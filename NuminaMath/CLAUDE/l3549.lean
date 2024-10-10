import Mathlib

namespace tangent_line_smallest_slope_l3549_354904

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem tangent_line_smallest_slope :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, f x = y → a*x + b*y + c = 0) ∧ 
    (∀ x₀ y₀ : ℝ, f x₀ = y₀ → ∀ m : ℝ, (∃ x y : ℝ, f x = y ∧ m = f' x) → m ≥ a) ∧
    a = 3 ∧ b = -1 ∧ c = -11 :=
sorry

end tangent_line_smallest_slope_l3549_354904


namespace unique_satisfying_function_l3549_354992

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2

theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ f = fun x ↦ x^2 :=
sorry

end unique_satisfying_function_l3549_354992


namespace smallest_n_congruence_l3549_354945

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n : ℤ) ≡ 409 [ZMOD 31] ∧ 
  ∀ m : ℕ+, (5 * m : ℤ) ≡ 409 [ZMOD 31] → n ≤ m → n = 2 := by
sorry

end smallest_n_congruence_l3549_354945


namespace apple_juice_production_l3549_354960

theorem apple_juice_production (total_production : ℝ) 
  (mixed_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 5.5 →
  mixed_percentage = 0.2 →
  juice_percentage = 0.5 →
  (1 - mixed_percentage) * juice_percentage * total_production = 2.2 := by
sorry

end apple_juice_production_l3549_354960


namespace count_points_is_ten_l3549_354955

def M : Finset Int := {1, -2, 3}
def N : Finset Int := {-4, 5, 6, -7}

def is_in_third_or_fourth_quadrant (p : Int × Int) : Bool :=
  p.2 < 0

def count_points : Nat :=
  (M.card * (N.filter (· < 0)).card) + (N.card * (M.filter (· < 0)).card)

theorem count_points_is_ten :
  count_points = 10 := by sorry

end count_points_is_ten_l3549_354955


namespace least_product_of_two_primes_above_30_l3549_354967

theorem least_product_of_two_primes_above_30 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p > 30 → q > 30 → 
  ∀ r s : ℕ, Prime r → Prime s → r ≠ s → r > 30 → s > 30 → 
  p * q ≤ r * s := by
  sorry

end least_product_of_two_primes_above_30_l3549_354967


namespace sqrt_three_squared_l3549_354908

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end sqrt_three_squared_l3549_354908


namespace complex_modulus_example_l3549_354952

theorem complex_modulus_example : Complex.abs (3 - 10*I) = Real.sqrt 109 := by
  sorry

end complex_modulus_example_l3549_354952


namespace purely_imaginary_complex_number_l3549_354956

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 - a - 2) (a + 1) ∧ z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end purely_imaginary_complex_number_l3549_354956


namespace line_segment_param_product_l3549_354948

/-- Given a line segment connecting (1, -3) and (6, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that (a+b) × (c+d) = 54. -/
theorem line_segment_param_product (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (1 = b ∧ -3 = d) →
  (6 = a + b ∧ 9 = c + d) →
  (a + b) * (c + d) = 54 := by
sorry

end line_segment_param_product_l3549_354948


namespace triangulations_count_l3549_354971

/-- The number of triangulations of a convex n-gon with exactly two internal triangles -/
def triangulations_with_two_internal_triangles (n : ℕ) : ℕ :=
  n * Nat.choose (n - 4) 4 * 2^(n - 9)

/-- Theorem stating the number of triangulations of a convex n-gon with exactly two internal triangles -/
theorem triangulations_count (n : ℕ) (hn : n > 7) :
  triangulations_with_two_internal_triangles n =
    n * Nat.choose (n - 4) 4 * 2^(n - 9) := by
  sorry

end triangulations_count_l3549_354971


namespace eulers_formula_l3549_354929

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  V : Type u  -- Vertex type
  E : Type v  -- Edge type
  F : Type w  -- Face type
  vertex_count : Nat
  edge_count : Nat
  face_count : Nat
  is_connected : Bool

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) :
  G.is_connected → G.vertex_count - G.edge_count + G.face_count = 2 := by
  sorry

#check eulers_formula

end eulers_formula_l3549_354929


namespace multiple_properties_l3549_354970

/-- Given that c is a multiple of 4 and d is a multiple of 8, prove the following statements -/
theorem multiple_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end multiple_properties_l3549_354970


namespace square_difference_from_sum_and_difference_l3549_354982

theorem square_difference_from_sum_and_difference (a b : ℚ) 
  (h1 : a + b = 9 / 17) (h2 : a - b = 1 / 51) : 
  a^2 - b^2 = 3 / 289 := by
  sorry

end square_difference_from_sum_and_difference_l3549_354982


namespace quartic_root_sum_l3549_354954

theorem quartic_root_sum (p q : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 4 + p * (Complex.I + 2 : ℂ) ^ 2 + q * (Complex.I + 2 : ℂ) + 1 = 0 →
  p + q = 10 := by
sorry

end quartic_root_sum_l3549_354954


namespace expression_value_l3549_354974

theorem expression_value : (2 * Real.sqrt 2) ^ (2/3) * (0.1)⁻¹ - Real.log 2 / Real.log 10 - Real.log 5 / Real.log 10 = 19 := by
  sorry

end expression_value_l3549_354974


namespace project_completion_time_l3549_354988

theorem project_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let time_together := (a * b) / (a + b)
  time_together > 0 ∧ 
  (1 / a + 1 / b) * time_together = 1 := by
  sorry

end project_completion_time_l3549_354988


namespace toy_store_fraction_l3549_354928

theorem toy_store_fraction (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_store_amount : ℚ) :
  weekly_allowance = 3 →
  arcade_fraction = 2/5 →
  candy_store_amount = 6/5 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry

end toy_store_fraction_l3549_354928


namespace secret_number_probability_l3549_354919

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  (units_digit n) % 3 = 0 ∧
  n > 75

theorem secret_number_probability :
  ∃! (valid_numbers : Finset ℕ),
    (∀ n, n ∈ valid_numbers ↔ satisfies_conditions n) ∧
    valid_numbers.card = 3 :=
sorry

end secret_number_probability_l3549_354919


namespace equation_equivalence_l3549_354950

theorem equation_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : 2*b - a ≠ 0) :
  (a + 2*b) / a = b / (2*b - a) ↔ 
  (a = -b * ((1 + Real.sqrt 17) / 2) ∨ a = -b * ((1 - Real.sqrt 17) / 2)) :=
by sorry

end equation_equivalence_l3549_354950


namespace ellipse_eccentricity_l3549_354937

/-- Given an ellipse and a hyperbola with shared foci, prove that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → a > b →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2/m^2 - y^2/n^2 = 1) →  -- Hyperbola equation
  c^2 = a^2 - b^2 →                     -- Shared foci condition for ellipse
  c^2 = m^2 + n^2 →                     -- Shared foci condition for hyperbola
  c^2 = a * m →                         -- c is geometric mean of a and m
  n^2 = m^2 + c^2/2 →                   -- n^2 is arithmetic mean of 2m^2 and c^2
  c/a = 1/2 := by
sorry

end ellipse_eccentricity_l3549_354937


namespace abc_divisibility_problem_l3549_354915

theorem abc_divisibility_problem (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b ∣ a^3 + b^3 + c^3) ∧
  (b^2 * c ∣ a^3 + b^3 + c^3) ∧
  (c^2 * a ∣ a^3 + b^3 + c^3) →
  ∃ k : ℕ, a = k ∧ b = k ∧ c = k := by
sorry

end abc_divisibility_problem_l3549_354915


namespace range_of_a_theorem_l3549_354903

-- Define the set A (condition p)
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}

-- Define the set B (condition q)
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {a | 1 ≤ a ∧ a ≤ 3 ∨ a = -1}

-- Statement of the theorem
theorem range_of_a_theorem :
  (∀ a : ℝ, A a ⊆ B a) → 
  (∀ a : ℝ, a ∈ RangeOfA ↔ (A a ⊆ B a)) :=
by sorry

end range_of_a_theorem_l3549_354903


namespace man_walking_time_l3549_354951

theorem man_walking_time (usual_time : ℝ) (reduced_time : ℝ) : 
  reduced_time = usual_time + 24 →
  (1 : ℝ) / 0.4 = reduced_time / usual_time →
  usual_time = 16 := by
sorry

end man_walking_time_l3549_354951


namespace diophantine_equation_solution_l3549_354910

theorem diophantine_equation_solution (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∃ (x y z : ℤ), (x, y, z) ≠ (0, 0, 0) ∧ a * x^2 + b * y^2 + c * z^2 = 0) :
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := by
  sorry

end diophantine_equation_solution_l3549_354910


namespace total_skips_theorem_l3549_354979

/-- Represents the number of skips a person can do with one rock -/
structure SkipAbility :=
  (skips : ℕ)

/-- Represents the number of rocks a person skipped -/
structure RocksSkipped :=
  (rocks : ℕ)

/-- Calculates the total skips for a person -/
def totalSkips (ability : SkipAbility) (skipped : RocksSkipped) : ℕ :=
  ability.skips * skipped.rocks

theorem total_skips_theorem 
  (bob_ability : SkipAbility)
  (jim_ability : SkipAbility)
  (sally_ability : SkipAbility)
  (bob_skipped : RocksSkipped)
  (jim_skipped : RocksSkipped)
  (sally_skipped : RocksSkipped)
  (h1 : bob_ability.skips = 12)
  (h2 : jim_ability.skips = 15)
  (h3 : sally_ability.skips = 18)
  (h4 : bob_skipped.rocks = 10)
  (h5 : jim_skipped.rocks = 8)
  (h6 : sally_skipped.rocks = 12) :
  totalSkips bob_ability bob_skipped + 
  totalSkips jim_ability jim_skipped + 
  totalSkips sally_ability sally_skipped = 456 := by
  sorry

#check total_skips_theorem

end total_skips_theorem_l3549_354979


namespace comic_books_sale_proof_l3549_354991

/-- The number of comic books sold by Scott and Sam -/
def comic_books_sold (initial_total remaining : ℕ) : ℕ :=
  initial_total - remaining

theorem comic_books_sale_proof :
  comic_books_sold 90 25 = 65 := by
  sorry

end comic_books_sale_proof_l3549_354991


namespace line_through_circle_center_perpendicular_to_l_l3549_354934

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define the line l
def l (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 2*x - 3*y + 3 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_l :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 4) ∧ 
    (result_line x₀ y₀) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧ x₁ ≠ x₂ → 
      (y₂ - y₁) * (x₀ - x₁) = -(x₂ - x₁) * (y₀ - y₁)) :=
sorry

end line_through_circle_center_perpendicular_to_l_l3549_354934


namespace seven_plums_balance_one_pear_l3549_354973

-- Define the weights of fruits as real numbers
variable (apple pear plum : ℝ)

-- Condition 1: 3 apples and 1 pear weigh as much as 10 plums
def condition1 : Prop := 3 * apple + pear = 10 * plum

-- Condition 2: 1 apple and 6 plums balance 1 pear
def condition2 : Prop := apple + 6 * plum = pear

-- Condition 3: Fruits of the same kind have the same weight
-- (This is implicitly assumed by using single variables for each fruit type)

-- Theorem: 7 plums balance one pear
theorem seven_plums_balance_one_pear
  (h1 : condition1 apple pear plum)
  (h2 : condition2 apple pear plum) :
  7 * plum = pear := by sorry

end seven_plums_balance_one_pear_l3549_354973


namespace hexagons_in_nth_ring_hexagons_in_100th_ring_l3549_354987

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring (n : ℕ) :
  hexagons_in_ring n = 6 * n := by sorry

/-- Corollary: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring :
  hexagons_in_ring 100 = 600 := by sorry

end hexagons_in_nth_ring_hexagons_in_100th_ring_l3549_354987


namespace log_30_8_l3549_354977

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- Define the conditions
axiom lg_5 : lg 5 = a
axiom lg_3 : lg 3 = b

-- State the theorem
theorem log_30_8 : (Real.log 8) / (Real.log 30) = 3 * (1 - a) / (b + 1) :=
sorry

end log_30_8_l3549_354977


namespace radius_of_larger_circle_l3549_354976

/-- Given a configuration of four circles of radius 2 that are externally tangent to two others
    and internally tangent to a larger circle, the radius of the larger circle is 2√3 + 2. -/
theorem radius_of_larger_circle (r : ℝ) (h1 : r > 0) :
  let small_radius : ℝ := 2
  let diagonal : ℝ := 4 * Real.sqrt 2
  let large_radius : ℝ := r
  (small_radius > 0) →
  (diagonal = 4 * Real.sqrt 2) →
  (large_radius = 2 * Real.sqrt 3 + 2) :=
by
  sorry

#check radius_of_larger_circle

end radius_of_larger_circle_l3549_354976


namespace no_prime_solution_l3549_354980

theorem no_prime_solution : ¬∃ (p q : Nat), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end no_prime_solution_l3549_354980


namespace parabola_fixed_point_l3549_354940

/-- Parabola E: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P -/
def P : ℝ × ℝ := (7, 3)

/-- Line with slope k passing through point P -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = k * (x - P.1)

/-- Line with slope 2/3 passing through point A -/
def line_AC (A : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - A.2 = (2/3) * (x - A.1)

theorem parabola_fixed_point :
  ∀ (k : ℝ) (A B C : ℝ × ℝ),
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola C.1 C.2 →
  line_through_P k A.1 A.2 →
  line_through_P k B.1 B.2 →
  line_AC A C.1 C.2 →
  ∃ (m : ℝ), y - C.2 = m * (x - C.1) ∧ y - B.2 = m * (x - B.1) →
  y - 3 = m * (x + 5/2) :=
sorry

end parabola_fixed_point_l3549_354940


namespace divisible_integers_count_l3549_354949

-- Define the range of integers
def lower_bound : ℕ := 2000
def upper_bound : ℕ := 3000

-- Define the factors
def factor1 : ℕ := 30
def factor2 : ℕ := 45
def factor3 : ℕ := 75

-- Function to count integers in the range divisible by all factors
def count_divisible_integers : ℕ := sorry

-- Theorem statement
theorem divisible_integers_count : count_divisible_integers = 2 := by sorry

end divisible_integers_count_l3549_354949


namespace problem_statement_l3549_354920

theorem problem_statement :
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧
  (∀ (a : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a/2 ≥ x + 2*y + 2*z) ↔ (a ≤ 0 ∨ a ≥ 4)) :=
by sorry

end problem_statement_l3549_354920


namespace pet_store_count_l3549_354964

/-- Represents the count of animals in a pet store -/
structure PetStore :=
  (birds : ℕ)
  (puppies : ℕ)
  (cats : ℕ)
  (spiders : ℕ)

/-- Calculates the total number of animals in the pet store -/
def totalAnimals (store : PetStore) : ℕ :=
  store.birds + store.puppies + store.cats + store.spiders

/-- Represents the changes in animal counts -/
structure Changes :=
  (birdsSold : ℕ)
  (puppiesAdopted : ℕ)
  (spidersLoose : ℕ)

/-- Applies changes to the pet store counts -/
def applyChanges (store : PetStore) (changes : Changes) : PetStore :=
  { birds := store.birds - changes.birdsSold,
    puppies := store.puppies - changes.puppiesAdopted,
    cats := store.cats,
    spiders := store.spiders - changes.spidersLoose }

theorem pet_store_count : 
  let initialStore : PetStore := { birds := 12, puppies := 9, cats := 5, spiders := 15 }
  let changes : Changes := { birdsSold := 6, puppiesAdopted := 3, spidersLoose := 7 }
  let finalStore := applyChanges initialStore changes
  totalAnimals finalStore = 25 := by
  sorry

end pet_store_count_l3549_354964


namespace b_share_is_2200_l3549_354933

/- Define the investments and a's share -/
def investment_a : ℕ := 7000
def investment_b : ℕ := 11000
def investment_c : ℕ := 18000
def share_a : ℕ := 1400

/- Define the function to calculate b's share -/
def calculate_b_share (inv_a inv_b inv_c share_a : ℕ) : ℕ :=
  let total_ratio := inv_a + inv_b + inv_c
  let total_profit := share_a * total_ratio / inv_a
  inv_b * total_profit / total_ratio

/- Theorem statement -/
theorem b_share_is_2200 : 
  calculate_b_share investment_a investment_b investment_c share_a = 2200 := by
  sorry


end b_share_is_2200_l3549_354933


namespace train_speed_l3549_354918

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 400) (h2 : time = 10) :
  length / time = 40 := by
  sorry

end train_speed_l3549_354918


namespace add_point_four_to_fifty_six_point_seven_l3549_354926

theorem add_point_four_to_fifty_six_point_seven :
  0.4 + 56.7 = 57.1 := by sorry

end add_point_four_to_fifty_six_point_seven_l3549_354926


namespace product_of_solutions_l3549_354921

theorem product_of_solutions (x₁ x₂ : ℝ) 
  (h₁ : x₁ * Real.exp x₁ = Real.exp 2)
  (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end product_of_solutions_l3549_354921


namespace product_equals_20152015_l3549_354978

theorem product_equals_20152015 : 5 * 13 * 31 * 73 * 137 = 20152015 := by
  sorry

end product_equals_20152015_l3549_354978


namespace graph_connected_probability_l3549_354983

def n : ℕ := 20
def edges_removed : ℕ := 35

theorem graph_connected_probability :
  let total_edges := n * (n - 1) / 2
  let remaining_edges := total_edges - edges_removed
  let prob_disconnected := n * (Nat.choose remaining_edges (remaining_edges - n + 1)) / (Nat.choose total_edges edges_removed)
  (1 : ℚ) - prob_disconnected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end graph_connected_probability_l3549_354983


namespace division_simplification_l3549_354996

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by sorry

end division_simplification_l3549_354996


namespace xiaoman_dumpling_probability_l3549_354947

theorem xiaoman_dumpling_probability :
  let total_dumplings : ℕ := 10
  let egg_dumplings : ℕ := 3
  let probability : ℚ := egg_dumplings / total_dumplings
  probability = 3 / 10 := by
sorry

end xiaoman_dumpling_probability_l3549_354947


namespace modulus_of_z_l3549_354925

theorem modulus_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l3549_354925


namespace four_weavers_four_days_eight_mats_l3549_354924

/-- The rate at which mat-weavers work, in mats per weaver per day -/
def weaving_rate (mats : ℕ) (weavers : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / (weavers * days)

/-- The number of mats that can be woven given a number of weavers, days, and a weaving rate -/
def mats_woven (weavers : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (weavers : ℚ) * days * rate

theorem four_weavers_four_days_eight_mats 
  (h : weaving_rate 16 8 8 = weaving_rate 8 4 4) : 
  mats_woven 4 4 (weaving_rate 16 8 8) = 8 := by
  sorry

end four_weavers_four_days_eight_mats_l3549_354924


namespace cos_eight_arccos_one_fourth_l3549_354959

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = -16286/16384 := by
  sorry

end cos_eight_arccos_one_fourth_l3549_354959


namespace phase_shift_sine_specific_phase_shift_l3549_354968

/-- The phase shift of a sine function of the form y = A sin(Bx + C) is -C/B -/
theorem phase_shift_sine (A B C : ℝ) (h : B ≠ 0) :
  let f := λ x : ℝ => A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x : ℝ, f (x + phase_shift) = A * Real.sin (B * x) := by
  sorry

/-- The phase shift of y = 3 sin(3x + π/4) is -π/12 -/
theorem specific_phase_shift :
  let f := λ x : ℝ => 3 * Real.sin (3 * x + π/4)
  let phase_shift := -π/12
  ∀ x : ℝ, f (x + phase_shift) = 3 * Real.sin (3 * x) := by
  sorry

end phase_shift_sine_specific_phase_shift_l3549_354968


namespace cakes_after_school_l3549_354998

theorem cakes_after_school (croissants_per_person breakfast_people pizzas_per_person bedtime_people total_food : ℕ) 
  (h1 : croissants_per_person = 7)
  (h2 : breakfast_people = 2)
  (h3 : pizzas_per_person = 30)
  (h4 : bedtime_people = 2)
  (h5 : total_food = 110) :
  ∃ (cakes_per_person : ℕ), 
    croissants_per_person * breakfast_people + cakes_per_person * 2 + pizzas_per_person * bedtime_people = total_food ∧ 
    cakes_per_person = 18 := by
  sorry

end cakes_after_school_l3549_354998


namespace inequality_solution_l3549_354930

theorem inequality_solution (x : ℕ+) : 
  (12 * x + 5 < 10 * x + 15) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := by
sorry

end inequality_solution_l3549_354930


namespace sample_size_calculation_l3549_354911

/-- Given a sample divided into groups, this theorem proves that when one group
    has a frequency of 36 and a rate of 0.25, the total sample size is 144. -/
theorem sample_size_calculation (n : ℕ) (f : ℕ) (r : ℚ)
  (h1 : f = 36)
  (h2 : r = 1/4)
  (h3 : r = f / n) :
  n = 144 := by
  sorry

end sample_size_calculation_l3549_354911


namespace tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3549_354932

/-- The volume of a tetrahedron formed from a regular pentagon -/
theorem tetrahedron_volume_from_pentagon (side_length : ℝ) 
  (h_side : side_length = 1) : ℝ :=
let diagonal_length := (1 + Real.sqrt 5) / 2
let base_area := Real.sqrt 3 / 4 * side_length ^ 2
let height := Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)
let volume := (1 / 3) * base_area * height
(1 + Real.sqrt 5) / 24

/-- The theorem statement -/
theorem tetrahedron_volume_proof : 
  ∃ (v : ℝ), tetrahedron_volume_from_pentagon 1 rfl = v ∧ v = (1 + Real.sqrt 5) / 24 := by
sorry

end tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3549_354932


namespace chess_problem_l3549_354939

/-- Represents a chess piece (rook or king) -/
inductive Piece
| Rook
| King

/-- Represents a position on the chess board -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the state of the chess board -/
structure ChessBoard :=
  (size : Nat)
  (whiteRooks : List Position)
  (blackKing : Position)

/-- Checks if a position is in check -/
def isInCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can get into check after some finite number of moves -/
def canGetIntoCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check after its move (excluding initial moves) -/
def canAlwaysBeInCheckAfterMove (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check (even after white's move, excluding initial moves) -/
def canAlwaysBeInCheck (board : ChessBoard) : Bool :=
  sorry

theorem chess_problem (board : ChessBoard) 
  (h1 : board.size = 1000) 
  (h2 : board.whiteRooks.length = 499) :
  (canGetIntoCheck board = true) ∧ 
  (canAlwaysBeInCheckAfterMove board = false) ∧
  (canAlwaysBeInCheck board = false) :=
  sorry

end chess_problem_l3549_354939


namespace midpoint_region_area_l3549_354938

/-- A regular hexagon with area 16 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq_16 : area = 16)

/-- The midpoint of a side of the hexagon -/
structure Midpoint :=
  (hexagon : RegularHexagon)

/-- A region formed by connecting four consecutive midpoints -/
structure MidpointRegion :=
  (hexagon : RegularHexagon)
  (midpoints : Fin 4 → Midpoint)
  (consecutive : ∀ i : Fin 3, (midpoints i).hexagon = (midpoints (i + 1)).hexagon)

/-- The theorem statement -/
theorem midpoint_region_area (region : MidpointRegion) : 
  (region.hexagon.area / 2) = 8 :=
sorry

end midpoint_region_area_l3549_354938


namespace walter_chores_l3549_354989

theorem walter_chores (total_days : ℕ) (regular_pay : ℕ) (exceptional_pay : ℕ) (total_earnings : ℕ) :
  total_days = 15 ∧ regular_pay = 4 ∧ exceptional_pay = 6 ∧ total_earnings = 78 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 9 :=
by sorry

end walter_chores_l3549_354989


namespace saroj_current_age_l3549_354909

/-- Represents the age of a person at different points in time -/
structure PersonAge where
  sixYearsAgo : ℕ
  current : ℕ
  fourYearsHence : ℕ

/-- The problem statement -/
theorem saroj_current_age 
  (vimal saroj : PersonAge)
  (h1 : vimal.sixYearsAgo * 5 = saroj.sixYearsAgo * 6)
  (h2 : vimal.fourYearsHence * 10 = saroj.fourYearsHence * 11)
  (h3 : vimal.current = vimal.sixYearsAgo + 6)
  (h4 : saroj.current = saroj.sixYearsAgo + 6)
  (h5 : vimal.fourYearsHence = vimal.current + 4)
  (h6 : saroj.fourYearsHence = saroj.current + 4)
  : saroj.current = 16 := by
  sorry

end saroj_current_age_l3549_354909


namespace expression_equals_one_l3549_354984

theorem expression_equals_one : 
  (105^2 - 8^2) / (80^2 - 13^2) * ((80 - 13) * (80 + 13)) / ((105 - 8) * (105 + 8)) = 1 := by
  sorry

end expression_equals_one_l3549_354984


namespace nina_money_theorem_l3549_354972

theorem nina_money_theorem (x : ℝ) 
  (h1 : 6 * x = 8 * (x - 1.5)) : 6 * x = 36 := by
  sorry

end nina_money_theorem_l3549_354972


namespace lcm_count_l3549_354936

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (6^9) (Nat.lcm (9^9) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (6^9) (Nat.lcm (9^9) k) ≠ 18^18)) := by
  sorry

end lcm_count_l3549_354936


namespace percentage_without_full_time_jobs_l3549_354953

theorem percentage_without_full_time_jobs :
  let total_parents : ℝ := 100
  let mothers : ℝ := 0.6 * total_parents
  let fathers : ℝ := 0.4 * total_parents
  let mothers_with_jobs : ℝ := (7/8) * mothers
  let fathers_with_jobs : ℝ := (3/4) * fathers
  let parents_with_jobs : ℝ := mothers_with_jobs + fathers_with_jobs
  let parents_without_jobs : ℝ := total_parents - parents_with_jobs
  (parents_without_jobs / total_parents) * 100 = 18 :=
by sorry

end percentage_without_full_time_jobs_l3549_354953


namespace power_five_mod_eighteen_l3549_354986

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end power_five_mod_eighteen_l3549_354986


namespace prime_divisibility_l3549_354981

theorem prime_divisibility (p q : ℕ) (n : ℕ) 
  (h_p_prime : Prime p) 
  (h_q_prime : Prime q) 
  (h_distinct : p ≠ q) 
  (h_pq_div : (p * q) ∣ (n^(p*q) + 1)) 
  (h_p3q3_div : (p^3 * q^3) ∣ (n^(p*q) + 1)) :
  p^2 ∣ (n + 1) ∨ q^2 ∣ (n + 1) := by
sorry

end prime_divisibility_l3549_354981


namespace garden_area_l3549_354965

-- Define the garden structure
structure Garden where
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

-- Define the conditions
def garden_conditions (g : Garden) : Prop :=
  g.perimeter = 4 * g.side_length ∧
  g.area = g.side_length * g.side_length ∧
  1500 = 30 * g.side_length ∧
  1500 = 15 * g.perimeter

-- Theorem statement
theorem garden_area (g : Garden) (h : garden_conditions g) : g.area = 625 := by
  sorry

end garden_area_l3549_354965


namespace seventh_term_of_geometric_sequence_l3549_354958

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- The seventh term of a geometric sequence with first term -4 and second term 8 is -256 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℝ := -4
  let a₂ : ℝ := 8
  let r : ℝ := a₂ / a₁
  geometric_sequence a₁ r 7 = -256 := by
sorry

end seventh_term_of_geometric_sequence_l3549_354958


namespace committee_selection_ways_l3549_354916

theorem committee_selection_ways : Nat.choose 15 4 = 1365 := by
  sorry

end committee_selection_ways_l3549_354916


namespace pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l3549_354917

/-- A regular pentagon with numbers from 1 to 5 at its vertices -/
def Pentagon : Type := Fin 5 → Fin 5

/-- A stack of pentagons -/
def PentagonStack : Type := List Pentagon

/-- The sum of numbers at a vertex in a stack of pentagons -/
def vertexSum (stack : PentagonStack) (vertex : Fin 5) : ℕ :=
  (stack.map (λ p => p vertex)).sum

/-- A predicate that checks if all vertex sums in a stack are equal -/
def allVertexSumsEqual (stack : PentagonStack) : Prop :=
  ∀ v1 v2 : Fin 5, vertexSum stack v1 = vertexSum stack v2

/-- Main theorem: For any natural number n ≠ 1 and n ≠ 3, there exists a valid pentagon stack of size n -/
theorem pentagon_stack_exists (n : ℕ) (h1 : n ≠ 1) (h3 : n ≠ 3) :
  ∃ (stack : PentagonStack), stack.length = n ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 1 -/
theorem no_pentagon_stack_for_one :
  ¬∃ (stack : PentagonStack), stack.length = 1 ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 3 -/
theorem no_pentagon_stack_for_three :
  ¬∃ (stack : PentagonStack), stack.length = 3 ∧ allVertexSumsEqual stack :=
sorry

end pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l3549_354917


namespace duck_weight_calculation_l3549_354935

theorem duck_weight_calculation (num_ducks : ℕ) (cost_per_duck : ℚ) (selling_price_per_pound : ℚ) (profit : ℚ) : 
  num_ducks = 30 →
  cost_per_duck = 10 →
  selling_price_per_pound = 5 →
  profit = 300 →
  (profit + num_ducks * cost_per_duck) / (selling_price_per_pound * num_ducks) = 4 := by
sorry

end duck_weight_calculation_l3549_354935


namespace twelfth_day_is_monday_l3549_354957

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- Axiom: The month has exactly 5 Fridays -/
axiom five_fridays (m : Month) : m.fridayCount = 5

/-- Axiom: The first day of the month is not a Friday -/
axiom first_not_friday (m : Month) : m.firstDay ≠ DayOfWeek.Friday

/-- Axiom: The last day of the month is not a Friday -/
axiom last_not_friday (m : Month) : m.lastDay ≠ DayOfWeek.Friday

/-- Function to get the day of week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end twelfth_day_is_monday_l3549_354957


namespace correct_possible_values_l3549_354993

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of possible values for 'a' -/
def PossibleValues : Set ℝ := {1/3, 3, -6}

/-- Function to count the number of intersection points between three lines -/
def countIntersections (l1 l2 l3 : Line) : ℕ := sorry

/-- Theorem stating that the set of possible values of 'a' is correct -/
theorem correct_possible_values :
  ∀ a : ℝ,
  (∃ l3 : Line,
    l3.a = a ∧ l3.b = 3 ∧ l3.c = -5 ∧
    countIntersections ⟨1, 1, 1⟩ ⟨2, -1, 8⟩ l3 ≤ 2) ↔
  a ∈ PossibleValues :=
sorry

end correct_possible_values_l3549_354993


namespace uncle_dave_nieces_l3549_354927

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end uncle_dave_nieces_l3549_354927


namespace trailing_zeroes_sum_factorials_l3549_354901

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZeroes (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

theorem trailing_zeroes_sum_factorials :
  trailingZeroes (factorial 60 + factorial 120) = 14 := by
  sorry

end trailing_zeroes_sum_factorials_l3549_354901


namespace unique_m_value_l3549_354997

/-- Given a set A and a real number m, proves that m = 3 is the only valid solution -/
theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end unique_m_value_l3549_354997


namespace tangent_line_theorem_intersecting_line_theorem_l3549_354906

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 6
def tangent_line_2 (x y : ℝ) : Prop := 5*x + 12*y - 78 = 0

-- Define the intersecting line equation
def intersecting_line (x y : ℝ) : Prop := 
  ∃ k, y - 4 = k*(x - 6) ∧ (k = (4 + Real.sqrt 17)/3 ∨ k = (4 - Real.sqrt 17)/3)

theorem tangent_line_theorem :
  ∀ x y : ℝ, 
  (∃ l : ℝ → ℝ → Prop, (l x y ↔ tangent_line_1 x ∨ tangent_line_2 x y) ∧ 
    (l (point_P.1) (point_P.2) ∧ 
     ∀ a b : ℝ, circle_equation a b → (l a b → a = (point_P.1) ∧ b = (point_P.2)))) :=
sorry

theorem intersecting_line_theorem :
  ∀ x y : ℝ,
  (intersecting_line x y →
    (x = point_P.1 ∧ y = point_P.2 ∨
     (∃ a b : ℝ, circle_equation a b ∧ intersecting_line a b ∧
      ∃ c d : ℝ, circle_equation c d ∧ intersecting_line c d ∧
      (a - c)^2 + (b - d)^2 = 18))) :=
sorry

end tangent_line_theorem_intersecting_line_theorem_l3549_354906


namespace area_of_M_l3549_354923

-- Define the region M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (|y| + |4 - y| ≤ 4) ∧
               ((y^2 + x - 4*y + 1) / (2*y + x - 7) ≤ 0)}

-- State the theorem
theorem area_of_M : MeasureTheory.volume M = 8 := by
  sorry

end area_of_M_l3549_354923


namespace tom_climbing_time_l3549_354946

/-- Given that Elizabeth takes 30 minutes to climb a hill and Tom takes four times as long,
    prove that Tom's climbing time is 2 hours. -/
theorem tom_climbing_time :
  let elizabeth_time : ℕ := 30 -- Elizabeth's climbing time in minutes
  let tom_factor : ℕ := 4 -- Tom takes four times as long as Elizabeth
  let tom_time : ℕ := elizabeth_time * tom_factor -- Tom's climbing time in minutes
  tom_time / 60 = 2 -- Tom's climbing time in hours
:= by sorry

end tom_climbing_time_l3549_354946


namespace units_digit_of_product_over_1000_l3549_354942

theorem units_digit_of_product_over_1000 : 
  (20 * 21 * 22 * 23 * 24 * 25) / 1000 ≡ 2 [ZMOD 10] := by
  sorry

end units_digit_of_product_over_1000_l3549_354942


namespace percentage_of_360_equals_93_6_l3549_354943

theorem percentage_of_360_equals_93_6 (total : ℝ) (part : ℝ) (percentage : ℝ) : 
  total = 360 → part = 93.6 → percentage = (part / total) * 100 → percentage = 26 := by
  sorry

end percentage_of_360_equals_93_6_l3549_354943


namespace sector_central_angle_l3549_354905

theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 6 →
  (1 / 2) * l * r = 2 →
  α = l / r →
  α = 1 ∨ α = 4 := by
  sorry

end sector_central_angle_l3549_354905


namespace factor_decomposition_l3549_354963

theorem factor_decomposition (a b : Int) : 
  a * b = 96 → a^2 + b^2 = 208 → 
  ((a = 8 ∧ b = 12) ∨ (a = -8 ∧ b = -12) ∨ (a = 12 ∧ b = 8) ∨ (a = -12 ∧ b = -8)) :=
by sorry

end factor_decomposition_l3549_354963


namespace expression_value_l3549_354914

theorem expression_value (a b c : ℤ) : 
  (-a = 2) → (abs b = 6) → (-c + b = -10) → (8 - a + b - c = 0) := by
sorry

end expression_value_l3549_354914


namespace square_of_difference_l3549_354995

theorem square_of_difference (y : ℝ) (h : 4 * y^2 - 36 ≥ 0) :
  (10 - Real.sqrt (4 * y^2 - 36))^2 = 4 * y^2 + 64 - 20 * Real.sqrt (4 * y^2 - 36) := by
  sorry

end square_of_difference_l3549_354995


namespace fixed_point_of_exponential_function_l3549_354944

/-- The function f(x) = a^(2-x) + 2 passes through the point (2, 3) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2 - x) + 2
  f 2 = 3 := by sorry

end fixed_point_of_exponential_function_l3549_354944


namespace sum_of_roots_l3549_354985

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 25*c - 75 = 0) 
  (hd : 9*d^3 - 72*d^2 - 345*d + 3060 = 0) : 
  c + d = 10 := by
sorry

end sum_of_roots_l3549_354985


namespace factors_of_x_fourth_plus_81_l3549_354902

theorem factors_of_x_fourth_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end factors_of_x_fourth_plus_81_l3549_354902


namespace simplify_expressions_l3549_354912

variable (x y a : ℝ)

theorem simplify_expressions :
  (5 * x - 3 * (2 * x - 3 * y) + x = 9 * y) ∧
  (3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3) := by sorry

end simplify_expressions_l3549_354912


namespace smallest_book_count_l3549_354941

theorem smallest_book_count (physics chemistry biology : ℕ) : 
  physics = 3 * (chemistry / 2) →  -- ratio of physics to chemistry is 3:2
  4 * biology = 3 * chemistry →    -- ratio of chemistry to biology is 4:3
  physics + chemistry + biology > 0 →  -- total number of books is more than 0
  ∀ n : ℕ, n > 0 → 
    (∃ p c b : ℕ, p = 3 * (c / 2) ∧ 4 * b = 3 * c ∧ p + c + b = n) →
    n ≥ 15 :=
by sorry

end smallest_book_count_l3549_354941


namespace hannah_late_times_l3549_354900

/-- Represents the number of times Hannah was late to work in a week. -/
def times_late (hourly_rate : ℕ) (hours_worked : ℕ) (dock_amount : ℕ) (actual_pay : ℕ) : ℕ :=
  (hourly_rate * hours_worked - actual_pay) / dock_amount

/-- Theorem stating that Hannah was late 3 times given the problem conditions. -/
theorem hannah_late_times :
  times_late 30 18 5 525 = 3 := by
  sorry

end hannah_late_times_l3549_354900


namespace rectangular_field_perimeter_l3549_354990

/-- The perimeter of a rectangular field with length 7/5 of its width and width 70 meters is 336 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 70 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 336 := by
sorry

end rectangular_field_perimeter_l3549_354990


namespace fountain_area_l3549_354913

-- Define the fountain
structure Fountain :=
  (ab : ℝ)  -- Length of AB
  (dc : ℝ)  -- Length of DC
  (h_ab_positive : ab > 0)
  (h_dc_positive : dc > 0)
  (h_d_midpoint : True)  -- Represents that D is the midpoint of AB
  (h_c_center : True)    -- Represents that C is the center of the fountain

-- Define the theorem
theorem fountain_area (f : Fountain) (h_ab : f.ab = 20) (h_dc : f.dc = 12) : 
  (π * (f.ab / 2) ^ 2 + π * f.dc ^ 2) = 244 * π := by
  sorry


end fountain_area_l3549_354913


namespace arccos_cos_eq_three_halves_x_implies_x_zero_l3549_354961

theorem arccos_cos_eq_three_halves_x_implies_x_zero 
  (x : ℝ) 
  (h1 : -π ≤ x ∧ x ≤ π) 
  (h2 : Real.arccos (Real.cos x) = (3 * x) / 2) : 
  x = 0 := by
  sorry

end arccos_cos_eq_three_halves_x_implies_x_zero_l3549_354961


namespace bah_equivalent_to_yahs_l3549_354969

/-- Conversion rate between bahs and rahs -/
def bah_to_rah : ℚ := 18 / 10

/-- Conversion rate between rahs and yahs -/
def rah_to_yah : ℚ := 10 / 6

/-- The number of yahs to convert -/
def yahs_to_convert : ℕ := 1500

theorem bah_equivalent_to_yahs : 
  ∃ (n : ℕ), n * bah_to_rah * rah_to_yah = yahs_to_convert ∧ n = 500 := by
  sorry

end bah_equivalent_to_yahs_l3549_354969


namespace hexagonal_prism_diagonals_l3549_354994

/-- A regular hexagonal prism --/
structure RegularHexagonalPrism where
  /-- Number of sides in the base --/
  n : ℕ
  /-- Assertion that the base has 6 sides --/
  base_is_hexagon : n = 6

/-- The number of diagonals in a regular hexagonal prism --/
def num_diagonals (prism : RegularHexagonalPrism) : ℕ := prism.n * (prism.n - 3)

/-- Theorem: The number of diagonals in a regular hexagonal prism is 18 --/
theorem hexagonal_prism_diagonals (prism : RegularHexagonalPrism) : 
  num_diagonals prism = 18 := by
  sorry

#check hexagonal_prism_diagonals

end hexagonal_prism_diagonals_l3549_354994


namespace solar_panel_height_P_l3549_354962

/-- Regular hexagon with side length 10 and pillars at vertices -/
structure SolarPanelSupport where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at L, M, and N
  height_L : ℝ
  height_M : ℝ
  height_N : ℝ

/-- The height of the pillar at P in the solar panel support system -/
def height_P (s : SolarPanelSupport) : ℝ := sorry

/-- Theorem stating the height of pillar P given specific conditions -/
theorem solar_panel_height_P (s : SolarPanelSupport) 
  (h_side : s.side_length = 10)
  (h_L : s.height_L = 15)
  (h_M : s.height_M = 12)
  (h_N : s.height_N = 13) : 
  height_P s = 22 := by sorry

end solar_panel_height_P_l3549_354962


namespace factor_expression_l3549_354931

theorem factor_expression (x : ℝ) :
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := by
  sorry

end factor_expression_l3549_354931


namespace sum_of_max_and_min_is_4022_l3549_354975

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval
def I : Set ℝ := Set.Icc (-2011) 2011

-- State the theorem
theorem sum_of_max_and_min_is_4022 
  (h1 : ∀ x ∈ I, ∀ y ∈ I, f (x + y) = f x + f y - 2011)
  (h2 : ∀ x > 0, x ∈ I → f x > 2011)
  : (⨆ x ∈ I, f x) + (⨅ x ∈ I, f x) = 4022 := by
  sorry

end sum_of_max_and_min_is_4022_l3549_354975


namespace profit_share_difference_l3549_354999

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_share_B : ℕ := 2500

theorem profit_share_difference :
  let ratio_A : ℕ := investment_A / 2000
  let ratio_B : ℕ := investment_B / 2000
  let ratio_C : ℕ := investment_C / 2000
  let part_value : ℕ := profit_share_B / ratio_B
  let profit_A : ℕ := ratio_A * part_value
  let profit_C : ℕ := ratio_C * part_value
  profit_C - profit_A = 1000 := by sorry

end profit_share_difference_l3549_354999


namespace well_depth_equation_l3549_354922

theorem well_depth_equation (d : ℝ) (u : ℝ) (h : u = Real.sqrt d) : 
  d = 14 * (10 - d / 1200)^2 → 14 * u^2 + 1200 * u - 12000 * Real.sqrt 14 = 0 := by
  sorry

#check well_depth_equation

end well_depth_equation_l3549_354922


namespace second_expression_proof_l3549_354966

theorem second_expression_proof (a x : ℝ) : 
  ((2 * a + 16 + x) / 2 = 84) → (a = 32) → (x = 88) := by
  sorry

end second_expression_proof_l3549_354966


namespace a_plus_b_values_l3549_354907

theorem a_plus_b_values (a b : ℝ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 := by
sorry

end a_plus_b_values_l3549_354907
