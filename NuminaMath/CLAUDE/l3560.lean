import Mathlib

namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3560_356099

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (n_weight o_weight : ℝ) (n_count o_count : ℕ) : ℝ :=
  n_weight * n_count + o_weight * o_count

/-- The molecular weight of the compound is 44.02 amu -/
theorem compound_molecular_weight :
  molecular_weight nitrogen_weight oxygen_weight nitrogen_count oxygen_count = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3560_356099


namespace NUMINAMATH_CALUDE_expression_value_l3560_356032

theorem expression_value (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 10) (hc : c = 7) (hk : k = 3) : 
  k * ((a - (b - c)) - ((a - b) - c)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3560_356032


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3560_356068

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first five terms of the specific geometric sequence -/
def specific_sum : ℚ := geometric_sum (1/3) (1/3) 5

theorem geometric_sequence_sum :
  specific_sum = 121/243 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3560_356068


namespace NUMINAMATH_CALUDE_four_points_cyclic_l3560_356039

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the necessary geometric relations
variable (collinear : Point → Point → Point → Prop)
variable (orthocenter : Point → Point → Point → Point)
variable (lies_on : Point → Line → Prop)
variable (concurrent : Line → Line → Line → Prop)
variable (cyclic : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_points_cyclic
  (A B C D P Q R : Point)
  (AP BQ CR : Line)
  (h1 : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)
  (h2 : orthocenter B C D ≠ D)
  (h3 : P = orthocenter B C D)
  (h4 : Q = orthocenter C A D)
  (h5 : R = orthocenter A B D)
  (h6 : lies_on A AP ∧ lies_on P AP)
  (h7 : lies_on B BQ ∧ lies_on Q BQ)
  (h8 : lies_on C CR ∧ lies_on R CR)
  (h9 : AP ≠ BQ ∧ BQ ≠ CR ∧ CR ≠ AP)
  (h10 : concurrent AP BQ CR)
  : cyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_four_points_cyclic_l3560_356039


namespace NUMINAMATH_CALUDE_linear_function_properties_l3560_356094

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 3)

-- Define the shifted function
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-3) = 0) ∧ 
  (g k 1 = -2 → k = -1) ∧
  (k < 0 → ∀ x₁ x₂ y₁ y₂ : ℝ, 
    f k x₁ = y₁ → f k x₂ = y₂ → y₁ < y₂ → x₁ > x₂) :=
by sorry


end NUMINAMATH_CALUDE_linear_function_properties_l3560_356094


namespace NUMINAMATH_CALUDE_system_solution_l3560_356072

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(2, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3560_356072


namespace NUMINAMATH_CALUDE_log_one_fifth_25_l3560_356015

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_fifth_25_l3560_356015


namespace NUMINAMATH_CALUDE_triangle_properties_l3560_356018

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * (Real.cos (t.B / 2))^2 + t.b * (Real.cos (t.A / 2))^2 = (3/2) * t.c)
  (h2 : t.a = 2 * t.b)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15) :
  (t.A > π / 2) ∧ (t.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3560_356018


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3560_356045

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (lst : List Nat) : ℚ :=
  (lst.map (λ x => (1 : ℚ) / x)).sum / lst.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3560_356045


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3560_356012

theorem cistern_filling_time (T : ℝ) : 
  T > 0 →  -- T must be positive
  (1 / 4 : ℝ) - (1 / T) = (3 / 28 : ℝ) → 
  T = 7 :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3560_356012


namespace NUMINAMATH_CALUDE_family_ages_l3560_356089

/-- Given the ages and relationships of family members, prove the ages of the younger siblings after 30 years -/
theorem family_ages (elder_son_age : ℕ) (declan_age_diff : ℕ) (younger_son_age_diff : ℕ) (third_sibling_age_diff : ℕ) (years_later : ℕ)
  (h1 : elder_son_age = 40)
  (h2 : declan_age_diff = 25)
  (h3 : younger_son_age_diff = 10)
  (h4 : third_sibling_age_diff = 5)
  (h5 : years_later = 30) :
  let younger_son_age := elder_son_age - younger_son_age_diff
  let third_sibling_age := younger_son_age - third_sibling_age_diff
  (younger_son_age + years_later = 60) ∧ (third_sibling_age + years_later = 55) :=
by sorry

end NUMINAMATH_CALUDE_family_ages_l3560_356089


namespace NUMINAMATH_CALUDE_sam_investment_result_l3560_356002

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 10000
def first_rate : ℝ := 0.20
def first_time : ℕ := 3
def multiplier : ℝ := 3
def second_rate : ℝ := 0.15
def second_time : ℕ := 1

-- Theorem statement
theorem sam_investment_result :
  let first_phase := compound_interest initial_investment first_rate first_time
  let second_phase := compound_interest (first_phase * multiplier) second_rate second_time
  second_phase = 59616 := by sorry

end NUMINAMATH_CALUDE_sam_investment_result_l3560_356002


namespace NUMINAMATH_CALUDE_combined_weight_l3560_356058

/-- The combined weight of John, Mary, and Jamison is 540 lbs -/
theorem combined_weight (mary_weight jamison_weight john_weight : ℝ) :
  mary_weight = 160 →
  jamison_weight = mary_weight + 20 →
  john_weight = mary_weight * (5/4) →
  mary_weight + jamison_weight + john_weight = 540 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_l3560_356058


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3560_356088

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube (x : ℕ) (h : x = 11 * 36 * 54) :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3560_356088


namespace NUMINAMATH_CALUDE_patio_rearrangement_l3560_356059

theorem patio_rearrangement (r c : ℕ) : 
  r * c = 48 ∧ 
  (r + 4) * (c - 2) = 48 ∧ 
  c > 2 →
  r = 6 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l3560_356059


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l3560_356017

/-- A circle tangent to the parabola y = x^2 at two points and lying inside the parabola --/
structure TangentCircle where
  /-- The x-coordinate of one tangent point (the other is at -a) --/
  a : ℝ
  /-- The y-coordinate of the circle's center --/
  b : ℝ
  /-- The radius of the circle --/
  r : ℝ
  /-- The circle lies inside the parabola --/
  inside : b > a^2
  /-- The circle is tangent to the parabola at (a, a^2) and (-a, a^2) --/
  tangent : (a^2 + (a^2 - b)^2 = r^2) ∧ (a^2 + (a^2 - b)^2 = r^2)

/-- The difference between the y-coordinate of the circle's center and the y-coordinate of either tangent point is 1/2 --/
theorem tangent_circle_height_difference (c : TangentCircle) : c.b - c.a^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l3560_356017


namespace NUMINAMATH_CALUDE_oranges_in_bin_l3560_356031

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 40 → thrown_away = 37 → final = 10 → final - (initial - thrown_away) = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_bin_l3560_356031


namespace NUMINAMATH_CALUDE_g_derivative_l3560_356067

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2 + 3^x

theorem g_derivative (x : ℝ) (h : x > 0) :
  deriv g x = 1 / (x * Real.log 2) + 3^x * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_g_derivative_l3560_356067


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_two_three_l3560_356060

theorem chinese_remainder_theorem_two_three :
  (∀ (a b : ℤ), ∃ (x : ℤ), x % 5 = a % 5 ∧ x % 6 = b % 6 ∧ x = 6*a + 25*b) ∧
  (∀ (a b c : ℤ), ∃ (y : ℤ), y % 5 = a % 5 ∧ y % 6 = b % 6 ∧ y % 7 = c % 7 ∧ y = 126*a + 175*b + 120*c) :=
by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_two_three_l3560_356060


namespace NUMINAMATH_CALUDE_negative_one_squared_plus_cubed_equals_zero_l3560_356076

theorem negative_one_squared_plus_cubed_equals_zero :
  (-1 : ℤ)^2 + (-1 : ℤ)^3 = 0 := by sorry

end NUMINAMATH_CALUDE_negative_one_squared_plus_cubed_equals_zero_l3560_356076


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l3560_356001

theorem polygon_interior_angle_sum (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 144 → 
  (n - 2) * 180 = n * interior_angle :=
by
  sorry

#check polygon_interior_angle_sum

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l3560_356001


namespace NUMINAMATH_CALUDE_last_ball_is_white_l3560_356095

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState :=
  (white : Nat)
  (black : Nat)

/-- The process of drawing balls and applying rules -/
def drawProcess (state : BoxState) : BoxState :=
  sorry

/-- The final state of the box after the process ends -/
def finalState (initial : BoxState) : BoxState :=
  sorry

/-- Theorem stating that the last ball is always white -/
theorem last_ball_is_white (initial : BoxState) :
  initial.white = 2011 → initial.black = 2012 →
  (finalState initial).white = 1 ∧ (finalState initial).black = 0 :=
sorry

end NUMINAMATH_CALUDE_last_ball_is_white_l3560_356095


namespace NUMINAMATH_CALUDE_crocodile_coloring_exists_l3560_356035

/-- A coloring function for an infinite checkerboard -/
def ColoringFunction := ℤ → ℤ → Fin 2

/-- The "crocodile" move on a checkerboard -/
def crocodileMove (m n : ℤ) (x y : ℤ) : Set (ℤ × ℤ) :=
  {(x + m, y + n), (x + m, y - n), (x - m, y + n), (x - m, y - n),
   (x + n, y + m), (x + n, y - m), (x - n, y + m), (x - n, y - m)}

/-- Theorem: For any m and n, there exists a coloring function such that
    any two squares connected by a crocodile move have different colors -/
theorem crocodile_coloring_exists (m n : ℤ) :
  ∃ (f : ColoringFunction),
    ∀ (x y : ℤ), ∀ (x' y' : ℤ), (x', y') ∈ crocodileMove m n x y →
      f x y ≠ f x' y' := by
  sorry

end NUMINAMATH_CALUDE_crocodile_coloring_exists_l3560_356035


namespace NUMINAMATH_CALUDE_system_positive_solution_l3560_356053

theorem system_positive_solution (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ - x₂ = a ∧ 
    x₃ - x₄ = b ∧ 
    x₁ + x₂ + x₃ + x₄ = 1 ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) ↔ 
  abs a + abs b < 1 :=
by sorry

end NUMINAMATH_CALUDE_system_positive_solution_l3560_356053


namespace NUMINAMATH_CALUDE_exists_floating_polyhedron_with_properties_l3560_356014

/-- A convex polyhedron floating in water -/
structure FloatingPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  submergedVolume : ℝ
  surfaceAreaAboveWater : ℝ
  volume_pos : 0 < volume
  surfaceArea_pos : 0 < surfaceArea
  submergedVolume_le_volume : submergedVolume ≤ volume
  surfaceAreaAboveWater_le_surfaceArea : surfaceAreaAboveWater ≤ surfaceArea

/-- Theorem stating the existence of a floating polyhedron with specific properties -/
theorem exists_floating_polyhedron_with_properties :
  ∀ ε > 0, ∃ (P : FloatingPolyhedron),
    P.submergedVolume / P.volume > 1 - ε ∧
    P.surfaceAreaAboveWater / P.surfaceArea > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_floating_polyhedron_with_properties_l3560_356014


namespace NUMINAMATH_CALUDE_smallest_sum_of_ten_numbers_l3560_356040

theorem smallest_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 ∧ 
  (∀ T ⊆ S, T.card = 5 → Even (T.prod id)) ∧
  Odd (S.sum id) →
  65 ≤ S.sum id :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_ten_numbers_l3560_356040


namespace NUMINAMATH_CALUDE_john_twice_frank_age_l3560_356000

/-- Given that Frank is 15 years younger than John and Frank will be 16 in 4 years,
    prove that John will be twice as old as Frank in 3 years. -/
theorem john_twice_frank_age (frank_age john_age x : ℕ) : 
  john_age = frank_age + 15 →
  frank_age + 4 = 16 →
  john_age + x = 2 * (frank_age + x) →
  x = 3 := by sorry

end NUMINAMATH_CALUDE_john_twice_frank_age_l3560_356000


namespace NUMINAMATH_CALUDE_jordan_list_count_l3560_356062

def smallest_square_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^2) n

def smallest_cube_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^3) n

theorem jordan_list_count : 
  let lower_bound := smallest_square_multiple 30
  let upper_bound := smallest_cube_multiple 30
  (upper_bound - lower_bound) / 30 + 1 = 871 := by sorry

end NUMINAMATH_CALUDE_jordan_list_count_l3560_356062


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l3560_356026

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def six_digit_number (B : ℕ) : ℕ := 303700 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303703 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l3560_356026


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3560_356057

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set of x that satisfy p
def p_set : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def q_set : Set ℝ := {x | q x}

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (q_set ⊆ p_set) ∧ ¬(p_set ⊆ q_set) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3560_356057


namespace NUMINAMATH_CALUDE_middle_number_proof_l3560_356043

theorem middle_number_proof (A B C : ℝ) (h1 : A < B) (h2 : B < C) 
  (h3 : B - C = A - B) (h4 : A * B = 85) (h5 : B * C = 115) : B = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3560_356043


namespace NUMINAMATH_CALUDE_log_equality_l3560_356025

theorem log_equality (a b : ℝ) (h1 : a = Real.log 484 / Real.log 4) (h2 : b = Real.log 22 / Real.log 2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3560_356025


namespace NUMINAMATH_CALUDE_expression_value_for_x_3_l3560_356077

theorem expression_value_for_x_3 :
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_for_x_3_l3560_356077


namespace NUMINAMATH_CALUDE_integer_roots_count_l3560_356065

theorem integer_roots_count : 
  let lower_bound := -5 - Real.sqrt 42
  let upper_bound := -5 + Real.sqrt 42
  let is_valid_root (x : ℤ) := 
    (Real.cos (2 * π * ↑x) + Real.cos (π * ↑x) = Real.sin (3 * π * ↑x) + Real.sin (π * ↑x)) ∧
    (lower_bound < x) ∧ (x < upper_bound)
  ∃! (roots : Finset ℤ), (Finset.card roots = 7) ∧ (∀ x, x ∈ roots ↔ is_valid_root x) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_count_l3560_356065


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l3560_356042

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) 
  (h2 : crayons_per_pack = 15) : 
  total_crayons / crayons_per_pack = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l3560_356042


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_two_l3560_356087

theorem imaginary_sum_equals_two (i : ℂ) (hi : i^2 = -1) :
  i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_two_l3560_356087


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l3560_356097

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessSubscription where
  /-- Subscription amount of person C -/
  c_amount : ℕ
  /-- Total profit of the business -/
  total_profit : ℕ
  /-- Profit received by person C -/
  c_profit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c_amount + 14000

/-- Theorem stating that the total subscription amount is 50,000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.total_profit = 35000)
  (h2 : bs.c_profit = 8400)
  (h3 : bs.c_profit * (total_subscription bs) = bs.total_profit * bs.c_amount) :
  total_subscription bs = 50000 := by
  sorry

#eval total_subscription { c_amount := 12000, total_profit := 35000, c_profit := 8400 }

end NUMINAMATH_CALUDE_total_subscription_is_50000_l3560_356097


namespace NUMINAMATH_CALUDE_relationship_abc_l3560_356029

theorem relationship_abc : 
  let a := (1/2)^(2/3)
  let b := (1/3)^(1/3)
  let c := Real.log 3
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3560_356029


namespace NUMINAMATH_CALUDE_book_price_percentage_l3560_356049

/-- The percentage of the suggested retail price that Bob paid for a book -/
theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let bob_paid := 0.6 * marked_price
  bob_paid / suggested_retail_price = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_book_price_percentage_l3560_356049


namespace NUMINAMATH_CALUDE_andy_wrong_answers_l3560_356009

/-- Represents the number of wrong answers for each person in a 30-question test. -/
structure TestResults where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The conditions of the problem and the theorem to be proved. -/
theorem andy_wrong_answers (t : TestResults) : 
  t.andy + t.beth = t.charlie + t.daniel →
  t.andy + t.daniel = t.beth + t.charlie + 4 →
  t.charlie = 5 →
  t.andy = 7 := by
  sorry

end NUMINAMATH_CALUDE_andy_wrong_answers_l3560_356009


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l3560_356041

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = red_balls + white_balls) (h2 : total_balls = 5) 
  (h3 : red_balls = 3) (h4 : white_balls = 2) (h5 : drawn_balls = 3) : 
  1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l3560_356041


namespace NUMINAMATH_CALUDE_cos_sin_eighteen_degrees_identity_l3560_356083

theorem cos_sin_eighteen_degrees_identity : 
  4 * (Real.cos (18 * π / 180))^2 - 1 = 1 / (4 * (Real.sin (18 * π / 180))^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_eighteen_degrees_identity_l3560_356083


namespace NUMINAMATH_CALUDE_matrix_power_four_l3560_356092

/-- The fourth power of a specific 2x2 matrix equals a specific result -/
theorem matrix_power_four : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]
  A^4 = !![(-8 : ℝ), 8 * Real.sqrt 3; -8 * Real.sqrt 3, (-8 : ℝ)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3560_356092


namespace NUMINAMATH_CALUDE_mary_stickers_left_l3560_356069

/-- The number of stickers Mary has left over after distributing them in class -/
def stickers_left_over (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (total_students : ℕ) (stickers_per_other : ℕ) : ℕ :=
  total_stickers - 
  (num_friends * stickers_per_friend + 
   (total_students - 1 - num_friends) * stickers_per_other)

/-- Theorem stating that Mary has 8 stickers left over -/
theorem mary_stickers_left : stickers_left_over 50 5 4 17 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_left_l3560_356069


namespace NUMINAMATH_CALUDE_car_trip_speed_l3560_356011

/-- Proves that given the conditions of the car trip, the return speed must be 37.5 mph -/
theorem car_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  ∃ speed_ba : ℝ,
    speed_ba = 37.5 ∧
    avg_speed = (2 * distance) / (distance / speed_ab + distance / speed_ba) :=
by sorry

end NUMINAMATH_CALUDE_car_trip_speed_l3560_356011


namespace NUMINAMATH_CALUDE_spade_calculation_l3560_356046

/-- The ⊙ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 5 ⊙ (6 ⊙ 3) = -704 -/
theorem spade_calculation : spade 5 (spade 6 3) = -704 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3560_356046


namespace NUMINAMATH_CALUDE_josh_new_marbles_l3560_356075

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 8

/-- The additional marbles Josh found compared to those he lost -/
def additional_marbles : ℕ := 2

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := marbles_lost + additional_marbles

theorem josh_new_marbles : new_marbles = 10 := by sorry

end NUMINAMATH_CALUDE_josh_new_marbles_l3560_356075


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_area_inequality_l3560_356071

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the perimeter of a quadrilateral
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the perimeter of the quadrilateral formed by the centers of inscribed circles
def inscribed_centers_perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Statement of the theorem
theorem quadrilateral_perimeter_area_inequality (q : ConvexQuadrilateral) :
  perimeter q * inscribed_centers_perimeter q > 4 * area q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_area_inequality_l3560_356071


namespace NUMINAMATH_CALUDE_six_awards_four_students_l3560_356034

/-- The number of ways to distribute awards to students. -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways to distribute 6 awards to 4 students. -/
theorem six_awards_four_students :
  distribute_awards 6 4 = 1260 :=
sorry

end NUMINAMATH_CALUDE_six_awards_four_students_l3560_356034


namespace NUMINAMATH_CALUDE_time_after_classes_l3560_356064

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt60 : minutes < 60

/-- Adds a duration in minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    hLt60 := by sorry }

/-- The starting time of classes -/
def startTime : Time := { hours := 12, minutes := 0, hLt60 := by simp }

/-- The number of completed classes -/
def completedClasses : ℕ := 4

/-- The duration of each class in minutes -/
def classDuration : ℕ := 45

/-- Theorem: After 4 classes of 45 minutes each, starting at 12 pm, the time is 3 pm -/
theorem time_after_classes :
  (addMinutes startTime (completedClasses * classDuration)).hours = 15 := by sorry

end NUMINAMATH_CALUDE_time_after_classes_l3560_356064


namespace NUMINAMATH_CALUDE_trig_identity_l3560_356093

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3560_356093


namespace NUMINAMATH_CALUDE_daniels_painting_area_l3560_356047

/-- Calculate the total paintable wall area for multiple bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- The problem statement -/
theorem daniels_painting_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end NUMINAMATH_CALUDE_daniels_painting_area_l3560_356047


namespace NUMINAMATH_CALUDE_line_slope_is_pi_over_three_l3560_356096

theorem line_slope_is_pi_over_three (x y : ℝ) :
  2 * Real.sqrt 3 * x - 2 * y - 1 = 0 →
  ∃ (m : ℝ), (∀ x y, y = m * x - 1 / 2) ∧ m = Real.tan (π / 3) :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_pi_over_three_l3560_356096


namespace NUMINAMATH_CALUDE_square_with_circles_theorem_l3560_356006

/-- A square with side length 6 and three congruent circles inside it -/
structure SquareWithCircles where
  /-- Side length of the square -/
  side_length : ℝ
  side_length_eq : side_length = 6
  /-- Radius of the congruent circles -/
  radius : ℝ
  /-- Center of circle X -/
  center_x : ℝ × ℝ
  /-- Center of circle Y -/
  center_y : ℝ × ℝ
  /-- Center of circle Z -/
  center_z : ℝ × ℝ
  /-- X is tangent to sides AB and AD -/
  x_tangent : center_x.1 = radius ∧ center_x.2 = radius
  /-- Y is tangent to sides AB and BC -/
  y_tangent : center_y.1 = side_length - radius ∧ center_y.2 = radius
  /-- Z is tangent to side CD and both circles X and Y -/
  z_tangent : center_z.1 = side_length / 2 ∧ center_z.2 = side_length - radius

/-- The theorem to be proved -/
theorem square_with_circles_theorem (s : SquareWithCircles) :
  ∃ (m n : ℕ), s.radius = m - Real.sqrt n ∧ m + n = 195 := by
  sorry

end NUMINAMATH_CALUDE_square_with_circles_theorem_l3560_356006


namespace NUMINAMATH_CALUDE_nontrivial_solution_iff_l3560_356078

/-- A system of linear equations with coefficients a, b, c has a non-trivial solution -/
def has_nontrivial_solution (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    a * x + b * y + c * z = 0 ∧
    b * x + c * y + a * z = 0 ∧
    c * x + a * y + b * z = 0

/-- The main theorem characterizing when the system has a non-trivial solution -/
theorem nontrivial_solution_iff (a b c : ℝ) :
  has_nontrivial_solution a b c ↔ a + b + c = 0 ∨ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_nontrivial_solution_iff_l3560_356078


namespace NUMINAMATH_CALUDE_symmetry_about_x_2_symmetry_about_2_0_l3560_356050

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Theorem for symmetry about x = 2
theorem symmetry_about_x_2 (h : ∀ x, f (1 - x) = f (3 + x)) :
  ∀ x, f (2 - x) = f (2 + x) := by sorry

-- Theorem for symmetry about (2,0)
theorem symmetry_about_2_0 (h : ∀ x, f (1 - x) = -f (3 + x)) :
  ∀ x, f (2 - x) = -f (2 + x) := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_2_symmetry_about_2_0_l3560_356050


namespace NUMINAMATH_CALUDE_f_max_value_l3560_356030

/-- The function f(x) = 8x - 3x^2 -/
def f (x : ℝ) : ℝ := 8 * x - 3 * x^2

/-- The maximum value of f(x) for any real x is 16/3 -/
theorem f_max_value : ∃ (M : ℝ), M = 16/3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3560_356030


namespace NUMINAMATH_CALUDE_tangent_lines_and_intersection_points_l3560_356052

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line l
def l (m x y : ℝ) : Prop := m * x - y - m + 1 = 0

theorem tangent_lines_and_intersection_points :
  -- Part 1: Tangent lines
  (∀ x y : ℝ, (x + 2*y - 7 = 0 → C x y) ∧ (2*x - y - 4 = 0 → C x y)) ∧
  (x + 2*y - 7 = 0 → x = M.1 ∧ y = M.2) ∧
  (2*x - y - 4 = 0 → x = M.1 ∧ y = M.2) ∧
  -- Part 2: Intersection points
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    l m A.1 A.2 ∧ l m B.1 B.2 ∧ 
    C A.1 A.2 ∧ C B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 17) → 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_tangent_lines_and_intersection_points_l3560_356052


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3560_356048

/-- The distance from the point (1, 0) to the line x - y + 1 = 0 is √2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 1 = 0
  Real.sqrt 2 = (|1 - 0 + 1|) / Real.sqrt (1^2 + (-1)^2) := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3560_356048


namespace NUMINAMATH_CALUDE_problem_statement_l3560_356086

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 10/3) 
  (h2 : x * y = 144) : 
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3560_356086


namespace NUMINAMATH_CALUDE_strawberries_left_l3560_356028

/-- Given an initial amount of strawberries and amounts eaten over two days, 
    calculate the remaining amount. -/
def remaining_strawberries (initial : ℝ) (eaten_day1 : ℝ) (eaten_day2 : ℝ) : ℝ :=
  initial - eaten_day1 - eaten_day2

/-- Theorem stating that given the specific amounts in the problem, 
    the remaining amount of strawberries is 0.5 kg. -/
theorem strawberries_left : 
  remaining_strawberries 1.6 0.8 0.3 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l3560_356028


namespace NUMINAMATH_CALUDE_temperature_at_night_l3560_356036

/-- Given the temperature changes throughout a day, prove the final temperature at night. -/
theorem temperature_at_night 
  (noon_temp : ℤ) 
  (afternoon_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : noon_temp = 5)
  (h2 : afternoon_temp = 7)
  (h3 : temp_drop = 9) : 
  afternoon_temp - temp_drop = -2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_night_l3560_356036


namespace NUMINAMATH_CALUDE_baseball_league_games_l3560_356055

/-- The number of games played in a baseball league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with every other team,
    the total number of games played is 180. --/
theorem baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_l3560_356055


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3560_356022

theorem unique_solution_for_equation : ∃! (n k : ℕ), k^5 + 5*n^4 = 81*k ∧ n = 2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3560_356022


namespace NUMINAMATH_CALUDE_mens_tshirt_interval_l3560_356027

/-- Represents the shop selling T-shirts -/
structure TShirtShop where
  womens_interval : ℕ  -- Minutes between women's T-shirt sales
  womens_price : ℕ     -- Price of women's T-shirts in dollars
  mens_price : ℕ        -- Price of men's T-shirts in dollars
  daily_hours : ℕ      -- Hours open per day
  weekly_days : ℕ      -- Days open per week
  weekly_revenue : ℕ   -- Total weekly revenue in dollars

/-- Calculates the interval between men's T-shirt sales -/
def mens_interval (shop : TShirtShop) : ℕ :=
  sorry

/-- Theorem stating that the men's T-shirt sale interval is 40 minutes -/
theorem mens_tshirt_interval (shop : TShirtShop) 
  (h1 : shop.womens_interval = 30)
  (h2 : shop.womens_price = 18)
  (h3 : shop.mens_price = 15)
  (h4 : shop.daily_hours = 12)
  (h5 : shop.weekly_days = 7)
  (h6 : shop.weekly_revenue = 4914) :
  mens_interval shop = 40 := by
    sorry

end NUMINAMATH_CALUDE_mens_tshirt_interval_l3560_356027


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3560_356033

-- Define a circle with center (1, 1) passing through (0, 0)
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 0) → 
      (x - 1)^2 + (y - 1)^2 = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l3560_356033


namespace NUMINAMATH_CALUDE_ellipse_property_l3560_356037

-- Define the basic concepts
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a moving point
def MovingPoint := ℝ → Point

-- Define the concept of an ellipse
def is_ellipse (trajectory : MovingPoint) : Prop := sorry

-- Define the concept of constant sum of distances
def constant_sum_distances (trajectory : MovingPoint) (f1 f2 : Point) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, distance (trajectory t) f1 + distance (trajectory t) f2 = k

-- State the theorem
theorem ellipse_property :
  (∀ trajectory : MovingPoint, ∀ f1 f2 : Point,
    is_ellipse trajectory → constant_sum_distances trajectory f1 f2) ∧
  (∃ trajectory : MovingPoint, ∃ f1 f2 : Point,
    constant_sum_distances trajectory f1 f2 ∧ ¬is_ellipse trajectory) :=
sorry

end NUMINAMATH_CALUDE_ellipse_property_l3560_356037


namespace NUMINAMATH_CALUDE_cube_cutting_l3560_356085

theorem cube_cutting (n : ℕ) : 
  (6 * (n - 2) * (n - 2) = 54) → (n^3 = 125) := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l3560_356085


namespace NUMINAMATH_CALUDE_expected_value_is_190_l3560_356038

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six

/-- The probability of rolling a specific outcome -/
def prob (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Six => 1/2
  | _ => 1/10

/-- The monetary value associated with each outcome -/
def value (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One => 1
  | DieOutcome.Two => 1
  | DieOutcome.Three => 1
  | DieOutcome.Four => 1
  | DieOutcome.Five => -10
  | DieOutcome.Six => 5

/-- The expected value of rolling the die -/
def expectedValue : ℚ :=
  (prob DieOutcome.One * value DieOutcome.One) +
  (prob DieOutcome.Two * value DieOutcome.Two) +
  (prob DieOutcome.Three * value DieOutcome.Three) +
  (prob DieOutcome.Four * value DieOutcome.Four) +
  (prob DieOutcome.Five * value DieOutcome.Five) +
  (prob DieOutcome.Six * value DieOutcome.Six)

theorem expected_value_is_190 : expectedValue = 19/10 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_190_l3560_356038


namespace NUMINAMATH_CALUDE_probability_red_then_blue_probability_red_then_blue_proof_l3560_356081

/-- The probability of drawing a red marble first and a blue marble second from a bag containing 
    4 red marbles and 6 blue marbles, when drawing two marbles sequentially without replacement. -/
theorem probability_red_then_blue (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : ℚ :=
  4 / 15

/-- Proof of the theorem -/
theorem probability_red_then_blue_proof (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : 
    probability_red_then_blue red blue h_red h_blue = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_blue_probability_red_then_blue_proof_l3560_356081


namespace NUMINAMATH_CALUDE_max_participants_l3560_356016

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Represents a chess tournament -/
structure ChessTournament where
  participants : Nat
  results : Fin participants → Fin participants → GameResult

/-- Calculates the score of a player against two other players -/
def score (t : ChessTournament) (p1 p2 p3 : Fin t.participants) : Rat :=
  let s1 := match t.results p1 p2 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  let s2 := match t.results p1 p3 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  s1 + s2

/-- The tournament satisfies the given conditions -/
def validTournament (t : ChessTournament) : Prop :=
  (∀ p1 p2 : Fin t.participants, p1 ≠ p2 → t.results p1 p2 ≠ t.results p2 p1) ∧
  (∀ p1 p2 p3 : Fin t.participants, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    (score t p1 p2 p3 = 3/2 ∨ score t p2 p1 p3 = 3/2 ∨ score t p3 p1 p2 = 3/2))

/-- The maximum number of participants in a valid tournament is 5 -/
theorem max_participants : ∀ t : ChessTournament, validTournament t → t.participants ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_participants_l3560_356016


namespace NUMINAMATH_CALUDE_oliver_remaining_money_l3560_356008

def initial_cash : ℕ := 40
def initial_quarters : ℕ := 200
def quarter_value : ℚ := 0.25
def cash_to_sister : ℕ := 5
def quarters_to_sister : ℕ := 120

theorem oliver_remaining_money :
  let total_initial := initial_cash + initial_quarters * quarter_value
  let total_to_sister := cash_to_sister + quarters_to_sister * quarter_value
  total_initial - total_to_sister = 55 := by
sorry

end NUMINAMATH_CALUDE_oliver_remaining_money_l3560_356008


namespace NUMINAMATH_CALUDE_range_of_a_l3560_356070

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 12*x + 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(q x a) → ¬(p x)) →
  (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3560_356070


namespace NUMINAMATH_CALUDE_roller_coaster_cost_l3560_356007

theorem roller_coaster_cost (total_tickets : ℕ) (ferris_wheel_cost : ℕ) 
  (h1 : total_tickets = 13)
  (h2 : ferris_wheel_cost = 5)
  (h3 : ∃ x : ℕ, total_tickets = ferris_wheel_cost + x + x) :
  ∃ roller_coaster_cost : ℕ, roller_coaster_cost = 4 ∧ 
    total_tickets = ferris_wheel_cost + roller_coaster_cost + roller_coaster_cost :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_l3560_356007


namespace NUMINAMATH_CALUDE_calculation_problems_l3560_356084

theorem calculation_problems :
  (((1 : ℚ) / 2 - 5 / 9 + 7 / 12) * (-36 : ℚ) = -19) ∧
  ((-199 - 24 / 25) * (5 : ℚ) = -999 - 4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_calculation_problems_l3560_356084


namespace NUMINAMATH_CALUDE_derivative_equals_negative_function_l3560_356044

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- State the theorem
theorem derivative_equals_negative_function (x₀ : ℝ) :
  x₀ ≠ 0 → -- Ensure x₀ is not zero to avoid division by zero
  (deriv f) x₀ = -f x₀ →
  x₀ = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_derivative_equals_negative_function_l3560_356044


namespace NUMINAMATH_CALUDE_snake_length_problem_l3560_356074

theorem snake_length_problem (penny_snake : ℕ) (jake_snake : ℕ) : 
  jake_snake = penny_snake + 12 →
  jake_snake + penny_snake = 70 →
  jake_snake = 41 := by
sorry

end NUMINAMATH_CALUDE_snake_length_problem_l3560_356074


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l3560_356005

theorem quadratic_solution_square (x : ℝ) :
  7 * x^2 + 6 = 5 * x + 11 →
  (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l3560_356005


namespace NUMINAMATH_CALUDE_ladder_wood_length_l3560_356019

/-- Calculates the total length of wood needed for ladder rungs -/
theorem ladder_wood_length 
  (rung_length : ℚ)      -- Length of each rung in inches
  (rung_spacing : ℚ)     -- Space between rungs in inches
  (climb_height : ℚ)     -- Height to climb in feet
  (h1 : rung_length = 18)
  (h2 : rung_spacing = 6)
  (h3 : climb_height = 50) :
  (climb_height * 12 / rung_spacing) * (rung_length / 12) = 150 :=
by sorry

end NUMINAMATH_CALUDE_ladder_wood_length_l3560_356019


namespace NUMINAMATH_CALUDE_multiply_65_55_l3560_356024

theorem multiply_65_55 : 65 * 55 = 3575 := by sorry

end NUMINAMATH_CALUDE_multiply_65_55_l3560_356024


namespace NUMINAMATH_CALUDE_wall_bricks_l3560_356066

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time1 : ℝ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time2 : ℝ := 12

/-- Represents the reduction in productivity when working together (in bricks per hour) -/
def reduction : ℝ := 15

/-- Represents the time taken by both bricklayers working together to build the wall -/
def timeJoint : ℝ := 6

/-- Represents the total number of bricks in the wall -/
def totalBricks : ℝ := 360

theorem wall_bricks : 
  timeJoint * (totalBricks / time1 + totalBricks / time2 - reduction) = totalBricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l3560_356066


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l3560_356063

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l3560_356063


namespace NUMINAMATH_CALUDE_inequality_solution_l3560_356080

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≤ 5/4) ↔ (x < -8 ∨ (-2 < x ∧ x ≤ -8/5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3560_356080


namespace NUMINAMATH_CALUDE_emily_garden_seeds_l3560_356013

theorem emily_garden_seeds (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : small_gardens = 3)
  (h3 : seeds_per_small_garden = 4) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 29 := by
  sorry

end NUMINAMATH_CALUDE_emily_garden_seeds_l3560_356013


namespace NUMINAMATH_CALUDE_elastic_band_radius_increase_l3560_356004

theorem elastic_band_radius_increase (r₁ r₂ : ℝ) : 
  2 * π * r₁ = 40 →  -- Initial circumference
  2 * π * r₂ = 80 →  -- Final circumference
  r₂ - r₁ = 20 / π := by
  sorry

end NUMINAMATH_CALUDE_elastic_band_radius_increase_l3560_356004


namespace NUMINAMATH_CALUDE_like_terms_sum_l3560_356079

theorem like_terms_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^2 * y^4 = 3 * y - x^n * y^(2*m)) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l3560_356079


namespace NUMINAMATH_CALUDE_missing_number_equation_l3560_356073

theorem missing_number_equation (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l3560_356073


namespace NUMINAMATH_CALUDE_smallest_number_above_threshold_l3560_356061

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let above_threshold := numbers.filter (λ x => x > threshold)
  above_threshold.minimum? = some 1.2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_above_threshold_l3560_356061


namespace NUMINAMATH_CALUDE_gcd_98_63_l3560_356098

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l3560_356098


namespace NUMINAMATH_CALUDE_total_vowels_written_l3560_356090

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 4

/-- Theorem: The total number of vowels written on the board is 20 -/
theorem total_vowels_written : num_vowels * times_written = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_vowels_written_l3560_356090


namespace NUMINAMATH_CALUDE_charity_draw_winnings_calculation_l3560_356054

/-- Calculates the charity draw winnings given initial amount, expenses, lottery winnings, and final amount -/
def charity_draw_winnings (initial_amount expenses lottery_winnings final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - expenses + lottery_winnings)

/-- Theorem stating that given the specific values from the problem, the charity draw winnings must be 19 -/
theorem charity_draw_winnings_calculation :
  charity_draw_winnings 10 (4 + 1 + 1) 65 94 = 19 := by
  sorry

end NUMINAMATH_CALUDE_charity_draw_winnings_calculation_l3560_356054


namespace NUMINAMATH_CALUDE_raisins_amount_l3560_356082

/-- The amount of peanuts used in the trail mix -/
def peanuts : ℝ := 0.16666666666666666

/-- The amount of chocolate chips used in the trail mix -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The total amount of trail mix -/
def total_mix : ℝ := 0.4166666666666667

/-- The amount of raisins used in the trail mix -/
def raisins : ℝ := total_mix - (peanuts + chocolate_chips)

theorem raisins_amount : raisins = 0.08333333333333337 := by sorry

end NUMINAMATH_CALUDE_raisins_amount_l3560_356082


namespace NUMINAMATH_CALUDE_keith_missed_games_l3560_356020

theorem keith_missed_games (total_games : ℕ) (attended_games : ℕ) 
  (h1 : total_games = 8)
  (h2 : attended_games = 4) :
  total_games - attended_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_missed_games_l3560_356020


namespace NUMINAMATH_CALUDE_number_of_parents_l3560_356010

theorem number_of_parents (girls : ℕ) (boys : ℕ) (playgroups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  playgroups = 3 → 
  group_size = 25 → 
  playgroups * group_size - (girls + boys) = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_parents_l3560_356010


namespace NUMINAMATH_CALUDE_pedal_to_original_triangle_l3560_356023

/-- Given the sides of a pedal triangle, calculate the sides of the original triangle --/
theorem pedal_to_original_triangle 
  (a₁ b₁ c₁ : ℝ) 
  (h_pos : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁) :
  ∃ (a b c : ℝ),
    let s₁ := (a₁ + b₁ + c₁) / 2
    a = a₁ * Real.sqrt (b₁ * c₁ / ((s₁ - b₁) * (s₁ - c₁))) ∧
    b = b₁ * Real.sqrt (a₁ * c₁ / ((s₁ - a₁) * (s₁ - c₁))) ∧
    c = c₁ * Real.sqrt (a₁ * b₁ / ((s₁ - a₁) * (s₁ - b₁))) :=
by
  sorry


end NUMINAMATH_CALUDE_pedal_to_original_triangle_l3560_356023


namespace NUMINAMATH_CALUDE_marias_salary_l3560_356003

theorem marias_salary (S : ℝ) : 
  (S * 0.2 + S * 0.05 + (S - S * 0.2 - S * 0.05) * 0.25 + 1125 = S) → S = 2000 := by
sorry

end NUMINAMATH_CALUDE_marias_salary_l3560_356003


namespace NUMINAMATH_CALUDE_quadratic_maximum_l3560_356091

theorem quadratic_maximum (r : ℝ) : 
  -7 * r^2 + 50 * r - 20 ≤ 5 ∧ ∃ r, -7 * r^2 + 50 * r - 20 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l3560_356091


namespace NUMINAMATH_CALUDE_least_months_to_triple_l3560_356056

def interest_rate : ℝ := 1.06

theorem least_months_to_triple (t : ℕ) : t = 19 ↔ 
  (∀ n : ℕ, n < 19 → interest_rate ^ n ≤ 3) ∧ 
  interest_rate ^ 19 > 3 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l3560_356056


namespace NUMINAMATH_CALUDE_group_size_problem_l3560_356051

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 4624) : ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3560_356051


namespace NUMINAMATH_CALUDE_identify_brothers_l3560_356021

-- Define the brothers
inductive Brother
| trulya
| tralya

-- Define a function to represent whether a brother tells the truth
def tellsTruth : Brother → Prop
| Brother.trulya => true
| Brother.tralya => false

-- Define the statements made by the brothers
def firstBrotherStatement (first second : Brother) : Prop :=
  first = Brother.trulya

def secondBrotherStatement (first second : Brother) : Prop :=
  second = Brother.tralya

def cardSuitStatement : Prop := false  -- Cards are not of the same suit

-- The main theorem
theorem identify_brothers :
  ∃ (first second : Brother),
    first ≠ second ∧
    (tellsTruth first → firstBrotherStatement first second) ∧
    (tellsTruth second → secondBrotherStatement first second) ∧
    (tellsTruth first → cardSuitStatement) ∧
    first = Brother.tralya ∧
    second = Brother.trulya :=
  sorry

end NUMINAMATH_CALUDE_identify_brothers_l3560_356021
