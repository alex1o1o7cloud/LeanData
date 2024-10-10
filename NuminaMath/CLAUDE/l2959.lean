import Mathlib

namespace gcd_power_two_minus_one_l2959_295920

theorem gcd_power_two_minus_one :
  Nat.gcd (2^2024 - 1) (2^2015 - 1) = 2^9 - 1 := by
sorry

end gcd_power_two_minus_one_l2959_295920


namespace floor_sum_inequality_floor_fractional_part_max_n_value_l2959_295998

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Proposition B
theorem floor_sum_inequality (x y : ℝ) : floor x + floor y ≤ floor (x + y) := by sorry

-- Proposition C
theorem floor_fractional_part (x : ℝ) : 0 ≤ x - floor x ∧ x - floor x < 1 := by sorry

-- Proposition D
def satisfies_conditions (t : ℝ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range (n - 2), floor (t ^ (k + 3)) = k + 1

theorem max_n_value :
  (∃ n : ℕ, n > 2 ∧ ∃ t : ℝ, satisfies_conditions t n) →
  (∀ n : ℕ, n > 5 → ¬∃ t : ℝ, satisfies_conditions t n) := by sorry

end floor_sum_inequality_floor_fractional_part_max_n_value_l2959_295998


namespace copper_zinc_mass_ranges_l2959_295964

/-- Represents the properties of a copper-zinc mixture -/
structure CopperZincMixture where
  total_mass : ℝ
  total_volume : ℝ
  copper_density_min : ℝ
  copper_density_max : ℝ
  zinc_density_min : ℝ
  zinc_density_max : ℝ

/-- Theorem stating the mass ranges of copper and zinc in the mixture -/
theorem copper_zinc_mass_ranges (mixture : CopperZincMixture)
  (h_total_mass : mixture.total_mass = 400)
  (h_total_volume : mixture.total_volume = 50)
  (h_copper_density : mixture.copper_density_min = 8.8 ∧ mixture.copper_density_max = 9)
  (h_zinc_density : mixture.zinc_density_min = 7.1 ∧ mixture.zinc_density_max = 7.2) :
  ∃ (copper_mass zinc_mass : ℝ),
    200 ≤ copper_mass ∧ copper_mass ≤ 233 ∧
    167 ≤ zinc_mass ∧ zinc_mass ≤ 200 ∧
    copper_mass + zinc_mass = mixture.total_mass ∧
    copper_mass / mixture.copper_density_max + zinc_mass / mixture.zinc_density_min = mixture.total_volume :=
by sorry

end copper_zinc_mass_ranges_l2959_295964


namespace quadratic_equation_roots_l2959_295962

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ -1 :=
by sorry

end quadratic_equation_roots_l2959_295962


namespace square_of_sum_fifteen_three_l2959_295960

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end square_of_sum_fifteen_three_l2959_295960


namespace subset_intersection_iff_bounds_l2959_295919

theorem subset_intersection_iff_bounds (a : ℝ) : 
  let A := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) → (A ⊆ A ∩ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end subset_intersection_iff_bounds_l2959_295919


namespace tile_difference_8_9_and_9_10_l2959_295986

/-- Represents the number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2

/-- The difference in tiles between two consecutive squares -/
def tile_difference (n : ℕ) : ℕ := tiles (n + 1) - tiles n

theorem tile_difference_8_9_and_9_10 :
  (tile_difference 8 = 17) ∧ (tile_difference 9 = 19) := by
  sorry

end tile_difference_8_9_and_9_10_l2959_295986


namespace midpoint_coordinate_sum_l2959_295952

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (9, 18) is 17 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 3
  let y1 : ℝ := 4
  let x2 : ℝ := 9
  let y2 : ℝ := 18
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end midpoint_coordinate_sum_l2959_295952


namespace polar_to_rectangular_conversion_l2959_295979

theorem polar_to_rectangular_conversion 
  (r φ x y : ℝ) 
  (h1 : r = 7 / (2 * Real.cos φ - 5 * Real.sin φ))
  (h2 : x = r * Real.cos φ)
  (h3 : y = r * Real.sin φ) :
  2 * x - 5 * y = 7 := by
sorry

end polar_to_rectangular_conversion_l2959_295979


namespace second_term_is_half_l2959_295931

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 1 / 4
  property : a 3 * a 5 = 4 * (a 4 - 1)

/-- The second term of the geometric sequence is 1/2 -/
theorem second_term_is_half (seq : GeometricSequence) : seq.a 2 = 1 / 2 := by
  sorry

end second_term_is_half_l2959_295931


namespace total_posters_proof_l2959_295974

def mario_posters : ℕ := 18
def samantha_extra_posters : ℕ := 15

theorem total_posters_proof :
  mario_posters + (mario_posters + samantha_extra_posters) = 51 := by
  sorry

end total_posters_proof_l2959_295974


namespace maaza_amount_l2959_295945

/-- The amount of Pepsi in liters -/
def pepsi : ℕ := 144

/-- The amount of Sprite in liters -/
def sprite : ℕ := 368

/-- The number of cans available -/
def num_cans : ℕ := 143

/-- The function to calculate the amount of Maaza given the constraints -/
def calculate_maaza (p s c : ℕ) : ℕ :=
  c * (Nat.gcd p s) - (p + s)

/-- Theorem stating that the amount of Maaza is 1776 liters -/
theorem maaza_amount : calculate_maaza pepsi sprite num_cans = 1776 := by
  sorry

end maaza_amount_l2959_295945


namespace equal_expressions_l2959_295970

theorem equal_expressions (x : ℝ) (h : x > 0) : 
  (∃! n : ℕ, n = (if x^x + x^x = 2*x^x then 1 else 0) + 
              (if x^x + x^x = x^(2*x) then 1 else 0) + 
              (if x^x + x^x = (2*x)^x then 1 else 0) + 
              (if x^x + x^x = (2*x)^(2*x) then 1 else 0)) ∧
  (x^x + x^x = 2*x^x) ∧
  (x^x + x^x ≠ x^(2*x)) ∧
  (x^x + x^x ≠ (2*x)^x) ∧
  (x^x + x^x ≠ (2*x)^(2*x)) :=
by sorry

end equal_expressions_l2959_295970


namespace eggs_remaining_l2959_295978

theorem eggs_remaining (initial_eggs : ℕ) (eggs_removed : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_removed = 5 → eggs_left = initial_eggs - eggs_removed → eggs_left = 42 := by
  sorry

end eggs_remaining_l2959_295978


namespace quadratic_integer_value_l2959_295916

theorem quadratic_integer_value (a b c : ℚ) :
  (∀ n : ℤ, ∃ m : ℤ, a * n^2 + b * n + c = m) ↔ 
  (∃ k l m : ℤ, 2 * a = k ∧ a + b = l ∧ c = m) :=
sorry

end quadratic_integer_value_l2959_295916


namespace five_percent_problem_l2959_295967

theorem five_percent_problem (x : ℝ) : (5 / 100) * x = 12.75 → x = 255 := by
  sorry

end five_percent_problem_l2959_295967


namespace john_ducks_count_l2959_295937

/-- Proves that John bought 30 ducks given the problem conditions -/
theorem john_ducks_count :
  let cost_per_duck : ℕ := 10
  let weight_per_duck : ℕ := 4
  let price_per_pound : ℕ := 5
  let total_profit : ℕ := 300
  let num_ducks : ℕ := (total_profit / (weight_per_duck * price_per_pound - cost_per_duck))
  num_ducks = 30 := by sorry

end john_ducks_count_l2959_295937


namespace min_value_of_sum_l2959_295951

theorem min_value_of_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x) + (x*z/y) + (x*y/z) ≥ Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀^2 + y₀^2 + z₀^2 = 1 ∧
    (y₀*z₀/x₀) + (x₀*z₀/y₀) + (x₀*y₀/z₀) = Real.sqrt 3 :=
by sorry

end min_value_of_sum_l2959_295951


namespace nail_boxes_theorem_l2959_295906

theorem nail_boxes_theorem : ∃ (a b c d : ℕ), 24 * a + 23 * b + 17 * c + 16 * d = 100 := by
  sorry

end nail_boxes_theorem_l2959_295906


namespace parabola_equation_l2959_295928

/-- A parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line that passes through the focus of a parabola and intersects it at two points -/
structure IntersectingLine (P : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola_A : A.2^2 = 2 * P.p * A.1
  h_on_parabola_B : B.2^2 = 2 * P.p * B.1
  h_through_focus : True  -- We don't need to specify this condition explicitly for the proof

/-- The theorem stating the conditions and the result to be proved -/
theorem parabola_equation (P : Parabola) (L : IntersectingLine P)
  (h_length : Real.sqrt ((L.A.1 - L.B.1)^2 + (L.A.2 - L.B.2)^2) = 8)
  (h_midpoint : (L.A.1 + L.B.1) / 2 = 2) :
  P.p = 4 ∧ ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*P.p*x := by
  sorry

end parabola_equation_l2959_295928


namespace regular_rate_is_three_dollars_l2959_295980

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay based on the pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  p.regularRate * p.regularHours + 2 * p.regularRate * p.overtimeHours

/-- Theorem: Given the specified pay structure, the regular rate is $3 per hour -/
theorem regular_rate_is_three_dollars (p : PayStructure) 
    (h1 : p.regularHours = 40)
    (h2 : p.overtimeHours = 10)
    (h3 : p.totalPay = 180)
    (h4 : calculateTotalPay p = p.totalPay) : 
    p.regularRate = 3 := by
  sorry

#check regular_rate_is_three_dollars

end regular_rate_is_three_dollars_l2959_295980


namespace valid_numbers_l2959_295910

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a m X : ℕ),
    (a = 1 ∨ a = 2) ∧
    m > 0 ∧
    X < 10^(m-1) ∧
    n = a * 10^(m-1) + X ∧
    3 * n = 10 * X + a

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {142857, 285714, 428571, 571428, 714285, 857142} :=
by sorry

end valid_numbers_l2959_295910


namespace total_distance_driven_l2959_295969

theorem total_distance_driven (renaldo_distance : ℝ) (ernesto_extra : ℝ) (marcos_percentage : ℝ) : 
  renaldo_distance = 15 →
  ernesto_extra = 7 →
  marcos_percentage = 0.2 →
  let ernesto_distance := renaldo_distance / 3 + ernesto_extra
  let marcos_distance := (renaldo_distance + ernesto_distance) * (1 + marcos_percentage)
  renaldo_distance + ernesto_distance + marcos_distance = 59.4 := by
sorry

end total_distance_driven_l2959_295969


namespace interest_difference_approx_l2959_295990

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  compound_interest principal compound_rate time - simple_interest principal simple_rate time

theorem interest_difference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |interest_difference 15000 0.06 0.08 20 - 9107| < ε :=
sorry

end interest_difference_approx_l2959_295990


namespace minimum_value_problem_l2959_295903

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.sqrt 2 = Real.sqrt (2^a * 4^b) → 1/x + x/y ≤ 1/a + a/b) ∧
  (1/x + x/y = 2 * Real.sqrt 2 + 1) :=
by sorry

end minimum_value_problem_l2959_295903


namespace martha_crayons_count_l2959_295925

/-- Calculate the final number of crayons Martha has after losing half and buying new ones. -/
def final_crayons (initial : ℕ) (new_set : ℕ) : ℕ :=
  initial / 2 + new_set

/-- Theorem stating that Martha's final crayon count is correct. -/
theorem martha_crayons_count : final_crayons 18 20 = 29 := by
  sorry

end martha_crayons_count_l2959_295925


namespace initial_puppies_count_l2959_295909

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_remaining : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies remaining -/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_remaining := by sorry

end initial_puppies_count_l2959_295909


namespace matrix_product_equals_C_l2959_295946

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -2; 0, 2, 4]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 0, 2, -1; 3, 0, 1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, -3; -5, 5, -5; 12, 4, 2]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end matrix_product_equals_C_l2959_295946


namespace unique_cube_root_between_9_and_9_2_l2959_295901

theorem unique_cube_root_between_9_and_9_2 :
  ∃! n : ℕ+, 27 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.2 :=
by
  -- The proof goes here
  sorry

end unique_cube_root_between_9_and_9_2_l2959_295901


namespace nicky_running_time_l2959_295943

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 400)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  ∃ (t : ℝ), t = 30 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end nicky_running_time_l2959_295943


namespace cubic_roots_geometric_progression_l2959_295971

/-- 
A cubic polynomial with coefficients a, b, and c has roots that form 
a geometric progression if and only if a^3 * c = b^3.
-/
theorem cubic_roots_geometric_progression 
  (a b c : ℝ) : 
  (∃ x y z : ℝ, (x^3 + a*x^2 + b*x + c = 0) ∧ 
                (y^3 + a*y^2 + b*y + c = 0) ∧ 
                (z^3 + a*z^2 + b*z + c = 0) ∧ 
                (y^2 = x*z)) ↔ 
  (a^3 * c = b^3) := by
sorry

end cubic_roots_geometric_progression_l2959_295971


namespace line_PB_equation_l2959_295914

-- Define the points A, B, and P
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (5, 0)
def P : ℝ × ℝ := (2, 3)

-- Define the equations of lines PA and PB
def line_PA (x y : ℝ) : Prop := x - y + 1 = 0
def line_PB (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem line_PB_equation :
  (A.1 = -1 ∧ A.2 = 0) →  -- A is on x-axis
  (B.1 = 5 ∧ B.2 = 0) →   -- B is on x-axis
  P.1 = 2 →               -- x-coordinate of P is 2
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- PA = PB
  (∀ x y, line_PA x y ↔ x - y + 1 = 0) →  -- Equation of PA
  (∀ x y, line_PB x y ↔ x + y - 5 = 0) :=  -- Equation of PB
by sorry

end line_PB_equation_l2959_295914


namespace negation_is_false_l2959_295908

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False := by sorry

end negation_is_false_l2959_295908


namespace unique_solution_power_equation_l2959_295959

theorem unique_solution_power_equation :
  ∀ x y : ℕ+, 3^(x:ℕ) + 7 = 2^(y:ℕ) → x = 2 ∧ y = 4 := by sorry

end unique_solution_power_equation_l2959_295959


namespace intersection_integer_coordinates_l2959_295900

theorem intersection_integer_coordinates (n : ℕ+) 
  (h : ∃ (x y : ℤ), 17 * x + 7 * y = 833 ∧ y = n * x - 3) : n = 15 := by
  sorry

end intersection_integer_coordinates_l2959_295900


namespace correct_ranking_l2959_295972

-- Define the set of friends
inductive Friend : Type
| Amy : Friend
| Bill : Friend
| Celine : Friend

-- Define the age relation
def older_than : Friend → Friend → Prop := sorry

-- Define the statements
def statement_I : Prop := older_than Friend.Bill Friend.Amy ∧ older_than Friend.Bill Friend.Celine
def statement_II : Prop := ¬(older_than Friend.Amy Friend.Bill ∧ older_than Friend.Amy Friend.Celine)
def statement_III : Prop := ¬(older_than Friend.Amy Friend.Celine ∧ older_than Friend.Bill Friend.Celine)

-- Define the theorem
theorem correct_ranking :
  -- Conditions
  (∀ (x y : Friend), x ≠ y → (older_than x y ∨ older_than y x)) →
  (∀ (x y z : Friend), older_than x y → older_than y z → older_than x z) →
  (statement_I ∨ statement_II ∨ statement_III) →
  (¬statement_I ∨ ¬statement_II) →
  (¬statement_I ∨ ¬statement_III) →
  (¬statement_II ∨ ¬statement_III) →
  -- Conclusion
  older_than Friend.Amy Friend.Celine ∧ older_than Friend.Celine Friend.Bill :=
by sorry

end correct_ranking_l2959_295972


namespace solution_set_inequality_proof_l2959_295992

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 5|

-- Part 1: Solution set of f(x) < 10
theorem solution_set (x : ℝ) : f x < 10 ↔ x ∈ Set.Ioo (-19/3) (-1) := by sorry

-- Part 2: Prove |a+b| + |a-b| < f(x) given |a| < 3 and |b| < 3
theorem inequality_proof (x a b : ℝ) (ha : |a| < 3) (hb : |b| < 3) :
  |a + b| + |a - b| < f x := by sorry

end solution_set_inequality_proof_l2959_295992


namespace absolute_value_sum_difference_l2959_295957

theorem absolute_value_sum_difference : |(-8)| + (-6) - (-12) = 14 := by sorry

end absolute_value_sum_difference_l2959_295957


namespace complex_exponential_conjugate_l2959_295936

theorem complex_exponential_conjugate (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I := by
  sorry

end complex_exponential_conjugate_l2959_295936


namespace lakers_win_probability_l2959_295987

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 540/16384

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers :=
sorry

end lakers_win_probability_l2959_295987


namespace no_solution_exists_l2959_295907

theorem no_solution_exists : ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 10467 [ZMOD 7] := by
  sorry

end no_solution_exists_l2959_295907


namespace correct_calculation_l2959_295911

theorem correct_calculation : -1^4 * (-1)^3 = 1 := by
  sorry

end correct_calculation_l2959_295911


namespace paulines_garden_l2959_295989

/-- Represents the number of kinds of cucumbers in Pauline's garden -/
def cucumber_kinds : ℕ := sorry

/-- The total number of spaces in the garden -/
def total_spaces : ℕ := 10 * 15

/-- The number of tomatoes planted -/
def tomatoes : ℕ := 3 * 5

/-- The number of cucumbers planted -/
def cucumbers : ℕ := cucumber_kinds * 4

/-- The number of potatoes planted -/
def potatoes : ℕ := 30

/-- The number of additional vegetables that can be planted -/
def additional_vegetables : ℕ := 85

theorem paulines_garden :
  cucumber_kinds = 5 :=
by sorry

end paulines_garden_l2959_295989


namespace x_lt_neg_one_necessary_not_sufficient_l2959_295999

theorem x_lt_neg_one_necessary_not_sufficient :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end x_lt_neg_one_necessary_not_sufficient_l2959_295999


namespace factory_production_l2959_295940

theorem factory_production (x : ℝ) 
  (h1 : (2200 / x) - (2400 / (1.2 * x)) = 1) : x = 200 := by
  sorry

end factory_production_l2959_295940


namespace slope_product_theorem_l2959_295947

theorem slope_product_theorem (m n : ℝ) (θ₁ θ₂ : ℝ) : 
  θ₁ = 3 * θ₂ →
  m = 9 * n →
  m ≠ 0 →
  m = Real.tan θ₁ →
  n = Real.tan θ₂ →
  m * n = 27 / 13 :=
by sorry

end slope_product_theorem_l2959_295947


namespace firewood_collection_l2959_295975

theorem firewood_collection (total kimberley houston : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : houston = 12) :
  ∃ ela : ℕ, total = kimberley + houston + ela ∧ ela = 13 := by
  sorry

end firewood_collection_l2959_295975


namespace job_completion_time_l2959_295963

/-- Given a job that can be completed by a man in 10 days and his son in 20/3 days,
    prove that they can complete the job together in 4 days. -/
theorem job_completion_time (man_time son_time combined_time : ℚ) : 
  man_time = 10 → son_time = 20 / 3 → 
  combined_time = 1 / (1 / man_time + 1 / son_time) → 
  combined_time = 4 := by sorry

end job_completion_time_l2959_295963


namespace scientific_notation_equality_l2959_295924

theorem scientific_notation_equality : 3422000 = 3.422 * (10 ^ 6) := by sorry

end scientific_notation_equality_l2959_295924


namespace car_speed_calculation_car_speed_is_21_l2959_295982

/-- Calculates the speed of a car given the walking speed of a person and the number of steps taken --/
theorem car_speed_calculation (walking_speed : ℝ) (steps_while_car_visible : ℕ) (steps_after_car_disappeared : ℕ) : ℝ :=
  let total_steps := steps_while_car_visible + steps_after_car_disappeared
  let speed_ratio := total_steps / steps_while_car_visible
  speed_ratio * walking_speed

/-- Proves that the car's speed is 21 km/h given the specific conditions --/
theorem car_speed_is_21 : 
  car_speed_calculation 3.5 27 135 = 21 := by
  sorry

end car_speed_calculation_car_speed_is_21_l2959_295982


namespace rectangular_prism_parallel_edges_l2959_295902

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallelEdgePairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism with different dimensions has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallelEdgePairs prism = 12 := by
  sorry

end rectangular_prism_parallel_edges_l2959_295902


namespace largest_value_l2959_295953

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (a^2 + b^2) (max (2*a*b) (max a (1/2))) = a^2 + b^2 := by
  sorry

end largest_value_l2959_295953


namespace complex_modulus_squared_l2959_295994

theorem complex_modulus_squared (w : ℂ) (h : w + 3 * Complex.abs w = -1 + 12 * Complex.I) :
  Complex.abs w ^ 2 = 2545 := by
  sorry

end complex_modulus_squared_l2959_295994


namespace log_four_one_sixtyfourth_l2959_295988

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_four_one_sixtyfourth : log 4 (1/64) = -3 := by
  sorry

end log_four_one_sixtyfourth_l2959_295988


namespace max_belts_is_five_l2959_295934

/-- Represents the shopping problem with hats, ties, and belts. -/
structure ShoppingProblem where
  hatPrice : ℕ
  tiePrice : ℕ
  beltPrice : ℕ
  totalBudget : ℕ

/-- Represents a valid shopping solution. -/
structure ShoppingSolution where
  hats : ℕ
  ties : ℕ
  belts : ℕ

/-- Checks if a solution is valid for a given problem. -/
def isValidSolution (problem : ShoppingProblem) (solution : ShoppingSolution) : Prop :=
  solution.hats ≥ 1 ∧
  solution.ties ≥ 1 ∧
  solution.belts ≥ 1 ∧
  problem.hatPrice * solution.hats +
  problem.tiePrice * solution.ties +
  problem.beltPrice * solution.belts = problem.totalBudget

/-- The main theorem stating that the maximum number of belts is 5. -/
theorem max_belts_is_five (problem : ShoppingProblem)
    (h1 : problem.hatPrice = 3)
    (h2 : problem.tiePrice = 4)
    (h3 : problem.beltPrice = 9)
    (h4 : problem.totalBudget = 60) :
    (∀ s : ShoppingSolution, isValidSolution problem s → s.belts ≤ 5) ∧
    (∃ s : ShoppingSolution, isValidSolution problem s ∧ s.belts = 5) :=
  sorry

end max_belts_is_five_l2959_295934


namespace green_valley_olympiad_l2959_295977

theorem green_valley_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) 
  (h_participation : (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s) : j = 2 * s :=
sorry

end green_valley_olympiad_l2959_295977


namespace head_start_calculation_l2959_295930

/-- Represents a runner in the race -/
inductive Runner : Type
  | A | B | C | D

/-- The head start (in meters) that Runner A can give to another runner -/
def headStart (r : Runner) : ℕ :=
  match r with
  | Runner.A => 0
  | Runner.B => 150
  | Runner.C => 310
  | Runner.D => 400

/-- The head start one runner can give to another -/
def headStartBetween (r1 r2 : Runner) : ℤ :=
  (headStart r2 : ℤ) - (headStart r1 : ℤ)

theorem head_start_calculation :
  (headStartBetween Runner.B Runner.C = 160) ∧
  (headStartBetween Runner.C Runner.D = 90) ∧
  (headStartBetween Runner.B Runner.D = 250) := by
  sorry

#check head_start_calculation

end head_start_calculation_l2959_295930


namespace apple_difference_l2959_295985

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_count : adam_apples = 9) 
  (jackie_count : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
  sorry

end apple_difference_l2959_295985


namespace perpendicular_to_third_not_implies_perpendicular_l2959_295913

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for two lines being perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

/-- Theorem stating that the perpendicularity of two lines to a third line
    does not imply their perpendicularity to each other -/
theorem perpendicular_to_third_not_implies_perpendicular :
  ∃ (l1 l2 l3 : Line3D),
    perpendicular l1 l3 ∧ perpendicular l2 l3 ∧ ¬perpendicular l1 l2 := by
  sorry

end perpendicular_to_third_not_implies_perpendicular_l2959_295913


namespace cone_sphere_ratio_l2959_295981

/-- A cone with three spheres inside it -/
structure ConeWithSpheres where
  R : ℝ  -- radius of the base of the cone
  r : ℝ  -- radius of each sphere
  slant_height : ℝ  -- slant height of the cone
  spheres_touch : Bool  -- spheres touch each other externally
  two_touch_base : Bool  -- two spheres touch the lateral surface and base
  third_in_plane : Bool  -- third sphere touches at a point in the same plane as centers

/-- The properties of the cone and spheres arrangement -/
def cone_sphere_properties (c : ConeWithSpheres) : Prop :=
  c.R > 0 ∧ c.r > 0 ∧
  c.slant_height = 2 * c.R ∧  -- base diameter equals slant height
  c.spheres_touch ∧
  c.two_touch_base ∧
  c.third_in_plane

/-- The theorem stating the ratio of cone base radius to sphere radius -/
theorem cone_sphere_ratio (c : ConeWithSpheres) 
  (h : cone_sphere_properties c) : c.R / c.r = 5 / 4 + Real.sqrt 3 := by
  sorry

end cone_sphere_ratio_l2959_295981


namespace hyperbola_asymptote_l2959_295941

/-- Given a hyperbola with equation x^2 - y^2/m^2 = 1 where m > 0,
    if one of its asymptotes is x + √3 * y = 0, then m = √3/3 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x^2 - y^2/m^2 = 1 ∧ x + Real.sqrt 3 * y = 0) : 
  m = Real.sqrt 3 / 3 := by
  sorry

end hyperbola_asymptote_l2959_295941


namespace triangle_side_lengths_l2959_295918

theorem triangle_side_lengths 
  (a b c : ℝ) 
  (h1 : a + b + c = 23)
  (h2 : 3 * a + b + c = 43)
  (h3 : a + b + 3 * c = 35)
  (h4 : 2 * (a + b + c) = 46) :
  a = 10 ∧ b = 7 ∧ c = 6 := by
sorry

end triangle_side_lengths_l2959_295918


namespace min_reciprocal_sum_min_reciprocal_sum_attained_l2959_295938

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1/x + 1/y + 1/z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (1/a + 1/b + 1/c) = 3 := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_attained_l2959_295938


namespace min_votes_to_win_l2959_295976

-- Define the voting structure
def total_voters : ℕ := 135
def num_districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Define winning conditions
def win_precinct (votes : ℕ) : Prop := votes > voters_per_precinct / 2
def win_district (precincts_won : ℕ) : Prop := precincts_won > precincts_per_district / 2
def win_final (districts_won : ℕ) : Prop := districts_won > num_districts / 2

-- Theorem statement
theorem min_votes_to_win (min_votes : ℕ) : 
  (∃ (districts_won precincts_won votes_per_precinct : ℕ),
    win_final districts_won ∧
    win_district precincts_won ∧
    win_precinct votes_per_precinct ∧
    min_votes = districts_won * precincts_won * votes_per_precinct) →
  min_votes = 30 := by sorry

end min_votes_to_win_l2959_295976


namespace outfit_combinations_l2959_295993

theorem outfit_combinations (s p h : ℕ) (hs : s = 5) (hp : p = 6) (hh : h = 3) :
  s * p * h = 90 := by sorry

end outfit_combinations_l2959_295993


namespace ghee_mixture_problem_l2959_295939

theorem ghee_mixture_problem (Q : ℝ) : 
  (0.6 * Q = Q - 0.4 * Q) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * Q = 0.2 * (Q + 10)) →  -- After adding 10 kg, vanaspati is 20%
  Q = 10 := by
sorry

end ghee_mixture_problem_l2959_295939


namespace percentage_of_hindu_boys_l2959_295950

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) 
  (h1 : total = 850)
  (h2 : muslim_percent = 46/100)
  (h3 : sikh_percent = 10/100)
  (h4 : other = 136) :
  (total - (muslim_percent * total).num - (sikh_percent * total).num - other) / total = 28/100 := by
  sorry

end percentage_of_hindu_boys_l2959_295950


namespace art_exhibition_tickets_l2959_295973

/-- Calculates the total number of tickets sold given the conditions -/
def totalTicketsSold (advancedPrice : ℕ) (doorPrice : ℕ) (totalCollected : ℕ) (advancedSold : ℕ) : ℕ :=
  let doorSold := (totalCollected - advancedPrice * advancedSold) / doorPrice
  advancedSold + doorSold

/-- Theorem stating that under the given conditions, 165 tickets were sold in total -/
theorem art_exhibition_tickets :
  totalTicketsSold 8 14 1720 100 = 165 := by
  sorry

#eval totalTicketsSold 8 14 1720 100

end art_exhibition_tickets_l2959_295973


namespace min_value_of_3a_plus_2_l2959_295995

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (m : ℝ), m = -5/2 ∧ ∀ x, 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 2 ≥ m :=
by sorry

end min_value_of_3a_plus_2_l2959_295995


namespace larger_ball_radius_l2959_295958

/-- The radius of a larger steel ball formed from the same amount of material as 12 smaller balls -/
theorem larger_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) 
  (h1 : small_radius = 2)
  (h2 : num_small_balls = 12) : 
  ∃ (large_radius : ℝ), large_radius^3 = num_small_balls * small_radius^3 :=
by sorry

end larger_ball_radius_l2959_295958


namespace factory_profit_l2959_295929

noncomputable section

-- Define the daily cost function
def C (x : ℝ) : ℝ := 3 + x

-- Define the daily sales revenue function
def S (x k : ℝ) : ℝ := 
  if 0 < x ∧ x < 6 then 3*x + k/(x-8) + 5
  else if x ≥ 6 then 14
  else 0  -- undefined for x ≤ 0

-- Define the daily profit function
def L (x k : ℝ) : ℝ := S x k - C x

-- State the theorem
theorem factory_profit (k : ℝ) :
  (L 2 k = 3) →  -- Condition: when x = 2, L = 3
  (k = 18 ∧ 
   ∀ x, 0 < x → L x k ≤ 6 ∧
   L 5 k = 6) := by
  sorry

end

end factory_profit_l2959_295929


namespace subtraction_of_fractions_l2959_295984

theorem subtraction_of_fractions : (1 : ℚ) / 2 - (1 : ℚ) / 8 = (3 : ℚ) / 8 := by
  sorry

end subtraction_of_fractions_l2959_295984


namespace arithmetic_mean_problem_l2959_295904

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 5 + 17 + 3*x + 11 + 3*x + 6) / 5 = 19 → x = 8 := by
  sorry

end arithmetic_mean_problem_l2959_295904


namespace carmen_fudge_delights_sales_l2959_295922

/-- Represents the number of boxes sold for each cookie type -/
structure CookieSales where
  samoas : Nat
  thin_mints : Nat
  fudge_delights : Nat
  sugar_cookies : Nat

/-- Represents the price of each cookie type -/
structure CookiePrices where
  samoas : Rat
  thin_mints : Rat
  fudge_delights : Rat
  sugar_cookies : Rat

/-- Calculates the total revenue from cookie sales -/
def total_revenue (sales : CookieSales) (prices : CookiePrices) : Rat :=
  sales.samoas * prices.samoas +
  sales.thin_mints * prices.thin_mints +
  sales.fudge_delights * prices.fudge_delights +
  sales.sugar_cookies * prices.sugar_cookies

/-- The main theorem stating that Carmen sold 1 box of fudge delights -/
theorem carmen_fudge_delights_sales
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thin_mints = 2)
  (h3 : sales.sugar_cookies = 9)
  (h4 : prices.samoas = 4)
  (h5 : prices.thin_mints = 7/2)
  (h6 : prices.fudge_delights = 5)
  (h7 : prices.sugar_cookies = 2)
  (h8 : total_revenue sales prices = 42) :
  sales.fudge_delights = 1 := by
  sorry


end carmen_fudge_delights_sales_l2959_295922


namespace lighthouse_signals_lighthouse_signals_minimum_l2959_295965

theorem lighthouse_signals (x : ℕ) : 
  (x % 15 = 2 ∧ x % 28 = 8) → x ≥ 92 :=
by sorry

theorem lighthouse_signals_minimum : 
  ∃ (x : ℕ), x % 15 = 2 ∧ x % 28 = 8 ∧ x = 92 :=
by sorry

end lighthouse_signals_lighthouse_signals_minimum_l2959_295965


namespace geometric_progression_formula_l2959_295961

/-- A geometric progression with positive terms, where a₁ = 1 and a₂ + a₃ = 6 -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) ∧
  a 1 = 1 ∧
  a 2 + a 3 = 6

/-- The general term of the geometric progression is 2^(n-1) -/
theorem geometric_progression_formula (a : ℕ → ℝ) (h : GeometricProgression a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end geometric_progression_formula_l2959_295961


namespace unique_four_digit_square_with_repeated_digits_l2959_295942

/-- A four-digit number with repeated first two digits and last two digits -/
def FourDigitRepeated (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b

/-- The property of being a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ FourDigitRepeated n ∧ IsPerfectSquare n ∧ n = 7744 := by
  sorry

end unique_four_digit_square_with_repeated_digits_l2959_295942


namespace petya_winning_strategy_exists_l2959_295915

/-- Represents a player in the coin game -/
inductive Player
| Vasya
| Petya

/-- Represents the state of the game -/
structure GameState where
  chests : Nat
  coins : Nat
  currentPlayer : Player

/-- Defines a strategy for Petya -/
def PetyaStrategy := GameState → Nat

/-- Checks if a game state is valid -/
def isValidGameState (state : GameState) : Prop :=
  state.chests > 0 ∧ state.coins ≥ state.chests

/-- Represents the initial game state -/
def initialState : GameState :=
  { chests := 1011, coins := 2022, currentPlayer := Player.Vasya }

/-- Theorem stating Petya's winning strategy exists -/
theorem petya_winning_strategy_exists :
  ∃ (strategy : PetyaStrategy),
    ∀ (game : GameState),
      isValidGameState game →
      game.coins = 2 →
      ∃ (chest : Nat),
        chest < game.chests ∧
        strategy game = chest :=
  sorry

end petya_winning_strategy_exists_l2959_295915


namespace trigonometric_inequality_l2959_295966

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * (π/180)) - (Real.sqrt 3 / 2) * Real.sin (6 * (π/180))
  let b := (2 * Real.tan (13 * (π/180))) / (1 + Real.tan (13 * (π/180)) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * (π/180))) / 2)
  a < c ∧ c < b := by
sorry

end trigonometric_inequality_l2959_295966


namespace coordinates_determine_location_kunming_location_determined_l2959_295954

-- Define a structure for geographical coordinates
structure GeoCoordinates where
  longitude : Real
  latitude : Real

-- Define a function to check if coordinates are valid
def isValidCoordinates (coords : GeoCoordinates) : Prop :=
  -180 ≤ coords.longitude ∧ coords.longitude ≤ 180 ∧
  -90 ≤ coords.latitude ∧ coords.latitude ≤ 90

-- Define a function to determine if coordinates specify a unique location
def specifiesUniqueLocation (coords : GeoCoordinates) : Prop :=
  isValidCoordinates coords

-- Theorem stating that valid coordinates determine a specific location
theorem coordinates_determine_location (coords : GeoCoordinates) :
  isValidCoordinates coords → specifiesUniqueLocation coords :=
by
  sorry

-- Example using the coordinates from the problem
def kunming_coords : GeoCoordinates :=
  { longitude := 102, latitude := 24 }

-- Theorem stating that the Kunming coordinates determine a specific location
theorem kunming_location_determined :
  specifiesUniqueLocation kunming_coords :=
by
  sorry

end coordinates_determine_location_kunming_location_determined_l2959_295954


namespace cubic_equation_ratio_l2959_295944

theorem cubic_equation_ratio (p q r s : ℝ) : 
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) →
  r / s = 11 / 6 := by
  sorry

end cubic_equation_ratio_l2959_295944


namespace initial_number_proof_l2959_295912

theorem initial_number_proof (x : ℕ) : 7899665 - (3 * 2 * x) = 7899593 ↔ x = 12 := by
  sorry

end initial_number_proof_l2959_295912


namespace subtraction_result_l2959_295927

theorem subtraction_result (x : ℝ) (h : 96 / x = 6) : 34 - x = 18 := by
  sorry

end subtraction_result_l2959_295927


namespace money_left_after_game_l2959_295935

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hot_dog_cost : ℕ := 3

theorem money_left_after_game : 
  initial_amount - (ticket_cost + hot_dog_cost) = 9 := by sorry

end money_left_after_game_l2959_295935


namespace smallest_prime_12_less_than_square_l2959_295932

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, n > 0 ∧ Nat.Prime n ∧ (∃ m : ℕ, n = m^2 - 12) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < n → ¬(Nat.Prime k ∧ ∃ m : ℕ, k = m^2 - 12)) ∧
  n = 13 :=
by sorry

end smallest_prime_12_less_than_square_l2959_295932


namespace wheel_radius_increase_l2959_295921

-- Define constants
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem wheel_radius_increase 
  (D d₁ d₂ r : ℝ) 
  (h₁ : D > 0)
  (h₂ : d₁ > 0)
  (h₃ : d₂ > 0)
  (h₄ : r > 0)
  (h₅ : d₁ > d₂)
  (h₆ : D = d₁) :
  ∃ Δr : ℝ, Δr = (D * (30 * π / inches_per_mile) * inches_per_mile) / (2 * π * d₂) - r :=
by
  sorry

#check wheel_radius_increase

end wheel_radius_increase_l2959_295921


namespace symbiotic_pair_negation_l2959_295949

theorem symbiotic_pair_negation (m n : ℚ) : 
  (m - n = m * n + 1) → (-n - (-m) = (-n) * (-m) + 1) := by
  sorry

end symbiotic_pair_negation_l2959_295949


namespace factorial_120_121_is_perfect_square_l2959_295996

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Theorem: 120! · 121! is a perfect square -/
theorem factorial_120_121_is_perfect_square :
  is_perfect_square (factorial 120 * factorial 121) := by
  sorry

end factorial_120_121_is_perfect_square_l2959_295996


namespace odd_square_difference_plus_one_is_perfect_square_l2959_295956

theorem odd_square_difference_plus_one_is_perfect_square 
  (m n : ℤ) 
  (h_m_odd : Odd m) 
  (h_n_odd : Odd n) 
  (h_divides : (m^2 - n^2 + 1) ∣ (n^2 - 1)) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 :=
sorry

end odd_square_difference_plus_one_is_perfect_square_l2959_295956


namespace num_chords_is_45_num_triangles_is_120_l2959_295955

/- Define the combination function -/
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Define the number of points on the circle -/
def num_points : ℕ := 10

/- Theorem for the number of chords -/
theorem num_chords_is_45 : combination num_points 2 = 45 := by sorry

/- Theorem for the number of triangles -/
theorem num_triangles_is_120 : combination num_points 3 = 120 := by sorry

end num_chords_is_45_num_triangles_is_120_l2959_295955


namespace ones_digit_sum_powers_l2959_295948

theorem ones_digit_sum_powers (n : Nat) : n = 2023 → 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 10 = 5 := by
  sorry

end ones_digit_sum_powers_l2959_295948


namespace shaded_area_of_carpet_l2959_295905

/-- Theorem: Total shaded area of a square carpet with specific shaded squares -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  S = 12 / 4 →              -- S is 1/4 of the carpet side length
  T = S / 4 →               -- T is 1/4 of S
  S^2 + 4 * T^2 = 11.25 :=  -- Total shaded area
by sorry

end shaded_area_of_carpet_l2959_295905


namespace simplify_polynomial_l2959_295923

theorem simplify_polynomial (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end simplify_polynomial_l2959_295923


namespace even_heads_probability_l2959_295917

def coin_flips : ℕ := 8

theorem even_heads_probability : 
  (Finset.filter (fun n => Even n) (Finset.range (coin_flips + 1))).card / 2^coin_flips = 1 / 2 :=
by sorry

end even_heads_probability_l2959_295917


namespace point_difference_l2959_295997

def wildcats_rate : ℝ := 2.5
def panthers_rate : ℝ := 1.3
def half_duration : ℝ := 24

theorem point_difference : 
  wildcats_rate * half_duration - panthers_rate * half_duration = 28.8 := by
sorry

end point_difference_l2959_295997


namespace coefficient_of_a_half_power_l2959_295983

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (a - 1/√a)^5
def expansion (a : ℝ) : ℝ → ℝ := sorry

-- Theorem statement
theorem coefficient_of_a_half_power (a : ℝ) :
  ∃ (c : ℝ), c = -10 ∧ 
  (∀ (k : ℕ), k ≠ 3 → (binomial 5 k) * (-1)^k * a^(5 - k - k/2) ≠ c * a^(1/2)) ∧
  (binomial 5 3) * (-1)^3 * a^(5 - 3 - 3/2) = c * a^(1/2) :=
sorry

end coefficient_of_a_half_power_l2959_295983


namespace units_digit_of_fraction_l2959_295926

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 5000 % 10 = 6 := by
  sorry

end units_digit_of_fraction_l2959_295926


namespace factors_of_1320_l2959_295968

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1320 : number_of_factors 1320 = 32 := by
  sorry

end factors_of_1320_l2959_295968


namespace cube_preserves_order_l2959_295933

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_order_l2959_295933


namespace mike_office_visits_l2959_295991

/-- The number of pull-ups Mike does each time he enters his office -/
def pull_ups_per_visit : ℕ := 2

/-- The number of pull-ups Mike does in a week -/
def pull_ups_per_week : ℕ := 70

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Mike goes into his office each day -/
def office_visits_per_day : ℕ := 5

theorem mike_office_visits :
  office_visits_per_day * days_per_week * pull_ups_per_visit = pull_ups_per_week :=
by sorry

end mike_office_visits_l2959_295991
