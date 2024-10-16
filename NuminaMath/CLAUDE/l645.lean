import Mathlib

namespace NUMINAMATH_CALUDE_money_distribution_l645_64599

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 310) :
  C = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l645_64599


namespace NUMINAMATH_CALUDE_texas_tech_profit_calculation_l645_64511

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The money made from selling t-shirts during the Texas Tech game -/
def texas_tech_profit : ℕ := (total_shirts - arkansas_shirts) * profit_per_shirt

theorem texas_tech_profit_calculation : texas_tech_profit = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_profit_calculation_l645_64511


namespace NUMINAMATH_CALUDE_bottles_bought_l645_64518

theorem bottles_bought (initial : ℕ) (drunk : ℕ) (final : ℕ) : 
  initial = 14 → drunk = 8 → final = 51 → final - (initial - drunk) = 45 := by
  sorry

end NUMINAMATH_CALUDE_bottles_bought_l645_64518


namespace NUMINAMATH_CALUDE_similar_triangle_shorter_sides_sum_l645_64509

theorem similar_triangle_shorter_sides_sum (a b c : ℝ) (k : ℝ) :
  a = 8 ∧ b = 10 ∧ c = 12 →
  k * (a + b + c) = 180 →
  k * a + k * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_shorter_sides_sum_l645_64509


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l645_64541

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l645_64541


namespace NUMINAMATH_CALUDE_exam_score_proof_l645_64587

/-- Represents the average score of students who took the exam on the assigned day -/
def average_score_assigned_day : ℝ := 55

theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℝ) 
  (makeup_percentage : ℝ) (makeup_average : ℝ) (class_average : ℝ) : 
  total_students = 100 →
  assigned_day_percentage = 70 →
  makeup_percentage = 30 →
  makeup_average = 95 →
  class_average = 67 →
  (assigned_day_percentage * average_score_assigned_day + 
   makeup_percentage * makeup_average) / 100 = class_average :=
by
  sorry

#check exam_score_proof

end NUMINAMATH_CALUDE_exam_score_proof_l645_64587


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_103_l645_64540

theorem largest_n_divisible_by_103 : 
  ∀ n : ℕ, n < 103 ∧ 103 ∣ (n^3 - 1) → n ≤ 52 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_103_l645_64540


namespace NUMINAMATH_CALUDE_book_reading_problem_l645_64529

theorem book_reading_problem (n t k : ℕ) 
  (h1 : (k + 1) * n + k * (k + 1) / 2 = 374)
  (h2 : (k + 1) * t + k * (k + 1) / 2 = 319)
  (h3 : n > 0)
  (h4 : t > 0)
  (h5 : k > 0) :
  n + t = 53 := by
sorry

end NUMINAMATH_CALUDE_book_reading_problem_l645_64529


namespace NUMINAMATH_CALUDE_correct_categorization_l645_64580

def given_numbers : List ℚ := [-13.5, 5, 0, -10, 3.14, 27, -4/5, -15/100, 21/3]

def is_negative (x : ℚ) : Prop := x < 0
def is_non_negative (x : ℚ) : Prop := x ≥ 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ ¬(is_integer x)

def negative_numbers : List ℚ := [-13.5, -10, -4/5, -15/100]
def non_negative_numbers : List ℚ := [5, 0, 3.14, 27, 21/3]
def integers : List ℚ := [5, 0, -10, 27]
def negative_fractions : List ℚ := [-13.5, -4/5, -15/100]

theorem correct_categorization :
  (∀ x ∈ negative_numbers, is_negative x) ∧
  (∀ x ∈ non_negative_numbers, is_non_negative x) ∧
  (∀ x ∈ integers, is_integer x) ∧
  (∀ x ∈ negative_fractions, is_negative_fraction x) ∧
  (∀ x ∈ given_numbers, 
    (x ∈ negative_numbers ∨ x ∈ non_negative_numbers) ∧
    (x ∈ integers ∨ x ∈ negative_fractions ∨ (is_non_negative x ∧ ¬(is_integer x)))) := by
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l645_64580


namespace NUMINAMATH_CALUDE_system_solution_l645_64570

theorem system_solution : 
  {(x, y) : ℝ × ℝ | x^2 + y^2 + x + y = 50 ∧ x * y = 20} = 
  {(5, 4), (4, 5), (-5 + Real.sqrt 5, -5 - Real.sqrt 5), (-5 - Real.sqrt 5, -5 + Real.sqrt 5)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l645_64570


namespace NUMINAMATH_CALUDE_prime_equation_value_l645_64502

theorem prime_equation_value (p q : ℕ) : 
  Prime p → Prime q → (∃ x : ℤ, p * x + 5 * q = 97) → (40 * p + 101 * q + 4 = 2003) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_value_l645_64502


namespace NUMINAMATH_CALUDE_triangle_equilateral_l645_64575

/-- 
A triangle with sides a, b, and c is equilateral if:
1) a, b, and c form an arithmetic sequence
2) √a, √b, and √c form an arithmetic sequence
-/
theorem triangle_equilateral (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 2 * Real.sqrt b = Real.sqrt a + Real.sqrt c) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l645_64575


namespace NUMINAMATH_CALUDE_street_running_distances_l645_64543

/-- Represents the distance run around a square block -/
def run_distance (block_side : ℝ) (street_width : ℝ) (position : ℕ) : ℝ :=
  match position with
  | 0 => 4 * (block_side - 2 * street_width) -- inner side
  | 1 => 4 * block_side -- block side
  | 2 => 4 * (block_side + 2 * street_width) -- outer side
  | _ => 0 -- invalid position

theorem street_running_distances 
  (block_side : ℝ) (street_width : ℝ) 
  (h1 : block_side = 500) 
  (h2 : street_width = 25) : 
  run_distance block_side street_width 2 - run_distance block_side street_width 1 = 200 ∧
  run_distance block_side street_width 1 - run_distance block_side street_width 0 = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_street_running_distances_l645_64543


namespace NUMINAMATH_CALUDE_expected_weekly_rain_l645_64590

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def probSun : ℝ := 0.3

/-- The probability of 5 inches of rain --/
def probRain5 : ℝ := 0.4

/-- The probability of 12 inches of rain --/
def probRain12 : ℝ := 0.3

/-- The amount of rain on a sunny day --/
def rainSun : ℝ := 0

/-- The amount of rain on a day with 5 inches --/
def rain5 : ℝ := 5

/-- The amount of rain on a day with 12 inches --/
def rain12 : ℝ := 12

/-- The expected value of rainfall for one day --/
def expectedDailyRain : ℝ := probSun * rainSun + probRain5 * rain5 + probRain12 * rain12

/-- Theorem: The expected value of total rainfall for the week is 39.2 inches --/
theorem expected_weekly_rain : days * expectedDailyRain = 39.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rain_l645_64590


namespace NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l645_64507

/-- Given a quadratic function f(x) = 3x^2 - x + 7, shifting it 5 units to the right
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that the sum of the coefficients a + b + c equals 59. -/
theorem shifted_quadratic_coefficient_sum :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - x + 7
  let g : ℝ → ℝ := λ x ↦ f (x - 5)
  ∃ a b c : ℝ, (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 59) :=
by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l645_64507


namespace NUMINAMATH_CALUDE_sum_p_q_form_l645_64515

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h1 : ∀ x, ∃ a b c, q x = a * x^2 + b * x + c  -- q(x) is quadratic
  h2 : p 1 = 4  -- p(1) = 4
  h3 : q 3 = 0  -- q(3) = 0
  h4 : ∃ k, ∀ x, q x = k * (x - 3)^2  -- q(x) has a double root at x = 3

/-- The main theorem about the sum of p(x) and q(x) -/
theorem sum_p_q_form (f : RationalFunction) :
  ∃ a c : ℝ, (∀ x, f.p x + f.q x = x^2 + (a - 6) * x + 13) ∧ a + c = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_p_q_form_l645_64515


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l645_64548

theorem base_conversion_theorem (n A B : ℕ) : 
  (0 < n) →
  (0 ≤ A) ∧ (A < 8) →
  (0 ≤ B) ∧ (B < 6) →
  (n = 8 * A + B) →
  (n = 6 * B + A) →
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l645_64548


namespace NUMINAMATH_CALUDE_min_value_expression_l645_64563

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x^2 + 1 / y^2 + 1 / (x * y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l645_64563


namespace NUMINAMATH_CALUDE_ms_cole_students_l645_64546

/-- Represents the number of students in each math level class taught by Ms. Cole -/
structure MathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- Calculates the total number of students Ms. Cole teaches -/
def total_students (classes : MathClasses) : ℕ :=
  classes.sixth_level + classes.fourth_level + classes.seventh_level

/-- Theorem stating the total number of students Ms. Cole teaches -/
theorem ms_cole_students : ∃ (classes : MathClasses), 
  classes.sixth_level = 40 ∧ 
  classes.fourth_level = 4 * classes.sixth_level ∧
  classes.seventh_level = 2 * classes.fourth_level ∧
  total_students classes = 520 := by
  sorry

end NUMINAMATH_CALUDE_ms_cole_students_l645_64546


namespace NUMINAMATH_CALUDE_max_slope_product_l645_64579

theorem max_slope_product (m₁ m₂ : ℝ) : 
  (m₁ = 5 * m₂) →                    -- One slope is 5 times the other
  (|((m₂ - m₁) / (1 + m₁ * m₂))| = 1) →  -- Lines intersect at 45° angle
  (∀ n₁ n₂ : ℝ, (n₁ = 5 * n₂) → (|((n₂ - n₁) / (1 + n₁ * n₂))| = 1) → m₁ * m₂ ≥ n₁ * n₂) →
  m₁ * m₂ = 1.8 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_l645_64579


namespace NUMINAMATH_CALUDE_equilateral_triangle_between_poles_l645_64526

theorem equilateral_triangle_between_poles (pole1 pole2 : ℝ) (h1 : pole1 = 11) (h2 : pole2 = 13) :
  let a := 8 * Real.sqrt 3
  (a ^ 2 = pole1 ^ 2 + 2 ^ 2) ∧
  (a ^ 2 = pole2 ^ 2 + 2 ^ 2) ∧
  (Real.sqrt (a ^ 2 - pole1 ^ 2) + Real.sqrt (a ^ 2 - pole2 ^ 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_between_poles_l645_64526


namespace NUMINAMATH_CALUDE_moles_NaHCO3_equals_moles_HCl_l645_64510

/-- Represents a chemical species in the reaction -/
inductive Species
| NaHCO3
| HCl
| NaCl
| H2O
| CO2

/-- Represents the balanced chemical equation -/
def balanced_equation (reactants products : Species → ℕ) : Prop :=
  reactants Species.NaHCO3 = 1 ∧
  reactants Species.HCl = 1 ∧
  products Species.NaCl = 1 ∧
  products Species.H2O = 1 ∧
  products Species.CO2 = 1

/-- The number of moles of HCl given -/
def moles_HCl : ℕ := 3

/-- The number of moles of products formed -/
def moles_products : Species → ℕ
| Species.NaCl => 3
| Species.H2O => 3
| Species.CO2 => 3
| _ => 0

/-- Theorem stating that the number of moles of NaHCO3 required equals the number of moles of HCl -/
theorem moles_NaHCO3_equals_moles_HCl 
  (eq : balanced_equation (λ _ => 1) (λ _ => 1))
  (prod : ∀ s, moles_products s = moles_HCl ∨ moles_products s = 0) :
  moles_HCl = moles_HCl := by sorry

end NUMINAMATH_CALUDE_moles_NaHCO3_equals_moles_HCl_l645_64510


namespace NUMINAMATH_CALUDE_xyz_sum_product_range_l645_64537

theorem xyz_sum_product_range :
  ∀ x y z : ℝ,
  0 < x ∧ x < 1 →
  0 < y ∧ y < 1 →
  0 < z ∧ z < 1 →
  x + y + z = 2 →
  ∃ S : ℝ, S = x*y + y*z + z*x ∧ 1 < S ∧ S ≤ 4/3 ∧
  ∀ T : ℝ, (∃ a b c : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
                        a + b + c = 2 ∧ T = a*b + b*c + c*a) →
            1 < T ∧ T ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_product_range_l645_64537


namespace NUMINAMATH_CALUDE_valid_numbers_l645_64528

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_valid_number (abcd : ℕ) : Prop :=
  1000 ≤ abcd ∧ abcd < 10000 ∧
  abcd % 11 = 0 ∧
  (abcd / 100 % 10 + abcd / 10 % 10 = abcd / 1000) ∧
  is_perfect_square ((abcd / 100 % 10) * 10 + (abcd / 10 % 10))

theorem valid_numbers :
  {abcd : ℕ | is_valid_number abcd} = {9812, 1012, 4048, 9361, 9097} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l645_64528


namespace NUMINAMATH_CALUDE_mountain_speed_decrease_l645_64530

/-- The problem of finding the percentage decrease in vehicle speed when ascending a mountain. -/
theorem mountain_speed_decrease (initial_speed : ℝ) (ascend_distance descend_distance : ℝ) 
  (total_time : ℝ) (descend_increase : ℝ) :
  initial_speed = 30 →
  ascend_distance = 60 →
  descend_distance = 72 →
  total_time = 6 →
  descend_increase = 0.2 →
  ∃ (x : ℝ),
    x = 0.5 ∧
    (ascend_distance / (initial_speed * (1 - x))) + 
    (descend_distance / (initial_speed * (1 + descend_increase))) = total_time :=
by sorry

end NUMINAMATH_CALUDE_mountain_speed_decrease_l645_64530


namespace NUMINAMATH_CALUDE_log_inequality_l645_64560

theorem log_inequality : 
  let a := Real.log 2 / Real.log (1/3)
  let b := (1/3)^2
  let c := 2^(1/3)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_log_inequality_l645_64560


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l645_64589

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 30 < 0 → n ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l645_64589


namespace NUMINAMATH_CALUDE_min_additional_weeks_equals_additional_wins_needed_l645_64517

/-- Represents the number of dollars Bob has won so far -/
def initial_winnings : ℕ := 200

/-- Represents the number of additional wins needed to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- Represents the prize money for each win in dollars -/
def prize_money : ℕ := 100

/-- Proves that the minimum number of additional weeks Bob must win first place is equal to the number of additional wins needed -/
theorem min_additional_weeks_equals_additional_wins_needed :
  additional_wins_needed = additional_wins_needed := by sorry

end NUMINAMATH_CALUDE_min_additional_weeks_equals_additional_wins_needed_l645_64517


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l645_64534

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) : 
  (initial_players * initial_average + new_player1_weight + new_player2_weight) / (initial_players + 2) = 92 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l645_64534


namespace NUMINAMATH_CALUDE_base5_calculation_l645_64574

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Theorem stating that the result of (2434₅ + 132₅) ÷ 21₅ in base 5 is 122₅ --/
theorem base5_calculation : 
  base10ToBase5 ((base5ToBase10 [4, 3, 4, 2] + base5ToBase10 [2, 3, 1]) / base5ToBase10 [1, 2]) = [2, 2, 1] := by
  sorry


end NUMINAMATH_CALUDE_base5_calculation_l645_64574


namespace NUMINAMATH_CALUDE_sparrow_seeds_count_l645_64569

theorem sparrow_seeds_count : ∃ n : ℕ+, 
  (9 * n < 1001) ∧ 
  (10 * n > 1100) ∧ 
  (n = 111) := by
sorry

end NUMINAMATH_CALUDE_sparrow_seeds_count_l645_64569


namespace NUMINAMATH_CALUDE_l_shape_surface_area_l645_64582

/-- Represents the "L" shaped solid formed by unit cubes -/
structure LShape where
  bottom_row : ℕ
  vertical_stack : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the L-shaped solid -/
def surface_area (shape : LShape) : ℕ :=
  let bottom_exposure := shape.bottom_row + (shape.bottom_row - 1)
  let vertical_stack_exposure := 4 * shape.vertical_stack + 1
  let bottom_sides := 2 + shape.bottom_row
  bottom_exposure + vertical_stack_exposure + bottom_sides

/-- Theorem stating that the surface area of the specific L-shaped solid is 26 square units -/
theorem l_shape_surface_area :
  let shape : LShape := ⟨4, 3, 7⟩
  surface_area shape = 26 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_surface_area_l645_64582


namespace NUMINAMATH_CALUDE_sphere_surface_area_l645_64552

/-- Given a sphere with volume 72π cubic inches, its surface area is 36π * 2^(2/3) square inches. -/
theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l645_64552


namespace NUMINAMATH_CALUDE_square_area_ratio_l645_64565

theorem square_area_ratio (x : ℝ) (hx : x > 0) : (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l645_64565


namespace NUMINAMATH_CALUDE_divisor_condition_solutions_l645_64584

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The condition that the number of divisors equals the cube root of 4n -/
def divisor_condition (n : ℕ) : Prop :=
  num_divisors n = (4 * n : ℝ) ^ (1/3 : ℝ)

/-- The main theorem stating that the divisor condition is satisfied only for 2, 128, and 2000 -/
theorem divisor_condition_solutions :
  ∀ n : ℕ, n > 0 → (divisor_condition n ↔ n = 2 ∨ n = 128 ∨ n = 2000) := by
  sorry


end NUMINAMATH_CALUDE_divisor_condition_solutions_l645_64584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l645_64576

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l645_64576


namespace NUMINAMATH_CALUDE_total_cookies_eq_eaten_plus_left_l645_64591

/-- The number of cookies Mom made initially -/
def total_cookies : ℕ := sorry

/-- The number of cookies eaten by Julie and Matt -/
def cookies_eaten : ℕ := 9

/-- The number of cookies left after Julie and Matt ate -/
def cookies_left : ℕ := 23

/-- Theorem stating that the total number of cookies is the sum of eaten and left cookies -/
theorem total_cookies_eq_eaten_plus_left : 
  total_cookies = cookies_eaten + cookies_left := by sorry

end NUMINAMATH_CALUDE_total_cookies_eq_eaten_plus_left_l645_64591


namespace NUMINAMATH_CALUDE_gcd_98_63_l645_64519

theorem gcd_98_63 : Int.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l645_64519


namespace NUMINAMATH_CALUDE_distance_difference_around_block_l645_64567

/-- The difference in distance run by two people around a square block -/
def distanceDifference (blockSideLength : ℝ) (streetWidth : ℝ) : ℝ :=
  4 * (2 * streetWidth)

theorem distance_difference_around_block :
  let blockSideLength : ℝ := 400
  let streetWidth : ℝ := 20
  distanceDifference blockSideLength streetWidth = 160 := by sorry

end NUMINAMATH_CALUDE_distance_difference_around_block_l645_64567


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l645_64551

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first question
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for the second question
theorem A_subset_C_implies_a_geq_7 (a : ℝ) :
  A ⊆ C a → a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l645_64551


namespace NUMINAMATH_CALUDE_high_school_student_count_l645_64595

theorem high_school_student_count :
  let total_students : ℕ := 325
  let glasses_percentage : ℚ := 40 / 100
  let non_glasses_count : ℕ := 195
  (1 - glasses_percentage) * total_students = non_glasses_count :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_student_count_l645_64595


namespace NUMINAMATH_CALUDE_hundredth_power_mod_125_l645_64573

theorem hundredth_power_mod_125 (n : ℤ) : (n^100 : ℤ) % 125 = 0 ∨ (n^100 : ℤ) % 125 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_power_mod_125_l645_64573


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l645_64504

theorem max_abs_sum_on_circle : ∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l645_64504


namespace NUMINAMATH_CALUDE_katy_brownies_theorem_l645_64539

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end NUMINAMATH_CALUDE_katy_brownies_theorem_l645_64539


namespace NUMINAMATH_CALUDE_dividend_calculation_l645_64544

theorem dividend_calculation (k : ℕ) (quotient : ℕ) (h1 : k = 8) (h2 : quotient = 8) :
  k * quotient = 64 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l645_64544


namespace NUMINAMATH_CALUDE_rectangle_image_is_curved_region_l645_64556

-- Define the rectangle OAPB
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (0, 3)

-- Define the transformation
def u (x y : ℝ) : ℝ := x^2 - y^2
def v (x y : ℝ) : ℝ := x * y

-- Define the image of a point under the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (u p.1 p.2, v p.1 p.2)

-- Theorem statement
theorem rectangle_image_is_curved_region :
  ∃ (R : Set (ℝ × ℝ)), 
    (∀ p ∈ R, ∃ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, p = transform q) ∧
    (∀ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, transform q ∈ R) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform O ∧ (f 1, g 1) = transform A) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform A ∧ (f 1, g 1) = transform P) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform P ∧ (f 1, g 1) = transform B) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform B ∧ (f 1, g 1) = transform O) :=
sorry

end NUMINAMATH_CALUDE_rectangle_image_is_curved_region_l645_64556


namespace NUMINAMATH_CALUDE_polygonal_chain_existence_l645_64557

/-- A type representing a line in a plane -/
structure Line where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a polygonal chain -/
structure PolygonalChain (n : ℕ) where
  vertices : Fin (n + 1) → Point
  segments : Fin n → Line

/-- Predicate to check if a polygonal chain is non-self-intersecting -/
def is_non_self_intersecting (chain : PolygonalChain n) : Prop :=
  sorry

/-- Predicate to check if each segment of a polygonal chain lies on a unique line -/
def segments_on_unique_lines (chain : PolygonalChain n) (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no two lines are parallel -/
def no_parallel_lines (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no three lines intersect at the same point -/
def no_three_lines_intersect (lines : Fin n → Line) : Prop :=
  sorry

/-- Main theorem statement -/
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h1 : no_parallel_lines lines) 
  (h2 : no_three_lines_intersect lines) : 
  ∃ (chain : PolygonalChain n), 
    is_non_self_intersecting chain ∧ 
    segments_on_unique_lines chain lines :=
  sorry

end NUMINAMATH_CALUDE_polygonal_chain_existence_l645_64557


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l645_64553

theorem power_of_eight_sum_equals_power_of_two (y : ℕ) : 
  8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l645_64553


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l645_64545

theorem medicine_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = original_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) : 
  x = 0.2 := by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l645_64545


namespace NUMINAMATH_CALUDE_no_discount_possible_l645_64559

theorem no_discount_possible (purchase_price : ℝ) (marked_price_each : ℝ) 
  (h1 : purchase_price = 50)
  (h2 : marked_price_each = 22.5) :
  2 * marked_price_each < purchase_price := by
  sorry

#eval 2 * 22.5 -- This will output 45.0, confirming the contradiction

end NUMINAMATH_CALUDE_no_discount_possible_l645_64559


namespace NUMINAMATH_CALUDE_floor_plus_double_eq_sixteen_l645_64522

theorem floor_plus_double_eq_sixteen (r : ℝ) : (⌊r⌋ : ℝ) + 2 * r = 16 ↔ r = (5.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_double_eq_sixteen_l645_64522


namespace NUMINAMATH_CALUDE_function_characterization_l645_64596

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the equation must be of the form f(x) = cx -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l645_64596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l645_64554

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, a₆ < a₇ if and only if a₆ < a₈ -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 6 < a 7 ↔ a 6 < a 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l645_64554


namespace NUMINAMATH_CALUDE_egypt_trip_total_cost_l645_64542

def egypt_trip_cost (base_price upgrade_cost transportation_cost : ℕ) 
                    (individual_discount transportation_discount : ℚ) 
                    (num_people : ℕ) : ℚ :=
  let discounted_tour_price := base_price - individual_discount
  let total_per_person := discounted_tour_price + upgrade_cost
  let discounted_transportation := transportation_cost * (1 - transportation_discount)
  (total_per_person + discounted_transportation) * num_people

theorem egypt_trip_total_cost :
  egypt_trip_cost 147 65 80 14 (1/10) 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_egypt_trip_total_cost_l645_64542


namespace NUMINAMATH_CALUDE_page_lines_increase_l645_64592

theorem page_lines_increase (original_lines : ℕ) 
  (h1 : (110 : ℝ) = 0.8461538461538461 * original_lines) 
  (h2 : original_lines + 110 = 240) : 
  (original_lines + 110 : ℕ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l645_64592


namespace NUMINAMATH_CALUDE_smallest_sum_l645_64583

theorem smallest_sum (E F G H : ℕ+) : 
  (∃ d : ℤ, (E : ℤ) + d = F ∧ (F : ℤ) + d = G) →  -- arithmetic sequence
  (∃ r : ℚ, F * r = G ∧ G * r = H) →  -- geometric sequence
  G = (7 : ℚ) / 4 * F →  -- G/F = 7/4
  E + F + G + H ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_l645_64583


namespace NUMINAMATH_CALUDE_volume_of_tetrahedron_OCDE_l645_64581

/-- Square ABCD with side length 2 -/
def square_ABCD : Set (ℝ × ℝ) := sorry

/-- Point E is the midpoint of AB -/
def point_E : ℝ × ℝ := sorry

/-- Point O is formed when A and B coincide after folding -/
def point_O : ℝ × ℝ := sorry

/-- Triangle OCD formed after folding -/
def triangle_OCD : Set (ℝ × ℝ) := sorry

/-- Tetrahedron O-CDE formed after folding -/
def tetrahedron_OCDE : Set (ℝ × ℝ × ℝ) := sorry

/-- Volume of a tetrahedron -/
def tetrahedron_volume (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem volume_of_tetrahedron_OCDE : 
  tetrahedron_volume tetrahedron_OCDE = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_tetrahedron_OCDE_l645_64581


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l645_64597

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l645_64597


namespace NUMINAMATH_CALUDE_largest_number_with_equal_costs_l645_64524

/-- Calculates the sum of squares of decimal digits for a given number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of 1's in the binary representation of a given number -/
def count_ones_in_binary (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 2000 where 
    sum of squares of digits equals the number of 1's in binary representation -/
theorem largest_number_with_equal_costs : 
  ∀ n : ℕ, n < 2000 → n > 503 → 
    sum_of_squares_of_digits n ≠ count_ones_in_binary n := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_costs_l645_64524


namespace NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l645_64533

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by sorry

end NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l645_64533


namespace NUMINAMATH_CALUDE_range_of_a_l645_64562

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (m : ℝ),
    x₁^2 - m*x₁ - 1 = 0 ∧
    x₂^2 - m*x₂ - 1 = 0 ∧
    a^2 + 4*a - 3 ≤ |x₁ - x₂|

def q (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

-- Define the theorem
theorem range_of_a :
  ∀ (a : ℝ), (p a ∨ q a) ∧ ¬(p a ∧ q a) → a = 1 ∨ a < -5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l645_64562


namespace NUMINAMATH_CALUDE_garden_area_l645_64561

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : Real
  length : Real
  perimeter : Real
  area : Real
  length_condition : length = 3 * width + 10
  perimeter_condition : perimeter = 2 * (length + width)
  area_condition : area = length * width

/-- Theorem stating the area of a specific rectangular garden. -/
theorem garden_area (g : RectangularGarden) (h : g.perimeter = 400) :
  g.area = 7243.75 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l645_64561


namespace NUMINAMATH_CALUDE_imaginary_difference_condition_l645_64549

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end NUMINAMATH_CALUDE_imaginary_difference_condition_l645_64549


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l645_64578

def gauss_family_ages : List ℕ := [8, 8, 8, 8, 16, 17]

theorem gauss_family_mean_age : 
  (gauss_family_ages.sum : ℚ) / gauss_family_ages.length = 65 / 6 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l645_64578


namespace NUMINAMATH_CALUDE_complex_equation_solution_l645_64538

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l645_64538


namespace NUMINAMATH_CALUDE_negation_of_no_left_handed_in_chess_club_l645_64514

-- Define the universe of students
variable (Student : Type)

-- Define predicates for left-handedness and chess club membership
variable (isLeftHanded : Student → Prop)
variable (isInChessClub : Student → Prop)

-- State the theorem
theorem negation_of_no_left_handed_in_chess_club :
  (¬ ∀ (s : Student), isLeftHanded s → ¬ isInChessClub s) ↔
  (∃ (s : Student), isLeftHanded s ∧ isInChessClub s) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_no_left_handed_in_chess_club_l645_64514


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l645_64535

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i : Nat, i < arr.length - 1 →
    (10 * arr[i]! + arr[i+1]!) % 7 = 0

theorem no_valid_arrangement :
  ¬∃ arr : List Nat, arr.toFinset = {1, 2, 3, 4, 5, 6, 8, 9} ∧ is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l645_64535


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l645_64585

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (a b : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem exactly_two_correct_propositions 
  (l m n : Line) (α β γ : Plane) : 
  (∃! (correct : List Prop), 
    correct.length = 2 ∧ 
    correct ⊆ [
      (parallel m l ∧ perpendicular m α → perpendicular l α),
      (parallel m l ∧ parallel m α → parallel l α),
      (intersect α β l ∧ intersect β γ m ∧ intersect γ α n → 
        parallel l m ∧ parallel m n ∧ parallel l n),
      (intersect α β m ∧ intersect β γ l ∧ intersect α γ n ∧ 
        parallel n β → parallel m l)
    ] ∧
    (∀ p ∈ correct, p)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l645_64585


namespace NUMINAMATH_CALUDE_remainder_problem_l645_64512

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1012 % d = r ∧ 
  1548 % d = r ∧ 
  2860 % d = r → 
  d - r = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l645_64512


namespace NUMINAMATH_CALUDE_perfect_square_condition_l645_64586

theorem perfect_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + (a - 1)*x + 16 = (x + b)^2) → (a = 9 ∨ a = -7) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l645_64586


namespace NUMINAMATH_CALUDE_domain_intersection_l645_64508

-- Define the domain of y = e^x
def M : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Define the domain of y = ln x
def N : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_domain_intersection_l645_64508


namespace NUMINAMATH_CALUDE_system_solution_unique_l645_64525

theorem system_solution_unique :
  ∃! (x y : ℚ), (3 * (x - 1) = y + 6) ∧ (x / 2 + y / 3 = 2) ∧ (x = 10 / 3) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l645_64525


namespace NUMINAMATH_CALUDE_negation_of_exists_ln_positive_l645_64568

theorem negation_of_exists_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_ln_positive_l645_64568


namespace NUMINAMATH_CALUDE_find_x_given_exponential_equation_l645_64505

theorem find_x_given_exponential_equation : ∃ x : ℝ, (2 : ℝ)^(x - 4) = 4^2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_x_given_exponential_equation_l645_64505


namespace NUMINAMATH_CALUDE_joan_balloon_count_l645_64564

theorem joan_balloon_count (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → joan + melanie = total → joan = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l645_64564


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l645_64532

open Complex

theorem smallest_distance_between_circles (z w : ℂ) : 
  abs (z - (2 + 4*I)) = 2 →
  abs (w - (5 + 2*I)) = 4 →
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), abs (z' - (2 + 4*I)) = 2 → abs (w' - (5 + 2*I)) = 4 → abs (z' - w') ≥ min_dist) ∧
    min_dist = 6 - Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l645_64532


namespace NUMINAMATH_CALUDE_philip_orange_collection_l645_64566

/-- The number of oranges in Philip's collection -/
def num_oranges : ℕ := 178 * 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := 178

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- Theorem stating that the number of oranges in Philip's collection is 356 -/
theorem philip_orange_collection : num_oranges = 356 := by
  sorry

#eval num_oranges -- This will output 356

end NUMINAMATH_CALUDE_philip_orange_collection_l645_64566


namespace NUMINAMATH_CALUDE_circle_area_tripled_l645_64506

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l645_64506


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l645_64593

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l645_64593


namespace NUMINAMATH_CALUDE_intersection_sum_l645_64577

/-- Given two graphs y = -2|x-a| + b and y = 2|x-c| + d intersecting at (1, 6) and (5, 2), prove a + c = 6 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2*|x - a| + b = 2*|x - c| + d → x = 1 ∧ -2*|x - a| + b = 6 ∨ x = 5 ∧ -2*|x - a| + b = 2) →
  a + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l645_64577


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l645_64500

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15/100)
  (h3 : candidate_valid_votes = 333200) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l645_64500


namespace NUMINAMATH_CALUDE_fifth_term_is_nine_l645_64588

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem fifth_term_is_nine : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_nine_l645_64588


namespace NUMINAMATH_CALUDE_no_nonzero_triple_sum_zero_l645_64594

theorem no_nonzero_triple_sum_zero :
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a = b + c ∧ b = c + a ∧ c = a + b ∧
    a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_triple_sum_zero_l645_64594


namespace NUMINAMATH_CALUDE_triangle_inequality_l645_64572

-- Define a structure for a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  nondeg : min a b < c ∧ c < a + b -- Nondegenerate condition
  unit_perimeter : a + b + c = 1 -- Unit perimeter condition

-- Define the theorem
theorem triangle_inequality (t : Triangle) :
  |((t.a - t.b)/(t.c + t.a*t.b))| + |((t.b - t.c)/(t.a + t.b*t.c))| + |((t.c - t.a)/(t.b + t.a*t.c))| < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l645_64572


namespace NUMINAMATH_CALUDE_perfect_square_condition_l645_64555

theorem perfect_square_condition (n : ℤ) : 
  (∃ k : ℤ, 9 + 8 * n = k^2) ↔ (∃ m : ℤ, n = (m - 1) * (m + 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l645_64555


namespace NUMINAMATH_CALUDE_square_root_sum_difference_l645_64558

theorem square_root_sum_difference (x y : ℝ) : 
  x = Real.sqrt 7 + Real.sqrt 3 →
  y = Real.sqrt 7 - Real.sqrt 3 →
  x * y = 4 ∧ x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_difference_l645_64558


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l645_64536

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l645_64536


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l645_64550

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ + x₂ - x₁ * x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l645_64550


namespace NUMINAMATH_CALUDE_equation_solution_l645_64501

theorem equation_solution :
  ∀ x y : ℝ, x^2 - y^4 = Real.sqrt (18*x - x^2 - 81) ↔ (x = 9 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l645_64501


namespace NUMINAMATH_CALUDE_red_to_blue_ratio_l645_64531

/-- Represents the number of beads of each color in Michelle's necklace. -/
structure Necklace where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ
  silver : ℕ

/-- The conditions of Michelle's necklace. -/
def michelle_necklace : Necklace where
  total := 40
  blue := 5
  red := 10  -- This is derived, not given directly
  white := 15 -- This is derived, not given directly
  silver := 10

/-- The ratio of red beads to blue beads is 2:1. -/
theorem red_to_blue_ratio (n : Necklace) (h1 : n = michelle_necklace) 
    (h2 : n.white = n.blue + n.red) 
    (h3 : n.total = n.blue + n.red + n.white + n.silver) : 
  n.red / n.blue = 2 := by
  sorry

#check red_to_blue_ratio

end NUMINAMATH_CALUDE_red_to_blue_ratio_l645_64531


namespace NUMINAMATH_CALUDE_thread_length_problem_l645_64503

theorem thread_length_problem (current_length : ℝ) : 
  current_length + (3/4 * current_length) = 21 → current_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_problem_l645_64503


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l645_64598

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_f_at_2 : 
  deriv f 2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l645_64598


namespace NUMINAMATH_CALUDE_min_value_of_f_l645_64547

def f (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 7

theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l645_64547


namespace NUMINAMATH_CALUDE_car_travel_time_l645_64516

theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) 
  (h1 : speed_A = 50)
  (h2 : speed_B = 25)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : speed_A > 0)
  (h6 : speed_B > 0)
  (h7 : time_A > 0)
  (h8 : ratio > 0) :
  (speed_A * time_A) / (speed_B * ((speed_A * time_A) / (ratio * speed_B))) = 4 := by
  sorry

#check car_travel_time

end NUMINAMATH_CALUDE_car_travel_time_l645_64516


namespace NUMINAMATH_CALUDE_no_perfect_square_natural_l645_64520

theorem no_perfect_square_natural (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_natural_l645_64520


namespace NUMINAMATH_CALUDE_toothpaste_cost_is_three_l645_64521

/-- Represents the shopping scenario with given conditions -/
structure Shopping where
  budget : ℕ
  showerGelPrice : ℕ
  showerGelCount : ℕ
  laundryDetergentPrice : ℕ
  remaining : ℕ

/-- Calculates the cost of toothpaste given the shopping conditions -/
def toothpasteCost (s : Shopping) : ℕ :=
  s.budget - s.remaining - (s.showerGelPrice * s.showerGelCount) - s.laundryDetergentPrice

/-- Theorem stating that the toothpaste costs $3 under the given conditions -/
theorem toothpaste_cost_is_three :
  let s : Shopping := {
    budget := 60,
    showerGelPrice := 4,
    showerGelCount := 4,
    laundryDetergentPrice := 11,
    remaining := 30
  }
  toothpasteCost s = 3 := by sorry

end NUMINAMATH_CALUDE_toothpaste_cost_is_three_l645_64521


namespace NUMINAMATH_CALUDE_election_abstention_percentage_l645_64527

theorem election_abstention_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_b_percentage : ℚ) 
  (candidate_c_percentage : ℚ) 
  (candidate_d_percentage : ℚ) 
  (h1 : total_members = 1600) 
  (h2 : votes_cast = 900) 
  (h3 : candidate_a_percentage = 45/100) 
  (h4 : candidate_b_percentage = 35/100) 
  (h5 : candidate_c_percentage = 15/100) 
  (h6 : candidate_d_percentage = 5/100) 
  (h7 : candidate_a_percentage + candidate_b_percentage + candidate_c_percentage + candidate_d_percentage = 1) :
  (total_members - votes_cast : ℚ) / total_members * 100 = 43.75 := by
sorry

end NUMINAMATH_CALUDE_election_abstention_percentage_l645_64527


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l645_64523

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the "within" relation for a line being in a plane
variable (within : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary 
  (l m n : Line) (α : Plane)
  (m_in_α : within m α) (n_in_α : within n α) :
  (∀ l m n α, perp_plane l α → perp_line l m ∧ perp_line l n) ∧
  ¬(∀ l m n α, perp_line l m ∧ perp_line l n → perp_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l645_64523


namespace NUMINAMATH_CALUDE_cubic_root_sum_l645_64571

/-- Given p, q, and r are the roots of x^3 - 8x^2 + 6x - 3 = 0,
    prove that p/(qr - 1) + q/(pr - 1) + r/(pq - 1) = 21.75 -/
theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 → 
  q^3 - 8*q^2 + 6*q - 3 = 0 → 
  r^3 - 8*r^2 + 6*r - 3 = 0 → 
  p/(q*r - 1) + q/(p*r - 1) + r/(p*q - 1) = 21.75 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l645_64571


namespace NUMINAMATH_CALUDE_triangle_side_length_l645_64513

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  a * (1 - Real.cos B) = b * Real.cos A →
  c = 3 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 →
  -- Conclusion
  b = 4 * Real.sqrt 2 ∨ b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l645_64513
