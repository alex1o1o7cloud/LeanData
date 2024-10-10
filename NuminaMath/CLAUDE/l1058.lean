import Mathlib

namespace twentieth_number_in_sequence_l1058_105846

theorem twentieth_number_in_sequence : ∃ (n : ℕ), 
  (n % 8 = 5) ∧ 
  (n % 3 = 2) ∧ 
  (∃ (k : ℕ), k = 19 ∧ n = 5 + 24 * k) ∧
  n = 461 := by
sorry

end twentieth_number_in_sequence_l1058_105846


namespace john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l1058_105864

/-- Calculates how many meters ahead John finishes compared to Steve in a race --/
theorem john_finish_distance_ahead_of_steve 
  (initial_distance_behind : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_push_time : ℝ) : ℝ :=
  let john_distance := john_speed * final_push_time
  let steve_distance := steve_speed * final_push_time
  let steve_effective_distance := steve_distance + initial_distance_behind
  john_distance - steve_effective_distance

/-- Proves that John finishes 2 meters ahead of Steve given the race conditions --/
theorem john_finishes_two_meters_ahead : 
  john_finish_distance_ahead_of_steve 15 4.2 3.8 42.5 = 2 := by
  sorry

end john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l1058_105864


namespace quadratic_discriminant_l1058_105867

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  let a : ℝ := 5
  let b : ℝ := 2
  let c : ℝ := -8
  discriminant a b c = 164 := by
sorry

end quadratic_discriminant_l1058_105867


namespace or_and_not_implication_l1058_105871

theorem or_and_not_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end or_and_not_implication_l1058_105871


namespace interest_rate_increase_specific_interest_rate_increase_l1058_105801

-- Define the initial interest rate
def last_year_rate : ℝ := 9.90990990990991

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Theorem to prove
theorem interest_rate_increase (last_year_rate : ℝ) (increase_percent : ℝ) :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

-- Apply the theorem to our specific values
theorem specific_interest_rate_increase :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

end interest_rate_increase_specific_interest_rate_increase_l1058_105801


namespace remaining_average_l1058_105861

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 5 ∧ subset = 3 ∧ total_avg = 11 ∧ subset_avg = 4 →
  ((total_avg * total) - (subset_avg * subset)) / (total - subset) = 21.5 :=
by sorry

end remaining_average_l1058_105861


namespace complex_modulus_of_fraction_l1058_105894

theorem complex_modulus_of_fraction (z : ℂ) : 
  z = (4 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_of_fraction_l1058_105894


namespace candy_sold_tuesday_l1058_105859

/-- Theorem: Candy sold on Tuesday --/
theorem candy_sold_tuesday (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end candy_sold_tuesday_l1058_105859


namespace right_triangle_leg_length_l1058_105828

/-- Given a right triangle with one leg of length 5 and an altitude to the hypotenuse of length 4,
    the length of the other leg is 20/3. -/
theorem right_triangle_leg_length (a b c h : ℝ) : 
  a = 5 →                        -- One leg has length 5
  h = 4 →                        -- Altitude to hypotenuse has length 4
  a^2 + b^2 = c^2 →              -- Pythagorean theorem
  (1/2) * a * b = (1/2) * c * h → -- Area equality
  b = 20/3 := by sorry

end right_triangle_leg_length_l1058_105828


namespace journey_speed_calculation_l1058_105865

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (bicycle_speed : ℝ) (foot_distance : ℝ) 
  (h1 : total_distance = 61) 
  (h2 : total_time = 9)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  ∃ (foot_speed : ℝ), 
    foot_speed = 4 ∧ 
    foot_distance / foot_speed + (total_distance - foot_distance) / bicycle_speed = total_time :=
by sorry

end journey_speed_calculation_l1058_105865


namespace smallest_four_digit_different_digits_l1058_105845

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_different_digits :
  ∀ n : ℕ, is_four_digit n → has_different_digits n → 1023 ≤ n :=
by sorry

end smallest_four_digit_different_digits_l1058_105845


namespace abs_sum_inequality_iff_l1058_105832

theorem abs_sum_inequality_iff (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| > a) ↔ a < 3 := by
  sorry

end abs_sum_inequality_iff_l1058_105832


namespace remaining_cube_height_l1058_105885

/-- The height of the remaining portion of a cube after cutting off a corner -/
theorem remaining_cube_height (cube_side : Real) (cut_distance : Real) : 
  cube_side = 2 → 
  cut_distance = 1 → 
  (cube_side - (Real.sqrt 3) / 3) = (5 * Real.sqrt 3) / 3 := by
  sorry

end remaining_cube_height_l1058_105885


namespace min_value_of_expression_l1058_105875

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end min_value_of_expression_l1058_105875


namespace printer_X_time_proof_l1058_105811

/-- The time it takes for printer X to complete the job alone -/
def printerX_time : ℝ := 16

/-- The time it takes for printer Y to complete the job alone -/
def printerY_time : ℝ := 10

/-- The time it takes for printer Z to complete the job alone -/
def printerZ_time : ℝ := 20

/-- The ratio of printer X's time to the combined time of printers Y and Z -/
def time_ratio : ℝ := 2.4

theorem printer_X_time_proof :
  printerX_time = 16 ∧
  printerY_time = 10 ∧
  printerZ_time = 20 ∧
  time_ratio = 2.4 →
  printerX_time = time_ratio * (1 / (1 / printerY_time + 1 / printerZ_time)) :=
by
  sorry

end printer_X_time_proof_l1058_105811


namespace inverse_square_problem_l1058_105877

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y * y) ∧ k ≠ 0

-- State the theorem
theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) :
  inverse_square_relation x₁ y₁ →
  inverse_square_relation x₂ y₂ →
  x₁ = 1 →
  y₁ = 3 →
  y₂ = 6 →
  x₂ = 1/4 := by
  sorry

end inverse_square_problem_l1058_105877


namespace prime_factors_of_1998_l1058_105895

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
sorry

end prime_factors_of_1998_l1058_105895


namespace limits_zero_l1058_105818

open Real

theorem limits_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n / (10 : ℝ)^n| < ε) ∧ 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |log n / n| < ε) := by
  sorry

end limits_zero_l1058_105818


namespace jim_marathon_training_l1058_105850

theorem jim_marathon_training (days_phase1 days_phase2 days_phase3 : ℕ)
  (miles_per_day_phase1 miles_per_day_phase2 miles_per_day_phase3 : ℕ)
  (h1 : days_phase1 = 30)
  (h2 : days_phase2 = 30)
  (h3 : days_phase3 = 30)
  (h4 : miles_per_day_phase1 = 5)
  (h5 : miles_per_day_phase2 = 10)
  (h6 : miles_per_day_phase3 = 20) :
  days_phase1 * miles_per_day_phase1 +
  days_phase2 * miles_per_day_phase2 +
  days_phase3 * miles_per_day_phase3 = 1050 := by
  sorry

end jim_marathon_training_l1058_105850


namespace quartic_integer_roots_l1058_105869

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  d_nonzero : d ≠ 0

/-- The number of integer roots of a quartic polynomial, counting multiplicities -/
def num_integer_roots (p : QuarticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem quartic_integer_roots (p : QuarticPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end quartic_integer_roots_l1058_105869


namespace quadrilateral_side_difference_l1058_105872

theorem quadrilateral_side_difference (a b c d : ℝ) (h1 : a + b + c + d = 120) 
  (h2 : a + c = 50) (h3 : a^2 + c^2 = 40^2) : |b - d| = 2 * Real.sqrt 775 := by
  sorry

end quadrilateral_side_difference_l1058_105872


namespace absolute_value_nonnegative_l1058_105844

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end absolute_value_nonnegative_l1058_105844


namespace inequality_reversal_l1058_105806

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(-x > -y) := by
  sorry

end inequality_reversal_l1058_105806


namespace shortest_translation_distance_line_to_circle_l1058_105842

/-- The shortest distance to translate a line to become tangent to a circle -/
theorem shortest_translation_distance_line_to_circle 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (h_line : ∀ x y, line x y ↔ x - y + 1 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ d : ℝ, d = Real.sqrt 2 - 1 ∧ 
    (∀ d' : ℝ, d' ≥ 0 → 
      (∃ c : ℝ, ∀ x y, (x - y + c = 0 → circle x y) → d' ≥ d)) :=
sorry

end shortest_translation_distance_line_to_circle_l1058_105842


namespace product_of_sums_equals_difference_of_powers_l1058_105809

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * 
  (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 3^128 - 4^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l1058_105809


namespace arithmetic_sequence_property_l1058_105873

/-- An arithmetic sequence with a specific condition -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition : a 3 + a 6 + 3 * a 7 = 20

/-- The theorem stating that for any arithmetic sequence satisfying the given condition, 2a₇ - a₈ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 2 * seq.a 7 - seq.a 8 = 4 := by
  sorry

end arithmetic_sequence_property_l1058_105873


namespace intersection_with_ratio_l1058_105827

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given parallel lines and points, there exists a line through B intersecting the parallel lines at X and Y with the given ratio -/
theorem intersection_with_ratio 
  (a c : Line) 
  (A B C : Point) 
  (m n : ℝ) 
  (h_parallel : Line.parallel a c)
  (h_A_on_a : A.on_line a)
  (h_C_on_c : C.on_line c)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0) :
  ∃ (X Y : Point) (l : Line),
    X.on_line a ∧
    Y.on_line c ∧
    B.on_line l ∧
    X.on_line l ∧
    Y.on_line l ∧
    ∃ (k : ℝ), k > 0 ∧ 
      (X.x - A.x)^2 + (X.y - A.y)^2 = k * m^2 ∧
      (Y.x - C.x)^2 + (Y.y - C.y)^2 = k * n^2 :=
sorry

end intersection_with_ratio_l1058_105827


namespace ann_has_36_blocks_l1058_105888

/-- The number of blocks Ann has at the end, given her initial blocks, 
    blocks found, and blocks lost. -/
def anns_final_blocks (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost

/-- Theorem stating that Ann ends up with 36 blocks -/
theorem ann_has_36_blocks : anns_final_blocks 9 44 17 = 36 := by
  sorry

end ann_has_36_blocks_l1058_105888


namespace selling_price_calculation_l1058_105878

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 110 →
  gain_percent = 13.636363636363626 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 125 := by
sorry

end selling_price_calculation_l1058_105878


namespace intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l1058_105883

theorem intersection_points_theorem : 
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x = 2 ∨ x = 4) :=
by sorry

theorem pair_c_not_solution :
  ¬∃ (x : ℝ), (x - 2 = x - 4) ∧ (x^2 - 6*x + 8 = 0) :=
by sorry

theorem pair_a_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 8 = 0 ∧ 0 = 0) :=
by sorry

theorem pair_b_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x = 8) :=
by sorry

theorem pair_d_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 9 = 1) :=
by sorry

theorem pair_e_solution :
  ∃ (x : ℝ), (x^2 - 5 = 6*x - 8) ∧ (x^2 - 6*x + 8 ≠ 0) :=
by sorry

end intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l1058_105883


namespace largest_root_divisibility_l1058_105814

theorem largest_root_divisibility (a : ℝ) : 
  (a^3 - 3*a^2 + 1 = 0) →
  (∀ x : ℝ, x^3 - 3*x^2 + 1 = 0 → x ≤ a) →
  (17 ∣ ⌊a^1788⌋) ∧ (17 ∣ ⌊a^1988⌋) := by
sorry

end largest_root_divisibility_l1058_105814


namespace parentheses_removal_correctness_l1058_105860

theorem parentheses_removal_correctness :
  let A := (x y : ℝ) → 5*x - (x - 2*y) = 5*x - x + 2*y
  let B := (a b : ℝ) → 2*a^2 + (3*a - b) = 2*a^2 + 3*a - b
  let C := (x y : ℝ) → (x - 2*y) - (x^2 - y^2) = x - 2*y - x^2 + y^2
  let D := (x : ℝ) → 3*x^2 - 3*(x + 6) = 3*x^2 - 3*x - 6
  A ∧ B ∧ C ∧ ¬D := by sorry

end parentheses_removal_correctness_l1058_105860


namespace units_digit_of_seven_to_sixth_l1058_105807

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end units_digit_of_seven_to_sixth_l1058_105807


namespace three_digit_sum_theorem_l1058_105802

-- Define a function to represent a three-digit number
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Define a function to represent a two-digit number
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Define the condition for the problem
def satisfies_condition (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  three_digit_number a b c = 
    two_digit_number a b + two_digit_number a c +
    two_digit_number b a + two_digit_number b c +
    two_digit_number c a + two_digit_number c b

-- State the theorem
theorem three_digit_sum_theorem :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 6 ∧ c = 4) ∨ 
    (a = 3 ∧ b = 9 ∧ c = 6) :=
by sorry

end three_digit_sum_theorem_l1058_105802


namespace equation_solutions_l1058_105808

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x + 3)^2 - 4*(x - 1)^2
  (f (-1/3) = 0) ∧ (f 5 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → (x = -1/3 ∨ x = 5)) := by
sorry

end equation_solutions_l1058_105808


namespace horse_food_bags_l1058_105813

/-- Calculates the number of food bags needed for horses over a period of time. -/
theorem horse_food_bags 
  (num_horses : ℕ) 
  (feedings_per_day : ℕ) 
  (food_per_feeding : ℕ) 
  (days : ℕ) 
  (bag_weight_in_pounds : ℕ) : 
  num_horses = 25 → 
  feedings_per_day = 2 → 
  food_per_feeding = 20 → 
  days = 60 → 
  bag_weight_in_pounds = 1000 → 
  (num_horses * feedings_per_day * food_per_feeding * days) / bag_weight_in_pounds = 60 := by
  sorry

#check horse_food_bags

end horse_food_bags_l1058_105813


namespace min_packages_correct_l1058_105810

/-- The minimum number of packages Mary must deliver to cover the cost of her bicycle -/
def min_packages : ℕ :=
  let bicycle_cost : ℕ := 800
  let revenue_per_package : ℕ := 12
  let maintenance_cost_per_package : ℕ := 4
  let profit_per_package : ℕ := revenue_per_package - maintenance_cost_per_package
  (bicycle_cost + profit_per_package - 1) / profit_per_package

theorem min_packages_correct : min_packages = 100 := by
  sorry

end min_packages_correct_l1058_105810


namespace d_investment_is_250_l1058_105857

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  c_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- Calculates D's investment based on the given conditions -/
def calculate_d_investment (b : BusinessInvestment) : ℕ :=
  b.c_investment * b.d_profit_share / (b.total_profit - b.d_profit_share)

/-- Theorem stating that D's investment is 250 given the conditions -/
theorem d_investment_is_250 (b : BusinessInvestment) 
  (h1 : b.c_investment = 1000)
  (h2 : b.total_profit = 500)
  (h3 : b.d_profit_share = 100) : 
  calculate_d_investment b = 250 := by
  sorry

#eval calculate_d_investment { c_investment := 1000, total_profit := 500, d_profit_share := 100 }

end d_investment_is_250_l1058_105857


namespace correct_distribution_ways_l1058_105817

/-- The number of ways to distribute four distinct balls into two boxes -/
def distribution_ways : ℕ := 10

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The minimum number of balls required in box 1 -/
def min_box1 : ℕ := 1

/-- The minimum number of balls required in box 2 -/
def min_box2 : ℕ := 2

/-- A function that calculates the number of ways to distribute balls -/
def calculate_distribution_ways (n : ℕ) (k : ℕ) (min1 : ℕ) (min2 : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to distribute the balls is correct -/
theorem correct_distribution_ways :
  calculate_distribution_ways num_balls num_boxes min_box1 min_box2 = distribution_ways := by sorry

end correct_distribution_ways_l1058_105817


namespace brothers_age_difference_l1058_105819

theorem brothers_age_difference (a b : ℕ) : 
  a > 0 → b > 0 → a + b = 60 → 3 * b = 2 * a → a - b = 12 := by
  sorry

end brothers_age_difference_l1058_105819


namespace exists_x_fx_equals_four_l1058_105886

open Real

theorem exists_x_fx_equals_four :
  ∃ x₀ ∈ Set.Ioo 0 (3 * π), 3 + cos (2 * x₀) = 4 := by
  sorry

end exists_x_fx_equals_four_l1058_105886


namespace rectangular_parallelepiped_side_lengths_l1058_105851

theorem rectangular_parallelepiped_side_lengths 
  (x y z : ℝ) 
  (sum_eq : x + y + z = 17)
  (area_eq : 2*x*y + 2*y*z + 2*z*x = 180)
  (sq_sum_eq : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 := by
sorry

end rectangular_parallelepiped_side_lengths_l1058_105851


namespace unique_solution_condition_l1058_105834

/-- The diamond-shaped region defined by |x| + |y - 1| = 1 -/
def DiamondRegion (x y : ℝ) : Prop :=
  (abs x) + (abs (y - 1)) = 1

/-- The line defined by y = a * x + 2012 -/
def Line (a x y : ℝ) : Prop :=
  y = a * x + 2012

/-- The system of equations has a unique solution -/
def UniqueSystemSolution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), DiamondRegion x y ∧ Line a x y

/-- The theorem stating the condition for a unique solution -/
theorem unique_solution_condition (a : ℝ) :
  UniqueSystemSolution a ↔ (a = 2011 ∨ a = -2011) :=
sorry

end unique_solution_condition_l1058_105834


namespace event_ticket_revenue_l1058_105866

theorem event_ticket_revenue :
  ∀ (full_price half_price : ℕ),
  full_price + half_price = 180 →
  full_price * 20 + half_price * 10 = 2750 →
  full_price * 20 = 1900 :=
by
  sorry

end event_ticket_revenue_l1058_105866


namespace factor_problems_l1058_105879

theorem factor_problems : 
  (∃ n : ℤ, 25 = 5 * n) ∧ (∃ m : ℤ, 200 = 10 * m) := by
  sorry

end factor_problems_l1058_105879


namespace tangent_line_parallel_to_3x_minus_y_equals_0_l1058_105870

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_to_3x_minus_y_equals_0 :
  let P : ℝ × ℝ := (1, 0)
  f P.1 = P.2 ∧ f' P.1 = 3 := by sorry

end tangent_line_parallel_to_3x_minus_y_equals_0_l1058_105870


namespace triangle_acute_iff_sum_squares_gt_8R_squared_l1058_105884

theorem triangle_acute_iff_sum_squares_gt_8R_squared 
  (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_R_def : 4 * R * (R - a) * (R - b) * (R - c) = a * b * c) :
  (∀ (A B C : ℝ), A + B + C = π → 
    0 < A ∧ A < π/2 ∧ 
    0 < B ∧ B < π/2 ∧ 
    0 < C ∧ C < π/2) ↔ 
  a^2 + b^2 + c^2 > 8 * R^2 :=
sorry

end triangle_acute_iff_sum_squares_gt_8R_squared_l1058_105884


namespace quadratic_inequality_always_true_l1058_105840

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 :=
by sorry

end quadratic_inequality_always_true_l1058_105840


namespace fifteen_multiple_and_divisor_of_itself_l1058_105892

theorem fifteen_multiple_and_divisor_of_itself : 
  ∃ n : ℕ, n % 15 = 0 ∧ 15 % n = 0 ∧ n = 15 := by
  sorry

end fifteen_multiple_and_divisor_of_itself_l1058_105892


namespace system_solution_l1058_105890

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  (x - y = -1 ∧ x + y = 3) ∧ 
  (1/3 * x^2 - 1/3 * y^2) * (x^2 - 2*x*y + y^2) = -1 := by
  sorry

end system_solution_l1058_105890


namespace pond_draining_time_l1058_105821

/-- The time taken by the first pump to drain one-half of the pond -/
def first_pump_time : ℝ := 5

/-- The time taken by the second pump to drain the entire pond alone -/
def second_pump_time : ℝ := 1.1111111111111112

/-- The time taken by both pumps to drain the remaining half of the pond -/
def combined_time : ℝ := 0.5

theorem pond_draining_time : 
  (1 / (2 * first_pump_time) + 1 / second_pump_time) * combined_time = 1 / 2 :=
sorry

end pond_draining_time_l1058_105821


namespace max_isosceles_triangles_2017gon_l1058_105862

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a division of a polygon into triangular regions using diagonals -/
structure PolygonDivision (n : ℕ) where
  polygon : RegularPolygon n
  num_triangles : ℕ
  num_diagonals : ℕ
  diagonals_dont_intersect : Bool

/-- Represents the number of isosceles triangles in a polygon division -/
def num_isosceles_triangles (d : PolygonDivision n) : ℕ :=
  sorry

/-- Theorem: The maximum number of isosceles triangles in a specific polygon division -/
theorem max_isosceles_triangles_2017gon :
  ∀ (d : PolygonDivision 2017),
    d.num_triangles = 2015 →
    d.num_diagonals = 2014 →
    d.diagonals_dont_intersect = true →
    num_isosceles_triangles d ≤ 2010 :=
  sorry

end max_isosceles_triangles_2017gon_l1058_105862


namespace tank_width_is_twelve_l1058_105838

/-- Represents the dimensions and plastering cost of a tank. -/
structure Tank where
  length : ℝ
  depth : ℝ
  width : ℝ
  plasteringRate : ℝ
  totalCost : ℝ

/-- Calculates the total surface area of the tank. -/
def surfaceArea (t : Tank) : ℝ :=
  2 * (t.length * t.depth) + 2 * (t.width * t.depth) + (t.length * t.width)

/-- Theorem stating that for a tank with given dimensions and plastering cost,
    the width is 12 meters. -/
theorem tank_width_is_twelve (t : Tank)
  (h1 : t.length = 25)
  (h2 : t.depth = 6)
  (h3 : t.plasteringRate = 0.25)
  (h4 : t.totalCost = 186)
  (h5 : t.totalCost = t.plasteringRate * surfaceArea t) :
  t.width = 12 := by
  sorry

#check tank_width_is_twelve

end tank_width_is_twelve_l1058_105838


namespace magnitude_of_z_l1058_105822

theorem magnitude_of_z (z : ℂ) (h : (z + 1) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l1058_105822


namespace smallest_y_with_given_remainders_l1058_105880

theorem smallest_y_with_given_remainders : ∃ y : ℕ, 
  y > 0 ∧ 
  y % 3 = 2 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧ 
  ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by sorry

end smallest_y_with_given_remainders_l1058_105880


namespace proposition_1_proposition_2_proposition_3_l1058_105876

-- Proposition 1
def p : Prop := ∃ x : ℝ, Real.tan x = 2
def q : Prop := ∀ x : ℝ, x^2 - x + 1/2 > 0

theorem proposition_1 : ¬(p ∧ ¬q) := by sorry

-- Proposition 2
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x + b * y + 1 = 0

theorem proposition_2 (a b : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a * 1 + 3 * b = 0)) ≠ 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a / b = -3)) := by sorry

-- Proposition 3
def original_statement (a b : ℝ) : Prop := 
  a * b ≥ 2 → a^2 + b^2 > 4

def negation_statement (a b : ℝ) : Prop := 
  a * b < 2 → a^2 + b^2 ≤ 4

theorem proposition_3 : 
  (∀ a b : ℝ, ¬(original_statement a b)) ↔ (∀ a b : ℝ, negation_statement a b) := by sorry

end proposition_1_proposition_2_proposition_3_l1058_105876


namespace cycle_selling_price_l1058_105825

/-- Calculates the selling price of a cycle given its original price and loss percentage. -/
theorem cycle_selling_price (original_price loss_percentage : ℝ) :
  original_price = 2300 →
  loss_percentage = 30 →
  original_price * (1 - loss_percentage / 100) = 1610 := by
sorry

end cycle_selling_price_l1058_105825


namespace rectangular_solid_volume_l1058_105839

theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 20)
  (front_area : b * c = 15)
  (bottom_area : a * c = 12)
  (dimension_relation : a = 2 * b) : 
  a * b * c = 12 * Real.sqrt 10 := by
sorry

end rectangular_solid_volume_l1058_105839


namespace P_intersect_Q_l1058_105893

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem P_intersect_Q : P ∩ Q = {1, 2} := by
  sorry

end P_intersect_Q_l1058_105893


namespace art_collection_remaining_l1058_105887

/-- Calculates the remaining number of art pieces after a donation --/
def remaining_art_pieces (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem: Given 70 initial pieces and 46 donated pieces, 24 pieces remain --/
theorem art_collection_remaining :
  remaining_art_pieces 70 46 = 24 := by
  sorry

end art_collection_remaining_l1058_105887


namespace winner_equal_victories_defeats_iff_odd_l1058_105830

/-- A tournament of n nations where each nation plays against every other nation exactly once. -/
structure Tournament (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- The number of victories for a team in a tournament. -/
def victories (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- The number of defeats for a team in a tournament. -/
def defeats (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- A team is a winner if it has the maximum number of victories. -/
def is_winner (t : Tournament n) (team : Fin n) : Prop :=
  ∀ other : Fin n, victories t team ≥ victories t other

theorem winner_equal_victories_defeats_iff_odd (n : ℕ) :
  (∃ (t : Tournament n) (w : Fin n), is_winner t w ∧ victories t w = defeats t w) ↔ Odd n :=
sorry

end winner_equal_victories_defeats_iff_odd_l1058_105830


namespace solution_set_l1058_105847

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0

theorem solution_set : 
  {x : ℝ | equation x} = {-9, -3, 3} := by sorry

end solution_set_l1058_105847


namespace square_sum_equality_l1058_105831

theorem square_sum_equality : 108 * 108 + 92 * 92 = 20128 := by
  sorry

end square_sum_equality_l1058_105831


namespace probability_nine_heads_in_twelve_flips_l1058_105824

theorem probability_nine_heads_in_twelve_flips : 
  (Nat.choose 12 9 : ℚ) / (2^12 : ℚ) = 220 / 4096 :=
by sorry

end probability_nine_heads_in_twelve_flips_l1058_105824


namespace power_two_plus_one_div_by_three_l1058_105804

theorem power_two_plus_one_div_by_three (n : ℕ) :
  n > 0 → (3 ∣ 2^n + 1 ↔ n % 2 = 1) := by sorry

end power_two_plus_one_div_by_three_l1058_105804


namespace balance_equals_132_l1058_105820

/-- Calculates the account balance after two years given an initial deposit,
    annual interest rate, and additional annual deposit. -/
def account_balance_after_two_years (initial_deposit : ℝ) (interest_rate : ℝ) (annual_deposit : ℝ) : ℝ :=
  let balance_after_first_year := initial_deposit * (1 + interest_rate) + annual_deposit
  balance_after_first_year * (1 + interest_rate) + annual_deposit

/-- Theorem stating that given the specified conditions, the account balance
    after two years will be $132. -/
theorem balance_equals_132 :
  account_balance_after_two_years 100 0.1 10 = 132 := by
  sorry

#eval account_balance_after_two_years 100 0.1 10

end balance_equals_132_l1058_105820


namespace marble_probability_theorem_l1058_105816

/-- Represents a box containing marbles -/
structure Box where
  total : ℕ
  red : ℕ
  blue : ℕ
  hSum : red + blue = total

/-- The probability of drawing a red marble from a box -/
def redProb (b : Box) : ℚ :=
  b.red / b.total

/-- The probability of drawing a blue marble from a box -/
def blueProb (b : Box) : ℚ :=
  b.blue / b.total

/-- The main theorem -/
theorem marble_probability_theorem
  (box1 box2 : Box)
  (hTotal : box1.total + box2.total = 30)
  (hRedProb : redProb box1 * redProb box2 = 2/3) :
  blueProb box1 * blueProb box2 = 3/4 := by
  sorry

end marble_probability_theorem_l1058_105816


namespace three_tangents_condition_l1058_105863

/-- The curve function f(x) = x³ + 3x² + ax + a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a*x + a - 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*x + a

/-- The tangent line equation passing through (0, 2) -/
def tangent_line (a : ℝ) (x₀ : ℝ) (x : ℝ) : ℝ :=
  f_deriv a x₀ * (x - x₀) + f a x₀

/-- The condition for a point x₀ to be on a tangent line passing through (0, 2) -/
def tangent_condition (a : ℝ) (x₀ : ℝ) : Prop :=
  tangent_line a x₀ 0 = 2

/-- The main theorem stating the condition for exactly three tangent lines -/
theorem three_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    tangent_condition a x₁ ∧ tangent_condition a x₂ ∧ tangent_condition a x₃ ∧
    (∀ x : ℝ, tangent_condition a x → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  4 < a ∧ a < 5 :=
sorry

end three_tangents_condition_l1058_105863


namespace students_playing_both_sports_l1058_105803

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 250 →
  football = 160 →
  cricket = 90 →
  neither = 50 →
  (total - neither) = (football + cricket - (football + cricket - (total - neither))) :=
by sorry

end students_playing_both_sports_l1058_105803


namespace inverse_f_sum_squares_l1058_105805

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_f_sum_squares : 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49) → 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49 ∧ y₁^2 + y₂^2 = 58) := by
sorry

end inverse_f_sum_squares_l1058_105805


namespace intersection_A_B_union_A_B_intersection_complements_A_B_l1058_105849

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 5} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (Aᶜ : Set ℝ) ∩ (Bᶜ : Set ℝ) = {x : ℝ | x < 1 ∨ x ≥ 8} := by sorry

end intersection_A_B_union_A_B_intersection_complements_A_B_l1058_105849


namespace parabola_right_angle_l1058_105823

theorem parabola_right_angle (a : ℝ) : 
  let f (x : ℝ) := -(x + 3) * (2 * x + a)
  let x₁ := -3
  let x₂ := -a / 2
  let y_c := f 0
  let A := (x₁, 0)
  let B := (x₂, 0)
  let C := (0, y_c)
  f x₁ = 0 ∧ f x₂ = 0 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  a = -1/6 := by
sorry

end parabola_right_angle_l1058_105823


namespace juan_number_problem_l1058_105897

theorem juan_number_problem (n : ℚ) : 
  ((n + 3) * 3 - 5) / 3 = 10 → n = 26 / 3 := by
  sorry

end juan_number_problem_l1058_105897


namespace f_min_value_l1058_105899

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that the minimum value of f(x, y) is 3 -/
theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ 3 := by sorry

end f_min_value_l1058_105899


namespace log_product_equals_five_thirds_l1058_105843

theorem log_product_equals_five_thirds :
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 3 := by
  sorry

end log_product_equals_five_thirds_l1058_105843


namespace min_cost_pool_l1058_105868

/-- Represents the dimensions and cost parameters of a rectangular pool -/
structure PoolParams where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of the pool given its length and width -/
def totalCost (p : PoolParams) (length width : ℝ) : ℝ :=
  p.bottomCost * length * width + p.wallCost * (2 * length * p.depth + 2 * width * p.depth)

/-- Theorem stating the minimum cost and dimensions of the pool -/
theorem min_cost_pool (p : PoolParams) 
    (hv : p.volume = 16)
    (hd : p.depth = 4)
    (hb : p.bottomCost = 110)
    (hw : p.wallCost = 90) :
    ∃ (length width : ℝ),
      length * width * p.depth = p.volume ∧
      length = 2 ∧
      width = 2 ∧
      totalCost p length width = 1880 ∧
      ∀ (l w : ℝ), l * w * p.depth = p.volume → totalCost p l w ≥ totalCost p length width :=
  sorry

end min_cost_pool_l1058_105868


namespace floor_of_5_7_l1058_105856

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end floor_of_5_7_l1058_105856


namespace total_earnings_l1058_105836

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem total_earnings :
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end total_earnings_l1058_105836


namespace consecutive_even_product_divisibility_l1058_105855

theorem consecutive_even_product_divisibility (n : ℤ) (h : Even n) :
  ∃ k : ℤ, n * (n + 2) * (n + 4) * (n + 6) = 48 * k := by
  sorry

end consecutive_even_product_divisibility_l1058_105855


namespace number_problem_l1058_105829

theorem number_problem (x : ℝ) : (x - 6) / 8 = 6 → (x - 5) / 7 = 7 := by
  sorry

end number_problem_l1058_105829


namespace max_volume_container_l1058_105898

/-- Represents a rectangular container made from a sheet with cut corners. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the container. -/
def volume (c : Container) : ℝ := (c.length - 2 * c.height) * (c.width - 2 * c.height) * c.height

/-- The original sheet dimensions. -/
def sheet_length : ℝ := 8
def sheet_width : ℝ := 5

/-- Theorem stating the maximum volume and the height at which it occurs. -/
theorem max_volume_container :
  ∃ (h : ℝ),
    h > 0 ∧
    h < min sheet_length sheet_width / 2 ∧
    (∀ (c : Container),
      c.length = sheet_length ∧
      c.width = sheet_width ∧
      c.height > 0 ∧
      c.height < min sheet_length sheet_width / 2 →
      volume c ≤ volume { length := sheet_length, width := sheet_width, height := h }) ∧
    volume { length := sheet_length, width := sheet_width, height := h } = 18 :=
  sorry

end max_volume_container_l1058_105898


namespace p_necessary_not_sufficient_l1058_105835

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 4
def q (x y : ℝ) : Prop := x + y ≠ 6

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end p_necessary_not_sufficient_l1058_105835


namespace intersection_line_equation_l1058_105881

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y - 1 = 0 -/
theorem intersection_line_equation (A₁ B₁ A₂ B₂ : ℝ) :
  (A₁ * 2 + B₁ * 3 = 1) →
  (A₂ * 2 + B₂ * 3 = 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (A₁ - A₂) * 2 + (B₁ - B₂) * 3 = k * (2 * (A₁ - A₂) + 3 * (B₁ - B₂) - 1) :=
by sorry

end intersection_line_equation_l1058_105881


namespace third_member_reels_six_l1058_105853

/-- Represents a fishing competition with three team members -/
structure FishingCompetition where
  days : ℕ
  fish_per_day_1 : ℕ
  fish_per_day_2 : ℕ
  total_fish : ℕ

/-- Calculates the number of fish the third member reels per day -/
def third_member_fish_per_day (comp : FishingCompetition) : ℕ :=
  (comp.total_fish - (comp.fish_per_day_1 + comp.fish_per_day_2) * comp.days) / comp.days

/-- Theorem stating that in the given conditions, the third member reels 6 fish per day -/
theorem third_member_reels_six (comp : FishingCompetition) 
  (h1 : comp.days = 5)
  (h2 : comp.fish_per_day_1 = 4)
  (h3 : comp.fish_per_day_2 = 8)
  (h4 : comp.total_fish = 90) : 
  third_member_fish_per_day comp = 6 := by
  sorry

#eval third_member_fish_per_day ⟨5, 4, 8, 90⟩

end third_member_reels_six_l1058_105853


namespace solution_satisfies_equations_l1058_105896

theorem solution_satisfies_equations :
  let a : ℚ := 4/7
  let b : ℚ := 19/7
  let c : ℚ := 29/19
  let d : ℚ := -6/19
  (8*a^2 - 3*b^2 + 5*c^2 + 16*d^2 - 10*a*b + 42*c*d + 18*a + 22*b - 2*c - 54*d = 42) ∧
  (15*a^2 - 3*b^2 + 21*c^2 - 5*d^2 + 4*a*b + 32*c*d - 28*a + 14*b - 54*c - 52*d = -22) :=
by sorry

end solution_satisfies_equations_l1058_105896


namespace geometric_sequence_middle_term_range_l1058_105848

/-- Given three numbers forming a geometric sequence with sum m, prove the range of the middle term -/
theorem geometric_sequence_middle_term_range (a b c m : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c) →  -- a, b, c form a geometric sequence
  (a + b + c = m) →                   -- sum of terms is m
  (m > 0) →                           -- m is positive
  (b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m/3)) :=  -- range of b
by sorry

end geometric_sequence_middle_term_range_l1058_105848


namespace amy_school_year_work_hours_l1058_105891

/-- Calculates the number of hours Amy needs to work per week during the school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ)
  (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_rate := summer_earnings / (summer_weeks * summer_hours_per_week : ℚ)
  let total_school_year_hours := school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks

/-- Proves that Amy needs to work approximately 18 hours per week during the school year -/
theorem amy_school_year_work_hours :
  let result := school_year_hours_per_week 10 36 3000 30 4500
  18 < result ∧ result < 19 := by
  sorry

end amy_school_year_work_hours_l1058_105891


namespace triangle_side_length_l1058_105826

/-- Represents a triangle with side lengths x, y, z and angles X, Y, Z --/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  X : ℝ
  Y : ℝ
  Z : ℝ

/-- The theorem stating the properties of the specific triangle and its side length y --/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.Z = 4 * t.X) 
  (h2 : t.x = 36) 
  (h3 : t.z = 72) : 
  t.y = 72 := by
  sorry

end triangle_side_length_l1058_105826


namespace alicia_satisfaction_l1058_105841

/-- Represents the satisfaction equation for Alicia's activities --/
def satisfaction (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Represents the constraint that t should be positive and less than 4 --/
def valid_t (t : ℝ) : Prop := 0 < t ∧ t < 4

theorem alicia_satisfaction (t : ℝ) : 
  valid_t t →
  satisfaction (12 - t) t = satisfaction (2*t + 2) (4 - t) →
  t = 2 :=
by sorry

end alicia_satisfaction_l1058_105841


namespace parabola_focus_l1058_105852

-- Define the parabola
def parabola (x : ℝ) : ℝ := -4 * x^2 - 8 * x + 1

-- Define the focus of a parabola
def focus (a b c : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem parabola_focus :
  focus (-4) (-8) 1 = (-1, 79/16) := by sorry

end parabola_focus_l1058_105852


namespace regression_line_intercept_l1058_105812

-- Define the number of data points
def n : ℕ := 8

-- Define the slope of the regression line
def m : ℚ := 1/3

-- Define the sum of x values
def sum_x : ℚ := 3

-- Define the sum of y values
def sum_y : ℚ := 5

-- Define the mean of x values
def mean_x : ℚ := sum_x / n

-- Define the mean of y values
def mean_y : ℚ := sum_y / n

-- Theorem statement
theorem regression_line_intercept :
  ∃ (a : ℚ), mean_y = m * mean_x + a ∧ a = 1/2 := by sorry

end regression_line_intercept_l1058_105812


namespace roots_of_composite_quadratic_l1058_105882

/-- A quadratic function with real coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given point -/
def evaluate (f : QuadraticFunction) (x : ℂ) : ℂ :=
  f.a * x^2 + f.b * x + f.c

/-- Predicate stating that a complex number is purely imaginary -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Predicate stating that all roots of the equation f(x) = 0 are purely imaginary -/
def hasPurelyImaginaryRoots (f : QuadraticFunction) : Prop :=
  ∀ x : ℂ, evaluate f x = 0 → isPurelyImaginary x

/-- Theorem stating the nature of roots for f(f(x)) = 0 -/
theorem roots_of_composite_quadratic
  (f : QuadraticFunction)
  (h : hasPurelyImaginaryRoots f) :
  ∀ x : ℂ, evaluate f (evaluate f x) = 0 →
    (¬ x.im = 0) ∧ ¬ isPurelyImaginary x :=
sorry

end roots_of_composite_quadratic_l1058_105882


namespace jaime_score_l1058_105800

theorem jaime_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) (jaime_score : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n - 1) * avg_without + jaime_score = n * avg_with →
  jaime_score = 105 :=
by
  sorry

end jaime_score_l1058_105800


namespace x_eighth_is_zero_l1058_105815

theorem x_eighth_is_zero (x : ℝ) (h : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 0 := by
  sorry

end x_eighth_is_zero_l1058_105815


namespace rectangular_box_volume_l1058_105874

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 10) 
  (h3 : c * a = 6) : 
  a * b * c = 30 := by
  sorry

end rectangular_box_volume_l1058_105874


namespace count_ordered_pairs_eq_18_l1058_105889

/-- Given that 1372 = 2^2 * 7^2 * 11, this function returns the number of ordered pairs of positive integers (x, y) satisfying x * y = 1372. -/
def count_ordered_pairs : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 2), (7, 2), (11, 1)]
  (prime_factorization.map (λ (p, e) => e + 1)).prod

/-- The number of ordered pairs of positive integers (x, y) satisfying x * y = 1372 is 18. -/
theorem count_ordered_pairs_eq_18 : count_ordered_pairs = 18 := by
  sorry

end count_ordered_pairs_eq_18_l1058_105889


namespace equation_implies_conditions_l1058_105833

theorem equation_implies_conditions (x y z w : ℝ) 
  (h : (2*x + y) / (y + z) = (z + w) / (w + 2*x)) :
  x = z/2 ∨ 2*x + y + z + w = 0 :=
by sorry

end equation_implies_conditions_l1058_105833


namespace no_such_function_exists_l1058_105837

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, (f^[f x]) x = x + 1 := by
  sorry

#check no_such_function_exists

end no_such_function_exists_l1058_105837


namespace square_sum_xy_l1058_105854

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 35)
  (h2 : y * (x + y) = 77) : 
  (x + y)^2 = 112 := by
sorry

end square_sum_xy_l1058_105854


namespace sector_area_l1058_105858

theorem sector_area (arc_length : Real) (central_angle : Real) (area : Real) :
  arc_length = 4 * Real.pi →
  central_angle = Real.pi / 3 →
  area = 24 * Real.pi :=
by
  sorry

end sector_area_l1058_105858
