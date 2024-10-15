import Mathlib

namespace NUMINAMATH_CALUDE_walking_speed_problem_l3385_338587

/-- Proves that the speed at which a person would have walked is 10 km/hr,
    given the conditions of the problem. -/
theorem walking_speed_problem (actual_distance : ℝ) (additional_distance : ℝ) 
  (actual_speed : ℝ) :
  actual_distance = 20 →
  additional_distance = 20 →
  actual_speed = 5 →
  ∃ (speed : ℝ),
    speed = actual_speed + 5 ∧
    actual_distance / actual_speed = (actual_distance + additional_distance) / speed ∧
    speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3385_338587


namespace NUMINAMATH_CALUDE_inconsistent_equation_l3385_338504

theorem inconsistent_equation : ¬ (3 * (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_equation_l3385_338504


namespace NUMINAMATH_CALUDE_second_book_length_is_100_l3385_338501

/-- The length of Yasna's first book in pages -/
def first_book_length : ℕ := 180

/-- The number of pages Yasna reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The total number of pages Yasna reads in two weeks -/
def total_pages : ℕ := pages_per_day * days_in_two_weeks

/-- The length of Yasna's second book in pages -/
def second_book_length : ℕ := total_pages - first_book_length

theorem second_book_length_is_100 : second_book_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_book_length_is_100_l3385_338501


namespace NUMINAMATH_CALUDE_door_height_is_eight_l3385_338569

/-- Represents the dimensions of a rectangular door and a pole satisfying specific conditions -/
structure DoorAndPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  horizontal_condition : pole_length = door_width + 4
  vertical_condition : pole_length = door_height + 2
  diagonal_condition : pole_length^2 = door_width^2 + door_height^2

/-- Theorem stating that for any DoorAndPole structure, the door height is 8 -/
theorem door_height_is_eight (d : DoorAndPole) : d.door_height = 8 := by
  sorry

#check door_height_is_eight

end NUMINAMATH_CALUDE_door_height_is_eight_l3385_338569


namespace NUMINAMATH_CALUDE_xyz_product_l3385_338551

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3385_338551


namespace NUMINAMATH_CALUDE_solve_equation_l3385_338506

theorem solve_equation : ∃ x : ℝ, (4 / 7) * (1 / 5) * x = 2 ∧ x = 35 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3385_338506


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l3385_338502

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) : 
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l3385_338502


namespace NUMINAMATH_CALUDE_triangle_properties_l3385_338505

/-- Given a triangle ABC with the following properties:
    - B has coordinates (1, -2)
    - The median CM on side AB has equation 2x - y + 1 = 0
    - The angle bisector of ∠BAC has equation x + 7y - 12 = 0
    Prove that:
    1. A has coordinates (-2, 2)
    2. The equation of line AC is 3x - 4y + 14 = 0
-/
theorem triangle_properties (B : ℝ × ℝ) (median_CM : ℝ → ℝ → ℝ) (angle_bisector : ℝ → ℝ → ℝ) 
  (hB : B = (1, -2))
  (hmedian : ∀ x y, median_CM x y = 2 * x - y + 1)
  (hbisector : ∀ x y, angle_bisector x y = x + 7 * y - 12) :
  ∃ (A : ℝ × ℝ) (line_AC : ℝ → ℝ → ℝ),
    A = (-2, 2) ∧ 
    (∀ x y, line_AC x y = 3 * x - 4 * y + 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3385_338505


namespace NUMINAMATH_CALUDE_smallest_valid_sum_of_cubes_l3385_338599

def is_valid (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p ∣ n → p > 18

def is_sum_of_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^3 + b^3

theorem smallest_valid_sum_of_cubes : 
  is_valid 1843 ∧ 
  is_sum_of_cubes 1843 ∧ 
  ∀ m : ℕ, m < 1843 → ¬(is_valid m ∧ is_sum_of_cubes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_sum_of_cubes_l3385_338599


namespace NUMINAMATH_CALUDE_am_gm_squared_max_value_on_interval_max_value_sqrt_function_l3385_338538

-- Statement 1
theorem am_gm_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ ((a + b) / 2) ^ 2 := by sorry

-- Statement 2
theorem max_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c := by sorry

theorem max_value_sqrt_function :
  ∃ c ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, x * Real.sqrt (4 - x^2) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_am_gm_squared_max_value_on_interval_max_value_sqrt_function_l3385_338538


namespace NUMINAMATH_CALUDE_bobby_candy_count_l3385_338576

/-- The total number of candy pieces Bobby ate -/
def total_candy (initial : ℕ) (more : ℕ) (chocolate : ℕ) : ℕ :=
  initial + more + chocolate

/-- Theorem stating that Bobby ate 133 pieces of candy in total -/
theorem bobby_candy_count :
  total_candy 28 42 63 = 133 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l3385_338576


namespace NUMINAMATH_CALUDE_max_value_of_function_l3385_338559

theorem max_value_of_function (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x * (1 - x^2)) →
  (∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f x ≤ f x₀) →
  (∃ x₀ ∈ Set.Icc 0 1, f x₀ = 2 * Real.sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3385_338559


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_intersection_l3385_338583

/-- The equation |x + a| = 1/x has exactly two solutions if and only if a = -2 -/
theorem absolute_value_reciprocal_intersection (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ + a| = 1/x₁ ∧ |x₂ + a| = 1/x₂) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_intersection_l3385_338583


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3385_338558

theorem quadratic_equation_solution :
  ∀ x : ℝ, x > 0 → (7 * x^2 - 8 * x - 6 = 0) → (x = 6/7 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3385_338558


namespace NUMINAMATH_CALUDE_dan_remaining_limes_l3385_338537

/-- The number of limes Dan has after giving some to Sara -/
def limes_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 5 limes remaining -/
theorem dan_remaining_limes :
  limes_remaining 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_remaining_limes_l3385_338537


namespace NUMINAMATH_CALUDE_cube_root_of_27_l3385_338565

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l3385_338565


namespace NUMINAMATH_CALUDE_lady_arrangements_proof_l3385_338580

def num_gentlemen : ℕ := 6
def num_ladies : ℕ := 3
def total_positions : ℕ := 9

def valid_arrangements : ℕ := 129600

theorem lady_arrangements_proof :
  (num_gentlemen + num_ladies = total_positions) →
  (valid_arrangements = num_gentlemen.factorial * (num_gentlemen + 1).choose num_ladies) :=
by sorry

end NUMINAMATH_CALUDE_lady_arrangements_proof_l3385_338580


namespace NUMINAMATH_CALUDE_new_average_after_22_innings_l3385_338597

def calculate_new_average (initial_innings : ℕ) (score_17th : ℕ) (average_increase : ℕ) 
  (additional_scores : List ℕ) : ℕ :=
  let total_innings := initial_innings + additional_scores.length
  let initial_average := (initial_innings - 1) * (average_increase + 1) / initial_innings
  let total_runs_17 := initial_innings * (initial_average + average_increase)
  let total_runs_22 := total_runs_17 + additional_scores.sum
  total_runs_22 / total_innings

theorem new_average_after_22_innings : 
  calculate_new_average 17 85 3 [100, 120, 45, 75, 65] = 47 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_22_innings_l3385_338597


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3385_338589

theorem two_numbers_problem (a b : ℝ) : 
  a + b = 90 ∧ 
  0.4 * a = 0.3 * b + 15 → 
  a = 60 ∧ b = 30 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3385_338589


namespace NUMINAMATH_CALUDE_base_six_units_digit_l3385_338508

theorem base_six_units_digit : 
  (123 * 78 - 156) % 6 = 0 := by sorry

end NUMINAMATH_CALUDE_base_six_units_digit_l3385_338508


namespace NUMINAMATH_CALUDE_coat_final_price_coat_price_is_81_l3385_338542

/-- The final price of a coat after discounts and tax -/
theorem coat_final_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (additional_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + tax_rate)
  final_price

/-- Proof that the final price of the coat is $81 -/
theorem coat_price_is_81 : 
  coat_final_price 100 0.2 5 0.08 = 81 := by
  sorry

end NUMINAMATH_CALUDE_coat_final_price_coat_price_is_81_l3385_338542


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3385_338590

theorem factorial_fraction_equals_one :
  (3 * Nat.factorial 5 + 15 * Nat.factorial 4) / Nat.factorial 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3385_338590


namespace NUMINAMATH_CALUDE_sum_even_coefficients_is_seven_l3385_338521

/-- Given a polynomial equation, prove that the sum of even-indexed coefficients (excluding a₀) is 7 -/
theorem sum_even_coefficients_is_seven (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^8 = a*x^12 + a₁*x^11 + a₂*x^10 + a₃*x^9 + a₄*x^8 + 
    a₅*x^7 + a₆*x^6 + a₇*x^5 + a₈*x^4 + a₉*x^3 + a₁₀*x^2 + a₁₁*x + a₁₂) →
  a = 1 →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 7 := by
sorry


end NUMINAMATH_CALUDE_sum_even_coefficients_is_seven_l3385_338521


namespace NUMINAMATH_CALUDE_candy_chocolate_price_difference_l3385_338518

def candy_bar_original_price : ℝ := 6
def candy_bar_discount : ℝ := 0.25
def chocolate_original_price : ℝ := 3
def chocolate_discount : ℝ := 0.10

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

theorem candy_chocolate_price_difference :
  discounted_price candy_bar_original_price candy_bar_discount -
  discounted_price chocolate_original_price chocolate_discount = 1.80 := by
sorry

end NUMINAMATH_CALUDE_candy_chocolate_price_difference_l3385_338518


namespace NUMINAMATH_CALUDE_max_value_problem_l3385_338535

theorem max_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3*x + 6*y < 108) :
  (x^2 * y * (108 - 3*x - 6*y)) ≤ 7776 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 6*y₀ < 108 ∧
    x₀^2 * y₀ * (108 - 3*x₀ - 6*y₀) = 7776 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l3385_338535


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l3385_338541

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x * Real.log x - a < 0) ↔ a > -1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l3385_338541


namespace NUMINAMATH_CALUDE_haley_money_received_l3385_338595

/-- Proves that Haley received 13 dollars from doing chores and her birthday -/
theorem haley_money_received (initial_amount : ℕ) (difference : ℕ) : 
  initial_amount = 2 → difference = 11 → initial_amount + difference = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_haley_money_received_l3385_338595


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l3385_338523

/-- The function f(x) = x - a/x - 2ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - 2 * Real.log x

/-- Predicate to check if x is an extreme point of f -/
def is_extreme_point (a : ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f a y ≠ f a x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  is_extreme_point a x₁ →
  is_extreme_point a x₂ →
  f a x₂ < x₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l3385_338523


namespace NUMINAMATH_CALUDE_unique_solution_l3385_338500

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x ^ (floor x) = 9 / 2

-- State the theorem
theorem unique_solution : 
  ∃! x : ℝ, equation x ∧ x = (3 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3385_338500


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3385_338584

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3385_338584


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l3385_338578

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l3385_338578


namespace NUMINAMATH_CALUDE_intersection_slope_l3385_338564

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

/-- Theorem stating that the slope of the line formed by the intersection points of the two circles is -1 -/
theorem intersection_slope : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ circle1 x2 y2 ∧ 
    circle2 x1 y1 ∧ circle2 x2 y2 ∧ 
    x1 ≠ x2 ∧
    (y2 - y1) / (x2 - x1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l3385_338564


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3385_338577

theorem factorial_equation_solution (m k : ℕ) (hm : m = 7) (hk : k = 12) :
  ∃ P : ℕ, (Nat.factorial 7) * (Nat.factorial 14) = 18 * P * (Nat.factorial 11) ∧ P = 54080 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3385_338577


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l3385_338528

theorem polynomial_always_positive (x y : ℝ) : x^2 + y^2 - 2*x - 4*y + 16 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l3385_338528


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3385_338566

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 45 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (a + b) / 2 = 40 →       -- The average weight of a and b is 40 kg
  b = 35 →                 -- The weight of b is 35 kg
  (b + c) / 2 = 45 :=      -- The average weight of b and c is 45 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3385_338566


namespace NUMINAMATH_CALUDE_system_solution_l3385_338503

theorem system_solution (x y z : ℝ) : 
  x * y = 15 - 3 * x - 2 * y →
  y * z = 8 - 2 * y - 4 * z →
  x * z = 56 - 5 * x - 6 * z →
  x > 0 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3385_338503


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3385_338546

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 4 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3385_338546


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3385_338568

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a - Complex.I) * (1 + a * Complex.I) = -4 + 3 * Complex.I →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3385_338568


namespace NUMINAMATH_CALUDE_visibility_condition_l3385_338512

/-- The curve C: y = 2x^2 -/
def C (x : ℝ) : ℝ := 2 * x^2

/-- Point A -/
def A : ℝ × ℝ := (0, -2)

/-- Point B -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- A point (x, y) is above the curve C -/
def is_above_curve (x y : ℝ) : Prop := y > C x

/-- A point (x, y) is on or below the line passing through two points -/
def is_on_or_below_line (x1 y1 x2 y2 x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) ≤ (y2 - y1) * (x - x1)

/-- B is visible from A without being obstructed by C -/
def is_visible (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 →
    is_above_curve x ((a + 2) / 3 * x - 2)

theorem visibility_condition (a : ℝ) :
  is_visible a ↔ a < 10 := by sorry

end NUMINAMATH_CALUDE_visibility_condition_l3385_338512


namespace NUMINAMATH_CALUDE_square_greater_than_abs_l3385_338550

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l3385_338550


namespace NUMINAMATH_CALUDE_system_solution_l3385_338560

theorem system_solution (x y b : ℚ) : 
  (4 * x + 3 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -21 / 5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3385_338560


namespace NUMINAMATH_CALUDE_three_Z_five_equals_32_l3385_338515

/-- The Z operation as defined in the problem -/
def Z (a b : ℝ) : ℝ := b + 12 * a - a^2

/-- Theorem stating that 3 Z 5 equals 32 -/
theorem three_Z_five_equals_32 : Z 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_32_l3385_338515


namespace NUMINAMATH_CALUDE_albatrocity_to_finchester_distance_l3385_338555

/-- The distance from Albatrocity to Finchester in miles -/
def distance : ℝ := 75

/-- The speed of the pigeon in still air in miles per hour -/
def pigeon_speed : ℝ := 40

/-- The wind speed from Albatrocity to Finchester in miles per hour -/
def wind_speed : ℝ := 10

/-- The time for a round trip without wind in hours -/
def no_wind_time : ℝ := 3.75

/-- The time for a round trip with wind in hours -/
def wind_time : ℝ := 4

theorem albatrocity_to_finchester_distance :
  (2 * distance / pigeon_speed = no_wind_time) ∧
  (distance / (pigeon_speed + wind_speed) + distance / (pigeon_speed - wind_speed) = wind_time) →
  distance = 75 := by sorry

end NUMINAMATH_CALUDE_albatrocity_to_finchester_distance_l3385_338555


namespace NUMINAMATH_CALUDE_prob_heads_and_five_l3385_338585

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1 / 2

/-- The probability of rolling a 5 on a regular eight-sided die -/
def prob_five : ℚ := 1 / 8

/-- The events (coin flip and die roll) are independent -/
axiom events_independent : True

theorem prob_heads_and_five : prob_heads * prob_five = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_heads_and_five_l3385_338585


namespace NUMINAMATH_CALUDE_cookies_eaten_difference_l3385_338531

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) 
  (h1 : initial_sweet = 37)
  (h2 : initial_salty = 11)
  (h3 : eaten_sweet = 5)
  (h4 : eaten_salty = 2) :
  eaten_sweet - eaten_salty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_difference_l3385_338531


namespace NUMINAMATH_CALUDE_total_days_2010_to_2015_l3385_338524

def is_leap_year (year : ℕ) : Bool :=
  year = 2012

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def years_in_period : List ℕ := [2010, 2011, 2012, 2013, 2014, 2015]

theorem total_days_2010_to_2015 :
  (years_in_period.map days_in_year).sum = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2015_l3385_338524


namespace NUMINAMATH_CALUDE_sqrt_three_bounds_l3385_338593

theorem sqrt_three_bounds (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3) ∧ (Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_bounds_l3385_338593


namespace NUMINAMATH_CALUDE_toms_allowance_l3385_338588

theorem toms_allowance (allowance : ℝ) : 
  (allowance - allowance / 3 - (allowance - allowance / 3) / 4 = 6) → allowance = 12 := by
  sorry

end NUMINAMATH_CALUDE_toms_allowance_l3385_338588


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3385_338571

-- Define the quadratic function
def f (a x : ℝ) := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- State the theorem
theorem quadratic_inequality_range :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3385_338571


namespace NUMINAMATH_CALUDE_non_vegan_gluten_cupcakes_l3385_338549

/-- Given a set of cupcakes with specific properties, prove that the number of non-vegan cupcakes containing gluten is 28. -/
theorem non_vegan_gluten_cupcakes
  (total : ℕ)
  (gluten_free : ℕ)
  (vegan : ℕ)
  (vegan_gluten_free : ℕ)
  (h1 : total = 80)
  (h2 : gluten_free = total / 2)
  (h3 : vegan = 24)
  (h4 : vegan_gluten_free = vegan / 2)
  : total - gluten_free - (vegan - vegan_gluten_free) = 28 := by
  sorry

#check non_vegan_gluten_cupcakes

end NUMINAMATH_CALUDE_non_vegan_gluten_cupcakes_l3385_338549


namespace NUMINAMATH_CALUDE_cyclist_round_trip_l3385_338509

/-- Cyclist's round trip problem -/
theorem cyclist_round_trip
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (second_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_speed : ℝ)
  (total_round_trip_time : ℝ)
  (h1 : total_distance = first_leg_distance + second_leg_distance)
  (h2 : first_leg_distance = 18)
  (h3 : second_leg_distance = 12)
  (h4 : first_leg_speed = 9)
  (h5 : second_leg_speed = 10)
  (h6 : total_round_trip_time = 7.2)
  : (2 * total_distance) / (total_round_trip_time - (first_leg_distance / first_leg_speed + second_leg_distance / second_leg_speed)) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_round_trip_l3385_338509


namespace NUMINAMATH_CALUDE_at_least_two_positive_l3385_338522

theorem at_least_two_positive (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c > 0) (h5 : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_positive_l3385_338522


namespace NUMINAMATH_CALUDE_larger_number_problem_l3385_338527

theorem larger_number_problem (a b : ℝ) : 
  a + b = 40 → a - b = 10 → a > b → a = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3385_338527


namespace NUMINAMATH_CALUDE_inverse_function_graph_point_l3385_338517

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse function of f
variable (h_inv : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f)

-- Given condition: f(3) = 0
variable (h_f_3 : f 3 = 0)

-- Theorem statement
theorem inverse_function_graph_point :
  (f_inv ((-1) + 1) = 3) ∧ (f_inv ∘ (fun x ↦ x + 1)) (-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_graph_point_l3385_338517


namespace NUMINAMATH_CALUDE_cafeteria_tile_problem_l3385_338536

theorem cafeteria_tile_problem :
  let current_tiles : ℕ := 630
  let current_area : ℕ := 18
  let new_tile_side : ℕ := 6
  let new_tiles : ℕ := 315
  (current_tiles * current_area = new_tiles * new_tile_side * new_tile_side) :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_tile_problem_l3385_338536


namespace NUMINAMATH_CALUDE_union_M_N_complement_intersection_M_N_l3385_338513

-- Define the universal set U
def U : Set ℝ := {x | -6 ≤ x ∧ x ≤ 5}

-- Define set M
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for ∁U(M ∩ N)
theorem complement_intersection_M_N : 
  (M ∩ N)ᶜ = {x ∈ U | x ≤ 0 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_intersection_M_N_l3385_338513


namespace NUMINAMATH_CALUDE_even_digits_in_base7_of_528_l3385_338553

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 528₁₀ is 0 -/
theorem even_digits_in_base7_of_528 : 
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base7_of_528_l3385_338553


namespace NUMINAMATH_CALUDE_line_through_point_l3385_338539

/-- Given a line ax + (a+1)y = a+2 that passes through the point (4, -8), prove that a = -2 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 1) * y = a + 2 → x = 4 ∧ y = -8) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l3385_338539


namespace NUMINAMATH_CALUDE_g_domain_l3385_338548

def f_domain : Set ℝ := Set.Icc 0 2

def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1)

theorem g_domain (f : ℝ → ℝ) (h : ∀ x, x ∈ f_domain ↔ f x ≠ 0) :
  ∀ x, x ∈ Set.Icc (-1) 1 ↔ g f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_g_domain_l3385_338548


namespace NUMINAMATH_CALUDE_spring_length_at_6kg_l3385_338570

/-- Represents the relationship between weight and spring length -/
def spring_length (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ) : ℝ :=
  initial_length + stretch_rate * weight

/-- Theorem stating that a spring with initial length 8 cm and stretch rate 0.5 cm/kg 
    will have a length of 11 cm when a 6 kg weight is hung -/
theorem spring_length_at_6kg 
  (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ)
  (h1 : initial_length = 8)
  (h2 : stretch_rate = 0.5)
  (h3 : weight = 6) :
  spring_length initial_length stretch_rate weight = 11 := by
  sorry

end NUMINAMATH_CALUDE_spring_length_at_6kg_l3385_338570


namespace NUMINAMATH_CALUDE_g_3_equals_25_l3385_338592

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^7 + q * x^3 + r * x + 7

-- State the theorem
theorem g_3_equals_25 (p q r : ℝ) :
  (g p q r (-3) = -11) →
  (∀ x, g p q r x + g p q r (-x) = 14) →
  g p q r 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_g_3_equals_25_l3385_338592


namespace NUMINAMATH_CALUDE_committee_selection_l3385_338533

theorem committee_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3385_338533


namespace NUMINAMATH_CALUDE_largest_multiple_in_sequence_l3385_338543

theorem largest_multiple_in_sequence : 
  ∀ (n : ℕ), 
  (3*n + 3*(n+1) + 3*(n+2) = 117) → 
  (max (3*n) (max (3*(n+1)) (3*(n+2))) = 42) := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_in_sequence_l3385_338543


namespace NUMINAMATH_CALUDE_smallest_cube_with_specific_digits_l3385_338567

/-- Returns the first n digits of a natural number -/
def firstNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Returns the last n digits of a natural number -/
def lastNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Checks if the first n digits of a natural number are all 1 -/
def firstNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  firstNDigits n x = 10^n - 1

/-- Checks if the last n digits of a natural number are all 1 -/
def lastNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  lastNDigits n x = 10^n - 1

theorem smallest_cube_with_specific_digits :
  ∀ x : ℕ, x ≥ 1038471 →
    (firstNDigitsAreOne 3 (x^3) ∧ lastNDigitsAreOne 4 (x^3)) →
    x = 1038471 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_with_specific_digits_l3385_338567


namespace NUMINAMATH_CALUDE_root_implies_k_value_l3385_338562

theorem root_implies_k_value (k : ℝ) : 
  (2 * (4 : ℝ)^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l3385_338562


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l3385_338532

theorem base_10_to_base_7_conversion :
  ∃ (a b c d : ℕ),
    746 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l3385_338532


namespace NUMINAMATH_CALUDE_min_absolute_difference_l3385_338556

/-- The minimum absolute difference between n and m, given f(m) = g(n) -/
theorem min_absolute_difference (f g : ℝ → ℝ) (m n : ℝ) : 
  (f = fun x ↦ Real.exp x + 2 * x) →
  (g = fun x ↦ 4 * x) →
  (f m = g n) →
  ∃ (min_diff : ℝ), 
    (∀ (m' n' : ℝ), f m' = g n' → |n' - m'| ≥ min_diff) ∧ 
    (min_diff = 1/2 - 1/2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_min_absolute_difference_l3385_338556


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3385_338514

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 5 * y = a →
  10 * y - 15 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3385_338514


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3385_338573

/-- Given a triangle ABC with angle B = 45°, side c = 2√2, and side b = 4√3/3,
    prove that angle A is either 7π/12 or π/12 -/
theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) : 
  B = π/4 → c = 2 * Real.sqrt 2 → b = 4 * Real.sqrt 3 / 3 →
  A = 7*π/12 ∨ A = π/12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3385_338573


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3385_338557

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  ¬(∀ x : ℝ, x > 1 → x > 3) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3385_338557


namespace NUMINAMATH_CALUDE_store_earnings_l3385_338581

/-- Calculates the total earnings from selling shirts and jeans --/
def total_earnings (shirt_price : ℕ) (shirt_quantity : ℕ) (jeans_quantity : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  shirt_price * shirt_quantity + jeans_price * jeans_quantity

/-- Proves that the total earnings from selling 20 shirts at $10 each and 10 pairs of jeans at twice the price of a shirt is $400 --/
theorem store_earnings : total_earnings 10 20 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_l3385_338581


namespace NUMINAMATH_CALUDE_special_circle_equation_l3385_338561

/-- A circle passing through the origin with center on the negative x-axis and radius 2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_negative_x_axis : center.1 < 0 ∧ center.2 = 0
  passes_through_origin : (center.1 ^ 2 + center.2 ^ 2) = radius ^ 2
  radius_is_two : radius = 2

/-- The equation of the special circle is (x+2)^2 + y^2 = 4 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ (x y : ℝ), ((x + 2) ^ 2 + y ^ 2 = 4) ↔ 
  ((x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l3385_338561


namespace NUMINAMATH_CALUDE_negation_equivalence_l3385_338594

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3385_338594


namespace NUMINAMATH_CALUDE_fraction_equality_l3385_338591

theorem fraction_equality : (2222 - 2121)^2 / 196 = 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3385_338591


namespace NUMINAMATH_CALUDE_unique_tangent_circle_l3385_338520

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Configuration of three unit circles each tangent to the other two -/
def unit_circle_configuration (c1 c2 c3 : Circle) : Prop :=
  c1.radius = 1 ∧ c2.radius = 1 ∧ c3.radius = 1 ∧
  are_tangent c1 c2 ∧ are_tangent c2 c3 ∧ are_tangent c3 c1

/-- A circle of radius 2 tangent to all three unit circles -/
def tangent_circle (c : Circle) (c1 c2 c3 : Circle) : Prop :=
  c.radius = 2 ∧ are_tangent c c1 ∧ are_tangent c c2 ∧ are_tangent c c3

theorem unique_tangent_circle (c1 c2 c3 : Circle) :
  unit_circle_configuration c1 c2 c3 →
  ∃! c : Circle, tangent_circle c c1 c2 c3 := by
  sorry

end NUMINAMATH_CALUDE_unique_tangent_circle_l3385_338520


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3385_338511

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0) → 
  (∃ k, k = (Nat.floor (999 / 7) - Nat.ceil (100 / 7) + 1) ∧ k = 128) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3385_338511


namespace NUMINAMATH_CALUDE_job_completion_time_l3385_338554

/-- Represents the job completion scenario with changing number of workers -/
structure JobCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  work_days_before_change : ℕ
  additional_workers : ℕ
  total_days : ℚ

/-- Theorem stating that under the given conditions, the job will be completed in 3.5 days -/
theorem job_completion_time (job : JobCompletion) :
  job.initial_workers = 6 ∧
  job.initial_days = 8 ∧
  job.work_days_before_change = 3 ∧
  job.additional_workers = 4 →
  job.total_days = 3.5 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l3385_338554


namespace NUMINAMATH_CALUDE_accurate_to_thousands_l3385_338563

/-- Represents a large number in millions with one decimal place -/
structure LargeNumber where
  whole : ℕ
  decimal : ℕ
  inv_ten : decimal < 10

/-- Converts a LargeNumber to its full integer representation -/
def LargeNumber.toInt (n : LargeNumber) : ℕ := n.whole * 1000000 + n.decimal * 100000

/-- Represents the place value in a number system -/
inductive PlaceValue
  | Thousands
  | Hundreds
  | Tens
  | Ones
  | Tenths
  | Hundredths

/-- Determines the smallest accurately represented place value for a given LargeNumber -/
def smallestAccuratePlaceValue (n : LargeNumber) : PlaceValue := 
  if n.decimal % 10 = 0 then PlaceValue.Hundreds else PlaceValue.Thousands

theorem accurate_to_thousands (n : LargeNumber) 
  (h : n.whole = 42 ∧ n.decimal = 3) : 
  smallestAccuratePlaceValue n = PlaceValue.Thousands := by
  sorry

end NUMINAMATH_CALUDE_accurate_to_thousands_l3385_338563


namespace NUMINAMATH_CALUDE_prism_volume_l3385_338545

/-- The volume of a right rectangular prism with face areas 100, 200, and 300 square units -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 100)
  (h2 : b * c = 200)
  (h3 : c * a = 300) : 
  a * b * c = 1000 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l3385_338545


namespace NUMINAMATH_CALUDE_power_four_times_four_equals_eight_l3385_338579

theorem power_four_times_four_equals_eight (a : ℝ) : a ^ 4 * a ^ 4 = a ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_four_equals_eight_l3385_338579


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3385_338530

theorem tangent_line_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + (y - 3)^2 = 5 ∧ y = 2*x) →
  (a = -1 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3385_338530


namespace NUMINAMATH_CALUDE_grape_juice_mixture_proof_l3385_338507

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture
    containing 20% grape juice results in a new mixture with 36% grape juice. -/
theorem grape_juice_mixture_proof :
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.20
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.36
  let initial_juice : ℝ := initial_mixture * initial_concentration
  let final_mixture : ℝ := initial_mixture + added_juice
  let final_juice : ℝ := initial_juice + added_juice
  final_juice / final_mixture = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_proof_l3385_338507


namespace NUMINAMATH_CALUDE_multiply_5915581_7907_l3385_338582

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := by
  sorry

end NUMINAMATH_CALUDE_multiply_5915581_7907_l3385_338582


namespace NUMINAMATH_CALUDE_abc_fraction_value_l3385_338525

theorem abc_fraction_value (a b c : ℕ+) 
  (h : a^2*b + b^2*c + a*c^2 + a + b + c = 2*(a*b + b*c + a*c)) :
  (c : ℚ)^2017 / ((a : ℚ)^2016 + (b : ℚ)^2018) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l3385_338525


namespace NUMINAMATH_CALUDE_exists_same_dimensions_l3385_338516

/-- Represents a rectangle with width and height as powers of two -/
structure Rectangle where
  width : Nat
  height : Nat
  width_pow_two : ∃ k : Nat, width = 2^k
  height_pow_two : ∃ k : Nat, height = 2^k

/-- Represents a tiling of a square -/
structure Tiling where
  n : Nat
  rectangles : List Rectangle
  at_least_two : rectangles.length ≥ 2
  covers_square : ∀ (x y : Nat), x < 2^n ∧ y < 2^n → 
    ∃ (r : Rectangle), r ∈ rectangles ∧ x < r.width ∧ y < r.height
  non_overlapping : ∀ (r1 r2 : Rectangle), r1 ∈ rectangles ∧ r2 ∈ rectangles ∧ r1 ≠ r2 →
    ∀ (x y : Nat), ¬(x < r1.width ∧ y < r1.height ∧ x < r2.width ∧ y < r2.height)

/-- Main theorem: There exist at least two rectangles with the same dimensions in any valid tiling -/
theorem exists_same_dimensions (t : Tiling) : 
  ∃ (r1 r2 : Rectangle), r1 ∈ t.rectangles ∧ r2 ∈ t.rectangles ∧ r1 ≠ r2 ∧ 
    r1.width = r2.width ∧ r1.height = r2.height :=
by sorry

end NUMINAMATH_CALUDE_exists_same_dimensions_l3385_338516


namespace NUMINAMATH_CALUDE_unique_solution_l3385_338575

theorem unique_solution : ∃! (x y : ℕ+), 
  (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) ∧ 
  3 * (x : ℝ)^(y : ℝ) = (y : ℝ)^(x : ℝ) + 13 ∧
  x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3385_338575


namespace NUMINAMATH_CALUDE_second_meeting_time_is_six_minutes_l3385_338552

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and the swimming scenario --/
structure Pool where
  length : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (pool : Pool) : ℝ :=
  sorry

/-- Theorem stating the conditions and the result to be proved --/
theorem second_meeting_time_is_six_minutes 
  (pool : Pool)
  (h1 : pool.length = 120)
  (h2 : pool.swimmer1.startPosition = 0)
  (h3 : pool.swimmer2.startPosition = 120)
  (h4 : pool.firstMeetingPosition = 40)
  (h5 : pool.firstMeetingTime = 2) :
  secondMeetingTime pool = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_six_minutes_l3385_338552


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l3385_338529

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l3385_338529


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l3385_338519

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (-1, -1)
  f2 : ℝ × ℝ := (-1, -3)
  -- Point on the ellipse
  p : ℝ × ℝ := (4, -2)
  -- Constants in the ellipse equation
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- The point p satisfies the ellipse equation
  eq_satisfied : (((p.1 - h)^2 / a^2) + ((p.2 - k)^2 / b^2)) = 1

/-- Theorem stating that a + k = 3 for the given ellipse -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l3385_338519


namespace NUMINAMATH_CALUDE_child_play_time_l3385_338572

theorem child_play_time (num_children : ℕ) (children_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 7)
  (h2 : children_per_game = 2)
  (h3 : total_time = 140)
  (h4 : children_per_game ≤ num_children)
  (h5 : children_per_game > 0)
  (h6 : total_time > 0) :
  (children_per_game * total_time) / num_children = 40 := by
sorry

end NUMINAMATH_CALUDE_child_play_time_l3385_338572


namespace NUMINAMATH_CALUDE_max_flour_mass_difference_l3385_338540

/-- The mass of a bag of flour in kg -/
structure FlourBag where
  mass : ℝ
  mass_range : mass ∈ Set.Icc (25 - 0.2) (25 + 0.2)

/-- The maximum difference in mass between two bags of flour -/
def max_mass_difference (bag1 bag2 : FlourBag) : ℝ :=
  |bag1.mass - bag2.mass|

/-- Theorem stating the maximum possible difference in mass between two bags of flour -/
theorem max_flour_mass_difference :
  ∃ (bag1 bag2 : FlourBag), max_mass_difference bag1 bag2 = 0.4 ∧
  ∀ (bag3 bag4 : FlourBag), max_mass_difference bag3 bag4 ≤ 0.4 := by
sorry

end NUMINAMATH_CALUDE_max_flour_mass_difference_l3385_338540


namespace NUMINAMATH_CALUDE_max_value_a_l3385_338547

theorem max_value_a (a : ℝ) : 
  (∀ x k, x ∈ Set.Ioo 0 6 → k ∈ Set.Icc (-1) 1 → 
    6 * Real.log x + x^2 - 8*x + a ≤ k*x) → 
  a ≤ 6 - 6 * Real.log 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l3385_338547


namespace NUMINAMATH_CALUDE_kangaroo_exhibition_arrangements_l3385_338586

/-- The number of ways to arrange n uniquely tall kangaroos in a row -/
def kangaroo_arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely tall kangaroos in a row,
    with the two tallest at the ends -/
def kangaroo_arrangements_with_tallest_at_ends (n : ℕ) : ℕ :=
  2 * kangaroo_arrangements (n - 2)

theorem kangaroo_exhibition_arrangements :
  kangaroo_arrangements_with_tallest_at_ends 8 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_exhibition_arrangements_l3385_338586


namespace NUMINAMATH_CALUDE_sqrt2_irrational_l3385_338534

theorem sqrt2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_l3385_338534


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l3385_338574

/-- Proves that the difference in ribbon length between two wrapping methods
    for a box matches one side of the box. -/
theorem ribbon_length_difference (l w h bow : ℕ) 
  (hl : l = 22) (hw : w = 22) (hh : h = 11) (hbow : bow = 24) :
  (2 * l + 4 * w + 2 * h + bow) - (2 * l + 2 * w + 4 * h + bow) = l := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l3385_338574


namespace NUMINAMATH_CALUDE_x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l3385_338544

theorem x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three :
  (∀ x : ℝ, x < 0 → x ≠ 3) ∧
  (∃ x : ℝ, x ≠ 3 ∧ x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l3385_338544


namespace NUMINAMATH_CALUDE_tony_lego_purchase_l3385_338526

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_price : ℕ
  sword_price : ℕ
  dough_price : ℕ
  sword_count : ℕ
  dough_count : ℕ
  total_paid : ℕ

/-- Calculates the number of Lego sets bought -/
def lego_sets_bought (purchase : ToyPurchase) : ℕ :=
  (purchase.total_paid - purchase.sword_price * purchase.sword_count - purchase.dough_price * purchase.dough_count) / purchase.lego_price

/-- Theorem stating that Tony bought 3 sets of Lego blocks -/
theorem tony_lego_purchase : 
  ∀ (purchase : ToyPurchase), 
  purchase.lego_price = 250 ∧ 
  purchase.sword_price = 120 ∧ 
  purchase.dough_price = 35 ∧ 
  purchase.sword_count = 7 ∧ 
  purchase.dough_count = 10 ∧ 
  purchase.total_paid = 1940 → 
  lego_sets_bought purchase = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_lego_purchase_l3385_338526


namespace NUMINAMATH_CALUDE_mother_bought_pencils_l3385_338510

def dozen : ℕ := 12

def initial_pencils : ℕ := 17

def total_pencils : ℕ := 2 * dozen

theorem mother_bought_pencils : total_pencils - initial_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_mother_bought_pencils_l3385_338510


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3385_338598

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 * i) / (1 - i)
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3385_338598


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3385_338596

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 57 → Nat.gcd (15*m - 9) (11*m + 10) = 1) ∧ 
  Nat.gcd (15*57 - 9) (11*57 + 10) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3385_338596
