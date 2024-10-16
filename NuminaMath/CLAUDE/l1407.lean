import Mathlib

namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_pow_fifteen_l1407_140730

theorem last_digit_of_one_over_two_pow_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / (2 ^ n) * 10^15 % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_pow_fifteen_l1407_140730


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_less_120_l1407_140708

theorem greatest_common_multiple_9_15_less_120 :
  ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  90 % 9 = 0 ∧ 90 % 15 = 0 ∧ 90 < 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_less_120_l1407_140708


namespace NUMINAMATH_CALUDE_incorrect_expression_l1407_140738

theorem incorrect_expression (x y : ℝ) (h : x / y = 2 / 5) :
  (x + 3 * y) / x ≠ 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1407_140738


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l1407_140729

/-- Proves that adding 3 liters of pure alcohol to a 6-liter solution
    that is 25% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.25
  let added_alcohol : ℝ := 3
  let final_concentration : ℝ := 0.5
  let final_volume : ℝ := initial_volume + added_alcohol
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_alcohol : ℝ := initial_alcohol + added_alcohol
  final_alcohol / final_volume = final_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l1407_140729


namespace NUMINAMATH_CALUDE_first_sequence_general_term_second_sequence_general_term_l1407_140721

/-- First sequence -/
def S₁ (n : ℕ) : ℚ := n^2 + (1/2) * n

/-- Second sequence -/
def S₂ (n : ℕ) : ℚ := (1/4) * n^2 + (2/3) * n + 3

/-- General term of the first sequence -/
def a₁ (n : ℕ) : ℚ := 2 * n - 1/2

/-- General term of the second sequence -/
def a₂ (n : ℕ) : ℚ :=
  if n = 1 then 47/12 else (6 * n + 5) / 12

theorem first_sequence_general_term (n : ℕ) :
  S₁ (n + 1) - S₁ n = a₁ (n + 1) :=
sorry

theorem second_sequence_general_term (n : ℕ) :
  S₂ (n + 1) - S₂ n = a₂ (n + 1) :=
sorry

end NUMINAMATH_CALUDE_first_sequence_general_term_second_sequence_general_term_l1407_140721


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l1407_140753

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 → b = 5 → c = 3 → d = 4 → 
  (a + b - c - d * e = a + (b - (c - (d * e)))) → e = 0 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l1407_140753


namespace NUMINAMATH_CALUDE_badminton_probability_l1407_140781

theorem badminton_probability (p : ℝ) (n : ℕ) : 
  p = 3/4 → n = 3 → 
  (1 - p)^n = 1/64 → 
  n.choose 1 * p * (1 - p)^(n-1) = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_badminton_probability_l1407_140781


namespace NUMINAMATH_CALUDE_simplify_fraction_l1407_140764

theorem simplify_fraction (a : ℝ) (h : a = 2) : 15 * a^5 / (75 * a^3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1407_140764


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1407_140734

-- Define the start time and one-third completion time
def start_time : ℕ := 7 * 60  -- 7:00 AM in minutes
def one_third_time : ℕ := 10 * 60 + 15  -- 10:15 AM in minutes

-- Define the time taken for one-third of the job
def one_third_duration : ℕ := one_third_time - start_time

-- Define the total duration of the job
def total_duration : ℕ := 3 * one_third_duration

-- Define the completion time
def completion_time : ℕ := start_time + total_duration

-- Theorem to prove
theorem doughnut_machine_completion_time :
  completion_time = 16 * 60 + 45  -- 4:45 PM in minutes
:= by sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1407_140734


namespace NUMINAMATH_CALUDE_double_and_square_reverse_digits_l1407_140759

/-- For any base greater than 2, doubling (base - 1) and squaring (base - 1) 
    result in numbers with the same digits in reverse order. -/
theorem double_and_square_reverse_digits (a : ℕ) (h : a > 2) :
  ∃ (d₁ d₂ : ℕ), d₁ < a ∧ d₂ < a ∧ 
  2 * (a - 1) = d₁ * a + d₂ ∧
  (a - 1)^2 = d₂ * a + d₁ :=
sorry

end NUMINAMATH_CALUDE_double_and_square_reverse_digits_l1407_140759


namespace NUMINAMATH_CALUDE_f_six_equals_one_half_l1407_140741

-- Define the function f
noncomputable def f : ℝ → ℝ := λ u => (u^2 - 8*u + 20) / 16

-- State the theorem
theorem f_six_equals_one_half :
  (∀ x : ℝ, f (4*x + 2) = x^2 - x + 1) → f 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_six_equals_one_half_l1407_140741


namespace NUMINAMATH_CALUDE_unique_function_on_rationals_l1407_140705

theorem unique_function_on_rationals
  (f : ℚ → ℝ)
  (h1 : ∀ x y : ℚ, f (x + y) - y * f x - x * f y = f x * f y - x - y + x * y)
  (h2 : ∀ x : ℚ, f x = 2 * f (x + 1) + 2 + x)
  (h3 : f 1 + 1 > 0) :
  ∀ x : ℚ, f x = -x / 2 := by sorry

end NUMINAMATH_CALUDE_unique_function_on_rationals_l1407_140705


namespace NUMINAMATH_CALUDE_x_range_l1407_140707

theorem x_range (x y : ℝ) (h : x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)) :
  x ∈ Set.Icc 0 20 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1407_140707


namespace NUMINAMATH_CALUDE_h_monotone_increasing_l1407_140735

/-- Given a real constant a and a function f(x) = x^2 - 2ax + a, 
    we define h(x) = f(x) / x and prove that h(x) is monotonically 
    increasing on [1, +∞) when a < 1. -/
theorem h_monotone_increasing (a : ℝ) (ha : a < 1) :
  ∀ x : ℝ, x ≥ 1 → (
    let f := fun x => x^2 - 2*a*x + a
    let h := fun x => f x / x
    (deriv h) x > 0
  ) := by
  sorry

end NUMINAMATH_CALUDE_h_monotone_increasing_l1407_140735


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1407_140712

/-- Given that 8y varies inversely as the cube of x, and y = 25 when x = 2,
    prove that y = 25/8 when x = 4. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, 8 * y x = k / x^3) →  -- 8y varies inversely as the cube of x
  y 2 = 25 →                 -- y = 25 when x = 2
  y 4 = 25 / 8 :=             -- y = 25/8 when x = 4
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1407_140712


namespace NUMINAMATH_CALUDE_cara_age_l1407_140784

/-- Given the age relationships in Cara's family, prove Cara's age --/
theorem cara_age :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = mom_age - 20 →
    mom_age = grandma_age - 15 →
    grandma_age = 75 →
    cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_l1407_140784


namespace NUMINAMATH_CALUDE_base_ratio_l1407_140737

/-- An isosceles trapezoid with bases a and b (a > b) and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_gt_b : a > b
  h_gt_zero : h > 0

/-- The property that the height divides the larger base in ratio 1:3 -/
def height_divides_base (t : IsoscelesTrapezoid) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (3 * x = t.a - x)

/-- The theorem stating the ratio of bases -/
theorem base_ratio (t : IsoscelesTrapezoid) 
  (h : height_divides_base t) : t.a / t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_ratio_l1407_140737


namespace NUMINAMATH_CALUDE_windows_installed_correct_l1407_140787

/-- Calculates the number of windows already installed given the total number of windows,
    the time to install each window, and the time left to install remaining windows. -/
def windows_installed (total_windows : ℕ) (time_per_window : ℕ) (time_left : ℕ) : ℕ :=
  total_windows - (time_left / time_per_window)

/-- Proves that the number of windows already installed is correct for the given problem. -/
theorem windows_installed_correct :
  windows_installed 14 4 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_windows_installed_correct_l1407_140787


namespace NUMINAMATH_CALUDE_smallest_covering_l1407_140748

/-- A rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A covering of rectangles -/
structure Covering where
  target : Rectangle
  tiles : List Rectangle

/-- Whether a covering is valid (complete and non-overlapping) -/
def is_valid_covering (c : Covering) : Prop :=
  (area c.target = (c.tiles.map area).sum) ∧
  (∀ r ∈ c.tiles, r.length = 3 ∧ r.width = 4)

/-- The main theorem -/
theorem smallest_covering :
  ∃ (c : Covering),
    is_valid_covering c ∧
    c.tiles.length = 2 ∧
    (∀ (c' : Covering), is_valid_covering c' → c'.tiles.length ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_l1407_140748


namespace NUMINAMATH_CALUDE_juice_remaining_l1407_140731

theorem juice_remaining (initial_amount : ℚ) (given_amount : ℚ) (result : ℚ) : 
  initial_amount = 5 → given_amount = 18 / 4 → result = initial_amount - given_amount → result = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_juice_remaining_l1407_140731


namespace NUMINAMATH_CALUDE_isoscelesTriangles29Count_l1407_140793

/-- An isosceles triangle with integer side lengths and perimeter 29 -/
structure IsoscelesTriangle29 where
  base : ℕ
  side : ℕ
  isIsosceles : side * 2 + base = 29
  isTriangle : base < side + side

/-- The count of valid isosceles triangles with perimeter 29 -/
def countIsoscelesTriangles29 : ℕ := sorry

/-- Theorem stating that there are exactly 5 isosceles triangles with integer side lengths and perimeter 29 -/
theorem isoscelesTriangles29Count : countIsoscelesTriangles29 = 5 := by sorry

end NUMINAMATH_CALUDE_isoscelesTriangles29Count_l1407_140793


namespace NUMINAMATH_CALUDE_restaurant_students_l1407_140754

theorem restaurant_students (burger_count : ℕ) (hotdog_count : ℕ) :
  burger_count = 30 →
  burger_count = 2 * hotdog_count →
  burger_count + hotdog_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_students_l1407_140754


namespace NUMINAMATH_CALUDE_bicycle_selling_price_l1407_140709

/-- Calculates the final selling price of a bicycle given the initial cost and profit percentages -/
theorem bicycle_selling_price (initial_cost : ℝ) (profit_a profit_b : ℝ) :
  initial_cost = 120 ∧ profit_a = 50 ∧ profit_b = 25 →
  initial_cost * (1 + profit_a / 100) * (1 + profit_b / 100) = 225 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_selling_price_l1407_140709


namespace NUMINAMATH_CALUDE_suraj_innings_l1407_140743

/-- Represents the cricket problem for Suraj's innings --/
def cricket_problem (n : ℕ) : Prop :=
  let A : ℚ := 10  -- Initial average (derived from the new average minus the increase)
  let new_average : ℚ := 16  -- New average after the last innings
  let runs_increase : ℚ := 6  -- Increase in average
  let last_innings_runs : ℕ := 112  -- Runs scored in the last innings
  
  -- The equation representing the new average
  (n * A + last_innings_runs) / (n + 1) = new_average ∧
  -- The equation representing the increase in average
  new_average = A + runs_increase

/-- Theorem stating that the number of innings before the last one is 16 --/
theorem suraj_innings : cricket_problem 16 := by sorry

end NUMINAMATH_CALUDE_suraj_innings_l1407_140743


namespace NUMINAMATH_CALUDE_ball_return_to_start_l1407_140740

def ball_throw (n : ℕ) : ℕ → ℕ := λ x => (x + 3) % n

theorem ball_return_to_start :
  ∀ (start : ℕ), start < 13 →
  ∃ (k : ℕ), k > 0 ∧ (Nat.iterate (ball_throw 13) k start) = start ∧
  k = 13 :=
sorry

end NUMINAMATH_CALUDE_ball_return_to_start_l1407_140740


namespace NUMINAMATH_CALUDE_rectangle_equal_diagonals_l1407_140790

-- Define a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define equal diagonals
def equal_diagonals (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem rectangle_equal_diagonals (A B C D : Point) :
  is_rectangle A B C D → equal_diagonals A B C D := by sorry

end NUMINAMATH_CALUDE_rectangle_equal_diagonals_l1407_140790


namespace NUMINAMATH_CALUDE_value_of_expression_l1407_140718

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1407_140718


namespace NUMINAMATH_CALUDE_triangle_problem_l1407_140720

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  -- Given conditions
  (Real.cos (B + C)) / (Real.cos C) = a / (2 * b + c) →
  b = 1 →
  Real.cos C = 2 * Real.sqrt 7 / 7 →
  -- Conclusions
  A = 2 * π / 3 ∧ a = Real.sqrt 7 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1407_140720


namespace NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l1407_140778

-- Define a function to check if a number is a three-digit integer
def isThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function to check if digits are distinct
def hasDistinctDigits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.length = digits.toFinset.card

-- Define a function to check if digits form a geometric sequence
def formsGeometricSequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

-- State the theorem
theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigitInteger n ∧ hasDistinctDigits n ∧ formsGeometricSequence n →
  124 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l1407_140778


namespace NUMINAMATH_CALUDE_total_distance_flown_l1407_140719

/-- The speed of an eagle in miles per hour -/
def eagle_speed : ℝ := 15

/-- The speed of a falcon in miles per hour -/
def falcon_speed : ℝ := 46

/-- The speed of a pelican in miles per hour -/
def pelican_speed : ℝ := 33

/-- The speed of a hummingbird in miles per hour -/
def hummingbird_speed : ℝ := 30

/-- The time the birds flew in hours -/
def flight_time : ℝ := 2

/-- Theorem stating that the total distance flown by all birds in 2 hours is 248 miles -/
theorem total_distance_flown : 
  eagle_speed * flight_time + 
  falcon_speed * flight_time + 
  pelican_speed * flight_time + 
  hummingbird_speed * flight_time = 248 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_flown_l1407_140719


namespace NUMINAMATH_CALUDE_second_quadrant_condition_l1407_140723

/-- Given a complex number z = i(i-a) where a is real, if z corresponds to a point in the second 
    quadrant of the complex plane, then a < 0. -/
theorem second_quadrant_condition (a : ℝ) : 
  let z : ℂ := Complex.I * (Complex.I - a)
  (z.re < 0 ∧ z.im > 0) → a < 0 := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_condition_l1407_140723


namespace NUMINAMATH_CALUDE_tire_usage_proof_l1407_140766

-- Define the number of tires
def total_tires : ℕ := 6

-- Define the total miles traveled by the car
def total_miles : ℕ := 45000

-- Define the number of tires used at any given time
def tires_in_use : ℕ := 4

-- Define the function to calculate miles per tire
def miles_per_tire (total_tires : ℕ) (total_miles : ℕ) (tires_in_use : ℕ) : ℕ :=
  (total_miles * tires_in_use) / total_tires

-- Theorem statement
theorem tire_usage_proof :
  miles_per_tire total_tires total_miles tires_in_use = 30000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_proof_l1407_140766


namespace NUMINAMATH_CALUDE_fourteen_binary_l1407_140795

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  sorry

theorem fourteen_binary : binary_repr 14 = [true, true, true, false] := by
  sorry

end NUMINAMATH_CALUDE_fourteen_binary_l1407_140795


namespace NUMINAMATH_CALUDE_original_number_from_percentage_l1407_140724

/-- Given a percentage value, returns the original number -/
def percentage_to_number (percentage : Float) : Float :=
  percentage / 100

/-- Theorem: The original number corresponding to 501.99999999999994% is 5.0199999999999994 -/
theorem original_number_from_percentage :
  percentage_to_number 501.99999999999994 = 5.0199999999999994 := by
  sorry

end NUMINAMATH_CALUDE_original_number_from_percentage_l1407_140724


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l1407_140780

theorem greatest_integer_with_gcf_two (n : ℕ) : n < 100 → Nat.gcd n 12 = 2 → n ≤ 98 := by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ Nat.gcd 98 12 = 2 := by
  sorry

theorem ninety_eight_is_greatest : 
  ∀ (m : ℕ), m < 100 → Nat.gcd m 12 = 2 → m ≤ 98 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l1407_140780


namespace NUMINAMATH_CALUDE_base5_413_equals_108_l1407_140758

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5^1 + c * 5^0

/-- The base 5 number 413₅ is equal to 108 in base 10 --/
theorem base5_413_equals_108 : base5ToBase10 4 1 3 = 108 := by sorry

end NUMINAMATH_CALUDE_base5_413_equals_108_l1407_140758


namespace NUMINAMATH_CALUDE_rabbit_chicken_problem_l1407_140722

theorem rabbit_chicken_problem (total : ℕ) (rabbits chickens : ℕ → ℕ) :
  total = 40 →
  (∀ x : ℕ, rabbits x + chickens x = total) →
  (∀ x : ℕ, 4 * rabbits x = 10 * 2 * chickens x - 8) →
  (∃ x : ℕ, rabbits x = 33) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_chicken_problem_l1407_140722


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1407_140773

theorem polynomial_division_remainder (x : ℝ) : 
  x^1000 % ((x^2 + 1) * (x + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1407_140773


namespace NUMINAMATH_CALUDE_rational_numbers_classification_l1407_140775

theorem rational_numbers_classification (x : ℚ) : 
  ¬(∀ x : ℚ, x > 0 ∨ x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_classification_l1407_140775


namespace NUMINAMATH_CALUDE_frog_food_theorem_l1407_140789

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty has caught -/
def flies_caught : ℕ := 10

/-- The number of additional flies Betty needs for a week's food -/
def additional_flies_needed : ℕ := 4

theorem frog_food_theorem :
  flies_per_day * days_in_week - flies_caught = additional_flies_needed :=
by sorry

end NUMINAMATH_CALUDE_frog_food_theorem_l1407_140789


namespace NUMINAMATH_CALUDE_log_25_between_consecutive_integers_l1407_140742

theorem log_25_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_25_between_consecutive_integers_l1407_140742


namespace NUMINAMATH_CALUDE_length_of_A_l1407_140763

def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 7)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    (p.1 + t * (r.1 - p.1) = q.1) ∧
    (p.2 + t * (r.2 - p.2) = q.2)

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, 
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect_at A A' C ∧
    intersect_at B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l1407_140763


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1407_140771

theorem sqrt_product_equals_sqrt_of_product :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1407_140771


namespace NUMINAMATH_CALUDE_line_properties_l1407_140736

structure Line where
  slope : ℝ
  inclination : ℝ

def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem line_properties (l1 l2 : Line) 
  (h_non_overlapping : l1 ≠ l2) : 
  (l1.slope = l2.slope → parallel l1 l2) ∧ 
  (parallel l1 l2 → l1.inclination = l2.inclination) ∧
  (l1.inclination = l2.inclination → parallel l1 l2) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l1407_140736


namespace NUMINAMATH_CALUDE_discounted_price_is_nine_l1407_140792

/-- The final price after applying a discount --/
def final_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Theorem: The final price of a $10 item after a 10% discount is $9 --/
theorem discounted_price_is_nine :
  final_price 10 0.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_is_nine_l1407_140792


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1407_140701

theorem complex_fraction_simplification :
  let z : ℂ := (10 : ℂ) - 8 * Complex.I
  let w : ℂ := (3 : ℂ) + 4 * Complex.I
  z / w = -(2 : ℂ) / 25 - (64 : ℂ) / 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1407_140701


namespace NUMINAMATH_CALUDE_triangle_perimeter_lower_bound_l1407_140772

theorem triangle_perimeter_lower_bound 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h : ℝ) -- Height on side BC
  (ha : a = 1) -- Side a equals 1
  (hh : h = Real.tan A) -- Height equals tan A
  (hA : 0 < A ∧ A < Real.pi / 2) -- A is in the range (0, π/2)
  (hS : (1/2) * a * h = (1/2) * b * c * Real.sin A) -- Area formula
  (hC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) -- Cosine rule
  : a + b + c > Real.sqrt 5 + 1 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_lower_bound_l1407_140772


namespace NUMINAMATH_CALUDE_count_even_positive_factors_l1407_140751

/-- The number of even positive factors of n, where n = 2^4 * 3^2 * 5^2 * 7 -/
def evenPositiveFactors (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of even positive factors of n is 72 -/
theorem count_even_positive_factors :
  ∃ n : ℕ, n = 2^4 * 3^2 * 5^2 * 7 ∧ evenPositiveFactors n = 72 :=
sorry

end NUMINAMATH_CALUDE_count_even_positive_factors_l1407_140751


namespace NUMINAMATH_CALUDE_peanuts_equation_initial_peanuts_count_l1407_140711

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 8

/-- The total number of peanuts after Mary adds more -/
def total_peanuts : ℕ := 12

/-- Theorem stating that the initial number of peanuts plus the added peanuts equals the total peanuts -/
theorem peanuts_equation : initial_peanuts + peanuts_added = total_peanuts := by sorry

/-- Theorem proving that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by sorry

end NUMINAMATH_CALUDE_peanuts_equation_initial_peanuts_count_l1407_140711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1407_140747

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  h1 : a 7 = 1  -- 7th term is 1
  h2 : S 4 = -32  -- Sum of first 4 terms is -32
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Constant difference property

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 13) ∧
  (∀ n : ℕ, seq.S n = (n - 6)^2 - 36) ∧
  (∀ n : ℕ, seq.S n ≥ -36) ∧
  (∃ n : ℕ, seq.S n = -36) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1407_140747


namespace NUMINAMATH_CALUDE_yoyo_cost_l1407_140706

/-- Given that Mrs. Hilt bought a yoyo and a whistle for a total of 38 cents,
    and the whistle costs 14 cents, prove that the yoyo costs 24 cents. -/
theorem yoyo_cost (total : ℕ) (whistle : ℕ) (yoyo : ℕ)
    (h1 : total = 38)
    (h2 : whistle = 14)
    (h3 : total = whistle + yoyo) :
  yoyo = 24 := by
  sorry

end NUMINAMATH_CALUDE_yoyo_cost_l1407_140706


namespace NUMINAMATH_CALUDE_planes_parallel_if_line_perpendicular_to_both_l1407_140761

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_line_perpendicular_to_both 
  (a : Line) (α β : Plane) (h1 : α ≠ β) :
  perp a α → perp a β → para α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_line_perpendicular_to_both_l1407_140761


namespace NUMINAMATH_CALUDE_valid_choices_count_l1407_140794

/-- The number of elements in the list -/
def n : ℕ := 2016

/-- The number of elements to be shuffled -/
def m : ℕ := 2014

/-- Function to calculate the number of valid ways to choose a and b -/
def count_valid_choices : ℕ := sorry

/-- Theorem stating that the number of valid choices is equal to 508536 -/
theorem valid_choices_count : count_valid_choices = 508536 := by sorry

end NUMINAMATH_CALUDE_valid_choices_count_l1407_140794


namespace NUMINAMATH_CALUDE_hikers_room_arrangements_l1407_140733

theorem hikers_room_arrangements (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_hikers_room_arrangements_l1407_140733


namespace NUMINAMATH_CALUDE_linear_function_theorem_l1407_140756

/-- A linear function f(x) = ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) : ℝ := a

theorem linear_function_theorem (a b : ℝ) :
  f a b 1 = 2 ∧ f_derivative a = 2 → f a b 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l1407_140756


namespace NUMINAMATH_CALUDE_absent_workers_l1407_140732

theorem absent_workers (total_workers : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_workers = 30)
  (h2 : original_days = 10)
  (h3 : actual_days = 12)
  (h4 : total_workers * original_days = (total_workers - absent) * actual_days) :
  absent = 5 :=
by sorry

end NUMINAMATH_CALUDE_absent_workers_l1407_140732


namespace NUMINAMATH_CALUDE_colonization_combinations_count_l1407_140725

/-- The number of habitable planets -/
def total_planets : ℕ := 15

/-- The number of Earth-like planets -/
def earth_like : ℕ := 6

/-- The number of Mars-like planets -/
def mars_like : ℕ := 9

/-- The number of colonization units required for an Earth-like planet -/
def earth_units : ℕ := 2

/-- The number of colonization units required for a Mars-like planet -/
def mars_units : ℕ := 1

/-- The total number of colonization units available -/
def total_units : ℕ := 16

/-- The function to calculate the number of combinations -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating the number of colonization combinations -/
theorem colonization_combinations_count : colonization_combinations = 765 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_count_l1407_140725


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1407_140717

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -24
  let c : ℝ := 98
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1407_140717


namespace NUMINAMATH_CALUDE_two_distinct_roots_root_three_implies_sum_l1407_140791

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 + 2*m*x + m^2 - 2 = 0

-- Part 1: The equation always has two distinct real roots
theorem two_distinct_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Part 2: If 3 is a root, then 2m^2 + 12m + 2043 = 2029
theorem root_three_implies_sum (m : ℝ) :
  quadratic_equation m 3 → 2*m^2 + 12*m + 2043 = 2029 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_root_three_implies_sum_l1407_140791


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l1407_140746

theorem unique_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 2) ∧ x^2 - a*x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l1407_140746


namespace NUMINAMATH_CALUDE_range_of_a_l1407_140782

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1407_140782


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1407_140755

/-- Given real numbers x and y satisfying |x-4| + √(y-10) = 0,
    prove that the perimeter of an isosceles triangle with side lengths x, y, and y is 24. -/
theorem isosceles_triangle_perimeter (x y : ℝ) 
  (h : |x - 4| + Real.sqrt (y - 10) = 0) : 
  x + y + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1407_140755


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1407_140785

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 7

theorem businessmen_neither_coffee_nor_tea :
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1407_140785


namespace NUMINAMATH_CALUDE_girl_sums_equal_iff_n_odd_l1407_140745

/-- Represents the sum of a girl's card number and the numbers of adjacent boys' cards -/
def girlSum (n : ℕ) (i : ℕ) : ℕ :=
  (n + i) + (i % n + 1) + ((i + 1) % n + 1)

/-- Theorem stating that all girl sums are equal if and only if n is odd -/
theorem girl_sums_equal_iff_n_odd (n : ℕ) (h : n ≥ 3) :
  (∀ i j, i < n → j < n → girlSum n i = girlSum n j) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_girl_sums_equal_iff_n_odd_l1407_140745


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_7_power_difference_l1407_140726

theorem sum_distinct_prime_factors_of_7_power_difference : 
  (Finset.sum (Finset.filter (Nat.Prime) (Finset.range ((7^7 - 7^4).factors.toFinset.card + 1)))
    (λ p => if p ∈ (7^7 - 7^4).factors.toFinset then p else 0)) = 31 := by sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_7_power_difference_l1407_140726


namespace NUMINAMATH_CALUDE_triangle_side_length_l1407_140728

/-- Given a triangle DEF with side lengths and median as specified, prove that DF = √130 -/
theorem triangle_side_length (DE EF DN : ℝ) (h1 : DE = 7) (h2 : EF = 9) (h3 : DN = 9/2) : 
  ∃ (DF : ℝ), DF = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1407_140728


namespace NUMINAMATH_CALUDE_unique_D_value_l1407_140702

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Definition of our addition problem -/
def AdditionProblem (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  1000 * A.val + 100 * A.val + 10 * C.val + B.val +
  1000 * B.val + 100 * C.val + 10 * B.val + D.val =
  1000 * B.val + 100 * D.val + 10 * A.val + B.val

theorem unique_D_value (A B C D : Digit) :
  AdditionProblem A B C D → D.val = 0 ∧ ∀ E : Digit, AdditionProblem A B C E → E = D :=
by sorry

end NUMINAMATH_CALUDE_unique_D_value_l1407_140702


namespace NUMINAMATH_CALUDE_scientific_notation_of_ten_billion_thirty_million_l1407_140727

theorem scientific_notation_of_ten_billion_thirty_million :
  (10030000000 : ℝ) = 1.003 * (10 : ℝ)^10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_ten_billion_thirty_million_l1407_140727


namespace NUMINAMATH_CALUDE_circle_diameter_l1407_140776

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l1407_140776


namespace NUMINAMATH_CALUDE_points_and_lines_l1407_140749

theorem points_and_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≤ 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_points_and_lines_l1407_140749


namespace NUMINAMATH_CALUDE_division_theorem_l1407_140713

theorem division_theorem (dividend divisor remainder quotient : ℕ) :
  dividend = 176 →
  divisor = 14 →
  remainder = 8 →
  quotient = 12 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_l1407_140713


namespace NUMINAMATH_CALUDE_peanuts_in_box_l1407_140768

/-- Calculate the final number of peanuts in a box after removing and adding some. -/
def final_peanuts (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that for the given values, the final number of peanuts is 13. -/
theorem peanuts_in_box : final_peanuts 4 3 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l1407_140768


namespace NUMINAMATH_CALUDE_range_of_f_l1407_140716

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1407_140716


namespace NUMINAMATH_CALUDE_no_coin_exchange_solution_l1407_140760

theorem no_coin_exchange_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 500 ∧ 
  36 * x + 6 * y + z = 3564 ∧ 
  x ≤ 99 := by
sorry

end NUMINAMATH_CALUDE_no_coin_exchange_solution_l1407_140760


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l1407_140710

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / x + 2 / y = 2 → a + b ≤ x + y ∧ 
  (a + b = (3 + 2 * Real.sqrt 2) / 2 ↔ a + b = x + y) :=
sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l1407_140710


namespace NUMINAMATH_CALUDE_smallest_quotient_two_digit_numbers_l1407_140798

theorem smallest_quotient_two_digit_numbers :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    a ≠ b →
    (10 * a + b : ℚ) / (a + b) ≥ 1.9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_quotient_two_digit_numbers_l1407_140798


namespace NUMINAMATH_CALUDE_pencil_rows_l1407_140796

/-- Given a total number of pencils and the number of pencils per row,
    calculate the number of complete rows that can be formed. -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem stating that 30 pencils arranged in rows of 5 will form 6 complete rows -/
theorem pencil_rows : calculate_rows 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_rows_l1407_140796


namespace NUMINAMATH_CALUDE_erased_line_length_l1407_140715

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem erased_line_length : 
  let initial_length_m : ℝ := 1
  let initial_length_cm : ℝ := initial_length_m * 100
  let erased_length_cm : ℝ := 33
  let final_length_cm : ℝ := initial_length_cm - erased_length_cm
  final_length_cm = 67 := by sorry

end NUMINAMATH_CALUDE_erased_line_length_l1407_140715


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1407_140786

/-- Proves that a parallelogram with area 288 sq m and altitude twice the base has a base length of 12 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 288 →
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1407_140786


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_defined_l1407_140799

theorem sqrt_x_minus_3_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_defined_l1407_140799


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1407_140769

/-- Given a quadratic equation (m-3)x^2 + 4x + 1 = 0 with real solutions,
    the range of values for m is m ≤ 7 and m ≠ 3 -/
theorem quadratic_equation_range (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1407_140769


namespace NUMINAMATH_CALUDE_halfway_fraction_l1407_140703

theorem halfway_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d > 1 / 2 ∧
  (n : ℚ) / d = (3 / 4 + 5 / 7) / 2 ∧
  n = 41 ∧ d = 56 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1407_140703


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l1407_140750

/-- The number of Popsicles consumed in a given time period -/
def popsicles_consumed (rate_minutes : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / rate_minutes

theorem megan_popsicle_consumption :
  popsicles_consumed 20 340 = 17 := by
  sorry

#eval popsicles_consumed 20 340

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l1407_140750


namespace NUMINAMATH_CALUDE_farm_area_is_1200_l1407_140777

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Calculates the total length of fencing required -/
def total_fencing_length (farm : RectangularFarm) : ℝ :=
  farm.short_side + farm.long_side + farm.diagonal

/-- The main theorem: If a rectangular farm satisfies the given conditions, its area is 1200 square meters -/
theorem farm_area_is_1200 (farm : RectangularFarm) 
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 13)
    (h3 : farm.total_fencing_cost = 1560)
    (h4 : farm.total_fencing_cost = total_fencing_length farm * farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end NUMINAMATH_CALUDE_farm_area_is_1200_l1407_140777


namespace NUMINAMATH_CALUDE_committee_formation_count_l1407_140765

/-- Represents a department in the science division -/
inductive Department : Type
| Biology : Department
| Physics : Department
| Chemistry : Department
| Mathematics : Department

/-- Represents the gender of a professor -/
inductive Gender : Type
| Male : Gender
| Female : Gender

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of departments -/
def total_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 4

/-- The number of departments that should contribute exactly 2 professors -/
def depts_with_two_profs : Nat := 2

/-- The minimum number of professors required from the Mathematics department -/
def min_math_profs : Nat := 2

/-- Calculates the number of ways to form the committee -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of ways to form the committee is 1944 -/
theorem committee_formation_count :
  count_committee_formations = 1944 :=
sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1407_140765


namespace NUMINAMATH_CALUDE_stick_swap_triangle_formation_l1407_140767

/-- Represents a set of three stick lengths -/
structure StickSet where
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  sum_is_one : s₁ + s₂ + s₃ = 1
  all_positive : 0 < s₁ ∧ 0 < s₂ ∧ 0 < s₃

/-- Checks if a triangle can be formed from the given stick lengths -/
def can_form_triangle (s : StickSet) : Prop :=
  s.s₁ < s.s₂ + s.s₃ ∧ s.s₂ < s.s₁ + s.s₃ ∧ s.s₃ < s.s₁ + s.s₂

theorem stick_swap_triangle_formation 
  (v_initial w_initial : StickSet)
  (v_can_form_initial : can_form_triangle v_initial)
  (w_can_form_initial : can_form_triangle w_initial)
  (v_final w_final : StickSet)
  (swap_occurred : ∃ (i j : Fin 3), 
    v_final.s₁ + v_final.s₂ + v_final.s₃ + w_final.s₁ + w_final.s₂ + w_final.s₃ = 
    v_initial.s₁ + v_initial.s₂ + v_initial.s₃ + w_initial.s₁ + w_initial.s₂ + w_initial.s₃)
  (v_cannot_form_final : ¬can_form_triangle v_final) :
  can_form_triangle w_final :=
sorry

end NUMINAMATH_CALUDE_stick_swap_triangle_formation_l1407_140767


namespace NUMINAMATH_CALUDE_tulips_to_remaining_ratio_l1407_140714

def total_flowers : ℕ := 12
def daisies : ℕ := 2
def sunflowers : ℕ := 4

def tulips : ℕ := total_flowers - (daisies + sunflowers)
def remaining_flowers : ℕ := tulips + sunflowers

theorem tulips_to_remaining_ratio :
  (tulips : ℚ) / (remaining_flowers : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tulips_to_remaining_ratio_l1407_140714


namespace NUMINAMATH_CALUDE_merchant_profit_l1407_140783

theorem merchant_profit (cost : ℝ) (cost_positive : cost > 0) : 
  let marked_price := cost * 1.2
  let discounted_price := marked_price * 0.9
  let profit := discounted_price - cost
  let profit_percentage := (profit / cost) * 100
  profit_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_l1407_140783


namespace NUMINAMATH_CALUDE_black_raisins_amount_l1407_140739

-- Define the variables
def yellow_raisins : ℝ := 0.3
def total_raisins : ℝ := 0.7

-- Define the theorem
theorem black_raisins_amount :
  total_raisins - yellow_raisins = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_black_raisins_amount_l1407_140739


namespace NUMINAMATH_CALUDE_park_tree_count_l1407_140797

/-- The number of dogwood trees in the park after 5 days of planting and one uprooting event -/
def final_tree_count (initial : ℕ) (day1 : ℕ) (day5 : ℕ) : ℕ :=
  let day2 := day1 / 2
  let day3 := day2 * 4
  let day4 := 5  -- Trees replaced due to uprooting
  initial + day1 + day2 + day3 + day4 + day5

/-- Theorem stating the final number of trees in the park -/
theorem park_tree_count : final_tree_count 39 24 15 = 143 := by
  sorry

end NUMINAMATH_CALUDE_park_tree_count_l1407_140797


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1407_140774

/-- Proves that a rectangular plot with length to breadth ratio 7:5 and perimeter 288 meters has an area of 5040 square meters. -/
theorem rectangular_plot_area (length width : ℝ) : 
  (length / width = 7 / 5) →  -- ratio condition
  (2 * (length + width) = 288) →  -- perimeter condition
  (length * width = 5040) :=  -- area to prove
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1407_140774


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l1407_140770

theorem divisibility_by_nine (A : ℕ) (h : A < 10) : 
  (7000 + 200 + 10 * A + 4) % 9 = 0 ↔ A = 5 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l1407_140770


namespace NUMINAMATH_CALUDE_flamingo_tail_feathers_l1407_140744

/-- The number of tail feathers per flamingo given the conditions for making feather boas --/
theorem flamingo_tail_feathers 
  (num_boas : ℕ) 
  (feathers_per_boa : ℕ) 
  (num_flamingoes : ℕ) 
  (safe_pluck_percentage : ℚ) : ℕ :=
  sorry

#check flamingo_tail_feathers 12 200 480 (1/4) = 20

end NUMINAMATH_CALUDE_flamingo_tail_feathers_l1407_140744


namespace NUMINAMATH_CALUDE_youngest_son_cotton_correct_l1407_140757

/-- The amount of cotton for the youngest son in the "Dividing Cotton among Eight Sons" problem -/
def youngest_son_cotton : ℕ := 184

/-- The total amount of cotton to be divided -/
def total_cotton : ℕ := 996

/-- The number of sons -/
def num_sons : ℕ := 8

/-- The difference in cotton amount between each son -/
def cotton_difference : ℕ := 17

/-- Theorem stating that the youngest son's cotton amount is correct given the problem conditions -/
theorem youngest_son_cotton_correct :
  youngest_son_cotton * num_sons + (num_sons * (num_sons - 1) / 2) * cotton_difference = total_cotton :=
by sorry

end NUMINAMATH_CALUDE_youngest_son_cotton_correct_l1407_140757


namespace NUMINAMATH_CALUDE_part_one_part_two_part_two_range_l1407_140779

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

-- Part I
theorem part_one :
  let a := 2
  {x : ℝ | f a x ≤ -1/2} = {x : ℝ | x ≥ 11/4} := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≥ a) → a ∈ Set.Iic (3/2) := by sorry

-- Additional theorem to show the full range of a
theorem part_two_range :
  {a : ℝ | ∃ x : ℝ, f a x ≥ a} = Set.Iic (3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_two_range_l1407_140779


namespace NUMINAMATH_CALUDE_x_positive_iff_reciprocal_positive_l1407_140788

theorem x_positive_iff_reciprocal_positive (x : ℝ) :
  x > 0 ↔ 1 / x > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_positive_iff_reciprocal_positive_l1407_140788


namespace NUMINAMATH_CALUDE_chocolate_bar_problem_l1407_140700

/-- The number of chocolate bars Min bought -/
def min_bars : ℕ := 67

/-- The initial number of chocolate bars in the store -/
def initial_bars : ℕ := 376

/-- The number of chocolate bars Max bought -/
def max_bars : ℕ := min_bars + 41

/-- The number of chocolate bars remaining in the store after purchases -/
def remaining_bars : ℕ := initial_bars - min_bars - max_bars

theorem chocolate_bar_problem :
  min_bars = 67 ∧
  initial_bars = 376 ∧
  max_bars = min_bars + 41 ∧
  remaining_bars = 3 * min_bars :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_problem_l1407_140700


namespace NUMINAMATH_CALUDE_colored_balls_theorem_l1407_140752

/-- Represents a box of colored balls -/
structure ColoredBalls where
  total : ℕ
  colors : ℕ
  min_same_color : ℕ → ℕ

/-- The problem statement -/
theorem colored_balls_theorem (box : ColoredBalls) 
  (h_total : box.total = 100)
  (h_colors : box.colors = 3)
  (h_min_same_color : box.min_same_color 26 ≥ 10) :
  box.min_same_color 66 ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_theorem_l1407_140752


namespace NUMINAMATH_CALUDE_not_perfect_square_l1407_140762

theorem not_perfect_square (t : ℤ) : ¬ ∃ k : ℤ, 7 * t + 3 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1407_140762


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l1407_140704

/-- Calculates the fraction of water remaining in a radiator after a given number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ numReplacements

theorem radiator_water_fraction :
  let initialVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numReplacements : ℕ := 5
  waterFraction initialVolume replacementVolume numReplacements = 1024 / 3125 := by
  sorry

#eval waterFraction 25 5 5

end NUMINAMATH_CALUDE_radiator_water_fraction_l1407_140704
