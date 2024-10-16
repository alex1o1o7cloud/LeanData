import Mathlib

namespace NUMINAMATH_CALUDE_periodic_function_zeros_l775_77506

/-- A function f: ℝ → ℝ that is periodic with period 5 and defined as x^2 - 2^x on (-1, 4] -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (x - 5)) ∧ 
  (∀ x, -1 < x ∧ x ≤ 4 → f x = x^2 - 2^x)

/-- The number of zeros of f on an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem periodic_function_zeros (f : ℝ → ℝ) (h : periodic_function f) :
  num_zeros f 0 2013 = 1207 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_zeros_l775_77506


namespace NUMINAMATH_CALUDE_coefficient_of_P_equals_30_l775_77562

/-- The generating function P as described in the problem -/
def P (x : Fin 6 → ℚ) : ℚ :=
  (1 / 24) * (
    (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^6 +
    6 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^4 + x 1^4 + x 2^4 + x 3^4 + x 4^4 + x 5^4) +
    3 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^2 +
    6 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^3 +
    8 * (x 0^3 + x 1^3 + x 2^3 + x 3^3 + x 4^3 + x 5^3)^2
  )

/-- The coefficient of x₁x₂x₃x₄x₅x₆ in the generating function P -/
def coefficient_x1x2x3x4x5x6 (P : (Fin 6 → ℚ) → ℚ) : ℚ :=
  sorry  -- Definition of how to extract the coefficient

theorem coefficient_of_P_equals_30 :
  coefficient_x1x2x3x4x5x6 P = 30 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_P_equals_30_l775_77562


namespace NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l775_77567

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The price of Dawn's lemonade in cents -/
def dawn_price : ℕ := 28

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- Theorem: Bea earned 26 cents more than Dawn -/
theorem bea_earned_more_than_dawn : 
  bea_price * bea_glasses - dawn_price * dawn_glasses = 26 := by
  sorry

end NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l775_77567


namespace NUMINAMATH_CALUDE_opposite_sides_line_constant_range_l775_77561

/-- Given two points on opposite sides of a line, prove the range of the constant term -/
theorem opposite_sides_line_constant_range :
  ∀ (a : ℝ),
  (((3 * 2 - 2 * 1 + a) * (3 * (-2) - 2 * 3 + a) < 0) ↔ (-4 < a ∧ a < 12)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_line_constant_range_l775_77561


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l775_77549

theorem largest_integer_satisfying_inequality :
  ∃ (n : ℕ), n > 0 ∧ n^200 < 5^300 ∧ ∀ (m : ℕ), m > n → m^200 ≥ 5^300 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l775_77549


namespace NUMINAMATH_CALUDE_average_pastry_sales_l775_77500

/-- Represents the daily sales of pastries over a week -/
def weeklySales : List Nat := [2, 3, 4, 5, 6, 7, 8]

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Calculates the average of a list of natural numbers -/
def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem average_pastry_sales : average weeklySales = 5 := by sorry

end NUMINAMATH_CALUDE_average_pastry_sales_l775_77500


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l775_77522

theorem meal_cost_calculation (initial_friends : ℕ) (additional_friends : ℕ) 
  (cost_decrease : ℚ) (total_cost : ℚ) : 
  initial_friends = 4 →
  additional_friends = 5 →
  cost_decrease = 6 →
  (total_cost / initial_friends.cast) - (total_cost / (initial_friends + additional_friends).cast) = cost_decrease →
  total_cost = 216/5 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l775_77522


namespace NUMINAMATH_CALUDE_time_2_to_7_is_50_l775_77523

/-- The time (in seconds) Fangfang takes to go from the 1st floor to the 5th floor -/
def time_1_to_5 : ℕ := 40

/-- The number of floors between the 1st and 5th floors -/
def floors_1_to_5 : ℕ := 5 - 1

/-- The number of floors between the 2nd and 7th floors -/
def floors_2_to_7 : ℕ := 7 - 2

/-- Theorem: The time Fangfang needs to go from the 2nd floor to the 7th floor is 50 seconds -/
theorem time_2_to_7_is_50 : 
  (time_1_to_5 / floors_1_to_5) * floors_2_to_7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_time_2_to_7_is_50_l775_77523


namespace NUMINAMATH_CALUDE_roses_in_vase_correct_l775_77598

/-- Given a total number of roses and the number of roses left,
    calculate the number of roses put in a vase. -/
def roses_in_vase (total : ℕ) (left : ℕ) : ℕ :=
  total - left

theorem roses_in_vase_correct (total : ℕ) (left : ℕ) 
  (h : left ≤ total) : 
  roses_in_vase total left = total - left :=
by
  sorry

#eval roses_in_vase 29 12  -- Should evaluate to 17

end NUMINAMATH_CALUDE_roses_in_vase_correct_l775_77598


namespace NUMINAMATH_CALUDE_max_clowns_proof_l775_77518

/-- The number of distinct colors available -/
def num_colors : ℕ := 12

/-- The minimum number of colors each clown must use -/
def min_colors_per_clown : ℕ := 5

/-- The maximum number of clowns that can use any particular color -/
def max_clowns_per_color : ℕ := 20

/-- The set of all possible color combinations for clowns -/
def color_combinations : Finset (Finset (Fin num_colors)) :=
  (Finset.powerset (Finset.univ : Finset (Fin num_colors))).filter (fun s => s.card ≥ min_colors_per_clown)

/-- The maximum number of clowns satisfying all conditions -/
def max_clowns : ℕ := num_colors * max_clowns_per_color

theorem max_clowns_proof :
  (∀ s : Finset (Fin num_colors), s ∈ color_combinations → s.card ≥ min_colors_per_clown) ∧
  (∀ c : Fin num_colors, (color_combinations.filter (fun s => c ∈ s)).card ≤ max_clowns_per_color) →
  color_combinations.card ≥ max_clowns ∧
  max_clowns = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_clowns_proof_l775_77518


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l775_77545

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l775_77545


namespace NUMINAMATH_CALUDE_time_puzzle_l775_77570

theorem time_puzzle : ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ 24 ∧ 
  (x / 4) + ((24 - x) / 2) = x ∧ 
  x = 9.6 := by
sorry

end NUMINAMATH_CALUDE_time_puzzle_l775_77570


namespace NUMINAMATH_CALUDE_factorial_square_root_square_l775_77542

-- Definition of factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_square_root_square :
  (((factorial 5 + 1) * factorial 4).sqrt ^ 2 : ℕ) = 2904 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_square_l775_77542


namespace NUMINAMATH_CALUDE_bike_shop_profit_l775_77544

/-- The profit calculation for Jim's bike shop -/
theorem bike_shop_profit (x : ℝ) 
  (h1 : x > 0) -- Charge for fixing bike tires is positive
  (h2 : 300 * x + 600 + 2000 - (300 * 5 + 100 + 4000) = 3000) -- Profit equation
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l775_77544


namespace NUMINAMATH_CALUDE_right_triangle_area_from_broken_stick_l775_77551

theorem right_triangle_area_from_broken_stick : ∀ a : ℝ,
  0 < a →
  a < 24 →
  a^2 + 24^2 = (48 - a)^2 →
  (1/2) * a * 24 = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_from_broken_stick_l775_77551


namespace NUMINAMATH_CALUDE_total_money_proof_l775_77543

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
sorry

end NUMINAMATH_CALUDE_total_money_proof_l775_77543


namespace NUMINAMATH_CALUDE_remaining_amount_is_14_90_l775_77558

-- Define the initial amount and item costs
def initial_amount : ℚ := 78
def kite_cost : ℚ := 8
def frisbee_cost : ℚ := 9
def roller_skates_cost : ℚ := 15
def roller_skates_discount : ℚ := 0.1
def lego_cost : ℚ := 25
def lego_coupon : ℚ := 5
def puzzle_cost : ℚ := 12
def puzzle_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def remaining_amount : ℚ :=
  initial_amount -
  (kite_cost +
   frisbee_cost +
   (roller_skates_cost * (1 - roller_skates_discount)) +
   (lego_cost - lego_coupon) +
   (puzzle_cost * (1 + puzzle_tax_rate)))

-- Theorem stating that the remaining amount is $14.90
theorem remaining_amount_is_14_90 :
  remaining_amount = 14.90 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_14_90_l775_77558


namespace NUMINAMATH_CALUDE_power_multiplication_l775_77560

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l775_77560


namespace NUMINAMATH_CALUDE_special_function_properties_l775_77563

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x < 0 → f x > 0)

/-- Main theorem encapsulating all parts of the problem -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ a x : ℝ, f (x^2) + 3 * f a > 3 * f x + f (a * x) ↔
    (a ≠ 0 ∧ ((a > 3 ∧ 3 < x ∧ x < a) ∨ (a < 3 ∧ a < x ∧ x < 3)))) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l775_77563


namespace NUMINAMATH_CALUDE_b_is_negative_l775_77575

def is_two_positive_two_negative (a b : ℝ) : Prop :=
  (((a + b > 0) ∧ (a - b > 0)) ∨ ((a + b > 0) ∧ (a * b > 0)) ∨ ((a + b > 0) ∧ (a / b > 0)) ∨
   ((a - b > 0) ∧ (a * b > 0)) ∨ ((a - b > 0) ∧ (a / b > 0)) ∨ ((a * b > 0) ∧ (a / b > 0))) ∧
  (((a + b < 0) ∧ (a - b < 0)) ∨ ((a + b < 0) ∧ (a * b < 0)) ∨ ((a + b < 0) ∧ (a / b < 0)) ∨
   ((a - b < 0) ∧ (a * b < 0)) ∨ ((a - b < 0) ∧ (a / b < 0)) ∨ ((a * b < 0) ∧ (a / b < 0)))

theorem b_is_negative (a b : ℝ) (h : b ≠ 0) (condition : is_two_positive_two_negative a b) : b < 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_negative_l775_77575


namespace NUMINAMATH_CALUDE_total_onions_grown_l775_77503

theorem total_onions_grown (sara_onions sally_onions fred_onions : ℕ) 
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_onions_grown_l775_77503


namespace NUMINAMATH_CALUDE_product_digits_l775_77577

def a : ℕ := 7123456789
def b : ℕ := 23567891234

theorem product_digits : (String.length (toString (a * b))) = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_l775_77577


namespace NUMINAMATH_CALUDE_seokjins_uncle_age_l775_77507

/-- Seokjin's uncle's age when Seokjin is 12 years old -/
def uncles_age (mothers_age_at_birth : ℕ) (age_difference : ℕ) : ℕ :=
  mothers_age_at_birth + 12 - age_difference

/-- Theorem stating that Seokjin's uncle's age is 41 when Seokjin is 12 -/
theorem seokjins_uncle_age :
  uncles_age 32 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_seokjins_uncle_age_l775_77507


namespace NUMINAMATH_CALUDE_quadratic_properties_l775_77587

/-- A quadratic function with vertex at (1, -4) and axis of symmetry at x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  (∀ x, f a b c x = a * (x - 1)^2 - 4) →
  (2 * a + b = 0) ∧
  (f a b c (-1) = 0 ∧ f a b c 3 = 0) ∧
  (∀ m, f a b c (m - 1) < f a b c m → m > 3/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l775_77587


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l775_77574

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 242) (h2 : num_friends = 12) : 
  total_balloons % num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l775_77574


namespace NUMINAMATH_CALUDE_domain_of_g_l775_77547

-- Define the function f with domain [-1, 2]
def f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2*x + 1) ∈ f

-- Theorem stating that the domain of g is [-1, 1/2]
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l775_77547


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l775_77566

theorem solution_of_linear_equation (x y a : ℝ) : 
  x = 1 → y = 3 → a * x - 2 * y = 4 → a = 10 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l775_77566


namespace NUMINAMATH_CALUDE_equivalent_representations_l775_77582

theorem equivalent_representations : ∀ (a b c d e : ℚ),
  (a = 15 ∧ b = 20 ∧ c = 6 ∧ d = 8 ∧ e = 75) →
  (a / b = c / d) ∧
  (a / b = 3 / 4) ∧
  (a / b = 0.75) ∧
  (a / b = e / 100) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_representations_l775_77582


namespace NUMINAMATH_CALUDE_asymptote_sum_l775_77526

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 3)*(x - 1)*(x - 3)) → A + B + C = 15 :=
by sorry

end NUMINAMATH_CALUDE_asymptote_sum_l775_77526


namespace NUMINAMATH_CALUDE_banana_arrangements_l775_77508

def word : String := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem banana_arrangements :
  (factorial letter_count) / (factorial b_count * factorial a_count * factorial n_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l775_77508


namespace NUMINAMATH_CALUDE_quadrilateral_bd_value_l775_77527

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- The quadrilateral satisfies the triangle inequality --/
def satisfies_triangle_inequality (q : Quadrilateral) : Prop :=
  q.AB + q.BD > q.DA ∧
  q.BC + q.CD > q.BD ∧
  q.DA + q.BD > q.AB ∧
  q.BD + q.CD > q.BC

/-- The theorem to be proved --/
theorem quadrilateral_bd_value (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 19)
  (h3 : q.CD = 6)
  (h4 : q.DA = 10)
  (h5 : satisfies_triangle_inequality q) :
  q.BD = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_bd_value_l775_77527


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l775_77573

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (age_15th_student : ℝ), 
    age_15th_student = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧ 
    age_15th_student = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l775_77573


namespace NUMINAMATH_CALUDE_cookie_ratio_l775_77590

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The total number of cookies Meena baked -/
def total_cookies : ℕ := 5 * dozen

/-- The number of cookies Mr. Stone bought -/
def stone_cookies : ℕ := 2 * dozen

/-- The number of cookies Brock bought -/
def brock_cookies : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 15

/-- The number of cookies Katy bought -/
def katy_cookies : ℕ := total_cookies - stone_cookies - brock_cookies - cookies_left

theorem cookie_ratio : 
  (katy_cookies : ℚ) / brock_cookies = 2 := by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l775_77590


namespace NUMINAMATH_CALUDE_f_is_K_function_l775_77565

-- Define a K function
def is_K_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Theorem stating that f is a K function
theorem f_is_K_function : is_K_function f := by sorry

end NUMINAMATH_CALUDE_f_is_K_function_l775_77565


namespace NUMINAMATH_CALUDE_sum_of_reflected_translated_quadratics_is_nonhorizontal_line_l775_77524

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure Quadratic (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a linear function of the form mx + b -/
structure Linear (α : Type*) [Ring α] where
  m : α
  b : α

/-- 
Given a quadratic function q, return a new quadratic function
that is the reflection of q about the x-axis and translated h units horizontally
-/
def reflect_and_translate (q : Quadratic ℝ) (h : ℝ) : Quadratic ℝ :=
  { a := -q.a
  , b := -q.b - 2 * q.a * h
  , c := -q.c - q.b * h - q.a * h^2 }

/-- 
Given two quadratic functions, return their sum as a linear function
-/
def sum_to_linear (q1 q2 : Quadratic ℝ) : Linear ℝ :=
  { m := (q1.a + q2.a) * 2
  , b := q1.b + q2.b }

theorem sum_of_reflected_translated_quadratics_is_nonhorizontal_line 
  (q : Quadratic ℝ) : 
  let f := reflect_and_translate q 5
  let g := reflect_and_translate q (-5)
  let sum := sum_to_linear f g
  sum.m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_reflected_translated_quadratics_is_nonhorizontal_line_l775_77524


namespace NUMINAMATH_CALUDE_seven_digit_palindrome_count_l775_77535

/-- A seven-digit palindrome is a number of the form abcdcba where a ≠ 0 -/
def SevenDigitPalindrome : Type := Nat

/-- The count of seven-digit palindromes -/
def countSevenDigitPalindromes : Nat := 9000

theorem seven_digit_palindrome_count :
  (Finset.filter (λ n : Nat => n ≥ 1000000 ∧ n ≤ 9999999 ∧ 
    (String.mk (List.reverse (String.toList (toString n)))) = toString n)
    (Finset.range 10000000)).card = countSevenDigitPalindromes := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindrome_count_l775_77535


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l775_77548

theorem arithmetic_mean_of_fractions : 
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l775_77548


namespace NUMINAMATH_CALUDE_car_speed_problem_l775_77501

/-- Proves that given the conditions of the car problem, the average speed of Car X is 50 mph -/
theorem car_speed_problem (Vx : ℝ) : 
  (∃ (T : ℝ), 
    T > 0 ∧ 
    Vx * 1.2 + Vx * T = 50 * T ∧ 
    Vx * T = 98) → 
  Vx = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l775_77501


namespace NUMINAMATH_CALUDE_sin_cos_15_ratio_eq_neg_sqrt3_div_3_l775_77550

theorem sin_cos_15_ratio_eq_neg_sqrt3_div_3 :
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) /
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_ratio_eq_neg_sqrt3_div_3_l775_77550


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l775_77532

theorem multiples_of_four_between_100_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 100 ∧ n < 300) (Finset.range 300)).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l775_77532


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l775_77556

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 4th term of the sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 6th term of the sequence -/
def a_6 (a : ℕ → ℝ) : ℝ := a 6

/-- The 8th term of the sequence -/
def a_8 (a : ℕ → ℝ) : ℝ := a 8

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a_4 a)^2 - 3*(a_4 a) + 2 = 0 →
  (a_8 a)^2 - 3*(a_8 a) + 2 = 0 →
  a_6 a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l775_77556


namespace NUMINAMATH_CALUDE_gas_fee_calculation_l775_77580

/-- Calculates the gas fee for a given usage --/
def gas_fee (usage : ℕ) : ℚ :=
  if usage ≤ 60 then
    0.8 * usage
  else
    0.8 * 60 + 1.2 * (usage - 60)

/-- Represents the average cost per cubic meter --/
def average_cost (usage : ℕ) (fee : ℚ) : ℚ :=
  fee / usage

theorem gas_fee_calculation (usage : ℕ) (h : average_cost usage (gas_fee usage) = 0.88) :
  gas_fee usage = 66 := by
  sorry

end NUMINAMATH_CALUDE_gas_fee_calculation_l775_77580


namespace NUMINAMATH_CALUDE_prime_factors_and_recalculation_l775_77584

def original_number : ℕ := 546

theorem prime_factors_and_recalculation (n : ℕ) (h : n = original_number) :
  (∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ n ∧ largest ∣ n ∧
    (∀ p : ℕ, p.Prime → p ∣ n → smallest ≤ p) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ≤ largest) ∧
    smallest + largest = 15) ∧
  (∃ (factors : List ℕ),
    (∀ p ∈ factors, p.Prime ∧ p ∣ n) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ∈ factors) ∧
    (List.prod (List.map (· * 2) factors) = 8736)) :=
by sorry

end NUMINAMATH_CALUDE_prime_factors_and_recalculation_l775_77584


namespace NUMINAMATH_CALUDE_range_of_m_l775_77589

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Define the main theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ (∃ x : ℝ, q x m ∧ p x)) →
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1) ∧ (∃ m : ℝ, m = -1 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l775_77589


namespace NUMINAMATH_CALUDE_unique_number_exists_l775_77546

theorem unique_number_exists : ∃! x : ℕ, (∃ k : ℕ, 3 * x = 9 * k) ∧ 4 * x = 108 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l775_77546


namespace NUMINAMATH_CALUDE_fraction_simplification_l775_77525

theorem fraction_simplification : (1 : ℚ) / 462 + 19 / 42 = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l775_77525


namespace NUMINAMATH_CALUDE_max_value_expression_l775_77579

theorem max_value_expression (x y z : ℝ) (h1 : x + 2*y + z = 7) (h2 : y ≥ 0) :
  ∃ M : ℝ, M = (10.5 : ℝ) ∧ ∀ x' y' z' : ℝ, x' + 2*y' + z' = 7 → y' ≥ 0 →
    x'*y' + x'*z' + y'*z' + y'^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l775_77579


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l775_77593

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l775_77593


namespace NUMINAMATH_CALUDE_zero_subset_X_l775_77576

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_X_l775_77576


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_product_upper_bound_l775_77509

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the condition x^2 + y^2 = x + y
def SumSquaresEqualSum (x y : ℝ) : Prop := x^2 + y^2 = x + y

-- Theorem 1: Minimum value of 1/x + 1/y is 2
theorem min_reciprocal_sum (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  1/x + 1/y ≥ 2 ∧ ∃ x y, x ∈ PositiveReals ∧ y ∈ PositiveReals ∧ SumSquaresEqualSum x y ∧ 1/x + 1/y = 2 :=
sorry

-- Theorem 2: (x+1)(y+1) < 5 for all x, y satisfying the conditions
theorem product_upper_bound (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  (x + 1) * (y + 1) < 5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_product_upper_bound_l775_77509


namespace NUMINAMATH_CALUDE_volunteer_distribution_l775_77585

theorem volunteer_distribution (n : ℕ) (k : ℕ) (m : ℕ) : n = 5 ∧ k = 3 ∧ m = 3 →
  (Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 1) / 2 * Nat.factorial k = 90 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l775_77585


namespace NUMINAMATH_CALUDE_expression_value_l775_77569

theorem expression_value (a b : ℝ) (h : a - 2*b = 3) : 2*a - 4*b - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l775_77569


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l775_77578

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 : ℚ) / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l775_77578


namespace NUMINAMATH_CALUDE_events_A_D_independent_l775_77528

structure Ball :=
  (label : Nat)

def Ω : Type := Ball × Ball

def P : Set Ω → ℝ := sorry

def A : Set Ω := {ω : Ω | ω.fst.label = 1}
def D : Set Ω := {ω : Ω | ω.fst.label + ω.snd.label = 7}

theorem events_A_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_events_A_D_independent_l775_77528


namespace NUMINAMATH_CALUDE_calculation_one_l775_77521

theorem calculation_one : (-3/8) + (-5/8) * (-6) = 27/8 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l775_77521


namespace NUMINAMATH_CALUDE_math_students_count_l775_77541

theorem math_students_count (total : ℕ) (difference : ℕ) (math_students : ℕ) : 
  total = 1256 →
  difference = 408 →
  math_students < 500 →
  math_students + difference + math_students = total →
  math_students = 424 := by
sorry

end NUMINAMATH_CALUDE_math_students_count_l775_77541


namespace NUMINAMATH_CALUDE_interlaced_roots_l775_77504

theorem interlaced_roots (p₁ p₂ q₁ q₂ : ℝ) 
  (h : (q₁ - q₂)^2 + (p₁ - p₂)*(p₁*q₂ - p₂*q₁) < 0) :
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x, x^2 + p₁*x + q₁ = 0 ↔ (x = r₁ ∨ x = r₂)) ∧
    (∀ x, x^2 + p₂*x + q₂ = 0 ↔ (x = r₃ ∨ x = r₄)) ∧
    ((r₁ < r₃ ∧ r₃ < r₂ ∧ r₂ < r₄) ∨ (r₃ < r₁ ∧ r₁ < r₄ ∧ r₄ < r₂)) :=
by sorry

end NUMINAMATH_CALUDE_interlaced_roots_l775_77504


namespace NUMINAMATH_CALUDE_infinite_image_is_infinite_l775_77572

-- Define the concept of an infinite set
def IsInfinite (α : Type*) : Prop := ∃ f : α → α, Function.Injective f ∧ ¬Function.Surjective f

-- State the theorem
theorem infinite_image_is_infinite {A B : Type*} (f : A → B) (h : IsInfinite A) : IsInfinite B := by
  sorry

end NUMINAMATH_CALUDE_infinite_image_is_infinite_l775_77572


namespace NUMINAMATH_CALUDE_inequality_solution_set_l775_77520

-- Define the inequality
def inequality (x : ℝ) : Prop := (2*x - 1) / (x + 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 1/2}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l775_77520


namespace NUMINAMATH_CALUDE_supermarket_spending_l775_77595

/-- Represents the total amount spent at the supermarket -/
def total_spent : ℝ := 120

/-- Represents the amount spent on candy -/
def candy_spent : ℝ := 8

/-- Theorem stating the total amount spent at the supermarket -/
theorem supermarket_spending :
  (1/2 + 1/3 + 1/10) * total_spent + candy_spent = total_spent :=
by sorry

end NUMINAMATH_CALUDE_supermarket_spending_l775_77595


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l775_77571

theorem one_fourth_divided_by_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l775_77571


namespace NUMINAMATH_CALUDE_only_four_and_six_have_three_solutions_l775_77583

def X : Finset ℕ := {1, 2, 5, 7, 11, 13, 16, 17}

def hasThreedifferentsolutions (k : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ), 
    x₁ ∈ X ∧ y₁ ∈ X ∧ x₂ ∈ X ∧ y₂ ∈ X ∧ x₃ ∈ X ∧ y₃ ∈ X ∧
    x₁ - y₁ = k ∧ x₂ - y₂ = k ∧ x₃ - y₃ = k ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)

theorem only_four_and_six_have_three_solutions :
  ∀ k : ℕ, k > 0 → (hasThreedifferentsolutions k ↔ k = 4 ∨ k = 6) := by sorry

end NUMINAMATH_CALUDE_only_four_and_six_have_three_solutions_l775_77583


namespace NUMINAMATH_CALUDE_longer_train_length_l775_77599

-- Define the given values
def speed_train1 : Real := 60  -- km/hr
def speed_train2 : Real := 40  -- km/hr
def length_shorter : Real := 140  -- meters
def crossing_time : Real := 11.519078473722104  -- seconds

-- Define the theorem
theorem longer_train_length :
  ∃ (length_longer : Real),
    length_longer = 180 ∧
    length_shorter + length_longer =
      (speed_train1 + speed_train2) * 1000 / 3600 * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_longer_train_length_l775_77599


namespace NUMINAMATH_CALUDE_largest_six_digit_number_l775_77552

/-- Represents a six-digit number where each digit, starting from the third,
    is the sum of the two preceding digits. -/
structure SixDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  h1 : c = a + b
  h2 : d = b + c
  h3 : e = c + d
  h4 : f = d + e
  h5 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

/-- Converts a SixDigitNumber to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The largest SixDigitNumber is 303369 -/
theorem largest_six_digit_number :
  ∀ n : SixDigitNumber, toNumber n ≤ 303369 := by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_l775_77552


namespace NUMINAMATH_CALUDE_z_in_terms_of_x_and_y_l775_77515

theorem z_in_terms_of_x_and_y (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y ≠ 2*x) 
  (h : 1/x - 2/y = 1/z) : z = x*y/(y - 2*x) := by
  sorry

end NUMINAMATH_CALUDE_z_in_terms_of_x_and_y_l775_77515


namespace NUMINAMATH_CALUDE_staircase_steps_l775_77510

/-- The number of steps Akvort skips at a time -/
def akvort_skip : ℕ := 3

/-- The number of steps Barnden skips at a time -/
def barnden_skip : ℕ := 4

/-- The number of steps Croft skips at a time -/
def croft_skip : ℕ := 5

/-- The minimum number of steps in the staircase -/
def min_steps : ℕ := 19

theorem staircase_steps :
  (min_steps + 1) % akvort_skip = 0 ∧
  (min_steps + 1) % barnden_skip = 0 ∧
  (min_steps + 1) % croft_skip = 0 ∧
  ∀ n : ℕ, n < min_steps →
    ((n + 1) % akvort_skip = 0 ∧
     (n + 1) % barnden_skip = 0 ∧
     (n + 1) % croft_skip = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_staircase_steps_l775_77510


namespace NUMINAMATH_CALUDE_fib_999_1001_minus_1000_squared_l775_77557

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem stating that F₉₉₉ * F₁₀₀₁ - F₁₀₀₀² = 1 for the Fibonacci sequence -/
theorem fib_999_1001_minus_1000_squared :
  fib 999 * fib 1001 - fib 1000 * fib 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_999_1001_minus_1000_squared_l775_77557


namespace NUMINAMATH_CALUDE_all_rules_correct_l775_77586

/-- Custom addition operation -/
def oplus (a b : ℝ) : ℝ := a + b + 1

/-- Custom subtraction operation -/
def ominus (a b : ℝ) : ℝ := a - b - 1

/-- Theorem stating the correctness of all three rules -/
theorem all_rules_correct (a b c : ℝ) : 
  (oplus a b = oplus b a) ∧ 
  (oplus a (oplus b c) = oplus (oplus a b) c) ∧ 
  (ominus a (oplus b c) = ominus (ominus a b) c) :=
sorry

end NUMINAMATH_CALUDE_all_rules_correct_l775_77586


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l775_77568

theorem three_digit_number_problem :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (x : ℚ) - (x : ℚ) / 10 = 201.6 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l775_77568


namespace NUMINAMATH_CALUDE_negation_existential_proposition_l775_77554

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > x - 2) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ x - 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_proposition_l775_77554


namespace NUMINAMATH_CALUDE_power_decomposition_l775_77529

/-- Sum of the first k odd numbers -/
def sum_odd (k : ℕ) : ℕ := k^2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2*n - 1

theorem power_decomposition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (n^2 = sum_odd 10) →
  (nth_odd ((m-1)^2 + 1) = 21) →
  m + n = 15 := by sorry

end NUMINAMATH_CALUDE_power_decomposition_l775_77529


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l775_77597

theorem sqrt_equation_solution :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x + Real.sqrt (x + 1) - Real.sqrt (x + 2) = 0 ∧ x = -1 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l775_77597


namespace NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l775_77513

theorem gcd_n_cubed_plus_16_and_n_plus_3 (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 16) (n + 3) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l775_77513


namespace NUMINAMATH_CALUDE_peter_money_brought_l775_77534

/-- The amount of money Peter brought to the store -/
def money_brought : ℚ := 2

/-- The cost of soda per ounce -/
def soda_cost_per_ounce : ℚ := 1/4

/-- The amount of soda Peter bought in ounces -/
def soda_amount : ℚ := 6

/-- The amount of money Peter left with -/
def money_left : ℚ := 1/2

/-- Proves that the amount of money Peter brought is correct -/
theorem peter_money_brought :
  money_brought = soda_cost_per_ounce * soda_amount + money_left :=
by sorry

end NUMINAMATH_CALUDE_peter_money_brought_l775_77534


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l775_77588

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 1 + I →
  (z₁.re = z₂.re ∧ z₁.im = -z₂.im) →
  z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l775_77588


namespace NUMINAMATH_CALUDE_divisibility_implication_l775_77591

theorem divisibility_implication (a b m n : ℕ) 
  (h1 : a > 1) 
  (h2 : Nat.gcd a b = 1) 
  (h3 : (a^n + b^n) ∣ (a^m + b^m)) : 
  n ∣ m := by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l775_77591


namespace NUMINAMATH_CALUDE_linear_equation_solution_l775_77540

theorem linear_equation_solution (a b : ℝ) :
  (a ≠ 0 → ∃! x : ℝ, a * x + b = 0 ∧ x = -b / a) ∧
  (a = 0 ∧ b = 0 → ∀ x : ℝ, a * x + b = 0) ∧
  (a = 0 ∧ b ≠ 0 → ¬∃ x : ℝ, a * x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l775_77540


namespace NUMINAMATH_CALUDE_mean_of_set_l775_77502

theorem mean_of_set (m : ℝ) : 
  (m + 8 = 16) → 
  (m + (m + 6) + (m + 8) + (m + 14) + (m + 21)) / 5 = 89 / 5 := by
sorry

end NUMINAMATH_CALUDE_mean_of_set_l775_77502


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l775_77536

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l775_77536


namespace NUMINAMATH_CALUDE_inequality_proof_l775_77539

theorem inequality_proof (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  let L := (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c))
  L ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l775_77539


namespace NUMINAMATH_CALUDE_relay_race_time_l775_77553

/-- The relay race problem -/
theorem relay_race_time (athlete1 athlete2 athlete3 athlete4 total : ℕ) : 
  athlete1 = 55 →
  athlete2 = athlete1 + 10 →
  athlete3 = athlete2 - 15 →
  athlete4 = athlete1 - 25 →
  total = athlete1 + athlete2 + athlete3 + athlete4 →
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l775_77553


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l775_77581

/-- The cost of peanuts per pound -/
def cost_per_pound : ℚ := 3

/-- The minimum purchase in pounds -/
def minimum_purchase : ℚ := 15

/-- The amount Baxter spent on peanuts -/
def amount_spent : ℚ := 105

/-- The number of pounds Baxter purchased over the minimum -/
def pounds_over_minimum : ℚ := (amount_spent / cost_per_pound) - minimum_purchase

theorem baxter_peanut_purchase :
  pounds_over_minimum = 20 :=
by sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l775_77581


namespace NUMINAMATH_CALUDE_min_value_exponential_product_l775_77594

theorem min_value_exponential_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 3) :
  Real.exp (1 / a) * Real.exp (1 / b) ≥ Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_product_l775_77594


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l775_77530

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (3 * x + 4 * y = 2) ∧ (2 * x - y = 5) ↔ (x = 2 ∧ y = -1) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 1) < 7) ∧ (x - 2 ≤ (2 * x - 3) / 3) ↔ (-2 < x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l775_77530


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l775_77564

theorem smallest_m_divisibility (n : ℕ) (h_odd : Odd n) :
  (∃ (m : ℕ), m > 0 ∧ ∀ (k : ℕ), k > 0 → k < m →
    ¬(262417 ∣ (529^n + k * 132^n))) ∧
  (262417 ∣ (529^n + 1 * 132^n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l775_77564


namespace NUMINAMATH_CALUDE_f_composition_equals_226_l775_77537

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

theorem f_composition_equals_226 : f (f (f 1)) = 226 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_226_l775_77537


namespace NUMINAMATH_CALUDE_total_sneaker_spending_l775_77592

/-- Geoff's sneaker spending over three days -/
def sneaker_spending (day1_spend : ℝ) : ℝ :=
  let day2_spend := 4 * day1_spend * (1 - 0.1)  -- 4 times day1 with 10% discount
  let day3_spend := 5 * day1_spend * (1 + 0.08) -- 5 times day1 with 8% tax
  day1_spend + day2_spend + day3_spend

/-- Theorem: Geoff's total sneaker spending over three days is $600 -/
theorem total_sneaker_spending :
  sneaker_spending 60 = 600 := by sorry

end NUMINAMATH_CALUDE_total_sneaker_spending_l775_77592


namespace NUMINAMATH_CALUDE_child_b_share_l775_77555

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 5400 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 1800 := by
sorry

end NUMINAMATH_CALUDE_child_b_share_l775_77555


namespace NUMINAMATH_CALUDE_meiosis_fertilization_result_l775_77538

/-- Represents a genetic combination -/
structure GeneticCombination where
  -- Add necessary fields

/-- Represents a gamete -/
structure Gamete where
  -- Add necessary fields

/-- Represents an organism -/
structure Organism where
  genetic_combination : GeneticCombination

/-- Meiosis process -/
def meiosis (parent : Organism) : List Gamete :=
  sorry

/-- Fertilization process -/
def fertilization (gamete1 gamete2 : Gamete) : Organism :=
  sorry

/-- Predicate to check if two genetic combinations are different -/
def are_different (gc1 gc2 : GeneticCombination) : Prop :=
  sorry

theorem meiosis_fertilization_result 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    (∃ (g1 : Gamete) (g2 : Gamete), 
      g1 ∈ meiosis parent1 ∧ 
      g2 ∈ meiosis parent2 ∧ 
      offspring = fertilization g1 g2) ∧
    are_different offspring.genetic_combination parent1.genetic_combination ∧
    are_different offspring.genetic_combination parent2.genetic_combination :=
  sorry

end NUMINAMATH_CALUDE_meiosis_fertilization_result_l775_77538


namespace NUMINAMATH_CALUDE_added_value_proof_l775_77505

theorem added_value_proof (N V : ℚ) : 
  N = 1280 → (N + V) / 125 = 7392 / 462 → V = 720 := by sorry

end NUMINAMATH_CALUDE_added_value_proof_l775_77505


namespace NUMINAMATH_CALUDE_john_final_amount_l775_77596

def calculate_final_amount (initial_amount : ℚ) (game_cost : ℚ) (candy_cost : ℚ) 
  (soda_cost : ℚ) (magazine_cost : ℚ) (coupon_value : ℚ) (discount_rate : ℚ) 
  (allowance : ℚ) : ℚ :=
  let discounted_soda_cost := soda_cost * (1 - discount_rate)
  let magazine_paid := magazine_cost - coupon_value
  let total_expenses := game_cost + candy_cost + discounted_soda_cost + magazine_paid
  let remaining_after_expenses := initial_amount - total_expenses
  remaining_after_expenses + allowance

theorem john_final_amount :
  calculate_final_amount 5 2 1 1.5 3 0.5 0.1 26 = 24.15 := by
  sorry

end NUMINAMATH_CALUDE_john_final_amount_l775_77596


namespace NUMINAMATH_CALUDE_condition_analysis_l775_77533

theorem condition_analysis (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l775_77533


namespace NUMINAMATH_CALUDE_courtney_marbles_count_l775_77517

/-- The number of marbles in Courtney's first jar -/
def first_jar : ℕ := 80

/-- The number of marbles in Courtney's second jar -/
def second_jar : ℕ := 2 * first_jar

/-- The number of marbles in Courtney's third jar -/
def third_jar : ℕ := first_jar / 4

/-- The total number of marbles Courtney has -/
def total_marbles : ℕ := first_jar + second_jar + third_jar

theorem courtney_marbles_count : total_marbles = 260 := by
  sorry

end NUMINAMATH_CALUDE_courtney_marbles_count_l775_77517


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l775_77559

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 →
  p_black = 0.5 →
  p_red + p_black + p_white = 1 →
  p_white = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l775_77559


namespace NUMINAMATH_CALUDE_sum_of_cubic_difference_l775_77514

theorem sum_of_cubic_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubic_difference_l775_77514


namespace NUMINAMATH_CALUDE_book_club_members_count_l775_77511

def annual_snack_fee : ℕ := 150
def hardcover_books_count : ℕ := 6
def hardcover_book_price : ℕ := 30
def paperback_books_count : ℕ := 6
def paperback_book_price : ℕ := 12
def total_collected : ℕ := 2412

theorem book_club_members_count :
  let cost_per_member := annual_snack_fee +
    hardcover_books_count * hardcover_book_price +
    paperback_books_count * paperback_book_price
  total_collected / cost_per_member = 6 :=
by sorry

end NUMINAMATH_CALUDE_book_club_members_count_l775_77511


namespace NUMINAMATH_CALUDE_bead_arrangement_theorem_l775_77531

/-- Represents a bead with a color --/
structure Bead where
  color : Nat

/-- Represents a necklace of beads --/
def Necklace := List Bead

/-- Checks if a segment of beads contains at least k different colors --/
def hasAtLeastKColors (segment : List Bead) (k : Nat) : Prop :=
  (segment.map (·.color)).toFinset.card ≥ k

/-- The property we want to prove --/
theorem bead_arrangement_theorem (total_beads : Nat) (num_colors : Nat) (beads_per_color : Nat)
    (h1 : total_beads = 1000)
    (h2 : num_colors = 50)
    (h3 : beads_per_color = 20)
    (h4 : total_beads = num_colors * beads_per_color) :
    ∃ (n : Nat),
      (∀ (necklace : Necklace),
        necklace.length = total_beads →
        (∀ (i : Nat),
          i + n ≤ necklace.length →
          hasAtLeastKColors (necklace.take n) 25)) ∧
      (∀ (m : Nat),
        m < n →
        ∃ (necklace : Necklace),
          necklace.length = total_beads ∧
          ∃ (i : Nat),
            i + m ≤ necklace.length ∧
            ¬hasAtLeastKColors (necklace.take m) 25) :=
  sorry

#check bead_arrangement_theorem

end NUMINAMATH_CALUDE_bead_arrangement_theorem_l775_77531


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l775_77516

theorem no_real_solution_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 7*x + 10)) ∧ 
             (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 7*x + 10 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l775_77516


namespace NUMINAMATH_CALUDE_min_value_c_l775_77512

-- Define the consecutive integers
def consecutive_integers (a b c d e : ℕ) : Prop :=
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e

-- Define perfect square and perfect cube
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

-- Main theorem
theorem min_value_c (a b c d e : ℕ) :
  consecutive_integers a b c d e →
  is_perfect_square (b + c + d) →
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ (∀ c' : ℕ, c' < 675 → 
    ¬(∃ a' b' d' e' : ℕ, consecutive_integers a' b' c' d' e' ∧
      is_perfect_square (b' + c' + d') ∧
      is_perfect_cube (a' + b' + c' + d' + e'))) :=
by sorry

end NUMINAMATH_CALUDE_min_value_c_l775_77512


namespace NUMINAMATH_CALUDE_parabola_vertex_l775_77519

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -x^2 + 15

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, 15)

/-- Theorem: The vertex of the parabola y = -x^2 + 15 is at the point (0, 15) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≤ parabola (vertex.1)) ∧ parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l775_77519
