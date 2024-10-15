import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l840_84025

theorem min_value_sum_reciprocals (a b : ℝ) (h : Real.log a + Real.log b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log x + Real.log y = 0 ∧ 2/x + 1/y < 2/a + 1/b) ∨ 
  (2/a + 1/b = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l840_84025


namespace NUMINAMATH_CALUDE_circle_equation_correct_l840_84080

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the center of the circle
def center : ℝ × ℝ := (1, 1)

-- Define the point that the circle passes through
def point_on_circle : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem circle_equation_correct :
  -- The circle passes through the point (1, 0)
  circle_equation point_on_circle.1 point_on_circle.2 ∧
  -- The center is at the intersection of x=1 and x+y=2
  center.1 = 1 ∧ center.1 + center.2 = 2 ∧
  -- The equation represents a circle with the given center
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l840_84080


namespace NUMINAMATH_CALUDE_book_arrangement_count_l840_84017

/-- Represents the number of math books -/
def num_math_books : Nat := 4

/-- Represents the number of history books -/
def num_history_books : Nat := 4

/-- Represents the condition that a math book must be at each end -/
def math_books_at_ends : Nat := 2

/-- Represents the remaining math books to be placed -/
def remaining_math_books : Nat := num_math_books - math_books_at_ends

/-- Represents the arrangement of books satisfying all conditions -/
def valid_arrangement (n m : Nat) : Nat :=
  (n * (n - 1)) * (m.factorial) * (remaining_math_books.factorial)

/-- Theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  valid_arrangement num_math_books num_history_books = 576 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l840_84017


namespace NUMINAMATH_CALUDE_sum_18_47_in_base5_l840_84058

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_47_in_base5 :
  toBase5 (18 + 47) = [2, 3, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_18_47_in_base5_l840_84058


namespace NUMINAMATH_CALUDE_solve_for_n_l840_84053

theorem solve_for_n (Q s r k : ℝ) (h : Q = (s * r) / (1 + k) ^ n) :
  n = Real.log ((s * r) / Q) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l840_84053


namespace NUMINAMATH_CALUDE_mobile_phone_cost_l840_84004

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 4
def mobile_profit_percent : ℝ := 9
def overall_profit : ℝ := 120

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent / 100) + 
   mobile_cost * (1 + mobile_profit_percent / 100)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 8000 := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_l840_84004


namespace NUMINAMATH_CALUDE_initial_persimmons_l840_84088

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_persimmons_l840_84088


namespace NUMINAMATH_CALUDE_double_angle_sine_15_degrees_l840_84089

theorem double_angle_sine_15_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_angle_sine_15_degrees_l840_84089


namespace NUMINAMATH_CALUDE_binary_sum_equals_141_l840_84011

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 111000₂ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- The sum of the two binary numbers in decimal -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_equals_141 : sum_decimal = 141 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_141_l840_84011


namespace NUMINAMATH_CALUDE_december_23_is_saturday_l840_84030

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (advanceDay d k)

-- Theorem statement
theorem december_23_is_saturday (thanksgiving : DayOfWeek) 
  (h : thanksgiving = DayOfWeek.Thursday) : 
  advanceDay thanksgiving 30 = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_december_23_is_saturday_l840_84030


namespace NUMINAMATH_CALUDE_f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l840_84012

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 8

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4 * a * x^2 - 12 * a^2 * x + 3 * a^3 - 8

-- Theorem 1: f(x) < 0 for all x ∈ [1, 2] iff a > 10
theorem f_negative_iff_a_greater_than_ten (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x < 0) ↔ a > 10 := by sorry

-- Theorem 2: No integer a exists such that g(x) has a local minimum in (0, 1)
theorem no_integer_a_for_g_local_minimum :
  ¬ ∃ a : ℤ, ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (g (a : ℝ)) x := by sorry

end NUMINAMATH_CALUDE_f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l840_84012


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l840_84007

/-- The rate per kg of grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 10

/-- The rate per kg of mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid to the shopkeeper -/
def total_paid : ℝ := 1195

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l840_84007


namespace NUMINAMATH_CALUDE_new_revenue_is_354375_l840_84040

/-- Calculates the total revenue at the new price given the conditions --/
def calculate_new_revenue (price_increase : ℕ) (sales_decrease : ℕ) (revenue_increase : ℕ) (new_sales : ℕ) : ℕ :=
  let original_sales := new_sales + sales_decrease
  let original_price := (revenue_increase + price_increase * new_sales) / sales_decrease
  let new_price := original_price + price_increase
  new_price * new_sales

/-- Theorem stating that the total revenue at the new price is $354,375 --/
theorem new_revenue_is_354375 :
  calculate_new_revenue 1000 8 26000 63 = 354375 := by
  sorry

end NUMINAMATH_CALUDE_new_revenue_is_354375_l840_84040


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l840_84062

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2
  {x : ℝ | f x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l840_84062


namespace NUMINAMATH_CALUDE_stickers_needed_for_both_prizes_l840_84045

def current_stickers : ℕ := 250
def small_prize_requirement : ℕ := 800
def big_prize_requirement : ℕ := 1500

theorem stickers_needed_for_both_prizes :
  (small_prize_requirement - current_stickers) + (big_prize_requirement - current_stickers) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_stickers_needed_for_both_prizes_l840_84045


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l840_84078

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l840_84078


namespace NUMINAMATH_CALUDE_quadratic_factorization_l840_84002

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l840_84002


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l840_84098

/-- Given that 14 oranges weigh the same as 10 apples, 
    prove that 42 oranges weigh the same as 30 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  14 * orange_weight = 10 * apple_weight →
  42 * orange_weight = 30 * apple_weight :=
by
  sorry

#check orange_apple_weight_equivalence

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l840_84098


namespace NUMINAMATH_CALUDE_trailing_zeros_28_factorial_l840_84066

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 28! is 6 -/
theorem trailing_zeros_28_factorial :
  trailingZeros 28 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_28_factorial_l840_84066


namespace NUMINAMATH_CALUDE_find_n_l840_84067

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ) : ℚ := 2 * n

/-- Theorem stating that given the conditions, n must equal 10 -/
theorem find_n : ∃ (n : ℕ), n > 0 ∧ a^2 - (b n)^2 = 0 ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_find_n_l840_84067


namespace NUMINAMATH_CALUDE_permutation_product_difference_divisible_l840_84073

def is_permutation (s : Fin 2016 → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 2016 → ∃ i : Fin 2016, s i = n

theorem permutation_product_difference_divisible
  (a b : Fin 2016 → ℕ)
  (ha : is_permutation a)
  (hb : is_permutation b) :
  ∃ i j : Fin 2016, i ≠ j ∧ (2017 ∣ a i * b i - a j * b j) :=
sorry

end NUMINAMATH_CALUDE_permutation_product_difference_divisible_l840_84073


namespace NUMINAMATH_CALUDE_vegetables_amount_l840_84038

def beef_initial : ℕ := 4
def beef_unused : ℕ := 1

def beef_used (initial unused : ℕ) : ℕ := initial - unused

def vegetables_used (beef : ℕ) : ℕ := 2 * beef

theorem vegetables_amount : vegetables_used (beef_used beef_initial beef_unused) = 6 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_amount_l840_84038


namespace NUMINAMATH_CALUDE_slips_with_three_l840_84079

/-- Given a bag of 20 slips with numbers 3 or 8, prove the number of 3s when expected value is 6 -/
theorem slips_with_three (total : ℕ) (value_one value_two : ℕ) (expected_value : ℚ) : 
  total = 20 →
  value_one = 3 →
  value_two = 8 →
  expected_value = 6 →
  ∃ (num_value_one : ℕ),
    num_value_one ≤ total ∧
    (num_value_one : ℚ) / total * value_one + (total - num_value_one : ℚ) / total * value_two = expected_value ∧
    num_value_one = 8 :=
by sorry

end NUMINAMATH_CALUDE_slips_with_three_l840_84079


namespace NUMINAMATH_CALUDE_value_of_a_l840_84003

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l840_84003


namespace NUMINAMATH_CALUDE_tangency_condition_single_intersection_condition_l840_84041

-- Define the line l: y = kx - 3k + 2
def line (k x : ℝ) : ℝ := k * x - 3 * k + 2

-- Define the curve C: (x-1)^2 + (y+1)^2 = 4
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4

-- Define the domain of x for the curve
def x_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- Theorem for tangency condition
theorem tangency_condition (k : ℝ) : 
  (∃ x, x_domain x ∧ curve x (line k x) ∧ 
   (∀ x', x' ≠ x → ¬curve x' (line k x'))) ↔ 
  k = 5/12 :=
sorry

-- Theorem for single intersection condition
theorem single_intersection_condition (k : ℝ) :
  (∃! x, x_domain x ∧ curve x (line k x)) ↔ 
  (1/2 < k ∧ k ≤ 5/2) ∨ k = 5/12 :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_single_intersection_condition_l840_84041


namespace NUMINAMATH_CALUDE_constant_term_product_l840_84024

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the conditions
axiom h1 : r = p * q
axiom h2 : p.coeff 0 = 5
axiom h3 : p.leadingCoeff = 2
axiom h4 : p.degree = 2
axiom h5 : r.coeff 0 = -15

-- Theorem statement
theorem constant_term_product :
  q.eval 0 = -3 :=
sorry

end NUMINAMATH_CALUDE_constant_term_product_l840_84024


namespace NUMINAMATH_CALUDE_average_weight_increase_l840_84035

theorem average_weight_increase (original_group_size : ℕ) 
  (original_weight : ℝ) (new_weight : ℝ) : 
  original_group_size = 5 → 
  original_weight = 50 → 
  new_weight = 70 → 
  (new_weight - original_weight) / original_group_size = 4 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l840_84035


namespace NUMINAMATH_CALUDE_emily_vacation_days_l840_84047

/-- The number of days food lasts for dogs -/
def vacation_days (num_dogs : ℕ) (food_per_dog : ℕ) (total_food : ℕ) : ℕ :=
  total_food * 1000 / (num_dogs * food_per_dog)

/-- Theorem: Emily's vacation lasts 14 days -/
theorem emily_vacation_days :
  vacation_days 4 250 14 = 14 := by
  sorry

end NUMINAMATH_CALUDE_emily_vacation_days_l840_84047


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l840_84043

/-- Given the total pears sold and the amount sold in the afternoon, 
    prove the ratio of afternoon sales to morning sales. -/
theorem pear_sales_ratio 
  (total_pears : ℕ) 
  (afternoon_pears : ℕ) 
  (h1 : total_pears = 480)
  (h2 : afternoon_pears = 320) :
  afternoon_pears / (total_pears - afternoon_pears) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l840_84043


namespace NUMINAMATH_CALUDE_scientific_notation_of_2720000_l840_84010

theorem scientific_notation_of_2720000 :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    2720000 = a * (10 : ℝ) ^ n ∧
    a = 2.72 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2720000_l840_84010


namespace NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l840_84001

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 4 and a right isosceles triangle DBE
    with DB = EB = 1 cut from it, the perimeter of the remaining quadrilateral ACED is 10 + √2 -/
theorem remaining_quadrilateral_perimeter :
  let abc : Triangle := { a := 4, b := 4, c := 4 }
  let dbe : Triangle := { a := 1, b := 1, c := Real.sqrt 2 }
  let aced : Quadrilateral := { a := 4, b := 3, c := Real.sqrt 2, d := 3 }
  aced.perimeter = 10 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l840_84001


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l840_84093

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

theorem increasing_quadratic_condition (a : ℝ) :
  (IncreasingOn (fun x => x^2 + 2*(a-1)*x + 2) 4) → a ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l840_84093


namespace NUMINAMATH_CALUDE_place_mat_length_l840_84008

-- Define the table and place mat properties
def table_radius : ℝ := 5
def num_mats : ℕ := 8
def mat_width : ℝ := 1.5

-- Define the length of the place mat
def mat_length (y : ℝ) : Prop :=
  y = table_radius * Real.sqrt (2 - Real.sqrt 2)

-- Define the arrangement of the place mats
def mats_arrangement (y : ℝ) : Prop :=
  ∃ (chord_length : ℝ),
    chord_length = 2 * table_radius * Real.sin (Real.pi / (2 * num_mats)) ∧
    y = chord_length

-- Theorem statement
theorem place_mat_length :
  ∃ y : ℝ, mat_length y ∧ mats_arrangement y :=
sorry

end NUMINAMATH_CALUDE_place_mat_length_l840_84008


namespace NUMINAMATH_CALUDE_standing_arrangements_eq_48_l840_84065

/-- The number of different standing arrangements for 5 students in a row,
    given the specified conditions. -/
def standing_arrangements : ℕ :=
  let total_students : ℕ := 5
  let positions_for_A : ℕ := total_students - 1
  let remaining_positions : ℕ := total_students - 1
  let arrangements_for_D_and_E : ℕ := remaining_positions * (remaining_positions - 1) / 2
  positions_for_A * arrangements_for_D_and_E

/-- Theorem stating that the number of standing arrangements is 48. -/
theorem standing_arrangements_eq_48 : standing_arrangements = 48 := by
  sorry

#eval standing_arrangements  -- This should output 48

end NUMINAMATH_CALUDE_standing_arrangements_eq_48_l840_84065


namespace NUMINAMATH_CALUDE_amount_left_after_purchase_l840_84028

/-- Represents the price of a single lollipop in dollars -/
def lollipop_price : ℚ := 3/2

/-- Represents the price of a pack of gummies in dollars -/
def gummies_price : ℚ := 2

/-- Represents the number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- Represents the number of packs of gummies bought -/
def num_gummies : ℕ := 2

/-- Represents the initial amount of money Chastity had in dollars -/
def initial_amount : ℚ := 15

/-- Theorem stating that the amount left after purchasing the candies is $5 -/
theorem amount_left_after_purchase : 
  initial_amount - (↑num_lollipops * lollipop_price + ↑num_gummies * gummies_price) = 5 := by
  sorry

end NUMINAMATH_CALUDE_amount_left_after_purchase_l840_84028


namespace NUMINAMATH_CALUDE_original_line_length_l840_84021

-- Define the units
def cm : ℝ := 1
def meter : ℝ := 100 * cm

-- Define the problem parameters
def erased_length : ℝ := 10 * cm
def remaining_length : ℝ := 90 * cm

-- State the theorem
theorem original_line_length :
  ∃ (original_length : ℝ),
    original_length = remaining_length + erased_length ∧
    original_length = 1 * meter :=
by sorry

end NUMINAMATH_CALUDE_original_line_length_l840_84021


namespace NUMINAMATH_CALUDE_largest_square_area_l840_84048

theorem largest_square_area (A B C : ℝ × ℝ) (h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_sum_squares : (B.1 - A.1)^2 + (B.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 + 2 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 500) :
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 125 := by
  sorry

#check largest_square_area

end NUMINAMATH_CALUDE_largest_square_area_l840_84048


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l840_84059

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_exactly_one : ℚ) :
  total = 800 →
  two_or_more = 64 →
  prob_exactly_one = 12 / 100 →
  (prob_exactly_one * total + two_or_more : ℚ) / total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l840_84059


namespace NUMINAMATH_CALUDE_dice_probability_l840_84097

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (at least one pair but not a three-of-a-kind) -/
def favorable_outcomes : ℕ := 27000

/-- The probability of rolling at least one pair but not a three-of-a-kind -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem dice_probability : probability = 625 / 1089 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l840_84097


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l840_84023

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c = 3) ↔ (a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l840_84023


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l840_84019

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (a 3)^2 - 10 * (a 3) + 16 = 0 ∧
  (a 6)^2 - 10 * (a 6) + 16 = 0

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ n : ℕ, a n = 2 * n - 4) ∧ (a 136 = 268) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l840_84019


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l840_84084

theorem quadratic_roots_property (a b : ℝ) : 
  (2 * a^2 + 6 * a - 14 = 0) → 
  (2 * b^2 + 6 * b - 14 = 0) → 
  (2 * a - 3) * (4 * b - 6) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l840_84084


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l840_84052

/-- Given that Sam initially had 9 dimes and gave 7 dimes away, prove that he now has 2 dimes. -/
theorem sams_remaining_dimes (initial_dimes : ℕ) (dimes_given_away : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : dimes_given_away = 7) :
  initial_dimes - dimes_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l840_84052


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l840_84057

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z →
    ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l840_84057


namespace NUMINAMATH_CALUDE_arithmetic_mean_not_less_than_harmonic_mean_l840_84054

theorem arithmetic_mean_not_less_than_harmonic_mean :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ (a + b) / 2 ≥ 2 / (1/a + 1/b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_not_less_than_harmonic_mean_l840_84054


namespace NUMINAMATH_CALUDE_m_divided_by_8_l840_84027

theorem m_divided_by_8 (m : ℕ) (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l840_84027


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_l840_84090

def thursday_sales : ℝ := 210
def saturday_sales : ℝ := 150
def average_daily_sales : ℝ := 260

theorem friday_to_thursday_ratio :
  let total_sales := average_daily_sales * 3
  let friday_sales := total_sales - thursday_sales - saturday_sales
  friday_sales / thursday_sales = 2 := by sorry

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_l840_84090


namespace NUMINAMATH_CALUDE_intersection_complement_eq_l840_84085

open Set

def U : Set ℝ := univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_eq : A ∩ (U \ B) = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_l840_84085


namespace NUMINAMATH_CALUDE_sam_total_money_l840_84076

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's coins in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_total_money : total_value = 10.05 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_money_l840_84076


namespace NUMINAMATH_CALUDE_closest_to_200_l840_84081

def problem_value : ℝ := 2.54 * 7.89 * (4.21 + 5.79)

def options : List ℝ := [150, 200, 250, 300, 350]

theorem closest_to_200 :
  ∀ x ∈ options, x ≠ 200 → |problem_value - 200| < |problem_value - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_200_l840_84081


namespace NUMINAMATH_CALUDE_earth_fresh_water_coverage_l840_84022

theorem earth_fresh_water_coverage : 
  ∀ (land_coverage : ℝ) (salt_water_percentage : ℝ),
  land_coverage = 3 / 10 →
  salt_water_percentage = 97 / 100 →
  (1 - land_coverage) * (1 - salt_water_percentage) = 21 / 1000 := by
sorry

end NUMINAMATH_CALUDE_earth_fresh_water_coverage_l840_84022


namespace NUMINAMATH_CALUDE_f_increasing_max_b_value_ln_2_bounds_l840_84005

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem max_b_value : 
  (∀ x : ℝ, x > 0 → g 2 x > 0) ∧ 
  (∀ b : ℝ, b > 2 → ∃ x : ℝ, x > 0 ∧ g b x ≤ 0) := by sorry

theorem ln_2_bounds : 0.692 < Real.log 2 ∧ Real.log 2 < 0.694 := by sorry

end NUMINAMATH_CALUDE_f_increasing_max_b_value_ln_2_bounds_l840_84005


namespace NUMINAMATH_CALUDE_parabolas_intersection_circle_l840_84020

/-- The parabolas y = (x + 2)^2 and x + 8 = (y - 2)^2 intersect at four points that lie on a circle with radius squared equal to 4 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x + 2)^2 ∧ x + 8 = (y - 2)^2) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_circle_l840_84020


namespace NUMINAMATH_CALUDE_corner_cut_cube_edges_l840_84018

/-- Represents a solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerCutCube where
  original_side_length : ℝ
  removed_side_length : ℝ

/-- Calculates the number of edges in the resulting solid -/
def edge_count (c : CornerCutCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 5 with corners of side length 2 removed has 48 edges -/
theorem corner_cut_cube_edges :
  let c : CornerCutCube := { original_side_length := 5, removed_side_length := 2 }
  edge_count c = 48 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_cube_edges_l840_84018


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l840_84086

/-- Given the number of stickers for Fred, calculate the number of stickers for Jerry. -/
def jerrys_stickers (freds_stickers : ℕ) : ℕ :=
  let georges_stickers := freds_stickers - 6
  3 * georges_stickers

/-- Prove that Jerry has 36 stickers given the conditions in the problem. -/
theorem jerry_has_36_stickers :
  jerrys_stickers 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l840_84086


namespace NUMINAMATH_CALUDE_sum_cube_plus_twice_sum_squares_l840_84049

theorem sum_cube_plus_twice_sum_squares : (3 + 7)^3 + 2*(3^2 + 7^2) = 1116 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_plus_twice_sum_squares_l840_84049


namespace NUMINAMATH_CALUDE_negation_of_implication_l840_84033

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l840_84033


namespace NUMINAMATH_CALUDE_jeremy_age_l840_84051

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) : 
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 :=
by sorry

end NUMINAMATH_CALUDE_jeremy_age_l840_84051


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l840_84077

theorem base_2_representation_of_123 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l840_84077


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l840_84064

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 3 * x + 2 = 2 * y ∧ x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l840_84064


namespace NUMINAMATH_CALUDE_trees_needed_l840_84009

/-- Represents a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the planting scheme for trees -/
structure PlantingScheme where
  treeSpacing : ℕ
  alternateTrees : Bool

/-- Calculates the total number of trees needed for a given playground and planting scheme -/
def totalTrees (p : Playground) (scheme : PlantingScheme) : ℕ :=
  (perimeter p) / scheme.treeSpacing

/-- Theorem stating the total number of trees required for the given playground and planting scheme -/
theorem trees_needed (p : Playground) (scheme : PlantingScheme) :
  p.length = 150 ∧ p.width = 60 ∧ scheme.treeSpacing = 10 ∧ scheme.alternateTrees = true →
  totalTrees p scheme = 42 := by
  sorry

end NUMINAMATH_CALUDE_trees_needed_l840_84009


namespace NUMINAMATH_CALUDE_playground_length_is_687_5_l840_84042

/-- A rectangular playground with given perimeter, breadth, and diagonal -/
structure Playground where
  perimeter : ℝ
  breadth : ℝ
  diagonal : ℝ

/-- The length of a rectangular playground -/
def length (p : Playground) : ℝ :=
  ((p.diagonal ^ 2) - (p.breadth ^ 2)) ^ (1/2)

/-- Theorem stating the length of the specific playground -/
theorem playground_length_is_687_5 (p : Playground) 
  (h1 : p.perimeter = 1200)
  (h2 : p.breadth = 500)
  (h3 : p.diagonal = 850) : 
  length p = 687.5 := by
  sorry

end NUMINAMATH_CALUDE_playground_length_is_687_5_l840_84042


namespace NUMINAMATH_CALUDE_find_m_l840_84094

theorem find_m (a : ℝ) (n m : ℕ) (h1 : a^n = 2) (h2 : a^(m*n) = 16) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l840_84094


namespace NUMINAMATH_CALUDE_unmanned_supermarket_prices_l840_84083

/-- Represents the unit price of keychains in yuan -/
def keychain_price : ℝ := 24

/-- Represents the unit price of plush toys in yuan -/
def plush_toy_price : ℝ := 36

/-- The total number of items bought -/
def total_items : ℕ := 15

/-- The total amount spent on keychains in yuan -/
def total_keychain_cost : ℝ := 240

/-- The total amount spent on plush toys in yuan -/
def total_plush_toy_cost : ℝ := 180

theorem unmanned_supermarket_prices :
  (total_keychain_cost / keychain_price + total_plush_toy_cost / plush_toy_price = total_items) ∧
  (plush_toy_price = 1.5 * keychain_price) := by
  sorry

end NUMINAMATH_CALUDE_unmanned_supermarket_prices_l840_84083


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l840_84095

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if the given side lengths form a valid triangle -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.c + t.a - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The initial triangle T₁ -/
def T₁ : Triangle := { a := 401, b := 403, c := 405 }

/-- The sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℚ := t.a + t.b + t.c

theorem last_triangle_perimeter :
  ∃ n : ℕ, 
    (Triangle.isValid (triangleSequence n)) ∧ 
    ¬(Triangle.isValid (triangleSequence (n + 1))) ∧
    (Triangle.perimeter (triangleSequence n) = 1209 / 512) := by
  sorry

#check last_triangle_perimeter

end NUMINAMATH_CALUDE_last_triangle_perimeter_l840_84095


namespace NUMINAMATH_CALUDE_correct_answer_l840_84099

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l840_84099


namespace NUMINAMATH_CALUDE_number_puzzle_l840_84056

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 13) = 93 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l840_84056


namespace NUMINAMATH_CALUDE_tower_count_mod_1000_l840_84092

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => 4 * T n

/-- The main theorem stating that the number of towers with 9 cubes is congruent to 768 mod 1000 -/
theorem tower_count_mod_1000 : T 9 ≡ 768 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_tower_count_mod_1000_l840_84092


namespace NUMINAMATH_CALUDE_decreasing_condition_passes_through_origin_l840_84069

/-- Given linear function y = (2-k)x - k^2 + 4 -/
def y (k x : ℝ) : ℝ := (2 - k) * x - k^2 + 4

/-- y decreases as x increases iff k > 2 -/
theorem decreasing_condition (k : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y k x₁ > y k x₂) ↔ k > 2 :=
sorry

/-- The graph passes through the origin iff k = -2 -/
theorem passes_through_origin (k : ℝ) :
  y k 0 = 0 ↔ k = -2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_passes_through_origin_l840_84069


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l840_84060

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

-- State the theorem
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l840_84060


namespace NUMINAMATH_CALUDE_minimum_workers_needed_l840_84075

/-- The number of units completed per worker per day in the first process -/
def process1_rate : ℕ := 48

/-- The number of units completed per worker per day in the second process -/
def process2_rate : ℕ := 32

/-- The number of units completed per worker per day in the third process -/
def process3_rate : ℕ := 28

/-- The minimum number of workers needed for the first process -/
def workers1 : ℕ := 14

/-- The minimum number of workers needed for the second process -/
def workers2 : ℕ := 21

/-- The minimum number of workers needed for the third process -/
def workers3 : ℕ := 24

/-- The theorem stating the minimum number of workers needed for each process -/
theorem minimum_workers_needed :
  (∃ n : ℕ, n > 0 ∧ 
    n = process1_rate * workers1 ∧ 
    n = process2_rate * workers2 ∧ 
    n = process3_rate * workers3) ∧
  (∀ w1 w2 w3 : ℕ, 
    (∃ m : ℕ, m > 0 ∧ 
      m = process1_rate * w1 ∧ 
      m = process2_rate * w2 ∧ 
      m = process3_rate * w3) →
    w1 ≥ workers1 ∧ w2 ≥ workers2 ∧ w3 ≥ workers3) :=
by sorry

end NUMINAMATH_CALUDE_minimum_workers_needed_l840_84075


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l840_84082

/-- Given a line L1 with equation 3x - 6y = 9, prove that the line L2 with equation y = (1/2)x - 1
    is parallel to L1 and passes through the point (2,0). -/
theorem parallel_line_through_point (x y : ℝ) : 
  (3 * x - 6 * y = 9) →  -- Equation of line L1
  (y = (1/2) * x - 1) →  -- Equation of line L2
  (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) →  -- L2 is in slope-intercept form with slope 1/2
  (0 = (1/2) * 2 - 1) →  -- L2 passes through (2,0)
  (∀ x₁ y₁ x₂ y₂ : ℝ, (3 * x₁ - 6 * y₁ = 9 ∧ 3 * x₂ - 6 * y₂ = 9) → 
    ((y₂ - y₁) / (x₂ - x₁) = 1/2)) →  -- Slope of L1 is 1/2
  (y = (1/2) * x - 1)  -- Conclusion: equation of L2
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l840_84082


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l840_84015

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l840_84015


namespace NUMINAMATH_CALUDE_additional_license_plates_l840_84032

def initial_first_letter : Nat := 5
def initial_second_letter : Nat := 3
def initial_first_number : Nat := 5
def initial_second_number : Nat := 5

def new_first_letter : Nat := 5
def new_second_letter : Nat := 4
def new_first_number : Nat := 7
def new_second_number : Nat := 5

def initial_combinations : Nat := initial_first_letter * initial_second_letter * initial_first_number * initial_second_number

def new_combinations : Nat := new_first_letter * new_second_letter * new_first_number * new_second_number

theorem additional_license_plates :
  new_combinations - initial_combinations = 325 := by
  sorry

end NUMINAMATH_CALUDE_additional_license_plates_l840_84032


namespace NUMINAMATH_CALUDE_exists_four_mutual_l840_84037

-- Define a type for people
def Person : Type := ℕ

-- Define a relation for familiarity
def familiar : Person → Person → Prop := sorry

-- Define a group of 18 people
def group : Finset Person := sorry

-- Axiom: The group has exactly 18 people
axiom group_size : Finset.card group = 18

-- Axiom: Any two people are either familiar or unfamiliar
axiom familiar_or_unfamiliar (p q : Person) : p ∈ group → q ∈ group → p ≠ q → 
  familiar p q ∨ ¬familiar p q

-- Theorem to prove
theorem exists_four_mutual (group : Finset Person) 
  (h₁ : Finset.card group = 18) 
  (h₂ : ∀ p q : Person, p ∈ group → q ∈ group → p ≠ q → familiar p q ∨ ¬familiar p q) :
  ∃ (s : Finset Person), Finset.card s = 4 ∧ s ⊆ group ∧
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → familiar p q) ∨
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → ¬familiar p q) :=
sorry

end NUMINAMATH_CALUDE_exists_four_mutual_l840_84037


namespace NUMINAMATH_CALUDE_difference_of_squares_102_99_l840_84087

theorem difference_of_squares_102_99 : 102^2 - 99^2 = 603 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_102_99_l840_84087


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l840_84036

def sora_numbers : List ℕ := [4, 6]
def heesu_numbers : List ℕ := [7, 5]
def jiyeon_numbers : List ℕ := [3, 8]

def sum_list (l : List ℕ) : ℕ := l.sum

theorem heesu_has_greatest_sum :
  sum_list heesu_numbers > sum_list sora_numbers ∧
  sum_list heesu_numbers > sum_list jiyeon_numbers :=
by sorry

end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l840_84036


namespace NUMINAMATH_CALUDE_finite_common_terms_l840_84070

/-- Two sequences of natural numbers with specific recurrence relations have only finitely many common terms -/
theorem finite_common_terms 
  (a b : ℕ → ℕ) 
  (ha : ∀ n : ℕ, n ≥ 1 → a (n + 1) = n * a n + 1)
  (hb : ∀ n : ℕ, n ≥ 1 → b (n + 1) = n * b n - 1) :
  Set.Finite {n : ℕ | ∃ m : ℕ, a n = b m} :=
sorry

end NUMINAMATH_CALUDE_finite_common_terms_l840_84070


namespace NUMINAMATH_CALUDE_centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l840_84031

-- Define basic biological concepts
def Chromosome : Type := Unit
def Centromere : Type := Unit
def Cell : Type := Unit
def Ribosome : Type := Unit
def DNAMolecule : Type := Unit
def Chromatid : Type := Unit
def HomologousChromosome : Type := Unit

-- Define the properties
def has_ribosome (c : Cell) : Prop := sorry
def is_eukaryotic (c : Cell) : Prop := sorry
def number_of_centromeres (c : Cell) : ℕ := sorry
def number_of_chromosomes (c : Cell) : ℕ := sorry
def number_of_dna_molecules (c : Cell) : ℕ := sorry
def number_of_chromatids (c : Cell) : ℕ := sorry
def size_and_shape (h : HomologousChromosome) : ℕ := sorry

-- State the theorem
theorem centromeres_equal_chromosomes :
  ∀ (c : Cell), number_of_centromeres c = number_of_chromosomes c :=
sorry

-- State the conditions
axiom cells_with_ribosomes :
  ∃ (c : Cell), has_ribosome c ∧ ¬is_eukaryotic c

axiom dna_chromatid_ratio :
  ∀ (c : Cell), 
    (number_of_dna_molecules c = number_of_chromatids c) ∨
    (number_of_dna_molecules c = 1 ∧ number_of_chromatids c = 0)

axiom homologous_chromosomes_different :
  ∃ (h1 h2 : HomologousChromosome), size_and_shape h1 ≠ size_and_shape h2

-- The main theorem stating that the statement is false
theorem centromeres_necessarily_equal_chromosomes :
  ¬(∃ (c : Cell), number_of_centromeres c ≠ number_of_chromosomes c) :=
sorry

end NUMINAMATH_CALUDE_centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l840_84031


namespace NUMINAMATH_CALUDE_x_minus_y_value_l840_84091

theorem x_minus_y_value (x y : ℝ) 
  (hx : |x| = 4)
  (hy : |y| = 2)
  (hxy : x * y < 0) :
  x - y = 6 ∨ x - y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l840_84091


namespace NUMINAMATH_CALUDE_m_range_l840_84061

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 0 ∧ m < 1/3

def q (m : ℝ) : Prop := ∃ (e : ℝ), ∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0 ∧ m < 15

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (1/3 ≤ m ∧ m < 15) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l840_84061


namespace NUMINAMATH_CALUDE_binary_101110_equals_octal_56_l840_84000

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101110₂ -/
def binary_101110 : List Bool := [false, true, true, true, true, false]

theorem binary_101110_equals_octal_56 :
  decimal_to_octal (binary_to_decimal binary_101110) = [6, 5] :=
by sorry

end NUMINAMATH_CALUDE_binary_101110_equals_octal_56_l840_84000


namespace NUMINAMATH_CALUDE_probability_one_authentic_one_defective_l840_84055

def total_products : ℕ := 5
def authentic_products : ℕ := 4
def defective_products : ℕ := 1

theorem probability_one_authentic_one_defective :
  (authentic_products * defective_products : ℚ) / (total_products.choose 2) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_authentic_one_defective_l840_84055


namespace NUMINAMATH_CALUDE_sequence_problem_l840_84063

/-- Given a sequence of positive integers x₁, x₂, ..., x₇ satisfying
    x₆ = 144 and x_{n+3} = x_{n+2}(x_{n+1} + x_n) for n = 1, 2, 3, 4,
    prove that x₇ = 3456. -/
theorem sequence_problem (x : Fin 7 → ℕ+) 
    (h1 : x 6 = 144)
    (h2 : ∀ n : Fin 4, x (n + 3) = x (n + 2) * (x (n + 1) + x n)) :
  x 7 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l840_84063


namespace NUMINAMATH_CALUDE_min_sum_max_product_l840_84068

theorem min_sum_max_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * b = 1 → a + b ≥ 2) ∧ (a + b = 1 → a * b ≤ 1/4) := by sorry

end NUMINAMATH_CALUDE_min_sum_max_product_l840_84068


namespace NUMINAMATH_CALUDE_function_properties_l840_84096

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : increasing_on f (-1) 0) :
  (periodic f 2) ∧ 
  (symmetric_about f 1) ∧ 
  (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l840_84096


namespace NUMINAMATH_CALUDE_complex_number_location_l840_84039

theorem complex_number_location (z : ℂ) : 
  z = Complex.mk (Real.sin (2019 * π / 180)) (Real.cos (2019 * π / 180)) →
  Real.sin (2019 * π / 180) < 0 ∧ Real.cos (2019 * π / 180) < 0 :=
by
  sorry

#check complex_number_location

end NUMINAMATH_CALUDE_complex_number_location_l840_84039


namespace NUMINAMATH_CALUDE_square_plus_one_ge_twice_abs_l840_84016

theorem square_plus_one_ge_twice_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_twice_abs_l840_84016


namespace NUMINAMATH_CALUDE_equation_implies_equal_variables_l840_84044

theorem equation_implies_equal_variables (a b : ℝ) 
  (h : (1 / (3 * a)) + (2 / (3 * b)) = 3 / (a + 2 * b)) : a = b :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_equal_variables_l840_84044


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l840_84074

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 5) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 74 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l840_84074


namespace NUMINAMATH_CALUDE_locus_of_M_l840_84072

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- The setup of the problem -/
structure Configuration :=
  (A B C : Point)
  (D : Point)
  (l : Line)
  (P Q N M L : Point)

/-- Condition that A, B, and C are collinear -/
def collinear (A B C : Point) : Prop := sorry

/-- Condition that a point is not on a line -/
def not_on_line (P : Point) (l : Line) : Prop := sorry

/-- Condition that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Condition that a line passes through a point -/
def passes_through (l : Line) (P : Point) : Prop := sorry

/-- Condition that a point is the foot of the perpendicular from another point to a line -/
def is_foot_of_perpendicular (M C : Point) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem locus_of_M (config : Configuration) :
  collinear config.A config.B config.C →
  not_on_line config.D config.l →
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CP parallel to AD
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CQ parallel to BD
  is_foot_of_perpendicular config.M config.C (Line.mk 0 0 0) →  -- PQ line
  (config.C.x - config.N.x) / (config.A.x - config.N.x) = (config.C.x - config.B.x) / (config.A.x - config.C.x) →
  ∃ (l_M : Line),
    passes_through l_M config.L ∧
    parallel l_M (Line.mk 0 0 0) ∧  -- MN line
    ∀ (M : Point), passes_through l_M M ↔ 
      ∃ (D : Point), is_foot_of_perpendicular M config.C (Line.mk 0 0 0) :=
sorry

end NUMINAMATH_CALUDE_locus_of_M_l840_84072


namespace NUMINAMATH_CALUDE_problem_1_l840_84046

theorem problem_1 : (-2)^2 + Real.sqrt 12 - 2 * Real.sin (π / 3) = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l840_84046


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l840_84050

def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 4 / 11 ∧
  4 + 11 = 15 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l840_84050


namespace NUMINAMATH_CALUDE_license_plate_count_l840_84029

/-- The number of digits used in the license plate -/
def num_digits : ℕ := 4

/-- The number of letters used in the license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of letters in the alphabet -/
def letter_choices : ℕ := 32

/-- The maximum number of different car license plates -/
def max_license_plates : ℕ := digit_choices ^ num_digits * letter_choices ^ num_letters

theorem license_plate_count : max_license_plates = 327680000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l840_84029


namespace NUMINAMATH_CALUDE_expression_value_l840_84034

theorem expression_value : 
  Real.sqrt (2018 * 2021 * 2022 * 2023 + 2024^2) - 2024^2 = -12138 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l840_84034


namespace NUMINAMATH_CALUDE_flight_duration_is_two_hours_l840_84026

/-- Calculates the flight duration in hours given the number of peanut bags, 
    peanuts per bag, and consumption rate. -/
def flight_duration (bags : ℕ) (peanuts_per_bag : ℕ) (minutes_per_peanut : ℕ) : ℚ :=
  (bags * peanuts_per_bag * minutes_per_peanut) / 60

/-- Proves that the flight duration is 2 hours given the specified conditions. -/
theorem flight_duration_is_two_hours : 
  flight_duration 4 30 1 = 2 := by
  sorry

#eval flight_duration 4 30 1

end NUMINAMATH_CALUDE_flight_duration_is_two_hours_l840_84026


namespace NUMINAMATH_CALUDE_quadrilateral_area_l840_84013

-- Define a structure for the partitioned triangle
structure PartitionedTriangle where
  -- Areas of the three smaller triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  areaQuad : ℝ
  -- Total area of the original triangle
  totalArea : ℝ
  -- Condition: The sum of all areas equals the total area
  sum_areas : area1 + area2 + area3 + areaQuad = totalArea

-- Theorem statement
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.areaQuad = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l840_84013


namespace NUMINAMATH_CALUDE_initial_candies_l840_84006

theorem initial_candies (package_size : ℕ) (added_candies : ℕ) (total_candies : ℕ) :
  package_size = 15 →
  added_candies = 4 →
  total_candies = 10 →
  total_candies - added_candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l840_84006


namespace NUMINAMATH_CALUDE_dividend_calculation_l840_84014

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h1 : quotient = 36)
  (h2 : divisor = 85)
  (h3 : remainder = 26) :
  divisor * quotient + remainder = 3086 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l840_84014


namespace NUMINAMATH_CALUDE_lawn_length_is_80_l840_84071

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  road_width : ℝ
  travel_cost_per_sqm : ℝ
  total_travel_cost : ℝ

/-- Calculates the area of the roads on the lawn -/
def road_area (l : LawnWithRoads) : ℝ :=
  l.road_width * l.length + l.road_width * (l.width - l.road_width)

/-- Theorem stating the length of the lawn given specific conditions -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.road_width = 10)
    (h3 : l.travel_cost_per_sqm = 5)
    (h4 : l.total_travel_cost = 6500)
    (h5 : l.total_travel_cost = l.travel_cost_per_sqm * road_area l) :
  l.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_80_l840_84071
