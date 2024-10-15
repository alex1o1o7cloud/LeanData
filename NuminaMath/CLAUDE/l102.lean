import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l102_10244

theorem consecutive_even_integers_sum (a : ℤ) : 
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a % 2 = 0 ∧ 
    a + c = 146) →
  a + (a + 2) + (a + 4) + (a + 6) = 296 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l102_10244


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l102_10211

theorem no_integer_pairs_with_square_diff_150 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l102_10211


namespace NUMINAMATH_CALUDE_linda_expenditure_l102_10241

def notebook_price : ℝ := 1.20
def notebook_quantity : ℕ := 3
def pencil_box_price : ℝ := 1.50
def pen_box_price : ℝ := 1.70
def marker_pack_price : ℝ := 2.80
def calculator_price : ℝ := 12.50
def item_discount_rate : ℝ := 0.15
def coupon_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

def total_expenditure : ℝ := 19.52

theorem linda_expenditure :
  let discountable_items_total := notebook_price * notebook_quantity + pencil_box_price + pen_box_price + marker_pack_price
  let discounted_items_total := discountable_items_total * (1 - item_discount_rate)
  let total_after_item_discount := discounted_items_total + calculator_price
  let total_after_coupon := total_after_item_discount * (1 - coupon_discount_rate)
  let final_total := total_after_coupon * (1 + sales_tax_rate)
  final_total = total_expenditure := by
sorry

end NUMINAMATH_CALUDE_linda_expenditure_l102_10241


namespace NUMINAMATH_CALUDE_line_cannot_contain_point_l102_10222

theorem line_cannot_contain_point (m b : ℝ) (h : m * b < 0) :
  ¬∃ x y : ℝ, x = -2022 ∧ y = 0 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_cannot_contain_point_l102_10222


namespace NUMINAMATH_CALUDE_divisibility_theorem_l102_10268

def group_digits (n : ℕ) : List ℕ :=
  sorry

def alternating_sum (groups : List ℕ) : ℤ :=
  sorry

theorem divisibility_theorem (A : ℕ) :
  let groups := group_digits A
  let B := alternating_sum groups
  (7 ∣ (A - B) ∧ 11 ∣ (A - B) ∧ 13 ∣ (A - B)) ↔ (7 ∣ A ∧ 11 ∣ A ∧ 13 ∣ A) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l102_10268


namespace NUMINAMATH_CALUDE_same_expected_defects_l102_10226

/-- Represents a worker's probability distribution of defective products -/
structure Worker where
  p0 : ℝ  -- Probability of 0 defective products
  p1 : ℝ  -- Probability of 1 defective product
  p2 : ℝ  -- Probability of 2 defective products
  p3 : ℝ  -- Probability of 3 defective products
  sum_to_one : p0 + p1 + p2 + p3 = 1
  non_negative : p0 ≥ 0 ∧ p1 ≥ 0 ∧ p2 ≥ 0 ∧ p3 ≥ 0

/-- Calculate the expected number of defective products for a worker -/
def expected_defects (w : Worker) : ℝ :=
  0 * w.p0 + 1 * w.p1 + 2 * w.p2 + 3 * w.p3

/-- Worker A's probability distribution -/
def worker_A : Worker := {
  p0 := 0.4
  p1 := 0.3
  p2 := 0.2
  p3 := 0.1
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Worker B's probability distribution -/
def worker_B : Worker := {
  p0 := 0.4
  p1 := 0.2
  p2 := 0.4
  p3 := 0
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Theorem stating that the expected number of defective products is the same for both workers -/
theorem same_expected_defects : expected_defects worker_A = expected_defects worker_B := by
  sorry

end NUMINAMATH_CALUDE_same_expected_defects_l102_10226


namespace NUMINAMATH_CALUDE_lisa_marbles_l102_10264

/-- The number of marbles each person has -/
structure Marbles where
  connie : ℕ
  juan : ℕ
  mark : ℕ
  lisa : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.connie = 323 ∧
  m.juan = m.connie + 175 ∧
  m.mark = 3 * m.juan ∧
  m.lisa = m.mark / 2 - 200

/-- The theorem stating that Lisa has 547 marbles -/
theorem lisa_marbles (m : Marbles) (h : marble_problem m) : m.lisa = 547 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_l102_10264


namespace NUMINAMATH_CALUDE_quadratic_properties_l102_10289

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Theorem statement
theorem quadratic_properties :
  (∀ x, f x = (x - 2)^2 - 1) ∧
  (∀ x, f x ≥ f 2) ∧
  (f 2 = -1) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Ioc 2 3, f x > f y) ∧
  (∀ y ∈ Set.Icc (-1 : ℝ) 8, ∃ x ∈ Set.Ico (-1 : ℝ) 3, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l102_10289


namespace NUMINAMATH_CALUDE_daily_class_schedule_l102_10296

theorem daily_class_schedule (n m : ℕ) (hn : n = 10) (hm : m = 6) :
  (n.factorial / (n - m).factorial) = 151200 :=
sorry

end NUMINAMATH_CALUDE_daily_class_schedule_l102_10296


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l102_10287

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l102_10287


namespace NUMINAMATH_CALUDE_complex_power_modulus_l102_10217

theorem complex_power_modulus : 
  Complex.abs ((2/3 : ℂ) + (1/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l102_10217


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l102_10203

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l102_10203


namespace NUMINAMATH_CALUDE_unique_solution_l102_10221

theorem unique_solution : ∃! (x : ℝ), x ≥ 0 ∧ x + 10 * Real.sqrt x = 39 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l102_10221


namespace NUMINAMATH_CALUDE_june_science_book_price_l102_10277

/-- Calculates the price of each science book given June's school supply purchases. -/
theorem june_science_book_price (total_budget : ℕ) (math_book_price : ℕ) (math_book_count : ℕ)
  (art_book_price : ℕ) (music_book_cost : ℕ) :
  total_budget = 500 →
  math_book_price = 20 →
  math_book_count = 4 →
  art_book_price = 20 →
  music_book_cost = 160 →
  let science_book_count := math_book_count + 6
  let art_book_count := 2 * math_book_count
  let total_spent := math_book_price * math_book_count +
                     art_book_price * art_book_count +
                     music_book_cost
  let remaining_budget := total_budget - total_spent
  remaining_budget / science_book_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_june_science_book_price_l102_10277


namespace NUMINAMATH_CALUDE_sachin_age_l102_10230

/-- Proves that Sachin's age is 14 years given the conditions -/
theorem sachin_age (sachin rahul : ℕ) 
  (h1 : rahul = sachin + 4)
  (h2 : (sachin : ℚ) / rahul = 7 / 9) : 
  sachin = 14 := by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l102_10230


namespace NUMINAMATH_CALUDE_t_range_for_inequality_l102_10252

theorem t_range_for_inequality (t : ℝ) : 
  (∀ x : ℝ, abs x ≤ 1 → t + 1 > (t^2 - 4) * x) ↔ 
  (t > (Real.sqrt 13 - 1) / 2 ∧ t < (Real.sqrt 21 + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_t_range_for_inequality_l102_10252


namespace NUMINAMATH_CALUDE_electric_vehicle_analysis_l102_10267

-- Define the variables
variable (x : ℝ) -- Number of vehicles a skilled worker can install per month
variable (y : ℝ) -- Number of vehicles a new worker can install per month
variable (m : ℝ) -- Average cost per kilometer of the electric vehicle
variable (a : ℝ) -- Annual mileage

-- Define the theorem
theorem electric_vehicle_analysis :
  -- Part 1: Installation capacity
  (2 * x + y = 10 ∧ x + 3 * y = 10) →
  (x = 4 ∧ y = 2) ∧
  -- Part 2: Cost per kilometer
  (200 / m = 4 * (200 / (m + 0.6))) →
  m = 0.2 ∧
  -- Part 3: Annual cost comparison
  (0.2 * a + 6400 < 0.8 * a + 4000) →
  a > 4000 :=
by sorry

end NUMINAMATH_CALUDE_electric_vehicle_analysis_l102_10267


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l102_10261

theorem alcohol_mixture_percentage (x : ℝ) : 
  (8 * 0.25 + 2 * (x / 100)) / (8 + 2) = 0.224 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l102_10261


namespace NUMINAMATH_CALUDE_total_bottles_l102_10282

theorem total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ)
  (h1 : regular = 57)
  (h2 : diet = 26)
  (h3 : lite = 27) :
  regular + diet + lite = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l102_10282


namespace NUMINAMATH_CALUDE_triple_equation_solutions_l102_10258

theorem triple_equation_solutions :
  ∀ a b c : ℝ, 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
    a^2 + a*b = c ∧ 
    b^2 + b*c = a ∧ 
    c^2 + c*a = b →
    (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
    (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_triple_equation_solutions_l102_10258


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l102_10288

theorem brianna_cd_purchase (total_money : ℚ) (total_cds : ℚ) (h : total_money > 0) (h' : total_cds > 0) :
  (1 / 4 : ℚ) * total_money = (1 / 4 : ℚ) * (total_cds * (total_money / total_cds)) →
  total_money - (total_cds * (total_money / total_cds)) = 0 := by
sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l102_10288


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l102_10250

/-- Given two externally tangent circles with common external tangents,
    prove that the area of the smaller circle is π(625 + 200√2) / 49 -/
theorem area_of_smaller_circle (P A B A' B' S L : ℝ × ℝ) : 
  let r := Real.sqrt ((5 + 10 * Real.sqrt 2) ^ 2 / 49)
  -- Two circles are externally tangent
  (∃ T : ℝ × ℝ, ‖S - T‖ = r ∧ ‖L - T‖ = 2*r) →
  -- PAB and PA'B' are common external tangents
  (‖P - A‖ = 5 ∧ ‖A - B‖ = 5 ∧ ‖P - A'‖ = 5 ∧ ‖A' - B'‖ = 5) →
  -- A and A' are on the smaller circle
  (‖S - A‖ = r ∧ ‖S - A'‖ = r) →
  -- B and B' are on the larger circle
  (‖L - B‖ = 2*r ∧ ‖L - B'‖ = 2*r) →
  -- Area of the smaller circle
  π * r^2 = π * (625 + 200 * Real.sqrt 2) / 49 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l102_10250


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l102_10284

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 / Real.sqrt 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l102_10284


namespace NUMINAMATH_CALUDE_negative_three_inequality_l102_10260

theorem negative_three_inequality (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_inequality_l102_10260


namespace NUMINAMATH_CALUDE_part_one_part_two_l102_10255

-- Part I
theorem part_one (t : ℝ) (h1 : t^2 - 5*t + 4 < 0) (h2 : (t-2)*(t-6) < 0) : 
  2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a ≠ 0) 
  (h_suff : ∀ t : ℝ, 2 < t ∧ t < 6 → t^2 - 5*a*t + 4*a^2 < 0) : 
  3/2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l102_10255


namespace NUMINAMATH_CALUDE_power_order_l102_10270

theorem power_order : 
  let p := (2 : ℕ) ^ 3009
  let q := (3 : ℕ) ^ 2006
  let r := (5 : ℕ) ^ 1003
  r < p ∧ p < q := by sorry

end NUMINAMATH_CALUDE_power_order_l102_10270


namespace NUMINAMATH_CALUDE_octal_sum_l102_10220

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of 444₈, 44₈, and 4₈ in base 8 is 514₈ --/
theorem octal_sum : 
  decimal_to_octal (octal_to_decimal 444 + octal_to_decimal 44 + octal_to_decimal 4) = 514 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_l102_10220


namespace NUMINAMATH_CALUDE_infinite_series_not_computable_l102_10272

/-- An infinite series of natural numbers -/
def infinite_series (n : ℕ) : ℕ := n

/-- A predicate indicating whether a series can be computed algorithmically -/
def is_algorithmically_computable (f : ℕ → ℕ) : Prop :=
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → f n = 0

/-- The theorem stating that the infinite series cannot be computed algorithmically -/
theorem infinite_series_not_computable :
  ¬ (is_algorithmically_computable infinite_series) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_not_computable_l102_10272


namespace NUMINAMATH_CALUDE_min_t_value_l102_10251

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem min_t_value : 
  (∃ (t : ℝ), ∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ (s : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ s) → s ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l102_10251


namespace NUMINAMATH_CALUDE_problem_solution_l102_10214

theorem problem_solution : ∃ x : ℚ, ((15 - 2 + 4 / 1) / x) * 8 = 77 ∧ x = 136 / 77 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l102_10214


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l102_10228

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 3 →
  downstream_distance = 3.6 →
  downstream_time = 1/5 →
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l102_10228


namespace NUMINAMATH_CALUDE_coins_problem_l102_10246

theorem coins_problem (A B C D : ℕ) : 
  A = 21 →
  B = A - 9 →
  C = B + 17 →
  A + B + 5 = C + D →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_coins_problem_l102_10246


namespace NUMINAMATH_CALUDE_count_five_digit_palindromes_l102_10227

/-- A five-digit palindrome is a number of the form abcba where a, b, c are digits and a ≠ 0. -/
def FiveDigitPalindrome (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a ≥ 1 ∧ a ≤ 9 ∧
    b ≥ 0 ∧ b ≤ 9 ∧
    c ≥ 0 ∧ c ≤ 9 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The count of five-digit palindromes. -/
def CountFiveDigitPalindromes : ℕ := 
  (Finset.range 9).card * (Finset.range 10).card * (Finset.range 10).card

theorem count_five_digit_palindromes :
  CountFiveDigitPalindromes = 900 :=
sorry

end NUMINAMATH_CALUDE_count_five_digit_palindromes_l102_10227


namespace NUMINAMATH_CALUDE_two_hundred_fiftieth_letter_l102_10273

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
         | 0 => 'C'
         | 1 => 'A'
         | _ => 'B'

theorem two_hundred_fiftieth_letter : repeating_pattern 250 = 'A' := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_fiftieth_letter_l102_10273


namespace NUMINAMATH_CALUDE_H_function_iff_non_decreasing_l102_10201

/-- A function f: ℝ → ℝ is an H function if for any x₁ ≠ x₂, x₁f(x₁) + x₂f(x₂) ≥ x₁f(x₂) + x₂f(x₁) -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is non-decreasing if for any x₁ < x₂, f(x₁) ≤ f(x₂) -/
def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

theorem H_function_iff_non_decreasing (f : ℝ → ℝ) :
  is_H_function f ↔ is_non_decreasing f := by
  sorry

end NUMINAMATH_CALUDE_H_function_iff_non_decreasing_l102_10201


namespace NUMINAMATH_CALUDE_square_of_negative_three_x_squared_y_l102_10266

theorem square_of_negative_three_x_squared_y (x y : ℝ) :
  (-3 * x^2 * y)^2 = 9 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_three_x_squared_y_l102_10266


namespace NUMINAMATH_CALUDE_solve_star_equation_l102_10253

-- Define the operation ★
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ x : ℝ, star 5 x = 37 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l102_10253


namespace NUMINAMATH_CALUDE_only_b_opens_upwards_l102_10232

def quadratic_a (x : ℝ) : ℝ := 1 - x - 6*x^2
def quadratic_b (x : ℝ) : ℝ := -8*x + x^2 + 1
def quadratic_c (x : ℝ) : ℝ := (1 - x)*(x + 5)
def quadratic_d (x : ℝ) : ℝ := 2 - (5 - x)^2

def opens_upwards (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, ∃ b c : ℝ, ∀ x, f x = a*x^2 + b*x + c

theorem only_b_opens_upwards :
  opens_upwards quadratic_b ∧
  ¬opens_upwards quadratic_a ∧
  ¬opens_upwards quadratic_c ∧
  ¬opens_upwards quadratic_d :=
by sorry

end NUMINAMATH_CALUDE_only_b_opens_upwards_l102_10232


namespace NUMINAMATH_CALUDE_cuboid_breadth_l102_10207

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 9) (harea : area = 432) :
  ∃ b : ℝ, cuboidSurfaceArea l b h = area ∧ b = 144 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_breadth_l102_10207


namespace NUMINAMATH_CALUDE_floor_of_4_8_l102_10231

theorem floor_of_4_8 : ⌊(4.8 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_8_l102_10231


namespace NUMINAMATH_CALUDE_sin_cos_lt_cos_sin_acute_l102_10209

theorem sin_cos_lt_cos_sin_acute (x : ℝ) (h : 0 < x ∧ x < π / 2) : 
  Real.sin (Real.cos x) < Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_lt_cos_sin_acute_l102_10209


namespace NUMINAMATH_CALUDE_percentage_calculation_l102_10265

theorem percentage_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l102_10265


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l102_10297

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = 64) : 
  (x^3 * 64 = 64) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l102_10297


namespace NUMINAMATH_CALUDE_sum_of_park_areas_l102_10204

theorem sum_of_park_areas :
  let park1_side : ℝ := 11
  let park2_side : ℝ := 5
  let park1_area := park1_side * park1_side
  let park2_area := park2_side * park2_side
  park1_area + park2_area = 146 := by
sorry

end NUMINAMATH_CALUDE_sum_of_park_areas_l102_10204


namespace NUMINAMATH_CALUDE_professor_count_proof_l102_10299

/-- The number of professors in the first year -/
def initial_professors : ℕ := 5

/-- The number of failing grades given in the first year -/
def first_year_grades : ℕ := 6480

/-- The number of failing grades given in the second year -/
def second_year_grades : ℕ := 11200

/-- The increase in the number of professors in the second year -/
def professor_increase : ℕ := 3

theorem professor_count_proof :
  (first_year_grades % initial_professors = 0) ∧
  (second_year_grades % (initial_professors + professor_increase) = 0) ∧
  (first_year_grades / initial_professors < second_year_grades / (initial_professors + professor_increase)) ∧
  (∀ p : ℕ, p < initial_professors →
    (first_year_grades % p = 0 ∧ second_year_grades % (p + professor_increase) = 0) →
    (first_year_grades / p < second_year_grades / (p + professor_increase)) →
    False) :=
by
  sorry

end NUMINAMATH_CALUDE_professor_count_proof_l102_10299


namespace NUMINAMATH_CALUDE_locus_all_importance_l102_10294

/-- Definition of a locus --/
def Locus (P : Type*) (condition : P → Prop) : Set P :=
  {p : P | condition p}

/-- Property of comprehensiveness --/
def Comprehensive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, condition p → p ∈ S

/-- Property of exclusivity --/
def Exclusive (S : Set P) (condition : P → Prop) : Prop :=
  ∀ p, p ∈ S → condition p

/-- Theorem: The definition of locus ensures both comprehensiveness and exclusivity --/
theorem locus_all_importance {P : Type*} (condition : P → Prop) :
  let L := Locus P condition
  Comprehensive L condition ∧ Exclusive L condition := by
  sorry

end NUMINAMATH_CALUDE_locus_all_importance_l102_10294


namespace NUMINAMATH_CALUDE_no_solution_for_floor_equation_l102_10257

theorem no_solution_for_floor_equation :
  ¬ ∃ s : ℝ, (⌊s⌋ : ℝ) + s = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_equation_l102_10257


namespace NUMINAMATH_CALUDE_odd_function_sum_l102_10247

def f (x a b : ℝ) : ℝ := (x - 1)^2 + a * x^2 + b

theorem odd_function_sum (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) → a + b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l102_10247


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_l102_10205

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (f a b 1 = 1) →
  (∀ x ∈ Set.Ioo 2 5, f a b x > 0) →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (a b x : ℝ) :
  (f a b 1 = 1) →
  (∀ a ∈ Set.Icc (-2) (-1), f a b x > 0) →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_l102_10205


namespace NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_16_l102_10206

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of two terms equidistant from the beginning and end of the sequence is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n k : ℕ, m < n → a m * a n = a (m + k) * a (n - k) := by sorry

theorem geometric_sequence_product_16 (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h2 : a 4 * a 8 = 16) : a 2 * a 10 = 16 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_16_l102_10206


namespace NUMINAMATH_CALUDE_marble_problem_l102_10279

theorem marble_problem (initial_marbles : ℕ) : 
  (initial_marbles * 40 / 100 / 2 = 20) → initial_marbles = 100 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l102_10279


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l102_10239

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)
  p 1 = 45 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l102_10239


namespace NUMINAMATH_CALUDE_correct_average_l102_10276

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ incorrect_num = 26 ∧ correct_num = 66 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l102_10276


namespace NUMINAMATH_CALUDE_max_min_difference_z_l102_10242

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3)
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ),
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6.5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l102_10242


namespace NUMINAMATH_CALUDE_initial_number_of_people_l102_10281

theorem initial_number_of_people (avg_weight_increase : ℝ) (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  avg_weight_increase * (weight_difference / avg_weight_increase) = weight_difference →
  (weight_difference / avg_weight_increase : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_people_l102_10281


namespace NUMINAMATH_CALUDE_triangle_two_solutions_range_l102_10235

theorem triangle_two_solutions_range (a b : ℝ) (B : ℝ) (h1 : b = 2) (h2 : B = 45 * π / 180) :
  (∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧ 
   a * Real.sin B < b ∧ b < a) ↔ 
  (2 < a ∧ a < 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_range_l102_10235


namespace NUMINAMATH_CALUDE_inequality_always_true_l102_10275

theorem inequality_always_true (a b : ℝ) (h : a * b > 0) :
  b / a + a / b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_always_true_l102_10275


namespace NUMINAMATH_CALUDE_divisors_not_mult_6_l102_10216

/-- The smallest integer satisfying the given conditions -/
def n : ℕ := 2^30 * 3^15 * 5^25

/-- n/2 is a perfect square -/
axiom n_div_2_is_square : ∃ k : ℕ, n / 2 = k^2

/-- n/4 is a perfect cube -/
axiom n_div_4_is_cube : ∃ j : ℕ, n / 4 = j^3

/-- n/5 is a perfect fifth -/
axiom n_div_5_is_fifth : ∃ m : ℕ, n / 5 = m^5

/-- The number of divisors of n -/
def total_divisors : ℕ := (30 + 1) * (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 2 -/
def divisors_mult_2 : ℕ := (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 3 -/
def divisors_mult_3 : ℕ := (29 + 1) * (25 + 1)

/-- Theorem: The number of divisors of n that are not multiples of 6 is 11740 -/
theorem divisors_not_mult_6 : total_divisors - divisors_mult_2 - divisors_mult_3 = 11740 := by
  sorry

end NUMINAMATH_CALUDE_divisors_not_mult_6_l102_10216


namespace NUMINAMATH_CALUDE_local_minimum_implies_m_eq_one_l102_10240

/-- The function f(x) = x(x-m)^2 has a local minimum at x = 1 -/
def has_local_minimum_at_one (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1

/-- The main theorem: if f(x) = x(x-m)^2 has a local minimum at x = 1, then m = 1 -/
theorem local_minimum_implies_m_eq_one (m : ℝ) :
  has_local_minimum_at_one (fun x => x * (x - m)^2) m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_m_eq_one_l102_10240


namespace NUMINAMATH_CALUDE_cary_calorie_deficit_l102_10224

/-- Calculates the net calorie deficit for Cary's grocery store trip -/
theorem cary_calorie_deficit :
  let miles_walked : ℕ := 3
  let calories_per_mile : ℕ := 150
  let candy_bar_calories : ℕ := 200
  let total_calories_burned := miles_walked * calories_per_mile
  let net_deficit := total_calories_burned - candy_bar_calories
  net_deficit = 250 := by sorry

end NUMINAMATH_CALUDE_cary_calorie_deficit_l102_10224


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l102_10219

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁.re = 1 →
  z₁.im = 1 →
  z₂.re = 0 →
  z₂.im = 1 →
  Complex.im (z₁ / z₂) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l102_10219


namespace NUMINAMATH_CALUDE_fraction_simplification_l102_10237

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l102_10237


namespace NUMINAMATH_CALUDE_average_transformation_l102_10218

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 8) :
  ((2 * x₁ - 1) + (2 * x₂ - 1) + (2 * x₃ - 1)) / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l102_10218


namespace NUMINAMATH_CALUDE_shina_probability_l102_10223

def word : Finset Char := {'М', 'А', 'Ш', 'И', 'Н', 'А'}

def draw_probability (word : Finset Char) (target : List Char) : ℚ :=
  let n := word.card
  let prob := target.foldl (λ acc c =>
    acc * (word.filter (λ x => x = c)).card / n) 1
  prob * (n - 1) * (n - 2) * (n - 3) / n

theorem shina_probability :
  draw_probability word ['Ш', 'И', 'Н', 'А'] = 1 / 180 := by
  sorry

end NUMINAMATH_CALUDE_shina_probability_l102_10223


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l102_10290

/-- Given a quadratic function y = -x^2 + 8x - 7 -/
def f (x : ℝ) : ℝ := -x^2 + 8*x - 7

theorem quadratic_function_properties :
  /- (1) y increases as x increases for x < 4 -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f x₁ < f x₂) ∧
  /- (2) y < 0 for x < 1 or x > 7 -/
  (∀ x : ℝ, (x < 1 ∨ x > 7) → f x < 0) :=
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l102_10290


namespace NUMINAMATH_CALUDE_match_probabilities_and_expectation_l102_10202

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_wins
| B_wins

/-- Represents the state of the match after the first two games -/
structure MatchState :=
  (A_wins : Nat)
  (B_wins : Nat)

/-- The probability of A winning a single game -/
def p_A_win : ℝ := 0.6

/-- The probability of B winning a single game -/
def p_B_win : ℝ := 0.4

/-- The initial state of the match after two games -/
def initial_state : MatchState := ⟨1, 1⟩

/-- The number of wins required to win the match -/
def wins_required : Nat := 3

/-- Calculates the probability of A winning the match given the current state -/
def prob_A_wins_match (state : MatchState) : ℝ :=
  sorry

/-- Calculates the expected number of additional games played -/
def expected_additional_games (state : MatchState) : ℝ :=
  sorry

theorem match_probabilities_and_expectation :
  prob_A_wins_match initial_state = 0.648 ∧
  expected_additional_games initial_state = 2.48 := by
  sorry

end NUMINAMATH_CALUDE_match_probabilities_and_expectation_l102_10202


namespace NUMINAMATH_CALUDE_units_digit_of_2009_pow_2008_plus_2013_l102_10254

theorem units_digit_of_2009_pow_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2009_pow_2008_plus_2013_l102_10254


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l102_10271

/-- A cubic polynomial with specific values at 0, 1, and -1 -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  value_at_zero : P 0 = k
  value_at_one : P 1 = 2 * k
  value_at_neg_one : P (-1) = 3 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 14k -/
theorem cubic_polynomial_sum (k : ℝ) (p : CubicPolynomial k) :
  p.P 2 + p.P (-2) = 14 * k := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l102_10271


namespace NUMINAMATH_CALUDE_circle_area_equivalence_l102_10293

theorem circle_area_equivalence (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 33) (h₂ : r₂ = 24) : 
  (π * r₁^2 - π * r₂^2 = π * r₃^2) → r₃ = 3 * Real.sqrt 57 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equivalence_l102_10293


namespace NUMINAMATH_CALUDE_sin_over_two_minus_cos_max_value_l102_10274

theorem sin_over_two_minus_cos_max_value (x : ℝ) : 
  (Real.sin x) / (2 - Real.cos x) ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_over_two_minus_cos_max_value_l102_10274


namespace NUMINAMATH_CALUDE_officers_on_duty_l102_10215

theorem officers_on_duty (total_female : ℕ) (on_duty : ℕ) 
  (h1 : total_female = 1000)
  (h2 : on_duty / 2 = total_female / 4) : 
  on_duty = 500 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l102_10215


namespace NUMINAMATH_CALUDE_mapping_count_l102_10229

-- Define the sets P and Q
variable (P Q : Type)

-- Define the conditions
variable (h1 : Fintype Q)
variable (h2 : Fintype.card Q = 3)
variable (h3 : Fintype P)
variable (h4 : (Fintype.card P) ^ (Fintype.card Q) = 81)

-- The theorem to prove
theorem mapping_count : (Fintype.card Q) ^ (Fintype.card P) = 64 := by
  sorry

end NUMINAMATH_CALUDE_mapping_count_l102_10229


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l102_10286

theorem fourth_circle_radius (r₁ r₂ r : ℝ) (h₁ : r₁ = 17) (h₂ : r₂ = 27) :
  π * r^2 = π * (r₂^2 - r₁^2) → r = 2 * Real.sqrt 110 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l102_10286


namespace NUMINAMATH_CALUDE_bobs_remaining_amount_l102_10225

/-- Calculates the remaining amount after Bob's spending over three days. -/
def remaining_amount (initial : ℚ) (mon_frac : ℚ) (tue_frac : ℚ) (wed_frac : ℚ) : ℚ :=
  let after_mon := initial * (1 - mon_frac)
  let after_tue := after_mon * (1 - tue_frac)
  after_tue * (1 - wed_frac)

/-- Theorem stating that Bob's remaining amount is $20 after three days of spending. -/
theorem bobs_remaining_amount :
  remaining_amount 80 (1/2) (1/5) (3/8) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bobs_remaining_amount_l102_10225


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l102_10285

theorem arithmetic_calculation : (20 * 24) / (2 * 0 + 2 * 4) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l102_10285


namespace NUMINAMATH_CALUDE_solution_check_unique_non_solution_l102_10243

theorem solution_check : ℝ → ℝ → Prop :=
  fun x y => x + y = 5

theorem unique_non_solution :
  (solution_check 2 3 ∧ 
   solution_check (-2) 7 ∧ 
   solution_check 0 5) ∧ 
  ¬(solution_check 1 6) := by
  sorry

end NUMINAMATH_CALUDE_solution_check_unique_non_solution_l102_10243


namespace NUMINAMATH_CALUDE_machines_count_l102_10283

/-- Given that n machines produce x units in 6 days and 12 machines produce 3x units in 6 days,
    where all machines work at an identical constant rate, prove that n = 4. -/
theorem machines_count (n : ℕ) (x : ℝ) (h1 : x > 0) :
  (n * x / 6 = x / 6) →
  (12 * (3 * x) / 6 = 3 * x / 6) →
  (n * x / (6 * n) = 12 * (3 * x) / (6 * 12)) →
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_machines_count_l102_10283


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l102_10236

/-- Represents a trapezoid ABCD with diagonal AC -/
structure Trapezoid where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Lengths
  AB : ℝ
  DC : ℝ
  AD : ℝ
  -- Properties
  is_trapezoid : (B.2 = C.2) ∧ (A.2 = D.2) -- BC parallel to AD
  AB_length : dist A B = AB
  DC_length : dist D C = DC
  AD_length : dist A D = AD

/-- The length of AC in the trapezoid is approximately 30.1 -/
theorem trapezoid_diagonal_length (t : Trapezoid) (h1 : t.AB = 15) (h2 : t.DC = 24) (h3 : t.AD = 7) :
  ∃ ε > 0, abs (dist t.A t.C - 30.1) < ε :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l102_10236


namespace NUMINAMATH_CALUDE_perfect_square_expression_l102_10249

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l102_10249


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_prod_l102_10238

theorem repeating_decimal_sum_diff_prod : 
  let repeating_decimal (n : ℕ) := n / 9
  (repeating_decimal 6) + (repeating_decimal 2) - (repeating_decimal 4) * (repeating_decimal 3) = 20 / 27 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_prod_l102_10238


namespace NUMINAMATH_CALUDE_base7_subtraction_l102_10212

/-- Represents a number in base 7 as a list of digits (least significant first) -/
def Base7 := List Nat

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : Base7) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers -/
def base7Difference (a b : Base7) : Base7 :=
  sorry -- Implementation not required for the statement

/-- Statement: The difference between 4512₇ and 2345₇ in base 7 is 2144₇ -/
theorem base7_subtraction :
  base7Difference [2, 1, 5, 4] [5, 4, 3, 2] = [4, 4, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base7_subtraction_l102_10212


namespace NUMINAMATH_CALUDE_ratio_problem_l102_10278

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a + b) / (b + c) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l102_10278


namespace NUMINAMATH_CALUDE_average_lifespan_of_sampled_products_l102_10210

/-- Represents a factory producing electronic products -/
structure Factory where
  production_ratio : ℚ
  average_lifespan : ℚ

/-- Calculates the weighted average lifespan of products from multiple factories -/
def weighted_average_lifespan (factories : List Factory) (total_samples : ℕ) : ℚ :=
  let total_ratio := factories.map (λ f => f.production_ratio) |>.sum
  let weighted_sum := factories.map (λ f => f.production_ratio * f.average_lifespan) |>.sum
  weighted_sum / total_ratio

/-- The main theorem proving the average lifespan of sampled products -/
theorem average_lifespan_of_sampled_products : 
  let factories := [
    { production_ratio := 1, average_lifespan := 980 },
    { production_ratio := 2, average_lifespan := 1020 },
    { production_ratio := 1, average_lifespan := 1032 }
  ]
  let total_samples := 100
  weighted_average_lifespan factories total_samples = 1013 := by
  sorry

end NUMINAMATH_CALUDE_average_lifespan_of_sampled_products_l102_10210


namespace NUMINAMATH_CALUDE_zero_point_condition_sufficient_condition_not_necessary_condition_l102_10248

def f (a : ℝ) (x : ℝ) := a * x + 3

theorem zero_point_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 2, f a x ≠ 0) ∨
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0) :=
by sorry

theorem sufficient_condition (a : ℝ) (h : a < -3) :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0 :=
by sorry

theorem not_necessary_condition :
  ∃ a ≥ -3, ∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_point_condition_sufficient_condition_not_necessary_condition_l102_10248


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l102_10245

theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (6 * w = 3 * (2 * w^2)) → w = 1 ∧ 2 * w = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l102_10245


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l102_10200

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 ≥ -3 ∧ -2 * (x + 3) > 0}
  S = {x : ℝ | -4 ≤ x ∧ x < -3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l102_10200


namespace NUMINAMATH_CALUDE_inequality_proof_l102_10233

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l102_10233


namespace NUMINAMATH_CALUDE_decimal_20_equals_base4_110_l102_10292

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem: The decimal number 20 is equivalent to 110 in base 4 -/
theorem decimal_20_equals_base4_110 : toBase4 20 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_20_equals_base4_110_l102_10292


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l102_10256

theorem gcf_lcm_sum_8_12 : 
  Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l102_10256


namespace NUMINAMATH_CALUDE_allocation_methods_for_three_schools_l102_10213

/-- The number of ways to allocate doctors and nurses to schools. -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.factorial) * (num_nurses.choose 2 * (num_nurses - 2).choose 2)

/-- Theorem stating that there are 540 different allocation methods for 3 doctors and 6 nurses to 3 schools. -/
theorem allocation_methods_for_three_schools :
  allocation_methods 3 6 3 = 540 := by
  sorry

#eval allocation_methods 3 6 3

end NUMINAMATH_CALUDE_allocation_methods_for_three_schools_l102_10213


namespace NUMINAMATH_CALUDE_inequality_proof_l102_10269

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l102_10269


namespace NUMINAMATH_CALUDE_centroid_equal_areas_l102_10208

/-- The centroid of a triangle divides it into three equal-area triangles -/
theorem centroid_equal_areas (P Q R S : ℝ × ℝ) : 
  P = (-1, 3) → Q = (2, 7) → R = (4, 0) → 
  S.1 = (P.1 + Q.1 + R.1) / 3 → 
  S.2 = (P.2 + Q.2 + R.2) / 3 → 
  8 * S.1 + 3 * S.2 = 70 / 3 := by
  sorry

#check centroid_equal_areas

end NUMINAMATH_CALUDE_centroid_equal_areas_l102_10208


namespace NUMINAMATH_CALUDE_certain_number_problem_l102_10298

theorem certain_number_problem (N x : ℝ) (h1 : N / x * 2 = 12) (h2 : x = 0.1) : N = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l102_10298


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l102_10234

theorem subtraction_of_fractions : 
  (2 + 1/4) - 2/3 = 1 + 7/12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l102_10234


namespace NUMINAMATH_CALUDE_sum_odd_numbers_less_than_20_l102_10291

theorem sum_odd_numbers_less_than_20 : 
  (Finset.range 10).sum (fun n => 2 * n + 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_less_than_20_l102_10291


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l102_10259

theorem fraction_to_decimal :
  (53 : ℚ) / (4 * 5^7) = (1325 : ℚ) / 10^7 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l102_10259


namespace NUMINAMATH_CALUDE_green_pill_cost_l102_10295

theorem green_pill_cost (weeks : ℕ) (daily_green : ℕ) (daily_pink : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_green = 1 →
  daily_pink = 1 →
  green_pink_diff = 3 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 21 ∧ 
    (weeks * 7 * (green_cost + (green_cost - green_pink_diff))) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l102_10295


namespace NUMINAMATH_CALUDE_green_balloons_l102_10280

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by sorry

end NUMINAMATH_CALUDE_green_balloons_l102_10280


namespace NUMINAMATH_CALUDE_todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l102_10262

theorem todds_initial_gum_pieces : ℝ → Prop :=
  fun x =>
    let additional_pieces : ℝ := 150
    let percentage_increase : ℝ := 0.25
    let final_total : ℝ := 890
    (x + additional_pieces = final_total) ∧
    (additional_pieces = percentage_increase * x) →
    x = 712

-- The proof is omitted
theorem todds_initial_gum_pieces_proof : todds_initial_gum_pieces 712 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l102_10262


namespace NUMINAMATH_CALUDE_percentage_of_75_to_125_l102_10263

theorem percentage_of_75_to_125 : (75 : ℝ) / 125 * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_75_to_125_l102_10263
