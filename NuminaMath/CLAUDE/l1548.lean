import Mathlib

namespace NUMINAMATH_CALUDE_shopping_mall_desk_lamps_l1548_154894

/-- Shopping mall desk lamp problem -/
theorem shopping_mall_desk_lamps :
  let total_lamps : ℕ := 50
  let total_cost : ℕ := 2500
  let cost_a : ℕ := 40
  let cost_b : ℕ := 65
  let marked_a : ℕ := 60
  let marked_b : ℕ := 100
  let discount_a : ℚ := 1 - 10 / 100
  let discount_b : ℚ := 1 - 30 / 100
  
  let num_a : ℕ := 30
  let num_b : ℕ := total_lamps - num_a
  
  (num_a * cost_a + num_b * cost_b = total_cost) ∧
  (num_a + num_b = total_lamps) ∧
  (num_a * (marked_a * discount_a - cost_a) + num_b * (marked_b * discount_b - cost_b) = 520) :=
by
  sorry


end NUMINAMATH_CALUDE_shopping_mall_desk_lamps_l1548_154894


namespace NUMINAMATH_CALUDE_baby_hippos_per_female_l1548_154836

theorem baby_hippos_per_female (initial_elephants initial_hippos total_animals : ℕ)
  (female_hippo_ratio : ℚ) :
  initial_elephants = 20 →
  initial_hippos = 35 →
  female_hippo_ratio = 5 / 7 →
  total_animals = 315 →
  ∃ (baby_hippos_per_female : ℕ),
    baby_hippos_per_female = 5 ∧
    total_animals = initial_elephants + initial_hippos +
      (female_hippo_ratio * initial_hippos).num * baby_hippos_per_female +
      ((female_hippo_ratio * initial_hippos).num * baby_hippos_per_female + 10) :=
by
  sorry

end NUMINAMATH_CALUDE_baby_hippos_per_female_l1548_154836


namespace NUMINAMATH_CALUDE_k_range_when_proposition_p_false_l1548_154829

theorem k_range_when_proposition_p_false (k : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, k * 4^x - k * 2^(x + 1) + 6 * (k - 5) ≠ 0) →
  k ∈ Set.Iio 5 ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_k_range_when_proposition_p_false_l1548_154829


namespace NUMINAMATH_CALUDE_test_average_l1548_154834

theorem test_average (male_count : ℕ) (female_count : ℕ) 
  (male_avg : ℝ) (female_avg : ℝ) : 
  male_count = 8 → 
  female_count = 32 → 
  male_avg = 82 → 
  female_avg = 92 → 
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end NUMINAMATH_CALUDE_test_average_l1548_154834


namespace NUMINAMATH_CALUDE_fence_cost_l1548_154897

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 55) :
  4 * price_per_foot * Real.sqrt area = 3740 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l1548_154897


namespace NUMINAMATH_CALUDE_paint_usage_l1548_154895

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) :
  total_paint = 360 ∧ 
  first_week_fraction = 1/4 ∧ 
  second_week_fraction = 1/2 →
  first_week_fraction * total_paint + 
  second_week_fraction * (total_paint - first_week_fraction * total_paint) = 225 :=
by sorry

end NUMINAMATH_CALUDE_paint_usage_l1548_154895


namespace NUMINAMATH_CALUDE_total_cost_of_promotional_items_l1548_154860

/-- The cost of a calendar in dollars -/
def calendar_cost : ℚ := 3/4

/-- The cost of a date book in dollars -/
def date_book_cost : ℚ := 1/2

/-- The number of calendars ordered -/
def calendars_ordered : ℕ := 300

/-- The number of date books ordered -/
def date_books_ordered : ℕ := 200

/-- The total number of items ordered -/
def total_items : ℕ := 500

/-- Theorem stating the total cost of promotional items -/
theorem total_cost_of_promotional_items :
  calendars_ordered * calendar_cost + date_books_ordered * date_book_cost = 325/1 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_of_promotional_items_l1548_154860


namespace NUMINAMATH_CALUDE_fred_remaining_cards_l1548_154831

/-- Given that Fred initially had 40 baseball cards and Keith bought 22 of them,
    prove that Fred now has 18 baseball cards. -/
theorem fred_remaining_cards (initial_cards : ℕ) (cards_bought : ℕ) (h1 : initial_cards = 40) (h2 : cards_bought = 22) :
  initial_cards - cards_bought = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_remaining_cards_l1548_154831


namespace NUMINAMATH_CALUDE_count_valid_digits_l1548_154871

def is_valid_digit (n : ℕ) : Prop :=
  n < 10

def appended_number (n : ℕ) : ℕ :=
  7580 + n

theorem count_valid_digits :
  ∃ (valid_digits : Finset ℕ),
    (∀ d ∈ valid_digits, is_valid_digit d ∧ (appended_number d).mod 4 = 0) ∧
    (∀ d, is_valid_digit d ∧ (appended_number d).mod 4 = 0 → d ∈ valid_digits) ∧
    valid_digits.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_digits_l1548_154871


namespace NUMINAMATH_CALUDE_installation_cost_calculation_l1548_154811

/-- Calculates the installation cost given the purchase details of a refrigerator. -/
theorem installation_cost_calculation
  (purchase_price_after_discount : ℝ)
  (discount_rate : ℝ)
  (transport_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price_after_discount = 12500)
  (h2 : discount_rate = 0.20)
  (h3 : transport_cost = 125)
  (h4 : selling_price = 18400)
  (h5 : selling_price = 1.15 * (purchase_price_after_discount + transport_cost + installation_cost)) :
  installation_cost = 3375 :=
by sorry


end NUMINAMATH_CALUDE_installation_cost_calculation_l1548_154811


namespace NUMINAMATH_CALUDE_cards_per_layer_calculation_l1548_154823

def number_of_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def number_of_layers : ℕ := 32

def total_cards : ℕ := number_of_decks * cards_per_deck

theorem cards_per_layer_calculation :
  total_cards / number_of_layers = 26 := by sorry

end NUMINAMATH_CALUDE_cards_per_layer_calculation_l1548_154823


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1548_154875

theorem complex_fraction_evaluation : 
  (2 / (3 + 1/5) + ((3 + 1/4) / 13) / (2/3) + (2 + 5/18 - 17/36) * (18/65)) * (1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1548_154875


namespace NUMINAMATH_CALUDE_max_regions_five_lines_l1548_154838

/-- The number of regions created by n intersecting lines in a plane --/
def num_regions (n : ℕ) : ℕ := sorry

/-- The maximum number of regions created by n intersecting lines in a rectangle --/
def max_regions_rectangle (n : ℕ) : ℕ := num_regions n

theorem max_regions_five_lines : 
  max_regions_rectangle 5 = 16 := by sorry

end NUMINAMATH_CALUDE_max_regions_five_lines_l1548_154838


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1500_l1548_154887

def hulk_jump (n : ℕ) : ℝ := 2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_1500 :
  ∀ k < 8, hulk_jump k ≤ 1500 ∧ hulk_jump 8 > 1500 := by sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1500_l1548_154887


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1548_154808

theorem inscribed_circle_radius 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 5) 
  (hc : c = 7) 
  (h_area : (a + b + c) / 2 - 2 = (a + b + c) / 2 * r) : 
  r = 1.8 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1548_154808


namespace NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l1548_154858

open Real

/-- The function f parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

/-- The function g -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The theorem statement -/
theorem f_less_than_g_implies_a_bound 
  (h : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) : 
  a > log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l1548_154858


namespace NUMINAMATH_CALUDE_binomial_prime_divisors_l1548_154891

theorem binomial_prime_divisors (k : ℕ+) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → (Nat.choose n k.val).factors.card ≥ k.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_prime_divisors_l1548_154891


namespace NUMINAMATH_CALUDE_unique_solution_for_k_l1548_154813

/-- The equation has exactly one solution when k = -3/4 -/
theorem unique_solution_for_k (k : ℝ) : 
  (∃! x, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_l1548_154813


namespace NUMINAMATH_CALUDE_major_axis_length_l1548_154869

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- State the theorem
theorem major_axis_length :
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧
  (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * a = 6 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_l1548_154869


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l1548_154853

theorem max_y_over_x_on_circle :
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - Real.sqrt 3)^2 = 3}
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → p.2 / p.1 ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l1548_154853


namespace NUMINAMATH_CALUDE_sequence_inequality_l1548_154800

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1

theorem sequence_inequality (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → |a p / p - a q / q| < 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1548_154800


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_and_perimeter_l1548_154876

/-- A rectangle with integer sides where the area equals twice the perimeter -/
structure SpecialRectangle where
  a : ℕ
  b : ℕ
  h1 : a ≠ b
  h2 : a * b = 2 * (2 * a + 2 * b)

theorem special_rectangle_dimensions_and_perimeter (rect : SpecialRectangle) :
  (rect.a = 12 ∧ rect.b = 6) ∨ (rect.a = 6 ∧ rect.b = 12) ∧
  2 * (rect.a + rect.b) = 36 := by
  sorry

#check special_rectangle_dimensions_and_perimeter

end NUMINAMATH_CALUDE_special_rectangle_dimensions_and_perimeter_l1548_154876


namespace NUMINAMATH_CALUDE_fraction_equality_l1548_154820

theorem fraction_equality (a b : ℝ) (h : ((1/a) + (1/b)) / ((1/a) - (1/b)) = 2020) : 
  (a + b) / (a - b) = 2020 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1548_154820


namespace NUMINAMATH_CALUDE_am_gm_inequality_l1548_154888

theorem am_gm_inequality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  ((c + d) / 2 - Real.sqrt (c * d)) < (d - c)^3 / (8 * c) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l1548_154888


namespace NUMINAMATH_CALUDE_max_value_d_l1548_154846

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_d_l1548_154846


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1548_154840

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 →
  area = side^2 →
  perimeter = 4 * side →
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1548_154840


namespace NUMINAMATH_CALUDE_range_of_x_given_integer_part_l1548_154893

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the theorem
theorem range_of_x_given_integer_part (x : ℝ) :
  integerPart ((1 - 3*x) / 2) = -1 → 1/3 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_given_integer_part_l1548_154893


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1548_154827

/-- Given a function f where f(x) = f'(π/4) * cos(x) + sin(x), prove that f(π/4) = 1 -/
theorem function_value_at_pi_over_four (f : ℝ → ℝ) 
  (h : ∀ x, f x = (deriv f (π/4)) * Real.cos x + Real.sin x) : 
  f (π/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1548_154827


namespace NUMINAMATH_CALUDE_jane_drawing_paper_l1548_154862

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper :
  total_sheets = 55 := by sorry

end NUMINAMATH_CALUDE_jane_drawing_paper_l1548_154862


namespace NUMINAMATH_CALUDE_number_with_special_average_l1548_154843

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
sorry

end NUMINAMATH_CALUDE_number_with_special_average_l1548_154843


namespace NUMINAMATH_CALUDE_binary_conversion_and_subtraction_l1548_154816

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binaryNumber : List Bool := [true, false, true, true, false, true]

/-- The main theorem to prove -/
theorem binary_conversion_and_subtraction :
  (binaryToDecimal binaryNumber) - 5 = 40 := by sorry

end NUMINAMATH_CALUDE_binary_conversion_and_subtraction_l1548_154816


namespace NUMINAMATH_CALUDE_three_number_product_l1548_154822

theorem three_number_product (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_product_l1548_154822


namespace NUMINAMATH_CALUDE_continuous_midpoint_property_implies_affine_l1548_154868

open Real

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that a continuous function with the midpoint property is affine -/
theorem continuous_midpoint_property_implies_affine
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ c b : ℝ, ∀ x, f x = c * x + b := by
  sorry

end NUMINAMATH_CALUDE_continuous_midpoint_property_implies_affine_l1548_154868


namespace NUMINAMATH_CALUDE_samson_schedule_solution_l1548_154812

/-- Utility function -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := 2 * math * frisbee

/-- Wednesday's utility -/
def wednesday_utility (s : ℝ) : ℝ := utility (10 - 2*s) s

/-- Thursday's utility -/
def thursday_utility (s : ℝ) : ℝ := utility (2*s + 4) (3 - s)

/-- The theorem stating that s = 2 is the unique solution -/
theorem samson_schedule_solution :
  ∃! s : ℝ, wednesday_utility s = thursday_utility s ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_samson_schedule_solution_l1548_154812


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1548_154878

/-- Proves that in a class of 27 students with 15 girls, the ratio of boys to girls is 4:5 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) 
  (h1 : total_students = 27) 
  (h2 : girls = 15) : 
  (total_students - girls) * 5 = girls * 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1548_154878


namespace NUMINAMATH_CALUDE_solve_exponential_system_l1548_154807

theorem solve_exponential_system (x y : ℝ) 
  (h1 : (6 : ℝ) ^ (x + y) = 36)
  (h2 : (6 : ℝ) ^ (x + 5 * y) = 216) :
  x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_system_l1548_154807


namespace NUMINAMATH_CALUDE_dante_remaining_coconuts_l1548_154833

/-- Paolo's number of coconuts -/
def paolo_coconuts : ℕ := 14

/-- Dante's initial number of coconuts in terms of Paolo's -/
def dante_initial_coconuts : ℕ := 3 * paolo_coconuts

/-- Number of coconuts Dante sold -/
def dante_sold_coconuts : ℕ := 10

/-- Theorem: Dante has 32 coconuts left after selling -/
theorem dante_remaining_coconuts : 
  dante_initial_coconuts - dante_sold_coconuts = 32 := by sorry

end NUMINAMATH_CALUDE_dante_remaining_coconuts_l1548_154833


namespace NUMINAMATH_CALUDE_marbles_gcd_l1548_154826

theorem marbles_gcd (blue : Nat) (white : Nat) (red : Nat) (green : Nat) (yellow : Nat)
  (h_blue : blue = 24)
  (h_white : white = 17)
  (h_red : red = 13)
  (h_green : green = 7)
  (h_yellow : yellow = 5) :
  Nat.gcd blue (Nat.gcd white (Nat.gcd red (Nat.gcd green yellow))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_marbles_gcd_l1548_154826


namespace NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_purely_imaginary_z_not_in_second_quadrant_l1548_154835

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 5*m + 6) (m^2 - 3*m)

-- 1. z is real iff m = 0 or m = 3
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 0 ∨ m = 3 := by sorry

-- 2. z is complex iff m ≠ 0 and m ≠ 3
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 0 ∧ m ≠ 3 := by sorry

-- 3. z is purely imaginary iff m = 2
theorem z_is_purely_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := by sorry

-- 4. z is never in the second quadrant
theorem z_not_in_second_quadrant (m : ℝ) : ¬((z m).re < 0 ∧ (z m).im > 0) := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_purely_imaginary_z_not_in_second_quadrant_l1548_154835


namespace NUMINAMATH_CALUDE_infinitely_many_satisfy_property_l1548_154863

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property that n divides F_{F_n} but not F_n -/
def satisfies_property (n : ℕ) : Prop :=
  n > 0 ∧ (n ∣ fib (fib n)) ∧ ¬(n ∣ fib n)

theorem infinitely_many_satisfy_property :
  ∀ k : ℕ, k > 0 → satisfies_property (12 * k) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfy_property_l1548_154863


namespace NUMINAMATH_CALUDE_inspection_probability_l1548_154828

theorem inspection_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90)
  (h2 : pass_rate2 = 0.95) :
  let fail_rate1 := 1 - pass_rate1
  let fail_rate2 := 1 - pass_rate2
  pass_rate1 * fail_rate2 + fail_rate1 * pass_rate2 = 0.14 :=
by sorry

end NUMINAMATH_CALUDE_inspection_probability_l1548_154828


namespace NUMINAMATH_CALUDE_tom_younger_than_bob_by_three_l1548_154832

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- The age difference between Bob and Tom -/
def ageDifference (ages : SiblingAges) : ℕ :=
  ages.bob - ages.tom

theorem tom_younger_than_bob_by_three (ages : SiblingAges) 
  (susan_age : ages.susan = 15)
  (arthur_age : ages.arthur = ages.susan + 2)
  (bob_age : ages.bob = 11)
  (total_age : ages.susan + ages.arthur + ages.tom + ages.bob = 51) :
  ageDifference ages = 3 := by
sorry

end NUMINAMATH_CALUDE_tom_younger_than_bob_by_three_l1548_154832


namespace NUMINAMATH_CALUDE_factor_implies_m_equals_one_l1548_154899

theorem factor_implies_m_equals_one (m : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 42 = (x + 6) * k) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_m_equals_one_l1548_154899


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1548_154815

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.95 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1548_154815


namespace NUMINAMATH_CALUDE_intersection_property_l1548_154841

/-- The curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = m

/-- The theorem statement -/
theorem intersection_property (m : ℝ) :
  m > 0 →
  ∃ (ρ_A ρ_M ρ_N : ℝ),
    line_l ρ_A (π/4) m ∧
    curve_C ρ_M (π/4) ∧
    curve_C ρ_N (π/4) ∧
    ρ_A * ρ_M * ρ_N = 6 →
  m = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_property_l1548_154841


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1548_154885

theorem integer_sum_problem : ∃ (a b : ℕ+), 
  (a * b + a + b = 167) ∧ 
  (Nat.gcd a.val b.val = 1) ∧ 
  (a < 30) ∧ (b < 30) ∧ 
  (a + b = 24) := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1548_154885


namespace NUMINAMATH_CALUDE_system_solution_l1548_154861

theorem system_solution (x y : ℝ) 
  (eq1 : 4 * x - y = 2) 
  (eq2 : 3 * x - 2 * y = -1) : 
  x - y = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1548_154861


namespace NUMINAMATH_CALUDE_legislation_approval_probability_l1548_154872

/-- The probability of a voter approving the legislation -/
def p_approve : ℝ := 0.6

/-- The number of voters surveyed -/
def n : ℕ := 4

/-- The number of approving voters we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k out of n voters approving the legislation -/
def prob_k_approve (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem legislation_approval_probability :
  prob_k_approve p_approve n k = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_legislation_approval_probability_l1548_154872


namespace NUMINAMATH_CALUDE_goats_in_field_l1548_154877

theorem goats_in_field (total : Nat) (cows : Nat) (sheep : Nat) (chickens : Nat) 
  (h1 : total = 900)
  (h2 : cows = 250)
  (h3 : sheep = 310)
  (h4 : chickens = 180) :
  total - (cows + sheep + chickens) = 160 := by
  sorry

end NUMINAMATH_CALUDE_goats_in_field_l1548_154877


namespace NUMINAMATH_CALUDE_montero_trip_feasibility_l1548_154844

/-- Represents the parameters of Mr. Montero's trip -/
structure TripParameters where
  normal_efficiency : Real
  traffic_efficiency_reduction : Real
  total_distance : Real
  traffic_distance : Real
  initial_gas : Real
  gas_price : Real
  price_increase : Real
  budget : Real

/-- Calculates whether Mr. Montero can complete his trip within budget -/
def can_complete_trip (params : TripParameters) : Prop :=
  let reduced_efficiency := params.normal_efficiency * (1 - params.traffic_efficiency_reduction)
  let normal_distance := params.total_distance - params.traffic_distance
  let gas_needed := normal_distance / params.normal_efficiency + 
                    params.traffic_distance / reduced_efficiency
  let gas_to_buy := gas_needed - params.initial_gas
  let half_trip_gas := (params.total_distance / 2) / params.normal_efficiency - params.initial_gas
  let first_half_cost := min half_trip_gas gas_to_buy * params.gas_price
  let second_half_cost := max 0 (gas_to_buy - half_trip_gas) * (params.gas_price * (1 + params.price_increase))
  first_half_cost + second_half_cost ≤ params.budget

theorem montero_trip_feasibility :
  let params : TripParameters := {
    normal_efficiency := 20,
    traffic_efficiency_reduction := 0.2,
    total_distance := 600,
    traffic_distance := 100,
    initial_gas := 8,
    gas_price := 2.5,
    price_increase := 0.1,
    budget := 75
  }
  can_complete_trip params := by sorry

end NUMINAMATH_CALUDE_montero_trip_feasibility_l1548_154844


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l1548_154805

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (t1 : ℝ) (v1 : ℝ) (t2 : ℝ) (v2 : ℝ) 
  (h1 : t1 = 5) 
  (h2 : v1 = 40) 
  (h3 : t2 = 3) 
  (h4 : v2 = 80) : 
  (t1 * v1 + t2 * v2) / (t1 + t2) = 55 := by
  sorry

#check two_part_journey_average_speed

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l1548_154805


namespace NUMINAMATH_CALUDE_divisibility_by_three_l1548_154855

theorem divisibility_by_three (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A ^ 2 + B ^ 2 = A * B)
  (h2 : IsUnit (B * A - A * B)) :
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l1548_154855


namespace NUMINAMATH_CALUDE_abc_value_l1548_154814

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 90 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1548_154814


namespace NUMINAMATH_CALUDE_remainder_equality_l1548_154837

theorem remainder_equality (A B C S T s t : ℕ) 
  (h1 : A > B)
  (h2 : A^2 % C = S)
  (h3 : B^2 % C = T)
  (h4 : (A^2 * B^2) % C = s)
  (h5 : (S * T) % C = t) :
  s = t := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l1548_154837


namespace NUMINAMATH_CALUDE_tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l1548_154879

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x + 1

-- Theorem for the tangent line when a = 0
theorem tangent_line_a_zero :
  ∀ x y : ℝ, f 0 1 = 1 → (2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1)) := by sorry

-- Theorem for the maximum value of g when a > 0
theorem max_value_g_positive_a :
  ∀ a : ℝ, a > 0 → ∃ max_val : ℝ, max_val = g a (1/a) ∧ 
  ∀ x : ℝ, x > 0 → g a x ≤ max_val := by sorry

-- Theorem for the inequality when a = -2
theorem inequality_a_negative_two :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
  f (-2) x₁ + f (-2) x₂ + x₁ * x₂ = 0 → 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l1548_154879


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l1548_154810

theorem dog_paws_on_ground (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_legs : ℕ) : 
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_legs = total_dogs / 2 →
  dogs_on_back_legs * 2 + dogs_on_all_legs * 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_paws_on_ground_l1548_154810


namespace NUMINAMATH_CALUDE_max_sides_of_special_polygon_existence_of_five_sided_polygon_l1548_154809

-- Define a convex polygon
def ConvexPolygon (n : ℕ) := Unit

-- Define a property that a polygon has at least one side of length 1
def HasSideOfLengthOne (p : ConvexPolygon n) : Prop := sorry

-- Define a property that all diagonals of a polygon have integer lengths
def AllDiagonalsInteger (p : ConvexPolygon n) : Prop := sorry

-- State the theorem
theorem max_sides_of_special_polygon :
  ∀ n : ℕ, n > 5 →
  ¬∃ (p : ConvexPolygon n), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

theorem existence_of_five_sided_polygon :
  ∃ (p : ConvexPolygon 5), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

end NUMINAMATH_CALUDE_max_sides_of_special_polygon_existence_of_five_sided_polygon_l1548_154809


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l1548_154856

theorem divisors_of_8_factorial (n : ℕ) : n = 8 → (Finset.card (Nat.divisors (Nat.factorial n))) = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l1548_154856


namespace NUMINAMATH_CALUDE_euler_line_equation_l1548_154865

/-- Triangle ABC with vertices A(1,3) and B(2,1), and |AC| = |BC| -/
structure Triangle :=
  (C : ℝ × ℝ)
  (ac_eq_bc : (C.1 - 1)^2 + (C.2 - 3)^2 = (C.2 - 2)^2 + (C.2 - 1)^2)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0}

/-- Theorem: The Euler line of triangle ABC is 2x - 4y + 5 = 0 -/
theorem euler_line_equation (t : Triangle) : 
  EulerLine t = {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0} := by
  sorry

end NUMINAMATH_CALUDE_euler_line_equation_l1548_154865


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l1548_154839

/-- The number of cards in one dozen -/
def cards_per_dozen : ℕ := 12

/-- The number of dozens each person has -/
def dozens_per_person : ℕ := 9

/-- The number of people -/
def number_of_people : ℕ := 4

/-- Theorem: The total number of Pokemon cards owned by 4 people, each having 9 dozen cards, is equal to 432 -/
theorem total_pokemon_cards : 
  (cards_per_dozen * dozens_per_person * number_of_people) = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l1548_154839


namespace NUMINAMATH_CALUDE_solve_star_equation_l1548_154864

-- Define the * operation
def star_op (a b : ℚ) : ℚ :=
  if a ≥ b then a^2 * b else a * b^2

-- Theorem statement
theorem solve_star_equation :
  ∃! m : ℚ, star_op 3 m = 48 ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1548_154864


namespace NUMINAMATH_CALUDE_larger_tart_flour_usage_l1548_154870

theorem larger_tart_flour_usage
  (small_tarts : ℕ)
  (large_tarts : ℕ)
  (small_flour : ℚ)
  (h1 : small_tarts = 50)
  (h2 : large_tarts = 25)
  (h3 : small_flour = 1 / 8)
  (h4 : small_tarts * small_flour = large_tarts * large_flour) :
  large_flour = 1 / 4 :=
by
  sorry

#check larger_tart_flour_usage

end NUMINAMATH_CALUDE_larger_tart_flour_usage_l1548_154870


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1548_154804

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1548_154804


namespace NUMINAMATH_CALUDE_pythagorean_triple_has_even_number_l1548_154896

theorem pythagorean_triple_has_even_number (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  Even a ∨ Even b ∨ Even c := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_has_even_number_l1548_154896


namespace NUMINAMATH_CALUDE_triangle_height_equals_30_l1548_154854

/-- Given a rectangle with perimeter 60 cm and a right triangle with base 15 cm,
    if their areas are equal, then the height of the triangle is 30 cm. -/
theorem triangle_height_equals_30 (rectangle_perimeter : ℝ) (triangle_base : ℝ) (h : ℝ) :
  rectangle_perimeter = 60 →
  triangle_base = 15 →
  (rectangle_perimeter / 4) * (rectangle_perimeter / 4) = (1 / 2) * triangle_base * h →
  h = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_equals_30_l1548_154854


namespace NUMINAMATH_CALUDE_intersection_point_correct_l1548_154884

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space defined by a point and a direction vector -/
structure Line2D where
  point : Point2D
  direction : Point2D

/-- The first line -/
def line1 : Line2D := {
  point := { x := 3, y := 0 },
  direction := { x := 1, y := 2 }
}

/-- The second line -/
def line2 : Line2D := {
  point := { x := -1, y := 4 },
  direction := { x := 3, y := -1 }
}

/-- The proposed intersection point -/
def intersectionPoint : Point2D := {
  x := 30 / 7,
  y := 18 / 7
}

/-- Function to check if a point lies on a line -/
def isPointOnLine (p : Point2D) (l : Line2D) : Prop :=
  ∃ t : ℚ, p.x = l.point.x + t * l.direction.x ∧ p.y = l.point.y + t * l.direction.y

/-- Theorem stating that the proposed intersection point lies on both lines -/
theorem intersection_point_correct :
  isPointOnLine intersectionPoint line1 ∧ isPointOnLine intersectionPoint line2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l1548_154884


namespace NUMINAMATH_CALUDE_calvin_haircut_goal_l1548_154890

/-- Calculate the percentage of progress towards a goal -/
def progressPercentage (completed : ℕ) (total : ℕ) : ℚ :=
  (completed : ℚ) / (total : ℚ) * 100

/-- Calvin's haircut goal problem -/
theorem calvin_haircut_goal :
  let total_haircuts : ℕ := 10
  let completed_haircuts : ℕ := 8
  progressPercentage completed_haircuts total_haircuts = 80 := by
  sorry

end NUMINAMATH_CALUDE_calvin_haircut_goal_l1548_154890


namespace NUMINAMATH_CALUDE_fifteen_customers_tipped_l1548_154867

/-- Calculates the number of customers who left a tip --/
def customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : ℕ :=
  initial_customers + additional_customers - non_tipping_customers

/-- Theorem: Given the conditions, prove that 15 customers left a tip --/
theorem fifteen_customers_tipped :
  customers_who_tipped 29 20 34 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_customers_tipped_l1548_154867


namespace NUMINAMATH_CALUDE_instantaneous_speed_at_4_l1548_154882

-- Define the motion equation
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the instantaneous speed (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem: The instantaneous speed at t = 4 is 6 m/s
theorem instantaneous_speed_at_4 : v 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_4_l1548_154882


namespace NUMINAMATH_CALUDE_seating_arrangements_l1548_154821

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial n - 2 * (factorial (n - 2) * factorial k) + factorial (n - 2) * (factorial k)^2

theorem seating_arrangements :
  arrangements 10 3 = 4596480 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1548_154821


namespace NUMINAMATH_CALUDE_round_trip_percentage_proof_l1548_154830

/-- The percentage of passengers with round-trip tickets -/
def round_trip_percentage : ℝ := 25

/-- The percentage of all passengers who have round-trip tickets and took their cars aboard -/
def round_trip_with_car_percentage : ℝ := 20

/-- The percentage of passengers with round-trip tickets who did not take their cars aboard -/
def round_trip_without_car_percentage : ℝ := 20

theorem round_trip_percentage_proof :
  round_trip_percentage = 25 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_percentage_proof_l1548_154830


namespace NUMINAMATH_CALUDE_permutations_5_3_l1548_154842

/-- The number of permutations of k elements chosen from n elements -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations A_5^3 equals 60 -/
theorem permutations_5_3 : permutations 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_permutations_5_3_l1548_154842


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1548_154802

theorem digit_sum_theorem (A B C D : ℕ) : 
  A ≠ 0 →
  A < 10 → B < 10 → C < 10 → D < 10 →
  1000 * A + 100 * B + 10 * C + D = (10 * C + D)^2 - (10 * A + B)^2 →
  A + B + C + D = 21 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1548_154802


namespace NUMINAMATH_CALUDE_valid_sequences_count_l1548_154898

/-- Represents a binary sequence with no consecutive 1s -/
inductive ValidSequence : Nat → Type
  | zero : ValidSequence 0
  | one : ValidSequence 1
  | appendZero : ValidSequence n → ValidSequence (n + 1)
  | appendOneZero : ValidSequence n → ValidSequence (n + 2)

/-- Counts the number of valid sequences of length n or less -/
def countValidSequences (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => countValidSequences (n+1) + countValidSequences n

theorem valid_sequences_count :
  countValidSequences 11 = 233 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l1548_154898


namespace NUMINAMATH_CALUDE_binary_to_decimal_conversion_l1548_154881

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (digits : List Bool) : ℕ :=
  digits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary representation of the number we want to convert. -/
def binaryNumber : List Bool :=
  [true, true, true, false, true, true, false, false, true, false, false, true]

theorem binary_to_decimal_conversion :
  binaryToNat binaryNumber = 3785 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_conversion_l1548_154881


namespace NUMINAMATH_CALUDE_largest_k_value_l1548_154806

/-- Triangle side lengths are positive real numbers that satisfy the triangle inequality --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

/-- The inequality holds for all triangles --/
def inequality_holds (k : ℝ) : Prop :=
  ∀ t : Triangle, (t.a + t.b + t.c)^3 ≥ (5/2) * (t.a^3 + t.b^3 + t.c^3) + k * t.a * t.b * t.c

/-- 39/2 is the largest real number satisfying the inequality --/
theorem largest_k_value : 
  (∀ k : ℝ, k > 39/2 → ¬(inequality_holds k)) ∧ 
  inequality_holds (39/2) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_value_l1548_154806


namespace NUMINAMATH_CALUDE_modular_difference_in_range_l1548_154859

def problem (a b : ℤ) : Prop :=
  a % 36 = 22 ∧ b % 36 = 85

def valid_range (n : ℤ) : Prop :=
  120 ≤ n ∧ n ≤ 161

theorem modular_difference_in_range (a b : ℤ) (h : problem a b) :
  ∃! n : ℤ, valid_range n ∧ (a - b) % 36 = n % 36 ∧ n = 153 := by sorry

end NUMINAMATH_CALUDE_modular_difference_in_range_l1548_154859


namespace NUMINAMATH_CALUDE_extremum_properties_l1548_154889

noncomputable section

variable (x : ℝ)

def f (x : ℝ) : ℝ := x * Real.log x + (1/2) * x^2

def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, x > 0 → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_properties (x₀ : ℝ) 
  (h₁ : x₀ > 0) 
  (h₂ : is_extremum_point f x₀) : 
  (0 < x₀ ∧ x₀ < Real.exp (-1)) ∧ 
  (f x₀ + x₀ < 0) := by
  sorry

end

end NUMINAMATH_CALUDE_extremum_properties_l1548_154889


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l1548_154825

theorem stadium_fee_difference (capacity : ℕ) (fee : ℕ) (h1 : capacity = 2000) (h2 : fee = 20) :
  capacity * fee - (3 * capacity / 4) * fee = 10000 := by
  sorry

end NUMINAMATH_CALUDE_stadium_fee_difference_l1548_154825


namespace NUMINAMATH_CALUDE_full_price_revenue_l1548_154845

def total_tickets : ℕ := 180
def total_revenue : ℕ := 2400

def ticket_revenue (full_price : ℕ) (num_full_price : ℕ) : Prop :=
  ∃ (half_price : ℕ),
    half_price = full_price / 2 ∧
    num_full_price + (total_tickets - num_full_price) = total_tickets ∧
    num_full_price * full_price + (total_tickets - num_full_price) * half_price = total_revenue

theorem full_price_revenue : 
  ∃ (full_price : ℕ) (num_full_price : ℕ), 
    ticket_revenue full_price num_full_price ∧ 
    full_price * num_full_price = 300 :=
by sorry

end NUMINAMATH_CALUDE_full_price_revenue_l1548_154845


namespace NUMINAMATH_CALUDE_handbag_price_l1548_154852

/-- Calculates the total selling price of a product given its original price, discount rate, and tax rate. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  discountedPrice * (1 + taxRate)

/-- Theorem stating that the total selling price of a $100 product with 30% discount and 8% tax is $75.6 -/
theorem handbag_price : 
  totalSellingPrice 100 0.3 0.08 = 75.6 := by
  sorry

end NUMINAMATH_CALUDE_handbag_price_l1548_154852


namespace NUMINAMATH_CALUDE_four_people_name_condition_l1548_154883

/-- Represents a person with a first name, patronymic, and last name -/
structure Person where
  firstName : String
  patronymic : String
  lastName : String

/-- Checks if two people share any attribute -/
def shareAttribute (p1 p2 : Person) : Prop :=
  p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.lastName = p2.lastName

/-- Theorem stating the existence of 4 people satisfying the given conditions -/
theorem four_people_name_condition : ∃ (people : Finset Person),
  (Finset.card people = 4) ∧
  (∀ (attr : Person → String),
    ∀ (p1 p2 p3 : Person),
      p1 ∈ people → p2 ∈ people → p3 ∈ people →
      p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
      ¬(attr p1 = attr p2 ∧ attr p2 = attr p3)) ∧
  (∀ (p1 p2 : Person),
    p1 ∈ people → p2 ∈ people → p1 ≠ p2 →
    shareAttribute p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_four_people_name_condition_l1548_154883


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_l1548_154847

/-- A regular polygon with exterior angles each measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18 :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_l1548_154847


namespace NUMINAMATH_CALUDE_smallest_x_value_l1548_154851

theorem smallest_x_value (x y : ℕ+) (h : (3 : ℚ) / 5 = y / (468 + x)) : 2 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1548_154851


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l1548_154803

/-- Given a tank with specified capacity and emptying times, calculate the inlet pipe rate -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (outlet_time : ℝ) (combined_time : ℝ) :
  tank_capacity = 3200 →
  outlet_time = 5 →
  combined_time = 8 →
  (tank_capacity / combined_time - tank_capacity / outlet_time) * (1 / 60) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inlet_pipe_rate_l1548_154803


namespace NUMINAMATH_CALUDE_households_with_car_l1548_154873

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end NUMINAMATH_CALUDE_households_with_car_l1548_154873


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_in_cones_l1548_154857

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ :=
  sorry

/-- Theorem stating the maximum squared radius of a sphere in the given configuration -/
theorem max_sphere_radius_squared_in_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSphereRadiusSquared ic = 144 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_in_cones_l1548_154857


namespace NUMINAMATH_CALUDE_inequality_proof_l1548_154801

theorem inequality_proof (n : ℕ+) (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c > 0) : 
  ((a * b + c)^n.val - c) / ((b + c)^n.val - c) ≤ a^n.val := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1548_154801


namespace NUMINAMATH_CALUDE_unique_solution_l1548_154818

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (6 - y) = 9)
  (eq2 : y * (6 - z) = 9)
  (eq3 : z * (6 - x) = 9) :
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1548_154818


namespace NUMINAMATH_CALUDE_driving_meeting_problem_l1548_154817

/-- A problem about two people driving and meeting on the road. -/
theorem driving_meeting_problem (wife_delay : Real) (wife_speed : Real) (meeting_time : Real) :
  wife_delay = 0.5 →
  wife_speed = 50 →
  meeting_time = 2 →
  ∃ man_speed : Real, man_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_driving_meeting_problem_l1548_154817


namespace NUMINAMATH_CALUDE_factory_production_average_l1548_154819

theorem factory_production_average (first_25_avg : ℝ) (last_5_avg : ℝ) (total_days : ℕ) :
  first_25_avg = 60 →
  last_5_avg = 48 →
  total_days = 30 →
  (25 * first_25_avg + 5 * last_5_avg) / total_days = 58 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_average_l1548_154819


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1548_154880

def set_A : Set ℝ := {x | |x| ≤ 1}
def set_B : Set ℝ := {y | ∃ x, y = x^2}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1548_154880


namespace NUMINAMATH_CALUDE_arccos_sin_one_point_five_l1548_154892

theorem arccos_sin_one_point_five (π : Real) :
  π = 3.14159265358979323846 →
  Real.arccos (Real.sin 1.5) = 0.0708 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_point_five_l1548_154892


namespace NUMINAMATH_CALUDE_lower_bound_of_exponential_sum_l1548_154849

theorem lower_bound_of_exponential_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
  ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, (2^a + 2^b + 2^c < x ↔ m ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_of_exponential_sum_l1548_154849


namespace NUMINAMATH_CALUDE_circle_condition_l1548_154848

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + 2*y - m = 0) ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1548_154848


namespace NUMINAMATH_CALUDE_product_xyz_l1548_154824

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2 * y = z) 
  (h2 : x^2 - 4 * y^2 + z^2 = 310) : 
  x * y * z = 11935 ∨ x * y * z = 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l1548_154824


namespace NUMINAMATH_CALUDE_center_is_five_l1548_154866

/-- Represents a 3x3 array of integers -/
def Array3x3 := Fin 3 → Fin 3 → ℕ

/-- Checks if two positions in the array are adjacent -/
def is_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if the array contains all numbers from 1 to 9 -/
def contains_all_numbers (a : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, a i j = n + 1

/-- Checks if consecutive numbers are adjacent in the array -/
def consecutive_adjacent (a : Array3x3) : Prop :=
  ∀ n : Fin 8, ∃ i₁ j₁ i₂ j₂ : Fin 3,
    a i₁ j₁ = n + 1 ∧ a i₂ j₂ = n + 2 ∧ is_adjacent (i₁, j₁) (i₂, j₂)

/-- The sum of corner numbers is 20 -/
def corner_sum_20 (a : Array3x3) : Prop :=
  a 0 0 + a 0 2 + a 2 0 + a 2 2 = 20

/-- The product of top-left and bottom-right corner numbers is 9 -/
def corner_product_9 (a : Array3x3) : Prop :=
  a 0 0 * a 2 2 = 9

theorem center_is_five (a : Array3x3)
  (h1 : contains_all_numbers a)
  (h2 : consecutive_adjacent a)
  (h3 : corner_sum_20 a)
  (h4 : corner_product_9 a) :
  a 1 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_center_is_five_l1548_154866


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1548_154850

theorem complex_fraction_equality (z : ℂ) (h : z = 1 + I) :
  (3 * I) / (z + 1) = 3/5 + 6/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1548_154850


namespace NUMINAMATH_CALUDE_garden_area_l1548_154886

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l1548_154886


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l1548_154874

/-- Calculates the number of granola bars per kid given the number of kids, bars per box, and boxes purchased. -/
def granola_bars_per_kid (num_kids : ℕ) (bars_per_box : ℕ) (boxes_purchased : ℕ) : ℕ :=
  (bars_per_box * boxes_purchased) / num_kids

/-- Proves that given 30 kids, 12 bars per box, and 5 boxes purchased, the number of granola bars per kid is 2. -/
theorem granola_bar_distribution : granola_bars_per_kid 30 12 5 = 2 := by
  sorry

#eval granola_bars_per_kid 30 12 5

end NUMINAMATH_CALUDE_granola_bar_distribution_l1548_154874
