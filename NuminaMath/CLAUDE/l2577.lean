import Mathlib

namespace NUMINAMATH_CALUDE_gumballs_last_days_l2577_257715

def gumballs_per_earring : ℕ := 9
def first_day_earrings : ℕ := 3
def second_day_earrings : ℕ := 2 * first_day_earrings
def third_day_earrings : ℕ := second_day_earrings - 1
def daily_consumption : ℕ := 3

def total_earrings : ℕ := first_day_earrings + second_day_earrings + third_day_earrings
def total_gumballs : ℕ := total_earrings * gumballs_per_earring

theorem gumballs_last_days : total_gumballs / daily_consumption = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_days_l2577_257715


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2577_257700

theorem tan_alpha_value (α : Real) (h : Real.tan (α - Real.pi/4) = 1/6) : 
  Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2577_257700


namespace NUMINAMATH_CALUDE_decimal_value_l2577_257709

theorem decimal_value (x : ℚ) : (10^5 - 10^3) * x = 31 → x = 1 / 3168 := by
  sorry

end NUMINAMATH_CALUDE_decimal_value_l2577_257709


namespace NUMINAMATH_CALUDE_sqrt_98_plus_sqrt_32_l2577_257753

theorem sqrt_98_plus_sqrt_32 : Real.sqrt 98 + Real.sqrt 32 = 11 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_98_plus_sqrt_32_l2577_257753


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l2577_257764

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation ax^2 + bx + c = 0 has real roots iff its discriminant is nonnegative -/
def has_real_roots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

theorem quadratic_roots_existence :
  ¬(has_real_roots 1 1 1) ∧
  (has_real_roots 1 2 1) ∧
  (has_real_roots 1 (-2) (-1)) ∧
  (has_real_roots 1 (-1) (-2)) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l2577_257764


namespace NUMINAMATH_CALUDE_garden_ratio_l2577_257787

theorem garden_ratio (area width length : ℝ) : 
  area = 588 → width = 14 → length * width = area → (length / width = 3) := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l2577_257787


namespace NUMINAMATH_CALUDE_omi_age_l2577_257747

/-- Given the ages of Kimiko, Arlette, and Omi, prove Omi's age -/
theorem omi_age (kimiko_age : ℕ) (arlette_age : ℕ) (omi_age : ℕ) : 
  kimiko_age = 28 →
  arlette_age = (3 * kimiko_age) / 4 →
  (kimiko_age + arlette_age + omi_age) / 3 = 35 →
  omi_age = 56 := by
sorry

end NUMINAMATH_CALUDE_omi_age_l2577_257747


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2577_257703

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = 3/2 ∧ c = 1/2 →
  (b * total) / (a + b + c) = 39 := by
sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2577_257703


namespace NUMINAMATH_CALUDE_square_difference_equality_l2577_257724

theorem square_difference_equality (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2577_257724


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l2577_257760

theorem solution_set_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - m| > 4) → (m > 3 ∨ m < -5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l2577_257760


namespace NUMINAMATH_CALUDE_shop_weekly_earnings_value_l2577_257751

/-- Represents the shop's weekly earnings calculation -/
def shop_weekly_earnings : ℝ :=
  let open_minutes : ℕ := 12 * 60
  let womens_tshirts_sold : ℕ := open_minutes / 30
  let mens_tshirts_sold : ℕ := open_minutes / 40
  let womens_jeans_sold : ℕ := open_minutes / 45
  let mens_jeans_sold : ℕ := open_minutes / 60
  let unisex_hoodies_sold : ℕ := open_minutes / 70

  let daily_earnings : ℝ :=
    womens_tshirts_sold * 18 +
    mens_tshirts_sold * 15 +
    womens_jeans_sold * 40 +
    mens_jeans_sold * 45 +
    unisex_hoodies_sold * 35

  let wednesday_earnings : ℝ := daily_earnings * 0.9
  let saturday_earnings : ℝ := daily_earnings * 1.05
  let other_days_earnings : ℝ := daily_earnings * 5

  wednesday_earnings + saturday_earnings + other_days_earnings

theorem shop_weekly_earnings_value :
  shop_weekly_earnings = 15512.40 := by sorry

end NUMINAMATH_CALUDE_shop_weekly_earnings_value_l2577_257751


namespace NUMINAMATH_CALUDE_evaluate_expression_l2577_257766

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2577_257766


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l2577_257730

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → |a| = |b|

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  |a| = |b| → a = b

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_proposition_correct :
  ∀ a b : ℝ, inverse_proposition a b ↔ ¬(original_proposition a b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_correct_l2577_257730


namespace NUMINAMATH_CALUDE_park_track_area_increase_l2577_257782

def small_diameter : ℝ := 15
def large_diameter : ℝ := 20

theorem park_track_area_increase :
  let small_area := π * (small_diameter / 2)^2
  let large_area := π * (large_diameter / 2)^2
  (large_area - small_area) / small_area = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_park_track_area_increase_l2577_257782


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_y_l2577_257702

theorem tan_theta_in_terms_of_x_y (θ x y : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sin (θ/2) = Real.sqrt ((y - x)/(y + x))) : 
  Real.tan θ = (2 * Real.sqrt (x * y)) / (3 * x - y) := by
sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_y_l2577_257702


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l2577_257712

theorem stratified_sampling_sample_size (total_population : ℕ) (elderly_population : ℕ) (elderly_sample : ℕ) (sample_size : ℕ) :
  total_population = 162 →
  elderly_population = 27 →
  elderly_sample = 6 →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population →
  sample_size = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l2577_257712


namespace NUMINAMATH_CALUDE_max_problems_solved_l2577_257744

theorem max_problems_solved (n : ℕ) (avg : ℕ) (h1 : n = 25) (h2 : avg = 6) :
  ∃ (max : ℕ), max = 126 ∧
  ∀ (problems : Fin n → ℕ),
  (∀ i, problems i ≥ 1) →
  (Finset.sum Finset.univ problems = n * avg) →
  ∀ i, problems i ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_problems_solved_l2577_257744


namespace NUMINAMATH_CALUDE_equation_solution_l2577_257732

theorem equation_solution : ∃ x : ℕ, (81^20 + 81^20 + 81^20 + 81^20 + 81^20 + 81^20 = 3^x) ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2577_257732


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l2577_257723

/-- Given real numbers 2, b, and a form a geometric sequence, 
    the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots 
  (b a : ℝ) 
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ 
    (∀ (z : ℝ), a * z^2 + b * z + 1/3 = 0 ↔ z = x ∨ z = y)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l2577_257723


namespace NUMINAMATH_CALUDE_readers_overlap_l2577_257743

theorem readers_overlap (total : ℕ) (sci_fi : ℕ) (literary : ℕ) (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end NUMINAMATH_CALUDE_readers_overlap_l2577_257743


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l2577_257769

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l2577_257769


namespace NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l2577_257711

/-- Calculates the percentage of carbonated water in a mixture of two solutions -/
def carbonated_water_percentage (solution1_percent : ℝ) (solution1_carbonated : ℝ) 
  (solution2_carbonated : ℝ) : ℝ :=
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated

theorem mixture_carbonated_water_percentage :
  carbonated_water_percentage 0.1999999999999997 0.80 0.55 = 0.5999999999999999 := by
  sorry

#eval carbonated_water_percentage 0.1999999999999997 0.80 0.55

end NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l2577_257711


namespace NUMINAMATH_CALUDE_function_ordering_l2577_257767

/-- A function f is even with respect to x = -1 -/
def IsEvenShifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = f (-x - 1)

/-- The function f is strictly decreasing in terms of its values when x > -1 -/
def IsStrictlyDecreasingShifted (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → (f x₂ - f x₁) * (x₂ - x₁) < 0

theorem function_ordering (f : ℝ → ℝ) 
    (h1 : IsEvenShifted f) 
    (h2 : IsStrictlyDecreasingShifted f) : 
    f (-2) < f 1 ∧ f 1 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_ordering_l2577_257767


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2577_257729

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a (-x) = f a x) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2577_257729


namespace NUMINAMATH_CALUDE_smallest_multiples_product_l2577_257735

theorem smallest_multiples_product (c d : ℕ) : 
  (c ≥ 10 ∧ c < 100 ∧ c % 7 = 0 ∧ ∀ x, x ≥ 10 ∧ x < 100 ∧ x % 7 = 0 → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 5 = 0 ∧ ∀ y, y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 → d ≤ y) →
  (c * d) - 100 = 1300 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiples_product_l2577_257735


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2577_257756

/-- The time it takes for the second pipe to empty the tank -/
def empty_time : ℝ := 24

/-- The time after which the second pipe is closed when both pipes are open -/
def close_time : ℝ := 48

/-- The total time it takes to fill the tank -/
def total_fill_time : ℝ := 30

/-- The time it takes for the first pipe to fill the tank -/
def fill_time : ℝ := 22

theorem pipe_fill_time : 
  (close_time * (1 / fill_time - 1 / empty_time) + (total_fill_time - close_time) * (1 / fill_time) = 1) →
  fill_time = 22 := by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2577_257756


namespace NUMINAMATH_CALUDE_fraction_equality_l2577_257728

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2577_257728


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l2577_257708

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 4}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem intersection_A_B_when_a_is_3 :
  A ∩ B 3 = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem A_subset_B_iff_a_greater_than_5 :
  ∀ a : ℝ, A ⊆ B a ↔ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l2577_257708


namespace NUMINAMATH_CALUDE_fraction_equality_l2577_257770

theorem fraction_equality : (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2577_257770


namespace NUMINAMATH_CALUDE_barney_towel_problem_l2577_257731

/-- The number of days without clean towels for Barney -/
def days_without_clean_towels (total_towels : ℕ) (towels_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  let towels_used_in_missed_week := towels_per_day * days_in_week
  let remaining_towels := total_towels - towels_used_in_missed_week
  let days_with_clean_towels := remaining_towels / towels_per_day
  days_in_week - days_with_clean_towels

/-- Theorem stating that Barney will not have clean towels for 5 days in the following week -/
theorem barney_towel_problem :
  days_without_clean_towels 18 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_barney_towel_problem_l2577_257731


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2023_l2577_257717

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2023 :
  (sum_factorials 2023) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2023_l2577_257717


namespace NUMINAMATH_CALUDE_coal_duration_coal_lasts_100_days_l2577_257736

/-- The number of days a batch of coal will last given planned and actual consumption rates. -/
theorem coal_duration (planned_daily_consumption : ℝ) (planned_duration : ℝ) (consumption_reduction : ℝ) : 
  planned_daily_consumption > 0 →
  planned_duration > 0 →
  consumption_reduction > 0 →
  consumption_reduction < 1 →
  (planned_daily_consumption * planned_duration) / (planned_daily_consumption * (1 - consumption_reduction)) = 
    planned_duration / (1 - consumption_reduction) := by
  sorry

/-- The specific problem of coal lasting 100 days. -/
theorem coal_lasts_100_days : 
  let planned_daily_consumption : ℝ := 0.25
  let planned_duration : ℝ := 80
  let consumption_reduction : ℝ := 0.2
  (planned_daily_consumption * planned_duration) / (planned_daily_consumption * (1 - consumption_reduction)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_coal_duration_coal_lasts_100_days_l2577_257736


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2577_257746

def f (x : ℝ) := x^2 - 4*x + 3
def g (x : ℝ) := -3*x + 3

theorem quadratic_function_properties :
  (∃ (x : ℝ), g x = 0 ∧ f x = 0) ∧
  (g 0 = f 0) ∧
  (∀ (x : ℝ), f x ≥ -1) ∧
  (∃ (x : ℝ), f x = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2577_257746


namespace NUMINAMATH_CALUDE_average_score_proof_l2577_257714

theorem average_score_proof (total_students : Nat) (abc_students : Nat) (de_students : Nat)
  (total_average : ℚ) (abc_average : ℚ) :
  total_students = 5 →
  abc_students = 3 →
  de_students = 2 →
  total_average = 80 →
  abc_average = 78 →
  (total_students * total_average - abc_students * abc_average) / de_students = 83 := by
  sorry

end NUMINAMATH_CALUDE_average_score_proof_l2577_257714


namespace NUMINAMATH_CALUDE_salary_percent_increase_l2577_257781

theorem salary_percent_increase 
  (original_salary new_salary increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : increase = 25000)
  (h3 : original_salary = new_salary - increase) :
  (increase / original_salary) * 100 = (25000 / (90000 - 25000)) * 100 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l2577_257781


namespace NUMINAMATH_CALUDE_find_number_l2577_257742

theorem find_number : ∃ x : ℝ, x = 800 ∧ 0.4 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2577_257742


namespace NUMINAMATH_CALUDE_lcm_18_24_l2577_257797

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l2577_257797


namespace NUMINAMATH_CALUDE_light_distance_theorem_l2577_257741

/-- The distance light travels in one year in miles -/
def light_year_miles : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def miles_to_km : ℝ := 1.60934

/-- The distance light travels in the given number of years in kilometers -/
def light_distance_km : ℝ := light_year_miles * years * miles_to_km

theorem light_distance_theorem : 
  light_distance_km = 4.723e15 := by sorry

end NUMINAMATH_CALUDE_light_distance_theorem_l2577_257741


namespace NUMINAMATH_CALUDE_total_marks_calculation_l2577_257798

theorem total_marks_calculation (num_candidates : ℕ) (average_marks : ℕ) 
  (h1 : num_candidates = 120) (h2 : average_marks = 35) : 
  num_candidates * average_marks = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l2577_257798


namespace NUMINAMATH_CALUDE_subtraction_with_division_l2577_257791

theorem subtraction_with_division : 3034 - (1002 / 20.04) = 2984 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l2577_257791


namespace NUMINAMATH_CALUDE_dry_cleaning_time_is_ten_l2577_257789

def total_time : ℕ := 180 -- 3 hours = 180 minutes
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dog_groomer_time : ℕ := 20
def cooking_time : ℕ := 90

def dry_cleaning_time : ℕ := total_time - commute_time - grocery_time - dog_groomer_time - cooking_time

theorem dry_cleaning_time_is_ten : dry_cleaning_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_dry_cleaning_time_is_ten_l2577_257789


namespace NUMINAMATH_CALUDE_gina_tip_percentage_is_five_percent_l2577_257755

/-- The bill amount in dollars -/
def bill_amount : ℝ := 26

/-- The minimum tip percentage for good tippers -/
def good_tipper_percentage : ℝ := 20

/-- The additional amount in cents Gina needs to tip to be a good tipper -/
def additional_tip_cents : ℝ := 390

/-- Gina's tip percentage -/
def gina_tip_percentage : ℝ := 5

/-- Theorem stating that Gina's tip percentage is 5% given the conditions -/
theorem gina_tip_percentage_is_five_percent :
  (gina_tip_percentage / 100) * bill_amount + (additional_tip_cents / 100) =
  (good_tipper_percentage / 100) * bill_amount :=
by sorry

end NUMINAMATH_CALUDE_gina_tip_percentage_is_five_percent_l2577_257755


namespace NUMINAMATH_CALUDE_similar_triangle_coordinates_l2577_257761

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def ratio : ℝ := 2

-- Define the possible coordinates of C'
def C'_pos : ℝ × ℝ := (6, 4)
def C'_neg : ℝ × ℝ := (-6, -4)

-- Theorem statement
theorem similar_triangle_coordinates :
  ∀ (C' : ℝ × ℝ), 
    (∃ (k : ℝ), k = ratio ∧ C' = (k * C.1, k * C.2)) ∨
    (∃ (k : ℝ), k = -ratio ∧ C' = (k * C.1, k * C.2)) →
    C' = C'_pos ∨ C' = C'_neg :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_coordinates_l2577_257761


namespace NUMINAMATH_CALUDE_oldest_harper_child_age_l2577_257780

/-- The age of the oldest Harper child given the ages of the other three and the average age of all four. -/
theorem oldest_harper_child_age 
  (average_age : ℝ) 
  (younger_child1 : ℕ) 
  (younger_child2 : ℕ) 
  (younger_child3 : ℕ) 
  (h1 : average_age = 9) 
  (h2 : younger_child1 = 6) 
  (h3 : younger_child2 = 8) 
  (h4 : younger_child3 = 10) : 
  ∃ (oldest_child : ℕ), 
    (younger_child1 + younger_child2 + younger_child3 + oldest_child) / 4 = average_age ∧ 
    oldest_child = 12 := by
  sorry

end NUMINAMATH_CALUDE_oldest_harper_child_age_l2577_257780


namespace NUMINAMATH_CALUDE_shift_left_sum_l2577_257727

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

theorem shift_left_sum (f : QuadraticFunction) :
  let g := shift_left f 6
  g.a + g.b + g.c = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_shift_left_sum_l2577_257727


namespace NUMINAMATH_CALUDE_remainder_problem_l2577_257762

theorem remainder_problem (y : ℤ) : 
  ∃ (k : ℤ), y = 276 * k + 42 → ∃ (m : ℤ), y = 23 * m + 19 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2577_257762


namespace NUMINAMATH_CALUDE_survey_min_overlap_l2577_257777

/-- Given a survey of 120 people, where 95 like Mozart, 80 like Bach, and 75 like Beethoven,
    the minimum number of people who like both Mozart and Bach but not Beethoven is 45. -/
theorem survey_min_overlap (total : ℕ) (mozart : ℕ) (bach : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_bach : bach = 80)
  (h_beethoven : beethoven = 75)
  (h_mozart_le : mozart ≤ total)
  (h_bach_le : bach ≤ total)
  (h_beethoven_le : beethoven ≤ total) :
  ∃ (overlap : ℕ), overlap ≥ 45 ∧
    overlap ≤ mozart ∧
    overlap ≤ bach ∧
    overlap ≤ total - beethoven ∧
    ∀ (x : ℕ), x < overlap →
      ¬(x ≤ mozart ∧ x ≤ bach ∧ x ≤ total - beethoven) :=
by sorry

end NUMINAMATH_CALUDE_survey_min_overlap_l2577_257777


namespace NUMINAMATH_CALUDE_outfits_count_l2577_257707

/-- The number of different outfits that can be created given a set of clothing items. -/
def number_of_outfits (shirts : Nat) (pants : Nat) (ties : Nat) (shoes : Nat) : Nat :=
  shirts * pants * (ties + 1) * shoes

/-- Theorem stating that the number of outfits is 240 given the specific clothing items. -/
theorem outfits_count :
  number_of_outfits 5 4 5 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2577_257707


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l2577_257752

theorem coffee_shop_sales (coffee_customers : ℕ) (coffee_price : ℕ) 
  (tea_customers : ℕ) (tea_price : ℕ) : 
  coffee_customers = 7 → 
  coffee_price = 5 → 
  tea_customers = 8 → 
  tea_price = 4 → 
  coffee_customers * coffee_price + tea_customers * tea_price = 67 := by
sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l2577_257752


namespace NUMINAMATH_CALUDE_grid_game_winner_l2577_257754

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | Player1Wins
  | Player2Wins

/-- Represents the game state on a 1 × N grid strip -/
structure GameState (N : ℕ) where
  grid : Fin N → Option Bool
  turn : Bool

/-- Defines the game rules and winning conditions -/
def gameResult (N : ℕ) : GameOutcome :=
  if N = 1 then
    GameOutcome.Player1Wins
  else
    GameOutcome.Player2Wins

/-- Theorem stating the winning player based on the grid size -/
theorem grid_game_winner (N : ℕ) :
  (N = 1 → gameResult N = GameOutcome.Player1Wins) ∧
  (N > 1 → gameResult N = GameOutcome.Player2Wins) := by
  sorry

/-- Lemma: Player 1 wins when N = 1 -/
lemma player1_wins_n1 (N : ℕ) (h : N = 1) :
  gameResult N = GameOutcome.Player1Wins := by
  sorry

/-- Lemma: Player 2 wins when N > 1 -/
lemma player2_wins_n_gt1 (N : ℕ) (h : N > 1) :
  gameResult N = GameOutcome.Player2Wins := by
  sorry

end NUMINAMATH_CALUDE_grid_game_winner_l2577_257754


namespace NUMINAMATH_CALUDE_pet_show_big_dogs_l2577_257740

theorem pet_show_big_dogs :
  ∀ (big_dogs small_dogs : ℕ),
  (big_dogs : ℚ) / small_dogs = 3 / 17 →
  big_dogs + small_dogs = 80 →
  big_dogs = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_show_big_dogs_l2577_257740


namespace NUMINAMATH_CALUDE_max_value_of_vector_difference_l2577_257721

/-- Given plane vectors a and b satisfying |b| = 2|a| = 2, 
    the maximum value of |a - 2b| is 5. -/
theorem max_value_of_vector_difference (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2 * ‖a‖) (h2 : ‖b‖ = 2) : 
  ∃ (max : ℝ), max = 5 ∧ ∀ (x : ℝ × ℝ), x = a - 2 • b → ‖x‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_vector_difference_l2577_257721


namespace NUMINAMATH_CALUDE_choir_dance_team_equation_l2577_257774

theorem choir_dance_team_equation (x : ℤ) : 
  (46 + x = 3 * (30 - x)) ↔ 
  (∃ (initial_choir initial_dance final_choir final_dance : ℤ),
    initial_choir = 46 ∧ 
    initial_dance = 30 ∧ 
    final_choir = initial_choir + x ∧ 
    final_dance = initial_dance - x ∧ 
    final_choir = 3 * final_dance) :=
by sorry

end NUMINAMATH_CALUDE_choir_dance_team_equation_l2577_257774


namespace NUMINAMATH_CALUDE_sqrt_problem_l2577_257790

theorem sqrt_problem (m n a : ℝ) : 
  (∃ (x : ℝ), x^2 = m ∧ x = 3) → 
  (∃ (y z : ℝ), y^2 = n ∧ z^2 = n ∧ y = a + 4 ∧ z = 2*a - 16) →
  m = 9 ∧ n = 64 ∧ (7*m - n)^(1/3) = -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_l2577_257790


namespace NUMINAMATH_CALUDE_cost_equation_solution_l2577_257763

/-- Given the cost equations for products A and B, prove that the solution (16, 4) satisfies both equations. -/
theorem cost_equation_solution :
  let x : ℚ := 16
  let y : ℚ := 4
  (20 * x + 15 * y = 380) ∧ (15 * x + 10 * y = 280) := by
  sorry

end NUMINAMATH_CALUDE_cost_equation_solution_l2577_257763


namespace NUMINAMATH_CALUDE_find_number_l2577_257726

theorem find_number (x : ℕ) : 102 * 102 + x * x = 19808 → x = 97 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2577_257726


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2577_257796

theorem quadratic_inequality_solution_set (x : ℝ) : x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2577_257796


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2577_257771

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2577_257771


namespace NUMINAMATH_CALUDE_induction_even_numbers_l2577_257768

theorem induction_even_numbers (P : ℕ → Prop) (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) 
  (h_base : P 2) (h_inductive : ∀ m : ℕ, m ≥ 2 → Even m → P m → P (m + 2)) :
  (P k → P (k + 2)) ∧ ¬(P k → P (k + 1)) ∧ ¬(P k → P (2*k + 2)) ∧ ¬(P k → P (2*(k + 2))) :=
sorry

end NUMINAMATH_CALUDE_induction_even_numbers_l2577_257768


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2577_257785

/-- Given that i is the imaginary unit and z is a complex number defined as
    z = ((1+i)^2 + 3(1-i)) / (2+i), prove that if z^2 + az + b = 1 + i
    where a and b are real numbers, then a = -3 and b = 4. -/
theorem complex_equation_solution (i : ℂ) (a b : ℝ) :
  i^2 = -1 →
  let z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)
  z^2 + a*z + b = 1 + i →
  a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2577_257785


namespace NUMINAMATH_CALUDE_prob_red_ball_l2577_257750

def urn1_red : ℚ := 3 / 8
def urn1_total : ℚ := 8
def urn2_red : ℚ := 1 / 2
def urn2_total : ℚ := 8
def urn3_red : ℚ := 0
def urn3_total : ℚ := 8

def prob_urn_selection : ℚ := 1 / 3

theorem prob_red_ball : 
  prob_urn_selection * (urn1_red * (urn1_total / urn1_total) + 
                        urn2_red * (urn2_total / urn2_total) + 
                        urn3_red * (urn3_total / urn3_total)) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_ball_l2577_257750


namespace NUMINAMATH_CALUDE_right_triangle_area_from_sticks_l2577_257795

/-- Represents a stick of length 24 cm that can be broken into two pieces -/
structure Stick :=
  (length : ℝ := 24)
  (piece1 : ℝ)
  (piece2 : ℝ)
  (break_constraint : piece1 + piece2 = length)

/-- Represents a right triangle formed from three sticks -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (hypotenuse : ℝ)
  (pythagorean : leg1^2 + leg2^2 = hypotenuse^2)

/-- Theorem stating that if a right triangle can be formed from three 24 cm sticks
    (one of which is broken), then its area is 216 square centimeters -/
theorem right_triangle_area_from_sticks 
  (s1 s2 : Stick) (s3 : Stick) (t : RightTriangle)
  (h1 : s1.length = 24 ∧ s2.length = 24 ∧ s3.length = 24)
  (h2 : t.leg1 = s1.piece1 ∧ t.leg2 = s2.length ∧ t.hypotenuse = s1.piece2 + s3.length) :
  t.leg1 * t.leg2 / 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_from_sticks_l2577_257795


namespace NUMINAMATH_CALUDE_product_to_power_minus_one_l2577_257745

theorem product_to_power_minus_one :
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_to_power_minus_one_l2577_257745


namespace NUMINAMATH_CALUDE_inequality_implies_b_minus_a_equals_two_l2577_257794

theorem inequality_implies_b_minus_a_equals_two (a b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) →
  b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_b_minus_a_equals_two_l2577_257794


namespace NUMINAMATH_CALUDE_x_minus_y_is_perfect_square_l2577_257759

theorem x_minus_y_is_perfect_square (x y : ℕ+) 
  (h : 3 * x ^ 2 + x = 4 * y ^ 2 + y) : 
  ∃ (k : ℕ), x - y = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_is_perfect_square_l2577_257759


namespace NUMINAMATH_CALUDE_inequality_solution_l2577_257713

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ x < -7 ∨ (-7 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2577_257713


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l2577_257737

/-- The number of FGH supermarkets in the US and Canada -/
structure FGHSupermarkets where
  total : ℕ
  us : ℕ
  canada : ℕ

/-- The conditions for FGH supermarkets -/
def validFGHSupermarkets (s : FGHSupermarkets) : Prop :=
  s.total = 60 ∧
  s.us + s.canada = s.total ∧
  s.us = 37 ∧
  s.us > s.canada

/-- Theorem: The difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference (s : FGHSupermarkets) 
  (h : validFGHSupermarkets s) : s.us - s.canada = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l2577_257737


namespace NUMINAMATH_CALUDE_product_inequality_l2577_257738

theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≥ 25/4 := by sorry

end NUMINAMATH_CALUDE_product_inequality_l2577_257738


namespace NUMINAMATH_CALUDE_scientific_notation_of_31400000_l2577_257765

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_31400000 :
  toScientificNotation 31400000 = ScientificNotation.mk 3.14 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_31400000_l2577_257765


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l2577_257758

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l2577_257758


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l2577_257734

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : ℚ
  beth : ℚ
  cyril : ℚ
  dan : ℚ
  eve : ℚ

/-- Calculates the pizza consumption for each sibling based on the given conditions -/
def calculate_consumption : PizzaConsumption := {
  alex := 1/6,
  beth := 2/7,
  cyril := 1/3,
  dan := 1 - (1/6 + 2/7 + 1/3 + 1/8) - 2/168,
  eve := 1/8 + 2/168
}

/-- Represents the correct order of siblings based on pizza consumption -/
def correct_order := ["Cyril", "Beth", "Eve", "Alex", "Dan"]

/-- Theorem stating that the calculated consumption leads to the correct order -/
theorem pizza_consumption_order : 
  let c := calculate_consumption
  (c.cyril > c.beth) ∧ (c.beth > c.eve) ∧ (c.eve > c.alex) ∧ (c.alex > c.dan) := by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l2577_257734


namespace NUMINAMATH_CALUDE_tan_twenty_seventy_product_is_one_l2577_257748

theorem tan_twenty_seventy_product_is_one :
  Real.tan (20 * π / 180) * Real.tan (70 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_twenty_seventy_product_is_one_l2577_257748


namespace NUMINAMATH_CALUDE_division_problem_l2577_257720

theorem division_problem (d q : ℚ) : 
  (100 / d = q) → 
  (d * ⌊q⌋ ≤ 100) → 
  (100 - d * ⌊q⌋ = 4) → 
  (d = 16 ∧ q = 6.65) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2577_257720


namespace NUMINAMATH_CALUDE_min_value_trig_fraction_min_value_is_one_l2577_257799

theorem min_value_trig_fraction (x : ℝ) :
  (Real.sin x)^5 + (Real.cos x)^5 + 1 ≥ (Real.sin x)^3 + (Real.cos x)^3 + 1 := by
  sorry

theorem min_value_is_one :
  ∀ x : ℝ, ((Real.sin x)^5 + (Real.cos x)^5 + 1) / ((Real.sin x)^3 + (Real.cos x)^3 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_fraction_min_value_is_one_l2577_257799


namespace NUMINAMATH_CALUDE_train_passing_time_l2577_257783

/-- Two trains passing problem -/
theorem train_passing_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) (h1 : length1 = 280)
    (h2 : length2 = 350) (h3 : speed1 = 72) (h4 : speed2 = 54) :
    (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2577_257783


namespace NUMINAMATH_CALUDE_daragh_initial_bears_l2577_257792

/-- The number of stuffed bears Daragh initially had -/
def initial_bears : ℕ := 20

/-- The number of favorite bears Daragh took out -/
def favorite_bears : ℕ := 8

/-- The number of sisters Daragh divided the remaining bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_bears_before : ℕ := 10

/-- The number of bears Eden had after receiving more -/
def eden_bears_after : ℕ := 14

theorem daragh_initial_bears :
  initial_bears = favorite_bears + (eden_bears_after - eden_bears_before) * num_sisters :=
by sorry

end NUMINAMATH_CALUDE_daragh_initial_bears_l2577_257792


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l2577_257705

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (3 * x + 4) ∣ (4 * x^3 + 17 * x^2 + d * x + 28)) →
  d = 155 / 9 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l2577_257705


namespace NUMINAMATH_CALUDE_circle_chords_area_theorem_l2577_257793

/-- Given a circle with radius 48, two chords of length 84 intersecting at a point 24 units from
    the center, the area of the region consisting of a smaller sector and one triangle formed by
    the chords and intersection point can be expressed as m*π - n*√d, where m, n, d are positive
    integers, d is not divisible by any prime square, and m + n + d = 1302. -/
theorem circle_chords_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
    (h1 : r = 48)
    (h2 : chord_length = 84)
    (h3 : intersection_distance = 24) :
    ∃ (m n d : ℕ), 
      (m > 0 ∧ n > 0 ∧ d > 0) ∧ 
      (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ d)) ∧
      (m + n + d = 1302) ∧
      (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end NUMINAMATH_CALUDE_circle_chords_area_theorem_l2577_257793


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2577_257786

/-- A geometric sequence with first term 3 and the sum of 1st, 3rd, and 5th terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 
    a 1 = 3 ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) ∧
    a 1 + a 3 + a 5 = 21

/-- The product of the 2nd and 6th terms of the geometric sequence is 72 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 2 * a 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2577_257786


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l2577_257710

theorem red_marbles_in_bag (total : ℕ) (prob : ℚ) (h_total : total = 84) (h_prob : prob = 36/49) :
  ∃ red : ℕ, red = 12 ∧ (1 - (red : ℚ) / total) * (1 - (red : ℚ) / total) = prob :=
sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l2577_257710


namespace NUMINAMATH_CALUDE_larger_integer_value_l2577_257719

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * b = 160) : 
  max a b = 20 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2577_257719


namespace NUMINAMATH_CALUDE_multiple_with_four_digits_l2577_257779

theorem multiple_with_four_digits (k : ℕ) (h : k > 1) :
  ∃ w : ℕ, w > 0 ∧ k ∣ w ∧ w < k^4 ∧ 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a = 0 ∨ a = 1 ∨ a = 8 ∨ a = 9) ∧
    (b = 0 ∨ b = 1 ∨ b = 8 ∨ b = 9) ∧
    (c = 0 ∨ c = 1 ∨ c = 8 ∨ c = 9) ∧
    (d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9) ∧
    w = a * 1000 + b * 100 + c * 10 + d := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_four_digits_l2577_257779


namespace NUMINAMATH_CALUDE_add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l2577_257716

-- 1. (-51) + (-37) = -88
theorem add_negative_numbers : (-51) + (-37) = -88 := by sorry

-- 2. (+2) + (-11) = -9
theorem add_positive_negative : (2 : Int) + (-11) = -9 := by sorry

-- 3. (-12) + (+12) = 0
theorem add_negative_positive_inverse : (-12) + (12 : Int) = 0 := by sorry

-- 4. 8 - 14 = -6
theorem subtract_larger_from_smaller : (8 : Int) - 14 = -6 := by sorry

-- 5. 15 - (-8) = 23
theorem subtract_negative : (15 : Int) - (-8) = 23 := by sorry

-- 6. (-3.4) + 4.3 = 0.9
theorem add_negative_positive_real : (-3.4) + 4.3 = 0.9 := by sorry

-- 7. |-2.25| + (-0.5) = 1.75
theorem abs_value_add_negative : |(-2.25 : ℝ)| + (-0.5) = 1.75 := by sorry

-- 8. -4 * 1.5 = -6
theorem multiply_negative_mixed : (-4 : ℝ) * 1.5 = -6 := by sorry

-- 9. -3 * (-6) = 18
theorem multiply_two_negatives : (-3 : Int) * (-6) = 18 := by sorry

end NUMINAMATH_CALUDE_add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l2577_257716


namespace NUMINAMATH_CALUDE_coopers_age_l2577_257757

theorem coopers_age (cooper_age dante_age maria_age : ℕ) : 
  cooper_age + dante_age + maria_age = 31 →
  dante_age = 2 * cooper_age →
  maria_age = dante_age + 1 →
  cooper_age = 6 := by
sorry

end NUMINAMATH_CALUDE_coopers_age_l2577_257757


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2577_257749

theorem power_fraction_equality : (2^2016 + 2^2014) / (2^2016 - 2^2014) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2577_257749


namespace NUMINAMATH_CALUDE_cone_volume_l2577_257718

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters. -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2577_257718


namespace NUMINAMATH_CALUDE_solve_for_P_l2577_257784

theorem solve_for_P : ∃ P : ℝ, (P^3).sqrt = 81 * Real.rpow 81 (1/3) → P = Real.rpow 3 (32/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l2577_257784


namespace NUMINAMATH_CALUDE_third_person_profit_is_800_l2577_257788

/-- Calculates the third person's share of the profit in a joint business investment. -/
def third_person_profit (total_investment : ℕ) (investment_difference : ℕ) (total_profit : ℕ) : ℕ :=
  let first_investment := (total_investment - 3 * investment_difference) / 3
  let second_investment := first_investment + investment_difference
  let third_investment := second_investment + investment_difference
  (third_investment * total_profit) / total_investment

/-- Theorem stating that under the given conditions, the third person's profit share is 800. -/
theorem third_person_profit_is_800 :
  third_person_profit 9000 1000 1800 = 800 := by
  sorry

end NUMINAMATH_CALUDE_third_person_profit_is_800_l2577_257788


namespace NUMINAMATH_CALUDE_blue_eyed_percentage_l2577_257776

/-- Represents the number of kittens with blue eyes for a cat -/
def blue_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 3 else 4

/-- Represents the number of kittens with brown eyes for a cat -/
def brown_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 7 else 6

/-- The total number of kittens -/
def total_kittens : Nat :=
  (blue_eyed_kittens 1 + brown_eyed_kittens 1) + (blue_eyed_kittens 2 + brown_eyed_kittens 2)

/-- The total number of blue-eyed kittens -/
def total_blue_eyed : Nat :=
  blue_eyed_kittens 1 + blue_eyed_kittens 2

/-- Theorem stating that 35% of all kittens have blue eyes -/
theorem blue_eyed_percentage :
  (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyed_percentage_l2577_257776


namespace NUMINAMATH_CALUDE_binomial_and_permutation_60_3_l2577_257778

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem binomial_and_permutation_60_3 :
  binomial 60 3 = 34220 ∧ permutation 60 3 = 205320 :=
by sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_60_3_l2577_257778


namespace NUMINAMATH_CALUDE_gcd_1549_1023_l2577_257773

theorem gcd_1549_1023 : Nat.gcd 1549 1023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1549_1023_l2577_257773


namespace NUMINAMATH_CALUDE_lewis_savings_l2577_257772

/-- Lewis's savings calculation -/
theorem lewis_savings (weekly_earnings weekly_rent harvest_weeks : ℕ) 
  (h1 : weekly_earnings = 491)
  (h2 : weekly_rent = 216)
  (h3 : harvest_weeks = 1181) : 
  (weekly_earnings - weekly_rent) * harvest_weeks = 324775 := by
  sorry

#eval (491 - 216) * 1181  -- To verify the result

end NUMINAMATH_CALUDE_lewis_savings_l2577_257772


namespace NUMINAMATH_CALUDE_glasses_displayed_is_70_l2577_257704

/-- Represents the cupboard system with given capacities and a broken shelf --/
structure CupboardSystem where
  tall_capacity : ℕ
  wide_capacity : ℕ
  narrow_capacity : ℕ
  narrow_shelves : ℕ
  broken_shelves : ℕ

/-- Calculates the total number of glasses displayed in the cupboard system --/
def total_glasses_displayed (cs : CupboardSystem) : ℕ :=
  cs.tall_capacity + cs.wide_capacity + 
  (cs.narrow_capacity / cs.narrow_shelves) * (cs.narrow_shelves - cs.broken_shelves)

/-- Theorem stating that the total number of glasses displayed is 70 --/
theorem glasses_displayed_is_70 : ∃ (cs : CupboardSystem), 
  cs.tall_capacity = 20 ∧
  cs.wide_capacity = 2 * cs.tall_capacity ∧
  cs.narrow_capacity = 15 ∧
  cs.narrow_shelves = 3 ∧
  cs.broken_shelves = 1 ∧
  total_glasses_displayed cs = 70 := by
  sorry

end NUMINAMATH_CALUDE_glasses_displayed_is_70_l2577_257704


namespace NUMINAMATH_CALUDE_goldbach_132_max_diff_l2577_257725

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_132_max_diff :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 132 ∧ p < q ∧
  ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 132 → r < s →
  s - r ≤ q - p ∧
  q - p = 122 :=
sorry

end NUMINAMATH_CALUDE_goldbach_132_max_diff_l2577_257725


namespace NUMINAMATH_CALUDE_max_value_expression_l2577_257739

theorem max_value_expression (x y : ℝ) :
  (Real.sqrt (8 - 4 * Real.sqrt 3) * Real.sin x - 3 * Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
  (3 + 2 * Real.sqrt (11 - Real.sqrt 3) * Real.cos y - Real.cos (2 * y)) ≤ 33 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2577_257739


namespace NUMINAMATH_CALUDE_triangle_ABC_is_right_angled_l2577_257722

/-- Triangle ABC is defined by points A(5, -2), B(1, 5), and C(-1, 2) in a 2D Euclidean space -/
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (1, 5)
def C : ℝ × ℝ := (-1, 2)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Triangle ABC is right-angled -/
theorem triangle_ABC_is_right_angled : 
  dist_squared A B = dist_squared B C + dist_squared C A :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_right_angled_l2577_257722


namespace NUMINAMATH_CALUDE_car_profit_theorem_l2577_257775

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
def profit_percentage (discount : ℝ) (markup : ℝ) : ℝ :=
  let purchase_price := 1 - discount
  let sale_price := purchase_price * (1 + markup)
  (sale_price - 1) * 100

/-- Theorem stating that buying a car at 5% discount and selling at 60% markup
    results in a 52% profit on the original price. -/
theorem car_profit_theorem :
  profit_percentage 0.05 0.60 = 52 := by sorry

end NUMINAMATH_CALUDE_car_profit_theorem_l2577_257775


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2577_257733

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 28) →
  (total / 2 * 4 / 5 + total / 2 / 2 = total - remaining) →
  total = 80 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2577_257733


namespace NUMINAMATH_CALUDE_apartment_cost_difference_l2577_257701

-- Define the parameters for each apartment
def rent1 : ℕ := 800
def utilities1 : ℕ := 260
def miles1 : ℕ := 31

def rent2 : ℕ := 900
def utilities2 : ℕ := 200
def miles2 : ℕ := 21

-- Define common parameters
def workdays : ℕ := 20
def cost_per_mile : ℚ := 58 / 100

-- Function to calculate total monthly cost
def total_cost (rent : ℕ) (utilities : ℕ) (miles : ℕ) : ℚ :=
  rent + utilities + (miles * workdays * cost_per_mile)

-- Theorem statement
theorem apartment_cost_difference :
  ⌊total_cost rent1 utilities1 miles1 - total_cost rent2 utilities2 miles2⌋ = 76 := by
  sorry


end NUMINAMATH_CALUDE_apartment_cost_difference_l2577_257701


namespace NUMINAMATH_CALUDE_correct_calculation_l2577_257706

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2577_257706
