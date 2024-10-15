import Mathlib

namespace NUMINAMATH_CALUDE_vector_not_parallel_implies_m_l1572_157270

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_not_parallel_implies_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  ¬(are_parallel a b) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_not_parallel_implies_m_l1572_157270


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1572_157257

/-- The polynomial being divided -/
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + x^2 + 4

/-- The divisor -/
def g (x : ℝ) : ℝ := (x - 2)^2

/-- The remainder -/
def r (x : ℝ) : ℝ := 35*x + 48

/-- The quotient -/
def q (x : ℝ) : ℝ := sorry

theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1572_157257


namespace NUMINAMATH_CALUDE_quadratic_form_identity_l1572_157219

theorem quadratic_form_identity 
  (a b c d e f x y z : ℝ) 
  (h : a * x^2 + b * y^2 + c * z^2 + 2 * d * y * z + 2 * e * z * x + 2 * f * x * y = 0) :
  (d * y * z + e * z * x + f * x * y)^2 - b * c * y^2 * z^2 - c * a * z^2 * x^2 - a * b * x^2 * y^2 = 
  (1/4) * (x * Real.sqrt a + y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a + y * Real.sqrt b - z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b - z * Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_identity_l1572_157219


namespace NUMINAMATH_CALUDE_sean_apples_count_l1572_157227

/-- Proves that the number of apples Sean has after receiving apples from Susan
    is equal to the total number of apples mentioned. -/
theorem sean_apples_count (initial_apples : ℕ) (apples_from_susan : ℕ) (total_apples : ℕ)
    (h1 : initial_apples = 9)
    (h2 : apples_from_susan = 8)
    (h3 : total_apples = 17) :
    initial_apples + apples_from_susan = total_apples := by
  sorry

end NUMINAMATH_CALUDE_sean_apples_count_l1572_157227


namespace NUMINAMATH_CALUDE_customer_buys_score_of_eggs_l1572_157212

/-- Definition of a score in terms of units -/
def score : ℕ := 20

/-- Definition of a dozen in terms of units -/
def dozen : ℕ := 12

/-- The number of eggs a customer receives when buying a score of eggs -/
def eggs_in_score : ℕ := score

theorem customer_buys_score_of_eggs : eggs_in_score = 20 := by sorry

end NUMINAMATH_CALUDE_customer_buys_score_of_eggs_l1572_157212


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l1572_157293

/-- The percentage of the list price at which goods should be marked to achieve
    the desired profit and discount conditions. -/
theorem merchant_markup_percentage 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (selling_discount : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : purchase_discount = 0.2)
  (h2 : selling_discount = 0.2)
  (h3 : profit_percentage = 0.2)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 1.25 ∧ 
    (1 - purchase_discount) * list_price = 
    (1 - profit_percentage) * ((1 - selling_discount) * (markup_percentage * list_price)) :=
by sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l1572_157293


namespace NUMINAMATH_CALUDE_larger_number_proof_l1572_157297

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) :
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1572_157297


namespace NUMINAMATH_CALUDE_swimming_pool_length_l1572_157273

/-- Given a rectangular swimming pool with width 22 feet, surrounded by a deck of uniform width 3 feet,
    prove that if the total area of the pool and deck is 728 square feet, then the length of the pool is 20 feet. -/
theorem swimming_pool_length (pool_width deck_width total_area : ℝ) : 
  pool_width = 22 →
  deck_width = 3 →
  (pool_width + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area →
  total_area = 728 →
  ∃ pool_length : ℝ, pool_length = 20 ∧ (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area :=
by sorry

end NUMINAMATH_CALUDE_swimming_pool_length_l1572_157273


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1572_157261

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x * (x - 3) > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1572_157261


namespace NUMINAMATH_CALUDE_range_of_f_l1572_157214

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1572_157214


namespace NUMINAMATH_CALUDE_intersection_points_f_squared_f_sixth_l1572_157268

theorem intersection_points_f_squared_f_sixth (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x : ℝ, f (x^2) = f (x^6) ↔ x ∈ s)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_f_squared_f_sixth_l1572_157268


namespace NUMINAMATH_CALUDE_lydia_porch_flowers_l1572_157285

/-- The number of flowers on Lydia's porch --/
def flowers_on_porch (total_plants : ℕ) (flowering_percent : ℚ) 
  (seven_flower_percent : ℚ) (seven_flower_plants : ℕ) (four_flower_plants : ℕ) : ℕ :=
  seven_flower_plants * 7 + four_flower_plants * 4

/-- Theorem stating the number of flowers on Lydia's porch --/
theorem lydia_porch_flowers :
  flowers_on_porch 120 (35/100) (60/100) 8 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lydia_porch_flowers_l1572_157285


namespace NUMINAMATH_CALUDE_intersection_point_property_l1572_157249

theorem intersection_point_property (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  b = -2 / a ∧ b = a + 3 → 1 / a - 1 / b = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l1572_157249


namespace NUMINAMATH_CALUDE_exercise_distribution_properties_l1572_157218

/-- Represents the frequency distribution of daily exercise time --/
structure ExerciseDistribution :=
  (less_than_70 : ℕ)
  (between_70_and_80 : ℕ)
  (between_80_and_90 : ℕ)
  (greater_than_90 : ℕ)

/-- Theorem stating the properties of the exercise distribution --/
theorem exercise_distribution_properties
  (dist : ExerciseDistribution)
  (total_surveyed : ℕ)
  (h1 : dist.less_than_70 = 14)
  (h2 : dist.between_70_and_80 = 40)
  (h3 : dist.between_80_and_90 = 35)
  (h4 : total_surveyed = 100) :
  let m := (dist.between_70_and_80 : ℚ) / total_surveyed * 100
  let n := dist.greater_than_90
  let estimated_80_plus := ((dist.between_80_and_90 + dist.greater_than_90 : ℚ) / total_surveyed * 1000).floor
  let p := 86
  (m = 40 ∧ n = 11) ∧
  estimated_80_plus = 460 ∧
  (((11 : ℚ) / total_surveyed * 100 ≤ 25) ∧ 
   ((11 + 35 : ℚ) / total_surveyed * 100 ≥ 25)) := by
  sorry

#check exercise_distribution_properties

end NUMINAMATH_CALUDE_exercise_distribution_properties_l1572_157218


namespace NUMINAMATH_CALUDE_calculate_new_interest_rate_l1572_157206

/-- Given a principal amount and interest rates, proves the new interest rate -/
theorem calculate_new_interest_rate
  (P : ℝ)
  (h1 : P * 0.045 = 405)
  (h2 : P * 0.05 = 450) :
  0.05 = (405 + 45) / P :=
by sorry

end NUMINAMATH_CALUDE_calculate_new_interest_rate_l1572_157206


namespace NUMINAMATH_CALUDE_middle_number_is_four_l1572_157228

theorem middle_number_is_four (a b c : ℕ) : 
  a < b ∧ b < c  -- numbers are in increasing order
  → a + b + c = 15  -- numbers sum to 15
  → a ≠ b ∧ b ≠ c ∧ a ≠ c  -- numbers are all different
  → a > 0 ∧ b > 0 ∧ c > 0  -- numbers are positive
  → (∀ x y, x < y ∧ x + y < 15 → ∃ z, x < z ∧ z < y ∧ x + z + y = 15)  -- leftmost doesn't uniquely determine
  → (∀ x y, x < y ∧ x + y > 0 → ∃ z, z < x ∧ x < y ∧ z + x + y = 15)  -- rightmost doesn't uniquely determine
  → b = 4  -- middle number is 4
  := by sorry

end NUMINAMATH_CALUDE_middle_number_is_four_l1572_157228


namespace NUMINAMATH_CALUDE_triple_value_equation_l1572_157213

theorem triple_value_equation (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_triple_value_equation_l1572_157213


namespace NUMINAMATH_CALUDE_exists_fib_with_three_trailing_zeros_l1572_157230

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem exists_fib_with_three_trailing_zeros :
  ∃ n : ℕ, fib n % 1000 = 0 ∧ fib (n + 1) % 1000 = 0 ∧ fib (n + 2) % 1000 = 0 := by
  sorry


end NUMINAMATH_CALUDE_exists_fib_with_three_trailing_zeros_l1572_157230


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1572_157223

theorem trig_identity_proof : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1572_157223


namespace NUMINAMATH_CALUDE_total_spent_is_thirteen_l1572_157255

-- Define the cost of items
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := 6

-- Define the total spent
def total_spent : ℕ := candy_bar_cost + chocolate_cost

-- Theorem to prove
theorem total_spent_is_thirteen : total_spent = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_thirteen_l1572_157255


namespace NUMINAMATH_CALUDE_complex_division_l1572_157248

theorem complex_division (z : ℂ) : z = -2 + I → z / (1 + I) = -1/2 + 3/2 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_l1572_157248


namespace NUMINAMATH_CALUDE_will_baseball_card_pages_l1572_157209

/-- Calculates the number of pages needed to arrange baseball cards. -/
def pages_needed (cards_per_page : ℕ) (cards_2020 : ℕ) (cards_2015_2019 : ℕ) (duplicates : ℕ) : ℕ :=
  let unique_2020 := cards_2020
  let unique_2015_2019 := cards_2015_2019 - duplicates
  let pages_2020 := (unique_2020 + cards_per_page - 1) / cards_per_page
  let pages_2015_2019 := (unique_2015_2019 + cards_per_page - 1) / cards_per_page
  pages_2020 + pages_2015_2019

/-- Theorem stating the number of pages needed for Will's baseball card arrangement. -/
theorem will_baseball_card_pages :
  pages_needed 3 8 10 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_baseball_card_pages_l1572_157209


namespace NUMINAMATH_CALUDE_scientific_notation_300_billion_l1572_157263

theorem scientific_notation_300_billion :
  ∃ (a : ℝ) (n : ℤ), 300000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_300_billion_l1572_157263


namespace NUMINAMATH_CALUDE_intersection_point_implies_n_equals_two_l1572_157280

theorem intersection_point_implies_n_equals_two (n : ℕ+) 
  (x y : ℤ) -- x and y are integers
  (h1 : 15 * x + 18 * y = 1005) -- First line equation
  (h2 : y = n * x + 2) -- Second line equation
  : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_implies_n_equals_two_l1572_157280


namespace NUMINAMATH_CALUDE_olivias_wallet_after_supermarket_l1572_157205

/-- The amount left in Olivia's wallet after visiting the supermarket -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

theorem olivias_wallet_after_supermarket :
  money_left 94 16 = 78 := by
  sorry

end NUMINAMATH_CALUDE_olivias_wallet_after_supermarket_l1572_157205


namespace NUMINAMATH_CALUDE_area_enclosed_by_curve_and_x_axis_l1572_157267

-- Define the curve function
def f (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem area_enclosed_by_curve_and_x_axis : 
  ∫ x in (-1)..1, f x = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_curve_and_x_axis_l1572_157267


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l1572_157296

theorem units_digit_of_sum_of_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l1572_157296


namespace NUMINAMATH_CALUDE_fraction_addition_l1572_157299

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1572_157299


namespace NUMINAMATH_CALUDE_square_sum_value_l1572_157245

theorem square_sum_value (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1572_157245


namespace NUMINAMATH_CALUDE_poverty_decline_rate_l1572_157287

/-- The annual average decline rate of impoverished people -/
def annual_decline_rate : ℝ := 0.5

/-- The initial number of impoverished people in 2018 -/
def initial_population : ℕ := 40000

/-- The number of impoverished people in 2020 -/
def final_population : ℕ := 10000

/-- The time period in years -/
def time_period : ℕ := 2

theorem poverty_decline_rate :
  (↑initial_population * (1 - annual_decline_rate) ^ time_period = ↑final_population) ∧
  (0 < annual_decline_rate) ∧
  (annual_decline_rate < 1) := by
  sorry

end NUMINAMATH_CALUDE_poverty_decline_rate_l1572_157287


namespace NUMINAMATH_CALUDE_min_selection_for_sum_multiple_of_10_l1572_157284

/-- The set of numbers from 11 to 30 -/
def S : Set ℕ := {n | 11 ≤ n ∧ n ≤ 30}

/-- A function that checks if the sum of two numbers is a multiple of 10 -/
def sumIsMultipleOf10 (a b : ℕ) : Prop := (a + b) % 10 = 0

/-- The theorem stating the minimum number of integers to be selected -/
theorem min_selection_for_sum_multiple_of_10 :
  ∃ (k : ℕ), k = 11 ∧
  (∀ (T : Set ℕ), T ⊆ S → T.ncard ≥ k →
    ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ sumIsMultipleOf10 a b) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (T : Set ℕ), T ⊆ S ∧ T.ncard = k' ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(sumIsMultipleOf10 a b)) :=
sorry

end NUMINAMATH_CALUDE_min_selection_for_sum_multiple_of_10_l1572_157284


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1572_157279

theorem arithmetic_mean_of_fractions : 
  (3 / 7 + 5 / 8) / 2 = 59 / 112 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1572_157279


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l1572_157277

/-- Exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def yah_amount : ℕ := 2000

/-- The expected number of bahs after conversion -/
def expected_bah_amount : ℕ := 375

theorem yah_to_bah_conversion :
  (yah_amount : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = expected_bah_amount := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l1572_157277


namespace NUMINAMATH_CALUDE_exponential_inequality_l1572_157201

theorem exponential_inequality (x y a b : ℝ) 
  (h1 : x > y) (h2 : y > 1) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : 
  a^x < b^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1572_157201


namespace NUMINAMATH_CALUDE_science_fiction_section_pages_l1572_157222

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_section_pages :
  total_pages = 3824 := by sorry

end NUMINAMATH_CALUDE_science_fiction_section_pages_l1572_157222


namespace NUMINAMATH_CALUDE_toilet_paper_cost_l1572_157265

/-- Prove that the cost of one roll of toilet paper is $1.50 -/
theorem toilet_paper_cost 
  (total_toilet_paper : ℕ) 
  (total_paper_towels : ℕ) 
  (total_tissues : ℕ) 
  (total_cost : ℚ) 
  (paper_towel_cost : ℚ) 
  (tissue_cost : ℚ) 
  (h1 : total_toilet_paper = 10)
  (h2 : total_paper_towels = 7)
  (h3 : total_tissues = 3)
  (h4 : total_cost = 35)
  (h5 : paper_towel_cost = 2)
  (h6 : tissue_cost = 2) :
  (total_cost - (total_paper_towels * paper_towel_cost + total_tissues * tissue_cost)) / total_toilet_paper = (3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_toilet_paper_cost_l1572_157265


namespace NUMINAMATH_CALUDE_hyperbola_focus_l1572_157216

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus (k : ℝ) : 
  (∀ x y : ℝ, x^2 - k*y^2 = 1 → (∃ c : ℝ, c^2 = 9 ∧ c^2 = 1 + 1/k)) → 
  k = 1/8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l1572_157216


namespace NUMINAMATH_CALUDE_number_of_divisors_2310_l1572_157272

theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2310_l1572_157272


namespace NUMINAMATH_CALUDE_division_result_l1572_157236

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1572_157236


namespace NUMINAMATH_CALUDE_children_to_women_ratio_l1572_157204

theorem children_to_women_ratio 
  (total_spectators : ℕ) 
  (men_spectators : ℕ) 
  (children_spectators : ℕ) 
  (h1 : total_spectators = 10000)
  (h2 : men_spectators = 7000)
  (h3 : children_spectators = 2500) :
  (children_spectators : ℚ) / (total_spectators - men_spectators - children_spectators) = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_children_to_women_ratio_l1572_157204


namespace NUMINAMATH_CALUDE_range_of_x_l1572_157234

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  (x ∈ Set.Ioo 1 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1572_157234


namespace NUMINAMATH_CALUDE_course_selection_theorem_l1572_157217

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_theorem :
  (ways_to_choose type_a_courses 2 * ways_to_choose type_b_courses 1) +
  (ways_to_choose type_a_courses 1 * ways_to_choose type_b_courses 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l1572_157217


namespace NUMINAMATH_CALUDE_min_sum_x_y_l1572_157260

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) :
  x + y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2 * x₀ + Real.sqrt (4 * x₀^2 + 1)) * (Real.sqrt (y₀^2 + 4) - 2) ≥ y₀ ∧
    x₀ + y₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_x_y_l1572_157260


namespace NUMINAMATH_CALUDE_abc_value_l1572_157298

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 5 * b = -20)
  (eq2 : b * c + 5 * c = -20)
  (eq3 : c * a + 5 * a = -20) :
  a * b * c = 100 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l1572_157298


namespace NUMINAMATH_CALUDE_tshirts_sold_equals_45_l1572_157276

/-- The number of t-shirts sold by the Razorback t-shirt Shop last week -/
def num_tshirts_sold : ℕ := 45

/-- The price of each t-shirt in dollars -/
def price_per_tshirt : ℕ := 16

/-- The total amount of money made in dollars -/
def total_money_made : ℕ := 720

/-- Theorem: The number of t-shirts sold is equal to 45 -/
theorem tshirts_sold_equals_45 :
  num_tshirts_sold = total_money_made / price_per_tshirt :=
by sorry

end NUMINAMATH_CALUDE_tshirts_sold_equals_45_l1572_157276


namespace NUMINAMATH_CALUDE_find_m_l1572_157241

def U : Set Int := {-1, 2, 3, 6}

def A (m : Int) : Set Int := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : Int, A m = {-1, 6} ∧ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1572_157241


namespace NUMINAMATH_CALUDE_area_ratio_when_diameter_tripled_l1572_157254

/-- The ratio of areas when a circle's diameter is tripled -/
theorem area_ratio_when_diameter_tripled (d : ℝ) (h : d > 0) :
  let r := d / 2
  let new_r := 3 * r
  (π * new_r ^ 2) / (π * r ^ 2) = 9 := by
sorry


end NUMINAMATH_CALUDE_area_ratio_when_diameter_tripled_l1572_157254


namespace NUMINAMATH_CALUDE_poly_has_four_nonzero_terms_l1572_157271

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x + 5)*(3*x^2 - x + 4) + 4*(x^3 + x^2 - 6*x)

/-- The expansion of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 10*x^3 + 17*x^2 - 21*x + 20

/-- Theorem stating that the polynomial has exactly 4 nonzero terms -/
theorem poly_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (n : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (∀ x, poly x = expanded_poly x) ∧
    (∀ x, expanded_poly x = a*x^3 + b*x^2 + c*x + d) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_poly_has_four_nonzero_terms_l1572_157271


namespace NUMINAMATH_CALUDE_smallest_n_value_l1572_157238

theorem smallest_n_value (N : ℕ) (c₁ c₂ c₃ c₄ : ℕ) : 
  (c₁ ≤ N) ∧ (c₂ ≤ N) ∧ (c₃ ≤ N) ∧ (c₄ ≤ N) ∧ 
  (c₁ = 4 * c₂ - 3) ∧
  (N + c₂ = 4 * c₄) ∧
  (2 * N + c₃ = 4 * c₃ - 1) ∧
  (3 * N + c₄ = 4 * c₁ - 3) →
  N = 1 ∧ c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 ∧ c₄ = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1572_157238


namespace NUMINAMATH_CALUDE_initial_investment_l1572_157224

/-- Proves that the initial investment is 8000 given the specified conditions -/
theorem initial_investment (x : ℝ) : 
  (0.05 * x + 0.08 * 4000 = 0.06 * (x + 4000)) → x = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_investment_l1572_157224


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1572_157243

/-- The average speed of a round trip, given the outbound speed and relative duration of return trip -/
theorem round_trip_average_speed 
  (outbound_speed : ℝ) 
  (return_time_factor : ℝ) 
  (h1 : outbound_speed = 48) 
  (h2 : return_time_factor = 2) : 
  (2 * outbound_speed) / (1 + return_time_factor) = 32 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1572_157243


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l1572_157292

/-- Represents the neighborhood cable layout problem -/
structure NeighborhoodCable where
  east_west_streets : Nat
  east_west_length : Nat
  north_south_streets : Nat
  north_south_length : Nat
  cable_per_mile : Nat
  cable_cost_per_mile : Nat

/-- Calculates the total cost of cable for the neighborhood -/
def total_cable_cost (n : NeighborhoodCable) : Nat :=
  let total_street_length := n.east_west_streets * n.east_west_length + n.north_south_streets * n.north_south_length
  let total_cable_length := total_street_length * n.cable_per_mile
  total_cable_length * n.cable_cost_per_mile

/-- The theorem stating the total cost of cable for the given neighborhood -/
theorem neighborhood_cable_cost :
  let n : NeighborhoodCable := {
    east_west_streets := 18,
    east_west_length := 2,
    north_south_streets := 10,
    north_south_length := 4,
    cable_per_mile := 5,
    cable_cost_per_mile := 2000
  }
  total_cable_cost n = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l1572_157292


namespace NUMINAMATH_CALUDE_quadratic_function_zero_equivalence_l1572_157244

/-- A quadratic function f(x) = ax² + bx + c where a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value for a given x -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The set of zeros of the function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.value x = 0}

/-- The composition of the function with itself -/
def QuadraticFunction.compose_self (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.value (f.value x)

theorem quadratic_function_zero_equivalence (f : QuadraticFunction) :
  (f.zeros = {x : ℝ | f.compose_self x = 0}) ↔ f.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_zero_equivalence_l1572_157244


namespace NUMINAMATH_CALUDE_expression_equals_eight_l1572_157295

theorem expression_equals_eight :
  ((18^18 / 18^17)^3 * 9^3) / 3^6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l1572_157295


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l1572_157266

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- Theorem statement
theorem sum_of_polynomials (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l1572_157266


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l1572_157232

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- State the theorem
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l1572_157232


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1572_157237

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  x^2 + (1/4) * y^2 ≥ 1/8 := by
sorry

theorem min_value_achieved (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  (x^2 + (1/4) * y^2 = 1/8) ↔ (x = 1/4 ∧ y = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1572_157237


namespace NUMINAMATH_CALUDE_jogging_distance_l1572_157275

/-- Calculates the total distance jogged over a period of days given a constant speed and daily jogging time. -/
def total_distance_jogged (speed : ℝ) (hours_per_day : ℝ) (days : ℕ) : ℝ :=
  speed * hours_per_day * days

/-- Proves that jogging at 5 miles per hour for 2 hours a day for 5 days results in a total distance of 50 miles. -/
theorem jogging_distance : total_distance_jogged 5 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l1572_157275


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1572_157281

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors (x : ℝ) :
  dot_product (λ i => a i + b x i) (λ i => a i - b x i) = 0 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1572_157281


namespace NUMINAMATH_CALUDE_max_value_of_sine_function_l1572_157221

theorem max_value_of_sine_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ (y : ℝ), y = 3 * Real.sin x + 2 ∧ y ≤ 2 ∧ ∃ (x₀ : ℝ), -π/2 ≤ x₀ ∧ x₀ ≤ 0 ∧ 3 * Real.sin x₀ + 2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sine_function_l1572_157221


namespace NUMINAMATH_CALUDE_x_equals_y_l1572_157291

theorem x_equals_y (x y : ℝ) : x = 2 + Real.sqrt 3 → y = 1 / (2 - Real.sqrt 3) → x = y := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l1572_157291


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1572_157207

theorem complex_modulus_problem (z : ℂ) : z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1572_157207


namespace NUMINAMATH_CALUDE_red_peaches_per_basket_l1572_157246

/-- Given 6 baskets of peaches with a total of 96 red peaches,
    prove that each basket contains 16 red peaches. -/
theorem red_peaches_per_basket :
  let total_baskets : ℕ := 6
  let total_red_peaches : ℕ := 96
  let green_peaches_per_basket : ℕ := 18
  (total_red_peaches / total_baskets : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_per_basket_l1572_157246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1572_157269

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 20) 
  (h_a6 : a 6 = 11) : 
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1572_157269


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_collinear_vectors_k_l1572_157294

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication for 2D vectors
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mult k w

-- Theorem 1
theorem perpendicular_vectors_m (m : ℝ) : 
  dot_product a (vector_add b (c m)) = 0 → m = -1 := by sorry

-- Theorem 2
theorem collinear_vectors_k (k : ℝ) :
  collinear (vector_add (scalar_mult k a) b) (vector_add (scalar_mult 2 a) (scalar_mult (-1) b)) → k = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_collinear_vectors_k_l1572_157294


namespace NUMINAMATH_CALUDE_percentage_problem_l1572_157250

theorem percentage_problem : ∃ p : ℝ, (p / 100) * 16 = 0.04 ∧ p = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1572_157250


namespace NUMINAMATH_CALUDE_tower_arrangements_eq_4200_l1572_157286

/-- The number of ways to arrange 9 cubes out of 10 cubes (3 red, 3 blue, 4 green) -/
def tower_arrangements : ℕ := 
  Nat.choose 10 9 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))

/-- Theorem stating that the number of tower arrangements is 4200 -/
theorem tower_arrangements_eq_4200 : tower_arrangements = 4200 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_eq_4200_l1572_157286


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l1572_157247

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- The ratio of the areas of triangles ABC and ADC
  area_ratio : ℚ
  -- The combined length of bases AB and CD
  total_base_length : ℝ
  -- The length of base AB
  ab_length : ℝ

/-- Theorem: If the area ratio is 8:2 and the total base length is 120,
    then the length of AB is 96 -/
theorem trapezoid_ab_length (t : Trapezoid) :
  t.area_ratio = 8 / 2 ∧ t.total_base_length = 120 → t.ab_length = 96 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ab_length_l1572_157247


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l1572_157274

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l1572_157274


namespace NUMINAMATH_CALUDE_max_rectangles_correct_l1572_157202

/-- The maximum number of 1 × (n + 1) rectangles that can be cut from a 2n × 2n square -/
def max_rectangles (n : ℕ) : ℕ :=
  if n ≥ 4 then 4 * (n - 1)
  else if n = 1 then 2
  else if n = 2 then 5
  else 8

theorem max_rectangles_correct (n : ℕ) :
  max_rectangles n = 
    if n ≥ 4 then 4 * (n - 1)
    else if n = 1 then 2
    else if n = 2 then 5
    else 8 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_correct_l1572_157202


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1572_157203

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Main theorem about the ellipse and the maximum product -/
theorem ellipse_theorem (C : Ellipse) (P Q : Point) (l : Line) (F : Point) (circle : Circle) :
  (P.x = 1 ∧ P.y = Real.sqrt 2 / 2) →
  (Q.x = -Real.sqrt 2 ∧ Q.y = 0) →
  (C.a^2 * P.y^2 + C.b^2 * P.x^2 = C.a^2 * C.b^2) →
  (C.a^2 * Q.y^2 + C.b^2 * Q.x^2 = C.a^2 * C.b^2) →
  (∃ (A B E : Point), A ≠ B ∧ E ≠ F ∧
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) ∧
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) ∧
    (l.p1 = F ∧ l.p2 = A) ∧
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2)) →
  (C.a = Real.sqrt 2 ∧ C.b = 1) ∧
  (∀ (A B E : Point), 
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) →
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) →
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2) →
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) * Real.sqrt ((F.x - E.x)^2 + (F.y - E.y)^2) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1572_157203


namespace NUMINAMATH_CALUDE_jade_ball_problem_l1572_157231

/-- Represents the state of boxes as a list of natural numbers (0-6) -/
def BoxState := List Nat

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : Nat) : BoxState :=
  sorry

/-- Counts the number of carries (resets) needed to increment from 1 to n in base 7 -/
def countCarries (n : Nat) : Nat :=
  sorry

/-- Sums the digits in a BoxState -/
def sumDigits (state : BoxState) : Nat :=
  sorry

theorem jade_ball_problem (n : Nat) : 
  n = 1876 → 
  sumDigits (toBase7 n) = 10 ∧ 
  countCarries n = 3 := by
  sorry

end NUMINAMATH_CALUDE_jade_ball_problem_l1572_157231


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l1572_157235

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least one white ball when randomly selecting 3 balls from a bag 
    containing 5 white balls and 4 red balls -/
theorem prob_at_least_one_white : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l1572_157235


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l1572_157210

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  7 * p^6 * q = 21 * p^5 * q^2 → 
  p = 3/4 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l1572_157210


namespace NUMINAMATH_CALUDE_otts_money_fraction_l1572_157226

theorem otts_money_fraction (moe loki nick ott : ℚ) : 
  moe > 0 → loki > 0 → nick > 0 → ott = 0 →
  ∃ (x : ℚ), x > 0 ∧ 
    x = moe / 3 ∧ 
    x = loki / 5 ∧ 
    x = nick / 4 →
  (3 * x) / (moe + loki + nick + 3 * x) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_otts_money_fraction_l1572_157226


namespace NUMINAMATH_CALUDE_playground_ball_cost_l1572_157253

theorem playground_ball_cost (jump_rope_cost board_game_cost savings_from_allowance savings_from_uncle additional_needed : ℕ) :
  jump_rope_cost = 7 →
  board_game_cost = 12 →
  savings_from_allowance = 6 →
  savings_from_uncle = 13 →
  additional_needed = 4 →
  ∃ (playground_ball_cost : ℕ),
    playground_ball_cost = 4 ∧
    jump_rope_cost + board_game_cost + playground_ball_cost = savings_from_allowance + savings_from_uncle + additional_needed :=
by sorry

end NUMINAMATH_CALUDE_playground_ball_cost_l1572_157253


namespace NUMINAMATH_CALUDE_eighteen_games_equation_l1572_157288

/-- The number of games in a competition where each pair of teams plays once. -/
def numGames (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for x teams, 18 total games is equivalent to the equation 1/2 * x * (x-1) = 18 -/
theorem eighteen_games_equation (x : ℕ) :
  numGames x = 18 ↔ (x * (x - 1)) / 2 = 18 := by sorry

end NUMINAMATH_CALUDE_eighteen_games_equation_l1572_157288


namespace NUMINAMATH_CALUDE_problem_solution_l1572_157262

theorem problem_solution : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1572_157262


namespace NUMINAMATH_CALUDE_total_weight_AlF3_is_839_8_l1572_157233

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in AlF3 -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in AlF3 -/
def num_F : ℕ := 3

/-- The number of moles of AlF3 -/
def num_moles : ℝ := 10

/-- The molecular weight of AlF3 in g/mol -/
def molecular_weight_AlF3 : ℝ := atomic_weight_Al * num_Al + atomic_weight_F * num_F

/-- The total weight of AlF3 in grams -/
def total_weight_AlF3 : ℝ := molecular_weight_AlF3 * num_moles

theorem total_weight_AlF3_is_839_8 : total_weight_AlF3 = 839.8 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_AlF3_is_839_8_l1572_157233


namespace NUMINAMATH_CALUDE_percent_of_x_l1572_157200

theorem percent_of_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y + z)) 
  (h2 : 0.4 * (y - z) = 0.2 * (y + x - z)) : 
  y - z = x := by sorry

end NUMINAMATH_CALUDE_percent_of_x_l1572_157200


namespace NUMINAMATH_CALUDE_train_combined_speed_l1572_157290

/-- The combined speed of two trains crossing a bridge simultaneously -/
theorem train_combined_speed
  (bridge_length : ℝ)
  (train1_length train1_time : ℝ)
  (train2_length train2_time : ℝ)
  (h1 : bridge_length = 300)
  (h2 : train1_length = 100)
  (h3 : train1_time = 30)
  (h4 : train2_length = 150)
  (h5 : train2_time = 45) :
  (train1_length + bridge_length) / train1_time +
  (train2_length + bridge_length) / train2_time =
  23.33 :=
sorry

end NUMINAMATH_CALUDE_train_combined_speed_l1572_157290


namespace NUMINAMATH_CALUDE_negation_equivalence_l1572_157229

theorem negation_equivalence (p q : Prop) : 
  let m := p ∧ q
  (¬p ∨ ¬q) ↔ ¬m := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1572_157229


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1572_157251

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1572_157251


namespace NUMINAMATH_CALUDE_base_6_number_identification_l1572_157278

def is_base_6_digit (d : ℕ) : Prop := d < 6

def is_base_6_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_base_6_digit d

theorem base_6_number_identification :
  ¬ is_base_6_number 66 ∧
  ¬ is_base_6_number 207 ∧
  ¬ is_base_6_number 652 ∧
  is_base_6_number 3142 :=
sorry

end NUMINAMATH_CALUDE_base_6_number_identification_l1572_157278


namespace NUMINAMATH_CALUDE_inflection_points_collinear_l1572_157259

/-- The function f(x) = 9x^5 - 30x^3 + 19x -/
def f (x : ℝ) : ℝ := 9*x^5 - 30*x^3 + 19*x

/-- The inflection points of f(x) -/
def inflection_points : List (ℝ × ℝ) := [(-1, 2), (0, 0), (1, -2)]

/-- Theorem: The inflection points of f(x) are collinear -/
theorem inflection_points_collinear : 
  let points := inflection_points
  ∃ (m c : ℝ), ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + c :=
by sorry

end NUMINAMATH_CALUDE_inflection_points_collinear_l1572_157259


namespace NUMINAMATH_CALUDE_equation_solutions_l1572_157256

-- Define the equation
def equation (m n : ℕ+) : Prop := 3^(m.val) - 2^(n.val) = 1

-- State the theorem
theorem equation_solutions :
  ∀ m n : ℕ+, equation m n ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1572_157256


namespace NUMINAMATH_CALUDE_cross_product_result_l1572_157225

def a : ℝ × ℝ × ℝ := (4, 2, -1)
def b : ℝ × ℝ × ℝ := (3, -5, 6)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result :
  cross_product a b = (7, -27, -26) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l1572_157225


namespace NUMINAMATH_CALUDE_product_of_numbers_l1572_157220

theorem product_of_numbers (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y / x = 15) (h4 : x + y = 400) : x * y = 9375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1572_157220


namespace NUMINAMATH_CALUDE_complex_absolute_value_squared_l1572_157239

theorem complex_absolute_value_squared : 
  (Complex.abs (-3 - (8/5)*Complex.I))^2 = 289/25 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_squared_l1572_157239


namespace NUMINAMATH_CALUDE_manny_marbles_l1572_157240

/-- Given a total of 120 marbles distributed in the ratio 4:5:6,
    prove that the person with the middle ratio (5) receives 40 marbles. -/
theorem manny_marbles (total : ℕ) (ratio_sum : ℕ) (manny_ratio : ℕ) :
  total = 120 →
  ratio_sum = 4 + 5 + 6 →
  manny_ratio = 5 →
  manny_ratio * (total / ratio_sum) = 40 :=
by sorry

end NUMINAMATH_CALUDE_manny_marbles_l1572_157240


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l1572_157252

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem solution_set_f_leq_6 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_f_plus_g_geq_3 :
  {a : ℝ | ∀ x, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l1572_157252


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1572_157289

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1572_157289


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1572_157211

theorem pure_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 3) = (x^2 + 2*x - 3) + Complex.I * (x + 3)) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1572_157211


namespace NUMINAMATH_CALUDE_houses_on_one_side_l1572_157282

theorem houses_on_one_side (x : ℕ) : 
  x + 3 * x = 160 → x = 40 := by sorry

end NUMINAMATH_CALUDE_houses_on_one_side_l1572_157282


namespace NUMINAMATH_CALUDE_negative_quartic_count_l1572_157264

theorem negative_quartic_count :
  (∃! (s : Finset ℤ), (∀ x ∈ s, x^4 - 51*x^2 + 100 < 0) ∧ s.card = 10) := by
  sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l1572_157264


namespace NUMINAMATH_CALUDE_point_below_right_of_line_range_of_a_below_right_of_line_l1572_157283

/-- A point (a, 1) is below and to the right of the line x-2y+4=0 if and only if a > -2 -/
theorem point_below_right_of_line (a : ℝ) : 
  (a - 2 * 1 + 4 > 0) ↔ (a > -2) :=
sorry

/-- The range of a for points (a, 1) below and to the right of the line x-2y+4=0 is (-2, +∞) -/
theorem range_of_a_below_right_of_line : 
  {a : ℝ | a - 2 * 1 + 4 > 0} = Set.Ioi (-2) :=
sorry

end NUMINAMATH_CALUDE_point_below_right_of_line_range_of_a_below_right_of_line_l1572_157283


namespace NUMINAMATH_CALUDE_millionth_digit_of_three_forty_first_l1572_157208

def fraction : ℚ := 3 / 41

def decimal_expansion (q : ℚ) : ℕ → ℕ := sorry

def nth_digit_after_decimal_point (q : ℚ) (n : ℕ) : ℕ :=
  decimal_expansion q n

theorem millionth_digit_of_three_forty_first (n : ℕ) (h : n = 1000000) :
  nth_digit_after_decimal_point fraction n = 7 := by sorry

end NUMINAMATH_CALUDE_millionth_digit_of_three_forty_first_l1572_157208


namespace NUMINAMATH_CALUDE_diploma_percentage_theorem_l1572_157258

/-- Represents the four income groups in country Z -/
inductive IncomeGroup
  | Low
  | LowerMiddle
  | UpperMiddle
  | High

/-- Returns the population percentage for a given income group -/
def population_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.25
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.25
  | IncomeGroup.High => 0.15

/-- Returns the percentage of people with a university diploma for a given income group -/
def diploma_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.05
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.60
  | IncomeGroup.High => 0.80

/-- Calculates the total percentage of the population with a university diploma -/
def total_diploma_percentage : Real :=
  (population_percentage IncomeGroup.Low * diploma_percentage IncomeGroup.Low) +
  (population_percentage IncomeGroup.LowerMiddle * diploma_percentage IncomeGroup.LowerMiddle) +
  (population_percentage IncomeGroup.UpperMiddle * diploma_percentage IncomeGroup.UpperMiddle) +
  (population_percentage IncomeGroup.High * diploma_percentage IncomeGroup.High)

theorem diploma_percentage_theorem :
  total_diploma_percentage = 0.405 := by
  sorry

end NUMINAMATH_CALUDE_diploma_percentage_theorem_l1572_157258


namespace NUMINAMATH_CALUDE_race_car_cost_l1572_157215

theorem race_car_cost (mater_cost sally_cost race_car_cost : ℝ) : 
  mater_cost = 0.1 * race_car_cost →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  race_car_cost = 140000 := by
sorry

end NUMINAMATH_CALUDE_race_car_cost_l1572_157215


namespace NUMINAMATH_CALUDE_remainder_8347_mod_9_l1572_157242

theorem remainder_8347_mod_9 : 8347 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8347_mod_9_l1572_157242
