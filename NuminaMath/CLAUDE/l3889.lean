import Mathlib

namespace NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l3889_388944

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * π + θ) + 2 * Real.sin ((11 / 10) * π - θ) = 0) : 
  Real.tan ((2 / 5) * π + θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l3889_388944


namespace NUMINAMATH_CALUDE_monroe_collection_legs_l3889_388974

def spider_count : ℕ := 8
def ant_count : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

def total_legs : ℕ := spider_count * spider_legs + ant_count * ant_legs

theorem monroe_collection_legs : total_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_monroe_collection_legs_l3889_388974


namespace NUMINAMATH_CALUDE_fraction_simplification_l3889_388936

theorem fraction_simplification (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = (100 * a + a) / (100 * b + b)) 
  (h4 : a / b = (10000 * a + 100 * a + a) / (10000 * b + 100 * b + b)) :
  ∀ (d : ℕ), d > 1 → d ∣ a → d ∣ b → False :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3889_388936


namespace NUMINAMATH_CALUDE_solve_equation_l3889_388977

-- Define the * operation
def star (a b : ℚ) : ℚ := 2 * a + 3 * b

-- Theorem statement
theorem solve_equation (x : ℚ) :
  star 5 (star 7 x) = -4 → x = -56/9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3889_388977


namespace NUMINAMATH_CALUDE_power_function_increasing_exponent_l3889_388991

theorem power_function_increasing_exponent (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → x^a < y^a) → a > 0 := by sorry

end NUMINAMATH_CALUDE_power_function_increasing_exponent_l3889_388991


namespace NUMINAMATH_CALUDE_total_notes_count_l3889_388993

/-- Given a total amount of 400 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes, 
    the total number of notes is 75. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 400 →
  note_count * (1 + 5 + 10) = total_amount →
  3 * note_count = 75 := by
  sorry

#check total_notes_count

end NUMINAMATH_CALUDE_total_notes_count_l3889_388993


namespace NUMINAMATH_CALUDE_tamara_kim_height_ratio_l3889_388949

/-- Given Tamara's height and the combined height of Tamara and Kim, 
    prove that Tamara is 17/6 times taller than Kim. -/
theorem tamara_kim_height_ratio :
  ∀ (tamara_height kim_height : ℝ),
    tamara_height = 68 →
    tamara_height + kim_height = 92 →
    tamara_height / kim_height = 17 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tamara_kim_height_ratio_l3889_388949


namespace NUMINAMATH_CALUDE_ratio_of_negatives_l3889_388958

theorem ratio_of_negatives (x y : ℝ) (hx : x < 0) (hy : y < 0) (h : 3 * x - 2 * y = Real.sqrt (x * y)) : 
  y / x = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_negatives_l3889_388958


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3889_388916

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3889_388916


namespace NUMINAMATH_CALUDE_driving_time_ratio_l3889_388938

theorem driving_time_ratio : 
  ∀ (t_28 t_60 : ℝ),
  t_28 + t_60 = 30 →
  t_28 * 28 + t_60 * 60 = 11 * 120 →
  t_28 = 15 ∧ t_28 / (t_28 + t_60) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_driving_time_ratio_l3889_388938


namespace NUMINAMATH_CALUDE_subset_condition_l3889_388997

def A : Set ℝ := {x | (x - 3) / (x + 1) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : 
  B a ⊆ A ↔ a ∈ Set.Icc (-1/3) 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3889_388997


namespace NUMINAMATH_CALUDE_bicycle_price_l3889_388985

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) :
  upfront_payment = 200 →
  upfront_percentage = 0.20 →
  upfront_payment = upfront_percentage * total_price →
  total_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_l3889_388985


namespace NUMINAMATH_CALUDE_range_of_a_l3889_388973

def p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1

def q (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ∧ 
  (∃ x : ℝ, ¬p x ∧ q a x)) → 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2) ∧ 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 → 
    (∀ x : ℝ, p x → q a x) ∧ 
    (∃ x : ℝ, ¬p x ∧ q a x)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3889_388973


namespace NUMINAMATH_CALUDE_problem_solution_l3889_388905

theorem problem_solution : 
  ∀ M : ℚ, (5 + 7 + 9) / 3 = (2005 + 2007 + 2009) / M → M = 860 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3889_388905


namespace NUMINAMATH_CALUDE_multiply_polynomial_difference_of_cubes_l3889_388946

theorem multiply_polynomial_difference_of_cubes (x : ℝ) :
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_difference_of_cubes_l3889_388946


namespace NUMINAMATH_CALUDE_books_remaining_l3889_388920

/-- Given Sandy has 10 books, Tim has 33 books, and Benny lost 24 of their books,
    prove that they have 19 books together now. -/
theorem books_remaining (sandy_books tim_books lost_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : lost_books = 24) : 
  sandy_books + tim_books - lost_books = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_l3889_388920


namespace NUMINAMATH_CALUDE_company_blocks_l3889_388933

/-- Calculates the number of blocks in a company based on gift budget and workers per block -/
theorem company_blocks (total_amount : ℝ) (gift_worth : ℝ) (workers_per_block : ℝ) :
  total_amount = 4000 ∧ gift_worth = 4 ∧ workers_per_block = 100 →
  (total_amount / gift_worth) / workers_per_block = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l3889_388933


namespace NUMINAMATH_CALUDE_unique_age_group_split_l3889_388956

theorem unique_age_group_split (total_students : ℕ) 
  (under_10_fraction : ℚ) (between_10_12_fraction : ℚ) (between_12_14_fraction : ℚ) :
  total_students = 60 →
  under_10_fraction = 1/4 →
  between_10_12_fraction = 1/2 →
  between_12_14_fraction = 1/6 →
  ∃! (under_10 between_10_12 between_12_14 above_14 : ℕ),
    under_10 + between_10_12 + between_12_14 + above_14 = total_students ∧
    under_10 = (under_10_fraction * total_students).num ∧
    between_10_12 = (between_10_12_fraction * total_students).num ∧
    between_12_14 = (between_12_14_fraction * total_students).num ∧
    above_14 = total_students - (under_10 + between_10_12 + between_12_14) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_age_group_split_l3889_388956


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3889_388975

theorem fraction_sum_equals_decimal : 
  (4 : ℚ) / 100 - 8 / 10 + 3 / 1000 + 2 / 10000 = -0.7568 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3889_388975


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3889_388972

theorem middle_part_of_proportional_division (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = (1 : ℝ) / 2 ∧ c = (1 : ℝ) / 4 →
  ∃ x : ℝ, a * x + b * x + c * x = total ∧ b * x = 20.8 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3889_388972


namespace NUMINAMATH_CALUDE_total_birds_is_148_l3889_388903

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_148_l3889_388903


namespace NUMINAMATH_CALUDE_homework_problem_count_l3889_388909

theorem homework_problem_count (math_pages reading_pages problems_per_page : ℕ) : 
  math_pages = 4 → reading_pages = 6 → problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 := by
sorry

end NUMINAMATH_CALUDE_homework_problem_count_l3889_388909


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3889_388976

/-- 
Given a quadratic equation kx^2 - 6x + 9 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 1
and not equal to 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6*x + 9 = 0 ∧ k * y^2 - 6*y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3889_388976


namespace NUMINAMATH_CALUDE_specific_gathering_interactions_l3889_388934

/-- The number of interactions in a gathering of witches and zombies -/
def interactions (num_witches num_zombies : ℕ) : ℕ :=
  (num_zombies * (num_zombies - 1)) / 2 + num_witches * num_zombies

/-- Theorem stating the number of interactions in a specific gathering -/
theorem specific_gathering_interactions :
  interactions 25 18 = 603 := by
  sorry

end NUMINAMATH_CALUDE_specific_gathering_interactions_l3889_388934


namespace NUMINAMATH_CALUDE_a_range_l3889_388984

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem a_range (a : ℝ) : (A ∩ B a).Nonempty → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3889_388984


namespace NUMINAMATH_CALUDE_triangle_side_length_l3889_388983

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle_BAC := Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))
  AC = 4 ∧ BC = 2 * Real.sqrt 7 ∧ angle_BAC = π / 3 → AB = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3889_388983


namespace NUMINAMATH_CALUDE_jerry_sticker_count_jerry_has_36_stickers_l3889_388964

theorem jerry_sticker_count (fred_stickers : ℕ) (george_diff : ℕ) (jerry_multiplier : ℕ) : ℕ :=
  let george_stickers := fred_stickers - george_diff
  let jerry_stickers := jerry_multiplier * george_stickers
  jerry_stickers

theorem jerry_has_36_stickers : 
  jerry_sticker_count 18 6 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_sticker_count_jerry_has_36_stickers_l3889_388964


namespace NUMINAMATH_CALUDE_principal_is_600_l3889_388947

/-- Proves that given the conditions of the problem, the principal amount is 600 --/
theorem principal_is_600 (P R : ℝ) (h : (P * (R + 4) * 6) / 100 - (P * R * 6) / 100 = 144) : P = 600 := by
  sorry

#check principal_is_600

end NUMINAMATH_CALUDE_principal_is_600_l3889_388947


namespace NUMINAMATH_CALUDE_m2_defective_percent_is_one_percent_l3889_388927

-- Define the percentage of products from each machine
def m1_percent : ℝ := 0.4
def m2_percent : ℝ := 0.3
def m3_percent : ℝ := 1 - m1_percent - m2_percent

-- Define the percentage of defective products for m1 and m3
def m1_defective_percent : ℝ := 0.03
def m3_non_defective_percent : ℝ := 0.93

-- Define the total percentage of defective products
def total_defective_percent : ℝ := 0.036

-- State the theorem
theorem m2_defective_percent_is_one_percent :
  ∃ (m2_defective_percent : ℝ),
    m2_defective_percent = 0.01 ∧
    m1_percent * m1_defective_percent +
    m2_percent * m2_defective_percent +
    m3_percent * (1 - m3_non_defective_percent) =
    total_defective_percent :=
sorry

end NUMINAMATH_CALUDE_m2_defective_percent_is_one_percent_l3889_388927


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3889_388945

theorem min_value_x_plus_2y (x y : ℝ) (hx : x + 2 > 0) (hy : y + 2 > 0) 
  (h : 3 / (x + 2) + 3 / (y + 2) = 1) : 
  x + 2*y ≥ 3 + 6 * Real.sqrt 2 := by
  sorry

#check min_value_x_plus_2y

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3889_388945


namespace NUMINAMATH_CALUDE_race_car_probability_l3889_388980

/-- A circular racetrack with a given length -/
structure CircularTrack where
  length : ℝ
  length_positive : length > 0

/-- A race car on the circular track -/
structure RaceCar where
  track : CircularTrack
  start_position : ℝ
  travel_distance : ℝ

/-- The probability of the car ending within a certain distance of a specific point -/
def end_probability (car : RaceCar) (target : ℝ) (range : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability for the specific problem -/
theorem race_car_probability (track : CircularTrack) 
  (h1 : track.length = 3)
  (car : RaceCar)
  (h2 : car.track = track)
  (h3 : car.travel_distance = 0.5)
  (target : ℝ)
  (h4 : target = 2.5) :
  end_probability car target 0.5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_race_car_probability_l3889_388980


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3889_388978

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x + 1) * (x - 4) = p x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3889_388978


namespace NUMINAMATH_CALUDE_projectile_max_height_l3889_388965

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 feet -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3889_388965


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l3889_388960

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  a + b = 70 := by
sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l3889_388960


namespace NUMINAMATH_CALUDE_max_divisible_arrangement_l3889_388930

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i : ℕ, i < arr.length - 1 → 
    is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i+1, by sorry⟩) ∨ 
    is_divisible (arr.get ⟨i+1, by sorry⟩) (arr.get ⟨i, by sorry⟩)

def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem max_divisible_arrangement :
  (∃ (arr : List ℕ), arr.length = 8 ∧ 
    (∀ x ∈ arr, x ∈ cards) ∧ 
    valid_arrangement arr) ∧
  (∀ (arr : List ℕ), arr.length > 8 → 
    (∀ x ∈ arr, x ∈ cards) → 
    ¬valid_arrangement arr) := by sorry

end NUMINAMATH_CALUDE_max_divisible_arrangement_l3889_388930


namespace NUMINAMATH_CALUDE_problem_solution_l3889_388970

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem problem_solution :
  (f (Real.pi / 6) = 0) ∧
  (∀ α : ℝ, α > 0 ∧ α < Real.pi ∧ f (α / 2) = 1/4 - Real.sqrt 3 / 2 → Real.sin α = (1 + 3 * Real.sqrt 5) / 8) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3889_388970


namespace NUMINAMATH_CALUDE_original_number_proof_l3889_388926

theorem original_number_proof :
  ∃ (a x y q : ℕ), 
    7 * a = 10 * x + y ∧
    y ≤ 9 ∧
    9 * x = 80 + q ∧
    q ≤ 9 ∧
    (a = 13 ∨ a = 14) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l3889_388926


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3889_388937

/-- A pentagon formed by cutting a triangular corner from a rectangular sheet. -/
structure CornerCutPentagon where
  sides : Finset ℝ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of a CornerCutPentagon is 759.5 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ∃ (area : ℝ), area = 759.5 := by
  sorry

#check corner_cut_pentagon_area

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3889_388937


namespace NUMINAMATH_CALUDE_greatest_n_squared_l3889_388989

theorem greatest_n_squared (n : ℤ) (V : ℝ) : 
  (∀ m : ℤ, 102 * m^2 ≤ V → m ≤ 8) →
  (102 * 8^2 ≤ V) →
  V = 6528 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_squared_l3889_388989


namespace NUMINAMATH_CALUDE_digit_sum_power_property_l3889_388954

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The property that the fifth power of the sum of digits equals the square of the number -/
def has_property (n : ℕ) : Prop := (sum_of_digits n)^5 = n^2

/-- Theorem stating that only 1 and 243 satisfy the property -/
theorem digit_sum_power_property :
  ∀ n : ℕ, has_property n ↔ n = 1 ∨ n = 243 := by sorry

end NUMINAMATH_CALUDE_digit_sum_power_property_l3889_388954


namespace NUMINAMATH_CALUDE_triangle_sine_relations_l3889_388900

theorem triangle_sine_relations (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = 7 * a * Real.sin B →
  Real.sin A = 1/7 ∧ 
  (B = π/3 → Real.sin C = 13/14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_relations_l3889_388900


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3889_388932

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals :
  repeating_decimal_4 + repeating_decimal_7 - repeating_decimal_3 = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3889_388932


namespace NUMINAMATH_CALUDE_power_equation_l3889_388990

theorem power_equation (y : ℝ) : (12 : ℝ)^2 * 6^y / 432 = 72 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l3889_388990


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l3889_388904

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l3889_388904


namespace NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l3889_388998

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 4 * ((n - 1) / 2) + 2 else 4 * (n / 2 - 1) + 3

def b : ℕ → ℕ
  | n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2 - 1) + 6

theorem odd_square_sum_of_consecutive_b (k : ℕ) (hk : k > 0) :
  ∃ r : ℕ, (2 * k + 1)^2 = b r + b (r + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l3889_388998


namespace NUMINAMATH_CALUDE_elena_pen_purchase_l3889_388979

theorem elena_pen_purchase (cost_x : ℝ) (cost_y : ℝ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_x = 4 →
  cost_y = 2.8 →
  total_pens = 12 →
  total_cost = 40 →
  ∃ (x y : ℕ), x + y = total_pens ∧ x * cost_x + y * cost_y = total_cost ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_elena_pen_purchase_l3889_388979


namespace NUMINAMATH_CALUDE_bounded_function_periodic_l3889_388907

/-- A bounded real function satisfying a specific functional equation is periodic with period 1. -/
theorem bounded_function_periodic (f : ℝ → ℝ) 
  (hbounded : ∃ M, ∀ x, |f x| ≤ M) 
  (hcond : ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)) : 
  ∀ x, f (x + 1) = f x := by
  sorry

end NUMINAMATH_CALUDE_bounded_function_periodic_l3889_388907


namespace NUMINAMATH_CALUDE_line_configuration_theorem_l3889_388971

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The theorem statement -/
theorem line_configuration_theorem (config : LineConfiguration) :
  config.n > 0 ∧
  config.total_intersections = 16 ∧
  config.triple_intersections = 6 ∧
  (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) ∧
  (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n →
    i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
    ¬∃ (p : ℕ), p < config.total_intersections) →
  config.n = 8 :=
sorry

end NUMINAMATH_CALUDE_line_configuration_theorem_l3889_388971


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3889_388928

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3889_388928


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3889_388914

theorem quadratic_inequality (x : ℝ) : -9 * x^2 + 6 * x + 8 > 0 ↔ -2/3 < x ∧ x < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3889_388914


namespace NUMINAMATH_CALUDE_worksheets_graded_l3889_388995

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3889_388995


namespace NUMINAMATH_CALUDE_smallest_divisible_by_999_l3889_388943

theorem smallest_divisible_by_999 :
  ∃ (a : ℕ), (∀ (n : ℕ), Odd n → (999 ∣ 2^(5*n) + a*5^n)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (m : ℕ), Odd m ∧ ¬(999 ∣ 2^(5*m) + b*5^m)) ∧
  a = 539 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_999_l3889_388943


namespace NUMINAMATH_CALUDE_product_inequality_l3889_388931

theorem product_inequality (a b c x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (sum_abc : a + b + c = 1)
  (prod_x : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁^2 + b * x₁ + c) * 
  (a * x₂^2 + b * x₂ + c) * 
  (a * x₃^2 + b * x₃ + c) * 
  (a * x₄^2 + b * x₄ + c) * 
  (a * x₅^2 + b * x₅ + c) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l3889_388931


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3889_388950

theorem sqrt_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.sqrt x ≠ 1 ∧ Real.sqrt x + 1 = 1 / (Real.sqrt x - 1) ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3889_388950


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l3889_388901

/-- The function f(x) = -x³ + x² + x - 2 -/
def f (x : ℝ) := -x^3 + x^2 + x - 2

/-- The theorem stating that f has exactly one zero in (-∞, -1/3) -/
theorem f_has_unique_zero_in_interval :
  ∃! x, x < -1/3 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l3889_388901


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l3889_388940

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 7, -3]

-- Define the original region's area
def S_area : ℝ := 10

-- Theorem statement
theorem transformed_area_theorem :
  let det := Matrix.det A
  let scale_factor := |det|
  scale_factor * S_area = 130 := by sorry

end NUMINAMATH_CALUDE_transformed_area_theorem_l3889_388940


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l3889_388951

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l3889_388951


namespace NUMINAMATH_CALUDE_fraction_transformation_l3889_388924

theorem fraction_transformation (x : ℝ) (h : x ≠ 3) : -1 / (3 - x) = 1 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3889_388924


namespace NUMINAMATH_CALUDE_fourth_root_logarithm_equality_l3889_388913

theorem fourth_root_logarithm_equality : 
  (16 ^ 3) ^ (1/4) - (25/4) ^ (1/2) + (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_logarithm_equality_l3889_388913


namespace NUMINAMATH_CALUDE_williams_books_l3889_388925

theorem williams_books (w : ℕ) : 
  (3 * w + 8 + 4 = w + 2 * 8) → w = 2 := by
  sorry

end NUMINAMATH_CALUDE_williams_books_l3889_388925


namespace NUMINAMATH_CALUDE_elvis_songwriting_time_l3889_388918

/-- Given Elvis's album recording scenario, prove that the time spent writing each song is 15 minutes. -/
theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time_per_song : ℕ) (total_editing_time : ℕ) :
  total_songs = 10 →
  studio_time = 5 * 60 →
  recording_time_per_song = 12 →
  total_editing_time = 30 →
  (studio_time - (total_songs * recording_time_per_song + total_editing_time)) / total_songs = 15 :=
by sorry

end NUMINAMATH_CALUDE_elvis_songwriting_time_l3889_388918


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l3889_388948

/-- The number of distinct permutations of the word BANANA -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l3889_388948


namespace NUMINAMATH_CALUDE_f_properties_l3889_388923

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 2) ∧
  (∃ x : ℝ, f a x + f a (2*x) < 1/2 ↔ -1 < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3889_388923


namespace NUMINAMATH_CALUDE_valentines_given_to_children_l3889_388917

/-- The number of Valentines Mrs. Wong had initially -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong was left with -/
def remaining_valentines : ℕ := 22

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := initial_valentines - remaining_valentines

theorem valentines_given_to_children :
  given_valentines = 8 :=
by sorry

end NUMINAMATH_CALUDE_valentines_given_to_children_l3889_388917


namespace NUMINAMATH_CALUDE_f_of_3_eq_neg_1_l3889_388941

-- Define the function f
def f (x : ℝ) : ℝ := 
  let t := 2 * x + 1
  x^2 - 2*x

-- Theorem statement
theorem f_of_3_eq_neg_1 : f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_neg_1_l3889_388941


namespace NUMINAMATH_CALUDE_line_equation_correct_l3889_388935

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if a line equation represents a line with a given slope -/
def hasSlope (eq : LineEquation) (m : ℝ) : Prop :=
  eq.b ≠ 0 ∧ -eq.a / eq.b = m

theorem line_equation_correct (l : Line) (eq : LineEquation) :
    l.point = (2, 1) →
    l.slope = -2 →
    satisfiesEquation l.point eq →
    hasSlope eq l.slope →
    eq = { a := 2, b := 1, c := -5 } :=
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l3889_388935


namespace NUMINAMATH_CALUDE_max_power_under_500_l3889_388952

theorem max_power_under_500 :
  ∃ (c d : ℕ), d > 1 ∧ c^d < 500 ∧
  (∀ (x y : ℕ), y > 1 → x^y < 500 → x^y ≤ c^d) ∧
  c + d = 24 :=
sorry

end NUMINAMATH_CALUDE_max_power_under_500_l3889_388952


namespace NUMINAMATH_CALUDE_correct_marked_price_l3889_388982

/-- Represents the pricing structure of a book -/
structure BookPricing where
  cost_price : ℝ
  marked_price : ℝ
  first_discount_rate : ℝ
  additional_discount_rate : ℝ
  profit_rate : ℝ
  commission_rate : ℝ

/-- Calculates the final selling price after all discounts and commissions -/
def final_selling_price (b : BookPricing) : ℝ :=
  let price_after_first_discount := b.marked_price * (1 - b.first_discount_rate)
  let price_after_additional_discount := price_after_first_discount * (1 - b.additional_discount_rate)
  let commission := price_after_first_discount * b.commission_rate
  price_after_additional_discount + commission

/-- Theorem stating the correct marked price for the given conditions -/
theorem correct_marked_price :
  ∃ (b : BookPricing),
    b.cost_price = 75 ∧
    b.first_discount_rate = 0.12 ∧
    b.additional_discount_rate = 0.05 ∧
    b.profit_rate = 0.3 ∧
    b.commission_rate = 0.1 ∧
    b.marked_price = 99.35 ∧
    final_selling_price b = b.cost_price * (1 + b.profit_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_marked_price_l3889_388982


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3889_388959

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Counts the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def countDistributions (balls : Nat) (boxes : Nat) : Nat :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : countDistributions 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3889_388959


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3889_388929

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = 740 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 ≥ 740) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3889_388929


namespace NUMINAMATH_CALUDE_line_through_center_perpendicular_to_axis_l3889_388921

/-- The polar equation of a circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- The center of the circle in Cartesian coordinates -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The polar equation of the line -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- The line passes through the center of the circle and is perpendicular to the polar axis -/
theorem line_through_center_perpendicular_to_axis :
  (∀ ρ θ : ℝ, circle_equation ρ θ → line_equation ρ θ) ∧
  (line_equation (circle_center.1) 0) ∧
  (∀ ρ : ℝ, line_equation ρ (Real.pi / 2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_center_perpendicular_to_axis_l3889_388921


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3889_388910

theorem merchant_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := list_price * 0.7
  let marked_price := list_price * 1.25
  let selling_price := marked_price * 0.8
  let profit := selling_price - purchase_price
  profit = selling_price * 0.3 := by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3889_388910


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3889_388942

theorem arithmetic_sequence_sum : 
  ∀ (a₁ l d : ℤ) (n : ℕ),
    a₁ = -48 →
    l = 0 →
    d = 2 →
    n = 25 →
    l = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + l) / 2 = -600 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3889_388942


namespace NUMINAMATH_CALUDE_percentage_free_lunch_l3889_388922

/-- Proves that 40% of students receive a free lunch given the specified conditions --/
theorem percentage_free_lunch (total_students : ℕ) (total_cost : ℚ) (paying_price : ℚ) :
  total_students = 50 →
  total_cost = 210 →
  paying_price = 7 →
  (∃ (paying_students : ℕ), paying_students * paying_price = total_cost) →
  (total_students - (total_cost / paying_price : ℚ)) / total_students = 2/5 := by
  sorry

#check percentage_free_lunch

end NUMINAMATH_CALUDE_percentage_free_lunch_l3889_388922


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l3889_388992

theorem subset_implies_a_value (A B : Set ℝ) (a : ℝ) :
  A = {-3} →
  B = {x : ℝ | a * x + 1 = 0} →
  B ⊆ A →
  a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l3889_388992


namespace NUMINAMATH_CALUDE_temperature_change_over_700_years_l3889_388981

/-- Calculates the total temperature change over 700 years and converts it to Fahrenheit. -/
theorem temperature_change_over_700_years :
  let rate1 : ℝ := 3  -- rate for first 300 years (units per century)
  let rate2 : ℝ := 5  -- rate for next 200 years (units per century)
  let rate3 : ℝ := 2  -- rate for last 200 years (units per century)
  let period1 : ℝ := 3  -- first period in centuries
  let period2 : ℝ := 2  -- second period in centuries
  let period3 : ℝ := 2  -- third period in centuries
  let total_change_celsius : ℝ := rate1 * period1 + rate2 * period2 + rate3 * period3
  let total_change_fahrenheit : ℝ := total_change_celsius * (9/5) + 32
  total_change_celsius = 23 ∧ total_change_fahrenheit = 73.4 := by
  sorry

end NUMINAMATH_CALUDE_temperature_change_over_700_years_l3889_388981


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3889_388994

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft berries_per_plant : ℕ) :
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  berries_per_plant = 8 →
  length * width * plants_per_sqft * berries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l3889_388994


namespace NUMINAMATH_CALUDE_min_sum_squares_l3889_388912

/-- Parabola defined by y² = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Line passing through (4, 0) -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

/-- Intersection points of the line and parabola -/
def Intersection (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧ Line k x₁ y₁ ∧ Line k x₂ y₂ ∧ x₁ ≠ x₂

theorem min_sum_squares :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  Intersection k x₁ y₁ x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3889_388912


namespace NUMINAMATH_CALUDE_no_solution_of_double_composition_l3889_388911

theorem no_solution_of_double_composition
  (P : ℝ → ℝ)
  (h_continuous : Continuous P)
  (h_no_solution : ∀ x : ℝ, P x ≠ x) :
  ∀ x : ℝ, P (P x) ≠ x :=
by sorry

end NUMINAMATH_CALUDE_no_solution_of_double_composition_l3889_388911


namespace NUMINAMATH_CALUDE_perfect_square_with_powers_of_three_l3889_388953

/-- For which integers k (0 ≤ k ≤ 9) do there exist positive integers m and n
    so that 3^m + 3^n + k is a perfect square? -/
theorem perfect_square_with_powers_of_three (k : ℕ) : 
  (k ≤ 9) → 
  (∃ (m n : ℕ+), ∃ (s : ℕ), (3^m.val + 3^n.val + k = s^2)) ↔ 
  (k = 0 ∨ k = 3 ∨ k = 4 ∨ k = 6 ∨ k = 7) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_with_powers_of_three_l3889_388953


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3889_388999

theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3889_388999


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3889_388915

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3889_388915


namespace NUMINAMATH_CALUDE_ultra_high_yield_interest_l3889_388906

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The interest earned from an investment -/
def interest_earned (principal : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem: The interest earned on a 500-dollar investment compounded annually at 3% for 10 years is approximately 172 dollars -/
theorem ultra_high_yield_interest :
  let principal : ℝ := 500
  let rate : ℝ := 0.03
  let years : ℕ := 10
  let final_amount := compound_interest principal rate years
  let earned := interest_earned principal final_amount
  ∃ ε > 0, |earned - 172| < ε :=
by sorry

end NUMINAMATH_CALUDE_ultra_high_yield_interest_l3889_388906


namespace NUMINAMATH_CALUDE_number_of_arrangements_l3889_388919

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of exhibition areas that should have one person --/
def k : ℕ := 2

/-- The number of exhibition areas that should have two people --/
def l : ℕ := 2

/-- The constraint that two specific volunteers cannot be in the same group --/
def constraint : Prop := True

/-- The function that calculates the number of arrangement plans --/
def arrangement_plans (n m k l : ℕ) (constraint : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangement plans is 156 --/
theorem number_of_arrangements :
  arrangement_plans n m k l constraint = 156 := by sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l3889_388919


namespace NUMINAMATH_CALUDE_odd_integer_quadratic_function_property_l3889_388988

theorem odd_integer_quadratic_function_property (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (Nat.gcd a n = 1) ∧ (Nat.gcd b n = 1) ∧
  (∃ (k : ℕ), n * k = (a^2 + b)) ∧
  (∀ (x : ℕ), x ≥ 1 → ∃ (p : ℕ), Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_quadratic_function_property_l3889_388988


namespace NUMINAMATH_CALUDE_apple_percentage_is_fifty_percent_l3889_388939

-- Define the initial number of apples and oranges
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5

-- Define the number of oranges added
def added_oranges : ℕ := 5

-- Define the total number of fruits after adding oranges
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

-- Define the percentage of apples
def apple_percentage : ℚ := (initial_apples : ℚ) / (total_fruits : ℚ) * 100

-- Theorem statement
theorem apple_percentage_is_fifty_percent :
  apple_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_apple_percentage_is_fifty_percent_l3889_388939


namespace NUMINAMATH_CALUDE_diamond_seven_three_l3889_388957

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_seven_three : diamond 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_diamond_seven_three_l3889_388957


namespace NUMINAMATH_CALUDE_value_of_expression_l3889_388955

theorem value_of_expression (m n : ℤ) (h : m - n = -2) : 2 - 5*m + 5*n = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3889_388955


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l3889_388986

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = -1) 
  (h2 : y = 1) 
  (h3 : ρ > 0) 
  (h4 : 0 < θ ∧ θ < π) 
  (h5 : ρ = Real.sqrt (x^2 + y^2)) 
  (h6 : θ = Real.arctan (y / x) + π) : 
  (ρ, θ) = (Real.sqrt 2, 3 * π / 4) := by
sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l3889_388986


namespace NUMINAMATH_CALUDE_ali_fish_weight_l3889_388996

/-- Proves that Ali caught 12 kg of fish given the conditions of the fishing problem -/
theorem ali_fish_weight (peter_weight : ℝ) 
  (h1 : peter_weight + 2 * peter_weight + (peter_weight + 1) = 25) : 
  2 * peter_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_ali_fish_weight_l3889_388996


namespace NUMINAMATH_CALUDE_inequality_proof_l3889_388987

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.exp b < b * Real.exp a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3889_388987


namespace NUMINAMATH_CALUDE_first_class_size_l3889_388969

/-- The number of students in the first class -/
def x : ℕ := sorry

/-- The average mark of the first class -/
def avg_first : ℝ := 40

/-- The number of students in the second class -/
def n_second : ℕ := 30

/-- The average mark of the second class -/
def avg_second : ℝ := 60

/-- The average mark of all students combined -/
def avg_total : ℝ := 50.90909090909091

theorem first_class_size :
  (x * avg_first + n_second * avg_second) / (x + n_second) = avg_total ∧ 
  x = 25 := by sorry

end NUMINAMATH_CALUDE_first_class_size_l3889_388969


namespace NUMINAMATH_CALUDE_train_crossing_time_l3889_388902

/-- Time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (fast_speed : ℝ) : 
  train_length = 100 →
  fast_speed = 24 →
  (50 : ℝ) / 9 = (2 * train_length) / (fast_speed + fast_speed / 2) := by
  sorry

#eval (50 : ℚ) / 9

end NUMINAMATH_CALUDE_train_crossing_time_l3889_388902


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3889_388961

theorem cubic_factorization_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3889_388961


namespace NUMINAMATH_CALUDE_election_results_l3889_388967

theorem election_results (total_votes : ℕ) (invalid_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (candidate_B_percentage : ℚ) :
  total_votes = 1250000 →
  invalid_percentage = 1/5 →
  candidate_A_percentage = 9/20 →
  candidate_B_percentage = 7/20 →
  ∃ (valid_votes : ℕ) (votes_A votes_B votes_C : ℕ),
    valid_votes = total_votes * (1 - invalid_percentage) ∧
    votes_A = valid_votes * candidate_A_percentage ∧
    votes_B = valid_votes * candidate_B_percentage ∧
    votes_C = valid_votes * (1 - candidate_A_percentage - candidate_B_percentage) ∧
    votes_A = 450000 ∧
    votes_B = 350000 ∧
    votes_C = 200000 :=
by sorry

end NUMINAMATH_CALUDE_election_results_l3889_388967


namespace NUMINAMATH_CALUDE_product_of_distinct_prime_factors_of_B_l3889_388908

def divisors_of_60 : List ℕ := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

def B : ℕ := (List.prod divisors_of_60)

theorem product_of_distinct_prime_factors_of_B :
  (Finset.prod (Finset.filter Nat.Prime (Finset.range (B + 1))) id) = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_prime_factors_of_B_l3889_388908


namespace NUMINAMATH_CALUDE_geometry_book_pages_multiple_l3889_388966

/-- Given that:
    - The old edition of a Geometry book has 340 pages
    - The new edition has 450 pages
    - The new edition has 230 pages less than m times the old edition's pages
    Prove that m = 2 -/
theorem geometry_book_pages_multiple (old_pages new_pages less_pages : ℕ) 
    (h1 : old_pages = 340)
    (h2 : new_pages = 450)
    (h3 : less_pages = 230) :
    ∃ m : ℚ, old_pages * m - less_pages = new_pages ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometry_book_pages_multiple_l3889_388966


namespace NUMINAMATH_CALUDE_steven_has_14_peaches_l3889_388963

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jake : ℕ
  jill : ℕ

/-- Given conditions about peach counts -/
def peach_conditions (p : PeachCount) : Prop :=
  p.jake + 6 = p.steven ∧ 
  p.jake = p.jill + 3 ∧ 
  p.jill = 5

/-- Theorem stating Steven has 14 peaches -/
theorem steven_has_14_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.steven = 14 := by
  sorry

end NUMINAMATH_CALUDE_steven_has_14_peaches_l3889_388963


namespace NUMINAMATH_CALUDE_polynomial_product_simplification_l3889_388968

theorem polynomial_product_simplification (x y : ℝ) :
  (3 * x^4 - 2 * y^3) * (9 * x^8 + 6 * x^4 * y^3 + 4 * y^6) = 27 * x^12 - 8 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_simplification_l3889_388968


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3889_388962

/-- 
Given an initial value of 550 and an increase of 35%,
prove that the final value is 742.5.
-/
theorem increase_by_percentage (initial_value : ℝ) (percentage_increase : ℝ) :
  initial_value = 550 →
  percentage_increase = 35 →
  initial_value * (1 + percentage_increase / 100) = 742.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3889_388962
