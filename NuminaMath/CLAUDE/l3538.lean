import Mathlib

namespace NUMINAMATH_CALUDE_job_completion_time_l3538_353885

/-- Given two people A and B who can complete a job individually in 9 and 18 days respectively,
    this theorem proves that they can complete the job together in 6 days. -/
theorem job_completion_time (a_time b_time combined_time : ℚ) 
  (ha : a_time = 9)
  (hb : b_time = 18)
  (hc : combined_time = 6)
  (h_combined : (1 / a_time + 1 / b_time)⁻¹ = combined_time) : 
  combined_time = 6 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3538_353885


namespace NUMINAMATH_CALUDE_nested_radical_floor_l3538_353889

theorem nested_radical_floor (y : ℝ) (B : ℤ) : 
  y > 0 → y^2 = 10 + y → B = ⌊10 + y⌋ → B = 13 := by sorry

end NUMINAMATH_CALUDE_nested_radical_floor_l3538_353889


namespace NUMINAMATH_CALUDE_expression_evaluation_l3538_353807

theorem expression_evaluation (x y : ℝ) (h : x > y ∧ y > 0) :
  (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x/y)^(y-x) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3538_353807


namespace NUMINAMATH_CALUDE_hash_composition_20_l3538_353891

-- Define the # operation
def hash (N : ℝ) : ℝ := (0.5 * N)^2 + 1

-- State the theorem
theorem hash_composition_20 : hash (hash (hash 20)) = 1627102.64 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_20_l3538_353891


namespace NUMINAMATH_CALUDE_total_eggs_count_l3538_353858

/-- The number of Easter eggs found at the club house -/
def club_house_eggs : ℕ := 60

/-- The number of Easter eggs found at the park -/
def park_eggs : ℕ := 40

/-- The number of Easter eggs found at the town hall -/
def town_hall_eggs : ℕ := 30

/-- The number of Easter eggs found at the local library -/
def local_library_eggs : ℕ := 50

/-- The number of Easter eggs found at the community center -/
def community_center_eggs : ℕ := 35

/-- The total number of Easter eggs found that day -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs + local_library_eggs + community_center_eggs

theorem total_eggs_count : total_eggs = 215 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_count_l3538_353858


namespace NUMINAMATH_CALUDE_cube_skeleton_theorem_l3538_353868

/-- The number of small cubes forming the skeleton of an n x n x n cube -/
def skeleton_cubes (n : ℕ) : ℕ := 12 * n - 16

/-- The number of small cubes to be removed to obtain the skeleton of an n x n x n cube -/
def removed_cubes (n : ℕ) : ℕ := n^3 - skeleton_cubes n

theorem cube_skeleton_theorem (n : ℕ) (h : n > 2) :
  skeleton_cubes n = 12 * n - 16 ∧
  removed_cubes n = n^3 - (12 * n - 16) := by
  sorry

#eval skeleton_cubes 6  -- Expected: 56
#eval removed_cubes 7   -- Expected: 275

end NUMINAMATH_CALUDE_cube_skeleton_theorem_l3538_353868


namespace NUMINAMATH_CALUDE_multiply_72_28_l3538_353866

theorem multiply_72_28 : 72 * 28 = 4896 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72_28_l3538_353866


namespace NUMINAMATH_CALUDE_mersenne_prime_implies_exponent_prime_l3538_353841

theorem mersenne_prime_implies_exponent_prime (n : ℕ) : 
  Prime (2^n - 1) → Prime n := by
  sorry

end NUMINAMATH_CALUDE_mersenne_prime_implies_exponent_prime_l3538_353841


namespace NUMINAMATH_CALUDE_six_objects_three_parts_l3538_353834

/-- The number of ways to partition n indistinguishable objects into at most k non-empty parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to partition 6 indistinguishable objects into at most 3 non-empty parts -/
theorem six_objects_three_parts : partition_count 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_objects_three_parts_l3538_353834


namespace NUMINAMATH_CALUDE_harmonic_geometric_sequence_ratio_l3538_353873

theorem harmonic_geometric_sequence_ratio (x y z : ℝ) :
  (1 / y - 1 / x) / (1 / x - 1 / z) = 1 →  -- harmonic sequence condition
  (5 * y * z) / (3 * x * y) = (7 * z * x) / (5 * y * z) →  -- geometric sequence condition
  y / z + z / y = 58 / 21 := by sorry

end NUMINAMATH_CALUDE_harmonic_geometric_sequence_ratio_l3538_353873


namespace NUMINAMATH_CALUDE_exists_remarkable_polygon_for_n_gt_4_l3538_353800

/-- A remarkable polygon is a grid polygon that is not a rectangle and can form a similar polygon from several of its copies. -/
structure RemarkablePolygon (n : ℕ) where
  cells : ℕ
  not_rectangle : cells ≠ 4
  can_form_similar : True  -- Simplified condition for similarity

/-- For all integers n > 4, there exists a remarkable polygon with n cells. -/
theorem exists_remarkable_polygon_for_n_gt_4 (n : ℕ) (h : n > 4) :
  ∃ (P : RemarkablePolygon n), P.cells = n :=
sorry

end NUMINAMATH_CALUDE_exists_remarkable_polygon_for_n_gt_4_l3538_353800


namespace NUMINAMATH_CALUDE_money_left_proof_l3538_353863

def salary : ℚ := 150000.00000000003

def food_fraction : ℚ := 1 / 5
def rent_fraction : ℚ := 1 / 10
def clothes_fraction : ℚ := 3 / 5

def money_left : ℚ := salary - (salary * food_fraction + salary * rent_fraction + salary * clothes_fraction)

theorem money_left_proof : money_left = 15000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_proof_l3538_353863


namespace NUMINAMATH_CALUDE_expand_product_l3538_353818

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3538_353818


namespace NUMINAMATH_CALUDE_f_2014_equals_zero_l3538_353854

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The given property of f
def f_property (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 4) = f x + f 2

-- Theorem statement
theorem f_2014_equals_zero 
  (h_even : even_function f) 
  (h_prop : f_property f) : 
  f 2014 = 0 := by sorry

end NUMINAMATH_CALUDE_f_2014_equals_zero_l3538_353854


namespace NUMINAMATH_CALUDE_f_properties_l3538_353874

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (x + 2) * Real.exp (-x) - 2
  else (x - 2) * Real.exp x + 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x ≤ 0, f x = (x + 2) * Real.exp (-x) - 2) →
  (∀ x > 0, f x = (x - 2) * Real.exp x + 2) ∧
  (∀ m : ℝ, (∃ x ∈ Set.Icc 0 2, f x = m) ↔ m ∈ Set.Icc (2 - Real.exp 1) 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3538_353874


namespace NUMINAMATH_CALUDE_three_digit_append_divisibility_l3538_353876

theorem three_digit_append_divisibility :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (594000 + n) % 651 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_three_digit_append_divisibility_l3538_353876


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3538_353843

theorem sum_of_x_and_y (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 27) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3538_353843


namespace NUMINAMATH_CALUDE_mode_is_25_l3538_353884

def sales_volumes : List ℕ := [10, 14, 25, 13]

def is_mode (x : ℕ) (list : List ℕ) : Prop :=
  ∀ y ∈ list, (list.count x ≥ list.count y)

theorem mode_is_25 (s : ℕ) : is_mode 25 (sales_volumes ++ [s]) := by
  sorry

end NUMINAMATH_CALUDE_mode_is_25_l3538_353884


namespace NUMINAMATH_CALUDE_division_calculation_l3538_353832

theorem division_calculation : (-1/30) / (2/3 - 1/10 + 1/6 - 2/5) = -1/10 := by
  sorry

end NUMINAMATH_CALUDE_division_calculation_l3538_353832


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l3538_353826

theorem distance_between_complex_points :
  let z₁ : ℂ := -3 + I
  let z₂ : ℂ := 1 - I
  Complex.abs (z₂ - z₁) = Real.sqrt 20 := by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l3538_353826


namespace NUMINAMATH_CALUDE_own_square_and_cube_root_l3538_353835

theorem own_square_and_cube_root : 
  ∀ x : ℝ, (x^2 = x ∧ x^3 = x) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_own_square_and_cube_root_l3538_353835


namespace NUMINAMATH_CALUDE_derivative_ln_inverse_sqrt_plus_one_squared_l3538_353851

open Real

theorem derivative_ln_inverse_sqrt_plus_one_squared (x : ℝ) :
  deriv (λ x => Real.log (1 / Real.sqrt (1 + x^2))) x = -x / (1 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_inverse_sqrt_plus_one_squared_l3538_353851


namespace NUMINAMATH_CALUDE_luxury_to_suv_ratio_l3538_353875

/-- Represents the number of cars of each type -/
structure CarInventory where
  economy : ℕ
  luxury : ℕ
  suv : ℕ

/-- The ratio of economy cars to luxury cars is 3:2 -/
def economy_to_luxury_ratio (inventory : CarInventory) : Prop :=
  3 * inventory.luxury = 2 * inventory.economy

/-- The ratio of economy cars to SUVs is 4:1 -/
def economy_to_suv_ratio (inventory : CarInventory) : Prop :=
  4 * inventory.suv = inventory.economy

/-- The theorem stating the ratio of luxury cars to SUVs -/
theorem luxury_to_suv_ratio (inventory : CarInventory) 
  (h1 : economy_to_luxury_ratio inventory) 
  (h2 : economy_to_suv_ratio inventory) : 
  8 * inventory.suv = 3 * inventory.luxury := by
  sorry

#check luxury_to_suv_ratio

end NUMINAMATH_CALUDE_luxury_to_suv_ratio_l3538_353875


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3538_353821

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 = 12, 
    then a_1 + a_2 + ... + a_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3538_353821


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l3538_353845

/-- Represents the earnings from a day's sale of drinks at a country store. -/
def store_earnings (cola_price : ℚ) (juice_price : ℚ) (water_price : ℚ) (sports_drink_price : ℚ)
                   (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (sports_drink_sold : ℕ)
                   (sports_drink_paid : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold +
  sports_drink_price * sports_drink_paid

/-- Theorem stating the total earnings of the store given the specific conditions. -/
theorem store_earnings_calculation :
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let sports_drink_price : ℚ := 5/2
  let cola_sold : ℕ := 18
  let juice_sold : ℕ := 15
  let water_sold : ℕ := 30
  let sports_drink_sold : ℕ := 44
  let sports_drink_paid : ℕ := 22
  store_earnings cola_price juice_price water_price sports_drink_price
                 cola_sold juice_sold water_sold sports_drink_sold sports_drink_paid = 161.5 := by
  sorry


end NUMINAMATH_CALUDE_store_earnings_calculation_l3538_353845


namespace NUMINAMATH_CALUDE_snow_difference_l3538_353848

def mrs_hilt_snow : ℕ := 29
def brecknock_snow : ℕ := 17

theorem snow_difference : mrs_hilt_snow - brecknock_snow = 12 := by
  sorry

end NUMINAMATH_CALUDE_snow_difference_l3538_353848


namespace NUMINAMATH_CALUDE_fraction_B_is_02_l3538_353838

-- Define the fractions of students receiving grades
def fraction_A : ℝ := 0.7
def fraction_A_or_B : ℝ := 0.9

-- Theorem statement
theorem fraction_B_is_02 : 
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_B_is_02_l3538_353838


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3538_353882

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 210 > 0 ∧ x + x + 210 = 360 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3538_353882


namespace NUMINAMATH_CALUDE_leesburg_population_l3538_353865

theorem leesburg_population (salem_factor : ℕ) (moved_out : ℕ) (women_ratio : ℚ) (women_count : ℕ) :
  salem_factor = 15 →
  moved_out = 130000 →
  women_ratio = 1/2 →
  women_count = 377050 →
  ∃ (leesburg_pop : ℕ), leesburg_pop = 58940 ∧ 
    salem_factor * leesburg_pop = 2 * women_count + moved_out :=
sorry

end NUMINAMATH_CALUDE_leesburg_population_l3538_353865


namespace NUMINAMATH_CALUDE_g_of_3_eq_200_l3538_353849

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_eq_200 : g 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_200_l3538_353849


namespace NUMINAMATH_CALUDE_blue_shirts_count_l3538_353827

/-- Represents the number of people at a school dance --/
structure DanceAttendance where
  boys : ℕ
  girls : ℕ
  teachers : ℕ

/-- Calculates the number of people wearing blue shirts at the dance --/
def blueShirts (attendance : DanceAttendance) : ℕ :=
  let blueShirtedBoys := (attendance.boys * 20) / 100
  let blueShirtedMaleTeachers := (attendance.teachers * 25) / 100
  blueShirtedBoys + blueShirtedMaleTeachers

/-- Theorem stating the number of people wearing blue shirts at the dance --/
theorem blue_shirts_count (attendance : DanceAttendance) :
  attendance.boys * 4 = attendance.girls * 3 →
  attendance.teachers * 9 = attendance.boys + attendance.girls →
  attendance.girls = 108 →
  blueShirts attendance = 21 := by
  sorry

#eval blueShirts { boys := 81, girls := 108, teachers := 21 }

end NUMINAMATH_CALUDE_blue_shirts_count_l3538_353827


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_225parts_l3538_353860

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a division of the board -/
structure Division :=
  (num_parts : ℕ)
  (equal_area : Bool)

/-- Calculates the maximum possible total length of cuts for a given board and division -/
def max_cut_length (b : Board) (d : Division) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_cut_length_30x30_225parts (b : Board) (d : Division) :
  b.size = 30 ∧ d.num_parts = 225 ∧ d.equal_area = true →
  max_cut_length b d = 1065 :=
sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_225parts_l3538_353860


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3538_353840

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem quadratic_function_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, ∃ y ∈ Set.Icc (-4 : ℝ) 12, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-4 : ℝ) 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3538_353840


namespace NUMINAMATH_CALUDE_second_pipe_rate_l3538_353890

def well_capacity : ℝ := 1200
def first_pipe_rate : ℝ := 48
def filling_time : ℝ := 5

theorem second_pipe_rate : 
  ∃ (rate : ℝ), 
    rate * filling_time + first_pipe_rate * filling_time = well_capacity ∧ 
    rate = 192 :=
by sorry

end NUMINAMATH_CALUDE_second_pipe_rate_l3538_353890


namespace NUMINAMATH_CALUDE_square_difference_120_pairs_l3538_353896

theorem square_difference_120_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > n ∧ m^2 - n^2 = 120) ∧
    pairs.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_120_pairs_l3538_353896


namespace NUMINAMATH_CALUDE_solution_to_system_l3538_353804

/-- The system of equations -/
def equation1 (x y : ℝ) : Prop := x^2*y - x*y^2 - 5*x + 5*y + 3 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The theorem stating that (4, 1) is the solution to the system of equations -/
theorem solution_to_system : equation1 4 1 ∧ equation2 4 1 := by sorry

end NUMINAMATH_CALUDE_solution_to_system_l3538_353804


namespace NUMINAMATH_CALUDE_total_marbles_l3538_353867

/-- Given a collection of red, blue, and green marbles, where:
  1. There are 25% more red marbles than blue marbles
  2. There are 60% more green marbles than red marbles
  3. The number of red marbles is r
Prove that the total number of marbles in the collection is 3.4r -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) 
  (h1 : r = 1.25 * b) 
  (h2 : g = 1.6 * r) : 
  r + b + g = 3.4 * r := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l3538_353867


namespace NUMINAMATH_CALUDE_sequence_representation_l3538_353830

theorem sequence_representation (a : ℕ → ℕ) 
  (h_increasing : ∀ k : ℕ, k ≥ 1 → a k < a (k + 1)) :
  ∀ N : ℕ, ∃ m p q x y : ℕ, 
    m > N ∧ 
    p ≠ q ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    a m = x * a p + y * a q :=
sorry

end NUMINAMATH_CALUDE_sequence_representation_l3538_353830


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3538_353871

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3538_353871


namespace NUMINAMATH_CALUDE_sofia_survey_l3538_353895

theorem sofia_survey (liked : ℕ) (disliked : ℕ) (h1 : liked = 235) (h2 : disliked = 165) :
  liked + disliked = 400 := by
  sorry

end NUMINAMATH_CALUDE_sofia_survey_l3538_353895


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l3538_353859

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the domain
def domain (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 5

-- Theorem: The curve is a line segment
theorem curve_is_line_segment :
  ∃ (a b : ℝ), ∀ (t : ℝ), domain t →
    ∃ (k : ℝ), 0 ≤ k ∧ k ≤ 1 ∧
      x t = a + k * (b - a) ∧
      y t = (x t - 2) / 3 ∧
      -1 ≤ y t ∧ y t ≤ 24 :=
by sorry


end NUMINAMATH_CALUDE_curve_is_line_segment_l3538_353859


namespace NUMINAMATH_CALUDE_stratified_sampling_eleventh_grade_l3538_353828

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def student_ratio : Fin 3 → ℕ
  | 0 => 4  -- 10th grade
  | 1 => 3  -- 11th grade
  | 2 => 3  -- 12th grade

/-- Total number of parts in the ratio -/
def total_ratio : ℕ := (student_ratio 0) + (student_ratio 1) + (student_ratio 2)

/-- Total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students drawn from the 11th grade -/
def eleventh_grade_sample : ℕ := 
  (student_ratio 1 * sample_size) / total_ratio

theorem stratified_sampling_eleventh_grade :
  eleventh_grade_sample = 15 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_eleventh_grade_l3538_353828


namespace NUMINAMATH_CALUDE_prob_rain_holiday_l3538_353869

/-- The probability of rain on Friday without a storm -/
def prob_rain_friday : ℝ := 0.3

/-- The probability of rain on Monday without a storm -/
def prob_rain_monday : ℝ := 0.6

/-- The increase in probability of rain if a storm develops -/
def storm_increase : ℝ := 0.2

/-- The probability of a storm developing -/
def prob_storm : ℝ := 0.5

/-- Assumption that all probabilities are independent -/
axiom probabilities_independent : True

/-- The probability of rain on at least one day during the holiday -/
def prob_rain_at_least_one_day : ℝ := 
  1 - (prob_storm * (1 - (prob_rain_friday + storm_increase)) * (1 - (prob_rain_monday + storm_increase)) + 
       (1 - prob_storm) * (1 - prob_rain_friday) * (1 - prob_rain_monday))

theorem prob_rain_holiday : prob_rain_at_least_one_day = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_holiday_l3538_353869


namespace NUMINAMATH_CALUDE_function_inequality_condition_l3538_353815

theorem function_inequality_condition (f : ℝ → ℝ) (a c : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  c > 0 →
  (∀ x, |x + 5| < c → |f x + 5| < a) ↔
  c > a / 2 := by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l3538_353815


namespace NUMINAMATH_CALUDE_f_difference_l3538_353813

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

/-- Theorem stating that f(3) - f(-3) = 210 -/
theorem f_difference : f 3 - f (-3) = 210 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3538_353813


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3538_353850

def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) ∧ y > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, m > 0 → (∀ b ∈ B, m ∣ b) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3538_353850


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3538_353808

def U : Set Int := {-2, 0, 1, 2}

def A : Set Int := {x ∈ U | x^2 + x - 2 = 0}

theorem complement_of_A_in_U :
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3538_353808


namespace NUMINAMATH_CALUDE_max_value_theorem_l3538_353812

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 4 + 9 * y * z ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3538_353812


namespace NUMINAMATH_CALUDE_f_range_l3538_353877

def closest_multiple (k : ℤ) (n : ℤ) : ℤ :=
  n * round (k / n)

def f (k : ℤ) : ℤ :=
  closest_multiple k 3 + closest_multiple (2*k) 5 + closest_multiple (3*k) 7 - 6*k

theorem f_range :
  (∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6) ∧
  (∀ m : ℤ, -6 ≤ m ∧ m ≤ 6 → ∃ k : ℤ, f k = m) :=
sorry

end NUMINAMATH_CALUDE_f_range_l3538_353877


namespace NUMINAMATH_CALUDE_stating_investment_plans_count_l3538_353842

/-- Represents the number of cities available for investment --/
def num_cities : ℕ := 4

/-- Represents the number of projects to be distributed --/
def num_projects : ℕ := 3

/-- Represents the maximum number of projects allowed in a single city --/
def max_projects_per_city : ℕ := 2

/-- 
Calculates the number of ways to distribute distinct projects among cities,
with a limit on the number of projects per city.
--/
def investment_plans (cities : ℕ) (projects : ℕ) (max_per_city : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of investment plans 
for the given conditions is 60.
--/
theorem investment_plans_count : 
  investment_plans num_cities num_projects max_projects_per_city = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_investment_plans_count_l3538_353842


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l3538_353825

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny adds more -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l3538_353825


namespace NUMINAMATH_CALUDE_acute_and_less_than_90_subset_l3538_353892

-- Define the sets
def A : Set ℝ := {x | ∃ k : ℤ, k * 360 < x ∧ x < k * 360 + 90}
def B : Set ℝ := {x | 0 < x ∧ x < 90}
def C : Set ℝ := {x | x < 90}

-- Theorem statement
theorem acute_and_less_than_90_subset :
  B ∪ C ⊆ C := by sorry

end NUMINAMATH_CALUDE_acute_and_less_than_90_subset_l3538_353892


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l3538_353897

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (2 * x₁) + 4 = 38) ∧ 
   (abs (2 * x₂) + 4 = 38) ∧ 
   x₁ * x₂ = -289) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l3538_353897


namespace NUMINAMATH_CALUDE_cookies_problem_l3538_353814

theorem cookies_problem (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine < mona →
  rachel = jasmine + 10 →
  mona + jasmine + rachel = 60 →
  jasmine = 15 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l3538_353814


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l3538_353886

theorem correct_fraction_proof (x y : ℕ) (h : x > 0 ∧ y > 0) :
  (5 : ℚ) / 6 * 480 = x / y * 480 + 250 → x / y = (5 : ℚ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l3538_353886


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3538_353839

theorem polynomial_divisibility (a b : ℝ) : 
  (∀ (X : ℝ), (X - 1)^2 ∣ (a * X^4 + b * X^3 + 1)) ↔ 
  (a = 3 ∧ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3538_353839


namespace NUMINAMATH_CALUDE_units_digit_of_composite_product_l3538_353855

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_composite_product :
  (first_four_composites.prod % 10) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_composite_product_l3538_353855


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_a_value_l3538_353852

/-- Given two curves C₁ and C₂, where C₁ is defined by y = ax³ - 6x² + 12x and C₂ is defined by y = e^x,
    if their tangent lines at x = 1 are parallel, then a = e/3. -/
theorem parallel_tangents_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (3 * a * x^2 - 12 * x + 12) = Real.exp x) → a = Real.exp 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_a_value_l3538_353852


namespace NUMINAMATH_CALUDE_marks_car_repair_cost_l3538_353837

/-- The total cost of fixing Mark's car -/
def total_cost (part_cost : ℕ) (num_parts : ℕ) (labor_rate : ℚ) (hours_worked : ℕ) : ℚ :=
  (part_cost * num_parts : ℚ) + labor_rate * (hours_worked * 60)

/-- Theorem stating that the total cost of fixing Mark's car is $220 -/
theorem marks_car_repair_cost :
  total_cost 20 2 0.5 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_marks_car_repair_cost_l3538_353837


namespace NUMINAMATH_CALUDE_car_uphill_speed_l3538_353809

/-- Proves that the uphill speed of a car is 30 km/hr given the specified conditions -/
theorem car_uphill_speed (V_up : ℝ) : 
  V_up > 0 →
  (100 / V_up + 50 / 60 : ℝ) = 150 / 36 →
  V_up = 30 := by
sorry

end NUMINAMATH_CALUDE_car_uphill_speed_l3538_353809


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l3538_353824

/-- Calculates the share of profit for an investor based on their investment amount, duration, and total profit -/
def calculate_share_of_profit (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 27000 →
  calculate_share_of_profit jose_investment jose_duration (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 15000 := by
  sorry

#eval calculate_share_of_profit 45000 10 810000 27000

end NUMINAMATH_CALUDE_jose_share_of_profit_l3538_353824


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l3538_353820

/-- A single-elimination tournament with no ties -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games required to declare a winner in a tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games required to declare a winner is 23 -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 24) (h2 : t.no_ties = true) : 
  games_to_winner t = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l3538_353820


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3538_353811

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3538_353811


namespace NUMINAMATH_CALUDE_smallest_number_l3538_353881

/-- Converts a number from base k to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary number 111111₍₂₎ --/
def binary_num : List Nat := [1, 1, 1, 1, 1, 1]

/-- The base-6 number 150₍₆₎ --/
def base6_num : List Nat := [0, 5, 1]

/-- The base-4 number 1000₍₄₎ --/
def base4_num : List Nat := [0, 0, 0, 1]

/-- The octal number 101₍₈₎ --/
def octal_num : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_num 2 < to_decimal base6_num 6 ∧
  to_decimal binary_num 2 < to_decimal base4_num 4 ∧
  to_decimal binary_num 2 < to_decimal octal_num 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3538_353881


namespace NUMINAMATH_CALUDE_fraction_value_l3538_353805

theorem fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : m * n / (m - n) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3538_353805


namespace NUMINAMATH_CALUDE_bakery_ratio_l3538_353878

/-- Given the conditions of a bakery's storage room, prove the ratio of flour to baking soda --/
theorem bakery_ratio (sugar flour baking_soda : ℕ) : 
  sugar = 6000 ∧ 
  5 * flour = 2 * sugar ∧ 
  8 * (baking_soda + 60) = flour → 
  10 * baking_soda = flour := by sorry

end NUMINAMATH_CALUDE_bakery_ratio_l3538_353878


namespace NUMINAMATH_CALUDE_frequency_table_purpose_l3538_353817

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  /-- The table analyzes sample data -/
  analyzes_sample_data : Bool
  /-- The table groups data into categories -/
  groups_data : Bool

/-- The purpose of creating a frequency distribution table -/
def purpose_of_frequency_table (table : FrequencyDistributionTable) : Prop :=
  table.analyzes_sample_data ∧ 
  table.groups_data → 
  (∃ (proportion_understanding : Prop) (population_estimation : Prop),
    proportion_understanding ∧ population_estimation)

/-- Theorem stating the purpose of creating a frequency distribution table -/
theorem frequency_table_purpose (table : FrequencyDistributionTable) : 
  purpose_of_frequency_table table :=
sorry

end NUMINAMATH_CALUDE_frequency_table_purpose_l3538_353817


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l3538_353819

theorem chess_game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 2/5)
  (h_not_lose : p_not_lose = 9/10) :
  p_not_lose - p_win = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l3538_353819


namespace NUMINAMATH_CALUDE_triple_overlap_is_six_l3538_353806

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the auditorium and the placement of carpets -/
structure Auditorium where
  width : ℝ
  height : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap given an auditorium setup -/
def tripleOverlapArea (a : Auditorium) : ℝ :=
  2 * 3

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (a : Auditorium) 
    (h1 : a.width = 10 ∧ a.height = 10)
    (h2 : a.carpet1.width = 6 ∧ a.carpet1.height = 8)
    (h3 : a.carpet2.width = 6 ∧ a.carpet2.height = 6)
    (h4 : a.carpet3.width = 5 ∧ a.carpet3.height = 7) :
  tripleOverlapArea a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_is_six_l3538_353806


namespace NUMINAMATH_CALUDE_remainder_of_p_l3538_353862

-- Define the polynomial p(x)
def p (x : ℝ) (r : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (x + 1) * (x - 2)^2 * r x + a * x + b

-- State the theorem
theorem remainder_of_p (r : ℝ → ℝ) (a b : ℝ) :
  (p 2 r a b = 6) →
  (p (-1) r a b = 0) →
  ∃ q : ℝ → ℝ, ∀ x, p x r a b = (x + 1) * (x - 2)^2 * q x + 2 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_p_l3538_353862


namespace NUMINAMATH_CALUDE_tennis_tournament_l3538_353870

theorem tennis_tournament (n : ℕ) : n > 0 → (
  let total_players := 4 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * n * (3 * n)
  let men_wins := 3 * n * n
  women_wins + men_wins = total_matches ∧
  3 * men_wins = 2 * women_wins
) → n = 4 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_l3538_353870


namespace NUMINAMATH_CALUDE_strawberry_cost_l3538_353831

/-- The price of one basket of strawberries in dollars -/
def price_per_basket : ℚ := 16.5

/-- The number of baskets to be purchased -/
def number_of_baskets : ℕ := 4

/-- The total cost of purchasing the strawberries -/
def total_cost : ℚ := price_per_basket * number_of_baskets

/-- Theorem stating that the total cost of 4 baskets of strawberries at $16.50 each is $66.00 -/
theorem strawberry_cost : total_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cost_l3538_353831


namespace NUMINAMATH_CALUDE_abcd_addition_l3538_353801

theorem abcd_addition (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * b + 10 * c + d) = 5472 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_abcd_addition_l3538_353801


namespace NUMINAMATH_CALUDE_prime_sum_2003_l3538_353894

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (heq : a^2 + b = 2003) :
  a + b = 2001 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2003_l3538_353894


namespace NUMINAMATH_CALUDE_revenue_is_78_l3538_353893

/-- The revenue per t-shirt for a shop selling t-shirts during two games -/
def revenue_per_tshirt (total_tshirts : ℕ) (first_game_tshirts : ℕ) (second_game_revenue : ℕ) : ℚ :=
  second_game_revenue / (total_tshirts - first_game_tshirts)

/-- Theorem stating that the revenue per t-shirt is $78 given the specified conditions -/
theorem revenue_is_78 :
  revenue_per_tshirt 186 172 1092 = 78 := by
  sorry

#eval revenue_per_tshirt 186 172 1092

end NUMINAMATH_CALUDE_revenue_is_78_l3538_353893


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l3538_353872

/-- A line through (1, -8) and (k, 15) is parallel to 6x + 9y = -12 iff k = -33.5 -/
theorem parallel_line_k_value : ∀ k : ℝ,
  (∃ m b : ℝ, (∀ x y : ℝ, y = m*x + b ↔ (x = 1 ∧ y = -8) ∨ (x = k ∧ y = 15)) ∧
               m = -2/3) ↔
  k = -33.5 := by sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l3538_353872


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3538_353823

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a^6*b + a^6*c + b^6*a + b^6*c + c^6*a + c^6*b) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3538_353823


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l3538_353822

theorem scooter_gain_percent : 
  let initial_cost : ℚ := 900
  let repair1 : ℚ := 150
  let repair2 : ℚ := 75
  let repair3 : ℚ := 225
  let selling_price : ℚ := 1800
  let total_cost : ℚ := initial_cost + repair1 + repair2 + repair3
  let gain : ℚ := selling_price - total_cost
  let gain_percent : ℚ := (gain / total_cost) * 100
  gain_percent = 33.33 := by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l3538_353822


namespace NUMINAMATH_CALUDE_mystery_discount_rate_l3538_353810

/-- Represents the discount rate for books -/
structure DiscountRate :=
  (biography : ℝ)
  (mystery : ℝ)

/-- Represents the problem parameters -/
structure BookstoreParams :=
  (biography_price : ℝ)
  (mystery_price : ℝ)
  (biography_count : ℕ)
  (mystery_count : ℕ)
  (total_savings : ℝ)
  (total_discount_rate : ℝ)

/-- Theorem stating that given the problem conditions, the discount rate on mysteries is 37.5% -/
theorem mystery_discount_rate 
  (params : BookstoreParams)
  (h1 : params.biography_price = 20)
  (h2 : params.mystery_price = 12)
  (h3 : params.biography_count = 5)
  (h4 : params.mystery_count = 3)
  (h5 : params.total_savings = 19)
  (h6 : params.total_discount_rate = 43)
  : ∃ (d : DiscountRate), 
    d.biography + d.mystery = params.total_discount_rate ∧ 
    params.biography_count * params.biography_price * (d.biography / 100) + 
    params.mystery_count * params.mystery_price * (d.mystery / 100) = params.total_savings ∧
    d.mystery = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_mystery_discount_rate_l3538_353810


namespace NUMINAMATH_CALUDE_unique_start_day_l3538_353879

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- A function that determines if a given day is the first day of a 30-day month with equal Saturdays and Sundays -/
def is_valid_start_day (d : DayOfWeek) : Prop :=
  ∃ (sat_count sun_count : ℕ),
    sat_count = sun_count ∧
    sat_count + sun_count ≤ 30 ∧
    (match d with
      | DayOfWeek.Monday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Tuesday   => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Wednesday => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Thursday  => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Friday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Saturday  => sat_count = 5 ∧ sun_count = 5
      | DayOfWeek.Sunday    => sat_count = 5 ∧ sun_count = 4)

/-- Theorem stating that there is exactly one day of the week that can be the first day of a 30-day month with equal Saturdays and Sundays -/
theorem unique_start_day :
  ∃! (d : DayOfWeek), is_valid_start_day d :=
sorry

end NUMINAMATH_CALUDE_unique_start_day_l3538_353879


namespace NUMINAMATH_CALUDE_train_length_l3538_353847

/-- The length of a train that passes a stationary man in 8 seconds and crosses a 270-meter platform in 20 seconds is 180 meters. -/
theorem train_length : ℝ → Prop :=
  fun L : ℝ =>
    (L / 8 = (L + 270) / 20) →
    L = 180

/-- Proof of the train length theorem -/
lemma train_length_proof : ∃ L : ℝ, train_length L :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3538_353847


namespace NUMINAMATH_CALUDE_audiobook_listening_time_l3538_353861

theorem audiobook_listening_time 
  (num_books : ℕ) 
  (book_length : ℕ) 
  (daily_listening : ℕ) 
  (h1 : num_books = 6) 
  (h2 : book_length = 30) 
  (h3 : daily_listening = 2) : 
  (num_books * book_length) / daily_listening = 90 := by
sorry

end NUMINAMATH_CALUDE_audiobook_listening_time_l3538_353861


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3538_353844

/-- Proves that a rectangle with length double its width, when modified as described, has an original length of 40. -/
theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  (2*w - 5) * (w + 5) = 2*w*w + 75 → 2*w = 40 := by
  sorry

#check rectangle_length_proof

end NUMINAMATH_CALUDE_rectangle_length_proof_l3538_353844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3538_353846

/-- For an arithmetic sequence with first term a₁ and common difference d,
    if the sum of the first 10 terms is 4 times the sum of the first 5 terms,
    then a₁/d = 1/2. -/
theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  let S : ℕ → ℝ := λ n => n * a₁ + (n * (n - 1) / 2) * d
  S 10 = 4 * S 5 → a₁ / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3538_353846


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3538_353898

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 140000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.4
    exponent := 8
    coeff_range := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3538_353898


namespace NUMINAMATH_CALUDE_hyperbola_center_l3538_353802

/-- The hyperbola is defined by the equation 9x^2 - 54x - 16y^2 + 128y - 400 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola is the point (h, k) where h and k are the coordinates that make
    the equation symmetric about the vertical and horizontal axes passing through (h, k) -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y, hyperbola_equation x y ↔ hyperbola_equation (2*h - x) y ∧ hyperbola_equation x (2*k - y)

/-- The center of the hyperbola defined by 9x^2 - 54x - 16y^2 + 128y - 400 = 0 is (3, 4) -/
theorem hyperbola_center : is_center 3 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3538_353802


namespace NUMINAMATH_CALUDE_office_network_connections_l3538_353857

/-- A network of switches with connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- The theorem stating that a network of 40 switches, each connected to 4 others, has 80 connections. -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 40, connections_per_switch := 4 }
  total_connections network = 80 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l3538_353857


namespace NUMINAMATH_CALUDE_magic_square_sum_l3538_353833

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : Fin 3 → Fin 3 → ℕ
  sum_row : ∀ i, (Finset.univ.sum (λ j => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_col : ∀ j, (Finset.univ.sum (λ i => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag1 : (Finset.univ.sum (λ i => a i i)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag2 : (Finset.univ.sum (λ i => a i (2 - i))) = (Finset.univ.sum (λ j => a 0 j))

/-- The theorem to be proved -/
theorem magic_square_sum (s : MagicSquare) 
  (h1 : s.a 0 0 = 25)
  (h2 : s.a 0 2 = 23)
  (h3 : s.a 1 0 = 18)
  (h4 : s.a 2 1 = 22) :
  s.a 1 2 + s.a 0 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l3538_353833


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l3538_353899

theorem hippopotamus_crayons (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 62)
  (h2 : remaining_crayons = 10) :
  initial_crayons - remaining_crayons = 52 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l3538_353899


namespace NUMINAMATH_CALUDE_range_of_a_l3538_353864

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : (∃ x, x ∈ A ∩ B a) → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3538_353864


namespace NUMINAMATH_CALUDE_jersey_cost_l3538_353856

theorem jersey_cost (initial_amount : ℕ) (num_jerseys : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 ∧
  num_jerseys = 5 ∧
  basketball_cost = 18 ∧
  shorts_cost = 8 ∧
  remaining_amount = 14 →
  ∃ (jersey_cost : ℕ), jersey_cost = 2 ∧ initial_amount = num_jerseys * jersey_cost + basketball_cost + shorts_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_jersey_cost_l3538_353856


namespace NUMINAMATH_CALUDE_domain_of_f_half_x_l3538_353880

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(lg x)
def domain_f_lg_x : Set ℝ := { x | 0.1 ≤ x ∧ x ≤ 100 }

-- State the theorem
theorem domain_of_f_half_x (h : ∀ x ∈ domain_f_lg_x, f (Real.log x / Real.log 10) = f (Real.log x / Real.log 10)) :
  { x : ℝ | f (x / 2) = f (x / 2) } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_half_x_l3538_353880


namespace NUMINAMATH_CALUDE_transformation_interval_l3538_353816

theorem transformation_interval (x : ℝ) :
  x ∈ Set.Icc 0 1 → (8 * x - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end NUMINAMATH_CALUDE_transformation_interval_l3538_353816


namespace NUMINAMATH_CALUDE_f_greater_than_log_over_x_minus_one_l3538_353887

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1) + 1 / x

theorem f_greater_than_log_over_x_minus_one (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  f x > Real.log x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_log_over_x_minus_one_l3538_353887


namespace NUMINAMATH_CALUDE_min_value_a_squared_l3538_353803

/-- In an acute-angled triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if b^2 * sin(C) = 4√2 * sin(B) and the area of triangle ABC is 8/3,
    then the minimum value of a^2 is 16√2/3. -/
theorem min_value_a_squared (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute-angled triangle
  b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B →     -- Given condition
  (1/2) * b * c * Real.sin A = 8/3 →                    -- Area of triangle
  ∀ x, x^2 ≥ a^2 → x^2 ≥ (16 * Real.sqrt 2) / 3 :=      -- Minimum value of a^2
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_l3538_353803


namespace NUMINAMATH_CALUDE_basketball_weight_prove_basketball_weight_l3538_353853

theorem basketball_weight : ℝ → ℝ → ℝ → Prop :=
  fun basketball_weight tricycle_weight motorbike_weight =>
    (9 * basketball_weight = 6 * tricycle_weight) ∧
    (6 * tricycle_weight = 4 * motorbike_weight) ∧
    (2 * motorbike_weight = 144) →
    basketball_weight = 32

-- Proof
theorem prove_basketball_weight :
  ∃ (b t m : ℝ), basketball_weight b t m :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_weight_prove_basketball_weight_l3538_353853


namespace NUMINAMATH_CALUDE_problem_statement_l3538_353883

theorem problem_statement : (-1)^53 + 2^(4^4 + 3^3 - 5^2) = -1 + 2^258 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3538_353883


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l3538_353888

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * (N - 2)) / Nat.factorial (N + 2) = 
  (Nat.factorial N * (N - 2)) / (N + 2) := by
sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l3538_353888


namespace NUMINAMATH_CALUDE_triangle_side_length_l3538_353836

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (b * Real.sin A = 3 * c * Real.sin B) →
  (a = 3) →
  (Real.cos B = 2/3) →
  -- Triangle inequality (to ensure it's a valid triangle)
  (a + b > c) → (b + c > a) → (c + a > b) →
  -- Positive side lengths
  (a > 0) → (b > 0) → (c > 0) →
  -- Conclusion
  b = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3538_353836


namespace NUMINAMATH_CALUDE_sally_cards_total_l3538_353829

/-- The number of cards Sally has now is equal to the sum of her initial cards,
    the cards Dan gave her, and the cards she bought. -/
theorem sally_cards_total
  (initial : ℕ)  -- Sally's initial number of cards
  (from_dan : ℕ) -- Number of cards Dan gave Sally
  (bought : ℕ)   -- Number of cards Sally bought
  (h1 : initial = 27)
  (h2 : from_dan = 41)
  (h3 : bought = 20) :
  initial + from_dan + bought = 88 :=
by sorry

end NUMINAMATH_CALUDE_sally_cards_total_l3538_353829
