import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l659_65953

theorem polynomial_roots_and_factorization (m : ℤ) : 
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → 
    (∃ a b c d : ℤ, x = a ∨ x = b ∨ x = c ∨ x = d)) →
  (m = -10 ∧ 
   ∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l659_65953


namespace NUMINAMATH_CALUDE_class_composition_l659_65986

theorem class_composition (total_students : ℕ) (total_planes : ℕ) (girls_planes : ℕ) (boys_planes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_planes = 69)
  (h3 : girls_planes = 2)
  (h4 : boys_planes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧
    boys * boys_planes + girls * girls_planes = total_planes ∧
    boys = 9 ∧
    girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l659_65986


namespace NUMINAMATH_CALUDE_halloween_candy_proof_l659_65997

/-- The number of candy pieces Faye scored on Halloween -/
def initial_candy : ℕ := 47

/-- The number of candy pieces Faye ate on the first night -/
def eaten_candy : ℕ := 25

/-- The number of candy pieces Faye's sister gave her -/
def gifted_candy : ℕ := 40

/-- The number of candy pieces Faye has now -/
def current_candy : ℕ := 62

/-- Theorem stating that the initial number of candy pieces is correct -/
theorem halloween_candy_proof : 
  initial_candy - eaten_candy + gifted_candy = current_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_proof_l659_65997


namespace NUMINAMATH_CALUDE_stratified_sampling_high_school_l659_65904

theorem stratified_sampling_high_school
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 950)
  (h_freshmen : freshmen = 350)
  (h_sophomores : sophomores = 400)
  (h_sample : sample_size = 190) :
  let juniors := total_students - freshmen - sophomores
  let sample_ratio := sample_size / total_students
  let freshmen_sample := (sample_ratio * freshmen : ℚ).num
  let sophomores_sample := (sample_ratio * sophomores : ℚ).num
  let juniors_sample := (sample_ratio * juniors : ℚ).num
  (freshmen_sample, sophomores_sample, juniors_sample) = (70, 80, 40) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_high_school_l659_65904


namespace NUMINAMATH_CALUDE_total_cost_price_calculation_l659_65961

theorem total_cost_price_calculation (sp1 sp2 sp3 : ℚ) (profit1 loss2 profit3 : ℚ) :
  sp1 = 600 ∧ profit1 = 25/100 ∧
  sp2 = 800 ∧ loss2 = 20/100 ∧
  sp3 = 1000 ∧ profit3 = 30/100 →
  ∃ (cp1 cp2 cp3 : ℚ),
    cp1 = sp1 / (1 + profit1) ∧
    cp2 = sp2 / (1 - loss2) ∧
    cp3 = sp3 / (1 + profit3) ∧
    cp1 + cp2 + cp3 = 2249.23 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_price_calculation_l659_65961


namespace NUMINAMATH_CALUDE_one_third_comparison_l659_65993

theorem one_third_comparison : (1 / 3 : ℚ) - (33333333 / 100000000 : ℚ) = 1 / (3 * 100000000) := by
  sorry

end NUMINAMATH_CALUDE_one_third_comparison_l659_65993


namespace NUMINAMATH_CALUDE_special_triangle_AB_length_l659_65971

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point K on BC -/
  K : ℝ × ℝ
  /-- Point M on AB -/
  M : ℝ × ℝ
  /-- Point N on AC -/
  N : ℝ × ℝ
  /-- AC length is 18 -/
  h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 18
  /-- BC length is 21 -/
  h_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 21
  /-- K is midpoint of BC -/
  h_K_midpoint : K = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  /-- M is midpoint of AB -/
  h_M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- AN length is 6 -/
  h_AN : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 6
  /-- MN = KN -/
  h_MN_eq_KN : Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = Real.sqrt ((N.1 - K.1)^2 + (N.2 - K.2)^2)

/-- The length of AB in the special triangle is 15 -/
theorem special_triangle_AB_length (t : SpecialTriangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_AB_length_l659_65971


namespace NUMINAMATH_CALUDE_average_of_first_25_odd_primes_l659_65906

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def first_25_odd_primes : List ℕ := 
  [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

theorem average_of_first_25_odd_primes : 
  (∀ p ∈ first_25_odd_primes, is_prime p ∧ is_odd p) → 
  (List.sum first_25_odd_primes).toFloat / 25 = 47.48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_25_odd_primes_l659_65906


namespace NUMINAMATH_CALUDE_profit_share_b_is_1800_l659_65990

/-- Represents the profit share calculation for a business partnership --/
def ProfitShare (investment_a investment_b investment_c : ℕ) (profit_diff_ac : ℕ) : ℕ :=
  let ratio_sum := (investment_a / 2000) + (investment_b / 2000) + (investment_c / 2000)
  let part_value := profit_diff_ac / ((investment_c / 2000) - (investment_a / 2000))
  (investment_b / 2000) * part_value

/-- Theorem stating that given the investments and profit difference, 
    the profit share of b is 1800 --/
theorem profit_share_b_is_1800 :
  ProfitShare 8000 10000 12000 720 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_b_is_1800_l659_65990


namespace NUMINAMATH_CALUDE_students_in_grades_2_and_3_l659_65940

theorem students_in_grades_2_and_3 (boys_grade_2 girls_grade_2 : ℕ) 
  (h1 : boys_grade_2 = 20)
  (h2 : girls_grade_2 = 11)
  (h3 : ∀ x, x = boys_grade_2 + girls_grade_2 → 2 * x = students_grade_3) :
  boys_grade_2 + girls_grade_2 + students_grade_3 = 93 :=
by
  sorry

#check students_in_grades_2_and_3

end NUMINAMATH_CALUDE_students_in_grades_2_and_3_l659_65940


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l659_65957

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l659_65957


namespace NUMINAMATH_CALUDE_car_speed_calculation_l659_65966

/-- Calculates the car speed given train and car travel information -/
theorem car_speed_calculation (train_speed : ℝ) (train_time : ℝ) (remaining_distance : ℝ) (car_time : ℝ) :
  train_speed = 120 →
  train_time = 2 →
  remaining_distance = 2.4 →
  car_time = 3 →
  (train_speed * train_time + remaining_distance) / car_time = 80.8 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_calculation_l659_65966


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l659_65999

theorem smallest_c_for_inequality : 
  ∃ c : ℝ, c > 0 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c * |x - y| ≥ (x + y) / 2) ∧
  (∀ c' : ℝ, c' > 0 → 
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c' * |x - y| ≥ (x + y) / 2) → 
    c' ≥ c) ∧
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l659_65999


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l659_65926

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l659_65926


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_bicycle_price_correct_l659_65944

theorem bicycle_price_calculation (initial_price : ℝ) 
  (first_quarter_increase : ℝ) 
  (second_quarter_increase : ℝ) 
  (third_quarter_decrease : ℝ) 
  (sales_tax : ℝ) : ℝ :=
  let price_after_first_quarter := initial_price * (1 + first_quarter_increase)
  let price_after_second_quarter := price_after_first_quarter * (1 + second_quarter_increase)
  let price_after_third_quarter := price_after_second_quarter * (1 - third_quarter_decrease)
  let final_price := price_after_third_quarter * (1 + sales_tax)
  final_price

theorem bicycle_price_correct : 
  bicycle_price_calculation 220 0.08 0.10 0.05 0.07 = 265.67 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_bicycle_price_correct_l659_65944


namespace NUMINAMATH_CALUDE_book_pages_proof_l659_65979

/-- Proves that a book has 72 pages given the reading conditions -/
theorem book_pages_proof (total_days : ℕ) (fraction_per_day : ℚ) (extra_pages : ℕ) : 
  total_days = 3 → 
  fraction_per_day = 1/4 → 
  extra_pages = 6 → 
  (total_days : ℚ) * (fraction_per_day * (72 : ℚ) + extra_pages) = 72 := by
  sorry

#check book_pages_proof

end NUMINAMATH_CALUDE_book_pages_proof_l659_65979


namespace NUMINAMATH_CALUDE_train_length_calculation_l659_65914

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed_kmh : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed_kmh = 90 →
  time_to_cross = 9.679225661947045 →
  bridge_length = 132 →
  ∃ train_length : ℝ, abs (train_length - 109.98) < 0.01 ∧
    train_length = train_speed_kmh * (1000 / 3600) * time_to_cross - bridge_length :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l659_65914


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l659_65921

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 128 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l659_65921


namespace NUMINAMATH_CALUDE_moon_carbon_percentage_l659_65930

/-- Represents the composition and weight of a celestial body -/
structure CelestialBody where
  weight : ℝ
  iron_percent : ℝ
  carbon_percent : ℝ
  other_percent : ℝ
  other_weight : ℝ

/-- The moon's composition and weight -/
def moon : CelestialBody := {
  weight := 250,
  iron_percent := 50,
  carbon_percent := 20,  -- This is what we want to prove
  other_percent := 30,
  other_weight := 75
}

/-- Mars' composition and weight -/
def mars : CelestialBody := {
  weight := 500,
  iron_percent := 50,
  carbon_percent := 20,
  other_percent := 30,
  other_weight := 150
}

/-- Theorem stating that the moon's carbon percentage is 20% -/
theorem moon_carbon_percentage :
  moon.carbon_percent = 20 ∧
  moon.iron_percent = 50 ∧
  moon.other_percent = 100 - moon.iron_percent - moon.carbon_percent ∧
  moon.weight = 250 ∧
  mars.weight = 2 * moon.weight ∧
  mars.iron_percent = moon.iron_percent ∧
  mars.carbon_percent = moon.carbon_percent ∧
  mars.other_percent = moon.other_percent ∧
  mars.other_weight = 150 ∧
  moon.other_weight = mars.other_weight / 2 := by
  sorry


end NUMINAMATH_CALUDE_moon_carbon_percentage_l659_65930


namespace NUMINAMATH_CALUDE_largest_side_is_sixty_l659_65929

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 180
  area_eq : length * width = 10 * 180

/-- The largest side of a SpecialRectangle is 60 feet -/
theorem largest_side_is_sixty (r : SpecialRectangle) : 
  max r.length r.width = 60 := by
  sorry

#check largest_side_is_sixty

end NUMINAMATH_CALUDE_largest_side_is_sixty_l659_65929


namespace NUMINAMATH_CALUDE_masters_sample_size_l659_65901

/-- Calculates the sample size for a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalRatio : ℕ) (stratumRatio : ℕ) (totalSample : ℕ) : ℕ :=
  (stratumRatio * totalSample) / totalRatio

/-- Proves that the sample size for master's students is 36 given the conditions -/
theorem masters_sample_size :
  let totalRatio : ℕ := 5 + 15 + 9 + 1
  let mastersRatio : ℕ := 9
  let totalSample : ℕ := 120
  stratifiedSampleSize totalRatio mastersRatio totalSample = 36 := by
  sorry

#eval stratifiedSampleSize 30 9 120

end NUMINAMATH_CALUDE_masters_sample_size_l659_65901


namespace NUMINAMATH_CALUDE_f_inequality_l659_65920

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality (x : ℝ) : f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l659_65920


namespace NUMINAMATH_CALUDE_rogers_reading_rate_l659_65951

/-- Roger's book reading problem -/
theorem rogers_reading_rate (total_books : ℕ) (weeks : ℕ) (books_per_week : ℕ) 
  (h1 : total_books = 30)
  (h2 : weeks = 5)
  (h3 : books_per_week * weeks = total_books) :
  books_per_week = 6 := by
sorry

end NUMINAMATH_CALUDE_rogers_reading_rate_l659_65951


namespace NUMINAMATH_CALUDE_gem_bonus_percentage_l659_65972

theorem gem_bonus_percentage (purchase : ℝ) (rate : ℝ) (final_gems : ℝ) : 
  purchase = 250 → 
  rate = 100 → 
  final_gems = 30000 → 
  (final_gems - purchase * rate) / (purchase * rate) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gem_bonus_percentage_l659_65972


namespace NUMINAMATH_CALUDE_set_intersection_problem_l659_65912

theorem set_intersection_problem (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l659_65912


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l659_65919

/-- The number of tan chips in the bag -/
def num_tan : ℕ := 4

/-- The number of pink chips in the bag -/
def num_pink : ℕ := 3

/-- The number of violet chips in the bag -/
def num_violet : ℕ := 5

/-- The number of green chips in the bag -/
def num_green : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := num_tan + num_pink + num_violet + num_green

/-- The probability of drawing the chips in the specified arrangement -/
def probability : ℚ := (num_tan.factorial * num_pink.factorial * num_violet.factorial * 6) / total_chips.factorial

theorem chip_drawing_probability : probability = 1440 / total_chips.factorial :=
sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l659_65919


namespace NUMINAMATH_CALUDE_percentage_increase_l659_65991

theorem percentage_increase (x : ℝ) (h : x = 89.6) :
  ((x - 80) / 80) * 100 = 12 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l659_65991


namespace NUMINAMATH_CALUDE_trapezoid_square_area_equality_l659_65978

/-- Given a trapezoid with upper side 15 cm, lower side 9 cm, and height 12 cm,
    the side length of a square with the same area as the trapezoid is 12 cm. -/
theorem trapezoid_square_area_equality (upper_side lower_side height : ℝ) 
    (h1 : upper_side = 15)
    (h2 : lower_side = 9)
    (h3 : height = 12) :
    ∃ (square_side : ℝ), 
      (1/2 * (upper_side + lower_side) * height = square_side^2) ∧ 
      square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_square_area_equality_l659_65978


namespace NUMINAMATH_CALUDE_problem_solution_l659_65952

theorem problem_solution :
  (∀ a b : ℝ, 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b) ∧
  (∀ x y : ℝ, (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = -6*x*y + 5*y^2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l659_65952


namespace NUMINAMATH_CALUDE_suv_length_sum_l659_65982

/-- Represents the length of a line segment in the grid -/
inductive SegmentLength
  | Straight : SegmentLength  -- Length 1
  | Slanted : SegmentLength   -- Length √2

/-- Counts the number of each type of segment in a letter -/
structure LetterSegments :=
  (straight : ℕ)
  (slanted : ℕ)

/-- Represents the SUV acronym -/
structure SUVAcronym :=
  (S : LetterSegments)
  (U : LetterSegments)
  (V : LetterSegments)

def suv : SUVAcronym :=
  { S := { straight := 5, slanted := 4 },
    U := { straight := 6, slanted := 0 },
    V := { straight := 0, slanted := 2 } }

theorem suv_length_sum :
  let total_straight := suv.S.straight + suv.U.straight + suv.V.straight
  let total_slanted := suv.S.slanted + suv.U.slanted + suv.V.slanted
  total_straight + total_slanted * Real.sqrt 2 = 11 + 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_suv_length_sum_l659_65982


namespace NUMINAMATH_CALUDE_count_seven_up_to_2017_l659_65947

/-- Count of digit 7 in a natural number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for all numbers from 1 to n -/
def sum_count_seven (n : ℕ) : ℕ := sorry

theorem count_seven_up_to_2017 : sum_count_seven 2017 = 602 := by sorry

end NUMINAMATH_CALUDE_count_seven_up_to_2017_l659_65947


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_undefined_l659_65946

theorem sqrt_x_minus_3_undefined (x : ℕ+) : 
  (¬ ∃ (y : ℝ), y^2 = (x : ℝ) - 3) ↔ (x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_undefined_l659_65946


namespace NUMINAMATH_CALUDE_cab_driver_fifth_day_income_verify_cab_driver_income_l659_65950

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
theorem cab_driver_fifth_day_income 
  (income_day1 income_day2 income_day3 income_day4 : ℚ) 
  (average_income : ℚ) : ℚ :=
  let total_income := 5 * average_income
  let sum_four_days := income_day1 + income_day2 + income_day3 + income_day4
  total_income - sum_four_days

/-- Verifies that the calculated fifth day income is correct given the specific values from the problem. -/
theorem verify_cab_driver_income : 
  cab_driver_fifth_day_income 300 150 750 400 420 = 500 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_fifth_day_income_verify_cab_driver_income_l659_65950


namespace NUMINAMATH_CALUDE_test_time_calculation_l659_65913

theorem test_time_calculation (total_questions : ℕ) (unanswered : ℕ) (time_per_question : ℕ) : 
  total_questions = 100 →
  unanswered = 40 →
  time_per_question = 2 →
  (total_questions - unanswered) * time_per_question / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_test_time_calculation_l659_65913


namespace NUMINAMATH_CALUDE_sum_of_squares_l659_65976

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 6*y = -17)
  (eq2 : y^2 + 4*z = 1)
  (eq3 : z^2 + 2*x = 2) :
  x^2 + y^2 + z^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l659_65976


namespace NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l659_65962

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_special_triangle : 
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h₁ := (2 * area) / a
  let h₂ := (2 * area) / b
  let h₃ := (2 * area) / c
  min h₁ (min h₂ h₃) = 7.2 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l659_65962


namespace NUMINAMATH_CALUDE_product_probabilities_l659_65924

/-- The probability of a product having a defect -/
def p₁ : ℝ := 0.1

/-- The probability of the controller detecting an existing defect -/
def p₂ : ℝ := 0.8

/-- The probability of the controller mistakenly rejecting a non-defective product -/
def p₃ : ℝ := 0.3

/-- The probability of a product being mistakenly rejected -/
def P_A₁ : ℝ := (1 - p₁) * p₃

/-- The probability of a product being passed into finished goods with a defect -/
def P_A₂ : ℝ := p₁ * (1 - p₂)

/-- The probability of a product being rejected -/
def P_A₃ : ℝ := p₁ * p₂ + (1 - p₁) * p₃

theorem product_probabilities :
  P_A₁ = 0.27 ∧ P_A₂ = 0.02 ∧ P_A₃ = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_product_probabilities_l659_65924


namespace NUMINAMATH_CALUDE_base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l659_65933

/-- Converts a base 7 number of the form 52x4 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The digit x in 52x4₇ makes the number divisible by 19 --/
theorem base7_divisible_by_19 :
  ∃ x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is unique --/
theorem unique_x_divisible_by_19 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is 4 --/
theorem x_is_4 :
  ∃ x : ℕ, x = 4 ∧ x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

end NUMINAMATH_CALUDE_base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l659_65933


namespace NUMINAMATH_CALUDE_distinct_collections_count_l659_65907

/-- Represents the number of each letter in BIOLOGY --/
structure LetterCount where
  o : Nat
  i : Nat
  y : Nat
  b : Nat
  g : Nat

/-- The initial count of letters in BIOLOGY --/
def initial_count : LetterCount :=
  { o := 2, i := 1, y := 1, b := 1, g := 2 }

/-- A collection of letters that can be put in the bag --/
structure BagCollection where
  vowels : Nat
  consonants : Nat

/-- Check if a collection is valid (3 vowels and 2 consonants) --/
def is_valid_collection (c : BagCollection) : Prop :=
  c.vowels = 3 ∧ c.consonants = 2

/-- Count the number of distinct vowel combinations --/
def count_vowel_combinations (lc : LetterCount) : Nat :=
  sorry

/-- Count the number of distinct consonant combinations --/
def count_consonant_combinations (lc : LetterCount) : Nat :=
  sorry

/-- The main theorem: there are 12 distinct possible collections --/
theorem distinct_collections_count :
  count_vowel_combinations initial_count * count_consonant_combinations initial_count = 12 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l659_65907


namespace NUMINAMATH_CALUDE_parabola_coefficient_l659_65945

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, 2h) and y-intercept at (0, -3h),
    where h ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + 2 * h) → 
  (a * 0^2 + b * 0 + c = -3 * h) →
  b = 10 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l659_65945


namespace NUMINAMATH_CALUDE_cuboid_breadth_l659_65936

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 12) (harea : area = 960) :
  ∃ w : ℝ, cuboidSurfaceArea l w h = area ∧ w = 19.2 := by sorry

end NUMINAMATH_CALUDE_cuboid_breadth_l659_65936


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l659_65949

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l659_65949


namespace NUMINAMATH_CALUDE_min_value_of_f_l659_65984

def f (x : ℝ) : ℝ := 
  Finset.sum (Finset.range 2015) (fun i => (i + 1) * x^(2014 - i))

theorem min_value_of_f :
  ∃ (min : ℝ), min = 1008 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l659_65984


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l659_65989

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 / 3) :
  (a + 1)^2 + a * (1 - a) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l659_65989


namespace NUMINAMATH_CALUDE_gain_amount_calculation_l659_65927

theorem gain_amount_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percentage = 0.10) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 10 := by
sorry

end NUMINAMATH_CALUDE_gain_amount_calculation_l659_65927


namespace NUMINAMATH_CALUDE_profit_calculation_l659_65902

/-- Given that the cost price of 30 articles equals the selling price of x articles,
    and the profit is 25%, prove that x = 24. -/
theorem profit_calculation (x : ℝ) 
  (h1 : 30 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 24 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l659_65902


namespace NUMINAMATH_CALUDE_jim_total_miles_l659_65998

/-- Represents Jim's running schedule over 90 days -/
structure RunningSchedule where
  first_month : Nat  -- Miles per day for the first 30 days
  second_month : Nat -- Miles per day for the second 30 days
  third_month : Nat  -- Miles per day for the third 30 days

/-- Calculates the total miles run given a RunningSchedule -/
def total_miles (schedule : RunningSchedule) : Nat :=
  30 * schedule.first_month + 30 * schedule.second_month + 30 * schedule.third_month

/-- Theorem stating that Jim's total miles run is 1050 -/
theorem jim_total_miles :
  let jim_schedule : RunningSchedule := { first_month := 5, second_month := 10, third_month := 20 }
  total_miles jim_schedule = 1050 := by
  sorry


end NUMINAMATH_CALUDE_jim_total_miles_l659_65998


namespace NUMINAMATH_CALUDE_similar_pentagons_longest_side_l659_65916

/-- A structure representing a pentagon with its longest and shortest sides -/
structure Pentagon where
  longest : ℝ
  shortest : ℝ
  longest_ge_shortest : longest ≥ shortest

/-- Two pentagons are similar if the ratio of their corresponding sides is constant -/
def similar_pentagons (p1 p2 : Pentagon) : Prop :=
  p1.longest / p2.longest = p1.shortest / p2.shortest

theorem similar_pentagons_longest_side 
  (p1 p2 : Pentagon)
  (h_similar : similar_pentagons p1 p2)
  (h_p1_longest : p1.longest = 20)
  (h_p1_shortest : p1.shortest = 4)
  (h_p2_shortest : p2.shortest = 3) :
  p2.longest = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_pentagons_longest_side_l659_65916


namespace NUMINAMATH_CALUDE_max_brownies_l659_65911

theorem max_brownies (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 2 * ((m - 2) * (n - 2)) = 2 * m + 2 * n - 4) : 
  m * n ≤ 84 := by
  sorry

end NUMINAMATH_CALUDE_max_brownies_l659_65911


namespace NUMINAMATH_CALUDE_system_solution_l659_65917

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (4 * x - 3 * y = 5) ∧ 
  (x = 41 / 7) ∧ 
  (y = 43 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l659_65917


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l659_65937

/-- A geometric sequence with third term 5 and fifth term 45 has 5/3 as a possible second term -/
theorem geometric_sequence_second_term (a r : ℝ) : 
  a * r^2 = 5 → a * r^4 = 45 → a * r = 5/3 ∨ a * r = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l659_65937


namespace NUMINAMATH_CALUDE_hana_stamp_collection_l659_65977

/-- The value of Hana's entire stamp collection -/
def total_value : ℚ := 980 / 27

/-- The fraction of the collection Hana sold -/
def sold_fraction : ℚ := 4/7 + 1/3 * (3/7) + 1/5 * (2/7)

/-- The amount Hana earned from her sales -/
def earned_amount : ℚ := 28

theorem hana_stamp_collection :
  sold_fraction * total_value = earned_amount :=
sorry

end NUMINAMATH_CALUDE_hana_stamp_collection_l659_65977


namespace NUMINAMATH_CALUDE_hotdog_cost_l659_65908

/-- The total cost of hot dogs given the number of hot dogs and the price per hot dog. -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℚ) : ℚ :=
  num_hotdogs * price_per_hotdog

/-- Theorem stating that the total cost of 6 hot dogs at 50 cents each is $3.00 -/
theorem hotdog_cost : total_cost 6 (50 / 100) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_cost_l659_65908


namespace NUMINAMATH_CALUDE_train_speed_equation_l659_65960

theorem train_speed_equation (x : ℝ) (h1 : x > 0) (h2 : x + 20 > 0) : 
  (400 / x) - (400 / (x + 20)) = 0.5 ↔ 
  (400 / x) - (400 / (x + 20)) = (30 : ℝ) / 60 ∧
  (400 / x) > (400 / (x + 20)) ∧
  (400 / x) - (400 / (x + 20)) = (400 / x - 400 / (x + 20)) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l659_65960


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l659_65935

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of convex quadrilaterals that can be formed -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l659_65935


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l659_65942

theorem complex_fraction_equality (z : ℂ) :
  z = 2 + I →
  (2 * I) / (z - 1) = 1 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l659_65942


namespace NUMINAMATH_CALUDE_bag_balls_problem_l659_65943

theorem bag_balls_problem (b g : ℕ) (p : ℚ) : 
  b = 8 →
  p = 1/3 →
  p = b / (b + g) →
  g = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bag_balls_problem_l659_65943


namespace NUMINAMATH_CALUDE_total_sums_attempted_l659_65988

/-- Given a student's performance on a set of math problems, calculate the total number of problems attempted. -/
theorem total_sums_attempted
  (correct : ℕ)  -- Number of sums solved correctly
  (h1 : correct = 12)  -- The student solved 12 sums correctly
  (h2 : ∃ wrong : ℕ, wrong = 2 * correct)  -- The student got twice as many sums wrong as right
  : ∃ total : ℕ, total = 3 * correct :=
by sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l659_65988


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l659_65915

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem (ξ η : BinomialRV)
  (hξ : ξ.n = 2)
  (hη : η.n = 4)
  (hp : ξ.p = η.p)
  (hprob : prob_ge ξ 1 = 5/9) :
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l659_65915


namespace NUMINAMATH_CALUDE_inequality_proof_l659_65987

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a / b + b / c > a / c + c / a := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l659_65987


namespace NUMINAMATH_CALUDE_orange_packing_l659_65965

/-- Given a fruit farm that packs oranges in boxes with a variable capacity,
    this theorem proves the relationship between the number of boxes used,
    the total number of oranges, and the capacity of each box. -/
theorem orange_packing (x : ℕ+) :
  (5623 : ℕ) / x.val = (5623 : ℕ) / x.val := by sorry

end NUMINAMATH_CALUDE_orange_packing_l659_65965


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l659_65909

theorem absolute_value_simplification : |(-4^3 + 5^2 - 6)| = 45 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l659_65909


namespace NUMINAMATH_CALUDE_same_solution_equations_l659_65992

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 9 = 0) ∧ (c * x - 5 = -11) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l659_65992


namespace NUMINAMATH_CALUDE_square_sum_squares_l659_65983

theorem square_sum_squares (n : ℕ) : n < 200 → (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_squares_l659_65983


namespace NUMINAMATH_CALUDE_brother_birthday_and_carlos_age_l659_65954

def days_to_weekday (start_day : Nat) (days : Nat) : Nat :=
  (start_day + days) % 7

def years_from_days (days : Nat) : Nat :=
  days / 365

theorem brother_birthday_and_carlos_age 
  (start_day : Nat) 
  (carlos_age : Nat) 
  (days_until_brother_birthday : Nat) :
  start_day = 2 → 
  carlos_age = 7 → 
  days_until_brother_birthday = 2000 → 
  days_to_weekday start_day days_until_brother_birthday = 0 ∧ 
  years_from_days days_until_brother_birthday + carlos_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_brother_birthday_and_carlos_age_l659_65954


namespace NUMINAMATH_CALUDE_percentage_of_b_l659_65958

/-- Given that 8 is 4% of a, a certain percentage of b is 4, and c equals b / a, 
    prove that the percentage of b is 1 / (50c) -/
theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 1 / (50 * c) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l659_65958


namespace NUMINAMATH_CALUDE_a_values_l659_65903

/-- The set of real numbers x such that x^2 - 2x - 8 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

/-- The set of real numbers x such that x^2 + a*x + a^2 - 12 = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

/-- The set of all possible values for a given the conditions -/
def possible_a : Set ℝ := {a | a < -4 ∨ a = -2 ∨ a ≥ 4}

theorem a_values (h : A ∪ B a = A) : a ∈ possible_a := by
  sorry

end NUMINAMATH_CALUDE_a_values_l659_65903


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l659_65967

/-- The trajectory of the midpoint of a line segment PQ, where P is fixed at (4, 0) and Q is on the circle x^2 + y^2 = 4 -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (qx qy : ℝ), qx^2 + qy^2 = 4 ∧ x = (4 + qx) / 2 ∧ y = qy / 2) → 
  (x - 2)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l659_65967


namespace NUMINAMATH_CALUDE_johns_naps_per_week_l659_65922

/-- Given that John takes 60 hours of naps in 70 days, and each nap is 2 hours long,
    prove that he takes 3 naps per week. -/
theorem johns_naps_per_week 
  (nap_duration : ℝ) 
  (total_days : ℝ) 
  (total_nap_hours : ℝ) 
  (h1 : nap_duration = 2)
  (h2 : total_days = 70)
  (h3 : total_nap_hours = 60) :
  (total_nap_hours / (total_days / 7)) / nap_duration = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_naps_per_week_l659_65922


namespace NUMINAMATH_CALUDE_find_y_value_l659_65938

theorem find_y_value (x y : ℝ) (h1 : 3 * (x^2 + x + 1) = y - 6) (h2 : x = -3) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l659_65938


namespace NUMINAMATH_CALUDE_quadratic_inequality_l659_65974

theorem quadratic_inequality (y : ℝ) : y^2 - 6*y - 16 > 0 ↔ y < -2 ∨ y > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l659_65974


namespace NUMINAMATH_CALUDE_equation_solution_l659_65973

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(2*x+8) = 16^(2*x+5) :=
  by
    use -3
    constructor
    · -- Prove that x = -3 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l659_65973


namespace NUMINAMATH_CALUDE_divisibility_of_consecutive_numbers_l659_65985

theorem divisibility_of_consecutive_numbers (n : ℕ) 
  (h1 : ∀ p : ℕ, Prime p → p ∣ n → p^2 ∣ n)
  (h2 : ∀ p : ℕ, Prime p → p ∣ (n + 1) → p^2 ∣ (n + 1))
  (h3 : ∀ p : ℕ, Prime p → p ∣ (n + 2) → p^2 ∣ (n + 2)) :
  ∃ p : ℕ, Prime p ∧ p^3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_consecutive_numbers_l659_65985


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l659_65918

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l659_65918


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l659_65975

theorem sin_product_equals_one_eighth :
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l659_65975


namespace NUMINAMATH_CALUDE_equation_general_form_l659_65969

theorem equation_general_form :
  ∀ x : ℝ, (x - 1) * (2 * x + 1) = 2 ↔ 2 * x^2 - x - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_general_form_l659_65969


namespace NUMINAMATH_CALUDE_male_attendees_fraction_l659_65941

theorem male_attendees_fraction (M F : ℝ) : 
  M + F = 1 → 
  (7/8 : ℝ) * M + (4/5 : ℝ) * F = 0.845 → 
  M = 0.6 := by
sorry

end NUMINAMATH_CALUDE_male_attendees_fraction_l659_65941


namespace NUMINAMATH_CALUDE_problem_1_l659_65948

theorem problem_1 : Real.sqrt 9 * 3⁻¹ + 2^3 / |(-2)| = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_l659_65948


namespace NUMINAMATH_CALUDE_inequality_equivalence_l659_65928

theorem inequality_equivalence (x : ℝ) : (x + 2) / (x - 1) > 3 ↔ 1 < x ∧ x < 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l659_65928


namespace NUMINAMATH_CALUDE_max_negative_integers_in_equation_l659_65963

theorem max_negative_integers_in_equation (a b c d : ℤ) 
  (eq : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  ∀ (n : ℕ), n ≤ (if a < 0 then 1 else 0) + 
              (if b < 0 then 1 else 0) + 
              (if c < 0 then 1 else 0) + 
              (if d < 0 then 1 else 0) → n = 0 :=
sorry

end NUMINAMATH_CALUDE_max_negative_integers_in_equation_l659_65963


namespace NUMINAMATH_CALUDE_complex_number_sum_of_parts_l659_65931

theorem complex_number_sum_of_parts (m : ℝ) : 
  let z : ℂ := m / (1 - Complex.I) + (1 - Complex.I) / 2 * Complex.I
  (z.re + z.im = 1) → m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_sum_of_parts_l659_65931


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l659_65939

theorem opposite_of_negative_2023 : 
  -((-2023 : ℤ)) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l659_65939


namespace NUMINAMATH_CALUDE_josh_marbles_l659_65980

def marble_problem (initial_marbles found_marbles : ℕ) : Prop :=
  initial_marbles + found_marbles = 28

theorem josh_marbles : marble_problem 21 7 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l659_65980


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l659_65934

theorem crayons_in_drawer (initial_crayons : ℕ) :
  (initial_crayons + 3 = 10) → initial_crayons = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l659_65934


namespace NUMINAMATH_CALUDE_polynomial_equation_l659_65955

/-- Given polynomials h and p such that h(x) + p(x) = 3x^2 - x + 4 
    and h(x) = x^4 - 5x^2 + x + 6, prove that p(x) = -x^4 + 8x^2 - 2x - 2 -/
theorem polynomial_equation (x : ℝ) (h p : ℝ → ℝ) 
    (h_p_sum : ∀ x, h x + p x = 3 * x^2 - x + 4)
    (h_def : ∀ x, h x = x^4 - 5 * x^2 + x + 6) :
  p x = -x^4 + 8 * x^2 - 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_l659_65955


namespace NUMINAMATH_CALUDE_problem_solution_l659_65923

theorem problem_solution (x y z : ℚ) 
  (sum_condition : x + y + z = 120)
  (equal_condition : x + 10 = y - 5 ∧ y - 5 = 4*z) : 
  y = 545/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l659_65923


namespace NUMINAMATH_CALUDE_hundred_day_previous_year_is_saturday_l659_65995

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Returns true if the year is a leap year, false otherwise -/
def isLeapYear (y : Year) : Bool := sorry

/-- The number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  if isLeapYear y then 366 else 365

theorem hundred_day_previous_year_is_saturday 
  (N : Year)
  (h1 : dayOfWeek N 400 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 300 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_hundred_day_previous_year_is_saturday_l659_65995


namespace NUMINAMATH_CALUDE_calculation_1_l659_65964

theorem calculation_1 : (1 * (-1/9)) - (1/2) = -11/18 := by sorry

end NUMINAMATH_CALUDE_calculation_1_l659_65964


namespace NUMINAMATH_CALUDE_min_detectors_for_ship_detection_l659_65925

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 2x2 ship placement on the board -/
structure Ship :=
  (topLeft : Cell)

/-- Represents a detector placement on the board -/
structure Detector :=
  (position : Cell)

/-- A function that determines if a ship occupies a given cell -/
def shipOccupies (s : Ship) (c : Cell) : Prop :=
  s.topLeft.x ≤ c.x ∧ c.x < s.topLeft.x + 2 ∧
  s.topLeft.y ≤ c.y ∧ c.y < s.topLeft.y + 2

/-- A function that determines if a detector can detect a ship -/
def detectorDetects (d : Detector) (s : Ship) : Prop :=
  shipOccupies s d.position

/-- The main theorem stating that 16 detectors are sufficient and necessary -/
theorem min_detectors_for_ship_detection :
  ∃ (detectors : Finset Detector),
    (detectors.card = 16) ∧
    (∀ (s : Ship), ∃ (d : Detector), d ∈ detectors ∧ detectorDetects d s) ∧
    (∀ (detectors' : Finset Detector),
      detectors'.card < 16 →
      ∃ (s : Ship), ∀ (d : Detector), d ∈ detectors' → ¬detectorDetects d s) :=
by sorry

end NUMINAMATH_CALUDE_min_detectors_for_ship_detection_l659_65925


namespace NUMINAMATH_CALUDE_removal_doubles_probability_l659_65910

/-- The number of red clips initially -/
def red_clips : ℕ := 4

/-- The total number of clips initially -/
def total_clips : ℕ := 16

/-- The number of clips removed -/
def k : ℕ := 12

/-- The probability of choosing a red clip is doubled after removal -/
def probability_doubled (red : ℕ) (total : ℕ) (removed : ℕ) : Prop :=
  (red : ℚ) / (total - removed : ℚ) = 2 * ((red : ℚ) / (total : ℚ))

theorem removal_doubles_probability :
  probability_doubled red_clips total_clips k := by sorry

end NUMINAMATH_CALUDE_removal_doubles_probability_l659_65910


namespace NUMINAMATH_CALUDE_math_olympiad_reform_l659_65970

-- Define the probability of achieving a top-20 ranking in a single competition
def top20_prob : ℚ := 1/4

-- Define the maximum number of competitions
def max_competitions : ℕ := 5

-- Define the number of top-20 rankings needed to join the provincial team
def required_top20 : ℕ := 2

-- Define the function to calculate the probability of joining the provincial team
def prob_join_team : ℚ := sorry

-- Define the random variable ξ representing the number of competitions participated
def ξ : ℕ → ℚ
| 2 => 1/16
| 3 => 3/32
| 4 => 27/64
| 5 => 27/64
| _ => 0

-- Define the expected value of ξ
def expected_ξ : ℚ := sorry

-- Theorem statement
theorem math_olympiad_reform :
  (prob_join_team = 67/256) ∧ (expected_ξ = 356/256) := by sorry

end NUMINAMATH_CALUDE_math_olympiad_reform_l659_65970


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l659_65996

/-- Pascal's triangle as a function from row and column to the value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- The set of all numbers in Pascal's triangle -/
def pascalNumbers : Set ℕ := sorry

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {n ∈ pascalNumbers | 1000 ≤ n ∧ n ≤ 9999}

/-- The third smallest element in a set of natural numbers -/
def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal : 
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l659_65996


namespace NUMINAMATH_CALUDE_inverse_variation_with_increase_l659_65981

/-- Given two quantities a and b that vary inversely, prove that when their product increases by 50% and a becomes 1600, b equals 0.375 -/
theorem inverse_variation_with_increase (a b a' b' : ℝ) : 
  (a * b = 800 * 0.5) →  -- Initial condition
  (a' * b' = 1.5 * a * b) →  -- 50% increase in product
  (a' = 1600) →  -- New value of a
  (b' = 0.375) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_with_increase_l659_65981


namespace NUMINAMATH_CALUDE_expression_evaluation_l659_65968

theorem expression_evaluation :
  let x : ℚ := -2
  let expr := (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1))
  expr = 3/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l659_65968


namespace NUMINAMATH_CALUDE_range_of_f_l659_65994

noncomputable def f (x : ℝ) : ℝ := (3 - 2^x) / (1 + 2^x)

theorem range_of_f :
  (∀ y ∈ Set.range f, -1 < y ∧ y < 3) ∧
  (∀ ε > 0, ∃ x₁ x₂, f x₁ < -1 + ε ∧ f x₂ > 3 - ε) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l659_65994


namespace NUMINAMATH_CALUDE_min_weighings_is_two_l659_65900

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A strategy for finding the real medal -/
def Strategy := List WeighResult → Nat

/-- The total number of medals -/
def totalMedals : Nat := 9

/-- The number of real medals -/
def realMedals : Nat := 1

/-- A weighing operation that compares two sets of medals -/
def weigh (leftSet rightSet : List Nat) : WeighResult := sorry

/-- Checks if a strategy correctly identifies the real medal -/
def isValidStrategy (s : Strategy) : Prop := sorry

/-- The minimum number of weighings required to find the real medal -/
def minWeighings : Nat := sorry

theorem min_weighings_is_two :
  minWeighings = 2 := by sorry

end NUMINAMATH_CALUDE_min_weighings_is_two_l659_65900


namespace NUMINAMATH_CALUDE_profit_calculation_l659_65959

/-- The number of pencils bought by the store owner -/
def total_pencils : ℕ := 2000

/-- The cost price of each pencil in dollars -/
def cost_price : ℚ := 15 / 100

/-- The selling price of each pencil in dollars -/
def selling_price : ℚ := 30 / 100

/-- The desired profit in dollars -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def pencils_to_sell : ℕ := 1500

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * cost_price = desired_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l659_65959


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l659_65932

/-- The parabolas y = (x - 1)^2 and x - 2 = (y + 1)^2 intersect at four points. 
    These points lie on a circle with radius squared equal to 1/4. -/
theorem intersection_points_on_circle : 
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 1)^2 ∧ p.1 - 2 = (p.2 + 1)^2) → 
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l659_65932


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_105_l659_65956

theorem last_three_digits_of_7_to_105 : 7^105 ≡ 783 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_105_l659_65956


namespace NUMINAMATH_CALUDE_paint_fraction_proof_l659_65905

def paint_problem (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : Prop :=
  let remaining_after_first_week : ℚ := initial_paint - (first_week_fraction * initial_paint)
  let used_second_week : ℚ := total_used - (first_week_fraction * initial_paint)
  (used_second_week / remaining_after_first_week) = 1 / 6

theorem paint_fraction_proof :
  paint_problem 360 (1 / 4) 135 := by
  sorry

end NUMINAMATH_CALUDE_paint_fraction_proof_l659_65905
