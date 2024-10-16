import Mathlib

namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l631_63172

/-- A five-digit palindromic number -/
def FiveDigitPalindrome (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- The count of five-digit palindromic numbers -/
def CountFiveDigitPalindromes : ℕ := 90

theorem five_digit_palindromes_count :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = FiveDigitPalindrome a b c) ↔
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = 10000 * a + 1000 * b + 100 * c + 10 * b + a)) →
  CountFiveDigitPalindromes = (9 : ℕ) * (10 : ℕ) * (1 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l631_63172


namespace NUMINAMATH_CALUDE_race_time_proof_l631_63178

/-- The time A takes to complete the race -/
def race_time_A : ℝ := 390

/-- The distance of the race in meters -/
def race_distance : ℝ := 1000

/-- The difference in distance between A and B at the finish -/
def distance_diff_AB : ℝ := 25

/-- The time difference between A and B -/
def time_diff_AB : ℝ := 10

/-- The difference in distance between A and C at the finish -/
def distance_diff_AC : ℝ := 40

/-- The time difference between A and C -/
def time_diff_AC : ℝ := 8

/-- The difference in distance between B and C at the finish -/
def distance_diff_BC : ℝ := 15

/-- The time difference between B and C -/
def time_diff_BC : ℝ := 2

theorem race_time_proof :
  let v_a := race_distance / race_time_A
  let v_b := (race_distance - distance_diff_AB) / race_time_A
  let v_c := (race_distance - distance_diff_AC) / race_time_A
  (v_b * (race_time_A + time_diff_AB) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AC) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AB + time_diff_BC) = race_distance) →
  race_time_A = 390 := by
sorry

end NUMINAMATH_CALUDE_race_time_proof_l631_63178


namespace NUMINAMATH_CALUDE_money_distribution_l631_63167

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (BC_sum : B + C = 150) : 
  C = 50 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l631_63167


namespace NUMINAMATH_CALUDE_curling_survey_probability_l631_63135

/-- Represents the survey data and selection process for the Winter Olympic Games curling interest survey. -/
structure CurlingSurvey where
  total_participants : Nat
  male_to_female_ratio : Rat
  interested_ratio : Rat
  uninterested_females : Nat
  selected_interested : Nat
  chosen_promoters : Nat

/-- Calculates the probability of selecting at least one female from the chosen promoters. -/
def probability_at_least_one_female (survey : CurlingSurvey) : Rat :=
  sorry

/-- Theorem stating that given the survey conditions, the probability of selecting at least one female is 9/14. -/
theorem curling_survey_probability (survey : CurlingSurvey) 
  (h1 : survey.total_participants = 600)
  (h2 : survey.male_to_female_ratio = 2/1)
  (h3 : survey.interested_ratio = 2/3)
  (h4 : survey.uninterested_females = 50)
  (h5 : survey.selected_interested = 8)
  (h6 : survey.chosen_promoters = 2) :
  probability_at_least_one_female survey = 9/14 :=
sorry

end NUMINAMATH_CALUDE_curling_survey_probability_l631_63135


namespace NUMINAMATH_CALUDE_nickel_piles_count_l631_63105

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value Rocco has in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of nickels is 9 -/
theorem nickel_piles_count : 
  ∃ (nickel_piles : ℕ), 
    nickel_piles = 9 ∧
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = 
    total_value :=
by sorry

end NUMINAMATH_CALUDE_nickel_piles_count_l631_63105


namespace NUMINAMATH_CALUDE_equal_distribution_of_sweets_l631_63108

/-- Proves that each student receives 4 sweet treats given the conditions -/
theorem equal_distribution_of_sweets
  (cookies : ℕ) (cupcakes : ℕ) (brownies : ℕ) (students : ℕ)
  (h_cookies : cookies = 20)
  (h_cupcakes : cupcakes = 25)
  (h_brownies : brownies = 35)
  (h_students : students = 20)
  : (cookies + cupcakes + brownies) / students = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_sweets_l631_63108


namespace NUMINAMATH_CALUDE_fish_tagging_ratio_l631_63106

theorem fish_tagging_ratio : 
  ∀ (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) (total_fish : ℕ),
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  total_fish = 3200 →
  (tagged_in_second : ℚ) / second_catch = 1 / 40 := by
sorry

end NUMINAMATH_CALUDE_fish_tagging_ratio_l631_63106


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_range_l631_63100

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def base_n_value (n : ℕ) : ℕ :=
  n^3 + 2*n^2 + 3*n + 4

theorem no_perfect_squares_in_range : 
  ¬ ∃ n : ℕ, 5 ≤ n ∧ n ≤ 20 ∧ is_perfect_square (base_n_value n) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_range_l631_63100


namespace NUMINAMATH_CALUDE_complement_of_M_l631_63176

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem complement_of_M : Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l631_63176


namespace NUMINAMATH_CALUDE_point_position_l631_63144

/-- An isosceles triangle with a point on its base satisfying certain conditions -/
structure IsoscelesTriangleWithPoint where
  -- The length of the base of the isosceles triangle
  a : ℝ
  -- The height of the isosceles triangle
  h : ℝ
  -- The distance from one endpoint of the base to the point on the base
  x : ℝ
  -- Condition: a > 0 (positive base length)
  a_pos : a > 0
  -- Condition: h > 0 (positive height)
  h_pos : h > 0
  -- Condition: 0 < x < a (point is on the base)
  x_on_base : 0 < x ∧ x < a
  -- Condition: BM + MA = 2h
  sum_condition : x + (2 * h - x) = 2 * h

/-- Theorem: The position of the point on the base satisfies the quadratic equation -/
theorem point_position (t : IsoscelesTriangleWithPoint) : 
  t.x = t.h + (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 ∨ 
  t.x = t.h - (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_point_position_l631_63144


namespace NUMINAMATH_CALUDE_complex_number_equality_l631_63128

theorem complex_number_equality (z : ℂ) :
  Complex.abs (z - 2) = 5 ∧ 
  Complex.abs (z + 4) = 5 ∧ 
  Complex.abs (z - 2*I) = 5 → 
  z = -1 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l631_63128


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l631_63182

theorem factorization_of_quadratic (x : ℝ) : 4 * x^2 - 2 * x = 2 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l631_63182


namespace NUMINAMATH_CALUDE_log_8_x_equals_3_75_l631_63110

theorem log_8_x_equals_3_75 (x : ℝ) :
  Real.log x / Real.log 8 = 3.75 → x = 1024 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_x_equals_3_75_l631_63110


namespace NUMINAMATH_CALUDE_tony_fish_count_l631_63160

/-- The number of fish Tony's parents buy each year -/
def fish_bought_yearly : ℕ := 2

/-- The number of years that pass -/
def years : ℕ := 5

/-- The number of fish Tony starts with -/
def initial_fish : ℕ := 2

/-- The number of fish that die each year -/
def fish_lost_yearly : ℕ := 1

/-- The number of fish Tony has after 5 years -/
def final_fish : ℕ := 7

theorem tony_fish_count :
  initial_fish + years * (fish_bought_yearly - fish_lost_yearly) = final_fish :=
by sorry

end NUMINAMATH_CALUDE_tony_fish_count_l631_63160


namespace NUMINAMATH_CALUDE_triangle_properties_l631_63116

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi / 2 ∧
  Real.cos (2 * A) = -1 / 3 ∧
  c = Real.sqrt 3 ∧
  Real.sin A = Real.sqrt 6 * Real.sin C

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : 
  a = 3 * Real.sqrt 2 ∧ 
  b = 5 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (5 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l631_63116


namespace NUMINAMATH_CALUDE_complex_number_real_minus_imag_l631_63140

theorem complex_number_real_minus_imag : 
  let z : ℂ := 5 / (-3 - Complex.I)
  let a : ℝ := z.re
  let b : ℝ := z.im
  a - b = -2 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_minus_imag_l631_63140


namespace NUMINAMATH_CALUDE_cd_combined_length_l631_63133

/-- The combined length of 3 CDs is 6 hours, given that two CDs are 1.5 hours each and the third CD is twice as long as the shorter ones. -/
theorem cd_combined_length : 
  let short_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * short_cd_length
  let total_length : ℝ := 2 * short_cd_length + long_cd_length
  total_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_combined_length_l631_63133


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_l631_63192

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

theorem rachel_furniture_assembly :
  ∀ (chairs tables : ℕ) (time_per_piece total_time : ℕ),
  chairs = 7 →
  time_per_piece = 4 →
  total_time = 40 →
  total_time = time_per_piece * (chairs + tables) →
  tables = num_tables :=
by sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_l631_63192


namespace NUMINAMATH_CALUDE_factorial_calculation_l631_63152

theorem factorial_calculation : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l631_63152


namespace NUMINAMATH_CALUDE_polynomial_parity_and_divisibility_l631_63174

theorem polynomial_parity_and_divisibility (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ p % 2 = 1 ∧ q % 2 = 0) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ p % 2 = 1 ∧ q % 2 = 1) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_parity_and_divisibility_l631_63174


namespace NUMINAMATH_CALUDE_water_depth_for_specific_cylinder_l631_63183

/-- Represents a cylindrical tower partially submerged in water -/
structure SubmergedCylinder where
  height : ℝ
  radius : ℝ
  aboveWaterRatio : ℝ

/-- Calculates the depth of water at the base of a partially submerged cylinder -/
def waterDepth (c : SubmergedCylinder) : ℝ :=
  c.height * (1 - c.aboveWaterRatio)

/-- Theorem stating the water depth for a specific cylinder -/
theorem water_depth_for_specific_cylinder :
  let c : SubmergedCylinder := {
    height := 1200,
    radius := 100,
    aboveWaterRatio := 1/3
  }
  waterDepth c = 400 := by sorry

end NUMINAMATH_CALUDE_water_depth_for_specific_cylinder_l631_63183


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l631_63158

theorem stratified_sampling_female_count 
  (total_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : female_students = 800)
  (h3 : sample_size = 50) :
  (sample_size * female_students) / total_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l631_63158


namespace NUMINAMATH_CALUDE_point_on_angle_terminal_side_l631_63141

theorem point_on_angle_terminal_side (P : ℝ × ℝ) (θ : ℝ) (h1 : θ = 2 * π / 3) (h2 : P.1 = -1) :
  P.2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_angle_terminal_side_l631_63141


namespace NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l631_63149

theorem nilpotent_matrix_square_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l631_63149


namespace NUMINAMATH_CALUDE_min_value_theorem_l631_63120

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l631_63120


namespace NUMINAMATH_CALUDE_parkers_richies_ratio_l631_63155

/-- Given that Parker's share is $50 and the total shared amount is $125,
    prove that the ratio of Parker's share to Richie's share is 2:3. -/
theorem parkers_richies_ratio (parker_share : ℝ) (total_share : ℝ) :
  parker_share = 50 →
  total_share = 125 →
  parker_share < total_share →
  ∃ (a b : ℕ), a = 2 ∧ b = 3 ∧ parker_share / (total_share - parker_share) = a / b :=
by sorry

end NUMINAMATH_CALUDE_parkers_richies_ratio_l631_63155


namespace NUMINAMATH_CALUDE_sqrt_sum_max_value_l631_63162

theorem sqrt_sum_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ m :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_max_value_l631_63162


namespace NUMINAMATH_CALUDE_hyperbola_equation_l631_63177

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m * 2 = Real.sqrt 3) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →
  a = 2 ∧ b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l631_63177


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l631_63199

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the longer base -/
  a : ℝ
  /-- The length of the shorter base -/
  b : ℝ
  /-- The height of the trapezoid -/
  h : ℝ
  /-- The condition that the trapezoid is isosceles -/
  isIsosceles : True
  /-- The condition that the diagonals are perpendicular -/
  diagonalsPerpendicular : True
  /-- The midline length is 5 -/
  midline_eq : (a + b) / 2 = 5

/-- The area of an isosceles trapezoid with perpendicular diagonals and midline length 5 is 25 -/
theorem isosceles_trapezoid_area (T : IsoscelesTrapezoid) : (T.a + T.b) * T.h / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l631_63199


namespace NUMINAMATH_CALUDE_max_sum_sides_is_ten_l631_63113

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  
/-- Represents a region formed by the intersection of lines -/
structure Region where
  num_sides : ℕ

/-- Represents two neighboring regions -/
structure NeighboringRegions where
  region1 : Region
  region2 : Region

/-- The maximum sum of sides for two neighboring regions in a configuration with 7 lines -/
def max_sum_sides (config : LineConfiguration) : ℕ :=
  10

/-- Theorem: The maximum sum of sides for two neighboring regions in a configuration with 7 lines is 10 -/
theorem max_sum_sides_is_ten (config : LineConfiguration) 
  (h : config.num_lines = 7) : 
  ∀ (neighbors : NeighboringRegions), 
    neighbors.region1.num_sides + neighbors.region2.num_sides ≤ max_sum_sides config :=
by
  sorry

#check max_sum_sides_is_ten

end NUMINAMATH_CALUDE_max_sum_sides_is_ten_l631_63113


namespace NUMINAMATH_CALUDE_final_brownies_count_l631_63175

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The initial number of brownies made by Mother -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mooney ate -/
def mooney_ate : ℕ := 4

/-- The number of new brownies Mother made the next day -/
def new_brownies : ℕ := 2 * dozen

/-- Theorem stating the final number of brownies on the counter -/
theorem final_brownies_count :
  initial_brownies - father_ate - mooney_ate + new_brownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_brownies_count_l631_63175


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l631_63171

/-- Given a point P, return its symmetric point with respect to the y-axis -/
def symmetric_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Given a point P, return its symmetric point with respect to the x-axis -/
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-10, -1)
  let P₁ : ℝ × ℝ := symmetric_y P
  let P₂ : ℝ × ℝ := symmetric_x P₁
  P₂ = (10, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l631_63171


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l631_63129

/-- Represents a gathering of people with specific knowledge relationships. -/
structure Gathering where
  total : Nat
  know_each_other : Nat
  know_no_one : Nat

/-- Calculates the number of handshakes in a gathering. -/
def count_handshakes (g : Gathering) : Nat :=
  g.know_no_one * (g.total - 1)

/-- Theorem stating that in a specific gathering, 217 handshakes occur. -/
theorem handshakes_in_specific_gathering :
  ∃ (g : Gathering),
    g.total = 30 ∧
    g.know_each_other = 15 ∧
    g.know_no_one = 15 ∧
    count_handshakes g = 217 := by
  sorry

#check handshakes_in_specific_gathering

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l631_63129


namespace NUMINAMATH_CALUDE_prob_non_defective_product_l631_63109

theorem prob_non_defective_product (prob_grade_b prob_grade_c : ℝ) 
  (h1 : prob_grade_b = 0.03)
  (h2 : prob_grade_c = 0.01)
  (h3 : 0 ≤ prob_grade_b ∧ prob_grade_b ≤ 1)
  (h4 : 0 ≤ prob_grade_c ∧ prob_grade_c ≤ 1)
  (h5 : prob_grade_b + prob_grade_c ≤ 1) :
  1 - (prob_grade_b + prob_grade_c) = 0.96 := by
sorry

end NUMINAMATH_CALUDE_prob_non_defective_product_l631_63109


namespace NUMINAMATH_CALUDE_gcd_5040_13860_l631_63101

theorem gcd_5040_13860 : Nat.gcd 5040 13860 = 420 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5040_13860_l631_63101


namespace NUMINAMATH_CALUDE_fraction_equality_l631_63111

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l631_63111


namespace NUMINAMATH_CALUDE_gold_medals_count_l631_63112

theorem gold_medals_count (total : ℕ) (silver : ℕ) (bronze : ℕ) (h1 : total = 67) (h2 : silver = 32) (h3 : bronze = 16) :
  total - silver - bronze = 19 := by
  sorry

end NUMINAMATH_CALUDE_gold_medals_count_l631_63112


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l631_63169

def complex_i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (1 - complex_i) / (1 + complex_i)^2 = -1/2 - complex_i/2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l631_63169


namespace NUMINAMATH_CALUDE_negative_squared_greater_than_product_l631_63153

theorem negative_squared_greater_than_product {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > a*b := by
  sorry

end NUMINAMATH_CALUDE_negative_squared_greater_than_product_l631_63153


namespace NUMINAMATH_CALUDE_non_adjacent_white_balls_arrangements_select_balls_with_min_score_l631_63122

/-- Represents the number of red balls in the bag -/
def num_red_balls : ℕ := 5

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := 4

/-- Represents the score for taking out a red ball -/
def red_ball_score : ℕ := 2

/-- Represents the score for taking out a white ball -/
def white_ball_score : ℕ := 1

/-- Represents the minimum required score -/
def min_score : ℕ := 8

/-- Represents the number of balls to be taken out -/
def balls_to_take : ℕ := 5

/-- Theorem for the number of ways to arrange balls with non-adjacent white balls -/
theorem non_adjacent_white_balls_arrangements : ℕ := by sorry

/-- Theorem for the number of ways to select balls with a minimum score -/
theorem select_balls_with_min_score : ℕ := by sorry

end NUMINAMATH_CALUDE_non_adjacent_white_balls_arrangements_select_balls_with_min_score_l631_63122


namespace NUMINAMATH_CALUDE_adjacent_number_in_triangular_arrangement_l631_63130

/-- Function to calculate the first number in the k-th row -/
def first_number_in_row (k : ℕ) : ℕ := (k - 1)^2 + 1

/-- Function to calculate the last number in the k-th row -/
def last_number_in_row (k : ℕ) : ℕ := k^2

/-- Function to determine if a number is in the k-th row -/
def is_in_row (n : ℕ) (k : ℕ) : Prop :=
  first_number_in_row k ≤ n ∧ n ≤ last_number_in_row k

/-- Function to calculate the number below a given number in the triangular arrangement -/
def number_below (n : ℕ) : ℕ :=
  let k := (n.sqrt + 1 : ℕ)
  let position := n - first_number_in_row k + 1
  first_number_in_row (k + 1) + position - 1

theorem adjacent_number_in_triangular_arrangement :
  is_in_row 267 17 → number_below 267 = 301 := by sorry

end NUMINAMATH_CALUDE_adjacent_number_in_triangular_arrangement_l631_63130


namespace NUMINAMATH_CALUDE_tyson_basketball_scores_l631_63166

theorem tyson_basketball_scores (three_pointers : ℕ) (one_pointers : ℕ) (total_points : ℕ) :
  three_pointers = 15 →
  one_pointers = 6 →
  total_points = 75 →
  ∃ (two_pointers : ℕ), 
    3 * three_pointers + 2 * two_pointers + one_pointers = total_points ∧
    two_pointers = 12 := by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_scores_l631_63166


namespace NUMINAMATH_CALUDE_stating_two_cookies_per_guest_l631_63147

/-- 
Given a total number of cookies and guests, calculates the number of cookies per guest,
assuming each guest receives the same number of cookies.
-/
def cookiesPerGuest (totalCookies guests : ℕ) : ℚ :=
  totalCookies / guests

/-- 
Theorem stating that when there are 10 cookies and 5 guests,
each guest receives 2 cookies.
-/
theorem two_cookies_per_guest :
  cookiesPerGuest 10 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_cookies_per_guest_l631_63147


namespace NUMINAMATH_CALUDE_nates_dropped_matches_l631_63186

/-- Proves that Nate dropped 10 matches in the creek given the initial conditions. -/
theorem nates_dropped_matches (initial_matches : ℕ) (remaining_matches : ℕ) (dropped_matches : ℕ) :
  initial_matches = 70 →
  remaining_matches = 40 →
  initial_matches - remaining_matches = dropped_matches + 2 * dropped_matches →
  dropped_matches = 10 := by
sorry

end NUMINAMATH_CALUDE_nates_dropped_matches_l631_63186


namespace NUMINAMATH_CALUDE_complement_of_A_l631_63118

def A : Set ℝ := {x | (x - 1) / (x - 2) ≥ 0}

theorem complement_of_A : (Set.univ \ A) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l631_63118


namespace NUMINAMATH_CALUDE_jasper_sold_31_drinks_l631_63107

/-- Represents the number of items sold by Jasper -/
structure JasperSales where
  chips : ℕ
  hot_dogs : ℕ
  drinks : ℕ

/-- Calculates the number of drinks sold by Jasper -/
def calculate_drinks (sales : JasperSales) : ℕ :=
  sales.chips - 8 + 12

/-- Theorem stating that Jasper sold 31 drinks -/
theorem jasper_sold_31_drinks (sales : JasperSales) 
  (h1 : sales.chips = 27)
  (h2 : sales.hot_dogs = sales.chips - 8)
  (h3 : sales.drinks = sales.hot_dogs + 12) :
  sales.drinks = 31 := by
  sorry

end NUMINAMATH_CALUDE_jasper_sold_31_drinks_l631_63107


namespace NUMINAMATH_CALUDE_mixed_beads_cost_l631_63185

/-- The cost per box of mixed beads -/
def cost_per_box_mixed (red_cost yellow_cost : ℚ) (total_boxes red_boxes yellow_boxes : ℕ) : ℚ :=
  (red_cost * red_boxes + yellow_cost * yellow_boxes) / total_boxes

/-- Theorem stating the cost per box of mixed beads is $1.32 -/
theorem mixed_beads_cost :
  cost_per_box_mixed (13/10) 2 10 4 4 = 132/100 := by
  sorry

end NUMINAMATH_CALUDE_mixed_beads_cost_l631_63185


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l631_63188

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : 10^2022 ≤ q ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2022 ≤ p ∧ p < 10^2023 → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l631_63188


namespace NUMINAMATH_CALUDE_negative_sqrt_point_eight_one_equals_negative_point_nine_l631_63180

theorem negative_sqrt_point_eight_one_equals_negative_point_nine :
  -Real.sqrt 0.81 = -0.9 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_point_eight_one_equals_negative_point_nine_l631_63180


namespace NUMINAMATH_CALUDE_inequality_system_solution_l631_63196

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, x > 1 ↔ (x - 1 > 0 ∧ 2*x - a > 0)) →
  a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l631_63196


namespace NUMINAMATH_CALUDE_find_divisor_l631_63173

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 125 → 
  quotient = 8 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder →
  divisor = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l631_63173


namespace NUMINAMATH_CALUDE_wax_required_for_feathers_l631_63114

/-- The amount of wax Icarus has, in grams. -/
def total_wax : ℕ := 557

/-- The amount of wax needed for the feathers, in grams. -/
def wax_needed : ℕ := 17

/-- Theorem stating that the amount of wax required for the feathers is equal to the amount needed, regardless of the total amount available. -/
theorem wax_required_for_feathers : wax_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_wax_required_for_feathers_l631_63114


namespace NUMINAMATH_CALUDE_unique_function_theorem_l631_63163

-- Define the function type
def IntFunction := ℤ → ℤ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : IntFunction) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

-- State the theorem
theorem unique_function_theorem :
  ∀ f : IntFunction, SatisfiesEquation f → ∀ n : ℤ, f n = 2 * n + 1007 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l631_63163


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l631_63125

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (liters_per_tank : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * liters_per_tank

/-- Theorem stating the total water capacity for the farmer's specific setup -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_farmer_water_capacity_l631_63125


namespace NUMINAMATH_CALUDE_dave_ticket_difference_l631_63136

theorem dave_ticket_difference (toys clothes : ℕ) 
  (h1 : toys = 12) 
  (h2 : clothes = 7) : 
  toys - clothes = 5 := by
  sorry

end NUMINAMATH_CALUDE_dave_ticket_difference_l631_63136


namespace NUMINAMATH_CALUDE_quadratic_greater_than_linear_l631_63134

theorem quadratic_greater_than_linear (x : ℝ) :
  let y₁ : ℝ → ℝ := λ x => x + 1
  let y₂ : ℝ → ℝ := λ x => (1/2) * x^2 - (1/2) * x - 1
  (y₂ x > y₁ x) ↔ (x < -1 ∨ x > 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_greater_than_linear_l631_63134


namespace NUMINAMATH_CALUDE_train_speed_calculation_l631_63145

/-- The speed of a train given another train passing in the opposite direction -/
theorem train_speed_calculation (passing_time : ℝ) (goods_train_length : ℝ) (goods_train_speed : ℝ) :
  passing_time = 9 →
  goods_train_length = 280 →
  goods_train_speed = 52 →
  ∃ (man_train_speed : ℝ), abs (man_train_speed - 60.16) < 0.01 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l631_63145


namespace NUMINAMATH_CALUDE_second_shop_payment_l631_63156

/-- The amount paid for books from the second shop -/
def amount_second_shop (books_first_shop : ℕ) (books_second_shop : ℕ) 
  (price_first_shop : ℚ) (average_price : ℚ) : ℚ := 
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_first_shop

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_payment : 
  amount_second_shop 65 50 1160 (18088695652173913 / 1000000000000000) = 920 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_payment_l631_63156


namespace NUMINAMATH_CALUDE_total_legs_calculation_l631_63143

theorem total_legs_calculation (total_tables : ℕ) (four_legged_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : four_legged_tables = 16)
  (h3 : four_legged_tables ≤ total_tables) :
  four_legged_tables * 4 + (total_tables - four_legged_tables) * 3 = 124 := by
  sorry

#check total_legs_calculation

end NUMINAMATH_CALUDE_total_legs_calculation_l631_63143


namespace NUMINAMATH_CALUDE_percentage_without_muffin_l631_63138

theorem percentage_without_muffin (muffin yogurt fruit granola : ℝ) :
  muffin = 38 →
  yogurt = 10 →
  fruit = 27 →
  granola = 25 →
  muffin + yogurt + fruit + granola = 100 →
  100 - muffin = 62 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_muffin_l631_63138


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l631_63164

theorem stewart_farm_ratio : ∀ (num_sheep num_horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
  num_sheep = 24 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  num_horses * horse_food_per_day = total_horse_food →
  num_sheep * 7 = num_horses * 3 :=
by sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l631_63164


namespace NUMINAMATH_CALUDE_pyramid_x_value_l631_63168

structure Pyramid where
  top : ℕ
  row2_left : ℕ
  row3_left : ℕ
  row4_left : ℕ
  row4_right : ℕ
  row5_left : ℕ
  row5_right : ℕ

def pyramid_sum (a b : ℕ) : ℕ := a + b

def calculate_x (pyr : Pyramid) : ℕ :=
  let p := pyramid_sum pyr.row2_left (pyr.top - pyr.row2_left)
  let q := p - pyr.row3_left
  let r := pyr.row2_left - q
  let s := r - pyr.row4_left
  let t := pyr.row4_left - pyr.row5_left
  s - t

theorem pyramid_x_value (pyr : Pyramid) 
  (h1 : pyr.top = 105)
  (h2 : pyr.row2_left = 47)
  (h3 : pyr.row3_left = 31)
  (h4 : pyr.row4_left = 13)
  (h5 : pyr.row4_right = 9)
  (h6 : pyr.row5_left = 9)
  (h7 : pyr.row5_right = 4) :
  calculate_x pyr = 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_x_value_l631_63168


namespace NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l631_63148

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l631_63148


namespace NUMINAMATH_CALUDE_rectangle_area_l631_63142

/-- The area of a rectangle with length 47.3 cm and width 24 cm is 1135.2 square centimeters. -/
theorem rectangle_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  length * width = 1135.2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l631_63142


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_range_l631_63159

/-- A function f with an extreme value only at x = 0 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + 2*x^2 + b

/-- f has an extreme value only at x = 0 -/
def has_extreme_only_at_zero (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → ¬(∀ y : ℝ, f a b y ≤ f a b x ∨ ∀ y : ℝ, f a b y ≥ f a b x)

/-- The main theorem: if f has an extreme value only at x = 0, then -8/3 ≤ a ≤ 8/3 -/
theorem extreme_value_implies_a_range (a b : ℝ) :
  has_extreme_only_at_zero a b → -8/3 ≤ a ∧ a ≤ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_range_l631_63159


namespace NUMINAMATH_CALUDE_zoey_finishes_on_friday_l631_63139

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (sets : ℕ) : ℕ :=
  (List.range sets).map (λ i => days_to_read (i + 1)) |>.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_finishes_on_friday :
  let sets := 8
  let start_day := 3  -- Wednesday (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
  day_of_week start_day (total_days sets) = 5  -- Friday
  := by sorry

end NUMINAMATH_CALUDE_zoey_finishes_on_friday_l631_63139


namespace NUMINAMATH_CALUDE_line_slope_angle_l631_63154

theorem line_slope_angle (x y : ℝ) : 
  y - Real.sqrt 3 * x + 5 = 0 → 
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l631_63154


namespace NUMINAMATH_CALUDE_good_games_count_l631_63170

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem good_games_count : 
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l631_63170


namespace NUMINAMATH_CALUDE_max_clock_digit_sum_l631_63137

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  clock_digit_sum h m ≥ clock_digit_sum h' m' ∧
  clock_digit_sum h m = 24 :=
sorry

end NUMINAMATH_CALUDE_max_clock_digit_sum_l631_63137


namespace NUMINAMATH_CALUDE_roberto_chicken_investment_l631_63121

def initial_cost : ℝ := 25 + 30 + 22 + 35
def weekly_feed_cost : ℝ := 1.5 + 1.3 + 1.1 + 0.9
def weekly_egg_production : ℕ := 4 + 3 + 5 + 2
def previous_egg_cost : ℝ := 2

def break_even_weeks : ℕ := 40

theorem roberto_chicken_investment (w : ℕ) :
  w = break_even_weeks ↔ 
  initial_cost + w * weekly_feed_cost = w * previous_egg_cost :=
sorry

end NUMINAMATH_CALUDE_roberto_chicken_investment_l631_63121


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l631_63126

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = a + 3 ∧ 
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → x + 3 = 3 * y + 1 → x + 3 ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l631_63126


namespace NUMINAMATH_CALUDE_find_extremes_l631_63119

/-- Represents the result of a weighing -/
inductive CompareResult
  | Less : CompareResult
  | Equal : CompareResult
  | Greater : CompareResult

/-- Represents a weight -/
structure Weight where
  id : Nat

/-- Represents a weighing operation -/
def weighing (w1 w2 : Weight) : CompareResult := sorry

/-- Represents the set of 5 weights -/
def Weights : Type := Fin 5 → Weight

/-- The heaviest weight in the set -/
def heaviest (ws : Weights) : Weight := sorry

/-- The lightest weight in the set -/
def lightest (ws : Weights) : Weight := sorry

/-- Axiom: Three weights have the same weight -/
axiom three_same_weight (ws : Weights) : 
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    weighing (ws i) (ws j) = CompareResult.Equal ∧ 
    weighing (ws j) (ws k) = CompareResult.Equal

/-- Axiom: One weight is heavier than the three identical weights -/
axiom one_heavier (ws : Weights) : 
  ∃ (h : Fin 5), ∀ (i : Fin 5), 
    weighing (ws i) (ws h) = CompareResult.Less ∨ 
    weighing (ws i) (ws h) = CompareResult.Equal

/-- Axiom: One weight is lighter than the three identical weights -/
axiom one_lighter (ws : Weights) : 
  ∃ (l : Fin 5), ∀ (i : Fin 5), 
    weighing (ws l) (ws i) = CompareResult.Less ∨ 
    weighing (ws l) (ws i) = CompareResult.Equal

/-- Theorem: It's possible to determine the heaviest and lightest weights in at most three weighings -/
theorem find_extremes (ws : Weights) : 
  ∃ (w1 w2 w3 w4 w5 w6 : Weight), 
    (weighing w1 w2 = CompareResult.Less ∨ 
     weighing w1 w2 = CompareResult.Equal ∨ 
     weighing w1 w2 = CompareResult.Greater) ∧
    (weighing w3 w4 = CompareResult.Less ∨ 
     weighing w3 w4 = CompareResult.Equal ∨ 
     weighing w3 w4 = CompareResult.Greater) ∧
    (weighing w5 w6 = CompareResult.Less ∨ 
     weighing w5 w6 = CompareResult.Equal ∨ 
     weighing w5 w6 = CompareResult.Greater) →
    (heaviest ws = heaviest ws ∧ lightest ws = lightest ws) :=
  sorry

end NUMINAMATH_CALUDE_find_extremes_l631_63119


namespace NUMINAMATH_CALUDE_union_complement_A_with_B_l631_63117

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 5}
def B : Set Nat := {1, 3, 5}

theorem union_complement_A_with_B :
  (U \ A) ∪ B = {1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_A_with_B_l631_63117


namespace NUMINAMATH_CALUDE_problem_statement_l631_63187

theorem problem_statement : (2351 - 2250)^2 / 121 = 84 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l631_63187


namespace NUMINAMATH_CALUDE_composite_divisibility_l631_63102

theorem composite_divisibility (n : ℕ) (k : ℕ) 
  (h_composite : ¬ Nat.Prime n)
  (h_n_gt_4 : n > 4)
  (h_k_bounds : 1 ≤ k ∧ k ≤ Int.floor (Real.sqrt (n - 1 : ℝ))) :
  (k * n) ∣ Nat.factorial (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_composite_divisibility_l631_63102


namespace NUMINAMATH_CALUDE_samantha_score_l631_63197

/-- Calculates the score for a revised AMC 8 contest --/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * correct - incorrect

/-- Proves that Samantha's score is 25 given the problem conditions --/
theorem samantha_score :
  let correct : ℕ := 15
  let incorrect : ℕ := 5
  let unanswered : ℕ := 5
  let total_questions : ℕ := correct + incorrect + unanswered
  total_questions = 25 →
  calculate_score correct incorrect unanswered = 25 := by
  sorry

end NUMINAMATH_CALUDE_samantha_score_l631_63197


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l631_63191

/-- If (a - i) / (1 + i) is a pure imaginary number where a ∈ ℝ, then 3a + 4i is in the first quadrant of the complex plane. -/
theorem complex_number_quadrant (a : ℝ) :
  (((a : ℂ) - I) / (1 + I)).im ≠ 0 ∧ (((a : ℂ) - I) / (1 + I)).re = 0 →
  (3 * a : ℝ) > 0 ∧ 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l631_63191


namespace NUMINAMATH_CALUDE_bank_transfer_result_l631_63104

def initial_balance : ℕ := 27004
def transfer_amount : ℕ := 69

theorem bank_transfer_result :
  initial_balance - transfer_amount = 26935 :=
by sorry

end NUMINAMATH_CALUDE_bank_transfer_result_l631_63104


namespace NUMINAMATH_CALUDE_vector_ab_and_magnitude_l631_63190

/-- Given two points A and B in a 2D Cartesian coordinate system,
    prove that the vector from A to B is (1, 1) and its magnitude is √2. -/
theorem vector_ab_and_magnitude (A B : ℝ × ℝ) : 
  A = (1, 2) → B = (2, 3) → 
  (B.1 - A.1, B.2 - A.2) = (1, 1) ∧ 
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_ab_and_magnitude_l631_63190


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l631_63194

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 48156 ≡ n [ZMOD 17] ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l631_63194


namespace NUMINAMATH_CALUDE_average_monthly_sales_l631_63189

def monthly_sales : List ℝ := [120, 80, 50, 130, 110, 90]
def discount_rate : ℝ := 0.1
def num_months : ℕ := 6

def may_index : ℕ := 4

theorem average_monthly_sales :
  let adjusted_sales := monthly_sales.mapIdx (fun i x => 
    if i = may_index then x / (1 - discount_rate) else x)
  (adjusted_sales.sum / num_months) = 98.70 := by
sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l631_63189


namespace NUMINAMATH_CALUDE_combined_salaries_l631_63184

/-- Given the salary of E and the average salary of five individuals including E,
    calculate the combined salaries of the other four individuals. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) :
  salary_E = 9000 →
  average_salary = 8800 →
  num_individuals = 5 →
  (num_individuals * average_salary) - salary_E = 35000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l631_63184


namespace NUMINAMATH_CALUDE_triangle_properties_l631_63181

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the theorem
theorem triangle_properties (ABC : Triangle) 
  (h1 : Real.cos (ABC.A / 2) = 2 * Real.sqrt 5 / 5)
  (h2 : ABC.b * ABC.c * Real.cos ABC.A = 15)
  (h3 : Real.tan ABC.B = 2) : 
  (1/2 * ABC.b * ABC.c * Real.sin ABC.A = 10) ∧ 
  (ABC.a = 2 * Real.sqrt 5) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l631_63181


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l631_63193

theorem polynomial_product_expansion :
  ∀ x : ℝ, (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l631_63193


namespace NUMINAMATH_CALUDE_odd_integers_between_9_and_39_l631_63195

theorem odd_integers_between_9_and_39 :
  let first_term := 9
  let last_term := 39
  let sum := 384
  let n := (last_term - first_term) / 2 + 1
  n = 16 ∧ sum = n / 2 * (first_term + last_term) := by
sorry

end NUMINAMATH_CALUDE_odd_integers_between_9_and_39_l631_63195


namespace NUMINAMATH_CALUDE_jelly_bean_division_l631_63124

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 36 →
  eaten_amount = 6 →
  num_piles = 3 →
  (initial_amount - eaten_amount) / num_piles = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l631_63124


namespace NUMINAMATH_CALUDE_intersection_formula_l631_63198

/-- Given complex numbers a and b on a circle centered at the origin,
    u is the intersection of tangents at a and b -/
def intersection_of_tangents (a b : ℂ) : ℂ := sorry

/-- a and b lie on a circle centered at the origin -/
def on_circle (a b : ℂ) : Prop := sorry

theorem intersection_formula {a b : ℂ} (h : on_circle a b) :
  intersection_of_tangents a b = 2 * a * b / (a + b) := by sorry

end NUMINAMATH_CALUDE_intersection_formula_l631_63198


namespace NUMINAMATH_CALUDE_square_triangle_count_l631_63115

theorem square_triangle_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h_total_shapes : total_shapes = 35)
  (h_total_edges : total_edges = 120) :
  ∃ (squares triangles : ℕ),
    squares + triangles = total_shapes ∧
    4 * squares + 3 * triangles = total_edges ∧
    squares = 20 ∧
    triangles = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_count_l631_63115


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_2018_l631_63123

theorem sum_of_prime_factors_of_2018 :
  ∀ p q : ℕ, 
  Prime p → Prime q → p * q = 2018 → p + q = 1011 := by
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_2018_l631_63123


namespace NUMINAMATH_CALUDE_strip_length_is_14_l631_63150

/-- Represents a folded rectangular strip of paper -/
structure FoldedStrip :=
  (width : ℝ)
  (ap_length : ℝ)
  (bm_length : ℝ)

/-- Calculates the total length of the folded strip -/
def total_length (strip : FoldedStrip) : ℝ :=
  strip.ap_length + strip.width + strip.bm_length

/-- Theorem: The length of the rectangular strip is 14 cm -/
theorem strip_length_is_14 (strip : FoldedStrip) 
  (h_width : strip.width = 4)
  (h_ap : strip.ap_length = 5)
  (h_bm : strip.bm_length = 5) : 
  total_length strip = 14 :=
by
  sorry

#eval total_length { width := 4, ap_length := 5, bm_length := 5 }

end NUMINAMATH_CALUDE_strip_length_is_14_l631_63150


namespace NUMINAMATH_CALUDE_anoop_joining_time_l631_63127

/-- Proves that Anoop joined after 6 months given the investment conditions -/
theorem anoop_joining_time (arjun_investment anoop_investment : ℕ) 
  (total_months : ℕ) (x : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  arjun_investment * total_months = anoop_investment * (total_months - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_anoop_joining_time_l631_63127


namespace NUMINAMATH_CALUDE_min_odd_counties_for_valid_island_l631_63131

/-- A rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  is_rectangular : Bool
  has_closed_path : Bool
  no_self_intersections : Bool

/-- Predicate to check if an island satisfies all conditions -/
def satisfies_conditions (island : Island) : Prop :=
  island.is_rectangular ∧
  island.has_closed_path ∧
  island.no_self_intersections ∧
  island.counties.length % 2 = 1 ∧
  ∀ c ∈ island.counties, c.has_diagonal_road

theorem min_odd_counties_for_valid_island :
  ∀ island : Island,
    satisfies_conditions island →
    island.counties.length ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_counties_for_valid_island_l631_63131


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_11_l631_63161

theorem smallest_multiple_of_45_and_75_not_11 : 
  (∃ n : ℕ+, n * 45 = 225 ∧ n * 75 = 225) ∧ 
  (¬ ∃ m : ℕ+, m * 11 = 225) ∧
  (∀ k : ℕ+, k < 225 → ¬(∃ p : ℕ+, p * 45 = k ∧ p * 75 = k) ∨ (∃ q : ℕ+, q * 11 = k)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_11_l631_63161


namespace NUMINAMATH_CALUDE_man_speed_on_bridge_l631_63103

/-- The speed of a man crossing a bridge -/
theorem man_speed_on_bridge (bridge_length : ℝ) (crossing_time : ℝ) (h1 : bridge_length = 1250) (h2 : crossing_time = 15) :
  bridge_length / crossing_time * (60 / 1000) = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_on_bridge_l631_63103


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l631_63157

theorem largest_of_three_consecutive_evens (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- a, b, c are consecutive even integers
  a + b + c = 1194 →                               -- their sum is 1194
  c = 400                                          -- the largest (c) is 400
:= by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l631_63157


namespace NUMINAMATH_CALUDE_inequality_solution_l631_63179

theorem inequality_solution (x : ℕ+) : 4 - (x : ℝ) > 1 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l631_63179


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l631_63151

theorem sqrt_sum_problem (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l631_63151


namespace NUMINAMATH_CALUDE_market_price_calculation_l631_63146

/-- Proves that given a reduction in sales tax from 3.5% to 3 1/3% resulting in a
    difference of Rs. 12.99999999999999 in tax amount, the market price of the article is Rs. 7800. -/
theorem market_price_calculation (initial_tax : ℚ) (reduced_tax : ℚ) (tax_difference : ℚ) 
  (h1 : initial_tax = 7/200)  -- 3.5%
  (h2 : reduced_tax = 1/30)   -- 3 1/3%
  (h3 : tax_difference = 12999999999999999/1000000000000000) : -- 12.99999999999999
  ∃ (market_price : ℕ), 
    (initial_tax - reduced_tax) * market_price = tax_difference ∧ 
    market_price = 7800 := by
sorry

end NUMINAMATH_CALUDE_market_price_calculation_l631_63146


namespace NUMINAMATH_CALUDE_cindy_math_problem_l631_63165

theorem cindy_math_problem (x : ℤ) : (x - 7) / 5 = 53 → (x - 5) / 7 = 38 := by
  sorry

end NUMINAMATH_CALUDE_cindy_math_problem_l631_63165


namespace NUMINAMATH_CALUDE_sum_of_x1_and_x2_l631_63132

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem sum_of_x1_and_x2 (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = 101 → f x₂ = 101 → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x1_and_x2_l631_63132
