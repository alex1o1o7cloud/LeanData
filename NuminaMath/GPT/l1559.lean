import Mathlib

namespace NUMINAMATH_GPT_stratified_sample_l1559_155941

theorem stratified_sample 
  (total_households : ℕ) 
  (high_income_households : ℕ) 
  (middle_income_households : ℕ) 
  (low_income_households : ℕ) 
  (sample_size : ℕ)
  (H1 : total_households = 600) 
  (H2 : high_income_households = 150)
  (H3 : middle_income_households = 360)
  (H4 : low_income_households = 90)
  (H5 : sample_size = 100) : 
  (middle_income_households * sample_size / total_households = 60) := 
by 
  sorry

end NUMINAMATH_GPT_stratified_sample_l1559_155941


namespace NUMINAMATH_GPT_least_positive_x_l1559_155931

theorem least_positive_x (x : ℕ) : ((2 * x) ^ 2 + 2 * 41 * 2 * x + 41 ^ 2) % 53 = 0 ↔ x = 6 := 
sorry

end NUMINAMATH_GPT_least_positive_x_l1559_155931


namespace NUMINAMATH_GPT_num_divisors_of_m_cubed_l1559_155924

theorem num_divisors_of_m_cubed (m : ℕ) (h : ∃ p : ℕ, Nat.Prime p ∧ m = p ^ 4) :
    Nat.totient (m ^ 3) = 13 := 
sorry

end NUMINAMATH_GPT_num_divisors_of_m_cubed_l1559_155924


namespace NUMINAMATH_GPT_lice_checks_l1559_155993

theorem lice_checks (t_first t_second t_third t_total t_per_check n_first n_second n_third n_total n_per_check n_kg : ℕ) 
 (h1 : t_first = 19 * t_per_check)
 (h2 : t_second = 20 * t_per_check)
 (h3 : t_third = 25 * t_per_check)
 (h4 : t_total = 3 * 60)
 (h5 : t_per_check = 2)
 (h6 : n_first = t_first / t_per_check)
 (h7 : n_second = t_second / t_per_check)
 (h8 : n_third = t_third / t_per_check)
 (h9 : n_total = (t_total - (t_first + t_second + t_third)) / t_per_check) :
 n_total = 26 :=
sorry

end NUMINAMATH_GPT_lice_checks_l1559_155993


namespace NUMINAMATH_GPT_bricks_required_l1559_155965

   -- Definitions from the conditions
   def courtyard_length_meters : ℝ := 42
   def courtyard_width_meters : ℝ := 22
   def brick_length_cm : ℝ := 16
   def brick_width_cm : ℝ := 10

   -- The Lean statement to prove
   theorem bricks_required : (courtyard_length_meters * courtyard_width_meters * 10000) / (brick_length_cm * brick_width_cm) = 57750 :=
   by 
       sorry
   
end NUMINAMATH_GPT_bricks_required_l1559_155965


namespace NUMINAMATH_GPT_discount_percentage_l1559_155999

theorem discount_percentage (C M A : ℝ) (h1 : M = 1.40 * C) (h2 : A = 1.05 * C) :
    (M - A) / M * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1559_155999


namespace NUMINAMATH_GPT_inequality_proof_l1559_155970

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  1 + x^(n+1) ≥ (2*x)^n / (1 + x)^(n-1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1559_155970


namespace NUMINAMATH_GPT_division_value_l1559_155945

theorem division_value (x : ℝ) (h : 800 / x - 154 = 6) : x = 5 := by
  sorry

end NUMINAMATH_GPT_division_value_l1559_155945


namespace NUMINAMATH_GPT_complex_addition_zero_l1559_155997

theorem complex_addition_zero (a b : ℝ) (i : ℂ) (h1 : (1 + i) * i = a + b * i) (h2 : i * i = -1) : a + b = 0 :=
sorry

end NUMINAMATH_GPT_complex_addition_zero_l1559_155997


namespace NUMINAMATH_GPT_evaluate_fraction_sum_l1559_155961

variable (a b c : ℝ)

theorem evaluate_fraction_sum
  (h : (a / (30 - a)) + (b / (70 - b)) + (c / (80 - c)) = 9) :
  (6 / (30 - a)) + (14 / (70 - b)) + (16 / (80 - c)) = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_sum_l1559_155961


namespace NUMINAMATH_GPT_find_x_l1559_155959

theorem find_x (x : ℝ) (h : x * 1.6 - (2 * 1.4) / 1.3 = 4) : x = 3.846154 :=
sorry

end NUMINAMATH_GPT_find_x_l1559_155959


namespace NUMINAMATH_GPT_toothpicks_for_10_squares_l1559_155949

theorem toothpicks_for_10_squares : (4 + 3 * (10 - 1)) = 31 :=
by 
  sorry

end NUMINAMATH_GPT_toothpicks_for_10_squares_l1559_155949


namespace NUMINAMATH_GPT_football_team_goal_l1559_155936

-- Definitions of the conditions
def L1 : ℤ := -5
def G2 : ℤ := 13
def L3 : ℤ := -(L1 ^ 2)
def G4 : ℚ := - (L3 : ℚ) / 2

def total_yardage : ℚ := L1 + G2 + L3 + G4

-- The statement to be proved
theorem football_team_goal : total_yardage < 30 := by
  -- sorry for now since no proof is needed
  sorry

end NUMINAMATH_GPT_football_team_goal_l1559_155936


namespace NUMINAMATH_GPT_mandy_yoga_time_l1559_155952

-- Define the conditions
def ratio_swimming := 1
def ratio_running := 2
def ratio_gym := 3
def ratio_biking := 5
def ratio_yoga := 4

def time_biking := 30

-- Define the Lean 4 statement to prove
theorem mandy_yoga_time : (time_biking / ratio_biking) * ratio_yoga = 24 :=
by
  sorry

end NUMINAMATH_GPT_mandy_yoga_time_l1559_155952


namespace NUMINAMATH_GPT_words_added_to_removed_ratio_l1559_155996

-- Conditions in the problem
def Yvonnes_words : ℕ := 400
def Jannas_extra_words : ℕ := 150
def words_removed : ℕ := 20
def words_needed : ℕ := 1000 - 930

-- Definitions derived from the conditions
def Jannas_words : ℕ := Yvonnes_words + Jannas_extra_words
def total_words_before_editing : ℕ := Yvonnes_words + Jannas_words
def total_words_after_removal : ℕ := total_words_before_editing - words_removed
def words_added : ℕ := words_needed

-- The theorem we need to prove
theorem words_added_to_removed_ratio :
  (words_added : ℚ) / words_removed = 7 / 2 :=
sorry

end NUMINAMATH_GPT_words_added_to_removed_ratio_l1559_155996


namespace NUMINAMATH_GPT_matthew_and_zac_strawberries_l1559_155954

theorem matthew_and_zac_strawberries (total_strawberries jonathan_and_matthew_strawberries zac_strawberries : ℕ) (h1 : total_strawberries = 550) (h2 : jonathan_and_matthew_strawberries = 350) (h3 : zac_strawberries = 200) : (total_strawberries - (jonathan_and_matthew_strawberries - zac_strawberries) = 400) :=
by { sorry }

end NUMINAMATH_GPT_matthew_and_zac_strawberries_l1559_155954


namespace NUMINAMATH_GPT_star_3_4_equals_8_l1559_155988

def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

theorem star_3_4_equals_8 : star 3 4 = 8 := by
  sorry

end NUMINAMATH_GPT_star_3_4_equals_8_l1559_155988


namespace NUMINAMATH_GPT_simplify_expression_l1559_155969

theorem simplify_expression : (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1559_155969


namespace NUMINAMATH_GPT_factor_expression_l1559_155948

variable (y : ℝ)

theorem factor_expression : 64 - 16 * y ^ 3 = 16 * (2 - y) * (4 + 2 * y + y ^ 2) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l1559_155948


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1559_155971

-- Define the first problem statement and the correct answers
theorem solve_eq1 (x : ℝ) (h : (x - 2) ^ 2 = 169) : x = 15 ∨ x = -11 := 
  by sorry

-- Define the second problem statement and the correct answer
theorem solve_eq2 (x : ℝ) (h : 3 * (x - 3) ^ 3 - 24 = 0) : x = 5 := 
  by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1559_155971


namespace NUMINAMATH_GPT_total_pears_picked_l1559_155984

def mikes_pears : Nat := 8
def jasons_pears : Nat := 7
def freds_apples : Nat := 6

theorem total_pears_picked : (mikes_pears + jasons_pears) = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_picked_l1559_155984


namespace NUMINAMATH_GPT_circle_iff_m_gt_neg_1_over_2_l1559_155907

noncomputable def represents_circle (m: ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + x + y - m = 0) → m > -1/2

theorem circle_iff_m_gt_neg_1_over_2 (m : ℝ) : represents_circle m ↔ m > -1/2 := by
  sorry

end NUMINAMATH_GPT_circle_iff_m_gt_neg_1_over_2_l1559_155907


namespace NUMINAMATH_GPT_math_problem_l1559_155992

theorem math_problem :
  2537 + 240 * 3 / 60 - 347 = 2202 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1559_155992


namespace NUMINAMATH_GPT_percentage_is_40_l1559_155940

variables (num : ℕ) (perc : ℕ)

-- Conditions
def ten_percent_eq_40 : Prop := 10 * num = 400
def certain_percentage_eq_160 : Prop := perc * num = 160 * 100

-- Statement to prove
theorem percentage_is_40 (h1 : ten_percent_eq_40 num) (h2 : certain_percentage_eq_160 num perc) : perc = 40 :=
sorry

end NUMINAMATH_GPT_percentage_is_40_l1559_155940


namespace NUMINAMATH_GPT_part1_part2_l1559_155964

-- Definitions and conditions
def a : ℕ := 60
def b : ℕ := 40
def c : ℕ := 80
def d : ℕ := 20
def n : ℕ := a + b + c + d

-- Given critical value for 99% certainty
def critical_value_99 : ℝ := 6.635

-- Calculate K^2 using the given formula
noncomputable def K_squared : ℝ := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculation of probability of selecting 2 qualified products from 5 before renovation
def total_sampled : ℕ := 5
def qualified_before_renovation : ℕ := 3
def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def prob_selecting_2_qualified : ℚ := (total_combinations qualified_before_renovation 2 : ℚ) / 
                                      (total_combinations total_sampled 2 : ℚ)

-- Proof statements
theorem part1 : K_squared > critical_value_99 := by
  sorry

theorem part2 : prob_selecting_2_qualified = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1559_155964


namespace NUMINAMATH_GPT_ratio_of_canoes_to_kayaks_l1559_155989

theorem ratio_of_canoes_to_kayaks 
    (canoe_cost kayak_cost total_revenue : ℕ) 
    (canoe_to_kayak_ratio extra_canoes : ℕ)
    (h1 : canoe_cost = 14)
    (h2 : kayak_cost = 15)
    (h3 : total_revenue = 288)
    (h4 : extra_canoes = 4)
    (h5 : canoe_to_kayak_ratio = 3) 
    (c k : ℕ)
    (h6 : c = k + extra_canoes)
    (h7 : c = canoe_to_kayak_ratio * k)
    (h8 : canoe_cost * c + kayak_cost * k = total_revenue) :
    c / k = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_canoes_to_kayaks_l1559_155989


namespace NUMINAMATH_GPT_least_boxes_l1559_155934
-- Definitions and conditions
def isPerfectCube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

def isFactor (a b : ℕ) : Prop := ∃ k, a * k = b

def numBoxes (N boxSize : ℕ) : ℕ := N / boxSize

-- Specific conditions for our problem
theorem least_boxes (N : ℕ) (boxSize : ℕ) 
  (h1 : N ≠ 0) 
  (h2 : isPerfectCube N)
  (h3 : isFactor boxSize N)
  (h4 : boxSize = 45): 
  numBoxes N boxSize = 75 :=
by
  sorry

end NUMINAMATH_GPT_least_boxes_l1559_155934


namespace NUMINAMATH_GPT_min_value_of_b_minus_2c_plus_1_over_a_l1559_155982

theorem min_value_of_b_minus_2c_plus_1_over_a
  (a b c : ℝ)
  (h₁ : (a ≠ 0))
  (h₂ : ∀ x, -1 < x ∧ x < 3 → ax^2 + bx + c < 0) :
  b - 2 * c + (1 / a) = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_b_minus_2c_plus_1_over_a_l1559_155982


namespace NUMINAMATH_GPT_measure_of_angle_E_l1559_155983

theorem measure_of_angle_E
    (A B C D E F : ℝ)
    (h1 : A = B)
    (h2 : B = C)
    (h3 : C = D)
    (h4 : E = F)
    (h5 : A = E - 30)
    (h6 : A + B + C + D + E + F = 720) :
  E = 140 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_measure_of_angle_E_l1559_155983


namespace NUMINAMATH_GPT_tim_score_in_math_l1559_155937

def even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def sum_even_numbers (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem tim_score_in_math : sum_even_numbers even_numbers = 56 := by
  -- Proof steps would be here
  sorry

end NUMINAMATH_GPT_tim_score_in_math_l1559_155937


namespace NUMINAMATH_GPT_inequality_holds_l1559_155981

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x * y + y * z + z * x = 1) :
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1559_155981


namespace NUMINAMATH_GPT_determine_n_l1559_155929

noncomputable def average_value (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1) : ℚ) / (6 * (n * (n + 1) / 2))

theorem determine_n :
  ∃ n : ℕ, average_value n = 2020 ∧ n = 3029 :=
sorry

end NUMINAMATH_GPT_determine_n_l1559_155929


namespace NUMINAMATH_GPT_min_value_of_angle_function_l1559_155947

theorem min_value_of_angle_function (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : 0 < α) (h3 : α < Real.pi) :
  ∃ α, α = (2 * Real.pi / 3) ∧ (4 / α + 1 / (Real.pi - α)) = (9 / Real.pi) := by
  sorry

end NUMINAMATH_GPT_min_value_of_angle_function_l1559_155947


namespace NUMINAMATH_GPT_calculate_expression_l1559_155950

theorem calculate_expression : 6^3 - 5 * 7 + 2^4 = 197 := 
by
  -- Generally, we would provide the proof here, but it's not required.
  sorry

end NUMINAMATH_GPT_calculate_expression_l1559_155950


namespace NUMINAMATH_GPT_students_taking_only_science_l1559_155913

theorem students_taking_only_science (total_students : ℕ) (students_science : ℕ) (students_math : ℕ)
  (h1 : total_students = 120) (h2 : students_science = 80) (h3 : students_math = 75) :
  (students_science - (students_science + students_math - total_students)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_only_science_l1559_155913


namespace NUMINAMATH_GPT_polynomial_abs_sum_l1559_155978

theorem polynomial_abs_sum (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  (1 - (2:ℝ) * x) ^ 8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = (3:ℝ) ^ 8 :=
sorry

end NUMINAMATH_GPT_polynomial_abs_sum_l1559_155978


namespace NUMINAMATH_GPT_series_sum_equals_four_l1559_155923

/-- 
  Proof of the sum of the series: 
  ∑ (n=1 to ∞) (6n² - n + 1) / (n⁵ - n⁴ + n³ - n² + n) = 4 
--/
theorem series_sum_equals_four :
  (∑' n : ℕ, (if n > 0 then (6 * n^2 - n + 1 : ℝ) / (n^5 - n^4 + n^3 - n^2 + n) else 0)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_equals_four_l1559_155923


namespace NUMINAMATH_GPT_kevin_age_l1559_155918

theorem kevin_age (x : ℕ) :
  (∃ n : ℕ, x - 2 = n^2) ∧ (∃ m : ℕ, x + 2 = m^3) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_kevin_age_l1559_155918


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1559_155919

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1559_155919


namespace NUMINAMATH_GPT_less_sum_mult_l1559_155922

theorem less_sum_mult {a b : ℝ} (h1 : a < 1) (h2 : b > 1) : a * b < a + b :=
sorry

end NUMINAMATH_GPT_less_sum_mult_l1559_155922


namespace NUMINAMATH_GPT_pond_volume_extraction_l1559_155955

/--
  Let length (l), width (w), and depth (h) be dimensions of a pond.
  Given:
  l = 20,
  w = 10,
  h = 5,
  Prove that the volume of the soil extracted from the pond is 1000 cubic meters.
-/
theorem pond_volume_extraction (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
  by
    sorry

end NUMINAMATH_GPT_pond_volume_extraction_l1559_155955


namespace NUMINAMATH_GPT_solve_for_y_l1559_155946

theorem solve_for_y (x y : ℝ) (h : 2 * x - 3 * y = 4) : y = (2 * x - 4) / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1559_155946


namespace NUMINAMATH_GPT_january_first_is_tuesday_l1559_155979

-- Define the days of the week for convenience
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the problem conditions
def daysInJanuary : Nat := 31
def weeksInJanuary : Nat := daysInJanuary / 7   -- This is 4 weeks
def extraDays : Nat := daysInJanuary % 7         -- This leaves 3 extra days

-- Define the problem as proving January 1st is a Tuesday
theorem january_first_is_tuesday (fridaysInJanuary : Nat) (mondaysInJanuary : Nat)
    (h_friday : fridaysInJanuary = 4) (h_monday: mondaysInJanuary = 4) : Weekday :=
  -- Avoid specific proof steps from the solution; assume conditions and directly prove the result
  sorry

end NUMINAMATH_GPT_january_first_is_tuesday_l1559_155979


namespace NUMINAMATH_GPT_intersection_point_of_given_lines_l1559_155926

theorem intersection_point_of_given_lines :
  ∃ (x y : ℚ), 2 * y = -x + 3 ∧ -y = 5 * x + 1 ∧ x = -5 / 9 ∧ y = 16 / 9 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_given_lines_l1559_155926


namespace NUMINAMATH_GPT_solution_mix_percentage_l1559_155915

theorem solution_mix_percentage
  (x y z : ℝ)
  (hx1 : x + y + z = 100)
  (hx2 : 0.40 * x + 0.50 * y + 0.30 * z = 46)
  (hx3 : z = 100 - x - y) :
  x = 40 ∧ y = 60 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_mix_percentage_l1559_155915


namespace NUMINAMATH_GPT_solve_length_BF_l1559_155921

-- Define the problem conditions
def rectangular_paper (short_side long_side : ℝ) : Prop :=
  short_side = 12 ∧ long_side > short_side

def vertex_touch_midpoint (vmp mid : ℝ) : Prop :=
  vmp = mid / 2

def congruent_triangles (triangle1 triangle2 : ℝ) : Prop :=
  triangle1 = triangle2

-- Theorem to prove the length of BF
theorem solve_length_BF (short_side long_side vmp mid triangle1 triangle2 : ℝ) 
  (h1 : rectangular_paper short_side long_side)
  (h2 : vertex_touch_midpoint vmp mid)
  (h3 : congruent_triangles triangle1 triangle2) :
  -- The length of BF is 10
  mid = 6 → 18 - 6 = 12 + 6 - 10 → 10 = 12 - (18 - 10) → vmp = 6 → 6 * 2 = 12 →
  sorry :=
sorry

end NUMINAMATH_GPT_solve_length_BF_l1559_155921


namespace NUMINAMATH_GPT_min_project_time_l1559_155962

theorem min_project_time (A B C : ℝ) (D : ℝ := 12) :
  (1 / B + 1 / C) = 1 / 2 →
  (1 / A + 1 / C) = 1 / 3 →
  (1 / A + 1 / B) = 1 / 4 →
  (1 / D) = 1 / 12 →
  ∃ x : ℝ, x = 8 / 5 ∧ 1 / x = 1 / A + 1 / B + 1 / C + 1 / (12:ℝ) :=
by
  intros h1 h2 h3 h4
  -- Combination of given hypotheses to prove the goal
  sorry

end NUMINAMATH_GPT_min_project_time_l1559_155962


namespace NUMINAMATH_GPT_probability_nan_kai_l1559_155903

theorem probability_nan_kai :
  let total_outcomes := Nat.choose 6 4
  let successful_outcomes := Nat.choose 4 4
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_nan_kai_l1559_155903


namespace NUMINAMATH_GPT_binary_to_base5_l1559_155917

theorem binary_to_base5 : Nat.digits 5 (Nat.ofDigits 2 [1, 0, 1, 1, 0, 0, 1]) = [4, 2, 3] :=
by
  sorry

end NUMINAMATH_GPT_binary_to_base5_l1559_155917


namespace NUMINAMATH_GPT_car_rental_cost_l1559_155930

def day1_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day2_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day3_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def total_cost (day1 : ℝ) (day2 : ℝ) (day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem car_rental_cost :
  let day1_base_rate := 150
  let day2_base_rate := 100
  let day3_base_rate := 75
  let day1_miles_driven := 620
  let day2_miles_driven := 744
  let day3_miles_driven := 510
  let day1_cost_per_mile := 0.50
  let day2_cost_per_mile := 0.40
  let day3_cost_per_mile := 0.30
  day1_cost day1_base_rate day1_miles_driven day1_cost_per_mile +
  day2_cost day2_base_rate day2_miles_driven day2_cost_per_mile +
  day3_cost day3_base_rate day3_miles_driven day3_cost_per_mile = 1085.60 :=
by
  let day1 := day1_cost 150 620 0.50
  let day2 := day2_cost 100 744 0.40
  let day3 := day3_cost 75 510 0.30
  let total := total_cost day1 day2 day3
  show total = 1085.60
  sorry

end NUMINAMATH_GPT_car_rental_cost_l1559_155930


namespace NUMINAMATH_GPT_weight_differences_correct_l1559_155944

-- Define the weights of Heather, Emily, Elizabeth, and Emma
def H : ℕ := 87
def E1 : ℕ := 58
def E2 : ℕ := 56
def E3 : ℕ := 64

-- Proof problem statement
theorem weight_differences_correct :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) :=
by
  -- Note: 'sorry' is used to skip the proof itself
  sorry

end NUMINAMATH_GPT_weight_differences_correct_l1559_155944


namespace NUMINAMATH_GPT_loss_percentage_is_10_l1559_155957

-- Define the conditions
def cost_price (CP : ℝ) : Prop :=
  (550 : ℝ) = 1.1 * CP

def selling_price (SP : ℝ) : Prop :=
  SP = 450

-- Define the main proof statement
theorem loss_percentage_is_10 (CP SP : ℝ) (HCP : cost_price CP) (HSP : selling_price SP) :
  ((CP - SP) / CP) * 100 = 10 :=
by
  -- Translation of the condition into Lean statement
  sorry

end NUMINAMATH_GPT_loss_percentage_is_10_l1559_155957


namespace NUMINAMATH_GPT_master_wang_resting_on_sunday_again_l1559_155951

theorem master_wang_resting_on_sunday_again (n : ℕ) 
  (works_days := 8) 
  (rest_days := 2) 
  (week_days := 7) 
  (cycle_days := works_days + rest_days) 
  (initial_rest_saturday_sunday : Prop) : 
  (initial_rest_saturday_sunday → ∃ n : ℕ, (week_days * n) % cycle_days = rest_days) → 
  (∃ n : ℕ, n = 7) :=
by
  sorry

end NUMINAMATH_GPT_master_wang_resting_on_sunday_again_l1559_155951


namespace NUMINAMATH_GPT_tan_add_pi_over_4_l1559_155914

theorem tan_add_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_4_l1559_155914


namespace NUMINAMATH_GPT_correct_quotient_is_243_l1559_155968

-- Define the given conditions
def mistaken_divisor : ℕ := 121
def mistaken_quotient : ℕ := 432
def correct_divisor : ℕ := 215
def remainder : ℕ := 0

-- Calculate the dividend based on mistaken values
def dividend : ℕ := mistaken_divisor * mistaken_quotient + remainder

-- State the theorem for the correct quotient
theorem correct_quotient_is_243
  (h_dividend : dividend = mistaken_divisor * mistaken_quotient + remainder)
  (h_divisible : dividend % correct_divisor = remainder) :
  dividend / correct_divisor = 243 :=
sorry

end NUMINAMATH_GPT_correct_quotient_is_243_l1559_155968


namespace NUMINAMATH_GPT_solve_w_from_system_of_equations_l1559_155912

open Real

variables (w x y z : ℝ)

theorem solve_w_from_system_of_equations
  (h1 : 2 * w + x + y + z = 1)
  (h2 : w + 2 * x + y + z = 2)
  (h3 : w + x + 2 * y + z = 2)
  (h4 : w + x + y + 2 * z = 1) :
  w = -1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_w_from_system_of_equations_l1559_155912


namespace NUMINAMATH_GPT_range_of_a_l1559_155975

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) :
  a ∈ Set.Icc (4 / 5) 8 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1559_155975


namespace NUMINAMATH_GPT_find_t_l1559_155958

theorem find_t (t a b : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60) →
  t = -9 :=
by
  intros h
  -- We'll skip the proof part
  sorry

end NUMINAMATH_GPT_find_t_l1559_155958


namespace NUMINAMATH_GPT_area_comparison_l1559_155927

namespace Quadrilaterals

open Real

-- Define the vertices of both quadrilaterals
def quadrilateral_I_vertices : List (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 1)]
def quadrilateral_II_vertices : List (ℝ × ℝ) := [(0, 0), (3, 0), (3, 1), (0, 2)]

-- Area calculation function (example function for clarity)
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- This would use the actual geometry to compute the area
  2.5 -- placeholder for the area of quadrilateral I
  -- 4.5 -- placeholder for the area of quadrilateral II

theorem area_comparison :
  (area_of_quadrilateral quadrilateral_I_vertices) < (area_of_quadrilateral quadrilateral_II_vertices) :=
  sorry

end Quadrilaterals

end NUMINAMATH_GPT_area_comparison_l1559_155927


namespace NUMINAMATH_GPT_puppy_ratios_l1559_155933

theorem puppy_ratios :
  ∀(total_puppies : ℕ)(golden_retriever_females golden_retriever_males : ℕ)
   (labrador_females labrador_males : ℕ)(poodle_females poodle_males : ℕ)
   (beagle_females beagle_males : ℕ),
  total_puppies = golden_retriever_females + golden_retriever_males +
                  labrador_females + labrador_males +
                  poodle_females + poodle_males +
                  beagle_females + beagle_males →
  golden_retriever_females = 2 →
  golden_retriever_males = 4 →
  labrador_females = 1 →
  labrador_males = 3 →
  poodle_females = 3 →
  poodle_males = 2 →
  beagle_females = 1 →
  beagle_males = 2 →
  (golden_retriever_females / golden_retriever_males = 1 / 2) ∧
  (labrador_females / labrador_males = 1 / 3) ∧
  (poodle_females / poodle_males = 3 / 2) ∧
  (beagle_females / beagle_males = 1 / 2) ∧
  (7 / 11 = (golden_retriever_females + labrador_females + poodle_females + beagle_females) / 
            (golden_retriever_males + labrador_males + poodle_males + beagle_males)) :=
by intros;
   sorry

end NUMINAMATH_GPT_puppy_ratios_l1559_155933


namespace NUMINAMATH_GPT_complement_intersection_l1559_155942

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2 * x > 0}

-- Define complement of A in U
def C_U_A : Set ℝ := U \ A

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_intersection (x : ℝ) : x ∈ C_U_A ∩ B ↔ 1 < x ∧ x ≤ 2 :=
by
   sorry

end NUMINAMATH_GPT_complement_intersection_l1559_155942


namespace NUMINAMATH_GPT_f_cos_x_l1559_155980

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (hx : f (Real.sin x) = 2 - Real.cos (2 * x)) :
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
sorry

end NUMINAMATH_GPT_f_cos_x_l1559_155980


namespace NUMINAMATH_GPT_term_in_sequence_l1559_155904

   theorem term_in_sequence (n : ℕ) (h1 : 1 ≤ n) (h2 : 6 * n + 1 = 2005) : n = 334 :=
   by
     sorry
   
end NUMINAMATH_GPT_term_in_sequence_l1559_155904


namespace NUMINAMATH_GPT_sum_of_roots_eq_p_l1559_155910

variable (p q : ℝ)
variable (hq : q = p^2 - 1)

theorem sum_of_roots_eq_p (h : q = p^2 - 1) : 
  let r1 := p
  let r2 := q
  r1 + r2 = p := 
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_p_l1559_155910


namespace NUMINAMATH_GPT_range_of_expression_l1559_155976

theorem range_of_expression (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3 / 2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l1559_155976


namespace NUMINAMATH_GPT_work_completion_l1559_155925

theorem work_completion (p q : ℝ) (h1 : p = 1.60 * q) (h2 : (1 / p + 1 / q) = 1 / 16) : p = 1 / 26 := 
by {
  -- This will be followed by the proof steps, but we add sorry since only the statement is required
  sorry
}

end NUMINAMATH_GPT_work_completion_l1559_155925


namespace NUMINAMATH_GPT_subtract_two_percent_is_multiplying_l1559_155905

theorem subtract_two_percent_is_multiplying (a : ℝ) : (a - 0.02 * a) = 0.98 * a := by
  sorry

end NUMINAMATH_GPT_subtract_two_percent_is_multiplying_l1559_155905


namespace NUMINAMATH_GPT_maximize_revenue_at_175_l1559_155987

def price (x : ℕ) : ℕ :=
  if x ≤ 150 then 200 else 200 - (x - 150)

def revenue (x : ℕ) : ℕ :=
  price x * x

theorem maximize_revenue_at_175 :
  ∀ x : ℕ, revenue 175 ≥ revenue x := 
sorry

end NUMINAMATH_GPT_maximize_revenue_at_175_l1559_155987


namespace NUMINAMATH_GPT_largest_integer_solving_inequality_l1559_155973

theorem largest_integer_solving_inequality :
  ∃ (x : ℤ), (7 - 5 * x > 22) ∧ ∀ (y : ℤ), (7 - 5 * y > 22) → x ≥ y ∧ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_solving_inequality_l1559_155973


namespace NUMINAMATH_GPT_determine_x_l1559_155977

theorem determine_x (x : ℚ) (n : ℤ) (d : ℚ) 
  (h_cond : x = n + d)
  (h_floor : n = ⌊x⌋)
  (h_d : 0 ≤ d ∧ d < 1)
  (h_eq : ⌊x⌋ + x = 17 / 4) :
  x = 9 / 4 := sorry

end NUMINAMATH_GPT_determine_x_l1559_155977


namespace NUMINAMATH_GPT_calc_expression_l1559_155943

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1559_155943


namespace NUMINAMATH_GPT_product_of_469111_and_9999_l1559_155985

theorem product_of_469111_and_9999 : 469111 * 9999 = 4690418889 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_469111_and_9999_l1559_155985


namespace NUMINAMATH_GPT_Sarah_substitution_l1559_155902

theorem Sarah_substitution :
  ∀ (f g h i j : ℤ), 
    f = 2 → g = 4 → h = 5 → i = 10 →
    (f - (g - (h * (i - j))) = 48 - 5 * j) →
    (f - g - h * i - j = -52 - j) →
    j = 25 :=
by
  intros f g h i j hfg hi hhi hmf hCm hRn
  sorry

end NUMINAMATH_GPT_Sarah_substitution_l1559_155902


namespace NUMINAMATH_GPT_fill_tank_without_leak_l1559_155994

theorem fill_tank_without_leak (T : ℕ) : 
  (1 / T - 1 / 110 = 1 / 11) ↔ T = 10 :=
by 
  sorry

end NUMINAMATH_GPT_fill_tank_without_leak_l1559_155994


namespace NUMINAMATH_GPT_number_of_raccoons_l1559_155909

/-- Jason pepper-sprays some raccoons and 6 times as many squirrels. 
Given that he pepper-sprays a total of 84 animals, the number of raccoons he pepper-sprays is 12. -/
theorem number_of_raccoons (R : Nat) (h1 : 84 = R + 6 * R) : R = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_raccoons_l1559_155909


namespace NUMINAMATH_GPT_raman_salary_loss_l1559_155963

theorem raman_salary_loss : 
  ∀ (S : ℝ), S > 0 →
  let decreased_salary := S - (0.5 * S) 
  let final_salary := decreased_salary + (0.5 * decreased_salary) 
  let loss := S - final_salary 
  let percentage_loss := (loss / S) * 100
  percentage_loss = 25 := 
by
  intros S hS
  let decreased_salary := S - (0.5 * S)
  let final_salary := decreased_salary + (0.5 * decreased_salary)
  let loss := S - final_salary
  let percentage_loss := (loss / S) * 100
  have h1 : decreased_salary = 0.5 * S := by sorry
  have h2 : final_salary = 0.75 * S := by sorry
  have h3 : loss = 0.25 * S := by sorry
  have h4 : percentage_loss = 25 := by sorry
  exact h4

end NUMINAMATH_GPT_raman_salary_loss_l1559_155963


namespace NUMINAMATH_GPT_smallest_n_exceeds_15_l1559_155920

noncomputable def g (n : ℕ) : ℕ :=
  sorry  -- Define the sum of the digits of 1 / 3^n to the right of the decimal point

theorem smallest_n_exceeds_15 : ∃ n : ℕ, n > 0 ∧ g n > 15 ∧ ∀ m : ℕ, m > 0 ∧ g m > 15 → n ≤ m :=
  sorry  -- Prove the smallest n such that g(n) > 15

end NUMINAMATH_GPT_smallest_n_exceeds_15_l1559_155920


namespace NUMINAMATH_GPT_speed_in_terms_of_time_l1559_155938

variable (a b x : ℝ)

-- Conditions
def condition1 : Prop := 1000 = a * x
def condition2 : Prop := 833 = b * x

-- The theorem to prove
theorem speed_in_terms_of_time (h1 : condition1 a x) (h2 : condition2 b x) :
  a = 1000 / x ∧ b = 833 / x :=
by
  sorry

end NUMINAMATH_GPT_speed_in_terms_of_time_l1559_155938


namespace NUMINAMATH_GPT_jenny_coins_value_l1559_155916

theorem jenny_coins_value (n d : ℕ) (h1 : d = 30 - n) (h2 : 150 + 5 * n = 300 - 5 * n + 120) :
  (300 - 5 * n : ℚ) / 100 = 1.65 := 
by
  sorry

end NUMINAMATH_GPT_jenny_coins_value_l1559_155916


namespace NUMINAMATH_GPT_unique_real_root_count_l1559_155953

theorem unique_real_root_count :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 := by
  sorry

end NUMINAMATH_GPT_unique_real_root_count_l1559_155953


namespace NUMINAMATH_GPT_distance_between_islands_l1559_155990

theorem distance_between_islands (AB : ℝ) (angle_BAC angle_ABC : ℝ) : 
  AB = 20 ∧ angle_BAC = 60 ∧ angle_ABC = 75 → 
  (∃ BC : ℝ, BC = 10 * Real.sqrt 6) := by
  intro h
  sorry

end NUMINAMATH_GPT_distance_between_islands_l1559_155990


namespace NUMINAMATH_GPT_scientific_notation_of_50000_l1559_155966

theorem scientific_notation_of_50000 :
  50000 = 5 * 10^4 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_50000_l1559_155966


namespace NUMINAMATH_GPT_cards_difference_l1559_155901

theorem cards_difference
  (H : ℕ)
  (F : ℕ)
  (B : ℕ)
  (hH : H = 200)
  (hF : F = 4 * H)
  (hTotal : B + F + H = 1750) :
  F - B = 50 :=
by
  sorry

end NUMINAMATH_GPT_cards_difference_l1559_155901


namespace NUMINAMATH_GPT_greatest_product_from_sum_2004_l1559_155995

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_from_sum_2004_l1559_155995


namespace NUMINAMATH_GPT_no_x_intersections_geometric_sequence_l1559_155967

theorem no_x_intersections_geometric_sequence (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a * c > 0) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) = false :=
by
  sorry

end NUMINAMATH_GPT_no_x_intersections_geometric_sequence_l1559_155967


namespace NUMINAMATH_GPT_percentage_decrease_correct_l1559_155932

theorem percentage_decrease_correct :
  ∀ (p : ℝ), (1 + 0.25) * (1 - p) = 1 → p = 0.20 :=
by
  intro p
  intro h
  sorry

end NUMINAMATH_GPT_percentage_decrease_correct_l1559_155932


namespace NUMINAMATH_GPT_eight_pow_15_div_sixtyfour_pow_6_l1559_155908

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_GPT_eight_pow_15_div_sixtyfour_pow_6_l1559_155908


namespace NUMINAMATH_GPT_monotonicity_and_k_range_l1559_155998

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1 / 2 : ℝ) * x^2 - x

theorem monotonicity_and_k_range :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) ↔ k ∈ Set.Iic (-2) := sorry

end NUMINAMATH_GPT_monotonicity_and_k_range_l1559_155998


namespace NUMINAMATH_GPT_general_term_formula_l1559_155906

theorem general_term_formula (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → (n+1) * a (n+1) - n * a n^2 + (n+1) * a n * a (n+1) - n * a n = 0) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l1559_155906


namespace NUMINAMATH_GPT_converted_land_eqn_l1559_155974

theorem converted_land_eqn (forest_land dry_land converted_dry_land : ℝ)
  (h1 : forest_land = 108)
  (h2 : dry_land = 54)
  (h3 : converted_dry_land = x) :
  (dry_land - converted_dry_land = 0.2 * (forest_land + converted_dry_land)) :=
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_converted_land_eqn_l1559_155974


namespace NUMINAMATH_GPT_factor_expression_l1559_155935

theorem factor_expression (x : ℝ) : (45 * x^3 - 135 * x^7) = 45 * x^3 * (1 - 3 * x^4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1559_155935


namespace NUMINAMATH_GPT_range_of_m_l1559_155928

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) →
  (1 < m) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1559_155928


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1559_155960

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1559_155960


namespace NUMINAMATH_GPT_complement_of_M_is_34_l1559_155956

open Set

noncomputable def U : Set ℝ := univ
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}
def complement_M (U : Set ℝ) (M : Set ℝ) : Set ℝ := U \ M

theorem complement_of_M_is_34 : complement_M U M = {x | 3 ≤ x ∧ x ≤ 4} := 
by sorry

end NUMINAMATH_GPT_complement_of_M_is_34_l1559_155956


namespace NUMINAMATH_GPT_hypotenuse_square_l1559_155972

theorem hypotenuse_square (a : ℕ) : (a + 1)^2 + a^2 = 2 * a^2 + 2 * a + 1 := 
by sorry

end NUMINAMATH_GPT_hypotenuse_square_l1559_155972


namespace NUMINAMATH_GPT_crafts_club_necklaces_l1559_155939

theorem crafts_club_necklaces (members : ℕ) (total_beads : ℕ) (beads_per_necklace : ℕ)
  (h1 : members = 9) (h2 : total_beads = 900) (h3 : beads_per_necklace = 50) :
  (total_beads / beads_per_necklace) / members = 2 :=
by
  sorry

end NUMINAMATH_GPT_crafts_club_necklaces_l1559_155939


namespace NUMINAMATH_GPT_radius_formula_l1559_155911

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  let angle := 42 * Real.pi / 180 -- converting 42 degrees to radians
  let R := a / (Real.sqrt 3)
  let h := R * Real.tan angle
  Real.sqrt ((R * R) + (h * h))

theorem radius_formula (a : ℝ) : radius_of_circumscribed_sphere a = (a * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_formula_l1559_155911


namespace NUMINAMATH_GPT_units_digit_p2_plus_3p_l1559_155986

-- Define p
def p : ℕ := 2017^3 + 3^2017

-- Define the theorem to be proved
theorem units_digit_p2_plus_3p : (p^2 + 3^p) % 10 = 5 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_units_digit_p2_plus_3p_l1559_155986


namespace NUMINAMATH_GPT_compute_sin_product_l1559_155900

theorem compute_sin_product : 
  (1 - Real.sin (Real.pi / 12)) *
  (1 - Real.sin (5 * Real.pi / 12)) *
  (1 - Real.sin (7 * Real.pi / 12)) *
  (1 - Real.sin (11 * Real.pi / 12)) = 
  (1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_compute_sin_product_l1559_155900


namespace NUMINAMATH_GPT_find_m_eq_4_l1559_155991

theorem find_m_eq_4 (m : ℝ) (h₁ : ∃ (A B C : ℝ × ℝ), A = (m, -m+3) ∧ B = (2, m-1) ∧ C = (-1, 4)) (h₂ : (4 - (-m+3)) / (-1-m) = 3 * ((m-1) - 4) / (2 - (-1))) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_eq_4_l1559_155991
