import Mathlib

namespace vasya_can_guess_number_in_10_questions_l1873_187309

noncomputable def log2 (n : ℕ) : ℝ := 
  Real.log n / Real.log 2

theorem vasya_can_guess_number_in_10_questions (n q : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 1000) (h3 : q = 10) :
  q ≥ log2 n := 
by
  sorry

end vasya_can_guess_number_in_10_questions_l1873_187309


namespace approx_num_chars_in_ten_thousand_units_l1873_187395

-- Define the number of characters in the book
def num_chars : ℕ := 731017

-- Define the conversion factor from characters to units of 'ten thousand'
def ten_thousand : ℕ := 10000

-- Define the number of characters in units of 'ten thousand'
def chars_in_ten_thousand_units : ℚ := num_chars / ten_thousand

-- Define the rounded number of units to the nearest whole number
def rounded_chars_in_ten_thousand_units : ℤ := round chars_in_ten_thousand_units

-- Theorem to state the approximate number of characters in units of 'ten thousand' is 73
theorem approx_num_chars_in_ten_thousand_units : rounded_chars_in_ten_thousand_units = 73 := 
by sorry

end approx_num_chars_in_ten_thousand_units_l1873_187395


namespace cyclic_quadrilaterals_count_l1873_187353

theorem cyclic_quadrilaterals_count :
  ∃ n : ℕ, n = 568 ∧
  ∀ (a b c d : ℕ), 
    a + b + c + d = 32 ∧
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c > d) ∧ (b + c + d > a) ∧ (c + d + a > b) ∧ (d + a + b > c) ∧
    (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2
      → n = 568 := 
sorry

end cyclic_quadrilaterals_count_l1873_187353


namespace simplify_complex_expression_l1873_187390

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l1873_187390


namespace max_value_l1873_187306

def a_n (n : ℕ) : ℤ := -2 * (n : ℤ)^2 + 29 * (n : ℤ) + 3

theorem max_value : ∃ n : ℕ, a_n n = 108 ∧ ∀ m : ℕ, a_n m ≤ 108 := by
  sorry

end max_value_l1873_187306


namespace range_of_m_intersection_l1873_187365

theorem range_of_m_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, y - k * x - 1 = 0 ∧ (x^2 / 4) + (y^2 / m) = 1) ↔ (m ∈ Set.Ico 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end range_of_m_intersection_l1873_187365


namespace ratio_of_boys_to_girls_l1873_187360

theorem ratio_of_boys_to_girls (total_students : ℕ) (girls : ℕ) (boys : ℕ)
  (h_total : total_students = 1040)
  (h_girls : girls = 400)
  (h_boys : boys = total_students - girls) :
  (boys / Nat.gcd boys girls = 8) ∧ (girls / Nat.gcd boys girls = 5) :=
sorry

end ratio_of_boys_to_girls_l1873_187360


namespace intersection_A_B_l1873_187369

open Set

def U := ℝ
def A := { x : ℝ | (2 * x + 3) / (x - 2) > 0 }
def B := { x : ℝ | abs (x - 1) < 2 }

theorem intersection_A_B : (A ∩ B) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l1873_187369


namespace tenth_day_of_month_is_monday_l1873_187378

def total_run_minutes_in_month (hours : ℕ) : ℕ := hours * 60

def run_minutes_per_week (runs_per_week : ℕ) (minutes_per_run : ℕ) : ℕ := 
  runs_per_week * minutes_per_run

def weeks_in_month (total_minutes : ℕ) (minutes_per_week : ℕ) : ℕ := 
  total_minutes / minutes_per_week

def identify_day_of_week (first_day : ℕ) (target_day : ℕ) : ℕ := 
  (first_day + target_day - 1) % 7

theorem tenth_day_of_month_is_monday :
  let hours := 5
  let runs_per_week := 3
  let minutes_per_run := 20
  let first_day := 6 -- Assuming 0=Sunday, ..., 6=Saturday
  let target_day := 10
  total_run_minutes_in_month hours = 300 ∧
  run_minutes_per_week runs_per_week minutes_per_run = 60 ∧
  weeks_in_month 300 60 = 5 ∧
  identify_day_of_week first_day target_day = 1 := -- 1 represents Monday
sorry

end tenth_day_of_month_is_monday_l1873_187378


namespace inequality_solution_l1873_187387

theorem inequality_solution (x : ℝ) : 
  (x^2 + 4 * x + 13 > 0) -> ((x - 4) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ≥ 4) :=
by
  intro h_pos
  sorry

end inequality_solution_l1873_187387


namespace selection_count_l1873_187311

theorem selection_count (word : String) (vowels : Finset Char) (consonants : Finset Char)
  (hword : word = "УЧЕБНИК")
  (hvowels : vowels = {'У', 'Е', 'И'})
  (hconsonants : consonants = {'Ч', 'Б', 'Н', 'К'})
  :
  vowels.card * consonants.card = 12 :=
by {
  sorry
}

end selection_count_l1873_187311


namespace intersection_M_N_l1873_187394

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_M_N : M ∩ N = { (1, 0) } := by
  sorry

end intersection_M_N_l1873_187394


namespace numberOfCubesWithNoMoreThanFourNeighbors_l1873_187314

def unitCubesWithAtMostFourNeighbors (a b c : ℕ) (h1 : a > 4) (h2 : b > 4) (h3 : c > 4) 
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) : ℕ := 
  4 * (a - 2 + b - 2 + c - 2) + 8

theorem numberOfCubesWithNoMoreThanFourNeighbors (a b c : ℕ) 
(h1 : a > 4) (h2 : b > 4) (h3 : c > 4)
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) :
  unitCubesWithAtMostFourNeighbors a b c h1 h2 h3 h4 = 144 :=
sorry

end numberOfCubesWithNoMoreThanFourNeighbors_l1873_187314


namespace greatest_possible_difference_l1873_187313

theorem greatest_possible_difference (x y : ℝ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) :
  ∃ n : ℤ, n = 9 ∧ ∀ x' y' : ℤ, (6 < x' ∧ x' < 10 ∧ 10 < y' ∧ y' < 17) → (y' - x' ≤ n) :=
by {
  -- here goes the actual proof
  sorry
}

end greatest_possible_difference_l1873_187313


namespace probability_at_least_one_white_l1873_187317

def total_number_of_pairs : ℕ := 10
def number_of_pairs_with_at_least_one_white_ball : ℕ := 7

theorem probability_at_least_one_white :
  (number_of_pairs_with_at_least_one_white_ball : ℚ) / (total_number_of_pairs : ℚ) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_l1873_187317


namespace number_of_students_is_four_l1873_187312

-- Definitions from the conditions
def average_weight_decrease := 8
def replaced_student_weight := 96
def new_student_weight := 64
def weight_decrease := replaced_student_weight - new_student_weight

-- Goal: Prove that the number of students is 4
theorem number_of_students_is_four
  (average_weight_decrease: ℕ)
  (replaced_student_weight new_student_weight: ℕ)
  (weight_decrease: ℕ) :
  weight_decrease / average_weight_decrease = 4 := 
by
  sorry

end number_of_students_is_four_l1873_187312


namespace kevin_marbles_l1873_187324

theorem kevin_marbles (M : ℕ) (h1 : 40 * 3 = 120) (h2 : 4 * M = 320 - 120) :
  M = 50 :=
by {
  sorry
}

end kevin_marbles_l1873_187324


namespace sum_of_all_three_digit_positive_even_integers_l1873_187318

def sum_of_three_digit_even_integers : ℕ :=
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_all_three_digit_positive_even_integers :
  sum_of_three_digit_even_integers = 247050 :=
by
  -- proof to be completed
  sorry

end sum_of_all_three_digit_positive_even_integers_l1873_187318


namespace inradius_of_triangle_l1873_187300

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 16) (h3 : c = 17) : 
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    let r := area / s
    r = Real.sqrt 21 := by
  sorry

end inradius_of_triangle_l1873_187300


namespace value_of_M_l1873_187336

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)
noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)
noncomputable def M : ℝ := a - b

theorem value_of_M : M = 4 :=
by
  sorry

end value_of_M_l1873_187336


namespace person_B_reads_more_than_A_l1873_187319

-- Assuming people are identifiers for Person A and Person B.
def pages_read_A (days : ℕ) (daily_read : ℕ) : ℕ := days * daily_read

def pages_read_B (days : ℕ) (daily_read : ℕ) (rest_cycle : ℕ) : ℕ := 
  let full_cycles := days / rest_cycle
  let remainder_days := days % rest_cycle
  let active_days := days - full_cycles
  active_days * daily_read

-- Given conditions
def daily_read_A := 8
def daily_read_B := 13
def rest_cycle_B := 3
def total_days := 7

-- The main theorem to prove
theorem person_B_reads_more_than_A : 
  (pages_read_B total_days daily_read_B rest_cycle_B) - (pages_read_A total_days daily_read_A) = 9 :=
by
  sorry

end person_B_reads_more_than_A_l1873_187319


namespace monthly_income_calculation_l1873_187374

variable (deposit : ℝ)
variable (percentage : ℝ)
variable (monthly_income : ℝ)

theorem monthly_income_calculation 
    (h1 : deposit = 3800) 
    (h2 : percentage = 0.32) 
    (h3 : deposit = percentage * monthly_income) : 
    monthly_income = 11875 :=
by
  sorry

end monthly_income_calculation_l1873_187374


namespace expand_expression_l1873_187381

variable (x y z : ℝ)

theorem expand_expression : (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end expand_expression_l1873_187381


namespace solution_to_equation_l1873_187391

theorem solution_to_equation (x y : ℕ → ℕ) (h1 : x 1 = 2) (h2 : y 1 = 3)
  (h3 : ∀ k, x (k + 1) = 3 * x k + 2 * y k)
  (h4 : ∀ k, y (k + 1) = 4 * x k + 3 * y k) :
  ∀ n, 2 * (x n)^2 + 1 = (y n)^2 := 
by
  sorry

end solution_to_equation_l1873_187391


namespace find_a_range_l1873_187354

open Real

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (-1, -6)
def C : (ℝ × ℝ) := (-3, 2)

-- Define the system of inequalities representing the region D
def region_D (x y : ℝ) : Prop :=
  7 * x - 5 * y - 23 ≤ 0 ∧
  x + 7 * y - 11 ≤ 0 ∧
  4 * x + y + 10 ≥ 0

-- Define the inequality condition for points B and C on opposite sides of the line 4x - 3y - a = 0
def opposite_sides (a : ℝ) : Prop :=
  (14 - a) * (-18 - a) < 0

-- Lean statement to prove the given problem
theorem find_a_range : 
  ∃ a : ℝ, region_D 0 0 ∧ opposite_sides a → -18 < a ∧ a < 14 :=
by 
  sorry

end find_a_range_l1873_187354


namespace length_major_axis_eq_six_l1873_187343

-- Define the given equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 9) = 1

-- The theorem stating the length of the major axis
theorem length_major_axis_eq_six (x y : ℝ) (h : ellipse_equation x y) : 
  2 * (Real.sqrt 9) = 6 :=
by
  sorry

end length_major_axis_eq_six_l1873_187343


namespace total_chairs_l1873_187303

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l1873_187303


namespace ascorbic_acid_molecular_weight_l1873_187332

theorem ascorbic_acid_molecular_weight (C H O : ℕ → ℝ)
  (C_weight : C 6 = 6 * 12.01)
  (H_weight : H 8 = 8 * 1.008)
  (O_weight : O 6 = 6 * 16.00)
  (total_mass_given : 528 = 6 * 12.01 + 8 * 1.008 + 6 * 16.00) :
  6 * 12.01 + 8 * 1.008 + 6 * 16.00 = 176.124 := 
by 
  sorry

end ascorbic_acid_molecular_weight_l1873_187332


namespace geo_sequence_sum_l1873_187344

theorem geo_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 2)
  (h2 : a 4 + a 5 = 4)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  a 10 + a 11 = 16 := by
  -- Insert proof here
  sorry  -- skipping the proof

end geo_sequence_sum_l1873_187344


namespace power_function_no_origin_l1873_187330

theorem power_function_no_origin (m : ℝ) : 
  (m^2 - m - 1 <= 0) ∧ (m^2 - 3 * m + 3 = 1) → m = 1 :=
by
  intros
  sorry

end power_function_no_origin_l1873_187330


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l1873_187355

noncomputable def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 4 * (n - 1)

def S (n : ℕ) : ℕ := 2 * n^2 - 2 * n

theorem geometric_sequence_general_term
    (a1 : ℕ := 2)
    (a4 : ℕ := 16)
    (h1 : a 1 = a1)
    (h2 : a 4 = a4)
    : ∀ n : ℕ, a n = a 1 * 2^(n-1) :=
by
  sorry

theorem arithmetic_sequence_sum
    (a2 : ℕ := 4)
    (a5 : ℕ := 32)
    (b2 : ℕ := a 2)
    (b9 : ℕ := a 5)
    (h1 : b 2 = b2)
    (h2 : b 9 = b9)
    : ∀ n : ℕ, S n = n * (n - 1) * 2 :=
by
  sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l1873_187355


namespace number_conversion_l1873_187350

theorem number_conversion (a b c d : ℕ) : 
  4090000 = 409 * 10000 ∧ (a = 800000) ∧ (b = 5000) ∧ (c = 20) ∧ (d = 4) → 
  (a + b + c + d = 805024) :=
by
  sorry

end number_conversion_l1873_187350


namespace floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l1873_187345

theorem floor_of_sqrt_sum_eq_floor_of_sqrt_expr (n : ℤ): 
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
sorry

end floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l1873_187345


namespace no_real_solution_l1873_187323

theorem no_real_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : 1 / a + 1 / b = 1 / (a + b)) : False :=
by
  sorry

end no_real_solution_l1873_187323


namespace train_speed_in_km_per_hr_l1873_187308

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l1873_187308


namespace box_volume_l1873_187352

-- Given conditions
variables (a b c : ℝ)
axiom ab_eq : a * b = 30
axiom bc_eq : b * c = 18
axiom ca_eq : c * a = 45

-- Prove that the volume of the box (a * b * c) equals 90 * sqrt(3)
theorem box_volume : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end box_volume_l1873_187352


namespace pow_sum_geq_pow_prod_l1873_187361

theorem pow_sum_geq_pow_prod (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 ≥ x^4 * y + x * y^4 :=
 by sorry

end pow_sum_geq_pow_prod_l1873_187361


namespace p_and_not_q_l1873_187331

def p : Prop :=
  ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≥ (1 / 3) ^ x

def q : Prop :=
  ∃ x : ℕ, x > 0 ∧ 2^x + 2^(1-x) = 2 * Real.sqrt 2

theorem p_and_not_q : p ∧ ¬q :=
by
  have h_p : p := sorry
  have h_not_q : ¬q := sorry
  exact ⟨h_p, h_not_q⟩

end p_and_not_q_l1873_187331


namespace y_time_to_complete_work_l1873_187366

-- Definitions of the conditions
def work_rate_x := 1 / 40
def work_done_by_x_in_8_days := 8 * work_rate_x
def remaining_work := 1 - work_done_by_x_in_8_days
def y_completion_time := 32
def work_rate_y := remaining_work / y_completion_time

-- Lean theorem
theorem y_time_to_complete_work :
  y_completion_time * work_rate_y = 1 →
  (1 / work_rate_y = 40) :=
by
  sorry

end y_time_to_complete_work_l1873_187366


namespace expand_simplify_expression_l1873_187376

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l1873_187376


namespace heptagon_diagonals_l1873_187337

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l1873_187337


namespace average_salary_for_company_l1873_187359

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l1873_187359


namespace cycle_selling_price_l1873_187384

theorem cycle_selling_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (percentage_gain : ℝ)
  (h_cost_price : cost_price = 1000)
  (h_percentage_gain : percentage_gain = 8) :
  selling_price = cost_price + (percentage_gain / 100) * cost_price :=
by
  sorry

end cycle_selling_price_l1873_187384


namespace find_a_value_l1873_187310

noncomputable def solve_for_a (y : ℝ) (a : ℝ) : Prop :=
  0 < y ∧ (a * y) / 20 + (3 * y) / 10 = 0.6499999999999999 * y 

theorem find_a_value (y : ℝ) (a : ℝ) (h : solve_for_a y a) : a = 7 := 
by 
  sorry

end find_a_value_l1873_187310


namespace shaded_area_10x12_floor_l1873_187371

theorem shaded_area_10x12_floor :
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  total_shaded_area = 90 - 30 * π :=
by
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  show total_shaded_area = 90 - 30 * π
  sorry

end shaded_area_10x12_floor_l1873_187371


namespace inv_prop_x_y_l1873_187339

theorem inv_prop_x_y (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 4) (h3 : y = 2) (h4 : y = 10) : x = 4 / 5 :=
by
  sorry

end inv_prop_x_y_l1873_187339


namespace race_course_length_l1873_187377

variable (v_A v_B d : ℝ)

theorem race_course_length (h1 : v_A = 4 * v_B) (h2 : (d - 60) / v_B = d / v_A) : d = 80 := by
  sorry

end race_course_length_l1873_187377


namespace roots_k_m_l1873_187347

theorem roots_k_m (k m : ℝ) 
  (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 11 ∧ a * b + b * c + c * a = k ∧ a * b * c = m)
  : k + m = 52 :=
sorry

end roots_k_m_l1873_187347


namespace range_of_m_l1873_187375

-- Define the proposition
def P : Prop := ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x + 1) + m = 0

-- Given that the negation of P is false
axiom neg_P_false : ¬¬P

-- Prove the range of m
theorem range_of_m : ∀ m : ℝ, (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l1873_187375


namespace solve_fraction_equation_l1873_187348

theorem solve_fraction_equation (t : ℝ) (h₀ : t ≠ 6) (h₁ : t ≠ -4) :
  (t = -2 ∨ t = -5) ↔ (t^2 - 3 * t - 18) / (t - 6) = 2 / (t + 4) := 
by
  sorry

end solve_fraction_equation_l1873_187348


namespace find_m_l1873_187379

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_m (m : ℝ) (h : is_perpendicular vector_a (vector_b m)) : m = 1 / 2 :=
by 
  sorry

end find_m_l1873_187379


namespace seating_arrangements_l1873_187335

/-- 
Given seven seats in a row, with four people sitting such that exactly two adjacent seats are empty,
prove that the number of different seating arrangements is 480.
-/
theorem seating_arrangements (seats people : ℕ) (adj_empty : ℕ) : 
  seats = 7 → people = 4 → adj_empty = 2 → 
  (∃ count : ℕ, count = 480) :=
by
  sorry

end seating_arrangements_l1873_187335


namespace area_of_billboard_l1873_187346

variable (L W : ℕ) (P : ℕ)
variable (hW : W = 8) (hP : P = 46)

theorem area_of_billboard (h1 : P = 2 * L + 2 * W) : L * W = 120 :=
by
  sorry

end area_of_billboard_l1873_187346


namespace interval_intersection_l1873_187302

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l1873_187302


namespace gcd_m_n_l1873_187316

namespace GCDProof

def m : ℕ := 33333333
def n : ℕ := 666666666

theorem gcd_m_n : gcd m n = 2 := 
  sorry

end GCDProof

end gcd_m_n_l1873_187316


namespace max_sum_value_l1873_187356

noncomputable def maxSum (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : ℤ :=
  i + j + k

theorem max_sum_value (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  maxSum i j k h ≤ 77 :=
  sorry

end max_sum_value_l1873_187356


namespace sum_of_three_consecutive_integers_product_336_l1873_187380

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l1873_187380


namespace find_n_l1873_187399

def digit_sum (n : ℕ) : ℕ :=
-- This function needs a proper definition for the digit sum, we leave it as sorry for this example.
sorry

def num_sevens (n : ℕ) : ℕ :=
7 * (10^n - 1) / 9

def product (n : ℕ) : ℕ :=
8 * num_sevens n

theorem find_n (n : ℕ) : digit_sum (product n) = 800 ↔ n = 788 :=
sorry

end find_n_l1873_187399


namespace recurring_decimal_difference_fraction_l1873_187304

noncomputable def recurring_decimal_seventy_three := 73 / 99
noncomputable def decimal_seventy_three := 73 / 100

theorem recurring_decimal_difference_fraction :
  recurring_decimal_seventy_three - decimal_seventy_three = 73 / 9900 := sorry

end recurring_decimal_difference_fraction_l1873_187304


namespace geometry_problem_l1873_187327

-- Definitions for geometrical entities
variable {Point : Type} -- type representing points

variable (Line : Type) -- type representing lines
variable (Plane : Type) -- type representing planes

-- Parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop) 
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Given conditions
axiom condition1 : perpendicular α β
axiom condition2 : perpendicular_line_plane m β
axiom condition3 : ¬ contained_in m α

-- Statement of the problem in Lean 4
theorem geometry_problem : parallel m α :=
by
  -- proof will involve using the axioms and definitions
  sorry

end geometry_problem_l1873_187327


namespace complement_of_M_l1873_187307

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {a | a ^ 2 - 2 * a > 0}
noncomputable def C_U_M : Set ℝ := U \ M

theorem complement_of_M :
  C_U_M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end complement_of_M_l1873_187307


namespace power_function_increasing_is_3_l1873_187325

theorem power_function_increasing_is_3 (m : ℝ) :
  (∀ x : ℝ, x > 0 → (m^2 - m - 5) * (x^(m)) > 0) ∧ (m^2 - m - 5 = 1) → m = 3 :=
by
  sorry

end power_function_increasing_is_3_l1873_187325


namespace rohan_salary_l1873_187386

variable (S : ℝ)

theorem rohan_salary (h₁ : (0.20 * S = 2500)) : S = 12500 :=
by
  sorry

end rohan_salary_l1873_187386


namespace police_officers_on_duty_l1873_187349

theorem police_officers_on_duty
  (female_officers : ℕ)
  (percent_female_on_duty : ℚ)
  (total_female_on_duty : ℕ)
  (total_officers_on_duty : ℕ)
  (H1 : female_officers = 1000)
  (H2 : percent_female_on_duty = 15 / 100)
  (H3 : total_female_on_duty = percent_female_on_duty * female_officers)
  (H4 : 2 * total_female_on_duty = total_officers_on_duty) :
  total_officers_on_duty = 300 :=
by
  sorry

end police_officers_on_duty_l1873_187349


namespace steven_owes_jeremy_l1873_187333

-- Define the payment per room
def payment_per_room : ℚ := 13 / 3

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 5 / 2

-- Calculate the total amount owed
def total_amount_owed : ℚ := payment_per_room * rooms_cleaned

-- The theorem statement to prove
theorem steven_owes_jeremy :
  total_amount_owed = 65 / 6 :=
by
  sorry

end steven_owes_jeremy_l1873_187333


namespace union_complements_eq_l1873_187358

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complements_eq :
  U = {0, 1, 3, 5, 6, 8} →
  A = {1, 5, 8} →
  B = {2} →
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  -- Prove that (U \ A) ∪ B = {0, 2, 3, 6}
  sorry

end union_complements_eq_l1873_187358


namespace smallest_natural_number_B_l1873_187340

theorem smallest_natural_number_B (A : ℕ) (h : A % 2 = 0 ∧ A % 3 = 0) :
    ∃ B : ℕ, (360 / (A^3 / B) = 5) ∧ B = 3 :=
by
  sorry

end smallest_natural_number_B_l1873_187340


namespace intersection_of_intervals_l1873_187305

theorem intersection_of_intervals (m n x : ℝ) (h1 : -1 < m) (h2 : m < 0) (h3 : 0 < n) :
  (m < x ∧ x < n) ∧ (-1 < x ∧ x < 0) ↔ -1 < x ∧ x < 0 :=
by sorry

end intersection_of_intervals_l1873_187305


namespace range_of_quadratic_function_l1873_187372

theorem range_of_quadratic_function : 
  ∀ x : ℝ, ∃ y : ℝ, y = x^2 - 1 :=
by
  sorry

end range_of_quadratic_function_l1873_187372


namespace smallest_four_digit_divisible_by_53_l1873_187398

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l1873_187398


namespace expression_for_f_l1873_187393

theorem expression_for_f {f : ℤ → ℤ} (h : ∀ x, f (x + 1) = 3 * x + 4) : ∀ x, f x = 3 * x + 1 :=
by
  sorry

end expression_for_f_l1873_187393


namespace rectangular_prism_sum_l1873_187364

theorem rectangular_prism_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l1873_187364


namespace roger_first_bag_correct_l1873_187397

noncomputable def sandra_total_pieces : ℕ := 2 * 6
noncomputable def roger_total_pieces : ℕ := sandra_total_pieces + 2
noncomputable def roger_known_bag_pieces : ℕ := 3
noncomputable def roger_first_bag_pieces : ℕ := 11

theorem roger_first_bag_correct :
  roger_total_pieces - roger_known_bag_pieces = roger_first_bag_pieces := 
  by sorry

end roger_first_bag_correct_l1873_187397


namespace smallest_number_of_coins_l1873_187328

theorem smallest_number_of_coins :
  ∃ pennies nickels dimes quarters half_dollars : ℕ,
    pennies + nickels + dimes + quarters + half_dollars = 6 ∧
    (∀ amount : ℕ, amount < 100 →
      ∃ p n d q h : ℕ,
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧ h ≤ half_dollars ∧
        1 * p + 5 * n + 10 * d + 25 * q + 50 * h = amount) :=
sorry

end smallest_number_of_coins_l1873_187328


namespace find_g_5_l1873_187301

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem find_g_5 : g 5 = 1 :=
by
  sorry

end find_g_5_l1873_187301


namespace probability_of_making_pro_shot_l1873_187396

-- Define the probabilities given in the problem
def P_free_throw : ℚ := 4 / 5
def P_high_school_3 : ℚ := 1 / 2
def P_at_least_one : ℚ := 0.9333333333333333

-- Define the unknown probability for professional 3-pointer
def P_pro := 1 / 3

-- Calculate the probability of missing each shot
def P_miss_free_throw : ℚ := 1 - P_free_throw
def P_miss_high_school_3 : ℚ := 1 - P_high_school_3
def P_miss_pro : ℚ := 1 - P_pro

-- Define the probability of missing all shots
def P_miss_all := P_miss_free_throw * P_miss_high_school_3 * P_miss_pro

-- Now state what needs to be proved
theorem probability_of_making_pro_shot :
  (1 - P_miss_all = P_at_least_one) → P_pro = 1 / 3 :=
by
  sorry

end probability_of_making_pro_shot_l1873_187396


namespace center_square_side_length_l1873_187362

theorem center_square_side_length (s : ℝ) :
    let total_area := 120 * 120
    let l_shape_area := (5 / 24) * total_area
    let l_shape_total_area := 4 * l_shape_area
    let center_square_area := total_area - l_shape_total_area
    s^2 = center_square_area → s = 49 :=
by
  intro total_area l_shape_area l_shape_total_area center_square_area h
  sorry

end center_square_side_length_l1873_187362


namespace bjorn_cannot_prevent_vakha_l1873_187320

-- Define the primary settings and objects involved
def n_points : ℕ := 99
inductive Color
| red 
| blue 

structure GameState :=
  (turn : ℕ)
  (points : Fin n_points → Option Color)

-- Define the valid states of the game where turn must be within the range of points
def valid_state (s : GameState) : Prop :=
  s.turn ≤ n_points ∧ ∀ p, s.points p ≠ none

-- Define what it means for an equilateral triangle to be monochromatically colored
def monochromatic_equilateral_triangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Fin n_points), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (p1.val + (n_points/3) % n_points) = p2.val ∧
    (p2.val + (n_points/3) % n_points) = p3.val ∧
    (p3.val + (n_points/3) % n_points) = p1.val ∧
    (state.points p1 = state.points p2) ∧ 
    (state.points p2 = state.points p3)

-- Vakha's winning condition
def vakha_wins (state : GameState) : Prop := 
  monochromatic_equilateral_triangle state

-- Bjorn's winning condition prevents Vakha from winning
def bjorn_can_prevent_vakha (initial_state : GameState) : Prop :=
  ¬ vakha_wins initial_state

-- Main theorem stating Bjorn cannot prevent Vakha from winning
theorem bjorn_cannot_prevent_vakha : ∀ (initial_state : GameState),
  valid_state initial_state → ¬ bjorn_can_prevent_vakha initial_state :=
sorry

end bjorn_cannot_prevent_vakha_l1873_187320


namespace Berengere_contribution_l1873_187385

theorem Berengere_contribution (cake_cost_in_euros : ℝ) (emily_dollars : ℝ) (exchange_rate : ℝ)
  (h1 : cake_cost_in_euros = 6)
  (h2 : emily_dollars = 5)
  (h3 : exchange_rate = 1.25) :
  cake_cost_in_euros - emily_dollars * (1 / exchange_rate) = 2 := by
  sorry

end Berengere_contribution_l1873_187385


namespace analogy_reasoning_conducts_electricity_l1873_187383

theorem analogy_reasoning_conducts_electricity (Gold Silver Copper Iron : Prop) (conducts : Prop)
  (h1 : Gold) (h2 : Silver) (h3 : Copper) (h4 : Iron) :
  (Gold ∧ Silver ∧ Copper ∧ Iron → conducts) → (conducts → !CompleteInductive ∧ !Inductive ∧ !Deductive ∧ Analogical) :=
by
  sorry

end analogy_reasoning_conducts_electricity_l1873_187383


namespace coordinates_of_B_l1873_187388

-- Define the initial conditions
def A : ℝ × ℝ := (-2, 1)
def jump_units : ℝ := 4

-- Define the function to compute the new coordinates after the jump
def new_coordinates (start : ℝ × ℝ) (jump : ℝ) : ℝ × ℝ :=
  let (x, y) := start
  (x + jump, y)

-- State the theorem to be proved
theorem coordinates_of_B
  (A : ℝ × ℝ) (jump_units : ℝ)
  (hA : A = (-2, 1))
  (h_jump : jump_units = 4) :
  new_coordinates A jump_units = (2, 1) := 
by
  -- Placeholder for the actual proof
  sorry

end coordinates_of_B_l1873_187388


namespace sufficient_but_not_necessary_l1873_187370

theorem sufficient_but_not_necessary (x : ℝ) :
  (x < -1 → x^2 - 1 > 0) ∧ (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l1873_187370


namespace interest_calculation_years_l1873_187338

theorem interest_calculation_years
  (principal : ℤ) (rate : ℝ) (difference : ℤ) (n : ℤ)
  (h_principal : principal = 2400)
  (h_rate : rate = 0.04)
  (h_difference : difference = 1920)
  (h_equation : (principal : ℝ) * rate * n = principal - difference) :
  n = 5 := 
sorry

end interest_calculation_years_l1873_187338


namespace sandwich_not_condiment_percentage_l1873_187368

theorem sandwich_not_condiment_percentage :
  (total_weight : ℝ) → (condiment_weight : ℝ) →
  total_weight = 150 → condiment_weight = 45 →
  ((total_weight - condiment_weight) / total_weight) * 100 = 70 :=
by
  intros total_weight condiment_weight h_total h_condiment
  sorry

end sandwich_not_condiment_percentage_l1873_187368


namespace find_xyz_l1873_187334

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := 
sorry

end find_xyz_l1873_187334


namespace bacteria_growth_rate_l1873_187321

theorem bacteria_growth_rate (r : ℝ) :
  (1 + r)^6 = 64 → r = 1 :=
by
  intro h
  sorry

end bacteria_growth_rate_l1873_187321


namespace reflect_P_y_axis_l1873_187357

def P : ℝ × ℝ := (2, 1)

def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem reflect_P_y_axis :
  reflect_y_axis P = (-2, 1) :=
by
  sorry

end reflect_P_y_axis_l1873_187357


namespace linemen_count_l1873_187315

-- Define the initial conditions
def linemen_drink := 8
def skill_position_players_drink := 6
def total_skill_position_players := 10
def cooler_capacity := 126
def skill_position_players_drink_first := 5

-- Define the number of ounces drunk by skill position players during the first break
def skill_position_players_first_break := skill_position_players_drink_first * skill_position_players_drink

-- Define the theorem stating that the number of linemen (L) is 12 given the conditions
theorem linemen_count :
  ∃ L : ℕ, linemen_drink * L + skill_position_players_first_break = cooler_capacity ∧ L = 12 :=
by {
  sorry -- Proof to be provided.
}

end linemen_count_l1873_187315


namespace find_a_l1873_187326

-- Conditions as definitions:
variable (a : ℝ) (b : ℝ)
variable (A : ℝ × ℝ := (0, 0)) (B : ℝ × ℝ := (a, 0)) (C : ℝ × ℝ := (0, b))
noncomputable def area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Given conditions:
axiom h1 : b = 4
axiom h2 : area a b = 28
axiom h3 : a > 0

-- The proof goal:
theorem find_a : a = 14 := by
  -- proof omitted
  sorry

end find_a_l1873_187326


namespace remainder_div_30_l1873_187373

-- Define the conditions as Lean definitions
variables (x y z p q : ℕ)

-- Hypotheses based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- assuming the conditions
axiom x_div_by_4 : is_divisible_by x 4
axiom y_div_by_5 : is_divisible_by y 5
axiom z_div_by_6 : is_divisible_by z 6
axiom p_div_by_7 : is_divisible_by p 7
axiom q_div_by_3 : is_divisible_by q 3

-- Statement to be proved
theorem remainder_div_30 : ((x^3) * (y^2) * (z * p * q + (x + y)^3) - 10) % 30 = 20 :=
by {
  sorry -- the proof will go here
}

end remainder_div_30_l1873_187373


namespace arithmetic_progression_a6_l1873_187351

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l1873_187351


namespace sixth_graders_more_than_seventh_l1873_187389

def total_payment_seventh_graders : ℕ := 143
def total_payment_sixth_graders : ℕ := 195
def cost_per_pencil : ℕ := 13

theorem sixth_graders_more_than_seventh :
  (total_payment_sixth_graders / cost_per_pencil) - (total_payment_seventh_graders / cost_per_pencil) = 4 :=
  by
  sorry

end sixth_graders_more_than_seventh_l1873_187389


namespace fraction_of_married_men_l1873_187329

theorem fraction_of_married_men (prob_single_woman : ℚ) (H : prob_single_woman = 3 / 7) :
  ∃ (fraction_married_men : ℚ), fraction_married_men = 4 / 11 :=
by
  -- Further proof steps would go here if required
  sorry

end fraction_of_married_men_l1873_187329


namespace sam_distance_l1873_187342

theorem sam_distance (miles_marguerite : ℕ) (hours_marguerite : ℕ) (hours_sam : ℕ) 
  (speed_increase : ℚ) (avg_speed_marguerite : ℚ) (speed_sam : ℚ) (distance_sam : ℚ) :
  miles_marguerite = 120 ∧ hours_marguerite = 3 ∧ hours_sam = 4 ∧ speed_increase = 1.20 ∧
  avg_speed_marguerite = miles_marguerite / hours_marguerite ∧ 
  speed_sam = avg_speed_marguerite * speed_increase ∧
  distance_sam = speed_sam * hours_sam →
  distance_sam = 192 :=
by
  intros h
  sorry

end sam_distance_l1873_187342


namespace tangent_integer_values_l1873_187392

/-- From point P outside a circle with circumference 12π units, a tangent and a secant are drawn.
      The secant divides the circle into arcs with lengths m and n. Given that the length of the
      tangent t is the geometric mean between m and n, and that m is three times n, there are zero
      possible integer values for t. -/
theorem tangent_integer_values
  (circumference : ℝ) (m n t : ℝ)
  (h_circumference : circumference = 12 * Real.pi)
  (h_sum : m + n = 12 * Real.pi)
  (h_ratio : m = 3 * n)
  (h_tangent : t = Real.sqrt (m * n)) :
  ¬(∃ k : ℤ, t = k) := 
sorry

end tangent_integer_values_l1873_187392


namespace value_of_x_l1873_187382

theorem value_of_x (x y : ℝ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end value_of_x_l1873_187382


namespace third_set_candies_l1873_187322

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l1873_187322


namespace customer_payment_probability_l1873_187341

theorem customer_payment_probability :
  let total_customers := 100
  let age_40_50_non_mobile := 13
  let age_50_60_non_mobile := 27
  let total_40_60_non_mobile := age_40_50_non_mobile + age_50_60_non_mobile
  let probability := (total_40_60_non_mobile : ℚ) / total_customers
  probability = 2 / 5 := by
sorry

end customer_payment_probability_l1873_187341


namespace sandy_age_l1873_187363

variables (S M J : ℕ)

def Q1 : Prop := S = M - 14  -- Sandy is younger than Molly by 14 years
def Q2 : Prop := J = S + 6  -- John is older than Sandy by 6 years
def Q3 : Prop := 7 * M = 9 * S  -- The ratio of Sandy's age to Molly's age is 7:9
def Q4 : Prop := 5 * J = 6 * S  -- The ratio of Sandy's age to John's age is 5:6

theorem sandy_age (h1 : Q1 S M) (h2 : Q2 S J) (h3 : Q3 S M) (h4 : Q4 S J) : S = 49 :=
by sorry

end sandy_age_l1873_187363


namespace expression_value_l1873_187367

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 :=
by
  sorry

end expression_value_l1873_187367
