import Mathlib

namespace NUMINAMATH_GPT_park_available_spaces_l629_62954

theorem park_available_spaces :
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  total_available_spaces = 175 := 
by
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  sorry

end NUMINAMATH_GPT_park_available_spaces_l629_62954


namespace NUMINAMATH_GPT_length_of_second_train_l629_62990

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (crossing_time : ℝ)
  (total_distance : ℝ)
  (relative_speed_mps : ℝ)
  (length_second_train : ℝ) :
  length_first_train = 130 ∧ 
  speed_first_train = 60 ∧
  speed_second_train = 40 ∧
  crossing_time = 10.439164866810657 ∧
  relative_speed_mps = (speed_first_train + speed_second_train) * (5/18) ∧
  total_distance = relative_speed_mps * crossing_time ∧
  length_first_train + length_second_train = total_distance →
  length_second_train = 160 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l629_62990


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l629_62983

theorem hyperbola_real_axis_length : 
  (∃ (x y : ℝ), (x^2 / 2) - (y^2 / 4) = 1) → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l629_62983


namespace NUMINAMATH_GPT_find_line_eq_l629_62971

theorem find_line_eq
  (l : ℝ → ℝ → Prop)
  (bisects_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 → l x y)
  (perpendicular_to_line : ∀ x y : ℝ, l x y ↔ y = -1/2 * x)
  : ∀ x y : ℝ, l x y ↔ 2*x - y = 0 := by
  sorry

end NUMINAMATH_GPT_find_line_eq_l629_62971


namespace NUMINAMATH_GPT_complement_of_60_is_30_l629_62980

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

theorem complement_of_60_is_30 : complement 60 = 30 :=
by 
  sorry

end NUMINAMATH_GPT_complement_of_60_is_30_l629_62980


namespace NUMINAMATH_GPT_impossible_condition_l629_62903

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem impossible_condition (a b c : ℝ) (h : f a > f b ∧ f b > f c) : ¬ (b < a ∧ a < c) :=
by
  sorry

end NUMINAMATH_GPT_impossible_condition_l629_62903


namespace NUMINAMATH_GPT_remainder_of_3_pow_101_plus_5_mod_11_l629_62931

theorem remainder_of_3_pow_101_plus_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := by
  -- The theorem statement includes the condition that (3^101 + 5) mod 11 equals 8.
  -- The proof will make use of repetitive behavior and modular arithmetic.
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_101_plus_5_mod_11_l629_62931


namespace NUMINAMATH_GPT_no_n_nat_powers_l629_62902

theorem no_n_nat_powers (n : ℕ) : ∀ n : ℕ, ¬∃ m k : ℕ, k ≥ 2 ∧ n * (n + 1) = m ^ k := 
by 
  sorry

end NUMINAMATH_GPT_no_n_nat_powers_l629_62902


namespace NUMINAMATH_GPT_sum_max_min_eq_four_l629_62968

noncomputable def f (x : ℝ) : ℝ :=
  (|2 * x| + x^3 + 2) / (|x| + 1)

-- Define the maximum value M and minimum value m
noncomputable def M : ℝ := sorry -- The maximum value of the function f(x)
noncomputable def m : ℝ := sorry -- The minimum value of the function f(x)

theorem sum_max_min_eq_four : M + m = 4 := by
  sorry

end NUMINAMATH_GPT_sum_max_min_eq_four_l629_62968


namespace NUMINAMATH_GPT_train_speed_is_72_kmh_l629_62984

-- Length of the train in meters
def length_train : ℕ := 600

-- Length of the platform in meters
def length_platform : ℕ := 600

-- Time to cross the platform in minutes
def time_crossing_platform : ℕ := 1

-- Convert meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Convert minutes to hours
def minutes_to_hours (m : ℕ) : ℕ := m * 60

-- Speed of the train in km/hr given lengths in meters and time in minutes
def speed_train_kmh (distance_m : ℕ) (time_min : ℕ) : ℕ :=
  (meters_to_kilometers distance_m) / (minutes_to_hours time_min)

theorem train_speed_is_72_kmh :
  speed_train_kmh (length_train + length_platform) time_crossing_platform = 72 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_train_speed_is_72_kmh_l629_62984


namespace NUMINAMATH_GPT_solve_eq_l629_62957

theorem solve_eq (x : ℝ) : x^6 - 19*x^3 = 216 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l629_62957


namespace NUMINAMATH_GPT_original_employee_salary_l629_62935

-- Given conditions
def emily_original_salary : ℝ := 1000000
def emily_new_salary : ℝ := 850000
def number_of_employees : ℕ := 10
def employee_new_salary : ℝ := 35000

-- Prove the original salary of each employee
theorem original_employee_salary :
  (emily_original_salary - emily_new_salary) / number_of_employees = employee_new_salary - 20000 := 
by
  sorry

end NUMINAMATH_GPT_original_employee_salary_l629_62935


namespace NUMINAMATH_GPT_number_of_men_is_15_l629_62917

-- Define the conditions
def number_of_people : Prop :=
  ∃ (M W B : ℕ), M = 8 ∧ W = 8 ∧ B = 8 ∧ 8 * M = 120

-- Define the final statement to be proven
theorem number_of_men_is_15 (h: number_of_people) : ∃ M : ℕ, M = 15 :=
by
  obtain ⟨M, W, B, hM, hW, hB, htotal⟩ := h
  use M
  rw [hM] at htotal
  have hM15 : M = 15 := by linarith
  exact hM15

end NUMINAMATH_GPT_number_of_men_is_15_l629_62917


namespace NUMINAMATH_GPT_pages_read_on_fourth_day_l629_62933

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end NUMINAMATH_GPT_pages_read_on_fourth_day_l629_62933


namespace NUMINAMATH_GPT_B_and_C_complete_task_l629_62985

noncomputable def A_work_rate : ℚ := 1 / 12
noncomputable def B_work_rate : ℚ := 1.2 * A_work_rate
noncomputable def C_work_rate : ℚ := 2 * A_work_rate

theorem B_and_C_complete_task (B_work_rate C_work_rate : ℚ) 
    (A_work_rate : ℚ := 1 / 12) :
  B_work_rate = 1.2 * A_work_rate →
  C_work_rate = 2 * A_work_rate →
  (B_work_rate + C_work_rate) = 4 / 15 :=
by intros; sorry

end NUMINAMATH_GPT_B_and_C_complete_task_l629_62985


namespace NUMINAMATH_GPT_div_by_16_l629_62907

theorem div_by_16 (n : ℕ) : 
  ((2*n - 1)^3 - (2*n)^2 + 2*n + 1) % 16 = 0 :=
sorry

end NUMINAMATH_GPT_div_by_16_l629_62907


namespace NUMINAMATH_GPT_abs_inequality_solution_l629_62959

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ -9 / 2 < x ∧ x < 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l629_62959


namespace NUMINAMATH_GPT_find_k_l629_62941

-- The expression in terms of x, y, and k
def expression (k x y : ℝ) :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

-- The mathematical statement to be proved
theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, expression k x y ≥ 0) ∧ (∃ (x y : ℝ), expression k x y = 0) :=
sorry

end NUMINAMATH_GPT_find_k_l629_62941


namespace NUMINAMATH_GPT_distance_closer_to_R_after_meeting_l629_62951

def distance_between_R_and_S : ℕ := 80
def rate_of_man_from_R : ℕ := 5
def initial_rate_of_man_from_S : ℕ := 4

theorem distance_closer_to_R_after_meeting 
  (t : ℕ) 
  (x : ℕ) 
  (h1 : t ≠ 0) 
  (h2 : distance_between_R_and_S = 80) 
  (h3 : rate_of_man_from_R = 5) 
  (h4 : initial_rate_of_man_from_S = 4) 
  (h5 : (rate_of_man_from_R * t) 
        + (t * initial_rate_of_man_from_S 
        + ((t - 1) * t / 2)) = distance_between_R_and_S) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_distance_closer_to_R_after_meeting_l629_62951


namespace NUMINAMATH_GPT_tan_pi_over_12_minus_tan_pi_over_6_l629_62950

theorem tan_pi_over_12_minus_tan_pi_over_6 :
  (Real.tan (Real.pi / 12) - Real.tan (Real.pi / 6)) = 7 - 4 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_tan_pi_over_12_minus_tan_pi_over_6_l629_62950


namespace NUMINAMATH_GPT_sara_total_cents_l629_62952

-- Define the conditions as constants
def quarters : ℕ := 11
def value_per_quarter : ℕ := 25

-- Define the total amount formula based on the conditions
def total_cents (q : ℕ) (v : ℕ) : ℕ := q * v

-- The theorem to be proven
theorem sara_total_cents : total_cents quarters value_per_quarter = 275 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sara_total_cents_l629_62952


namespace NUMINAMATH_GPT_find_smaller_number_l629_62999

theorem find_smaller_number (L S : ℕ) (h1 : L - S = 2468) (h2 : L = 8 * S + 27) : S = 349 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l629_62999


namespace NUMINAMATH_GPT_linear_function_details_l629_62986

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end NUMINAMATH_GPT_linear_function_details_l629_62986


namespace NUMINAMATH_GPT_find_parabola_equation_l629_62991

noncomputable def parabola_equation (a : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ), 
    F.1 = a / 4 ∧ F.2 = 0 ∧
    A.1 = 0 ∧ A.2 = a / 2 ∧
    (abs (F.1 * A.2) / 2) = 4

theorem find_parabola_equation :
  ∀ (a : ℝ), parabola_equation a → a = 8 ∨ a = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_parabola_equation_l629_62991


namespace NUMINAMATH_GPT_sqrt_expression_equality_l629_62916

theorem sqrt_expression_equality :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_equality_l629_62916


namespace NUMINAMATH_GPT_average_age_before_new_students_l629_62922

theorem average_age_before_new_students
  (N : ℕ) (A : ℚ) 
  (hN : N = 8) 
  (new_avg : (A - 4) = ((A * N) + (32 * 8)) / (N + 8)) 
  : A = 40 := 
by
  sorry

end NUMINAMATH_GPT_average_age_before_new_students_l629_62922


namespace NUMINAMATH_GPT_fraction_of_science_liking_students_l629_62921

open Real

theorem fraction_of_science_liking_students (total_students math_fraction english_fraction no_fav_students math_students english_students fav_students remaining_students science_students fraction_science) :
  total_students = 30 ∧
  math_fraction = 1/5 ∧
  english_fraction = 1/3 ∧
  no_fav_students = 12 ∧
  math_students = total_students * math_fraction ∧
  english_students = total_students * english_fraction ∧
  fav_students = total_students - no_fav_students ∧
  remaining_students = fav_students - (math_students + english_students) ∧
  science_students = remaining_students ∧
  fraction_science = science_students / remaining_students →
  fraction_science = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_science_liking_students_l629_62921


namespace NUMINAMATH_GPT_quadratic_form_ratio_l629_62947

theorem quadratic_form_ratio (x y u v : ℤ) (h : ∃ k : ℤ, k * (u^2 + 3*v^2) = x^2 + 3*y^2) :
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 := sorry

end NUMINAMATH_GPT_quadratic_form_ratio_l629_62947


namespace NUMINAMATH_GPT_count_defective_pens_l629_62936

theorem count_defective_pens
  (total_pens : ℕ) (prob_non_defective : ℚ)
  (h1 : total_pens = 12)
  (h2 : prob_non_defective = 0.5454545454545454) :
  ∃ (D : ℕ), D = 1 := by
  sorry

end NUMINAMATH_GPT_count_defective_pens_l629_62936


namespace NUMINAMATH_GPT_diagonal_square_grid_size_l629_62900

theorem diagonal_square_grid_size (n : ℕ) (h : 2 * n - 1 = 2017) : n = 1009 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_square_grid_size_l629_62900


namespace NUMINAMATH_GPT_angle_B_is_pi_div_3_sin_C_value_l629_62995

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (cos_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (triangle_ineq : 0 < A ∧ A < Real.pi)
variable (sin_positive : Real.sin A > 0)
variable (a_eq_2 : a = 2)
variable (c_eq_3 : c = 3)

-- Proving B = π / 3 under given conditions
theorem angle_B_is_pi_div_3 : B = Real.pi / 3 := sorry

-- Proving sin C under given additional conditions
theorem sin_C_value : Real.sin C = 3 * Real.sqrt 14 / 14 := sorry

end NUMINAMATH_GPT_angle_B_is_pi_div_3_sin_C_value_l629_62995


namespace NUMINAMATH_GPT_hyperbola_equation_l629_62966

theorem hyperbola_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 - 2 → (∃ k : ℝ, k ≠ 0 ∧ x * y = k) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l629_62966


namespace NUMINAMATH_GPT_slope_of_tangent_line_l629_62914

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_line_l629_62914


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l629_62908

-- First problem
theorem problem1 : 24 - |(-2)| + (-16) - 8 = -2 := by
  sorry

-- Second problem
theorem problem2 : (-2) * (3 / 2) / (-3 / 4) * 4 = 4 := by
  sorry

-- Third problem
theorem problem3 : -1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l629_62908


namespace NUMINAMATH_GPT_probability_heads_at_least_10_in_12_flips_l629_62953

theorem probability_heads_at_least_10_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 79 / 4096 := by
  sorry

end NUMINAMATH_GPT_probability_heads_at_least_10_in_12_flips_l629_62953


namespace NUMINAMATH_GPT_fraction_second_year_students_l629_62937

theorem fraction_second_year_students
  (total_students : ℕ)
  (third_year_students : ℕ)
  (second_year_students : ℕ)
  (h1 : third_year_students = total_students * 30 / 100)
  (h2 : second_year_students = total_students * 10 / 100) :
  (second_year_students : ℚ) / (total_students - third_year_students) = 1 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_second_year_students_l629_62937


namespace NUMINAMATH_GPT_probability_of_consonant_initials_l629_62961

def number_of_students : Nat := 30
def alphabet_size : Nat := 26
def redefined_vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
def number_of_vowels : Nat := redefined_vowels.card
def number_of_consonants : Nat := alphabet_size - number_of_vowels

theorem probability_of_consonant_initials :
  (number_of_consonants : ℝ) / (number_of_students : ℝ) = 2/3 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_consonant_initials_l629_62961


namespace NUMINAMATH_GPT_total_paint_correct_l629_62977

-- Define the current gallons of paint he has
def current_paint : ℕ := 36

-- Define the gallons of paint he bought
def bought_paint : ℕ := 23

-- Define the additional gallons of paint he needs
def needed_paint : ℕ := 11

-- The total gallons of paint he needs for finishing touches
def total_paint_needed : ℕ := current_paint + bought_paint + needed_paint

-- The proof statement to show that the total paint needed is 70
theorem total_paint_correct : total_paint_needed = 70 := by
  sorry

end NUMINAMATH_GPT_total_paint_correct_l629_62977


namespace NUMINAMATH_GPT_Jessie_initial_weight_l629_62970

def lost_first_week : ℕ := 56
def after_first_week : ℕ := 36

theorem Jessie_initial_weight :
  (after_first_week + lost_first_week = 92) :=
by
  sorry

end NUMINAMATH_GPT_Jessie_initial_weight_l629_62970


namespace NUMINAMATH_GPT_determine_range_of_x_l629_62920

theorem determine_range_of_x (x : ℝ) (h₁ : 1/x < 3) (h₂ : 1/x > -2) : x > 1/3 ∨ x < -1/2 :=
sorry

end NUMINAMATH_GPT_determine_range_of_x_l629_62920


namespace NUMINAMATH_GPT_molecular_weight_of_1_mole_l629_62982

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_1_mole_l629_62982


namespace NUMINAMATH_GPT_cost_effective_plan1_l629_62945

/-- 
Plan 1 involves purchasing a 80 yuan card and a subsequent fee of 10 yuan per session.
Plan 2 involves a fee of 20 yuan per session without purchasing the card.
We want to prove that Plan 1 is more cost-effective than Plan 2 for any number of sessions x > 8.
-/
theorem cost_effective_plan1 (x : ℕ) (h : x > 8) : 
  10 * x + 80 < 20 * x :=
sorry

end NUMINAMATH_GPT_cost_effective_plan1_l629_62945


namespace NUMINAMATH_GPT_pyramid_side_length_difference_l629_62944

theorem pyramid_side_length_difference (x : ℕ) (h1 : 1 + x^2 + (x + 1)^2 + (x + 2)^2 = 30) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_side_length_difference_l629_62944


namespace NUMINAMATH_GPT_adult_ticket_cost_l629_62911

/--
Tickets at a local theater cost a certain amount for adults and 2 dollars for kids under twelve.
Given that 175 tickets were sold and the profit was 750 dollars, and 75 kid tickets were sold,
prove that an adult ticket costs 6 dollars.
-/
theorem adult_ticket_cost
  (kid_ticket_price : ℕ := 2)
  (kid_tickets_sold : ℕ := 75)
  (total_tickets_sold : ℕ := 175)
  (total_profit : ℕ := 750)
  (adult_tickets_sold : ℕ := total_tickets_sold - kid_tickets_sold)
  (adult_ticket_revenue : ℕ := total_profit - kid_ticket_price * kid_tickets_sold)
  (adult_ticket_cost : ℕ := adult_ticket_revenue / adult_tickets_sold) :
  adult_ticket_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l629_62911


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_find_xy_l629_62928

variable {x y : ℝ}

theorem find_x_squared_plus_y_squared (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x^2 + y^2 = 34 :=
sorry

theorem find_xy (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x * y = 15 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_find_xy_l629_62928


namespace NUMINAMATH_GPT_sum_of_c_d_l629_62929

theorem sum_of_c_d (c d : ℝ) (g : ℝ → ℝ) 
(hg : ∀ x, g x = (x + 5) / (x^2 + c * x + d)) 
(hasymp : ∀ x, (x = 2 ∨ x = -3) → x^2 + c * x + d = 0) : 
c + d = -5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_c_d_l629_62929


namespace NUMINAMATH_GPT_mushroom_children_count_l629_62960

variables {n : ℕ} {A V S R : ℕ}

-- Conditions:
def condition1 (n : ℕ) (A : ℕ) (V : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → V + A / 2 = k

def condition2 (S : ℕ) (A : ℕ) (R : ℕ) (V : ℕ) : Prop :=
  S + A = R + V + A

-- Proof statement
theorem mushroom_children_count (n : ℕ) (A : ℕ) (V : ℕ) (S : ℕ) (R : ℕ) :
  condition1 n A V → condition2 S A R V → n = 6 :=
by
  intros hcondition1 hcondition2
  sorry

end NUMINAMATH_GPT_mushroom_children_count_l629_62960


namespace NUMINAMATH_GPT_fewest_tiles_to_cover_region_l629_62932

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end NUMINAMATH_GPT_fewest_tiles_to_cover_region_l629_62932


namespace NUMINAMATH_GPT_even_m_n_l629_62918

variable {m n : ℕ}

theorem even_m_n
  (h_m : ∃ k : ℕ, m = 2 * k + 1)
  (h_n : ∃ k : ℕ, n = 2 * k + 1) :
  Even ((m - n) ^ 2) ∧ Even ((m - n - 4) ^ 2) ∧ Even (2 * m * n + 4) :=
by
  sorry

end NUMINAMATH_GPT_even_m_n_l629_62918


namespace NUMINAMATH_GPT_pete_numbers_count_l629_62969

theorem pete_numbers_count :
  ∃ x_values : Finset Nat, x_values.card = 4 ∧
  ∀ x ∈ x_values, ∃ y z : Nat, 
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x + y) * z = 14 ∧ (x * y) + z = 14 :=
by
  sorry

end NUMINAMATH_GPT_pete_numbers_count_l629_62969


namespace NUMINAMATH_GPT_fixed_point_line_l629_62901

theorem fixed_point_line (k : ℝ) :
  ∃ A : ℝ × ℝ, (3 + k) * A.1 - 2 * A.2 + 1 - k = 0 ∧ (A = (1, 2)) :=
by
  let A : ℝ × ℝ := (1, 2)
  use A
  sorry

end NUMINAMATH_GPT_fixed_point_line_l629_62901


namespace NUMINAMATH_GPT_polynomial_factorization_proof_l629_62997

noncomputable def factorizable_binary_quadratic (m : ℚ) : Prop :=
  ∃ (a b : ℚ), (3*a - 5*b = 17) ∧ (a*b = -4) ∧ (m = 2*a + 3*b)

theorem polynomial_factorization_proof :
  ∀ (m : ℚ), factorizable_binary_quadratic m ↔ (m = 5 ∨ m = -58 / 15) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_proof_l629_62997


namespace NUMINAMATH_GPT_cost_to_fill_half_of_can_B_l629_62998

theorem cost_to_fill_half_of_can_B (r h : ℝ) (cost_fill_V : ℝ) (cost_fill_V_eq : cost_fill_V = 16)
  (V_radius_eq : 2 * r = radius_of_can_V)
  (V_height_eq: h / 2 = height_of_can_V) :
  cost_fill_half_of_can_B = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_fill_half_of_can_B_l629_62998


namespace NUMINAMATH_GPT_cos_alpha_plus_20_eq_neg_alpha_l629_62987

variable (α : ℝ)

theorem cos_alpha_plus_20_eq_neg_alpha (h : Real.sin (α - 70 * Real.pi / 180) = α) :
    Real.cos (α + 20 * Real.pi / 180) = -α :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_20_eq_neg_alpha_l629_62987


namespace NUMINAMATH_GPT_no_two_ways_for_z_l629_62946

theorem no_two_ways_for_z (z : ℤ) (x y x' y' : ℕ) 
  (hx : x ≤ y) (hx' : x' ≤ y') : ¬ (z = x! + y! ∧ z = x'! + y'! ∧ (x ≠ x' ∨ y ≠ y')) :=
by
  sorry

end NUMINAMATH_GPT_no_two_ways_for_z_l629_62946


namespace NUMINAMATH_GPT_BrotherUpperLimit_l629_62940

variable (w : ℝ) -- Arun's weight
variable (b : ℝ) -- Upper limit of Arun's weight according to his brother's opinion

-- Conditions as per the problem
def ArunOpinion (w : ℝ) := 64 < w ∧ w < 72
def BrotherOpinion (w b : ℝ) := 60 < w ∧ w < b
def MotherOpinion (w : ℝ) := w ≤ 67

-- The average of probable weights
def AverageWeight (weights : Set ℝ) (avg : ℝ) := (∀ w ∈ weights, 64 < w ∧ w ≤ 67) ∧ avg = 66

-- The main theorem to be proven
theorem BrotherUpperLimit (hA : ArunOpinion w) (hB : BrotherOpinion w b) (hM : MotherOpinion w) (hAvg : AverageWeight {w | 64 < w ∧ w ≤ 67} 66) : b = 67 := by
  sorry

end NUMINAMATH_GPT_BrotherUpperLimit_l629_62940


namespace NUMINAMATH_GPT_math_problem_l629_62948

variable {p q r x y : ℝ}

theorem math_problem (h1 : p / q = 6 / 7)
                     (h2 : p / r = 8 / 9)
                     (h3 : q / r = x / y) :
                     x = 28 ∧ y = 27 ∧ 2 * p + q = (19 / 6) * p := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l629_62948


namespace NUMINAMATH_GPT_total_households_in_apartment_complex_l629_62926

theorem total_households_in_apartment_complex :
  let buildings := 25
  let floors_per_building := 10
  let households_per_floor := 8
  buildings * floors_per_building * households_per_floor = 2000 :=
by
  sorry

end NUMINAMATH_GPT_total_households_in_apartment_complex_l629_62926


namespace NUMINAMATH_GPT_first_chinese_supercomputer_is_milkyway_l629_62915

-- Define the names of the computers
inductive ComputerName
| Universe
| Taihu
| MilkyWay
| Dawn

-- Define a structure to hold the properties of the computer
structure Computer :=
  (name : ComputerName)
  (introduction_year : Nat)
  (calculations_per_second : Nat)

-- Define the properties of the specific computer in the problem
def first_chinese_supercomputer := 
  Computer.mk ComputerName.MilkyWay 1983 100000000

-- The theorem to be proven
theorem first_chinese_supercomputer_is_milkyway :
  first_chinese_supercomputer.name = ComputerName.MilkyWay :=
by
  -- Provide the conditions that lead to the conclusion (proof steps will be added here)
  sorry

end NUMINAMATH_GPT_first_chinese_supercomputer_is_milkyway_l629_62915


namespace NUMINAMATH_GPT_blue_paint_amount_l629_62965

/-- 
Prove that if Giselle uses 15 quarts of white paint, then according to the ratio 4:3:5, she should use 12 quarts of blue paint.
-/
theorem blue_paint_amount (white_paint : ℚ) (h1 : white_paint = 15) : 
  let blue_ratio := 4;
  let white_ratio := 5;
  blue_ratio / white_ratio * white_paint = 12 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_amount_l629_62965


namespace NUMINAMATH_GPT_double_acute_angle_lt_180_l629_62993

theorem double_acute_angle_lt_180
  (α : ℝ) (h : 0 < α ∧ α < 90) : 2 * α < 180 := 
sorry

end NUMINAMATH_GPT_double_acute_angle_lt_180_l629_62993


namespace NUMINAMATH_GPT_right_triangle_of_pythagorean_l629_62978

theorem right_triangle_of_pythagorean
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC CA : ℝ)
  (h : AB^2 = BC^2 + CA^2) : ∃ (c : ℕ), c = 90 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_of_pythagorean_l629_62978


namespace NUMINAMATH_GPT_num_ordered_pairs_squares_diff_30_l629_62904

theorem num_ordered_pairs_squares_diff_30 :
  ∃ (n : ℕ), n = 0 ∧
  ∀ (m n: ℕ), 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 30 → false :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_squares_diff_30_l629_62904


namespace NUMINAMATH_GPT_conditionA_is_necessary_for_conditionB_l629_62981

-- Definitions for conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (area : ℝ) -- area of the triangle

def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

def conditionA (t1 t2 : Triangle) : Prop :=
  t1.area = t2.area ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem conditionA_is_necessary_for_conditionB (t1 t2 : Triangle) :
  congruent t1 t2 → conditionA t1 t2 :=
by sorry

end NUMINAMATH_GPT_conditionA_is_necessary_for_conditionB_l629_62981


namespace NUMINAMATH_GPT_subtraction_result_l629_62943

theorem subtraction_result : 3.05 - 5.678 = -2.628 := 
by
  sorry

end NUMINAMATH_GPT_subtraction_result_l629_62943


namespace NUMINAMATH_GPT_neg_prop_l629_62963

theorem neg_prop : ∃ (a : ℝ), ∀ (x : ℝ), (a * x^2 - 3 * x + 2 = 0) → x ≤ 0 :=
sorry

end NUMINAMATH_GPT_neg_prop_l629_62963


namespace NUMINAMATH_GPT_intersection_points_l629_62964

noncomputable def hyperbola : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 9 - y^2 = 1 }

noncomputable def line : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = (1 / 3) * (x + 1) }

theorem intersection_points :
  ∃! (p : ℝ × ℝ), p ∈ hyperbola ∧ p ∈ line :=
sorry

end NUMINAMATH_GPT_intersection_points_l629_62964


namespace NUMINAMATH_GPT_toothpicks_per_card_l629_62967

-- Define the conditions of the problem
def numCardsInDeck : ℕ := 52
def numCardsNotUsed : ℕ := 16
def numCardsUsed : ℕ := numCardsInDeck - numCardsNotUsed

def numBoxesToothpicks : ℕ := 6
def toothpicksPerBox : ℕ := 450
def totalToothpicksUsed : ℕ := numBoxesToothpicks * toothpicksPerBox

-- Prove the number of toothpicks used per card
theorem toothpicks_per_card : totalToothpicksUsed / numCardsUsed = 75 := 
  by sorry

end NUMINAMATH_GPT_toothpicks_per_card_l629_62967


namespace NUMINAMATH_GPT_ratio_a_b_l629_62913

-- Definitions of the arithmetic sequences
open Classical

noncomputable def sequence1 (a y b : ℕ) : ℕ → ℕ
| 0 => a
| 1 => y
| 2 => b
| 3 => 14
| _ => 0 -- only the first four terms are given for sequence1

noncomputable def sequence2 (x y : ℕ) : ℕ → ℕ
| 0 => 2
| 1 => x
| 2 => 6
| 3 => y
| _ => 0 -- only the first four terms are given for sequence2

theorem ratio_a_b (a y b x : ℕ) (h1 : sequence1 a y b 0 = a) (h2 : sequence1 a y b 1 = y) 
  (h3 : sequence1 a y b 2 = b) (h4 : sequence1 a y b 3 = 14)
  (h5 : sequence2 x y 0 = 2) (h6 : sequence2 x y 1 = x) 
  (h7 : sequence2 x y 2 = 6) (h8 : sequence2 x y 3 = y) :
  (a:ℚ) / b = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_a_b_l629_62913


namespace NUMINAMATH_GPT_percentage_very_satisfactory_l629_62955

-- Definitions based on conditions
def total_parents : ℕ := 120
def needs_improvement_count : ℕ := 6
def excellent_percentage : ℕ := 15
def satisfactory_remaining_percentage : ℕ := 80

-- Theorem statement
theorem percentage_very_satisfactory 
  (total_parents : ℕ) 
  (needs_improvement_count : ℕ) 
  (excellent_percentage : ℕ) 
  (satisfactory_remaining_percentage : ℕ) 
  (result : ℕ) : result = 16 :=
by
  sorry

end NUMINAMATH_GPT_percentage_very_satisfactory_l629_62955


namespace NUMINAMATH_GPT_log_relationship_l629_62956

theorem log_relationship (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log 2) 
  (hb : b = Real.log 4 / Real.log 3) 
  (hc : c = Real.log 5 / Real.log 4) : 
  c < b ∧ b < a :=
by 
  sorry

end NUMINAMATH_GPT_log_relationship_l629_62956


namespace NUMINAMATH_GPT_tetradecagon_edge_length_correct_l629_62949

-- Define the parameters of the problem
def regular_tetradecagon_perimeter (n : ℕ := 14) : ℕ := 154

-- Define the length of one edge
def edge_length (P : ℕ) (n : ℕ) : ℕ := P / n

-- State the theorem
theorem tetradecagon_edge_length_correct :
  edge_length (regular_tetradecagon_perimeter 14) 14 = 11 := by
  sorry

end NUMINAMATH_GPT_tetradecagon_edge_length_correct_l629_62949


namespace NUMINAMATH_GPT_GAUSS_1998_LCM_l629_62972

/-- The periodicity of cycling the word 'GAUSS' -/
def period_GAUSS : ℕ := 5

/-- The periodicity of cycling the number '1998' -/
def period_1998 : ℕ := 4

/-- The least common multiple (LCM) of the periodicities of 'GAUSS' and '1998' is 20 -/
theorem GAUSS_1998_LCM : Nat.lcm period_GAUSS period_1998 = 20 :=
by
  sorry

end NUMINAMATH_GPT_GAUSS_1998_LCM_l629_62972


namespace NUMINAMATH_GPT_simplify_expression_l629_62930

theorem simplify_expression : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l629_62930


namespace NUMINAMATH_GPT_sum_series_l629_62912

theorem sum_series (s : ℕ → ℝ) 
  (h : ∀ n : ℕ, s n = (n+1) / (4 : ℝ)^(n+1)) : 
  tsum s = (4 / 9 : ℝ) :=
sorry

end NUMINAMATH_GPT_sum_series_l629_62912


namespace NUMINAMATH_GPT_product_of_integers_cubes_sum_to_35_l629_62925

-- Define the conditions
def integers_sum_of_cubes (a b : ℤ) : Prop :=
  a^3 + b^3 = 35

-- Define the theorem that the product of integers whose cubes sum to 35 is 6
theorem product_of_integers_cubes_sum_to_35 :
  ∃ a b : ℤ, integers_sum_of_cubes a b ∧ a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_cubes_sum_to_35_l629_62925


namespace NUMINAMATH_GPT_daisy_milk_problem_l629_62989

theorem daisy_milk_problem (total_milk : ℝ) (kids_percentage : ℝ) (remaining_milk : ℝ) (used_milk : ℝ) :
  total_milk = 16 →
  kids_percentage = 0.75 →
  remaining_milk = total_milk * (1 - kids_percentage) →
  used_milk = 2 →
  (used_milk / remaining_milk) * 100 = 50 :=
by
  intros _ _ _ _ 
  sorry

end NUMINAMATH_GPT_daisy_milk_problem_l629_62989


namespace NUMINAMATH_GPT_alice_weekly_walk_distance_l629_62942

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end NUMINAMATH_GPT_alice_weekly_walk_distance_l629_62942


namespace NUMINAMATH_GPT_museum_rid_paintings_l629_62996

def initial_paintings : ℕ := 1795
def leftover_paintings : ℕ := 1322

theorem museum_rid_paintings : initial_paintings - leftover_paintings = 473 := by
  sorry

end NUMINAMATH_GPT_museum_rid_paintings_l629_62996


namespace NUMINAMATH_GPT_village_population_l629_62939

theorem village_population (P : ℝ) (h : 0.8 * P = 32000) : P = 40000 := by
  sorry

end NUMINAMATH_GPT_village_population_l629_62939


namespace NUMINAMATH_GPT_ones_digit_of_9_pow_46_l629_62973

theorem ones_digit_of_9_pow_46 : (9 ^ 46) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_9_pow_46_l629_62973


namespace NUMINAMATH_GPT_no_solution_l629_62906

theorem no_solution (x : ℝ) : ¬ (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 5) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_l629_62906


namespace NUMINAMATH_GPT_ratio_of_white_to_yellow_balls_l629_62994

theorem ratio_of_white_to_yellow_balls (original_white original_yellow extra_yellow : ℕ) 
(h1 : original_white = 32) 
(h2 : original_yellow = 32) 
(h3 : extra_yellow = 20) : 
(original_white : ℚ) / (original_yellow + extra_yellow) = 8 / 13 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_white_to_yellow_balls_l629_62994


namespace NUMINAMATH_GPT_symmetric_line_eq_l629_62927

-- Definitions for the given line equations
def l1 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
def l2 (x y : ℝ) : Prop := x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x - 3 * y - 1 = 0

-- The theorem to prove
theorem symmetric_line_eq (x y : ℝ) (h1: l1 x y) (h2: l2 x y) : l3 x y :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l629_62927


namespace NUMINAMATH_GPT_probability_greater_than_two_on_three_dice_l629_62975

theorem probability_greater_than_two_on_three_dice :
  (4 / 6 : ℚ) ^ 3 = (8 / 27 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_greater_than_two_on_three_dice_l629_62975


namespace NUMINAMATH_GPT_complement_union_l629_62979

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 3 }
noncomputable def CR (S : Set ℝ) : Set ℝ := { x : ℝ | x ∉ S }

theorem complement_union (A B : Set ℝ) :
  (CR A ∪ B) = (Set.univ \ A ∪ Set.Ioo 1 3) := by
  sorry

end NUMINAMATH_GPT_complement_union_l629_62979


namespace NUMINAMATH_GPT_people_lost_l629_62974

-- Define the given constants
def win_ratio : ℕ := 4
def lose_ratio : ℕ := 1
def people_won : ℕ := 28

-- The statement to prove that 7 people lost
theorem people_lost (win_ratio lose_ratio people_won : ℕ) (H : win_ratio * 7 = people_won * lose_ratio) : 7 = people_won * lose_ratio / win_ratio :=
by { sorry }

end NUMINAMATH_GPT_people_lost_l629_62974


namespace NUMINAMATH_GPT_problem1_problem2_l629_62988

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l629_62988


namespace NUMINAMATH_GPT_power_cycle_i_pow_2012_l629_62958

-- Define the imaginary unit i as a complex number
def i : ℂ := Complex.I

-- Define the periodic properties of i
theorem power_cycle (n : ℕ) : Complex := 
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 0 -- this case should never happen

-- Using the periodic properties
theorem i_pow_2012 : (i ^ 2012) = 1 := by
  sorry

end NUMINAMATH_GPT_power_cycle_i_pow_2012_l629_62958


namespace NUMINAMATH_GPT_equivalent_proof_problem_l629_62934

theorem equivalent_proof_problem (x : ℤ) (h : (x + 2) * (x - 2) = 1221) :
    (x = 35 ∨ x = -35) ∧ ((x + 1) * (x - 1) = 1224) :=
sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l629_62934


namespace NUMINAMATH_GPT_Mary_forgot_pigs_l629_62924

theorem Mary_forgot_pigs (Mary_thinks : ℕ) (actual_animals : ℕ) (double_counted_sheep : ℕ)
  (H_thinks : Mary_thinks = 60) (H_actual : actual_animals = 56)
  (H_double_counted : double_counted_sheep = 7) :
  ∃ pigs_forgot : ℕ, pigs_forgot = 3 :=
by
  let counted_animals := Mary_thinks - double_counted_sheep
  have H_counted_correct : counted_animals = 53 := by sorry -- 60 - 7 = 53
  have pigs_forgot := actual_animals - counted_animals
  have H_pigs_forgot : pigs_forgot = 3 := by sorry -- 56 - 53 = 3
  exact ⟨pigs_forgot, H_pigs_forgot⟩

end NUMINAMATH_GPT_Mary_forgot_pigs_l629_62924


namespace NUMINAMATH_GPT_sin_double_angle_l629_62976

theorem sin_double_angle (A : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : Real.cos A = 3 / 5) :
  Real.sin (2 * A) = 24 / 25 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l629_62976


namespace NUMINAMATH_GPT_count_linear_eqs_l629_62909

-- Define each equation as conditions
def eq1 (x y : ℝ) := 3 * x - y = 2
def eq2 (x : ℝ) := x + 1 / x + 2 = 0
def eq3 (x : ℝ) := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) := x = 0
def eq5 (x : ℝ) := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) := 1 / 2 * x = 1 / 2
def eq7 (x : ℝ) := (2 * x + 1) / 3 = 1 / 6 * x

-- Proof statement: there are exactly 3 linear equations
theorem count_linear_eqs : 
  (∃ x y, eq1 x y) ∧ eq4 0 ∧ (∃ x, eq6 x) ∧ (∃ x, eq7 x) ∧ 
  ¬ (∃ x, eq2 x) ∧ ¬ (∃ x, eq3 x) ∧ ¬ (∃ x, eq5 x) → 
  3 = 3 :=
sorry

end NUMINAMATH_GPT_count_linear_eqs_l629_62909


namespace NUMINAMATH_GPT_ellipse_properties_l629_62938

theorem ellipse_properties (h k a b : ℝ) (θ : ℝ)
  (h_def : h = -2)
  (k_def : k = 3)
  (a_def : a = 6)
  (b_def : b = 4)
  (θ_def : θ = 45) :
  h + k + a + b = 11 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l629_62938


namespace NUMINAMATH_GPT_tan_alpha_eq_2_implies_sin_2alpha_inverse_l629_62962

theorem tan_alpha_eq_2_implies_sin_2alpha_inverse (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_2_implies_sin_2alpha_inverse_l629_62962


namespace NUMINAMATH_GPT_class_B_more_uniform_l629_62910

def x_A : ℝ := 80
def x_B : ℝ := 80
def S2_A : ℝ := 240
def S2_B : ℝ := 180

theorem class_B_more_uniform (h1 : x_A = 80) (h2 : x_B = 80) (h3 : S2_A = 240) (h4 : S2_B = 180) : 
  S2_B < S2_A :=
by {
  exact sorry
}

end NUMINAMATH_GPT_class_B_more_uniform_l629_62910


namespace NUMINAMATH_GPT_digits_of_2048_in_base_9_l629_62919

def digits_base9 (n : ℕ) : ℕ :=
if n < 9 then 1 else 1 + digits_base9 (n / 9)

theorem digits_of_2048_in_base_9 : digits_base9 2048 = 4 :=
by sorry

end NUMINAMATH_GPT_digits_of_2048_in_base_9_l629_62919


namespace NUMINAMATH_GPT_joska_has_higher_probability_l629_62923

open Nat

def num_4_digit_with_all_diff_digits := 10 * 9 * 8 * 7
def total_4_digit_combinations := 10^4
def num_4_digit_with_repeated_digits := total_4_digit_combinations - num_4_digit_with_all_diff_digits

-- Calculate probabilities
noncomputable def prob_joska := (num_4_digit_with_all_diff_digits : ℝ) / (total_4_digit_combinations : ℝ)
noncomputable def prob_gabor := (num_4_digit_with_repeated_digits : ℝ) / (total_4_digit_combinations : ℝ)

theorem joska_has_higher_probability :
  prob_joska > prob_gabor :=
  by
    sorry

end NUMINAMATH_GPT_joska_has_higher_probability_l629_62923


namespace NUMINAMATH_GPT_similarity_of_triangle_l629_62992

noncomputable def side_length (AB BC AC : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 1 → (AB, BC, AC) = (k * AB, k * BC, k * AC)

theorem similarity_of_triangle (AB BC AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : AC > 0) :
  side_length (2 * AB) (2 * BC) (2 * AC) = side_length AB BC AC :=
by sorry

end NUMINAMATH_GPT_similarity_of_triangle_l629_62992


namespace NUMINAMATH_GPT_inequality_solution_set_is_correct_l629_62905

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  (3 * x - 1) / (2 - x) ≥ 1

theorem inequality_solution_set_is_correct :
  { x : ℝ | inequality_solution_set x } = { x : ℝ | 3 / 4 ≤ x ∧ x < 2 } :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_is_correct_l629_62905
