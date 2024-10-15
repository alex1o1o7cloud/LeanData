import Mathlib

namespace NUMINAMATH_GPT_percentage_reduction_is_20_percent_l1766_176693

-- Defining the initial and final prices
def initial_price : ℝ := 25
def final_price : ℝ := 16

-- Defining the percentage reduction
def percentage_reduction (x : ℝ) := 1 - x

-- The equation representing the two reductions:
def equation (x : ℝ) := initial_price * (percentage_reduction x) * (percentage_reduction x)

theorem percentage_reduction_is_20_percent :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ equation x = final_price ∧ x = 0.20 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_20_percent_l1766_176693


namespace NUMINAMATH_GPT_abscissa_of_tangent_point_l1766_176633

theorem abscissa_of_tangent_point (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = Real.exp x + a * Real.exp (-x))
  (h_odd : ∀ x, (D^[2] f x) = - (D^[2] f (-x)))
  (slope_cond : ∀ x, (D f x) = 3 / 2) : 
  ∃ x ∈ Set.Ioo (-Real.log 2) (Real.log 2), x = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_abscissa_of_tangent_point_l1766_176633


namespace NUMINAMATH_GPT_part_a_part_b_l1766_176650

theorem part_a (x : ℝ) (n : ℕ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) (hn_pos : 0 < n) :
  Real.log x < n * (x ^ (1 / n) - 1) ∧ n * (x ^ (1 / n) - 1) < (x ^ (1 / n)) * Real.log x := sorry

theorem part_b (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
  (Real.log x) = (Real.log x) := sorry

end NUMINAMATH_GPT_part_a_part_b_l1766_176650


namespace NUMINAMATH_GPT_value_of_k_l1766_176612

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x - 8

theorem value_of_k:
  (f 5) - (g 5 k) = 20 → k = -10.8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1766_176612


namespace NUMINAMATH_GPT_length_of_AB_l1766_176620

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1766_176620


namespace NUMINAMATH_GPT_option_D_min_value_is_2_l1766_176654

noncomputable def funcD (x : ℝ) : ℝ :=
  (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem option_D_min_value_is_2 :
  ∃ x : ℝ, funcD x = 2 :=
sorry

end NUMINAMATH_GPT_option_D_min_value_is_2_l1766_176654


namespace NUMINAMATH_GPT_solve_system_and_compute_l1766_176635

-- Given system of equations
variables {x y : ℝ}
variables (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5)

-- Statement to prove
theorem solve_system_and_compute :
  (x - y = -1) ∧ (x + y = 3) ∧ ((1/3 * (x^2 - y^2)) * (x^2 - 2*x*y + y^2) = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_and_compute_l1766_176635


namespace NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1766_176630

theorem circumscribed_sphere_surface_area 
    (x y z : ℝ) 
    (h1 : x * y = Real.sqrt 6) 
    (h2 : y * z = Real.sqrt 2) 
    (h3 : z * x = Real.sqrt 3) : 
    4 * Real.pi * ((Real.sqrt (x^2 + y^2 + z^2)) / 2)^2 = 6 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1766_176630


namespace NUMINAMATH_GPT_find_pairs_satisfying_conditions_l1766_176617

theorem find_pairs_satisfying_conditions :
  ∀ (m n : ℕ), (0 < m ∧ 0 < n) →
               (∃ k : ℤ, m^2 - 4 * n = k^2) →
               (∃ l : ℤ, n^2 - 4 * m = l^2) →
               (m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5) :=
by
  intros m n hmn h1 h2
  sorry

end NUMINAMATH_GPT_find_pairs_satisfying_conditions_l1766_176617


namespace NUMINAMATH_GPT_complex_solution_count_l1766_176618

theorem complex_solution_count : 
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^3 - 8) / (z^2 - 3 * z + 2) = 0) ∧ s.card = 2 := 
by
  sorry

end NUMINAMATH_GPT_complex_solution_count_l1766_176618


namespace NUMINAMATH_GPT_expression_as_fraction_l1766_176686

theorem expression_as_fraction :
  1 + (4 / (5 + (6 / 7))) = (69 : ℚ) / 41 := 
by
  sorry

end NUMINAMATH_GPT_expression_as_fraction_l1766_176686


namespace NUMINAMATH_GPT_total_money_spent_l1766_176639

theorem total_money_spent {s j : ℝ} (hs : s = 14.28) (hj : j = 4.74) : s + j = 19.02 :=
by
  sorry

end NUMINAMATH_GPT_total_money_spent_l1766_176639


namespace NUMINAMATH_GPT_remainder_when_a_squared_times_b_divided_by_n_l1766_176609

theorem remainder_when_a_squared_times_b_divided_by_n (n : ℕ) (a : ℤ) (h1 : a * 3 ≡ 1 [ZMOD n]) : 
  (a^2 * 3) % n = a % n := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_a_squared_times_b_divided_by_n_l1766_176609


namespace NUMINAMATH_GPT_solution_eq1_solution_eq2_l1766_176621

theorem solution_eq1 (x : ℝ) : 
  2 * x^2 - 4 * x - 1 = 0 ↔ 
  (x = 1 + (Real.sqrt 6) / 2 ∨ x = 1 - (Real.sqrt 6) / 2) := by
sorry

theorem solution_eq2 (x : ℝ) :
  (x - 1) * (x + 2) = 28 ↔ 
  (x = -6 ∨ x = 5) := by
sorry

end NUMINAMATH_GPT_solution_eq1_solution_eq2_l1766_176621


namespace NUMINAMATH_GPT_marked_cells_in_grid_l1766_176699

theorem marked_cells_in_grid :
  ∀ (grid : Matrix (Fin 5) (Fin 5) Bool), 
  (∀ (i j : Fin 3), ∃! (a b : Fin 3), grid (i + a + 1) (j + b + 1) = true) → ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_marked_cells_in_grid_l1766_176699


namespace NUMINAMATH_GPT_total_students_stratified_sampling_l1766_176684

namespace HighSchool

theorem total_students_stratified_sampling 
  (sample_size : ℕ)
  (sample_grade10 : ℕ)
  (sample_grade11 : ℕ)
  (students_grade12 : ℕ) 
  (n : ℕ)
  (H1 : sample_size = 100)
  (H2 : sample_grade10 = 24)
  (H3 : sample_grade11 = 26)
  (H4 : students_grade12 = 600)
  (H5 : ∀ n, (students_grade12 / n * sample_size = sample_size - sample_grade10 - sample_grade11) → n = 1200) :
  n = 1200 :=
sorry

end HighSchool

end NUMINAMATH_GPT_total_students_stratified_sampling_l1766_176684


namespace NUMINAMATH_GPT_total_days_to_finish_job_l1766_176652

noncomputable def workers_job_completion
  (initial_workers : ℕ)
  (additional_workers : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)
  (work_completion_days : ℕ)
  (remaining_work : ℝ)
  (additional_days_needed : ℝ)
  : ℝ :=
  initial_days + additional_days_needed

theorem total_days_to_finish_job
  (initial_workers : ℕ := 6)
  (additional_workers : ℕ := 4)
  (initial_days : ℕ := 3)
  (total_days : ℕ := 8)
  (work_completion_days : ℕ := 8)
  : workers_job_completion initial_workers additional_workers initial_days total_days work_completion_days (1 - (initial_days : ℝ) / work_completion_days) (remaining_work / (((initial_workers + additional_workers) : ℝ) / work_completion_days)) = 3.5 :=
  sorry

end NUMINAMATH_GPT_total_days_to_finish_job_l1766_176652


namespace NUMINAMATH_GPT_terminating_decimal_expansion_7_over_625_l1766_176625

theorem terminating_decimal_expansion_7_over_625 : (7 / 625 : ℚ) = 112 / 10000 := by
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_7_over_625_l1766_176625


namespace NUMINAMATH_GPT_find_b_l1766_176689

-- Definitions from conditions
def f (x : ℚ) := 3 * x - 2
def g (x : ℚ) := 7 - 2 * x

-- Problem statement
theorem find_b (b : ℚ) (h : g (f b) = 1) : b = 5 / 3 := sorry

end NUMINAMATH_GPT_find_b_l1766_176689


namespace NUMINAMATH_GPT_cos_double_angle_identity_l1766_176614

open Real

theorem cos_double_angle_identity (α : ℝ) 
  (h : tan (α + π / 4) = 1 / 3) : cos (2 * α) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_identity_l1766_176614


namespace NUMINAMATH_GPT_find_LN_l1766_176668

noncomputable def LM : ℝ := 9
noncomputable def sin_N : ℝ := 3 / 5
noncomputable def LN : ℝ := 15

theorem find_LN (h₁ : sin_N = 3 / 5) (h₂ : LM = 9) (h₃ : sin_N = LM / LN) : LN = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_LN_l1766_176668


namespace NUMINAMATH_GPT_mean_and_variance_of_y_l1766_176604

noncomputable def mean (xs : List ℝ) : ℝ :=
  if h : xs.length > 0 then xs.sum / xs.length else 0

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  if h : xs.length > 0 then (xs.map (λ x => (x - m)^2)).sum / xs.length else 0

theorem mean_and_variance_of_y
  (x : List ℝ)
  (hx_len : x.length = 20)
  (hx_mean : mean x = 1)
  (hx_var : variance x = 8) :
  let y := x.map (λ xi => 2 * xi + 3)
  mean y = 5 ∧ variance y = 32 :=
by
  let y := x.map (λ xi => 2 * xi + 3)
  sorry

end NUMINAMATH_GPT_mean_and_variance_of_y_l1766_176604


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1766_176653

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1766_176653


namespace NUMINAMATH_GPT_speed_of_sound_l1766_176662

theorem speed_of_sound (d₁ d₂ t : ℝ) (speed_car : ℝ) (speed_km_hr_to_m_s : ℝ) :
  d₁ = 1200 ∧ speed_car = 108 ∧ speed_km_hr_to_m_s = (speed_car * 1000 / 3600) ∧ t = 3.9669421487603307 →
  (d₁ + speed_km_hr_to_m_s * t) / t = 332.59 :=
by sorry

end NUMINAMATH_GPT_speed_of_sound_l1766_176662


namespace NUMINAMATH_GPT_gail_has_two_ten_dollar_bills_l1766_176637

-- Define the given conditions
def total_amount : ℕ := 100
def num_five_bills : ℕ := 4
def num_twenty_bills : ℕ := 3
def value_five_bill : ℕ := 5
def value_twenty_bill : ℕ := 20
def value_ten_bill : ℕ := 10

-- The function to determine the number of ten-dollar bills
noncomputable def num_ten_bills : ℕ := 
  (total_amount - (num_five_bills * value_five_bill + num_twenty_bills * value_twenty_bill)) / value_ten_bill

-- Proof statement
theorem gail_has_two_ten_dollar_bills : num_ten_bills = 2 := by
  sorry

end NUMINAMATH_GPT_gail_has_two_ten_dollar_bills_l1766_176637


namespace NUMINAMATH_GPT_new_class_mean_l1766_176631

theorem new_class_mean {X Y : ℕ} {mean_a mean_b : ℚ}
  (hx : X = 30) (hy : Y = 6) 
  (hmean_a : mean_a = 72) (hmean_b : mean_b = 78) :
  (X * mean_a + Y * mean_b) / (X + Y) = 73 := 
by 
  sorry

end NUMINAMATH_GPT_new_class_mean_l1766_176631


namespace NUMINAMATH_GPT_students_at_school_yy_l1766_176696

theorem students_at_school_yy (X Y : ℝ) 
    (h1 : X + Y = 4000)
    (h2 : 0.07 * X - 0.03 * Y = 40) : 
    Y = 2400 :=
by
  sorry

end NUMINAMATH_GPT_students_at_school_yy_l1766_176696


namespace NUMINAMATH_GPT_last_digit_expr_is_4_l1766_176665

-- Definitions for last digits.
def last_digit (n : ℕ) : ℕ := n % 10

def a : ℕ := 287
def b : ℕ := 269

def expr := (a * a) + (b * b) - (2 * a * b)

-- Conjecture stating that the last digit of the given expression is 4.
theorem last_digit_expr_is_4 : last_digit expr = 4 := 
by sorry

end NUMINAMATH_GPT_last_digit_expr_is_4_l1766_176665


namespace NUMINAMATH_GPT_girl_attendance_l1766_176607

theorem girl_attendance (g b : ℕ) (h1 : g + b = 1500) (h2 : (3 / 4 : ℚ) * g + (1 / 3 : ℚ) * b = 900) :
  (3 / 4 : ℚ) * g = 720 :=
by
  sorry

end NUMINAMATH_GPT_girl_attendance_l1766_176607


namespace NUMINAMATH_GPT_jennifer_cards_left_l1766_176673

-- Define the initial number of cards and the number of cards eaten
def initial_cards : ℕ := 72
def eaten_cards : ℕ := 61

-- Define the final number of cards
def final_cards (initial_cards eaten_cards : ℕ) : ℕ :=
  initial_cards - eaten_cards

-- Proposition stating that Jennifer has 11 cards left
theorem jennifer_cards_left : final_cards initial_cards eaten_cards = 11 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_jennifer_cards_left_l1766_176673


namespace NUMINAMATH_GPT_percentage_increase_in_rectangle_area_l1766_176676

theorem percentage_increase_in_rectangle_area (L W : ℝ) :
  (1.35 * 1.35 * L * W - L * W) / (L * W) * 100 = 82.25 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_in_rectangle_area_l1766_176676


namespace NUMINAMATH_GPT_simplify_fractions_l1766_176683

theorem simplify_fractions :
  (36 / 51) * (35 / 24) * (68 / 49) = (20 / 7) :=
by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 51 = 3 * 17 := by norm_num
  have h3 : 35 = 5 * 7 := by norm_num
  have h4 : 24 = 2^3 * 3 := by norm_num
  have h5 : 68 = 2^2 * 17 := by norm_num
  have h6 : 49 = 7^2 := by norm_num
  sorry

end NUMINAMATH_GPT_simplify_fractions_l1766_176683


namespace NUMINAMATH_GPT_proof_problem_l1766_176675

def diamondsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem proof_problem :
  { (x, y) : ℝ × ℝ | diamondsuit x y = diamondsuit y x } =
  { (x, y) | x = 0 } ∪ { (x, y) | y = 0 } ∪ { (x, y) | x = y } ∪ { (x, y) | x = -y } :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1766_176675


namespace NUMINAMATH_GPT_interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l1766_176634

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem interval_increase_for_k_eq_2 :
  ∃ k : ℝ, k = 2 → 
  ∃ a b : ℝ, 0 < b ∧ b = 1 / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → (Real.log x - 2 * x + 1 < Real.log x - 2 * x + 1)) := 
sorry

theorem range_of_k_if_f_leq_0 :
  ∀ (k : ℝ), (∀ x : ℝ, 0 < x → Real.log x - k * x + 1 ≤ 0) →
  ∃ k_min : ℝ, k_min = 1 ∧ k ≥ k_min :=
sorry

end NUMINAMATH_GPT_interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l1766_176634


namespace NUMINAMATH_GPT_ratio_final_to_initial_l1766_176687

theorem ratio_final_to_initial (P R T : ℝ) (hR : R = 5) (hT : T = 20) :
  let SI := P * R * T / 100
  let A := P + SI
  A / P = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_final_to_initial_l1766_176687


namespace NUMINAMATH_GPT_train_speed_is_85_kmh_l1766_176674

noncomputable def speed_of_train_in_kmh (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_mps := speed_of_man_kmh * 1000 / 3600
  let relative_speed_mps := length_of_train / time_to_cross
  let speed_of_train_mps := relative_speed_mps - speed_of_man_mps
  speed_of_train_mps * 3600 / 1000

theorem train_speed_is_85_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (speed_of_man_kmh : ℝ)
  (h1 : length_of_train = 150)
  (h2 : time_to_cross = 6)
  (h3 : speed_of_man_kmh = 5) :
  speed_of_train_in_kmh length_of_train time_to_cross speed_of_man_kmh = 85 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_85_kmh_l1766_176674


namespace NUMINAMATH_GPT_find_sum_l1766_176640

def f (x : ℝ) : ℝ := sorry

axiom f_non_decreasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 ≤ 1 → 0 ≤ x2 → x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (x / 3) = (1 / 2) * f x
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (1 - x) = 1 - f x

theorem find_sum :
  f (1 / 3) + f (2 / 3) + f (1 / 9) + f (1 / 6) + f (1 / 8) = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l1766_176640


namespace NUMINAMATH_GPT_number_of_lawns_mowed_l1766_176613

noncomputable def ChargePerLawn : ℕ := 33
noncomputable def TotalTips : ℕ := 30
noncomputable def TotalEarnings : ℕ := 558

theorem number_of_lawns_mowed (L : ℕ) 
  (h1 : ChargePerLawn * L + TotalTips = TotalEarnings) : L = 16 := 
by
  sorry

end NUMINAMATH_GPT_number_of_lawns_mowed_l1766_176613


namespace NUMINAMATH_GPT_survey_steps_correct_l1766_176660

theorem survey_steps_correct :
  ∀ steps : (ℕ → ℕ), (steps 1 = 2) → (steps 2 = 4) → (steps 3 = 3) → (steps 4 = 1) → True :=
by
  intros steps h1 h2 h3 h4
  exact sorry

end NUMINAMATH_GPT_survey_steps_correct_l1766_176660


namespace NUMINAMATH_GPT_problem_inverse_range_m_l1766_176685

theorem problem_inverse_range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 / x + 1 / y = 1) : 
  (2 * x + y > m^2 + 8 * m) ↔ (m > -9 ∧ m < 1) := 
by
  sorry

end NUMINAMATH_GPT_problem_inverse_range_m_l1766_176685


namespace NUMINAMATH_GPT_proposition_statementC_l1766_176655

-- Definitions of each statement
def statementA := "Draw a parallel line to line AB"
def statementB := "Take a point C on segment AB"
def statementC := "The complement of equal angles are equal"
def statementD := "Is the perpendicular segment the shortest?"

-- Proving that among the statements A, B, C, and D, statement C is the proposition
theorem proposition_statementC : 
  (statementC = "The complement of equal angles are equal") :=
by
  -- We assume it directly from the equivalence given in the problem statement
  sorry

end NUMINAMATH_GPT_proposition_statementC_l1766_176655


namespace NUMINAMATH_GPT_problem_l1766_176659

-- Conditions
variables (x y : ℚ)
def condition1 := 3 * x + 5 = 12
def condition2 := 10 * y - 2 = 5

-- Theorem to prove
theorem problem (h1 : condition1 x) (h2 : condition2 y) : x + y = 91 / 30 := sorry

end NUMINAMATH_GPT_problem_l1766_176659


namespace NUMINAMATH_GPT_triangle_area_ratio_l1766_176623

-- Define parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : (ℝ × ℝ) := (2, 0)

-- Define the line passing through the focus and intersecting the parabola
def line_through_focus (f : ℝ × ℝ) (a b : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l (f.1) = f.2 ∧ parabola a.1 a.2 ∧ parabola b.1 b.2 ∧   -- line passes through the focus and intersects parabola at a and b
  l a.1 = a.2 ∧ l b.1 = b.2 ∧ 
  |a.1 - f.1| + |a.2 - f.2| = 3 ∧ -- condition |AF| = 3
  (f = (2, 0))

-- The proof problem
theorem triangle_area_ratio (a b : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_line : line_through_focus focus a b l) :
  ∃ r, r = (1 / 2) := 
sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1766_176623


namespace NUMINAMATH_GPT_min_max_value_expression_l1766_176656

theorem min_max_value_expression
  (x1 x2 x3 : ℝ) 
  (hx : x1 + x2 + x3 = 1)
  (hx1 : 0 ≤ x1)
  (hx2 : 0 ≤ x2)
  (hx3 : 0 ≤ x3) :
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5) = 1 := 
sorry

end NUMINAMATH_GPT_min_max_value_expression_l1766_176656


namespace NUMINAMATH_GPT_wall_height_correct_l1766_176697

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.2
def brick_width  : ℝ := 0.1
def brick_height : ℝ := 0.08

-- Define the volume of one brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Total number of bricks used
def number_of_bricks : ℕ := 12250

-- Define the wall dimensions except height
def wall_length : ℝ := 10
def wall_width  : ℝ := 24.5

-- Total volume of all bricks
def volume_total_bricks : ℝ := number_of_bricks * volume_brick

-- Volume of the wall
def volume_wall (h : ℝ) : ℝ := wall_length * h * wall_width

-- The height of the wall
def wall_height : ℝ := 0.08

-- The theorem to prove
theorem wall_height_correct : volume_total_bricks = volume_wall wall_height :=
by
  sorry

end NUMINAMATH_GPT_wall_height_correct_l1766_176697


namespace NUMINAMATH_GPT_square_inscribed_in_hexagon_has_side_length_l1766_176664

-- Definitions for the conditions given
noncomputable def side_length_square (AB EF : ℝ) : ℝ :=
  if AB = 30 ∧ EF = 19 * (Real.sqrt 3 - 1) then 10 * Real.sqrt 3 else 0

-- The theorem stating the specified equality
theorem square_inscribed_in_hexagon_has_side_length (AB EF : ℝ)
  (hAB : AB = 30) (hEF : EF = 19 * (Real.sqrt 3 - 1)) :
  side_length_square AB EF = 10 * Real.sqrt 3 := 
by 
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_square_inscribed_in_hexagon_has_side_length_l1766_176664


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1766_176682

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 540) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1766_176682


namespace NUMINAMATH_GPT_arrangements_of_masters_and_apprentices_l1766_176641

theorem arrangements_of_masters_and_apprentices : 
  ∃ n : ℕ, n = 48 ∧ 
     let pairs := 3 
     let ways_to_arrange_pairs := pairs.factorial 
     let ways_to_arrange_within_pairs := 2 ^ pairs 
     ways_to_arrange_pairs * ways_to_arrange_within_pairs = n := 
sorry

end NUMINAMATH_GPT_arrangements_of_masters_and_apprentices_l1766_176641


namespace NUMINAMATH_GPT_find_input_values_f_l1766_176615

theorem find_input_values_f (f : ℤ → ℤ) 
  (h_def : ∀ x, f (2 * x + 3) = (x - 3) * (x + 4))
  (h_val : ∃ y, f y = 170) : 
  ∃ (a b : ℤ), (a = -25 ∧ b = 29) ∧ (f a = 170 ∧ f b = 170) :=
by
  sorry

end NUMINAMATH_GPT_find_input_values_f_l1766_176615


namespace NUMINAMATH_GPT_number_of_ways_is_64_l1766_176672

-- Definition of the problem conditions
def ways_to_sign_up (students groups : ℕ) : ℕ :=
  groups ^ students

-- Theorem statement asserting that for 3 students and 4 groups, the number of ways is 64
theorem number_of_ways_is_64 : ways_to_sign_up 3 4 = 64 :=
by sorry

end NUMINAMATH_GPT_number_of_ways_is_64_l1766_176672


namespace NUMINAMATH_GPT_sum_of_values_l1766_176678

theorem sum_of_values (N : ℝ) (h : N * (N + 4) = 8) : N + (4 - N - 8 / N) = -4 := 
sorry

end NUMINAMATH_GPT_sum_of_values_l1766_176678


namespace NUMINAMATH_GPT_find_digits_for_divisibility_l1766_176643

theorem find_digits_for_divisibility (d1 d2 : ℕ) (h1 : d1 < 10) (h2 : d2 < 10) :
  (32 * 10^7 + d1 * 10^6 + 35717 * 10 + d2) % 72 = 0 →
  d1 = 2 ∧ d2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_digits_for_divisibility_l1766_176643


namespace NUMINAMATH_GPT_problem_statement_l1766_176649

def p (x : ℝ) : ℝ := x^2 - x + 1

theorem problem_statement (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1766_176649


namespace NUMINAMATH_GPT_unique_two_scoop_sundaes_l1766_176600

theorem unique_two_scoop_sundaes (n : ℕ) (hn : n = 8) : ∃ k, k = Nat.choose 8 2 :=
by
  use 28
  sorry

end NUMINAMATH_GPT_unique_two_scoop_sundaes_l1766_176600


namespace NUMINAMATH_GPT_find_alpha_l1766_176632

-- Define the problem in Lean terms
variable (x y α : ℝ)

-- Conditions
def condition1 : Prop := 3 + α + y = 4 + α + x
def condition2 : Prop := 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)

-- The theorem to prove
theorem find_alpha (h1 : condition1 x y α) (h2 : condition2 x y α) : α = 5 := 
  sorry

end NUMINAMATH_GPT_find_alpha_l1766_176632


namespace NUMINAMATH_GPT_product_of_terms_eq_72_l1766_176692

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end NUMINAMATH_GPT_product_of_terms_eq_72_l1766_176692


namespace NUMINAMATH_GPT_find_g_of_polynomial_l1766_176688

variable (x : ℝ)

theorem find_g_of_polynomial :
  ∃ g : ℝ → ℝ, (4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) → (g x = -2 * x^4 - 13 * x^3 + 7 * x + 4) :=
sorry

end NUMINAMATH_GPT_find_g_of_polynomial_l1766_176688


namespace NUMINAMATH_GPT_problem_l1766_176638

theorem problem (m : ℝ) (h : m + 1/m = 6) : m^2 + 1/m^2 + 3 = 37 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1766_176638


namespace NUMINAMATH_GPT_vector_magnitude_l1766_176691

noncomputable def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude :
  ∀ (x y : ℝ), let a := (x, 2)
               let b := (1, y)
               let c := (2, -6)
               (a.1 * c.1 + a.2 * c.2 = 0) →
               (b.1 * (-c.2) - b.2 * c.1 = 0) →
               magnitude (vec_add a b) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h₁ h₂
  let a := (x, 2)
  let b := (1, y)
  let c := (2, -6)
  sorry

end NUMINAMATH_GPT_vector_magnitude_l1766_176691


namespace NUMINAMATH_GPT_tank_width_problem_l1766_176670

noncomputable def tank_width (cost_per_sq_meter : ℚ) (total_cost : ℚ) (length depth : ℚ) : ℚ :=
  let total_cost_in_paise := total_cost * 100
  let total_area := total_cost_in_paise / cost_per_sq_meter
  let w := (total_area - (2 * length * depth) - (2 * depth * 6)) / (length + 2 * depth)
  w

theorem tank_width_problem :
  tank_width 55 409.20 25 6 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_tank_width_problem_l1766_176670


namespace NUMINAMATH_GPT_M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l1766_176658

noncomputable def M : ℕ → ℕ → ℕ → ℝ := sorry

theorem M_less_equal_fraction_M (n k h : ℕ) : 
  M n k h ≤ (n / h) * M (n-1) (k-1) (h-1) :=
sorry

theorem M_greater_equal_fraction_M (n k h : ℕ) : 
  M n k h ≥ (n / (n - h)) * M (n-1) k k :=
sorry

theorem M_less_equal_sum_M (n k h : ℕ) : 
  M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h :=
sorry

end NUMINAMATH_GPT_M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l1766_176658


namespace NUMINAMATH_GPT_second_number_l1766_176695

theorem second_number (A B : ℝ) (h1 : A = 200) (h2 : 0.30 * A = 0.60 * B + 30) : B = 50 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_second_number_l1766_176695


namespace NUMINAMATH_GPT_average_is_correct_l1766_176626

def nums : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_is_correct :
  (nums.sum / nums.length) = 125830.8 :=
by sorry

end NUMINAMATH_GPT_average_is_correct_l1766_176626


namespace NUMINAMATH_GPT_proof_problem_l1766_176605

noncomputable def problem_statement : Prop :=
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ 
    x1 + x2 + x3 + x4 = 8 ∧ 
    x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ 
    (x1 + x2) = 2 * 2 ∧ 
    (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 - 4 * 2 * (x1 + x2 + x3 + x4) + 4 * 4) = 4 ∧ 
    (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3)

theorem proof_problem : problem_statement :=
sorry

end NUMINAMATH_GPT_proof_problem_l1766_176605


namespace NUMINAMATH_GPT_blue_segments_count_l1766_176603

def grid_size : ℕ := 16
def total_dots : ℕ := grid_size * grid_size
def red_dots : ℕ := 133
def boundary_red_dots : ℕ := 32
def corner_red_dots : ℕ := 2
def yellow_segments : ℕ := 196

-- Dummy hypotheses representing the given conditions
axiom red_dots_on_grid : red_dots <= total_dots
axiom boundary_red_dots_count : boundary_red_dots = 32
axiom corner_red_dots_count : corner_red_dots = 2
axiom total_yellow_segments : yellow_segments = 196

-- Proving the number of blue line segments
theorem blue_segments_count :  ∃ (blue_segments : ℕ), blue_segments = 134 := 
sorry

end NUMINAMATH_GPT_blue_segments_count_l1766_176603


namespace NUMINAMATH_GPT_board_partition_possible_l1766_176680

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end NUMINAMATH_GPT_board_partition_possible_l1766_176680


namespace NUMINAMATH_GPT_triangle_land_area_l1766_176616

theorem triangle_land_area :
  let base_cm := 12
  let height_cm := 9
  let scale_cm_to_miles := 3
  let square_mile_to_acres := 640
  let area_cm2 := (1 / 2 : Float) * base_cm * height_cm
  let area_miles2 := area_cm2 * (scale_cm_to_miles ^ 2)
  let area_acres := area_miles2 * square_mile_to_acres
  area_acres = 311040 :=
by
  -- Skipped proofs
  sorry

end NUMINAMATH_GPT_triangle_land_area_l1766_176616


namespace NUMINAMATH_GPT_tom_candy_pieces_l1766_176608

/-!
# Problem Statement
Tom bought 14 boxes of chocolate candy, 10 boxes of fruit candy, and 8 boxes of caramel candy. 
He gave 8 chocolate boxes and 5 fruit boxes to his little brother. 
If each chocolate box has 3 pieces inside, each fruit box has 4 pieces, and each caramel box has 5 pieces, 
prove that Tom still has 78 pieces of candy.
-/

theorem tom_candy_pieces 
  (chocolate_boxes : ℕ := 14)
  (fruit_boxes : ℕ := 10)
  (caramel_boxes : ℕ := 8)
  (gave_away_chocolate_boxes : ℕ := 8)
  (gave_away_fruit_boxes : ℕ := 5)
  (chocolate_pieces_per_box : ℕ := 3)
  (fruit_pieces_per_box : ℕ := 4)
  (caramel_pieces_per_box : ℕ := 5)
  : chocolate_boxes * chocolate_pieces_per_box + 
    fruit_boxes * fruit_pieces_per_box + 
    caramel_boxes * caramel_pieces_per_box - 
    (gave_away_chocolate_boxes * chocolate_pieces_per_box + 
     gave_away_fruit_boxes * fruit_pieces_per_box) = 78 :=
by
  sorry

end NUMINAMATH_GPT_tom_candy_pieces_l1766_176608


namespace NUMINAMATH_GPT_children_working_initially_l1766_176661

theorem children_working_initially (W C : ℝ) (n : ℕ) 
  (h1 : 10 * W = 1 / 5) 
  (h2 : n * C = 1 / 10) 
  (h3 : 5 * W + 10 * C = 1 / 5) : 
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_children_working_initially_l1766_176661


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1766_176669

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 4 = 8)
  (h3 : ∀ k : ℕ, a (k + 1) = a k * q) : a 3 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1766_176669


namespace NUMINAMATH_GPT_grain_milling_l1766_176629

theorem grain_milling (A : ℚ) (h1 : 0.9 * A = 100) : A = 111 + 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_grain_milling_l1766_176629


namespace NUMINAMATH_GPT_advertisement_probability_l1766_176646

theorem advertisement_probability
  (ads_time_hour : ℕ)
  (total_time_hour : ℕ)
  (h1 : ads_time_hour = 20)
  (h2 : total_time_hour = 60) :
  ads_time_hour / total_time_hour = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_advertisement_probability_l1766_176646


namespace NUMINAMATH_GPT_minimum_value_f_x_l1766_176651

theorem minimum_value_f_x (x : ℝ) (h : 1 < x) : 
  x + (1 / (x - 1)) ≥ 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_x_l1766_176651


namespace NUMINAMATH_GPT_eval_abc_l1766_176642

theorem eval_abc (a b c : ℚ) (h1 : a = 1 / 2) (h2 : b = 3 / 4) (h3 : c = 8) :
  a^3 * b^2 * c = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_eval_abc_l1766_176642


namespace NUMINAMATH_GPT_area_of_right_triangle_ABC_l1766_176681

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def area_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) (hAB : AB = 30) (hBC : BC = 40) : ℝ :=
  1 / 2 * AB * BC

theorem area_of_right_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) 
  (hAB : AB = 30) (hBC : BC = 40) : 
  area_triangle_ABC AB BC angleB hangle hAB hBC = 600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_ABC_l1766_176681


namespace NUMINAMATH_GPT_sum_of_3_consecutive_multiples_of_3_l1766_176698

theorem sum_of_3_consecutive_multiples_of_3 (a b c : ℕ) (h₁ : a = b + 3) (h₂ : b = c + 3) (h₃ : a = 42) : a + b + c = 117 :=
by sorry

end NUMINAMATH_GPT_sum_of_3_consecutive_multiples_of_3_l1766_176698


namespace NUMINAMATH_GPT_water_heater_ratio_l1766_176611

variable (Wallace_capacity : ℕ) (Catherine_capacity : ℕ)
variable (Wallace_fullness : ℚ := 3/4) (Catherine_fullness : ℚ := 3/4)
variable (total_water : ℕ := 45)

theorem water_heater_ratio :
  Wallace_capacity = 40 →
  (Wallace_fullness * Wallace_capacity : ℚ) + (Catherine_fullness * Catherine_capacity : ℚ) = total_water →
  ((Wallace_capacity : ℚ) / (Catherine_capacity : ℚ)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_water_heater_ratio_l1766_176611


namespace NUMINAMATH_GPT_part1_part2_l1766_176645

noncomputable def f (a x : ℝ) : ℝ := a * x - a * Real.log x - Real.exp x / x

theorem part1 (a : ℝ) :
  (∀ x > 0, f a x < 0) → a < Real.exp 1 :=
sorry

theorem part2 (a : ℝ) (x1 x2 x3 : ℝ) :
  (∀ x, f a x = 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧
  f a x1 + f a x2 + f a x3 ≤ 3 * Real.exp 2 - Real.exp 1 →
  Real.exp 1 < a ∧ a ≤ Real.exp 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1766_176645


namespace NUMINAMATH_GPT_total_seeds_l1766_176648

-- Definitions and conditions
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds
def Eun_seeds : ℕ := 2 * Gwi_seeds

-- Theorem statement
theorem total_seeds : Bom_seeds + Gwi_seeds + Yeon_seeds + Eun_seeds = 2340 :=
by
  -- Skipping the proof steps with sorry
  sorry

end NUMINAMATH_GPT_total_seeds_l1766_176648


namespace NUMINAMATH_GPT_rectangles_in_grid_l1766_176671

-- Define a function that calculates the number of rectangles formed
def number_of_rectangles (n m : ℕ) : ℕ :=
  ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4

-- Prove that the number_of_rectangles function correctly calculates the number of rectangles given n and m 
theorem rectangles_in_grid (n m : ℕ) :
  number_of_rectangles n m = ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4 := 
by
  sorry

end NUMINAMATH_GPT_rectangles_in_grid_l1766_176671


namespace NUMINAMATH_GPT_compute_fg_neg_2_l1766_176622

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 4 * x + 4

theorem compute_fg_neg_2 : f (g (-2)) = -5 :=
by
-- sorry is used to skip the proof
sorry

end NUMINAMATH_GPT_compute_fg_neg_2_l1766_176622


namespace NUMINAMATH_GPT_original_price_l1766_176667

theorem original_price (x : ℝ) (h1 : x > 0) (h2 : 1.12 * x - x = 270) : x = 2250 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1766_176667


namespace NUMINAMATH_GPT_sum_even_then_diff_even_sum_odd_then_diff_odd_l1766_176657

theorem sum_even_then_diff_even (a b : ℤ) (h : (a + b) % 2 = 0) : (a - b) % 2 = 0 := by
  sorry

theorem sum_odd_then_diff_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a - b) % 2 = 1 := by
  sorry

end NUMINAMATH_GPT_sum_even_then_diff_even_sum_odd_then_diff_odd_l1766_176657


namespace NUMINAMATH_GPT_range_of_a_l1766_176644

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (2 * x + 2) - abs (2 * x - 2) ≤ a) ↔ 4 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1766_176644


namespace NUMINAMATH_GPT_correct_sample_size_l1766_176663

-- Definitions based on conditions:
def population_size : ℕ := 1800
def sample_size : ℕ := 1000
def surveyed_parents : ℕ := 1000

-- The proof statement we need: 
-- Prove that the sample size is 1000, given the surveyed parents are 1000
theorem correct_sample_size (ps : ℕ) (sp : ℕ) (ss : ℕ) (h1 : ps = population_size) (h2 : sp = surveyed_parents) : ss = sample_size :=
  sorry

end NUMINAMATH_GPT_correct_sample_size_l1766_176663


namespace NUMINAMATH_GPT_convert_to_spherical_l1766_176602

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if y / x < 0 then Real.arctan (-y / x) + 2 * Real.pi else Real.arctan (y / x)
  (ρ, θ, φ)

theorem convert_to_spherical :
  let x := 1
  let y := -4 * Real.sqrt 3
  let z := 4
  spherical_coordinates x y z = (Real.sqrt 65, Real.arctan (-4 * Real.sqrt 3) + 2 * Real.pi, Real.arccos (4 / (Real.sqrt 65))) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_spherical_l1766_176602


namespace NUMINAMATH_GPT_prove_statement_II_must_be_true_l1766_176690

-- Definitions of the statements
def statement_I (d : ℕ) : Prop := d = 5
def statement_II (d : ℕ) : Prop := d ≠ 6
def statement_III (d : ℕ) : Prop := d = 7
def statement_IV (d : ℕ) : Prop := d ≠ 8

-- Condition: Exactly three of these statements are true and one is false
def exactly_three_true (P Q R S : Prop) : Prop :=
  (P ∧ Q ∧ R ∧ ¬S) ∨ (P ∧ Q ∧ ¬R ∧ S) ∨ (P ∧ ¬Q ∧ R ∧ S) ∨ (¬P ∧ Q ∧ R ∧ S)

-- Problem statement
theorem prove_statement_II_must_be_true (d : ℕ) (h : exactly_three_true (statement_I d) (statement_II d) (statement_III d) (statement_IV d)) : 
  statement_II d :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_prove_statement_II_must_be_true_l1766_176690


namespace NUMINAMATH_GPT_area_of_quadrilateral_AXYD_l1766_176679

open Real

noncomputable def area_quadrilateral_AXYD: ℝ :=
  let A := (0, 0)
  let B := (20, 0)
  let C := (20, 12)
  let D := (0, 12)
  let Z := (20, 30)
  let E := (6, 6)
  let X := (2.5, 0)
  let Y := (9.5, 12)
  let base1 := (B.1 - X.1)  -- Length from B to X
  let base2 := (Y.1 - A.1)  -- Length from D to Y
  let height := (C.2 - A.2) -- Height common for both bases
  (base1 + base2) * height / 2

theorem area_of_quadrilateral_AXYD : area_quadrilateral_AXYD = 72 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_AXYD_l1766_176679


namespace NUMINAMATH_GPT_find_x0_range_l1766_176619

variable {x y x0 : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def angle_condition (x0 : ℝ) : Prop :=
  let OM := Real.sqrt (x0^2 + 3)
  OM ≤ 2

theorem find_x0_range (h1 : circle_eq x y) (h2 : angle_condition x0) :
  -1 ≤ x0 ∧ x0 ≤ 1 := 
sorry

end NUMINAMATH_GPT_find_x0_range_l1766_176619


namespace NUMINAMATH_GPT_regular_polygon_properties_l1766_176677

theorem regular_polygon_properties
  (exterior_angle : ℝ := 18) :
  (∃ (n : ℕ), n = 20) ∧ (∃ (interior_angle : ℝ), interior_angle = 162) := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_properties_l1766_176677


namespace NUMINAMATH_GPT_pencil_notebook_cost_l1766_176606

theorem pencil_notebook_cost (p n : ℝ)
  (h1 : 9 * p + 10 * n = 5.35)
  (h2 : 6 * p + 4 * n = 2.50) :
  24 * 0.9 * p + 15 * n = 9.24 :=
by 
  sorry

end NUMINAMATH_GPT_pencil_notebook_cost_l1766_176606


namespace NUMINAMATH_GPT_fraction_product_equals_12_l1766_176694

theorem fraction_product_equals_12 :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_equals_12_l1766_176694


namespace NUMINAMATH_GPT_Sum_a2_a3_a7_l1766_176666

-- Definitions from the conditions
variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers
variable {S : ℕ → ℝ} -- Define the sum of the first n terms as a function from natural numbers to real numbers

-- Given conditions
axiom Sn_formula : ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))
axiom S7_eq_42 : S 7 = 42

theorem Sum_a2_a3_a7 :
  a 2 + a 3 + a 7 = 18 :=
sorry

end NUMINAMATH_GPT_Sum_a2_a3_a7_l1766_176666


namespace NUMINAMATH_GPT_gcd_pow_of_subtraction_l1766_176636

noncomputable def m : ℕ := 2^2100 - 1
noncomputable def n : ℕ := 2^1950 - 1

theorem gcd_pow_of_subtraction : Nat.gcd m n = 2^150 - 1 :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_gcd_pow_of_subtraction_l1766_176636


namespace NUMINAMATH_GPT_number_of_pencils_purchased_l1766_176627

variable {total_pens : ℕ} (total_cost : ℝ) (avg_price_pencil avg_price_pen : ℝ)

theorem number_of_pencils_purchased 
  (h1 : total_pens = 30)
  (h2 : total_cost = 570)
  (h3 : avg_price_pencil = 2.00)
  (h4 : avg_price_pen = 14)
  : 
  ∃ P : ℕ, P = 75 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pencils_purchased_l1766_176627


namespace NUMINAMATH_GPT_system_infinite_solutions_a_eq_neg2_l1766_176624

theorem system_infinite_solutions_a_eq_neg2 
  (x y a : ℝ)
  (h1 : 2 * x + 2 * y = -1)
  (h2 : 4 * x + a^2 * y = a) 
  (infinitely_many_solutions : ∃ (a : ℝ), ∀ (c : ℝ), 4 * x + a^2 * y = c) :
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_system_infinite_solutions_a_eq_neg2_l1766_176624


namespace NUMINAMATH_GPT_ambiguous_dates_count_l1766_176601

theorem ambiguous_dates_count : 
  ∃ n : ℕ, n = 132 ∧ ∀ d m : ℕ, 1 ≤ d ∧ d ≤ 31 ∧ 1 ≤ m ∧ m ≤ 12 →
  ((d ≥ 1 ∧ d ≤ 12 ∧ m ≥ 1 ∧ m ≤ 12) → n = 132)
  :=
by 
  let ambiguous_days := 12 * 12
  let non_ambiguous_days := 12
  let total_ambiguous := ambiguous_days - non_ambiguous_days
  use total_ambiguous
  sorry

end NUMINAMATH_GPT_ambiguous_dates_count_l1766_176601


namespace NUMINAMATH_GPT_final_temperature_is_100_l1766_176610

-- Definitions based on conditions
def initial_temperature := 20  -- in degrees
def heating_rate := 5          -- in degrees per minute
def heating_time := 16         -- in minutes

-- The proof statement
theorem final_temperature_is_100 :
  initial_temperature + heating_rate * heating_time = 100 := by
  sorry

end NUMINAMATH_GPT_final_temperature_is_100_l1766_176610


namespace NUMINAMATH_GPT_sum_of_midpoint_coords_l1766_176628

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 3) (hy1 : y1 = 5) (hx2 : x2 = 11) (hy2 : y2 = 21) :
  ((x1 + x2) / 2 + (y1 + y2) / 2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coords_l1766_176628


namespace NUMINAMATH_GPT_max_branch_diameter_l1766_176647

theorem max_branch_diameter (d : ℝ) (w : ℝ) (angle : ℝ) (H: w = 1 ∧ angle = 90) :
  d ≤ 2 * Real.sqrt 2 + 2 := 
sorry

end NUMINAMATH_GPT_max_branch_diameter_l1766_176647
