import Mathlib

namespace NUMINAMATH_GPT_second_remainder_l561_56155

theorem second_remainder (n : ℕ) : n = 210 ∧ n % 13 = 3 → n % 17 = 6 :=
by
  sorry

end NUMINAMATH_GPT_second_remainder_l561_56155


namespace NUMINAMATH_GPT_evaluate_expression_l561_56123

-- Define the conditions
def num : ℤ := 900^2
def a : ℤ := 306
def b : ℤ := 294
def denom : ℤ := a^2 - b^2

-- State the theorem to be proven
theorem evaluate_expression : (num : ℚ) / denom = 112.5 :=
by
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_evaluate_expression_l561_56123


namespace NUMINAMATH_GPT_length_of_XY_l561_56135

theorem length_of_XY (A B C D P Q X Y : ℝ) (h₁ : A = B) (h₂ : C = D) 
  (h₃ : A + B = 13) (h₄ : C + D = 21) (h₅ : A + P = 7) 
  (h₆ : C + Q = 8) (h₇ : P ≠ Q) (h₈ : P + Q = 30) :
  ∃ k : ℝ, XY = 2 * k + 30 + 31 / 15 :=
by sorry

end NUMINAMATH_GPT_length_of_XY_l561_56135


namespace NUMINAMATH_GPT_kanul_cash_percentage_l561_56160

-- Define the conditions
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 1000
def total_amount : ℝ := 5714.29
def total_spent := raw_materials_cost + machinery_cost
def cash := total_amount - total_spent

-- The goal is to prove the percentage of the total amount as cash is 30%
theorem kanul_cash_percentage :
  (cash / total_amount) * 100 = 30 := 
sorry

end NUMINAMATH_GPT_kanul_cash_percentage_l561_56160


namespace NUMINAMATH_GPT_arithmetic_sequence_root_arithmetic_l561_56154

theorem arithmetic_sequence_root_arithmetic (a : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_root : ∀ x : ℝ, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) : 
  a 6 = -6 := 
by
  -- We skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_root_arithmetic_l561_56154


namespace NUMINAMATH_GPT_negation_of_proposition_l561_56112

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l561_56112


namespace NUMINAMATH_GPT_complex_plane_squares_areas_l561_56141

theorem complex_plane_squares_areas (z : ℂ) 
  (h1 : z^3 - z = i * (z^2 - z) ∨ z^3 - z = -i * (z^2 - z))
  (h2 : z^4 - z = i * (z^3 - z) ∨ z^4 - z = -i * (z^3 - z)) :
  ( ∃ A₁ A₂ : ℝ, (A₁ = 10 ∨ A₁ = 18) ∧ (A₂ = 10 ∨ A₂ = 18) ) := 
sorry

end NUMINAMATH_GPT_complex_plane_squares_areas_l561_56141


namespace NUMINAMATH_GPT_commute_time_l561_56144

theorem commute_time (start_time : ℕ) (first_station_time : ℕ) (work_time : ℕ) 
  (h1 : start_time = 6 * 60) 
  (h2 : first_station_time = 40) 
  (h3 : work_time = 9 * 60) : 
  work_time - (start_time + first_station_time) = 140 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_l561_56144


namespace NUMINAMATH_GPT_total_days_stayed_l561_56175

-- Definitions of given conditions as variables
def cost_first_week := 18
def days_first_week := 7
def cost_additional_week := 13
def total_cost := 334

-- Formulation of the target statement in Lean
theorem total_days_stayed :
  (days_first_week + 
  ((total_cost - (days_first_week * cost_first_week)) / cost_additional_week)) = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_days_stayed_l561_56175


namespace NUMINAMATH_GPT_find_integers_a_l561_56166

theorem find_integers_a (a : ℤ) : 
  (∃ n : ℤ, (a^3 + 1 = (a - 1) * n)) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := 
sorry

end NUMINAMATH_GPT_find_integers_a_l561_56166


namespace NUMINAMATH_GPT_monomials_like_terms_l561_56174

theorem monomials_like_terms (a b : ℤ) (h1 : a + 1 = 2) (h2 : b - 2 = 3) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_monomials_like_terms_l561_56174


namespace NUMINAMATH_GPT_max_ab_ac_bc_l561_56153

noncomputable def maxValue (a b c : ℝ) := a * b + a * c + b * c

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : maxValue a b c ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_ac_bc_l561_56153


namespace NUMINAMATH_GPT_five_digit_palindromes_count_l561_56183

def num_five_digit_palindromes : ℕ :=
  let choices_for_A := 9
  let choices_for_B := 10
  let choices_for_C := 10
  choices_for_A * choices_for_B * choices_for_C

theorem five_digit_palindromes_count : num_five_digit_palindromes = 900 :=
by
  unfold num_five_digit_palindromes
  sorry

end NUMINAMATH_GPT_five_digit_palindromes_count_l561_56183


namespace NUMINAMATH_GPT_cone_to_prism_volume_ratio_l561_56190

noncomputable def ratio_of_volumes (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) : ℝ :=
  let r := a / 2
  let V_cone := (1/3) * Real.pi * r^2 * h
  let V_prism := a * (2 * a) * h
  V_cone / V_prism

theorem cone_to_prism_volume_ratio (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) :
  ratio_of_volumes a h pos_a pos_h = Real.pi / 24 := by
  sorry

end NUMINAMATH_GPT_cone_to_prism_volume_ratio_l561_56190


namespace NUMINAMATH_GPT_solve_cubic_equation_l561_56180

theorem solve_cubic_equation (x y z : ℤ) (h : x^3 - 3*y^3 - 9*z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l561_56180


namespace NUMINAMATH_GPT_suraj_new_average_l561_56106

noncomputable def suraj_average (A : ℝ) : ℝ := A + 8

theorem suraj_new_average (A : ℝ) (h_conditions : 14 * A + 140 = 15 * (A + 8)) :
  suraj_average A = 28 :=
by
  sorry

end NUMINAMATH_GPT_suraj_new_average_l561_56106


namespace NUMINAMATH_GPT_complete_square_eqn_l561_56130

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_eqn_l561_56130


namespace NUMINAMATH_GPT_find_number_l561_56127

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 10) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l561_56127


namespace NUMINAMATH_GPT_determinant_property_l561_56150

variable {R : Type} [CommRing R]
variable (x y z w : R)

theorem determinant_property 
  (h : x * w - y * z = 7) :
  (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by sorry

end NUMINAMATH_GPT_determinant_property_l561_56150


namespace NUMINAMATH_GPT_find_side_a_l561_56171

noncomputable def side_a (b : ℝ) (A : ℝ) (S : ℝ) : ℝ :=
  2 * S / (b * Real.sin A)

theorem find_side_a :
  let b := 2
  let A := Real.pi * 2 / 3 -- 120 degrees in radians
  let S := 2 * Real.sqrt 3
  side_a b A S = 4 :=
by
  let b := 2
  let A := Real.pi * 2 / 3
  let S := 2 * Real.sqrt 3
  show side_a b A S = 4
  sorry

end NUMINAMATH_GPT_find_side_a_l561_56171


namespace NUMINAMATH_GPT_cars_sold_first_day_l561_56107

theorem cars_sold_first_day (c_2 c_3 : ℕ) (total : ℕ) (h1 : c_2 = 16) (h2 : c_3 = 27) (h3 : total = 57) :
  ∃ c_1 : ℕ, c_1 + c_2 + c_3 = total ∧ c_1 = 14 :=
by
  sorry

end NUMINAMATH_GPT_cars_sold_first_day_l561_56107


namespace NUMINAMATH_GPT_angle_C_is_80_l561_56172

-- Define the angles A, B, and C
def isoscelesTriangle (A B C : ℕ) : Prop :=
  -- Triangle ABC is isosceles with A = B, and C is 30 degrees more than A
  A = B ∧ C = A + 30 ∧ A + B + C = 180

-- Problem: Prove that angle C is 80 degrees given the conditions
theorem angle_C_is_80 (A B C : ℕ) (h : isoscelesTriangle A B C) : C = 80 :=
by sorry

end NUMINAMATH_GPT_angle_C_is_80_l561_56172


namespace NUMINAMATH_GPT_first_plot_germination_rate_l561_56169

-- Define the known quantities and conditions
def plot1_seeds : ℕ := 300
def plot2_seeds : ℕ := 200
def plot2_germination_rate : ℚ := 35 / 100
def total_germination_percentage : ℚ := 26 / 100

-- Define a statement to prove the percentage of seeds that germinated in the first plot
theorem first_plot_germination_rate : 
  ∃ (x : ℚ), (x / 100) * plot1_seeds + (plot2_germination_rate * plot2_seeds) = total_germination_percentage * (plot1_seeds + plot2_seeds) ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_first_plot_germination_rate_l561_56169


namespace NUMINAMATH_GPT_approximation_irrational_quotient_l561_56138

theorem approximation_irrational_quotient 
  (r1 r2 : ℝ) (irrational : ¬ ∃ q : ℚ, r1 = q * r2) 
  (x : ℝ) (p : ℝ) (pos_p : p > 0) : 
  ∃ (k1 k2 : ℤ), |x - (k1 * r1 + k2 * r2)| < p :=
sorry

end NUMINAMATH_GPT_approximation_irrational_quotient_l561_56138


namespace NUMINAMATH_GPT_tetrahedron_solution_l561_56139

noncomputable def num_triangles (a : ℝ) (E F G : ℝ → ℝ → ℝ) : ℝ :=
  if a > 3 then 3 else 0

theorem tetrahedron_solution (a : ℝ) (E F G : ℝ → ℝ → ℝ) :
  a > 3 → num_triangles a E F G = 3 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_solution_l561_56139


namespace NUMINAMATH_GPT_triangle_inequality_l561_56188
-- Import necessary libraries

-- Define the problem
theorem triangle_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (α β γ : ℝ) (h_alpha : α = 2 * Real.sqrt (b * c)) (h_beta : β = 2 * Real.sqrt (c * a)) (h_gamma : γ = 2 * Real.sqrt (a * b)) :
  (a / α) + (b / β) + (c / γ) ≥ (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l561_56188


namespace NUMINAMATH_GPT_fitness_club_alpha_is_more_advantageous_l561_56149

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end NUMINAMATH_GPT_fitness_club_alpha_is_more_advantageous_l561_56149


namespace NUMINAMATH_GPT_johns_personal_payment_l561_56124

theorem johns_personal_payment 
  (cost_per_hearing_aid : ℕ)
  (num_hearing_aids : ℕ)
  (deductible : ℕ)
  (coverage_percent : ℕ)
  (coverage_limit : ℕ) 
  (total_payment : ℕ)
  (insurance_payment_over_limit : ℕ) : 
  cost_per_hearing_aid = 2500 ∧ 
  num_hearing_aids = 2 ∧ 
  deductible = 500 ∧ 
  coverage_percent = 80 ∧ 
  coverage_limit = 3500 →
  total_payment = cost_per_hearing_aid * num_hearing_aids - deductible →
  insurance_payment_over_limit = max 0 (coverage_percent * total_payment / 100 - coverage_limit) →
  (total_payment - min (coverage_percent * total_payment / 100) coverage_limit + deductible = 1500) :=
by
  intros
  sorry

end NUMINAMATH_GPT_johns_personal_payment_l561_56124


namespace NUMINAMATH_GPT_converse_proposition_l561_56110

-- Define a proposition for vertical angles
def vertical_angles (α β : ℕ) : Prop := α = β

-- Define the converse of the vertical angle proposition
def converse_vertical_angles (α β : ℕ) : Prop := β = α

-- Prove that the converse of "Vertical angles are equal" is 
-- "Angles that are equal are vertical angles"
theorem converse_proposition (α β : ℕ) : vertical_angles α β ↔ converse_vertical_angles α β :=
by
  sorry

end NUMINAMATH_GPT_converse_proposition_l561_56110


namespace NUMINAMATH_GPT_solve_problem_1_solve_problem_2_l561_56196

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_1_solve_problem_2_l561_56196


namespace NUMINAMATH_GPT_train_cross_signal_pole_in_18_seconds_l561_56116

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 550
noncomputable def crossing_time_platform : ℝ := 51
noncomputable def signal_pole_crossing_time : ℝ := 18

theorem train_cross_signal_pole_in_18_seconds (t l_p t_p t_s : ℝ)
    (h1 : t = train_length)
    (h2 : l_p = platform_length)
    (h3 : t_p = crossing_time_platform)
    (h4 : t_s = signal_pole_crossing_time) : 
    (t + l_p) / t_p = train_length / signal_pole_crossing_time :=
by
  unfold train_length platform_length crossing_time_platform signal_pole_crossing_time at *
  -- proof will go here
  sorry

end NUMINAMATH_GPT_train_cross_signal_pole_in_18_seconds_l561_56116


namespace NUMINAMATH_GPT_line_equation_l561_56125

theorem line_equation (a T : ℝ) (h : 0 < a ∧ 0 < T) :
  ∃ (x y : ℝ), (2 * T * x - a^2 * y + 2 * a * T = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l561_56125


namespace NUMINAMATH_GPT_find_n_l561_56119

variable (n : ℚ)

theorem find_n (h : (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4)) : 
  n = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l561_56119


namespace NUMINAMATH_GPT_negation_proposition_l561_56195

theorem negation_proposition : ∀ (a : ℝ), (a > 3) → (a^2 ≥ 9) :=
by
  intros a ha
  sorry

end NUMINAMATH_GPT_negation_proposition_l561_56195


namespace NUMINAMATH_GPT_compute_expression_l561_56133

-- Define the operation a Δ b
def Delta (a b : ℝ) : ℝ := a^2 - 2 * b

theorem compute_expression :
  let x := 3 ^ (Delta 4 10)
  let y := 4 ^ (Delta 2 3)
  Delta x y = ( -819.125 / 6561) :=
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l561_56133


namespace NUMINAMATH_GPT_main_theorem_l561_56145

-- Let x be a real number
variable {x : ℝ}

-- Define the given identity
def identity (M₁ M₂ : ℝ) : Prop :=
  ∀ x, (50 * x - 42) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)

-- The proposition to prove the numerical value of M₁M₂
def prove_M1M2_value : Prop :=
  ∀ (M₁ M₂ : ℝ), identity M₁ M₂ → M₁ * M₂ = -6264

theorem main_theorem : prove_M1M2_value :=
  sorry

end NUMINAMATH_GPT_main_theorem_l561_56145


namespace NUMINAMATH_GPT_crayon_ratio_l561_56187

theorem crayon_ratio :
  ∀ (Karen Beatrice Gilbert Judah : ℕ),
    Karen = 128 →
    Beatrice = Karen / 2 →
    Beatrice = Gilbert →
    Gilbert = 4 * Judah →
    Judah = 8 →
    Beatrice / Gilbert = 1 :=
by
  intros Karen Beatrice Gilbert Judah hKaren hBeatrice hEqual hGilbert hJudah
  sorry

end NUMINAMATH_GPT_crayon_ratio_l561_56187


namespace NUMINAMATH_GPT_car_distance_covered_l561_56192

def distance_covered_by_car (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time

theorem car_distance_covered :
  distance_covered_by_car (3 + 1/5 : ℝ) 195 = 624 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_covered_l561_56192


namespace NUMINAMATH_GPT_arjun_starting_amount_l561_56167

theorem arjun_starting_amount (X : ℝ) (h1 : Anoop_investment = 4000) (h2 : Anoop_months = 6) (h3 : Arjun_months = 12) (h4 : (X * 12) = (4000 * 6)) :
  X = 2000 :=
sorry

end NUMINAMATH_GPT_arjun_starting_amount_l561_56167


namespace NUMINAMATH_GPT_community_members_after_five_years_l561_56189

theorem community_members_after_five_years:
  ∀ (a : ℕ → ℕ),
  a 0 = 20 →
  (∀ k : ℕ, a (k + 1) = 4 * a k - 15) →
  a 5 = 15365 :=
by
  intros a h₀ h₁
  sorry

end NUMINAMATH_GPT_community_members_after_five_years_l561_56189


namespace NUMINAMATH_GPT_trajectory_of_M_l561_56132

-- Define the two circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for the moving circle M being tangent to both circles
def isTangent (Mx My : ℝ) : Prop := 
  let distC1 := (Mx + 3)^2 + My^2
  let distC2 := (Mx - 3)^2 + My^2
  distC2 - distC1 = 4

-- The equation of the trajectory of M
theorem trajectory_of_M (Mx My : ℝ) (h : isTangent Mx My) : 
  Mx^2 - (My^2 / 8) = 1 ∧ Mx < 0 :=
sorry

end NUMINAMATH_GPT_trajectory_of_M_l561_56132


namespace NUMINAMATH_GPT_total_weight_kg_l561_56182

def envelope_weight_grams : ℝ := 8.5
def num_envelopes : ℝ := 800

theorem total_weight_kg : (envelope_weight_grams * num_envelopes) / 1000 = 6.8 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_kg_l561_56182


namespace NUMINAMATH_GPT_first_meet_at_starting_point_l561_56173

-- Definitions
def track_length := 300
def speed_A := 2
def speed_B := 4

-- Theorem: A and B will meet at the starting point for the first time after 400 seconds.
theorem first_meet_at_starting_point : 
  (∃ (t : ℕ), t = 400 ∧ (
    (∃ (n : ℕ), n * (track_length * (speed_B - speed_A)) = t * (speed_A + speed_B) * track_length) ∨
    (∃ (m : ℕ), m * (track_length * (speed_B + speed_A)) = t * (speed_A - speed_B) * track_length))) := 
    sorry

end NUMINAMATH_GPT_first_meet_at_starting_point_l561_56173


namespace NUMINAMATH_GPT_football_team_practice_missed_days_l561_56142

theorem football_team_practice_missed_days 
(daily_practice_hours : ℕ) 
(total_practice_hours : ℕ) 
(days_in_week : ℕ) 
(h1 : daily_practice_hours = 5) 
(h2 : total_practice_hours = 30) 
(h3 : days_in_week = 7) : 
days_in_week - (total_practice_hours / daily_practice_hours) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_football_team_practice_missed_days_l561_56142


namespace NUMINAMATH_GPT_multiple_solutions_no_solution_2891_l561_56194

theorem multiple_solutions (n : ℤ) (x y : ℤ) (h1 : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (u v : ℤ), u ≠ x ∧ v ≠ y ∧ u^3 - 3 * u * v^2 + v^3 = n :=
  sorry

theorem no_solution_2891 (x y : ℤ) (h2 : x^3 - 3 * x * y^2 + y^3 = 2891) :
  false :=
  sorry

end NUMINAMATH_GPT_multiple_solutions_no_solution_2891_l561_56194


namespace NUMINAMATH_GPT_total_money_l561_56159

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end NUMINAMATH_GPT_total_money_l561_56159


namespace NUMINAMATH_GPT_sector_area_is_correct_l561_56158

noncomputable def area_of_sector (r : ℝ) (α : ℝ) : ℝ := 1/2 * α * r^2

theorem sector_area_is_correct (circumference : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) 
  (h1 : circumference = 8) 
  (h2 : central_angle = 2) 
  (h3 : circumference = central_angle * r + 2 * r)
  (h4 : r = 2) : area = 4 :=
by
  have h5: area = 1/2 * central_angle * r^2 := sorry
  exact sorry

end NUMINAMATH_GPT_sector_area_is_correct_l561_56158


namespace NUMINAMATH_GPT_usual_time_to_cover_distance_l561_56177

variable (S T : ℝ)

-- Conditions:
-- 1. The man walks at 40% of his usual speed.
-- 2. He takes 24 minutes more to cover the same distance at this reduced speed.
-- 3. Usual speed is S.
-- 4. Usual time to cover the distance is T.

def usual_speed := S
def usual_time := T
def reduced_speed := 0.4 * S
def extra_time := 24

-- Question: Prove the man's usual time to cover the distance is 16 minutes.
theorem usual_time_to_cover_distance : T = 16 := 
by
  have speed_relation : S / (0.4 * S) = (T + 24) / T :=
    sorry
  have simplified_speed_relation : 2.5 = (T + 24) / T :=
    sorry
  have cross_multiplication_step : 2.5 * T = T + 24 :=
    sorry
  have solve_for_T_step : 1.5 * T = 24 :=
    sorry
  have final_step : T = 16 :=
    sorry
  exact final_step

end NUMINAMATH_GPT_usual_time_to_cover_distance_l561_56177


namespace NUMINAMATH_GPT_simplest_quadratic_radical_l561_56121

noncomputable def optionA := Real.sqrt 7
noncomputable def optionB := Real.sqrt 9
noncomputable def optionC := Real.sqrt 12
noncomputable def optionD := Real.sqrt (2 / 3)

theorem simplest_quadratic_radical :
  optionA = Real.sqrt 7 ∧
  optionB = Real.sqrt 9 ∧
  optionC = Real.sqrt 12 ∧
  optionD = Real.sqrt (2 / 3) ∧
  (optionB = 3 ∧ optionC = 2 * Real.sqrt 3 ∧ optionD = Real.sqrt 6 / 3) ∧
  (optionA < 3 ∧ optionA < 2 * Real.sqrt 3 ∧ optionA < Real.sqrt 6 / 3) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_simplest_quadratic_radical_l561_56121


namespace NUMINAMATH_GPT_joe_time_to_store_l561_56181

theorem joe_time_to_store :
  ∀ (r_w : ℝ) (r_r : ℝ) (t_w t_r t_total : ℝ), 
   (r_r = 2 * r_w) → (t_w = 10) → (t_r = t_w / 2) → (t_total = t_w + t_r) → (t_total = 15) := 
by
  intros r_w r_r t_w t_r t_total hrw hrw_eq hr_tw hr_t_total
  sorry

end NUMINAMATH_GPT_joe_time_to_store_l561_56181


namespace NUMINAMATH_GPT_cartons_loaded_l561_56165

def total_cartons : Nat := 50
def cans_per_carton : Nat := 20
def cans_left_to_load : Nat := 200

theorem cartons_loaded (C : Nat) (h : cans_per_carton ≠ 0) : 
  C = total_cartons - (cans_left_to_load / cans_per_carton) := by
  sorry

end NUMINAMATH_GPT_cartons_loaded_l561_56165


namespace NUMINAMATH_GPT_intersection_P_Q_l561_56128

-- Defining the two sets P and Q
def P := { x : ℤ | abs x ≤ 2 }
def Q := { x : ℝ | -1 < x ∧ x < 5/2 }

-- Statement to prove
theorem intersection_P_Q : 
  { x : ℤ | abs x ≤ 2 } ∩ { x : ℤ | -1 < ((x : ℝ)) ∧ ((x : ℝ)) < 5/2 } = {0, 1, 2} := sorry

end NUMINAMATH_GPT_intersection_P_Q_l561_56128


namespace NUMINAMATH_GPT_carrie_remaining_money_l561_56101

def initial_money : ℝ := 200
def sweater_cost : ℝ := 36
def tshirt_cost : ℝ := 12
def tshirt_discount : ℝ := 0.10
def shoes_cost : ℝ := 45
def jeans_cost : ℝ := 52
def scarf_cost : ℝ := 18
def sales_tax_rate : ℝ := 0.05

-- Calculate tshirt price after discount
def tshirt_final_price : ℝ := tshirt_cost * (1 - tshirt_discount)

-- Sum all the item costs before tax
def total_cost_before_tax : ℝ := sweater_cost + tshirt_final_price + shoes_cost + jeans_cost + scarf_cost

-- Calculate the total sales tax
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

-- Calculate total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

-- Calculate the remaining money
def remaining_money (initial : ℝ) (total : ℝ) : ℝ := initial - total

theorem carrie_remaining_money
  (initial_money : ℝ)
  (sweater_cost : ℝ)
  (tshirt_cost : ℝ)
  (tshirt_discount : ℝ)
  (shoes_cost : ℝ)
  (jeans_cost : ℝ)
  (scarf_cost : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : initial_money = 200)
  (h₂ : sweater_cost = 36)
  (h₃ : tshirt_cost = 12)
  (h₄ : tshirt_discount = 0.10)
  (h₅ : shoes_cost = 45)
  (h₆ : jeans_cost = 52)
  (h₇ : scarf_cost = 18)
  (h₈ : sales_tax_rate = 0.05) :
  remaining_money initial_money (total_cost_after_tax) = 30.11 := 
by 
  simp only [remaining_money, total_cost_after_tax, total_cost_before_tax, tshirt_final_price, sales_tax];
  sorry

end NUMINAMATH_GPT_carrie_remaining_money_l561_56101


namespace NUMINAMATH_GPT_fg_of_2_eq_81_l561_56170

def f (x : ℝ) : ℝ := x ^ 2
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1

theorem fg_of_2_eq_81 : f (g 2) = 81 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_eq_81_l561_56170


namespace NUMINAMATH_GPT_percentage_increase_on_bought_price_l561_56184

-- Define the conditions as Lean definitions
def original_price (P : ℝ) : ℝ := P
def bought_price (P : ℝ) : ℝ := 0.90 * P
def selling_price (P : ℝ) : ℝ := 1.62000000000000014 * P

-- Lean statement to prove the required result
theorem percentage_increase_on_bought_price (P : ℝ) :
  (selling_price P - bought_price P) / bought_price P * 100 = 80.00000000000002 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_on_bought_price_l561_56184


namespace NUMINAMATH_GPT_extinction_probability_l561_56157

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_extinction_probability_l561_56157


namespace NUMINAMATH_GPT_rational_solutions_k_values_l561_56109

theorem rational_solutions_k_values (k : ℕ) (h₁ : k > 0) 
    (h₂ : ∃ (m : ℤ), 900 - 4 * (k:ℤ)^2 = m^2) : k = 9 ∨ k = 15 := 
by
  sorry

end NUMINAMATH_GPT_rational_solutions_k_values_l561_56109


namespace NUMINAMATH_GPT_quotient_remainder_scaled_l561_56104

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end NUMINAMATH_GPT_quotient_remainder_scaled_l561_56104


namespace NUMINAMATH_GPT_largest_negative_is_l561_56129

def largest_of_negatives (a b c d : ℚ) (largest : ℚ) : Prop := largest = max (max a b) (max c d)

theorem largest_negative_is (largest : ℚ) : largest_of_negatives (-2/3) (-2) (-1) (-5) largest → largest = -2/3 :=
by
  intro h
  -- We assume the definition and the theorem are sufficient to say largest = -2/3
  sorry

end NUMINAMATH_GPT_largest_negative_is_l561_56129


namespace NUMINAMATH_GPT_smallest_k_l561_56114

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l561_56114


namespace NUMINAMATH_GPT_select_students_l561_56191

-- Definitions for the conditions
variables (A B C D E : Prop)

-- Conditions
def condition1 : Prop := A → B ∧ ¬E
def condition2 : Prop := (B ∨ E) → ¬D
def condition3 : Prop := C ∨ D

-- The main theorem
theorem select_students (hA : A) (h1 : condition1 A B E) (h2 : condition2 B E D) (h3 : condition3 C D) : B ∧ C :=
by 
  sorry

end NUMINAMATH_GPT_select_students_l561_56191


namespace NUMINAMATH_GPT_diamond_19_98_l561_56156

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end NUMINAMATH_GPT_diamond_19_98_l561_56156


namespace NUMINAMATH_GPT_fourth_root_cubed_eq_729_l561_56118

theorem fourth_root_cubed_eq_729 (x : ℝ) (hx : (x^(1/4))^3 = 729) : x = 6561 :=
  sorry

end NUMINAMATH_GPT_fourth_root_cubed_eq_729_l561_56118


namespace NUMINAMATH_GPT_fraction_product_l561_56102

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_l561_56102


namespace NUMINAMATH_GPT_least_three_digit_eleven_heavy_l561_56186

def isElevenHeavy (n : ℕ) : Prop :=
  n % 11 > 6

theorem least_three_digit_eleven_heavy : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ isElevenHeavy n ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ isElevenHeavy m) → n ≤ m :=
sorry

end NUMINAMATH_GPT_least_three_digit_eleven_heavy_l561_56186


namespace NUMINAMATH_GPT_unique_solution_single_element_l561_56140

theorem unique_solution_single_element (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x^2 + a * x + 1 = 0) → (a * y^2 + a * y + 1 = 0) → x = y) : a = 4 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_single_element_l561_56140


namespace NUMINAMATH_GPT_sequence_value_l561_56117

theorem sequence_value (a b c d x : ℕ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 17) (h4 : d = 33)
  (h5 : b - a = 4) (h6 : c - b = 8) (h7 : d - c = 16) (h8 : x - d = 32) : x = 65 := by
  sorry

end NUMINAMATH_GPT_sequence_value_l561_56117


namespace NUMINAMATH_GPT_average_speed_of_Car_X_l561_56168

noncomputable def average_speed_CarX (V_x : ℝ) : Prop :=
  let head_start_time := 1.2
  let distance_traveled_by_CarX := 98
  let speed_CarY := 50
  let time_elapsed := distance_traveled_by_CarX / speed_CarY
  (distance_traveled_by_CarX / time_elapsed) = V_x

theorem average_speed_of_Car_X : average_speed_CarX 50 :=
  sorry

end NUMINAMATH_GPT_average_speed_of_Car_X_l561_56168


namespace NUMINAMATH_GPT_inequality_of_function_inequality_l561_56193

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + Real.sqrt (x^2 + 1))) + 2 * x + Real.sin x

theorem inequality_of_function_inequality (x1 x2 : ℝ) (h : f x1 + f x2 > 0) : x1 + x2 > 0 :=
sorry

end NUMINAMATH_GPT_inequality_of_function_inequality_l561_56193


namespace NUMINAMATH_GPT_solution_contains_non_zero_arrays_l561_56115

noncomputable def verify_non_zero_array (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + 
  (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

theorem solution_contains_non_zero_arrays (x y z w : ℝ) (non_zero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) :
  verify_non_zero_array x y z w ↔ 
  (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) ∧
  (if x = -1 then y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if y = -2 then x ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if z = -3 then x ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0 else 
   x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :=
sorry

end NUMINAMATH_GPT_solution_contains_non_zero_arrays_l561_56115


namespace NUMINAMATH_GPT_find_naturals_for_divisibility_l561_56108

theorem find_naturals_for_divisibility (n : ℕ) (h1 : 3 * n ≠ 1) :
  (∃ k : ℤ, 7 * n + 5 = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_naturals_for_divisibility_l561_56108


namespace NUMINAMATH_GPT_weight_difference_l561_56198

open Real

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h1 : (W_A + W_B + W_C) / 3 = 50)
  (h2 : W_A = 73)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 53)
  (h4 : (W_B + W_C + W_D + W_E) / 4 = 51) :
  W_E - W_D = 3 := 
sorry

end NUMINAMATH_GPT_weight_difference_l561_56198


namespace NUMINAMATH_GPT_minimum_value_is_138_l561_56111

-- Definition of problem conditions and question
def is_digit (n : ℕ) : Prop := n < 10
def digits (A : ℕ) : List ℕ := A.digits 10

def multiple_of_3_not_9 (A : ℕ) : Prop :=
  A % 3 = 0 ∧ A % 9 ≠ 0

def product_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· * ·) 1

def sum_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· + ·) 0

def given_condition (A : ℕ) : Prop :=
  A % 9 = 0 → False ∧
  (A + product_of_digits A) % 9 = 0

-- Main goal: Prove that the minimum value A == 138 satisfies the given conditions
theorem minimum_value_is_138 : ∃ A, A = 138 ∧
  multiple_of_3_not_9 A ∧
  given_condition A :=
sorry

end NUMINAMATH_GPT_minimum_value_is_138_l561_56111


namespace NUMINAMATH_GPT_regression_line_zero_corr_l561_56197

-- Definitions based on conditions
variables {X Y : Type}
variables [LinearOrder X] [LinearOrder Y]
variables {f : X → Y}  -- representing the regression line

-- Condition: Regression coefficient b = 0
def regression_coefficient_zero (b : ℝ) : Prop := b = 0

-- Definition of correlation coefficient; here symbolically represented since full derivation requires in-depth statistics definitions
def correlation_coefficient (r : ℝ) : ℝ := r

-- The mathematical goal to prove
theorem regression_line_zero_corr {b r : ℝ} 
  (hb : regression_coefficient_zero b) : correlation_coefficient r = 0 := 
by
  sorry

end NUMINAMATH_GPT_regression_line_zero_corr_l561_56197


namespace NUMINAMATH_GPT_point_A_in_second_quadrant_l561_56131

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end NUMINAMATH_GPT_point_A_in_second_quadrant_l561_56131


namespace NUMINAMATH_GPT_average_speed_of_trip_is_correct_l561_56162

-- Definitions
def total_distance : ℕ := 450
def distance_part1 : ℕ := 300
def speed_part1 : ℕ := 20
def distance_part2 : ℕ := 150
def speed_part2 : ℕ := 15

-- The average speed problem
theorem average_speed_of_trip_is_correct :
  (total_distance : ℤ) / (distance_part1 / speed_part1 + distance_part2 / speed_part2 : ℤ) = 18 := by
  sorry

end NUMINAMATH_GPT_average_speed_of_trip_is_correct_l561_56162


namespace NUMINAMATH_GPT_find_V_y_l561_56164

-- Define the volumes and percentages given in the problem
def V_x : ℕ := 300
def percent_x : ℝ := 0.10
def percent_y : ℝ := 0.30
def desired_percent : ℝ := 0.22

-- Define the alcohol volumes in the respective solutions
def alcohol_x := percent_x * V_x
def total_volume (V_y : ℕ) := V_x + V_y
def desired_alcohol (V_y : ℕ) := desired_percent * (total_volume V_y)

-- Define our main statement
theorem find_V_y : ∃ (V_y : ℕ), alcohol_x + (percent_y * V_y) = desired_alcohol V_y ∧ V_y = 450 :=
by
  sorry

end NUMINAMATH_GPT_find_V_y_l561_56164


namespace NUMINAMATH_GPT_rectangle_area_l561_56199

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l561_56199


namespace NUMINAMATH_GPT_geometric_sum_l561_56105

theorem geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
    (h1 : S 3 = 8)
    (h2 : S 6 = 7)
    (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 7 + a 8 + a 9 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_l561_56105


namespace NUMINAMATH_GPT_train_crossing_time_l561_56120

theorem train_crossing_time
    (train_speed_kmph : ℕ)
    (platform_length_meters : ℕ)
    (crossing_time_platform_seconds : ℕ)
    (crossing_time_man_seconds : ℕ)
    (train_speed_mps : ℤ)
    (train_length_meters : ℤ)
    (T : ℤ)
    (h1 : train_speed_kmph = 72)
    (h2 : platform_length_meters = 340)
    (h3 : crossing_time_platform_seconds = 35)
    (h4 : train_speed_mps = 20)
    (h5 : train_length_meters = 360)
    (h6 : train_length_meters = train_speed_mps * crossing_time_man_seconds)
    : T = 18 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l561_56120


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l561_56103

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l561_56103


namespace NUMINAMATH_GPT_all_three_selected_l561_56179

-- Define the probabilities
def P_R : ℚ := 6 / 7
def P_Rv : ℚ := 1 / 5
def P_Rs : ℚ := 2 / 3
def P_Rv_given_R : ℚ := 2 / 5
def P_Rs_given_Rv : ℚ := 1 / 2

-- The probability that all three are selected
def P_all : ℚ := P_R * P_Rv_given_R * P_Rs_given_Rv

-- Prove that the calculated probability is equal to the given answer
theorem all_three_selected : P_all = 6 / 35 :=
by
  sorry

end NUMINAMATH_GPT_all_three_selected_l561_56179


namespace NUMINAMATH_GPT_find_x_from_ratio_l561_56152

theorem find_x_from_ratio (x y k: ℚ) 
  (h1 : ∀ x y, (5 * x - 3) / (y + 20) = k) 
  (h2 : 5 * 1 - 3 = 2 * 22) (hy : y = 5) : 
  x = 58 / 55 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_from_ratio_l561_56152


namespace NUMINAMATH_GPT_hyperbola_asymptotes_and_point_l561_56163

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_asymptotes_and_point 
  (x y : ℝ)
  (asymptote1 : ∀ x, y = (1/2) * x)
  (asymptote2 : ∀ x, y = (-1/2) * x)
  (point : (x, y) = (4, Real.sqrt 2))
: hyperbola_equation x y :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_and_point_l561_56163


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt_3_div_2_l561_56178

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt_3_div_2_l561_56178


namespace NUMINAMATH_GPT_pages_left_to_read_correct_l561_56151

def total_pages : Nat := 563
def pages_read : Nat := 147
def pages_left_to_read : Nat := 416

theorem pages_left_to_read_correct : total_pages - pages_read = pages_left_to_read := by
  sorry

end NUMINAMATH_GPT_pages_left_to_read_correct_l561_56151


namespace NUMINAMATH_GPT_total_number_of_edges_in_hexahedron_is_12_l561_56185

-- Define a hexahedron
structure Hexahedron where
  face_count : Nat
  edges_per_face : Nat
  edge_sharing : Nat

-- Total edges calculation function
def total_edges (h : Hexahedron) : Nat := (h.face_count * h.edges_per_face) / h.edge_sharing

-- The specific hexahedron (cube) in question
def cube : Hexahedron := {
  face_count := 6,
  edges_per_face := 4,
  edge_sharing := 2
}

-- The theorem to prove the number of edges in a hexahedron
theorem total_number_of_edges_in_hexahedron_is_12 : total_edges cube = 12 := by
  sorry

end NUMINAMATH_GPT_total_number_of_edges_in_hexahedron_is_12_l561_56185


namespace NUMINAMATH_GPT_work_completion_time_l561_56100

noncomputable def work_done (hours : ℕ) (a_rate : ℚ) (b_rate : ℚ) : ℚ :=
  if hours % 2 = 0 then (hours / 2) * (a_rate + b_rate)
  else ((hours - 1) / 2) * (a_rate + b_rate) + a_rate

theorem work_completion_time :
  let a_rate := 1/4
  let b_rate := 1/12
  (∃ t, work_done t a_rate b_rate = 1) → t = 6 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_work_completion_time_l561_56100


namespace NUMINAMATH_GPT_division_by_fraction_equiv_neg_multiplication_l561_56136

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end NUMINAMATH_GPT_division_by_fraction_equiv_neg_multiplication_l561_56136


namespace NUMINAMATH_GPT_ratio_bananas_apples_is_3_to_1_l561_56147

def ratio_of_bananas_to_apples (oranges apples bananas peaches total_fruit : ℕ) : ℚ :=
if oranges = 6 ∧ apples = oranges - 2 ∧ peaches = bananas / 2 ∧ total_fruit = 28
   ∧ 6 + apples + bananas + peaches = total_fruit then
    bananas / apples
else 0

theorem ratio_bananas_apples_is_3_to_1 : ratio_of_bananas_to_apples 6 4 12 6 28 = 3 := by
sorry

end NUMINAMATH_GPT_ratio_bananas_apples_is_3_to_1_l561_56147


namespace NUMINAMATH_GPT_airplane_fraction_l561_56122

noncomputable def driving_time : ℕ := 195

noncomputable def airport_drive_time : ℕ := 10

noncomputable def waiting_time : ℕ := 20

noncomputable def get_off_time : ℕ := 10

noncomputable def faster_by : ℕ := 90

theorem airplane_fraction :
  ∃ x : ℕ, 195 = 40 + x + 90 ∧ x = 65 ∧ x = driving_time / 3 := sorry

end NUMINAMATH_GPT_airplane_fraction_l561_56122


namespace NUMINAMATH_GPT_distance_from_A_to_B_l561_56126

-- Definitions of the conditions
def avg_speed : ℝ := 25
def distance_AB (D : ℝ) : Prop := ∃ T : ℝ, D / (4 * T) = avg_speed ∧ D = 3 * (T * avg_speed)∧ (D / 2) = (T * avg_speed)

theorem distance_from_A_to_B : ∃ D : ℝ, distance_AB D ∧ D = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_A_to_B_l561_56126


namespace NUMINAMATH_GPT_convert_300_degree_to_radian_l561_56146

theorem convert_300_degree_to_radian : (300 : ℝ) * π / 180 = 5 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_convert_300_degree_to_radian_l561_56146


namespace NUMINAMATH_GPT_find_k_l561_56134

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2 + b.2) (3 * a.1, 3 * a.2 - b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l561_56134


namespace NUMINAMATH_GPT_find_positive_number_l561_56161

theorem find_positive_number (x : ℕ) (h_pos : 0 < x) (h_equation : x * x / 100 + 6 = 10) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_number_l561_56161


namespace NUMINAMATH_GPT_find_y_l561_56113

theorem find_y (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := 
sorry

end NUMINAMATH_GPT_find_y_l561_56113


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l561_56176

def p (x y : ℝ) : Prop := x > 1 ∧ y > 1
def q (x y : ℝ) : Prop := x + y > 3

theorem neither_necessary_nor_sufficient :
  ¬ (∀ x y, q x y → p x y) ∧ ¬ (∀ x y, p x y → q x y) :=
by
  sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l561_56176


namespace NUMINAMATH_GPT_sum_of_coordinates_A_l561_56137

-- Define the points A, B, and C and the given conditions
variables (A B C : ℝ × ℝ)
variables (h_ratio1 : dist A C / dist A B = 1 / 3)
variables (h_ratio2 : dist B C / dist A B = 1 / 3)
variables (h_B : B = (2, 8))
variables (h_C : C = (0, 2))

-- Lean 4 statement to prove the sum of the coordinates of A is -14
theorem sum_of_coordinates_A : (A.1 + A.2) = -14 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_A_l561_56137


namespace NUMINAMATH_GPT_value_of_x_l561_56148

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def f_inv (y : ℝ) : ℝ := sorry -- Placeholder for the inverse of f

noncomputable def g (x : ℝ) : ℝ := 3 * f_inv x

theorem value_of_x (h : g 18 = 18) : x = 30 / 11 :=
by
  -- Proof is not required.
  sorry

end NUMINAMATH_GPT_value_of_x_l561_56148


namespace NUMINAMATH_GPT_als_initial_portion_l561_56143

theorem als_initial_portion (a b c : ℝ)
  (h1 : a + b + c = 1200)
  (h2 : a - 150 + 3 * b + 3 * c = 1800) :
  a = 825 :=
sorry

end NUMINAMATH_GPT_als_initial_portion_l561_56143
