import Mathlib

namespace sphere_volume_l2202_220267

theorem sphere_volume (π : ℝ) (r : ℝ):
  4 * π * r^2 = 144 * π →
  (4 / 3) * π * r^3 = 288 * π :=
by
  sorry

end sphere_volume_l2202_220267


namespace total_fertilizer_usage_l2202_220287

theorem total_fertilizer_usage :
  let daily_A : ℝ := 3 / 12
  let daily_B : ℝ := 4 / 10
  let daily_C : ℝ := 5 / 8
  let final_A : ℝ := daily_A + 6
  let final_B : ℝ := daily_B + 5
  let final_C : ℝ := daily_C + 7
  (final_A + final_B + final_C) = 19.275 := by
  sorry

end total_fertilizer_usage_l2202_220287


namespace possible_values_of_b_l2202_220238

-- Set up the basic definitions and conditions
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Assuming the conditions provided in the problem
axiom cond1 : a * (1 - Real.cos B) = b * Real.cos A
axiom cond2 : c = 3
axiom cond3 : 1 / 2 * a * c * Real.sin B = 2 * Real.sqrt 2

-- The theorem expressing the question and the correct answer
theorem possible_values_of_b : b = 2 ∨ b = 4 * Real.sqrt 2 := sorry

end possible_values_of_b_l2202_220238


namespace pears_more_than_apples_l2202_220236

theorem pears_more_than_apples (red_apples green_apples pears : ℕ) (h1 : red_apples = 15) (h2 : green_apples = 8) (h3 : pears = 32) : (pears - (red_apples + green_apples) = 9) :=
by
  sorry

end pears_more_than_apples_l2202_220236


namespace quadratic_geometric_sequence_root_l2202_220296

theorem quadratic_geometric_sequence_root {a b c : ℝ} (r : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b = a * r) 
  (h3 : c = a * r^2)
  (h4 : a ≥ b) 
  (h5 : b ≥ c) 
  (h6 : c ≥ 0) 
  (h7 : (a * r)^2 - 4 * a * (a * r^2) = 0) : 
  -b / (2 * a) = -1 / 8 := 
sorry

end quadratic_geometric_sequence_root_l2202_220296


namespace whitney_money_leftover_l2202_220246

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end whitney_money_leftover_l2202_220246


namespace shaded_area_l2202_220241

theorem shaded_area (r : ℝ) (sector_area : ℝ) (h1 : r = 4) (h2 : sector_area = 2 * Real.pi) : 
  sector_area - (1 / 2 * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = 2 * Real.pi - 4 :=
by 
  -- Lean proof follows
  sorry

end shaded_area_l2202_220241


namespace weight_lifting_requirement_l2202_220278

-- Definitions based on conditions
def weight_25 : Int := 25
def weight_10 : Int := 10
def lifts_25 := 16
def total_weight_25 := 2 * weight_25 * lifts_25

def n_lifts_10 (n : Int) := 2 * weight_10 * n

-- Problem statement and theorem to prove
theorem weight_lifting_requirement (n : Int) : n_lifts_10 n = total_weight_25 ↔ n = 40 := by
  sorry

end weight_lifting_requirement_l2202_220278


namespace gcd_1343_816_l2202_220260

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l2202_220260


namespace solve_problem_l2202_220284

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (0.65 * x > 26) ∧ (0.40 * y < -3) ∧ ((x - y)^2 ≥ 100) 
  → (x > 40) ∧ (y < -7.5)

theorem solve_problem (x y : ℝ) (h : proof_problem x y) : (x > 40) ∧ (y < -7.5) := 
sorry

end solve_problem_l2202_220284


namespace minimum_value_l2202_220277

variable {a b : ℝ}

noncomputable def given_conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + 2 * b = 2

theorem minimum_value :
  given_conditions a b →
  ∃ x, x = (1 + 4 * a + 3 * b) / (a * b) ∧ x ≥ 25 / 2 :=
by
  sorry

end minimum_value_l2202_220277


namespace girls_in_club_l2202_220203

/-
A soccer club has 30 members. For a recent team meeting, only 18 members could attend:
one-third of the girls attended but all of the boys attended. Prove that the number of 
girls in the soccer club is 18.
-/

variables (B G : ℕ)

-- Conditions
def total_members (B G : ℕ) := B + G = 30
def meeting_attendance (B G : ℕ) := (1/3 : ℚ) * G + B = 18

theorem girls_in_club (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : G = 18 :=
  sorry

end girls_in_club_l2202_220203


namespace variation_of_x_l2202_220268

theorem variation_of_x (k j z : ℝ) : ∃ m : ℝ, ∀ x y : ℝ, (x = k * y^2) ∧ (y = j * z^(1 / 3)) → (x = m * z^(2 / 3)) :=
sorry

end variation_of_x_l2202_220268


namespace find_m_l2202_220297

-- Conditions given
def ellipse (x y m : ℝ) : Prop := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- The theorem to prove
theorem find_m (m : ℝ) (h₁ : ellipse 1 1 m) (h₂ : eccentricity 2) : m = 3 ∨ m = 5 :=
  sorry

end find_m_l2202_220297


namespace arithmetic_sequence_ratio_l2202_220201

/-- 
  Given the ratio of the sum of the first n terms of two arithmetic sequences,
  prove the ratio of the 11th terms of these sequences.
-/
theorem arithmetic_sequence_ratio (S T : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 1 : ℚ) / (4 * n + 2)) : 
  S 21 / T 21 = 74 / 43 :=
sorry

end arithmetic_sequence_ratio_l2202_220201


namespace discount_on_pony_jeans_l2202_220265

theorem discount_on_pony_jeans 
  (F P : ℕ)
  (h1 : F + P = 25)
  (h2 : 5 * F + 4 * P = 100) : P = 25 :=
by
  sorry

end discount_on_pony_jeans_l2202_220265


namespace A_intersection_B_eq_C_l2202_220205

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x < 3}
def C := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem A_intersection_B_eq_C : A ∩ B = C := 
by sorry

end A_intersection_B_eq_C_l2202_220205


namespace seungjun_clay_cost_l2202_220245

theorem seungjun_clay_cost (price_per_gram : ℝ) (qty1 qty2 : ℝ) 
  (h1 : price_per_gram = 17.25) 
  (h2 : qty1 = 1000) 
  (h3 : qty2 = 10) :
  (qty1 * price_per_gram + qty2 * price_per_gram) = 17422.5 :=
by
  sorry

end seungjun_clay_cost_l2202_220245


namespace greatest_possible_value_of_squares_l2202_220251

theorem greatest_possible_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 15)
  (h2 : ab + c + d = 78)
  (h3 : ad + bc = 160)
  (h4 : cd = 96) :
  a^2 + b^2 + c^2 + d^2 ≤ 717 ∧ ∃ a b c d, a + b = 15 ∧ ab + c + d = 78 ∧ ad + bc = 160 ∧ cd = 96 ∧ a^2 + b^2 + c^2 + d^2 = 717 :=
sorry

end greatest_possible_value_of_squares_l2202_220251


namespace binom_10_3_l2202_220208

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l2202_220208


namespace problem_equivalent_l2202_220232

def modified_op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem problem_equivalent (x y : ℝ) : 
  modified_op ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 := 
by 
  sorry

end problem_equivalent_l2202_220232


namespace completing_square_solution_l2202_220295

theorem completing_square_solution (x : ℝ) :
  2 * x^2 + 4 * x - 3 = 0 →
  (x + 1)^2 = 5 / 2 :=
by
  sorry

end completing_square_solution_l2202_220295


namespace solution_set_f_x_minus_2_ge_zero_l2202_220252

-- Define the necessary conditions and prove the statement
noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_minus_2_ge_zero (f_even : ∀ x, f x = f (-x))
  (f_mono : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_zero : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x | x ≥ 3 ∨ x ≤ 1} :=
by {
  sorry
}

end solution_set_f_x_minus_2_ge_zero_l2202_220252


namespace sum_mod_7_l2202_220247

/-- Define the six numbers involved. -/
def a := 102345
def b := 102346
def c := 102347
def d := 102348
def e := 102349
def f := 102350

/-- State the theorem to prove the remainder of their sum when divided by 7. -/
theorem sum_mod_7 : 
  (a + b + c + d + e + f) % 7 = 5 := 
by sorry

end sum_mod_7_l2202_220247


namespace jebb_total_spent_l2202_220257

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l2202_220257


namespace data_instances_in_one_hour_l2202_220292

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l2202_220292


namespace range_of_t_l2202_220285

theorem range_of_t (x y a t : ℝ) 
  (h1 : x + 3 * y + a = 4) 
  (h2 : x - y - 3 * a = 0) 
  (h3 : -1 ≤ a ∧ a ≤ 1) 
  (h4 : t = x + y) : 
  1 ≤ t ∧ t ≤ 3 := 
sorry

end range_of_t_l2202_220285


namespace convert_15_deg_to_rad_l2202_220299

theorem convert_15_deg_to_rad (deg_to_rad : ℝ := Real.pi / 180) : 
  15 * deg_to_rad = Real.pi / 12 :=
by sorry

end convert_15_deg_to_rad_l2202_220299


namespace classify_triangles_by_angles_l2202_220200

-- Define the basic types and properties for triangles and their angle classifications
def acute_triangle (α β γ : ℝ) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def right_triangle (α β γ : ℝ) : Prop :=
  α = 90 ∨ β = 90 ∨ γ = 90

def obtuse_triangle (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- Problem: Classify triangles by angles and prove that the correct classification is as per option A
theorem classify_triangles_by_angles :
  (∀ (α β γ : ℝ), acute_triangle α β γ ∨ right_triangle α β γ ∨ obtuse_triangle α β γ) :=
sorry

end classify_triangles_by_angles_l2202_220200


namespace ram_krish_together_time_l2202_220270

theorem ram_krish_together_time : 
  let t_R := 36
  let t_K := t_R / 2
  let task_per_day_R := 1 / t_R
  let task_per_day_K := 1 / t_K
  let task_per_day_together := task_per_day_R + task_per_day_K
  let T := 1 / task_per_day_together
  T = 12 := 
by
  sorry

end ram_krish_together_time_l2202_220270


namespace find_f_l2202_220243

theorem find_f
  (d e f : ℝ)
  (vertex_x vertex_y : ℝ)
  (p_x p_y : ℝ)
  (vertex_cond : vertex_x = 3 ∧ vertex_y = -1)
  (point_cond : p_x = 5 ∧ p_y = 1)
  (equation : ∀ y : ℝ, ∃ x : ℝ, x = d * y^2 + e * y + f) :
  f = 7 / 2 :=
by
  sorry

end find_f_l2202_220243


namespace solve_trig_inequality_l2202_220263

noncomputable def sin_triple_angle_identity (x : ℝ) : ℝ :=
  3 * (Real.sin x) - 4 * (Real.sin x) ^ 3

theorem solve_trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
  (8 / (3 * Real.sin x - sin_triple_angle_identity x) + 3 * (Real.sin x) ^ 2) ≤ 5 ↔
  x = Real.pi / 2 :=
by
  sorry

end solve_trig_inequality_l2202_220263


namespace value_ne_one_l2202_220254

theorem value_ne_one (a b: ℝ) (h : a * b ≠ 0) : (|a| / a) + (|b| / b) ≠ 1 := 
by 
  sorry

end value_ne_one_l2202_220254


namespace no_real_roots_range_a_l2202_220282

theorem no_real_roots_range_a (a : ℝ) : (¬∃ x : ℝ, 2 * x^2 + (a - 5) * x + 2 = 0) → 1 < a ∧ a < 9 :=
by
  sorry

end no_real_roots_range_a_l2202_220282


namespace mass_percentage_correct_l2202_220255

noncomputable def mass_percentage_C_H_N_O_in_C20H25N3O 
  (m_C : ℚ) (m_H : ℚ) (m_N : ℚ) (m_O : ℚ) 
  (atoms_C : ℚ) (atoms_H : ℚ) (atoms_N : ℚ) (atoms_O : ℚ)
  (total_mass : ℚ)
  (percentage_C : ℚ) (percentage_H : ℚ) (percentage_N : ℚ) (percentage_O : ℚ) :=
  atoms_C = 20 ∧ atoms_H = 25 ∧ atoms_N = 3 ∧ atoms_O = 1 ∧ 
  m_C = 12.01 ∧ m_H = 1.008 ∧ m_N = 14.01 ∧ m_O = 16 ∧ 
  total_mass = (atoms_C * m_C) + (atoms_H * m_H) + (atoms_N * m_N) + (atoms_O * m_O) ∧ 
  percentage_C = (atoms_C * m_C / total_mass) * 100 ∧ 
  percentage_H = (atoms_H * m_H / total_mass) * 100 ∧ 
  percentage_N = (atoms_N * m_N / total_mass) * 100 ∧ 
  percentage_O = (atoms_O * m_O / total_mass) * 100 

theorem mass_percentage_correct : 
  mass_percentage_C_H_N_O_in_C20H25N3O 12.01 1.008 14.01 16 20 25 3 1 323.43 74.27 7.79 12.99 4.95 :=
by {
  sorry
}

end mass_percentage_correct_l2202_220255


namespace jennifer_score_l2202_220262

theorem jennifer_score 
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (unanswered_questions : ℕ)
  (points_per_correct : ℤ)
  (points_deduction_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (h_total : total_questions = 30)
  (h_correct : correct_answers = 15)
  (h_incorrect : incorrect_answers = 10)
  (h_unanswered : unanswered_questions = 5)
  (h_points_correct : points_per_correct = 2)
  (h_deduction_incorrect : points_deduction_incorrect = -1)
  (h_points_unanswered : points_per_unanswered = 0) : 
  ∃ (score : ℤ), score = (correct_answers * points_per_correct 
                          + incorrect_answers * points_deduction_incorrect 
                          + unanswered_questions * points_per_unanswered) 
                        ∧ score = 20 := 
by
  sorry

end jennifer_score_l2202_220262


namespace green_hat_cost_l2202_220209

theorem green_hat_cost (G : ℝ) (total_hats : ℕ) (blue_hats : ℕ) (green_hats : ℕ) (blue_cost : ℝ) (total_cost : ℝ) 
    (h₁ : blue_hats = 85) (h₂ : blue_cost = 6) (h₃ : green_hats = 90) (h₄ : total_cost = 600) 
    (h₅ : total_hats = blue_hats + green_hats) 
    (h₆ : total_cost = blue_hats * blue_cost + green_hats * G) : 
    G = 1 := by
  sorry

end green_hat_cost_l2202_220209


namespace largest_digit_change_l2202_220224

-- Definitions
def initial_number : ℝ := 0.12345

def change_digit (k : Fin 5) : ℝ :=
  match k with
  | 0 => 0.92345
  | 1 => 0.19345
  | 2 => 0.12945
  | 3 => 0.12395
  | 4 => 0.12349

theorem largest_digit_change :
  ∀ k : Fin 5, k ≠ 0 → change_digit 0 > change_digit k :=
by
  intros k hk
  sorry

end largest_digit_change_l2202_220224


namespace quadratic_to_square_l2202_220293

theorem quadratic_to_square (x h k : ℝ) : 
  (x * x - 4 * x + 3 = 0) →
  ((x + h) * (x + h) = k) →
  k = 1 :=
by
  sorry

end quadratic_to_square_l2202_220293


namespace technicians_count_l2202_220202

noncomputable def total_salary := 8000 * 21
noncomputable def average_salary_all := 8000
noncomputable def average_salary_technicians := 12000
noncomputable def average_salary_rest := 6000
noncomputable def total_workers := 21

theorem technicians_count :
  ∃ (T R : ℕ),
  T + R = total_workers ∧
  average_salary_technicians * T + average_salary_rest * R = total_salary ∧
  T = 7 :=
by
  sorry

end technicians_count_l2202_220202


namespace find_number_l2202_220204

theorem find_number (x : ℤ) (h : x - 27 = 49) : x = 76 := by
  sorry

end find_number_l2202_220204


namespace minimum_area_l2202_220271

-- Define point A
def A : ℝ × ℝ := (-4, 0)

-- Define point B
def B : ℝ × ℝ := (0, 4)

-- Define the circle
def on_circle (C : ℝ × ℝ) : Prop := (C.1 - 2)^2 + C.2^2 = 2

-- Instantiating the proof of the minimum area of △ABC = 8
theorem minimum_area (C : ℝ × ℝ) (h : on_circle C) : 
  ∃ C : ℝ × ℝ, on_circle C ∧ 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 8 := 
sorry

end minimum_area_l2202_220271


namespace smallest_n_divisible_by_5_l2202_220212

def is_not_divisible_by_5 (x : ℤ) : Prop :=
  ¬ (x % 5 = 0)

def avg_is_integer (xs : List ℤ) : Prop :=
  (List.sum xs) % 5 = 0

theorem smallest_n_divisible_by_5 (n : ℕ) (h1 : n > 1980)
  (h2 : ∀ x ∈ List.range n, is_not_divisible_by_5 x)
  : n = 1985 :=
by
  -- The proof would go here
  sorry

end smallest_n_divisible_by_5_l2202_220212


namespace diagonal_of_rectangular_prism_l2202_220234

noncomputable def diagonal_length (a b c : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 12 18 15 = 3 * Real.sqrt 77 :=
by
  sorry

end diagonal_of_rectangular_prism_l2202_220234


namespace simplify_expression_d_l2202_220228

variable (a b c : ℝ)

theorem simplify_expression_d : a - (b - c) = a - b + c :=
  sorry

end simplify_expression_d_l2202_220228


namespace express_in_scientific_notation_l2202_220256

theorem express_in_scientific_notation : (0.0000028 = 2.8 * 10^(-6)) :=
sorry

end express_in_scientific_notation_l2202_220256


namespace vehicles_with_cd_player_but_no_pw_or_ab_l2202_220258

-- Definitions based on conditions from step a)
def P : ℝ := 0.60 -- percentage of vehicles with power windows
def A : ℝ := 0.25 -- percentage of vehicles with anti-lock brakes
def C : ℝ := 0.75 -- percentage of vehicles with a CD player
def PA : ℝ := 0.10 -- percentage of vehicles with both power windows and anti-lock brakes
def AC : ℝ := 0.15 -- percentage of vehicles with both anti-lock brakes and a CD player
def PC : ℝ := 0.22 -- percentage of vehicles with both power windows and a CD player
def PAC : ℝ := 0.00 -- no vehicle has all 3 features

-- The statement we want to prove
theorem vehicles_with_cd_player_but_no_pw_or_ab : C - (PC + AC) = 0.38 := by
  sorry

end vehicles_with_cd_player_but_no_pw_or_ab_l2202_220258


namespace solution_set_of_inequality_l2202_220239

theorem solution_set_of_inequality (a x : ℝ) (h : a > 0) : 
  (x^2 - (a + 1/a + 1) * x + a + 1/a < 0) ↔ (1 < x ∧ x < a + 1/a) :=
by sorry

end solution_set_of_inequality_l2202_220239


namespace find_number_satisfying_condition_l2202_220259

-- Define the condition where fifteen percent of x equals 150
def fifteen_percent_eq (x : ℝ) : Prop :=
  (15 / 100) * x = 150

-- Statement to prove the existence of a number x that satisfies the condition, and this x equals 1000
theorem find_number_satisfying_condition : ∃ x : ℝ, fifteen_percent_eq x ∧ x = 1000 :=
by
  -- Proof will be added here
  sorry

end find_number_satisfying_condition_l2202_220259


namespace ways_to_write_1800_as_sum_of_4s_and_5s_l2202_220225

theorem ways_to_write_1800_as_sum_of_4s_and_5s : 
  ∃ S : Finset (ℕ × ℕ), S.card = 91 ∧ ∀ (nm : ℕ × ℕ), nm ∈ S ↔ 4 * nm.1 + 5 * nm.2 = 1800 ∧ nm.1 ≥ 0 ∧ nm.2 ≥ 0 :=
by
  sorry

end ways_to_write_1800_as_sum_of_4s_and_5s_l2202_220225


namespace division_of_15_by_neg_5_l2202_220221

theorem division_of_15_by_neg_5 : 15 / (-5) = -3 :=
by
  sorry

end division_of_15_by_neg_5_l2202_220221


namespace replace_all_cardio_machines_cost_l2202_220276

noncomputable def totalReplacementCost : ℕ :=
  let numGyms := 20
  let bikesPerGym := 10
  let treadmillsPerGym := 5
  let ellipticalsPerGym := 5
  let costPerBike := 700
  let costPerTreadmill := costPerBike * 3 / 2
  let costPerElliptical := costPerTreadmill * 2
  let totalBikes := numGyms * bikesPerGym
  let totalTreadmills := numGyms * treadmillsPerGym
  let totalEllipticals := numGyms * ellipticalsPerGym
  (totalBikes * costPerBike) + (totalTreadmills * costPerTreadmill) + (totalEllipticals * costPerElliptical)

theorem replace_all_cardio_machines_cost :
  totalReplacementCost = 455000 :=
by
  -- All the calculation steps provided as conditions and intermediary results need to be verified here.
  sorry

end replace_all_cardio_machines_cost_l2202_220276


namespace area_of_rectangle_l2202_220226

noncomputable def rectangle_area : ℚ :=
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  length * width

theorem area_of_rectangle : rectangle_area = 392 / 9 :=
  by 
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  have : length * width = 392 / 9 := sorry
  exact this

end area_of_rectangle_l2202_220226


namespace covered_ratio_battonya_covered_ratio_sopron_l2202_220222

noncomputable def angular_diameter_sun : ℝ := 1899 / 2
noncomputable def angular_diameter_moon : ℝ := 1866 / 2

def max_phase_battonya : ℝ := 0.766
def max_phase_sopron : ℝ := 0.678

def center_distance (R_M R_S f : ℝ) : ℝ :=
  R_M - (2 * f - 1) * R_S

-- Defining the hypothetical calculation (details omitted for brevity)
def covered_ratio (R_S R_M d : ℝ) : ℝ := 
  -- Placeholder for the actual calculation logic
  sorry

theorem covered_ratio_battonya :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_battonya) = 0.70 :=
  sorry

theorem covered_ratio_sopron :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_sopron) = 0.59 :=
  sorry

end covered_ratio_battonya_covered_ratio_sopron_l2202_220222


namespace chris_eats_donuts_l2202_220237

def daily_donuts := 10
def days := 12
def donuts_eaten_per_day := 1
def boxes_filled := 10
def donuts_per_box := 10

-- Define the total number of donuts made.
def total_donuts := daily_donuts * days

-- Define the total number of donuts Jeff eats.
def jeff_total_eats := donuts_eaten_per_day * days

-- Define the remaining donuts after Jeff eats his share.
def remaining_donuts := total_donuts - jeff_total_eats

-- Define the total number of donuts in the boxes.
def donuts_in_boxes := boxes_filled * donuts_per_box

-- The proof problem:
theorem chris_eats_donuts : remaining_donuts - donuts_in_boxes = 8 :=
by
  -- Placeholder for proof
  sorry

end chris_eats_donuts_l2202_220237


namespace simplify_expression_l2202_220275

theorem simplify_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -(m + n) :=
by sorry

end simplify_expression_l2202_220275


namespace harry_total_cost_l2202_220206

-- Define the price of each type of seed packet
def pumpkin_price : ℝ := 2.50
def tomato_price : ℝ := 1.50
def chili_pepper_price : ℝ := 0.90
def zucchini_price : ℝ := 1.20
def eggplant_price : ℝ := 1.80

-- Define the quantities Harry wants to buy
def pumpkin_qty : ℕ := 4
def tomato_qty : ℕ := 6
def chili_pepper_qty : ℕ := 7
def zucchini_qty : ℕ := 3
def eggplant_qty : ℕ := 5

-- Calculate the total cost
def total_cost : ℝ :=
  pumpkin_qty * pumpkin_price +
  tomato_qty * tomato_price +
  chili_pepper_qty * chili_pepper_price +
  zucchini_qty * zucchini_price +
  eggplant_qty * eggplant_price

-- The proof problem
theorem harry_total_cost : total_cost = 38.90 := by
  sorry

end harry_total_cost_l2202_220206


namespace solve_equation_l2202_220290

theorem solve_equation (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
    x^2 * (Real.log 27 / Real.log x) * (Real.log x / Real.log 9) = x + 4 → x = 2 :=
by
  sorry

end solve_equation_l2202_220290


namespace tom_initial_money_l2202_220213

theorem tom_initial_money (spent_on_game : ℕ) (toy_cost : ℕ) (number_of_toys : ℕ)
    (total_spent : ℕ) (h1 : spent_on_game = 49) (h2 : toy_cost = 4)
    (h3 : number_of_toys = 2) (h4 : total_spent = spent_on_game + number_of_toys * toy_cost) :
  total_spent = 57 := by
  sorry

end tom_initial_money_l2202_220213


namespace stratified_leader_selection_probability_of_mixed_leaders_l2202_220207

theorem stratified_leader_selection :
  let num_first_grade := 150
  let num_second_grade := 100
  let total_leaders := 5
  let leaders_first_grade := (total_leaders * num_first_grade) / (num_first_grade + num_second_grade)
  let leaders_second_grade := (total_leaders * num_second_grade) / (num_first_grade + num_second_grade)
  leaders_first_grade = 3 ∧ leaders_second_grade = 2 :=
by
  sorry

theorem probability_of_mixed_leaders :
  let num_first_grade_leaders := 3
  let num_second_grade_leaders := 2
  let total_leaders := 5
  let total_ways := 10
  let favorable_ways := 6
  (favorable_ways / total_ways) = (3 / 5) :=
by
  sorry

end stratified_leader_selection_probability_of_mixed_leaders_l2202_220207


namespace tan_sum_pi_div_12_l2202_220215

theorem tan_sum_pi_div_12 (h1 : Real.tan (Real.pi / 12) ≠ 0) (h2 : Real.tan (5 * Real.pi / 12) ≠ 0) :
  Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 := 
by
  sorry

end tan_sum_pi_div_12_l2202_220215


namespace reflection_line_sum_l2202_220231

-- Prove that the sum of m and b is 10 given the reflection conditions

theorem reflection_line_sum
    (m b : ℚ)
    (H : ∀ (x y : ℚ), (2, 2) = (x, y) → (8, 6) = (2 * (5 - (3 / 2) * (2 - x)), 2 + m * (y - 2)) ∧ y = m * x + b) :
  m + b = 10 :=
sorry

end reflection_line_sum_l2202_220231


namespace dot_product_property_l2202_220250

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

variables (x_P y_P : ℝ) (F1 F2 : ℝ × ℝ)

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0) ∨ F = (-1, 0)

def radius_of_inscribed_circle (r : ℝ) : Prop :=
  r = 1 / 2

theorem dot_product_property (h1 : point_on_ellipse x_P y_P)
  (h2 : is_focus F1) (h3 : is_focus F2) (h4: radius_of_inscribed_circle (1/2)):
  (x_P^2 - 1 + y_P^2) = 9 / 4 :=
sorry

end dot_product_property_l2202_220250


namespace solution_set_transformation_l2202_220261

variables (a b c α β : ℝ) (h_root : (α : ℝ) > 0)

open Set

def quadratic_inequality (x : ℝ) : Prop :=
  a * x^2 + b * x + c > 0

def transformed_inequality (x : ℝ) : Prop :=
  c * x^2 + b * x + a < 0

theorem solution_set_transformation :
  (∀ x, quadratic_inequality a b c x ↔ (α < x ∧ x < β)) →
  (∃ α β : ℝ, α > 0 ∧ (∀ x, transformed_inequality c b a x ↔ (x < 1/β ∨ x > 1/α))) :=
by
  sorry

end solution_set_transformation_l2202_220261


namespace absolute_diff_half_l2202_220219

theorem absolute_diff_half (x y : ℝ) 
  (h : ((x + y = x - y ∧ x - y = x * y) ∨ 
       (x + y = x * y ∧ x * y = x / y) ∨ 
       (x - y = x * y ∧ x * y = x / y))
       ∧ x ≠ 0 ∧ y ≠ 0) : 
     |y| - |x| = 1 / 2 := 
sorry

end absolute_diff_half_l2202_220219


namespace min_value_9x_plus_3y_l2202_220279

noncomputable def minimum_value_of_expression : ℝ := 6

theorem min_value_9x_plus_3y (x y : ℝ) 
  (h1 : (x - 1) * 4 + 2 * y = 0) 
  (ha : ∃ (a1 a2 : ℝ), (a1, a2) = (x - 1, 2)) 
  (hb : ∃ (b1 b2 : ℝ), (b1, b2) = (4, y)) : 
  9^x + 3^y = minimum_value_of_expression :=
by
  sorry

end min_value_9x_plus_3y_l2202_220279


namespace hip_hop_final_percentage_is_39_l2202_220298

noncomputable def hip_hop_percentage (total_songs percentage_country: ℝ):
  ℝ :=
  let percentage_non_country := 1 - percentage_country
  let original_ratio_hip_hop := 0.65
  let original_ratio_pop := 0.35
  let total_non_country := original_ratio_hip_hop + original_ratio_pop
  let hip_hop_percentage := original_ratio_hip_hop / total_non_country * percentage_non_country
  hip_hop_percentage

theorem hip_hop_final_percentage_is_39 (total_songs : ℕ) :
  hip_hop_percentage total_songs 0.40 = 0.39 :=
by
  sorry

end hip_hop_final_percentage_is_39_l2202_220298


namespace cookies_per_batch_l2202_220210

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end cookies_per_batch_l2202_220210


namespace smallest_number_am_median_largest_l2202_220274

noncomputable def smallest_number (a b c : ℕ) : ℕ :=
if a ≤ b ∧ a ≤ c then a
else if b ≤ a ∧ b ≤ c then b
else c

theorem smallest_number_am_median_largest (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 28) (h3 : c = b + 6) :
  smallest_number a b c = 28 :=
sorry

end smallest_number_am_median_largest_l2202_220274


namespace edge_length_increase_l2202_220233

theorem edge_length_increase (e e' : ℝ) (A : ℝ) (hA : ∀ e, A = 6 * e^2)
  (hA' : 2.25 * A = 6 * e'^2) :
  (e' - e) / e * 100 = 50 :=
by
  sorry

end edge_length_increase_l2202_220233


namespace calculation_result_l2202_220214

theorem calculation_result :
  1500 * 451 * 0.0451 * 25 = 7627537500 :=
by
  -- Simply state without proof as instructed
  sorry

end calculation_result_l2202_220214


namespace girls_attending_sports_event_l2202_220280

theorem girls_attending_sports_event 
  (total_students attending_sports_event : ℕ) 
  (girls boys : ℕ)
  (h1 : total_students = 1500)
  (h2 : attending_sports_event = 900)
  (h3 : girls + boys = total_students)
  (h4 : (1 / 2) * girls + (3 / 5) * boys = attending_sports_event) :
  (1 / 2) * girls = 500 := 
by
  sorry

end girls_attending_sports_event_l2202_220280


namespace slices_eaten_l2202_220244

theorem slices_eaten (total_slices : Nat) (slices_left : Nat) (expected_slices_eaten : Nat) :
  total_slices = 32 →
  slices_left = 7 →
  expected_slices_eaten = 25 →
  total_slices - slices_left = expected_slices_eaten :=
by
  intros
  sorry

end slices_eaten_l2202_220244


namespace min_value_inequality_l2202_220288

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l2202_220288


namespace base7_to_base10_and_frac_l2202_220273

theorem base7_to_base10_and_frac (c d e : ℕ) 
  (h1 : (761 : ℕ) = 7^2 * 7 + 6 * 7^1 + 1 * 7^0)
  (h2 : (10 * 10 * c + 10 * d + e) = 386)
  (h3 : c = 3)
  (h4 : d = 8)
  (h5 : e = 6) :
  (d * e) / 15 = 48 / 15 := 
sorry

end base7_to_base10_and_frac_l2202_220273


namespace find_t_find_s_find_a_find_c_l2202_220230

-- Proof Problem I4.1
theorem find_t (p q r t : ℝ) (h1 : (p + q + r) / 3 = 12) (h2 : (p + q + r + t + 2 * t) / 5 = 15) : t = 13 :=
sorry

-- Proof Problem I4.2
theorem find_s (k t s : ℝ) (hk : k ≠ 0) (h1 : k^4 + (1 / k^4) = t + 1) (h2 : t = 13) (h_s : s = k^2 + (1 / k^2)) : s = 4 :=
sorry

-- Proof Problem I4.3
theorem find_a (s a b : ℝ) (hxₘ : 1 ≠ 11) (hyₘ : 2 ≠ 7) (h1 : (a, b) = ((1 * 11 + s * 1) / (1 + s), (1 * 7 + s * 2) / (1 + s))) (h_s : s = 4) : a = 3 :=
sorry

-- Proof Problem I4.4
theorem find_c (a c : ℝ) (h1 : ∀ x, a * x^2 + 12 * x + c = 0 → (a*x^2 + 12 * x + c = 0)) (h2 : ∃ x, a * x^2 + 12 * x + c = 0) : c = 36 / a :=
sorry

end find_t_find_s_find_a_find_c_l2202_220230


namespace cars_without_features_l2202_220235

theorem cars_without_features (total_cars cars_with_air_bags cars_with_power_windows cars_with_sunroofs 
                               cars_with_air_bags_and_power_windows cars_with_air_bags_and_sunroofs 
                               cars_with_power_windows_and_sunroofs cars_with_all_features: ℕ)
                               (h1 : total_cars = 80)
                               (h2 : cars_with_air_bags = 45)
                               (h3 : cars_with_power_windows = 40)
                               (h4 : cars_with_sunroofs = 25)
                               (h5 : cars_with_air_bags_and_power_windows = 20)
                               (h6 : cars_with_air_bags_and_sunroofs = 15)
                               (h7 : cars_with_power_windows_and_sunroofs = 10)
                               (h8 : cars_with_all_features = 8) : 
    total_cars - (cars_with_air_bags + cars_with_power_windows + cars_with_sunroofs 
                 - cars_with_air_bags_and_power_windows - cars_with_air_bags_and_sunroofs 
                 - cars_with_power_windows_and_sunroofs + cars_with_all_features) = 7 :=
by sorry

end cars_without_features_l2202_220235


namespace factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l2202_220248

-- Problem 1
theorem factorize_polynomial (x y : ℝ) : 
  x^2 - y^2 + 2*x - 2*y = (x - y)*(x + y + 2) := 
sorry

-- Problem 2
theorem triangle_equilateral (a b c : ℝ) (h : a^2 + c^2 - 2*b*(a - b + c) = 0) : 
  a = b ∧ b = c :=
sorry

-- Problem 3
theorem prove_2p_eq_m_plus_n (m n p : ℝ) (h : 1/4*(m - n)^2 = (p - n)*(m - p)) : 
  2*p = m + n :=
sorry

end factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l2202_220248


namespace translation_of_graph_l2202_220286

theorem translation_of_graph (f : ℝ → ℝ) (x : ℝ) :
  f x = 2 ^ x →
  f (x - 1) + 2 = 2 ^ (x - 1) + 2 :=
by
  intro
  sorry

end translation_of_graph_l2202_220286


namespace prove_system_of_equations_l2202_220281

variables (x y : ℕ)

def system_of_equations (x y : ℕ) : Prop :=
  x = 2*y + 4 ∧ x = 3*y - 9

theorem prove_system_of_equations :
  ∀ (x y : ℕ), system_of_equations x y :=
by sorry

end prove_system_of_equations_l2202_220281


namespace number_divided_by_four_l2202_220291

variable (x : ℝ)

theorem number_divided_by_four (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by {
  sorry
}

end number_divided_by_four_l2202_220291


namespace tigers_home_games_l2202_220217

-- Definitions based on the conditions
def losses : ℕ := 12
def ties : ℕ := losses / 2
def wins : ℕ := 38

-- Statement to prove
theorem tigers_home_games : losses + ties + wins = 56 := by
  sorry

end tigers_home_games_l2202_220217


namespace no_prime_ratio_circle_l2202_220253

theorem no_prime_ratio_circle (A : Fin 2007 → ℕ) :
  ¬ (∀ i : Fin 2007, (∃ p : ℕ, Nat.Prime p ∧ (p = A i / A ((i + 1) % 2007) ∨ p = A ((i + 1) % 2007) / A i))) := by
  sorry

end no_prime_ratio_circle_l2202_220253


namespace line_intersects_hyperbola_left_branch_l2202_220240

noncomputable def problem_statement (k : ℝ) : Prop :=
  ∀ (x y : ℝ), y = k * x - 1 ∧ x^2 - y^2 = 1 ∧ y < 0 → 
  k ∈ Set.Ioo (-Real.sqrt 2) (-1)

theorem line_intersects_hyperbola_left_branch (k : ℝ) :
  problem_statement k :=
by
  sorry

end line_intersects_hyperbola_left_branch_l2202_220240


namespace julia_investment_l2202_220227

-- Define the total investment and the relationship between the investments
theorem julia_investment:
  ∀ (m : ℕ), 
  m + 6 * m = 200000 → 6 * m = 171428 := 
by
  sorry

end julia_investment_l2202_220227


namespace pigs_total_l2202_220269

theorem pigs_total (initial_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joined_pigs = 22) 
  : total_pigs = 86 :=
by
  sorry

end pigs_total_l2202_220269


namespace calculate_shaded_area_l2202_220223

noncomputable def square_shaded_area : ℝ := 
  let a := 10 -- side length of the square
  let s := a / 2 -- half side length, used for midpoints
  let total_area := a * a / 2 -- total area of a right triangle with legs a and a
  let triangle_DMA := total_area / 2 -- area of triangle DAM
  let triangle_DNG := triangle_DMA / 5 -- area of triangle DNG
  let triangle_CDM := total_area -- area of triangle CDM
  let shaded_area := triangle_CDM + triangle_DNG - triangle_DMA -- area of shaded region
  shaded_area

theorem calculate_shaded_area : square_shaded_area = 35 := 
by 
sorry

end calculate_shaded_area_l2202_220223


namespace repeating_decimal_fraction_l2202_220289

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end repeating_decimal_fraction_l2202_220289


namespace ratio_of_radii_of_cylinders_l2202_220283

theorem ratio_of_radii_of_cylinders
  (r_V r_B h_V h_B : ℝ)
  (h1 : h_V = 1/2 * h_B)
  (h2 : π * r_B^2 * h_B / 2  = 4)
  (h3 : π * r_V^2 * h_V = 16) :
  r_V / r_B = 2 := 
by 
  sorry

end ratio_of_radii_of_cylinders_l2202_220283


namespace only_solution_for_triplet_l2202_220266

theorem only_solution_for_triplet (x y z : ℤ) (h : x^2 + y^2 + z^2 - 2 * x * y * z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end only_solution_for_triplet_l2202_220266


namespace difference_two_smallest_integers_l2202_220294

/--
There is more than one integer greater than 1 which, when divided by any integer k such that 2 ≤ k ≤ 11, has a remainder of 1.
Prove that the difference between the two smallest such integers is 27720.
-/
theorem difference_two_smallest_integers :
  ∃ n₁ n₂ : ℤ, 
  (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (n₁ % k = 1 ∧ n₂ % k = 1)) ∧ 
  n₁ > 1 ∧ n₂ > 1 ∧ 
  ∀ m : ℤ, (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (m % k =  1)) ∧ m > 1 → m = n₁ ∨ m = n₂ → 
  (n₂ - n₁ = 27720) := 
sorry

end difference_two_smallest_integers_l2202_220294


namespace sin_double_angle_neg_one_l2202_220242

theorem sin_double_angle_neg_one (α : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, Real.cos α)) (h₂ : b = (Real.sin α, 1)) (h₃ : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sin (2 * α) = -1 :=
sorry

end sin_double_angle_neg_one_l2202_220242


namespace solution_l2202_220264

noncomputable def problem : Prop :=
  (2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2) ∧
  (1 - 2 * Real.sin (Real.pi / 12) ^ 2 ≠ 1 / 2) ∧
  (Real.cos (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 
   Real.sin (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 2) ∧
  ( (Real.tan (77 * Real.pi / 180) - Real.tan (32 * Real.pi / 180)) /
    (2 * (1 + Real.tan (77 * Real.pi / 180) * Real.tan (32 * Real.pi / 180))) = 1 / 2 )

theorem solution : problem :=
  by 
    sorry

end solution_l2202_220264


namespace willie_final_stickers_l2202_220218

-- Definitions of initial stickers and given stickers
def willie_initial_stickers : ℝ := 36.0
def emily_gives : ℝ := 7.0

-- The statement to prove
theorem willie_final_stickers : willie_initial_stickers + emily_gives = 43.0 := by
  sorry

end willie_final_stickers_l2202_220218


namespace count_two_digit_or_less_numbers_l2202_220220

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l2202_220220


namespace pizzas_served_dinner_eq_6_l2202_220249

-- Definitions based on the conditions
def pizzas_served_lunch : Nat := 9
def pizzas_served_today : Nat := 15

-- The theorem to prove the number of pizzas served during dinner
theorem pizzas_served_dinner_eq_6 : pizzas_served_today - pizzas_served_lunch = 6 := by
  sorry

end pizzas_served_dinner_eq_6_l2202_220249


namespace sum_series_eq_two_l2202_220211

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l2202_220211


namespace axis_of_symmetry_l2202_220229

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) :
  ∀ y : ℝ, (∃ x₁ x₂ : ℝ, y = f x₁ ∧ y = f x₂ ∧ (x₁ + x₂) / 2 = 2) :=
by
  sorry

end axis_of_symmetry_l2202_220229


namespace no_solutions_l2202_220272

theorem no_solutions (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a + b ≠ 0) :
  ¬ (1 / a + 2 / b = 3 / (a + b)) :=
by { sorry }

end no_solutions_l2202_220272


namespace five_pow_10000_mod_1000_l2202_220216

theorem five_pow_10000_mod_1000 (h : 5^500 ≡ 1 [MOD 1000]) : 5^10000 ≡ 1 [MOD 1000] := sorry

end five_pow_10000_mod_1000_l2202_220216
