import Mathlib

namespace bhupathi_amount_l1771_177125

variable (A B : ℝ)

theorem bhupathi_amount :
  (A + B = 1210 ∧ (4 / 15) * A = (2 / 5) * B) → B = 484 :=
by
  sorry

end bhupathi_amount_l1771_177125


namespace sum_y_coordinates_of_other_vertices_of_parallelogram_l1771_177106

theorem sum_y_coordinates_of_other_vertices_of_parallelogram :
  let x1 := 4
  let y1 := 26
  let x2 := 12
  let y2 := -8
  let midpoint_y := (y1 + y2) / 2
  2 * midpoint_y = 18 := by
    sorry

end sum_y_coordinates_of_other_vertices_of_parallelogram_l1771_177106


namespace quadratic_expression_negative_for_all_x_l1771_177170

theorem quadratic_expression_negative_for_all_x (k : ℝ) :
  (∀ x : ℝ, (5-k) * x^2 - 2 * (1-k) * x + 2 - 2 * k < 0) ↔ k > 9 :=
sorry

end quadratic_expression_negative_for_all_x_l1771_177170


namespace second_particle_catches_first_l1771_177188

open Real

-- Define the distance functions for both particles
def distance_first (t : ℝ) : ℝ := 34 + 5 * t
def distance_second (t : ℝ) : ℝ := 0.25 * t^2 + 2.75 * t

-- The proof statement
theorem second_particle_catches_first : ∃ t : ℝ, distance_second t = distance_first t ∧ t = 17 :=
by
  have : distance_first 17 = 34 + 5 * 17 := by sorry
  have : distance_second 17 = 0.25 * 17^2 + 2.75 * 17 := by sorry
  sorry

end second_particle_catches_first_l1771_177188


namespace f_odd_f_inequality_solution_l1771_177194

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd: 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = - f x := 
by
  sorry

theorem f_inequality_solution:
  { x : ℝ // -1 < x ∧ x < 1 ∧ f x < -1 } = { x : ℝ // -1 < x ∧ x < -1/3 } := 
by 
  sorry

end f_odd_f_inequality_solution_l1771_177194


namespace inequality_proof_equality_condition_l1771_177162

variable {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
axiom x1_pos : x1 > 0
axiom x2_pos : x2 > 0
axiom x1y1_gz1sq : x1 * y1 > z1 ^ 2
axiom x2y2_gz2sq : x2 * y2 > z2 ^ 2

theorem inequality_proof : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) <= 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

theorem equality_condition : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) = 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) ↔ 
  (x1 = x2 ∧ y1 = y2 ∧ z1 = z2) :=
sorry

end inequality_proof_equality_condition_l1771_177162


namespace length_of_first_platform_is_140_l1771_177186

-- Definitions based on problem conditions
def train_length : ℝ := 190
def time_first_platform : ℝ := 15
def time_second_platform : ℝ := 20
def length_second_platform : ℝ := 250

-- Definition for the length of the first platform (what we're proving)
def length_first_platform (L : ℝ) : Prop :=
  (time_first_platform * (train_length + L) = time_second_platform * (train_length + length_second_platform))

-- Theorem: The length of the first platform is 140 meters
theorem length_of_first_platform_is_140 : length_first_platform 140 :=
  by sorry

end length_of_first_platform_is_140_l1771_177186


namespace jet_bar_sales_difference_l1771_177195

variable (monday_sales : ℕ) (total_target : ℕ) (remaining_target : ℕ)
variable (sales_so_far : ℕ) (tuesday_sales : ℕ)
def JetBarsDifference : Prop :=
  monday_sales = 45 ∧ total_target = 90 ∧ remaining_target = 16 ∧
  sales_so_far = total_target - remaining_target ∧
  tuesday_sales = sales_so_far - monday_sales ∧
  (monday_sales - tuesday_sales = 16)

theorem jet_bar_sales_difference :
  JetBarsDifference 45 90 16 (90 - 16) (90 - 16 - 45) :=
by
  sorry

end jet_bar_sales_difference_l1771_177195


namespace max_min_f_values_l1771_177198

noncomputable def f (a b c d : ℝ) : ℝ := (Real.sqrt (5 * a + 9) + Real.sqrt (5 * b + 9) + Real.sqrt (5 * c + 9) + Real.sqrt (5 * d + 9))

theorem max_min_f_values (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h₁ : a + b + c + d = 32) :
  (f a b c d ≤ 28) ∧ (f a b c d ≥ 22) := by
  sorry

end max_min_f_values_l1771_177198


namespace prob_le_45_l1771_177167

-- Define the probability conditions
def prob_between_1_and_45 : ℚ := 7 / 15
def prob_ge_1 : ℚ := 14 / 15

-- State the theorem to prove
theorem prob_le_45 : prob_between_1_and_45 = 7 / 15 := by
  sorry

end prob_le_45_l1771_177167


namespace rectangle_area_inscribed_circle_l1771_177151

theorem rectangle_area_inscribed_circle {r w l : ℕ} (h1 : r = 7) (h2 : w = 2 * r) (h3 : l = 3 * w) : l * w = 588 :=
by 
  -- The proof details are omitted as per instructions.
  sorry

end rectangle_area_inscribed_circle_l1771_177151


namespace ratio_new_values_l1771_177144

theorem ratio_new_values (x y x2 y2 : ℝ) (h1 : x / y = 7 / 5) (h2 : x2 = x * y) (h3 : y2 = y * x) : x2 / y2 = 1 := by
  sorry

end ratio_new_values_l1771_177144


namespace range_of_a_l1771_177185

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a > 0) ↔ a > -3 := 
by sorry

end range_of_a_l1771_177185


namespace LCM_of_numbers_with_HCF_and_ratio_l1771_177111

theorem LCM_of_numbers_with_HCF_and_ratio (a b x : ℕ)
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x)
  (h3 : ∀ y : ℕ, y ∣ a → y ∣ b → y ∣ x)
  (hx : x = 5) :
  Nat.lcm a b = 60 := 
by
  sorry

end LCM_of_numbers_with_HCF_and_ratio_l1771_177111


namespace min_value_of_n_for_constant_term_l1771_177165

theorem min_value_of_n_for_constant_term :
  ∃ (n : ℕ) (r : ℕ) (h₁ : r > 0) (h₂ : n > 0), 
  (2 * n - 7 * r / 3 = 0) ∧ n = 7 :=
by
  sorry

end min_value_of_n_for_constant_term_l1771_177165


namespace injective_function_equality_l1771_177161

def injective (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b : ℕ⦄, f a = f b → a = b

theorem injective_function_equality
  {f : ℕ → ℕ}
  (h_injective : injective f)
  (h_eq : ∀ n m : ℕ, (1 / f n) + (1 / f m) = 4 / (f n + f m)) :
  ∀ n m : ℕ, m = n :=
by
  sorry

end injective_function_equality_l1771_177161


namespace pos_sum_of_powers_l1771_177119

theorem pos_sum_of_powers (a b c : ℝ) (n : ℕ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) : 
  a^n + b^n + c^n > 0 :=
sorry

end pos_sum_of_powers_l1771_177119


namespace logan_money_left_l1771_177101

-- Defining the given conditions
def income : ℕ := 65000
def rent_expense : ℕ := 20000
def groceries_expense : ℕ := 5000
def gas_expense : ℕ := 8000
def additional_income_needed : ℕ := 10000

-- Calculating total expenses
def total_expense : ℕ := rent_expense + groceries_expense + gas_expense

-- Desired income
def desired_income : ℕ := income + additional_income_needed

-- The theorem to prove
theorem logan_money_left : (desired_income - total_expense) = 42000 :=
by
  -- A placeholder for the proof
  sorry

end logan_money_left_l1771_177101


namespace bamboo_sections_volume_l1771_177154

theorem bamboo_sections_volume (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a n = a 0 + n * d) →
  (a 0 + a 1 + a 2 = 4) →
  (a 5 + a 6 + a 7 + a 8 = 3) →
  (a 3 + a 4 = 2 + 3 / 22) :=
sorry

end bamboo_sections_volume_l1771_177154


namespace odd_numbers_divisibility_l1771_177109

theorem odd_numbers_divisibility 
  (a b c : ℤ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : (ab - 1) % 4 = 0 ∨ (bc - 1) % 4 = 0 ∨ (ca - 1) % 4 = 0 := 
sorry

end odd_numbers_divisibility_l1771_177109


namespace P_2017_eq_14_l1771_177158

def sumOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def numberOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.length

def P (n : Nat) : Nat :=
  sumOfDigits n + numberOfDigits n

theorem P_2017_eq_14 : P 2017 = 14 :=
by
  sorry

end P_2017_eq_14_l1771_177158


namespace find_abc_l1771_177112

noncomputable def x (t : ℝ) := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) := 3 * Real.sin t

theorem find_abc :
  ∃ a b c : ℝ, 
  (a = 1/9) ∧ 
  (b = 4/27) ∧ 
  (c = 5/27) ∧ 
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1) :=
by
  sorry

end find_abc_l1771_177112


namespace max_length_of_cuts_l1771_177147

-- Define the dimensions of the board and the number of parts
def board_size : ℕ := 30
def num_parts : ℕ := 225

-- Define the total possible length of the cuts
def max_possible_cuts_length : ℕ := 1065

-- Define the condition that the board is cut into parts of equal area
def equal_area_partition (board_size num_parts : ℕ) : Prop :=
  ∃ (area_per_part : ℕ), (board_size * board_size) / num_parts = area_per_part

-- Define the theorem to prove the maximum possible total length of the cuts
theorem max_length_of_cuts (h : equal_area_partition board_size num_parts) :
  max_possible_cuts_length = 1065 :=
by
  -- Proof to be filled in
  sorry

end max_length_of_cuts_l1771_177147


namespace midpoint_trajectory_l1771_177159

theorem midpoint_trajectory (x y p q : ℝ) (h_parabola : p^2 = 4 * q)
  (h_focus : ∀ (p q : ℝ), p^2 = 4 * q → q = (p/2)^2) 
  (h_midpoint_x : x = (p + 1) / 2)
  (h_midpoint_y : y = q / 2):
  y^2 = 2 * x - 1 :=
by
  sorry

end midpoint_trajectory_l1771_177159


namespace complement_of_beta_l1771_177197

variable (α β : ℝ)
variable (compl : α + β = 180)
variable (alpha_greater_beta : α > β)

theorem complement_of_beta (h : α + β = 180) (h' : α > β) : 90 - β = (1 / 2) * (α - β) :=
by
  sorry

end complement_of_beta_l1771_177197


namespace only_one_statement_is_true_l1771_177153

theorem only_one_statement_is_true (A B C D E: Prop)
  (hA : A ↔ B)
  (hB : B ↔ ¬ E)
  (hC : C ↔ (A ∧ B ∧ C ∧ D ∧ E))
  (hD : D ↔ ¬ (A ∨ B ∨ C ∨ D ∨ E))
  (hE : E ↔ ¬ A)
  (h_unique : ∃! x, x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∧ x = True) : E :=
by
  sorry

end only_one_statement_is_true_l1771_177153


namespace area_of_circle_l1771_177120

-- Define the given conditions
def pi_approx : ℝ := 3
def radius : ℝ := 0.6

-- Prove that the area is 1.08 given the conditions
theorem area_of_circle : π = pi_approx → radius = 0.6 → 
  (pi_approx * radius^2 = 1.08) :=
by
  intros hπ hr
  sorry

end area_of_circle_l1771_177120


namespace gcd_bc_eq_one_l1771_177191

theorem gcd_bc_eq_one (a b c x y : ℕ)
  (h1 : Nat.gcd a b = 120)
  (h2 : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  Nat.gcd b c = 1 :=
by
  sorry

end gcd_bc_eq_one_l1771_177191


namespace max_marks_l1771_177103

theorem max_marks (score shortfall passing_threshold : ℝ) (h1 : score = 212) (h2 : shortfall = 19) (h3 : passing_threshold = 0.30) :
  ∃ M, M = 770 :=
by
  sorry

end max_marks_l1771_177103


namespace number_of_seedlings_l1771_177131

theorem number_of_seedlings (packets : ℕ) (seeds_per_packet : ℕ) (h1 : packets = 60) (h2 : seeds_per_packet = 7) : packets * seeds_per_packet = 420 :=
by
  sorry

end number_of_seedlings_l1771_177131


namespace sin_60_eq_sqrt3_div_2_l1771_177177

-- Problem statement translated to Lean
theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l1771_177177


namespace find_initial_interest_rate_l1771_177132

-- Definitions of the initial conditions
def P1 : ℝ := 3000
def P2 : ℝ := 1499.9999999999998
def P_total : ℝ := 4500
def r2 : ℝ := 0.08
def total_annual_income : ℝ := P_total * 0.06

-- Defining the problem as a statement to prove
theorem find_initial_interest_rate (r1 : ℝ) :
  (P1 * r1) + (P2 * r2) = total_annual_income → r1 = 0.05 := by
  sorry

end find_initial_interest_rate_l1771_177132


namespace find_divisor_l1771_177113

theorem find_divisor (d : ℕ) (h1 : d ∣ (9671 - 1)) : d = 9670 :=
by
  sorry

end find_divisor_l1771_177113


namespace max_blocks_fit_l1771_177114

theorem max_blocks_fit :
  ∃ (blocks : ℕ), blocks = 12 ∧ 
  (∀ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 1 → 
  ∀ (x y z : ℕ), x = 5 ∧ y = 4 ∧ z = 4 → 
  blocks = (x * y * z) / (a * b * c) ∧
  blocks = (y * z / (b * c) * (5 / a))) :=
sorry

end max_blocks_fit_l1771_177114


namespace geometric_sequence_a4_l1771_177164

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ {m n p q}, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : geometric_sequence a) (h2 : a 2 = 4) (h6 : a 6 = 16) :
  a 4 = 8 :=
by {
  -- Here you can provide the proof steps if needed
  sorry
}

end geometric_sequence_a4_l1771_177164


namespace parametric_to_cartesian_l1771_177183

theorem parametric_to_cartesian (θ : ℝ) (x y : ℝ) :
  (x = 1 + 2 * Real.cos θ) →
  (y = 2 * Real.sin θ) →
  (x - 1) ^ 2 + y ^ 2 = 4 :=
by 
  sorry

end parametric_to_cartesian_l1771_177183


namespace sequence_count_is_correct_l1771_177180

def has_integer_root (a_i a_i_plus_1 : ℕ) : Prop :=
  ∃ r : ℕ, r^2 - a_i * r + a_i_plus_1 = 0

def valid_sequence (seq : Fin 16 → ℕ) : Prop :=
  ∀ i : Fin 15, has_integer_root (seq i.val + 1) (seq (i + 1).val + 1) ∧ seq 15 = seq 0

-- This noncomputable definition is used because we are estimating a specific number without providing a concrete computable function.
noncomputable def sequence_count : ℕ :=
  1409

theorem sequence_count_is_correct :
  ∃ N, valid_sequence seq → N = 1409 :=
sorry 

end sequence_count_is_correct_l1771_177180


namespace journey_speed_first_half_l1771_177104

noncomputable def speed_first_half (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem journey_speed_first_half
  (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : total_distance = 224)
  (h3 : second_half_speed = 24) :
  speed_first_half total_time total_distance second_half_speed = 21 := by
  sorry

end journey_speed_first_half_l1771_177104


namespace contradiction_proof_real_root_l1771_177136

theorem contradiction_proof_real_root (a b : ℝ) :
  (∀ x : ℝ, x^3 + a * x + b ≠ 0) → (∃ x : ℝ, x + a * x + b = 0) :=
sorry

end contradiction_proof_real_root_l1771_177136


namespace intersection_points_l1771_177178

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * (x - y) - 18 = 0

theorem intersection_points : 
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by sorry

end intersection_points_l1771_177178


namespace induction_proof_l1771_177190

def f (n : ℕ) : ℕ := (List.range (2 * n - 1)).sum + n

theorem induction_proof (n : ℕ) (h : n > 0) : f (n + 1) - f n = 8 * n := by
  sorry

end induction_proof_l1771_177190


namespace total_students_l1771_177149

theorem total_students (h1 : ∀ (n : ℕ), n = 5 → Jaya_ranks_nth_from_top)
                       (h2 : ∀ (m : ℕ), m = 49 → Jaya_ranks_mth_from_bottom) :
  ∃ (total : ℕ), total = 53 :=
by
  sorry

end total_students_l1771_177149


namespace weighted_avg_M_B_eq_l1771_177133

-- Define the weightages and the given weighted total marks equation
def weight_physics : ℝ := 1.5
def weight_chemistry : ℝ := 2
def weight_mathematics : ℝ := 1.25
def weight_biology : ℝ := 1.75
def weighted_total_M_B : ℝ := 250
def weighted_sum_M_B : ℝ := weight_mathematics + weight_biology

-- Theorem statement: Prove that the weighted average mark for mathematics and biology is 83.33
theorem weighted_avg_M_B_eq :
  (weighted_total_M_B / weighted_sum_M_B) = 83.33 :=
by
  sorry

end weighted_avg_M_B_eq_l1771_177133


namespace no_feasible_distribution_l1771_177127

-- Define the initial conditions
def initial_runs_player_A : ℕ := 320
def initial_runs_player_B : ℕ := 450
def initial_runs_player_C : ℕ := 550

def initial_innings : ℕ := 10

def required_increase_A : ℕ := 4
def required_increase_B : ℕ := 5
def required_increase_C : ℕ := 6

def total_run_limit : ℕ := 250

-- Define the total runs required after 11 innings
def total_required_runs_after_11_innings (initial_runs avg_increase : ℕ) : ℕ :=
  (initial_runs / initial_innings + avg_increase) * 11

-- Calculate the additional runs needed in the next innings
def additional_runs_needed (initial_runs avg_increase : ℕ) : ℕ :=
  total_required_runs_after_11_innings initial_runs avg_increase - initial_runs

-- Calculate the total additional runs needed for all players
def total_additional_runs_needed : ℕ :=
  additional_runs_needed initial_runs_player_A required_increase_A +
  additional_runs_needed initial_runs_player_B required_increase_B +
  additional_runs_needed initial_runs_player_C required_increase_C

-- The statement to verify if the total additional required runs exceed the limit
theorem no_feasible_distribution :
  total_additional_runs_needed > total_run_limit :=
by 
  -- Skipping proofs and just stating the condition is what we aim to show.
  sorry

end no_feasible_distribution_l1771_177127


namespace ratio_of_investments_l1771_177141

theorem ratio_of_investments (P Q : ℝ)
  (h_ratio_profits : (20 * P) / (40 * Q) = 7 / 10) : P / Q = 7 / 5 := 
sorry

end ratio_of_investments_l1771_177141


namespace math_proof_problem_l1771_177100

-- Defining the problem condition
def condition (x y z : ℝ) := 
  x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0

-- Adding constraints to x, y, z
def constraints (x y z : ℝ) :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x ≠ y ∨ y ≠ z ∨ z ≠ x)

-- Stating the main theorem
theorem math_proof_problem (x y z : ℝ) (h_condition : condition x y z) (h_constraints : constraints x y z) :
  x + y + z = 3 ∧ x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6 := 
sorry

end math_proof_problem_l1771_177100


namespace find_a_for_even_function_l1771_177175

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = (x + 1) * (x + a) ∧ f (-x) = f x) : a = -1 := by 
  sorry

end find_a_for_even_function_l1771_177175


namespace trader_gain_percentage_l1771_177137

-- Definition of the given conditions
def cost_per_pen (C : ℝ) := C
def num_pens_sold := 90
def gain_from_sale (C : ℝ) := 15 * C
def total_cost (C : ℝ) := 90 * C

-- Statement of the problem
theorem trader_gain_percentage (C : ℝ) : 
  (((gain_from_sale C) / (total_cost C)) * 100) = 16.67 :=
by
  -- This part will contain the step-by-step proof, omitted here
  sorry

end trader_gain_percentage_l1771_177137


namespace price_change_theorem_l1771_177182

-- Define initial prices
def candy_box_price_before : ℝ := 10
def soda_can_price_before : ℝ := 9
def popcorn_bag_price_before : ℝ := 5
def gum_pack_price_before : ℝ := 2

-- Define price changes
def candy_box_price_increase := candy_box_price_before * 0.25
def soda_can_price_decrease := soda_can_price_before * 0.15
def popcorn_bag_price_factor := 2
def gum_pack_price_change := 0

-- Compute prices after the policy changes
def candy_box_price_after := candy_box_price_before + candy_box_price_increase
def soda_can_price_after := soda_can_price_before - soda_can_price_decrease
def popcorn_bag_price_after := popcorn_bag_price_before * popcorn_bag_price_factor
def gum_pack_price_after := gum_pack_price_before

-- Compute total costs
def total_cost_before := candy_box_price_before + soda_can_price_before + popcorn_bag_price_before + gum_pack_price_before
def total_cost_after := candy_box_price_after + soda_can_price_after + popcorn_bag_price_after + gum_pack_price_after

-- The statement to be proven
theorem price_change_theorem :
  total_cost_before = 26 ∧ total_cost_after = 32.15 :=
by
  -- This part requires proof, add 'sorry' for now
  sorry

end price_change_theorem_l1771_177182


namespace find_f_at_1_l1771_177171

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem find_f_at_1 : f 1 = 2 := by
  sorry

end find_f_at_1_l1771_177171


namespace circumscribed_sphere_surface_area_l1771_177118

-- Define the setup and conditions for the right circular cone and its circumscribed sphere
theorem circumscribed_sphere_surface_area (PA PB PC AB R : ℝ)
  (h1 : AB = Real.sqrt 2)
  (h2 : PA = 1)
  (h3 : PB = 1)
  (h4 : PC = 1)
  (h5 : R = Real.sqrt 3 / 2 * PA) :
  4 * Real.pi * R ^ 2 = 3 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l1771_177118


namespace new_member_younger_by_160_l1771_177139

theorem new_member_younger_by_160 
  (A : ℕ)  -- average age 8 years ago and today
  (O N : ℕ)  -- age of the old member and the new member respectively
  (h1 : 20 * A = 20 * A + O - N)  -- condition derived from the problem
  (h2 : 20 * 8 = 160)  -- age increase over 8 years for 20 members
  (h3 : O - N = 160) : O - N = 160 :=
by
  sorry

end new_member_younger_by_160_l1771_177139


namespace fraction_difference_of_squares_l1771_177117

theorem fraction_difference_of_squares :
  (175^2 - 155^2) / 20 = 330 :=
by
  -- Proof goes here
  sorry

end fraction_difference_of_squares_l1771_177117


namespace number_of_valid_polynomials_l1771_177126

noncomputable def count_polynomials_meeting_conditions : ℕ := sorry

theorem number_of_valid_polynomials :
  count_polynomials_meeting_conditions = 7200 :=
sorry

end number_of_valid_polynomials_l1771_177126


namespace f_periodic_l1771_177124

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_periodic (f : ℝ → ℝ)
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

end f_periodic_l1771_177124


namespace vertical_asymptote_l1771_177146

theorem vertical_asymptote (x : ℝ) : (y = (2*x - 3) / (4*x + 5)) → (4*x + 5 = 0) → x = -5/4 := 
by 
  intros h1 h2
  sorry

end vertical_asymptote_l1771_177146


namespace shaded_region_area_l1771_177102

theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  8 * (π * r * r / 4 - r * r / 2) / 2 = 50 * (π - 2) :=
by
  sorry

end shaded_region_area_l1771_177102


namespace range_of_m_l1771_177134

theorem range_of_m (A : Set ℝ) (m : ℝ) (h : ∃ x, x ∈ A ∩ {x | x ≠ 0}) :
  -4 < m ∧ m < 0 :=
by
  have A_def : A = {x | x^2 + (m+2)*x + 1 = 0} := sorry
  have h_non_empty : ∃ x, x ∈ A ∧ x ≠ 0 := sorry
  have discriminant : (m+2)^2 - 4 < 0 := sorry
  exact ⟨sorry, sorry⟩

end range_of_m_l1771_177134


namespace rectangle_perimeters_l1771_177150

theorem rectangle_perimeters (w h : ℝ) 
  (h1 : 2 * (w + h) = 20)
  (h2 : 2 * (4 * w + h) = 56) : 
  4 * (w + h) = 40 ∧ 2 * (w + 4 * h) = 44 := 
by
  sorry

end rectangle_perimeters_l1771_177150


namespace line_passes_through_fixed_point_l1771_177189

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : x = 6) 
  (h3 : y = 4) : 
  a * x + b * y - 10 = 0 := 
by
  sorry

end line_passes_through_fixed_point_l1771_177189


namespace remaining_pictures_l1771_177173

-- Definitions based on the conditions
def pictures_in_first_book : ℕ := 44
def pictures_in_second_book : ℕ := 35
def pictures_in_third_book : ℕ := 52
def pictures_in_fourth_book : ℕ := 48
def colored_pictures : ℕ := 37

-- Statement of the theorem based on the question and correct answer
theorem remaining_pictures :
  pictures_in_first_book + pictures_in_second_book + pictures_in_third_book + pictures_in_fourth_book - colored_pictures = 142 := by
  sorry

end remaining_pictures_l1771_177173


namespace price_of_Microtron_stock_l1771_177163

theorem price_of_Microtron_stock
  (n d : ℕ) (p_d p p_m : ℝ) 
  (h1 : n = 300) 
  (h2 : d = 150) 
  (h3 : p_d = 44) 
  (h4 : p = 40) 
  (h5 : p_m = 36) : 
  (d * p_d + (n - d) * p_m) / n = p := 
sorry

end price_of_Microtron_stock_l1771_177163


namespace no_integer_solution_l1771_177105

theorem no_integer_solution (x y z : ℤ) (h : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solution_l1771_177105


namespace hyperbola_condition_l1771_177148

theorem hyperbola_condition (k : ℝ) (x y : ℝ) :
  (k ≠ 0 ∧ k ≠ 3 ∧ (x^2 / k + y^2 / (k - 3) = 1)) → 0 < k ∧ k < 3 :=
by
  sorry

end hyperbola_condition_l1771_177148


namespace mutually_exclusive_not_complementary_l1771_177169

def event_odd (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_greater_than_5 (n : ℕ) : Prop := n = 6

theorem mutually_exclusive_not_complementary :
  (∀ n : ℕ, event_odd n → ¬ event_greater_than_5 n) ∧
  (∃ n : ℕ, ¬ event_odd n ∧ ¬ event_greater_than_5 n) :=
by
  sorry

end mutually_exclusive_not_complementary_l1771_177169


namespace positive_difference_two_numbers_l1771_177140

theorem positive_difference_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) : |x - y| = 8 := by
  sorry

end positive_difference_two_numbers_l1771_177140


namespace solve_quadratic_eq_l1771_177107

theorem solve_quadratic_eq (x : ℝ) : (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l1771_177107


namespace length_of_the_train_l1771_177174

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end length_of_the_train_l1771_177174


namespace male_salmon_count_l1771_177199

theorem male_salmon_count (total_count : ℕ) (female_count : ℕ) (male_count : ℕ) :
  total_count = 971639 →
  female_count = 259378 →
  male_count = (total_count - female_count) →
  male_count = 712261 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end male_salmon_count_l1771_177199


namespace money_distribution_l1771_177160

theorem money_distribution (p q r : ℝ) 
  (h1 : p + q + r = 9000) 
  (h2 : r = (2/3) * (p + q)) : 
  r = 3600 := 
by 
  sorry

end money_distribution_l1771_177160


namespace total_shaded_area_l1771_177138

def rectangle_area (R : ℝ) : ℝ := R * R
def square_area (S : ℝ) : ℝ := S * S

theorem total_shaded_area 
  (R S : ℝ)
  (h1 : 18 = 2 * R)
  (h2 : R = 4 * S) :
  rectangle_area R + 12 * square_area S = 141.75 := 
  by 
    sorry

end total_shaded_area_l1771_177138


namespace village_population_l1771_177145

variable (Px : ℕ)
variable (py : ℕ := 42000)
variable (years : ℕ := 16)
variable (rate_decrease_x : ℕ := 1200)
variable (rate_increase_y : ℕ := 800)

theorem village_population (Px : ℕ) (py : ℕ := 42000)
  (years : ℕ := 16) (rate_decrease_x : ℕ := 1200)
  (rate_increase_y : ℕ := 800) :
  Px - rate_decrease_x * years = py + rate_increase_y * years → Px = 74000 := by
  sorry

end village_population_l1771_177145


namespace smallest_number_of_roses_to_buy_l1771_177196

-- Definitions representing the conditions
def group_size1 : ℕ := 9
def group_size2 : ℕ := 19

-- Statement representing the problem and solution
theorem smallest_number_of_roses_to_buy : Nat.lcm group_size1 group_size2 = 171 := 
by 
  sorry

end smallest_number_of_roses_to_buy_l1771_177196


namespace bucket_full_weight_l1771_177116

variable (c d : ℝ)

def total_weight_definition (x y : ℝ) := x + y

theorem bucket_full_weight (x y : ℝ) 
  (h₁ : x + 3/4 * y = c) 
  (h₂ : x + 1/3 * y = d) : 
  total_weight_definition x y = (8 * c - 3 * d) / 5 :=
sorry

end bucket_full_weight_l1771_177116


namespace last_digit_2_pow_2023_l1771_177130

-- Definitions
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem last_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 :=
by
  -- We will assume and use the properties mentioned in the solution steps.
  -- The proof process is skipped here with 'sorry'.
  sorry

end last_digit_2_pow_2023_l1771_177130


namespace loss_equals_cost_price_of_balls_l1771_177166

variable (selling_price : ℕ) (cost_price_ball : ℕ)
variable (number_of_balls : ℕ) (loss_incurred : ℕ) (x : ℕ)

-- Conditions
def condition1 : selling_price = 720 := sorry -- Selling price of 11 balls is Rs. 720
def condition2 : cost_price_ball = 120 := sorry -- Cost price of one ball is Rs. 120
def condition3 : number_of_balls = 11 := sorry -- Number of balls is 11

-- Cost price of 11 balls
def cost_price (n : ℕ) (cp_ball : ℕ): ℕ := n * cp_ball

-- Loss incurred on selling 11 balls
def loss (cp : ℕ) (sp : ℕ): ℕ := cp - sp

-- Equation for number of balls the loss equates to
def loss_equation (l : ℕ) (cp_ball : ℕ): ℕ := l / cp_ball

theorem loss_equals_cost_price_of_balls : 
  ∀ (n sp cp_ball cp l: ℕ), 
  sp = 720 ∧ cp_ball = 120 ∧ n = 11 ∧ 
  cp = cost_price n cp_ball ∧ 
  l = loss cp sp →
  loss_equation l cp_ball = 5 := sorry

end loss_equals_cost_price_of_balls_l1771_177166


namespace eval_f_at_3_l1771_177192

def f (x : ℝ) : ℝ := 3 * x + 1

theorem eval_f_at_3 : f 3 = 10 :=
by
  -- computation of f at x = 3
  sorry

end eval_f_at_3_l1771_177192


namespace rhombus_area_correct_l1771_177179

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 80 120 = 4800 :=
by 
  -- the proof is skipped by including sorry
  sorry

end rhombus_area_correct_l1771_177179


namespace smallest_k_divisibility_l1771_177110

theorem smallest_k_divisibility : ∃ (k : ℕ), k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_divisibility_l1771_177110


namespace expand_and_simplify_l1771_177121

variable (x : ℝ)

theorem expand_and_simplify : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_and_simplify_l1771_177121


namespace polynomial_inequality_l1771_177123

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
sorry

end polynomial_inequality_l1771_177123


namespace bill_sunday_miles_l1771_177135

variable (B : ℕ)

-- Conditions
def miles_Bill_Saturday : ℕ := B
def miles_Bill_Sunday : ℕ := B + 4
def miles_Julia_Sunday : ℕ := 2 * (B + 4)
def total_miles : ℕ := miles_Bill_Saturday B + miles_Bill_Sunday B + miles_Julia_Sunday B

theorem bill_sunday_miles (h : total_miles B = 32) : miles_Bill_Sunday B = 9 := by
  sorry

end bill_sunday_miles_l1771_177135


namespace describe_shape_cylinder_l1771_177187

-- Define cylindrical coordinates
structure CylindricalCoordinates where
  r : ℝ -- radial distance
  θ : ℝ -- azimuthal angle
  z : ℝ -- height

-- Define the positive constant c
variable (c : ℝ) (hc : 0 < c)

-- The theorem statement
theorem describe_shape_cylinder (p : CylindricalCoordinates) (h : p.r = c) : 
  ∃ (p : CylindricalCoordinates), p.r = c :=
by
  sorry

end describe_shape_cylinder_l1771_177187


namespace inequality_proof_l1771_177152

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^3 + b^3 = 2) :
  (1 / a) + (1 / b) ≥ 2 * (a^2 - a + 1) * (b^2 - b + 1) := 
by
  sorry

end inequality_proof_l1771_177152


namespace factorial_expression_equals_l1771_177168

theorem factorial_expression_equals :
  7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end factorial_expression_equals_l1771_177168


namespace t_mobile_first_two_lines_cost_l1771_177184

theorem t_mobile_first_two_lines_cost :
  ∃ T : ℝ,
  (T + 16 * 3) = (45 + 14 * 3 + 11) → T = 50 :=
by
  sorry

end t_mobile_first_two_lines_cost_l1771_177184


namespace sum_200_to_299_l1771_177172

variable (a : ℕ)

-- Condition: Sum of the first 100 natural numbers is equal to a
def sum_100 := (100 * 101) / 2

-- Main Theorem: Sum from 200 to 299 in terms of a
theorem sum_200_to_299 (h : sum_100 = a) : (299 * 300 / 2 - 199 * 200 / 2) = 19900 + a := by
  sorry

end sum_200_to_299_l1771_177172


namespace largest_a_l1771_177181

open Real

theorem largest_a (a b c : ℝ) (h1 : a + b + c = 6) (h2 : ab + ac + bc = 11) : 
  a ≤ 2 + 2 * sqrt 3 / 3 :=
sorry

end largest_a_l1771_177181


namespace net_sag_calculation_l1771_177193

open Real

noncomputable def sag_of_net (m1 m2 h1 h2 x1 : ℝ) : ℝ :=
  let g := 9.81
  let a := 28
  let b := -1.75
  let c := -50.75
  let D := b^2 - 4*a*c
  let sqrtD := sqrt D
  (1.75 + sqrtD) / (2 * a)

theorem net_sag_calculation :
  let m1 := 78.75
  let x1 := 1
  let h1 := 15
  let m2 := 45
  let h2 := 29
  sag_of_net m1 m2 h1 h2 x1 = 1.38 := 
by
  sorry

end net_sag_calculation_l1771_177193


namespace number_of_girls_l1771_177122

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l1771_177122


namespace range_of_a_l1771_177142

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end range_of_a_l1771_177142


namespace total_hours_over_two_weeks_l1771_177156

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l1771_177156


namespace time_needed_by_Alpha_and_Beta_l1771_177157

theorem time_needed_by_Alpha_and_Beta (A B C h : ℝ)
  (h₀ : 1 / (A - 4) = 1 / (B - 2))
  (h₁ : 1 / A + 1 / B + 1 / C = 3 / C)
  (h₂ : A = B + 2)
  (h₃ : 1 / 12 + 1 / 10 = 11 / 60)
  : h = 60 / 11 :=
sorry

end time_needed_by_Alpha_and_Beta_l1771_177157


namespace smallest_even_x_l1771_177176

theorem smallest_even_x (x : ℤ) (h1 : x < 3 * x - 10) (h2 : ∃ k : ℤ, x = 2 * k) : x = 6 :=
by {
  sorry
}

end smallest_even_x_l1771_177176


namespace percentage_more_l1771_177128

variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.90 * J
def Mary_income : Prop := M = 1.44 * J

-- Theorem to be proved
theorem percentage_more (h1 : Tim_income J T) (h2 : Mary_income J M) :
  ((M - T) / T) * 100 = 60 :=
sorry

end percentage_more_l1771_177128


namespace no_solution_fraction_eq_l1771_177129

theorem no_solution_fraction_eq {x m : ℝ} : 
  (∀ x, ¬ (1 - x = 0) → (2 - x) / (1 - x) = (m + x) / (1 - x) + 1) ↔ m = 0 := 
by
  sorry

end no_solution_fraction_eq_l1771_177129


namespace car_speed_40_kmph_l1771_177108

theorem car_speed_40_kmph (v : ℝ) (h : 1 / v = 1 / 48 + 15 / 3600) : v = 40 := 
sorry

end car_speed_40_kmph_l1771_177108


namespace fbox_eval_correct_l1771_177143

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end fbox_eval_correct_l1771_177143


namespace trigonometric_identity_l1771_177115

open Real

theorem trigonometric_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) : 
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = 2 / tan α := 
by
  sorry

end trigonometric_identity_l1771_177115


namespace commercial_break_total_time_l1771_177155

theorem commercial_break_total_time (c1 c2 c3 : ℕ) (c4 : ℕ → ℕ) (interrupt restart : ℕ) 
  (h1 : c1 = 5) (h2 : c2 = 6) (h3 : c3 = 7) 
  (h4 : ∀ i, i < 11 → c4 i = 2) 
  (h_interrupt : interrupt = 3)
  (h_restart : restart = 2) :
  c1 + c2 + c3 + (11 * 2) + interrupt + 2 * restart = 47 := 
  by
  sorry

end commercial_break_total_time_l1771_177155
