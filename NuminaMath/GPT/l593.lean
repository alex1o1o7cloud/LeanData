import Mathlib

namespace NUMINAMATH_GPT_find_incomes_l593_59309

theorem find_incomes (M N O P Q : ℝ) 
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (O + P) / 2 = 6800)
  (h4 : (P + Q) / 2 = 7500)
  (h5 : (M + O + Q) / 3 = 6000) :
  M = 300 ∧ N = 9800 ∧ O = 2700 ∧ P = 10900 ∧ Q = 4100 :=
by
  sorry


end NUMINAMATH_GPT_find_incomes_l593_59309


namespace NUMINAMATH_GPT_find_three_digit_number_l593_59315

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l593_59315


namespace NUMINAMATH_GPT_find_b_l593_59351

variable {a b d m : ℝ}

theorem find_b (h : m = d * a * b / (a + b)) : b = m * a / (d * a - m) :=
sorry

end NUMINAMATH_GPT_find_b_l593_59351


namespace NUMINAMATH_GPT_eval_x_squared_minus_y_squared_l593_59385

theorem eval_x_squared_minus_y_squared (x y : ℝ) (h1 : 3 * x + 2 * y = 30) (h2 : 4 * x + 2 * y = 34) : x^2 - y^2 = -65 :=
by
  sorry

end NUMINAMATH_GPT_eval_x_squared_minus_y_squared_l593_59385


namespace NUMINAMATH_GPT_domain_of_function_l593_59310

theorem domain_of_function :
  {x : ℝ | ∀ k : ℤ, 2 * x + (π / 4) ≠ k * π + (π / 2)}
  = {x : ℝ | ∀ k : ℤ, x ≠ (k * π / 2) + (π / 8)} :=
sorry

end NUMINAMATH_GPT_domain_of_function_l593_59310


namespace NUMINAMATH_GPT_find_length_of_AC_l593_59366

theorem find_length_of_AC
  (A B C : Type)
  (AB : Real)
  (AC : Real)
  (Area : Real)
  (angle_A : Real)
  (h1 : AB = 8)
  (h2 : angle_A = (30 * Real.pi / 180)) -- converting degrees to radians
  (h3 : Area = 16) :
  AC = 8 :=
by
  -- Skipping proof as requested
  sorry

end NUMINAMATH_GPT_find_length_of_AC_l593_59366


namespace NUMINAMATH_GPT_correct_speed_l593_59379

noncomputable def distance (t : ℝ) := 50 * (t + 5 / 60)
noncomputable def distance2 (t : ℝ) := 70 * (t - 5 / 60)

theorem correct_speed : 
  ∃ r : ℝ, 
    (∀ t : ℝ, distance t = distance2 t → r = 55) := 
by
  sorry

end NUMINAMATH_GPT_correct_speed_l593_59379


namespace NUMINAMATH_GPT_eval_infinite_product_l593_59350

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, (3:ℝ)^(2 * n / (3:ℝ)^n)

theorem eval_infinite_product : infinite_product = (3:ℝ)^(9 / 2) := by
  sorry

end NUMINAMATH_GPT_eval_infinite_product_l593_59350


namespace NUMINAMATH_GPT_probability_red_ball_l593_59357

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_ball_l593_59357


namespace NUMINAMATH_GPT_vera_operations_impossible_l593_59326

theorem vera_operations_impossible (N : ℕ) : (N % 3 ≠ 0) → ¬(∃ k : ℕ, ((N + 3 * k) % 5 = 0) → ((N + 3 * k) / 5) = 1) :=
by
  sorry

end NUMINAMATH_GPT_vera_operations_impossible_l593_59326


namespace NUMINAMATH_GPT_average_points_per_player_l593_59305

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end NUMINAMATH_GPT_average_points_per_player_l593_59305


namespace NUMINAMATH_GPT_x_finishes_in_24_days_l593_59302

variable (x y : Type) [Inhabited x] [Inhabited y]

/-- 
  y can finish the work in 16 days,
  y worked for 10 days and left the job,
  x alone needs 9 days to finish the remaining work,
  How many days does x need to finish the work alone?
-/
theorem x_finishes_in_24_days
  (days_y : ℕ := 16)
  (work_done_y : ℕ := 10)
  (work_left_x : ℕ := 9)
  (D_x : ℕ) :
  (1 / days_y : ℚ) * work_done_y + (1 / D_x) * work_left_x = 1 / D_x :=
by
  sorry

end NUMINAMATH_GPT_x_finishes_in_24_days_l593_59302


namespace NUMINAMATH_GPT_g_value_at_4_l593_59342

noncomputable def g : ℝ → ℝ := sorry -- We will define g here

def functional_condition (g : ℝ → ℝ) := ∀ x y : ℝ, x * g y = y * g x
def g_value_at_12 := g 12 = 30

theorem g_value_at_4 (g : ℝ → ℝ) (h₁ : functional_condition g) (h₂ : g_value_at_12) : g 4 = 10 := 
sorry

end NUMINAMATH_GPT_g_value_at_4_l593_59342


namespace NUMINAMATH_GPT_solution_set_of_inequality_l593_59330

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x ≥ 0} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l593_59330


namespace NUMINAMATH_GPT_polynomial_product_c_l593_59300

theorem polynomial_product_c (b c : ℝ) (h1 : b = 2 * c - 1) (h2 : (x^2 + b * x + c) = 0 → (∃ r : ℝ, x = r)) :
  c = 1 / 2 :=
sorry

end NUMINAMATH_GPT_polynomial_product_c_l593_59300


namespace NUMINAMATH_GPT_solve_equation_l593_59387

noncomputable def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * x ^ (2021 / 2022) - 1

theorem solve_equation : ∀ x : ℝ, equation x ↔ x = 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l593_59387


namespace NUMINAMATH_GPT_ratio_of_boys_l593_59394

theorem ratio_of_boys (p : ℚ) (hp : p = (3 / 4) * (1 - p)) : p = 3 / 7 :=
by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l593_59394


namespace NUMINAMATH_GPT_trigonometric_ratio_l593_59328

theorem trigonometric_ratio (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_ratio_l593_59328


namespace NUMINAMATH_GPT_geometric_sequence_a4_value_l593_59311

theorem geometric_sequence_a4_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h1 : a 1 + (2 / 3) * a 2 = 3) 
  (h2 : a 4^2 = (1 / 9) * a 3 * a 7) 
  :
  a 4 = 27 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a4_value_l593_59311


namespace NUMINAMATH_GPT_no_integer_n_gt_1_satisfies_inequality_l593_59371

open Int

theorem no_integer_n_gt_1_satisfies_inequality :
  ∀ (n : ℤ), n > 1 → ¬ (⌊(Real.sqrt (↑n - 2) + 2 * Real.sqrt (↑n + 2))⌋ < ⌊Real.sqrt (9 * (↑n : ℝ) + 6)⌋) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_no_integer_n_gt_1_satisfies_inequality_l593_59371


namespace NUMINAMATH_GPT_preimages_of_f_l593_59356

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem preimages_of_f (k : ℝ) : (∃ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ ≠ x₂) ↔ k < 1 := by
  sorry

end NUMINAMATH_GPT_preimages_of_f_l593_59356


namespace NUMINAMATH_GPT_pigeons_in_house_l593_59380

variable (x F c : ℝ)

theorem pigeons_in_house 
  (H1 : F = (x - 75) * 20 * c)
  (H2 : F = (x + 100) * 15 * c) :
  x = 600 := by
  sorry

end NUMINAMATH_GPT_pigeons_in_house_l593_59380


namespace NUMINAMATH_GPT_original_cost_price_l593_59388

theorem original_cost_price (P : ℝ) 
  (h1 : P - 0.07 * P = 0.93 * P)
  (h2 : 0.93 * P + 0.02 * 0.93 * P = 0.9486 * P)
  (h3 : 0.9486 * P * 1.05 = 0.99603 * P)
  (h4 : 0.93 * P * 0.95 = 0.8835 * P)
  (h5 : 0.8835 * P + 0.02 * 0.8835 * P = 0.90117 * P)
  (h6 : 0.99603 * P - 5 = (0.90117 * P) * 1.10)
: P = 5 / 0.004743 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_l593_59388


namespace NUMINAMATH_GPT_parabola_shift_right_by_3_l593_59331

theorem parabola_shift_right_by_3 :
  ∀ (x : ℝ), (∃ y₁ y₂ : ℝ, y₁ = 2 * x^2 ∧ y₂ = 2 * (x - 3)^2) →
  (∃ (h : ℝ), h = 3) :=
sorry

end NUMINAMATH_GPT_parabola_shift_right_by_3_l593_59331


namespace NUMINAMATH_GPT_find_greater_number_l593_59367

theorem find_greater_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 6) (h3 : x * y = 216) (h4 : x > y) : x = 18 := 
sorry

end NUMINAMATH_GPT_find_greater_number_l593_59367


namespace NUMINAMATH_GPT_A_finishes_work_in_9_days_l593_59322

noncomputable def B_work_rate : ℝ := 1 / 15
noncomputable def B_work_10_days : ℝ := 10 * B_work_rate
noncomputable def remaining_work_by_A : ℝ := 1 - B_work_10_days

theorem A_finishes_work_in_9_days (A_days : ℝ) (B_days : ℝ) (B_days_worked : ℝ) (A_days_worked : ℝ) :
  (B_days = 15) ∧ (B_days_worked = 10) ∧ (A_days_worked = 3) ∧ 
  (remaining_work_by_A = (1 / 3)) → A_days = 9 :=
by sorry

end NUMINAMATH_GPT_A_finishes_work_in_9_days_l593_59322


namespace NUMINAMATH_GPT_players_taking_chemistry_l593_59354

theorem players_taking_chemistry (total_players biology_players both_sci_players: ℕ) 
  (h1 : total_players = 12)
  (h2 : biology_players = 7)
  (h3 : both_sci_players = 2)
  (h4 : ∀ p, p <= total_players) : 
  ∃ chemistry_players, chemistry_players = 7 := 
sorry

end NUMINAMATH_GPT_players_taking_chemistry_l593_59354


namespace NUMINAMATH_GPT_frac_addition_l593_59395

theorem frac_addition :
  (3 / 5) + (2 / 15) = 11 / 15 :=
sorry

end NUMINAMATH_GPT_frac_addition_l593_59395


namespace NUMINAMATH_GPT_determinant_matrices_equivalence_l593_59333

-- Define the problem as a Lean theorem statement
theorem determinant_matrices_equivalence (p q r s : ℝ) 
  (h : p * s - q * r = 3) : 
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_determinant_matrices_equivalence_l593_59333


namespace NUMINAMATH_GPT_relay_race_total_time_is_correct_l593_59398

-- Define the time taken by each runner
def time_Ainslee : ℕ := 72
def time_Bridget : ℕ := (10 * time_Ainslee) / 9
def time_Cecilia : ℕ := (3 * time_Bridget) / 4
def time_Dana : ℕ := (5 * time_Cecilia) / 6

-- Define the total time and convert to minutes and seconds
def total_time_seconds : ℕ := time_Ainslee + time_Bridget + time_Cecilia + time_Dana
def total_time_minutes := total_time_seconds / 60
def total_time_remainder := total_time_seconds % 60

theorem relay_race_total_time_is_correct :
  total_time_minutes = 4 ∧ total_time_remainder = 22 :=
by
  -- All intermediate values can be calculated using the definitions
  -- provided above correctly.
  sorry

end NUMINAMATH_GPT_relay_race_total_time_is_correct_l593_59398


namespace NUMINAMATH_GPT_valid_bases_for_625_l593_59303

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end NUMINAMATH_GPT_valid_bases_for_625_l593_59303


namespace NUMINAMATH_GPT_minimum_pizzas_needed_l593_59319

variables (p : ℕ)

def income_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4
def maintenance_cost_per_pizza : ℕ := 1
def car_cost : ℕ := 6500

theorem minimum_pizzas_needed :
  p ≥ 929 ↔ (income_per_pizza * p - (gas_cost_per_pizza + maintenance_cost_per_pizza) * p) ≥ car_cost :=
sorry

end NUMINAMATH_GPT_minimum_pizzas_needed_l593_59319


namespace NUMINAMATH_GPT_east_bound_cyclist_speed_l593_59390

-- Define the speeds of the cyclists and the relationship between them
def east_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * x
def west_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * (x + 4)

-- Condition: After 5 hours, they are 200 miles apart
def total_distance (t : ℕ) (x : ℕ) : ℕ := east_bound_speed t x + west_bound_speed t x

theorem east_bound_cyclist_speed :
  ∃ x : ℕ, total_distance 5 x = 200 ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_east_bound_cyclist_speed_l593_59390


namespace NUMINAMATH_GPT_sum_of_first_3n_terms_l593_59341

theorem sum_of_first_3n_terms (n : ℕ) (sn s2n s3n : ℕ) 
  (h1 : sn = 48) (h2 : s2n = 60)
  (h3 : s2n - sn = s3n - s2n) (h4 : 2 * (s2n - sn) = sn + (s3n - s2n)) :
  s3n = 36 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_first_3n_terms_l593_59341


namespace NUMINAMATH_GPT_problem1_problem2_l593_59314

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l593_59314


namespace NUMINAMATH_GPT_total_reams_l593_59316

theorem total_reams (h_r : ℕ) (s_r : ℕ) : h_r = 2 → s_r = 3 → h_r + s_r = 5 :=
by
  intro h_eq s_eq
  sorry

end NUMINAMATH_GPT_total_reams_l593_59316


namespace NUMINAMATH_GPT_triangle_inequality_l593_59399

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l593_59399


namespace NUMINAMATH_GPT_curve_intersection_l593_59376

theorem curve_intersection (a m : ℝ) (a_pos : 0 < a) :
  (∀ x y : ℝ, 
     (x^2 / a^2 + y^2 = 1) ∧ (y^2 = 2 * (x + m)) 
     → 
     (1 / 2 * (a^2 + 1) = m) ∨ (-a < m ∧ m <= a))
  ∨ (a >= 1 → -a < m ∧ m < a) := 
sorry

end NUMINAMATH_GPT_curve_intersection_l593_59376


namespace NUMINAMATH_GPT_valid_q_range_l593_59383

noncomputable def polynomial_has_nonneg_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ (x^4 + q*x^3 + x^2 + q*x + 4 = 0)

theorem valid_q_range (q : ℝ) : polynomial_has_nonneg_root q → q ≤ -2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_valid_q_range_l593_59383


namespace NUMINAMATH_GPT_marcus_calzones_total_time_l593_59304

/-
Conditions:
1. It takes Marcus 20 minutes to saute the onions.
2. It takes a quarter of the time to saute the garlic and peppers that it takes to saute the onions.
3. It takes 30 minutes to knead the dough.
4. It takes twice as long to let the dough rest as it takes to knead it.
5. It takes 1/10th of the combined kneading and resting time to assemble the calzones.
-/

def time_saute_onions : ℕ := 20
def time_saute_garlic_peppers : ℕ := time_saute_onions / 4
def time_knead : ℕ := 30
def time_rest : ℕ := 2 * time_knead
def time_assemble : ℕ := (time_knead + time_rest) / 10

def total_time_making_calzones : ℕ :=
  time_saute_onions + time_saute_garlic_peppers + time_knead + time_rest + time_assemble

theorem marcus_calzones_total_time : total_time_making_calzones = 124 := by
  -- All steps and proof details to be filled in
  sorry

end NUMINAMATH_GPT_marcus_calzones_total_time_l593_59304


namespace NUMINAMATH_GPT_problem_equivalent_proof_statement_l593_59364

-- Definition of a line with a definite slope
def has_definite_slope (m : ℝ) : Prop :=
  ∃ slope : ℝ, slope = -m 

-- Definition of the equation of a line passing through two points being correct
def line_through_two_points (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2) : Prop :=
  ∀ x y : ℝ, (y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)) ↔ y = ((y2 - y1) * (x - x1) / (x2 - x1)) + y1 

-- Formalizing and proving the given conditions
theorem problem_equivalent_proof_statement : 
  (∀ m : ℝ, has_definite_slope m) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2), line_through_two_points x1 y1 x2 y2 h) :=
by 
  sorry

end NUMINAMATH_GPT_problem_equivalent_proof_statement_l593_59364


namespace NUMINAMATH_GPT_father_l593_59365

-- Definitions based on conditions in a)
def cost_MP3_player : ℕ := 120
def cost_CD : ℕ := 19
def total_cost : ℕ := cost_MP3_player + cost_CD
def savings : ℕ := 55
def amount_lacking : ℕ := 64

-- Statement of the proof problem
theorem father's_contribution : (savings + (148:ℕ) - amount_lacking = total_cost) := by
  -- Add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_father_l593_59365


namespace NUMINAMATH_GPT_farmer_pays_per_acre_per_month_l593_59332

-- Define the conditions
def total_payment : ℕ := 600
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Define the problem to prove
theorem farmer_pays_per_acre_per_month :
  length_of_plot * width_of_plot / square_feet_per_acre > 0 ∧
  total_payment / (length_of_plot * width_of_plot / square_feet_per_acre) = 60 :=
by
  -- skipping the actual proof for now
  sorry

end NUMINAMATH_GPT_farmer_pays_per_acre_per_month_l593_59332


namespace NUMINAMATH_GPT_total_dolls_l593_59355

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end NUMINAMATH_GPT_total_dolls_l593_59355


namespace NUMINAMATH_GPT_total_votes_l593_59343

-- Define the given conditions
def candidate_votes (V : ℝ) : ℝ := 0.35 * V
def rival_votes (V : ℝ) : ℝ := 0.35 * V + 1800

-- Prove the total number of votes cast
theorem total_votes (V : ℝ) (h : candidate_votes V + rival_votes V = V) : V = 6000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l593_59343


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l593_59345

-- Problem 1
theorem problem1
  (α : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (hα : 0 < α ∧ α < 2 * Real.pi / 3) :
  (a + b) • (a - b) = 0 :=
sorry

-- Problem 2
theorem problem2
  (α k : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (x : ℝ × ℝ := k • a + 3 • b)
  (y : ℝ × ℝ := a + (1 / k) • b)
  (hk : 0 < k)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hxy : x • y = 0) :
  k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0 :=
sorry

-- Problem 3
theorem problem3
  (α k : ℝ)
  (h_eq : k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hk : 0 < k) :
  Real.pi / 2 ≤ α ∧ α < 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l593_59345


namespace NUMINAMATH_GPT_triangle_base_length_l593_59329

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 24) (h_height : height = 8) (h_area_formula : area = (base * height) / 2) :
  base = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l593_59329


namespace NUMINAMATH_GPT_equal_total_areas_of_checkerboard_pattern_l593_59361

-- Definition representing the convex quadrilateral and its subdivisions
structure ConvexQuadrilateral :=
  (A B C D : ℝ × ℝ) -- vertices of the quadrilateral

-- Predicate indicating the subdivision and coloring pattern
inductive CheckerboardColor
  | Black
  | White

-- Function to determine the area of the resulting smaller quadrilateral
noncomputable def area_of_subquadrilateral 
  (quad : ConvexQuadrilateral) 
  (subdivision : ℕ) -- subdivision factor
  (color : CheckerboardColor) 
  : ℝ := -- returns the area based on the subdivision and color
  -- Simplified implementation of area calculation
  -- (detailed geometric computation should replace this placeholder)
  sorry

-- Function to determine the total area of quadrilaterals of a given color
noncomputable def total_area_of_color 
  (quad : ConvexQuadrilateral) 
  (substution : ℕ) 
  (color : CheckerboardColor) 
  : ℝ := -- Total area of subquadrilaterals of the given color
  sorry

-- Theorem stating the required proof
theorem equal_total_areas_of_checkerboard_pattern
  (quad : ConvexQuadrilateral)
  (subdivision : ℕ)
  : total_area_of_color quad subdivision CheckerboardColor.Black = total_area_of_color quad subdivision CheckerboardColor.White :=
  sorry

end NUMINAMATH_GPT_equal_total_areas_of_checkerboard_pattern_l593_59361


namespace NUMINAMATH_GPT_problem_statement_l593_59339

theorem problem_statement (p q : ℝ)
  (α β : ℝ) (h1 : α ≠ β) (h1' : α + β = -p) (h1'' : α * β = -2)
  (γ δ : ℝ) (h2 : γ ≠ δ) (h2' : γ + δ = -q) (h2'' : γ * δ = -3) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2) - 2 * q + 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l593_59339


namespace NUMINAMATH_GPT_negation_of_universal_statement_l593_59389

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l593_59389


namespace NUMINAMATH_GPT_smallest_five_digit_divisible_by_53_l593_59308

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_divisible_by_53_l593_59308


namespace NUMINAMATH_GPT_largest_positive_real_root_l593_59358

theorem largest_positive_real_root (b2 b1 b0 : ℤ) (h2 : |b2| ≤ 3) (h1 : |b1| ≤ 3) (h0 : |b0| ≤ 3) :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + (b2 : ℝ) * r^2 + (b1 : ℝ) * r + (b0 : ℝ) = 0) ∧ 3.5 < r ∧ r < 4.0 :=
sorry

end NUMINAMATH_GPT_largest_positive_real_root_l593_59358


namespace NUMINAMATH_GPT_solve_equation_l593_59320

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l593_59320


namespace NUMINAMATH_GPT_water_wheel_effective_horsepower_l593_59353

noncomputable def effective_horsepower 
  (velocity : ℝ) (width : ℝ) (thickness : ℝ) (density : ℝ) 
  (diameter : ℝ) (efficiency : ℝ) (g : ℝ) (hp_conversion : ℝ) : ℝ :=
  let mass_flow_rate := velocity * width * thickness * density
  let kinetic_energy_per_second := 0.5 * mass_flow_rate * velocity^2
  let potential_energy_per_second := mass_flow_rate * diameter * g
  let indicated_power := kinetic_energy_per_second + potential_energy_per_second
  let horsepower := indicated_power / hp_conversion
  efficiency * horsepower

theorem water_wheel_effective_horsepower :
  effective_horsepower 1.4 0.5 0.13 1000 3 0.78 9.81 745.7 = 2.9 :=
by
  sorry

end NUMINAMATH_GPT_water_wheel_effective_horsepower_l593_59353


namespace NUMINAMATH_GPT_sammy_pickles_l593_59318

theorem sammy_pickles 
  (T S R : ℕ) 
  (h1 : T = 2 * S) 
  (h2 : R = 8 * T / 10) 
  (h3 : R = 24) : 
  S = 15 :=
by
  sorry

end NUMINAMATH_GPT_sammy_pickles_l593_59318


namespace NUMINAMATH_GPT_number_of_solutions_l593_59323

theorem number_of_solutions (n : ℕ) : (4 * n) = 80 ↔ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l593_59323


namespace NUMINAMATH_GPT_bruce_anne_clean_in_4_hours_l593_59324

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end NUMINAMATH_GPT_bruce_anne_clean_in_4_hours_l593_59324


namespace NUMINAMATH_GPT_henry_books_l593_59392

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end NUMINAMATH_GPT_henry_books_l593_59392


namespace NUMINAMATH_GPT_rectangle_diagonals_not_perpendicular_l593_59306

-- Definition of a rectangle through its properties
structure Rectangle (α : Type _) [LinearOrderedField α] :=
  (angle_eq : ∀ (a : α), a = 90)
  (diagonals_eq : ∀ (d1 d2 : α), d1 = d2)
  (diagonals_bisect : ∀ (d1 d2 : α), d1 / 2 = d2 / 2)

-- Theorem stating that a rectangle's diagonals are not necessarily perpendicular
theorem rectangle_diagonals_not_perpendicular (α : Type _) [LinearOrderedField α] (R : Rectangle α) : 
  ¬ (∀ (d1 d2 : α), d1 * d2 = 0) :=
sorry

end NUMINAMATH_GPT_rectangle_diagonals_not_perpendicular_l593_59306


namespace NUMINAMATH_GPT_percent_of_b_is_50_l593_59359

variable (a b c : ℝ)

-- Conditions
def c_is_25_percent_of_a : Prop := c = 0.25 * a
def b_is_50_percent_of_a : Prop := b = 0.50 * a

-- Proof
theorem percent_of_b_is_50 :
  c_is_25_percent_of_a c a → b_is_50_percent_of_a b a → c = 0.50 * b :=
by sorry

end NUMINAMATH_GPT_percent_of_b_is_50_l593_59359


namespace NUMINAMATH_GPT_number_of_boys_l593_59378

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end NUMINAMATH_GPT_number_of_boys_l593_59378


namespace NUMINAMATH_GPT_solution_set_empty_l593_59369

theorem solution_set_empty (x : ℝ) : ¬ (|x| + |2023 - x| < 2023) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_empty_l593_59369


namespace NUMINAMATH_GPT_inequality_proof_l593_59325

theorem inequality_proof (x : ℝ) (hx : 0 < x) : (1 / x) + 4 * (x ^ 2) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l593_59325


namespace NUMINAMATH_GPT_not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l593_59321

def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ * x₁ + b * x₁ + c = 0 ∧ a * x₂ * x₂ + b * x₂ + c = 0 
  ∧ (x₁ - x₂ = 1 ∨ x₂ - x₁ = 1)

theorem not_neighboring_root_equation_x2_x_2 : 
  ¬ is_neighboring_root_equation 1 1 (-2) :=
sorry

theorem neighboring_root_equation_k_values (k : ℝ) : 
  is_neighboring_root_equation 1 (-(k-3)) (-3*k) ↔ k = -2 ∨ k = -4 :=
sorry

end NUMINAMATH_GPT_not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l593_59321


namespace NUMINAMATH_GPT_female_democrats_l593_59347

theorem female_democrats :
  ∀ (F M : ℕ),
  F + M = 720 →
  F/2 + M/4 = 240 →
  F / 2 = 120 :=
by
  intros F M h1 h2
  sorry

end NUMINAMATH_GPT_female_democrats_l593_59347


namespace NUMINAMATH_GPT_cost_of_each_barbell_l593_59352

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_barbell_l593_59352


namespace NUMINAMATH_GPT_power_division_result_l593_59338

theorem power_division_result : (-2)^(2014) / (-2)^(2013) = -2 :=
by
  sorry

end NUMINAMATH_GPT_power_division_result_l593_59338


namespace NUMINAMATH_GPT_arctan_arcsin_arccos_sum_l593_59391

theorem arctan_arcsin_arccos_sum :
  (Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1 / 2) + Real.arccos 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_arctan_arcsin_arccos_sum_l593_59391


namespace NUMINAMATH_GPT_domain_of_v_l593_59375

def domain_v (x : ℝ) : Prop :=
  x ≥ 2 ∧ x ≠ 5

theorem domain_of_v :
  {x : ℝ | domain_v x} = { x | 2 < x ∧ x < 5 } ∪ { x | 5 < x }
:= by
  sorry

end NUMINAMATH_GPT_domain_of_v_l593_59375


namespace NUMINAMATH_GPT_sum_a_b_eq_five_l593_59382

theorem sum_a_b_eq_five (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_sum_a_b_eq_five_l593_59382


namespace NUMINAMATH_GPT_find_x_given_y_and_ratio_l593_59381

variable (x y k : ℝ)

theorem find_x_given_y_and_ratio :
  (∀ x y, (5 * x - 6) / (2 * y + 20) = k) →
  (5 * 3 - 6) / (2 * 5 + 20) = k →
  y = 15 →
  x = 21 / 5 :=
by 
  intro h1 h2 hy
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_find_x_given_y_and_ratio_l593_59381


namespace NUMINAMATH_GPT_sculpture_paint_area_l593_59373

/-- An artist creates a sculpture using 15 cubes, each with a side length of 1 meter. 
The cubes are organized into a wall-like structure with three layers: 
the top layer consists of 3 cubes, 
the middle layer consists of 5 cubes, 
and the bottom layer consists of 7 cubes. 
Some of the cubes in the middle and bottom layers are spaced apart, exposing additional side faces. 
Prove that the total exposed surface area painted is 49 square meters. -/
theorem sculpture_paint_area :
  let cubes_sizes : ℕ := 15
  let layer_top : ℕ := 3
  let layer_middle : ℕ := 5
  let layer_bottom : ℕ := 7
  let side_exposed_area_layer_top : ℕ := layer_top * 5
  let side_exposed_area_layer_middle : ℕ := 2 * 3 + 3 * 2
  let side_exposed_area_layer_bottom : ℕ := layer_bottom * 1
  let exposed_side_faces : ℕ := side_exposed_area_layer_top + side_exposed_area_layer_middle + side_exposed_area_layer_bottom
  let exposed_top_faces : ℕ := layer_top * 1 + layer_middle * 1 + layer_bottom * 1
  let total_exposed_area : ℕ := exposed_side_faces + exposed_top_faces
  total_exposed_area = 49 := 
sorry

end NUMINAMATH_GPT_sculpture_paint_area_l593_59373


namespace NUMINAMATH_GPT_inequality_solution_l593_59377

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end NUMINAMATH_GPT_inequality_solution_l593_59377


namespace NUMINAMATH_GPT_find_a_range_l593_59362

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x| + 2 else x + 2 / x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l593_59362


namespace NUMINAMATH_GPT_f_2018_eq_2017_l593_59349

-- Define f(1) and f(2)
def f : ℕ → ℕ 
| 1 => 1
| 2 => 1
| n => if h : n ≥ 3 then (f (n - 1) - f (n - 2) + n) else 0

-- State the theorem to prove f(2018) = 2017
theorem f_2018_eq_2017 : f 2018 = 2017 := 
by 
  sorry

end NUMINAMATH_GPT_f_2018_eq_2017_l593_59349


namespace NUMINAMATH_GPT_length_of_box_l593_59360

theorem length_of_box 
  (width height num_cubes length : ℕ)
  (h_width : width = 16)
  (h_height : height = 13)
  (h_cubes : num_cubes = 3120)
  (h_volume : length * width * height = num_cubes) :
  length = 15 :=
by
  sorry

end NUMINAMATH_GPT_length_of_box_l593_59360


namespace NUMINAMATH_GPT_fraction_value_l593_59301

variable (x y : ℝ)

theorem fraction_value (hx : x = 4) (hy : y = -3) : (x - 2 * y) / (x + y) = 10 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l593_59301


namespace NUMINAMATH_GPT_find_chosen_number_l593_59372

-- Define the conditions
def condition (x : ℝ) : Prop := (3 / 2) * x + 53.4 = -78.9

-- State the theorem
theorem find_chosen_number : ∃ x : ℝ, condition x ∧ x = -88.2 :=
sorry

end NUMINAMATH_GPT_find_chosen_number_l593_59372


namespace NUMINAMATH_GPT_bobby_total_candy_l593_59368

theorem bobby_total_candy (candy1 candy2 : ℕ) (h1 : candy1 = 26) (h2 : candy2 = 17) : candy1 + candy2 = 43 := 
by 
  sorry

end NUMINAMATH_GPT_bobby_total_candy_l593_59368


namespace NUMINAMATH_GPT_sum_of_common_ratios_l593_59336

theorem sum_of_common_ratios (k p r : ℝ) (h1 : k ≠ 0) (h2 : k * (p^2) - k * (r^2) = 5 * (k * p - k * r)) (h3 : p ≠ r) : p + r = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l593_59336


namespace NUMINAMATH_GPT_arithmetic_sequence_twenty_fourth_term_l593_59393

-- Given definitions (conditions)
def third_term (a d : ℚ) : ℚ := a + 2 * d
def tenth_term (a d : ℚ) : ℚ := a + 9 * d
def twenty_fourth_term (a d : ℚ) : ℚ := a + 23 * d

-- The main theorem to be proved
theorem arithmetic_sequence_twenty_fourth_term 
  (a d : ℚ) 
  (h1 : third_term a d = 7) 
  (h2 : tenth_term a d = 27) :
  twenty_fourth_term a d = 67 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_twenty_fourth_term_l593_59393


namespace NUMINAMATH_GPT_abs_sub_lt_five_solution_set_l593_59397

theorem abs_sub_lt_five_solution_set (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 :=
by sorry

end NUMINAMATH_GPT_abs_sub_lt_five_solution_set_l593_59397


namespace NUMINAMATH_GPT_smallest_prime_12_less_than_square_l593_59334

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end NUMINAMATH_GPT_smallest_prime_12_less_than_square_l593_59334


namespace NUMINAMATH_GPT_container_capacity_l593_59337

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 18 = 0.75 * C) : 
  C = 40 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_container_capacity_l593_59337


namespace NUMINAMATH_GPT_maria_savings_after_purchase_l593_59313

theorem maria_savings_after_purchase
  (cost_sweater : ℕ)
  (cost_scarf : ℕ)
  (cost_mittens : ℕ)
  (num_family_members : ℕ)
  (savings : ℕ)
  (total_cost_one_set : ℕ)
  (total_cost_all_sets : ℕ)
  (amount_left : ℕ)
  (h1 : cost_sweater = 35)
  (h2 : cost_scarf = 25)
  (h3 : cost_mittens = 15)
  (h4 : num_family_members = 10)
  (h5 : savings = 800)
  (h6 : total_cost_one_set = cost_sweater + cost_scarf + cost_mittens)
  (h7 : total_cost_all_sets = total_cost_one_set * num_family_members)
  (h8 : amount_left = savings - total_cost_all_sets)
  : amount_left = 50 :=
sorry

end NUMINAMATH_GPT_maria_savings_after_purchase_l593_59313


namespace NUMINAMATH_GPT_intersection_point_l593_59307

variables (g : ℤ → ℤ) (b a : ℤ)
def g_def := ∀ x : ℤ, g x = 4 * x + b
def inv_def := ∀ y : ℤ, g y = -4 → y = a
def point_intersection := ∀ y : ℤ, (g y = -4) → (y = a) → (a = -16 + b)
def solution : ℤ := -4

theorem intersection_point (b a : ℤ) (h₁ : g_def g b) (h₂ : inv_def g a) (h₃ : point_intersection g a b) :
  a = solution :=
  sorry

end NUMINAMATH_GPT_intersection_point_l593_59307


namespace NUMINAMATH_GPT_find_a_l593_59386

theorem find_a (a b c : ℕ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c) (h_eq : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end NUMINAMATH_GPT_find_a_l593_59386


namespace NUMINAMATH_GPT_ab_zero_if_conditions_l593_59340

theorem ab_zero_if_conditions 
  (a b : ℤ)
  (h : |a - b| + |a * b| = 2) : a * b = 0 :=
  sorry

end NUMINAMATH_GPT_ab_zero_if_conditions_l593_59340


namespace NUMINAMATH_GPT_habitable_land_area_l593_59346

noncomputable def area_of_habitable_land : ℝ :=
  let length : ℝ := 23
  let diagonal : ℝ := 33
  let radius_of_pond : ℝ := 3
  let width : ℝ := Real.sqrt (diagonal ^ 2 - length ^ 2)
  let area_of_rectangle : ℝ := length * width
  let area_of_pond : ℝ := Real.pi * (radius_of_pond ^ 2)
  area_of_rectangle - area_of_pond

theorem habitable_land_area :
  abs (area_of_habitable_land - 515.91) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_habitable_land_area_l593_59346


namespace NUMINAMATH_GPT_exists_nat_number_gt_1000_l593_59348

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_nat_number_gt_1000 (S : ℕ → ℕ) :
  (∀ n : ℕ, S (2^n) = sum_of_digits (2^n)) →
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n + 1)) :=
by sorry

end NUMINAMATH_GPT_exists_nat_number_gt_1000_l593_59348


namespace NUMINAMATH_GPT_percentage_spent_on_household_items_l593_59396

def Raja_income : ℝ := 37500
def clothes_percentage : ℝ := 0.20
def medicines_percentage : ℝ := 0.05
def savings_amount : ℝ := 15000

theorem percentage_spent_on_household_items : 
  (Raja_income - (clothes_percentage * Raja_income + medicines_percentage * Raja_income + savings_amount)) / Raja_income * 100 = 35 :=
  sorry

end NUMINAMATH_GPT_percentage_spent_on_household_items_l593_59396


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l593_59317

-- Definitions of conditions
variable (x : ℝ)

-- Statement of the problem in Lean 4
theorem necessary_not_sufficient_condition (h : |x - 1| ≤ 1) : 2 - x ≥ 0 := sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l593_59317


namespace NUMINAMATH_GPT_proof_m_div_x_plus_y_l593_59312

variables (a b c x y m : ℝ)

-- 1. The ratio of 'a' to 'b' is 4 to 5
axiom h1 : a / b = 4 / 5

-- 2. 'c' is half of 'a'.
axiom h2 : c = a / 2

-- 3. 'x' equals 'a' increased by 27 percent of 'a'.
axiom h3 : x = 1.27 * a

-- 4. 'y' equals 'b' decreased by 16 percent of 'b'.
axiom h4 : y = 0.84 * b

-- 5. 'm' equals 'c' increased by 14 percent of 'c'.
axiom h5 : m = 1.14 * c

theorem proof_m_div_x_plus_y : m / (x + y) = 0.2457 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_proof_m_div_x_plus_y_l593_59312


namespace NUMINAMATH_GPT_jack_mopping_rate_l593_59363

variable (bathroom_floor_area : ℕ) (kitchen_floor_area : ℕ) (time_mopped : ℕ)

theorem jack_mopping_rate
  (h_bathroom : bathroom_floor_area = 24)
  (h_kitchen : kitchen_floor_area = 80)
  (h_time : time_mopped = 13) :
  (bathroom_floor_area + kitchen_floor_area) / time_mopped = 8 :=
by
  sorry

end NUMINAMATH_GPT_jack_mopping_rate_l593_59363


namespace NUMINAMATH_GPT_waiters_hired_correct_l593_59370

noncomputable def waiters_hired (W H : ℕ) : Prop :=
  let cooks := 9
  (cooks / W = 3 / 8) ∧ (cooks / (W + H) = 1 / 4) ∧ (H = 12)

theorem waiters_hired_correct (W H : ℕ) : waiters_hired W H :=
  sorry

end NUMINAMATH_GPT_waiters_hired_correct_l593_59370


namespace NUMINAMATH_GPT_freddy_age_l593_59374

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end NUMINAMATH_GPT_freddy_age_l593_59374


namespace NUMINAMATH_GPT_plane_distance_l593_59327

variable (a b c p : ℝ)

def plane_intercept := (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (p = 1 / (Real.sqrt ((1 / a^2) + (1 / b^2) + (1 / c^2))))

theorem plane_distance
  (h : plane_intercept a b c p) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 1 / p^2 := 
sorry

end NUMINAMATH_GPT_plane_distance_l593_59327


namespace NUMINAMATH_GPT_maximum_rectangle_area_l593_59335

theorem maximum_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 44) : 
  ∃ (l_max w_max : ℕ), l_max * w_max = 121 :=
by
  sorry

end NUMINAMATH_GPT_maximum_rectangle_area_l593_59335


namespace NUMINAMATH_GPT_find_a_l593_59344

def M : Set ℝ := {x | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem find_a (a : ℝ) : N a ⊆ M ↔ a = -1 ∨ a = 0 ∨ a = 2/3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l593_59344


namespace NUMINAMATH_GPT_original_number_l593_59384

theorem original_number (x : ℕ) : 
  (∃ y : ℕ, y = x + 28 ∧ (y % 5 = 0) ∧ (y % 6 = 0) ∧ (y % 4 = 0) ∧ (y % 3 = 0)) → x = 32 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l593_59384
