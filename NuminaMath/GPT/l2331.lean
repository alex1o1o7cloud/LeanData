import Mathlib

namespace NUMINAMATH_GPT_vertical_asymptote_at_9_over_4_l2331_233106

def vertical_asymptote (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', x' ≠ x → abs (x' - x) < δ → abs (y x') > ε)

noncomputable def function_y (x : ℝ) : ℝ :=
  (2 * x + 3) / (4 * x - 9)

theorem vertical_asymptote_at_9_over_4 :
  vertical_asymptote function_y (9 / 4) :=
sorry

end NUMINAMATH_GPT_vertical_asymptote_at_9_over_4_l2331_233106


namespace NUMINAMATH_GPT_sum_geometric_seq_l2331_233100

theorem sum_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1)
  (h2 : 4 * a 2 = 4 * a 1 + a 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_geometric_seq_l2331_233100


namespace NUMINAMATH_GPT_solve_for_x_l2331_233121

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2331_233121


namespace NUMINAMATH_GPT_certain_number_is_65_l2331_233124

-- Define the conditions
variables (N : ℕ)
axiom condition1 : N < 81
axiom condition2 : ∀ k : ℕ, k ≤ 15 → N + k < 81
axiom last_consecutive : N + 15 = 80

-- Prove the theorem
theorem certain_number_is_65 (h1 : N < 81) (h2 : ∀ k : ℕ, k ≤ 15 → N + k < 81) (h3 : N + 15 = 80) : N = 65 :=
sorry

end NUMINAMATH_GPT_certain_number_is_65_l2331_233124


namespace NUMINAMATH_GPT_no_integer_solutions_l2331_233185

theorem no_integer_solutions : ¬∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := 
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l2331_233185


namespace NUMINAMATH_GPT_correct_equation_l2331_233102

theorem correct_equation (x y a b : ℝ) :
  ¬ (-(x - 6) = -x - 6) ∧
  ¬ (-y^2 - y^2 = 0) ∧
  ¬ (9 * a^2 * b - 9 * a * b^2 = 0) ∧
  (-9 * y^2 + 16 * y^2 = 7 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l2331_233102


namespace NUMINAMATH_GPT_eval_expression_l2331_233175

theorem eval_expression : 5 - 7 * (8 - 12 / 3^2) * 6 = -275 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l2331_233175


namespace NUMINAMATH_GPT_proof_OPQ_Constant_l2331_233105

open Complex

def OPQ_Constant :=
  ∀ (z1 z2 : ℂ) (θ : ℝ), abs z1 = 5 ∧
    (z1^2 - z1 * z2 * Real.sin θ + z2^2 = 0) →
      abs z2 = 5

theorem proof_OPQ_Constant : OPQ_Constant :=
by
  sorry

end NUMINAMATH_GPT_proof_OPQ_Constant_l2331_233105


namespace NUMINAMATH_GPT_arithmetic_mean_of_geometric_sequence_l2331_233138

theorem arithmetic_mean_of_geometric_sequence (a r : ℕ) (h_a : a = 4) (h_r : r = 3) :
    ((a) + (a * r) + (a * r^2)) / 3 = (52 / 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_geometric_sequence_l2331_233138


namespace NUMINAMATH_GPT_unclaimed_candy_fraction_l2331_233109

-- Definitions for the shares taken by each person.
def al_share (x : ℕ) : ℚ := 3 / 7 * x
def bert_share (x : ℕ) : ℚ := 2 / 7 * (x - al_share x)
def carl_share (x : ℕ) : ℚ := 1 / 7 * ((x - al_share x) - bert_share x)
def dana_share (x : ℕ) : ℚ := 1 / 7 * (((x - al_share x) - bert_share x) - carl_share x)

-- The amount of candy that goes unclaimed.
def remaining_candy (x : ℕ) : ℚ := x - (al_share x + bert_share x + carl_share x + dana_share x)

-- The theorem we want to prove.
theorem unclaimed_candy_fraction (x : ℕ) : remaining_candy x / x = 584 / 2401 :=
by
  sorry

end NUMINAMATH_GPT_unclaimed_candy_fraction_l2331_233109


namespace NUMINAMATH_GPT_car_speed_second_hour_l2331_233174

theorem car_speed_second_hour
  (S : ℕ)
  (first_hour_speed : ℕ := 98)
  (avg_speed : ℕ := 79)
  (total_time : ℕ := 2)
  (h_avg_speed : avg_speed = (first_hour_speed + S) / total_time) :
  S = 60 :=
by
  -- Proof steps omitted
  sorry

end NUMINAMATH_GPT_car_speed_second_hour_l2331_233174


namespace NUMINAMATH_GPT_total_avg_donation_per_person_l2331_233114

-- Definition of variables and conditions
variables (avgA avgB : ℝ) (numA numB : ℕ)
variables (h1 : avgB = avgA - 100)
variables (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
variables (h3 : numA = numB / 4)

-- Lean 4 statement to prove the total average donation per person is 120
theorem total_avg_donation_per_person (h1 :  avgB = avgA - 100)
    (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
    (h3 : numA = numB / 4) : 
    ( (numA * avgA + numB * avgB) / (numA + numB) ) = 120 :=
sorry

end NUMINAMATH_GPT_total_avg_donation_per_person_l2331_233114


namespace NUMINAMATH_GPT_problem1_problem2a_problem2b_problem2c_l2331_233182

theorem problem1 {x : ℝ} : 3 * x ^ 2 - 5 * x - 2 < 0 → -1 / 3 < x ∧ x < 2 :=
sorry

theorem problem2a {x a : ℝ} (ha : -1 / 2 < a ∧ a < 0) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x < 2 ∨ x > -1 / a :=
sorry

theorem problem2b {x a : ℝ} (ha : a = -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x ≠ 2 :=
sorry

theorem problem2c {x a : ℝ} (ha : a < -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x > 2 ∨ x < -1 / a :=
sorry

end NUMINAMATH_GPT_problem1_problem2a_problem2b_problem2c_l2331_233182


namespace NUMINAMATH_GPT_min_value_f_l2331_233155

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (4 / (1 - 2 * x))

theorem min_value_f : ∃ (x : ℝ), (0 < x ∧ x < 1 / 2) ∧ f x = 6 + 4 * sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_f_l2331_233155


namespace NUMINAMATH_GPT_pairs_of_boys_girls_l2331_233176

theorem pairs_of_boys_girls (a_g b_g a_b b_b : ℕ) 
  (h1 : a_b = 3 * a_g)
  (h2 : b_b = 4 * b_g) :
  ∃ c : ℕ, b_b = 7 * b_g :=
sorry

end NUMINAMATH_GPT_pairs_of_boys_girls_l2331_233176


namespace NUMINAMATH_GPT_quadratic_is_perfect_square_l2331_233184

theorem quadratic_is_perfect_square (a b c x : ℝ) (h : b^2 - 4 * a * c = 0) :
  a * x^2 + b * x + c = 0 ↔ (2 * a * x + b)^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_is_perfect_square_l2331_233184


namespace NUMINAMATH_GPT_greatest_prime_factor_294_l2331_233157

theorem greatest_prime_factor_294 : ∃ p, Nat.Prime p ∧ p ∣ 294 ∧ ∀ q, Nat.Prime q ∧ q ∣ 294 → q ≤ p := 
by
  let prime_factors := [2, 3, 7]
  have h1 : 294 = 2 * 3 * 7 * 7 := by
    -- Proof of factorization should be inserted here
    sorry

  have h2 : ∀ p, p ∣ 294 → p = 2 ∨ p = 3 ∨ p = 7 := by
    -- Proof of prime factor correctness should be inserted here
    sorry

  use 7
  -- Prove 7 is the greatest prime factor here
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_294_l2331_233157


namespace NUMINAMATH_GPT_trig_identity_l2331_233117

variable {α : Real}

theorem trig_identity (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2331_233117


namespace NUMINAMATH_GPT_train_speed_l2331_233116

theorem train_speed (v t : ℝ) (h1 : 16 * t + v * t = 444) (h2 : v * t = 16 * t + 60) : v = 21 := 
sorry

end NUMINAMATH_GPT_train_speed_l2331_233116


namespace NUMINAMATH_GPT_part1_part2_l2331_233128

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2331_233128


namespace NUMINAMATH_GPT_intersection_point_a_l2331_233152

-- Definitions for the given conditions 
def f (x : ℤ) (b : ℤ) : ℤ := 3 * x + b
def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 3 -- Considering that f is invertible for integer b

-- The problem statement
theorem intersection_point_a (a b : ℤ) (h1 : a = f (-3) b) (h2 : a = f_inv (-3)) (h3 : f (-3) b = -3):
  a = -3 := sorry

end NUMINAMATH_GPT_intersection_point_a_l2331_233152


namespace NUMINAMATH_GPT_square_side_length_l2331_233181

theorem square_side_length (P : ℝ) (s : ℝ) (h1 : P = 36) (h2 : P = 4 * s) : s = 9 := 
by sorry

end NUMINAMATH_GPT_square_side_length_l2331_233181


namespace NUMINAMATH_GPT_ten_years_less_average_age_l2331_233148

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_ten_years_less_average_age_l2331_233148


namespace NUMINAMATH_GPT_intersection_A_B_l2331_233192

-- Define the set A as natural numbers greater than 1
def A : Set ℕ := {x | x > 1}

-- Define the set B as numbers less than or equal to 3
def B : Set ℕ := {x | x ≤ 3}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x | x ∈ A ∧ x ∈ B}

-- State the theorem we want to prove
theorem intersection_A_B : A_inter_B = {2, 3} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2331_233192


namespace NUMINAMATH_GPT_find_first_month_sales_l2331_233104

noncomputable def avg_sales (sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ) : ℕ :=
(sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6

theorem find_first_month_sales :
  let sales_2 := 6927
  let sales_3 := 6855
  let sales_4 := 7230
  let sales_5 := 6562
  let sales_6 := 5091
  let avg_sales_needed := 6500
  ∃ sales_1, avg_sales sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 = avg_sales_needed := 
by
  sorry

end NUMINAMATH_GPT_find_first_month_sales_l2331_233104


namespace NUMINAMATH_GPT_root_ratio_equiv_l2331_233133

theorem root_ratio_equiv :
  (81 ^ (1 / 3)) / (81 ^ (1 / 4)) = 81 ^ (1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_root_ratio_equiv_l2331_233133


namespace NUMINAMATH_GPT_graveling_cost_is_correct_l2331_233122

noncomputable def cost_of_graveling (lawn_length : ℕ) (lawn_breadth : ℕ) 
(road_width : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_parallel_to_length := road_width * lawn_breadth
  let area_road_parallel_to_breadth := road_width * lawn_length
  let area_overlap := road_width * road_width
  let total_area := area_road_parallel_to_length + area_road_parallel_to_breadth - area_overlap
  total_area * cost_per_sq_m

theorem graveling_cost_is_correct : cost_of_graveling 90 60 10 3 = 4200 := by
  sorry

end NUMINAMATH_GPT_graveling_cost_is_correct_l2331_233122


namespace NUMINAMATH_GPT_option_c_not_equivalent_l2331_233158

theorem option_c_not_equivalent :
  ¬ (785 * 10^(-9) = 7.845 * 10^(-6)) :=
by
  sorry

end NUMINAMATH_GPT_option_c_not_equivalent_l2331_233158


namespace NUMINAMATH_GPT_MF1_dot_MF2_range_proof_l2331_233197

noncomputable def MF1_dot_MF2_range : Set ℝ :=
  Set.Icc (24 - 16 * Real.sqrt 3) (24 + 16 * Real.sqrt 3)

theorem MF1_dot_MF2_range_proof :
  ∀ (M : ℝ × ℝ), (Prod.snd M + 4) ^ 2 + (Prod.fst M) ^ 2 = 12 →
    (Prod.fst M) ^ 2 + (Prod.snd M) ^ 2 - 4 ∈ MF1_dot_MF2_range :=
by
  sorry

end NUMINAMATH_GPT_MF1_dot_MF2_range_proof_l2331_233197


namespace NUMINAMATH_GPT_min_value_xy_l2331_233132

theorem min_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : x * y ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_value_xy_l2331_233132


namespace NUMINAMATH_GPT_actual_distance_traveled_l2331_233137

theorem actual_distance_traveled :
  ∀ (t : ℝ) (d1 d2 : ℝ),
  d1 = 15 * t →
  d2 = 30 * t →
  d2 = d1 + 45 →
  d1 = 45 := by
  intro t d1 d2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l2331_233137


namespace NUMINAMATH_GPT_correct_answer_l2331_233142

def A : Set ℝ := { x | x^2 + 2 * x - 3 > 0 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem correct_answer : A ∩ B = { 2 } :=
  sorry

end NUMINAMATH_GPT_correct_answer_l2331_233142


namespace NUMINAMATH_GPT_misha_total_shots_l2331_233166

theorem misha_total_shots (x y : ℕ) 
  (h1 : 18 * x + 5 * y = 99) 
  (h2 : 2 * x + y = 15) 
  (h3 : (15 / 0.9375 : ℝ) = 16) : 
  (¬(x = 0) ∧ ¬(y = 24)) ->
  16 = 16 :=
by
  sorry

end NUMINAMATH_GPT_misha_total_shots_l2331_233166


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2331_233162

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 0): (x = 1 → x > 0) ∧ ¬(x > 0 → x = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2331_233162


namespace NUMINAMATH_GPT_contrapositive_proposition_l2331_233130

theorem contrapositive_proposition (x a b : ℝ) : (x < 2 * a * b) → (x < a^2 + b^2) :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l2331_233130


namespace NUMINAMATH_GPT_time_per_potato_l2331_233198

-- Definitions from the conditions
def total_potatoes : ℕ := 12
def cooked_potatoes : ℕ := 6
def remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
def total_time : ℕ := 36
def remaining_time_per_potato : ℕ := total_time / remaining_potatoes

-- Theorem to be proved
theorem time_per_potato : remaining_time_per_potato = 6 := by
  sorry

end NUMINAMATH_GPT_time_per_potato_l2331_233198


namespace NUMINAMATH_GPT_moles_of_water_used_l2331_233156

-- Define the balanced chemical equation's molar ratios
def balanced_reaction (Li3N_moles : ℕ) (H2O_moles : ℕ) (LiOH_moles : ℕ) (NH3_moles : ℕ) : Prop :=
  Li3N_moles = 1 ∧ H2O_moles = 3 ∧ LiOH_moles = 3 ∧ NH3_moles = 1

-- Given 1 mole of lithium nitride and 3 moles of lithium hydroxide produced, 
-- prove that 3 moles of water were used.
theorem moles_of_water_used (Li3N_moles : ℕ) (LiOH_moles : ℕ) (H2O_moles : ℕ) :
  Li3N_moles = 1 → LiOH_moles = 3 → H2O_moles = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_moles_of_water_used_l2331_233156


namespace NUMINAMATH_GPT_integer_solution_count_l2331_233112

theorem integer_solution_count (x : ℤ) : (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 330 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_count_l2331_233112


namespace NUMINAMATH_GPT_mrs_hilt_remaining_cents_l2331_233119

-- Define the initial amount of money Mrs. Hilt had
def initial_cents : ℕ := 43

-- Define the cost of the pencil
def pencil_cost : ℕ := 20

-- Define the cost of the candy
def candy_cost : ℕ := 5

-- Define the remaining money Mrs. Hilt has after the purchases
def remaining_cents : ℕ := initial_cents - (pencil_cost + candy_cost)

-- Theorem statement to prove that the remaining amount is 18 cents
theorem mrs_hilt_remaining_cents : remaining_cents = 18 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mrs_hilt_remaining_cents_l2331_233119


namespace NUMINAMATH_GPT_trig_inequality_sin_cos_l2331_233191

theorem trig_inequality_sin_cos :
  Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_inequality_sin_cos_l2331_233191


namespace NUMINAMATH_GPT_picked_clovers_when_one_four_found_l2331_233113

-- Definition of conditions
def total_leaves : ℕ := 100
def leaves_three_leaved_clover : ℕ := 3
def leaves_four_leaved_clover : ℕ := 4
def one_four_leaved_clover : ℕ := 1

-- Proof Statement
theorem picked_clovers_when_one_four_found (three_leaved_count : ℕ) :
  (total_leaves - leaves_four_leaved_clover) / leaves_three_leaved_clover = three_leaved_count → 
  three_leaved_count = 32 :=
by
  sorry

end NUMINAMATH_GPT_picked_clovers_when_one_four_found_l2331_233113


namespace NUMINAMATH_GPT_slope_tangent_line_l2331_233190

variable {f : ℝ → ℝ}

-- Assumption: f is differentiable
def differentiable_at (f : ℝ → ℝ) (x : ℝ) := ∃ f', ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x + h) - f x) / h - f'| < ε

-- Hypothesis: limit condition
axiom limit_condition : (∀ x, differentiable_at f (1 - x)) → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε)

-- Theorem: the slope of the tangent line to the curve y = f(x) at (1, f(1)) is -2
theorem slope_tangent_line : differentiable_at f 1 → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε) → deriv f 1 = -2 :=
by
    intro h_diff h_lim
    sorry

end NUMINAMATH_GPT_slope_tangent_line_l2331_233190


namespace NUMINAMATH_GPT_ratio_of_areas_eq_nine_sixteenth_l2331_233177

-- Definitions based on conditions
def side_length_C : ℝ := 45
def side_length_D : ℝ := 60
def area (s : ℝ) : ℝ := s * s

-- Theorem stating the desired proof problem
theorem ratio_of_areas_eq_nine_sixteenth :
  (area side_length_C) / (area side_length_D) = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_eq_nine_sixteenth_l2331_233177


namespace NUMINAMATH_GPT_football_starting_lineup_count_l2331_233143

variable (n_team_members n_offensive_linemen : ℕ)
variable (H_team_members : 12 = n_team_members)
variable (H_offensive_linemen : 5 = n_offensive_linemen)

theorem football_starting_lineup_count :
  n_team_members = 12 → n_offensive_linemen = 5 →
  (n_offensive_linemen * (n_team_members - 1) * (n_team_members - 2) * ((n_team_members - 3) * (n_team_members - 4) / 2)) = 19800 := 
by
  intros
  sorry

end NUMINAMATH_GPT_football_starting_lineup_count_l2331_233143


namespace NUMINAMATH_GPT_fill_bathtub_with_drain_open_l2331_233172

theorem fill_bathtub_with_drain_open :
  let fill_rate := 1 / 10
  let drain_rate := 1 / 12
  let net_fill_rate := fill_rate - drain_rate
  fill_rate = 1 / 10 ∧ drain_rate = 1 / 12 → 1 / net_fill_rate = 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fill_bathtub_with_drain_open_l2331_233172


namespace NUMINAMATH_GPT_integer_solutions_to_system_l2331_233108

theorem integer_solutions_to_system (x y z : ℤ) (h1 : x + y + z = 2) (h2 : x^3 + y^3 + z^3 = -10) :
  (x = 3 ∧ y = 3 ∧ z = -4) ∨
  (x = 3 ∧ y = -4 ∧ z = 3) ∨
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end NUMINAMATH_GPT_integer_solutions_to_system_l2331_233108


namespace NUMINAMATH_GPT_simplify_and_evaluate_div_fraction_l2331_233193

theorem simplify_and_evaluate_div_fraction (a : ℤ) (h : a = -3) : 
  (a - 2) / (1 + 2 * a + a^2) / (a - 3 * a / (a + 1)) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_div_fraction_l2331_233193


namespace NUMINAMATH_GPT_remove_terms_l2331_233188

-- Define the fractions
def f1 := 1 / 3
def f2 := 1 / 6
def f3 := 1 / 9
def f4 := 1 / 12
def f5 := 1 / 15
def f6 := 1 / 18

-- Define the total sum
def total_sum := f1 + f2 + f3 + f4 + f5 + f6

-- Define the target sum after removal
def target_sum := 2 / 3

-- Define the condition to be proven
theorem remove_terms {x y : Real} (h1 : (x = f4) ∧ (y = f5)) : 
  total_sum - (x + y) = target_sum := by
  sorry

end NUMINAMATH_GPT_remove_terms_l2331_233188


namespace NUMINAMATH_GPT_day_care_center_toddlers_l2331_233149

theorem day_care_center_toddlers (I T : ℕ) (h_ratio1 : 7 * I = 3 * T) (h_ratio2 : 7 * (I + 12) = 5 * T) :
  T = 42 :=
by
  sorry

end NUMINAMATH_GPT_day_care_center_toddlers_l2331_233149


namespace NUMINAMATH_GPT_position_of_2019_in_splits_l2331_233165

def sum_of_consecutive_odds (n : ℕ) : ℕ :=
  n^2 - (n - 1)

theorem position_of_2019_in_splits : ∃ n : ℕ, sum_of_consecutive_odds n = 2019 ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_position_of_2019_in_splits_l2331_233165


namespace NUMINAMATH_GPT_exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l2331_233154

theorem exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012 :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧ 
    a ∣ (a * b * c + 2012) ∧ b ∣ (a * b * c + 2012) ∧ c ∣ (a * b * c + 2012) :=
by
  sorry

end NUMINAMATH_GPT_exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l2331_233154


namespace NUMINAMATH_GPT_parallelepiped_analogy_l2331_233196

-- Define plane figures and the concept of analogy for a parallelepiped 
-- (specifically here as a parallelogram) in space
inductive PlaneFigure where
  | triangle
  | parallelogram
  | trapezoid
  | rectangle

open PlaneFigure

/-- 
  Given the properties and definitions of a parallelepiped and plane figures,
  we want to show that the appropriate analogy for a parallelepiped in space
  is a parallelogram.
-/
theorem parallelepiped_analogy : 
  (analogy : PlaneFigure) = parallelogram :=
sorry

end NUMINAMATH_GPT_parallelepiped_analogy_l2331_233196


namespace NUMINAMATH_GPT_find_value_of_a_b_ab_l2331_233186

variable (a b : ℝ)

theorem find_value_of_a_b_ab
  (h1 : 2 * a + 2 * b + a * b = 1)
  (h2 : a + b + 3 * a * b = -2) :
  a + b + a * b = 0 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_b_ab_l2331_233186


namespace NUMINAMATH_GPT_problem1_problem2a_problem2b_l2331_233111

-- Problem 1: Deriving y in terms of x
theorem problem1 (x y : ℕ) (h1 : 30 * x + 10 * y = 2000) : y = 200 - 3 * x :=
by sorry

-- Problem 2(a): Minimum ingredient B for at least 220 yuan profit with a=3
theorem problem2a (x y a w : ℕ) (h1 : a = 3) 
  (h2 : 3 * x + 2 * y ≥ 220) (h3 : y = 200 - 3 * x) 
  (h4 : w = 15 * x + 20 * y) : w = 1300 :=
by sorry

-- Problem 2(b): Profit per portion of dessert A for 450 yuan profit with 3100 grams of B
theorem problem2b (x : ℕ) (a : ℕ) (B : ℕ) 
  (h1 : B = 3100) (h2 : 15 * x + 20 * (200 - 3 * x) ≤ B) 
  (h3 : a * x + 2 * (200 - 3 * x) = 450) 
  (h4 : x ≥ 20) : a = 8 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2a_problem2b_l2331_233111


namespace NUMINAMATH_GPT_miguel_paint_area_l2331_233168

def wall_height := 10
def wall_length := 15
def window_side := 3

theorem miguel_paint_area :
  (wall_height * wall_length) - (window_side * window_side) = 141 := 
by
  sorry

end NUMINAMATH_GPT_miguel_paint_area_l2331_233168


namespace NUMINAMATH_GPT_max_ab_is_5_l2331_233161

noncomputable def max_ab : ℝ :=
  sorry

theorem max_ab_is_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h : a / 4 + b / 5 = 1) : max_ab = 5 :=
  sorry

end NUMINAMATH_GPT_max_ab_is_5_l2331_233161


namespace NUMINAMATH_GPT_ordering_l2331_233153

noncomputable def a : ℝ := 1 / (Real.exp 0.6)
noncomputable def b : ℝ := 0.4
noncomputable def c : ℝ := Real.log 1.4 / 1.4

theorem ordering : a > b ∧ b > c :=
by
  have ha : a = 1 / (Real.exp 0.6) := rfl
  have hb : b = 0.4 := rfl
  have hc : c = Real.log 1.4 / 1.4 := rfl
  sorry

end NUMINAMATH_GPT_ordering_l2331_233153


namespace NUMINAMATH_GPT_a_minus_b_l2331_233160

theorem a_minus_b (a b : ℚ) :
  (∀ x y, (x = 3 → y = 7) ∨ (x = 10 → y = 19) → y = a * x + b) →
  a - b = -(1/7) :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_l2331_233160


namespace NUMINAMATH_GPT_smallest_four_digit_equiv_8_mod_9_l2331_233125

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_equiv_8_mod_9_l2331_233125


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l2331_233179

theorem abs_inequality_solution_set (x : ℝ) :
  |x| + |x - 1| < 2 ↔ - (1 / 2) < x ∧ x < (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l2331_233179


namespace NUMINAMATH_GPT_probability_at_least_one_succeeds_l2331_233170

variable (p1 p2 : ℝ)

theorem probability_at_least_one_succeeds : 
  0 ≤ p1 ∧ p1 ≤ 1 → 0 ≤ p2 ∧ p2 ≤ 1 → (1 - (1 - p1) * (1 - p2)) = 1 - (1 - p1) * (1 - p2) :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_probability_at_least_one_succeeds_l2331_233170


namespace NUMINAMATH_GPT_smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l2331_233126

-- Problem (a): Smallest n such that n! is divisible by 2016
theorem smallest_n_divisible_by_2016 : ∃ (n : ℕ), n = 8 ∧ 2016 ∣ n.factorial :=
by
  sorry

-- Problem (b): Smallest n such that n! is divisible by 2016^10
theorem smallest_n_divisible_by_2016_pow_10 : ∃ (n : ℕ), n = 63 ∧ 2016^10 ∣ n.factorial :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l2331_233126


namespace NUMINAMATH_GPT_car_with_highest_avg_speed_l2331_233120

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end NUMINAMATH_GPT_car_with_highest_avg_speed_l2331_233120


namespace NUMINAMATH_GPT_money_made_arkansas_game_is_8722_l2331_233145

def price_per_tshirt : ℕ := 98
def tshirts_sold_arkansas_game : ℕ := 89
def total_money_made_arkansas_game (price_per_tshirt tshirts_sold_arkansas_game : ℕ) : ℕ :=
  price_per_tshirt * tshirts_sold_arkansas_game

theorem money_made_arkansas_game_is_8722 :
  total_money_made_arkansas_game price_per_tshirt tshirts_sold_arkansas_game = 8722 :=
by
  sorry

end NUMINAMATH_GPT_money_made_arkansas_game_is_8722_l2331_233145


namespace NUMINAMATH_GPT_find_difference_l2331_233107

variable (d : ℕ) (A B : ℕ)
open Nat

theorem find_difference (hd : d > 7)
  (hAB : d * A + B + d * A + A = d * d + 7 * d + 4)  (hA_gt_B : A > B):
  A - B = 3 :=
sorry

end NUMINAMATH_GPT_find_difference_l2331_233107


namespace NUMINAMATH_GPT_no_valid_rectangles_l2331_233187

theorem no_valid_rectangles 
  (a b x y : ℝ) (h_ab_lt : a < b) (h_xa_lt : x < a) (h_ya_lt : y < a) 
  (h_perimeter : 2 * (x + y) = (2 * (a + b)) / 3) 
  (h_area : x * y = (a * b) / 3) : false := 
sorry

end NUMINAMATH_GPT_no_valid_rectangles_l2331_233187


namespace NUMINAMATH_GPT_cost_price_l2331_233164

theorem cost_price (SP MP CP : ℝ) (discount_rate : ℝ) 
  (h1 : MP = CP * 1.15)
  (h2 : SP = MP * (1 - discount_rate))
  (h3 : SP = 459)
  (h4 : discount_rate = 0.2608695652173913) : CP = 540 :=
by
  -- We use the hints given as conditions to derive the statement
  sorry

end NUMINAMATH_GPT_cost_price_l2331_233164


namespace NUMINAMATH_GPT_christmas_bonus_remainder_l2331_233167

theorem christmas_bonus_remainder (B P R : ℕ) (hP : P = 8 * B + 5) (hR : (4 * P) % 8 = R) : R = 4 :=
by
  sorry

end NUMINAMATH_GPT_christmas_bonus_remainder_l2331_233167


namespace NUMINAMATH_GPT_Phil_quarters_l2331_233141

theorem Phil_quarters (initial_amount : ℝ)
  (pizza : ℝ) (soda : ℝ) (jeans : ℝ) (book : ℝ) (gum : ℝ) (ticket : ℝ)
  (quarter_value : ℝ) (spent := pizza + soda + jeans + book + gum + ticket)
  (remaining := initial_amount - spent)
  (quarters := remaining / quarter_value) :
  initial_amount = 40 ∧ pizza = 2.75 ∧ soda = 1.50 ∧ jeans = 11.50 ∧
  book = 6.25 ∧ gum = 1.75 ∧ ticket = 8.50 ∧ quarter_value = 0.25 →
  quarters = 31 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Phil_quarters_l2331_233141


namespace NUMINAMATH_GPT_calculate_expression_l2331_233173

theorem calculate_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2331_233173


namespace NUMINAMATH_GPT_find_angle_A_find_AB_l2331_233101

theorem find_angle_A (A B C : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C)) (h2 : A + B + C = Real.pi) :
  A = Real.pi / 3 := by
  sorry

theorem find_AB (A B C : ℝ) (AB BC AC : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C))
  (h2 : BC = 2) (h3 : 1 / 2 * AB * AC * Real.sin (Real.pi / 3) = Real.sqrt 3)
  (h4 : A = Real.pi / 3) :
  AB = 2 := by
  sorry

end NUMINAMATH_GPT_find_angle_A_find_AB_l2331_233101


namespace NUMINAMATH_GPT_snack_eaters_left_after_second_newcomers_l2331_233171

theorem snack_eaters_left_after_second_newcomers
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (half_left_1 : ℕ)
  (new_outsiders_2 : ℕ)
  (final_snackers : ℕ)
  (H1 : initial_snackers = 100)
  (H2 : new_outsiders_1 = 20)
  (H3 : half_left_1 = (initial_snackers + new_outsiders_1) / 2)
  (H4 : new_outsiders_2 = 10)
  (H5 : final_snackers = 20)
  : (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - final_snackers * 2)) = 30 :=
by 
  sorry

end NUMINAMATH_GPT_snack_eaters_left_after_second_newcomers_l2331_233171


namespace NUMINAMATH_GPT_parabola_directrix_l2331_233194

theorem parabola_directrix (x y : ℝ) (h : y = 4 * x^2) : y = -1 / 16 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2331_233194


namespace NUMINAMATH_GPT_lending_period_C_l2331_233103

theorem lending_period_C (P_B P_C : ℝ) (R : ℝ) (T_B I_total : ℝ) (T_C_months : ℝ) :
  P_B = 5000 ∧ P_C = 3000 ∧ R = 0.10 ∧ T_B = 2 ∧ I_total = 2200 ∧ 
  T_C_months = (2 / 3) * 12 → T_C_months = 8 := by
  intros h
  sorry

end NUMINAMATH_GPT_lending_period_C_l2331_233103


namespace NUMINAMATH_GPT_angle_GDA_is_135_l2331_233139

-- Definitions for the geometric entities and conditions mentioned
structure Triangle :=
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)

structure Square :=
  (angle : ℝ := 90)

def BCD : Triangle :=
  { angle_A := 45, angle_B := 45, angle_C := 90 }

def ABCD : Square :=
  {}

def DEFG : Square :=
  {}

-- The proof problem stated in Lean 4
theorem angle_GDA_is_135 :
  ∃ θ : ℝ, θ = 135 ∧ 
  (∀ (BCD : Triangle), BCD.angle_C = 90 ∧ BCD.angle_A = 45 ∧ BCD.angle_B = 45) ∧ 
  (∀ (Square : Square), Square.angle = 90) → 
  θ = 135 :=
by
  sorry

end NUMINAMATH_GPT_angle_GDA_is_135_l2331_233139


namespace NUMINAMATH_GPT_correct_sentence_is_D_l2331_233144

-- Define the sentences as strings
def sentence_A : String :=
  "Between any two adjacent integers on the number line, an infinite number of fractions can be inserted to fill the gaps on the number line; mathematicians once thought that with this approach, the entire number line was finally filled."

def sentence_B : String :=
  "With zero as the center, all integers are arranged from right to left at equal distances, and then connected with a horizontal line; this is what we call the 'number line'."

def sentence_C : String :=
  "The vast collection of books in the Beijing Library contains an enormous amount of information, but it is still finite, whereas the number pi contains infinite information, which is awe-inspiring."

def sentence_D : String :=
  "Pi is fundamentally the exact ratio of a circle's circumference to its diameter, but the infinite sequence it produces has the greatest uncertainty; we cannot help but be amazed and shaken by the marvel and mystery of nature."

-- Define the problem statement
theorem correct_sentence_is_D :
  sentence_D ≠ "" := by
  sorry

end NUMINAMATH_GPT_correct_sentence_is_D_l2331_233144


namespace NUMINAMATH_GPT_ribbon_tape_remaining_l2331_233147

theorem ribbon_tape_remaining 
  (initial_length used_for_ribbon used_for_gift : ℝ)
  (h_initial: initial_length = 1.6)
  (h_ribbon: used_for_ribbon = 0.8)
  (h_gift: used_for_gift = 0.3) : 
  initial_length - used_for_ribbon - used_for_gift = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_ribbon_tape_remaining_l2331_233147


namespace NUMINAMATH_GPT_tangent_line_at_point_l2331_233163

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (Real.exp (-(x - 1)) - x) else (Real.exp (x - 1) + x)

theorem tangent_line_at_point (f_even : ∀ x : ℝ, f x = f (-x)) :
    ∀ (x y : ℝ), x = 1 → y = 2 → (∃ m b : ℝ, y = m * x + b ∧ m = 2 ∧ b = 0) := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2331_233163


namespace NUMINAMATH_GPT_solve_inequality_l2331_233123

theorem solve_inequality (x : ℝ) :
  (x^2 - 4 * x - 12) / (x - 3) < 0 ↔ (-2 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2331_233123


namespace NUMINAMATH_GPT_island_challenge_probability_l2331_233189
open Nat

theorem island_challenge_probability :
  let total_ways := choose 20 3
  let ways_one_tribe := choose 10 3
  let combined_ways := 2 * ways_one_tribe
  let probability := combined_ways / total_ways
  probability = (20 : ℚ) / 95 :=
by
  sorry

end NUMINAMATH_GPT_island_challenge_probability_l2331_233189


namespace NUMINAMATH_GPT_min_value_expr_l2331_233135

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l2331_233135


namespace NUMINAMATH_GPT_problem_inequality_l2331_233169

theorem problem_inequality 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) 
  (h8 : a + b + c + d ≥ 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := 
sorry

end NUMINAMATH_GPT_problem_inequality_l2331_233169


namespace NUMINAMATH_GPT_find_sum_of_cubes_l2331_233199

-- Define the distinct real numbers p, q, and r
variables {p q r : ℝ}

-- Conditions
-- Distinctness condition
axiom h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p

-- Given condition
axiom h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r

-- Proof goal
theorem find_sum_of_cubes : p^3 + q^3 + r^3 = -21 :=
sorry

end NUMINAMATH_GPT_find_sum_of_cubes_l2331_233199


namespace NUMINAMATH_GPT_maximize_value_l2331_233178

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  3 * x - 2 * y

theorem maximize_value (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : maximum_value x y ≤ 5 :=
sorry

end NUMINAMATH_GPT_maximize_value_l2331_233178


namespace NUMINAMATH_GPT_refrigerator_cost_is_15000_l2331_233151

theorem refrigerator_cost_is_15000 (R : ℝ) 
  (phone_cost : ℝ := 8000)
  (phone_profit : ℝ := 0.10) 
  (fridge_loss : ℝ := 0.03) 
  (overall_profit : ℝ := 350) :
  (0.97 * R + phone_cost * (1 + phone_profit) = (R + phone_cost) + overall_profit) →
  (R = 15000) :=
by
  sorry

end NUMINAMATH_GPT_refrigerator_cost_is_15000_l2331_233151


namespace NUMINAMATH_GPT_emily_can_see_emerson_l2331_233150

theorem emily_can_see_emerson : 
  ∀ (emily_speed emerson_speed : ℝ) 
    (initial_distance final_distance : ℝ), 
  emily_speed = 15 → 
  emerson_speed = 9 → 
  initial_distance = 1 → 
  final_distance = 1 →
  (initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)) * 60 = 20 :=
by
  intros emily_speed emerson_speed initial_distance final_distance
  sorry

end NUMINAMATH_GPT_emily_can_see_emerson_l2331_233150


namespace NUMINAMATH_GPT_surprise_shop_daily_revenue_l2331_233131

def closed_days_per_year : ℕ := 3
def years_active : ℕ := 6
def total_revenue_lost : ℚ := 90000

def total_closed_days : ℕ :=
  closed_days_per_year * years_active

def daily_revenue : ℚ :=
  total_revenue_lost / total_closed_days

theorem surprise_shop_daily_revenue :
  daily_revenue = 5000 := by
  sorry

end NUMINAMATH_GPT_surprise_shop_daily_revenue_l2331_233131


namespace NUMINAMATH_GPT_janet_needs_9_dog_collars_l2331_233136

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end NUMINAMATH_GPT_janet_needs_9_dog_collars_l2331_233136


namespace NUMINAMATH_GPT_probability_triangle_nonagon_l2331_233195

-- Define the total number of ways to choose 3 vertices from 9 vertices
def total_ways_to_choose_triangle : ℕ := Nat.choose 9 3

-- Define the number of favorable outcomes
def favorable_outcomes_one_side : ℕ := 9 * 5
def favorable_outcomes_two_sides : ℕ := 9

def total_favorable_outcomes : ℕ := favorable_outcomes_one_side + favorable_outcomes_two_sides

-- Define the probability as a rational number
def probability_at_least_one_side_nonagon (total: ℕ) (favorable: ℕ) : ℚ :=
  favorable / total
  
-- Theorem stating the probability
theorem probability_triangle_nonagon :
  probability_at_least_one_side_nonagon total_ways_to_choose_triangle total_favorable_outcomes = 9 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_triangle_nonagon_l2331_233195


namespace NUMINAMATH_GPT_value_of_expression_l2331_233140

theorem value_of_expression :
  (10^2 - 10) / 9 = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2331_233140


namespace NUMINAMATH_GPT_compute_d1e1_d2e2_d3e3_l2331_233127

-- Given polynomials and conditions
variables {R : Type*} [CommRing R]

noncomputable def P (x : R) : R :=
  x^7 - x^6 + x^4 - x^3 + x^2 - x + 1

noncomputable def Q (x : R) (d1 d2 d3 e1 e2 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3)

-- Given conditions
theorem compute_d1e1_d2e2_d3e3 
  (d1 d2 d3 e1 e2 e3 : R)
  (h : ∀ x : R, P x = Q x d1 d2 d3 e1 e2 e3) : 
  d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
by
  sorry

end NUMINAMATH_GPT_compute_d1e1_d2e2_d3e3_l2331_233127


namespace NUMINAMATH_GPT_find_larger_number_l2331_233180

theorem find_larger_number (a b : ℕ) (h_diff : a - b = 3) (h_sum_squares : a^2 + b^2 = 117) (h_pos : 0 < a ∧ 0 < b) : a = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l2331_233180


namespace NUMINAMATH_GPT_count_rectangles_with_perimeter_twenty_two_l2331_233115

theorem count_rectangles_with_perimeter_twenty_two : 
  (∃! (n : ℕ), n = 11) :=
by
  sorry

end NUMINAMATH_GPT_count_rectangles_with_perimeter_twenty_two_l2331_233115


namespace NUMINAMATH_GPT_maple_logs_correct_l2331_233146

/-- Each pine tree makes 80 logs. -/
def pine_logs := 80

/-- Each walnut tree makes 100 logs. -/
def walnut_logs := 100

/-- Jerry cuts up 8 pine trees. -/
def pine_trees := 8

/-- Jerry cuts up 3 maple trees. -/
def maple_trees := 3

/-- Jerry cuts up 4 walnut trees. -/
def walnut_trees := 4

/-- The total number of logs is 1220. -/
def total_logs := 1220

/-- The number of logs each maple tree makes. -/
def maple_logs := 60

theorem maple_logs_correct :
  (pine_trees * pine_logs) + (maple_trees * maple_logs) + (walnut_trees * walnut_logs) = total_logs :=
by
  -- (8 * 80) + (3 * 60) + (4 * 100) = 1220
  sorry

end NUMINAMATH_GPT_maple_logs_correct_l2331_233146


namespace NUMINAMATH_GPT_remainder_division_l2331_233134

-- Define the polynomial f(x) = x^51 + 51
def f (x : ℤ) : ℤ := x^51 + 51

-- State the theorem to be proven
theorem remainder_division : f (-1) = 50 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_remainder_division_l2331_233134


namespace NUMINAMATH_GPT_correct_operation_l2331_233110

theorem correct_operation (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2331_233110


namespace NUMINAMATH_GPT_sledding_small_hills_l2331_233183

theorem sledding_small_hills (total_sleds tall_hills_sleds sleds_per_tall_hill sleds_per_small_hill small_hills : ℕ) 
  (h1 : total_sleds = 14)
  (h2 : tall_hills_sleds = 2)
  (h3 : sleds_per_tall_hill = 4)
  (h4 : sleds_per_small_hill = sleds_per_tall_hill / 2)
  (h5 : total_sleds = tall_hills_sleds * sleds_per_tall_hill + small_hills * sleds_per_small_hill)
  : small_hills = 3 := 
sorry

end NUMINAMATH_GPT_sledding_small_hills_l2331_233183


namespace NUMINAMATH_GPT_dots_not_visible_on_3_dice_l2331_233159

theorem dots_not_visible_on_3_dice :
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  hidden_dots = 35 := 
by 
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  show total_dots - visible_dots = 35
  sorry

end NUMINAMATH_GPT_dots_not_visible_on_3_dice_l2331_233159


namespace NUMINAMATH_GPT_evaluate_expression_l2331_233118

noncomputable def lg (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression :
  lg 5 * lg 50 - lg 2 * lg 20 - lg 625 = -2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2331_233118


namespace NUMINAMATH_GPT_common_ratio_of_geometric_progression_l2331_233129

theorem common_ratio_of_geometric_progression (a1 q : ℝ) (S3 : ℝ) (a2 : ℝ)
  (h1 : S3 = a1 * (1 + q + q^2))
  (h2 : a2 = a1 * q)
  (h3 : a2 + S3 = 0) :
  q = -1 := 
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_progression_l2331_233129
