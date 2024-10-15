import Mathlib

namespace NUMINAMATH_GPT_preparation_start_month_l827_82747

variable (ExamMonth : ℕ)
def start_month (ExamMonth : ℕ) : ℕ :=
  (ExamMonth - 5) % 12

theorem preparation_start_month :
  ∀ (ExamMonth : ℕ), start_month ExamMonth = (ExamMonth - 5) % 12 :=
by
  sorry

end NUMINAMATH_GPT_preparation_start_month_l827_82747


namespace NUMINAMATH_GPT_problem_statement_l827_82751

/-- For any positive integer n, given θ ∈ (0, π) and x ∈ ℂ such that 
x + 1/x = 2√2 cos θ - sin θ, it follows that x^n + 1/x^n = 2 cos (n α). -/
theorem problem_statement (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (x : ℂ) (hx : x + 1/x = 2 * (2:ℝ).sqrt * θ.cos - θ.sin)
  (n : ℕ) (hn : 0 < n) : x^n + x⁻¹^n = 2 * θ.cos * n := 
  sorry

end NUMINAMATH_GPT_problem_statement_l827_82751


namespace NUMINAMATH_GPT_probability_of_second_ball_white_is_correct_l827_82760

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_second_ball_white_is_correct_l827_82760


namespace NUMINAMATH_GPT_linear_combination_solution_l827_82797

theorem linear_combination_solution :
  ∃ a b c : ℚ, 
    a • (⟨1, -2, 3⟩ : ℚ × ℚ × ℚ) + b • (⟨4, 1, -1⟩ : ℚ × ℚ × ℚ) + c • (⟨-3, 2, 1⟩ : ℚ × ℚ × ℚ) = ⟨0, 1, 4⟩ ∧
    a = -491/342 ∧
    b = 233/342 ∧
    c = 49/38 :=
by
  sorry

end NUMINAMATH_GPT_linear_combination_solution_l827_82797


namespace NUMINAMATH_GPT_remainder_sum_of_squares_mod_13_l827_82734

-- Define the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Prove that the remainder when the sum of squares of the first 20 natural numbers
-- is divided by 13 is 10
theorem remainder_sum_of_squares_mod_13 : sum_of_squares 20 % 13 = 10 := 
by
  -- Here you can imagine the relevant steps or intermediate computations might go, if needed.
  sorry -- Placeholder for the proof.

end NUMINAMATH_GPT_remainder_sum_of_squares_mod_13_l827_82734


namespace NUMINAMATH_GPT_sqrt_sum_of_fractions_l827_82782

theorem sqrt_sum_of_fractions :
  (Real.sqrt ((25 / 36) + (16 / 9)) = (Real.sqrt 89) / 6) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_of_fractions_l827_82782


namespace NUMINAMATH_GPT_count_books_in_row_on_tuesday_l827_82761

-- Define the given conditions
def tiles_count_monday : ℕ := 38
def books_count_monday : ℕ := 75
def total_count_tuesday : ℕ := 301
def tiles_count_tuesday := tiles_count_monday * 2

-- The Lean statement we need to prove
theorem count_books_in_row_on_tuesday (hcbooks : books_count_monday = 75) 
(hc1 : total_count_tuesday = 301) 
(hc2 : tiles_count_tuesday = tiles_count_monday * 2):
  (total_count_tuesday - tiles_count_tuesday) / books_count_monday = 3 :=
by
  sorry

end NUMINAMATH_GPT_count_books_in_row_on_tuesday_l827_82761


namespace NUMINAMATH_GPT_train_cross_pole_time_l827_82719

noncomputable def train_time_to_cross_pole (length : ℕ) (speed_km_per_hr : ℕ) : ℕ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  length / speed_m_per_s

theorem train_cross_pole_time :
  train_time_to_cross_pole 100 72 = 5 :=
by
  unfold train_time_to_cross_pole
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l827_82719


namespace NUMINAMATH_GPT_complex_number_solution_l827_82723

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (h_i : z * i = 2 + i) : z = 1 - 2 * i := by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l827_82723


namespace NUMINAMATH_GPT_last_two_digits_of_sum_l827_82743

noncomputable def last_two_digits_sum_factorials : ℕ :=
  let fac : List ℕ := List.map (fun n => Nat.factorial (n * 3)) [1, 2, 3, 4, 5, 6, 7]
  fac.foldl (fun acc x => (acc + x) % 100) 0

theorem last_two_digits_of_sum : last_two_digits_sum_factorials = 6 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_sum_l827_82743


namespace NUMINAMATH_GPT_min_value_x3_l827_82772

noncomputable def min_x3 (x1 x2 x3 : ℝ) : ℝ := -21 / 11

theorem min_value_x3 (x1 x2 x3 : ℝ) 
  (h1 : x1 + (1 / 2) * x2 + (1 / 3) * x3 = 1)
  (h2 : x1^2 + (1 / 2) * x2^2 + (1 / 3) * x3^2 = 3) 
  : x3 ≥ - (21 / 11) := 
by sorry

end NUMINAMATH_GPT_min_value_x3_l827_82772


namespace NUMINAMATH_GPT_paint_price_max_boxes_paint_A_l827_82742

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end NUMINAMATH_GPT_paint_price_max_boxes_paint_A_l827_82742


namespace NUMINAMATH_GPT_det_scaled_matrix_l827_82718

variable (a b c d : ℝ)
variable (h : Matrix.det ![![a, b], ![c, d]] = 5)

theorem det_scaled_matrix : Matrix.det ![![3 * a, 3 * b], ![4 * c, 4 * d]] = 60 := by
  sorry

end NUMINAMATH_GPT_det_scaled_matrix_l827_82718


namespace NUMINAMATH_GPT_problem_statement_l827_82736

def T : Set ℤ :=
  {n^2 + (n+2)^2 + (n+4)^2 | n : ℤ }

theorem problem_statement :
  (∀ x ∈ T, ¬ (4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l827_82736


namespace NUMINAMATH_GPT_union_A_B_complement_union_l827_82790

-- Define \( U \), \( A \), and \( B \)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

-- Define complement in the universe \( U \)
def complement_U (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- Statements to prove
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
  sorry

theorem complement_union : complement_U A ∪ complement_U B = {x | x < 3 ∨ x ≥ 5} :=
  sorry

end NUMINAMATH_GPT_union_A_B_complement_union_l827_82790


namespace NUMINAMATH_GPT_range_of_a_same_solution_set_l827_82779

-- Define the inequality (x-2)(x-5) ≤ 0
def ineq1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the first inequality in the system (x-2)(x-5) ≤ 0
def ineq_system_1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the second inequality in the system x(x-a) ≥ 0
def ineq_system_2 (x a : ℝ) : Prop :=
  x * (x - a) ≥ 0

-- The final proof statement
theorem range_of_a_same_solution_set (a : ℝ) :
  (∀ x : ℝ, ineq_system_1 x ↔ ineq1 x) →
  (∀ x : ℝ, ineq_system_2 x a → ineq1 x) →
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_same_solution_set_l827_82779


namespace NUMINAMATH_GPT_perimeter_of_regular_polygon_l827_82738

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : n = 3) (h2 : side_length = 5) (h3 : exterior_angle = 120) : 
  n * side_length = 15 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_polygon_l827_82738


namespace NUMINAMATH_GPT_graph_passes_quadrants_l827_82763

theorem graph_passes_quadrants (a b : ℝ) (h_a : 1 < a) (h_b : -1 < b ∧ b < 0) : 
    ∀ x : ℝ, (0 < a^x + b ∧ x > 0) ∨ (a^x + b < 0 ∧ x < 0) ∨ (0 < x ∧ a^x + b = 0) → x ≠ 0 ∧ 0 < x :=
sorry

end NUMINAMATH_GPT_graph_passes_quadrants_l827_82763


namespace NUMINAMATH_GPT_min_max_sum_eq_one_l827_82737

theorem min_max_sum_eq_one 
  (x : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_eq_one : (x 1 + x 2 + x 3 + x 4 + x 5) = 1) :
  (min (max (x 1 + x 2) (max (x 2 + x 3) (max (x 3 + x 4) (x 4 + x 5)))) = (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_min_max_sum_eq_one_l827_82737


namespace NUMINAMATH_GPT_percentage_died_by_bombardment_l827_82741

theorem percentage_died_by_bombardment (P_initial : ℝ) (P_remaining : ℝ) (died_percentage : ℝ) (fear_percentage : ℝ) :
  P_initial = 3161 → P_remaining = 2553 → fear_percentage = 0.15 → 
  P_initial - (died_percentage/100) * P_initial - fear_percentage * (P_initial - (died_percentage/100) * P_initial) = P_remaining → 
  abs (died_percentage - 4.98) < 0.01 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_percentage_died_by_bombardment_l827_82741


namespace NUMINAMATH_GPT_bowling_average_l827_82705

theorem bowling_average (A : ℝ) (W : ℕ) (hW : W = 145) (hW7 : W + 7 ≠ 0)
  (h : ( A * W + 26 ) / ( W + 7 ) = A - 0.4) : A = 12.4 := 
by 
  sorry

end NUMINAMATH_GPT_bowling_average_l827_82705


namespace NUMINAMATH_GPT_game_result_2013_game_result_2014_l827_82710

inductive Player
| Barbara
| Jenna

def winning_player (n : ℕ) : Option Player :=
  if n % 5 = 3 then some Player.Jenna
  else if n % 5 = 4 then some Player.Barbara
  else none

theorem game_result_2013 : winning_player 2013 = some Player.Jenna := 
by sorry

theorem game_result_2014 : (winning_player 2014 = some Player.Barbara) ∨ (winning_player 2014 = some Player.Jenna) :=
by sorry

end NUMINAMATH_GPT_game_result_2013_game_result_2014_l827_82710


namespace NUMINAMATH_GPT_time_solution_l827_82702

-- Define the condition as a hypothesis
theorem time_solution (x : ℝ) (h : x / 4 + (24 - x) / 2 = x) : x = 9.6 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_time_solution_l827_82702


namespace NUMINAMATH_GPT_minimum_value_is_six_l827_82731

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z)

theorem minimum_value_is_six
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + z = 9) (h2 : y = 2 * x) :
  minimum_value_expression x y z = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_is_six_l827_82731


namespace NUMINAMATH_GPT_adam_apples_l827_82791

theorem adam_apples (x : ℕ) 
  (h1 : 15 + 75 * x = 240) : x = 3 :=
sorry

end NUMINAMATH_GPT_adam_apples_l827_82791


namespace NUMINAMATH_GPT_apples_more_than_grapes_l827_82765

theorem apples_more_than_grapes 
  (total_weight : ℕ) (weight_ratio_apples : ℕ) (weight_ratio_peaches : ℕ) (weight_ratio_grapes : ℕ) : 
  weight_ratio_apples = 12 → 
  weight_ratio_peaches = 8 → 
  weight_ratio_grapes = 7 → 
  total_weight = 54 →
  ((12 * total_weight / (12 + 8 + 7)) - (7 * total_weight / (12 + 8 + 7))) = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_apples_more_than_grapes_l827_82765


namespace NUMINAMATH_GPT_no_rational_solutions_l827_82777

theorem no_rational_solutions (a b c d : ℚ) (n : ℕ) :
  ¬ ((a + b * (Real.sqrt 2))^(2 * n) + (c + d * (Real.sqrt 2))^(2 * n) = 5 + 4 * (Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_no_rational_solutions_l827_82777


namespace NUMINAMATH_GPT_Lucy_retirement_month_l827_82787

theorem Lucy_retirement_month (start_month : ℕ) (duration : ℕ) (March : ℕ) (May : ℕ) : 
  (start_month = March) ∧ (duration = 3) → (start_month + duration - 1 = May) :=
by
  intro h
  have h_start_month := h.1
  have h_duration := h.2
  sorry

end NUMINAMATH_GPT_Lucy_retirement_month_l827_82787


namespace NUMINAMATH_GPT_find_A_minus_B_l827_82755

def A : ℤ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℤ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem find_A_minus_B : A - B = 128 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_A_minus_B_l827_82755


namespace NUMINAMATH_GPT_middle_number_is_14_5_l827_82732

theorem middle_number_is_14_5 (x y z : ℝ) (h1 : x + y = 24) (h2 : x + z = 29) (h3 : y + z = 34) : y = 14.5 :=
sorry

end NUMINAMATH_GPT_middle_number_is_14_5_l827_82732


namespace NUMINAMATH_GPT_relationship_abcd_l827_82764

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end NUMINAMATH_GPT_relationship_abcd_l827_82764


namespace NUMINAMATH_GPT_correct_mms_packs_used_l827_82707

variable (num_sundaes_monday : ℕ) (mms_per_sundae_monday : ℕ)
variable (num_sundaes_tuesday : ℕ) (mms_per_sundae_tuesday : ℕ)
variable (mms_per_pack : ℕ)

-- Conditions
def conditions : Prop := 
  num_sundaes_monday = 40 ∧ 
  mms_per_sundae_monday = 6 ∧ 
  num_sundaes_tuesday = 20 ∧
  mms_per_sundae_tuesday = 10 ∧ 
  mms_per_pack = 40

-- Question: How many m&m packs does Kekai use?
def number_of_mms_packs (num_sundaes_monday mms_per_sundae_monday 
                         num_sundaes_tuesday mms_per_sundae_tuesday 
                         mms_per_pack : ℕ) : ℕ := 
  (num_sundaes_monday * mms_per_sundae_monday + num_sundaes_tuesday * mms_per_sundae_tuesday) / mms_per_pack

-- Theorem to prove the correct number of m&m packs used
theorem correct_mms_packs_used (h : conditions num_sundaes_monday mms_per_sundae_monday 
                                              num_sundaes_tuesday mms_per_sundae_tuesday 
                                              mms_per_pack) : 
  number_of_mms_packs num_sundaes_monday mms_per_sundae_monday 
                      num_sundaes_tuesday mms_per_sundae_tuesday 
                      mms_per_pack = 11 := by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_correct_mms_packs_used_l827_82707


namespace NUMINAMATH_GPT_solve_for_x_l827_82757

theorem solve_for_x (x : ℝ) (h : 24 - 6 = 3 * x + 3) : x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l827_82757


namespace NUMINAMATH_GPT_area_of_circle_l827_82785

open Real

theorem area_of_circle :
  ∃ (A : ℝ), (∀ x y : ℝ, (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) → A = 16 * π) :=
sorry

end NUMINAMATH_GPT_area_of_circle_l827_82785


namespace NUMINAMATH_GPT_gold_copper_alloy_ratio_l827_82799

theorem gold_copper_alloy_ratio 
  (water : ℝ) 
  (G : ℝ) 
  (C : ℝ) 
  (H1 : G = 10 * water)
  (H2 : C = 6 * water)
  (H3 : 10 * G + 6 * C = 8 * (G + C)) : 
  G / C = 1 :=
by
  sorry

end NUMINAMATH_GPT_gold_copper_alloy_ratio_l827_82799


namespace NUMINAMATH_GPT_new_average_weight_l827_82730

def num_people := 6
def avg_weight1 := 154
def weight_seventh := 133

theorem new_average_weight :
  (num_people * avg_weight1 + weight_seventh) / (num_people + 1) = 151 := by
  sorry

end NUMINAMATH_GPT_new_average_weight_l827_82730


namespace NUMINAMATH_GPT_comparison_abc_l827_82721

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := 0.5 * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

theorem comparison_abc : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_comparison_abc_l827_82721


namespace NUMINAMATH_GPT_max_correct_answers_l827_82746

theorem max_correct_answers (a b c : ℕ) (n : ℕ := 60) (p_correct : ℤ := 5) (p_blank : ℤ := 0) (p_incorrect : ℤ := -2) (S : ℤ := 150) :
        a + b + c = n ∧ p_correct * a + p_blank * b + p_incorrect * c = S → a ≤ 38 :=
by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l827_82746


namespace NUMINAMATH_GPT_find_pos_int_l827_82704

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_pos_int_l827_82704


namespace NUMINAMATH_GPT_arithmetic_expression_value_l827_82748

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_value_l827_82748


namespace NUMINAMATH_GPT_school_raised_amount_correct_l827_82795

def school_fundraising : Prop :=
  let mrsJohnson := 2300
  let mrsSutton := mrsJohnson / 2
  let missRollin := mrsSutton * 8
  let topThreeTotal := missRollin * 3
  let mrEdward := missRollin * 0.75
  let msAndrea := mrEdward * 1.5
  let totalRaised := mrsJohnson + mrsSutton + missRollin + mrEdward + msAndrea
  let adminFee := totalRaised * 0.02
  let maintenanceExpense := totalRaised * 0.05
  let totalDeductions := adminFee + maintenanceExpense
  let finalAmount := totalRaised - totalDeductions
  finalAmount = 28737

theorem school_raised_amount_correct : school_fundraising := 
by 
  sorry

end NUMINAMATH_GPT_school_raised_amount_correct_l827_82795


namespace NUMINAMATH_GPT_range_of_a_l827_82711

theorem range_of_a 
  (e : ℝ) (h_e_pos : 0 < e) 
  (a : ℝ) 
  (h_equation : ∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ (1 / e ^ x₁ - a / x₁ = 0) ∧ (1 / e ^ x₂ - a / x₂ = 0)) :
  0 < a ∧ a < 1 / e :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l827_82711


namespace NUMINAMATH_GPT_solve_equation_l827_82722

theorem solve_equation (x : ℝ) (h : x ≠ 2) : -x^2 = (4 * x + 2) / (x - 2) ↔ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l827_82722


namespace NUMINAMATH_GPT_number_of_people_third_day_l827_82739

variable (X : ℕ)
variable (total : ℕ := 246)
variable (first_day : ℕ := 79)
variable (second_day_third_day_diff : ℕ := 47)

theorem number_of_people_third_day :
  (first_day + (X + second_day_third_day_diff) + X = total) → 
  X = 60 := by
  sorry

end NUMINAMATH_GPT_number_of_people_third_day_l827_82739


namespace NUMINAMATH_GPT_integer_solutions_of_system_l827_82744

theorem integer_solutions_of_system (x y z : ℤ) :
  x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10 ↔ 
  (x = 3 ∧ y = 3 ∧ z = -4) ∨ 
  (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end NUMINAMATH_GPT_integer_solutions_of_system_l827_82744


namespace NUMINAMATH_GPT_martha_knits_hat_in_2_hours_l827_82778

-- Definitions based on given conditions
variables (H : ℝ)
def knit_times (H : ℝ) : ℝ := H + 3 + 2 + 3 + 6

def total_knitting_time (H : ℝ) : ℝ := 3 * knit_times H

-- The main statement to be proven
theorem martha_knits_hat_in_2_hours (H : ℝ) (h : total_knitting_time H = 48) : H = 2 := 
by
  sorry

end NUMINAMATH_GPT_martha_knits_hat_in_2_hours_l827_82778


namespace NUMINAMATH_GPT_union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l827_82750

open Set

variable (a : ℝ)

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := univ

theorem union_A_B :
  A ∪ B = {x | 1 ≤ x ∧ x ≤ 8} := by
  sorry

theorem compl_A_inter_B :
  (U \ A) ∩ B = {x | 1 ≤ x ∧ x < 2} := by
  sorry

theorem intersection_A_C_not_empty :
  (A ∩ C a ≠ ∅) → a < 8 := by
  sorry

end NUMINAMATH_GPT_union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l827_82750


namespace NUMINAMATH_GPT_original_employee_count_l827_82758

theorem original_employee_count (employees_operations : ℝ) 
                                (employees_sales : ℝ) 
                                (employees_finance : ℝ) 
                                (employees_hr : ℝ) 
                                (employees_it : ℝ) 
                                (h1 : employees_operations / 0.82 = 192)
                                (h2 : employees_sales / 0.75 = 135)
                                (h3 : employees_finance / 0.85 = 123)
                                (h4 : employees_hr / 0.88 = 66)
                                (h5 : employees_it / 0.90 = 90) : 
                                employees_operations + employees_sales + employees_finance + employees_hr + employees_it = 734 :=
sorry

end NUMINAMATH_GPT_original_employee_count_l827_82758


namespace NUMINAMATH_GPT_solve_inequality_l827_82756

open Set Real

theorem solve_inequality (x : ℝ) : { x : ℝ | x^2 - 4 * x > 12 } = {x : ℝ | x < -2} ∪ {x : ℝ | 6 < x} := 
sorry

end NUMINAMATH_GPT_solve_inequality_l827_82756


namespace NUMINAMATH_GPT_evaluate_expression_l827_82714

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem evaluate_expression : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l827_82714


namespace NUMINAMATH_GPT_right_triangle_least_side_l827_82716

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end NUMINAMATH_GPT_right_triangle_least_side_l827_82716


namespace NUMINAMATH_GPT_area_of_region_l827_82703

noncomputable def circle_radius : ℝ := 3

noncomputable def segment_length : ℝ := 4

theorem area_of_region : ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l827_82703


namespace NUMINAMATH_GPT_min_pq_value_l827_82783

theorem min_pq_value : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 98 * p = q ^ 3 ∧ (∀ p' q' : ℕ, p' > 0 ∧ q' > 0 ∧ 98 * p' = q' ^ 3 → p' + q' ≥ p + q) ∧ p + q = 42 :=
sorry

end NUMINAMATH_GPT_min_pq_value_l827_82783


namespace NUMINAMATH_GPT_algebraic_expression_value_l827_82715

-- Define the conditions given
variables {a b : ℝ}
axiom h1 : a ≠ b
axiom h2 : a^2 - 8 * a + 5 = 0
axiom h3 : b^2 - 8 * b + 5 = 0

-- Main theorem to prove the expression equals -20
theorem algebraic_expression_value:
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l827_82715


namespace NUMINAMATH_GPT_probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l827_82769

-- Definitions for the conditions
def total_balls : ℕ := 18
def initial_red_balls : ℕ := 12
def initial_white_balls : ℕ := 6
def probability_red_ball : ℚ := initial_red_balls / total_balls
def probability_white_ball_after_removal (x : ℕ) : ℚ := initial_white_balls / (total_balls - x)

-- Statement of the proof problem
theorem probability_red_ball_is_two_thirds : probability_red_ball = 2 / 3 := 
by sorry

theorem red_balls_taken_out_is_three : ∃ x : ℕ, probability_white_ball_after_removal x = 2 / 5 ∧ x = 3 := 
by sorry

end NUMINAMATH_GPT_probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l827_82769


namespace NUMINAMATH_GPT_xiaoyu_money_left_l827_82752

def box_prices (x y z : ℝ) : Prop :=
  2 * x + 5 * y = z + 3 ∧ 5 * x + 2 * y = z - 3

noncomputable def money_left (x y z : ℝ) : ℝ :=
  z - 7 * x
  
theorem xiaoyu_money_left (x y z : ℝ) (hx : box_prices x y z) :
  money_left x y z = 7 := by
  sorry

end NUMINAMATH_GPT_xiaoyu_money_left_l827_82752


namespace NUMINAMATH_GPT_hari_contribution_correct_l827_82713

-- Translate the conditions into definitions
def praveen_investment : ℝ := 3360
def praveen_duration : ℝ := 12
def hari_duration : ℝ := 7
def profit_ratio_praveen : ℝ := 2
def profit_ratio_hari : ℝ := 3

-- The target Hari's contribution that we need to prove
def hari_contribution : ℝ := 2160

-- Problem statement: prove Hari's contribution given the conditions
theorem hari_contribution_correct :
  (praveen_investment * praveen_duration) / (hari_contribution * hari_duration) = profit_ratio_praveen / profit_ratio_hari :=
by {
  -- The statement is set up to prove equality of the ratios as given in the problem
  sorry
}

end NUMINAMATH_GPT_hari_contribution_correct_l827_82713


namespace NUMINAMATH_GPT_solve_system_of_equations_l827_82768

theorem solve_system_of_equations (a b c x y z : ℝ)
  (h1 : a^3 + a^2 * x + a * y + z = 0)
  (h2 : b^3 + b^2 * x + b * y + z = 0)
  (h3 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + ac + bc ∧ z = -abc :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_of_equations_l827_82768


namespace NUMINAMATH_GPT_concert_cost_l827_82798

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end NUMINAMATH_GPT_concert_cost_l827_82798


namespace NUMINAMATH_GPT_square_side_length_l827_82729

theorem square_side_length (s : ℝ) (h : s^2 = 1/9) : s = 1/3 :=
sorry

end NUMINAMATH_GPT_square_side_length_l827_82729


namespace NUMINAMATH_GPT_find_number_l827_82724

-- We define n, x, y as real numbers
variables (n x y : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := n * (x - y) = 4
def condition2 : Prop := 6 * x - 3 * y = 12

-- Define the theorem we need to prove: If the conditions hold, then n = 2
theorem find_number (h1 : condition1 n x y) (h2 : condition2 x y) : n = 2 := 
sorry

end NUMINAMATH_GPT_find_number_l827_82724


namespace NUMINAMATH_GPT_arithmetic_sequence_range_of_m_l827_82725

-- Conditions
variable {a : ℕ+ → ℝ} -- Sequence of positive terms
variable {S : ℕ+ → ℝ} -- Sum of the first n terms
variable (h : ∀ n, 2 * Real.sqrt (S n) = a n + 1) -- Relationship condition

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ+)
    (h1 : ∀ n, 2 * Real.sqrt (S n) = a n + 1)
    (h2 : S 1 = 1 / 4 * (a 1 + 1)^2) :
    ∃ d : ℝ, ∀ n, a (n + 1) = a n + d :=
sorry

-- Part 2: Find range of m
theorem range_of_m (T : ℕ+ → ℝ)
    (hT : ∀ n, T n = 1 / 4 * n + 1 / 8 * (1 - 1 / (2 * n + 1))) :
    ∃ m : ℝ, (6 / 7 : ℝ) < m ∧ m ≤ 10 / 9 ∧
    (∃ n₁ n₂ n₃ : ℕ+, (n₁ < n₂ ∧ n₂ < n₃) ∧ (∀ n, T n < m ↔ n₁ ≤ n ∧ n ≤ n₃)) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_range_of_m_l827_82725


namespace NUMINAMATH_GPT_max_diff_units_digit_l827_82700

theorem max_diff_units_digit (n : ℕ) (h1 : n = 850 ∨ n = 855) : ∃ d, d = 5 :=
by 
  sorry

end NUMINAMATH_GPT_max_diff_units_digit_l827_82700


namespace NUMINAMATH_GPT_commission_8000_l827_82776

variable (C k : ℝ)

def commission_5000 (C k : ℝ) : Prop := C + 5000 * k = 110
def commission_11000 (C k : ℝ) : Prop := C + 11000 * k = 230

theorem commission_8000 
  (h1 : commission_5000 C k) 
  (h2 : commission_11000 C k)
  : C + 8000 * k = 170 :=
sorry

end NUMINAMATH_GPT_commission_8000_l827_82776


namespace NUMINAMATH_GPT_special_hash_calculation_l827_82712

-- Definition of the operation #
def special_hash (a b : ℤ) : ℚ := 2 * a + (a / b) + 3

-- Statement of the proof problem
theorem special_hash_calculation : special_hash 7 3 = 19 + 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_special_hash_calculation_l827_82712


namespace NUMINAMATH_GPT_log2_sufficient_not_necessary_l827_82775

noncomputable def baseTwoLog (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (baseTwoLog a > baseTwoLog b) ↔ (a > b) :=
sorry

end NUMINAMATH_GPT_log2_sufficient_not_necessary_l827_82775


namespace NUMINAMATH_GPT_volleyball_team_ways_l827_82735

def num_ways_choose_starers : ℕ :=
  3 * (Nat.choose 12 6 + Nat.choose 12 5)

theorem volleyball_team_ways :
  num_ways_choose_starers = 5148 := by
  sorry

end NUMINAMATH_GPT_volleyball_team_ways_l827_82735


namespace NUMINAMATH_GPT_total_baseball_fans_l827_82745

theorem total_baseball_fans (Y M B : ℕ)
  (h1 : Y = 3 / 2 * M)
  (h2 : M = 88)
  (h3 : B = 5 / 4 * M) :
  Y + M + B = 330 :=
by
  sorry

end NUMINAMATH_GPT_total_baseball_fans_l827_82745


namespace NUMINAMATH_GPT_total_weight_new_group_l827_82762

variable (W : ℝ) -- Total weight of the original group of 20 people
variable (weights_old : List ℝ) 
variable (weights_new : List ℝ)

-- Given conditions
def five_weights_old : List ℝ := [40, 55, 60, 75, 80]
def average_weight_increase : ℝ := 2
def group_size : ℕ := 20
def num_replaced : ℕ := 5

-- Define theorem
theorem total_weight_new_group :
(W - five_weights_old.sum + group_size * average_weight_increase) -
(W - five_weights_old.sum) = weights_new.sum → 
weights_new.sum = 350 := 
by
  sorry

end NUMINAMATH_GPT_total_weight_new_group_l827_82762


namespace NUMINAMATH_GPT_solve_for_x_l827_82708

theorem solve_for_x (x : ℝ) (h : 3 / (x + 10) = 1 / (2 * x)) : x = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l827_82708


namespace NUMINAMATH_GPT_phones_left_is_7500_l827_82794

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end NUMINAMATH_GPT_phones_left_is_7500_l827_82794


namespace NUMINAMATH_GPT_pandas_bamboo_consumption_l827_82781

/-- Given:
  1. An adult panda can eat 138 pounds of bamboo each day.
  2. A baby panda can eat 50 pounds of bamboo a day.
Prove: the total pounds of bamboo eaten by both pandas in a week is 1316 pounds. -/
theorem pandas_bamboo_consumption :
  let adult_daily_bamboo := 138
  let baby_daily_bamboo := 50
  let days_in_week := 7
  (adult_daily_bamboo * days_in_week) + (baby_daily_bamboo * days_in_week) = 1316 := by
  sorry

end NUMINAMATH_GPT_pandas_bamboo_consumption_l827_82781


namespace NUMINAMATH_GPT_value_of_playstation_l827_82774

theorem value_of_playstation (V : ℝ) (H1 : 700 + 200 = 900) (H2 : V - 0.2 * V = 0.8 * V) (H3 : 0.8 * V = 900 - 580) : V = 400 :=
by
  sorry

end NUMINAMATH_GPT_value_of_playstation_l827_82774


namespace NUMINAMATH_GPT_rectangle_area_l827_82759

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l827_82759


namespace NUMINAMATH_GPT_inequality_proof_l827_82789

theorem inequality_proof (a b x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l827_82789


namespace NUMINAMATH_GPT_share_of_b_l827_82793

theorem share_of_b (x : ℝ) (h : 3300 / ((7/2) * x) = 2 / 7) :  
   let total_profit := 3300
   let B_share := (x / ((7/2) * x)) * total_profit
   B_share = 942.86 :=
by sorry

end NUMINAMATH_GPT_share_of_b_l827_82793


namespace NUMINAMATH_GPT_students_in_game_divisors_of_119_l827_82767

theorem students_in_game_divisors_of_119 (n : ℕ) (h1 : ∃ (k : ℕ), k * n = 119) :
  n = 7 ∨ n = 17 :=
sorry

end NUMINAMATH_GPT_students_in_game_divisors_of_119_l827_82767


namespace NUMINAMATH_GPT_parallel_lines_a_value_l827_82766

theorem parallel_lines_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
      (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_value_l827_82766


namespace NUMINAMATH_GPT_sum_base6_l827_82780

theorem sum_base6 (a b c : ℕ) 
  (ha : a = 1 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 1 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hc : c = 1 * 6^1 + 5 * 6^0) :
  a + b + c = 2 * 6^3 + 2 * 6^2 + 0 * 6^1 + 3 * 6^0 :=
by 
  sorry

end NUMINAMATH_GPT_sum_base6_l827_82780


namespace NUMINAMATH_GPT_ellipse_tangency_construction_l827_82728

theorem ellipse_tangency_construction
  (a : ℝ)
  (e1 e2 : ℝ → Prop)  -- Representing the parallel lines as propositions
  (F1 F2 : ℝ × ℝ)  -- Foci represented as points in the plane
  (d : ℝ)  -- Distance between the parallel lines
  (angle_condition : ℝ)
  (conditions : 2 * a > d ∧ angle_condition = 1 / 3) : 
  ∃ O : ℝ × ℝ,  -- Midpoint O
    ∃ (T1 T1' T2 T2' : ℝ × ℝ),  -- Points of tangency
      (∃ E1 E2 : ℝ, e1 E1 ∧ e2 E2) ∧  -- Intersection points on the lines
      (F1.1 * (T1.1 - F1.1) + F1.2 * (T1.2 - F1.2)) / 
      (F2.1 * (T2.1 - F2.1) + F2.2 * (T2.2 - F2.2)) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ellipse_tangency_construction_l827_82728


namespace NUMINAMATH_GPT_positive_even_representation_l827_82701

theorem positive_even_representation (k : ℕ) (h : k > 0) :
  ∃ (a b : ℤ), (2 * k : ℤ) = a * b ∧ a + b = 0 := 
by
  sorry

end NUMINAMATH_GPT_positive_even_representation_l827_82701


namespace NUMINAMATH_GPT_mmobile_additional_line_cost_l827_82706

noncomputable def cost_tmobile (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * 16

noncomputable def cost_mmobile (x : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * x

theorem mmobile_additional_line_cost
  (x : ℕ)
  (ht : cost_tmobile 5 = 98)
  (hm : cost_tmobile 5 - cost_mmobile x 5 = 11) :
  x = 14 :=
by
  sorry

end NUMINAMATH_GPT_mmobile_additional_line_cost_l827_82706


namespace NUMINAMATH_GPT_pictures_on_front_l827_82740

-- Conditions
variable (total_pictures : ℕ)
variable (pictures_on_back : ℕ)

-- Proof obligation
theorem pictures_on_front (h1 : total_pictures = 15) (h2 : pictures_on_back = 9) : total_pictures - pictures_on_back = 6 :=
sorry

end NUMINAMATH_GPT_pictures_on_front_l827_82740


namespace NUMINAMATH_GPT_distinct_integers_sum_l827_82709

theorem distinct_integers_sum (m n p q : ℕ) (h1 : m ≠ n) (h2 : m ≠ p) (h3 : m ≠ q) (h4 : n ≠ p)
  (h5 : n ≠ q) (h6 : p ≠ q) (h71 : m > 0) (h72 : n > 0) (h73 : p > 0) (h74 : q > 0)
  (h_eq : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 := by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_l827_82709


namespace NUMINAMATH_GPT_sample_size_stratified_sampling_l827_82720

theorem sample_size_stratified_sampling (n : ℕ) 
  (total_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (middle_aged_sample : ℕ)
  (stratified_sampling : n * middle_aged_employees = middle_aged_sample * total_employees)
  (total_employees_pos : total_employees = 750)
  (middle_aged_employees_pos : middle_aged_employees = 250) :
  n = 15 := 
by
  rw [total_employees_pos, middle_aged_employees_pos] at stratified_sampling
  sorry

end NUMINAMATH_GPT_sample_size_stratified_sampling_l827_82720


namespace NUMINAMATH_GPT_find_x_l827_82733

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) : hash x 7 = 63 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l827_82733


namespace NUMINAMATH_GPT_no_distinct_nat_numbers_eq_l827_82784

theorem no_distinct_nat_numbers_eq (x y z t : ℕ) (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t) 
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) : x ^ x + y ^ y ≠ z ^ z + t ^ t := 
by 
  sorry

end NUMINAMATH_GPT_no_distinct_nat_numbers_eq_l827_82784


namespace NUMINAMATH_GPT_arccos_cos_9_eq_2_717_l827_82788

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end NUMINAMATH_GPT_arccos_cos_9_eq_2_717_l827_82788


namespace NUMINAMATH_GPT_point_quadrant_I_or_IV_l827_82771

def is_point_on_line (x y : ℝ) : Prop := 4 * x + 3 * y = 12
def is_equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

def point_in_quadrant_I (x y : ℝ) : Prop := (x > 0 ∧ y > 0)
def point_in_quadrant_IV (x y : ℝ) : Prop := (x > 0 ∧ y < 0)

theorem point_quadrant_I_or_IV (x y : ℝ) 
  (h1 : is_point_on_line x y) 
  (h2 : is_equidistant_from_axes x y) :
  point_in_quadrant_I x y ∨ point_in_quadrant_IV x y :=
sorry

end NUMINAMATH_GPT_point_quadrant_I_or_IV_l827_82771


namespace NUMINAMATH_GPT_moles_of_CaCl2_l827_82796

theorem moles_of_CaCl2 (HCl moles_of_HCl : ℕ) (CaCO3 moles_of_CaCO3 : ℕ) 
  (reaction : (CaCO3 = 1) → (HCl = 2) → (moles_of_HCl = 6) → (moles_of_CaCO3 = 3)) :
  ∃ moles_of_CaCl2 : ℕ, moles_of_CaCl2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_CaCl2_l827_82796


namespace NUMINAMATH_GPT_log_cut_problem_l827_82786

theorem log_cut_problem (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x + 4 * y = 100) :
  2 * x + 3 * y = 70 := by
  sorry

end NUMINAMATH_GPT_log_cut_problem_l827_82786


namespace NUMINAMATH_GPT_widgets_per_shipping_box_l827_82717

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end NUMINAMATH_GPT_widgets_per_shipping_box_l827_82717


namespace NUMINAMATH_GPT_find_value_of_a_plus_b_l827_82727

variables (a b : ℝ)

theorem find_value_of_a_plus_b
  (h1 : a^3 - 3 * a^2 + 5 * a = 1)
  (h2 : b^3 - 3 * b^2 + 5 * b = 5) :
  a + b = 2 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_plus_b_l827_82727


namespace NUMINAMATH_GPT_avg_remaining_two_l827_82792

theorem avg_remaining_two (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 8) (h2 : (a + b + c) / 3 = 4) :
  (d + e) / 2 = 14 := by
  sorry

end NUMINAMATH_GPT_avg_remaining_two_l827_82792


namespace NUMINAMATH_GPT_range_of_m_for_roots_greater_than_2_l827_82753

theorem range_of_m_for_roots_greater_than_2 :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + (m-2)*x + 5 - m = 0 → x > 2) ↔ (-5 < m ∧ m ≤ -4) :=
  sorry

end NUMINAMATH_GPT_range_of_m_for_roots_greater_than_2_l827_82753


namespace NUMINAMATH_GPT_rationalize_denominator_div_l827_82726

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_rationalize_denominator_div_l827_82726


namespace NUMINAMATH_GPT_probability_of_choosing_red_base_l827_82749

theorem probability_of_choosing_red_base (A B : Prop) (C D : Prop) : 
  let red_bases := 2
  let total_bases := 4
  let probability := red_bases / total_bases
  probability = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_choosing_red_base_l827_82749


namespace NUMINAMATH_GPT_equilateral_triangle_sum_l827_82770

noncomputable def equilateral_triangle (a b c : Complex) (s : ℝ) : Prop :=
  Complex.abs (a - b) = s ∧ Complex.abs (b - c) = s ∧ Complex.abs (c - a) = s

theorem equilateral_triangle_sum (a b c : Complex):
  equilateral_triangle a b c 18 →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  intros h_triangle h_sum
  sorry

end NUMINAMATH_GPT_equilateral_triangle_sum_l827_82770


namespace NUMINAMATH_GPT_integer_modulo_solution_l827_82754

theorem integer_modulo_solution (a : ℤ) : 
  (5 ∣ a^3 + 3 * a + 1) ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  exact sorry

end NUMINAMATH_GPT_integer_modulo_solution_l827_82754


namespace NUMINAMATH_GPT_fraction_difference_l827_82773

theorem fraction_difference :
  (↑(1+4+7) / ↑(2+5+8)) - (↑(2+5+8) / ↑(1+4+7)) = - (9 / 20) :=
by
  sorry

end NUMINAMATH_GPT_fraction_difference_l827_82773
