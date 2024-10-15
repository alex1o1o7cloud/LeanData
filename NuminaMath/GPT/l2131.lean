import Mathlib

namespace NUMINAMATH_GPT_max_area_of_right_triangle_with_hypotenuse_4_l2131_213187

theorem max_area_of_right_triangle_with_hypotenuse_4 : 
  (∀ (a b : ℝ), a^2 + b^2 = 16 → (∃ S, S = 1/2 * a * b ∧ S ≤ 4)) ∧ 
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ a = b ∧ 1/2 * a * b = 4) :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_right_triangle_with_hypotenuse_4_l2131_213187


namespace NUMINAMATH_GPT_remainder_is_90_l2131_213103

theorem remainder_is_90:
  let larger_number := 2982
  let smaller_number := 482
  let quotient := 6
  (larger_number - smaller_number = 2500) ∧ 
  (larger_number = quotient * smaller_number + r) →
  (r = 90) :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_90_l2131_213103


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l2131_213130

noncomputable def probability_of_red_ball (total_balls red_balls : ℕ) : ℚ :=
  red_balls / total_balls

theorem probability_of_drawing_red_ball:
  probability_of_red_ball 5 3 = 3 / 5 :=
by
  unfold probability_of_red_ball
  norm_num

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l2131_213130


namespace NUMINAMATH_GPT_quadratic_real_roots_quadratic_product_of_roots_l2131_213131

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * m * x + m^2 + m - 3 = 0) ↔ m ≤ 3 := by
{
  sorry
}

theorem quadratic_product_of_roots (m : ℝ) (α β : ℝ) :
  α * β = 17 ∧ α^2 - 2 * m * α + m^2 + m - 3 = 0 ∧ β^2 - 2 * m * β + m^2 + m - 3 = 0 →
  m = -5 := by
{
  sorry
}

end NUMINAMATH_GPT_quadratic_real_roots_quadratic_product_of_roots_l2131_213131


namespace NUMINAMATH_GPT_middle_digit_base7_l2131_213156

theorem middle_digit_base7 (a b c : ℕ) 
  (h1 : N = 49 * a + 7 * b + c) 
  (h2 : N = 81 * c + 9 * b + a)
  (h3 : a < 7 ∧ b < 7 ∧ c < 7) : 
  b = 0 :=
by sorry

end NUMINAMATH_GPT_middle_digit_base7_l2131_213156


namespace NUMINAMATH_GPT_cherries_on_June_5_l2131_213186

theorem cherries_on_June_5 : 
  ∃ c : ℕ, (c + (c + 8) + (c + 16) + (c + 24) + (c + 32) = 130) ∧ (c + 32 = 42) :=
by
  sorry

end NUMINAMATH_GPT_cherries_on_June_5_l2131_213186


namespace NUMINAMATH_GPT_calculate_8b_l2131_213161

-- Define the conditions \(6a + 3b = 0\), \(b - 3 = a\), and \(b + c = 5\)
variables (a b c : ℝ)

theorem calculate_8b :
  (6 * a + 3 * b = 0) → (b - 3 = a) → (b + c = 5) → (8 * b = 16) :=
by
  intros h1 h2 h3
  -- Proof goes here, but we will use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_calculate_8b_l2131_213161


namespace NUMINAMATH_GPT_sqrt_four_eq_two_l2131_213155

theorem sqrt_four_eq_two : Real.sqrt 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_four_eq_two_l2131_213155


namespace NUMINAMATH_GPT_minimum_value_l2131_213172

theorem minimum_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ x : ℝ, x = 4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ∧ x ≥ 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_minimum_value_l2131_213172


namespace NUMINAMATH_GPT_find_p7_value_l2131_213124

def quadratic (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem find_p7_value (d e f : ℝ)
  (h1 : quadratic d e f 1 = 4)
  (h2 : quadratic d e f 2 = 4) :
  quadratic d e f 7 = 5 := by
  sorry

end NUMINAMATH_GPT_find_p7_value_l2131_213124


namespace NUMINAMATH_GPT_eighth_square_more_tiles_than_seventh_l2131_213116

-- Define the total number of tiles in the nth square
def total_tiles (n : ℕ) : ℕ := n^2 + 2 * n

-- Formulate the theorem statement
theorem eighth_square_more_tiles_than_seventh :
  total_tiles 8 - total_tiles 7 = 17 := by
  sorry

end NUMINAMATH_GPT_eighth_square_more_tiles_than_seventh_l2131_213116


namespace NUMINAMATH_GPT_find_y_when_x_is_4_l2131_213100

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_4_l2131_213100


namespace NUMINAMATH_GPT_value_of_expr_l2131_213139

theorem value_of_expr (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expr_l2131_213139


namespace NUMINAMATH_GPT_intersection_complement_l2131_213176

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set M
def M : Set ℕ := {0, 3, 5}

-- Define set N
def N : Set ℕ := {1, 4, 5}

-- Define the complement of N in U
def complement_U_N : Set ℕ := U \ N

-- The main theorem statement
theorem intersection_complement : M ∩ complement_U_N = {0, 3} :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_intersection_complement_l2131_213176


namespace NUMINAMATH_GPT_water_outflow_time_l2131_213136

theorem water_outflow_time (H R : ℝ) (flow_rate : ℝ → ℝ)
  (h_initial : ℝ) (t_initial : ℝ) (empty_height : ℝ) :
  H = 12 →
  R = 3 →
  (∀ h, flow_rate h = -h) →
  h_initial = 12 →
  t_initial = 0 →
  empty_height = 0 →
  ∃ t, t = (72 : ℝ) * π / 16 :=
by
  intros hL R_eq flow_rate_eq h_initial_eq t_initial_eq empty_height_eq
  sorry

end NUMINAMATH_GPT_water_outflow_time_l2131_213136


namespace NUMINAMATH_GPT_decimal_representation_of_7_over_12_eq_0_point_5833_l2131_213141

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end NUMINAMATH_GPT_decimal_representation_of_7_over_12_eq_0_point_5833_l2131_213141


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2131_213142

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2131_213142


namespace NUMINAMATH_GPT_log_sum_property_l2131_213191

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def f_inv (a : ℝ) (y : ℝ) := a ^ y

theorem log_sum_property (a : ℝ) (h1 : f_inv a 2 = 9) (h2 : f a 9 = 2) : f a 9 + f a 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_sum_property_l2131_213191


namespace NUMINAMATH_GPT_smallest_k_l2131_213166

theorem smallest_k (p : ℕ) (hp : p = 997) : 
  ∃ k : ℕ, (p^2 - k) % 10 = 0 ∧ k = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l2131_213166


namespace NUMINAMATH_GPT_mass_percentage_oxygen_NaBrO3_l2131_213199

-- Definitions
def molar_mass_Na : ℝ := 22.99
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

-- Theorem: proof that the mass percentage of oxygen in NaBrO3 is 31.81%
theorem mass_percentage_oxygen_NaBrO3 :
  ((3 * molar_mass_O) / molar_mass_NaBrO3) * 100 = 31.81 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_oxygen_NaBrO3_l2131_213199


namespace NUMINAMATH_GPT_Jace_post_break_time_correct_l2131_213184

noncomputable def Jace_post_break_time (total_distance : ℝ) (speed : ℝ) (pre_break_time : ℝ) : ℝ :=
  (total_distance - (speed * pre_break_time)) / speed

theorem Jace_post_break_time_correct :
  Jace_post_break_time 780 60 4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_Jace_post_break_time_correct_l2131_213184


namespace NUMINAMATH_GPT_smallest_n_inequality_l2131_213115

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
           (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) ∧
           n = 4 :=
by
  let n := 4
  sorry

end NUMINAMATH_GPT_smallest_n_inequality_l2131_213115


namespace NUMINAMATH_GPT_geometric_sequence_sum_5_is_75_l2131_213153

noncomputable def geometric_sequence_sum_5 (a r : ℝ) : ℝ :=
  a * (1 + r + r^2 + r^3 + r^4)

theorem geometric_sequence_sum_5_is_75 (a r : ℝ)
  (h1 : a * (1 + r + r^2) = 13)
  (h2 : a * (1 - r^7) / (1 - r) = 183) :
  geometric_sequence_sum_5 a r = 75 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_5_is_75_l2131_213153


namespace NUMINAMATH_GPT_hyperbola_s_eq_l2131_213180

theorem hyperbola_s_eq (s : ℝ) 
  (hyp1 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (5, -3) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp2 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (3, 0) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp3 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (s, -1) → x^2 / 9 - y^2 / b^2 = 1) :
  s^2 = 873 / 81 :=
sorry

end NUMINAMATH_GPT_hyperbola_s_eq_l2131_213180


namespace NUMINAMATH_GPT_width_of_rectangle_l2131_213174

-- Define the given values
def length : ℝ := 2
def area : ℝ := 8

-- State the theorem
theorem width_of_rectangle : ∃ width : ℝ, area = length * width ∧ width = 4 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_width_of_rectangle_l2131_213174


namespace NUMINAMATH_GPT_freq_distribution_correct_l2131_213105

variable (freqTable_isForm : Prop)
variable (freqHistogram_isForm : Prop)
variable (freqTable_isAccurate : Prop)
variable (freqHistogram_isIntuitive : Prop)

theorem freq_distribution_correct :
  ((freqTable_isForm ∧ freqHistogram_isForm) ∧
   (freqTable_isAccurate ∧ freqHistogram_isIntuitive)) →
  True :=
by
  intros _
  exact trivial

end NUMINAMATH_GPT_freq_distribution_correct_l2131_213105


namespace NUMINAMATH_GPT_g_g_g_3_l2131_213125

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_g_g_3 : g (g (g 3)) = 241 := by
  sorry

end NUMINAMATH_GPT_g_g_g_3_l2131_213125


namespace NUMINAMATH_GPT_missed_field_goals_l2131_213143

theorem missed_field_goals (TotalAttempts MissedFraction WideRightPercentage : ℕ) 
  (TotalAttempts_eq : TotalAttempts = 60)
  (MissedFraction_eq : MissedFraction = 15)
  (WideRightPercentage_eq : WideRightPercentage = 3) : 
  (TotalAttempts * (1 / 4) * (20 / 100) = 3) :=
  by
    sorry

end NUMINAMATH_GPT_missed_field_goals_l2131_213143


namespace NUMINAMATH_GPT_solve_eq_roots_l2131_213126

noncomputable def solve_equation (x : ℝ) : Prop :=
  (7 * x + 2) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2)

theorem solve_eq_roots (x : ℝ) (h₁ : x ≠ 2 / 3) :
  solve_equation x ↔ (x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_roots_l2131_213126


namespace NUMINAMATH_GPT_point_P_through_graph_l2131_213181

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem point_P_through_graph (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
  f a 1 = 5 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_point_P_through_graph_l2131_213181


namespace NUMINAMATH_GPT_hans_deposit_l2131_213160

noncomputable def calculate_deposit : ℝ :=
  let flat_fee := 30
  let kid_deposit := 2 * 3
  let adult_deposit := 8 * 6
  let senior_deposit := 5 * 4
  let student_deposit := 3 * 4.5
  let employee_deposit := 2 * 2.5
  let total_deposit_before_service := flat_fee + kid_deposit + adult_deposit + senior_deposit + student_deposit + employee_deposit
  let service_charge := total_deposit_before_service * 0.05
  total_deposit_before_service + service_charge

theorem hans_deposit : calculate_deposit = 128.63 :=
by
  sorry

end NUMINAMATH_GPT_hans_deposit_l2131_213160


namespace NUMINAMATH_GPT_proof_of_ratio_l2131_213147

def f (x : ℤ) : ℤ := 3 * x + 4

def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_of_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 :=
by
  sorry

end NUMINAMATH_GPT_proof_of_ratio_l2131_213147


namespace NUMINAMATH_GPT_intersection_points_l2131_213107

def curve (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x + 1

theorem intersection_points :
  {p : ℝ × ℝ | curve p.1 p.2 ∧ line p.1 p.2} = {(-1, 0), (0, 1)} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_points_l2131_213107


namespace NUMINAMATH_GPT_regular_polygon_sides_l2131_213198

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 = 144 * n) : n = 10 := 
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2131_213198


namespace NUMINAMATH_GPT_marble_weight_l2131_213190

theorem marble_weight (m d : ℝ) : (9 * m = 4 * d) → (3 * d = 36) → (m = 16 / 3) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_marble_weight_l2131_213190


namespace NUMINAMATH_GPT_max_correct_answers_l2131_213152

/--
In a 50-question multiple-choice math contest, students receive 5 points for a correct answer, 
0 points for an answer left blank, and -2 points for an incorrect answer. Jesse’s total score 
on the contest was 115. Prove that the maximum number of questions that Jesse could have answered 
correctly is 30.
-/
theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 50) (h2 : 5 * a - 2 * c = 115) : a ≤ 30 :=
by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l2131_213152


namespace NUMINAMATH_GPT_factorization_cd_c_l2131_213188

theorem factorization_cd_c (C D : ℤ) (h : ∀ y : ℤ, 20*y^2 - 117*y + 72 = (C*y - 8) * (D*y - 9)) : C * D + C = 25 :=
sorry

end NUMINAMATH_GPT_factorization_cd_c_l2131_213188


namespace NUMINAMATH_GPT_sea_star_collection_l2131_213132

theorem sea_star_collection (S : ℕ) (initial_seashells : ℕ) (initial_snails : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
  initial_seashells = 21 →
  initial_snails = 29 →
  lost_sea_creatures = 25 →
  remaining_items = 59 →
  S + initial_seashells + initial_snails = remaining_items + lost_sea_creatures →
  S = 34 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end NUMINAMATH_GPT_sea_star_collection_l2131_213132


namespace NUMINAMATH_GPT_determine_original_number_l2131_213108

theorem determine_original_number (a b c : ℕ) (m : ℕ) (N : ℕ) 
  (h1 : N = 4410) 
  (h2 : (a + b + c) % 2 = 0)
  (h3 : m = 100 * a + 10 * b + c)
  (h4 : N + m = 222 * (a + b + c)) : 
  a = 4 ∧ b = 4 ∧ c = 4 :=
by 
  sorry

end NUMINAMATH_GPT_determine_original_number_l2131_213108


namespace NUMINAMATH_GPT_medium_sized_fir_trees_count_l2131_213185

theorem medium_sized_fir_trees_count 
  (total_trees : ℕ) (ancient_oaks : ℕ) (saplings : ℕ)
  (h1 : total_trees = 96)
  (h2 : ancient_oaks = 15)
  (h3 : saplings = 58) :
  total_trees - ancient_oaks - saplings = 23 :=
by 
  sorry

end NUMINAMATH_GPT_medium_sized_fir_trees_count_l2131_213185


namespace NUMINAMATH_GPT_students_play_long_tennis_l2131_213177

-- Define the parameters for the problem
def total_students : ℕ := 38
def football_players : ℕ := 26
def both_sports_players : ℕ := 17
def neither_sports_players : ℕ := 9

-- Total students playing at least one sport
def at_least_one := total_students - neither_sports_players

-- Define the Lean theorem statement
theorem students_play_long_tennis : at_least_one = football_players + (20 : ℕ) - both_sports_players := 
by 
  -- Translate the given facts into the Lean proof structure
  have h1 : at_least_one = 29 := by rfl -- total_students - neither_sports_players
  have h2 : football_players = 26 := by rfl
  have h3 : both_sports_players = 17 := by rfl
  show 29 = 26 + 20 - 17
  sorry

end NUMINAMATH_GPT_students_play_long_tennis_l2131_213177


namespace NUMINAMATH_GPT_range_of_m_l2131_213145

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) / x

theorem range_of_m (a m : ℝ) (h₀ : a > 0) (h₁ : f a (m^2 + 1) > f a (m^2 - m + 3)) 
  : m > 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2131_213145


namespace NUMINAMATH_GPT_scientific_notation_example_l2131_213170

theorem scientific_notation_example : 3790000 = 3.79 * 10^6 := 
sorry

end NUMINAMATH_GPT_scientific_notation_example_l2131_213170


namespace NUMINAMATH_GPT_pythagorean_triple_9_12_15_l2131_213112

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end NUMINAMATH_GPT_pythagorean_triple_9_12_15_l2131_213112


namespace NUMINAMATH_GPT_salt_solution_problem_l2131_213157

theorem salt_solution_problem
  (x y : ℝ)
  (h1 : 70 + x + y = 200)
  (h2 : 0.20 * 70 + 0.60 * x + 0.35 * y = 0.45 * 200) :
  x = 122 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_salt_solution_problem_l2131_213157


namespace NUMINAMATH_GPT_find_divided_number_l2131_213168

theorem find_divided_number:
  ∃ x : ℕ, (x % 127 = 6) ∧ (2037 % 127 = 5) ∧ x = 2038 :=
by
  sorry

end NUMINAMATH_GPT_find_divided_number_l2131_213168


namespace NUMINAMATH_GPT_probability_sum_six_two_dice_l2131_213146

theorem probability_sum_six_two_dice :
  let total_outcomes := 36
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes = 5 / 36 := by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  sorry

end NUMINAMATH_GPT_probability_sum_six_two_dice_l2131_213146


namespace NUMINAMATH_GPT_Shiela_stars_per_bottle_l2131_213122

theorem Shiela_stars_per_bottle (total_stars : ℕ) (total_classmates : ℕ) (h1 : total_stars = 45) (h2 : total_classmates = 9) :
  total_stars / total_classmates = 5 := 
by 
  sorry

end NUMINAMATH_GPT_Shiela_stars_per_bottle_l2131_213122


namespace NUMINAMATH_GPT_no_unhappy_days_l2131_213110

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end NUMINAMATH_GPT_no_unhappy_days_l2131_213110


namespace NUMINAMATH_GPT_conditional_probability_l2131_213169

def prob_event_A : ℚ := 7 / 8 -- Probability of event A (at least one occurrence of tails)
def prob_event_AB : ℚ := 3 / 8 -- Probability of both events A and B happening (at least one occurrence of tails and exactly one occurrence of heads)

theorem conditional_probability (prob_A : ℚ) (prob_AB : ℚ) 
  (h1: prob_A = 7 / 8) (h2: prob_AB = 3 / 8) : 
  (prob_AB / prob_A) = 3 / 7 := 
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_conditional_probability_l2131_213169


namespace NUMINAMATH_GPT_number_of_zeros_among_50_numbers_l2131_213111

theorem number_of_zeros_among_50_numbers :
  ∀ (m n p : ℕ), (m + n + p = 50) → (m * p = 500) → n = 5 :=
by
  intros m n p h1 h2
  sorry

end NUMINAMATH_GPT_number_of_zeros_among_50_numbers_l2131_213111


namespace NUMINAMATH_GPT_exact_value_range_l2131_213171

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end NUMINAMATH_GPT_exact_value_range_l2131_213171


namespace NUMINAMATH_GPT_sum_S9_l2131_213134

variable (a : ℕ → ℤ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given condition for the sum of specific terms
def condition_given (a : ℕ → ℤ) : Prop :=
  a 2 + a 5 + a 8 = 12

-- Sum of the first 9 terms
def sum_of_first_nine_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8

-- Problem statement: Prove that given the arithmetic sequence and the condition,
-- the sum of the first 9 terms is 36
theorem sum_S9 :
  arithmetic_sequence a → condition_given a → sum_of_first_nine_terms a = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_S9_l2131_213134


namespace NUMINAMATH_GPT_one_thirds_in_nine_thirds_l2131_213154

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_one_thirds_in_nine_thirds_l2131_213154


namespace NUMINAMATH_GPT_problem_equivalence_l2131_213183

theorem problem_equivalence :
  (1 / Real.sin (Real.pi / 18) - Real.sqrt 3 / Real.sin (4 * Real.pi / 18)) = 4 := 
sorry

end NUMINAMATH_GPT_problem_equivalence_l2131_213183


namespace NUMINAMATH_GPT_find_number_l2131_213129

theorem find_number :
  ∃ x : ℝ, (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2131_213129


namespace NUMINAMATH_GPT_ab_value_l2131_213197

theorem ab_value (a b : ℝ) (log_two_3 : ℝ := Real.log 3 / Real.log 2) :
  a * log_two_3 = 1 ∧ (4 : ℝ)^b = 3 → a * b = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ab_value_l2131_213197


namespace NUMINAMATH_GPT_welders_correct_l2131_213164

-- Define the initial number of welders
def initial_welders := 12

-- Define the conditions:
-- 1. Total work is 1 job that welders can finish in 3 days.
-- 2. 9 welders leave after the first day.
-- 3. The remaining work is completed by (initial_welders - 9) in 8 days.

theorem welders_correct (W : ℕ) (h1 : W * 1/3 = 1) (h2 : (W - 9) * 8 = 2 * W) : 
  W = initial_welders :=
by
  sorry

end NUMINAMATH_GPT_welders_correct_l2131_213164


namespace NUMINAMATH_GPT_Sophie_l2131_213182

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_Sophie_l2131_213182


namespace NUMINAMATH_GPT_customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l2131_213178

theorem customer_B_cost_effectiveness (box_orig_cost box_spec_cost : ℕ) (orig_price spec_price eggs_per_box remaining_eggs : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : remaining_eggs = 20) : 
    ¬ (spec_price * 2 / (eggs_per_box * 2 - remaining_eggs) < orig_price / eggs_per_box) :=
by
  sorry

theorem customer_A_boxes_and_consumption (orig_price spec_price eggs_per_box total_cost_savings : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : total_cost_savings = 90): 
  ∃ (boxes_bought : ℕ) (avg_daily_consumption : ℕ), 
    (spec_price * boxes_bought = orig_price * boxes_bought * 2 - total_cost_savings) ∧ 
    (avg_daily_consumption = eggs_per_box * boxes_bought / 15) :=
by
  sorry

end NUMINAMATH_GPT_customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l2131_213178


namespace NUMINAMATH_GPT_SeedMixtureWeights_l2131_213150

theorem SeedMixtureWeights (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x / 3 = y / 2) (h3 : x / 3 = z / 3) :
  x = 3 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_SeedMixtureWeights_l2131_213150


namespace NUMINAMATH_GPT_mow_lawn_time_l2131_213163

noncomputable def time_to_mow (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert inches to feet
  let strips_needed := width / effective_swath
  let total_distance := strips_needed * length
  total_distance / speed

theorem mow_lawn_time : time_to_mow 100 140 30 6 4500 = 1.6 :=
by
  sorry

end NUMINAMATH_GPT_mow_lawn_time_l2131_213163


namespace NUMINAMATH_GPT_difference_between_q_and_r_l2131_213127

-- Define the variables for shares with respect to the common multiple x
def p_share (x : Nat) : Nat := 3 * x
def q_share (x : Nat) : Nat := 7 * x
def r_share (x : Nat) : Nat := 12 * x

-- Given condition: The difference between q's share and p's share is Rs. 4000
def condition_1 (x : Nat) : Prop := (q_share x - p_share x = 4000)

-- Define the theorem to prove the difference between r and q's share is Rs. 5000
theorem difference_between_q_and_r (x : Nat) (h : condition_1 x) : r_share x - q_share x = 5000 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_q_and_r_l2131_213127


namespace NUMINAMATH_GPT_original_price_of_painting_l2131_213140

theorem original_price_of_painting (purchase_price : ℝ) (fraction : ℝ) (original_price : ℝ) :
  purchase_price = 200 → fraction = 1/4 → purchase_price = original_price * fraction → original_price = 800 :=
by
  intros h1 h2 h3
  -- proof steps here
  sorry

end NUMINAMATH_GPT_original_price_of_painting_l2131_213140


namespace NUMINAMATH_GPT_cost_per_kg_paint_l2131_213189

-- Define the basic parameters
variables {sqft_per_kg : ℝ} -- the area covered by 1 kg of paint
variables {total_cost : ℝ} -- the total cost to paint the cube
variables {side_length : ℝ} -- the side length of the cube
variables {num_faces : ℕ} -- the number of faces of the cube

-- Define the conditions given in the problem
def conditions (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) : Prop :=
  sqft_per_kg = 16 ∧
  total_cost = 876 ∧
  side_length = 8 ∧
  num_faces = 6

-- Define the statement to prove, which is the cost per kg of paint
theorem cost_per_kg_paint (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) :
  conditions sqft_per_kg total_cost side_length num_faces →
  ∃ cost_per_kg : ℝ, cost_per_kg = 36.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_kg_paint_l2131_213189


namespace NUMINAMATH_GPT_cat_and_mouse_positions_after_317_moves_l2131_213173

-- Define the conditions of the problem
def cat_positions : List String := ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
def mouse_positions : List String := ["Top Left", "Top Middle", "Top Right", "Right Middle", "Bottom Right", "Bottom Middle", "Bottom Left", "Left Middle"]

-- Calculate the position of the cat after n moves
def cat_position_after_moves (n : Nat) : String :=
  cat_positions.get! (n % 4)

-- Calculate the position of the mouse after n moves
def mouse_position_after_moves (n : Nat) : String :=
  mouse_positions.get! (n % 8)

-- Prove the final positions of the cat and mouse after 317 moves
theorem cat_and_mouse_positions_after_317_moves :
  cat_position_after_moves 317 = "Top Left" ∧ mouse_position_after_moves 317 = "Bottom Middle" :=
by
  sorry

end NUMINAMATH_GPT_cat_and_mouse_positions_after_317_moves_l2131_213173


namespace NUMINAMATH_GPT_measure_of_angle_C_l2131_213113

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 360) (h2 : C = 5 * D) : C = 300 := 
by sorry

end NUMINAMATH_GPT_measure_of_angle_C_l2131_213113


namespace NUMINAMATH_GPT_depletion_rate_l2131_213195

theorem depletion_rate (initial_value final_value : ℝ) (years: ℕ) (r : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2256.25)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) :
  r = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_depletion_rate_l2131_213195


namespace NUMINAMATH_GPT_find_p_q_r_l2131_213118

def is_rel_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem find_p_q_r (x : ℝ) (p q r : ℕ)
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r)
  (hpq_rel_prime : is_rel_prime p q)
  (hp : 0 < p)
  (hq : 0 < q)
  (hr : 0 < r) :
  p + q + r = 26 :=
sorry

end NUMINAMATH_GPT_find_p_q_r_l2131_213118


namespace NUMINAMATH_GPT_condition1_a_geq_1_l2131_213128

theorem condition1_a_geq_1 (a : ℝ) :
  (∀ x ∈ ({1, 2, 3} : Set ℝ), a * x - 1 ≥ 0) → a ≥ 1 :=
by
sorry

end NUMINAMATH_GPT_condition1_a_geq_1_l2131_213128


namespace NUMINAMATH_GPT_surface_area_small_prism_l2131_213193

-- Definitions and conditions
variables (a b c : ℝ)

def small_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * a * b + 2 * a * c + 2 * b * c

def large_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * (3 * b) * (3 * b) + 2 * (3 * b) * (4 * c) + 2 * (4 * c) * (3 * b)

-- Conditions
def conditions : Prop :=
  (3 * b = 2 * a) ∧ (a = 3 * c) ∧ (large_cuboid_surface_area a b c = 360)

-- Desired result
def result : Prop :=
  small_cuboid_surface_area a b c = 88

-- The theorem
theorem surface_area_small_prism (a b c : ℝ) (h : conditions a b c) : result a b c :=
by
  sorry

end NUMINAMATH_GPT_surface_area_small_prism_l2131_213193


namespace NUMINAMATH_GPT_proof_problem_l2131_213109

def intelligentFailRate (r1 r2 r3 : ℚ) : ℚ :=
  1 - r1 * r2 * r3

def phi (p : ℚ) : ℚ :=
  30 * p * (1 - p)^29

def derivativePhi (p : ℚ) : ℚ :=
  30 * (1 - p)^28 * (1 - 30 * p)

def qualifiedPassRate (intelligentPassRate comprehensivePassRate : ℚ) : ℚ :=
  intelligentPassRate * comprehensivePassRate

theorem proof_problem :
  let r1 := (99 : ℚ) / 100
  let r2 := (98 : ℚ) / 99
  let r3 := (97 : ℚ) / 98
  let p0 := (1 : ℚ) / 30
  let comprehensivePassRate := 1 - p0
  let qualifiedRate := qualifiedPassRate (r1 * r2 * r3) comprehensivePassRate
  (intelligentFailRate r1 r2 r3 = 3 / 100) ∧
  (derivativePhi p0 = 0) ∧
  (qualifiedRate < 96 / 100) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2131_213109


namespace NUMINAMATH_GPT_area_of_rectangular_field_l2131_213162

theorem area_of_rectangular_field (L W A : ℕ) (h1 : L = 10) (h2 : 2 * W + L = 130) :
  A = 600 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l2131_213162


namespace NUMINAMATH_GPT_tallest_is_jie_l2131_213194

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end NUMINAMATH_GPT_tallest_is_jie_l2131_213194


namespace NUMINAMATH_GPT_angles_sum_540_l2131_213165

theorem angles_sum_540 (p q r s : ℝ) (h1 : ∀ a, a + (180 - a) = 180)
  (h2 : ∀ a b, (180 - a) + (180 - b) = 360 - a - b)
  (h3 : ∀ p q r, (360 - p - q) + (180 - r) = 540 - p - q - r) :
  p + q + r + s = 540 :=
sorry

end NUMINAMATH_GPT_angles_sum_540_l2131_213165


namespace NUMINAMATH_GPT_initial_men_invited_l2131_213133

theorem initial_men_invited (M W C : ℕ) (h1 : W = M / 2) (h2 : C + 10 = 30) (h3 : M + W + C = 80) (h4 : C = 20) : M = 40 :=
sorry

end NUMINAMATH_GPT_initial_men_invited_l2131_213133


namespace NUMINAMATH_GPT_ratio_R_U_l2131_213167

theorem ratio_R_U : 
  let spacing := 1 / 4
  let R := 3 * spacing
  let U := 6 * spacing
  R / U = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_R_U_l2131_213167


namespace NUMINAMATH_GPT_power_sum_l2131_213175

theorem power_sum : (-2) ^ 2007 + (-2) ^ 2008 = 2 ^ 2007 := by
  sorry

end NUMINAMATH_GPT_power_sum_l2131_213175


namespace NUMINAMATH_GPT_michael_ratio_zero_l2131_213104

theorem michael_ratio_zero (M : ℕ) (h1: M ≤ 60) (h2: 15 = (60 - M) / 2 - 15) : M = 0 := by
  sorry 

end NUMINAMATH_GPT_michael_ratio_zero_l2131_213104


namespace NUMINAMATH_GPT_cylinder_surface_area_l2131_213158

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) 
  (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (circumference_formula : c = 2 * Real.pi * r) : 
  2 * (Real.pi * r^2) + (2 * Real.pi * r * h) = 6 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l2131_213158


namespace NUMINAMATH_GPT_tangent_line_at_point_l2131_213114

theorem tangent_line_at_point
  (x y : ℝ)
  (h_curve : y = x^3 - 3 * x^2 + 1)
  (h_point : (x, y) = (1, -1)) :
  ∃ m b : ℝ, (m = -3) ∧ (b = 2) ∧ (y = m * x + b) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2131_213114


namespace NUMINAMATH_GPT_tom_roses_per_day_l2131_213138

-- Define variables and conditions
def total_roses := 168
def days_in_week := 7
def dozen := 12

-- Theorem to prove
theorem tom_roses_per_day : (total_roses / dozen) / days_in_week = 2 :=
by
  -- The actual proof would go here, using the sorry placeholder
  sorry

end NUMINAMATH_GPT_tom_roses_per_day_l2131_213138


namespace NUMINAMATH_GPT_cost_price_proof_l2131_213106

def trader_sells_66m_for_660 : Prop := ∃ cp profit sp : ℝ, sp = 660 ∧ cp * 66 + profit * 66 = sp
def profit_5_per_meter : Prop := ∃ profit : ℝ, profit = 5
def cost_price_per_meter_is_5 : Prop := ∃ cp : ℝ, cp = 5

theorem cost_price_proof : trader_sells_66m_for_660 → profit_5_per_meter → cost_price_per_meter_is_5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_price_proof_l2131_213106


namespace NUMINAMATH_GPT_oil_needed_to_half_fill_tanker_l2131_213120

theorem oil_needed_to_half_fill_tanker :
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  let current_tanker_oil := initial_tanker_oil + poured_oil
  let half_tanker_capacity := initial_tanker_capacity / 2
  let needed_oil := half_tanker_capacity - current_tanker_oil
  needed_oil = 4000 :=
by
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  have h1 : poured_oil = 3000 := by sorry
  let current_tanker_oil := initial_tanker_oil + poured_oil
  have h2 : current_tanker_oil = 6000 := by sorry
  let half_tanker_capacity := initial_tanker_capacity / 2
  have h3 : half_tanker_capacity = 10000 := by sorry
  let needed_oil := half_tanker_capacity - current_tanker_oil
  have h4 : needed_oil = 4000 := by sorry
  exact h4

end NUMINAMATH_GPT_oil_needed_to_half_fill_tanker_l2131_213120


namespace NUMINAMATH_GPT_find_principal_l2131_213101

theorem find_principal (r t1 t2 ΔI : ℝ) (h_r : r = 0.15) (h_t1 : t1 = 3.5) (h_t2 : t2 = 5) (h_ΔI : ΔI = 144) :
  ∃ P : ℝ, P = 640 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l2131_213101


namespace NUMINAMATH_GPT_prob_draw_l2131_213159

theorem prob_draw (p_not_losing p_winning p_drawing : ℝ) (h1 : p_not_losing = 0.6) (h2 : p_winning = 0.5) :
  p_drawing = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_prob_draw_l2131_213159


namespace NUMINAMATH_GPT_number_of_true_propositions_l2131_213119

noncomputable def f : ℝ → ℝ := sorry -- since it's not specified, we use sorry here

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Original proposition
def original_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, is_odd f → f 0 = 0

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) :=
  f 0 = 0 → ∀ x : ℝ, is_odd f

-- Inverse proposition (logically equivalent to the converse)
def inverse_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, ¬(is_odd f) → f 0 ≠ 0

-- Contrapositive proposition (logically equivalent to the original)
def contrapositive_proposition (f : ℝ → ℝ) :=
  f 0 ≠ 0 → ∀ x : ℝ, ¬(is_odd f)

-- Theorem statement
theorem number_of_true_propositions (f : ℝ → ℝ) :
  (original_proposition f → true) ∧
  (converse_proposition f → false) ∧
  (inverse_proposition f → false) ∧
  (contrapositive_proposition f → true) →
  2 = 2 := 
by 
  sorry -- proof to be inserted

end NUMINAMATH_GPT_number_of_true_propositions_l2131_213119


namespace NUMINAMATH_GPT_gnome_voting_l2131_213196

theorem gnome_voting (n : ℕ) :
  (∀ g : ℕ, g < n →  
   (g % 3 = 0 → (∃ k : ℕ, k * 4 = n))
   ∧ (n ≠ 0 ∧ (∀ i : ℕ, i < n → (i + 1) % n ≠ (i + 2) % n) → (∃ k : ℕ, k * 4 = n))) := 
sorry

end NUMINAMATH_GPT_gnome_voting_l2131_213196


namespace NUMINAMATH_GPT_ball_returns_to_bella_after_13_throws_l2131_213135

def girl_after_throws (start : ℕ) (throws : ℕ) : ℕ :=
  (start + throws * 5) % 13

theorem ball_returns_to_bella_after_13_throws :
  girl_after_throws 1 13 = 1 :=
sorry

end NUMINAMATH_GPT_ball_returns_to_bella_after_13_throws_l2131_213135


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2131_213102

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, (x^2)/4 - y^2 = 1) →
  (∀ x : ℝ, y = x / 2 ∨ y = -x / 2) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2131_213102


namespace NUMINAMATH_GPT_xyz_sum_48_l2131_213148

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end NUMINAMATH_GPT_xyz_sum_48_l2131_213148


namespace NUMINAMATH_GPT_height_of_Linda_room_l2131_213192

theorem height_of_Linda_room (w l: ℝ) (h a1 a2 a3 paint_area: ℝ) 
  (hw: w = 20) (hl: l = 20) 
  (d1_h: a1 = 3) (d1_w: a2 = 7) 
  (d2_h: a3 = 4) (d2_w: a4 = 6) 
  (d3_h: a5 = 5) (d3_w: a6 = 7) 
  (total_paint_area: paint_area = 560):
  h = 6 := 
by
  sorry

end NUMINAMATH_GPT_height_of_Linda_room_l2131_213192


namespace NUMINAMATH_GPT_exists_n_for_all_k_l2131_213149

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 5^k ∣ (n^2 + 1) :=
sorry

end NUMINAMATH_GPT_exists_n_for_all_k_l2131_213149


namespace NUMINAMATH_GPT_square_garden_perimeter_l2131_213117

theorem square_garden_perimeter (A : ℝ) (hA : A = 450) : 
    ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  by
    sorry

end NUMINAMATH_GPT_square_garden_perimeter_l2131_213117


namespace NUMINAMATH_GPT_smallest_positive_integer_23n_mod_5678_mod_11_l2131_213123

theorem smallest_positive_integer_23n_mod_5678_mod_11 :
  ∃ n : ℕ, 0 < n ∧ 23 * n % 11 = 5678 % 11 ∧ ∀ m : ℕ, 0 < m ∧ 23 * m % 11 = 5678 % 11 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_23n_mod_5678_mod_11_l2131_213123


namespace NUMINAMATH_GPT_fraction_remains_unchanged_l2131_213179

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (3 * x)) / (2 * (3 * x) - 3 * y) = (3 * x) / (2 * x - y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_remains_unchanged_l2131_213179


namespace NUMINAMATH_GPT_five_minus_x_eight_l2131_213137

theorem five_minus_x_eight (x y : ℤ) (h1 : 5 + x = 3 - y) (h2 : 2 + y = 6 + x) : 5 - x = 8 :=
by
  sorry

end NUMINAMATH_GPT_five_minus_x_eight_l2131_213137


namespace NUMINAMATH_GPT_find_a_2_find_a_n_l2131_213151

-- Define the problem conditions and questions as types
def S_3 (a_1 a_2 a_3 : ℝ) : Prop := a_1 + a_2 + a_3 = 7
def arithmetic_mean_condition (a_1 a_2 a_3 : ℝ) : Prop :=
  (a_1 + 3 + a_3 + 4) / 2 = 3 * a_2

-- Prove that a_2 = 2 given the conditions
theorem find_a_2 (a_1 a_2 a_3 : ℝ) (h1 : S_3 a_1 a_2 a_3) (h2: arithmetic_mean_condition a_1 a_2 a_3) :
  a_2 = 2 := 
sorry

-- Define the general term for a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Prove the formula for the general term of the geometric sequence given the conditions and a_2 found
theorem find_a_n (a : ℕ → ℝ) (q : ℝ) (h1 : S_3 (a 1) (a 2) (a 3)) (h2 : arithmetic_mean_condition (a 1) (a 2) (a 3)) (h3 : geometric_sequence a q) : 
  (q = (1/2) → ∀ n, a n = (1 / 2)^(n - 3))
  ∧ (q = 2 → ∀ n, a n = 2^(n - 1)) := 
sorry

end NUMINAMATH_GPT_find_a_2_find_a_n_l2131_213151


namespace NUMINAMATH_GPT_correct_assignment_statement_l2131_213121

-- Definitions according to the problem conditions
def input_statement (x : Nat) : Prop := x = 3
def assignment_statement1 (A B : Nat) : Prop := A = B ∧ B = 2
def assignment_statement2 (T : Nat) : Prop := T = T * T
def output_statement (A : Nat) : Prop := A = 4

-- Lean statement for the problem. We need to prove that the assignment_statement2 is correct.
theorem correct_assignment_statement (T : Nat) : assignment_statement2 T :=
by sorry

end NUMINAMATH_GPT_correct_assignment_statement_l2131_213121


namespace NUMINAMATH_GPT_largest_angle_triangle_l2131_213144

-- Definition of constants and conditions
def right_angle : ℝ := 90
def angle_sum : ℝ := 120
def angle_difference : ℝ := 20

-- Given two angles of a triangle sum to 120 degrees and one is 20 degrees greater than the other,
-- Prove the largest angle in the triangle is 70 degrees
theorem largest_angle_triangle (A B C : ℝ) (hA : A + B = angle_sum) (hB : B = A + angle_difference) (hC : A + B + C = 180) : 
  max A (max B C) = 70 := 
by 
  sorry

end NUMINAMATH_GPT_largest_angle_triangle_l2131_213144
