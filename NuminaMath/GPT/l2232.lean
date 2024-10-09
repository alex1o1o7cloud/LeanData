import Mathlib

namespace michael_choose_classes_l2232_223274

-- Michael's scenario setup
def total_classes : ℕ := 10
def compulsory_class : ℕ := 1
def remaining_classes : ℕ := total_classes - compulsory_class
def total_to_choose : ℕ := 4
def additional_to_choose : ℕ := total_to_choose - compulsory_class

-- Correct answer based on the conditions
def correct_answer : ℕ := 84

-- Function to compute the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove the number of ways Michael can choose his classes
theorem michael_choose_classes : binomial 9 3 = correct_answer := by
  rw [binomial, Nat.factorial]
  sorry

end michael_choose_classes_l2232_223274


namespace range_of_a_l2232_223210

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l2232_223210


namespace vector_sum_to_zero_l2232_223295

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V] {A B C : V}

theorem vector_sum_to_zero (AB BC CA : V) (hAB : AB = B - A) (hBC : BC = C - B) (hCA : CA = A - C) :
  AB + BC + CA = 0 := by
  sorry

end vector_sum_to_zero_l2232_223295


namespace range_of_a_l2232_223276

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3) ↔ (-1 < a ∧ a < 2) :=
by 
  sorry

end range_of_a_l2232_223276


namespace total_apples_l2232_223262

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end total_apples_l2232_223262


namespace track_width_l2232_223216

variable (r1 r2 r3 : ℝ)

def cond1 : Prop := 2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi
def cond2 : Prop := 2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi

theorem track_width (h1 : cond1 r1 r2) (h2 : cond2 r2 r3) : r3 - r1 = 25 := by
  sorry

end track_width_l2232_223216


namespace perpendicular_lines_l2232_223209

theorem perpendicular_lines (a : ℝ) : 
  (a = -1 → (∀ x y : ℝ, 4 * x - (a + 1) * y + 9 = 0 → x ≠ 0 →  y ≠ 0 → 
  ∃ b : ℝ, (b^2 + 1) * x - b * y + 6 = 0)) ∧ 
  (∀ x y : ℝ, (4 * x - (a + 1) * y + 9 = 0) ∧ (∃ x y : ℝ, (a^2 - 1) * x - a * y + 6 = 0) → a ≠ -1) := 
sorry

end perpendicular_lines_l2232_223209


namespace arithmetic_sequence_property_l2232_223259

-- Define arithmetic sequence and given condition
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Lean 4 statement
theorem arithmetic_sequence_property {a : ℕ → ℝ} (h : arithmetic_sequence a) (h1 : a 6 = 30) : a 3 + a 9 = 60 :=
by
  sorry

end arithmetic_sequence_property_l2232_223259


namespace juggling_contest_l2232_223255

theorem juggling_contest (B : ℕ) (rot_baseball : ℕ := 80)
    (rot_per_apple : ℕ := 101) (num_apples : ℕ := 4)
    (winner_rotations : ℕ := 404) :
    (num_apples * rot_per_apple = winner_rotations) :=
by
  sorry

end juggling_contest_l2232_223255


namespace proof_problem_l2232_223294

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 / 3 ∧ x ≤ 2

theorem proof_problem (x : ℝ) (h : valid_x x) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 :=
sorry

end proof_problem_l2232_223294


namespace integer_solution_inequality_l2232_223235

theorem integer_solution_inequality (x : ℤ) : ((x - 1)^2 ≤ 4) → ([-1, 0, 1, 2, 3].count x = 5) :=
by
  sorry

end integer_solution_inequality_l2232_223235


namespace possible_second_game_scores_count_l2232_223298

theorem possible_second_game_scores_count :
  ∃ (A1 A3 B2 : ℕ),
  (A1 + A3 = 22) ∧ (B2 = 11) ∧ (A1 < 11) ∧ (A3 < 11) ∧ ((B2 - A2 = 2) ∨ (B2 >= A2 + 2)) ∧ (A1 + B1 + A2 + B2 + A3 + B3 = 62) :=
  sorry

end possible_second_game_scores_count_l2232_223298


namespace compute_combination_l2232_223275

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l2232_223275


namespace part1_l2232_223289

theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), 2 * x + a / (x + 1) ≥ 0) → a ≥ -3 / 2 :=
sorry

end part1_l2232_223289


namespace maximum_value_fraction_sum_l2232_223232

theorem maximum_value_fraction_sum (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : 0 < c) (hd : 0 < d) (h1 : a + c = 20) (h2 : (a : ℝ) / b + (c : ℝ) / d < 1) :
  (a : ℝ) / b + (c : ℝ) / d ≤ 1385 / 1386 :=
sorry

end maximum_value_fraction_sum_l2232_223232


namespace smallest_n_l2232_223220

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l2232_223220


namespace problem_l2232_223270

open Real

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (1 / a) + (4 / b) + (9 / c) ≤ 36 / (a + b + c)) 
  : (2 * b + 3 * c) / (a + b + c) = 13 / 6 :=
sorry

end problem_l2232_223270


namespace least_element_in_T_l2232_223231

variable (S : Finset ℕ)
variable (T : Finset ℕ)
variable (hS : S = Finset.range 16 \ {0})
variable (hT : T.card = 5)
variable (hTsubS : T ⊆ S)
variable (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0))

theorem least_element_in_T (S T : Finset ℕ) (hT : T.card = 5) (hTsubS : T ⊆ S)
  (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) : 
  ∃ m ∈ T, m = 5 :=
by
  sorry

end least_element_in_T_l2232_223231


namespace solve_for_N_l2232_223293

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l2232_223293


namespace Eve_age_l2232_223265

theorem Eve_age (Adam_age : ℕ) (Eve_age : ℕ) (h1 : Adam_age = 9) (h2 : Eve_age = Adam_age + 5)
  (h3 : ∃ k : ℕ, Eve_age + 1 = k * (Adam_age - 4)) : Eve_age = 14 :=
sorry

end Eve_age_l2232_223265


namespace donovan_points_needed_l2232_223280

-- Definitions based on conditions
def average_points := 26
def games_played := 15
def total_games := 20
def goal_average := 30

-- Assertion
theorem donovan_points_needed :
  let total_points_needed := goal_average * total_games
  let points_already_scored := average_points * games_played
  let remaining_games := total_games - games_played
  let remaining_points_needed := total_points_needed - points_already_scored
  let points_per_game_needed := remaining_points_needed / remaining_games
  points_per_game_needed = 42 :=
  by
    -- Proof skipped
    sorry

end donovan_points_needed_l2232_223280


namespace sum_of_first_five_terms_l2232_223205

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_sum_n : ∀ n, S n = n / 2 * (a 1 + a n))
  (h_roots : ∀ x, x^2 - x - 3 = 0 → x = a 2 ∨ x = a 4)
  (h_vieta : a 2 + a 4 = 1) :
  S 5 = 5 / 2 :=
  sorry

end sum_of_first_five_terms_l2232_223205


namespace martin_total_waste_is_10_l2232_223287

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l2232_223287


namespace pete_total_blocks_traveled_l2232_223249

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled_l2232_223249


namespace simplify_and_evaluate_div_expr_l2232_223230

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l2232_223230


namespace sum_mean_median_mode_l2232_223263

theorem sum_mean_median_mode (l : List ℚ) (h : l = [1, 2, 2, 3, 3, 3, 3, 4, 5]) :
    let mean := (1 + 2 + 2 + 3 + 3 + 3 + 3 + 4 + 5) / 9
    let median := 3
    let mode := 3
    mean + median + mode = 8.888 :=
by
  sorry

end sum_mean_median_mode_l2232_223263


namespace incorrect_operation_D_l2232_223225

theorem incorrect_operation_D (x y: ℝ) : ¬ (-2 * x * (x - y) = -2 * x^2 - 2 * x * y) :=
by sorry

end incorrect_operation_D_l2232_223225


namespace solving_linear_equations_problems_l2232_223292

def num_total_math_problems : ℕ := 140
def percent_algebra_problems : ℝ := 0.40
def fraction_solving_linear_equations : ℝ := 0.50

theorem solving_linear_equations_problems :
  let num_algebra_problems := percent_algebra_problems * num_total_math_problems
  let num_solving_linear_equations := fraction_solving_linear_equations * num_algebra_problems
  num_solving_linear_equations = 28 :=
by
  sorry

end solving_linear_equations_problems_l2232_223292


namespace simson_line_properties_l2232_223200

-- Given a triangle ABC
variables {A B C M P Q R H : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] 
variables [Inhabited M] [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited H]

-- Conditions
def is_point_on_circumcircle (A B C : Type) (M : Type) : Prop :=
sorry  -- formal definition that M is on the circumcircle of triangle ABC

def perpendicular_dropped_to_side (M : Type) (side : Type) (foot : Type) : Prop :=
sorry  -- formal definition of a perpendicular dropping from M to a side

def is_orthocenter (A B C H : Type) : Prop := 
sorry  -- formal definition that H is the orthocenter of triangle ABC

-- Proof Goal 1: The points P, Q, R are collinear (Simson line)
def simson_line (A B C M P Q R : Type) : Prop :=
sorry  -- formal definition and proof that P, Q, R are collinear

-- Proof Goal 2: The Simson line is equidistant from point M and the orthocenter H
def simson_line_equidistant (M H P Q R : Type) : Prop :=
sorry  -- formal definition and proof that Simson line is equidistant from M and H

-- Main theorem combining both proof goals
theorem simson_line_properties 
  (A B C M P Q R H : Type)
  (M_on_circumcircle : is_point_on_circumcircle A B C M)
  (perp_to_BC : perpendicular_dropped_to_side M (B × C) P)
  (perp_to_CA : perpendicular_dropped_to_side M (C × A) Q)
  (perp_to_AB : perpendicular_dropped_to_side M (A × B) R)
  (H_is_orthocenter : is_orthocenter A B C H) :
  simson_line A B C M P Q R ∧ simson_line_equidistant M H P Q R := 
by sorry

end simson_line_properties_l2232_223200


namespace trigonometric_identity_proof_l2232_223222

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem trigonometric_identity_proof : 2 * α - β = π / 2 := 
by 
  sorry

end trigonometric_identity_proof_l2232_223222


namespace min_value_l2232_223261

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l2232_223261


namespace twelfth_term_geometric_sequence_l2232_223268

-- Define the first term and common ratio
def a1 : Int := 5
def r : Int := -3

-- Define the formula for the nth term of the geometric sequence
def nth_term (n : Nat) : Int := a1 * r^(n-1)

-- The statement to be proved: that the twelfth term is -885735
theorem twelfth_term_geometric_sequence : nth_term 12 = -885735 := by
  sorry

end twelfth_term_geometric_sequence_l2232_223268


namespace theater_rows_l2232_223286

theorem theater_rows (R : ℕ) (h1 : R < 30 → ∃ r : ℕ, r < R ∧ r * 2 ≥ 30) (h2 : R ≥ 29 → 26 + 3 ≤ R) : R = 29 :=
by
  sorry

end theater_rows_l2232_223286


namespace perfect_square_trinomial_l2232_223223

theorem perfect_square_trinomial (m x : ℝ) : 
  ∃ a b : ℝ, (4 * x^2 + (m - 3) * x + 1 = (a + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_l2232_223223


namespace square_area_l2232_223253

theorem square_area (p : ℝ → ℝ) (a b : ℝ) (h₁ : ∀ x, p x = x^2 + 3 * x + 2) (h₂ : p a = 5) (h₃ : p b = 5) (h₄ : a ≠ b) : (b - a)^2 = 21 :=
by
  sorry

end square_area_l2232_223253


namespace green_shirt_pairs_l2232_223245

theorem green_shirt_pairs (r g : ℕ) (p total_pairs red_pairs : ℕ) :
  r = 63 → g = 69 → p = 66 → red_pairs = 25 → (g - (r - red_pairs * 2)) / 2 = 28 :=
by
  intros hr hg hp hred_pairs
  sorry

end green_shirt_pairs_l2232_223245


namespace no_nat_exists_perfect_cubes_l2232_223251

theorem no_nat_exists_perfect_cubes : ¬ ∃ n : ℕ, ∃ a b : ℤ, 2^(n + 1) - 1 = a^3 ∧ 2^(n - 1)*(2^n - 1) = b^3 := 
by
  sorry

end no_nat_exists_perfect_cubes_l2232_223251


namespace contrapositive_x_squared_eq_one_l2232_223206

theorem contrapositive_x_squared_eq_one (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) := by
  sorry

end contrapositive_x_squared_eq_one_l2232_223206


namespace total_shaded_area_correct_l2232_223256
-- Let's import the mathematical library.

-- Define the problem-related conditions.
def first_rectangle_length : ℕ := 4
def first_rectangle_width : ℕ := 15
def second_rectangle_length : ℕ := 5
def second_rectangle_width : ℕ := 12
def third_rectangle_length : ℕ := 2
def third_rectangle_width : ℕ := 2

-- Define the areas based on the problem conditions.
def A1 : ℕ := first_rectangle_length * first_rectangle_width
def A2 : ℕ := second_rectangle_length * second_rectangle_width
def A_overlap_12 : ℕ := first_rectangle_length * second_rectangle_length
def A3 : ℕ := third_rectangle_length * third_rectangle_width

-- Define the total shaded area formula.
def total_shaded_area : ℕ := A1 + A2 - A_overlap_12 + A3

-- Statement of the theorem to prove.
theorem total_shaded_area_correct :
  total_shaded_area = 104 :=
by
  sorry

end total_shaded_area_correct_l2232_223256


namespace triangle_weight_l2232_223218

variables (S C T : ℕ)

def scale1 := (S + C = 8)
def scale2 := (S + 2 * C = 11)
def scale3 := (C + 2 * T = 15)

theorem triangle_weight (h1 : scale1 S C) (h2 : scale2 S C) (h3 : scale3 C T) : T = 6 :=
by 
  sorry

end triangle_weight_l2232_223218


namespace expected_value_is_correct_l2232_223211

-- Given conditions
def prob_heads : ℚ := 2 / 5
def prob_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def loss_amount_tails : ℚ := -3

-- Expected value calculation
def expected_value : ℚ := prob_heads * win_amount_heads + prob_tails * loss_amount_tails

-- Property to prove
theorem expected_value_is_correct : expected_value = 0.2 := sorry

end expected_value_is_correct_l2232_223211


namespace central_angle_of_probability_l2232_223233

theorem central_angle_of_probability (x : ℝ) (h1 : x / 360 = 1 / 6) : x = 60 := by
  have h2 : x = 60 := by
    linarith
  exact h2

end central_angle_of_probability_l2232_223233


namespace large_pile_toys_l2232_223207

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l2232_223207


namespace selected_student_in_eighteenth_group_l2232_223290

def systematic_sampling (first_number common_difference nth_term : ℕ) : ℕ :=
  first_number + (nth_term - 1) * common_difference

theorem selected_student_in_eighteenth_group :
  systematic_sampling 22 50 18 = 872 :=
by
  sorry

end selected_student_in_eighteenth_group_l2232_223290


namespace total_amount_paid_l2232_223243

-- Definitions based on the conditions in step a)
def ring_cost : ℕ := 24
def ring_quantity : ℕ := 2

-- Statement to prove that the total cost is $48.
theorem total_amount_paid : ring_quantity * ring_cost = 48 := 
by
  sorry

end total_amount_paid_l2232_223243


namespace combined_score_of_three_students_left_l2232_223277

variable (T S : ℕ) (avg16 avg13 : ℝ) (N16 N13 : ℕ)

theorem combined_score_of_three_students_left (h_avg16 : avg16 = 62.5) 
  (h_avg13 : avg13 = 62.0) (h_N16 : N16 = 16) (h_N13 : N13 = 13) 
  (h_total16 : T = avg16 * N16) (h_total13 : T - S = avg13 * N13) :
  S = 194 :=
by
  sorry

end combined_score_of_three_students_left_l2232_223277


namespace count_multiples_12_9_l2232_223236

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end count_multiples_12_9_l2232_223236


namespace function_correct_max_min_values_l2232_223283

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

@[simp]
theorem function_correct : (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧ 
                           (f (3 * Real.pi / 8) = 0) ∧ 
                           (f (Real.pi / 8) = 2) :=
by
  sorry

theorem max_min_values : (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = -2) ∧ 
                         (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = 2) :=
by
  sorry

end function_correct_max_min_values_l2232_223283


namespace sqrt_5th_of_x_sqrt_4th_x_l2232_223204

theorem sqrt_5th_of_x_sqrt_4th_x (x : ℝ) (hx : 0 < x) : Real.sqrt (x * Real.sqrt (x ^ (1 / 4))) = x ^ (1 / 4) :=
by
  sorry

end sqrt_5th_of_x_sqrt_4th_x_l2232_223204


namespace particular_solution_ODE_l2232_223264

theorem particular_solution_ODE (y : ℝ → ℝ) (h : ∀ x, deriv y x + y x * Real.tan x = 0) (h₀ : y 0 = 2) :
  ∀ x, y x = 2 * Real.cos x :=
sorry

end particular_solution_ODE_l2232_223264


namespace range_of_a_l2232_223227

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) → (-1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l2232_223227


namespace scientific_notation_240000_l2232_223228

theorem scientific_notation_240000 :
  240000 = 2.4 * 10^5 :=
by
  sorry

end scientific_notation_240000_l2232_223228


namespace find_rate_of_interest_l2232_223240

-- Conditions
def principal : ℕ := 4200
def time : ℕ := 2
def interest_12 : ℕ := principal * 12 * time / 100
def additional_interest : ℕ := 504
def total_interest_r : ℕ := interest_12 + additional_interest

-- Theorem Statement
theorem find_rate_of_interest (r : ℕ) (h : 1512 = principal * r * time / 100) : r = 18 :=
by sorry

end find_rate_of_interest_l2232_223240


namespace general_term_l2232_223299

noncomputable def S : ℕ → ℤ
| n => 3 * n ^ 2 - 2 * n + 1

def a : ℕ → ℤ
| 0 => 2  -- Since sequences often start at n=1 and MATLAB indexing starts at 0.
| 1 => 2
| (n+2) => 6 * (n + 2) - 5

theorem general_term (n : ℕ) : 
  a n = if n = 1 then 2 else 6 * n - 5 :=
by sorry

end general_term_l2232_223299


namespace james_payment_l2232_223226

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end james_payment_l2232_223226


namespace cookies_in_one_row_l2232_223278

theorem cookies_in_one_row
  (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ)
  (h_trays : num_trays = 4) (h_rows : rows_per_tray = 5) (h_cookies : total_cookies = 120) :
  total_cookies / (num_trays * rows_per_tray) = 6 := by
  sorry

end cookies_in_one_row_l2232_223278


namespace tables_count_l2232_223238

theorem tables_count (c t : Nat) (h1 : c = 8 * t) (h2 : 3 * c + 5 * t = 580) : t = 20 :=
by
  sorry

end tables_count_l2232_223238


namespace total_trees_in_forest_l2232_223247

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l2232_223247


namespace MrsHilt_money_left_l2232_223285

theorem MrsHilt_money_left (initial_amount pencil_cost remaining_amount : ℕ) 
  (h_initial : initial_amount = 15) 
  (h_cost : pencil_cost = 11) 
  (h_remaining : remaining_amount = 4) : 
  initial_amount - pencil_cost = remaining_amount := 
by 
  sorry

end MrsHilt_money_left_l2232_223285


namespace find_ac_bd_l2232_223257

variable (a b c d : ℝ)

axiom cond1 : a^2 + b^2 = 1
axiom cond2 : c^2 + d^2 = 1
axiom cond3 : a * d - b * c = 1 / 7

theorem find_ac_bd : a * c + b * d = 4 * Real.sqrt 3 / 7 := by
  sorry

end find_ac_bd_l2232_223257


namespace no_integer_solutions_l2232_223296

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), (x ≠ 1 ∧ (x^7 - 1) / (x - 1) = (y^5 - 1)) :=
sorry

end no_integer_solutions_l2232_223296


namespace required_connections_l2232_223271

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l2232_223271


namespace linda_five_dollar_bills_l2232_223272

theorem linda_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end linda_five_dollar_bills_l2232_223272


namespace anna_plants_needed_l2232_223229

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l2232_223229


namespace correct_system_of_equations_l2232_223213

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : 3 * x = 5 * y - 6)
  (h2 : y = 2 * x - 10) : 
  (3 * x = 5 * y - 6) ∧ (y = 2 * x - 10) :=
by
  sorry

end correct_system_of_equations_l2232_223213


namespace vector_subtraction_l2232_223221

theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - (2 : ℝ) • b = (7, 3) :=
by
  rw [h1, h2]
  simp
  sorry

end vector_subtraction_l2232_223221


namespace find_x_l2232_223281

theorem find_x (n x : ℚ) (h1 : 3 * n + x = 6 * n - 10) (h2 : n = 25 / 3) : x = 15 :=
by
  sorry

end find_x_l2232_223281


namespace problem_f_increasing_l2232_223203

theorem problem_f_increasing (a : ℝ) 
  (h1 : ∀ x, 2 ≤ x → 0 < x^2 - a * x + 3 * a) 
  (h2 : ∀ x, 2 ≤ x → 0 ≤ 2 * x - a) : 
  -4 < a ∧ a ≤ 4 := by
  sorry

end problem_f_increasing_l2232_223203


namespace divisibility_l2232_223241

theorem divisibility (a : ℤ) : (5 ∣ a^3) ↔ (5 ∣ a) := 
by sorry

end divisibility_l2232_223241


namespace luke_fish_fillets_l2232_223215

theorem luke_fish_fillets (daily_fish : ℕ) (days : ℕ) (fillets_per_fish : ℕ) 
  (h1 : daily_fish = 2) (h2 : days = 30) (h3 : fillets_per_fish = 2) : 
  daily_fish * days * fillets_per_fish = 120 := 
by 
  sorry

end luke_fish_fillets_l2232_223215


namespace rs_division_l2232_223279

theorem rs_division (a b c : ℝ) 
  (h1 : a = 1 / 2 * b)
  (h2 : b = 1 / 2 * c)
  (h3 : a + b + c = 700) : 
  c = 400 :=
sorry

end rs_division_l2232_223279


namespace expression_value_l2232_223224

theorem expression_value (x y z : ℤ) (h1: x = 2) (h2: y = -3) (h3: z = 1) :
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := 
by
  sorry

end expression_value_l2232_223224


namespace sin_beta_value_l2232_223202

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 4 / 5) 
  (h4 : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 33 / 65 := 
by 
  sorry

end sin_beta_value_l2232_223202


namespace coin_flip_sequences_count_l2232_223246

theorem coin_flip_sequences_count : (2 ^ 16) = 65536 :=
by
  sorry

end coin_flip_sequences_count_l2232_223246


namespace triangles_congruent_alternative_condition_l2232_223208

theorem triangles_congruent_alternative_condition
  (A B C A' B' C' : Type)
  (AB A'B' AC A'C' : ℝ)
  (angleA angleA' : ℝ)
  (h1 : AB = A'B')
  (h2 : angleA = angleA')
  (h3 : AC = A'C') :
  ∃ (triangleABC triangleA'B'C' : Type), (triangleABC = triangleA'B'C') :=
by sorry

end triangles_congruent_alternative_condition_l2232_223208


namespace quadratic_solutions_l2232_223269

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l2232_223269


namespace num_adult_tickets_is_35_l2232_223250

noncomputable def num_adult_tickets_sold (A C: ℕ): Prop :=
  A + C = 85 ∧ 5 * A + 2 * C = 275

theorem num_adult_tickets_is_35: ∃ A C: ℕ, num_adult_tickets_sold A C ∧ A = 35 :=
by
  -- Definitions based on the provided conditions
  sorry

end num_adult_tickets_is_35_l2232_223250


namespace common_root_quadratic_l2232_223217

theorem common_root_quadratic (a x1: ℝ) :
  (x1^2 + a * x1 + 1 = 0) ∧ (x1^2 + x1 + a = 0) ↔ a = -2 :=
sorry

end common_root_quadratic_l2232_223217


namespace ab_value_l2232_223260

variables {a b : ℝ}

theorem ab_value (h₁ : a - b = 6) (h₂ : a^2 + b^2 = 50) : ab = 7 :=
sorry

end ab_value_l2232_223260


namespace total_cars_parked_l2232_223288

theorem total_cars_parked
  (area_a : ℕ) (util_a : ℕ)
  (area_b : ℕ) (util_b : ℕ)
  (area_c : ℕ) (util_c : ℕ)
  (area_d : ℕ) (util_d : ℕ)
  (space_per_car : ℕ) 
  (ha: area_a = 400 * 500)
  (hu_a: util_a = 80)
  (hb: area_b = 600 * 700)
  (hu_b: util_b = 75)
  (hc: area_c = 500 * 800)
  (hu_c: util_c = 65)
  (hd: area_d = 300 * 900)
  (hu_d: util_d = 70)
  (h_sp: space_per_car = 10) :
  (util_a * area_a / 100 / space_per_car + 
   util_b * area_b / 100 / space_per_car + 
   util_c * area_c / 100 / space_per_car + 
   util_d * area_d / 100 / space_per_car) = 92400 :=
by sorry

end total_cars_parked_l2232_223288


namespace kindergarten_students_percentage_is_correct_l2232_223273

-- Definitions based on conditions
def total_students_annville : ℕ := 150
def total_students_cleona : ℕ := 250
def percent_kindergarten_annville : ℕ := 14
def percent_kindergarten_cleona : ℕ := 10

-- Calculation of number of kindergarten students
def kindergarten_students_annville : ℕ := total_students_annville * percent_kindergarten_annville / 100
def kindergarten_students_cleona : ℕ := total_students_cleona * percent_kindergarten_cleona / 100
def total_kindergarten_students : ℕ := kindergarten_students_annville + kindergarten_students_cleona
def total_students : ℕ := total_students_annville + total_students_cleona
def kindergarten_percentage : ℚ := (total_kindergarten_students * 100) / total_students

-- The theorem to be proved using the conditions
theorem kindergarten_students_percentage_is_correct : kindergarten_percentage = 11.5 := by
  sorry

end kindergarten_students_percentage_is_correct_l2232_223273


namespace find_num_female_workers_l2232_223284

-- Defining the given constants and equations
def num_male_workers : Nat := 20
def num_child_workers : Nat := 5
def wage_male_worker : Nat := 35
def wage_female_worker : Nat := 20
def wage_child_worker : Nat := 8
def avg_wage_paid : Nat := 26

-- Defining the total number of workers and total daily wage
def total_workers (num_female_workers : Nat) : Nat := 
  num_male_workers + num_female_workers + num_child_workers

def total_wage (num_female_workers : Nat) : Nat :=
  (num_male_workers * wage_male_worker) + (num_female_workers * wage_female_worker) + (num_child_workers * wage_child_worker)

-- Proving the number of female workers given the average wage
theorem find_num_female_workers (F : Nat) 
  (h : avg_wage_paid * total_workers F = total_wage F) : 
  F = 15 :=
by
  sorry

end find_num_female_workers_l2232_223284


namespace tan_alpha_tan_beta_value_l2232_223214

theorem tan_alpha_tan_beta_value
  (α β : ℝ)
  (h1 : Real.cos (α + β) = 1 / 5)
  (h2 : Real.cos (α - β) = 3 / 5) :
  Real.tan α * Real.tan β = 1 / 2 :=
by
  sorry

end tan_alpha_tan_beta_value_l2232_223214


namespace larger_number_is_1617_l2232_223258

-- Given conditions
variables (L S : ℤ)
axiom condition1 : L - S = 1515
axiom condition2 : L = 16 * S + 15

-- To prove
theorem larger_number_is_1617 : L = 1617 := by
  sorry

end larger_number_is_1617_l2232_223258


namespace largest_coins_l2232_223242

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l2232_223242


namespace goods_train_cross_platform_time_l2232_223254

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let total_distance : ℝ := length_train + length_platform
  total_distance / speed_mps

theorem goods_train_cross_platform_time :
  time_to_cross_platform 72 290.04 230 = 26.002 :=
by
  -- The proof is omitted
  sorry

end goods_train_cross_platform_time_l2232_223254


namespace half_angle_in_second_quadrant_l2232_223219

theorem half_angle_in_second_quadrant (α : Real) (h1 : 180 < α ∧ α < 270)
        (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
        90 < α / 2 ∧ α / 2 < 180 :=
sorry

end half_angle_in_second_quadrant_l2232_223219


namespace minimum_value_l2232_223291

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1487

theorem minimum_value : ∃ x : ℝ, f x = 1484 := 
sorry

end minimum_value_l2232_223291


namespace sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l2232_223244

-- Problem 1
theorem sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3 : |Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2 : Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l2232_223244


namespace distance_MF_l2232_223239

-- Define the conditions for the problem
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus : (ℝ × ℝ) := (2, 0)

def lies_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

def distance_to_line (M : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  abs (M.1 - line_x)

def point_M_conditions (M : ℝ × ℝ) : Prop :=
  distance_to_line M (-3) = 6 ∧ lies_on_parabola M

-- The final proof problem statement in Lean
theorem distance_MF (M : ℝ × ℝ) (h : point_M_conditions M) : dist M focus = 5 :=
by sorry

end distance_MF_l2232_223239


namespace car_arrives_before_bus_l2232_223237

theorem car_arrives_before_bus
  (d : ℝ) (s_bus : ℝ) (s_car : ℝ) (v : ℝ)
  (h1 : d = 240)
  (h2 : s_bus = 40)
  (h3 : s_car = v)
  : 56 < v ∧ v < 120 := 
sorry

end car_arrives_before_bus_l2232_223237


namespace f_2_eq_1_l2232_223234

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

theorem f_2_eq_1 (a b : ℝ) (h : f a b (-2) = 1) : f a b 2 = 1 :=
by {
  sorry 
}

end f_2_eq_1_l2232_223234


namespace ratio_is_one_to_five_l2232_223297

def ratio_of_minutes_to_hour (twelve_minutes : ℕ) (one_hour : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd twelve_minutes one_hour
  (twelve_minutes / gcd, one_hour / gcd)

theorem ratio_is_one_to_five : ratio_of_minutes_to_hour 12 60 = (1, 5) := 
by 
  sorry

end ratio_is_one_to_five_l2232_223297


namespace smallest_nat_number_l2232_223267

theorem smallest_nat_number (x : ℕ) 
  (h1 : ∃ z : ℕ, x + 3 = 5 * z) 
  (h2 : ∃ n : ℕ, x - 3 = 6 * n) : x = 27 := 
sorry

end smallest_nat_number_l2232_223267


namespace cody_needs_total_steps_l2232_223266

theorem cody_needs_total_steps 
  (weekly_steps : ℕ → ℕ)
  (h1 : ∀ n, weekly_steps n = (n + 1) * 1000 * 7)
  (h2 : 4 * 7 * 1000 + 3 * 7 * 1000 + 2 * 7 * 1000 + 1 * 7 * 1000 = 70000) 
  (h3 : 70000 + 30000 = 100000) :
  ∃ total_steps, total_steps = 100000 := 
by
  sorry

end cody_needs_total_steps_l2232_223266


namespace find_first_term_l2232_223282

def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem find_first_term (a r : ℝ) (h1 : r = 2/3) (h2 : geom_seq a r 3 = 18) (h3 : geom_seq a r 4 = 12) : a = 40.5 := 
by sorry

end find_first_term_l2232_223282


namespace initial_black_pens_correct_l2232_223212

-- Define the conditions
def initial_blue_pens : ℕ := 9
def removed_blue_pens : ℕ := 4
def remaining_blue_pens : ℕ := initial_blue_pens - removed_blue_pens

def initial_red_pens : ℕ := 6
def removed_red_pens : ℕ := 0
def remaining_red_pens : ℕ := initial_red_pens - removed_red_pens

def total_remaining_pens : ℕ := 25
def removed_black_pens : ℕ := 7

-- Assume B is the initial number of black pens
def B : ℕ := 21

-- Prove the initial number of black pens condition
theorem initial_black_pens_correct : 
  (initial_blue_pens + B + initial_red_pens) - (removed_blue_pens + removed_black_pens) = total_remaining_pens :=
by 
  have h1 : initial_blue_pens - removed_blue_pens = remaining_blue_pens := rfl
  have h2 : initial_red_pens - removed_red_pens = remaining_red_pens := rfl
  have h3 : remaining_blue_pens + (B - removed_black_pens) + remaining_red_pens = total_remaining_pens := sorry
  exact h3

end initial_black_pens_correct_l2232_223212


namespace tangent_length_external_tangent_length_internal_l2232_223248

noncomputable def tangent_length_ext (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R + r) / R)

noncomputable def tangent_length_int (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R - r) / R)

theorem tangent_length_external (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_ext R r a h hAB :=
sorry

theorem tangent_length_internal (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_int R r a h hAB :=
sorry

end tangent_length_external_tangent_length_internal_l2232_223248


namespace trajectory_midpoint_eq_C2_length_CD_l2232_223252

theorem trajectory_midpoint_eq_C2 {x y x' y' : ℝ} :
  (x' - 0)^2 + (y' - 4)^2 = 16 →
  x = (x' + 4) / 2 →
  y = y' / 2 →
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  sorry

theorem length_CD {x y Cx Cy Dx Dy : ℝ} :
  ((x - 2)^2 + (y - 2)^2 = 4) →
  (x^2 + (y - 4)^2 = 16) →
  ((Cx - Dx)^2 + (Cy - Dy)^2 = 14) :=
by
  sorry

end trajectory_midpoint_eq_C2_length_CD_l2232_223252


namespace square_side_length_l2232_223201

theorem square_side_length
  (P : ℕ) (A : ℕ) (s : ℕ)
  (h1 : P = 44)
  (h2 : A = 121)
  (h3 : P = 4 * s)
  (h4 : A = s * s) :
  s = 11 :=
sorry

end square_side_length_l2232_223201
