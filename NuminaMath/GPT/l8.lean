import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Combinatorial.Binom
import Mathlib.Algebra.Factorial.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Special_Functions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Polynomial.Division
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Real

namespace sum_of_youngest_and_oldest_cousins_l8_8094

theorem sum_of_youngest_and_oldest_cousins :
  ∃ (ages : Fin 5 → ℝ), (∃ (a1 a5 : ℝ), ages 0 = a1 ∧ ages 4 = a5 ∧ a1 + a5 = 29) ∧
                        (∃ (median : ℝ), median = ages 2 ∧ median = 7) ∧
                        (∃ (mean : ℝ), mean = (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 ∧ mean = 10) :=
by sorry

end sum_of_youngest_and_oldest_cousins_l8_8094


namespace percents_multiplication_l8_8794

theorem percents_multiplication :
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  (p1 * p2 * p3 * p4) * 100 = 5.88 := 
by
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  sorry

end percents_multiplication_l8_8794


namespace evaluate_integral_l8_8139

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, (2 * x + sqrt (1 - x^2))

theorem evaluate_integral :
  integral_value = 1 + (π / 4) :=
sorry

end evaluate_integral_l8_8139


namespace basketball_team_lineup_l8_8060

-- Definitions as per conditions
def total_players : ℕ := 16
def quadruplets : set ℕ := {Alex, Bella, Chris, Dana}  -- Assume Alex, Bella, Chris, Dana are distinct elements of ℕ

-- Defining a valid lineup function
noncomputable def valid_lineups (total_players : ℕ) (quadruplets : set ℕ) : ℕ :=
  let unrestricted_selections := nat.choose total_players 7 in
  let invalid_selections := 
    (nat.choose 4 3 * nat.choose (total_players - 4) 4) +  -- choosing 3 quadruplets
    (nat.choose 4 4 * nat.choose (total_players - 4) 3)  -- choosing 4 quadruplets
  in
  unrestricted_selections - invalid_selections

-- Theorem to prove
theorem basketball_team_lineup : valid_lineups total_players quadruplets = 9240 := by
  sorry

end basketball_team_lineup_l8_8060


namespace quiz_answer_key_count_l8_8416

theorem quiz_answer_key_count :
  let tf_combinations := 6 -- Combinations of true-false questions
  let mc_combinations := 4 ^ 3 -- Combinations of multiple-choice questions
  tf_combinations * mc_combinations = 384 := by
  -- The values and conditions are directly taken from the problem statement.
  let tf_combinations := 6
  let mc_combinations := 4 ^ 3
  sorry

end quiz_answer_key_count_l8_8416


namespace probability_of_intersection_independent_events_l8_8554

theorem probability_of_intersection_independent_events
  (A B : Type) [DecidablePred A] [DecidablePred B]
  (P : Set (A ∩ B) → ℝ)
  (hA : P A = 1/3)
  (hB : P B = 3/7)
  (h_independent : ∀ x ∈ A, ∀ y ∈ B, x ≠ y → P (A ∩ B) = P A * P B) :
  P (A ∩ B) = 1/7 :=
by
  sorry

end probability_of_intersection_independent_events_l8_8554


namespace bombardment_percentage_l8_8316

-- Given conditions
variables (P D : ℕ)
variables (H1 : P = 6249)
variables (H2 : 0.8 * (P - D) = 4500)

-- Define the percentage of people who died
def percentage_died : ℚ := (D : ℚ) / (P : ℚ) * 100

-- The theorem to prove
theorem bombardment_percentage (H1 : P = 6249) (H2 : 0.8 * (P - D) = 4500) : 
  percentage_died P D ≈ 9.99 :=
by
  sorry

end bombardment_percentage_l8_8316


namespace total_journey_distance_l8_8216

/-- Given a journey with specified travel speeds and breaks, the total journey distance is found. -/
theorem total_journey_distance (D : ℝ) :
  let T1 := (D / 3) / 18
  let T2 := (D / 3) / 24
  let T3 := (D / 3) / 30
  let Breaks := 1 -- 1 hour in breaks (2 * 30 minutes)
  T1 + T2 + T3 + Breaks = 12 →
  D ≈ 253 :=
by sorry

end total_journey_distance_l8_8216


namespace scarves_per_box_l8_8469

variable (B : Nat) -- number of boxes
variable (M : Nat) -- number of mittens per box
variable (T : Nat) -- total number of pieces of winter clothing
variable (S : Nat) -- number of scarves per box

theorem scarves_per_box (hB : B = 4) (hM : M = 6) (hT : T = 32) : S = 2 :=
  -- Define total mittens
  let total_mittens := B * M
  -- Define total scarves
  let total_scarves := T - total_mittens
  -- Define scarves per box
  let scarves_per_box := total_scarves / B
  show S = scarves_per_box by
  simp [hB, hM, hT]
  sorry

end scarves_per_box_l8_8469


namespace largest_m_divides_30_fact_l8_8515

theorem largest_m_divides_30_fact : 
  let pow2_in_fact := 15 + 7 + 3 + 1,
      pow3_in_fact := 10 + 3 + 1,
      max_m_from_2 := pow2_in_fact,
      max_m_from_3 := pow3_in_fact / 2
  in max_m_from_2 >= 7 ∧ max_m_from_3 >= 7 → 7 = 7 :=
by
  sorry

end largest_m_divides_30_fact_l8_8515


namespace triangle_angles_l8_8609

theorem triangle_angles
  (h_a a h_b b : ℝ)
  (h_a_ge_a : h_a ≥ a)
  (h_b_ge_b : h_b ≥ b)
  (a_ge_h_b : a ≥ h_b)
  (b_ge_h_a : b ≥ h_a) : 
  a = b ∧ 
  (a = h_a ∧ b = h_b) → 
  ∃ A B C : ℝ, Set.toFinset ({A, B, C} : Set ℝ) = {90, 45, 45} := 
by 
  sorry

end triangle_angles_l8_8609


namespace choice_related_to_gender_choose_route_A_l8_8740

-- Definitions for the first problem
def total_tourists := 300
def good_route_A := 50
def average_route_A := 75
def total_route_A := 125
def good_route_B := 75
def average_route_B := 100
def total_route_B := 175
def gender_table : List (Nat × Nat × Nat) := [(30, 90, 120), (120, 60, 180)]

def K_squared (n a b c d : ℕ) : ℝ := n * ((a * d - b * c) ^ 2) / (a + b) / (c + d) / (a + c) / (b + d)

def k2_calculated : ℝ := K_squared total_tourists 30 90 120 60

-- Lean theorem for Problem (1)
theorem choice_related_to_gender : k2_calculated > 10.828 := by
  sorry

-- Definitions for the second problem
def P_good_A := 2 / 5 : ℝ
def P_good_B := 3 / 7 : ℝ
def points_good := 5
def points_average := 2

def expected_score (P_good : ℝ) : ℝ :=
  3 * (P_good * points_good + (1 - P_good) * points_average)

def E_A : ℝ := expected_score P_good_A
def E_B : ℝ := expected_score P_good_B

-- Lean theorem for Problem (2)
theorem choose_route_A : E_A > E_B := by
  sorry

end choice_related_to_gender_choose_route_A_l8_8740


namespace kim_average_round_correct_answers_l8_8229

theorem kim_average_round_correct_answers (x : ℕ) :
  (6 * 2) + (x * 3) + (4 * 5) = 38 → x = 2 :=
by
  intros h
  sorry

end kim_average_round_correct_answers_l8_8229


namespace jesse_bananas_l8_8639

def number_of_bananas_shared (friends : ℕ) (bananas_per_friend : ℕ) : ℕ :=
  friends * bananas_per_friend

theorem jesse_bananas :
  number_of_bananas_shared 3 7 = 21 :=
by
  sorry

end jesse_bananas_l8_8639


namespace company_sales_difference_l8_8748

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l8_8748


namespace number_of_students_studying_numeric_methods_l8_8618

theorem number_of_students_studying_numeric_methods
  (ac : ℕ)
  (both : ℕ)
  (fraction_second_year : ℝ)
  (total_students : ℕ)
  (number_second_year : ℕ)
  (N : ℕ) :
  ac = 423 →
  both = 134 →
  fraction_second_year = 0.80 →
  total_students = 653 →
  number_second_year = (fraction_second_year * total_students).round →
  number_second_year = 522 →
  N = (number_second_year - ac + both) →
  N = 233 :=
by
  intros h_ac h_both h_fraction h_total h_round h_total_second_year h_eq
  sorry

end number_of_students_studying_numeric_methods_l8_8618


namespace ab_over_ef_is_three_l8_8462

-- Definitions and conditions:
variables {A B C D E F G I : Type}
variables (AB EF : ℝ)
axiom square_ABCD : quadrilateral A B C D
axiom square_CEFG : quadrilateral C E F G
axiom area_ratio : S_triangle A B I / S_triangle E F I = 27

-- The theorem statement:
theorem ab_over_ef_is_three (h1 : quadrilateral square_ABCD) (h2 : quadrilateral square_CEFG) (h3 : S_triangle A B I / S_triangle E F I = 27) :
  AB / EF = 3 :=
sorry

end ab_over_ef_is_three_l8_8462


namespace common_solution_y_l8_8134

theorem common_solution_y (x y : ℝ) :
  x^2 + y^2 - 4 = 0 ∧ x^2 - 4y + 8 = 0 → y = 2 :=
by
  sorry

end common_solution_y_l8_8134


namespace sin_double_angle_l8_8942

theorem sin_double_angle (α : ℝ) (h1 : sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  sin (2 * α) = - (4 * real.sqrt 2) / 9 := 
by 
  sorry

end sin_double_angle_l8_8942


namespace jack_needs_more_money_l8_8629

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l8_8629


namespace correct_proposition_is_D_l8_8824

variables {m n : Line} {α β : Plane}

-- Define the conditions as hypotheses.
def propositionA (m_parallel_alpha : m ∥ α) (n_parallel_alpha : n ∥ α) : Prop :=
  m ∥ n

def propositionB (m_in_alpha : m ⊆ α) (n_in_alpha : n ⊆ α) 
                 (m_parallel_beta : m ∥ β) (n_parallel_beta : n ∥ β) : Prop :=
  α ∥ β

def propositionC (alpha_perp_beta : α ⊥ β) (m_in_alpha : m ⊆ α) : Prop :=
  m ⊥ β

def propositionD (alpha_perp_beta : α ⊥ β) (m_perp_beta : m ⊥ β) (m_not_in_alpha : m ⊬ α) : Prop :=
  m ∥ α

theorem correct_proposition_is_D (alpha_perp_beta : α ⊥ β) (m_perp_beta : m ⊥ β) (m_not_in_alpha : m ⊬ α) : 
  propositionD alpha_perp_beta m_perp_beta m_not_in_alpha :=
sorry

end correct_proposition_is_D_l8_8824


namespace density_increase_l8_8902

variables (m V₁ V₂ ρ₁ ρ₂ : ℝ)
hypothesis mass_constant : m > 0
hypothesis volume_relation : V₂ = 0.8 * V₁
definition density_before := ρ₁ = m / V₁
definition density_after := ρ₂ = m / V₂

theorem density_increase :
  ρ₂ = 1.25 * ρ₁ :=
by {
  sorry
}

end density_increase_l8_8902


namespace length_AB_is_8_l8_8964

def find_length_AB (a b c : ℝ) (h : a ≠ 0) (A B : ℝ × ℝ) (hA : A = (-2, 0)) (h_symmetry : ∃ x : ℝ, x = 2) : ℝ :=
  let B := (6, 0) in
  abs (fst B - fst A)

theorem length_AB_is_8 (a b c : ℝ) (h : a ≠ 0) :
  find_length_AB a b c h (-2, 0) 6 (by simp) = 8 :=
by
  unfold find_length_AB
  sorry

end length_AB_is_8_l8_8964


namespace evaluate_expr_equiv_l8_8489

theorem evaluate_expr_equiv : 64^(-1 / 3) + 81^(-2 / 4) = 13 / 36 := by
  have h1 : 64 = 2^6 := rfl
  have h2 : 81 = 3^4 := rfl
  sorry

end evaluate_expr_equiv_l8_8489


namespace max_points_of_intersection_l8_8917

theorem max_points_of_intersection : 
  (nat.choose 15 2) * (nat.choose 6 2) = 1575 := 
by 
  sorry

end max_points_of_intersection_l8_8917


namespace polynomial_sum_l8_8658

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8658


namespace joan_football_games_l8_8256

theorem joan_football_games (games_this_year games_last_year total_games: ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_last_year = 9)
  (h3 : total_games = games_this_year + games_last_year) :
  total_games = 13 := 
by
  sorry

end joan_football_games_l8_8256


namespace last_passenger_probability_last_passenger_probability_l8_8337

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l8_8337


namespace even_function_value_l8_8983

theorem even_function_value (a b : ℝ) (h : ∀ x : ℝ, f x = ax^2 + bx + 3a + b)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_domain : 2 * a + (a - 1) = 0) : a + b = 1 / 3 := by
  sorry

end even_function_value_l8_8983


namespace hex_to_decimal_B4E_l8_8126

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end hex_to_decimal_B4E_l8_8126


namespace cosine_of_angle_between_planes_l8_8558

def n1 : ℝ × ℝ × ℝ := (1, 2, 3)
def n2 : ℝ × ℝ × ℝ := (-1, 0, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def cosine_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cosine_of_angle_between_planes :
  cosine_angle n1 n2 = 5 * Real.sqrt 70 / 70 :=
by
  sorry

end cosine_of_angle_between_planes_l8_8558


namespace find_multiple_l8_8453

variables (total_questions correct_answers score : ℕ)
variable (m : ℕ)
variable (incorrect_answers : ℕ := total_questions - correct_answers)

-- Given conditions
axiom total_questions_eq : total_questions = 100
axiom correct_answers_eq : correct_answers = 92
axiom score_eq : score = 76

-- Define the scoring method
def score_formula : ℕ := correct_answers - m * incorrect_answers

-- Statement to prove
theorem find_multiple : score = 76 → correct_answers = 92 → total_questions = 100 → score_formula total_questions correct_answers m = score → m = 2 := by
  intros h1 h2 h3 h4
  sorry

end find_multiple_l8_8453


namespace num_perfect_squares_diff_consecutive_under_20000_l8_8905

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l8_8905


namespace last_passenger_probability_l8_8332

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l8_8332


namespace brendan_total_wins_l8_8108

theorem brendan_total_wins : 
  let round1_matches := 6
  let round2_matches := 6
  let last_round_matches := 4
  let first_two_rounds_wins := round1_matches + round2_matches
  let last_round_wins := last_round_matches / 2
  let total_wins := first_two_rounds_wins + last_round_wins
  total_wins = 14 := by
  -- Definitions and assumptions
  have h1 : first_two_rounds_wins = 6 + 6 := by rw [round1_matches, round2_matches]
  have h2 : last_round_wins = 4 / 2 := by rw [last_round_matches]
  -- Prove the total number of wins
  have h3 : first_two_rounds_wins = 12 := by
    rw h1
    norm_num
  have h4 : last_round_wins = 2 := by
    rw h2
    norm_num
  -- Sum up to get the total wins
  have h5 : total_wins = first_two_rounds_wins + last_round_wins := by rw [first_two_rounds_wins, last_round_wins]
  have h6 : total_wins = 12 + 2 := by
    rw [h3, h4]
  norm_num
  exact h6

end brendan_total_wins_l8_8108


namespace last_passenger_sits_in_assigned_seat_l8_8341

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l8_8341


namespace find_n_l8_8050

-- Definitions based on conditions
def fixed_cost : ℝ := 12000
def marginal_cost : ℝ := 200
def total_cost : ℝ := 16000

-- Theorem statement
theorem find_n (n : ℝ) (H : total_cost - fixed_cost = 4000) : 
  n = (total_cost - fixed_cost) / marginal_cost → n = 20 :=
by
  intro H1
  rw [H] at H1
  rw [div_eq_iff_mul_eq] at H1
  norm_num at H1
  exact H1
  all_goals {norm_num, linarith}

end find_n_l8_8050


namespace average_age_of_10_students_l8_8712

theorem average_age_of_10_students
  (avg_age_25_students : ℕ)
  (num_students_25 : ℕ)
  (avg_age_14_students : ℕ)
  (num_students_14 : ℕ)
  (age_25th_student : ℕ)
  (avg_age_10_students : ℕ)
  (h_avg_age_25 : avg_age_25_students = 25)
  (h_num_students_25 : num_students_25 = 25)
  (h_avg_age_14 : avg_age_14_students = 28)
  (h_num_students_14 : num_students_14 = 14)
  (h_age_25th : age_25th_student = 13)
  : avg_age_10_students = 22 :=
by
  sorry

end average_age_of_10_students_l8_8712


namespace SimplifyAndRationalize_l8_8697

theorem SimplifyAndRationalize :
  ( (√3 / √7) * (√5 / √8) * (√6 / √9) ) = ( √35 / 42 ) :=
sorry

end SimplifyAndRationalize_l8_8697


namespace cos_330_eq_sqrt3_over_2_l8_8481

theorem cos_330_eq_sqrt3_over_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_over_2_l8_8481


namespace last_passenger_sits_in_assigned_seat_l8_8340

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l8_8340


namespace seating_arrangements_count_l8_8862

theorem seating_arrangements_count : 
  (∀ (p : Fin 6 → Fin 6 → Bool), 
    (∀ i, (p i (i + 1)) = tt → (p i i.succ.succ) = tt)
    → (∀ j, (p j (j + 2)) = tt → (p j j.succ.succ.succ) = tt) 
  )
  → (Alice ≠ Bob) 
  ∧ (Alice ≠ Carla)
  ∧ (Derek ≠ Eric)
  ∧ (Derek ≠ Frank)
  → (num_of_seating_arrangements(6) = 1800) := 
sorry

end seating_arrangements_count_l8_8862


namespace circle_intersect_y_axis_l8_8351

noncomputable def c (c : ℝ) : Prop :=
  ∃ (P : Point), 
  (∀ (A B : Point), 
    (A.y * 2 + c) = 0 ∧ 
    (B.y * 2 + c) = 0 ∧
    (A.x = 0) ∧
    (B.x = 0) ∧
    (P.x = 2) ∧
    (P.y = -1) ∧
    angle A P B = 90) → 
    (c = -3)

theorem circle_intersect_y_axis (c : ℝ) : c c 
:= sorry

end circle_intersect_y_axis_l8_8351


namespace area_of_region_l8_8388

theorem area_of_region (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y = 5 → 
  ∃ A : ℝ, A = 10 * Real.pi := 
begin
  sorry
end

end area_of_region_l8_8388


namespace range_mu_l8_8197

-- Defining the parabola, vector equations, and range conditions
variables {p : ℝ} (hp : p > 0) {λ μ : ℝ}

-- Define points A, B, P with coordinates
variables (x1 y1 x2 y2 y0 : ℝ)
  
-- Assume the vector relationships and ratio bound
axiom vector_eq1 : (x1 - p / 2, y1) = (λ * -x1, λ * (y0 - y1))
axiom vector_eq2 : (p / 2 - x2, -y2) = (μ * (x1 - p / 2), μ * y1)
axiom ratio_bound : λ / μ ∈ set.Icc (1 / 4) (1 / 2)
  
open set
  
-- Proving the range of μ
theorem range_mu : μ ∈ Icc (4 / 3) 2 := 
sorry

end range_mu_l8_8197


namespace log_base_10_of_2_bounds_l8_8385

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) →
  (10^5 = 100000) →
  (2^15 = 32768) →
  (2^16 = 65536) →
  (1 / 5 < log 2 / log 10 ∧ log 2 / log 10 < 5 / 16) := 
by
  intros h1 h2 h3 h4
  sorry

end log_base_10_of_2_bounds_l8_8385


namespace max_value_of_b_minus_a_l8_8171

theorem max_value_of_b_minus_a (a b : ℝ) (h₀ : a < 0)
  (h₁ : ∀ x : ℝ, a < x ∧ x < b → (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) :
  b - a ≤ 2017 :=
sorry

end max_value_of_b_minus_a_l8_8171


namespace preimage_exists_l8_8571

-- Define the mapping function f
def f (x y : ℚ) : ℚ × ℚ :=
  (x + 2 * y, 2 * x - y)

-- Define the statement
theorem preimage_exists (x y : ℚ) :
  f x y = (3, 1) → (x, y) = (-1/3, 5/3) :=
by
  sorry

end preimage_exists_l8_8571


namespace simplify_and_rationalize_l8_8695

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 :=
by sorry

end simplify_and_rationalize_l8_8695


namespace eval_modulus_complex_expr_l8_8916

-- The expression to be evaluated
def complex_expr := 3 - 5 * complex.i + (-2 + (3 / 4) * complex.i)

-- The given modulus to be verified
def modulus := real.sqrt 305 / 4

-- The main theorem stating the equality that needs to be proved
theorem eval_modulus_complex_expr : complex.abs complex_expr = modulus := by
  sorry

end eval_modulus_complex_expr_l8_8916


namespace problem_statement_l8_8528

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2 ^ x) / 2 - 2 / (2 ^ x) - x + 1

theorem problem_statement (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) : g x₁ + g x₂ > 0 :=
sorry

end problem_statement_l8_8528


namespace probability_of_forming_triangle_l8_8230

variables (a b c d : ℕ)
variables (sticks : finset ℕ)
variables (triples : finset (ℕ × ℕ × ℕ))

-- Define the stick lengths
def stick_lengths : finset ℕ := {2, 5, 6, 10}

-- Define the property that checks if three sticks can form a triangle
def forms_triangle (x y z: ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Define the set of all possible triples from four lengths
def all_triples : finset (ℕ × ℕ × ℕ) := 
  stick_lengths.image (λ (t : finset ℕ), t.to_list.combinations 3).bind finset.of_list

-- Count how many triples can form a triangle
def triangle_count : ℕ :=
  finset.card (all_triples.filter (λ (t : ℕ × ℕ × ℕ), forms_triangle t.1 t.2 t.3))

-- Total number of triples
def total_triples : ℕ := finset.card all_triples

-- The main statement to prove: the probability of forming a triangle is 1/2
theorem probability_of_forming_triangle :
  (triangle_count stick_lengths all_triples forms_triangle : ℚ) / total_triples all_triples = 1 / 2 :=
sorry

end probability_of_forming_triangle_l8_8230


namespace simplify_expression_l8_8266

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem simplify_expression (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = a + b) :
  (a / b) + (b / a) - (1 / (a * b)) = 1 :=
by sorry

end simplify_expression_l8_8266


namespace find_least_n_l8_8921

-- Define a telescoping_sum that calculates the sum from 30 to 88
noncomputable def telescoping_sum : ℝ :=
  ∑ k in finset.range 59, (1 / (Real.sin (30 + k) * Real.sin (30 + k + 1))) -- sum over k=30 to k=88

-- Define the condition for cos_89
noncomputable def cos_89 : ℝ := Real.cos 89

-- The least positive integer n such that the equation holds
theorem find_least_n :
  ∃ n : ℕ, n > 0 ∧ telescoping_sum + cos_89 = 1 / Real.sin n ∧ n = 1 :=
by
  -- The proof is skipped
  sorry

end find_least_n_l8_8921


namespace circus_juggling_l8_8355

theorem circus_juggling (jugglers : ℕ) (balls_per_juggler : ℕ) (total_balls : ℕ)
  (h1 : jugglers = 5000)
  (h2 : balls_per_juggler = 12)
  (h3 : total_balls = jugglers * balls_per_juggler) :
  total_balls = 60000 :=
by
  rw [h1, h2] at h3
  exact h3

end circus_juggling_l8_8355


namespace mean_height_calc_l8_8726

/-- Heights of players on the soccer team -/
def heights : List ℕ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

/-- Total number of players -/
def total_players : ℕ := heights.length

/-- Sum of heights of players -/
def sum_heights : ℕ := heights.sum

/-- Mean height of players on the soccer team -/
def mean_height : ℚ := sum_heights / total_players

/-- Proof that the mean height is correct -/
theorem mean_height_calc : mean_height = 56.25 := by
  sorry

end mean_height_calc_l8_8726


namespace area_of_given_region_l8_8111

noncomputable def area_of_region (x y : ℝ) : ℝ :=
  if (x - 2)^2 + (y + 1)^2 ≤ 4 then π * (2^2) else 0

theorem area_of_given_region :
  ∀ (x y : ℝ), (x^2 + y^2 - 4x + 2y = -4) → (area_of_region x y = 4 * π) :=
by
  sorry

end area_of_given_region_l8_8111


namespace dark_tile_fraction_l8_8439

theorem dark_tile_fraction 
  (floor : Type) 
  [∀ (x y : ℤ), floor] 
  (pattern : ℤ → ℤ → bool) 
  (repeated_pattern : ∀ i j, pattern (i + 4) j = pattern i j ∧ pattern i (j + 4) = pattern i j)
  (dark_tile : ∀ i j, i % 4 < 2 → j % 4 < 2 → pattern i j = tt)
  (remaining_light_tile : ∀ i j, (i % 4 >= 2 ∨ j % 4 >= 2) → pattern i j = ff) :
  (∑ i in range 4, ∑ j in range 4, if pattern i j then 1 else 0) / 16 = 1 / 4 :=
by 
  sorry

end dark_tile_fraction_l8_8439


namespace samuel_observes_gabriella_l8_8690

theorem samuel_observes_gabriella :
  ∀ (s g : ℕ), (s = 30) → (g = 22) → 
  (15 = ((2 : ℕ) * (1 * 60) / (s - g))) := 
by
  intros s g hs hg
  rw [hs, hg]
  norm_num
  sorry

end samuel_observes_gabriella_l8_8690


namespace inequality_proof_l8_8306

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≥ b) (h5 : b ≥ c) :
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) :=
by
  sorry

end inequality_proof_l8_8306


namespace terez_pregnant_female_cows_l8_8322

def total_cows := 44
def percent_females := 0.50
def percent_pregnant_females := 0.50

def female_cows := total_cows * percent_females
def pregnant_female_cows := female_cows * percent_pregnant_females

theorem terez_pregnant_female_cows : pregnant_female_cows = 11 := by
  sorry

end terez_pregnant_female_cows_l8_8322


namespace train_crossing_time_l8_8411

-- Define the conditions
def train_length : ℝ := 120
def bridge_length : ℝ := 200
def speed_kmph : ℝ := 36

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * (5 / 18)

-- Define the equivalent proof problem
theorem train_crossing_time :
  let speed_mps := kmph_to_mps speed_kmph in
  let total_distance := train_length + bridge_length in
  let time := total_distance / speed_mps in
  time = 32 :=
by
  -- lean will verify this statement when proof steps are filled in
  sorry

end train_crossing_time_l8_8411


namespace find_k_when_lines_perpendicular_l8_8579

theorem find_k_when_lines_perpendicular (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (3-k) * y + 1 = 0 → ∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0 → -((k-3)/(3-k)) * (k-3) = -1) → 
  k = 2 :=
by
  sorry

end find_k_when_lines_perpendicular_l8_8579


namespace greatest_m_value_l8_8910

theorem greatest_m_value (x y m : ℝ) 
  (h₁: x^2 + y^2 = 1)
  (h₂ : |x^3 - y^3| + |x - y| = m^3) : 
  m ≤ 2^(1/3) :=
sorry

end greatest_m_value_l8_8910


namespace distance_between_points_on_polar_eqn_l8_8614

noncomputable def parametric_line : ℝ → ℝ × ℝ :=
  λ t, (2 * sqrt 3 - sqrt 3 / 2 * t, 1 / 2 * t)

noncomputable def parametric_curve : ℝ → ℝ × ℝ :=
  λ α, (sqrt 3 + sqrt 3 * cos α, sqrt 3 * sin α)

def polar_equation_line (ρ θ : ℝ) : Prop :=
  ρ * cos θ + sqrt 3 * ρ * sin θ = 2 * sqrt 3

def polar_equation_curve (ρ θ : ℝ) : Prop :=
  ρ = 2 * sqrt 3 * cos θ

def on_polar_coords (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * cos θ ∧ y = ρ * sin θ

theorem distance_between_points_on_polar_eqn :
  ∀ (θ : ℝ), 0 < θ ∧ θ < (π / 2) →
  (polar_equation_line 2 θ → polar_equation_curve ρ θ → 
  ∃ (ρ_N : ℝ), on_polar_coords 2 θ (2 * cos θ) (2 * sin θ) ∧
  on_polar_coords ρ_N θ (2 * sqrt 3 * cos θ) (2 * sqrt 3 * sin θ) ∧
  |ρ_N - 2| = 1) :=
sorry

end distance_between_points_on_polar_eqn_l8_8614


namespace largest_multiple_of_8_less_than_100_l8_8765

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8765


namespace tan_alpha_eq_l8_8244

-- Definitions from conditions
variables (a h : ℝ) (n : ℕ) [hn : Fact (n % 2 = 1)] [Fact (n > 0)]
-- α is the angle at vertex A whose tangent we need to prove
def α : ℝ := sorry

-- Statement of the theorem
theorem tan_alpha_eq: tan α = (4 * n * h) / ((n^2 - 1) * a) :=
sorry

end tan_alpha_eq_l8_8244


namespace supercomputer_additions_in_half_hour_l8_8088

theorem supercomputer_additions_in_half_hour :
  let rate := 20000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 36000000 :=
by
  let rate := 20000
  let seconds_in_half_hour := 1800
  calc
    rate * seconds_in_half_hour = 20000 * 1800 : rfl
    ... = 36000000 : sorry

end supercomputer_additions_in_half_hour_l8_8088


namespace distance_between_given_lines_l8_8478

def distance_between_lines (A B c1 c2 : ℝ) : ℝ :=
  abs ((c1 - c2) / (Real.sqrt (A ^ 2 + B ^ 2)))

theorem distance_between_given_lines :
  let line1 := (6 : ℝ, -8 : ℝ, -19 : ℝ)
  let line2 := (3 : ℝ, -4 : ℝ, 0.5 : ℝ)
  let A := 3
  let B := -4
  let c1 := -9.5
  let c2 := 0.5
  distance_between_lines A B c1 c2 = 2 :=
by
  sorry

end distance_between_given_lines_l8_8478


namespace monotonicity_intervals_f_nonpositive_l8_8563

section problem_two
variables {a : ℝ} {x : ℝ}

def f (a x : ℝ) : ℝ := a * Real.log x - x + 1

theorem monotonicity_intervals (h_pos : 0 < x) :
  if a ≤ 0 then 
    ∀ x > 0, f a x < f a (x + 1)
  else 
    (∀ x ∈ Ioo 0 a, f a x < f a (x + 1)) ∧ (∀ x ∈ Ioo a (0 : ℝ), f a x > f a (x + 1)) :=
sorry

theorem f_nonpositive (h : ∀ x > 0, f a x ≤ 0) : a = 1 :=
sorry

end problem_two

end monotonicity_intervals_f_nonpositive_l8_8563


namespace prove_props_l8_8544

def Prop1 (α β γ : Plane) : Prop :=
  (α ≠ β ∧ α ∥ γ ∧ β ∥ γ) → α ∥ β

def Prop2 (α β : Plane) (l : Line) : Prop :=
  (α ≠ β ∧ α ∥ l ∧ β ∥ l) → α ∥ β

def Prop3 (α β γ : Plane) : Prop :=
  (α ≠ β ∧ α ⊥ γ ∧ β ⊥ γ) → α ∥ β

def Prop4 (α β : Plane) (l : Line) : Prop :=
  (α ≠ β ∧ l ⊥ α ∧ l ⊥ β) → α ∥ β

theorem prove_props (α β γ : Plane) (l : Line) :
  Prop1 α β γ ∧ ¬Prop2 α β l ∧ ¬Prop3 α β γ ∧ Prop4 α β l :=
by {
  sorry
}

end prove_props_l8_8544


namespace cost_of_dvd_player_l8_8001

/-- The ratio of the cost of a DVD player to the cost of a movie is 9:2.
    A DVD player costs $63 more than a movie.
    Prove that the cost of the DVD player is $81. -/
theorem cost_of_dvd_player 
(D M : ℝ)
(h1 : D = (9 / 2) * M)
(h2 : D = M + 63) : 
D = 81 := 
sorry

end cost_of_dvd_player_l8_8001


namespace trigonometric_expression_value_l8_8169

variable (θ : ℝ)

-- Conditions
axiom tan_theta_eq_two : Real.tan θ = 2

-- Theorem to prove
theorem trigonometric_expression_value : 
  Real.sin θ * Real.sin θ + 
  Real.sin θ * Real.cos θ - 
  2 * Real.cos θ * Real.cos θ = 4 / 5 := 
by
  sorry

end trigonometric_expression_value_l8_8169


namespace factor_polynomial_l8_8552

open Polynomial

noncomputable def polynomial1 := (2 : ℚ) * X^4 + X^3 - 16 * X^2 + 3 * X + 16 + 3 - 1
noncomputable def polynomial2 := X^2 + X - 6

theorem factor_polynomial : polynomial2 ∣ polynomial1 :=
by
  -- Here we should find the quotient and check it exactly divides, but proof is omitted.
  sorry

end factor_polynomial_l8_8552


namespace laptop_total_selling_price_l8_8442

-- Define the original price of the laptop
def originalPrice : ℝ := 1200

-- Define the discount rate
def discountRate : ℝ := 0.30

-- Define the redemption coupon amount
def coupon : ℝ := 50

-- Define the tax rate
def taxRate : ℝ := 0.15

-- Calculate the discount amount
def discountAmount : ℝ := originalPrice * discountRate

-- Calculate the sale price after discount
def salePrice : ℝ := originalPrice - discountAmount

-- Calculate the new sale price after applying the coupon
def newSalePrice : ℝ := salePrice - coupon

-- Calculate the tax amount
def taxAmount : ℝ := newSalePrice * taxRate

-- Calculate the total selling price after tax
def totalSellingPrice : ℝ := newSalePrice + taxAmount

-- Prove that the total selling price is 908.5 dollars
theorem laptop_total_selling_price : totalSellingPrice = 908.5 := by
  unfold totalSellingPrice newSalePrice taxAmount salePrice discountAmount
  norm_num
  sorry

end laptop_total_selling_price_l8_8442


namespace polynomial_sum_l8_8661

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8661


namespace solution_l8_8502

noncomputable def satisfies_equation (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x * f(x) - y * f(y) = (x - y) * f(x + y)

theorem solution (f : ℝ → ℝ) :
  satisfies_equation f → ∃ (a b : ℝ), ∀ x : ℝ, f(x) = a * x + b :=
sorry

end solution_l8_8502


namespace weight_square_proof_l8_8448

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

def density (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  (side_length^2) * density

theorem weight_square_proof :
  let side_triangle := 4
  let weight_triangle := 18
  let side_square := 6
  let area_triangle := area_of_equilateral_triangle side_triangle
  let area_square := side_square ^ 2
  let density_triangle := density weight_triangle area_triangle
  weight_of_square side_square density_triangle = 93.5 :=
by
  have side_triangle := 4
  have weight_triangle := 18
  have side_square := 6
  have area_triangle := area_of_equilateral_triangle side_triangle
  have area_square := side_square ^ 2
  have density_triangle := density weight_triangle area_triangle
  show weight_of_square side_square density_triangle = 93.5
  sorry

end weight_square_proof_l8_8448


namespace gemstone_necklaces_count_l8_8141

-- Conditions
def num_bead_necklaces : ℕ := 3
def price_per_necklace : ℕ := 7
def total_earnings : ℕ := 70

-- Proof Problem
theorem gemstone_necklaces_count : (total_earnings - num_bead_necklaces * price_per_necklace) / price_per_necklace = 7 := by
  sorry

end gemstone_necklaces_count_l8_8141


namespace books_withdrawn_is_15_l8_8011

-- Define the initial condition
def initial_books : ℕ := 250

-- Define the books taken out on Tuesday
def books_taken_out_tuesday : ℕ := 120

-- Define the books returned on Wednesday
def books_returned_wednesday : ℕ := 35

-- Define the books left in library on Thursday
def books_left_thursday : ℕ := 150

-- Define the problem: Determine the number of books withdrawn on Thursday
def books_withdrawn_thursday : ℕ :=
  (initial_books - books_taken_out_tuesday + books_returned_wednesday) - books_left_thursday

-- The statement we want to prove
theorem books_withdrawn_is_15 : books_withdrawn_thursday = 15 := by sorry

end books_withdrawn_is_15_l8_8011


namespace find_height_of_box_l8_8034

-- Define the conditions as formal variables and statements.
variable (h : ℝ) -- height of the box

-- The dimensions of the base of the box
def box_length : ℝ := 20
def box_width : ℝ := 20

-- The area of the base of the box
def box_base_area := box_length * box_width

-- Volume required to package the collection
def total_volume_needed : ℝ := 2400000

-- Volume of one box
def box_volume := box_base_area * h

-- Number of boxes needed
def num_boxes := total_volume_needed / box_volume

-- Cost per box
def cost_per_box : ℝ := 0.5

-- Total cost for the boxes
def total_cost := num_boxes * cost_per_box

-- The minimum amount the university must spend on boxes
def min_spent : ℝ := 250

-- The proof problem statement
theorem find_height_of_box : total_cost ≤ min_spent ↔ h = 12 := by
  sorry

end find_height_of_box_l8_8034


namespace limit_sequence_l8_8553

-- Definition of the sequence {a_n}
variable {a : ℕ → ℝ}

-- Conditions
axiom seq_positive : ∀ n, a n > 0
axiom seq_condition : ∀ n ∈ ℕ+, (∑ i in Finset.range (n + 1), (a i).sqrt) = n^2 + 3 * n

-- Statement of the problem
theorem limit_sequence (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n ∈ ℕ+, (∑ i in Finset.range (n + 1), sqrt (a i)) = n^2 + 3 * n) :
  (tendsto (λ n, (1 / n^2) * (∑ i in Finset.range (n + 1), (a i) / (i + 2))) at_top (nhds 4)) :=
sorry

end limit_sequence_l8_8553


namespace complex_expression_evaluation_explicit_formula_of_f_l8_8054

open Complex Real

-- Problem (1)

theorem complex_expression_evaluation :
  i^2010 + (sqrt 2 + sqrt 2 * i)^2 - (sqrt 2 / (1 - i))^4 = -1 + 4 * i := by
  sorry

-- Problem (2)

noncomputable def f (x : ℝ) : ℝ := f'(1) * exp (x - 1) - f(0) * x + 1/2 * x^2

theorem explicit_formula_of_f (f : ℝ → ℝ) (h : ∀ x, f x = f' 1 * exp (x - 1) - f 0 * x + 1/2 * x^2) :
  f = λ x, exp x - x + 1/2 * x^2 := by
  sorry

end complex_expression_evaluation_explicit_formula_of_f_l8_8054


namespace machine_value_depletion_years_l8_8444

theorem machine_value_depletion_years (initial_value : ℝ) (depletion_rate : ℝ) (final_value : ℝ) : 
  initial_value = 1100 → depletion_rate = 0.10 → final_value = 891 → 
  ∃ (time : ℝ), time = 2 :=
by
  intros h_initial h_rate h_final
  have h_value := final_value = initial_value * (1 - depletion_rate) ^ time
  sorry

end machine_value_depletion_years_l8_8444


namespace polynomial_sum_l8_8660

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8660


namespace complex_translation_l8_8090

theorem complex_translation (z1 z2 z3 z4 : ℂ) (w : ℂ) 
  (h1 : z1 + w = z2) (h2 : z3 + w = z4) 
  (h3 : z1 = 1 - 3*complex.I) (h4 : z2 = 6 + 2*complex.I) 
  (h5 : z3 = 2 - complex.I) : 
  z4 = 7 + 4*complex.I := 
sorry

end complex_translation_l8_8090


namespace largest_multiple_of_8_less_than_100_l8_8782

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8782


namespace lines_concurrent_or_parallel_l8_8248

noncomputable theory
open_locale classical
open Set

variables 
  (A B C : Point) -- Vertices of the triangle
  (I D G : Point) -- Given points i.e., incenter, A-excircle tangency point, G
  (B1 B2 B3 B4 C1 C2 C3 C4 : Point) -- Constructed points

-- Conditions
variables 
  [triangle : is_triangle A B C]
  [incenter : is_incenter I A B C]
  [excircle_touchpoint : is_touchpoint D]
  [midpoint_arc : midpoint_arc G A B C]
  [parallel_GB1_AB : is_parallel (line_through G B1) (line_through A B)]
  [circumpoint_B1 : on_circumcircle B1 A B C]
  [intersection_B1I : intersection (line_through B1 I) (circumcircle A B C) B2]
  [parallel_B2B3_BC : is_parallel (line_through B2 B3) (line_through B C)]
  [circumpoint_B3 : on_circumcircle B3 A B C]
  [intersection_DB3_GB1 : intersection (line_through D B3) (line_through G B1) B4]
  [parallel_GC1_AC : is_parallel (line_through G C1) (line_through A C)]
  [circumpoint_C1 : on_circumcircle C1 A B C]
  [intersection_C1I : intersection (line_through C1 I) (circumcircle A B C) C2]
  [parallel_C2C3_BC : is_parallel (line_through C2 C3) (line_through B C)]
  [circumpoint_C3 : on_circumcircle C3 A B C]
  [intersection_DC3_GC1 : intersection (line_through D C3) (line_through G C1) C4]

-- Goal
theorem lines_concurrent_or_parallel 
  (A B C I D G B1 B2 B3 B4 C1 C2 C3 C4 : Point)
  [is_triangle A B C]
  [is_incenter I A B C]
  [is_touchpoint D]
  [midpoint_arc G A B C]
  [is_parallel (line_through G B1) (line_through A B)]
  [on_circumcircle B1 A B C]
  [intersection (line_through B1 I) (circumcircle A B C) B2]
  [is_parallel (line_through B2 B3) (line_through B C)]
  [on_circumcircle B3 A B C]
  [intersection (line_through D B3) (line_through G B1) B4]
  [is_parallel (line_through G C1) (line_through A C)]
  [on_circumcircle C1 A B C]
  [intersection (line_through C1 I) (circumcircle A B C) C2]
  [is_parallel (line_through C2 C3) (line_through B C)]
  [on_circumcircle C3 A B C]
  [intersection (line_through D C3) (line_through G C1) C4] :
  concurrent_or_parallel (line_through A G) (line_through B B4) (line_through C C4) := sorry

end lines_concurrent_or_parallel_l8_8248


namespace find_pairs_l8_8919

noncomputable def pairs_satisfying_conditions : list (ℝ × ℝ) :=
[(real.sqrt 975, real.sqrt 980.99), (real.sqrt 975, -real.sqrt 980.99), (-real.sqrt 975, real.sqrt 1043.99), (-real.sqrt 975, -real.sqrt 1043.99)]

theorem find_pairs (x y : ℝ) (hx : y^2 - real.floor x^2 = 19.99) (hy : x^2 + real.floor y^2 = 1999) :
  (x, y) ∈ pairs_satisfying_conditions :=
sorry

end find_pairs_l8_8919


namespace total_watch_time_eq_l8_8258

variable (R : ℝ) (hR : 0 < R ∧ R ≤ 100)

def short_video_time : ℝ := 2
def long_video_time : ℝ := 6 * short_video_time
def daily_video_time : ℝ := 2 * short_video_time + long_video_time
def weekly_video_time : ℝ := 7 * daily_video_time
def total_watch_time (R : ℝ) : ℝ := weekly_video_time * (R / 100)

theorem total_watch_time_eq (R : ℝ) (hR : 0 < R ∧ R ≤ 100) : 
  total_watch_time R = 112 * (R / 100) := 
by
  sorry

end total_watch_time_eq_l8_8258


namespace right_triangles_congruent_l8_8599

theorem right_triangles_congruent (a b c d e f S P : ℝ) 
    (h1 : S = (1 / 2) * a * b) 
    (h2 : c = sqrt (a^2 + b^2))
    (h3 : P = a + b + c)
    (h4 : S = (1 / 2) * d * e) 
    (h5 : f = sqrt (d^2 + e^2))
    (h6 : P = d + e + f) 
    : (a = d ∧ b = e ∧ c = f) ∨ (a = e ∧ b = d ∧ c = f) :=
by
  sorry

end right_triangles_congruent_l8_8599


namespace sum_of_x_when_ap_l8_8954

open List Real

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def mode (l : List ℝ) : ℝ :=
  l.maximalByNameFrequency!.val

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· < ·)
  if (sorted.length % 2 = 1) then
    sorted.get! (sorted.length / 2)
  else
    sorry -- Median calculation for even length lists

theorem sum_of_x_when_ap (L : List ℝ) (h : ∃ x, (mean L, median L, mode L) form_ap_with_common_diff 3) :
  L = [12, 3, 6, 3, 5, 3, x] → 
  let x_values := { x | (mean (12::3::6::3::5::3::x::nil), median (12::3::6::3::5::3::x::nil), mode (12::3::6::3::5::3::x::nil)) form_ap_with_common_diff 3 }
  in x_values.sum = 53 / 13 :=
sorry -- Proof goes here

end sum_of_x_when_ap_l8_8954


namespace analogous_to_tetrahedron_is_triangle_l8_8396

-- Define the objects as types
inductive Object
| Quadrilateral
| Pyramid
| Triangle
| Prism
| Tetrahedron

-- Define the analogous relationship
def analogous (a b : Object) : Prop :=
  (a = Object.Tetrahedron ∧ b = Object.Triangle)
  ∨ (b = Object.Tetrahedron ∧ a = Object.Triangle)

-- The main statement to prove
theorem analogous_to_tetrahedron_is_triangle :
  ∃ (x : Object), analogous Object.Tetrahedron x ∧ x = Object.Triangle :=
by
  sorry

end analogous_to_tetrahedron_is_triangle_l8_8396


namespace tetrahedron_distance_sum_eq_l8_8165

-- Defining the necessary conditions
variables {V K : ℝ}
variables {S_1 S_2 S_3 S_4 H_1 H_2 H_3 H_4 : ℝ}

axiom ratio_eq (i : ℕ) (Si : ℝ) (K : ℝ) : (Si / i = K)
axiom volume_eq : S_1 * H_1 + S_2 * H_2 + S_3 * H_3 + S_4 * H_4 = 3 * V

-- Main theorem stating that the desired result holds under the given conditions
theorem tetrahedron_distance_sum_eq :
  H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4 = 3 * V / K :=
by
have h1 : S_1 = K * 1 := by sorry
have h2 : S_2 = K * 2 := by sorry
have h3 : S_3 = K * 3 := by sorry
have h4 : S_4 = K * 4 := by sorry
have sum_eq : K * (H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4) = 3 * V := by sorry
exact sorry

end tetrahedron_distance_sum_eq_l8_8165


namespace AT_midpoint_PQ_l8_8180

-- Given a triangle ABC
variables (A B C X Y P Q T : Type) 
-- X is on side AB
variable (X_on_AB : X ∈ [A, B])
-- Y is on side AC
variable (Y_on_AC : Y ∈ [A, C])
-- P and Q are on side BC 
variables (P_on_BC Q_on_BC : P ∈ [B, C] ∧ Q ∈ [B, C])
-- AX = AY
variable (AX_eq_AY : AX = AY)
-- BX = BP
variable (BX_eq_BP : BX = BP)
-- CY = CQ
variable (CY_eq_CQ : CY = CQ)
-- XP and YQ intersect at T
variable (XP_YQ_intersect_T : ∃ T, T ∈ [XP ∩ YQ])

-- To prove AT passes through the midpoint of PQ
theorem AT_midpoint_PQ (h : AX_eq_AY ∧ BX_eq_BP ∧ CY_eq_CQ ∧ XP_YQ_intersect_T) : 
  passes_midpoint (AT, midpoint(P,Q)) := 
sorry

end AT_midpoint_PQ_l8_8180


namespace expand_and_simplify_l8_8140

theorem expand_and_simplify (x : ℝ) :
  2 * (x + 3) * (x^2 + 2 * x + 7) = 2 * x^3 + 10 * x^2 + 26 * x + 42 :=
by
  sorry

end expand_and_simplify_l8_8140


namespace lumberjack_trees_chopped_l8_8836

-- Statement of the problem in Lean 4
theorem lumberjack_trees_chopped
  (logs_per_tree : ℕ) 
  (firewood_per_log : ℕ) 
  (total_firewood : ℕ) 
  (logs_per_tree_eq : logs_per_tree = 4) 
  (firewood_per_log_eq : firewood_per_log = 5) 
  (total_firewood_eq : total_firewood = 500)
  : (total_firewood / firewood_per_log) / logs_per_tree = 25 := 
by
  rw [total_firewood_eq, firewood_per_log_eq, logs_per_tree_eq]
  norm_num
  sorry

end lumberjack_trees_chopped_l8_8836


namespace find_angle_between_vectors_l8_8203

theorem find_angle_between_vectors
  (a b : ℝ^n)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 2)
  (h : a ⬝ (a - b) = 0) : 
  ∃ θ : ℝ, θ = 60 ∧ θ ∈ set.Icc 0 180 := sorry

end find_angle_between_vectors_l8_8203


namespace largest_multiple_of_8_less_than_100_l8_8783

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8783


namespace percentage_of_knives_is_40_l8_8468

def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives

def total_knives : ℕ := initial_knives + 10
def total_spoons : ℕ := initial_spoons - 6

def total_silverware : ℕ := total_knives + initial_forks + total_spoons

def percentage_knives : ℚ := (total_knives.to_rat / total_silverware.to_rat) * 100

theorem percentage_of_knives_is_40 :
  percentage_knives = 40 :=
by
  sorry

end percentage_of_knives_is_40_l8_8468


namespace floor_2_7_l8_8362

def floor_function_property (x : ℝ) : ℤ := Int.floor x

theorem floor_2_7 : floor_function_property 2.7 = 2 :=
by
  sorry

end floor_2_7_l8_8362


namespace angle_MDC_half_x_l8_8646

noncomputable def isosceles_triangle (A B C : Type) [EuclideanGeometry] :=
isosceles A B C

theorem angle_MDC_half_x
(A B C M D : Type) [EuclideanGeometry]
(h_isosceles : isosceles A B C)
(h_midpoint_M : midpoint M B C)
(h_symmetric_D : symmetric D M A C)
(x : ℝ) :
  angle A B C = x →
  angle M D C = x / 2 :=
by
  sorry

end angle_MDC_half_x_l8_8646


namespace solve_sqrt_equation_l8_8706

theorem solve_sqrt_equation (x : ℚ) (hx : sqrt (5 * x - 4) + 15 / sqrt (5 * x - 4) = 8) : 
  x = 29 / 5 ∨ x = 13 / 5 := 
by
  sorry

end solve_sqrt_equation_l8_8706


namespace conjugate_of_z_squared_l8_8146

/-- Define the complex number z as 1 + 2i -/
def z : ℂ := 1 + 2 * complex.I

/-- Define the square of z -/
def z_squared : ℂ := z * z

/-- State the problem of finding the conjugate of z squared -/
theorem conjugate_of_z_squared : (conj z_squared) = -3 - 4 * complex.I :=
by sorry

end conjugate_of_z_squared_l8_8146


namespace evaluate_expression_l8_8485

def a := (64 : ℝ) ^ (-1 / 3 : ℝ)
def b := (81 : ℝ) ^ (-1 / 2 : ℝ)
def result := a + b

theorem evaluate_expression : result = (13 / 36 : ℝ) :=
by 
  sorry

end evaluate_expression_l8_8485


namespace arcsin_cos_solution_l8_8314

theorem arcsin_cos_solution (x : ℝ) (h : -π/2 ≤ x/3 ∧ x/3 ≤ π/2) :
  x = 3*π/10 ∨ x = 3*π/8 := 
sorry

end arcsin_cos_solution_l8_8314


namespace trigonometric_simplification_l8_8691

theorem trigonometric_simplification (A : ℝ) (h₀ : sin A ≠ 0) (h₁ : cos A ≠ 0) 
    (h₂ : sin A ^ 2 + cos A ^ 2 = 1) :
  (2 + 2 * (cos A / sin A) - 3 * (1 / sin A)) * (3 + 2 * (sin A / cos A) + (1 / cos A)) = 11 :=
by
  sorry

end trigonometric_simplification_l8_8691


namespace minimum_a_plus_8b_l8_8172

theorem minimum_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : a + 2*b = 4*a*b) : a + 8*b = 9/2 :=
begin
  sorry
end

end minimum_a_plus_8b_l8_8172


namespace solve_last_remaining_prisoner_l8_8622

def last_remaining_prisoner : ℕ → ℕ
| 0        := 0  -- assuming no prisoners if n = 0
| n@(n+1)  := 2 * n - (2^(Nat.log2 n + 1)) + 1 -- this function matches the problem definition

theorem solve_last_remaining_prisoner :
  ∀ (n : ℕ), n > 0 → last_remaining_prisoner n = 2 * n - 2^(Nat.log2 n + 1) + 1 :=
by {
  sorry -- proof required
}

end solve_last_remaining_prisoner_l8_8622


namespace rhombus_area_three_times_diagonals_l8_8371

theorem rhombus_area_three_times_diagonals :
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  (new_d1 * new_d2) / 2 = 108 :=
by
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  have h : (new_d1 * new_d2) / 2 = 108 := sorry
  exact h

end rhombus_area_three_times_diagonals_l8_8371


namespace find_difference_l8_8591

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end find_difference_l8_8591


namespace find_second_number_l8_8816

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 280 ∧
  a = 2 * b ∧
  c = 2 / 3 * a ∧
  d = b + c

theorem find_second_number (a b c d : ℚ) (h : problem a b c d) : b = 52.5 :=
by
  -- Proof will go here.
  sorry

end find_second_number_l8_8816


namespace minimum_value_f_when_a_is_2_range_of_a_for_two_intersections_l8_8981

-- Part (Ⅰ): Minimum value for f(x) when a = 2
theorem minimum_value_f_when_a_is_2 : 
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.exp x - 2 * 2 * x - Real.exp 1 + 1) → 
  ∃ x : ℝ, x = Real.log 4 ∧ f x = 5 - 4 * Real.log 4 - Real.exp 1 :=
by sorry

-- Part (Ⅱ): Range of a for which f(x) intersects y = -a at two distinct points in (0, 1)
theorem range_of_a_for_two_intersections : 
  (∀ f : ℝ → ℝ, 
    (∀ x, f x = Real.exp x - 2 * (λ a : ℝ, a) x - Real.exp 1 + (λ a : ℝ, a) + 1) ∧
    (∀ a, ∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f x1 = -a ∧ f x2 = -a)) →
    ∀ a, (e - 2 < a ∧ a < 1) :=
by sorry

end minimum_value_f_when_a_is_2_range_of_a_for_two_intersections_l8_8981


namespace cylindrical_tank_water_volume_l8_8436

-- Definitions for the problem
def radius : ℝ := 6
def height : ℝ := 7
def depth : ℝ := 3

-- The volume of water in the tank
def water_volume : ℝ := 84 * Real.pi - 63 * Real.sqrt 3

-- The statement that needs to be proved
theorem cylindrical_tank_water_volume :
  volume_of_water_in_tank radius height depth = water_volume :=
sorry

end cylindrical_tank_water_volume_l8_8436


namespace quadratic_vertex_form_and_axis_of_symmetry_l8_8309

theorem quadratic_vertex_form_and_axis_of_symmetry :
  ∀ (x : ℝ) (a b c : ℝ), a = 2 → b = -8 → c = 10 → 
  let y := a * x^2 + b * x + c in
  let h := -b / (2 * a) in
  let k := a * (h^2) + b * h + c in
  (y = a * (x - h)^2 + k) ∧ (h = 2) :=
by
  intros x a b c ha hb hc
  rw [ha, hb, hc]
  let y := a * x^2 + b * x + c
  let h := -b / (2 * a)
  let k := a * (h^2) + b * h + c
  sorry

end quadratic_vertex_form_and_axis_of_symmetry_l8_8309


namespace simplify_fraction_sum_l8_8277

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end simplify_fraction_sum_l8_8277


namespace sum_of_series_l8_8999

theorem sum_of_series :
  (∑ n in Finset.range (2018 - 1) + 1, Int.floor (1 / (n + 2 - Real.sqrt ((n + 2) * (n + 1)) : ℝ))) = 2017 :=
by
  sorry

end sum_of_series_l8_8999


namespace total_annual_donation_l8_8289

-- Defining the conditions provided in the problem
def monthly_donation : ℕ := 1707
def months_in_year : ℕ := 12

-- Stating the theorem that answers the question
theorem total_annual_donation : monthly_donation * months_in_year = 20484 := 
by
  -- The proof is omitted for brevity
  sorry

end total_annual_donation_l8_8289


namespace triangle_area_ratio_l8_8297

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let DE := (s * Real.sqrt 3) / 4
  let CE := DE / Real.sqrt 3
  let CD := s - CE
  let AD := s - CD
  AD / CD

theorem triangle_area_ratio (s : ℝ) (h_pos : 0 < s)
  (ABC_equilateral : true)  -- Assuming the given conditions
  (D_on_AC : D ∈ segment ℝ @A @C)                         -- Point D lies on AC
  (angle_30 : ∠DB C = 30) : 
  area_ratio s = 1 / 3 :=
by sorry

end triangle_area_ratio_l8_8297


namespace history_books_count_l8_8368

-- Definitions based on conditions
def total_books : Nat := 100
def geography_books : Nat := 25
def math_books : Nat := 43

-- Problem statement: proving the number of history books
theorem history_books_count : total_books - geography_books - math_books = 32 := by
  sorry

end history_books_count_l8_8368


namespace solutions_of_cubic_eq_l8_8155

theorem solutions_of_cubic_eq : 
  let z := Complex in
  ∃ (a b c : z), 
    (a = -3) ∧ 
    (b = Complex.mk (3 / 2) (3 * Real.sqrt 3 / 2)) ∧ 
    (c = Complex.mk (3 / 2) (-(3 * Real.sqrt 3 / 2))) ∧
    (∀ x : z, x^3 = -27 ↔ (x = a ∨ x = b ∨ x = c)) :=
by
  sorry

end solutions_of_cubic_eq_l8_8155


namespace quiz_passing_condition_l8_8298

theorem quiz_passing_condition (P Q : Prop) :
  (Q → P) → 
    (¬P → ¬Q) ∧ 
    (¬Q → ¬P) ∧ 
    (P → Q) :=
by sorry

end quiz_passing_condition_l8_8298


namespace coffee_price_decrease_is_37_5_l8_8641

-- Define the initial and new prices
def initial_price_per_packet := 12 / 3
def new_price_per_packet := 10 / 4

-- Define the calculation of the percent decrease
def percent_decrease (initial_price : ℚ) (new_price : ℚ) : ℚ :=
  ((initial_price - new_price) / initial_price) * 100

-- The theorem statement
theorem coffee_price_decrease_is_37_5 :
  percent_decrease initial_price_per_packet new_price_per_packet = 37.5 := by
  sorry

end coffee_price_decrease_is_37_5_l8_8641


namespace last_passenger_probability_l8_8331

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l8_8331


namespace largest_multiple_of_8_less_than_100_l8_8771

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8771


namespace polynomial_sum_l8_8659

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8659


namespace evaluation_l8_8131

def bracket (a b c : ℕ) (hc : c ≠ 0) : ℚ := (a + b) / c

def nested_bracket (x y z : ℚ) (hz : z ≠ 0) : ℚ := (x + y) / z

theorem evaluation :
  bracket 2^4 2^3 2^5 (ne_of_gt (pow_pos (nat.zero_lt_succ 1) 5)) = 3 / 4 →
  bracket 3^2 3 ((3^2) + 1) (ne_of_gt (nat.zero_lt_succ 9)) = 6 / 5 →
  bracket 5^2 5 ((5^2) + 1) (ne_of_gt (nat.zero_lt_succ 24)) = 15 / 13 →
  nested_bracket (3 / 4) (6 / 5) (15 / 13) (ne_of_gt (by norm_num)) = 169 / 100 :=
sorry

end evaluation_l8_8131


namespace cross_product_calculation_l8_8506

def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (-2, 4, 6)
def expected_cross_product : ℝ × ℝ × ℝ := (16, -16, 16)

theorem cross_product_calculation :
  let ⟨a₁, a₂, a₃⟩ := vector_a,
      ⟨b₁, b₂, b₃⟩ := vector_b,
      ⟨c₁, c₂, c₃⟩ := expected_cross_product in
  (a₂ * b₃ - a₃ * b₂,
   a₃ * b₁ - a₁ * b₃,
   a₁ * b₂ - a₂ * b₁) = (c₁, c₂, c₃) :=
by
  sorry

end cross_product_calculation_l8_8506


namespace last_passenger_sits_in_assigned_seat_l8_8328

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l8_8328


namespace percentage_increase_area_l8_8221

-- Definitions for original and new dimensions
def original_length (L : ℝ) : ℝ := L
def original_width (W : ℝ) : ℝ := W
def new_length (L : ℝ) (x : ℝ) : ℝ := L * (1 + x / 100)
def new_width (W : ℝ) (y : ℝ) : ℝ := W * (1 + y / 100)

-- Define the original and new areas
def original_area (L W : ℝ) := L * W
def new_area (L W : ℝ) (x y : ℝ) := (L * (1 + x / 100)) * (W * (1 + y / 100))

-- The theorem to prove the percentage increase in the area
theorem percentage_increase_area (L W x y : ℝ) :
  ((new_area L W x y - original_area L W) / original_area L W) * 100 = x + y + (x * y / 100) :=
by
  sorry

end percentage_increase_area_l8_8221


namespace simplify_expression_l8_8274

variables {R : Type*} [LinearOrderedField R]
variables {a b c x : R}
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem simplify_expression (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ a b c x : R, 
   (h_distinct) →
   ((a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)) →
   ( 
     ( (x + a)^2 / ( (a - b) * (a - c) ) + 
       (x + b)^2 / ( (b - a) * (b - c) ) + 
       (x + c)^2 / ( (c - a) * (c - b) )
     ) = -1
   )
  ) := sorry

end simplify_expression_l8_8274


namespace polynomial_sum_l8_8662

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8662


namespace area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l8_8428

-- 1. Prove that the area enclosed by x = π/2, x = 3π/2, y = 0 and y = cos x is 2
theorem area_enclosed_by_lines_and_curve : 
  ∫ (x : ℝ) in (Real.pi / 2)..(3 * Real.pi / 2), (-Real.cos x) = 2 := sorry

-- 2. Prove that the cylindrical coordinates (sqrt(2), π/4, 1) correspond to Cartesian coordinates (1, 1, 1)
theorem cylindrical_to_cartesian_coordinates :
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  let z := 1
  (r * Real.cos θ, r * Real.sin θ, z) = (1, 1, 1) := sorry

-- 3. Prove that (3 + 2i) / (2 - 3i) - (3 - 2i) / (2 + 3i) = 2i
theorem complex_number_evaluation : 
  ((3 + 2 * Complex.I) / (2 - 3 * Complex.I)) - ((3 - 2 * Complex.I) / (2 + 3 * Complex.I)) = 2 * Complex.I := sorry

-- 4. Prove that the area of triangle AOB with given polar coordinates is 2
theorem area_of_triangle_AOB :
  let A := (2, Real.pi / 6)
  let B := (4, Real.pi / 3)
  let area := 1 / 2 * (2 * 4 * Real.sin (Real.pi / 3 - Real.pi / 6))
  area = 2 := sorry

end area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l8_8428


namespace num_perfect_squares_diff_consecutive_under_20000_l8_8906

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l8_8906


namespace partition_bound_exists_l8_8318

noncomputable def p (n : ℕ) : ℕ := 
  sorry  -- definition of the number of partitions of n

theorem partition_bound_exists : 
  ∃ c > 0, ∀ n : ℕ, p(n) ≥ n ^ (c * log (n)) :=
by 
  sorry

end partition_bound_exists_l8_8318


namespace sin_sum_zero_points_l8_8183

noncomputable def f (x m : ℝ) := 2 * sin (2 * x) + cos (2 * x) - m

theorem sin_sum_zero_points (x₁ x₂ m : ℝ) (hₓ₁_in : x₁ ∈ set.Icc 0 (Real.pi / 2))
  (hₓ₂_in : x₂ ∈ set.Icc 0 (Real.pi / 2))
  (hx₁_zero : f x₁ m = 0)
  (hx₂_zero : f x₂ m = 0) :
  sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 :=
sorry -- proof goes here

end sin_sum_zero_points_l8_8183


namespace obtuse_triangle_has_one_obtuse_angle_equilateral_triangle_angles_l8_8847

theorem obtuse_triangle_has_one_obtuse_angle (T : Type) [triangle T] (h : is_obtuse T) : 
exists! (angle : T), is_obtuse_angle angle := sorry

theorem equilateral_triangle_angles (T : Type) [triangle T] (h : is_equilateral T) : 
forall angle : T, measure_angle angle = 60 := sorry

end obtuse_triangle_has_one_obtuse_angle_equilateral_triangle_angles_l8_8847


namespace permissible_k_values_l8_8366

-- Definitions
variable {k : ℝ} -- ratio V_parallelepiped / V_sphere = k
def r := ℝ -- radius of the sphere

-- Conditions
variables (h1 : k >= 6 / Real.pi) -- condition on k

-- Statement to be proved
theorem permissible_k_values (k : ℝ) (h1 : k >= 6 / Real.pi) :
  ∃ α : ℝ, α = Real.arcsin(6 / (Real.pi * k)) ∧ (0 < α) ∧ (α ≤ Real.pi) :=
sorry

end permissible_k_values_l8_8366


namespace trees_chopped_l8_8840

def pieces_of_firewood_per_log : Nat := 5
def logs_per_tree : Nat := 4
def total_firewood_chopped : Nat := 500

theorem trees_chopped (pieces_of_firewood_per_log = 5) (logs_per_tree = 4)
    (total_firewood_chopped = 500) :
    total_firewood_chopped / pieces_of_firewood_per_log / logs_per_tree = 25 := by
  sorry

end trees_chopped_l8_8840


namespace imo_2007_hktst_1_l8_8807

-- Define the main theorem
theorem imo_2007_hktst_1 (p q r s : ℝ) (h : p^2 + q^2 + r^2 - s^2 + 4 = 0) :
  (3 * p + 2 * q + r - 4 * (|s|)) ≤ -2 * real.sqrt 2 :=
sorry

end imo_2007_hktst_1_l8_8807


namespace square_free_bounded_by_powers_of_two_l8_8644

theorem square_free_bounded_by_powers_of_two (n : ℕ) (a b : ℕ) (ε : ℝ) (h1 : n.factorial = a * b^2) (h2 : ∀ c : ℕ, c^2 ∣ a → c = 1) (h3 : ε > 0) (h4 : ∃ N : ℕ, ∀ m ≥ N, ((m = n) → True)) : 
  2^(ℝ.ofNat (1-ε)*n) < a ∧ a < 2^(ℝ.ofNat (1+ε)*n) :=
by 
  sorry

end square_free_bounded_by_powers_of_two_l8_8644


namespace pizzas_needed_l8_8735

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end pizzas_needed_l8_8735


namespace susan_more_cats_than_bob_after_transfer_l8_8319

-- Definitions and conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def cats_transferred : ℕ := 4

-- Question statement translated to Lean
theorem susan_more_cats_than_bob_after_transfer :
  (susan_initial_cats - cats_transferred) - bob_initial_cats = 14 :=
by
  sorry

end susan_more_cats_than_bob_after_transfer_l8_8319


namespace cannot_form_right_triangle_l8_8801

-- Define that the following lengths cannot form a right triangle
theorem cannot_form_right_triangle : 
  ¬ ( ∃ a b c : ℝ, a = real.sqrt 5 ∧ b = 2 ∧ c = real.sqrt 8 ∧ a^2 + b^2 = c^2 ) :=
by
  -- Write the contradictory statement which shows that the Pythagorean theorem does not hold for these lengths
  intro h
  rcases h with ⟨a, b, c, ha, hb, hc, hpythag⟩
  rw [ha, hb, hc] at hpythag
  simp at hpythag
  -- Real computations
  have : sqrt 8^2 = 8 ∧ sqrt 5^2 = 5 := by norm_num
  rw [this.1, this.2] at hpythag
  linarith

end cannot_form_right_triangle_l8_8801


namespace num_primes_between_60_and_90_l8_8994

theorem num_primes_between_60_and_90 :
  (finset.filter nat.prime (finset.Icc 60 90)).card = 7 :=
by {
  sorry
}

end num_primes_between_60_and_90_l8_8994


namespace exists_unique_plane_through_line_parallel_skew_lines_l8_8751

-- Definitions for points, lines, and planes
variables {Point : Type} {Line Plane : Type}
variables [IncidenceGeometry Point Line Plane]

-- Definitions for skew lines, parallel lines, and planes
def skew (l1 l2 : Line) : Prop :=
  ¬ ∃ P : Point, Incident P l1 ∧ Incident P l2 -- l1 and l2 do not intersect

def parallel (l1 l2 : Line) : Prop :=
  ∀ P Q : Point, Incident P l1 → Incident Q l2 → ∀ R : Point, Incident R l1 → Incident R l2 → P = Q

noncomputable def plane_through_line_parallel (l1 l2 : Line) : Prop :=
  ∃! π : Plane, Incident l1 π ∧ (∃ a : Line, parallel a l2 ∧ Incident a π)

-- Prove that there is exactly one such plane
theorem exists_unique_plane_through_line_parallel_skew_lines (l1 l2 : Line)
  (h_skew : skew l1 l2) :
  plane_through_line_parallel l1 l2 :=
sorry

end exists_unique_plane_through_line_parallel_skew_lines_l8_8751


namespace convert_2458_decimal_to_base_7_l8_8899

theorem convert_2458_decimal_to_base_7 : ∃ (rep : List ℕ), (rep = [1, 0, 1, 1, 1] ∧ 2458 = rep.foldr (λ (d acc), d + 7 * acc) 0) :=
by
  use [1, 0, 1, 1, 1]
  sorry

end convert_2458_decimal_to_base_7_l8_8899


namespace limit_proof_l8_8561

theorem limit_proof (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.log x + 8 * x) :
  filter.tendsto (λ (Δx : ℝ), (f (1 - 2 * Δx) - f 1) / Δx) (nhds 0) (nhds (-20)) :=
by
  sorry

end limit_proof_l8_8561


namespace sqrt_two_between_one_and_two_l8_8009

theorem sqrt_two_between_one_and_two : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := 
by
  -- sorry placeholder
  sorry

end sqrt_two_between_one_and_two_l8_8009


namespace largest_multiple_of_8_less_than_100_l8_8781

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8781


namespace sqrt_x_minus_1_meaningful_l8_8378

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_x_minus_1_meaningful_l8_8378


namespace smallest_C_inequality_l8_8888

theorem smallest_C_inequality (x y : ℝ) : 
  (x^2 * (1 + y) + y^2 * (1 + x) ≤ sqrt((x^4 + 4) * (y^4 + 4)) + 4) :=
by
  sorry

end smallest_C_inequality_l8_8888


namespace length_of_field_l8_8359

theorem length_of_field (width : ℕ) (distance_covered : ℕ) (n : ℕ) (L : ℕ) 
  (h1 : width = 15) 
  (h2 : distance_covered = 540) 
  (h3 : n = 3) 
  (h4 : 2 * (L + width) = perimeter)
  (h5 : n * perimeter = distance_covered) : 
  L = 75 :=
by 
  sorry

end length_of_field_l8_8359


namespace largest_multiple_of_8_less_than_100_l8_8761

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8761


namespace parallel_vectors_m_plus_n_l8_8204

-- Given vectors
def a : (ℝ × ℝ × ℝ) := (-2, 3, -1)
variables (m n : ℝ)
def b : (ℝ × ℝ × ℝ) := (4, m, n)

-- a is parallel to b
def are_parallel (a b : (ℝ × ℝ × ℝ)) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2, k * a.3)

theorem parallel_vectors_m_plus_n (h : are_parallel a b) : m + n = -4 :=
sorry

end parallel_vectors_m_plus_n_l8_8204


namespace determinant_of_matrix4x5_2x3_l8_8882

def matrix4x5_2x3 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 5], ![2, 3]]

theorem determinant_of_matrix4x5_2x3 : matrix4x5_2x3.det = 2 := 
by
  sorry

end determinant_of_matrix4x5_2x3_l8_8882


namespace celine_library_charge_l8_8827

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l8_8827


namespace angle_ACB_is_67_5_l8_8623

-- Definition of given condition
variables {A B C D E F : Point}
variables (x : Real)
variables [Triangle ABC]
variables [Isosceles ABC A B C]
variables [PointOnLineSegment D AB]
variables [PointOnLineSegment E BC] 
variables [Intersection F AE CD]
variables [IsoscelesRightTriangle CFE] (angle_CFE : ∠CFE = 90°)
variables (angle_BAE : ∠BAE = x) (angle_ACD : ∠ACD = x)

-- Final statement of the problem 
theorem angle_ACB_is_67_5:
  ∠ACB = 67.5° :=
sorry

end angle_ACB_is_67_5_l8_8623


namespace last_passenger_probability_last_passenger_probability_l8_8336

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l8_8336


namespace last_passenger_probability_l8_8346

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l8_8346


namespace total_books_proof_l8_8386

def total_books (math_books history_books : ℕ) : ℕ := math_books + history_books

theorem total_books_proof :
    (math_book_cost history_book_cost total_cost math_books : ℕ) 
    (math_book_cost = 4) 
    (history_book_cost = 5) 
    (total_cost = 390) 
    (math_books = 60) 
    (history_books : ℕ) 
    (cost_of_math_books = math_books * math_book_cost) 
    (cost_of_history_books = total_cost - cost_of_math_books) 
    (history_books = cost_of_history_books / history_book_cost) 
    (total_number_of_books = total_books math_books history_books) 
    : total_number_of_books = 90 :=
sorry

end total_books_proof_l8_8386


namespace evaluate_expr_equiv_l8_8491

theorem evaluate_expr_equiv : 64^(-1 / 3) + 81^(-2 / 4) = 13 / 36 := by
  have h1 : 64 = 2^6 := rfl
  have h2 : 81 = 3^4 := rfl
  sorry

end evaluate_expr_equiv_l8_8491


namespace find_original_radius_l8_8249

noncomputable def volume_increase_radius (r : ℝ) : ℝ :=
  5 * π * ((r + 3)^2 - r^2)

noncomputable def volume_increase_height (r : ℝ) : ℝ :=
  π * r^2 * 3

theorem find_original_radius (r : ℝ) (z : ℝ)
  (h_eq : 5 = 5) 
  (radius_condition : volume_increase_radius r = z)
  (height_condition : volume_increase_height r = z) :
  r = 5 + 2 * Real.sqrt 10 :=
by
  sorry

end find_original_radius_l8_8249


namespace find_m_find_z1_conj_z2_l8_8945

open Complex

noncomputable def z1 (m : ℝ) : ℂ := (m^2 : ℂ) + (1/(m+1) * I)
noncomputable def z2 (m : ℝ) : ℂ := ((2*m - 3) : ℂ) + (1/2 * I)

-- Problem statement for part 1
theorem find_m (m : ℝ) (H : (z1 m + z2 m).re = 0) : m = 1 :=
sorry

-- Problem statement for part 2
theorem find_z1_conj_z2
  (m : ℝ)
  (H1 : m = 1)
  : z1 m * conj (z2 m) = -3/4 - I :=
sorry

end find_m_find_z1_conj_z2_l8_8945


namespace total_sacks_needed_l8_8869

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l8_8869


namespace integral_squared_geq_twelve_integral_squared_l8_8642

variable (f : ℝ → ℝ)

noncomputable theory

open Real

def is_derivable_and_continuous (f : ℝ → ℝ) (a b : ℝ) := 
  differentiable_on ℝ f (set.Icc a b) ∧ continuous_on f.deriv (set.Icc a b)

theorem integral_squared_geq_twelve_integral_squared 
  (f : ℝ → ℝ) 
  (hf : is_derivable_and_continuous f 0 1) 
  (hf_mid : f (1/2) = 0) 
  : 
  (∫ x in 0..1, (f.deriv x)^2) ≥ 12 * (∫ x in 0..1, f x)^2 := 
sorry

end integral_squared_geq_twelve_integral_squared_l8_8642


namespace pascals_triangle_ratio_sum_l8_8313

theorem pascals_triangle_ratio_sum:
  (∑ i in Finset.range (2007), (Nat.choose 2006 i) / (Nat.choose 2007 i)) -
  (∑ i in Finset.range (2006), (Nat.choose 2005 i) / (Nat.choose 2006 i)) =
  (1 / 2 : ℚ) :=
by
  sorry

end pascals_triangle_ratio_sum_l8_8313


namespace find_angle_A_find_cos_C_l8_8602

open Real

variable (a b c : ℝ)
variable (A B C : ℝ) -- angles in radians
variable (m n : ℝ × ℝ)
variable (triangle_ABC : (A + B + C = π))
variable (parallel_m_n : m.1 / n.1 = m.2 / n.2)
variable (dot_product_m_n : m.1 * n.1 + m.2 * n.2 = 3 * b * sin B)
variable (lines_parallel : parallel_m_n)
variable (cos_A : cos A = 4 / 5)
variable (vector_m : m = (a, c))
variable (vector_n : n = (cos C, cos A))
variable (relation_c : c = sqrt 3 * a)

-- (1) Proving A given parallel vectors and relation on c
theorem find_angle_A (triangle_ABC) (parallel_m_n) (relation_c) : A = π / 6 :=
  sorry

-- (2) Proving cos C given dot product and cos A
theorem find_cos_C (triangle_ABC) (dot_product_m_n) (cos_A) : cos C = (3 - 8 * sqrt 2) / 15 :=
  sorry

end find_angle_A_find_cos_C_l8_8602


namespace train_crossing_time_l8_8059

noncomputable def length_of_train : ℝ := 480
noncomputable def length_of_platform : ℝ := 620
noncomputable def speed_of_train_kmh : ℝ := 55
noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def expected_time : ℝ := 71.98

theorem train_crossing_time :
  (total_distance / speed_of_train_ms) ≈ expected_time :=
sorry

end train_crossing_time_l8_8059


namespace earnings_difference_l8_8744

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l8_8744


namespace exists_unique_plane_perpendicular_l8_8534

variable (m : Line) (α : Plane)

-- Conditions
axiom m_intersects_α : intersects m α
axiom m_not_perpendicular_α : ¬perpendicular m α

-- Statement: There exists exactly one plane through line m that is perpendicular to plane α
theorem exists_unique_plane_perpendicular (m_intersects_α : intersects m α) (m_not_perpendicular_α : ¬perpendicular m α) :
  ∃! β : Plane, (intersects m β) ∧ (perpendicular β α) :=
sorry

end exists_unique_plane_perpendicular_l8_8534


namespace polar_to_rectangular_eq_l8_8128

def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_eq :
  polar_to_rectangular 8 (Real.pi / 4) = (4 * Real.sqrt 2, 4 * Real.sqrt 2) :=
by
  sorry

end polar_to_rectangular_eq_l8_8128


namespace eccentricity_of_conic_curve_l8_8960

-- Define the geometric mean condition
def is_geometric_mean (a b m : ℝ) : Prop :=
  m^2 = a * b

-- Define the conic curve condition
def is_conic_curve (m : ℝ) : Prop :=
  ∃ e : ℝ, (e = (Math.sqrt 3) / 2 ∨ e = Math.sqrt 5) ∧
  (x^2 + y^2 / m = 1)

-- The main statement, proving the eccentricity condition
theorem eccentricity_of_conic_curve :
  ∀ (m : ℝ), is_geometric_mean 2 8 m → is_conic_curve m :=
begin
  assume m h,
  sorry -- Proof goes here
end

end eccentricity_of_conic_curve_l8_8960


namespace correct_subtraction_l8_8584

theorem correct_subtraction :
  ∃ x : ℤ, (x - 32 = 25) → (x - 23 = 34) :=
begin
  -- Placeholder for proof
  sorry
end

end correct_subtraction_l8_8584


namespace num_times_teams_face_each_other_l8_8010

-- Conditions
variable (teams games total_games : ℕ)
variable (k : ℕ)
variable (h1 : teams = 17)
variable (h2 : games = teams * (teams - 1) * k / 2)
variable (h3 : total_games = 1360)

-- Proof problem
theorem num_times_teams_face_each_other : k = 5 := 
by 
  sorry

end num_times_teams_face_each_other_l8_8010


namespace sum_digits_square_Y_eq_162_l8_8029

theorem sum_digits_square_Y_eq_162 :
  let Y := 222222222
  in sum_of_digits (Y^2) = 162 :=
by
  admit  -- Replace with appropriate proof

end sum_digits_square_Y_eq_162_l8_8029


namespace find_c_for_radius_6_l8_8911

-- Define the circle equation and the radius condition.
theorem find_c_for_radius_6 (c : ℝ) :
  (∃ (x y : ℝ), x^2 + 8 * x + y^2 + 2 * y + c = 0) ∧ 6 = 6 -> c = -19 := 
by
  sorry

end find_c_for_radius_6_l8_8911


namespace sum_lent_l8_8047

theorem sum_lent (R T : ℝ) (P I : ℝ) (h1 : R = 4) (h2 : T = 8) (h3 : I = P - 306) :
    I = (P * R * T) / 100 → P = 450 :=
by
  intros h4
  rw [h1, h2] at h4
  have hI : I = P * 0.32 := by linarith
  rw hI at h3
  linarith [h3]

-- To complete the proof we'll need to solve I = P - 306 = 0.32P

end sum_lent_l8_8047


namespace solve_students_and_apples_l8_8913

noncomputable def students_and_apples : Prop :=
  ∃ (x y : ℕ), y = 4 * x + 3 ∧ 6 * (x - 1) ≤ y ∧ y ≤ 6 * (x - 1) + 2 ∧ x = 4 ∧ y = 19

theorem solve_students_and_apples : students_and_apples :=
  sorry

end solve_students_and_apples_l8_8913


namespace other_root_of_equation_l8_8185

theorem other_root_of_equation (m : ℤ) (h₁ : (2 : ℤ) ∈ {x : ℤ | x ^ 2 - 3 * x - m = 0}) : 
  ∃ x, x ≠ 2 ∧ (x ^ 2 - 3 * x - m = 0) ∧ x = 1 :=
by {
  sorry
}

end other_root_of_equation_l8_8185


namespace roots_reciprocal_l8_8792

theorem roots_reciprocal (a b c x1 x2 x3 x4 : ℝ) 
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (hx3 : c * x3^2 + b * x3 + a = 0)
  (hx4 : c * x4^2 + b * x4 + a = 0) :
  (x3 = 1/x1 ∧ x4 = 1/x2) :=
  sorry

end roots_reciprocal_l8_8792


namespace garden_length_l8_8081

theorem garden_length
  (width : ℝ) (area : ℝ)
  (h_width : width = 5)
  (h_area : area = 60) :
  ∃ length : ℝ, length = 12 :=
by
  -- Variables definition
  let length := area / width
  -- Applying given conditions
  have h_length: length = 60 / 5, by simp [h_area, h_width]
  -- Showing the required length
  have h_final: length = 12, by norm_num [h_length]
  -- Conclusion
  exact ⟨length, h_final⟩

end garden_length_l8_8081


namespace lumberjack_trees_l8_8842

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l8_8842


namespace product_of_midpoint_coordinates_l8_8685

def point (α : Type) := prod α α
def reflect_y (pt : point ℝ) : point ℝ := (-pt.1, pt.2)
def midpoint (p1 p2 : point ℝ) : point ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem product_of_midpoint_coordinates :
  let A := (4, 2) : point ℝ
  let B := (16, 18) : point ℝ
  let N := midpoint A B
  let A' := reflect_y A
  let B' := reflect_y B
  let N' := midpoint A' B'
  N'.1 * N'.2 = -100 := 
begin
  sorry
end

end product_of_midpoint_coordinates_l8_8685


namespace solve_abs_inequality_l8_8315

theorem solve_abs_inequality (x : ℝ) : (abs (x - 1) + abs (x + 2) ≤ 5) ↔ x ∈ set.Icc (-3 : ℝ) (2 : ℝ) := by
  sorry

end solve_abs_inequality_l8_8315


namespace exists_diagonal_not_parallel_to_side_l8_8300

-- Define what it means to be a convex polygon with 2n sides
variables {n : ℕ} (P : Type) [polygon P] -- Assuming a polygon structure is defined

-- Define the main theorem
theorem exists_diagonal_not_parallel_to_side (convex : convex P) (sides_eq : ∃ k, count_sides P = 2 * k) :
  ∃ d, diagonal d P ∧ ¬ ∃ s, side s P ∧ parallel d s :=
by
  -- A placeholder for the actual proof
  sorry

end exists_diagonal_not_parallel_to_side_l8_8300


namespace find_a_find_inverse_function_l8_8667

section proof_problem

variables (f : ℝ → ℝ) (a : ℝ)

-- Condition that f is an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f (x)

-- Condition from the problem
def condition_1 : Prop := (a - 1) * (2 ^ (1:ℝ) + 1) = 0
def function_definition : Prop := f = λ x, 2^x - 1

-- Question 1: Find the value of a
theorem find_a
  (h1 : is_odd_function f)
  (h2 : condition_1 a)
  (h3 : function_definition f) : a = 1 :=
sorry

-- Question 2: Find the inverse function f⁻¹(x)
noncomputable def inverse_function (y : ℝ) : ℝ := real.log (y + 1) / real.log 2

theorem find_inverse_function
  (h3 : function_definition f) : 
  ∀ x, f (inverse_function x) = x :=
sorry

end proof_problem

end find_a_find_inverse_function_l8_8667


namespace second_shift_widget_fraction_l8_8464

theorem second_shift_widget_fraction (x y : ℕ) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3 : ℚ) * x * (4 / 3 : ℚ) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  (second_shift_widgets / total_widgets) = (8 / 17 : ℚ) :=
by
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3 : ℚ) * x * (4 / 3 : ℚ) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  have h1 : second_shift_widgets = (8 / 9 : ℚ) * (x * y) := by 
    simp [second_shift_widgets, mul_assoc]
  have h2 : total_widgets = first_shift_widgets + second_shift_widgets := rfl
  have h3 : first_shift_widgets = x * y := rfl
  have h4 : first_shift_widgets + second_shift_widgets = (9 / 9 : ℚ) * (x * y) + (8 / 9 : ℚ) * (x * y) := by
    simp [h3, h1]
  have h5 : total_widgets = (17 / 9 : ℚ) * (x * y) := by
    simp [h2, h4]
  show second_shift_widgets / total_widgets = (8 / 17 : ℚ) from by
    rw [h1, h5]
    field_simp
    ring

end second_shift_widget_fraction_l8_8464


namespace find_xy_pairs_l8_8920

theorem find_xy_pairs (x y: ℝ) :
  x + y + 4 = (12 * x + 11 * y) / (x ^ 2 + y ^ 2) ∧
  y - x + 3 = (11 * x - 12 * y) / (x ^ 2 + y ^ 2) ↔
  (x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5) :=
by
  sorry

end find_xy_pairs_l8_8920


namespace max_val_and_period_of_f_find_a_and_b_l8_8191

-- (1) Define f(x) and show its maximum value and smallest positive period
noncomputable def f (x : ℝ) := 2 * real.sqrt 3 * real.sin x * real.cos x - 2 * (real.cos x) ^ 2 + 3

-- Prove the maximum value and period
theorem max_val_and_period_of_f :
  (∀ x : ℝ, f x ≤ 4) ∧ (∃ x : ℝ, f x = 4) ∧ (∀ x : ℝ, f (x + real.pi) = f x) :=
by
  sorry

-- (2) Triangle ABC problem
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : c = real.sqrt 3)
variables (h2 : f C = 4)
variables (h3 : real.sin A = 2 * real.sin B)

-- Prove the values of a and b
theorem find_a_and_b :
  a = 2 ∧ b = 1 :=
by
  sorry

end max_val_and_period_of_f_find_a_and_b_l8_8191


namespace sum_of_all_possible_values_of_c_l8_8652

variable (b c : ℝ)
def poly := (x : ℝ) → x^2 + b*x + c

theorem sum_of_all_possible_values_of_c :
  (∀ (x : ℝ), poly b c x = 0 → poly b c x = 0 → (c = 3 + 2*Real.sqrt 2 ∨ c = 3 - 2*Real.sqrt 2)) →
  b = c - 1 →
  ∀ (Δ : ℝ), (Δ = b^2 - 4*c) → Δ > 0 →
  (3 + 2*Real.sqrt 2) + (3 - 2*Real.sqrt 2) = 6 :=
by
  intros hpoly hb hΔ hΔ_pos
  sorry

end sum_of_all_possible_values_of_c_l8_8652


namespace simplify_and_rationalize_l8_8701

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l8_8701


namespace min_dist_PA_on_tangent_l8_8186

theorem min_dist_PA_on_tangent (P A : ℝ × ℝ)
  (hP : 3 * P.1 + 4 * P.2 - 10 = 0)
  (hA : A.1^2 + A.2^2 = 1)
  (tangent : ∃ k : ℝ, A = (P.1 + k * (-4), P.2 + k * 3)) :
  |PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)| = sqrt 3 := 
sorry

end min_dist_PA_on_tangent_l8_8186


namespace sin_alpha_third_quadrant_unit_circle_l8_8188

theorem sin_alpha_third_quadrant_unit_circle :
  (∃ m : ℝ, (α : ℝ) (P : ℝ × ℝ) (origin : ℝ × ℝ),
        P = (- (√ 5 / 5), m)
        ∧ origin = (0, 0)
        ∧ 0 ≤ P.1
        ∧ P.2 < 0
        ∧ P.1 ^ 2 + P.2 ^ 2 = 1) 
  → α.sin = - ((2 * (√ 5)) / 5) :=
by
  sorry

end sin_alpha_third_quadrant_unit_circle_l8_8188


namespace solutions_to_cubic_l8_8156

noncomputable theory

def cubic_polynomial (z : ℂ) : ℂ := z^3 + 27

theorem solutions_to_cubic :
  {z : ℂ | cubic_polynomial z = 0} = {-3, (3/2) + (3 * complex.I * real.sqrt 3)/2, (3/2) - (3 * complex.I * real.sqrt 3)/2} :=
by
  sorry

end solutions_to_cubic_l8_8156


namespace ellipse_major_axis_l8_8969

theorem ellipse_major_axis (h : ∀ (x y : ℝ), 2 * x^2 + 3 * y^2 = 1) : ∃ a : ℝ, 2 * a = sqrt 2 :=
sorry

end ellipse_major_axis_l8_8969


namespace players_odd_sum_probability_l8_8380

noncomputable def tile_probability_odd_sum : ℚ := 1 / 410

theorem players_odd_sum_probability :
  let m := 1 in
  let n := 410 in
  ∑ i in finset.range 1, ∑ j in finset.range 410, (m * 410 + n = 411) :=
begin
  let m := 1,
  let n := 410,
  exact (m + n = 411),
end

end players_odd_sum_probability_l8_8380


namespace exists_sum_power_of_2_l8_8260

open Set

def is_power_of_2 (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 ^ k

theorem exists_sum_power_of_2 (n : ℕ) (A B : Set ℕ) (hA : ∀ a ∈ A, a ≤ n)
    (hB : ∀ b ∈ B, b ≤ n) (h_card : (A ∩ Icc 0 n).card + (B ∩ Icc 0 n).card ≥ n + 2) :
  ∃ a ∈ A, ∃ b ∈ B, is_power_of_2 (a + b) := by
  sorry

end exists_sum_power_of_2_l8_8260


namespace equation_of_curve_C_ratio_of_areas_l8_8577

-- Define the necessary points and curves
structure Point := (x : ℝ) (y : ℝ)

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-2, 1⟩
def B : Point := ⟨2, 1⟩

-- Define curve C with the condition
def on_curve_C (M : Point) :=
  let MA := Point.mk (-2 - M.x) (1 - M.y)
  let MB := Point.mk (2 - M.x) (1 - M.y)
  let OM := M
  ∥MA + MB∥ = M.x * (A.x + B.x) + M.y * (A.y + B.y) + 2

-- Given point Q on curve C
axiom Q_is_on_curve_C (x0 y0 : ℝ) (h : -2 < x0 ∧ x0 < 2) : on_curve_C ⟨x0, y0⟩

-- Prove the equation of the curve
theorem equation_of_curve_C :
  ∀ (M : Point), on_curve_C M → M.x ^ 2 = 4 * M.y := sorry

-- Prove the ratio of the areas
theorem ratio_of_areas (x0 y0 : ℝ) (h : -2 < x0 ∧ x0 < 2) :
  on_curve_C ⟨x0, y0⟩ →
  let D := ⟨(x0 - 2) / 2, 1 - (x0 - 2) / 2 - 1⟩
  let E := ⟨(x0 + 2) / 2, (x0 + 2) / 2 - 1⟩
  let S_QAB := (4 - (x0 ^ 2)) / 2
  let S_PDE := (4 - (x0 ^ 2)) / 4
  S_QAB / S_PDE = 2 := sorry

end equation_of_curve_C_ratio_of_areas_l8_8577


namespace sum_of_multiples_of_integer_contract_l8_8793

theorem sum_of_multiples_of_integer_contract (n : ℤ) 
  (h : ∃ m, m ∈ Icc (63 : ℤ) 151 ∧ m = n * k) 
  (Σ : ℤ) : 
  (Σ = 2316) → false := 
by
  sorry

end sum_of_multiples_of_integer_contract_l8_8793


namespace largest_multiple_of_8_less_than_100_l8_8763

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8763


namespace ants_calculation_l8_8080

noncomputable def ants_in_usable_area (field_width field_length pond_radius ants_per_sq_inch : ℕ) : ℝ :=
  let width_in_inches := field_width * 12
  let length_in_inches := field_length * 12
  let total_area := width_in_inches * length_in_inches
  let radius_in_inches := pond_radius * 12
  let pond_area := Real.pi * radius_in_inches^2
  let usable_area := total_area - pond_area
  (usable_area * ants_per_sq_inch : ℝ)

theorem ants_calculation :
  ants_in_usable_area 200 300 25 5 ≈ 38758855 :=
by
  sorry

end ants_calculation_l8_8080


namespace hyperbola_eccentricity_l8_8098

theorem hyperbola_eccentricity :
  (∃ x y, x^2 / 4 - y^2 / 5 = 1 ∧ ecc(x^2 / 4 - y^2 / 5 = 1) = 3 / 2) := 
sorry

end hyperbola_eccentricity_l8_8098


namespace overall_loss_percentage_l8_8450

def cost_price_radio := 1500
def selling_price_radio := 1200
def cost_price_calculator := 800
def selling_price_calculator := 700
def cost_price_mobile := 8000
def selling_price_mobile := 7500

def loss_on_radio := cost_price_radio - selling_price_radio
def loss_on_calculator := cost_price_calculator - selling_price_calculator
def loss_on_mobile := cost_price_mobile - selling_price_mobile

def total_cost_price := cost_price_radio + cost_price_calculator + cost_price_mobile
def total_selling_price := selling_price_radio + selling_price_calculator + selling_price_mobile
def total_loss := loss_on_radio + loss_on_calculator + loss_on_mobile

def loss_percentage := (total_loss / total_cost_price) * 100

theorem overall_loss_percentage : abs (loss_percentage - 8.74) < 0.01 := by
  sorry

end overall_loss_percentage_l8_8450


namespace angle_around_point_l8_8619

noncomputable def sum_of_angles (a b c : ℝ) : Prop := a + b + c = 360

theorem angle_around_point (GFQ PFH : ℝ) (h_GFQ : GFQ = 90) (h_PFH : PFH = 68) :
  ∃ (HFQ : ℝ), HFQ = 202 ∧ sum_of_angles GFQ PFH HFQ :=
by
  use 360 - GFQ - PFH
  split
  · exact 202
  · sorry

end angle_around_point_l8_8619


namespace coefficient_of_x_squared_in_binomial_expansion_l8_8617

theorem coefficient_of_x_squared_in_binomial_expansion :
  (coeff (expand (1 + 2 * x) 6) 2) = 60 :=
by
  sorry

end coefficient_of_x_squared_in_binomial_expansion_l8_8617


namespace domain_length_l8_8909

noncomputable def g (x : ℝ) : ℝ :=
  log 2 (log 4 (exp (log (1/8) (log 8 x))))

theorem domain_length (p q : ℕ) (hpq : Nat.coprime p q) :
  (∀ x, g x  ∈ ℝ → (1 < x ∧ x < 8)) → p = 7 ∧ q = 1 ∧ p + q = 8 :=
by
  sorry

end domain_length_l8_8909


namespace angle_F_after_decrease_l8_8625

theorem angle_F_after_decrease (D E F : ℝ) (h1 : D = 60) (h2 : E = 60) (h3 : F = 60) (h4 : E = D) :
  F - 20 = 40 := by
  simp [h3]
  sorry

end angle_F_after_decrease_l8_8625


namespace max_sum_of_lengths_l8_8521

def length_of_integer (k : ℤ) (hk : k > 1) : ℤ := sorry

theorem max_sum_of_lengths (x y : ℤ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
  length_of_integer x hx + length_of_integer y hy = 15 :=
sorry

end max_sum_of_lengths_l8_8521


namespace participants_with_3_points_l8_8235

theorem participants_with_3_points (n : ℕ) (h : n > 4) :
  ∃ (C : ℕ → ℕ → ℕ), (∑ k in finset.range (n + 1), C n k = 2^n) →
    (number_of_participants = 2^n + 4) →
    (number_of_3_points_scorers = C n 3 + 1) :=
by sorry

end participants_with_3_points_l8_8235


namespace sum_of_squares_of_roots_eq_213_l8_8159

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l8_8159


namespace mean_of_solutions_l8_8517

open Polynomial Rat

theorem mean_of_solutions (f : ℚ[X]) (h_f : f = X^3 + 5 * X^2 - 14 * X) :
  mean_of_solutions f = -5/3 :=
by
  sorry

noncomputable def mean_of_solutions (f : Polynomial ℚ) : ℚ :=
if h_solutions : (roots f).toList.length > 0 then
  (roots f).toList.sum / (roots f).toList.length
else 0

end mean_of_solutions_l8_8517


namespace exists_minimum_n_l8_8537

noncomputable def sequence (n : ℕ) : ℕ := sorry -- define the sequence function

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 1, 2 * sum_first_n_terms a n = a n * a (n + 1)

def log_sum_exceeds_five (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (Finset.range n).sum (λ i, Real.log2 (1 + 1 / a (i + 1))) > 5

theorem exists_minimum_n (a : ℕ → ℕ) :
  satisfies_condition a → ∃ n : ℕ, log_sum_exceeds_five a n ∧
    ∀ m : ℕ, m < n → ¬log_sum_exceeds_five a m :=
λ h,
  sorry

end exists_minimum_n_l8_8537


namespace total_votes_l8_8240

theorem total_votes (V : ℕ) (h1 : ∃ c : ℕ, c = 84) (h2 : ∃ m : ℕ, m = 476) (h3 : ∃ d : ℕ, d = ((84 * V - 16 * V) / 100)) : 
  V = 700 := 
by 
  sorry 

end total_votes_l8_8240


namespace problem1_problem2_problem3_l8_8562

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := log (2 * x / (a * x + b))

def conditions (a b : ℝ) : Prop := 
  f 1 a b = 0 ∧
  (∀ x > 0, f x a b - f (1 / x) a b = log x)

theorem problem1 (a b : ℝ) (hc : conditions a b) : 
  a = 1 → b = 1 ∧ (∀ x, f x a b = log (2 * x / (x + 1))) :=
sorry

theorem problem2 (a b t : ℝ) (hc : conditions a b) : 
  (∃ x, f x a b = log t) → (0 < t ∧ t < 2) ∨ (2 < t) :=
sorry

theorem problem3 (a b m : ℝ) (hc : conditions a b) : 
  (∀ x, ¬ (f x a b = log (8 * x + m))) → 0 ≤ m ∧ m < 18 :=
sorry

end problem1_problem2_problem3_l8_8562


namespace merchant_markup_percentage_l8_8845

theorem merchant_markup_percentage
  (CP : ℕ) (discount_percent : ℚ) (profit_percent : ℚ)
  (mp : ℚ := CP + x)
  (sp : ℚ := (1 - discount_percent) * mp)
  (final_sp : ℚ := CP * (1 + profit_percent)) :
  discount_percent = 15 / 100 ∧ profit_percent = 19 / 100 ∧ CP = 100 → 
  sp = 85 + 0.85 * x → 
  final_sp = 119 →
  x = 40 :=
by 
  sorry

end merchant_markup_percentage_l8_8845


namespace polygon_sides_eq_eight_l8_8222

theorem polygon_sides_eq_eight (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by 
  sorry

end polygon_sides_eq_eight_l8_8222


namespace laplace_transform_brownian_bridge_l8_8301

noncomputable def laplace_transform_brownian_bridge_range (λ : ℝ) (B_t : ℝ → ℝ) : ℝ :=
  ∫ (x : ℝ) in 0..∞, exp (-λ * x) * (∑ (n : ℤ), (1 - n^2 * real.pi^2 * x) * exp (-n^2 * real.pi^2 * x / 2))

theorem laplace_transform_brownian_bridge (λ : ℝ) (hλ : 0 < λ)
  (B_t : ℝ → ℝ)
  (hR : ∀ t, 0 ≤ t ∧ t ≤ 1 → B_t t = B_t (1 - t))
  (hP : ∀ x ≥ 0, 
    ∑ (n : ℤ), (1 - n^2 * real.pi^2 * x) * exp (-n^2 * real.pi^2 * x / 2) =
    ∑ (n : ℤ), if n = 0 then 1 else (1 - n^2 * real.pi^2 * x) * exp (-n^2 * real.pi^2 * x / 2)) :
  laplace_transform_brownian_bridge_range λ B_t =
  (sqrt (2 * λ) / real.sinh (sqrt (2 * λ)))^2 :=
sorry

end laplace_transform_brownian_bridge_l8_8301


namespace max_area_quadrilateral_l8_8176

variable {R : ℝ} {k : ℝ}

noncomputable def maxArea (R k : ℝ) : ℝ :=
if k ≤ real.sqrt 2 - 1 then (k + 1) * R * R
else (2 * R * R * real.sqrt (k * (k + 2))) / (k + 1)

theorem max_area_quadrilateral (hR : 0 < R) (hk : 0 < k) :
  ∃ S : ℝ, S = maxArea R k :=
begin
  use maxArea R k,
  sorry
end

end max_area_quadrilateral_l8_8176


namespace probability_of_event_l8_8372

-- Definitions for the problem setup

-- Box C and its range
def boxC := {i : ℕ | 1 ≤ i ∧ i ≤ 30}

-- Box D and its range
def boxD := {i : ℕ | 21 ≤ i ∧ i ≤ 50}

-- Condition for a tile from box C being less than 20
def tile_from_C_less_than_20 (i : ℕ) : Prop := i ∈ boxC ∧ i < 20

-- Condition for a tile from box D being odd or greater than 45
def tile_from_D_odd_or_greater_than_45 (i : ℕ) : Prop := i ∈ boxD ∧ (i % 2 = 1 ∨ i > 45)

-- Main statement
theorem probability_of_event :
  (19 / 30 : ℚ) * (17 / 30 : ℚ) = (323 / 900 : ℚ) :=
by sorry

end probability_of_event_l8_8372


namespace option_c_correct_l8_8395

theorem option_c_correct : (3 * Real.sqrt 2) ^ 2 = 18 :=
by 
  -- Proof to be provided here
  sorry

end option_c_correct_l8_8395


namespace find_n_l8_8547

theorem find_n
  (log_sin_x_log_cos_x : log 10 (sin x * cos x) = -2)
  (log_sin_cos_x : log 10 (sin x + cos x) = 1 / 2 * (log 10 n - 2)) :
  n = 102 := 
sorry

end find_n_l8_8547


namespace largest_m_divides_30_fact_l8_8514

theorem largest_m_divides_30_fact : 
  let pow2_in_fact := 15 + 7 + 3 + 1,
      pow3_in_fact := 10 + 3 + 1,
      max_m_from_2 := pow2_in_fact,
      max_m_from_3 := pow3_in_fact / 2
  in max_m_from_2 >= 7 ∧ max_m_from_3 >= 7 → 7 = 7 :=
by
  sorry

end largest_m_divides_30_fact_l8_8514


namespace scientific_notation_of_12480_l8_8293

theorem scientific_notation_of_12480 : (12480 = 1.248 * 10^4) ∧ (↑12480 ≈ 1.25 * 10^4) :=
by
  sorry

end scientific_notation_of_12480_l8_8293


namespace solve_for_star_l8_8812

theorem solve_for_star { * : ℝ } (h : 45 - (28 - (37 - (15 - (*^2)))) = 59) : * = 2 * Real.sqrt 5 :=
sorry

end solve_for_star_l8_8812


namespace perimeter_of_parallelogram_l8_8871

theorem perimeter_of_parallelogram (a b : ℕ) (hyp : a = 3 ∧ b = 4) : 
  let c := nat.sqrt (a^2 + b^2) in
  let base := c in
  let height := a + b in
  2 * (base + height) = 24 :=
by
  sorry

end perimeter_of_parallelogram_l8_8871


namespace gcd_abcd_plus_dcba_is_1111_l8_8676

theorem gcd_abcd_plus_dcba_is_1111 (a : ℕ) (ha : a < 7) : 
  let abcd := 1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6),
      dcba := 1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a,
      n := abcd + dcba in gcd n 1111 = 1111 :=
by
  sorry

end gcd_abcd_plus_dcba_is_1111_l8_8676


namespace ted_age_l8_8106

variable (t s : ℕ)

theorem ted_age (h1 : t = 3 * s - 10) (h2 : t + s = 65) : t = 46 := by
  sorry

end ted_age_l8_8106


namespace product_of_odds_2000_4015_l8_8387

noncomputable def product_of_odds (n m : ℕ) : ℕ :=
  List.prod (List.filter (λ x : ℕ, (x > n) ∧ (x ≤ m) ∧ (x % 2 = 1)) (List.range (m + 1)))

theorem product_of_odds_2000_4015 :
  let p := product_of_odds 2000 4015 in
  (p > 0) ∧ (p % 10 = 5) :=
by
  sorry

end product_of_odds_2000_4015_l8_8387


namespace point_inside_curve_iff_m_range_chord_length_range_l8_8198

noncomputable def parametric_line (t α : ℝ) : ℝ × ℝ :=
(t * Real.cos α, t * Real.sin α)

noncomputable def polar_curve (ρ θ m : ℝ) : ℝ :=
ρ^2 - 2 * m * ρ * Real.cos θ - 4

theorem point_inside_curve_iff_m_range (m : ℝ) (hm : 0 < m) :
  (3 - m)^2 + 9 < m^2 + 4 ↔ m ∈ set.Ioi (7 / 3) :=
sorry

theorem chord_length_range (α : ℝ) :
  ∀ m = 3,
  (let ρ_1 := 6 * Real.cos α;
       ρ_2 := 6 * Real.cos α + sqrt(36 * Real.cos α * Real.cos α + 16) in
   abs (ρ_1 - ρ_2) ∈ set.Icc 4 (2 * Real.sqrt 13)) :=
sorry


end point_inside_curve_iff_m_range_chord_length_range_l8_8198


namespace jack_needs_more_money_l8_8634

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l8_8634


namespace has_property_M_1234_inequality_for_set_A_max_elements_in_set_A_l8_8199

namespace ProofProblems

-- (I) Prove that the set {1, 2, 3, 4} has property M
theorem has_property_M_1234 : 
  let A := {1, 2, 3, 4} in
  ∀ (x y : ℕ), x ∈ A → y ∈ A → x ≠ y → |x - y| > (x * y) / 25 :=
by sorry

-- (II) Prove the inequality for the set A with property M
theorem inequality_for_set_A 
  (A : Finset ℕ) (a : ℕ → ℕ) (h_ordered : ∀ i j, i < j → a i < a j)
  (h_property_M : ∀ (i j : ℕ), i ≠ j → (a i ∈ A) → (a j ∈ A) → |a i - a j| > (a i * a j) / 25) :
  let n := A.card in
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → (1 / (a 1 : ℝ) - 1 / (a n : ℝ)) ≥ (n - 1) / 25 :=
by sorry

-- (III) Prove that the maximum number of elements in set A is 9
theorem max_elements_in_set_A 
  (A : Finset ℕ) (a : ℕ → ℕ) (h_ordered : ∀ i j, i < j → a i < a j)
  (h_property_M : ∀ (i j : ℕ), i ≠ j → (a i ∈ A) → (a j ∈ A) → |a i - a j| > (a i * a j) / 25) :
  A.card ≤ 9 :=
by sorry

end ProofProblems

end has_property_M_1234_inequality_for_set_A_max_elements_in_set_A_l8_8199


namespace find_x7_l8_8000

-- Definitions for the conditions
def seq (x : ℕ → ℕ) : Prop :=
  (x 6 = 144) ∧ ∀ n, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → (x (n + 3) = x (n + 2) * (x (n + 1) + x n))

-- Theorem statement to prove x_7 = 3456
theorem find_x7 (x : ℕ → ℕ) (h : seq x) : x 7 = 3456 := sorry

end find_x7_l8_8000


namespace rounding_to_nearest_hundredth_l8_8020

def x : ℝ := 0.6457

theorem rounding_to_nearest_hundredth : 
  (real.frac (x * 100) / 100 = 0.65) :=
sorry

end rounding_to_nearest_hundredth_l8_8020


namespace cos_value_l8_8587

theorem cos_value (A : ℝ) (h : Real.sin (π + A) = 1/2) : Real.cos (3*π/2 - A) = 1/2 :=
sorry

end cos_value_l8_8587


namespace log2_bounds_l8_8755

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
  (h3 : 2^10 = 1024) (h4 : 2^11 = 2048) (h5 : 2^12 = 4096) 
  (h6 : 2^13 = 8192) (h7 : 2^14 = 16384) :
  (3 : ℝ) / 10 < log2 10 ∧ log2 10 < (2 : ℝ) / 7 :=
by
  sorry

end log2_bounds_l8_8755


namespace probability_within_two_units_of_origin_l8_8075

def rectangle_area (width height : ℝ) : ℝ := width * height
def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2
def probability_inside_circle (rect_area circle_area : ℝ) : ℝ := circle_area / rect_area

theorem probability_within_two_units_of_origin :
  let r_area := rectangle_area 6 8 in
  let c_area := circle_area 2 in
  probability_inside_circle r_area c_area = Real.pi / 12 :=
by
  sorry

end probability_within_two_units_of_origin_l8_8075


namespace evaluate_expr_equiv_l8_8490

theorem evaluate_expr_equiv : 64^(-1 / 3) + 81^(-2 / 4) = 13 / 36 := by
  have h1 : 64 = 2^6 := rfl
  have h2 : 81 = 3^4 := rfl
  sorry

end evaluate_expr_equiv_l8_8490


namespace largest_prime_factor_4290_l8_8789

theorem largest_prime_factor_4290 : ∀ (n : ℕ), n = 4290 → ∃ p : ℕ, p.prime ∧ p ∣ 4290 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 4290 → q ≤ p := 
by
  sorry

end largest_prime_factor_4290_l8_8789


namespace part1_S3_values_part2_unique_Sn_part3_parity_Sn_l8_8246

-- Definitions for statements
def filled_table (2 : ℕ) (n : ℕ) (cells : fin 2 → fin n → ℕ) : Prop :=
  ∀ i : fin 2, ∃ p : finset ℕ, p.card = n ∧ ∀ j, cells i j ∈ p

def table_sum (n : ℕ) (cells : fin 2 → fin n → ℕ) : ℕ :=
  finset.univ.sum (λ j, abs (cells 0 j - cells 1 j))

-- Statements
-- Part (Ⅰ)
theorem part1_S3_values (cells : fin 2 → fin 3 → ℕ)
  (h_filled : filled_table 2 3 cells)
  (h_a : ∀ j, cells 0 j ∈ finset.of_list [1, 3, 5]):
  {table_sum 3 cells | ∀ (i : fin 3), cells 1 i ∈ finset.of_list [2, 4, 6]} = {3, 5, 7, 9} := sorry

-- Part (Ⅱ)
theorem part2_unique_Sn (n : ℕ) (cells : fin 2 → fin n → ℕ)
  (h_filled : filled_table 2 n cells)
  (h_a : ∀ j, cells 0 j = j + 1):
  (∀ b, {table_sum n cells | ∀ (i : fin n), cells 1 i ∈ b} = {n^2}) := sorry

-- Part (Ⅲ)
theorem part3_parity_Sn (n : ℕ) (cells : fin 2 → fin n → ℕ)
  (h_filled : filled_table 2 n cells):
  (∀ b₁ b₂, parity (table_sum n cells | ∀ (i : fin n), cells 1 i ∈ b₁) =
            parity (table_sum n cells | ∀ (i, fin n), cells 1 i ∈ b₂)) := sorry

end part1_S3_values_part2_unique_Sn_part3_parity_Sn_l8_8246


namespace min_value_hyperbola_l8_8596

theorem min_value_hyperbola (x y : ℝ) (h : x^2 / 4 - y^2 = 1) :
  3 * x^2 - 2 * x * y ≥ 6 + 4 * real.sqrt 2 := by
  sorry

end min_value_hyperbola_l8_8596


namespace always_possible_subset_l8_8523

theorem always_possible_subset (n : ℕ) (h : n % 2 = 0) : 
  ∀ (a : Fin n → ℕ), (∀ k, a k ≤ n) → (∑ k, a k = 2 * n) → 
  ∃ (s : Fin n → Prop), (∑ k in {i : Fin n | s i}, a k = n) := 
sorry

end always_possible_subset_l8_8523


namespace complex_modulus_fraction_l8_8110

theorem complex_modulus_fraction :
  abs ((√3 + √2 * Complex.I) * (√5 + √2 * Complex.I) * (√5 + √3 * Complex.I)
    / ((√2 - √3 * Complex.I) * (√2 - √5 * Complex.I))) = 2 * √2 :=
by sorry

end complex_modulus_fraction_l8_8110


namespace sum_binomials_l8_8877

-- Defining binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem sum_binomials : binom 12 4 + binom 10 3 = 615 :=
by
  -- Here we state the problem, and the proof will be left as 'sorry'.
  sorry

end sum_binomials_l8_8877


namespace sum_of_abc_l8_8593

theorem sum_of_abc (a b c : ℝ) (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0) :
  a + b + c = 18 :=
sorry

end sum_of_abc_l8_8593


namespace spherical_to_rectangular_conversion_l8_8848

variables (ρ θ φ : ℝ)

def initial_coordinates := (-5, -7, 4)
def spherical_to_rectangular_conditions := 
  (-5 = ρ * sin φ * cos θ) ∧
  (-7 = ρ * sin φ * sin θ) ∧
  (4 = ρ * cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular_conditions ρ θ φ →
  (5, 7, 4) = (ρ * sin (-φ) * cos (θ + π), ρ * sin (-φ) * sin (θ + π), ρ * cos (-φ)) :=
by
  intro h
  sorry

end spherical_to_rectangular_conversion_l8_8848


namespace max_value_fraction_l8_8394

theorem max_value_fraction {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (4 * x * z + y * z) / (x^2 + y^2 + z^2) ≤ sqrt 17 / 2 := 
sorry

end max_value_fraction_l8_8394


namespace find_fg3_l8_8210

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := (x + 2)^2 - 4 * x

theorem find_fg3 : f (g 3) = 49 :=
by
  sorry

end find_fg3_l8_8210


namespace students_taken_exam_l8_8684

def students_failed (T : ℕ) : ℝ :=
  (1 / 2) * (3 / 5) * T

theorem students_taken_exam (T : ℕ) (ht : students_failed T = 24) : T = 80 :=
  sorry

end students_taken_exam_l8_8684


namespace carlos_laundry_time_l8_8467

def washing_time1 := 30
def washing_time2 := 45
def washing_time3 := 40
def washing_time4 := 50
def washing_time5 := 35
def drying_time1 := 85
def drying_time2 := 95

def total_laundry_time := washing_time1 + washing_time2 + washing_time3 + washing_time4 + washing_time5 + drying_time1 + drying_time2

theorem carlos_laundry_time : total_laundry_time = 380 :=
by
  sorry

end carlos_laundry_time_l8_8467


namespace blue_pill_cost_is_25_l8_8471

variable (blue_pill_cost red_pill_cost : ℕ)

-- Clara takes one blue pill and one red pill each day for 10 days.
-- A blue pill costs $2 more than a red pill.
def pill_cost_condition (blue_pill_cost red_pill_cost : ℕ) : Prop :=
  blue_pill_cost = red_pill_cost + 2 ∧
  10 * blue_pill_cost + 10 * red_pill_cost = 480

-- Prove that the cost of one blue pill is $25.
theorem blue_pill_cost_is_25 (h : pill_cost_condition blue_pill_cost red_pill_cost) : blue_pill_cost = 25 :=
  sorry

end blue_pill_cost_is_25_l8_8471


namespace distinct_diagonals_of_convex_nonagon_l8_8886

def is_convex (P : polygon) : Prop := sorry -- definition placeholder for convex polygon
def is_nonagon (P : polygon) : Prop := polygon.num_sides P = 9

theorem distinct_diagonals_of_convex_nonagon (P : polygon) (h1 : is_convex P) (h2 : is_nonagon P) :
  polygon.num_diagonals P = 27 :=
sorry

end distinct_diagonals_of_convex_nonagon_l8_8886


namespace max_value_of_trig_function_l8_8360

theorem max_value_of_trig_function :
  ∃ x : ℝ, y = sin^2 x - 4 * cos x + 2 ∧ -1 ≤ cos x ∧ cos x ≤ 1 ∧ y = 6 :=
sorry

end max_value_of_trig_function_l8_8360


namespace problem1_max_min_value_problem2_min_value_exists_l8_8564

noncomputable section

-- Helper definitions to express the functions involved
def f1 (x : ℝ) : ℝ := -x^2 + 3 * x - Real.log x
def f2 (b : ℝ) (x : ℝ) : ℝ := b * x - Real.log x

-- Problem 1
theorem problem1_max_min_value :
  (∀ x ∈ Set.Icc (1/2) 2, x > 0) ∧
  (∀ x ∈ Set.Icc (1/2) 2, f1 x ≤ 2) ∧
  (2 = f1 1) ∧
  (Real.log 2 + 5/4 = f1 (1/2)) := by
  sorry

-- Problem 2
theorem problem2_min_value_exists :
  ∃ (b : ℝ) (hb : 0 < b), b = Real.exp 2 ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.exp 1), f2 b x ≥ 3) ∧
  (f2 b (Real.exp 1) = 3) := by
  sorry

end problem1_max_min_value_problem2_min_value_exists_l8_8564


namespace jessica_blueberry_pies_l8_8640

theorem jessica_blueberry_pies 
  (total_pies : ℕ)
  (ratio_apple : ℕ)
  (ratio_blueberry : ℕ)
  (ratio_cherry : ℕ)
  (h_total : total_pies = 36)
  (h_ratios : ratio_apple = 2)
  (h_ratios_b : ratio_blueberry = 5)
  (h_ratios_c : ratio_cherry = 3) : 
  total_pies * ratio_blueberry / (ratio_apple + ratio_blueberry + ratio_cherry) = 18 := 
by
  sorry

end jessica_blueberry_pies_l8_8640


namespace math_problem_l8_8503

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0):
  ((2^(2^n) + 1) * (2^(2^m) + 1)) % (m * n) = 0 →
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) :=
by
  sorry

end math_problem_l8_8503


namespace find_f_4_l8_8973

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then x^2 else f (x - 2)

theorem find_f_4 : f 4 = 0 :=
by
  sorry

end find_f_4_l8_8973


namespace tangent_line_at_1_neg1_tangency_points_for_perpendicular_l8_8567

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Prove the equation of the tangent line at the point (1, -1).
theorem tangent_line_at_1_neg1 : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -3 ∧ (∀ x y : ℝ, y = f'(1) * (x - 1) + f(1) ↔ a * x + b * y + c = 0) :=
begin
  sorry
end

-- Define the line that is perpendicular to the given line
def g (x : ℝ) : ℝ := - (1/2) * x + 3

-- Prove the coordinates of the tangency point for the tangent that is perpendicular to given line.
theorem tangency_points_for_perpendicular :
  ∃ (m n : ℝ), (m = 1 ∨ m = -1) ∧ n = m^3 - m - 1 :=
begin
  sorry
end

end tangent_line_at_1_neg1_tangency_points_for_perpendicular_l8_8567


namespace correct_propositions_l8_8038

-- Define the conditions
def cond1 (a : ℝ) : Prop := (a > 1) → (1 / a < 1) ∧ (¬((1 / a < 1) → a > 1))

def negation_of_prop : Prop := 
  (∀ x : ℝ, x < 1 → x^2 < 1) ↔ ¬∃ x : ℝ, x < 1 ∧ x^2 ≥ 1

def min_value_inequality (x : ℝ) (h : x ≥ 2) : Prop := 
  x + 2 / x ≥ 2 * Real.sqrt 2

def cond4 (a b: ℝ) : Prop := 
  (a ≠ 0 → ¬((a ≠ 0) → ab ≠ 0))

-- State the problem as proving the conjunction of the correct options A and D
theorem correct_propositions (a b: ℝ) : cond1 a ∧ cond4 a b :=
sorry

end correct_propositions_l8_8038


namespace triangle_angle_condition_iff_dist_to_orthocenter_eq_circumradius_l8_8305

theorem triangle_angle_condition_iff_dist_to_orthocenter_eq_circumradius
  (A B C H O : Point)
  (∠_A : ℝ) (∠_B : ℝ) (∠_C : ℝ)
  (R : ℝ)
  (is_triangle : Triangle A B C)
  (orthocenter : Orthocenter A B C = H)
  (circumcenter : Circumcenter A B C = O)
  (circumradius : Circumradius A B C = R) :
  (∠_A = 60 ∨ ∠_A = 120) ↔ (dist A H = R) :=
by sorry

end triangle_angle_condition_iff_dist_to_orthocenter_eq_circumradius_l8_8305


namespace geometric_figure_total_length_l8_8455

def rectangle_segments_length (l w l_remove_sd w_remove_sd w_mid) :=
  (l - l_remove_sd) + w_remove_sd + w_remove_sd + w_mid + 2 * l_remove_sd

theorem geometric_figure_total_length:
  ∀(l w l_remove_sd w_remove_sd w_mid : ℕ),
  l = 10 ∧ w = 5 ∧ 
  l_remove_sd = 3 ∧ w_remove_sd = 2 ∧ 
  w_mid = 4 →  -- given explicitly middle side involved
  rectangle_segments_length l w l_remove_sd w_remove_sd w_mid = 16 :=
by
  intros l w l_remove_sd w_remove_sd w_mid h,
  rw [and_assoc] at h,
  cases h with hl h_rest,
  cases h_rest with hw h_rest,
  cases h_rest with hl_side h_rest,
  cases h_rest with hw_side h_mid,
  simp [rectangle_segments_length],
  rw [hl, hw, hl_side, hw_side, h_mid],
  norm_num,
  sorry

end geometric_figure_total_length_l8_8455


namespace find_c_trajectory_eqn_l8_8531

-- Define circle equation and conditions
def circle_eqn (x y : ℝ) (c : ℝ) := x^2 + y^2 - 4 * x + 2 * y + c = 0

-- Points A and B intersect y-axis, Center is M
def y_axis_intersect (x : ℝ) := x = 0
def center (x y : ℝ) := x = 2 ∧ y = -1

-- Given angle AMB is 90 degrees
def angle_AMB_90 : Prop := ∀ A B : ℝ × ℝ, y_axis_intersect A.1 → y_axis_intersect B.1 → (A.1 - 2) * (B.1 - 2) + (A.2 + 1) * (B.2 + 1) = 0

-- Given circle intersects the line x + y - 1 = 0 at points E and F
def intersection_eqn (x y : ℝ) := x + y - 1 = 0

-- Moving point H with ratio of distances to points E and F as λ
def distance_ratio_eqn (x₁ y₁ x₂ y₂ λ : ℝ) : Prop :=
  (1 - λ^2) * x₁^2 + (1 - λ^2) * y₁^2 + 8 * λ^2 * x₁ - (2 + 6 * λ^2) * y₁ + 1 - 25 * λ^2 = 0

-- Theorem statements
theorem find_c : ∃ c : ℝ, angle_AMB_90 → c = -3 :=
  sorry

theorem trajectory_eqn (λ : ℝ) (hλ : λ > 0) : 
  ∃ H : ℝ × ℝ, (λ = 1 → (H.1 - H.2 - 3 = 0)) ∧ (λ ≠ 1 → distance_ratio_eqn H.1 H.2 E.1 E.2 λ) :=
  sorry

end find_c_trajectory_eqn_l8_8531


namespace diameter_of_circle_A_l8_8117

theorem diameter_of_circle_A
  (diameter_B : ℝ)
  (r : ℝ)
  (h1 : diameter_B = 16)
  (h2 : r^2 = (r / 8)^2 * 4):
  2 * (r / 2) = 8 :=
by
  sorry

end diameter_of_circle_A_l8_8117


namespace AIE_degree_measure_l8_8624

theorem AIE_degree_measure 
  (α β γ : ℝ)
  (ΔABC : Triangle ℝ) 
  (D E F I : Point ℝ)
  (h₁ : ∠BAC = 50)
  (h₂ : AngleBisector AD ABC I)
  (h₃ : AngleBisector BE ABC I)
  (h₄ : AngleBisector CF ABC I) 
  (h₅ : Incenter I ΔABC) :
  ∠AIE = 65 :=
by
  sorry

end AIE_degree_measure_l8_8624


namespace patients_per_doctor_l8_8830

theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) (h_patients : total_patients = 400) (h_doctors : total_doctors = 16) : 
  (total_patients / total_doctors) = 25 :=
by
  sorry

end patients_per_doctor_l8_8830


namespace locus_of_P_is_circle_l8_8281

open Real

-- Define the basic structure of an ellipse
axiom ellipse (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the foci of the ellipse
axiom foci (c a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  c^2 = a^2 - b^2 ∧ F1 = (-c, 0) ∧ F2 = (c, 0)

-- Define points on the ellipse not at vertices
axiom point_on_ellipse_not_at_vertices (x y a b : ℝ) : Prop :=
  ellipse x y a b ∧ ¬(y = 0 ∧ (x = a ∨ x = -a))

-- Define the conditions for Q, P on ellipse
axiom condition_QP (F1 F2 Q P : ℝ × ℝ) (a : ℝ) : Prop :=
  (Q ∈ { (x, y) | ellipse x y a (sqrt (a^2 - (fst F2)^2)) ∧
                 ¬(y = 0 ∧ (x = a ∨ x = -a)) }) ∧
  ∃ (ext_bisector : ℝ × ℝ → ℝ × ℝ → ℝ), -- external bisector of ∠F1QF2
    (P = foot_perpendicular ext_bisector F1 ∨ P = foot_perpendicular ext_bisector F2)

-- Main theorem statement: Prove the locus of P is a circle with the described conditions.
theorem locus_of_P_is_circle (a b c : ℝ) (F1 F2 Q P : ℝ × ℝ) :
  foci c a b F1 F2 →
  point_on_ellipse_not_at_vertices (fst Q) (snd Q) a b →
  condition_QP F1 F2 Q P a →
  (locus P = { point : ℝ × ℝ | point_distance (0, 0) point = a } \ {(-a, 0), (a, 0)} ) :=
sorry

end locus_of_P_is_circle_l8_8281


namespace prob_neither_defective_l8_8227

-- Definitions for the conditions
def totalPens : ℕ := 8
def defectivePens : ℕ := 2
def nonDefectivePens : ℕ := totalPens - defectivePens
def selectedPens : ℕ := 2

-- Theorem statement for the probability that neither of the two selected pens is defective
theorem prob_neither_defective : 
  (nonDefectivePens / totalPens) * ((nonDefectivePens - 1) / (totalPens - 1)) = 15 / 28 := 
  sorry

end prob_neither_defective_l8_8227


namespace solve_z_for_complex_eq_l8_8705

theorem solve_z_for_complex_eq (i : ℂ) (h : i^2 = -1) : ∀ (z : ℂ), 3 - 2 * i * z = -4 + 5 * i * z → z = -i :=
by
  intro z
  intro eqn
  -- The proof would go here
  sorry

end solve_z_for_complex_eq_l8_8705


namespace tangent_inclination_range_l8_8950

-- Define the sine function
def sin_curve (x : ℝ) : ℝ := Real.sin x

-- Define the derivative of the sine function, which is the cosine function
def derivative_sin_curve (x : ℝ) : ℝ := Real.cos x

-- Define the range of cosine to help in the proof
def range_of_cosine : Set ℝ := Set.Icc (-1 : ℝ) (1 : ℝ)

-- Define the corresponding angles from the range of slope
noncomputable def inclination_angles_range : Set ℝ :=
  Set.union (Set.Icc 0 (Real.pi / 4)) (Set.Ico (3 * Real.pi / 4) Real.pi)

theorem tangent_inclination_range :
  (∀ (x : ℝ), (x ∈ Set.Icc 0 (2 * Real.pi)) → 
    ∃(m : ℝ) (θ : ℝ), 
      m = derivative_sin_curve x ∧ 
      θ ∈ inclination_angles_range ∧ 
      m = Real.cos θ) :=
by
  sorry

end tangent_inclination_range_l8_8950


namespace find_common_difference_l8_8541

-- Definitions for the arithmetic sequence and conditions.
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n-1) * d

-- Conditions derived from the original problem.
def condition_1 := arithmetic_sequence 2 d 2 + arithmetic_sequence 2 d 4 = arithmetic_sequence 2 d 6

-- The main statement to prove.
theorem find_common_difference (d : ℝ) (h1 : condition_1) : d = 2 :=
by
  sorry

end find_common_difference_l8_8541


namespace total_cost_of_fencing_l8_8505

noncomputable def π : ℝ := 3.14159

def diameter (d : ℝ) : Prop := d = 22

def rate (r : ℝ) : Prop := r = 3

theorem total_cost_of_fencing (d : ℝ) (r : ℝ) (C : ℝ) :
  diameter d → rate r → C = π * d → (C * r) = 207.36 :=
by
  intros h_d h_r h_C
  rw [h_d, h_r]
  sorry

end total_cost_of_fencing_l8_8505


namespace largest_multiple_of_8_less_than_100_l8_8768

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8768


namespace simplify_and_rationalize_l8_8694

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 :=
by sorry

end simplify_and_rationalize_l8_8694


namespace find_OC_dot_AB_l8_8181

open Real EuclideanGeometry

variables (OA OB OC : ℝ^3)
variables (O : ℝ^3)

def triangleInscribedInCircle := ∥OA∥ = 1 ∧ ∥OB∥ = 1 ∧ ∥OC∥ = 1 ∧ (3 • OA + 4 • OB + 5 • OC = 0)

theorem find_OC_dot_AB (h : triangleInscribedInCircle OA OB OC) :
  (OC) • (OB - OA) = -1/5 :=
sorry

end find_OC_dot_AB_l8_8181


namespace original_message_is_net_l8_8426

noncomputable def decode_message (x : Fin 13 → ℕ) (k1 k2 : ℕ) : Prop :=
  let y1 := (2 * x 0 + x 2 + (-1)^1 * k1) % 32
  let y2 := (x 0 + x 1 + (-1)^1 * k2) % 32
  let y3 := (2 * x 2 + x 4 + (-1)^1 * k1) % 32
  let y4 := (x 2 + x 3 + (-1)^2 * k2) % 32
  let y5 := (2 * x 4 + x 6 + (-1)^1 * k1) % 32
  let y6 := (x 4 + x 5 + (-1)^3 * k2) % 32
  let y7 := (2 * x 6 + x 8 + (-1)^1 * k1) % 32
  let y8 := (x 6 + x 7 + (-1)^4 * k2) % 32
  let y9 := (2 * x 8 + x 10 + (-1)^1 * k1) % 32
  let y10 := (x 8 + x 9 + (-1)^5 * k2) % 32
  let y11 := (2 * x 10 + x 12 + (-1)^1 * k1) % 32
  let y12 := (x 10 + x 11 + (-1)^6 * k2) % 32
  let y13 := (2 * x 12 + (x 0 + x 1 + ⋯ + x 12 + k1) + (-1)^1 * k1) % 32
  y1 = 23 ∧ y2 = 4 ∧ y3 = 21 ∧ y4 = 7 ∧ y5 = 24 ∧
  y6 = 2 ∧ y7 = 26 ∧ y8 = 28 ∧ y9 = 28 ∧ y10 = 4 ∧
  y11 = 2 ∧ y12 = 16 ∧ y13 = 10

theorem original_message_is_net (x : Fin 13 → ℕ) (k1 k2 : ℕ) :
  decode_message x k1 k2 → first_word x = "нет" :=
sorry

end original_message_is_net_l8_8426


namespace incorrect_statement_E_l8_8290

theorem incorrect_statement_E :
  ¬ (∀ x y : ℝ, x > 0 → y > 0 → x * y = p → x + y = max) :=
by sorry

end incorrect_statement_E_l8_8290


namespace rectangle_area_l8_8850

-- Definitions for conditions
variables (r w : ℝ)

-- Conditions given in the problem
def condition1 := w / 2 = r
def condition2 := 3 * w / 2 = sqrt (10 * w^2) / 2

-- Given the conditions and the radius, prove the area of the rectangle
theorem rectangle_area (h1 : condition1 r w) (h2 : condition2 r w) : 3 * (w * w) = 30 * r * r :=
by sorry

end rectangle_area_l8_8850


namespace bob_wins_if_both_even_alice_wins_if_n_even_and_m_odd_l8_8420

-- Definitions for the problem conditions
def grid (n m : ℕ) : Type := (fin n) × (fin m)

def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Ontological propositions regarding winning strategies
def has_winning_strategy (player : string) (n m : ℕ) : Prop :=
match player with
| "Alice" => ∀ (g : grid n m), ∃ strategy : fin (n * m div 2) → grid n m, True
| "Bob" => ∀ (g : grid n m), ∃ strategy : fin (n * m div 2) → grid n m, True
| _ => False
end

-- Theorems stating the winning strategies
theorem bob_wins_if_both_even (n m : ℕ) (hn_even : even n) (hm_even : even m) : has_winning_strategy "Bob" n m :=
sorry

theorem alice_wins_if_n_even_and_m_odd (n m : ℕ) (hn_even : even n) (hm_odd : odd m) : has_winning_strategy "Alice" n m :=
sorry

end bob_wins_if_both_even_alice_wins_if_n_even_and_m_odd_l8_8420


namespace distance_between_ships_l8_8383

theorem distance_between_ships
  (h_lighthouse : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (tan_30 : Real.tan 30 = 1 / Real.sqrt 3)
  (tan_45 : Real.tan 45 = 1)
  (AL : ℝ)
  (BL : ℝ)
  (hAL : AL = h_lighthouse / Real.tan 30)
  (hBL : BL = h_lighthouse / Real.tan 45)
  : AL + BL = h_lighthouse * Real.sqrt(3) + h_lighthouse := 
sorry

end distance_between_ships_l8_8383


namespace eduardo_winning_strategy_l8_8261

theorem eduardo_winning_strategy (p : ℕ) (hp : p.prime) (hp2 : 2 ≤ p) :
  ∃ strategy : (fin p → ℕ) → option (fin p × ℕ), -- a strategy function for Eduardo's moves
  ∀ (moves : fin p → option (fin p × ℕ)),        -- possible moves by both players
   ∃ M : ℕ,                                      -- resulting number M
   (M = ∑ i in finset.range p, moves i * 10 ^ i)  -- M is constructed based on game rules
   ∧ M % p = 0 :=                                  -- Eduardo's goal: M is divisible by p
begin
  sorry -- The proof will be here
end

end eduardo_winning_strategy_l8_8261


namespace complex_number_simplification_l8_8671

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : 
  (↑(1 : ℂ) - i) / (↑(1 : ℂ) + i) ^ 2017 = -i :=
sorry

end complex_number_simplification_l8_8671


namespace num_regular_soda_l8_8441

theorem num_regular_soda (t d r : ℕ) (h₁ : t = 17) (h₂ : d = 8) (h₃ : r = t - d) : r = 9 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end num_regular_soda_l8_8441


namespace polynomial_sum_l8_8663

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8663


namespace PlayerB_prevents_A_winning_l8_8752

theorem PlayerB_prevents_A_winning :
  ∀ (grid : ℕ × ℕ → option bool), 
  (∀ i j, grid (i, j) = some true ∨ grid (i, j) = some false) →
  (∃ strategy : (ℕ × ℕ → option bool) → (ℕ × ℕ), ∀ g : (ℕ × ℕ → option bool),
    (∀ m n, ¬ ∃ k, (k ≤ 11) ∧ (∀ d, (d < k) → g (m + d, n) = some true ∨ 
                                g (m, n + d) = some true ∨ 
                                g (m + d, n + d) = some true) ∧
                            strategy g ≠ (m + d, n) ∧
                            strategy g ≠ (m, n + d) ∧
                            strategy g ≠ (m + d, n + d))) :=
begin
  sorry
end

end PlayerB_prevents_A_winning_l8_8752


namespace largest_prime_factor_of_binary_numbers_l8_8500

noncomputable def binary_to_decimal (L : List ℕ) : ℕ :=
  List.sum (List.mapWithIndex (λ i d, d * 2^i) L.reverse)

def prime_factors (n : ℕ) : List ℕ :=
  if h : n > 0 then
    (Nat.factors n).eraseDup
  else
    []

def largest_prime_factor (a b : ℕ) : ℕ :=
  List.maximum (List.filter Nat.prime (prime_factors (Nat.gcd a b))).getD 0

theorem largest_prime_factor_of_binary_numbers :
  largest_prime_factor (binary_to_decimal [0,0,0,0,1,0,0,1]) (binary_to_decimal [0,0,0,0,0,1,0,0,1]) = binary_to_decimal [1,1] :=
begin
  sorry
end

end largest_prime_factor_of_binary_numbers_l8_8500


namespace large_pizza_slices_l8_8861

variable (L : ℕ)

theorem large_pizza_slices :
  (2 * L + 2 * 8 = 48) → (L = 16) :=
by 
  sorry

end large_pizza_slices_l8_8861


namespace remainder_theorem_example_l8_8151

-- Given polynomial and divisor variables
def polynomial : ℚ[X] := 8 * X^5 - 10 * X^4 + 3 * X^3 + 5 * X^2 - 7 * X - 35
def divisor : ℚ := 5

-- Statement to prove
theorem remainder_theorem_example : polynomial.eval divisor = 19180 := by
  sorry

end remainder_theorem_example_l8_8151


namespace symmetric_product_l8_8280

-- Definition of symmetry about the origin for complex numbers
def symmetric_about_origin (z1 z2 : ℂ) : Prop :=
  z2 = -z1

-- Problem statement
theorem symmetric_product (z1 z2 : ℂ) (hz : symmetric_about_origin z1 z2) (h1 : z1 = 2 - 1 * Complex.i) :
  z1 * z2 = -3 + 4 * Complex.i :=
by
  rw [symmetric_about_origin] at hz
  sorry

end symmetric_product_l8_8280


namespace unique_real_solution_l8_8132

theorem unique_real_solution (x y z : ℝ) :
  (x^3 - 3 * x = 4 - y) ∧ 
  (2 * y^3 - 6 * y = 6 - z) ∧ 
  (3 * z^3 - 9 * z = 8 - x) ↔ 
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end unique_real_solution_l8_8132


namespace sum_of_coeffs_eq_32_l8_8113

def polynomial := 4 * (2 * x ^ 8 + 5 * x ^ 6 - 3 * x + 9) + 5 * (x ^ 7 - 3 * x ^ 3 + 2 * x ^ 2 - 4)

def sum_of_coefficients (p : ℕ → ℤ) : ℤ :=
  p 0

theorem sum_of_coeffs_eq_32 : 
  sum_of_coefficients polynomial = 32 :=
  sorry

end sum_of_coeffs_eq_32_l8_8113


namespace continuous_functional_equation_solution_l8_8918

noncomputable def exponential_solution (a c : ℝ) : ℝ → ℝ :=
λ x, a * c^x

theorem continuous_functional_equation_solution (f : ℝ → ℝ) (a c : ℝ) 
  (h_cont : Continuous f) 
  (h_eq : ∀ x y : ℝ, f(x + y) * f(x - y) = (f x)^2) :
  ∀ x : ℝ, f x = exponential_solution a c x :=
by
  sorry

end continuous_functional_equation_solution_l8_8918


namespace range_of_m_l8_8556

theorem range_of_m (m : ℝ) :
  let p := (2 < m ∧ m < 4)
  let q := (m > 1 ∧ 4 - 4 * m < 0)
  (¬ (p ∧ q) ∧ (p ∨ q)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 4) :=
by intros p q h
   let p := 2 < m ∧ m < 4
   let q := m > 1 ∧ 4 - 4 * m < 0
   sorry

end range_of_m_l8_8556


namespace quadrilateral_diagonals_bisect_is_parallelogram_quadrilateral_diagonals_perpendicular_not_rhombus_parallelogram_equal_diagonals_not_rhombus_parallelogram_right_angle_is_rectangle_correct_propositions_l8_8798

theorem quadrilateral_diagonals_bisect_is_parallelogram
  (Q : Type) [quadrilateral Q] (bis : ∀(d1 d2 : diagonal Q), bisects d1 d2) : parallelogram Q := by
  sorry

theorem quadrilateral_diagonals_perpendicular_not_rhombus
  (Q : Type) [quadrilateral Q] (perp : ∀(d1 d2 : diagonal Q), perpendicular d1 d2) : ¬rhombus Q := by
  sorry

theorem parallelogram_equal_diagonals_not_rhombus
  (P : Type) [parallelogram P] (eqDiags : ∀(d1 d2 : diagonal P), equal d1 d2) : ¬rhombus P := by
  sorry

theorem parallelogram_right_angle_is_rectangle
  (P : Type) [parallelogram P] (rightAngle : ∃(a : angle P), is_right_angle a) : rectangle P := by
  sorry

theorem correct_propositions :
  quadrilateral_diagonals_bisect_is_parallelogram ∧ parallelogram_right_angle_is_rectangle ∧ 
  quadrilateral_diagonals_perpendicular_not_rhombus ∧ parallelogram_equal_diagonals_not_rhombus := by
  apply And.intro;
  apply quadrilateral_diagonals_bisect_is_parallelogram;
  apply parallelogram_right_angle_is_rectangle;
  apply quadrilateral_diagonals_perpendicular_not_rhombus;
  apply parallelogram_equal_diagonals_not_rhombus;

end quadrilateral_diagonals_bisect_is_parallelogram_quadrilateral_diagonals_perpendicular_not_rhombus_parallelogram_equal_diagonals_not_rhombus_parallelogram_right_angle_is_rectangle_correct_propositions_l8_8798


namespace inequality_holds_for_all_x_l8_8520

theorem inequality_holds_for_all_x (m : ℝ) (h : ∀ x : ℝ, |x + 5| ≥ m + 2) : m ≤ -2 :=
sorry

end inequality_holds_for_all_x_l8_8520


namespace find_S6_l8_8948

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ)

-- The sequence {a_n} is given as a geometric sequence
-- Partial sums are given as S_2 = 1 and S_4 = 3

-- Conditions
axiom geom_sequence : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0
axiom S2 : S_n 2 = 1
axiom S4 : S_n 4 = 3

-- Theorem statement
theorem find_S6 : S_n 6 = 7 :=
sorry

end find_S6_l8_8948


namespace basketball_team_selection_l8_8433

theorem basketball_team_selection : 
  let total_members := 12
  let lineup_size := 5
  let captain_count := 1
  ∀ (choose : ℕ → ℕ → ℕ), 
    choose total_members captain_count * choose (total_members - captain_count) (lineup_size - captain_count) = 3960 :=
by
  let total_members := 12
  let lineup_size := 5
  let captain_count := 1
  exact fun choose => by
    have remaining_players := total_members - captain_count
    have players_to_choose := lineup_size - captain_count
    have number_of_ways_captain := choose total_members captain_count
    have number_of_ways_remaining := choose remaining_players players_to_choose
    exact calc
      number_of_ways_captain * number_of_ways_remaining
        = choose 12 1 * choose 11 4 : by rw [number_of_ways_captain, number_of_ways_remaining]
        = 12 * 330 : by sorry
        = 3960 : by norm_num

end basketball_team_selection_l8_8433


namespace last_passenger_probability_l8_8329

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l8_8329


namespace decreasing_function_interval_l8_8565

theorem decreasing_function_interval (a : ℝ) :
  (∀ x y : ℝ, x ∈ Iic (6 : ℝ) → y ∈ Iic (6 : ℝ) → x ≤ y → f x ≥ f y) →
    a ∈ Iic (-5) := 
by 
  let f : ℝ → ℝ := λ x, x^2 + 2*(a-1)*x + 2
  sorry

end decreasing_function_interval_l8_8565


namespace knights_minimum_count_l8_8422

/-- There are 1001 people sitting around a round table, each of whom is either a knight (always tells the truth) or a liar (always lies).
Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight.
Prove that the minimum number of knights that can be sitting at the table is 502. -/
theorem knights_minimum_count (n : ℕ) (h : n = 1001) (N : ℕ) (L : ℕ) 
  (h1 : N + L = n) (h2 : ∀ i, (i < n) → 
    ((is_knight i ∧ is_liar ((i + 1) % n)) ∨ (is_liar i ∧ is_knight ((i + 1) % n)))) 
  : N = 502 :=
sorry

end knights_minimum_count_l8_8422


namespace wall_thickness_l8_8832

def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

def wall_area (length width : ℝ) : ℝ :=
  length * width

def required_volume (number_of_bricks volume_per_brick : ℝ) : ℝ :=
  number_of_bricks * volume_per_brick

theorem wall_thickness {
  number_of_bricks : ℝ,
  brick_length : ℝ,
  brick_width : ℝ,
  brick_height : ℝ,
  wall_length : ℝ,
  wall_width : ℝ
} (h_vol: brick_volume brick_length brick_width brick_height = 1650)
  (h_area: wall_area wall_length wall_width = 60000)
  (h_num_bricks: number_of_bricks = 72.72727272727273)
  (h_wall_length: wall_length = 200)
  (h_wall_width: wall_width = 300):
  wall_length > 0 ∧ wall_width > 0 → 
  required_volume number_of_bricks (brick_volume brick_length brick_width brick_height) / (wall_area wall_length wall_width) = 2 :=
by sorry

end wall_thickness_l8_8832


namespace brand_b_contains_65_percent_millet_l8_8437

def percentage_millet_B (x : ℝ) (pa pb : ℝ) (pb_safflower : ℝ) (mix_millet : ℝ) (mix_A : ℝ) (mix_B : ℝ) : Prop :=
  mix_A * pa + mix_B * x = mix_millet

theorem brand_b_contains_65_percent_millet (x : ℝ) (pa pb : ℝ) (pb_safflower : ℝ) (mix_millet : ℝ) (mix_A : ℝ) (mix_B : ℝ) :
  pa = 0.40 →
  pb_safflower = 0.35 →
  mix_millet = 0.50 →
  mix_A = 0.60 →
  mix_B = 0.40 →
  percentage_millet_B x pa pb pb_safflower mix_millet mix_A mix_B →
  x = 0.65 :=
by {
  intros,
  sorry
}

end brand_b_contains_65_percent_millet_l8_8437


namespace find_m_l8_8206

-- Definitions based on given conditions
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Sum of the vectors
def vec_sum (m : ℝ) : ℝ × ℝ := (vec_a m).fst + vec_b.fst, (vec_a m).snd + vec_b.snd

-- Dot product function
def dot_prod (u : ℝ × ℝ) (v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

-- Theorem we need to prove
theorem find_m (m : ℝ) : dot_prod (vec_sum m) vec_b = 0 → m = 8 :=
by
  sorry

end find_m_l8_8206


namespace hyperbola_equation_l8_8835

/-- Given an ellipse with the equation 4x^2 + y^2 = 1 and a hyperbola sharing the same foci
    with one of its asymptotes given by the equation y = sqrt(2) * x, prove that the equation 
    of the hyperbola is 2y^2 - 4x^2 = 1. -/
theorem hyperbola_equation (x y : ℝ) (h_ellipse : 4 * x^2 + y^2 = 1) 
    (h_asymptote : y = √2 * x) : 2 * y^2 - 4 * x^2 = 1 :=
sorry

end hyperbola_equation_l8_8835


namespace area_of_given_rhombus_perimeter_of_given_rhombus_l8_8350

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ  -- First diagonal
  d2 : ℝ  -- Second diagonal
  perp_bisectors : True  -- They are perpendicular bisectors

-- Define the given rhombus
def given_rhombus : Rhombus :=
  { d1 := 6, d2 := 8, perp_bisectors := trivial }

-- Proposition to prove the area
theorem area_of_given_rhombus : (given_rhombus.d1 * given_rhombus.d2 / 2) = 24 := 
sorry

-- Proposition to prove the perimeter
theorem perimeter_of_given_rhombus : 
  let side := Real.sqrt ((given_rhombus.d1 / 2)^2 + (given_rhombus.d2 / 2)^2)
  in 4 * side = 20 :=
sorry

end area_of_given_rhombus_perimeter_of_given_rhombus_l8_8350


namespace right_angle_triangle_rotation_generates_solid_l8_8855

theorem right_angle_triangle_rotation_generates_solid :
  ∀ (T : Triangle), is_right_angle T → 
  (∀ (S : Surface), S = rotate T (perpendicular_side T) → 
  generates_solid S) :=
by
  intros T hT S hS
  sorry

end right_angle_triangle_rotation_generates_solid_l8_8855


namespace net_profit_is_correct_l8_8811

-- Define the conditions
def selling_price_house_1 : ℝ := 10000
def selling_price_house_2 : ℝ := 10000
def profit_percentage_house_1 : ℝ := 0.30
def loss_percentage_house_2 : ℝ := 0.10

-- Calculate cost price of the 1st house
def cost_price_house_1 : ℝ := selling_price_house_1 / (1 + profit_percentage_house_1)

-- Calculate profit from the 1st house
def profit_house_1 : ℝ := selling_price_house_1 - cost_price_house_1

-- Calculate cost price of the 2nd house
def cost_price_house_2 : ℝ := selling_price_house_2 / (1 - loss_percentage_house_2)

-- Calculate loss from the 2nd house
def loss_house_2 : ℝ := cost_price_house_2 - selling_price_house_2

-- Calculate net profit or loss
def net_profit_or_loss : ℝ := profit_house_1 - loss_house_2

-- Prove that the net profit or loss is 1196.58
theorem net_profit_is_correct : net_profit_or_loss = 1196.58 :=
by 
  -- sorry statement to be filled in by proof
  sorry

end net_profit_is_correct_l8_8811


namespace sum_nonpositive_inequality_l8_8822

theorem sum_nonpositive_inequality (x : ℝ) : x + 5 ≤ 0 ↔ x + 5 ≤ 0 :=
by
  sorry

end sum_nonpositive_inequality_l8_8822


namespace gcd_exponentiation_l8_8025

def m : ℕ := 2^2050 - 1
def n : ℕ := 2^2040 - 1

theorem gcd_exponentiation : Nat.gcd m n = 1023 := by
  sorry

end gcd_exponentiation_l8_8025


namespace percentage_with_repeated_digits_l8_8215

theorem percentage_with_repeated_digits :
  let num_total := 900
  let num_without_repeat := (9 * 9 * 8)
  let num_with_repeat := (num_total - num_without_repeat)
  let y := ((num_with_repeat: ℝ / num_total) * 100).round
  y = 28.0 :=
by
  -- Placeholder for the actual proof
  sorry

end percentage_with_repeated_digits_l8_8215


namespace overlapping_area_of_congruent_isosceles_triangles_l8_8019

noncomputable def isosceles_right_triangle (hypotenuse : ℝ) := 
  {l : ℝ // l = hypotenuse / Real.sqrt 2}

theorem overlapping_area_of_congruent_isosceles_triangles (hypotenuse : ℝ) 
  (A₁ A₂ : isosceles_right_triangle hypotenuse) (h_congruent : A₁ = A₂) :
  hypotenuse = 10 → 
  let leg := hypotenuse / Real.sqrt 2 
  let area := (leg * leg) / 2 
  let shared_area := area / 2 
  shared_area = 12.5 :=
by
  sorry

end overlapping_area_of_congruent_isosceles_triangles_l8_8019


namespace class_student_count_l8_8604

theorem class_student_count 
  (average_height_of_40_girls : ℝ)
  (average_height_of_remaining_girls : ℝ)
  (average_height_of_class : ℝ)
  (num_40_girls : ℕ)
  (total_class_height : ℝ)
  (N R : ℕ) :
  average_height_of_40_girls = 169 →
  average_height_of_remaining_girls = 167 →
  average_height_of_class = 168.6 →
  num_40_girls = 40 →
  total_class_height = N * average_height_of_class →
  (40 * 169 + R * 167 = N * 168.6) →
  N = 40 + R →
  N = 50 :=
by
  intros
  sorry

end class_student_count_l8_8604


namespace experimental_value_of_pi_l8_8806

theorem experimental_value_of_pi 
  (side_length : ℝ)
  (n m : ℕ)
  (h_nonzero : n ≠ 0)
  (h_side_length_3 : side_length = 3)
  (h_distance_condition : ∀ (p : ℝ × ℝ) (d : ℝ), d = 1 → 
      p.1 = 0 ∨ p.1 = 3 ∨ p.2 = 0 ∨ p.2 = 3 → 
      m = ∑ (x in finset.range n), ite ((p.1 - x)^2 + (p.2 - x)^2 < d^2) 1 0)
  : Real.pi = (9 * (m : ℝ)) / (4 * (n : ℝ)) := 
sorry

end experimental_value_of_pi_l8_8806


namespace product_of_roots_l8_8480

theorem product_of_roots : (∀ x : ℝ, (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 5) → x ∈ {r : ℝ | r * (x) = 3}) :=
begin
  sorry,
end

end product_of_roots_l8_8480


namespace range_of_x_l8_8194

def f (x : ℝ) : ℝ :=
  log (x^2 + exp (-1)) - abs (x / exp 1)

theorem range_of_x :
  ∀ x, 0 < x ∧ x < 2 → f (x + 1) < f (2 * x - 1) :=
by
  -- provide proof here
  sorry

end range_of_x_l8_8194


namespace sweet_numbers_count_l8_8043

-- Define the conditions as functions in Lean
def sequence_rule (n : ℕ) : ℕ :=
  if n ≤ 20 then 2 * n else n - 15

def is_sweet_number (G : ℕ) : Prop :=
  let rec generate (n : ℕ) : ℕ :=
    if n = 18 then 18
    else if n ≤ 20 then generate (2 * n)
    else generate (n - 15)
  in generate G ≠ 18

-- The Lean 4 statement of the problem
theorem sweet_numbers_count : {n : ℕ | 1 ≤ n ∧ n ≤ 50 ∧ is_sweet_number n}.card = 34 := by
  sorry

end sweet_numbers_count_l8_8043


namespace geometric_sequence_a4_value_l8_8228

variable {α : Type} [LinearOrderedField α]

noncomputable def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, n < m → ∃ r : α, 0 < r ∧ a m = a n * r^(m - n)

theorem geometric_sequence_a4_value (a : ℕ → α)
  (pos : ∀ n, 0 < a n)
  (geo_seq : is_geometric_sequence a)
  (h : a 1 * a 7 = 36) :
  a 4 = 6 :=
by 
  sorry

end geometric_sequence_a4_value_l8_8228


namespace tangent_line_is_y2_or_3x4y8_no_suitable_k_l8_8533

noncomputable def circle_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 12 * x + 32 = 0

noncomputable def line_eq (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y = k * x + 2

noncomputable def is_tangent (k : ℝ) : Prop :=
  let d := (abs (6 * k + 2)) / (real.sqrt (1 + k^2)) in
  d = 2

noncomputable def center := (6, 0 : ℝ)
noncomputable def radius := 2

noncomputable def tangent_lines : set (ℝ → ℝ → Prop) := 
  {f | ∃k, f = line_eq k ∧ is_tangent k}

theorem tangent_line_is_y2_or_3x4y8 : 
  tangent_lines = {λ x y, y = 2, 
                   λ x y, 3 * x + 4 * y - 8 = 0} :=
sorry

noncomputable def collinearity (k : ℝ) (A B P Q : ℝ × ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (px, py) := P in
  let (qx, qy) := Q in
  -2 * (x1 + x2) = 6 * (y1 + y2)

theorem no_suitable_k : 
  ¬∃ (k : ℝ) (A B : ℝ × ℝ), 
    let l := line_eq k in 
    ∀t, l t → ∃ (x y), circle_eq x y ∧ collinearity k (x, y) A B center:= 
sorry

end tangent_line_is_y2_or_3x4y8_no_suitable_k_l8_8533


namespace part1_max_area_part2_find_a_l8_8979

-- Part (1): Define the function and prove maximum area of the triangle
noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp x - 3 * a * x + 2 * Real.sin x - 1

theorem part1_max_area (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f' := a^2 - 3 * a + 2
  ∃ h_a_max, h_a_max == 3 / 8 :=
  sorry

-- Part (2): Prove that the function reaches an extremum at x = 0 and determine the value of a.
theorem part2_find_a (a : ℝ) : (a^2 - 3 * a + 2 = 0) → (a = 1 ∨ a = 2) :=
  sorry

end part1_max_area_part2_find_a_l8_8979


namespace part1_part2_l8_8665

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end part1_part2_l8_8665


namespace domain_of_function_l8_8354

/-- Prove the domain of the function f(x) = log10(2 * cos x - 1) + sqrt(49 - x^2) -/
theorem domain_of_function :
  { x : ℝ | -7 ≤ x ∧ x < - (5 * Real.pi) / 3 ∨ - Real.pi / 3 < x ∧ x < Real.pi / 3 ∨ (5 * Real.pi) / 3 < x ∧ x ≤ 7 }
  = { x : ℝ | 2 * Real.cos x - 1 > 0 ∧ 49 - x^2 ≥ 0 } :=
by {
  sorry
}

end domain_of_function_l8_8354


namespace smallest_n_l8_8391

-- Definitions based on conditions
def expression (n : ℕ) : ℤ := 9 * (n - 2) ^ 6 - n ^ 3 + 20 * n - 48

def is_multiple_of_7 (n : ℕ) : Prop := expression n % 7 = 0

-- The proof problem statement
theorem smallest_n (h : 50000 < 50001) : ∃ n > 50000, is_multiple_of_7 n ∧ n = 50001 :=
by
  have h2 : 50000 < 50001 := by exact h
  existsi 50001
  split
  repeat { sorry }

end smallest_n_l8_8391


namespace lcm_of_two_numbers_l8_8709

theorem lcm_of_two_numbers (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_sum : a + b = 30) : Nat.lcm a b = 18 :=
  sorry

end lcm_of_two_numbers_l8_8709


namespace lumberjack_trees_chopped_l8_8837

-- Statement of the problem in Lean 4
theorem lumberjack_trees_chopped
  (logs_per_tree : ℕ) 
  (firewood_per_log : ℕ) 
  (total_firewood : ℕ) 
  (logs_per_tree_eq : logs_per_tree = 4) 
  (firewood_per_log_eq : firewood_per_log = 5) 
  (total_firewood_eq : total_firewood = 500)
  : (total_firewood / firewood_per_log) / logs_per_tree = 25 := 
by
  rw [total_firewood_eq, firewood_per_log_eq, logs_per_tree_eq]
  norm_num
  sorry

end lumberjack_trees_chopped_l8_8837


namespace lumberjack_trees_l8_8843

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l8_8843


namespace total_profit_is_92000_l8_8409

def investment_a : ℚ := 24000
def investment_b : ℚ := 32000
def investment_c : ℚ := 36000
def profit_c : ℚ := 36000

theorem total_profit_is_92000 (total_profit : ℚ) : 
  ((investment_c / (investment_a + investment_b + investment_c)) * total_profit = profit_c) → 
  total_profit = 92000 :=
by
  sorry

end total_profit_is_92000_l8_8409


namespace compute_f_1986_l8_8669

noncomputable def f : ℕ → ℝ := sorry
def g (x : ℕ) : ℝ := f x + x

lemma f_initial : f 1 = 1 := sorry

lemma f_functional (a b : ℕ) : f (a + b) = f a + f b + g (a * b) - 1 := sorry

theorem compute_f_1986 : f 1986 = -991.5 := sorry

end compute_f_1986_l8_8669


namespace ice_cream_combinations_l8_8997

-- Definition and question rephrased into a Lean definition
theorem ice_cream_combinations : (nat.choose 8 3) = 56 := by
  sorry

end ice_cream_combinations_l8_8997


namespace floor_2_7_l8_8363

def floor_function_property (x : ℝ) : ℤ := Int.floor x

theorem floor_2_7 : floor_function_property 2.7 = 2 :=
by
  sorry

end floor_2_7_l8_8363


namespace min_distance_between_curves_l8_8569

noncomputable def distance_between_intersections : ℝ :=
  let f (x : ℝ) := (2 * x + 1) - (x + Real.log x)
  let f' (x : ℝ) := 1 - 1 / x
  let minimum_distance :=
    if hs : 1 < 1 then 2 else
    if hs : 1 > 1 then 2 else
    2
  minimum_distance

theorem min_distance_between_curves : distance_between_intersections = 2 :=
by
  sorry

end min_distance_between_curves_l8_8569


namespace medial_lines_concyclic_points_l8_8303

-- Define points D, E, F as midpoints of sides of triangle ABC
variables {A B C : Point}
variables (D E F : Point) (hD : midpoint D A B) (hE : midpoint E A C) (hF : midpoint F B C)

-- Define points I_A, I_B, I_C as excenters of triangle ABC
variables (I_A I_B I_C : Point) 
          (hI_A : excenter I_A α β γ)
          (hI_B : excenter I_B β γ α)
          (hI_C : excenter I_C γ α β)

-- Define the intersection points M, L, J, O, K, N
variables (M L J O K N : Point)
          (hM : intersect M (line_through D E) (line_through I_A I_C))
          (hL : intersect L (line_through D E) (line_through I_A I_B))
          (hJ : intersect J (line_through D F) (line_through I_C I_B))
          (hO : intersect O (line_through D F) (line_through I_A I_B))
          (hK : intersect K (line_through E F) (line_through I_C I_B))
          (hN : intersect N (line_through E F) (line_through I_C I_A))

-- Prove that points M, L, J, O, K, N are concyclic
theorem medial_lines_concyclic_points : concyclic M L J O K N :=
by
  sorry

end medial_lines_concyclic_points_l8_8303


namespace correctly_calculated_equation_l8_8397

theorem correctly_calculated_equation {a b c : ℝ} :
  (2 * (2 * b - 1) ≠ 4 * b - 1) ∧
  (3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  (6 * a - 5 * a ≠ 1) ∧
  (a - (2 * b - 3 * c) ≠ a + 2 * b - 3 * c) := 
by 
  split; 
  linarith; 
  sorry; 
  linarith; 
  linarith

end correctly_calculated_equation_l8_8397


namespace subset_with_three_colors_l8_8621

def telephone_graph (n : ℕ) (c : ℕ) (colors_present : ℕ → Bool) :=
  n = 2004 ∧ c = 4 ∧ colors_present 4

theorem subset_with_three_colors (n : ℕ) (c : ℕ) (colors_present : ℕ → Bool) : telephone_graph n c colors_present → ∃ (S : Finset ℕ), (∀ (i j ∈ S), i ≠ j → ∃ k, k ∈ Finset.range c ∧ colors_present k) ∧ (Finset.card (Finset.filter colors_present (Finset.range c)) = 3) :=
by
  sorry

end subset_with_three_colors_l8_8621


namespace inverse_and_eigenvalues_l8_8572

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; -1, 4]

theorem inverse_and_eigenvalues (A_inv : Matrix (Fin 2) (Fin 2) ℚ)
  (λ1 λ2 : ℚ) (α1 α2 : Fin 2 → ℚ) :
  A_inv = !![2/3, -1/3; 1/6, 1/6] ∧
  λ1 = 2 ∧ λ2 = 3 ∧
  α1 = !![2, 1] ∧ α2 = !![1, 1] :=
sorry

end inverse_and_eigenvalues_l8_8572


namespace modulus_conjugate_l8_8560

theorem modulus_conjugate (z : ℂ) (h : z = (3 + 1 * complex.I) / (1 - 1 * complex.I)) : complex.abs (complex.conj z) = real.sqrt 5 :=
by
  -- Sorry to skip the proof
  sorry

end modulus_conjugate_l8_8560


namespace polynomial_sum_l8_8656

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l8_8656


namespace number_of_trees_after_cutting_and_planting_l8_8095

theorem number_of_trees_after_cutting_and_planting 
    (initial_trees : ℕ) (cut_percentage : ℝ) (trees_planted_per_cut : ℕ) : 
    initial_trees = 400 →
    cut_percentage = 0.20 →
    trees_planted_per_cut = 5 →
    ∃ final_trees : ℕ, final_trees = 720 :=
by
  intros initial_trees_eq cut_percentage_eq trees_planted_per_cut_eq
  
  let trees_cut := (cut_percentage * initial_trees.to_real).nat_abs
  have trees_cut_eq : trees_cut = 80,
  { sorry },

  let remaining_trees := initial_trees - trees_cut
  have remaining_trees_eq : remaining_trees = 320,
  { sorry },

  let new_trees_planted := trees_cut * trees_planted_per_cut
  have new_trees_planted_eq : new_trees_planted = 400,
  { sorry },

  let final_trees := remaining_trees + new_trees_planted
  have final_trees_eq : final_trees = 720,
  { sorry },

  use final_trees
  exact final_trees_eq

end number_of_trees_after_cutting_and_planting_l8_8095


namespace joey_studies_2_hours_per_night_l8_8257

def joey_nightly_study_hours (x : ℝ) : Prop :=
  let weekend_study_hours := 3 * 2 * 6
  let total_hours := 96
  let weekday_total_hours := total_hours - weekend_study_hours
  let weekly_weekday_hours := weekday_total_hours / 6
  let nightly_weekday_hours := weekly_weekday_hours / 5
  nightly_weekday_hours = x

theorem joey_studies_2_hours_per_night :
  joey_nightly_study_hours 2 :=
by
  let weekend_study_hours := 3 * 2 * 6
  let total_hours := 96
  let weekday_total_hours := total_hours - weekend_study_hours
  let weekly_weekday_hours := weekday_total_hours / 6
  let nightly_weekday_hours := weekly_weekday_hours / 5
  show nightly_weekday_hours = 2 from sorry

end joey_studies_2_hours_per_night_l8_8257


namespace part1_part2_part3a_part3b_l8_8532

noncomputable def f (a x : ℝ) : ℝ := Real.log x + x^2 + a * x

-- (Ⅰ) If f(x) attains an extreme value at x = 1/2, show that a = -3
theorem part1 (a : ℝ) (h : ∃ c : ℝ, c = 1/2 ∧ deriv (f a) c = 0) : a = -3 := by
  sorry

-- (Ⅱ) If f(x) is an increasing function within its domain, show that the range of a is [-2*sqrt(2), +∞)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) : -2 * Real.sqrt 2 ≤ a := by
  sorry

-- (Ⅲ) Show g(x) = ln x - x + 1 with a = -1 has g(x) ≤ 0 within its domain, and another inequality about logarithms
noncomputable def g (x : ℝ) : ℝ := Real.log x - x + 1

theorem part3a : ∀ x : ℝ, 0 < x → g x ≤ 0 := by
  sorry

theorem part3b (n : ℕ) (h : 2 ≤ n) :
  ∑ k in Finset.range (n - 1) + 2, Real.log k.succ.succ^2 / k.succ.succ^2 < (2 * n^2 - n - 1) / (2 * (n + 1)) := by
  sorry

end part1_part2_part3a_part3b_l8_8532


namespace certain_number_l8_8023

theorem certain_number (x : ℤ) (h : 12 + x = 27) : x = 15 :=
by
  sorry

end certain_number_l8_8023


namespace price_of_each_lemon_square_l8_8252

-- Given
def brownies_sold : Nat := 4
def price_per_brownie : Nat := 3
def lemon_squares_sold : Nat := 5
def goal_amount : Nat := 50
def cookies_sold : Nat := 7
def price_per_cookie : Nat := 4

-- Prove
theorem price_of_each_lemon_square :
  (brownies_sold * price_per_brownie + lemon_squares_sold * L + cookies_sold * price_per_cookie = goal_amount) →
  L = 2 :=
by
  sorry

end price_of_each_lemon_square_l8_8252


namespace simplify_and_rationalize_l8_8700

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l8_8700


namespace solve_system_correct_l8_8707

noncomputable def solve_system (a b c d e : ℝ) : Prop :=
  3 * a = (b + c + d) ^ 3 ∧ 
  3 * b = (c + d + e) ^ 3 ∧ 
  3 * c = (d + e + a) ^ 3 ∧ 
  3 * d = (e + a + b) ^ 3 ∧ 
  3 * e = (a + b + c) ^ 3

theorem solve_system_correct :
  ∀ (a b c d e : ℝ), solve_system a b c d e → 
    (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨ 
    (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨ 
    (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3) :=
by
  sorry

end solve_system_correct_l8_8707


namespace minesweeper_A_minesweeper_B_minesweeper_C_minesweeper_D_l8_8607

-- Define the grid as a 2D array with known numbers indicating mine counts around specific cells.
def grid := array (9 × 6) (option ℕ)

-- Define what it means to have a minefield and analyze the sections A, B, C, D.
def minesweeper_problem (g : grid) : Prop :=
  ∃ (count_A count_B count_C count_D : ℕ),
    (count_A = 7) ∧
    (count_B = 8) ∧
    (count_C = 9) ∧
    (count_D = 10)

-- Part A grid configuration and mine count.
def grid_A : grid := sorry  -- Assume we have the specific grid configuration for part A.
def count_mines_A := 7
theorem minesweeper_A : ∃ g_A : grid, (g_A = grid_A) → (count_mines_A = 7) := sorry

-- Part B grid configuration and mine count.
def grid_B : grid := sorry  -- Assume we have the specific grid configuration for part B.
def count_mines_B := 8
theorem minesweeper_B : ∃ g_B : grid, (g_B = grid_B) → (count_mines_B = 8) := sorry

-- Part C grid configuration and mine count.
def grid_C : grid := sorry  -- Assume we have the specific grid configuration for part C.
def count_mines_C := 9
theorem minesweeper_C : ∃ g_C : grid, (g_C = grid_C) → (count_mines_C = 9) := sorry

-- Part D grid configuration and mine count.
def grid_D : grid := sorry  -- Assume we have the specific grid configuration for part D.
def count_mines_D := 10
theorem minesweeper_D : ∃ g_D : grid, (g_D = grid_D) → (count_mines_D = 10) := sorry

end minesweeper_A_minesweeper_B_minesweeper_C_minesweeper_D_l8_8607


namespace area_of_KLMN_l8_8818

-- Define points and lengths in the triangle
def AB : ℝ := 21
def BC : ℝ := 20
def AC : ℝ := 13
def AK : ℝ := 1
def CN : ℝ := 12
def AL_ratio : ℝ := 13 / 21

-- Define points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Coordinates for points A, B, and C
def A : Point := ⟨0, 0⟩
def B : Point := ⟨21, 0⟩
noncomputable def C : Point := 
  let x_C := 5
  let y_C := 12
  in ⟨x_C, y_C⟩

-- Define points on sides based on given conditions
noncomputable def N : Point := ⟨13, 7.2⟩
noncomputable def K : Point := ⟨1, 12⟩
noncomputable def L : Point := ⟨13, 0⟩ -- This should be placed correctly with provided ratio

-- Hypothetical point M would be calculated based on geometric transformations but not directly needed here

-- Define area calculation (using some formula for quadrilateral)
noncomputable def quadrilateral_area (p1 p2 p3 p4 : Point) : ℝ := 
  -- Placeholder for the actual area calculation
  14236 / 325

-- The goal is to prove the area of KLMN is 14236 / 325
theorem area_of_KLMN : 
  quadrilateral_area K L M N = 14236 / 325 :=
sorry

end area_of_KLMN_l8_8818


namespace mass_percentage_H_correct_l8_8027

/-- Definitions of the molar masses of elements -/
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

/-- Chemical formula for Citric Acid -/
def citric_acid_formula : string := "C6H8O7"

/-- Molar mass calculation for C6H8O7 -/
def molar_mass_C6H8O7 : ℝ := (6 * molar_mass_C) + (8 * molar_mass_H) + (7 * molar_mass_O)

/-- Total mass of Hydrogen in one mole of C6H8O7 -/
def total_mass_H_in_C6H8O7 : ℝ := 8 * molar_mass_H

/-- Mass percentage calculation for Hydrogen in C6H8O7 -/
def mass_percentage_H_in_C6H8O7 : ℝ := (total_mass_H_in_C6H8O7 / molar_mass_C6H8O7) * 100

/-- Theorem: mass percentage of Hydrogen in C6H8O7 is 4.20%. -/
theorem mass_percentage_H_correct : mass_percentage_H_in_C6H8O7 = 4.20 := 
by
  -- Proof goes here
  sorry

end mass_percentage_H_correct_l8_8027


namespace exists_tangent_circle_l8_8955

variable {P : Type} [MetricSpace P]

-- Definitions of circles and quadrilateral
structure Circle (P: Type) := 
  (center : P)
  (radius : ℝ)

structure Quadrilateral (P: Type) :=
  (A B C D : P)

-- Given conditions
def is_inscribable (Q : Quadrilateral P) : Prop := sorry
def has_diameter_circles (Q : Quadrilateral P) 
  (o1 o2 o3 o4 : Circle P) : Prop := sorry

-- Main theorem statement
theorem exists_tangent_circle (Q : Quadrilateral P)
  (o1 o2 o3 o4 : Circle P)
  (h1 : is_inscribable Q)
  (h2 : has_diameter_circles Q o1 o2 o3 o4) : 
  ∃ k : Circle P, ∀ i, 
    (i = o1 ∨ i = o2 ∨ i = o3 ∨ i = o4) → k.center.dist i.center = k.radius + i.radius :=
sorry

end exists_tangent_circle_l8_8955


namespace find_tangent_c_l8_8160

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → (c = 1) :=
by
  intros h
  sorry

end find_tangent_c_l8_8160


namespace circumradius_product_equality_l8_8272

variables {A B C P A1 B1 C1 : Type*} [inhabited A] [inhabited B] [inhabited C] [inhabited P] [inhabited A1] [inhabited B1] [inhabited C1]

def circumradius {A B C : Type*} [inhabited A] [inhabited B] [inhabited C] (a b c : A) : ℝ := sorry

theorem circumradius_product_equality 
  (h1 : P ∈ interior (triangle A B C))
  (h2 : A1 ∈ interior (segment B C))
  (h3 : B1 ∈ interior (segment C A))
  (h4 : C1 ∈ interior (segment A B)) :
  (circumradius A C1 P) * (circumradius B A1 P) * (circumradius C B1 P) = 
  (circumradius C1 B P) * (circumradius A1 C P) * (circumradius B1 A P) :=
sorry

end circumradius_product_equality_l8_8272


namespace sum_arithmetic_sequence_l8_8926

theorem sum_arithmetic_sequence (n : ℕ) : 
  ∑ k in Finset.range n, (3 * k + 2) = (3 * n^2 + n) / 2 := 
by 
  sorry

end sum_arithmetic_sequence_l8_8926


namespace rectangular_to_cylindrical_l8_8127

theorem rectangular_to_cylindrical (x y z : ℝ) (hx : x = 3) (hy : y = -3 * sqrt 3) (hz : z = 2) :
  ∃ (r θ : ℝ), r = 6 ∧ θ = 5 * π / 3 ∧ z = 2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π := 
by {
  sorry
}

end rectangular_to_cylindrical_l8_8127


namespace angle_C_measure_triangle_perimeter_l8_8555

-- Define the problem conditions
variables {A B C a b c : ℝ}
variables (h1 : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)

-- Part (1): Prove measure of angle C
theorem angle_C_measure : 
  (∀ (a b c : ℝ) (A B C : ℝ), 
      a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B → 
      C = Real.pi / 3) :=
sorry

-- Conditions for Part (2)
variables (h2 : c = 2 * Real.sqrt 3)
variables (h3 : let S := 2 * Real.sqrt 3 in S = 1/2 * a * b * Real.sin (Real.pi / 3))

-- Part (2): Prove perimeter of triangle ABC
theorem triangle_perimeter :
  (∀ (a b c : ℝ), c = 2 * Real.sqrt 3 → 
      (let S := 2 * Real.sqrt 3 in S = 1/2 * a * b * Real.sin (Real.pi / 3)) → 
      a + b + c = 6 + 2 * Real.sqrt 3) :=
sorry

end angle_C_measure_triangle_perimeter_l8_8555


namespace intersection_union_complement_l8_8576

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def universal_set := U = univ
def set_A := A = {x : ℝ | -1 ≤ x ∧ x < 2}
def set_B := B = {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

theorem union (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry

theorem complement (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) :
  U \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := sorry

end intersection_union_complement_l8_8576


namespace third_prize_probability_winning_prize_probability_l8_8234

noncomputable def probability_winning_third_prize : ℚ :=
  1 / 4

noncomputable def probability_winning_prize : ℚ :=
  9 / 16

theorem third_prize_probability (draws : Finset (ℕ × ℕ))
    (balls : Fin 4)
    (draws_considered : Finset.univ = {(0, 3), (1, 2), (2, 1), (3, 0)} ) :
    probability_winning_third_prize = (4 / 16 : ℚ) := 
  by sorry

theorem winning_prize_probability (draws : Finset (ℕ × ℕ))
    (balls : Fin 4)
    (draws_considered : Finset.univ = {(0, 3), (1, 2), (2, 1), (3, 0), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2)}) :
    probability_winning_prize = (9 / 16 : ℚ) := 
  by sorry

end third_prize_probability_winning_prize_probability_l8_8234


namespace simplify_fraction_sum_l8_8276

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end simplify_fraction_sum_l8_8276


namespace cyclic_quadrilateral_inequality_l8_8647

theorem cyclic_quadrilateral_inequality 
  (AB CD AD BC AC BD : ℝ) 
  (h_cyclic : is_cyclic_quadrilateral AB CD AD BC AC BD) : 
  |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| :=
sorry

end cyclic_quadrilateral_inequality_l8_8647


namespace quadratic_has_real_roots_l8_8724

theorem quadratic_has_real_roots (a : ℝ) : 
  ∃ x : ℝ, x^2 + a * x + (a - 1) = 0 :=
by
  let Δ := a^2 - 4 * 1 * (a - 1)
  have hΔ : Δ = (a - 2)^2 := by ring
  have hΔ_nonneg : Δ ≥ 0 := by { rw hΔ, apply pow_two_nonneg }
  sorry

end quadratic_has_real_roots_l8_8724


namespace proposition_correct_l8_8400

theorem proposition_correct :
  (¬(∀ x : ℝ, x^2 = 1 → x = 1) ∨ (∀ x : ℝ, x^2 = 1 → x ≠ 1)) ∧
  (¬(∀ x : ℝ, x = -1 → x^2 - 5 * x - 6 = 0) ∨ (∀ x : ℝ, x = -1 → x^2 - 5 * x - 6 = 0)) ∧
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ∨ (∀ x : ℝ, x^2 + x + 1 < 0)) →
  (∀ x y : ℝ, x = y → sin x = sin y) :=
begin
  sorry
end

end proposition_correct_l8_8400


namespace find_x_solutions_l8_8072

theorem find_x_solutions :
  ∀ {x : ℝ}, (x = (1/x) + (-x)^2 + 3) → (x = -1 ∨ x = 1) :=
by
  sorry

end find_x_solutions_l8_8072


namespace avg_divisible_by_4_between_15_and_55_eq_34_l8_8878

theorem avg_divisible_by_4_between_15_and_55_eq_34 :
  let numbers := (List.filter (λ x => x % 4 = 0) (List.range' 16 37))
  (List.sum numbers) / (numbers.length) = 34 := by
  sorry

end avg_divisible_by_4_between_15_and_55_eq_34_l8_8878


namespace intervals_union_l8_8643

open Set

noncomputable def I (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

theorem intervals_union {I1 I2 I3 : Set ℝ} (h1 : ∃ (a1 b1 : ℝ), I1 = I a1 b1)
  (h2 : ∃ (a2 b2 : ℝ), I2 = I a2 b2) (h3 : ∃ (a3 b3 : ℝ), I3 = I a3 b3)
  (h_non_empty : (I1 ∩ I2 ∩ I3).Nonempty) (h_not_contained : ¬ (I1 ⊆ I2) ∧ ¬ (I1 ⊆ I3) ∧ ¬ (I2 ⊆ I1) ∧ ¬ (I2 ⊆ I3) ∧ ¬ (I3 ⊆ I1) ∧ ¬ (I3 ⊆ I2)) :
  I1 ⊆ (I2 ∪ I3) ∨ I2 ⊆ (I1 ∪ I3) ∨ I3 ⊆ (I1 ∪ I2) :=
sorry

end intervals_union_l8_8643


namespace largest_multiple_of_8_less_than_100_l8_8760

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8760


namespace interval_monotonic_decrease_and_period_and_range_k_l8_8580

noncomputable def vector_m (x : ℝ) : ℝ × ℝ :=
  (√3 * Real.sin (x / 4), 1)

noncomputable def vector_n (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x / 4), Real.cos (x / 4) ^ 2)

noncomputable def f (x : ℝ) : ℝ :=
  let m := vector_m x
  let n := vector_n x
  m.1 * n.1 + m.2 * n.2

theorem interval_monotonic_decrease_and_period_and_range_k :
  ∃ (k : ℕ),
    (∀ x : ℝ, (4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3) → (Real.sin (x / 2 + π / 6) + 1 / 2) = f x) ∧
    (∃ T : ℝ, 0 < T ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * π) ∧
    (∀ k : ℝ, 0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ (set.Icc (0 : ℝ) (7 * π / 3)), Real.sin (x / 2 - π / 6) + 1 / 2 - k = 0) :=
sorry

end interval_monotonic_decrease_and_period_and_range_k_l8_8580


namespace inequality_sqrt_a_b_c_l8_8689

noncomputable def sqrt (x : ℝ) := x ^ (1 / 2)

theorem inequality_sqrt_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a ^ (1 - a) * b ^ (1 - b) * c ^ (1 - c)) ≤ 1 / 3 := 
sorry

end inequality_sqrt_a_b_c_l8_8689


namespace calculation_result_l8_8028

theorem calculation_result :
  3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 :=
sorry

end calculation_result_l8_8028


namespace profit_percent_300_l8_8414

theorem profit_percent_300 (SP : ℝ) (h : SP ≠ 0) (CP : ℝ) (h1 : CP = 0.25 * SP) : 
  (SP - CP) / CP * 100 = 300 := 
  sorry

end profit_percent_300_l8_8414


namespace non_neg_int_solutions_l8_8361

-- Define the property of being a non-negative integer solution
def non_neg_int_solution (x : ℕ) : Prop := x + 1 < 4

-- Define the set of non-negative integer solutions
def solution_set : set ℕ := { x | non_neg_int_solution x }

-- State the main theorem
theorem non_neg_int_solutions {x : ℕ} : x ∈ solution_set ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end non_neg_int_solutions_l8_8361


namespace workers_complete_job_together_in_time_l8_8041

theorem workers_complete_job_together_in_time :
  let work_rate_A := 1 / 10 
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  time = 60 / 13 :=
by
  let work_rate_A := 1 / 10
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  sorry

end workers_complete_job_together_in_time_l8_8041


namespace subtract_from_sum_base8_l8_8153

def add_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) + (b % 8)) % 8
  + (((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) % 8) * 8
  + (((((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) / 8) + ((a / 64) % 8 + (b / 64) % 8)) % 8) * 64

def subtract_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) - (b % 8) + 8) % 8
  + (((a / 8) % 8 - (b / 8) % 8 - if (a % 8) < (b % 8) then 1 else 0 + 8) % 8) * 8
  + (((a / 64) - (b / 64) - if (a / 8) % 8 < (b / 8) % 8 then 1 else 0) % 8) * 64

theorem subtract_from_sum_base8 :
  subtract_in_base_8 (add_in_base_8 652 147) 53 = 50 := by
  sorry

end subtract_from_sum_base8_l8_8153


namespace range_of_a_l8_8984

theorem range_of_a 
  (a : ℤ) 
  (ineq1 : ∀ x : ℤ, x - a ≤ 0)
  (ineq2 : ∀ x : ℤ, 7 + 2 * x > 1)
  (five_int_solutions : ∃ (x : ℤ), 5 = (↑(-3) < x ∧ x ≤ ↑a)) : 
  2 ≤ a ∧ a < 3 := 
begin
  sorry
end

end range_of_a_l8_8984


namespace response_rate_increase_l8_8452

-- Define the variables for the conditions
variables (n1 n2 r1 r2 : ℕ)
-- Condition 1: Number of customers surveyed in the original survey is 80
axiom h1 : n1 = 80
-- Condition 2: Number of respondents in the original survey is 7
axiom h2 : r1 = 7
-- Condition 3: Number of customers surveyed in the redesigned survey is 63
axiom h3 : n2 = 63
-- Condition 4: Number of respondents in the redesigned survey is 9
axiom h4 : r2 = 9

-- Define the response rates
def response_rate (respondents : ℕ) (surveyed : ℕ) : ℝ :=
  (respondents.to_real / surveyed.to_real) * 100

-- Define the proof theorem statement
theorem response_rate_increase (n1 n2 r1 r2 : ℕ) 
  (h1 : n1 = 80) (h2 : r1 = 7) (h3 : n2 = 63) (h4 : r2 = 9) : 
  (response_rate r2 n2 - response_rate r1 n1) / response_rate r1 n1 * 100 ≈ 63.24 :=
  sorry

end response_rate_increase_l8_8452


namespace total_sacks_needed_l8_8870

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l8_8870


namespace sum_of_consecutive_even_numbers_l8_8817

theorem sum_of_consecutive_even_numbers (x : ℤ) (h : (x + 2)^2 - x^2 = 84) : x + (x + 2) = 42 :=
by 
  sorry

end sum_of_consecutive_even_numbers_l8_8817


namespace circle_intersects_line_at_A_and_B_l8_8951

noncomputable def midpoint (A B O : Point) : Prop :=
  O = (A + B) / 2  -- assuming points operate in a vector space.

theorem circle_intersects_line_at_A_and_B 
(O A B : Point) (l : Line)
(h1 : segment OA parallel_to l)
(h2 : circle_centered_at O radius OA)
(h3 : midpoint A B O) :
  intersects (circle O (dist O A)) l = {A, B} :=
by 
  sorry

end circle_intersects_line_at_A_and_B_l8_8951


namespace allison_more_glue_sticks_l8_8863

theorem allison_more_glue_sticks (A M C : ℕ) (Marie_M_sticks:  M = 15) 
(Marie_M_paper: M_paper = 30) 
(paper_ratio: M_paper = 6 * C)
(total_items: A + C = 28) : 
  A - M = 8 :=
begin
  -- proof to be filled
  sorry
end

end allison_more_glue_sticks_l8_8863


namespace longest_segment_CD_l8_8894
open_locale big_operators

noncomputable def A : ℝ × ℝ := (-2, 1)
noncomputable def B : ℝ × ℝ := (2, 2)
noncomputable def C : ℝ × ℝ := (3, -1)
noncomputable def D : ℝ × ℝ := (0, -2)

-- Given angles in degrees
noncomputable def angle_DAB : ℝ := 30
noncomputable def angle_ABC : ℝ := 85
noncomputable def angle_BCD : ℝ := 60
noncomputable def angle_CDA : ℝ := 55

-- Define the line segments
noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def AB : ℝ := length A B
noncomputable def BC : ℝ := length B C
noncomputable def CD : ℝ := length C D
noncomputable def DA : ℝ := length D A
noncomputable def BD : ℝ := length B D

theorem longest_segment_CD :
  max (max (max AB BC) (max CD DA)) BD = CD :=
begin
  sorry
end

end longest_segment_CD_l8_8894


namespace ellipse_and_line_conditions_proof_l8_8557

noncomputable def ellipse_equation_eccentricity : Prop :=
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ c = 1 ∧ a = sqrt 2 ∧
  (x y : ℝ) (e : ℝ := sqrt 2 / 2), 
  e = c / a ∧
  (∃ x y, y^2 = 4 * x ∧ x = 1 ∧ y = 0) ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a = sqrt 2 ∧ b = 1) ∧
  equation_of_ellipse_eq : (x^2 / 2 + y^2 = 1)

noncomputable def line_intersects_ellipse (m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
  y1 = x1 + m ∧ y2 = x2 + m ∧
  (x1^2 / 2 + y1^2 = 1) ∧ (x2^2 / 2 + y2^2 = 1) ∧
  (P : (ℝ × ℝ)), P.1 = x1 + x2 ∧ P.2 = y1 + y2 ∧
  (P.1^2 / 2 + P.2^2 = 1) ∧
  m = sqrt 3 / 2 ∨ m = -sqrt 3 / 2

theorem ellipse_and_line_conditions_proof :
  ellipse_equation_eccentricity ∧ (∀ m, line_intersects_ellipse m) :=
by sorry

end ellipse_and_line_conditions_proof_l8_8557


namespace min_value_l8_8590

theorem min_value (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 :=
sorry

end min_value_l8_8590


namespace range_m_ge_two_l8_8282

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (g : ℝ → ℝ)

noncomputable def has_deriv (f : ℝ → ℝ) :=
  ∀ x : ℝ, ∃ l : ℝ, let Df_x := deriv f in Df_x x = l

def decreasing (g : ℝ → ℝ) :=
  ∀ x y : ℝ, x ≤ y → g x ≥ g y

axiom f_deriv_exists : has_deriv f

axiom f_prime_condition : ∀ x : ℝ, f' x < x

axiom g_def : ∀ x : ℝ, g x = f x - 0.5 * x^2

axiom f_ineq : ∀ m : ℝ, f (4 - m) - f m ≥ 8 - 4 * m

theorem range_m_ge_two : ∀ m : ℝ, f_prime_condition f' → g_def g → f_ineq f → m ≥ 2 :=
by
  sorry

end range_m_ge_two_l8_8282


namespace largest_integer_m_dividing_30_factorial_l8_8508

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l8_8508


namespace radical_axis_passes_through_X_l8_8819

-- Definitions based on conditions
variable (S1 S2 S3 S4 : Circle)

-- Given condition: Circle tangency
axiom tangent_ext (S : Circle) (S' : Circle) : TangentExternally S S'

-- Prove statement
theorem radical_axis_passes_through_X
  (tangency12 : tangent_ext S1 S2)
  (tangency23 : tangent_ext S2 S3)
  (tangency34 : tangent_ext S3 S4)
  (tangency41 : tangent_ext S4 S1) :
  ∃ X : Point, intersect_common_external_tangent S2 S4 X → lies_on_radical_axis X S1 S3 :=
by
  sorry

end radical_axis_passes_through_X_l8_8819


namespace find_s_for_g3_eq_0_l8_8670

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem find_s_for_g3_eq_0 : (g 3 s = 0) ↔ (s = -573) :=
by
  sorry

end find_s_for_g3_eq_0_l8_8670


namespace distinct_pairwise_products_l8_8546

theorem distinct_pairwise_products
  (n a b c d : ℕ) (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bounds: n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2) :
  (a * b ≠ a * c ∧ a * b ≠ a * d ∧ a * b ≠ b * c ∧ a * b ≠ b * d ∧ a * b ≠ c * d) ∧
  (a * c ≠ a * d ∧ a * c ≠ b * c ∧ a * c ≠ b * d ∧ a * c ≠ c * d) ∧
  (a * d ≠ b * c ∧ a * d ≠ b * d ∧ a * d ≠ c * d) ∧
  (b * c ≠ b * d ∧ b * c ≠ c * d) ∧
  (b * d ≠ c * d) :=
sorry

end distinct_pairwise_products_l8_8546


namespace SimplifyAndRationalize_l8_8698

theorem SimplifyAndRationalize :
  ( (√3 / √7) * (√5 / √8) * (√6 / √9) ) = ( √35 / 42 ) :=
sorry

end SimplifyAndRationalize_l8_8698


namespace binomial_expansion_coefficient_l8_8352

theorem binomial_expansion_coefficient :
  (finset.sum (finset.range 5) (λ r, (-2:ℤ)^r * (nat.choose 4 r) * x^r)).coeff 1 = -8 := 
sorry

end binomial_expansion_coefficient_l8_8352


namespace polynomial_divisors_l8_8182

theorem polynomial_divisors (n : ℕ) :
  { (a, b) : ℤ × ℤ // (∃ α β : ℂ, (α + β = -(a:ℂ) ∧ α * β = (b:ℂ)) ∧ 
    (α ^ n = α ∨ α ^ n = β) ∧ (β ^ n = α ∨ β ^ n = β))  } =
  if n % 3 = 0 then { (0, 0), (-1, 0), (1, 1), (-2, 1) }
  else if (n % 6 = 1 ∨ n % 6 = 5) then { (0, 0), (0, 1), (0, -1), (1, 1), (2, 1), (-2, 1) }
  else if (n % 2 = 0 ∧ n % 3 ≠ 0) then { (0, 0), (-2, 1), (0, -1) }
  else ∅ :=
by sorry

end polynomial_divisors_l8_8182


namespace solutions_to_cubic_l8_8157

noncomputable theory

def cubic_polynomial (z : ℂ) : ℂ := z^3 + 27

theorem solutions_to_cubic :
  {z : ℂ | cubic_polynomial z = 0} = {-3, (3/2) + (3 * complex.I * real.sqrt 3)/2, (3/2) - (3 * complex.I * real.sqrt 3)/2} :=
by
  sorry

end solutions_to_cubic_l8_8157


namespace correct_option_is_B_l8_8805

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l8_8805


namespace prob_second_white_given_first_white_l8_8608

/-- 
  In a pocket, there are 5 white balls and 4 black balls.
  If two balls are drawn consecutively without replacement,
  the probability that the second ball drawn is white, 
  given that the first ball drawn is white, is 1/2.
-/
theorem prob_second_white_given_first_white (totalBalls : Nat) (whiteBalls : Nat) (blackBalls : Nat) :
  totalBalls = 9 → whiteBalls = 5 → blackBalls = 4 → 
  let P_A : ℚ := whiteBalls / totalBalls in
  let P_AB : ℚ := (whiteBalls * (whiteBalls - 1)) / (totalBalls * (totalBalls - 1)) in
  let P_B_given_A : ℚ := P_AB / P_A in
  P_B_given_A = 1 / 2 :=
by
  intros htow htow_w htow_b
  rw [htow, htow_w, htow_b]
  let P_A : ℚ := 5 / 9
  let P_AB : ℚ := (5 * 4) / (9 * 8)
  let P_B_given_A : ℚ := P_AB / P_A
  have h1 : P_A = 5 / 9 := rfl
  have h2 : P_AB = 5 / 18 := rfl
  have h3 : P_B_given_A = (5 / 18) / (5 / 9) := rfl
  simp [h1, h2, h3]
  norm_num
  sorry

end prob_second_white_given_first_white_l8_8608


namespace largest_multiple_of_8_less_than_100_l8_8784

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8784


namespace largest_multiple_of_8_less_than_100_l8_8786

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8786


namespace num_integers_satisfy_condition_l8_8583

theorem num_integers_satisfy_condition :
  let f (n : ℤ) : ℤ := (n - 3) * (n + 5) * (n + 9)
  in ((finset.Icc (-15 : ℤ) 15).filter (λ n, f n < 0)).card = 13 := by
  sorry

end num_integers_satisfy_condition_l8_8583


namespace common_ratio_arithmetic_geometric_sequence_l8_8267

noncomputable def log_c (a c : ℝ) := Real.log a / Real.log c
noncomputable def log_a (b a : ℝ) := Real.log b / Real.log a
noncomputable def log_b (c b : ℝ) := Real.log c / Real.log b

theorem common_ratio_arithmetic_geometric_sequence 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_arith_seq : 2 * b = a + c)
  : (let x := log_c a c in
     let k := log_c 2 c in
     let l := log_a 2 a in
     let r := log_b c b / x in
     r = 4 / (3 + (1 + 8 * l).sqrt - 2 * k)) :=
by sorry

end common_ratio_arithmetic_geometric_sequence_l8_8267


namespace company_sales_difference_l8_8750

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l8_8750


namespace problem_1_problem_2_l8_8543

-- Define the geometric setup of the problem
variables {a b c x y : ℝ}

-- Given conditions for the ellipse and symmetry
def ellipse_equation : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def symmetry_line : Prop := (x - y + a = 0)

def symmetric_point_condition (x_M y_M : ℝ) : Prop := 
  (x_M + y_M - a = 0) ∧ (3 * x_M + 2 * (y_M) = 0)

-- Define foci
def foci : Prop := ∃ c : ℝ, c = a * sqrt(1 - (b^2 / a^2))

-- Define the eccentricity calculation
def eccentricity : ℝ := c / a

-- Problem 1: Prove the eccentricity e of the ellipse
theorem problem_1 (h_ellipse : ellipse_equation) (h_symmetry : symmetry_line)
  (h_symmetric_point : ∃ (x_M y_M : ℝ), symmetric_point_condition x_M y_M): 
  ∃ e : ℝ, e = 1 / 2 :=
sorry

-- Define the major axis length and line slope for second problem
def major_axis : ℝ := 4
def line_slope : ℝ := 1 / 2

-- Define the intersection points A and B with their slopes relative to a point P
def slopes_sum_constant (P A B : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, (A.1 + B.1 = -2 * t) ∧ 
    (A.1 * B.1 = t^2 - 3) ∧ 
    (line_slope * (A.2 - P.2) / (A.1 - P.1) + line_slope * (B.2 - P.2) / (B.1 - P.1) = constant_value)

-- Problem 2: Prove the existence of fixed points P such that the slopes sum to a constant
theorem problem_2 
  (h_major_axis : major_axis = 4) 
  (h_slope : line_slope = 1 / 2)
  (h_eccentricity : eccentricity = 1 / 2)
  (h_slopes_sum_constant : ∃ (A B P : ℝ × ℝ), slopes_sum_constant P A B) : 
  ∃ P₁ P₂ : ℝ × ℝ, 
    (P₁ = (-1, -3 / 2)) ∧ (P₂ = (1, 3 / 2)) :=
sorry

end problem_1_problem_2_l8_8543


namespace turn_off_lamps_l8_8733

theorem turn_off_lamps : 
  ∃ (ways : ℕ), 
  (ways = 20) ∧ 
  (∀ (l : list ℕ), 
    (l.length = 3) ∧ 
    (∀ x, x ∈ l → x ≠ 1 ∧ x ≠ 10) ∧ 
    (list.pairwise (λ x y, abs (x - y) > 1) l) 
    → 
    ways = 20) :=
sorry

end turn_off_lamps_l8_8733


namespace shaded_area_correct_l8_8852

def side_length : ℝ := 3
def large_radius : ℝ := side_length / 2
def small_radius : ℝ := large_radius / 2

def hexagon_area : ℝ := (3 * Real.sqrt 3 / 2 ) * side_length^2
def larger_semi_circle_area : ℝ := (1/2) * Real.pi * large_radius^2
def smaller_semi_circle_area : ℝ := (1/2) * Real.pi * small_radius^2

def total_semi_circle_area : ℝ := 3 * larger_semi_circle_area + 3 * smaller_semi_circle_area

def area_of_shaded_region : ℝ := hexagon_area - total_semi_circle_area

theorem shaded_area_correct :
  area_of_shaded_region = 13.5 * Real.sqrt 3 - (135 * Real.pi / 32) := by
    sorry

end shaded_area_correct_l8_8852


namespace solved_men_and_women_problem_l8_8813

def men_and_women_problem : Prop :=
  ∃ (x : ℕ),
  nmen = 4*x ∧
  nwomen = 5*x ∧
  final_nmen = 4*x + 2 ∧
  final_nwomen = 2 * (5 * x - 3) ∧
  final_nmen = 14 ∧
  final_nwomen = 24

theorem solved_men_and_women_problem : men_and_women_problem :=
sorry

end solved_men_and_women_problem_l8_8813


namespace tan_sum_identity_l8_8499

theorem tan_sum_identity :
  let α : ℝ := 10 * (ℝ.pi / 180)
  let β : ℝ := 50 * (ℝ.pi / 180)
  tan α + tan β + real.sqrt 3 * tan α * tan β = real.sqrt 3 :=
by
  sorry

end tan_sum_identity_l8_8499


namespace function_has_two_zeros_iff_a_in_range_l8_8972

theorem function_has_two_zeros_iff_a_in_range (a : ℝ) :
    (∃ x1 x2 : ℝ, (x1 < 1 ∧ ln (1 - x1) = 0) ∧ (x2 ≥ 1 ∧ sqrt x2 - a = 0) ∧ x1 ≠ x2)
    ↔ (a ∈ set.Ici 1) :=
by
  sorry

end function_has_two_zeros_iff_a_in_range_l8_8972


namespace a_n_general_formula_b_n_minus_a_n_geometric_b_n_sum_minimum_value_l8_8952

section Sequences

-- Definitions of sequences {a_n} and {b_n}
def a_seq (n : ℕ) : ℝ :=
if h : n > 0 then 
  (1 / 2) * n + 1 / 4 
else 3 / 4

def b_seq (n : ℕ) : ℝ :=
nat.rec_on n (-37 / 4) (λ n b, (b + (n + 2) / 3))

-- Prove the general formula for {a_n}
theorem a_n_general_formula (n : ℕ) (h : n ≥ 1) : 
  (a_seq n = (1 / 2) * n + 1 / 4) :=
by
  sorry

-- Prove that {b_n - a_n} is a geometric sequence
theorem b_n_minus_a_n_geometric (n : ℕ) (h1 : n ≥ 2) :
  ∃ r : ℝ, ∀ n, n ≥ 2 → b_seq n - a_seq n = r * (b_seq (n-1) - a_seq (n-1)) :=
by
  sorry

-- Prove the minimum value of the sum of the first n terms of {b_n}
theorem b_n_sum_minimum_value (n : ℕ) (h1 : n ≥ 2) : 
  ∃ k : ℝ, k = ((b_seq 1) + (b_seq 2) + ∑ i in range (3), b_seq i) ∧ k = -34 / 3 :=
by
  sorry

end Sequences

end a_n_general_formula_b_n_minus_a_n_geometric_b_n_sum_minimum_value_l8_8952


namespace quadrilateral_perimeter_l8_8389

-- Define constants for lengths and angles
constant AB : ℝ := 5
constant CD : ℝ := 6
constant BC : ℝ := 10
constant angle_BCD : ℝ := 120
constant angle_ABC : ℝ := 90

-- Define a quadrilateral with the given properties
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (AB_perpendicular_BC : angle_ABC = 90)
  (BCD_angle : angle_BCD = 120)
  (AB_length : dist A B = AB)
  (BC_length : dist B C = BC)
  (CD_length : dist C D = CD)

-- Define the function to calculate the perimeter of the quadrilateral
noncomputable def perimeter (Q : Quadrilateral) : ℝ :=
  let AD := dist Q.A Q.D in
  Q.AB_length + Q.BC_length + Q.CD_length + AD

-- State the theorem to prove the perimeter is 34.08 cm
theorem quadrilateral_perimeter (Q : Quadrilateral) : perimeter Q = 34.08 :=
by
  sorry

end quadrilateral_perimeter_l8_8389


namespace simplify_and_rationalize_l8_8693

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 :=
by sorry

end simplify_and_rationalize_l8_8693


namespace tetrahedron_volume_from_cube_l8_8068

theorem tetrahedron_volume_from_cube {s : ℝ} (h : s = 8) :
  let cube_volume := s^3
  let smaller_tetrahedron_volume := (1/3) * (1/2) * s * s * s
  let total_smaller_tetrahedron_volume := 4 * smaller_tetrahedron_volume
  let tetrahedron_volume := cube_volume - total_smaller_tetrahedron_volume
  tetrahedron_volume = 170.6666 :=
by
  sorry

end tetrahedron_volume_from_cube_l8_8068


namespace find_a_value_l8_8974

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a_value :
  ∃ a : ℝ, f a (1 - a) = f a (1 + a) ∧ a = -3 / 4 :=
begin
  use -3 / 4,
  split,
  {
    have h1 : 1 - (-3 / 4) < 1, by norm_num,
    have h2 : 1 + (-3 / 4) < 1, by norm_num,
    rw f,
    rw [if_pos h1, if_neg (by norm_num : ¬1 + -3 / 4 < 1)],
    norm_num,
  },
  refl,
end

end find_a_value_l8_8974


namespace convert_A03_to_decimal_l8_8900

theorem convert_A03_to_decimal :
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  hex_value = 2563 :=
by
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  have : hex_value = 2563 := sorry
  exact this

end convert_A03_to_decimal_l8_8900


namespace lemon_pie_degrees_l8_8226

noncomputable def num_students := 45
noncomputable def chocolate_pie_students := 15
noncomputable def apple_pie_students := 9
noncomputable def blueberry_pie_students := 9
noncomputable def other_pie_students := num_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
noncomputable def each_remaining_pie_students := other_pie_students / 3
noncomputable def fraction_lemon_pie := each_remaining_pie_students / num_students
noncomputable def degrees_lemon_pie := fraction_lemon_pie * 360

theorem lemon_pie_degrees : degrees_lemon_pie = 32 :=
sorry

end lemon_pie_degrees_l8_8226


namespace extremum_points_range_of_b_inequality_exp_l8_8982

-- Definition for condition (1)
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 - Real.log x

-- Problem 1: Extrema of f(x)
theorem extremum_points (a : ℝ) :
  (a > 0 → ∀ x > 0, f a x = f a (1/a)) ∧ (a ≤ 0 → ∀ x > 0, ¬ (∃ y, y ∈ Set.localExtr f a x)) := by
  sorry

-- Problem 2: Range of b given the conditions
theorem range_of_b (b : ℝ) :
  (∀ x > 0, f 1 x ≥ b * x - 2) → b ≤ (Real.exp 2 - 1) / Real.exp 2 := by
  sorry

-- Problem 3: Inequality for e^(x - y)
theorem inequality_exp (x y : ℝ) (h : x > y) (hx : x > Real.exp 1 - 1) (hy : y > Real.exp 1 - 1) :
  Real.exp (x - y) > Real.log (x + 1) / Real.log (y + 1) := by
  sorry

end extremum_points_range_of_b_inequality_exp_l8_8982


namespace maximum_positive_value_a_l8_8150

theorem maximum_positive_value_a :
  ∃ (a : ℤ), (∀ (b : ℤ), (0 < b) ∧ b > a → ¬ (real.sqrt 3 + real.sqrt 8 > 1 + real.sqrt b)) ∧ a = 12 :=
begin
  sorry
end

end maximum_positive_value_a_l8_8150


namespace knights_minimum_count_l8_8423

/-- There are 1001 people sitting around a round table, each of whom is either a knight (always tells the truth) or a liar (always lies).
Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight.
Prove that the minimum number of knights that can be sitting at the table is 502. -/
theorem knights_minimum_count (n : ℕ) (h : n = 1001) (N : ℕ) (L : ℕ) 
  (h1 : N + L = n) (h2 : ∀ i, (i < n) → 
    ((is_knight i ∧ is_liar ((i + 1) % n)) ∨ (is_liar i ∧ is_knight ((i + 1) % n)))) 
  : N = 502 :=
sorry

end knights_minimum_count_l8_8423


namespace correct_statement_is_B_l8_8802

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l8_8802


namespace earnings_difference_l8_8743

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l8_8743


namespace solutions_of_cubic_eq_l8_8154

theorem solutions_of_cubic_eq : 
  let z := Complex in
  ∃ (a b c : z), 
    (a = -3) ∧ 
    (b = Complex.mk (3 / 2) (3 * Real.sqrt 3 / 2)) ∧ 
    (c = Complex.mk (3 / 2) (-(3 * Real.sqrt 3 / 2))) ∧
    (∀ x : z, x^3 = -27 ↔ (x = a ∨ x = b ∨ x = c)) :=
by
  sorry

end solutions_of_cubic_eq_l8_8154


namespace open_number_1029_maximum_M_when_G_divisible_by_7_l8_8164

noncomputable def is_open_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  let A := 10 * a + d
  let B := 10 * c + b
  A - B = -(a + b)

noncomputable def G (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  (b + 13) / (10 * a - 9 * c)

theorem open_number_1029 : is_open_number 1029 :=
  sorry

theorem maximum_M_when_G_divisible_by_7 : ∃ M : ℕ, is_open_number M ∧ G M % 7 = 0 ∧ M = 8892 :=
  sorry

end open_number_1029_maximum_M_when_G_divisible_by_7_l8_8164


namespace train_speed_l8_8089

theorem train_speed (train_length : ℕ) (crossing_time : ℕ) (train_length = 3000) (crossing_time = 120) : 
  let distance_km := train_length / 1000 in
  let time_hr := crossing_time / 3600 in
  let speed := distance_km / time_hr in
  speed = 90 :=
begin
  simp [train_length, crossing_time],
  sorry
end

end train_speed_l8_8089


namespace product_p_yi_eq_neg26_l8_8270

-- Definitions of the polynomials h and p.
def h (y : ℂ) : ℂ := y^3 - 3 * y + 1
def p (y : ℂ) : ℂ := y^3 + 2

-- Given that y1, y2, y3 are roots of h(y)
variables (y1 y2 y3 : ℂ) (H1 : h y1 = 0) (H2 : h y2 = 0) (H3 : h y3 = 0)

-- State the theorem to show p(y1) * p(y2) * p(y3) = -26
theorem product_p_yi_eq_neg26 : p y1 * p y2 * p y3 = -26 :=
sorry

end product_p_yi_eq_neg26_l8_8270


namespace max_cut_trees_l8_8022

theorem max_cut_trees (G : Type) [fintype G] (gard : G → G → Prop) :
  (forall i j : G, gard i j) →
  (∀ cut_trees : set (G × G), (∀ t1 t2 ∈ cut_trees, t1 ≠ t2 → ∃ u ∈ (G × G) \ cut_trees, gard u t1 ∧ gard u t2) →
    cut_trees.finite → cut_trees.card ≤ 2500) :=
begin
  sorry
end

end max_cut_trees_l8_8022


namespace polynomial_root_reciprocal_square_sum_l8_8122

theorem polynomial_root_reciprocal_square_sum :
  ∀ (a b c : ℝ), (a + b + c = 6) → (a * b + b * c + c * a = 11) → (a * b * c = 6) →
  (1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 = 49 / 36) :=
by
  intros a b c h_sum h_prod_sum h_prod
  sorry

end polynomial_root_reciprocal_square_sum_l8_8122


namespace largest_m_dividing_factorial_l8_8510

theorem largest_m_dividing_factorial :
  (∃ m : ℕ, (∀ n : ℕ, (18^n ∣ 30!) ↔ n ≤ m) ∧ m = 7) :=
by
  sorry

end largest_m_dividing_factorial_l8_8510


namespace infinite_palindromic_multiples_l8_8074

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

theorem infinite_palindromic_multiples (n : ℕ) (h_n : ∃ (k : ℕ) (p : Fin k → ℕ), (∀ i, Nat.Prime (p i)) ∧ (∀ i, ¬(p i) ∣ 10) ∧ (n = ∏ i, p i)) :
  ∀ (ℓ : ℕ), ∃ m, m = 10 ^ (ℓ * (∏ i in Finset.range (Fin  k), (p i - 1))) - 1 ∧ is_palindrome m ∧ n ∣ m :=
sorry

end infinite_palindromic_multiples_l8_8074


namespace sub_neg_of_lt_l8_8212

theorem sub_neg_of_lt {x y : ℝ} (h : x < y) : x - y < 0 := by
  sorry

end sub_neg_of_lt_l8_8212


namespace last_passenger_sits_in_assigned_seat_l8_8342

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l8_8342


namespace SimplifyAndRationalize_l8_8696

theorem SimplifyAndRationalize :
  ( (√3 / √7) * (√5 / √8) * (√6 / √9) ) = ( √35 / 42 ) :=
sorry

end SimplifyAndRationalize_l8_8696


namespace exists_point_F_on_PC_parallel_to_plane_AEC_l8_8463

theorem exists_point_F_on_PC_parallel_to_plane_AEC 
  (a : ℝ) 
  (angle_ABC : ℝ) 
  (PA_eq_AC : PA = a)
  (PB_eq_PD : PB = sqrt 2 * a)
  (E_partition : ∃ E, E ∈ line_segment PD ∧ PE / ED = 2 / 1) :
  ∃ F ∈ (segment P C), BF ∥ (span ℝ {A, E, C}) := 
sorry

end exists_point_F_on_PC_parallel_to_plane_AEC_l8_8463


namespace twentieth_century_years_as_powers_of_two_diff_l8_8033

theorem twentieth_century_years_as_powers_of_two_diff :
  ∀ (y : ℕ), (1900 ≤ y ∧ y < 2000) →
    ∃ (n k : ℕ), y = 2^n - 2^k ↔ y = 1984 ∨ y = 1920 := 
by
  sorry

end twentieth_century_years_as_powers_of_two_diff_l8_8033


namespace total_squares_after_erasing_lines_l8_8901

theorem total_squares_after_erasing_lines :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ), a = 16 → b = 4 → c = 9 → d = 2 → 
  a - b + c - d + (a / 16) = 22 := 
by
  intro a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_squares_after_erasing_lines_l8_8901


namespace simplify_and_rationalize_l8_8702

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l8_8702


namespace length_of_midsegment_in_trapezoid_l8_8729

-- Definitions based on the conditions
def base_angle_sum := 90
def upper_base_length := 5
def lower_base_length := 11

-- Theorem stating the problem
theorem length_of_midsegment_in_trapezoid :
  let midsegment_length := (lower_base_length - upper_base_length) / 2
  in midsegment_length = 3 :=
by
  sorry

end length_of_midsegment_in_trapezoid_l8_8729


namespace right_triangle_arithmetic_prog_inradius_eq_d_l8_8688

theorem right_triangle_arithmetic_prog_inradius_eq_d 
  (a b c d : ℕ) (h₁ : a ≤ b) (h₂ : b < c) 
  (h₃ : b = a + d) (h₄ : c = a + 2 * d)
  (h₅ : c^2 = a^2 + b^2) : 
  let s := (a + b + c) / 2 in
  let A := a * b / 2 in
  let r := A / s in
  r = d := 
sorry

end right_triangle_arithmetic_prog_inradius_eq_d_l8_8688


namespace warehouse_goods_problem_l8_8435

-- Given definitions
def changes : List Int := [24, -48, -13, 37, -52, 57, -13, -33]
def final_goods_in_warehouse : Int := 217
def handling_fee_per_ton : Int := 15

-- Proof statement
theorem warehouse_goods_problem :
  (changes.sum = -41) ∧
  (final_goods_in_warehouse + 41 = 258) ∧
  (changes.map (λ x, abs x)).sum * handling_fee_per_ton = 4155 :=
by
  sorry

end warehouse_goods_problem_l8_8435


namespace area_of_triangle_l8_8265

open Matrix

def vector_a : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![3], ![2]]

def vector_b : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![1], ![5]]

theorem area_of_triangle (a b : Matrix (Fin 2) (Fin 1) ℝ) (h₁ : a = vector_a) (h₂ : b = vector_b) :
  0.5 * abs (a.data 0 0 * b.data 1 0 - a.data 1 0 * b.data 0 0) = 6.5 :=
by
  rw [h₁, h₂]
  sorry

end area_of_triangle_l8_8265


namespace lumberjack_trees_l8_8844

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l8_8844


namespace terez_pregnant_female_cows_l8_8320

def total_cows : ℕ := 44
def percent_female : ℚ := 0.50
def percent_pregnant_female : ℚ := 0.50
def female_cows : ℕ := (total_cows * percent_female).toNat
def pregnant_female_cows : ℕ := (female_cows * percent_pregnant_female).toNat

theorem terez_pregnant_female_cows : pregnant_female_cows = 11 := 
by
  sorry

end terez_pregnant_female_cows_l8_8320


namespace first_player_wins_l8_8820

def initial_box1 : ℕ := 2017
def initial_box2 : ℕ := 2018

noncomputable def move (box1 box2 : ℕ) : (ℕ × ℕ) := sorry

theorem first_player_wins :
  ∃ strat : (ℕ × ℕ) → (ℕ × ℕ), 
    ∀ (box1 box2 : ℕ) (n : ℕ), 
      (box1, box2) = (initial_box1, initial_box2) ∨ (box1, box2) = (2 * n, 2 * n + 1) 
      → (box1, strat (box1, box2)).fst % (box1, strat (box1, box2)).snd ≠ 0 
      ∧ (box1, strat (box1, box2)).snd % (box1, strat (box1, box2)).fst ≠ 0 
      ∧ (box2, strat (box1, box2)).fst % (box2, strat (box1, box2)).snd ≠ 0 
      ∧ (box2, strat (box1, box2)).snd % (box2, strat (box1, box2)).fst ≠ 0 := sorry

end first_player_wins_l8_8820


namespace rectangle_parallelepiped_angles_l8_8757

theorem rectangle_parallelepiped_angles 
  (a b c d : ℝ) 
  (α β : ℝ) 
  (h_a : a = d * Real.sin β)
  (h_b : b = d * Real.sin α)
  (h_d : d^2 = (d * Real.sin β)^2 + c^2 + (d * Real.sin α)^2) :
  (α > 0 ∧ β > 0 ∧ α + β < 90) := sorry

end rectangle_parallelepiped_angles_l8_8757


namespace angle_measure_l8_8100

theorem angle_measure (α : ℝ) (h1 : α - (90 - α) = 20) : α = 55 := by
  -- Proof to be provided here
  sorry

end angle_measure_l8_8100


namespace Mary_books_check_out_l8_8286

theorem Mary_books_check_out
  (initial_books : ℕ)
  (returned_unhelpful_books : ℕ)
  (returned_later_books : ℕ)
  (checked_out_later_books : ℕ)
  (total_books_now : ℕ)
  (h1 : initial_books = 5)
  (h2 : returned_unhelpful_books = 3)
  (h3 : returned_later_books = 2)
  (h4 : checked_out_later_books = 7)
  (h5 : total_books_now = 12) :
  ∃ (x : ℕ), (initial_books - returned_unhelpful_books + x - returned_later_books + checked_out_later_books = total_books_now) ∧ x = 5 :=
by {
  sorry
}

end Mary_books_check_out_l8_8286


namespace valid_root_for_A_valid_root_for_E_l8_8120

noncomputable def has_complex_root (p : Polynomial ℝ) (z : ℂ) : Prop :=
  p.eval (coe z) = 0

def valid_poly (r s : ℝ) : Polynomial ℝ :=
  Polynomial.X * (Polynomial.X - Polynomial.C r) * 
  (Polynomial.X - Polynomial.C s) * 
  (Polynomial.X^2 + Polynomial.C (-1) * Polynomial.X + Polynomial.C 8) -- This choice corresponds for a and b for (A)

def valid_poly2 (r s : ℝ) : Polynomial ℝ :=
  Polynomial.X * (Polynomial.X - Polynomial.C r) * 
  (Polynomial.X - Polynomial.C s) * 
  (Polynomial.X^2 + Polynomial.C (-1) * Polynomial.X + Polynomial.C 9) -- This choice corresponds for a and b for (E)

theorem valid_root_for_A (r s : ℝ) :
  has_complex_root (valid_poly r s) (⟨1 / 2, real.sqrt 15 / 2⟩) :=
sorry

theorem valid_root_for_E (r s : ℝ) :
  has_complex_root (valid_poly2 r s) (⟨1 / 2, real.sqrt 17 / 2⟩) :=
sorry

end valid_root_for_A_valid_root_for_E_l8_8120


namespace earnings_difference_l8_8742

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l8_8742


namespace f_2015_l8_8959

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, cos x
| (n + 1) := λ x, (f n)' x

theorem f_2015 (x : ℝ) : f 2015 x = sin x :=
by
  sorry

end f_2015_l8_8959


namespace angle_equality_l8_8741

-- Definitions of the points (implicitly assumes they satisfy any relevant geometric properties)
variables {O1 O2 A B M1 M2 : Point}
-- Definitions of the circles
variables (circle1 : Circle O1) (circle2 : Circle O2)
-- Conditions of the problem
variables (h1 : A ∈ circle1) (h2 : B ∈ circle1)
variables (h3 : A ∈ circle2) (h4 : B ∈ circle2)
variables (h5 : M1 ∈ circle1) (h6 : M2 ∈ circle2)
variables (h7 : LineThroughA : Line A B)

-- The proof problem to show
theorem angle_equality : 
  ∠ B O1 M1 = ∠ B O2 M2 :=
sorry

end angle_equality_l8_8741


namespace smallest_x_plus_3956_palindrome_l8_8390

def is_palindrome (n : ℕ) : Prop :=
  let s := to_string n
  s = s.reverse

theorem smallest_x_plus_3956_palindrome :
  ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 3956) ∧ x = 48 :=
by
  sorry

end smallest_x_plus_3956_palindrome_l8_8390


namespace smallest_part_l8_8585

theorem smallest_part {total parts : ℕ} (h_total : total = 120) (h_ratio : parts = [3, 5, 7]) :
  ∃ x : ℕ, 3 * x = 24 :=
sorry

end smallest_part_l8_8585


namespace total_people_hired_l8_8460

theorem total_people_hired (H L : ℕ) (hL : L = 1) (payroll : ℕ) (hPayroll : 129 * H + 82 * L = 3952) : H + L = 31 := by
  sorry

end total_people_hired_l8_8460


namespace largest_multiple_of_8_less_than_100_l8_8785

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8785


namespace sarah_mean_score_l8_8936

noncomputable def john_mean_score : ℝ := 86
noncomputable def john_num_tests : ℝ := 4
noncomputable def test_scores : List ℝ := [78, 80, 85, 87, 90, 95, 100]
noncomputable def total_sum : ℝ := test_scores.sum
noncomputable def sarah_num_tests : ℝ := 3

theorem sarah_mean_score :
  let john_total_score := john_mean_score * john_num_tests
  let sarah_total_score := total_sum - john_total_score
  let sarah_mean_score := sarah_total_score / sarah_num_tests
  sarah_mean_score = 90.3 :=
by
  sorry

end sarah_mean_score_l8_8936


namespace min_knights_proof_l8_8425

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end min_knights_proof_l8_8425


namespace isosceles_triangles_exist_l8_8947

theorem isosceles_triangles_exist (V : Finset (Fin 2005)) (colors : Fin 2005 → Fin 2) :
  ∃ T : Finset (Finset (Fin 2005)), T.card ≥ 101 ∧ ∀ {a b c : Fin 2005}, a ∈ V ∧ b ∈ V ∧ c ∈ V ∧ 
    (colors a = colors b ∧ colors b = colors c) ∧ (Isosceles (a, b, c)) :=
sorry

end isosceles_triangles_exist_l8_8947


namespace range_of_a_l8_8971

def f (x : ℝ) : ℝ :=
  if x < 0 then -x^3 else x^3

theorem range_of_a (a : ℝ) : f (3 * a - 1) ≥ 8 * f a ↔ a ≤ 1/5 ∨ a ≥ 1 :=
by sorry

end range_of_a_l8_8971


namespace coinciding_rest_days_count_l8_8885

-- Define the schedules as repetitive cycles
def charlie_schedule (n : ℕ) : ℕ := n % 6 -- 4 work-days, 2 rest-days
def dana_schedule (n : ℕ) : ℕ := n % 10 -- 9 work-days, 1 rest-day

-- Define rest-days for Charlie and Dana
def is_rest_day_charlie (n : ℕ) : Prop := (charlie_schedule n) ≥ 4
def is_rest_day_dana (n : ℕ) : Prop := (dana_schedule n) = 9

-- Define the condition for coinciding rest-days
def coinciding_rest_day (n : ℕ) : Prop := is_rest_day_charlie n ∧ is_rest_day_dana n

-- Main theorem statement
theorem coinciding_rest_days_count : 
    (finset.range 1200).filter coinciding_rest_day).card = 40 :=
by trivial_tagging

end coinciding_rest_days_count_l8_8885


namespace tangent_difference_problem_l8_8946

-- Define the conditions and the problem statement
theorem tangent_difference_problem (α : ℝ) (hα1 : α ∈ set.Ioo (π / 2) π) (hα2 : Real.sin α = 3 / 5) :
  Real.tan (α - π / 4) = -7 :=
sorry

end tangent_difference_problem_l8_8946


namespace C_should_pay_45_rs_l8_8859

def ox_months (oxen : ℕ) (months : ℕ) : ℕ :=
  oxen * months

def total_ox_months (A B C : ℕ) : ℕ :=
  A + B + C

def cost_per_ox_month (total_rent : ℕ) (total_ox_months : ℕ) : ℕ :=
  total_rent / total_ox_months

def C_rent (C_ox_months : ℕ) (cost_per_ox_month : ℕ) : ℕ :=
  C_ox_months * cost_per_ox_month

theorem C_should_pay_45_rs (A_oxen A_months B_oxen B_months C_oxen C_months total_rent : ℕ) :
  let A_ox_months := ox_months A_oxen A_months in
  let B_ox_months := ox_months B_oxen B_months in
  let C_ox_months := ox_months C_oxen C_months in
  let total_ox := total_ox_months A_ox_months B_ox_months C_ox_months in
  let cost_per_month := cost_per_ox_month total_rent total_ox in
  C_rent C_ox_months cost_per_month = 45 := by
  -- Definitions from the conditions
  let A_oxen := 10
  let A_months := 7
  let B_oxen := 12
  let B_months := 5
  let C_oxen := 15
  let C_months := 3
  let total_rent := 175

  -- Compute intermediate values
  let A_ox_months := ox_months A_oxen A_months
  let B_ox_months := ox_months B_oxen B_months
  let C_ox_months := ox_months C_oxen C_months
  let total_ox := total_ox_months A_ox_months B_ox_months C_ox_months
  let cost_per_month := cost_per_ox_month total_rent total_ox

  -- Assert final value
  sorry

end C_should_pay_45_rs_l8_8859


namespace pyramid_base_edge_length_l8_8713

-- Definitions and conditions
def radius : ℝ := 3
def height : ℝ := 8

-- Theorem statement: the edge-length of the base of the pyramid
theorem pyramid_base_edge_length :
  ∃ s : ℝ, s = (24 * real.sqrt 110) / 55 :=
sorry

end pyramid_base_edge_length_l8_8713


namespace distance_traveled_average_speed_l8_8007

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 0.1 * t ^ 3

-- Define the time interval
def T : ℝ := 10

-- State the distance traveled over the time interval is 250 meters
theorem distance_traveled : (∫ t in 0..T, velocity t) = 250 := 
sorry

-- State the average speed over the time interval is 25 meters per second
theorem average_speed : ((∫ t in 0..T, velocity t) / T) = 25 := 
sorry

end distance_traveled_average_speed_l8_8007


namespace daily_round_trip_miles_l8_8739

def cost_per_gallon := 2
def total_spent := 80
def miles_per_gallon := 25
def work_days_per_week := 5

theorem daily_round_trip_miles :
  let total_gas_used := total_spent / cost_per_gallon in
  let total_miles_driven := total_gas_used * miles_per_gallon in
  let weekly_miles_driven := total_miles_driven / 4 in
  let daily_miles_driven := weekly_miles_driven / work_days_per_week in
  daily_miles_driven = 50 :=
by
  -- Proof omitted
  sorry

end daily_round_trip_miles_l8_8739


namespace runners_together_after_2000_seconds_l8_8938

theorem runners_together_after_2000_seconds :
  (∃ t : ℕ, 0 < t ∧ (t = 2000) ∧
            (∀ (d1 d2 d3 d4 : ℕ),
            let d1 := 4.2 * t,
                d2 := 4.5 * t,
                d3 := 4.8 * t,
                d4 := 5.1 * t in
            d1 % 600 = d2 % 600 ∧
            d2 % 600 = d3 % 600 ∧
            d3 % 600 = d4 % 600)) :=
begin
  -- skipped proof
  sorry
end

end runners_together_after_2000_seconds_l8_8938


namespace last_passenger_probability_l8_8330

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l8_8330


namespace simplify_and_rationalize_l8_8692

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 :=
by sorry

end simplify_and_rationalize_l8_8692


namespace monotonically_increasing_interval_symmetry_center_area_of_triangle_ABC_l8_8976

def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.sin (real.pi / 4 + x) ^ 2 + 2 * real.sin (real.pi / 4 + x) * real.cos (real.pi / 4 + x)

open set

theorem monotonically_increasing_interval :
  ∃ (k : ℤ), ∀ x ∈ Icc (-real.pi / 3 + k * real.pi) (real.pi / 6 + k * real.pi), 
    monotone_on f (Icc (-real.pi / 3 + k * real.pi) (real.pi / 6 + k * real.pi)) :=
sorry

theorem symmetry_center :
  ∃ (k : ℤ), ( - real.pi / 12 + k * real.pi / 2, real.sqrt 3) ∈ center_of_symmetry f :=
sorry

theorem area_of_triangle_ABC 
  (A a : ℝ) (median_BC : ℝ) (area : ℝ) 
  (h_A : f A = real.sqrt 3 + 1) (h_a : a = 3) (h_median_BC : median_BC = 3) :
  area = 27 * real.sqrt 3 / 8 :=
sorry

end monotonically_increasing_interval_symmetry_center_area_of_triangle_ABC_l8_8976


namespace fixed_point_on_graph_l8_8714

theorem fixed_point_on_graph {a : ℝ} (h₁ : 0 < a) (h₂ : a ≠ 1) :
  (∃ x : ℝ, f x = -1) :=
  begin
    let f := λ x, 2 * a^(x + 1) - 3,
    use -1,
    calc f(-1) = 2 * a^(0) - 3 : by { rw [add_neg_one, pow_zero]}
          ... = 2 * 1 - 3 : by { rw mul_one }
          ... = -1 : by rw sub_self
  end

end fixed_point_on_graph_l8_8714


namespace find_x_values_l8_8161

theorem find_x_values (x : ℝ) : 
  ((x + 1)^2 = 36 ∨ (x + 10)^3 = -27) ↔ (x = 5 ∨ x = -7 ∨ x = -13) :=
by
  sorry

end find_x_values_l8_8161


namespace combined_original_price_l8_8285

theorem combined_original_price (S P : ℝ) 
  (hS : 0.25 * S = 6) 
  (hP : 0.60 * P = 12) :
  S + P = 44 :=
by
  sorry

end combined_original_price_l8_8285


namespace maximum_value_of_expression_l8_8268

theorem maximum_value_of_expression
  (a b c : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + 2 * c^2 = 1) :
  ab * Real.sqrt 3 + 3 * bc ≤ Real.sqrt 7 :=
sorry

end maximum_value_of_expression_l8_8268


namespace equivalent_single_discount_l8_8865

theorem equivalent_single_discount (P : ℝ) (hP : 0 < P) : 
    let first_discount : ℝ := 0.15
    let second_discount : ℝ := 0.25
    let single_discount : ℝ := 0.3625
    (1 - first_discount) * (1 - second_discount) * P = (1 - single_discount) * P := by
    sorry

end equivalent_single_discount_l8_8865


namespace simplify_expression_l8_8275

variables {R : Type*} [LinearOrderedField R]
variables {a b c x : R}
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem simplify_expression (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ a b c x : R, 
   (h_distinct) →
   ((a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)) →
   ( 
     ( (x + a)^2 / ( (a - b) * (a - c) ) + 
       (x + b)^2 / ( (b - a) * (b - c) ) + 
       (x + c)^2 / ( (c - a) * (c - b) )
     ) = -1
   )
  ) := sorry

end simplify_expression_l8_8275


namespace product_not_square_l8_8302

theorem product_not_square (n : ℕ) : 
  ¬ is_square (∏ i in finset.range (n + 1).succ, i^4 + i^2 + 1) :=
sorry

end product_not_square_l8_8302


namespace total_plums_picked_l8_8093

theorem total_plums_picked :
  let alyssa_rate1 := 17
  let alyssa_rate2 := 3 * alyssa_rate1
  let alyssa_rate3 := alyssa_rate1
  let alyssa_dropped_rate := 0.07 * alyssa_rate3
  let alyssa_total := alyssa_rate1 + alyssa_rate2 + (alyssa_rate3 - alyssa_dropped_rate)

  let jason_rate1 := 10
  let jason_rate2 := jason_rate1 * 1.4
  let jason_rate3 := 2 * jason_rate1
  let jason_dropped_rate := 0.07 * jason_rate3
  let jason_total := jason_rate1 + jason_rate2 + (jason_rate3 - jason_dropped_rate)

  alyssa_total + jason_total = 127 :=
by
  let alyssa_rate1 := 17
  let alyssa_rate2 := 3 * alyssa_rate1
  let alyssa_rate3 := alyssa_rate1
  let alyssa_dropped_rate := 0.07 * alyssa_rate3
  let alyssa_total := alyssa_rate1 + alyssa_rate2 + (alyssa_rate3 - alyssa_dropped_rate)

  let jason_rate1 := 10
  let jason_rate2 := jason_rate1 * 1.4
  let jason_rate3 := 2 * jason_rate1
  let jason_dropped_rate := 0.07 * jason_rate3
  let jason_total := jason_rate1 + jason_rate2 + (jason_rate3 - jason_dropped_rate)

  have h₁ : alyssa_total = 84 :=
    by sorry  -- proof of Alyssa's total
  have h₂ : jason_total = 43 :=
    by sorry  -- proof of Jason's total
  show alyssa_total + jason_total = 127
  by rw [h₁, h₂]
  sorry

end total_plums_picked_l8_8093


namespace tom_found_seashells_l8_8017

theorem tom_found_seashells : ∀ (days : ℕ) (seashells_per_day : ℕ), days = 5 ∧ seashells_per_day = 7 → days * seashells_per_day = 35 := 
by
  intros days seashells_per_day h
  cases h with h_days h_seashells_per_day
  rw [h_days, h_seashells_per_day]
  exact nat.mul_comm 5 7 ▸ rfl

end tom_found_seashells_l8_8017


namespace eval_fraction_sum_l8_8494

theorem eval_fraction_sum : 64^(-1/3) + 81^(-1/2) = 13/36 := by
  sorry

end eval_fraction_sum_l8_8494


namespace conical_funnel_optimal_height_l8_8377

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def radius_from_height (l h : ℝ) : ℝ :=
  (l^2 - h^2).sqrt

def cone_maximizing_height (l : ℝ) : ℝ :=
  10 * real.sqrt 3

-- Statement to be proved
theorem conical_funnel_optimal_height (h l : ℝ) (h_slant: l = 30) :
  let r := radius_from_height l h in
  ∀ (h_optim : ℝ), h_optim = cone_maximizing_height l → volume_of_cone r h_optim = volume_of_cone r h →
  h = 10 * real.sqrt 3 :=
sorry

end conical_funnel_optimal_height_l8_8377


namespace annulus_area_correct_l8_8099

noncomputable def area_of_annulus (r₁ r₂ : ℝ) (k : ℝ) (XZ : ℝ) : ℝ :=
  if k > 1 ∧ r₁ = k * r₂ ∧ r₁^2 = r₂^2 + XZ^2 then π * XZ^2 else 0

theorem annulus_area_correct {r₁ r₂ k XZ : ℝ} (h_r₁ : r₁ = k * r₂) (h_k : k > 1) (h_XZ : r₁^2 = r₂^2 + XZ^2) :
  area_of_annulus r₁ r₂ k XZ = π * XZ^2 :=
by
  simp [area_of_annulus, h_r₁, h_k, h_XZ]
  sorry

end annulus_area_correct_l8_8099


namespace length_of_segment_BD_is_sqrt_3_l8_8239

open Real

-- Define the triangle ABC and the point D according to the problem conditions
def triangle_ABC (A B C : ℝ × ℝ) :=
  B.1 = 0 ∧ B.2 = 0 ∧
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 3 ∧
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = 7 ∧
  C.2 = 0 ∧ (A.1 - C.1) ^ 2 + A.2 ^ 2 = 10

def point_D (A B C D : ℝ × ℝ) :=
  ∃ BD DC : ℝ, BD + DC = sqrt 7 ∧
  BD / DC = sqrt 3 / sqrt 7 ∧
  D.1 = BD / sqrt 7 ∧ D.2 = 0

-- The theorem to prove
theorem length_of_segment_BD_is_sqrt_3 (A B C D : ℝ × ℝ)
  (h₁ : triangle_ABC A B C)
  (h₂ : point_D A B C D) :
  (sqrt ((D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2)) = sqrt 3 :=
sorry

end length_of_segment_BD_is_sqrt_3_l8_8239


namespace Emberly_not_walked_days_l8_8484

theorem Emberly_not_walked_days : 
  (∀ total_days walks_per_day := 31,
   ∀ miles_per_walk := 4, 
   ∀ total_miles_walked := 108,
   ∃ days_not_walked,
   days_not_walked = total_days - (total_miles_walked / miles_per_walk))
  → 4 = days_not_walked :=
by
  intros total_days walks_per_day miles_per_walk total_miles_walked H
  have daily_walk_count : ℕ := total_miles_walked / miles_per_walk
  have days_not_walked : ℕ := total_days - daily_walk_count
  exact Eq.symm (H daily_walk_count days_not_walked)
  sorry

end Emberly_not_walked_days_l8_8484


namespace polynomial_sum_l8_8657

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8657


namespace paper_hole_distribution_l8_8851

theorem paper_hole_distribution :
  let original_paper := (1, 1) in
  let folded_paper := (1 / 2, 1 / 2) in
  let X := (2 / 3 * folded_paper.1, 1 / 3 * folded_paper.2) in
  let Y := (1 / 3 * folded_paper.1, 1 / 3 * folded_paper.2) in
  -- After unfolding the paper
  let unfolded_positions := [
    (2 / 3, 1 / 3), (2 / 3, 2 / 3), (1 / 3, 1 / 3), (1 / 3, 2 / 3),
    (1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)
    ] in
  (∃ grid : fin 3 × fin 3 → Prop, ∀ pos ∈ unfolded_positions, (∃ i j : fin 3, pos = (i / 3, j / 3))) :=
sorry

end paper_hole_distribution_l8_8851


namespace students_taking_neither_music_nor_art_l8_8065

open Set

-- Definitions extracted from the problem conditions
def total_students := 500
def music_students := 30
def art_students := 20
def both_music_and_art_students := 10

-- Lean statement to show the number of students taking neither music nor art
theorem students_taking_neither_music_nor_art : 
  total_students - (music_students + art_students - both_music_and_art_students) = 460 :=
by
  let students_taking_music_or_art := music_students + art_students - both_music_and_art_students
  have students_taking_neither := total_students - students_taking_music_or_art
  show students_taking_neither = 460
  sorry  -- proof here

end students_taking_neither_music_nor_art_l8_8065


namespace quadrilateral_propositions_l8_8797

theorem quadrilateral_propositions :
  (∀ (Q : Type) [quadrilateral Q], (diagonals_bisect Q) → parallelogram Q) ∧
  (∀ (Q : Type) [quadrilateral Q], (diagonals_perpendicular Q) → ¬rhombus Q) ∧
  (∀ (P : Type) [parallelogram P], (diagonals_equal P) → ¬rhombus P) ∧
  (∀ (P : Type) [parallelogram P], (right_angle P) → rectangle P) := 
  by sorry

-- Definitions for the predicates used in the statement for completeness
class quadrilateral (Q : Type) :=
(diagonals_bisect : Q → Prop)
(diagonals_perpendicular : Q → Prop)

class parallelogram (P : Type) extends quadrilateral P :=
(diagonals_equal : P → Prop)
(right_angle : P → Prop)

class rhombus (R : Type) extends parallelogram R :=
()

class rectangle (R : Type) extends parallelogram R :=
()

end quadrilateral_propositions_l8_8797


namespace melinda_math_textbooks_l8_8681

theorem melinda_math_textbooks:
  -- Given conditions
  let boxes := 4
  let total_textbooks := 15
  let math_textbooks := 4
  let box_capacities := [3, 5, 4, 3]
  
  -- Hypothesis: Total ways textbooks can be packed
  let total_ways := Nat.binomial 15 3 * Nat.binomial 12 5 * Nat.binomial 7 4 * Nat.binomial 3 3
  
  -- Favorable ways: All math textbooks in one box
  let favorable_ways :=
    -- Box with capacity 5
    (Nat.binomial 11 1 * Nat.binomial 10 3 * Nat.binomial 7 4 * Nat.binomial 3 3) +
    -- Box with capacity 4
    (Nat.binomial 11 5 * Nat.binomial 6 3 * Nat.binomial 3 3)
  
  -- Probability computation
  let probability := (favorable_ways.toRational / total_ways.toRational)
  let gcd_val := Nat.gcd 50820 6306300
  let m := 11
  let n := 1365
  
  -- Result
  m + n = 1376 :=
by admit

end melinda_math_textbooks_l8_8681


namespace find_smallest_number_l8_8937

theorem find_smallest_number
  (a1 a2 a3 a4 : ℕ)
  (h1 : (a1 + a2 + a3 + a4) / 4 = 30)
  (h2 : a2 = 28)
  (h3 : a2 = 35 - 7) :
  a1 = 27 :=
sorry

end find_smallest_number_l8_8937


namespace eval_fraction_sum_l8_8496

theorem eval_fraction_sum : 64^(-1/3) + 81^(-1/2) = 13/36 := by
  sorry

end eval_fraction_sum_l8_8496


namespace sum_ineq_l8_8269

theorem sum_ineq (n : ℕ) (a : ℕ → ℕ) (h_distinct : function.injective a) :
  ∑ i in finset.range n, (a (i + 1) : ℚ) / ((i + 1) ^ 2) ≥ ∑ i in finset.range n, 1 / (i + 1) := sorry

end sum_ineq_l8_8269


namespace max_quadratic_equations_l8_8790

-- Define a type for quadratic equations
structure QuadraticEquations :=
  (a b c : ℝ)
  (discriminant_ne_zero : b ^ 2 - 4 * a * c ≠ 0)

-- Define the condition that two equations share a common root
def share_common_root (q1 q2 : QuadraticEquations) : Prop :=
  ∃ r, r * r * q1.a + r * q1.b + q1.c = 0 ∧ r * r * q2.a + r * q2.b + q2.c = 0

-- Define the condition that no four equations share a common root
def no_four_share_common_root (qs : set QuadraticEquations) : Prop :=
  ∀ (q1 q2 q3 q4 : QuadraticEquations), 
  q1 ∈ qs → q2 ∈ qs → q3 ∈ qs → q4 ∈ qs →
  ¬ ∃ r, (r * r * q1.a + r * q1.b + q1.c = 0) ∧ 
         (r * r * q2.a + r * q2.b + q2.c = 0) ∧ 
         (r * r * q3.a + r * q3.b + q3.c = 0) ∧ 
         (r * r * q4.a + r * q4.b + q4.c = 0)

-- Prove the maximum number of such quadratic equations is 3
theorem max_quadratic_equations (qs : set QuadraticEquations)
  (h1 : ∀ q1 q2 ∈ qs, q1 ≠ q2 → share_common_root q1 q2)
  (h2 : no_four_share_common_root qs) :
  set.card qs ≤ 3 :=
sorry

end max_quadratic_equations_l8_8790


namespace exists_v_mod_eq_l8_8645

noncomputable def v (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n+1).log2, n / 2^(k + 1)

theorem exists_v_mod_eq (a m : ℕ) (ha : 0 < a) (hm : 0 < m) : 
  ∃ n > 1, v(n) % m = a % m :=
by 
  sorry

end exists_v_mod_eq_l8_8645


namespace probability_residue_1_mod_7_and_divisible_by_3_in_range_l8_8102

open BigOperators

def is_residue_1_mod_7 (N : ℕ) : Prop := N^12 % 7 = 1
def is_divisible_by_3 (N : ℕ) : Prop := N % 3 = 0
def is_in_range (N : ℕ) : Prop := 1 ≤ N ∧ N ≤ 504

theorem probability_residue_1_mod_7_and_divisible_by_3_in_range :
  ∑' (N : ℕ) in (set.Icc 1 504), 
    indicator (fun n => is_residue_1_mod_7 n ∧ is_divisible_by_3 n) N
  = (1/7 : ℚ) :=
sorry

end probability_residue_1_mod_7_and_divisible_by_3_in_range_l8_8102


namespace length_segment_AB_reciprocal_sum_squares_maximum_major_axis_l8_8177

-- Define the basic conditions
def line_eqn (x : ℝ) : ℝ := -x + 1

def ellipse_eqn (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def ellipse_eccentricity_eqn (a b e : ℝ) : Prop :=
  sqrt(a^2 - b^2) / a = e

def focal_distance_eqn (a b : ℝ) : Prop :=
  2 * sqrt(a^2 - b^2) = 2

-- Define the proof problems

-- Proof Problem 1
theorem length_segment_AB
  (a b : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_eccentricity_eqn a b (sqrt 3 / 3))
  (h4 : focal_distance_eqn a b) :
  ∃ x1 x2 y1 y2, (x1 + x2 = 6 / 5) ∧ (x1 * x2 = -3 / 5) ∧ 
  sqrt((x1 - x2)^2 + (y1 - y2)^2) = (8 * sqrt 3) / 5 := 
sorry

-- Proof Problem 2
theorem reciprocal_sum_squares
  (a b : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : line_eqn = - (fun x => x) + 1)
  (h4 : ∃ x1 x2 y1 y2, x1 * x2 + y1 * y2 = 0 
    ∧ ellipse_eqn x1 y1 a b 
    ∧ ellipse_eqn x2 y2 a b) :
  1 / a^2 + 1 / b^2 = 2 := 
sorry

-- Proof Problem 3
theorem maximum_major_axis
  (a b e : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : e ∈ Set.Icc (1 / 2) (sqrt 2 / 2))
  (h4 : ellipse_eqn = fun (x y : ℝ) => (x^2 / a^2) + (y^2 / b^2) = 1)
  (h5 : ∃ x1 x2 y1 y2, x1 * x2 + y1 * y2 = 0 
    ∧ ellipse_eqn x1 y1 a b 
    ∧ ellipse_eqn x2 y2 a b) :
  2 * a = sqrt 6 := 
sorry

end length_segment_AB_reciprocal_sum_squares_maximum_major_axis_l8_8177


namespace average_monthly_balance_l8_8860

theorem average_monthly_balance
  (jan feb mar apr may : ℕ) 
  (Hjan : jan = 200)
  (Hfeb : feb = 300)
  (Hmar : mar = 100)
  (Hapr : apr = 250)
  (Hmay : may = 150) :
  (jan + feb + mar + apr + may) / 5 = 200 := 
  by
  sorry

end average_monthly_balance_l8_8860


namespace SimplifyAndRationalize_l8_8699

theorem SimplifyAndRationalize :
  ( (√3 / √7) * (√5 / √8) * (√6 / √9) ) = ( √35 / 42 ) :=
sorry

end SimplifyAndRationalize_l8_8699


namespace find_number_l8_8308

def crab := 2
def goat := 3
def bear := 4
def cat := 1
def chicken := 5

-- Conditions from Rows and Columns
def row4 := 5 * crab = 10
def col5 := 4 * crab + goat = 11
def row2 := 2 * goat + crab + 2 * bear = 16
def col2 := cat + bear + 2 * goat + crab = 13
def col3 := 2 * crab + 2 * chicken + goat = 17

-- The theorem to prove the final correct answer
theorem find_number : 
  row4 ∧ col5 ∧ row2 ∧ col2 ∧ col3 →
  nat.digits 10 [cat, chicken, crab, bear, goat] = [1, 5, 2, 4, 3] :=
by
  sorry

end find_number_l8_8308


namespace similar_triangles_iff_sides_proportional_l8_8304

theorem similar_triangles_iff_sides_proportional
  (a b c a1 b1 c1 : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a1 ∧ 0 < b1 ∧ 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) =
   Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end similar_triangles_iff_sides_proportional_l8_8304


namespace min_value_l8_8545

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  ∃ c : ℝ, c = 4 ∧ 
  ∀ x y : ℝ, (x = 1 / (a - 1) ∧ y = 4 / (b - 1)) → (x + y ≥ c) :=
sorry

end min_value_l8_8545


namespace largest_multiple_of_8_less_than_100_l8_8787

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8787


namespace cricket_player_average_l8_8067

theorem cricket_player_average :
  ∃ A : ℝ, (20 * A + 200) / 21 = A + 8 ∧ A = 32 :=
begin
  use 32,
  sorry
end

end cricket_player_average_l8_8067


namespace dice_prime_probability_l8_8456

def probability_three_primes (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem dice_prime_probability :
  let p := (5 / 12 : ℚ) in
  let n := 5 in
  let k := 3 in
  probability_three_primes n k p = 6125 / 24883 :=
by
  sorry

end dice_prime_probability_l8_8456


namespace Clea_escalator_time_l8_8628

theorem Clea_escalator_time :
  ∀ (c s d : ℝ), d = 80 * c → d = 20 * (c + s) → t = d / s := by
  intros c s d h1 h2
  have h3: s = 3 * c := by
    rw [←h1, ←h2]
    linarith
  rw [h1, h2, h3]
  have t_eval: t = 80 * c / (3 * c) := by
    norm_num
    rw [div_eq_mul_inv]
    field_simp
  exact eq.mpr sorry

#check Clea_escalator_time

end Clea_escalator_time_l8_8628


namespace problem1_problem2_l8_8578

-- Definitions to set up the conditions
def l1 (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l2 (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Problem 1: If l1 is parallel to l2, then b is in the range (-∞, -6) ∪ (-6, 0]
theorem problem1 (a b : ℝ) (h_parallel : ∀ (x y : ℝ), l1 a x y → l2 a b x y) : b ∈ set.Ioo (0 : ℝ) (-∞) ∪ set.Icc (-6 : ℝ) 0 :=
sorry

-- Problem 2: If l1 is perpendicular to l2, the minimum of |ab| is 2
theorem problem2 (a b : ℝ) (h_perp : ∀ (x y : ℝ), l1 a x y → l2 a b x y) : abs (a * b) ≥ 2 :=
sorry

end problem1_problem2_l8_8578


namespace intersection_with_complement_N_l8_8202

open Set Real

def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}
def complement_N : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem intersection_with_complement_N : M ∩ complement_N = Ico 2 3 :=
by {
  sorry
}

end intersection_with_complement_N_l8_8202


namespace bicycle_final_price_l8_8061

theorem bicycle_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (h1 : original_price = 200) (h2 : discount1 = 0.4) (h3 : discount2 = 0.2) :
  (original_price * (1 - discount1) * (1 - discount2)) = 96 :=
by
  -- sorry proof here
  sorry

end bicycle_final_price_l8_8061


namespace evaluate_expression_l8_8486

def a := (64 : ℝ) ^ (-1 / 3 : ℝ)
def b := (81 : ℝ) ^ (-1 / 2 : ℝ)
def result := a + b

theorem evaluate_expression : result = (13 / 36 : ℝ) :=
by 
  sorry

end evaluate_expression_l8_8486


namespace distinct_prime_factors_of_x_l8_8317

variable (x y : ℕ)

-- Definitions based on conditions
def gcd_has_9_distinct_primes : Prop :=
  (nat.gcd x y).prime_factors.to_finset.card = 9

def lcm_has_36_distinct_primes : Prop :=
  (nat.lcm x y).prime_factors.to_finset.card = 36

def x_has_fewer_prime_factors_than_y : Prop :=
  x.prime_factors.to_finset.card < y.prime_factors.to_finset.card

-- The statement to prove
theorem distinct_prime_factors_of_x (h1 : gcd_has_9_distinct_primes x y)
                                    (h2 : lcm_has_36_distinct_primes x y)
                                    (h3 : x_has_fewer_prime_factors_than_y x y) :
  x.prime_factors.to_finset.card ≤ 22 := 
sorry

end distinct_prime_factors_of_x_l8_8317


namespace find_rate_of_interest_l8_8408

-- Define the main problem components.
variables (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 

-- Define the given conditions.
def condition_2_years (P : ℝ) (r : ℝ) : Prop :=
  P * (1 + r / 100) ^ 2 = 3650

def condition_3_years (P : ℝ) (r : ℝ) : Prop :=
  P * (1 + r / 100) ^ 3 = 4015

-- Define the problem statement.
theorem find_rate_of_interest (P r : ℝ):
  (condition_2_years P r) → (condition_3_years P r) → r = 10 :=
begin
  -- Proof steps would go here.
  sorry,
end

end find_rate_of_interest_l8_8408


namespace interval_increasing_cos_2alpha_l8_8207

noncomputable def m (x : Real) : Vector := (sqrt 3 * cos x, -cos x)
noncomputable def n (x : Real) : Vector := (cos (x - π / 2), cos x)
noncomputable def f (x : Real) : Real := (m x).dotProduct (n x) + 0.5

theorem interval_increasing : ∀ (x : Real), 0 ≤ x ∧ x ≤ π / 3 → strict_mono_incr_on f (Icc 0 (π / 3)) :=
sorry

theorem cos_2alpha : ∀ (α : Real), (0 ≤ α ∧ α ≤ π / 4) ∧ (f α = 5 / 13) → cos (2 * α) = (12 * sqrt 3 - 5) / 26 :=
sorry

end interval_increasing_cos_2alpha_l8_8207


namespace trajectory_of_centroid_is_line_l8_8957

-- We define the points A and B, and a line l
variables {A B : ℝ × ℝ} {l : ℝ × ℝ → Prop}

-- We define a function that checks if a point P(x, y) lies on a line l.
def lies_on (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

-- Given conditions
variable hA : lies_on A l
variable hB : lies_on B l

-- Defining the centroid function for a triangle given three vertices
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Main theorem statement
theorem trajectory_of_centroid_is_line (C : ℝ × ℝ) (hC : lies_on C l) :
  ∃ m b : ℝ, ∀ C ∈ {p : ℝ × ℝ | lies_on p l}, let M := centroid A B C in M.2 = m * M.1 + b :=
sorry

end trajectory_of_centroid_is_line_l8_8957


namespace stream_speed_l8_8445

variable (v : ℝ) -- speed of the stream

-- Conditions
variable (swimming_speed : ℝ) (upstream_time_ratio : ℝ)
hypothesis (H1 : swimming_speed = 1.5) -- man's swimming speed in still water
hypothesis (H2 : upstream_time_ratio = 2) -- upstream takes twice as long as downstream

theorem stream_speed (H1 : swimming_speed = 1.5) (H2 : upstream_time_ratio = 2) : v = 0.5 :=
by
  sorry

end stream_speed_l8_8445


namespace sin_C_value_ratio_equality_l8_8538

variable (A B C : ℝ) -- Angles of the triangle
variable (a b c S : ℝ) -- Sides and area of the triangle

-- Condition definitions
def S_condition : Prop := S = (a + b)^2 - c^2
def sum_condition : Prop := a + b = 4

-- Proving statements
theorem sin_C_value (h1 : S_condition) (h2 : sum_condition) : 
  ∃ k, sin C = k ∧ k = 8 / 17 := by
  sorry

theorem ratio_equality (h : a^2 - b^2 = c^2 * (sin A * (cos B / sin B) - sin B * (cos A / sin A)) ) :
  ∃ k, (a^2 - b^2) / c^2 = k ∧ k = sin (A - B) / sin C := by
  sorry

end sin_C_value_ratio_equality_l8_8538


namespace find_CD_l8_8129

-- Define the cyclic pentagon with the given properties
variables {A B C D E : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables {dist : A → A → ℝ}
variables [AB : dist A B = 15]
variables [BC : dist B C = 20]
variables [EqABDEREA : dist A B = dist D E ∧ dist A B = dist E A]
variables [CyclicPentagon : IsCyclicPentagon A B C D E]
variables [RightAngle : dist A B C = 90]

-- Define the proof goal
theorem find_CD : dist C D = 20 :=
sorry

end find_CD_l8_8129


namespace volume_relation_l8_8854

variable (r : ℝ)

-- Define the volumes
def volumeCone (r : ℝ) : ℝ := (1 / 3) * π * r^3
def volumeCylinder (r : ℝ) : ℝ := 2 * π * r^3
def volumeHemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3

-- Define the statement to be proven
theorem volume_relation (r : ℝ) : volumeHemisphere r + 2 * volumeCone r ≠ volumeCylinder r := 
by 
  sorry

end volume_relation_l8_8854


namespace option_b_correct_l8_8795

theorem option_b_correct :
  ∀ (a b : ℝ), sqrt 2 * sqrt 3 = sqrt 6 :=
by sorry

end option_b_correct_l8_8795


namespace half_AB_correct_l8_8991

noncomputable theory

variables (OA OB : ℝ × ℝ)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

def half_AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (AB OA OB).1 / 2, (AB OA OB).2 / 2

theorem half_AB_correct (hOA : OA = (1, -2)) (hOB : OB = (-3, 2)) :
  half_AB OA OB = (-2, 2) :=
by
  simp [hOA, hOB, AB, half_AB]
  sorry

end half_AB_correct_l8_8991


namespace number_of_two_digit_integers_with_conditions_l8_8970

def digits := {2, 4, 7, 9}
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem number_of_two_digit_integers_with_conditions :
  (∑ d₁ in digits, ∑ d₂ in digits \ {d₁}, if is_odd d₁ then 1 else 0) = 6 := by
sorry

end number_of_two_digit_integers_with_conditions_l8_8970


namespace tennis_tournament_points_l8_8237

theorem tennis_tournament_points (n : ℕ) (hn : 4 < n) :
  let total_participants := 2^n + 4 in 
  let participants_with_3_points := (nat.choose n 3) + 1 in 
  (C n 3 + 1) = participants_with_3_points :=
by
  sorry

end tennis_tournament_points_l8_8237


namespace find_k_l8_8904

theorem find_k (k : ℝ) (hk : 0 < k) (slope_eq : (2 - k) / (k - 1) = k^2) : k = 1 :=
by sorry

end find_k_l8_8904


namespace switches_finite_operations_l8_8021

variable (k : ℕ)
variables (Directions : Fin 4 → Fin 4 → Fin 4 → Prop) -- A predicate for directions

theorem switches_finite_operations (k : ℕ)
  (switches : Fin k → Fin 4) 
  (operation : ∀ {i : Fin (k-2)}, Directions (switches i) (switches ⟨i+1, sorry⟩) (switches ⟨i+2, sorry⟩) → (Fin k → Fin 4)) :
  ∃ n : ℕ, n = natural_bound ∧ ∀ m : ℕ, m > n → ¬ ∃ i : Fin (k-2), Directions (switches i) (switches ⟨i+1, sorry⟩) (switches ⟨i+2, sorry⟩) :=
sorry

end switches_finite_operations_l8_8021


namespace polynomial_sum_l8_8653

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l8_8653


namespace radius_sphere_equiv_l8_8737

noncomputable def radius_of_sphere (d : ℝ) : ℝ :=
  d * sqrt 13 / 6 

theorem radius_sphere_equiv (d : ℝ) (R : ℝ) (h1 : d > 0) 
  (h2 : ∀ a b c : ℝ, 
        a = b ∧ b = c ∧ c = a) 
  (h3 : ∀ θ : ℝ, θ = 60 ∧ (cos θ - opp reciprocal segment)):
  radius_of_sphere d = R :=  sorry

end radius_sphere_equiv_l8_8737


namespace largest_integer_m_dividing_30_factorial_l8_8509

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l8_8509


namespace tom_found_seashells_l8_8018

theorem tom_found_seashells : ∀ (days : ℕ) (seashells_per_day : ℕ), days = 5 ∧ seashells_per_day = 7 → days * seashells_per_day = 35 := 
by
  intros days seashells_per_day h
  cases h with h_days h_seashells_per_day
  rw [h_days, h_seashells_per_day]
  exact nat.mul_comm 5 7 ▸ rfl

end tom_found_seashells_l8_8018


namespace total_time_to_complete_project_l8_8101

-- Define the initial conditions
def initial_people : ℕ := 6
def initial_days : ℕ := 35
def fraction_completed : ℚ := 1 / 3

-- Define the additional conditions after more people joined
def additional_people : ℕ := initial_people
def total_people : ℕ := initial_people + additional_people
def remaining_fraction : ℚ := 1 - fraction_completed

-- Total time taken to complete the project
theorem total_time_to_complete_project (initial_people initial_days additional_people : ℕ) (fraction_completed remaining_fraction : ℚ)
  (h1 : initial_people * initial_days * fraction_completed = 1/3) 
  (h2 : additional_people = initial_people) 
  (h3 : total_people = initial_people + additional_people)
  (h4 : remaining_fraction = 1 - fraction_completed) : 
  (initial_days + (remaining_fraction / (total_people * (fraction_completed / (initial_people * initial_days)))) = 70) :=
sorry

end total_time_to_complete_project_l8_8101


namespace bonus_for_each_positive_review_l8_8993

theorem bonus_for_each_positive_review (wage_per_hour : ℕ) (ride_bonus : ℕ) 
  (review_bonus : ℕ) (hours_worked : ℕ) (rides_given : ℕ) 
  (gallons_gas : ℕ) (cost_per_gallon : ℕ) (total_owed : ℕ) 
  (reviews : ℕ) 
  (h1 : wage_per_hour = 15) 
  (h2 : ride_bonus = 5) 
  (h3 : hours_worked = 8) 
  (h4 : rides_given = 3)
  (h5 : gallons_gas = 17) 
  (h6 : cost_per_gallon = 3) 
  (h7 : reviews = 2) 
  (h8 : total_owed = 226) : 
  review_bonus = 20 := 
by
  -- Definitions
  let hourly_wage := hours_worked * wage_per_hour
  let rides_earnings := rides_given * ride_bonus
  let gas_reimbursement := gallons_gas * cost_per_gallon

  -- Calculations
  have sum_earnings : hourly_wage + rides_earnings + gas_reimbursement = 186 := sorry

  have remaining_amount : total_owed - (hourly_wage + rides_earnings + gas_reimbursement) = 40 := sorry

  have final_bonus : remaining_amount / reviews = 20 := sorry

  -- Result
  exact final_bonus

-- Dummy implementation to compile successfully
noncomputable def dummy : ℕ := 0

end bonus_for_each_positive_review_l8_8993


namespace largest_multiple_of_8_less_than_100_l8_8779

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8779


namespace drops_per_minute_l8_8358

-- Definitions based on conditions
def drop_volume : ℕ := 20  -- Each drop is 20 ml
def pot_capacity : ℕ := 3000  -- Pot capacity in ml (3 liters)
def fill_time : ℕ := 50  -- Time to fill the pot in minutes

-- Theorem statement
theorem drops_per_minute (drop_volume pot_capacity fill_time : ℕ) (h1 : drop_volume = 20) (h2 : pot_capacity = 3000) (h3 : fill_time = 50) : pot_capacity / drop_volume / fill_time = 3 :=
by
  rw [h1, h2, h3]
  sorry

end drops_per_minute_l8_8358


namespace bowling_ball_weight_l8_8927

theorem bowling_ball_weight (b c : ℝ) (h1 : c = 36) (h2 : 5 * b = 4 * c) : b = 28.8 := by
  sorry

end bowling_ball_weight_l8_8927


namespace solve_tangents_equation_l8_8968

open Real

def is_deg (x : ℝ) : Prop := ∃ k : ℤ, x = 30 + 180 * k

theorem solve_tangents_equation (x : ℝ) (h : tan (x * π / 180) * tan (20 * π / 180) + tan (20 * π / 180) * tan (40 * π / 180) + tan (40 * π / 180) * tan (x * π / 180) = 1) :
  is_deg x :=
sorry

end solve_tangents_equation_l8_8968


namespace relationship_among_abc_l8_8550

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x^(-1/3)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x^(1/2)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x^3

theorem relationship_among_abc : c < b ∧ b < a := by
  -- proof would go here
  sorry

end relationship_among_abc_l8_8550


namespace differences_l8_8149

def seq (n : ℕ) : ℕ := n^2 + 1

def first_diff (n : ℕ) : ℕ := (seq (n + 1)) - (seq n)

def second_diff (n : ℕ) : ℕ := (first_diff (n + 1)) - (first_diff n)

def third_diff (n : ℕ) : ℕ := (second_diff (n + 1)) - (second_diff n)

theorem differences (n : ℕ) : first_diff n = 2 * n + 1 ∧ 
                             second_diff n = 2 ∧ 
                             third_diff n = 0 := by 
  sorry

end differences_l8_8149


namespace min_value_cos2x_minus_3cosx_plus_2_l8_8923

theorem min_value_cos2x_minus_3cosx_plus_2 : 
  ∀ (x : ℝ), -1 ≤ Real.cos x ∧ Real.cos x ≤ 1 → 
    ∃ t ∈ set.Icc (-1 : ℝ) (1 : ℝ), y = t^2 - 3 * t + 2 ∧ y = 0
    :=
by
  sorry

end min_value_cos2x_minus_3cosx_plus_2_l8_8923


namespace jack_needs_more_money_l8_8631

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l8_8631


namespace cube_splitting_height_l8_8600

/-- If we split a cube with an edge of 1 meter into small cubes with an edge of 1 millimeter,
what will be the height of a column formed by stacking all the small cubes one on top of another? -/
theorem cube_splitting_height :
  let edge_meter := 1
  let edge_mm := 1000
  let num_cubes := (edge_meter * edge_mm) ^ 3
  let height_mm := num_cubes * edge_mm
  let height_km := height_mm / (1000 * 1000 * 1000)
  height_km = 1000 :=
by
  sorry

end cube_splitting_height_l8_8600


namespace radii_of_inscribed_circles_l8_8373

-- Definitions for given conditions
variable (r a : ℝ)

-- Statement of the problem in Lean 4
theorem radii_of_inscribed_circles
  (h_r_pos : r > 0) (h_a_pos : a > r) :
  let x := a * r / (a - r) in
  let y := a * a * r / (a - r) / (a - r) in
  (x = a * r / (a - r)) ∧ (y = a^2 * r / (a - r)^2) :=
by
  -- Proof is omitted
  sorry

end radii_of_inscribed_circles_l8_8373


namespace percent_of_x_is_y_in_terms_of_z_l8_8213

theorem percent_of_x_is_y_in_terms_of_z (x y z : ℝ) (h1 : 0.7 * (x - y) = 0.3 * (x + y))
    (h2 : 0.6 * (x + z) = 0.4 * (y - z)) : y / x = 0.4 :=
  sorry

end percent_of_x_is_y_in_terms_of_z_l8_8213


namespace difference_between_hexagonal_and_pentagonal_prism_vertices_l8_8147

-- Definitions based on the conditions
def num_vertices_hexagon : Nat := 6
def num_vertices_pentagon : Nat := 5

def num_vertices_prism (num_vertices_polygon : Nat) : Nat :=
  2 * num_vertices_polygon

def num_vertices_hexagonal_prism : Nat := num_vertices_prism num_vertices_hexagon
def num_vertices_pentagonal_prism : Nat := num_vertices_prism num_vertices_pentagon

-- The problem statement to be proved
theorem difference_between_hexagonal_and_pentagonal_prism_vertices :
  num_vertices_hexagonal_prism - num_vertices_pentagonal_prism = 2 :=
by
  calc
    num_vertices_hexagonal_prism - num_vertices_pentagonal_prism
      = (2 * num_vertices_hexagon) - (2 * num_vertices_pentagon) : by rfl
  ... = 2 * (num_vertices_hexagon - num_vertices_pentagon) : by sorry -- algebraic manipulation
  ... = 2 * (6 - 5) : by sorry -- substituting values
  ... = 2 * 1 : by sorry -- simple arithmetic
  ... = 2 : by sorry

end difference_between_hexagonal_and_pentagonal_prism_vertices_l8_8147


namespace male_athletes_drawn_in_sample_l8_8857

theorem male_athletes_drawn_in_sample (total_male total_female sample_size : ℕ) 
  (h_male : total_male = 48) 
  (h_female : total_female = 36) 
  (h_sample : sample_size = 21) : 
  let total_athletes := total_male + total_female in
  let fraction_sample_size := sample_size.toRat / total_athletes.toRat in
  let number_of_males_drawn := (fraction_sample_size * total_male.toRat).toNat in
  number_of_males_drawn = 12 := by
  sorry

end male_athletes_drawn_in_sample_l8_8857


namespace intersection_of_sets_l8_8594

def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | x^2 > 1 }
def C := { x : ℝ | 1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : 
  (A ∩ B) = C := 
by sorry

end intersection_of_sets_l8_8594


namespace area_of_region_l8_8136

open Set Real Topology

theorem area_of_region :
  let region : Set (ℝ × ℝ) := {p | (p.1 - abs p.1)^2 + (p.2 - abs p.2)^2 ≤ 4 ∧ p.2 + 2 * p.1 ≤ 0}
  measure_theory.measure_space.volume (region) = (5 + π) / 4 :=
by
  -- The proof will verify that the area of the defined region is (5 + π) / 4.
  sorry

end area_of_region_l8_8136


namespace largest_divisor_of_consecutive_even_product_l8_8650

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℕ, k = 8 ∧ 8 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) := 
by {
  intros n,
  use 8,
  split,
  { refl, },
  { sorry, }  -- Proof placeholder
}

end largest_divisor_of_consecutive_even_product_l8_8650


namespace op_7_3_eq_70_l8_8166

noncomputable def op (x y : ℝ) : ℝ := sorry

axiom ax1 : ∀ x : ℝ, op x 0 = x
axiom ax2 : ∀ x y : ℝ, op x y = op y x
axiom ax3 : ∀ x y : ℝ, op (x + 1) y = (op x y) + y + 2

theorem op_7_3_eq_70 : op 7 3 = 70 := by
  sorry

end op_7_3_eq_70_l8_8166


namespace largest_possible_m_value_l8_8364

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_possible_m_value :
  ∃ (m x y : ℕ), is_three_digit m ∧ is_prime x ∧ is_prime y ∧ x ≠ y ∧
  x < 10 ∧ y < 10 ∧ is_prime (10 * x - y) ∧ m = x * y * (10 * x - y) ∧ m = 705 := sorry

end largest_possible_m_value_l8_8364


namespace calculate_H_iterates_l8_8078

def H (x : ℝ) : ℝ :=
  if x = 2 then -1
  else if x = -1 then 3
  else if x = 3 then 3
  else 0 -- This handles any other cases which are not specificied, assuming default behavior

theorem calculate_H_iterates :
  H (H (H (H (H 2)))) = 3 :=
by
  -- proof would go here, but for now we just add sorry to skip the proof
  sorry

end calculate_H_iterates_l8_8078


namespace jack_money_proof_l8_8637

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l8_8637


namespace quadrilateral_propositions_l8_8796

theorem quadrilateral_propositions :
  (∀ (Q : Type) [quadrilateral Q], (diagonals_bisect Q) → parallelogram Q) ∧
  (∀ (Q : Type) [quadrilateral Q], (diagonals_perpendicular Q) → ¬rhombus Q) ∧
  (∀ (P : Type) [parallelogram P], (diagonals_equal P) → ¬rhombus P) ∧
  (∀ (P : Type) [parallelogram P], (right_angle P) → rectangle P) := 
  by sorry

-- Definitions for the predicates used in the statement for completeness
class quadrilateral (Q : Type) :=
(diagonals_bisect : Q → Prop)
(diagonals_perpendicular : Q → Prop)

class parallelogram (P : Type) extends quadrilateral P :=
(diagonals_equal : P → Prop)
(right_angle : P → Prop)

class rhombus (R : Type) extends parallelogram R :=
()

class rectangle (R : Type) extends parallelogram R :=
()

end quadrilateral_propositions_l8_8796


namespace ex_one_divisible_by_10_l8_8458

theorem ex_one_divisible_by_10 {a b c d e : ℤ} (h_distinct : list.nodup [a, b, c, d, e])
  (h_cond : ∀ (x y z : ℤ), x ∈ [a, b, c, d, e] → y ∈ [a, b, c, d, e] → z ∈ [a, b, c, d, e] → x ≠ y → y ≠ z → x ≠ z → 10 ∣ (x * y * z)) :
  ∃ x ∈ [a, b, c, d, e], 10 ∣ x :=
begin
  sorry
end

end ex_one_divisible_by_10_l8_8458


namespace math_problem_equivalency_l8_8193

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x + a * Real.log x

-- 1. Monotonic intervals
def monotonic_intervals (a : ℝ) : Prop :=
  (a >= 0 → ∀ x y : ℝ, 0 < x → x < y → f x a < f y a) ∧
  (a < 0 → (∀ x : ℝ, 0 < x → x < -a → f x a > f (-a) a) ∧ (∀ y : ℝ, y > -a → f (-a) a < f y a))

-- 2. Range of a for f(x) > 0 on [1, 2]
def range_of_a : Prop :=
  ∀ a : ℝ, 
    (∀ x : ℝ, 1 <= x ∧ x <= 2 → f x a > 0) ↔ a > -2 / Real.log 2

-- 3. Number of tangent lines through P(1, 3)
def num_tangent_lines (a : ℝ) : Prop :=
  ∃ n : ℕ, 
    (a > 0 → n = 2) ∧ (a <= 0 → n = 0)

-- The main theorem combining all the statements
theorem math_problem_equivalency : ∀ a : ℝ,
  monotonic_intervals a ∧
  range_of_a ∧
  num_tangent_lines a :=
sorry

end math_problem_equivalency_l8_8193


namespace probability_X1_lt_X2_lt_X3_is_1_6_l8_8262

noncomputable def probability_X1_lt_X2_lt_X3 (n : ℕ) (h : n ≥ 3) : ℚ :=
if h : n ≥ 3 then
  1/6
else
  0

theorem probability_X1_lt_X2_lt_X3_is_1_6 (n : ℕ) (h : n ≥ 3) :
  probability_X1_lt_X2_lt_X3 n h = 1/6 :=
sorry

end probability_X1_lt_X2_lt_X3_is_1_6_l8_8262


namespace circle_positions_n_l8_8357

theorem circle_positions_n (n : ℕ) (h1 : n ≥ 23) (h2 : (23 - 7) * 2 + 2 = n) : n = 32 :=
sorry

end circle_positions_n_l8_8357


namespace total_accepted_cartons_l8_8259

-- Definitions for the number of cartons delivered and damaged for each customer
def cartons_delivered_first_two : Nat := 300
def cartons_delivered_last_three : Nat := 200

def cartons_damaged_first : Nat := 70
def cartons_damaged_second : Nat := 50
def cartons_damaged_third : Nat := 40
def cartons_damaged_fourth : Nat := 30
def cartons_damaged_fifth : Nat := 20

-- Statement to prove
theorem total_accepted_cartons :
  let accepted_first := cartons_delivered_first_two - cartons_damaged_first
  let accepted_second := cartons_delivered_first_two - cartons_damaged_second
  let accepted_third := cartons_delivered_last_three - cartons_damaged_third
  let accepted_fourth := cartons_delivered_last_three - cartons_damaged_fourth
  let accepted_fifth := cartons_delivered_last_three - cartons_damaged_fifth
  accepted_first + accepted_second + accepted_third + accepted_fourth + accepted_fifth = 990 :=
by
  sorry

end total_accepted_cartons_l8_8259


namespace sum_of_y_values_is_5_over_2_l8_8522

noncomputable def sum_real_y_values : ℝ :=
∑ y in { y : ℝ | ∃ x : ℝ, x^2 + x^2 * y^2 + x^2 * y^4 = 525 ∧ x + x * y + x * y^2 = 35 }, y

theorem sum_of_y_values_is_5_over_2 :
  sum_real_y_values = 5 / 2 :=
sorry

end sum_of_y_values_is_5_over_2_l8_8522


namespace optimal_area_is_2500_l8_8376

noncomputable def optimal_garden_area : ℝ :=
  let A (l w : ℝ) : ℝ := l * w in
  let perimeter (l w : ℝ) : Prop := 2 * l + 2 * w = 200 in
  let length_constraint (l : ℝ) : Prop := l ≥ 60 in
  let width_constraint (w : ℝ) : Prop := w ≥ 30 in
  (arg_max (λ l w, A l w)
    (λ l w, perimeter l w ∧ length_constraint l ∧ width_constraint w)).2

theorem optimal_area_is_2500 : optimal_garden_area = 2500 := by
  sorry

end optimal_area_is_2500_l8_8376


namespace solve_equation_l8_8214

theorem solve_equation (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 / x + 4 / y = 1) : 
  x = 3 * y / (y - 4) :=
sorry

end solve_equation_l8_8214


namespace crows_and_trees_l8_8461

variable (x y : ℕ)

theorem crows_and_trees (h1 : x = 3 * y + 5) (h2 : x = 5 * (y - 1)) : 
  (x - 5) / 3 = y ∧ x / 5 = y - 1 :=
by
  sorry

end crows_and_trees_l8_8461


namespace largest_integer_m_dividing_30_factorial_l8_8507

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l8_8507


namespace find_natural_numbers_l8_8144

theorem find_natural_numbers (n : ℕ) :
  (∀ k : ℕ, k^2 + ⌊ (n : ℝ) / (k^2 : ℝ) ⌋ ≥ 1991) ∧
  (∃ k_0 : ℕ, k_0^2 + ⌊ (n : ℝ) / (k_0^2 : ℝ) ⌋ < 1992) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by sorry

end find_natural_numbers_l8_8144


namespace area_of_border_l8_8083

theorem area_of_border (height_painting width_painting border_width : ℕ)
    (area_painting framed_height framed_width : ℕ)
    (H1 : height_painting = 12)
    (H2 : width_painting = 15)
    (H3 : border_width = 3)
    (H4 : area_painting = height_painting * width_painting)
    (H5 : framed_height = height_painting + 2 * border_width)
    (H6 : framed_width = width_painting + 2 * border_width)
    (area_framed : ℕ)
    (H7 : area_framed = framed_height * framed_width) :
    area_framed - area_painting = 198 := 
sorry

end area_of_border_l8_8083


namespace relationship_ab_c_l8_8943

def a := 0.8 ^ 0.8
def b := 0.8 ^ 0.9
def c := 1.2 ^ 0.8

theorem relationship_ab_c : c > a ∧ a > b := 
by
  -- The proof would go here
  sorry

end relationship_ab_c_l8_8943


namespace vector_subtraction_simplification_vector_projection_cosine_from_sine_trigonometric_statements_l8_8055

-- Problem (1)
theorem vector_subtraction_simplification (AB CD BE DE AC : ℝ^2) :
  (AB - CD) + (BE - DE) = AC := 
sorry

-- Problem (2)
theorem vector_projection (a b : ℝ × ℝ)
  (ha : a = (2, 3))
  (hb : b = (-4, 7)) :
  let projection := (a.1 * b.1 + a.2 * b.2) / real.sqrt ((b.1)^2 + (b.2)^2)
  in projection = sqrt 65 / 5 := 
sorry

-- Problem (3)
theorem cosine_from_sine (a : ℝ) (h : real.sin (110 * real.pi / 180) = a) :
  real.cos (20 * real.pi / 180) = a := 
sorry

-- Problem (4)
theorem trigonometric_statements :
  let f(x : ℝ) := real.cos ((2 / 3) * x + real.pi / 2)
  let g(x : ℝ) := real.sin (2 * x + (5 / 4) * real.pi)
  bool :=
  (¬(real.tan x is_strictly_monotone_in its_domain)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g x = g (x + real.pi/8)) :=
sorry

end vector_subtraction_simplification_vector_projection_cosine_from_sine_trigonometric_statements_l8_8055


namespace horizontal_asymptote_of_f_l8_8479

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 + 4) / (4 * x^2 + 2 * x - 1)

theorem horizontal_asymptote_of_f : 
  tendsto (f) atTop (𝓝 (7 / 4)) :=
by
  sorry

end horizontal_asymptote_of_f_l8_8479


namespace equivalent_operations_l8_8399

theorem equivalent_operations:
  (∀ x : ℚ, (x * (5/6)) / (2/7) = x * (35/12)) :=
by
  intros x
  calc
    (x * (5/6)) / (2/7)
        = x * (5/6) * (7/2)   : by rw [div_eq_mul_inv]
    ... = x * ((5 * 7) / (6 * 2)) : by rw [mul_div_assoc, mul_comm 7, div_mul_cancel, div_mul_cancel]
    ... = x * (35 / 12) : by rw [mul_assoc, ← mul_comm 5]

#check equivalent_operations

end equivalent_operations_l8_8399


namespace pyramid_inscribed_sphere_distance_l8_8610

noncomputable def pyramid_distance (AB CD AC: ℝ) : ℝ :=
  if h : AB = 12 ∧ CD = 4 ∧ AC = 7 then
    (3 * Real.sqrt 13) / (Real.sqrt 5 + Real.sqrt 13)
  else
    0

theorem pyramid_inscribed_sphere_distance :
  let AB := 12
  let CD := 4
  let AC := 7 in
  pyramid_distance AB CD AC = (3 * Real.sqrt 13) / (Real.sqrt 5 + Real.sqrt 13) := by
  sorry

end pyramid_inscribed_sphere_distance_l8_8610


namespace margo_total_distance_l8_8680

-- Definitions of the given conditions
def time_to_friend : ℝ := 15 / 60 -- hours
def time_to_home : ℝ := 30 / 60 -- hours
def speed_to_friend : ℝ := 5   -- miles per hour
def speed_to_home : ℝ := 3    -- miles per hour
def average_speed_total : ℝ := 3.6 -- miles per hour

-- The proof problem
theorem margo_total_distance :
  let distance_to_friend := speed_to_friend * time_to_friend
  let distance_to_home := speed_to_home * time_to_home
  let total_distance := distance_to_friend + distance_to_home
  total_distance = 2.75 :=
by
  sorry -- proof goes here

end margo_total_distance_l8_8680


namespace B4E_base16_to_base10_l8_8124

theorem B4E_base16_to_base10 : 
  let B := 11
  let four := 4
  let E := 14
  (B * 16^2 + four * 16^1 + E * 16^0) = 2894 := 
by 
  let B := 11
  let four := 4
  let E := 14
  calc
    B * 16^2 + four * 16^1 + E * 16^0 = 11 * 256 + 4 * 16 + 14 : by rfl
    ... = 2816 + 64 + 14 : by rfl
    ... = 2894 : by rfl

end B4E_base16_to_base10_l8_8124


namespace original_numerical_expression_l8_8231

-- Define a proof that the expression satisfies the given conditions
theorem original_numerical_expression (A B C D : ℕ) (AA BCDA : ℕ) : 
    (AA = 11 * A) →
    (2018 * 10 + A = 201_0 + 18 * 10 + A) →
    (BCDA = 100 * B + 10 * C + D) →
    (2018 * 10 + A % BCDA = AA) :=
begin
    sorry
end

end original_numerical_expression_l8_8231


namespace abs_of_neg_one_third_l8_8349

theorem abs_of_neg_one_third : abs (- (1 / 3)) = (1 / 3) := by
  sorry

end abs_of_neg_one_third_l8_8349


namespace wyatt_bought_4_cartons_of_juice_l8_8404

/-- 
Wyatt's mother gave him $74 to go to the store.
Wyatt bought 5 loaves of bread, each costing $5.
Each carton of orange juice cost $2.
Wyatt has $41 left.
We need to prove that Wyatt bought 4 cartons of orange juice.
-/
theorem wyatt_bought_4_cartons_of_juice (initial_money spent_money loaves_price juice_price loaves_qty money_left juice_qty : ℕ)
  (h1 : initial_money = 74)
  (h2 : money_left = 41)
  (h3 : loaves_price = 5)
  (h4 : juice_price = 2)
  (h5 : loaves_qty = 5)
  (h6 : spent_money = initial_money - money_left)
  (h7 : spent_money = loaves_qty * loaves_price + juice_qty * juice_price) :
  juice_qty = 4 :=
by
  -- the proof would go here
  sorry

end wyatt_bought_4_cartons_of_juice_l8_8404


namespace trig_identity_sin_cos_l8_8526

theorem trig_identity_sin_cos
  (a : ℝ)
  (h : Real.sin (Real.pi / 3 - a) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - a) = -1 / 3 :=
by
  sorry

end trig_identity_sin_cos_l8_8526


namespace all_positive_7_numbers_l8_8627

theorem all_positive_7_numbers
  (a : ℕ → ℝ)
  (h : ∀ (i j k : ℕ) (H1 : i < j) (H2 : j < k) (H3 : k < 7), 
       a i + a j + a k < (∑ n in finset.range 7, a n) - (a i + a j + a k)) :
  ∀ n, n < 7 → 0 < a n :=
by
  sorry

end all_positive_7_numbers_l8_8627


namespace range_of_x1_plus_x2_l8_8527

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else Real.exp x

noncomputable def F (x : ℝ) : ℝ :=
  (f x)^2

theorem range_of_x1_plus_x2 (a : ℝ) (h1 : a > 1) (h2 : ∀ x, F x = a) : -1 < (-Real.sqrt a + Real.log a / 2) :=
  sorry

end range_of_x1_plus_x2_l8_8527


namespace jack_needs_more_money_l8_8633

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l8_8633


namespace smallest_positive_period_of_f_is_pi_extreme_values_of_f_monotonically_increasing_interval_l8_8975

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)

theorem smallest_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π :=
sorry

theorem extreme_values_of_f :
  ∃ a b, (∀ x, f x ≤ a ∧ f x ≥ b) ∧ a = 2 ∧ b = -2 :=
sorry

theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x, (k * π - π / 3) ≤ x ∧ x ≤ (k * π + π / 6) → ∀ y, x ≤ y ∧ y ≤ (k * π + π / 6) → f x ≤ f y :=
sorry

end smallest_positive_period_of_f_is_pi_extreme_values_of_f_monotonically_increasing_interval_l8_8975


namespace solve_for_x_l8_8754

theorem solve_for_x :
  ∀ (x : ℚ), (let δ := (λ x, 3 * x + 8) in let φ := (λ x, 8 * x + 10) in δ (φ x) = 9) ↔ x = -29 / 24 := by
  sorry

end solve_for_x_l8_8754


namespace common_root_eq_one_l8_8559

theorem common_root_eq_one (a b : ℝ) (m : ℝ) :
  (m^2 + a*m + b = 0) → (m^2 + b*m + a = 0) → m = 1 :=
by
  intros h1 h2
  have h := eq_sub_of_add_eq (h1 - h2)
  simp at h
  sorry

end common_root_eq_one_l8_8559


namespace sum_sin_mod_89_l8_8472

noncomputable def complex_sum : ℤ :=
  ∑ k in Finset.range 11, (Real.sin (2^(k + 4) * Real.pi / 89) / 
                           Real.sin (2^k * Real.pi / 89))

theorem sum_sin_mod_89 : complex_sum = -2 :=
sorry

end sum_sin_mod_89_l8_8472


namespace tan_angle_sum_identity_l8_8184

theorem tan_angle_sum_identity 
  (α : ℝ) 
  (h1 : α ∈ Ioo (π / 2) π) 
  (h2 : Real.sin α = 5 / 13) : 
  Real.tan (α + π / 4) = 7 / 17 := 
sorry

end tan_angle_sum_identity_l8_8184


namespace calculate_slacks_l8_8035

theorem calculate_slacks (blouses skirts total_clothes : ℕ) (percent_blouses percent_skirts percent_slacks : ℝ)
  (in_hamper_blouses in_hamper_skirts in_hamper_slacks : ℕ)
  (H1 : blouses = 12)
  (H2 : skirts = 6)
  (H3 : percent_blouses = 0.75)
  (H4 : percent_skirts = 0.5)
  (H5 : percent_slacks = 0.25)
  (H6 : in_hamper_blouses = (percent_blouses * blouses).nat_abs)
  (H7 : in_hamper_skirts = (percent_skirts * skirts).nat_abs)
  (H8 : total_clothes = in_hamper_blouses + in_hamper_skirts + in_hamper_slacks)
  (H9 : total_clothes = 14)
  : blouses = 12 → skirts = 6 → in_hamper_blouses = 9 → in_hamper_skirts = 3 → in_hamper_slacks = 2 → (2 / percent_slacks).nat_abs = 8 := 
by
  sorry

end calculate_slacks_l8_8035


namespace track_length_is_450_l8_8465

theorem track_length_is_450 (x : ℝ) (d₁ : ℝ) (d₂ : ℝ)
  (h₁ : d₁ = 150)
  (h₂ : x - d₁ = 120)
  (h₃ : d₂ = 200)
  (h₄ : ∀ (d₁ d₂ : ℝ) (t₁ t₂ : ℝ), t₁ / t₂ = d₁ / d₂)
  : x = 450 := by
  sorry

end track_length_is_450_l8_8465


namespace max_sin_sum_in_triangle_l8_8730

theorem max_sin_sum_in_triangle
  (A B C : ℝ)
  (h_sum_angles : A + B + C = Real.pi)
  (h_A_in : 0 < A ∧ A < Real.pi)
  (h_B_in : 0 < B ∧ B < Real.pi)
  (h_C_in : 0 < C ∧ C < Real.pi)
  (convex_sin : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < Real.pi → 0 < x2 ∧ x2 < Real.pi → (sin x1 + sin x2) / 2 ≤ sin ((x1 + x2) / 2))
  : sin A + sin B + sin C ≤ 3 * sin (Real.pi / 3) := 
begin
  sorry
end

end max_sin_sum_in_triangle_l8_8730


namespace participants_with_3_points_l8_8236

theorem participants_with_3_points (n : ℕ) (h : n > 4) :
  ∃ (C : ℕ → ℕ → ℕ), (∑ k in finset.range (n + 1), C n k = 2^n) →
    (number_of_participants = 2^n + 4) →
    (number_of_3_points_scorers = C n 3 + 1) :=
by sorry

end participants_with_3_points_l8_8236


namespace p_plus_q_l8_8721

theorem p_plus_q (p q : ℕ) (a : ℚ)
  (h_a : a = p / q)
  (h_coprime : Nat.coprime p q)
  (sum_x : (∑ x in { x : ℝ | ∃ k : ℤ, k ≤ x ∧ x < k + 1 ∧ k * (x - k) = a * x^2}, x) = 540)
  : p + q = 1331 := sorry

end p_plus_q_l8_8721


namespace women_entered_city_Y_l8_8062

theorem women_entered_city_Y :
  ∀ (M W : ℕ), (W = M / 2) ∧ (M + W = 72) ∧ (M - 16 = W) →
  let women_entered := W - (M / 2)
  in women_entered = 8 :=
by
  intros M W h
  sorry

end women_entered_city_Y_l8_8062


namespace Tim_Linda_Mow_Lawn_l8_8251

theorem Tim_Linda_Mow_Lawn :
  let tim_time := 1.5
  let linda_time := 2
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let combined_rate := tim_rate + linda_rate
  let combined_time_hours := 1 / combined_rate
  let combined_time_minutes := combined_time_hours * 60
  combined_time_minutes = 51.43 := 
by
    sorry

end Tim_Linda_Mow_Lawn_l8_8251


namespace monotonic_intervals_of_f_range_of_g_x1_x2_l8_8175

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, (0 < x ∧ x < 1 / 2 → deriv (λ x : ℝ, (1 / 2) * log x - x) x > 0) ∧
          (1 / 2 < x ∧ x < +∞ → deriv (λ x : ℝ, (1 / 2) * log x - x) x < 0) :=
by
  sorry

theorem range_of_g_x1_x2 (x1 x2 : ℝ) (m : ℝ) (h1 : 0 < x1)
    (h2 : x1 ≠ x2) (h3 : m > 0 ∧ m < 1 / 4) :
  let g := λ x : ℝ, (1 / 2) * log x - x + m * x^2
  in ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧  (deriv g x₁ = 0) ∧ (deriv g x₂ = 0) →
  g x₁ + g x₂ < -3 / 2 :=
by
  sorry

end monotonic_intervals_of_f_range_of_g_x1_x2_l8_8175


namespace parallel_planes_l8_8201

-- Define lines and planes, and their characteristics
variables {P : Type} [AddGroup P] [Module ℝ P]
variables {l m : Submodule ℝ P} -- lines
variables {α β : Submodule ℝ P} -- planes
noncomputable def pointP : P := sorry
variables (hl_in_alpha : l ≤ α)
variables (hm_in_alpha : m ≤ α)
variables (hl_parallel_beta : l.⊓ β = ⊥)
variables (hm_parallel_beta : m.⊓ β = ⊥)
variables (l_inter_m : l ⊓ m = Submodule.span ℝ {pointP})

-- Goal: show that α and β are parallel
theorem parallel_planes : α.⊓ β = ⊥ :=
sorry

end parallel_planes_l8_8201


namespace max_buildings_in_street_l8_8873

-- Define the maximum number of stories per the city rules
def max_stories : ℕ := 9

-- Define a predicate that describes the condition between buildings
def valid_building_sequence (sequence : list ℕ) : Prop :=
  ∀ i j, 
  (i < j) → 
  (sequence.get(i) = sequence.get(j) → ∃ k, (i < k) ∧ (k < j) ∧ (sequence.get(i) < sequence.get(k)))

-- Define the proof problem that checks if the maximum number of buildings is 511
theorem max_buildings_in_street : ∃ sequence : list ℕ, 
  (∀ n ∈ sequence, n ≤ max_stories) 
  ∧ valid_building_sequence sequence 
  ∧ sequence.length = 511 := 
by 
  sorry

end max_buildings_in_street_l8_8873


namespace last_passenger_probability_l8_8345

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l8_8345


namespace find_slope_BT_l8_8421

-- Defining the ellipse and given points
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 9 = 1

-- Defining the coordinates
variables {x1 y1 x2 y2 : ℝ}

-- Given points A and C
def point_A := ellipse x1 y1
def point_C := ellipse x2 y2

-- Point B and its coordinates
def point_B : ℝ × ℝ := (4, 9 / 5)

-- Focus point
def focus_F : ℝ × ℝ := (4, 0)

-- Distance from a point to the focus
def distance_to_focus (x y : ℝ) : ℝ :=
  sqrt ((x - 4)^2 + y^2)

-- Condition that distances form an arithmetic sequence
def arithmetic_sequence_distances : Prop :=
  2 * distance_to_focus 4 (9 / 5) = distance_to_focus x1 y1 + distance_to_focus x2 y2

-- Perpendicular bisector intersection with x-axis
noncomputable def intersection_T_x : ℝ :=
  (y1^2 - y2^2) / (2 * (x1 - x2)) + (x1 + x2) / 2

-- Slope of the line BT
noncomputable def slope_B_T : ℝ :=
  (9 / 5) / (4 - intersection_T_x)

-- The theorem to be proved
theorem find_slope_BT 
  (hA : ellipse x1 y1) 
  (hC : ellipse x2 y2) 
  (h_dist_sequence : arithmetic_sequence_distances) :
  slope_B_T = 5 / 4 :=
sorry

end find_slope_BT_l8_8421


namespace last_passenger_probability_l8_8333

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l8_8333


namespace vector_subtraction_correct_l8_8109

theorem vector_subtraction_correct : 
  let v1 := (3, -8)
  let s := 3
  let v2 := (2, 6)
  (v1.1 - (s * v2.1), v1.2 - (s * v2.2)) = (-3, -26) :=
by
  let v1 := (3, -8)
  let s := 3
  let v2 := (2, 6)
  show (v1.1 - (s * v2.1), v1.2 - (s * v2.2)) = (-3, -26)
  sorry

end vector_subtraction_correct_l8_8109


namespace sacks_required_in_4_weeks_l8_8868

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l8_8868


namespace largest_mersenne_prime_lt_1000_l8_8758

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop := 
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_lt_1000 : 
  ∃ p, is_mersenne_prime p ∧ p < 1000 ∧ ∀ q, is_mersenne_prime q ∧ q < 1000 → q ≤ p :=
sorry

end largest_mersenne_prime_lt_1000_l8_8758


namespace trees_chopped_l8_8841

def pieces_of_firewood_per_log : Nat := 5
def logs_per_tree : Nat := 4
def total_firewood_chopped : Nat := 500

theorem trees_chopped (pieces_of_firewood_per_log = 5) (logs_per_tree = 4)
    (total_firewood_chopped = 500) :
    total_firewood_chopped / pieces_of_firewood_per_log / logs_per_tree = 25 := by
  sorry

end trees_chopped_l8_8841


namespace range_of_m_num_of_subsets_l8_8284

section Part1
  variable (A B : Set ℝ)
  variable (m : ℝ)
  variable (H_sub : B ⊆ A)

  def A_def := {x | -1 ≤ x ∧ x ≤ 6}
  def B_def := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

  theorem range_of_m (H_sub : B ⊆ A) : 
    B = ∅ → m < -2 ∨ (0 ≤ m ∧ m ≤ (5:ℝ)/2) :=
    sorry
end Part1

section Part2
  variable (A : Set ℕ)
  def A_def := {0, 1, 2, 3, 4, 5, 6}

  theorem num_of_subsets (H : ∀ x, x ∈ A → ∃! n, n = 7) : 
    ∃ n, n = 2^7 :=
    sorry
end Part2

end range_of_m_num_of_subsets_l8_8284


namespace triangle_perimeter_l8_8966

-- We're dealing with sides of a triangle a, b, and even c
variables {a b c : ℕ}

-- Conditions given in the problem
def triangle_condition (a b : ℕ) : Prop :=
  (a - 2)^2 + |b - 4| = 0

def even_and_in_range (c : ℕ) : Prop :=
  Even c ∧ 2 < c ∧ c < 6

-- The perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove
theorem triangle_perimeter
  (h1 : triangle_condition a b)
  (h2 : even_and_in_range c) :
  a = 2 ∧ b = 4 ∧ perimeter a b c = 10 := 
by {
  sorry
}

end triangle_perimeter_l8_8966


namespace largest_multiple_of_8_less_than_100_l8_8759

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8759


namespace eval_fraction_sum_l8_8493

theorem eval_fraction_sum : 64^(-1/3) + 81^(-1/2) = 13/36 := by
  sorry

end eval_fraction_sum_l8_8493


namespace sequence_an_formula_l8_8245

noncomputable def a : ℕ+ → ℚ
| ⟨1, _⟩   := 1
| ⟨n+1, p⟩ := a ⟨n, nat.succ_pos n⟩ / (1 + a ⟨n, nat.succ_pos n⟩)

theorem sequence_an_formula (n : ℕ+) : a n = 1 / n := 
sorry

end sequence_an_formula_l8_8245


namespace polynomial_simplification_l8_8403

noncomputable def a := 3 * x^3 + 4 * x^2 - 5 * x + 8
noncomputable def b := 2 * x^3 - 7 * x^2 + 10
noncomputable def c := 7 * x - 15
noncomputable def d := 2 * x + 1
noncomputable def e := x - 2

theorem polynomial_simplification (x : ℝ) :
  (a * e - e * b + c * e * d) = x^4 + 23 * x^3 - 78 * x^2 + 39 * x + 34 :=
sorry

end polynomial_simplification_l8_8403


namespace minimum_M_l8_8566

def f (x : ℝ) : ℝ := x ^ 2 - (1/2) * x + (1/4)

def b_seq : ℕ → ℝ
| 0       := 1
| (n+1) := 2 * f (b_seq n)

theorem minimum_M (M : ℝ) : (∀ n : ℕ, (∑ i in Finset.range n, 1 / b_seq i) < M) → M = 2 :=
begin
  sorry
end

end minimum_M_l8_8566


namespace valid_three_digit_numbers_count_l8_8995

theorem valid_three_digit_numbers_count : 
  let total_numbers := 900 
  let form_AAA := 9
  let form_ABA := 81
  let excluded_numbers := form_AAA + form_ABA
  let remaining_numbers := total_numbers - excluded_numbers
  remaining_numbers = 810 :=
by
  simp [total_numbers, form_AAA, form_ABA, excluded_numbers, remaining_numbers]
  done

end valid_three_digit_numbers_count_l8_8995


namespace polynomial_sum_l8_8655

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l8_8655


namespace sacks_required_in_4_weeks_l8_8867

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l8_8867


namespace max_value_of_f_cos_value_given_f_theta_l8_8205

open Real

/-- Definition of vector a -/
def a (x : ℝ) : ℝ × ℝ := (1 + sin (2 * x), sin x - cos x)

/-- Definition of vector b -/
def b (x : ℝ) : ℝ × ℝ := (1, sin x + cos x)

/-- Definition of dot product -/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Function f(x) -/
def f (x : ℝ) : ℝ := dot (a x) (b x)

theorem max_value_of_f (k : ℤ) (x : ℝ) :
  (∃ x, f x = sqrt 2 + 1) ∧ (∃ (k : ℤ), x = k * π + 3 / 8 * π) := 
sorry

theorem cos_value_given_f_theta (θ : ℝ) (hθ : f θ = 8 / 5) :
  cos (2 * (π / 4 - 2 * θ)) = 16 / 25 := 
sorry

end max_value_of_f_cos_value_given_f_theta_l8_8205


namespace count_perfect_squares_diff_two_consecutive_squares_l8_8907

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l8_8907


namespace part1_part2_l8_8429

-- Define the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := x^3 - x^2 + b * x + c

-- Part 1: Prove that if f(x) is an increasing function on (-∞, ∞), then b ≥ 1/12
theorem part1 (b c : ℝ) :
  (∀ x : ℝ, deriv (λ x, f x b c) x ≥ 0) → b ≥ 1/12 := sorry

-- Part 2: Prove that if f(x) takes an extreme value at x = 1 and f(x) < c^2 for all x ∈ [-1, 2],
-- then c ∈ (-∞, -1) ∪ (2, ∞)
theorem part2 (b c : ℝ) :
  (deriv (λ x, f x b c) 1 = 0) →
  (∀ x : ℝ, x ∈ set.Icc (-1) 2 → f x b c < c^2) →
  (c < -1 ∨ c > 2) := sorry

end part1_part2_l8_8429


namespace min_moves_to_equalize_boxes_l8_8291

/--
Olya has a black box containing 5 apples and 7 pears, and a white box containing 12 pears.
Olya can take a fruit from any box without looking and either eat it or move it to the other box.
The goal is to make the contents of the black box and white box the same.
Prove that the minimum number of moves required to achieve this is 18.
-/
theorem min_moves_to_equalize_boxes : 
  ∃ (moves : ℕ), moves = 18 ∧ 
    (∀ (initial_state : (fin 2 → ℕ) × (fin 2 → ℕ)),
      (initial_state.1 0 = 5 ∧ initial_state.1 1 = 7) →
      (initial_state.2 0 = 0 ∧ initial_state.2 1 = 12) →
      ∃ (final_state : (fin 2 → ℕ) × (fin 2 → ℕ)), 
        (final_state.1 = final_state.2) ∧
        (∃ (steps : list ((fin 2) × (fin 2))), 
          steps.length = moves ∧
          ∀ (s: (fin 2 → ℕ) × (fin 2 → ℕ)),
            s = initial_state ∨ 
            (∀ (step : (fin 2) × (fin 2)), 
              step ∈ steps → 
              s.1 step.1 = s.1 step.1 - 1 ∧ s.1 step.2 = s.1 step.2 + 1 ∨ 
              s.2 step.1 = s.2 step.1 - 1 ∧ s.2 step.2 = s.2 step.2 + 1) ∧
            (∀ (a b : fin 2), s.1 a = final_state.1 a ∧ s.2 b = final_state.2 b))) := 
begin
  sorry
end

end min_moves_to_equalize_boxes_l8_8291


namespace average_of_combined_numbers_l8_8218

theorem average_of_combined_numbers (M N : ℕ) (X Y : ℝ) (hM_avg : ∑ i in finset.range M, i / M = X) (hN_avg : ∑ i in finset.range N, i / N = Y) :
  (∑ i in finset.range (M + N), i) / (M + N) = (M * X + N * Y) / (M + N) :=
by {
  sorry
}

end average_of_combined_numbers_l8_8218


namespace three_digit_numbers_count_l8_8996

theorem three_digit_numbers_count : 
  ∃ n : ℕ, n = 55 ∧ 
  (∀ a b c : ℕ,
    (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (b = a + c) → 
    (1 ≤ (10 * a + b * 10 + c) ∧ (10 * a + b * 10 + c) ≤ 999)
  ): sorry

end three_digit_numbers_count_l8_8996


namespace puzzles_pieces_count_l8_8295

theorem puzzles_pieces_count :
  let pieces_per_hour := 100
  let hours_per_day := 7
  let days := 7
  let total_pieces_can_put_together := pieces_per_hour * hours_per_day * days
  let pieces_per_puzzle1 := 300
  let number_of_puzzles1 := 8
  let total_pieces_puzzles1 := pieces_per_puzzle1 * number_of_puzzles1
  let remaining_pieces := total_pieces_can_put_together - total_pieces_puzzles1
  let number_of_puzzles2 := 5
  remaining_pieces / number_of_puzzles2 = 500
:= by
  sorry

end puzzles_pieces_count_l8_8295


namespace find_pentahedron_l8_8037

/-- Define the property of being a tetrahedron -/
def is_tetrahedron (x : Type) : Prop := sorry

/-- Define the property of being a pentahedron -/
def is_pentahedron (x : Type) : Prop := sorry

/-- Define the property of being a hexahedron -/
def is_hexahedron (x : Type) : Prop := sorry

/-- Define specific polyhedra as types -/
inductive Polyhedron
| TriangularPyramid
| TriangularPrism
| QuadrangularPrism
| PentagonalPyramid

open Polyhedron

/-- Define the conditions -/
axiom cond1 : is_tetrahedron TriangularPyramid
axiom cond2 : is_pentahedron TriangularPrism
axiom cond3 : is_hexahedron QuadrangularPrism
axiom cond4 : is_hexahedron PentagonalPyramid

/-- Prove that the TriangularPrism is the pentahedron -/
theorem find_pentahedron : is_pentahedron TriangularPrism := cond2

end find_pentahedron_l8_8037


namespace obtuse_angle_half_in_first_quadrant_l8_8586

-- Define α to be an obtuse angle
variable {α : ℝ}

-- The main theorem we want to prove
theorem obtuse_angle_half_in_first_quadrant (h_obtuse : (π / 2) < α ∧ α < π) :
  0 < α / 2 ∧ α / 2 < π / 2 :=
  sorry

end obtuse_angle_half_in_first_quadrant_l8_8586


namespace cot_neg_45_deg_l8_8143

-- Define the necessary trigonometric identities.
def cot (θ : ℝ) := 1 / (Real.tan θ)
def neg_angle_identity : ∀ θ, Real.tan (-θ) = -Real.tan θ := sorry
def tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove: 
theorem cot_neg_45_deg : cot (-Real.pi / 4) = -1 := by
  -- Given identities
  have h1: cot (-Real.pi / 4) = 1 / Real.tan (-Real.pi / 4) := rfl
  have h2: Real.tan (-Real.pi / 4) = -Real.tan (Real.pi / 4) := neg_angle_identity (Real.pi / 4)
  have h3: Real.tan (Real.pi / 4) = 1 := tan_45
  -- Proof steps would be placed here
  sorry

end cot_neg_45_deg_l8_8143


namespace find_tangent_line_l8_8717

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end find_tangent_line_l8_8717


namespace factorize_polynomial_l8_8501

def p (a b : ℝ) : ℝ := a^2 - b^2 + 2 * a + 1

theorem factorize_polynomial (a b : ℝ) : 
  p a b = (a + 1 + b) * (a + 1 - b) :=
by
  sorry

end factorize_polynomial_l8_8501


namespace sum_second_third_smallest_l8_8370

theorem sum_second_third_smallest (a b c d : ℕ) :
  ({a, b, c, d} = {10, 11, 12, 13}) →
  (∀ x y z w : ℕ, ({x, y, z, w} = {a, b, c, d}) →
    (x ≤ y ∧ y ≤ z ∧ z ≤ w) →
    nth_element_of_set ({x, y, z, w}, 1) + nth_element_of_set ({x, y, z, w}, 2) = 23) :=
sorry

end sum_second_third_smallest_l8_8370


namespace find_P_Q_l8_8142

noncomputable def P := 11 / 3
noncomputable def Q := -2 / 3

theorem find_P_Q :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
    (3 * x + 12) / (x ^ 2 - 5 * x - 14) = P / (x - 7) + Q / (x + 2) :=
by
  intros x hx1 hx2
  dsimp [P, Q]  -- Unfold the definitions of P and Q
  -- The actual proof would go here, but we are skipping it
  sorry

end find_P_Q_l8_8142


namespace polynomial_sum_l8_8664

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l8_8664


namespace range_of_f_l8_8722

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (x^2 + 4)

theorem range_of_f :
  set.Ico 0 2 = {y : ℝ | ∃ x : ℝ, f(x) = y} :=
by
  sorry

end range_of_f_l8_8722


namespace percentage_problem_l8_8497

theorem percentage_problem (X : ℝ) (h : 0.28 * X + 0.45 * 250 = 224.5) : X = 400 :=
sorry

end percentage_problem_l8_8497


namespace bobby_pizzas_l8_8287

theorem bobby_pizzas (B : ℕ) (h_slices : (1 / 4 : ℝ) * B = 3) (h_slices_per_pizza : 6 > 0) :
  B / 6 = 2 := by
  sorry

end bobby_pizzas_l8_8287


namespace earnings_difference_l8_8746

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l8_8746


namespace trisomy_21_due_to_sperm_with_two_chromosomes_21_l8_8723

-- Definitions based on problem conditions
def genotype (individual : String) : String :=
  match individual with
  | "child" => "++-"
  | "father" => "+-"
  | "mother" => "--"
  | _ => ""

-- The proposition we need to prove based on the problem
theorem trisomy_21_due_to_sperm_with_two_chromosomes_21 :
  genotype "child" = "++-" →
  genotype "father" = "+-" →
  genotype "mother" = "--" →
  "C: The child was produced due to the sperm having 2 chromosome 21s" :=
by
  intros h_child h_father h_mother
  -- Replace with the actual proof
  sorry

end trisomy_21_due_to_sperm_with_two_chromosomes_21_l8_8723


namespace tetrahedron_division_l8_8482

def divide_tetrahedron := by
  /- Given conditions -/
  let T := regular_tetrahedron 1 -- A regular tetrahedron of side length 1
  let T_divided_edges := divide_edges T 3 -- Divide each edge into three equal parts
  let planes := constructed_planes T_divided_edges
  
  /- Correct answer  -/
  let num_parts := 15
  
  /- Statement to prove -/
  theorem tetrahedron_division : (number_of_parts T planes = num_parts) :=
  sorry

end tetrahedron_division_l8_8482


namespace boxes_contain_fruits_l8_8012

-- Define the weights of the boxes
def box_weights : List ℕ := [15, 16, 18, 19, 20, 31]

-- Define the weight requirement for apples and pears
def weight_rel (apple_weight pear_weight : ℕ) : Prop := apple_weight = pear_weight / 2

-- Define the statement with the constraints, given conditions and assignments.
theorem boxes_contain_fruits (h1 : box_weights = [15, 16, 18, 19, 20, 31])
                             (h2 : ∃ apple_weight pear_weight, 
                                   weight_rel apple_weight pear_weight ∧ 
                                   pear_weight ∈ box_weights ∧ apple_weight ∈ box_weights)
                             (h3 : ∃ orange_weight, orange_weight ∈ box_weights ∧ 
                                   ∀ w, w ∈ box_weights → w ≠ orange_weight)
                             : (15 = 2 ∧ 19 = 3 ∧ 20 = 1 ∧ 31 = 3) := 
                             sorry

end boxes_contain_fruits_l8_8012


namespace graph_intersects_self_28_times_l8_8896

noncomputable def x (t : ℝ) : ℝ := cos t + t / 3
noncomputable def y (t : ℝ) : ℝ := sin t

theorem graph_intersects_self_28_times :
  ∃ t_values : (ℕ → ℝ), (∀ n : ℕ, x (t_values n) = 60) ∧ (x (t_values 177) = 0) ∧ (t_values.length = 28) := 
sorry

end graph_intersects_self_28_times_l8_8896


namespace max_value_of_quadratic_l8_8940

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) : (6 - x) * x ≤ 9 := 
by
  sorry

end max_value_of_quadratic_l8_8940


namespace trees_chopped_l8_8839

def pieces_of_firewood_per_log : Nat := 5
def logs_per_tree : Nat := 4
def total_firewood_chopped : Nat := 500

theorem trees_chopped (pieces_of_firewood_per_log = 5) (logs_per_tree = 4)
    (total_firewood_chopped = 500) :
    total_firewood_chopped / pieces_of_firewood_per_log / logs_per_tree = 25 := by
  sorry

end trees_chopped_l8_8839


namespace hexagon_coloring_l8_8138

-- Definitions
def vertices := ["A", "B", "C", "D", "E", "F"]

def different_color (coloring : vertices → ℕ) (c1 c2 : vertices) :=
  coloring c1 ≠ coloring c2

def valid_coloring (coloring : vertices → ℕ) :=
  ∀ (A B C D E F: vertices),
    (different_color coloring A B) ∧
    (different_color coloring B C) ∧
    (different_color coloring C D) ∧
    (different_color coloring D E) ∧
    (different_color coloring E F) ∧
    (different_color coloring F A) ∧
    (different_color coloring A D) ∧
    (different_color coloring B E) ∧
    (different_color coloring C F)

-- Main statement
theorem hexagon_coloring : 
  ∃ (coloring : vertices → ℕ), valid_coloring coloring ∧ 
  (card {coloring | valid_coloring coloring} = 45360) := 
sorry

end hexagon_coloring_l8_8138


namespace number_of_routes_jack_to_jill_l8_8253

def num_routes_avoiding (start goal avoid : ℕ × ℕ) : ℕ := sorry

theorem number_of_routes_jack_to_jill : 
  num_routes_avoiding (0,0) (3,2) (1,1) = 4 :=
sorry

end number_of_routes_jack_to_jill_l8_8253


namespace line_equation_correct_find_m_value_l8_8963

-- Line l is through point M(1,1) and parallel to the line 2x + 4y + 9 = 0
class Line :=
  (point : ℝ × ℝ)
  (parallel_to : ℝ × ℝ × ℝ)

-- Circle with equation x^2 + y^2 + x - 6y + m = 0
def Circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 + x - 6y + m = 0

-- Perpendicularity of vectors OP and OQ
def Perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.1 * Q.1 + P.2 * Q.2 = 0)

-- Given Line l with specified properties
def given_line : Line :=
  { point := (1,1),
    parallel_to := (2, 4, 9) }

-- The equation of line l
def equation_of_line (l : Line) : String := sorry -- Placeholder for the equation derivation

-- Proving the line equation
theorem line_equation_correct :
  equation_of_line given_line = "x + 2y - 3 = 0" :=
sorry

-- Proving the value of m given the conditions
theorem find_m_value :
  ∀ {P Q : ℝ × ℝ}
    (h1 : Circle 3)
    (h2 : Perpendicular P Q),
  3 = 3 :=
sorry

end line_equation_correct_find_m_value_l8_8963


namespace reflection_periodicity_l8_8679

noncomputable def theta := Real.arctan (17 / 75)
noncomputable def alpha := Real.pi / 60
noncomputable def beta := Real.pi / 45

-- Define the reflection operation R
def R (theta : ℝ) (alpha : ℝ) (beta : ℝ) : ℝ :=
  let reflected_once := 2 * alpha - theta
  in 2 * beta - reflected_once

-- Prove that applying R 36 times results in the original angle
theorem reflection_periodicity (theta alpha beta : ℝ) :
  R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R (R theta alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta) alpha beta = theta :=
sorry

end reflection_periodicity_l8_8679


namespace goals_even_more_likely_l8_8440

theorem goals_even_more_likely (p_1 : ℝ) (q_1 : ℝ) (h1 : p_1 + q_1 = 1) :
  let p := p_1^2 + q_1^2 
  let q := 2 * p_1 * q_1
  p ≥ q := by
    sorry

end goals_even_more_likely_l8_8440


namespace matches_in_each_matchbook_l8_8255

-- Conditions given in the problem
def one_stamp_worth_matches (s : ℕ) : Prop := s = 12
def tonya_initial_stamps (t : ℕ) : Prop := t = 13
def tonya_final_stamps (t : ℕ) : Prop := t = 3
def jimmy_initial_matchbooks (j : ℕ) : Prop := j = 5

-- Goal: prove M = 24
theorem matches_in_each_matchbook (M : ℕ) (s t_initial t_final j : ℕ) 
  (h1 : one_stamp_worth_matches s) 
  (h2 : tonya_initial_stamps t_initial) 
  (h3 : tonya_final_stamps t_final) 
  (h4 : jimmy_initial_matchbooks j) : M = 24 := by
  sorry

end matches_in_each_matchbook_l8_8255


namespace midpoint_of_arc_l8_8105

-- Define the centers of the circles and the intersecting points
variables {O1 O2 A B X Y X' K : Type}
variables [MetricSpace O1] [MetricSpace O2]
variables [Inhabited O1] [Inhabited O2]

-- Assume the existence of circles and points with given properties
variable (w1 : MetricSpace.Sphere O1)
variable (w2 : MetricSpace.Sphere O2)

-- Assume necessary conditions 
variables (h1 : A ∈ w1) (h2 : B ∈ w1) (h3 : A ∈ w2) (h4 : B ∈ w2)
variables (h5 : X ∈ w2) (h6 : Y ∈ w1) 
variables (h7 : ∠X B Y = 90) 
variables (h8 : intersects (O1, X) w2 X') 
variables (h9 : intersects (X', Y) w2 K) 

-- Prove X is the midpoint of the arc AK on w2
theorem midpoint_of_arc (w1 : MetricSpace.Sphere O1) (w2 : MetricSpace.Sphere O2) 
    (A B X Y X' K: Type) [MetricSpace O1] [MetricSpace O2] [Inhabited O1] [Inhabited O2]
    (h1 : A ∈ w1) (h2 : B ∈ w1) (h3 : A ∈ w2) (h4 : B ∈ w2)
    (h5 : X ∈ w2) (h6 : Y ∈ w1)
    (h7 : ∠X B Y = 90)
    (h8 : intersects (O1, X) w2 X') 
    (h9 : intersects (X', Y) w2 K) : 
    is_midpoint_of_arc X A K w2 :=
sorry

end midpoint_of_arc_l8_8105


namespace count_valid_n_l8_8167

theorem count_valid_n : ∀ (n : ℕ),
  (1 ≤ n ∧ n ≤ 2008) →
  (∃ (s : Finset (Finset ℕ)), s.card = n ∧ ∀ (t ∈ s), t.card = 4 ∧ (∃ a b c d ∈ t, a + b + c + d = 4 * a)) →
  ∃ (k : ℕ), k = 1004 :=
by
  sorry

end count_valid_n_l8_8167


namespace log_inequality_l8_8998

variable {x : ℝ}

-- Given condition
def ineq (x : ℝ) :=
  1 < x ∧ x < 10

-- Prove the inequality
theorem log_inequality (h : ineq x) :
  log(log x) < (log x)^2 ∧ (log x)^2 < log (x^2) :=
by
  sorry

end log_inequality_l8_8998


namespace last_passenger_sits_in_assigned_seat_l8_8339

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l8_8339


namespace clothing_store_profit_l8_8064

theorem clothing_store_profit 
  (cost_price selling_price : ℕ)
  (initial_items_per_day items_increment items_reduction : ℕ)
  (initial_profit_per_day : ℕ) :
  -- Conditions
  cost_price = 50 ∧
  selling_price = 90 ∧
  initial_items_per_day = 20 ∧
  items_increment = 2 ∧
  items_reduction = 1 ∧
  initial_profit_per_day = 1200 →
  -- Question
  exists x, 
  (selling_price - x - cost_price) * (initial_items_per_day + items_increment * x) = initial_profit_per_day ∧
  x = 20 := 
sorry

end clothing_store_profit_l8_8064


namespace portion_replaced_25_units_l8_8066

def initial_solution (V : ℝ) := 0.10 * V
def second_solution (W : ℝ) := 0.34 * W
def resulting_solution (V W : ℝ) := V + W
def sugar_content (S V : ℝ) := S / V

theorem portion_replaced_25_units :
  ∀ (V W : ℝ), 
    sugar_content (initial_solution (100 - V) + second_solution V) 100 = 0.16 → 
    V = 25 :=
by
  intros V W h
  sorry

end portion_replaced_25_units_l8_8066


namespace complement_M_in_U_l8_8988

def M (x : ℝ) : Prop := 0 < x ∧ x < 2

def complement_M (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

theorem complement_M_in_U (x : ℝ) : ¬ M x ↔ complement_M x :=
by sorry

end complement_M_in_U_l8_8988


namespace largest_multiple_of_8_less_than_100_l8_8772

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8772


namespace smallest_b_such_that_N_is_fourth_power_l8_8044

theorem smallest_b_such_that_N_is_fourth_power :
  let N (b : ℕ) : ℕ := 7 * b^2 + 7 * b + 7
  in ∃ (b : ℕ), N b = m^4 for some m ∈ ℕ ∧ ∀ b' < b, ¬ ∃ m' ∈ ℕ, N b' = m'^4 :=
begin
  sorry
end

end smallest_b_such_that_N_is_fourth_power_l8_8044


namespace jack_needs_more_money_l8_8632

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l8_8632


namespace gravel_cost_correct_l8_8406

-- Definitions from the conditions
def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 60
def road_width : ℕ := 15
def gravel_cost_per_sq_m : ℕ := 3

-- Calculate areas of the roads
def area_road_length : ℕ := lawn_length * road_width
def area_road_breadth : ℕ := (lawn_breadth - road_width) * road_width

-- Total area to be graveled
def total_area : ℕ := area_road_length + area_road_breadth

-- Total cost
def total_cost : ℕ := total_area * gravel_cost_per_sq_m

-- Prove the total cost is 5625 Rs
theorem gravel_cost_correct : total_cost = 5625 := by
  sorry

end gravel_cost_correct_l8_8406


namespace conical_funnel_area_l8_8014

def slant_height : ℝ := 6
def base_circumference : ℝ := 6 * real.pi

theorem conical_funnel_area : 
  ∃ (A : ℝ), A = 18 * real.pi := 
sorry

end conical_funnel_area_l8_8014


namespace count_perfect_squares_diff_two_consecutive_squares_l8_8908

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l8_8908


namespace maximize_parabola_area_l8_8535

variable {a b : ℝ}

/--
The parabola y = ax^2 + bx is tangent to the line x + y = 4 within the first quadrant. 
Prove that the values of a and b that maximize the area S enclosed by this parabola and 
the x-axis are a = -1 and b = 3, and that the maximum value of S is 9/2.
-/
theorem maximize_parabola_area (hab_tangent : ∃ x y, y = a * x^2 + b * x ∧ y = 4 - x ∧ x > 0 ∧ y > 0) 
  (area_eqn : S = 1/6 * (b^3 / a^2)) : 
  a = -1 ∧ b = 3 ∧ S = 9/2 := 
sorry

end maximize_parabola_area_l8_8535


namespace probability_select_changjin_lake_l8_8914

theorem probability_select_changjin_lake :
  let movies := ["Changjin Lake", "Hello, Beijing", "Young Lion", "Prosecution Storm"]
  in ∃ (P : String → ℚ), (P "Changjin Lake" = 1 / 4) ∧
                         (∀ (m : String), m ∈ movies → P m = 1 / (movies.length : ℚ)) :=
by
  sorry

end probability_select_changjin_lake_l8_8914


namespace tennis_tournament_points_l8_8238

theorem tennis_tournament_points (n : ℕ) (hn : 4 < n) :
  let total_participants := 2^n + 4 in 
  let participants_with_3_points := (nat.choose n 3) + 1 in 
  (C n 3 + 1) = participants_with_3_points :=
by
  sorry

end tennis_tournament_points_l8_8238


namespace avg_speed_in_mph_l8_8071

/-- 
Given conditions:
1. The man travels 10,000 feet due north.
2. He travels 6,000 feet due east in 1/4 less time than he took heading north, traveling at 3 miles per minute.
3. He returns to his starting point by traveling south at 1 mile per minute.
4. He travels back west at the same speed as he went east.
We aim to prove that the average speed for the entire trip is 22.71 miles per hour.
-/
theorem avg_speed_in_mph :
  let distance_north_feet := 10000
  let distance_east_feet := 6000
  let speed_east_miles_per_minute := 3
  let speed_south_miles_per_minute := 1
  let feet_per_mile := 5280
  let distance_north_mil := (distance_north_feet / feet_per_mile : ℝ)
  let distance_east_mil := (distance_east_feet / feet_per_mile : ℝ)
  let time_north_min := distance_north_mil / (1 / 3)
  let time_east_min := time_north_min * 0.75
  let time_south_min := distance_north_mil / speed_south_miles_per_minute
  let time_west_min := time_east_min
  let total_time_hr := (time_north_min + time_east_min + time_south_min + time_west_min) / 60
  let total_distance_miles := 2 * (distance_north_mil + distance_east_mil)
  let avg_speed_mph := total_distance_miles / total_time_hr
  avg_speed_mph = 22.71 := by
sorry

end avg_speed_in_mph_l8_8071


namespace find_shaded_area_l8_8616

noncomputable def shaded_area (r : ℝ) : ℝ :=
  let quarter_circle_area := (π * r^2) / 4
  let right_triangle_area := (r * r) / 2
  let shaded_segment_area := quarter_circle_area - right_triangle_area
  8 * shaded_segment_area

theorem find_shaded_area : shaded_area 6 = 72 * π - 144 :=
  by sorry

end find_shaded_area_l8_8616


namespace master_bedroom_suite_size_l8_8069

-- Define the conditions based on the problem statement
def living_room_dining_kitchen_area : ℕ := 1000
def total_house_area : ℕ := 2300

def guest_room_to_master_ratio : ℕ := 4

-- Define the problem in terms of Lean 4 statement
theorem master_bedroom_suite_size :
  ∃ (M : ℕ), let G := M / guest_room_to_master_ratio in
  M + G = total_house_area - living_room_dining_kitchen_area ∧ M = 1040 :=
sorry

end master_bedroom_suite_size_l8_8069


namespace find_possible_positive_integers_l8_8504

theorem find_possible_positive_integers (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pairwise_coprime : Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime c a) :
  ∃ n : ℕ, n = 6 ∨ n = 7 ∨ n = 8 ∧ n = ((b / a) + (c / a) + (c / b) + (a / b) + (a / c) + (b / c)) :=
sorry

end find_possible_positive_integers_l8_8504


namespace teams_not_played_each_other_l8_8606

theorem teams_not_played_each_other :
  ∃ (A B C : Fin 18),
  (∀ (i : Fin 18), (A ≠ i → B ≠ i → C ≠ i 
  → ¬((has_played A i) ∧ (has_played B i) ∧ (has_played C i)))) :=
begin
  sorry
end

-- Here, has_played is a predicate that determines whether two teams have played against each other.
-- Fin 18 represents the set of 18 teams, indexed from 0 to 17.

end teams_not_played_each_other_l8_8606


namespace sequence_problem_l8_8986

open Nat

def sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = n^2 → S n = (finset.range (n + 1)).sum a

def correct_a3_and_general_term (a : ℕ → ℕ) : Prop :=
  a 3 = 5 ∧ (∀ n : ℕ, a n = 2 * n - 1)

theorem sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = n^2) →
  sequence_sum S a →
  correct_a3_and_general_term a :=
by {
  intros hS hSeqSum,
  -- Proof would go here
  sorry
}

end sequence_problem_l8_8986


namespace container_capacity_is_80_l8_8405

noncomputable theory

def container_capacity (C : ℝ) : Prop :=
  let initial_water := 0.5 * C
  let final_water := initial_water + 20
  final_water = 0.75 * C

theorem container_capacity_is_80 :
  ∃ C, container_capacity C ∧ C = 80 := 
by
  sorry

end container_capacity_is_80_l8_8405


namespace eccentricity_range_l8_8476

noncomputable def ellipseEccentricityRange (a b c e : ℝ) (P : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  ((P.1 = a^2 / c) ∧ 
   (∃ m : ℝ, P = (a^2 / c, m) ∧
    let F1 := (-c, 0) in
    let F2 := (c, 0) in
    let K := ((a^2 - c^2) / (2 * c), m / 2) in
    ( (P.2 - 0) / (a^2 / c + c) * 
      ( (m / 2) - 0 ) /
      ((a^2 - c^2) / (2 * c) - c) = -1) ∧
    (3 * e^4 + 2 * e^2 - 1 ≥ 0)) ∧
  (a^2 = b^2 + c^2) ∧ 
  (e = c / a))

theorem eccentricity_range (a b c e : ℝ) (P : ℝ × ℝ) : 
  ellipseEccentricityRange a b c e P → 
  e ∈ Set.Icc (real.sqrt 3 / 3) 1 := 
by 
  sorry

end eccentricity_range_l8_8476


namespace card_prob_eq_one_div_seventy_five_l8_8312

/-- Seventy-five cards, numbered 1 to 75, are placed in a box.
One card is randomly selected. What is the probability 
that the number on the card is prime and is a multiple of 5? -/
def card_prob_prime_and_multiple_of_five : ℚ :=
  let cards := {1..75}
  let prime_five := 5
  let favorable_events := {n | n = prime_five} -- numbers that are both prime and multiples of 5
  let total_events := cards.card -- total number of cards
  let favorable_number := favorable_events.card -- count of favorable events
  (favorable_number : ℚ) / (total_events : ℚ)

theorem card_prob_eq_one_div_seventy_five : card_prob_prime_and_multiple_of_five = 1 / 75 := by
  -- Details of the proof will go here
  sorry

end card_prob_eq_one_div_seventy_five_l8_8312


namespace smallest_k_DIVISIBLE_by_3_67_l8_8933

theorem smallest_k_DIVISIBLE_by_3_67 :
  ∃ k : ℕ, (∀ n : ℕ, (2016^k % 3^67 = 0 ∧ (2016^n % 3^67 = 0 → k ≤ n)) ∧ k = 34) := by
  sorry

end smallest_k_DIVISIBLE_by_3_67_l8_8933


namespace sum_of_digits_is_10_l8_8727

theorem sum_of_digits_is_10 (x y : ℕ) (a : ℕ) (hxy : x * y = 32) (ha : a = (10 ^ (x * y) - 64).digits.sum) : a = 10 :=
sorry

end sum_of_digits_is_10_l8_8727


namespace ball_hits_ground_at_5_over_2_l8_8716

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 + 40 * t + 60

theorem ball_hits_ground_at_5_over_2 :
  ∃ t : ℝ, t = 5 / 2 ∧ ball_height t = 0 :=
sorry

end ball_hits_ground_at_5_over_2_l8_8716


namespace equation_is_point_l8_8477

-- Definition of the condition in the problem
def equation (x y : ℝ) := x^2 + 36*y^2 - 12*x - 72*y + 36 = 0

-- The theorem stating the equivalence to the point (6, 1)
theorem equation_is_point :
  ∀ (x y : ℝ), equation x y → (x = 6 ∧ y = 1) :=
by
  intros x y h
  -- The proof steps would go here
  sorry

end equation_is_point_l8_8477


namespace area_of_rhombus_given_diagonals_l8_8815

def rhombus_diagonals_to_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem area_of_rhombus_given_diagonals :
  rhombus_diagonals_to_area 12 20 = 120 :=
  by sorry

end area_of_rhombus_given_diagonals_l8_8815


namespace value_of_n_l8_8168

theorem value_of_n : ∃ n : ℕ, 3 * 4 * 5 * n = 6! ∧ n = 12 := 
  sorry

end value_of_n_l8_8168


namespace f_2011_equals_1_l8_8719

-- Define odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define function with period property
def has_period_3 (f : ℤ → ℤ) : Prop :=
  ∀ x, f (x + 3) = f (x)

-- Define main problem statement
theorem f_2011_equals_1 
  (f : ℤ → ℤ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : f (-1) = -1) 
  : f 2011 = 1 :=
sorry

end f_2011_equals_1_l8_8719


namespace find_prices_l8_8738

def price_system_of_equations (x y : ℕ) : Prop :=
  3 * x + 2 * y = 474 ∧ x - y = 8

theorem find_prices (x y : ℕ) :
  price_system_of_equations x y :=
by
  sorry

end find_prices_l8_8738


namespace calculation_of_exponents_l8_8881

theorem calculation_of_exponents :
  (81:ℝ)^(0.25) * (81:ℝ)^(0.20) = 6.86 :=
  sorry

end calculation_of_exponents_l8_8881


namespace bn_is_not_geometric_l8_8209

def is_geometric (s : ℕ → ℝ) : Prop :=
  ∃ r a, ∀ n, s n = a * r^n

def sequence_an : ℕ → ℝ
| 0 => 1
| n+1 => (-1) * sequence_an n

def sequence_bn (a_seq : ℕ → ℝ) : ℕ → ℝ :=
  λ n, a_seq (2 * n - 1) + a_seq (2 * n)

theorem bn_is_not_geometric : ¬ is_geometric (sequence_bn sequence_an)
                                                              :=
begin
  sorry
end

end bn_is_not_geometric_l8_8209


namespace derivative_sufficiency_l8_8195

theorem derivative_sufficiency (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x ∈ Ioo a b, f' x > 0) :
  (∀ x y ∈ Ioo a b, x < y → f x < f y) ∧ ¬(∀ x ∈ Ioo a b, f' x > 0) → false := by sorry

end derivative_sufficiency_l8_8195


namespace distribute_books_l8_8058

theorem distribute_books (students books : ℕ) (h_books : books = 3) (h_students : students = 4) :
  (students ^ books) = 4 ^ 3 := by
  rw [h_books, h_students]
  rfl

end distribute_books_l8_8058


namespace value_of_a_2018_l8_8985

def sequence_a : ℕ → ℝ
| 0     := 2
| (n+1) := 1 - 1 / sequence_a n

theorem value_of_a_2018 : sequence_a 2017 = 1 / 2 := -- Using 2017 because the sequence indices start from 0 in Lean
by
  sorry

end value_of_a_2018_l8_8985


namespace simple_interest_sum_l8_8087

theorem simple_interest_sum (SI R T : ℝ) (hSI : SI = 4016.25) (hR : R = 0.01) (hT : T = 3) :
  SI / (R * T) = 133875 := by
  sorry

end simple_interest_sum_l8_8087


namespace combinatorial_count_l8_8939

-- Define the function f_r(n, k) that we need to prove the correct answer for
def f_r (n k r : ℕ) : ℕ :=
  Nat.choose (n - k * r + r) k

-- Lean statement representing the proof problem
theorem combinatorial_count (n k r : ℕ) (h : n + r ≥ k * r + k) : 
  f_r n k r = Nat.choose (n - k * r + r) k := 
sorry

end combinatorial_count_l8_8939


namespace eval_fraction_sum_l8_8495

theorem eval_fraction_sum : 64^(-1/3) + 81^(-1/2) = 13/36 := by
  sorry

end eval_fraction_sum_l8_8495


namespace new_supervisor_salary_l8_8814

theorem new_supervisor_salary
  (n_workers : ℕ)
  (old_supervisor_salary : ℕ)
 -- Given data: average salary with old supervisor is 430, old supervisor's salary is 870, new average is 390
  (avg_salary_with_old_supervisor : ℕ)
  (new_avg_salary : ℕ)
  -- Values to plug in
  (n_workers = 8)
  (old_supervisor_salary = 870)
  (avg_salary_with_old_supervisor = 430)
  (new_avg_salary = 390) :
  -- Let's calculate the salary of new supervisor
  let W := avg_salary_with_old_supervisor * (n_workers + 1) - old_supervisor_salary in
  let new_supervisor_salary := new_avg_salary * (n_workers + 1) - W in
  new_supervisor_salary = 510 := 
sorry

end new_supervisor_salary_l8_8814


namespace distance_from_atlanta_to_seattle_l8_8893

-- Definitions of the coordinates on the complex plane and distance preservation
def seattle : ℂ := 0
def miami : ℂ := 3000 * complex.I
def atlanta : ℂ := 900 + 1200 * complex.I

-- Distance between Seattle and Atlanta
noncomputable def distance_sa : ℝ := complex.abs (atlanta - seattle)

-- The theorem we want to prove
theorem distance_from_atlanta_to_seattle : distance_sa = 1500 := by
  sorry

end distance_from_atlanta_to_seattle_l8_8893


namespace tangent_circumcircles_l8_8626

theorem tangent_circumcircles
  (ABC A_B A_C O_a O_b O_c : Type) 
  [triangle ABC]
  [perpendicular_bisector_intersects_bc (BC : line) (AB : line) (AC : line) A_B A_C]
  (circumcenter_aa_ba_c : Type) 
  (circumcenter_bb_cb_c : Type)
  (circumcenter_cc_ac_b : Type)
  (O_a_center : is_circumcenter ABC A_B A_C O_a)
  (O_b_center : is_circumcenter ABC B_A B_C O_b)
  (O_c_center : is_circumcenter ABC C_A C_B O_c) :
  tangent (circumcircle O_a O_b O_c) (circumcircle ABC) :=
sorry

end tangent_circumcircles_l8_8626


namespace main_theorem_l8_8053

noncomputable def a : ℝ := sorry

noncomputable def x1 : ℝ :=
  classical.some (exists_quadratic_roots (a - 1) 2)

noncomputable def x2 : ℝ :=
  classical.some_spec (exists_quadratic_roots (a - 1) 2)

noncomputable def x3 : ℝ :=
  classical.some (exists_quadratic_roots (-a - 1) 2)

noncomputable def x4 : ℝ :=
  classical.some_spec (exists_quadratic_roots (-a - 1) 2)

theorem main_theorem : x3 - x1 = 3 * (x4 - x2) → x4 - x2 = a / 2 :=
by
  sorry

end main_theorem_l8_8053


namespace company_sales_difference_l8_8749

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l8_8749


namespace equal_frac_implies_x_zero_l8_8934

theorem equal_frac_implies_x_zero (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
sorry

end equal_frac_implies_x_zero_l8_8934


namespace find_k_l8_8070

-- Definitions of the points and the slope condition.
def point1 : ℝ × ℝ := (-1, -4)
def x1 : ℝ := point1.1
def y1 : ℝ := point1.2

def x2 : ℝ := 3
def y2 : ℝ := k

def slope : ℝ := (k - y1) / (x2 - x1)

theorem find_k (k : ℝ) (h : slope = k) : k = 4 / 3 :=
by 
  have h1 : slope = (k + 4) / 4 := by sorry
  rw [h] at h1
  have h2 : k = (k + 4) / 4 := by sorry
  linarith

end find_k_l8_8070


namespace number_of_ways_at_least_one_different_l8_8097

open Finset Nat

theorem number_of_ways_at_least_one_different :
  (∑ x in (finset.range 4).powersetLen 2, ∑ y in (finset.range (4 - x.card)).powersetLen 2, 1) +
  (∑ x in (finset.range 4).powersetLen 1, ∑ y in (finset.range (3 - x.card)).powersetLen 1, ∑ z in (finset.range (2 - y.card)).powersetLen 1, 1) = 30 :=
by
  sorry

end number_of_ways_at_least_one_different_l8_8097


namespace sum_of_roots_angle_l8_8003

def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ * real.pi / 180)

theorem sum_of_roots_angle :
  ∑ k in finset.range 6, (225 + 360 * k) / 6 = 1170 :=
by sorry

end sum_of_roots_angle_l8_8003


namespace equilateral_triangle_lines_l8_8475

theorem equilateral_triangle_lines (T : Type) [equilateral_triangle T]
  (altitude_median_bisector : ∀ v : vertex T, ∃ l : line T, is_altitude l v ∧ is_median l v ∧ is_angle_bisector l v) :
  distinct_lines T = 3 := 
by 
  sorry

end equilateral_triangle_lines_l8_8475


namespace problem1_problem2_l8_8179

def A (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a i
def B (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1) + 1, a i
def C (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1) + 2, a i

theorem problem1 (a : ℕ → ℕ) (h₁ : a 0 = 1) (h₂ : a 1 = 5)
  (h₃ : ∀ n : ℕ, B a n - A a n = C a n - B a n) :
  ∀ n : ℕ, a n = 4 * n - 3 :=
sorry

theorem problem2 (a : ℕ → ℕ) (q : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n * q) ↔
  (∀ n : ℕ, B a n = q * A a n ∧ C a n = q * B a n) :=
sorry

end problem1_problem2_l8_8179


namespace unit_is_liters_l8_8107

def unit_for_amount_of_water_beibei_uses (amount : ℕ) : String :=
  if amount = 2 then "Liters" else "Unknown"

theorem unit_is_liters (amount: ℕ) (h : amount = 2) : unit_for_amount_of_water_beibei_uses amount = "Liters" :=
by
  rw [unit_for_amount_of_water_beibei_uses]
  simp [h]
  sorry

end unit_is_liters_l8_8107


namespace B4E_base16_to_base10_l8_8123

theorem B4E_base16_to_base10 : 
  let B := 11
  let four := 4
  let E := 14
  (B * 16^2 + four * 16^1 + E * 16^0) = 2894 := 
by 
  let B := 11
  let four := 4
  let E := 14
  calc
    B * 16^2 + four * 16^1 + E * 16^0 = 11 * 256 + 4 * 16 + 14 : by rfl
    ... = 2816 + 64 + 14 : by rfl
    ... = 2894 : by rfl

end B4E_base16_to_base10_l8_8123


namespace largest_multiple_of_8_less_than_100_l8_8780

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8780


namespace polynomial_sum_l8_8654

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l8_8654


namespace sum_of_a_and_b_in_sequence_l8_8597

theorem sum_of_a_and_b_in_sequence (a b : ℕ) 
    (h : ∏ n in (finset.range (a - 4) + 3), (n + 1) / n = 16) : 
    a + b = 95 :=
sorry

end sum_of_a_and_b_in_sequence_l8_8597


namespace janice_purchase_l8_8254

theorem janice_purchase (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 30 * a + 200 * b + 300 * c = 3000) : a = 20 :=
sorry

end janice_purchase_l8_8254


namespace inclination_angle_of_line_l8_8356

theorem inclination_angle_of_line : 
  ∃ α : ℝ, (∀ (x y : ℝ), x + sqrt 3 * y + 1 = 0 → tan α = -sqrt 3 / 3) ∧ α = 5 * Real.pi / 6 := 
by {
  sorry
}

end inclination_angle_of_line_l8_8356


namespace total_ticket_cost_l8_8374

theorem total_ticket_cost 
  (young_discount : ℝ := 0.55) 
  (old_discount : ℝ := 0.30) 
  (full_price : ℝ := 10)
  (num_young : ℕ := 2) 
  (num_middle : ℕ := 2) 
  (num_old : ℕ := 2) 
  (grandma_ticket_cost : ℝ := 7) :
  2 * (full_price * young_discount) + 2 * full_price + 2 * grandma_ticket_cost = 43 :=
by 
  sorry

end total_ticket_cost_l8_8374


namespace circle_subset_exists_l8_8967

-- Define the notion of "circle" and "intersection" for the problem
structure Circle (α : Type) :=
(center : α × α)
(radius : ℝ)

def intersect (c1 c2 : Circle ℝ) : Prop :=
let d := dist c1.center c2.center in d ≤ c1.radius + c2.radius && d ≥ real.abs (c1.radius - c2.radius)

-- The problem statement
theorem circle_subset_exists (circles : fin 2015 → Circle ℝ):
  ∃ (S : set (fin 2015)), S.card = 27 ∧ ∀ (i j ∈ S), (intersect (circles i) (circles j) ∨ ¬intersect (circles i) (circles j)) :=
by sorry

end circle_subset_exists_l8_8967


namespace area_of_triangle_tangent_at_origin_l8_8189

-- Define the given curve
def curve (x : ℝ) : ℝ := exp (-2 * x) + 1

-- Definitions for the tangent line, y=0, and y=x, and calculation of area
theorem area_of_triangle_tangent_at_origin :
  let tangent_line : ℝ → ℝ := λ x, -2 * x + 2,
      x_axis : ℝ → ℝ := λ x, 0,
      y_equals_x : ℝ → ℝ := λ x, x,
      intersection_x_axis : ℝ × ℝ := (1, 0),
      intersection_y_equals_x : ℝ × ℝ := (2 / 3, 2 / 3),
      base : ℝ := fst intersection_x_axis - (fst intersection_y_equals_x),
      height : ℝ := 2,
      area : ℝ := 1 / 2 * base * height
  in area = 1 / 3 := sorry

end area_of_triangle_tangent_at_origin_l8_8189


namespace find_heaviest_or_lightest_l8_8734

theorem find_heaviest_or_lightest (stones : Fin 10 → ℝ)
  (h_distinct: ∀ i j : Fin 10, i ≠ j → stones i ≠ stones j)
  (h_pairwise_sums_distinct : ∀ i j k l : Fin 10, 
    i ≠ j → k ≠ l → stones i + stones j ≠ stones k + stones l) :
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≥ stones j) ∨ 
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≤ stones j) :=
sorry

end find_heaviest_or_lightest_l8_8734


namespace find_max_n_l8_8057

def max_n (b : ℕ → ℕ) : ℕ :=
  ∃ N, ∀ n, (n ≤ N → (∀ i, i < n → b i % 2 = 1) ∧ ((∑ i in range n, b i ^ 2) % 4 = 0))

theorem find_max_n : max_n (λ i, 2 * i + 1) = 4 := by
  sorry

end find_max_n_l8_8057


namespace library_charge_l8_8825

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l8_8825


namespace pascal_triangle_row_20_sum_l8_8890

theorem pascal_triangle_row_20_sum :
  (Nat.choose 20 2) + (Nat.choose 20 3) + (Nat.choose 20 4) = 6175 :=
by
  sorry

end pascal_triangle_row_20_sum_l8_8890


namespace bagged_rice_probability_l8_8367

noncomputable def Φ : ℝ → ℝ := sorry -- Cumulative distribution function for the standard normal distribution

theorem bagged_rice_probability :
  let μ := 10
  let σ := 0.1
  let ξ := sorry -- Random variable following normal distribution N(10, 0.01)
  let P := sorry -- Probability function for the random variable ξ
  P(9.8 < ξ ∧ ξ < 10.2) = 2 * Φ(2) - 1 :=
by
  sorry

end bagged_rice_probability_l8_8367


namespace part1_part2_l8_8666

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end part1_part2_l8_8666


namespace function_properties_l8_8977

noncomputable def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l8_8977


namespace minimum_value_l8_8958

theorem minimum_value (a : ℝ) (h₀ : 0 < a) (h₁ : a < 3) :
  ∃ a : ℝ, (0 < a ∧ a < 3) ∧ (1 / a + 4 / (8 - a) = 9 / 8) := by
sorry

end minimum_value_l8_8958


namespace proposition_d_correct_l8_8864

theorem proposition_d_correct : ∀ x ∈ set.Ioi 0, exp x > 1 + x := 
by {
  sorry
}

end proposition_d_correct_l8_8864


namespace find_retail_price_l8_8407

-- Define the conditions
def wholesale_price : ℝ := 90
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.20

-- Calculate the necessary values from conditions
def profit : ℝ := profit_rate * wholesale_price
def selling_price : ℝ := wholesale_price + profit
def discount_factor : ℝ := 1 - discount_rate

-- Rewrite the main theorem statement
theorem find_retail_price : ∃ w : ℝ, discount_factor * w = selling_price → w = 120 :=
by sorry

end find_retail_price_l8_8407


namespace expenses_as_percentage_of_revenue_l8_8225

variables {x y : ℝ} -- Revenue and Expenses in 2005
variables {revenue2006 expenses2006 : ℝ}

-- Conditions on revenue and expenses increases
def revenueIncrease := revenue2006 = 1.25 * x
def expensesIncrease := expenses2006 = 1.15 * y

-- Condition on profit increase
def profitIncrease := (1.25 * x - 1.15 * y) = 1.4 * (x - y)

theorem expenses_as_percentage_of_revenue (h₁ : revenueIncrease) (h₂ : expensesIncrease) (h₃ : profitIncrease) :
  expenses2006 / revenue2006 = 0.552 := by
  sorry

end expenses_as_percentage_of_revenue_l8_8225


namespace max_value_on_interval_max_value_of_a_l8_8573

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Define the maximum value M(a) within the interval [-4, -2]
def M (a : ℝ) (h : a > 0) : ℝ :=
  if h1 : 0 < a ∧ a ≤ 1/6 then f a (-2)
  else if h2 : a > 1/6 then f a (-4)
  else sorry

-- Prove that M(a) is as defined above within the range [-4, -2]
theorem max_value_on_interval (a : ℝ) (h : a > 0) :
  M a h = if h1 : 0 < a ∧ a ≤ 1/6 then 4 * a - 1
          else if h2 : a > 1/6 then 16 * a - 3
          else 0 := sorry

-- Define the constraints for roots x1, x2, and ratio
theorem max_value_of_a (a : ℝ) (h : a > 0)
  (x1 x2 : ℝ)
  (hx : (x1 * x2 = 1 / a) ∧ (x1 + x2 = -1 / a) ∧ (x1 / x2 ∈ Icc (1/10) 10)) :
  a ≤ 1/4 := sorry

end max_value_on_interval_max_value_of_a_l8_8573


namespace megatek_manufacturing_percentage_l8_8417

theorem megatek_manufacturing_percentage 
  (total_degrees : ℝ := 360)
  (manufacturing_degrees : ℝ := 18)
  (is_proportional : (manufacturing_degrees / total_degrees) * 100 = 5) :
  (manufacturing_degrees / total_degrees) * 100 = 5 := 
  by
  exact is_proportional

end megatek_manufacturing_percentage_l8_8417


namespace largest_n_for_arithmetic_sequence_product_c_n_d_n_max_value_l8_8872

theorem largest_n_for_arithmetic_sequence_product :
  ∃ (u v : ℤ) (n : ℕ), (∀ i, c i = 1 + (i - 1) * u) ∧ (∀ i, d i = 1 + (i - 1) * v) ∧ ((c n) * (d n) = 1764) ∧ n ≤ 1764 :=
sorry

noncomputable def c (i : ℕ) (u : ℤ) : ℤ := 1 + (i - 1) * u
noncomputable def d (i : ℕ) (v : ℤ) : ℤ := 1 + (i - 1) * v

theorem c_n_d_n_max_value (u v : ℤ) (hn : 1 ≤ n) (hc1 : c 1 u = 1)
  (hd1 : d 1 v = 1) (hc2_le_hd2 : c 2 u ≤ d 2 v) (prod_eq : (c n u) * (d n v) = 1764) :
  n = 1764 := 
sorry

end largest_n_for_arithmetic_sequence_product_c_n_d_n_max_value_l8_8872


namespace range_of_b_l8_8678

def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * log (x + 1)

def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 + 2 * x + b

theorem range_of_b (b : ℝ) (f_has_extrema : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ (f x₁ b < f x₂ b) ∧ (∀ x ∈ set.Icc x₁ x₂, f x₁ b ≤ f x b)) : 
  0 < b ∧ b < 1 / 2 :=
sorry

end range_of_b_l8_8678


namespace probability_is_one_twelfth_l8_8013

def probability_red_gt4_green_odd_blue_lt4 : ℚ :=
  let total_outcomes := 6 * 6 * 6
  let successful_outcomes := 2 * 3 * 3
  successful_outcomes / total_outcomes

theorem probability_is_one_twelfth :
  probability_red_gt4_green_odd_blue_lt4 = 1 / 12 :=
by
  -- proof here
  sorry

end probability_is_one_twelfth_l8_8013


namespace find_interest_rate_l8_8451

-- Define the initial conditions
def simple_interest := 70.0
def principal := 499.99999999999994  -- We are not approximating to 500 here
def time := 4.0

-- Define the problem to prove
theorem find_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) : 
  SI = simple_interest → 
  P = principal → 
  T = time → 
  (SI / (P * T)) = 0.035 :=
by
  intros hSI hP hT
  sorry

end find_interest_rate_l8_8451


namespace exists_irrational_r_l8_8299

noncomputable def r (k : ℕ) (hk : k ≥ 2) : ℝ :=
  k + Real.sqrt (k^2 - k)

theorem exists_irrational_r (k : ℕ) (hk : k ≥ 2) : 
  ∃ (r : ℝ), (irrational r) ∧ (∀ (m : ℕ), (⌊r^m⌋ % k = k - 1))  :=
begin
  sorry
end

end exists_irrational_r_l8_8299


namespace exists_six_subjects_l8_8311

-- Define the number of students and subjects
def students := Fin 7
def subjects := Fin 12

-- Assume each student has a unique 12-tuple of marks represented by a function from students to subjects
variables (marks : students → subjects → ℕ) 

-- Condition: No two students have identical marks in all 12 subjects
axiom unique_marks : ∀ x y : students, x ≠ y → ∃ s : subjects, marks x s ≠ marks y s

-- Prove that we can choose 6 subjects such that any two of the students have different marks in at least one of these subjects
theorem exists_six_subjects : ∃ (S : Finset subjects) (h : S.card = 6), 
  ∀ x y : students, x ≠ y → ∃ s ∈ S, marks x s ≠ marks y s :=
sorry

end exists_six_subjects_l8_8311


namespace value_ratio_l8_8620

theorem value_ratio : 
  (let ten_thousands := 10000 in 
   let tenths := 0.1 in 
   ten_thousands / tenths = 100000) :=
by 
  let ten_thousands := 10000
  let tenths := 0.1
  sorry

end value_ratio_l8_8620


namespace jack_money_proof_l8_8635

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l8_8635


namespace fuel_consumption_range_min_fuel_consumption_l8_8829

-- Conditions
variable (x k : ℝ)
variable (hx : 60 ≤ x ∧ x ≤ 120)
variable (hk : 60 ≤ k ∧ k ≤ 100)
variable (fuel_consumption : ℝ := (1/5) * (x - k + 4500 / x))

-- Problem Part 1
theorem fuel_consumption_range (hk_eq : k = 100)
  (hfuel : fuel_consumption ≤ 9) : 60 ≤ x ∧ x ≤ 100 :=
by sorry

-- Problem Part 2
def fuel_consumption_for_100_km (x : ℝ) (k : ℝ) : ℝ :=
  20 - (20 * k) / x + (90000 / (x^2))

theorem min_fuel_consumption (k : ℝ) 
  (hk : 60 ≤ k ∧ k ≤ 100) :
  (75 ≤ k ∧ k < 100 → ∃ y_min, y_min = 20 - k^2 / 900) ∧
  (60 ≤ k ∧ k < 75 → ∃ y_min, y_min = (105 / 4) - k / 6) :=
by sorry

end fuel_consumption_range_min_fuel_consumption_l8_8829


namespace rope_segments_divided_l8_8856

theorem rope_segments_divided (folds1 folds2 : ℕ) (cut : ℕ) (h_folds1 : folds1 = 3) (h_folds2 : folds2 = 2) (h_cut : cut = 1) :
  (folds1 * folds2 + cut = 7) :=
by {
  -- Proof steps would go here
  sorry
}

end rope_segments_divided_l8_8856


namespace cost_vs_selling_price_ratio_l8_8296

-- Define variables for cost price per pencil and selling price per pencil
variables (C S : ℝ)

-- Define conditions
def cost_price_70_pencils := 70 * C
def loss_20_pencils := 20 * S
def selling_price_70_pencils := cost_price_70_pencils - loss_20_pencils

-- Define the theorem to be proven
theorem cost_vs_selling_price_ratio 
  (h1 : selling_price_70_pencils = cost_price_70_pencils - loss_20_pencils)
  : (cost_price_70_pencils / selling_price_70_pencils) = (C / (C - 2 * S / 7)) := 
by {
  -- Proof will be provided here
  sorry
}

end cost_vs_selling_price_ratio_l8_8296


namespace largest_multiple_of_8_less_than_100_l8_8770

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8770


namespace irrational_minus_sqrt3_l8_8800

theorem irrational_minus_sqrt3 : 
  ∀ x : ℝ, (x = -2) ∨ (x = 0.1010) ∨ (x = 1/3) ∨ (x = -sqrt 3) → irrational x ↔ x = -sqrt 3 :=
by
  sorry

end irrational_minus_sqrt3_l8_8800


namespace locus_equation_sum_of_slopes_l8_8941

-- Define the conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def is_on_locus (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  y*y / ((x + 2) * (x - 2)) = -3/4

-- The main statement for the first part of the problem: the equation of the locus
theorem locus_equation (P : ℝ × ℝ) (h : is_on_locus P) : 
  (P.1^2) / 4 + (P.2^2) / 3 = 1 := 
sorry

-- Define the conditions for the second part of the problem
def is_inside_ellipse (D : ℝ × ℝ) : Prop := 
  (D.1)^2 / 4 + (D.2)^2 / 3 < 1

-- Define the cyclic quadrilateral condition and sum of slopes
theorem sum_of_slopes (D : ℝ × ℝ) 
  (hD : is_inside_ellipse D) 
  (l1 l2 : ℝ × ℝ → Prop) -- Representing the lines passing through D and forming a cyclic quadrilateral
  (h_cyclic : ∀ P Q R S, l1 P → l1 Q → l2 R → l2 S → 
    P ≠ D ∧ Q ≠ D ∧ R ≠ D ∧ S ≠ D ∧
    (P, Q, R, S) forms_cyclic_quadrilateral) : 
  slope l1 D + slope l2 D = 0 :=
sorry

end locus_equation_sum_of_slopes_l8_8941


namespace bug_final_position_l8_8163

theorem bug_final_position (n : ℕ) (start : ℕ) (points : Finset ℕ) 
  (hpoints : points = {1, 2, 3, 4, 5}) 
  (hstart : start = 5) 
  (hodd_move : ∀ p ∈ points, p % 2 = 1 → (p + 2) % 5 ∈ points) 
  (heven_move : ∀ p ∈ points, p % 2 = 0 → (p + 3) % 5 ∈ points) 
  : ∀ k, k = 2010 → find_position k start = 2 := 
by 
  sorry

noncomputable def find_position : ℕ → ℕ → ℕ
| 0, p := p
| (n+1), p := if p % 2 = 0 then 
                find_position n ((p + 3) % 5)
              else 
                find_position n ((p + 2) % 5)

end bug_final_position_l8_8163


namespace evaluate_expression_l8_8487

def a := (64 : ℝ) ^ (-1 / 3 : ℝ)
def b := (81 : ℝ) ^ (-1 / 2 : ℝ)
def result := a + b

theorem evaluate_expression : result = (13 / 36 : ℝ) :=
by 
  sorry

end evaluate_expression_l8_8487


namespace resistance_per_meter_l8_8002

@[inline] def resistance_between_points (R0 : ℝ) (distance : ℝ) : ℝ :=
    if distance = 2 then
        15
    else
        0  -- inappropriate distance is not considered.

theorem resistance_per_meter :
    resistance_between_points 15 2 = 15 :=
by
    rfl

end resistance_per_meter_l8_8002


namespace correct_option_is_B_l8_8804

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l8_8804


namespace last_passenger_probability_l8_8347

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l8_8347


namespace largest_prime_factor_12321_l8_8516

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : Nat) : Nat :=
  if n = 1 then 1
  else
    let factors := (List.range (n + 1)).reverse.filter (λ m => m ∣ n ∧ is_prime m)
    factors.head!

theorem largest_prime_factor_12321 :
  largest_prime_factor 12321 = 47 :=
by
  sorry

end largest_prime_factor_12321_l8_8516


namespace no_integer_n_7n_plus_n3_divisible_by_9_l8_8935

theorem no_integer_n_7n_plus_n3_divisible_by_9 :
  ∀ (n : ℤ), (7^n + n^3) % 9 ≠ 0 := 
by
  sorry

end no_integer_n_7n_plus_n3_divisible_by_9_l8_8935


namespace identify_conic_section_hyperbola_l8_8401

-- Define the given equation as part of the conditions.
def given_equation (x y : ℝ) : Prop := (x - 3)^2 = 3 * (2 * y + 4)^2 - 75

-- State the problem to prove the type of conic section is a hyperbola.
theorem identify_conic_section_hyperbola (x y : ℝ) (h : given_equation x y) : 
  "H" :=
begin
  sorry
end

end identify_conic_section_hyperbola_l8_8401


namespace last_passenger_probability_l8_8344

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l8_8344


namespace f_nested_value_l8_8211

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x * (1 - x)
  else if 1 < x ∧ x ≤ 2 then Real.cos (π * x)
  else f (x - 4)

theorem f_nested_value : f (f (29 / 3)) = 1 / 4 := sorry

end f_nested_value_l8_8211


namespace question1_solution_question2_solution_l8_8536

-- Definitions based on the conditions:

def circle_radius : ℝ := 2

-- Defining point P with coordinates (1, a)
def point_P (a : ℝ) : ℝ × ℝ := (1, a)

-- Circle equation: x^2 + y^2 = 4, which means radius squared is 4
def is_on_circle (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = circle_radius ^ 2

-- Tangent equation construction given point on circle and slope/condition
def tangent_equation (P : ℝ × ℝ) (m : ℝ) : (ℝ × ℝ) → Prop :=
  λ Q, Q.1 + m * Q.2 = m * P.2 + P.1 - circle_radius

-- Question 1 proof statement:
theorem question1_solution (a : ℝ) (h: is_on_circle (point_P a)) :
  a = sqrt 3 ∨ a = -sqrt 3 :=
begin
  sorry
end

-- Definitions for chords and area calculation, given a = sqrt(2)

-- Distance from origin to a line equation representation
def distance_to_origin (line : ℝ × ℝ → Prop) : ℝ := 
  -- Placeholder for distance formula
  sorry 

-- Length of chord AC or BD given distance from origin
def chord_length (d : ℝ) : ℝ := 2 * sqrt(circle_radius^2 - d^2)

-- Area calculation when AC and BD are perpendicular
def area_ABCD (d1 d2 : ℝ) : ℝ :=
  2 * sqrt(16 - 4 * (d1^2 + d2^2) + d1^2 * d2^2)

-- Question 2 proof statement given specific conditions for maximum area
theorem question2_solution (a := sqrt 2) :
  ∃ d1 d2, distance_to_origin sorry = d1 ∧ distance_to_origin sorry = d2 ∧
  area_ABCD d1 d2 ≤ 5 :=
begin
  sorry
end

end question1_solution_question2_solution_l8_8536


namespace largest_positive_integer_l8_8413

def binary_op (n : ℕ) : ℤ := n - (n * 5)

theorem largest_positive_integer (n : ℕ) (h : binary_op n < 21) : n ≤ 1 := 
sorry

end largest_positive_integer_l8_8413


namespace min_distance_l8_8271

def parabola (x : ℝ) : ℝ := x^2 - 6 * x + 11
def line (x : ℝ) : ℝ := 2 * x - 5

theorem min_distance : 
  ∃ a b d : ℝ, 
  parabola a = b ∧ 
  d = (|2 * a - b - 5|) / sqrt 5 ∧ 
  ∀ a', d ≤ (|2 * a' - (parabola a') - 5|) / sqrt 5 ∧ 
  d = 16 * sqrt 5 / 5 := 
sorry

end min_distance_l8_8271


namespace largest_multiple_of_8_less_than_100_l8_8776

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8776


namespace no_intersection_implies_t_ge_one_l8_8575

variables {M P : set ℝ}
def M : set ℝ := {x ∈ ℝ | x ≤ 1}
def P (t : ℝ) : set ℝ := {x ∈ ℝ | x > t}

theorem no_intersection_implies_t_ge_one (t : ℝ) (h : M ∩ P t = ∅) : t ≥ 1 :=
by sorry

end no_intersection_implies_t_ge_one_l8_8575


namespace sixth_piggy_bank_coins_l8_8039

theorem sixth_piggy_bank_coins :
  ∀ (n1 n2 n3 n4 n5 : ℕ),
    n1 = 72 →
    n2 = n1 + 9 →
    n3 = n2 + 9 →
    n4 = n3 + 9 →
    n5 = n4 + 9 →
    n5 + 9 = 117 :=
by
  intros n1 n2 n3 n4 n5 h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact rfl

end sixth_piggy_bank_coins_l8_8039


namespace decreasing_f_l8_8551

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then (2 * a - 1) * x + 4 * a else -x + 1

theorem decreasing_f (a : ℝ) (h1 : 2 * a - 1 < 0) (h2 : 6 * a ≥ 1) : a ∈ set.Icc (1/6 : ℝ) (1/2 : ℝ) :=
by {
  sorry,
}

end decreasing_f_l8_8551


namespace length_BX_l8_8273

theorem length_BX
  (A B C O D E X Y : Type)
  (h1 : ∠ BAC < 90)
  (h2 : ∠ ABC < 90)
  (h3 : ∠ ACB < 90)
  (h_triangle : triangle A B C)
  (h_circumcenter : circumcenter O A B C)
  (h_AB : dist A B = 4)
  (h_AC : dist A C = 5)
  (h_BC : dist B C = 6)
  (h_D : foot D A B C)
  (h_E : line_through A O ∧ lies_on E O B C)
  (h_X : lies_on X B C ∧ between D X E)
  (h_exists_Y : ∃ Y, lies_on Y A D ∧ parallel XY AO ∧ perpendicular YO AX) :
  dist B X = 96 / 41 :=
begin
  sorry
end

end length_BX_l8_8273


namespace num_square_tiles_is_zero_l8_8434

def triangular_tiles : ℕ := sorry
def square_tiles : ℕ := sorry
def hexagonal_tiles : ℕ := sorry

axiom tile_count_eq : triangular_tiles + square_tiles + hexagonal_tiles = 30
axiom edge_count_eq : 3 * triangular_tiles + 4 * square_tiles + 6 * hexagonal_tiles = 120

theorem num_square_tiles_is_zero : square_tiles = 0 :=
by
  sorry

end num_square_tiles_is_zero_l8_8434


namespace jack_money_proof_l8_8636

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l8_8636


namespace triangle_concurrency_l8_8178

-- Define the vertices, feet of the altitudes, and other necessary points on the triangle
variables {A B C D E F A1 B1 C1 A2 B2 C2 : Type}

-- Non-isosceles acute triangle condition
axiom triangle_non_isosceles_acute {α : Type} (t : α)
  (non_isosceles_acute : ∀ (A B C : t), non_isosceles_acute_triangle A B C) : Prop

-- Definitions for points on the triangle and feet
axiom altitude_feet {α : Type} (t : α) 
  (A B C D E F : t) (AD_CF BE_CF : is_ALT D A) : Prop

-- Circumcircle intersections and concurrency
axiom circumcircle_intersections {α : Type} 
  (A B C A1 B1 C1 : α) (ABC_circumcircle : is_CIRCC A B C) (intersects_circumcircle : ∀ (δ : Type), intersects δ (ABC_circumcircle)) : Prop

-- Points specific for AA_2, BB_2, CC_2 concurrency
axiom concurrency_points {α : Type}
  (A B C A2 B2 C2 : α) (concurrency_circumcircle : is_CIRCC A B C) (intersects_circumcircle_A2_B2_C2 : ∀ (δ : Type), intersects_A2_B2_C2 δ (concurrency_circumcircle)) : Prop

-- Variables for lengths of triangle sides
variables {a b c : ℝ}

-- Main theorem statement
theorem triangle_concurrency
    ( hoc : triangle_non_isosceles_acute triangle )
    ( af : altitude_feet triangle )
    ( cc_int : circumcircle_intersections )
    ( cc_pts : concurrency_points )
    ( lengths : lengths_of_triangle_sides a b c ) :
  (( AA_2 * AA_2 + BB_2 * BB_2 + CC_2 * CC_2 ) ≥ ((4 * (a*b + b*c + c*a)) ^ 2) / ((3 * (a^2 + b^2 + c^2)))) := 
sorry

end triangle_concurrency_l8_8178


namespace hexagon_angle_sequences_l8_8720

theorem hexagon_angle_sequences :
  ∃ n : ℕ, n = 4 ∧
  ∃ (x d : ℕ), 
  (30 ≤ x) ∧ 
  (∀ i, i ∈ (list.range 6) → 
    let ai := x + i * d in ai < 160 ∧ 
    ai ∉ (list.map (λ j, x + j * d) (list.range i)) ∧
    6 * x + 15 * d = 720) :=
by
  sorry

end hexagon_angle_sequences_l8_8720


namespace circle_points_AC_length_l8_8686

theorem circle_points_AC_length :
  ∀ (O A B C E : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space E]
  (r : ℝ)
  (h_circle: ∀ X ∈ O, metric_space.dist O X = r)  -- X lies on a circle with radius r
  (A B : O) (h_dist_AB : metric_space.dist A B = 8)
  (C : O) (C_midpoint_minor_arc : true)  -- C is the midpoint of minor arc AB
  ,
  metric_space.dist A C = real.sqrt (98 - 14 * real.sqrt 33) :=
begin
  sorry
end

end circle_points_AC_length_l8_8686


namespace sticker_price_l8_8582

theorem sticker_price (x : ℝ) (h1 : 0.8 * x - 100 = 0.7 * x - 25) : x = 750 :=
by
  sorry

end sticker_price_l8_8582


namespace andre_max_points_visited_l8_8104
noncomputable def largest_points_to_visit_in_alphabetical_order : ℕ :=
  10

theorem andre_max_points_visited : largest_points_to_visit_in_alphabetical_order = 10 := 
by
  sorry

end andre_max_points_visited_l8_8104


namespace harmonic_mean_of_2_3_6_l8_8112

def harmonic_mean (a b c : ℕ) : ℚ :=
  3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_of_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  -- Definition of harmonic mean
  let h_mean := harmonic_mean 2 3 6
  -- Reciprocals sum
  have h_sum : (1 / 2 : ℚ) + (1 / 3) + (1 / 6) = 1 := by norm_num
  -- Harmonic mean calculation
  have h_mean : h_mean = 3 := by
    unfold harmonic_mean
    rw [h_sum]
    norm_num
  exact h_mean

-- Harmonically calculate the mean to show equivalence.
example : harmonic_mean 2 3 6 = 3 := harmonic_mean_of_2_3_6

end harmonic_mean_of_2_3_6_l8_8112


namespace evaluate_expr_equiv_l8_8492

theorem evaluate_expr_equiv : 64^(-1 / 3) + 81^(-2 / 4) = 13 / 36 := by
  have h1 : 64 = 2^6 := rfl
  have h2 : 81 = 3^4 := rfl
  sorry

end evaluate_expr_equiv_l8_8492


namespace last_passenger_probability_l8_8348

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l8_8348


namespace largest_multiple_of_8_less_than_100_l8_8774

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8774


namespace largest_m_dividing_factorial_l8_8511

theorem largest_m_dividing_factorial :
  (∃ m : ℕ, (∀ n : ℕ, (18^n ∣ 30!) ↔ n ≤ m) ∧ m = 7) :=
by
  sorry

end largest_m_dividing_factorial_l8_8511


namespace find_smallest_even_number_l8_8092

theorem find_smallest_even_number (x : ℕ) (h1 : 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14)) = 424) : 
  x = 46 := 
by
  sorry

end find_smallest_even_number_l8_8092


namespace product_of_abc_l8_8008

-- Define the constants and conditions
variables (a b c m : ℝ)
axiom h1 : a + b + c = 180
axiom h2 : 5 * a = m
axiom h3 : b = m + 12
axiom h4 : c = m - 6

-- Prove that the product of a, b, and c is 42184
theorem product_of_abc : a * b * c = 42184 :=
by {
  sorry
}

end product_of_abc_l8_8008


namespace recording_time_per_tape_l8_8446

theorem recording_time_per_tape (total_time : ℕ) (max_time_per_tape : ℕ) (h_total : total_time = 480) (h_max : max_time_per_tape = 60) :
  (total_time / max_time_per_tape) = 8 → total_time / (total_time / max_time_per_tape) = max_time_per_tape :=
by
  intro h_tapes
  rw [h_total, h_max, h_tapes]
  sorry

end recording_time_per_tape_l8_8446


namespace fiona_unique_pairs_l8_8232

/--
In a park, Fiona observes two groups of people: one with 12 teenagers and another with 8 adults. 
She decides to count how many unique pairs she can observe where each pair could be either two teenagers, 
two adults, or one teenager and one adult.
-/
theorem fiona_unique_pairs : 
  let num_teenagers := 12
  let num_adults := 8
  nat.choose num_teenagers 2 + nat.choose num_adults 2 + num_teenagers * num_adults = 190 :=
by
  sorry

end fiona_unique_pairs_l8_8232


namespace hyperbola_focal_length_l8_8568

noncomputable def hyperbola_data : Type :=
  Σ (a : ℝ), (h_a_pos : a > 0) × (∃ (c : ℝ), 2 * c = 10 ∧ (20 / a^2 = 4))

theorem hyperbola_focal_length (data : hyperbola_data) : ∃ (c : ℝ), 2 * c = 10 :=
  by
    obtain ⟨a, ⟨h_a_pos, ⟨c, h_c⟩⟩⟩ := data
    sorry

end hyperbola_focal_length_l8_8568


namespace three_solutions_system_l8_8145

theorem three_solutions_system (a : ℝ) :
  (∃ x y : ℝ, (|x| + |y| - 2)^2 = 1 ∧ y = a * x + 5) ∧
  ∃! p, ∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    ((|x1| + |y1| - 2)^2 = 1 ∧ y1 = a * x1 + 5) ∧ 
    ((|x2| + |y2| - 2)^2 = 1 ∧ y2 = a * x2 + 5) ∧
    ((|x3| + |y3| - 2)^2 = 1 ∧ y3 = a * x3 + 5) :=
  a = 5 ∨ a = -5 ∨ a = 5 / 3 ∨ a = -5 / 3 :=
sorry

end three_solutions_system_l8_8145


namespace library_charge_l8_8826

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l8_8826


namespace exists_k_lt_2003_l8_8006

noncomputable def a : ℕ → ℝ
| 0     := 56
| (n+1) := a n - 1 / a n

theorem exists_k_lt_2003 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2002 ∧ a k < 0 := by
  sorry

end exists_k_lt_2003_l8_8006


namespace largest_multiple_of_8_less_than_100_l8_8762

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8762


namespace scientific_notation_example_l8_8603

theorem scientific_notation_example : 10500 = 1.05 * 10^4 :=
by
  sorry

end scientific_notation_example_l8_8603


namespace number_of_distinct_points_l8_8924

def is_solution (x y : ℝ) : Prop := (x ^ 2 + 9 * y ^ 2 = 9) ∧ (9 * x ^ 2 + y ^ 2 = 1)

theorem number_of_distinct_points : 
  {p : ℝ × ℝ | is_solution p.1 p.2}.to_finset.card = 2 :=
sorry

end number_of_distinct_points_l8_8924


namespace evaluate_expression_l8_8498

theorem evaluate_expression : 
  -20 + 15 * (4 ^ (-1 : ℤ) * 2) = -12.5 := 
by {
  -- Proof intentionally omitted
  sorry 
}

end evaluate_expression_l8_8498


namespace num_possible_values_of_n_l8_8728

theorem num_possible_values_of_n:
  ∀ (a n : ℤ), n > 1 → (n * (a + n - 1) = 153) → (n ∈ [3, 9, 17, 51, 153]) → 5 := sorry

end num_possible_values_of_n_l8_8728


namespace division_result_l8_8925

open Polynomial

noncomputable def P : Polynomial ℝ := X^5 - 25 * X^3 + 9 * X^2 - 17 * X + 12
noncomputable def Q : Polynomial ℝ := X - 3
noncomputable def Quot : Polynomial ℝ := X^4 + 3 * X^3 - 16 * X^2 - 39 * X - 134
noncomputable def Rem : ℝ := -312

theorem division_result :
  P = Q * Quot + C Rem :=
sorry

end division_result_l8_8925


namespace solve_system_of_equations_l8_8052

theorem solve_system_of_equations :
  ∃ (x y z u v : ℝ), 
    (x + y + z = 5) ∧ 
    (x^2 + y^2 + z^2 = 9) ∧ 
    (x * y + u + v * x + v * y = 0) ∧ 
    (y * z + u + v * y + v * z = 0) ∧ 
    (z * x + u + v * z + v * x = 0) ∧ 
    ((x, y, z, u, v) = (2, 2, 1, 4, -2) ∨
    (x, y, z, u, v) = (4/3, 4/3, 7/3, 16/9, -4/3) ∨
    (x, y, z, u, v) = (2, 1, 2, 4, -2) ∨
    (x, y, z, u, v) = (4/3, 7/3, 4/3, 16/9, -4/3) ∨
    (x, y, z, u, v) = (1, 2, 2, 4, -2) ∨
    (x, y, z, u, v) = (7/3, 4/3, 4/3, 16/9, -4/3)) :=
by decide

end solve_system_of_equations_l8_8052


namespace max_convenient_set_exists_l8_8084

def is_convenient_set (n : ℕ) (S : Fin n → Fin n → Bool) : Prop :=
  (∀ i : Fin n, ∃ j1 : Fin n, ∃ j2 : Fin n, j1 ≠ j2 ∧ S i j1 ∧ S i j2) ∧
  (∀ j : Fin n, ∃ i1 : Fin n, ∃ i2 : Fin n, i1 ≠ i2 ∧ S i1 j ∧ S i2 j)

def max_convenient_set_size (n : ℕ) : ℕ :=
  4 * (n - 2)

theorem max_convenient_set_exists (n : ℕ) (h : n ≥ 5) :
  ∃ S : Fin n → Fin n → Bool,
    is_convenient_set n S ∧
    (∀ T : Fin n → Fin n → Bool, (is_convenient_set n T → ∃ x y : Fin n, T x y = false) ∧ (∑ i j, if S i j then 1 else 0) = max_convenient_set_size n) :=
sorry

end max_convenient_set_exists_l8_8084


namespace hex_to_decimal_B4E_l8_8125

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end hex_to_decimal_B4E_l8_8125


namespace sum_of_solutions_of_fx_eq_0_l8_8278

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 7 * x + 10 else 3 * x - 15

theorem sum_of_solutions_of_fx_eq_0 :
  let x1 := -10 / 7
  let x2 := 5
  f x1 = 0 ∧ f x2 = 0 ∧ x1 ≤ 1 ∧ x2 > 1 → x1 + x2 = 25 / 7 :=
by
  sorry

end sum_of_solutions_of_fx_eq_0_l8_8278


namespace fly_position_changes_l8_8892

theorem fly_position_changes : 
  ∀ (flies : Cube → ℕ), 
  (∀ v : Cube, flies v = 1) ∧ 
  (∀ v : Cube, (new_position flies) v ≠ 0) → 
  number_of_ways (flies) = 81 :=
by
  -- Definitions and positions of flies
  def Cube := fin 8
  def faces (x : Cube) := {i : Cube // i ≠ x ∧ (opposite_face i x)}
  
  -- Moving flies according to conditions
  def new_position (flies : Cube → ℕ) (v : Cube) : Cube :=
    let face_v := faces v
    -- Function to determine new vertex position in the same face but diagonally opposite
    sorry
  
  -- Calculate the number of ways flies can move without overlapping
  def number_of_ways (flies : Cube → ℕ) : ℕ :=
    sorry
  
  -- The complete statement in Lean
  assume (flies : Cube → ℕ)
  assume (h1 : ∀ v : Cube, flies v = 1)
  assume (h2 : ∀ v : Cube, (new_position flies) v ≠ 0)
  show number_of_ways flies = 81
  sorry

end fly_position_changes_l8_8892


namespace sequence_is_arithmetic_sum_of_first_twenty_terms_l8_8574

noncomputable def a_n (n : ℕ) : ℚ := 9 / 2 - n

-- Statement (1)
theorem sequence_is_arithmetic : ∀ (n : ℕ), n ≥ 2 → a_n n - a_n (n - 1) = 1 := 
by sorry

-- Statement (2)
theorem sum_of_first_twenty_terms : ∑ i in finset.range 20, a_n i = -120 := 
by sorry

end sequence_is_arithmetic_sum_of_first_twenty_terms_l8_8574


namespace simplify_and_rationalize_l8_8703

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 7) * (sqrt 5 / sqrt 8) * (sqrt 6 / sqrt 9) = sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l8_8703


namespace tan_sub_pi_over_4_l8_8549

theorem tan_sub_pi_over_4 (θ : ℝ) (hθ1 : 0 < θ ∧ θ < 2 * π ∧ π / 2 < θ < π)
  (hθ2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = - 4 / 3 :=
sorry

end tan_sub_pi_over_4_l8_8549


namespace last_passenger_sits_in_assigned_seat_l8_8327

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l8_8327


namespace period_of_f_zeros_of_f_max_value_of_f_f_not_increasing_on_interval_l8_8932

noncomputable def f (x : ℝ) : ℝ := Real.sin x + (1/2) * Real.sin (2 * x)

theorem period_of_f : f (x + 2 * Real.pi) = f x :=
by
  sorry

theorem zeros_of_f : ∃ x y z ∈ set.Icc (0 : ℝ) (2 * Real.pi), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x = 0 ∧ f y = 0 ∧ f z = 0 :=
by
  sorry

theorem max_value_of_f : (∃ x ∈ set.Icc (0 : ℝ) (2 * Real.pi), f x = (3 * Real.sqrt 3) / 4) :=
by
  sorry

theorem f_not_increasing_on_interval : ¬∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), ∀ y ∈ set.Icc (0 : ℝ) (Real.pi / 2), x ≤ y → f x ≤ f y :=
by
  sorry

end period_of_f_zeros_of_f_max_value_of_f_f_not_increasing_on_interval_l8_8932


namespace prove_expression_l8_8931

-- Define the operation for real numbers
def op (a b c : ℝ) : ℝ := (a - b + c) ^ 2

-- Stating the theorem for the given expression
theorem prove_expression (x z : ℝ) :
  op ((x + z) ^ 2) ((z - x) ^ 2) ((x - z) ^ 2) = (x + z) ^ 4 := 
by  sorry

end prove_expression_l8_8931


namespace sum_of_digits_of_4_plus_2_pow_21_l8_8392

theorem sum_of_digits_of_4_plus_2_pow_21 :
  let x := (4 + 2)
  (x^(21) % 100).div 10 + (x^(21) % 100).mod 10 = 6 :=
by
  let x := (4 + 2)
  sorry

end sum_of_digits_of_4_plus_2_pow_21_l8_8392


namespace tangent_line_at_one_l8_8192

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_one : ∃ k b : ℝ, 
  (∀ x : ℝ, f'(x) = (1 / x) - 3) ∧ f'(1) = -2 ∧ (1,-3) → y = k * x + b ∧ k = -2 ∧ b = 1 :=
by
  sorry

end tangent_line_at_one_l8_8192


namespace parallel_lines_iff_a_eq_1_l8_8173

variable (a : ℝ)

def line1 := λ (x y : ℝ), a * x + 2 * y - 1 = 0
def line2 := λ (x y : ℝ), x + (a + 1) * y + 4 = 0

theorem parallel_lines_iff_a_eq_1 (a : ℝ) :
  (∀ x y, line1 a x y ↔ line2 a x y) ↔ (a = 1) :=
sorry

end parallel_lines_iff_a_eq_1_l8_8173


namespace range_of_m_l8_8190

def f (m x : ℝ) : ℝ := Real.log x - m * x + m

theorem range_of_m:
  (∀ x : ℝ, 0 < x → f m x ≤ 0) ↔ m = 1 := by sorry

end range_of_m_l8_8190


namespace problem_remainder_6_pow_83_add_8_pow_83_mod_49_l8_8152

-- Definitions based on the conditions.
def euler_totient_49 : ℕ := 42

theorem problem_remainder_6_pow_83_add_8_pow_83_mod_49 
  (h1 : 6 ^ euler_totient_49 ≡ 1 [MOD 49])
  (h2 : 8 ^ euler_totient_49 ≡ 1 [MOD 49]) :
  (6 ^ 83 + 8 ^ 83) % 49 = 35 :=
by
  sorry

end problem_remainder_6_pow_83_add_8_pow_83_mod_49_l8_8152


namespace num_possible_pairs_l8_8121

noncomputable def M (a d : ℝ) : matrix (fin 2) (fin 2) ℝ :=
  ![(a : ℝ), 4; -9, d]

theorem num_possible_pairs (a d : ℝ) (h : M a d ⬝ M a d = 1) : ∃ (N : ℕ), N = 2 :=
by sorry

end num_possible_pairs_l8_8121


namespace compute_expression_l8_8473

theorem compute_expression : 12 + 5 * (4 - 9)^2 - 3 = 134 := by
  sorry

end compute_expression_l8_8473


namespace option_C_same_function_l8_8133

-- Define the functions from Option C
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := 5 * x^5

-- State the proposition that Option C represents the same function
theorem option_C_same_function : ∀ (x : ℝ), f1 x = f2 x := 
begin
  -- The actual proof should go here, but we use sorry to skip the proof
  sorry
end

end option_C_same_function_l8_8133


namespace extra_flowers_l8_8427

theorem extra_flowers (tulips: ℕ) (roses: ℕ) (used: ℕ) (h_tulips: tulips = 39) (h_roses: roses = 49) (h_used: used = 81) :
    tulips + roses - used = 7 :=
by
  rw [h_tulips, h_roses, h_used]
  norm_num
  -- The proof is omitted
  sorry

end extra_flowers_l8_8427


namespace sum_faces_of_cube_l8_8704

theorem sum_faces_of_cube (p u q v r w : ℕ) (hp : 0 < p) (hu : 0 < u) (hq : 0 < q) (hv : 0 < v)
    (hr : 0 < r) (hw : 0 < w)
    (h_sum_vertices : p * q * r + p * v * r + p * q * w + p * v * w 
        + u * q * r + u * v * r + u * q * w + u * v * w = 2310) : 
    p + u + q + v + r + w = 40 := 
sorry

end sum_faces_of_cube_l8_8704


namespace rectangle_length_l8_8449

theorem rectangle_length (P w l : ℕ) (h₀ : P = 42) (h₁ : w = 4) : l = 17 :=
by
  -- P = 2 (l + w)
  have h₂ : 2 * (l + w) = P, from sorry,
  -- 42 = 2 (l + 4)
  have h₃ : 42 = 2 * (l + 4), from sorry,
  -- 21 = l + 4
  have h₄ : 21 = l + 4, from sorry,
  -- l = 21 - 4
  have h₅ : l = 17, from sorry,
  assumption

example : ∃ l, rectangle_length 42 4 l := ⟨17, rectangle_length 42 4 17⟩

end rectangle_length_l8_8449


namespace right_triangle_sin_sum_l8_8233

/--
In a right triangle ABC with ∠A = 90°, prove that sin A + sin^2 B + sin^2 C = 2.
-/
theorem right_triangle_sin_sum (A B C : ℝ) (hA : A = 90) (hABC : A + B + C = 180) :
  Real.sin (A * π / 180) + Real.sin (B * π / 180) ^ 2 + Real.sin (C * π / 180) ^ 2 = 2 :=
sorry

end right_triangle_sin_sum_l8_8233


namespace earnings_difference_l8_8747

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l8_8747


namespace calculate_expression_l8_8114

theorem calculate_expression : ((-1 + 2) * 3 + 2^2 / (-4)) = 2 :=
by
  sorry

end calculate_expression_l8_8114


namespace shaded_area_of_floor_l8_8431

theorem shaded_area_of_floor
  (floor_length : ℝ) (floor_width : ℝ)
  (tile_length : ℝ) (tile_width : ℝ)
  (circle_radius : ℝ)
  (total_tiles : ℕ)
  (total_shaded_area : ℝ)
  (h1 : floor_length = 16)
  (h2 : floor_width = 20)
  (h3 : tile_length = 2)
  (h4 : tile_width = 2)
  (h5 : circle_radius = 1)
  (h6 : total_tiles = (floor_length * floor_width / (tile_length * tile_width)).to_nat)
  (h7 : total_shaded_area = total_tiles * (tile_length * tile_width - π * circle_radius^2)) :
  total_shaded_area = 320 - 80 * π :=
by
  sorry

end shaded_area_of_floor_l8_8431


namespace rod_length_l8_8524

/--
Prove that given the number of pieces that can be cut from the rod is 40 and the length of each piece is 85 cm, the length of the rod is 3400 cm.
-/
theorem rod_length (number_of_pieces : ℕ) (length_of_each_piece : ℕ) (h_pieces : number_of_pieces = 40) (h_length_piece : length_of_each_piece = 85) : number_of_pieces * length_of_each_piece = 3400 := 
by
  -- We need to prove that 40 * 85 = 3400
  sorry

end rod_length_l8_8524


namespace p_lim_p20_minus_p15_l8_8708

noncomputable def p : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2) * p n + (1 / 2) * p (n - 1)

theorem p_lim (n : ℕ) : filter.tendsto p filter.at_top (nhds 1) :=
sorry

theorem p20_minus_p15 : p 20 - p 15 = 0 :=
begin
  have h_lim := p_lim,
  sorry -- Proof that p 20 - p 15 = 0
end

end p_lim_p20_minus_p15_l8_8708


namespace tan_half_odd_tan_half_monotonic_tan_half_symmetric_l8_8912

def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_half_odd : ∀ x : ℝ, tan_half (-x) = -tan_half x :=
by
  intro x
  sorry

theorem tan_half_monotonic : ∀ a b : ℝ, 0 < a → a < b → b < π / 2 → tan_half a < tan_half b :=
by
  intro a b ha hab hb
  sorry

theorem tan_half_symmetric : ∀ k : ℤ, tan_half (k * π) = 0 :=
by
  intro k
  sorry

end tan_half_odd_tan_half_monotonic_tan_half_symmetric_l8_8912


namespace angle_A_is_60_degrees_value_of_b_plus_c_l8_8601

noncomputable def triangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  let area := (3 * Real.sqrt 3) / 2
  c + 2 * a * Real.cos C = 2 * b ∧
  1/2 * b * c * Real.sin A = area 

theorem angle_A_is_60_degrees (A B C : ℝ) (a b c : ℝ) :
  triangleABC A B C a b c →
  Real.cos A = 1 / 2 → 
  A = 60 :=
by
  intros h1 h2 
  sorry

theorem value_of_b_plus_c (A B C : ℝ) (b c : ℝ) :
  triangleABC A B C (Real.sqrt 7) b c →
  b * c = 6 →
  (b + c) = 5 :=
by 
  intros h1 h2 
  sorry

end angle_A_is_60_degrees_value_of_b_plus_c_l8_8601


namespace minimum_value_arithmetic_sequence_geometric_property_l8_8540

theorem minimum_value_arithmetic_sequence_geometric_property :
  ∀ (a_n : ℕ → ℤ) (S : ℕ → ℤ),
  (∀ n, a_n = 1 + (n - 1) * 2) -- This defines the arithmetic sequence with a_1 = 1 and d = 2
  → (∀ n, S n = n * n) -- Sum of the first n terms
  → ∃ n ∈ (finset.range 1).succ, -- Ensure n is in positive natural numbers
      (2 * S n + 16) / (a_n + 3) = 4 := 
begin
  intros a_n S h_seq h_sum,
  sorry
end

end minimum_value_arithmetic_sequence_geometric_property_l8_8540


namespace cost_effective_l8_8042

variable {n : ℕ} (h_n : n > 1) 
variable {a : Fin n → ℝ} (h_a : ∀ i, a i > 0)

def x : ℝ := (∑ i, a i) / n
def y : ℝ := n / (∑ i, 1 / (a i))

theorem cost_effective (h_n : n > 1) (h_a : ∀ i, a i > 0) : x ≥ y := by
  sorry

end cost_effective_l8_8042


namespace sum_sqrt_ineq_l8_8687

theorem sum_sqrt_ineq {n : ℕ} : (∑ k in Finset.range (n^2 + 1), (Real.fract (Real.sqrt k))) ≤ (n^2 - 1) / 2 := 
by
  sorry

end sum_sqrt_ineq_l8_8687


namespace last_passenger_probability_last_passenger_probability_l8_8334

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l8_8334


namespace find_equation_of_ellipse_find_range_of_m_l8_8542

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), 
    (a > b) ∧ (b > 0) ∧ 
    Eccentricity a b (sqrt 3 / 2) ∧ 
    slope_AF sqrt 3 ∧ 
    Equation_E a b

noncomputable def range_m (P : ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  P 0 m ∧ 
  ∃ (x1 x2 : ℝ),
    intersects_l_E l E x1 x2 ∧ 
    vector_relation x1 x2 (lambda 3) ∧ 
    OM_ON_OP_relation 4

theorem find_equation_of_ellipse :
  ellipse_equation ↔ (∃ (a b : ℝ), (a = 2) ∧ (b = 1))

theorem find_range_of_m (P : ℝ → ℝ → Prop) :
  (∀ m, range_m P m) ↔ (-2 < m ∧ m < -1) ∨ (1 < m ∧ m < 2)

end find_equation_of_ellipse_find_range_of_m_l8_8542


namespace probability_Q_within_2_of_origin_eq_pi_div_9_l8_8076

noncomputable def probability_within_circle (π : ℝ) : ℝ :=
  let area_of_square := (2 * 3)^2
  let area_of_circle := π * 2^2
  area_of_circle / area_of_square

theorem probability_Q_within_2_of_origin_eq_pi_div_9 :
  probability_within_circle Real.pi = Real.pi / 9 :=
by
  sorry

end probability_Q_within_2_of_origin_eq_pi_div_9_l8_8076


namespace cindy_correct_answer_l8_8470

/-- 
Cindy accidentally first subtracted 9 from a number, then multiplied the result 
by 2 before dividing by 6, resulting in an answer of 36. 
Following these steps, she was actually supposed to subtract 12 from the 
number and then divide by 8. What would her answer have been had she worked the 
problem correctly?
-/
theorem cindy_correct_answer :
  ∀ (x : ℝ), (2 * (x - 9) / 6 = 36) → ((x - 12) / 8 = 13.125) :=
by
  intro x
  sorry

end cindy_correct_answer_l8_8470


namespace find_slope_of_chord_l8_8961

noncomputable def slope_of_chord (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem find_slope_of_chord :
  (∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 → ∃ (x1 x2 y1 y2 : ℝ),
    x1 + x2 = 8 ∧ y1 + y2 = 4 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ slope_of_chord x1 x2 y1 y2 = -1 / 2) := sorry

end find_slope_of_chord_l8_8961


namespace intersection_distance_l8_8613

-- Define the initial and final points in 3D space.
def startPoint : ℝ × ℝ × ℝ := (3, 1, 2)
def endPoint : ℝ × ℝ × ℝ := (-2, -4, -5)

-- Define the sphere's radius and center.
def sphereRadius : ℝ := 2
def sphereCenter : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the parameters p and q for final equation
def p : ℕ := 2
def q : ℕ := 154

-- The equation that needs proof.
theorem intersection_distance (p q : ℕ) (p = 2) (q = 154) : 
  p + q = 156 := 
  sorry

end intersection_distance_l8_8613


namespace product_of_sequence_l8_8880

theorem product_of_sequence :
  (∏ n in Finset.range 2024, (n + 3) / (n + 2)) = 1013 := by
  sorry

end product_of_sequence_l8_8880


namespace independence_of_A_and_B_l8_8375

variable (Ω : Type)
variable [Fintype Ω]
variable [DecidableEq Ω]

def event (E : Set Ω) := {ω | ω ∈ E}

def A : event Ω := {ω | ω.1 = "H"}
def B : event Ω := {ω | ω.2 = "T"}

theorem independence_of_A_and_B : 
  (∃ (Ω : Type) [Fintype Ω] [DecidableEq Ω], 
  (P (A ∩ B) = P A * P B)) :=
sorry

end independence_of_A_and_B_l8_8375


namespace cos_x_eq_range_f_B_l8_8990

-- Definition for Question 1 conditions
def vector_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
def vector_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
def f (x : ℝ) : ℝ := vector_m x.1 * vector_n x.1 + vector_m x.2 * vector_n x.2 + 1

-- Question 1 main statement
theorem cos_x_eq (x : ℝ) (h1 : x ∈ Icc 0 (Real.pi / 2)) (h2 : f x = 11 / 10) : 
  Real.cos x = (4 * Real.sqrt 3 - 3) / 10 := 
sorry

-- Definition for Question 2 conditions
variables {A B C : Angle}
variables {a b c : ℝ}
def side_opposite (θ : Angle) (∆ : Triangle) : ℝ :=
  match θ with
  | A => a
  | B => b
  | C => c

-- Question 2 main statement
theorem range_f_B 
  (hA : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  ∃ f_B_range : Set ℝ, f_B_range = Ioc 0 (1 / 2) :=
sorry

end cos_x_eq_range_f_B_l8_8990


namespace sum_of_cubes_of_roots_l8_8889

-- We define the original equation
def equation (x : ℝ) : Prop := x^(1/3) - 8 * x + 9 * x^(1/3) - 2 = 0

-- We assume all roots are real and nonnegative
axiom roots_are_real_nonnegative (x : ℝ) : equation x → 0 ≤ x

-- Proof statement
theorem sum_of_cubes_of_roots : ∀ (r s t : ℝ), equation r → equation s → equation t →
  r + s + t = -5/4 → r * s + r * t + s * t = 0 → r * s * t = 1/4 → r^3 + s^3 + t^3 = -77/64 :=
begin
  intros r s t hr hs ht h_sum h_rs h_rst,
  -- Proof is skipped
  sorry,
end

end sum_of_cubes_of_roots_l8_8889


namespace pizzas_needed_l8_8736

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end pizzas_needed_l8_8736


namespace solve_triangle_and_circle_problem_l8_8365

noncomputable def triangle_and_circle_problem : Prop :=
  let perimeter_AQM := 300
  let angle_QAM := 120 -- in degrees
  let radius_O := 30   -- radius of the circle
  let relatively_prime (a b : ℕ) : Prop := (Nat.gcd a b) = 1
  exists (p q : ℕ), p + q = 16 ∧ (relatively_prime p q) ∧ 
    let OQ := p / q
    OQ = 15

theorem solve_triangle_and_circle_problem : triangle_and_circle_problem :=
begin
  sorry
end

end solve_triangle_and_circle_problem_l8_8365


namespace find_a_find_inverse_function_l8_8668

section proof_problem

variables (f : ℝ → ℝ) (a : ℝ)

-- Condition that f is an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f (x)

-- Condition from the problem
def condition_1 : Prop := (a - 1) * (2 ^ (1:ℝ) + 1) = 0
def function_definition : Prop := f = λ x, 2^x - 1

-- Question 1: Find the value of a
theorem find_a
  (h1 : is_odd_function f)
  (h2 : condition_1 a)
  (h3 : function_definition f) : a = 1 :=
sorry

-- Question 2: Find the inverse function f⁻¹(x)
noncomputable def inverse_function (y : ℝ) : ℝ := real.log (y + 1) / real.log 2

theorem find_inverse_function
  (h3 : function_definition f) : 
  ∀ x, f (inverse_function x) = x :=
sorry

end proof_problem

end find_a_find_inverse_function_l8_8668


namespace part1_part2_l8_8677

noncomputable def f (a : ℝ) (x : ℝ) := (a * x - 1) * (x - 1)

theorem part1 (h : ∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) : a = 1/2 :=
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 1/a) ∨
  (a = 1 → ∀ x : ℝ, ¬(f a x < 0)) ∨
  (∀ x : ℝ, f a x < 0 ↔ 1/a < x ∧ x < 1) :=
  sorry

end part1_part2_l8_8677


namespace problem_statement_l8_8103

noncomputable def probability_no_distinct_real_roots_greater_than_2 : ℚ :=
  let pairs := [(a, c) | a ∈ (-6..6), c ∈ (-6..6)].toFinset.toList
  let valid_pairs := pairs.filter (λ (ac : ℤ × ℤ), let ⟨a, c⟩ := ac in
    ¬((9 * a^2 - 4 * a * c > 0) ∧ (a ≠ 0) ∧ (c > 4 * a)))
  (valid_pairs.length : ℚ) / (pairs.length : ℚ)

theorem problem_statement : probability_no_distinct_real_roots_greater_than_2 = 167 / 169 :=
  sorry

end problem_statement_l8_8103


namespace last_passenger_probability_last_passenger_probability_l8_8338

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l8_8338


namespace chord_length_polar_coordinates_l8_8243

theorem chord_length_polar_coordinates :
  ∀ (ρ θ : ℝ), ρ * Real.cos θ = 1 / 2 → ρ = 2 * Real.cos θ → 
  let x := ρ * Real.cos θ in
  let y := ρ * Real.sin θ in
  let circle_eq := x^2 + y^2 - 2 * x = 0 in
  let center := (1, 0) in
  let radius := 1 in
  let line_distance := 1 / 2 in
  let chord_length := 2 * Real.sqrt (1 - (1 / 4)) in
  chord_length = Real.sqrt 3 := 
sorry

end chord_length_polar_coordinates_l8_8243


namespace machines_complete_order_l8_8049

theorem machines_complete_order (h1 : ℝ) (h2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ)
  (h1_def : h1 = 9)
  (h2_def : h2 = 8)
  (rate1_def : rate1 = 1 / h1)
  (rate2_def : rate2 = 1 / h2)
  (combined_rate : ℝ := rate1 + rate2) :
  time = 72 / 17 :=
by
  sorry

end machines_complete_order_l8_8049


namespace smallest_X_divisible_by_60_l8_8651

/-
  Let \( T \) be a positive integer consisting solely of 0s and 1s.
  If \( X = \frac{T}{60} \) and \( X \) is an integer, prove that the smallest possible value of \( X \) is 185.
-/
theorem smallest_X_divisible_by_60 (T X : ℕ) 
  (hT_digit : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) 
  (h1 : X = T / 60) 
  (h2 : T % 60 = 0) : 
  X = 185 :=
sorry

end smallest_X_divisible_by_60_l8_8651


namespace select_p_elements_with_integer_mean_l8_8263

theorem select_p_elements_with_integer_mean {p : ℕ} (hp : Nat.Prime p) (p_odd : p % 2 = 1) :
  ∃ (M : Finset ℕ), (M.card = (p^2 + 1) / 2) ∧ ∃ (S : Finset ℕ), (S.card = p) ∧ ((S.sum id) % p = 0) :=
by
  -- sorry to skip the proof
  sorry

end select_p_elements_with_integer_mean_l8_8263


namespace distance_correct_l8_8148

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line equation
def line (x y : ℝ) : Prop := x - (sqrt 3) * y = 0

-- Define the focus of the parabola
def focus : ℝ×ℝ := (1/2, 0)

-- Define the formula for distance from a point to a line
def point_to_line_distance (p : ℝ×ℝ) (a b c : ℝ) : ℝ :=
  (|a * p.1 + b * p.2 + c|) / sqrt (a^2 + b^2)

-- Specific case for the given problem
def distance_from_focus_to_line : ℝ :=
  point_to_line_distance focus 1 (-sqrt 3) 0

-- The theorem to prove
theorem distance_correct :
  distance_from_focus_to_line = 1/4 :=
  sorry

end distance_correct_l8_8148


namespace second_crane_height_l8_8903

noncomputable def height_of_second_crane : ℝ :=
  let crane1 := 228
  let building1 := 200
  let building2 := 100
  let crane3 := 147
  let building3 := 140
  let avg_building_height := (building1 + building2 + building3) / 3
  let avg_crane_height := avg_building_height * 1.13
  let h := (avg_crane_height * 3) - (crane1 - building1 + crane3 - building3) + building2
  h

theorem second_crane_height : height_of_second_crane = 122 := 
  sorry

end second_crane_height_l8_8903


namespace numbers_divisible_l8_8208

theorem numbers_divisible (n : ℕ) (d1 d2 : ℕ) (lcm_d1_d2 : ℕ) (limit : ℕ) (h_lcm: lcm d1 d2 = lcm_d1_d2) (h_limit : limit = 2011)
(h_d1 : d1 = 117) (h_d2 : d2 = 2) : 
  ∃ k : ℕ, k = 8 ∧ ∀ m : ℕ, m < limit → (m % lcm_d1_d2 = 0 ↔ ∃ i : ℕ, i < k ∧ m = lcm_d1_d2 * (i + 1)) :=
by
  sorry

end numbers_divisible_l8_8208


namespace sum_first_eight_terms_l8_8187

-- Define the arithmetic sequence and its sum.
def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) := a_1 + (n - 1) * d

def sum_arithmetic_sequence (a_1 d : ℝ) (n : ℕ) := (n / 2) * (2 * a_1 + (n - 1) * d)

-- Define the conditions from the problem.
def condition (a_1 d : ℝ) : Prop :=
  a_1 + d = 18 - (a_1 + 6 * d)

-- The theorem to be proven: under these conditions, the sum of the first 8 terms is 72.
theorem sum_first_eight_terms (a_1 d : ℝ) (h : condition a_1 d) : sum_arithmetic_sequence a_1 d 8 = 72 :=
by
  sorry

end sum_first_eight_terms_l8_8187


namespace remaining_distance_of_second_pedestrian_l8_8382

-- Definitions based on the conditions in the problem
variables (x : ℝ) -- distance between points A and B
variables (y : ℝ) -- ratio of the speeds of the first pedestrian to the second pedestrian

-- Conditions given in the problem
variables (hx1 : y = (x / 2) / (x - 24)) -- first condition
variables (hx2 : y = (x - 15) / (x / 2)) -- second condition

-- The theorem we need to prove
theorem remaining_distance_of_second_pedestrian 
  (hx1 : y = (x / 2) / (x - 24))
  (hx2 : y = (x - 15) / (x / 2))
  : let S2 := x * (2 / 3) / y + x * (1 / 3) in
    x - S2 = 8 :=
by 
  -- This is where the proof would go
  sorry

end remaining_distance_of_second_pedestrian_l8_8382


namespace largest_multiple_of_8_less_than_100_l8_8788

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8788


namespace smallest_angle_solution_l8_8887

theorem smallest_angle_solution (y : ℝ) (deg_y : ℝ) (h: y = deg_y * (π / 180) ∧ 6 * sin y * cos^3 y - 6 * sin^3 y * cos y = 1) : deg_y = 10.4525 :=
sorry

end smallest_angle_solution_l8_8887


namespace largest_multiple_of_8_less_than_100_l8_8777

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8777


namespace yarn_total_length_l8_8831

theorem yarn_total_length (parts: ℕ) (parts_used: ℕ) (total_used: ℕ) 
  (equal_parts: parts = 5) (used_parts: parts_used = 3) (used_total: total_used = 6) : 
  let part_length := total_used / parts_used in
  let total_length := part_length * parts in
  total_length = 10 :=
by {
  sorry
}

end yarn_total_length_l8_8831


namespace find_theta_perpendicular_l8_8570

noncomputable def l1_perpendicular_l2 (θ : ℝ) : Prop :=
  let l1 := λ x y : ℝ, x * Real.sin θ - y * Real.cos θ = 1
  let l2 := λ x y : ℝ, √3 * x + y - 1 = 0
  let slope_l2 := -√3
  let slope_l1 := Real.tan θ
  slope_l1 * slope_l2 = -1

theorem find_theta_perpendicular : 
    ∃ θ : ℝ, l1_perpendicular_l2 θ ∧ θ = π / 6 :=
by
  sorry

end find_theta_perpendicular_l8_8570


namespace value_of_y_l8_8753

variable {x y : ℝ}

theorem value_of_y (h1 : x > 2) (h2 : y > 2) (h3 : 1/x + 1/y = 3/4) (h4 : x * y = 8) : y = 4 :=
sorry

end value_of_y_l8_8753


namespace number_of_trees_after_cutting_and_planting_l8_8096

theorem number_of_trees_after_cutting_and_planting 
    (initial_trees : ℕ) (cut_percentage : ℝ) (trees_planted_per_cut : ℕ) : 
    initial_trees = 400 →
    cut_percentage = 0.20 →
    trees_planted_per_cut = 5 →
    ∃ final_trees : ℕ, final_trees = 720 :=
by
  intros initial_trees_eq cut_percentage_eq trees_planted_per_cut_eq
  
  let trees_cut := (cut_percentage * initial_trees.to_real).nat_abs
  have trees_cut_eq : trees_cut = 80,
  { sorry },

  let remaining_trees := initial_trees - trees_cut
  have remaining_trees_eq : remaining_trees = 320,
  { sorry },

  let new_trees_planted := trees_cut * trees_planted_per_cut
  have new_trees_planted_eq : new_trees_planted = 400,
  { sorry },

  let final_trees := remaining_trees + new_trees_planted
  have final_trees_eq : final_trees = 720,
  { sorry },

  use final_trees
  exact final_trees_eq

end number_of_trees_after_cutting_and_planting_l8_8096


namespace probability_three_dice_prime_l8_8457

noncomputable def probability_three_prime (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_three_dice_prime : 
  probability_three_prime 6 3 (3 / 6) = 5 / 16 := by
  sorry

end probability_three_dice_prime_l8_8457


namespace speed_of_car_in_second_hour_l8_8725

-- Given conditions
def speed_first_hour : ℝ := 120
def average_speed : ℝ := 95
def total_time : ℝ := 2

-- Definition of total distance
def total_distance (speed_second_hour : ℝ) : ℝ :=
  speed_first_hour + speed_second_hour

-- Definition of average speed
def avg_speed (speed_second_hour : ℝ) : ℝ :=
  total_distance(speed_second_hour) / total_time

-- Main theorem to prove
theorem speed_of_car_in_second_hour (speed_second_hour : ℝ) : 
  avg_speed(speed_second_hour) = average_speed → speed_second_hour = 70 := 
by
  sorry

end speed_of_car_in_second_hour_l8_8725


namespace linear_function_parallel_intersection_l8_8219

theorem linear_function_parallel_intersection (b : ℝ) :
  (∀ x : ℝ, (-3 : ℝ) * x + b = 3 * x + 5 → b = 5) → 
  (∀ k : ℝ, k = -3 → (y : ℝ → ℝ) (x) = k * x + b) → 
  (∀ y : ℝ → ℝ, ∃ x : ℝ, y = -3 * x + 5) :=
by
  sorry

end linear_function_parallel_intersection_l8_8219


namespace problem1_problem2_l8_8196

def f (x: ℝ) : ℝ := (Real.log (1 + x)) / x

theorem problem1 (x : ℝ) (hx : x > 0) : f(x) > (2 / (x + 2)) :=
by sorry

theorem problem2 : ∃ k : ℝ, k = 0.5 ∧ ∀ x : ℝ, -1 < x ∧ x ≠ 0 → f(x) < ((1 + k * x) / (1 + x)) :=
by sorry

end problem1_problem2_l8_8196


namespace platform_length_l8_8432

theorem platform_length
  (train_length : ℤ)
  (speed_kmph : ℤ)
  (time_sec : ℤ)
  (speed_mps : speed_kmph * 1000 / 3600 = 20)
  (distance_eq : (train_length + 220) = (20 * time_sec))
  (train_length_val : train_length = 180)
  (time_sec_val : time_sec = 20) :
  220 = 220 := by
  sorry

end platform_length_l8_8432


namespace largest_multiple_of_8_less_than_100_l8_8778

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), (n % 8 = 0) ∧ (n < 100) ∧ (∀ m : ℕ, (m % 8 = 0) ∧ (m < 100) → m ≤ n) :=
begin
  use 96,
  split,
  { -- 96 is a multiple of 8
    exact nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 96),
  },
  split,
  { -- 96 is less than 100
    norm_num,
  },
  { -- 96 is the largest multiple of 8 less than 100
    intros m hm,
    obtain ⟨k, rfl⟩ := (nat.dvd_iff_mod_eq_zero.mp hm.1),
    have : k ≤ 12, by linarith,
    linarith [mul_le_mul (zero_le _ : (0 : ℕ) ≤ 8) this (zero_le _ : (0 : ℕ) ≤ 12) (zero_le _ : (0 : ℕ) ≤ 8)],
  },
end

end largest_multiple_of_8_less_than_100_l8_8778


namespace volume_of_triangular_pyramid_l8_8858

-- Define the conditions from the problem
variables {a b c : ℝ}
variables (h1 : a * b = 3) (h2 : b * c = 4) (h3 : a * c = 12)

-- Define the requirement to prove
theorem volume_of_triangular_pyramid (h_perpendicular : true) : 
  ∃ (a b c : ℝ), a * b * c = 12 → (1 / 6) * a * b * c = 2 :=
begin
  -- Since h_perpendicular is just a placeholder for the geometric condition,
  -- it is not used directly in the computation.
  use [a, b, c],
  intro h_abc,
  rw h_abc,
  norm_num,
end

end volume_of_triangular_pyramid_l8_8858


namespace correct_quotient_l8_8415

theorem correct_quotient (incorrect_divisor incorrect_quotient correct_divisor : ℕ) (h1: incorrect_divisor = 48) 
(h2: incorrect_quotient = 24) (h3: correct_divisor = 36) : 
(incorrect_divisor * incorrect_quotient) / correct_divisor = 32 :=
by
  -- Assumptions based on the conditions
  have h4: incorrect_divisor * incorrect_quotient = 48 * 24, from by rw [h1, h2],
  -- Calculate product of incorrect_divisor and incorrect_quotient
  have h5: 48 * 24 = 1152, from by norm_num,
  -- Replace product in the theorem statement using previously calculated value
  have h6: (48 * 24) / 36 = 1152 / 36, from by rw [h5],
  -- Calculate the correct quotient
  have h7: 1152 / 36 = 32, from by norm_num,
  -- Conclude that the statement holds by combining the previous results
  exact eq.trans (by rw [h4, h6]) h7

#check correct_quotient

end correct_quotient_l8_8415


namespace smallest_positive_value_of_sum_l8_8483

-- Define the variables
variables {a : Fin 100 → Int}

-- Define the conditions and the theorem
theorem smallest_positive_value_of_sum (h1 : ∀ i : Fin 100, a i = 1 ∨ a i = -1)
    (h2 : ∃ p q : ℕ, p + q = 100 ∧ ∀ i : Fin 100, a i = 1 → p ∧ a i = -1 → q)
    (h3 : (∑ i : Fin 100, a i) ^ 2 = 100 + 2 * (T : ℤ)) :
  let sum := ∑ i in Finset.range 100, ∑ j in Finset.range i, a (Fin.ofNat j) * a (Fin.ofNat i) in
  sum = 22 :=
by
  sorry

end smallest_positive_value_of_sum_l8_8483


namespace carrie_bought_tshirts_l8_8884

/-- 
Carrie likes to buy t-shirts at the local clothing store. They cost $9.15 each. 
One day, she bought some t-shirts and spent $201. How many t-shirts did she buy? 
-/
theorem carrie_bought_tshirts (cost_per_tshirt : ℝ) (total_spent : ℝ) (h_cost : cost_per_tshirt = 9.15) (h_total : total_spent = 201) : 
  ⌊total_spent / cost_per_tshirt⌋ = 21 := 
  by {
    rw [h_cost, h_total],
    norm_num,
    -- skip the proof with sorry
    sorry
  }

end carrie_bought_tshirts_l8_8884


namespace f_at_2_l8_8962

noncomputable def f (x : ℝ) : ℝ := 2 * x * (f' 2) + Real.log (x - 1)
noncomputable def f' (x : ℝ) : ℝ := 2 * (f' 2) + 1 / (x - 1)

theorem f_at_2 : f 2 = -4 :=
by
  -- the proof goes here
  sorry

end f_at_2_l8_8962


namespace fraction_under_11_is_one_third_l8_8288

def fraction_under_11 (T : ℕ) (fraction_above_11_under_13 : ℚ) (students_above_13 : ℕ) : ℚ :=
  let fraction_under_11 := 1 - (fraction_above_11_under_13 + students_above_13 / T)
  fraction_under_11

theorem fraction_under_11_is_one_third :
  fraction_under_11 45 (2/5) 12 = 1/3 :=
by
  sorry

end fraction_under_11_is_one_third_l8_8288


namespace tangent_lines_between_circles_l8_8294

theorem tangent_lines_between_circles
  (r1 r2 d : ℝ)
  (hr1 : r1 = 5) (hr2 : r2 = 8)
  (hd : d = 13) :
  ∃ m : ℕ, m = 3 :=
by
  use 3
  sorry

end tangent_lines_between_circles_l8_8294


namespace prob_point_closer_to_4_than_0_is_0_l8_8077

noncomputable def probability_closer_to_4_than_0 : ℝ := 
  let favorable_length := (5 - 2)      -- Length of the interval where points are closer to 4 than 0
  let total_length := (5 - 0)           -- Total length of the interval [0,5]
  (favorable_length / total_length)

theorem prob_point_closer_to_4_than_0_is_0.6 :
  probability_closer_to_4_than_0 = 0.6 := by
  sorry

end prob_point_closer_to_4_than_0_is_0_l8_8077


namespace last_passenger_probability_last_passenger_probability_l8_8335

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l8_8335


namespace tangent_CF_l8_8241

variables {α : Type*} [euclidean_geometry α]

/-- All given conditions and the task to prove CF is tangent to the circumcircle -/
theorem tangent_CF (A B C D E F : α)
  (hABC : ∠A + ∠B + ∠C = 180)
  (hABC_acute : ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90)
  (hCD_perp_AB : CD ⊥ AB)
  (hCD_perp_AT_D : ∃ D : α, bc_on_line A B D)
  (hE_on_CD_bisector : E ∈ CD)
  (hF_on_circum_ADE : F ∈ circumcircle (A, D, E))
  (hADF : ∠ADF = 45) 
  : CF ∈ tangent_of_circle (circumcircle (A, D, E)) := 
sorry

end tangent_CF_l8_8241


namespace distance_between_A1_and_A12_l8_8292

-- Define the initial setup conditions
constant A : ℕ → ℝ
axiom A1_0 : A 1 = 0 -- A1 is point O
axiom A2_1 : A 2 = 1 -- A2 is point P, with A1A2 = 1

-- Define the midpoint property for all relevant points
axiom midpoint_property (n : ℕ) : A n = (A (n + 1) + A (n + 2)) / 2

-- Define the theorem we want to prove
theorem distance_between_A1_and_A12 : abs (A 12 - A 1) = 683 :=
by
  sorry

end distance_between_A1_and_A12_l8_8292


namespace distance_between_points_A_and_B_is_122_l8_8051

-- Define the parameters
variables (d : ℝ) (v₁ v₂ v₃ : ℝ) (x : ℝ) (s_p : ℝ) (s_e : ℝ) (t : ℝ)

-- Initial speeds and condition constants
def passenger_initial_speed := 30 -- km/h
def express_speed := 60 -- km/h
def reduced_passenger_speed := passenger_initial_speed / 2 -- 15 km/h
def breakdown_fraction := 2/3
def catchup_distance := 27 -- km

-- Distance between points A and B
def distance_AB := d

-- Distance travelled before breakdown
def distance_before_breakdown := breakdown_fraction * d

-- Distance remaining after breakdown
def remaining_distance := d - distance_before_breakdown

-- Adjusted speed after breakdown
def passenger_speed_after_reduction := reduced_passenger_speed

-- Previous remaining distance before breakdown (remaining distance before failure)
def remaining_distance_in_passenger_speed := remaining_distance - catchup_distance

-- Proof statement
theorem distance_between_points_A_and_B_is_122:
    ∀ (d : ℝ),
    (passenger_initial_speed * remaining_distance_in_passenger_speed) = (distance_AB * (1 - catchup_distance / d)) →
    (passenger_initial_speed * distance_before_breakdown + passenger_speed_after_reduction * remaining_distance_in_passenger_speed) = remaining_distance_in_passenger_speed * express_speed →
    d = 122 := 
by
  intros,
  sorry

end distance_between_points_A_and_B_is_122_l8_8051


namespace lumberjack_trees_chopped_l8_8838

-- Statement of the problem in Lean 4
theorem lumberjack_trees_chopped
  (logs_per_tree : ℕ) 
  (firewood_per_log : ℕ) 
  (total_firewood : ℕ) 
  (logs_per_tree_eq : logs_per_tree = 4) 
  (firewood_per_log_eq : firewood_per_log = 5) 
  (total_firewood_eq : total_firewood = 500)
  : (total_firewood / firewood_per_log) / logs_per_tree = 25 := 
by
  rw [total_firewood_eq, firewood_per_log_eq, logs_per_tree_eq]
  norm_num
  sorry

end lumberjack_trees_chopped_l8_8838


namespace well_depth_l8_8085

variable (d : ℝ)

-- Conditions
def total_time (t₁ t₂ : ℝ) : Prop := t₁ + t₂ = 8.5
def stone_fall (t₁ : ℝ) : Prop := d = 16 * t₁^2 
def sound_travel (t₂ : ℝ) : Prop := t₂ = d / 1100

theorem well_depth : 
  ∃ t₁ t₂ : ℝ, total_time t₁ t₂ ∧ stone_fall d t₁ ∧ sound_travel d t₂ → d = 918.09 := 
by
  sorry

end well_depth_l8_8085


namespace geese_flock_size_l8_8834

theorem geese_flock_size : 
  ∃ x : ℕ, x + x + (x / 2) + (x / 4) + 1 = 100 ∧ x = 36 := 
by
  sorry

end geese_flock_size_l8_8834


namespace solve_for_x_l8_8519

theorem solve_for_x (x : ℝ) (h : 25^(-3) = 5^(72 / x) / (5^(42 / x) * 25^(30 / x))) : x = 5 :=
by
  sorry

end solve_for_x_l8_8519


namespace total_candies_correct_l8_8115

-- Define the number of candies each has
def caleb_jellybeans := 3 * 12
def caleb_chocolate_bars := 5
def caleb_gummy_bears := 8
def caleb_total := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears

def sophie_jellybeans := (caleb_jellybeans / 2)
def sophie_chocolate_bars := 3
def sophie_gummy_bears := 12
def sophie_total := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears

def max_jellybeans := (2 * 12) + sophie_jellybeans
def max_chocolate_bars := 6
def max_gummy_bears := 10
def max_total := max_jellybeans + max_chocolate_bars + max_gummy_bears

-- Define the total number of candies
def total_candies := caleb_total + sophie_total + max_total

-- Theorem statement
theorem total_candies_correct : total_candies = 140 := by
  sorry

end total_candies_correct_l8_8115


namespace find_n_l8_8897

def sequence (t : ℕ → ℚ) : Prop :=
  t 1 = 1 ∧ 
  (∀ n : ℕ, n > 1 → even n → t n = 1 + t (n / 2)) ∧
  (∀ n : ℕ, n > 1 → odd n → t n = 1 / t (n - 1))

theorem find_n
  (t : ℕ → ℚ)
  (hseq : sequence t)
  (ht : t 1905 = 19 / 87) :
  n = 1905 :=
sorry

end find_n_l8_8897


namespace BD_perpendicular_AM_l8_8247

-- Define the given conditions
variables (A B C D M : Point)
variables (h1 : Trapezium A B C D)
variables (h2 : Parallel AD BC)
variables (h3 : OnLineSegment M C D)
variables (h4 : Ratio CM MD = 2 / 3)
variables (h5 : EqLength AB AD)
variables (h6 : Ratio BC AD = 1 / 3)

-- State the theorem to prove
theorem BD_perpendicular_AM : Perpendicular BD AM :=
by
  sorry

end BD_perpendicular_AM_l8_8247


namespace area_under_cos_3_over_2_pi_l8_8711

theorem area_under_cos_3_over_2_pi : 
  ∫ x in 0..(3 * Real.pi / 2), abs (Real.cos x) = 3 := 
by 
  -- Proof omitted; here we only state the theorem.
  sorry

end area_under_cos_3_over_2_pi_l8_8711


namespace painting_time_l8_8875

noncomputable def bob_rate : ℕ := 120 / 8
noncomputable def alice_rate : ℕ := 150 / 10
noncomputable def combined_rate : ℕ := bob_rate + alice_rate
noncomputable def total_area : ℕ := 120 + 150
noncomputable def working_time : ℕ := total_area / combined_rate
noncomputable def lunch_break : ℕ := 1
noncomputable def total_time : ℕ := working_time + lunch_break

theorem painting_time : total_time = 10 := by
  -- Proof skipped
  sorry

end painting_time_l8_8875


namespace inequality_A_inequality_B_inequality_C_inequality_D_l8_8398

theorem inequality_A (x : ℝ) (hx : x > 0) : ln x ≥ 1 - 1 / x :=
sorry

theorem inequality_B (x : ℝ) (hx : x > 0) : ¬ (sin (2 * x) < x) :=
sorry

theorem inequality_C : (1 + tan (Real.pi / 12)) / (1 - tan (Real.pi / 12)) > Real.pi / 3 :=
sorry

theorem inequality_D (x : ℝ) (hx : x > 0) : Real.exp x > 2 * Real.sin x :=
sorry

end inequality_A_inequality_B_inequality_C_inequality_D_l8_8398


namespace weight_of_pipe_approx_l8_8045

-- Defining the conditions
def length_of_pipe := 21 -- cm
def external_diameter := 8 -- cm
def thickness_of_pipe := 1 -- cm
def density_of_iron := 8 -- g/cm³

def weight_of_pipe : ℝ :=
  let r_ext := external_diameter / 2 in -- external radius is half of external diameter
  let r_int := r_ext - thickness_of_pipe in -- internal radius is external radius minus thickness
  let V_ext := Real.pi * r_ext^2 * length_of_pipe in -- external volume
  let V_int := Real.pi * r_int^2 * length_of_pipe in -- internal volume
  let V_iron := V_ext - V_int in -- volume of iron
  V_iron * density_of_iron -- weight of the pipe

-- Statement to prove that the weight of the pipe is approximately 3696.47 grams
theorem weight_of_pipe_approx : |weight_of_pipe - 3696.47| < 0.01 := by
  sorry

end weight_of_pipe_approx_l8_8045


namespace vector_addition_example_l8_8548

noncomputable def OA : ℝ × ℝ := (-2, 3)
noncomputable def AB : ℝ × ℝ := (-1, -4)
noncomputable def OB : ℝ × ℝ := (OA.1 + AB.1, OA.2 + AB.2)

theorem vector_addition_example :
  OB = (-3, -1) :=
by
  sorry

end vector_addition_example_l8_8548


namespace min_value_l8_8965

open Real

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : log 2 a + log 2 b = 3) :
  ∃ min : ℝ, min = (1 / a + 1 / b) ∧ min = sqrt 2 / 2 := by
  sorry

end min_value_l8_8965


namespace minimum_value_condition_l8_8949

theorem minimum_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) 
                                (h_line : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1) 
                                (h_chord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ (x1 + 3)^2 + (y1 + 1)^2 = 1 ∧
                                           m * x2 + n * y2 + 2 = 0 ∧ (x2 + 3)^2 + (y2 + 1)^2 = 1 ∧
                                           (x1 - x2)^2 + (y1 - y2)^2 = 4) 
                                (h_relation : 3 * m + n = 2) : 
    ∃ (C : ℝ), C = 6 ∧ (C = (1 / m + 3 / n)) := 
by
  sorry

end minimum_value_condition_l8_8949


namespace sqrt_11_bounds_l8_8598

theorem sqrt_11_bounds : ∃ a : ℤ, a < Real.sqrt 11 ∧ Real.sqrt 11 < a + 1 ∧ a = 3 := 
by
  sorry

end sqrt_11_bounds_l8_8598


namespace isosceles_trapezoid_x_squared_l8_8648

-- Define the isosceles trapezoid structure
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD AD BC : ℝ)
  (hAB : AB = 122)
  (hCD : CD = 26)
  (hAD_BC : AD = BC)

-- Define the proof of the value of x^2
theorem isosceles_trapezoid_x_squared
  (A B C D : Type)
  (h : IsoscelesTrapezoid A B C D)
  (center_on_AB : ∃ M : Type, M = (h.AB / 2))
  (tangent_points_PQ : ∃ (P Q : Type), P = tangent_to_circle_on h.AD
                           ∧ Q = tangent_to_circle_on h.BC)
  (tangent_line_L : ∃ T : Type, midpoint h.CD ∈ L ∧ T ∈ circle ∧ is_tangent L T) :
  let x := h.AD in
  x^2 = 2135 :=
begin
  -- Proof omitted
  sorry,
end

end isosceles_trapezoid_x_squared_l8_8648


namespace common_remainder_proof_l8_8032

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end common_remainder_proof_l8_8032


namespace greatest_number_of_consecutive_integers_whose_sum_is_36_l8_8026

/-- 
Given that the sum of N consecutive integers starting from a is 36, 
prove that the greatest possible value of N is 72.
-/
theorem greatest_number_of_consecutive_integers_whose_sum_is_36 :
  ∀ (N a : ℤ), (N > 0) → (N * (2 * a + N - 1)) = 72 → N ≤ 72 := 
by
  intros N a hN h
  sorry

end greatest_number_of_consecutive_integers_whose_sum_is_36_l8_8026


namespace proof_problem_l8_8307

variable (μ σ p : ℝ)

-- Given: X ~ N(μ, σ^2), P(X ≤ 2) = 0.5
axiom X_normal : ∀ X : ℝ, X ~ ℕormal(μ, σ^2)
axiom P_X_le_2 : P (λ X, X ≤ 2) = 0.5

-- Given: Y ~ B(3, p), E(Y) = E(X)
axiom Y_binomial : ∀ Y : ℕ, Y ~ Binomial(3, p)
axiom E_Y_eq_E_X : E Y = E (λ X, X)

-- Statements to prove:
theorem proof_problem : μ = 2 ∧ p = 2/3 ∧ D (3 * Y) = 6 :=
by
  sorry

end proof_problem_l8_8307


namespace fish_population_estimate_l8_8438

-- Definition for tagged fish and total fish count estimation.
def tagged_fish (initial_catch : ℕ) (released_back : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  if second_catch = 0 then 0 else (initial_catch * second_catch) / tagged_in_second

-- Claim to prove
theorem fish_population_estimate :
  tagged_fish 30 30 40 2 = 600 :=
by
  have h1 : tagged_fish 30 30 40 2 = (30 * 40) / 2 := rfl
  have h2 : (30 * 40) / 2 = 600 := by norm_num
  rw [h1, h2]
  rfl

end fish_population_estimate_l8_8438


namespace no_geometric_sequence_l8_8675

-- Define an arithmetic sequence and its conditions
structure ArithSeq (α : Type*) [LinearOrderedField α] :=
(a : ℕ → α)
(common_difference : α)

-- Specify the conditions given in the problem
variables {α : Type*} [LinearOrderedField α]
noncomputable def contains1AndSqrt2 (seq : ArithSeq α) :=
  ∃ k l : ℕ, seq.a k = 1 ∧ seq.a l = real.sqrt 2

-- The theorem statement
theorem no_geometric_sequence (seq : ArithSeq α) (h : contains1AndSqrt2 seq) :
  ∀ m n p : ℕ, seq.a n ^ 2 ≠ seq.a m * seq.a p :=
sorry

end no_geometric_sequence_l8_8675


namespace at_least_11_distinct_remainders_l8_8419

theorem at_least_11_distinct_remainders :
  ∀ (A : Fin 100 → ℕ),
    (∀ i, A i ∈ Finset.range 100) →
    Function.Bijective A →
    let B := λ n, ∑ i in Finset.range (n + 1), A i in
      Finset.card (Finset.image (λ n, (B n % 100)) (Finset.range 100)) ≥ 11 := 
by 
  sorry

end at_least_11_distinct_remainders_l8_8419


namespace problem_1_problem_2_problem_3_problem_4_l8_8879

theorem problem_1 : (1 * -2.48) + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem problem_2 : 2 * (23 / 6 : ℚ) + - (36 / 7 : ℚ) + - (13 / 6 : ℚ) + - (230 / 7 : ℚ) = -(36 + 1 / 3 : ℚ) := by
  sorry

theorem problem_3 : (4 / 5 : ℚ) - (5 / 6 : ℚ) - (3 / 5 : ℚ) + (1 / 6 : ℚ) = - (7 / 15 : ℚ) := by
  sorry

theorem problem_4 : (-1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3) ^ 2) = 1 / 6 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l8_8879


namespace terez_pregnant_female_cows_l8_8321

def total_cows : ℕ := 44
def percent_female : ℚ := 0.50
def percent_pregnant_female : ℚ := 0.50
def female_cows : ℕ := (total_cows * percent_female).toNat
def pregnant_female_cows : ℕ := (female_cows * percent_pregnant_female).toNat

theorem terez_pregnant_female_cows : pregnant_female_cows = 11 := 
by
  sorry

end terez_pregnant_female_cows_l8_8321


namespace earnings_difference_l8_8745

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l8_8745


namespace probability_red_next_ball_l8_8116

-- Definitions of initial conditions
def initial_red_balls : ℕ := 50
def initial_blue_balls : ℕ := 50
def initial_yellow_balls : ℕ := 30
def total_pulled_balls : ℕ := 65

-- Condition that Calvin pulled out 5 more red balls than blue balls
def red_balls_pulled (blue_balls_pulled : ℕ) : ℕ := blue_balls_pulled + 5

-- Compute the remaining balls
def remaining_balls (blue_balls_pulled : ℕ) : Prop :=
  let remaining_red_balls := initial_red_balls - red_balls_pulled blue_balls_pulled
  let remaining_blue_balls := initial_blue_balls - blue_balls_pulled
  let remaining_yellow_balls := initial_yellow_balls - (total_pulled_balls - red_balls_pulled blue_balls_pulled - blue_balls_pulled)
  (remaining_red_balls + remaining_blue_balls + remaining_yellow_balls) = 15

-- Main theorem to be proven
theorem probability_red_next_ball (blue_balls_pulled : ℕ) (h : remaining_balls blue_balls_pulled) :
  (initial_red_balls - red_balls_pulled blue_balls_pulled) / 15 = 9 / 26 :=
sorry

end probability_red_next_ball_l8_8116


namespace value_of_v1_at_neg2_l8_8384

noncomputable def f (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x) * x + 1) * x + 0.3) * x + 2

theorem value_of_v1_at_neg2 : 
  let v1 := 1 * (-2) - 5 in 
  v1 = -7 :=
by 
  sorry

end value_of_v1_at_neg2_l8_8384


namespace adam_points_per_round_l8_8402

noncomputable def points_per_round (total_points : ℕ) (number_of_rounds : ℕ) : ℕ :=
  (total_points : ℤ) / (number_of_rounds : ℤ)

theorem adam_points_per_round :
  points_per_round 283 4 = 71 :=
by
  sorry

end adam_points_per_round_l8_8402


namespace eleven_hash_five_l8_8929

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five_l8_8929


namespace smallest_palindrome_in_base_3_and_base_5_l8_8119

def is_palindrome {α : Type*} [decidable_eq α] (l : list α) : Prop :=
l = l.reverse

def base_3_representation (n : ℕ) : list ℕ := sorry -- implement base 3 conversion
def base_5_representation (n : ℕ) : list ℕ := sorry -- implement base 5 conversion

theorem smallest_palindrome_in_base_3_and_base_5 :
  ∃ n : ℕ, n > 15 ∧ is_palindrome (base_3_representation n) ∧ is_palindrome (base_5_representation n) ∧
  (∀ m : ℕ, m > 15 ∧ is_palindrome (base_3_representation m) ∧ is_palindrome (base_5_representation m) → n ≤ m) ∧ n = 26 :=
sorry

end smallest_palindrome_in_base_3_and_base_5_l8_8119


namespace solve_for_n_l8_8592

-- Problem statement:
theorem solve_for_n :
    ∀ n : ℕ, (2^n = 8^20) → (8 = 2^3) → n = 60 :=
begin
    intros n h₁ h₂,
    sorry
end

end solve_for_n_l8_8592


namespace sum_x_coordinates_eq_3_l8_8353

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end sum_x_coordinates_eq_3_l8_8353


namespace final_value_after_determinant_and_addition_l8_8895

theorem final_value_after_determinant_and_addition :
  let a := 5
  let b := 7
  let c := 3
  let d := 4
  let det := a * d - b * c
  det + 3 = 2 :=
by
  sorry

end final_value_after_determinant_and_addition_l8_8895


namespace intersection_of_equilateral_triangles_forms_equilateral_l8_8731

theorem intersection_of_equilateral_triangles_forms_equilateral
  (O : Point) 
  (ABC A'B'C' : Triangle)
  (h1 : Equilateral ABC)
  (h2 : Equilateral A'B'C')
  (h3 : Inscribed ABC O)
  (h4 : Inscribed A'B'C' O)
  (A1 B1 C1 : Point)
  (h5 : IntersectionOfSides ABC A'B'C' A1 B1 C1) :
  Equilateral (Triangle.mk A1 B1 C1) := 
begin
  sorry
end

end intersection_of_equilateral_triangles_forms_equilateral_l8_8731


namespace largest_multiple_of_8_less_than_100_l8_8766

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8766


namespace igor_arrangement_l8_8223

theorem igor_arrangement : 
  let n := 7 in 
  let k := 3 in 
  ∃ (ways : ℕ), ways = (n.factorial * (nat.choose (n-1) (k-1))) ∧ ways = 75600 := 
by
  let n := 7
  let k := 3
  use (n.factorial * (nat.choose (n-1) (k-1)))
  split
  · rfl
  · sorry

end igor_arrangement_l8_8223


namespace angle_A_value_side_a_value_l8_8224

-- Given conditions
variable {a b c A B C : ℝ}
variable (h_triangle : a * sin B = sqrt 3 * b * cos A)
variable (h_area : 1/2 * b * c * sin A = sqrt 3)
variable (h_perimeter : a + b + c = 6)

-- Prove
theorem angle_A_value (h_triangle : a * sin B = sqrt 3 * b * cos A) :
  A = π / 3 :=
by
  sorry

theorem side_a_value (h_triangle : a * sin B = sqrt 3 * b * cos A)
    (h_area : 1/2 * b * c * sin A = sqrt 3)
    (h_perimeter : a + b + c = 6)
    (h_A : A = π / 3) :
  a = 2 :=
by
  sorry

end angle_A_value_side_a_value_l8_8224


namespace perp_DM_PN_l8_8283

-- Definitions of the triangle and its elements
variables {A B C M N P D : Point}
variables (triangle_incircle_touch : ∀ (A B C : Point) (triangle : Triangle ABC),
  touches_incircle_at triangle B C M ∧ 
  touches_incircle_at triangle C A N ∧ 
  touches_incircle_at triangle A B P)
variables (point_D : lies_on_segment D N P)
variables {BD CD DP DN : ℝ}
variables (ratio_condition : DP / DN = BD / CD)

-- The theorem statement
theorem perp_DM_PN 
  (h1 : triangle_incircle_touch A B C) 
  (h2 : point_D)
  (h3 : ratio_condition) : 
  is_perpendicular D M P N := 
sorry

end perp_DM_PN_l8_8283


namespace similar_rotated_rhombus_l8_8898

open_locale classical

variables {α : Type*} [LinearOrder α] [AddGroup α]

-- Define lines and relationships
variables {l1 l2 l3 l4 : α → α}

-- Condition: l1 ∥ l2
def parallel_l1_l2 (l1 l2 : α → α) : Prop := ∀ x, l1 x = l2 x + c1

-- Condition: l3 ∥ l4
def parallel_l3_l4 (l3 l4 : α → α) : Prop := ∀ x, l3 x = l4 x + c2

-- Condition: l1 ⊥ l3
def perpendicular_l1_l3 (l1 l3 : α → α) : Prop := ∀ x, l1 x * l3 x = 0

-- Define the points forming rhombus and its rotated counterpart
structure point (α : Type*) := (x y : α)
variables {P Q R S P1 Q1 R1 S1 M : point α}

-- Conditions for points 
variables
  (inscribed_PQRS : Inscribed PQRS l1 l2 l3 l4)
  (center_M : is_center_of_diagonals M PQRS)
  (rotated_PQRS : rotated_rhombus PQRS α M P1 Q1 R1 S1)

-- Theorem statement
theorem similar_rotated_rhombus :
  ∀ α, similar_rhombus PQRS (rotated_rhombus PQRS α M P1 Q1 R1 S1) :=
begin
  sorry
end

end similar_rotated_rhombus_l8_8898


namespace sum_of_first_five_lovely_numbers_l8_8846

def is_proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def sum_squares_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d => is_proper_divisor d n) (Finset.range (n + 1))).sum (λ x => x * x)

def lovely (n : ℕ) : Prop := sum_squares_of_proper_divisors n = n

theorem sum_of_first_five_lovely_numbers : 
  (Finset.filter lovely (Finset.range (11^2 + 1))).sum id = 208 :=
sorry

end sum_of_first_five_lovely_numbers_l8_8846


namespace largest_multiple_of_8_less_than_100_l8_8775

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8775


namespace number_of_pairings_l8_8732

-- Define the conditions: standing equally spaced around a circle, each person knows 4 specific others
def knows (i j : ℕ) : Prop :=
  (i + 1) % 12 = j % 12 ∨
  (i - 1) % 12 = j % 12 ∨
  (i + 6) % 12 = j % 12 ∨
  (i + 3) % 12 = j % 12

-- The proof problem:
theorem number_of_pairings : ∃! (pairs : list (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ pairs → knows p.1 p.2) ∧
  (pairs.length = 6) ∧
  (∀ (i : ℕ), (∃ (p : ℕ × ℕ), p ∈ pairs ∧ i = p.1) ↔ (i < 12)) := 
sorry

end number_of_pairings_l8_8732


namespace m_value_l8_8987

theorem m_value (A : Set ℝ) (B : Set ℝ) (m : ℝ) 
                (hA : A = {0, 1, 2}) 
                (hB : B = {1, m}) 
                (h_subset : B ⊆ A) : 
                m = 0 ∨ m = 2 :=
by
  sorry

end m_value_l8_8987


namespace induced_voltage_constant_magnetic_induction_zero_l8_8454

-- Define the parameters given in the problem
def I (t : ℝ) : ℝ := (10^3) * t
def r : ℝ := 0.2
def R : ℝ := 10^(-2)
def l : ℝ := 0.05
def μ₀ : ℝ := 4 * π * 10^(-7)
def k : ℝ := 10^3

-- The induced voltage U is constant
theorem induced_voltage_constant :
  ∃ U : ℝ, U ≈ 6.44 * 10^(-5) ∧ (∀ t, U = (μ₀ * r * k / (2 * π)) * log (1 + r / l)) :=
sorry

-- Time when the magnetic induction at the center of the square is zero
theorem magnetic_induction_zero :
  ∃ t : ℝ, t ≈ 2.73 * 10^(-4) ∧ (I t / (l + r / 2) = 2 * sqrt 2 * (μ₀ * (6.44 * 10^-3) / (π * r))) :=
sorry

end induced_voltage_constant_magnetic_induction_zero_l8_8454


namespace theta_quadrant_l8_8525

theorem theta_quadrant (θ : ℝ) (h : cos θ * tan θ < 0) : 
  (π / 2 < θ ∧ θ < π) ∨ (π < θ ∧ θ < 3 * π / 2) := 
sorry

end theta_quadrant_l8_8525


namespace number_of_white_balls_l8_8612

theorem number_of_white_balls (x : ℕ) (h1 : 3 + x ≠ 0) (h2 : (3 : ℚ) / (3 + x) = 1 / 5) : x = 12 :=
sorry

end number_of_white_balls_l8_8612


namespace volume_of_inscribed_cube_not_largest_l8_8250

theorem volume_of_inscribed_cube_not_largest (r : ℝ) :
  ∃ (P : Type) [polyhedron P] (hP : inscribed_in_sphere P r), num_vertices P = 8 ∧
  volume_of_polyhedron P > volume_of_cube (2 * r / sqrt 3 * r / sqrt 3 * r / sqrt 3) := 
sorry

end volume_of_inscribed_cube_not_largest_l8_8250


namespace odd_function_increasing_and_min_value_neg_l8_8220

variable {α : Type*} [LinearOrder α]

noncomputable def is_increasing (f : α → α) (a b : α) : Prop :=
∀ x y ∈ Icc a b, x < y → f x < f y

noncomputable def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = -f x

theorem odd_function_increasing_and_min_value_neg
  (f : α → α) (a b : α) (h_odd : is_odd f) (h_inc : is_increasing f a b) (h_min : ∃ x ∈ Icc a b, f x = 1) :
  is_increasing f (-b) (-a) ∧ ∃ x ∈ Icc (-b) (-a), f x = -1 :=
by
  sorry

end odd_function_increasing_and_min_value_neg_l8_8220


namespace no_a_b_domain_range_m_value_range_l8_8980

noncomputable def f (x : ℝ) : ℝ := abs (1 - (1 / x))

theorem no_a_b_domain_range (a b : ℝ) (h : a < b) :
  ¬(∀ x : ℝ, x ≠ 0 → f x ∈ set.Icc a b ∧ x ∈ set.Icc a b) :=
by
  sorry

theorem m_value_range (a b : ℝ) (m : ℝ) (h1 : a < b) (h2 : m ≠ 0) 
  (h3 : ∀ x : ℝ, x ∈ set.Icc a b → f x ∈ set.Icc (m * a) (m * b)) :
  0 < m ∧ m < 1 / 4 :=
by
  sorry

end no_a_b_domain_range_m_value_range_l8_8980


namespace math_problem_l8_8992

-- Definitions and conditions
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 1)
def c : ℝ × ℝ := (sqrt 5 / 5, -2 * sqrt 5 / 5)

-- Support functions
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := real.sqrt (u.1 ^ 2 + u.2 ^ 2)
def projection (u v : ℝ × ℝ) : ℝ × ℝ := (dot_product u v / (magnitude v) ^ 2) • v

-- Theorem statement
theorem math_problem :
  projection a b = (-1 / 2) • b ∧
  dot_product a (a - b) = (2 * sqrt 5) / 5 ∧
  dot_product a c = 0 := by
  sorry

end math_problem_l8_8992


namespace find_k_l8_8953

noncomputable def seq_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (h_inc : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n ≥ 2, S (n - 1) + S n + S (n + 1) = 3 * n^2 + 2)
  (h_a1 : a 1 = 1) (h_S3k : ∃ k : ℕ, S (3 * k) = 324) :
  ∃ k : ℕ, k = 6 := by
exact ⟨6, sorry⟩

end find_k_l8_8953


namespace excircle_radius_opposite_A_eq_six_l8_8539

variables (ABC : Type) [triangle ABC] (C1 B1 A1 D E G : Point ABC)
variables (r : Real) (R : Real) (AB AC BC CE CB1 : Real)
variable (inscribed_circle_radius : r = 1)
variable (CE_segment_length : CE = 6)
variable (CB1_segment_length : CB1 = 1)
variable (excircle_radius_opposite_A : Point ABC → Point ABC → Real → Real → Real → Real)

theorem excircle_radius_opposite_A_eq_six :
  excircle_radius_opposite_A A B C CE CB1 = 6 :=
sorry

end excircle_radius_opposite_A_eq_six_l8_8539


namespace lines_intersect_l8_8443

structure Point where
  x : ℝ
  y : ℝ

def line1 (t : ℝ) : Point :=
  ⟨1 + 2 * t, 4 - 3 * t⟩

def line2 (u : ℝ) : Point :=
  ⟨5 + 4 * u, -2 - 5 * u⟩

theorem lines_intersect (x y t u : ℝ) 
  (h1 : x = 1 + 2 * t)
  (h2 : y = 4 - 3 * t)
  (h3 : x = 5 + 4 * u)
  (h4 : y = -2 - 5 * u) :
  x = 5 ∧ y = -2 := 
sorry

end lines_intersect_l8_8443


namespace last_passenger_sits_in_assigned_seat_l8_8324

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l8_8324


namespace trapezoid_perimeter_l8_8418

structure Point := (x : ℝ) (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def perimeter (A B C D : Point) : ℝ :=
  distance A B + distance B C + distance C D + distance D A

theorem trapezoid_perimeter :
  let J := Point.mk (-2) (-4)
  let K := Point.mk (-2) 2
  let L := Point.mk 6 8
  let M := Point.mk 6 (-4)
  perimeter J K L M = 36 :=
by
  sorry

end trapezoid_perimeter_l8_8418


namespace complex_coordinate_l8_8615

theorem complex_coordinate (z : ℂ) : z = (3 + 5 * complex.i) / (1 + complex.i) → z = 4 + complex.i := by
  intro h
  rw [complex.div_eq_mul_inv, complex.inv_def, complex.conj, complex.mul_comm, complex.mul_div_cancel', complex.add_comm]
  sorry

end complex_coordinate_l8_8615


namespace log_sum_geometric_sequence_l8_8242

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem log_sum_geometric_sequence
  {a r : ℝ}
  (h_pos : ∀ n : ℕ, geometric_sequence a r n > 0)
  (h_eq : geometric_sequence a r 3 * geometric_sequence a r 6 * geometric_sequence a r 9 = 4) :
  (Real.log 2 (geometric_sequence a r 2) + Real.log 2 (geometric_sequence a r 4) 
  + Real.log 2 (geometric_sequence a r 8) + Real.log 2 (geometric_sequence a r 10)) = (16/3) :=
by
  sorry

end log_sum_geometric_sequence_l8_8242


namespace admissible_sequences_count_valid_values_of_n_l8_8048

/-- Part (a): The number of admissible sequences of 4 even digits is 540 -/
theorem admissible_sequences_count : 
  let even_digits := {0, 2, 4, 6, 8}
  let count_admissible_sequences := 540
  ∀ (seq : list ℕ), (∀ x ∈ seq, x ∈ even_digits) → (seq.length = 4 ∧ (∀ x ∈ seq, seq.count x ≤ 2)) → 
  seq.permutations.length = count_admissible_sequences :=
sorry

/-- Part (b): The values of n for which d_n+1 / d_n is an integer -/
theorem valid_values_of_n {n: ℕ} (hn: n ≥ 2) :
   let d (n: ℕ) := n * 540^(n-1)
   let res := (d (n + 1)) / (d n)
   res ∈ ℕ → n ∈ {2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 27, 30, 36, 45, 54, 60, 90, 108, 135, 180, 270, 540} :=
sorry

end admissible_sequences_count_valid_values_of_n_l8_8048


namespace min_knights_proof_l8_8424

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end min_knights_proof_l8_8424


namespace pascals_triangle_row_20_fifth_element_l8_8466

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- State the theorem about Row 20, fifth element in Pascal's triangle
theorem pascals_triangle_row_20_fifth_element :
  binomial 20 4 = 4845 := 
by
  sorry

end pascals_triangle_row_20_fifth_element_l8_8466


namespace sum_of_favorites_l8_8682

/-- Define the favorite numbers of Misty, Glory, and Dawn --/
variables (M G D : ℕ)
variable (hG : G = 450)
variable (hM : M = G / 3)
variable (hD : D = 2 * G)

theorem sum_of_favorites : M + G + D = 1500 := by
  sorry

end sum_of_favorites_l8_8682


namespace twelfth_odd_multiple_of_5_is_115_l8_8030

theorem twelfth_odd_multiple_of_5_is_115 :
  let sequence := λ n : ℕ, 10 * n + 5 in
  sequence 11 = 115 :=
by
  simp only [sequence]
  sorry

end twelfth_odd_multiple_of_5_is_115_l8_8030


namespace tetrahedrons_not_necessarily_similar_l8_8989

theorem tetrahedrons_not_necessarily_similar 
  (T₁ T₂ : Type)
  [tetrahedron T₁] [tetrahedron T₂]
  (no_two_faces_similar₁ : ∀ (f₁ f₂ : face T₁), f₁ ≠ f₂ → ¬similar f₁ f₂) 
  (no_two_faces_similar₂ : ∀ (f₁ f₂ : face T₂), f₁ ≠ f₂ → ¬similar f₁ f₂)
  (each_face_similar : ∀ (f₁ : face T₁), ∃ (f₂ : face T₂), similar f₁ f₂)
  : ¬similar T₁ T₂ :=
sorry

end tetrahedrons_not_necessarily_similar_l8_8989


namespace sum_of_squares_of_roots_eq_213_l8_8158

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l8_8158


namespace log_inequality_l8_8529

theorem log_inequality (a x y : ℝ) (ha : 0 < a) (ha_lt_1 : a < 1) 
(h : x^2 + y = 0) : 
  Real.log (a^x + a^y) / Real.log a ≤ Real.log 2 / Real.log a + 1 / 8 :=
sorry

end log_inequality_l8_8529


namespace train_distance_difference_l8_8381

theorem train_distance_difference 
  (speed1 speed2 : ℕ) (distance : ℕ) (meet_time : ℕ)
  (h_speed1 : speed1 = 16)
  (h_speed2 : speed2 = 21)
  (h_distance : distance = 444)
  (h_meet_time : meet_time = distance / (speed1 + speed2)) :
  (speed2 * meet_time) - (speed1 * meet_time) = 60 :=
by
  sorry

end train_distance_difference_l8_8381


namespace jack_needs_more_money_l8_8630

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l8_8630


namespace final_number_is_80_or_66_l8_8040

-- Define the initial number as having 100 fives
def initial_number : ℕ := nat.mk_nat (list.repeat 5 100)

-- Define Operation 1: Discard the last four digits and subtract the remaining number from those four digits
def operation1 (n : ℕ) : ℕ := 
  let head := n / 10000 in
  let tail := n % 10000 in
  let diff := abs (head - tail) in
  diff

-- Define Operation 2: Replace three consecutive digits a, b, c with a-1, b+3, c-3 if a > 0, b < 7, c > 2
def operation2 (n : ℕ) : ℕ := 
  sorry /- Operation 2 is more complex to directly model but assume defined -/

-- State that if you start from initial_number, after performing the operations, you get either 80 or 66.
theorem final_number_is_80_or_66 :
  ∃ n, (n = 80 ∨ n = 66) ∧ (operation1 ∘ operation2) (initial_number) = n := 
  sorry

end final_number_is_80_or_66_l8_8040


namespace series_sum_convergence_l8_8056

noncomputable def aₙ (x : ℝ) : ℕ → ℝ
| 0       := 0
| 1       := x
| (n + 1) := x * (finset.range (n + 1)).sum (λ i, aₙ x i)

theorem series_sum_convergence (x : ℝ) (h : -2 < x ∧ x < 0) :
  ∑' n, aₙ x n = -1 :=
by
  sorry

end series_sum_convergence_l8_8056


namespace ratio_of_volumes_l8_8791

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ratio_of_volumes :
  let r_C := 10
  let h_C := 20
  let r_D := 18
  let h_D := 12
  volume_cone r_C h_C / volume_cone r_D h_D = 125 / 243 :=
by
  sorry

end ratio_of_volumes_l8_8791


namespace gas_pipe_probability_l8_8073

theorem gas_pipe_probability :
  let p := (75 * 75 : ℝ) / (100 * 100)
  in p = 9 / 16 :=
by
  -- Definitions and assumptions
  let x := 100.0
  let y := 75.0
  let area_total := (1 / 2 * x * x : ℝ)
  let area_usable := (1 / 2 * y * y : ℝ)
  -- Calculation and proof
  have h1 : area_total = 5000.0 := sorry
  have h2 : area_usable = 2812.5 := sorry
  let p := area_usable / area_total
  show p = 9 / 16, from sorry

end gas_pipe_probability_l8_8073


namespace celine_library_charge_l8_8828

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l8_8828


namespace tom_seashells_l8_8016

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end tom_seashells_l8_8016


namespace find_constants_l8_8874

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_constants (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_min_value : ∃ x : ℝ, a * csc (b * x + c) = 3)
  (h_period : ∀ x, a * csc (b * (x + 4 * Real.pi) + c) = a * csc (b * x + c)) :
  a = 3 ∧ b = (1 / 2) :=
by
  sorry

end find_constants_l8_8874


namespace percentage_half_fee_concession_l8_8611

theorem percentage_half_fee_concession (T : ℕ) (h1 : T = 750):
  let boys := 0.60 * T,
      girls := 0.40 * T,
      boys_fee_waiver := 0.15 * boys,
      girls_fee_waiver := 0.075 * girls,
      total_fee_waiver := boys_fee_waiver + girls_fee_waiver,
      total_not_fee_waiver := T - total_fee_waiver,
      half_fee_concession := 330 in 
  T = 750 →
  boys_fee_waiver + girls_fee_waiver = 90 →
  total_not_fee_waiver = 0.88 * T →
  (half_fee_concession / total_not_fee_waiver) * 100 = 50 :=
by
  sorry

end percentage_half_fee_concession_l8_8611


namespace cricket_bat_profit_percentage_l8_8809

theorem cricket_bat_profit_percentage 
  (selling_price profit : ℝ) 
  (h_sp: selling_price = 850) 
  (h_p: profit = 230) : 
  (profit / (selling_price - profit) * 100) = 37.10 :=
by
  sorry

end cricket_bat_profit_percentage_l8_8809


namespace quadrilateral_diagonals_bisect_is_parallelogram_quadrilateral_diagonals_perpendicular_not_rhombus_parallelogram_equal_diagonals_not_rhombus_parallelogram_right_angle_is_rectangle_correct_propositions_l8_8799

theorem quadrilateral_diagonals_bisect_is_parallelogram
  (Q : Type) [quadrilateral Q] (bis : ∀(d1 d2 : diagonal Q), bisects d1 d2) : parallelogram Q := by
  sorry

theorem quadrilateral_diagonals_perpendicular_not_rhombus
  (Q : Type) [quadrilateral Q] (perp : ∀(d1 d2 : diagonal Q), perpendicular d1 d2) : ¬rhombus Q := by
  sorry

theorem parallelogram_equal_diagonals_not_rhombus
  (P : Type) [parallelogram P] (eqDiags : ∀(d1 d2 : diagonal P), equal d1 d2) : ¬rhombus P := by
  sorry

theorem parallelogram_right_angle_is_rectangle
  (P : Type) [parallelogram P] (rightAngle : ∃(a : angle P), is_right_angle a) : rectangle P := by
  sorry

theorem correct_propositions :
  quadrilateral_diagonals_bisect_is_parallelogram ∧ parallelogram_right_angle_is_rectangle ∧ 
  quadrilateral_diagonals_perpendicular_not_rhombus ∧ parallelogram_equal_diagonals_not_rhombus := by
  apply And.intro;
  apply quadrilateral_diagonals_bisect_is_parallelogram;
  apply parallelogram_right_angle_is_rectangle;
  apply quadrilateral_diagonals_perpendicular_not_rhombus;
  apply parallelogram_equal_diagonals_not_rhombus;

end quadrilateral_diagonals_bisect_is_parallelogram_quadrilateral_diagonals_perpendicular_not_rhombus_parallelogram_equal_diagonals_not_rhombus_parallelogram_right_angle_is_rectangle_correct_propositions_l8_8799


namespace eighth_roots_sum_property_l8_8031

noncomputable theory

open Complex

-- Define the set of 8th roots of unity
def eighth_roots_of_unity : Finset ℂ :=
  {z | z^8 = 1}.to_finset

-- Define the sum
def root_sum := ∑ z in eighth_roots_of_unity, 1 / (abs (1 - z))^2

-- Theorem statement proving the sum is a specific non-zero value
theorem eighth_roots_sum_property : ∃ c ≠ 0, ∑ z in eighth_roots_of_unity, 1 / (abs (1 - z))^2 = c :=
by
  sorry

end eighth_roots_sum_property_l8_8031


namespace ellipse_parameters_l8_8079

theorem ellipse_parameters 
  (x y : ℝ)
  (h : 2 * x^2 + y^2 + 42 = 8 * x + 36 * y) :
  ∃ (h k : ℝ) (a b : ℝ), 
    (h = 2) ∧ (k = 18) ∧ (a = Real.sqrt 290) ∧ (b = Real.sqrt 145) ∧ 
    ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
sorry

end ellipse_parameters_l8_8079


namespace bruce_money_left_to_buy_more_clothes_l8_8876

def calculate_remaining_money 
  (amount_given : ℝ) 
  (shirt_price : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ)
  (sock_price : ℝ) (num_socks : ℕ)
  (belt_original_price : ℝ) (belt_discount : ℝ)
  (total_discount : ℝ) : ℝ := 
let shirts_cost := shirt_price * num_shirts
let socks_cost := sock_price * num_socks
let belt_price := belt_original_price * (1 - belt_discount)
let total_cost := shirts_cost + pants_price + socks_cost + belt_price
let discount_cost := total_cost * total_discount
let final_cost := total_cost - discount_cost
amount_given - final_cost

theorem bruce_money_left_to_buy_more_clothes 
  : calculate_remaining_money 71 5 5 26 3 2 12 0.25 0.10 = 11.60 := 
by
  sorry

end bruce_money_left_to_buy_more_clothes_l8_8876


namespace sector_area_l8_8710

noncomputable def area_of_sector (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) : ℝ :=
  1 / 2 * arc_length * radius

theorem sector_area (R : ℝ)
  (arc_length : ℝ) (central_angle : ℝ)
  (h_arc : arc_length = 4 * Real.pi)
  (h_angle : central_angle = Real.pi / 3)
  (h_radius : arc_length = central_angle * R) :
  area_of_sector arc_length central_angle 12 = 24 * Real.pi :=
by
  -- Proof skipped
  sorry

#check sector_area

end sector_area_l8_8710


namespace vector_magnitude_problem_l8_8217

noncomputable def angle_cos (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

variables (a b : ℝ × ℝ)
variables (a_magnitude b_magnitude : ℝ)
variables (angle_ab : ℝ)

/-- The magnitude of a vector -/
noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- The dot product of two vectors -/
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

/-- The proof problem -/
theorem vector_magnitude_problem
  (h1 : a = (3 / 5, -4 / 5))
  (h2 : vector_magnitude b = 2)
  (h3 : angle_ab = 120) :
  let c := (2 * a.1 - b.1, 2 * a.2 - b.2) in
  vector_magnitude c = 2 * Real.sqrt 3 := 
by
  sorry

end vector_magnitude_problem_l8_8217


namespace operation_applied_twice_to_v_l8_8930

def O (v : ℝ) : ℝ := v - v / 3

theorem operation_applied_twice_to_v {v : ℝ} (h : v = 45) : O (O v) = 20 :=
by
  rw h
  sorry

end operation_applied_twice_to_v_l8_8930


namespace value_of_r2_l8_8589

-- Definitions of the conditions
variables {x : ℝ} 

-- Given conditions
def r1 : ℝ := (1 / 3) ^ 6
def q1 (x : ℝ) : ℝ := (x ^ 5 - (5/3 * x^4) + (10/9 * x^3) - (10/27 * x^2) + (5/(3^4) * x) - 1/(3^5))

-- We need to prove that r2 is 1/1024
def r2 : ℝ := q1 (1 / 4)

-- Main statement to be proved
theorem value_of_r2 : r2 = 1 / 1024 := by
  sorry

end value_of_r2_l8_8589


namespace largest_m_divides_30_fact_l8_8513

theorem largest_m_divides_30_fact : 
  let pow2_in_fact := 15 + 7 + 3 + 1,
      pow3_in_fact := 10 + 3 + 1,
      max_m_from_2 := pow2_in_fact,
      max_m_from_3 := pow3_in_fact / 2
  in max_m_from_2 >= 7 ∧ max_m_from_3 >= 7 → 7 = 7 :=
by
  sorry

end largest_m_divides_30_fact_l8_8513


namespace repeating_decimal_subtraction_simplified_l8_8393

theorem repeating_decimal_subtraction_simplified :
  let x := (567 / 999 : ℚ)
  let y := (234 / 999 : ℚ)
  let z := (891 / 999 : ℚ)
  x - y - z = -186 / 333 :=
by
  sorry

end repeating_decimal_subtraction_simplified_l8_8393


namespace pentagon_area_ratio_l8_8264

theorem pentagon_area_ratio
    (FGHIJ : Type)
    (FG IJ GH FI HJ : FGHIJ)
    (angle_FGH : real)
    (parallel_FG_IJ : FG ∥ IJ)
    (parallel_GH_FI : GH ∥ FI)
    (parallel_GI_HJ : GI ∥ HJ)
    (FG_len : ℝ)
    (GH_len : ℝ)
    (HJ_len : ℝ)
    (area_ratio : ℝ)
    (h1 : angle_FGH = 100)
    (h2 : FG_len = 4)
    (h3 : GH_len = 7)
    (h4 : HJ_len = 20)
    (h5 : area_ratio = (1 / 15)) :
    1 + 15 = 16 :=
by
  sorry

end pentagon_area_ratio_l8_8264


namespace g_sum_l8_8279

def g (x : ℝ) : ℝ :=
  if x > 6 then x^2 - 1
  else if -6 ≤ x ∧ x ≤ 6 then 3 * x + 2
  else -2 

theorem g_sum :
  g (-8) + g (0) + g (8) = 63 :=
by
  sorry

end g_sum_l8_8279


namespace intersection_of_A_and_B_l8_8200

-- Define the sets A and B
def A : Set ℤ := {-2, -1, 0, 1}
def B : Set ℕ := {x : ℕ | true} -- all natural numbers (x ≥ 0 by definition)

-- Define the proof problem
theorem intersection_of_A_and_B :
  A ∩ B = {0, 1} := 
sorry

end intersection_of_A_and_B_l8_8200


namespace largest_multiple_of_8_less_than_100_l8_8773

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l8_8773


namespace tony_money_left_after_game_l8_8379

variable (initial_amount : ℕ) (ticket_cost : ℕ) (hotdog_cost : ℕ)

def money_left_after_game (initial_amount ticket_cost hotdog_cost : ℕ) : ℕ :=
  initial_amount - (ticket_cost + hotdog_cost)

theorem tony_money_left_after_game 
  (h1 : initial_amount = 20) 
  (h2 : ticket_cost = 8) 
  (h3 : hotdog_cost = 3) : 
  money_left_after_game initial_amount ticket_cost hotdog_cost = 9 := 
by
  rw [money_left_after_game, h1, h2, h3]
  norm_num

end tony_money_left_after_game_l8_8379


namespace sum_of_coefficients_l8_8036

def original_function (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 4

def transformed_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2 * (x + 2) + 4 + 5

theorem sum_of_coefficients : (3 : ℝ) + 10 + 17 = 30 :=
by
  sorry

end sum_of_coefficients_l8_8036


namespace compare_a_b_c_l8_8944

noncomputable def a : ℝ := 2 ^ (1 / 2)
noncomputable def b : ℝ := (2 ^ (Real.log 3 / Real.log 2)) ^ (-1 / 2)
noncomputable def c : ℝ := (Real.cos (50 * Real.pi / 180) * Real.cos (10 * Real.pi / 180)) 
                           + (Real.cos (140 * Real.pi / 180) * Real.sin (170 * Real.pi / 180))

theorem compare_a_b_c : a > b ∧ b > c :=
by
  sorry

end compare_a_b_c_l8_8944


namespace last_passenger_sits_in_assigned_seat_l8_8326

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l8_8326


namespace largest_m_dividing_factorial_l8_8512

theorem largest_m_dividing_factorial :
  (∃ m : ℕ, (∀ n : ℕ, (18^n ∣ 30!) ↔ n ≤ m) ∧ m = 7) :=
by
  sorry

end largest_m_dividing_factorial_l8_8512


namespace ratio_of_areas_l8_8082

-- Define the initial conditions as given in the problem.
def initial_length := 10
def initial_width := 5
def new_length := initial_length + (0.5 * initial_length)
def new_width := initial_width + (1.0 * initial_width)

-- Define the original and new areas in terms of the initial conditions.
def original_area := initial_length * initial_width
def new_area := new_length * new_width

-- State the theorem to prove the given ratio.
theorem ratio_of_areas :
  (original_area : ℚ) / (new_area : ℚ) = 1 / 3 :=
by
  sorry

end ratio_of_areas_l8_8082


namespace tangent_identity_proof_l8_8883

noncomputable def tan_tangent_identity (x : ℝ) : Prop :=
  tan (real.pi * (18 / 180) - x) * tan (real.pi * (12 / 180) + x) + 
  real.sqrt 3 * (tan (real.pi * (18 / 180) - x) + tan (real.pi * (12 / 180) + x)) = 1

theorem tangent_identity_proof (x : ℝ) : tan_tangent_identity x :=
by {
  sorry
}

end tangent_identity_proof_l8_8883


namespace sum_F_1_to_100_l8_8649

def E (n : ℕ) : ℕ := (n.digits 10).filter (λ x, x % 2 = 0).sum

def F (n : ℕ) : ℕ := if n % 10 = 4 then E n else 0

theorem sum_F_1_to_100 : (∑ n in Finset.range 101, F n) = 68 := 
by 
  sorry

end sum_F_1_to_100_l8_8649


namespace max_value_of_y_l8_8174

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logBase 3 x

theorem max_value_of_y :
  ∃ x ∈ Set.Icc 1 9, 
    let y := (f x)^2 + (f (x^2))
    y = 13 ∧ x = 3 :=
by
  sorry

end max_value_of_y_l8_8174


namespace trigonometric_expression_zero_l8_8823

theorem trigonometric_expression_zero (α : ℝ) :
  sin (2 * α - 3 / 2 * Real.pi) + cos (2 * α - 8 / 3 * Real.pi) + cos (2 / 3 * Real.pi + 2 * α) = 0 :=
by
  sorry

end trigonometric_expression_zero_l8_8823


namespace cakes_difference_l8_8853

theorem cakes_difference :
  let l := 6
  let d := 9
  d - l = 3 :=
by
  unfold l d
  sorry

end cakes_difference_l8_8853


namespace growing_path_max_points_value_l8_8891

-- Definition of the grid and growing path conditions
def grid_points : list (ℕ × ℕ) := 
  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

def distance (p1 p2 : ℕ × ℕ) : ℂ :=
  complex.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).to_complex

def is_growing_path (path : list (ℕ × ℕ)) : Prop :=
  ∀(i j : ℕ), i < j → distance (path.nth i) (path.nth (i + 1)) < distance (path.nth j) (path.nth (j + 1))

-- Definition of the problem
theorem growing_path_max_points_value :
  ∃ m r, m = 5 ∧ r = 4 ∧ m * r = 20 :=
sorry

end growing_path_max_points_value_l8_8891


namespace evaluate_expression_l8_8488

def a := (64 : ℝ) ^ (-1 / 3 : ℝ)
def b := (81 : ℝ) ^ (-1 / 2 : ℝ)
def result := a + b

theorem evaluate_expression : result = (13 / 36 : ℝ) :=
by 
  sorry

end evaluate_expression_l8_8488


namespace Hank_sold_10_bicycles_l8_8683

theorem Hank_sold_10_bicycles (B S : ℤ) 
  (Friday_eqn : B - S + 15)
  (Saturday_eqn : (B - S + 15) - 12 + 8)
  (Sunday_eqn : ((B - S + 15) - 12 + 8) - 9 + 11)
  (net_increase : ((B - S + 15) - 12 + 8 - 9 + 11 - B = 3)) : S = 10 :=
by
  sorry

end Hank_sold_10_bicycles_l8_8683


namespace no_such_function_exists_l8_8137

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 :=
by
  sorry

end no_such_function_exists_l8_8137


namespace problem_l8_8978

noncomputable def f (ω φ : ℝ) (x : ℝ) := 4 * Real.sin (ω * x + φ)

theorem problem (ω : ℝ) (φ : ℝ) (x1 x2 α : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h0 : f ω φ 0 = 2 * Real.sqrt 3)
  (hx1 : f ω φ x1 = 0) (hx2 : f ω φ x2 = 0) (hx1x2 : |x1 - x2| = Real.pi / 2)
  (hα : α ∈ Set.Ioo (Real.pi / 12) (Real.pi / 2)) :
  f 2 (Real.pi / 3) α = 12 / 5 ∧ Real.sin (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
sorry

end problem_l8_8978


namespace repeating_decimal_count_eq_182_l8_8928

/-- 
Given \( 1 \leq n \leq 200 \):
- \( \frac{n}{n+1} \text{ is in simplest form} \) because gcd(n, n+1) = 1
- For \( \frac{n}{n+1} \text{ to be repeating, } n+1 \text{ should have prime factors other than 2 and 5}\)
- We count such n where \( 1 ≤ n ≤ 200 \) and \( n+1 \) is not of the form \( 2^a * 5^b \)

Prove that the number of such integers \( n \) is 182.
-/
theorem repeating_decimal_count_eq_182 : 
  (Finset.filter (λ n : ℕ, Nat.gcd n (n+1) = 1 ∧ Nat.PrimeFactorOtherThan2And5 (n+1))
    (Finset.range 201)).card = 182 :=
sorry

end repeating_decimal_count_eq_182_l8_8928


namespace terez_pregnant_female_cows_l8_8323

def total_cows := 44
def percent_females := 0.50
def percent_pregnant_females := 0.50

def female_cows := total_cows * percent_females
def pregnant_female_cows := female_cows * percent_pregnant_females

theorem terez_pregnant_female_cows : pregnant_female_cows = 11 := by
  sorry

end terez_pregnant_female_cows_l8_8323


namespace find_set_A_l8_8672

open Set Real

noncomputable def log2 := λ x : ℝ, Real.log x / Real.log 2

theorem find_set_A (A : Set ℝ) (hA : A = {1, log2 3, log2 5}) :
  (∃ A : Set ℝ, (∀ x1 x2 x3 ∈ A, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧
    (let B := { x + y | x ∈ A ∧ y ∈ A ∧ x ≠ y } in
    B = { log2 6, log2 10, log2 15 })) :=
begin
  use {1, log2 3, log2 5},
  split,
  { intros x1 x2 x3 hx1 hx2 hx3,
    -- Proof of distinctness of elements in {1, log2 3, log2 5}
    sorry },
  { let A := {1, log2 3, log2 5},
    let B := { x + y | x ∈ A ∧ y ∈ A ∧ x ≠ y },
    show B = { log2 6, log2 10, log2 15 },
    -- Proof that B is { log2 6, log2 10, log2 15 }
    sorry }
end

end find_set_A_l8_8672


namespace digit_in_hundredths_place_l8_8024

theorem digit_in_hundredths_place : 
  let d := (7: ℚ) / 25 in 
  (d * 100).floor % 10 = 8 :=
by
  let d := (7: ℚ) / 25
  have h₁ : d = 0.28 := by norm_num
  have h₂ : (d * 100).floor = 28 := by norm_num
  show 28 % 10 = 8
  exact rfl

end digit_in_hundredths_place_l8_8024


namespace goods_train_speed_l8_8046

variables (speed_train_kmph : ℝ) (length_goods_m : ℝ) (time_seconds : ℝ)

theorem goods_train_speed (h1 : speed_train_kmph = 50)
                         (h2 : length_goods_m = 280)
                         (h3 : time_seconds = 9) :
  let speed_goods_kmph := ( (length_goods_m / time_seconds) * 3.6 - (speed_train_kmph * 1000 / 3600) * 3.6 ) in
  abs speed_goods_kmph = 62 :=
by
  calc sorry -- Proof steps are omitted.

end goods_train_speed_l8_8046


namespace sequence_sum_to_44_l8_8005

noncomputable def sequence (n : ℕ) : ℤ := 
  if n = 0 then 0 else a n
where
  a : ℕ → ℤ
  | 1 => 0 -- Let a₁=0 for the sake of defining initial value
  | n + 1 => 2 (n - 1) - (-1) ^ (n - 1) * a n

theorem sequence_sum_to_44 : 
  (Finset.range 44).sum sequence = 990 :=
by
  sorry

end sequence_sum_to_44_l8_8005


namespace dan_money_left_l8_8130

def money_left (initial_amount spent_on_candy spent_on_gum : ℝ) : ℝ :=
  initial_amount - (spent_on_candy + spent_on_gum)

theorem dan_money_left :
  money_left 3.75 1.25 0.80 = 1.70 :=
by
  sorry

end dan_money_left_l8_8130


namespace slope_of_line_l8_8135

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end slope_of_line_l8_8135


namespace arithmetic_mean_745_l8_8086

theorem arithmetic_mean_745 (x : ℤ) (h : 3 + 117 + 915 + 138 + 1917 + 2114 + x ≡ 7 [MOD 7]) :
  (3 + 117 + 915 + 138 + 1917 + 2114 + x) / 7 = 745 :=
by 
  sorry

end arithmetic_mean_745_l8_8086


namespace tom_seashells_l8_8015

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end tom_seashells_l8_8015


namespace DX_eq_BX_l8_8715

variables {A B C D E X : Type} [Points A B C D E X]
variables {side_length : ℝ}
variables {angle_90 : ℝ := 90}

-- The pentagon is regular
def is_regular_pentagon (ABCDE : Set Points) : Prop :=
  ∀ (p q : A ∩ B ∩ C ∩ D ∩ E), side_length (p, q) = side_length ∧ 
    (angle_pq p q = 90 → ∀ angle (angle_pq p q), angle = 90 → is_parallel (side pq, side pr))

-- Pentagon conditions
variables (ABCDE : Set Points)
variable hx : is_regular_pentagon (ABCDE) ∧ (angles_in_pentagon ABCDE) angle_90

-- Point X defined as the intersection
variable hx_intersection : X = intersection (AD BE)

theorem DX_eq_BX : DX = BX :=
sorry

end DX_eq_BX_l8_8715


namespace fraction_of_quarters_from_1800_to_1809_l8_8638

def num_total_quarters := 26
def num_states_1800s := 8

theorem fraction_of_quarters_from_1800_to_1809 : 
  (num_states_1800s / num_total_quarters : ℚ) = 4 / 13 :=
by
  sorry

end fraction_of_quarters_from_1800_to_1809_l8_8638


namespace triplet_sequence_bound_l8_8821

/-- 
A triplet of numbers is defined as three numbers where one of them is the arithmetic mean of the other two. 
There is an infinite sequence \(a_n\) consisting of natural numbers. 
It is known that \(a_1 = a_2 = 1\) and for \(n > 2\), the number \(a_n\) is the smallest natural number 
such that among the numbers \(a_1, a_2, \ldots, a_n\), there are no three numbers forming a triplet. 
Prove that \(a_n \leq \frac{n^2 + 7}{8}\) for any \(n\).
-/
theorem triplet_sequence_bound (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 1)
  (h₃ : ∀ n > 2, a n = smallest_nat_not_forming_triplet (λ k, k < n)) :
  ∀ n, a n ≤ (n^2 + 7) / 8 := 
sorry

end triplet_sequence_bound_l8_8821


namespace circle_digits_divisible_by_27_l8_8430

theorem circle_digits_divisible_by_27 (n : ℕ) (digits : Fin n → ℕ) (h1 : n = 1953)
  (h2 : ∃ k : Fin n, (∑ i : Fin n, digits ((k + i) % n) * 10 ^ i) % 27 = 0) :
  ∀ k' : Fin n, (∑ i : Fin n, digits ((k' + i) % n) * 10 ^ i) % 27 = 0 := by
  sorry

end circle_digits_divisible_by_27_l8_8430


namespace largest_multiple_of_8_less_than_100_l8_8764

theorem largest_multiple_of_8_less_than_100 : ∃ n, n < 100 ∧ n % 8 = 0 ∧ ∀ m, m < 100 ∧ m % 8 = 0 → m ≤ n :=
begin
  use 96,
  split,
  { -- prove 96 < 100
    norm_num,
  },
  split,
  { -- prove 96 is a multiple of 8
    norm_num,
  },
  { -- prove 96 is the largest such multiple
    intros m hm,
    cases hm with h1 h2,
    have h3 : m / 8 < 100 / 8,
    { exact_mod_cast h1 },
    interval_cases (m / 8) with H,
    all_goals { 
      try { norm_num, exact le_refl _ },
    },
  },
end

end largest_multiple_of_8_less_than_100_l8_8764


namespace number_of_dogs_l8_8447

theorem number_of_dogs (total_animals cats : ℕ) (probability : ℚ) (h1 : total_animals = 7) (h2 : cats = 2) (h3 : probability = 2 / 7) :
  total_animals - cats = 5 := 
by
  sorry

end number_of_dogs_l8_8447


namespace correct_statement_is_B_l8_8803

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l8_8803


namespace slope_of_tangent_line_l8_8518
open Real

-- Define the curve function y
def curve (x : ℝ) : ℝ := exp x - (1 / x)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := exp x + (1 / (x^2))

-- Define the point of interest
def point_of_interest : ℝ := 1

-- Prove that the slope of the tangent line at the point of interest is e + 1
theorem slope_of_tangent_line : curve_derivative point_of_interest = exp 1 + 1 := 
by
  sorry

end slope_of_tangent_line_l8_8518


namespace triangular_grid_signs_exists_l8_8474

-- Definition of signs
inductive Sign
| Plus : Sign
| Minus : Sign

-- Function to check the condition of the triangular grid
def triangle_condition (a b c : Sign) : Prop :=
  if a = b then c = Sign.Plus else c = Sign.Minus

-- Proof statement
theorem triangular_grid_signs_exists :
  ∃ (vertex_signs : ℕ × ℕ → Sign),
  (∀ x y : ℕ × ℕ, ∀ (a b c : Sign),
    (vertex_signs x = a) → (vertex_signs y = b) →
    (vertex_signs (x.1 + y.1, x.2 + y.2) = c) →
    triangle_condition a b c) ∧
  ¬(∀ x y : ℕ × ℕ, vertex_signs (x, y) = Sign.Plus) :=
sorry

end triangular_grid_signs_exists_l8_8474


namespace largest_angle_correct_l8_8833

-- Define the variables and their relationships based on the given conditions
variables (x : ℝ)
def angle1 := x + 2
def angle2 := 2 * x - 1
def angle3 := 2 * x + 1
def angle4 := 3 * x
def angle5 := 4 * x - 3
def angle6 := 5 * x - 2

-- The sum of interior angles of the hexagon
def hex_sum := angle1 + angle2 + angle3 + angle4 + angle5 + angle6

-- Define the fact that the sum of interior angles in a hexagon is 720 degrees
axiom hex_sum_720 : hex_sum = 720

-- Define the function to get the largest angle
def largest_angle := max angle1 (max angle2 (max angle3 (max angle4 (max angle5 angle6))))

-- Define the problem statement to check that the largest angle is equal to 3581/17 degrees
theorem largest_angle_correct
    (hex_cond : hex_sum_720) : largest_angle x = 3581 / 17 := sorry

end largest_angle_correct_l8_8833


namespace tangent_y_intercept_l8_8118

theorem tangent_y_intercept :
  ∀ (x1 y1 r1 x2 y2 r2 m b : ℝ),
    x1 = 3 →
    y1 = 2 →
    r1 = 5 →
    x2 = 12 →
    y2 = 10 →
    r2 = 7 →
    m > 0 →
    (∀ x y, (y = m * x + b) → (x1 - x)^2 + (y1 - y)^2 = r1^2 ∧ (x2 - x)^2 + (y2 - y)^2 = r2^2) →
    b = -313 / 17 :=
begin
  intros x1 y1 r1 x2 y2 r2 m b hx1 hy1 hr1 hx2 hy2 hr2 hm tangents,
  -- The proof will be here
  sorry
end

end tangent_y_intercept_l8_8118


namespace smallest_number_among_neg2_neg1_0_1_l8_8459

theorem smallest_number_among_neg2_neg1_0_1 : 
    ∀ x ∈ ({-2, -1, 0, 1} : Set ℤ), -2 ≤ x :=
by
  intros x hx
  cases hx with
  | inl hx => exact hx.le
  | inr hx =>
    cases hx with
    | inl hx := exact Int.zero_le_add_one.trans (hx.symm.le)
    | inr hx =>
      cases hx with
      | inl hx := exact zero_le_one.trans (hx.symm.le)
      | inr hx := exact Int.zero_le_add_one.trans (one_le_two.trans (hx.symm.le))

end smallest_number_among_neg2_neg1_0_1_l8_8459


namespace max_terms_sequence_l8_8605

noncomputable def max_terms (seq : ℕ → ℝ) : ℕ :=
  if (∀ n, seq (n) + seq (n + 1) + seq (n + 2) < 0 ∧
  ∀ m, seq (m) + seq (m + 1) + seq (m + 2) + seq (m + 3) > 0) then 5 else sorry -- Assuming no other r satisfies the conditions

theorem max_terms_sequence : max_terms = 5 := 
by 
  intros seq;
  intro h;
  sorry -- the proof steps would go here

end max_terms_sequence_l8_8605


namespace Ryanne_is_7_years_older_than_Hezekiah_l8_8310

theorem Ryanne_is_7_years_older_than_Hezekiah
  (H : ℕ) (R : ℕ)
  (h1 : H = 4)
  (h2 : R + H = 15) :
  R - H = 7 := by
  sorry

end Ryanne_is_7_years_older_than_Hezekiah_l8_8310


namespace bus_driver_regular_rate_l8_8063

-- Conditions:
def regular_rate_applies_to_hours := 40
def overtime_rate_factor := 1.75
def total_compensation := 998
def hours_worked := 58
def overtime_hours := hours_worked - regular_rate_applies_to_hours

-- Main statement to prove
theorem bus_driver_regular_rate :
  ∃ (R : ℝ), 998 = (regular_rate_applies_to_hours * R) + (overtime_hours * (R * overtime_rate_factor)) ∧ R ≈ 13.95 :=
begin
  sorry
end

end bus_driver_regular_rate_l8_8063


namespace find_expression_for_f_l8_8530

theorem find_expression_for_f (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 6 * x) :
  ∀ x, f x = x^2 + 8 * x + 7 :=
by
  sorry

end find_expression_for_f_l8_8530


namespace largest_multiple_of_8_less_than_100_l8_8769

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8769


namespace max_money_earned_l8_8369

-- Definitions
def piles_initial (a b c : ℕ) : Prop := True

def move_stone (a b c : ℕ) (m n o : ℕ) : Prop :=
  ∃ (x y z : ℕ), (x + y + z = a + b + c) ∧ 
  ((x = a + 1 ∧ y = b - 1 ∧ z = c) ∨
   (x = a - 1 ∧ y = b + 1 ∧ z = c) ∨
   (x = a ∧ y = b + 1 ∧ z = c - 1) ∨
   (x = a ∧ y = b - 1 ∧ z = c + 1) ∨
   (x = a + 1 ∧ y = b ∧ z = c - 1) ∨
   (x = a - 1 ∧ y = b ∧ z = c + 1))

def final_state (a b c : ℕ) (initial_a initial_b initial_c : ℕ) : Prop :=
  a = initial_a ∧ b = initial_b ∧ c = initial_c

-- Theorem
theorem max_money_earned (a b c : ℕ) :
  piles_initial a b c →
  (∀ m n o, move_stone a b c m n o) →
  final_state a b c a b c →
  (moneysum : ℤ :=
    let funds := λ (x y z : ℕ), if x > y then x - y else - (y - x)
    in (funds a b) + (funds b c) + (funds c a) : ℤ ) = 0 :=
begin
  intros h_initial h_move h_final,
  sorry
end

end max_money_earned_l8_8369


namespace probability_units_digit_7_l8_8162

noncomputable def units_digit_mod_10 (n : ℕ) : ℕ := n % 10

theorem probability_units_digit_7 (c d : ℕ) (hc : c ∈ {i | 1 ≤ i ∧ i ≤ 100}) (hd : d ∈ {i | 1 ≤ i ∧ i ≤ 100}) :
  1 / 4 =
  let count := (∑ c in (finset.range 100).filter (λ c, units_digit_mod_10 (2 ^ c) = 2), 
                ∑ d in (finset.range 100), if units_digit_mod_10 (2 ^ c + 5 ^ d) = 7 then 1 else 0) in
  (count.to_real / (100 * 100).to_real) := sorry

end probability_units_digit_7_l8_8162


namespace minimum_number_of_bricks_l8_8922

def width := 18
def depth := 12
def height := 9
def lcm_dimensions := 36 -- LCM of 18, 12, and 9
def volume_of_cube := lcm_dimensions ^ 3
def volume_of_brick := width * depth * height

theorem minimum_number_of_bricks : volume_of_cube / volume_of_brick = 24 :=
by
  sorry

end minimum_number_of_bricks_l8_8922


namespace last_passenger_sits_in_assigned_seat_l8_8343

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l8_8343


namespace find_m_pure_imaginary_l8_8595

noncomputable def find_m (m : ℝ) : ℝ := m

theorem find_m_pure_imaginary (m : ℝ) (h : (m^2 - 5 * m + 6 : ℂ) = 0) :
  find_m m = 2 :=
by
  sorry

end find_m_pure_imaginary_l8_8595


namespace printer_completion_time_l8_8849

def start_time := 9 -- 9:00 AM in hours
def quarter_task_time := 3 -- time taken for quarter of the task in hours
def noon_time := 12 -- 12:00 PM in hours
def speed_increase := 1.25 -- 25% speed increase

theorem printer_completion_time :
  let original_completion_time := quarter_task_time * 4 in
  let remaining_task := 3 / 4 in
  let new_speed_time_per_quarter := quarter_task_time / speed_increase in
  let remaining_task_time := remaining_task * new_speed_time_per_quarter in
  noon_time + remaining_task_time = 7 + 12 / 60 :=
sorry

end printer_completion_time_l8_8849


namespace eric_bike_speed_l8_8915

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 1
def run_distance : ℝ := 2
def run_speed : ℝ := 8
def bike_distance : ℝ := 12
def total_time_limit : ℝ := 2

theorem eric_bike_speed :
  (swim_distance / swim_speed) + (run_distance / run_speed) + (bike_distance / (48/5)) < total_time_limit :=
by
  sorry

end eric_bike_speed_l8_8915


namespace sum_expression_is_160_l8_8674

noncomputable def T : ℝ :=
  ∑ n in finset.range 10000, 1 / real.sqrt (n + real.sqrt (n^2 - 4))

theorem sum_expression_is_160 :
  ∃ (p q r : ℕ), T = p + q * real.sqrt r ∧ r.is_squarefree ∧ p + q + r = 160 :=
by
-- The detailed proof is omitted since the task requires only the theorem statement.
sorry

end sum_expression_is_160_l8_8674


namespace reduced_price_per_dozen_bananas_l8_8810

variables (P R : ℝ) (B : ℝ)
def condition1 := R = 0.60 * P
def condition2 := P = 40.00001 / B
def condition3 := R = 40.00001 / (B + 64)

theorem reduced_price_per_dozen_bananas (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  R * 12 = 3.00000075 :=
sorry

end reduced_price_per_dozen_bananas_l8_8810


namespace total_amount_received_l8_8808

theorem total_amount_received (B : ℕ) (price_per_book : ℝ) (sold_fraction : ℝ) (remaining_books : ℕ) :
  sold_fraction = (2 / 3) →
  remaining_books = 30 →
  price_per_book = 4.25 →
  B = 3 * remaining_books →
  (2 / 3) * B * price_per_book = 255 :=
by
  intros h1 h2 h3 h4
  rw h1 h2 h3 h4
  norm_num

end total_amount_received_l8_8808


namespace number_of_trucks_l8_8412
open Nat

theorem number_of_trucks (each_truck_packages total_packages : ℕ) (h_each_truck : each_truck_packages = 70) (h_total : total_packages = 490) : total_packages / each_truck_packages = 7 :=
by
  rw [h_each_truck, h_total]
  norm_num
  sorry

end number_of_trucks_l8_8412


namespace height_difference_meters_height_difference_feet_l8_8581

theorem height_difference_meters (h_eiffel_m : ℕ) (h_burj_m : ℕ) (h_e : h_eiffel_m = 324) (h_b : h_burj_m = 830) :
  h_burj_m - h_eiffel_m = 506 :=
by
  rw [h_e, h_b]
  exact Nat.sub_eq_of_eq_add sorry

theorem height_difference_feet (h_eiffel_ft : ℕ) (h_burj_ft : ℕ) (h_e_f : h_eiffel_ft = 1063) (h_b_f : h_burj_ft = 2722) :
  h_burj_ft - h_eiffel_ft = 1659 :=
by
  rw [h_e_f, h_b_f]
  exact Nat.sub_eq_of_eq_add sorry

end height_difference_meters_height_difference_feet_l8_8581


namespace find_zero_in_range_l8_8004

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n > 1, a n = a (n / 2) + (-1) ^ (n * (n + 1) / 2)

theorem find_zero_in_range : ∀ k > 0, ∃ n, 2^k ≤ n ∧ n < 2^(k+1) ∧ a n = 0 :=
by
  assume a : ℕ → ℤ
  assume seq_def : sequence a
  induction k with k ih
  case zero =>
    trivial
  case succ k ih =>
    sorry

end find_zero_in_range_l8_8004


namespace gain_percentage_is_25_l8_8410

def selling_price : ℝ := 100
def gain : ℝ := 20
def cost_price : ℝ := selling_price - gain
def gain_percentage : ℝ := (gain / cost_price) * 100

theorem gain_percentage_is_25 : gain_percentage = 25 := by
  sorry

end gain_percentage_is_25_l8_8410


namespace find_point_Q_l8_8956

def point (x y : ℝ) : Prop := True

def on_line (Q : ℝ × ℝ) : Prop :=
  let (x, y) := Q in x - y + 1 = 0

def perpendicular_slope_condition (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := Q in let (px, py) := P in 
  x ≠ 0 ∧ (y + 1) / x * (-1 / 2) = -1

theorem find_point_Q :
  ∃ Q : ℝ × ℝ, point 0 (-1) ∧ on_line Q ∧ perpendicular_slope_condition Q (0, -1) ∧ Q = (2, 3) :=
by
  sorry

end find_point_Q_l8_8956


namespace last_passenger_sits_in_assigned_seat_l8_8325

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l8_8325


namespace greater_area_circle_l8_8756

-- Define areas and perimeters of convex figures and circles
def Area (fig : Type) : ℝ := sorry
def Perimeter (fig : Type) : ℝ := sorry

-- Define the convex figure and circle types
def ConvexFigure := sorry
def Circle := sorry

-- Assume result of problem 59 (implicitly satisfied assumptions about polygons inscribed)
axiom problem_59 (polygon : ConvexFigure) (circum_circle : Circle) : 
  Area polygon / (Perimeter polygon)^2 ≤ Area circum_circle / (Perimeter circum_circle)^2

-- Define the problem to be proved
theorem greater_area_circle (ϕ: ConvexFigure) (K: Circle) :
  (Perimeter ϕ = Perimeter K) → 
  ((Area K) / (Perimeter K)^2 > (Area ϕ) / (Perimeter ϕ)^2) :=
by {
  intro perimeter_eq,
  have ineq := problem_59 ϕ K,
  sorry -- Remaining steps to prove the greater area property using the given inequality and condition
}

end greater_area_circle_l8_8756


namespace fifth_term_arithmetic_sequence_l8_8718

-- Conditions provided
def first_term (x y : ℝ) := x + y^2
def second_term (x y : ℝ) := x - y^2
def third_term (x y : ℝ) := x - 3*y^2
def fourth_term (x y : ℝ) := x - 5*y^2

-- Proof to determine the fifth term
theorem fifth_term_arithmetic_sequence (x y : ℝ) :
  (fourth_term x y) - (third_term x y) = -2*y^2 →
  (x - 5 * y^2) - 2 * y^2 = x - 7 * y^2 :=
by sorry

end fifth_term_arithmetic_sequence_l8_8718


namespace equation_solutions_l8_8588

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃ x : ℝ, ax + b = 0) ∨ (∃ x : ℝ, ∀ y : ℝ, ax + b = 0 → x = y) :=
sorry

end equation_solutions_l8_8588


namespace largest_multiple_of_8_less_than_100_l8_8767

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l8_8767


namespace expected_value_coin_flip_l8_8866

-- Define the conditions
def probability_heads := 2 / 3
def probability_tails := 1 / 3
def gain_heads := 5
def loss_tails := -10

-- Define the expected value calculation
def expected_value := (probability_heads * gain_heads) + (probability_tails * loss_tails)

-- Prove that the expected value is 0.00
theorem expected_value_coin_flip : expected_value = 0 := 
by sorry

end expected_value_coin_flip_l8_8866


namespace sum_of_c_sn_ineq_l8_8170

def an (n : ℕ) : ℕ := n
def bn (n : ℕ) : ℕ := 2^(n-1)
def Sn (n : ℕ) : ℕ := n * (n + 1) / 2

def cn (n : ℕ) : ℚ := 
if n % 2 == 1 then 
  (3 * an n - 2) * bn n / (an n * an (n + 2))
else
  an (n - 1) / bn (n + 1)

theorem sum_of_c (n : ℕ) : 
  (finset.range (2 * n + 1)).sum cn = 4^n / (2 * n + 1) - (6 * n + 5) / (9 * 4^n) - 4 / 9 :=
sorry

theorem sn_ineq (n : ℕ) : 
  Sn n * Sn (n + 2) < Sn (n + 1)^2 :=
sorry

end sum_of_c_sn_ineq_l8_8170


namespace area_of_circle_outside_triangle_l8_8673

theorem area_of_circle_outside_triangle (AB : ℝ) (r : ℝ) (h_AB : AB = 8) 
  (h_r : r = 2) : 
  let quarter_circle_area := (1/4) * real.pi * r^2 in
  let triangle_area := (1/2) * r^2 in
  quarter_circle_area - triangle_area = real.pi - 2 := 
by
  sorry

end area_of_circle_outside_triangle_l8_8673


namespace trapezoid_circumscribed_radius_l8_8091

variables (a b α : ℝ)

-- Define the expression for the radius of the circumscribed circle
def circumscribed_radius (a b α : ℝ) : ℝ :=
  (Real.sqrt ((b - a)^2 + (b + a)^2 * (Real.tan α)^2)) / (4 * Real.sin α)

theorem trapezoid_circumscribed_radius
  (A B C D : Type)
  (BC AD : ℝ) 
  (angle_CAD : ℝ)
  [circumscribed : CyclicQuadrilateral ABCD]
  (parallel_BC_AD : BC = a ∧ AD = b ∧ ∠ (line_segment C A) (line_segment A D) = α) :
  circumscribed_radius a b α = (Real.sqrt ((b - a)^2 + (b + a)^2 * (Real.tan α)^2)) / (4 * Real.sin α) :=
sorry

end trapezoid_circumscribed_radius_l8_8091
