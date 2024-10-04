import Mathlib

namespace xyz_values_l233_233996

theorem xyz_values (x y z : ℝ)
  (h1 : x * y - 5 * y = 20)
  (h2 : y * z - 5 * z = 20)
  (h3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := 
by sorry

end xyz_values_l233_233996


namespace rectangle_area_change_area_analysis_l233_233961

noncomputable def original_area (a b : ℝ) : ℝ := a * b

noncomputable def new_area (a b : ℝ) : ℝ := (a - 3) * (b + 3)

theorem rectangle_area_change (a b : ℝ) :
  let S := original_area a b
  let S₁ := new_area a b
  S₁ - S = 3 * (a - b - 3) :=
by
  sorry

theorem area_analysis (a b : ℝ) :
  if a - b - 3 = 0 then new_area a b = original_area a b
  else if a - b - 3 > 0 then new_area a b > original_area a b
  else new_area a b < original_area a b :=
by
  sorry

end rectangle_area_change_area_analysis_l233_233961


namespace people_joined_after_leaving_l233_233659

theorem people_joined_after_leaving 
  (p_initial : ℕ) (p_left : ℕ) (p_final : ℕ) (p_joined : ℕ) :
  p_initial = 30 → p_left = 10 → p_final = 25 → p_joined = p_final - (p_initial - p_left) → p_joined = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end people_joined_after_leaving_l233_233659


namespace union_complement_eq_l233_233281

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l233_233281


namespace complement_union_of_M_and_N_l233_233890

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233890


namespace find_value_of_x2001_plus_y2001_l233_233418

theorem find_value_of_x2001_plus_y2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
x ^ 2001 + y ^ 2001 = 2 ^ 2001 ∨ x ^ 2001 + y ^ 2001 = -2 ^ 2001 := by
  sorry

end find_value_of_x2001_plus_y2001_l233_233418


namespace union_complement_l233_233301

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l233_233301


namespace tickets_needed_to_ride_l233_233062

noncomputable def tickets_required : Float :=
let ferris_wheel := 3.5
let roller_coaster := 8.0
let bumper_cars := 5.0
let additional_ride_discount := 0.5
let newspaper_coupon := 1.5
let teacher_discount := 2.0

let total_cost_without_discounts := ferris_wheel + roller_coaster + bumper_cars
let total_additional_discounts := additional_ride_discount * 2
let total_coupons_discounts := newspaper_coupon + teacher_discount

let total_cost_with_discounts := total_cost_without_discounts - total_additional_discounts - total_coupons_discounts
total_cost_with_discounts

theorem tickets_needed_to_ride : tickets_required = 12.0 := by
  sorry

end tickets_needed_to_ride_l233_233062


namespace range_of_a_if_f_decreasing_l233_233563

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (x^2 - a * x + 4)

theorem range_of_a_if_f_decreasing:
  ∀ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
    2 ≤ a ∧ a ≤ 5 :=
by
  intros a h
  sorry

end range_of_a_if_f_decreasing_l233_233563


namespace phantom_additional_money_needed_l233_233331

theorem phantom_additional_money_needed
  (given_money : ℕ)
  (black_inks_cost : ℕ)
  (red_inks_cost : ℕ)
  (yellow_inks_cost : ℕ)
  (blue_inks_cost : ℕ)
  (total_money_needed : ℕ)
  (additional_money_needed : ℕ) :
  given_money = 50 →
  black_inks_cost = 3 * 12 →
  red_inks_cost = 4 * 16 →
  yellow_inks_cost = 3 * 14 →
  blue_inks_cost = 2 * 17 →
  total_money_needed = black_inks_cost + red_inks_cost + yellow_inks_cost + blue_inks_cost →
  additional_money_needed = total_money_needed - given_money →
  additional_money_needed = 126 :=
by
  intros h_given_money h_black h_red h_yellow h_blue h_total h_additional
  sorry

end phantom_additional_money_needed_l233_233331


namespace trees_in_garden_l233_233572

theorem trees_in_garden (yard_length distance_between_trees : ℕ) (h1 : yard_length = 800) (h2 : distance_between_trees = 32) :
  ∃ n : ℕ, n = (yard_length / distance_between_trees) + 1 ∧ n = 26 :=
by
  sorry

end trees_in_garden_l233_233572


namespace verify_total_bill_l233_233624

def fixed_charge : ℝ := 20
def daytime_rate : ℝ := 0.10
def evening_rate : ℝ := 0.05
def free_evening_minutes : ℕ := 200

def daytime_minutes : ℕ := 200
def evening_minutes : ℕ := 300

noncomputable def total_bill : ℝ :=
  fixed_charge + (daytime_minutes * daytime_rate) +
  ((evening_minutes - free_evening_minutes) * evening_rate)

theorem verify_total_bill : total_bill = 45 := by
  sorry

end verify_total_bill_l233_233624


namespace probability_A_not_losing_is_80_percent_l233_233803

def probability_A_winning : ℝ := 0.30
def probability_draw : ℝ := 0.50
def probability_A_not_losing : ℝ := probability_A_winning + probability_draw

theorem probability_A_not_losing_is_80_percent : probability_A_not_losing = 0.80 :=
by 
  sorry

end probability_A_not_losing_is_80_percent_l233_233803


namespace opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l233_233498

theorem opposite_number_of_2_eq_neg2 : -2 = -2 := by
  sorry

theorem abs_val_eq_2_iff_eq_2_or_neg2 (x : ℝ) : abs x = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l233_233498


namespace sequence_general_term_l233_233690

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), a 1 = 2 ^ (5 / 2) ∧ 
  (∀ n, a (n+1) = 4 * (4 * a n) ^ (1/4)) →
  ∀ n, a n = 2 ^ (10 / 3 * (1 - 1 / 4 ^ n)) :=
by
  intros a h1 h_rec
  sorry

end sequence_general_term_l233_233690


namespace A_less_B_C_A_relationship_l233_233838

variable (a : ℝ)
def A := a + 2
def B := 2 * a^2 - 3 * a + 10
def C := a^2 + 5 * a - 3

theorem A_less_B : A a - B a < 0 := by
  sorry

theorem C_A_relationship :
  if a < -5 then C a > A a
  else if a = -5 then C a = A a
  else if a < 1 then C a < A a
  else if a = 1 then C a = A a
  else C a > A a := by
  sorry

end A_less_B_C_A_relationship_l233_233838


namespace root_interval_l233_233840

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h₁ : f 1 < 0) (h₂ : f 1.5 > 0) (h₃ : f 1.25 < 0) (h₄ : f 2 > 0) :
  ∃ x, 1.25 < x ∧ x < 1.5 ∧ f x = 0 :=
sorry

end root_interval_l233_233840


namespace correct_transformation_l233_233793

theorem correct_transformation (x : ℝ) :
  3 + x = 7 ∧ ¬ (x = 7 + 3) ∧
  5 * x = -4 ∧ ¬ (x = -5 / 4) ∧
  (7 / 4) * x = 3 ∧ ¬ (x = 3 * (7 / 4)) ∧
  -((x - 2) / 4) = 1 ∧ (-(x - 2)) = 4 :=
by
  sorry

end correct_transformation_l233_233793


namespace find_initial_mice_l233_233084

theorem find_initial_mice : 
  ∃ x : ℕ, (∀ (h1 : ∀ (m : ℕ), m * 2 = m + m), (35 * x = 280) → x = 8) :=
by
  existsi 8
  intro h1 h2
  sorry

end find_initial_mice_l233_233084


namespace min_performances_l233_233073

theorem min_performances (total_singers : ℕ) (m : ℕ) (n_pairs : ℕ := 28) (pairs_performance : ℕ := 6)
  (condition : total_singers = 108) 
  (const_pairs : ∀ (r : ℕ), (n_pairs * r = pairs_performance * m)) : m ≥ 14 :=
by
  sorry

end min_performances_l233_233073


namespace exists_subset_with_property_P_l233_233746

open Nat

-- Define the property P
def has_property_P (A : Set ℕ) : Prop :=
  ∃ (m : ℕ), ∀ (k : ℕ), k > 0 → ∃ (a : Fin k → ℕ), (∀ j < k-1, 1 ≤ a j.succ - a j ∧ a j.succ - a j ≤ m) ∧ (∀ j < k, a j ∈ A)

-- Main theorem statement
theorem exists_subset_with_property_P (N : Set ℕ) (r : ℕ) (A : Fin r → Set ℕ)
  (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j))
  (h_union : (⋃ i, A i) = N) :
  ∃ i, has_property_P (A i) :=
sorry

end exists_subset_with_property_P_l233_233746


namespace speaker_is_tweedledee_l233_233660

-- Definitions
variable (Speaks : Prop) (is_tweedledum : Prop) (has_black_card : Prop)

-- Condition: If the speaker is Tweedledum, then the card in the speaker's pocket is not a black suit.
axiom A1 : is_tweedledum → ¬ has_black_card

-- Goal: Prove that the speaker is Tweedledee.
theorem speaker_is_tweedledee (h1 : Speaks) : ¬ is_tweedledum :=
by
  sorry

end speaker_is_tweedledee_l233_233660


namespace det_of_matrix_l233_233272

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_of_matrix (h1 : 1 ≤ n)
  (h2 : A ^ 7 + A ^ 5 + A ^ 3 + A - 1 = 0) :
  0 < Matrix.det A :=
sorry

end det_of_matrix_l233_233272


namespace find_m_l233_233608

-- Definitions
variable {A B C O H : Type}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {H_is_altitude_intersection : is_altitude_intersection H A B C}
variable (AH BH CH OA OB OC : ℝ)

-- Problem Statement
theorem find_m (h : AH * BH * CH = m * (OA * OB * OC)) : m = 1 :=
sorry

end find_m_l233_233608


namespace perimeter_of_square_D_l233_233604

-- Definitions based on the conditions in the problem
def square (s : ℝ) := s * s

def perimeter (s : ℝ) := 4 * s

-- Given conditions
def perimeter_C : ℝ := 40
def side_length_C : ℝ := perimeter_C / 4
def area_C : ℝ := square side_length_C
def area_D : ℝ := area_C / 3
def side_length_D : ℝ := real.sqrt area_D

-- Proof statement to be proved
theorem perimeter_of_square_D : perimeter side_length_D = (40 * real.sqrt 3) / 3 := by
  sorry

end perimeter_of_square_D_l233_233604


namespace gifts_wrapped_with_third_roll_l233_233758

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l233_233758


namespace part1_part2_1_part2_2_l233_233026

theorem part1 (n : ℚ) :
  (2 / 2 + n / 5 = (2 + n) / 7) → n = -25 / 2 :=
by sorry

theorem part2_1 (m n : ℚ) :
  (m / 2 + n / 5 = (m + n) / 7) → m = -4 / 25 * n :=
by sorry

theorem part2_2 (m n: ℚ) :
  (m = -4 / 25 * n) → (25 * m + n = 6) → (m = 8 / 25 ∧ n = -2) :=
by sorry

end part1_part2_1_part2_2_l233_233026


namespace lowest_score_l233_233767

theorem lowest_score 
    (mean_15 : ℕ → ℕ → ℕ → ℕ)
    (mean_13 : ℕ → ℕ → ℕ)
    (S15 : ℕ := mean_15 15 85)
    (S13 : ℕ := mean_13 13 87)
    (highest_score : ℕ := 105)
    (S_removed : ℕ := S15 - S13) :
    S_removed - highest_score = 39 := 
sorry

end lowest_score_l233_233767


namespace percentage_students_taking_music_l233_233788

theorem percentage_students_taking_music
  (total_students : ℕ)
  (students_take_dance : ℕ)
  (students_take_art : ℕ)
  (students_take_music : ℕ)
  (percentage_students_taking_music : ℕ) :
  total_students = 400 →
  students_take_dance = 120 →
  students_take_art = 200 →
  students_take_music = total_students - students_take_dance - students_take_art →
  percentage_students_taking_music = (students_take_music * 100) / total_students →
  percentage_students_taking_music = 20 :=
by
  sorry

end percentage_students_taking_music_l233_233788


namespace factorial_expression_evaluation_l233_233529

theorem factorial_expression_evaluation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := 
by 
  sorry

end factorial_expression_evaluation_l233_233529


namespace simplify_expression_l233_233341

theorem simplify_expression (y : ℝ) : 
  3 * y - 5 * y ^ 2 + 12 - (7 - 3 * y + 5 * y ^ 2) = -10 * y ^ 2 + 6 * y + 5 :=
by 
  sorry

end simplify_expression_l233_233341


namespace range_of_f_4_l233_233420

theorem range_of_f_4 {a b c d : ℝ} 
  (h1 : 1 ≤ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ∧ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ≤ 2) 
  (h2 : 1 ≤ a*1^3 + b*1^2 + c*1 + d ∧ a*1^3 + b*1^2 + c*1 + d ≤ 3) 
  (h3 : 2 ≤ a*2^3 + b*2^2 + c*2 + d ∧ a*2^3 + b*2^2 + c*2 + d ≤ 4) 
  (h4 : -1 ≤ a*3^3 + b*3^2 + c*3 + d ∧ a*3^3 + b*3^2 + c*3 + d ≤ 1) :
  -21.75 ≤ a*4^3 + b*4^2 + c*4 + d ∧ a*4^3 + b*4^2 + c*4 + d ≤ 1 :=
sorry

end range_of_f_4_l233_233420


namespace t_lt_s_l233_233417

noncomputable def t : ℝ := Real.sqrt 11 - 3
noncomputable def s : ℝ := Real.sqrt 7 - Real.sqrt 5

theorem t_lt_s : t < s :=
by
  sorry

end t_lt_s_l233_233417


namespace cos_pi_div_4_add_alpha_l233_233107

variable (α : ℝ)

theorem cos_pi_div_4_add_alpha (h : Real.sin (Real.pi / 4 - α) = Real.sqrt 2 / 2) :
  Real.cos (Real.pi / 4 + α) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_pi_div_4_add_alpha_l233_233107


namespace rationalize_denominator_sum_l233_233754

noncomputable def rationalize_denominator (x y z : ℤ) :=
  x = 4 ∧ y = 49 ∧ z = 35 ∧ y ∣ 343 ∧ z > 0 

theorem rationalize_denominator_sum : 
  ∃ A B C : ℤ, rationalize_denominator A B C ∧ A + B + C = 88 :=
by
  sorry

end rationalize_denominator_sum_l233_233754


namespace stephanie_oranges_l233_233163

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l233_233163


namespace min_value_of_abs_diff_l233_233854
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem min_value_of_abs_diff (x1 x2 x : ℝ) (h1 : f x1 ≤ f x) (h2: f x ≤ f x2) : |x1 - x2| = π := by
  sorry

end min_value_of_abs_diff_l233_233854


namespace length_of_AC_l233_233265

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end length_of_AC_l233_233265


namespace complement_of_union_is_singleton_five_l233_233913

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233913


namespace quadratic_complete_the_square_l233_233500

theorem quadratic_complete_the_square :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 + 1500 * x + 1500 = (x + b) ^ 2 + c)
      ∧ b = 750
      ∧ c = -748 * 750
      ∧ c / b = -748 := 
by {
  sorry
}

end quadratic_complete_the_square_l233_233500


namespace unique_solution_k_l233_233091

theorem unique_solution_k (k : ℝ) :
  (∀ x : ℝ, (x + 3) / (k * x + 2) = x) ↔ (k = -1 / 12) :=
  sorry

end unique_solution_k_l233_233091


namespace chapters_per_day_l233_233414

theorem chapters_per_day (chapters : ℕ) (total_days : ℕ) : ℝ :=
  let chapters := 2
  let total_days := 664
  chapters / total_days

example : chapters_per_day 2 664 = 2 / 664 := by sorry

end chapters_per_day_l233_233414


namespace proof_problem_l233_233693

-- Conditions for the ellipse
def ellipse_coefs (a b : ℝ) := 0 < b ∧ b < a 
def focal_length (a b : ℝ) := (a^2 - b^2 = 8)
def minor_semi_axis (b : ℝ) := b = 2

-- Equation of the Ellipse
noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, ellipse_coefs a b ∧ focal_length a b ∧ minor_semi_axis b ∧ 
    (∀ x y : ℝ, ((x^2/a^2 + y^2/b^2 = 1) ↔ ((x^2/12 + y^2/4 = 1)))

-- Line passing through point P with slope 1
noncomputable def line_through_p (y x : ℝ) : Prop := 
  (∀ x y : ℝ, (y = x + 3 ↔ ((x+2)*(x+2) + 1 = y)))

-- Length of the Chord
noncomputable def chord_length : ℝ :=
  ∃ (x1 x2 : ℝ), (-6) * (253 / 286) = 0 ∧ 4 * ((x1 * x2) - (x1 / 2 * x2 * y)) = 42

-- Combined, for readability
theorem proof_problem : Prop :=
  ellipse_equation ∧ line_through_p ∧ chord_length = (sqrt 42 / 2)

end proof_problem_l233_233693


namespace s_plough_time_l233_233068

theorem s_plough_time (r_s_combined_time : ℝ) (r_time : ℝ) (t_time : ℝ) (s_time : ℝ) :
  r_s_combined_time = 10 → r_time = 15 → t_time = 20 → s_time = 30 :=
by
  sorry

end s_plough_time_l233_233068


namespace hyperbola_asymptotes_n_l233_233773

theorem hyperbola_asymptotes_n {y x : ℝ} (n : ℝ) (H : ∀ x y, (y^2 / 16) - (x^2 / 9) = 1 → y = n * x ∨ y = -n * x) : n = 4/3 :=
  sorry

end hyperbola_asymptotes_n_l233_233773


namespace geom_seq_common_ratio_l233_233985

theorem geom_seq_common_ratio (S_3 S_6 : ℕ) (h1 : S_3 = 7) (h2 : S_6 = 63) : 
  ∃ q : ℕ, q = 2 := 
by
  sorry

end geom_seq_common_ratio_l233_233985


namespace price_adjustment_l233_233079

theorem price_adjustment (P : ℝ) (x : ℝ) (hx : P * (1 - (x / 100)^2) = 0.75 * P) : 
  x = 50 :=
by
  -- skipping the proof with sorry
  sorry

end price_adjustment_l233_233079


namespace no_integer_solutions_l233_233100

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
  sorry

end no_integer_solutions_l233_233100


namespace marble_sharing_l233_233747

theorem marble_sharing 
  (total_marbles : ℕ) 
  (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 30) 
  (h2 : marbles_per_friend = 6) : 
  total_marbles / marbles_per_friend = 5 := 
by 
  sorry

end marble_sharing_l233_233747


namespace problem_l233_233291

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l233_233291


namespace shirts_sold_l233_233388

theorem shirts_sold (initial_shirts remaining_shirts shirts_sold : ℕ) (h1 : initial_shirts = 49) (h2 : remaining_shirts = 28) : 
  shirts_sold = initial_shirts - remaining_shirts → 
  shirts_sold = 21 := 
by 
  sorry

end shirts_sold_l233_233388


namespace number_of_ways_to_fill_grid_with_conditions_l233_233406

/-- 
Number of ways to arrange numbers 1 to 9 in a 3x3 matrix, where each row is increasing
from left to right and each column is decreasing from top to bottom, is 42.
------------------ -/
theorem number_of_ways_to_fill_grid_with_conditions : 
  let grid := Matrix (Fin 3) (Fin 3) Nat
  ∃ (m : grid),
    (∀ i : Fin 3, StrictMono (λ j : Fin 3 => m i j)) ∧
    (∀ j : Fin 3, StrictAnti (λ i : Fin 3 => m i j)) ∧
    (∀ v : Fin 9, ∃ i j, m i j = v.succ) →
    ∃! (n : Nat), n = 42 := 
sorry

end number_of_ways_to_fill_grid_with_conditions_l233_233406


namespace average_breadth_of_plot_l233_233444

theorem average_breadth_of_plot :
  ∃ B L : ℝ, (L - B = 10) ∧ (23 * B = (1/2) * (L + B) * B) ∧ (B = 18) :=
by
  sorry

end average_breadth_of_plot_l233_233444


namespace cell_X_is_Red_l233_233340

-- Define the colours as an enumeration type
inductive Colour
| Red
| Blue
| Yellow
| Green

open Colour

-- Define a function representing the grid colouring constraint
def vertex_colour_constraint (grid : ℕ → ℕ → Colour) : Prop :=
  ∀ x y, (∀ dx dy, ((dx ≠ 0) ∨ (dy ≠ 0)) → grid x y ≠ grid (x + dx) (y + dy))

-- Assume a predefined partially coloured grid (not specified in the problem)
-- For example purpose, let's assume: 
-- grid 0 0 = Red, grid 0 1 = Blue, grid 1 0 = Green,
-- The implementation detail of the grid is abstracted away for simplicity here.

noncomputable def partially_coloured_grid : ℕ → ℕ → Colour := sorry

-- The final goal is to determine the colour of cell marked X
def cell_X : ℕ × ℕ := (2, 2)  -- Marking cell X at coordinates (2, 2) for example

-- The hypothesized grid function should satisfy the constraints
axiom correct_grid : vertex_colour_constraint partially_coloured_grid

-- Prove that the colour of cell X is Red
theorem cell_X_is_Red : partially_coloured_grid cell_X.fst cell_X.snd = Red :=
by
  -- The proof steps would be here, but they are omitted since we only need the statement
  sorry

end cell_X_is_Red_l233_233340


namespace infinite_primes_of_form_m2_mn_n2_l233_233598

theorem infinite_primes_of_form_m2_mn_n2 : ∀ m n : ℤ, ∃ p : ℕ, ∃ k : ℕ, (p = k^2 + k * m + n^2) ∧ Prime k :=
sorry

end infinite_primes_of_form_m2_mn_n2_l233_233598


namespace min_value_PQ_l233_233726

variable (t : ℝ) (x y : ℝ)

-- Parametric equations of line l
def line_l : Prop := (x = 4 * t - 1) ∧ (y = 3 * t - 3 / 2)

-- Polar equation of circle C
def polar_eq_circle_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4)

-- General equation of line l
def general_eq_line_l (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Rectangular equation of circle C
def rectangular_eq_circle_c (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Definition of the condition where P is on line l
def p_on_line_l (x y : ℝ) : Prop := ∃ t : ℝ, line_l t x y

-- Minimum value of |PQ|
theorem min_value_PQ :
  p_on_line_l x y →
  general_eq_line_l x y →
  rectangular_eq_circle_c x y →
  ∃ d : ℝ, d = Real.sqrt 2 :=
by intros; sorry

end min_value_PQ_l233_233726


namespace volume_ratio_spheres_l233_233777

theorem volume_ratio_spheres (r1 r2 r3 v1 v2 v3 : ℕ)
  (h_rad_ratio : r1 = 1 ∧ r2 = 2 ∧ r3 = 3)
  (h_vol_ratio : v1 = r1^3 ∧ v2 = r2^3 ∧ v3 = r3^3) :
  v3 = 3 * (v1 + v2) := by
  -- main proof goes here
  sorry

end volume_ratio_spheres_l233_233777


namespace truck_sand_at_arrival_l233_233083

-- Definitions based on conditions in part a)
def initial_sand : ℝ := 4.1
def lost_sand : ℝ := 2.4

-- Theorem statement corresponding to part c)
theorem truck_sand_at_arrival : initial_sand - lost_sand = 1.7 :=
by
  -- "sorry" placeholder to skip the proof
  sorry

end truck_sand_at_arrival_l233_233083


namespace marcus_scored_50_percent_l233_233320

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l233_233320


namespace investment_duration_l233_233231

theorem investment_duration 
  (P SI R : ℕ) (T : ℕ) 
  (hP : P = 800) 
  (hSI : SI = 128) 
  (hR : R = 4) 
  (h : SI = P * R * T / 100) 
  : T = 4 :=
by 
  rw [hP, hSI, hR] at h
  sorry

end investment_duration_l233_233231


namespace initial_men_count_l233_233162

theorem initial_men_count (M : ℕ) (P : ℕ) :
  P = M * 20 →
  P = (M + 650) * 109 / 9 →
  M = 1000 :=
by
  sorry

end initial_men_count_l233_233162


namespace complement_union_M_N_l233_233950

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233950


namespace three_days_earning_l233_233013

theorem three_days_earning
  (charge : ℤ := 2)
  (day_before_yesterday_wash : ℤ := 5)
  (yesterday_wash : ℤ := day_before_yesterday_wash + 5)
  (today_wash : ℤ := 2 * yesterday_wash)
  (three_days_earning : ℤ := charge * (day_before_yesterday_wash + yesterday_wash + today_wash)) :
  three_days_earning = 70 := 
by
  have h1 : day_before_yesterday_wash = 5 := by rfl
  have h2 : yesterday_wash = day_before_yesterday_wash + 5 := by rfl
  have h3 : today_wash = 2 * yesterday_wash := by rfl
  have h4 : charge * (day_before_yesterday_wash + yesterday_wash + today_wash) = 70 := sorry
  exact h4

end three_days_earning_l233_233013


namespace not_divisible_by_121_l233_233476

theorem not_divisible_by_121 (n : ℤ) : ¬ 121 ∣ (n^2 + 3 * n + 5) :=
sorry

end not_divisible_by_121_l233_233476


namespace complement_union_l233_233901

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233901


namespace syllogism_arrangement_l233_233369

theorem syllogism_arrangement : 
  (∀ n : ℕ, Odd n → ¬ (n % 2 = 0)) → 
  Odd 2013 → 
  (¬ (2013 % 2 = 0)) :=
by
  intros h1 h2
  exact h1 2013 h2

end syllogism_arrangement_l233_233369


namespace square_of_binomial_is_25_l233_233973

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l233_233973


namespace axis_of_symmetry_range_of_t_l233_233982

section
variables (a b m n p t : ℝ)

-- Assume the given conditions
def parabola (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Part (1): Find the axis of symmetry
theorem axis_of_symmetry (h_a_pos : a > 0) 
    (hM : parabola a b 2 = m) 
    (hN : parabola a b 4 = n) 
    (hmn : m = n) : 
    -b / (2 * a) = 3 := 
  sorry

-- Part (2): Find the range of values for t
theorem range_of_t (h_a_pos : a > 0) 
    (hP : parabola a b (-1) = p)
    (axis : -b / (2 * a) = t) 
    (hmn_neg : m * n < 0) 
    (hmpn : m < p ∧ p < n) :
    1 < t ∧ t < 3 / 2 := 
  sorry
end

end axis_of_symmetry_range_of_t_l233_233982


namespace union_complement_eq_l233_233295

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l233_233295


namespace number_of_men_is_15_l233_233379

-- Define the conditions
def number_of_people : Prop :=
  ∃ (M W B : ℕ), M = 8 ∧ W = 8 ∧ B = 8 ∧ 8 * M = 120

-- Define the final statement to be proven
theorem number_of_men_is_15 (h: number_of_people) : ∃ M : ℕ, M = 15 :=
by
  obtain ⟨M, W, B, hM, hW, hB, htotal⟩ := h
  use M
  rw [hM] at htotal
  have hM15 : M = 15 := by linarith
  exact hM15

end number_of_men_is_15_l233_233379


namespace dvds_bought_online_l233_233007

theorem dvds_bought_online (total_dvds : ℕ) (store_dvds : ℕ) (online_dvds : ℕ) :
  total_dvds = 10 → store_dvds = 8 → online_dvds = total_dvds - store_dvds → online_dvds = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvds_bought_online_l233_233007


namespace cylinder_height_l233_233038

variable (r h : ℝ) (SA : ℝ)

theorem cylinder_height (h : ℝ) (r : ℝ) (SA : ℝ) (h_eq : h = 2) (r_eq : r = 3) (SA_eq : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h → h = 2 :=
by
  intros
  sorry

end cylinder_height_l233_233038


namespace complement_union_l233_233885

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233885


namespace throwers_count_l233_233152

variable (totalPlayers : ℕ) (rightHandedPlayers : ℕ) (nonThrowerLeftHandedFraction nonThrowerRightHandedFraction : ℚ)

theorem throwers_count
  (h1 : totalPlayers = 70)
  (h2 : rightHandedPlayers = 64)
  (h3 : nonThrowerLeftHandedFraction = 1 / 3)
  (h4 : nonThrowerRightHandedFraction = 2 / 3)
  (h5 : nonThrowerLeftHandedFraction + nonThrowerRightHandedFraction = 1) : 
  ∃ T : ℕ, T = 52 := by
  sorry

end throwers_count_l233_233152


namespace orange_balls_count_l233_233213

-- Define the constants
constant total_balls : ℕ := 50
constant red_balls : ℕ := 20
constant blue_balls : ℕ := 10

-- Define the conditions
axiom total_parts : total_balls = red_balls + blue_balls + (total_balls - red_balls - blue_balls)
axiom pink_or_orange_balls : total_balls - red_balls - blue_balls = 20
axiom pink_is_three_times_orange {O P : ℕ} : P = 3 * O
axiom sum_pink_orange {O P : ℕ} : P + O = 20

-- Main statement to prove
theorem orange_balls_count : ∃ O : ℕ, ∀ P : ℕ, P = 3 * O → P + O = 20 → O = 5 :=
by
  sorry

end orange_balls_count_l233_233213


namespace intersection_A_B_l233_233115

def A : Set ℝ := { x | Real.log x > 0 }

def B : Set ℝ := { x | Real.exp x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < Real.log 3 / Real.log 2 } :=
sorry

end intersection_A_B_l233_233115


namespace complement_of_union_is_singleton_five_l233_233917

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233917


namespace log_identity_l233_233662

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_identity :
    2 * log_base_10 2 + log_base_10 (5 / 8) - log_base_10 25 = -1 :=
by 
  sorry

end log_identity_l233_233662


namespace son_age_is_26_l233_233798

-- Definitions based on conditions in the problem
variables (S F : ℕ)
axiom cond1 : F = S + 28
axiom cond2 : F + 2 = 2 * (S + 2)

-- Statement to prove that S = 26
theorem son_age_is_26 : S = 26 :=
by 
  -- Proof steps go here
  sorry

end son_age_is_26_l233_233798


namespace evaluate_polynomial_at_2_l233_233537

def polynomial (x : ℝ) := x^2 + 5*x - 14

theorem evaluate_polynomial_at_2 : polynomial 2 = 0 := by
  sorry

end evaluate_polynomial_at_2_l233_233537


namespace proof_l233_233686

variable (p : ℕ) (ε : ℤ)
variable (RR NN NR RN : ℕ)

-- Conditions
axiom h1 : ∀ n ≤ p - 2, 
  (n % 2 = 0 ∧ (n + 1) % 2 = 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 ≠ 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 = 0 ) ∨ 
  (n % 2 = 0 ∧ (n + 1) % 2 ≠ 0) 

axiom h2 :  RR + NN - RN - NR = 1

axiom h3 : ε = (-1) ^ ((p - 1) / 2)

axiom h4 : RR + RN = (p - 2 - ε) / 2

axiom h5 : RR + NR = (p - 1) / 2 - 1

axiom h6 : NR + NN = (p - 2 + ε) / 2

axiom h7 : RN + NN = (p - 1) / 2  

-- To prove
theorem proof : 
  RR = (p / 4) - (ε + 4) / 4 ∧ 
  RN = (p / 4) - (ε) / 4 ∧ 
  NN = (p / 4) + (ε - 2) / 4 ∧ 
  NR = (p / 4) + (ε - 2) / 4 := 
sorry

end proof_l233_233686


namespace smallest_multiple_36_45_not_11_l233_233189

theorem smallest_multiple_36_45_not_11 (n : ℕ) :
  (n = 180) ↔ (n > 0 ∧ (36 ∣ n) ∧ (45 ∣ n) ∧ ¬ (11 ∣ n)) :=
by
  sorry

end smallest_multiple_36_45_not_11_l233_233189


namespace solve_eq1_solve_system_l233_233156

theorem solve_eq1 : ∃ x y : ℝ, (3 / x) + (2 / y) = 4 :=
by
  use 1
  use 2
  sorry

theorem solve_system :
  ∃ x y : ℝ,
    (3 / x + 2 / y = 4) ∧ (5 / x - 6 / y = 2) ∧ (x = 1) ∧ (y = 2) :=
by
  use 1
  use 2
  sorry

end solve_eq1_solve_system_l233_233156


namespace find_a_l233_233772

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.sin x

theorem find_a (a : ℝ) : (∀ f', f' = (fun x => a * Real.exp x - Real.cos x) → f' 0 = 0) → a = 1 :=
by
  intros h
  specialize h (fun x => a * Real.exp x - Real.cos x) rfl
  sorry  -- proof is omitted

end find_a_l233_233772


namespace distance_between_lines_l233_233354

noncomputable def distance_between_parallel_lines_eq : Prop :=
  let line1 (x y : ℝ) := (2 * x - y = 0)
  let line2 (x y : ℝ) := (2 * x - y + 5 = 0)

  let distance : ℝ := |0 - 5| / real.sqrt (2^2 + (-1)^2)

  distance = real.sqrt 5

theorem distance_between_lines :
  distance_between_parallel_lines_eq := by
  sorry

end distance_between_lines_l233_233354


namespace polynomial_no_positive_real_roots_l233_233490

theorem polynomial_no_positive_real_roots : 
  ¬ ∃ x : ℝ, x > 0 ∧ x^3 + 6 * x^2 + 11 * x + 6 = 0 :=
sorry

end polynomial_no_positive_real_roots_l233_233490


namespace rates_of_interest_l233_233206

theorem rates_of_interest (P_B P_C T_B T_C SI_B SI_C : ℝ) (R_B R_C : ℝ)
  (hB1 : P_B = 5000) (hB2: T_B = 5) (hB3: SI_B = 2200)
  (hC1 : P_C = 3000) (hC2 : T_C = 7) (hC3 : SI_C = 2730)
  (simple_interest : ∀ {P R T SI : ℝ}, SI = (P * R * T) / 100)
  : R_B = 8.8 ∧ R_C = 13 := by
  sorry

end rates_of_interest_l233_233206


namespace complement_union_l233_233936

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233936


namespace degree_not_determined_from_characteristic_l233_233989

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l233_233989


namespace more_money_from_mom_is_correct_l233_233412

noncomputable def more_money_from_mom : ℝ :=
  let money_from_mom := 8.25
  let money_from_dad := 6.50
  let money_from_grandparents := 12.35
  let money_from_aunt := 5.10
  let money_spent_toy := 4.45
  let money_spent_snacks := 6.25
  let total_received := money_from_mom + money_from_dad + money_from_grandparents + money_from_aunt
  let total_spent := money_spent_toy + money_spent_snacks
  let money_remaining := total_received - total_spent
  let money_spent_books := 0.25 * money_remaining
  let money_left_after_books := money_remaining - money_spent_books
  money_from_mom - money_from_dad

theorem more_money_from_mom_is_correct : more_money_from_mom = 1.75 := by
  sorry

end more_money_from_mom_is_correct_l233_233412


namespace find_share_of_A_l233_233796

noncomputable def investment_share_A (initial_investment_A initial_investment_B withdraw_A add_B after_months end_of_year_profit : ℝ) : ℝ :=
  let investment_months_A := (initial_investment_A * after_months) + ((initial_investment_A - withdraw_A) * (12 - after_months))
  let investment_months_B := (initial_investment_B * after_months) + ((initial_investment_B + add_B) * (12 - after_months))
  let total_investment_months := investment_months_A + investment_months_B
  let ratio_A := investment_months_A / total_investment_months
  ratio_A * end_of_year_profit

theorem find_share_of_A : 
  investment_share_A 3000 4000 1000 1000 8 630 = 240 := 
by 
  sorry

end find_share_of_A_l233_233796


namespace remaining_ribbon_l233_233002

-- Definitions of the conditions
def total_ribbon : ℕ := 18
def gifts : ℕ := 6
def ribbon_per_gift : ℕ := 2

-- The statement to prove the remaining ribbon
theorem remaining_ribbon 
  (initial_ribbon : ℕ) (num_gifts : ℕ) (ribbon_each_gift : ℕ) 
  (H1 : initial_ribbon = total_ribbon) 
  (H2 : num_gifts = gifts) 
  (H3 : ribbon_each_gift = ribbon_per_gift) : 
  initial_ribbon - (ribbon_each_gift * num_gifts) = 6 := 
  by 
    simp [H1, H2, H3, total_ribbon, gifts, ribbon_per_gift]
    linarith
    sorry 

end remaining_ribbon_l233_233002


namespace homework_duration_decrease_l233_233188

variable (a b x : ℝ)

theorem homework_duration_decrease (h: a * (1 - x)^2 = b) :
  a * (1 - x)^2 = b := 
by
  sorry

end homework_duration_decrease_l233_233188


namespace four_positive_reals_inequality_l233_233333

theorem four_positive_reals_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + b^2 * c + c^2 * d + d^2 * a :=
sorry

end four_positive_reals_inequality_l233_233333


namespace distance_range_l233_233175

theorem distance_range (A_school_distance : ℝ) (B_school_distance : ℝ) (x : ℝ)
  (hA : A_school_distance = 3) (hB : B_school_distance = 2) :
  1 ≤ x ∧ x ≤ 5 :=
sorry

end distance_range_l233_233175


namespace find_a_l233_233697

theorem find_a (a : ℝ) : (4, -5).2 = (a - 2, a + 1).2 → a = -6 :=
by
  intro h
  sorry

end find_a_l233_233697


namespace jane_wins_prob_correct_l233_233209

-- Definitions based on conditions
def sectors : Finset ℕ := {1, 2, 3, 4, 5, 6}
def possible_outcomes := sectors.product sectors
def losing_conditions : Finset (ℕ × ℕ) := {(1, 5), (1, 6), (2, 6), (5, 1), (6, 1), (6, 2)}

-- noncomputable and proof skeleton
noncomputable def jane_wins_probability : ℚ :=
  1 - (losing_conditions.card / possible_outcomes.card : ℚ)

theorem jane_wins_prob_correct : jane_wins_probability = 5 / 6 :=
by
  -- proof steps will be filled here
  sorry

end jane_wins_prob_correct_l233_233209


namespace count_teams_of_6_l233_233362

theorem count_teams_of_6 
  (students : Fin 12 → Type)
  (played_together_once : ∀ (s : Finset (Fin 12)) (h : s.card = 5), ∃! t : Finset (Fin 12), t.card = 6 ∧ s ⊆ t) :
  (∃ team_count : ℕ, team_count = 132) :=
by
  -- Proof omitted
  sorry

end count_teams_of_6_l233_233362


namespace union_complement_l233_233302

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l233_233302


namespace range_of_m_l233_233714

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m+1)*x^2 + (m+1)*x + (m+2) ≥ 0) ↔ m ≥ -1 := by
  sorry

end range_of_m_l233_233714


namespace length_of_AC_l233_233264

/-- 
Given AB = 15 cm, DC = 24 cm, and AD = 7 cm, 
prove that the length of AC to the nearest tenth of a centimeter is 30.3 cm.
-/
theorem length_of_AC {A B C D : Point} 
  (hAB : dist A B = 15) (hDC : dist D C = 24) (hAD : dist A D = 7) : 
  dist A C ≈ 30.3 :=
sorry

end length_of_AC_l233_233264


namespace complement_union_l233_233910

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233910


namespace parabola_transformation_l233_233491

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def shifted_left (x : ℝ) : ℝ := original_parabola (x + 1)

def shifted_down (x : ℝ) : ℝ := shifted_left x - 2

theorem parabola_transformation :
  shifted_down x = 3 * (x + 1)^2 - 2 :=
sorry

end parabola_transformation_l233_233491


namespace sin_X_value_l233_233268

variables (a b X : ℝ)

-- Conditions
def conditions :=
  (1/2 * a * b * Real.sin X = 100) ∧ (Real.sqrt (a * b) = 15)

theorem sin_X_value (h : conditions a b X) : Real.sin X = 8 / 9 := by
  sorry

end sin_X_value_l233_233268


namespace time_to_cross_pole_is_correct_l233_233141

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end time_to_cross_pole_is_correct_l233_233141


namespace chair_cost_l233_233221

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l233_233221


namespace heidi_and_liam_paint_in_15_minutes_l233_233564

-- Definitions
def Heidi_rate : ℚ := 1 / 60
def Liam_rate : ℚ := 1 / 90
def combined_rate : ℚ := Heidi_rate + Liam_rate
def painting_time : ℚ := 15

-- Theorem to Prove
theorem heidi_and_liam_paint_in_15_minutes : painting_time * combined_rate = 5 / 12 := by
  sorry

end heidi_and_liam_paint_in_15_minutes_l233_233564


namespace max_cos2_sinx_l233_233178

noncomputable def cos2_sinx (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_cos2_sinx : ∃ x : ℝ, cos2_sinx x = 5 / 4 := 
by
  existsi (Real.arcsin (-1 / 2))
  rw [cos2_sinx]
  -- We need further steps to complete the proof
  sorry

end max_cos2_sinx_l233_233178


namespace sequence_term_n_l233_233724

theorem sequence_term_n (a : ℕ → ℕ) (a1 d : ℕ) (n : ℕ) (h1 : a 1 = a1) (h2 : d = 2)
  (h3 : a n = 19) (h_seq : ∀ n, a n = a1 + (n - 1) * d) : n = 10 :=
by
  sorry

end sequence_term_n_l233_233724


namespace cost_price_computer_table_l233_233039

theorem cost_price_computer_table (S : ℝ) (C : ℝ) (h1 : S = C * 1.15) (h2 : S = 5750) : C = 5000 :=
by
  sorry

end cost_price_computer_table_l233_233039


namespace complement_union_l233_233906

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233906


namespace nets_win_in_7_games_probability_l233_233137

theorem nets_win_in_7_games_probability :
  let warriors_win_prob := (1 : ℚ) / 4
  let nets_win_prob := (3 : ℚ) / 4
  let binom_coeff := Nat.choose 6 3
  let game_6_warriors_prob := (warriors_win_prob ^ 3) * (nets_win_prob ^ 3)
  let prob_before_game_7 := binom_coeff * game_6_warriors_prob
  let final_prob := prob_before_game_7 * nets_win_prob
  final_prob = 405 / 4096 :=
by
  let warriors_win_prob := (1 : ℚ) / 4
  let nets_win_prob := (3 : ℚ) / 4
  let binom_coeff := Nat.choose 6 3
  let game_6_warriors_prob := (warriors_win_prob ^ 3) * (nets_win_prob ^ 3)
  let prob_before_game_7 := binom_coeff * game_6_warriors_prob
  let final_prob := prob_before_game_7 * nets_win_prob
  have h1 : binom_coeff = 20 := by sorry
  have h2 : game_6_warriors_prob = 27 / 4096 := by sorry
  have h3 : prob_before_game_7 = 540 / 4096 := by sorry
  have h4 : final_prob = (540 / 4096) * (3 / 4) := by sorry
  have h5 : final_prob = 405 / 4096 := by sorry
  exact h5

end nets_win_in_7_games_probability_l233_233137


namespace solution_l233_233740

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l233_233740


namespace gcd_1407_903_l233_233507

theorem gcd_1407_903 : Nat.gcd 1407 903 = 21 := 
  sorry

end gcd_1407_903_l233_233507


namespace solve_inequality_1_solve_inequality_2_l233_233433

-- Definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ := abs (x + 1)

-- Lean statement for the first proof problem
theorem solve_inequality_1 :
  ∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Lean statement for the second proof problem
theorem solve_inequality_2 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 2 * f x + abs (x + a) ≤ x + 4) ↔ -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end solve_inequality_1_solve_inequality_2_l233_233433


namespace largest_divisor_n4_minus_n2_l233_233824

theorem largest_divisor_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_n4_minus_n2_l233_233824


namespace find_a_l233_233407

theorem find_a (r s : ℚ) (a : ℚ) :
  (∀ x : ℚ, (ax^2 + 18 * x + 16 = (r * x + s)^2)) → 
  s = 4 ∨ s = -4 →
  a = (9 / 4) * (9 / 4)
:= sorry

end find_a_l233_233407


namespace find_300th_term_excl_squares_l233_233627

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l233_233627


namespace range_of_a_l233_233855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a * x - 1 else a / x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def func_increasing_on_R (a : ℝ) : Prop :=
  is_increasing_on (f a) Set.univ

theorem range_of_a (a : ℝ) : func_increasing_on_R a ↔ a < -2 :=
sorry

end range_of_a_l233_233855


namespace fewest_tiles_needed_l233_233650

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end fewest_tiles_needed_l233_233650


namespace marcus_percentage_of_team_points_l233_233323

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l233_233323


namespace average_speed_round_trip_l233_233525

theorem average_speed_round_trip
  (n : ℕ)
  (distance_km : ℝ := n / 1000)
  (pace_west_min_per_km : ℝ := 2)
  (speed_east_kmh : ℝ := 3)
  (wait_time_hr : ℝ := 30 / 60) :
  (2 * distance_km) / 
  ((pace_west_min_per_km * distance_km / 60) + wait_time_hr + (distance_km / speed_east_kmh)) = 
  60 * n / (11 * n + 150000) := by
  sorry

end average_speed_round_trip_l233_233525


namespace complement_of_union_is_singleton_five_l233_233918

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233918


namespace degree_not_determined_by_A_P_l233_233987

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l233_233987


namespace solve_equation_l233_233761

theorem solve_equation (x : ℝ) : 
  (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = -3 - Real.sqrt 6) ↔ 
  (x^4 / (2 * x + 1) + x^2 = 6 * (2 * x + 1)) := by
  sorry

end solve_equation_l233_233761


namespace triangle_inequality_equality_condition_l233_233737

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_equality_condition_l233_233737


namespace term_omit_perfect_squares_300_l233_233629

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l233_233629


namespace number_of_petri_dishes_l233_233574

noncomputable def total_germs : ℝ := 0.036 * 10^5
noncomputable def germs_per_dish : ℝ := 99.99999999999999

theorem number_of_petri_dishes : 36 = total_germs / germs_per_dish :=
by sorry

end number_of_petri_dishes_l233_233574


namespace find_difference_l233_233067

variables (a b c : ℝ)

theorem find_difference (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 := by
  sorry

end find_difference_l233_233067


namespace probability_not_siblings_l233_233641

-- Define the number of people and the sibling condition
def number_of_people : ℕ := 6
def siblings_count (x : Fin number_of_people) : ℕ := 2

-- Define the probability that two individuals randomly selected are not siblings
theorem probability_not_siblings (P Q : Fin number_of_people) (h : P ≠ Q) :
  let K := number_of_people - 1
  let non_siblings := K - siblings_count P
  (non_siblings / K : ℚ) = 3 / 5 :=
by
  intros
  sorry

end probability_not_siblings_l233_233641


namespace prime_divisors_difference_l233_233585

def prime_factors (n : ℕ) : ℕ := sorry -- definition placeholder

theorem prime_divisors_difference (n : ℕ) (hn : 0 < n) : 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ k - m = n ∧ prime_factors k - prime_factors m = 1 := 
sorry

end prime_divisors_difference_l233_233585


namespace team_arrangements_l233_233718

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem team_arrangements :
  let num_players := 10
  let team_blocks := 4
  let cubs_players := 3
  let red_sox_players := 3
  let yankees_players := 2
  let dodgers_players := 2
  (factorial team_blocks) * (factorial cubs_players) * (factorial red_sox_players) * (factorial yankees_players) * (factorial dodgers_players) = 3456 := 
by
  -- Proof steps will be inserted here
  sorry

end team_arrangements_l233_233718


namespace divisible_by_3_l233_233099

theorem divisible_by_3 (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end divisible_by_3_l233_233099


namespace union_complement_eq_l233_233304

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l233_233304


namespace example_theorem_l233_233865

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233865


namespace problem1_solution_problem2_solution_l233_233664

theorem problem1_solution (x : ℝ) : (x^2 - 4 * x = 5) → (x = 5 ∨ x = -1) :=
by sorry

theorem problem2_solution (x : ℝ) : (2 * x^2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1/2) :=
by sorry

end problem1_solution_problem2_solution_l233_233664


namespace example_theorem_l233_233859

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233859


namespace external_bisector_l233_233467

open Triangle

variable {Point : Type} [EuclideanGeometry Point]

variables {A T C L K : Point}

theorem external_bisector (h : angle_bisector A T C L) : external_bisector A T C K :=
sorry

end external_bisector_l233_233467


namespace equilateral_triangle_ratio_l233_233510

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let perimeter := 3 * s
  let area := (s * s * Real.sqrt 3) / 4
  perimeter / area = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l233_233510


namespace value_of_t_l233_233975

theorem value_of_t (k : ℤ) (t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 68) : t = 20 :=
by
  sorry

end value_of_t_l233_233975


namespace am_gm_inequality_l233_233959

variable {x y z : ℝ}

theorem am_gm_inequality (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y + z) / 3 ≥ Real.sqrt (Real.sqrt (x * y) * Real.sqrt z) :=
by
  sorry

end am_gm_inequality_l233_233959


namespace longest_side_range_of_obtuse_triangle_l233_233260

theorem longest_side_range_of_obtuse_triangle (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  a^2 + b^2 < c^2 → (Real.sqrt 5 < c ∧ c < 3) ∨ c = 2 :=
by
  sorry

end longest_side_range_of_obtuse_triangle_l233_233260


namespace prod_72516_9999_l233_233684

theorem prod_72516_9999 : 72516 * 9999 = 724987484 :=
by
  sorry

end prod_72516_9999_l233_233684


namespace expression_simplifies_to_62_l233_233097

theorem expression_simplifies_to_62 (a b c : ℕ) (h1 : a = 14) (h2 : b = 19) (h3 : c = 29) :
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 62 := by {
  sorry -- Proof goes here
}

end expression_simplifies_to_62_l233_233097


namespace simon_paid_amount_l233_233602

theorem simon_paid_amount:
  let pansy_price := 2.50
  let hydrangea_price := 12.50
  let petunia_price := 1.00
  let pansies_count := 5
  let hydrangeas_count := 1
  let petunias_count := 5
  let discount_rate := 0.10
  let change_received := 23.00

  let total_cost_before_discount := (pansies_count * pansy_price) + (hydrangeas_count * hydrangea_price) + (petunias_count * petunia_price)
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  let amount_paid_with := total_cost_after_discount + change_received

  amount_paid_with = 50.00 :=
by
  sorry

end simon_paid_amount_l233_233602


namespace nabla_2_3_2_eq_4099_l233_233533

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem nabla_2_3_2_eq_4099 : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end nabla_2_3_2_eq_4099_l233_233533


namespace find_radius_of_circle_B_l233_233820

noncomputable def radius_of_circle_B : Real :=
  sorry

theorem find_radius_of_circle_B :
  let A := 2
  let R := 4
  -- Define x as the horizontal distance (FG) and y as the vertical distance (GH)
  ∃ (x y : Real), 
  (y = x + (x^2 / 2)) ∧
  (y = 2 - (x^2 / 4)) ∧
  (5 * x^2 + 4 * x - 8 = 0) ∧
  -- Contains only the positive solution among possible valid radii
  (radius_of_circle_B = (22 / 25) + (2 * Real.sqrt 11 / 25))
:= 
sorry

end find_radius_of_circle_B_l233_233820


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l233_233235

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l233_233235


namespace arithmetic_sequence_a2_value_l233_233242

open Nat

theorem arithmetic_sequence_a2_value (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 + a 3 = 12) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) : 
  a 2 = 5 :=
  sorry

end arithmetic_sequence_a2_value_l233_233242


namespace FirstCandidatePercentage_l233_233200

noncomputable def percentage_of_first_candidate_marks (PassingMarks TotalMarks MarksFirstCandidate : ℝ) :=
  (MarksFirstCandidate / TotalMarks) * 100

theorem FirstCandidatePercentage 
  (PassingMarks TotalMarks MarksFirstCandidate : ℝ)
  (h1 : PassingMarks = 200)
  (h2 : 0.45 * TotalMarks = PassingMarks + 25)
  (h3 : MarksFirstCandidate = PassingMarks - 50)
  : percentage_of_first_candidate_marks PassingMarks TotalMarks MarksFirstCandidate = 30 :=
sorry

end FirstCandidatePercentage_l233_233200


namespace expand_polynomial_l233_233538

variable {R : Type*} [CommRing R]

theorem expand_polynomial (x : R) : (2 * x + 3) * (x + 6) = 2 * x^2 + 15 * x + 18 := 
sorry

end expand_polynomial_l233_233538


namespace sin_cos_equiv_l233_233106

theorem sin_cos_equiv (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 := 
by
  sorry

end sin_cos_equiv_l233_233106


namespace algebraic_identity_l233_233426

theorem algebraic_identity (x y : ℝ) (h₁ : x * y = 4) (h₂ : x - y = 5) : 
  x^2 + 5 * x * y + y^2 = 53 := 
by 
  sorry

end algebraic_identity_l233_233426


namespace polar_curve_symmetry_l233_233446

theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), ρ = 4 * Real.sin (θ - π / 3) → 
  ∃ k : ℤ, θ = 5 * π / 6 + k * π :=
sorry

end polar_curve_symmetry_l233_233446


namespace min_additional_packs_needed_l233_233472

-- Defining the problem conditions
def total_sticker_packs : ℕ := 40
def packs_per_basket : ℕ := 7

-- The statement to prove
theorem min_additional_packs_needed : 
  ∃ (additional_packs : ℕ), 
    (total_sticker_packs + additional_packs) % packs_per_basket = 0 ∧ 
    (total_sticker_packs + additional_packs) / packs_per_basket = 6 ∧ 
    additional_packs = 2 :=
by 
  sorry

end min_additional_packs_needed_l233_233472


namespace triangle_DEF_area_l233_233575

theorem triangle_DEF_area (DE height : ℝ) (hDE : DE = 12) (hHeight : height = 15) : 
  (1/2) * DE * height = 90 :=
by
  rw [hDE, hHeight]
  norm_num

end triangle_DEF_area_l233_233575


namespace minimum_value_of_expression_l233_233691

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  1 / x + 4 / y + 9 / z

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  min_value_expression x y z ≥ 36 :=
sorry

end minimum_value_of_expression_l233_233691


namespace pow_two_ge_square_l233_233760

theorem pow_two_ge_square {n : ℕ} (hn : n ≥ 4) : 2^n ≥ n^2 :=
sorry

end pow_two_ge_square_l233_233760


namespace problem_solution_l233_233120

theorem problem_solution (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3) →
  (a₁ + a₂ + a₃ = 19) :=
by
  -- Given condition: for any real number x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3
  -- We need to prove: a₁ + a₂ + a₃ = 19
  sorry

end problem_solution_l233_233120


namespace maximum_overtakes_l233_233133

-- Definitions based on problem conditions
structure Team where
  members : List ℕ
  speed_const : ℕ → ℝ -- Speed of each member is constant but different
  run_segment : ℕ → ℕ -- Each member runs exactly one segment
  
def relay_race_condition (team1 team2 : Team) : Prop :=
  team1.members.length = 20 ∧
  team2.members.length = 20 ∧
  ∀ i, (team1.speed_const i ≠ team2.speed_const i)

def transitions (team : Team) : ℕ :=
  team.members.length - 1

-- The theorem to be proved
theorem maximum_overtakes (team1 team2 : Team) (hcond : relay_race_condition team1 team2) : 
  ∃ n, n = 38 :=
by
  sorry

end maximum_overtakes_l233_233133


namespace complement_union_l233_233904

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233904


namespace ratio_of_ages_l233_233616

-- Given conditions
def present_age_sum (H J : ℕ) : Prop :=
  H + J = 43

def present_ages (H J : ℕ) : Prop := 
  H = 27 ∧ J = 16

def multiple_of_age (H J k : ℕ) : Prop :=
  H - 5 = k * (J - 5)

-- Prove that the ratio of Henry's age to Jill's age 5 years ago was 2:1
theorem ratio_of_ages (H J k : ℕ) 
  (h_sum : present_age_sum H J)
  (h_present : present_ages H J)
  (h_multiple : multiple_of_age H J k) :
  (H - 5) / (J - 5) = 2 :=
by
  sorry

end ratio_of_ages_l233_233616


namespace arccos_zero_eq_pi_div_two_l233_233670

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l233_233670


namespace marbles_leftover_l233_233600

theorem marbles_leftover (r p g : ℕ) (hr : r % 7 = 5) (hp : p % 7 = 4) (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 :=
by
  sorry

end marbles_leftover_l233_233600


namespace hexagonalPrismCannotIntersectAsCircle_l233_233441

-- Define each geometric shape as a type
inductive GeometricShape
| Sphere
| Cone
| Cylinder
| HexagonalPrism

-- Define a function that checks if a shape can be intersected by a plane to form a circular cross-section
def canIntersectAsCircle (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True -- Sphere can always form a circular cross-section
  | GeometricShape.Cone => True -- Cone can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.Cylinder => True -- Cylinder can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.HexagonalPrism => False -- Hexagonal Prism cannot form a circular cross-section

-- The theorem to prove
theorem hexagonalPrismCannotIntersectAsCircle :
  ∀ shape : GeometricShape,
  (shape = GeometricShape.HexagonalPrism) ↔ ¬ canIntersectAsCircle shape := by
  sorry

end hexagonalPrismCannotIntersectAsCircle_l233_233441


namespace total_capacity_of_bowl_l233_233004

theorem total_capacity_of_bowl (L C : ℕ) (h1 : L / C = 3 / 5) (h2 : C = L + 18) : L + C = 72 := 
by
  sorry

end total_capacity_of_bowl_l233_233004


namespace fewest_tiles_needed_l233_233651

-- Define the dimensions of the tile
def tile_width : ℕ := 2
def tile_height : ℕ := 5

-- Define the dimensions of the floor in feet
def floor_width_ft : ℕ := 3
def floor_height_ft : ℕ := 6

-- Convert the floor dimensions to inches
def floor_width_inch : ℕ := floor_width_ft * 12
def floor_height_inch : ℕ := floor_height_ft * 12

-- Calculate the areas in square inches
def tile_area : ℕ := tile_width * tile_height
def floor_area : ℕ := floor_width_inch * floor_height_inch

-- Calculate the minimum number of tiles required, rounding up
def min_tiles_required : ℕ := Float.ceil (floor_area / tile_area)

-- The theorem statement: prove that the minimum tiles required is 260
theorem fewest_tiles_needed : min_tiles_required = 260 := 
  by 
    sorry

end fewest_tiles_needed_l233_233651


namespace evaluate_expr_at_neg3_l233_233676

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l233_233676


namespace school_student_ratio_l233_233185

theorem school_student_ratio :
  ∀ (F S T : ℕ), (T = 200) → (S = T + 40) → (F + S + T = 920) → (F : ℚ) / (S : ℚ) = 2 / 1 :=
by
  intros F S T hT hS hSum
  sorry

end school_student_ratio_l233_233185


namespace price_and_max_units_proof_l233_233827

/-- 
Given the conditions of purchasing epidemic prevention supplies: 
- 60 units of type A and 45 units of type B costing 1140 yuan
- 45 units of type A and 30 units of type B costing 840 yuan
- A total of 600 units with a cost not exceeding 8000 yuan

Prove:
1. The price of each unit of type A is 16 yuan, and type B is 4 yuan.
2. The maximum number of units of type A that can be purchased is 466.
--/
theorem price_and_max_units_proof 
  (x y : ℕ) 
  (m : ℕ)
  (h1 : 60 * x + 45 * y = 1140) 
  (h2 : 45 * x + 30 * y = 840) 
  (h3 : 16 * m + 4 * (600 - m) ≤ 8000) 
  (h4 : m ≤ 600) :
  x = 16 ∧ y = 4 ∧ m = 466 := 
by 
  sorry

end price_and_max_units_proof_l233_233827


namespace voldemort_lunch_calories_l233_233505

def dinner_cake_calories : Nat := 110
def chips_calories : Nat := 310
def coke_calories : Nat := 215
def breakfast_calories : Nat := 560
def daily_intake_limit : Nat := 2500
def remaining_calories : Nat := 525

def total_dinner_snacks_breakfast : Nat :=
  dinner_cake_calories + chips_calories + coke_calories + breakfast_calories

def total_remaining_allowance : Nat :=
  total_dinner_snacks_breakfast + remaining_calories

def lunch_calories : Nat :=
  daily_intake_limit - total_remaining_allowance

theorem voldemort_lunch_calories:
  lunch_calories = 780 := by
  sorry

end voldemort_lunch_calories_l233_233505


namespace candy_cost_l233_233370

theorem candy_cost (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) 
  (total_tickets : ℕ) (candies : ℕ) (cost_per_candy : ℕ) 
  (h1 : tickets_whack_a_mole = 8) (h2 : tickets_skee_ball = 7)
  (h3 : total_tickets = tickets_whack_a_mole + tickets_skee_ball)
  (h4 : candies = 3) (h5 : total_tickets = candies * cost_per_candy) :
  cost_per_candy = 5 :=
by
  sorry

end candy_cost_l233_233370


namespace roy_is_6_years_older_than_julia_l233_233336

theorem roy_is_6_years_older_than_julia :
  ∀ (R J K : ℕ) (x : ℕ), 
    R = J + x →
    R = K + x / 2 →
    R + 4 = 2 * (J + 4) →
    (R + 4) * (K + 4) = 108 →
    x = 6 :=
by
  intros R J K x h1 h2 h3 h4
  -- Proof goes here (using sorry to skip the proof)
  sorry

end roy_is_6_years_older_than_julia_l233_233336


namespace complement_union_eq_l233_233925

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233925


namespace max_sum_l233_233614

open Real

theorem max_sum (a b c : ℝ) (h : a^2 + (b^2) / 4 + (c^2) / 9 = 1) : a + b + c ≤ sqrt 14 :=
sorry

end max_sum_l233_233614


namespace initial_ratio_l233_233580

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l233_233580


namespace number_of_teams_l233_233445

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end number_of_teams_l233_233445


namespace other_endpoint_coordinates_sum_l233_233021

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l233_233021


namespace evaluate_expression_at_neg3_l233_233678

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l233_233678


namespace walking_speed_10_mph_l233_233382

theorem walking_speed_10_mph 
  (total_minutes : ℕ)
  (distance : ℕ)
  (rest_per_segment : ℕ)
  (rest_time : ℕ)
  (segments : ℕ)
  (walk_time : ℕ)
  (walk_time_hours : ℕ) :
  total_minutes = 328 → 
  distance = 50 → 
  rest_per_segment = 7 → 
  segments = 4 →
  rest_time = segments * rest_per_segment →
  walk_time = total_minutes - rest_time →
  walk_time_hours = walk_time / 60 →
  distance / walk_time_hours = 10 :=
by
  sorry

end walking_speed_10_mph_l233_233382


namespace range_for_k_solutions_when_k_eq_1_l233_233233

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l233_233233


namespace consecutive_composite_numbers_bound_l233_233477

theorem consecutive_composite_numbers_bound (n : ℕ) (hn: 0 < n) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ Nat.Prime (seq i)) ∧ (∀ i, seq i < 4^(n+1)) :=
sorry

end consecutive_composite_numbers_bound_l233_233477


namespace sum_of_fractions_eq_sum_of_cubes_l233_233752

theorem sum_of_fractions_eq_sum_of_cubes (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  ( (x-1)*(x+1) / (x*(x-1) + 1) + (2*(0.5-x)) / (x*(1-x) -1) ) = 
  ( ((x-1)*(x+1) / (x*(x-1) + 1))^3 + ((2*(0.5-x)) / (x*(1-x) -1))^3 ) :=
sorry

end sum_of_fractions_eq_sum_of_cubes_l233_233752


namespace arccos_zero_eq_pi_div_two_l233_233666

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l233_233666


namespace saturday_earnings_l233_233074

-- Lean 4 Statement

theorem saturday_earnings 
  (S Wednesday_earnings : ℝ)
  (h1 : S + Wednesday_earnings = 5182.50)
  (h2 : Wednesday_earnings = S - 142.50) 
  : S = 2662.50 := 
by
  sorry

end saturday_earnings_l233_233074


namespace problem_l233_233292

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l233_233292


namespace pizza_slices_with_both_toppings_l233_233198

theorem pizza_slices_with_both_toppings :
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  n = 6 :=
by
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  show n = 6
  sorry

end pizza_slices_with_both_toppings_l233_233198


namespace continuity_f_at_1_l233_233334

theorem continuity_f_at_1 (f : ℝ → ℝ) (x0 : ℝ)
  (h1 : f x0 = -12)
  (h2 : ∀ x : ℝ, f x = -5 * x^2 - 7)
  (h3 : x0 = 1) :
  ∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_f_at_1_l233_233334


namespace external_bisector_TK_l233_233469

-- Definitions of points and segments
variables {A T C L K : Type}

-- Conditions given in the problem
variable (angle_bisector_TL : is_angle_bisector T L A C)
variable (is_triangle : is_triangle T A C)

-- Prove that TK is the external angle bisector
theorem external_bisector_TK : is_external_angle_bisector T K A C :=
sorry

end external_bisector_TK_l233_233469


namespace total_students_l233_233095

theorem total_students (x : ℕ) (h1 : (x + 6) / (2*x + 6) = 2 / 3) : 2 * x + 6 = 18 :=
sorry

end total_students_l233_233095


namespace union_complement_set_l233_233285

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l233_233285


namespace union_with_complement_l233_233317

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l233_233317


namespace integer_multiple_of_ten_l233_233657

theorem integer_multiple_of_ten (x : ℤ) :
  10 * x = 30 ↔ x = 3 :=
by
  sorry

end integer_multiple_of_ten_l233_233657


namespace goods_train_cross_platform_time_l233_233645

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let total_distance : ℝ := length_train + length_platform
  total_distance / speed_mps

theorem goods_train_cross_platform_time :
  time_to_cross_platform 72 290.04 230 = 26.002 :=
by
  -- The proof is omitted
  sorry

end goods_train_cross_platform_time_l233_233645


namespace find_p_q_l233_233738

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l233_233738


namespace equal_sum_partition_l233_233416

theorem equal_sum_partition (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin n, a i ≤ a i.succ ∧ a i.succ ≤ 2 * a i)
  (h3 : (Finset.univ : Finset (Fin n.succ)).sum a % 2 = 0) :
  ∃ (partition : Finset (Fin n.succ)), 
    (partition.sum a = (partitionᶜ : Finset (Fin n.succ)).sum a) :=
by sorry

end equal_sum_partition_l233_233416


namespace probability_prime_sum_l233_233380

def is_roll_result_valid (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

def is_prime_sum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem probability_prime_sum (t : ℚ) :
  (∀ (a b : ℕ), is_roll_result_valid a → is_roll_result_valid b → a + b ∈ {2, 3, 5, 7, 11}) →
  t = 5 / 12 :=
by
  sorry  -- Proof to be filled in

end probability_prime_sum_l233_233380


namespace value_at_minus_two_l233_233556

def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 1

theorem value_at_minus_two (a b c : ℝ) (h : f 2 a b c = -1) : f (-2) a b c = 3 := by
  sorry

end value_at_minus_two_l233_233556


namespace solution_system_equations_l233_233481

theorem solution_system_equations :
  ∀ (x y : ℝ) (k n : ℤ),
    (4 * (Real.cos x) ^ 2 - 4 * Real.cos x * (Real.cos (6 * x)) ^ 2 + (Real.cos (6 * x)) ^ 2 = 0) ∧
    (Real.sin x = Real.cos y) →
    (∃ k n : ℤ, (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = (Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = -(Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = (5 * Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = -(5 * Real.pi / 6) + 2 * Real.pi * n)) :=
by
  sorry

end solution_system_equations_l233_233481


namespace cannot_eat_166_candies_l233_233527

-- Define parameters for sandwiches and candies equations
def sandwiches_eq (x y z : ℕ) := x + 2 * y + 3 * z = 100
def candies_eq (x y z : ℕ) := 3 * x + 4 * y + 5 * z = 166

theorem cannot_eat_166_candies (x y z : ℕ) : ¬ (sandwiches_eq x y z ∧ candies_eq x y z) :=
by {
  -- Proof will show impossibility of (x, y, z) as nonnegative integers solution
  sorry
}

end cannot_eat_166_candies_l233_233527


namespace complement_union_eq_singleton_five_l233_233873

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233873


namespace find_numbers_l233_233353

-- Definitions for the conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0
def difference_is_three (x y : ℕ) : Prop := x - y = 3

-- Statement of the proof problem
theorem find_numbers (x y : ℕ) (h1 : is_three_digit x) (h2 : is_even_two_digit y) (h3 : difference_is_three x y) :
  x = 101 ∧ y = 98 :=
sorry

end find_numbers_l233_233353


namespace Jackson_to_Williams_Ratio_l233_233270

-- Define the amounts of money Jackson and Williams have, given the conditions.
def JacksonMoney : ℤ := 125
def TotalMoney : ℤ := 150
-- Define Williams' money based on the given conditions.
def WilliamsMoney : ℤ := TotalMoney - JacksonMoney

-- State the theorem that the ratio of Jackson's money to Williams' money is 5:1
theorem Jackson_to_Williams_Ratio : JacksonMoney / WilliamsMoney = 5 := 
by
  -- Proof steps are omitted as per the instruction.
  sorry

end Jackson_to_Williams_Ratio_l233_233270


namespace total_marks_l233_233459

theorem total_marks (k l d : ℝ) (hk : k = 3.5) (hl : l = 3.2 * k) (hd : d = l + 5.7) : k + l + d = 31.6 :=
by
  rw [hk] at hl
  rw [hl] at hd
  rw [hk, hl, hd]
  sorry

end total_marks_l233_233459


namespace rectangular_park_length_l233_233193

noncomputable def length_of_rectangular_park
  (P : ℕ) (B : ℕ) (L : ℕ) : Prop :=
  (P = 1000) ∧ (B = 200) ∧ (P = 2 * (L + B)) → (L = 300)

theorem rectangular_park_length : length_of_rectangular_park 1000 200 300 :=
by {
  sorry
}

end rectangular_park_length_l233_233193


namespace gifts_wrapped_with_third_roll_l233_233757

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l233_233757


namespace steven_needs_more_seeds_l233_233168

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l233_233168


namespace thm_300th_term_non_square_seq_l233_233635

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l233_233635


namespace minimum_students_l233_233570

variables (b g : ℕ) -- Define variables for boys and girls

-- Define the conditions
def boys_passed : ℕ := (3 * b) / 4
def girls_passed : ℕ := (2 * g) / 3
def equal_passed := boys_passed b = girls_passed g

def total_students := b + g + 4

-- Statement to prove minimum students in the class
theorem minimum_students (h1 : equal_passed b g)
  (h2 : ∃ multiple_of_nine : ℕ, g = 9 * multiple_of_nine ∧ 3 * b = 4 * multiple_of_nine * 2) :
  total_students b g = 21 :=
sorry

end minimum_students_l233_233570


namespace sufficiency_but_not_necessity_l233_233069

theorem sufficiency_but_not_necessity (a b : ℝ) :
  (a = 0 → a * b = 0) ∧ (a * b = 0 → a = 0) → False :=
by
   -- Proof is skipped
   sorry

end sufficiency_but_not_necessity_l233_233069


namespace check_point_on_curve_l233_233535

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x * y + 2 * y + 1 = 0

theorem check_point_on_curve :
  point_on_curve 0 (-1/2) :=
by
  sorry

end check_point_on_curve_l233_233535


namespace expand_product_l233_233405

noncomputable def a (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := x^2 + x + 3

theorem expand_product (x : ℝ) : (a x) * (b x) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 :=
by
  sorry

end expand_product_l233_233405


namespace find_a_even_function_l233_233852

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f x = (x + 1) * (x + a))  
  (h2 : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_even_function_l233_233852


namespace union_complement_eq_l233_233283

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l233_233283


namespace total_worth_of_stock_l233_233136

theorem total_worth_of_stock :
  let cost_expensive := 10
  let cost_cheaper := 3.5
  let total_modules := 11
  let cheaper_modules := 10
  let expensive_modules := total_modules - cheaper_modules
  let worth_cheaper_modules := cheaper_modules * cost_cheaper
  let worth_expensive_module := expensive_modules * cost_expensive 
  worth_cheaper_modules + worth_expensive_module = 45 := by
  sorry

end total_worth_of_stock_l233_233136


namespace least_number_leaving_remainder_4_l233_233637

theorem least_number_leaving_remainder_4 (x : ℤ) : 
  (x % 6 = 4) ∧ (x % 9 = 4) ∧ (x % 12 = 4) ∧ (x % 18 = 4) → x = 40 :=
by
  sorry

end least_number_leaving_remainder_4_l233_233637


namespace jewelry_store_gross_profit_l233_233797

theorem jewelry_store_gross_profit (purchase_price selling_price new_selling_price gross_profit : ℝ)
    (h1 : purchase_price = 240)
    (h2 : markup = 0.25 * selling_price)
    (h3 : selling_price = purchase_price + markup)
    (h4 : decrease = 0.20 * selling_price)
    (h5 : new_selling_price = selling_price - decrease)
    (h6 : gross_profit = new_selling_price - purchase_price) :
    gross_profit = 16 :=
by
    sorry

end jewelry_store_gross_profit_l233_233797


namespace inequality_holds_l233_233332

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) :=
sorry

end inequality_holds_l233_233332


namespace complement_union_eq_l233_233927

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233927


namespace marcus_scored_50_percent_l233_233321

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l233_233321


namespace find_range_of_m_l233_233846

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  m^2 - 4 > 0

def inequality_holds_for_all_real_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * (m + 1) * x + m * (m + 1) > 0

def p (m : ℝ) : Prop := has_two_distinct_real_roots m
def q (m : ℝ) : Prop := inequality_holds_for_all_real_x m

theorem find_range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
sorry

end find_range_of_m_l233_233846


namespace segment_bisected_at_A_l233_233521

variable {P Q A : EuclideanSpace ℝ ℝ}

theorem segment_bisected_at_A (A_inside_angle : ∃ (l1 l2 : set (EuclideanSpace ℝ ℝ)), 
    A ∈ l1 ∧ A ∈ l2 ∧ angle A l1 l2 = 2 ∧ ∀ (l : set (EuclideanSpace ℝ ℝ)), 
    (A ∈ l) → (area (triangle A l1 l2 l) = smallest_area_triangle)) : 
    bisection A P Q :=
sorry

end segment_bisected_at_A_l233_233521


namespace divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l233_233374

theorem divides_2_pow_n_sub_1 (n : ℕ) : 7 ∣ (2 ^ n - 1) ↔ 3 ∣ n := by
  sorry

theorem no_n_divides_2_pow_n_add_1 (n : ℕ) : ¬ 7 ∣ (2 ^ n + 1) := by
  sorry

end divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l233_233374


namespace solution_set_of_inequality_l233_233780

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2*x + 3 > 0 ↔ (-1 < x ∧ x < 3) :=
sorry

end solution_set_of_inequality_l233_233780


namespace simplify_expression_l233_233005

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by
  sorry

end simplify_expression_l233_233005


namespace complement_union_l233_233945

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233945


namespace max_value_of_expression_l233_233857

variables (a x1 x2 : ℝ)

theorem max_value_of_expression :
  (x1 < 0) → (0 < x2) → (∀ x, x^2 - a * x + a - 2 > 0 ↔ (x < x1) ∨ (x > x2)) →
  (x1 * x2 = a - 2) → 
  x1 + x2 + 2 / x1 + 2 / x2 ≤ 0 :=
by
  intros h1 h2 h3 h4
  -- Proof goes here
  sorry

end max_value_of_expression_l233_233857


namespace michael_total_cost_l233_233326

def peach_pies : ℕ := 5
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3

def pounds_per_pie : ℕ := 3

def price_per_pound_peaches : ℝ := 2.0
def price_per_pound_apples : ℝ := 1.0
def price_per_pound_blueberries : ℝ := 1.0

def total_peach_pounds : ℕ := peach_pies * pounds_per_pie
def total_apple_pounds : ℕ := apple_pies * pounds_per_pie
def total_blueberry_pounds : ℕ := blueberry_pies * pounds_per_pie

def cost_peaches : ℝ := total_peach_pounds * price_per_pound_peaches
def cost_apples : ℝ := total_apple_pounds * price_per_pound_apples
def cost_blueberries : ℝ := total_blueberry_pounds * price_per_pound_blueberries

def total_cost : ℝ := cost_peaches + cost_apples + cost_blueberries

theorem michael_total_cost :
  total_cost = 51.0 := by
  sorry

end michael_total_cost_l233_233326


namespace jane_mean_score_l233_233582

-- Define the six quiz scores Jane took
def score1 : ℕ := 86
def score2 : ℕ := 91
def score3 : ℕ := 89
def score4 : ℕ := 95
def score5 : ℕ := 88
def score6 : ℕ := 94

-- The number of quizzes
def num_quizzes : ℕ := 6

-- The sum of all quiz scores
def total_score : ℕ := score1 + score2 + score3 + score4 + score5 + score6 

-- The expected mean score
def mean_score : ℚ := 90.5

-- The proof statement
theorem jane_mean_score (h : total_score = 543) : total_score / num_quizzes = mean_score := 
by sorry

end jane_mean_score_l233_233582


namespace part_a_part_b_l233_233993

noncomputable def f (g n : ℕ) : ℕ := g^n + 1

theorem part_a (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → f g n ∣ f g (3*n) ∧ f g n ∣ f g (5*n) ∧ f g n ∣ f g (7*n) :=
sorry

theorem part_b (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → ∀ k : ℕ, 1 ≤ k → gcd (f g n) (f g (2*k*n)) = 1 :=
sorry

end part_a_part_b_l233_233993


namespace tv_price_increase_percentage_l233_233041

theorem tv_price_increase_percentage (P Q : ℝ) (x : ℝ) :
  (P * (1 + x / 100) * Q * 0.8 = P * Q * 1.28) → x = 60 :=
by sorry

end tv_price_increase_percentage_l233_233041


namespace clock_equiv_l233_233329

theorem clock_equiv (h : ℕ) (h_gt_6 : h > 6) : h ≡ h^2 [MOD 12] ∧ h ≡ h^3 [MOD 12] → h = 9 :=
by
  sorry

end clock_equiv_l233_233329


namespace volume_relationship_l233_233652

open Real

theorem volume_relationship (r : ℝ) (A M C : ℝ)
  (hA : A = (1/3) * π * r^3)
  (hM : M = π * r^3)
  (hC : C = (4/3) * π * r^3) :
  A + M + (1/2) * C = 2 * π * r^3 :=
by
  sorry

end volume_relationship_l233_233652


namespace no_natural_numbers_satisfy_conditions_l233_233826

theorem no_natural_numbers_satisfy_conditions : 
  ¬ ∃ (a b : ℕ), 
    (∃ (k : ℕ), k^2 = a^2 + 2 * b^2) ∧ 
    (∃ (m : ℕ), m^2 = b^2 + 2 * a) :=
by {
  -- Proof steps and logical deductions can be written here.
  sorry
}

end no_natural_numbers_satisfy_conditions_l233_233826


namespace apps_left_on_phone_l233_233093

-- Definitions for the given conditions
def initial_apps : ℕ := 15
def added_apps : ℕ := 71
def deleted_apps : ℕ := added_apps + 1

-- Proof statement
theorem apps_left_on_phone : initial_apps + added_apps - deleted_apps = 14 := by
  sorry

end apps_left_on_phone_l233_233093


namespace knocks_to_knicks_l233_233705

def knicks := ℕ
def knacks := ℕ
def knocks := ℕ

axiom knicks_to_knacks_ratio (k : knicks) (n : knacks) : 5 * k = 3 * n
axiom knacks_to_knocks_ratio (n : knacks) (o : knocks) : 4 * n = 6 * o

theorem knocks_to_knicks (k : knicks) (n : knacks) (o : knocks) (h1 : 5 * k = 3 * n) (h2 : 4 * n = 6 * o) :
  36 * o = 40 * k :=
sorry

end knocks_to_knicks_l233_233705


namespace weight_removed_l233_233375

-- Definitions for the given conditions
def weight_sugar : ℕ := 16
def weight_salt : ℕ := 30
def new_combined_weight : ℕ := 42

-- The proof problem statement
theorem weight_removed : (weight_sugar + weight_salt) - new_combined_weight = 4 := by
  -- Proof will be provided here
  sorry

end weight_removed_l233_233375


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l233_233234

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l233_233234


namespace sixth_term_of_arithmetic_sequence_l233_233783

noncomputable def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem sixth_term_of_arithmetic_sequence
  (a d : ℕ)
  (h₁ : sum_first_n_terms a d 4 = 10)
  (h₂ : a + 4 * d = 5) :
  a + 5 * d = 6 :=
by {
  sorry
}

end sixth_term_of_arithmetic_sequence_l233_233783


namespace total_preparation_and_cooking_time_l233_233815

def time_to_chop_pepper := 3
def time_to_chop_onion := 4
def time_to_slice_mushroom := 2
def time_to_dice_tomato := 3
def time_to_grate_cheese := 1
def time_to_assemble_and_cook_omelet := 6

def num_peppers := 8
def num_onions := 4
def num_mushrooms := 6
def num_tomatoes := 6
def num_omelets := 10

theorem total_preparation_and_cooking_time :
  (num_peppers * time_to_chop_pepper) +
  (num_onions * time_to_chop_onion) +
  (num_mushrooms * time_to_slice_mushroom) +
  (num_tomatoes * time_to_dice_tomato) +
  (num_omelets * time_to_grate_cheese) +
  (num_omelets * time_to_assemble_and_cook_omelet) = 140 :=
by
  sorry

end total_preparation_and_cooking_time_l233_233815


namespace megatech_astrophysics_degrees_l233_233371

theorem megatech_astrophysics_degrees :
  let microphotonics := 10
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let astrophysics_percentage := 100 - total_percentage
  let total_degrees := 360
  let astrophysics_degrees := (astrophysics_percentage / 100) * total_degrees
  astrophysics_degrees = 50.4 :=
by
  sorry

end megatech_astrophysics_degrees_l233_233371


namespace complement_of_union_is_singleton_five_l233_233921

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233921


namespace complement_union_l233_233877

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233877


namespace max_lift_times_l233_233473

theorem max_lift_times (n : ℕ) :
  (2 * 30 * 10) = (2 * 25 * n) → n = 12 :=
by
  sorry

end max_lift_times_l233_233473


namespace sum_of_other_endpoint_coordinates_l233_233019

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l233_233019


namespace milkshake_cost_proof_l233_233658

-- Define the problem
def milkshake_cost (total_money : ℕ) (hamburger_cost : ℕ) (n_hamburgers : ℕ)
                   (n_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_hamburgers_cost := n_hamburgers * hamburger_cost
  let money_after_hamburgers := total_money - total_hamburgers_cost
  let milkshake_cost := (money_after_hamburgers - remaining_money) / n_milkshakes
  milkshake_cost

-- Statement to prove
theorem milkshake_cost_proof : milkshake_cost 120 4 8 6 70 = 3 :=
by
  -- we skip the proof steps as the problem statement does not require it
  sorry

end milkshake_cost_proof_l233_233658


namespace find_p_q_l233_233742

theorem find_p_q (p q : ℚ)
  (h1 : (4 : ℚ) * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2):
  (p, q) = (-29/12 : ℚ, 43/12 : ℚ) :=
by 
  sorry

end find_p_q_l233_233742


namespace batsman_average_increase_l233_233520

theorem batsman_average_increase
  (A : ℕ)
  (h_average_after_17th : (16 * A + 90) / 17 = 42) :
  42 - A = 3 :=
by
  sorry

end batsman_average_increase_l233_233520


namespace calculate_p_op_l233_233999

def op (x y : ℝ) := x * y^2 - x

theorem calculate_p_op (p : ℝ) : op p (op p p) = p^7 - 2*p^5 + p^3 - p :=
by
  sorry

end calculate_p_op_l233_233999


namespace distinguishable_arrangements_l233_233116

-- Define the conditions: number of tiles of each color
def num_brown_tiles := 2
def num_purple_tile := 1
def num_green_tiles := 3
def num_yellow_tiles := 4

-- Total number of tiles
def total_tiles := num_brown_tiles + num_purple_tile + num_green_tiles + num_yellow_tiles

-- Factorials (using Lean's built-in factorial function)
def brown_factorial := Nat.factorial num_brown_tiles
def purple_factorial := Nat.factorial num_purple_tile
def green_factorial := Nat.factorial num_green_tiles
def yellow_factorial := Nat.factorial num_yellow_tiles
def total_factorial := Nat.factorial total_tiles

-- The result of the permutation calculation
def number_of_arrangements := total_factorial / (brown_factorial * purple_factorial * green_factorial * yellow_factorial)

-- The theorem stating the expected correct answer
theorem distinguishable_arrangements : number_of_arrangements = 12600 := 
by
    simp [number_of_arrangements, total_tiles, brown_factorial, purple_factorial, green_factorial, yellow_factorial, total_factorial]
    sorry

end distinguishable_arrangements_l233_233116


namespace negation_of_proposition_l233_233495

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≥ 0) ↔ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by sorry

end negation_of_proposition_l233_233495


namespace find_number_of_male_students_l233_233154

/- Conditions: 
 1. n ≡ 2 [MOD 4]
 2. n ≡ 1 [MOD 5]
 3. n > 15
 4. There are 15 female students
 5. There are more female students than male students
-/
theorem find_number_of_male_students (n : ℕ) (females : ℕ) (h1 : n % 4 = 2) (h2 : n % 5 = 1) (h3 : n > 15) (h4 : females = 15) (h5 : females > n - females) : (n - females) = 11 :=
by
  sorry

end find_number_of_male_students_l233_233154


namespace isosceles_triangle_perimeter_l233_233181

theorem isosceles_triangle_perimeter (perimeter_eq_tri : ℕ) (side_eq_tri : ℕ) (base_iso_tri : ℕ) (perimeter_iso_tri : ℕ) 
  (h1 : perimeter_eq_tri = 60) 
  (h2 : side_eq_tri = perimeter_eq_tri / 3) 
  (h3 : base_iso_tri = 5)
  (h4 : perimeter_iso_tri = 2 * side_eq_tri + base_iso_tri) : 
  perimeter_iso_tri = 45 := by
  sorry

end isosceles_triangle_perimeter_l233_233181


namespace quadrilateral_inequality_l233_233782

-- Definitions based on conditions in a)
variables {A B C D : Type}
variables (AB AC AD BC CD : ℝ)
variable (angleA angleC: ℝ)
variable (convex := angleA + angleC < 180)

-- Lean statement that encodes the problem
theorem quadrilateral_inequality 
  (Hconvex : convex = true)
  : AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l233_233782


namespace simplify_fraction_l233_233029

theorem simplify_fraction (n : ℕ) : 
  (3 ^ (n + 3) - 3 * (3 ^ n)) / (3 * 3 ^ (n + 2)) = 8 / 9 :=
by sorry

end simplify_fraction_l233_233029


namespace truncated_pyramid_volume_l233_233212

theorem truncated_pyramid_volume :
  let unit_cube_vol := 1
  let tetrahedron_base_area := 1 / 2
  let tetrahedron_height := 1 / 2
  let tetrahedron_vol := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let two_tetrahedra_vol := 2 * tetrahedron_vol
  let truncated_pyramid_vol := unit_cube_vol - two_tetrahedra_vol
  truncated_pyramid_vol = 5 / 6 :=
by
  sorry

end truncated_pyramid_volume_l233_233212


namespace complement_union_l233_233903

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233903


namespace solve_equation_l233_233030

-- Define the equation to be solved
def equation (x : ℝ) : Prop := (x + 2)^4 + (x - 4)^4 = 272

-- State the theorem we want to prove
theorem solve_equation : ∃ x : ℝ, equation x :=
  sorry

end solve_equation_l233_233030


namespace stickers_initial_count_l233_233271

variable (initial : ℕ) (lost : ℕ)

theorem stickers_initial_count (lost_stickers : lost = 6) (remaining_stickers : initial - lost = 87) : initial = 93 :=
by {
  sorry
}

end stickers_initial_count_l233_233271


namespace evaluate_expr_at_neg3_l233_233675

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l233_233675


namespace sequence_a4_value_l233_233182

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 1) ∧ a 4 = 15 :=
by
  sorry

end sequence_a4_value_l233_233182


namespace find_a_l233_233565

theorem find_a (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 :=
sorry

end find_a_l233_233565


namespace find_a_l233_233970

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l233_233970


namespace largest_sum_of_two_3_digit_numbers_l233_233509

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
    (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧ 
    (100 * (a + d) + 10 * (b + e) + (c + f) = 1173) :=
by
  sorry

end largest_sum_of_two_3_digit_numbers_l233_233509


namespace number_of_boxwoods_l233_233327

variables (x : ℕ)
def charge_per_trim := 5
def charge_per_shape := 15
def number_of_shaped_boxwoods := 4
def total_charge := 210
def total_shaping_charge := number_of_shaped_boxwoods * charge_per_shape

theorem number_of_boxwoods (h : charge_per_trim * x + total_shaping_charge = total_charge) : x = 30 :=
by
  sorry

end number_of_boxwoods_l233_233327


namespace camryn_flute_practice_interval_l233_233399

theorem camryn_flute_practice_interval (x : ℕ) 
  (h1 : ∃ n : ℕ, n * 11 = 33) 
  (h2 : x ∣ 33) 
  (h3 : x < 11) 
  (h4 : x > 1) 
  : x = 3 := 
sorry

end camryn_flute_practice_interval_l233_233399


namespace middle_card_is_four_l233_233621

theorem middle_card_is_four (a b c : ℕ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                            (h2 : a + b + c = 15)
                            (h3 : a < b ∧ b < c)
                            (h_casey : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_tracy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_stacy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            : b = 4 := 
sorry

end middle_card_is_four_l233_233621


namespace cannot_determine_degree_from_char_set_l233_233991

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l233_233991


namespace largest_angle_consecutive_even_pentagon_l233_233043

theorem largest_angle_consecutive_even_pentagon :
  ∀ (n : ℕ), (2 * n + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 540) →
  (2 * n + 8 = 112) :=
by
  intros n h
  sorry

end largest_angle_consecutive_even_pentagon_l233_233043


namespace complement_union_of_M_and_N_l233_233893

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233893


namespace complement_union_l233_233931

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233931


namespace regular_polygon_sides_l233_233828

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l233_233828


namespace arithmetic_sequence_sum_l233_233778

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end arithmetic_sequence_sum_l233_233778


namespace ratio_of_white_socks_l233_233457

theorem ratio_of_white_socks 
  (total_socks : ℕ) (blue_socks : ℕ)
  (h_total_socks : total_socks = 180)
  (h_blue_socks : blue_socks = 60) :
  (total_socks - blue_socks) * 3 = total_socks * 2 :=
by
  sorry

end ratio_of_white_socks_l233_233457


namespace union_complement_eq_l233_233305

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l233_233305


namespace complement_union_l233_233909

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233909


namespace logarithm_argument_positive_l233_233569

open Real

theorem logarithm_argument_positive (a : ℝ) : 
  (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + a * sin x * cos x > 0) ↔ -1 / 2 < a ∧ a < 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end logarithm_argument_positive_l233_233569


namespace find_m_value_l233_233694

-- Definitions based on conditions
variables {a b m : ℝ} (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1)

-- Lean 4 statement of the problem
theorem find_m_value (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1) : m = 10 := sorry

end find_m_value_l233_233694


namespace div_condition_l233_233408

theorem div_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  4 * (m * n + 1) % (m + n)^2 = 0 ↔ m = n := 
sorry

end div_condition_l233_233408


namespace joan_lost_balloons_l233_233143

theorem joan_lost_balloons :
  let initial_balloons := 9
  let current_balloons := 7
  let balloons_lost := initial_balloons - current_balloons
  balloons_lost = 2 :=
by
  sorry

end joan_lost_balloons_l233_233143


namespace ratio_female_to_male_l233_233218

variable {f m c : ℕ}

/-- 
  The following conditions are given:
  - The average age of female members is 35 years.
  - The average age of male members is 30 years.
  - The average age of children members is 10 years.
  - The average age of the entire membership is 25 years.
  - The number of children members is equal to the number of male members.
  We need to show that the ratio of female to male members is 1.
-/
theorem ratio_female_to_male (h1 : c = m)
  (h2 : 35 * f + 40 * m = 25 * (f + 2 * m)) :
  f = m :=
by sorry

end ratio_female_to_male_l233_233218


namespace joan_total_spending_l233_233729

def basketball_game_price : ℝ := 5.20
def basketball_game_discount : ℝ := 0.15 * basketball_game_price
def basketball_game_discounted : ℝ := basketball_game_price - basketball_game_discount

def racing_game_price : ℝ := 4.23
def racing_game_discount : ℝ := 0.10 * racing_game_price
def racing_game_discounted : ℝ := racing_game_price - racing_game_discount

def puzzle_game_price : ℝ := 3.50

def total_before_tax : ℝ := basketball_game_discounted + racing_game_discounted + puzzle_game_price
def sales_tax : ℝ := 0.08 * total_before_tax
def total_with_tax : ℝ := total_before_tax + sales_tax

theorem joan_total_spending : (total_with_tax : ℝ) = 12.67 := by
  sorry

end joan_total_spending_l233_233729


namespace range_of_m_l233_233248

open Set Real

-- Define over the real numbers ℝ
noncomputable def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }
noncomputable def CRB (m : ℝ) : Set ℝ := { x : ℝ | x < m - 2 ∨ x > m + 2 }

-- Main theorem statement
theorem range_of_m (m : ℝ) (h : A ⊆ CRB m) : m < -3 ∨ m > 5 :=
sorry

end range_of_m_l233_233248


namespace Anne_carrying_four_cats_weight_l233_233217

theorem Anne_carrying_four_cats_weight : 
  let w1 := 2
  let w2 := 1.5 * w1
  let m1 := 2 * w1
  let m2 := w1 + w2
  w1 + w2 + m1 + m2 = 14 :=
by
  sorry

end Anne_carrying_four_cats_weight_l233_233217


namespace problem_statement_l233_233571

noncomputable def C : ℝ := 49
noncomputable def D : ℝ := 3.75

theorem problem_statement : C + D = 52.75 := by
  sorry

end problem_statement_l233_233571


namespace find_greater_number_l233_233501

-- Define the two numbers x and y
variables (x y : ℕ)

-- Conditions
theorem find_greater_number (h1 : x + y = 36) (h2 : x - y = 12) : x = 24 := 
by
  sorry

end find_greater_number_l233_233501


namespace absolute_value_inequality_solution_l233_233183

theorem absolute_value_inequality_solution (x : ℝ) : abs (x - 3) < 2 ↔ 1 < x ∧ x < 5 :=
by
  sorry

end absolute_value_inequality_solution_l233_233183


namespace complement_union_eq_l233_233926

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233926


namespace number_of_persons_in_first_group_eq_39_l233_233763

theorem number_of_persons_in_first_group_eq_39 :
  ∀ (P : ℕ),
    (P * 12 * 5 = 15 * 26 * 6) →
    P = 39 :=
by
  intros P h
  have h1 : P = (15 * 26 * 6) / (12 * 5) := sorry
  simp at h1
  exact h1

end number_of_persons_in_first_group_eq_39_l233_233763


namespace three_days_earning_l233_233012

theorem three_days_earning
  (charge : ℤ := 2)
  (day_before_yesterday_wash : ℤ := 5)
  (yesterday_wash : ℤ := day_before_yesterday_wash + 5)
  (today_wash : ℤ := 2 * yesterday_wash)
  (three_days_earning : ℤ := charge * (day_before_yesterday_wash + yesterday_wash + today_wash)) :
  three_days_earning = 70 := 
by
  have h1 : day_before_yesterday_wash = 5 := by rfl
  have h2 : yesterday_wash = day_before_yesterday_wash + 5 := by rfl
  have h3 : today_wash = 2 * yesterday_wash := by rfl
  have h4 : charge * (day_before_yesterday_wash + yesterday_wash + today_wash) = 70 := sorry
  exact h4

end three_days_earning_l233_233012


namespace external_bisector_of_triangle_l233_233463

variables {A T C K L : Type} [noncomputable_field A T C K L]
variable {triangle ATC : Triangle A T C}

noncomputable def internal_angle_bisector (T L : Point) := 
    ∃ R, is_bisector TL ∧ TL ∝ AR RL

noncomputable def external_angle_bisector (T K : Point) := 
    ∃ M, is_bisector TK ∧ TK ∝ AM MK

theorem external_bisector_of_triangle {A T C K L : Point} 
    (hL : internal_angle_bisector T L) 
    : external_angle_bisector T K :=
sorry

end external_bisector_of_triangle_l233_233463


namespace family_spent_36_dollars_l233_233648

def ticket_cost : ℝ := 5

def popcorn_cost : ℝ := 0.8 * ticket_cost

def soda_cost : ℝ := 0.5 * popcorn_cost

def tickets_bought : ℕ := 4

def popcorn_bought : ℕ := 2

def sodas_bought : ℕ := 4

def total_spent : ℝ :=
  (tickets_bought * ticket_cost) +
  (popcorn_bought * popcorn_cost) +
  (sodas_bought * soda_cost)

theorem family_spent_36_dollars : total_spent = 36 := by
  sorry

end family_spent_36_dollars_l233_233648


namespace complement_union_M_N_l233_233956

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233956


namespace degree_not_determined_by_A_P_l233_233988

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l233_233988


namespace triangle_area_l233_233040

theorem triangle_area (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ) :
  P = 42 → r = 5 → s = P / 2 → A = r * s → A = 105 :=
by
  intro hP hr hs hA
  sorry

end triangle_area_l233_233040


namespace find_f3_l233_233345

theorem find_f3 (f : ℚ → ℚ)
  (h : ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x) / x = x^3) :
  f 3 = 7753 / 729 :=
sorry

end find_f3_l233_233345


namespace tangent_line_through_B_l233_233430

theorem tangent_line_through_B (x : ℝ) (y : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  (y₀ = x₀^2) →
  (y - y₀ = 2*x₀*(x - x₀)) →
  (3, 5) ∈ ({p : ℝ × ℝ | ∃ t, p.2 - t^2 = 2*t*(p.1 - t)}) →
  (x = 2 * x₀) ∧ (y = y₀) →
  (2*x - y - 1 = 0 ∨ 10*x - y - 25 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end tangent_line_through_B_l233_233430


namespace right_triangle_third_side_l233_233977

theorem right_triangle_third_side (a b c : ℝ) (ha : a = 8) (hb : b = 6) (h_right_triangle : a^2 + b^2 = c^2) :
  c = 10 :=
by
  sorry

end right_triangle_third_side_l233_233977


namespace unique_solution_f_eq_x_l233_233835

theorem unique_solution_f_eq_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2 * y + f x ^ 2) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_eq_x_l233_233835


namespace three_hundredth_term_without_squares_l233_233633

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l233_233633


namespace sum_of_palindromes_l233_233499

/-- Definition of a three-digit palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n / 100 = n % 10

theorem sum_of_palindromes (a b : ℕ) (h1 : is_palindrome a)
  (h2 : is_palindrome b) (h3 : a * b = 334491) (h4 : 100 ≤ a)
  (h5 : a < 1000) (h6 : 100 ≤ b) (h7 : b < 1000) : a + b = 1324 :=
sorry

end sum_of_palindromes_l233_233499


namespace william_shared_marble_count_l233_233511

theorem william_shared_marble_count : ∀ (initial_marbles shared_marbles remaining_marbles : ℕ),
  initial_marbles = 10 → remaining_marbles = 7 → 
  shared_marbles = initial_marbles - remaining_marbles → 
  shared_marbles = 3 := by 
    intros initial_marbles shared_marbles remaining_marbles h_initial h_remaining h_shared
    rw [h_initial, h_remaining] at h_shared
    exact h_shared

end william_shared_marble_count_l233_233511


namespace oliver_earning_correct_l233_233010

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l233_233010


namespace example_theorem_l233_233860

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233860


namespace square_of_binomial_is_25_l233_233972

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l233_233972


namespace books_in_bin_after_actions_l233_233588

theorem books_in_bin_after_actions (x y : ℕ) (z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 250) : x - y + (z / 100) * x = 11 :=
by
  rw [hx, hy, hz]
  -- x - y + (z / 100) * x = 4 - 3 + (250 / 100) * 4
  norm_num
  sorry

end books_in_bin_after_actions_l233_233588


namespace man_work_days_l233_233077

theorem man_work_days :
  ∃ M : ℕ, (∀ M, ((1 : ℚ) / M + 1 / 6 = 1 / 3) -> M = 6) := by
  sorry

end man_work_days_l233_233077


namespace complement_union_eq_singleton_five_l233_233868

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233868


namespace coeff_xcubed_expansion_l233_233351

theorem coeff_xcubed_expansion : 
  coefficient (expand (2*x - 3)^5) x^3 = 720 := 
sorry

end coeff_xcubed_expansion_l233_233351


namespace complement_union_l233_233900

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233900


namespace complement_union_l233_233948

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233948


namespace prob_no_rain_correct_l233_233776

-- Define the probability of rain on each of the next five days
def prob_rain_each_day : ℚ := 1 / 2

-- Define the probability of no rain on a single day
def prob_no_rain_one_day : ℚ := 1 - prob_rain_each_day

-- Define the probability of no rain in any of the next five days
def prob_no_rain_five_days : ℚ := prob_no_rain_one_day ^ 5

-- Theorem statement
theorem prob_no_rain_correct : prob_no_rain_five_days = 1 / 32 := by
  sorry

end prob_no_rain_correct_l233_233776


namespace nonempty_solution_set_iff_a_gt_2_l233_233478

theorem nonempty_solution_set_iff_a_gt_2 (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) ↔ a > 2 :=
sorry

end nonempty_solution_set_iff_a_gt_2_l233_233478


namespace steven_needs_more_seeds_l233_233167

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l233_233167


namespace union_complement_l233_233299

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l233_233299


namespace cube_root_of_8_is_2_l233_233037

theorem cube_root_of_8_is_2 : ∃ x : ℝ, x ^ 3 = 8 ∧ x = 2 :=
by
  have h : (2 : ℝ) ^ 3 = 8 := by norm_num
  exact ⟨2, h, rfl⟩

end cube_root_of_8_is_2_l233_233037


namespace sun_salutations_per_year_l233_233032

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l233_233032


namespace number_of_charms_l233_233360

-- Let x be the number of charms used to make each necklace
variable (x : ℕ)

-- Each charm costs $15
variable (cost_per_charm : ℕ)
axiom cost_per_charm_is_15 : cost_per_charm = 15

-- Tim sells each necklace for $200
variable (selling_price : ℕ)
axiom selling_price_is_200 : selling_price = 200

-- Tim makes a profit of $1500 if he sells 30 necklaces
variable (total_profit : ℕ)
axiom total_profit_is_1500 : total_profit = 1500

theorem number_of_charms (h : 30 * (selling_price - cost_per_charm * x) = total_profit) : x = 10 :=
sorry

end number_of_charms_l233_233360


namespace expression_factorization_l233_233196

variables (a b c : ℝ)

theorem expression_factorization :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3)
  = (a - b) * (b - c) * (c - a) * (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
sorry

end expression_factorization_l233_233196


namespace quantiville_jacket_junction_l233_233986

theorem quantiville_jacket_junction :
  let sales_tax_rate := 0.07
  let original_price := 120.0
  let discount := 0.25
  let amy_total := (original_price * (1 + sales_tax_rate)) * (1 - discount)
  let bob_total := (original_price * (1 - discount)) * (1 + sales_tax_rate)
  let carla_total := ((original_price * (1 + sales_tax_rate)) * (1 - discount)) * (1 + sales_tax_rate)
  (carla_total - amy_total) = 6.744 :=
by
  sorry

end quantiville_jacket_junction_l233_233986


namespace total_area_of_combined_figure_l233_233348

noncomputable def combined_area (A_triangle : ℕ) (b : ℕ) : ℕ :=
  let h := (2 * A_triangle) / b
  let A_square := b * b
  A_square + A_triangle

theorem total_area_of_combined_figure :
  combined_area 720 40 = 2320 := by
  sorry

end total_area_of_combined_figure_l233_233348


namespace directrix_of_parabola_l233_233174

theorem directrix_of_parabola :
  ∀ (a h k : ℝ), (a < 0) → (∀ x, y = a * (x - h) ^ 2 + k) → (h = 0) → (k = 0) → 
  (directrix = 1 / (4 * a)) → (directrix = 1 / 4) :=
by
  sorry

end directrix_of_parabola_l233_233174


namespace monotonic_increase_interval_range_of_a_l233_233111

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + 2 * Real.exp x - a * x^2
def h (x : ℝ) : ℝ := x

theorem monotonic_increase_interval :
  ∃ I : Set ℝ, I = Set.Ioi 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y := 
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (g x1 a - h x1) * (g x2 a - h x2) > 0) ↔ a ∈ Set.Iic 1 :=
  sorry

end monotonic_increase_interval_range_of_a_l233_233111


namespace star_15_star_eq_neg_15_l233_233685

-- Define the operations as given
def y_star (y : ℤ) := 9 - y
def star_y (y : ℤ) := y - 9

-- The theorem stating the required proof
theorem star_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by
  sorry

end star_15_star_eq_neg_15_l233_233685


namespace problem1_solution_problem2_solution_l233_233817

noncomputable def problem1 : Real :=
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2)

noncomputable def problem2 : Real :=
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6)

theorem problem1_solution : problem1 = 0 := by
  sorry

theorem problem2_solution : problem2 = 6 := by
  sorry

end problem1_solution_problem2_solution_l233_233817


namespace max_common_ratio_arithmetic_geometric_sequence_l233_233429

open Nat

theorem max_common_ratio_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (k : ℕ) (q : ℝ) 
  (hk : k ≥ 2) (ha : ∀ n, a (n + 1) = a n + d)
  (hg : (a 1) * (a (2 * k)) = (a k) ^ 2) :
  q ≤ 2 :=
by
  sorry

end max_common_ratio_arithmetic_geometric_sequence_l233_233429


namespace find_a_l233_233969

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l233_233969


namespace real_roots_of_quadratic_l233_233532

theorem real_roots_of_quadratic (k : ℝ) : (k ≤ 0 ∨ 1 ≤ k) →
  ∃ x : ℝ, x^2 + 2 * k * x + k = 0 :=
by
  intro h
  sorry

end real_roots_of_quadratic_l233_233532


namespace union_complement_eq_l233_233307

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l233_233307


namespace maximum_value_of_linear_expression_l233_233045

theorem maximum_value_of_linear_expression (m n : ℕ) (h_sum : (m*(m + 1) + n^2 = 1987)) : 3 * m + 4 * n ≤ 221 :=
sorry

end maximum_value_of_linear_expression_l233_233045


namespace equation1_solution_equation2_solutions_l233_233161

theorem equation1_solution (x : ℝ) : (x - 2) * (x - 3) = x - 2 → (x = 2 ∨ x = 4) :=
by
  intro h
  have h1 : (x - 2) * (x - 3) - (x - 2) = 0 := by sorry
  have h2 : (x - 2) * (x - 4) = 0 := by sorry
  have h3 : x - 2 = 0 ∨ x - 4 = 0 := by sorry
  cases h3 with
  | inl h4 => left; exact eq_of_sub_eq_zero h4
  | inr h5 => right; exact eq_of_sub_eq_zero h5

theorem equation2_solutions (x : ℝ) : 2 * x^2 - 5 * x + 1 = 0 → (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by
  intro h
  have h1 : (-5)^2 - 4 * 2 * 1 = 17 := by sorry
  have h2 : 2 * x^2 - 5 * x + 1 = 2 * ((x - (5 + Real.sqrt 17) / 4) * (x - (5 - Real.sqrt 17) / 4)) := by sorry
  have h3 : (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) := by sorry
  exact h3

end equation1_solution_equation2_solutions_l233_233161


namespace finite_operations_invariant_final_set_l233_233373

theorem finite_operations (n : ℕ) (a : Fin n → ℕ) :
  ∃ N : ℕ, ∀ k, k > N → ((∃ i j, i ≠ j ∧ ¬ (a i ∣ a j ∨ a j ∣ a i)) → False) :=
sorry

theorem invariant_final_set (n : ℕ) (a : Fin n → ℕ) :
  ∃ b : Fin n → ℕ, (∀ i, ∃ j, b i = a j) ∧ ∀ (c : Fin n → ℕ), (∀ i, ∃ j, c i = a j) → c = b :=
sorry

end finite_operations_invariant_final_set_l233_233373


namespace greatest_radius_l233_233709

theorem greatest_radius (A : ℝ) (hA : A < 60 * Real.pi) : ∃ r : ℕ, r = 7 ∧ (r : ℝ) * (r : ℝ) < 60 :=
by
  sorry

end greatest_radius_l233_233709


namespace complement_union_l233_233883

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233883


namespace exists_power_of_two_with_last_n_digits_ones_and_twos_l233_233753

theorem exists_power_of_two_with_last_n_digits_ones_and_twos (N : ℕ) (hN : 0 < N) :
  ∃ k : ℕ, ∀ i < N, ∃ (d : ℕ), d = 1 ∨ d = 2 ∧ 
    (2^k % 10^N) / 10^i % 10 = d :=
sorry

end exists_power_of_two_with_last_n_digits_ones_and_twos_l233_233753


namespace sum_of_other_endpoint_coordinates_l233_233018

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l233_233018


namespace probability_point_above_parabola_l233_233765

theorem probability_point_above_parabola : 
  (∑ a b, if b > a + a * a then 1 else 0).toRat / (9 * 9) = 23 / 27 :=
sorry

end probability_point_above_parabola_l233_233765


namespace algebraic_expression_values_l233_233958

-- Defining the given condition
def condition (x y : ℝ) : Prop :=
  x^4 + 6 * x^2 * y + 9 * y^2 + 2 * x^2 + 6 * y + 4 = 7

-- Defining the target expression
def target_expression (x y : ℝ) : ℝ :=
  x^4 + 6 * x^2 * y + 9 * y^2 - 2 * x^2 - 6 * y - 1

-- Stating the theorem to be proved
theorem algebraic_expression_values (x y : ℝ) (h : condition x y) :
  target_expression x y = -2 ∨ target_expression x y = 14 :=
by
  sorry

end algebraic_expression_values_l233_233958


namespace vector_simplification_l233_233480

-- Define vectors AB, CD, AC, and BD
variables {V : Type*} [AddCommGroup V]

-- Given vectors
variables (AB CD AC BD : V)

-- Theorem to be proven
theorem vector_simplification :
  (AB - CD) - (AC - BD) = (0 : V) :=
sorry

end vector_simplification_l233_233480


namespace complement_union_M_N_l233_233952

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233952


namespace wrapping_third_roll_l233_233755

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l233_233755


namespace fraction_of_smaller_part_l233_233338

theorem fraction_of_smaller_part (A B : ℕ) (x : ℚ) (h1 : A + B = 66) (h2 : A = 50) (h3 : 0.40 * A = x * B + 10) : x = 5 / 8 :=
by
  sorry

end fraction_of_smaller_part_l233_233338


namespace solve_quadratic_eq_l233_233603

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 4 * x - 1 = 0) : x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

end solve_quadratic_eq_l233_233603


namespace complement_union_of_M_and_N_l233_233887

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233887


namespace maximize_perimeter_OIH_l233_233448

/-- In triangle ABC, given certain angles and side lengths, prove that
    angle ABC = 70° maximizes the perimeter of triangle OIH, where O, I,
    and H are the circumcenter, incenter, and orthocenter of triangle ABC. -/
theorem maximize_perimeter_OIH 
  (A : ℝ) (B : ℝ) (C : ℝ)
  (BC : ℝ) (AB : ℝ) (AC : ℝ)
  (BOC : ℝ) (BIC : ℝ) (BHC : ℝ) :
  A = 75 ∧ BC = 2 ∧ AB ≥ AC ∧
  BOC = 150 ∧ BIC = 127.5 ∧ BHC = 105 → 
  B = 70 :=
by
  sorry

end maximize_perimeter_OIH_l233_233448


namespace min_deg_q_l233_233425

-- Definitions of polynomials requirements
variables (p q r : Polynomial ℝ)

-- Given Conditions
def polynomials_relation : Prop := 5 * p + 6 * q = r
def deg_p : Prop := p.degree = 10
def deg_r : Prop := r.degree = 12

-- The main theorem we want to prove
theorem min_deg_q (h1 : polynomials_relation p q r) (h2 : deg_p p) (h3 : deg_r r) : q.degree ≥ 12 :=
sorry

end min_deg_q_l233_233425


namespace rebecca_tent_stakes_l233_233157

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l233_233157


namespace example_theorem_l233_233863

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233863


namespace greatest_possible_remainder_l233_233700

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end greatest_possible_remainder_l233_233700


namespace contrapositive_equiv_l233_233769

variable (a b : ℝ)

def original_proposition : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

def contrapositive_proposition : Prop := a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0

theorem contrapositive_equiv : original_proposition a b ↔ contrapositive_proposition a b :=
by
  sorry

end contrapositive_equiv_l233_233769


namespace b_is_geometric_T_sum_l233_233108

noncomputable def a (n : ℕ) : ℝ := 1/2 + (n-1) * (1/2)
noncomputable def S (n : ℕ) : ℝ := n * (1/2) + (n * (n-1) / 2) * (1/2)
noncomputable def b (n : ℕ) : ℝ := 4 ^ (a n)
noncomputable def c (n : ℕ) : ℝ := a n + b n
noncomputable def T (n : ℕ) : ℝ := (n * (n+1) / 4) + 2^(n+1) - 2

theorem b_is_geometric : ∀ n : ℕ, (n > 0) → b (n+1) / b n = 2 := by
  sorry

theorem T_sum : ∀ n : ℕ, T n = (n * (n + 1) / 4) + 2^(n + 1) - 2 := by
  sorry

end b_is_geometric_T_sum_l233_233108


namespace manuscript_copy_cost_l233_233088

theorem manuscript_copy_cost (total_cost : ℝ) (binding_cost : ℝ) (num_manuscripts : ℕ) (pages_per_manuscript : ℕ) (x : ℝ) :
  total_cost = 250 ∧ binding_cost = 5 ∧ num_manuscripts = 10 ∧ pages_per_manuscript = 400 →
  x = (total_cost - binding_cost * num_manuscripts) / (num_manuscripts * pages_per_manuscript) →
  x = 0.05 :=
by
  sorry

end manuscript_copy_cost_l233_233088


namespace sum_seven_terms_l233_233845

-- Define the arithmetic sequence and sum of first n terms
variable {a : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S : ℕ → ℝ} -- The sum of the first n terms S_n

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition: a_4 = 4
def a_4_eq_4 (a : ℕ → ℝ) : Prop :=
  a 4 = 4

-- Proposition we want to prove: S_7 = 28 given a_4 = 4
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_arithmetic_sequence a S)
  (h : a_4_eq_4 a) : 
  S 7 = 28 := 
sorry

end sum_seven_terms_l233_233845


namespace find_f_property_l233_233731

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_property :
  (f 0 = 3) ∧ (∀ x y : ℝ, f (xy) = f ((x^2 + y^2) / 2) + (x - y)^2) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 - 2 * x) :=
by
  intros hypothesis
  -- Proof would be placed here
  sorry

end find_f_property_l233_233731


namespace find_CD_l233_233540

theorem find_CD (C D : ℚ) :
  (∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 → (7 * x - 4) / (x ^ 2 - 9 * x - 36) = C / (x - 12) + D / (x + 3))
  → C = 16 / 3 ∧ D = 5 / 3 :=
by
  sorry

end find_CD_l233_233540


namespace price_of_sports_equipment_l233_233377

theorem price_of_sports_equipment (x y : ℕ) (a b : ℕ) :
  (2 * x + y = 330) → (5 * x + 2 * y = 780) → x = 120 ∧ y = 90 ∧
  (120 * a + 90 * b = 810) → a = 3 ∧ b = 5 :=
by
  intros h1 h2 h3
  sorry

end price_of_sports_equipment_l233_233377


namespace multiple_of_15_bounds_and_difference_l233_233672

theorem multiple_of_15_bounds_and_difference :
  ∃ (n : ℕ), 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 :=
by
  sorry

end multiple_of_15_bounds_and_difference_l233_233672


namespace fraction_home_l233_233081

-- Defining the conditions
def fractionFun := 5 / 13
def fractionYouth := 4 / 13

-- Stating the theorem to be proven
theorem fraction_home : 1 - (fractionFun + fractionYouth) = 4 / 13 := by
  sorry

end fraction_home_l233_233081


namespace inequality_1_inequality_2_inequality_3_inequality_4_l233_233744

noncomputable def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = Real.pi

theorem inequality_1 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin a + Real.sin b + Real.sin c ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_2 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos (a / 2) + Real.cos (b / 2) + Real.cos (c / 2) ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_3 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos a * Real.cos b * Real.cos c ≤ (1 / 8) :=
sorry

theorem inequality_4 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin (2 * a) + Real.sin (2 * b) + Real.sin (2 * c) ≤ Real.sin a + Real.sin b + Real.sin c :=
sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l233_233744


namespace find_a_l233_233968

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l233_233968


namespace union_complement_l233_233303

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l233_233303


namespace ratio_of_40_to_8_l233_233054

theorem ratio_of_40_to_8 : 40 / 8 = 5 := 
by
  sorry

end ratio_of_40_to_8_l233_233054


namespace new_person_weight_l233_233768

theorem new_person_weight (W x : ℝ) (h1 : (W - 55 + x) / 8 = (W / 8) + 2.5) : x = 75 := by
  -- Proof omitted
  sorry

end new_person_weight_l233_233768


namespace real_part_of_z_given_condition_l233_233241

open Complex

noncomputable def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z_given_condition :
  ∀ (z : ℂ), (i * (z + 1) = -3 + 2 * i) → real_part_of_z z = 1 :=
by
  intro z h
  sorry

end real_part_of_z_given_condition_l233_233241


namespace brendan_weekly_capacity_l233_233661

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end brendan_weekly_capacity_l233_233661


namespace num_diagonals_increase_by_n_l233_233807

-- Definitions of the conditions
def num_diagonals (n : ℕ) : ℕ := sorry  -- Consider f(n) to be a function that calculates diagonals for n-sided polygon

-- Lean 4 proof problem statement
theorem num_diagonals_increase_by_n (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n :=
sorry

end num_diagonals_increase_by_n_l233_233807


namespace complement_of_union_is_singleton_five_l233_233919

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233919


namespace lcm_condition_proof_l233_233841

theorem lcm_condition_proof (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2 * n)
  (h4 : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > n * 2 / 3 := 
sorry

end lcm_condition_proof_l233_233841


namespace complex_pow_difference_l233_233703

theorem complex_pow_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 12 - (1 - i) ^ 12 = 0 :=
  sorry

end complex_pow_difference_l233_233703


namespace aku_mother_packages_l233_233230

theorem aku_mother_packages
  (friends : Nat)
  (cookies_per_package : Nat)
  (cookies_per_child : Nat)
  (total_children : Nat)
  (birthday : Nat)
  (H_friends : friends = 4)
  (H_cookies_per_package : cookies_per_package = 25)
  (H_cookies_per_child : cookies_per_child = 15)
  (H_total_children : total_children = friends + 1)
  (H_birthday : birthday = 10) :
  (total_children * cookies_per_child) / cookies_per_package = 3 :=
by
  sorry

end aku_mother_packages_l233_233230


namespace n_cubed_plus_20n_div_48_l233_233595

theorem n_cubed_plus_20n_div_48 (n : ℕ) (h_even : n % 2 = 0) : (n^3 + 20 * n) % 48 = 0 :=
sorry

end n_cubed_plus_20n_div_48_l233_233595


namespace example_theorem_l233_233867

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233867


namespace count_integer_radii_l233_233819

theorem count_integer_radii (r : ℕ) (h : r < 150) :
  (∃ n : ℕ, n = 11 ∧ (∀ r, 0 < r ∧ r < 150 → (150 % r = 0)) ∧ (r ≠ 150)) := sorry

end count_integer_radii_l233_233819


namespace union_with_complement_l233_233315

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l233_233315


namespace union_complement_set_l233_233288

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l233_233288


namespace angus_token_count_l233_233395

theorem angus_token_count (elsa_tokens : ℕ) (token_value : ℕ) 
  (tokens_less_than_elsa_value : ℕ) (elsa_token_value_relation : elsa_tokens = 60) 
  (token_value_relation : token_value = 4) (tokens_less_value_relation : tokens_less_than_elsa_value = 20) :
  elsa_tokens - (tokens_less_than_elsa_value / token_value) = 55 :=
by
  rw [elsa_token_value_relation, token_value_relation, tokens_less_value_relation]
  norm_num
  sorry

end angus_token_count_l233_233395


namespace ab_value_l233_233552

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 240) :
  a * b = 255 :=
sorry

end ab_value_l233_233552


namespace angle_bisector_segments_l233_233576

noncomputable def divide_leg_BC (A B C S : Point ℝ) (BF FC : ℝ) :=
  -- Define the properties of the triangle ABC
  right_triangle ABC ∧ 
  distance A B = 1 ∧
  angle A B C = 30 ∧ 
  centroid S A B C ∧
  angle_bisector S A B C B F C ∧
  
  -- Define the lengths of the segments BF and FC
  BF + FC = distance B C ∧
  BF/FC = distance B S / distance S C

-- Define the theorem we want to state and prove
theorem angle_bisector_segments : 
  ∀ (A B C S : Point ℝ) (BF FC : ℝ), 
  divide_leg_BC A B C S BF FC → 
  BF / FC = sorry ∧ BF + FC = BC := 
-- The proof is left as an exercise
by 
  sorry
  -- The proof should involve setting up the coordinates of points, using trigonometric identities, and applying geometric theorems.

end angle_bisector_segments_l233_233576


namespace area_is_25_l233_233526

noncomputable def area_of_square (x : ℝ) : ℝ :=
  let side1 := 5 * x - 20
  let side2 := 25 - 4 * x
  if h : side1 = side2 then 
    side1 * side1
  else 
    0

theorem area_is_25 (x : ℝ) (h_eq : 5 * x - 20 = 25 - 4 * x) : area_of_square x = 25 :=
by
  sorry

end area_is_25_l233_233526


namespace union_with_complement_l233_233318

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l233_233318


namespace mean_goals_correct_l233_233492

-- Definitions based on problem conditions
def players_with_3_goals := 4
def players_with_4_goals := 3
def players_with_5_goals := 1
def players_with_6_goals := 2

-- The total number of goals scored
def total_goals := (3 * players_with_3_goals) + (4 * players_with_4_goals) + (5 * players_with_5_goals) + (6 * players_with_6_goals)

-- The total number of players
def total_players := players_with_3_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals

-- The mean number of goals
def mean_goals := total_goals.toFloat / total_players.toFloat

theorem mean_goals_correct : mean_goals = 4.1 := by
  sorry

end mean_goals_correct_l233_233492


namespace burger_cost_l233_233391

theorem burger_cost :
  ∃ b s f : ℕ, 4 * b + 2 * s + 3 * f = 480 ∧ 3 * b + s + 2 * f = 360 ∧ b = 80 :=
by
  sorry

end burger_cost_l233_233391


namespace range_of_c_l233_233422

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) : 1 < c ∧ c ≤ 4 / 3 :=
by
  sorry

end range_of_c_l233_233422


namespace sum_of_cubes_div_xyz_l233_233998

-- Given: x, y, z are non-zero real numbers, and x + y + z = 0.
-- Prove: (x^3 + y^3 + z^3) / (xyz) = 3.
theorem sum_of_cubes_div_xyz (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by
  sorry

end sum_of_cubes_div_xyz_l233_233998


namespace find_a_m_l233_233105

theorem find_a_m :
  ∃ a m : ℤ,
    (a = -2) ∧ (m = -1 ∨ m = 3) ∧ 
    (∀ x : ℝ, (a - 1) * x^2 + a * x + 1 = 0 → 
               (m^2 + m) * x^2 + 3 * m * x - 3 = 0) := sorry

end find_a_m_l233_233105


namespace complement_union_l233_233943

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233943


namespace complement_union_eq_singleton_five_l233_233869

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233869


namespace inclination_angle_is_45_degrees_l233_233044

open Real

-- Define the coordinates of the points.
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (-1, -1)

-- Define the slope m of the line passing through the points.
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the inclination angle using the arctangent function.
def inclination_angle (m : ℝ) : ℝ := arctan m

-- The main theorem to prove that the inclination angle of the line 
-- passing through the given points is 45 degrees.
theorem inclination_angle_is_45_degrees :
  inclination_angle (slope point1 point2) = π / 4 :=
by
  -- Provide the proof here.
  sorry

end inclination_angle_is_45_degrees_l233_233044


namespace meaningful_expression_range_l233_233257

theorem meaningful_expression_range (x : ℝ) : (¬ (x - 1 = 0)) ↔ (x ≠ 1) := 
by
  sorry

end meaningful_expression_range_l233_233257


namespace percentage_by_which_x_is_more_than_y_l233_233125

variable {z : ℝ} 

-- Define x and y based on the given conditions
def x (z : ℝ) : ℝ := 0.78 * z
def y (z : ℝ) : ℝ := 0.60 * z

-- The main theorem we aim to prove
theorem percentage_by_which_x_is_more_than_y (z : ℝ) : x z = y z + 0.30 * y z := by
  sorry

end percentage_by_which_x_is_more_than_y_l233_233125


namespace complement_union_l233_233899

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233899


namespace value_of_a_l233_233277

noncomputable def a : ℕ := 4

def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a*a}
def C : Set ℕ := {0, 1, 2, 4, 16}

theorem value_of_a : A ∪ B = C → a = 4 := by
  intro h
  sorry

end value_of_a_l233_233277


namespace problem_solution_l233_233104

-- Define the problem
noncomputable def a_b_sum : ℝ := 
  let a := 5
  let b := 3
  a + b

-- Theorem statement
theorem problem_solution (a b i : ℝ) (h1 : a + b * i = (11 - 7 * i) / (1 - 2 * i)) (hi : i * i = -1) :
  a + b = 8 :=
by sorry

end problem_solution_l233_233104


namespace no_integer_solutions_l233_233801

theorem no_integer_solutions (x y z : ℤ) (h₀ : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solutions_l233_233801


namespace every_algorithm_must_have_sequential_structure_l233_233359

def is_sequential_structure (alg : Type) : Prop := sorry -- This defines what a sequential structure is

def must_have_sequential_structure (alg : Type) : Prop :=
∀ alg, is_sequential_structure alg

theorem every_algorithm_must_have_sequential_structure :
  must_have_sequential_structure nat := sorry

end every_algorithm_must_have_sequential_structure_l233_233359


namespace example_theorem_l233_233866

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233866


namespace garage_sale_records_l233_233601

/--
Roberta started off with 8 vinyl records. Her friends gave her 12
records for her birthday and she bought some more at a garage
sale. It takes her 2 days to listen to 1 record. It will take her
100 days to listen to her record collection. Prove that she bought
30 records at the garage sale.
-/
theorem garage_sale_records :
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale
  records_bought = 30 := 
by
  -- Variable assumptions
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100

  -- Definitions
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale

  -- Conclusion to prove
  show records_bought = 30
  sorry

end garage_sale_records_l233_233601


namespace marcus_scored_50_percent_l233_233319

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l233_233319


namespace find_other_number_l233_233184

theorem find_other_number (x : ℕ) (h1 : 10 + x = 30) : x = 20 := by
  sorry

end find_other_number_l233_233184


namespace probability_of_drawing_2_black_and_2_white_l233_233199

def total_balls : ℕ := 17
def black_balls : ℕ := 9
def white_balls : ℕ := 8
def balls_drawn : ℕ := 4
def favorable_outcomes := (Nat.choose 9 2) * (Nat.choose 8 2)
def total_outcomes := Nat.choose 17 4
def probability_draw : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_drawing_2_black_and_2_white :
  probability_draw = 168 / 397 :=
by
  sorry

end probability_of_drawing_2_black_and_2_white_l233_233199


namespace union_with_complement_l233_233314

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l233_233314


namespace percentage_of_products_by_m1_l233_233719

theorem percentage_of_products_by_m1
  (x : ℝ)
  (h1 : 30 / 100 > 0)
  (h2 : 3 / 100 > 0)
  (h3 : 1 / 100 > 0)
  (h4 : 7 / 100 > 0)
  (h_total_defective : 
    0.036 = 
      (0.03 * x / 100) + 
      (0.01 * 30 / 100) + 
      (0.07 * (100 - x - 30) / 100)) :
  x = 40 :=
by
  sorry

end percentage_of_products_by_m1_l233_233719


namespace external_angle_bisector_l233_233466

open EuclideanGeometry

variables {A T C K L : Point}

theorem external_angle_bisector
    (h1 : is_internal_angle_bisector T L A T C) : is_external_angle_bisector T K A T C :=
sorry

end external_angle_bisector_l233_233466


namespace jills_daily_earnings_first_month_l233_233583

-- Definitions based on conditions
variable (x : ℕ) -- daily earnings in the first month
def total_earnings_first_month := 30 * x
def total_earnings_second_month := 30 * (2 * x)
def total_earnings_third_month := 15 * (2 * x)
def total_earnings_three_months := total_earnings_first_month x + total_earnings_second_month x + total_earnings_third_month x

-- The theorem we need to prove
theorem jills_daily_earnings_first_month
  (h : total_earnings_three_months x = 1200) : x = 10 :=
sorry

end jills_daily_earnings_first_month_l233_233583


namespace complement_union_l233_233898

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233898


namespace valid_reasonings_l233_233368

-- Define the conditions as hypotheses
def analogical_reasoning (R1 : Prop) : Prop := R1
def inductive_reasoning (R2 R4 : Prop) : Prop := R2 ∧ R4
def invalid_generalization (R3 : Prop) : Prop := ¬R3

-- Given the conditions, prove that the valid reasonings are (1), (2), and (4)
theorem valid_reasonings
  (R1 : Prop) (R2 : Prop) (R3 : Prop) (R4 : Prop)
  (h1 : analogical_reasoning R1) 
  (h2 : inductive_reasoning R2 R4) 
  (h3 : invalid_generalization R3) : 
  R1 ∧ R2 ∧ R4 :=
by 
  sorry

end valid_reasonings_l233_233368


namespace chair_cost_l233_233223

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l233_233223


namespace solve_problem_l233_233266

theorem solve_problem
    (product_trailing_zeroes : ∃ (x y z w v u p q r : ℕ), (10 ∣ (x * y * z * w * v * u * p * q * r)) ∧ B = 0)
    (digit_sequences : (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) % 10 = 8 ∧
                       (11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19) % 10 = 4 ∧
                       (21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29) % 10 = 4 ∧
                       (31 * 32 * 33 * 34 * 35) % 10 = 4 ∧
                       A = 2 ∧ B = 0)
    (divisibility_rule_11 : ∀ C D, (71 + C) - (68 + D) = 11 → C - D = -3 ∨ C - D = 8)
    (divisibility_rule_9 : ∀ C D, (139 + C + D) % 9 = 0 → C + D = 5 ∨ C + D = 14)
    (system_of_equations : ∀ C D, (C - D = -3 ∧ C + D = 5) → (C = 1 ∧ D = 4)) :
  A = 2 ∧ B = 0 ∧ C = 1 ∧ D = 4 :=
by
  sorry

end solve_problem_l233_233266


namespace positive_integer_solutions_of_inequality_l233_233356

theorem positive_integer_solutions_of_inequality :
  {x : ℕ | 2 * (x - 1) < 7 - x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_l233_233356


namespace knicks_from_knocks_l233_233704

variable (knicks knacks knocks : Type)
variable [HasSmul ℚ knicks] [HasSmul ℚ knacks] [HasSmul ℚ knocks]

variable (k1 : knicks) (k2 : knacks) (k3 : knocks)
variable (h1 : 5 • k1 = 3 • k2)
variable (h2 : 4 • k2 = 6 • k3)

theorem knicks_from_knocks : 36 • k3 = 40 • k1 :=
by {
  sorry
}

end knicks_from_knocks_l233_233704


namespace parallelogram_sum_l233_233402

open Finset

noncomputable def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (vertices : Finset (ℤ × ℤ)) : ℝ :=
  let [a, b, c, d] := vertices.sort (· ≤ ·)
  distance a b + distance b c + distance c d + distance d a

noncomputable def area (vertices : Finset (ℤ × ℤ)) : ℝ :=
  let [a, b, _, _] := vertices.sort (· ≤ ·)
  let base_vector := (b.1 - a.1, b.2 - a.2)
  let height_vector := (d.1 - a.1, d.2 - a.2)
  Real.abs (base_vector.1 * height_vector.2 - base_vector.2 * height_vector.1)

theorem parallelogram_sum (vertices : Finset (ℤ × ℤ)) : 
  (vertices = {(1,3), (5,6), (11,6), (7,3)}) → 
  (perimeter vertices + area vertices = 22 + 18) :=
by
  sorry

end parallelogram_sum_l233_233402


namespace sum_of_smallest_and_largest_is_correct_l233_233229

-- Define the conditions
def digits : Set ℕ := {0, 3, 4, 8}

-- Define the smallest and largest valid four-digit number using the digits
def smallest_number : ℕ := 3048
def largest_number : ℕ := 8430

-- Define the sum of the smallest and largest numbers
def sum_of_numbers : ℕ := smallest_number + largest_number

-- The theorem to be proven
theorem sum_of_smallest_and_largest_is_correct : 
  sum_of_numbers = 11478 := 
by
  -- Proof omitted
  sorry

end sum_of_smallest_and_largest_is_correct_l233_233229


namespace sum_of_three_consecutive_even_numbers_l233_233190

theorem sum_of_three_consecutive_even_numbers (a : ℤ) (h : a * (a + 2) * (a + 4) = 960) : a + (a + 2) + (a + 4) = 30 := by
  sorry

end sum_of_three_consecutive_even_numbers_l233_233190


namespace problem_l233_233293

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l233_233293


namespace complement_union_l233_233902

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233902


namespace surface_area_is_correct_l233_233822

structure CubicSolid where
  base_layer : ℕ
  second_layer : ℕ
  third_layer : ℕ
  top_layer : ℕ

def conditions : CubicSolid := ⟨4, 4, 3, 1⟩

theorem surface_area_is_correct : 
  (conditions.base_layer + conditions.second_layer + conditions.third_layer + conditions.top_layer + 7 + 7 + 3 + 3) = 28 := 
  by
  sorry

end surface_area_is_correct_l233_233822


namespace ratio_problem_l233_233966

variable (a b c d : ℚ)

theorem ratio_problem
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end ratio_problem_l233_233966


namespace arccos_zero_l233_233669

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l233_233669


namespace initial_ratio_l233_233579

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l233_233579


namespace steven_needs_more_seeds_l233_233170

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l233_233170


namespace range_of_a_l233_233979

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by
  sorry

end range_of_a_l233_233979


namespace quadratic_inequality_solution_l233_233440

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), x^2 - 9 * x + 14 ≤ 0 → 2 ≤ x ∧ x ≤ 7 :=
by
  intros x h
  sorry

end quadratic_inequality_solution_l233_233440


namespace chair_cost_l233_233224

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l233_233224


namespace percentage_of_students_owning_only_cats_l233_233127

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ) (students_owning_dogs : ℕ) (students_owning_cats : ℕ) (students_owning_both : ℕ)
  (h1 : total_students = 500) (h2 : students_owning_dogs = 200) (h3 : students_owning_cats = 100) (h4 : students_owning_both = 50) :
  ((students_owning_cats - students_owning_both) * 100 / total_students) = 10 :=
by
  -- Placeholder for proof
  sorry

end percentage_of_students_owning_only_cats_l233_233127


namespace complement_union_M_N_l233_233951

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233951


namespace product_of_square_roots_of_nine_l233_233117

theorem product_of_square_roots_of_nine (a b : ℝ) (ha : a^2 = 9) (hb : b^2 = 9) : a * b = -9 :=
sorry

end product_of_square_roots_of_nine_l233_233117


namespace people_owning_only_cats_and_dogs_l233_233047

theorem people_owning_only_cats_and_dogs 
  (total_people : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) 
  (total_snakes : ℕ) 
  (only_cats_and_dogs : ℕ) 
  (h1 : total_people = 89) 
  (h2 : only_dogs = 15) 
  (h3 : only_cats = 10) 
  (h4 : cats_dogs_snakes = 3) 
  (h5 : total_snakes = 59) 
  (h6 : total_people = only_dogs + only_cats + only_cats_and_dogs + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) : 
  only_cats_and_dogs = 5 := 
by 
  sorry

end people_owning_only_cats_and_dogs_l233_233047


namespace find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l233_233558

def U := Set ℝ
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem find_intersection (x : ℝ) : x ∈ A ∧ x ∈ B ↔ 4 < x ∧ x < 6 :=
by
  sorry

theorem complement_B (x : ℝ) : x ∉ B ↔ x ≥ 6 ∨ x ≤ -6 :=
by
  sorry

def A_minus_B : Set ℝ := {x | x ∈ A ∧ x ∉ B}

theorem find_A_minus_B (x : ℝ) : x ∈ A_minus_B ↔ x ≥ 6 :=
by
  sorry

theorem find_A_minus_A_minus_B (x : ℝ) : x ∈ (A \ A_minus_B) ↔ 4 < x ∧ x < 6 :=
by
  sorry

end find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l233_233558


namespace number_of_binders_l233_233150

-- Definitions of given conditions
def book_cost : Nat := 16
def binder_cost : Nat := 2
def notebooks_cost : Nat := 6
def total_cost : Nat := 28

-- Variable for the number of binders
variable (b : Nat)

-- Proposition that the number of binders Léa bought is 3
theorem number_of_binders (h : book_cost + binder_cost * b + notebooks_cost = total_cost) : b = 3 :=
by
  sorry

end number_of_binders_l233_233150


namespace average_is_5x_minus_10_implies_x_is_50_l233_233035

theorem average_is_5x_minus_10_implies_x_is_50 (x : ℝ) 
  (h : (1 / 3) * ((3 * x + 8) + (7 * x + 3) + (4 * x + 9)) = 5 * x - 10) : 
  x = 50 :=
by
  sorry

end average_is_5x_minus_10_implies_x_is_50_l233_233035


namespace f_zero_f_odd_f_inequality_solution_l233_233848

open Real

-- Given definitions
variables {f : ℝ → ℝ}
variable (h_inc : ∀ x y, x < y → f x < f y)
variable (h_eq : ∀ x y, y * f x - x * f y = x * y * (x^2 - y^2))

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := 
sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := 
sorry

-- Prove the range of x satisfying the given inequality
theorem f_inequality_solution : {x : ℝ | f (x^2 + 1) + f (3 * x - 5) < 0} = {x : ℝ | -4 < x ∧ x < 1} :=
sorry

end f_zero_f_odd_f_inequality_solution_l233_233848


namespace find_x0_l233_233587

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^2 + c
noncomputable def int_f (a c : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, f x a c

theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_eq : int_f a c = f x0 a c) : x0 = Real.sqrt 3 / 3 := sorry

end find_x0_l233_233587


namespace hike_took_one_hour_l233_233702

-- Define the constants and conditions
def initial_cups : ℕ := 6
def remaining_cups : ℕ := 1
def leak_rate : ℕ := 1 -- cups per hour
def drank_last_mile : ℚ := 1
def drank_first_3_miles_per_mile : ℚ := 2/3
def first_3_miles : ℕ := 3

-- Define the hike duration we want to prove
def hike_duration := 1

-- The total water drank
def total_drank := drank_last_mile + drank_first_3_miles_per_mile * first_3_miles

-- Prove the hike took 1 hour
theorem hike_took_one_hour :
  ∃ hours : ℕ, (initial_cups - remaining_cups = hours * leak_rate + total_drank) ∧ (hours = hike_duration) :=
by
  sorry

end hike_took_one_hour_l233_233702


namespace andrea_needs_to_buy_sod_squares_l233_233799

theorem andrea_needs_to_buy_sod_squares :
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  1500 = total_area / area_of_sod_square :=
by
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  sorry

end andrea_needs_to_buy_sod_squares_l233_233799


namespace largest_n_divisibility_condition_l233_233683

def S1 (n : ℕ) : ℕ := (n * (n + 1)) / 2
def S2 (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem largest_n_divisibility_condition : ∀ (n : ℕ), (n = 1) → (S2 n) % (S1 n) = 0 :=
by
  intros n hn
  rw [hn]
  sorry

end largest_n_divisibility_condition_l233_233683


namespace family_raised_percentage_l233_233482

theorem family_raised_percentage :
  ∀ (total_funds friends_percentage own_savings family_funds remaining_funds : ℝ),
    total_funds = 10000 →
    friends_percentage = 0.40 →
    own_savings = 4200 →
    remaining_funds = total_funds - (friends_percentage * total_funds) →
    family_funds = remaining_funds - own_savings →
    (family_funds / remaining_funds) * 100 = 30 :=
by
  intros total_funds friends_percentage own_savings family_funds remaining_funds
  intros h_total_funds h_friends_percentage h_own_savings h_remaining_funds h_family_funds
  sorry

end family_raised_percentage_l233_233482


namespace loan_proof_l233_233092

-- Definition of the conditions
def interest_rate_year_1 : ℝ := 0.10
def interest_rate_year_2 : ℝ := 0.12
def interest_rate_year_3 : ℝ := 0.14
def total_interest_paid : ℝ := 5400

-- Theorem proving the results
theorem loan_proof (P : ℝ) 
                   (annual_repayment : ℝ)
                   (remaining_principal : ℝ) :
  (interest_rate_year_1 * P) + 
  (interest_rate_year_2 * P) + 
  (interest_rate_year_3 * P) = total_interest_paid →
  3 * annual_repayment = total_interest_paid →
  remaining_principal = P →
  P = 15000 ∧ 
  annual_repayment = 1800 ∧ 
  remaining_principal = 15000 :=
by
  intros h1 h2 h3
  sorry

end loan_proof_l233_233092


namespace complement_union_of_M_and_N_l233_233888

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233888


namespace first_discount_l233_233215

theorem first_discount (P F : ℕ) (D₂ : ℝ) (D₁ : ℝ) 
  (hP : P = 150) 
  (hF : F = 105)
  (hD₂ : D₂ = 12.5)
  (hF_eq : F = P * (1 - D₁ / 100) * (1 - D₂ / 100)) : 
  D₁ = 20 :=
by
  sorry

end first_discount_l233_233215


namespace intersection_sum_l233_233204

noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1) ^ 2 / 3

theorem intersection_sum :
  ∃ a b : ℝ, f a = f (a - 4) ∧ b = f a ∧ a + b = 16 / 3 :=
sorry

end intersection_sum_l233_233204


namespace first_negative_term_at_14_l233_233262

-- Define the n-th term of the arithmetic sequence
def a_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Given values
def a₁ := 51
def d := -4

-- Proof statement
theorem first_negative_term_at_14 : ∃ n : ℕ, a_n a₁ d n < 0 ∧ ∀ m < n, a_n a₁ d m ≥ 0 :=
  by sorry

end first_negative_term_at_14_l233_233262


namespace workers_task_solution_l233_233050

-- Defining the variables for the number of days worked by A and B
variables (x y : ℕ)

-- Defining the total earnings for A and B
def total_earnings_A := 30
def total_earnings_B := 14

-- Condition: B worked 3 days less than A
def condition1 := y = x - 3

-- Daily wages of A and B
def daily_wage_A := total_earnings_A / x
def daily_wage_B := total_earnings_B / y

-- New scenario conditions
def new_days_A := x - 2
def new_days_B := y + 5

-- New total earnings in the scenario where they work changed days
def new_earnings_A := new_days_A * daily_wage_A
def new_earnings_B := new_days_B * daily_wage_B

-- Final proof to show the number of days worked and daily wages satisfying the conditions
theorem workers_task_solution 
  (h1 : y = x - 3)
  (h2 : new_earnings_A = new_earnings_B) 
  (hx : x = 10)
  (hy : y = 7) 
  (wageA : daily_wage_A = 3) 
  (wageB : daily_wage_B = 2) : 
  x = 10 ∧ y = 7 ∧ daily_wage_A = 3 ∧ daily_wage_B = 2 :=
by {
  sorry  -- Proof is skipped as instructed
}

end workers_task_solution_l233_233050


namespace edward_initial_money_l233_233227

variable (spent_books : ℕ) (spent_pens : ℕ) (money_left : ℕ)

theorem edward_initial_money (h_books : spent_books = 6) 
                             (h_pens : spent_pens = 16)
                             (h_left : money_left = 19) : 
                             spent_books + spent_pens + money_left = 41 := by
  sorry

end edward_initial_money_l233_233227


namespace erin_walks_less_l233_233096

variable (total_distance : ℕ)
variable (susan_distance : ℕ)

theorem erin_walks_less (h1 : total_distance = 15) (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 := by
  sorry

end erin_walks_less_l233_233096


namespace wall_length_is_7_5_meters_l233_233437

noncomputable def brick_volume : ℚ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℚ := 6000 * brick_volume

noncomputable def wall_cross_section : ℚ := 600 * 22.5

noncomputable def wall_length (total_volume : ℚ) (cross_section : ℚ) : ℚ := total_volume / cross_section

theorem wall_length_is_7_5_meters :
  wall_length total_brick_volume wall_cross_section = 7.5 := by
sorry

end wall_length_is_7_5_meters_l233_233437


namespace mary_added_peanuts_l233_233046

-- Defining the initial number of peanuts
def initial_peanuts : ℕ := 4

-- Defining the final number of peanuts
def total_peanuts : ℕ := 10

-- Defining the number of peanuts added by Mary
def peanuts_added : ℕ := total_peanuts - initial_peanuts

-- The proof problem is to show that Mary added 6 peanuts
theorem mary_added_peanuts : peanuts_added = 6 :=
by
  -- We leave the proof part as a sorry as per instruction
  sorry

end mary_added_peanuts_l233_233046


namespace alarm_prob_l233_233361

theorem alarm_prob (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.90) : 
  (1 - (1 - pA) * (1 - pB)) = 0.98 :=
by 
  sorry

end alarm_prob_l233_233361


namespace BigDigMiningCopperOutput_l233_233814

theorem BigDigMiningCopperOutput :
  (∀ (total_output : ℝ) (nickel_percentage : ℝ) (iron_percentage : ℝ) (amount_of_nickel : ℝ),
      nickel_percentage = 0.10 → 
      iron_percentage = 0.60 → 
      amount_of_nickel = 720 →
      total_output = amount_of_nickel / nickel_percentage →
      (1 - nickel_percentage - iron_percentage) * total_output = 2160) :=
sorry

end BigDigMiningCopperOutput_l233_233814


namespace inequality_interval_l233_233736

def differentiable_on_R (f : ℝ → ℝ) : Prop := Differentiable ℝ f
def strictly_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

theorem inequality_interval (f : ℝ → ℝ)
  (h_diff : differentiable_on_R f)
  (h_cond : ∀ x : ℝ, f x > deriv f x)
  (h_init : f 0 = 1) :
  ∀ x : ℝ, (x > 0) ↔ (f x / Real.exp x < 1) := 
by
  sorry

end inequality_interval_l233_233736


namespace marcus_percentage_of_team_points_l233_233324

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l233_233324


namespace polynomial_roots_arithmetic_progression_l233_233610

theorem polynomial_roots_arithmetic_progression (m n : ℝ)
  (h : ∃ a : ℝ, ∃ d : ℝ, ∃ b : ℝ,
   (a = b ∧ (b + d) + (b + 2*d) + (b + 3*d) + b = 0) ∧
   (b * (b + d) * (b + 2*d) * (b + 3*d) = 144) ∧
   b ≠ (b + d) ∧ (b + d) ≠ (b + 2*d) ∧ (b + 2*d) ≠ (b + 3*d)) :
  m = -40 := sorry

end polynomial_roots_arithmetic_progression_l233_233610


namespace max_average_hours_l233_233592

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l233_233592


namespace isabel_initial_amount_l233_233269

theorem isabel_initial_amount (X : ℝ) (h : X / 2 - X / 4 = 51) : X = 204 :=
sorry

end isabel_initial_amount_l233_233269


namespace compute_expression_l233_233821

theorem compute_expression : (3 + 5) ^ 2 + (3 ^ 2 + 5 ^ 2) = 98 := by
  sorry

end compute_expression_l233_233821


namespace evaluate_expression_at_neg3_l233_233677

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l233_233677


namespace curve_intersects_x_axis_at_4_over_5_l233_233723

-- Define the function for the curve
noncomputable def curve (x : ℝ) : ℝ :=
  (3 * x - 1) * (Real.sqrt (9 * x ^ 2 - 6 * x + 5) + 1) +
  (2 * x - 3) * (Real.sqrt (4 * x ^ 2 - 12 * x + 13) + 1)

-- Prove that curve(x) = 0 when x = 4 / 5
theorem curve_intersects_x_axis_at_4_over_5 :
  curve (4 / 5) = 0 :=
by
  sorry

end curve_intersects_x_axis_at_4_over_5_l233_233723


namespace steven_name_day_44_l233_233059

def W (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day_44 : ∃ n : ℕ, W n = 44 :=
  by 
  existsi 16
  sorry

end steven_name_day_44_l233_233059


namespace point_slope_form_l233_233243

theorem point_slope_form (k : ℝ) (p : ℝ × ℝ) (h_slope : k = 2) (h_point : p = (2, -3)) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ y = 2 * (x - 2) + (-3)) := 
sorry

end point_slope_form_l233_233243


namespace set_equality_l233_233557

theorem set_equality : 
  { x : ℕ | ∃ k : ℕ, 6 - x = k ∧ 8 % k = 0 } = { 2, 4, 5 } :=
by
  sorry

end set_equality_l233_233557


namespace complement_union_l233_233905

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233905


namespace example_theorem_l233_233861

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233861


namespace river_flow_rate_l233_233385

noncomputable def volume_per_minute : ℝ := 3200
noncomputable def depth_of_river : ℝ := 3
noncomputable def width_of_river : ℝ := 32
noncomputable def cross_sectional_area : ℝ := depth_of_river * width_of_river

noncomputable def flow_rate_m_per_minute : ℝ := volume_per_minute / cross_sectional_area
-- Conversion factors
noncomputable def minutes_per_hour : ℝ := 60
noncomputable def meters_per_km : ℝ := 1000

noncomputable def flow_rate_kmph : ℝ := (flow_rate_m_per_minute * minutes_per_hour) / meters_per_km

theorem river_flow_rate :
  flow_rate_kmph = 2 :=
by
  -- We skip the proof and use sorry to focus on the statement structure.
  sorry

end river_flow_rate_l233_233385


namespace final_apples_count_l233_233581

-- Definitions from the problem conditions
def initialApples : ℕ := 150
def soldToJill (initial : ℕ) : ℕ := initial * 30 / 100
def remainingAfterJill (initial : ℕ) := initial - soldToJill initial
def soldToJune (remaining : ℕ) : ℕ := remaining * 20 / 100
def remainingAfterJune (remaining : ℕ) := remaining - soldToJune remaining
def givenToFriend (current : ℕ) : ℕ := current - 2
def soldAfterFriend (current : ℕ) : ℕ := current * 10 / 100
def remainingAfterAll (current : ℕ) := current - soldAfterFriend current

theorem final_apples_count : remainingAfterAll (givenToFriend (remainingAfterJune (remainingAfterJill initialApples))) = 74 :=
by
  sorry

end final_apples_count_l233_233581


namespace factorize_difference_of_squares_l233_233833

theorem factorize_difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := 
by
  sorry

end factorize_difference_of_squares_l233_233833


namespace original_number_l233_233048

theorem original_number (x : ℤ) (h : x / 2 = 9) : x = 18 := by
  sorry

end original_number_l233_233048


namespace hh3_eq_2943_l233_233707

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Prove that h(h(3)) = 2943
theorem hh3_eq_2943 : h (h 3) = 2943 :=
by
  sorry

end hh3_eq_2943_l233_233707


namespace complement_of_union_is_singleton_five_l233_233915

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233915


namespace poly_constant_or_sum_constant_l233_233145

-- definitions of the polynomials as real-coefficient polynomials
variables (P Q R : Polynomial ℝ)

-- conditions
#check ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ) -- Considering 'constant' as 1 for simplicity

-- target
theorem poly_constant_or_sum_constant 
  (h : ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ)) :
  (∃ c : ℝ, ∀ x, P.eval x = c) ∨ (∃ c : ℝ, ∀ x, Q.eval x + R.eval x = c) :=
sorry

end poly_constant_or_sum_constant_l233_233145


namespace thm_300th_term_non_square_seq_l233_233634

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l233_233634


namespace range_satisfying_f_inequality_l233_233432

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (1 + |x|) - (1 / (1 + x^2))

theorem range_satisfying_f_inequality : 
  ∀ x : ℝ, (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) :=
by
  intro x hx
  sorry

end range_satisfying_f_inequality_l233_233432


namespace min_value_of_y_l233_233460

noncomputable def y (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - abs (x - 3)

theorem min_value_of_y : ∃ x : ℝ, (∀ x' : ℝ, y x' ≥ y x) ∧ y x = -1 :=
sorry

end min_value_of_y_l233_233460


namespace brother_combined_age_l233_233503

-- Define the ages of the brothers as integers
variable (x y : ℕ)

-- Define the condition given in the problem
def combined_age_six_years_ago : Prop := (x - 6) + (y - 6) = 100

-- State the theorem to prove the current combined age
theorem brother_combined_age (h : combined_age_six_years_ago x y): x + y = 112 :=
  sorry

end brother_combined_age_l233_233503


namespace rebecca_tent_stakes_l233_233159

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l233_233159


namespace select_students_with_A_or_B_l233_233759

-- Defining the parameters of the problem
def totalStudents : Nat := 9
def groupSize : Nat := 4
def excludedStudents : Nat := 7

-- Calculate total ways to choose 4 from 9
def totalWays : Nat := Nat.choose totalStudents groupSize

-- Calculate ways to choose 4 from 7 (excluding A and B)
def excludedWays : Nat := Nat.choose excludedStudents groupSize

-- Prove that the number of ways to select the group with at least one of A or B is 91
theorem select_students_with_A_or_B : 
  totalWays - excludedWays = 91 :=
by {
  -- Total ways to choose 4 out of 9
  have h1 : totalWays = Nat.choose 9 4 := rfl,
  -- Ways to choose 4 out of 7 excluding A and B
  have h2 : excludedWays = Nat.choose 7 4 := rfl,
  -- Perform the calculation
  calc
    totalWays - excludedWays 
      = Nat.choose 9 4 - Nat.choose 7 4  : by rw [h1, h2]
      = 126 - 35 : by rfl
      = 91 : by rfl
}

end select_students_with_A_or_B_l233_233759


namespace complement_union_l233_233947

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233947


namespace quincy_more_stuffed_animals_l233_233451

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l233_233451


namespace complement_union_M_N_l233_233953

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233953


namespace system1_solution_system2_solution_l233_233344

theorem system1_solution (p q : ℝ) 
  (h1 : p + q = 4)
  (h2 : 2 * p - q = 5) : 
  p = 3 ∧ q = 1 := 
sorry

theorem system2_solution (v t : ℝ)
  (h3 : 2 * v + t = 3)
  (h4 : 3 * v - 2 * t = 3) :
  v = 9 / 7 ∧ t = 3 / 7 :=
sorry

end system1_solution_system2_solution_l233_233344


namespace total_spent_by_pete_and_raymond_l233_233475

def pete_initial_amount := 250
def pete_spending_on_stickers := 4 * 5
def pete_spending_on_candy := 3 * 10
def pete_spending_on_toy_car := 2 * 25
def pete_spending_on_keychain := 5
def pete_total_spent := pete_spending_on_stickers + pete_spending_on_candy + pete_spending_on_toy_car + pete_spending_on_keychain
def raymond_initial_amount := 250
def raymond_left_dimes := 7 * 10
def raymond_left_quarters := 4 * 25
def raymond_left_nickels := 5 * 5
def raymond_left_pennies := 3 * 1
def raymond_total_left := raymond_left_dimes + raymond_left_quarters + raymond_left_nickels + raymond_left_pennies
def raymond_total_spent := raymond_initial_amount - raymond_total_left
def total_spent := pete_total_spent + raymond_total_spent

theorem total_spent_by_pete_and_raymond : total_spent = 157 := by
  have h1 : pete_total_spent = 105 := sorry
  have h2 : raymond_total_spent = 52 := sorry
  exact sorry

end total_spent_by_pete_and_raymond_l233_233475


namespace find_p_value_l233_233112

open Set

/-- Given the parabola C: y^2 = 2px with p > 0, point A(0, sqrt(3)),
    and point B on the parabola such that AB is perpendicular to AF,
    and |BF| = 4. Determine the value of p. -/
theorem find_p_value (p : ℝ) (h : p > 0) :
  ∃ p, p = 2 ∨ p = 6 :=
sorry

end find_p_value_l233_233112


namespace find_k_l233_233102

theorem find_k (n m : ℕ) (hn : n > 0) (hm : m > 0) (h : (1 : ℚ) / n^2 + 1 / m^2 = k / (n^2 + m^2)) : k = 4 :=
sorry

end find_k_l233_233102


namespace carl_lawn_area_l233_233086

theorem carl_lawn_area :
  ∃ (width height : ℤ), 
    (width + 1) + (height + 1) - 4 = 24 ∧
    3 * width = height ∧
    3 * ((width + 1) * 3) * ((height + 1) * 3) = 243 :=
by
  sorry

end carl_lawn_area_l233_233086


namespace avg_speed_BC_60_mph_l233_233063

theorem avg_speed_BC_60_mph 
  (d_AB : ℕ) (d_BC : ℕ) (avg_speed_total : ℚ) (time_ratio : ℚ) (t_AB : ℕ) :
  d_AB = 120 ∧ d_BC = 60 ∧ avg_speed_total = 45 ∧ time_ratio = 3 ∧
  t_AB = 3 → (d_BC / (t_AB / time_ratio) = 60) :=
by
  sorry

end avg_speed_BC_60_mph_l233_233063


namespace pizza_remained_l233_233130

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end pizza_remained_l233_233130


namespace find_z_l233_233114

def M (z : ℂ) : Set ℂ := {1, 2, z * Complex.I}
def N : Set ℂ := {3, 4}

theorem find_z (z : ℂ) (h : M z ∩ N = {4}) : z = -4 * Complex.I := by
  sorry

end find_z_l233_233114


namespace steven_needs_more_seeds_l233_233169

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l233_233169


namespace only_set_C_is_pythagorean_triple_l233_233060

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem only_set_C_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 15 20 25 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 1 3 5 :=
by {
  -- Proof goes here
  sorry
}

end only_set_C_is_pythagorean_triple_l233_233060


namespace exists_points_same_color_one_meter_apart_l233_233528

-- Predicate to describe points in the 2x2 square
structure Point where
  x : ℝ
  y : ℝ
  h_x : 0 ≤ x ∧ x ≤ 2
  h_y : 0 ≤ y ∧ y ≤ 2

-- Function to describe the color assignment
def color (p : Point) : Prop := sorry -- True = Black, False = White

-- The main theorem to be proven
theorem exists_points_same_color_one_meter_apart :
  ∃ p1 p2 : Point, color p1 = color p2 ∧ dist (p1.1, p1.2) (p2.1, p2.2) = 1 :=
by
  sorry

end exists_points_same_color_one_meter_apart_l233_233528


namespace complement_union_l233_233942

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233942


namespace complement_union_eq_singleton_five_l233_233875

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233875


namespace complement_union_l233_233896

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233896


namespace orange_balls_count_l233_233214

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end orange_balls_count_l233_233214


namespace trains_meet_in_time_l233_233363

noncomputable def time_to_meet (length1 length2 distance_between speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_to_meet 150 250 850 110 130 = 18.75 :=
by 
  -- here would go the proof steps, but since we are not required,
  sorry

end trains_meet_in_time_l233_233363


namespace root_magnitude_bound_l233_233006

open Complex

theorem root_magnitude_bound (n : ℕ) (hn : n ≥ 1) (a : Fin n → ℂ) 
  (z : Fin n → ℂ) (hz : ∀ i, eval₂ id z i = 0) :
  let A := Finset.max' (Finset.univ.image (λ i, abs (a i))) (by apply Finset.nonempty_univ) in
  ∀ j : Fin n, abs (z j) ≤ 1 + A :=
sorry

end root_magnitude_bound_l233_233006


namespace students_helped_on_fourth_day_l233_233653

theorem students_helped_on_fourth_day (total_books : ℕ) (books_per_student : ℕ)
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ)
  (H1 : total_books = 120) (H2 : books_per_student = 5)
  (H3 : day1_students = 4) (H4 : day2_students = 5) (H5 : day3_students = 6) :
  (total_books - (day1_students * books_per_student + day2_students * books_per_student + day3_students * books_per_student)) / books_per_student = 9 :=
by
  sorry

end students_helped_on_fourth_day_l233_233653


namespace complement_union_l233_233937

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233937


namespace complement_union_l233_233933

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233933


namespace choose_president_and_secretary_l233_233024

theorem choose_president_and_secretary (total_members boys girls : ℕ) (h_total : total_members = 30) (h_boys : boys = 18) (h_girls : girls = 12) : 
  (boys * girls = 216) :=
by
  sorry

end choose_president_and_secretary_l233_233024


namespace complement_union_eq_singleton_five_l233_233871

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233871


namespace min_max_values_l233_233276

theorem min_max_values (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 = 1) :
  1 ≤ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ∧ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ≤ 9/5 :=
by sorry

end min_max_values_l233_233276


namespace octagon_diagonals_l233_233066

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l233_233066


namespace pages_revised_only_once_l233_233042

theorem pages_revised_only_once 
  (total_pages : ℕ)
  (cost_per_page_first_time : ℝ)
  (cost_per_page_revised : ℝ)
  (revised_twice_pages : ℕ)
  (total_cost : ℝ)
  (pages_revised_only_once : ℕ) :
  total_pages = 100 →
  cost_per_page_first_time = 10 →
  cost_per_page_revised = 5 →
  revised_twice_pages = 30 →
  total_cost = 1400 →
  10 * (total_pages - pages_revised_only_once - revised_twice_pages) + 
  15 * pages_revised_only_once + 
  20 * revised_twice_pages = total_cost →
  pages_revised_only_once = 20 :=
by
  intros 
  sorry

end pages_revised_only_once_l233_233042


namespace function_equation_l233_233834

noncomputable def f (n : ℕ) : ℕ := sorry

theorem function_equation (h : ∀ m n : ℕ, m > 0 → n > 0 →
  f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) : 
  ∀ n : ℕ, n > 0 → f n = n := 
sorry

end function_equation_l233_233834


namespace union_complement_eq_target_l233_233310

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l233_233310


namespace sum_of_other_endpoint_l233_233015

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l233_233015


namespace non_square_300th_term_l233_233630

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l233_233630


namespace find_values_of_k_l233_233983

noncomputable def complex_distance (z w : ℂ) : ℝ := complex.abs (z - w)

theorem find_values_of_k (k : ℝ) :
  (∀ z : ℂ, (complex_distance z 2 = 3 * complex_distance z (-2)) ↔ (complex.abs z = k)) ->
  k = 1.5 ∨ k = 4.5 ∨ k = 5.5 :=
by sorry

end find_values_of_k_l233_233983


namespace hyperbola_constant_ellipse_constant_l233_233250

variables {a b : ℝ} (a_pos_b_gt_a : 0 < a ∧ a < b)
variables {A B : ℝ × ℝ} (on_hyperbola_A : A.1^2 / a^2 - A.2^2 / b^2 = 1)
variables (on_hyperbola_B : B.1^2 / a^2 - B.2^2 / b^2 = 1) (perp_OA_OB : A.1 * B.1 + A.2 * B.2 = 0)

-- Hyperbola statement
theorem hyperbola_constant :
  (1 / (A.1^2 + A.2^2)) + (1 / (B.1^2 + B.2^2)) = 1 / a^2 - 1 / b^2 :=
sorry

variables {C D : ℝ × ℝ} (on_ellipse_C : C.1^2 / a^2 + C.2^2 / b^2 = 1)
variables (on_ellipse_D : D.1^2 / a^2 + D.2^2 / b^2 = 1) (perp_OC_OD : C.1 * D.1 + C.2 * D.2 = 0)

-- Ellipse statement
theorem ellipse_constant :
  (1 / (C.1^2 + C.2^2)) + (1 / (D.1^2 + D.2^2)) = 1 / a^2 + 1 / b^2 :=
sorry

end hyperbola_constant_ellipse_constant_l233_233250


namespace initial_ratio_men_to_women_l233_233578

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l233_233578


namespace smallest_number_of_rectangles_l233_233791

-- Defining the given problem conditions
def rectangle_area : ℕ := 3 * 4
def smallest_square_side_length : ℕ := 12

-- Lean 4 statement to prove the problem
theorem smallest_number_of_rectangles 
    (h : ∃ n : ℕ, n * n = smallest_square_side_length * smallest_square_side_length)
    (h1 : ∃ m : ℕ, m * rectangle_area = smallest_square_side_length * smallest_square_side_length) :
    m = 9 :=
by
  sorry

end smallest_number_of_rectangles_l233_233791


namespace triangle_right_angle_l233_233715

variable {A B C a b c : ℝ}

theorem triangle_right_angle (h1 : Real.sin (A / 2) ^ 2 = (c - b) / (2 * c)) 
                             (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : 
                             a^2 + b^2 = c^2 :=
by
  sorry

end triangle_right_angle_l233_233715


namespace central_angle_radian_measure_l233_233850

namespace SectorProof

variables (R l : ℝ)
variables (α : ℝ)

-- Given conditions
def condition1 : Prop := 2 * R + l = 20
def condition2 : Prop := 1 / 2 * l * R = 9
def α_definition : Prop := α = l / R

-- Central angle result
theorem central_angle_radian_measure (h1 : condition1 R l) (h2 : condition2 R l) :
  α_definition α l R → α = 2 / 9 :=
by
  intro h_α
  -- proof steps would be here, but we skip them with sorry
  sorry

end SectorProof

end central_angle_radian_measure_l233_233850


namespace probability_black_then_red_l233_233387

/-- Definition of a standard deck -/
def standard_deck := {cards : Finset (Fin 52) // cards.card = 52}

/-- Definition of black cards in the deck -/
def black_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Definition of red cards in the deck -/
def red_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Probability of drawing the top card as black and the second card as red -/
def prob_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) : ℚ :=
  (26 * 26) / (52 * 51)

theorem probability_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) :
  prob_black_then_red deck black red = 13 / 51 :=
sorry

end probability_black_then_red_l233_233387


namespace knicks_eq_knocks_l233_233706

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l233_233706


namespace pizza_remained_l233_233131

theorem pizza_remained (total_people : ℕ) (fraction_eating_pizza : ℚ)
  (total_pizza_pieces : ℕ) (pieces_per_person : ℕ)
  (h1 : total_people = 15)
  (h2 : fraction_eating_pizza = 3 / 5)
  (h3 : total_pizza_pieces = 50)
  (h4 : pieces_per_person = 4) :
  total_pizza_pieces - (((total_people : ℚ) * fraction_eating_pizza).natCast * pieces_per_person) = 14 := by
  sorry

end pizza_remained_l233_233131


namespace solve_system1_l233_233795

theorem solve_system1 (x y : ℝ) :
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25 →
  x = 4 ∧ y = 3 :=
by
  sorry

end solve_system1_l233_233795


namespace convex_pentagon_largest_angle_l233_233202

theorem convex_pentagon_largest_angle 
  (x : ℝ)
  (h1 : (x + 2) + (2 * x + 3) + (3 * x + 6) + (4 * x + 5) + (5 * x + 4) = 540) :
  5 * x + 4 = 532 / 3 :=
by
  sorry

end convex_pentagon_largest_angle_l233_233202


namespace mrs_wong_initial_valentines_l233_233593

theorem mrs_wong_initial_valentines (x : ℕ) (given left : ℕ) (h_given : given = 8) (h_left : left = 22) (h_initial : x = left + given) : x = 30 :=
by
  rw [h_left, h_given] at h_initial
  exact h_initial

end mrs_wong_initial_valentines_l233_233593


namespace work_done_l233_233804

noncomputable def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

theorem work_done (W : ℝ) (h : W = ∫ x in (1:ℝ)..(5:ℝ), F x) : W = 112 :=
by sorry

end work_done_l233_233804


namespace fraction_ratio_l233_233439

theorem fraction_ratio
  (m n p q r : ℚ)
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 :=
by
  sorry

end fraction_ratio_l233_233439


namespace marble_287_is_blue_l233_233386

def marble_color (n : ℕ) : String :=
  if n % 15 < 6 then "blue"
  else if n % 15 < 11 then "green"
  else "red"

theorem marble_287_is_blue : marble_color 287 = "blue" :=
by
  sorry

end marble_287_is_blue_l233_233386


namespace range_of_m_l233_233711

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end range_of_m_l233_233711


namespace school_class_student_count_l233_233259

theorem school_class_student_count
  (num_classes : ℕ) (num_students : ℕ)
  (h_classes : num_classes = 30)
  (h_students : num_students = 1000)
  (h_max_students_per_class : ∀(n : ℕ), n < 30 → ∀(s : ℕ), s ≤ 33 → s ≤ 1000 / 30) :
  ∃ c, c ≤ num_classes ∧ ∃s, s ≥ 34 :=
by
  sorry

end school_class_student_count_l233_233259


namespace complement_of_union_is_singleton_five_l233_233916

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233916


namespace complement_union_eq_l233_233930

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233930


namespace eval_gg3_l233_233462

def g (x : ℕ) : ℕ := 3 * x^2 + 3 * x - 2

theorem eval_gg3 : g (g 3) = 3568 :=
by 
  sorry

end eval_gg3_l233_233462


namespace probability_correct_l233_233197

noncomputable def probability_at_most_3_sixes : ℝ :=
  let p : ℝ := 1/6 in
  let q : ℝ := 5/6 in
  (Nat.choose 10 0) * q^10 +
  (Nat.choose 10 1) * p * q^9 +
  (Nat.choose 10 2) * p^2 * q^8 +
  (Nat.choose 10 3) * p^3 * q^7

-- Now we need to state the theorem to prove
theorem probability_correct : probability_at_most_3_sixes =
  (Nat.choose 10 0) * (5 / 6)^10 +
  (Nat.choose 10 1) * (1 / 6) * (5 / 6)^9 +
  (Nat.choose 10 2) * (1 / 6)^2 * (5 / 6)^8 +
  (Nat.choose 10 3) * (1 / 6)^3 * (5 / 6)^7 := by
  sorry

end probability_correct_l233_233197


namespace average_playtime_l233_233590

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l233_233590


namespace range_of_2a_minus_b_l233_233560

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := 
sorry

end range_of_2a_minus_b_l233_233560


namespace max_distance_ellipse_line_l233_233228

theorem max_distance_ellipse_line :
  let P (α : ℝ) := (4 * Real.cos α, 2 * Real.sin α)
  let d (α : ℝ) := abs((4 * Real.cos α + 4 * Real.sin α - Real.sqrt 2) / Real.sqrt 5)
  ∃ α₀ : ℝ, d α₀ = Real.sqrt 10 :=
begin
  sorry
end

end max_distance_ellipse_line_l233_233228


namespace total_questions_attempted_l233_233135

theorem total_questions_attempted (C W T : ℕ) 
    (hC : C = 36) 
    (hScore : 120 = (4 * C) - W) 
    (hT : T = C + W) : 
    T = 60 := 
by 
  sorry

end total_questions_attempted_l233_233135


namespace sequence_formula_l233_233545

noncomputable def seq (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, (a n - 3) * a (n + 1) - a n + 4 = 0

theorem sequence_formula (a : ℕ+ → ℚ) (h : seq a) :
  ∀ n : ℕ+, a n = (2 * n - 1) / n :=
by
  sorry

end sequence_formula_l233_233545


namespace chord_length_eq_l233_233246

def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem chord_length_eq : 
  ∀ (x y : ℝ), 
  (line_eq x y) ∧ (circle_eq x y) → 
  ∃ l, l = 2 * Real.sqrt 3 :=
sorry

end chord_length_eq_l233_233246


namespace additional_time_needed_l233_233647

theorem additional_time_needed (total_parts apprentice_first_phase remaining_parts apprentice_rate master_rate combined_rate : ℕ)
  (h1 : total_parts = 500)
  (h2 : apprentice_first_phase = 45)
  (h3 : remaining_parts = total_parts - apprentice_first_phase)
  (h4 : apprentice_rate = 15)
  (h5 : master_rate = 20)
  (h6 : combined_rate = apprentice_rate + master_rate) :
  remaining_parts / combined_rate = 13 := 
by {
  sorry
}

end additional_time_needed_l233_233647


namespace least_grapes_in_heap_l233_233205

theorem least_grapes_in_heap :
  ∃ n : ℕ, (n % 19 = 1) ∧ (n % 23 = 1) ∧ (n % 29 = 1) ∧ n = 12209 :=
by
  sorry

end least_grapes_in_heap_l233_233205


namespace union_complement_eq_target_l233_233312

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l233_233312


namespace complement_union_l233_233932

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233932


namespace union_complement_eq_l233_233282

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l233_233282


namespace sum_of_solutions_equation_l233_233226

theorem sum_of_solutions_equation (x : ℝ) :
  let eqn := (4 * x + 7) * (3 * x - 8) = -12 in
  (eqn → root_sum (12 * x^2 - 11 * x - 68) = 11 / 12) :=
by
  sorry

end sum_of_solutions_equation_l233_233226


namespace cost_of_three_pencils_and_two_pens_l233_233770

theorem cost_of_three_pencils_and_two_pens
  (p q : ℝ)
  (h₁ : 8 * p + 3 * q = 5.20)
  (h₂ : 2 * p + 5 * q = 4.40) :
  3 * p + 2 * q = 2.5881 :=
by
  sorry

end cost_of_three_pencils_and_two_pens_l233_233770


namespace parameter_exists_solution_l233_233836

theorem parameter_exists_solution (b : ℝ) (h : b ≥ -2 * Real.sqrt 2 - 1 / 4) :
  ∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y) :=
by
  sorry

end parameter_exists_solution_l233_233836


namespace sequence_sum_l233_233615

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = 2^n) →
  (a 1 = S 1) ∧ (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 3 + a 4 = 12 :=
by
  sorry

end sequence_sum_l233_233615


namespace tangent_ln_at_origin_l233_233428

theorem tangent_ln_at_origin {k : ℝ} (h : ∀ x : ℝ, (k * x = Real.log x) → k = 1 / x) : k = 1 / Real.exp 1 :=
by
  sorry

end tangent_ln_at_origin_l233_233428


namespace trains_cross_each_other_in_5_76_seconds_l233_233195

noncomputable def trains_crossing_time (l1 l2 v1_kmh v2_kmh : ℕ) : ℚ :=
  let v1 := (v1_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let v2 := (v2_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let total_distance := (l1 : ℚ) + (l2 : ℚ)
  let relative_velocity := v1 + v2
  total_distance / relative_velocity

theorem trains_cross_each_other_in_5_76_seconds :
  trains_crossing_time 100 60 60 40 = 160 / 27.78 := by
  sorry

end trains_cross_each_other_in_5_76_seconds_l233_233195


namespace max_sum_abs_values_l233_233995

-- Define the main problem in Lean
theorem max_sum_abs_values (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 :=
by
  intros h
  sorry

end max_sum_abs_values_l233_233995


namespace identical_digits_has_37_factor_l233_233155

theorem identical_digits_has_37_factor (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 37 ∣ (100 * a + 10 * a + a) :=
by
  sorry

end identical_digits_has_37_factor_l233_233155


namespace geometric_sequence_first_term_and_ratio_l233_233357

theorem geometric_sequence_first_term_and_ratio (b : ℕ → ℚ) 
  (hb2 : b 2 = 37 + 1/3) 
  (hb6 : b 6 = 2 + 1/3) : 
  ∃ (b1 q : ℚ), b 1 = b1 ∧ (∀ n, b n = b1 * q^(n-1)) ∧ b1 = 224 / 3 ∧ q = 1 / 2 :=
by 
  sorry

end geometric_sequence_first_term_and_ratio_l233_233357


namespace maximum_number_of_workers_l233_233210

theorem maximum_number_of_workers :
  ∀ (n : ℕ), n ≤ 5 → 2 * n + 6 ≤ 16 :=
by
  intro n h
  have hn : n ≤ 5 := h
  linarith

end maximum_number_of_workers_l233_233210


namespace new_tax_rate_is_correct_l233_233713

noncomputable def new_tax_rate (old_rate : ℝ) (income : ℝ) (savings : ℝ) : ℝ := 
  let old_tax := old_rate * income / 100
  let new_tax := (income - savings) / income * old_tax
  let rate := new_tax / income * 100
  rate

theorem new_tax_rate_is_correct :
  ∀ (income : ℝ) (old_rate : ℝ) (savings : ℝ),
    old_rate = 42 →
    income = 34500 →
    savings = 4830 →
    new_tax_rate old_rate income savings = 28 := 
by
  intros income old_rate savings h1 h2 h3
  sorry

end new_tax_rate_is_correct_l233_233713


namespace exists_nat_concat_is_perfect_square_l233_233094

theorem exists_nat_concat_is_perfect_square :
  ∃ A : ℕ, ∃ n : ℕ, ∃ B : ℕ, (B * B = (10^n + 1) * A) :=
by sorry

end exists_nat_concat_is_perfect_square_l233_233094


namespace arccos_zero_l233_233668

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l233_233668


namespace ratio_of_sum_to_first_term_l233_233853

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - (2 ^ n)) / (1 - 2)

-- Main statement to be proven
theorem ratio_of_sum_to_first_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geo : geometric_sequence a 2) (h_sum : sum_of_first_n_terms a S) :
  S 3 / a 0 = 7 :=
sorry

end ratio_of_sum_to_first_term_l233_233853


namespace distance_of_ladder_to_building_l233_233381

theorem distance_of_ladder_to_building :
  ∀ (c a b : ℕ), c = 25 ∧ a = 20 ∧ (a^2 + b^2 = c^2) → b = 15 :=
by
  intros c a b h
  rcases h with ⟨hc, ha, hpyth⟩
  have h1 : c = 25 := hc
  have h2 : a = 20 := ha
  have h3 : a^2 + b^2 = c^2 := hpyth
  sorry

end distance_of_ladder_to_building_l233_233381


namespace external_angle_bisector_proof_l233_233464

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l233_233464


namespace _l233_233113

-- Here we define our conditions

def parabola (x y : ℝ) := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

example (y_P : ℝ) (hP : parabola 4 y_P) :
  dist (point_on_parabola y_P) focus = 6 := by
  -- Since we only need the theorem statement, we finish with sorry
  sorry

end _l233_233113


namespace find_x_of_arithmetic_mean_l233_233427

theorem find_x_of_arithmetic_mean (x : ℝ) (h : (6 + 13 + 18 + 4 + x) / 5 = 10) : x = 9 :=
by
  sorry

end find_x_of_arithmetic_mean_l233_233427


namespace solution_set_of_inequality_l233_233358

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l233_233358


namespace complement_union_l233_233935

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233935


namespace salary_spending_l233_233518

theorem salary_spending (S_A S_B : ℝ) (P_A P_B : ℝ) 
  (h1 : S_A = 4500) 
  (h2 : S_A + S_B = 6000)
  (h3 : P_B = 0.85) 
  (h4 : S_A * (1 - P_A) = S_B * (1 - P_B)) : 
  P_A = 0.95 :=
by
  -- Start proofs here
  sorry

end salary_spending_l233_233518


namespace TK_is_external_bisector_of_ATC_l233_233465

variable {A T C L K : Type} [LinearOrder A] [LinearOrder T] [LinearOrder C] [LinearOrder L] [LinearOrder K]
variable (triangle_ATC : Triangle A T C)
variable (point_T : Point T)
variable (point_L : Point L)
variable (point_K : Point K)
variable (internal_bisector_TL : AngleBisector (angle_ATC) T L)
variable (external_bisector_TK : AngleBisector (external_angle_ATC) T K)

theorem TK_is_external_bisector_of_ATC :
  external_bisector_TK = true :=
sorry

end TK_is_external_bisector_of_ATC_l233_233465


namespace street_length_l233_233523

theorem street_length
  (time_minutes : ℕ)
  (speed_kmph : ℕ)
  (length_meters : ℕ)
  (h1 : time_minutes = 12)
  (h2 : speed_kmph = 9)
  (h3 : length_meters = 1800) :
  length_meters = (speed_kmph * 1000 / 60) * time_minutes :=
by sorry

end street_length_l233_233523


namespace find_divisor_l233_233568

theorem find_divisor (x d : ℕ) (h1 : x ≡ 7 [MOD d]) (h2 : (x + 11) ≡ 18 [MOD 31]) : d = 31 := 
sorry

end find_divisor_l233_233568


namespace solve_D_l233_233263

-- Define the digits represented by each letter
variable (P M T D E : ℕ)

-- Each letter represents a different digit (0-9) and should be distinct
axiom distinct_digits : (P ≠ M) ∧ (P ≠ T) ∧ (P ≠ D) ∧ (P ≠ E) ∧ 
                        (M ≠ T) ∧ (M ≠ D) ∧ (M ≠ E) ∧ 
                        (T ≠ D) ∧ (T ≠ E) ∧ 
                        (D ≠ E)

-- Each letter is a digit from 0 to 9
axiom digit_range : 0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ M ∧ M ≤ 9 ∧ 
                    0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 
                    0 ≤ E ∧ E ≤ 9

-- Each column sums to the digit below it, considering carry overs from right to left
axiom column1 : T + T + E = E ∨ T + T + E = 10 + E
axiom column2 : E + D + T + (if T + T + E = 10 + E then 1 else 0) = P
axiom column3 : P + M + (if E + D + T + (if T + T + E = 10 + E then 1 else 0) = 10 + P then 1 else 0) = M

-- Prove that D = 4 given the above conditions
theorem solve_D : D = 4 :=
by sorry

end solve_D_l233_233263


namespace smallest_denominator_between_l233_233216

theorem smallest_denominator_between :
  ∃ (a b : ℕ), b > 0 ∧ a < b ∧ 6 / 17 < (a : ℚ) / b ∧ (a : ℚ) / b < 9 / 25 ∧ (∀ (c d : ℕ), d > 0 → c < d → 6 / 17 < (c : ℚ) / d → (c : ℚ) / d < 9 / 25 → b ≤ d) ∧ a = 5 ∧ b = 14 :=
by
  existsi 5
  existsi 14
  sorry

end smallest_denominator_between_l233_233216


namespace part1_part2_l233_233644

theorem part1 (x y : ℝ) (h1 : x + 3 * y = 26) (h2 : 2 * x + y = 22) : x = 8 ∧ y = 6 :=
by
  sorry

theorem part2 (m : ℝ) (h : 8 * m + 6 * (15 - m) ≤ 100) : m ≤ 5 :=
by
  sorry

end part1_part2_l233_233644


namespace corporate_event_handshakes_l233_233813

def GroupHandshakes (A B C : Nat) (knows_all_A : Nat) (knows_none : Nat) (C_knows_none : Nat) : Nat :=
  -- Handshakes between Group A and Group B
  let handshakes_AB := knows_none * A
  -- Handshakes within Group B
  let handshakes_B := (knows_none * (knows_none - 1)) / 2
  -- Handshakes between Group B and Group C
  let handshakes_BC := B * C_knows_none
  -- Total handshakes
  handshakes_AB + handshakes_B + handshakes_BC

theorem corporate_event_handshakes : GroupHandshakes 15 20 5 5 15 = 430 :=
by
  sorry

end corporate_event_handshakes_l233_233813


namespace complement_union_l233_233908

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233908


namespace complement_union_of_M_and_N_l233_233892

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233892


namespace union_complement_eq_target_l233_233311

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l233_233311


namespace logician1_max_gain_l233_233810

noncomputable def maxCoinsDistribution (logician1 logician2 logician3 : ℕ) := (logician1, logician2, logician3)

theorem logician1_max_gain 
  (total_coins : ℕ) 
  (coins1 coins2 coins3 : ℕ) 
  (H : total_coins = 10)
  (H1 : ¬ (coins1 = 9 ∧ coins2 = 0 ∧ coins3 = 1) → coins1 = 2):
  maxCoinsDistribution coins1 coins2 coins3 = (9, 0, 1) :=
by
  sorry

end logician1_max_gain_l233_233810


namespace arithmetic_mean_common_difference_l233_233109

theorem arithmetic_mean_common_difference (a : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 + a 4 = 2 * (a 2 + 1))
    : d = 2 := 
by 
  -- Proof is omitted as it is not required.
  sorry

end arithmetic_mean_common_difference_l233_233109


namespace total_number_of_pieces_paper_l233_233750

-- Define the number of pieces of paper each person picked up
def olivia_pieces : ℝ := 127.5
def edward_pieces : ℝ := 345.25
def sam_pieces : ℝ := 518.75

-- Define the total number of pieces of paper picked up
def total_pieces : ℝ := olivia_pieces + edward_pieces + sam_pieces

-- The theorem to be proven
theorem total_number_of_pieces_paper :
  total_pieces = 991.5 :=
by
  -- Sorry is used as we are not required to provide a proof here
  sorry

end total_number_of_pieces_paper_l233_233750


namespace correct_mutually_exclusive_events_l233_233072

variables (balls : Finset (Fin 4)) (draws : Finset (Finset (Fin 4)))

-- Define events
def is_at_least_one_white_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b ∈ drawn, b < 2

def is_both_white_balls (drawn : Finset (Fin 4)) : Prop :=
  drawn = {0, 1}

def is_at_least_one_red_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b ∈ drawn, b >= 2

def is_both_red_balls (drawn : Finset (Fin 4)) : Prop :=
  drawn = {2, 3}

def is_exactly_one_white_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b1 b2 ∈ drawn, b1 < 2 ∧ b2 >= 2

-- Define the event pairs
def event_pair_A (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_both_white_balls drawn

def event_pair_B (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_at_least_one_red_ball drawn

def event_pair_C (drawn : Finset (Fin 4)) : Prop :=
  is_exactly_one_white_ball drawn ∧ is_both_white_balls drawn

def event_pair_D (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_both_red_balls drawn

-- The goal is to prove that the correct answer is event pair D
theorem correct_mutually_exclusive_events : event_pair_D = true :=
by {
  sorry,  -- skipping the proof
}

end correct_mutually_exclusive_events_l233_233072


namespace find_a_l233_233960

theorem find_a (a : ℝ) 
  (h1 : ∀ x y : ℝ, 2*x + y - 2 = 0)
  (h2 : ∀ x y : ℝ, a*x + 4*y + 1 = 0)
  (perpendicular : ∀ (m1 m2 : ℝ), m1 = -2 → m2 = -a/4 → m1 * m2 = -1) :
  a = -2 :=
sorry

end find_a_l233_233960


namespace sec_neg_450_undefined_l233_233541

theorem sec_neg_450_undefined : ¬ ∃ x, x = 1 / Real.cos (-450 * Real.pi / 180) :=
by
  -- Proof skipped using 'sorry'
  sorry

end sec_neg_450_undefined_l233_233541


namespace jori_remaining_water_l233_233000

-- Having the necessary libraries for arithmetic and fractions.

-- Definitions directly from the conditions in a).
def initial_water_quantity : ℚ := 4
def used_water_quantity : ℚ := 9 / 4 -- Converted 2 1/4 to an improper fraction

-- The statement proving the remaining quantity of water is 1 3/4 gallons.
theorem jori_remaining_water : initial_water_quantity - used_water_quantity = 7 / 4 := by
  sorry

end jori_remaining_water_l233_233000


namespace solve_xy_l233_233119

theorem solve_xy (x y a b : ℝ) (h1 : x * y = 2 * b) (h2 : (1 / x^2) + (1 / y^2) = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := 
by 
  sorry

end solve_xy_l233_233119


namespace total_charge_for_2_hours_l233_233512

theorem total_charge_for_2_hours (A F : ℕ) (h1 : F = A + 35) (h2 : F + 4 * A = 350) : 
  F + A = 161 := 
by 
  sorry

end total_charge_for_2_hours_l233_233512


namespace value_of_x_plus_y_l233_233858

theorem value_of_x_plus_y (x y : ℝ) 
  (h1 : 2 * x - y = -1) 
  (h2 : x + 4 * y = 22) : 
  x + y = 7 :=
sorry

end value_of_x_plus_y_l233_233858


namespace paint_cost_l233_233355

theorem paint_cost (l : ℝ) (b : ℝ) (rate : ℝ) (area : ℝ) (cost : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : l = 18.9999683334125) 
  (h3 : rate = 3.00001) 
  (h4 : area = l * b) 
  (h5 : cost = area * rate) : 
  cost = 361.00 :=
by
  sorry

end paint_cost_l233_233355


namespace part_a_part_b_l233_233506

def happy (n : ℕ) : Prop :=
  ∃ (a b : ℤ), a^2 + b^2 = n

theorem part_a (t : ℕ) (ht : happy t) : happy (2 * t) := 
sorry

theorem part_b (t : ℕ) (ht : happy t) : ¬ happy (3 * t) := 
sorry

end part_a_part_b_l233_233506


namespace find_roots_and_m_l233_233849

theorem find_roots_and_m (m a : ℝ) (h_root : (-2)^2 - 4 * (-2) + m = 0) :
  m = -12 ∧ a = 6 :=
by
  sorry

end find_roots_and_m_l233_233849


namespace ellen_painted_roses_l233_233536

theorem ellen_painted_roses :
  ∀ (r : ℕ),
    (5 * 17 + 7 * r + 3 * 6 + 2 * 20 = 213) → (r = 10) :=
by
  intros r h
  sorry

end ellen_painted_roses_l233_233536


namespace fractions_non_integer_l233_233255

theorem fractions_non_integer (a b c d : ℤ) : 
  ∃ (a b c d : ℤ), 
    ¬((a-b) % 2 = 0 ∧ 
      (b-c) % 2 = 0 ∧ 
      (c-d) % 2 = 0 ∧ 
      (d-a) % 2 = 0) :=
sorry

end fractions_non_integer_l233_233255


namespace squareD_perimeter_l233_233605

-- Let perimeterC be the perimeter of square C
def perimeterC : ℝ := 40

-- Let sideC be the side length of square C
def sideC := perimeterC / 4

-- Let areaC be the area of square C
def areaC := sideC * sideC

-- Let areaD be the area of square D, which is one-third the area of square C
def areaD := (1 / 3) * areaC

-- Let sideD be the side length of square D
def sideD := Real.sqrt areaD

-- Let perimeterD be the perimeter of square D
def perimeterD := 4 * sideD

-- The theorem to prove
theorem squareD_perimeter (h : perimeterC = 40) (h' : areaD = (1 / 3) * areaC) : perimeterD = (40 * Real.sqrt 3) / 3 := by
  sorry

end squareD_perimeter_l233_233605


namespace arccos_zero_eq_pi_div_two_l233_233667

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l233_233667


namespace sum_of_other_endpoint_coordinates_l233_233020

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l233_233020


namespace negation_of_p_l233_233122

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 1 > 0

-- State the theorem that the negation of p is equivalent to the given existential statement
theorem negation_of_p :
  ¬p ↔ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0 :=
by
  sorry

end negation_of_p_l233_233122


namespace m_is_perfect_square_l233_233479

-- Given definitions and conditions
def is_odd (k : ℤ) : Prop := ∃ n : ℤ, k = 2 * n + 1

def is_perfect_square (m : ℕ) : Prop := ∃ a : ℕ, m = a * a

theorem m_is_perfect_square (k m n : ℕ) (h1 : (2 + Real.sqrt 3) ^ k = 1 + m + n * Real.sqrt 3)
  (h2 : 0 < m) (h3 : 0 < n) (h4 : 0 < k) (h5 : is_odd k) : is_perfect_square m := 
sorry

end m_is_perfect_square_l233_233479


namespace complement_of_union_is_singleton_five_l233_233914

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233914


namespace complement_union_eq_l233_233928

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233928


namespace parallel_vectors_x_value_l233_233688

theorem parallel_vectors_x_value (x : ℝ) :
  (∀ k : ℝ, k ≠ 0 → (4, 2) = (k * x, k * (-3))) → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l233_233688


namespace quadratic_to_general_form_l233_233401

theorem quadratic_to_general_form (x : ℝ) :
  ∃ b : ℝ, (∀ a c : ℝ, (a = 3) ∧ (c = 1) → (a * x^2 + c = 6 * x) → b = -6) :=
by
  sorry

end quadratic_to_general_form_l233_233401


namespace height_on_hypotenuse_of_right_triangle_l233_233712

theorem height_on_hypotenuse_of_right_triangle (a b : ℝ) (h_a : a = 2) (h_b : b = 3) :
  ∃ h : ℝ, h = (6 * Real.sqrt 13) / 13 :=
by
  sorry

end height_on_hypotenuse_of_right_triangle_l233_233712


namespace complement_union_M_N_l233_233954

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233954


namespace regular_polygon_sides_l233_233829

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l233_233829


namespace polynomials_divide_x15_minus_1_l233_233597

open Polynomial

-- Define the polynomial f(x) = x^15 - 1
def f : Polynomial ℤ := (X ^ 15) - 1

-- Now, we need to prove the main theorem
theorem polynomials_divide_x15_minus_1 :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 14 → ∃ g : Polynomial ℤ, degree g = k ∧ g ∣ f :=
by
  sorry

end polynomials_divide_x15_minus_1_l233_233597


namespace range_for_k_solutions_when_k_eq_1_l233_233232

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l233_233232


namespace count_three_digit_numbers_with_digit_sum_24_l233_233559

-- Define the conditions:
def isThreeDigitNumber (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c ≥ 100)

def digitSumEquals24 (a b c : ℕ) : Prop :=
  a + b + c = 24

-- State the theorem:
theorem count_three_digit_numbers_with_digit_sum_24 :
  (∃ (count : ℕ), count = 10 ∧ 
   ∀ (a b c : ℕ), isThreeDigitNumber a b c ∧ digitSumEquals24 a b c → (count = 10)) :=
sorry

end count_three_digit_numbers_with_digit_sum_24_l233_233559


namespace train_arrival_day_l233_233502

-- Definitions for the start time and journey duration
def start_time : ℕ := 0  -- early morning (0 hours) on Tuesday
def journey_duration : ℕ := 28  -- 28 hours

-- Proving the arrival time
theorem train_arrival_day (start_time journey_duration : ℕ) :
  journey_duration == 28 → 
  start_time == 0 → 
  (journey_duration / 24, journey_duration % 24) == (1, 4) → 
  true := 
by
  intros
  sorry

end train_arrival_day_l233_233502


namespace average_first_8_matches_l233_233607

/--
Assume we have the following conditions:
1. The average score for 12 matches is 48 runs.
2. The average score for the last 4 matches is 64 runs.
Prove that the average score for the first 8 matches is 40 runs.
-/
theorem average_first_8_matches (A1 A2 : ℕ) :
  (A1 / 12 = 48) → 
  (A2 / 4 = 64) →
  ((A1 - A2) / 8 = 40) :=
by
  sorry

end average_first_8_matches_l233_233607


namespace point_transformation_correct_l233_233447

-- Define the rectangular coordinate system O-xyz
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the point in the original coordinate system
def originalPoint : Point3D := { x := 1, y := -2, z := 3 }

-- Define the transformation function for the yOz plane
def transformToYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

-- Define the expected transformed point
def transformedPoint : Point3D := { x := -1, y := -2, z := 3 }

-- State the theorem to be proved
theorem point_transformation_correct :
  transformToYOzPlane originalPoint = transformedPoint :=
by
  sorry

end point_transformation_correct_l233_233447


namespace complement_union_l233_233939

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233939


namespace discount_percentage_l233_233992

theorem discount_percentage (shirts : ℕ) (total_cost : ℕ) (price_after_discount : ℕ) 
  (h1 : shirts = 3) (h2 : total_cost = 60) (h3 : price_after_discount = 12) : 
  ∃ discount_percentage : ℕ, discount_percentage = 40 := 
by 
  sorry

end discount_percentage_l233_233992


namespace sandy_spent_correct_amount_l233_233337

-- Definitions
def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def shoes_price : ℝ := 8.50
def accessories_price : ℝ := 10.75
def discount_rate : ℝ := 0.10
def coupon_amount : ℝ := 5.00
def tax_rate : ℝ := 0.075

-- Sum of all items before discounts and coupons
def total_before_discount : ℝ :=
  shorts_price + shirt_price + jacket_price + shoes_price + accessories_price

-- Total after applying the discount
def total_after_discount : ℝ :=
  total_before_discount * (1 - discount_rate)

-- Total after applying the coupon
def total_after_coupon : ℝ :=
  total_after_discount - coupon_amount

-- Total after applying the tax
def total_after_tax : ℝ :=
  total_after_coupon * (1 + tax_rate)

-- Theorem assertion that total amount spent is equal to $45.72
theorem sandy_spent_correct_amount : total_after_tax = 45.72 := by
  sorry

end sandy_spent_correct_amount_l233_233337


namespace complement_union_l233_233946

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233946


namespace roots_odd_even_l233_233976

theorem roots_odd_even (n : ℤ) (x1 x2 : ℤ) (h_eqn : x1^2 + (4 * n + 1) * x1 + 2 * n = 0) (h_eqn' : x2^2 + (4 * n + 1) * x2 + 2 * n = 0) :
  ((x1 % 2 = 0 ∧ x2 % 2 ≠ 0) ∨ (x1 % 2 ≠ 0 ∧ x2 % 2 = 0)) :=
sorry

end roots_odd_even_l233_233976


namespace sequence_periodicity_l233_233779

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : a 1 = 6 / 7)
  (h_rec : ∀ n, 0 ≤ a n ∧ a n < 1 → a (n+1) = if a n ≤ 1/2 then 2 * a n else 2 * a n - 1) :
  a 2017 = 6 / 7 :=
  sorry

end sequence_periodicity_l233_233779


namespace yogurt_amount_l233_233404

-- Conditions
def total_ingredients : ℝ := 0.5
def strawberries : ℝ := 0.2
def orange_juice : ℝ := 0.2

-- Question and Answer (Proof Goal)
theorem yogurt_amount : total_ingredients - strawberries - orange_juice = 0.1 := by
  -- Since calculation involves specifics, we add sorry to indicate the proof is skipped
  sorry

end yogurt_amount_l233_233404


namespace min_value_m_l233_233695

theorem min_value_m (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + a 1)
  (h_geometric : ∀ n, b (n + 1) = b 1 * (b 1 ^ n))
  (h_b1_mean : 2 * b 1 = a 1 + a 2)
  (h_a3 : a 3 = 5)
  (h_b3 : b 3 = a 4 + 1)
  (h_S_formula : ∀ n, S n = n^2)
  (h_S_le_b : ∀ n ≥ 4, S n ≤ b n) :
  ∃ m, ∀ n, (n ≥ m → S n ≤ b n) ∧ m = 4 := sorry

end min_value_m_l233_233695


namespace find_a_subtract_two_l233_233252

theorem find_a_subtract_two (a b : ℤ) 
    (h1 : 2 + a = 5 - b) 
    (h2 : 5 + b = 8 + a) : 
    2 - a = 2 := 
by
  sorry

end find_a_subtract_two_l233_233252


namespace dice_prob_not_one_l233_233058

theorem dice_prob_not_one : 
  let outcomes := [1, 2, 3, 4, 5, 6]
  let prob_not_1 := 5 / 6
  let total_outcomes := 6
  let number_of_dice := 4
  let prob := prob_not_1 ^ number_of_dice 
  prob = 625 / 1296 :=
by
  sorry

end dice_prob_not_one_l233_233058


namespace dogwood_trees_after_planting_l233_233787

-- Define the number of current dogwood trees and the number to be planted.
def current_dogwood_trees : ℕ := 34
def trees_to_be_planted : ℕ := 49

-- Problem statement to prove the total number of dogwood trees after planting.
theorem dogwood_trees_after_planting : current_dogwood_trees + trees_to_be_planted = 83 := by
  -- A placeholder for proof
  sorry

end dogwood_trees_after_planting_l233_233787


namespace height_difference_l233_233494

variables (H1 H2 H3 : ℕ)
variable (x : ℕ)
variable (h_ratio : H1 = 4 * x ∧ H2 = 5 * x ∧ H3 = 6 * x)
variable (h_lightest : H1 = 120)

theorem height_difference :
  (H1 + H3) - H2 = 150 :=
by
  -- Proof will go here
  sorry

end height_difference_l233_233494


namespace perpendicular_OA_PQ_eq_square_AP_2AD_OM_l233_233749

variables (A B C O D E F P Q M : Point) (triangle_ABC : Triangle A B C)
variables [acute_triangle : Triangle.AngleSumEqPI A B C]
noncomputable theory

def circumcenter (triangle_ABC : Triangle A B C) := O
def altitude_A (A B C D : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left A D B C
def altitude_B (A B C E : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left B E A C
def altitude_C (A B C F : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left C F A B
def midpoint_BC (B C M : Point) := Midpoint B C M

theorem perpendicular_OA_PQ
  (circumcenter_O : circumcenter triangle_ABC = O)
  (altitude_AD : altitude_A A B C D triangle_ABC)
  (altitude_BE : altitude_B A B C E triangle_ABC)
  (altitude_CF : altitude_C A B C F triangle_ABC)
  (points_PQ : ∃ (P Q : Point), Line.cut_circle EF P Q)
  : Line.perpendicular O A P Q :=
sorry

theorem eq_square_AP_2AD_OM
  (midpoint_M : midpoint_BC B C M)
  : (AP.square = 2 * AD.length * OM.length) :=
sorry

end perpendicular_OA_PQ_eq_square_AP_2AD_OM_l233_233749


namespace angle_P_is_90_degrees_l233_233267

open EuclideanGeometry

noncomputable def problem_triangle (P Q R S: Point) :=
  IsIsoscelesTriangle P Q R ∧
  (S ∈ Segment R P) ∧
  (IsAngleBisector Q S P R) ∧
  (distance Q S = distance Q R)

theorem angle_P_is_90_degrees (P Q R S : Point) 
  (h1 : IsIsoscelesTriangle P Q R)
  (h2 : S ∈ Segment R P)
  (h3 : IsAngleBisector Q S P R)
  (h4 : distance Q S = distance Q R) : 
  angle P ≤ \pi/2 := 
sorry

end angle_P_is_90_degrees_l233_233267


namespace circle_intersection_unique_point_l233_233984

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end circle_intersection_unique_point_l233_233984


namespace minimal_length_AX_XB_l233_233586

theorem minimal_length_AX_XB 
  (AA' BB' : ℕ) (A'B' : ℕ) 
  (h1 : AA' = 680) (h2 : BB' = 2000) (h3 : A'B' = 2010) 
  : ∃ X : ℕ, AX + XB = 3350 := 
sorry

end minimal_length_AX_XB_l233_233586


namespace no_solution_exists_l233_233781

theorem no_solution_exists :
  ¬ ∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) :=
by sorry

end no_solution_exists_l233_233781


namespace middle_number_probability_l233_233497

theorem middle_number_probability :
  let k := 6
  let A := { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }

  -- Conditions:
  (∀ (arrangement : Set (Finset ℕ)) (harrangement : arrangement.card = 11)
    (hmid : (arrangement.val.nth k).val > (arrangement.val.nth <$> (Finset.range k))),
    -- Prove question:
    let total_permutations := A.card.choose 6
    let configurations := (∑ (k : ℕ) in {2, 3, 4, 5, 6, 7}, (k - 1) * Nat.choose (11 - k) 4)
    P := 2 * 70 / 462
    P = 10 / 33 :=
  sorry

end middle_number_probability_l233_233497


namespace divides_lcm_condition_l233_233766

theorem divides_lcm_condition (x y : ℕ) (h₀ : 1 < x) (h₁ : 1 < y)
  (h₂ : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x := 
sorry

end divides_lcm_condition_l233_233766


namespace complement_union_l233_233907

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233907


namespace sin_B_plus_pi_over_6_eq_l233_233449

noncomputable def sin_b_plus_pi_over_6 (B : ℝ) : ℝ :=
  Real.sin B * (Real.sqrt 3 / 2) + (Real.sqrt (1 - (Real.sin B) ^ 2)) * (1 / 2)

theorem sin_B_plus_pi_over_6_eq :
  ∀ (A B : ℝ) (b c : ℝ),
    A = (2 * Real.pi / 3) →
    b = 1 →
    (1 / 2 * b * c * Real.sin A) = Real.sqrt 3 →
    c = 2 →
    sin_b_plus_pi_over_6 B = (2 * Real.sqrt 7 / 7) :=
by
  intros A B b c hA hb hArea hc
  sorry

end sin_B_plus_pi_over_6_eq_l233_233449


namespace all_numbers_same_parity_in_tame_array_all_numbers_equal_in_turbo_tame_array_l233_233656

-- Definitions for tame and turbo tame arrays
def is_tame (a : list ℤ) (h_len : a.length = 13) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 13 → ∃ l1 l2, l1.sum = l2.sum ∧ l1 ++ l2 = (a.removeAt i)

def is_turbo_tame (a : list ℤ) (h_len : a.length = 13) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 13 → ∃ l1 l2, l1.length = 6 ∧ l2.length = 6 ∧ l1.sum = l2.sum ∧ l1 ++ l2 = (a.removeAt i)

-- Proof statements
theorem all_numbers_same_parity_in_tame_array (a : list ℤ) (h_len : a.length = 13) (h_tame : is_tame a h_len) : 
  ∀ i1 i2, 0 ≤ i1 ∧ i1 < 13 ∧ 0 ≤ i2 ∧ i2 < 13 → (a.nth i1) % 2 = (a.nth i2) % 2 :=
sorry

theorem all_numbers_equal_in_turbo_tame_array (a : list ℤ) (h_len : a.length = 13) (h_turbo_tame : is_turbo_tame a h_len) : 
  ∀ i1 i2, 0 ≤ i1 ∧ i1 < 13 ∧ 0 ≤ i2 ∧ i2 < 13 → a.nth i1 = a.nth i2 :=
sorry

end all_numbers_same_parity_in_tame_array_all_numbers_equal_in_turbo_tame_array_l233_233656


namespace swim_time_l233_233522

-- Definitions based on conditions:
def speed_in_still_water : ℝ := 6.5 -- speed of the man in still water (km/h)
def distance_downstream : ℝ := 16 -- distance swam downstream (km)
def distance_upstream : ℝ := 10 -- distance swam upstream (km)
def time_downstream := 2 -- time taken to swim downstream (hours)
def time_upstream := 2 -- time taken to swim upstream (hours)

-- Defining the speeds taking the current into account:
def speed_downstream (c : ℝ) : ℝ := speed_in_still_water + c
def speed_upstream (c : ℝ) : ℝ := speed_in_still_water - c

-- Assumption that the time took for both downstream and upstream are equal
def time_eq (c : ℝ) : Prop :=
  distance_downstream / (speed_downstream c) = distance_upstream / (speed_upstream c)

-- The proof we need to establish:
theorem swim_time (c : ℝ) (h : time_eq c) : time_downstream = time_upstream := by
  sorry

end swim_time_l233_233522


namespace intersection_P_Q_l233_233435

def P (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = -x^2 + 2

def Q (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = x

theorem intersection_P_Q :
  { y : ℝ | P y } ∩ { y : ℝ | Q y } = { y : ℝ | y ≤ 2 } :=
by
  sorry

end intersection_P_Q_l233_233435


namespace other_endpoint_coordinates_sum_l233_233022

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l233_233022


namespace external_angle_bisector_of_triangle_l233_233468

variables {A T C L K : Type}
variables [IsTriangle A T C] [IsInternalAngleBisector T L (A T C)]

theorem external_angle_bisector_of_triangle 
  (h1 : is_internal_angle_bisector TL ATC) :
  is_external_angle_bisector TK ATC :=
sorry

end external_angle_bisector_of_triangle_l233_233468


namespace frequency_distribution_table_understanding_l233_233673

theorem frequency_distribution_table_understanding (size_sample_group : Prop) :
  (∃ (size_proportion : Prop) (corresponding_situation : Prop),
    size_sample_group → size_proportion ∧ corresponding_situation) :=
sorry

end frequency_distribution_table_understanding_l233_233673


namespace union_with_complement_l233_233316

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l233_233316


namespace complement_union_l233_233938

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233938


namespace range_of_a_l233_233244

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - (x^2 / 2) - a * x - 1

theorem range_of_a (x : ℝ) (a : ℝ) (h : 1 ≤ x) : (0 ≤ f a x) → (a ≤ Real.exp 1 - 3 / 2) :=
by
  sorry

end range_of_a_l233_233244


namespace fractions_problem_l233_233974

theorem fractions_problem (x y : ℚ) (hx : x = 2 / 3) (hy : y = 3 / 2) :
  (1 / 3) * x^5 * y^6 = 3 / 2 := by
  sorry

end fractions_problem_l233_233974


namespace smallest_x_for_perfect_cube_l233_233461

theorem smallest_x_for_perfect_cube (x N : ℕ) (hN : 1260 * x = N^3) (h_fact : 1260 = 2^2 * 3^2 * 5 * 7): x = 7350 := sorry

end smallest_x_for_perfect_cube_l233_233461


namespace winston_initial_gas_l233_233794

theorem winston_initial_gas (max_gas : ℕ) (store_gas : ℕ) (doctor_gas : ℕ) :
  store_gas = 6 → doctor_gas = 2 → max_gas = 12 → max_gas - (store_gas + doctor_gas) = 4 → max_gas = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end winston_initial_gas_l233_233794


namespace least_number_of_coins_l233_233057

theorem least_number_of_coins : ∃ (n : ℕ), 
  (n % 6 = 3) ∧ 
  (n % 4 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, (m % 6 = 3) ∧ (m % 4 = 1) ∧ (m % 7 = 2) → n ≤ m) :=
by
  exists 9
  simp
  sorry

end least_number_of_coins_l233_233057


namespace total_snakes_owned_l233_233981

theorem total_snakes_owned 
  (total_people : ℕ)
  (only_dogs only_cats only_birds only_snakes : ℕ)
  (cats_and_dogs birds_and_dogs birds_and_cats snakes_and_dogs snakes_and_cats snakes_and_birds : ℕ)
  (cats_dogs_snakes cats_dogs_birds cats_birds_snakes dogs_birds_snakes all_four_pets : ℕ)
  (h1 : total_people = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four_pets = 10) : 
  7 + 3 + 4 + 2 + 5 + 6 + 9 + 10 = 46 := 
sorry

end total_snakes_owned_l233_233981


namespace cyclic_quadrilateral_tangency_l233_233808

theorem cyclic_quadrilateral_tangency (a b c d x y : ℝ) (h_cyclic : a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120) 
  (h_tangency: x + y = 140) : |x - y| = 5 := 
sorry

end cyclic_quadrilateral_tangency_l233_233808


namespace non_square_300th_term_l233_233631

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l233_233631


namespace perimeter_of_square_D_l233_233606

-- Definition of the perimeter of square C
def perimeter_C := 40
-- Definition of the area of square D in terms of the area of square C
def area_C := ((perimeter_C / 4) ^ 2)
def area_D := area_C / 3
-- Define the side of square D in terms of its area
def side_D := Real.sqrt area_D
-- Prove the perimeter of square D
def perimeter_D := 4 * side_D

-- Statement to prove the perimeter of square D equals the given value
theorem perimeter_of_square_D :
  perimeter_D = 40 * Real.sqrt 3 / 3 :=
by
  sorry

end perimeter_of_square_D_l233_233606


namespace union_complement_eq_l233_233308

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l233_233308


namespace complement_union_of_M_and_N_l233_233894

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233894


namespace units_digit_6_power_l233_233638

theorem units_digit_6_power (n : ℕ) : (6^n % 10) = 6 :=
sorry

end units_digit_6_power_l233_233638


namespace three_hundredth_term_without_squares_l233_233632

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l233_233632


namespace problem_l233_233290

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l233_233290


namespace two_pow_2001_mod_127_l233_233364

theorem two_pow_2001_mod_127 : (2^2001) % 127 = 64 := 
by
  sorry

end two_pow_2001_mod_127_l233_233364


namespace product_of_functions_l233_233245

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_of_functions (x : ℝ) (h : x ≠ 0) : f x * g x = -6 * x + 2 := by
  sorry

end product_of_functions_l233_233245


namespace empty_pencil_cases_l233_233619

theorem empty_pencil_cases (total_cases pencil_cases pen_cases both_cases : ℕ) 
  (h1 : total_cases = 10)
  (h2 : pencil_cases = 5)
  (h3 : pen_cases = 4)
  (h4 : both_cases = 2) : total_cases - (pencil_cases + pen_cases - both_cases) = 3 := by
  sorry

end empty_pencil_cases_l233_233619


namespace num_solutions_zero_l233_233411

open Matrix

noncomputable def num_solutions : ℕ :=
  if ∃ (a b c d : ℝ), (λ M : Matrix (Fin 2) (Fin 2) ℝ, M⁻¹ = ![![1 / (a + d), 1 / (b + c)], ![1 / (c + b), 1 / (a + d)]] ) (Matrix.fin2x2 a b c d)
  then 1 else 0

theorem num_solutions_zero : num_solutions = 0 := by 
  sorry

end num_solutions_zero_l233_233411


namespace symmetric_point_about_x_l233_233488

-- Define the coordinates of the point A
def A : ℝ × ℝ := (-2, 3)

-- Define the function that computes the symmetric point about the x-axis
def symmetric_about_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The concrete symmetric point of A
def A' := symmetric_about_x A

-- The original problem and proof statement
theorem symmetric_point_about_x :
  A' = (-2, -3) :=
by
  -- Proof goes here
  sorry

end symmetric_point_about_x_l233_233488


namespace hyperbola_equation_l233_233786

noncomputable def distance_between_vertices : ℝ := 8
noncomputable def eccentricity : ℝ := 5 / 4

theorem hyperbola_equation :
  ∃ a b c : ℝ, 2 * a = distance_between_vertices ∧ 
               c = a * eccentricity ∧ 
               b^2 = c^2 - a^2 ∧ 
               (a = 4 ∧ c = 5 ∧ b^2 = 9) ∧ 
               ∀ x y : ℝ, (x^2 / (a:ℝ)^2) - (y^2 / (b:ℝ)^2) = 1 :=
by 
  sorry

end hyperbola_equation_l233_233786


namespace log_floor_probability_l233_233335

-- We define the conditions: x and y are real numbers in the interval (0, 4)
def uniform_interval (a b : ℝ) : set ℝ := {x | a < x ∧ x < b}

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
  let distribution : measure ℝ := volume.restrict (uniform_interval 0 4) in
  probability_space ℝ distribution,
  ∫⁺ x, ∫⁴ y, indicator (λ (z : ℝ × ℝ), ⌊log 4 (fst z)⌋ = ⌊log 4 (snd z)⌋) (λ _, 1) (x, y) ∂distribution

-- The claim to be proved that the probability is 5 / 8
theorem log_floor_probability: probability_event = 5 / 8 := 
by sorry

end log_floor_probability_l233_233335


namespace total_value_of_item_l233_233383

theorem total_value_of_item (V : ℝ) 
  (h1 : ∃ V > 1000, 
              0.07 * (V - 1000) + 
              (if 55 > 50 then (55 - 50) * 0.15 else 0) + 
              0.05 * V = 112.70) :
  V = 1524.58 :=
by 
  sorry

end total_value_of_item_l233_233383


namespace find_original_sum_of_money_l233_233372

theorem find_original_sum_of_money
  (R : ℝ)
  (P : ℝ)
  (h1 : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 63) :
  P = 2100 :=
sorry

end find_original_sum_of_money_l233_233372


namespace triangle_ineq_l233_233070

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 2 * (a^2 + b^2) > c^2 := 
by 
  sorry

end triangle_ineq_l233_233070


namespace candy_store_truffle_price_l233_233376

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price_l233_233376


namespace find_p_q_l233_233739

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l233_233739


namespace find_x_l233_233191

-- Let \( x \) be a real number.
variable (x : ℝ)

-- Condition given in the problem.
def condition : Prop := x = (3 / 7) * x + 200

-- The main statement to be proved.
theorem find_x (h : condition x) : x = 350 :=
  sorry

end find_x_l233_233191


namespace union_of_A_and_B_l233_233423

open Set

variable {x : ℝ}

-- Define sets A and B based on the given conditions
def A : Set ℝ := { x | 0 < 3 - x ∧ 3 - x ≤ 2 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end union_of_A_and_B_l233_233423


namespace Oliver_9th_l233_233129

def person := ℕ → Prop

axiom Ruby : person
axiom Oliver : person
axiom Quinn : person
axiom Pedro : person
axiom Nina : person
axiom Samuel : person
axiom place : person → ℕ → Prop

-- Conditions given in the problem
axiom Ruby_Oliver : ∀ n, place Ruby n → place Oliver (n + 7)
axiom Quinn_Pedro : ∀ n, place Quinn n → place Pedro (n - 2)
axiom Nina_Oliver : ∀ n, place Nina n → place Oliver (n + 3)
axiom Pedro_Samuel : ∀ n, place Pedro n → place Samuel (n - 3)
axiom Samuel_Ruby : ∀ n, place Samuel n → place Ruby (n + 2)
axiom Quinn_5th : place Quinn 5

-- Question: Prove that Oliver finished in 9th place
theorem Oliver_9th : place Oliver 9 :=
sorry

end Oliver_9th_l233_233129


namespace amount_of_bill_l233_233194

theorem amount_of_bill (TD R FV T : ℝ) (hTD : TD = 270) (hR : R = 16) (hT : T = 9/12) 
(h_formula : TD = (R * T * FV) / (100 + (R * T))) : FV = 2520 :=
by
  sorry

end amount_of_bill_l233_233194


namespace largest_divisor_for_consecutive_seven_odds_l233_233508

theorem largest_divisor_for_consecutive_seven_odds (n : ℤ) (h_even : 2 ∣ n) (h_pos : 0 < n) : 
  105 ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) :=
sorry

end largest_divisor_for_consecutive_seven_odds_l233_233508


namespace find_abc_squares_l233_233144

variable (a b c x : ℕ)

theorem find_abc_squares (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 99 * (c - a) = 65 * x) (h4 : 495 = 65 * x) : a^2 + b^2 + c^2 = 53 :=
  sorry

end find_abc_squares_l233_233144


namespace mean_greater_than_median_by_six_l233_233065

theorem mean_greater_than_median_by_six (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 :=
by
  sorry

end mean_greater_than_median_by_six_l233_233065


namespace problem1_problem2_l233_233519

-- Definitions
def total_questions := 5
def multiple_choice := 3
def true_false := 2
def total_outcomes := total_questions * (total_questions - 1)

-- (1) Probability of A drawing a true/false question and B drawing a multiple-choice question
def favorable_outcomes_1 := true_false * multiple_choice

-- (2) Probability of at least one of A or B drawing a multiple-choice question
def unfavorable_outcomes_2 := true_false * (true_false - 1)

-- Statements to be proved
theorem problem1 : favorable_outcomes_1 / total_outcomes = 3 / 10 := by sorry

theorem problem2 : 1 - (unfavorable_outcomes_2 / total_outcomes) = 9 / 10 := by sorry

end problem1_problem2_l233_233519


namespace max_remainder_is_10_l233_233701

theorem max_remainder_is_10 (x : ℕ) (h : x % 11 ≠ 0) : x % 11 = 10 :=
begin
  sorry
end

end max_remainder_is_10_l233_233701


namespace sum_of_D_coordinates_l233_233751

-- Definition of the midpoint condition
def is_midpoint (N C D : ℝ × ℝ) : Prop :=
  N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2

-- Given points
def N : ℝ × ℝ := (5, -1)
def C : ℝ × ℝ := (11, 10)

-- Statement of the problem
theorem sum_of_D_coordinates :
  ∃ D : ℝ × ℝ, is_midpoint N C D ∧ (D.1 + D.2 = -13) :=
  sorry

end sum_of_D_coordinates_l233_233751


namespace BD_eq_DE_l233_233347

-- Given data definitions
def is_isosceles_triangle (ABC : Triangle) : Prop := 
  ABC.is_isosceles ∧ ABC.angle_ABC = 108

def angle_bisector (A B C D : Point) (ABC : Triangle) : Prop := 
  angle_eq (line A D).angle_with (line B C) (Bachelor_720.B.part_cube \C) 54

def perpendicular_from_D (D : Point) (AD : Line) (E : Point) (AC : Line) : Prop := 
  D.is_foot_of_perpendicular AD ∧ E ∈ intersection_of_perpendicular AD AC ∧ 
  AC.angle_between AD = 90

-- Prove the stated geometric relationships
theorem BD_eq_DE (A B C D E : Point) (ABC : Triangle) :
  is_isosceles_triangle ABC → angle_bisector A B C D ABC → 
  perpendicular_from_D D (angle_bisector_line A D) E (line A C) → 
  dist (B, D) = dist (D, E) :=
by
  intros
  sorry

end BD_eq_DE_l233_233347


namespace time_for_B_alone_l233_233075

theorem time_for_B_alone (h1 : 4 * (1/15 + 1/x) = 7/15) : x = 20 :=
sorry

end time_for_B_alone_l233_233075


namespace find_certain_number_l233_233126

noncomputable def certain_number (x : ℝ) : Prop :=
  3005 - 3000 + x = 2705

theorem find_certain_number : ∃ x : ℝ, certain_number x ∧ x = 2700 :=
by
  use 2700
  unfold certain_number
  sorry

end find_certain_number_l233_233126


namespace number_of_proper_subsets_l233_233496

theorem number_of_proper_subsets (S : Finset ℕ) (h : S = {1, 2, 3, 4}) : S.powerset.card - 1 = 15 := by
  sorry

end number_of_proper_subsets_l233_233496


namespace oak_trees_remaining_l233_233789

theorem oak_trees_remaining (initial_trees cut_down_trees remaining_trees : ℕ)
  (h1 : initial_trees = 9)
  (h2 : cut_down_trees = 2)
  (h3 : remaining_trees = initial_trees - cut_down_trees) :
  remaining_trees = 7 :=
by 
  sorry

end oak_trees_remaining_l233_233789


namespace compute_expression_l233_233400

theorem compute_expression : (46 + 15)^2 - (46 - 15)^2 = 2760 :=
by
  sorry

end compute_expression_l233_233400


namespace find_a_plus_b_l233_233561

theorem find_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 :=
sorry

end find_a_plus_b_l233_233561


namespace minimum_value_of_expression_l233_233434

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression_l233_233434


namespace ship_speeds_l233_233049

theorem ship_speeds (x : ℝ) 
  (h1 : (2 * x) ^ 2 + (2 * (x + 3)) ^ 2 = 174 ^ 2) :
  x = 60 ∧ x + 3 = 63 :=
by
  sorry

end ship_speeds_l233_233049


namespace union_complement_eq_l233_233297

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l233_233297


namespace bridge_length_is_correct_l233_233655

noncomputable def speed_km_per_hour_to_m_per_s (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed_m_per_s time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

def bridge_length (total_distance train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem bridge_length_is_correct : 
  let train_length := 110 
  let speed_kmph := 72
  let time_s := 12.099
  let speed_m_per_s := speed_km_per_hour_to_m_per_s speed_kmph
  let total_distance := total_distance_covered speed_m_per_s time_s
  bridge_length total_distance train_length = 131.98 := 
by
  sorry

end bridge_length_is_correct_l233_233655


namespace example_theorem_l233_233862

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233862


namespace probability_of_selection_l233_233716

-- defining necessary parameters and the systematic sampling method
def total_students : ℕ := 52
def selected_students : ℕ := 10
def exclusion_probability := 2 / total_students
def inclusion_probability_exclude := selected_students / (total_students - 2)
def final_probability := (1 - exclusion_probability) * inclusion_probability_exclude

-- the main theorem stating the probability calculation
theorem probability_of_selection :
  final_probability = 5 / 26 :=
by
  -- we skip the proof part and end with sorry since it is not required
  sorry

end probability_of_selection_l233_233716


namespace square_of_binomial_is_25_l233_233971

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l233_233971


namespace find_first_discount_percentage_l233_233378

def first_discount_percentage 
  (price_initial : ℝ) 
  (price_final : ℝ) 
  (discount_x : ℝ) 
  : Prop := 
  price_initial * (1 - discount_x / 100) * 0.9 * 0.95 = price_final

theorem find_first_discount_percentage :
  first_discount_percentage 9941.52 6800 20.02 :=
by
  sorry

end find_first_discount_percentage_l233_233378


namespace custom_op_example_l233_233253

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_example : (custom_op 7 4) - (custom_op 4 7) = -9 :=
by
  sorry

end custom_op_example_l233_233253


namespace total_trip_cost_l233_233134

def distance_AC : ℝ := 4000
def distance_AB : ℝ := 4250
def bus_rate : ℝ := 0.10
def plane_rate : ℝ := 0.15
def boarding_fee : ℝ := 150

theorem total_trip_cost :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2)
  let flight_cost := distance_AB * plane_rate + boarding_fee
  let bus_cost := distance_BC * bus_rate
  flight_cost + bus_cost = 931.15 :=
by
  sorry

end total_trip_cost_l233_233134


namespace total_cats_correct_l233_233454

-- Jamie's cats
def Jamie_Persian_cats : ℕ := 4
def Jamie_Maine_Coons : ℕ := 2

-- Gordon's cats
def Gordon_Persian_cats : ℕ := Jamie_Persian_cats / 2
def Gordon_Maine_Coons : ℕ := Jamie_Maine_Coons + 1

-- Hawkeye's cats
def Hawkeye_Persian_cats : ℕ := 0
def Hawkeye_Maine_Coons : ℕ := Gordon_Maine_Coons - 1

-- Total cats for each person
def Jamie_total_cats : ℕ := Jamie_Persian_cats + Jamie_Maine_Coons
def Gordon_total_cats : ℕ := Gordon_Persian_cats + Gordon_Maine_Coons
def Hawkeye_total_cats : ℕ := Hawkeye_Persian_cats + Hawkeye_Maine_Coons

-- Proof that the total number of cats is 13
theorem total_cats_correct : Jamie_total_cats + Gordon_total_cats + Hawkeye_total_cats = 13 :=
by sorry

end total_cats_correct_l233_233454


namespace max_m_value_l233_233839

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 1 / b = 1 / 4) : ∃ m : ℝ, (∀ a b : ℝ,  a > 0 ∧ b > 0 ∧ (2 / a + 1 / b = 1 / 4) → 2 * a + b ≥ 4 * m) ∧ m = 7 / 4 :=
sorry

end max_m_value_l233_233839


namespace inequality_holds_infinitely_many_times_l233_233584

variable {a : ℕ → ℝ}

theorem inequality_holds_infinitely_many_times
    (h_pos : ∀ n, 0 < a n) :
    ∃ᶠ n in at_top, 1 + a n > a (n - 1) * 2^(1 / n) :=
sorry

end inequality_holds_infinitely_many_times_l233_233584


namespace real_y_values_for_given_x_l233_233674

theorem real_y_values_for_given_x (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ (x ≤ -2 / 3 ∨ x ≥ 4) :=
by
  sorry

end real_y_values_for_given_x_l233_233674


namespace tutors_meet_in_lab_l233_233443

theorem tutors_meet_in_lab (c a j t : ℕ)
  (hC : c = 5) (hA : a = 6) (hJ : j = 8) (hT : t = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm c a) j) t = 360 :=
by
  rw [hC, hA, hJ, hT]
  rfl

end tutors_meet_in_lab_l233_233443


namespace complement_union_eq_singleton_five_l233_233872

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233872


namespace expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l233_233064

-- Proof for (x + 3y)^2 = x^2 + 6xy + 9y^2
theorem expand_x_plus_3y_squared (x y : ℝ) : 
  (x + 3 * y) ^ 2 = x ^ 2 + 6 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (2x + 3y)^2 = 4x^2 + 12xy + 9y^2
theorem expand_2x_plus_3y_squared (x y : ℝ) : 
  (2 * x + 3 * y) ^ 2 = 4 * x ^ 2 + 12 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (m^3 + n^5)^2 = m^6 + 2m^3n^5 + n^10
theorem expand_m3_plus_n5_squared (m n : ℝ) : 
  (m ^ 3 + n ^ 5) ^ 2 = m ^ 6 + 2 * m ^ 3 * n ^ 5 + n ^ 10 := 
  sorry

-- Proof for (5x - 3y)^2 = 25x^2 - 30xy + 9y^2
theorem expand_5x_minus_3y_squared (x y : ℝ) : 
  (5 * x - 3 * y) ^ 2 = 25 * x ^ 2 - 30 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (3m^5 - 4n^2)^2 = 9m^10 - 24m^5n^2 + 16n^4
theorem expand_3m5_minus_4n2_squared (m n : ℝ) : 
  (3 * m ^ 5 - 4 * n ^ 2) ^ 2 = 9 * m ^ 10 - 24 * m ^ 5 * n ^ 2 + 16 * n ^ 4 := 
  sorry

end expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l233_233064


namespace gcd_7429_12345_l233_233681

theorem gcd_7429_12345 : Int.gcd 7429 12345 = 1 := 
by 
  sorry

end gcd_7429_12345_l233_233681


namespace distance_scientific_notation_l233_233609

theorem distance_scientific_notation :
  55000000 = 5.5 * 10^7 :=
sorry

end distance_scientific_notation_l233_233609


namespace solve_polynomial_l233_233542

theorem solve_polynomial (z : ℂ) : z^6 - 9 * z^3 + 8 = 0 ↔ z = 1 ∨ z = 2 := 
by
  sorry

end solve_polynomial_l233_233542


namespace arithmetic_sequence_sum_first_three_terms_l233_233611

theorem arithmetic_sequence_sum_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 4) (h5 : a 5 = 7) (h6 : a 6 = 10) : a 1 + a 2 + a 3 = -6 :=
sorry

end arithmetic_sequence_sum_first_three_terms_l233_233611


namespace probability_of_success_l233_233124

open Finset

noncomputable def numbers : Finset ℕ := {5, 14, 28, 35, 49, 54, 63}

def product_is_multiple_of_126 (a b : ℕ) : Prop :=
  ∃ (x y z : ℕ), 2^x * 3^(2*y) * 7^z ∣ a * b ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1

def successful_pairings : Finset (ℕ × ℕ) :=
  numbers.product numbers.filter (λ x, x ≠ x.1)

def count_successful : ℚ :=
  (successful_pairings.filter (λ p, product_is_multiple_of_126 p.1 p.2)).card

def total_pairings : ℚ := (numbers.card.choose 2 : ℚ)

theorem probability_of_success : (count_successful / total_pairings) = 4 / 21 :=
by
  sorry

end probability_of_success_l233_233124


namespace lakshmi_share_annual_gain_l233_233599

theorem lakshmi_share_annual_gain (x : ℝ) (annual_gain : ℝ) (Raman_inv_months : ℝ) (Lakshmi_inv_months : ℝ) (Muthu_inv_months : ℝ) (Gowtham_inv_months : ℝ) (Pradeep_inv_months : ℝ)
  (total_inv_months : ℝ) (lakshmi_share : ℝ) :
  Raman_inv_months = x * 12 →
  Lakshmi_inv_months = 2 * x * 6 →
  Muthu_inv_months = 3 * x * 4 →
  Gowtham_inv_months = 4 * x * 9 →
  Pradeep_inv_months = 5 * x * 1 →
  total_inv_months = Raman_inv_months + Lakshmi_inv_months + Muthu_inv_months + Gowtham_inv_months + Pradeep_inv_months →
  annual_gain = 58000 →
  lakshmi_share = (Lakshmi_inv_months / total_inv_months) * annual_gain →
  lakshmi_share = 9350.65 :=
by
  sorry

end lakshmi_share_annual_gain_l233_233599


namespace leak_emptying_time_l233_233153

theorem leak_emptying_time (A_rate L_rate : ℚ) 
  (hA : A_rate = 1 / 4)
  (hCombined : A_rate - L_rate = 1 / 8) :
  1 / L_rate = 8 := 
by
  sorry

end leak_emptying_time_l233_233153


namespace problem_l233_233289

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l233_233289


namespace spelling_bee_participants_l233_233717

theorem spelling_bee_participants (n : ℕ)
  (h1 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (k - 1 < 74 ∨ k - 1 > 74))
  (h2 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (75 - k > 0 ∨ k - 1 > 74)) :
  n = 149 := by
  sorry

end spelling_bee_participants_l233_233717


namespace twelve_times_reciprocal_sum_l233_233625

theorem twelve_times_reciprocal_sum (a b c : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/4) (h₃ : c = 1/6) :
  12 * (a + b + c)⁻¹ = 16 := 
by
  sorry

end twelve_times_reciprocal_sum_l233_233625


namespace subsets_neither_A_nor_B_l233_233249

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B_l233_233249


namespace intersection_with_y_axis_l233_233484

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l233_233484


namespace chord_bisected_line_eq_l233_233036

theorem chord_bisected_line_eq (x y : ℝ) (hx1 : x^2 + 4 * y^2 = 36) (hx2 : (4, 2) = ((x1 + x2) / 2, (y1 + y2) / 2)) :
  x + 2 * y - 8 = 0 :=
sorry

end chord_bisected_line_eq_l233_233036


namespace total_spent_on_burgers_l233_233413

def days_in_june := 30
def burgers_per_day := 4
def cost_per_burger := 13

theorem total_spent_on_burgers (total_spent : Nat) :
  total_spent = days_in_june * burgers_per_day * cost_per_burger :=
sorry

end total_spent_on_burgers_l233_233413


namespace most_suitable_for_comprehensive_survey_l233_233061

-- Definitions of the survey options
inductive SurveyOption
| A
| B
| C
| D

-- Condition definitions based on the problem statement
def comprehensive_survey (option : SurveyOption) : Prop :=
  option = SurveyOption.B

-- The theorem stating that the most suitable survey is option B
theorem most_suitable_for_comprehensive_survey : ∀ (option : SurveyOption), comprehensive_survey option ↔ option = SurveyOption.B :=
by
  intro option
  sorry

end most_suitable_for_comprehensive_survey_l233_233061


namespace complement_union_l233_233941

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233941


namespace average_of_second_set_of_two_numbers_l233_233349

theorem average_of_second_set_of_two_numbers
  (S : ℝ)
  (avg1 avg2 avg3 : ℝ)
  (h1 : S = 6 * 3.95)
  (h2 : avg1 = 3.4)
  (h3 : avg3 = 4.6) :
  (S - (2 * avg1) - (2 * avg3)) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_of_two_numbers_l233_233349


namespace matrix_diagonal_neg5_l233_233534

variable (M : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_diagonal_neg5 
    (h : ∀ v : Fin 3 → ℝ, (M.mulVec v) = -5 • v) : 
    M = !![-5, 0, 0; 0, -5, 0; 0, 0, -5] :=
by
  sorry

end matrix_diagonal_neg5_l233_233534


namespace union_complement_eq_l233_233306

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l233_233306


namespace volume_of_solid_of_revolution_l233_233392

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let h := a / 2
  let r := (Real.sqrt 3 / 2) * a
  2 * (1 / 3) * π * r^2 * h = (π * a^3) / 4 :=
by
  sorry

end volume_of_solid_of_revolution_l233_233392


namespace complement_union_l233_233881

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233881


namespace sum_of_ages_26_l233_233003

-- Define an age predicate to manage the three ages
def is_sum_of_ages (kiana twin : ℕ) : Prop :=
  kiana < twin ∧ twin * twin * kiana = 180 ∧ (kiana + twin + twin = 26)

theorem sum_of_ages_26 : 
  ∃ (kiana twin : ℕ), is_sum_of_ages kiana twin :=
by 
  sorry

end sum_of_ages_26_l233_233003


namespace max_blue_points_l233_233832

theorem max_blue_points (n : ℕ) (r b : ℕ)
  (h1 : n = 2009)
  (h2 : b + r = n)
  (h3 : ∀(k : ℕ), b ≤ k * (k - 1) / 2 → r ≥ k) :
  b = 1964 :=
by
  sorry

end max_blue_points_l233_233832


namespace cos_inequality_for_triangle_l233_233140

theorem cos_inequality_for_triangle (A B C : ℝ) (h : A + B + C = π) :
  (1 / 3) * (Real.cos A + Real.cos B + Real.cos C) ≤ (1 / 2) ∧
  (1 / 2) ≤ Real.sqrt ((1 / 3) * (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2)) :=
by
  sorry

end cos_inequality_for_triangle_l233_233140


namespace count_positive_integers_satisfy_l233_233438

theorem count_positive_integers_satisfy :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 5) * (n - 3) * (n - 12) * (n - 17) < 0) ∧ S.card = 4 :=
by
  sorry

end count_positive_integers_satisfy_l233_233438


namespace pipe_pumping_rate_l233_233623

theorem pipe_pumping_rate (R : ℕ) (h : 5 * R + 5 * 192 = 1200) : R = 48 := by
  sorry

end pipe_pumping_rate_l233_233623


namespace quadratic_with_real_roots_l233_233687

theorem quadratic_with_real_roots: 
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4 * x₁ + k = 0 ∧ x₂^2 + 4 * x₂ + k = 0) ↔ (k ≤ 4) := 
by 
  sorry

end quadratic_with_real_roots_l233_233687


namespace solve_inequality_l233_233343

theorem solve_inequality (x : ℝ) : 2 * x + 6 > 5 * x - 3 → x < 3 :=
by
  -- Proof steps would go here
  sorry

end solve_inequality_l233_233343


namespace min_number_of_students_l233_233076

theorem min_number_of_students 
  (n : ℕ)
  (h1 : 25 ≡ 99 [MOD n])
  (h2 : 8 ≡ 119 [MOD n]) : 
  n = 37 :=
by sorry

end min_number_of_students_l233_233076


namespace max_value_of_M_l233_233101

theorem max_value_of_M (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)) ≤ 1 :=
sorry -- Proof placeholder

end max_value_of_M_l233_233101


namespace complement_union_eq_singleton_five_l233_233876

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233876


namespace find_Y_payment_l233_233622

theorem find_Y_payment 
  (P X Z : ℝ)
  (total_payment : ℝ)
  (h1 : P + X + Z = total_payment)
  (h2 : X = 1.2 * P)
  (h3 : Z = 0.96 * P) :
  P = 332.28 := by
  sorry

end find_Y_payment_l233_233622


namespace rebecca_tent_stakes_l233_233160

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l233_233160


namespace cos_sum_series_l233_233554

theorem cos_sum_series : 
  (∑' n : ℤ, if (n % 2 = 1 ∨ n % 2 = -1) then (1 : ℝ) / (n : ℝ)^2 else 0) = (π^2) / 8 := by
  sorry

end cos_sum_series_l233_233554


namespace exists_line_equidistant_from_AB_CD_l233_233547

noncomputable def Line : Type := sorry  -- This would be replaced with an appropriate definition of a line in space

def Point : Type := sorry  -- Similarly, a point in space type definition

variables (A B C D : Point)

def perpendicularBisector (P Q : Point) : Type := sorry  -- Definition for perpendicular bisector plane of two points

def is_perpendicularBisector_of (e : Line) (P Q : Point) : Prop := sorry  -- e is perpendicular bisector plane of P and Q

theorem exists_line_equidistant_from_AB_CD (A B C D : Point) :
  ∃ e : Line, is_perpendicularBisector_of e A C ∧ is_perpendicularBisector_of e B D :=
by
  sorry

end exists_line_equidistant_from_AB_CD_l233_233547


namespace choir_singers_joined_final_verse_l233_233806

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end choir_singers_joined_final_verse_l233_233806


namespace sum_of_a_b_l233_233843

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l233_233843


namespace average_of_seven_consecutive_l233_233339

variable (a : ℕ) 

def average_of_consecutive_integers (x : ℕ) : ℕ :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) / 7

theorem average_of_seven_consecutive (a : ℕ) :
  average_of_consecutive_integers (average_of_consecutive_integers a) = a + 6 :=
by
  sorry

end average_of_seven_consecutive_l233_233339


namespace cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l233_233727

-- Define the production cost function
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Define the profit function
def profit (n : ℕ) : ℤ := 90 * n - 4000 - 50 * n

-- 1. Prove that the cost for producing 1000 pairs of shoes is 54,000 yuan
theorem cost_of_1000_pairs : production_cost 1000 = 54000 := 
by sorry

-- 2. Prove that if the production cost is 48,000 yuan, then 880 pairs of shoes were produced
theorem pairs_for_48000_yuan (n : ℕ) (h : production_cost n = 48000) : n = 880 := 
by sorry

-- 3. Prove that at least 100 pairs of shoes must be produced each day to avoid a loss
theorem minimum_pairs_to_avoid_loss (n : ℕ) : profit n ≥ 0 ↔ n ≥ 100 := 
by sorry

end cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l233_233727


namespace radius_of_circle_is_4_l233_233431

noncomputable def circle_radius
  (a : ℝ) 
  (radius : ℝ) 
  (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 9 = 0 ∧ (-a, 0) = (5, 0) ∧ radius = 4

theorem radius_of_circle_is_4 
  (a x y : ℝ) 
  (radius : ℝ) 
  (h : circle_radius a radius x y) : 
  radius = 4 :=
by 
  sorry

end radius_of_circle_is_4_l233_233431


namespace minimum_value_m_l233_233251

theorem minimum_value_m (x0 : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 1| ≤ m) → m = 2 :=
by
  sorry

end minimum_value_m_l233_233251


namespace union_complement_set_l233_233286

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l233_233286


namespace min_disks_required_l233_233728

def num_files : ℕ := 35
def disk_size : ℕ := 2
def file_size_0_9 : ℕ := 4
def file_size_0_8 : ℕ := 15
def file_size_0_5 : ℕ := num_files - file_size_0_9 - file_size_0_8

-- Prove the minimum number of disks required to store all files.
theorem min_disks_required 
  (n : ℕ) 
  (disk_storage : ℕ)
  (num_files_0_9 : ℕ)
  (num_files_0_8 : ℕ)
  (num_files_0_5 : ℕ) :
  n = num_files → disk_storage = disk_size → num_files_0_9 = file_size_0_9 → num_files_0_8 = file_size_0_8 → num_files_0_5 = file_size_0_5 → 
  ∃ (d : ℕ), d = 15 :=
by 
  intros H1 H2 H3 H4 H5
  sorry

end min_disks_required_l233_233728


namespace find_principal_l233_233514

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem find_principal
  (A : ℝ) (r : ℝ) (n t : ℕ)
  (hA : A = 4410)
  (hr : r = 0.05)
  (hn : n = 1)
  (ht : t = 2) :
  ∃ (P : ℝ), compound_interest P r n t = A ∧ P = 4000 :=
by
  sorry

end find_principal_l233_233514


namespace sum_of_altitudes_less_than_sum_of_sides_l233_233028

theorem sum_of_altitudes_less_than_sum_of_sides 
  (a b c h_a h_b h_c K : ℝ) 
  (triangle_area : K = (1/2) * a * h_a)
  (h_a_def : h_a = 2 * K / a) 
  (h_b_def : h_b = 2 * K / b)
  (h_c_def : h_c = 2 * K / c) : 
  h_a + h_b + h_c < a + b + c := by
  sorry

end sum_of_altitudes_less_than_sum_of_sides_l233_233028


namespace number_divided_is_144_l233_233474

theorem number_divided_is_144 (n divisor quotient remainder : ℕ) (h_divisor : divisor = 11) (h_quotient : quotient = 13) (h_remainder : remainder = 1) (h_division : n = (divisor * quotient) + remainder) : n = 144 :=
by
  sorry

end number_divided_is_144_l233_233474


namespace chair_cost_l233_233222

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l233_233222


namespace find_300th_term_excl_squares_l233_233626

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l233_233626


namespace points_symmetric_about_y_eq_x_l233_233149

theorem points_symmetric_about_y_eq_x (x y r : ℝ) :
  (x^2 + y^2 ≤ r^2 ∧ x + y > 0) →
  (∃ p q : ℝ, (q = p ∧ p + q = 0) ∨ (p = q ∧ q = -p)) :=
sorry

end points_symmetric_about_y_eq_x_l233_233149


namespace total_fruit_in_buckets_l233_233186

theorem total_fruit_in_buckets (A B C : ℕ) 
  (h1 : A = B + 4)
  (h2 : B = C + 3)
  (h3 : C = 9) :
  A + B + C = 37 := by
  sorry

end total_fruit_in_buckets_l233_233186


namespace satisfies_equation_l233_233517

noncomputable def y (x : ℝ) : ℝ := -Real.sqrt (x^4 - x^2)
noncomputable def dy (x : ℝ) : ℝ := x * (1 - 2 * x^2) / Real.sqrt (x^4 - x^2)

theorem satisfies_equation (x : ℝ) (hx : x ≠ 0) : x * y x * dy x - (y x)^2 = x^4 := 
sorry

end satisfies_equation_l233_233517


namespace range_of_values_for_k_l233_233238

theorem range_of_values_for_k (k : ℝ) (h : k ≠ 0) :
  (1 : ℝ) ∈ { x : ℝ | k^2 * x^2 - 6 * k * x + 8 ≥ 0 } ↔ (k ≥ 4 ∨ k ≤ 2) := 
by
  -- proof 
  sorry

end range_of_values_for_k_l233_233238


namespace union_complement_eq_l233_233298

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l233_233298


namespace union_complement_set_l233_233284

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l233_233284


namespace average_age_union_l233_233800

theorem average_age_union
    (A B C : Set Person)
    (a b c : ℕ)
    (sum_A sum_B sum_C : ℝ)
    (h_disjoint_AB : Disjoint A B)
    (h_disjoint_AC : Disjoint A C)
    (h_disjoint_BC : Disjoint B C)
    (h_avg_A : sum_A / a = 40)
    (h_avg_B : sum_B / b = 25)
    (h_avg_C : sum_C / c = 35)
    (h_avg_AB : (sum_A + sum_B) / (a + b) = 33)
    (h_avg_AC : (sum_A + sum_C) / (a + c) = 37.5)
    (h_avg_BC : (sum_B + sum_C) / (b + c) = 30) :
  (sum_A + sum_B + sum_C) / (a + b + c) = 51.6 :=
sorry

end average_age_union_l233_233800


namespace selection_events_mutually_exclusive_not_complementary_l233_233980

-- Definitions representing the individuals
def boys : Finset ℕ := {1, 2}
def girls : Finset ℕ := {3, 4}

-- Event definitions
def event_exactly_one_girl (selection : Finset ℕ) : Prop :=
  selection.card = 2 ∧ ∃ girl ∈ selection, ∃ boy ∈ selection, boy ∉ girls

def event_exactly_two_girls (selection : Finset ℕ) : Prop :=
  selection.card = 2 ∧ ∀ x ∈ selection, x ∈ girls

-- Sample space of selecting 2 people out of 4
def sample_space : Finset (Finset ℕ) := (Finset.powerset (boys ∪ girls)).filter (λ s, s.card = 2)

-- The theorem statement
theorem selection_events_mutually_exclusive_not_complementary :
  ∀ selection ∈ sample_space,
  (event_exactly_one_girl selection ∧ event_exactly_two_girls selection) = false :=
begin
  sorry -- Proof to be filled in
end

end selection_events_mutually_exclusive_not_complementary_l233_233980


namespace point_not_similar_inflection_point_ln_l233_233256

noncomputable def similar_inflection_point (C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ (m : ℝ → ℝ), (∀ x, m x = (deriv C P.1) * (x - P.1) + P.2) ∧
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → (C (P.1 + h) > m (P.1 + h) ∧ C (P.1 - h) < m (P.1 - h)) ∨ 
                     (C (P.1 + h) < m (P.1 + h) ∧ C (P.1 - h) > m (P.1 - h))

theorem point_not_similar_inflection_point_ln :
  ¬ similar_inflection_point (fun x => Real.log x) (1, 0) :=
sorry

end point_not_similar_inflection_point_ln_l233_233256


namespace complement_union_eq_l233_233923

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233923


namespace roots_order_l233_233118

theorem roots_order {a b m n : ℝ} (h1 : m < n) (h2 : a < b)
  (hm : 1 - (m - a) * (m - b) = 0) (hn : 1 - (n - a) * (n - b) = 0) :
  m < a ∧ a < b ∧ b < n :=
sorry

end roots_order_l233_233118


namespace sum_of_first_cards_l233_233025

variables (a b c d : ℕ)

theorem sum_of_first_cards (a b c d : ℕ) : 
  ∃ x, x = b * (c + 1) + d - a :=
by
  sorry

end sum_of_first_cards_l233_233025


namespace complement_union_l233_233911

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233911


namespace standard_heat_of_formation_Fe2O3_l233_233051

def Q_form_Al2O3 := 1675.5 -- kJ/mol

def Q1 := 854.2 -- kJ

-- Definition of the standard heat of formation of Fe2O3
def Q_form_Fe2O3 := Q_form_Al2O3 - Q1

-- The proof goal
theorem standard_heat_of_formation_Fe2O3 : Q_form_Fe2O3 = 821.3 := by
  sorry

end standard_heat_of_formation_Fe2O3_l233_233051


namespace not_divisible_by_n_plus_4_l233_233596

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 8*n + 15 = k * (n + 4) :=
sorry

end not_divisible_by_n_plus_4_l233_233596


namespace complement_union_l233_233884

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233884


namespace shaded_area_is_20_l233_233539

-- Represents the square PQRS with the necessary labeled side lengths
noncomputable def square_side_length : ℝ := 8

-- Represents the four labeled smaller squares' positions and their side lengths
noncomputable def smaller_square_side_lengths : List ℝ := [2, 2, 2, 6]

-- The coordinates or relations to describe their overlaying positions are not needed for the proof.

-- Define the calculated areas from the solution steps
noncomputable def vertical_rectangle_area : ℝ := 6 * 2
noncomputable def horizontal_rectangle_area : ℝ := 6 * 2
noncomputable def overlap_area : ℝ := 2 * 2

-- The total shaded T-shaped region area calculation
noncomputable def total_shaded_area : ℝ := vertical_rectangle_area + horizontal_rectangle_area - overlap_area

-- Theorem statement to prove the area of the T-shaped region is 20
theorem shaded_area_is_20 : total_shaded_area = 20 :=
by
  -- Proof steps are not required as per the instruction.
  sorry

end shaded_area_is_20_l233_233539


namespace complement_union_of_M_and_N_l233_233886

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233886


namespace value_of_M_l233_233962

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l233_233962


namespace restaurant_production_in_june_l233_233078

def cheese_pizzas_per_day (hot_dogs_per_day : ℕ) : ℕ :=
  hot_dogs_per_day + 40

def pepperoni_pizzas_per_day (cheese_pizzas_per_day : ℕ) : ℕ :=
  2 * cheese_pizzas_per_day

def hot_dogs_per_day := 60
def beef_hot_dogs_per_day := 30
def chicken_hot_dogs_per_day := 30
def days_in_june := 30

theorem restaurant_production_in_june :
  (cheese_pizzas_per_day hot_dogs_per_day * days_in_june = 3000) ∧
  (pepperoni_pizzas_per_day (cheese_pizzas_per_day hot_dogs_per_day) * days_in_june = 6000) ∧
  (beef_hot_dogs_per_day * days_in_june = 900) ∧
  (chicken_hot_dogs_per_day * days_in_june = 900) :=
by
  sorry

end restaurant_production_in_june_l233_233078


namespace odd_function_value_l233_233236

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

theorem odd_function_value (a b : ℝ) (h1 : ∀ x, f x b = -f (-x) b) (h2 : a - 4 + 2 * a - 2 = 0) : f a b + f (2 * -a) b = 0 := by
  sorry

end odd_function_value_l233_233236


namespace jar_a_marbles_l233_233455

theorem jar_a_marbles : ∃ A : ℕ, (∃ B : ℕ, B = A + 12) ∧ (∃ C : ℕ, C = 2 * (A + 12)) ∧ (A + (A + 12) + 2 * (A + 12) = 148) ∧ (A = 28) :=
by
sorry

end jar_a_marbles_l233_233455


namespace min_number_of_stamps_exists_l233_233397

theorem min_number_of_stamps_exists : 
  ∃ s t : ℕ, 5 * s + 7 * t = 50 ∧ ∀ (s' t' : ℕ), 5 * s' + 7 * t' = 50 → s + t ≤ s' + t' := 
by
  sorry

end min_number_of_stamps_exists_l233_233397


namespace line_tangent_to_ellipse_l233_233978

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 → ∃! y : ℝ, 3 * x^2 + 6 * y^2 = 6) →
  m^2 = 3 / 2 :=
by
  sorry

end line_tangent_to_ellipse_l233_233978


namespace chess_tournament_games_l233_233642

theorem chess_tournament_games (n : ℕ) (h : n = 17) (k : n - 1 = 16) :
  (n * (n - 1)) / 2 = 136 := by
  sorry

end chess_tournament_games_l233_233642


namespace complement_union_eq_singleton_five_l233_233874

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233874


namespace wrapping_third_roll_l233_233756

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l233_233756


namespace charlotte_total_dog_walking_time_l233_233818

def poodles_monday : ℕ := 4
def chihuahuas_monday : ℕ := 2
def poodles_tuesday : ℕ := 4
def chihuahuas_tuesday : ℕ := 2
def labradors_wednesday : ℕ := 4

def time_poodle : ℕ := 2
def time_chihuahua : ℕ := 1
def time_labrador : ℕ := 3

def total_time_monday : ℕ := poodles_monday * time_poodle + chihuahuas_monday * time_chihuahua
def total_time_tuesday : ℕ := poodles_tuesday * time_poodle + chihuahuas_tuesday * time_chihuahua
def total_time_wednesday : ℕ := labradors_wednesday * time_labrador

def total_time_week : ℕ := total_time_monday + total_time_tuesday + total_time_wednesday

theorem charlotte_total_dog_walking_time : total_time_week = 32 := by
  -- Lean allows us to state the theorem without proving it.
  sorry

end charlotte_total_dog_walking_time_l233_233818


namespace total_students_l233_233080

theorem total_students (rank_right rank_left : ℕ) (h_right : rank_right = 18) (h_left : rank_left = 12) : rank_right + rank_left - 1 = 29 := 
by
  sorry

end total_students_l233_233080


namespace smallest_five_digit_perfect_square_and_cube_l233_233365

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ n = 15625 :=
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l233_233365


namespace union_complement_eq_l233_233296

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l233_233296


namespace lemonade_calories_l233_233544

theorem lemonade_calories 
    (lime_juice_weight : ℕ)
    (lime_juice_calories_per_grams : ℕ)
    (sugar_weight : ℕ)
    (sugar_calories_per_grams : ℕ)
    (water_weight : ℕ)
    (water_calories_per_grams : ℕ)
    (mint_weight : ℕ)
    (mint_calories_per_grams : ℕ)
    :
    lime_juice_weight = 150 →
    lime_juice_calories_per_grams = 30 →
    sugar_weight = 200 →
    sugar_calories_per_grams = 390 →
    water_weight = 500 →
    water_calories_per_grams = 0 →
    mint_weight = 50 →
    mint_calories_per_grams = 7 →
    (300 * ((150 * 30 + 200 * 390 + 500 * 0 + 50 * 7) / 900) = 276) :=
by
  sorry

end lemonade_calories_l233_233544


namespace other_endpoint_coordinates_sum_l233_233023

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l233_233023


namespace oliver_earnings_l233_233009

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l233_233009


namespace find_k_such_that_product_minus_one_is_perfect_power_l233_233098

noncomputable def product_of_first_n_primes (n : ℕ) : ℕ :=
  (List.take n (List.filter (Nat.Prime) (List.range n.succ))).prod

theorem find_k_such_that_product_minus_one_is_perfect_power :
  ∀ k : ℕ, ∃ a n : ℕ, (product_of_first_n_primes k) - 1 = a^n ∧ n > 1 ∧ k = 1 :=
by
  sorry

end find_k_such_that_product_minus_one_is_perfect_power_l233_233098


namespace unique_solution_m_n_l233_233837

theorem unique_solution_m_n (m n : ℕ) (h1 : m > 1) (h2 : (n - 1) % (m - 1) = 0) 
  (h3 : ¬ ∃ k : ℕ, n = m ^ k) :
  ∃! (a b c : ℕ), a + m * b = n ∧ a + b = m * c := 
sorry

end unique_solution_m_n_l233_233837


namespace identify_incorrect_calculation_l233_233792

theorem identify_incorrect_calculation : 
  (∀ x : ℝ, x^2 * x^3 = x^5) ∧ 
  (∀ x : ℝ, x^3 + x^3 = 2 * x^3) ∧ 
  (∀ x : ℝ, x^6 / x^2 = x^4) ∧ 
  ¬ (∀ x : ℝ, (-3 * x)^2 = 6 * x^2) := 
by
  sorry

end identify_incorrect_calculation_l233_233792


namespace choir_third_verse_joiners_l233_233805

theorem choir_third_verse_joiners:
  let total_singers := 30 in
  let first_verse_singers := total_singers / 2 in
  let remaining_after_first := total_singers - first_verse_singers in
  let second_verse_singers := remaining_after_first / 3 in
  let remaining_after_second := remaining_after_first - second_verse_singers in
  let third_verse_singers := remaining_after_second in
  third_verse_singers = 10 := 
by
  sorry

end choir_third_verse_joiners_l233_233805


namespace num_possible_pairs_l233_233384

theorem num_possible_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 8) * (b - 8) = 32) : 
    (∃ n, n = 3) :=
by { sorry }

end num_possible_pairs_l233_233384


namespace find_two_digit_number_l233_233211

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b

theorem find_two_digit_number (a b : ℕ) (h1 : a = 2 * b) (h2 : original_number b a = original_number a b - 36) : original_number a b = 84 :=
by
  sorry

end find_two_digit_number_l233_233211


namespace least_number_subtracted_divisible_17_l233_233410

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17_l233_233410


namespace find_LN_l233_233171

noncomputable def LM : ℝ := 25
noncomputable def sinN : ℝ := 4 / 5

theorem find_LN (LN : ℝ) (h_sin : sinN = LM / LN) : LN = 125 / 4 :=
by
  sorry

end find_LN_l233_233171


namespace min_max_product_l233_233734

noncomputable def min_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the minimum value of 3x^2 + 4xy + 3y^2
  sorry

noncomputable def max_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the maximum value of 3x^2 + 4xy + 3y^2
  sorry

theorem min_max_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) :
  min_value x y h * max_value x y h = 7 / 16 :=
sorry

end min_max_product_l233_233734


namespace shaded_area_in_6x6_grid_l233_233802

def total_shaded_area (grid_size : ℕ) (triangle_squares : ℕ) (num_triangles : ℕ)
  (trapezoid_squares : ℕ) (num_trapezoids : ℕ) : ℕ :=
  (triangle_squares * num_triangles) + (trapezoid_squares * num_trapezoids)

theorem shaded_area_in_6x6_grid :
  total_shaded_area 6 2 2 3 4 = 16 :=
by
  -- Proof omitted for demonstration purposes
  sorry

end shaded_area_in_6x6_grid_l233_233802


namespace sum_of_other_endpoint_l233_233017

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l233_233017


namespace number_of_real_solutions_l233_233825

theorem number_of_real_solutions (x : ℝ) (n : ℤ) : 
  (3 : ℝ) * x^2 - 27 * (n : ℝ) + 29 = 0 → n = ⌊x⌋ →  ∃! x, (3 : ℝ) * x^2 - 27 * (⌊x⌋ : ℝ) + 29 = 0 := 
sorry

end number_of_real_solutions_l233_233825


namespace sum_same_probability_l233_233176

-- Definition for standard dice probability problem
def dice_problem (n : ℕ) (target_sum : ℕ) (target_sum_of_faces : ℕ) : Prop :=
  let faces := [1, 2, 3, 4, 5, 6]
  let min_sum := n * 1
  let max_sum := n * 6
  let average_sum := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * average_sum - target_sum
  symmetric_sum = target_sum_of_faces

-- The proof statement (no proof included, just the declaration)
theorem sum_same_probability : dice_problem 8 12 44 :=
by sorry

end sum_same_probability_l233_233176


namespace largest_odd_not_sum_of_three_distinct_composites_l233_233682

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem largest_odd_not_sum_of_three_distinct_composites :
  ∀ n : ℕ, is_odd n → (¬ ∃ (a b c : ℕ), is_composite a ∧ is_composite b ∧ is_composite c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a + b + c) → n ≤ 17 :=
by
  sorry

end largest_odd_not_sum_of_three_distinct_composites_l233_233682


namespace player_A_wins_even_n_l233_233147

theorem player_A_wins_even_n (n : ℕ) (hn : n > 0) (even_n : Even n) :
  ∃ strategy_A : ℕ → Bool, 
    ∀ (P Q : ℕ), P % 2 = 0 → (Q + P) % 2 = 0 :=
by 
  sorry

end player_A_wins_even_n_l233_233147


namespace sun_salutations_per_year_l233_233033

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l233_233033


namespace student_loses_one_mark_per_wrong_answer_l233_233573

noncomputable def marks_lost_per_wrong_answer (x : ℝ) : Prop :=
  let total_questions := 60
  let correct_answers := 42
  let wrong_answers := total_questions - correct_answers
  let marks_per_correct := 4
  let total_marks := 150
  correct_answers * marks_per_correct - wrong_answers * x = total_marks

theorem student_loses_one_mark_per_wrong_answer : marks_lost_per_wrong_answer 1 :=
by
  sorry

end student_loses_one_mark_per_wrong_answer_l233_233573


namespace range_of_t_l233_233419

theorem range_of_t (a t : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  a * x^2 + t * y^2 ≥ (a * x + t * y)^2 ↔ 0 ≤ t ∧ t ≤ 1 - a :=
sorry

end range_of_t_l233_233419


namespace intersection_M_N_l233_233278

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l233_233278


namespace three_consecutive_odd_numbers_l233_233784

theorem three_consecutive_odd_numbers (x : ℤ) (h : x - 2 + x + x + 2 = 27) : 
  (x + 2, x, x - 2) = (11, 9, 7) :=
by
  sorry

end three_consecutive_odd_numbers_l233_233784


namespace oliver_earnings_l233_233008

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l233_233008


namespace smallest_three_digit_number_l233_233254

theorem smallest_three_digit_number (x : ℤ) (h1 : x - 7 % 7 = 0) (h2 : x - 8 % 8 = 0) (h3 : x - 9 % 9 = 0) : x = 504 := 
sorry

end smallest_three_digit_number_l233_233254


namespace stones_in_10th_pattern_l233_233812

def stones_in_nth_pattern (n : ℕ) : ℕ :=
n * (3 * n - 1) / 2 + 1

theorem stones_in_10th_pattern : stones_in_nth_pattern 10 = 145 :=
by
  sorry

end stones_in_10th_pattern_l233_233812


namespace angus_tokens_count_l233_233394

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l233_233394


namespace point_in_second_quadrant_l233_233548

structure Point where
  x : Int
  y : Int

-- Define point P
def P : Point := { x := -1, y := 2 }

-- Define the second quadrant condition
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- The mathematical statement to prove
theorem point_in_second_quadrant : second_quadrant P := by
  sorry

end point_in_second_quadrant_l233_233548


namespace find_a_value_l233_233566

namespace Proof

-- Define the context and variables
variables (a b c : ℝ)
variables (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variables (h2 : a * 15 * 2 = 4)

-- State the theorem we want to prove
theorem find_a_value: a = 6 :=
by
  sorry

end Proof

end find_a_value_l233_233566


namespace union_complement_eq_target_l233_233309

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l233_233309


namespace count_pairs_sets_condition_l233_233275

open Finset

/-! 
  We want to find the number of pairs of non-empty subsets A and B of {1, 2, ..., 10} such that 
  the smallest element of A is not less than the largest element of B. 
  We need to prove that the total number of such pairs is 9217.
-/

theorem count_pairs_sets_condition :
  let S := (range 10).map (λ n, n + 1) in
  let valid_pairs := 
    (S.powerset.filter (λ A, A ≠ ∅)).bind (λ A,
      (S.powerset.filter (λ B, B ≠ ∅ ∧ B.max ≤ A.min)).image (λ B, (A, B))) in
  valid_pairs.card = 9217 := sorry

end count_pairs_sets_condition_l233_233275


namespace regular_polygon_sides_l233_233831

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l233_233831


namespace sun_salutations_per_year_l233_233034

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l233_233034


namespace compute_geometric_sum_l233_233090

open Complex

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem compute_geometric_sum : 
  let ω := omega in
  (ω ^ 1 + ω ^ 2 + ω ^ 3 + ω ^ 4 + ω ^ 5 + ω ^ 6 + 
  ω ^ 7 + ω ^ 8 + ω ^ 9 + ω ^ 10 + ω ^ 11 + ω ^ 12 + 
  ω ^ 13 + ω ^ 14 + ω ^ 15 + ω ^ 16) = -1 :=
by 
  let ω := omega
  have h : ω ^ 17 = 1 := 
    by sorry
  have h1 : ω ^ 16 = 1 / ω := 
    by sorry
  sorry

end compute_geometric_sum_l233_233090


namespace consequent_in_ratio_4_6_l233_233132

theorem consequent_in_ratio_4_6 (h : 4 = 6 * (20 / x)) : x = 30 := 
by
  have h' : 4 * x = 6 * 20 := sorry -- cross-multiplication
  have h'' : x = 120 / 4 := sorry -- solving for x
  have hx : x = 30 := sorry -- simplifying 120 / 4

  exact hx

end consequent_in_ratio_4_6_l233_233132


namespace quincy_more_stuffed_animals_l233_233450

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l233_233450


namespace complement_union_eq_l233_233924

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233924


namespace expand_expression_l233_233680

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := 
by
  sorry

end expand_expression_l233_233680


namespace sample_capacity_l233_233524

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ)
  (h1 : frequency = 30)
  (h2 : frequency_rate = 25 / 100) :
  n = 120 :=
by
  sorry

end sample_capacity_l233_233524


namespace angle_problem_l233_233663

-- Definitions for degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Adding two angles
def add_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := a1.minutes + a2.minutes
  let extra_degrees := total_minutes / 60
  { degrees := a1.degrees + a2.degrees + extra_degrees,
    minutes := total_minutes % 60 }

-- Subtracting two angles
def sub_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := if a1.minutes < a2.minutes then a1.minutes + 60 else a1.minutes
  let extra_deg := if a1.minutes < a2.minutes then 1 else 0
  { degrees := a1.degrees - a2.degrees - extra_deg,
    minutes := total_minutes - a2.minutes }

-- Multiplying an angle by a constant
def mul_angle (a : Angle) (k : ℕ) : Angle :=
  let total_minutes := a.minutes * k
  let extra_degrees := total_minutes / 60
  { degrees := a.degrees * k + extra_degrees,
    minutes := total_minutes % 60 }

-- Given angles
def angle1 : Angle := { degrees := 24, minutes := 31}
def angle2 : Angle := { degrees := 62, minutes := 10}

-- Prove the problem statement
theorem angle_problem : sub_angles (mul_angle angle1 4) angle2 = { degrees := 35, minutes := 54} :=
  sorry

end angle_problem_l233_233663


namespace c_minus_3_eq_neg3_l233_233220

variable (g : ℝ → ℝ)
variable (c : ℝ)

-- defining conditions
axiom invertible_g : Function.Injective g
axiom g_c_eq_3 : g c = 3
axiom g_3_eq_5 : g 3 = 5

-- The goal is to prove that c - 3 = -3
theorem c_minus_3_eq_neg3 : c - 3 = -3 :=
by
  sorry

end c_minus_3_eq_neg3_l233_233220


namespace find_p_q_l233_233743

theorem find_p_q (p q : ℚ)
  (h1 : (4 : ℚ) * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2):
  (p, q) = (-29/12 : ℚ, 43/12 : ℚ) :=
by 
  sorry

end find_p_q_l233_233743


namespace Quincy_sold_more_l233_233453

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l233_233453


namespace probability_within_circle_eq_pi_over_nine_l233_233649

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let circle_area := Real.pi * (2 ^ 2)
  let square_area := 6 * 6
  circle_area / square_area

theorem probability_within_circle_eq_pi_over_nine :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_circle_eq_pi_over_nine_l233_233649


namespace rebecca_tent_stakes_l233_233158

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l233_233158


namespace oliver_earning_correct_l233_233011

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l233_233011


namespace sum_of_17th_roots_of_unity_except_1_l233_233089

theorem sum_of_17th_roots_of_unity_except_1 :
  Complex.exp (2 * Real.pi * Complex.I / 17) +
  Complex.exp (4 * Real.pi * Complex.I / 17) +
  Complex.exp (6 * Real.pi * Complex.I / 17) +
  Complex.exp (8 * Real.pi * Complex.I / 17) +
  Complex.exp (10 * Real.pi * Complex.I / 17) +
  Complex.exp (12 * Real.pi * Complex.I / 17) +
  Complex.exp (14 * Real.pi * Complex.I / 17) +
  Complex.exp (16 * Real.pi * Complex.I / 17) +
  Complex.exp (18 * Real.pi * Complex.I / 17) +
  Complex.exp (20 * Real.pi * Complex.I / 17) +
  Complex.exp (22 * Real.pi * Complex.I / 17) +
  Complex.exp (24 * Real.pi * Complex.I / 17) +
  Complex.exp (26 * Real.pi * Complex.I / 17) +
  Complex.exp (28 * Real.pi * Complex.I / 17) +
  Complex.exp (30 * Real.pi * Complex.I / 17) +
  Complex.exp (32 * Real.pi * Complex.I / 17) = 0 := sorry

end sum_of_17th_roots_of_unity_except_1_l233_233089


namespace contrapositive_example_l233_233352

theorem contrapositive_example 
  (x y : ℝ) (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_example_l233_233352


namespace double_acute_angle_lt_180_l233_233237

theorem double_acute_angle_lt_180
  (α : ℝ) (h : 0 < α ∧ α < 90) : 2 * α < 180 := 
sorry

end double_acute_angle_lt_180_l233_233237


namespace largest_divisor_of_expression_l233_233708

theorem largest_divisor_of_expression (x : ℤ) (h_odd : x % 2 = 1) : 
  1200 ∣ ((10 * x - 4) * (10 * x) * (5 * x + 15)) := 
  sorry

end largest_divisor_of_expression_l233_233708


namespace union_complement_eq_l233_233280

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l233_233280


namespace complement_union_of_M_and_N_l233_233891

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233891


namespace complement_union_l233_233934

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l233_233934


namespace inequality_system_two_integer_solutions_l233_233710

theorem inequality_system_two_integer_solutions (m : ℝ) : (-1 : ℝ) ≤ m ∧ m < 0 ↔ ∃ x : ℤ, (x < 1) ∧ (x > m - 1) ∧ {
  (∃ y : ℤ, (y < 1) ∧ (y > m - 1) ∧ x ≠ y)
  ∧ ∀ z : ℤ, (z < 1) ∧ (z > m - 1) → (z = x ∨ z = y)
}

end inequality_system_two_integer_solutions_l233_233710


namespace min_star_value_l233_233515

theorem min_star_value :
  ∃ (star : ℕ), (98348 * 10 + star) % 72 = 0 ∧ (∀ (x : ℕ), (98348 * 10 + x) % 72 = 0 → star ≤ x) := sorry

end min_star_value_l233_233515


namespace point_in_fourth_quadrant_l233_233698

open Complex

theorem point_in_fourth_quadrant (z : ℂ) (h : (3 + 4 * I) * z = 25) : 
  Complex.arg z > -π / 2 ∧ Complex.arg z < 0 := 
by
  sorry

end point_in_fourth_quadrant_l233_233698


namespace value_of_M_l233_233965

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l233_233965


namespace smallest_possible_integer_l233_233415

theorem smallest_possible_integer (a b : ℤ)
  (a_lt_10 : a < 10)
  (b_lt_10 : b < 10)
  (a_lt_b : a < b)
  (sum_eq_45 : a + b + 32 = 45)
  : a = 4 :=
by
  sorry

end smallest_possible_integer_l233_233415


namespace elisa_math_books_l233_233403

theorem elisa_math_books (N M L : ℕ) (h₀ : 24 + M + L + 1 = N + 1) (h₁ : (N + 1) % 9 = 0) (h₂ : (N + 1) % 4 = 0) (h₃ : N < 100) : M = 7 :=
by
  sorry

end elisa_math_books_l233_233403


namespace plane_speeds_l233_233187

-- Define the speeds of the planes
def speed_slower (x : ℕ) := x
def speed_faster (x : ℕ) := 2 * x

-- Define the distances each plane travels in 3 hours
def distance_slower (x : ℕ) := 3 * speed_slower x
def distance_faster (x : ℕ) := 3 * speed_faster x

-- Define the total distance
def total_distance (x : ℕ) := distance_slower x + distance_faster x

-- Prove the speeds given the total distance
theorem plane_speeds (x : ℕ) (h : total_distance x = 2700) : speed_slower x = 300 ∧ speed_faster x = 600 :=
by {
  sorry
}

end plane_speeds_l233_233187


namespace min_value_of_squares_attains_min_value_l233_233730

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  (a^2 + b^2 + c^2) ≥ (t^2 / 3) :=
sorry

theorem attains_min_value (a b c t : ℝ) (h : a = t / 3 ∧ b = t / 3 ∧ c = t / 3) :
  (a^2 + b^2 + c^2) = (t^2 / 3) :=
sorry

end min_value_of_squares_attains_min_value_l233_233730


namespace balance_blue_balls_l233_233014

noncomputable def weight_balance (G B Y W : ℝ) : ℝ :=
  3 * G + 3 * Y + 5 * W

theorem balance_blue_balls (G B Y W : ℝ)
  (hG : G = 2 * B)
  (hY : Y = 2 * B)
  (hW : W = (5 / 3) * B) :
  weight_balance G B Y W = (61 / 3) * B :=
by
  sorry

end balance_blue_balls_l233_233014


namespace ratio_m_n_l233_233562

theorem ratio_m_n (m n : ℕ) (h : (n : ℚ) / m = 3 / 7) : (m + n : ℚ) / m = 10 / 7 := by 
  sorry

end ratio_m_n_l233_233562


namespace num_of_solutions_eq_28_l233_233180

def num_solutions : Nat :=
  sorry

theorem num_of_solutions_eq_28 : num_solutions = 28 :=
  sorry

end num_of_solutions_eq_28_l233_233180


namespace sun_salutations_per_year_l233_233031

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l233_233031


namespace solution_l233_233741

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l233_233741


namespace stamens_in_bouquet_l233_233516

-- Define the number of pistils, leaves, stamens for black roses and crimson flowers
def pistils_black_rose : ℕ := 4
def stamens_black_rose : ℕ := 4
def leaves_black_rose : ℕ := 2

def pistils_crimson_flower : ℕ := 8
def stamens_crimson_flower : ℕ := 10
def leaves_crimson_flower : ℕ := 3

-- Define the number of black roses and crimson flowers (as variables x and y)
variables (x y : ℕ)

-- Define the total number of pistils and leaves in the bouquet
def total_pistils : ℕ := pistils_black_rose * x + pistils_crimson_flower * y
def total_leaves : ℕ := leaves_black_rose * x + leaves_crimson_flower * y

-- Condition: There are 108 fewer leaves than pistils
axiom leaves_pistils_relation : total_leaves = total_pistils - 108

-- Calculate the total number of stamens in the bouquet
def total_stamens : ℕ := stamens_black_rose * x + stamens_crimson_flower * y

-- The theorem to be proved
theorem stamens_in_bouquet : total_stamens = 216 :=
by
  sorry

end stamens_in_bouquet_l233_233516


namespace complement_of_union_is_singleton_five_l233_233920

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l233_233920


namespace solve_inequality_l233_233762

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end solve_inequality_l233_233762


namespace algebraic_expression_value_l233_233261

theorem algebraic_expression_value {m n : ℝ} 
  (h1 : n = m - 2022) 
  (h2 : m * n = -2022) : 
  (2022 / m) + ((m^2 - 2022 * m) / n) = 2022 := 
by sorry

end algebraic_expression_value_l233_233261


namespace tangent_line_eqn_c_range_l233_233555

noncomputable def f (x : ℝ) := 3 * x * Real.log x + 2

theorem tangent_line_eqn :
  let k := 3 
  let x₀ := 1 
  let y₀ := f x₀ 
  y = k * (x - x₀) + y₀ ↔ 3*x - y - 1 = 0 :=
by sorry

theorem c_range (x : ℝ) (hx : 1 < x) (c : ℝ) :
  f x ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 :=
by sorry

end tangent_line_eqn_c_range_l233_233555


namespace solve_for_x_l233_233342

theorem solve_for_x (x : ℝ) (h : (x^2 - 36) / 3 = (x^2 + 3 * x + 9) / 6) : x = 9 ∨ x = -9 := 
by 
  sorry

end solve_for_x_l233_233342


namespace intersection_with_y_axis_l233_233487

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l233_233487


namespace part1_part2_l233_233771

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x y : ℝ, f (x - y) = f x / f y
axiom h2 : ∀ x : ℝ, f x > 0
axiom h3 : ∀ x y : ℝ, x < y → f x > f y

-- First part: f(0) = 1 and proving f(x + y) = f(x) * f(y)
theorem part1 : f 0 = 1 ∧ (∀ x y : ℝ, f (x + y) = f x * f y) :=
sorry

-- Second part: Given f(-1) = 3, solve the inequality
axiom h4 : f (-1) = 3

theorem part2 : {x : ℝ | (x ≤ 3) ∨ (x ≥ 4)} = {x : ℝ | f (x^2 - 7*x + 10) ≤ f (-2)} :=
sorry

end part1_part2_l233_233771


namespace simplify_expression_l233_233530

theorem simplify_expression (a : ℝ) (h : a ≠ -1) : a - 1 + 1 / (a + 1) = a^2 / (a + 1) :=
  sorry

end simplify_expression_l233_233530


namespace product_g_roots_l233_233735

noncomputable def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 3

theorem product_g_roots (x_1 x_2 x_3 x_4 : ℝ) (hx : ∀ x, (x = x_1 ∨ x = x_2 ∨ x = x_3 ∨ x = x_4) ↔ f x = 0) :
  g x_1 * g x_2 * g x_3 * g x_4 = 142 :=
by sorry

end product_g_roots_l233_233735


namespace complement_union_eq_l233_233922

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233922


namespace complement_union_M_N_l233_233955

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233955


namespace bicycle_speed_l233_233790

theorem bicycle_speed (x : ℝ) (h : (2.4 / x) - (2.4 / (4 * x)) = 0.5) : 4 * x = 14.4 :=
by
  sorry

end bicycle_speed_l233_233790


namespace problem_to_prove_l233_233551

theorem problem_to_prove
  (α : ℝ)
  (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = -7 / 9 :=
by
  sorry -- proof required

end problem_to_prove_l233_233551


namespace train_speed_l233_233082

def train_length : ℝ := 110
def bridge_length : ℝ := 265
def crossing_time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem train_speed (train_length bridge_length crossing_time conversion_factor : ℝ) :
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 :=
by
  sorry

end train_speed_l233_233082


namespace cannot_determine_degree_from_A_P_l233_233990

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l233_233990


namespace current_in_circuit_l233_233720

open Complex

theorem current_in_circuit
  (V : ℂ := 2 + 3 * I)
  (Z : ℂ := 4 - 2 * I) :
  (V / Z) = (1 / 10 + 4 / 5 * I) :=
  sorry

end current_in_circuit_l233_233720


namespace proportion_solution_l233_233567

theorem proportion_solution (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) : 
  x = 17.5 * c / (4 * b) := 
sorry

end proportion_solution_l233_233567


namespace original_number_is_24_l233_233207

def number_parts (x y original_number : ℝ) : Prop :=
  7 * x + 5 * y = 146 ∧ x = 13 ∧ original_number = x + y

theorem original_number_is_24 :
  ∃ (x y original_number : ℝ), number_parts x y original_number ∧ original_number = 24 :=
by
  sorry

end original_number_is_24_l233_233207


namespace find_number_l233_233513

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 :=
by
  sorry

end find_number_l233_233513


namespace range_of_a_l233_233442

theorem range_of_a (a : ℝ) (H : ∀ x : ℝ, x ≤ 1 → 4 - a * 2^x > 0) : a < 2 :=
sorry

end range_of_a_l233_233442


namespace regular_polygon_sides_l233_233830

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l233_233830


namespace union_complement_eq_l233_233279

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l233_233279


namespace complement_union_M_N_l233_233949

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233949


namespace union_complement_eq_l233_233294

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l233_233294


namespace total_candy_count_l233_233620

def numberOfRedCandies : ℕ := 145
def numberOfBlueCandies : ℕ := 3264
def totalNumberOfCandies : ℕ := numberOfRedCandies + numberOfBlueCandies

theorem total_candy_count :
  totalNumberOfCandies = 3409 :=
by
  unfold totalNumberOfCandies
  unfold numberOfRedCandies
  unfold numberOfBlueCandies
  sorry

end total_candy_count_l233_233620


namespace car_speed_ratio_l233_233085

theorem car_speed_ratio (v_A v_B : ℕ) (h1 : v_B = 50) (h2 : 6 * v_A + 2 * v_B = 1000) :
  v_A / v_B = 3 :=
sorry

end car_speed_ratio_l233_233085


namespace metro_station_closure_l233_233721

theorem metro_station_closure (G : SimpleGraph (fin n)) [G.Connected] :
  ∃ s : fin n, G.Subgraph_fair s → G.Subgraph_fair (fin n \ s) :=
sorry

end metro_station_closure_l233_233721


namespace negation_of_universal_proposition_l233_233179

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) :=
by sorry

end negation_of_universal_proposition_l233_233179


namespace project_completion_time_l233_233811

theorem project_completion_time :
  let A_work_rate := (1 / 30) * (2 / 3)
  let B_work_rate := (1 / 60) * (3 / 4)
  let C_work_rate := (1 / 40) * (5 / 6)
  let combined_work_rate_per_12_days := 12 * (A_work_rate + B_work_rate + C_work_rate)
  let remaining_work_after_12_days := 1 - (2 / 3)
  let additional_work_rates_over_5_days := 
        5 * A_work_rate + 
        5 * B_work_rate + 
        5 * C_work_rate
  let remaining_work_after_5_days := remaining_work_after_12_days - additional_work_rates_over_5_days
  let B_additional_time := remaining_work_after_5_days / B_work_rate
  12 + 5 + B_additional_time = 17.5 :=
sorry

end project_completion_time_l233_233811


namespace shopper_saved_percentage_l233_233654

theorem shopper_saved_percentage (amount_paid : ℝ) (amount_saved : ℝ) (original_price : ℝ)
  (h1 : amount_paid = 45) (h2 : amount_saved = 5) (h3 : original_price = amount_paid + amount_saved) :
  (amount_saved / original_price) * 100 = 10 :=
by
  -- The proof is omitted
  sorry

end shopper_saved_percentage_l233_233654


namespace problem_l233_233436

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end problem_l233_233436


namespace max_average_hours_l233_233591

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l233_233591


namespace complement_union_eq_singleton_five_l233_233870

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l233_233870


namespace quadratic_equation_single_solution_l233_233192

theorem quadratic_equation_single_solution (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 = 0) ∧ (∀ x1 x2 : ℝ, a * x1^2 + a * x1 + 1 = 0 → a * x2^2 + a * x2 + 1 = 0 → x1 = x2) → a = 4 :=
by sorry

end quadratic_equation_single_solution_l233_233192


namespace stephanie_oranges_l233_233165

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l233_233165


namespace cos_pi_over_3_plus_double_alpha_l233_233967

theorem cos_pi_over_3_plus_double_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end cos_pi_over_3_plus_double_alpha_l233_233967


namespace final_grey_cats_l233_233258

def initially_total_cats : Nat := 16
def initial_white_cats : Nat := 2
def percent_black_cats : Nat := 25
def black_cats_left_fraction : Nat := 2
def new_white_cats : Nat := 2
def new_grey_cats : Nat := 1

/- We will calculate the number of grey cats after all specified events -/
theorem final_grey_cats :
  let total_cats := initially_total_cats
  let white_cats := initial_white_cats + new_white_cats
  let black_cats := (percent_black_cats * total_cats / 100) / black_cats_left_fraction
  let initial_grey_cats := total_cats - white_cats - black_cats
  let final_grey_cats := initial_grey_cats + new_grey_cats
  final_grey_cats = 11 := by
  sorry

end final_grey_cats_l233_233258


namespace seven_power_expression_l233_233274

theorem seven_power_expression (x y z : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) (h₂ : xy + xz + yz ≠ 0) :
  (x^7 + y^7 + z^7) / (xyz * (x^2 + y^2 + z^2)) = 14 :=
by
  sorry

end seven_power_expression_l233_233274


namespace distance_foci_l233_233493

noncomputable def distance_between_foci := 
  let F1 := (4, 5)
  let F2 := (-6, 9)
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) 

theorem distance_foci : 
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (4, 5) ∧ 
    F2 = (-6, 9) ∧ 
    distance_between_foci = 2 * Real.sqrt 29 := by {
  sorry
}

end distance_foci_l233_233493


namespace complement_union_l233_233895

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233895


namespace car_b_speed_l233_233665

theorem car_b_speed (v : ℕ) (h1 : ∀ (v : ℕ), CarA_speed = 3 * v)
                   (h2 : ∀ (time : ℕ), CarA_time = 6)
                   (h3 : ∀ (time : ℕ), CarB_time = 2)
                   (h4 : Car_total_distance = 1000) :
    v = 50 :=
by
  sorry

end car_b_speed_l233_233665


namespace sequence_value_l233_233138

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 1 * 3) ∧ 
  (11 - 5 = 2 * 3) ∧ 
  (20 - 11 = 3 * 3) ∧ 
  (x - 20 = 4 * 3) ∧ 
  (47 - x = 5 * 3) → 
  x = 32 :=
by 
  intros h 
  sorry

end sequence_value_l233_233138


namespace problem_statement_l233_233847

noncomputable def sequence_def (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  (a ≠ 0) ∧
  (S 1 = a) ∧
  (S 2 = 2 / S 1) ∧
  (∀ n, n ≥ 3 → S n = 2 / S (n - 1))

theorem problem_statement (a : ℝ) (S : ℕ → ℝ) (h : sequence_def a S 2018) : 
  S 2018 = 2 / a := 
by 
  sorry

end problem_statement_l233_233847


namespace average_playtime_l233_233589

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l233_233589


namespace wendy_full_face_time_l233_233053

theorem wendy_full_face_time (a b c : ℕ) (H1 : a = 5) (H2 : b = 5) (H3 : c = 30) : a * b + c = 55 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end wendy_full_face_time_l233_233053


namespace cocos_August_bill_l233_233128

noncomputable def total_cost (a_monthly_cost: List (Float × Float)) :=
a_monthly_cost.foldr (fun x acc => (x.1 * x.2 * 0.09) + acc) 0

theorem cocos_August_bill :
  let oven        := (2.4, 25)
  let air_cond    := (1.6, 150)
  let refrigerator := (0.15, 720)
  let washing_mach := (0.5, 20) 
  total_cost [oven, air_cond, refrigerator, washing_mach] = 37.62 :=
by
  sorry

end cocos_August_bill_l233_233128


namespace sum_of_z_values_l233_233273

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem sum_of_z_values (z1 z2 : ℝ) (hz1 : f (3 * z1) = 11) (hz2 : f (3 * z2) = 11) :
  z1 + z2 = - (2 / 9) :=
sorry

end sum_of_z_values_l233_233273


namespace motorist_spent_on_petrol_l233_233390

def original_price_per_gallon : ℝ := 5.56
def reduction_percentage : ℝ := 0.10
def new_price_per_gallon := original_price_per_gallon - (0.10 * original_price_per_gallon)
def gallons_more_after_reduction : ℝ := 5

theorem motorist_spent_on_petrol (X : ℝ) 
  (h1 : new_price_per_gallon = original_price_per_gallon - (reduction_percentage * original_price_per_gallon))
  (h2 : (X / new_price_per_gallon) - (X / original_price_per_gallon) = gallons_more_after_reduction) :
  X = 250.22 :=
by
  sorry

end motorist_spent_on_petrol_l233_233390


namespace angus_tokens_eq_l233_233393

-- Define the conditions
def worth_per_token : ℕ := 4
def elsa_tokens : ℕ := 60
def angus_less_worth : ℕ := 20

-- Define the main theorem to prove
theorem angus_tokens_eq :
  let elsaTokens := elsa_tokens,
      worthPerToken := worth_per_token,
      angusLessTokens := angus_less_worth / worth_per_token
  in (elsaTokens - angusLessTokens) = 55 := by
  sorry

end angus_tokens_eq_l233_233393


namespace union_of_M_and_Q_is_correct_l233_233549

-- Given sets M and Q
def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

-- Statement to prove
theorem union_of_M_and_Q_is_correct : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_of_M_and_Q_is_correct_l233_233549


namespace union_complement_l233_233300

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l233_233300


namespace complement_union_l233_233897

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l233_233897


namespace q_minus_p_897_l233_233997

def smallest_three_digit_integer_congruent_7_mod_13 := ∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7
def smallest_four_digit_integer_congruent_7_mod_13 := ∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7

theorem q_minus_p_897 : 
  (∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7) → 
  (∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7) → 
  ∀ p q : ℕ, 
    (p = 8*13+7) → 
    (q = 77*13+7) → 
    q - p = 897 :=
by
  intros h1 h2 p q hp hq
  sorry

end q_minus_p_897_l233_233997


namespace stephanie_oranges_l233_233166

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l233_233166


namespace maries_trip_distance_l233_233325

theorem maries_trip_distance (x : ℚ)
  (h1 : x = x / 4 + 15 + x / 6) :
  x = 180 / 7 :=
by
  sorry

end maries_trip_distance_l233_233325


namespace term_omit_perfect_squares_300_l233_233628

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l233_233628


namespace triangle_inequality_l233_233146

theorem triangle_inequality
  (α β γ a b c : ℝ)
  (h_angles_sum : α + β + γ = Real.pi)
  (h_pos_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 2 * (a / α + b / β + c / γ) := by
  sorry

end triangle_inequality_l233_233146


namespace total_herd_l233_233203

theorem total_herd (n : ℕ) (h : n > 0) (h1 : (1 / 3 : ℚ) * n ∈ ℤ) (h2: (1 / 6 : ℚ) * n ∈ ℤ) (h3: (1 / 9 : ℚ) * n ∈ ℤ) (h4 : (2 / 9 : ℚ) * n = 11) :
  n = 54 :=
by
  sorry

end total_herd_l233_233203


namespace stephanie_oranges_l233_233164

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l233_233164


namespace ellipse_foci_cond_l233_233173

theorem ellipse_foci_cond (m n : ℝ) (h_cond : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → (m > n ∧ n > 0)) ∧ ((m > n ∧ n > 0) → ∀ x y : ℝ, mx^2 + ny^2 = 1) :=
sorry

end ellipse_foci_cond_l233_233173


namespace total_books_l233_233456

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) : 
  joan_books + tom_books + sarah_books + alex_books = 118 := 
by 
  sorry

end total_books_l233_233456


namespace sum_of_squares_eq_three_l233_233110

theorem sum_of_squares_eq_three
  (a b s : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a * s^2 + b * s + b = 0)
  (h₂ : a * (1 / s)^2 + a * (1 / s) + b = 0)
  (h₃ : s * (1 / s) = 1) :
  s^2 + (1 / s)^2 = 3 := 
sorry

end sum_of_squares_eq_three_l233_233110


namespace min_y_squared_l233_233148

noncomputable def isosceles_trapezoid_bases (EF GH : ℝ) := EF = 102 ∧ GH = 26

noncomputable def trapezoid_sides (EG FH y : ℝ) := EG = y ∧ FH = y

noncomputable def tangent_circle (center_on_EF tangent_to_EG_FH : Prop) := 
  ∃ P : ℝ × ℝ, true -- center P exists somewhere and lies on EF

theorem min_y_squared (EF GH EG FH y : ℝ) (center_on_EF tangent_to_EG_FH : Prop) 
  (h1 : isosceles_trapezoid_bases EF GH)
  (h2 : trapezoid_sides EG FH y)
  (h3 : tangent_circle center_on_EF tangent_to_EG_FH) : 
  ∃ n : ℝ, n^2 = 1938 :=
sorry

end min_y_squared_l233_233148


namespace smallest_possible_N_l233_233732

theorem smallest_possible_N :
  ∀ (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0),
  p + q + r + s + t = 4020 →
  (∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1342) :=
by
  intros p q r s t hp hq hr hs ht h
  use 1342
  sorry

end smallest_possible_N_l233_233732


namespace time_left_to_room_l233_233151

theorem time_left_to_room (total_time minutes_to_gate minutes_to_building : ℕ) 
  (h1 : total_time = 30) 
  (h2 : minutes_to_gate = 15) 
  (h3 : minutes_to_building = 6) : 
  total_time - (minutes_to_gate + minutes_to_building) = 9 :=
by 
  sorry

end time_left_to_room_l233_233151


namespace interval_of_a_l233_233643

theorem interval_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_monotone : ∀ x y, x < y → f y ≤ f x)
  (h_condition : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) : 
  a ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 5 :=
by
  sorry

end interval_of_a_l233_233643


namespace fewest_occupied_seats_l233_233618

theorem fewest_occupied_seats (n m : ℕ) (h₁ : n = 150) (h₂ : (m * 4 + 3 < 150)) : m = 37 :=
by
  sorry

end fewest_occupied_seats_l233_233618


namespace complement_union_eq_l233_233929

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l233_233929


namespace wendy_full_face_time_l233_233052

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time_l233_233052


namespace Jack_remaining_money_l233_233142

-- Definitions based on conditions
def initial_money : ℕ := 100
def initial_bottles : ℕ := 4
def bottle_cost : ℕ := 2
def extra_bottles : ℕ := 8
def cheese_cost_per_pound : ℕ := 10
def cheese_weight : ℚ := 1 / 2

-- The statement we want to prove
theorem Jack_remaining_money :
  let total_water_cost := (initial_bottles + extra_bottles) * bottle_cost,
      total_cheese_cost := cheese_cost_per_pound * cheese_weight,
      total_cost := total_water_cost + total_cheese_cost
  in initial_money - total_cost = 71 :=
by
  sorry

end Jack_remaining_money_l233_233142


namespace sufficient_but_not_necessary_condition_l233_233421

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : -1 < x ∧ x < 3) (q : x^2 - 5 * x - 6 < 0) : 
  (-1 < x ∧ x < 3) → (x^2 - 5 * x - 6 < 0) ∧ ¬((x^2 - 5 * x - 6 < 0) → (-1 < x ∧ x < 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l233_233421


namespace wood_pieces_gathered_l233_233087

theorem wood_pieces_gathered (sacks : ℕ) (pieces_per_sack : ℕ) (total_pieces : ℕ)
  (h1 : sacks = 4)
  (h2 : pieces_per_sack = 20)
  (h3 : total_pieces = sacks * pieces_per_sack) :
  total_pieces = 80 :=
by
  sorry

end wood_pieces_gathered_l233_233087


namespace initial_ratio_men_to_women_l233_233577

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l233_233577


namespace determine_remainder_l233_233056

-- Define the sequence and its sum
def geom_series_sum_mod (a r n m : ℕ) : ℕ := 
  ((r^(n+1) - 1) / (r - 1)) % m

-- Define the specific geometric series and modulo
theorem determine_remainder :
  geom_series_sum_mod 1 11 1800 500 = 1 :=
by
  -- Using geom_series_sum_mod to define the series
  let S := geom_series_sum_mod 1 11 1800 500
  -- Remainder when the series is divided by 500
  show S = 1
  sorry

end determine_remainder_l233_233056


namespace complement_union_of_M_and_N_l233_233889

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l233_233889


namespace evaluate_expression_l233_233679

theorem evaluate_expression (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
    (x - (y - z)) - ((x - y) - z) = 14 := by 
  sorry

end evaluate_expression_l233_233679


namespace income_to_expenditure_ratio_l233_233177

theorem income_to_expenditure_ratio (I E S : ℕ) (hI : I = 15000) (hS : S = 7000) (hSavings : S = I - E) :
  I / E = 15 / 8 := by
  -- Lean proof goes here
  sorry

end income_to_expenditure_ratio_l233_233177


namespace num_vec_a_exists_l233_233725

-- Define the vectors and the conditions
def vec_a (x y : ℝ) : (ℝ × ℝ) := (x, y)
def vec_b (x y : ℝ) : (ℝ × ℝ) := (x^2, y^2)
def vec_c : (ℝ × ℝ) := (1, 1)

-- Define the dot product
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the conditions
def cond_1 (x y : ℝ) : Prop := (x + y = 1)
def cond_2 (x y : ℝ) : Prop := (x^2 / 4 + (1 - x)^2 / 9 = 1)

-- The proof problem statement
theorem num_vec_a_exists : ∃! (x y : ℝ), cond_1 x y ∧ cond_2 x y := by
  sorry

end num_vec_a_exists_l233_233725


namespace complement_union_l233_233882

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233882


namespace intersection_with_y_axis_l233_233485

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis_l233_233485


namespace arithmetic_series_sum_l233_233816

def a := 5
def l := 20
def n := 16
def S := (n / 2) * (a + l)

theorem arithmetic_series_sum :
  S = 200 :=
by
  sorry

end arithmetic_series_sum_l233_233816


namespace sample_size_l233_233201

theorem sample_size (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : 8 = n * 2 / 10) : n = 40 :=
by
  sorry

end sample_size_l233_233201


namespace find_original_number_l233_233489

theorem find_original_number (x : ℝ) (h : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end find_original_number_l233_233489


namespace arithmetic_square_root_l233_233121

theorem arithmetic_square_root (n : ℝ) (h : (-5)^2 = n) : Real.sqrt n = 5 :=
by
  sorry

end arithmetic_square_root_l233_233121


namespace sum_of_first_three_tests_l233_233471

variable (A B C: ℕ)

def scores (A B C test4 : ℕ) : Prop := (A + B + C + test4) / 4 = 85

theorem sum_of_first_three_tests (h : scores A B C 100) : A + B + C = 240 :=
by
  -- Proof goes here
  sorry

end sum_of_first_three_tests_l233_233471


namespace problem_l233_233733

theorem problem
  (r s t : ℝ)
  (h₀ : r^3 - 15 * r^2 + 13 * r - 8 = 0)
  (h₁ : s^3 - 15 * s^2 + 13 * s - 8 = 0)
  (h₂ : t^3 - 15 * t^2 + 13 * t - 8 = 0) :
  (r / (1 / r + s * t) + s / (1 / s + t * r) + t / (1 / t + r * s) = 199 / 9) :=
sorry

end problem_l233_233733


namespace find_p_l233_233123

theorem find_p 
  (p q x y : ℤ)
  (h1 : p * x + q * y = 8)
  (h2 : 3 * x - q * y = 38)
  (hx : x = 2)
  (hy : y = -4) : 
  p = 20 := 
by 
  subst hx
  subst hy
  sorry

end find_p_l233_233123


namespace correct_option_is_C_l233_233639

theorem correct_option_is_C 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (D : Prop)
  (hA : ¬ A)
  (hB : ¬ B)
  (hD : ¬ D)
  (hC : C) :
  C := by
  exact hC

end correct_option_is_C_l233_233639


namespace solve_for_V_l233_233550

open Real

theorem solve_for_V :
  ∃ k V, 
    (U = k * (V / W) ∧ (U = 16 ∧ W = 1 / 4 ∧ V = 2) ∧ (U = 25 ∧ W = 1 / 5 ∧ V = 2.5)) :=
by {
  sorry
}

end solve_for_V_l233_233550


namespace complement_union_M_N_l233_233957

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l233_233957


namespace groupA_forms_triangle_l233_233774

theorem groupA_forms_triangle (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 20) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  sorry
}

end groupA_forms_triangle_l233_233774


namespace boat_travel_distance_along_stream_l233_233722

theorem boat_travel_distance_along_stream :
  ∀ (v_s : ℝ), (5 - v_s = 2) → (5 + v_s) * 1 = 8 :=
by
  intro v_s
  intro h1
  have vs_value : v_s = 3 := by linarith
  rw [vs_value]
  norm_num

end boat_travel_distance_along_stream_l233_233722


namespace number_of_children_l233_233328

-- Define the number of adults and their ticket price
def num_adults := 9
def adult_ticket_price := 11

-- Define the children's ticket price and the total cost difference
def child_ticket_price := 7
def cost_difference := 50

-- Define the total cost for adult tickets
def total_adult_cost := num_adults * adult_ticket_price

-- Given the conditions, prove that the number of children is 7
theorem number_of_children : ∃ c : ℕ, total_adult_cost = c * child_ticket_price + cost_difference ∧ c = 7 :=
by
  sorry

end number_of_children_l233_233328


namespace external_bisector_l233_233470

-- Define the relevant points and segments in the triangle
variables {A T C L K : Type} [inhabited A] [inhabited T] [inhabited C] [inhabited L] [inhabited K]

-- Assuming T, L, K are points such that TL divides angle ATC internally
axiom internal_bisector (h : L) : ∃ TL, TL.bisects_angle (A T C) internally

-- Prove that TK is the external angle bisector given the condition above
theorem external_bisector (h : L) (h_internal : internal_bisector h) : 
  ∃ TK, TK.bisects_angle (A T C) externally :=
sorry

end external_bisector_l233_233470


namespace ratio_of_a_to_b_l233_233613

variable (a b x m : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_x : x = 1.25 * a) (h_m : m = 0.6 * b)
variable (h_ratio : m / x = 0.6)

theorem ratio_of_a_to_b (h_x : x = 1.25 * a) (h_m : m = 0.6 * b) (h_ratio : m / x = 0.6) : a / b = 0.8 :=
by
  sorry

end ratio_of_a_to_b_l233_233613


namespace variance_of_transformed_binomial_l233_233745

open ProbabilityTheory

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

noncomputable def D (Y : ℚ) : ℚ := binomial_variance 3 ((5/9 : ℚ) - (4/9 : ℚ))

theorem variance_of_transformed_binomial :
  ∃ (p : ℚ),
    (Pr (X ≥ 1) = 5 / 9) ∧ (D (3 * Y + 1) = 6) :=
begin
  sorry
end

end variance_of_transformed_binomial_l233_233745


namespace example_theorem_l233_233864

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l233_233864


namespace MountainRidgeAcademy_l233_233396

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy_l233_233396


namespace ribbon_left_l233_233001

-- Define the variables
def T : ℕ := 18 -- Total ribbon in yards
def G : ℕ := 6  -- Number of gifts
def P : ℕ := 2  -- Ribbon per gift in yards

-- Statement of the theorem
theorem ribbon_left (T G P : ℕ) : (T - G * P) = 6 :=
by
  -- Add conditions as Lean assumptions
  have hT : T = 18 := sorry
  have hG : G = 6 := sorry
  have hP : P = 2 := sorry
  -- Now prove the final result
  sorry

end ribbon_left_l233_233001


namespace pastries_left_to_take_home_l233_233103

def initial_cupcakes : ℕ := 7
def initial_cookies : ℕ := 5
def pastries_sold : ℕ := 4

theorem pastries_left_to_take_home :
  initial_cupcakes + initial_cookies - pastries_sold = 8 := by
  sorry

end pastries_left_to_take_home_l233_233103


namespace quadratic_inequality_condition_l233_233775

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 3 := 
sorry

end quadratic_inequality_condition_l233_233775


namespace fraction_of_ABCD_is_shaded_l233_233764

noncomputable def squareIsDividedIntoTriangles : Type := sorry
noncomputable def areTrianglesIdentical (s : squareIsDividedIntoTriangles) : Prop := sorry
noncomputable def isFractionShadedCorrect : Prop := 
  ∃ (s : squareIsDividedIntoTriangles), 
  areTrianglesIdentical s ∧ 
  (7 / 16 : ℚ) = 7 / 16

theorem fraction_of_ABCD_is_shaded (s : squareIsDividedIntoTriangles) :
  areTrianglesIdentical s → (7 / 16 : ℚ) = 7 / 16 :=
sorry

end fraction_of_ABCD_is_shaded_l233_233764


namespace find_nine_day_segment_l233_233389

/-- 
  Definitions:
  - ws_day: The Winter Solstice day, December 21, 2012.
  - j1_day: New Year's Day, January 1, 2013.
  - Calculate the total days difference between ws_day and j1_day.
  - Check that the distribution of days into 9-day segments leads to January 1, 2013, being the third day of the second segment.
-/
def ws_day : ℕ := 21
def j1_day : ℕ := 1
def days_in_december : ℕ := 31
def days_ws_to_end_dec : ℕ := days_in_december - ws_day + 1
def total_days : ℕ := days_ws_to_end_dec + j1_day

theorem find_nine_day_segment : (total_days % 9) = 3 ∧ (total_days / 9) = 1 := by
  sorry  -- Proof skipped

end find_nine_day_segment_l233_233389


namespace factorial_div_sub_factorial_equality_l233_233366

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end factorial_div_sub_factorial_equality_l233_233366


namespace nonneg_real_sum_inequality_l233_233696

theorem nonneg_real_sum_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end nonneg_real_sum_inequality_l233_233696


namespace new_cooks_waiters_ratio_l233_233219

-- Definitions based on the conditions
variables (cooks waiters new_waiters : ℕ)

-- Given conditions
def ratio := 3
def initial_waiters := (ratio * cooks) / 3 -- Derived from 3 cooks / 11 waiters = 9 cooks / x waiters
def hired_waiters := 12
def total_waiters := initial_waiters + hired_waiters

-- The restaurant has 9 cooks
def restaurant_cooks := 9

-- Conclusion to prove
theorem new_cooks_waiters_ratio :
  (ratio = 3) →
  (restaurant_cooks = 9) →
  (initial_waiters = (ratio * restaurant_cooks) / 3) →
  (cooks = restaurant_cooks) →
  (waiters = initial_waiters) →
  (new_waiters = waiters + hired_waiters) →
  (new_waiters = 45) →
  (cooks / new_waiters = 1 / 5) :=
by
  intros
  sorry

end new_cooks_waiters_ratio_l233_233219


namespace min_value_l233_233689

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) :
  2 * a + b + c ≥ 4 :=
sorry

end min_value_l233_233689


namespace charlie_cookies_l233_233504

theorem charlie_cookies (father_cookies mother_cookies total_cookies charlie_cookies : ℕ)
  (h1 : father_cookies = 10) (h2 : mother_cookies = 5) (h3 : total_cookies = 30) :
  father_cookies + mother_cookies + charlie_cookies = total_cookies → charlie_cookies = 15 :=
by
  intros h
  sorry

end charlie_cookies_l233_233504


namespace intersection_A_B_l233_233247

def A : Set Int := {-1, 0, 1, 5, 8}
def B : Set Int := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} :=
by
  sorry

end intersection_A_B_l233_233247


namespace complement_union_l233_233940

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233940


namespace lily_pads_cover_half_l233_233640

theorem lily_pads_cover_half (P D : ℕ) (cover_entire : P * (2 ^ 25) = D) : P * (2 ^ 24) = D / 2 :=
by sorry

end lily_pads_cover_half_l233_233640


namespace rows_per_floor_l233_233458

theorem rows_per_floor
  (right_pos : ℕ) (left_pos : ℕ)
  (floors : ℕ) (total_cars : ℕ)
  (h_right : right_pos = 5) (h_left : left_pos = 4)
  (h_floors : floors = 10) (h_total : total_cars = 1600) :
  ∃ rows_per_floor : ℕ, rows_per_floor = 20 :=
by {
  sorry
}

end rows_per_floor_l233_233458


namespace Quincy_sold_more_l233_233452

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l233_233452


namespace parabola_distance_l233_233240

theorem parabola_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (h_distance_focus : (P.1 - 1)^2 + P.2^2 = 9) : 
  Real.sqrt (P.1^2 + P.2^2) = 2 * Real.sqrt 3 :=
by
  sorry

end parabola_distance_l233_233240


namespace sum_of_other_endpoint_l233_233016

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l233_233016


namespace complement_union_l233_233944

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l233_233944


namespace determine_omega_value_l233_233699

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + (π / 6))

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x - (ω * π / 6) + (π / 6))

theorem determine_omega_value (ω : ℝ) : 
  (ω > 0) →
  (∀ x, g ω (x + π / 6) = f ω x) →
  (∀ x, g ω (-π / 6 - x) = g ω (-π / 6 + x)) →
  (∀ x, (π / 6) ≤ x ∧ x ≤ (π / 3) → monotone_decreasing (f ω x)) →
  ω = 2 :=
by
  sorry

end determine_omega_value_l233_233699


namespace min_garden_cost_l233_233027

theorem min_garden_cost : 
  let flower_cost (flower : String) : Real :=
    if flower = "Asters" then 1 else
    if flower = "Begonias" then 2 else
    if flower = "Cannas" then 2 else
    if flower = "Dahlias" then 3 else
    if flower = "Easter lilies" then 2.5 else
    0
  let region_area (region : String) : Nat :=
    if region = "Bottom left" then 10 else
    if region = "Top left" then 9 else
    if region = "Bottom right" then 20 else
    if region = "Top middle" then 2 else
    if region = "Top right" then 7 else
    0
  let min_cost : Real :=
    (flower_cost "Dahlias" * region_area "Top middle") + 
    (flower_cost "Easter lilies" * region_area "Top right") + 
    (flower_cost "Cannas" * region_area "Top left") + 
    (flower_cost "Begonias" * region_area "Bottom left") + 
    (flower_cost "Asters" * region_area "Bottom right")
  min_cost = 81.5 :=
by
  sorry

end min_garden_cost_l233_233027


namespace total_number_of_workers_l233_233350

theorem total_number_of_workers 
    (W : ℕ) 
    (average_salary_all : ℕ := 8000) 
    (average_salary_technicians : ℕ := 12000) 
    (average_salary_rest : ℕ := 6000) 
    (total_salary_all : ℕ := average_salary_all * W) 
    (salary_technicians : ℕ := 6 * average_salary_technicians) 
    (N : ℕ := W - 6) 
    (salary_rest : ℕ := average_salary_rest * N) 
    (salary_equation : total_salary_all = salary_technicians + salary_rest) 
  : W = 18 := 
sorry

end total_number_of_workers_l233_233350


namespace sum_of_fractions_l233_233398

theorem sum_of_fractions : 
  (2 / 5 : ℚ) + (4 / 50 : ℚ) + (3 / 500 : ℚ) + (8 / 5000 : ℚ) = 4876 / 10000 :=
by
  -- The proof can be completed by converting fractions and summing them accurately.
  sorry

end sum_of_fractions_l233_233398


namespace value_of_M_l233_233964

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l233_233964


namespace quadratic_solution_range_l233_233823

noncomputable def quadratic_inequality_real_solution (c : ℝ) : Prop :=
  0 < c ∧ c < 16

theorem quadratic_solution_range :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 8 * x + c < 0) ↔ quadratic_inequality_real_solution c :=
by
  intro c
  simp only [quadratic_inequality_real_solution]
  sorry

end quadratic_solution_range_l233_233823


namespace choose_7_starters_with_at_least_one_quadruplet_l233_233330

-- Given conditions
variable (n : ℕ := 18) -- total players
variable (k : ℕ := 7)  -- number of starters
variable (q : ℕ := 4)  -- number of quadruplets

-- Lean statement
theorem choose_7_starters_with_at_least_one_quadruplet 
  (h : n = 18) 
  (h1 : k = 7) 
  (h2 : q = 4) :
  (Nat.choose 18 7 - Nat.choose 14 7) = 28392 :=
by
  sorry

end choose_7_starters_with_at_least_one_quadruplet_l233_233330


namespace graph_not_in_first_quadrant_l233_233612

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

-- Prove that the graph of f(x) does not pass through the first quadrant
theorem graph_not_in_first_quadrant : ∀ (x : ℝ), x > 0 → f x ≤ 0 := by
  intro x hx
  sorry

end graph_not_in_first_quadrant_l233_233612


namespace range_of_phi_l233_233856

theorem range_of_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : |φ| < (Real.pi / 2))
  (h3 : ∀ x, f x = Real.sin (ω * x + φ))
  (h4 : ∀ x, f (x + (Real.pi / ω)) = f x)
  (h5 : ∀ x y, (x ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) ∧
                  (y ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) → 
                  (x < y → f x ≤ f y)) :
  (φ ∈ Set.Icc (- Real.pi / 6) (- Real.pi / 10)) :=
by
  sorry

end range_of_phi_l233_233856


namespace complement_union_l233_233879

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233879


namespace b5_b9_equal_16_l233_233844

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} (h_arith : ∀ n m, a m = a n + (m - n) * (a 1 - a 0))
variable (h_non_zero : ∀ n, a n ≠ 0)
variable (h_cond : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)

-- Define the geometric sequence and condition
variables {b : ℕ → ℝ} (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
variable (h_b7 : b 7 = a 7)

-- State the theorem to prove
theorem b5_b9_equal_16 : b 5 * b 9 = 16 :=
sorry

end b5_b9_equal_16_l233_233844


namespace arccos_zero_eq_pi_div_two_l233_233671

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l233_233671


namespace complement_union_l233_233878

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233878


namespace first_day_is_sunday_l233_233346

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open Day

-- Function to determine the day of the week for a given day number
def day_of_month (n : ℕ) (start_day : Day) : Day :=
  match n % 7 with
  | 0 => start_day
  | 1 => match start_day with
          | Sunday    => Monday
          | Monday    => Tuesday
          | Tuesday   => Wednesday
          | Wednesday => Thursday
          | Thursday  => Friday
          | Friday    => Saturday
          | Saturday  => Sunday
  | 2 => match start_day with
          | Sunday    => Tuesday
          | Monday    => Wednesday
          | Tuesday   => Thursday
          | Wednesday => Friday
          | Thursday  => Saturday
          | Friday    => Sunday
          | Saturday  => Monday
-- ... and so on for the rest of the days of the week.
  | _ => start_day -- Assuming the pattern continues accordingly.

-- Prove that the first day of the month is a Sunday given that the 18th day of the month is a Wednesday.
theorem first_day_is_sunday (h : day_of_month 18 Wednesday = Wednesday) : day_of_month 1 Wednesday = Sunday :=
  sorry

end first_day_is_sunday_l233_233346


namespace find_n_in_range_l233_233543

theorem find_n_in_range : ∃ n, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [MOD 7] ∧ n = 8 := 
by
  sorry

end find_n_in_range_l233_233543


namespace find_n_l233_233994

theorem find_n (n : ℕ) (m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_div : (2^n - 1) ∣ (m^2 + 81)) : 
  ∃ k : ℕ, n = 2^k := 
sorry

end find_n_l233_233994


namespace value_of_M_l233_233963

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l233_233963


namespace find_x_l233_233055

theorem find_x
  (x : ℤ)
  (h1 : 71 * x % 9 = 8) :
  x = 1 :=
sorry

end find_x_l233_233055


namespace find_f1_increasing_on_positive_solve_inequality_l233_233851

-- Given conditions
axiom f : ℝ → ℝ
axiom domain : ∀ x, 0 < x → true
axiom f4 : f 4 = 1
axiom multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom less_than_zero : ∀ x, 0 < x ∧ x < 1 → f x < 0

-- Required proofs
theorem find_f1 : f 1 = 0 := sorry

theorem increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem solve_inequality : {x : ℝ // 3 < x ∧ x ≤ 5} := sorry

end find_f1_increasing_on_positive_solve_inequality_l233_233851


namespace cylinder_volume_l233_233617

-- Define the volume of the cone
def V_cone : ℝ := 18.84

-- Define the volume of the cylinder
def V_cylinder : ℝ := 3 * V_cone

-- Prove that the volume of the cylinder is 56.52 cubic meters
theorem cylinder_volume :
  V_cylinder = 56.52 := 
by 
  -- the proof will go here
  sorry

end cylinder_volume_l233_233617


namespace total_soda_bottles_l233_233646

def regular_soda : ℕ := 57
def diet_soda : ℕ := 26
def lite_soda : ℕ := 27

theorem total_soda_bottles : regular_soda + diet_soda + lite_soda = 110 := by
  sorry

end total_soda_bottles_l233_233646


namespace triangle_altitude_length_l233_233139

variable (AB AC BC BA1 AA1 : ℝ)
variable (eq1 : AB = 8)
variable (eq2 : AC = 10)
variable (eq3 : BC = 12)

theorem triangle_altitude_length (h : ∃ AA1, AA1 * AA1 + BA1 * BA1 = 64 ∧ 
                                AA1 * AA1 + (BC - BA1) * (BC - BA1) = 100) :
    BA1 = 4.5 := by
  sorry 

end triangle_altitude_length_l233_233139


namespace find_q_l233_233553

noncomputable def q_value (m q : ℕ) : Prop := 
  ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (q * 10 ^ 31)

theorem find_q (m : ℕ) (q : ℕ) (h1 : m = 31) (h2 : q_value m q) : q = 2 :=
by
  sorry

end find_q_l233_233553


namespace both_players_same_score_probability_l233_233239

theorem both_players_same_score_probability :
  let p_A_score := 0.6
  let p_B_score := 0.8
  let p_A_miss := 1 - p_A_score
  let p_B_miss := 1 - p_B_score
  (p_A_score * p_B_score + p_A_miss * p_B_miss = 0.56) :=
by
  sorry

end both_players_same_score_probability_l233_233239


namespace marked_price_l233_233071

theorem marked_price (x : ℝ) (purchase_price : ℝ) (selling_price : ℝ) (profit_margin : ℝ) 
  (h_purchase_price : purchase_price = 100)
  (h_profit_margin : profit_margin = 0.2)
  (h_selling_price : selling_price = purchase_price * (1 + profit_margin))
  (h_price_relation : 0.8 * x = selling_price) : 
  x = 150 :=
by sorry

end marked_price_l233_233071


namespace nadine_white_pebbles_l233_233594

variable (W R : ℝ)

theorem nadine_white_pebbles :
  (R = 1/2 * W) →
  (W + R = 30) →
  W = 20 :=
by
  sorry

end nadine_white_pebbles_l233_233594


namespace union_A_B_eq_neg2_neg1_0_l233_233424

def setA : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}
def setB : Set ℤ := {-2, -1}

theorem union_A_B_eq_neg2_neg1_0 : (setA ∪ setB) = ({-2, -1, 0} : Set ℤ) :=
by
  sorry

end union_A_B_eq_neg2_neg1_0_l233_233424


namespace regular_polygon_sides_l233_233208

theorem regular_polygon_sides (P s : ℕ) (hP : P = 150) (hs : s = 15) :
  P / s = 10 :=
by
  sorry

end regular_polygon_sides_l233_233208


namespace zero_positive_integers_prime_polynomial_l233_233225

noncomputable def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem zero_positive_integers_prime_polynomial :
  ∀ (n : ℕ), ¬ is_prime (n^3 - 7 * n^2 + 16 * n - 12) :=
by
  sorry

end zero_positive_integers_prime_polynomial_l233_233225


namespace marcus_percentage_of_team_points_l233_233322

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l233_233322


namespace intersection_with_y_axis_l233_233486

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l233_233486


namespace largest_solution_of_equation_l233_233636

theorem largest_solution_of_equation :
  let eq := λ x : ℝ => x^4 - 50 * x^2 + 625
  ∃ x : ℝ, eq x = 0 ∧ ∀ y : ℝ, eq y = 0 → y ≤ x :=
sorry

end largest_solution_of_equation_l233_233636


namespace largest_x_value_l233_233409

def largest_solution_inequality (x : ℝ) : Prop := 
  (-(Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25)))^3 + 
  abs (Real.log 3 ((100 + 2 * x * Real.sqrt (2 * x + 25)) / (x^2 + 2 * x + 4)^4))) / 
  (3 * Real.log 6 (50 + 2 * x * Real.sqrt (2 * x + 25)) - 
  2 * Real.log 3 (100 + 2 * x * Real.sqrt (2 * x + 25))) ≥ 0 → 
  x ≤ 12 + 4 * Real.sqrt 3

theorem largest_x_value : ∃ x : ℝ, largest_solution_inequality x :=
sorry

end largest_x_value_l233_233409


namespace correct_inequality_l233_233809

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_increasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

theorem correct_inequality : f (-2) < f 1 ∧ f 1 < f 3 :=
by 
  sorry

end correct_inequality_l233_233809


namespace triangle_inequality_l233_233483

-- Define the triangle angles, semiperimeter, and circumcircle radius
variables (α β γ s R : Real)

-- Define the sum of angles in a triangle
axiom angle_sum : α + β + γ = Real.pi

-- The inequality to prove
theorem triangle_inequality (h_sum : α + β + γ = Real.pi) :
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (Real.pi / Real.sqrt 3)^3 * R / s := sorry

end triangle_inequality_l233_233483


namespace sum_of_a_b_l233_233842

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l233_233842


namespace pentagon_total_area_l233_233531

-- Conditions definition
variables {a b c d e : ℕ}
variables {side1 side2 side3 side4 side5 : ℕ} 
variables {h : ℕ}
variables {triangle_area : ℕ}
variables {trapezoid_area : ℕ}
variables {total_area : ℕ}

-- Specific conditions given in the problem
def pentagon_sides (a b c d e : ℕ) : Prop :=
  a = 18 ∧ b = 25 ∧ c = 30 ∧ d = 28 ∧ e = 25

def can_be_divided (triangle_area trapezoid_area total_area : ℕ) : Prop :=
  triangle_area = 225 ∧ trapezoid_area = 770 ∧ total_area = 995

-- Total area of the pentagon under given conditions
theorem pentagon_total_area 
  (h_div: can_be_divided triangle_area trapezoid_area total_area) 
  (h_sides: pentagon_sides a b c d e)
  (h: triangle_area + trapezoid_area = total_area) :
  total_area = 995 := 
by
  sorry

end pentagon_total_area_l233_233531


namespace union_complement_eq_target_l233_233313

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l233_233313


namespace bucket_weight_one_third_l233_233367

theorem bucket_weight_one_third 
    (x y c b : ℝ) 
    (h1 : x + 3/4 * y = c)
    (h2 : x + 1/2 * y = b) :
    x + 1/3 * y = 5/3 * b - 2/3 * c :=
by
  sorry

end bucket_weight_one_third_l233_233367


namespace union_complement_set_l233_233287

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l233_233287


namespace find_q_sum_of_bn_l233_233546

-- Defining the sequences and conditions
def a (n : ℕ) (q : ℝ) : ℝ := q^(n-1)

def b (n : ℕ) (q : ℝ) : ℝ := a n q + n

-- Given that 2a_1, (1/2)a_3, a_2 form an arithmetic sequence
def condition_arithmetic_sequence (q : ℝ) : Prop :=
  2 * a 1 q + a 2 q = (1 / 2) * a 3 q + (1 / 2) * a 3 q

-- To be proved: Given conditions, prove q = 2
theorem find_q : ∃ q > 0, a 1 q = 1 ∧ a 2 q = q ∧ a 3 q = q^2 ∧ condition_arithmetic_sequence q ∧ q = 2 :=
by {
  sorry
}

-- Given b_n = a_n + n, prove T_n = (n(n+1))/2 + 2^n - 1
theorem sum_of_bn (n : ℕ) : 
  ∃ T_n : ℕ → ℝ, T_n n = (n * (n + 1)) / 2 + (2^n) - 1 :=
by {
  sorry
}

end find_q_sum_of_bn_l233_233546


namespace maria_must_earn_l233_233748

-- Define the given conditions
def retail_price : ℕ := 600
def maria_savings : ℕ := 120
def mother_contribution : ℕ := 250

-- Total amount Maria has from savings and her mother's contribution
def total_savings : ℕ := maria_savings + mother_contribution

-- Prove that Maria must earn $230 to be able to buy the bike
theorem maria_must_earn : 600 - total_savings = 230 :=
by sorry

end maria_must_earn_l233_233748


namespace linear_function_quadrants_l233_233692

theorem linear_function_quadrants
  (k : ℝ) (h₀ : k ≠ 0) (h₁ : ∀ x : ℝ, x > 0 → k*x < 0) :
  (∃ x > 0, 2*x + k > 0) ∧
  (∃ x > 0, 2*x + k < 0) ∧
  (∃ x < 0, 2*x + k < 0) :=
  by
  sorry

end linear_function_quadrants_l233_233692


namespace lollipop_cases_l233_233172

theorem lollipop_cases (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) 
  (h1 : total_cases = 80) (h2 : chocolate_cases = 25) : lollipop_cases = 55 :=
by
  sorry

end lollipop_cases_l233_233172


namespace total_corn_yield_l233_233785

/-- 
The total corn yield in centners, harvested from a certain field area, is expressed 
as a four-digit number composed of the digits 0, 2, 3, and 5. When the average 
yield per hectare was calculated, it was found to be the same number of centners 
as the number of hectares of the field area. 
This statement proves that the total corn yield is 3025. 
-/
theorem total_corn_yield : ∃ (Y A : ℕ), (Y = A^2) ∧ (A >= 10 ∧ A < 100) ∧ 
  (Y / 1000 != 0) ∧ (Y / 1000 != 1) ∧ (Y / 10 % 10 != 4) ∧ 
  (Y % 10 != 1) ∧ (Y % 10 = 0 ∨ Y % 10 = 5) ∧ 
  (Y / 100 % 10 == 0 ∨ Y / 100 % 10 == 2 ∨ Y / 100 % 10 == 3 ∨ Y / 100 % 10 == 5) ∧ 
  Y = 3025 := 
by 
  sorry

end total_corn_yield_l233_233785


namespace complement_union_l233_233912

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l233_233912


namespace complement_union_l233_233880

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l233_233880
