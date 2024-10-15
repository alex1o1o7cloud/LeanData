import Mathlib

namespace NUMINAMATH_GPT_production_today_l2000_200026

theorem production_today (n x: ℕ) (avg_past: ℕ) 
  (h1: avg_past = 50) 
  (h2: n = 1) 
  (h3: (avg_past * n + x) / (n + 1) = 55): 
  x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_production_today_l2000_200026


namespace NUMINAMATH_GPT_Rachel_picked_apples_l2000_200007

theorem Rachel_picked_apples :
  let apples_from_first_tree := 8
  let apples_from_second_tree := 10
  let apples_from_third_tree := 12
  let apples_from_fifth_tree := 6
  apples_from_first_tree + apples_from_second_tree + apples_from_third_tree + apples_from_fifth_tree = 36 :=
by
  sorry

end NUMINAMATH_GPT_Rachel_picked_apples_l2000_200007


namespace NUMINAMATH_GPT_swallow_distance_flew_l2000_200097

/-- The TGV departs from Paris at 150 km/h toward Marseille, which is 800 km away, while an intercité departs from Marseille at 50 km/h toward Paris at the same time. A swallow perched on the TGV takes off at that moment, flying at 200 km/h toward Marseille. We aim to prove that the distance flown by the swallow when the two trains meet is 800 km. -/
theorem swallow_distance_flew :
  let distance := 800 -- distance between Paris and Marseille in km
  let speed_TGV := 150 -- speed of TGV in km/h
  let speed_intercite := 50 -- speed of intercité in km/h
  let speed_swallow := 200 -- speed of swallow in km/h
  let combined_speed := speed_TGV + speed_intercite
  let time_to_meet := distance / combined_speed
  let distance_swallow_traveled := speed_swallow * time_to_meet
  distance_swallow_traveled = 800 := 
by
  sorry

end NUMINAMATH_GPT_swallow_distance_flew_l2000_200097


namespace NUMINAMATH_GPT_winning_post_distance_l2000_200040

theorem winning_post_distance (v_A v_B D : ℝ) (hvA : v_A = (5 / 3) * v_B) (head_start : 80 ≤ D) :
  (D / v_A = (D - 80) / v_B) → D = 200 :=
by
  sorry

end NUMINAMATH_GPT_winning_post_distance_l2000_200040


namespace NUMINAMATH_GPT_S_inter_T_eq_T_l2000_200052

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end NUMINAMATH_GPT_S_inter_T_eq_T_l2000_200052


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2000_200049

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 9*x + 14 < 0) : 2 < x ∧ x < 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2000_200049


namespace NUMINAMATH_GPT_product_of_roots_cubic_eq_l2000_200070

theorem product_of_roots_cubic_eq (α : Type _) [Field α] :
  (∃ (r1 r2 r3 : α), (r1 * r2 * r3 = 6) ∧ (r1 + r2 + r3 = 6) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 11)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_cubic_eq_l2000_200070


namespace NUMINAMATH_GPT_remainder_of_349_divided_by_17_l2000_200032

theorem remainder_of_349_divided_by_17 : 
  (349 % 17 = 9) := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_349_divided_by_17_l2000_200032


namespace NUMINAMATH_GPT_sum_of_cube_faces_l2000_200019

theorem sum_of_cube_faces (a b c d e f : ℕ) (h1 : a % 2 = 0) (h2 : b = a + 2) (h3 : c = b + 2) (h4 : d = c + 2) (h5 : e = d + 2) (h6 : f = e + 2)
(h_pairs : (a + f + 2) = (b + e + 2) ∧ (b + e + 2) = (c + d + 2)) :
  a + b + c + d + e + f = 90 :=
  sorry

end NUMINAMATH_GPT_sum_of_cube_faces_l2000_200019


namespace NUMINAMATH_GPT_suitable_for_sampling_l2000_200001

-- Definitions based on conditions
def optionA_requires_comprehensive : Prop := true
def optionB_requires_comprehensive : Prop := true
def optionC_requires_comprehensive : Prop := true
def optionD_allows_sampling : Prop := true

-- Problem in Lean: Prove that option D is suitable for a sampling survey
theorem suitable_for_sampling : optionD_allows_sampling := by
  sorry

end NUMINAMATH_GPT_suitable_for_sampling_l2000_200001


namespace NUMINAMATH_GPT_quadratic_discriminant_single_solution_l2000_200027

theorem quadratic_discriminant_single_solution :
  ∃ (n : ℝ), (∀ x : ℝ, 9 * x^2 + n * x + 36 = 0 → x = (-n) / (2 * 9)) → n = 36 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_single_solution_l2000_200027


namespace NUMINAMATH_GPT_pranks_combinations_correct_l2000_200002

noncomputable def pranks_combinations : ℕ := by
  let monday_choice := 1
  let tuesday_choice := 2
  let wednesday_choice := 4
  let thursday_choice := 5
  let friday_choice := 1
  let total_combinations := monday_choice * tuesday_choice * wednesday_choice * thursday_choice * friday_choice
  exact 40

theorem pranks_combinations_correct : pranks_combinations = 40 := by
  unfold pranks_combinations
  sorry -- Proof omitted

end NUMINAMATH_GPT_pranks_combinations_correct_l2000_200002


namespace NUMINAMATH_GPT_convert_to_scientific_notation_9600000_l2000_200080

theorem convert_to_scientific_notation_9600000 :
  9600000 = 9.6 * 10^6 := 
sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_9600000_l2000_200080


namespace NUMINAMATH_GPT_good_walker_catch_up_l2000_200058

theorem good_walker_catch_up :
  ∀ x y : ℕ, 
    (x = (100:ℕ) + y) ∧ (x = ((100:ℕ)/(60:ℕ) : ℚ) * y) := 
by
  sorry

end NUMINAMATH_GPT_good_walker_catch_up_l2000_200058


namespace NUMINAMATH_GPT_line_third_quadrant_l2000_200015

theorem line_third_quadrant (A B C : ℝ) (h_origin : C = 0)
  (h_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x - B * y = 0) :
  A * B < 0 :=
by
  sorry

end NUMINAMATH_GPT_line_third_quadrant_l2000_200015


namespace NUMINAMATH_GPT_distinct_solutions_abs_eq_l2000_200011

theorem distinct_solutions_abs_eq : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (|2 * x1 - 14| = |x1 + 4| ∧ |2 * x2 - 14| = |x2 + 4|) ∧ (∀ x, |2 * x - 14| = |x + 4| → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_distinct_solutions_abs_eq_l2000_200011


namespace NUMINAMATH_GPT_rice_and_flour_bags_l2000_200008

theorem rice_and_flour_bags (x : ℕ) (y : ℕ) 
  (h1 : x + y = 351)
  (h2 : x + 20 = 3 * (y - 50) + 1) : 
  x = 221 ∧ y = 130 :=
by
  sorry

end NUMINAMATH_GPT_rice_and_flour_bags_l2000_200008


namespace NUMINAMATH_GPT_max_value_Tn_l2000_200006

noncomputable def geom_seq (a : ℕ → ℝ) : Prop := 
∀ n : ℕ, a (n+1) = 2 * a n

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (2 : ℝ)^n) / (1 - (2 : ℝ))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(9 * sum_first_n_terms a n - sum_first_n_terms a (2 * n)) / (a n * (2 : ℝ)^n)

theorem max_value_Tn (a : ℕ → ℝ) (h : geom_seq a) : 
  ∃ n, T_n a n ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_Tn_l2000_200006


namespace NUMINAMATH_GPT_incenter_closest_to_median_l2000_200082

variables (a b c : ℝ) (s_a s_b s_c d_a d_b d_c : ℝ)

noncomputable def median_length (a b c : ℝ) : ℝ := 
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def distance_to_median (x y median_length : ℝ) : ℝ := 
  (y - x) / (2 * median_length)

theorem incenter_closest_to_median
  (h₀ : a = 4) (h₁ : b = 5) (h₂ : c = 8) 
  (h₃ : s_a = median_length a b c)
  (h₄ : s_b = median_length b a c)
  (h₅ : s_c = median_length c a b)
  (h₆ : d_a = distance_to_median b c s_a)
  (h₇ : d_b = distance_to_median a c s_b)
  (h₈ : d_c = distance_to_median a b s_c) : 
  d_a = d_c := 
sorry

end NUMINAMATH_GPT_incenter_closest_to_median_l2000_200082


namespace NUMINAMATH_GPT_rational_solutions_of_quadratic_l2000_200085

theorem rational_solutions_of_quadratic (k : ℕ) (h_positive : k > 0) :
  (∃ p q : ℚ, p * p + 30 * p * q + k * (q * q) = 0) ↔ k = 9 ∨ k = 15 :=
sorry

end NUMINAMATH_GPT_rational_solutions_of_quadratic_l2000_200085


namespace NUMINAMATH_GPT_adjusted_distance_buoy_fourth_l2000_200061

theorem adjusted_distance_buoy_fourth :
  let a1 := 20  -- distance to the first buoy
  let d := 4    -- common difference (distance between consecutive buoys)
  let ocean_current_effect := 3  -- effect of ocean current
  
  -- distances from the beach to buoys based on their sequence
  let a2 := a1 + d 
  let a3 := a2 + d
  let a4 := a3 + d
  
  -- distance to the fourth buoy without external factors
  let distance_to_fourth_buoy := a1 + 3 * d
  
  -- adjusted distance considering the ocean current
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  adjusted_distance = 29 := 
by
  let a1 := 20
  let d := 4
  let ocean_current_effect := 3
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let distance_to_fourth_buoy := a1 + 3 * d
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  sorry

end NUMINAMATH_GPT_adjusted_distance_buoy_fourth_l2000_200061


namespace NUMINAMATH_GPT_integers_satisfy_equation_l2000_200087

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end NUMINAMATH_GPT_integers_satisfy_equation_l2000_200087


namespace NUMINAMATH_GPT_monotonically_increasing_intervals_min_and_max_values_l2000_200048

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x + Real.pi / 4) + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
    -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → 
    f (x + 1) ≥ f x := sorry

theorem min_and_max_values :
  ∃ min max, 
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min ∧ f x ≤ max) ∧ 
    (min = 0) ∧ 
    (max = Real.sqrt 2 + 1) := sorry

end NUMINAMATH_GPT_monotonically_increasing_intervals_min_and_max_values_l2000_200048


namespace NUMINAMATH_GPT_Mona_grouped_with_one_player_before_in_second_group_l2000_200060

/-- Mona plays in groups with four other players, joined 9 groups, and grouped with 33 unique players. 
    One of the groups included 2 players she had grouped with before. 
    Prove that the number of players she had grouped with before in the second group is 1. -/
theorem Mona_grouped_with_one_player_before_in_second_group 
    (total_groups : ℕ) (group_size : ℕ) (unique_players : ℕ) 
    (repeat_players_in_group1 : ℕ) : 
    (total_groups = 9) → (group_size = 5) → (unique_players = 33) → (repeat_players_in_group1 = 2) 
        → ∃ repeat_players_in_group2 : ℕ, repeat_players_in_group2 = 1 :=
by
    sorry

end NUMINAMATH_GPT_Mona_grouped_with_one_player_before_in_second_group_l2000_200060


namespace NUMINAMATH_GPT_ricciana_jump_distance_l2000_200014

theorem ricciana_jump_distance (R : ℕ) :
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1
  Total_distance_Margarita = Total_distance_Ricciana → R = 22 :=
by
  -- Definitions
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1

  -- Given condition
  intro h
  sorry

end NUMINAMATH_GPT_ricciana_jump_distance_l2000_200014


namespace NUMINAMATH_GPT_smallest_number_of_students_l2000_200025

theorem smallest_number_of_students (n : ℕ) : 
  (6 * n + 2 > 40) → (∃ n, 4 * n + 2 * (n + 1) = 44) :=
 by
  intro h
  exact sorry

end NUMINAMATH_GPT_smallest_number_of_students_l2000_200025


namespace NUMINAMATH_GPT_election_votes_l2000_200024

noncomputable def third_candidate_votes (total_votes first_candidate_votes second_candidate_votes : ℕ) (winning_fraction : ℚ) : ℕ :=
  total_votes - (first_candidate_votes + second_candidate_votes)

theorem election_votes :
  ∃ total_votes : ℕ, 
  ∃ first_candidate_votes : ℕ,
  ∃ second_candidate_votes : ℕ,
  ∃ winning_fraction : ℚ,
  first_candidate_votes = 5000 ∧ 
  second_candidate_votes = 15000 ∧ 
  winning_fraction = 2/3 ∧ 
  total_votes = 60000 ∧ 
  third_candidate_votes total_votes first_candidate_votes second_candidate_votes winning_fraction = 40000 :=
    sorry

end NUMINAMATH_GPT_election_votes_l2000_200024


namespace NUMINAMATH_GPT_toms_total_cost_l2000_200062

theorem toms_total_cost :
  let costA := 4 * 15
  let costB := 3 * 12
  let discountB := 0.20 * costB
  let costBDiscounted := costB - discountB
  let costC := 2 * 18
  costA + costBDiscounted + costC = 124.80 := 
by
  sorry

end NUMINAMATH_GPT_toms_total_cost_l2000_200062


namespace NUMINAMATH_GPT_equilibrium_constant_relationship_l2000_200047

def given_problem (K1 K2 : ℝ) : Prop :=
  K2 = (1 / K1)^(1 / 2)

theorem equilibrium_constant_relationship (K1 K2 : ℝ) (h : given_problem K1 K2) :
  K1 = 1 / K2^2 :=
by sorry

end NUMINAMATH_GPT_equilibrium_constant_relationship_l2000_200047


namespace NUMINAMATH_GPT_distance_between_points_l2000_200096

theorem distance_between_points :
  let p1 := (3, -5)
  let p2 := (-4, 4)
  dist p1 p2 = Real.sqrt 130 := by
  sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

end NUMINAMATH_GPT_distance_between_points_l2000_200096


namespace NUMINAMATH_GPT_determine_k_l2000_200000

variables (x y z k : ℝ)

theorem determine_k (h1 : (5 / (x - z)) = (k / (y + z))) 
                    (h2 : (k / (y + z)) = (12 / (x + y))) 
                    (h3 : y + z = 2 * x) : 
                    k = 17 := 
by 
  sorry

end NUMINAMATH_GPT_determine_k_l2000_200000


namespace NUMINAMATH_GPT_solve_for_x_l2000_200079

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.07 * (25 + x) = 15.1) : x = 111.25 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2000_200079


namespace NUMINAMATH_GPT_find_P_at_1_l2000_200033

noncomputable def P (x : ℝ) : ℝ := x ^ 2 + x + 1008

theorem find_P_at_1 :
  (∀ x : ℝ, P (P x) - (P x) ^ 2 = x ^ 2 + x + 2016) →
  P 1 = 1010 := by
  intros H
  sorry

end NUMINAMATH_GPT_find_P_at_1_l2000_200033


namespace NUMINAMATH_GPT_books_not_read_l2000_200030

theorem books_not_read (total_books read_books : ℕ) (h1 : total_books = 20) (h2 : read_books = 15) : total_books - read_books = 5 := by
  sorry

end NUMINAMATH_GPT_books_not_read_l2000_200030


namespace NUMINAMATH_GPT_greatest_possible_large_chips_l2000_200068

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end NUMINAMATH_GPT_greatest_possible_large_chips_l2000_200068


namespace NUMINAMATH_GPT_value_of_x_plus_y_l2000_200069

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l2000_200069


namespace NUMINAMATH_GPT_sum_of_money_l2000_200010

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end NUMINAMATH_GPT_sum_of_money_l2000_200010


namespace NUMINAMATH_GPT_students_at_end_of_year_l2000_200051

def n_start := 10
def n_left := 4
def n_new := 42

theorem students_at_end_of_year : n_start - n_left + n_new = 48 := by
  sorry

end NUMINAMATH_GPT_students_at_end_of_year_l2000_200051


namespace NUMINAMATH_GPT_largest_power_of_2_that_divides_n_l2000_200005

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_2_that_divides_n :
  ∃ k : ℕ, 2^k ∣ n ∧ ¬ (2^(k+1) ∣ n) ∧ k = 5 := sorry

end NUMINAMATH_GPT_largest_power_of_2_that_divides_n_l2000_200005


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l2000_200074

theorem algebraic_expression_evaluation
  (x y p q : ℝ)
  (h1 : x + y = 0)
  (h2 : p * q = 1) : (x + y) - 2 * (p * q) = -2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l2000_200074


namespace NUMINAMATH_GPT_penny_initial_money_l2000_200016

theorem penny_initial_money
    (pairs_of_socks : ℕ)
    (cost_per_pair : ℝ)
    (number_of_pairs : ℕ)
    (cost_of_hat : ℝ)
    (money_left : ℝ)
    (initial_money : ℝ)
    (H1 : pairs_of_socks = 4)
    (H2 : cost_per_pair = 2)
    (H3 : number_of_pairs = pairs_of_socks)
    (H4 : cost_of_hat = 7)
    (H5 : money_left = 5)
    (H6 : initial_money = (number_of_pairs * cost_per_pair) + cost_of_hat + money_left) : initial_money = 20 :=
sorry

end NUMINAMATH_GPT_penny_initial_money_l2000_200016


namespace NUMINAMATH_GPT_range_of_a_l2000_200041

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then (a - 5) * x + 8 else 2 * a / x

theorem range_of_a (a : ℝ) : 
  (∀ x y, x < y → f a x ≥ f a y) → (2 ≤ a ∧ a < 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2000_200041


namespace NUMINAMATH_GPT_number_of_female_students_l2000_200073

-- Given conditions
variables (F : ℕ)

-- The average score of all students (90)
def avg_all_students := 90
-- The total number of male students (8)
def num_male_students := 8
-- The average score of male students (87)
def avg_male_students := 87
-- The average score of female students (92)
def avg_female_students := 92

-- We want to prove the following statement
theorem number_of_female_students :
  num_male_students * avg_male_students + F * avg_female_students = (num_male_students + F) * avg_all_students →
  F = 12 :=
sorry

end NUMINAMATH_GPT_number_of_female_students_l2000_200073


namespace NUMINAMATH_GPT_complement_union_l2000_200076

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_union : U \ (A ∪ B) = {4} := by
  sorry

end NUMINAMATH_GPT_complement_union_l2000_200076


namespace NUMINAMATH_GPT_relatively_prime_solutions_l2000_200055

theorem relatively_prime_solutions  (x y : ℤ) (h_rel_prime : gcd x y = 1) : 
  2 * (x^3 - x) = 5 * (y^3 - y) ↔ 
  (x = 0 ∧ (y = 1 ∨ y = -1)) ∨ 
  (x = 1 ∧ y = 0) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 4 ∧ (y = 3 ∨ y = -3)) ∨ 
  (x = -4 ∧ (y = -3 ∨ y = 3)) ∨
  (x = 1 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨
  (x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_GPT_relatively_prime_solutions_l2000_200055


namespace NUMINAMATH_GPT_find_coordinates_of_B_l2000_200065

-- Define the conditions from the problem
def point_A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def point_B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- The proof problem: The coordinates of B are (4, -4)
theorem find_coordinates_of_B (a : ℝ) (h : point_A a = (0, a + 1)) : point_B a = (4, -4) := by
  -- This is skipping the proof part.
  sorry

end NUMINAMATH_GPT_find_coordinates_of_B_l2000_200065


namespace NUMINAMATH_GPT_graphs_intersect_once_l2000_200037

variable {a b c d : ℝ}

theorem graphs_intersect_once 
(h1: ∃ x, (2 * a + 1 / (x - b)) = (2 * c + 1 / (x - d)) ∧ 
∃ y₁ y₂: ℝ, ∀ x, (2 * a + 1 / (x - b)) ≠ 2 * c + 1 / (x - d)) : 
∃ x, ((2 * b + 1 / (x - a)) = (2 * d + 1 / (x - c))) ∧ 
∃ y₁ y₂: ℝ, ∀ x, 2 * b + 1 / (x - a) ≠ 2 * d + 1 / (x - c) := 
sorry

end NUMINAMATH_GPT_graphs_intersect_once_l2000_200037


namespace NUMINAMATH_GPT_solve_system_l2000_200077

theorem solve_system :
  ∃ x y : ℝ, (x^3 + y^3) * (x^2 + y^2) = 64 ∧ x + y = 2 ∧ 
  ((x = 1 + Real.sqrt (5 / 3) ∧ y = 1 - Real.sqrt (5 / 3)) ∨ 
   (x = 1 - Real.sqrt (5 / 3) ∧ y = 1 + Real.sqrt (5 / 3))) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2000_200077


namespace NUMINAMATH_GPT_tan_theta_value_l2000_200056

noncomputable def tan_theta (θ : ℝ) : ℝ :=
  if (0 < θ) ∧ (θ < 2 * Real.pi) ∧ (Real.cos (θ / 2) = 1 / 3) then
    (2 * (2 * Real.sqrt 2) / (1 - (2 * Real.sqrt 2) ^ 2))
  else
    0 -- added default value for well-definedness

theorem tan_theta_value (θ : ℝ) (h₀: 0 < θ) (h₁ : θ < 2 * Real.pi) (h₂ : Real.cos (θ / 2) = 1 / 3) : 
  tan_theta θ = -4 * Real.sqrt 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_value_l2000_200056


namespace NUMINAMATH_GPT_average_monthly_income_is_2125_l2000_200093

noncomputable def calculate_average_monthly_income (expenses_3_months: ℕ) (expenses_4_months: ℕ) (expenses_5_months: ℕ) (savings_per_year: ℕ) : ℕ :=
  (expenses_3_months * 3 + expenses_4_months * 4 + expenses_5_months * 5 + savings_per_year) / 12

theorem average_monthly_income_is_2125 :
  calculate_average_monthly_income 1700 1550 1800 5200 = 2125 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_income_is_2125_l2000_200093


namespace NUMINAMATH_GPT_fraction_to_decimal_l2000_200066

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2000_200066


namespace NUMINAMATH_GPT_washer_total_cost_l2000_200013

variable (C : ℝ)
variable (h : 0.25 * C = 200)

theorem washer_total_cost : C = 800 :=
by
  sorry

end NUMINAMATH_GPT_washer_total_cost_l2000_200013


namespace NUMINAMATH_GPT_binomial_19_10_l2000_200057

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end NUMINAMATH_GPT_binomial_19_10_l2000_200057


namespace NUMINAMATH_GPT_find_point_A_l2000_200009

theorem find_point_A :
  (∃ A : ℤ, A + 2 = -2) ∨ (∃ A : ℤ, A - 2 = -2) → (∃ A : ℤ, A = 0 ∨ A = -4) :=
by
  sorry

end NUMINAMATH_GPT_find_point_A_l2000_200009


namespace NUMINAMATH_GPT_passenger_drop_ratio_l2000_200075

theorem passenger_drop_ratio (initial_passengers passengers_at_first passengers_at_second final_passengers x : ℕ)
  (h0 : initial_passengers = 288)
  (h1 : passengers_at_first = initial_passengers - (initial_passengers / 3) + 280)
  (h2 : passengers_at_second = passengers_at_first - x + 12)
  (h3 : final_passengers = 248)
  (h4 : passengers_at_second = final_passengers) :
  x / passengers_at_first = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_passenger_drop_ratio_l2000_200075


namespace NUMINAMATH_GPT_min_value_condition_l2000_200094

theorem min_value_condition 
  (a b : ℝ) 
  (h1 : 4 * a + b = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 1 - 4 * x → x = 16) := 
sorry

end NUMINAMATH_GPT_min_value_condition_l2000_200094


namespace NUMINAMATH_GPT_correct_omega_l2000_200017

theorem correct_omega (Ω : ℕ) (h : Ω * Ω = 2 * 2 * 2 * 2 * 3 * 3) : Ω = 2 * 2 * 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_omega_l2000_200017


namespace NUMINAMATH_GPT_triangles_fit_in_pan_l2000_200098

theorem triangles_fit_in_pan (pan_length pan_width triangle_base triangle_height : ℝ)
  (h1 : pan_length = 15) (h2 : pan_width = 24) (h3 : triangle_base = 3) (h4 : triangle_height = 4) :
  (pan_length * pan_width) / (1/2 * triangle_base * triangle_height) = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangles_fit_in_pan_l2000_200098


namespace NUMINAMATH_GPT_graphs_intersect_at_one_point_l2000_200039

theorem graphs_intersect_at_one_point (a : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 3 * x + 1 = -x - 1) ↔ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_graphs_intersect_at_one_point_l2000_200039


namespace NUMINAMATH_GPT_number_of_solutions_eq_six_l2000_200018

/-- 
The number of ordered pairs (m, n) of positive integers satisfying the equation
6/m + 3/n = 1 is 6.
-/
theorem number_of_solutions_eq_six : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, (1 < p.1 ∧ 1 < p.2) ∧ 6 / p.1 + 3 / p.2 = 1) ∧ s.card = 6 :=
sorry

end NUMINAMATH_GPT_number_of_solutions_eq_six_l2000_200018


namespace NUMINAMATH_GPT_volume_between_spheres_l2000_200050

theorem volume_between_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  (4 / 3) * Real.pi * (r_large ^ 3) - (4 / 3) * Real.pi * (r_small ^ 3) = (1792 / 3) * Real.pi := 
by
  rw [h_small, h_large]
  sorry

end NUMINAMATH_GPT_volume_between_spheres_l2000_200050


namespace NUMINAMATH_GPT_set_membership_proof_l2000_200031

variable (A : Set ℕ) (B : Set (Set ℕ))

theorem set_membership_proof :
  A = {0, 1} → B = {x | x ⊆ A} → A ∈ B :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_set_membership_proof_l2000_200031


namespace NUMINAMATH_GPT_proportion_of_mothers_full_time_jobs_l2000_200064

theorem proportion_of_mothers_full_time_jobs
  (P : ℝ) (W : ℝ) (F : ℝ → Prop) (M : ℝ)
  (hwomen : W = 0.4 * P)
  (hfathers_full_time : ∀ p, F p → p = 0.75)
  (hno_full_time : P - (W + 0.75 * (P - W)) = 0.19 * P) :
  M = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_mothers_full_time_jobs_l2000_200064


namespace NUMINAMATH_GPT_division_of_cubics_l2000_200095

theorem division_of_cubics (c d : ℕ) (h1 : c = 7) (h2 : d = 3) : 
  (c^3 + d^3) / (c^2 - c * d + d^2) = 10 := by
  sorry

end NUMINAMATH_GPT_division_of_cubics_l2000_200095


namespace NUMINAMATH_GPT_positive_integer_solution_inequality_l2000_200045

theorem positive_integer_solution_inequality (x : ℕ) (h : 2 * (x + 1) ≥ 5 * x - 3) : x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integer_solution_inequality_l2000_200045


namespace NUMINAMATH_GPT_decreased_value_l2000_200023

noncomputable def original_expression (x y: ℝ) : ℝ :=
  x * y^2

noncomputable def decreased_expression (x y: ℝ) : ℝ :=
  (1 / 2) * x * (1 / 2 * y) ^ 2

theorem decreased_value (x y: ℝ) :
  decreased_expression x y = (1 / 8) * original_expression x y :=
by
  sorry

end NUMINAMATH_GPT_decreased_value_l2000_200023


namespace NUMINAMATH_GPT_grace_charges_for_pulling_weeds_l2000_200042

theorem grace_charges_for_pulling_weeds :
  (∃ (W : ℕ ), 63 * 6 + 9 * W + 10 * 9 = 567 → W = 11) :=
by
  use 11
  intro h
  sorry

end NUMINAMATH_GPT_grace_charges_for_pulling_weeds_l2000_200042


namespace NUMINAMATH_GPT_moles_of_H2O_formed_l2000_200034

theorem moles_of_H2O_formed (moles_NH4NO3 moles_NaOH : ℕ) (percent_NaOH_reacts : ℝ)
  (h_decomposition : moles_NH4NO3 = 2) (h_NaOH : moles_NaOH = 2) 
  (h_percent : percent_NaOH_reacts = 0.85) : 
  (moles_NaOH * percent_NaOH_reacts = 1.7) :=
by
  sorry

end NUMINAMATH_GPT_moles_of_H2O_formed_l2000_200034


namespace NUMINAMATH_GPT_odd_natural_of_form_l2000_200004

/-- 
  Prove that the only odd natural number n in the form (p + q) / (p - q)
  where p and q are prime numbers and p > q is 5.
-/
theorem odd_natural_of_form (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p > q) 
  (h2 : ∃ n : ℕ, n = (p + q) / (p - q) ∧ n % 2 = 1) : ∃ n : ℕ, n = 5 :=
sorry

end NUMINAMATH_GPT_odd_natural_of_form_l2000_200004


namespace NUMINAMATH_GPT_proportional_function_decreases_l2000_200083

-- Define the function y = -2x
def proportional_function (x : ℝ) : ℝ := -2 * x

-- State the theorem to prove that y decreases as x increases
theorem proportional_function_decreases (x y : ℝ) (h : y = proportional_function x) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → proportional_function x₁ > proportional_function x₂ := 
sorry

end NUMINAMATH_GPT_proportional_function_decreases_l2000_200083


namespace NUMINAMATH_GPT_total_surface_area_correct_l2000_200028

-- Definitions for side lengths of the cubes
def side_length_large := 5
def side_length_medium := 2
def side_length_small := 1

-- Surface area calculation for a single cube
def surface_area (side_length : ℕ) : ℕ := 6 * side_length^2

-- Surface areas for each size of the cube
def surface_area_large := surface_area side_length_large
def surface_area_medium := surface_area side_length_medium
def surface_area_small := surface_area side_length_small

-- Total surface areas for medium and small cubes
def surface_area_medium_total := 4 * surface_area_medium
def surface_area_small_total := 4 * surface_area_small

-- Total surface area of the structure
def total_surface_area := surface_area_large + surface_area_medium_total + surface_area_small_total

-- Expected result
def expected_surface_area := 270

-- Proof statement
theorem total_surface_area_correct : total_surface_area = expected_surface_area := by
  sorry

end NUMINAMATH_GPT_total_surface_area_correct_l2000_200028


namespace NUMINAMATH_GPT_proof_l2000_200086

-- Define the universal set U.
def U : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set M.
def M : Set ℕ := {1, 2, 3}

-- Define set N.
def N : Set ℕ := {3, 4, 5, 6}

-- The complement of M with respect to U.
def compl_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- The intersection of complement of M and N.
def result : Set ℕ := compl_U_M ∩ N

-- The theorem to be proven.
theorem proof : result = {4, 5, 6} := by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_proof_l2000_200086


namespace NUMINAMATH_GPT_problem_solution_l2000_200003

variable (a : ℝ)

theorem problem_solution (h : a ≠ 0) : a^2 + 1 > 1 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2000_200003


namespace NUMINAMATH_GPT_sequence_diff_exists_l2000_200091

theorem sequence_diff_exists (x : ℕ → ℕ) (h1 : x 1 = 1) (h2 : ∀ n : ℕ, 1 ≤ n → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end NUMINAMATH_GPT_sequence_diff_exists_l2000_200091


namespace NUMINAMATH_GPT_product_in_third_quadrant_l2000_200036

def z1 : ℂ := 1 - 3 * Complex.I
def z2 : ℂ := 3 - 2 * Complex.I
def z := z1 * z2

theorem product_in_third_quadrant : z.re < 0 ∧ z.im < 0 := 
sorry

end NUMINAMATH_GPT_product_in_third_quadrant_l2000_200036


namespace NUMINAMATH_GPT_simple_random_sampling_methods_proof_l2000_200020

-- Definitions based on conditions
def equal_probability (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
∀ s1 s2 : samples, p s1 = p s2

-- Define that Lottery Drawing Method and Random Number Table Method are part of simple random sampling
def is_lottery_drawing_method (samples : Type) : Prop := sorry
def is_random_number_table_method (samples : Type) : Prop := sorry

def simple_random_sampling_methods (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
  equal_probability samples p ∧ is_lottery_drawing_method samples ∧ is_random_number_table_method samples

-- Statement to be proven
theorem simple_random_sampling_methods_proof (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) :
  (∀ s1 s2 : samples, p s1 = p s2) → simple_random_sampling_methods samples p :=
by
  intro h
  unfold simple_random_sampling_methods
  constructor
  exact h
  constructor
  sorry -- Proof for is_lottery_drawing_method
  sorry -- Proof for is_random_number_table_method

end NUMINAMATH_GPT_simple_random_sampling_methods_proof_l2000_200020


namespace NUMINAMATH_GPT_no_hexagonal_pyramid_with_equal_edges_l2000_200038

theorem no_hexagonal_pyramid_with_equal_edges (edges : ℕ → ℝ)
  (regular_polygon : ℕ → ℝ → Prop)
  (equal_length_edges : ∀ (n : ℕ), regular_polygon n (edges n) → ∀ i j, edges i = edges j)
  (apex_above_centroid : ∀ (n : ℕ) (h : regular_polygon n (edges n)), True) :
  ¬ regular_polygon 6 (edges 6) :=
by
  sorry

end NUMINAMATH_GPT_no_hexagonal_pyramid_with_equal_edges_l2000_200038


namespace NUMINAMATH_GPT_original_wage_before_increase_l2000_200071

theorem original_wage_before_increase (W : ℝ) 
  (h1 : W * 1.4 = 35) : W = 25 := by
  sorry

end NUMINAMATH_GPT_original_wage_before_increase_l2000_200071


namespace NUMINAMATH_GPT_domain_of_f_2x_plus_1_l2000_200043

theorem domain_of_f_2x_plus_1 {f : ℝ → ℝ} :
  (∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 3 → (-3 : ℝ) ≤ x - 1 ∧ x - 1 ≤ 2) →
  (∀ x, (-3 : ℝ) ≤ x ∧ x ≤ 2 → (-2 : ℝ) ≤ (x : ℝ) ∧ x ≤ 1/2) →
  ∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 1 / 2 → ∀ y, y = 2 * x + 1 → (-3 : ℝ) ≤ y ∧ y ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_2x_plus_1_l2000_200043


namespace NUMINAMATH_GPT_max_value_k_eq_1_range_k_no_zeros_l2000_200090

-- Define the function f(x)
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Note: 'by' and 'sorry' are placeholders to skip the proof; actual proofs are not required.

-- Proof Problem 1: Prove that when k = 1, the maximum value of f(x) is 0.
theorem max_value_k_eq_1 : ∀ x : ℝ, 1 < x → f x 1 ≤ 0 := 
by
  sorry

-- Proof Problem 2: Prove that k ∈ (1, +∞) is the range such that f(x) has no zeros.
theorem range_k_no_zeros : ∀ k : ℝ, (∀ x : ℝ, 1 < x → f x k ≠ 0) → 1 < k :=
by
  sorry

end NUMINAMATH_GPT_max_value_k_eq_1_range_k_no_zeros_l2000_200090


namespace NUMINAMATH_GPT_combined_difference_is_correct_l2000_200022

-- Define the number of cookies each person has
def alyssa_cookies : Nat := 129
def aiyanna_cookies : Nat := 140
def carl_cookies : Nat := 167

-- Define the differences between each pair of people's cookies
def diff_alyssa_aiyanna : Nat := aiyanna_cookies - alyssa_cookies
def diff_alyssa_carl : Nat := carl_cookies - alyssa_cookies
def diff_aiyanna_carl : Nat := carl_cookies - aiyanna_cookies

-- Define the combined difference
def combined_difference : Nat := diff_alyssa_aiyanna + diff_alyssa_carl + diff_aiyanna_carl

-- State the theorem to be proved
theorem combined_difference_is_correct : combined_difference = 76 := by
  sorry

end NUMINAMATH_GPT_combined_difference_is_correct_l2000_200022


namespace NUMINAMATH_GPT_find_a_plus_c_l2000_200099

theorem find_a_plus_c (a b c d : ℝ) (h1 : ab + bc + cd + da = 40) (h2 : b + d = 8) : a + c = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_c_l2000_200099


namespace NUMINAMATH_GPT_cost_of_math_book_l2000_200078

-- The definitions based on the conditions from the problem
def total_books : ℕ := 90
def math_books : ℕ := 54
def history_books := total_books - math_books -- 36
def cost_history_book : ℝ := 5
def total_cost : ℝ := 396

-- The theorem we want to prove: the cost of each math book
theorem cost_of_math_book (M : ℝ) : (math_books * M + history_books * cost_history_book = total_cost) → M = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_math_book_l2000_200078


namespace NUMINAMATH_GPT_distinct_paths_l2000_200046

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem distinct_paths (right_steps up_steps : ℕ) : right_steps = 7 → up_steps = 3 →
  binom (right_steps + up_steps) up_steps = 120 := 
by
  intros h1 h2
  rw [h1, h2]
  unfold binom
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_distinct_paths_l2000_200046


namespace NUMINAMATH_GPT_product_of_x_values_product_of_all_possible_x_values_l2000_200072

theorem product_of_x_values (x : ℚ) (h : abs ((18 : ℚ) / x - 4) = 3) :
  x = 18 ∨ x = 18 / 7 :=
sorry

theorem product_of_all_possible_x_values (x1 x2 : ℚ) (h1 : abs ((18 : ℚ) / x1 - 4) = 3) (h2 : abs ((18 : ℚ) / x2 - 4) = 3) :
  x1 * x2 = 324 / 7 :=
sorry

end NUMINAMATH_GPT_product_of_x_values_product_of_all_possible_x_values_l2000_200072


namespace NUMINAMATH_GPT_red_beads_count_is_90_l2000_200084

-- Define the arithmetic sequence for red beads
def red_bead_count (n : ℕ) : ℕ := 2 * n

-- The sum of the first n terms in our sequence
def sum_red_beads (n : ℕ) : ℕ := n * (n + 1)

-- Verify the number of terms n such that the sum of red beads remains under 100
def valid_num_terms : ℕ := Nat.sqrt 99

-- Calculate total number of red beads on the necklace
def total_red_beads : ℕ := sum_red_beads valid_num_terms

theorem red_beads_count_is_90 (num_beads : ℕ) (valid : num_beads = 99) : 
  total_red_beads = 90 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_red_beads_count_is_90_l2000_200084


namespace NUMINAMATH_GPT_george_initial_socks_l2000_200044

theorem george_initial_socks (S : ℕ) (h : S - 4 + 36 = 60) : S = 28 :=
by
  sorry

end NUMINAMATH_GPT_george_initial_socks_l2000_200044


namespace NUMINAMATH_GPT_triangle_cut_20_sided_polygon_l2000_200029

-- Definitions based on the conditions
def is_triangle (T : Type) : Prop := ∃ (a b c : ℝ), a + b + c = 180 

def can_form_20_sided_polygon (pieces : List (ℝ × ℝ)) : Prop := pieces.length = 20

-- Theorem statement
theorem triangle_cut_20_sided_polygon (T : Type) (P1 P2 : (ℝ × ℝ)) :
  is_triangle T → 
  (P1 ≠ P2) → 
  can_form_20_sided_polygon [P1, P2] :=
sorry

end NUMINAMATH_GPT_triangle_cut_20_sided_polygon_l2000_200029


namespace NUMINAMATH_GPT_linear_function_diff_l2000_200092

noncomputable def g : ℝ → ℝ := sorry

theorem linear_function_diff (h_linear : ∀ x y z w : ℝ, (g y - g x) / (y - x) = (g w - g z) / (w - z))
                            (h_condition : g 8 - g 1 = 21) : 
  g 16 - g 1 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_linear_function_diff_l2000_200092


namespace NUMINAMATH_GPT_tangent_line_at_one_l2000_200089

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one : ∀ (x y : ℝ), y = 2 * Real.exp 1 * x - Real.exp 1 → 
  ∃ m b : ℝ, (∀ x: ℝ, f x = m * x + b) ∧ (m = 2 * Real.exp 1) ∧ (b = -Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_l2000_200089


namespace NUMINAMATH_GPT_min_value_inequality_l2000_200067

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3 * a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 25 ∧ (∀ x y, (x > 0) → (y > 0) → (3 * x + 2 * y = 1) → (3 / x + 2 / y) ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l2000_200067


namespace NUMINAMATH_GPT_max_x_on_circle_l2000_200059

theorem max_x_on_circle : 
  ∀ x y : ℝ,
  (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_max_x_on_circle_l2000_200059


namespace NUMINAMATH_GPT_sum_xyz_l2000_200081

theorem sum_xyz (x y z : ℝ) (h1 : x + y = 1) (h2 : y + z = 1) (h3 : z + x = 1) : x + y + z = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_sum_xyz_l2000_200081


namespace NUMINAMATH_GPT_base_b_addition_correct_base_b_l2000_200035

theorem base_b_addition (b : ℕ) (hb : b > 5) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 :=
  by
    sorry

theorem correct_base_b : ∃ (b : ℕ), b > 5 ∧ 
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 ∧
  (4 + 5 = b + 1) ∧
  (2 + 1 + 1 = 4) :=
  ⟨8, 
   by decide,
   base_b_addition 8 (by decide),
   by decide,
   by decide⟩ 

end NUMINAMATH_GPT_base_b_addition_correct_base_b_l2000_200035


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2000_200021

theorem repeating_decimal_to_fraction :
  7.4646464646 = (739 / 99) :=
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2000_200021


namespace NUMINAMATH_GPT_farmer_flax_acres_l2000_200053

-- Definitions based on conditions
def total_acres : ℕ := 240
def extra_sunflower_acres : ℕ := 80

-- Problem statement
theorem farmer_flax_acres (F : ℕ) (S : ℕ) 
    (h1 : F + S = total_acres) 
    (h2 : S = F + extra_sunflower_acres) : 
    F = 80 :=
by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_farmer_flax_acres_l2000_200053


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l2000_200054

theorem min_value_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l2000_200054


namespace NUMINAMATH_GPT_exponential_function_pass_through_point_l2000_200088

theorem exponential_function_pass_through_point
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a^(1 - 1) + 1 = 2) :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_pass_through_point_l2000_200088


namespace NUMINAMATH_GPT_probability_of_two_in_decimal_rep_of_eight_over_eleven_l2000_200063

theorem probability_of_two_in_decimal_rep_of_eight_over_eleven : 
  (∃ B : List ℕ, (B = [7, 2]) ∧ (1 = (B.count 2) / (B.length)) ∧ 
  (0 + B.sum + 1) / 11 = 8 / 11) := sorry

end NUMINAMATH_GPT_probability_of_two_in_decimal_rep_of_eight_over_eleven_l2000_200063


namespace NUMINAMATH_GPT_empty_vessel_mass_l2000_200012

theorem empty_vessel_mass
  (m1 : ℝ) (m2 : ℝ) (rho_K : ℝ) (rho_B : ℝ) (V : ℝ) (m_c : ℝ)
  (h1 : m1 = m_c + rho_K * V)
  (h2 : m2 = m_c + rho_B * V)
  (h_mass_kerosene : m1 = 31)
  (h_mass_water : m2 = 33)
  (h_rho_K : rho_K = 800)
  (h_rho_B : rho_B = 1000) :
  m_c = 23 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_empty_vessel_mass_l2000_200012
