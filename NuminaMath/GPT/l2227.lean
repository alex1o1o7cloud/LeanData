import Mathlib

namespace base_k_conversion_l2227_222716

theorem base_k_conversion (k : ℕ) (hk : 4 * k + 4 = 36) : 6 * 8 + 7 = 55 :=
by
  -- Proof skipped
  sorry

end base_k_conversion_l2227_222716


namespace x_plus_y_equals_two_l2227_222725

variable (x y : ℝ)

def condition1 : Prop := (x - 1) ^ 2017 + 2013 * (x - 1) = -1
def condition2 : Prop := (y - 1) ^ 2017 + 2013 * (y - 1) = 1

theorem x_plus_y_equals_two (h1 : condition1 x) (h2 : condition2 y) : x + y = 2 :=
  sorry

end x_plus_y_equals_two_l2227_222725


namespace lucas_change_l2227_222740

-- Define the costs of items and the initial amount.
def initial_amount : ℝ := 20.00
def cost_avocados : ℝ := 1.50 + 2.25 + 3.00
def cost_water : ℝ := 2 * 1.75
def cost_apples : ℝ := 4 * 0.75

-- Define the total cost.
def total_cost : ℝ := cost_avocados + cost_water + cost_apples

-- Define the expected change.
def expected_change : ℝ := initial_amount - total_cost

-- The proposition (statement) we want to prove.
theorem lucas_change : expected_change = 6.75 :=
by
  sorry -- Proof to be completed.

end lucas_change_l2227_222740


namespace tape_mounting_cost_correct_l2227_222784

-- Define the given conditions as Lean definitions
def os_overhead_cost : ℝ := 1.07
def cost_per_millisecond : ℝ := 0.023
def total_cost : ℝ := 40.92
def runtime_seconds : ℝ := 1.5

-- Define the required target cost for mounting a data tape
def cost_of_data_tape : ℝ := 5.35

-- Prove that the cost of mounting a data tape is correct given the conditions
theorem tape_mounting_cost_correct :
  let computer_time_cost := cost_per_millisecond * (runtime_seconds * 1000)
  let total_cost_computed := os_overhead_cost + computer_time_cost
  cost_of_data_tape = total_cost - total_cost_computed := by
{
  sorry
}

end tape_mounting_cost_correct_l2227_222784


namespace quadratic_root_a_value_l2227_222717

theorem quadratic_root_a_value (a : ℝ) (h : 2^2 - 2 * a + 6 = 0) : a = 5 :=
sorry

end quadratic_root_a_value_l2227_222717


namespace arithmetic_sequence_divisible_by_2005_l2227_222701

-- Problem Statement
theorem arithmetic_sequence_divisible_by_2005
  (a : ℕ → ℕ) -- Define the arithmetic sequence
  (d : ℕ) -- Common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence condition
  (h_product_div_2005 : ∀ n, 2005 ∣ (a n) * (a (n + 31))) -- Given condition on product divisibility
  : ∀ n, 2005 ∣ a n := 
sorry

end arithmetic_sequence_divisible_by_2005_l2227_222701


namespace negation_of_every_function_has_parity_l2227_222703

-- Assume the initial proposition
def every_function_has_parity := ∀ f : ℕ → ℕ, ∃ (p : ℕ), p = 0 ∨ p = 1

-- Negation of the original proposition
def exists_function_without_parity := ∃ f : ℕ → ℕ, ∀ p : ℕ, p ≠ 0 ∧ p ≠ 1

-- The theorem to prove
theorem negation_of_every_function_has_parity : 
  ¬ every_function_has_parity ↔ exists_function_without_parity := 
by
  unfold every_function_has_parity exists_function_without_parity
  sorry

end negation_of_every_function_has_parity_l2227_222703


namespace twentieth_fisherman_caught_l2227_222799

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l2227_222799


namespace number_of_pictures_in_first_coloring_book_l2227_222707

-- Define the conditions
variable (X : ℕ)
variable (total_pictures_colored : ℕ := 44)
variable (pictures_left : ℕ := 11)
variable (pictures_in_second_coloring_book : ℕ := 32)
variable (total_pictures : ℕ := total_pictures_colored + pictures_left)

-- The theorem statement
theorem number_of_pictures_in_first_coloring_book :
  X + pictures_in_second_coloring_book = total_pictures → X = 23 :=
by
  intro h
  sorry

end number_of_pictures_in_first_coloring_book_l2227_222707


namespace part1_part2_l2227_222785

theorem part1 (a m n : ℕ) (ha : a > 1) (hdiv : a^m + 1 ∣ a^n + 1) : n ∣ m :=
sorry

theorem part2 (a b m n : ℕ) (ha : a > 1) (coprime_ab : Nat.gcd a b = 1) (hdiv : a^m + b^m ∣ a^n + b^n) : n ∣ m :=
sorry

end part1_part2_l2227_222785


namespace symmetric_trapezoid_construction_possible_l2227_222706

-- Define lengths of legs and distance from intersection point
variables (a b : ℝ)

-- Symmetric trapezoid feasibility condition
theorem symmetric_trapezoid_construction_possible : 3 * b > 2 * a := sorry

end symmetric_trapezoid_construction_possible_l2227_222706


namespace binary_divisible_by_136_l2227_222766

theorem binary_divisible_by_136 :
  let N := 2^139 + 2^105 + 2^15 + 2^13
  N % 136 = 0 :=
by {
  let N := 2^139 + 2^105 + 2^15 + 2^13;
  sorry
}

end binary_divisible_by_136_l2227_222766


namespace find_number_of_olives_l2227_222733

theorem find_number_of_olives (O : ℕ)
  (lettuce_choices : 2 = 2)
  (tomato_choices : 3 = 3)
  (soup_choices : 2 = 2)
  (total_combos : 2 * 3 * O * 2 = 48) :
  O = 4 :=
by
  sorry

end find_number_of_olives_l2227_222733


namespace count_integers_satisfying_inequality_l2227_222776

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end count_integers_satisfying_inequality_l2227_222776


namespace value_ranges_l2227_222768

theorem value_ranges 
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_eq1 : 3 * a + 2 * b + c = 5)
  (h_eq2 : 2 * a + b - 3 * c = 1) :
  (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧ 
  (-5 / 7 ≤ (3 * a + b - 7 * c) ∧ (3 * a + b - 7 * c) ≤ -1 / 11) :=
by 
  sorry

end value_ranges_l2227_222768


namespace mixed_number_expression_l2227_222739

open Real

-- Definitions of the given mixed numbers
def mixed_number1 : ℚ := (37 / 7)
def mixed_number2 : ℚ := (18 / 5)
def mixed_number3 : ℚ := (19 / 6)
def mixed_number4 : ℚ := (9 / 4)

-- Main theorem statement
theorem mixed_number_expression :
  25 * (mixed_number1 - mixed_number2) / (mixed_number3 + mixed_number4) = 7 + 49 / 91 :=
by
  sorry

end mixed_number_expression_l2227_222739


namespace average_of_original_set_l2227_222704

theorem average_of_original_set (A : ℝ) (h1 : (35 * A) = (7 * 75)) : A = 15 := 
by sorry

end average_of_original_set_l2227_222704


namespace total_distance_covered_l2227_222729

theorem total_distance_covered :
  let t1 := 30 / 60 -- time in hours for first walking session
  let s1 := 3       -- speed in mph for first walking session
  let t2 := 20 / 60 -- time in hours for running session
  let s2 := 8       -- speed in mph for running session
  let t3 := 10 / 60 -- time in hours for second walking session
  let s3 := 2       -- speed in mph for second walking session
  let d1 := s1 * t1 -- distance for first walking session
  let d2 := s2 * t2 -- distance for running session
  let d3 := s3 * t3 -- distance for second walking session
  d1 + d2 + d3 = 4.5 :=
by
  sorry

end total_distance_covered_l2227_222729


namespace plane_equation_l2227_222788

theorem plane_equation 
  (s t : ℝ)
  (x y z : ℝ)
  (parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ)
  (plane_eq : ℝ × ℝ × ℝ → Prop) :
  parametric_plane s t = (2 + 2 * s - t, 1 + 2 * s, 4 - 3 * s + t) →
  plane_eq (x, y, z) ↔ 2 * x - 5 * y + 2 * z - 7 = 0 :=
by
  sorry

end plane_equation_l2227_222788


namespace range_of_q_l2227_222756

variable (x : ℝ)

def q (x : ℝ) := (3 * x^2 + 1)^2

theorem range_of_q : ∀ y, (∃ x : ℝ, x ≥ 0 ∧ y = q x) ↔ y ≥ 1 := by
  sorry

end range_of_q_l2227_222756


namespace school_anniversary_problem_l2227_222751

theorem school_anniversary_problem
    (total_cost : ℕ)
    (cost_commemorative_albums cost_bone_china_cups : ℕ)
    (num_commemorative_albums num_bone_china_cups : ℕ)
    (price_commemorative_album price_bone_china_cup : ℕ)
    (H1 : total_cost = 312000)
    (H2 : cost_commemorative_albums + cost_bone_china_cups = total_cost)
    (H3 : cost_commemorative_albums = 3 * cost_bone_china_cups)
    (H4 : price_commemorative_album = 3 / 2 * price_bone_china_cup)
    (H5 : num_bone_china_cups = 4 * num_commemorative_albums + 1600) :
    (cost_commemorative_albums = 72000 ∧ cost_bone_china_cups = 240000) ∧
    (price_commemorative_album = 45 ∧ price_bone_china_cup = 30) :=
by
  sorry

end school_anniversary_problem_l2227_222751


namespace number_of_parents_l2227_222713

theorem number_of_parents (n m : ℕ) 
  (h1 : n + m = 31) 
  (h2 : 15 + m = n) 
  : n = 23 := 
by 
  sorry

end number_of_parents_l2227_222713


namespace min_flash_drives_needed_l2227_222721

theorem min_flash_drives_needed (total_files : ℕ) (capacity_per_drive : ℝ)  
  (num_files_0_9 : ℕ) (size_0_9 : ℝ) 
  (num_files_0_8 : ℕ) (size_0_8 : ℝ) 
  (size_0_6 : ℝ) 
  (remaining_files : ℕ) :
  total_files = 40 →
  capacity_per_drive = 2.88 →
  num_files_0_9 = 5 →
  size_0_9 = 0.9 →
  num_files_0_8 = 18 →
  size_0_8 = 0.8 →
  remaining_files = total_files - (num_files_0_9 + num_files_0_8) →
  size_0_6 = 0.6 →
  (num_files_0_9 * size_0_9 + num_files_0_8 * size_0_8 + remaining_files * size_0_6) / capacity_per_drive ≤ 13 :=
by {
  sorry
}

end min_flash_drives_needed_l2227_222721


namespace simple_interest_rate_l2227_222752

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ) : 
  T = 6 → I = (7/6) * P - P → I = P * R * T / 100 → R = 100 / 36 :=
by
  intros T_eq I_eq simple_interest_eq
  sorry

end simple_interest_rate_l2227_222752


namespace solution_of_phi_l2227_222782

theorem solution_of_phi 
    (φ : ℝ) 
    (H : ∃ k : ℤ, 2 * (π / 6) + φ = k * π) :
    φ = - (π / 3) := 
sorry

end solution_of_phi_l2227_222782


namespace solve_for_y_l2227_222779

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l2227_222779


namespace kibble_recommendation_difference_l2227_222797

theorem kibble_recommendation_difference :
  (0.2 * 1000 : ℝ) < (0.3 * 1000) ∧ ((0.3 * 1000) - (0.2 * 1000)) = 100 :=
by
  sorry

end kibble_recommendation_difference_l2227_222797


namespace students_like_both_l2227_222796

theorem students_like_both (total_students French_fries_likers burger_likers neither_likers : ℕ)
(H1 : total_students = 25)
(H2 : French_fries_likers = 15)
(H3 : burger_likers = 10)
(H4 : neither_likers = 6)
: (French_fries_likers + burger_likers + neither_likers - total_students) = 12 :=
by sorry

end students_like_both_l2227_222796


namespace proof_problem_l2227_222777

-- Definitions
variable (T : Type) (Sam : T)
variable (solves_all : T → Prop) (passes : T → Prop)

-- Given condition (Dr. Evans's statement)
axiom dr_evans_statement : ∀ x : T, solves_all x → passes x

-- Statement to be proven
theorem proof_problem : ¬ (passes Sam) → ¬ (solves_all Sam) :=
  by sorry

end proof_problem_l2227_222777


namespace total_stamps_received_l2227_222798

theorem total_stamps_received
  (initial_stamps : ℕ)
  (final_stamps : ℕ)
  (received_stamps : ℕ)
  (h_initial : initial_stamps = 34)
  (h_final : final_stamps = 61)
  (h_received : received_stamps = final_stamps - initial_stamps) :
  received_stamps = 27 :=
by 
  sorry

end total_stamps_received_l2227_222798


namespace train_pass_time_l2227_222737

def train_length : ℕ := 250
def train_speed_kmph : ℕ := 36
def station_length : ℕ := 200

def total_distance : ℕ := train_length + station_length

noncomputable def train_speed_mps : ℚ := (train_speed_kmph : ℚ) * 1000 / 3600

noncomputable def time_to_pass_station : ℚ := total_distance / train_speed_mps

theorem train_pass_time : time_to_pass_station = 45 := by
  sorry

end train_pass_time_l2227_222737


namespace worth_of_presents_l2227_222755

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l2227_222755


namespace range_of_a_l2227_222774

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x a ≥ a) ↔ -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 := 
sorry

end range_of_a_l2227_222774


namespace smallest_N_is_14_l2227_222744

-- Definition of depicted number and cyclic arrangement
def depicted_number : Type := List (Fin 2) -- Depicted numbers are lists of digits (0 corresponds to 1, 1 corresponds to 2)

-- A condition representing the function that checks if a list contains all possible four-digit combinations
def contains_all_four_digit_combinations (arr: List (Fin 2)) : Prop :=
  ∀ (seq: List (Fin 2)), seq.length = 4 → seq ⊆ arr

-- The problem statement: find the smallest N where an arrangement contains all four-digit combinations
def smallest_N (N: Nat) (arr: List (Fin 2)) : Prop :=
  N = arr.length ∧ contains_all_four_digit_combinations arr

theorem smallest_N_is_14 : ∃ (N : Nat) (arr: List (Fin 2)), smallest_N N arr ∧ N = 14 :=
by
  -- Placeholder for the proof
  sorry

end smallest_N_is_14_l2227_222744


namespace machine_a_sprockets_per_hour_l2227_222747

theorem machine_a_sprockets_per_hour (s h : ℝ)
    (H1 : 1.1 * s * h = 550)
    (H2 : s * (h + 10) = 550) : s = 5 := by
  sorry

end machine_a_sprockets_per_hour_l2227_222747


namespace find_C_plus_D_l2227_222726

theorem find_C_plus_D
  (C D : ℕ)
  (h1 : D = C + 2)
  (h2 : 2 * C^2 + 5 * C + 3 - (7 * D + 5) = (C + D)^2 + 6 * (C + D) + 8)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D) :
  C + D = 26 := by
  sorry

end find_C_plus_D_l2227_222726


namespace remaining_perimeter_l2227_222787

-- Definitions based on conditions
noncomputable def GH : ℝ := 2
noncomputable def HI : ℝ := 2
noncomputable def GI : ℝ := Real.sqrt (GH^2 + HI^2)
noncomputable def side_JKL : ℝ := 5
noncomputable def JI : ℝ := side_JKL - GH
noncomputable def IK : ℝ := side_JKL - HI
noncomputable def JK : ℝ := side_JKL

-- Problem statement in Lean 4
theorem remaining_perimeter :
  JI + IK + JK = 11 :=
by
  sorry

end remaining_perimeter_l2227_222787


namespace common_ratio_l2227_222793

theorem common_ratio (a_3 S_3 : ℝ) (q : ℝ) 
  (h1 : a_3 = 3 / 2) 
  (h2 : S_3 = 9 / 2)
  (h3 : S_3 = (1 + q + q^2) * a_3 / q^2) :
  q = 1 ∨ q = -1 / 2 := 
by 
  sorry

end common_ratio_l2227_222793


namespace quadratic_real_roots_k_le_one_fourth_l2227_222778

theorem quadratic_real_roots_k_le_one_fourth (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4 * k - 2) * x + k^2 = 0) ↔ k ≤ 1/4 :=
sorry

end quadratic_real_roots_k_le_one_fourth_l2227_222778


namespace sin_150_eq_half_l2227_222769

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l2227_222769


namespace special_pair_example_1_special_pair_example_2_special_pair_negation_l2227_222771

-- Definition of "special rational number pair"
def is_special_rational_pair (a b : ℚ) : Prop := a + b = a * b - 1

-- Problem (1)
theorem special_pair_example_1 : is_special_rational_pair 5 (3 / 2) :=
  by sorry

-- Problem (2)
theorem special_pair_example_2 (a : ℚ) : is_special_rational_pair a 3 → a = 2 :=
  by sorry

-- Problem (3)
theorem special_pair_negation (m n : ℚ) : is_special_rational_pair m n → ¬ is_special_rational_pair (-n) (-m) :=
  by sorry

end special_pair_example_1_special_pair_example_2_special_pair_negation_l2227_222771


namespace find_difference_l2227_222762

theorem find_difference (P : ℝ) (hP : P > 150) :
  let q := P - 150
  let A := 0.2 * P
  let B := 40
  let C := 0.3 * q
  ∃ w z, (0.2 * (150 + 50) >= B) ∧ (30 + 0.2 * q >= 0.3 * q) ∧ 150 + 50 = w ∧ 150 + 300 = z ∧ z - w = 250 :=
by
  sorry

end find_difference_l2227_222762


namespace paul_sandwiches_l2227_222714

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l2227_222714


namespace polar_to_rectangular_l2227_222761

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 4) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 :=
by
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l2227_222761


namespace closest_ratio_adults_children_l2227_222767

theorem closest_ratio_adults_children 
  (a c : ℕ) 
  (H1 : 30 * a + 15 * c = 2550) 
  (H2 : a > 0) 
  (H3 : c > 0) : 
  (a = 57 ∧ c = 56) ∨ (a = 56 ∧ c = 58) :=
by
  sorry

end closest_ratio_adults_children_l2227_222767


namespace quadratic_completion_l2227_222754

theorem quadratic_completion (x d e f : ℤ) (h1 : 100*x^2 + 80*x - 144 = 0) (hd : d > 0) 
  (hde : (d * x + e)^2 = f) : d + e + f = 174 :=
sorry

end quadratic_completion_l2227_222754


namespace trigonometric_inequality_l2227_222719

theorem trigonometric_inequality (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 := 
sorry

end trigonometric_inequality_l2227_222719


namespace prob_four_of_a_kind_after_re_roll_l2227_222738

noncomputable def probability_of_four_of_a_kind : ℚ :=
sorry

theorem prob_four_of_a_kind_after_re_roll :
  (probability_of_four_of_a_kind =
    (1 : ℚ) / 6) :=
sorry

end prob_four_of_a_kind_after_re_roll_l2227_222738


namespace range_of_m_l2227_222724

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y - x * y = 0) :
    (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y - x * y = 0 → x + 2 * y > m^2 + 2 * m) ↔ (-4 : ℝ) < m ∧ m < 2 :=
by
  sorry

end range_of_m_l2227_222724


namespace vectors_parallel_x_eq_four_l2227_222731

theorem vectors_parallel_x_eq_four (x : ℝ) :
  (x > 0) →
  (∃ k : ℝ, (8 + 1/2 * x, x) = k • (x + 1, 2)) →
  x = 4 :=
by
  intro h1 h2
  sorry

end vectors_parallel_x_eq_four_l2227_222731


namespace average_age_in_club_l2227_222773

theorem average_age_in_club :
  let women_avg_age := 32
  let men_avg_age := 38
  let children_avg_age := 10
  let women_count := 12
  let men_count := 18
  let children_count := 10
  let total_ages := (women_avg_age * women_count) + (men_avg_age * men_count) + (children_avg_age * children_count)
  let total_people := women_count + men_count + children_count
  let overall_avg_age := (total_ages : ℝ) / (total_people : ℝ)
  overall_avg_age = 29.2 := by
  sorry

end average_age_in_club_l2227_222773


namespace linear_function_quadrant_l2227_222712

theorem linear_function_quadrant (x y : ℝ) (h : y = -3 * x + 2) :
  ¬ (x > 0 ∧ y > 0) :=
by
  sorry

end linear_function_quadrant_l2227_222712


namespace micah_total_envelopes_l2227_222772

-- Define the conditions as hypotheses
def weight_threshold := 5
def stamps_for_heavy := 5
def stamps_for_light := 2
def total_stamps := 52
def light_envelopes := 6

-- Noncomputable because we are using abstract reasoning rather than computational functions
noncomputable def total_envelopes : ℕ :=
  light_envelopes + (total_stamps - light_envelopes * stamps_for_light) / stamps_for_heavy

-- The theorem to prove
theorem micah_total_envelopes : total_envelopes = 14 := by
  sorry

end micah_total_envelopes_l2227_222772


namespace minimum_value_of_x_plus_2y_l2227_222735

open Real

theorem minimum_value_of_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 := by
  sorry

end minimum_value_of_x_plus_2y_l2227_222735


namespace no_increasing_sequence_with_unique_sum_l2227_222711

theorem no_increasing_sequence_with_unique_sum :
  ¬ (∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ N, ∃ k ≥ N, ∀ m ≥ k, 
    (∃! (i j : ℕ), a i + a j = m))) := sorry

end no_increasing_sequence_with_unique_sum_l2227_222711


namespace perimeter_of_square_l2227_222775

theorem perimeter_of_square (s : ℝ) (h : s^2 = 588) : (4 * s) = 56 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_l2227_222775


namespace value_of_A_l2227_222700

theorem value_of_A (M A T E H : ℤ) (hH : H = 8) (h1 : M + A + T + H = 31) (h2 : T + E + A + M = 40) (h3 : M + E + E + T = 44) (h4 : M + A + T + E = 39) : A = 12 :=
by
  sorry

end value_of_A_l2227_222700


namespace even_and_odd_functions_satisfying_equation_l2227_222734

theorem even_and_odd_functions_satisfying_equation :
  ∀ (f g : ℝ → ℝ),
    (∀ x : ℝ, f (-x) = f x) →                      -- condition 1: f is even
    (∀ x : ℝ, g (-x) = -g x) →                    -- condition 2: g is odd
    (∀ x : ℝ, f x - g x = x^3 + x^2 + 1) →        -- condition 3: f(x) - g(x) = x^3 + x^2 + 1
    f 1 + g 1 = 1 :=                              -- question: proof of f(1) + g(1) = 1
by
  intros f g h_even h_odd h_eqn
  sorry

end even_and_odd_functions_satisfying_equation_l2227_222734


namespace probability_same_color_correct_l2227_222728

def number_of_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def probability_two_balls_same_color : ℚ :=
  ((green_balls / number_of_balls)^2 + (red_balls / number_of_balls)^2 + (blue_balls / number_of_balls)^2)

theorem probability_same_color_correct :
  probability_two_balls_same_color = 49 / 128 := sorry

end probability_same_color_correct_l2227_222728


namespace total_apples_correct_l2227_222748

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end total_apples_correct_l2227_222748


namespace nonagon_blue_quadrilateral_l2227_222760

theorem nonagon_blue_quadrilateral :
  ∀ (vertices : Finset ℕ) (red blue : ℕ → ℕ → Prop),
    (vertices.card = 9) →
    (∀ a b, red a b ∨ blue a b) →
    (∀ a b c, (red a b ∧ red b c ∧ red c a) → False) →
    (∃ A B C D, blue A B ∧ blue B C ∧ blue C D ∧ blue D A ∧ blue A C ∧ blue B D) := 
by
  -- Proof goes here
  sorry

end nonagon_blue_quadrilateral_l2227_222760


namespace inequality_am_gm_l2227_222722

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / c + c / b) ≥ (4 * a / (a + b)) ∧ (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by
  -- Proof steps
  sorry

end inequality_am_gm_l2227_222722


namespace symmetry_implies_condition_l2227_222781

open Function

variable {R : Type*} [Field R]
variables (p q r s : R)

theorem symmetry_implies_condition
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0) 
  (h_symmetry : ∀ x y : R, y = (p * x + q) / (r * x - s) → 
                          -x = (p * (-y) + q) / (r * (-y) - s)) :
  r + s = 0 := 
sorry

end symmetry_implies_condition_l2227_222781


namespace one_fourth_more_equals_thirty_percent_less_l2227_222718

theorem one_fourth_more_equals_thirty_percent_less :
  ∃ n : ℝ, 80 - 0.30 * 80 = (5 / 4) * n ∧ n = 44.8 :=
by
  sorry

end one_fourth_more_equals_thirty_percent_less_l2227_222718


namespace twice_perimeter_is_72_l2227_222708

def twice_perimeter_of_square_field (s : ℝ) : ℝ := 2 * 4 * s

theorem twice_perimeter_is_72 (a P : ℝ) (h1 : a = s^2) (h2 : P = 36) 
    (h3 : 6 * a = 6 * (2 * P + 9)) : twice_perimeter_of_square_field s = 72 := 
by
  sorry

end twice_perimeter_is_72_l2227_222708


namespace difference_xy_l2227_222730

theorem difference_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := sorry

end difference_xy_l2227_222730


namespace sum_is_constant_l2227_222709

variable (a b c d : ℚ) -- declare variables states as rational numbers

theorem sum_is_constant :
  (a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) →
  a + b + c + d = -(14 / 3) :=
by
  intros h
  sorry

end sum_is_constant_l2227_222709


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l2227_222742

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l2227_222742


namespace count_success_permutations_l2227_222727

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l2227_222727


namespace hyperbola_focal_length_l2227_222758

theorem hyperbola_focal_length (x y : ℝ) : 
  (∃ h : x^2 / 9 - y^2 / 4 = 1, 
   ∀ a b : ℝ, a^2 = 9 → b^2 = 4 → 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13) :=
by sorry

end hyperbola_focal_length_l2227_222758


namespace range_of_a_l2227_222745

open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → -1 ≤ a ∧ a ≤ 3 :=
by
  intro h
  -- insert the actual proof here
  sorry

end range_of_a_l2227_222745


namespace solve_for_x_l2227_222783

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (x + 2) / 5) ^ (1 / 4)

theorem solve_for_x : 
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -404 / 201 := 
by {
  sorry
}

end solve_for_x_l2227_222783


namespace alternating_binomial_sum_l2227_222794

open BigOperators Finset

theorem alternating_binomial_sum :
  ∑ k in range 34, (-1 : ℤ)^k * (Nat.choose 99 (3 * k)) = -1 := by
  sorry

end alternating_binomial_sum_l2227_222794


namespace infinite_solutions_l2227_222746

theorem infinite_solutions (x y : ℕ) (h : x ≥ 1 ∧ y ≥ 1) : ∃ (x y : ℕ), x^2 + y^2 = x^3 :=
by {
  sorry 
}

end infinite_solutions_l2227_222746


namespace halfway_between_one_eighth_and_one_third_l2227_222791

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 + 1 / 3) / 2 = 11 / 48 :=
by
  -- Skipping the proof here
  sorry

end halfway_between_one_eighth_and_one_third_l2227_222791


namespace John_profit_is_correct_l2227_222723

-- Definitions of conditions as necessary in Lean
variable (initial_puppies : ℕ) (given_away_puppies : ℕ) (kept_puppy : ℕ) (price_per_puppy : ℤ) (payment_to_stud_owner : ℤ)

-- Specific values from the problem
def John_initial_puppies := 8
def John_given_away_puppies := 4
def John_kept_puppy := 1
def John_price_per_puppy := 600
def John_payment_to_stud_owner := 300

-- Calculate the number of puppies left to sell
def John_remaining_puppies := John_initial_puppies - John_given_away_puppies - John_kept_puppy

-- Calculate total earnings from selling puppies
def John_earnings := John_remaining_puppies * John_price_per_puppy

-- Calculate the profit by subtracting payment to the stud owner from earnings
def John_profit := John_earnings - John_payment_to_stud_owner

-- Statement to prove
theorem John_profit_is_correct : 
  John_profit = 1500 := 
by 
  -- The proof will be here but we use sorry to skip it as requested.
  sorry

-- This ensures the definitions match the given problem conditions
#eval (John_initial_puppies, John_given_away_puppies, John_kept_puppy, John_price_per_puppy, John_payment_to_stud_owner)

end John_profit_is_correct_l2227_222723


namespace neither_sufficient_nor_necessary_l2227_222753

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(-1 < x ∧ x < 2 → |x - 2| < 1) ∧ ¬(|x - 2| < 1 → -1 < x ∧ x < 2) :=
by
  sorry

end neither_sufficient_nor_necessary_l2227_222753


namespace quadratic_root_2020_l2227_222705

theorem quadratic_root_2020 (a b : ℝ) (h₀ : a ≠ 0) (h₁ : a * 2019^2 + b * 2019 - 1 = 0) :
    ∃ x : ℝ, (a * (x - 1)^2 + b * (x - 1) = 1) ∧ x = 2020 :=
by
  sorry

end quadratic_root_2020_l2227_222705


namespace initial_overs_l2227_222732

theorem initial_overs (x : ℝ) (r1 : ℝ) (r2 : ℝ) (target : ℝ) (overs_remaining : ℝ) :
  r1 = 3.2 ∧ overs_remaining = 22 ∧ r2 = 11.363636363636363 ∧ target = 282 ∧
  (r1 * x + r2 * overs_remaining = target) → x = 10 :=
by
  intro h
  obtain ⟨hr1, ho, hr2, ht, heq⟩ := h
  sorry

end initial_overs_l2227_222732


namespace total_travel_time_l2227_222749

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l2227_222749


namespace carla_final_payment_l2227_222702

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l2227_222702


namespace min_value_f_l2227_222743

theorem min_value_f (x : ℝ) (h : 0 < x) : 
  ∃ c: ℝ, c = 2.5 ∧ (∀ x, 0 < x → x^2 + 1 / x^2 + 1 / (x^2 + 1 / x^2) ≥ c) :=
by sorry

end min_value_f_l2227_222743


namespace sacks_after_days_l2227_222741

-- Define the number of sacks harvested per day
def harvest_per_day : ℕ := 74

-- Define the number of sacks discarded per day
def discard_per_day : ℕ := 71

-- Define the days of harvest
def days_of_harvest : ℕ := 51

-- Define the number of sacks that are not discarded per day
def net_sacks_per_day : ℕ := harvest_per_day - discard_per_day

-- Define the total number of sacks after the specified days of harvest
def total_sacks : ℕ := days_of_harvest * net_sacks_per_day

theorem sacks_after_days :
  total_sacks = 153 := by
  sorry

end sacks_after_days_l2227_222741


namespace motorcycle_materials_cost_l2227_222710

theorem motorcycle_materials_cost 
  (car_material_cost : ℕ) (cars_per_month : ℕ) (car_sale_price : ℕ)
  (motorcycles_per_month : ℕ) (motorcycle_sale_price : ℕ)
  (additional_profit : ℕ) :
  car_material_cost = 100 →
  cars_per_month = 4 →
  car_sale_price = 50 →
  motorcycles_per_month = 8 →
  motorcycle_sale_price = 50 →
  additional_profit = 50 →
  car_material_cost + additional_profit = 250 := by
  sorry

end motorcycle_materials_cost_l2227_222710


namespace find_second_term_l2227_222759

theorem find_second_term (A B : ℕ) (h1 : A / B = 3 / 4) (h2 : (A + 10) / (B + 10) = 4 / 5) : B = 40 :=
sorry

end find_second_term_l2227_222759


namespace triangle_side_length_l2227_222715

theorem triangle_side_length (A B : ℝ) (b : ℝ) (a : ℝ) 
  (hA : A = 60) (hB : B = 45) (hb : b = 2) 
  (h : a = b * (Real.sin A) / (Real.sin B)) :
  a = Real.sqrt 6 := by
  sorry

end triangle_side_length_l2227_222715


namespace tan_945_equals_1_l2227_222757

noncomputable def tan_circular (x : ℝ) : ℝ := Real.tan x

theorem tan_945_equals_1 :
  tan_circular 945 = 1 := 
by
  sorry

end tan_945_equals_1_l2227_222757


namespace triangle_area_correct_l2227_222764

/-- Define the points of the triangle -/
def x1 : ℝ := -4
def y1 : ℝ := 2
def x2 : ℝ := 2
def y2 : ℝ := 8
def x3 : ℝ := -2
def y3 : ℝ := -2

/-- Define the area calculation function -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Define the area of the given triangle -/
def given_triangle_area : ℝ :=
  triangle_area x1 y1 x2 y2 x3 y3

/-- The goal is to prove that the area of the given triangle is 22 square units -/
theorem triangle_area_correct : given_triangle_area = 22 := by
  sorry

end triangle_area_correct_l2227_222764


namespace initial_shed_bales_zero_l2227_222770

def bales_in_barn_initial : ℕ := 47
def bales_added_by_benny : ℕ := 35
def bales_in_barn_total : ℕ := 82

theorem initial_shed_bales_zero (b_shed : ℕ) :
  bales_in_barn_initial + bales_added_by_benny = bales_in_barn_total → b_shed = 0 :=
by
  intro h
  sorry

end initial_shed_bales_zero_l2227_222770


namespace ratio_of_rooms_l2227_222790

def rooms_in_danielle_apartment : Nat := 6
def rooms_in_heidi_apartment : Nat := 3 * rooms_in_danielle_apartment
def rooms_in_grant_apartment : Nat := 2

theorem ratio_of_rooms :
  (rooms_in_grant_apartment : ℚ) / (rooms_in_heidi_apartment : ℚ) = 1 / 9 := 
by 
  sorry

end ratio_of_rooms_l2227_222790


namespace mike_earnings_l2227_222750

theorem mike_earnings (total_games non_working_games price_per_game : ℕ) 
  (h1 : total_games = 15) (h2 : non_working_games = 9) (h3 : price_per_game = 5) : 
  total_games - non_working_games * price_per_game = 30 :=
by
  rw [h1, h2, h3]
  show 15 - 9 * 5 = 30
  sorry

end mike_earnings_l2227_222750


namespace smallest_number_among_options_l2227_222795

noncomputable def binary_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 111111 => 63
  | _ => 0

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 210 => 2 * 6^2 + 1 * 6
  | _ => 0

noncomputable def base_nine_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 85 => 8 * 9 + 5
  | _ => 0

theorem smallest_number_among_options :
  min 75 (min (binary_to_decimal 111111) (min (base_six_to_decimal 210) (base_nine_to_decimal 85))) = binary_to_decimal 111111 :=
by 
  sorry

end smallest_number_among_options_l2227_222795


namespace seed_grow_prob_l2227_222720

theorem seed_grow_prob (P_G P_S_given_G : ℝ) (hP_G : P_G = 0.9) (hP_S_given_G : P_S_given_G = 0.8) :
  P_G * P_S_given_G = 0.72 :=
by
  rw [hP_G, hP_S_given_G]
  norm_num

end seed_grow_prob_l2227_222720


namespace solution_set_M_abs_ineq_l2227_222765

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- The first statement to prove the solution set M for the inequality
theorem solution_set_M : ∀ x, f x < 3 ↔ x ∈ M :=
by sorry

-- The second statement to prove the inequality when a, b ∈ M
theorem abs_ineq (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + ab| :=
by sorry

end solution_set_M_abs_ineq_l2227_222765


namespace inequality_solution_l2227_222780

theorem inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 5) →
  ((x * x - 4 * x - 5) / (x * x + 3 * x + 2) < 0 ↔ (x ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∨ x ∈ Set.Ioo (-1:ℝ) (5:ℝ))) :=
by
  sorry

end inequality_solution_l2227_222780


namespace equilateral_triangle_AB_length_l2227_222792

noncomputable def Q := 2
noncomputable def R := 3
noncomputable def S := 4

theorem equilateral_triangle_AB_length :
  ∀ (AB BC CA : ℝ), 
  AB = BC ∧ BC = CA ∧ (∃ P : ℝ × ℝ, (Q = 2) ∧ (R = 3) ∧ (S = 4)) →
  AB = 6 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_AB_length_l2227_222792


namespace parker_total_weight_l2227_222763

def twenty_pound := 20
def thirty_pound := 30
def forty_pound := 40

def first_set_weight := (2 * twenty_pound) + (1 * thirty_pound) + (1 * forty_pound)
def second_set_weight := (1 * twenty_pound) + (2 * thirty_pound) + (2 * forty_pound)
def third_set_weight := (3 * thirty_pound) + (3 * forty_pound)

def total_weight := first_set_weight + second_set_weight + third_set_weight

theorem parker_total_weight :
  total_weight = 480 := by
  sorry

end parker_total_weight_l2227_222763


namespace f_of_8_l2227_222736

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom function_property : ∀ x : ℝ, f (x + 2) = -1 / f (x)

-- Statement to prove
theorem f_of_8 : f 8 = 0 :=
sorry

end f_of_8_l2227_222736


namespace arithmetic_seq_40th_term_l2227_222789

theorem arithmetic_seq_40th_term (a₁ d : ℕ) (n : ℕ) (h1 : a₁ = 3) (h2 : d = 4) (h3 : n = 40) : 
  a₁ + (n - 1) * d = 159 :=
by
  sorry

end arithmetic_seq_40th_term_l2227_222789


namespace pounds_lost_per_month_l2227_222786

variable (starting_weight : ℕ) (ending_weight : ℕ) (months_in_year : ℕ) 

theorem pounds_lost_per_month
    (h_start : starting_weight = 250)
    (h_end : ending_weight = 154)
    (h_months : months_in_year = 12) :
    (starting_weight - ending_weight) / months_in_year = 8 := 
sorry

end pounds_lost_per_month_l2227_222786
