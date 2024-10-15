import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l349_34985

noncomputable def setM (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def setN : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : setM a ∪ setN = setN ↔ (-2 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l349_34985


namespace NUMINAMATH_GPT_cost_of_goods_l349_34980

-- Define variables and conditions
variables (x y z : ℝ)

-- Assume the given conditions
axiom h1 : x + 2 * y + 3 * z = 136
axiom h2 : 3 * x + 2 * y + z = 240

-- Statement to prove
theorem cost_of_goods : x + y + z = 94 := 
sorry

end NUMINAMATH_GPT_cost_of_goods_l349_34980


namespace NUMINAMATH_GPT_min_third_side_of_right_triangle_l349_34922

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end NUMINAMATH_GPT_min_third_side_of_right_triangle_l349_34922


namespace NUMINAMATH_GPT_peter_speed_l349_34994

theorem peter_speed (p : ℝ) (v_juan : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_juan = p + 3) 
  (h2 : d = t * p + t * v_juan) 
  (h3 : t = 1.5) 
  (h4 : d = 19.5) : 
  p = 5 :=
by
  sorry

end NUMINAMATH_GPT_peter_speed_l349_34994


namespace NUMINAMATH_GPT_complement_union_l349_34938

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union : 
  U = {0, 1, 2, 3, 4} →
  (U \ A = {1, 2}) →
  B = {1, 3} →
  (A ∪ B = {0, 1, 3, 4}) :=
by
  intros hU hA hB
  sorry

end NUMINAMATH_GPT_complement_union_l349_34938


namespace NUMINAMATH_GPT_exponent_multiplication_l349_34981

-- Define the variables and exponentiation property
variable (a : ℝ)

-- State the theorem
theorem exponent_multiplication : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l349_34981


namespace NUMINAMATH_GPT_margaret_score_l349_34931

theorem margaret_score (average_score marco_score margaret_score : ℝ)
  (h1: average_score = 90)
  (h2: marco_score = average_score - 0.10 * average_score)
  (h3: margaret_score = marco_score + 5) : 
  margaret_score = 86 := 
by
  sorry

end NUMINAMATH_GPT_margaret_score_l349_34931


namespace NUMINAMATH_GPT_badges_total_l349_34995

theorem badges_total :
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  hermione_badges + luna_badges + celestia_badges = 83 :=
by
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  sorry

end NUMINAMATH_GPT_badges_total_l349_34995


namespace NUMINAMATH_GPT_line_intersects_circle_l349_34937

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end NUMINAMATH_GPT_line_intersects_circle_l349_34937


namespace NUMINAMATH_GPT_stamps_max_l349_34945

theorem stamps_max (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 25) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, (n * price_per_stamp ≤ total_cents) ∧ (∀ m : ℕ, (m > n) → (m * price_per_stamp > total_cents)) ∧ n = 200 := 
by
  sorry

end NUMINAMATH_GPT_stamps_max_l349_34945


namespace NUMINAMATH_GPT_gcd_eq_gcd_of_eq_add_mul_l349_34978

theorem gcd_eq_gcd_of_eq_add_mul (a b q r : Int) (h_q : b > 0) (h_r : 0 ≤ r) (h_ar : a = b * q + r) : Int.gcd a b = Int.gcd b r :=
by
  -- Conditions: constraints and assertion
  exact sorry

end NUMINAMATH_GPT_gcd_eq_gcd_of_eq_add_mul_l349_34978


namespace NUMINAMATH_GPT_sum_possible_values_for_k_l349_34959

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end NUMINAMATH_GPT_sum_possible_values_for_k_l349_34959


namespace NUMINAMATH_GPT_road_trip_cost_l349_34916

theorem road_trip_cost 
  (x : ℝ)
  (initial_cost_per_person: ℝ) 
  (redistributed_cost_per_person: ℝ)
  (cost_difference: ℝ) :
  initial_cost_per_person = x / 4 →
  redistributed_cost_per_person = x / 7 →
  cost_difference = 8 →
  initial_cost_per_person - redistributed_cost_per_person = cost_difference →
  x = 74.67 :=
by
  intro h1 h2 h3 h4
  -- starting the proof
  rw [h1, h2] at h4
  sorry

end NUMINAMATH_GPT_road_trip_cost_l349_34916


namespace NUMINAMATH_GPT_sibling_discount_is_correct_l349_34973

-- Defining the given conditions
def tuition_per_person : ℕ := 45
def total_cost_with_discount : ℕ := 75

-- Defining the calculation of sibling discount
def sibling_discount : ℕ :=
  let original_cost := 2 * tuition_per_person
  let discount := original_cost - total_cost_with_discount
  discount

-- Statement to prove
theorem sibling_discount_is_correct : sibling_discount = 15 :=
by
  unfold sibling_discount
  simp
  sorry

end NUMINAMATH_GPT_sibling_discount_is_correct_l349_34973


namespace NUMINAMATH_GPT_total_balls_estimate_l349_34982

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end NUMINAMATH_GPT_total_balls_estimate_l349_34982


namespace NUMINAMATH_GPT_intersection_complement_l349_34992

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of B with respect to U
def comp_B : Set ℕ := U \ B

-- Statement to be proven
theorem intersection_complement : A ∩ comp_B = {1, 3} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_l349_34992


namespace NUMINAMATH_GPT_largest_class_students_l349_34912

theorem largest_class_students :
  ∃ x : ℕ, (x + (x - 4) + (x - 8) + (x - 12) + (x - 16) + (x - 20) + (x - 24) +
  (x - 28) + (x - 32) + (x - 36) = 100) ∧ x = 28 :=
by
  sorry

end NUMINAMATH_GPT_largest_class_students_l349_34912


namespace NUMINAMATH_GPT_time_to_cross_first_platform_l349_34934

-- Define the given conditions
def length_first_platform : ℕ := 140
def length_second_platform : ℕ := 250
def length_train : ℕ := 190
def time_cross_second_platform : Nat := 20
def speed := (length_train + length_second_platform) / time_cross_second_platform

-- The theorem to be proved
theorem time_to_cross_first_platform : 
  (length_train + length_first_platform) / speed = 15 :=
sorry

end NUMINAMATH_GPT_time_to_cross_first_platform_l349_34934


namespace NUMINAMATH_GPT_integer_pair_condition_l349_34968

theorem integer_pair_condition (m n : ℤ) (h : (m^2 + m * n + n^2 : ℚ) / (m + 2 * n) = 13 / 3) : m + 2 * n = 9 :=
sorry

end NUMINAMATH_GPT_integer_pair_condition_l349_34968


namespace NUMINAMATH_GPT_min_value_of_F_on_neg_infinity_l349_34905

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions provided in the problem
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom g_odd : ∀ x : ℝ, g (-x) = - g x
noncomputable def F (x : ℝ) := a * f x + b * g x + 2
axiom F_max_on_pos : ∃ x ∈ (Set.Ioi 0), F x = 5

-- Prove the conclusion of the problem
theorem min_value_of_F_on_neg_infinity : ∃ y ∈ (Set.Iio 0), F y = -1 :=
sorry

end NUMINAMATH_GPT_min_value_of_F_on_neg_infinity_l349_34905


namespace NUMINAMATH_GPT_coefficient_x8_expansion_l349_34960

-- Define the problem statement in Lean
theorem coefficient_x8_expansion : 
  (Nat.choose 7 4) * (1 : ℤ)^3 * (-2 : ℤ)^4 = 560 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x8_expansion_l349_34960


namespace NUMINAMATH_GPT_radius_of_circle_B_l349_34977

theorem radius_of_circle_B (r_A r_D : ℝ) (r_B : ℝ) (hA : r_A = 2) (hD : r_D = 4) 
  (congruent_BC : r_B = r_B) (tangent_condition : true) -- placeholder conditions
  (center_pass : true) -- placeholder conditions
  : r_B = (4 / 3) * (Real.sqrt 7 - 1) :=
sorry

end NUMINAMATH_GPT_radius_of_circle_B_l349_34977


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_is_986_l349_34993

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_is_986_l349_34993


namespace NUMINAMATH_GPT_apps_left_on_phone_l349_34984

-- Definitions for the given conditions
def initial_apps : ℕ := 15
def added_apps : ℕ := 71
def deleted_apps : ℕ := added_apps + 1

-- Proof statement
theorem apps_left_on_phone : initial_apps + added_apps - deleted_apps = 14 := by
  sorry

end NUMINAMATH_GPT_apps_left_on_phone_l349_34984


namespace NUMINAMATH_GPT_strawberries_remaining_l349_34965

theorem strawberries_remaining (initial : ℝ) (eaten_yesterday : ℝ) (eaten_today : ℝ) :
  initial = 1.6 ∧ eaten_yesterday = 0.8 ∧ eaten_today = 0.3 → initial - eaten_yesterday - eaten_today = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_remaining_l349_34965


namespace NUMINAMATH_GPT_binary_operation_correct_l349_34983

-- Define the binary numbers involved
def bin1 := 0b110110 -- 110110_2
def bin2 := 0b101010 -- 101010_2
def bin3 := 0b100    -- 100_2

-- Define the operation in binary
def result := 0b111001101100 -- 111001101100_2

-- Lean statement to verify the operation result
theorem binary_operation_correct : (bin1 * bin2) / bin3 = result :=
by sorry

end NUMINAMATH_GPT_binary_operation_correct_l349_34983


namespace NUMINAMATH_GPT_gcd_of_consecutive_digit_sums_is_1111_l349_34963

theorem gcd_of_consecutive_digit_sums_is_1111 (p q r s : ℕ) (hc : q = p+1 ∧ r = p+2 ∧ s = p+3) :
  ∃ d, d = 1111 ∧ ∀ n : ℕ, n = (1000 * p + 100 * q + 10 * r + s) + (1000 * s + 100 * r + 10 * q + p) → d ∣ n := by
  use 1111
  sorry

end NUMINAMATH_GPT_gcd_of_consecutive_digit_sums_is_1111_l349_34963


namespace NUMINAMATH_GPT_find_k_l349_34957

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : ∀ x ∈ Set.Icc (2 : ℝ) 4, y = k / x → y ≥ 5) : k = 20 :=
sorry

end NUMINAMATH_GPT_find_k_l349_34957


namespace NUMINAMATH_GPT_find_larger_number_l349_34901

theorem find_larger_number (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l349_34901


namespace NUMINAMATH_GPT_intersection_two_sets_l349_34924

theorem intersection_two_sets (M N : Set ℤ) (h1 : M = {1, 2, 3, 4}) (h2 : N = {-2, 2}) :
  M ∩ N = {2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_two_sets_l349_34924


namespace NUMINAMATH_GPT_points_equidistant_from_circle_and_tangents_l349_34930

noncomputable def circle_radius := 4
noncomputable def tangent_distance := 6

theorem points_equidistant_from_circle_and_tangents :
  ∃! (P : ℝ × ℝ), dist P (0, 0) = circle_radius ∧
                 dist P (0, tangent_distance) = tangent_distance - circle_radius ∧
                 dist P (0, -tangent_distance) = tangent_distance - circle_radius :=
by {
  sorry
}

end NUMINAMATH_GPT_points_equidistant_from_circle_and_tangents_l349_34930


namespace NUMINAMATH_GPT_Liza_initial_balance_l349_34946

theorem Liza_initial_balance
  (W: Nat)   -- Liza's initial balance on Tuesday
  (rent: Nat := 450)
  (deposit: Nat := 1500)
  (electricity: Nat := 117)
  (internet: Nat := 100)
  (phone: Nat := 70)
  (final_balance: Nat := 1563) 
  (balance_eq: W - rent + deposit - electricity - internet - phone = final_balance) 
  : W = 800 :=
sorry

end NUMINAMATH_GPT_Liza_initial_balance_l349_34946


namespace NUMINAMATH_GPT_multiple_of_persons_l349_34997

variable (Persons Work : ℕ) (Rate : ℚ)

def work_rate (P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D
def multiple_work_rate (m P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D

theorem multiple_of_persons
  (P : ℕ) (W : ℕ)
  (h1 : work_rate P W 12 = W / 12)
  (h2 : multiple_work_rate 1 P (W / 2) 3 = (W / 6)) :
  m = 2 :=
by sorry

end NUMINAMATH_GPT_multiple_of_persons_l349_34997


namespace NUMINAMATH_GPT_fraction_irreducible_l349_34935

open Nat

theorem fraction_irreducible (m n : ℕ) : Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 :=
  sorry

end NUMINAMATH_GPT_fraction_irreducible_l349_34935


namespace NUMINAMATH_GPT_yangyang_departure_time_l349_34904

noncomputable def departure_time : Nat := 373 -- 6:13 in minutes from midnight (6 * 60 + 13)

theorem yangyang_departure_time :
  let arrival_at_60_mpm := 413 -- 6:53 in minutes from midnight
  let arrival_at_75_mpm := 405 -- 6:45 in minutes from midnight
  let difference := arrival_at_60_mpm - arrival_at_75_mpm -- time difference
  let x := 40 -- time taken to walk to school at 60 meters per minute
  departure_time = arrival_at_60_mpm - x :=
by
  -- Definitions
  let arrival_at_60_mpm := 413
  let arrival_at_75_mpm := 405
  let difference := 8
  let x := 40
  have h : departure_time = (413 - 40) := rfl
  sorry

end NUMINAMATH_GPT_yangyang_departure_time_l349_34904


namespace NUMINAMATH_GPT_quadratic_eq_has_real_root_l349_34927

theorem quadratic_eq_has_real_root (a b : ℝ) :
  ¬(∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_has_real_root_l349_34927


namespace NUMINAMATH_GPT_commission_percentage_is_4_l349_34932

-- Define the given conditions
def commission := 12.50
def total_sales := 312.5

-- The problem is to prove the commission percentage
theorem commission_percentage_is_4 :
  (commission / total_sales) * 100 = 4 := by
  sorry

end NUMINAMATH_GPT_commission_percentage_is_4_l349_34932


namespace NUMINAMATH_GPT_doubled_sum_of_squares_l349_34926

theorem doubled_sum_of_squares (a b : ℝ) : 
  2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := 
by
  sorry

end NUMINAMATH_GPT_doubled_sum_of_squares_l349_34926


namespace NUMINAMATH_GPT_determine_A_plus_B_l349_34925

theorem determine_A_plus_B :
  ∃ (A B : ℚ), ((∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → 
  (Bx - 23) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) ∧
  (A + B = 11 / 9)) :=
sorry

end NUMINAMATH_GPT_determine_A_plus_B_l349_34925


namespace NUMINAMATH_GPT_at_least_one_miss_l349_34928

variables (p q : Prop)

-- Proposition stating the necessary and sufficient condition.
theorem at_least_one_miss : ¬(p ∧ q) ↔ (¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_GPT_at_least_one_miss_l349_34928


namespace NUMINAMATH_GPT_common_chord_eqn_l349_34976

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 6 * y + 1 = 0) ∧
  (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) →
  3 * x - 4 * y + 6 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_common_chord_eqn_l349_34976


namespace NUMINAMATH_GPT_leaves_blew_away_correct_l349_34921

-- Definitions based on conditions
def original_leaves : ℕ := 356
def leaves_left : ℕ := 112
def leaves_blew_away : ℕ := original_leaves - leaves_left

-- Theorem statement based on the question and correct answer
theorem leaves_blew_away_correct : leaves_blew_away = 244 := by {
  -- Proof goes here (omitted for now)
  sorry
}

end NUMINAMATH_GPT_leaves_blew_away_correct_l349_34921


namespace NUMINAMATH_GPT_farmer_ear_count_l349_34902

theorem farmer_ear_count
    (seeds_per_ear : ℕ)
    (price_per_ear : ℝ)
    (cost_per_bag : ℝ)
    (seeds_per_bag : ℕ)
    (profit : ℝ)
    (target_profit : ℝ) :
  seeds_per_ear = 4 →
  price_per_ear = 0.1 →
  cost_per_bag = 0.5 →
  seeds_per_bag = 100 →
  target_profit = 40 →
  profit = price_per_ear - ((cost_per_bag / seeds_per_bag) * seeds_per_ear) →
  target_profit / profit = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_farmer_ear_count_l349_34902


namespace NUMINAMATH_GPT_probability_three_hearts_l349_34941

noncomputable def probability_of_three_hearts : ℚ :=
  (13/52) * (12/51) * (11/50)

theorem probability_three_hearts :
  probability_of_three_hearts = 26/2025 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_hearts_l349_34941


namespace NUMINAMATH_GPT_polynomial_transformation_l349_34947

-- Given the conditions of the polynomial function g and the provided transformation
-- We aim to prove the equivalence in a mathematically formal way using Lean

theorem polynomial_transformation (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 3 * x^2 - 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_polynomial_transformation_l349_34947


namespace NUMINAMATH_GPT_find_x_value_l349_34964

theorem find_x_value (x : ℝ) :
  |x - 25| + |x - 21| = |3 * x - 75| → x = 71 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l349_34964


namespace NUMINAMATH_GPT_value_expression_l349_34936

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_expression (p q r s t : ℝ) (h : g p q r s t (-3) = 9) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = -9 := 
by
  sorry

end NUMINAMATH_GPT_value_expression_l349_34936


namespace NUMINAMATH_GPT_new_unemployment_rate_is_66_percent_l349_34944

theorem new_unemployment_rate_is_66_percent
  (initial_unemployment_rate : ℝ)
  (initial_employment_rate : ℝ)
  (u_increases_by_10_percent : initial_unemployment_rate * 1.1 = new_unemployment_rate)
  (e_decreases_by_15_percent : initial_employment_rate * 0.85 = new_employment_rate)
  (sum_is_100_percent : initial_unemployment_rate + initial_employment_rate = 100) :
  new_unemployment_rate = 66 :=
by
  sorry

end NUMINAMATH_GPT_new_unemployment_rate_is_66_percent_l349_34944


namespace NUMINAMATH_GPT_equal_number_of_experienced_fishermen_and_children_l349_34979

theorem equal_number_of_experienced_fishermen_and_children 
  (n : ℕ)
  (total_fish : ℕ)
  (children_catch : ℕ)
  (fishermen_catch : ℕ)
  (h1 : total_fish = n^2 + 5 * n + 22)
  (h2 : fishermen_catch - 10 = children_catch)
  (h3 : total_fish = n * children_catch + 11 * fishermen_catch)
  (h4 : fishermen_catch > children_catch)
  : n = 11 := 
sorry

end NUMINAMATH_GPT_equal_number_of_experienced_fishermen_and_children_l349_34979


namespace NUMINAMATH_GPT_math_books_count_l349_34900

theorem math_books_count (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 396) : M = 54 :=
sorry

end NUMINAMATH_GPT_math_books_count_l349_34900


namespace NUMINAMATH_GPT_find_number_l349_34948

theorem find_number (n : ℤ) (h : 7 * n = 3 * n + 12) : n = 3 :=
sorry

end NUMINAMATH_GPT_find_number_l349_34948


namespace NUMINAMATH_GPT_complex_division_l349_34996

theorem complex_division (i : ℂ) (h : i ^ 2 = -1) : (3 - 4 * i) / i = -4 - 3 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l349_34996


namespace NUMINAMATH_GPT_complex_seventh_root_identity_l349_34971

open Complex

theorem complex_seventh_root_identity (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 :=
by
  sorry

end NUMINAMATH_GPT_complex_seventh_root_identity_l349_34971


namespace NUMINAMATH_GPT_nathan_final_temperature_l349_34911

theorem nathan_final_temperature : ∃ (final_temp : ℝ), final_temp = 77.4 :=
  let initial_temp : ℝ := 50
  let type_a_increase : ℝ := 2
  let type_b_increase : ℝ := 3.5
  let type_c_increase : ℝ := 4.8
  let type_d_increase : ℝ := 7.2
  let type_a_quantity : ℚ := 6
  let type_b_quantity : ℚ := 5
  let type_c_quantity : ℚ := 9
  let type_d_quantity : ℚ := 3
  let temp_after_a := initial_temp + 3 * type_a_increase
  let temp_after_b := temp_after_a + 2 * type_b_increase
  let temp_after_c := temp_after_b + 3 * type_c_increase
  let final_temp := temp_after_c
  ⟨final_temp, sorry⟩

end NUMINAMATH_GPT_nathan_final_temperature_l349_34911


namespace NUMINAMATH_GPT_find_t_l349_34954

theorem find_t (t : ℝ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1)^(n-1)) → -- Geometric sequence condition
  (∀ n, S_n n = 2017 * 2016^n - 2018 * t) →     -- Given sum formula
  t = 2017 / 2018 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l349_34954


namespace NUMINAMATH_GPT_determine_p_l349_34910

theorem determine_p (p : ℝ) (h : (2 * p - 1) * (-1)^2 + 2 * (1 - p) * (-1) + 3 * p = 0) : p = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_determine_p_l349_34910


namespace NUMINAMATH_GPT_union_set_solution_l349_34950

theorem union_set_solution (M N : Set ℝ) 
    (hM : M = { x | 0 ≤ x ∧ x ≤ 3 }) 
    (hN : N = { x | x < 1 }) : 
    M ∪ N = { x | x ≤ 3 } := 
by 
    sorry

end NUMINAMATH_GPT_union_set_solution_l349_34950


namespace NUMINAMATH_GPT_find_numbers_l349_34952

theorem find_numbers :
  ∃ a d : ℝ, 
    ((a - d) + a + (a + d) = 12) ∧ 
    ((a - d) * a * (a + d) = 48) ∧
    (a = 4) ∧ 
    (d = -2) ∧ 
    (a - d = 6) ∧ 
    (a + d = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l349_34952


namespace NUMINAMATH_GPT_quadratic_is_binomial_square_l349_34962

theorem quadratic_is_binomial_square 
  (a : ℤ) : 
  (∃ b : ℤ, 9 * (x: ℤ)^2 - 24 * x + a = (3 * x + b)^2) ↔ a = 16 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_is_binomial_square_l349_34962


namespace NUMINAMATH_GPT_prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l349_34919

-- Definitions
def groupA : List ℕ := [2, 4, 6]
def groupB : List ℕ := [3, 5]
def card_count_A : ℕ := groupA.length
def card_count_B : ℕ := groupB.length

-- Condition 1: Probability of drawing the card with number 2 from group A
def prob_draw_2_groupA : ℚ := 1 / card_count_A

-- Condition 2: Game Rule Outcomes
def is_multiple_of_3 (n : ℕ) : Bool := n % 3 == 0

def outcomes : List (ℕ × ℕ) := [(2, 3), (2, 5), (4, 3), (4, 5), (6, 3), (6, 5)]

def winning_outcomes_A : List (ℕ × ℕ) :=List.filter (λ p => is_multiple_of_3 (p.1 * p.2)) outcomes
def winning_outcomes_B : List (ℕ × ℕ) := List.filter (λ p => ¬ is_multiple_of_3 (p.1 * p.2)) outcomes

def prob_win_A : ℚ := winning_outcomes_A.length / outcomes.length
def prob_win_B : ℚ := winning_outcomes_B.length / outcomes.length

-- Proof problems
theorem prob_draw_2_groupA_is_one_third : prob_draw_2_groupA = 1 / 3 := sorry

theorem game_rule_is_unfair : prob_win_A ≠ prob_win_B := sorry

end NUMINAMATH_GPT_prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l349_34919


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l349_34986

theorem triangle_angle_contradiction (α β γ : ℝ) (h : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) -> false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l349_34986


namespace NUMINAMATH_GPT_xy_sum_greater_two_l349_34967

theorem xy_sum_greater_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := 
by 
  sorry

end NUMINAMATH_GPT_xy_sum_greater_two_l349_34967


namespace NUMINAMATH_GPT_find_number_l349_34915

theorem find_number (x : ℝ) (h : 0.7 * x = 48 + 22) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l349_34915


namespace NUMINAMATH_GPT_x_plus_y_possible_values_l349_34918

theorem x_plus_y_possible_values (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < 20) (h4 : y < 20) (h5 : x + y + x * y = 99) : 
  x + y = 23 ∨ x + y = 18 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_y_possible_values_l349_34918


namespace NUMINAMATH_GPT_cost_of_jeans_and_shirts_l349_34914

theorem cost_of_jeans_and_shirts 
  (S : ℕ) (J : ℕ) (X : ℕ)
  (hS : S = 18)
  (h2J3S : 2 * J + 3 * S = 76)
  (h3J2S : 3 * J + 2 * S = X) :
  X = 69 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_jeans_and_shirts_l349_34914


namespace NUMINAMATH_GPT_service_charge_percentage_is_correct_l349_34951

-- Define the conditions
def orderAmount : ℝ := 450
def totalAmountPaid : ℝ := 468
def serviceCharge : ℝ := totalAmountPaid - orderAmount

-- Define the target percentage
def expectedServiceChargePercentage : ℝ := 4.0

-- Proof statement: the service charge percentage is expectedServiceChargePercentage
theorem service_charge_percentage_is_correct : 
  (serviceCharge / orderAmount) * 100 = expectedServiceChargePercentage :=
by
  sorry

end NUMINAMATH_GPT_service_charge_percentage_is_correct_l349_34951


namespace NUMINAMATH_GPT_speed_of_stream_l349_34969

-- Definitions based on given conditions
def speed_still_water := 24 -- km/hr
def distance_downstream := 140 -- km
def time_downstream := 5 -- hours

-- Proof problem statement
theorem speed_of_stream (v : ℕ) :
  24 + v = distance_downstream / time_downstream → v = 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l349_34969


namespace NUMINAMATH_GPT_term_five_eq_nine_l349_34909

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- The sum of the first n terms of the sequence equals n^2.
axiom sum_formula : ∀ n, S n = n^2

-- Definition of the nth term in terms of the sequence sum.
def a_n (n : ℕ) : ℕ := S n - S (n - 1)

-- Goal: Prove that the 5th term, a(5), equals 9.
theorem term_five_eq_nine : a_n S 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_term_five_eq_nine_l349_34909


namespace NUMINAMATH_GPT_find_K_values_l349_34966

-- Define summation of first K natural numbers
def sum_natural_numbers (K : ℕ) : ℕ :=
  K * (K + 1) / 2

-- Define the main problem conditions
theorem find_K_values (K N : ℕ) (hN_positive : N > 0) (hN_bound : N < 150) (h_sum_eq : sum_natural_numbers K = 3 * N^2) :
  K = 2 ∨ K = 12 ∨ K = 61 :=
  sorry

end NUMINAMATH_GPT_find_K_values_l349_34966


namespace NUMINAMATH_GPT_symmetric_line_equation_l349_34987

theorem symmetric_line_equation {l : ℝ} (h1 : ∀ x y : ℝ, x + y - 1 = 0 → (-x) - y + 1 = l) : l = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l349_34987


namespace NUMINAMATH_GPT_possible_values_l349_34999

theorem possible_values (m n : ℕ) (h1 : 10 ≥ m) (h2 : m > n) (h3 : n ≥ 4) (h4 : (m - n) ^ 2 = m + n) :
    (m, n) = (10, 6) :=
sorry

end NUMINAMATH_GPT_possible_values_l349_34999


namespace NUMINAMATH_GPT_perpendicular_lines_unique_a_l349_34989

open Real

theorem perpendicular_lines_unique_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, (a - 1) * x + y - 1 = 0) 
  (l2 : ∀ x y : ℝ, 3 * x + a * y + 2 = 0) 
  (perpendicular : True) : 
  a = 3 / 4 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_unique_a_l349_34989


namespace NUMINAMATH_GPT_compute_expression_at_4_l349_34923

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_GPT_compute_expression_at_4_l349_34923


namespace NUMINAMATH_GPT_sqrt_one_half_eq_sqrt_two_over_two_l349_34920

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_one_half_eq_sqrt_two_over_two_l349_34920


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l349_34939

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → α)
  (q : α)
  (h1 : is_geometric_sequence a q)
  (h2 : a 3 = 6)
  (h3 : a 0 + a 1 + a 2 = 18) :
  q = 1 ∨ q = - (1 / 2) := 
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l349_34939


namespace NUMINAMATH_GPT_LimingFatherAge_l349_34998

theorem LimingFatherAge
  (age month day : ℕ)
  (age_condition : 18 ≤ age ∧ age ≤ 70)
  (product_condition : age * month * day = 2975)
  (valid_month : 1 ≤ month ∧ month ≤ 12)
  (valid_day : 1 ≤ day ∧ day ≤ 31)
  : age = 35 := sorry

end NUMINAMATH_GPT_LimingFatherAge_l349_34998


namespace NUMINAMATH_GPT_part1_part2_l349_34975

def setA := {x : ℝ | -3 < x ∧ x < 4}
def setB (a : ℝ) := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 = 0}

theorem part1 (a : ℝ) : (setA ∩ setB a = ∅) ↔ (a ≤ -3 ∨ a ≥ 4) :=
sorry

theorem part2 (a : ℝ) : (setA ∪ setB a = setA) ↔ (-1 < a ∧ a < 4/3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l349_34975


namespace NUMINAMATH_GPT_find_non_equivalent_fraction_l349_34961

-- Define the fractions mentioned in the problem
def sevenSixths := 7 / 6
def optionA := 14 / 12
def optionB := 1 + 1 / 6
def optionC := 1 + 5 / 30
def optionD := 1 + 2 / 6
def optionE := 1 + 14 / 42

-- The main problem statement
theorem find_non_equivalent_fraction :
  optionD ≠ sevenSixths := by
  -- We put a 'sorry' here because we are not required to provide the proof
  sorry

end NUMINAMATH_GPT_find_non_equivalent_fraction_l349_34961


namespace NUMINAMATH_GPT_geometric_sequence_sum_2018_l349_34990

noncomputable def geometric_sum (n : ℕ) (a1 q : ℝ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_2018 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, S n = geometric_sum n (a 1) 2) →
    a 1 = 1 / 2 →
    (a 1 * 2^2)^2 = 8 * a 1 * 2^3 - 16 →
    S 2018 = 2^2017 - 1 / 2 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sum_2018_l349_34990


namespace NUMINAMATH_GPT_remaining_paint_needed_l349_34906

-- Define the conditions
def total_paint_needed : ℕ := 70
def paint_bought : ℕ := 23
def paint_already_have : ℕ := 36

-- Lean theorem statement
theorem remaining_paint_needed : (total_paint_needed - (paint_already_have + paint_bought)) = 11 := by
  sorry

end NUMINAMATH_GPT_remaining_paint_needed_l349_34906


namespace NUMINAMATH_GPT_find_g2_l349_34940

-- Define the conditions of the problem
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x

-- Prove the desired value of g(2)
theorem find_g2 (g : ℝ → ℝ) (h : satisfies_condition g) : g 2 = -19 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_g2_l349_34940


namespace NUMINAMATH_GPT_trishul_investment_less_than_raghu_l349_34907

noncomputable def VishalInvestment (T : ℝ) : ℝ := 1.10 * T

noncomputable def TotalInvestment (T : ℝ) (R : ℝ) : ℝ :=
  T + VishalInvestment T + R

def RaghuInvestment : ℝ := 2100

def TotalSumInvested : ℝ := 6069

theorem trishul_investment_less_than_raghu :
  ∃ T : ℝ, TotalInvestment T RaghuInvestment = TotalSumInvested → (RaghuInvestment - T) / RaghuInvestment * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_trishul_investment_less_than_raghu_l349_34907


namespace NUMINAMATH_GPT_roots_exist_for_all_K_l349_34955

theorem roots_exist_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  -- Applied conditions and approach
  sorry

end NUMINAMATH_GPT_roots_exist_for_all_K_l349_34955


namespace NUMINAMATH_GPT_expression_evaluation_l349_34991

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = -1 ^ 2023) :
  (2 * m + n) * (2 * m - n) - (2 * m - n) ^ 2 + 2 * n * (m + n) = -12 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l349_34991


namespace NUMINAMATH_GPT_peter_total_food_l349_34943

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end NUMINAMATH_GPT_peter_total_food_l349_34943


namespace NUMINAMATH_GPT_deschamps_cows_l349_34933

theorem deschamps_cows (p v : ℕ) (h1 : p + v = 160) (h2 : 2 * p + 4 * v = 400) : v = 40 :=
by sorry

end NUMINAMATH_GPT_deschamps_cows_l349_34933


namespace NUMINAMATH_GPT_space_between_trees_l349_34956

theorem space_between_trees (n_trees : ℕ) (tree_space : ℕ) (total_length : ℕ) (spaces_between_trees : ℕ) (result_space : ℕ) 
  (h1 : n_trees = 8)
  (h2 : tree_space = 1)
  (h3 : total_length = 148)
  (h4 : spaces_between_trees = n_trees - 1)
  (h5 : result_space = (total_length - n_trees * tree_space) / spaces_between_trees) : 
  result_space = 20 := 
by sorry

end NUMINAMATH_GPT_space_between_trees_l349_34956


namespace NUMINAMATH_GPT_sufficient_condition_transitive_l349_34974

theorem sufficient_condition_transitive
  (C B A : Prop) (h1 : (C → B)) (h2 : (B → A)) : (C → A) :=
  sorry

end NUMINAMATH_GPT_sufficient_condition_transitive_l349_34974


namespace NUMINAMATH_GPT_cupcakes_initial_count_l349_34942

theorem cupcakes_initial_count (x : ℕ) (h1 : x - 5 + 10 = 24) : x = 19 :=
by sorry

end NUMINAMATH_GPT_cupcakes_initial_count_l349_34942


namespace NUMINAMATH_GPT_x0_y0_sum_eq_31_l349_34908

theorem x0_y0_sum_eq_31 :
  ∃ x0 y0 : ℕ, (0 ≤ x0 ∧ x0 < 37) ∧ (0 ≤ y0 ∧ y0 < 37) ∧ 
  (2 * x0 ≡ 1 [MOD 37]) ∧ (3 * y0 ≡ 36 [MOD 37]) ∧ 
  (x0 + y0 = 31) :=
sorry

end NUMINAMATH_GPT_x0_y0_sum_eq_31_l349_34908


namespace NUMINAMATH_GPT_pure_imaginary_condition_l349_34913

variable (a : ℝ)

def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition :
  isPureImaginary (a - 17 / (4 - (i : ℂ))) → a = 4 := 
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_condition_l349_34913


namespace NUMINAMATH_GPT_bus_weight_conversion_l349_34953

noncomputable def round_to_nearest (x : ℚ) : ℤ := Int.floor (x + 0.5)

theorem bus_weight_conversion (kg_to_pound : ℚ) (bus_weight_kg : ℚ) 
  (h : kg_to_pound = 0.4536) (h_bus : bus_weight_kg = 350) : 
  round_to_nearest (bus_weight_kg / kg_to_pound) = 772 := by
  sorry

end NUMINAMATH_GPT_bus_weight_conversion_l349_34953


namespace NUMINAMATH_GPT_selling_price_before_brokerage_l349_34970

theorem selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) (final_cash : ℝ) : 
  final_cash = 104.25 → brokerage_rate = 1 / 400 → cash_realized = 104.51 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_selling_price_before_brokerage_l349_34970


namespace NUMINAMATH_GPT_dilution_plate_count_lower_than_actual_l349_34972

theorem dilution_plate_count_lower_than_actual
  (bacteria_count : ℕ)
  (colony_count : ℕ)
  (dilution_factor : ℕ)
  (plate_count : ℕ)
  (count_error_margin : ℕ)
  (method_estimation_error : ℕ)
  (H1 : method_estimation_error > 0)
  (H2 : colony_count = bacteria_count / dilution_factor - method_estimation_error)
  : colony_count < bacteria_count :=
by
  sorry

end NUMINAMATH_GPT_dilution_plate_count_lower_than_actual_l349_34972


namespace NUMINAMATH_GPT_Emir_needs_more_money_l349_34958

theorem Emir_needs_more_money
  (cost_dictionary : ℝ)
  (cost_dinosaur_book : ℝ)
  (cost_cookbook : ℝ)
  (cost_science_kit : ℝ)
  (cost_colored_pencils : ℝ)
  (saved_amount : ℝ)
  (total_cost : ℝ := cost_dictionary + cost_dinosaur_book + cost_cookbook + cost_science_kit + cost_colored_pencils)
  (more_money_needed : ℝ := total_cost - saved_amount) :
  cost_dictionary = 5.50 →
  cost_dinosaur_book = 11.25 →
  cost_cookbook = 5.75 →
  cost_science_kit = 8.40 →
  cost_colored_pencils = 3.60 →
  saved_amount = 24.50 →
  more_money_needed = 10.00 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_Emir_needs_more_money_l349_34958


namespace NUMINAMATH_GPT_population_of_missing_village_l349_34917

theorem population_of_missing_village 
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ) 
  (avg_pop : ℕ) 
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1023)
  (h4 : pop4 = 945)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000) :
  ∃ (pop_missing : ℕ), pop_missing = 1100 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_population_of_missing_village_l349_34917


namespace NUMINAMATH_GPT_evaluate_expression_l349_34949

theorem evaluate_expression :
  (3 ^ 1002 + 7 ^ 1003) ^ 2 - (3 ^ 1002 - 7 ^ 1003) ^ 2 = 56 * 10 ^ 1003 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l349_34949


namespace NUMINAMATH_GPT_range_of_a_l349_34903

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, 1 < x → 2 * a * Real.log x ≤ 2 * x^2 + f a (2 * x - 1)) :
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l349_34903


namespace NUMINAMATH_GPT_smallest_value_of_n_l349_34929

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_n_l349_34929


namespace NUMINAMATH_GPT_apartments_per_floor_l349_34988

theorem apartments_per_floor (floors apartments_per: ℕ) (total_people : ℕ) (each_apartment_houses : ℕ)
    (h1 : floors = 25)
    (h2 : each_apartment_houses = 2)
    (h3 : total_people = 200)
    (h4 : floors * apartments_per * each_apartment_houses = total_people) :
    apartments_per = 4 := 
sorry

end NUMINAMATH_GPT_apartments_per_floor_l349_34988
