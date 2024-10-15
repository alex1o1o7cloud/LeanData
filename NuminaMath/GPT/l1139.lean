import Mathlib

namespace NUMINAMATH_GPT_value_of_one_TV_mixer_blender_l1139_113932

variables (M T B : ℝ)

-- The given conditions
def eq1 : Prop := 2 * M + T + B = 10500
def eq2 : Prop := T + M + 2 * B = 14700

-- The problem: find the combined value of one TV, one mixer, and one blender
theorem value_of_one_TV_mixer_blender :
  eq1 M T B → eq2 M T B → (T + M + B = 18900) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_one_TV_mixer_blender_l1139_113932


namespace NUMINAMATH_GPT_cow_difference_l1139_113965

variables (A M R : Nat)

def Aaron_has_four_times_as_many_cows_as_Matthews : Prop := A = 4 * M
def Matthews_has_cows : Prop := M = 60
def Total_cows_for_three := A + M + R = 570

theorem cow_difference (h1 : Aaron_has_four_times_as_many_cows_as_Matthews A M) 
                       (h2 : Matthews_has_cows M)
                       (h3 : Total_cows_for_three A M R) :
  (A + M) - R = 30 :=
by
  sorry

end NUMINAMATH_GPT_cow_difference_l1139_113965


namespace NUMINAMATH_GPT_distribution_scheme_count_l1139_113925

-- Definitions based on conditions
variable (village1 village2 village3 village4 : Type)
variables (quota1 quota2 quota3 quota4 : ℕ)

-- Conditions as given in the problem
def valid_distribution (v1 v2 v3 v4 : ℕ) : Prop :=
  v1 = 1 ∧ v2 = 2 ∧ v3 = 3 ∧ v4 = 4

-- The goal is to prove the number of permutations is equal to 24
theorem distribution_scheme_count :
  (∃ v1 v2 v3 v4 : ℕ, valid_distribution v1 v2 v3 v4) → 
  (4 * 3 * 2 * 1 = 24) :=
by 
  sorry

end NUMINAMATH_GPT_distribution_scheme_count_l1139_113925


namespace NUMINAMATH_GPT_problem_statement_l1139_113964

theorem problem_statement (n : ℤ) (h_odd: Odd n) (h_pos: n > 0) (h_not_divisible_by_3: ¬(3 ∣ n)) : 24 ∣ (n^2 - 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1139_113964


namespace NUMINAMATH_GPT_correct_ratio_l1139_113963

theorem correct_ratio (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_ratio_l1139_113963


namespace NUMINAMATH_GPT_find_triangle_lengths_l1139_113954

-- Conditions:
-- 1. Two right-angled triangles are similar.
-- 2. Bigger triangle sides: x + 1 and y + 5, Area larger by 8 cm^2

def triangle_lengths (x y : ℝ) : Prop := 
  (y = 5 * x ∧ 
  (5 / 2) * (x + 1) ^ 2 - (5 / 2) * x ^ 2 = 8)

theorem find_triangle_lengths (x y : ℝ) : triangle_lengths x y ↔ (x = 1.1 ∧ y = 5.5) :=
sorry

end NUMINAMATH_GPT_find_triangle_lengths_l1139_113954


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_l1139_113958

def Point := (ℝ × ℝ)

def isMidpoint (M C D : Point) : Prop :=
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem sum_of_coordinates_of_D (M C : Point) (D : Point) (hM : isMidpoint M C D) (hC : C = (2, 10)) :
  D.1 + D.2 = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_l1139_113958


namespace NUMINAMATH_GPT_cosine_squared_identity_l1139_113946

theorem cosine_squared_identity (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) : Real.cos (α - (π / 4)) ^ 2 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_cosine_squared_identity_l1139_113946


namespace NUMINAMATH_GPT_number_of_terms_in_expansion_l1139_113940

def first_factor : List Char := ['x', 'y']
def second_factor : List Char := ['u', 'v', 'w', 'z', 's']

theorem number_of_terms_in_expansion :
  first_factor.length * second_factor.length = 10 :=
by
  -- Lean expects a proof here, but the problem statement specifies to use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_number_of_terms_in_expansion_l1139_113940


namespace NUMINAMATH_GPT_squared_greater_abs_greater_l1139_113967

theorem squared_greater_abs_greater {a b : ℝ} : a^2 > b^2 ↔ |a| > |b| :=
by sorry

end NUMINAMATH_GPT_squared_greater_abs_greater_l1139_113967


namespace NUMINAMATH_GPT_cubic_polynomials_integer_roots_l1139_113992

theorem cubic_polynomials_integer_roots (a b : ℤ) :
  (∀ α1 α2 α3 : ℤ, α1 + α2 + α3 = 0 ∧ α1 * α2 + α2 * α3 + α3 * α1 = a ∧ α1 * α2 * α3 = -b) →
  (∀ β1 β2 β3 : ℤ, β1 + β2 + β3 = 0 ∧ β1 * β2 + β2 * β3 + β3 * β1 = b ∧ β1 * β2 * β3 = -a) →
  a = 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomials_integer_roots_l1139_113992


namespace NUMINAMATH_GPT_total_chickens_l1139_113973

open Nat

theorem total_chickens 
  (Q S C : ℕ) 
  (h1 : Q = 2 * S + 25) 
  (h2 : S = 3 * C - 4) 
  (h3 : C = 37) : 
  Q + S + C = 383 := by
  sorry

end NUMINAMATH_GPT_total_chickens_l1139_113973


namespace NUMINAMATH_GPT_carrey_fixed_amount_l1139_113912

theorem carrey_fixed_amount :
  ∃ C : ℝ, 
    (C + 0.25 * 44.44444444444444 = 24 + 0.16 * 44.44444444444444) →
    C = 20 :=
by
  sorry

end NUMINAMATH_GPT_carrey_fixed_amount_l1139_113912


namespace NUMINAMATH_GPT_unique_zero_property_l1139_113952

theorem unique_zero_property (x : ℝ) (h1 : ∀ a : ℝ, x * a = x) (h2 : ∀ (a : ℝ), a ≠ 0 → x / a = x) :
  x = 0 :=
sorry

end NUMINAMATH_GPT_unique_zero_property_l1139_113952


namespace NUMINAMATH_GPT_persimmons_count_l1139_113986

theorem persimmons_count (x : ℕ) (h : x - 5 = 12) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_persimmons_count_l1139_113986


namespace NUMINAMATH_GPT_simplify_expression_l1139_113976

theorem simplify_expression (x : ℝ) : 3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1139_113976


namespace NUMINAMATH_GPT_solve_for_x_l1139_113909

theorem solve_for_x (x : ℝ) (h : (x - 75) / 3 = (8 - 3 * x) / 4) : 
  x = 324 / 13 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1139_113909


namespace NUMINAMATH_GPT_pieces_per_box_l1139_113903

theorem pieces_per_box (boxes : ℕ) (total_pieces : ℕ) (h_boxes : boxes = 7) (h_total : total_pieces = 21) : 
  total_pieces / boxes = 3 :=
by
  sorry

end NUMINAMATH_GPT_pieces_per_box_l1139_113903


namespace NUMINAMATH_GPT_prime_not_fourth_power_l1139_113928

theorem prime_not_fourth_power (p : ℕ) (hp : p > 5) (prime : Prime p) : 
  ¬ ∃ a : ℕ, p = a^4 + 4 :=
by
  sorry

end NUMINAMATH_GPT_prime_not_fourth_power_l1139_113928


namespace NUMINAMATH_GPT_rhombus_diagonal_l1139_113984

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) 
  (h_d1 : d1 = 70) 
  (h_area : area = 5600): 
  (area = (d1 * d2) / 2) → d2 = 160 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1139_113984


namespace NUMINAMATH_GPT_range_of_function_l1139_113915

theorem range_of_function : 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 12) ∧ 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 18) ∧ 
  (∀ y : ℝ, (12 ≤ y ∧ y ≤ 18) → 
    ∃ x : ℝ, y = |x + 5| - |x - 3| + 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1139_113915


namespace NUMINAMATH_GPT_gcf_450_144_l1139_113917

theorem gcf_450_144 : Nat.gcd 450 144 = 18 := by
  sorry

end NUMINAMATH_GPT_gcf_450_144_l1139_113917


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1139_113977

theorem solution_set_of_inequality (x : ℝ) : 3 * x - 7 ≤ 2 → x ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1139_113977


namespace NUMINAMATH_GPT_sqrt_57_in_range_l1139_113956

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end NUMINAMATH_GPT_sqrt_57_in_range_l1139_113956


namespace NUMINAMATH_GPT_biased_coin_probability_l1139_113914

theorem biased_coin_probability :
  let P1 := 3 / 4
  let P2 := 1 / 2
  let P3 := 3 / 4
  let P4 := 2 / 3
  let P5 := 1 / 3
  let P6 := 2 / 5
  let P7 := 3 / 7
  P1 * P2 * P3 * P4 * P5 * P6 * P7 = 3 / 560 :=
by sorry

end NUMINAMATH_GPT_biased_coin_probability_l1139_113914


namespace NUMINAMATH_GPT_graph_equation_l1139_113920

theorem graph_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end NUMINAMATH_GPT_graph_equation_l1139_113920


namespace NUMINAMATH_GPT_program_selection_count_l1139_113985

theorem program_selection_count :
  let courses := ["English", "Algebra", "Geometry", "History", "Science", "Art", "Latin"]
  let english := 1
  let math_courses := ["Algebra", "Geometry"]
  let science_courses := ["Science"]
  ∃ (programs : Finset (Finset String)) (count : ℕ),
    (count = 9) ∧
    (programs.card = count) ∧
    ∀ p ∈ programs,
      "English" ∈ p ∧
      (∃ m ∈ p, m ∈ math_courses) ∧
      (∃ s ∈ p, s ∈ science_courses) ∧
      p.card = 5 :=
sorry

end NUMINAMATH_GPT_program_selection_count_l1139_113985


namespace NUMINAMATH_GPT_profit_percentage_l1139_113906

def cost_price : ℝ := 60
def selling_price : ℝ := 78

theorem profit_percentage : ((selling_price - cost_price) / cost_price) * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1139_113906


namespace NUMINAMATH_GPT_smallest_w_value_l1139_113902

theorem smallest_w_value (x y z w : ℝ) 
    (hx : -2 ≤ x ∧ x ≤ 5) 
    (hy : -3 ≤ y ∧ y ≤ 7) 
    (hz : 4 ≤ z ∧ z ≤ 8) 
    (hw : w = x * y - z) : 
    w ≥ -23 :=
sorry

end NUMINAMATH_GPT_smallest_w_value_l1139_113902


namespace NUMINAMATH_GPT_total_courses_l1139_113961

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end NUMINAMATH_GPT_total_courses_l1139_113961


namespace NUMINAMATH_GPT_y_intercept_of_linear_function_l1139_113935

theorem y_intercept_of_linear_function 
  (k : ℝ)
  (h : (∃ k: ℝ, ∀ x y: ℝ, y = k * (x - 1) ∧ (x, y) = (-1, -2))) : 
  ∃ y : ℝ, (0, y) = (0, -1) :=
by {
  -- Skipping the proof as per the instruction
  sorry
}

end NUMINAMATH_GPT_y_intercept_of_linear_function_l1139_113935


namespace NUMINAMATH_GPT_smaller_angle_36_degrees_l1139_113960

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_36_degrees_l1139_113960


namespace NUMINAMATH_GPT_probability_same_flips_l1139_113975

-- Define the probability of getting the first head on the nth flip
def prob_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (1 / 2) ^ n

-- Define the probability that all three get the first head on the nth flip
def prob_all_three_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (prob_first_head_on_nth_flip n) ^ 3

-- Define the total probability considering all n
noncomputable def total_prob_all_three_same_flips : ℚ :=
  ∑' n, prob_all_three_first_head_on_nth_flip (n + 1)

-- The statement to prove
theorem probability_same_flips : total_prob_all_three_same_flips = 1 / 7 :=
by sorry

end NUMINAMATH_GPT_probability_same_flips_l1139_113975


namespace NUMINAMATH_GPT_book_selection_l1139_113974

theorem book_selection :
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  tier1 + tier2 + tier3 = 16 :=
by
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  sorry

end NUMINAMATH_GPT_book_selection_l1139_113974


namespace NUMINAMATH_GPT_rangeOfA_l1139_113990

theorem rangeOfA (a : ℝ) : 
  (∃ x : ℝ, 9^x + a * 3^x + 4 = 0) → a ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_rangeOfA_l1139_113990


namespace NUMINAMATH_GPT_Tanya_accompanied_two_l1139_113901

-- Define the number of songs sung by each girl
def Anya_songs : ℕ := 8
def Tanya_songs : ℕ := 6
def Olya_songs : ℕ := 3
def Katya_songs : ℕ := 7

-- Assume each song is sung by three girls
def total_songs : ℕ := (Anya_songs + Tanya_songs + Olya_songs + Katya_songs) / 3

-- Define the number of times Tanya accompanied
def Tanya_accompanied : ℕ := total_songs - Tanya_songs

-- Prove that Tanya accompanied 2 times
theorem Tanya_accompanied_two : Tanya_accompanied = 2 :=
by sorry

end NUMINAMATH_GPT_Tanya_accompanied_two_l1139_113901


namespace NUMINAMATH_GPT_non_neg_int_solutions_inequality_l1139_113922

theorem non_neg_int_solutions_inequality :
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_inequality_l1139_113922


namespace NUMINAMATH_GPT_child_support_owed_l1139_113921

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end NUMINAMATH_GPT_child_support_owed_l1139_113921


namespace NUMINAMATH_GPT_total_profit_is_correct_l1139_113991

-- Definitions for the investments and profit shares
def x_investment : ℕ := 5000
def y_investment : ℕ := 15000
def x_share_of_profit : ℕ := 400

-- The theorem states that the total profit is Rs. 1600 given the conditions
theorem total_profit_is_correct (h1 : x_share_of_profit = 400) (h2 : x_investment = 5000) (h3 : y_investment = 15000) : 
  let y_share_of_profit := 3 * x_share_of_profit
  let total_profit := x_share_of_profit + y_share_of_profit
  total_profit = 1600 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_is_correct_l1139_113991


namespace NUMINAMATH_GPT_find_x_minus_y_l1139_113955

open Real

theorem find_x_minus_y (x y : ℝ) (h : (sin x ^ 2 - cos x ^ 2 + cos x ^ 2 * cos y ^ 2 - sin x ^ 2 * sin y ^ 2) / sin (x + y) = 1) :
  ∃ k : ℤ, x - y = π / 2 + 2 * k * π :=
by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l1139_113955


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1139_113996

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1139_113996


namespace NUMINAMATH_GPT_taylor_scores_l1139_113916

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_taylor_scores_l1139_113916


namespace NUMINAMATH_GPT_evaluate_at_neg_one_l1139_113972

def f (x : ℝ) : ℝ := -2 * x ^ 2 + 1

theorem evaluate_at_neg_one : f (-1) = -1 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_at_neg_one_l1139_113972


namespace NUMINAMATH_GPT_primes_less_or_equal_F_l1139_113929

-- Definition of F_n
def F (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- The main theorem statement
theorem primes_less_or_equal_F (n : ℕ) : ∃ S : Finset ℕ, S.card ≥ n + 1 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ≤ F n := 
sorry

end NUMINAMATH_GPT_primes_less_or_equal_F_l1139_113929


namespace NUMINAMATH_GPT_alice_always_wins_l1139_113971

theorem alice_always_wins (n : ℕ) (initial_coins : ℕ) (alice_first_move : ℕ) (total_coins : ℕ) :
  initial_coins = 1331 → alice_first_move = 1 → total_coins = 1331 →
  (∀ (k : ℕ), 
    let alice_total := (k * (k + 1)) / 2;
    let basilio_min_total := (k * (k - 1)) / 2;
    let basilio_max_total := (k * (k + 1)) / 2 - 1;
    k * k ≤ total_coins ∧ total_coins ≤ k * (k + 1) - 1 →
    ¬ (total_coins = k * k + k - 1 ∨ total_coins = k * (k + 1) - 1)) →
  alice_first_move = 1 ∧ initial_coins = 1331 ∧ total_coins = 1331 → alice_wins :=
sorry

end NUMINAMATH_GPT_alice_always_wins_l1139_113971


namespace NUMINAMATH_GPT_dogs_in_school_l1139_113980

theorem dogs_in_school
  (sit: ℕ) (sit_and_stay: ℕ) (stay: ℕ) (stay_and_roll_over: ℕ)
  (roll_over: ℕ) (sit_and_roll_over: ℕ) (all_three: ℕ) (none: ℕ)
  (h1: sit = 50) (h2: sit_and_stay = 17) (h3: stay = 29)
  (h4: stay_and_roll_over = 12) (h5: roll_over = 34)
  (h6: sit_and_roll_over = 18) (h7: all_three = 9) (h8: none = 9) :
  sit + stay + roll_over + sit_and_stay + stay_and_roll_over + sit_and_roll_over - 2 * all_three + none = 84 :=
by sorry

end NUMINAMATH_GPT_dogs_in_school_l1139_113980


namespace NUMINAMATH_GPT_distinct_pairs_reciprocal_sum_l1139_113926

theorem distinct_pairs_reciprocal_sum : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ (m n : ℕ), ((m, n) ∈ S) ↔ (m > 0 ∧ n > 0 ∧ (1/m + 1/n = 1/5))) ∧ S.card = 3 :=
sorry

end NUMINAMATH_GPT_distinct_pairs_reciprocal_sum_l1139_113926


namespace NUMINAMATH_GPT_part1_part2_l1139_113927

def A := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x + 1 - m^2 ≤ 0}

theorem part1 (m : ℝ) (hm : m = 2) :
  A ∩ {x : ℝ | x < -1 ∨ 3 < x} = {x : ℝ | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} :=
sorry

theorem part2 :
  (∀ x, x ∈ A → x ∈ B (m : ℝ)) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1139_113927


namespace NUMINAMATH_GPT_square_side_length_l1139_113951

/-- Define OPEN as a square and T a point on side NO
    such that the areas of triangles TOP and TEN are 
    respectively 62 and 10. Prove that the side length 
    of the square is 12. -/
theorem square_side_length (s x y : ℝ) (T : x + y = s)
    (h1 : 0 < s) (h2 : 0 < x) (h3 : 0 < y)
    (a1 : 1 / 2 * x * s = 62)
    (a2 : 1 / 2 * y * s = 10) :
    s = 12 :=
by
    sorry

end NUMINAMATH_GPT_square_side_length_l1139_113951


namespace NUMINAMATH_GPT_find_borrowed_amount_l1139_113998

noncomputable def borrowed_amount (P : ℝ) : Prop :=
  let interest_paid := P * (4 / 100) * 2
  let interest_earned := P * (6 / 100) * 2
  let total_gain := 120 * 2
  interest_earned - interest_paid = total_gain

theorem find_borrowed_amount : ∃ P : ℝ, borrowed_amount P ∧ P = 3000 :=
by
  use 3000
  unfold borrowed_amount
  simp
  sorry

end NUMINAMATH_GPT_find_borrowed_amount_l1139_113998


namespace NUMINAMATH_GPT_no_ghost_not_multiple_of_p_l1139_113994

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence_S (p : ℕ) (S : ℕ → ℕ) : Prop :=
  (is_prime p ∧ p % 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i < p → S i = i) ∧
  (∀ n, n ≥ p → (S n > S (n-1) ∧ 
    ∀ (a b c : ℕ), (a < b ∧ b < c ∧ c < n ∧ S a < S b ∧ S b < S c ∧
    S b - S a = S c - S b → false)))

def is_ghost (p : ℕ) (S : ℕ → ℕ) (g : ℕ) : Prop :=
  ∀ n : ℕ, S n ≠ g

theorem no_ghost_not_multiple_of_p (p : ℕ) (S : ℕ → ℕ) :
  (is_prime p ∧ p % 2 = 1) ∧ sequence_S p S → 
  ∀ g : ℕ, is_ghost p S g → p ∣ g :=
by 
  sorry

end NUMINAMATH_GPT_no_ghost_not_multiple_of_p_l1139_113994


namespace NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_8_l1139_113919

theorem remainder_of_sum_of_squares_mod_8 :
  let a := 445876
  let b := 985420
  let c := 215546
  let d := 656452
  let e := 387295
  a % 8 = 4 → b % 8 = 4 → c % 8 = 6 → d % 8 = 4 → e % 8 = 7 →
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_8_l1139_113919


namespace NUMINAMATH_GPT_same_color_probability_is_correct_l1139_113953

-- Define the variables and conditions
def total_sides : ℕ := 12
def pink_sides : ℕ := 3
def green_sides : ℕ := 4
def blue_sides : ℕ := 5

-- Calculate individual probabilities
def pink_probability : ℚ := (pink_sides : ℚ) / total_sides
def green_probability : ℚ := (green_sides : ℚ) / total_sides
def blue_probability : ℚ := (blue_sides : ℚ) / total_sides

-- Calculate the probabilities that both dice show the same color
def both_pink_probability : ℚ := pink_probability ^ 2
def both_green_probability : ℚ := green_probability ^ 2
def both_blue_probability : ℚ := blue_probability ^ 2

-- The final probability that both dice come up the same color
def same_color_probability : ℚ := both_pink_probability + both_green_probability + both_blue_probability

theorem same_color_probability_is_correct : same_color_probability = 25 / 72 := by
  sorry

end NUMINAMATH_GPT_same_color_probability_is_correct_l1139_113953


namespace NUMINAMATH_GPT_calculate_perimeter_l1139_113978

def four_squares_area : ℝ := 144 -- total area of the figure in cm²
noncomputable def area_of_one_square : ℝ := four_squares_area / 4 -- area of one square in cm²
noncomputable def side_length_of_square : ℝ := Real.sqrt area_of_one_square -- side length of one square in cm

def number_of_vertical_segments : ℕ := 4 -- based on the arrangement
def number_of_horizontal_segments : ℕ := 6 -- based on the arrangement

noncomputable def total_perimeter : ℝ := (number_of_vertical_segments + number_of_horizontal_segments) * side_length_of_square

theorem calculate_perimeter : total_perimeter = 60 := by
  sorry

end NUMINAMATH_GPT_calculate_perimeter_l1139_113978


namespace NUMINAMATH_GPT_horse_total_value_l1139_113934

theorem horse_total_value (n : ℕ) (a r : ℕ) (h₁ : n = 32) (h₂ : a = 1) (h₃ : r = 2) :
  (a * (r ^ n - 1) / (r - 1)) = 4294967295 :=
by 
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_horse_total_value_l1139_113934


namespace NUMINAMATH_GPT_inequality_solution_l1139_113900

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l1139_113900


namespace NUMINAMATH_GPT_problem_l1139_113931

theorem problem (a b : ℤ) (h1 : |a - 2| = 5) (h2 : |b| = 9) (h3 : a + b < 0) :
  a - b = 16 ∨ a - b = 6 := 
sorry

end NUMINAMATH_GPT_problem_l1139_113931


namespace NUMINAMATH_GPT_find_triangle_angles_l1139_113945

theorem find_triangle_angles (α β γ : ℝ)
  (h1 : (180 - α) / (180 - β) = 13 / 9)
  (h2 : β - α = 45)
  (h3 : α + β + γ = 180) :
  (α = 33.75) ∧ (β = 78.75) ∧ (γ = 67.5) :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_angles_l1139_113945


namespace NUMINAMATH_GPT_calories_per_pound_of_body_fat_l1139_113969

theorem calories_per_pound_of_body_fat (gained_weight : ℕ) (calories_burned_per_day : ℕ) 
  (days_to_lose_weight : ℕ) (calories_consumed_per_day : ℕ) : 
  gained_weight = 5 → 
  calories_burned_per_day = 2500 → 
  days_to_lose_weight = 35 → 
  calories_consumed_per_day = 2000 → 
  (calories_burned_per_day * days_to_lose_weight - calories_consumed_per_day * days_to_lose_weight) / gained_weight = 3500 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_calories_per_pound_of_body_fat_l1139_113969


namespace NUMINAMATH_GPT_trig_identity_l1139_113949

theorem trig_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (α - 15 * Real.pi / 180) + Real.cos (105 * Real.pi / 180 - α) = -2 / 3 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1139_113949


namespace NUMINAMATH_GPT_repair_cost_l1139_113966

variable (R : ℝ)

theorem repair_cost (purchase_price transportation_charges profit_rate selling_price : ℝ) (h1 : purchase_price = 12000) (h2 : transportation_charges = 1000) (h3 : profit_rate = 0.5) (h4 : selling_price = 27000) :
  R = 5000 :=
by
  have total_cost := purchase_price + R + transportation_charges
  have selling_price_eq := 1.5 * total_cost
  have sp_eq_27000 := selling_price = 27000
  sorry

end NUMINAMATH_GPT_repair_cost_l1139_113966


namespace NUMINAMATH_GPT_system_solution_l1139_113936

theorem system_solution :
  ∃ x y : ℝ, (3 * x + y = 11 ∧ x - y = 1) ∧ (x = 3 ∧ y = 2) := 
by
  sorry

end NUMINAMATH_GPT_system_solution_l1139_113936


namespace NUMINAMATH_GPT_tangents_product_is_constant_MN_passes_fixed_point_l1139_113910

-- Define the parabola C and the tangency conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

variables {x1 y1 x2 y2 : ℝ}

-- Point G is on the axis of the parabola C (we choose the y-axis for part 2)
def point_G_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Two tangent points from G to the parabola at A (x1, y1) and B (x2, y2)
def tangent_points (G : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂

-- Question 1 proof statement
theorem tangents_product_is_constant (G : ℝ × ℝ) (hG : point_G_on_axis G)
  (hT : tangent_points G x1 y1 x2 y2) : x1 * x2 + y1 * y2 = -3 := sorry

variables {M N : ℝ × ℝ}

-- Question 2 proof statement
theorem MN_passes_fixed_point {G : ℝ × ℝ} (hG : G.1 = 0) (xM yM xN yN : ℝ)
 (hMA : parabola M.1 M.2) (hMB : parabola N.1 N.2)
 (h_perpendicular : (M.1 - G.1) * (N.1 - G.1) + (M.2 - G.2) * (N.2 - G.2) = 0)
 : ∃ P, P = (2, 5) := sorry

end NUMINAMATH_GPT_tangents_product_is_constant_MN_passes_fixed_point_l1139_113910


namespace NUMINAMATH_GPT_gcd_of_28430_and_39674_l1139_113904

theorem gcd_of_28430_and_39674 : Nat.gcd 28430 39674 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_28430_and_39674_l1139_113904


namespace NUMINAMATH_GPT_number_of_pumpkin_pies_l1139_113968

-- Definitions for the conditions
def apple_pies : ℕ := 2
def pecan_pies : ℕ := 4
def total_pies : ℕ := 13

-- The proof statement
theorem number_of_pumpkin_pies
  (h_apple : apple_pies = 2)
  (h_pecan : pecan_pies = 4)
  (h_total : total_pies = 13) : 
  total_pies - (apple_pies + pecan_pies) = 7 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_pumpkin_pies_l1139_113968


namespace NUMINAMATH_GPT_wax_total_is_correct_l1139_113988

-- Define the given conditions
def current_wax : ℕ := 20
def additional_wax : ℕ := 146

-- The total amount of wax required is the sum of current_wax and additional_wax
def total_wax := current_wax + additional_wax

-- The proof goal is to show that the total_wax equals 166 grams
theorem wax_total_is_correct : total_wax = 166 := by
  sorry

end NUMINAMATH_GPT_wax_total_is_correct_l1139_113988


namespace NUMINAMATH_GPT_john_spends_on_memory_cards_l1139_113930

theorem john_spends_on_memory_cards :
  (10 * (3 * 365)) / 50 * 60 = 13140 :=
by
  sorry

end NUMINAMATH_GPT_john_spends_on_memory_cards_l1139_113930


namespace NUMINAMATH_GPT_find_a61_l1139_113979

def seq (a : ℕ → ℕ) : Prop :=
  (∀ n, a (2 * n + 1) = a n + a (n + 1)) ∧
  (∀ n, a (2 * n) = a n) ∧
  a 1 = 1

theorem find_a61 (a : ℕ → ℕ) (h : seq a) : a 61 = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_a61_l1139_113979


namespace NUMINAMATH_GPT_max_sum_x_y_min_diff_x_y_l1139_113939

def circle_points (x y : ℤ) : Prop := (x - 1)^2 + (y + 2)^2 = 36

theorem max_sum_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x + y ≥ x' + y') :=
  by sorry

theorem min_diff_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x - y ≤ x' - y') :=
  by sorry

end NUMINAMATH_GPT_max_sum_x_y_min_diff_x_y_l1139_113939


namespace NUMINAMATH_GPT_num_sets_satisfying_union_is_four_l1139_113941

variable (M : Set ℕ) (N : Set ℕ)

def num_sets_satisfying_union : Prop :=
  M = {1, 2} ∧ (M ∪ N = {1, 2, 6} → (N = {6} ∨ N = {1, 6} ∨ N = {2, 6} ∨ N = {1, 2, 6}))

theorem num_sets_satisfying_union_is_four :
  (∃ M : Set ℕ, M = {1, 2}) →
  (∃ N : Set ℕ, M ∪ N = {1, 2, 6}) →
  (∃ (num_sets : ℕ), num_sets = 4) :=
by
  sorry

end NUMINAMATH_GPT_num_sets_satisfying_union_is_four_l1139_113941


namespace NUMINAMATH_GPT_mn_value_l1139_113989

-- Definitions
def exponent_m := 2
def exponent_n := 2

-- Theorem statement
theorem mn_value : exponent_m * exponent_n = 4 :=
by
  sorry

end NUMINAMATH_GPT_mn_value_l1139_113989


namespace NUMINAMATH_GPT_cubic_poly_sum_l1139_113947

noncomputable def q (x : ℕ) : ℤ := sorry

axiom h0 : q 1 = 5
axiom h1 : q 6 = 24
axiom h2 : q 10 = 16
axiom h3 : q 15 = 34

theorem cubic_poly_sum :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) +
  (q 7) + (q 8) + (q 9) + (q 10) + (q 11) + (q 12) + (q 13) +
  (q 14) + (q 15) + (q 16) = 340 :=
by
  sorry

end NUMINAMATH_GPT_cubic_poly_sum_l1139_113947


namespace NUMINAMATH_GPT_min_expression_l1139_113983

theorem min_expression : ∀ x y : ℝ, ∃ x, 4 * x^2 + 4 * x * (Real.sin y) - (Real.cos y)^2 = -1 := by
  sorry

end NUMINAMATH_GPT_min_expression_l1139_113983


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1139_113957

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) :
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  e = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1139_113957


namespace NUMINAMATH_GPT_roots_quadratic_expression_l1139_113937

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + m - 2023 = 0) (h2 : n^2 + n - 2023 = 0) :
  m^2 + 2 * m + n = 2022 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_roots_quadratic_expression_l1139_113937


namespace NUMINAMATH_GPT_remainder_div_l1139_113938

theorem remainder_div (n : ℕ) : (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 + 
  90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 - 90^7 * Nat.choose 10 7 + 
  90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 + 90^10 * Nat.choose 10 10) % 88 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_div_l1139_113938


namespace NUMINAMATH_GPT_redistribution_amount_l1139_113913

theorem redistribution_amount
    (earnings : Fin 5 → ℕ)
    (h : earnings = ![18, 22, 30, 35, 45]) :
    (earnings 4 - ((earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_redistribution_amount_l1139_113913


namespace NUMINAMATH_GPT_smallest_odd_n_3_product_gt_5000_l1139_113943

theorem smallest_odd_n_3_product_gt_5000 :
  ∃ n : ℕ, (∃ k : ℤ, n = 2 * k + 1 ∧ n > 0) ∧ (3 ^ ((n + 1)^2 / 8)) > 5000 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_odd_n_3_product_gt_5000_l1139_113943


namespace NUMINAMATH_GPT_find_range_a_l1139_113981

-- Define the parabola equation y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line equation y = (√3/3) * (x - a)
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the focus of the parabola
def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the condition that F is outside the circle with diameter CD
def F_outside_circle_CD (x1 y1 x2 y2 a : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 > 0

-- Define the parabola-line intersection points and the related Vieta's formulas
def intersection_points (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 2 * a + 12 ∧ x1 * x2 = a^2

-- Define the final condition for a
def range_a (a : ℝ) : Prop :=
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3

-- Main theorem statement
theorem find_range_a (a : ℝ) (hneg : a < 0)
  (x1 x2 y1 y2 : ℝ)
  (hparabola1 : parabola x1 y1)
  (hparabola2 : parabola x2 y2)
  (hline1 : line x1 y1 a)
  (hline2 : line x2 y2 a)
  (hfocus : focus 1 0)
  (hF_out : F_outside_circle_CD x1 y1 x2 y2 a)
  (hintersect : intersection_points a x1 x2) :
  range_a a := 
sorry

end NUMINAMATH_GPT_find_range_a_l1139_113981


namespace NUMINAMATH_GPT_trip_time_difference_l1139_113950

theorem trip_time_difference (speed distance1 distance2 : ℕ) (h1 : speed > 0) (h2 : distance2 > distance1) 
  (h3 : speed = 60) (h4 : distance1 = 540) (h5 : distance2 = 570) : 
  (distance2 - distance1) / speed * 60 = 30 := 
by
  sorry

end NUMINAMATH_GPT_trip_time_difference_l1139_113950


namespace NUMINAMATH_GPT_find_original_manufacturing_cost_l1139_113905

noncomputable def originalManufacturingCost (P : ℝ) : ℝ := 0.70 * P

theorem find_original_manufacturing_cost (P : ℝ) (currentCost : ℝ) 
  (h1 : currentCost = 50) 
  (h2 : currentCost = P - 0.50 * P) : originalManufacturingCost P = 70 :=
by
  -- The actual proof steps would go here, but we'll add sorry for now
  sorry

end NUMINAMATH_GPT_find_original_manufacturing_cost_l1139_113905


namespace NUMINAMATH_GPT_g_triple_evaluation_l1139_113948

def g (x : ℤ) : ℤ := 
if x < 8 then x ^ 2 - 6 
else x - 15

theorem g_triple_evaluation :
  g (g (g 20)) = 4 :=
by sorry

end NUMINAMATH_GPT_g_triple_evaluation_l1139_113948


namespace NUMINAMATH_GPT_gcd_280_2155_l1139_113911

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := 
sorry

end NUMINAMATH_GPT_gcd_280_2155_l1139_113911


namespace NUMINAMATH_GPT_power_mod_1000_l1139_113933

theorem power_mod_1000 (N : ℤ) (h : Int.gcd N 10 = 1) : (N ^ 101 ≡ N [ZMOD 1000]) :=
  sorry

end NUMINAMATH_GPT_power_mod_1000_l1139_113933


namespace NUMINAMATH_GPT_janna_wrote_more_words_than_yvonne_l1139_113993

theorem janna_wrote_more_words_than_yvonne :
  ∃ (janna_words_written yvonne_words_written : ℕ), 
    yvonne_words_written = 400 ∧
    janna_words_written > yvonne_words_written ∧
    ∃ (removed_words added_words : ℕ),
      removed_words = 20 ∧
      added_words = 2 * removed_words ∧
      (janna_words_written + yvonne_words_written - removed_words + added_words + 30 = 1000) ∧
      (janna_words_written - yvonne_words_written = 130) :=
by
  sorry

end NUMINAMATH_GPT_janna_wrote_more_words_than_yvonne_l1139_113993


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1139_113908

theorem geometric_sequence_a4 {a_2 a_6 a_4 : ℝ} 
  (h1 : ∃ a_1 r : ℝ, a_2 = a_1 * r ∧ a_6 = a_1 * r^5) 
  (h2 : a_2 * a_6 = 64) 
  (h3 : a_2 = a_1 * r)
  (h4 : a_6 = a_1 * r^5)
  : a_4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1139_113908


namespace NUMINAMATH_GPT_relationship_among_sets_l1139_113959

-- Definitions based on the conditions
def RegularQuadrilateralPrism (x : Type) : Prop := -- prisms with a square base and perpendicular lateral edges
  sorry

def RectangularPrism (x : Type) : Prop := -- prisms with a rectangular base and perpendicular lateral edges
  sorry

def RightQuadrilateralPrism (x : Type) : Prop := -- prisms whose lateral edges are perpendicular to the base, and the base can be any quadrilateral
  sorry

def RightParallelepiped (x : Type) : Prop := -- prisms with lateral edges perpendicular to the base
  sorry

-- Sets
def M : Set Type := { x | RegularQuadrilateralPrism x }
def P : Set Type := { x | RectangularPrism x }
def N : Set Type := { x | RightQuadrilateralPrism x }
def Q : Set Type := { x | RightParallelepiped x }

-- Proof problem statement
theorem relationship_among_sets : M ⊂ P ∧ P ⊂ Q ∧ Q ⊂ N := 
  by
    sorry

end NUMINAMATH_GPT_relationship_among_sets_l1139_113959


namespace NUMINAMATH_GPT_fraction_covered_by_triangle_l1139_113924

structure Point where
  x : ℤ
  y : ℤ

def area_of_triangle (A B C : Point) : ℚ :=
  (1/2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_grid (length width : ℤ) : ℚ :=
  (length * width : ℚ)

def fraction_of_grid_covered (A B C : Point) (length width : ℤ) : ℚ :=
  (area_of_triangle A B C) / (area_of_grid length width)

theorem fraction_covered_by_triangle :
  fraction_of_grid_covered ⟨2, 4⟩ ⟨7, 2⟩ ⟨6, 5⟩ 8 6 = 13 / 96 :=
by
  sorry

end NUMINAMATH_GPT_fraction_covered_by_triangle_l1139_113924


namespace NUMINAMATH_GPT_probability_at_least_four_8s_in_five_rolls_l1139_113918

-- Definitions 
def prob_three_favorable : ℚ := 3 / 10

def prob_at_least_four_times_in_five_rolls : ℚ := 5 * (prob_three_favorable^4) * ((7 : ℚ)/10) + (prob_three_favorable)^5

-- The proof statement
theorem probability_at_least_four_8s_in_five_rolls : prob_at_least_four_times_in_five_rolls = 2859.3 / 10000 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_four_8s_in_five_rolls_l1139_113918


namespace NUMINAMATH_GPT_find_a_find_b_l1139_113999

section Problem1

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^4 - 4 * x^3 + a * x^2 - 1

-- Condition 1: f is monotonically increasing on [0, 1]
def f_increasing_on_interval_01 (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≤ y → f x a ≤ f y a

-- Condition 2: f is monotonically decreasing on [1, 2]
def f_decreasing_on_interval_12 (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x ≤ y → f y a ≤ f x a

-- Proof of a part
theorem find_a : ∃ a, f_increasing_on_interval_01 a ∧ f_decreasing_on_interval_12 a ∧ a = 4 :=
  sorry

end Problem1

section Problem2

noncomputable def f_fixed (x : ℝ) : ℝ := x^4 - 4 * x^3 + 4 * x^2 - 1
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x^2 - 1

-- Condition for intersections
def intersect_at_two_points (b : ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f_fixed x1 = g x1 b ∧ f_fixed x2 = g x2 b

-- Proof of b part
theorem find_b : ∃ b, intersect_at_two_points b ∧ (b = 0 ∨ b = 4) :=
  sorry

end Problem2

end NUMINAMATH_GPT_find_a_find_b_l1139_113999


namespace NUMINAMATH_GPT_sequence_a10_l1139_113944

theorem sequence_a10 : 
  (∃ (a : ℕ → ℤ), 
    a 1 = -1 ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n) - a (2*n - 1) = 2^(2*n-1)) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n + 1) - a (2*n) = 2^(2*n))) → 
  (∃ a : ℕ → ℤ, a 10 = 1021) :=
by
  intro h
  obtain ⟨a, h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_sequence_a10_l1139_113944


namespace NUMINAMATH_GPT_total_price_of_hats_l1139_113907

-- Declare the conditions as Lean definitions
def total_hats : Nat := 85
def green_hats : Nat := 38
def blue_hat_cost : Nat := 6
def green_hat_cost : Nat := 7

-- The question becomes proving the total cost of the hats is $548
theorem total_price_of_hats :
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  total_blue_cost + total_green_cost = 548 := by
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  show total_blue_cost + total_green_cost = 548
  sorry

end NUMINAMATH_GPT_total_price_of_hats_l1139_113907


namespace NUMINAMATH_GPT_original_price_l1139_113962

theorem original_price (P : ℝ) (h : P * 0.80 = 960) : P = 1200 :=
sorry

end NUMINAMATH_GPT_original_price_l1139_113962


namespace NUMINAMATH_GPT_regression_analysis_correct_statement_l1139_113942

variables (x : Type) (y : Type)

def is_deterministic (v : Type) : Prop := sorry -- A placeholder definition
def is_random (v : Type) : Prop := sorry -- A placeholder definition

theorem regression_analysis_correct_statement :
  (is_deterministic x) → (is_random y) →
  ("The independent variable is a deterministic variable, and the dependent variable is a random variable" = "C") :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_regression_analysis_correct_statement_l1139_113942


namespace NUMINAMATH_GPT_inequality_proof_l1139_113982

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (hx1 : x ≤ 1) (hy1 : y ≤ 1) (hz1 : z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1139_113982


namespace NUMINAMATH_GPT_initial_num_nuts_l1139_113970

theorem initial_num_nuts (total_nuts : ℕ) (h1 : 1/6 * total_nuts = 5) : total_nuts = 30 := 
sorry

end NUMINAMATH_GPT_initial_num_nuts_l1139_113970


namespace NUMINAMATH_GPT_smallest_integer_ending_in_9_and_divisible_by_11_l1139_113987

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end NUMINAMATH_GPT_smallest_integer_ending_in_9_and_divisible_by_11_l1139_113987


namespace NUMINAMATH_GPT_circle_radius_l1139_113997

theorem circle_radius (C : ℝ) (r : ℝ) (h1 : C = 72 * Real.pi) (h2 : C = 2 * Real.pi * r) : r = 36 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1139_113997


namespace NUMINAMATH_GPT_length_of_segment_AB_l1139_113923

theorem length_of_segment_AB :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, y^2 = 8 * x ∧ y = (y - 0) / (4 - 2) * (x - 2))
  ∧ (A.1 + B.1) / 2 = 4
  → dist A B = 12 := 
by
  sorry

end NUMINAMATH_GPT_length_of_segment_AB_l1139_113923


namespace NUMINAMATH_GPT_interest_credited_cents_l1139_113995

theorem interest_credited_cents (P : ℝ) (rt : ℝ) (A : ℝ) (interest : ℝ) :
  A = 255.31 →
  rt = 1 + 0.05 * (1/6) →
  P = A / rt →
  interest = A - P →
  (interest * 100) % 100 = 10 :=
by
  intro hA
  intro hrt
  intro hP
  intro hint
  sorry

end NUMINAMATH_GPT_interest_credited_cents_l1139_113995
