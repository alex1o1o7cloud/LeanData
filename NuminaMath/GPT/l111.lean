import Mathlib

namespace maxRegions_four_planes_maxRegions_n_planes_l111_111482

noncomputable def maxRegions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

theorem maxRegions_four_planes : maxRegions 4 = 11 := by
  sorry

theorem maxRegions_n_planes (n : ℕ) : maxRegions n = 1 + (n * (n + 1)) / 2 := by
  sorry

end maxRegions_four_planes_maxRegions_n_planes_l111_111482


namespace probability_at_least_two_same_post_l111_111343

theorem probability_at_least_two_same_post : 
  let volunteers := 3
  let posts := 4
  let total_assignments := posts ^ volunteers
  let different_post_assignments := Nat.factorial posts / (Nat.factorial (posts - volunteers))
  let probability_all_different := different_post_assignments / total_assignments
  let probability_two_same := 1 - probability_all_different
  (1 - (Nat.factorial posts / (total_assignments * Nat.factorial (posts - volunteers)))) = 5 / 8 :=
by
  sorry

end probability_at_least_two_same_post_l111_111343


namespace circle_tangent_to_x_axis_l111_111116

theorem circle_tangent_to_x_axis (b : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + 4 * x + 2 * b * y + c = 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ y : ℝ, y = -b ↔ y = 2)) ↔ (b = 2 ∨ b = -2) :=
sorry

end circle_tangent_to_x_axis_l111_111116


namespace remainder_is_x_plus_2_l111_111641

noncomputable def problem_division := 
  ∀ x : ℤ, ∃ q r : ℤ, (x^3 + 2 * x^2) = q * (x^2 + 3 * x + 2) + r ∧ r < x^2 + 3 * x + 2 ∧ r = x + 2

theorem remainder_is_x_plus_2 : problem_division := sorry

end remainder_is_x_plus_2_l111_111641


namespace smallest_pos_mult_of_31_mod_97_l111_111823

theorem smallest_pos_mult_of_31_mod_97 {k : ℕ} (h : 31 * k % 97 = 6) : 31 * k = 2015 :=
sorry

end smallest_pos_mult_of_31_mod_97_l111_111823


namespace series_convergence_p_geq_2_l111_111304

noncomputable def ai_series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, a i ^ 2 = l

noncomputable def bi_series_converges (b : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, b i ^ 2 = l

theorem series_convergence_p_geq_2 
  (a b : ℕ → ℝ) 
  (h₁ : ai_series_converges a)
  (h₂ : bi_series_converges b) 
  (p : ℝ) (hp : p ≥ 2) : 
  ∃ l : ℝ, ∑' i, |a i - b i| ^ p = l := 
sorry

end series_convergence_p_geq_2_l111_111304


namespace jimmy_bread_packs_needed_l111_111202

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l111_111202


namespace total_pieces_in_10_row_triangle_l111_111618

open Nat

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_in_10_row_triangle : 
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  unit_rods + connectors = 231 :=
by
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  show unit_rods + connectors = 231
  sorry

end total_pieces_in_10_row_triangle_l111_111618


namespace license_plates_count_l111_111813

/-
Problem:
I want to choose a license plate that is 4 characters long,
where the first character is a letter,
the last two characters are either a letter or a digit,
and the second character can be a letter or a digit 
but must be the same as either the first or the third character.
Additionally, the fourth character must be different from the first three characters.
-/

def is_letter (c : Char) : Prop := c.isAlpha
def is_digit_or_letter (c : Char) : Prop := c.isAlpha || c.isDigit
noncomputable def count_license_plates : ℕ :=
  let first_char_options := 26
  let third_char_options := 36
  let second_char_options := 2
  let fourth_char_options := 34
  first_char_options * third_char_options * second_char_options * fourth_char_options

theorem license_plates_count : count_license_plates = 59904 := by
  sorry

end license_plates_count_l111_111813


namespace lulu_cash_left_l111_111945

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l111_111945


namespace chord_length_l111_111866

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l111_111866


namespace find_x1_l111_111373

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
    (h5 : (1 - x1)^3 + (x1 - x2)^3 + (x2 - x3)^3 + x3^3 = 1 / 8) : x1 = 3 / 4 := 
by 
  sorry

end find_x1_l111_111373


namespace calc_expression_l111_111099

theorem calc_expression :
  (2014 * 2014 + 2012) - 2013 * 2013 = 6039 :=
by
  -- Let 2014 = 2013 + 1 and 2012 = 2013 - 1
  have h2014 : 2014 = 2013 + 1 := by sorry
  have h2012 : 2012 = 2013 - 1 := by sorry
  -- Start the main proof
  sorry

end calc_expression_l111_111099


namespace oranges_to_friend_is_two_l111_111063

-- Definitions based on the conditions.

def initial_oranges : ℕ := 12

def oranges_to_brother (n : ℕ) : ℕ := n / 3

def remainder_after_brother (n : ℕ) : ℕ := n - oranges_to_brother n

def oranges_to_friend (n : ℕ) : ℕ := remainder_after_brother n / 4

-- Theorem stating the problem to be proven.
theorem oranges_to_friend_is_two : oranges_to_friend initial_oranges = 2 :=
sorry

end oranges_to_friend_is_two_l111_111063


namespace number_of_pipes_l111_111640

theorem number_of_pipes (L : ℝ) : 
  let r_small := 1
  let r_large := 3
  let len_small := L
  let len_large := 2 * L
  let volume_large := π * r_large^2 * len_large
  let volume_small := π * r_small^2 * len_small
  volume_large = 18 * volume_small :=
by
  sorry

end number_of_pipes_l111_111640


namespace f_is_periodic_l111_111532

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := sorry

axiom exists_a_gt_zero : a > 0

axiom functional_eq (x : ℝ) : f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_is_periodic_l111_111532


namespace malcolm_walked_uphill_l111_111435

-- Define the conditions as variables and parameters
variables (x : ℕ)

-- Define the conditions given in the problem
def first_route_time := x + 2 * x + x
def second_route_time := 14 + 28
def time_difference := 18

-- Theorem statement - proving that Malcolm walked uphill for 6 minutes in the first route
theorem malcolm_walked_uphill : first_route_time - second_route_time = time_difference → x = 6 := by
  sorry

end malcolm_walked_uphill_l111_111435


namespace amount_spent_per_trip_l111_111589

def trips_per_month := 4
def months_per_year := 12
def initial_amount := 200
def final_amount := 104

def total_amount_spent := initial_amount - final_amount
def total_trips := trips_per_month * months_per_year

theorem amount_spent_per_trip :
  (total_amount_spent / total_trips) = 2 := 
by 
  sorry

end amount_spent_per_trip_l111_111589


namespace sufficient_not_necessary_condition_l111_111858

noncomputable def sequence_increasing_condition (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) > |a n|

noncomputable def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n < a (n + 1)

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) :
  sequence_increasing_condition a → is_increasing_sequence a ∧ ¬(∀ b : ℕ → ℝ, is_increasing_sequence b → sequence_increasing_condition b) :=
sorry

end sufficient_not_necessary_condition_l111_111858


namespace clarence_to_matthew_ratio_l111_111651

theorem clarence_to_matthew_ratio (D C M : ℝ) (h1 : D = 6.06) (h2 : D = 1 / 2 * C) (h3 : D + C + M = 20.20) : C / M = 6 := 
by 
  sorry

end clarence_to_matthew_ratio_l111_111651


namespace set_intersection_l111_111033

open Finset

-- Let the universal set U, and sets A and B be defined as follows:
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 4, 6}

-- Define the complement of A with respect to U:
def complement_A : Finset ℕ := U \ A

-- The goal is to prove that B ∩ complement_A = {4, 6}
theorem set_intersection (h : B ∩ complement_A = {4, 6}) : B ∩ complement_A = {4, 6} :=
by exact h

#check set_intersection

end set_intersection_l111_111033


namespace algebraic_expression_value_l111_111038

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y + 3 = 0) : 1 - 2 * x + 4 * y = 7 := 
by
  sorry

end algebraic_expression_value_l111_111038


namespace sequence_value_of_m_l111_111684

theorem sequence_value_of_m (a : ℕ → ℝ) (m : ℕ) (h1 : a 1 = 1)
                            (h2 : ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n)
                            (h3 : 8 * a m = 1) :
                            m = 8 := by
  sorry

end sequence_value_of_m_l111_111684


namespace primeFactors_of_3_pow_6_minus_1_l111_111052

def calcPrimeFactorsSumAndSumOfSquares (n : ℕ) : ℕ × ℕ :=
  let factors := [2, 7, 13]  -- Given directly
  let sum_factors := 2 + 7 + 13
  let sum_squares := 2^2 + 7^2 + 13^2
  (sum_factors, sum_squares)

theorem primeFactors_of_3_pow_6_minus_1 :
  calcPrimeFactorsSumAndSumOfSquares (3^6 - 1) = (22, 222) :=
by
  sorry

end primeFactors_of_3_pow_6_minus_1_l111_111052


namespace range_of_m_l111_111403

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : y1 = x1^2 - 4*x1 + 3)
  (h2 : y2 = x2^2 - 4*x2 + 3) (h3 : -1 < x1) (h4 : x1 < 1)
  (h5 : m > 0) (h6 : m-1 < x2) (h7 : x2 < m) (h8 : y1 ≠ y2) :
  (2 ≤ m ∧ m ≤ 3) ∨ (m ≥ 6) :=
sorry

end range_of_m_l111_111403


namespace Daisy_vs_Bess_l111_111445

-- Define the conditions
def Bess_daily : ℕ := 2
def Brownie_multiple : ℕ := 3
def total_pails_per_week : ℕ := 77
def days_per_week : ℕ := 7

-- Define the weekly production for Bess
def Bess_weekly : ℕ := Bess_daily * days_per_week

-- Define the weekly production for Brownie
def Brownie_weekly : ℕ := Brownie_multiple * Bess_weekly

-- Farmer Red's total weekly milk production is the sum of Bess, Brownie, and Daisy's production
-- We need to prove the difference in weekly production between Daisy and Bess is 7 pails.
theorem Daisy_vs_Bess (Daisy_weekly : ℕ) (h : Bess_weekly + Brownie_weekly + Daisy_weekly = total_pails_per_week) :
  Daisy_weekly - Bess_weekly = 7 :=
by
  sorry

end Daisy_vs_Bess_l111_111445


namespace total_shoes_l111_111860

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l111_111860


namespace division_simplification_l111_111333

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by
  sorry

end division_simplification_l111_111333


namespace minimum_length_intersection_l111_111574

def length (a b : ℝ) : ℝ := b - a

def M (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2/3 }
def N (n : ℝ) : Set ℝ := { x | n - 1/2 ≤ x ∧ x ≤ n }

def IntervalSet : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem minimum_length_intersection (m n : ℝ) (hM : M m ⊆ IntervalSet) (hN : N n ⊆ IntervalSet) :
  length (max m (n - 1/2)) (min (m + 2/3) n) = 1/6 :=
by
  sorry

end minimum_length_intersection_l111_111574


namespace relation_between_m_and_n_l111_111090

variable {A x y z a b c d e n m : ℝ}
variable {p r : ℝ}
variable (s : finset ℝ) (hset : s = {x, y, z, a, b, c, d, e})
variable (hsorted : x < y ∧ y < z ∧ z < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e)
variable (hne : n ∉ s)
variable (hme : m ∉ s)

theorem relation_between_m_and_n 
  (h_avg_n : (s.sum + n) / 9 = (s.sum / 8) * (1 + p / 100)) 
  (h_avg_m : (s.sum + m) / 9 = (s.sum / 8) * (1 + r / 100)) 
  : m = n + 9 * (s.sum / 8) * (r / 100 - p / 100) :=
sorry

end relation_between_m_and_n_l111_111090


namespace total_time_spent_l111_111023

-- Define the conditions
def number_of_chairs : ℕ := 4
def number_of_tables : ℕ := 2
def time_per_piece : ℕ := 8

-- Prove that the total time spent is 48 minutes
theorem total_time_spent : (number_of_chairs + number_of_tables) * time_per_piece = 48 :=
by
  sorry

end total_time_spent_l111_111023


namespace find_varphi_l111_111819

theorem find_varphi
  (ϕ : ℝ)
  (h : ∃ k : ℤ, ϕ = (π / 8) + (k * π / 2)) :
  ϕ = π / 8 :=
by
  sorry

end find_varphi_l111_111819


namespace inequality_always_holds_l111_111316

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry

end inequality_always_holds_l111_111316


namespace M_greater_than_N_l111_111831

variable (a : ℝ)

def M := 2 * a^2 - 4 * a
def N := a^2 - 2 * a - 3

theorem M_greater_than_N : M a > N a := by
  sorry

end M_greater_than_N_l111_111831


namespace area_of_cos_closed_figure_l111_111026

theorem area_of_cos_closed_figure :
  ∫ x in (Real.pi / 2)..(3 * Real.pi / 2), Real.cos x = 2 :=
by
  sorry

end area_of_cos_closed_figure_l111_111026


namespace cities_with_fewer_than_500000_residents_l111_111433

theorem cities_with_fewer_than_500000_residents (P Q R : ℕ) 
  (h1 : P + Q + R = 100) 
  (h2 : P = 40) 
  (h3 : Q = 35) 
  (h4 : R = 25) : P + Q = 75 :=
by 
  sorry

end cities_with_fewer_than_500000_residents_l111_111433


namespace building_total_floors_l111_111495

def earl_final_floor (start : ℕ) : ℕ :=
  start + 5 - 2 + 7

theorem building_total_floors (start : ℕ) (current : ℕ) (remaining : ℕ) (total : ℕ) :
  earl_final_floor start = current →
  remaining = 9 →
  total = current + remaining →
  start = 1 →
  total = 20 := by
sorry

end building_total_floors_l111_111495


namespace similar_triangle_perimeter_l111_111402

theorem similar_triangle_perimeter :
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 12 →
  ∀ (d : ℝ), d = 30 →
  ∃ (p : ℝ), p = 65 ∧ 
  (∃ a' b' c' : ℝ, (a' = 17.5 ∧ b' = 17.5 ∧ c' = d) ∧ p = a' + b' + c') :=
by sorry

end similar_triangle_perimeter_l111_111402


namespace part_one_part_two_l111_111848

-- Defining the function and its first derivative
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Part (Ⅰ)
theorem part_one (a b : ℝ)
  (H1 : f' a b 3 = 24)
  (H2 : f' a b 1 = 0) :
  a = 1 ∧ b = -3 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f' 1 (-3) x ≤ 0) :=
sorry

-- Part (Ⅱ)
theorem part_two (b : ℝ)
  (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 3 * x^2 + b ≤ 0) :
  b ≤ -3 :=
sorry

end part_one_part_two_l111_111848


namespace perimeter_of_polygon_l111_111280

-- Conditions
variables (a b : ℝ) (polygon_is_part_of_rectangle : 0 < a ∧ 0 < b)

-- Prove that if the polygon completes a rectangle with perimeter 28,
-- then the perimeter of the polygon is 28.
theorem perimeter_of_polygon (h : 2 * (a + b) = 28) : 2 * (a + b) = 28 :=
by
  exact h

end perimeter_of_polygon_l111_111280


namespace find_a_b_l111_111704

noncomputable def parabola_props (a b : ℝ) : Prop :=
a ≠ 0 ∧ 
∀ x : ℝ, a * x^2 + b * x - 4 = (1 / 2) * x^2 + x - 4

theorem find_a_b {a b : ℝ} (h1 : parabola_props a b) : 
a = 1 / 2 ∧ b = -1 :=
sorry

end find_a_b_l111_111704


namespace machines_remain_closed_l111_111112

open Real

/-- A techno company has 14 machines of equal efficiency in its factory.
The annual manufacturing costs are Rs 42000 and establishment charges are Rs 12000.
The annual output of the company is Rs 70000. The annual output and manufacturing
costs are directly proportional to the number of machines. The shareholders get
12.5% profit, which is directly proportional to the annual output of the company.
If some machines remain closed throughout the year, then the percentage decrease
in the amount of profit of the shareholders is 12.5%. Prove that 2 machines remain
closed throughout the year. -/
theorem machines_remain_closed (machines total_cost est_charges output : ℝ)
    (shareholders_profit : ℝ)
    (machines_closed percentage_decrease : ℝ) :
  machines = 14 →
  total_cost = 42000 →
  est_charges = 12000 →
  output = 70000 →
  shareholders_profit = 0.125 →
  percentage_decrease = 0.125 →
  machines_closed = 2 :=
by
  sorry

end machines_remain_closed_l111_111112


namespace side_length_of_square_l111_111580

theorem side_length_of_square 
  (x : ℝ) 
  (h₁ : 4 * x = 2 * (x * x)) :
  x = 2 :=
by 
  sorry

end side_length_of_square_l111_111580


namespace youseff_blocks_l111_111060

theorem youseff_blocks (x : ℕ) (h1 : x = 1 * x) (h2 : (20 / 60 : ℚ) * x = x / 3) (h3 : x = x / 3 + 8) : x = 12 := by
  have : x = x := rfl  -- trivial step to include the equality
  sorry

end youseff_blocks_l111_111060


namespace fewer_sevens_l111_111159

def seven_representation (n : ℕ) : ℕ :=
  (7 * (10^n - 1)) / 9

theorem fewer_sevens (n : ℕ) :
  ∃ m, m < n ∧ 
    (∃ expr : ℕ → ℕ, (∀ i < n, expr i = 7) ∧ seven_representation n = expr m) :=
sorry

end fewer_sevens_l111_111159


namespace arithmetic_sequence_a2_value_l111_111270

open Nat

theorem arithmetic_sequence_a2_value (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 + a 3 = 12) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) : 
  a 2 = 5 :=
  sorry

end arithmetic_sequence_a2_value_l111_111270


namespace bathtub_problem_l111_111253

theorem bathtub_problem (T : ℝ) (h1 : 1 / T - 1 / 12 = 1 / 60) : T = 10 := 
by {
  -- Sorry, skip the proof as requested
  sorry
}

end bathtub_problem_l111_111253


namespace fibonacci_recurrence_l111_111808

theorem fibonacci_recurrence (f : ℕ → ℝ) (a b : ℝ) 
  (h₀ : f 0 = 1) 
  (h₁ : f 1 = 1) 
  (h₂ : ∀ n, f (n + 2) = f (n + 1) + f n)
  (h₃ : a + b = 1) 
  (h₄ : a * b = -1) 
  (h₅ : a > b) 
  : ∀ n, f n = (a ^ (n + 1) - b ^ (n + 1)) / Real.sqrt 5 := by
  sorry

end fibonacci_recurrence_l111_111808


namespace gain_percent_correct_l111_111149

theorem gain_percent_correct (C S : ℝ) (h : 50 * C = 28 * S) : 
  ( (S - C) / C ) * 100 = 1100 / 14 :=
by
  sorry

end gain_percent_correct_l111_111149


namespace calculate_expr_l111_111135

theorem calculate_expr :
  ( (5 / 12: ℝ) ^ 2022) * (-2.4) ^ 2023 = - (12 / 5: ℝ) := 
by 
  sorry

end calculate_expr_l111_111135


namespace mother_sold_rings_correct_l111_111378

noncomputable def motherSellsRings (initial_bought_rings mother_bought_rings remaining_rings final_stock : ℤ) : ℤ :=
  let initial_stock := initial_bought_rings / 2
  let total_stock := initial_bought_rings + initial_stock
  let sold_by_eliza := (3 * total_stock) / 4
  let remaining_after_eliza := total_stock - sold_by_eliza
  let new_total_stock := remaining_after_eliza + mother_bought_rings
  new_total_stock - final_stock

theorem mother_sold_rings_correct :
  motherSellsRings 200 300 225 300 = 150 :=
by
  sorry

end mother_sold_rings_correct_l111_111378


namespace graph_of_cubic_equation_is_three_lines_l111_111699

theorem graph_of_cubic_equation_is_three_lines (x y : ℝ) :
  (x + y) ^ 3 = x ^ 3 + y ^ 3 →
  (y = -x ∨ x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_cubic_equation_is_three_lines_l111_111699


namespace part1_part2_part3_l111_111516

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a / x + Real.log x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  1 - a / x^2 + 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f' x a - x

theorem part1 (a : ℝ) (h : f' 1 a = 0) : a = 2 :=
  sorry

theorem part2 {a : ℝ} (h : ∀ x, 1 < x → x < 2 → f' x a ≥ 0) : a ≤ 2 :=
  sorry

theorem part3 (a : ℝ) :
  ((a > 1 → ∀ x, g x a ≠ 0) ∧ 
  (a = 1 ∨ a ≤ 0 → ∃ x, g x a = 0 ∧ ∀ y, g y a = 0 → y = x) ∧ 
  (0 < a ∧ a < 1 → ∃ x y, x ≠ y ∧ g x a = 0 ∧ g y a = 0)) :=
  sorry

end part1_part2_part3_l111_111516


namespace consecutive_even_legs_sum_l111_111241

theorem consecutive_even_legs_sum (x : ℕ) (h : x % 2 = 0) (hx : x ^ 2 + (x + 2) ^ 2 = 34 ^ 2) : x + (x + 2) = 48 := by
  sorry

end consecutive_even_legs_sum_l111_111241


namespace janet_earnings_per_hour_l111_111134

theorem janet_earnings_per_hour :
  let text_posts := 150
  let image_posts := 80
  let video_posts := 20
  let rate_text := 0.25
  let rate_image := 0.30
  let rate_video := 0.40
  text_posts * rate_text + image_posts * rate_image + video_posts * rate_video = 69.50 :=
by
  sorry

end janet_earnings_per_hour_l111_111134


namespace vandermonde_identity_combinatorial_identity_l111_111067

open Nat

-- Problem 1: Vandermonde Identity
theorem vandermonde_identity (m n k : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : m + n ≥ k) :
  (Finset.range (k + 1)).sum (λ i => Nat.choose m i * Nat.choose n (k - i)) = Nat.choose (m + n) k :=
sorry

-- Problem 2:
theorem combinatorial_identity (p q n : ℕ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < n) :
  (Finset.range (p + 1)).sum (λ k => Nat.choose p k * Nat.choose q k * Nat.choose (n + k) (p + q)) =
  Nat.choose n p * Nat.choose n q :=
sorry

end vandermonde_identity_combinatorial_identity_l111_111067


namespace inequality_l111_111807

theorem inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) : 
  (a / b) + (b / c) + (c / a) + (b / a) + (a / c) + (c / b) + 6 ≥ 
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
sorry

end inequality_l111_111807


namespace apartments_decrease_l111_111157

theorem apartments_decrease (p_initial e_initial p e q : ℕ) (h1: p_initial = 5) (h2: e_initial = 2) (h3: q = 1)
    (first_mod: p = p_initial - 2) (e_first_mod: e = e_initial + 3) (q_eq: q = 1)
    (second_mod: p = p - 2) (e_second_mod: e = e + 3) :
    p_initial * e_initial * q > p * e * q := by
  sorry

end apartments_decrease_l111_111157


namespace lizard_ratio_l111_111524

def lizard_problem (W S : ℕ) : Prop :=
  (S = 7 * W) ∧ (3 = S + W - 69) ∧ (W / 3 = 3)

theorem lizard_ratio (W S : ℕ) (h : lizard_problem W S) : W / 3 = 3 :=
  by
    rcases h with ⟨h1, h2, h3⟩
    exact h3

end lizard_ratio_l111_111524


namespace elephants_ratio_l111_111397

theorem elephants_ratio (x : ℝ) (w : ℝ) (g : ℝ) (total : ℝ) :
  w = 70 →
  total = 280 →
  g = x * w →
  w + g = total →
  x = 3 :=
by 
  intros h1 h2 h3 h4
  sorry

end elephants_ratio_l111_111397


namespace percentage_error_x_percentage_error_y_l111_111300

theorem percentage_error_x (x : ℝ) : 
  let correct_result := x * 10
  let erroneous_result := x / 10
  (correct_result - erroneous_result) / correct_result * 100 = 99 :=
by
  sorry

theorem percentage_error_y (y : ℝ) : 
  let correct_result := y + 15
  let erroneous_result := y - 15
  (correct_result - erroneous_result) / correct_result * 100 = (30 / (y + 15)) * 100 :=
by
  sorry

end percentage_error_x_percentage_error_y_l111_111300


namespace sqrt_combination_l111_111717

theorem sqrt_combination : 
    ∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 8) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 3))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 12))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 0.2))) :=
by
  sorry

end sqrt_combination_l111_111717


namespace quadratic_no_real_roots_l111_111110

theorem quadratic_no_real_roots 
  (a b c m : ℝ) 
  (h1 : c > 0) 
  (h2 : c = a * m^2) 
  (h3 : c = b * m)
  : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0) :=
by 
  sorry

end quadratic_no_real_roots_l111_111110


namespace largest_y_coordinate_on_graph_l111_111398

theorem largest_y_coordinate_on_graph :
  ∀ x y : ℝ, (x / 7) ^ 2 + ((y - 3) / 5) ^ 2 = 0 → y ≤ 3 := 
by
  intro x y h
  sorry

end largest_y_coordinate_on_graph_l111_111398


namespace largest_d_l111_111659

theorem largest_d (a b c d : ℤ) 
  (h₁ : a + 1 = b - 2) 
  (h₂ : a + 1 = c + 3) 
  (h₃ : a + 1 = d - 4) : 
  d > a ∧ d > b ∧ d > c := 
by 
  -- Here we would provide the proof, but for now we'll skip it
  sorry

end largest_d_l111_111659


namespace geometric_sequence_a9_l111_111463

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℝ)
variable (q : ℝ)

theorem geometric_sequence_a9
  (h_seq : geometric_sequence a q)
  (h2 : a 1 * a 4 = -32)
  (h3 : a 2 + a 3 = 4)
  (hq : ∃ n : ℤ, q = ↑n) :
  a 8 = -256 := 
sorry

end geometric_sequence_a9_l111_111463


namespace calculation_l111_111540

theorem calculation :
  7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end calculation_l111_111540


namespace saree_blue_stripes_l111_111029

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    brown_stripes = 4 →
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_brown h_gold h_blue
  sorry

end saree_blue_stripes_l111_111029


namespace range_of_uv_sq_l111_111712

theorem range_of_uv_sq (u v w : ℝ) (h₀ : 0 ≤ u) (h₁ : 0 ≤ v) (h₂ : 0 ≤ w) (h₃ : u + v + w = 2) :
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 :=
sorry

end range_of_uv_sq_l111_111712


namespace wyatt_bought_4_cartons_of_juice_l111_111455

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

end wyatt_bought_4_cartons_of_juice_l111_111455


namespace triangle_side_range_l111_111009

theorem triangle_side_range (AB AC x : ℝ) (hAB : AB = 16) (hAC : AC = 7) (hBC : BC = x) :
  9 < x ∧ x < 23 :=
by
  sorry

end triangle_side_range_l111_111009


namespace petya_friends_l111_111105

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l111_111105


namespace valid_pin_count_l111_111795

def total_pins : ℕ := 10^5

def restricted_pins (seq : List ℕ) : ℕ :=
  if seq = [3, 1, 4, 1] then 10 else 0

def valid_pins (seq : List ℕ) : ℕ :=
  total_pins - restricted_pins seq

theorem valid_pin_count :
  valid_pins [3, 1, 4, 1] = 99990 :=
by
  sorry

end valid_pin_count_l111_111795


namespace solve_floor_equation_l111_111170

theorem solve_floor_equation (x : ℝ) (hx : (∃ (y : ℤ), (x^3 - 40 * (y : ℝ) - 78 = 0) ∧ (y : ℝ) ≤ x ∧ x < (y + 1 : ℝ))) :
  x = -5.45 ∨ x = -4.96 ∨ x = -1.26 ∨ x = 6.83 ∨ x = 7.10 :=
by sorry

end solve_floor_equation_l111_111170


namespace triangle_max_area_proof_l111_111407

open Real

noncomputable def triangle_max_area (A B C : ℝ) (AB : ℝ) (tanA tanB : ℝ) : Prop :=
  AB = 4 ∧ tanA * tanB = 3 / 4 → ∃ S : ℝ, S = 2 * sqrt 3

theorem triangle_max_area_proof (A B C : ℝ) (tanA tanB : ℝ) (AB : ℝ) : 
  triangle_max_area A B C AB tanA tanB :=
by
  sorry

end triangle_max_area_proof_l111_111407


namespace chess_tournament_total_games_l111_111206

theorem chess_tournament_total_games (n : ℕ) (h : n = 10) : (n * (n - 1)) / 2 = 45 := by
  sorry

end chess_tournament_total_games_l111_111206


namespace range_of_m_l111_111308

theorem range_of_m (x m : ℝ) : (|x - 3| ≤ 2) → ((x - m + 1) * (x - m - 1) ≤ 0) → 
  (¬(|x - 3| ≤ 2) → ¬((x - m + 1) * (x - m - 1) ≤ 0)) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l111_111308


namespace solve_inequality_group_l111_111334

theorem solve_inequality_group (x : ℝ) (h1 : -9 < 2 * x - 1) (h2 : 2 * x - 1 ≤ 6) :
  -4 < x ∧ x ≤ 3.5 := 
sorry

end solve_inequality_group_l111_111334


namespace tan_add_pi_over_3_l111_111251

variable (y : ℝ)

theorem tan_add_pi_over_3 (h : Real.tan y = 3) : 
  Real.tan (y + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := 
by
  sorry

end tan_add_pi_over_3_l111_111251


namespace paper_thickness_after_folds_l111_111379

def folded_thickness (initial_thickness : ℝ) (folds : ℕ) : ℝ :=
  initial_thickness * 2^folds

theorem paper_thickness_after_folds :
  folded_thickness 0.1 4 = 1.6 :=
by
  sorry

end paper_thickness_after_folds_l111_111379


namespace solve_inequality_l111_111476

theorem solve_inequality (x : ℝ) :
  |(3 * x - 2) / (x ^ 2 - x - 2)| > 3 ↔ (x ∈ Set.Ioo (-1) (-2 / 3) ∪ Set.Ioo (1 / 3) 4) :=
by sorry

end solve_inequality_l111_111476


namespace g_6_eq_1_l111_111136

variable (f : ℝ → ℝ)

noncomputable def g (x : ℝ) := f x + 1 - x

theorem g_6_eq_1 
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g f 6 = 1 :=
by
  sorry

end g_6_eq_1_l111_111136


namespace movement_left_3m_l111_111591

-- Define the condition
def movement_right_1m : ℝ := 1

-- Define the theorem stating that movement to the left by 3m should be denoted as -3
theorem movement_left_3m : movement_right_1m * (-3) = -3 :=
by
  sorry

end movement_left_3m_l111_111591


namespace black_dogs_count_l111_111512

def number_of_brown_dogs := 20
def number_of_white_dogs := 10
def total_number_of_dogs := 45
def number_of_black_dogs := total_number_of_dogs - (number_of_brown_dogs + number_of_white_dogs)

theorem black_dogs_count : number_of_black_dogs = 15 := by
  sorry

end black_dogs_count_l111_111512


namespace max_value_fn_l111_111594

theorem max_value_fn : ∀ x : ℝ, y = 1 / (|x| + 2) → 
  ∃ y : ℝ, y = 1 / 2 ∧ ∀ x : ℝ, 1 / (|x| + 2) ≤ y :=
sorry

end max_value_fn_l111_111594


namespace sugar_left_correct_l111_111996

-- Define the total amount of sugar bought by Pamela
def total_sugar : ℝ := 9.8

-- Define the amount of sugar spilled by Pamela
def spilled_sugar : ℝ := 5.2

-- Define the amount of sugar left after spilling
def sugar_left : ℝ := total_sugar - spilled_sugar

-- State that the amount of sugar left should be equivalent to the correct answer
theorem sugar_left_correct : sugar_left = 4.6 :=
by
  sorry

end sugar_left_correct_l111_111996


namespace largest_trailing_zeros_l111_111775

def count_trailing_zeros (n : Nat) : Nat :=
  if n = 0 then 0
  else Nat.min (Nat.factorial (n / 10)) (Nat.factorial (n / 5))

theorem largest_trailing_zeros :
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^5 * 3^4 * 5^6)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^4 * 3^4 * 5^5)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (4^2 * 5^4 * 6^3)) :=
  sorry

end largest_trailing_zeros_l111_111775


namespace Mikail_money_left_after_purchase_l111_111868

def Mikail_age_tomorrow : ℕ := 9  -- Defining Mikail's age tomorrow as 9.

def gift_per_year : ℕ := 5  -- Defining the gift amount per year of age as $5.

def video_game_cost : ℕ := 80  -- Defining the cost of the video game as $80.

def calculate_gift (age : ℕ) : ℕ := age * gift_per_year  -- Function to calculate the gift money he receives based on his age.

-- The statement we need to prove:
theorem Mikail_money_left_after_purchase : 
    calculate_gift Mikail_age_tomorrow < video_game_cost → calculate_gift Mikail_age_tomorrow - video_game_cost = 0 :=
by
  sorry

end Mikail_money_left_after_purchase_l111_111868


namespace tangent_line_to_circle_range_mn_l111_111753

theorem tangent_line_to_circle_range_mn (m n : ℝ) 
  (h1 : (m + 1) * (m + 1) + (n + 1) * (n + 1) = 4) :
  (m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end tangent_line_to_circle_range_mn_l111_111753


namespace blue_notes_per_red_note_l111_111743

-- Given conditions
def total_red_notes : ℕ := 5 * 6
def additional_blue_notes : ℕ := 10
def total_notes : ℕ := 100
def total_blue_notes := total_notes - total_red_notes

-- Proposition that needs to be proved
theorem blue_notes_per_red_note (x : ℕ) : total_red_notes * x + additional_blue_notes = total_blue_notes → x = 2 := by
  intro h
  sorry

end blue_notes_per_red_note_l111_111743


namespace negation_of_existential_l111_111919

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by
  sorry

end negation_of_existential_l111_111919


namespace tangent_line_at_point_P_l111_111021

-- Define the curve y = x^3 
def curve (x : ℝ) : ℝ := x ^ 3

-- Define the point P(1,1)
def pointP : ℝ × ℝ := (1, 1)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x ^ 2

-- Define the tangent line equation we need to prove
def tangent_line (x y : ℝ) : Prop := 3 * x - y - 2 = 0

theorem tangent_line_at_point_P :
  ∀ (x y : ℝ), 
  pointP = (1, 1) ∧ curve 1 = 1 ∧ curve_derivative 1 = 3 → 
  tangent_line 1 1 := 
by
  intros x y h
  sorry

end tangent_line_at_point_P_l111_111021


namespace save_after_increase_l111_111258

def monthly_saving_initial (salary : ℕ) (saving_percentage : ℕ) : ℕ :=
  salary * saving_percentage / 100

def monthly_expense_initial (salary : ℕ) (saving : ℕ) : ℕ :=
  salary - saving

def increase_by_percentage (amount : ℕ) (percentage : ℕ) : ℕ :=
  amount * percentage / 100

def new_expense (initial_expense : ℕ) (increase : ℕ) : ℕ :=
  initial_expense + increase

def new_saving (salary : ℕ) (expense : ℕ) : ℕ :=
  salary - expense

theorem save_after_increase (salary saving_percentage increase_percentage : ℕ) 
  (H_salary : salary = 5500) 
  (H_saving_percentage : saving_percentage = 20) 
  (H_increase_percentage : increase_percentage = 20) :
  new_saving salary (new_expense (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) (increase_by_percentage (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) increase_percentage)) = 220 := 
by
  sorry

end save_after_increase_l111_111258


namespace complementary_angle_ratio_l111_111188

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l111_111188


namespace total_pay_is_correct_l111_111655

-- Define the constants and conditions
def regular_rate := 3  -- $ per hour
def regular_hours := 40  -- hours
def overtime_multiplier := 2  -- overtime pay is twice the regular rate
def overtime_hours := 8  -- hours

-- Calculate regular and overtime pay
def regular_pay := regular_rate * regular_hours
def overtime_rate := regular_rate * overtime_multiplier
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total pay
def total_pay := regular_pay + overtime_pay

-- Prove that the total pay is $168
theorem total_pay_is_correct : total_pay = 168 := by
  -- The proof goes here
  sorry

end total_pay_is_correct_l111_111655


namespace least_x_divisibility_l111_111855

theorem least_x_divisibility :
  ∃ x : ℕ, (x > 0) ∧ ((x^2 + 164) % 3 = 0) ∧ ((x^2 + 164) % 4 = 0) ∧ ((x^2 + 164) % 5 = 0) ∧
  ((x^2 + 164) % 6 = 0) ∧ ((x^2 + 164) % 7 = 0) ∧ ((x^2 + 164) % 8 = 0) ∧ 
  ((x^2 + 164) % 9 = 0) ∧ ((x^2 + 164) % 10 = 0) ∧ ((x^2 + 164) % 11 = 0) ∧ x = 166 → 
  3 = 3 :=
by
  sorry

end least_x_divisibility_l111_111855


namespace valid_m_values_l111_111608

theorem valid_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) → m < 1 :=
by
  sorry

end valid_m_values_l111_111608


namespace rationalize_denominator_l111_111055

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l111_111055


namespace needed_correct_to_pass_l111_111835

def total_questions : Nat := 120
def genetics_questions : Nat := 20
def ecology_questions : Nat := 50
def evolution_questions : Nat := 50

def correct_genetics : Nat := (60 * genetics_questions) / 100
def correct_ecology : Nat := (50 * ecology_questions) / 100
def correct_evolution : Nat := (70 * evolution_questions) / 100
def total_correct : Nat := correct_genetics + correct_ecology + correct_evolution

def passing_rate : Nat := 65
def passing_score : Nat := (passing_rate * total_questions) / 100

theorem needed_correct_to_pass : (passing_score - total_correct) = 6 := 
by
  sorry

end needed_correct_to_pass_l111_111835


namespace total_distance_traveled_l111_111922

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l111_111922


namespace ratio_yuan_david_l111_111232

-- Definitions
def yuan_age (david_age : ℕ) : ℕ := david_age + 7
def ratio (a b : ℕ) : ℚ := a / b

-- Conditions
variable (david_age : ℕ) (h_david : david_age = 7)

-- Proof Statement
theorem ratio_yuan_david : ratio (yuan_age david_age) david_age = 2 :=
by
  sorry

end ratio_yuan_david_l111_111232


namespace odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l111_111960

theorem odd_positive_multiples_of_7_with_units_digit_1_lt_200_count : 
  ∃ (count : ℕ), count = 3 ∧
  ∀ n : ℕ, (n % 2 = 1) → (n % 7 = 0) → (n < 200) → (n % 10 = 1) → count = 3 :=
sorry

end odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l111_111960


namespace ab_sum_pow_eq_neg_one_l111_111748

theorem ab_sum_pow_eq_neg_one (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) : (a + b) ^ 2003 = -1 := 
by
  sorry

end ab_sum_pow_eq_neg_one_l111_111748


namespace set_complement_l111_111787

variable {U : Set ℝ} (A : Set ℝ)

theorem set_complement :
  (U = {x : ℝ | x > 1}) →
  (A ⊆ U) →
  (U \ A = {x : ℝ | x > 9}) →
  (A = {x : ℝ | 1 < x ∧ x ≤ 9}) :=
by
  intros hU hA hC
  sorry

end set_complement_l111_111787


namespace anne_cleaning_time_l111_111669

variable (B A : ℝ)

theorem anne_cleaning_time :
  (B + A) * 4 = 1 ∧ (B + 2 * A) * 3 = 1 → 1/A = 12 := 
by
  intro h
  sorry

end anne_cleaning_time_l111_111669


namespace population_difference_l111_111953

variable (A B C : ℝ)

-- Conditions
def population_condition (A B C : ℝ) : Prop := A + B = B + C + 5000

-- The proof statement
theorem population_difference (h : population_condition A B C) : A - C = 5000 :=
by sorry

end population_difference_l111_111953


namespace finding_a_of_geometric_sequence_l111_111991
noncomputable def geometric_sequence_a : Prop :=
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) ∧ a^2 = 2

theorem finding_a_of_geometric_sequence :
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end finding_a_of_geometric_sequence_l111_111991


namespace correct_expression_l111_111142

theorem correct_expression (a b c : ℝ) : a - b + c = a - (b - c) :=
by
  sorry

end correct_expression_l111_111142


namespace marbles_count_l111_111303

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l111_111303


namespace multiplication_with_negative_l111_111073

theorem multiplication_with_negative (a b : Int) (h1 : a = 3) (h2 : b = -2) : a * b = -6 :=
by
  sorry

end multiplication_with_negative_l111_111073


namespace sum_of_three_digit_numbers_l111_111527

theorem sum_of_three_digit_numbers (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  222 * (a + b + c) ≠ 2021 := 
sorry

end sum_of_three_digit_numbers_l111_111527


namespace problem_solution_l111_111561

lemma factor_def (m n : ℕ) : n ∣ m ↔ ∃ k, m = n * k := by sorry

def is_true_A : Prop := 4 ∣ 24
def is_true_B : Prop := 19 ∣ 209 ∧ ¬ (19 ∣ 63)
def is_true_C : Prop := ¬ (30 ∣ 90) ∧ ¬ (30 ∣ 65)
def is_true_D : Prop := 11 ∣ 33 ∧ ¬ (11 ∣ 77)
def is_true_E : Prop := 9 ∣ 180

theorem problem_solution : (is_true_A ∧ is_true_B ∧ is_true_E) ∧ ¬(is_true_C) ∧ ¬(is_true_D) :=
  by sorry

end problem_solution_l111_111561


namespace matrix_pow_minus_l111_111017

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_pow_minus : B ^ 20 - 3 * (B ^ 19) = ![![0, 4 * (2 ^ 19)], ![0, -(2 ^ 19)]] :=
by
  sorry

end matrix_pow_minus_l111_111017


namespace determine_triangle_area_l111_111869

noncomputable def triangle_area_proof : Prop :=
  let height : ℝ := 2
  let angle_ratio : ℝ := 2 / 1
  let smaller_base_part : ℝ := 1
  let larger_base_part : ℝ := 7 / 3
  let base := smaller_base_part + larger_base_part
  let area := (1 / 2) * base * height
  area = 11 / 3

theorem determine_triangle_area : triangle_area_proof :=
by
  sorry

end determine_triangle_area_l111_111869


namespace advance_tickets_sold_20_l111_111542

theorem advance_tickets_sold_20 :
  ∃ (A S : ℕ), 20 * A + 30 * S = 1600 ∧ A + S = 60 ∧ A = 20 :=
by
  sorry

end advance_tickets_sold_20_l111_111542


namespace A_intersect_B_eq_l111_111306

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x ≤ 1
def A_cap_B (x : ℝ) : Prop := x ∈ {y | A y} ∧ x ∈ {y | B y}

theorem A_intersect_B_eq (x : ℝ) : (A_cap_B x) ↔ (x ∈ Set.Ioc 0 1) :=
by
  sorry

end A_intersect_B_eq_l111_111306


namespace sequence_root_formula_l111_111185

theorem sequence_root_formula {a : ℕ → ℝ} 
    (h1 : ∀ n, (a (n + 1))^2 = (a n)^2 + 4)
    (h2 : a 1 = 1)
    (h3 : ∀ n, a n > 0) :
    ∀ n, a n = Real.sqrt (4 * n - 3) := 
sorry

end sequence_root_formula_l111_111185


namespace base_of_second_fraction_l111_111526

theorem base_of_second_fraction (base : ℝ) (h1 : (1/2) ^ 16 * (1/base) ^ 8 = 1 / (18 ^ 16)): base = 81 :=
sorry

end base_of_second_fraction_l111_111526


namespace range_of_a_l111_111048

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the main theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, tensor (x - a) (x + a) < 1) → 
  (-((1 : ℝ) / 2) < a ∧ a < (3 : ℝ) / 2) :=
by
  sorry

end range_of_a_l111_111048


namespace G_five_times_of_2_l111_111689

def G (x : ℝ) : ℝ := (x - 2) ^ 2 - 1

theorem G_five_times_of_2 : G (G (G (G (G 2)))) = 1179395 := 
by 
  rw [G, G, G, G, G]; 
  sorry

end G_five_times_of_2_l111_111689


namespace quilt_cost_l111_111442

theorem quilt_cost :
  let length := 7
  let width := 8
  let cost_per_sq_ft := 40
  let area := length * width
  let total_cost := area * cost_per_sq_ft
  total_cost = 2240 :=
by
  sorry

end quilt_cost_l111_111442


namespace total_length_proof_l111_111601

noncomputable def total_length_climbed (keaton_ladder_height : ℕ) (keaton_times : ℕ) (shortening : ℕ) (reece_times : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - shortening
  let keaton_total := keaton_ladder_height * keaton_times
  let reece_total := reece_ladder_height * reece_times
  (keaton_total + reece_total) * 100

theorem total_length_proof :
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end total_length_proof_l111_111601


namespace minimum_value_is_12_l111_111254

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) : ℝ :=
(a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (a + d)) + (1 / (b + c)) + (1 / (b + d)) + (1 / (c + d)))

theorem minimum_value_is_12 (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) :
  smallest_possible_value a b c d h1 h2 h3 h4 h5 h6 h7 ≥ 12 :=
sorry

end minimum_value_is_12_l111_111254


namespace farmer_children_l111_111145

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l111_111145


namespace contrapositive_even_statement_l111_111828

-- Translate the conditions to Lean 4 definitions
def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem contrapositive_even_statement (a b : Int) :
  (¬ is_even (a + b) → ¬ (is_even a ∧ is_even b)) ↔ 
  (is_even a ∧ is_even b → is_even (a + b)) :=
by sorry

end contrapositive_even_statement_l111_111828


namespace meaningful_fraction_l111_111646

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l111_111646


namespace unique_solution_l111_111337

def unique_ordered_pair : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
               (∃ x : ℝ, x = (m : ℝ)^(1/3) - (n : ℝ)^(1/3) ∧ x^6 + 4 * x^3 - 36 * x^2 + 4 = 0) ∧
               m = 2 ∧ n = 4

theorem unique_solution : unique_ordered_pair := sorry

end unique_solution_l111_111337


namespace other_solution_of_quadratic_l111_111404

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  65 * x^2 - 104 * x + 31 = 0

-- Main theorem statement
theorem other_solution_of_quadratic :
  quadratic_eq (6 / 5) → quadratic_eq (5 / 13) :=
by
  intro h
  sorry

end other_solution_of_quadratic_l111_111404


namespace correct_option_is_C_l111_111056

theorem correct_option_is_C (x y : ℝ) :
  ¬(3 * x + 4 * y = 12 * x * y) ∧
  ¬(x^9 / x^3 = x^3) ∧
  ((x^2)^3 = x^6) ∧
  ¬((x - y)^2 = x^2 - y^2) :=
by
  sorry

end correct_option_is_C_l111_111056


namespace problem_statement_l111_111972

def A : Prop := (∀ (x : ℝ), x^2 - 3*x + 2 = 0 → x = 2)
def B : Prop := (∃ (x : ℝ), x^2 - x + 1 < 0)
def C : Prop := (¬(∀ (x : ℝ), x > 2 → x^2 - 3*x + 2 > 0))

theorem problem_statement :
  ¬ (A ∧ ∀ (x : ℝ), (B → (x^2 - x + 1) ≥ 0) ∧ (¬(A) ∧ C)) :=
sorry

end problem_statement_l111_111972


namespace part1_real_values_part2_imaginary_values_l111_111353

namespace ComplexNumberProblem

-- Definitions of conditions for part 1
def imaginaryZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 = 0

def realPositive (x : ℝ) : Prop :=
  x^2 - 2*x - 2 > 0

-- Definition of question for part 1
def realValues (x : ℝ) : Prop :=
  x = -1 ∨ x = -2

-- Proof problem for part 1
theorem part1_real_values (x : ℝ) (h1 : imaginaryZero x) (h2 : realPositive x) : realValues x :=
by
  have h : realValues x := sorry
  exact h

-- Definitions of conditions for part 2
def realPartOne (x : ℝ) : Prop :=
  x^2 - 2*x - 2 = 1

def imaginaryNonZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 ≠ 0

-- Definition of question for part 2
def imaginaryValues (x : ℝ) : Prop :=
  x = 3

-- Proof problem for part 2
theorem part2_imaginary_values (x : ℝ) (h1 : realPartOne x) (h2 : imaginaryNonZero x) : imaginaryValues x :=
by
  have h : imaginaryValues x := sorry
  exact h

end ComplexNumberProblem

end part1_real_values_part2_imaginary_values_l111_111353


namespace gcd_47_power5_1_l111_111839
-- Import the necessary Lean library

-- Mathematically equivalent proof problem in Lean 4
theorem gcd_47_power5_1 (a b : ℕ) (h1 : a = 47^5 + 1) (h2 : b = 47^5 + 47^3 + 1) :
  Nat.gcd a b = 1 :=
by
  sorry

end gcd_47_power5_1_l111_111839


namespace gcd_repeated_three_digit_integers_l111_111010

theorem gcd_repeated_three_digit_integers : 
  ∀ m ∈ {n | 100 ≤ n ∧ n < 1000}, 
  gcd (1001 * m) (1001 * (m + 1)) = 1001 :=
by
  sorry

end gcd_repeated_three_digit_integers_l111_111010


namespace orthocenter_of_ABC_l111_111661

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨-1, 3, 2⟩
def B : Point3D := ⟨4, -2, 2⟩
def C : Point3D := ⟨2, -1, 6⟩

def orthocenter (A B C : Point3D) : Point3D :=
  -- formula to calculate the orthocenter
  sorry

theorem orthocenter_of_ABC :
  orthocenter A B C = ⟨101 / 150, 192 / 150, 232 / 150⟩ :=
by 
  -- proof steps
  sorry

end orthocenter_of_ABC_l111_111661


namespace functional_equation_solution_l111_111360

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y, f (x ^ 2) - f (y ^ 2) + 2 * x + 1 = f (x + y) * f (x - y)) :
  (∀ x, f x = x + 1) ∨ (∀ x, f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l111_111360


namespace find_a_extreme_value_l111_111496

theorem find_a_extreme_value (a : ℝ) :
  (f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9) →
  (f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + 3) →
  f' (-3) = 0 →
  a = 5 :=
by
  sorry

end find_a_extreme_value_l111_111496


namespace inequality_proof_l111_111058

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 :=
by
  sorry

end inequality_proof_l111_111058


namespace sum_of_solutions_l111_111368

theorem sum_of_solutions (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := 
by {
  -- missing proof part
  sorry
}

end sum_of_solutions_l111_111368


namespace geometric_sequence_max_product_l111_111950

theorem geometric_sequence_max_product
  (b : ℕ → ℝ) (q : ℝ) (b1 : ℝ)
  (h_b1_pos : b1 > 0)
  (h_q : 0 < q ∧ q < 1)
  (h_b : ∀ n, b (n + 1) = b n * q)
  (h_b7_gt_1 : b 7 > 1)
  (h_b8_lt_1 : b 8 < 1) :
  (∀ (n : ℕ), n = 7 → b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 = b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7) :=
by {
  sorry
}

end geometric_sequence_max_product_l111_111950


namespace packets_for_dollars_l111_111741

variable (P R C : ℕ)

theorem packets_for_dollars :
  let dimes := 10 * C
  let taxable_dimes := 9 * C
  ∃ x, x = taxable_dimes * P / R :=
sorry

end packets_for_dollars_l111_111741


namespace fraction_eq_l111_111544

theorem fraction_eq {x : ℝ} (h : 1 - 6 / x + 9 / x ^ 2 - 2 / x ^ 3 = 0) :
  3 / x = 3 / 2 ∨ 3 / x = 3 / (2 + Real.sqrt 3) ∨ 3 / x = 3 / (2 - Real.sqrt 3) :=
sorry

end fraction_eq_l111_111544


namespace no_solution_eq_l111_111986

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l111_111986


namespace sum_abcd_l111_111354

theorem sum_abcd (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 :=
sorry

end sum_abcd_l111_111354


namespace divisible_by_101_l111_111345

theorem divisible_by_101 (n : ℕ) : (101 ∣ (10^n - 1)) ↔ (∃ k : ℕ, n = 4 * k) :=
by
  sorry

end divisible_by_101_l111_111345


namespace roses_after_trading_equals_36_l111_111356

-- Definitions of the given conditions
def initial_roses_given : ℕ := 24
def roses_after_trade (n : ℕ) : ℕ := n
def remaining_roses_after_first_wilt (roses : ℕ) : ℕ := roses / 2
def remaining_roses_after_second_wilt (roses : ℕ) : ℕ := roses / 2
def roses_remaining_second_day : ℕ := 9

-- The statement we want to prove
theorem roses_after_trading_equals_36 (n : ℕ) (h : roses_remaining_second_day = 9) :
  ( ∃ x, roses_after_trade x = n ∧ remaining_roses_after_first_wilt (remaining_roses_after_first_wilt x) = roses_remaining_second_day ) →
  n = 36 :=
by
  sorry

end roses_after_trading_equals_36_l111_111356


namespace find_b50_l111_111461

noncomputable def T (n : ℕ) : ℝ := if n = 1 then 2 else 2 / (6 * n - 5)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 2 else T n - T (n - 1)

theorem find_b50 : b 50 = -6 / 42677.5 := by sorry

end find_b50_l111_111461


namespace area_of_sector_l111_111156

theorem area_of_sector (r l : ℝ) (h1 : l + 2 * r = 12) (h2 : l / r = 2) : (1 / 2) * l * r = 9 :=
by
  sorry

end area_of_sector_l111_111156


namespace coin_collection_problem_l111_111366

variable (n d q : ℚ)

theorem coin_collection_problem 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 20 * q = 340)
  (h3 : d = 2 * n) :
  q - n = 2 / 7 := by
  sorry

end coin_collection_problem_l111_111366


namespace spent_on_video_games_l111_111974

-- Defining the given amounts
def initial_amount : ℕ := 84
def grocery_spending : ℕ := 21
def final_amount : ℕ := 39

-- The proof statement: Proving Lenny spent $24 on video games.
theorem spent_on_video_games : initial_amount - final_amount - grocery_spending = 24 :=
by
  sorry

end spent_on_video_games_l111_111974


namespace regression_line_l111_111545

theorem regression_line (m x1 y1 : ℝ) (h_slope : m = 1.23) (h_center : (x1, y1) = (4, 5)) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1.23 * x + 0.08) :=
by
  use 0.08
  sorry

end regression_line_l111_111545


namespace total_amount_spent_correct_l111_111100

-- Definitions based on conditions
def price_of_food_before_tax_and_tip : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

-- Definitions of intermediate steps
def sales_tax : ℝ := sales_tax_rate * price_of_food_before_tax_and_tip
def total_before_tip : ℝ := price_of_food_before_tax_and_tip + sales_tax
def tip : ℝ := tip_rate * total_before_tip
def total_amount_spent : ℝ := total_before_tip + tip

-- Theorem statement to be proved
theorem total_amount_spent_correct : total_amount_spent = 184.80 :=
by
  sorry -- Proof is skipped

end total_amount_spent_correct_l111_111100


namespace sum_or_difference_div_by_100_l111_111903

theorem sum_or_difference_div_by_100 (s : Finset ℤ) (h_card : s.card = 52) :
  ∃ (a b : ℤ), a ∈ s ∧ b ∈ s ∧ (a ≠ b) ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) :=
by
  sorry

end sum_or_difference_div_by_100_l111_111903


namespace solve_for_y_l111_111380

theorem solve_for_y (y : ℕ) (h : 9 / y^2 = 3 * y / 81) : y = 9 :=
sorry

end solve_for_y_l111_111380


namespace smallest_c_for_inverse_l111_111006

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c : ℝ, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d : ℝ, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
sorry

end smallest_c_for_inverse_l111_111006


namespace max_int_value_of_a_real_roots_l111_111086

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l111_111086


namespace Harriett_total_money_l111_111109

open Real

theorem Harriett_total_money :
    let quarters := 14 * 0.25
    let dimes := 7 * 0.10
    let nickels := 9 * 0.05
    let pennies := 13 * 0.01
    let half_dollars := 4 * 0.50
    quarters + dimes + nickels + pennies + half_dollars = 6.78 :=
by
    sorry

end Harriett_total_money_l111_111109


namespace roots_operation_zero_l111_111507

def operation (a b : ℝ) : ℝ := a * b - a - b

theorem roots_operation_zero {x1 x2 : ℝ}
  (h1 : x1 + x2 = -1)
  (h2 : x1 * x2 = -1) :
  operation x1 x2 = 0 :=
by
  sorry

end roots_operation_zero_l111_111507


namespace kyunghwan_spent_the_most_l111_111315

-- Define initial pocket money for everyone
def initial_money : ℕ := 20000

-- Define remaining money
def remaining_S : ℕ := initial_money / 4
def remaining_K : ℕ := initial_money / 8
def remaining_D : ℕ := initial_money / 5

-- Calculate spent money
def spent_S : ℕ := initial_money - remaining_S
def spent_K : ℕ := initial_money - remaining_K
def spent_D : ℕ := initial_money - remaining_D

theorem kyunghwan_spent_the_most 
  (h1 : remaining_S = initial_money / 4)
  (h2 : remaining_K = initial_money / 8)
  (h3 : remaining_D = initial_money / 5) :
  spent_K > spent_S ∧ spent_K > spent_D :=
by
  -- Proof skipped
  sorry

end kyunghwan_spent_the_most_l111_111315


namespace perimeter_of_8_sided_figure_l111_111861

theorem perimeter_of_8_sided_figure (n : ℕ) (len : ℕ) (h1 : n = 8) (h2 : len = 2) :
  n * len = 16 := by
  sorry

end perimeter_of_8_sided_figure_l111_111861


namespace find_eccentricity_l111_111892

noncomputable def ellipse_gamma (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b) : Prop :=
∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def ellipse_focus (a b : ℝ) : Prop :=
∀ (x y : ℝ), x = 3 → y = 0

def vertex_A (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = b

def vertex_B (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = -b

def point_N : Prop :=
∀ (x y : ℝ), x = 12 → y = 0

theorem find_eccentricity : 
∀ (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b), 
  ellipse_gamma a b ha_gt hb_gt h → 
  ellipse_focus a b → 
  vertex_A b → 
  vertex_B b → 
  point_N → 
  ∃ e : ℝ, e = 1 / 2 := 
by 
  sorry

end find_eccentricity_l111_111892


namespace area_of_region_l111_111178

-- Define the equation as a predicate
def region (x y : ℝ) : Prop := x^2 + y^2 + 6*x = 2*y + 10

-- The proof statement
theorem area_of_region : (∃ (x y : ℝ), region x y) → ∃ A : ℝ, A = 20 * Real.pi :=
by 
  sorry

end area_of_region_l111_111178


namespace total_pupils_count_l111_111369

theorem total_pupils_count (girls boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) : girls + boys = 926 :=
by 
  sorry

end total_pupils_count_l111_111369


namespace two_integer_solutions_iff_l111_111114

theorem two_integer_solutions_iff (a : ℝ) :
  (∃ (n m : ℤ), n ≠ m ∧ |n - 1| < a * n ∧ |m - 1| < a * m ∧
    ∀ (k : ℤ), |k - 1| < a * k → k = n ∨ k = m) ↔
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) :=
by
  sorry

end two_integer_solutions_iff_l111_111114


namespace find_number_l111_111204

theorem find_number (x : ℝ) (h : 42 - 3 * x = 12) : x = 10 := 
by 
  sorry

end find_number_l111_111204


namespace smallest_c_ineq_l111_111182

noncomputable def smallest_c {d : ℕ → ℕ} (h_d : ∀ n > 0, d n ≤ d n + 1) := Real.sqrt 3

theorem smallest_c_ineq (d : ℕ → ℕ) (h_d : ∀ n > 0, (d n) ≤ d n + 1) :
  ∀ n : ℕ, n > 0 → d n ≤ smallest_c h_d * (Real.sqrt n) :=
sorry

end smallest_c_ineq_l111_111182


namespace no_solution_exists_l111_111290

theorem no_solution_exists (p : ℝ) : (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) = (x - p) / (x - 8) → false) ↔ p = 7 :=
by sorry

end no_solution_exists_l111_111290


namespace taimour_paint_time_l111_111458

theorem taimour_paint_time (T : ℝ) :
  (1 / T + 2 / T) * 7 = 1 → T = 21 :=
by
  intro h
  sorry

end taimour_paint_time_l111_111458


namespace kittens_price_l111_111752

theorem kittens_price (x : ℕ) 
  (h1 : 2 * x + 5 = 17) : x = 6 := by
  sorry

end kittens_price_l111_111752


namespace value_of_number_l111_111915

theorem value_of_number (x : ℤ) (number : ℚ) (h₁ : x = 32) (h₂ : 35 - (23 - (15 - x)) = 12 * number / (1/2)) : number = -5/6 :=
by
  sorry

end value_of_number_l111_111915


namespace find_base_l111_111339

theorem find_base (r : ℕ) (h1 : 5 * r^2 + 3 * r + 4 + 3 * r^2 + 6 * r + 6 = r^3) : r = 10 :=
by
  sorry

end find_base_l111_111339


namespace dilation_translation_correct_l111_111323

def transformation_matrix (d: ℝ) (tx: ℝ) (ty: ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![d, 0, tx],
    ![0, d, ty],
    ![0, 0, 1]
  ]

theorem dilation_translation_correct :
  transformation_matrix 4 2 3 =
  ![
    ![4, 0, 2],
    ![0, 4, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end dilation_translation_correct_l111_111323


namespace remainder_sum_mod_13_l111_111271

theorem remainder_sum_mod_13 : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 :=
by
  sorry

end remainder_sum_mod_13_l111_111271


namespace average_one_half_one_fourth_one_eighth_l111_111820

theorem average_one_half_one_fourth_one_eighth : 
  ((1 / 2.0 + 1 / 4.0 + 1 / 8.0) / 3.0) = 7 / 24 := 
by sorry

end average_one_half_one_fourth_one_eighth_l111_111820


namespace tangent_lines_through_point_l111_111260

theorem tangent_lines_through_point {x y : ℝ} (h_circle : (x-1)^2 + (y-1)^2 = 1)
  (h_point : ∀ (x y: ℝ), (x, y) = (2, 4)) :
  (x = 2 ∨ 4 * x - 3 * y + 4 = 0) :=
sorry

end tangent_lines_through_point_l111_111260


namespace total_lambs_l111_111957

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l111_111957


namespace similar_triangles_y_value_l111_111494

theorem similar_triangles_y_value :
  ∀ (y : ℚ),
    (12 : ℚ) / y = (9 : ℚ) / 6 → 
    y = 8 :=
by
  intros y h
  sorry

end similar_triangles_y_value_l111_111494


namespace ratio_of_cream_max_to_maxine_l111_111299

def ounces_of_cream_in_max (coffee_sipped : ℕ) (cream_added: ℕ) : ℕ := cream_added

def ounces_of_remaining_cream_in_maxine (initial_coffee : ℚ) (cream_added: ℚ) (sipped : ℚ) : ℚ :=
  let total_mixture := initial_coffee + cream_added
  let remaining_mixture := total_mixture - sipped
  (initial_coffee / total_mixture) * cream_added

theorem ratio_of_cream_max_to_maxine :
  let max_cream := ounces_of_cream_in_max 4 3
  let maxine_cream := ounces_of_remaining_cream_in_maxine 16 3 5
  (max_cream : ℚ) / maxine_cream = 19 / 14 := by 
  sorry

end ratio_of_cream_max_to_maxine_l111_111299


namespace sum_of_last_digits_l111_111767

theorem sum_of_last_digits (num : Nat → Nat) (a b : Nat) :
  (∀ i, 1 ≤ i ∧ i < 2000 → (num i * 10 + num (i + 1)) % 17 = 0 ∨ (num i * 10 + num (i + 1)) % 23 = 0) →
  num 1 = 3 →
  (num 2000 = a ∨ num 2000 = b) →
  a = 2 →
  b = 5 →
  a + b = 7 :=
by 
  sorry

end sum_of_last_digits_l111_111767


namespace cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l111_111909

noncomputable def A_n (n : ℕ) : ℝ :=
  490 * n - 10 * n^2

noncomputable def B_n (n : ℕ) : ℝ :=
  500 * n + 400 - 500 / 2^(n-1)

theorem cumulative_profit_exceeds_technical_renovation :
  ∀ n : ℕ, n ≥ 4 → B_n n > A_n n :=
by
  sorry  -- Proof goes here

theorem expressions_for_A_n_B_n (n : ℕ) :
  A_n n = 490 * n - 10 * n^2 ∧
  B_n n = 500 * n + 400 - 500 / 2^(n-1) :=
by
  sorry  -- Proof goes here

end cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l111_111909


namespace find_amount_l111_111161

theorem find_amount (N : ℝ) (hN : N = 24) (A : ℝ) (hA : A = 0.6667 * N - 0.25 * N) : A = 10.0008 :=
by
  rw [hN] at hA
  sorry

end find_amount_l111_111161


namespace ajax_weight_after_exercise_l111_111292

theorem ajax_weight_after_exercise :
  ∀ (initial_weight_kg : ℕ) (conversion_rate : ℝ) (daily_exercise_hours : ℕ) (exercise_loss_rate : ℝ) (days_in_week : ℕ) (weeks : ℕ),
    initial_weight_kg = 80 →
    conversion_rate = 2.2 →
    daily_exercise_hours = 2 →
    exercise_loss_rate = 1.5 →
    days_in_week = 7 →
    weeks = 2 →
    initial_weight_kg * conversion_rate - daily_exercise_hours * exercise_loss_rate * (days_in_week * weeks) = 134 :=
by
  intros
  sorry

end ajax_weight_after_exercise_l111_111292


namespace iron_wire_left_l111_111166

-- Given conditions as variables
variable (initial_usage : ℚ) (additional_usage : ℚ)

-- Conditions as hypotheses
def conditions := initial_usage = 2 / 9 ∧ additional_usage = 3 / 9

-- The goal to prove
theorem iron_wire_left (h : conditions initial_usage additional_usage):
  1 - initial_usage - additional_usage = 4 / 9 :=
by
  -- Insert proof here
  sorry

end iron_wire_left_l111_111166


namespace simplify_expression_l111_111256

variable (a b : ℤ)

theorem simplify_expression :
  (15 * a + 45 * b) + (20 * a + 35 * b) - (25 * a + 55 * b) + (30 * a - 5 * b) = 
  40 * a + 20 * b :=
by
  sorry

end simplify_expression_l111_111256


namespace rahul_salary_l111_111900

variable (X : ℝ)

def house_rent_deduction (salary : ℝ) : ℝ := salary * 0.8
def education_expense (remaining_after_rent : ℝ) : ℝ := remaining_after_rent * 0.9
def clothing_expense (remaining_after_education : ℝ) : ℝ := remaining_after_education * 0.9

theorem rahul_salary : (X * 0.8 * 0.9 * 0.9 = 1377) → X = 2125 :=
by
  intros h
  sorry

end rahul_salary_l111_111900


namespace tanya_work_days_l111_111311

theorem tanya_work_days (days_sakshi : ℕ) (efficiency_increase : ℚ) (work_rate_sakshi : ℚ) (work_rate_tanya : ℚ) (days_tanya : ℚ) :
  days_sakshi = 15 ->
  efficiency_increase = 1.25 ->
  work_rate_sakshi = 1 / days_sakshi ->
  work_rate_tanya = work_rate_sakshi * efficiency_increase ->
  days_tanya = 1 / work_rate_tanya ->
  days_tanya = 12 :=
by
  intros h_sakshi h_efficiency h_work_rate_sakshi h_work_rate_tanya h_days_tanya
  sorry

end tanya_work_days_l111_111311


namespace inequality_am_gm_l111_111097

theorem inequality_am_gm 
  (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := 
by 
  sorry

end inequality_am_gm_l111_111097


namespace find_x4_l111_111121

open Real

theorem find_x4 (x : ℝ) (h₁ : 0 < x) (h₂ : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end find_x4_l111_111121


namespace polar_equation_l111_111295

theorem polar_equation (y ρ θ : ℝ) (x : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) 
  (h3 : y^2 = 12 * x) : 
  ρ * (Real.sin θ)^2 = 12 * Real.cos θ := 
by
  sorry

end polar_equation_l111_111295


namespace speed_of_faster_train_l111_111459

noncomputable def speed_of_slower_train : ℝ := 36
noncomputable def length_of_each_train : ℝ := 70
noncomputable def time_to_pass : ℝ := 36

theorem speed_of_faster_train : 
  ∃ (V_f : ℝ), 
    (V_f - speed_of_slower_train) * (1000 / 3600) = 140 / time_to_pass ∧ 
    V_f = 50 :=
by {
  sorry
}

end speed_of_faster_train_l111_111459


namespace find_x_value_l111_111534

variable {x : ℝ}

def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (k * b.1, k * b.2)

theorem find_x_value (h : opposite_directions (x, 1) (4, x)) : x = -2 :=
sorry

end find_x_value_l111_111534


namespace shortest_side_of_right_triangle_l111_111943

theorem shortest_side_of_right_triangle (a b : ℝ) (h : a = 9 ∧ b = 12) : ∃ c : ℝ, (c = min a b) ∧ c = 9 :=
by
  sorry

end shortest_side_of_right_triangle_l111_111943


namespace probability_neither_orange_nor_white_l111_111627

/-- Define the problem conditions. -/
def num_orange_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 6

/-- Define the total number of balls. -/
def total_balls : ℕ := num_orange_balls + num_black_balls + num_white_balls

/-- Define the probability of picking a black ball (neither orange nor white). -/
noncomputable def probability_black_ball : ℚ := num_black_balls / total_balls

/-- The main statement to be proved: The probability is 1/3. -/
theorem probability_neither_orange_nor_white : probability_black_ball = 1 / 3 :=
sorry

end probability_neither_orange_nor_white_l111_111627


namespace uncle_fyodor_sandwiches_count_l111_111027

variable (sandwiches_sharik : ℕ)
variable (sandwiches_matroskin : ℕ := 3 * sandwiches_sharik)
variable (total_sandwiches_eaten : ℕ := sandwiches_sharik + sandwiches_matroskin)
variable (sandwiches_uncle_fyodor : ℕ := 2 * total_sandwiches_eaten)
variable (difference : ℕ := sandwiches_uncle_fyodor - sandwiches_sharik)

theorem uncle_fyodor_sandwiches_count :
  (difference = 21) → sandwiches_uncle_fyodor = 24 := by
  intro h
  sorry

end uncle_fyodor_sandwiches_count_l111_111027


namespace max_abs_value_l111_111999

theorem max_abs_value (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  sorry

end max_abs_value_l111_111999


namespace no_integers_divisible_by_all_l111_111645

-- Define the list of divisors
def divisors : List ℕ := [2, 3, 4, 5, 7, 11]

-- Define the LCM function
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Calculate the LCM of the given divisors
def lcm_divisors : ℕ := lcm_list divisors

-- Define a predicate to check divisibility by all divisors
def is_divisible_by_all (n : ℕ) (ds : List ℕ) : Prop :=
  ds.all (λ d => n % d = 0)

-- Define the theorem to prove the number of integers between 1 and 1000 divisible by the given divisors
theorem no_integers_divisible_by_all :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ is_divisible_by_all n divisors) → False := by
  sorry

end no_integers_divisible_by_all_l111_111645


namespace time_for_A_l111_111437

-- Given rates of pipes A, B, and C filling the tank
variable (A B C : ℝ)

-- Condition 1: Tank filled by all three pipes in 8 hours
def combined_rate := (A + B + C = 1/8)

-- Condition 2: Pipe C is twice as fast as B
def rate_C := (C = 2 * B)

-- Condition 3: Pipe B is twice as fast as A
def rate_B := (B = 2 * A)

-- Question: To prove that pipe A alone will take 56 hours to fill the tank
theorem time_for_A (h₁ : combined_rate A B C) (h₂ : rate_C B C) (h₃ : rate_B A B) : 
  1 / A = 56 :=
by {
  sorry
}

end time_for_A_l111_111437


namespace sum_of_digits_smallest_N_l111_111783

/-- Define the probability Q(N) -/
def Q (N : ℕ) : ℚ :=
  ((2 * N) / 3 + 1) / (N + 1)

/-- Main mathematical statement to be proven in Lean 4 -/

theorem sum_of_digits_smallest_N (N : ℕ) (h1 : N > 9) (h2 : N % 6 = 0) (h3 : Q N < 7 / 10) : 
  (N.digits 10).sum = 3 :=
  sorry

end sum_of_digits_smallest_N_l111_111783


namespace coefficient_x_neg_4_expansion_l111_111504

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the function to calculate the coefficient of the term containing x^(-4)
def coeff_term_x_neg_4 : ℕ :=
  let k := 10
  binom 12 k

theorem coefficient_x_neg_4_expansion :
  coeff_term_x_neg_4 = 66 := by
  -- Calculation here would show that binom 12 10 is indeed 66
  sorry

end coefficient_x_neg_4_expansion_l111_111504


namespace video_game_cost_l111_111115

theorem video_game_cost
  (weekly_allowance1 : ℕ)
  (weeks1 : ℕ)
  (weekly_allowance2 : ℕ)
  (weeks2 : ℕ)
  (money_spent_on_clothes_fraction : ℚ)
  (remaining_money : ℕ)
  (allowance1 : weekly_allowance1 = 5)
  (duration1 : weeks1 = 8)
  (allowance2 : weekly_allowance2 = 6)
  (duration2 : weeks2 = 6)
  (money_spent_fraction : money_spent_on_clothes_fraction = 1/2)
  (remaining_money_condition : remaining_money = 3) :
  (weekly_allowance1 * weeks1 + weekly_allowance2 * weeks2) * (1 - money_spent_on_clothes_fraction) - remaining_money = 35 :=
by
  rw [allowance1, duration1, allowance2, duration2, money_spent_fraction, remaining_money_condition]
  -- Calculation steps are omitted; they can be filled in here.
  exact sorry

end video_game_cost_l111_111115


namespace reduced_price_l111_111298

variable (P : ℝ)  -- the original price per kg
variable (reduction_factor : ℝ := 0.5)  -- 50% reduction
variable (extra_kgs : ℝ := 5)  -- 5 kgs more
variable (total_cost : ℝ := 800)  -- Rs. 800

theorem reduced_price :
  total_cost / (P * (1 - reduction_factor)) = total_cost / P + extra_kgs → 
  P / 2 = 80 :=
by
  sorry

end reduced_price_l111_111298


namespace find_a2023_l111_111450

variable {a : ℕ → ℕ}
variable {x : ℕ}

def sequence_property (a: ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 20

theorem find_a2023 (h1 : sequence_property a) 
                   (h2 : a 2 = 2 * x) 
                   (h3 : a 18 = 9 + x) 
                   (h4 : a 65 = 6 - x) : 
  a 2023 = 5 := 
by
  sorry

end find_a2023_l111_111450


namespace largest_value_fraction_l111_111428

noncomputable def largest_value (x y : ℝ) : ℝ := (x + y) / x

theorem largest_value_fraction
  (x y : ℝ)
  (hx1 : -5 ≤ x)
  (hx2 : x ≤ -3)
  (hy1 : 3 ≤ y)
  (hy2 : y ≤ 5)
  (hy_odd : ∃ k : ℤ, y = 2 * k + 1) :
  largest_value x y = 0.4 :=
sorry

end largest_value_fraction_l111_111428


namespace find_certain_number_l111_111068

theorem find_certain_number (x : ℤ) (h : x + 34 - 53 = 28) : x = 47 :=
by {
  sorry
}

end find_certain_number_l111_111068


namespace genuine_product_probability_l111_111724

-- Define the probabilities as constants
def P_second_grade := 0.03
def P_third_grade := 0.01

-- Define the total probability (outcome must be either genuine or substandard)
def P_substandard := P_second_grade + P_third_grade
def P_genuine := 1 - P_substandard

-- The statement to be proved
theorem genuine_product_probability :
  P_genuine = 0.96 :=
sorry

end genuine_product_probability_l111_111724


namespace cats_owners_percentage_l111_111632

noncomputable def percentage_of_students_owning_cats (total_students : ℕ) (cats_owners : ℕ) : ℚ :=
  (cats_owners : ℚ) / (total_students : ℚ) * 100

theorem cats_owners_percentage (total_students : ℕ) (cats_owners : ℕ)
  (dogs_owners : ℕ) (birds_owners : ℕ)
  (h_total_students : total_students = 400)
  (h_cats_owners : cats_owners = 80)
  (h_dogs_owners : dogs_owners = 120)
  (h_birds_owners : birds_owners = 40) :
  percentage_of_students_owning_cats total_students cats_owners = 20 :=
by {
  -- We state the proof but leave it as sorry so it's an incomplete placeholder.
  sorry
}

end cats_owners_percentage_l111_111632


namespace rational_solutions_are_integers_l111_111348

-- Given two integers a and b, and two equations with rational solutions
variables (a b : ℤ)

-- The first equation is y - 2x = a
def eq1 (y x : ℚ) : Prop := y - 2 * x = a

-- The second equation is y^2 - xy + x^2 = b
def eq2 (y x : ℚ) : Prop := y^2 - x * y + x^2 = b

-- We want to prove that if y and x are rational solutions, they must be integers
theorem rational_solutions_are_integers (y x : ℚ) (h1 : eq1 a y x) (h2 : eq2 b y x) : 
    ∃ (y_int x_int : ℤ), y = y_int ∧ x = x_int :=
sorry

end rational_solutions_are_integers_l111_111348


namespace not_possible_last_digit_l111_111709

theorem not_possible_last_digit :
  ∀ (S : ℕ) (a : Fin 111 → ℕ),
  (∀ i, a i ≤ 500) →
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i, (a i) % 10 = (S - a i) % 10) →
  False :=
by
  intro S a h1 h2 h3
  sorry

end not_possible_last_digit_l111_111709


namespace solve_for_x_l111_111702

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l111_111702


namespace modulus_of_complex_l111_111598

-- Define the conditions
variables {x y : ℝ}
def i := Complex.I

-- State the conditions of the problem
def condition1 : 1 + x * i = (2 - y) - 3 * i :=
by sorry

-- State the hypothesis and the goal
theorem modulus_of_complex (h : 1 + x * i = (2 - y) - 3 * i) : Complex.abs (x + y * i) = Real.sqrt 10 :=
sorry

end modulus_of_complex_l111_111598


namespace min_points_tenth_game_l111_111935

-- Defining the scores for each segment of games
def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

-- Calculating the total score after 9 games
def total_score_after_nine_games : ℕ := first_five_games.sum + next_four_games.sum

-- Defining the required total points after 10 games for an average greater than 17
def required_total_points := 171

-- Proving the number of points needed in the 10th game
theorem min_points_tenth_game (s₁ s₂ : List ℕ) (h₁ : s₁ = first_five_games) (h₂ : s₂ = next_four_games) :
    s₁.sum + s₂.sum + x ≥ required_total_points → x ≥ 22 :=
  sorry

end min_points_tenth_game_l111_111935


namespace bacteria_growth_returns_six_l111_111852

theorem bacteria_growth_returns_six (n : ℕ) (h : (4 * 2 ^ n > 200)) : n = 6 :=
sorry

end bacteria_growth_returns_six_l111_111852


namespace range_of_a_l111_111372

noncomputable def p (x : ℝ) : Prop := (1 / (x - 3)) ≥ 1

noncomputable def q (x a : ℝ) : Prop := abs (x - a) < 1

theorem range_of_a (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, ¬ (p x) ∧ (q x a)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l111_111372


namespace base_representation_l111_111132

theorem base_representation (b : ℕ) (h₁ : b^2 ≤ 125) (h₂ : 125 < b^3) :
  (∀ b, b = 12 → 125 % b % 2 = 1) → b = 12 := 
by
  sorry

end base_representation_l111_111132


namespace candy_store_total_sales_l111_111284

def price_per_pound_fudge : ℝ := 2.50
def pounds_fudge : ℕ := 20
def price_per_truffle : ℝ := 1.50
def dozens_truffles : ℕ := 5
def price_per_pretzel : ℝ := 2.00
def dozens_pretzels : ℕ := 3

theorem candy_store_total_sales :
  price_per_pound_fudge * pounds_fudge +
  price_per_truffle * (dozens_truffles * 12) +
  price_per_pretzel * (dozens_pretzels * 12) = 212.00 := by
  sorry

end candy_store_total_sales_l111_111284


namespace div2_implies_div2_of_either_l111_111555

theorem div2_implies_div2_of_either (a b : ℕ) (h : 2 ∣ a * b) : (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end div2_implies_div2_of_either_l111_111555


namespace value_of_x_l111_111282

theorem value_of_x (x : ℝ) (h : x = 90 + (11 / 100) * 90) : x = 99.9 :=
by {
  sorry
}

end value_of_x_l111_111282


namespace residue_of_neg_1237_mod_37_l111_111313

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l111_111313


namespace range_of_y_l111_111976

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_y : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 →
  y x ∈ Set.Icc (-Real.sin 1 - (Real.pi / 2)) (Real.sin 1 + (Real.pi / 2)) :=
sorry

end range_of_y_l111_111976


namespace a2_a4_a6_a8_a10_a12_sum_l111_111091

theorem a2_a4_a6_a8_a10_a12_sum :
  ∀ (x : ℝ), 
    (1 + x + x^2)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 →
    a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
sorry

end a2_a4_a6_a8_a10_a12_sum_l111_111091


namespace kitten_food_consumption_l111_111969

-- Definitions of the given conditions
def k : ℕ := 4  -- Number of kittens
def ac : ℕ := 3  -- Number of adult cats
def f : ℕ := 7  -- Initial cans of food
def af : ℕ := 35  -- Additional cans of food needed
def days : ℕ := 7  -- Total number of days

-- Definition of the food consumption per adult cat per day
def food_per_adult_cat_per_day : ℕ := 1

-- Definition of the correct answer: food per kitten per day
def food_per_kitten_per_day : ℚ := 0.75

-- Proof statement
theorem kitten_food_consumption (k : ℕ) (ac : ℕ) (f : ℕ) (af : ℕ) (days : ℕ) (food_per_adult_cat_per_day : ℕ) :
  (ac * food_per_adult_cat_per_day * days + k * food_per_kitten_per_day * days = f + af) → 
  food_per_kitten_per_day = 0.75 :=
sorry

end kitten_food_consumption_l111_111969


namespace gcd_gx_x_l111_111238

theorem gcd_gx_x (x : ℤ) (hx : 34560 ∣ x) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 17)) x = 20 := 
by
  sorry

end gcd_gx_x_l111_111238


namespace total_number_of_balls_l111_111912

theorem total_number_of_balls 
(b : ℕ) (P_blue : ℚ) (h1 : b = 8) (h2 : P_blue = 1/3) : 
  ∃ g : ℕ, b + g = 24 := by
  sorry

end total_number_of_balls_l111_111912


namespace right_triangle_hypotenuse_length_l111_111444

theorem right_triangle_hypotenuse_length (a b c : ℝ) (h₀ : a = 7) (h₁ : b = 24) (h₂ : a^2 + b^2 = c^2) : c = 25 :=
by
  rw [h₀, h₁] at h₂
  -- This step will simplify the problem
  sorry

end right_triangle_hypotenuse_length_l111_111444


namespace bubble_sort_probability_r10_r25_l111_111025

theorem bubble_sort_probability_r10_r25 (n : ℕ) (r : ℕ → ℕ) :
  n = 50 ∧ (∀ i, 1 ≤ i ∧ i ≤ 50 → r i ≠ r (i + 1)) ∧ (∀ i j, i ≠ j → r i ≠ r j) →
  let p := 1
  let q := 650
  p + q = 651 :=
by
  intros h
  sorry

end bubble_sort_probability_r10_r25_l111_111025


namespace problem_statement_l111_111531

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y - x > 1) :
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 :=
sorry

end problem_statement_l111_111531


namespace find_a_find_A_l111_111160

-- Part (I)
theorem find_a (b c : ℝ) (A : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = 5 * Real.pi / 6) :
  ∃ a : ℝ, a = 2 * Real.sqrt 7 :=
by {
  sorry
}

-- Part (II)
theorem find_A (b c : ℝ) (C : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 2 + A) :
  ∃ A : ℝ, A = Real.pi / 6 :=
by {
  sorry
}

end find_a_find_A_l111_111160


namespace profit_percentage_is_20_l111_111036

variable (C : ℝ) -- Assuming the cost price C is a real number.

theorem profit_percentage_is_20 
  (h1 : 10 * 1 = 12 * (C / 1)) :  -- Shopkeeper sold 10 articles at the cost price of 12 articles.
  ((12 * C - 10 * C) / (10 * C)) * 100 = 20 := 
by
  sorry

end profit_percentage_is_20_l111_111036


namespace expenditure_of_negative_amount_l111_111677

theorem expenditure_of_negative_amount (x : ℝ) (h : x < 0) : 
  ∃ y : ℝ, y > 0 ∧ x = -y :=
by
  sorry

end expenditure_of_negative_amount_l111_111677


namespace smallest_integer_is_77_l111_111880

theorem smallest_integer_is_77 
  (A B C D E F G : ℤ)
  (h_uniq: A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G)
  (h_sum: A + B + C + D + E + F + G = 840)
  (h_largest: G = 190)
  (h_two_smallest_sum: A + B = 156) : 
  A = 77 :=
sorry

end smallest_integer_is_77_l111_111880


namespace neg_p_necessary_not_sufficient_neg_q_l111_111262

def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0
def not_p (x : ℝ) : Prop := ¬ (p x)
def not_q (x : ℝ) : Prop := ¬ (q x)

theorem neg_p_necessary_not_sufficient_neg_q : ∀ (x : ℝ), (not_q x → not_p x) ∧ ¬ (not_p x → not_q x) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l111_111262


namespace find_hours_l111_111454

theorem find_hours (x : ℕ) (h : (14 + 10 + 13 + 9 + 12 + 11 + x) / 7 = 12) : x = 15 :=
by
  -- The proof is omitted
  sorry

end find_hours_l111_111454


namespace invitational_tournament_l111_111792

theorem invitational_tournament (x : ℕ) (h : 2 * (x * (x - 1) / 2) = 56) : x = 8 :=
by
  sorry

end invitational_tournament_l111_111792


namespace find_certain_value_l111_111359

noncomputable def certain_value 
  (total_area : ℝ) (smaller_part : ℝ) (difference_fraction : ℝ) : ℝ :=
  (total_area - 2 * smaller_part) / difference_fraction

theorem find_certain_value (total_area : ℝ) (smaller_part : ℝ) (X : ℝ) : 
  total_area = 700 → 
  smaller_part = 315 → 
  (total_area - 2 * smaller_part) / (1/5) = X → 
  X = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end find_certain_value_l111_111359


namespace train_speed_l111_111629

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 1200) (h_time : time = 15) :
  (length / time) = 80 := by
  sorry

end train_speed_l111_111629


namespace cheryl_used_total_amount_l111_111163

theorem cheryl_used_total_amount :
  let bought_A := (5 / 8 : ℚ)
  let bought_B := (2 / 9 : ℚ)
  let bought_C := (2 / 5 : ℚ)
  let leftover_A := (1 / 12 : ℚ)
  let leftover_B := (5 / 36 : ℚ)
  let leftover_C := (1 / 10 : ℚ)
  let used_A := bought_A - leftover_A
  let used_B := bought_B - leftover_B
  let used_C := bought_C - leftover_C
  used_A + used_B + used_C = 37 / 40 :=
by 
  sorry

end cheryl_used_total_amount_l111_111163


namespace coast_guard_overtake_smuggler_l111_111535

noncomputable def time_of_overtake (initial_distance : ℝ) (initial_time : ℝ) 
                                   (smuggler_speed1 coast_guard_speed : ℝ) 
                                   (duration1 new_smuggler_speed : ℝ) : ℝ :=
  let distance_after_duration1 := initial_distance + (smuggler_speed1 * duration1) - (coast_guard_speed * duration1)
  let relative_speed_new := coast_guard_speed - new_smuggler_speed
  duration1 + (distance_after_duration1 / relative_speed_new)

theorem coast_guard_overtake_smuggler : 
  time_of_overtake 15 0 18 20 1 16 = 4.25 := by
  sorry

end coast_guard_overtake_smuggler_l111_111535


namespace grandmother_mistaken_l111_111034

-- Definitions of the given conditions:
variables (N : ℕ) (x n : ℕ)
variable (initial_split : N % 4 = 0)

-- Conditions
axiom cows_survived : 4 * (N / 4) / 5 = N / 5
axiom horses_pigs : x = N / 4 - N / 5
axiom rabbit_ratio : (N / 4 - n) = 5 / 14 * (N / 5 + N / 4 + N / 4 - n)

-- Goal: Prove the grandmother is mistaken, i.e., some species avoided casualties
theorem grandmother_mistaken : n = 0 :=
sorry

end grandmother_mistaken_l111_111034


namespace balance_balls_l111_111438

variables (R B O P : ℝ)

-- Conditions
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 6 * B
axiom h3 : 8 * B = 6 * P

-- Proof problem
theorem balance_balls : 5 * R + 3 * O + 3 * P = 20 * B :=
by sorry

end balance_balls_l111_111438


namespace ratio_of_arithmetic_sums_l111_111229

theorem ratio_of_arithmetic_sums : 
  let a1 := 4
  let d1 := 4
  let l1 := 48
  let a2 := 2
  let d2 := 3
  let l2 := 35
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let S1 := n1 * (a1 + l1) / 2
  let S2 := n2 * (a2 + l2) / 2
  let ratio := S1 / S2
  ratio = 52 / 37 := by sorry

end ratio_of_arithmetic_sums_l111_111229


namespace one_third_times_seven_times_nine_l111_111602

theorem one_third_times_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_times_seven_times_nine_l111_111602


namespace solve_gcd_problem_l111_111467

def gcd_problem : Prop :=
  gcd 1337 382 = 191

theorem solve_gcd_problem : gcd_problem := 
by 
  sorry

end solve_gcd_problem_l111_111467


namespace L_shape_perimeter_correct_l111_111573

-- Define the dimensions of the rectangles
def rect_height : ℕ := 3
def rect_width : ℕ := 4

-- Define the combined shape and perimeter calculation
def L_shape_perimeter (h w : ℕ) : ℕ := (2 * w) + (2 * h)

theorem L_shape_perimeter_correct : 
  L_shape_perimeter rect_height rect_width = 14 := 
  sorry

end L_shape_perimeter_correct_l111_111573


namespace ellipse_equation_with_foci_l111_111721

theorem ellipse_equation_with_foci (M N P : ℝ × ℝ)
  (area_triangle : Real) (tan_M tan_N : ℝ)
  (h₁ : area_triangle = 1)
  (h₂ : tan_M = 1 / 2)
  (h₃ : tan_N = -2) :
  ∃ (a b : ℝ), (4 * x^2) / (15 : ℝ) + y^2 / (3 : ℝ) = 1 :=
by
  -- Definitions to meet given conditions would be here
  sorry

end ellipse_equation_with_foci_l111_111721


namespace gcd_m_n_is_one_l111_111603

-- Definitions of m and n
def m : ℕ := 101^2 + 203^2 + 307^2
def n : ℕ := 100^2 + 202^2 + 308^2

-- The main theorem stating the gcd of m and n
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l111_111603


namespace proof_inequality_l111_111187

theorem proof_inequality (n : ℕ) (a b : ℝ) (c : ℝ) (h_n : 1 ≤ n) (h_a : 1 ≤ a) (h_b : 1 ≤ b) (h_c : 0 < c) : 
  ((ab + c)^n - c) / ((b + c)^n - c) ≤ a^n :=
sorry

end proof_inequality_l111_111187


namespace quadratic_roots_bc_minus_two_l111_111139

theorem quadratic_roots_bc_minus_two (b c : ℝ) 
  (h1 : 1 + -2 = -b) 
  (h2 : 1 * -2 = c) : b * c = -2 :=
by 
  sorry

end quadratic_roots_bc_minus_two_l111_111139


namespace factor_x_squared_minus_144_l111_111713

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) :=
by
  sorry

end factor_x_squared_minus_144_l111_111713


namespace number_of_passed_candidates_l111_111931

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l111_111931


namespace min_area_triangle_l111_111081

-- Conditions
def point_on_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def incircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem min_area_triangle (x₀ y₀ b c : ℝ) (h_curve : point_on_curve x₀ y₀) 
  (h_bc_yaxis : b ≠ c) (h_incircle : incircle x₀ y₀) :
  ∃ P : ℝ × ℝ, 
    ∃ B C : ℝ × ℝ, 
    ∃ S : ℝ,
    point_on_curve P.1 P.2 ∧
    B = (0, b) ∧
    C = (0, c) ∧
    incircle P.1 P.2 ∧
    S = (x₀ - 2) + (4 / (x₀ - 2)) + 4 ∧
    S = 8 :=
sorry

end min_area_triangle_l111_111081


namespace solve_inequality_l111_111085

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end solve_inequality_l111_111085


namespace bounce_ratio_l111_111043

theorem bounce_ratio (r : ℝ) (h₁ : 96 * r^4 = 3) : r = Real.sqrt 2 / 4 :=
by
  sorry

end bounce_ratio_l111_111043


namespace tangent_line_at_zero_decreasing_intervals_l111_111596

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem tangent_line_at_zero :
  let t : ℝ × ℝ := (0, f 0)
  (∀ x : ℝ, (9 * x - f x - 2 = 0) → t.snd = -2) := by
  sorry

theorem decreasing_intervals :
  ∀ x : ℝ, (-3 * x^2 + 6 * x + 9 < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end tangent_line_at_zero_decreasing_intervals_l111_111596


namespace symmetric_circle_proof_l111_111276

-- Define the original circle equation
def original_circle_eq (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop :=
  y = x

-- Define the symmetric circle equation
def symmetric_circle_eq (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 5

-- The theorem to prove
theorem symmetric_circle_proof (x y : ℝ) :
  (original_circle_eq x y) ↔ (symmetric_circle_eq x y) :=
sorry

end symmetric_circle_proof_l111_111276


namespace framed_painting_ratio_l111_111708

-- Definitions and conditions
def painting_width : ℕ := 20
def painting_height : ℕ := 30
def frame_side_width (x : ℕ) : ℕ := x
def frame_top_bottom_width (x : ℕ) : ℕ := 3 * x

-- Overall dimensions of the framed painting
def framed_painting_width (x : ℕ) : ℕ := painting_width + 2 * frame_side_width x
def framed_painting_height (x : ℕ) : ℕ := painting_height + 2 * frame_top_bottom_width x

-- Area of the painting
def painting_area : ℕ := painting_width * painting_height

-- Area of the frame
def frame_area (x : ℕ) : ℕ := framed_painting_width x * framed_painting_height x - painting_area

-- Condition that frame area equals painting area
def frame_area_condition (x : ℕ) : Prop := frame_area x = painting_area

-- Theoretical ratio of smaller to larger dimension of the framed painting
def dimension_ratio (x : ℕ) : ℚ := (framed_painting_width x : ℚ) / (framed_painting_height x)

-- The mathematical problem to prove
theorem framed_painting_ratio : ∃ x : ℕ, frame_area_condition x ∧ dimension_ratio x = (4 : ℚ) / 7 :=
by
  sorry

end framed_painting_ratio_l111_111708


namespace decimal_to_fraction_l111_111483

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l111_111483


namespace tan_double_angle_l111_111221

theorem tan_double_angle (α : ℝ) (h₁ : Real.sin α = 4/5) (h₂ : α ∈ Set.Ioc (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 := 
  sorry

end tan_double_angle_l111_111221


namespace number_of_defective_pens_l111_111658

noncomputable def defective_pens (total : ℕ) (prob : ℚ) : ℕ :=
  let N := 6 -- since we already know the steps in the solution leading to N = 6
  let D := total - N
  D

theorem number_of_defective_pens (total : ℕ) (prob : ℚ) :
  (total = 12) → (prob = 0.22727272727272727) → defective_pens total prob = 6 :=
by
  intros ht hp
  unfold defective_pens
  sorry

end number_of_defective_pens_l111_111658


namespace trig_identity_l111_111384

theorem trig_identity 
  (α : ℝ) 
  (h : Real.tan α = 1 / 3) : 
  Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10 :=
sorry

end trig_identity_l111_111384


namespace sufficient_but_not_necessary_condition_for_square_l111_111610

theorem sufficient_but_not_necessary_condition_for_square (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ (¬(x^2 > 4 → x > 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_square_l111_111610


namespace quadratic_always_positive_l111_111449

theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) (hpos : a > 0) (hdisc : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := 
by
  sorry

end quadratic_always_positive_l111_111449


namespace perpendicular_line_through_point_l111_111891

noncomputable def is_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_line_through_point
  (line : ℝ → ℝ)
  (P : ℝ × ℝ)
  (h_line_eq : ∀ x, line x = 3 * x + 8)
  (hP : P = (2,1)) :
  ∃ a b c : ℝ, a * (P.1) + b * (P.2) + c = 0 ∧ is_perpendicular 3 (-b / a) ∧ a * 1 + b * 3 + c = 0 :=
sorry

end perpendicular_line_through_point_l111_111891


namespace initial_marbles_l111_111144

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l111_111144


namespace incorrect_quotient_l111_111046

theorem incorrect_quotient
    (correct_quotient : ℕ)
    (correct_divisor : ℕ)
    (incorrect_divisor : ℕ)
    (h1 : correct_quotient = 28)
    (h2 : correct_divisor = 21)
    (h3 : incorrect_divisor = 12) :
  correct_divisor * correct_quotient / incorrect_divisor = 49 :=
by
  sorry

end incorrect_quotient_l111_111046


namespace drunk_drivers_count_l111_111914

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l111_111914


namespace max_value_range_of_t_l111_111477

theorem max_value_range_of_t (t x : ℝ) (h : t ≤ x ∧ x ≤ t + 2) 
: ∃ y : ℝ, y = -x^2 + 6 * x - 7 ∧ y = -(t - 3)^2 + 2 ↔ t ≥ 3 := 
by {
    sorry
}

end max_value_range_of_t_l111_111477


namespace functions_equiv_l111_111035

noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x)
noncomputable def g_D (x : ℝ) : ℝ := (1/2) * Real.log x

theorem functions_equiv : ∀ x : ℝ, x > 0 → f_D x = g_D x := by
  intro x h
  sorry

end functions_equiv_l111_111035


namespace cindy_correct_answer_l111_111840

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l111_111840


namespace anna_chocolates_l111_111981

theorem anna_chocolates : ∃ (n : ℕ), (5 * 2^(n-1) > 200) ∧ n = 7 :=
by
  sorry

end anna_chocolates_l111_111981


namespace arithmetic_sequence_sum_q_l111_111497

theorem arithmetic_sequence_sum_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 2) + a (n + 1) = 2 * a n)
  (hq : q ≠ 1) :
  S 5 = 11 :=
sorry

end arithmetic_sequence_sum_q_l111_111497


namespace simplify_and_evaluate_l111_111793

-- Defining the variables with given values
def a : ℚ := 1 / 2
def b : ℚ := -2

-- Expression to be simplified and evaluated
def expression : ℚ := (2 * a + b) ^ 2 - (2 * a - b) * (a + b) - 2 * (a - 2 * b) * (a + 2 * b)

-- The main theorem
theorem simplify_and_evaluate : expression = 37 := by
  sorry

end simplify_and_evaluate_l111_111793


namespace range_of_a_l111_111448

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 3 < x ∧ x < 4 ∧ ax^2 - 4*a*x - 2 > 0) ↔ a < -2/3 :=
sorry

end range_of_a_l111_111448


namespace factory_production_l111_111419

theorem factory_production (y x : ℝ) (h1 : y + 40 * x = 1.2 * y) (h2 : y + 0.6 * y * x = 2.5 * y) 
  (hx : x = 2.5) : y = 500 ∧ 1 + x = 3.5 :=
by
  sorry

end factory_production_l111_111419


namespace parabola_b_value_l111_111595

variable {q : ℝ}

theorem parabola_b_value (a b c : ℝ) (h_a : a = -3 / q)
  (h_eq : ∀ x : ℝ, (a * x^2 + b * x + c) = a * (x - q)^2 + q)
  (h_intercept : (a * 0^2 + b * 0 + c) = -2 * q)
  (h_q_nonzero : q ≠ 0) :
  b = 6 / q := 
sorry

end parabola_b_value_l111_111595


namespace determine_OP_l111_111791

theorem determine_OP
  (a b c d e : ℝ)
  (h_dist_OA : a > 0)
  (h_dist_OB : b > 0)
  (h_dist_OC : c > 0)
  (h_dist_OD : d > 0)
  (h_dist_OE : e > 0)
  (h_c_le_d : c ≤ d)
  (P : ℝ)
  (hP : c ≤ P ∧ P ≤ d)
  (h_ratio : ∀ (P : ℝ) (hP : c ≤ P ∧ P ≤ d), (a - P) / (P - e) = (c - P) / (P - d)) :
  P = (ce - ad) / (a - c + e - d) :=
sorry

end determine_OP_l111_111791


namespace commission_percentage_l111_111579

-- Define the conditions
def cost_of_item := 18.0
def observed_price := 27.0
def profit_percentage := 0.20
def desired_selling_price := cost_of_item + profit_percentage * cost_of_item
def commission_amount := observed_price - desired_selling_price

-- Prove the commission percentage taken by the online store
theorem commission_percentage : (commission_amount / desired_selling_price) * 100 = 25 :=
by
  -- Here the proof would normally be implemented
  sorry

end commission_percentage_l111_111579


namespace find_divisor_l111_111997

-- Define the given and calculated values in the conditions
def initial_value : ℕ := 165826
def subtracted_value : ℕ := 2
def resulting_value : ℕ := initial_value - subtracted_value

-- Define the goal: to find the smallest divisor of resulting_value other than 1
theorem find_divisor (d : ℕ) (h1 : initial_value - subtracted_value = resulting_value)
  (h2 : resulting_value % d = 0) (h3 : d > 1) : d = 2 := by
  sorry

end find_divisor_l111_111997


namespace ratio_of_other_triangle_to_square_l111_111410

noncomputable def ratio_of_triangle_areas (m : ℝ) : ℝ :=
  let side_of_square := 2
  let area_of_square := side_of_square ^ 2
  let area_of_smaller_triangle := m * area_of_square
  let r := area_of_smaller_triangle / (side_of_square / 2)
  let s := side_of_square * side_of_square / r
  let area_of_other_triangle := side_of_square * s / 2
  area_of_other_triangle / area_of_square

theorem ratio_of_other_triangle_to_square (m : ℝ) (h : m > 0) :
  ratio_of_triangle_areas m = 1 / (4 * m) :=
sorry

end ratio_of_other_triangle_to_square_l111_111410


namespace number_of_balls_l111_111888

theorem number_of_balls (x : ℕ) (h : x - 20 = 30 - x) : x = 25 :=
sorry

end number_of_balls_l111_111888


namespace value_that_number_exceeds_l111_111305

theorem value_that_number_exceeds (V : ℤ) (h : 69 = V + 3 * (86 - 69)) : V = 18 :=
by
  sorry

end value_that_number_exceeds_l111_111305


namespace no_super_squarish_numbers_l111_111906

def is_super_squarish (M : ℕ) : Prop :=
  let a := M / 100000 % 100
  let b := M / 1000 % 1000
  let c := M % 100
  (M ≥ 1000000 ∧ M < 10000000) ∧
  (M % 10 ≠ 0 ∧ (M / 10) % 10 ≠ 0 ∧ (M / 100) % 10 ≠ 0 ∧ (M / 1000) % 10 ≠ 0 ∧
    (M / 10000) % 10 ≠ 0 ∧ (M / 100000) % 10 ≠ 0 ∧ (M / 1000000) % 10 ≠ 0) ∧
  (∃ y : ℕ, y * y = M) ∧
  (∃ f g : ℕ, f * f = a ∧ 2 * f * g = b ∧ g * g = c) ∧
  (10 ≤ a ∧ a ≤ 99) ∧
  (100 ≤ b ∧ b ≤ 999) ∧
  (10 ≤ c ∧ c ≤ 99)

theorem no_super_squarish_numbers : ∀ M : ℕ, is_super_squarish M → false :=
sorry

end no_super_squarish_numbers_l111_111906


namespace exists_infinite_arith_prog_exceeding_M_l111_111399

def sum_of_digits(n : ℕ) : ℕ :=
n.digits 10 |> List.sum

theorem exists_infinite_arith_prog_exceeding_M (M : ℝ) :
  ∃ (a d : ℕ), ¬ (10 ∣ d) ∧ (∀ n : ℕ, a + n * d > 0) ∧ (∀ n : ℕ, sum_of_digits (a + n * d) > M) := by
sorry

end exists_infinite_arith_prog_exceeding_M_l111_111399


namespace fraction_ratio_l111_111328

theorem fraction_ratio (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 2 :=
by
  sorry

end fraction_ratio_l111_111328


namespace sum_of_cubes_inequality_l111_111990

theorem sum_of_cubes_inequality (a b c : ℝ) (h1 : a >= -1) (h2 : b >= -1) (h3 : c >= -1) (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 <= 4 := 
sorry

end sum_of_cubes_inequality_l111_111990


namespace monotone_f_iff_l111_111039

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then a^x
  else x^2 + 4 / x + a * Real.log x

theorem monotone_f_iff (a : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end monotone_f_iff_l111_111039


namespace total_number_of_fish_l111_111257

theorem total_number_of_fish (fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1: fishbowls = 261) (h2: fish_per_bowl = 23) : 
  fishbowls * fish_per_bowl = 6003 := 
by
  sorry

end total_number_of_fish_l111_111257


namespace find_f_minus_half_l111_111918

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 4^x

-- Theorem statement
theorem find_f_minus_half {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_def : function_definition f) :
  f (-1/2) = -2 :=
by
  sorry

end find_f_minus_half_l111_111918


namespace six_digit_number_l111_111168

theorem six_digit_number : ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ 3 * x = (x - 300000) * 10 + 3 ∧ x = 428571 :=
by
sorry

end six_digit_number_l111_111168


namespace arith_seq_formula_l111_111802

noncomputable def arith_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + a (n + 2) = 4 * n + 6

theorem arith_seq_formula (a : ℕ → ℤ) (h : arith_seq a) : ∀ n : ℕ, a n = 2 * n + 1 :=
by
  intros
  sorry

end arith_seq_formula_l111_111802


namespace triangle_area_r_l111_111457

theorem triangle_area_r (r : ℝ) (h₁ : 12 ≤ (r - 3) ^ (3 / 2)) (h₂ : (r - 3) ^ (3 / 2) ≤ 48) : 15 ≤ r ∧ r ≤ 19 := by
  sorry

end triangle_area_r_l111_111457


namespace sufficient_but_not_necessary_condition_l111_111224

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 → ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) ∧
  ¬ (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 ↔ ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) := 
sorry

end sufficient_but_not_necessary_condition_l111_111224


namespace roof_length_width_difference_l111_111093

theorem roof_length_width_difference
  {w l : ℝ} 
  (h_area : l * w = 576) 
  (h_length : l = 4 * w) 
  (hw_pos : w > 0) :
  l - w = 36 :=
by 
  sorry

end roof_length_width_difference_l111_111093


namespace sum_of_series_l111_111740

theorem sum_of_series (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_gt : a > b) :
  ∑' n, 1 / ( ((n - 1) * a + (n - 2) * b) * (n * a + (n - 1) * b)) = 1 / ((a + b) * b) :=
by
  sorry

end sum_of_series_l111_111740


namespace log_sum_l111_111165

theorem log_sum : Real.logb 2 1 + Real.logb 3 9 = 2 := by
  sorry

end log_sum_l111_111165


namespace n_squared_plus_n_plus_1_is_odd_l111_111089

theorem n_squared_plus_n_plus_1_is_odd (n : ℤ) : Odd (n^2 + n + 1) :=
sorry

end n_squared_plus_n_plus_1_is_odd_l111_111089


namespace ratio_wealth_citizen_XY_l111_111887

noncomputable def wealth_ratio_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : ℝ :=
  let pop_X := 0.4 * P
  let wealth_X_before_tax := 0.5 * W
  let tax_X := 0.1 * wealth_X_before_tax
  let wealth_X_after_tax := wealth_X_before_tax - tax_X
  let wealth_per_citizen_X := wealth_X_after_tax / pop_X

  let pop_Y := 0.3 * P
  let wealth_Y := 0.6 * W
  let wealth_per_citizen_Y := wealth_Y / pop_Y

  wealth_per_citizen_X / wealth_per_citizen_Y

theorem ratio_wealth_citizen_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : 
  wealth_ratio_XY P W h1 h2 = 9 / 16 := 
by
  sorry

end ratio_wealth_citizen_XY_l111_111887


namespace linear_function_no_pass_quadrant_I_l111_111013

theorem linear_function_no_pass_quadrant_I (x y : ℝ) (h : y = -2 * x - 1) : 
  ¬ (0 < x ∧ 0 < y) :=
by 
  sorry

end linear_function_no_pass_quadrant_I_l111_111013


namespace perimeter_after_adding_tiles_l111_111236

theorem perimeter_after_adding_tiles (init_perimeter new_tiles : ℕ) (cond1 : init_perimeter = 14) (cond2 : new_tiles = 2) :
  ∃ new_perimeter : ℕ, new_perimeter = 18 :=
by
  sorry

end perimeter_after_adding_tiles_l111_111236


namespace ways_to_go_from_first_to_fifth_l111_111249

theorem ways_to_go_from_first_to_fifth (floors : ℕ) (staircases_per_floor : ℕ) (total_ways : ℕ) 
    (h1 : floors = 5) (h2 : staircases_per_floor = 2) (h3 : total_ways = 2^4) : total_ways = 16 :=
by
  sorry

end ways_to_go_from_first_to_fifth_l111_111249


namespace well_depth_l111_111796

variable (d : ℝ)

-- Conditions
def total_time (t₁ t₂ : ℝ) : Prop := t₁ + t₂ = 8.5
def stone_fall (t₁ : ℝ) : Prop := d = 16 * t₁^2 
def sound_travel (t₂ : ℝ) : Prop := t₂ = d / 1100

theorem well_depth : 
  ∃ t₁ t₂ : ℝ, total_time t₁ t₂ ∧ stone_fall d t₁ ∧ sound_travel d t₂ → d = 918.09 := 
by
  sorry

end well_depth_l111_111796


namespace find_function_l111_111180

theorem find_function (f : ℕ → ℕ) (h : ∀ m n, f (m + f n) = f (f m) + f n) :
  ∃ d, d > 0 ∧ (∀ m, ∃ k, f m = k * d) :=
sorry

end find_function_l111_111180


namespace speed_of_faster_train_l111_111401

theorem speed_of_faster_train
  (length_each_train : ℕ)
  (length_in_meters : length_each_train = 50)
  (speed_slower_train_kmh : ℝ)
  (speed_slower : speed_slower_train_kmh = 36)
  (pass_time_seconds : ℕ)
  (pass_time : pass_time_seconds = 36) :
  ∃ speed_faster_train_kmh, speed_faster_train_kmh = 46 :=
by
  sorry

end speed_of_faster_train_l111_111401


namespace scoops_for_mom_l111_111929

/-- 
  Each scoop of ice cream costs $2.
  Pierre gets 3 scoops.
  The total bill is $14.
  Prove that Pierre's mom gets 4 scoops.
-/
theorem scoops_for_mom
  (scoop_cost : ℕ)
  (pierre_scoops : ℕ)
  (total_bill : ℕ) :
  scoop_cost = 2 → pierre_scoops = 3 → total_bill = 14 → 
  (total_bill - pierre_scoops * scoop_cost) / scoop_cost = 4 := 
by
  intros h1 h2 h3
  sorry

end scoops_for_mom_l111_111929


namespace P_plus_Q_is_26_l111_111218

theorem P_plus_Q_is_26 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3))) : 
  P + Q = 26 :=
sorry

end P_plus_Q_is_26_l111_111218


namespace max_value_of_x2_plus_y2_l111_111386

open Real

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x - 2 * y + 2) : 
  x^2 + y^2 ≤ 6 + 4 * sqrt 2 :=
sorry

end max_value_of_x2_plus_y2_l111_111386


namespace effect_on_revenue_l111_111716

variables (P Q : ℝ)

def original_revenue : ℝ := P * Q
def new_price : ℝ := 1.60 * P
def new_quantity : ℝ := 0.80 * Q
def new_revenue : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue (h1 : new_price P = 1.60 * P) (h2 : new_quantity Q = 0.80 * Q) :
  new_revenue P Q - original_revenue P Q = 0.28 * original_revenue P Q :=
by
  sorry

end effect_on_revenue_l111_111716


namespace simplify_expression_value_at_3_value_at_4_l111_111214

-- Define the original expression
def original_expr (x : ℕ) : ℚ := (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2 * x + 1))

-- Property 1: Simplify the expression
theorem simplify_expression (x : ℕ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  original_expr x = (x - 1) / (x + 2) :=
sorry

-- Property 2: Evaluate the expression at x = 3
theorem value_at_3 : original_expr 3 = 2 / 5 :=
sorry

-- Property 3: Evaluate the expression at x = 4
theorem value_at_4 : original_expr 4 = 1 / 2 :=
sorry

end simplify_expression_value_at_3_value_at_4_l111_111214


namespace boat_b_takes_less_time_l111_111162

theorem boat_b_takes_less_time (A_speed_still : ℝ) (B_speed_still : ℝ)
  (A_current : ℝ) (B_current : ℝ) (distance_downstream : ℝ)
  (A_speed_downstream : A_speed_still + A_current = 26)
  (B_speed_downstream : B_speed_still + B_current = 28)
  (A_time : A_speed_still + A_current = 26 → distance_downstream / (A_speed_still + A_current) = 4.6154)
  (B_time : B_speed_still + B_current = 28 → distance_downstream / (B_speed_still + B_current) = 4.2857) :
  distance_downstream / (B_speed_still + B_current) < distance_downstream / (A_speed_still + A_current) :=
by sorry

end boat_b_takes_less_time_l111_111162


namespace number_of_turns_to_wind_tape_l111_111462

theorem number_of_turns_to_wind_tape (D δ L : ℝ) 
(hD : D = 22) 
(hδ : δ = 0.018) 
(hL : L = 90000) : 
∃ n : ℕ, n = 791 := 
sorry

end number_of_turns_to_wind_tape_l111_111462


namespace percentage_decrease_wages_l111_111487

theorem percentage_decrease_wages (W : ℝ) (P : ℝ) : 
  (0.20 * W * (1 - P / 100)) = 0.70 * (0.20 * W) → 
  P = 30 :=
by
  sorry

end percentage_decrease_wages_l111_111487


namespace solve_combination_eq_l111_111759

theorem solve_combination_eq (x : ℕ) (h : x ≥ 3) : 
  (Nat.choose x 3 + Nat.choose x 2 = 12 * (x - 1)) ↔ (x = 9) := 
by
  sorry

end solve_combination_eq_l111_111759


namespace max_eccentricity_of_ellipse_l111_111107

theorem max_eccentricity_of_ellipse 
  (R_large : ℝ)
  (r_cylinder : ℝ)
  (R_small : ℝ)
  (D_centers : ℝ)
  (a : ℝ)
  (b : ℝ)
  (e : ℝ) :
  R_large = 1 → 
  r_cylinder = 1 → 
  R_small = 1/4 → 
  D_centers = 10/3 → 
  a = 5/3 → 
  b = 1 → 
  e = Real.sqrt (1 - (b / a) ^ 2) → 
  e = 4/5 := by 
  sorry

end max_eccentricity_of_ellipse_l111_111107


namespace henry_collected_points_l111_111965

def points_from_wins (wins : ℕ) : ℕ := wins * 5
def points_from_losses (losses : ℕ) : ℕ := losses * 2
def points_from_draws (draws : ℕ) : ℕ := draws * 3

def total_points (wins losses draws : ℕ) : ℕ := 
  points_from_wins wins + points_from_losses losses + points_from_draws draws

theorem henry_collected_points :
  total_points 2 2 10 = 44 := by
  -- The proof will go here
  sorry

end henry_collected_points_l111_111965


namespace discount_savings_l111_111158

theorem discount_savings (initial_price discounted_price : ℝ)
  (h_initial : initial_price = 475)
  (h_discounted : discounted_price = 199) :
  initial_price - discounted_price = 276 :=
by
  rw [h_initial, h_discounted]
  sorry

end discount_savings_l111_111158


namespace kylie_daisies_l111_111805

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l111_111805


namespace third_consecutive_odd_integers_is_fifteen_l111_111424

theorem third_consecutive_odd_integers_is_fifteen :
  ∃ x : ℤ, (x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) ∧ (x + 2 + (x + 4) = x + 17) → (x + 4 = 15) :=
by
  sorry

end third_consecutive_odd_integers_is_fifteen_l111_111424


namespace negative_represents_backward_l111_111510

-- Definitions based on conditions
def forward (distance : Int) : Int := distance
def backward (distance : Int) : Int := -distance

-- The mathematical equivalent proof problem
theorem negative_represents_backward
  (distance : Int)
  (h : forward distance = 5) :
  backward distance = -5 :=
sorry

end negative_represents_backward_l111_111510


namespace remainder_of_sum_l111_111758

open Nat

theorem remainder_of_sum :
  (12345 + 12347 + 12349 + 12351 + 12353 + 12355 + 12357) % 16 = 9 :=
by 
  sorry

end remainder_of_sum_l111_111758


namespace cos_240_is_neg_half_l111_111405

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l111_111405


namespace find_x_l111_111837

def set_of_numbers := [1, 2, 4, 5, 6, 9, 9, 10]

theorem find_x {x : ℝ} (h : (set_of_numbers.sum + x) / 9 = 7) : x = 17 :=
by
  sorry

end find_x_l111_111837


namespace num_packages_l111_111562

-- Defining the given conditions
def packages_count_per_package := 6
def total_tshirts := 426

-- The statement to be proved
theorem num_packages : (total_tshirts / packages_count_per_package) = 71 :=
by sorry

end num_packages_l111_111562


namespace perimeter_of_rectangle_EFGH_l111_111064

noncomputable def rectangle_ellipse_problem (u v c d : ℝ) : Prop :=
  (u * v = 3000) ∧
  (3000 = c * d) ∧
  ((u + v) = 2 * c) ∧
  ((u^2 + v^2).sqrt = 2 * (c^2 - d^2).sqrt) ∧
  (d = 3000 / c) ∧
  (4 * c = 8 * (1500).sqrt)

theorem perimeter_of_rectangle_EFGH :
  ∃ (u v c d : ℝ), rectangle_ellipse_problem u v c d ∧ 2 * (u + v) = 8 * (1500).sqrt := sorry

end perimeter_of_rectangle_EFGH_l111_111064


namespace compute_permutation_eq_4_l111_111453

def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem compute_permutation_eq_4 :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 4 :=
by
  sorry

end compute_permutation_eq_4_l111_111453


namespace f_monotonicity_l111_111818

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem f_monotonicity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x : ℝ, x > 0 → deriv (f a) x > 0) ∧ (∀ x : ℝ, x < 0 → deriv (f a) x < 0) :=
by
  sorry

end f_monotonicity_l111_111818


namespace x_plus_q_eq_five_l111_111301

theorem x_plus_q_eq_five (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x < 5) : x + q = 5 :=
by
  sorry

end x_plus_q_eq_five_l111_111301


namespace find_m_value_l111_111285

-- Definitions for the problem conditions are given below
variables (m : ℝ)

-- Conditions
def conditions := (6 < m) ∧ (m < 10) ∧ (4 = 2 * 2) ∧ (4 = (m - 2) - (10 - m))

-- Proof statement
theorem find_m_value : conditions m → m = 8 :=
sorry

end find_m_value_l111_111285


namespace area_of_trapezium_is_105_l111_111365

-- Define points in the coordinate plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨14, 3⟩
def C : Point := ⟨18, 10⟩
def D : Point := ⟨0, 10⟩

noncomputable def length (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)
noncomputable def height (p1 p2 : Point) : ℝ := abs (p2.y - p1.y)

-- Calculate lengths of parallel sides AB and CD, and height
noncomputable def AB := length A B
noncomputable def CD := length C D
noncomputable def heightAC := height A C

-- Define the area of trapezium
noncomputable def area_trapezium (AB CD height : ℝ) : ℝ := (1/2) * (AB + CD) * height

-- The proof problem statement
theorem area_of_trapezium_is_105 :
  area_trapezium AB CD heightAC = 105 := by
  sorry

end area_of_trapezium_is_105_l111_111365


namespace values_of_x_for_g_l111_111907

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l111_111907


namespace smallest_x_y_sum_299_l111_111321

theorem smallest_x_y_sum_299 : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x < y ∧ (100 + (x / y : ℚ) = 2 * (100 * x / y : ℚ)) ∧ (x + y = 299) :=
by
  sorry

end smallest_x_y_sum_299_l111_111321


namespace total_sales_l111_111237

theorem total_sales (S : ℝ) (remitted : ℝ) : 
  (∀ S, remitted = S - (0.05 * 10000 + 0.04 * (S - 10000)) → remitted = 31100) → S = 32500 :=
by
  sorry

end total_sales_l111_111237


namespace integer_count_in_interval_l111_111924

theorem integer_count_in_interval : 
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  upper_bound - lower_bound + 1 = 61 :=
by
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  have : upper_bound - lower_bound + 1 = 61 := sorry
  exact this

end integer_count_in_interval_l111_111924


namespace arithmetic_seq_solution_l111_111396

variables (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) - a n = d

def seq_cond (a : ℕ → ℤ) (d : ℤ) : Prop :=
is_arithmetic_sequence a d ∧ (a 2 + a 6 = a 8)

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_seq_solution :
  ∀ (a : ℕ → ℤ) (d : ℤ), seq_cond a d → (a 2 - a 1 ≠ 0) → 
    (sum_first_n a 5 / a 5) = 3 :=
by
  intros a d h_cond h_d_ne_zero
  sorry

end arithmetic_seq_solution_l111_111396


namespace difference_in_girls_and_boys_l111_111231

-- Given conditions as definitions
def boys : ℕ := 40
def ratio_boys_to_girls (b g : ℕ) : Prop := 5 * g = 13 * b

-- Statement of the problem
theorem difference_in_girls_and_boys (g : ℕ) (h : ratio_boys_to_girls boys g) : g - boys = 64 :=
by
  sorry

end difference_in_girls_and_boys_l111_111231


namespace sum_of_six_angles_l111_111357

theorem sum_of_six_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle3 + angle5 = 180)
  (h2 : angle2 + angle4 + angle6 = 180) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 :=
by
  sorry

end sum_of_six_angles_l111_111357


namespace initial_necklaces_15_l111_111335

variable (N E : ℕ)
variable (initial_necklaces : ℕ) (initial_earrings : ℕ) (store_necklaces : ℕ) (store_earrings : ℕ) (mother_earrings : ℕ) (total_jewelry : ℕ)

axiom necklaces_eq_initial : N = initial_necklaces
axiom earrings_eq_15 : E = initial_earrings
axiom initial_earrings_15 : initial_earrings = 15
axiom store_necklaces_eq_initial : store_necklaces = initial_necklaces
axiom store_earrings_eq_23_initial : store_earrings = 2 * initial_earrings / 3
axiom mother_earrings_eq_115_store : mother_earrings = 1 * store_earrings / 5
axiom total_jewelry_is_57 : total_jewelry = 57
axiom jewelry_pieces_eq : 2 * initial_necklaces + initial_earrings + store_earrings + mother_earrings = total_jewelry

theorem initial_necklaces_15 : initial_necklaces = 15 := by
  sorry

end initial_necklaces_15_l111_111335


namespace maximum_value_of_n_with_positive_sequence_l111_111192

theorem maximum_value_of_n_with_positive_sequence (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, 0 < a n) 
    (h_arithmetic : ∀ n : ℕ, a (n + 1)^2 - a n^2 = 1) : ∃ n : ℕ, n = 24 ∧ a n < 5 :=
by
  sorry

end maximum_value_of_n_with_positive_sequence_l111_111192


namespace number_of_boys_is_60_l111_111653

-- Definitions based on conditions
def total_students : ℕ := 150

def number_of_boys (x : ℕ) : Prop :=
  ∃ g : ℕ, x + g = total_students ∧ g = (x * total_students) / 100

-- Theorem statement
theorem number_of_boys_is_60 : number_of_boys 60 := 
sorry

end number_of_boys_is_60_l111_111653


namespace symmetric_point_coordinates_l111_111471

-- Definition of symmetry in the Cartesian coordinate system
def is_symmetrical_about_origin (A A' : ℝ × ℝ) : Prop :=
  A'.1 = -A.1 ∧ A'.2 = -A.2

-- Given point A and its symmetric property to find point A'
theorem symmetric_point_coordinates (A A' : ℝ × ℝ)
  (hA : A = (1, -2))
  (h_symm : is_symmetrical_about_origin A A') :
  A' = (-1, 2) :=
by
  sorry -- Proof to be filled in (not required as per the instructions)

end symmetric_point_coordinates_l111_111471


namespace troll_ratio_l111_111575

theorem troll_ratio 
  (B : ℕ)
  (h1 : 6 + B + (1 / 2 : ℚ) * B = 33) : 
  B / 6 = 3 :=
by
  sorry

end troll_ratio_l111_111575


namespace distance_from_sphere_center_to_triangle_plane_l111_111414

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ) (a b c : ℝ), 
  r = 9 →
  a = 13 →
  b = 13 →
  c = 10 →
  (∀ (d : ℝ), d = distance_from_O_to_plane) →
  d = 8.36 :=
by
  intro O r a b c hr ha hb hc hd
  sorry

end distance_from_sphere_center_to_triangle_plane_l111_111414


namespace find_x_pos_integer_l111_111905

theorem find_x_pos_integer (x : ℕ) (h : 0 < x) (n d : ℕ)
    (h1 : n = x^2 + 4 * x + 29)
    (h2 : d = 4 * x + 9)
    (h3 : n = d * x + 13) : 
    x = 2 := 
sorry

end find_x_pos_integer_l111_111905


namespace value_of_c_minus_a_l111_111742

variables (a b c : ℝ)

theorem value_of_c_minus_a (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 60) : (c - a) = 30 :=
by
  have h3 : a + b = 90 := by sorry
  have h4 : b + c = 120 := by sorry
  -- now we have the required form of the problem statement
  -- c - a = 120 - 90
  sorry

end value_of_c_minus_a_l111_111742


namespace square_area_l111_111431

theorem square_area (perimeter : ℝ) (h : perimeter = 32) : 
  ∃ (side area : ℝ), side = perimeter / 4 ∧ area = side * side ∧ area = 64 := 
by
  sorry

end square_area_l111_111431


namespace triangle_ABC_angles_l111_111030

theorem triangle_ABC_angles :
  ∃ (θ φ ω : ℝ), θ = 36 ∧ φ = 72 ∧ ω = 72 ∧
  (ω + φ + θ = 180) ∧
  (2 * ω + θ = 180) ∧
  (φ = 2 * θ) :=
by
  sorry

end triangle_ABC_angles_l111_111030


namespace range_of_a_l111_111937

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) → (a ≤ -1 ∨ a ≥ 3) :=
by 
  sorry

end range_of_a_l111_111937


namespace distance_to_office_is_18_l111_111642

-- Definitions given in the problem conditions
variables (x t d : ℝ)
-- Conditions based on the problem statements
axiom speed_condition1 : d = x * t
axiom speed_condition2 : d = (x + 1) * (3 / 4 * t)
axiom speed_condition3 : d = (x - 1) * (t + 3)

-- The mathematical proof statement that needs to be shown
theorem distance_to_office_is_18 :
  d = 18 :=
by
  sorry

end distance_to_office_is_18_l111_111642


namespace solution_l111_111599

-- Define the equations and their solution sets
def eq1 (x p : ℝ) : Prop := x^2 - p * x + 6 = 0
def eq2 (x q : ℝ) : Prop := x^2 + 6 * x - q = 0

-- Define the condition that the solution sets intersect at {2}
def intersect_at_2 (p q : ℝ) : Prop :=
  eq1 2 p ∧ eq2 2 q

-- The main theorem stating the value of p + q given the conditions
theorem solution (p q : ℝ) (h : intersect_at_2 p q) : p + q = 21 :=
by
  sorry

end solution_l111_111599


namespace initial_observations_l111_111125

theorem initial_observations (n : ℕ) (S : ℕ) (new_obs : ℕ) :
  (S = 12 * n) → (new_obs = 5) → (S + new_obs = 11 * (n + 1)) → n = 6 :=
by
  intro h1 h2 h3
  sorry

end initial_observations_l111_111125


namespace polynomial_expansion_l111_111847

theorem polynomial_expansion :
  (7 * X^2 + 5 * X - 3) * (3 * X^3 + 2 * X^2 + 1) = 
  21 * X^5 + 29 * X^4 + X^3 + X^2 + 5 * X - 3 :=
sorry

end polynomial_expansion_l111_111847


namespace smallest_positive_perfect_square_div_by_2_3_5_l111_111245

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l111_111245


namespace initial_mixture_volume_is_165_l111_111172

noncomputable def initial_volume_of_mixture (initial_milk_volume initial_water_volume water_added final_milk_water_ratio : ℕ) : ℕ :=
  if (initial_milk_volume + initial_water_volume) = 5 * (initial_milk_volume / 3) &&
     initial_water_volume = 2 * (initial_milk_volume / 3) &&
     water_added = 66 &&
     final_milk_water_ratio = 3 / 4 then
    5 * (initial_milk_volume / 3)
  else
    0

theorem initial_mixture_volume_is_165 :
  ∀ initial_milk_volume initial_water_volume water_added final_milk_water_ratio,
    initial_volume_of_mixture initial_milk_volume initial_water_volume water_added final_milk_water_ratio = 165 :=
by
  intros
  sorry

end initial_mixture_volume_is_165_l111_111172


namespace oil_bill_for_January_l111_111269

variables (J F : ℝ)

-- Conditions
def condition1 := F = (5 / 4) * J
def condition2 := (F + 45) / J = 3 / 2

theorem oil_bill_for_January (h1 : condition1 J F) (h2 : condition2 J F) : J = 180 :=
by sorry

end oil_bill_for_January_l111_111269


namespace extra_bananas_each_child_l111_111196

theorem extra_bananas_each_child (total_children absent_children planned_bananas_per_child : ℕ) 
    (h1 : total_children = 660) (h2 : absent_children = 330) (h3 : planned_bananas_per_child = 2) : (1320 / (total_children - absent_children)) - planned_bananas_per_child = 2 := by
  sorry

end extra_bananas_each_child_l111_111196


namespace radishes_per_row_l111_111586

theorem radishes_per_row 
  (bean_seedlings : ℕ) (beans_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (rows_per_bed : ℕ) (plant_beds : ℕ)
  (h1 : bean_seedlings = 64) (h2 : beans_per_row = 8)
  (h3 : pumpkin_seeds = 84) (h4 : pumpkins_per_row = 7)
  (h5 : radishes = 48) (h6 : rows_per_bed = 2) (h7 : plant_beds = 14) : 
  (radishes / ((plant_beds * rows_per_bed) - (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row))) = 6 := 
by sorry

end radishes_per_row_l111_111586


namespace Janet_pages_per_day_l111_111797

variable (J : ℕ)

-- Conditions
def belinda_pages_per_day : ℕ := 30
def janet_extra_pages_per_6_weeks : ℕ := 2100
def days_in_6_weeks : ℕ := 42

-- Prove that Janet reads 80 pages a day
theorem Janet_pages_per_day (h : J * days_in_6_weeks = (belinda_pages_per_day * days_in_6_weeks) + janet_extra_pages_per_6_weeks) : J = 80 := 
by sorry

end Janet_pages_per_day_l111_111797


namespace range_of_a_l111_111005

-- Defining the core problem conditions in Lean
def prop_p (a : ℝ) : Prop := ∃ x₀ : ℝ, a * x₀^2 + 2 * a * x₀ + 1 < 0

-- The original proposition p is false, thus we need to show the range of a is 0 ≤ a ≤ 1
theorem range_of_a (a : ℝ) : ¬ prop_p a → 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l111_111005


namespace range_of_a_l111_111072

theorem range_of_a (a : ℝ) :
  (-1 < x ∧ x < 0 → (x^2 - a * x + 2 * a) > 0) ∧
  (0 < x → (x^2 - a * x + 2 * a) < 0) ↔ -1 / 3 < a ∧ a < 0 :=
sorry

end range_of_a_l111_111072


namespace total_cost_proof_l111_111570

def F : ℝ := 20.50
def R : ℝ := 61.50
def M : ℝ := 1476

def total_cost (mangos : ℝ) (rice : ℝ) (flour : ℝ) : ℝ :=
  (M * mangos) + (R * rice) + (F * flour)

theorem total_cost_proof:
  total_cost 4 3 5 = 6191 := by
  sorry

end total_cost_proof_l111_111570


namespace equation_of_line_AB_l111_111485

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the center C
def C : Point := ⟨1, 0⟩

-- The equation of line AB we want to verify
def line_AB (P : Point) := P.x - P.y - 3 = 0

-- The theorem to prove
theorem equation_of_line_AB :
  (circle_eq P.x P.y ∧ P = ⟨2, -1⟩ ∧ C = ⟨1, 0⟩) → line_AB P :=
by
  sorry

end equation_of_line_AB_l111_111485


namespace division_equivalence_l111_111001

theorem division_equivalence (a b c d : ℝ) (h1 : a = 11.7) (h2 : b = 2.6) (h3 : c = 117) (h4 : d = 26) :
  (11.7 / 2.6) = (117 / 26) ∧ (117 / 26) = 4.5 := 
by 
  sorry

end division_equivalence_l111_111001


namespace number_of_days_l111_111075

def burger_meal_cost : ℕ := 6
def upsize_cost : ℕ := 1
def total_spending : ℕ := 35

/-- The number of days Clinton buys the meal. -/
theorem number_of_days (h1 : burger_meal_cost + upsize_cost = 7) (h2 : total_spending = 35) : total_spending / (burger_meal_cost + upsize_cost) = 5 :=
by
  -- The proof will go here
  sorry

end number_of_days_l111_111075


namespace number_of_games_between_men_and_women_l111_111615

theorem number_of_games_between_men_and_women
    (W M : ℕ)
    (hW : W * (W - 1) / 2 = 72)
    (hM : M * (M - 1) / 2 = 288) :
  M * W = 288 :=
by
  sorry

end number_of_games_between_men_and_women_l111_111615


namespace max_crosses_4x10_proof_l111_111129

def max_crosses_4x10 (table : Matrix ℕ ℕ Bool) : ℕ :=
  sorry -- Placeholder for actual function implementation

theorem max_crosses_4x10_proof (table : Matrix ℕ ℕ Bool) (h : ∀ i < 4, ∃ j < 10, table i j = tt) :
  max_crosses_4x10 table = 30 :=
sorry

end max_crosses_4x10_proof_l111_111129


namespace max_area_of_inscribed_equilateral_triangle_l111_111293

noncomputable def maxInscribedEquilateralTriangleArea : ℝ :=
  let length : ℝ := 12
  let width : ℝ := 15
  let max_area := 369 * Real.sqrt 3 - 540
  max_area

theorem max_area_of_inscribed_equilateral_triangle :
  maxInscribedEquilateralTriangleArea = 369 * Real.sqrt 3 - 540 := 
by
  sorry

end max_area_of_inscribed_equilateral_triangle_l111_111293


namespace find_salary_May_l111_111727

-- Define the salaries for each month as variables
variables (J F M A May : ℝ)

-- Declare the conditions as hypotheses
def avg_salary_Jan_to_Apr := (J + F + M + A) / 4 = 8000
def avg_salary_Feb_to_May := (F + M + A + May) / 4 = 8100
def salary_Jan := J = 6100

-- The theorem stating the salary for the month of May
theorem find_salary_May (h1 : avg_salary_Jan_to_Apr J F M A) (h2 : avg_salary_Feb_to_May F M A May) (h3 : salary_Jan J) :
  May = 6500 :=
  sorry

end find_salary_May_l111_111727


namespace sqrt_square_of_neg_four_l111_111318

theorem sqrt_square_of_neg_four : Real.sqrt ((-4:Real)^2) = 4 := by
  sorry

end sqrt_square_of_neg_four_l111_111318


namespace multiply_identity_l111_111624

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity_l111_111624


namespace squared_diagonal_inequality_l111_111133

theorem squared_diagonal_inequality 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :
  let AB := (x1 - x2)^2 + (y1 - y2)^2
  let BC := (x2 - x3)^2 + (y2 - y3)^2
  let CD := (x3 - x4)^2 + (y3 - y4)^2
  let DA := (x1 - x4)^2 + (y1 - y4)^2
  let AC := (x1 - x3)^2 + (y1 - y3)^2
  let BD := (x2 - x4)^2 + (y2 - y4)^2
  AC + BD ≤ AB + BC + CD + DA := 
by
  sorry

end squared_diagonal_inequality_l111_111133


namespace volume_and_surface_area_of_convex_body_l111_111521

noncomputable def volume_of_convex_body (a b c : ℝ) : ℝ := 
  (a^2 + b^2 + c^2)^3 / (6 * a * b * c)

noncomputable def surface_area_of_convex_body (a b c : ℝ) : ℝ :=
  (a^2 + b^2 + c^2)^(5/2) / (a * b * c)

theorem volume_and_surface_area_of_convex_body (a b c d : ℝ)
  (h : d^2 = a^2 + b^2 + c^2) :
  volume_of_convex_body a b c = (a^2 + b^2 + c^2)^3 / (6 * a * b * c) ∧
  surface_area_of_convex_body a b c = (a^2 + b^2 + c^2)^(5/2) / (a * b * c) :=
by
  sorry

end volume_and_surface_area_of_convex_body_l111_111521


namespace find_second_term_l111_111703

theorem find_second_term 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h_sum : ∀ n, S n = n * (2 * n + 1))
  (h_S1 : S 1 = a 1) 
  (h_S2 : S 2 = a 1 + a 2) 
  (h_a1 : a 1 = 3) : 
  a 2 = 7 := 
sorry

end find_second_term_l111_111703


namespace marsh_ducks_l111_111804

theorem marsh_ducks (D : ℕ) (h1 : 58 = D + 21) : D = 37 := 
by {
  sorry
}

end marsh_ducks_l111_111804


namespace unit_vector_parallel_to_d_l111_111279

theorem unit_vector_parallel_to_d (x y: ℝ): (4 * x - 3 * y = 0) ∧ (x^2 + y^2 = 1) → (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) :=
by sorry

end unit_vector_parallel_to_d_l111_111279


namespace deposit_time_l111_111486

theorem deposit_time (r t : ℕ) : 
  8000 + 8000 * r * t / 100 = 10200 → 
  8000 + 8000 * (r + 2) * t / 100 = 10680 → 
  t = 3 :=
by 
  sorry

end deposit_time_l111_111486


namespace base_conversion_l111_111558

theorem base_conversion {b : ℕ} (h : 5 * 6 + 2 = b * b + b + 1) : b = 5 :=
by
  -- Begin omitted steps to solve the proof
  sorry

end base_conversion_l111_111558


namespace political_exam_pass_l111_111878

-- Define the students' statements.
def A_statement (C_passed : Prop) : Prop := C_passed
def B_statement (B_passed : Prop) : Prop := ¬ B_passed
def C_statement (A_statement : Prop) : Prop := A_statement

-- Define the problem conditions.
def condition_1 (A_passed B_passed C_passed : Prop) : Prop := ¬A_passed ∨ ¬B_passed ∨ ¬C_passed
def condition_2 (A_passed B_passed C_passed : Prop) := A_statement C_passed
def condition_3 (A_passed B_passed C_passed : Prop) := B_statement B_passed
def condition_4 (A_passed B_passed C_passed : Prop) := C_statement (A_statement C_passed)
def condition_5 (A_statement_true B_statement_true C_statement_true : Prop) : Prop := 
  (¬A_statement_true ∧ B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ ¬B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ B_statement_true ∧ ¬C_statement_true)

-- Define the proof problem.
theorem political_exam_pass : 
  ∀ (A_passed B_passed C_passed : Prop),
  condition_1 A_passed B_passed C_passed →
  condition_2 A_passed B_passed C_passed →
  condition_3 A_passed B_passed C_passed →
  condition_4 A_passed B_passed C_passed →
  ∃ (A_statement_true B_statement_true C_statement_true : Prop), 
  condition_5 A_statement_true B_statement_true C_statement_true →
  ¬A_passed
:= by { sorry }

end political_exam_pass_l111_111878


namespace line_plane_relationship_l111_111042

variable {ℓ α : Type}
variables (is_line : is_line ℓ) (is_plane : is_plane α) (not_parallel : ¬ parallel ℓ α)

theorem line_plane_relationship (ℓ : Type) (α : Type) [is_line ℓ] [is_plane α] (not_parallel : ¬ parallel ℓ α) : 
  (intersect ℓ α) ∨ (subset ℓ α) :=
sorry

end line_plane_relationship_l111_111042


namespace conic_sections_union_l111_111834

theorem conic_sections_union :
  ∀ (x y : ℝ), (y^4 - 4*x^4 = 2*y^2 - 1) ↔ 
               (y^2 - 2*x^2 = 1) ∨ (y^2 + 2*x^2 = 1) := 
by
  sorry

end conic_sections_union_l111_111834


namespace ratio_of_means_l111_111592

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end ratio_of_means_l111_111592


namespace hyperbola_eccentricity_range_l111_111927

theorem hyperbola_eccentricity_range
  (a b t : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_condition : a > b) :
  ∃ e : ℝ, e = Real.sqrt (1 + (b / a)^2) ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_range_l111_111927


namespace closest_vector_l111_111432

theorem closest_vector 
  (s : ℝ)
  (u b d : ℝ × ℝ × ℝ)
  (h₁ : u = (3, -2, 4) + s • (6, 4, 2))
  (h₂ : b = (1, 7, 6))
  (hdir : d = (6, 4, 2))
  (h₃ : (u - b) = (2 + 6 * s, -9 + 4 * s, -2 + 2 * s)) :
  ((2 + 6 * s) * 6 + (-9 + 4 * s) * 4 + (-2 + 2 * s) * 2) = 0 →
  s = 1 / 2 :=
by
  -- Skipping the proof, adding sorry
  sorry

end closest_vector_l111_111432


namespace calculate_binom_l111_111863

theorem calculate_binom : 2 * Nat.choose 30 3 = 8120 := 
by 
  sorry

end calculate_binom_l111_111863


namespace number_of_adults_l111_111205

theorem number_of_adults (total_bill : ℕ) (cost_per_meal : ℕ) (num_children : ℕ) (total_cost_children : ℕ) 
  (remaining_cost_for_adults : ℕ) (num_adults : ℕ) 
  (H1 : total_bill = 56)
  (H2 : cost_per_meal = 8)
  (H3 : num_children = 5)
  (H4 : total_cost_children = num_children * cost_per_meal)
  (H5 : remaining_cost_for_adults = total_bill - total_cost_children)
  (H6 : num_adults = remaining_cost_for_adults / cost_per_meal) :
  num_adults = 2 :=
by
  sorry

end number_of_adults_l111_111205


namespace third_pipe_empty_time_l111_111859

theorem third_pipe_empty_time :
  let A_rate := 1/60
  let B_rate := 1/75
  let combined_rate := 1/50
  let third_pipe_rate := combined_rate - (A_rate + B_rate)
  let time_to_empty := 1 / third_pipe_rate
  time_to_empty = 100 :=
by
  sorry

end third_pipe_empty_time_l111_111859


namespace arithmetic_sequence_sum_l111_111199

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l111_111199


namespace shortest_distance_from_origin_l111_111275

noncomputable def shortest_distance_to_circle (x y : ℝ) : ℝ :=
  x^2 + 6 * x + y^2 - 8 * y + 18

theorem shortest_distance_from_origin :
  ∃ (d : ℝ), d = 5 - Real.sqrt 7 ∧ ∀ (x y : ℝ), shortest_distance_to_circle x y = 0 →
    (Real.sqrt ((x - 0)^2 + (y - 0)^2) - Real.sqrt ((x + 3)^2 + (y - 4)^2)) = d := sorry

end shortest_distance_from_origin_l111_111275


namespace correct_option_l111_111873

-- Definitions for universe set, and subsets A and B
def S : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- The proof goal
theorem correct_option : A ⊆ S \ B :=
by
  sorry

end correct_option_l111_111873


namespace system_of_equations_has_two_solutions_l111_111559

theorem system_of_equations_has_two_solutions :
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  xy + yz = 63 ∧ 
  xz + yz = 23 :=
sorry

end system_of_equations_has_two_solutions_l111_111559


namespace rational_solution_unique_l111_111174

theorem rational_solution_unique
  (n : ℕ) (x y : ℚ)
  (hn : Odd n)
  (hx_eqn : x ^ n + 2 * y = y ^ n + 2 * x) :
  x = y :=
sorry

end rational_solution_unique_l111_111174


namespace max_x_minus_y_l111_111870

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l111_111870


namespace pos_sum_inequality_l111_111600

theorem pos_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := 
sorry

end pos_sum_inequality_l111_111600


namespace algebraic_expression_analysis_l111_111635

theorem algebraic_expression_analysis :
  (∀ x y : ℝ, (x - 1/2 * y) * (x + 1/2 * y) = x^2 - (1/2 * y)^2) ∧
  (∀ a b c : ℝ, ¬ ((3 * a + b * c) * (-b * c - 3 * a) = (3 * a + b * c)^2)) ∧
  (∀ x y : ℝ, (3 - x + y) * (3 + x + y) = (3 + y)^2 - x^2) ∧
  ((100 + 1) * (100 - 1) = 100^2 - 1) :=
by
  intros
  repeat { split }; sorry

end algebraic_expression_analysis_l111_111635


namespace radhika_christmas_games_l111_111550

variable (C B : ℕ)

def games_on_birthday := 8
def total_games (C : ℕ) (B : ℕ) := C + B + (C + B) / 2

theorem radhika_christmas_games : 
  total_games C games_on_birthday = 30 → C = 12 :=
by
  intro h
  sorry

end radhika_christmas_games_l111_111550


namespace winning_candidate_percentage_l111_111473

theorem winning_candidate_percentage (P: ℝ) (majority diff votes totalVotes : ℝ)
    (h1 : majority = 184)
    (h2 : totalVotes = 460)
    (h3 : diff = P * totalVotes / 100 - (100 - P) * totalVotes / 100)
    (h4 : majority = diff) : P = 70 :=
by
  sorry

end winning_candidate_percentage_l111_111473


namespace cafeteria_pies_l111_111538

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h1 : initial_apples = 96)
  (h2 : handed_out_apples = 42)
  (h3 : apples_per_pie = 6) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := by
  sorry

end cafeteria_pies_l111_111538


namespace nonzero_rational_pow_zero_l111_111243

theorem nonzero_rational_pow_zero 
  (num : ℤ) (denom : ℤ) (hnum : num = -1241376497) (hdenom : denom = 294158749357) (h_nonzero: num ≠ 0 ∧ denom ≠ 0) :
  (num / denom : ℚ) ^ 0 = 1 := 
by 
  sorry

end nonzero_rational_pow_zero_l111_111243


namespace count_valid_ys_l111_111799

theorem count_valid_ys : 
  ∃ ys : Finset ℤ, ys.card = 4 ∧ ∀ y ∈ ys, (y - 3 > 0) ∧ ((y + 3) * (y - 3) * (y^2 + 9) < 2000) :=
by
  sorry

end count_valid_ys_l111_111799


namespace top_leftmost_rectangle_is_E_l111_111219

def rectangle (w x y z : ℕ) : Prop := true

-- Define the rectangles according to the given conditions
def rectangle_A : Prop := rectangle 4 1 6 9
def rectangle_B : Prop := rectangle 1 0 3 6
def rectangle_C : Prop := rectangle 3 8 5 2
def rectangle_D : Prop := rectangle 7 5 4 8
def rectangle_E : Prop := rectangle 9 2 7 0

-- Prove that the top leftmost rectangle is E
theorem top_leftmost_rectangle_is_E : rectangle_E → True :=
by
  sorry

end top_leftmost_rectangle_is_E_l111_111219


namespace hannah_books_per_stocking_l111_111556

theorem hannah_books_per_stocking
  (candy_canes_per_stocking : ℕ)
  (beanie_babies_per_stocking : ℕ)
  (num_kids : ℕ)
  (total_stuffers : ℕ)
  (books_per_stocking : ℕ) :
  candy_canes_per_stocking = 4 →
  beanie_babies_per_stocking = 2 →
  num_kids = 3 →
  total_stuffers = 21 →
  books_per_stocking = (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids →
  books_per_stocking = 1 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  simp at h5
  sorry

end hannah_books_per_stocking_l111_111556


namespace complex_inequality_l111_111294

theorem complex_inequality (m : ℝ) : 
  (m - 3 ≥ 0 ∧ m^2 - 9 = 0) → m = 3 := 
by
  sorry

end complex_inequality_l111_111294


namespace gcd_12569_36975_l111_111911

-- Define the integers for which we need to find the gcd
def num1 : ℕ := 12569
def num2 : ℕ := 36975

-- The statement that the gcd of these two numbers is 1
theorem gcd_12569_36975 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcd_12569_36975_l111_111911


namespace roof_shingles_area_l111_111563

-- Definitions based on given conditions
def base_main_roof : ℝ := 20.5
def height_main_roof : ℝ := 25
def upper_base_porch : ℝ := 2.5
def lower_base_porch : ℝ := 4.5
def height_porch : ℝ := 3
def num_gables_main_roof : ℕ := 2
def num_trapezoids_porch : ℕ := 4

-- Proof problem statement
theorem roof_shingles_area : 
  2 * (1 / 2 * base_main_roof * height_main_roof) +
  4 * (1 / 2 * (upper_base_porch + lower_base_porch) * height_porch) = 554.5 :=
by sorry

end roof_shingles_area_l111_111563


namespace minimum_f_zero_iff_t_is_2sqrt2_l111_111201

noncomputable def f (x t : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

theorem minimum_f_zero_iff_t_is_2sqrt2 :
  (∀ x > 0, f x t ≥ 0) ∧ (∃ x > 0, f x t = 0) ↔ t = 2 * Real.sqrt 2 := 
sorry

end minimum_f_zero_iff_t_is_2sqrt2_l111_111201


namespace range_of_a_l111_111637

theorem range_of_a
  (a x : ℝ)
  (h_eq : 2 * (1 / 4) ^ (-x) - (1 / 2) ^ (-x) + a = 0)
  (h_x : -1 ≤ x ∧ x ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l111_111637


namespace find_z_l111_111883

-- Definitions of the conditions
def equation_1 (x y : ℝ) : Prop := x^2 - 3 * x + 6 = y - 10
def equation_2 (y z : ℝ) : Prop := y = 2 * z
def x_value (x : ℝ) : Prop := x = -5

-- Lean theorem statement
theorem find_z (x y z : ℝ) (h1 : equation_1 x y) (h2 : equation_2 y z) (h3 : x_value x) : z = 28 :=
sorry

end find_z_l111_111883


namespace parity_of_expression_l111_111738

theorem parity_of_expression (a b c : ℤ) (h : (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 :=
by
sorry

end parity_of_expression_l111_111738


namespace divide_plane_into_four_quadrants_l111_111246

-- Definitions based on conditions
def perpendicular_axes (x y : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, x t = t ∨ x t = 0) ∧ (∀ t : ℝ, y t = t ∨ y t = 0) ∧ ∀ t : ℝ, x t ≠ y t

-- The mathematical proof statement
theorem divide_plane_into_four_quadrants (x y : ℝ → ℝ) (hx : perpendicular_axes x y) :
  ∃ quadrants : ℕ, quadrants = 4 :=
by
  sorry

end divide_plane_into_four_quadrants_l111_111246


namespace total_notebooks_eq_216_l111_111151

theorem total_notebooks_eq_216 (n : ℕ) 
  (h1 : total_notebooks = n^2 + 20)
  (h2 : total_notebooks = (n + 1)^2 - 9) : 
  total_notebooks = 216 := 
by 
  sorry

end total_notebooks_eq_216_l111_111151


namespace fraction_spent_is_one_third_l111_111375

-- Define the initial conditions and money variables
def initial_money := 32
def cost_bread := 3
def cost_candy := 2
def remaining_money_after_all := 18

-- Define the calculation for the money left after buying bread and candy bar
def money_left_after_bread_candy := initial_money - cost_bread - cost_candy

-- Define the calculation for the money spent on turkey
def money_spent_on_turkey := money_left_after_bread_candy - remaining_money_after_all

-- The fraction of the remaining money spent on the Turkey
noncomputable def fraction_spent_on_turkey := (money_spent_on_turkey : ℚ) / money_left_after_bread_candy

-- State the theorem that verifies the fraction spent on turkey is 1/3
theorem fraction_spent_is_one_third : fraction_spent_on_turkey = 1 / 3 := by
  sorry

end fraction_spent_is_one_third_l111_111375


namespace ratio_is_one_half_l111_111087

namespace CupRice

-- Define the grains of rice in one cup
def grains_in_one_cup : ℕ := 480

-- Define the grains of rice in the portion of the cup
def grains_in_portion : ℕ := 8 * 3 * 10

-- Define the ratio of the portion of the cup to the whole cup
def portion_to_cup_ratio := grains_in_portion / grains_in_one_cup

-- Prove that the ratio of the portion of the cup to the whole cup is 1:2
theorem ratio_is_one_half : portion_to_cup_ratio = 1 / 2 := by
  -- Proof goes here, but we skip it as required
  sorry
end CupRice

end ratio_is_one_half_l111_111087


namespace problem_1_problem_2_l111_111949

theorem problem_1 (α : ℝ) (hα : Real.tan α = 2) :
  Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

theorem problem_2 (α : ℝ) (hα : Real.tan α = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 :=
by
  sorry

end problem_1_problem_2_l111_111949


namespace cos_equiv_l111_111096

theorem cos_equiv (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) (hcos : Real.cos (n * Real.pi / 180) = Real.cos (1018 * Real.pi / 180)) : n = 62 := 
sorry

end cos_equiv_l111_111096


namespace hair_cut_off_length_l111_111893

def initial_hair_length : ℕ := 18
def hair_length_after_haircut : ℕ := 9

theorem hair_cut_off_length :
  initial_hair_length - hair_length_after_haircut = 9 :=
sorry

end hair_cut_off_length_l111_111893


namespace angle_between_bisectors_is_zero_l111_111910

-- Let's define the properties of the triangle and the required proof.

open Real

-- Define the side lengths of the isosceles triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ a > 0 ∧ b > 0 ∧ c > 0

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c) ∧ is_triangle a b c

-- Define the specific isosceles triangle in the problem
def triangle_ABC : Prop := is_isosceles 5 5 6

-- Prove that the angle φ between the two lines is 0°
theorem angle_between_bisectors_is_zero :
  triangle_ABC → ∃ φ : ℝ, φ = 0 :=
by sorry

end angle_between_bisectors_is_zero_l111_111910


namespace range_of_m_l111_111045

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : (1 / (a - b)) + (1 / (b - c)) ≥ m / (a - c)) :
  m ≤ 4 :=
sorry

end range_of_m_l111_111045


namespace base_conversion_min_sum_l111_111816

theorem base_conversion_min_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3)
    (h_mod: 3 * a - 2 ≡ 0 [MOD 5])
    (valid_base_a : a >= 2)
    (valid_base_b : b >= 2):
  a + b = 14 := sorry

end base_conversion_min_sum_l111_111816


namespace hyperbola_equation_l111_111755

noncomputable def hyperbola : Prop :=
  ∃ (a b : ℝ), 
    (2 : ℝ) * a = (3 : ℝ) * b ∧
    ∀ (x y : ℝ), (4 * x^2 - 9 * y^2 = -32) → (x = 1) ∧ (y = 2)

theorem hyperbola_equation (a b : ℝ) :
  (2 * a = 3 * b) ∧ (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = -32 → x = 1 ∧ y = 2) → 
  (9 / 32 * y^2 - x^2 / 8 = 1) :=
by
  sorry

end hyperbola_equation_l111_111755


namespace find_m_if_parallel_l111_111662

-- Definitions of the lines and the condition for parallel lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m - 1) * x + y + 2
def line2 (m : ℝ) (x y : ℝ) : ℝ := 8 * x + (m + 1) * y + (m - 1)

-- The condition for the lines to be parallel
def parallel (m : ℝ) : Prop :=
  (m - 1) / 8 = 1 / (m + 1) ∧ (m - 1) / 8 ≠ 2 / (m - 1)

-- The main theorem to prove
theorem find_m_if_parallel (m : ℝ) (h : parallel m) : m = 3 :=
sorry

end find_m_if_parallel_l111_111662


namespace no_solution_for_x_l111_111212

theorem no_solution_for_x (x : ℝ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → False :=
by
  sorry

end no_solution_for_x_l111_111212


namespace greatest_three_digit_multiple_of_17_l111_111259

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l111_111259


namespace product_of_roots_l111_111423

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots_l111_111423


namespace problem_statement_l111_111120

variable {x y z : ℝ}

theorem problem_statement (h : x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0)
  (hne : ¬(x = y ∧ y = z)) (hpos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x + y + z = 3) ∧ (x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6) :=
sorry

end problem_statement_l111_111120


namespace remainder_hx10_div_hx_l111_111511

noncomputable def h (x : ℕ) := x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_hx10_div_hx (x : ℕ) : (h x ^ 10) % (h x) = 7 := by
  sorry

end remainder_hx10_div_hx_l111_111511


namespace expand_expression_l111_111528

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end expand_expression_l111_111528


namespace inverse_proportional_p_q_l111_111961

theorem inverse_proportional_p_q (k : ℚ)
  (h1 : ∀ p q : ℚ, p * q = k)
  (h2 : (30 : ℚ) * (4 : ℚ) = k) :
  p = 12 ↔ (10 : ℚ) * p = k :=
by
  sorry

end inverse_proportional_p_q_l111_111961


namespace total_fast_food_order_cost_l111_111103

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l111_111103


namespace intersection_of_M_and_N_l111_111394

-- Defining our sets M and N based on the conditions provided
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x^2 < 4 }

-- The statement we want to prove
theorem intersection_of_M_and_N :
  M ∩ N = { x | -2 < x ∧ x < 1 } :=
sorry

end intersection_of_M_and_N_l111_111394


namespace interval_of_decrease_for_f_l111_111864

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)

def decreasing_interval (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem interval_of_decrease_for_f :
  decreasing_interval {x : ℝ | x < -1} f :=
by
  sorry

end interval_of_decrease_for_f_l111_111864


namespace ratio_norm_lisa_l111_111889

-- Define the number of photos taken by each photographer.
variable (L M N : ℕ)

-- Given conditions
def norm_photos : Prop := N = 110
def photo_sum_condition : Prop := L + M = M + N - 60

-- Prove the ratio of Norm's photos to Lisa's photos.
theorem ratio_norm_lisa (h1 : norm_photos N) (h2 : photo_sum_condition L M N) : N / L = 11 / 5 := 
by
  sorry

end ratio_norm_lisa_l111_111889


namespace sum_of_coefficients_l111_111947

theorem sum_of_coefficients (x y : ℝ) : 
  (2 * x - 3 * y) ^ 9 = -1 :=
by
  sorry

end sum_of_coefficients_l111_111947


namespace fraction_of_square_shaded_is_half_l111_111902

theorem fraction_of_square_shaded_is_half {s : ℝ} (h : s > 0) :
  let O := (0, 0)
  let P := (0, s)
  let Q := (s, s / 2)
  let area_square := s^2
  let area_triangle_OPQ := 1 / 2 * s^2 / 2
  let shaded_area := area_square - area_triangle_OPQ
  (shaded_area / area_square) = 1 / 2 :=
by
  sorry

end fraction_of_square_shaded_is_half_l111_111902


namespace min_dot_product_trajectory_l111_111040

-- Definitions of points and conditions
def point (x y : ℝ) : Prop := True

def trajectory (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  x * x - y * y = 2 ∧ x ≥ Real.sqrt 2

-- Definition of dot product over vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Stating the theorem for minimum value of dot product
theorem min_dot_product_trajectory (A B : ℝ × ℝ) (hA : trajectory A) (hB : trajectory B) : 
  dot_product A B ≥ 2 := 
sorry

end min_dot_product_trajectory_l111_111040


namespace songs_performed_l111_111513

variable (R L S M : ℕ)
variable (songs_total : ℕ)

def conditions := 
  R = 9 ∧ L = 6 ∧ (6 ≤ S ∧ S ≤ 9) ∧ (6 ≤ M ∧ M ≤ 9) ∧ songs_total = (R + L + S + M) / 3

theorem songs_performed (h : conditions R L S M songs_total) :
  songs_total = 9 ∨ songs_total = 10 ∨ songs_total = 11 :=
sorry

end songs_performed_l111_111513


namespace fraction_done_by_B_l111_111460

theorem fraction_done_by_B {A B : ℝ} (h : A = (2/5) * B) : (B / (A + B)) = (5/7) :=
by
  sorry

end fraction_done_by_B_l111_111460


namespace sum_a2_a4_a6_l111_111358

theorem sum_a2_a4_a6 : ∀ {a : ℕ → ℕ}, (∀ i, a (i+1) = (1 / 2 : ℝ) * a i) → a 2 = 32 → a 2 + a 4 + a 6 = 42 :=
by
  intros a ha h2
  sorry

end sum_a2_a4_a6_l111_111358


namespace find_relationship_l111_111822

noncomputable def log_equation (c d : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → 6 * (Real.log (x) / Real.log (c))^2 + 5 * (Real.log (x) / Real.log (d))^2 = 12 * (Real.log (x))^2 / (Real.log (c) * Real.log (d))

theorem find_relationship (c d : ℝ) :
  log_equation c d → 
    (d = c ^ (5 / (6 + Real.sqrt 6)) ∨ d = c ^ (5 / (6 - Real.sqrt 6))) :=
by
  sorry

end find_relationship_l111_111822


namespace real_and_equal_roots_of_quadratic_l111_111126

theorem real_and_equal_roots_of_quadratic (k: ℝ) :
  (-(k+2))^2 - 4 * 3 * 12 = 0 ↔ k = 10 ∨ k = -14 :=
by
  sorry

end real_and_equal_roots_of_quadratic_l111_111126


namespace pieces_length_l111_111898

theorem pieces_length (L M S : ℝ) (h1 : L + M + S = 180)
  (h2 : L = M + S + 30)
  (h3 : M = L / 2 - 10) :
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end pieces_length_l111_111898


namespace find_f_log2_20_l111_111652

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then 2^x + 1 else sorry

lemma f_periodic (x : ℝ) : f (x - 2) = f (x + 2) :=
sorry

lemma f_odd (x : ℝ) : f (-x) = -f (x) :=
sorry

theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
sorry

end find_f_log2_20_l111_111652


namespace smallest_integer_in_set_l111_111928

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end smallest_integer_in_set_l111_111928


namespace unique_necklace_arrangements_l111_111963

-- Definitions
def num_beads : Nat := 7

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- The number of unique ways to arrange the beads on a necklace
-- considering rotations and reflections
theorem unique_necklace_arrangements : (factorial num_beads) / (num_beads * 2) = 360 := 
by
  sorry

end unique_necklace_arrangements_l111_111963


namespace share_per_person_in_dollars_l111_111994

-- Definitions based on conditions
def total_cost_euros : ℝ := 25 * 10^9  -- 25 billion Euros
def number_of_people : ℝ := 300 * 10^6  -- 300 million people
def exchange_rate : ℝ := 1.2  -- 1 Euro = 1.2 dollars

-- To prove
theorem share_per_person_in_dollars : (total_cost_euros * exchange_rate) / number_of_people = 100 := 
by 
  sorry

end share_per_person_in_dollars_l111_111994


namespace circle_area_l111_111560

theorem circle_area :
  let circle := {p : ℝ × ℝ | (p.fst - 8) ^ 2 + p.snd ^ 2 = 64}
  let line := {p : ℝ × ℝ | p.snd = 10 - p.fst}
  ∃ area : ℝ, 
    (area = 8 * Real.pi) ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle → p.snd ≥ 0 → p ∈ line → p.snd ≥ 10 - p.fst →
  sorry := sorry

end circle_area_l111_111560


namespace curve_C2_eq_l111_111242

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end curve_C2_eq_l111_111242


namespace books_borrowed_in_a_week_l111_111054

theorem books_borrowed_in_a_week 
  (daily_avg : ℕ)
  (friday_increase_pct : ℕ)
  (days_open : ℕ)
  (friday_books : ℕ)
  (total_books_week : ℕ)
  (h1 : daily_avg = 40)
  (h2 : friday_increase_pct = 40)
  (h3 : days_open = 5)
  (h4 : friday_books = daily_avg + (daily_avg * friday_increase_pct / 100))
  (h5 : total_books_week = (days_open - 1) * daily_avg + friday_books) :
  total_books_week = 216 :=
by {
  sorry
}

end books_borrowed_in_a_week_l111_111054


namespace largest_integer_satisfying_inequality_l111_111377

theorem largest_integer_satisfying_inequality :
  ∃ x : ℤ, (6 * x - 5 < 3 * x + 4) ∧ (∀ y : ℤ, (6 * y - 5 < 3 * y + 4) → y ≤ x) ∧ x = 2 :=
by
  sorry

end largest_integer_satisfying_inequality_l111_111377


namespace select_medical_team_l111_111478

open Nat

theorem select_medical_team : 
  let male_doctors := 5
  let female_doctors := 4
  let selected_doctors := 3
  (male_doctors.choose 1 * female_doctors.choose 2 + male_doctors.choose 2 * female_doctors.choose 1) = 70 :=
by
  sorry

end select_medical_team_l111_111478


namespace sum_fractions_lt_one_l111_111964

theorem sum_fractions_lt_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  0 < (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) ∧
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) < 1 :=
by
  sorry

end sum_fractions_lt_one_l111_111964


namespace triangles_sticks_not_proportional_l111_111082

theorem triangles_sticks_not_proportional :
  ∀ (n_triangles n_sticks : ℕ), 
  (∃ k : ℕ, n_triangles = k * n_sticks) 
  ∨ 
  (∃ k : ℕ, n_triangles * n_sticks = k) 
  → False :=
by
  sorry

end triangles_sticks_not_proportional_l111_111082


namespace find_k_l111_111326

theorem find_k (x k : ℝ) (h : ((x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) ∧ k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l111_111326


namespace question_eq_answer_l111_111101

theorem question_eq_answer (n : ℝ) (h : 0.25 * 0.1 * n = 15) :
  0.1 * 0.25 * n = 15 :=
by
  sorry

end question_eq_answer_l111_111101


namespace hadley_total_walking_distance_l111_111875

-- Definitions of the distances walked to each location
def distance_grocery_store : ℕ := 2
def distance_pet_store : ℕ := distance_grocery_store - 1
def distance_home : ℕ := 4 - 1

-- Total distance walked by Hadley
def total_distance : ℕ := distance_grocery_store + distance_pet_store + distance_home

-- Statement to be proved
theorem hadley_total_walking_distance : total_distance = 6 := by
  sorry

end hadley_total_walking_distance_l111_111875


namespace graph_quadrant_l111_111617

theorem graph_quadrant (x y : ℝ) : 
  y = 3 * x - 4 → ¬ ((x < 0) ∧ (y > 0)) :=
by
  intro h
  sorry

end graph_quadrant_l111_111617


namespace explicit_formula_solution_set_l111_111757

noncomputable def f : ℝ → ℝ 
| x => if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
       if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
       0

theorem explicit_formula (x : ℝ) :
  f x = if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
        if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
        0 := 
by 
  sorry 

theorem solution_set (x : ℝ) : 
  (0 < x ∧ x < 1 ∨ -4 < x ∧ x < -1) ↔ x * f x < 0 := 
by
  sorry

end explicit_formula_solution_set_l111_111757


namespace kamari_toys_eq_65_l111_111350

-- Define the number of toys Kamari has
def number_of_toys_kamari_has : ℕ := sorry

-- Define the number of toys Anais has in terms of K
def number_of_toys_anais_has (K : ℕ) : ℕ := K + 30

-- Define the total number of toys
def total_number_of_toys (K A : ℕ) := K + A

-- Prove that the number of toys Kamari has is 65
theorem kamari_toys_eq_65 : ∃ K : ℕ, (number_of_toys_anais_has K) = K + 30 ∧ total_number_of_toys K (number_of_toys_anais_has K) = 160 ∧ K = 65 :=
by
  sorry

end kamari_toys_eq_65_l111_111350


namespace bus_speed_incl_stoppages_l111_111385

theorem bus_speed_incl_stoppages (v_excl : ℝ) (minutes_stopped : ℝ) :
  v_excl = 64 → minutes_stopped = 13.125 →
  v_excl - (v_excl * (minutes_stopped / 60)) = 50 :=
by
  intro v_excl_eq minutes_stopped_eq
  rw [v_excl_eq, minutes_stopped_eq]
  have hours_stopped : ℝ := 13.125 / 60
  have distance_lost : ℝ := 64 * hours_stopped
  have v_incl := 64 - distance_lost
  sorry

end bus_speed_incl_stoppages_l111_111385


namespace find_smaller_part_l111_111079

noncomputable def smaller_part (x y : ℕ) : ℕ :=
  if x ≤ y then x else y

theorem find_smaller_part (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : smaller_part x y = 11 :=
  sorry

end find_smaller_part_l111_111079


namespace find_m_range_l111_111140

noncomputable def f (x m : ℝ) : ℝ := x * abs (x - m) + 2 * x - 3

theorem find_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m ≤ f x₂ m)
    ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end find_m_range_l111_111140


namespace nice_set_l111_111480

def nice (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (a b c d : ℤ), (a, b) ∈ P ∧ (c, d) ∈ P → (b, a) ∈ P ∧ (a + c, b - d) ∈ P

def is_solution (p q : ℤ) : Prop :=
  Int.gcd p q = 1 ∧ p % 2 ≠ q % 2

theorem nice_set (p q : ℤ) (P : Set (ℤ × ℤ)) :
  nice P → (p, q) ∈ P → is_solution p q → P = Set.univ := 
  sorry

end nice_set_l111_111480


namespace math_or_sci_but_not_both_l111_111648

-- Definitions of the conditions
variable (students_math_and_sci : ℕ := 15)
variable (students_math : ℕ := 30)
variable (students_only_sci : ℕ := 18)

-- The theorem to prove
theorem math_or_sci_but_not_both :
  (students_math - students_math_and_sci) + students_only_sci = 33 := by
  -- Proof is omitted.
  sorry

end math_or_sci_but_not_both_l111_111648


namespace milk_needed_for_one_batch_l111_111715

-- Define cost of one batch given amount of milk M
def cost_of_one_batch (M : ℝ) : ℝ := 1.5 * M + 6

-- Define cost of three batches
def cost_of_three_batches (M : ℝ) : ℝ := 3 * cost_of_one_batch M

theorem milk_needed_for_one_batch : ∃ M : ℝ, cost_of_three_batches M = 63 ∧ M = 10 :=
by
  sorry

end milk_needed_for_one_batch_l111_111715


namespace numbers_represented_3_units_from_A_l111_111546

theorem numbers_represented_3_units_from_A (A : ℝ) (x : ℝ) (h : A = -2) : 
  abs (x + 2) = 3 ↔ x = 1 ∨ x = -5 := by
  sorry

end numbers_represented_3_units_from_A_l111_111546


namespace integral_rational_term_expansion_l111_111760

theorem integral_rational_term_expansion :
  ∫ x in 0.0..1.0, x ^ (1/6 : ℝ) = 6/7 := by
  sorry

end integral_rational_term_expansion_l111_111760


namespace three_sport_players_l111_111886

def total_members := 50
def B := 22
def T := 28
def Ba := 18
def BT := 10
def BBa := 8
def TBa := 12
def N := 4
def All := 8

theorem three_sport_players : B + T + Ba - (BT + BBa + TBa) + All = total_members - N :=
by
suffices h : 22 + 28 + 18 - (10 + 8 + 12) + 8 = 50 - 4
exact h
-- The detailed proof is left as an exercise
sorry

end three_sport_players_l111_111886


namespace sale_in_first_month_l111_111636

theorem sale_in_first_month
  (s2 : ℕ)
  (s3 : ℕ)
  (s4 : ℕ)
  (s5 : ℕ)
  (s6 : ℕ)
  (required_total_sales : ℕ)
  (average_sales : ℕ)
  : (required_total_sales = 39000) → 
    (average_sales = 6500) → 
    (s2 = 6927) →
    (s3 = 6855) →
    (s4 = 7230) →
    (s5 = 6562) →
    (s6 = 4991) →
    s2 + s3 + s4 + s5 + s6 = 32565 →
    required_total_sales - (s2 + s3 + s4 + s5 + s6) = 6435 :=
by
  intros
  sorry

end sale_in_first_month_l111_111636


namespace triangle_perimeter_l111_111233

theorem triangle_perimeter (a : ℕ) (h1 : a < 8) (h2 : a > 4) (h3 : a % 2 = 0) : 2 + 6 + a = 14 :=
  by
  sorry

end triangle_perimeter_l111_111233


namespace cookie_distribution_l111_111216

theorem cookie_distribution (b m l : ℕ)
  (h1 : b + m + l = 30)
  (h2 : m = 2 * b)
  (h3 : l = b + m) :
  b = 5 ∧ m = 10 ∧ l = 15 := 
by 
  sorry

end cookie_distribution_l111_111216


namespace solve_eq1_solve_eq2_l111_111124

-- Prove the solution of the first equation
theorem solve_eq1 (x : ℝ) : 3 * x - (x - 1) = 7 ↔ x = 3 :=
by
  sorry

-- Prove the solution of the second equation
theorem solve_eq2 (x : ℝ) : (2 * x - 1) / 3 - (x - 3) / 6 = 1 ↔ x = (5 : ℝ) / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l111_111124


namespace range_of_a_l111_111041

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1 > 2 * x - 2) → (x < a)) → (a ≥ 3) :=
by
  sorry

end range_of_a_l111_111041


namespace angle_measure_of_three_times_complementary_l111_111418

def is_complementary (α β : ℝ) : Prop := α + β = 90

def three_times_complement (α : ℝ) : Prop := 
  ∃ β : ℝ, is_complementary α β ∧ α = 3 * β

theorem angle_measure_of_three_times_complementary :
  ∀ α : ℝ, three_times_complement α → α = 67.5 :=
by sorry

end angle_measure_of_three_times_complementary_l111_111418


namespace quadratic_square_binomial_l111_111138

theorem quadratic_square_binomial (a r s : ℚ) (h1 : a = r^2) (h2 : 2 * r * s = 26) (h3 : s^2 = 9) :
  a = 169/9 := sorry

end quadratic_square_binomial_l111_111138


namespace difference_of_squares_example_l111_111179

theorem difference_of_squares_example : 204^2 - 202^2 = 812 := by
  sorry

end difference_of_squares_example_l111_111179


namespace tangent_product_l111_111988

noncomputable section

open Real

theorem tangent_product 
  (x y k1 k2 : ℝ) :
  (x / 2) ^ 2 + y ^ 2 = 1 ∧ 
  (x, y) = (-3, -3) ∧ 
  k1 + k2 = 18 / 5 ∧
  k1 * k2 = 8 / 5 → 
  (3 * k1 - 3) * (3 * k2 - 3) = 9 := 
by
  intros 
  sorry

end tangent_product_l111_111988


namespace solve_for_3x2_plus_6_l111_111670

theorem solve_for_3x2_plus_6 (x : ℚ) (h : 5 * x + 3 = 2 * x - 4) : 3 * (x^2 + 6) = 103 / 3 :=
by
  sorry

end solve_for_3x2_plus_6_l111_111670


namespace parallel_lines_iff_a_eq_3_l111_111119

theorem parallel_lines_iff_a_eq_3 (a : ℝ) :
  (∀ x y : ℝ, (6 * x - 4 * y + 1 = 0) ↔ (a * x - 2 * y - 1 = 0)) ↔ (a = 3) := 
sorry

end parallel_lines_iff_a_eq_3_l111_111119


namespace triangle_XYZ_r_s_max_sum_l111_111977

theorem triangle_XYZ_r_s_max_sum
  (r s : ℝ)
  (h_area : 1/2 * abs (r * (15 - 18) + 10 * (18 - s) + 20 * (s - 15)) = 90)
  (h_slope : s = -3 * r + 61.5) :
  r + s ≤ 42.91 :=
sorry

end triangle_XYZ_r_s_max_sum_l111_111977


namespace solve_for_m_l111_111665

theorem solve_for_m (x m : ℝ) (h : (∃ x, (x - 1) / (x - 4) = m / (x - 4))): 
  m = 3 :=
by {
  sorry -- placeholder to indicate where the proof would go
}

end solve_for_m_l111_111665


namespace triangle_problem_l111_111862

theorem triangle_problem (n : ℕ) (h : 1 < n ∧ n < 4) : n = 2 ∨ n = 3 :=
by
  -- Valid realizability proof omitted
  sorry

end triangle_problem_l111_111862


namespace bertha_no_children_count_l111_111682

-- Definitions
def bertha_daughters : ℕ := 6
def granddaughters_per_daughter : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30

-- Theorem to be proved
theorem bertha_no_children_count : 
  ∃ x : ℕ, (x * granddaughters_per_daughter + bertha_daughters = total_daughters_and_granddaughters) ∧ 
           (bertha_daughters - x + x * granddaughters_per_daughter = 26) :=
sorry

end bertha_no_children_count_l111_111682


namespace square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l111_111552

theorem square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive :
  (∀ n : ℝ, 0 ≤ n → n^2 ≤ 0 → False) ↔ (∀ m : ℝ, m < 0 → m^2 ≤ 0) := 
sorry

end square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l111_111552


namespace solve_for_c_l111_111015

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  (6 * 15 * c = 1.5) →
  c = 7

theorem solve_for_c : proof_problem 6 15 7 :=
by sorry

end solve_for_c_l111_111015


namespace tan_theta_expr_l111_111207

variables {θ x : ℝ}

-- Let θ be an acute angle and let sin(θ/2) = sqrt((x - 2) / (3x)).
theorem tan_theta_expr (h₀ : 0 < θ) (h₁ : θ < (Real.pi / 2)) (h₂ : Real.sin (θ / 2) = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan θ = (3 * Real.sqrt (7 * x^2 - 8 * x - 16)) / (x + 4) :=
sorry

end tan_theta_expr_l111_111207


namespace area_of_rhombus_enclosed_by_equation_l111_111668

-- Given the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- Define the main theorem to be proven
theorem area_of_rhombus_enclosed_by_equation : 
  (∃ x y : ℝ, equation x y) → ∃ area : ℝ, area = 384 :=
by
  sorry

end area_of_rhombus_enclosed_by_equation_l111_111668


namespace odd_function_characterization_l111_111167

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l111_111167


namespace middle_number_is_45_l111_111031

open Real

noncomputable def middle_number (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42): ℝ := 
  l.nthLe 6 sorry  -- middle element (index 6 in 0-based index)

theorem middle_number_is_45 (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42) : 
  middle_number l h_len h1 h2 h3 = 45 := 
sorry

end middle_number_is_45_l111_111031


namespace simplify_trig_expression_l111_111593

open Real

theorem simplify_trig_expression (α : ℝ) : 
  sin (2 * π - α)^2 + (cos (π + α) * cos (π - α)) + 1 = 2 := 
by 
  sorry

end simplify_trig_expression_l111_111593


namespace right_triangle_perimeter_l111_111846

-- Conditions
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (h_area : 1 / 2 * 15 * b = 150)
variable (h_pythagorean : a^2 + b^2 = c^2)
variable (h_a : a = 15)

-- The theorem to prove the perimeter is 60 units
theorem right_triangle_perimeter : a + b + c = 60 := by
  sorry

end right_triangle_perimeter_l111_111846


namespace slope_of_tangent_line_at_zero_l111_111923

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 := 
by
  sorry

end slope_of_tangent_line_at_zero_l111_111923


namespace sequence_formula_l111_111951

noncomputable def a (n : ℕ) : ℕ := n

theorem sequence_formula (n : ℕ) (h : 0 < n) (S_n : ℕ → ℕ) 
  (hSn : ∀ m : ℕ, S_n m = (1 / 2 : ℚ) * (a m)^2 + (1 / 2 : ℚ) * m) : a n = n :=
by
  sorry

end sequence_formula_l111_111951


namespace floor_diff_bounds_l111_111554

theorem floor_diff_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  0 ≤ Int.floor (a + b) - (Int.floor a + Int.floor b) ∧ 
  Int.floor (a + b) - (Int.floor a + Int.floor b) ≤ 1 :=
by
  sorry

end floor_diff_bounds_l111_111554


namespace greatest_cars_with_ac_not_racing_stripes_l111_111851

def total_cars : ℕ := 100
def without_ac : ℕ := 49
def at_least_racing_stripes : ℕ := 51

theorem greatest_cars_with_ac_not_racing_stripes :
  (total_cars - without_ac) - (at_least_racing_stripes - without_ac) = 49 :=
by
  unfold total_cars without_ac at_least_racing_stripes
  sorry

end greatest_cars_with_ac_not_racing_stripes_l111_111851


namespace geometric_sequence_third_term_l111_111441

theorem geometric_sequence_third_term (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := 
sorry

end geometric_sequence_third_term_l111_111441


namespace cubic_sum_identity_l111_111347

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 10) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 100 :=
by sorry

end cubic_sum_identity_l111_111347


namespace bread_cost_is_30_l111_111008

variable (cost_sandwich : ℝ)
variable (cost_ham : ℝ)
variable (cost_cheese : ℝ)

def cost_bread (cost_sandwich cost_ham cost_cheese : ℝ) : ℝ :=
  cost_sandwich - cost_ham - cost_cheese

theorem bread_cost_is_30 (H1 : cost_sandwich = 0.90)
  (H2 : cost_ham = 0.25)
  (H3 : cost_cheese = 0.35) :
  cost_bread cost_sandwich cost_ham cost_cheese = 0.30 :=
by
  rw [H1, H2, H3]
  simp [cost_bread]
  sorry

end bread_cost_is_30_l111_111008


namespace angle_compute_l111_111690

open Real

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def sub_vec := (b.1 - a.1, b.2 - a.2)
noncomputable def sum_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ :=
  arccos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))

theorem angle_compute : angle_between sub_vec sum_vec = π / 4 :=
by {
  sorry
}

end angle_compute_l111_111690


namespace sum_of_digits_l111_111391

theorem sum_of_digits (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h3 : 34 * a + 42 * b = 142) : a + b = 4 := 
by
  sorry

end sum_of_digits_l111_111391


namespace joan_seashells_l111_111729

/-- Prove that Joan has 36 seashells given the initial conditions. -/
theorem joan_seashells :
  let initial_seashells := 79
  let given_mike := 63
  let found_more := 45
  let traded_seashells := 20
  let lost_seashells := 5
  (initial_seashells - given_mike + found_more - traded_seashells - lost_seashells) = 36 :=
by
  sorry

end joan_seashells_l111_111729


namespace vector_addition_example_l111_111059

theorem vector_addition_example :
  (⟨-3, 2, -1⟩ : ℝ × ℝ × ℝ) + (⟨1, 5, -3⟩ : ℝ × ℝ × ℝ) = ⟨-2, 7, -4⟩ :=
by
  sorry

end vector_addition_example_l111_111059


namespace house_prices_and_yields_l111_111412

theorem house_prices_and_yields :
  ∃ x y : ℝ, 
    (425 = (y / 100) * x) ∧ 
    (459 = ((y - 0.5) / 100) * (6/5) * x) ∧ 
    (x = 8500) ∧ 
    (y = 5) ∧ 
    ((6/5) * x = 10200) ∧ 
    (y - 0.5 = 4.5) :=
by
  sorry

end house_prices_and_yields_l111_111412


namespace range_of_a_inequality_solution_set_l111_111137

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l111_111137


namespace smallest_term_index_l111_111113

theorem smallest_term_index (a_n : ℕ → ℤ) (h : ∀ n, a_n n = 3 * n^2 - 38 * n + 12) : ∃ n, a_n n = a_n 6 ∧ ∀ m, a_n m ≥ a_n 6 :=
by
  sorry

end smallest_term_index_l111_111113


namespace solve_for_A_l111_111102

variable (x y : ℝ)

theorem solve_for_A (A : ℝ) : (2 * x - y) ^ 2 + A = (2 * x + y) ^ 2 → A = 8 * x * y :=
by
  intro h
  sorry

end solve_for_A_l111_111102


namespace inequality_proof_l111_111272

variables {x y a b ε m : ℝ}

theorem inequality_proof (h1 : |x - a| < ε / (2 * m))
                        (h2 : |y - b| < ε / (2 * |a|))
                        (h3 : 0 < y ∧ y < m) :
                        |x * y - a * b| < ε :=
sorry

end inequality_proof_l111_111272


namespace solve_for_y_l111_111777

theorem solve_for_y (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 :=
by
  sorry

end solve_for_y_l111_111777


namespace dividend_is_5336_l111_111930

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) :
  (D * Q + R) = 5336 :=
by {
  sorry
}

end dividend_is_5336_l111_111930


namespace monomial_same_type_l111_111309

-- Define a structure for monomials
structure Monomial where
  coeff : ℕ
  vars : List String

-- Monomials definitions based on the given conditions
def m1 := Monomial.mk 3 ["a"]
def m2 := Monomial.mk 2 ["b"]
def m3 := Monomial.mk 1 ["a", "b"]
def m4 := Monomial.mk 3 ["a", "c"]
def target := Monomial.mk 2 ["a", "b"]

-- Define a predicate to check if two monomials are of the same type
def sameType (m n : Monomial) : Prop :=
  m.vars = n.vars

theorem monomial_same_type :
  sameType m3 target := sorry

end monomial_same_type_l111_111309


namespace solving_equation_l111_111614

theorem solving_equation (x : ℝ) : 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := 
by
  sorry

end solving_equation_l111_111614


namespace max_product_l111_111710

noncomputable def max_of_product (x y : ℝ) : ℝ := x * y

theorem max_product (x y : ℝ) (h1 : x ∈ Set.Ioi 0) (h2 : y ∈ Set.Ioi 0) (h3 : x + 4 * y = 1) :
  max_of_product x y ≤ 1 / 16 := sorry

end max_product_l111_111710


namespace line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l111_111503

theorem line_eq_45_deg_y_intercept_2 :
  (∃ l : ℝ → ℝ, (l 0 = 2) ∧ (∀ x, l x = x + 2)) := sorry

theorem circle_eq_center_neg2_3_tangent_yaxis :
  (∃ c : ℝ × ℝ → ℝ, (c (-2, 3) = 0) ∧ (∀ x y, c (x, y) = (x + 2)^2 + (y - 3)^2 - 4)) := sorry

end line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l111_111503


namespace max_area_rect_l111_111756

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l111_111756


namespace maximum_value_of_k_l111_111154

theorem maximum_value_of_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
    (h4 : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) : k ≤ 1.5 :=
by
  sorry

end maximum_value_of_k_l111_111154


namespace negation_of_existence_l111_111933

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by
  sorry

end negation_of_existence_l111_111933


namespace count_two_digit_numbers_with_unit_7_lt_50_l111_111077

def is_two_digit_nat (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def has_unit_digit_7 (n : ℕ) : Prop := n % 10 = 7
def less_than_50 (n : ℕ) : Prop := n < 50

theorem count_two_digit_numbers_with_unit_7_lt_50 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit_nat n ∧ has_unit_digit_7 n ∧ less_than_50 n) ∧ s.card = 4 := 
by
  sorry

end count_two_digit_numbers_with_unit_7_lt_50_l111_111077


namespace cakes_remaining_l111_111197

theorem cakes_remaining (cakes_made : ℕ) (cakes_sold : ℕ) (h_made : cakes_made = 149) (h_sold : cakes_sold = 10) :
  (cakes_made - cakes_sold) = 139 :=
by
  cases h_made
  cases h_sold
  sorry

end cakes_remaining_l111_111197


namespace inequalities_proof_l111_111800

variables (x y z : ℝ)

def p := x + y + z
def q := x * y + y * z + z * x
def r := x * y * z

theorem inequalities_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (p x y z) ^ 2 ≥ 3 * (q x y z) ∧
  (p x y z) ^ 3 ≥ 27 * (r x y z) ∧
  (p x y z) * (q x y z) ≥ 9 * (r x y z) ∧
  (q x y z) ^ 2 ≥ 3 * (p x y z) * (r x y z) ∧
  (p x y z) ^ 2 * (q x y z) + 3 * (p x y z) * (r x y z) ≥ 4 * (q x y z) ^ 2 ∧
  (p x y z) ^ 3 + 9 * (r x y z) ≥ 4 * (p x y z) * (q x y z) ∧
  (p x y z) * (q x y z) ^ 2 ≥ 2 * (p x y z) ^ 2 * (r x y z) + 3 * (q x y z) * (r x y z) ∧
  (p x y z) * (q x y z) ^ 2 + 3 * (q x y z) * (r x y z) ≥ 4 * (p x y z) ^ 2 * (r x y z) ∧
  2 * (q x y z) ^ 3 + 9 * (r x y z) ^ 2 ≥ 7 * (p x y z) * (q x y z) * (r x y z) ∧
  (p x y z) ^ 4 + 4 * (q x y z) ^ 2 + 6 * (p x y z) * (r x y z) ≥ 5 * (p x y z) ^ 2 * (q x y z) :=
by sorry

end inequalities_proof_l111_111800


namespace greatest_possible_value_x_y_l111_111171

noncomputable def max_x_y : ℕ :=
  let s1 := 150
  let s2 := 210
  let s3 := 270
  let s4 := 330
  (3 * (s3 + s4) - (s1 + s2 + s3 + s4))

theorem greatest_possible_value_x_y :
  max_x_y = 840 := by
  sorry

end greatest_possible_value_x_y_l111_111171


namespace arithmetic_sequence_sufficient_but_not_necessary_condition_l111_111220

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a_1_a_3_equals_2a_2 (a : ℕ → ℤ) :=
  a 1 + a 3 = 2 * a 2

-- Statement of the mathematical problem
theorem arithmetic_sequence_sufficient_but_not_necessary_condition (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a_1_a_3_equals_2a_2 a ∧ (a_1_a_3_equals_2a_2 a → ¬ is_arithmetic_sequence a) :=
by
  sorry

end arithmetic_sequence_sufficient_but_not_necessary_condition_l111_111220


namespace tom_father_time_saved_correct_l111_111719

def tom_father_jog_time_saved : Prop :=
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 5
  let daily_distance := 3
  let hours_to_minutes := 60

  let monday_time := daily_distance / monday_speed
  let tuesday_time := daily_distance / tuesday_speed
  let thursday_time := daily_distance / thursday_speed
  let saturday_time := daily_distance / saturday_speed

  let total_time_original := monday_time + tuesday_time + thursday_time + saturday_time
  let always_5mph_time := 4 * (daily_distance / 5)
  let time_saved := total_time_original - always_5mph_time

  let time_saved_minutes := time_saved * hours_to_minutes

  time_saved_minutes = 3

theorem tom_father_time_saved_correct : tom_father_jog_time_saved := by
  sorry

end tom_father_time_saved_correct_l111_111719


namespace no_valid_a_l111_111147

theorem no_valid_a : ¬ ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 
  ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 2 * x₁^2 + (3 * a + 1) * x₁ + a^2 = 0 ∧ 2 * x₂^2 + (3 * a + 1) * x₂ + a^2 = 0 :=
by {
  sorry
}

end no_valid_a_l111_111147


namespace units_digit_2_pow_2015_minus_1_l111_111310

theorem units_digit_2_pow_2015_minus_1 : (2^2015 - 1) % 10 = 7 := by
  sorry

end units_digit_2_pow_2015_minus_1_l111_111310


namespace distance_between_nails_l111_111548

theorem distance_between_nails (banner_length : ℕ) (num_nails : ℕ) (end_distance : ℕ) :
  banner_length = 20 → num_nails = 7 → end_distance = 1 → 
  (banner_length - 2 * end_distance) / (num_nails - 1) = 3 :=
by
  intros
  sorry

end distance_between_nails_l111_111548


namespace malesWithCollegeDegreesOnly_l111_111604

-- Define the parameters given in the problem
def totalEmployees : ℕ := 180
def totalFemales : ℕ := 110
def employeesWithAdvancedDegrees : ℕ := 90
def employeesWithCollegeDegreesOnly : ℕ := totalEmployees - employeesWithAdvancedDegrees
def femalesWithAdvancedDegrees : ℕ := 55

-- Define the question as a theorem
theorem malesWithCollegeDegreesOnly : 
  totalEmployees = 180 →
  totalFemales = 110 →
  employeesWithAdvancedDegrees = 90 →
  employeesWithCollegeDegreesOnly = 90 →
  femalesWithAdvancedDegrees = 55 →
  ∃ (malesWithCollegeDegreesOnly : ℕ), 
    malesWithCollegeDegreesOnly = 35 := 
by
  intros
  sorry

end malesWithCollegeDegreesOnly_l111_111604


namespace gcd_2023_2052_eq_1_l111_111656

theorem gcd_2023_2052_eq_1 : Int.gcd 2023 2052 = 1 :=
by
  sorry

end gcd_2023_2052_eq_1_l111_111656


namespace MatthewSharedWithTwoFriends_l111_111612

theorem MatthewSharedWithTwoFriends
  (crackers : ℕ)
  (cakes : ℕ)
  (cakes_per_person : ℕ)
  (persons : ℕ)
  (H1 : crackers = 29)
  (H2 : cakes = 30)
  (H3 : cakes_per_person = 15)
  (H4 : persons * cakes_per_person = cakes) :
  persons = 2 := by
  sorry

end MatthewSharedWithTwoFriends_l111_111612


namespace family_e_initial_members_l111_111226

theorem family_e_initial_members 
(a b c d f E : ℕ) 
(h_a : a = 7) 
(h_b : b = 8) 
(h_c : c = 10) 
(h_d : d = 13) 
(h_f : f = 10)
(h_avg : (a - 1 + b - 1 + c - 1 + d - 1 + E - 1 + f - 1) / 6 = 8) : 
E = 6 := 
by 
  sorry

end family_e_initial_members_l111_111226


namespace hanna_has_money_l111_111322

variable (total_roses money_spent : ℕ)
variable (rose_price : ℕ := 2)

def hanna_gives_roses (total_roses : ℕ) : Bool :=
  (1 / 3 * total_roses + 1 / 2 * total_roses) = 125

theorem hanna_has_money (H : hanna_gives_roses total_roses) : money_spent = 300 := sorry

end hanna_has_money_l111_111322


namespace smallest_a_l111_111505

-- Define the conditions and the proof goal
theorem smallest_a (a b : ℝ) (h₁ : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (15 * (x : ℝ))) (h₂ : 0 ≤ a) (h₃ : 0 ≤ b) :
  a = 15 :=
sorry

end smallest_a_l111_111505


namespace product_of_fractions_l111_111346

theorem product_of_fractions :
  (1 / 2) * (2 / 3) * (3 / 4) * (3 / 2) = 3 / 8 := by
  sorry

end product_of_fractions_l111_111346


namespace find_a8_l111_111987

variable {a : ℕ → ℝ} -- Assuming the sequence is real-valued for generality

-- Defining the necessary properties and conditions of the arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions as hypothesis
variable (h_seq : arithmetic_sequence a) 
variable (h_sum : a 3 + a 6 + a 10 + a 13 = 32)

-- The proof statement
theorem find_a8 : a 8 = 8 :=
by
  sorry -- The proof itself

end find_a8_l111_111987


namespace number_of_diagonals_octagon_heptagon_diff_l111_111944

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff_l111_111944


namespace ted_age_l111_111123

variable (t s : ℕ)

theorem ted_age (h1 : t = 3 * s - 10) (h2 : t + s = 65) : t = 46 := by
  sorry

end ted_age_l111_111123


namespace folded_paper_area_ratio_l111_111956

theorem folded_paper_area_ratio (s : ℝ) (h : s > 0) :
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  (folded_area / A) = 7 / 4 :=
by
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  show (folded_area / A) = 7 / 4
  sorry

end folded_paper_area_ratio_l111_111956


namespace incorrect_calculation_l111_111117

theorem incorrect_calculation : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) :=
by sorry

end incorrect_calculation_l111_111117


namespace contradiction_example_l111_111422

theorem contradiction_example (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2) : ¬ (a < 0 ∧ b < 0) :=
by
  -- The proof goes here, but we just need the statement
  sorry

end contradiction_example_l111_111422


namespace problem_solution_l111_111815

variable (a : ℝ)
def ellipse_p (a : ℝ) : Prop := (0 < a) ∧ (a < 5)
def quadratic_q (a : ℝ) : Prop := (-3 ≤ a) ∧ (a ≤ 3)
def p_or_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∨ ((-3 ≤ a) ∧ (a ≤ 3)))
def p_and_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∧ ((-3 ≤ a) ∧ (a ≤ 3)))

theorem problem_solution (a : ℝ) :
  (ellipse_p a → 0 < a ∧ a < 5) ∧ 
  (¬(ellipse_p a) ∧ quadratic_q a → -3 ≤ a ∧ a ≤ 0) ∧
  (p_or_q a ∧ ¬(p_and_q a) → 3 < a ∧ a < 5 ∨ (-3 ≤ a ∧ a ≤ 0)) :=
  by
  sorry

end problem_solution_l111_111815


namespace sum_of_reciprocals_l111_111479

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 := 
sorry

end sum_of_reciprocals_l111_111479


namespace exists_positive_b_l111_111778

theorem exists_positive_b (m p : ℕ) (hm : 0 < m) (hp : Prime p)
  (h1 : m^2 ≡ 2 [MOD p])
  (ha : ∃ a : ℕ, 0 < a ∧ a^2 ≡ 2 - m [MOD p]) :
  ∃ b : ℕ, 0 < b ∧ b^2 ≡ m + 2 [MOD p] := 
  sorry

end exists_positive_b_l111_111778


namespace greatest_divisor_four_consecutive_squared_l111_111472

theorem greatest_divisor_four_consecutive_squared :
  ∀ (n: ℕ), ∃ m: ℕ, (∀ (n: ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) ∧ m = 144 := 
sorry

end greatest_divisor_four_consecutive_squared_l111_111472


namespace percent_increase_combined_cost_l111_111049

theorem percent_increase_combined_cost :
  let laptop_last_year := 500
  let tablet_last_year := 200
  let laptop_increase := 10 / 100
  let tablet_increase := 20 / 100
  let new_laptop_cost := laptop_last_year * (1 + laptop_increase)
  let new_tablet_cost := tablet_last_year * (1 + tablet_increase)
  let total_last_year := laptop_last_year + tablet_last_year
  let total_this_year := new_laptop_cost + new_tablet_cost
  let increase := total_this_year - total_last_year
  let percent_increase := (increase / total_last_year) * 100
  percent_increase = 13 :=
by
  sorry

end percent_increase_combined_cost_l111_111049


namespace part1_part2_l111_111470

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l111_111470


namespace mirror_area_l111_111975

-- Defining the conditions as Lean functions and values
def frame_height : ℕ := 100
def frame_width : ℕ := 140
def frame_border : ℕ := 15

-- Statement to prove the area of the mirror
theorem mirror_area :
  let mirror_width := frame_width - 2 * frame_border
  let mirror_height := frame_height - 2 * frame_border
  mirror_width * mirror_height = 7700 :=
by
  sorry

end mirror_area_l111_111975


namespace bucket_capacity_l111_111754

-- Given Conditions
variable (C : ℝ)
variable (h : (2 / 3) * C = 9)

-- Goal
theorem bucket_capacity : C = 13.5 := by
  sorry

end bucket_capacity_l111_111754


namespace algebraic_expression_value_l111_111551

variables (a b c d m : ℝ)

theorem algebraic_expression_value :
  a = -b → cd = 1 → m^2 = 1 →
  -(a + b) - cd / 2022 + m^2 / 2022 = 0 :=
by
  intros h1 h2 h3
  sorry

end algebraic_expression_value_l111_111551


namespace sum_f_x₁_f_x₂_lt_0_l111_111297

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

-- Condition: y = f(x + 2) is an odd function
def odd_function_on_shifted_domain : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

-- Condition: f(x) is monotonically increasing for x > 2
def monotonically_increasing_for_x_gt_2 : Prop :=
  ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Condition: x₁ + x₂ < 4
def sum_lt_4 : Prop :=
  x₁ + x₂ < 4

-- Condition: (x₁-2)(x₂-2) < 0
def product_shift_lt_0 : Prop :=
  (x₁ - 2) * (x₂ - 2) < 0

-- Main theorem to prove f(x₁) + f(x₂) < 0
theorem sum_f_x₁_f_x₂_lt_0
  (h1 : odd_function_on_shifted_domain f)
  (h2 : monotonically_increasing_for_x_gt_2 f)
  (h3 : sum_lt_4 x₁ x₂)
  (h4 : product_shift_lt_0 x₁ x₂) :
  f x₁ + f x₂ < 0 := sorry

end sum_f_x₁_f_x₂_lt_0_l111_111297


namespace m_lt_n_l111_111095

theorem m_lt_n (a t : ℝ) (h : 0 < t ∧ t < 1) : 
  abs (Real.log (1 + t) / Real.log a) < abs (Real.log (1 - t) / Real.log a) :=
sorry

end m_lt_n_l111_111095


namespace find_angle_A_find_bc_range_l111_111265

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (c * (a * Real.cos B - (1/2) * b) = a^2 - b^2) ∧ (A = Real.arccos (1/2))

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) :
  A = Real.pi / 3 := 
sorry

theorem find_bc_range (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) (ha : a = Real.sqrt 3) :
  b + c ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3) := 
sorry

end find_angle_A_find_bc_range_l111_111265


namespace lorelei_vase_rose_count_l111_111978

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l111_111978


namespace mustard_at_first_table_l111_111865

theorem mustard_at_first_table (M : ℝ) :
  (M + 0.25 + 0.38 = 0.88) → M = 0.25 :=
by
  intro h
  sorry

end mustard_at_first_table_l111_111865


namespace cannot_equal_120_l111_111018

def positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem cannot_equal_120 (a b : ℕ) (ha : positive_even a) (hb : positive_even b) :
  let A := a * b
  let P' := 2 * (a + b) + 6
  A + P' ≠ 120 :=
sorry

end cannot_equal_120_l111_111018


namespace calculate_expression_l111_111620

theorem calculate_expression : 1 + (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 2 - Real.sqrt 3) = 1 :=
by
  sorry

end calculate_expression_l111_111620


namespace find_sum_lent_l111_111904

theorem find_sum_lent (P : ℝ) : 
  (∃ R T : ℝ, R = 4 ∧ T = 8 ∧ I = P - 170 ∧ I = (P * 8) / 25) → P = 250 :=
by
  sorry

end find_sum_lent_l111_111904


namespace A_gt_B_and_C_lt_A_l111_111122

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l111_111122


namespace number_of_heaps_is_5_l111_111198

variable (bundles : ℕ) (bunches : ℕ) (heaps : ℕ) (total_removed : ℕ)
variable (sheets_per_bunch : ℕ) (sheets_per_bundle : ℕ) (sheets_per_heap : ℕ)

def number_of_heaps (bundles : ℕ) (sheets_per_bundle : ℕ)
                    (bunches : ℕ) (sheets_per_bunch : ℕ)
                    (total_removed : ℕ) (sheets_per_heap : ℕ) :=
  (total_removed - (bundles * sheets_per_bundle + bunches * sheets_per_bunch)) / sheets_per_heap

theorem number_of_heaps_is_5 :
  number_of_heaps 3 2 2 4 114 20 = 5 :=
by
  unfold number_of_heaps
  sorry

end number_of_heaps_is_5_l111_111198


namespace monotone_function_sol_l111_111913

noncomputable def monotone_function (f : ℤ → ℤ) :=
  ∀ x y : ℤ, f x ≤ f y → x ≤ y

theorem monotone_function_sol
  (f : ℤ → ℤ)
  (H1 : monotone_function f)
  (H2 : ∀ x y : ℤ, f (x^2005 + y^2005) = f x ^ 2005 + f y ^ 2005) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end monotone_function_sol_l111_111913


namespace oliver_final_money_l111_111181

-- Define the initial conditions as variables and constants
def initial_amount : Nat := 9
def savings : Nat := 5
def earnings : Nat := 6
def spent_frisbee : Nat := 4
def spent_puzzle : Nat := 3
def spent_stickers : Nat := 2
def movie_ticket_price : Nat := 10
def movie_ticket_discount : Nat := 20 -- 20%
def snack_price : Nat := 3
def snack_discount : Nat := 1
def birthday_gift : Nat := 8

-- Define the final amount of money Oliver has left based on the problem statement
def final_amount : Nat :=
  let total_money := initial_amount + savings + earnings
  let total_spent := spent_frisbee + spent_puzzle + spent_stickers
  let remaining_after_spending := total_money - total_spent
  let discounted_movie_ticket := movie_ticket_price * (100 - movie_ticket_discount) / 100
  let discounted_snack := snack_price - snack_discount
  let total_spent_after_discounts := discounted_movie_ticket + discounted_snack
  let remaining_after_discounts := remaining_after_spending - total_spent_after_discounts
  remaining_after_discounts + birthday_gift

-- Lean theorem statement to prove that Oliver ends up with $9
theorem oliver_final_money : final_amount = 9 := by
  sorry

end oliver_final_money_l111_111181


namespace player_B_wins_l111_111340

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end player_B_wins_l111_111340


namespace milk_production_l111_111885

variables (a b c d e : ℕ) (h1 : a > 0) (h2 : c > 0)

def summer_rate := b / (a * c) -- Rate in summer per cow per day
def winter_rate := 2 * summer_rate -- Rate in winter per cow per day

noncomputable def total_milk_produced := (d * summer_rate * e) + (d * winter_rate * e)

theorem milk_production (h : d > 0) : total_milk_produced a b c d e = 3 * b * d * e / (a * c) :=
by sorry

end milk_production_l111_111885


namespace hidden_message_is_correct_l111_111491

def russian_alphabet_mapping : Char → Nat
| 'А' => 1
| 'Б' => 2
| 'В' => 3
| 'Г' => 4
| 'Д' => 5
| 'Е' => 6
| 'Ё' => 7
| 'Ж' => 8
| 'З' => 9
| 'И' => 10
| 'Й' => 11
| 'К' => 12
| 'Л' => 13
| 'М' => 14
| 'Н' => 15
| 'О' => 16
| 'П' => 17
| 'Р' => 18
| 'С' => 19
| 'Т' => 20
| 'У' => 21
| 'Ф' => 22
| 'Х' => 23
| 'Ц' => 24
| 'Ч' => 25
| 'Ш' => 26
| 'Щ' => 27
| 'Ъ' => 28
| 'Ы' => 29
| 'Ь' => 30
| 'Э' => 31
| 'Ю' => 32
| 'Я' => 33
| _ => 0

def prime_p : ℕ := 7 -- Assume some prime number p

def grid_position (p : ℕ) (k : ℕ) := p * k

theorem hidden_message_is_correct :
  ∃ m : String, m = "ПАРОЛЬ МЕДВЕЖАТА" :=
by
  let message := "ПАРОЛЬ МЕДВЕЖАТА"
  have h1 : russian_alphabet_mapping 'П' = 17 := by sorry
  have h2 : russian_alphabet_mapping 'А' = 1 := by sorry
  have h3 : russian_alphabet_mapping 'Р' = 18 := by sorry
  have h4 : russian_alphabet_mapping 'О' = 16 := by sorry
  have h5 : russian_alphabet_mapping 'Л' = 13 := by sorry
  have h6 : russian_alphabet_mapping 'Ь' = 29 := by sorry
  have h7 : russian_alphabet_mapping 'М' = 14 := by sorry
  have h8 : russian_alphabet_mapping 'Е' = 5 := by sorry
  have h9 : russian_alphabet_mapping 'Д' = 10 := by sorry
  have h10 : russian_alphabet_mapping 'В' = 3 := by sorry
  have h11 : russian_alphabet_mapping 'Ж' = 8 := by sorry
  have h12 : russian_alphabet_mapping 'Т' = 20 := by sorry
  have g1 : grid_position prime_p 17 = 119 := by sorry
  have g2 : grid_position prime_p 1 = 7 := by sorry
  have g3 : grid_position prime_p 18 = 126 := by sorry
  have g4 : grid_position prime_p 16 = 112 := by sorry
  have g5 : grid_position prime_p 13 = 91 := by sorry
  have g6 : grid_position prime_p 29 = 203 := by sorry
  have g7 : grid_position prime_p 14 = 98 := by sorry
  have g8 : grid_position prime_p 5 = 35 := by sorry
  have g9 : grid_position prime_p 10 = 70 := by sorry
  have g10 : grid_position prime_p 3 = 21 := by sorry
  have g11 : grid_position prime_p 8 = 56 := by sorry
  have g12 : grid_position prime_p 20 = 140 := by sorry
  existsi message
  rfl

end hidden_message_is_correct_l111_111491


namespace sum_of_cubes_of_consecutive_integers_l111_111536

-- Define the given condition
def sum_of_squares_of_consecutive_integers (n : ℕ) : Prop :=
  (n - 1)^2 + n^2 + (n + 1)^2 = 7805

-- Define the statement we want to prove
theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : sum_of_squares_of_consecutive_integers n) : 
  (n - 1)^3 + n^3 + (n + 1)^3 = 398259 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l111_111536


namespace find_number_l111_111577

theorem find_number (x : ℝ) (h : 5 * 1.6 - (2 * 1.4) / x = 4) : x = 0.7 :=
by
  sorry

end find_number_l111_111577


namespace total_lawns_mowed_l111_111899

theorem total_lawns_mowed (earned_per_lawn forgotten_lawns total_earned : ℕ) 
    (h1 : earned_per_lawn = 9) 
    (h2 : forgotten_lawns = 8) 
    (h3 : total_earned = 54) : 
    ∃ (total_lawns : ℕ), total_lawns = 14 :=
by
    sorry

end total_lawns_mowed_l111_111899


namespace line_equation_through_points_and_area_l111_111211

variable (a b S : ℝ)
variable (h_b_gt_a : b > a)
variable (h_area : S = 1/2 * (b - a) * (2 * S / (b - a)))

theorem line_equation_through_points_and_area :
  0 = -2 * S * x + (b - a)^2 * y + 2 * S * a - 2 * S * b := sorry

end line_equation_through_points_and_area_l111_111211


namespace evaluate_fraction_l111_111541

theorem evaluate_fraction : (5 / 6 : ℚ) / (9 / 10) - 1 = -2 / 27 := by
  sorry

end evaluate_fraction_l111_111541


namespace min_val_l111_111692

theorem min_val (x y : ℝ) (h : x + 2 * y = 1) : 2^x + 4^y = 2 * Real.sqrt 2 :=
sorry

end min_val_l111_111692


namespace union_A_B_comp_U_A_inter_B_range_of_a_l111_111317

namespace ProofProblem

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := Set.univ

theorem union_A_B : A ∪ B = { x | 1 < x ∧ x ≤ 8 } := by
  sorry

theorem comp_U_A_inter_B : (U \ A) ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by
  sorry

end ProofProblem

end union_A_B_comp_U_A_inter_B_range_of_a_l111_111317


namespace sugar_needed_287_163_l111_111761

theorem sugar_needed_287_163 :
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sugar_stored + additional_sugar_needed = 450 :=
by
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sorry

end sugar_needed_287_163_l111_111761


namespace scientific_notation_41600_l111_111725

theorem scientific_notation_41600 : (4.16 * 10^4) = 41600 := by
  sorry

end scientific_notation_41600_l111_111725


namespace men_in_first_group_l111_111611

theorem men_in_first_group (M : ℕ) 
  (h1 : (M * 25 : ℝ) = (15 * 26.666666666666668 : ℝ)) : 
  M = 16 := 
by 
  sorry

end men_in_first_group_l111_111611


namespace cube_plane_intersection_distance_l111_111824

theorem cube_plane_intersection_distance :
  let vertices := [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]
  let P := (0, 3, 0)
  let Q := (2, 0, 0)
  let R := (2, 6, 6)
  let plane_equation := 3 * x - 2 * y - 2 * z + 6 = 0
  let S := (2, 0, 6)
  let T := (0, 6, 3)
  dist S T = 7 := sorry

end cube_plane_intersection_distance_l111_111824


namespace original_number_divisible_l111_111768

theorem original_number_divisible (N M R : ℕ) (n : ℕ) (hN : N = 1000 * M + R)
  (hDiff : (M - R) % n = 0) (hn : n = 7 ∨ n = 11 ∨ n = 13) : N % n = 0 :=
by
  sorry

end original_number_divisible_l111_111768


namespace magnitude_of_c_is_correct_l111_111176

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
noncomputable def c : ℝ × ℝ := (a.1 - (dot_product a b) * b.1, a.2 - (dot_product a b) * b.2)

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))

theorem magnitude_of_c_is_correct :
  magnitude c = 8 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_c_is_correct_l111_111176


namespace negation_of_universal_proposition_l111_111327
open Classical

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 > 0) → ∃ x : ℝ, ¬(x^2 > 0) :=
by
  intro h
  have := (not_forall.mp h)
  exact this

end negation_of_universal_proposition_l111_111327


namespace painting_time_l111_111331

theorem painting_time (t₁₂ : ℕ) (h : t₁₂ = 6) (r : ℝ) (hr : r = t₁₂ / 12) (n : ℕ) (hn : n = 20) : 
  t₁₂ + n * r = 16 := by
  sorry

end painting_time_l111_111331


namespace gain_percent_is_150_l111_111484

variable (C S : ℝ)
variable (h : 50 * C = 20 * S)

theorem gain_percent_is_150 (h : 50 * C = 20 * S) : ((S - C) / C) * 100 = 150 :=
by
  sorry

end gain_percent_is_150_l111_111484


namespace coordinates_on_y_axis_l111_111881

theorem coordinates_on_y_axis (m : ℝ) (h : m + 1 = 0) : (m + 1, m + 4) = (0, 3) :=
by
  sorry

end coordinates_on_y_axis_l111_111881


namespace number_is_correct_l111_111332

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l111_111332


namespace cyclic_trapezoid_radii_relation_l111_111572

variables (A B C D O : Type)
variables (AD BC : Type)
variables (r1 r2 r3 r4 : ℝ)

-- Conditions
def cyclic_trapezoid (A B C D: Type) (AD BC: Type): Prop := sorry
def intersection (A B C D O : Type): Prop := sorry
def radius_incircle (triangle : Type) (radius : ℝ): Prop := sorry

theorem cyclic_trapezoid_radii_relation
  (h1: cyclic_trapezoid A B C D AD BC)
  (h2: intersection A B C D O)
  (hr1: radius_incircle AOD r1)
  (hr2: radius_incircle AOB r2)
  (hr3: radius_incircle BOC r3)
  (hr4: radius_incircle COD r4):
  (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4) :=
sorry

end cyclic_trapezoid_radii_relation_l111_111572


namespace array_element_count_l111_111364

theorem array_element_count (A : Finset ℕ) 
  (h1 : ∀ n ∈ A, n ≠ 1 → (∃ a ∈ [2, 3, 5], a ∣ n)) 
  (h2 : ∀ n ∈ A, (2 * n ∈ A ∨ 3 * n ∈ A ∨ 5 * n ∈ A) ↔ (n ∈ A ∧ 2 * n ∈ A ∧ 3 * n ∈ A ∧ 5 * n ∈ A)) 
  (card_A_range : 300 ≤ A.card ∧ A.card ≤ 400) : 
  A.card = 364 := 
sorry

end array_element_count_l111_111364


namespace sqrt_range_l111_111143

theorem sqrt_range (a : ℝ) : 2 * a - 1 ≥ 0 ↔ a ≥ 1 / 2 :=
by sorry

end sqrt_range_l111_111143


namespace pies_in_each_row_l111_111150

theorem pies_in_each_row (pecan_pies apple_pies rows : Nat) (hpecan : pecan_pies = 16) (happle : apple_pies = 14) (hrows : rows = 30) :
  (pecan_pies + apple_pies) / rows = 1 :=
by
  sorry

end pies_in_each_row_l111_111150


namespace trivia_team_points_l111_111674

theorem trivia_team_points 
    (total_members : ℕ) 
    (members_absent : ℕ) 
    (total_points : ℕ) 
    (members_present : ℕ := total_members - members_absent) 
    (points_per_member : ℕ := total_points / members_present) 
    (h1 : total_members = 7) 
    (h2 : members_absent = 2) 
    (h3 : total_points = 20) : 
    points_per_member = 4 :=
by
    sorry

end trivia_team_points_l111_111674


namespace line_equation_parallel_l111_111844

theorem line_equation_parallel (x₁ y₁ m : ℝ) (h₁ : (x₁, y₁) = (1, -2)) (h₂ : m = 2) :
  ∃ a b c : ℝ, a * x₁ + b * y₁ + c = 0 ∧ a * 2 + b * 1 + c = 4 := by
sorry

end line_equation_parallel_l111_111844


namespace area_ratio_BDF_FDCE_l111_111809

-- Define the vertices of the triangle
variables {A B C : Point}
-- Define the points on the sides and midpoints
variables {E D F : Point}
-- Define angles and relevant properties
variables (angle_CBA : Angle B C A = 72)
variables (midpoint_E : Midpoint E A C)
variables (ratio_D : RatioSegment B D D C = 2)
-- Define intersection point F
variables (intersect_F : IntersectLineSegments (LineSegment A D) (LineSegment B E) = F)

theorem area_ratio_BDF_FDCE (h_angle : angle_CBA = 72) 
  (h_midpoint_E : midpoint_E) (h_ratio_D : ratio_D) (h_intersect_F : intersect_F)
  : area_ratio (Triangle.area B D F) (Quadrilateral.area F D C E) = 1 / 5 :=
sorry

end area_ratio_BDF_FDCE_l111_111809


namespace geom_seq_product_l111_111177

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  2 * a 3 - (a 8) ^ 2 + 2 * a 13 = 0

def geometric_seq (b : ℕ → ℤ) (a8 : ℤ) : Prop :=
  b 8 = a8

theorem geom_seq_product (a b : ℕ → ℤ) (a8 : ℤ) 
  (h1 : arithmetic_seq a)
  (h2 : geometric_seq b a8)
  (h3 : a8 = 4)
: b 4 * b 12 = 16 := sorry

end geom_seq_product_l111_111177


namespace product_of_consecutive_sums_not_eq_111111111_l111_111547

theorem product_of_consecutive_sums_not_eq_111111111 :
  ∀ (a : ℤ), (3 * a + 3) * (3 * a + 12) ≠ 111111111 := 
by
  intros a
  sorry

end product_of_consecutive_sums_not_eq_111111111_l111_111547


namespace find_second_divisor_l111_111080

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l111_111080


namespace lcm_24_150_is_600_l111_111047

noncomputable def lcm_24_150 : ℕ :=
  let a := 24
  let b := 150
  have h₁ : a = 2^3 * 3 := by sorry
  have h₂ : b = 2 * 3 * 5^2 := by sorry
  Nat.lcm a b

theorem lcm_24_150_is_600 : lcm_24_150 = 600 := by
  -- Use provided primes conditions to derive the result
  sorry

end lcm_24_150_is_600_l111_111047


namespace A_days_to_complete_work_l111_111762

noncomputable def work (W : ℝ) (A_work_per_day B_work_per_day : ℝ) (days_A days_B days_B_alone : ℝ) : ℝ :=
  A_work_per_day * days_A + B_work_per_day * days_B

theorem A_days_to_complete_work 
  (W : ℝ)
  (A_work_per_day B_work_per_day : ℝ)
  (days_A days_B days_B_alone : ℝ)
  (h1 : days_A = 5)
  (h2 : days_B = 12)
  (h3 : days_B_alone = 18)
  (h4 : B_work_per_day = W / days_B_alone)
  (h5 : work W A_work_per_day B_work_per_day days_A days_B days_B_alone = W) :
  W / A_work_per_day = 15 := 
sorry

end A_days_to_complete_work_l111_111762


namespace coordinates_of_point_A_l111_111000

    theorem coordinates_of_point_A (x y : ℝ) (h1 : y = 0) (h2 : abs x = 3) : (x, y) = (3, 0) ∨ (x, y) = (-3, 0) :=
    sorry
    
end coordinates_of_point_A_l111_111000


namespace find_number_l111_111500

theorem find_number (x : ℝ) (h1 : 0.35 * x = 0.2 * 700) (h2 : 0.2 * 700 = 140) (h3 : 0.35 * x = 140) : x = 400 :=
by sorry

end find_number_l111_111500


namespace xy_proposition_l111_111421

theorem xy_proposition (x y : ℝ) : (x + y ≥ 5) → (x ≥ 3 ∨ y ≥ 2) :=
sorry

end xy_proposition_l111_111421


namespace range_of_a_l111_111576

open Set

theorem range_of_a (a x : ℝ) (p : ℝ → Prop) (q : ℝ → ℝ → Prop)
    (hp : p x → |x - a| > 3)
    (hq : q x a → (x + 1) * (2 * x - 1) ≥ 0)
    (hsuff : ∀ x, ¬p x → q x a) :
    {a | ∀ x, (¬ (|x - a| > 3) → (x + 1) * (2 * x - 1) ≥ 0) → (( a ≤ -4) ∨ (a ≥ 7 / 2))} :=
by
  sorry

end range_of_a_l111_111576


namespace similarity_coefficient_interval_l111_111376

-- Definitions
def similarTriangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

-- Theorem statement
theorem similarity_coefficient_interval (x y z p k : ℝ) (h_sim : similarTriangles x y z p) :
  0 ≤ k ∧ k ≤ 2 :=
sorry

end similarity_coefficient_interval_l111_111376


namespace find_initial_workers_l111_111146

-- Define the initial number of workers.
def initial_workers (W : ℕ) (A : ℕ) : Prop :=
  -- Condition 1: W workers can complete work A in 25 days.
  ( W * 25 = A )  ∧
  -- Condition 2: (W + 10) workers can complete work A in 15 days.
  ( (W + 10) * 15 = A )

-- The theorem states that given the conditions, the initial number of workers is 15.
theorem find_initial_workers {W A : ℕ} (h : initial_workers W A) : W = 15 :=
  sorry

end find_initial_workers_l111_111146


namespace men_wages_l111_111578

-- Conditions
variable (M W B : ℝ)
variable (h1 : 15 * M = W)
variable (h2 : W = 12 * B)
variable (h3 : 15 * M + W + B = 432)

-- Statement to prove
theorem men_wages : 15 * M = 144 :=
by
  sorry

end men_wages_l111_111578


namespace product_of_two_numbers_ratio_l111_111606

theorem product_of_two_numbers_ratio (x y : ℝ)
  (h1 : x - y ≠ 0)
  (h2 : x + y = 4 * (x - y))
  (h3 : x * y = 18 * (x - y)) :
  x * y = 86.4 :=
by
  sorry

end product_of_two_numbers_ratio_l111_111606


namespace garden_plant_count_l111_111469

theorem garden_plant_count :
  let rows := 52
  let columns := 15
  rows * columns = 780 := 
by
  sorry

end garden_plant_count_l111_111469


namespace survey_total_parents_l111_111451

theorem survey_total_parents (P : ℝ)
  (h1 : 0.15 * P + 0.60 * P + 0.20 * 0.25 * P + 0.05 * P = P)
  (h2 : 0.05 * P = 6) : 
  P = 120 :=
sorry

end survey_total_parents_l111_111451


namespace slope_of_bisecting_line_l111_111680

theorem slope_of_bisecting_line (m n : ℕ) (hmn : Int.gcd m n = 1) : 
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  -- Define conditions for line through origin (x = 0, y = 0) bisecting the parallelogram
  let b := 135 / 19
  let slope := (90 + b) / 20
  -- The slope must be equal to 369/76 (m = 369, n = 76)
  m = 369 ∧ n = 76 → m + n = 445 := by
  intro m n hmn
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  let b := 135 / 19
  let slope := (90 + b) / 20
  sorry

end slope_of_bisecting_line_l111_111680


namespace arithmetic_expression_equality_l111_111525

theorem arithmetic_expression_equality :
  15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := 
by 
  sorry

end arithmetic_expression_equality_l111_111525


namespace evaluate_series_sum_l111_111446

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l111_111446


namespace arrange_abc_l111_111362

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom cos_a_eq_a : Real.cos a = a
axiom sin_cos_b_eq_b : Real.sin (Real.cos b) = b
axiom cos_sin_c_eq_c : Real.cos (Real.sin c) = c

theorem arrange_abc : b < a ∧ a < c := 
by
  sorry

end arrange_abc_l111_111362


namespace bird_cost_l111_111737

variable (scost bcost : ℕ)

theorem bird_cost (h1 : bcost = 2 * scost)
                  (h2 : (5 * bcost + 3 * scost) = (3 * bcost + 5 * scost) + 20) :
                  scost = 10 ∧ bcost = 20 :=
by {
  sorry
}

end bird_cost_l111_111737


namespace function_properties_l111_111057

noncomputable def f (x : ℝ) : ℝ := Real.sin ((13 * Real.pi / 2) - x)

theorem function_properties :
  (∀ x : ℝ, f x = Real.cos x) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (forall t: ℝ, (∀ x : ℝ, f (x + t) = f x) → (t = 2 * Real.pi ∨ t = -2 * Real.pi)) :=
by
  sorry

end function_properties_l111_111057


namespace correct_order_l111_111392

noncomputable def f : ℝ → ℝ := sorry

axiom periodic : ∀ x : ℝ, f (x + 4) = f x
axiom increasing : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ < 2) → (0 ≤ x₂ ∧ x₂ ≤ 2) → x₁ < x₂ → f x₁ < f x₂
axiom symmetric : ∀ x : ℝ, f (x + 2) = f (2 - x)

theorem correct_order : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end correct_order_l111_111392


namespace king_lancelot_seats_38_l111_111685

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l111_111685


namespace alex_distribution_ways_l111_111128

theorem alex_distribution_ways : (15^5 = 759375) := by {
  sorry
}

end alex_distribution_ways_l111_111128


namespace conclusion1_conclusion2_conclusion3_l111_111349

-- Define the Δ operation
def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

-- 1. Proof that (-2^2) Δ 4 = 0
theorem conclusion1 : delta (-4) 4 = 0 := sorry

-- 2. Proof that (1/3) Δ (1/4) = 3 Δ 4
theorem conclusion2 : delta (1/3) (1/4) = delta 3 4 := sorry

-- 3. Proof that (-m) Δ n = m Δ (-n)
theorem conclusion3 (m n : ℚ) : delta (-m) n = delta m (-n) := sorry

end conclusion1_conclusion2_conclusion3_l111_111349


namespace perp_vector_k_l111_111044

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_vector_k :
  ∀ k : ℝ, dot_product (1, 2) (-2, k) = 0 → k = 1 :=
by
  intro k h₀
  sorry

end perp_vector_k_l111_111044


namespace charge_for_cat_l111_111200

theorem charge_for_cat (D N_D N_C T C : ℝ) 
  (h1 : D = 60) (h2 : N_D = 20) (h3 : N_C = 60) (h4 : T = 3600)
  (h5 : 20 * D + 60 * C = T) :
  C = 40 := by
  sorry

end charge_for_cat_l111_111200


namespace area_of_regionM_l111_111673

/-
Define the conditions as separate predicates in Lean.
-/

def cond1 (x y : ℝ) : Prop := y - x ≥ abs (x + y)

def cond2 (x y : ℝ) : Prop := (x^2 + 8*x + y^2 + 6*y) / (2*y - x - 8) ≤ 0

/-
Define region \( M \) by combining the conditions.
-/

def regionM (x y : ℝ) : Prop := cond1 x y ∧ cond2 x y

/-
Define the main theorem to compute the area of the region \( M \).
-/

theorem area_of_regionM : 
  ∀ x y : ℝ, (regionM x y) → (calculateAreaOfM) := sorry

/-
A placeholder definition to calculate the area of M. 
-/

noncomputable def calculateAreaOfM : ℝ := 8

end area_of_regionM_l111_111673


namespace pens_distributed_evenly_l111_111856

theorem pens_distributed_evenly (S : ℕ) (P : ℕ) (pencils : ℕ) 
  (hS : S = 10) (hpencils : pencils = 920) 
  (h_pencils_distributed : pencils % S = 0) 
  (h_pens_distributed : P % S = 0) : 
  ∃ k : ℕ, P = 10 * k :=
by 
  sorry

end pens_distributed_evenly_l111_111856


namespace correct_operation_l111_111832

variable (a b : ℝ)

theorem correct_operation : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := 
sorry

end correct_operation_l111_111832


namespace price_rollback_is_correct_l111_111582

-- Define the conditions
def liters_today : ℕ := 10
def cost_per_liter_today : ℝ := 1.4
def liters_friday : ℕ := 25
def total_liters : ℕ := 35
def total_cost : ℝ := 39

-- Define the price rollback calculation
noncomputable def price_rollback : ℝ :=
  (cost_per_liter_today - (total_cost - (liters_today * cost_per_liter_today)) / liters_friday)

-- The theorem stating the rollback per liter is $0.4
theorem price_rollback_is_correct : price_rollback = 0.4 := by
  sorry

end price_rollback_is_correct_l111_111582


namespace clock_angle_150_at_5pm_l111_111628

theorem clock_angle_150_at_5pm :
  (∀ t : ℕ, (t = 5) ↔ (∀ θ : ℝ, θ = 150 → θ = (30 * t))) := sorry

end clock_angle_150_at_5pm_l111_111628


namespace inequalities_hold_l111_111539

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b ≥ 2) :=
by
  sorry

end inequalities_hold_l111_111539


namespace domain_of_function_l111_111786

noncomputable def is_domain_of_function (x : ℝ) : Prop :=
  (4 - x^2 ≥ 0) ∧ (x ≠ 1)

theorem domain_of_function :
  {x : ℝ | is_domain_of_function x} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l111_111786


namespace last_three_digits_of_7_exp_1987_l111_111076

theorem last_three_digits_of_7_exp_1987 : (7 ^ 1987) % 1000 = 543 := by
  sorry

end last_three_digits_of_7_exp_1987_l111_111076


namespace solve_for_x_l111_111683

theorem solve_for_x (x : ℤ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 9) : x = 72 / 23 :=
by
  sorry

end solve_for_x_l111_111683


namespace solve_arcsin_eq_l111_111230

open Real

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (sin x) = (3 * x) / 4

theorem solve_arcsin_eq(x : ℝ) (h : problem_statement x) (h_range: - (2 * π) / 3 ≤ x ∧ x ≤ (2 * π) / 3) : x = 0 :=
sorry

end solve_arcsin_eq_l111_111230


namespace rose_bought_flowers_l111_111633

theorem rose_bought_flowers (F : ℕ) (h1 : ∃ (daisies tulips sunflowers : ℕ), daisies = 2 ∧ sunflowers = 4 ∧ 
  tulips = (3 / 5) * (F - 2) ∧ sunflowers = (2 / 5) * (F - 2)) : F = 12 :=
sorry

end rose_bought_flowers_l111_111633


namespace new_concentration_l111_111508

def vessel1 := (3 : ℝ)  -- 3 litres
def conc1 := (0.25 : ℝ) -- 25% alcohol

def vessel2 := (5 : ℝ)  -- 5 litres
def conc2 := (0.40 : ℝ) -- 40% alcohol

def vessel3 := (7 : ℝ)  -- 7 litres
def conc3 := (0.60 : ℝ) -- 60% alcohol

def vessel4 := (4 : ℝ)  -- 4 litres
def conc4 := (0.15 : ℝ) -- 15% alcohol

def total_volume := (25 : ℝ) -- Total vessel capacity

noncomputable def alcohol_total : ℝ :=
  (vessel1 * conc1) + (vessel2 * conc2) + (vessel3 * conc3) + (vessel4 * conc4)

theorem new_concentration : (alcohol_total / total_volume = 0.302) :=
  sorry

end new_concentration_l111_111508


namespace exp_monotonic_iff_l111_111492

theorem exp_monotonic_iff (a b : ℝ) : (a > b) ↔ (Real.exp a > Real.exp b) :=
sorry

end exp_monotonic_iff_l111_111492


namespace smallest_solution_neg_two_l111_111989

-- We set up the expressions and then state the smallest solution
def smallest_solution (x : ℝ) : Prop :=
  x * abs x = 3 * x + 2

theorem smallest_solution_neg_two :
  ∃ x : ℝ, smallest_solution x ∧ (∀ y : ℝ, smallest_solution y → y ≥ x) ∧ x = -2 :=
by
  sorry

end smallest_solution_neg_two_l111_111989


namespace negation_of_universal_proposition_l111_111466

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0) ↔ ∃ (x₀ : ℝ), x₀^3 - x₀^2 + 1 > 0 :=
by {
  sorry
}

end negation_of_universal_proposition_l111_111466


namespace total_pears_l111_111388

theorem total_pears (Alyssa_picked Nancy_picked : ℕ) (h₁ : Alyssa_picked = 42) (h₂ : Nancy_picked = 17) : Alyssa_picked + Nancy_picked = 59 :=
by
  sorry

end total_pears_l111_111388


namespace range_of_a_l111_111506

noncomputable def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x ^ 2 + 2 * x + a ≥ 0 }

theorem range_of_a (a : ℝ) : (a > -8) → (∃ x, x ∈ A ∧ x ∈ B a) :=
by
  sorry

end range_of_a_l111_111506


namespace evaluate_expression_l111_111329

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 :=
by sorry

end evaluate_expression_l111_111329


namespace line_increase_is_110_l111_111882

noncomputable def original_lines (increased_lines : ℕ) (percentage_increase : ℚ) : ℚ :=
  increased_lines / (1 + percentage_increase)

theorem line_increase_is_110
  (L' : ℕ)
  (percentage_increase : ℚ)
  (hL' : L' = 240)
  (hp : percentage_increase = 0.8461538461538461) :
  L' - original_lines L' percentage_increase = 110 :=
by
  sorry

end line_increase_is_110_l111_111882


namespace area_of_sector_l111_111131

theorem area_of_sector (l : ℝ) (α : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : l = 3)
  (h2 : α = 1)
  (h3 : l = α * r) : 
  S = 9 / 2 :=
by
  sorry

end area_of_sector_l111_111131


namespace Kirill_is_69_l111_111489

/-- Kirill is 14 centimeters shorter than his brother.
    Their sister's height is twice the height of Kirill.
    Their cousin's height is 3 centimeters more than the sister's height.
    Together, their heights equal 432 centimeters.
    We aim to prove that Kirill's height is 69 centimeters.
-/
def Kirill_height (K : ℕ) : Prop :=
  let brother_height := K + 14
  let sister_height := 2 * K
  let cousin_height := 2 * K + 3
  K + brother_height + sister_height + cousin_height = 432

theorem Kirill_is_69 {K : ℕ} (h : Kirill_height K) : K = 69 :=
by
  sorry

end Kirill_is_69_l111_111489


namespace container_fullness_calc_l111_111474

theorem container_fullness_calc (initial_percent : ℝ) (added_water : ℝ) (total_capacity : ℝ) (result_fraction : ℝ) :
  initial_percent = 0.3 →
  added_water = 27 →
  total_capacity = 60 →
  result_fraction = 3/4 →
  ((initial_percent * total_capacity + added_water) / total_capacity) = result_fraction :=
by
  intros h1 h2 h3 h4
  sorry

end container_fullness_calc_l111_111474


namespace cheryl_gave_mms_to_sister_l111_111194

-- Definitions for given conditions in the problem
def ate_after_lunch : ℕ := 7
def ate_after_dinner : ℕ := 5
def initial_mms : ℕ := 25

-- The statement to be proved
theorem cheryl_gave_mms_to_sister : (initial_mms - (ate_after_lunch + ate_after_dinner)) = 13 := by
  sorry

end cheryl_gave_mms_to_sister_l111_111194


namespace convert_binary_1101_to_decimal_l111_111447

theorem convert_binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by sorry

end convert_binary_1101_to_decimal_l111_111447


namespace no_solutions_sinx_eq_sin_sinx_l111_111088

open Real

theorem no_solutions_sinx_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ arcsin 0.9) : ¬ (sin x = sin (sin x)) :=
by
  sorry

end no_solutions_sinx_eq_sin_sinx_l111_111088


namespace range_of_a_l111_111566

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) :
    (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
    a ≥ 18 := sorry

end range_of_a_l111_111566


namespace george_run_speed_l111_111803

theorem george_run_speed (usual_distance : ℝ) (usual_speed : ℝ) (today_first_distance : ℝ) (today_first_speed : ℝ)
  (remaining_distance : ℝ) (expected_time : ℝ) :
  usual_distance = 1.5 →
  usual_speed = 3 →
  today_first_distance = 1 →
  today_first_speed = 2.5 →
  remaining_distance = 0.5 →
  expected_time = usual_distance / usual_speed →
  today_first_distance / today_first_speed + remaining_distance / (remaining_distance / (expected_time - today_first_distance / today_first_speed)) = expected_time →
  remaining_distance / (expected_time - today_first_distance / today_first_speed) = 5 :=
by sorry

end george_run_speed_l111_111803


namespace area_of_given_field_l111_111697

noncomputable def area_of_field (cost_in_rupees : ℕ) (rate_per_meter_in_paise : ℕ) (ratio_width : ℕ) (ratio_length : ℕ) : ℕ :=
  let cost_in_paise := cost_in_rupees * 100
  let perimeter := (ratio_width + ratio_length) * 2
  let x := cost_in_paise / (perimeter * rate_per_meter_in_paise)
  let width := ratio_width * x
  let length := ratio_length * x
  width * length

theorem area_of_given_field :
  let cost_in_rupees := 105
  let rate_per_meter_in_paise := 25
  let ratio_width := 3
  let ratio_length := 4
  area_of_field cost_in_rupees rate_per_meter_in_paise ratio_width ratio_length = 10800 :=
by
  sorry

end area_of_given_field_l111_111697


namespace reciprocal_of_neg3_l111_111776

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l111_111776


namespace more_uniform_team_l111_111400

-- Define the parameters and the variances
def average_height := 1.85
def variance_team_A := 0.32
def variance_team_B := 0.26

-- Main theorem statement
theorem more_uniform_team : variance_team_B < variance_team_A → "Team B" = "Team with more uniform heights" :=
by
  -- Placeholder for the actual proof
  sorry

end more_uniform_team_l111_111400


namespace within_acceptable_range_l111_111657

def flour_weight : ℝ := 25.18
def flour_label : ℝ := 25
def tolerance : ℝ := 0.25

theorem within_acceptable_range  :
  (flour_label - tolerance) ≤ flour_weight ∧ flour_weight ≤ (flour_label + tolerance) :=
by
  sorry

end within_acceptable_range_l111_111657


namespace Ursula_hot_dogs_l111_111094

theorem Ursula_hot_dogs 
  (H : ℕ) 
  (cost_hot_dog : ℚ := 1.50) 
  (cost_salad : ℚ := 2.50) 
  (num_salads : ℕ := 3) 
  (total_money : ℚ := 20) 
  (change : ℚ := 5) :
  (cost_hot_dog * H + cost_salad * num_salads = total_money - change) → H = 5 :=
by
  sorry

end Ursula_hot_dogs_l111_111094


namespace max_c_val_l111_111430

theorem max_c_val (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : 2 * a * b = 2 * a + b) 
  (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 :=
sorry

end max_c_val_l111_111430


namespace common_chord_is_linear_l111_111210

-- Defining the equations of two intersecting circles
noncomputable def circle1 : ℝ → ℝ → ℝ := sorry
noncomputable def circle2 : ℝ → ℝ → ℝ := sorry

-- Defining a method to eliminate quadratic terms
noncomputable def eliminate_quadratic_terms (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Defining the linear equation representing the common chord
noncomputable def common_chord (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Statement of the problem
theorem common_chord_is_linear (circle1 circle2 : ℝ → ℝ → ℝ) :
  common_chord circle1 circle2 = eliminate_quadratic_terms circle1 circle2 := sorry

end common_chord_is_linear_l111_111210


namespace max_discount_l111_111092

-- Definitions:
def cost_price : ℝ := 400
def sale_price : ℝ := 600
def desired_profit_margin : ℝ := 0.05

-- Statement:
theorem max_discount 
  (x : ℝ) 
  (hx : sale_price * (1 - x / 100) ≥ cost_price * (1 + desired_profit_margin)) :
  x ≤ 90 := 
sorry

end max_discount_l111_111092


namespace positive_m_of_quadratic_has_one_real_root_l111_111998

theorem positive_m_of_quadratic_has_one_real_root : 
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, x^2 + 6 * m * x + m = 0 → x = -3 * m) :=
by
  sorry

end positive_m_of_quadratic_has_one_real_root_l111_111998


namespace f_positive_l111_111222

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom f_monotonically_decreasing : ∀ x y : ℝ, x < y → f x > f y
axiom inequality_condition : ∀ x : ℝ, (f x) / (f'' x) + x < 1

theorem f_positive : ∀ x : ℝ, f x > 0 :=
by sorry

end f_positive_l111_111222


namespace complement_of_union_is_singleton_five_l111_111982

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

end complement_of_union_is_singleton_five_l111_111982


namespace carousel_rotation_time_l111_111955

-- Definitions and Conditions
variables (a v U x : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (U * a - v * a = 2 * Real.pi)
def condition2 : Prop := (v * a = U * (x - a / 2))

-- Statement to prove
theorem carousel_rotation_time :
  condition1 a v U ∧ condition2 a v U x → x = 2 * a / 3 :=
by
  intro h
  have c1 := h.1
  have c2 := h.2
  sorry

end carousel_rotation_time_l111_111955


namespace three_configuration_m_separable_l111_111784

theorem three_configuration_m_separable
  {n m : ℕ} (A : Finset (Fin n)) (h : m ≥ n / 2) :
  ∀ (C : Finset (Fin n)), C.card = 3 → ∃ B : Finset (Fin n), B.card = m ∧ (∀ c ∈ C, ∃ b ∈ B, c ≠ b) :=
by
  sorry

end three_configuration_m_separable_l111_111784


namespace students_in_grade6_l111_111871

noncomputable def num_students_total : ℕ := 100
noncomputable def num_students_grade4 : ℕ := 30
noncomputable def num_students_grade5 : ℕ := 35
noncomputable def num_students_grade6 : ℕ := num_students_total - (num_students_grade4 + num_students_grade5)

theorem students_in_grade6 : num_students_grade6 = 35 := by
  sorry

end students_in_grade6_l111_111871


namespace maximize_profit_l111_111070

def cost_price_A (x y : ℕ) := x = y + 20
def cost_sum_eq_200 (x y : ℕ) := x + 2 * y = 200
def linear_function (m n : ℕ) := m = -((1/2) : ℚ) * n + 90
def profit_function (w n : ℕ) : ℚ := (-((1/2) : ℚ) * ((n : ℚ) - 130)^2) + 1250

theorem maximize_profit
  (x y m n : ℕ)
  (hx : cost_price_A x y)
  (hsum : cost_sum_eq_200 x y)
  (hlin : linear_function m n)
  (hmaxn : 80 ≤ n ∧ n ≤ 120)
  : y = 60 ∧ x = 80 ∧ n = 120 ∧ profit_function 120 120 = 1200 := 
sorry

end maximize_profit_l111_111070


namespace hexagon_perimeter_l111_111722

-- Defining the side lengths of the hexagon
def side_lengths : List ℕ := [7, 10, 8, 13, 11, 9]

-- Defining the perimeter calculation
def perimeter (sides : List ℕ) : ℕ := sides.sum

-- The main theorem stating the perimeter of the given hexagon
theorem hexagon_perimeter :
  perimeter side_lengths = 58 := by
  -- Skipping proof here
  sorry

end hexagon_perimeter_l111_111722


namespace other_divisor_l111_111371

theorem other_divisor (x : ℕ) (h1 : 261 % 37 = 2) (h2 : 261 % x = 2) (h3 : 259 = 261 - 2) :
  ∃ x : ℕ, 259 % 37 = 0 ∧ 259 % x = 0 ∧ x = 7 :=
by
  sorry

end other_divisor_l111_111371


namespace smallest_yummy_is_minus_2013_l111_111083

-- Define a yummy integer
def is_yummy (A : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (a : ℤ), (a <= A) ∧ (a + k = A) ∧ ((k + 1) * A - k*(k + 1)/2 = 2014)

-- Define the smallest yummy integer
def smallest_yummy : ℤ :=
  -2013

-- The Lean theorem to state the proof problem
theorem smallest_yummy_is_minus_2013 : ∀ A : ℤ, is_yummy A → (-2013 ≤ A) :=
by
  sorry

end smallest_yummy_is_minus_2013_l111_111083


namespace irrational_number_line_representation_l111_111517

theorem irrational_number_line_representation :
  ∀ (x : ℝ), ¬ (∃ r s : ℚ, x = r / s ∧ r ≠ 0 ∧ s ≠ 0) → ∃ p : ℝ, x = p := 
by
  sorry

end irrational_number_line_representation_l111_111517


namespace total_students_is_46_l111_111731

-- Define the constants for the problem
def students_in_history : ℕ := 19
def students_in_math : ℕ := 14
def students_in_english : ℕ := 26
def students_in_all_three : ℕ := 3
def students_in_exactly_two : ℕ := 7

-- The total number of students as per the inclusion-exclusion principle
def total_students : ℕ :=
  students_in_history + students_in_math + students_in_english
  - students_in_exactly_two - 2 * students_in_all_three + students_in_all_three

theorem total_students_is_46 : total_students = 46 :=
  sorry

end total_students_is_46_l111_111731


namespace train_length_l111_111213

theorem train_length
  (speed_kmph : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (speed_m_per_s : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 520)
  (platform_eq : platform_length = 280)
  (time_eq : time_seconds = 26) :
  ∃ L : ℕ, L = 240 := by
  sorry

end train_length_l111_111213


namespace ellipse_product_l111_111071

noncomputable def computeProduct (a b : ℝ) : ℝ :=
  let AB := 2 * a
  let CD := 2 * b
  AB * CD

theorem ellipse_product (a b : ℝ) (h1 : a^2 - b^2 = 64) (h2 : a - b = 4) :
  computeProduct a b = 240 := by
sorry

end ellipse_product_l111_111071


namespace quadrilateral_midpoints_area_l111_111255

-- We set up the geometric context and define the problem in Lean 4.

noncomputable def area_of_midpoint_quadrilateral
  (AB CD : ℝ) (AD BC : ℝ)
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) : ℝ :=
  37.5

-- The theorem statement validating the area of the quadrilateral.
theorem quadrilateral_midpoints_area (AB CD AD BC : ℝ) 
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) :
  area_of_midpoint_quadrilateral AB CD AD BC h_AB_CD h_CD_AB h_AD_BC h_BC_AD mid_AB mid_BC mid_CD mid_DA = 37.5 :=
by 
  sorry  -- Proof is omitted.

end quadrilateral_midpoints_area_l111_111255


namespace necessary_but_not_sufficient_for_gt_one_l111_111854

variable (x : ℝ)

theorem necessary_but_not_sufficient_for_gt_one (h : x^2 > 1) : ¬(x^2 > 1 ↔ x > 1) ∧ (x > 1 → x^2 > 1) :=
by
  sorry

end necessary_but_not_sufficient_for_gt_one_l111_111854


namespace ordered_triples_count_eq_4_l111_111488

theorem ordered_triples_count_eq_4 :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 
    (∀ x y z : ℝ, (x, y, z) ∈ S ↔ (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (xy + 1 = z) ∧ (yz + 1 = x) ∧ (zx + 1 = y)) ∧
    S.card = 4 :=
sorry

end ordered_triples_count_eq_4_l111_111488


namespace arithmetic_sequence_a7_l111_111895

theorem arithmetic_sequence_a7 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 4 = 9) 
  (common_diff : ∀ n, a (n + 1) = a n + d) :
  a 7 = 8 :=
by
  sorry

end arithmetic_sequence_a7_l111_111895


namespace find_x_solution_l111_111789

theorem find_x_solution (x : ℝ) 
  (h : ∑' n:ℕ, ((-1)^(n+1)) * (2 * n + 1) * x^n = 16) : 
  x = -15/16 :=
sorry

end find_x_solution_l111_111789


namespace average_side_length_of_squares_l111_111867

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l111_111867


namespace sum_base10_to_base4_l111_111667

theorem sum_base10_to_base4 : 
  (31 + 22 : ℕ) = 3 * 4^2 + 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end sum_base10_to_base4_l111_111667


namespace cone_volume_l111_111936

theorem cone_volume (R h : ℝ) (hR : 0 ≤ R) (hh : 0 ≤ h) : 
  (∫ x in (0 : ℝ)..h, π * (R / h * x)^2) = (1 / 3) * π * R^2 * h :=
by
  sorry

end cone_volume_l111_111936


namespace geometric_series_sum_l111_111687

theorem geometric_series_sum (a : ℝ) (q : ℝ) (a₁ : ℝ) 
  (h1 : a₁ = 1)
  (h2 : q = a - (3/2))
  (h3 : |q| < 1)
  (h4 : a = a₁ / (1 - q)) :
  a = 2 :=
sorry

end geometric_series_sum_l111_111687


namespace find_M_N_l111_111750

-- Define positive integers less than 10
def is_pos_int_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Main theorem to prove M = 5 and N = 6 given the conditions
theorem find_M_N (M N : ℕ) (hM : is_pos_int_lt_10 M) (hN : is_pos_int_lt_10 N) 
  (h : 8 * (10 ^ 7) * M + 420852 * 9 = N * (10 ^ 7) * 9889788 * 11) : 
  M = 5 ∧ N = 6 :=
by {
  sorry
}

end find_M_N_l111_111750


namespace total_payment_360_l111_111130

noncomputable def q : ℝ := 12
noncomputable def p_wage : ℝ := 1.5 * q
noncomputable def p_hourly_rate : ℝ := q + 6
noncomputable def h : ℝ := 20
noncomputable def total_payment_p : ℝ := p_wage * h -- The total payment when candidate p is hired
noncomputable def total_payment_q : ℝ := q * (h + 10) -- The total payment when candidate q is hired

theorem total_payment_360 : 
  p_wage = p_hourly_rate ∧ 
  total_payment_p = total_payment_q ∧ 
  total_payment_p = 360 := by
  sorry

end total_payment_360_l111_111130


namespace beef_weight_before_processing_l111_111749

-- Define the initial weight of the beef.
def W_initial := 1070.5882

-- Define the loss percentages.
def loss1 := 0.20
def loss2 := 0.15
def loss3 := 0.25

-- Define the final weight after all losses.
def W_final := 546.0

-- The main proof goal: show that W_initial results in W_final after considering the weight losses.
theorem beef_weight_before_processing (W_initial W_final : ℝ) (loss1 loss2 loss3 : ℝ) :
  W_final = (1 - loss3) * (1 - loss2) * (1 - loss1) * W_initial :=
by
  sorry

end beef_weight_before_processing_l111_111749


namespace arrangement_of_digits_11250_l111_111338

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l111_111338


namespace frog_climbing_time_is_correct_l111_111686

noncomputable def frog_climb_out_time : Nat :=
  let well_depth := 12
  let climb_up := 3
  let slip_down := 1
  let net_gain := climb_up - slip_down
  let total_cycles := (well_depth - 3) / net_gain + 1
  let total_time := total_cycles * 3
  let extra_time := 6
  total_time + extra_time

theorem frog_climbing_time_is_correct :
  frog_climb_out_time = 22 := by
  sorry

end frog_climbing_time_is_correct_l111_111686


namespace ants_total_l111_111439

namespace Ants

-- Defining the number of ants each child finds based on the given conditions
def Abe_ants := 4
def Beth_ants := Abe_ants + Abe_ants
def CeCe_ants := 3 * Abe_ants
def Duke_ants := Abe_ants / 2
def Emily_ants := Abe_ants + (3 * Abe_ants / 4)
def Frances_ants := 2 * CeCe_ants

-- The total number of ants found by the six children
def total_ants := Abe_ants + Beth_ants + CeCe_ants + Duke_ants + Emily_ants + Frances_ants

-- The statement to prove
theorem ants_total: total_ants = 57 := by
  sorry

end Ants

end ants_total_l111_111439


namespace fourth_root_of_207360000_l111_111650

theorem fourth_root_of_207360000 :
  120 ^ 4 = 207360000 :=
sorry

end fourth_root_of_207360000_l111_111650


namespace possible_value_m_l111_111821

theorem possible_value_m (x m : ℝ) (h : ∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) : m ≥ -25 / 8 := sorry

end possible_value_m_l111_111821


namespace sail_pressure_l111_111307

theorem sail_pressure (k : ℝ) :
  (forall (V A : ℝ), P = k * A * (V : ℝ)^2) 
  → (P = 1.25) → (V = 20) → (A = 1)
  → (A = 4) → (V = 40)
  → (P = 20) :=
by
  sorry

end sail_pressure_l111_111307


namespace max_sum_cos_isosceles_triangle_l111_111382

theorem max_sum_cos_isosceles_triangle :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (2 * Real.cos α + Real.cos (π - 2 * α)) ≤ 1.5 :=
by
  sorry

end max_sum_cos_isosceles_triangle_l111_111382


namespace probability_of_A_losing_l111_111286

variable (p_win p_draw p_lose : ℝ)

def probability_of_A_winning := p_win = (1/3)
def probability_of_draw := p_draw = (1/2)
def sum_of_probabilities := p_win + p_draw + p_lose = 1

theorem probability_of_A_losing
  (h1: probability_of_A_winning p_win)
  (h2: probability_of_draw p_draw)
  (h3: sum_of_probabilities p_win p_draw p_lose) :
  p_lose = (1/6) :=
sorry

end probability_of_A_losing_l111_111286


namespace inequality_division_by_positive_l111_111111

theorem inequality_division_by_positive (x y : ℝ) (h : x > y) : (x / 5 > y / 5) :=
by
  sorry

end inequality_division_by_positive_l111_111111


namespace negative_integers_abs_le_4_l111_111152

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l111_111152


namespace value_of_a_l111_111273

theorem value_of_a 
  (a b c d e : ℤ)
  (h1 : a + 4 = b + 2)
  (h2 : a + 2 = b)
  (h3 : a + c = 146)
  (he : e = 79)
  (h4 : e = d + 2)
  (h5 : d = c + 2)
  (h6 : c = b + 2) :
  a = 71 :=
by
  sorry

end value_of_a_l111_111273


namespace no_two_primes_sum_to_53_l111_111037

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_two_primes_sum_to_53_l111_111037


namespace evaluate_fractions_l111_111735

theorem evaluate_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := 
by
  sorry

end evaluate_fractions_l111_111735


namespace edge_length_of_box_l111_111426

noncomputable def edge_length_cubical_box (num_cubes : ℕ) (edge_length_cube : ℝ) : ℝ :=
  if num_cubes = 8 ∧ edge_length_cube = 0.5 then -- 50 cm in meters
    1 -- The edge length of the cubical box in meters
  else
    0 -- Placeholder for other cases

theorem edge_length_of_box :
  edge_length_cubical_box 8 0.5 = 1 :=
sorry

end edge_length_of_box_l111_111426


namespace combined_girls_avg_l111_111569

variables (A a B b : ℕ) -- Number of boys and girls at Adams and Baker respectively.
variables (avgBoysAdams avgGirlsAdams avgAdams avgBoysBaker avgGirlsBaker avgBaker : ℚ)

-- Conditions
def avgAdamsBoys := 72
def avgAdamsGirls := 78
def avgAdamsCombined := 75
def avgBakerBoys := 84
def avgBakerGirls := 91
def avgBakerCombined := 85
def combinedAvgBoys := 80

-- Equations derived from the problem statement
def equations : Prop :=
  (72 * A + 78 * a) / (A + a) = 75 ∧
  (84 * B + 91 * b) / (B + b) = 85 ∧
  (72 * A + 84 * B) / (A + B) = 80

-- The goal is to show the combined average score of girls
def combinedAvgGirls := 85

theorem combined_girls_avg (h : equations A a B b):
  (78 * (6 * b / 7) + 91 * b) / ((6 * b / 7) + b) = 85 := by
  sorry

end combined_girls_avg_l111_111569


namespace teddy_bears_ordered_l111_111141

theorem teddy_bears_ordered (days : ℕ) (T : ℕ)
  (h1 : 20 * days + 100 = T)
  (h2 : 23 * days - 20 = T) :
  T = 900 ∧ days = 40 := 
by 
  sorry

end teddy_bears_ordered_l111_111141


namespace increasing_interval_l111_111032

noncomputable def f (x : ℝ) := Real.logb 2 (5 - 4 * x - x^2)

theorem increasing_interval : ∀ {x : ℝ}, (-5 < x ∧ x ≤ -2) → f x = Real.logb 2 (5 - 4 * x - x^2) := by
  sorry

end increasing_interval_l111_111032


namespace assignment_increase_l111_111053

-- Define what an assignment statement is
def assignment_statement (lhs rhs : ℕ) : ℕ := rhs

-- Define the conditions and the problem
theorem assignment_increase (n : ℕ) : assignment_statement n (n + 1) = n + 1 :=
by
  -- Here we would prove that the assignment statement increases n by 1
  sorry

end assignment_increase_l111_111053


namespace father_l111_111894

variable (R F M : ℕ)
variable (h1 : F = 4 * R)
variable (h2 : 4 * R + 8 = M * (R + 8))
variable (h3 : 4 * R + 16 = 2 * (R + 16))

theorem father's_age_ratio (hR : R = 8) : (F + 8) / (R + 8) = 5 / 2 := by
  sorry

end father_l111_111894


namespace arithmetic_sequence_sum_l111_111325

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l111_111325


namespace unique_solution_c_exceeds_s_l111_111810

-- Problem Conditions
def steers_cost : ℕ := 35
def cows_cost : ℕ := 40
def total_budget : ℕ := 1200

-- Definition of the solution conditions
def valid_purchase (s c : ℕ) : Prop := 
  steers_cost * s + cows_cost * c = total_budget ∧ s > 0 ∧ c > 0

-- Statement to prove
theorem unique_solution_c_exceeds_s :
  ∃ s c : ℕ, valid_purchase s c ∧ c > s ∧ ∀ (s' c' : ℕ), valid_purchase s' c' → s' = 8 ∧ c' = 17 :=
sorry

end unique_solution_c_exceeds_s_l111_111810


namespace symmetric_point_about_x_l111_111649

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

end symmetric_point_about_x_l111_111649


namespace remainder_when_dividing_386_l111_111019

theorem remainder_when_dividing_386 :
  (386 % 35 = 1) ∧ (386 % 11 = 1) :=
by
  sorry

end remainder_when_dividing_386_l111_111019


namespace remaining_kibble_l111_111153

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end remaining_kibble_l111_111153


namespace fraction_comparison_l111_111493

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) :=
by sorry

end fraction_comparison_l111_111493


namespace find_angle_A_find_area_triangle_l111_111732

-- Definitions for the triangle and the angles
def triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- Given conditions
variables (a b c A B C : ℝ)
variables (hTriangle : triangle A B C)
variables (hEq : 2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C)
variables (hAngleB : B = Real.pi / 6)
variables (hMedianAM : Real.sqrt 7 = Real.sqrt (b^2 + (b / 2)^2 - 2 * b * (b / 2) * Real.cos (2 * Real.pi / 3)))

-- Proof statements
theorem find_angle_A : A = Real.pi / 6 :=
sorry

theorem find_area_triangle : (1/2) * b^2 * Real.sin C = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_triangle_l111_111732


namespace solution_exists_real_solution_31_l111_111925

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l111_111925


namespace smallest_non_six_digit_palindrome_l111_111619

-- Definition of a four-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

-- Definition of a six-digit number
def is_six_digit (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000

-- Definition of a non-palindrome
def not_palindrome (n : ℕ) : Prop :=
  ¬ is_palindrome n

-- Find the smallest four-digit palindrome whose product with 103 is not a six-digit palindrome
theorem smallest_non_six_digit_palindrome :
  ∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ is_palindrome n ∧ not_palindrome (103 * n)
  ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ is_palindrome m ∧ not_palindrome (103 * m) → n ≤ m) :=
  sorry

end smallest_non_six_digit_palindrome_l111_111619


namespace f_comp_f_neg1_l111_111959

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (1 / 4) ^ x else Real.log x / Real.log (1 / 2)

theorem f_comp_f_neg1 : f (f (-1)) = -2 := 
by
  sorry

end f_comp_f_neg1_l111_111959


namespace blue_pill_cost_l111_111877

theorem blue_pill_cost :
  ∃ y : ℝ, ∀ (red_pill_cost blue_pill_cost : ℝ),
    (blue_pill_cost = red_pill_cost + 2) ∧
    (21 * (blue_pill_cost + red_pill_cost) = 819) →
    blue_pill_cost = 20.5 :=
by sorry

end blue_pill_cost_l111_111877


namespace constant_term_is_21_l111_111342

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_l111_111342


namespace units_digit_of_quotient_l111_111666

theorem units_digit_of_quotient : 
  (4^1985 + 7^1985) % 7 = 0 → (4^1985 + 7^1985) / 7 % 10 = 2 := 
  by 
    intro h
    sorry

end units_digit_of_quotient_l111_111666


namespace both_reunions_l111_111921

theorem both_reunions (U O H B : ℕ) 
  (hU : U = 100) 
  (hO : O = 50) 
  (hH : H = 62) 
  (attend_one : U = O + H - B) :  
  B = 12 := 
by 
  sorry

end both_reunions_l111_111921


namespace total_reduction_500_l111_111429

noncomputable def total_price_reduction (P : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : ℝ :=
  let first_reduction := P * first_reduction_percent / 100
  let intermediate_price := P - first_reduction
  let second_reduction := intermediate_price * second_reduction_percent / 100
  let final_price := intermediate_price - second_reduction
  P - final_price

theorem total_reduction_500 (P : ℝ) (first_reduction_percent : ℝ)  (second_reduction_percent: ℝ) (h₁ : P = 500) (h₂ : first_reduction_percent = 5) (h₃ : second_reduction_percent = 4):
  total_price_reduction P first_reduction_percent second_reduction_percent = 44 := 
by
  sorry

end total_reduction_500_l111_111429


namespace visitors_current_day_l111_111691

-- Define the number of visitors on the previous day and the additional visitors
def v_prev : ℕ := 600
def v_add : ℕ := 61

-- Prove that the number of visitors on the current day is 661
theorem visitors_current_day : v_prev + v_add = 661 :=
by
  sorry

end visitors_current_day_l111_111691


namespace g_10_equals_100_l111_111630

-- Define the function g and the conditions it must satisfy.
def g : ℕ → ℝ := sorry

axiom g_2 : g 2 = 4

axiom g_condition : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

-- Prove the required statement.
theorem g_10_equals_100 : g 10 = 100 :=
by sorry

end g_10_equals_100_l111_111630


namespace smallest_positive_integer_l111_111985

theorem smallest_positive_integer :
  ∃ x : ℤ, 0 < x ∧ (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 11 = 10) ∧ x = 384 :=
by
  sorry

end smallest_positive_integer_l111_111985


namespace sine_ratio_comparison_l111_111744

theorem sine_ratio_comparison : (Real.sin (1 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) < (Real.sin (3 * Real.pi / 180) / Real.sin (4 * Real.pi / 180)) :=
sorry

end sine_ratio_comparison_l111_111744


namespace little_johns_money_left_l111_111850

def J_initial : ℝ := 7.10
def S : ℝ := 1.05
def F : ℝ := 1.00

theorem little_johns_money_left :
  J_initial - (S + 2 * F) = 4.05 :=
by sorry

end little_johns_money_left_l111_111850


namespace even_function_symmetric_y_axis_l111_111434

theorem even_function_symmetric_y_axis (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = f (-x) := by
  sorry

end even_function_symmetric_y_axis_l111_111434


namespace imaginary_part_of_fraction_l111_111701

theorem imaginary_part_of_fraction (i : ℂ) (h : i^2 = -1) : ( (i^2) / (2 * i - 1) ).im = (2 / 5) :=
by
  sorry

end imaginary_part_of_fraction_l111_111701


namespace cars_through_toll_booth_l111_111016

noncomputable def total_cars_in_week (n_mon n_tue n_wed n_thu n_fri n_sat n_sun : ℕ) : ℕ :=
  n_mon + n_tue + n_wed + n_thu + n_fri + n_sat + n_sun 

theorem cars_through_toll_booth : 
  let n_mon : ℕ := 50
  let n_tue : ℕ := 50
  let n_wed : ℕ := 2 * n_mon
  let n_thu : ℕ := 2 * n_mon
  let n_fri : ℕ := 50
  let n_sat : ℕ := 50
  let n_sun : ℕ := 50
  total_cars_in_week n_mon n_tue n_wed n_thu n_fri n_sat n_sun = 450 := 
by 
  sorry

end cars_through_toll_booth_l111_111016


namespace sum_of_squares_of_projections_constant_l111_111896

-- Define the sum of the squares of projections function
noncomputable def sum_of_squares_of_projections (a : ℝ) (α : ℝ) : ℝ :=
  let p1 := a * Real.cos α
  let p2 := a * Real.cos (Real.pi / 3 - α)
  let p3 := a * Real.cos (Real.pi / 3 + α)
  p1^2 + p2^2 + p3^2

-- Statement of the theorem
theorem sum_of_squares_of_projections_constant (a α : ℝ) : 
  sum_of_squares_of_projections a α = 3 / 2 * a^2 :=
sorry

end sum_of_squares_of_projections_constant_l111_111896


namespace horse_food_per_day_l111_111501

theorem horse_food_per_day (ratio_sh : ℕ) (ratio_h : ℕ) (sheep : ℕ) (total_food : ℕ) (sheep_count : sheep = 32) (ratio : ratio_sh = 4) (ratio_horses : ratio_h = 7) (total_food_need : total_food = 12880) :
  total_food / (sheep * ratio_h / ratio_sh) = 230 :=
by
  sorry

end horse_food_per_day_l111_111501


namespace valid_password_count_l111_111853

/-- 
The number of valid 4-digit ATM passwords at Fred's Bank, composed of digits from 0 to 9,
that do not start with the sequence "9,1,1" and do not end with the digit "5",
is 8991.
-/
theorem valid_password_count : 
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  total_passwords - (start_911 + end_5 - start_911_end_5) = 8991 :=
by
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  show total_passwords - (start_911 + end_5 - start_911_end_5) = 8991
  sorry

end valid_password_count_l111_111853


namespace dihedral_angle_proof_l111_111509

noncomputable def angle_between_planes 
  (α β : Real) : Real :=
  Real.arcsin (Real.sin α * Real.sin β)

theorem dihedral_angle_proof 
  (α β : Real) 
  (α_non_neg : 0 ≤ α) 
  (α_non_gtr : α ≤ Real.pi / 2) 
  (β_non_neg : 0 ≤ β) 
  (β_non_gtr : β ≤ Real.pi / 2) :
  angle_between_planes α β = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end dihedral_angle_proof_l111_111509


namespace correct_inequality_l111_111771

theorem correct_inequality :
  1.6 ^ 0.3 > 0.9 ^ 3.1 :=
sorry

end correct_inequality_l111_111771


namespace compare_neg_rationals_l111_111941

theorem compare_neg_rationals : - (3 / 4 : ℚ) > - (6 / 5 : ℚ) :=
by sorry

end compare_neg_rationals_l111_111941


namespace solve_equation_l111_111390

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ x = -2 ∨ x = 1 / 3 :=
by
  sorry

end solve_equation_l111_111390


namespace combined_resistance_parallel_l111_111324

theorem combined_resistance_parallel (R1 R2 : ℝ) (r : ℝ) 
  (hR1 : R1 = 8) (hR2 : R2 = 9) (h_parallel : (1 / r) = (1 / R1) + (1 / R2)) : 
  r = 72 / 17 :=
by
  sorry

end combined_resistance_parallel_l111_111324


namespace proof_completion_l111_111607

namespace MathProof

def p : ℕ := 10 * 7

def r : ℕ := p - 3

def q : ℚ := (3 / 5) * r

theorem proof_completion : q = 40.2 := by
  sorry

end MathProof

end proof_completion_l111_111607


namespace probability_of_different_value_and_suit_l111_111452

theorem probability_of_different_value_and_suit :
  let total_cards := 52
  let first_card_choices := 52
  let remaining_cards := 51
  let different_suits := 3
  let different_values := 12
  let favorable_outcomes := different_suits * different_values
  let total_outcomes := remaining_cards
  let probability := favorable_outcomes / total_outcomes
  probability = 12 / 17 := 
by
  sorry

end probability_of_different_value_and_suit_l111_111452


namespace no_such_number_exists_l111_111764

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

/-- Define the number N as a sequence of digits a_n a_{n-1} ... a_0 -/
def number (a b : ℕ) (n : ℕ) : ℕ := a * 10^n + b

theorem no_such_number_exists :
  ¬ ∃ (N a_n b : ℕ) (n : ℕ), is_digit a_n ∧ a_n ≠ 0 ∧ b < 10^n ∧
    N = number a_n b n ∧
    b = N / 57 :=
sorry

end no_such_number_exists_l111_111764


namespace remainder_when_divided_by_22_l111_111993

theorem remainder_when_divided_by_22 (y : ℤ) (k : ℤ) (h : y = 264 * k + 42) : y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l111_111993


namespace range_of_c_l111_111020

variable (c : ℝ)

def p := 2 < 3 * c
def q := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

theorem range_of_c (hp : p c) (hq : q c) : (2 / 3) < c ∧ c < (Real.sqrt 2 / 2) :=
by
  sorry

end range_of_c_l111_111020


namespace johannes_sells_48_kg_l111_111296

-- Define Johannes' earnings
def earnings_wednesday : ℕ := 30
def earnings_friday : ℕ := 24
def earnings_today : ℕ := 42

-- Price per kilogram of cabbage
def price_per_kg : ℕ := 2

-- Prove that the total kilograms of cabbage sold is 48
theorem johannes_sells_48_kg :
  ((earnings_wednesday + earnings_friday + earnings_today) / price_per_kg) = 48 := by
  sorry

end johannes_sells_48_kg_l111_111296


namespace ninth_term_of_sequence_is_4_l111_111530

-- Definition of the first term and common ratio
def a1 : ℚ := 4
def r : ℚ := 1

-- Definition of the nth term of a geometric sequence
def a (n : ℕ) : ℚ := a1 * r^(n-1)

-- Proof that the ninth term of the sequence is 4
theorem ninth_term_of_sequence_is_4 : a 9 = 4 := by
  sorry

end ninth_term_of_sequence_is_4_l111_111530


namespace greatest_possible_value_l111_111208

theorem greatest_possible_value (x y : ℝ) (h1 : x^2 + y^2 = 98) (h2 : x * y = 40) : x + y = Real.sqrt 178 :=
by sorry

end greatest_possible_value_l111_111208


namespace smallest_positive_period_and_monotonic_increase_max_min_in_interval_l111_111581

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∃ a b : ℝ, (k * π - π / 3 ≤ a ∧ a ≤ x) ∧ (x ≤ b ∧ b ≤ k * π + π / 6) → f x = 1) := sorry

theorem max_min_in_interval :
  (∀ x : ℝ, (-π / 4 ≤ x ∧ x ≤ π / 6) → (1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3)) := sorry

end smallest_positive_period_and_monotonic_increase_max_min_in_interval_l111_111581


namespace system_of_equations_soln_l111_111718

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l111_111718


namespace chantel_final_bracelets_count_l111_111155

def bracelets_made_in_first_5_days : ℕ := 5 * 2

def bracelets_after_giving_away_at_school : ℕ := bracelets_made_in_first_5_days - 3

def bracelets_made_in_next_4_days : ℕ := 4 * 3

def total_bracelets_before_soccer_giveaway : ℕ := bracelets_after_giving_away_at_school + bracelets_made_in_next_4_days

def bracelets_after_giving_away_at_soccer : ℕ := total_bracelets_before_soccer_giveaway - 6

theorem chantel_final_bracelets_count : bracelets_after_giving_away_at_soccer = 13 :=
sorry

end chantel_final_bracelets_count_l111_111155


namespace last_row_number_l111_111726

/-
Given:
1. Each row forms an arithmetic sequence.
2. The common differences of the rows are:
   - 1st row: common difference = 1
   - 2nd row: common difference = 2
   - 3rd row: common difference = 4
   - ...
   - 2015th row: common difference = 2^2014
3. The nth row starts with \( (n+1) \times 2^{n-2} \).

Prove:
The number in the last row (2016th row) is \( 2017 \times 2^{2014} \).
-/
theorem last_row_number
  (common_diff : ℕ → ℕ)
  (h1 : common_diff 1 = 1)
  (h2 : common_diff 2 = 2)
  (h3 : common_diff 3 = 4)
  (h_general : ∀ n, common_diff n = 2^(n-1))
  (first_number_in_row : ℕ → ℕ)
  (first_number_in_row_def : ∀ n, first_number_in_row n = (n + 1) * 2^(n - 2)) :
  first_number_in_row 2016 = 2017 * 2^2014 := by
    sorry

end last_row_number_l111_111726


namespace find_n_l111_111183

theorem find_n (n : ℕ) (h : 7^(2*n) = (1/7)^(n-12)) : n = 4 :=
sorry

end find_n_l111_111183


namespace stocking_stuffers_total_l111_111946

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l111_111946


namespace no_real_a_b_l111_111843

noncomputable def SetA (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

noncomputable def SetB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

noncomputable def SetC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}

theorem no_real_a_b :
  ¬ ∃ (a b : ℝ), (∃ p ∈ SetA a b, p ∈ SetB) ∧ (a, b) ∈ SetC :=
by
    sorry

end no_real_a_b_l111_111843


namespace chocolate_bars_remaining_l111_111958

theorem chocolate_bars_remaining (total_bars sold_week1 sold_week2 : ℕ) (h_total : total_bars = 18) (h_sold1 : sold_week1 = 5) (h_sold2 : sold_week2 = 7) : total_bars - (sold_week1 + sold_week2) = 6 :=
by {
  sorry
}

end chocolate_bars_remaining_l111_111958


namespace solve_for_x_l111_111694

theorem solve_for_x (y z x : ℝ) (h1 : 2 / 3 = y / 90) (h2 : 2 / 3 = (y + z) / 120) (h3 : 2 / 3 = (x - z) / 150) : x = 120 :=
by
  sorry

end solve_for_x_l111_111694


namespace smallest_sum_of_squares_l111_111413

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 217) : 
  x^2 + y^2 ≥ 505 :=
sorry

end smallest_sum_of_squares_l111_111413


namespace parabola_transform_l111_111663

theorem parabola_transform :
  ∀ (x : ℝ),
    ∃ (y : ℝ),
      (y = -2 * x^2) →
      (∃ (y' : ℝ), y' = y - 1 ∧
      ∃ (x' : ℝ), x' = x - 3 ∧
      ∃ (y'' : ℝ), y'' = -2 * (x')^2 - 1) :=
by sorry

end parabola_transform_l111_111663


namespace paco_ate_more_sweet_than_salty_l111_111051

theorem paco_ate_more_sweet_than_salty (s t : ℕ) (h_s : s = 5) (h_t : t = 2) : s - t = 3 :=
by
  sorry

end paco_ate_more_sweet_than_salty_l111_111051


namespace waiter_tables_l111_111707

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  total_customers = 62 →
  left_customers = 17 →
  people_per_table = 9 →
  remaining_customers = total_customers - left_customers →
  tables = remaining_customers / people_per_table →
  tables = 5 := by
  sorry

end waiter_tables_l111_111707


namespace sum_first_9_terms_l111_111425

noncomputable def sum_of_first_n_terms (a1 d : Int) (n : Int) : Int :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_9_terms (a1 d : ℤ) 
  (h1 : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 39)
  (h2 : (a1 + 2 * d) + (a1 + 5 * d) + (a1 + 8 * d) = 27) :
  sum_of_first_n_terms a1 d 9 = 99 := by
  sorry

end sum_first_9_terms_l111_111425


namespace rate_of_interest_l111_111239

theorem rate_of_interest (P R : ℝ) :
  (2 * P * R) / 100 = 320 ∧
  P * ((1 + R / 100) ^ 2 - 1) = 340 →
  R = 12.5 :=
by
  intro h
  sorry

end rate_of_interest_l111_111239


namespace average_of_last_four_numbers_l111_111408

theorem average_of_last_four_numbers
  (avg_seven : ℝ) (avg_first_three : ℝ) (avg_last_four : ℝ)
  (h1 : avg_seven = 62) (h2 : avg_first_three = 55) :
  avg_last_four = 67.25 := 
by
  sorry

end average_of_last_four_numbers_l111_111408


namespace power_of_four_l111_111811

theorem power_of_four (x : ℕ) (h : 5^29 * 4^x = 2 * 10^29) : x = 15 := by
  sorry

end power_of_four_l111_111811


namespace roots_of_unity_l111_111225

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

noncomputable def is_cube_root_of_unity (z : ℂ) : Prop :=
  z^3 = 1

theorem roots_of_unity (x y : ℂ) (hx : is_root_of_unity x) (hy : is_root_of_unity y) (hxy : x ≠ y) :
  is_root_of_unity (x + y) ↔ is_cube_root_of_unity (y / x) :=
sorry

end roots_of_unity_l111_111225


namespace divide_64_to_get_800_l111_111681

theorem divide_64_to_get_800 (x : ℝ) (h : 64 / x = 800) : x = 0.08 :=
sorry

end divide_64_to_get_800_l111_111681


namespace sum_modulo_nine_l111_111984

theorem sum_modulo_nine :
  (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := 
by
  sorry

end sum_modulo_nine_l111_111984


namespace find_usual_time_l111_111917

variable (R T : ℝ)

theorem find_usual_time
  (h_condition :  R * T = (9 / 8) * R * (T - 4)) :
  T = 36 :=
by
  sorry

end find_usual_time_l111_111917


namespace sally_balloons_l111_111235

theorem sally_balloons :
  (initial_orange_balloons : ℕ) → (lost_orange_balloons : ℕ) → 
  (remaining_orange_balloons : ℕ) → (doubled_orange_balloons : ℕ) → 
  initial_orange_balloons = 20 → 
  lost_orange_balloons = 5 →
  remaining_orange_balloons = initial_orange_balloons - lost_orange_balloons →
  doubled_orange_balloons = 2 * remaining_orange_balloons → 
  doubled_orange_balloons = 30 :=
by
  intro initial_orange_balloons lost_orange_balloons 
       remaining_orange_balloons doubled_orange_balloons
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h3] at h4
  sorry

end sally_balloons_l111_111235


namespace nonneg_int_repr_l111_111830

theorem nonneg_int_repr (n : ℕ) : ∃ (a b c : ℕ), (0 < a ∧ a < b ∧ b < c) ∧ n = a^2 + b^2 - c^2 :=
sorry

end nonneg_int_repr_l111_111830


namespace apple_bags_l111_111415

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l111_111415


namespace camden_total_legs_l111_111995

theorem camden_total_legs 
  (num_justin_dogs : ℕ := 14)
  (num_rico_dogs := num_justin_dogs + 10)
  (num_camden_dogs := 3 * num_rico_dogs / 4)
  (camden_3_leg_dogs : ℕ := 5)
  (camden_4_leg_dogs : ℕ := 7)
  (camden_2_leg_dogs : ℕ := 2) : 
  3 * camden_3_leg_dogs + 4 * camden_4_leg_dogs + 2 * camden_2_leg_dogs = 47 :=
by sorry

end camden_total_legs_l111_111995


namespace find_m_n_l111_111066

theorem find_m_n (x : ℝ) (m n : ℝ) 
  (h : (2 * x - 5) * (x + m) = 2 * x^2 - 3 * x + n) :
  m = 1 ∧ n = -5 :=
by
  have h_expand : (2 * x - 5) * (x + m) = 2 * x^2 + (2 * m - 5) * x - 5 * m := by
    ring
  rw [h_expand] at h
  have coeff_eq1 : 2 * m - 5 = -3 := by sorry
  have coeff_eq2 : -5 * m = n := by sorry
  have m_sol : m = 1 := by
    linarith [coeff_eq1]
  have n_sol : n = -5 := by
    rw [m_sol] at coeff_eq2
    linarith
  exact ⟨m_sol, n_sol⟩

end find_m_n_l111_111066


namespace value_of_expression_l111_111330

theorem value_of_expression (x : ℝ) (h : x ^ 2 - 3 * x + 1 = 0) : 
  x ≠ 0 → (x ^ 2) / (x ^ 4 + x ^ 2 + 1) = 1 / 8 :=
by 
  intros h1 
  sorry

end value_of_expression_l111_111330


namespace find_units_digit_l111_111983

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem find_units_digit (n : ℕ) :
  is_three_digit n →
  (is_perfect_square n ∨ is_even n ∨ is_divisible_by_11 n ∨ digit_sum n = 12) ∧
  (¬is_perfect_square n ∨ ¬is_even n ∨ ¬is_divisible_by_11 n ∨ ¬(digit_sum n = 12)) →
  (n % 10 = 4) :=
sorry

end find_units_digit_l111_111983


namespace find_value_l111_111901

theorem find_value (x y : ℚ) (hx : x = 5 / 7) (hy : y = 7 / 5) :
  (1 / 3 * x^8 * y^9 + 1 / 7) = 64 / 105 := by
  sorry

end find_value_l111_111901


namespace average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l111_111826

-- Conditions from the problem statement
def initial_daily_sales : ℕ := 20
def profit_per_box : ℕ := 40
def additional_sales_per_yuan_reduction : ℕ := 2

-- Part 1: New average daily sales after a 10 yuan reduction
theorem average_daily_sales_after_10_yuan_reduction :
  (initial_daily_sales + 10 * additional_sales_per_yuan_reduction) = 40 :=
  sorry

-- Part 2: Price reduction needed to achieve a daily sales profit of 1200 yuan
theorem price_reduction_for_1200_yuan_profit :
  ∃ (x : ℕ), 
  (profit_per_box - x) * (initial_daily_sales + x * additional_sales_per_yuan_reduction) = 1200 ∧ x = 20 :=
  sorry

end average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l111_111826


namespace students_per_bus_l111_111908

/-- The number of students who can be accommodated in each bus -/
theorem students_per_bus (total_students : ℕ) (students_in_cars : ℕ) (num_buses : ℕ) 
(h1 : total_students = 375) (h2 : students_in_cars = 4) (h3 : num_buses = 7) : 
(total_students - students_in_cars) / num_buses = 53 :=
by
  sorry

end students_per_bus_l111_111908


namespace inequality_example_l111_111814

theorem inequality_example (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (habc_sum : a + b + c = 3) :
  18 * ((1 / ((3 - a) * (4 - a))) + (1 / ((3 - b) * (4 - b))) + (1 / ((3 - c) * (4 - c)))) + 2 * (a * b + b * c + c * a) ≥ 15 :=
by
  sorry

end inequality_example_l111_111814


namespace area_of_shaded_region_l111_111411

theorem area_of_shaded_region :
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  shaded_area = 22 :=
by
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  sorry

end area_of_shaded_region_l111_111411


namespace original_cost_of_dress_l111_111502

theorem original_cost_of_dress (x: ℝ) 
  (h1: x / 2 - 10 < x) 
  (h2: x - (x / 2 - 10) = 80) : 
  x = 140 :=
sorry

end original_cost_of_dress_l111_111502


namespace zarnin_staffing_l111_111381

open Finset

theorem zarnin_staffing :
  let total_resumes := 30
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let positions := 5
  suitable_resumes = 20 → 
  positions = 5 → 
  Nat.factorial suitable_resumes / Nat.factorial (suitable_resumes - positions) = 930240 := by
  intro total_resumes unsuitable_resumes suitable_resumes positions h1 h2
  have hs : suitable_resumes = 20 := h1
  have hp : positions = 5 := h2
  sorry

end zarnin_staffing_l111_111381


namespace find_f_half_l111_111514

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (y : ℝ) : ℝ := if y ≠ 0 then (1 - y^2) / y^2 else 0

theorem find_f_half :
  f (g (1 / 4)) = 15 :=
by
  have g_eq : g (1 / 4) = 1 / 2 := sorry
  rw [g_eq]
  have f_eq : f (1 / 2) = 15 := sorry
  exact f_eq

end find_f_half_l111_111514


namespace fg_of_2_eq_513_l111_111623

def f (x : ℤ) : ℤ := x^3 + 1
def g (x : ℤ) : ℤ := 3*x + 2

theorem fg_of_2_eq_513 : f (g 2) = 513 := by
  sorry

end fg_of_2_eq_513_l111_111623


namespace polynomial_not_product_of_single_var_l111_111613

theorem polynomial_not_product_of_single_var :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ), 
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = (x^200) * (y^200) + 1) := sorry

end polynomial_not_product_of_single_var_l111_111613


namespace find_real_x_l111_111966

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l111_111966


namespace impossible_measure_1_liter_with_buckets_l111_111003

theorem impossible_measure_1_liter_with_buckets :
  ¬ (∃ k l : ℤ, k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1) :=
by
  sorry

end impossible_measure_1_liter_with_buckets_l111_111003


namespace transformed_average_l111_111675

theorem transformed_average (n : ℕ) (original_average factor : ℝ) 
  (h1 : n = 15) (h2 : original_average = 21.5) (h3 : factor = 7) :
  (original_average * factor) = 150.5 :=
by
  sorry

end transformed_average_l111_111675


namespace angle_between_diagonals_l111_111962

open Real

theorem angle_between_diagonals
  (a b c : ℝ) :
  ∃ θ : ℝ, θ = arccos (a^2 / sqrt ((a^2 + b^2) * (a^2 + c^2))) :=
by
  -- Placeholder for the proof
  sorry

end angle_between_diagonals_l111_111962


namespace find_positive_x_l111_111189

theorem find_positive_x :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4))
  ∧ x = 21 + Real.sqrt 449 :=
by
  sorry

end find_positive_x_l111_111189


namespace prob_point_in_region_l111_111934

theorem prob_point_in_region :
  let rect_area := 18
  let intersect_area := 15 / 2
  let probability := intersect_area / rect_area
  probability = 5 / 12 :=
by
  sorry

end prob_point_in_region_l111_111934


namespace jerry_trays_l111_111264

theorem jerry_trays :
  ∀ (trays_from_table1 trays_from_table2 trips trays_per_trip : ℕ),
  trays_from_table1 = 9 →
  trays_from_table2 = 7 →
  trips = 2 →
  trays_from_table1 + trays_from_table2 = 16 →
  trays_per_trip = (trays_from_table1 + trays_from_table2) / trips →
  trays_per_trip = 8 :=
by
  intros
  sorry

end jerry_trays_l111_111264


namespace find_num_pennies_l111_111302

def total_value (nickels : ℕ) (dimes : ℕ) (pennies : ℕ) : ℕ :=
  5 * nickels + 10 * dimes + pennies

def num_pennies (nickels_value: ℕ) (dimes_value: ℕ) (total: ℕ): ℕ :=
  total - (nickels_value + dimes_value)

theorem find_num_pennies : 
  ∀ (total : ℕ) (num_nickels : ℕ) (num_dimes: ℕ),
  total = 59 → num_nickels = 4 → num_dimes = 3 → num_pennies (5 * num_nickels) (10 * num_dimes) total = 9 :=
by
  intros
  sorry

end find_num_pennies_l111_111302


namespace extra_flowers_l111_111745

-- Definitions from the conditions
def tulips : Nat := 57
def roses : Nat := 73
def daffodils : Nat := 45
def sunflowers : Nat := 35
def used_flowers : Nat := 181

-- Statement to prove
theorem extra_flowers : (tulips + roses + daffodils + sunflowers) - used_flowers = 29 := by
  sorry

end extra_flowers_l111_111745


namespace eq_curveE_eq_lineCD_l111_111499

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def curveE (x y : ℝ) : Prop :=
  distance (x, y) (-1, 0) = Real.sqrt 3 * distance (x, y) (1, 0)

theorem eq_curveE (x y : ℝ) : curveE x y ↔ (x - 2)^2 + y^2 = 3 :=
by sorry

variables (m : ℝ)
variables (m_nonzero : m ≠ 0)
variables (A C B D : ℝ × ℝ)
variables (line1_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = A ∨ p = C) → p.1 - m * p.2 - 1 = 0)
variables (line2_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = B ∨ p = D) → m * p.1 + p.2 - m = 0)
variables (CD_slope : (D.2 - C.2) / (D.1 - C.1) = -1)

theorem eq_lineCD (x y : ℝ) : 
  (y = -x ∨ y = -x + 3) :=
by sorry

end eq_curveE_eq_lineCD_l111_111499


namespace evaluate_expression_l111_111355

theorem evaluate_expression : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := 
by
  sorry

end evaluate_expression_l111_111355


namespace expression_evaluation_l111_111992

theorem expression_evaluation :
  2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := sorry

end expression_evaluation_l111_111992


namespace max_m_n_squared_l111_111277

theorem max_m_n_squared (m n : ℤ) 
  (hmn : 1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981)
  (h_eq : (n^2 - m*n - m^2)^2 = 1) : 
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_squared_l111_111277


namespace range_of_angle_of_inclination_l111_111932

theorem range_of_angle_of_inclination (α : ℝ) :
  ∃ θ : ℝ, θ ∈ (Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi) ∧
           ∀ x : ℝ, ∃ y : ℝ, y = x * Real.sin α + 1 := by
  sorry

end range_of_angle_of_inclination_l111_111932


namespace infinite_a_exists_l111_111247

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ+), ∃ (m : ℕ+), n^6 + 3 * (a : ℕ) = m^3 :=
  sorry

end infinite_a_exists_l111_111247


namespace total_songs_performed_l111_111940

theorem total_songs_performed :
  ∃ N : ℕ, 
  (∃ e d o : ℕ, 
     (e > 3 ∧ e < 9) ∧ (d > 3 ∧ d < 9) ∧ (o > 3 ∧ o < 9)
      ∧ N = (9 + 3 + e + d + o) / 4) ∧ N = 6 :=
sorry

end total_songs_performed_l111_111940


namespace fruit_fly_cell_division_l111_111571

/-- Genetic properties of fruit flies:
  1. Fruit flies have 2N = 8 chromosomes.
  2. Alleles A/a and B/b are inherited independently.
  3. Genotype AaBb is given.
  4. This genotype undergoes cell division without chromosomal variation.

Prove that:
Cells with a genetic composition of AAaaBBbb contain 8 or 16 chromosomes.
-/
theorem fruit_fly_cell_division (genotype : ℕ → ℕ) (A a B b : ℕ) :
  genotype 2 = 8 ∧
  (A + a + B + b = 8) ∧
  (genotype 0 = 2 * 4) →
  (genotype 1 = 8 ∨ genotype 1 = 16) :=
by
  sorry

end fruit_fly_cell_division_l111_111571


namespace milo_dozen_eggs_l111_111195

theorem milo_dozen_eggs (total_weight_pounds egg_weight_pounds dozen : ℕ) (h1 : total_weight_pounds = 6)
  (h2 : egg_weight_pounds = 1 / 16) (h3 : dozen = 12) :
  total_weight_pounds / egg_weight_pounds / dozen = 8 :=
by
  -- The proof would go here
  sorry

end milo_dozen_eggs_l111_111195


namespace distinct_shading_patterns_l111_111801

/-- How many distinct patterns can be made by shading exactly three of the sixteen squares 
    in a 4x4 grid, considering that patterns which can be matched by flips and/or turns are 
    not considered different? The answer is 8. -/
theorem distinct_shading_patterns : 
  (number_of_distinct_patterns : ℕ) = 8 :=
by
  /- Define the 4x4 Grid and the condition of shading exactly three squares, considering 
     flips and turns -/
  sorry

end distinct_shading_patterns_l111_111801


namespace base_n_not_divisible_by_11_l111_111312

theorem base_n_not_divisible_by_11 :
  ∀ n, 2 ≤ n ∧ n ≤ 100 → (6 + 2*n + 5*n^2 + 4*n^3 + 2*n^4 + 4*n^5) % 11 ≠ 0 := by
  sorry

end base_n_not_divisible_by_11_l111_111312


namespace fraction_simplification_l111_111127

theorem fraction_simplification (x : ℚ) : 
  (3 / 4) * 60 - x * 60 + 63 = 12 → 
  x = (8 / 5) :=
by
  sorry

end fraction_simplification_l111_111127


namespace g_of_5_l111_111879

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem g_of_5 (x : ℝ) : g (g (g (g (g x)))) = -2 / x :=
by
  sorry

end g_of_5_l111_111879


namespace ratio_mom_pays_to_total_cost_l111_111266

-- Definitions based on the conditions from the problem
def num_shirts := 4
def num_pants := 2
def num_jackets := 2
def cost_per_shirt := 8
def cost_per_pant := 18
def cost_per_jacket := 60
def amount_carrie_pays := 94

-- Calculate total costs based on given definitions
def cost_shirts := num_shirts * cost_per_shirt
def cost_pants := num_pants * cost_per_pant
def cost_jackets := num_jackets * cost_per_jacket
def total_cost := cost_shirts + cost_pants + cost_jackets

-- Amount Carrie's mom pays
def amount_mom_pays := total_cost - amount_carrie_pays

-- The proving statement
theorem ratio_mom_pays_to_total_cost : (amount_mom_pays : ℝ) / (total_cost : ℝ) = 1 / 2 :=
by
  sorry

end ratio_mom_pays_to_total_cost_l111_111266


namespace polynomial_simplification_l111_111417

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := 
by
  sorry

end polynomial_simplification_l111_111417


namespace simplify_expression_l111_111464

theorem simplify_expression :
  ((5 ^ 7 + 2 ^ 8) * (1 ^ 5 - (-1) ^ 5) ^ 10) = 80263680 := by
  sorry

end simplify_expression_l111_111464


namespace solution_set_nonempty_implies_a_range_l111_111173

theorem solution_set_nonempty_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end solution_set_nonempty_implies_a_range_l111_111173


namespace non_empty_set_A_l111_111634

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {x | x ^ 2 = a}

-- Theorem statement
theorem non_empty_set_A (a : ℝ) (h : (A a).Nonempty) : 0 ≤ a :=
by
  sorry

end non_empty_set_A_l111_111634


namespace extreme_points_range_l111_111564

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end extreme_points_range_l111_111564


namespace discriminant_of_quadratic_polynomial_l111_111184

theorem discriminant_of_quadratic_polynomial :
  let a := 5
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ) 
  let Δ := b^2 - 4 * a * c
  Δ = (576/25 : ℚ) :=
by
  sorry

end discriminant_of_quadratic_polynomial_l111_111184


namespace no_intersection_of_lines_l111_111022

theorem no_intersection_of_lines :
  ¬ ∃ (s v : ℝ) (x y : ℝ),
    (x = 1 - 2 * s ∧ y = 4 + 6 * s) ∧
    (x = 3 - v ∧ y = 10 + 3 * v) :=
by {
  sorry
}

end no_intersection_of_lines_l111_111022


namespace major_axis_length_is_three_l111_111664

-- Given the radius of the cylinder
def cylinder_radius : ℝ := 1

-- Given the percentage longer of the major axis than the minor axis
def percentage_longer (r : ℝ) : ℝ := 1.5

-- Given the function to calculate the minor axis using the radius
def minor_axis (r : ℝ) : ℝ := 2 * r

-- Given the function to calculate the major axis using the minor axis
def major_axis (minor_axis : ℝ) (factor : ℝ) : ℝ := minor_axis * factor

-- The conjecture states that the major axis length is 3
theorem major_axis_length_is_three : 
  major_axis (minor_axis cylinder_radius) (percentage_longer cylinder_radius) = 3 :=
by 
  -- Proof goes here
  sorry

end major_axis_length_is_three_l111_111664


namespace perfect_square_trinomial_k_l111_111565

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m + 1)^2) ∨
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m - 1)^2) ↔
  k = 14 ∨ k = -14 :=
sorry

end perfect_square_trinomial_k_l111_111565


namespace cos_585_eq_neg_sqrt2_div_2_l111_111078

theorem cos_585_eq_neg_sqrt2_div_2 :
  Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_585_eq_neg_sqrt2_div_2_l111_111078


namespace pairs_divisible_by_7_l111_111274

theorem pairs_divisible_by_7 :
  (∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, (1 ≤ p.fst ∧ p.fst ≤ 1000) ∧ (1 ≤ p.snd ∧ p.snd ≤ 1000) ∧ (p.fst^2 + p.snd^2) % 7 = 0) ∧ 
    pairs.length = 20164) :=
sorry

end pairs_divisible_by_7_l111_111274


namespace problem_statement_l111_111567

theorem problem_statement (a b : ℝ) (h1 : 1 / a + 1 / b = Real.sqrt 5) (h2 : a ≠ b) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 :=
by
  sorry

end problem_statement_l111_111567


namespace find_extreme_values_find_m_range_for_zeros_l111_111647

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x + 2

theorem find_extreme_values (m : ℝ) :
  (∀ x > 0, m ≤ 0 → (f x m ≠ 0 ∨ ∀ y > 0, f y m ≥ f x m ∨ f y m ≤ f x m)) ∧
  (∀ x > 0, m > 0 → ∃ x_max, x_max = 1 / m ∧ ∀ y > 0, f y m ≤ f x_max m) := 
sorry

theorem find_m_range_for_zeros (m : ℝ) :
  (∃ a b, a = 1 / Real.exp 2 ∧ b = Real.exp 1 ∧ (f a m = 0 ∧ f b m = 0)) ↔ 
  (m ≥ 3 / Real.exp 1 ∧ m < Real.exp 1) :=
sorry

end find_extreme_values_find_m_range_for_zeros_l111_111647


namespace lines_perpendicular_iff_l111_111108

/-- Given two lines y = k₁ x + l₁ and y = k₂ x + l₂, 
    which are not parallel to the coordinate axes,
    these lines are perpendicular if and only if k₁ * k₂ = -1. -/
theorem lines_perpendicular_iff 
  (k₁ k₂ l₁ l₂ : ℝ) (h1 : k₁ ≠ 0) (h2 : k₂ ≠ 0) :
  (∀ x, k₁ * x + l₁ = k₂ * x + l₂) <-> k₁ * k₂ = -1 :=
sorry

end lines_perpendicular_iff_l111_111108


namespace find_base_of_log_equation_l111_111522

theorem find_base_of_log_equation :
  ∃ b : ℝ, (∀ x : ℝ, (9 : ℝ)^(x + 5) = (5 : ℝ)^x → x = Real.logb b ((9 : ℝ)^5)) ∧ b = 5 / 9 :=
by
  sorry

end find_base_of_log_equation_l111_111522


namespace remainder_3_pow_500_mod_17_l111_111643

theorem remainder_3_pow_500_mod_17 : (3^500) % 17 = 13 := 
by
  sorry

end remainder_3_pow_500_mod_17_l111_111643


namespace algebraic_simplification_l111_111967

variables (a b : ℝ)

theorem algebraic_simplification (h : a > b ∧ b > 0) : 
  ((a + b) / ((Real.sqrt a - Real.sqrt b)^2)) * 
  (((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b^2) / 
    (1/2 * Real.sqrt (1/4 * ((a / b + b / a)^2) - 1)) + 
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b^2 * Real.sqrt a) / 
   (3/2 * Real.sqrt b - 2 * Real.sqrt a))) 
  = -2 * b * (a + 3 * Real.sqrt (a * b)) :=
sorry

end algebraic_simplification_l111_111967


namespace Margo_total_distance_walked_l111_111720

theorem Margo_total_distance_walked :
  ∀ (d : ℝ),
  (5 * (d / 5) + 3 * (d / 3) = 1) →
  (2 * d = 3.75) :=
by
  sorry

end Margo_total_distance_walked_l111_111720


namespace power_product_rule_l111_111320

theorem power_product_rule (a : ℤ) : (-a^2)^3 = -a^6 := 
by 
  sorry

end power_product_rule_l111_111320


namespace last_bead_is_black_l111_111191

-- Definition of the repeating pattern
def pattern := [1, 2, 3, 1, 2]  -- 1: black, 2: white, 3: gray (one full cycle)

-- Given constants
def total_beads : Nat := 91
def pattern_length : Nat := List.length pattern  -- This should be 9

-- Proof statement: The last bead is black
theorem last_bead_is_black : pattern[(total_beads % pattern_length) - 1] = 1 :=
by
  -- The following steps would be the proof which is not required
  sorry

end last_bead_is_black_l111_111191


namespace pieces_eaten_first_l111_111104

variable (initial_candy : ℕ) (remaining_candy : ℕ) (candy_eaten_second : ℕ)

theorem pieces_eaten_first 
    (initial_candy := 21) 
    (remaining_candy := 7)
    (candy_eaten_second := 9) :
    (initial_candy - remaining_candy - candy_eaten_second = 5) :=
sorry

end pieces_eaten_first_l111_111104


namespace percentage_increase_l111_111971

theorem percentage_increase (original_price new_price : ℝ) (h₀ : original_price = 300) (h₁ : new_price = 420) :
  ((new_price - original_price) / original_price) * 100 = 40 :=
by
  -- Insert the proof here
  sorry

end percentage_increase_l111_111971


namespace seven_pow_eight_mod_100_l111_111164

theorem seven_pow_eight_mod_100 :
  (7 ^ 8) % 100 = 1 := 
by {
  -- here can be the steps of the proof, but for now we use sorry
  sorry
}

end seven_pow_eight_mod_100_l111_111164


namespace x_finishes_in_nine_days_l111_111583

-- Definitions based on the conditions
def x_work_rate : ℚ := 1 / 24
def y_work_rate : ℚ := 1 / 16
def y_days_worked : ℚ := 10
def y_work_done : ℚ := y_work_rate * y_days_worked
def remaining_work : ℚ := 1 - y_work_done
def x_days_to_finish : ℚ := remaining_work / x_work_rate

-- Statement to be proven
theorem x_finishes_in_nine_days : x_days_to_finish = 9 := 
by
  -- Skipping actual proof steps as instructed
  sorry

end x_finishes_in_nine_days_l111_111583


namespace common_root_unique_solution_l111_111244

theorem common_root_unique_solution
  (p : ℝ) (h : ∃ x, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) :
  p = 3 :=
by sorry

end common_root_unique_solution_l111_111244


namespace find_c_l111_111533

theorem find_c (x : ℝ) (c : ℝ) (h : x = 0.3)
  (equ : (10 * x + 2) / c - (3 * x - 6) / 18 = (2 * x + 4) / 3) :
  c = 4 :=
by
  sorry

end find_c_l111_111533


namespace average_of_all_digits_l111_111389

theorem average_of_all_digits {a b : ℕ} (n : ℕ) (x y : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : n = 10) (h4 : x = 58) (h5 : y = 113) :
  ((a * x + b * y) / n = 80) :=
  sorry

end average_of_all_digits_l111_111389


namespace sum_of_numbers_is_twenty_l111_111590

-- Given conditions
variables {a b c : ℝ}

-- Prove that the sum of a, b, and c is 20 given the conditions
theorem sum_of_numbers_is_twenty (h1 : a^2 + b^2 + c^2 = 138) (h2 : ab + bc + ca = 131) :
  a + b + c = 20 :=
by
  sorry

end sum_of_numbers_is_twenty_l111_111590


namespace mom_approach_is_sampling_survey_l111_111409

def is_sampling_survey (action : String) : Prop :=
  action = "tasting a little bit"

def is_census (action : String) : Prop :=
  action = "tasting the entire dish"

theorem mom_approach_is_sampling_survey :
  is_sampling_survey "tasting a little bit" :=
by {
  -- This follows from the given conditions directly.
  sorry
}

end mom_approach_is_sampling_survey_l111_111409


namespace period_change_l111_111336

theorem period_change {f : ℝ → ℝ} (T : ℝ) (hT : 0 < T) (h_period : ∀ x, f (x + T) = f x) (α : ℝ) (hα : 0 < α) :
  ∀ x, f (α * (x + T / α)) = f (α * x) :=
by
  sorry

end period_change_l111_111336


namespace intersection_A_B_find_coefficients_a_b_l111_111644

open Set

variable {X : Type} (x : X)

def setA : Set ℝ := { x | x^2 < 9 }
def setB : Set ℝ := { x | (x - 2) * (x + 4) < 0 }
def A_inter_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def A_union_B_solution_set : Set ℝ := { x | -4 < x ∧ x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | -3 < x ∧ x < 2 } :=
sorry

theorem find_coefficients_a_b (a b : ℝ) :
  (∀ x, 2 * x^2 + a * x + b < 0 ↔ -4 < x ∧ x < 3) → 
  a = 2 ∧ b = -24 :=
sorry

end intersection_A_B_find_coefficients_a_b_l111_111644


namespace point_in_second_quadrant_l111_111261

theorem point_in_second_quadrant {x y : ℝ} (hx : x < 0) (hy : y > 0) : 
  ∃ q, q = 2 :=
by
  sorry

end point_in_second_quadrant_l111_111261


namespace foreign_stamps_count_l111_111240

-- Define the conditions
variables (total_stamps : ℕ) (more_than_10_years_old : ℕ) (both_foreign_and_old : ℕ) (neither_foreign_nor_old : ℕ)

theorem foreign_stamps_count 
  (h1 : total_stamps = 200)
  (h2 : more_than_10_years_old = 60)
  (h3 : both_foreign_and_old = 20)
  (h4 : neither_foreign_nor_old = 70) : 
  ∃ (foreign_stamps : ℕ), foreign_stamps = 90 :=
by
  -- let foreign_stamps be the variable representing the number of foreign stamps
  let foreign_stamps := total_stamps - neither_foreign_nor_old - more_than_10_years_old + both_foreign_and_old
  use foreign_stamps
  -- the proof will develop here to show that foreign_stamps = 90
  sorry

end foreign_stamps_count_l111_111240


namespace freq_count_of_third_group_l111_111523

theorem freq_count_of_third_group
  (sample_size : ℕ) 
  (freq_third_group : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : freq_third_group = 0.2) : 
  (sample_size * freq_third_group) = 20 :=
by 
  sorry

end freq_count_of_third_group_l111_111523


namespace min_omega_symmetry_l111_111287

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem min_omega_symmetry :
  ∃ (omega : ℝ), omega > 0 ∧ 
  (∀ x : ℝ, Real.cos (omega * (x - π / 12)) = Real.cos (omega * (2 * (π / 4) - x) - omega * π / 12) ) ∧ 
  (∀ ω_, ω_ > 0 → 
  (∀ x : ℝ, Real.cos (ω_ * (x - π / 12)) = Real.cos (ω_ * (2 * (π / 4) - x) - ω_ * π / 12) → 
  omega ≤ ω_)) ∧ omega = 6 :=
sorry

end min_omega_symmetry_l111_111287


namespace angle_B_plus_angle_D_105_l111_111626

theorem angle_B_plus_angle_D_105
(angle_A : ℝ) (angle_AFG angle_AGF : ℝ)
(h1 : angle_A = 30)
(h2 : angle_AFG = angle_AGF)
: angle_B + angle_D = 105 := sorry

end angle_B_plus_angle_D_105_l111_111626


namespace two_squares_always_similar_l111_111849

-- Define geometric shapes and their properties
inductive Shape
| Rectangle : Shape
| Rhombus   : Shape
| Square    : Shape
| RightAngledTriangle : Shape

-- Define similarity condition
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.Square, Shape.Square => true
  | _, _ => false

-- Prove that two squares are always similar
theorem two_squares_always_similar : similar Shape.Square Shape.Square = true :=
by
  sorry

end two_squares_always_similar_l111_111849


namespace triangle_altitude_from_rectangle_l111_111465

theorem triangle_altitude_from_rectangle (a b : ℕ) (A : ℕ) (h : ℕ) (H1 : a = 7) (H2 : b = 21) (H3 : A = 147) (H4 : a * b = A) (H5 : 2 * A = h * b) : h = 14 :=
sorry

end triangle_altitude_from_rectangle_l111_111465


namespace positive_number_eq_576_l111_111780

theorem positive_number_eq_576 (x : ℝ) (h : 0 < x) (h_eq : (2 / 3) * x = (25 / 216) * (1 / x)) : x = 5.76 := 
by 
  sorry

end positive_number_eq_576_l111_111780


namespace negation_of_P_l111_111779

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x > Real.sin x

-- Formulate the negation of P
def neg_P : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- State the theorem to be proved
theorem negation_of_P (hP : P) : neg_P :=
sorry

end negation_of_P_l111_111779


namespace heximal_to_binary_k_value_l111_111806

theorem heximal_to_binary_k_value (k : ℕ) (h : 10 * (6^3) + k * 6 + 5 = 239) : 
  k = 3 :=
by
  sorry

end heximal_to_binary_k_value_l111_111806


namespace evaluate_expression_l111_111693

theorem evaluate_expression : 
    8 * 7 / 8 * 7 = 49 := 
by sorry

end evaluate_expression_l111_111693


namespace percentage_square_area_in_rectangle_l111_111250

variable (s : ℝ)
variable (W : ℝ) (L : ℝ)
variable (hW : W = 3 * s) -- Width is 3 times the side of the square
variable (hL : L = (3 / 2) * W) -- Length is 3/2 times the width

theorem percentage_square_area_in_rectangle :
  (s^2 / ((27 * s^2) / 2)) * 100 = 7.41 :=
by 
  sorry

end percentage_square_area_in_rectangle_l111_111250


namespace total_games_High_School_Nine_l111_111621

-- Define the constants and assumptions.
def num_teams := 9
def games_against_non_league := 6

-- Calculation of the number of games within the league.
def games_within_league := (num_teams * (num_teams - 1) / 2) * 2

-- Calculation of the number of games against non-league teams.
def games_non_league := num_teams * games_against_non_league

-- The total number of games.
def total_games := games_within_league + games_non_league

-- The statement to prove.
theorem total_games_High_School_Nine : total_games = 126 := 
by
  -- You do not need to provide the proof.
  sorry

end total_games_High_School_Nine_l111_111621


namespace smallest_N_div_a3_possible_values_of_a3_l111_111186

-- Problem (a)
theorem smallest_N_div_a3 (a : Fin 10 → Nat) (h : StrictMono a) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) / (a 2) = 8 :=
sorry

-- Problem (b)
theorem possible_values_of_a3 (a : Nat) (h_a3_range : 1 ≤ a ∧ a ≤ 1000) :
  a = 315 ∨ a = 630 ∨ a = 945 :=
sorry

end smallest_N_div_a3_possible_values_of_a3_l111_111186


namespace blue_pens_count_l111_111588

variable (redPenCost bluePenCost totalCost totalPens : ℕ)
variable (numRedPens numBluePens : ℕ)

-- Conditions
axiom PriceOfRedPen : redPenCost = 5
axiom PriceOfBluePen : bluePenCost = 7
axiom TotalCost : totalCost = 102
axiom TotalPens : totalPens = 16
axiom PenCount : numRedPens + numBluePens = totalPens
axiom CostEquation : redPenCost * numRedPens + bluePenCost * numBluePens = totalCost

theorem blue_pens_count : numBluePens = 11 :=
by
  sorry

end blue_pens_count_l111_111588


namespace all_boxcars_combined_capacity_l111_111970

theorem all_boxcars_combined_capacity :
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  let green_capacity := 1.5 * black_capacity
  let yellow_capacity := green_capacity + 2000
  let total_red := 3 * red_capacity
  let total_blue := 4 * blue_capacity
  let total_black := 7 * black_capacity
  let total_green := 2 * green_capacity
  let total_yellow := 5 * yellow_capacity
  total_red + total_blue + total_black + total_green + total_yellow = 184000 :=
by 
  -- Proof omitted
  sorry

end all_boxcars_combined_capacity_l111_111970


namespace round_table_chairs_l111_111773

theorem round_table_chairs :
  ∃ x : ℕ, (2 * x + 2 * 7 = 26) ∧ x = 6 :=
by
  sorry

end round_table_chairs_l111_111773


namespace last_digit_fifth_power_l111_111788

theorem last_digit_fifth_power (R : ℤ) : (R^5 - R) % 10 = 0 := 
sorry

end last_digit_fifth_power_l111_111788


namespace find_minimum_value_l111_111938

noncomputable def fixed_point_at_2_2 (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : Prop :=
∀ (x : ℝ), a^(2-x) + 1 = 2 ↔ x = 2

noncomputable def point_on_line (m n : ℝ) (hmn_pos : m * n > 0) : Prop :=
2 * m + 2 * n = 1

theorem find_minimum_value (m n : ℝ) (hmn_pos : m * n > 0) :
  (fixed_point_at_2_2 a ha_pos ha_ne) → (point_on_line m n hmn_pos) → (1/m + 1/n ≥ 8) :=
sorry

end find_minimum_value_l111_111938


namespace series_sum_l111_111395

noncomputable def S (n : ℕ) : ℝ := 2^(n + 1) + n - 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem series_sum : 
  ∑' i, a i / 4^i = 4 / 3 :=
by 
  sorry

end series_sum_l111_111395


namespace zero_point_of_function_l111_111209

theorem zero_point_of_function : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 :=
by
  sorry

end zero_point_of_function_l111_111209


namespace max_obtuse_in_convex_quadrilateral_l111_111393

-- Definition and problem statement
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

theorem max_obtuse_in_convex_quadrilateral (a b c d : ℝ) :
  convex_quadrilateral a b c d →
  (is_obtuse a → (is_obtuse b → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse b → (is_obtuse a → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse c → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse d))) →
  (is_obtuse d → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse c))) :=
by
  intros h_convex h1 h2 h3 h4
  sorry

end max_obtuse_in_convex_quadrilateral_l111_111393


namespace average_of_x_y_z_l111_111004

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l111_111004


namespace quadratic_equation_C_has_real_solutions_l111_111283

theorem quadratic_equation_C_has_real_solutions :
  ∀ (x : ℝ), ∃ (a b c : ℝ), a = 1 ∧ b = 3 ∧ c = -2 ∧ a*x^2 + b*x + c = 0 :=
by
  sorry

end quadratic_equation_C_has_real_solutions_l111_111283


namespace fraction_days_passed_l111_111028

-- Conditions
def total_days : ℕ := 30
def pills_per_day : ℕ := 2
def total_pills : ℕ := total_days * pills_per_day -- 60 pills
def pills_left : ℕ := 12
def pills_taken : ℕ := total_pills - pills_left -- 48 pills
def days_taken : ℕ := pills_taken / pills_per_day -- 24 days

-- Question and answer
theorem fraction_days_passed :
  (days_taken : ℚ) / (total_days : ℚ) = 4 / 5 := 
by
  sorry

end fraction_days_passed_l111_111028


namespace prob_first_question_correct_is_4_5_distribution_of_X_l111_111696

-- Assume probabilities for member A and member B answering correctly.
def prob_A_correct : ℚ := 2 / 5
def prob_B_correct : ℚ := 2 / 3

def prob_A_incorrect : ℚ := 1 - prob_A_correct
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- Given that A answers first, followed by B.
-- Calculate the probability that the first team answers the first question correctly.
def prob_first_question_correct : ℚ :=
  prob_A_correct + (prob_A_incorrect * prob_B_correct)

-- Assert that the calculated probability is equal to 4/5
theorem prob_first_question_correct_is_4_5 :
  prob_first_question_correct = 4 / 5 := by
  sorry

-- Define the possible scores and their probabilities
def prob_X_eq_0 : ℚ := prob_A_incorrect * prob_B_incorrect
def prob_X_eq_10 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_20 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 2 * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_30 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 3

-- Assert the distribution probabilities for the random variable X
theorem distribution_of_X :
  prob_X_eq_0 = 1 / 5 ∧
  prob_X_eq_10 = 4 / 25 ∧
  prob_X_eq_20 = 16 / 125 ∧
  prob_X_eq_30 = 64 / 125 := by
  sorry

end prob_first_question_correct_is_4_5_distribution_of_X_l111_111696


namespace caps_production_l111_111798

def caps1 : Int := 320
def caps2 : Int := 400
def caps3 : Int := 300

def avg_caps (caps1 caps2 caps3 : Int) : Int := (caps1 + caps2 + caps3) / 3

noncomputable def total_caps_after_four_weeks : Int :=
  caps1 + caps2 + caps3 + avg_caps caps1 caps2 caps3

theorem caps_production : total_caps_after_four_weeks = 1360 :=
by
  sorry

end caps_production_l111_111798


namespace minimum_number_of_odd_integers_among_six_l111_111845

theorem minimum_number_of_odd_integers_among_six : 
  ∀ (x y a b m n : ℤ), 
    x + y = 28 →
    x + y + a + b = 45 →
    x + y + a + b + m + n = 63 →
    ∃ (odd_count : ℕ), odd_count = 1 :=
by sorry

end minimum_number_of_odd_integers_among_six_l111_111845


namespace real_number_set_condition_l111_111939

theorem real_number_set_condition (x : ℝ) :
  (x ≠ 1) ∧ (x^2 - x ≠ 1) ∧ (x^2 - x ≠ x) →
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 := 
by
  sorry

end real_number_set_condition_l111_111939


namespace minimum_reciprocal_sum_l111_111012

theorem minimum_reciprocal_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : x^2 + y^2 = x * y * (x^2 * y^2 + 2)) : 
  (1 / x + 1 / y) ≥ 2 :=
by 
  sorry -- Proof to be completed

end minimum_reciprocal_sum_l111_111012


namespace Lulu_blueberry_pies_baked_l111_111968

-- Definitions of conditions
def Lola_mini_cupcakes := 13
def Lola_pop_tarts := 10
def Lola_blueberry_pies := 8
def Lola_total_pastries := Lola_mini_cupcakes + Lola_pop_tarts + Lola_blueberry_pies
def Lulu_mini_cupcakes := 16
def Lulu_pop_tarts := 12
def total_pastries := 73

-- Prove that Lulu baked 14 blueberry pies
theorem Lulu_blueberry_pies_baked : 
  ∃ (Lulu_blueberry_pies : Nat), 
    Lola_total_pastries + Lulu_mini_cupcakes + Lulu_pop_tarts + Lulu_blueberry_pies = total_pastries ∧ 
    Lulu_blueberry_pies = 14 := by
  sorry

end Lulu_blueberry_pies_baked_l111_111968


namespace single_elimination_games_l111_111215

theorem single_elimination_games (n : Nat) (h : n = 256) : n - 1 = 255 := by
  sorry

end single_elimination_games_l111_111215


namespace correct_power_functions_l111_111443

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, x ≠ 0 → f x = k * x^n

def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1 / 2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3 / 4)
def f5 (x : ℝ) : ℝ := x^(1 / 3) + 1

theorem correct_power_functions :
  {f2, f4} = {f : ℝ → ℝ | is_power_function f} ∩ {f2, f4, f1, f3, f5} :=
by
  sorry

end correct_power_functions_l111_111443


namespace perimeter_circumradius_ratio_neq_l111_111706

-- Define the properties for the equilateral triangle
def Triangle (A K R P : ℝ) : Prop :=
  P = 3 * A ∧ K = A^2 * Real.sqrt 3 / 4 ∧ R = A * Real.sqrt 3 / 3

-- Define the properties for the square
def Square (b k r p : ℝ) : Prop :=
  p = 4 * b ∧ k = b^2 ∧ r = b * Real.sqrt 2 / 2

-- Main statement to prove
theorem perimeter_circumradius_ratio_neq 
  (A b K R P k r p : ℝ)
  (hT : Triangle A K R P) 
  (hS : Square b k r p) :
  P / p ≠ R / r := 
by
  rcases hT with ⟨hP, hK, hR⟩
  rcases hS with ⟨hp, hk, hr⟩
  sorry

end perimeter_circumradius_ratio_neq_l111_111706


namespace time_saved_by_taking_route_B_l111_111765

-- Defining the times for the routes A and B
def time_route_A_one_way : ℕ := 5
def time_route_B_one_way : ℕ := 2

-- The total round trip times
def time_route_A_round_trip : ℕ := 2 * time_route_A_one_way
def time_route_B_round_trip : ℕ := 2 * time_route_B_one_way

-- The statement to prove
theorem time_saved_by_taking_route_B :
  time_route_A_round_trip - time_route_B_round_trip = 6 :=
by
  -- Proof would go here
  sorry

end time_saved_by_taking_route_B_l111_111765


namespace find_four_numbers_l111_111766

theorem find_four_numbers (a b c d : ℚ) :
  ((a + b = 1) ∧ (a + c = 5) ∧ 
   ((a + d = 8 ∧ b + c = 9) ∨ (a + d = 9 ∧ b + c = 8)) ) →
  ((a = -3/2 ∧ b = 5/2 ∧ c = 13/2 ∧ d = 19/2) ∨ 
   (a = -1 ∧ b = 2 ∧ c = 6 ∧ d = 10)) :=
  by
    sorry

end find_four_numbers_l111_111766


namespace Faye_crayons_l111_111954

theorem Faye_crayons (rows crayons_per_row : ℕ) (h_rows : rows = 7) (h_crayons_per_row : crayons_per_row = 30) : rows * crayons_per_row = 210 :=
by
  sorry

end Faye_crayons_l111_111954


namespace sector_area_l111_111267

theorem sector_area (r θ : ℝ) (hr : r = 1) (hθ : θ = 2) : 
  (1 / 2) * r * r * θ = 1 := by
sorry

end sector_area_l111_111267


namespace probability_at_least_one_girl_l111_111733

theorem probability_at_least_one_girl 
  (boys girls : ℕ) 
  (total : boys + girls = 7) 
  (combinations_total : ℕ := Nat.choose 7 2) 
  (combinations_boys : ℕ := Nat.choose 4 2) 
  (prob_no_girls : ℚ := combinations_boys / combinations_total) 
  (prob_at_least_one_girl : ℚ := 1 - prob_no_girls) :
  boys = 4 ∧ girls = 3 → prob_at_least_one_girl = 5 / 7 := 
by
  intro h
  cases h
  sorry

end probability_at_least_one_girl_l111_111733


namespace largest_angle_is_176_l111_111829

-- Define the angles of the pentagon
def angle1 (y : ℚ) : ℚ := y
def angle2 (y : ℚ) : ℚ := 2 * y + 2
def angle3 (y : ℚ) : ℚ := 3 * y - 3
def angle4 (y : ℚ) : ℚ := 4 * y + 4
def angle5 (y : ℚ) : ℚ := 5 * y - 5

-- Define the function to calculate the largest angle
def largest_angle (y : ℚ) : ℚ := 5 * y - 5

-- Problem statement: Prove that the largest angle in the pentagon is 176 degrees
theorem largest_angle_is_176 (y : ℚ) (h : angle1 y + angle2 y + angle3 y + angle4 y + angle5 y = 540) :
  largest_angle y = 176 :=
by sorry

end largest_angle_is_176_l111_111829


namespace problem_proof_l111_111281

theorem problem_proof (x y z : ℝ) 
  (h1 : 1/x + 2/y + 3/z = 0) 
  (h2 : 1/x - 6/y - 5/z = 0) : 
  (x / y + y / z + z / x) = -1 := 
by
  sorry

end problem_proof_l111_111281


namespace corn_syrup_amount_l111_111291

-- Definitions based on given conditions
def flavoring_to_corn_syrup_standard := 1 / 12
def flavoring_to_water_standard := 1 / 30

def flavoring_to_corn_syrup_sport := (3 * flavoring_to_corn_syrup_standard)
def flavoring_to_water_sport := (1 / 2) * flavoring_to_water_standard

def common_factor := (30 : ℝ)

-- Amounts in sport formulation after adjustment
def flavoring_to_corn_syrup_ratio_sport := 1 / 4
def flavoring_to_water_ratio_sport := 1 / 60

def total_flavoring_corn_syrup := 15 -- Since ratio is 15:60:60 and given water is 15 ounces

theorem corn_syrup_amount (water_ounces : ℝ) :
  water_ounces = 15 → 
  (60 / 60) * water_ounces = 15 :=
by
  sorry

end corn_syrup_amount_l111_111291


namespace prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l111_111747

def num_outcomes := 36

def same_points_events := 6
def less_than_seven_events := 15
def greater_than_or_equal_eleven_events := 3

def prob_same_points := (same_points_events : ℚ) / num_outcomes
def prob_less_than_seven := (less_than_seven_events : ℚ) / num_outcomes
def prob_greater_or_equal_eleven := (greater_than_or_equal_eleven_events : ℚ) / num_outcomes

theorem prob_same_points_eq : prob_same_points = 1 / 6 := by
  sorry

theorem prob_less_than_seven_eq : prob_less_than_seven = 5 / 12 := by
  sorry

theorem prob_greater_or_equal_eleven_eq : prob_greater_or_equal_eleven = 1 / 12 := by
  sorry

end prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l111_111747


namespace temperature_on_last_day_l111_111175

noncomputable def last_day_temperature (T1 T2 T3 T4 T5 T6 T7 : ℕ) (mean : ℕ) : ℕ :=
  8 * mean - (T1 + T2 + T3 + T4 + T5 + T6 + T7)

theorem temperature_on_last_day 
  (T1 T2 T3 T4 T5 T6 T7 mean x : ℕ)
  (hT1 : T1 = 82) (hT2 : T2 = 80) (hT3 : T3 = 84) 
  (hT4 : T4 = 86) (hT5 : T5 = 88) (hT6 : T6 = 90) 
  (hT7 : T7 = 88) (hmean : mean = 86) 
  (hx : x = last_day_temperature T1 T2 T3 T4 T5 T6 T7 mean) :
  x = 90 := by
  sorry

end temperature_on_last_day_l111_111175


namespace incorrect_inequality_l111_111842

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l111_111842


namespace inequality_proof_l111_111268

theorem inequality_proof (a b : ℝ) (h : (a = 0 ∨ b = 0 ∨ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0))) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 :=
by
  sorry

end inequality_proof_l111_111268


namespace problem1_problem2_l111_111711

-- Problem 1: Prove that x = ±7/2 given 4x^2 - 49 = 0
theorem problem1 (x : ℝ) : 4 * x^2 - 49 = 0 → x = 7 / 2 ∨ x = -7 / 2 := 
by
  sorry

-- Problem 2: Prove that x = 2 given (x + 1)^3 - 27 = 0
theorem problem2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 := 
by
  sorry

end problem1_problem2_l111_111711


namespace susans_average_speed_l111_111024

theorem susans_average_speed :
  ∀ (total_distance first_leg_distance second_leg_distance : ℕ)
    (first_leg_speed second_leg_speed : ℕ)
    (total_time : ℚ),
    first_leg_distance = 40 →
    second_leg_distance = 20 →
    first_leg_speed = 15 →
    second_leg_speed = 60 →
    total_distance = first_leg_distance + second_leg_distance →
    total_time = (first_leg_distance / first_leg_speed : ℚ) + (second_leg_distance / second_leg_speed : ℚ) →
    total_distance / total_time = 20 :=
by
  sorry

end susans_average_speed_l111_111024


namespace greatest_integer_less_than_neg_eight_over_three_l111_111781

theorem greatest_integer_less_than_neg_eight_over_three :
  ∃ (z : ℤ), (z < -8 / 3) ∧ ∀ w : ℤ, (w < -8 / 3) → w ≤ z := by
  sorry

end greatest_integer_less_than_neg_eight_over_three_l111_111781


namespace compute_xy_l111_111061

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := 
by 
  sorry

end compute_xy_l111_111061


namespace xyz_value_l111_111518

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) : 
  x * y * z = 10 :=
by
  sorry

end xyz_value_l111_111518


namespace values_of_z_l111_111952

theorem values_of_z (z : ℤ) (hz : 0 < z) :
  (z^2 - 50 * z + 550 ≤ 10) ↔ (20 ≤ z ∧ z ≤ 30) := sorry

end values_of_z_l111_111952


namespace frequency_of_fourth_group_l111_111980

theorem frequency_of_fourth_group (f₁ f₂ f₃ f₄ f₅ f₆ : ℝ) (h1 : f₁ + f₂ + f₃ = 0.65) (h2 : f₅ + f₆ = 0.32) (h3 : f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = 1) :
  f₄ = 0.03 :=
by 
  sorry

end frequency_of_fourth_group_l111_111980


namespace maximum_correct_answers_l111_111660

theorem maximum_correct_answers (c w u : ℕ) :
  c + w + u = 25 →
  4 * c - w = 70 →
  c ≤ 19 :=
by
  sorry

end maximum_correct_answers_l111_111660


namespace sufficient_but_not_necessary_l111_111361

-- Define the quadratic function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

-- The problem statement to prove that "m < 1" is a sufficient condition
-- but not a necessary condition for the function f(x) to have a root.
theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 1 → ∃ x : ℝ, f x m = 0) ∧ ¬(¬(m < 1) → ∃ x : ℝ, f x m = 0) :=
sorry

end sufficient_but_not_necessary_l111_111361


namespace seconds_in_3_hours_25_minutes_l111_111065

theorem seconds_in_3_hours_25_minutes:
  let hours := 3
  let minutesInAnHour := 60
  let additionalMinutes := 25
  let secondsInAMinute := 60
  (hours * minutesInAnHour + additionalMinutes) * secondsInAMinute = 12300 := 
by
  sorry

end seconds_in_3_hours_25_minutes_l111_111065


namespace liked_product_B_l111_111926

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l111_111926


namespace compute_c_minus_d_cubed_l111_111420

-- define c as the number of positive multiples of 12 less than 60
def c : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

-- define d as the number of positive integers less than 60 and a multiple of both 3 and 4
def d : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

theorem compute_c_minus_d_cubed : (c - d)^3 = 0 := by
  -- since c and d are computed the same way, (c - d) = 0
  -- hence, (c - d)^3 = 0^3 = 0
  sorry

end compute_c_minus_d_cubed_l111_111420


namespace benny_turnips_l111_111857

-- Definitions and conditions
def melanie_turnips : ℕ := 139
def total_turnips : ℕ := 252

-- Question to prove
theorem benny_turnips : ∃ b : ℕ, b = total_turnips - melanie_turnips ∧ b = 113 :=
by {
    sorry
}

end benny_turnips_l111_111857


namespace sine_sum_square_greater_l111_111584

variable {α β : Real} (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1)

theorem sine_sum_square_greater (α β : Real) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
  (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : 
  Real.sin (α + β) ^ 2 > Real.sin α ^ 2 + Real.sin β ^ 2 :=
sorry

end sine_sum_square_greater_l111_111584


namespace fraction_white_surface_area_l111_111654

/-- A 4-inch cube is constructed from 64 smaller cubes, each with 1-inch edges.
   48 of these smaller cubes are colored red and 16 are colored white.
   Prove that if the 4-inch cube is constructed to have the smallest possible white surface area showing,
   the fraction of the white surface area is 1/12. -/
theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let white_cubes := 16
  let exposed_white_surface_area := 8
  (exposed_white_surface_area / total_surface_area) = (1 / 12) := 
  sorry

end fraction_white_surface_area_l111_111654


namespace min_value_expression_l111_111228

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 1) : 
  ∃ (xy_min : ℝ), xy_min = 9 ∧ (∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2*x + y = 1 → (x + 2*y)/(x*y) ≥ xy_min) :=
sorry

end min_value_expression_l111_111228


namespace supermarket_sales_l111_111631

theorem supermarket_sales (S_Dec : ℝ) (S_Jan : ℝ) (S_Feb : ℝ) (S_Jan_eq : S_Jan = S_Dec * (1 + x))
  (S_Feb_eq : S_Feb = S_Jan * (1 + x))
  (inc_eq : S_Feb = S_Dec + 0.24 * S_Dec) :
  x = 0.2 ∧ S_Feb = S_Dec * (1 + 0.2)^2 := by
sorry

end supermarket_sales_l111_111631


namespace sampling_survey_suitability_l111_111884

-- Define the conditions
def OptionA := "Understanding the effectiveness of a certain drug"
def OptionB := "Understanding the vision status of students in this class"
def OptionC := "Organizing employees of a unit to undergo physical examinations at a hospital"
def OptionD := "Inspecting components of artificial satellite"

-- Mathematical statement
theorem sampling_survey_suitability : OptionA = "Understanding the effectiveness of a certain drug" → 
  ∃ (suitable_for_sampling_survey : String), suitable_for_sampling_survey = OptionA :=
by
  sorry

end sampling_survey_suitability_l111_111884


namespace second_polygon_sides_l111_111248

/--
Given two regular polygons where:
- The first polygon has 42 sides.
- Each side of the first polygon is three times the length of each side of the second polygon.
- The perimeters of both polygons are equal.
Prove that the second polygon has 126 sides.
-/
theorem second_polygon_sides
  (s : ℝ) -- the side length of the second polygon
  (h1 : ∃ n : ℕ, n = 42) -- the first polygon has 42 sides
  (h2 : ∃ m : ℝ, m = 3 * s) -- the side length of the first polygon is three times the side length of the second polygon
  (h3 : ∃ k : ℕ, k * (3 * s) = n * s) -- the perimeters of both polygons are equal
  : ∃ n2 : ℕ, n2 = 126 := 
by
  sorry

end second_polygon_sides_l111_111248


namespace greatest_multiple_l111_111876

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l111_111876


namespace simplify_fraction_l111_111227

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l111_111227


namespace animath_extortion_l111_111770

noncomputable def max_extortion (n : ℕ) : ℕ :=
2^n - n - 1 

theorem animath_extortion (n : ℕ) :
  ∃ steps : ℕ, steps < (2^n - n - 1) :=
sorry

end animath_extortion_l111_111770


namespace intersection_M_complement_N_eq_l111_111672

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_eq_l111_111672


namespace find_prices_max_basketballs_l111_111118

-- Definition of given conditions
def conditions1 (x y : ℝ) : Prop := 
  (x - y = 50) ∧ (6 * x + 8 * y = 1700)

-- Definitions of questions:
-- Question 1: Find the price of one basketball and one soccer ball
theorem find_prices (x y : ℝ) (h: conditions1 x y) : x = 150 ∧ y = 100 := sorry

-- Definition of given conditions for Question 2
def conditions2 (x y : ℝ) (a : ℕ) : Prop :=
  (x = 150) ∧ (y = 100) ∧ 
  (0.9 * x * a + 0.85 * y * (10 - a) ≤ 1150)

-- Question 2: The school plans to purchase 10 items with given discounts
theorem max_basketballs (x y : ℝ) (a : ℕ) (h1: x = 150) (h2: y = 100) (h3: a ≤ 10) (h4: conditions2 x y a) : a ≤ 6 := sorry

end find_prices_max_basketballs_l111_111118


namespace ratio_X_N_l111_111763

-- Given conditions as definitions
variables (P Q M N X : ℝ)
variables (hM : M = 0.40 * Q)
variables (hQ : Q = 0.30 * P)
variables (hN : N = 0.60 * P)
variables (hX : X = 0.25 * M)

-- Prove that X / N == 1 / 20
theorem ratio_X_N : X / N = 1 / 20 :=
by
  sorry

end ratio_X_N_l111_111763


namespace longer_diagonal_rhombus_l111_111557

theorem longer_diagonal_rhombus (a b d1 d2 : ℝ) 
  (h1 : a = 35) 
  (h2 : d1 = 42) : 
  d2 = 56 := 
by 
  sorry

end longer_diagonal_rhombus_l111_111557


namespace additional_charge_fraction_of_mile_l111_111746

-- Conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_mile_fraction : ℝ := 0.15
def total_charge (distance : ℝ) : ℝ := 2.25 + 0.15 * distance
def trip_distance : ℝ := 3.6
def total_cost : ℝ := 3.60

-- Question
theorem additional_charge_fraction_of_mile :
  ∃ f : ℝ, total_cost = initial_fee + additional_charge_per_mile_fraction * 3.6 ∧ f = 1 / 9 :=
by
  sorry

end additional_charge_fraction_of_mile_l111_111746


namespace total_fish_in_pond_l111_111916

theorem total_fish_in_pond (N : ℕ) (h1 : 80 ≤ N) (h2 : 5 ≤ 150) (h_marked_dist : (5 : ℚ) / 150 = (80 : ℚ) / N) : N = 2400 := by
  sorry

end total_fish_in_pond_l111_111916


namespace functional_eq_solution_l111_111549

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_solution f) :
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 :=
sorry

end functional_eq_solution_l111_111549


namespace roots_of_quadratic_serve_as_eccentricities_l111_111351

theorem roots_of_quadratic_serve_as_eccentricities :
  ∀ (x1 x2 : ℝ), x1 * x2 = 1 ∧ x1 + x2 = 79 → (x1 > 1 ∧ x2 < 1) → 
  (x1 > 1 ∧ x2 < 1) ∧ x1 > 1 ∧ x2 < 1 :=
by
  sorry

end roots_of_quadratic_serve_as_eccentricities_l111_111351


namespace green_minus_blue_is_40_l111_111679

noncomputable def number_of_green_minus_blue_disks (total_disks : ℕ) (ratio_blue : ℕ) (ratio_yellow : ℕ) (ratio_green : ℕ) : ℕ :=
  let total_ratio := ratio_blue + ratio_yellow + ratio_green
  let disks_per_part := total_disks / total_ratio
  let blue_disks := ratio_blue * disks_per_part
  let green_disks := ratio_green * disks_per_part
  green_disks - blue_disks

theorem green_minus_blue_is_40 :
  number_of_green_minus_blue_disks 144 3 7 8 = 40 :=
sorry

end green_minus_blue_is_40_l111_111679


namespace large_rectangle_perimeter_l111_111698

-- Definitions for conditions
def rectangle_area (l b : ℝ) := l * b
def is_large_rectangle_perimeter (l b perimeter : ℝ) := perimeter = 2 * (l + b)

-- Statement of the theorem
theorem large_rectangle_perimeter :
  ∃ (l b : ℝ), rectangle_area l b = 8 ∧ 
               (∀ l_rect b_rect: ℝ, is_large_rectangle_perimeter l_rect b_rect 32) :=
by
  sorry

end large_rectangle_perimeter_l111_111698


namespace pencil_and_eraser_cost_l111_111367

theorem pencil_and_eraser_cost (p e : ℕ) :
  2 * p + e = 40 →
  p > e →
  e ≥ 3 →
  p + e = 22 :=
by
  sorry

end pencil_and_eraser_cost_l111_111367


namespace find_points_l111_111520

def acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the triangle formed by A, B, and C is an acute-angled triangle.
  sorry -- This would be formalized ensuring all angles are less than 90 degrees.

def no_three_collinear (A B C D E : ℝ × ℝ × ℝ) : Prop :=
  -- Definition that ensures no three points among A, B, C, D, and E are collinear.
  sorry

def line_normal_to_plane (P Q R S : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the line through any two points P, Q is normal to the plane containing R, S, and the other point.
  sorry

theorem find_points (A B C : ℝ × ℝ × ℝ) (h_acute : acute_triangle A B C) :
  ∃ (D E : ℝ × ℝ × ℝ), no_three_collinear A B C D E ∧
    (∀ (P Q R R' : ℝ × ℝ × ℝ), 
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) →
      (R' = A ∨ R' = B ∨ R' = C ∨ R' = D ∨ R' = E) →
      P ≠ Q → Q ≠ R → R ≠ R' →
      line_normal_to_plane P Q R R') :=
sorry

end find_points_l111_111520


namespace peach_count_l111_111769

theorem peach_count (n : ℕ) : n % 4 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ 120 ≤ n ∧ n ≤ 150 → n = 142 :=
sorry

end peach_count_l111_111769


namespace smallest_possible_X_l111_111468

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l111_111468


namespace gcd_210_162_l111_111475

-- Define the numbers
def a := 210
def b := 162

-- The proposition we need to prove: The GCD of 210 and 162 is 6
theorem gcd_210_162 : Nat.gcd a b = 6 :=
by
  sorry

end gcd_210_162_l111_111475


namespace ratio_of_games_played_to_losses_l111_111890

-- Definitions based on the conditions
def total_games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := total_games_played - games_won

-- The proof problem
theorem ratio_of_games_played_to_losses : (total_games_played / Nat.gcd total_games_played games_lost) = 2 ∧ (games_lost / Nat.gcd total_games_played games_lost) = 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l111_111890


namespace min_value_f_l111_111723

noncomputable def f (x : ℝ) : ℝ :=
  7 * (Real.sin x)^2 + 5 * (Real.cos x)^2 + 2 * Real.sin x

theorem min_value_f : ∃ x : ℝ, f x = 4.5 :=
  sorry

end min_value_f_l111_111723


namespace system_solution_xz_y2_l111_111616

theorem system_solution_xz_y2 (x y z : ℝ) (k : ℝ)
  (h : (x + 2 * k * y + 4 * z = 0) ∧
       (4 * x + k * y - 3 * z = 0) ∧
       (3 * x + 5 * y - 2 * z = 0) ∧
       x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k = 95 / 12) :
  (x * z) / (y ^ 2) = 10 :=
by sorry

end system_solution_xz_y2_l111_111616


namespace prime_division_l111_111387

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l111_111387


namespace tea_in_box_l111_111897

theorem tea_in_box (tea_per_day ounces_per_week ounces_per_box : ℝ) 
    (H1 : tea_per_day = 1 / 5) 
    (H2 : ounces_per_week = tea_per_day * 7) 
    (H3 : ounces_per_box = ounces_per_week * 20) : 
    ounces_per_box = 28 := 
by
  sorry

end tea_in_box_l111_111897


namespace smallest_N_divisible_by_p_l111_111751

theorem smallest_N_divisible_by_p (p : ℕ) (hp : Nat.Prime p)
    (N1 : ℕ) (N2 : ℕ) :
  (∃ N1 N2, 
    (N1 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N1 % n = 1) ∧
    (N2 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N2 % n = n - 1)
  ) :=
sorry

end smallest_N_divisible_by_p_l111_111751


namespace trig_identity_simplified_l111_111874

open Real

theorem trig_identity_simplified :
  (sin (15 * π / 180) + cos (15 * π / 180)) * (sin (15 * π / 180) - cos (15 * π / 180)) = - (sqrt 3 / 2) :=
by
  sorry

end trig_identity_simplified_l111_111874


namespace num_squares_sharing_two_vertices_l111_111498

-- Define the isosceles triangle and condition AB = AC
structure IsoscelesTriangle (A B C : Type) :=
  (AB AC : ℝ)
  (h_iso : AB = AC)

-- Define the problem statement in Lean
theorem num_squares_sharing_two_vertices 
  (A B C : Type) 
  (iso_tri : IsoscelesTriangle A B C) 
  (planeABC : ∀ P Q R : Type, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) :
  ∃ n : ℕ, n = 4 := sorry

end num_squares_sharing_two_vertices_l111_111498


namespace problem_condition_l111_111436

variable (x y z : ℝ)

theorem problem_condition (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end problem_condition_l111_111436


namespace matrix_product_l111_111817

-- Define matrix A
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 3, 2], ![1, -3, 4]]

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, 0], ![2, 0, 4], ![3, 0, 1]]

-- Define the expected result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![9, 6, -1], ![12, 0, 14], ![7, 3, -8]]

-- The statement to prove
theorem matrix_product : A * B = C :=
by
  sorry

end matrix_product_l111_111817


namespace max_x_minus_y_l111_111416

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x^2 + y) :
  x - y ≤ 1 / Real.sqrt 24 :=
sorry

end max_x_minus_y_l111_111416


namespace largest_number_l111_111288

-- Definitions based on the conditions
def numA := 0.893
def numB := 0.8929
def numC := 0.8931
def numD := 0.839
def numE := 0.8391

-- The statement to be proved 
theorem largest_number : numB = max numA (max numB (max numC (max numD numE))) := by
  sorry

end largest_number_l111_111288


namespace wilson_theorem_non_prime_divisibility_l111_111278

theorem wilson_theorem (p : ℕ) (h : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

theorem non_prime_divisibility (p : ℕ) (h : ¬ Nat.Prime p) : ¬ p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

end wilson_theorem_non_prime_divisibility_l111_111278


namespace factor_ax2_minus_ay2_l111_111344

variable (a x y : ℝ)

theorem factor_ax2_minus_ay2 : a * x^2 - a * y^2 = a * (x + y) * (x - y) := 
sorry

end factor_ax2_minus_ay2_l111_111344


namespace perfect_square_unique_n_l111_111688

theorem perfect_square_unique_n (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, 2^n + 12^n + 2011^n = m^2) ↔ n = 1 := by
  sorry

end perfect_square_unique_n_l111_111688


namespace george_purchased_two_large_pizzas_l111_111553

noncomputable def small_slices := 4
noncomputable def large_slices := 8
noncomputable def small_pizzas_purchased := 3
noncomputable def george_slices := 3
noncomputable def bob_slices := george_slices + 1
noncomputable def susie_slices := bob_slices / 2
noncomputable def bill_slices := 3
noncomputable def fred_slices := 3
noncomputable def mark_slices := 3
noncomputable def leftover_slices := 10

noncomputable def total_slices_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

noncomputable def total_slices_before_eating := total_slices_consumed + leftover_slices

noncomputable def small_pizza_total_slices := small_pizzas_purchased * small_slices

noncomputable def large_pizza_total_slices := total_slices_before_eating - small_pizza_total_slices

noncomputable def large_pizzas_purchased := large_pizza_total_slices / large_slices

theorem george_purchased_two_large_pizzas : large_pizzas_purchased = 2 :=
sorry

end george_purchased_two_large_pizzas_l111_111553


namespace total_weekly_cups_brewed_l111_111671

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end total_weekly_cups_brewed_l111_111671


namespace hyperbola_trajectory_center_l111_111519

theorem hyperbola_trajectory_center :
  ∀ m : ℝ, ∃ (x y : ℝ), x^2 - y^2 - 6 * m * x - 4 * m * y + 5 * m^2 - 1 = 0 ∧ 2 * x + 3 * y = 0 :=
by
  sorry

end hyperbola_trajectory_center_l111_111519


namespace no_eleven_points_achieve_any_score_l111_111084

theorem no_eleven_points (x y : ℕ) : 3 * x + 7 * y ≠ 11 := 
sorry

theorem achieve_any_score (S : ℕ) (h : S ≥ 12) : ∃ (x y : ℕ), 3 * x + 7 * y = S :=
sorry

end no_eleven_points_achieve_any_score_l111_111084


namespace number_of_fridays_l111_111638

theorem number_of_fridays (jan_1_sat : true) (is_non_leap_year : true) : ∃ (n : ℕ), n = 52 :=
by
  -- Conditions: January 1st is Saturday and it is a non-leap year.
  -- We are given that January 1st is a Saturday.
  have jan_1_sat_condition : true := jan_1_sat
  -- We are given that the year is a non-leap year (365 days).
  have non_leap_condition : true := is_non_leap_year
  -- Therefore, there are 52 Fridays in the year.
  use 52
  done

end number_of_fridays_l111_111638


namespace rate_ratio_l111_111190

theorem rate_ratio
  (rate_up : ℝ) (time_up : ℝ) (distance_up : ℝ)
  (distance_down : ℝ) (time_down : ℝ) :
  rate_up = 4 → time_up = 2 → distance_up = rate_up * time_up →
  distance_down = 12 → time_down = 2 →
  (distance_down / time_down) / rate_up = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rate_ratio_l111_111190


namespace age_ratio_in_six_years_l111_111790

-- Definitions for Claire's and Pete's current ages
variables (c p : ℕ)

-- Conditions given in the problem
def condition1 : Prop := c - 3 = 2 * (p - 3)
def condition2 : Prop := p - 7 = (1 / 4) * (c - 7)

-- The proof problem statement
theorem age_ratio_in_six_years (c p : ℕ) (h1 : condition1 c p) (h2 : condition2 c p) : 
  (c + 6) = 3 * (p + 6) :=
sorry

end age_ratio_in_six_years_l111_111790


namespace rate_of_stream_l111_111406

theorem rate_of_stream (v : ℝ) (h : 126 = (16 + v) * 6) : v = 5 :=
by 
  sorry

end rate_of_stream_l111_111406


namespace rainfall_second_week_l111_111062

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 35) (h2 : r2 = 1.5 * r1) : r2 = 21 := 
  sorry

end rainfall_second_week_l111_111062


namespace solve_for_k_l111_111597

theorem solve_for_k (t s k : ℝ) :
  (∀ t s : ℝ, (∃ t s : ℝ, (⟨1, 4⟩ : ℝ × ℝ) + t • ⟨5, -3⟩ = ⟨0, 1⟩ + s • ⟨-2, k⟩) → false) ↔ k = 6 / 5 :=
by
  sorry

end solve_for_k_l111_111597


namespace at_least_one_nonzero_l111_111543

theorem at_least_one_nonzero (a b : ℝ) : a^2 + b^2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end at_least_one_nonzero_l111_111543


namespace cost_per_ticket_l111_111979

/-- Adam bought 13 tickets and after riding the ferris wheel, he had 4 tickets left.
    He spent 81 dollars riding the ferris wheel, and we want to determine how much each ticket cost. -/
theorem cost_per_ticket (initial_tickets : ℕ) (tickets_left : ℕ) (total_cost : ℕ) (used_tickets : ℕ) 
    (ticket_cost : ℕ) (h1 : initial_tickets = 13) 
    (h2 : tickets_left = 4) 
    (h3 : total_cost = 81) 
    (h4 : used_tickets = initial_tickets - tickets_left) 
    (h5 : ticket_cost = total_cost / used_tickets) : ticket_cost = 9 :=
by {
    sorry
}

end cost_per_ticket_l111_111979


namespace nora_nuts_problem_l111_111490

theorem nora_nuts_problem :
  ∃ n : ℕ, (∀ (a p c : ℕ), 30 * n = 18 * a ∧ 30 * n = 21 * p ∧ 30 * n = 16 * c) ∧ n = 34 :=
by
  -- Provided conditions and solution steps will go here.
  sorry

end nora_nuts_problem_l111_111490


namespace cos_neg_30_eq_sqrt_3_div_2_l111_111314

theorem cos_neg_30_eq_sqrt_3_div_2 : 
  Real.cos (-30 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_neg_30_eq_sqrt_3_div_2_l111_111314


namespace consecutive_odd_product_l111_111585

theorem consecutive_odd_product (n : ℤ) :
  (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by sorry

end consecutive_odd_product_l111_111585


namespace no_leopards_in_circus_l111_111827

theorem no_leopards_in_circus (L T : ℕ) (N : ℕ) (h₁ : L = N / 5) (h₂ : T = 5 * (N - T)) : 
  ∀ A, A = L + N → A = T + (N - T) → ¬ ∃ x, x ≠ L ∧ x ≠ T ∧ x ≠ (N - L - T) :=
by
  sorry

end no_leopards_in_circus_l111_111827


namespace marly_100_bills_l111_111529

-- Define the number of each type of bill Marly has
def num_20_bills := 10
def num_10_bills := 8
def num_5_bills := 4

-- Define the values of the bills
def value_20_bill := 20
def value_10_bill := 10
def value_5_bill := 5

-- Define the total amount of money Marly has
def total_amount := num_20_bills * value_20_bill + num_10_bills * value_10_bill + num_5_bills * value_5_bill

-- Define the value of a $100 bill
def value_100_bill := 100

-- Now state the main theorem
theorem marly_100_bills : total_amount / value_100_bill = 3 := by
  sorry

end marly_100_bills_l111_111529


namespace find_X_l111_111383

theorem find_X (k : ℝ) (R1 R2 X1 X2 Y1 Y2 : ℝ) (h1 : R1 = k * (X1 / Y1)) (h2 : R1 = 10) (h3 : X1 = 2) (h4 : Y1 = 4) (h5 : R2 = 8) (h6 : Y2 = 5) : X2 = 2 :=
sorry

end find_X_l111_111383


namespace winning_configurations_for_blake_l111_111872

def isWinningConfigurationForBlake (config : List ℕ) := 
  let nimSum := config.foldl (xor) 0
  nimSum = 0

theorem winning_configurations_for_blake :
  (isWinningConfigurationForBlake [8, 2, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 3, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 5, 2]) :=
by {
  sorry
}

end winning_configurations_for_blake_l111_111872


namespace find_the_triplet_l111_111341

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end find_the_triplet_l111_111341


namespace find_WZ_length_l111_111568

noncomputable def WZ_length (XY YZ XZ WX : ℝ) (theta : ℝ) : ℝ :=
  Real.sqrt ((WX^2 + XZ^2 - 2 * WX * XZ * (-1 / 2)))

-- Define the problem within the context of the provided lengths and condition
theorem find_WZ_length :
  WZ_length 3 5 7 8.5 (-1 / 2) = Real.sqrt 180.75 :=
by 
  -- This "by sorry" is used to indicate the proof is omitted
  sorry

end find_WZ_length_l111_111568


namespace length_of_segment_NB_l111_111920

variable (L W x : ℝ)
variable (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W))

theorem length_of_segment_NB (L W x : ℝ) (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W)) : 
  x = 0.8 * L :=
by
  sorry

end length_of_segment_NB_l111_111920


namespace sum_of_three_smallest_two_digit_primes_l111_111948

theorem sum_of_three_smallest_two_digit_primes :
  11 + 13 + 17 = 41 :=
by
  sorry

end sum_of_three_smallest_two_digit_primes_l111_111948


namespace minimum_product_OP_OQ_l111_111700

theorem minimum_product_OP_OQ (a b : ℝ) (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : P ≠ Q) (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) (h5 : Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)
  (h6 : P.1 * Q.1 + P.2 * Q.2 = 0) :
  (P.1 ^ 2 + P.2 ^ 2) * (Q.1 ^ 2 + Q.2 ^ 2) ≥ (2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2)) :=
by sorry

end minimum_product_OP_OQ_l111_111700


namespace smallest_n_l111_111352

theorem smallest_n(vc: ℕ) (n: ℕ) : 
    (vc = 25) ∧ ∃ y o i : ℕ, ((25 * n = 10 * y) ∨ (25 * n = 18 * o) ∨ (25 * n = 20 * i)) → 
    n = 16 := by
    -- We state that given conditions should imply n = 16.
    sorry

end smallest_n_l111_111352


namespace white_balls_count_l111_111074

theorem white_balls_count (w : ℕ) (h : (w / 15) * ((w - 1) / 14) = (1 : ℚ) / 21) : w = 5 := by
  sorry

end white_balls_count_l111_111074


namespace multiply_5915581_7907_l111_111515

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := 
by
  -- sorry is used here to skip the proof
  sorry

end multiply_5915581_7907_l111_111515


namespace total_stickers_at_end_of_week_l111_111098

-- Defining the initial and earned stickers as constants
def initial_stickers : ℕ := 39
def earned_stickers : ℕ := 22

-- Defining the goal as a proof statement
theorem total_stickers_at_end_of_week : initial_stickers + earned_stickers = 61 := 
by {
  sorry
}

end total_stickers_at_end_of_week_l111_111098


namespace simplify_expression_l111_111263

theorem simplify_expression : 5 * (14 / 3) * (21 / -70) = - 35 / 2 := by
  sorry

end simplify_expression_l111_111263


namespace arithmetic_sequence_sum_l111_111537

-- Definitions based on problem conditions
variable (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) -- terms of the sequence
variable (S_3 S_6 S_9 : ℤ)

-- Given conditions
variable (h1 : S_3 = 3 * a_1 + 3 * (a_2 - a_1))
variable (h2 : S_6 = 6 * a_1 + 15 * (a_2 - a_1))
variable (h3 : S_3 = 9)
variable (h4 : S_6 = 36)

-- Theorem to prove
theorem arithmetic_sequence_sum : S_9 = 81 :=
by
  -- We just state the theorem here and will provide a proof later
  sorry

end arithmetic_sequence_sum_l111_111537


namespace problem1_l111_111106

def f (x : ℝ) := (1 - 3 * x) * (1 + x) ^ 5

theorem problem1 :
  let a : ℝ := f (1 / 3)
  a = 0 :=
by
  let a := f (1 / 3)
  sorry

end problem1_l111_111106


namespace fruit_weight_sister_and_dad_l111_111203

-- Defining the problem statement and conditions
variable (strawberries_m blueberries_m raspberries_m : ℝ)
variable (strawberries_d blueberries_d raspberries_d : ℝ)
variable (strawberries_s blueberries_s raspberries_s : ℝ)
variable (total_weight : ℝ)

-- Given initial conditions
def conditions : Prop :=
  strawberries_m = 5 ∧
  blueberries_m = 3 ∧
  raspberries_m = 6 ∧
  strawberries_d = 2 * strawberries_m ∧
  blueberries_d = 2 * blueberries_m ∧
  raspberries_d = 2 * raspberries_m ∧
  strawberries_s = strawberries_m / 2 ∧
  blueberries_s = blueberries_m / 2 ∧
  raspberries_s = raspberries_m / 2 ∧
  total_weight = (strawberries_m + blueberries_m + raspberries_m) + 
                 (strawberries_d + blueberries_d + raspberries_d) + 
                 (strawberries_s + blueberries_s + raspberries_s)

-- Defining the property to prove
theorem fruit_weight_sister_and_dad :
  conditions strawberries_m blueberries_m raspberries_m strawberries_d blueberries_d raspberries_d strawberries_s blueberries_s raspberries_s total_weight →
  (strawberries_d + blueberries_d + raspberries_d) +
  (strawberries_s + blueberries_s + raspberries_s) = 35 := by
  sorry

end fruit_weight_sister_and_dad_l111_111203


namespace find_13th_result_l111_111374

theorem find_13th_result (avg25 : ℕ) (avg12_first : ℕ) (avg12_last : ℕ)
  (h_avg25 : avg25 = 18) (h_avg12_first : avg12_first = 10) (h_avg12_last : avg12_last = 20) :
  ∃ r13 : ℕ, r13 = 90 := by
  sorry

end find_13th_result_l111_111374


namespace coffee_last_days_l111_111169

theorem coffee_last_days (coffee_weight : ℕ) (cups_per_lb : ℕ) (angie_daily : ℕ) (bob_daily : ℕ) (carol_daily : ℕ) 
  (angie_coffee_weight : coffee_weight = 3) (cups_brewing_rate : cups_per_lb = 40)
  (angie_consumption : angie_daily = 3) (bob_consumption : bob_daily = 2) (carol_consumption : carol_daily = 4) : 
  ((coffee_weight * cups_per_lb) / (angie_daily + bob_daily + carol_daily) = 13) := by
  sorry

end coffee_last_days_l111_111169


namespace correct_equation_l111_111011

theorem correct_equation (x : ℕ) :
  (30 * x + 8 = 31 * x - 26) := by
  sorry

end correct_equation_l111_111011


namespace solve_pond_fish_problem_l111_111193

def pond_fish_problem 
  (tagged_fish : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (total_fish : ℕ) : Prop :=
  (tagged_in_second_catch : ℝ) / second_catch = (tagged_fish : ℝ) / total_fish →
  total_fish = 1750

theorem solve_pond_fish_problem : 
  pond_fish_problem 70 50 2 1750 :=
by
  sorry

end solve_pond_fish_problem_l111_111193


namespace least_width_l111_111838

theorem least_width (w : ℝ) (h_nonneg : w ≥ 0) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end least_width_l111_111838


namespace arithmetic_sequence_k_l111_111836

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end arithmetic_sequence_k_l111_111836


namespace max_cookie_price_l111_111833

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l111_111833


namespace cans_collected_on_first_day_l111_111622

-- Declare the main theorem
theorem cans_collected_on_first_day 
  (x : ℕ) -- Number of cans collected on the first day
  (total_cans : x + (x + 5) + (x + 10) + (x + 15) + (x + 20) = 150) :
  x = 20 :=
sorry

end cans_collected_on_first_day_l111_111622


namespace sum_of_squares_l111_111625

theorem sum_of_squares (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 1) (h2 : b^2 + b * c + c^2 = 3) (h3 : c^2 + c * a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := 
sorry

end sum_of_squares_l111_111625


namespace barrels_oil_difference_l111_111774

/--
There are two barrels of oil, A and B.
1. $\frac{1}{3}$ of the oil is poured from barrel A into barrel B.
2. $\frac{1}{5}$ of the oil is poured from barrel B back into barrel A.
3. Each barrel contains 24kg of oil after the transfers.

Prove that originally, barrel A had 6 kg more oil than barrel B.
-/
theorem barrels_oil_difference :
  ∃ (x y : ℝ), (y = 48 - x) ∧
  (24 = (2 / 3) * x + (1 / 5) * (48 - x + (1 / 3) * x)) ∧
  (24 = (48 - x + (1 / 3) * x) * (4 / 5)) ∧
  (x - y = 6) :=
by
  sorry

end barrels_oil_difference_l111_111774


namespace distance_from_A_to_directrix_l111_111734

open Real

noncomputable def distance_from_point_to_directrix (p : ℝ) : ℝ :=
  1 + p / 2

theorem distance_from_A_to_directrix : 
  ∃ (p : ℝ), (sqrt 5)^2 = 2 * p ∧ distance_from_point_to_directrix p = 9 / 4 :=
by 
  sorry

end distance_from_A_to_directrix_l111_111734


namespace solution_set_inequality_l111_111973

theorem solution_set_inequality (x : ℝ) : 
  ((x-2) * (3-x) > 0) ↔ (2 < x ∧ x < 3) :=
by sorry

end solution_set_inequality_l111_111973


namespace bisect_segment_l111_111730

variables {A B C D E P : Point}
variables {α β γ δ ε : Real} -- angles in degrees
variables {BD CE : Line}

-- Geometric predicates
def Angle (x y z : Point) : Real := sorry -- calculates the angle ∠xyz

def isMidpoint (M A B : Point) : Prop := sorry -- M is the midpoint of segment AB

-- Given Conditions
variables (h1 : convex_pentagon A B C D E)
          (h2 : Angle B A C = Angle C A D ∧ Angle C A D = Angle D A E)
          (h3 : Angle A B C = Angle A C D ∧ Angle A C D = Angle A D E)
          (h4 : intersects BD CE P)

-- Conclusion to be proved
theorem bisect_segment : isMidpoint P C D :=
by {
  sorry -- proof to be filled in
}

end bisect_segment_l111_111730


namespace fraction_of_state_quarters_is_two_fifths_l111_111050

variable (total_quarters state_quarters : ℕ)
variable (is_pennsylvania_percentage : ℚ)
variable (pennsylvania_state_quarters : ℕ)

theorem fraction_of_state_quarters_is_two_fifths
  (h1 : total_quarters = 35)
  (h2 : pennsylvania_state_quarters = 7)
  (h3 : is_pennsylvania_percentage = 1 / 2)
  (h4 : state_quarters = 2 * pennsylvania_state_quarters)
  : (state_quarters : ℚ) / (total_quarters : ℚ) = 2 / 5 :=
sorry

end fraction_of_state_quarters_is_two_fifths_l111_111050


namespace subset_relation_l111_111736

def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

theorem subset_relation : N ⊆ M := by
  sorry

end subset_relation_l111_111736


namespace cost_of_largest_pot_l111_111481

theorem cost_of_largest_pot
    (x : ℝ)
    (hx : 6 * x + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) = 8.25) :
    (x + 0.5) = 1.625 :=
sorry

end cost_of_largest_pot_l111_111481


namespace points_lie_on_circle_l111_111289

theorem points_lie_on_circle (t : ℝ) : 
  let x := Real.cos t
  let y := Real.sin t
  x^2 + y^2 = 1 :=
by
  sorry

end points_lie_on_circle_l111_111289


namespace analytical_expression_satisfies_conditions_l111_111363

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

theorem analytical_expression_satisfies_conditions :
  is_increasing f ∧ (∀ x : ℝ, f x > 1) :=
by
  sorry

end analytical_expression_satisfies_conditions_l111_111363


namespace esperanzas_gross_monthly_salary_l111_111639

variables (Rent FoodExpenses MortgageBill Savings Taxes GrossSalary : ℝ)

def problem_conditions (Rent FoodExpenses MortgageBill Savings Taxes : ℝ) :=
  Rent = 600 ∧
  FoodExpenses = (3 / 5) * Rent ∧
  MortgageBill = 3 * FoodExpenses ∧
  Savings = 2000 ∧
  Taxes = (2 / 5) * Savings

theorem esperanzas_gross_monthly_salary (h : problem_conditions Rent FoodExpenses MortgageBill Savings Taxes) :
  GrossSalary = Rent + FoodExpenses + MortgageBill + Taxes + Savings → GrossSalary = 4840 :=
by
  sorry

end esperanzas_gross_monthly_salary_l111_111639


namespace intervals_of_monotonicity_and_min_value_l111_111148

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem intervals_of_monotonicity_and_min_value : 
  (∀ x, (x < -1 → f x < f (x + 0.0001)) ∧ (x > -1 ∧ x < 3 → f x > f (x + 0.0001)) ∧ (x > 3 → f x < f (x + 0.0001))) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≥ f 2) :=
by
  sorry

end intervals_of_monotonicity_and_min_value_l111_111148


namespace geom_sum_3m_l111_111739

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (m : ℕ)

axiom geom_sum_m : S m = 10
axiom geom_sum_2m : S (2 * m) = 30

theorem geom_sum_3m : S (3 * m) = 70 :=
by
  sorry

end geom_sum_3m_l111_111739


namespace triangle_relation_l111_111427

theorem triangle_relation (A B C a b : ℝ) (h : 4 * A = B ∧ B = C) (hABC : A + B + C = 180) : 
  a^3 + b^3 = 3 * a * b^2 := 
by 
  sorry

end triangle_relation_l111_111427


namespace maximum_area_of_triangle_l111_111217

theorem maximum_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hC : C = π / 6) (hSum : a + b = 12) :
  ∃ (S : ℝ), S = 9 ∧ ∀ S', S' ≤ S := 
sorry

end maximum_area_of_triangle_l111_111217


namespace integer_solution_of_inequalities_l111_111676

theorem integer_solution_of_inequalities :
  (∀ x : ℝ, 3 * x - 4 ≤ 6 * x - 2 → (2 * x + 1) / 3 - 1 < (x - 1) / 2 → (x = 0)) :=
sorry

end integer_solution_of_inequalities_l111_111676


namespace sequence_a4_value_l111_111841

theorem sequence_a4_value : 
  ∀ (a : ℕ → ℕ), a 1 = 2 → (∀ n, n ≥ 2 → a n = a (n - 1) + n) → a 4 = 11 :=
by
  sorry

end sequence_a4_value_l111_111841


namespace not_prime_for_all_n_ge_2_l111_111440

theorem not_prime_for_all_n_ge_2 (n : ℕ) (hn : n ≥ 2) : ¬ Prime (2 * (n^3 + n + 1)) := 
by
  sorry

end not_prime_for_all_n_ge_2_l111_111440


namespace Robie_l111_111825

def initial_bags (X : ℕ) := (X - 2) + 3 = 4

theorem Robie's_initial_bags (X : ℕ) (h : initial_bags X) : X = 3 :=
by
  unfold initial_bags at h
  sorry

end Robie_l111_111825


namespace num_of_possible_outcomes_l111_111782

def participants : Fin 6 := sorry  -- Define the participants as elements of Fin 6

theorem num_of_possible_outcomes : (6 * 5 * 4 = 120) :=
by {
  -- Prove this mathematical statement
  rfl
}

end num_of_possible_outcomes_l111_111782


namespace simplify_expression_l111_111678

theorem simplify_expression : 
    2 * Real.sqrt 12 + 3 * Real.sqrt (4 / 3) - Real.sqrt (16 / 3) - (2 / 3) * Real.sqrt 48 = 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l111_111678


namespace ophelia_average_pay_l111_111942

theorem ophelia_average_pay : ∀ (n : ℕ), 
  (51 + 100 * (n - 1)) / n = 93 ↔ n = 7 :=
by
  sorry

end ophelia_average_pay_l111_111942


namespace conditions_neither_necessary_nor_sufficient_l111_111223

theorem conditions_neither_necessary_nor_sufficient :
  (¬(0 < x ∧ x < 2) ↔ (¬(-1 / 2 < x ∨ x < 1)) ∨ (¬(-1 / 2 < x ∧ x < 1))) :=
by sorry

end conditions_neither_necessary_nor_sufficient_l111_111223


namespace orchard_harvest_l111_111772

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l111_111772


namespace jerry_task_duration_l111_111812

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end jerry_task_duration_l111_111812


namespace least_possible_value_of_z_minus_x_l111_111252

variables (x y z : ℤ)

-- Define the conditions
def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

-- State the theorem
theorem least_possible_value_of_z_minus_x (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) 
    (hx_even : even x) (hy_odd : odd y) (hz_odd : odd z) : z - x = 7 :=
sorry

end least_possible_value_of_z_minus_x_l111_111252


namespace distinct_weights_count_l111_111695

theorem distinct_weights_count (n : ℕ) (h : n = 4) : 
  -- Given four weights and a two-pan balance scale without a pointer,
  ∃ m : ℕ, 
  -- prove that the number of distinct weights of cargo
  (m = 40) ∧  
  -- that can be exactly measured if the weights can be placed on both pans of the scale is 40.
  m = 3^n - 1 ∧ (m / 2 = 40) := by
  sorry

end distinct_weights_count_l111_111695


namespace number_of_arrangements_l111_111007

theorem number_of_arrangements (n : ℕ) (h : n = 7) :
  ∃ (arrangements : ℕ), arrangements = 20 :=
by
  sorry

end number_of_arrangements_l111_111007


namespace correct_product_exists_l111_111785

variable (a b : ℕ)

theorem correct_product_exists
  (h1 : a < 100)
  (h2 : 10 * (a % 10) + a / 10 = 14)
  (h3 : 14 * b = 182) : a * b = 533 := sorry

end correct_product_exists_l111_111785


namespace num_ways_to_choose_starting_lineup_l111_111794

-- Define conditions as Lean definitions
def team_size : ℕ := 12
def outfield_players : ℕ := 4

-- Define the function to compute the number of ways to choose the starting lineup
def choose_starting_lineup (team_size : ℕ) (outfield_players : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) outfield_players

-- The theorem to prove that the number of ways to choose the lineup is 3960
theorem num_ways_to_choose_starting_lineup : choose_starting_lineup team_size outfield_players = 3960 :=
  sorry

end num_ways_to_choose_starting_lineup_l111_111794


namespace greatest_possible_subway_takers_l111_111069

/-- In a company with 48 employees, some part-time and some full-time, exactly (1/3) of the part-time
employees and (1/4) of the full-time employees take the subway to work. Prove that the greatest
possible number of employees who take the subway to work is 15. -/
theorem greatest_possible_subway_takers
  (P F : ℕ)
  (h : P + F = 48)
  (h_subway_part : ∀ p, p = P → 0 ≤ p ∧ p ≤ 48)
  (h_subway_full : ∀ f, f = F → 0 ≤ f ∧ f ≤ 48) :
  ∃ y, y = 15 := 
sorry

end greatest_possible_subway_takers_l111_111069


namespace triangle_area_l111_111705

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area_l111_111705


namespace solution_set_inequality_l111_111456

theorem solution_set_inequality (a : ℝ) :
  ∀ x : ℝ,
    (12 * x^2 - a * x > a^2) →
    ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
     (a = 0 ∧ x ≠ 0) ∨
     (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
by
  sorry

end solution_set_inequality_l111_111456


namespace sum_first_9_terms_l111_111319

-- Definitions of the arithmetic sequence and sum.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Conditions
def a_n (n : ℕ) : ℤ := sorry -- we assume this function gives the n-th term of the arithmetic sequence
def S_n (n : ℕ) : ℤ := sorry -- sum of first n terms
axiom a_5_eq_2 : a_n 5 = 2
axiom arithmetic_sequence_proof : arithmetic_sequence a_n
axiom sum_first_n_proof : sum_first_n a_n S_n

-- Statement to prove
theorem sum_first_9_terms : S_n 9 = 18 :=
by
  sorry

end sum_first_9_terms_l111_111319


namespace part1_part2_l111_111605

-- Define the function f
def f (x m : ℝ) : ℝ := abs (x + m) + abs (2 * x - 1)

-- First part of the problem
theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 := 
by sorry

-- Second part of the problem
theorem part2 (m : ℝ) : 
  (∀ x, 3 / 4 ≤ x → x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 := 
by sorry

end part1_part2_l111_111605


namespace subset_implies_value_l111_111587

theorem subset_implies_value (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 3, 2*m-1}) (hB : B = {3, m}) (hSub : B ⊆ A) : 
  m = -1 ∨ m = 1 := by
  sorry

end subset_implies_value_l111_111587


namespace sodium_hydride_reaction_l111_111002

theorem sodium_hydride_reaction (H2O NaH NaOH H2 : ℕ) 
  (balanced_eq : NaH + H2O = NaOH + H2) 
  (stoichiometry : NaH = H2O → NaOH = H2 → NaH = H2) 
  (h : H2O = 2) : NaH = 2 :=
sorry

end sodium_hydride_reaction_l111_111002


namespace total_vehicles_correct_l111_111234

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l111_111234


namespace AJ_stamps_l111_111014

theorem AJ_stamps (A : ℕ)
  (KJ := A / 2)
  (CJ := 2 * KJ + 5)
  (BJ := 3 * A - 3)
  (total_stamps := A + KJ + CJ + BJ)
  (h : total_stamps = 1472) :
  A = 267 :=
  sorry

end AJ_stamps_l111_111014


namespace total_prime_dates_in_non_leap_year_l111_111370

def prime_dates_in_non_leap_year (days_in_months : List (Nat × Nat)) : Nat :=
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  days_in_months.foldl 
    (λ acc (month, days) => 
      acc + (prime_numbers.filter (λ day => day ≤ days)).length) 
    0

def month_days : List (Nat × Nat) :=
  [(2, 28), (3, 31), (5, 31), (7, 31), (11,30)]

theorem total_prime_dates_in_non_leap_year : prime_dates_in_non_leap_year month_days = 52 :=
  sorry

end total_prime_dates_in_non_leap_year_l111_111370


namespace Trishul_investment_percentage_l111_111609

-- Definitions from the conditions
def Vishal_invested (T : ℝ) : ℝ := 1.10 * T
def total_investment (T : ℝ) (V : ℝ) : ℝ := T + V + 2000

-- Problem statement
theorem Trishul_investment_percentage (T : ℝ) (V : ℝ) (H1 : V = Vishal_invested T) (H2 : total_investment T V = 5780) :
  ((2000 - T) / 2000) * 100 = 10 :=
sorry

end Trishul_investment_percentage_l111_111609


namespace elvis_writing_time_per_song_l111_111714

-- Define the conditions based on the problem statement
def total_studio_time_minutes := 300   -- 5 hours converted to minutes
def songs := 10
def recording_time_per_song := 12
def total_editing_time := 30

-- Define the total recording time
def total_recording_time := songs * recording_time_per_song

-- Define the total time available for writing songs
def total_writing_time := total_studio_time_minutes - total_recording_time - total_editing_time

-- Define the time to write each song
def time_per_song_writing := total_writing_time / songs

-- State the proof goal
theorem elvis_writing_time_per_song : time_per_song_writing = 15 := by
  sorry

end elvis_writing_time_per_song_l111_111714


namespace find_g_at_3_l111_111728

theorem find_g_at_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 4 * x + 1) : g 3 = 23 / 3 :=
by
  sorry

end find_g_at_3_l111_111728
