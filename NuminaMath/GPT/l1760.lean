import Mathlib

namespace NUMINAMATH_GPT_rectangle_width_length_ratio_l1760_176077

theorem rectangle_width_length_ratio (w l P : ℕ) (hP : P = 30) (hl : l = 10) (h_perimeter : P = 2*l + 2*w) :
  w / l = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_length_ratio_l1760_176077


namespace NUMINAMATH_GPT_chicken_feathers_after_crossing_l1760_176024

def feathers_remaining_after_crossings (cars_dodged feathers_before pulling_factor : ℕ) : ℕ :=
  let feathers_lost := cars_dodged * pulling_factor
  feathers_before - feathers_lost

theorem chicken_feathers_after_crossing 
  (cars_dodged : ℕ := 23)
  (feathers_before : ℕ := 5263)
  (pulling_factor : ℕ := 2) :
  feathers_remaining_after_crossings cars_dodged feathers_before pulling_factor = 5217 :=
by
  sorry

end NUMINAMATH_GPT_chicken_feathers_after_crossing_l1760_176024


namespace NUMINAMATH_GPT_Robie_chocolates_left_l1760_176060

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end NUMINAMATH_GPT_Robie_chocolates_left_l1760_176060


namespace NUMINAMATH_GPT_total_container_weight_is_correct_l1760_176061

-- Definitions based on the conditions
def copper_bar_weight : ℕ := 90
def steel_bar_weight : ℕ := copper_bar_weight + 20
def tin_bar_weight : ℕ := steel_bar_weight / 2
def aluminum_bar_weight : ℕ := tin_bar_weight + 10

-- Number of bars in the container
def count_steel_bars : ℕ := 10
def count_tin_bars : ℕ := 15
def count_copper_bars : ℕ := 12
def count_aluminum_bars : ℕ := 8

-- Total weight of each type of bar
def total_steel_weight : ℕ := count_steel_bars * steel_bar_weight
def total_tin_weight : ℕ := count_tin_bars * tin_bar_weight
def total_copper_weight : ℕ := count_copper_bars * copper_bar_weight
def total_aluminum_weight : ℕ := count_aluminum_bars * aluminum_bar_weight

-- Total weight of the container
def total_container_weight : ℕ := total_steel_weight + total_tin_weight + total_copper_weight + total_aluminum_weight

-- Theorem to prove
theorem total_container_weight_is_correct : total_container_weight = 3525 := by
  sorry

end NUMINAMATH_GPT_total_container_weight_is_correct_l1760_176061


namespace NUMINAMATH_GPT_number_of_functions_l1760_176042

open Nat

theorem number_of_functions (f : Fin 15 → Fin 15)
  (h : ∀ x, (f (f x) - 2 * f x + x : Int) % 15 = 0) :
  ∃! n : Nat, n = 375 := sorry

end NUMINAMATH_GPT_number_of_functions_l1760_176042


namespace NUMINAMATH_GPT_volunteer_org_percentage_change_l1760_176069

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end NUMINAMATH_GPT_volunteer_org_percentage_change_l1760_176069


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1760_176032

theorem recurring_decimal_to_fraction :
  let x := 0.4 + 67 / (99 : ℝ)
  (∀ y : ℝ, y = x ↔ y = 463 / 990) := 
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1760_176032


namespace NUMINAMATH_GPT_opposite_of_2023_l1760_176040

def opposite (n : Int) : Int := -n

theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l1760_176040


namespace NUMINAMATH_GPT_alice_savings_l1760_176081

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end NUMINAMATH_GPT_alice_savings_l1760_176081


namespace NUMINAMATH_GPT_rebus_solution_l1760_176043

-- We state the conditions:
variables (A B Γ D : ℤ)

-- Define the correct values
def A_correct := 2
def B_correct := 7
def Γ_correct := 1
def D_correct := 0

-- State the conditions as assumptions
axiom cond1 : A * B + 8 = 3 * B
axiom cond2 : Γ * D + B = 5  -- Adjusted assuming V = 5 from problem data
axiom cond3 : Γ * B + 3 = A * D

-- State the goal to be proved
theorem rebus_solution : A = A_correct ∧ B = B_correct ∧ Γ = Γ_correct ∧ D = D_correct :=
by
  sorry

end NUMINAMATH_GPT_rebus_solution_l1760_176043


namespace NUMINAMATH_GPT_find_angle_B_l1760_176002

open Real

theorem find_angle_B (A B : ℝ) 
  (h1 : 0 < B ∧ B < A ∧ A < π/2)
  (h2 : cos A = 1/7) 
  (h3 : cos (A - B) = 13/14) : 
  B = π/3 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l1760_176002


namespace NUMINAMATH_GPT_draw_odds_l1760_176071

theorem draw_odds (x : ℝ) (bet_Zubilo bet_Shaiba bet_Draw payout : ℝ) (h1 : bet_Zubilo = 3 * x) (h2 : bet_Shaiba = 2 * x) (h3 : payout = 6 * x) : 
  bet_Draw * 6 = payout :=
by
  sorry

end NUMINAMATH_GPT_draw_odds_l1760_176071


namespace NUMINAMATH_GPT_least_number_to_subtract_l1760_176097

theorem least_number_to_subtract (x : ℕ) :
  (2590 - x) % 9 = 6 ∧ 
  (2590 - x) % 11 = 6 ∧ 
  (2590 - x) % 13 = 6 ↔ 
  x = 16 := 
sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1760_176097


namespace NUMINAMATH_GPT_range_of_a_l1760_176051

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h_poly: ∀ x, x * x + (a * a - 1) * x + (a - 2) = 0 → x = x1 ∨ x = x2)
  (h_order: x1 < 1 ∧ 1 < x2) : 
  -2 < a ∧ a < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1760_176051


namespace NUMINAMATH_GPT_sum_faces_edges_vertices_eq_26_l1760_176068

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_sum_faces_edges_vertices_eq_26_l1760_176068


namespace NUMINAMATH_GPT_plane_boat_ratio_l1760_176034

theorem plane_boat_ratio (P B : ℕ) (h1 : P > B) (h2 : B ≤ 2) (h3 : P + B = 10) : P = 8 ∧ B = 2 ∧ P / B = 4 := by
  sorry

end NUMINAMATH_GPT_plane_boat_ratio_l1760_176034


namespace NUMINAMATH_GPT_trig_eq_solution_l1760_176052

open Real

theorem trig_eq_solution (x : ℝ) : 
  (cos (7 * x) + cos (3 * x) + sin (7 * x) - sin (3 * x) + sqrt 2 * cos (4 * x) = 0) ↔ 
  (∃ k : ℤ, 
    (x = -π / 8 + π * k / 2) ∨ 
    (x = -π / 4 + 2 * π * k / 3) ∨ 
    (x = 3 * π / 28 + 2 * π * k / 7)) :=
by sorry

end NUMINAMATH_GPT_trig_eq_solution_l1760_176052


namespace NUMINAMATH_GPT_find_room_length_l1760_176079

variable (w : ℝ) (C : ℝ) (r : ℝ)

theorem find_room_length (h_w : w = 4.75) (h_C : C = 29925) (h_r : r = 900) : (C / r) / w = 7 := by
  sorry

end NUMINAMATH_GPT_find_room_length_l1760_176079


namespace NUMINAMATH_GPT_determine_a_zeros_l1760_176006

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x = 3 then a else 2 / |x - 3|

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

theorem determine_a_zeros (a : ℝ) : (∃ c d, c ≠ 3 ∧ d ≠ 3 ∧ c ≠ d ∧ y c a = 0 ∧ y d a = 0 ∧ y 3 a = 0) → a = 4 :=
sorry

end NUMINAMATH_GPT_determine_a_zeros_l1760_176006


namespace NUMINAMATH_GPT_solve_for_x_l1760_176050

theorem solve_for_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1760_176050


namespace NUMINAMATH_GPT_allocate_teaching_positions_l1760_176038

theorem allocate_teaching_positions :
  ∃ (ways : ℕ), ways = 10 ∧ 
    (∃ (a b c : ℕ), a + b + c = 8 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 2 ≤ a) := 
sorry

end NUMINAMATH_GPT_allocate_teaching_positions_l1760_176038


namespace NUMINAMATH_GPT_count_subsets_l1760_176021

theorem count_subsets (S T : Set ℕ) (h1 : S = {1, 2, 3}) (h2 : T = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ n : ℕ, n = 16 ∧ ∀ X, S ⊆ X ∧ X ⊆ T ↔ X ∈ { X | ∃ m : ℕ, m = 16 }) := 
sorry

end NUMINAMATH_GPT_count_subsets_l1760_176021


namespace NUMINAMATH_GPT_simplify_expression_l1760_176055

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : 0 < b)

theorem simplify_expression : a ^ Real.log (1 / b ^ Real.log a) = 1 / b ^ (Real.log a) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1760_176055


namespace NUMINAMATH_GPT_first_day_exceeds_200_l1760_176031

-- Bacteria population doubling function
def bacteria_population (n : ℕ) : ℕ := 4 * 3 ^ n

-- Prove the smallest day where bacteria count exceeds 200 is 4
theorem first_day_exceeds_200 : ∃ n : ℕ, bacteria_population n > 200 ∧ ∀ m < n, bacteria_population m ≤ 200 :=
by 
    -- Proof will be filled here
    sorry

end NUMINAMATH_GPT_first_day_exceeds_200_l1760_176031


namespace NUMINAMATH_GPT_find_s_2_l1760_176018

def t (x : ℝ) : ℝ := 4 * x - 6
def s (y : ℝ) : ℝ := y^2 + 5 * y - 7

theorem find_s_2 : s 2 = 7 := by
  sorry

end NUMINAMATH_GPT_find_s_2_l1760_176018


namespace NUMINAMATH_GPT_find_angle_A_max_perimeter_incircle_l1760_176041

-- Definition of the triangle and the conditions
variables {A B C : Real} {a b c : Real} 

-- The conditions given in the problem
def triangle_conditions (a b c A B C : Real) : Prop :=
  (b + c = a * (Real.cos C + Real.sqrt 3 * Real.sin C)) ∧
  A + B + C = Real.pi

-- Part 1: Prove the value of angle A
theorem find_angle_A (a b c A B C : Real) 
(h : triangle_conditions a b c A B C) : 
A = Real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter of the incircle when a=2
theorem max_perimeter_incircle (b c A B C : Real) 
(h : triangle_conditions 2 b c A B C) : 
2 * Real.pi * (Real.sqrt 3 / 6 * (b + c - 2)) ≤ (2 * Real.sqrt 3 / 3) * Real.pi := sorry

end NUMINAMATH_GPT_find_angle_A_max_perimeter_incircle_l1760_176041


namespace NUMINAMATH_GPT_min_value_of_reciprocals_l1760_176099

theorem min_value_of_reciprocals (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) : 
  (1 / m) + (1 / n) = 2 :=
by
  -- the proof needs to be completed here.
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocals_l1760_176099


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1760_176073

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (3 / (a + b + c)^2) ≥ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1760_176073


namespace NUMINAMATH_GPT_ratio_tends_to_zero_as_n_tends_to_infinity_l1760_176056

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  -- Function to find the smallest prime not dividing n
  sorry

theorem ratio_tends_to_zero_as_n_tends_to_infinity :
  ∀ ε > 0, ∃ N, ∀ n > N, (smallest_prime_not_dividing n : ℝ) / (n : ℝ) < ε := by
  sorry

end NUMINAMATH_GPT_ratio_tends_to_zero_as_n_tends_to_infinity_l1760_176056


namespace NUMINAMATH_GPT_find_m_l1760_176062

theorem find_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l1760_176062


namespace NUMINAMATH_GPT_road_path_distance_l1760_176028

theorem road_path_distance (d_AB d_AC d_BC d_BD : ℕ) 
  (h1 : d_AB = 9) (h2 : d_AC = 13) (h3 : d_BC = 8) (h4 : d_BD = 14) : A_to_D = 19 :=
by
  sorry

end NUMINAMATH_GPT_road_path_distance_l1760_176028


namespace NUMINAMATH_GPT_prob_A_winning_l1760_176085

variable (P_draw P_B : ℚ)

def P_A_winning := 1 - P_draw - P_B

theorem prob_A_winning (h1 : P_draw = 1 / 2) (h2 : P_B = 1 / 3) :
  P_A_winning P_draw P_B = 1 / 6 :=
by
  rw [P_A_winning, h1, h2]
  norm_num
  done

end NUMINAMATH_GPT_prob_A_winning_l1760_176085


namespace NUMINAMATH_GPT_remainder_of_expression_l1760_176098

theorem remainder_of_expression (n : ℤ) (h : n % 60 = 1) : (n^2 + 2 * n + 3) % 60 = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_expression_l1760_176098


namespace NUMINAMATH_GPT_number_of_children_per_seat_l1760_176039

variable (children : ℕ) (seats : ℕ)

theorem number_of_children_per_seat (h1 : children = 58) (h2 : seats = 29) :
  children / seats = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_children_per_seat_l1760_176039


namespace NUMINAMATH_GPT_opposite_of_2023_l1760_176030

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l1760_176030


namespace NUMINAMATH_GPT_find_excluded_number_l1760_176091

-- Definition of the problem conditions
def avg (nums : List ℕ) : ℕ := (nums.sum / nums.length)

-- Problem condition: the average of 5 numbers is 27
def condition1 (nums : List ℕ) : Prop :=
  nums.length = 5 ∧ avg nums = 27

-- Problem condition: excluding one number, the average of remaining 4 numbers is 25
def condition2 (nums : List ℕ) (x : ℕ) : Prop :=
  let nums' := nums.filter (λ n => n ≠ x)
  nums.length = 5 ∧ nums'.length = 4 ∧ avg nums' = 25

-- Proof statement: finding the excluded number
theorem find_excluded_number (nums : List ℕ) (x : ℕ) (h1 : condition1 nums) (h2 : condition2 nums x) : x = 35 := 
by
  sorry

end NUMINAMATH_GPT_find_excluded_number_l1760_176091


namespace NUMINAMATH_GPT_jim_gas_tank_capacity_l1760_176037

/-- Jim has 2/3 of a tank left after a round-trip of 20 miles where he gets 5 miles per gallon.
    Prove that the capacity of Jim's gas tank is 12 gallons. --/
theorem jim_gas_tank_capacity
    (remaining_fraction : ℚ)
    (round_trip_distance : ℚ)
    (fuel_efficiency : ℚ)
    (used_fraction : ℚ)
    (used_gallons : ℚ)
    (total_capacity : ℚ)
    (h1 : remaining_fraction = 2/3)
    (h2 : round_trip_distance = 20)
    (h3 : fuel_efficiency = 5)
    (h4 : used_fraction = 1 - remaining_fraction)
    (h5 : used_gallons = round_trip_distance / fuel_efficiency)
    (h6 : used_gallons = used_fraction * total_capacity) :
  total_capacity = 12 :=
sorry

end NUMINAMATH_GPT_jim_gas_tank_capacity_l1760_176037


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1760_176025

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1760_176025


namespace NUMINAMATH_GPT_logical_equivalence_l1760_176059

variables (P Q : Prop)

theorem logical_equivalence :
  (¬P → ¬Q) ↔ (Q → P) :=
sorry

end NUMINAMATH_GPT_logical_equivalence_l1760_176059


namespace NUMINAMATH_GPT_john_marks_wrongly_entered_as_l1760_176046

-- Definitions based on the conditions
def john_correct_marks : ℤ := 62
def num_students : ℤ := 80
def avg_increase : ℤ := 1/2
def total_increase : ℤ := num_students * avg_increase

-- Statement to prove
theorem john_marks_wrongly_entered_as (x : ℤ) :
  (total_increase = (x - john_correct_marks)) → x = 102 :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_john_marks_wrongly_entered_as_l1760_176046


namespace NUMINAMATH_GPT_product_formula_l1760_176065

theorem product_formula :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) *
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) *
  (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end NUMINAMATH_GPT_product_formula_l1760_176065


namespace NUMINAMATH_GPT_baseball_team_earnings_l1760_176094

theorem baseball_team_earnings (S : ℝ) (W : ℝ) (Total : ℝ) 
    (h1 : S = 2662.50) 
    (h2 : W = S - 142.50) 
    (h3 : Total = W + S) : 
  Total = 5182.50 :=
sorry

end NUMINAMATH_GPT_baseball_team_earnings_l1760_176094


namespace NUMINAMATH_GPT_evaluate_expression_l1760_176000

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1760_176000


namespace NUMINAMATH_GPT_positive_root_of_cubic_eq_l1760_176076

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 3 * x^2 - x - Real.sqrt 2 = 0 ∧ x = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_positive_root_of_cubic_eq_l1760_176076


namespace NUMINAMATH_GPT_value_of_x_l1760_176007

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1760_176007


namespace NUMINAMATH_GPT_savings_account_amount_l1760_176057

-- Definitions and conditions from the problem
def checking_account_yen : ℕ := 6359
def total_yen : ℕ := 9844

-- Question we aim to prove - the amount in the savings account
def savings_account_yen : ℕ := total_yen - checking_account_yen

-- Lean statement to prove the equality
theorem savings_account_amount : savings_account_yen = 3485 :=
by
  sorry

end NUMINAMATH_GPT_savings_account_amount_l1760_176057


namespace NUMINAMATH_GPT_sequence_third_order_and_nth_term_l1760_176086

-- Define the given sequence
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 6
  | 2 => 13
  | 3 => 27
  | 4 => 50
  | 5 => 84
  | _ => sorry -- let’s define the general form for other terms later

-- Define first differences
def first_diff (n : ℕ) : ℤ := a (n + 1) - a n

-- Define second differences
def second_diff (n : ℕ) : ℤ := first_diff (n + 1) - first_diff n

-- Define third differences
def third_diff (n : ℕ) : ℤ := second_diff (n + 1) - second_diff n

-- Define the nth term formula
noncomputable def nth_term (n : ℕ) : ℚ := (1 / 6) * (2 * n^3 + 3 * n^2 - 11 * n + 30)

-- Theorem stating the least possible order is 3 and the nth term formula
theorem sequence_third_order_and_nth_term :
  (∀ n, third_diff n = 2) ∧ (∀ n, a n = nth_term n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_third_order_and_nth_term_l1760_176086


namespace NUMINAMATH_GPT_min_AB_DE_l1760_176088

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem min_AB_DE 
(F : (ℝ × ℝ)) 
(A B D E : ℝ × ℝ) 
(k1 k2 : ℝ) 
(hF : F = (1, 0)) 
(hk : k1^2 + k2^2 = 1) 
(hAB : ∀ x y, parabola x y → line_through_focus k1 x y → A = (x, y) ∨ B = (x, y)) 
(hDE : ∀ x y, parabola x y → line_through_focus k2 x y → D = (x, y) ∨ E = (x, y)) 
: |(A.1 - B.1)| + |(D.1 - E.1)| ≥ 24 := 
sorry

end NUMINAMATH_GPT_min_AB_DE_l1760_176088


namespace NUMINAMATH_GPT_boys_girls_students_l1760_176092

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_boys_girls_students_l1760_176092


namespace NUMINAMATH_GPT_speed_of_current_is_6_l1760_176010

noncomputable def speed_of_current : ℝ :=
  let Vm := 18  -- speed in still water in kmph
  let distance_m := 100  -- distance covered in meters
  let time_s := 14.998800095992323  -- time taken in seconds
  let distance_km := distance_m / 1000  -- converting distance to kilometers
  let time_h := time_s / 3600  -- converting time to hours
  let Vd := distance_km / time_h  -- speed downstream in kmph
  Vd - Vm  -- speed of the current

theorem speed_of_current_is_6 :
  speed_of_current = 6 := by
  sorry -- proof is skipped

end NUMINAMATH_GPT_speed_of_current_is_6_l1760_176010


namespace NUMINAMATH_GPT_complete_the_square_l1760_176004

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x + 5 = 0 ↔ (x - 4)^2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1760_176004


namespace NUMINAMATH_GPT_channel_bottom_width_l1760_176005

theorem channel_bottom_width
  (area : ℝ)
  (top_width : ℝ)
  (depth : ℝ)
  (h_area : area = 880)
  (h_top_width : top_width = 14)
  (h_depth : depth = 80) :
  ∃ (b : ℝ), b = 8 ∧ area = (1/2) * (top_width + b) * depth := 
by
  sorry

end NUMINAMATH_GPT_channel_bottom_width_l1760_176005


namespace NUMINAMATH_GPT_river_current_speed_l1760_176063

variable (c : ℝ)

def boat_speed_still_water : ℝ := 20
def round_trip_distance : ℝ := 182
def round_trip_time : ℝ := 10

theorem river_current_speed (h : (91 / (boat_speed_still_water - c)) + (91 / (boat_speed_still_water + c)) = round_trip_time) : c = 6 :=
sorry

end NUMINAMATH_GPT_river_current_speed_l1760_176063


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1760_176044

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 7 * x + 6 = 0 ↔ x = 1 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1760_176044


namespace NUMINAMATH_GPT_min_area_circle_tangent_l1760_176022

theorem min_area_circle_tangent (h : ∀ (x : ℝ), x > 0 → y = 2 / x) : 
  ∃ (a b r : ℝ), (∀ (x : ℝ), x > 0 → 2 * a + b = 2 + 2 / x) ∧
  (∀ (x : ℝ), x > 0 → (x - 1)^2 + (y - 2)^2 = 5) :=
sorry

end NUMINAMATH_GPT_min_area_circle_tangent_l1760_176022


namespace NUMINAMATH_GPT_rhombus_area_l1760_176093

theorem rhombus_area (x y : ℝ) (h : |x - 1| + |y - 1| = 1) : 
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1760_176093


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l1760_176053

theorem parallel_lines_slope_eq {a : ℝ} : (∀ x : ℝ, 2*x - 1 = a*x + 1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l1760_176053


namespace NUMINAMATH_GPT_equilateral_triangle_l1760_176029

theorem equilateral_triangle (a b c : ℝ) (h1 : b^2 = a * c) (h2 : 2 * b = a + c) : a = b ∧ b = c ∧ a = c := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_l1760_176029


namespace NUMINAMATH_GPT_find_m_l1760_176089

noncomputable def inverse_proportion (x : ℝ) : ℝ := 4 / x

theorem find_m (m n : ℝ) (h1 : ∀ x, -4 ≤ x ∧ x ≤ m → inverse_proportion x = 4 / x ∧ n ≤ inverse_proportion x ∧ inverse_proportion x ≤ n + 3) :
  m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1760_176089


namespace NUMINAMATH_GPT_son_l1760_176083

def woman's_age (W S : ℕ) : Prop := W = 2 * S + 3
def sum_of_ages (W S : ℕ) : Prop := W + S = 84

theorem son's_age_is_27 (W S : ℕ) (h1: woman's_age W S) (h2: sum_of_ages W S) : S = 27 :=
by
  sorry

end NUMINAMATH_GPT_son_l1760_176083


namespace NUMINAMATH_GPT_system_of_equations_solution_l1760_176013

theorem system_of_equations_solution (x y z : ℝ) (h1 : x + y = 1) (h2 : x + z = 0) (h3 : y + z = -1) : 
    x = 1 ∧ y = 0 ∧ z = -1 := 
by 
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1760_176013


namespace NUMINAMATH_GPT_lucinda_jelly_beans_l1760_176016

theorem lucinda_jelly_beans (g l : ℕ) 
  (h₁ : g = 3 * l) 
  (h₂ : g - 20 = 4 * (l - 20)) : 
  g = 180 := 
by 
  sorry

end NUMINAMATH_GPT_lucinda_jelly_beans_l1760_176016


namespace NUMINAMATH_GPT_min_value_expression_l1760_176019

theorem min_value_expression (x y z : ℝ) : ∃ v, v = 0 ∧ ∀ x y z : ℝ, x^2 + 2 * x * y + 3 * y^2 + 2 * x * z + 3 * z^2 ≥ v := 
by 
  use 0
  sorry

end NUMINAMATH_GPT_min_value_expression_l1760_176019


namespace NUMINAMATH_GPT_discs_angular_velocity_relation_l1760_176014

variables {r1 r2 ω1 ω2 : ℝ} -- Radii and angular velocities

-- Conditions:
-- Discs have radii r1 and r2, and angular velocities ω1 and ω2, respectively.
-- Discs come to a halt after being brought into contact via friction.
-- Discs have identical thickness and are made of the same material.
-- Prove the required relation.

theorem discs_angular_velocity_relation
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (halt_contact : ω1 * r1^3 = ω2 * r2^3) :
  ω1 * r1^3 = ω2 * r2^3 :=
sorry

end NUMINAMATH_GPT_discs_angular_velocity_relation_l1760_176014


namespace NUMINAMATH_GPT_cost_of_a_pen_l1760_176066

theorem cost_of_a_pen:
  ∃ x y : ℕ, 5 * x + 4 * y = 345 ∧ 3 * x + 6 * y = 285 ∧ x = 52 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_a_pen_l1760_176066


namespace NUMINAMATH_GPT_proposition_3_true_proposition_4_true_l1760_176058

def exp_pos (x : ℝ) : Prop := Real.exp x > 0

def two_power_gt_xsq (x : ℝ) : Prop := 2^x > x^2

def prod_gt_one (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop := a * b > 1

def geom_seq_nec_suff (a b c : ℝ) : Prop := ¬(b = Real.sqrt (a * c) ∨ (a * b = c * b ∧ b^2 = a * c))

theorem proposition_3_true (a b : ℝ) (ha : a > 1) (hb : b > 1) : prod_gt_one a b ha hb :=
sorry

theorem proposition_4_true (a b c : ℝ) : geom_seq_nec_suff a b c :=
sorry

end NUMINAMATH_GPT_proposition_3_true_proposition_4_true_l1760_176058


namespace NUMINAMATH_GPT_estimate_expr_l1760_176011

theorem estimate_expr : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_GPT_estimate_expr_l1760_176011


namespace NUMINAMATH_GPT_tangent_circles_t_value_l1760_176064

theorem tangent_circles_t_value (t : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = t^2 → x^2 + y^2 + 6 * x - 8 * y + 24 = 0 → dist (0, 0) (-3, 4) = t + 1) → t = 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_circles_t_value_l1760_176064


namespace NUMINAMATH_GPT_min_moves_to_break_chocolate_l1760_176026

theorem min_moves_to_break_chocolate (n m : ℕ) (tiles : ℕ) (moves : ℕ) :
    (n = 4) → (m = 10) → (tiles = n * m) → (moves = tiles - 1) → moves = 39 :=
by
  intros hnm hn4 hm10 htm
  sorry

end NUMINAMATH_GPT_min_moves_to_break_chocolate_l1760_176026


namespace NUMINAMATH_GPT_exist_common_divisor_l1760_176045

theorem exist_common_divisor (a : ℕ → ℕ) (m : ℕ) (h_positive : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i)
  (p : ℕ → ℤ) (h_poly : ∀ n : ℕ, ∃ i, 1 ≤ i ∧ i ≤ m ∧ (a i : ℤ) ∣ p n) :
  ∃ j, 1 ≤ j ∧ j ≤ m ∧ ∀ n, (a j : ℤ) ∣ p n :=
by
  sorry

end NUMINAMATH_GPT_exist_common_divisor_l1760_176045


namespace NUMINAMATH_GPT_toy_store_problem_l1760_176009

variables (x y : ℕ)

theorem toy_store_problem (h1 : 8 * x + 26 * y + 33 * (31 - x - y) / 2 = 370)
                          (h2 : x + y + (31 - x - y) / 2 = 31) :
    x = 20 :=
sorry

end NUMINAMATH_GPT_toy_store_problem_l1760_176009


namespace NUMINAMATH_GPT_max_discardable_grapes_l1760_176049

theorem max_discardable_grapes (n : ℕ) (k : ℕ) (h : k = 8) : 
  ∃ m : ℕ, m < k ∧ (∀ q : ℕ, q * k + m = n) ∧ m = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_discardable_grapes_l1760_176049


namespace NUMINAMATH_GPT_sum_of_coordinates_D_l1760_176001

theorem sum_of_coordinates_D (x y : ℝ) 
  (M_midpoint : (4, 10) = ((8 + x) / 2, (6 + y) / 2)) : 
  x + y = 14 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_D_l1760_176001


namespace NUMINAMATH_GPT_value_y1_y2_l1760_176096

variable {x1 x2 y1 y2 : ℝ}

-- Points on the inverse proportion function
def on_graph (x y : ℝ) : Prop := y = -3 / x

-- Given conditions
theorem value_y1_y2 (hx1 : on_graph x1 y1) (hx2 : on_graph x2 y2) (hxy : x1 * x2 = 2) : y1 * y2 = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_y1_y2_l1760_176096


namespace NUMINAMATH_GPT_count_valid_48_tuples_l1760_176048

open BigOperators

theorem count_valid_48_tuples : 
  ∃ n : ℕ, n = 54 ^ 48 ∧ 
  ( ∃ a : Fin 48 → ℕ, 
    (∀ i : Fin 48, 0 ≤ a i ∧ a i ≤ 100) ∧ 
    (∀ (i j : Fin 48), i < j → a i ≠ a j ∧ a i ≠ a j + 1) 
  ) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_48_tuples_l1760_176048


namespace NUMINAMATH_GPT_coordinates_of_C_l1760_176074

theorem coordinates_of_C (A B : ℝ × ℝ) (hA : A = (-2, -1)) (hB : B = (4, 9)) :
    ∃ C : ℝ × ℝ, (dist C A) = 4 * dist C B ∧ C = (-0.8, 1) :=
sorry

end NUMINAMATH_GPT_coordinates_of_C_l1760_176074


namespace NUMINAMATH_GPT_num_ways_to_tile_3x5_is_40_l1760_176054

-- Definition of the problem
def numTilings (tiles : List (ℕ × ℕ)) (m n : ℕ) : ℕ :=
  sorry -- Placeholder for actual tiling computation

-- Condition specific to this problem
def specificTiles : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

-- Problem statement in Lean 4
theorem num_ways_to_tile_3x5_is_40 :
  numTilings specificTiles 3 5 = 40 :=
sorry

end NUMINAMATH_GPT_num_ways_to_tile_3x5_is_40_l1760_176054


namespace NUMINAMATH_GPT_pizza_payment_difference_l1760_176067

theorem pizza_payment_difference
  (total_slices : ℕ := 12)
  (plain_cost : ℝ := 12)
  (onion_cost : ℝ := 3)
  (jack_onion_slices : ℕ := 4)
  (jack_plain_slices : ℕ := 3)
  (carl_plain_slices : ℕ := 5) :
  let total_cost := plain_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jack_onion_payment := jack_onion_slices * cost_per_slice
  let jack_plain_payment := jack_plain_slices * cost_per_slice
  let jack_total_payment := jack_onion_payment + jack_plain_payment
  let carl_total_payment := carl_plain_slices * cost_per_slice
  jack_total_payment - carl_total_payment = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_pizza_payment_difference_l1760_176067


namespace NUMINAMATH_GPT_intersection_of_sets_l1760_176087
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1760_176087


namespace NUMINAMATH_GPT_coronavirus_diameter_scientific_notation_l1760_176072

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end NUMINAMATH_GPT_coronavirus_diameter_scientific_notation_l1760_176072


namespace NUMINAMATH_GPT_pyramid_height_l1760_176075

theorem pyramid_height (perimeter_side_base : ℝ) (apex_distance_to_vertex : ℝ) (height_peak_to_center_base : ℝ) : 
  (perimeter_side_base = 32) → (apex_distance_to_vertex = 12) → 
  height_peak_to_center_base = 4 * Real.sqrt 7 := 
  by
    sorry

end NUMINAMATH_GPT_pyramid_height_l1760_176075


namespace NUMINAMATH_GPT_circle_radius_l1760_176035

theorem circle_radius (x y : ℝ) : (x^2 - 4 * x + y^2 - 21 = 0) → (∃ r : ℝ, r = 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1760_176035


namespace NUMINAMATH_GPT_solve_for_diamond_l1760_176003

theorem solve_for_diamond (d : ℕ) (h : d * 5 + 3 = d * 6 + 2) : d = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_diamond_l1760_176003


namespace NUMINAMATH_GPT_possible_values_of_m_l1760_176012

theorem possible_values_of_m (m : ℕ) (h1 : 3 * m + 15 > 3 * m + 8) 
  (h2 : 3 * m + 8 > 4 * m - 4) (h3 : m > 11) : m = 11 := 
by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l1760_176012


namespace NUMINAMATH_GPT_members_not_in_A_nor_B_l1760_176017

variable (U A B : Finset ℕ) -- We define the sets as finite sets of natural numbers.
variable (hU_size : U.card = 190) -- Size of set U is 190.
variable (hB_size : (U ∩ B).card = 49) -- 49 items are in set B.
variable (hAB_size : (A ∩ U ∩ B).card = 23) -- 23 items are in both A and B.
variable (hA_size : (U ∩ A).card = 105) -- 105 items are in set A.

theorem members_not_in_A_nor_B :
  (U \ (A ∪ B)).card = 59 := sorry

end NUMINAMATH_GPT_members_not_in_A_nor_B_l1760_176017


namespace NUMINAMATH_GPT_greatest_multiple_of_4_l1760_176033

theorem greatest_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x > 0) (h3 : x^3 < 500) : x ≤ 4 :=
by sorry

end NUMINAMATH_GPT_greatest_multiple_of_4_l1760_176033


namespace NUMINAMATH_GPT_smallest_difference_of_factors_l1760_176015

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2268) : 
  (a = 42 ∧ b = 54) ∨ (a = 54 ∧ b = 42) := sorry

end NUMINAMATH_GPT_smallest_difference_of_factors_l1760_176015


namespace NUMINAMATH_GPT_power_function_increasing_l1760_176020

theorem power_function_increasing {α : ℝ} (hα : α = 1 ∨ α = 3 ∨ α = 1 / 2) :
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → x ^ α ≤ y ^ α := 
sorry

end NUMINAMATH_GPT_power_function_increasing_l1760_176020


namespace NUMINAMATH_GPT_linear_avoid_third_quadrant_l1760_176008

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_avoid_third_quadrant_l1760_176008


namespace NUMINAMATH_GPT_experiment_implies_101_sq_1_equals_10200_l1760_176036

theorem experiment_implies_101_sq_1_equals_10200 :
    (5^2 - 1 = 24) →
    (7^2 - 1 = 48) →
    (11^2 - 1 = 120) →
    (13^2 - 1 = 168) →
    (101^2 - 1 = 10200) :=
by
  repeat { intro }
  sorry

end NUMINAMATH_GPT_experiment_implies_101_sq_1_equals_10200_l1760_176036


namespace NUMINAMATH_GPT_face_opposite_to_A_l1760_176047

-- Define the faces and their relationships
inductive Face : Type
| A | B | C | D | E | F
open Face

def adjacent (x y : Face) : Prop :=
  match x, y with
  | A, B => true
  | B, A => true
  | C, A => true
  | A, C => true
  | D, A => true
  | A, D => true
  | C, D => true
  | D, C => true
  | E, F => true
  | F, E => true
  | _, _ => false

-- Theorem stating that "F" is opposite to "A" given the provided conditions.
theorem face_opposite_to_A : ∀ x : Face, (adjacent A x = false) → (x = B ∨ x = C ∨ x = D → false) → (x = E ∨ x = F) → x = F := 
  by
    intros x h1 h2 h3
    sorry

end NUMINAMATH_GPT_face_opposite_to_A_l1760_176047


namespace NUMINAMATH_GPT_determine_values_l1760_176082

theorem determine_values (A B : ℚ) :
  (A + B = 4) ∧ (2 * A - 7 * B = 3) →
  A = 31 / 9 ∧ B = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_determine_values_l1760_176082


namespace NUMINAMATH_GPT_asymptotes_equation_l1760_176090

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 / 64 - y^2 / 36 = 1

theorem asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y → (y = (3/4) * x ∨ y = - (3/4) * x) :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_GPT_asymptotes_equation_l1760_176090


namespace NUMINAMATH_GPT_celsius_to_fahrenheit_conversion_l1760_176070

theorem celsius_to_fahrenheit_conversion (k b : ℝ) :
  (∀ C : ℝ, (C * k + b = C * 1.8 + 32)) → (k = 1.8 ∧ b = 32) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_celsius_to_fahrenheit_conversion_l1760_176070


namespace NUMINAMATH_GPT_square_distance_l1760_176023

theorem square_distance (a b c d e f: ℝ) 
  (side_length : ℝ)
  (AB : a = 0 ∧ b = side_length)
  (BC : c = side_length ∧ d = 0)
  (BE_dist : (a - b)^2 + (b - b)^2 = 25)
  (AE_dist : a^2 + (c - b)^2 = 144)
  (DF_dist : (d)^2 + (d)^2 = 25)
  (CF_dist : (d - c)^2 + e^2 = 144) :
  (f - d)^2 + (e - a)^2 = 578 :=
by
  -- Required to bypass the proof steps
  sorry

end NUMINAMATH_GPT_square_distance_l1760_176023


namespace NUMINAMATH_GPT_true_propositions_3_and_4_l1760_176078

-- Define the condition for Proposition ③
def prop3_statement (m : ℝ) : Prop :=
  (m > 2) → ∀ x : ℝ, (x^2 - 2*x + m > 0)

def prop3_contrapositive (m : ℝ) : Prop :=
  (∀ x : ℝ, (x^2 - 2*x + m > 0)) → (m > 2)

-- Define the condition for Proposition ④
def prop4_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (1 + x) = f (1 - x))

def prop4_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) = f (x))

-- Theorem to prove Propositions ③ and ④ are true
theorem true_propositions_3_and_4
  (m : ℝ) (f : ℝ → ℝ)
  (h3 : ∀ (m : ℝ), prop3_contrapositive m)
  (h4 : prop4_condition f): 
  prop3_statement m ∧ prop4_period_4 f :=
by {
  sorry
}

end NUMINAMATH_GPT_true_propositions_3_and_4_l1760_176078


namespace NUMINAMATH_GPT_units_digit_of_power_435_l1760_176027

def units_digit_cycle (n : ℕ) : ℕ :=
  n % 2

def units_digit_of_four_powers (cycle : ℕ) : ℕ :=
  if cycle = 0 then 6 else 4

theorem units_digit_of_power_435 : 
  units_digit_of_four_powers (units_digit_cycle (3^5)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_power_435_l1760_176027


namespace NUMINAMATH_GPT_rectangular_prism_dimensions_l1760_176080

theorem rectangular_prism_dimensions 
    (a b c : ℝ) -- edges of the rectangular prism
    (h_increase_volume : (2 * a * b = 90)) -- condition 2: increasing height increases volume by 90 cm³ 
    (h_volume_proportion : (a * (c + 2)) / 2 = (3 / 5) * (a * b * c)) -- condition 3: height change results in 3/5 of original volume
    (h_edge_relation : (a = 5 * b ∨ b = 5 * a ∨ a * b = 45)) -- condition 1: one edge 5 times longer
    : 
    (a = 0.9 ∧ b = 50 ∧ c = 10) ∨ (a = 2 ∧ b = 22.5 ∧ c = 10) ∨ (a = 3 ∧ b = 15 ∧ c = 10) :=
sorry

end NUMINAMATH_GPT_rectangular_prism_dimensions_l1760_176080


namespace NUMINAMATH_GPT_number_of_intersections_l1760_176095

/-- 
  Define the two curves as provided in the problem:
  curve1 is defined by the equation 3x² + 2y² = 6,
  curve2 is defined by the equation x² - 2y² = 1.
  We aim to prove that there are exactly 4 distinct intersection points.
--/
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

theorem number_of_intersections : ∃ (points : Finset (ℝ × ℝ)), (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 4 :=
sorry

end NUMINAMATH_GPT_number_of_intersections_l1760_176095


namespace NUMINAMATH_GPT_sheets_borrowed_l1760_176084

theorem sheets_borrowed (pages sheets borrowed remaining_sheets : ℕ) 
  (h1 : pages = 70) 
  (h2 : sheets = 35)
  (h3 : remaining_sheets = sheets - borrowed)
  (h4 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> 2*i-1 <= pages) 
  (h5 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> i + 1 != borrowed ∧ i <= remaining_sheets)
  (avg : ℕ) (h6 : avg = 28)
  : borrowed = 17 := by
  sorry

end NUMINAMATH_GPT_sheets_borrowed_l1760_176084
