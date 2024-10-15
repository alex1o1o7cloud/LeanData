import Mathlib

namespace NUMINAMATH_GPT_sector_angle_l187_18786

noncomputable def central_angle_of_sector (r l : ℝ) : ℝ := l / r

theorem sector_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  central_angle_of_sector r l = 1 ∨ central_angle_of_sector r l = 4 :=
by
  sorry

end NUMINAMATH_GPT_sector_angle_l187_18786


namespace NUMINAMATH_GPT_probability_of_same_length_l187_18737

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end NUMINAMATH_GPT_probability_of_same_length_l187_18737


namespace NUMINAMATH_GPT_rearrange_circles_sums13_l187_18732

def isSum13 (a b c d x y z w : ℕ) : Prop :=
  (a + 4 + b = 13) ∧ (b + 2 + d = 13) ∧ (d + 1 + c = 13) ∧ (c + 3 + a = 13)

theorem rearrange_circles_sums13 : 
  ∃ (a b c d x y z w : ℕ), 
  a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 6 ∧ 
  a + b = 9 ∧ b + z = 11 ∧ z + c = 12 ∧ c + a = 10 ∧ 
  isSum13 a b c d x y z w :=
by {
  sorry
}

end NUMINAMATH_GPT_rearrange_circles_sums13_l187_18732


namespace NUMINAMATH_GPT_simple_interest_sum_l187_18769

theorem simple_interest_sum (SI R T : ℝ) (hSI : SI = 4016.25) (hR : R = 0.01) (hT : T = 3) :
  SI / (R * T) = 133875 := by
  sorry

end NUMINAMATH_GPT_simple_interest_sum_l187_18769


namespace NUMINAMATH_GPT_edward_books_bought_l187_18767

def money_spent : ℕ := 6
def cost_per_book : ℕ := 3

theorem edward_books_bought : money_spent / cost_per_book = 2 :=
by
  sorry

end NUMINAMATH_GPT_edward_books_bought_l187_18767


namespace NUMINAMATH_GPT_problem1_problem2_l187_18717

theorem problem1 : 27^((2:ℝ)/(3:ℝ)) - 2^(Real.logb 2 3) * Real.logb 2 (1/8) = 18 := 
by
  sorry -- proof omitted

theorem problem2 : 1/(Real.sqrt 5 - 2) - (Real.sqrt 5 + 2)^0 - Real.sqrt ((2 - Real.sqrt 5)^2) = 2*(Real.sqrt 5 - 1) := 
by
  sorry -- proof omitted

end NUMINAMATH_GPT_problem1_problem2_l187_18717


namespace NUMINAMATH_GPT_work_completion_time_l187_18711

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 5) (hC : C = 1 / 20) :
  1 / (A + B + C) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_work_completion_time_l187_18711


namespace NUMINAMATH_GPT_sum_of_values_k_l187_18756

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_k_l187_18756


namespace NUMINAMATH_GPT_line_to_slope_intercept_l187_18704

noncomputable def line_equation (v p q : ℝ × ℝ) : Prop :=
  (v.1 * (p.1 - q.1) + v.2 * (p.2 - q.2)) = 0

theorem line_to_slope_intercept (x y m b : ℝ) :
  line_equation (3, -4) (x, y) (2, 8) → (m, b) = (3 / 4, 6.5) :=
  by
    sorry

end NUMINAMATH_GPT_line_to_slope_intercept_l187_18704


namespace NUMINAMATH_GPT_sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l187_18735

theorem sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3 :
  let largestThreeDigitMultipleOf4 := 996
  let smallestFourDigitMultipleOf3 := 1002
  largestThreeDigitMultipleOf4 + smallestFourDigitMultipleOf3 = 1998 :=
by
  sorry

end NUMINAMATH_GPT_sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l187_18735


namespace NUMINAMATH_GPT_james_toys_l187_18725

-- Define the conditions and the problem statement
theorem james_toys (x : ℕ) (h1 : ∀ x, 2 * x = 60 - x) : x = 20 :=
sorry

end NUMINAMATH_GPT_james_toys_l187_18725


namespace NUMINAMATH_GPT_x_eq_zero_sufficient_not_necessary_l187_18718

theorem x_eq_zero_sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2 * x = 0) ∧ (x^2 - 2 * x = 0 → x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_x_eq_zero_sufficient_not_necessary_l187_18718


namespace NUMINAMATH_GPT_boxes_needed_to_complete_flooring_l187_18743

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end NUMINAMATH_GPT_boxes_needed_to_complete_flooring_l187_18743


namespace NUMINAMATH_GPT_lending_period_C_l187_18714

theorem lending_period_C (R : ℝ) (P_B P_C T_B I : ℝ) (h1 : R = 13.75) (h2 : P_B = 4000) (h3 : P_C = 2000) (h4 : T_B = 2) (h5 : I = 2200) : 
  ∃ T_C : ℝ, T_C = 4 :=
by
  -- Definitions and known facts
  let I_B := (P_B * R * T_B) / 100
  let I_C := I - I_B
  let T_C := I_C / ((P_C * R) / 100)
  -- Prove the target
  use T_C
  sorry

end NUMINAMATH_GPT_lending_period_C_l187_18714


namespace NUMINAMATH_GPT_total_drums_hit_l187_18765

/-- 
Given the conditions of the problem, Juanita hits 4500 drums in total. 
-/
theorem total_drums_hit (entry_fee cost_per_drum_hit earnings_per_drum_hit_beyond_200_double
                         net_loss: ℝ) 
                         (first_200_drums hits_after_200: ℕ) :
  entry_fee = 10 → 
  cost_per_drum_hit = 0.02 →
  earnings_per_drum_hit_beyond_200_double = 0.025 →
  net_loss = -7.5 →
  hits_after_200 = 4300 →
  first_200_drums = 200 →
  (-net_loss = entry_fee + (first_200_drums * cost_per_drum_hit) +
   (hits_after_200 * (earnings_per_drum_hit_beyond_200_double - cost_per_drum_hit))) →
  first_200_drums + hits_after_200 = 4500 :=
by
  intro h_entry_fee h_cost_per_drum_hit h_earnings_per_drum_hit_beyond_200_double h_net_loss h_hits_after_200
       h_first_200_drums h_loss_equation
  sorry

end NUMINAMATH_GPT_total_drums_hit_l187_18765


namespace NUMINAMATH_GPT_probability_MAME_top_l187_18709

-- Conditions
def paper_parts : ℕ := 8
def desired_top : ℕ := 1

-- Question and Proof Problem (Probability calculation)
theorem probability_MAME_top : (1 : ℚ) / paper_parts = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_MAME_top_l187_18709


namespace NUMINAMATH_GPT_milton_sold_total_pies_l187_18751

-- Definitions for the given conditions.
def apple_pie_slices : ℕ := 8
def peach_pie_slices : ℕ := 6
def cherry_pie_slices : ℕ := 10

def apple_slices_ordered : ℕ := 88
def peach_slices_ordered : ℕ := 78
def cherry_slices_ordered : ℕ := 45

-- Function to compute the number of pies, rounding up as necessary
noncomputable def pies_sold (ordered : ℕ) (slices : ℕ) : ℕ :=
  (ordered + slices - 1) / slices  -- Using integer division to round up

-- The theorem asserting the total number of pies sold 
theorem milton_sold_total_pies : 
  pies_sold apple_slices_ordered apple_pie_slices +
  pies_sold peach_slices_ordered peach_pie_slices +
  pies_sold cherry_slices_ordered cherry_pie_slices = 29 :=
by sorry

end NUMINAMATH_GPT_milton_sold_total_pies_l187_18751


namespace NUMINAMATH_GPT_car_trip_problem_l187_18782

theorem car_trip_problem (a b c : ℕ) (x : ℕ) 
(h1 : 1 ≤ a) 
(h2 : a + b + c ≤ 9)
(h3 : 100 * b + 10 * c + a - 100 * a - 10 * b - c = 60 * x) 
: a^2 + b^2 + c^2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_car_trip_problem_l187_18782


namespace NUMINAMATH_GPT_initial_investment_proof_l187_18733

-- Definitions for the conditions
def initial_investment_A : ℝ := sorry
def contribution_B : ℝ := 15750
def profit_ratio_A : ℝ := 2
def profit_ratio_B : ℝ := 3
def time_A : ℝ := 12
def time_B : ℝ := 4

-- Lean statement to prove
theorem initial_investment_proof : initial_investment_A * time_A * profit_ratio_B = contribution_B * time_B * profit_ratio_A → initial_investment_A = 1750 :=
by
  sorry

end NUMINAMATH_GPT_initial_investment_proof_l187_18733


namespace NUMINAMATH_GPT_find_gamma_delta_l187_18775

theorem find_gamma_delta (γ δ : ℝ) (h : ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90 * x + 1980) / (x^2 + 60 * x - 3240)) : 
  γ + δ = 140 :=
sorry

end NUMINAMATH_GPT_find_gamma_delta_l187_18775


namespace NUMINAMATH_GPT_gcd_987654_876543_eq_3_l187_18726

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end NUMINAMATH_GPT_gcd_987654_876543_eq_3_l187_18726


namespace NUMINAMATH_GPT_solve_eq_l187_18758

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end NUMINAMATH_GPT_solve_eq_l187_18758


namespace NUMINAMATH_GPT_total_lives_l187_18791

theorem total_lives (initial_players new_players lives_per_person : ℕ)
  (h_initial : initial_players = 8)
  (h_new : new_players = 2)
  (h_lives : lives_per_person = 6)
  : (initial_players + new_players) * lives_per_person = 60 := 
by
  sorry

end NUMINAMATH_GPT_total_lives_l187_18791


namespace NUMINAMATH_GPT_problem_statement_l187_18708

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l187_18708


namespace NUMINAMATH_GPT_prob_three_red_cards_l187_18788

noncomputable def probability_of_three_red_cards : ℚ :=
  let total_ways := 52 * 51 * 50
  let ways_to_choose_red_cards := 26 * 25 * 24
  ways_to_choose_red_cards / total_ways

theorem prob_three_red_cards : probability_of_three_red_cards = 4 / 17 := sorry

end NUMINAMATH_GPT_prob_three_red_cards_l187_18788


namespace NUMINAMATH_GPT_sqrt_expr_eq_two_l187_18707

noncomputable def expr := Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2)

theorem sqrt_expr_eq_two : expr = 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expr_eq_two_l187_18707


namespace NUMINAMATH_GPT_Carter_card_number_l187_18797

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end NUMINAMATH_GPT_Carter_card_number_l187_18797


namespace NUMINAMATH_GPT_inverse_function_value_l187_18784

-- Defining the function g as a list of pairs
def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 6
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | 6 => 2
  | _ => 0 -- default case which should not be used

-- Defining the inverse function g_inv using the values determined from g
def g_inv (y : ℕ) : ℕ :=
  match y with
  | 3 => 1
  | 6 => 2
  | 1 => 3
  | 5 => 4
  | 4 => 5
  | 2 => 6
  | _ => 0 -- default case which should not be used

theorem inverse_function_value :
  g_inv (g_inv (g_inv 6)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_value_l187_18784


namespace NUMINAMATH_GPT_find_initial_days_provisions_last_l187_18750

def initial_days_provisions_last (initial_men reinforcements days_after_reinforcement : ℕ) (x : ℕ) : Prop :=
  initial_men * (x - 15) = (initial_men + reinforcements) * days_after_reinforcement

theorem find_initial_days_provisions_last
  (initial_men reinforcements days_after_reinforcement x : ℕ)
  (h1 : initial_men = 2000)
  (h2 : reinforcements = 1900)
  (h3 : days_after_reinforcement = 20)
  (h4 : initial_days_provisions_last initial_men reinforcements days_after_reinforcement x) :
  x = 54 :=
by
  sorry


end NUMINAMATH_GPT_find_initial_days_provisions_last_l187_18750


namespace NUMINAMATH_GPT_intersection_M_N_l187_18799

def M : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def N : Set ℝ := { x | Real.log (x - 2) / Real.log (1 / 2) ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l187_18799


namespace NUMINAMATH_GPT_fish_weight_l187_18730

variables (W G T : ℕ)

-- Define the known conditions
axiom tail_weight : W = 1
axiom head_weight : G = W + T / 2
axiom torso_weight : T = G + W

-- Define the proof statement
theorem fish_weight : W + G + T = 8 :=
by
  sorry

end NUMINAMATH_GPT_fish_weight_l187_18730


namespace NUMINAMATH_GPT_fraction_addition_l187_18760

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l187_18760


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l187_18747

theorem tangent_line_to_parabola (r : ℝ) :
  (∃ x : ℝ, 2 * x^2 - x - r = 0) ∧
  (∀ x1 x2 : ℝ, (2 * x1^2 - x1 - r = 0) ∧ (2 * x2^2 - x2 - r = 0) → x1 = x2) →
  r = -1 / 8 :=
sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l187_18747


namespace NUMINAMATH_GPT_domain_of_f_eq_l187_18712

def domain_of_fractional_function : Set ℝ := 
  { x : ℝ | x > -1 }

theorem domain_of_f_eq : 
  ∀ x : ℝ, x ∈ domain_of_fractional_function ↔ x > -1 :=
by
  sorry -- Proof this part in Lean 4. The domain of f(x) is (-1, +∞)

end NUMINAMATH_GPT_domain_of_f_eq_l187_18712


namespace NUMINAMATH_GPT_sum_is_3600_l187_18768

variables (P R T : ℝ)
variables (CI SI : ℝ)

theorem sum_is_3600
  (hR : R = 10)
  (hT : T = 2)
  (hCI : CI = P * (1 + R / 100) ^ T - P)
  (hSI : SI = P * R * T / 100)
  (h_diff : CI - SI = 36) :
  P = 3600 :=
sorry

end NUMINAMATH_GPT_sum_is_3600_l187_18768


namespace NUMINAMATH_GPT_isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l187_18729

open Real

theorem isosceles_triangle_of_sine_ratio (a b c : ℝ) (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h1 : a = b * sin C + c * cos B) :
  C = π / 4 :=
sorry

theorem obtuse_triangle_of_tan_sum_neg (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_tan_sum : tan A + tan B + tan C < 0) :
  ∃ (E : ℝ), (A = E ∨ B = E ∨ C = E) ∧ π / 2 < E :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l187_18729


namespace NUMINAMATH_GPT_calculate_rent_l187_18794

def monthly_income : ℝ := 3200
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def car_payment : ℝ := 350
def gas_maintenance : ℝ := 350

def total_expenses : ℝ := utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance
def rent : ℝ := monthly_income - total_expenses

theorem calculate_rent : rent = 1250 := by
  -- condition proof here
  sorry

end NUMINAMATH_GPT_calculate_rent_l187_18794


namespace NUMINAMATH_GPT_trackball_mice_count_l187_18731

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end NUMINAMATH_GPT_trackball_mice_count_l187_18731


namespace NUMINAMATH_GPT_pete_ran_least_distance_l187_18793

theorem pete_ran_least_distance
  (phil_distance : ℕ := 4)
  (tom_distance : ℕ := 6)
  (pete_distance : ℕ := 2)
  (amal_distance : ℕ := 8)
  (sanjay_distance : ℕ := 7) :
  pete_distance ≤ phil_distance ∧
  pete_distance ≤ tom_distance ∧
  pete_distance ≤ amal_distance ∧
  pete_distance ≤ sanjay_distance :=
by {
  sorry
}

end NUMINAMATH_GPT_pete_ran_least_distance_l187_18793


namespace NUMINAMATH_GPT_number_of_divisors_36_l187_18796

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_divisors_36_l187_18796


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_9_is_36_l187_18727

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * (a n)
noncomputable def Sn (b : ℕ → ℝ) (n : ℕ) : ℝ := n * (b 1 + b n) / 2

theorem arithmetic_sequence_sum_9_is_36 (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_cond : a 4 * a 6 = 2 * a 5) (h_b5 : b 5 = 2 * a 5) : Sn b 9 = 36 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_9_is_36_l187_18727


namespace NUMINAMATH_GPT_door_solution_l187_18773

def door_problem (x : ℝ) : Prop :=
  let w := x - 4
  let h := x - 2
  let diagonal := x
  (diagonal ^ 2 - (h) ^ 2 = (w) ^ 2)

theorem door_solution (x : ℝ) : door_problem x :=
  sorry

end NUMINAMATH_GPT_door_solution_l187_18773


namespace NUMINAMATH_GPT_log_pi_inequality_l187_18795

theorem log_pi_inequality (a b : ℝ) (π : ℝ) (h1 : 2^a = π) (h2 : 5^b = π) (h3 : a = Real.log π / Real.log 2) (h4 : b = Real.log π / Real.log 5) :
  (1 / a) + (1 / b) > 2 :=
by
  sorry

end NUMINAMATH_GPT_log_pi_inequality_l187_18795


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_6_mod_19_l187_18764

theorem least_five_digit_congruent_to_6_mod_19 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 19 = 6 ∧ n = 10011 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_6_mod_19_l187_18764


namespace NUMINAMATH_GPT_marbles_per_box_l187_18787

-- Define the total number of marbles
def total_marbles : Nat := 18

-- Define the number of boxes
def number_of_boxes : Nat := 3

-- Prove there are 6 marbles in each box
theorem marbles_per_box : total_marbles / number_of_boxes = 6 := by
  sorry

end NUMINAMATH_GPT_marbles_per_box_l187_18787


namespace NUMINAMATH_GPT_simplify_and_evaluate_l187_18728

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) : 
  (1 - (1 : ℚ) / (a + 1)) / (a / ((a * a) - 1)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l187_18728


namespace NUMINAMATH_GPT_sum_of_side_lengths_l187_18792

theorem sum_of_side_lengths (A B C : ℕ) (hA : A = 10) (h_nat_B : B > 0) (h_nat_C : C > 0)
(h_eq_area : B^2 + C^2 = A^2) : B + C = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_side_lengths_l187_18792


namespace NUMINAMATH_GPT_linear_combination_value_l187_18734

theorem linear_combination_value (x y : ℝ) (h₁ : 2 * x + y = 8) (h₂ : x + 2 * y = 10) :
  8 * x ^ 2 + 10 * x * y + 8 * y ^ 2 = 164 :=
sorry

end NUMINAMATH_GPT_linear_combination_value_l187_18734


namespace NUMINAMATH_GPT_solution_set_inequality1_solution_set_inequality2_l187_18762

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality1_solution_set_inequality2_l187_18762


namespace NUMINAMATH_GPT_total_cost_supplies_l187_18744

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end NUMINAMATH_GPT_total_cost_supplies_l187_18744


namespace NUMINAMATH_GPT_expression_evaluation_l187_18763

variable {x y : ℝ}

theorem expression_evaluation (h : (x-2)^2 + |y-3| = 0) :
  ( (x - 2 * y) * (x + 2 * y) - (x - y) ^ 2 + y * (y + 2 * x) ) / (-2 * y) = 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l187_18763


namespace NUMINAMATH_GPT_Carol_cleaning_time_l187_18700

theorem Carol_cleaning_time 
(Alice_time : ℕ) 
(Bob_time : ℕ) 
(Carol_time : ℕ) 
(h1 : Alice_time = 40) 
(h2 : Bob_time = 3 * Alice_time / 4) 
(h3 : Carol_time = 2 * Bob_time) :
  Carol_time = 60 := 
sorry

end NUMINAMATH_GPT_Carol_cleaning_time_l187_18700


namespace NUMINAMATH_GPT_percentage_chromium_first_alloy_l187_18748

theorem percentage_chromium_first_alloy
  (x : ℝ) (h : (x / 100) * 15 + (8 / 100) * 35 = (9.2 / 100) * 50) : x = 12 :=
sorry

end NUMINAMATH_GPT_percentage_chromium_first_alloy_l187_18748


namespace NUMINAMATH_GPT_fractional_inequality_solution_l187_18774

theorem fractional_inequality_solution (x : ℝ) :
  (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_fractional_inequality_solution_l187_18774


namespace NUMINAMATH_GPT_fractional_expression_value_l187_18766

theorem fractional_expression_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 2 * x - 3 * y - z = 0)
  (h2 : x + 3 * y - 14 * z = 0) :
  (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by sorry

end NUMINAMATH_GPT_fractional_expression_value_l187_18766


namespace NUMINAMATH_GPT_average_marks_increase_ratio_l187_18739

theorem average_marks_increase_ratio
  (T : ℕ)  -- The correct total marks of the class
  (n : ℕ)  -- The number of pupils in the class
  (h_n : n = 16) (wrong_mark : ℕ) (correct_mark : ℕ)  -- The wrong and correct marks
  (h_wrong : wrong_mark = 73) (h_correct : correct_mark = 65) :
  (8 : ℚ) / T = (wrong_mark - correct_mark : ℚ) / n * (n / T) :=
by
  sorry

end NUMINAMATH_GPT_average_marks_increase_ratio_l187_18739


namespace NUMINAMATH_GPT_adamek_marbles_l187_18746

theorem adamek_marbles : ∃ n : ℕ, (∀ k : ℕ, n = 4 * k ∧ n = 3 * (k + 8)) → n = 96 :=
by
  sorry

end NUMINAMATH_GPT_adamek_marbles_l187_18746


namespace NUMINAMATH_GPT_percentage_of_360_is_165_6_l187_18754

theorem percentage_of_360_is_165_6 :
  (165.6 / 360) * 100 = 46 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_360_is_165_6_l187_18754


namespace NUMINAMATH_GPT_find_f_2_l187_18777

def f (a b x : ℝ) := a * x^3 - b * x + 1

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = -1) : f a b 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l187_18777


namespace NUMINAMATH_GPT_anna_walk_distance_l187_18722

theorem anna_walk_distance (d: ℚ) 
  (hd: 22 * 1.25 - 4 * 1.25 = d)
  (d2: d = 3.7): d = 3.7 :=
by 
  sorry

end NUMINAMATH_GPT_anna_walk_distance_l187_18722


namespace NUMINAMATH_GPT_find_ratio_l187_18723

open Real

theorem find_ratio (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l187_18723


namespace NUMINAMATH_GPT_distance_to_convenience_store_l187_18785

def distance_work := 6
def days_work := 5
def distance_dog_walk := 2
def times_dog_walk := 2
def days_week := 7
def distance_friend_house := 1
def times_friend_visit := 1
def total_miles := 95
def trips_convenience_store := 2

theorem distance_to_convenience_store :
  ∃ x : ℝ,
    (distance_work * 2 * days_work) +
    (distance_dog_walk * times_dog_walk * days_week) +
    (distance_friend_house * 2 * times_friend_visit) +
    (x * trips_convenience_store) = total_miles
    → x = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_convenience_store_l187_18785


namespace NUMINAMATH_GPT_simple_interest_rate_l187_18753

theorem simple_interest_rate (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ) (h1 : T = 4) (h2 : SI = P / 5) (h3 : SI = (P * R * T) / 100) : R = 5 := by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l187_18753


namespace NUMINAMATH_GPT_points_on_fourth_board_l187_18703

-- Definition of the points scored on each dartboard
def points_board_1 : ℕ := 30
def points_board_2 : ℕ := 38
def points_board_3 : ℕ := 41

-- Statement to prove that points on the fourth board are 34
theorem points_on_fourth_board : (points_board_1 + points_board_2) / 2 = 34 :=
by
  -- Given points on first and second boards
  have h1 : points_board_1 + points_board_2 = 68 := by rfl
  sorry

end NUMINAMATH_GPT_points_on_fourth_board_l187_18703


namespace NUMINAMATH_GPT_scientific_notation_correct_l187_18752

theorem scientific_notation_correct :
  0.000000007 = 7 * 10^(-9) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l187_18752


namespace NUMINAMATH_GPT_crayons_in_new_set_l187_18749

theorem crayons_in_new_set (initial_crayons : ℕ) (half_loss : ℕ) (total_after_purchase : ℕ) (initial_crayons_eq : initial_crayons = 18) (half_loss_eq : half_loss = initial_crayons / 2) (total_eq : total_after_purchase = 29) :
  total_after_purchase - (initial_crayons - half_loss) = 20 :=
by
  sorry

end NUMINAMATH_GPT_crayons_in_new_set_l187_18749


namespace NUMINAMATH_GPT_vector_parallel_x_is_neg1_l187_18772

variables (a b : ℝ × ℝ)
variable (x : ℝ)

def vectors_parallel : Prop := 
  (a = (1, -1)) ∧ (b = (x, 1)) ∧ (a.1 * b.2 - a.2 * b.1 = 0)

theorem vector_parallel_x_is_neg1 (h : vectors_parallel a b x) : x = -1 :=
sorry

end NUMINAMATH_GPT_vector_parallel_x_is_neg1_l187_18772


namespace NUMINAMATH_GPT_value_of_x2_plus_9y2_l187_18706

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x2_plus_9y2_l187_18706


namespace NUMINAMATH_GPT_min_voters_tall_giraffe_win_l187_18724

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end NUMINAMATH_GPT_min_voters_tall_giraffe_win_l187_18724


namespace NUMINAMATH_GPT_max_value_of_function_l187_18780

noncomputable def function_to_maximize (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + 1 / ((Real.sin x)^2 + (Real.cos x)^2 + 1)

theorem max_value_of_function :
  ∃ x : ℝ, function_to_maximize x = 7 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_function_l187_18780


namespace NUMINAMATH_GPT_alfonso_initial_money_l187_18742

def daily_earnings : ℕ := 6
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10
def cost_of_helmet : ℕ := 340

theorem alfonso_initial_money :
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  cost_of_helmet - total_earnings = 40 :=
by
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  show cost_of_helmet - total_earnings = 40
  sorry

end NUMINAMATH_GPT_alfonso_initial_money_l187_18742


namespace NUMINAMATH_GPT_cindy_pens_ratio_is_one_l187_18710

noncomputable def pens_owned_initial : ℕ := 25
noncomputable def pens_given_by_mike : ℕ := 22
noncomputable def pens_given_to_sharon : ℕ := 19
noncomputable def pens_owned_final : ℕ := 75

def pens_before_cindy (initial_pens mike_pens : ℕ) : ℕ := initial_pens + mike_pens
def pens_before_sharon (final_pens sharon_pens : ℕ) : ℕ := final_pens + sharon_pens
def pens_given_by_cindy (pens_before_sharon pens_before_cindy : ℕ) : ℕ := pens_before_sharon - pens_before_cindy
def ratio_pens_given_cindy (cindy_pens pens_before_cindy : ℕ) : ℚ := cindy_pens / pens_before_cindy

theorem cindy_pens_ratio_is_one :
    ratio_pens_given_cindy
        (pens_given_by_cindy (pens_before_sharon pens_owned_final pens_given_to_sharon)
                             (pens_before_cindy pens_owned_initial pens_given_by_mike))
        (pens_before_cindy pens_owned_initial pens_given_by_mike) = 1 := by
    sorry

end NUMINAMATH_GPT_cindy_pens_ratio_is_one_l187_18710


namespace NUMINAMATH_GPT_james_total_vegetables_l187_18779

def james_vegetable_count (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

theorem james_total_vegetables 
    (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
    a = 22 → b = 18 → c = 15 → d = 10 → e = 12 →
    james_vegetable_count a b c d e = 77 :=
by
  intros ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end NUMINAMATH_GPT_james_total_vegetables_l187_18779


namespace NUMINAMATH_GPT_no_real_solution_to_system_l187_18705

theorem no_real_solution_to_system :
  ∀ (x y z : ℝ), (x + y - 2 - 4 * x * y = 0) ∧
                 (y + z - 2 - 4 * y * z = 0) ∧
                 (z + x - 2 - 4 * z * x = 0) → false := 
by 
    intros x y z h
    rcases h with ⟨h1, h2, h3⟩
    -- Here would be the proof steps, which are omitted.
    sorry

end NUMINAMATH_GPT_no_real_solution_to_system_l187_18705


namespace NUMINAMATH_GPT_sequence_general_term_l187_18798

theorem sequence_general_term (a : ℕ → ℤ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, 1 < n → a n = 2 * (n + a (n - 1))) :
  ∀ n, 1 ≤ n → a n = 2 ^ (n + 2) - 2 * n - 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l187_18798


namespace NUMINAMATH_GPT_Mary_work_days_l187_18741

theorem Mary_work_days :
  ∀ (M : ℝ), (∀ R : ℝ, R = M / 1.30) → (R = 20) → M = 26 :=
by
  intros M h1 h2
  sorry

end NUMINAMATH_GPT_Mary_work_days_l187_18741


namespace NUMINAMATH_GPT_people_per_van_is_six_l187_18781

noncomputable def n_vans : ℝ := 6.0
noncomputable def n_buses : ℝ := 8.0
noncomputable def p_bus : ℝ := 18.0
noncomputable def people_difference : ℝ := 108

theorem people_per_van_is_six (x : ℝ) (h : n_buses * p_bus = n_vans * x + people_difference) : x = 6.0 := 
by
  sorry

end NUMINAMATH_GPT_people_per_van_is_six_l187_18781


namespace NUMINAMATH_GPT_natural_number_increased_by_one_l187_18740

theorem natural_number_increased_by_one (a : ℕ) 
  (h : (a + 1) ^ 2 - a ^ 2 = 1001) : 
  a = 500 := 
sorry

end NUMINAMATH_GPT_natural_number_increased_by_one_l187_18740


namespace NUMINAMATH_GPT_A_is_11_years_older_than_B_l187_18771

-- Define the constant B as given in the problem
def B : ℕ := 41

-- Define the condition based on the problem statement
def condition (A : ℕ) := A + 10 = 2 * (B - 10)

-- Prove the main statement that A is 11 years older than B
theorem A_is_11_years_older_than_B (A : ℕ) (h : condition A) : A - B = 11 :=
by
  sorry

end NUMINAMATH_GPT_A_is_11_years_older_than_B_l187_18771


namespace NUMINAMATH_GPT_rectangle_perimeter_l187_18783

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangle_perimeter (x y : ℝ) (A : ℝ) (E : ℝ) (fA fB : Real) (p : ℝ) 
  (h1 : y = 2 * x)
  (h2 : x * y = 2015)
  (h3 : E = 2006 * π)
  (h4 : fA = x + y)
  (h5 : fB ^ 2 = (3 / 2)^2 * 1007.5 - (p / 2)^2)
  (h6 : 2 * (3 / 2 * sqrt 1007.5 * sqrt 1009.375) = 2006 / π) :
  2 * (x + y) = 6 * sqrt 1007.5 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l187_18783


namespace NUMINAMATH_GPT_cost_of_country_cd_l187_18755

theorem cost_of_country_cd
  (cost_rock_cd : ℕ) (cost_pop_cd : ℕ) (cost_dance_cd : ℕ)
  (num_each : ℕ) (julia_has : ℕ) (julia_short : ℕ)
  (total_cost : ℕ) (total_other_cds : ℕ) (cost_country_cd : ℕ) :
  cost_rock_cd = 5 →
  cost_pop_cd = 10 →
  cost_dance_cd = 3 →
  num_each = 4 →
  julia_has = 75 →
  julia_short = 25 →
  total_cost = julia_has + julia_short →
  total_other_cds = num_each * cost_rock_cd + num_each * cost_pop_cd + num_each * cost_dance_cd →
  total_cost = total_other_cds + num_each * cost_country_cd →
  cost_country_cd = 7 :=
by
  intros cost_rock_cost_pop_cost_dance_num julia_diff 
         calc_total_total_other sub_total total_cds
  sorry

end NUMINAMATH_GPT_cost_of_country_cd_l187_18755


namespace NUMINAMATH_GPT_find_number_of_children_l187_18720

theorem find_number_of_children (N : ℕ) (B : ℕ) 
    (h1 : B = 2 * N) 
    (h2 : B = 4 * (N - 160)) 
    : N = 320 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_children_l187_18720


namespace NUMINAMATH_GPT_exponentiation_example_l187_18789

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_GPT_exponentiation_example_l187_18789


namespace NUMINAMATH_GPT_max_marks_l187_18745

theorem max_marks (M : ℝ) (h1 : 0.45 * M = 225) : M = 500 :=
by {
sorry
}

end NUMINAMATH_GPT_max_marks_l187_18745


namespace NUMINAMATH_GPT_inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l187_18701

theorem inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d
  (a b c d : ℚ) 
  (h1 : a * d > b * c) 
  (h2 : (a : ℚ) / b > (c : ℚ) / d) : 
  (a / b > (a + c) / (b + d)) ∧ ((a + c) / (b + d) > c / d) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l187_18701


namespace NUMINAMATH_GPT_stocks_higher_price_l187_18715

theorem stocks_higher_price (total_stocks lower_price higher_price: ℝ)
  (h_total: total_stocks = 8000)
  (h_ratio: higher_price = 1.5 * lower_price)
  (h_sum: lower_price + higher_price = total_stocks) :
  higher_price = 4800 :=
by
  sorry

end NUMINAMATH_GPT_stocks_higher_price_l187_18715


namespace NUMINAMATH_GPT_x729_minus_inverse_l187_18736

theorem x729_minus_inverse (x : ℂ) (h : x - x⁻¹ = 2 * Complex.I) : x ^ 729 - x⁻¹ ^ 729 = 2 * Complex.I := 
by 
  sorry

end NUMINAMATH_GPT_x729_minus_inverse_l187_18736


namespace NUMINAMATH_GPT_rosy_fish_count_l187_18790

theorem rosy_fish_count (L R T : ℕ) (hL : L = 10) (hT : T = 19) : R = T - L := by
  sorry

end NUMINAMATH_GPT_rosy_fish_count_l187_18790


namespace NUMINAMATH_GPT_greater_combined_area_l187_18719

noncomputable def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def combined_area (length : ℝ) (width : ℝ) : ℝ :=
  2 * (area_of_rectangle length width)

theorem greater_combined_area 
  (length1 width1 length2 width2 : ℝ)
  (h1 : length1 = 11) (h2 : width1 = 13)
  (h3 : length2 = 6.5) (h4 : width2 = 11) :
  combined_area length1 width1 - combined_area length2 width2 = 143 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_greater_combined_area_l187_18719


namespace NUMINAMATH_GPT_union_sets_l187_18759

-- Define the sets A and B using their respective conditions.
def A : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 4 < x ∧ x ≤ 10}

-- The theorem we aim to prove.
theorem union_sets : A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} := 
by
  sorry

end NUMINAMATH_GPT_union_sets_l187_18759


namespace NUMINAMATH_GPT_real_solutions_eq_l187_18778

def satisfies_equations (x y : ℝ) : Prop :=
  (4 * x + 5 * y = 13) ∧ (2 * x - 3 * y = 1)

theorem real_solutions_eq {x y : ℝ} : satisfies_equations x y ↔ (x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_GPT_real_solutions_eq_l187_18778


namespace NUMINAMATH_GPT_average_first_21_multiples_of_4_l187_18716

-- Define conditions
def n : ℕ := 21
def a1 : ℕ := 4
def an : ℕ := 4 * n
def sum_series (n a1 an : ℕ) : ℕ := (n * (a1 + an)) / 2

-- The problem statement in Lean 4
theorem average_first_21_multiples_of_4 : 
    (sum_series n a1 an) / n = 44 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_average_first_21_multiples_of_4_l187_18716


namespace NUMINAMATH_GPT_price_cashews_l187_18761

noncomputable def price_per_pound_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ) : ℝ := 
  (price_mixed_nuts_per_pound * weight_mixed_nuts - price_peanuts_per_pound * weight_peanuts) / weight_cashews

open Real

theorem price_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ)
  (h1 : price_mixed_nuts_per_pound = 2.50) 
  (h2 : weight_mixed_nuts = 100) 
  (h3 : weight_peanuts = 40) 
  (h4 : price_peanuts_per_pound = 3.50) 
  (h5 : weight_cashews = 60) : 
  price_per_pound_cashews price_mixed_nuts_per_pound weight_mixed_nuts weight_peanuts price_peanuts_per_pound weight_cashews = 11 / 6 := by 
  sorry

end NUMINAMATH_GPT_price_cashews_l187_18761


namespace NUMINAMATH_GPT_simplify_expression_l187_18738

variables {a b : ℝ}

-- Define the conditions
def condition (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a^4 + b^4 = a + b)

-- Define the target goal
def goal (a b : ℝ) : Prop := 
  (a / b + b / a - 1 / (a * b^2)) = (-a - b) / (a * b^2)

-- Statement of the theorem
theorem simplify_expression (h : condition a b) : goal a b :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l187_18738


namespace NUMINAMATH_GPT_sum_of_cubes_l187_18713

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l187_18713


namespace NUMINAMATH_GPT_average_age_of_two_women_is_30_l187_18702

-- Given definitions
def avg_age_before_replacement (A : ℝ) := 8 * A
def avg_age_after_increase (A : ℝ) := 8 * (A + 2)
def ages_of_men_replaced := 20 + 24

-- The theorem to prove: the average age of the two women is 30 years
theorem average_age_of_two_women_is_30 (A : ℝ) :
  (avg_age_after_increase A) - (avg_age_before_replacement A) = 16 →
  (ages_of_men_replaced + 16) / 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_two_women_is_30_l187_18702


namespace NUMINAMATH_GPT_fill_bathtub_time_l187_18757

def rate_cold_water : ℚ := 3 / 20
def rate_hot_water : ℚ := 1 / 8
def rate_drain : ℚ := 3 / 40
def net_rate : ℚ := rate_cold_water + rate_hot_water - rate_drain

theorem fill_bathtub_time :
  net_rate = 1/5 → (1 / net_rate) = 5 := by
  sorry

end NUMINAMATH_GPT_fill_bathtub_time_l187_18757


namespace NUMINAMATH_GPT_sales_tax_calculation_l187_18776

theorem sales_tax_calculation 
  (total_amount_paid : ℝ)
  (tax_rate : ℝ)
  (cost_tax_free : ℝ) :
  total_amount_paid = 30 → tax_rate = 0.08 → cost_tax_free = 12.72 → 
  (∃ sales_tax : ℝ, sales_tax = 1.28) :=
by
  intros H1 H2 H3
  sorry

end NUMINAMATH_GPT_sales_tax_calculation_l187_18776


namespace NUMINAMATH_GPT_number_of_chords_with_integer_length_l187_18770

theorem number_of_chords_with_integer_length 
(centerP_dist radius : ℝ) 
(h1 : centerP_dist = 12) 
(h2 : radius = 20) : 
  ∃ n : ℕ, n = 9 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_chords_with_integer_length_l187_18770


namespace NUMINAMATH_GPT_charge_per_call_proof_l187_18721

-- Define the conditions as given in the problem
def fixed_rental : ℝ := 350
def free_calls_per_month : ℕ := 200
def charge_per_call_exceed_200 (x : ℝ) (calls : ℕ) : ℝ := 
  if calls > 200 then (calls - 200) * x else 0

def charge_per_call_exceed_400 : ℝ := 1.6
def discount_rate : ℝ := 0.28
def february_calls : ℕ := 150
def march_calls : ℕ := 250
def march_discount (x : ℝ) : ℝ := x * (1 - discount_rate)
def total_march_charge (x : ℝ) : ℝ := 
  fixed_rental + charge_per_call_exceed_200 (march_discount x) march_calls

-- Prove the correct charge per call when calls exceed 200 per month
theorem charge_per_call_proof (x : ℝ) : 
  charge_per_call_exceed_200 x february_calls = 0 ∧ 
  total_march_charge x = fixed_rental + (march_calls - free_calls_per_month) * (march_discount x) → 
  x = x := 
by { 
  sorry 
}

end NUMINAMATH_GPT_charge_per_call_proof_l187_18721
