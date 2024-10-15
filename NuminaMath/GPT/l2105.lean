import Mathlib

namespace NUMINAMATH_GPT_speed_of_stream_l2105_210552

variable (v : ℝ)

theorem speed_of_stream (h : (64 / (24 + v)) = (32 / (24 - v))) : v = 8 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2105_210552


namespace NUMINAMATH_GPT_intersection_A_B_l2105_210507
-- Lean 4 code statement

def set_A : Set ℝ := {x | |x - 1| > 2}
def set_B : Set ℝ := {x | x * (x - 5) < 0}
def set_intersection : Set ℝ := {x | 3 < x ∧ x < 5}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2105_210507


namespace NUMINAMATH_GPT_total_jokes_proof_l2105_210520

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end NUMINAMATH_GPT_total_jokes_proof_l2105_210520


namespace NUMINAMATH_GPT_find_m_value_l2105_210506

-- Condition: P(-m^2, 3) lies on the axis of symmetry of the parabola y^2 = mx
def point_on_axis_of_symmetry (m : ℝ) : Prop :=
  let P := (-m^2, 3)
  let axis_of_symmetry := (-m / 4)
  P.1 = axis_of_symmetry

theorem find_m_value (m : ℝ) (h : point_on_axis_of_symmetry m) : m = 1 / 4 :=
  sorry

end NUMINAMATH_GPT_find_m_value_l2105_210506


namespace NUMINAMATH_GPT_rod_length_of_weight_l2105_210570

theorem rod_length_of_weight (w10 : ℝ) (wL : ℝ) (L : ℝ) (h1 : w10 = 23.4) (h2 : wL = 14.04) : L = 6 :=
by
  sorry

end NUMINAMATH_GPT_rod_length_of_weight_l2105_210570


namespace NUMINAMATH_GPT_inverse_function_evaluation_l2105_210582

def g (x : ℕ) : ℕ :=
  if x = 1 then 4
  else if x = 2 then 5
  else if x = 3 then 2
  else if x = 4 then 3
  else if x = 5 then 1
  else 0  -- default case, though it shouldn't be used given the conditions

noncomputable def g_inv (y : ℕ) : ℕ :=
  if y = 4 then 1
  else if y = 5 then 2
  else if y = 2 then 3
  else if y = 3 then 4
  else if y = 1 then 5
  else 0  -- default case, though it shouldn't be used given the conditions

theorem inverse_function_evaluation : g_inv (g_inv (g_inv 4)) = 2 := by
  sorry

end NUMINAMATH_GPT_inverse_function_evaluation_l2105_210582


namespace NUMINAMATH_GPT_loss_percentage_is_ten_l2105_210574

variable (CP SP SP_new : ℝ)  -- introduce the cost price, selling price, and new selling price as variables

theorem loss_percentage_is_ten
  (h1 : CP = 2000)
  (h2 : SP_new = CP + 80)
  (h3 : SP_new = SP + 280)
  (h4 : SP = CP - (L / 100 * CP)) : L = 10 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_loss_percentage_is_ten_l2105_210574


namespace NUMINAMATH_GPT_banana_group_size_l2105_210562

theorem banana_group_size (bananas groups : ℕ) (h1 : bananas = 407) (h2 : groups = 11) : bananas / groups = 37 :=
by sorry

end NUMINAMATH_GPT_banana_group_size_l2105_210562


namespace NUMINAMATH_GPT_tangerine_and_orange_percentage_l2105_210566

-- Given conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17
def initial_grapes := 12
def initial_kiwis := 7

def removed_oranges := 2
def removed_tangerines := 10
def removed_grapes := 4
def removed_kiwis := 3

def added_oranges := 3
def added_tangerines := 6

-- Computed values based on the initial conditions and changes
def remaining_apples := initial_apples
def remaining_oranges := initial_oranges - removed_oranges + added_oranges
def remaining_tangerines := initial_tangerines - removed_tangerines + added_tangerines
def remaining_grapes := initial_grapes - removed_grapes
def remaining_kiwis := initial_kiwis - removed_kiwis

def total_remaining_fruits := remaining_apples + remaining_oranges + remaining_tangerines + remaining_grapes + remaining_kiwis
def total_citrus_fruits := remaining_oranges + remaining_tangerines

-- Statement to prove
def citrus_percentage := (total_citrus_fruits : ℚ) / total_remaining_fruits * 100

theorem tangerine_and_orange_percentage : citrus_percentage = 47.5 := by
  sorry

end NUMINAMATH_GPT_tangerine_and_orange_percentage_l2105_210566


namespace NUMINAMATH_GPT_part1_part2_l2105_210588

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2105_210588


namespace NUMINAMATH_GPT_part_a_not_divisible_by_29_part_b_divisible_by_11_l2105_210583
open Nat

-- Part (a): Checking divisibility of 5641713 by 29
def is_divisible_by_29 (n : ℕ) : Prop :=
  n % 29 = 0

theorem part_a_not_divisible_by_29 : ¬is_divisible_by_29 5641713 :=
  by sorry

-- Part (b): Checking divisibility of 1379235 by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem part_b_divisible_by_11 : is_divisible_by_11 1379235 :=
  by sorry

end NUMINAMATH_GPT_part_a_not_divisible_by_29_part_b_divisible_by_11_l2105_210583


namespace NUMINAMATH_GPT_integer_multiplication_for_ones_l2105_210540

theorem integer_multiplication_for_ones :
  ∃ x : ℤ, (10^9 - 1) * x = (10^81 - 1) / 9 :=
by
  sorry

end NUMINAMATH_GPT_integer_multiplication_for_ones_l2105_210540


namespace NUMINAMATH_GPT_solve_for_x_l2105_210559

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2105_210559


namespace NUMINAMATH_GPT_find_range_of_m_l2105_210532

theorem find_range_of_m:
  (∀ x: ℝ, ¬ ∃ x: ℝ, x^2 + (m - 3) * x + 1 = 0) →
  (∀ y: ℝ, ¬ ∀ y: ℝ, x^2 + y^2 / (m - 1) = 1) → 
  1 < m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l2105_210532


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2105_210557

theorem boat_speed_in_still_water (v s : ℝ) (h1 : v + s = 15) (h2 : v - s = 7) : v = 11 := 
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2105_210557


namespace NUMINAMATH_GPT_factor_expression_l2105_210511

theorem factor_expression (x : ℤ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := 
by sorry

end NUMINAMATH_GPT_factor_expression_l2105_210511


namespace NUMINAMATH_GPT_intersection_A_B_l2105_210575

-- Define the sets A and the function f
def A : Set ℤ := {-2, 0, 2}
def f (x : ℤ) : ℤ := |x|

-- Define the set B as the image of A under the function f
def B : Set ℤ := {b | ∃ a ∈ A, f a = b}

-- State the property that every element in B has a pre-image in A
axiom B_has_preimage : ∀ b ∈ B, ∃ a ∈ A, f a = b

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {0, 2} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l2105_210575


namespace NUMINAMATH_GPT_total_revenue_correct_l2105_210567

noncomputable def total_ticket_revenue : ℕ :=
  let revenue_2pm := 180 * 6 + 20 * 5 + 60 * 4 + 20 * 3 + 20 * 5
  let revenue_5pm := 95 * 8 + 30 * 7 + 110 * 5 + 15 * 6
  let revenue_8pm := 122 * 10 + 74 * 7 + 29 * 8
  revenue_2pm + revenue_5pm + revenue_8pm

theorem total_revenue_correct : total_ticket_revenue = 5160 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l2105_210567


namespace NUMINAMATH_GPT_integer_square_root_35_consecutive_l2105_210555

theorem integer_square_root_35_consecutive : 
  ∃ n : ℕ, ∀ k : ℕ, n^2 ≤ k ∧ k < (n+1)^2 ∧ ((n + 1)^2 - n^2 = 35) ∧ (n = 17) := by 
  sorry

end NUMINAMATH_GPT_integer_square_root_35_consecutive_l2105_210555


namespace NUMINAMATH_GPT_emails_in_afternoon_l2105_210502

theorem emails_in_afternoon (A : ℕ) 
  (morning_emails : A + 3 = 10) : A = 7 :=
by {
    sorry
}

end NUMINAMATH_GPT_emails_in_afternoon_l2105_210502


namespace NUMINAMATH_GPT_polygon_at_least_9_sides_l2105_210578

theorem polygon_at_least_9_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ θ, θ < 45 ∧ (∀ j, 1 ≤ j ∧ j ≤ n → θ = 360 / n))):
  9 ≤ n :=
sorry

end NUMINAMATH_GPT_polygon_at_least_9_sides_l2105_210578


namespace NUMINAMATH_GPT_proof_MrLalandeInheritance_l2105_210596

def MrLalandeInheritance : Nat := 18000
def initialPayment : Nat := 3000
def monthlyInstallment : Nat := 2500
def numInstallments : Nat := 6

theorem proof_MrLalandeInheritance :
  initialPayment + numInstallments * monthlyInstallment = MrLalandeInheritance := 
by 
  sorry

end NUMINAMATH_GPT_proof_MrLalandeInheritance_l2105_210596


namespace NUMINAMATH_GPT_money_out_of_pocket_l2105_210591

theorem money_out_of_pocket
  (old_system_cost : ℝ)
  (trade_in_percent : ℝ)
  (new_system_cost : ℝ)
  (discount_percent : ℝ)
  (trade_in_value : ℝ)
  (discount_value : ℝ)
  (discounted_price : ℝ)
  (money_out_of_pocket : ℝ) :
  old_system_cost = 250 →
  trade_in_percent = 80 / 100 →
  new_system_cost = 600 →
  discount_percent = 25 / 100 →
  trade_in_value = old_system_cost * trade_in_percent →
  discount_value = new_system_cost * discount_percent →
  discounted_price = new_system_cost - discount_value →
  money_out_of_pocket = discounted_price - trade_in_value →
  money_out_of_pocket = 250 := by
  intros
  sorry

end NUMINAMATH_GPT_money_out_of_pocket_l2105_210591


namespace NUMINAMATH_GPT_initial_avg_production_is_50_l2105_210531

-- Define the initial conditions and parameters
variables (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55)

-- State that the initial total production over n days
def initial_total_production (A : ℝ) (n : ℕ) : ℝ := A * n

-- State the total production after today's production is added
def post_total_production (A : ℝ) (n : ℕ) (today_prod : ℝ) : ℝ := initial_total_production A n + today_prod

-- State the new average production calculation
def new_avg_production (n : ℕ) (new_avg : ℝ) : ℝ := new_avg * (n + 1)

-- State the main claim: Prove that the initial average daily production was 50 units per day
theorem initial_avg_production_is_50 (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55) 
  (h : post_total_production A n today_prod = new_avg_production n new_avg) : 
  A = 50 := 
by {
  -- Preliminary setups (we don't need detailed proof steps here)
  sorry
}

end NUMINAMATH_GPT_initial_avg_production_is_50_l2105_210531


namespace NUMINAMATH_GPT_domain_log_sin_sqrt_l2105_210518

theorem domain_log_sin_sqrt (x : ℝ) : 
  (2 < x ∧ x < (5 * Real.pi) / 3) ↔ 
  (∃ k : ℤ, (Real.pi / 3) + (4 * k * Real.pi) < x ∧ x < (5 * Real.pi / 3) + (4 * k * Real.pi) ∧ 2 < x) :=
by
  sorry

end NUMINAMATH_GPT_domain_log_sin_sqrt_l2105_210518


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_mixture_A_l2105_210549

theorem percentage_of_alcohol_in_mixture_A (x : ℝ) :
  (10 * x / 100 + 5 * 50 / 100 = 15 * 30 / 100) → x = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_mixture_A_l2105_210549


namespace NUMINAMATH_GPT_min_nSn_l2105_210504

theorem min_nSn 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (m : ℕ)
  (h1 : m ≥ 2)
  (h2 : S (m-1) = -2) 
  (h3 : S m = 0) 
  (h4 : S (m+1) = 3) : 
  ∃ n : ℕ, n * S n = -9 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_nSn_l2105_210504


namespace NUMINAMATH_GPT_range_of_a_l2105_210571

-- Problem statement and conditions definition
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def Q (a : ℝ) : Prop := (5 - 2 * a) > 1

-- Proof problem statement
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2105_210571


namespace NUMINAMATH_GPT_proof_average_l2105_210581

def average_two (x y : ℚ) : ℚ := (x + y) / 2
def average_three (x y z : ℚ) : ℚ := (x + y + z) / 3

theorem proof_average :
  average_three (2 * average_three 3 2 0) (average_two 0 3) (1 * 3) = 47 / 18 :=
by
  sorry

end NUMINAMATH_GPT_proof_average_l2105_210581


namespace NUMINAMATH_GPT_negation_P_eq_Q_l2105_210565

-- Define the proposition P: For any x ∈ ℝ, x^2 - 2x - 3 ≤ 0
def P : Prop := ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0

-- Define its negation which is the proposition Q
def Q : Prop := ∃ x : ℝ, x^2 - 2*x - 3 > 0

-- Prove that the negation of P is equivalent to Q
theorem negation_P_eq_Q : ¬P = Q :=
  by
  sorry

end NUMINAMATH_GPT_negation_P_eq_Q_l2105_210565


namespace NUMINAMATH_GPT_soda_difference_l2105_210543

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end NUMINAMATH_GPT_soda_difference_l2105_210543


namespace NUMINAMATH_GPT_factor_expression_l2105_210501

theorem factor_expression (a b c : ℝ) : 
  ( (a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4 ) / 
  ( (a - b)^4 + (b - c)^4 + (c - a)^4 ) = 1 := 
by sorry

end NUMINAMATH_GPT_factor_expression_l2105_210501


namespace NUMINAMATH_GPT_project_completion_l2105_210551

theorem project_completion (x : ℕ) :
  (21 - x) * (1 / 12 : ℚ) + x * (1 / 30 : ℚ) = 1 → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_project_completion_l2105_210551


namespace NUMINAMATH_GPT_unique_real_solution_l2105_210505

theorem unique_real_solution :
  ∃! x : ℝ, -((x + 2) ^ 2) ≥ 0 :=
sorry

end NUMINAMATH_GPT_unique_real_solution_l2105_210505


namespace NUMINAMATH_GPT_length_of_real_axis_l2105_210592

noncomputable def hyperbola_1 : Prop :=
  ∃ (x y: ℝ), (x^2 / 16) - (y^2 / 4) = 1

noncomputable def hyperbola_2 (a b: ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ (x y: ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def same_eccentricity (a b: ℝ) : Prop :=
  (1 + b^2 / a^2) = (1 + 1 / 4 / 16)

noncomputable def area_of_triangle (a b: ℝ) : Prop :=
  (a * b) = 32

theorem length_of_real_axis (a b: ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_1 ∧ hyperbola_2 a b ha hb ∧ same_eccentricity a b ∧ area_of_triangle a b →
  2 * a = 16 :=
by
  sorry

end NUMINAMATH_GPT_length_of_real_axis_l2105_210592


namespace NUMINAMATH_GPT_Eva_numbers_l2105_210554

theorem Eva_numbers : ∃ (a b : ℕ), a + b = 43 ∧ a - b = 15 ∧ a = 29 ∧ b = 14 :=
by
  sorry

end NUMINAMATH_GPT_Eva_numbers_l2105_210554


namespace NUMINAMATH_GPT_max_distance_between_bus_stops_l2105_210580

theorem max_distance_between_bus_stops 
  (v_m : ℝ) (v_b : ℝ) (dist : ℝ) 
  (h1 : v_m = v_b / 3) (h2 : dist = 2) : 
  ∀ d : ℝ, d = 1.5 := sorry

end NUMINAMATH_GPT_max_distance_between_bus_stops_l2105_210580


namespace NUMINAMATH_GPT_area_of_triangle_pqr_l2105_210584

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ :=
  let PQ := P + Q
  let PR := P + R
  let QR := Q + R
  if PQ^2 = PR^2 + QR^2 then
    1 / 2 * PR * QR
  else
    0

theorem area_of_triangle_pqr : 
  area_of_triangle 3 2 1 = 6 :=
by
  simp [area_of_triangle]
  sorry

end NUMINAMATH_GPT_area_of_triangle_pqr_l2105_210584


namespace NUMINAMATH_GPT_cloth_gain_representation_l2105_210521

theorem cloth_gain_representation (C S : ℝ) (h1 : S = 1.20 * C) (h2 : ∃ gain, gain = 60 * S - 60 * C) :
  ∃ meters : ℝ, meters = (60 * S - 60 * C) / S ∧ meters = 12 :=
by
  sorry

end NUMINAMATH_GPT_cloth_gain_representation_l2105_210521


namespace NUMINAMATH_GPT_consecutive_even_numbers_sum_is_3_l2105_210553

-- Definitions from the conditions provided
def consecutive_even_numbers := [80, 82, 84]
def sum_of_numbers := 246

-- The problem is to prove that there are 3 consecutive even numbers summing up to 246
theorem consecutive_even_numbers_sum_is_3 :
  (consecutive_even_numbers.sum = sum_of_numbers) → consecutive_even_numbers.length = 3 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_numbers_sum_is_3_l2105_210553


namespace NUMINAMATH_GPT_derivative_of_f_at_pi_over_2_l2105_210595

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -5 :=
sorry

end NUMINAMATH_GPT_derivative_of_f_at_pi_over_2_l2105_210595


namespace NUMINAMATH_GPT_find_angle_B_l2105_210525

-- Define the necessary trigonometric identities and dependencies
open Real

-- Declare the conditions under which we are working
theorem find_angle_B : 
  ∀ {a b A B : ℝ}, 
    a = 1 → 
    b = sqrt 3 → 
    A = π / 6 → 
    (B = π / 3 ∨ B = 2 * π / 3) := 
  by 
    intros a b A B ha hb hA
    sorry

end NUMINAMATH_GPT_find_angle_B_l2105_210525


namespace NUMINAMATH_GPT_part1_part2_l2105_210514

-- Define properties for the first part of the problem
def condition1 (weightA weightB : ℕ) : Prop :=
  weightA + weightB = 7500 ∧ weightA = 3 * weightB / 2

def question1_answer : Prop :=
  ∃ weightA weightB : ℕ, condition1 weightA weightB ∧ weightA = 4500 ∧ weightB = 3000

-- Combined condition for the second part of the problem scenarios
def condition2a (y : ℕ) : Prop := y ≤ 1800 ∧ 18 * y - 10 * y = 17400
def condition2b (y : ℕ) : Prop := 1800 < y ∧ y ≤ 3000 ∧ 18 * y - (15 * y - 9000) = 17400
def condition2c (y : ℕ) : Prop := y > 3000 ∧ 18 * y - (20 * y - 24000) = 17400

def question2_answer : Prop :=
  (∃ y : ℕ, condition2b y ∧ y = 2800) ∨ (∃ y : ℕ, condition2c y ∧ y = 3300)

-- The Lean statements for both parts of the problem
theorem part1 : question1_answer := sorry

theorem part2 : question2_answer := sorry

end NUMINAMATH_GPT_part1_part2_l2105_210514


namespace NUMINAMATH_GPT_intersection_of_A_B_l2105_210510

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_of_A_B (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {x : ℝ | 0 < x ∧ x < 3}) :
  A ∩ B = {1, 2} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_B_l2105_210510


namespace NUMINAMATH_GPT_lena_calculation_l2105_210593

def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + (10 - n % 10)

theorem lena_calculation :
  round_to_nearest_ten (63 + 2 * 29) = 120 :=
by
  sorry

end NUMINAMATH_GPT_lena_calculation_l2105_210593


namespace NUMINAMATH_GPT_linear_term_coefficient_is_neg_two_l2105_210516

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the specific quadratic equation
def specific_quadratic_eq (x : ℝ) : Prop :=
  quadratic_eq 1 (-2) (-1) x

-- The statement to prove the coefficient of the linear term
theorem linear_term_coefficient_is_neg_two : ∀ x : ℝ, specific_quadratic_eq x → ∀ a b c : ℝ, quadratic_eq a b c x → b = -2 :=
by
  intros x h_eq a b c h_quadratic_eq
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_linear_term_coefficient_is_neg_two_l2105_210516


namespace NUMINAMATH_GPT_base_8_to_base_10_4652_l2105_210546

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (4 * 8^3) + (6 * 8^2) + (5 * 8^1) + (2 * 8^0)

theorem base_8_to_base_10_4652 :
  convert_base_8_to_base_10 4652 = 2474 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_base_8_to_base_10_4652_l2105_210546


namespace NUMINAMATH_GPT_ratio_of_logs_l2105_210515

noncomputable def log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem ratio_of_logs (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : log_base 8 a = log_base 18 b)
    (h4 : log_base 18 b = log_base 32 (a + b)) : 
    b / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_logs_l2105_210515


namespace NUMINAMATH_GPT_centers_distance_ABC_l2105_210535

-- Define triangle ABC with the given properties
structure RightTriangle (ABC : Type) :=
(angle_A : ℝ)
(angle_C : ℝ)
(shorter_leg : ℝ)

-- Given: angle A is 30 degrees, angle C is 90 degrees, and shorter leg AC is 1
def triangle_ABC : RightTriangle ℝ := {
  angle_A := 30,
  angle_C := 90,
  shorter_leg := 1
}

-- Define the distance between the centers of the inscribed circles of triangles ACD and BCD
noncomputable def distance_between_centers (ABC : RightTriangle ℝ): ℝ :=
  sorry  -- placeholder for the actual proof

-- Example problem statement
theorem centers_distance_ABC (ABC : RightTriangle ℝ) (h_ABC : ABC = triangle_ABC) :
  distance_between_centers ABC = (Real.sqrt 3 - 1) / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_centers_distance_ABC_l2105_210535


namespace NUMINAMATH_GPT_marilyn_bottle_caps_start_l2105_210558

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end NUMINAMATH_GPT_marilyn_bottle_caps_start_l2105_210558


namespace NUMINAMATH_GPT_Mickey_less_than_twice_Minnie_l2105_210594

def Minnie_horses_per_day : ℕ := 10
def Mickey_horses_per_day : ℕ := 14

theorem Mickey_less_than_twice_Minnie :
  2 * Minnie_horses_per_day - Mickey_horses_per_day = 6 := by
  sorry

end NUMINAMATH_GPT_Mickey_less_than_twice_Minnie_l2105_210594


namespace NUMINAMATH_GPT_total_accidents_l2105_210599

noncomputable def A (k x : ℕ) : ℕ := 96 + k * x

theorem total_accidents :
  let k_morning := 1
  let k_evening := 3
  let x_morning := 2000
  let x_evening := 1000
  A k_morning x_morning + A k_evening x_evening = 5192 := by
  sorry

end NUMINAMATH_GPT_total_accidents_l2105_210599


namespace NUMINAMATH_GPT_correct_option_l2105_210508

def M : Set ℝ := { x | x^2 - 4 = 0 }

theorem correct_option : -2 ∈ M :=
by
  -- Definitions and conditions from the problem
  -- Set M is defined as the set of all x such that x^2 - 4 = 0
  have hM : M = { x | x^2 - 4 = 0 } := rfl
  -- Goal is to show that -2 belongs to the set M
  sorry

end NUMINAMATH_GPT_correct_option_l2105_210508


namespace NUMINAMATH_GPT_triangle_interior_angle_contradiction_l2105_210564

theorem triangle_interior_angle_contradiction :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A > 60 ∧ B > 60 ∧ C > 60 → false) :=
by
  sorry

end NUMINAMATH_GPT_triangle_interior_angle_contradiction_l2105_210564


namespace NUMINAMATH_GPT_total_sleep_per_week_l2105_210524

namespace TotalSleep

def hours_sleep_wd (days: Nat) : Nat := 6 * days
def hours_sleep_wknd (days: Nat) : Nat := 10 * days

theorem total_sleep_per_week : 
  hours_sleep_wd 5 + hours_sleep_wknd 2 = 50 := by
  sorry

end TotalSleep

end NUMINAMATH_GPT_total_sleep_per_week_l2105_210524


namespace NUMINAMATH_GPT_right_triangle_angles_ratio_l2105_210500

theorem right_triangle_angles_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3) :
  α = 67.5 ∧ β = 22.5 :=
sorry

end NUMINAMATH_GPT_right_triangle_angles_ratio_l2105_210500


namespace NUMINAMATH_GPT_negation_of_existential_prop_l2105_210512

theorem negation_of_existential_prop :
  (¬ ∃ (x₀ : ℝ), x₀^2 + x₀ + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_prop_l2105_210512


namespace NUMINAMATH_GPT_remainder_of_sum_l2105_210539

theorem remainder_of_sum :
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l2105_210539


namespace NUMINAMATH_GPT_jim_age_is_55_l2105_210550

-- Definitions of the conditions
def jim_age (t : ℕ) : ℕ := 3 * t + 10

def sum_ages (j t : ℕ) : Prop := j + t = 70

-- Statement of the proof problem
theorem jim_age_is_55 : ∃ t : ℕ, jim_age t = 55 ∧ sum_ages (jim_age t) t :=
by
  sorry

end NUMINAMATH_GPT_jim_age_is_55_l2105_210550


namespace NUMINAMATH_GPT_roots_of_equation_l2105_210586

theorem roots_of_equation :
  ∀ x : ℝ, (x^4 + x^2 - 20 = 0) ↔ (x = 2 ∨ x = -2) :=
by
  -- This will be the proof.
  -- We are claiming that x is a root of the polynomial if and only if x = 2 or x = -2.
  sorry

end NUMINAMATH_GPT_roots_of_equation_l2105_210586


namespace NUMINAMATH_GPT_min_value_of_expression_l2105_210563

theorem min_value_of_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * (a + c) = 4) : 2 * a + b + c ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2105_210563


namespace NUMINAMATH_GPT_river_depth_mid_July_l2105_210529

theorem river_depth_mid_July :
  let d_May := 5
  let d_June := d_May + 10
  let d_July := 3 * d_June
  d_July = 45 :=
by
  sorry

end NUMINAMATH_GPT_river_depth_mid_July_l2105_210529


namespace NUMINAMATH_GPT_problem_proof_l2105_210528

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end NUMINAMATH_GPT_problem_proof_l2105_210528


namespace NUMINAMATH_GPT_problem_statement_l2105_210576

theorem problem_statement (g : ℝ → ℝ) :
  (∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - x + 2) →
  (∃ m t : ℝ, m = 1 ∧ t = 3 ∧ m * t = 3) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2105_210576


namespace NUMINAMATH_GPT_cube_iff_diagonal_perpendicular_l2105_210503

-- Let's define the rectangular parallelepiped as a type
structure RectParallelepiped :=
-- Define the property of being a cube
(isCube : Prop)

-- Define the property q: any diagonal of the parallelepiped is perpendicular to the diagonal of its non-intersecting face
def diagonal_perpendicular (S : RectParallelepiped) : Prop := 
 sorry -- This depends on how you define diagonals and perpendicularity within the structure

-- Prove the biconditional relationship
theorem cube_iff_diagonal_perpendicular (S : RectParallelepiped) :
 S.isCube ↔ diagonal_perpendicular S :=
sorry

end NUMINAMATH_GPT_cube_iff_diagonal_perpendicular_l2105_210503


namespace NUMINAMATH_GPT_events_mutually_exclusive_not_complementary_l2105_210579

-- Define the set of balls and people
inductive Ball : Type
| b1 | b2 | b3 | b4

inductive Person : Type
| A | B | C | D

-- Define the event types
structure Event :=
  (p : Person)
  (b : Ball)

-- Define specific events as follows
def EventA : Event := { p := Person.A, b := Ball.b1 }
def EventB : Event := { p := Person.B, b := Ball.b1 }

-- We want to prove the relationship between two specific events:
-- "Person A gets ball number 1" and "Person B gets ball number 1"
-- Namely, that they are mutually exclusive but not complementary.

theorem events_mutually_exclusive_not_complementary :
  (∀ e : Event, (e = EventA → ¬ (e = EventB)) ∧ ¬ (e = EventA ∨ e = EventB)) :=
sorry

end NUMINAMATH_GPT_events_mutually_exclusive_not_complementary_l2105_210579


namespace NUMINAMATH_GPT_subset_S_A_inter_B_nonempty_l2105_210537

open Finset

-- Definitions of sets A and B
def A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def B : Finset ℕ := {4, 5, 6, 7, 8}

-- Definition of the subset S and its condition
def S : Finset ℕ := {5, 6}

-- The statement to be proved
theorem subset_S_A_inter_B_nonempty : S ⊆ A ∧ S ∩ B ≠ ∅ :=
by {
  sorry -- proof to be provided
}

end NUMINAMATH_GPT_subset_S_A_inter_B_nonempty_l2105_210537


namespace NUMINAMATH_GPT_cost_price_percentage_l2105_210556

theorem cost_price_percentage (CP SP : ℝ) (h1 : SP = 4 * CP) : (CP / SP) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l2105_210556


namespace NUMINAMATH_GPT_walking_time_l2105_210509

theorem walking_time (distance_walking_rate : ℕ) 
                     (distance : ℕ)
                     (rest_distance : ℕ) 
                     (rest_time : ℕ) 
                     (total_walking_time : ℕ) : 
  distance_walking_rate = 10 → 
  rest_distance = 10 → 
  rest_time = 7 → 
  distance = 50 → 
  total_walking_time = 328 → 
  total_walking_time = (distance / distance_walking_rate) * 60 + ((distance / rest_distance) - 1) * rest_time :=
by
  sorry

end NUMINAMATH_GPT_walking_time_l2105_210509


namespace NUMINAMATH_GPT_parameterized_line_solution_l2105_210590

theorem parameterized_line_solution :
  ∃ s l : ℝ, s = 1 / 2 ∧ l = -10 ∧
    ∀ t : ℝ, ∃ x y : ℝ,
      (x = -7 + t * l → y = s + t * (-5)) ∧ (y = (1 / 2) * x + 4) :=
by
  sorry

end NUMINAMATH_GPT_parameterized_line_solution_l2105_210590


namespace NUMINAMATH_GPT_example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l2105_210573

-- Define what it means to be a three-digit number using only two distinct digits
def two_digit_natural (d1 d2 : ℕ) (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = d1 ∨ d = d2) ∧ 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem example_of_four_three_digit_numbers_sum_2012_two_digits_exists :
  ∃ a b c d : ℕ, 
    two_digit_natural 3 5 a ∧
    two_digit_natural 3 5 b ∧
    two_digit_natural 3 5 c ∧
    two_digit_natural 3 5 d ∧
    a + b + c + d = 2012 :=
by
  sorry

end NUMINAMATH_GPT_example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l2105_210573


namespace NUMINAMATH_GPT_heights_inequality_l2105_210597

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) :
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) :=
by
  sorry

end NUMINAMATH_GPT_heights_inequality_l2105_210597


namespace NUMINAMATH_GPT_find_quotient_l2105_210519

-- Definitions for the variables and conditions
variables (D d q r : ℕ)

-- Conditions
axiom eq1 : D = q * d + r
axiom eq2 : D + 65 = q * (d + 5) + r

-- Theorem statement
theorem find_quotient : q = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l2105_210519


namespace NUMINAMATH_GPT_bob_total_earnings_l2105_210560

def hourly_rate_regular := 5
def hourly_rate_overtime := 6
def regular_hours_per_week := 40

def hours_worked_week1 := 44
def hours_worked_week2 := 48

def earnings_week1 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week1 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def earnings_week2 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week2 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def total_earnings : ℕ := earnings_week1 + earnings_week2

theorem bob_total_earnings : total_earnings = 472 := by
  sorry

end NUMINAMATH_GPT_bob_total_earnings_l2105_210560


namespace NUMINAMATH_GPT_initial_speed_is_correct_l2105_210522

def initial_speed (v : ℝ) : Prop :=
  let D_total : ℝ := 70 * 5
  let D_2 : ℝ := 85 * 2
  let D_1 := v * 3
  D_total = D_1 + D_2

theorem initial_speed_is_correct :
  ∃ v : ℝ, initial_speed v ∧ v = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_speed_is_correct_l2105_210522


namespace NUMINAMATH_GPT_binomial_multiplication_subtract_240_l2105_210536

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_multiplication_subtract_240 :
  binom 10 3 * binom 8 3 - 240 = 6480 :=
by
  sorry

end NUMINAMATH_GPT_binomial_multiplication_subtract_240_l2105_210536


namespace NUMINAMATH_GPT_maximize_product_l2105_210589

variable (x y : ℝ)
variable (h_xy_pos : x > 0 ∧ y > 0)
variable (h_sum : x + y = 35)

theorem maximize_product : x^5 * y^2 ≤ (25: ℝ)^5 * (10: ℝ)^2 :=
by
  -- Here we need to prove that the product x^5 y^2 is maximized at (x, y) = (25, 10)
  sorry

end NUMINAMATH_GPT_maximize_product_l2105_210589


namespace NUMINAMATH_GPT_smallest_positive_period_tan_l2105_210538

noncomputable def max_value (a b x : ℝ) := b + a * Real.sin x = -1
noncomputable def min_value (a b x : ℝ) := b - a * Real.sin x = -5
noncomputable def a_negative (a : ℝ) := a < 0

theorem smallest_positive_period_tan :
  ∃ (a b : ℝ), (max_value a b 0) ∧ (min_value a b 0) ∧ (a_negative a) →
  (1 / |3 * a + b|) * Real.pi = Real.pi / 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_tan_l2105_210538


namespace NUMINAMATH_GPT_height_percentage_difference_l2105_210568

theorem height_percentage_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.5384615384615385) :
  (H_B - H_A) / H_B * 100 = 35 := 
sorry

end NUMINAMATH_GPT_height_percentage_difference_l2105_210568


namespace NUMINAMATH_GPT_min_english_score_l2105_210561

theorem min_english_score (A B : ℕ) (h_avg_AB : (A + B) / 2 = 90) : 
  ∀ E : ℕ, ((A + B + E) / 3 ≥ 92) ↔ E ≥ 96 := by
  sorry

end NUMINAMATH_GPT_min_english_score_l2105_210561


namespace NUMINAMATH_GPT_locus_of_feet_of_perpendiculars_from_focus_l2105_210544

def parabola_locus (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = (p / 2) * x)

theorem locus_of_feet_of_perpendiculars_from_focus (p : ℝ) :
    parabola_locus p :=
by
  sorry

end NUMINAMATH_GPT_locus_of_feet_of_perpendiculars_from_focus_l2105_210544


namespace NUMINAMATH_GPT_find_y_l2105_210517

-- Define the sequence from 1 to 50
def seq_sum : ℕ := (50 * 51) / 2

-- Define y and the average condition
def average_condition (y : ℚ) : Prop :=
  (seq_sum + y) / 51 = 51 * y

-- Theorem statement
theorem find_y (y : ℚ) (h : average_condition y) : y = 51 / 104 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2105_210517


namespace NUMINAMATH_GPT_heap_holds_20_sheets_l2105_210587

theorem heap_holds_20_sheets :
  ∀ (num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets : ℕ),
    num_bundles = 3 →
    num_bunches = 2 →
    num_heaps = 5 →
    sheets_per_bundle = 2 →
    sheets_per_bunch = 4 →
    total_sheets = 114 →
    (total_sheets - (num_bundles * sheets_per_bundle + num_bunches * sheets_per_bunch)) / num_heaps = 20 := 
by
  intros num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end NUMINAMATH_GPT_heap_holds_20_sheets_l2105_210587


namespace NUMINAMATH_GPT_find_x_plus_y_of_parallel_vectors_l2105_210527

theorem find_x_plus_y_of_parallel_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (x, 2, -2)) 
  (hb : b = (2, y, 4)) 
  (h_parallel : ∃ k : ℝ, a = k • b) 
  : x + y = -5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_plus_y_of_parallel_vectors_l2105_210527


namespace NUMINAMATH_GPT_age_twice_of_father_l2105_210530

theorem age_twice_of_father (S M Y : ℕ) (h₁ : S = 22) (h₂ : M = S + 24) (h₃ : M + Y = 2 * (S + Y)) : Y = 2 := by
  sorry

end NUMINAMATH_GPT_age_twice_of_father_l2105_210530


namespace NUMINAMATH_GPT_project_completion_time_saving_l2105_210585

/-- A theorem stating that if a project with initial and additional workforce configuration,
the project will be completed 10 days ahead of schedule. -/
theorem project_completion_time_saving
  (total_days : ℕ := 100)
  (initial_people : ℕ := 10)
  (initial_days : ℕ := 30)
  (initial_fraction : ℚ := 1 / 5)
  (additional_people : ℕ := 10)
  : (total_days - ((initial_days + (1 / (initial_people + additional_people * initial_fraction)) * (total_days * initial_fraction) / initial_fraction)) = 10) :=
sorry

end NUMINAMATH_GPT_project_completion_time_saving_l2105_210585


namespace NUMINAMATH_GPT_find_p_l2105_210541

theorem find_p (m n p : ℝ)
  (h1 : m = 5 * n + 5)
  (h2 : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2105_210541


namespace NUMINAMATH_GPT_john_finishes_ahead_l2105_210577

noncomputable def InitialDistanceBehind : ℝ := 12
noncomputable def JohnSpeed : ℝ := 4.2
noncomputable def SteveSpeed : ℝ := 3.7
noncomputable def PushTime : ℝ := 28

theorem john_finishes_ahead :
  (JohnSpeed * PushTime - InitialDistanceBehind) - (SteveSpeed * PushTime) = 2 := by
  sorry

end NUMINAMATH_GPT_john_finishes_ahead_l2105_210577


namespace NUMINAMATH_GPT_Nell_initial_cards_l2105_210523

theorem Nell_initial_cards (given_away : ℕ) (now_has : ℕ) : 
  given_away = 276 → now_has = 252 → (now_has + given_away) = 528 :=
by
  intros h_given_away h_now_has
  sorry

end NUMINAMATH_GPT_Nell_initial_cards_l2105_210523


namespace NUMINAMATH_GPT_isosceles_trapezoid_perimeter_l2105_210547

/-- In an isosceles trapezoid ABCD with bases AB = 10 units and CD = 18 units, 
and height from AB to CD is 4 units, the perimeter of ABCD is 28 + 8 * sqrt(2) units. -/
theorem isosceles_trapezoid_perimeter :
  ∃ (A B C D : Type) (AB CD AD BC h : ℝ), 
      AB = 10 ∧ 
      CD = 18 ∧ 
      AD = BC ∧ 
      h = 4 →
      ∀ (P : ℝ), P = AB + BC + CD + DA → 
      P = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_perimeter_l2105_210547


namespace NUMINAMATH_GPT_range_of_m_for_second_quadrant_l2105_210545

theorem range_of_m_for_second_quadrant (m : ℝ) :
  (P : ℝ × ℝ) → P = (1 + m, 3) → P.fst < 0 → m < -1 :=
by
  intro P hP hQ
  sorry

end NUMINAMATH_GPT_range_of_m_for_second_quadrant_l2105_210545


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l2105_210572

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x + 2 * y + z = 1) : 
    x^2 + y^2 + z^2 ≥ (1 / 6) := 
  sorry

noncomputable def min_val_xy2z (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  if h_sq : x^2 + y^2 + z^2 = 1 / 6 then (x^2 + y^2 + z^2) else if x = 1 / 6 ∧ z = 1 / 6 ∧ y = 1 / 3 then 1 / 6 else (1 / 6)

example (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + y^2 + z^2 = min_val_xy2z x y z h :=
  sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l2105_210572


namespace NUMINAMATH_GPT_blackjack_payout_ratio_l2105_210548

theorem blackjack_payout_ratio (total_payout original_bet : ℝ) (h1 : total_payout = 60) (h2 : original_bet = 40):
  total_payout - original_bet = (1 / 2) * original_bet :=
by
  sorry

end NUMINAMATH_GPT_blackjack_payout_ratio_l2105_210548


namespace NUMINAMATH_GPT_largest_AC_value_l2105_210513

theorem largest_AC_value : ∃ (a b c d : ℕ), 
  a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (AC BD : ℝ), AC * BD = a * c + b * d ∧
  AC ^ 2 + BD ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ∧
  AC = Real.sqrt 458) :=
sorry

end NUMINAMATH_GPT_largest_AC_value_l2105_210513


namespace NUMINAMATH_GPT_find_value_of_y_l2105_210598

theorem find_value_of_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = 7) : y = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_y_l2105_210598


namespace NUMINAMATH_GPT_marching_band_members_l2105_210533

theorem marching_band_members :
  ∃ (n : ℕ), 100 < n ∧ n < 200 ∧
             n % 4 = 1 ∧
             n % 5 = 2 ∧
             n % 7 = 3 :=
  by sorry

end NUMINAMATH_GPT_marching_band_members_l2105_210533


namespace NUMINAMATH_GPT_find_a_plus_b_l2105_210569

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2105_210569


namespace NUMINAMATH_GPT_combined_sum_correct_l2105_210526

-- Define the sum of integers in a range
def sum_of_integers (a b : Int) : Int := (b - a + 1) * (a + b) / 2

-- Define the sum of squares of integers in a range
def sum_of_squares (a b : Int) : Int :=
  let sum_sq (n : Int) : Int := n * (n + 1) * (2 * n + 1) / 6
  sum_sq b - sum_sq (a - 1)

-- Define the combined sum function
def combined_sum (a b c d : Int) : Int :=
  sum_of_integers a b + sum_of_squares c d

-- Theorem statement: Prove the combined sum of integers from -50 to 40 and squares of integers from 10 to 40 is 21220
theorem combined_sum_correct :
  combined_sum (-50) 40 10 40 = 21220 :=
by
  -- leaving the proof as a sorry
  sorry

end NUMINAMATH_GPT_combined_sum_correct_l2105_210526


namespace NUMINAMATH_GPT_non_degenerate_ellipse_l2105_210534

theorem non_degenerate_ellipse (k : ℝ) : (∃ (x y : ℝ), x^2 + 4*y^2 - 10*x + 56*y = k) ↔ k > -221 :=
sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_l2105_210534


namespace NUMINAMATH_GPT_green_ball_probability_l2105_210542

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_green_ball_probability_l2105_210542
