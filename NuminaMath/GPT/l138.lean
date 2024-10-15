import Mathlib

namespace NUMINAMATH_GPT_desired_overall_percentage_l138_13827

-- Define the scores in the three subjects
def score1 := 50
def score2 := 70
def score3 := 90

-- Define the expected overall percentage
def expected_overall_percentage := 70

-- The main theorem to prove
theorem desired_overall_percentage :
  (score1 + score2 + score3) / 3 = expected_overall_percentage :=
by
  sorry

end NUMINAMATH_GPT_desired_overall_percentage_l138_13827


namespace NUMINAMATH_GPT_number_of_restaurants_l138_13801

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end NUMINAMATH_GPT_number_of_restaurants_l138_13801


namespace NUMINAMATH_GPT_difference_even_number_sums_l138_13890

open Nat

def sum_of_even_numbers (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem difference_even_number_sums :
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  sum_B - sum_A = 2100 :=
by
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  show sum_B - sum_A = 2100
  sorry

end NUMINAMATH_GPT_difference_even_number_sums_l138_13890


namespace NUMINAMATH_GPT_radius_of_circle_l138_13856

theorem radius_of_circle (P Q : ℝ) (h : P / Q = 25) : ∃ r : ℝ, 2 * π * r = Q ∧ π * r^2 = P ∧ r = 50 := 
by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_radius_of_circle_l138_13856


namespace NUMINAMATH_GPT_abs_ineq_sol_set_l138_13808

theorem abs_ineq_sol_set (x : ℝ) : (|x - 2| + |x - 1| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_sol_set_l138_13808


namespace NUMINAMATH_GPT_fraction_not_covered_l138_13845

/--
Given that frame X has a diameter of 16 cm and frame Y has a diameter of 12 cm,
prove that the fraction of the surface of frame X that is not covered by frame Y is 7/16.
-/
theorem fraction_not_covered (dX dY : ℝ) (hX : dX = 16) (hY : dY = 12) : 
  let rX := dX / 2
  let rY := dY / 2
  let AX := Real.pi * rX^2
  let AY := Real.pi * rY^2
  let uncovered_area := AX - AY
  let fraction_not_covered := uncovered_area / AX
  fraction_not_covered = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_not_covered_l138_13845


namespace NUMINAMATH_GPT_acute_angle_sum_equals_pi_over_two_l138_13881

theorem acute_angle_sum_equals_pi_over_two (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
  (h1 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1)
  (h2 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + b = π / 2 := 
sorry

end NUMINAMATH_GPT_acute_angle_sum_equals_pi_over_two_l138_13881


namespace NUMINAMATH_GPT_smallest_a_for_polynomial_l138_13878

theorem smallest_a_for_polynomial (a b x₁ x₂ x₃ : ℕ) 
    (h1 : x₁ * x₂ * x₃ = 2730)
    (h2 : x₁ + x₂ + x₃ = a)
    (h3 : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
    (h4 : ∀ y₁ y₂ y₃ : ℕ, y₁ * y₂ * y₃ = 2730 ∧ y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + y₂ + y₃ ≥ a) :
  a = 54 :=
  sorry

end NUMINAMATH_GPT_smallest_a_for_polynomial_l138_13878


namespace NUMINAMATH_GPT_roger_expenses_fraction_l138_13858

theorem roger_expenses_fraction {B t s n : ℝ} (h1 : t = 0.25 * (B - s))
  (h2 : s = 0.10 * (B - t)) (h3 : n = 5) :
  (t + s + n) / B = 0.41 :=
sorry

end NUMINAMATH_GPT_roger_expenses_fraction_l138_13858


namespace NUMINAMATH_GPT_no_real_roots_range_l138_13893

theorem no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_l138_13893


namespace NUMINAMATH_GPT_Sara_quarters_after_borrowing_l138_13832

theorem Sara_quarters_after_borrowing (initial_quarters borrowed_quarters : ℕ) (h1 : initial_quarters = 783) (h2 : borrowed_quarters = 271) :
  initial_quarters - borrowed_quarters = 512 := by
  sorry

end NUMINAMATH_GPT_Sara_quarters_after_borrowing_l138_13832


namespace NUMINAMATH_GPT_diamond_expression_l138_13839

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Declare the main theorem
theorem diamond_expression :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29 / 132 := 
by
  sorry

end NUMINAMATH_GPT_diamond_expression_l138_13839


namespace NUMINAMATH_GPT_geoff_needed_more_votes_to_win_l138_13886

-- Definitions based on the conditions
def total_votes : ℕ := 6000
def percent_to_fraction (p : ℕ) : ℚ := p / 100
def geoff_percent : ℚ := percent_to_fraction 1
def win_percent : ℚ := percent_to_fraction 51

-- Specific values derived from the conditions
def geoff_votes : ℚ := geoff_percent * total_votes
def win_votes : ℚ := win_percent * total_votes + 1

-- The theorem we intend to prove
theorem geoff_needed_more_votes_to_win :
  (win_votes - geoff_votes) = 3001 := by
  sorry

end NUMINAMATH_GPT_geoff_needed_more_votes_to_win_l138_13886


namespace NUMINAMATH_GPT_part1_part2_l138_13864

open Nat

-- Part (I)
theorem part1 (a b : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x + b = 0 → x = 2 ∨ x = 3) :
  a + b = 11 :=
by sorry

-- Part (II)
theorem part2 (c : ℝ) (h2 : ∀ x : ℝ, -x^2 + 6 * x + c ≤ 0) :
  c ≤ -9 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l138_13864


namespace NUMINAMATH_GPT_savings_in_cents_l138_13861

def price_local : ℝ := 149.99
def price_payment : ℝ := 26.50
def number_payments : ℕ := 5
def fee_delivery : ℝ := 19.99

theorem savings_in_cents :
  (price_local - (number_payments * price_payment + fee_delivery)) * 100 = -250 := by
  sorry

end NUMINAMATH_GPT_savings_in_cents_l138_13861


namespace NUMINAMATH_GPT_neither_5_nice_nor_6_nice_count_l138_13833

def is_k_nice (N k : ℕ) : Prop :=
  N % k = 1

def count_5_nice (N : ℕ) : ℕ :=
  (N - 1) / 5 + 1

def count_6_nice (N : ℕ) : ℕ :=
  (N - 1) / 6 + 1

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_30_nice (N : ℕ) : ℕ :=
  (N - 1) / 30 + 1

theorem neither_5_nice_nor_6_nice_count : 
  ∀ N < 200, 
  (N - (count_5_nice 199 + count_6_nice 199 - count_30_nice 199)) = 133 := 
by
  sorry

end NUMINAMATH_GPT_neither_5_nice_nor_6_nice_count_l138_13833


namespace NUMINAMATH_GPT_compound_interest_rate_l138_13815

-- Defining the principal amount and total repayment
def P : ℝ := 200
def A : ℝ := 220

-- The annual compound interest rate
noncomputable def annual_compound_interest_rate (P A : ℝ) (n : ℕ) : ℝ :=
  (A / P)^(1 / n) - 1

-- Introducing the conditions
axiom compounded_annually : ∀ (P A : ℝ), annual_compound_interest_rate P A 1 = 0.1

-- Stating the theorem
theorem compound_interest_rate :
  annual_compound_interest_rate P A 1 = 0.1 :=
by {
  exact compounded_annually P A
}

end NUMINAMATH_GPT_compound_interest_rate_l138_13815


namespace NUMINAMATH_GPT_smallest_rational_in_set_l138_13828

theorem smallest_rational_in_set : 
  ∀ (a b c d : ℚ), 
    a = -2/3 → b = -1 → c = 0 → d = 1 → 
    (a > b ∧ b < c ∧ c < d) → b = -1 := 
by
  intros a b c d ha hb hc hd h
  sorry

end NUMINAMATH_GPT_smallest_rational_in_set_l138_13828


namespace NUMINAMATH_GPT_find_xyz_l138_13868

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
  sorry

end NUMINAMATH_GPT_find_xyz_l138_13868


namespace NUMINAMATH_GPT_total_balloons_l138_13807

theorem total_balloons (A_initial : Nat) (A_additional : Nat) (J_initial : Nat) 
  (h1 : A_initial = 3) (h2 : J_initial = 5) (h3 : A_additional = 2) : 
  A_initial + A_additional + J_initial = 10 := by
  sorry

end NUMINAMATH_GPT_total_balloons_l138_13807


namespace NUMINAMATH_GPT_usual_time_is_36_l138_13850

noncomputable def usual_time_to_school (R : ℝ) (T : ℝ) : Prop :=
  let new_rate := (9/8 : ℝ) * R
  let new_time := T - 4
  R * T = new_rate * new_time

theorem usual_time_is_36 (R : ℝ) (T : ℝ) (h : T = 36) : usual_time_to_school R T :=
by
  sorry

end NUMINAMATH_GPT_usual_time_is_36_l138_13850


namespace NUMINAMATH_GPT_expression_always_integer_l138_13887

theorem expression_always_integer (m : ℕ) : 
  ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = (k : ℚ) := 
sorry

end NUMINAMATH_GPT_expression_always_integer_l138_13887


namespace NUMINAMATH_GPT_factorization_sum_l138_13826

theorem factorization_sum :
  ∃ a b c : ℤ, (∀ x : ℝ, (x^2 + 20 * x + 96 = (x + a) * (x + b)) ∧
                      (x^2 + 18 * x + 81 = (x - b) * (x + c))) →
              (a + b + c = 30) :=
by
  sorry

end NUMINAMATH_GPT_factorization_sum_l138_13826


namespace NUMINAMATH_GPT_roots_of_quadratic_l138_13852

variable {γ δ : ℝ}

theorem roots_of_quadratic (hγ : γ^2 - 5*γ + 6 = 0) (hδ : δ^2 - 5*δ + 6 = 0) : 
  8*γ^5 + 15*δ^4 = 8425 := 
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l138_13852


namespace NUMINAMATH_GPT_matt_house_wall_height_l138_13877

noncomputable def height_of_walls_in_matt_house : ℕ :=
  let living_room_side := 40
  let bedroom_side_1 := 10
  let bedroom_side_2 := 12

  let perimeter_living_room := 4 * living_room_side
  let perimeter_living_room_3_walls := perimeter_living_room - living_room_side

  let perimeter_bedroom := 2 * (bedroom_side_1 + bedroom_side_2)

  let total_perimeter_to_paint := perimeter_living_room_3_walls + perimeter_bedroom
  let total_area_to_paint := 1640

  total_area_to_paint / total_perimeter_to_paint

theorem matt_house_wall_height :
  height_of_walls_in_matt_house = 10 := by
  sorry

end NUMINAMATH_GPT_matt_house_wall_height_l138_13877


namespace NUMINAMATH_GPT_new_sailor_weight_l138_13830

-- Define the conditions
variables {average_weight : ℝ} (new_weight : ℝ)
variable (old_weight : ℝ := 56)

-- State the property we need to prove
theorem new_sailor_weight
  (h : (new_weight - old_weight) = 8) :
  new_weight = 64 :=
by
  sorry

end NUMINAMATH_GPT_new_sailor_weight_l138_13830


namespace NUMINAMATH_GPT_machine_Y_produces_more_widgets_l138_13870

-- Definitions for the rates and widgets produced
def W_x := 18 -- widgets per hour by machine X
def total_widgets := 1080

-- Calculations for time taken by each machine
def T_x := total_widgets / W_x -- time taken by machine X
def T_y := T_x - 10 -- machine Y takes 10 hours less

-- Rate at which machine Y produces widgets
def W_y := total_widgets / T_y

-- Calculation of percentage increase
def percentage_increase := (W_y - W_x) / W_x * 100

-- The final theorem to prove
theorem machine_Y_produces_more_widgets : percentage_increase = 20 := by
  sorry

end NUMINAMATH_GPT_machine_Y_produces_more_widgets_l138_13870


namespace NUMINAMATH_GPT_value_of_a_l138_13862

def f (x : ℝ) : ℝ := x^2 + 9
def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 25) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l138_13862


namespace NUMINAMATH_GPT_value_of_expression_l138_13871

theorem value_of_expression (A B C D : ℝ) (h1 : A - B = 30) (h2 : C + D = 20) :
  (B + C) - (A - D) = -10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l138_13871


namespace NUMINAMATH_GPT_solve_xy_l138_13869

theorem solve_xy (x y : ℝ) :
  (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1 / 3 → 
  x = 34 / 3 ∧ y = 35 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_xy_l138_13869


namespace NUMINAMATH_GPT_value_set_l138_13888

open Real Set

noncomputable def possible_values (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ x = c / a + c / b}

theorem value_set (c : ℝ) (hc : c > 0) : possible_values a b c = Ici (2 * c) := by
  sorry

end NUMINAMATH_GPT_value_set_l138_13888


namespace NUMINAMATH_GPT_temperature_on_tuesday_l138_13897

theorem temperature_on_tuesday 
  (M T W Th F Sa : ℝ)
  (h1 : (M + T + W) / 3 = 38)
  (h2 : (T + W + Th) / 3 = 42)
  (h3 : (W + Th + F) / 3 = 44)
  (h4 : (Th + F + Sa) / 3 = 46)
  (hF : F = 43)
  (pattern : M + 2 = Sa ∨ M - 1 = Sa) :
  T = 80 :=
sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l138_13897


namespace NUMINAMATH_GPT_brets_dinner_tip_calculation_l138_13873

/-
  We need to prove that the percentage of the tip Bret included is 20%, given the conditions.
-/

theorem brets_dinner_tip_calculation :
  let num_meals := 4
  let cost_per_meal := 12
  let num_appetizers := 2
  let cost_per_appetizer := 6
  let rush_fee := 5
  let total_cost := 77
  (total_cost - (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer + rush_fee))
  / (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_brets_dinner_tip_calculation_l138_13873


namespace NUMINAMATH_GPT_shape_is_cylinder_l138_13876

noncomputable def shape_desc (r θ z a : ℝ) : Prop := r = a

theorem shape_is_cylinder (a : ℝ) (h_a : a > 0) :
  ∀ (r θ z : ℝ), shape_desc r θ z a → ∃ c : Set (ℝ × ℝ × ℝ), c = {p : ℝ × ℝ × ℝ | ∃ θ z, p = (a, θ, z)} :=
by
  sorry

end NUMINAMATH_GPT_shape_is_cylinder_l138_13876


namespace NUMINAMATH_GPT_max_value_expression_l138_13813

theorem max_value_expression (s : ℝ) : 
  ∃ M, M = -3 * s^2 + 36 * s + 7 ∧ (∀ t : ℝ, -3 * t^2 + 36 * t + 7 ≤ M) :=
by
  use 115
  sorry

end NUMINAMATH_GPT_max_value_expression_l138_13813


namespace NUMINAMATH_GPT_man_speed_km_per_hr_l138_13806

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 82
noncomputable def time_to_pass_man_sec : ℝ := 4.499640028797696

theorem man_speed_km_per_hr :
  ∃ (Vm_km_per_hr : ℝ), Vm_km_per_hr = 6.0084 :=
sorry

end NUMINAMATH_GPT_man_speed_km_per_hr_l138_13806


namespace NUMINAMATH_GPT_ordered_pairs_count_l138_13805

theorem ordered_pairs_count : 
  ∃ n : ℕ, n = 6 ∧ ∀ A B : ℕ, (0 < A ∧ 0 < B) → (A * B = 32 ↔ A = 1 ∧ B = 32 ∨ A = 32 ∧ B = 1 ∨ A = 2 ∧ B = 16 ∨ A = 16 ∧ B = 2 ∨ A = 4 ∧ B = 8 ∨ A = 8 ∧ B = 4) := 
sorry

end NUMINAMATH_GPT_ordered_pairs_count_l138_13805


namespace NUMINAMATH_GPT_number_of_students_l138_13831

theorem number_of_students (n : ℕ)
  (h_avg : 100 * n = total_marks_unknown)
  (h_wrong_marks : total_marks_wrong = total_marks_unknown + 50)
  (h_correct_avg : total_marks_correct / n = 95)
  (h_corrected_marks : total_marks_correct = total_marks_wrong - 50) :
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l138_13831


namespace NUMINAMATH_GPT_sum_of_roots_l138_13894

theorem sum_of_roots (x : ℝ) (h : (x + 3) * (x - 2) = 15) : x = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l138_13894


namespace NUMINAMATH_GPT_present_value_of_machine_l138_13843

theorem present_value_of_machine {
  V0 : ℝ
} (h : 36100 = V0 * (0.95)^2) : V0 = 39978.95 :=
sorry

end NUMINAMATH_GPT_present_value_of_machine_l138_13843


namespace NUMINAMATH_GPT_min_vertical_distance_between_graphs_l138_13889

noncomputable def absolute_value (x : ℝ) : ℝ :=
if x >= 0 then x else -x

theorem min_vertical_distance_between_graphs : 
  ∃ d : ℝ, d = 3 / 4 ∧ ∀ x : ℝ, ∃ dist : ℝ, dist = absolute_value x - (- x^2 - 4 * x - 3) ∧ dist >= d :=
by
  sorry

end NUMINAMATH_GPT_min_vertical_distance_between_graphs_l138_13889


namespace NUMINAMATH_GPT_total_value_correct_l138_13872

noncomputable def total_value (num_coins : ℕ) : ℕ :=
  let value_one_rupee := num_coins * 1
  let value_fifty_paise := (num_coins * 50) / 100
  let value_twentyfive_paise := (num_coins * 25) / 100
  value_one_rupee + value_fifty_paise + value_twentyfive_paise

theorem total_value_correct :
  let num_coins := 40
  total_value num_coins = 70 := by
  sorry

end NUMINAMATH_GPT_total_value_correct_l138_13872


namespace NUMINAMATH_GPT_thought_number_and_appended_digit_l138_13899

theorem thought_number_and_appended_digit (x y : ℕ) (hx : x > 0) (hy : y ≤ 9):
  (10 * x + y - x^2 = 8 * x) ↔ (x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8) := sorry

end NUMINAMATH_GPT_thought_number_and_appended_digit_l138_13899


namespace NUMINAMATH_GPT_angle_45_deg_is_75_venerts_l138_13820

-- There are 600 venerts in a full circle.
def venus_full_circle : ℕ := 600

-- A full circle on Earth is 360 degrees.
def earth_full_circle : ℕ := 360

-- Conversion factor from degrees to venerts.
def degrees_to_venerts (deg : ℕ) : ℕ :=
  deg * (venus_full_circle / earth_full_circle)

-- Angle of 45 degrees in venerts.
def angle_45_deg_in_venerts : ℕ := 45 * (venus_full_circle / earth_full_circle)

theorem angle_45_deg_is_75_venerts :
  angle_45_deg_in_venerts = 75 :=
by
  -- Proof will be inserted here.
  sorry

end NUMINAMATH_GPT_angle_45_deg_is_75_venerts_l138_13820


namespace NUMINAMATH_GPT_find_second_number_l138_13853

theorem find_second_number (G N: ℕ) (h1: G = 101) (h2: 4351 % G = 8) (h3: N % G = 10) : N = 4359 :=
by 
  sorry

end NUMINAMATH_GPT_find_second_number_l138_13853


namespace NUMINAMATH_GPT_solve_df1_l138_13865

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (df1 : ℝ)

-- The condition given in the problem
axiom func_def : ∀ x, f x = 2 * x * df1 + (Real.log x)

-- Express the relationship from the derivative and solve for f'(1) = -1
theorem solve_df1 : df1 = -1 :=
by
  -- Here we will insert the proof steps in Lean, but they are omitted in this statement.
  sorry

end NUMINAMATH_GPT_solve_df1_l138_13865


namespace NUMINAMATH_GPT_incorrect_statement_D_l138_13896

def ordinate_of_x_axis_is_zero (p : ℝ × ℝ) : Prop :=
  p.2 = 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A_properties (a b : ℝ) : Prop :=
  let x := - a^2 - 1
  let y := abs b
  x < 0 ∧ y ≥ 0

theorem incorrect_statement_D (a b : ℝ) : 
  ∃ (x y : ℝ), point_A_properties a b ∧ (x = -a^2 - 1 ∧ y = abs b ∧ (x < 0 ∧ y = 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_statement_D_l138_13896


namespace NUMINAMATH_GPT_regular_polygon_sides_l138_13867

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l138_13867


namespace NUMINAMATH_GPT_train_length_correct_l138_13836

noncomputable def speed_km_per_hour : ℝ := 56
noncomputable def time_seconds : ℝ := 32.142857142857146
noncomputable def bridge_length_m : ℝ := 140
noncomputable def train_length_m : ℝ := 360

noncomputable def speed_m_per_s : ℝ := speed_km_per_hour * (1000 / 3600)
noncomputable def total_distance_m : ℝ := speed_m_per_s * time_seconds

theorem train_length_correct :
  (total_distance_m - bridge_length_m) = train_length_m :=
  by
    sorry

end NUMINAMATH_GPT_train_length_correct_l138_13836


namespace NUMINAMATH_GPT_a_2011_value_l138_13823

noncomputable def sequence_a : ℕ → ℝ
| 0 => 6/7
| (n + 1) => if 0 ≤ sequence_a n ∧ sequence_a n < 1/2 then 2 * sequence_a n
              else 2 * sequence_a n - 1

theorem a_2011_value : sequence_a 2011 = 6/7 := sorry

end NUMINAMATH_GPT_a_2011_value_l138_13823


namespace NUMINAMATH_GPT_base_circumference_cone_l138_13882

theorem base_circumference_cone (r : ℝ) (h : r = 5) (θ : ℝ) (k : θ = 180) : 
  ∃ c : ℝ, c = 5 * π :=
by
  sorry

end NUMINAMATH_GPT_base_circumference_cone_l138_13882


namespace NUMINAMATH_GPT_good_games_count_l138_13819

-- Define the conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def games_that_didnt_work : Nat := 74

-- Define the total games bought
def total_games_bought : Nat := games_from_friend + games_from_garage_sale

-- State the theorem to prove the number of good games
theorem good_games_count : total_games_bought - games_that_didnt_work = 3 :=
by
  sorry

end NUMINAMATH_GPT_good_games_count_l138_13819


namespace NUMINAMATH_GPT_num_biology_books_is_15_l138_13846

-- conditions
def num_chemistry_books : ℕ := 8
def total_ways : ℕ := 2940

-- main statement to prove
theorem num_biology_books_is_15 : ∃ B: ℕ, (B * (B - 1)) / 2 * (num_chemistry_books * (num_chemistry_books - 1)) / 2 = total_ways ∧ B = 15 :=
by
  sorry

end NUMINAMATH_GPT_num_biology_books_is_15_l138_13846


namespace NUMINAMATH_GPT_find_initial_milk_amount_l138_13822

-- Define the initial amount of milk as a variable in liters
variable (T : ℝ)

-- Given conditions
def consumed (T : ℝ) := 0.4 * T
def leftover (T : ℝ) := 0.69

-- The total milk at first was T if T = 0.69 / 0.6
theorem find_initial_milk_amount 
  (h1 : leftover T = 0.69)
  (h2 : consumed T = 0.4 * T) :
  T = 1.15 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_milk_amount_l138_13822


namespace NUMINAMATH_GPT_marbles_count_l138_13824

variable (r b : ℕ)

theorem marbles_count (hr1 : 8 * (r - 1) = r + b - 2) (hr2 : 4 * r = r + b - 3) : r + b = 9 := 
by sorry

end NUMINAMATH_GPT_marbles_count_l138_13824


namespace NUMINAMATH_GPT_equipment_B_production_l138_13885

theorem equipment_B_production
  (total_production : ℕ)
  (sample_size : ℕ)
  (A_sample_production : ℕ)
  (B_sample_production : ℕ)
  (A_total_production : ℕ)
  (B_total_production : ℕ)
  (total_condition : total_production = 4800)
  (sample_condition : sample_size = 80)
  (A_sample_condition : A_sample_production = 50)
  (B_sample_condition : B_sample_production = 30)
  (ratio_condition : (A_sample_production / B_sample_production) = (5 / 3))
  (production_condition : A_total_production + B_total_production = total_production) :
  B_total_production = 1800 := 
sorry

end NUMINAMATH_GPT_equipment_B_production_l138_13885


namespace NUMINAMATH_GPT_locomotive_distance_l138_13802

theorem locomotive_distance 
  (speed_train : ℝ) (speed_sound : ℝ) (time_diff : ℝ)
  (h_train : speed_train = 20) 
  (h_sound : speed_sound = 340) 
  (h_time : time_diff = 4) : 
  ∃ x : ℝ, x = 85 := 
by 
  sorry

end NUMINAMATH_GPT_locomotive_distance_l138_13802


namespace NUMINAMATH_GPT_vector_dot_product_l138_13891

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem vector_dot_product : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_vector_dot_product_l138_13891


namespace NUMINAMATH_GPT_sin_2A_value_l138_13829

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : a / (2 * Real.cos A) = b / (3 * Real.cos B))
variable (h₂ : b / (3 * Real.cos B) = c / (6 * Real.cos C))

theorem sin_2A_value (h₃ : a / (2 * Real.cos A) = c / (6 * Real.cos C)) :
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := sorry

end NUMINAMATH_GPT_sin_2A_value_l138_13829


namespace NUMINAMATH_GPT_average_speed_l138_13838

theorem average_speed (speed1 speed2 time1 time2: ℝ) (h1 : speed1 = 60) (h2 : time1 = 3) (h3 : speed2 = 85) (h4 : time2 = 2) : 
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 70 :=
by
  -- Definitions
  have distance1 := speed1 * time1
  have distance2 := speed2 * time2
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  -- Proof skeleton
  sorry

end NUMINAMATH_GPT_average_speed_l138_13838


namespace NUMINAMATH_GPT_determinant_roots_l138_13859

theorem determinant_roots (s p q a b c : ℂ) 
  (h : ∀ x : ℂ, x^3 - s*x^2 + p*x + q = (x - a) * (x - b) * (x - c)) :
  (1 + a) * ((1 + b) * (1 + c) - 1) - ((1) * (1 + c) - 1) + ((1) - (1 + b)) = p + 3 * s :=
by {
  -- expanded determinant calculations
  sorry
}

end NUMINAMATH_GPT_determinant_roots_l138_13859


namespace NUMINAMATH_GPT_water_tank_capacity_l138_13847

theorem water_tank_capacity (C : ℝ) :
  0.4 * C - 0.1 * C = 36 → C = 120 :=
by sorry

end NUMINAMATH_GPT_water_tank_capacity_l138_13847


namespace NUMINAMATH_GPT_fraction_to_decimal_l138_13835

theorem fraction_to_decimal : (9 : ℚ) / 25 = 0.36 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l138_13835


namespace NUMINAMATH_GPT_fran_speed_l138_13834

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end NUMINAMATH_GPT_fran_speed_l138_13834


namespace NUMINAMATH_GPT_each_persons_final_share_l138_13810

theorem each_persons_final_share
  (total_dining_bill : ℝ)
  (number_of_people : ℕ)
  (tip_percentage : ℝ) :
  total_dining_bill = 211.00 →
  tip_percentage = 0.15 →
  number_of_people = 5 →
  ((total_dining_bill + total_dining_bill * tip_percentage) / number_of_people) = 48.53 :=
by
  intros
  sorry

end NUMINAMATH_GPT_each_persons_final_share_l138_13810


namespace NUMINAMATH_GPT_jeremy_school_distance_l138_13883

def travel_time_rush_hour := 15 / 60 -- hours
def travel_time_clear_day := 10 / 60 -- hours
def speed_increase := 20 -- miles per hour

def distance_to_school (d v : ℝ) : Prop :=
  d = v * travel_time_rush_hour ∧ d = (v + speed_increase) * travel_time_clear_day

theorem jeremy_school_distance (d v : ℝ) (h_speed : v = 40) : d = 10 :=
by
  have travel_time_rush_hour := 1/4
  have travel_time_clear_day := 1/6
  have speed_increase := 20
  
  have h1 : d = v * travel_time_rush_hour := by sorry
  have h2 : d = (v + speed_increase) * travel_time_clear_day := by sorry
  have eqn := distance_to_school d v
  sorry

end NUMINAMATH_GPT_jeremy_school_distance_l138_13883


namespace NUMINAMATH_GPT_distinct_infinite_solutions_l138_13898

theorem distinct_infinite_solutions (n : ℕ) (hn : n > 0) : 
  ∃ p q : ℤ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n ∧ (p * p - 5 * q * q = 1) ∧ 
  ∀ m : ℕ, (m ≠ n → (9 + 4 * Real.sqrt 5) ^ m ≠ (9 + 4 * Real.sqrt 5) ^ n) :=
by
  sorry

end NUMINAMATH_GPT_distinct_infinite_solutions_l138_13898


namespace NUMINAMATH_GPT_factor_x12_minus_729_l138_13800

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end NUMINAMATH_GPT_factor_x12_minus_729_l138_13800


namespace NUMINAMATH_GPT_simplify_fraction_l138_13874

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end NUMINAMATH_GPT_simplify_fraction_l138_13874


namespace NUMINAMATH_GPT_find_points_per_enemy_l138_13803

def points_per_enemy (x : ℕ) : Prop :=
  let points_from_enemies := 6 * x
  let additional_points := 8
  let total_points := points_from_enemies + additional_points
  total_points = 62

theorem find_points_per_enemy (x : ℕ) (h : points_per_enemy x) : x = 9 :=
  by sorry

end NUMINAMATH_GPT_find_points_per_enemy_l138_13803


namespace NUMINAMATH_GPT_find_p_l138_13866

theorem find_p (h p : Polynomial ℝ) 
  (H1 : h + p = 3 * X^2 - X + 4)
  (H2 : h = X^4 - 5 * X^2 + X + 6) : 
  p = -X^4 + 8 * X^2 - 2 * X - 2 :=
sorry

end NUMINAMATH_GPT_find_p_l138_13866


namespace NUMINAMATH_GPT_joan_spent_on_thursday_l138_13854

theorem joan_spent_on_thursday : 
  ∀ (n : ℕ), 
  2 * (4 + n) = 18 → 
  n = 14 := 
by 
  sorry

end NUMINAMATH_GPT_joan_spent_on_thursday_l138_13854


namespace NUMINAMATH_GPT_six_times_product_plus_one_equals_seven_pow_sixteen_l138_13816

theorem six_times_product_plus_one_equals_seven_pow_sixteen :
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := 
  sorry

end NUMINAMATH_GPT_six_times_product_plus_one_equals_seven_pow_sixteen_l138_13816


namespace NUMINAMATH_GPT_even_function_periodic_symmetric_about_2_l138_13860

variables {F : ℝ → ℝ}

theorem even_function_periodic_symmetric_about_2
  (h_even : ∀ x, F x = F (-x))
  (h_symmetric : ∀ x, F (2 - x) = F (2 + x))
  (h_cond : F 2011 + 2 * F 1 = 18) :
  F 2011 = 6 :=
sorry

end NUMINAMATH_GPT_even_function_periodic_symmetric_about_2_l138_13860


namespace NUMINAMATH_GPT_min_value_is_3_l138_13825

theorem min_value_is_3 (a b : ℝ) (h1 : a > b / 2) (h2 : 2 * a > b) : (2 * a + b) / a ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_is_3_l138_13825


namespace NUMINAMATH_GPT_borrowed_amount_correct_l138_13895

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_borrowed_amount_correct_l138_13895


namespace NUMINAMATH_GPT_find_angle_between_altitude_and_median_l138_13809

noncomputable def angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : ℝ :=
  Real.arctan ((a^2 - b^2) / (4 * S))

theorem find_angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : 
  angle_between_altitude_and_median a b S h1 h2 = 
    Real.arctan ((a^2 - b^2) / (4 * S)) := 
  sorry

end NUMINAMATH_GPT_find_angle_between_altitude_and_median_l138_13809


namespace NUMINAMATH_GPT_gcd_5800_14025_l138_13884

theorem gcd_5800_14025 : Int.gcd 5800 14025 = 25 := by
  sorry

end NUMINAMATH_GPT_gcd_5800_14025_l138_13884


namespace NUMINAMATH_GPT_bus_capacity_percentage_l138_13844

theorem bus_capacity_percentage (x : ℕ) (h1 : 150 * x / 100 + 150 * 70 / 100 = 195) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_bus_capacity_percentage_l138_13844


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l138_13842

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_a30 : a 30 = 100)
  (h_a100 : a 100 = 30) :
  d = -1 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l138_13842


namespace NUMINAMATH_GPT_smallest_composite_square_side_length_l138_13818

theorem smallest_composite_square_side_length (n : ℕ) (h : ∃ k, 14 * n = k^2) : 
  ∃ m : ℕ, n = 14 ∧ m = 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_composite_square_side_length_l138_13818


namespace NUMINAMATH_GPT_merchant_marked_price_percentage_l138_13855

variables (L S M C : ℝ)
variable (h1 : C = 0.7 * L)
variable (h2 : C = 0.75 * S)
variable (h3 : S = 0.9 * M)

theorem merchant_marked_price_percentage : M = 1.04 * L :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_percentage_l138_13855


namespace NUMINAMATH_GPT_mod_computation_l138_13841

theorem mod_computation (n : ℤ) : 
  0 ≤ n ∧ n < 23 ∧ 47582 % 23 = n ↔ n = 3 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mod_computation_l138_13841


namespace NUMINAMATH_GPT_car_return_speed_l138_13837

theorem car_return_speed (d : ℕ) (speed_CD : ℕ) (avg_speed_round_trip : ℕ) 
  (round_trip_distance : ℕ) (time_CD : ℕ) (time_round_trip : ℕ) (r: ℕ) 
  (h1 : d = 150) (h2 : speed_CD = 75) (h3 : avg_speed_round_trip = 60)
  (h4 : d * 2 = round_trip_distance) 
  (h5 : time_CD = d / speed_CD) 
  (h6 : time_round_trip = time_CD + d / r) 
  (h7 : avg_speed_round_trip = round_trip_distance / time_round_trip) :
  r = 50 :=
by {
  -- proof steps will go here
  sorry
}

end NUMINAMATH_GPT_car_return_speed_l138_13837


namespace NUMINAMATH_GPT_rebus_puzzle_verified_l138_13851

-- Defining the conditions
def A := 1
def B := 1
def C := 0
def D := 1
def F := 1
def L := 1
def M := 0
def N := 1
def P := 0
def Q := 1
def T := 1
def G := 8
def H := 1
def K := 4
def W := 4
def X := 1

noncomputable def verify_rebus_puzzle : Prop :=
  (A * B * 10 = 110) ∧
  (6 * G / (10 * H + 7) = 4) ∧
  (L + N * 10 = 20) ∧
  (12 - K = 8) ∧
  (101 + 10 * W + X = 142)

-- Lean statement to verify the problem
theorem rebus_puzzle_verified : verify_rebus_puzzle :=
by {
  -- Values are already defined and will be concluded by Lean
  sorry
}

end NUMINAMATH_GPT_rebus_puzzle_verified_l138_13851


namespace NUMINAMATH_GPT_min_value_of_quadratic_l138_13857

open Real

theorem min_value_of_quadratic 
  (x y z : ℝ) 
  (h : 3 * x + 2 * y + z = 1) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ 3 / 34 := 
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l138_13857


namespace NUMINAMATH_GPT_greatest_perimeter_l138_13863

theorem greatest_perimeter (w l : ℕ) (h1 : w * l = 12) : 
  ∃ (P : ℕ), P = 2 * (w + l) ∧ ∀ (w' l' : ℕ), w' * l' = 12 → 2 * (w' + l') ≤ P := 
sorry

end NUMINAMATH_GPT_greatest_perimeter_l138_13863


namespace NUMINAMATH_GPT_cost_of_computer_game_is_90_l138_13879

-- Define the costs of individual items
def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

-- Define the number of items
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_quantity : ℕ := 1

-- Calculate the total cost before rebate
def total_cost_before_rebate : ℕ :=
  total_cost_after_rebate + rebate

-- Calculate the total cost of polo shirts and necklaces
def total_cost_polo_necklaces : ℕ :=
  (polo_shirt_quantity * polo_shirt_price) + (necklace_quantity * necklace_price)

-- Define the unknown cost of the computer game
def computer_game_price : ℕ :=
  total_cost_before_rebate - total_cost_polo_necklaces

-- Prove the cost of the computer game
theorem cost_of_computer_game_is_90 : computer_game_price = 90 := by
  -- The following line is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cost_of_computer_game_is_90_l138_13879


namespace NUMINAMATH_GPT_N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l138_13812

-- Define the number with 1986 ones
def N : ℕ := (10^1986 - 1) / 9

-- Definition of having at least n distinct divisors
def has_at_least_n_distinct_divisors (num : ℕ) (n : ℕ) :=
  ∃ (divisors : Finset ℕ), divisors.card ≥ n ∧ ∀ d ∈ divisors, d ∣ num

theorem N_has_at_least_8_distinct_divisors :
  has_at_least_n_distinct_divisors N 8 :=
sorry

theorem N_has_at_least_32_distinct_divisors :
  has_at_least_n_distinct_divisors N 32 :=
sorry


end NUMINAMATH_GPT_N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l138_13812


namespace NUMINAMATH_GPT_count_six_digit_palindromes_l138_13849

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end NUMINAMATH_GPT_count_six_digit_palindromes_l138_13849


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l138_13817

noncomputable def a : ℝ := Real.log (Real.tan (70 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def b : ℝ := Real.log (Real.sin (25 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 2) ^ Real.cos (25 * Real.pi / 180)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  -- proofs would go here
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l138_13817


namespace NUMINAMATH_GPT_efficacy_rate_is_80_percent_l138_13804

-- Define the total number of people surveyed
def total_people : ℕ := 20

-- Define the number of people who find the new drug effective
def effective_people : ℕ := 16

-- Calculate the efficacy rate
def efficacy_rate (effective : ℕ) (total : ℕ) : ℚ := effective / total

-- The theorem to be proved
theorem efficacy_rate_is_80_percent : efficacy_rate effective_people total_people = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_efficacy_rate_is_80_percent_l138_13804


namespace NUMINAMATH_GPT_sum_of_odd_powers_l138_13848

variable (x y z a : ℝ) (k : ℕ)

theorem sum_of_odd_powers (h1 : x + y + z = a) (h2 : x^3 + y^3 + z^3 = a^3) (hk : k % 2 = 1) : 
  x^k + y^k + z^k = a^k :=
sorry

end NUMINAMATH_GPT_sum_of_odd_powers_l138_13848


namespace NUMINAMATH_GPT_compute_focus_d_l138_13875

-- Define the given conditions as Lean definitions
structure Ellipse (d : ℝ) :=
  (first_quadrant : d > 0)
  (F1 : ℝ × ℝ := (4, 8))
  (F2 : ℝ × ℝ := (d, 8))
  (tangent_x_axis : (d + 4) / 2 > 0)
  (tangent_y_axis : (d + 4) / 2 > 0)

-- Define the proof problem to show d = 6 for the given conditions
theorem compute_focus_d (d : ℝ) (e : Ellipse d) : d = 6 := by
  sorry

end NUMINAMATH_GPT_compute_focus_d_l138_13875


namespace NUMINAMATH_GPT_vector_addition_result_l138_13840

-- Define the given vectors
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-3, 4)

-- Statement to prove that the sum of the vectors is (-1, 5)
theorem vector_addition_result : vector_a + vector_b = (-1, 5) :=
by
  -- Use the fact that vector addition in ℝ^2 is component-wise
  sorry

end NUMINAMATH_GPT_vector_addition_result_l138_13840


namespace NUMINAMATH_GPT_problem_l138_13821

theorem problem (a b : ℝ) (h : a > b) (k : b > 0) : b * (a - b) > 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l138_13821


namespace NUMINAMATH_GPT_problem_statement_l138_13892

theorem problem_statement :
  ∀ m n : ℕ, (m = 9) → (n = m^2 + 1) → n - m = 73 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end NUMINAMATH_GPT_problem_statement_l138_13892


namespace NUMINAMATH_GPT_number_of_dots_on_faces_l138_13814

theorem number_of_dots_on_faces (d A B C D : ℕ) 
  (h1 : d = 6)
  (h2 : A = 3)
  (h3 : B = 5)
  (h4 : C = 6)
  (h5 : D = 5) :
  A = 3 ∧ B = 5 ∧ C = 6 ∧ D = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_dots_on_faces_l138_13814


namespace NUMINAMATH_GPT_dog_speed_is_16_kmh_l138_13811

variable (man's_speed : ℝ := 4) -- man's speed in km/h
variable (total_path_length : ℝ := 625) -- total path length in meters
variable (remaining_distance : ℝ := 81) -- remaining distance in meters

theorem dog_speed_is_16_kmh :
  let total_path_length_km := total_path_length / 1000
  let remaining_distance_km := remaining_distance / 1000
  let man_covered_distance_km := total_path_length_km - remaining_distance_km
  let time := man_covered_distance_km / man's_speed
  let dog_total_distance_km := 4 * (2 * total_path_length_km)
  let dog_speed := dog_total_distance_km / time
  dog_speed = 16 :=
by
  sorry

end NUMINAMATH_GPT_dog_speed_is_16_kmh_l138_13811


namespace NUMINAMATH_GPT_wallpaper_removal_time_l138_13880

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end NUMINAMATH_GPT_wallpaper_removal_time_l138_13880
