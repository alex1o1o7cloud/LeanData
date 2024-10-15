import Mathlib

namespace NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l1847_184794

-- Define the polynomial q(x) = D*x^4 + E*x^2 + F*x + 8
variable (D E F : ℝ)
def q (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- Given condition: q(2) = 12
axiom h1 : q D E F 2 = 12

-- Prove that q(-2) = 4
theorem remainder_when_divided_by_x_plus_2 : q D E F (-2) = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l1847_184794


namespace NUMINAMATH_GPT_difference_in_tiles_l1847_184737

theorem difference_in_tiles (n : ℕ) (hn : n = 9) : (n + 1)^2 - n^2 = 19 :=
by sorry

end NUMINAMATH_GPT_difference_in_tiles_l1847_184737


namespace NUMINAMATH_GPT_cost_per_liter_l1847_184763

/-
Given:
- Service cost per vehicle: $2.10
- Number of mini-vans: 3
- Number of trucks: 2
- Total cost: $299.1
- Mini-van's tank size: 65 liters
- Truck's tank is 120% bigger than a mini-van's tank
- All tanks are empty

Prove that the cost per liter of fuel is $0.60
-/

theorem cost_per_liter (service_cost_per_vehicle : ℝ) 
(number_of_minivans number_of_trucks : ℕ)
(total_cost : ℝ)
(minivan_tank_size : ℝ)
(truck_tank_multiplier : ℝ)
(fuel_cost : ℝ)
(total_fuel : ℝ) :
  service_cost_per_vehicle = 2.10 ∧
  number_of_minivans = 3 ∧
  number_of_trucks = 2 ∧
  total_cost = 299.1 ∧
  minivan_tank_size = 65 ∧
  truck_tank_multiplier = 1.2 ∧
  fuel_cost = (total_cost - (number_of_minivans + number_of_trucks) * service_cost_per_vehicle) ∧
  total_fuel = (number_of_minivans * minivan_tank_size + number_of_trucks * (minivan_tank_size * (1 + truck_tank_multiplier))) →
  (fuel_cost / total_fuel) = 0.60 :=
sorry

end NUMINAMATH_GPT_cost_per_liter_l1847_184763


namespace NUMINAMATH_GPT_max_product_of_real_roots_quadratic_eq_l1847_184773

theorem max_product_of_real_roots_quadratic_eq : ∀ (k : ℝ), (∃ x y : ℝ, 4 * x ^ 2 - 8 * x + k = 0 ∧ 4 * y ^ 2 - 8 * y + k = 0) 
    → k = 4 :=
sorry

end NUMINAMATH_GPT_max_product_of_real_roots_quadratic_eq_l1847_184773


namespace NUMINAMATH_GPT_percentage_increase_l1847_184768

theorem percentage_increase (A B x y : ℝ) (h1 : A / B = (5 * y^2) / (6 * x)) (h2 : 2 * x + 3 * y = 42) :  
  (B - A) / A * 100 = ((126 - 9 * y - 5 * y^2) / (5 * y^2)) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1847_184768


namespace NUMINAMATH_GPT_proof_problem_l1847_184798

open Set

def Point : Type := ℝ × ℝ

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def area_of_triangle (T : Triangle) : ℝ :=
   0.5 * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

def area_of_grid (length width : ℝ) : ℝ :=
   length * width

def problem_statement : Prop :=
   let T : Triangle := {A := (1,3), B := (5,1), C := (4,4)} 
   let S1 := area_of_triangle T
   let S := area_of_grid 6 5
   (S1 / S) = 1 / 6

theorem proof_problem : problem_statement := 
by
  sorry


end NUMINAMATH_GPT_proof_problem_l1847_184798


namespace NUMINAMATH_GPT_difference_five_three_numbers_specific_number_condition_l1847_184709

def is_five_three_number (A : ℕ) : Prop :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a = 5 + c ∧ b = 3 + d

def M (A : ℕ) : ℕ :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a + c + 2 * (b + d)

def N (A : ℕ) : ℕ :=
  let b := (A % 1000) / 100
  b - 3

noncomputable def largest_five_three_number := 9946
noncomputable def smallest_five_three_number := 5300

theorem difference_five_three_numbers :
  largest_five_three_number - smallest_five_three_number = 4646 := by
  sorry

noncomputable def specific_five_three_number := 5401

theorem specific_number_condition {A : ℕ} (hA : is_five_three_number A) :
  (M A) % (N A) = 0 ∧ (M A) / (N A) % 5 = 0 → A = specific_five_three_number := by
  sorry

end NUMINAMATH_GPT_difference_five_three_numbers_specific_number_condition_l1847_184709


namespace NUMINAMATH_GPT_softball_players_count_l1847_184708

theorem softball_players_count :
  ∀ (cricket hockey football total_players softball : ℕ),
  cricket = 15 →
  hockey = 12 →
  football = 13 →
  total_players = 55 →
  total_players = cricket + hockey + football + softball →
  softball = 15 :=
by
  intros cricket hockey football total_players softball h_cricket h_hockey h_football h_total_players h_total
  sorry

end NUMINAMATH_GPT_softball_players_count_l1847_184708


namespace NUMINAMATH_GPT_larger_circle_radius_l1847_184734

theorem larger_circle_radius (r R : ℝ) 
  (h : (π * R^2) / (π * r^2) = 5 / 2) : 
  R = r * Real.sqrt 2.5 :=
sorry

end NUMINAMATH_GPT_larger_circle_radius_l1847_184734


namespace NUMINAMATH_GPT_Stonewall_marching_band_max_members_l1847_184753

theorem Stonewall_marching_band_max_members (n : ℤ) (h1 : 30 * n % 34 = 2) (h2 : 30 * n < 1500) : 30 * n = 1260 :=
by
  sorry

end NUMINAMATH_GPT_Stonewall_marching_band_max_members_l1847_184753


namespace NUMINAMATH_GPT_chairlift_halfway_l1847_184752

theorem chairlift_halfway (total_chairs current_chair halfway_chair : ℕ) 
  (h_total_chairs : total_chairs = 96)
  (h_current_chair : current_chair = 66) : halfway_chair = 18 :=
sorry

end NUMINAMATH_GPT_chairlift_halfway_l1847_184752


namespace NUMINAMATH_GPT_halfway_between_3_4_and_5_7_l1847_184745

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end NUMINAMATH_GPT_halfway_between_3_4_and_5_7_l1847_184745


namespace NUMINAMATH_GPT_lcm_factor_is_one_l1847_184786

theorem lcm_factor_is_one
  (A B : ℕ)
  (hcf : A.gcd B = 42)
  (larger_A : A = 588)
  (other_factor : ∃ X, A.lcm B = 42 * X * 14) :
  ∃ X, X = 1 :=
  sorry

end NUMINAMATH_GPT_lcm_factor_is_one_l1847_184786


namespace NUMINAMATH_GPT_find_ordered_triple_l1847_184790

theorem find_ordered_triple :
  ∃ (a b c : ℝ), a > 2 ∧ b > 2 ∧ c > 2 ∧
    (a + b + c = 30) ∧
    ( (a = 13) ∧ (b = 11) ∧ (c = 6) ) ∧
    ( ( ( (a + 3)^2 / (b + c - 3) ) + ( (b + 5)^2 / (c + a - 5) ) + ( (c + 7)^2 / (a + b - 7) ) = 45 ) ) :=
sorry

end NUMINAMATH_GPT_find_ordered_triple_l1847_184790


namespace NUMINAMATH_GPT_quadrilateral_sides_equal_l1847_184739

theorem quadrilateral_sides_equal (a b c d : ℕ) (h1 : a ∣ b + c + d) (h2 : b ∣ a + c + d) (h3 : c ∣ a + b + d) (h4 : d ∣ a + b + c) : a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end NUMINAMATH_GPT_quadrilateral_sides_equal_l1847_184739


namespace NUMINAMATH_GPT_line_equation_exists_l1847_184799

noncomputable def P : ℝ × ℝ := (-2, 5)
noncomputable def m : ℝ := -3 / 4

theorem line_equation_exists (x y : ℝ) : 
  (y - 5 = -3 / 4 * (x + 2)) ↔ (3 * x + 4 * y - 14 = 0) := 
by 
  sorry

end NUMINAMATH_GPT_line_equation_exists_l1847_184799


namespace NUMINAMATH_GPT_eric_return_home_time_l1847_184774

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end NUMINAMATH_GPT_eric_return_home_time_l1847_184774


namespace NUMINAMATH_GPT_tan_alpha_proof_l1847_184715

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_proof_l1847_184715


namespace NUMINAMATH_GPT_linear_regression_passes_through_centroid_l1847_184736

noncomputable def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a + b * x

theorem linear_regression_passes_through_centroid 
  (a b : ℝ) (x_bar y_bar : ℝ) 
  (h_centroid : ∀ (x y : ℝ), (x = x_bar ∧ y = y_bar) → y = linear_regression a b x) :
  linear_regression a b x_bar = y_bar :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_linear_regression_passes_through_centroid_l1847_184736


namespace NUMINAMATH_GPT_chord_length_of_concentric_circles_l1847_184771

theorem chord_length_of_concentric_circles 
  (R r : ℝ) (h1 : R^2 - r^2 = 15) (h2 : ∀ s, s = 2 * R) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ ∀ x, x = c := 
by 
  sorry

end NUMINAMATH_GPT_chord_length_of_concentric_circles_l1847_184771


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1847_184789

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 75) (h₂ : b = 100) : ∃ c, c = 125 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1847_184789


namespace NUMINAMATH_GPT_prime_product_is_2009_l1847_184725

theorem prime_product_is_2009 (a b c : ℕ) 
  (h_primeA : Prime a) 
  (h_primeB : Prime b) 
  (h_primeC : Prime c)
  (h_div1 : a ∣ (b + 8)) 
  (h_div2a : a ∣ (b^2 - 1)) 
  (h_div2c : c ∣ (b^2 - 1)) 
  (h_sum : b + c = a^2 - 1) : 
  a * b * c = 2009 := 
sorry

end NUMINAMATH_GPT_prime_product_is_2009_l1847_184725


namespace NUMINAMATH_GPT_factorize_x_cubed_minus_9x_l1847_184764

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end NUMINAMATH_GPT_factorize_x_cubed_minus_9x_l1847_184764


namespace NUMINAMATH_GPT_smallest_b_base_l1847_184749

theorem smallest_b_base :
  ∃ b : ℕ, b^2 ≤ 25 ∧ 25 < b^3 ∧ (∀ c : ℕ, c < b → ¬(c^2 ≤ 25 ∧ 25 < c^3)) :=
sorry

end NUMINAMATH_GPT_smallest_b_base_l1847_184749


namespace NUMINAMATH_GPT_solve_system_eq_l1847_184756

theorem solve_system_eq (x y z : ℝ) :
    (x^2 - y^2 + z = 64 / (x * y)) ∧
    (y^2 - z^2 + x = 64 / (y * z)) ∧
    (z^2 - x^2 + y = 64 / (x * z)) ↔ 
    (x = 4 ∧ y = 4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = -4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = 4 ∧ z = -4) ∨ 
    (x = 4 ∧ y = -4 ∧ z = -4) := by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1847_184756


namespace NUMINAMATH_GPT_statue_of_liberty_ratio_l1847_184772

theorem statue_of_liberty_ratio :
  let H_statue := 305 -- height in feet
  let H_model := 10 -- height in inches
  H_statue / H_model = 30.5 := 
by
  let H_statue := 305
  let H_model := 10
  sorry

end NUMINAMATH_GPT_statue_of_liberty_ratio_l1847_184772


namespace NUMINAMATH_GPT_negation_of_proposition_l1847_184701
open Real

theorem negation_of_proposition :
  ¬ (∃ x₀ : ℝ, (2/x₀) + log x₀ ≤ 0) ↔ ∀ x : ℝ, (2/x) + log x > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1847_184701


namespace NUMINAMATH_GPT_car_rental_cost_l1847_184714

theorem car_rental_cost
  (rent_per_day : ℝ) (cost_per_mile : ℝ) (days_rented : ℕ) (miles_driven : ℝ)
  (h1 : rent_per_day = 30)
  (h2 : cost_per_mile = 0.25)
  (h3 : days_rented = 5)
  (h4 : miles_driven = 500) :
  rent_per_day * days_rented + cost_per_mile * miles_driven = 275 := 
  by
  sorry

end NUMINAMATH_GPT_car_rental_cost_l1847_184714


namespace NUMINAMATH_GPT_train_distance_difference_l1847_184713

theorem train_distance_difference (t : ℝ) (D₁ D₂ : ℝ)
(h_speed1 : D₁ = 20 * t)
(h_speed2 : D₂ = 25 * t)
(h_total_dist : D₁ + D₂ = 540) :
  D₂ - D₁ = 60 :=
by {
  -- These are the conditions as stated in step c)
  sorry
}

end NUMINAMATH_GPT_train_distance_difference_l1847_184713


namespace NUMINAMATH_GPT_sqrt_twentyfive_eq_five_l1847_184704

theorem sqrt_twentyfive_eq_five : Real.sqrt 25 = 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_twentyfive_eq_five_l1847_184704


namespace NUMINAMATH_GPT_equal_intercepts_no_second_quadrant_l1847_184742

/- Given line equation (a + 1)x + y + 2 - a = 0 and a \in ℝ. -/
def line_eq (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/- If the line l has equal intercepts on both coordinate axes, 
   then a = 0 or a = 2. -/
theorem equal_intercepts (a : ℝ) :
  (∃ x y : ℝ, line_eq a x 0 ∧ line_eq a 0 y ∧ x = y) →
  a = 0 ∨ a = 2 :=
sorry

/- If the line l does not pass through the second quadrant,
   then a ≤ -1. -/
theorem no_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → ¬ line_eq a x y) →
  a ≤ -1 :=
sorry

end NUMINAMATH_GPT_equal_intercepts_no_second_quadrant_l1847_184742


namespace NUMINAMATH_GPT_find_monthly_income_l1847_184766

-- Given condition
def deposit : ℝ := 3400
def percentage : ℝ := 0.15

-- Goal: Prove Sheela's monthly income
theorem find_monthly_income : (deposit / percentage) = 22666.67 := by
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_find_monthly_income_l1847_184766


namespace NUMINAMATH_GPT_proof_problem_l1847_184722

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (x + 1) * f x

axiom domain_f : ∀ x : ℝ, true
axiom even_f : ∀ x : ℝ, f (2 * x - 1) = f (-(2 * x - 1))
axiom mono_g_neg_inf_minus_1 : ∀ x y : ℝ, x ≤ y → x ≤ -1 → y ≤ -1 → g x ≤ g y

-- Proof Problem Statement
theorem proof_problem :
  (∀ x y : ℝ, x ≤ y → -1 ≤ x → -1 ≤ y → g x ≤ g y) ∧
  (∀ a b : ℝ, g a + g b > 0 → a + b + 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1847_184722


namespace NUMINAMATH_GPT_fraction_multiplication_exponent_l1847_184777

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_exponent_l1847_184777


namespace NUMINAMATH_GPT_operation_eval_l1847_184762

def my_operation (a b : ℤ) := a * (b + 2) + a * (b + 1)

theorem operation_eval : my_operation 3 (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_operation_eval_l1847_184762


namespace NUMINAMATH_GPT_factor_expression_l1847_184750

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1847_184750


namespace NUMINAMATH_GPT_oil_bill_january_l1847_184727

theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 :=
by
  sorry

end NUMINAMATH_GPT_oil_bill_january_l1847_184727


namespace NUMINAMATH_GPT_diego_can_carry_home_l1847_184778

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end NUMINAMATH_GPT_diego_can_carry_home_l1847_184778


namespace NUMINAMATH_GPT_joe_first_lift_weight_l1847_184720

variable (x y : ℕ)

def joe_lift_conditions (x y : ℕ) : Prop :=
  x + y = 600 ∧ 2 * x = y + 300

theorem joe_first_lift_weight (x y : ℕ) (h : joe_lift_conditions x y) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_joe_first_lift_weight_l1847_184720


namespace NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l1847_184770

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) : 
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l1847_184770


namespace NUMINAMATH_GPT_question_d_not_true_l1847_184755

variable {a b c d : ℚ}

theorem question_d_not_true (h : a * b = c * d) : (a + 1) / (c + 1) ≠ (d + 1) / (b + 1) := 
sorry

end NUMINAMATH_GPT_question_d_not_true_l1847_184755


namespace NUMINAMATH_GPT_range_of_func_l1847_184706

noncomputable def func (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_func :
  (∀ y : ℝ, 
    (∃ x : ℝ, (x < 1 ∨ (2 ≤ x ∧ x < 5)) ∧ y = func x) ↔ 
    (y < 0 ∨ (1/4 < y ∧ y ≤ 1))) :=
by
  sorry

end NUMINAMATH_GPT_range_of_func_l1847_184706


namespace NUMINAMATH_GPT_alice_students_count_l1847_184726

variable (S : ℕ)
variable (students_with_own_vests := 0.20 * S)
variable (students_needing_vests := 0.80 * S)
variable (instructors : ℕ := 10)
variable (life_vests_on_hand : ℕ := 20)
variable (additional_life_vests_needed : ℕ := 22)
variable (total_life_vests_needed := life_vests_on_hand + additional_life_vests_needed)
variable (life_vests_needed_for_instructors := instructors)
variable (life_vests_needed_for_students := total_life_vests_needed - life_vests_needed_for_instructors)

theorem alice_students_count : S = 40 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_alice_students_count_l1847_184726


namespace NUMINAMATH_GPT_sequence_general_term_l1847_184754

namespace SequenceSum

def Sn (n : ℕ) : ℕ :=
  2 * n^2 + n

def a₁ (n : ℕ) : ℕ :=
  if n = 1 then Sn n else (Sn n - Sn (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  a₁ n = 4 * n - 1 :=
sorry

end SequenceSum

end NUMINAMATH_GPT_sequence_general_term_l1847_184754


namespace NUMINAMATH_GPT_find_missing_number_l1847_184775

theorem find_missing_number (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1847_184775


namespace NUMINAMATH_GPT_binomial_coefficient_ratio_l1847_184718

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_ratio_l1847_184718


namespace NUMINAMATH_GPT_inequality_proof_l1847_184793

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a ^ a * b ^ b * c ^ c ≥ 1 / (a * b * c) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1847_184793


namespace NUMINAMATH_GPT_edge_length_box_l1847_184710

theorem edge_length_box (n : ℝ) (h : n = 999.9999999999998) : 
  ∃ (L : ℝ), L = 1 ∧ ((L * 100) ^ 3 / 10 ^ 3) = n := 
sorry

end NUMINAMATH_GPT_edge_length_box_l1847_184710


namespace NUMINAMATH_GPT_total_weight_of_plastic_rings_l1847_184729

-- Conditions
def orange_ring_weight : ℝ := 0.08
def purple_ring_weight : ℝ := 0.33
def white_ring_weight : ℝ := 0.42

-- Proof Statement
theorem total_weight_of_plastic_rings :
  orange_ring_weight + purple_ring_weight + white_ring_weight = 0.83 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_plastic_rings_l1847_184729


namespace NUMINAMATH_GPT_range_of_a_l1847_184738

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (x^2 + a*x + 4 < 0)) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1847_184738


namespace NUMINAMATH_GPT_grocer_sales_l1847_184731

theorem grocer_sales (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale2 = 900)
  (h2 : sale3 = 1000)
  (h3 : sale4 = 700)
  (h4 : sale5 = 800)
  (h5 : sale6 = 900)
  (h6 : (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 850) :
  sale1 = 800 :=
by
  sorry

end NUMINAMATH_GPT_grocer_sales_l1847_184731


namespace NUMINAMATH_GPT_find_subtracted_value_l1847_184792

theorem find_subtracted_value (N : ℕ) (V : ℕ) (hN : N = 2976) (h : (N / 12) - V = 8) : V = 240 := by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1847_184792


namespace NUMINAMATH_GPT_quadratic_inequality_l1847_184787

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1847_184787


namespace NUMINAMATH_GPT_smallest_positive_real_number_l1847_184702

noncomputable def smallest_x : ℝ := 71 / 8

theorem smallest_positive_real_number (x : ℝ) (h₁ : ∀ y : ℝ, 0 < y ∧ (⌊y^2⌋ - y * ⌊y⌋ = 7) → x ≤ y) (h₂ : 0 < x) (h₃ : ⌊x^2⌋ - x * ⌊x⌋ = 7) : x = smallest_x :=
sorry

end NUMINAMATH_GPT_smallest_positive_real_number_l1847_184702


namespace NUMINAMATH_GPT_fixed_monthly_fee_l1847_184716

theorem fixed_monthly_fee (f h : ℝ) 
  (feb_bill : f + h = 18.72)
  (mar_bill : f + 3 * h = 33.78) :
  f = 11.19 :=
by
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l1847_184716


namespace NUMINAMATH_GPT_breakable_iff_composite_l1847_184723

-- Definitions directly from the problem conditions
def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x / a : ℚ) + (y / b : ℚ) = 1

def is_composite (n : ℕ) : Prop :=
  ∃ (s t : ℕ), s > 1 ∧ t > 1 ∧ n = s * t

-- The proof statement
theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ is_composite n := sorry

end NUMINAMATH_GPT_breakable_iff_composite_l1847_184723


namespace NUMINAMATH_GPT_circle_equation_l1847_184719

theorem circle_equation (x y : ℝ) (h1 : (1 - 1)^2 + (1 - 1)^2 = 2) (h2 : (0 - 1)^2 + (0 - 1)^2 = r_sq) :
  (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

end NUMINAMATH_GPT_circle_equation_l1847_184719


namespace NUMINAMATH_GPT_square_area_l1847_184741

theorem square_area : ∃ (s: ℝ), (∀ x: ℝ, x^2 + 4*x + 1 = 7 → ∃ t: ℝ, t = x ∧ ∃ x2: ℝ, (x2 - x)^2 = s^2 ∧ ∀ y : ℝ, y = 7 ∧ y = x2^2 + 4*x2 + 1) ∧ s^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1847_184741


namespace NUMINAMATH_GPT_solve_mod_equation_l1847_184761

theorem solve_mod_equation (y b n : ℤ) (h1 : 15 * y + 4 ≡ 7 [ZMOD 18]) (h2 : y ≡ b [ZMOD n]) (h3 : 2 ≤ n) (h4 : b < n) : b + n = 11 :=
sorry

end NUMINAMATH_GPT_solve_mod_equation_l1847_184761


namespace NUMINAMATH_GPT_sarah_homework_problems_l1847_184703

theorem sarah_homework_problems (math_pages reading_pages problems_per_page : ℕ) 
  (h1 : math_pages = 4) 
  (h2 : reading_pages = 6) 
  (h3 : problems_per_page = 4) : 
  (math_pages + reading_pages) * problems_per_page = 40 :=
by 
  sorry

end NUMINAMATH_GPT_sarah_homework_problems_l1847_184703


namespace NUMINAMATH_GPT_eq_g_of_f_l1847_184740

def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := 6 * x - 29

theorem eq_g_of_f (x : ℝ) : 2 * (f x) - 19 = g x :=
by 
  sorry

end NUMINAMATH_GPT_eq_g_of_f_l1847_184740


namespace NUMINAMATH_GPT_employees_use_public_transportation_l1847_184711

theorem employees_use_public_transportation 
  (total_employees : ℕ)
  (percentage_drive : ℕ)
  (half_of_non_drivers_take_transport : ℕ)
  (h1 : total_employees = 100)
  (h2 : percentage_drive = 60)
  (h3 : half_of_non_drivers_take_transport = 1 / 2) 
  : (total_employees - percentage_drive * total_employees / 100) / 2 = 20 := 
  by
  sorry

end NUMINAMATH_GPT_employees_use_public_transportation_l1847_184711


namespace NUMINAMATH_GPT_no_solution_for_b_a_divides_a_b_minus_1_l1847_184767

theorem no_solution_for_b_a_divides_a_b_minus_1 :
  ¬ (∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ b^a ∣ a^b - 1) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_b_a_divides_a_b_minus_1_l1847_184767


namespace NUMINAMATH_GPT_central_symmetry_preserves_distance_l1847_184717

variables {Point : Type} [MetricSpace Point]

def central_symmetry (O A A' B B' : Point) : Prop :=
  dist O A = dist O A' ∧ dist O B = dist O B'

theorem central_symmetry_preserves_distance {O A A' B B' : Point}
  (h : central_symmetry O A A' B B') : dist A B = dist A' B' :=
sorry

end NUMINAMATH_GPT_central_symmetry_preserves_distance_l1847_184717


namespace NUMINAMATH_GPT_inverse_of_inverse_at_9_l1847_184733

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def f_inv (x : ℝ) : ℝ := (x - 5) / 4

theorem inverse_of_inverse_at_9 : f_inv (f_inv 9) = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_inverse_at_9_l1847_184733


namespace NUMINAMATH_GPT_men_in_room_l1847_184785
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end NUMINAMATH_GPT_men_in_room_l1847_184785


namespace NUMINAMATH_GPT_solution_set_of_x_l1847_184707

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_x_l1847_184707


namespace NUMINAMATH_GPT_width_of_door_is_correct_l1847_184784

theorem width_of_door_is_correct
  (L : ℝ) (W : ℝ) (H : ℝ := 12)
  (door_height : ℝ := 6) (window_height : ℝ := 4) (window_width : ℝ := 3)
  (cost_per_square_foot : ℝ := 10) (total_cost : ℝ := 9060) :
  (L = 25 ∧ W = 15) →
  2 * (L + W) * H - (door_height * width_door + 3 * (window_height * window_width)) * cost_per_square_foot = total_cost →
  width_door = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_width_of_door_is_correct_l1847_184784


namespace NUMINAMATH_GPT_anthony_has_more_pairs_l1847_184705

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end NUMINAMATH_GPT_anthony_has_more_pairs_l1847_184705


namespace NUMINAMATH_GPT_polynomial_perfect_square_l1847_184765

theorem polynomial_perfect_square (k : ℝ) 
  (h : ∃ a : ℝ, x^2 + 8*x + k = (x + a)^2) : 
  k = 16 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_perfect_square_l1847_184765


namespace NUMINAMATH_GPT_determine_remainder_l1847_184782

theorem determine_remainder (a b c : ℕ) (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (H1 : (a + 2 * b + 3 * c) % 7 = 1) 
  (H2 : (2 * a + 3 * b + c) % 7 = 2) 
  (H3 : (3 * a + b + 2 * c) % 7 = 1) : 
  (a * b * c) % 7 = 0 := 
sorry

end NUMINAMATH_GPT_determine_remainder_l1847_184782


namespace NUMINAMATH_GPT_range_of_y_is_correct_l1847_184795

noncomputable def range_of_y (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem range_of_y_is_correct :
  (∀ n, 0 < range_of_y n ∧ range_of_y n < 1 / 2 ∧ n > 2) ∨ (∀ n, 1 ≤ range_of_y n ∧ n ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_y_is_correct_l1847_184795


namespace NUMINAMATH_GPT_transformed_polynomial_l1847_184732

theorem transformed_polynomial (x y : ℝ) (h : y = x + 1 / x) :
  (x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0) → (x^2 * (y^2 - y - 3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_transformed_polynomial_l1847_184732


namespace NUMINAMATH_GPT_rectangle_perimeter_l1847_184759

theorem rectangle_perimeter (a b c d e f g : ℕ)
  (h1 : a + b + c = d)
  (h2 : d + e = g)
  (h3 : b + c = f)
  (h4 : c + f = g)
  (h5 : Nat.gcd (a + b + g) (d + e) = 1)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g) :
  2 * (a + b + g + d + e) = 40 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1847_184759


namespace NUMINAMATH_GPT_coefficient_of_x4_l1847_184751

theorem coefficient_of_x4 (n : ℕ) (f : ℕ → ℕ → ℝ)
  (h1 : (2 : ℕ) ^ n = 256) :
  (f 8 4) * (2 : ℕ) ^ 4 = 1120 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x4_l1847_184751


namespace NUMINAMATH_GPT_rose_bushes_unwatered_l1847_184776

theorem rose_bushes_unwatered (n V A : ℕ) (V_set A_set : Finset ℕ) (hV : V = 1003) (hA : A = 1003) (hTotal : n = 2006) (hIntersection : V_set.card = 3) :
  n - (V + A - V_set.card) = 3 :=
by
  sorry

end NUMINAMATH_GPT_rose_bushes_unwatered_l1847_184776


namespace NUMINAMATH_GPT_calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l1847_184797

def probability_10_ring : ℝ := 0.13
def probability_9_ring : ℝ := 0.28
def probability_8_ring : ℝ := 0.31

def probability_10_or_9_ring : ℝ := probability_10_ring + probability_9_ring

def probability_less_than_9_ring : ℝ := 1 - probability_10_or_9_ring

theorem calc_probability_10_or_9_ring :
  probability_10_or_9_ring = 0.41 :=
by
  sorry

theorem calc_probability_less_than_9_ring :
  probability_less_than_9_ring = 0.59 :=
by
  sorry

end NUMINAMATH_GPT_calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l1847_184797


namespace NUMINAMATH_GPT_min_value_expression_l1847_184760

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x * y * z) ≥ 216 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1847_184760


namespace NUMINAMATH_GPT_solve_quadratic_l1847_184735

theorem solve_quadratic (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 → ∃ r s, (x + r)^2 = s ∧ s = 40.04 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1847_184735


namespace NUMINAMATH_GPT_train_cross_time_l1847_184779

theorem train_cross_time (length_of_train : ℕ) (speed_in_kmh : ℕ) (conversion_factor : ℕ) (speed_in_mps : ℕ) (time : ℕ) :
  length_of_train = 120 →
  speed_in_kmh = 72 →
  conversion_factor = 1000 / 3600 →
  speed_in_mps = speed_in_kmh * conversion_factor →
  time = length_of_train / speed_in_mps →
  time = 6 :=
by
  intros hlength hspeed hconversion hspeed_mps htime
  have : conversion_factor = 5 / 18 := sorry
  have : speed_in_mps = 20 := sorry
  exact sorry

end NUMINAMATH_GPT_train_cross_time_l1847_184779


namespace NUMINAMATH_GPT_problem_inequality_l1847_184769

variable (a b : ℝ)

theorem problem_inequality (h_pos : 0 < a) (h_pos' : 0 < b) (h_sum : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31 / 8)^2 := 
  sorry

end NUMINAMATH_GPT_problem_inequality_l1847_184769


namespace NUMINAMATH_GPT_smallest_d_for_inverse_l1847_184744

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_inverse_l1847_184744


namespace NUMINAMATH_GPT_difference_in_circumferences_l1847_184783

theorem difference_in_circumferences (r_inner r_outer : ℝ) (h1 : r_inner = 15) (h2 : r_outer = r_inner + 8) : 
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 16 * Real.pi :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_difference_in_circumferences_l1847_184783


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1847_184757

theorem geometric_sequence_common_ratio (r : ℝ) (a : ℝ) (a3 : ℝ) :
  a = 3 → a3 = 27 → r = 3 ∨ r = -3 :=
by
  intros ha ha3
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1847_184757


namespace NUMINAMATH_GPT_find_y_l1847_184780

theorem find_y (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : y * 12 = 110) : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1847_184780


namespace NUMINAMATH_GPT_main_inequality_l1847_184730

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_main_inequality_l1847_184730


namespace NUMINAMATH_GPT_y_days_do_work_l1847_184724

theorem y_days_do_work (d : ℝ) (h : (1 / 30) + (1 / d) = 1 / 18) : d = 45 := 
by
  sorry

end NUMINAMATH_GPT_y_days_do_work_l1847_184724


namespace NUMINAMATH_GPT_num_stripes_on_us_flag_l1847_184748

-- Definitions based on conditions in the problem
def num_stars : ℕ := 50

def num_circles : ℕ := (num_stars / 2) - 3

def num_squares (S : ℕ) : ℕ := 2 * S + 6

def total_shapes (num_squares : ℕ) : ℕ := num_circles + num_squares

-- The theorem stating the number of stripes
theorem num_stripes_on_us_flag (S : ℕ) (h1 : num_circles = 22) (h2 : total_shapes (num_squares S) = 54) : S = 13 := by
  sorry

end NUMINAMATH_GPT_num_stripes_on_us_flag_l1847_184748


namespace NUMINAMATH_GPT_incorrect_expression_l1847_184746

variable (x y : ℝ)

theorem incorrect_expression (h : x > y) (hnx : x < 0) (hny : y < 0) : x^2 - 3 ≤ y^2 - 3 := by
sorry

end NUMINAMATH_GPT_incorrect_expression_l1847_184746


namespace NUMINAMATH_GPT_car_speed_l1847_184747

theorem car_speed (v : ℝ) (h : (1 / v) * 3600 = (1 / 450) * 3600 + 2) : v = 360 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1847_184747


namespace NUMINAMATH_GPT_max_value_of_f_l1847_184788

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : ∃ x ∈ Set.Icc (Real.pi / 2) Real.pi, ∀ y ∈ Set.Icc (Real.pi / 2) Real.pi, f y ≤ f x ∧ f x = Real.pi := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1847_184788


namespace NUMINAMATH_GPT_parabola_focus_l1847_184743

-- Define the given conditions
def parabola_equation (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- The proof statement that we need to show the focus of the given parabola
theorem parabola_focus :
  (∃ (h k : ℝ), (k = 1) ∧ (h = 1) ∧ (parabola_equation h = k) ∧ ((h, k + 1 / (4 * 4)) = (1, 17 / 16))) := 
sorry

end NUMINAMATH_GPT_parabola_focus_l1847_184743


namespace NUMINAMATH_GPT_find_ratio_a6_b6_l1847_184712

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def T (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

theorem find_ratio_a6_b6 
  (H1 : ∀ n: ℕ, n > 0 → (S n / T n : ℚ) = n / (2 * n + 1)) :
  (a 6 / b 6 : ℚ) = 11 / 23 :=
sorry

end NUMINAMATH_GPT_find_ratio_a6_b6_l1847_184712


namespace NUMINAMATH_GPT_absolute_value_inequality_solution_l1847_184700

theorem absolute_value_inequality_solution (x : ℝ) :
  abs ((3 * x + 2) / (x + 2)) > 3 ↔ (x < -2) ∨ (-2 < x ∧ x < -4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_solution_l1847_184700


namespace NUMINAMATH_GPT_power_eq_l1847_184781

open Real

theorem power_eq {x : ℝ} (h : x^3 + 4 * x = 8) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end NUMINAMATH_GPT_power_eq_l1847_184781


namespace NUMINAMATH_GPT_find_growth_rate_l1847_184728

noncomputable def donation_first_day : ℝ := 10000
noncomputable def donation_third_day : ℝ := 12100
noncomputable def growth_rate (x : ℝ) : Prop :=
  (donation_first_day * (1 + x) ^ 2 = donation_third_day)

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_find_growth_rate_l1847_184728


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_6_5_8_9_eq_360_l1847_184791

theorem smallest_three_digit_multiple_of_6_5_8_9_eq_360 :
  ∃ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧ n = 360 := 
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_6_5_8_9_eq_360_l1847_184791


namespace NUMINAMATH_GPT_ferris_wheel_seats_l1847_184796

-- Define the total number of seats S as a variable
variables (S : ℕ)

-- Define the conditions
def seat_capacity : ℕ := 15

def broken_seats : ℕ := 10

def max_riders : ℕ := 120

-- The theorem statement
theorem ferris_wheel_seats :
  ((S - broken_seats) * seat_capacity = max_riders) → S = 18 :=
by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seats_l1847_184796


namespace NUMINAMATH_GPT_bus_system_carry_per_day_l1847_184758

theorem bus_system_carry_per_day (total_people : ℕ) (weeks : ℕ) (days_in_week : ℕ) (people_per_day : ℕ) :
  total_people = 109200000 →
  weeks = 13 →
  days_in_week = 7 →
  people_per_day = total_people / (weeks * days_in_week) →
  people_per_day = 1200000 :=
by
  intros htotal hweeks hdays hcalc
  sorry

end NUMINAMATH_GPT_bus_system_carry_per_day_l1847_184758


namespace NUMINAMATH_GPT_mixed_groups_count_l1847_184721

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end NUMINAMATH_GPT_mixed_groups_count_l1847_184721
