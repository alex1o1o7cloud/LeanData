import Mathlib

namespace NUMINAMATH_GPT_three_digit_number_cubed_sum_l1224_122420

theorem three_digit_number_cubed_sum {a b c : ℕ} (h₁ : 1 ≤ a ∧ a ≤ 9)
                                      (h₂ : 0 ≤ b ∧ b ≤ 9)
                                      (h₃ : 0 ≤ c ∧ c ≤ 9) :
  (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999) →
  (100 * a + 10 * b + c = (a + b + c) ^ 3) →
  (100 * a + 10 * b + c = 512) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_cubed_sum_l1224_122420


namespace NUMINAMATH_GPT_ferry_speed_difference_l1224_122485

variable (v_P v_Q d_P d_Q t_P t_Q x : ℝ)

-- Defining the constants and conditions provided in the problem
axiom h1 : v_P = 8 
axiom h2 : t_P = 2 
axiom h3 : d_P = t_P * v_P 
axiom h4 : d_Q = 3 * d_P 
axiom h5 : t_Q = t_P + 2
axiom h6 : d_Q = v_Q * t_Q 
axiom h7 : x = v_Q - v_P 

-- The theorem that corresponds to the solution
theorem ferry_speed_difference : x = 4 := by
  sorry

end NUMINAMATH_GPT_ferry_speed_difference_l1224_122485


namespace NUMINAMATH_GPT_average_cars_given_per_year_l1224_122487

/-- Definition of initial conditions and the proposition -/
def initial_cars : ℕ := 3500
def final_cars : ℕ := 500
def years : ℕ := 60

theorem average_cars_given_per_year : (initial_cars - final_cars) / years = 50 :=
by
  sorry

end NUMINAMATH_GPT_average_cars_given_per_year_l1224_122487


namespace NUMINAMATH_GPT_Paige_recycled_pounds_l1224_122405

-- Definitions based on conditions from step a)
def points_per_pound := 1 / 4
def friends_pounds_recycled := 2
def total_points := 4

-- The proof statement (no proof required)
theorem Paige_recycled_pounds :
  let total_pounds_recycled := total_points * 4
  let paige_pounds_recycled := total_pounds_recycled - friends_pounds_recycled
  paige_pounds_recycled = 14 :=
by
  sorry

end NUMINAMATH_GPT_Paige_recycled_pounds_l1224_122405


namespace NUMINAMATH_GPT_Alpha_Beta_meet_at_Alpha_Beta_meet_again_l1224_122442

open Real

-- Definitions and conditions
def A : ℝ := -24
def B : ℝ := -10
def C : ℝ := 10
def Alpha_speed : ℝ := 4
def Beta_speed : ℝ := 6

-- Question 1: Prove that Alpha and Beta meet at -10.4
theorem Alpha_Beta_meet_at : 
  ∃ t : ℝ, (A + Alpha_speed * t = C - Beta_speed * t) ∧ (A + Alpha_speed * t = -10.4) :=
  sorry

-- Question 2: Prove that after reversing at t = 2, Alpha and Beta meet again at -44
theorem Alpha_Beta_meet_again :
  ∃ t z : ℝ, 
    ((t = 2) ∧ (4 * t + (14 - 4 * t) + (14 - 4 * t + 20) = 40) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = C - Beta_speed * t - Beta_speed * z) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = -44)) :=
  sorry  

end NUMINAMATH_GPT_Alpha_Beta_meet_at_Alpha_Beta_meet_again_l1224_122442


namespace NUMINAMATH_GPT_find_asterisk_l1224_122434

theorem find_asterisk : ∃ (x : ℕ), (63 / 21) * (x / 189) = 1 ∧ x = 63 :=
by
  sorry

end NUMINAMATH_GPT_find_asterisk_l1224_122434


namespace NUMINAMATH_GPT_inequality_solution_l1224_122488

theorem inequality_solution (x : ℝ) :
    (∀ t : ℝ, abs (t - 3) + abs (2 * t + 1) ≥ abs (2 * x - 1) + abs (x + 2)) ↔ 
    (-1 / 2 ≤ x ∧ x ≤ 5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1224_122488


namespace NUMINAMATH_GPT_find_theta_interval_l1224_122470

theorem find_theta_interval (θ : ℝ) (x : ℝ) :
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (0 ≤ x ∧ x ≤ 1) →
  (∀ k, k = 0.5 → x^2 * Real.sin θ - k * x * (1 - x) + (1 - x)^2 * Real.cos θ ≥ 0) ↔
  (0 ≤ θ ∧ θ ≤ π / 12) ∨ (23 * π / 12 ≤ θ ∧ θ ≤ 2 * π) := 
sorry

end NUMINAMATH_GPT_find_theta_interval_l1224_122470


namespace NUMINAMATH_GPT_hat_price_after_discounts_l1224_122490

-- Defining initial conditions
def initial_price : ℝ := 15
def first_discount_percent : ℝ := 0.25
def second_discount_percent : ℝ := 0.50

-- Defining the expected final price after applying both discounts
def expected_final_price : ℝ := 5.625

-- Lean statement to prove the final price after both discounts is as expected
theorem hat_price_after_discounts : 
  let first_reduced_price := initial_price * (1 - first_discount_percent)
  let second_reduced_price := first_reduced_price * (1 - second_discount_percent)
  second_reduced_price = expected_final_price := sorry

end NUMINAMATH_GPT_hat_price_after_discounts_l1224_122490


namespace NUMINAMATH_GPT_volume_of_defined_region_l1224_122400

noncomputable def volume_of_region (x y z : ℝ) : ℝ :=
if x + y ≤ 5 ∧ z ≤ 5 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x ≤ 2 then 15 else 0

theorem volume_of_defined_region :
  ∀ (x y z : ℝ),
  (0 ≤ x) → (0 ≤ y) → (0 ≤ z) → (x ≤ 2) →
  (|x + y + z| + |x + y - z| ≤ 10) →
  volume_of_region x y z = 15 :=
sorry

end NUMINAMATH_GPT_volume_of_defined_region_l1224_122400


namespace NUMINAMATH_GPT_correct_calculation_l1224_122448

theorem correct_calculation (y : ℤ) (h : (y + 4) * 5 = 140) : 5 * y + 4 = 124 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l1224_122448


namespace NUMINAMATH_GPT_arrangements_count_correct_l1224_122437

def arrangements_total : Nat :=
  let total_with_A_first := (Nat.factorial 5) -- A^5_5 = 120
  let total_with_B_first := (Nat.factorial 4) * 1 -- A^1_4 * A^4_4 = 96
  total_with_A_first + total_with_B_first

theorem arrangements_count_correct : arrangements_total = 216 := 
by
  -- Proof is required here
  sorry

end NUMINAMATH_GPT_arrangements_count_correct_l1224_122437


namespace NUMINAMATH_GPT_find_a_b_l1224_122499

theorem find_a_b (a b x y : ℝ) (h₀ : a + b = 10) (h₁ : a / x + b / y = 1) (h₂ : x + y = 16) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
    (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l1224_122499


namespace NUMINAMATH_GPT_find_N_l1224_122473

theorem find_N (x y N : ℝ) (h1 : 2 * x + y = N) (h2 : x + 2 * y = 5) (h3 : (x + y) / 3 = 1) : N = 4 :=
by
  have h4 : x + y = 3 := by
    linarith [h3]
  have h5 : y = 3 - x := by
    linarith [h4]
  have h6 : x + 2 * (3 - x) = 5 := by
    linarith [h2, h5]
  have h7 : x = 1 := by
    linarith [h6]
  have h8 : y = 2 := by
    linarith [h4, h7]
  have h9 : 2 * x + y = 4 := by
    linarith [h7, h8]
  linarith [h1, h9]

end NUMINAMATH_GPT_find_N_l1224_122473


namespace NUMINAMATH_GPT_geometric_series_sum_l1224_122469

theorem geometric_series_sum :
  2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))))))) = 2046 := 
by sorry

end NUMINAMATH_GPT_geometric_series_sum_l1224_122469


namespace NUMINAMATH_GPT_ramsey_6_3_3_l1224_122455

open Classical

theorem ramsey_6_3_3 (G : SimpleGraph (Fin 6)) :
  ∃ (A : Finset (Fin 6)), A.card = 3 ∧ (∀ (x y : Fin 6), x ∈ A → y ∈ A → x ≠ y → G.Adj x y) ∨ ∃ (B : Finset (Fin 6)), B.card = 3 ∧ (∀ (x y : Fin 6), x ∈ B → y ∈ B → x ≠ y → ¬ G.Adj x y) :=
by
  sorry

end NUMINAMATH_GPT_ramsey_6_3_3_l1224_122455


namespace NUMINAMATH_GPT_intersection_unique_point_l1224_122439

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 16 * x + 28

theorem intersection_unique_point :
  ∃ a : ℝ, f a = a ∧ a = -4 := sorry

end NUMINAMATH_GPT_intersection_unique_point_l1224_122439


namespace NUMINAMATH_GPT_agatha_initial_money_l1224_122461

/-
Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel.
If she has $20 left to spend on a seat and handlebar tape, prove that she had $60 initially.
-/

theorem agatha_initial_money (frame_cost wheel_cost remaining_money initial_money: ℕ) 
  (h1 : frame_cost = 15) 
  (h2 : wheel_cost = 25) 
  (h3 : remaining_money = 20) 
  (h4 : initial_money = frame_cost + wheel_cost + remaining_money) : 
  initial_money = 60 :=
by {
  -- We state explicitly that initial_money should be 60
  sorry
}

end NUMINAMATH_GPT_agatha_initial_money_l1224_122461


namespace NUMINAMATH_GPT_find_k_l1224_122475

theorem find_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) (hk : k ≠ 0) : k = 5 :=
sorry

end NUMINAMATH_GPT_find_k_l1224_122475


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1224_122491

variable (v : ℝ)

-- Given conditions
def speed_including_stoppages := 45 -- kmph
def stoppage_time_ratio := 1/6 -- 10 minutes per hour is 1/6 of the time

-- Prove that the speed excluding stoppages is 54 kmph
theorem bus_speed_excluding_stoppages (h1 : speed_including_stoppages = 45) 
                                      (h2 : stoppage_time_ratio = 1/6) : 
                                      v = 54 := by
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1224_122491


namespace NUMINAMATH_GPT_girls_count_in_leos_class_l1224_122409

def leo_class_girls_count (g b : ℕ) :=
  (g / b = 3 / 4) ∧ (g + b = 35) → g = 15

theorem girls_count_in_leos_class (g b : ℕ) :
  leo_class_girls_count g b :=
by
  sorry

end NUMINAMATH_GPT_girls_count_in_leos_class_l1224_122409


namespace NUMINAMATH_GPT_find_k_and_b_l1224_122413

variables (k b : ℝ)

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (k * p.1, p.2 + b)

theorem find_k_and_b
  (h : f k b (6, 2) = (3, 1)) :
  k = 2 ∧ b = -1 :=
by {
  -- proof steps would go here
  sorry
}

end NUMINAMATH_GPT_find_k_and_b_l1224_122413


namespace NUMINAMATH_GPT_fraction_least_l1224_122462

noncomputable def solve_fraction_least : Prop :=
  ∃ (x y : ℚ), x + y = 5/6 ∧ x * y = 1/8 ∧ (min x y = 1/6)
  
theorem fraction_least : solve_fraction_least :=
sorry

end NUMINAMATH_GPT_fraction_least_l1224_122462


namespace NUMINAMATH_GPT_tips_fraction_l1224_122419

theorem tips_fraction {S T I : ℚ} (h1 : T = (7/4) * S) (h2 : I = S + T) : (T / I) = 7 / 11 :=
by
  sorry

end NUMINAMATH_GPT_tips_fraction_l1224_122419


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1224_122416

noncomputable def f (a b x : ℝ) := x / (a * x + b)

theorem value_of_a_plus_b (a b : ℝ) (h₁: a ≠ 0) (h₂: f a b (-4) = 4)
    (h₃: ∀ x, f a b (f a b x) = x) : a + b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1224_122416


namespace NUMINAMATH_GPT_total_house_rent_l1224_122402

theorem total_house_rent (P S R : ℕ)
  (h1 : S = 5 * P)
  (h2 : R = 3 * P)
  (h3 : R = 1800) : 
  S + P + R = 5400 :=
by
  sorry

end NUMINAMATH_GPT_total_house_rent_l1224_122402


namespace NUMINAMATH_GPT_intersection_correct_l1224_122454

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x < 2}

theorem intersection_correct : P ∩ Q = {1} :=
by sorry

end NUMINAMATH_GPT_intersection_correct_l1224_122454


namespace NUMINAMATH_GPT_charlie_widgets_difference_l1224_122497

theorem charlie_widgets_difference (w t : ℕ) (hw : w = 3 * t) :
  w * t - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end NUMINAMATH_GPT_charlie_widgets_difference_l1224_122497


namespace NUMINAMATH_GPT_guppies_total_l1224_122418

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end NUMINAMATH_GPT_guppies_total_l1224_122418


namespace NUMINAMATH_GPT_quadratic_coefficients_l1224_122486

theorem quadratic_coefficients (x1 x2 p q : ℝ)
  (h1 : x1 - x2 = 5)
  (h2 : x1 ^ 3 - x2 ^ 3 = 35) :
  (x1 + x2 = -p ∧ x1 * x2 = q ∧ (p = 1 ∧ q = -6) ∨ 
   x1 + x2 = p ∧ x1 * x2 = q ∧ (p = -1 ∧ q = -6)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1224_122486


namespace NUMINAMATH_GPT_favorite_numbers_parity_l1224_122438

variables (D J A H : ℤ)

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem favorite_numbers_parity
  (h1 : odd (D + 3 * J))
  (h2 : odd ((A - H) * 5))
  (h3 : even (D * H + 17)) :
  odd D ∧ even J ∧ even A ∧ odd H := 
sorry

end NUMINAMATH_GPT_favorite_numbers_parity_l1224_122438


namespace NUMINAMATH_GPT_min_guests_l1224_122428

theorem min_guests (total_food : ℕ) (max_food : ℝ) 
  (H1 : total_food = 337) 
  (H2 : max_food = 2) : 
  ∃ n : ℕ, n = ⌈total_food / max_food⌉ ∧ n = 169 :=
by
  sorry

end NUMINAMATH_GPT_min_guests_l1224_122428


namespace NUMINAMATH_GPT_recurring_fraction_division_l1224_122498

noncomputable def recurring_833 := 5 / 6
noncomputable def recurring_1666 := 5 / 3

theorem recurring_fraction_division : 
  (recurring_833 / recurring_1666) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_recurring_fraction_division_l1224_122498


namespace NUMINAMATH_GPT_find_second_x_intercept_l1224_122411

theorem find_second_x_intercept (a b c : ℝ)
  (h_vertex : ∀ x, y = a * x^2 + b * x + c → x = 5 → y = -3)
  (h_intercept1 : ∀ y, y = a * 1^2 + b * 1 + c → y = 0) :
  ∃ x, y = a * x^2 + b * x + c ∧ y = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_GPT_find_second_x_intercept_l1224_122411


namespace NUMINAMATH_GPT_count_four_digit_integers_with_1_or_7_l1224_122463

/-- 
The total number of four-digit integers with at least one digit being 1 or 7 is 5416.
-/
theorem count_four_digit_integers_with_1_or_7 : 
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  with_1_or_7 = 5416
:= by
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  show with_1_or_7 = 5416
  sorry

end NUMINAMATH_GPT_count_four_digit_integers_with_1_or_7_l1224_122463


namespace NUMINAMATH_GPT_find_y_coordinate_l1224_122403

theorem find_y_coordinate (x2 : ℝ) (y1 : ℝ) :
  (∃ m : ℝ, m = (y1 - 0) / (10 - 4) ∧ (-8 - y1) = m * (x2 - 10)) →
  y1 = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_y_coordinate_l1224_122403


namespace NUMINAMATH_GPT_hypotenuse_length_l1224_122477

-- Definitions derived from conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Proposed theorem
theorem hypotenuse_length (a c : ℝ) 
  (h1 : is_isosceles_right_triangle a a c) 
  (h2 : perimeter a a c = 8 + 8 * Real.sqrt 2) :
  c = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1224_122477


namespace NUMINAMATH_GPT_colombian_coffee_amount_l1224_122432

theorem colombian_coffee_amount 
  (C B : ℝ) 
  (h1 : C + B = 100)
  (h2 : 8.75 * C + 3.75 * B = 635) :
  C = 52 := 
sorry

end NUMINAMATH_GPT_colombian_coffee_amount_l1224_122432


namespace NUMINAMATH_GPT_mowing_ratio_is_sqrt2_l1224_122436

noncomputable def mowing_ratio (s w : ℝ) (hw_half_area : w * (s * Real.sqrt 2) = s^2) : ℝ :=
  s / w

theorem mowing_ratio_is_sqrt2 (s w : ℝ) (hs_positive : s > 0) (hw_positive : w > 0)
  (hw_half_area : w * (s * Real.sqrt 2) = s^2) : mowing_ratio s w hw_half_area = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_mowing_ratio_is_sqrt2_l1224_122436


namespace NUMINAMATH_GPT_maximum_value_of_f_over_interval_l1224_122451

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * x - 2)

theorem maximum_value_of_f_over_interval :
  ∀ x : ℝ, -4 < x ∧ x < 1 → ∃ M : ℝ, (∀ y : ℝ, -4 < y ∧ y < 1 → f y ≤ M) ∧ M = -1 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_over_interval_l1224_122451


namespace NUMINAMATH_GPT_carl_owes_15300_l1224_122421

def total_property_damage : ℝ := 40000
def total_medical_bills : ℝ := 70000
def insurance_coverage_property_damage : ℝ := 0.80
def insurance_coverage_medical_bills : ℝ := 0.75
def carl_responsibility : ℝ := 0.60

def carl_personally_owes : ℝ :=
  let insurance_paid_property_damage := insurance_coverage_property_damage * total_property_damage
  let insurance_paid_medical_bills := insurance_coverage_medical_bills * total_medical_bills
  let remaining_property_damage := total_property_damage - insurance_paid_property_damage
  let remaining_medical_bills := total_medical_bills - insurance_paid_medical_bills
  let carl_share_property_damage := carl_responsibility * remaining_property_damage
  let carl_share_medical_bills := carl_responsibility * remaining_medical_bills
  carl_share_property_damage + carl_share_medical_bills

theorem carl_owes_15300 :
  carl_personally_owes = 15300 := by
  sorry

end NUMINAMATH_GPT_carl_owes_15300_l1224_122421


namespace NUMINAMATH_GPT_physics_marks_l1224_122464

variables (P C M : ℕ)

theorem physics_marks (h1 : P + C + M = 195)
                      (h2 : P + M = 180)
                      (h3 : P + C = 140) : P = 125 :=
by
  sorry

end NUMINAMATH_GPT_physics_marks_l1224_122464


namespace NUMINAMATH_GPT_pythagorean_triplet_unique_solution_l1224_122430

-- Define the conditions given in the problem
def is_solution (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  2000 ≤ a ∧ a ≤ 3000 ∧
  2000 ≤ b ∧ b ≤ 3000 ∧
  2000 ≤ c ∧ c ≤ 3000

-- Prove that the only set of integers (a, b, c) meeting the conditions
-- equals the specific tuple (2100, 2059, 2941)
theorem pythagorean_triplet_unique_solution : 
  ∀ a b c : ℕ, is_solution a b c ↔ (a = 2100 ∧ b = 2059 ∧ c = 2941) :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triplet_unique_solution_l1224_122430


namespace NUMINAMATH_GPT_corvette_trip_time_percentage_increase_l1224_122424

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end NUMINAMATH_GPT_corvette_trip_time_percentage_increase_l1224_122424


namespace NUMINAMATH_GPT_difference_of_squares_36_l1224_122453

theorem difference_of_squares_36 {x y : ℕ} (h₁ : x + y = 18) (h₂ : x * y = 80) (h₃ : x > y) : x^2 - y^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_36_l1224_122453


namespace NUMINAMATH_GPT_fractional_equation_solution_l1224_122426

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (3 / (x + 1) = 2 / (x - 1)) → (x = 5) :=
sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1224_122426


namespace NUMINAMATH_GPT_complement_U_A_eq_two_l1224_122465

open Set

universe u

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }
def comp_U_A : Set ℕ := U \ A

theorem complement_U_A_eq_two : comp_U_A = {2} :=
by 
  sorry

end NUMINAMATH_GPT_complement_U_A_eq_two_l1224_122465


namespace NUMINAMATH_GPT_problem_1_problem_2_l1224_122472

open BigOperators

-- Question 1
theorem problem_1 (a : Fin 2021 → ℝ) :
  (1 + 2 * x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  (∑ i in Finset.range 2021, (i * a i)) = 4040 * 3 ^ 2019 :=
sorry

-- Question 2
theorem problem_2 (a : Fin 2021 → ℝ) :
  (1 - x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  ((∑ i in Finset.range 2021, 1 / a i)) = 2021 / 1011 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1224_122472


namespace NUMINAMATH_GPT_evaluate_g_at_5_l1224_122468

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem evaluate_g_at_5 : g 5 = 15 :=
by
    -- proof steps here
    sorry

end NUMINAMATH_GPT_evaluate_g_at_5_l1224_122468


namespace NUMINAMATH_GPT_max_expression_l1224_122480

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end NUMINAMATH_GPT_max_expression_l1224_122480


namespace NUMINAMATH_GPT_sum_of_squares_l1224_122484

theorem sum_of_squares (x : ℚ) (h : x + 2 * x + 3 * x = 14) : 
  (x^2 + (2 * x)^2 + (3 * x)^2) = 686 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1224_122484


namespace NUMINAMATH_GPT_powderman_distance_when_hears_explosion_l1224_122492

noncomputable def powderman_speed_yd_per_s : ℝ := 10
noncomputable def blast_time_s : ℝ := 45
noncomputable def sound_speed_ft_per_s : ℝ := 1080
noncomputable def powderman_speed_ft_per_s : ℝ := 30

noncomputable def distance_powderman (t : ℝ) : ℝ := powderman_speed_ft_per_s * t
noncomputable def distance_sound (t : ℝ) : ℝ := sound_speed_ft_per_s * (t - blast_time_s)

theorem powderman_distance_when_hears_explosion :
  ∃ t, t > blast_time_s ∧ distance_powderman t = distance_sound t ∧ (distance_powderman t) / 3 = 463 :=
sorry

end NUMINAMATH_GPT_powderman_distance_when_hears_explosion_l1224_122492


namespace NUMINAMATH_GPT_taxi_speed_l1224_122466

theorem taxi_speed (v : ℕ) (h₁ : v > 30) (h₂ : ∃ t₁ t₂ : ℕ, t₁ = 3 ∧ t₂ = 3 ∧ 
                    v * t₁ = (v - 30) * (t₁ + t₂)) : 
                    v = 60 :=
by
  sorry

end NUMINAMATH_GPT_taxi_speed_l1224_122466


namespace NUMINAMATH_GPT_min_value_l1224_122458

-- Define the conditions
variables (x y : ℝ)
-- Assume x and y are in the positive real numbers
axiom pos_x : 0 < x
axiom pos_y : 0 < y
-- Given equation
axiom eq1 : x + 2 * y = 2 * x * y

-- The goal is to prove that the minimum value of 3x + 4y is 5 + 2sqrt(6)
theorem min_value (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (eq1 : x + 2 * y = 2 * x * y) : 
  3 * x + 4 * y ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_min_value_l1224_122458


namespace NUMINAMATH_GPT_shelves_filled_l1224_122449

theorem shelves_filled (carvings_per_shelf : ℕ) (total_carvings : ℕ) (h₁ : carvings_per_shelf = 8) (h₂ : total_carvings = 56) :
  total_carvings / carvings_per_shelf = 7 := by
  sorry

end NUMINAMATH_GPT_shelves_filled_l1224_122449


namespace NUMINAMATH_GPT_P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l1224_122467

-- Assume the definition of sum of digits of n and count of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum  -- Sum of digits in base 10 representation

def num_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).length  -- Number of digits in base 10 representation

def P (n : ℕ) : ℕ :=
  sum_of_digits n + num_of_digits n

-- Problem (a)
theorem P_2017 : P 2017 = 14 :=
sorry

-- Problem (b)
theorem P_eq_4 :
  {n : ℕ | P n = 4} = {3, 11, 20, 100} :=
sorry

-- Problem (c)
theorem exists_P_minus_P_succ_gt_50 : 
  ∃ n : ℕ, P n - P (n + 1) > 50 :=
sorry

end NUMINAMATH_GPT_P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l1224_122467


namespace NUMINAMATH_GPT_number_of_mismatching_socks_l1224_122401

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_number_of_mismatching_socks_l1224_122401


namespace NUMINAMATH_GPT_congruence_solution_exists_l1224_122410

theorem congruence_solution_exists {p n a : ℕ} (hp : Prime p) (hn : n % p ≠ 0) (ha : a % p ≠ 0)
  (hx : ∃ x : ℕ, x^n % p = a % p) :
  ∀ r : ℕ, ∃ x : ℕ, x^n % (p^(r + 1)) = a % (p^(r + 1)) :=
by
  intros r
  sorry

end NUMINAMATH_GPT_congruence_solution_exists_l1224_122410


namespace NUMINAMATH_GPT_second_offset_length_l1224_122441

noncomputable def quadrilateral_area (d o1 o2 : ℝ) : ℝ :=
  (1 / 2) * d * (o1 + o2)

theorem second_offset_length (d o1 A : ℝ) (h_d : d = 22) (h_o1 : o1 = 9) (h_A : A = 165) :
  ∃ o2, quadrilateral_area d o1 o2 = A ∧ o2 = 6 := by
  sorry

end NUMINAMATH_GPT_second_offset_length_l1224_122441


namespace NUMINAMATH_GPT_six_digit_palindromes_count_l1224_122483

theorem six_digit_palindromes_count : 
  ∃ n : ℕ, n = 27 ∧ 
  (∀ (A B C : ℕ), 
       (A = 6 ∨ A = 7 ∨ A = 8) ∧ 
       (B = 6 ∨ B = 7 ∨ B = 8) ∧ 
       (C = 6 ∨ C = 7 ∨ C = 8) → 
       ∃ p : ℕ, 
         p = (A * 10^5 + B * 10^4 + C * 10^3 + C * 10^2 + B * 10 + A) ∧ 
         (6 ≤ p / 10^5 ∧ p / 10^5 ≤ 8) ∧ 
         (6 ≤ (p / 10^4) % 10 ∧ (p / 10^4) % 10 ≤ 8) ∧ 
         (6 ≤ (p / 10^3) % 10 ∧ (p / 10^3) % 10 ≤ 8)) :=
  by sorry

end NUMINAMATH_GPT_six_digit_palindromes_count_l1224_122483


namespace NUMINAMATH_GPT_fruiting_plants_given_away_l1224_122414

noncomputable def roxy_fruiting_plants_given_away 
  (N_f : ℕ) -- initial flowering plants
  (N_ft : ℕ) -- initial fruiting plants
  (N_bsf : ℕ) -- flowering plants bought on Saturday
  (N_bst : ℕ) -- fruiting plants bought on Saturday
  (N_gsf : ℕ) -- flowering plant given away on Sunday
  (N_total_remaining : ℕ) -- total plants remaining 
  (H₁ : N_ft = 2 * N_f) -- twice as many fruiting plants
  (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) -- total plants equation
  : ℕ :=
  4

theorem fruiting_plants_given_away (N_f : ℕ) (N_ft : ℕ) (N_bsf : ℕ) (N_bst : ℕ) (N_gsf : ℕ) (N_total_remaining : ℕ)
  (H₁ : N_ft = 2 * N_f) (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) : N_ft - (N_total_remaining - (N_f + N_bsf - N_gsf)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_fruiting_plants_given_away_l1224_122414


namespace NUMINAMATH_GPT_positive_x_condition_l1224_122482

theorem positive_x_condition (x : ℝ) (h : x > 0 ∧ (0.01 * x * x = 9)) : x = 30 :=
sorry

end NUMINAMATH_GPT_positive_x_condition_l1224_122482


namespace NUMINAMATH_GPT_geometric_sequence_S5_l1224_122433

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_S5_l1224_122433


namespace NUMINAMATH_GPT_percentage_of_cars_with_no_features_l1224_122446

theorem percentage_of_cars_with_no_features (N S W R SW SR WR SWR : ℕ)
  (hN : N = 120)
  (hS : S = 70)
  (hW : W = 40)
  (hR : R = 30)
  (hSW : SW = 20)
  (hSR : SR = 15)
  (hWR : WR = 10)
  (hSWR : SWR = 5) :
  (120 - (S + W + R - SW - SR - WR + SWR)) / (N : ℝ) * 100 = 16.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_cars_with_no_features_l1224_122446


namespace NUMINAMATH_GPT_eval_expression_in_second_quadrant_l1224_122404

theorem eval_expression_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α > 0) (h3 : Real.cos α < 0) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / (Real.sin α) ^ 2 - 1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_in_second_quadrant_l1224_122404


namespace NUMINAMATH_GPT_find_x_l1224_122435

theorem find_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : 
  x = 7 :=
sorry

end NUMINAMATH_GPT_find_x_l1224_122435


namespace NUMINAMATH_GPT_watch_cost_l1224_122478

variables (w s : ℝ)

theorem watch_cost (h1 : w + s = 120) (h2 : w = 100 + s) : w = 110 :=
by
  sorry

end NUMINAMATH_GPT_watch_cost_l1224_122478


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l1224_122456

-- Problem (1) representation
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m x : ℝ) : Prop := 1 < m ∧ m < 2 ∧ x = (1 / 2)^(m - 1)

theorem range_of_x (x : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → x = (1 / 2)^(m - 1)) ∧ p (1/4) x →
  1/2 < x ∧ x < 3/4 :=
sorry

-- Problem (2) representation
theorem range_of_a (a : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → ∀ x, x = (1 / 2)^(m - 1) → p a x) →
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l1224_122456


namespace NUMINAMATH_GPT_angle_between_bisectors_l1224_122407

theorem angle_between_bisectors (β γ : ℝ) (h_sum : β + γ = 130) : (β / 2) + (γ / 2) = 65 :=
by
  have h : β + γ = 130 := h_sum
  sorry

end NUMINAMATH_GPT_angle_between_bisectors_l1224_122407


namespace NUMINAMATH_GPT_hundred_days_from_friday_is_sunday_l1224_122427

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end NUMINAMATH_GPT_hundred_days_from_friday_is_sunday_l1224_122427


namespace NUMINAMATH_GPT_complement_union_eq_l1224_122425

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}
def complement (A : Set ℝ) : Set ℝ := {x | x ∉ A}

theorem complement_union_eq :
  complement (M ∪ N) = {x | x ≥ 1} :=
sorry

end NUMINAMATH_GPT_complement_union_eq_l1224_122425


namespace NUMINAMATH_GPT_max_m_ratio_l1224_122450

theorem max_m_ratio (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∀ a b, (4 / a + 1 / b) ≥ m / (a + 4 * b)) :
  (m = 16) → (b / a = 1 / 4) :=
by sorry

end NUMINAMATH_GPT_max_m_ratio_l1224_122450


namespace NUMINAMATH_GPT_cos_4theta_l1224_122417

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end NUMINAMATH_GPT_cos_4theta_l1224_122417


namespace NUMINAMATH_GPT_tournament_players_l1224_122423

theorem tournament_players (n : ℕ) (h : n * (n - 1) / 2 = 56) : n = 14 :=
sorry

end NUMINAMATH_GPT_tournament_players_l1224_122423


namespace NUMINAMATH_GPT_solve_f_1991_2_1990_l1224_122415

-- Define the sum of digits function for an integer k
def sum_of_digits (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define f1(k) as the square of the sum of digits of k
def f1 (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

-- Define the recursive sequence fn as given in the problem
def fn : ℕ → ℕ → ℕ
| 0, k => k
| n + 1, k => f1 (fn n k)

-- Define the specific problem statement
theorem solve_f_1991_2_1990 : fn 1991 (2 ^ 1990) = 4 := sorry

end NUMINAMATH_GPT_solve_f_1991_2_1990_l1224_122415


namespace NUMINAMATH_GPT_area_of_region_l1224_122481

theorem area_of_region : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 8*y = 16) → 
  (π * 41) = (π * 41) :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l1224_122481


namespace NUMINAMATH_GPT_count_negative_numbers_l1224_122479

theorem count_negative_numbers : 
  let n1 := abs (-2)
  let n2 := - abs (3^2)
  let n3 := - (3^2)
  let n4 := (-2)^(2023)
  (if n1 < 0 then 1 else 0) + (if n2 < 0 then 1 else 0) + (if n3 < 0 then 1 else 0) + (if n4 < 0 then 1 else 0) = 3 := 
by
  sorry

end NUMINAMATH_GPT_count_negative_numbers_l1224_122479


namespace NUMINAMATH_GPT_minimum_k_value_l1224_122471

theorem minimum_k_value (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∀ a b, (1 / a + 1 / b + k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end NUMINAMATH_GPT_minimum_k_value_l1224_122471


namespace NUMINAMATH_GPT_algebra_square_formula_l1224_122495

theorem algebra_square_formula (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
sorry

end NUMINAMATH_GPT_algebra_square_formula_l1224_122495


namespace NUMINAMATH_GPT_difference_in_price_l1224_122460

-- Definitions based on the given conditions
def price_with_cork : ℝ := 2.10
def price_cork : ℝ := 0.05
def price_without_cork : ℝ := price_with_cork - price_cork

-- The theorem proving the given question and correct answer
theorem difference_in_price : price_with_cork - price_without_cork = price_cork :=
by
  -- Proof can be omitted
  sorry

end NUMINAMATH_GPT_difference_in_price_l1224_122460


namespace NUMINAMATH_GPT_simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l1224_122494

theorem simplify_expression (a : ℤ) (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 :=
by
  sorry

theorem evaluate_expression_at_neg1 (h : (-1 : ℤ) ≠ 0) (h' : (-1 : ℤ) ≠ 1) : 
  (-1 - (2 * (-1) - 1) / (-1)) / ((-1 - 1) / (-1)) = -2 :=
by
  sorry

theorem evaluate_expression_at_2 (h : (2 : ℤ) ≠ 0) (h' : (2 : ℤ) ≠ 1) : 
  (2 - (2 * 2 - 1) / 2) / ((2 - 1) / 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l1224_122494


namespace NUMINAMATH_GPT_find_missing_ratio_l1224_122444

theorem find_missing_ratio
  (x y : ℕ)
  (h : ((2 / 3 : ℚ) * (x / y : ℚ) * (11 / 2 : ℚ) = 2)) :
  x = 6 ∧ y = 11 :=
sorry

end NUMINAMATH_GPT_find_missing_ratio_l1224_122444


namespace NUMINAMATH_GPT_costs_equal_when_x_20_l1224_122429

noncomputable def costA (x : ℕ) : ℤ := 150 * x + 3300
noncomputable def costB (x : ℕ) : ℤ := 210 * x + 2100

theorem costs_equal_when_x_20 : costA 20 = costB 20 :=
by
  -- Statements representing the costs equal condition
  have ha : costA 20 = 150 * 20 + 3300 := rfl
  have hb : costB 20 = 210 * 20 + 2100 := rfl
  rw [ha, hb]
  -- Simplification steps (represented here in Lean)
  sorry

end NUMINAMATH_GPT_costs_equal_when_x_20_l1224_122429


namespace NUMINAMATH_GPT_unit_digit_7_pow_2023_l1224_122412

theorem unit_digit_7_pow_2023 : (7^2023) % 10 = 3 :=
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_unit_digit_7_pow_2023_l1224_122412


namespace NUMINAMATH_GPT_final_ratio_of_milk_to_water_l1224_122422

-- Initial conditions definitions
def initial_milk_ratio : ℚ := 5 / 8
def initial_water_ratio : ℚ := 3 / 8
def additional_milk : ℚ := 8
def total_capacity : ℚ := 72

-- Final ratio statement
theorem final_ratio_of_milk_to_water :
  (initial_milk_ratio * (total_capacity - additional_milk) + additional_milk) / (initial_water_ratio * (total_capacity - additional_milk)) = 2 := by
  sorry

end NUMINAMATH_GPT_final_ratio_of_milk_to_water_l1224_122422


namespace NUMINAMATH_GPT_cyclists_meet_after_24_minutes_l1224_122496

noncomputable def meet_time (D : ℝ) (vm vb : ℝ) : ℝ :=
  D / (2.5 * D - 12)

theorem cyclists_meet_after_24_minutes
  (D vm vb : ℝ)
  (h_vm : 1/3 * vm + 2 = D/2)
  (h_vb : 1/2 * vb = D/2 - 3) :
  meet_time D vm vb = 24 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_meet_after_24_minutes_l1224_122496


namespace NUMINAMATH_GPT_alix_more_chocolates_than_nick_l1224_122440

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end NUMINAMATH_GPT_alix_more_chocolates_than_nick_l1224_122440


namespace NUMINAMATH_GPT_grover_total_profit_l1224_122452

theorem grover_total_profit :
  let boxes := 3
  let masks_per_box := 20
  let price_per_mask := 0.50
  let cost := 15
  let total_masks := boxes * masks_per_box
  let total_revenue := total_masks * price_per_mask
  let total_profit := total_revenue - cost
  total_profit = 15 := by
sorry

end NUMINAMATH_GPT_grover_total_profit_l1224_122452


namespace NUMINAMATH_GPT_science_book_pages_l1224_122408

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end NUMINAMATH_GPT_science_book_pages_l1224_122408


namespace NUMINAMATH_GPT_card_distribution_count_l1224_122431

def card_distribution_ways : Nat := sorry

theorem card_distribution_count :
  card_distribution_ways = 9 := sorry

end NUMINAMATH_GPT_card_distribution_count_l1224_122431


namespace NUMINAMATH_GPT_knight_king_moves_incompatible_l1224_122447

-- Definitions for moves and chessboards
structure Board :=
  (numbering : Fin 64 → Nat)
  (different_board : Prop)

def knights_move (x y : Fin 64) : Prop :=
  (abs (x / 8 - y / 8) = 2 ∧ abs (x % 8 - y % 8) = 1) ∨
  (abs (x / 8 - y / 8) = 1 ∧ abs (x % 8 - y % 8) = 2)

def kings_move (x y : Fin 64) : Prop :=
  abs (x / 8 - y / 8) ≤ 1 ∧ abs (x % 8 - y % 8) ≤ 1 ∧ (x ≠ y)

-- Theorem stating the proof problem
theorem knight_king_moves_incompatible (vlad_board gosha_board : Board) (h_board_diff: vlad_board.different_board):
  ¬ ∀ i j : Fin 64, (knights_move i j ↔ kings_move (vlad_board.numbering i) (vlad_board.numbering j)) :=
by {
  -- Skipping proofs with sorry
  sorry
}

end NUMINAMATH_GPT_knight_king_moves_incompatible_l1224_122447


namespace NUMINAMATH_GPT_stationery_store_sales_l1224_122493

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end NUMINAMATH_GPT_stationery_store_sales_l1224_122493


namespace NUMINAMATH_GPT_no_solution_l1224_122457

theorem no_solution (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n :=
sorry

end NUMINAMATH_GPT_no_solution_l1224_122457


namespace NUMINAMATH_GPT_cost_of_plane_ticket_l1224_122443

theorem cost_of_plane_ticket 
  (total_cost : ℤ) (hotel_cost_per_day_per_person : ℤ) (num_people : ℤ) (num_days : ℤ) (plane_ticket_cost_per_person : ℤ) :
  total_cost = 120 →
  hotel_cost_per_day_per_person = 12 →
  num_people = 2 →
  num_days = 3 →
  (total_cost - num_people * hotel_cost_per_day_per_person * num_days) = num_people * plane_ticket_cost_per_person →
  plane_ticket_cost_per_person = 24 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_cost_of_plane_ticket_l1224_122443


namespace NUMINAMATH_GPT_total_games_in_season_l1224_122459

theorem total_games_in_season (n_teams : ℕ) (games_between_each_team : ℕ) (non_conf_games_per_team : ℕ) 
  (h_teams : n_teams = 8) (h_games_between : games_between_each_team = 3) (h_non_conf : non_conf_games_per_team = 3) :
  let games_within_league := (n_teams * (n_teams - 1) / 2) * games_between_each_team
  let games_outside_league := n_teams * non_conf_games_per_team
  games_within_league + games_outside_league = 108 := by
  sorry

end NUMINAMATH_GPT_total_games_in_season_l1224_122459


namespace NUMINAMATH_GPT_combined_weight_correct_l1224_122406

-- Define Jake's present weight
def Jake_weight : ℕ := 196

-- Define the weight loss
def weight_loss : ℕ := 8

-- Define Jake's weight after losing weight
def Jake_weight_after_loss : ℕ := Jake_weight - weight_loss

-- Define the relationship between Jake's weight after loss and his sister's weight
def sister_weight : ℕ := Jake_weight_after_loss / 2

-- Define the combined weight
def combined_weight : ℕ := Jake_weight + sister_weight

-- Prove that the combined weight is 290 pounds
theorem combined_weight_correct : combined_weight = 290 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_correct_l1224_122406


namespace NUMINAMATH_GPT_product_of_two_numbers_l1224_122445

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def greatestCommonDivisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem product_of_two_numbers (a b : ℕ) :
  leastCommonMultiple a b = 36 ∧ greatestCommonDivisor a b = 6 → a * b = 216 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1224_122445


namespace NUMINAMATH_GPT_dentist_age_is_32_l1224_122474

-- Define the conditions
def one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence (x : ℕ) : Prop :=
  (x - 8) / 6 = (x + 8) / 10

-- State the theorem
theorem dentist_age_is_32 : ∃ x : ℕ, one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence x ∧ x = 32 :=
by
  sorry

end NUMINAMATH_GPT_dentist_age_is_32_l1224_122474


namespace NUMINAMATH_GPT_correct_exp_operation_l1224_122476

theorem correct_exp_operation (a : ℝ) : (a^2 * a = a^3) := 
by
  -- Leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_correct_exp_operation_l1224_122476


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1224_122489

open Set

variable (U : Set ℤ := { -2, -1, 0, 1, 2 })
variable (A : Set ℤ := { x | 0 < Int.natAbs x ∧ Int.natAbs x < 2 })

theorem complement_of_A_in_U :
  U \ A = { -2, 0, 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1224_122489
