import Mathlib

namespace gcd_of_polynomial_and_multiple_l538_53833

theorem gcd_of_polynomial_and_multiple (b : ℕ) (hb : 714 ∣ b) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end gcd_of_polynomial_and_multiple_l538_53833


namespace problem_equivalent_l538_53824

variable (f : ℝ → ℝ)

theorem problem_equivalent (h₁ : ∀ x, deriv f x = deriv (deriv f) x)
                            (h₂ : ∀ x, deriv (deriv f) x < f x) : 
                            f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := sorry

end problem_equivalent_l538_53824


namespace earnings_difference_l538_53873

theorem earnings_difference :
  let oula_deliveries := 96
  let tona_deliveries := oula_deliveries * 3 / 4
  let area_A_fee := 100
  let area_B_fee := 125
  let area_C_fee := 150
  let oula_area_A_deliveries := 48
  let oula_area_B_deliveries := 32
  let oula_area_C_deliveries := 16
  let tona_area_A_deliveries := 27
  let tona_area_B_deliveries := 18
  let tona_area_C_deliveries := 9
  let oula_total_earnings := oula_area_A_deliveries * area_A_fee + oula_area_B_deliveries * area_B_fee + oula_area_C_deliveries * area_C_fee
  let tona_total_earnings := tona_area_A_deliveries * area_A_fee + tona_area_B_deliveries * area_B_fee + tona_area_C_deliveries * area_C_fee
  oula_total_earnings - tona_total_earnings = 4900 := by
sorry

end earnings_difference_l538_53873


namespace cos_210_eq_neg_sqrt3_over_2_l538_53830

theorem cos_210_eq_neg_sqrt3_over_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_over_2_l538_53830


namespace reciprocal_problem_l538_53888

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 5) : 150 * (x⁻¹) = 240 := 
by 
  sorry

end reciprocal_problem_l538_53888


namespace hyperbola_condition_l538_53813

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end hyperbola_condition_l538_53813


namespace stickers_at_end_of_week_l538_53876

theorem stickers_at_end_of_week (initial_stickers earned_stickers total_stickers : Nat) :
  initial_stickers = 39 →
  earned_stickers = 22 →
  total_stickers = initial_stickers + earned_stickers →
  total_stickers = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end stickers_at_end_of_week_l538_53876


namespace ages_l538_53892

-- Definitions of ages
variables (S M : ℕ) -- S: son's current age, M: mother's current age

-- Given conditions
def father_age : ℕ := 44
def son_father_relationship (S : ℕ) : Prop := father_age = S + S
def son_mother_relationship (S M : ℕ) : Prop := (S - 5) = (M - 10)

-- Theorem to prove the ages
theorem ages (S M : ℕ) (h1 : son_father_relationship S) (h2 : son_mother_relationship S M) :
  S = 22 ∧ M = 27 :=
by 
  sorry

end ages_l538_53892


namespace negation_proof_l538_53818

theorem negation_proof : ¬ (∃ x : ℝ, (x ≤ -1) ∨ (x ≥ 2)) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 := 
by 
  -- proof skipped
  sorry

end negation_proof_l538_53818


namespace smallest_gcd_bc_l538_53831

theorem smallest_gcd_bc (a b c : ℕ) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) : Nat.gcd b c = 1 :=
sorry

end smallest_gcd_bc_l538_53831


namespace average_price_of_remaining_cans_l538_53834

theorem average_price_of_remaining_cans (price_all price_returned : ℕ) (average_all average_returned : ℚ) 
    (h1 : price_all = 6) (h2 : average_all = 36.5) (h3 : price_returned = 2) (h4 : average_returned = 49.5) : 
    (price_all - price_returned) ≠ 0 → 
    4 * 30 = 6 * 36.5 - 2 * 49.5 :=
by
  intros hne
  sorry

end average_price_of_remaining_cans_l538_53834


namespace correct_option_l538_53851

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg_real (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂

theorem correct_option (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_decr : is_decreasing_on_nonneg_real f) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by
  sorry

end correct_option_l538_53851


namespace percentage_students_50_59_is_10_71_l538_53832

theorem percentage_students_50_59_is_10_71 :
  let n_90_100 := 3
  let n_80_89 := 6
  let n_70_79 := 8
  let n_60_69 := 4
  let n_50_59 := 3
  let n_below_50 := 4
  let total_students := n_90_100 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_below_50
  let fraction := (n_50_59 : ℚ) / total_students
  let percentage := (fraction * 100)
  percentage = 10.71 := by sorry

end percentage_students_50_59_is_10_71_l538_53832


namespace product_of_two_consecutive_integers_sum_lt_150_l538_53811

theorem product_of_two_consecutive_integers_sum_lt_150 :
  ∃ (n : Nat), n * (n + 1) = 5500 ∧ 2 * n + 1 < 150 :=
by
  sorry

end product_of_two_consecutive_integers_sum_lt_150_l538_53811


namespace guess_probability_greater_than_two_thirds_l538_53836

theorem guess_probability_greater_than_two_thirds :
  (1335 : ℝ) / 2002 > 2 / 3 :=
by {
  -- Placeholder for proof
  sorry
}

end guess_probability_greater_than_two_thirds_l538_53836


namespace max_value_fraction_l538_53852

theorem max_value_fraction (x : ℝ) : x ≠ 0 → 1 / (x^4 + 4*x^2 + 2 + 8/x^2 + 16/x^4) ≤ 1 / 31 :=
by sorry

end max_value_fraction_l538_53852


namespace sam_pens_count_l538_53802

-- Lean 4 statement
theorem sam_pens_count :
  ∃ (black_pens blue_pens pencils red_pens : ℕ),
    (black_pens = blue_pens + 10) ∧
    (blue_pens = 2 * pencils) ∧
    (pencils = 8) ∧
    (red_pens = pencils - 2) ∧
    (black_pens + blue_pens + red_pens = 48) :=
by {
  sorry
}

end sam_pens_count_l538_53802


namespace rhombus_area_l538_53855

theorem rhombus_area (d1 d2 : ℝ) (θ : ℝ) (h1 : d1 = 8) (h2 : d2 = 10) (h3 : Real.sin θ = 3 / 5) : 
  (1 / 2) * d1 * d2 * Real.sin θ = 24 :=
by
  sorry

end rhombus_area_l538_53855


namespace least_clock_equivalent_l538_53848

def clock_equivalent (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + 12 * k = b

theorem least_clock_equivalent (h : ℕ) (hh : h > 3) (hq : clock_equivalent h (h * h)) :
  h = 4 :=
by
  sorry

end least_clock_equivalent_l538_53848


namespace value_of_2a_minus_b_minus_4_l538_53899

theorem value_of_2a_minus_b_minus_4 (a b : ℝ) (h : 2 * a - b = 2) : 2 * a - b - 4 = -2 :=
by
  sorry

end value_of_2a_minus_b_minus_4_l538_53899


namespace time_to_pass_platform_l538_53893

-- Definitions based on the conditions
def train_length : ℕ := 1500 -- (meters)
def tree_crossing_time : ℕ := 120 -- (seconds)
def platform_length : ℕ := 500 -- (meters)

-- Define the train's speed
def train_speed := train_length / tree_crossing_time

-- Define the total distance the train needs to cover to pass the platform
def total_distance := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : 
  total_distance / train_speed = 160 :=
by sorry

end time_to_pass_platform_l538_53893


namespace mary_average_speed_l538_53895

noncomputable def trip_distance : ℝ := 1.5 + 1.5
noncomputable def trip_time_minutes : ℝ := 45 + 15
noncomputable def trip_time_hours : ℝ := trip_time_minutes / 60

theorem mary_average_speed :
  (trip_distance / trip_time_hours) = 3 := by
  sorry

end mary_average_speed_l538_53895


namespace joe_paint_usage_l538_53882

noncomputable def paint_used_after_four_weeks : ℝ := 
  let total_paint := 480
  let first_week_paint := (1/5) * total_paint
  let second_week_paint := (1/6) * (total_paint - first_week_paint)
  let third_week_paint := (1/7) * (total_paint - first_week_paint - second_week_paint)
  let fourth_week_paint := (2/9) * (total_paint - first_week_paint - second_week_paint - third_week_paint)
  first_week_paint + second_week_paint + third_week_paint + fourth_week_paint

theorem joe_paint_usage :
  abs (paint_used_after_four_weeks - 266.66) < 0.01 :=
sorry

end joe_paint_usage_l538_53882


namespace cryptarithm_solution_l538_53807

theorem cryptarithm_solution (A B : ℕ) (h_digit_A : A < 10) (h_digit_B : B < 10)
  (h_equation : 9 * (10 * A + B) = 110 * A + B) :
  A = 2 ∧ B = 5 :=
sorry

end cryptarithm_solution_l538_53807


namespace circle_equation_tangent_line1_tangent_line2_l538_53871

-- Definitions of points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)

-- Equation for the circle given the point constraints
def circle_eq : Prop := 
  ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) ↔ ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0))

-- Equations for the tangent lines passing through point P and tangent to the circle
def tangent_eq1 : Prop := 
  P.1 = 2

def tangent_eq2 : Prop :=
  4 * P.1 - 3 * P.2 + 1 = 0

-- Statements to be proven
theorem circle_equation : circle_eq := 
  sorry 

theorem tangent_line1 : tangent_eq1 := 
  sorry 

theorem tangent_line2 : tangent_eq2 := 
  sorry 

end circle_equation_tangent_line1_tangent_line2_l538_53871


namespace difference_between_length_and_breadth_l538_53816

theorem difference_between_length_and_breadth (L W : ℝ) (h1 : W = 1/2 * L) (h2 : L * W = 800) : L - W = 20 :=
by
  sorry

end difference_between_length_and_breadth_l538_53816


namespace area_R2_l538_53814

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end area_R2_l538_53814


namespace child_support_calculation_l538_53877

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l538_53877


namespace third_side_triangle_l538_53810

theorem third_side_triangle (a : ℝ) :
  (5 < a ∧ a < 13) → (a = 8) :=
sorry

end third_side_triangle_l538_53810


namespace find_k_value_l538_53820

theorem find_k_value (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ∧
  (∀ x1 x2 : ℝ, (k * x1^2 + 4 * x1 + 4 = 0 ∧ k * x2^2 + 4 * x2 + 4 = 0) → x1 = x2) →
  (k = 0 ∨ k = 1) :=
by
  intros h
  sorry

end find_k_value_l538_53820


namespace find_20_paise_coins_l538_53801

theorem find_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7100) : x = 200 :=
by
  -- Given the conditions, we need to prove x = 200.
  -- Steps and proofs are omitted here.
  sorry

end find_20_paise_coins_l538_53801


namespace correct_reflection_l538_53860

section
variable {Point : Type}
variables (PQ : Point → Point → Prop) (shaded_figure : Point → Prop)
variables (A B C D E : Point → Prop)

-- Condition: The line segment PQ is the axis of reflection.
-- Condition: The shaded figure is positioned above the line PQ and touches it at two points.
-- Define the reflection operation (assuming definitions for points and reflections are given).

def reflected (fig : Point → Prop) (axis : Point → Point → Prop) : Point → Prop := sorry  -- Define properly

-- The correct answer: The reflected figure should match figure (A).
theorem correct_reflection :
  reflected shaded_figure PQ = A :=
sorry
end

end correct_reflection_l538_53860


namespace lcm_14_18_20_l538_53880

theorem lcm_14_18_20 : Nat.lcm (Nat.lcm 14 18) 20 = 1260 :=
by
  -- Define the prime factorizations
  have fact_14 : 14 = 2 * 7 := by norm_num
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_20 : 20 = 2^2 * 5 := by norm_num
  
  -- Calculate the LCM based on the highest powers of each prime
  have lcm : Nat.lcm (Nat.lcm 14 18) 20 = 2^2 * 3^2 * 5 * 7 :=
    by
      sorry -- Proof details are not required

  -- Final verification that this calculation matches 1260
  exact lcm

end lcm_14_18_20_l538_53880


namespace empty_atm_l538_53879

theorem empty_atm (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 9 < b 9)
    (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ 8 → a k ≠ b k) 
    (n : ℕ) (h₀ : n = 1) : 
    ∃ (sequence : ℕ → ℕ), (∀ i, sequence i ≤ n) → (∀ k, ∃ i, k > i → sequence k = 0) :=
sorry

end empty_atm_l538_53879


namespace arabella_dance_steps_l538_53849

theorem arabella_dance_steps :
  exists T1 T2 T3 : ℕ,
    T1 = 30 ∧
    T3 = T1 + T2 ∧
    T1 + T2 + T3 = 90 ∧
    (T2 / T1 : ℚ) = 1 / 2 :=
by
  sorry

end arabella_dance_steps_l538_53849


namespace pencils_ratio_l538_53825

theorem pencils_ratio (T S Ti : ℕ) 
  (h1 : T = 6 * S)
  (h2 : T = 12)
  (h3 : Ti = 16) : Ti / S = 8 := by
  sorry

end pencils_ratio_l538_53825


namespace magnitude_b_magnitude_c_area_l538_53857

-- Define the triangle ABC and parameters
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi_div2 : A < Real.pi / 2)
variables (triangle_condition : a = Real.sqrt 15) (sin_A : Real.sin A = 1 / 4)

-- Problem 1
theorem magnitude_b (cos_B : Real.cos B = Real.sqrt 5 / 3) :
  b = (8 * Real.sqrt 15) / 3 := by
  sorry

-- Problem 2
theorem magnitude_c_area (b_eq_4a : b = 4 * a) :
  c = 15 ∧ (1 / 2 * b * c * Real.sin A = (15 / 2) * Real.sqrt 15) := by
  sorry

end magnitude_b_magnitude_c_area_l538_53857


namespace kai_ice_plate_division_l538_53838

-- Define the "L"-shaped ice plate with given dimensions
structure LShapedIcePlate (a : ℕ) :=
(horiz_length : ℕ)
(vert_length : ℕ)
(horiz_eq_vert : horiz_length = a ∧ vert_length = a)

-- Define the correctness of dividing the L-shaped plate into four equal parts
def can_be_divided_into_four_equal_parts (a : ℕ) (piece : LShapedIcePlate a) : Prop :=
∃ cut_points_v1 cut_points_v2 cut_points_h1 cut_points_h2,
  -- The cut points for vertical and horizontal cuts to turn the large "L" shape into four smaller "L" shapes
  piece.horiz_length = cut_points_v1 + cut_points_v2 ∧
  piece.vert_length = cut_points_h1 + cut_points_h2 ∧
  cut_points_v1 = a / 2 ∧ cut_points_v2 = a - a / 2 ∧
  cut_points_h1 = a / 2 ∧ cut_points_h2 = a - a / 2

-- Prove the main theorem
theorem kai_ice_plate_division (a : ℕ) (h : a > 0) (plate : LShapedIcePlate a) : 
  can_be_divided_into_four_equal_parts a plate :=
sorry

end kai_ice_plate_division_l538_53838


namespace percent_correct_both_l538_53809

-- Definitions based on given conditions in the problem
def P_A : ℝ := 0.63
def P_B : ℝ := 0.50
def P_not_A_and_not_B : ℝ := 0.20

-- Definition of the desired result using the inclusion-exclusion principle based on the given conditions
def P_A_and_B : ℝ := P_A + P_B - (1 - P_not_A_and_not_B)

-- Theorem stating our goal: proving the probability of both answering correctly is 0.33
theorem percent_correct_both : P_A_and_B = 0.33 := by
  sorry

end percent_correct_both_l538_53809


namespace no_valid_prime_pairs_l538_53865

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_valid_prime_pairs :
  ∀ x y : ℕ, is_prime x → is_prime y → y < x → x ≤ 200 → (x % y = 0) → ((x +1) % (y +1) = 0) → false :=
by
  sorry

end no_valid_prime_pairs_l538_53865


namespace point_on_coordinate_axes_l538_53854

-- Definitions and assumptions from the problem conditions
variables {a b : ℝ}

-- The theorem statement asserts that point M(a, b) must be located on the coordinate axes given ab = 0
theorem point_on_coordinate_axes (h : a * b = 0) : 
  (a = 0) ∨ (b = 0) :=
by
  sorry

end point_on_coordinate_axes_l538_53854


namespace compute_ratio_l538_53837

theorem compute_ratio (x y z a : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x + y + z = a) (h5 : a ≠ 0) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 1 / 3 :=
by
  -- Proof will be filled in here
  sorry

end compute_ratio_l538_53837


namespace value_of_x_l538_53891

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end value_of_x_l538_53891


namespace gcd_lcm_sum_correct_l538_53803

def gcd_lcm_sum : ℕ :=
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  gcd_40_60 + 2 * lcm_20_15

theorem gcd_lcm_sum_correct : gcd_lcm_sum = 140 := by
  -- Definitions based on conditions
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  
  -- sorry to skip the proof
  sorry

end gcd_lcm_sum_correct_l538_53803


namespace correct_addition_result_l538_53878

-- Definitions corresponding to the conditions
def mistaken_addend := 240
def correct_addend := 420
def incorrect_sum := 390

-- The proof statement
theorem correct_addition_result : 
  (incorrect_sum - mistaken_addend + correct_addend) = 570 :=
by
  sorry

end correct_addition_result_l538_53878


namespace find_number_l538_53889

theorem find_number (x : ℝ) : (0.75 * x = 0.45 * 1500 + 495) -> x = 1560 :=
by
  sorry

end find_number_l538_53889


namespace max_min_values_f_decreasing_interval_f_l538_53827

noncomputable def a : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := ((a.1 * (b x).1) + (a.2 * (b x).2)) + 2

theorem max_min_values_f (k : ℤ) :
  (∃ (x1 : ℝ), (x1 = 2 * k * Real.pi + Real.pi / 6) ∧ f x1 = 3) ∧
  (∃ (x2 : ℝ), (x2 = 2 * k * Real.pi - 5 * Real.pi / 6) ∧ f x2 = 1) := 
sorry

theorem decreasing_interval_f :
  ∀ x, (Real.pi / 6 ≤ x ∧ x ≤ 7 * Real.pi / 6) → (∀ y, f x ≥ f y → x ≤ y) := 
sorry

end max_min_values_f_decreasing_interval_f_l538_53827


namespace range_of_m_l538_53867

open Set Real

noncomputable def f (x m : ℝ) : ℝ := abs (x^2 - 4 * x + 9 - 2 * m) + 2 * m

theorem range_of_m
  (h1 : ∀ x ∈ Icc (0 : ℝ) 4, f x m ≤ 9) : m ≤ 7 / 2 :=
by
  sorry

end range_of_m_l538_53867


namespace surface_area_of_circumscribing_sphere_l538_53822

theorem surface_area_of_circumscribing_sphere :
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  A = 17 * Real.pi :=
by
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  show A = 17 * Real.pi
  sorry

end surface_area_of_circumscribing_sphere_l538_53822


namespace smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l538_53806

/-- Define what it means for a number to be a prime greater than 3 -/
def is_prime_gt_3 (n : ℕ) : Prop :=
  Prime n ∧ 3 < n

/-- Define a scalene triangle with side lengths that are distinct primes greater than 3 -/
def is_scalene_triangle_with_distinct_primes (a b c : ℕ) : Prop :=
  is_prime_gt_3 a ∧ is_prime_gt_3 b ∧ is_prime_gt_3 c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The proof problem statement -/
theorem smallest_possible_perimeter_of_scalene_triangle_with_prime_sides :
  ∃ (a b c : ℕ), is_scalene_triangle_with_distinct_primes a b c ∧ Prime (a + b + c) ∧ (a + b + c = 23) :=
sorry

end smallest_possible_perimeter_of_scalene_triangle_with_prime_sides_l538_53806


namespace arithmetic_seq_a12_l538_53894

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Prove that a_12 = 12 given the conditions
theorem arithmetic_seq_a12 :
  ∃ a₁, (arithmetic_seq a₁ 2 2 = -8) → (arithmetic_seq a₁ 2 12 = 12) :=
by
  sorry

end arithmetic_seq_a12_l538_53894


namespace investment_share_l538_53868

variable (P_investment Q_investment : ℝ)

theorem investment_share (h1 : Q_investment = 60000) (h2 : P_investment / Q_investment = 2 / 3) : P_investment = 40000 := by
  sorry

end investment_share_l538_53868


namespace x_intercept_of_line_is_7_over_2_l538_53884

-- Definitions for the conditions
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (6, 5)

-- Define what it means to be the x-intercept of the line
def x_intercept_of_line (x : ℝ) : Prop :=
  ∃ m b : ℝ, (point1.snd) = m * (point1.fst) + b ∧ (point2.snd) = m * (point2.fst) + b ∧ 0 = m * x + b

-- The theorem stating the x-intercept
theorem x_intercept_of_line_is_7_over_2 : x_intercept_of_line (7 / 2) :=
sorry

end x_intercept_of_line_is_7_over_2_l538_53884


namespace least_common_denominator_l538_53826

-- Define the list of numbers
def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the least common multiple function
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the main theorem
theorem least_common_denominator : lcm_list numbers = 2520 := 
  by sorry

end least_common_denominator_l538_53826


namespace number_of_sides_l538_53846

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l538_53846


namespace roll_probability_l538_53885

noncomputable def probability_allison_rolls_greater : ℚ :=
  let p_brian := 5 / 6  -- Probability of Brian rolling 5 or lower
  let p_noah := 1       -- Probability of Noah rolling 5 or lower (since all faces roll 5 or lower)
  p_brian * p_noah

theorem roll_probability :
  probability_allison_rolls_greater = 5 / 6 := by
  sorry

end roll_probability_l538_53885


namespace vector_product_magnitude_l538_53881

noncomputable def vector_magnitude (a b : ℝ) (theta : ℝ) : ℝ :=
  abs a * abs b * Real.sin theta

theorem vector_product_magnitude 
  (a b : ℝ) 
  (theta : ℝ) 
  (ha : abs a = 4) 
  (hb : abs b = 3) 
  (h_dot : a * b = -2) 
  (theta_range : 0 ≤ theta ∧ theta ≤ Real.pi)
  (cos_theta : Real.cos theta = -1/6) 
  (sin_theta : Real.sin theta = Real.sqrt 35 / 6) :
  vector_magnitude a b theta = 2 * Real.sqrt 35 :=
sorry

end vector_product_magnitude_l538_53881


namespace complex_distance_l538_53839

theorem complex_distance (i : Complex) (h : i = Complex.I) :
  Complex.abs (3 / (2 - i)^2) = 3 / 5 := 
by
  sorry

end complex_distance_l538_53839


namespace downstream_distance_l538_53842

theorem downstream_distance
    (speed_still_water : ℝ)
    (current_rate : ℝ)
    (travel_time_minutes : ℝ)
    (h_still_water : speed_still_water = 20)
    (h_current_rate : current_rate = 4)
    (h_travel_time : travel_time_minutes = 24) :
    (speed_still_water + current_rate) * (travel_time_minutes / 60) = 9.6 :=
by
  -- Proof goes here
  sorry

end downstream_distance_l538_53842


namespace find_parameters_infinite_solutions_l538_53898

def system_has_infinite_solutions (a b : ℝ) :=
  ∀ x y : ℝ, 2 * (a - b) * x + 6 * y = a ∧ 3 * b * x + (a - b) * b * y = 1

theorem find_parameters_infinite_solutions :
  ∀ (a b : ℝ), 
  system_has_infinite_solutions a b ↔ 
    (a = (3 + Real.sqrt 17) / 2 ∧ b = (Real.sqrt 17 - 3) / 2) ∨
    (a = (3 - Real.sqrt 17) / 2 ∧ b = (-3 - Real.sqrt 17) / 2) ∨
    (a = -2 ∧ b = 1) ∨
    (a = -1 ∧ b = 2) :=
sorry

end find_parameters_infinite_solutions_l538_53898


namespace steve_speed_ratio_l538_53897

variable (distance : ℝ)
variable (total_time : ℝ)
variable (speed_back : ℝ)
variable (speed_to : ℝ)

noncomputable def speed_ratio (distance : ℝ) (total_time : ℝ) (speed_back : ℝ) : ℝ := 
  let time_to := total_time - distance / speed_back
  let speed_to := distance / time_to
  speed_back / speed_to

theorem steve_speed_ratio (h1 : distance = 10) (h2 : total_time = 6) (h3 : speed_back = 5) :
  speed_ratio distance total_time speed_back = 2 := by
  sorry

end steve_speed_ratio_l538_53897


namespace no_real_roots_l538_53845

theorem no_real_roots (k : ℝ) (h : k ≠ 0) : ¬∃ x : ℝ, x^2 + k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_l538_53845


namespace rows_identical_l538_53829

theorem rows_identical {n : ℕ} {a : Fin n → ℝ} {k : Fin n → Fin n}
  (h_inc : ∀ i j : Fin n, i < j → a i < a j)
  (h_perm : ∀ i j : Fin n, k i ≠ k j → a (k i) ≠ a (k j))
  (h_sum_inc : ∀ i j : Fin n, i < j → a i + a (k i) < a j + a (k j)) :
  ∀ i : Fin n, a i = a (k i) :=
by
  sorry

end rows_identical_l538_53829


namespace company_max_revenue_l538_53808

structure Conditions where
  max_total_time : ℕ -- maximum total time in minutes
  max_total_cost : ℕ -- maximum total cost in yuan
  rate_A : ℕ -- rate per minute for TV A in yuan
  rate_B : ℕ -- rate per minute for TV B in yuan
  revenue_A : ℕ -- revenue per minute for TV A in million yuan
  revenue_B : ℕ -- revenue per minute for TV B in million yuan

def company_conditions : Conditions :=
  { max_total_time := 300,
    max_total_cost := 90000,
    rate_A := 500,
    rate_B := 200,
    revenue_A := 3, -- as 0.3 million yuan converted to 3 tenths (integer representation)
    revenue_B := 2  -- as 0.2 million yuan converted to 2 tenths (integer representation)
  }

def advertising_strategy
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : Prop :=
  time_A + time_B ≤ conditions.max_total_time ∧
  time_A * conditions.rate_A + time_B * conditions.rate_B ≤ conditions.max_total_cost

def revenue
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : ℕ :=
  time_A * conditions.revenue_A + time_B * conditions.revenue_B

theorem company_max_revenue (time_A time_B : ℕ)
  (h : advertising_strategy company_conditions time_A time_B) :
  revenue company_conditions time_A time_B = 70 := 
  by
  have h1 : time_A = 100 := sorry
  have h2 : time_B = 200 := sorry
  sorry

end company_max_revenue_l538_53808


namespace hours_spent_gaming_l538_53828

def total_hours_in_day : ℕ := 24

def sleeping_fraction : ℚ := 1/3

def studying_fraction : ℚ := 3/4

def gaming_fraction : ℚ := 1/4

theorem hours_spent_gaming :
  let sleeping_hours := total_hours_in_day * sleeping_fraction
  let remaining_hours_after_sleeping := total_hours_in_day - sleeping_hours
  let studying_hours := remaining_hours_after_sleeping * studying_fraction
  let remaining_hours_after_studying := remaining_hours_after_sleeping - studying_hours
  remaining_hours_after_studying * gaming_fraction = 1 :=
by
  sorry

end hours_spent_gaming_l538_53828


namespace circle_radius_l538_53815

/-- Let a circle have a maximum distance of 11 cm and a minimum distance of 5 cm from a point P.
Prove that the radius of the circle can be either 3 cm or 8 cm. -/
theorem circle_radius (max_dist min_dist : ℕ) (h_max : max_dist = 11) (h_min : min_dist = 5) :
  (∃ r : ℕ, r = 3 ∨ r = 8) :=
by
  sorry

end circle_radius_l538_53815


namespace inverse_proportion_quad_l538_53844

theorem inverse_proportion_quad (k : ℝ) : (∀ x : ℝ, x > 0 → (k + 1) / x < 0) ∧ (∀ x : ℝ, x < 0 → (k + 1) / x > 0) ↔ k < -1 :=
by
  sorry

end inverse_proportion_quad_l538_53844


namespace monotonic_decreasing_interval_l538_53805

open Real

noncomputable def decreasing_interval (k: ℤ): Set ℝ :=
  {x | k * π - π / 3 < x ∧ x < k * π + π / 6 }

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, x ∈ decreasing_interval k ↔ (k * π - π / 3 < x ∧ x < k * π + π / 6) :=
by 
  intros x
  sorry

end monotonic_decreasing_interval_l538_53805


namespace sum_of_number_and_reverse_divisible_by_11_l538_53847

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) (hA : 0 ≤ A) (hA9 : A ≤ 9) (hB : 0 ≤ B) (hB9 : B ≤ 9) :
  11 ∣ ((10 * A + B) + (10 * B + A)) :=
by
  sorry

end sum_of_number_and_reverse_divisible_by_11_l538_53847


namespace max_product_l538_53890

-- Problem statement: Define the conditions and the conclusion
theorem max_product (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 4) : mn ≤ 4 :=
by
  sorry -- Proof placeholder

end max_product_l538_53890


namespace john_will_lose_weight_in_80_days_l538_53864

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l538_53864


namespace probability_of_same_color_when_rolling_two_24_sided_dice_l538_53874

-- Defining the conditions
def numSides : ℕ := 24
def purpleSides : ℕ := 5
def blueSides : ℕ := 8
def redSides : ℕ := 10
def goldSides : ℕ := 1

-- Required to use rational numbers for probabilities
def probability (eventSides : ℕ) (totalSides : ℕ) : ℚ := eventSides / totalSides

-- Main theorem statement
theorem probability_of_same_color_when_rolling_two_24_sided_dice :
  probability purpleSides numSides * probability purpleSides numSides +
  probability blueSides numSides * probability blueSides numSides +
  probability redSides numSides * probability redSides numSides +
  probability goldSides numSides * probability goldSides numSides =
  95 / 288 :=
by
  sorry

end probability_of_same_color_when_rolling_two_24_sided_dice_l538_53874


namespace quotient_of_division_l538_53800

theorem quotient_of_division (L : ℕ) (S : ℕ) (Q : ℕ) (h1 : L = 1631) (h2 : L - S = 1365) (h3 : L = S * Q + 35) :
  Q = 6 :=
by
  sorry

end quotient_of_division_l538_53800


namespace units_digit_sum_factorials_500_l538_53859

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end units_digit_sum_factorials_500_l538_53859


namespace smallest_whole_number_divisible_by_8_leaves_remainder_1_l538_53861

theorem smallest_whole_number_divisible_by_8_leaves_remainder_1 :
  ∃ (n : ℕ), n ≡ 1 [MOD 2] ∧ n ≡ 1 [MOD 3] ∧ n ≡ 1 [MOD 4] ∧ n ≡ 1 [MOD 5] ∧ n ≡ 1 [MOD 7] ∧ n % 8 = 0 ∧ n = 7141 :=
by
  sorry

end smallest_whole_number_divisible_by_8_leaves_remainder_1_l538_53861


namespace cosine_120_eq_negative_half_l538_53887

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l538_53887


namespace chloe_total_score_l538_53843

def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

def score_first_level : ℕ := treasures_first_level * points_per_treasure
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := score_first_level + score_second_level

theorem chloe_total_score : total_score = 81 := by
  sorry

end chloe_total_score_l538_53843


namespace rectangle_area_from_square_area_and_proportions_l538_53896

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end rectangle_area_from_square_area_and_proportions_l538_53896


namespace y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l538_53840

variable (x y : ℝ)

-- Condition: y is defined as a function of x
def y_def := y = 2 * x + 5

-- Theorem: y > 0 if and only if x > -5/2
theorem y_positive_if_and_only_if_x_greater_than_negative_five_over_two 
  (h : y_def x y) : y > 0 ↔ x > -5 / 2 := by sorry

end y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l538_53840


namespace friend_gives_30_l538_53883

noncomputable def total_earnings := 10 + 30 + 50 + 40 + 70

noncomputable def equal_share := total_earnings / 5

noncomputable def contribution_of_highest_earner := 70

noncomputable def amount_to_give := contribution_of_highest_earner - equal_share

theorem friend_gives_30 : amount_to_give = 30 := by
  sorry

end friend_gives_30_l538_53883


namespace jellybean_ratio_l538_53817

theorem jellybean_ratio (gigi_je : ℕ) (rory_je : ℕ) (lorelai_je : ℕ) (h_gigi : gigi_je = 15) (h_rory : rory_je = gigi_je + 30) (h_lorelai : lorelai_je = 180) : lorelai_je / (rory_je + gigi_je) = 3 :=
by
  -- Introduce the given hypotheses
  rw [h_gigi, h_rory, h_lorelai]
  -- Simplify the expression
  sorry

end jellybean_ratio_l538_53817


namespace number_of_female_fish_l538_53862

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l538_53862


namespace complete_the_square_l538_53821

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x - 10 = 0

-- State the proof problem
theorem complete_the_square (x : ℝ) (h : quadratic_eq x) : (x - 3)^2 = 19 :=
by 
  -- Skip the proof using sorry
  sorry

end complete_the_square_l538_53821


namespace frosting_time_difference_l538_53812

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l538_53812


namespace pages_in_book_l538_53819

-- Define the initial conditions
variable (P : ℝ) -- total number of pages in the book
variable (h_read_20_percent : 0.20 * P = 320 * 0.20 / 0.80) -- Nate has read 20% of the book and the rest 80%

-- The goal is to show that P = 400
theorem pages_in_book (P : ℝ) :
  (0.80 * P = 320) → P = 400 :=
by
  sorry

end pages_in_book_l538_53819


namespace determinant_of_given_matrix_l538_53866

-- Define the given matrix
def given_matrix (z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![z + 2, z, z], ![z, z + 3, z], ![z, z, z + 4]]

-- Define the proof statement
theorem determinant_of_given_matrix (z : ℂ) : Matrix.det (given_matrix z) = 22 * z + 24 :=
by
  sorry

end determinant_of_given_matrix_l538_53866


namespace daily_construction_areas_minimum_area_A_must_build_l538_53850

-- Definitions based on conditions and questions
variable {area : ℕ}
variable {daily_A : ℕ}
variable {daily_B : ℕ}
variable (h_area : area = 5100)
variable (h_A_B_diff : daily_A = daily_B + 2)
variable (h_A_days : 900 / daily_A = 720 / daily_B)

-- Proof statements for the questions in the problem
theorem daily_construction_areas (daily_B : ℕ) (daily_A : ℕ) :
  daily_B = 8 ∧ daily_A = 10 :=
by sorry

theorem minimum_area_A_must_build (daily_A : ℕ) (daily_B : ℕ) (area_A : ℕ) :
  (area_A ≥ 2 * (5100 - area_A)) → (area_A ≥ 3400) :=
by sorry

end daily_construction_areas_minimum_area_A_must_build_l538_53850


namespace question_eq_answer_l538_53863

theorem question_eq_answer (w x y z k : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2520) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 24 :=
sorry

end question_eq_answer_l538_53863


namespace solve_equation_l538_53835

theorem solve_equation (x : ℝ) (h : (3 * x) / (x + 1) = 9 / (x + 1)) : x = 3 :=
by sorry

end solve_equation_l538_53835


namespace translation_2_units_left_l538_53841

-- Define the initial parabola
def parabola1 (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 5

-- State that parabola2 is obtained by translating parabola1
-- And prove that this translation is 2 units to the left
theorem translation_2_units_left :
  ∀ x : ℝ, parabola2 x = parabola1 (x + 2) := 
by
  sorry

end translation_2_units_left_l538_53841


namespace sequence_mono_iff_b_gt_neg3_l538_53872

theorem sequence_mono_iff_b_gt_neg3 (b : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → (n + 1) ^ 2 + b * (n + 1) > n ^ 2 + b * n) → b > -3 := 
by
  sorry

end sequence_mono_iff_b_gt_neg3_l538_53872


namespace cosine_difference_formula_l538_53853

theorem cosine_difference_formula
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < (Real.pi / 2))
  (h3 : Real.tan α = 2) :
  Real.cos (α - (Real.pi / 4)) = (3 * Real.sqrt 10) / 10 := 
by
  sorry

end cosine_difference_formula_l538_53853


namespace max_distance_from_center_of_square_l538_53858

theorem max_distance_from_center_of_square :
  let A := (0, 0)
  let B := (1, 0)
  let C := (1, 1)
  let D := (0, 1)
  let O := (0.5, 0.5)
  ∃ P : ℝ × ℝ, 
  (let u := dist P A
   let v := dist P B
   let w := dist P C
   u^2 + v^2 + w^2 = 2)
  → dist O P = (1 + 2 * Real.sqrt 2) / (3 * Real.sqrt 2) :=
by sorry

end max_distance_from_center_of_square_l538_53858


namespace sum_of_distinct_abc_eq_roots_l538_53875

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 * ((x + 2*y)^2 - y^2 + x - 1)

-- Main theorem statement
theorem sum_of_distinct_abc_eq_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : f a (b+c) = f b (c+a)) (h2 : f b (c+a) = f c (a+b)) :
  a + b + c = (1 + Real.sqrt 5) / 2 ∨ a + b + c = (1 - Real.sqrt 5) / 2 :=
sorry

end sum_of_distinct_abc_eq_roots_l538_53875


namespace min_transfers_to_uniform_cards_l538_53823

theorem min_transfers_to_uniform_cards (n : ℕ) (h : n = 101) (s : Fin n) :
  ∃ k : ℕ, (∀ s1 s2 : Fin n → ℕ, 
    (∀ i, s1 i = i + 1) ∧ (∀ j, s2 j = 51) → -- Initial and final conditions
    k ≤ 42925) := 
sorry

end min_transfers_to_uniform_cards_l538_53823


namespace infinite_series_sum_l538_53870

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l538_53870


namespace total_heads_of_cabbage_l538_53856

-- Problem definition for the first patch
def first_patch : ℕ := 12 * 15

-- Problem definition for the second patch
def second_patch : ℕ := 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24

-- Problem statement
theorem total_heads_of_cabbage : first_patch + second_patch = 316 := by
  sorry

end total_heads_of_cabbage_l538_53856


namespace triangle_acute_angle_l538_53804

theorem triangle_acute_angle 
  (a b c : ℝ) 
  (h1 : a^3 = b^3 + c^3)
  (h2 : a > b)
  (h3 : a > c)
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : a > 0) 
  : 
  (a^2 < b^2 + c^2) :=
sorry

end triangle_acute_angle_l538_53804


namespace ellipse_chord_through_focus_l538_53869

theorem ellipse_chord_through_focus (x y : ℝ) (a b : ℝ := 6) (c : ℝ := 3 * Real.sqrt 3)
  (F : ℝ × ℝ := (3 * Real.sqrt 3, 0)) (AF BF : ℝ) :
  (x^2 / 36) + (y^2 / 9) = 1 ∧ ((x - 3 * Real.sqrt 3)^2 + y^2 = (3/2)^2) ∧
  (AF = 3 / 2) ∧ F.1 = 3 * Real.sqrt 3 ∧ F.2 = 0 →
  BF = 3 / 2 :=
sorry

end ellipse_chord_through_focus_l538_53869


namespace acquaintances_at_ends_equal_l538_53886

theorem acquaintances_at_ends_equal 
  (n : ℕ) -- number of participants
  (a b : ℕ → ℕ) -- functions which return the number of acquaintances before/after for each participant
  (h_ai_bi : ∀ (i : ℕ), 1 < i ∧ i < n → a i = b i) -- condition for participants except first and last
  (h_a1 : a 1 = 0) -- the first person has no one before them
  (h_bn : b n = 0) -- the last person has no one after them
  :
  a n = b 1 :=
by
  sorry

end acquaintances_at_ends_equal_l538_53886
