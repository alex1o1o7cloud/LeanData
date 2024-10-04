import Mathlib

namespace complex_z_modulus_l250_250499

open Complex

theorem complex_z_modulus (z : ℂ) (h1 : (z + 2 * I).re = z + 2 * I) (h2 : (z / (2 - I)).re = z / (2 - I)) :
  (z = 4 - 2 * I) ∧ abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end complex_z_modulus_l250_250499


namespace find_f_2017_l250_250655

def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_shifted : ∀ x : ℝ, f (1 - x) = f (x + 1)
axiom f_neg_one : f (-1) = 2

theorem find_f_2017 : f 2017 = -2 :=
by
  sorry

end find_f_2017_l250_250655


namespace laptop_weight_l250_250121

-- Defining the weights
variables (B U L P : ℝ)
-- Karen's tote weight
def K := 8

-- Conditions from the problem
axiom tote_eq_two_briefcase : K = 2 * B
axiom umbrella_eq_half_briefcase : U = B / 2
axiom full_briefcase_eq_double_tote : B + L + P + U = 2 * K
axiom papers_eq_sixth_full_briefcase : P = (B + L + P) / 6

-- Theorem stating the weight of Kevin's laptop is 7.67 pounds
theorem laptop_weight (hB : B = 4) (hU : U = 2) (hL : L = 7.67) : 
  L - K = -0.33 :=
by
  sorry

end laptop_weight_l250_250121


namespace number_of_sacks_after_49_days_l250_250838

def sacks_per_day : ℕ := 38
def days_of_harvest : ℕ := 49
def total_sacks_after_49_days : ℕ := 1862

theorem number_of_sacks_after_49_days :
  sacks_per_day * days_of_harvest = total_sacks_after_49_days :=
by
  sorry

end number_of_sacks_after_49_days_l250_250838


namespace part_a_l250_250323

theorem part_a (m n : ℕ) (hm : m > 1) : n ∣ Nat.totient (m^n - 1) :=
sorry

end part_a_l250_250323


namespace election_votes_and_deposit_l250_250102

theorem election_votes_and_deposit (V : ℕ) (A B C D E : ℕ) (hA : A = 40 * V / 100) 
  (hB : B = 28 * V / 100) (hC : C = 20 * V / 100) (hDE : D + E = 12 * V / 100)
  (win_margin : A - B = 500) :
  V = 4167 ∧ (15 * V / 100 ≤ A) ∧ (15 * V / 100 ≤ B) ∧ (15 * V / 100 ≤ C) ∧ 
  ¬ (15 * V / 100 ≤ D) ∧ ¬ (15 * V / 100 ≤ E) :=
by 
  sorry

end election_votes_and_deposit_l250_250102


namespace cookies_per_bag_l250_250099

theorem cookies_per_bag (b T : ℕ) (h1 : b = 37) (h2 : T = 703) : (T / b) = 19 :=
by
  -- Placeholder for proof
  sorry

end cookies_per_bag_l250_250099


namespace part_a_part_b_l250_250328

section
-- Definitions based on the conditions
variable (n : ℕ)  -- Variable n representing the number of cities

-- Given a condition function T_n that returns an integer (number of ways to build roads)
def T_n (n : ℕ) : ℕ := sorry  -- Definition placeholder for T_n function

-- Part (a): For all odd n, T_n(n) is divisible by n
theorem part_a (hn : n % 2 = 1) : T_n n % n = 0 := sorry

-- Part (b): For all even n, T_n(n) is divisible by n / 2
theorem part_b (hn : n % 2 = 0) : T_n n % (n / 2) = 0 := sorry

end

end part_a_part_b_l250_250328


namespace unit_vector_opposite_AB_is_l250_250822

open Real

noncomputable def unit_vector_opposite_dir (A B : ℝ × ℝ) : ℝ × ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BA := (-AB.1, -AB.2)
  let mag_BA := sqrt (BA.1^2 + BA.2^2)
  (BA.1 / mag_BA, BA.2 / mag_BA)

theorem unit_vector_opposite_AB_is (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (-2, 6)) :
  unit_vector_opposite_dir A B = (3/5, -4/5) :=
by
  sorry

end unit_vector_opposite_AB_is_l250_250822


namespace minimum_odd_correct_answers_l250_250038

theorem minimum_odd_correct_answers (students : Fin 50 → Fin 5) :
  (∀ S : Finset (Fin 50), S.card = 40 → 
    (∃ x ∈ S, students x = 3) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, students x₁ = 2 ∧ x₁ ≠ x₂ ∧ students x₂ = 2) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, students x₁ = 1 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ students x₂ = 1 ∧ students x₃ = 1) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, ∃ x₄ ∈ S, students x₁ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ students x₂ = 0 ∧ students x₃ = 0 ∧ students x₄ = 0)) →
  (∃ S : Finset (Fin 50), (∀ x ∈ S, (students x = 1 ∨ students x = 3)) ∧ S.card = 23) :=
by
  sorry

end minimum_odd_correct_answers_l250_250038


namespace total_nominal_income_l250_250285

theorem total_nominal_income :
  let principal := 8700
  let rate := 0.06 / 12
  let income (n : ℕ) := principal * ((1 + rate) ^ n - 1)
  income 6 + income 5 + income 4 + income 3 + income 2 + income 1 = 921.15 := by
  sorry

end total_nominal_income_l250_250285


namespace sparrows_initial_count_l250_250891

theorem sparrows_initial_count (a b c : ℕ) 
  (h1 : a + b + c = 24)
  (h2 : a - 4 = b + 1)
  (h3 : b + 1 = c + 3) : 
  a = 12 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end sparrows_initial_count_l250_250891


namespace speed_of_second_train_l250_250927

theorem speed_of_second_train
  (t₁ : ℕ := 2)  -- Time the first train sets off (2:00 pm in hours)
  (s₁ : ℝ := 70) -- Speed of the first train in km/h
  (t₂ : ℕ := 3)  -- Time the second train sets off (3:00 pm in hours)
  (t₃ : ℕ := 10) -- Time when the second train catches the first train (10:00 pm in hours)
  : ∃ S : ℝ, S = 80 := sorry

end speed_of_second_train_l250_250927


namespace inequality_range_l250_250842

theorem inequality_range (a : ℝ) (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 := by
  sorry

end inequality_range_l250_250842


namespace workers_together_complete_work_in_14_days_l250_250606

noncomputable def efficiency (Wq : ℝ) := 1.4 * Wq

def work_done_in_one_day_p (Wp : ℝ) := Wp = 1 / 24

noncomputable def work_done_in_one_day_q (Wq : ℝ) := Wq = (1 / 24) / 1.4

noncomputable def combined_work_per_day (Wp Wq : ℝ) := Wp + Wq

noncomputable def days_to_complete_work (W : ℝ) := 1 / W

theorem workers_together_complete_work_in_14_days (Wp Wq : ℝ) 
  (h1 : Wp = efficiency Wq)
  (h2 : work_done_in_one_day_p Wp)
  (h3 : work_done_in_one_day_q Wq) :
  days_to_complete_work (combined_work_per_day Wp Wq) = 14 := 
sorry

end workers_together_complete_work_in_14_days_l250_250606


namespace find_integer_of_divisors_l250_250567

theorem find_integer_of_divisors:
  ∃ (N : ℕ), (∀ (l m n : ℕ), N = (2^l) * (3^m) * (5^n) → 
  (2^120) * (3^60) * (5^90) = (2^l * 3^m * 5^n)^( ((l+1)*(m+1)*(n+1)) / 2 ) ) → 
  N = 18000 :=
sorry

end find_integer_of_divisors_l250_250567


namespace mr_green_garden_yield_l250_250683

noncomputable def garden_yield (steps_length steps_width step_length yield_per_sqft : ℝ) : ℝ :=
  let length_ft := steps_length * step_length
  let width_ft := steps_width * step_length
  let area := length_ft * width_ft
  area * yield_per_sqft

theorem mr_green_garden_yield :
  garden_yield 18 25 2.5 0.5 = 1406.25 :=
by
  sorry

end mr_green_garden_yield_l250_250683


namespace find_y_l250_250637

theorem find_y (y : ℝ) (h : |2 * y - 44| + |y - 24| = |3 * y - 66|) : y = 23 := 
by 
  sorry

end find_y_l250_250637


namespace largest_x_l250_250781

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l250_250781


namespace find_x_eq_5_over_3_l250_250220

def g (x : ℝ) : ℝ := 4 * x - 5

def g_inv (y : ℝ) : ℝ := (y + 5) / 4

theorem find_x_eq_5_over_3 (x : ℝ) (hx : g x = g_inv x) : x = 5 / 3 :=
by
  sorry

end find_x_eq_5_over_3_l250_250220


namespace distance_light_travels_100_years_l250_250413

def distance_light_travels_one_year : ℝ := 5870e9 * 10^3

theorem distance_light_travels_100_years : distance_light_travels_one_year * 100 = 587 * 10^12 :=
by
  rw [distance_light_travels_one_year]
  sorry

end distance_light_travels_100_years_l250_250413


namespace find_m_eq_zero_l250_250393

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l250_250393


namespace small_cubes_for_larger_cube_l250_250732

theorem small_cubes_for_larger_cube (VL VS : ℕ) (h : VL = 125 * VS) : (VL / VS = 125) :=
by {
    sorry
}

end small_cubes_for_larger_cube_l250_250732


namespace angle_halving_quadrant_l250_250505

theorem angle_halving_quadrant (k : ℤ) (α : ℝ) 
  (h : k * 360 + 180 < α ∧ α < k * 360 + 270) : 
  k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135 :=
sorry

end angle_halving_quadrant_l250_250505


namespace initial_fish_count_l250_250684

-- Definitions based on the given conditions
def Fish_given : ℝ := 22.0
def Fish_now : ℝ := 25.0

-- The goal is to prove the initial number of fish Mrs. Sheridan had.
theorem initial_fish_count : (Fish_given + Fish_now) = 47.0 := by
  sorry

end initial_fish_count_l250_250684


namespace woman_worked_days_l250_250047

-- Define variables and conditions
variables (W I : ℕ)

-- Conditions
def total_days : Prop := W + I = 25
def net_earnings : Prop := 20 * W - 5 * I = 450

-- Main theorem statement
theorem woman_worked_days (h1 : total_days W I) (h2 : net_earnings W I) : W = 23 :=
sorry

end woman_worked_days_l250_250047


namespace sum_of_decimals_l250_250198

theorem sum_of_decimals : 5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end sum_of_decimals_l250_250198


namespace distinct_pairs_reciprocal_sum_l250_250839

theorem distinct_pairs_reciprocal_sum : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ (m n : ℕ), ((m, n) ∈ S) ↔ (m > 0 ∧ n > 0 ∧ (1/m + 1/n = 1/5))) ∧ S.card = 3 :=
sorry

end distinct_pairs_reciprocal_sum_l250_250839


namespace circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l250_250485

-- Part (a): Prove the center and radius for the given circle equation: (x-3)^2 + (y+2)^2 = 16
theorem circle_a_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), (x - 3) ^ 2 + (y + 2) ^ 2 = 16 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 3 ∧ b = -2 ∧ R = 4) :=
by {
  sorry
}

-- Part (b): Prove the center and radius for the given circle equation: x^2 + y^2 - 2(x - 3y) - 15 = 0
theorem circle_b_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), x^2 + y^2 - 2 * (x - 3 * y) - 15 = 0 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1 ∧ b = -3 ∧ R = 5) :=
by {
  sorry
}

-- Part (c): Prove the center and radius for the given circle equation: x^2 + y^2 = x + y + 1/2
theorem circle_c_center_radius :
  (∃ (a b : ℚ) (R : ℚ), (∀ (x y : ℚ), x^2 + y^2 = x + y + 1/2 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1/2 ∧ b = 1/2 ∧ R = 1) :=
by {
  sorry
}

end circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l250_250485


namespace shirt_price_correct_l250_250012

noncomputable def sweater_price := 43.885
noncomputable def shirt_price := 36.455
noncomputable def total_cost := 80.34
noncomputable def price_difference := 7.43

theorem shirt_price_correct :
  (shirt_price + sweater_price = total_cost) ∧ (sweater_price - shirt_price = price_difference) →
  shirt_price = 36.455 :=
by {
  intros h,
  sorry
}

end shirt_price_correct_l250_250012


namespace prime_triplets_l250_250621

theorem prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) := by
  sorry

end prime_triplets_l250_250621


namespace solve_for_z_l250_250969

open Complex

theorem solve_for_z (z : ℂ) (h : 2 * z * I = 1 + 3 * I) : 
  z = (3 / 2) - (1 / 2) * I :=
by
  sorry

end solve_for_z_l250_250969


namespace solution_set_l250_250145

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l250_250145


namespace value_of_y_l250_250028

theorem value_of_y :
  ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l250_250028


namespace largest_real_number_condition_l250_250810

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l250_250810


namespace intersection_is_correct_l250_250253

def A : Set ℝ := { x | x * (x - 2) < 0 }
def B : Set ℝ := { x | Real.log x > 0 }

theorem intersection_is_correct : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_is_correct_l250_250253


namespace calculate_F_l250_250818

def f(a : ℝ) : ℝ := a^2 - 5 * a + 6
def F(a b c : ℝ) : ℝ := b^2 + a * c + 1

theorem calculate_F : F 3 (f 3) (f 5) = 19 :=
by
  sorry

end calculate_F_l250_250818


namespace coefficient_x3_expansion_l250_250613

/--
Prove that the coefficient of \(x^{3}\) in the expansion of \(( \frac{x}{\sqrt{y}} - \frac{y}{\sqrt{x}})^{6}\) is \(15\).
-/
theorem coefficient_x3_expansion (x y : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ (x / y.sqrt - y / x.sqrt) ^ 6 = c * x ^ 3) :=
sorry

end coefficient_x3_expansion_l250_250613


namespace y_eq_fraction_x_l250_250034

theorem y_eq_fraction_x (p : ℝ) (x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) :=
sorry

end y_eq_fraction_x_l250_250034


namespace number_of_students_increased_l250_250717

theorem number_of_students_increased
  (original_number_of_students : ℕ) (increase_in_expenses : ℕ) (diminshed_average_expenditure : ℕ)
  (original_expenditure : ℕ) (increase_in_students : ℕ) :
  original_number_of_students = 35 →
  increase_in_expenses = 42 →
  diminshed_average_expenditure = 1 →
  original_expenditure = 420 →
  (35 + increase_in_students) * (12 - 1) - 420 = 42 →
  increase_in_students = 7 :=
by
  intros
  sorry

end number_of_students_increased_l250_250717


namespace largest_real_number_condition_l250_250811

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l250_250811


namespace chests_content_l250_250529

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l250_250529


namespace axis_of_symmetry_of_parabola_l250_250163

-- Definitions (from conditions):
def quadratic_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root_of_quadratic (a b c x : ℝ) : Prop := quadratic_equation a b c x = 0

-- Given conditions
variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)
variable (h_root1 : is_root_of_quadratic a b c 1)
variable (h_root2 : is_root_of_quadratic a b c 5)

-- Problem statement
theorem axis_of_symmetry_of_parabola : (3 : ℝ) = (1 + 5) / 2 :=
by
  -- proof omitted
  sorry

end axis_of_symmetry_of_parabola_l250_250163


namespace dan_gave_marbles_l250_250943

-- Conditions as definitions in Lean 4
def original_marbles : ℕ := 64
def marbles_left : ℕ := 50
def marbles_given : ℕ := original_marbles - marbles_left

-- Theorem statement proving the question == answer given the conditions.
theorem dan_gave_marbles : marbles_given = 14 := by
  sorry

end dan_gave_marbles_l250_250943


namespace range_of_a_l250_250661

theorem range_of_a (a : ℝ) (h1 : 2 * a + 1 < 17) (h2 : 2 * a + 1 > 7) : 3 < a ∧ a < 8 := by
  sorry

end range_of_a_l250_250661


namespace min_value_of_algebraic_sum_l250_250880

theorem min_value_of_algebraic_sum 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + 3 * b = 3) :
  ∃ (min_value : ℝ), min_value = 16 / 3 ∧ (∀ a b, a > 0 → b > 0 → a + 3 * b = 3 → 1 / a + 3 / b ≥ min_value) :=
sorry

end min_value_of_algebraic_sum_l250_250880


namespace sam_annual_income_l250_250670

theorem sam_annual_income
  (q : ℝ) (I : ℝ)
  (h1 : 30000 * 0.01 * q + 15000 * 0.01 * (q + 3) + (I - 45000) * 0.01 * (q + 5) = (q + 0.35) * 0.01 * I) :
  I = 48376 := 
sorry

end sam_annual_income_l250_250670


namespace additional_airplanes_needed_l250_250865

theorem additional_airplanes_needed (total_current_airplanes : ℕ) (airplanes_per_row : ℕ) 
  (h_current_airplanes : total_current_airplanes = 37) 
  (h_airplanes_per_row : airplanes_per_row = 8) : 
  ∃ additional_airplanes : ℕ, additional_airplanes = 3 ∧ 
  ((total_current_airplanes + additional_airplanes) % airplanes_per_row = 0) :=
by
  sorry

end additional_airplanes_needed_l250_250865


namespace sum_of_first_ten_terms_l250_250088

theorem sum_of_first_ten_terms (S : ℕ → ℕ) (h : ∀ n, S n = n^2 - 4 * n + 1) : S 10 = 61 :=
by
  sorry

end sum_of_first_ten_terms_l250_250088


namespace f_neg_two_l250_250832

def f (a b : ℝ) (x : ℝ) :=
  -a * x^5 - x^3 + b * x - 7

theorem f_neg_two (a b : ℝ) (h : f a b 2 = -9) : f a b (-2) = -5 :=
by sorry

end f_neg_two_l250_250832


namespace expenditure_on_house_rent_l250_250607

variable (X : ℝ) -- Let X be Bhanu's total income in rupees

-- Condition 1: Bhanu spends 300 rupees on petrol, which is 30% of his income
def condition_on_petrol : Prop := 0.30 * X = 300

-- Definition of remaining income
def remaining_income : ℝ := X - 300

-- Definition of house rent expenditure: 10% of remaining income
def house_rent : ℝ := 0.10 * remaining_income X

-- Theorem: If the condition on petrol holds, then the house rent expenditure is 70 rupees
theorem expenditure_on_house_rent (h : condition_on_petrol X) : house_rent X = 70 :=
  sorry

end expenditure_on_house_rent_l250_250607


namespace ones_digit_of_8_pow_47_l250_250026

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l250_250026


namespace min_value_point_on_line_l250_250238

theorem min_value_point_on_line (m n : ℝ) (h : m + 2 * n = 1) : 
  2^m + 4^n ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_point_on_line_l250_250238


namespace prime_factor_count_l250_250486

theorem prime_factor_count (n : ℕ) (H : 22 + n + 2 = 29) : n = 5 := 
  sorry

end prime_factor_count_l250_250486


namespace value_of_expression_l250_250728

theorem value_of_expression (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 3) :
  (2 * a - (3 * b - 4 * c)) - ((2 * a - 3 * b) - 4 * c) = 24 := by
  sorry

end value_of_expression_l250_250728


namespace washing_machine_capacity_l250_250191

-- Definitions of the conditions
def total_pounds_per_day : ℕ := 200
def number_of_machines : ℕ := 8

-- Main theorem to prove the question == answer given the conditions
theorem washing_machine_capacity :
  total_pounds_per_day / number_of_machines = 25 :=
by
  sorry

end washing_machine_capacity_l250_250191


namespace find_numbers_l250_250461

theorem find_numbers (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : 1000 * x + y = 7 * x * y) :
  x = 143 ∧ y = 143 :=
by
  sorry

end find_numbers_l250_250461


namespace tan_pi_plus_alpha_l250_250817

noncomputable def given_condition (α : ℝ) : Prop :=
  sin α = -2 / 3 ∧ (π < α ∧ α < 3 * π / 2)

theorem tan_pi_plus_alpha (α : ℝ) (h : given_condition α) : 
  tan (π + α) = 2 * real.sqrt 5 / 5 :=
by
  sorry

end tan_pi_plus_alpha_l250_250817


namespace number_of_tacos_you_ordered_l250_250909

variable {E : ℝ} -- E represents the cost of one enchilada in dollars

-- Conditions
axiom h1 : ∃ t : ℕ, 0.9 * (t : ℝ) + 3 * E = 7.80
axiom h2 : 0.9 * 3 + 5 * E = 12.70

theorem number_of_tacos_you_ordered (E : ℝ) : ∃ t : ℕ, t = 2 := by
  sorry

end number_of_tacos_you_ordered_l250_250909


namespace crayons_total_l250_250345

theorem crayons_total (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 15 → crayons_per_row = 42 → total_crayons = rows * crayons_per_row → total_crayons = 630 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end crayons_total_l250_250345


namespace range_of_m_l250_250659

theorem range_of_m (m : ℝ) (p : Prop) (q : Prop)
  (hp : (2 * m)^2 - 4 ≥ 0 ↔ p)
  (hq : 1 < (Real.sqrt (5 + m)) / (Real.sqrt 5) ∧ (Real.sqrt (5 + m)) / (Real.sqrt 5) < 2 ↔ q)
  (hnq : ¬q = False)
  (hpq : (p ∧ q) = False) :
  0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l250_250659


namespace find_f_neg_2_l250_250830

def f (x : ℝ) : ℝ := sorry -- The actual function f is undefined here.

theorem find_f_neg_2 (h : ∀ x ≠ 0, f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
sorry

end find_f_neg_2_l250_250830


namespace validColoringsCount_l250_250217

-- Define the initial conditions
def isValidColoring (n : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ i ∈ Finset.range (n - 1), 
    (i % 2 = 1 → (color i = 1 ∨ color i = 3)) ∧
    color i ≠ color (i + 1)

noncomputable def countValidColorings : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => 
    match n % 2 with
    | 0 => 2 * 3^(n/2)
    | _ => 4 * 3^((n-1)/2)

-- Main theorem
theorem validColoringsCount (n : ℕ) :
  (∀ color : ℕ → ℕ, isValidColoring n color) →
  (if n % 2 = 0 then countValidColorings n = 4 * 3^((n / 2) - 1) 
     else countValidColorings n = 2 * 3^(n / 2)) :=
by
  sorry

end validColoringsCount_l250_250217


namespace minimum_value_of_A_l250_250648

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end minimum_value_of_A_l250_250648


namespace exists_subseq_sum_div_by_100_l250_250989

theorem exists_subseq_sum_div_by_100 (a : Fin 100 → ℤ) :
  ∃ (s : Finset (Fin 100)), (s.sum (λ i, a i)) % 100 = 0 := 
by 
  sorry

end exists_subseq_sum_div_by_100_l250_250989


namespace solution_set_l250_250144

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l250_250144


namespace find_m_minus_n_l250_250077

noncomputable def m_abs := 4
noncomputable def n_abs := 6

theorem find_m_minus_n (m n : ℝ) (h1 : |m| = m_abs) (h2 : |n| = n_abs) (h3 : |m + n| = m + n) : m - n = -2 ∨ m - n = -10 :=
sorry

end find_m_minus_n_l250_250077


namespace simplify_fraction_l250_250876

-- Given
def num := 54
def denom := 972

-- Factorization condition
def factorization_54 : num = 2 * 3^3 := by 
  sorry

def factorization_972 : denom = 2^2 * 3^5 := by 
  sorry

-- GCD condition
def gcd_num_denom := 54

-- Division condition
def simplified_num := 1
def simplified_denom := 18

-- Statement to prove
theorem simplify_fraction : (num / denom) = (simplified_num / simplified_denom) := by 
  sorry

end simplify_fraction_l250_250876


namespace negation_of_rectangular_parallelepipeds_have_12_edges_l250_250565

-- Define a structure for Rectangular Parallelepiped and the property of having edges
structure RectangularParallelepiped where
  hasEdges : ℕ → Prop

-- Problem statement
theorem negation_of_rectangular_parallelepipeds_have_12_edges :
  (∀ rect_p : RectangularParallelepiped, rect_p.hasEdges 12) →
  ∃ rect_p : RectangularParallelepiped, ¬ rect_p.hasEdges 12 := 
by
  sorry

end negation_of_rectangular_parallelepipeds_have_12_edges_l250_250565


namespace triangle_type_and_area_l250_250888

theorem triangle_type_and_area (x : ℝ) (hpos : 0 < x) (h : 3 * x + 4 * x + 5 * x = 36) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = 54 :=
by {
  sorry
}

end triangle_type_and_area_l250_250888


namespace parallelogram_perimeter_area_sum_l250_250945

theorem parallelogram_perimeter_area_sum 
  (A B C D : Point)
  (hA : A = (1, 3))
  (hB : B = (6, 3))
  (hC : C = (4, 0))
  (hD : D = (-1, 0)) :
  let sideLength (p1 p2 : Point) : ℝ :=
    EuclideanGeometry.dist p1 p2 in
  let p := 2 * (sideLength A B + sideLength A D) in
  let base := sideLength A D in
  let height := 3 in
  let a := base * height in
  p + a = 25 + 2 * Real.sqrt 13 :=
by
  sorry

end parallelogram_perimeter_area_sum_l250_250945


namespace problem_inequality_l250_250155

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_pos : ∀ x : ℝ, x > 0 → f x > 0

axiom f_increasing : ∀ x y : ℝ, x > 0 → y > 0 → x ≤ y → (f x / x) ≤ (f y / y)

theorem problem_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
    3 * (f a + f b + f c) / (a + b + c) + (f a / a + f b / b + f c / c) :=
sorry

end problem_inequality_l250_250155


namespace possible_b_value_l250_250679

theorem possible_b_value (a b : ℤ) (h1 : a = 3^20) (h2 : a ≡ b [ZMOD 10]) : b = 2011 :=
by sorry

end possible_b_value_l250_250679


namespace find_x_l250_250487

theorem find_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (7 * x + 42)) : x = -3 / 2 :=
sorry

end find_x_l250_250487


namespace largest_x_eq_48_div_7_l250_250806

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l250_250806


namespace pradeep_maximum_marks_l250_250319

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.35 * M = 175) :
  M = 500 :=
by
  sorry

end pradeep_maximum_marks_l250_250319


namespace binomial_product_l250_250215

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l250_250215


namespace apples_sold_l250_250570

theorem apples_sold (a1 a2 a3 : ℕ) (h1 : a3 = a2 / 4 + 8) (h2 : a2 = a1 / 4 + 8) (h3 : a3 = 18) : a1 = 128 :=
by
  sorry

end apples_sold_l250_250570


namespace value_of_a_is_3_l250_250248

def symmetric_about_x1 (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| + |x - a| = |2 - x + 1| + |2 - x - a|

theorem value_of_a_is_3 : symmetric_about_x1 3 :=
sorry

end value_of_a_is_3_l250_250248


namespace simplify_fraction_l250_250449

theorem simplify_fraction :
  (6 * x ^ 3 + 13 * x ^ 2 + 15 * x - 25) / (2 * x ^ 3 + 4 * x ^ 2 + 4 * x - 10) =
  (6 * x - 5) / (2 * x - 2) :=
by
  sorry

end simplify_fraction_l250_250449


namespace rectangle_y_value_l250_250456

theorem rectangle_y_value (y : ℝ) (h₁ : (-2, y) ≠ (10, y))
  (h₂ : (-2, -1) ≠ (10, -1))
  (h₃ : 12 * (y + 1) = 108)
  (y_pos : 0 < y) :
  y = 8 :=
by
  sorry

end rectangle_y_value_l250_250456


namespace inequality_proof_l250_250815

theorem inequality_proof 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
  sorry

end inequality_proof_l250_250815


namespace not_or_false_imp_and_false_l250_250258

variable (p q : Prop)

theorem not_or_false_imp_and_false (h : ¬ (p ∨ q) = False) : ¬ (p ∧ q) :=
by
  sorry

end not_or_false_imp_and_false_l250_250258


namespace only_sqrt_three_is_irrational_l250_250753

-- Definitions based on conditions
def zero_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (0 : ℝ) = p / q
def neg_three_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (-3 : ℝ) = p / q
def one_third_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (1/3 : ℝ) = p / q
def sqrt_three_irrational : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ (Real.sqrt 3) = p / q

-- The proof problem statement
theorem only_sqrt_three_is_irrational :
  zero_rational ∧
  neg_three_rational ∧
  one_third_rational ∧
  sqrt_three_irrational :=
by sorry

end only_sqrt_three_is_irrational_l250_250753


namespace three_seventy_five_as_fraction_l250_250442

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l250_250442


namespace min_value_of_f_l250_250833

def f (x : ℝ) (a : ℝ) := - x^3 + a * x^2 - 4

def f_deriv (x : ℝ) (a : ℝ) := - 3 * x^2 + 2 * a * x

theorem min_value_of_f (h : f_deriv (2) a = 0)
  (hm : ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f m a + f_deriv m a ≥ f 0 3 + f_deriv (-1) 3) :
  f 0 3 + f_deriv (-1) 3 = -13 :=
by sorry

end min_value_of_f_l250_250833


namespace tutors_meet_again_l250_250571

theorem tutors_meet_again (tim uma victor xavier: ℕ) (h1: tim = 5) (h2: uma = 6) (h3: victor = 9) (h4: xavier = 8) :
  Nat.lcm (Nat.lcm tim uma) (Nat.lcm victor xavier) = 360 := 
by 
  rw [h1, h2, h3, h4]
  show Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 8) = 360
  sorry

end tutors_meet_again_l250_250571


namespace determine_d_l250_250614

theorem determine_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : d = 1/2 := by
  sorry

end determine_d_l250_250614


namespace problem_statement_l250_250687

theorem problem_statement :
  (1 / 3 * 1 / 6 * P = (1 / 4 * 1 / 8 * 64) + (1 / 5 * 1 / 10 * 100)) → 
  P = 72 :=
by
  sorry

end problem_statement_l250_250687


namespace find_y_given_conditions_l250_250557

def is_value_y (x y : ℕ) : Prop :=
  (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200

theorem find_y_given_conditions : ∃ y : ℕ, ∀ x : ℕ, (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200 → y = 50 :=
by
  sorry

end find_y_given_conditions_l250_250557


namespace remainder_of_x_l250_250993

theorem remainder_of_x (x : ℕ) 
(H1 : 4 + x ≡ 81 [MOD 16])
(H2 : 6 + x ≡ 16 [MOD 36])
(H3 : 8 + x ≡ 36 [MOD 64]) :
  x ≡ 37 [MOD 48] :=
sorry

end remainder_of_x_l250_250993


namespace jacques_suitcase_weight_l250_250519

noncomputable def suitcase_weight_on_return : ℝ := 
  let initial_weight := 12
  let perfume_weight := (5 * 1.2) / 16
  let chocolate_weight := 4 + 1.5 + 3.25
  let soap_weight := (2 * 5) / 16
  let jam_weight := (8 + 6 + 10 + 12) / 16
  let sculpture_weight := 3.5 * 2.20462
  let shirts_weight := (3 * 300 * 0.03527396) / 16
  let cookies_weight := (450 * 0.03527396) / 16
  let wine_weight := (190 * 0.03527396) / 16
  initial_weight + perfume_weight + chocolate_weight + soap_weight + jam_weight + sculpture_weight + shirts_weight + cookies_weight + wine_weight

theorem jacques_suitcase_weight : suitcase_weight_on_return = 35.111288 := 
by 
  -- Calculation to verify that the total is 35.111288
  sorry

end jacques_suitcase_weight_l250_250519


namespace third_bouquet_carnations_l250_250726

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end third_bouquet_carnations_l250_250726


namespace leading_digits_sum_l250_250387

-- Define the conditions
def M : ℕ := (888888888888888888888888888888888888888888888888888888888888888888888888888888) -- define the 400-digit number
-- Assume the function g(r) which finds the leading digit of the r-th root of M

/-- 
  Function g(r) definition:
  It extracts the leading digit of the r-th root of the given number M.
-/
noncomputable def g (r : ℕ) : ℕ := sorry

-- Define the problem statement in Lean 4
theorem leading_digits_sum :
  g 3 + g 4 + g 5 + g 6 + g 7 = 8 :=
sorry

end leading_digits_sum_l250_250387


namespace binomial_product_l250_250213

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l250_250213


namespace roger_cookie_price_l250_250147

noncomputable def price_per_roger_cookie (A_cookies: ℕ) (A_price_per_cookie: ℕ) (A_area_per_cookie: ℕ) (R_cookies: ℕ) (R_area_per_cookie: ℕ): ℕ :=
  by
  let A_total_earnings := A_cookies * A_price_per_cookie
  let R_total_area := A_cookies * A_area_per_cookie
  let price_per_R_cookie := A_total_earnings / R_cookies
  exact price_per_R_cookie
  
theorem roger_cookie_price {A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie : ℕ}
  (h1 : A_cookies = 12)
  (h2 : A_price_per_cookie = 60)
  (h3 : A_area_per_cookie = 12)
  (h4 : R_cookies = 18) -- assumed based on area calculation 144 / 8 (we need this input to match solution context)
  (h5 : R_area_per_cookie = 8) :
  price_per_roger_cookie A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie = 40 :=
  by
  sorry

end roger_cookie_price_l250_250147


namespace quadratic_solution_linear_factor_solution_l250_250153

theorem quadratic_solution (x : ℝ) : (5 * x^2 + 2 * x - 1 = 0) ↔ (x = (-1 + Real.sqrt 6) / 5 ∨ x = (-1 - Real.sqrt 6) / 5) := by
  sorry

theorem linear_factor_solution (x : ℝ) : (x * (x - 3) - 4 * (3 - x) = 0) ↔ (x = 3 ∨ x = -4) := by
  sorry

end quadratic_solution_linear_factor_solution_l250_250153


namespace loss_is_negative_one_point_twenty_seven_percent_l250_250592

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def gov_tax : ℝ := gov_tax_rate * book_price
noncomputable def seller_discount : ℝ := seller_discount_rate * book_price
noncomputable def total_cost : ℝ := book_price + gov_tax + shipping_fee - seller_discount
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def loss_percentage : ℝ := (profit / total_cost) * 100

theorem loss_is_negative_one_point_twenty_seven_percent :
  loss_percentage = -1.27 :=
by
  sorry

end loss_is_negative_one_point_twenty_seven_percent_l250_250592


namespace distinct_real_roots_exists_l250_250351

-- This statement encompasses the conditions and the conclusion
theorem distinct_real_roots_exists (a : ℝ) :
  a ∈ Ioo 0 (1/2) ↔ ∀ x : ℝ, x > 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x^2 + 2 * (a - 1) * x + a^2 = 0) ∧ 
    x1 + a > 0 ∧
    x2 + a > 0 ∧
    x1 + a ≠ 1 ∧
    x2 + a ≠ 1 :=
begin
  sorry
end

end distinct_real_roots_exists_l250_250351


namespace hockey_season_duration_l250_250719

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end hockey_season_duration_l250_250719


namespace gain_percent_l250_250030

theorem gain_percent (CP SP : ℝ) (hCP : CP = 20) (hSP : SP = 35) : 
  (SP - CP) / CP * 100 = 75 :=
by
  rw [hCP, hSP]
  sorry

end gain_percent_l250_250030


namespace sqrt_54_sub_sqrt_6_l250_250062

theorem sqrt_54_sub_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end sqrt_54_sub_sqrt_6_l250_250062


namespace sum_of_altitudes_l250_250007

theorem sum_of_altitudes (x y : ℝ) (h : 12 * x + 5 * y = 60) :
  let a := (if y = 0 then x else 0)
  let b := (if x = 0 then y else 0)
  let c := (60 / (Real.sqrt (12^2 + 5^2)))
  a + b + c = 281 / 13 :=
sorry

end sum_of_altitudes_l250_250007


namespace geometric_series_sum_l250_250769

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l250_250769


namespace rhombus_side_length_l250_250337

noncomputable def side_length_rhombus (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) : ℝ :=
  4

theorem rhombus_side_length (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) (x : ℝ) :
  side_length_rhombus AB BC AC condition1 condition2 condition3 = x ↔ x = 4 := by
  sorry

end rhombus_side_length_l250_250337


namespace average_of_remaining_two_l250_250158

theorem average_of_remaining_two (S S3 : ℚ) (h1 : S / 5 = 6) (h2 : S3 / 3 = 4) : (S - S3) / 2 = 9 :=
by
  sorry

end average_of_remaining_two_l250_250158


namespace largest_real_solution_l250_250795

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l250_250795


namespace events_related_with_99_confidence_l250_250899

theorem events_related_with_99_confidence (K_squared : ℝ) (h : K_squared > 6.635) : 
  events_A_B_related_with_99_confidence :=
sorry

end events_related_with_99_confidence_l250_250899


namespace dexter_total_cards_l250_250067

theorem dexter_total_cards 
  (boxes_basketball : ℕ) 
  (cards_per_basketball_box : ℕ) 
  (boxes_football : ℕ) 
  (cards_per_football_box : ℕ) 
   (h1 : boxes_basketball = 15)
   (h2 : cards_per_basketball_box = 20)
   (h3 : boxes_football = boxes_basketball - 7)
   (h4 : cards_per_football_box = 25) 
   : boxes_basketball * cards_per_basketball_box + boxes_football * cards_per_football_box = 500 := by 
sorry

end dexter_total_cards_l250_250067


namespace largest_x_satisfies_condition_l250_250791

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l250_250791


namespace third_bouquet_carnations_l250_250725

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end third_bouquet_carnations_l250_250725


namespace k_value_tangent_l250_250636

-- Defining the equations
def line (k : ℝ) (x y : ℝ) : Prop := 3 * x + 5 * y + k = 0
def parabola (x y : ℝ) : Prop := y^2 = 24 * x

-- The main theorem stating that k must be 50 for the line to be tangent to the parabola
theorem k_value_tangent (k : ℝ) : (∀ x y : ℝ, line k x y → parabola x y → True) → k = 50 :=
by 
  -- The proof can be constructed based on the discriminant condition provided in the problem
  sorry

end k_value_tangent_l250_250636


namespace leak_empties_tank_in_12_hours_l250_250041

theorem leak_empties_tank_in_12_hours 
  (capacity : ℕ) (inlet_rate : ℕ) (net_emptying_time : ℕ) (leak_rate : ℤ) (leak_emptying_time : ℕ) :
  capacity = 5760 →
  inlet_rate = 4 →
  net_emptying_time = 8 →
  (inlet_rate - leak_rate : ℤ) = (capacity / (net_emptying_time * 60)) →
  leak_emptying_time = (capacity / leak_rate) →
  leak_emptying_time = 12 * 60 / 60 :=
by sorry

end leak_empties_tank_in_12_hours_l250_250041


namespace tank_empty_time_l250_250330

noncomputable def capacity : ℝ := 5760
noncomputable def leak_rate_time : ℝ := 6
noncomputable def inlet_rate_per_minute : ℝ := 4

-- leak rate calculation
noncomputable def leak_rate : ℝ := capacity / leak_rate_time

-- inlet rate calculation in litres per hour
noncomputable def inlet_rate : ℝ := inlet_rate_per_minute * 60

-- net emptying rate calculation
noncomputable def net_empty_rate : ℝ := leak_rate - inlet_rate

-- time to empty the tank calculation
noncomputable def time_to_empty : ℝ := capacity / net_empty_rate

-- The statement to prove
theorem tank_empty_time : time_to_empty = 8 :=
by
  -- Definition step
  have h1 : leak_rate = capacity / leak_rate_time := rfl
  have h2 : inlet_rate = inlet_rate_per_minute * 60 := rfl
  have h3 : net_empty_rate = leak_rate - inlet_rate := rfl
  have h4 : time_to_empty = capacity / net_empty_rate := rfl

  -- Final proof (skipped with sorry)
  sorry

end tank_empty_time_l250_250330


namespace percentage_palm_oil_in_cheese_l250_250404

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end percentage_palm_oil_in_cheese_l250_250404


namespace fraction_water_by_volume_l250_250101

theorem fraction_water_by_volume
  (A W : ℝ) 
  (h1 : A / W = 0.5)
  (h2 : A / (A + W) = 1/7) : 
  W / (A + W) = 2/7 :=
by
  sorry

end fraction_water_by_volume_l250_250101


namespace geom_arith_sequence_l250_250082

theorem geom_arith_sequence (a b c m n : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : m = (a + b) / 2) 
  (h3 : n = (b + c) / 2) : 
  a / m + c / n = 2 := 
by 
  sorry

end geom_arith_sequence_l250_250082


namespace tom_read_in_five_months_l250_250308

def books_in_may : ℕ := 2
def books_in_june : ℕ := 6
def books_in_july : ℕ := 12
def books_in_august : ℕ := 20
def books_in_september : ℕ := 30

theorem tom_read_in_five_months : 
  books_in_may + books_in_june + books_in_july + books_in_august + books_in_september = 70 := by
  sorry

end tom_read_in_five_months_l250_250308


namespace solution_set_unique_line_l250_250139

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l250_250139


namespace youtube_dislikes_l250_250591

theorem youtube_dislikes (likes : ℕ) (initial_dislikes : ℕ) (final_dislikes : ℕ) :
  likes = 3000 →
  initial_dislikes = likes / 2 + 100 →
  final_dislikes = initial_dislikes + 1000 →
  final_dislikes = 2600 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end youtube_dislikes_l250_250591


namespace p_and_q_work_together_l250_250318

-- Given conditions
variable (Wp Wq : ℝ)

-- Condition that p is 50% more efficient than q
def efficiency_relation : Prop := Wp = 1.5 * Wq

-- Condition that p can complete the work in 25 days
def work_completion_by_p : Prop := Wp = 1 / 25

-- To be proved that p and q working together can complete the work in 15 days
theorem p_and_q_work_together (h1 : efficiency_relation Wp Wq)
                              (h2 : work_completion_by_p Wp) :
                              1 / (Wp + (Wp / 1.5)) = 15 :=
by
  sorry

end p_and_q_work_together_l250_250318


namespace find_k_l250_250510

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - 2 * y = -k) (h3 : 2 * x - y = 8) : k = 2 :=
by
  sorry

end find_k_l250_250510


namespace neg_q_true_l250_250651

theorem neg_q_true : (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_q_true_l250_250651


namespace range_of_a_fall_within_D_l250_250512

-- Define the conditions
variable (a : ℝ) (c : ℝ)
axiom A_through : c = 9
axiom D_through : a < 0 ∧ (6, 7) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove the range of a given the conditions
theorem range_of_a : -1/4 < a ∧ a < -1/18 := sorry

-- Define the additional condition for point P
axiom P_through : (2, 8.1) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove that the object can fall within interval D when passing through point P
theorem fall_within_D : a = -9/40 ∧ -1/4 < a ∧ a < -1/18 := sorry

end range_of_a_fall_within_D_l250_250512


namespace binom_multiplication_l250_250212

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l250_250212


namespace part1_part2_l250_250743

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l250_250743


namespace geometric_sequence_property_l250_250104

variable {a_n : ℕ → ℝ}

theorem geometric_sequence_property (h1 : ∀ m n p q : ℕ, m + n = p + q → a_n m * a_n n = a_n p * a_n q)
    (h2 : a_n 4 * a_n 5 * a_n 6 = 27) : a_n 1 * a_n 9 = 9 := by
  sorry

end geometric_sequence_property_l250_250104


namespace intersection_M_N_eq_l250_250503

open Set

theorem intersection_M_N_eq :
  let M := {x : ℝ | x - 2 > 0}
  let N := {y : ℝ | ∃ (x : ℝ), y = Real.sqrt (x^2 + 1)}
  M ∩ N = {x : ℝ | x > 2} :=
by
  sorry

end intersection_M_N_eq_l250_250503


namespace tangent_sum_l250_250937

theorem tangent_sum :
  (Finset.sum (Finset.range 2019) (λ k => Real.tan ((k + 1) * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47))) = -2021 :=
by
  -- proof will be completed here
  sorry

end tangent_sum_l250_250937


namespace max_checkers_on_chessboard_l250_250020

theorem max_checkers_on_chessboard : 
  ∃ (w b : ℕ), (∀ r c : ℕ, r < 8 ∧ c < 8 → w = 2 * b) ∧ (8 * (w + b) = 48) ∧ (w + b) * 8 ≤ 64 :=
by sorry

end max_checkers_on_chessboard_l250_250020


namespace find_base_b_l250_250095

theorem find_base_b (b : ℕ) : ( (2 * b + 5) ^ 2 = 6 * b ^ 2 + 5 * b + 5 ) → b = 9 := 
by 
  sorry  -- Proof is not required as per instruction

end find_base_b_l250_250095


namespace composition_is_rotation_or_translation_l250_250874

open Real

/-- Define the rotation function. -/
noncomputable def rotation (center : Point ℝ) (angle : ℝ) (p : Point ℝ) : Point ℝ :=
sorry -- Implementation of rotation function is skipped

/-- Define the composition of two rotations. -/
noncomputable def composition_of_rotations (A B : Point ℝ) (α β : ℝ) : Point ℝ → Point ℝ :=
rotation B β ∘ rotation A α

/-- The main theorem stating the required properties. -/
theorem composition_is_rotation_or_translation (A B : Point ℝ) (α β : ℝ) :
  A ≠ B →
  (∃ O : Point ℝ, ∃ θ : ℝ, θ = α + β ∧ (θ % 360 ≠ 0 → composition_of_rotations A B α β = rotation O θ) ∧
  (θ % 360 = 0 → ∃ T : Point ℝ → Point ℝ, composition_of_rotations A B α β = T)) :=
sorry -- Proof omitted

end composition_is_rotation_or_translation_l250_250874


namespace probability_not_face_card_l250_250185

-- Definitions based on the conditions
def total_cards : ℕ := 52
def face_cards  : ℕ := 12
def non_face_cards : ℕ := total_cards - face_cards

-- Statement of the theorem
theorem probability_not_face_card : (non_face_cards : ℚ) / (total_cards : ℚ) = 10 / 13 := by
  sorry

end probability_not_face_card_l250_250185


namespace inclination_line_eq_l250_250829

theorem inclination_line_eq (l : ℝ → ℝ) (h1 : ∃ x, l x = 2 ∧ ∃ y, l y = 2) (h2 : ∃ θ, θ = 135) :
  ∃ a b c, a = 1 ∧ b = 1 ∧ c = -4 ∧ ∀ x y, y = l x → a * x + b * y + c = 0 :=
by 
  sorry

end inclination_line_eq_l250_250829


namespace f_eq_2x_pow_5_l250_250978

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_eq_2x_pow_5 (x : ℝ) : f x = (2*x)^5 :=
by
  sorry

end f_eq_2x_pow_5_l250_250978


namespace soccer_league_games_l250_250736

open Nat

theorem soccer_league_games (n : ℕ) (h : n = 10) : (nat.choose n 2) = 45 := by
  rw h
  simp [nat.choose, factorial]
  sorry

end soccer_league_games_l250_250736


namespace incenter_and_circumcenter_parallel_l250_250671

open EuclideanGeometry

-- Definitions based on the given problem
variables {A B C D E F I1 I2 O1 O2 : Point}

-- Proof statement based on the given conditions
theorem incenter_and_circumcenter_parallel
  (h_acute: Triangle ABC)
  (h_D: foot A B C D)
  (h_E: foot B C A E)
  (h_F: foot C A B F)
  (h_I1: incenter I1 A E F)
  (h_I2: incenter I2 B D F)
  (h_O1: circumcenter O1 A C I1)
  (h_O2: circumcenter O2 B C I2):
  parallel (line_through I1 I2) (line_through O1 O2) :=
sorry

end incenter_and_circumcenter_parallel_l250_250671


namespace no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l250_250164

theorem no_real_solution_x_squared_minus_2x_plus_3_eq_zero :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≠ 0 :=
by
  sorry

end no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l250_250164


namespace find_a3_plus_a5_l250_250957

-- Define an arithmetic-geometric sequence
def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, 0 < r ∧ ∃ b : ℝ, a n = b * r ^ n

-- Define the given condition
def given_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

-- Define the target theorem statement
theorem find_a3_plus_a5 (a : ℕ → ℝ) 
  (pos_sequence : is_arithmetic_geometric a) 
  (cond : given_condition a) : 
  a 3 + a 5 = 5 :=
sorry

end find_a3_plus_a5_l250_250957


namespace infinite_nested_radical_l250_250900

theorem infinite_nested_radical : ∀ (x : ℝ), (x > 0) → (x = Real.sqrt (12 + x)) → x = 4 :=
by
  intro x
  intro hx_pos
  intro hx_eq
  sorry

end infinite_nested_radical_l250_250900


namespace ratio_of_smaller_to_trapezoid_l250_250056

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

def ratio_of_areas : ℝ :=
  let side_large := 10
  let side_small := 5
  let area_large := area_equilateral_triangle side_large
  let area_small := area_equilateral_triangle side_small
  let area_trapezoid := area_large - area_small
  area_small / area_trapezoid

theorem ratio_of_smaller_to_trapezoid :
  ratio_of_areas = 1 / 3 :=
sorry

end ratio_of_smaller_to_trapezoid_l250_250056


namespace ducks_at_Lake_Michigan_l250_250551

variable (D : ℕ)

def ducks_condition := 2 * D + 6 = 206

theorem ducks_at_Lake_Michigan (h : ducks_condition D) : D = 100 :=
by
  sorry

end ducks_at_Lake_Michigan_l250_250551


namespace solution_set_unique_line_l250_250141

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l250_250141


namespace joy_valid_rod_count_l250_250382

theorem joy_valid_rod_count : 
  let l := [4, 12, 21]
  let qs := [1, 2, 3, 5, 13, 20, 22, 40].filter (fun x => x != 4 ∧ x != 12 ∧ x != 21)
  (∀ d ∈ qs, 4 + 12 + 21 > d ∧ 4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) → 
  ∃ n, n = 28 :=
by sorry

end joy_valid_rod_count_l250_250382


namespace christian_age_in_eight_years_l250_250941

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years_l250_250941


namespace determine_n_from_average_l250_250453

-- Definitions derived from conditions
def total_cards (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_of_values (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
def average_value (n : ℕ) : ℚ := sum_of_values n / total_cards n

-- Main statement for proving equivalence
theorem determine_n_from_average :
  (∃ n : ℕ, average_value n = 2023) ↔ (n = 3034) :=
by
  sorry

end determine_n_from_average_l250_250453


namespace triangle_area_difference_l250_250674

-- Definitions based on given lengths and right angles.
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Note: Right angles are implicitly used in the area calculations and do not need to be represented directly in Lean.
-- Define areas for triangles involved.
def area_FGH : ℝ := 0.5 * FG * GH
def area_GHI : ℝ := 0.5 * GH * HI
def area_FHI : ℝ := 0.5 * FG * HI

-- Define areas of the triangles FGJ and HJI using variables.
variable (x y z : ℝ)
axiom area_FGJ : x = area_FHI - z
axiom area_HJI : y = area_GHI - z

-- The main proof statement involving the difference.
theorem triangle_area_difference : (x - y) = 14 := by
  sorry

end triangle_area_difference_l250_250674


namespace sallys_dad_nickels_l250_250697

theorem sallys_dad_nickels :
  ∀ (initial_nickels mother's_nickels total_nickels nickels_from_dad : ℕ), 
    initial_nickels = 7 → 
    mother's_nickels = 2 →
    total_nickels = 18 →
    total_nickels = initial_nickels + mother's_nickels + nickels_from_dad →
    nickels_from_dad = 9 :=
by
  intros initial_nickels mother's_nickels total_nickels nickels_from_dad
  intros h1 h2 h3 h4
  sorry

end sallys_dad_nickels_l250_250697


namespace totalNominalIncomeIsCorrect_l250_250284

def nominalIncomeForMonth (principal rate divisor months : ℝ) : ℝ :=
  principal * ((1 + rate / divisor) ^ months - 1)

def totalNominalIncomeForSixMonths : ℝ :=
  nominalIncomeForMonth 8700 0.06 12 6 +
  nominalIncomeForMonth 8700 0.06 12 5 +
  nominalIncomeForMonth 8700 0.06 12 4 +
  nominalIncomeForMonth 8700 0.06 12 3 +
  nominalIncomeForMonth 8700 0.06 12 2 +
  nominalIncomeForMonth 8700 0.06 12 1

theorem totalNominalIncomeIsCorrect : totalNominalIncomeForSixMonths = 921.15 := by
  sorry

end totalNominalIncomeIsCorrect_l250_250284


namespace equation_of_hyperbola_l250_250354

variable (a b c : ℝ)
variable (x y : ℝ)

theorem equation_of_hyperbola :
  (0 < a) ∧ (0 < b) ∧ (c / a = Real.sqrt 3) ∧ (a^2 / c = 1) ∧ (c = 3) ∧ (b = Real.sqrt 6)
  → (x^2 / 3 - y^2 / 6 = 1) :=
by
  sorry

end equation_of_hyperbola_l250_250354


namespace soccer_player_positions_exist_l250_250468

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l250_250468


namespace least_amount_of_money_l250_250372

variable (money : String → ℝ)
variable (Bo Coe Flo Jo Moe Zoe : String)

theorem least_amount_of_money :
  (money Bo ≠ money Coe) ∧ (money Bo ≠ money Flo) ∧ (money Bo ≠ money Jo) ∧ (money Bo ≠ money Moe) ∧ (money Bo ≠ money Zoe) ∧ 
  (money Coe ≠ money Flo) ∧ (money Coe ≠ money Jo) ∧ (money Coe ≠ money Moe) ∧ (money Coe ≠ money Zoe) ∧ 
  (money Flo ≠ money Jo) ∧ (money Flo ≠ money Moe) ∧ (money Flo ≠ money Zoe) ∧ 
  (money Jo ≠ money Moe) ∧ (money Jo ≠ money Zoe) ∧ 
  (money Moe ≠ money Zoe) ∧ 
  (money Flo > money Jo) ∧ (money Flo > money Bo) ∧
  (money Bo > money Moe) ∧ (money Coe > money Moe) ∧ 
  (money Jo > money Moe) ∧ (money Jo < money Bo) ∧ 
  (money Zoe > money Jo) ∧ (money Zoe < money Coe) →
  money Moe < money Bo ∧ money Moe < money Coe ∧ money Moe < money Flo ∧ money Moe < money Jo ∧ money Moe < money Zoe := 
sorry

end least_amount_of_money_l250_250372


namespace mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l250_250118

theorem mrs_mcpherson_contributes (rent : ℕ) (percentage : ℕ) (mrs_mcp_contribution : ℕ) : 
  mrs_mcp_contribution = (percentage * rent) / 100 := by
  sorry

theorem mr_mcpherson_raises (rent : ℕ) (mrs_mcp_contribution : ℕ) : 
  mr_mcp_contribution = rent - mrs_mcp_contribution := by
  sorry

theorem mr_mcpherson_complete_rent : 
  let rent := 1200
  let percentage := 30
  let mrs_mcp_contribution := (percentage * rent) / 100
  let mr_mcp_contribution := rent - mrs_mcp_contribution
  mr_mcp_contribution = 840 := by
  have mrs_contribution : mrs_mcp_contribution = (30 * 1200) / 100 := by
    exact mrs_mcpherson_contributes 1200 30 ((30 * 1200) / 100)
  have mr_contribution : mr_mcp_contribution = 1200 - ((30 * 1200) / 100) := by
    exact mr_mcpherson_raises 1200 ((30 * 1200) / 100)
  show 1200 - 360 = 840 from by
    rw [mrs_contribution, mr_contribution]
    rfl

end mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l250_250118


namespace largest_x_eq_48_div_7_l250_250804

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l250_250804


namespace net_profit_loan_payments_dividends_per_share_director_dividends_l250_250059

theorem net_profit (revenue expenses : ℕ) (tax_rate : ℚ) 
  (h_rev : revenue = 2500000)
  (h_exp : expenses = 1576250)
  (h_tax : tax_rate = 0.2) :
  ((revenue - expenses) - (revenue - expenses) * tax_rate).toNat = 739000 := by
  sorry

theorem loan_payments (monthly_payment : ℕ) 
  (h_monthly : monthly_payment = 25000) :
  (monthly_payment * 12) = 300000 := by
  sorry

theorem dividends_per_share (net_profit loan_payments : ℕ) (total_shares : ℕ)
  (h_net_profit : net_profit = 739000)
  (h_loan_payments : loan_payments = 300000)
  (h_shares : total_shares = 1600) :
  ((net_profit - loan_payments) / total_shares) = 274 := by
  sorry

theorem director_dividends (dividend_per_share : ℕ) (share_percentage : ℚ) (total_shares : ℕ)
  (h_dividend_per_share : dividend_per_share = 274)
  (h_percentage : share_percentage = 0.35)
  (h_shares : total_shares = 1600) :
  (dividend_per_share * share_percentage * total_shares).toNat = 153440 := by
  sorry

end net_profit_loan_payments_dividends_per_share_director_dividends_l250_250059


namespace correct_answer_l250_250089

def M : Set ℤ := {x | |x| < 5}

theorem correct_answer : {0} ⊆ M := by
  sorry

end correct_answer_l250_250089


namespace number_of_students_passed_l250_250334

theorem number_of_students_passed (total_students : ℕ) (failure_frequency : ℝ) (h1 : total_students = 1000) (h2 : failure_frequency = 0.4) : 
  (total_students - (total_students * failure_frequency)) = 600 :=
by
  sorry

end number_of_students_passed_l250_250334


namespace initial_people_on_train_l250_250048

theorem initial_people_on_train {x y z u v w : ℤ} 
  (h1 : y = 29) (h2 : z = 17) (h3 : u = 27) (h4 : v = 35) (h5 : w = 116) :
  x - (y - z) + (v - u) = w → x = 120 := 
by sorry

end initial_people_on_train_l250_250048


namespace number_of_blocks_l250_250934

theorem number_of_blocks (children_per_block : ℕ) (total_children : ℕ) (h1: children_per_block = 6) (h2: total_children = 54) : (total_children / children_per_block) = 9 :=
by {
  sorry
}

end number_of_blocks_l250_250934


namespace least_number_of_tiles_l250_250044

/-- A room of 544 cm long and 374 cm broad is to be paved with square tiles. 
    Prove that the least number of square tiles required to cover the floor is 176. -/
theorem least_number_of_tiles (length breadth : ℕ) (h1 : length = 544) (h2 : breadth = 374) :
  let gcd_length_breadth := Nat.gcd length breadth
  let num_tiles_length := length / gcd_length_breadth
  let num_tiles_breadth := breadth / gcd_length_breadth
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end least_number_of_tiles_l250_250044


namespace min_value_sum_l250_250417

def non_neg_int := {n : ℕ // 0 ≤ n}

theorem min_value_sum (a b c d : non_neg_int)
  (h : a.val * b.val + b.val * c.val + c.val * d.val + d.val * a.val = 707) :
  a.val + b.val + c.val + d.val ≥ 108 :=
begin
  -- The proof would go here, but it is omitted as per instructions.
  sorry
end

end min_value_sum_l250_250417


namespace job_completion_l250_250437

theorem job_completion (A_rate D_rate : ℝ) (h₁ : A_rate = 1 / 12) (h₂ : A_rate + D_rate = 1 / 4) : D_rate = 1 / 6 := 
by 
  sorry

end job_completion_l250_250437


namespace petya_friends_count_l250_250986

-- Define the number of classmates
def total_classmates : ℕ := 28

-- Each classmate has a unique number of friends from 0 to 27
def unique_friends (n : ℕ) : Prop :=
  n ≥ 0 ∧ n < total_classmates

-- We state the problem where Petya's number of friends is to be proven as 14
theorem petya_friends_count (friends : ℕ) (h : unique_friends friends) : friends = 14 :=
sorry

end petya_friends_count_l250_250986


namespace distinct_numbers_in_list_l250_250484

def count_distinct_floors (l : List ℕ) : ℕ :=
  l.eraseDups.length

def generate_list : List ℕ :=
  List.map (λ n => Nat.floor ((n * n : ℚ) / 2000)) (List.range' 1 2000)

theorem distinct_numbers_in_list : count_distinct_floors generate_list = 1501 :=
by
  sorry

end distinct_numbers_in_list_l250_250484


namespace sum_difference_20_l250_250317

def sum_of_even_integers (n : ℕ) : ℕ := (n / 2) * (2 + 2 * (n - 1))

def sum_of_odd_integers (n : ℕ) : ℕ := (n / 2) * (1 + 2 * (n - 1))

theorem sum_difference_20 : sum_of_even_integers (20) - sum_of_odd_integers (20) = 20 := by
  sorry

end sum_difference_20_l250_250317


namespace simplify_abs_neg_pow_sub_l250_250549

theorem simplify_abs_neg_pow_sub (a b : ℤ) (h : a = 4) (h' : b = 6) : 
  (|-(a ^ 2) - b| = 22) := 
by
  sorry

end simplify_abs_neg_pow_sub_l250_250549


namespace pow_mod_eq_l250_250173

theorem pow_mod_eq : (6 ^ 2040) % 50 = 26 := by
  sorry

end pow_mod_eq_l250_250173


namespace compute_Q3_Qneg3_l250_250678

noncomputable def Q (x : ℝ) (a b c m : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + m

theorem compute_Q3_Qneg3 (a b c m : ℝ)
  (h1 : Q 1 a b c m = 3 * m)
  (h2 : Q (-1) a b c m = 4 * m)
  (h3 : Q 0 a b c m = m) :
  Q 3 a b c m + Q (-3) a b c m = 47 * m :=
by
  sorry

end compute_Q3_Qneg3_l250_250678


namespace joe_used_paint_total_l250_250855

theorem joe_used_paint_total :
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  total_first_airport + total_second_airport = 415 :=
by
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  show total_first_airport + total_second_airport = 415
  sorry

end joe_used_paint_total_l250_250855


namespace algebra_expression_bound_l250_250667

theorem algebra_expression_bound (x y m : ℝ) 
  (h1 : x + y + m = 6) 
  (h2 : 3 * x - y + m = 4) : 
  (-2 * x * y + 1) ≤ 3 / 2 := 
by 
  sorry

end algebra_expression_bound_l250_250667


namespace equation_has_seven_real_solutions_l250_250087

def f (x : ℝ) : ℝ := abs (x^2 - 1) - 1

theorem equation_has_seven_real_solutions (b c : ℝ) : 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) ↔ 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
  x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
  x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
  x₆ ≠ x₇ ∧
  f x₁ ^ 2 - b * f x₁ + c = 0 ∧ f x₂ ^ 2 - b * f x₂ + c = 0 ∧
  f x₃ ^ 2 - b * f x₃ + c = 0 ∧ f x₄ ^ 2 - b * f x₄ + c = 0 ∧
  f x₅ ^ 2 - b * f x₅ + c = 0 ∧ f x₆ ^ 2 - b * f x₆ + c = 0 ∧
  f x₇ ^ 2 - b * f x₇ + c = 0 :=
sorry

end equation_has_seven_real_solutions_l250_250087


namespace trigonometric_simplification_l250_250581

noncomputable def tan : ℝ → ℝ := λ x => Real.sin x / Real.cos x
noncomputable def simp_expr : ℝ :=
  (tan (96 * Real.pi / 180) - tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))
  /
  (1 + tan (96 * Real.pi / 180) * tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))

theorem trigonometric_simplification : simp_expr = Real.sqrt 3 / 3 :=
by
  sorry

end trigonometric_simplification_l250_250581


namespace frac_sum_eq_one_l250_250955

variable {x y : ℝ}

theorem frac_sum_eq_one (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x) + (1 / y) = 1 :=
by sorry

end frac_sum_eq_one_l250_250955


namespace remainder_23_pow_2003_mod_7_l250_250898

theorem remainder_23_pow_2003_mod_7 : 23 ^ 2003 % 7 = 4 :=
by sorry

end remainder_23_pow_2003_mod_7_l250_250898


namespace correct_average_mark_l250_250037

theorem correct_average_mark (
  num_students : ℕ := 50)
  (incorrect_avg : ℚ := 85.4)
  (wrong_mark_A : ℚ := 73.6) (correct_mark_A : ℚ := 63.5)
  (wrong_mark_B : ℚ := 92.4) (correct_mark_B : ℚ := 96.7)
  (wrong_mark_C : ℚ := 55.3) (correct_mark_C : ℚ := 51.8) :
  (incorrect_avg*num_students + 
   (correct_mark_A - wrong_mark_A) + 
   (correct_mark_B - wrong_mark_B) + 
   (correct_mark_C - wrong_mark_C)) / 
   num_students = 85.214 :=
sorry

end correct_average_mark_l250_250037


namespace length_sawed_off_l250_250918

-- Define the lengths as constants
def original_length : ℝ := 8.9
def final_length : ℝ := 6.6

-- State the property to be proven
theorem length_sawed_off : original_length - final_length = 2.3 := by
  sorry

end length_sawed_off_l250_250918


namespace intersection_P_Q_l250_250388

def P (x : ℝ) : Prop := x + 2 ≥ x^2

def Q (x : ℕ) : Prop := x ≤ 3

theorem intersection_P_Q :
  {x : ℕ | P x} ∩ {x : ℕ | Q x} = {0, 1, 2} :=
by
  sorry

end intersection_P_Q_l250_250388


namespace factor_expression_l250_250427

theorem factor_expression (a b c d : ℝ) : 
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 
        = ((a - b) * (b - c) * (c - d) * (d - a)) * (a + b + c + d) := 
by
  sorry

end factor_expression_l250_250427


namespace solution_set_unique_line_l250_250138

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l250_250138


namespace ratio_of_areas_l250_250055

theorem ratio_of_areas 
  (s1 s2 : ℝ)
  (A_large A_small A_trapezoid : ℝ)
  (h1 : s1 = 10)
  (h2 : s2 = 5)
  (h3 : A_large = (sqrt 3 / 4) * s1^2)
  (h4 : A_small = (sqrt 3 / 4) * s2^2)
  (h5 : A_trapezoid = A_large - A_small) :
  (A_small / A_trapezoid = 1 / 3) :=
sorry

end ratio_of_areas_l250_250055


namespace min_sum_distinct_positive_integers_l250_250954

theorem min_sum_distinct_positive_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (1 / a + 1 / b = k1 * (1 / c)) ∧ (1 / a + 1 / c = k2 * (1 / b)) ∧ (1 / b + 1 / c = k3 * (1 / a))) :
  a + b + c ≥ 11 :=
sorry

end min_sum_distinct_positive_integers_l250_250954


namespace soccer_players_positions_l250_250471

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l250_250471


namespace total_nominal_income_l250_250283

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l250_250283


namespace convert_decimal_to_fraction_l250_250439

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l250_250439


namespace largest_real_number_condition_l250_250807

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l250_250807


namespace power_of_fraction_l250_250760

theorem power_of_fraction : ((1/3)^5 = (1/243)) :=
by
  sorry

end power_of_fraction_l250_250760


namespace finish_work_in_time_l250_250593

noncomputable def work_in_days_A (DA : ℕ) := DA
noncomputable def work_in_days_B (DA : ℕ) := DA / 2
noncomputable def combined_work_rate (DA : ℕ) : ℚ := 1 / work_in_days_A DA + 2 / work_in_days_A DA

theorem finish_work_in_time (DA : ℕ) (h_combined_rate : combined_work_rate DA = 0.25) : DA = 12 :=
sorry

end finish_work_in_time_l250_250593


namespace minimum_value_expression_l250_250649

theorem minimum_value_expression 
  (x y z : ℝ) 
  (h1 : x ≥ 3) (h2 : y ≥ 3) (h3 : z ≥ 3) :
  let A := (x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3) in
  let B := x * y + y * z + z * x in
  A / B ≥ 1 :=
sorry

end minimum_value_expression_l250_250649


namespace ones_digit_of_8_pow_47_l250_250021

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l250_250021


namespace possible_values_of_P_l250_250539

-- Definition of the conditions
variables (x y : ℕ) (h1 : x < y) (h2 : (x > 0)) (h3 : (y > 0))

-- Definition of P
def P : ℤ := (x^3 - y) / (1 + x * y)

-- Theorem statement
theorem possible_values_of_P : (P = 0) ∨ (P ≥ 2) :=
sorry

end possible_values_of_P_l250_250539


namespace sharks_problem_l250_250866

variable (F : ℝ)
variable (S : ℝ := 0.25 * (F + 3 * F))
variable (total_sharks : ℝ := 15)

theorem sharks_problem : 
  (0.25 * (F + 3 * F) = 15) ↔ (F = 15) :=
by 
  sorry

end sharks_problem_l250_250866


namespace star_3_2_l250_250009

-- Definition of the operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- The proof problem
theorem star_3_2 : star 3 2 = 22 :=
by
  sorry

end star_3_2_l250_250009


namespace soccer_players_arrangement_l250_250469

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l250_250469


namespace cost_of_each_ball_number_of_plans_highest_profit_l250_250186

-- Part 1: Cost of each ball
theorem cost_of_each_ball (x y : ℝ) 
  (h1 : 10 * x + 5 * y = 100) 
  (h2 : 5 * x + 3 * y = 55) : 
  x = 5 ∧ y = 10 :=
sorry

-- Part 2: Number of purchasing plans
theorem number_of_plans 
  (x y m : ℝ) 
  (h_costA : x = 5) 
  (h_costB : y = 10)
  (h_budget : 1000 = x * (200 - 2 * m) + y * m) 
  (h_quantity_A : 200 - 2 * m ≥ 6 * m) 
  (h_minimum_m : 23 ≤ m) :
  ∃ l, l = {23, 24, 25}.card ∧ (∀ k ∈ l, 23 ≤ k ∧ k ≤ 25) ∧ set.card l = 3 :=
sorry

-- Part 3: Maximum profit
theorem highest_profit 
  (m : ℝ)
  (h_costB : 10 = 10)
  (h_plans : m ∈ {23, 24, 25})
  (h_max_profit : ∀ m₁ ∈ {23, 24, 25}, 
    let profit := 3 * (200 - 2 * m) + 4 * m in 
    profit ≤ (3 * (200 - 2 * 23) + 4 * 23) :=
  ∀ m₁ ∈ {23, 24, 25}, 
    3 * (200 - 2 * m) + 4 * m ≤ 554 :=
sorry

end cost_of_each_ball_number_of_plans_highest_profit_l250_250186


namespace cost_of_each_ticket_l250_250638

theorem cost_of_each_ticket (x : ℝ) : 
  500 * x * 0.70 = 4 * 2625 → x = 30 :=
by 
  sorry

end cost_of_each_ticket_l250_250638


namespace problem1_problem2_l250_250276

-- problem (1): Prove that if a = 1 and (p ∨ q) is true, then the range of x is 1 < x < 3
def p (a x : ℝ) : Prop := x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : a = 1) (h₂ : p a x ∨ q x) : 
    1 < x ∧ x < 3 :=
sorry

-- problem (2): Prove that if p is a necessary but not sufficient condition for q,
-- then the range of a is 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end problem1_problem2_l250_250276


namespace geometric_sequence_problem_l250_250239

variable {a : ℕ → ℝ}
variable (r a1 : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_geom : ∀ n, a (n + 1) = a 1 * r ^ n)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025)

theorem geometric_sequence_problem :
  a 3 + a 5 = 45 :=
by
  sorry

end geometric_sequence_problem_l250_250239


namespace game_is_unfair_l250_250029

def pencil_game_unfair : Prop :=
∀ (take1 take2 : ℕ → ℕ),
  take1 1 = 1 ∨ take1 1 = 2 →
  take2 2 = 1 ∨ take2 2 = 2 →
  ∀ n : ℕ,
    n = 5 → (∃ first_move : ℕ, (take1 first_move = 2) ∧ (take2 (take1 first_move) = 1 ∨ take2 (take1 first_move) = 2) ∧ (take1 (take2 (n - take1 first_move)) = 1 ∨ take1 (take2 (n - take1 first_move)) = 2) ∧
    ∀ second_move : ℕ, (second_move = n - first_move - take2 (n - take1 first_move)) → 
    n - first_move - take2 (n - take1 first_move) = 1 ∨ n - first_move - take2 (n - take1 first_move) = 2)

theorem game_is_unfair : pencil_game_unfair := 
sorry

end game_is_unfair_l250_250029


namespace simplify_expression_l250_250698

theorem simplify_expression (x : ℝ) : (2 * x)^3 + (3 * x) * (x^2) = 11 * x^3 := 
  sorry

end simplify_expression_l250_250698


namespace minimum_value_of_f_l250_250227

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 - (Real.log x) / x

theorem minimum_value_of_f : (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
by
  sorry

end minimum_value_of_f_l250_250227


namespace max_playground_area_l250_250980

/-- Mara is setting up a fence around a rectangular playground with given constraints.
    We aim to prove that the maximum area the fence can enclose is 10000 square feet. --/
theorem max_playground_area (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 400) 
  (h2 : l ≥ 100) 
  (h3 : w ≥ 50) : 
  l * w ≤ 10000 :=
sorry

end max_playground_area_l250_250980


namespace derivative_of_f_l250_250298

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -x * Real.sin x := by
  sorry

end derivative_of_f_l250_250298


namespace option_B_more_cost_effective_l250_250511

def cost_option_A (x : ℕ) : ℕ := 60 + 18 * x
def cost_option_B (x : ℕ) : ℕ := 150 + 15 * x
def x : ℕ := 40

theorem option_B_more_cost_effective : cost_option_B x < cost_option_A x := by
  -- Placeholder for the proof steps
  sorry

end option_B_more_cost_effective_l250_250511


namespace least_product_xy_l250_250492

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l250_250492


namespace axis_of_symmetry_l250_250959

-- Define points and the parabola equation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5
def B := Point.mk 4 5

def parabola (b c : ℝ) (p : Point) : Prop :=
  p.y = 2 * p.x^2 + b * p.x + c

theorem axis_of_symmetry (b c : ℝ) (hA : parabola b c A) (hB : parabola b c B) : ∃ x_axis : ℝ, x_axis = 3 :=
by
  -- Proof to be provided
  sorry

end axis_of_symmetry_l250_250959


namespace g_increasing_on_minus_infty_one_l250_250086

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 + (2 * x) / (1 - x)

theorem g_increasing_on_minus_infty_one : (∀ x y : ℝ, x < y → x < 1 → y ≤ 1 → g x < g y) :=
sorry

end g_increasing_on_minus_infty_one_l250_250086


namespace cube_surface_area_l250_250706

-- Definitions based on conditions from the problem
def edge_length : ℕ := 7
def number_of_faces : ℕ := 6

-- Definition of the problem converted to a theorem in Lean 4
theorem cube_surface_area (edge_length : ℕ) (number_of_faces : ℕ) : 
  number_of_faces * (edge_length * edge_length) = 294 :=
by
  -- Proof steps are omitted, so we put sorry to indicate that the proof is required.
  sorry

end cube_surface_area_l250_250706


namespace problem_statement_l250_250896

theorem problem_statement : (29.7 + 83.45) - 0.3 = 112.85 := sorry

end problem_statement_l250_250896


namespace max_integer_a_for_real_roots_l250_250645

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l250_250645


namespace composite_19_8n_plus_17_l250_250987

theorem composite_19_8n_plus_17 (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
by 
  sorry

end composite_19_8n_plus_17_l250_250987


namespace inequality_proof_l250_250491

variable {a b : ℝ}

theorem inequality_proof (h : a > b) : 2 - a < 2 - b :=
by
  sorry

end inequality_proof_l250_250491


namespace Ken_and_Kendra_fish_count_l250_250113

def Ken_and_Kendra_bring_home (kendra_fish_caught : ℕ) (ken_ratio : ℕ) (ken_releases : ℕ) : ℕ :=
  let ken_fish_caught := ken_ratio * kendra_fish_caught
  let ken_fish_brought_home := ken_fish_caught - ken_releases
  ken_fish_brought_home + kendra_fish_caught

theorem Ken_and_Kendra_fish_count :
  let kendra_fish_caught := 30 in
  let ken_ratio := 2 in
  let ken_releases := 3 in
  Ken_and_Kendra_bring_home kendra_fish_caught ken_ratio ken_releases = 87 :=
by
  sorry

end Ken_and_Kendra_fish_count_l250_250113


namespace flight_relation_not_preserved_l250_250263

noncomputable def swap_city_flights (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) : Prop := sorry

theorem flight_relation_not_preserved (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) (M N : ℕ) (hM : M ∈ cities) (hN : N ∈ cities) : 
  ¬ swap_city_flights cities flights :=
sorry

end flight_relation_not_preserved_l250_250263


namespace f_6_plus_f_neg3_l250_250709

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- f is increasing in the interval [3,6]
def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) := a ≤ b → ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the given conditions
axiom h1 : is_odd_function f
axiom h2 : is_increasing_interval f 3 6
axiom h3 : f 6 = 8
axiom h4 : f 3 = -1

-- The statement to be proved
theorem f_6_plus_f_neg3 : f 6 + f (-3) = 9 :=
by
  sorry

end f_6_plus_f_neg3_l250_250709


namespace inequality_1_minimum_value_l250_250406

-- Definition for part (1)
theorem inequality_1 (a b m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (a^2 / m + b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

-- Definition for part (2)
theorem minimum_value (x : ℝ) (hx : 0 < x) (hx' : x < 1) : 
  (∃ (y : ℝ), y = (1 / x + 4 / (1 - x)) ∧ y = 9) :=
sorry

end inequality_1_minimum_value_l250_250406


namespace syllogism_correct_l250_250947

-- Define that natural numbers are integers
axiom nat_is_int : ∀ (n : ℕ), ∃ (m : ℤ), m = n

-- Define that 4 is a natural number
axiom four_is_nat : ∃ (n : ℕ), n = 4

-- The syllogism's conclusion: 4 is an integer
theorem syllogism_correct : ∃ (m : ℤ), m = 4 :=
by
  have h1 := nat_is_int 4
  have h2 := four_is_nat
  exact h1

end syllogism_correct_l250_250947


namespace system_of_equations_xy_l250_250964

theorem system_of_equations_xy (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = 5) :
  x - y = 2 := sorry

end system_of_equations_xy_l250_250964


namespace discount_percentage_l250_250286

theorem discount_percentage (original_price sale_price : ℝ) (h₁ : original_price = 128) (h₂ : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 :=
by
  sorry

end discount_percentage_l250_250286


namespace find_a_l250_250244

variable {x a : ℝ}

def A (x : ℝ) : Prop := x ≤ -1 ∨ x > 2
def B (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem find_a (hA : ∀ x, (x + 1) / (x - 2) ≥ 0 ↔ A x)
                (hB : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a > 0 ↔ B x a)
                (hSub : ∀ x, A x → B x a) :
  -1 < a ∧ a ≤ 1 :=
sorry

end find_a_l250_250244


namespace geometric_series_sum_l250_250768

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l250_250768


namespace min_value_of_quadratic_l250_250415

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end min_value_of_quadratic_l250_250415


namespace number_less_than_value_l250_250331

-- Definition for the conditions
def exceeds_condition (x y : ℕ) : Prop := x - 18 = 3 * (y - x)
def specific_value (x : ℕ) : Prop := x = 69

-- Statement of the theorem
theorem number_less_than_value : ∃ y : ℕ, (exceeds_condition 69 y) ∧ (specific_value 69) → y = 86 :=
by
  -- To be proved
  sorry

end number_less_than_value_l250_250331


namespace range_of_b_for_increasing_f_l250_250098

noncomputable def f (b x : ℝ) : ℝ :=
  if x > 1 then (2 * b - 1) / x + b + 3 else -x^2 + (2 - b) * x

theorem range_of_b_for_increasing_f :
  ∀ b : ℝ, (∀ x1 x2 : ℝ, x1 < x2 → f b x1 ≤ f b x2) ↔ -1/4 ≤ b ∧ b ≤ 0 := 
sorry

end range_of_b_for_increasing_f_l250_250098


namespace monotonically_decreasing_intervals_max_and_min_values_on_interval_l250_250247

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ x : ℝ, (x < -1 ∨ x > 3) → f x a < f (x+1) a :=
sorry

theorem max_and_min_values_on_interval : 
  (f (-1) (-2) = -7) ∧ (max (f (-2) (-2)) (f 2 (-2)) = 20) :=
sorry

end monotonically_decreasing_intervals_max_and_min_values_on_interval_l250_250247


namespace rate_of_interest_per_annum_l250_250585

theorem rate_of_interest_per_annum (P R : ℝ) (T : ℝ) 
  (h1 : T = 8)
  (h2 : (P / 5) = (P * R * T) / 100) : 
  R = 2.5 := 
by
  sorry

end rate_of_interest_per_annum_l250_250585


namespace existence_of_function_implies_a_le_1_l250_250677

open Real

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem existence_of_function_implies_a_le_1 (a : ℝ) :
  (∃ f : positive_reals → positive_reals, ∀ x : positive_reals, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) → a ≤ 1 :=
by
  sorry

end existence_of_function_implies_a_le_1_l250_250677


namespace ratio_B_C_l250_250294

def total_money := 595
def A_share := 420
def B_share := 105
def C_share := 70

-- The main theorem stating the expected ratio
theorem ratio_B_C : (B_share / C_share : ℚ) = 3 / 2 := by
  sorry

end ratio_B_C_l250_250294


namespace min_value_expression_l250_250979

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + z = 3) (h2 : z = (x + y) / 2) : 
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) = 3 / 2 :=
by sorry

end min_value_expression_l250_250979


namespace find_a8_l250_250360

variable {α : Type} [LinearOrderedField α]

/-- Given conditions of an arithmetic sequence -/
def arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a_n n = a1 + n * d

theorem find_a8 (a_n : ℕ → ℝ)
  (h_arith : arithmetic_sequence a_n)
  (h3 : a_n 3 = 5)
  (h5 : a_n 5 = 3) :
  a_n 8 = 0 :=
sorry

end find_a8_l250_250360


namespace no_solution_eq1_l250_250451

   theorem no_solution_eq1 : ¬ ∃ x, (3 - x) / (x - 4) - 1 / (4 - x) = 1 :=
   by
     sorry
   
end no_solution_eq1_l250_250451


namespace books_a_count_l250_250879

-- Variables representing the number of books (a) and (b)
variables (A B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := A + B = 20
def condition2 : Prop := A = B + 4

-- The theorem to prove
theorem books_a_count (h1 : condition1 A B) (h2 : condition2 A B) : A = 12 :=
sorry

end books_a_count_l250_250879


namespace horse_running_time_l250_250878

def area_of_square_field : Real := 625
def speed_of_horse_around_field : Real := 25

theorem horse_running_time : (4 : Real) = 
  let side_length := Real.sqrt area_of_square_field
  let perimeter := 4 * side_length
  perimeter / speed_of_horse_around_field :=
by
  sorry

end horse_running_time_l250_250878


namespace speed_of_first_train_l250_250171

noncomputable def speed_of_second_train : ℝ := 40 -- km/h
noncomputable def length_of_first_train : ℝ := 125 -- m
noncomputable def length_of_second_train : ℝ := 125.02 -- m
noncomputable def time_to_pass_each_other : ℝ := 1.5 / 60 -- hours (converted from minutes)

theorem speed_of_first_train (V1 V2 : ℝ) 
  (h1 : V2 = speed_of_second_train)
  (h2 : 125 + 125.02 = 250.02) 
  (h3 : 1.5 / 60 = 0.025) :
  V1 - V2 = 10.0008 → V1 = 50 :=
by 
  sorry

end speed_of_first_train_l250_250171


namespace triangle_inequality_l250_250434

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) : Prop :=
    a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 3 5 7 (by norm_num) (by norm_num) (by norm_num) :=
by simp; apply and.intro (by norm_num) (by norm_num)

end triangle_inequality_l250_250434


namespace percentage_of_3rd_graders_l250_250303

theorem percentage_of_3rd_graders (students_jackson students_madison : ℕ)
  (percent_3rd_grade_jackson percent_3rd_grade_madison : ℝ) :
  students_jackson = 200 → percent_3rd_grade_jackson = 25 →
  students_madison = 300 → percent_3rd_grade_madison = 35 →
  ((percent_3rd_grade_jackson / 100 * students_jackson +
    percent_3rd_grade_madison / 100 * students_madison) /
   (students_jackson + students_madison) * 100) = 31 :=
by 
  intros hjackson_percent hmpercent 
    hpercent_jack_percent hpercent_mad_percent
  -- Proof Placeholder
  sorry

end percentage_of_3rd_graders_l250_250303


namespace hallway_width_equals_four_l250_250398

-- Define the conditions: dimensions of the areas and total installed area.
def centralAreaLength : ℝ := 10
def centralAreaWidth : ℝ := 10
def centralArea : ℝ := centralAreaLength * centralAreaWidth

def totalInstalledArea : ℝ := 124
def hallwayLength : ℝ := 6

-- Total area minus central area's area yields hallway's area
def hallwayArea : ℝ := totalInstalledArea - centralArea

-- Statement to prove: the width of the hallway given its area and length.
theorem hallway_width_equals_four :
  (hallwayArea / hallwayLength) = 4 := 
by
  sorry

end hallway_width_equals_four_l250_250398


namespace inverse_function_l250_250575

noncomputable def f (x : ℝ) := 3 - 7 * x + x^2

noncomputable def g (x : ℝ) := (7 + Real.sqrt (37 + 4 * x)) / 2

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  intros x
  sorry

end inverse_function_l250_250575


namespace value_of_M_is_800_l250_250216

theorem value_of_M_is_800 : 
  let M := (sum (list.map (λ n, if n % 3 == 1 then (list.nth_le [50, 47, ..., 5] n (by sorry))^2 - (list.nth_le [49, 46, ..., 4] n (by sorry))^2 else (list.nth_le [48, 45, ..., 3] n (by sorry))^2) [0..15]))
  M = 800 :=
by sorry

end value_of_M_is_800_l250_250216


namespace shaded_region_area_l250_250673

theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  8 * (π * r * r / 4 - r * r / 2) / 2 = 50 * (π - 2) :=
by
  sorry

end shaded_region_area_l250_250673


namespace smallest_digit_divisible_by_9_l250_250349

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, (∃ m : ℕ, m = 2 + 4 + d + 6 + 0 ∧ m % 9 = 0 ∧ d < 10) ∧ d = 6 :=
by
  sorry

end smallest_digit_divisible_by_9_l250_250349


namespace income_increase_l250_250188

-- Definitions based on conditions
def original_price := 1.0
def original_items := 100.0
def discount := 0.10
def increased_sales := 0.15

-- Calculations for new values
def new_price := original_price * (1 - discount)
def new_items := original_items * (1 + increased_sales)
def original_income := original_price * original_items
def new_income := new_price * new_items

-- The percentage increase in income
def percentage_increase := ((new_income - original_income) / original_income) * 100

-- The theorem to prove that the percentage increase in gross income is 3.5%
theorem income_increase : percentage_increase = 3.5 := 
by
  -- This is where the proof would go
  sorry

end income_increase_l250_250188


namespace probability_four_1s_in_five_rolls_l250_250260

open ProbabilityTheory

theorem probability_four_1s_in_five_rolls :
  let p1 := (1 / 8 : ℚ) -- probability of rolling a 1
  let p_not_1 := (7 / 8 : ℚ) -- probability of not rolling a 1
  let comb := (Nat.choose 5 4 : ℚ) -- number of ways to choose four positions out of five
  let prob := comb * (p1 ^ 4) * p_not_1 -- required probability
  prob = 35 / 32768 := 
by
  sorry

end probability_four_1s_in_five_rolls_l250_250260


namespace remainder_p11_minus_3_div_p_minus_2_l250_250228

def f (p : ℕ) : ℕ := p^11 - 3

theorem remainder_p11_minus_3_div_p_minus_2 : f 2 = 2045 := 
by 
  sorry

end remainder_p11_minus_3_div_p_minus_2_l250_250228


namespace range_of_m_l250_250956

theorem range_of_m (x m : ℝ) (h₀ : -2 ≤ x ∧ x ≤ 11)
  (h₁ : 1 - 3 * m ≤ x ∧ x ≤ 3 + m)
  (h₂ : ¬ (-2 ≤ x ∧ x ≤ 11) → ¬ (1 - 3 * m ≤ x ∧ x ≤ 3 + m)) :
  m ≥ 8 :=
by
  sorry

end range_of_m_l250_250956


namespace binom_30_3_is_4060_l250_250202

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l250_250202


namespace molecular_weight_calc_l250_250577

namespace MolecularWeightProof

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def number_of_H : ℕ := 1
def number_of_Br : ℕ := 1
def number_of_O : ℕ := 3

theorem molecular_weight_calc :
  (number_of_H * atomic_weight_H + number_of_Br * atomic_weight_Br + number_of_O * atomic_weight_O) = 128.91 :=
by
  sorry

end MolecularWeightProof

end molecular_weight_calc_l250_250577


namespace cost_of_paving_l250_250300

noncomputable def length : Float := 5.5
noncomputable def width : Float := 3.75
noncomputable def cost_per_sq_meter : Float := 600

theorem cost_of_paving :
  (length * width * cost_per_sq_meter) = 12375 := by
  sorry

end cost_of_paving_l250_250300


namespace math_proof_l250_250974

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof_l250_250974


namespace sum_of_reciprocals_is_five_l250_250160

theorem sum_of_reciprocals_is_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = 3 * x * y) : 
  (1 / x) + (1 / y) = 5 :=
sorry

end sum_of_reciprocals_is_five_l250_250160


namespace convert_decimal_to_fraction_l250_250438

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l250_250438


namespace total_wages_of_12_men_l250_250093

variable {M W B x y : Nat}
variable {total_wages : Nat}

-- Condition 1: 12 men do the work equivalent to W women
axiom work_equivalent_1 : 12 * M = W

-- Condition 2: 12 men do the work equivalent to 20 boys
axiom work_equivalent_2 : 12 * M = 20 * B

-- Condition 3: All together earn Rs. 450
axiom total_earnings : (12 * M) + (x * (12 * M / W)) + (y * (12 * M / (20 * B))) = 450

-- The theorem to prove
theorem total_wages_of_12_men : total_wages = 12 * M → false :=
by sorry

end total_wages_of_12_men_l250_250093


namespace largest_x_exists_largest_x_largest_real_number_l250_250786

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l250_250786


namespace problem1_problem2_l250_250063

theorem problem1 :
  Real.sqrt 27 - (Real.sqrt 2 * Real.sqrt 6) + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := 
  by sorry

theorem problem2 :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 := 
  by sorry

end problem1_problem2_l250_250063


namespace irreducible_fraction_l250_250693

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end irreducible_fraction_l250_250693


namespace yacht_arrangement_l250_250168

theorem yacht_arrangement (n k : ℕ) (h_tourists : n = 5) (h_yachts : k = 2) (h_min_tourists_per_yacht : ∀ (a b : ℕ), a + b = n → a ≥ 2 → b ≥ 2) :
  ∃(arrangements : ℕ), arrangements = 20 :=
by
  sorry

end yacht_arrangement_l250_250168


namespace desired_cost_per_pound_l250_250745

/-- 
Let $p_1 = 8$, $w_1 = 25$, $p_2 = 5$, and $w_2 = 50$ represent the prices and weights of two types of candies.
Calculate the desired cost per pound $p_m$ of the mixture.
-/
theorem desired_cost_per_pound 
  (p1 : ℝ) (w1 : ℝ) (p2 : ℝ) (w2 : ℝ) (p_m : ℝ) 
  (h1 : p1 = 8) (h2 : w1 = 25) (h3 : p2 = 5) (h4 : w2 = 50) :
  p_m = (p1 * w1 + p2 * w2) / (w1 + w2) → p_m = 6 :=
by 
  intros
  sorry

end desired_cost_per_pound_l250_250745


namespace minimum_reciprocal_sum_l250_250537

noncomputable def minimum_value_of_reciprocal_sum (x y z : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 then 
    max (1/x + 1/y + 1/z) (9/2)
  else
    0
  
theorem minimum_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2): 
  1/x + 1/y + 1/z ≥ 9/2 :=
sorry

end minimum_reciprocal_sum_l250_250537


namespace candy_left_l250_250488

-- Definitions according to the conditions
def initialCandy : ℕ := 15
def candyGivenToHaley : ℕ := 6

-- Theorem statement formalizing the proof problem
theorem candy_left (c : ℕ) (h₁ : c = initialCandy - candyGivenToHaley) : c = 9 :=
by
  -- The proof is omitted as instructed.
  sorry

end candy_left_l250_250488


namespace ones_digit_of_8_pow_47_l250_250024

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l250_250024


namespace factoring_difference_of_squares_l250_250050

theorem factoring_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := 
sorry

end factoring_difference_of_squares_l250_250050


namespace john_needs_20_nails_l250_250489

-- Define the given conditions
def large_planks (n : ℕ) := n = 12
def small_planks (n : ℕ) := n = 10
def nails_for_large_planks (n : ℕ) := n = 15
def nails_for_small_planks (n : ℕ) := n = 5

-- Define the total number of nails needed
def total_nails_needed (n : ℕ) :=
  ∃ (lp sp np_large np_small : ℕ),
  large_planks lp ∧ small_planks sp ∧ nails_for_large_planks np_large ∧ nails_for_small_planks np_small ∧ n = np_large + np_small

-- The theorem statement
theorem john_needs_20_nails : total_nails_needed 20 :=
by { sorry }

end john_needs_20_nails_l250_250489


namespace parameterize_circle_l250_250339

noncomputable def parametrization (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 1) / (t^2 + 1), (-2 * t) / (t^2 + 1) )

theorem parameterize_circle (t : ℝ) : 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  (x^2 + y^2) = 1 :=
by 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  sorry

end parameterize_circle_l250_250339


namespace denis_fourth_board_score_l250_250773

theorem denis_fourth_board_score :
  ∀ (darts_per_board points_first_board points_second_board points_third_board points_total_boards : ℕ),
    darts_per_board = 3 →
    points_first_board = 30 →
    points_second_board = 38 →
    points_third_board = 41 →
    points_total_boards = (points_first_board + points_second_board + points_third_board) / 2 →
    points_total_boards = 34 :=
by
  intros darts_per_board points_first_board points_second_board points_third_board points_total_boards h1 h2 h3 h4 h5
  sorry

end denis_fourth_board_score_l250_250773


namespace Watson_class_student_count_l250_250425

def num_kindergartners : ℕ := 14
def num_first_graders : ℕ := 24
def num_second_graders : ℕ := 4

def total_students : ℕ := num_kindergartners + num_first_graders + num_second_graders

theorem Watson_class_student_count : total_students = 42 := 
by
    sorry

end Watson_class_student_count_l250_250425


namespace solution_set_line_l250_250124

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l250_250124


namespace equalize_costs_l250_250856

theorem equalize_costs (A B : ℝ) (h_lt : A < B) :
  (B - A) / 2 = (A + B) / 2 - A :=
by sorry

end equalize_costs_l250_250856


namespace percentage_vets_recommend_puppy_kibble_l250_250452

theorem percentage_vets_recommend_puppy_kibble :
  ∀ (P : ℝ), (30 / 100 * 1000 = 300) → (1000 * P / 100 + 100 = 300) → P = 20 :=
by
  intros P h1 h2
  sorry

end percentage_vets_recommend_puppy_kibble_l250_250452


namespace find_fencing_cost_l250_250481

theorem find_fencing_cost
  (d : ℝ) (cost_per_meter : ℝ) (π : ℝ)
  (h1 : d = 22)
  (h2 : cost_per_meter = 2.50)
  (hπ : π = Real.pi) :
  (cost_per_meter * (π * d) = 172.80) :=
sorry

end find_fencing_cost_l250_250481


namespace unique_positive_real_solution_l250_250627

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l250_250627


namespace area_ratio_XYZ_PQR_l250_250376

theorem area_ratio_XYZ_PQR 
  (PR PQ QR : ℝ)
  (p q r : ℝ) 
  (hPR : PR = 15) 
  (hPQ : PQ = 20) 
  (hQR : QR = 25)
  (hPX : p * PR = PR * p)
  (hQY : q * QR = QR * q) 
  (hPZ : r * PQ = PQ * r) 
  (hpq_sum : p + q + r = 3 / 4) 
  (hpq_sq_sum : p^2 + q^2 + r^2 = 9 / 16) : 
  (area_triangle_XYZ / area_triangle_PQR = 1 / 4) :=
sorry

end area_ratio_XYZ_PQR_l250_250376


namespace tan_thirteen_pi_over_four_l250_250620

theorem tan_thirteen_pi_over_four : 
  let θ := (13 * Real.pi) / 4 in
  Real.tan θ = 1 := by
  let θ := (13 * Real.pi) / 4
  have h1 : θ = (9 * Real.pi) / 4 + Real.pi := by
    sorry
  have h2 : Real.tan θ = Real.tan ((9 * Real.pi) / 4 + Real.pi) := by
    sorry
  have h3 : Real.tan ((9 * Real.pi) / 4 + Real.pi) = Real.tan (Real.pi / 4) := by
    sorry
  have h4 : Real.tan (Real.pi / 4) = 1 := by
    exact Real.tan_pi_over_four
  exact Eq.trans h2 (Eq.trans h3 h4)

end tan_thirteen_pi_over_four_l250_250620


namespace first_player_wins_l250_250374

-- Define the game state and requirements
inductive Player
| first : Player
| second : Player

-- Game state consists of a number of stones and whose turn it is
structure GameState where
  stones : Nat
  player : Player

-- Define a simple transition for the game
def take_stones (s : GameState) (n : Nat) : GameState :=
  { s with stones := s.stones - n, player := Player.second }

-- Determine if a player can take n stones
def can_take (s : GameState) (n : Nat) : Prop :=
  n >= 1 ∧ n <= 4 ∧ n <= s.stones

-- Define victory condition
def wins (s : GameState) : Prop :=
  s.stones = 0 ∧ s.player = Player.second

-- Prove that if the first player starts with 18 stones and picks 3 stones initially,
-- they can ensure victory
theorem first_player_wins :
  ∀ (s : GameState),
    s.stones = 18 ∧ s.player = Player.first →
    can_take s 3 →
    wins (take_stones s 3)
:= by
  sorry

end first_player_wins_l250_250374


namespace mower_value_drop_l250_250759

theorem mower_value_drop :
  ∀ (initial_value value_six_months value_after_year : ℝ) (percentage_drop_six_months percentage_drop_next_year : ℝ),
  initial_value = 100 →
  percentage_drop_six_months = 0.25 →
  value_six_months = initial_value * (1 - percentage_drop_six_months) →
  value_after_year = 60 →
  percentage_drop_next_year = 1 - (value_after_year / value_six_months) →
  percentage_drop_next_year * 100 = 20 :=
by
  intros initial_value value_six_months value_after_year percentage_drop_six_months percentage_drop_next_year
  intros h1 h2 h3 h4 h5
  sorry

end mower_value_drop_l250_250759


namespace min_students_l250_250157

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end min_students_l250_250157


namespace find_BF_pqsum_l250_250700

noncomputable def square_side_length : ℝ := 900
noncomputable def EF_length : ℝ := 400
noncomputable def m_angle_EOF : ℝ := 45
noncomputable def center_mid_to_side : ℝ := square_side_length / 2

theorem find_BF_pqsum :
  let G_mid : ℝ := center_mid_to_side
  let x : ℝ := G_mid - (2 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let y : ℝ := (1 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let BF := G_mid - y
  BF = 250 + 50 * Real.sqrt 7 ->
  250 + 50 + 7 = 307 := sorry

end find_BF_pqsum_l250_250700


namespace value_of_x_is_4_l250_250107

variable {A B C D E F G H P : ℕ}

theorem value_of_x_is_4 (h1 : 5 + A + B = 19)
                        (h2 : A + B + C = 19)
                        (h3 : C + D + E = 19)
                        (h4 : D + E + F = 19)
                        (h5 : F + x + G = 19)
                        (h6 : x + G + H = 19)
                        (h7 : H + P + 10 = 19) :
                        x = 4 :=
by
  sorry

end value_of_x_is_4_l250_250107


namespace quadratic_fraction_formula_l250_250251

theorem quadratic_fraction_formula (p q α β : ℝ) 
  (h1 : α + β = p) 
  (h2 : α * β = 6) 
  (h3 : p^2 ≠ 12) 
  (h4 : ∃ x : ℝ, x^2 - p * x + q = 0) :
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) :=
sorry

end quadratic_fraction_formula_l250_250251


namespace acute_angle_30_l250_250666

theorem acute_angle_30 (α : ℝ) (h : Real.cos (π / 6) * Real.sin α = Real.sqrt 3 / 4) : α = π / 6 := 
by 
  sorry

end acute_angle_30_l250_250666


namespace percentage_palm_oil_in_cheese_l250_250405

-- Define the conditions
variables (initial_cheese_price : ℝ) (initial_palm_oil_price : ℝ) (final_cheese_price : ℝ) (final_palm_oil_price : ℝ)

-- Condition 1: Initial price assumptions and price increase percentages
def conditions (initial_cheese_price initial_palm_oil_price : ℝ) : Prop :=
  final_cheese_price = initial_cheese_price * 1.03 ∧
  final_palm_oil_price = initial_palm_oil_price * 1.10

-- The main theorem to prove
theorem percentage_palm_oil_in_cheese
  (initial_cheese_price initial_palm_oil_price final_cheese_price final_palm_oil_price : ℝ)
  (h : conditions initial_cheese_price initial_palm_oil_price) :
  initial_palm_oil_price / initial_cheese_price = 0.30 :=
by
  sorry

end percentage_palm_oil_in_cheese_l250_250405


namespace least_x_for_inequality_l250_250950

theorem least_x_for_inequality : 
  ∃ (x : ℝ), (-x^2 + 9 * x - 20 ≤ 0) ∧ ∀ y, (-y^2 + 9 * y - 20 ≤ 0) → x ≤ y ∧ x = 4 := 
by
  sorry

end least_x_for_inequality_l250_250950


namespace number_of_flags_l250_250611

theorem number_of_flags (colors : Finset ℕ) (stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : stripes = 3) : 
  (colors.card ^ stripes) = 27 := 
by
  sorry

end number_of_flags_l250_250611


namespace find_first_week_customers_l250_250522

def commission_per_customer := 1
def first_week_customers (C : ℕ) := C
def second_week_customers (C : ℕ) := 2 * C
def third_week_customers (C : ℕ) := 3 * C
def salary := 500
def bonus := 50
def total_earnings := 760

theorem find_first_week_customers (C : ℕ) (H : salary + bonus + commission_per_customer * (first_week_customers C + second_week_customers C + third_week_customers C) = total_earnings) : 
  C = 35 :=
by
  sorry

end find_first_week_customers_l250_250522


namespace decode_division_problem_l250_250772

theorem decode_division_problem :
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  dividend / divisor = quotient :=
by {
  -- Definitions of given and derived values
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  -- The statement to prove
  sorry
}

end decode_division_problem_l250_250772


namespace triangle_statements_l250_250108

-- Define the fundamental properties of the triangle
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a = 45 ∧ a = 2 ∧ b = 2 * Real.sqrt 2 ∧ 
  (a - b = c * Real.cos B - c * Real.cos A)

-- Statement A
def statement_A (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  ∃ B, Real.sin B = 1

-- Statement B
def statement_B (A B C : ℝ) (v_AC v_AB : ℝ) : Prop :=
  v_AC * v_AB > 0 → Real.cos A > 0

-- Statement C
def statement_C (A B : ℝ) (a b : ℝ) : Prop :=
  Real.sin A > Real.sin B → a > b

-- Statement D
def statement_D (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  (a - b = c * Real.cos B - c * Real.cos A) →
  (a = b ∨ c^2 = a^2 + b^2)

-- Final proof statement
theorem triangle_statements (A B C a b c : ℝ) (v_AC v_AB : ℝ) 
  (h_triangle : triangle A B C a b c) :
  (statement_A A B C a b c h_triangle) ∧
  ¬(statement_B A B C v_AC v_AB) ∧
  (statement_C A B a b) ∧
  (statement_D A B C a b c h_triangle) :=
by sorry

end triangle_statements_l250_250108


namespace scientific_notation_equivalence_l250_250562

-- Define constants and variables
def scientific_notation {a b : ℝ} (n : ℝ) (a b : ℝ) := n = a * (10^b)

-- State the conditions
def seven_nm_equals := (7 : ℝ) * (10 : ℝ) ^ (-9 : ℝ) = 0.000000007

-- Theorem to prove
theorem scientific_notation_equivalence : scientific_notation 0.000000007 7 (-9) :=
by
  apply (seven_nm_equals)

end scientific_notation_equivalence_l250_250562


namespace ratio_of_areas_l250_250054

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l250_250054


namespace solve_for_x_l250_250151

theorem solve_for_x (x : ℤ) : (16 : ℝ) ^ (3 * x - 5) = ((1 : ℝ) / 4) ^ (2 * x + 6) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l250_250151


namespace maximum_b_value_l250_250249

noncomputable def f (x : ℝ) := Real.exp x - x - 1
def g (x : ℝ) := -x^2 + 4 * x - 3

theorem maximum_b_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : f a = g b) : b ≤ 3 := by
  sorry

end maximum_b_value_l250_250249


namespace yolanda_avg_three_point_baskets_l250_250907

noncomputable theory

def yolanda_points_season : ℕ := 345
def total_games : ℕ := 15
def free_throws_per_game : ℕ := 4
def two_point_baskets_per_game : ℕ := 5

theorem yolanda_avg_three_point_baskets :
  (345 - (15 * (4 * 1 + 5 * 2))) / 3 / 15 = 3 :=
by sorry

end yolanda_avg_three_point_baskets_l250_250907


namespace part_I_part_II_l250_250836

def f (x a : ℝ) : ℝ := |x - 4 * a| + |x|

theorem part_I (a : ℝ) (h : -4 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, f x a ≥ a^2 := 
sorry

theorem part_II (x y z : ℝ) (h : 4 * x + 2 * y + z = 4) :
  (x + y)^2 + y^2 + z^2 ≥ 16 / 21 :=
sorry

end part_I_part_II_l250_250836


namespace online_textbooks_cost_l250_250982

theorem online_textbooks_cost (x : ℕ) :
  (5 * 10) + x + 3 * x = 210 → x = 40 :=
by
  sorry

end online_textbooks_cost_l250_250982


namespace geometric_sequence_ratio_l250_250533

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h0 : q ≠ 1) 
  (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q)) 
  (h2 : ∀ n, a n = a 0 * q^n) 
  (h3 : 2 * S 3 = 7 * a 2) :
  (S 5 / a 2 = 31 / 2) ∨ (S 5 / a 2 = 31 / 8) :=
by sorry

end geometric_sequence_ratio_l250_250533


namespace shortest_ribbon_length_is_10_l250_250272

noncomputable def shortest_ribbon_length (L : ℕ) : Prop :=
  (∃ k1 : ℕ, L = 2 * k1) ∧ (∃ k2 : ℕ, L = 5 * k2)

theorem shortest_ribbon_length_is_10 : shortest_ribbon_length 10 :=
by
  sorry

end shortest_ribbon_length_is_10_l250_250272


namespace leo_class_girls_l250_250847

theorem leo_class_girls (g b : ℕ) 
  (h_ratio : 3 * b = 4 * g) 
  (h_total : g + b = 35) : g = 15 := 
by
  sorry

end leo_class_girls_l250_250847


namespace number_of_teachers_l250_250458

theorem number_of_teachers (total_people : ℕ) (sampled_individuals : ℕ) (sampled_students : ℕ) 
    (school_total : total_people = 2400) 
    (sample_total : sampled_individuals = 160) 
    (sample_students : sampled_students = 150) : 
    ∃ teachers : ℕ, teachers = 150 := 
by
  -- Proof omitted
  sorry

end number_of_teachers_l250_250458


namespace percentage_difference_l250_250752

-- Define the quantities involved
def milk_in_A : ℕ := 1264
def transferred_milk : ℕ := 158

-- Define the quantities of milk in container B and C after transfer
noncomputable def quantity_in_B : ℕ := milk_in_A / 2
noncomputable def quantity_in_C : ℕ := quantity_in_B

-- Prove that the percentage difference between the quantity of milk in container B
-- and the capacity of container A is 50%
theorem percentage_difference :
  ((milk_in_A - quantity_in_B) * 100 / milk_in_A) = 50 := sorry

end percentage_difference_l250_250752


namespace acute_angle_sine_solution_l250_250825

theorem acute_angle_sine_solution (α : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : sin (α - 10 * real.pi / 180) = real.sqrt 3 / 2) : α = 70 * real.pi / 180 := 
by
  sorry

end acute_angle_sine_solution_l250_250825


namespace least_xy_value_l250_250494

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l250_250494


namespace jenny_mother_age_l250_250543

theorem jenny_mother_age:
  (∀ x : ℕ, (50 + x = 2 * (10 + x)) → (2010 + x = 2040)) :=
by
  sorry

end jenny_mother_age_l250_250543


namespace yolanda_three_point_avg_l250_250908

-- Definitions based on conditions
def total_points_season := 345
def total_games := 15
def free_throws_per_game := 4
def two_point_baskets_per_game := 5

-- Definitions based on the derived quantities
def average_points_per_game := total_points_season / total_games
def points_from_two_point_baskets := two_point_baskets_per_game * 2
def points_from_free_throws := free_throws_per_game * 1
def points_from_non_three_point_baskets := points_from_two_point_baskets + points_from_free_throws
def points_from_three_point_baskets := average_points_per_game - points_from_non_three_point_baskets
def three_point_baskets_per_game := points_from_three_point_baskets / 3

-- The theorem to prove that Yolanda averaged 3 three-point baskets per game
theorem yolanda_three_point_avg:
  three_point_baskets_per_game = 3 := sorry

end yolanda_three_point_avg_l250_250908


namespace actual_revenue_is_60_percent_of_projected_l250_250735

variable (R : ℝ)

-- Condition: Projected revenue is 25% more than last year's revenue
def projected_revenue (R : ℝ) : ℝ := 1.25 * R

-- Condition: Actual revenue decreased by 25% compared to last year's revenue
def actual_revenue (R : ℝ) : ℝ := 0.75 * R

-- Theorem: Prove that the actual revenue is 60% of the projected revenue
theorem actual_revenue_is_60_percent_of_projected :
  (actual_revenue R) = 0.6 * (projected_revenue R) :=
  sorry

end actual_revenue_is_60_percent_of_projected_l250_250735


namespace unique_positive_solution_eq_15_l250_250630

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l250_250630


namespace equivalent_set_complement_intersection_l250_250365

def setM : Set ℝ := {x | -3 < x ∧ x < 1}
def setN : Set ℝ := {x | x ≤ 3}
def givenSet : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

theorem equivalent_set_complement_intersection :
  givenSet = (setM ∩ setN)ᶜ :=
sorry

end equivalent_set_complement_intersection_l250_250365


namespace problem_statement_l250_250694

theorem problem_statement (x : ℝ) (h : (2024 - x)^2 + (2022 - x)^2 = 4038) : 
  (2024 - x) * (2022 - x) = 2017 :=
sorry

end problem_statement_l250_250694


namespace g_is_even_and_symmetric_l250_250005

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem g_is_even_and_symmetric :
  (∀ x : ℝ, g x = g (-x)) ∧ (∀ k : ℤ, g ((2 * k - 1) * π / 8) = 0) :=
by
  sorry

end g_is_even_and_symmetric_l250_250005


namespace min_value_f_l250_250695

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4 * x + 20) + Real.sqrt (x^2 + 2 * x + 10)

theorem min_value_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 :=
by
  sorry

end min_value_f_l250_250695


namespace polar_to_rectangular_l250_250065

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 4) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 :=
by
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l250_250065


namespace ab_neither_sufficient_nor_necessary_l250_250275

theorem ab_neither_sufficient_nor_necessary (a b : ℝ) (h : a * b ≠ 0) :
  (¬ ((a * b > 1) → (a > 1 / b))) ∧ (¬ ((a > 1 / b) → (a * b > 1))) :=
by
  sorry

end ab_neither_sufficient_nor_necessary_l250_250275


namespace problem_statement_l250_250966

theorem problem_statement (x y : ℝ) (h₁ : 2.5 * x = 0.75 * y) (h₂ : x = 20) : y = 200 / 3 := by
  sorry

end problem_statement_l250_250966


namespace problem_1_problem_2_l250_250364

-- Define sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- First problem statement
theorem problem_1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  -- proof omitted
  sorry

-- Second problem statement
theorem problem_2 (a : ℝ) : (∅ ⊆ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  -- proof omitted
  sorry

end problem_1_problem_2_l250_250364


namespace recipe_total_cups_l250_250885

noncomputable def total_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℕ) : ℕ :=
  let part := sugar_cups / sugar_ratio
  let butter_cups := butter_ratio * part
  let flour_cups := flour_ratio * part
  butter_cups + flour_cups + sugar_cups

theorem recipe_total_cups : 
  total_cups 2 7 5 10 = 28 :=
by
  sorry

end recipe_total_cups_l250_250885


namespace triangle_area_l250_250545

variables {A B C D M N: Type}

-- Define the conditions and the proof 
theorem triangle_area
  (α β : ℝ)
  (CD : ℝ)
  (sin_Ratio : ℝ)
  (C_angle : ℝ)
  (MCN_Area : ℝ)
  (M_distance : ℝ)
  (N_distance : ℝ)
  (hCD : CD = Real.sqrt 13)
  (hSinRatio : (Real.sin α) / (Real.sin β) = 4 / 3)
  (hC_angle : C_angle = 120)
  (hMCN_Area : MCN_Area = 3 * Real.sqrt 3)
  (hDistance : M_distance = 2 * N_distance)
  : ∃ ABC_Area, ABC_Area = 27 * Real.sqrt 3 / 2 :=
sorry

end triangle_area_l250_250545


namespace percentage_of_adults_is_40_l250_250919

variables (A C : ℕ)

-- Given conditions as definitions
def total_members := 120
def more_children_than_adults := 24
def percentage_of_adults (A : ℕ) := (A.toFloat / total_members.toFloat) * 100

-- Lean 4 statement to prove the percentage of adults
theorem percentage_of_adults_is_40 (h1 : A + C = 120)
                                   (h2 : C = A + 24) :
  percentage_of_adults A = 40 :=
by
  sorry

end percentage_of_adults_is_40_l250_250919


namespace value_of_m_l250_250394

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l250_250394


namespace largest_x_l250_250780

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l250_250780


namespace train_cross_time_l250_250915

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

end train_cross_time_l250_250915


namespace previous_painting_price_l250_250546

-- Define the amount received for the most recent painting
def recentPainting (p : ℕ) := 5 * p - 1000

-- Define the target amount
def target := 44000

-- State that the target amount is achieved by the prescribed function
theorem previous_painting_price : recentPainting 9000 = target :=
by
  sorry

end previous_painting_price_l250_250546


namespace probability_is_two_thirds_l250_250243

noncomputable def probabilityOfEvent : ℚ :=
  let Ω := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 ∧ 2 * p.1 - p.2 + 2 ≥ 0 }
  let area_Ω := (2 - 0) * (6 - 0)
  let area_A := area_Ω - (1 / 2) * 2 * 4
  (area_A / area_Ω : ℚ)

theorem probability_is_two_thirds : probabilityOfEvent = (2 / 3 : ℚ) :=
  sorry

end probability_is_two_thirds_l250_250243


namespace product_of_common_ratios_l250_250277

theorem product_of_common_ratios (x p r a2 a3 b2 b3 : ℝ)
  (h1 : a2 = x * p) (h2 : a3 = x * p^2)
  (h3 : b2 = x * r) (h4 : b3 = x * r^2)
  (h5 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2))
  (h_nonconstant : x ≠ 0) (h_diff_ratios : p ≠ r) :
  p * r = 9 :=
by
  sorry

end product_of_common_ratios_l250_250277


namespace total_first_year_students_l250_250377

theorem total_first_year_students (males : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) (N : ℕ)
  (h1 : males = 570)
  (h2 : sample_size = 110)
  (h3 : female_in_sample = 53)
  (h4 : N = ((sample_size - female_in_sample) * males) / (sample_size - (sample_size - female_in_sample)))
  : N = 1100 := 
by
  sorry

end total_first_year_students_l250_250377


namespace probability_X_eq_2_l250_250820

namespace Hypergeometric

def combin (n k : ℕ) : ℕ := n.choose k

noncomputable def hypergeometric (N M n k : ℕ) : ℚ :=
  (combin M k * combin (N - M) (n - k)) / combin N n

theorem probability_X_eq_2 :
  hypergeometric 8 5 3 2 = 15 / 28 := by
  sorry

end Hypergeometric

end probability_X_eq_2_l250_250820


namespace parabola_vertex_is_two_one_l250_250998

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end parabola_vertex_is_two_one_l250_250998


namespace distance_to_focus_parabola_l250_250652

theorem distance_to_focus_parabola (F P : ℝ × ℝ) (hF : F = (0, -1/2))
  (hP : P = (1, 2)) (C : ℝ × ℝ → Prop)
  (hC : ∀ x, C (x, 2 * x^2)) : dist P F = 17 / 8 := by
sorry

end distance_to_focus_parabola_l250_250652


namespace additional_people_proof_l250_250069

variable (initialPeople additionalPeople mowingHours trimmingRate totalNewPeople totalMowingPeople requiredPersonHours totalPersonHours: ℕ)

noncomputable def mowingLawn (initialPeople mowingHours : ℕ) : ℕ :=
  initialPeople * mowingHours

noncomputable def mowingRate (requiredPersonHours : ℕ) (mowingHours : ℕ) : ℕ :=
  (requiredPersonHours / mowingHours)

noncomputable def trimmingEdges (totalMowingPeople trimmingRate : ℕ) : ℕ :=
  (totalMowingPeople / trimmingRate)

noncomputable def totalPeople (mowingPeople trimmingPeople : ℕ) : ℕ :=
  (mowingPeople + trimmingPeople)

noncomputable def additionalPeopleNeeded (totalPeople initialPeople : ℕ) : ℕ :=
  (totalPeople - initialPeople)

theorem additional_people_proof :
  initialPeople = 8 →
  mowingHours = 3 →
  totalPersonHours = mowingLawn initialPeople mowingHours →
  totalMowingPeople = mowingRate totalPersonHours 2 →
  trimmingRate = 3 →
  requiredPersonHours = totalPersonHours →
  totalNewPeople = totalPeople totalMowingPeople (trimmingEdges totalMowingPeople trimmingRate) →
  additionalPeople = additionalPeopleNeeded totalNewPeople initialPeople →
  additionalPeople = 8 :=
by
  sorry

end additional_people_proof_l250_250069


namespace lowest_price_per_component_l250_250910

theorem lowest_price_per_component (cost_per_component shipping_per_component fixed_costs num_components : ℕ) 
  (h_cost_per_component : cost_per_component = 80)
  (h_shipping_per_component : shipping_per_component = 5)
  (h_fixed_costs : fixed_costs = 16500)
  (h_num_components : num_components = 150) :
  (cost_per_component + shipping_per_component) * num_components + fixed_costs = 29250 ∧
  29250 / 150 = 195 :=
by
  sorry

end lowest_price_per_component_l250_250910


namespace bus_pickup_time_l250_250397

open Time  -- Open the Time namespace

/-- 
 Conditions:
 1. The bus takes forty minutes to arrive at the first station.
 2. Mr. Langsley arrives at work at 9:00 a.m.
 3. The total time from the first station to Mr. Langsley's workplace is 140 minutes.

 Goal:
 The bus picks Mr. Langsley up at 6:00 a.m.
-/
theorem bus_pickup_time :
  let bus_to_station := 40  -- minutes
  let arrival_time := Time.mk 9 0  -- 9:00 am
  let first_station_to_work := 140  -- minutes
  (arrival_time - (bus_to_station + first_station_to_work)) = Time.mk 6 0 := 
by
  sorry

end bus_pickup_time_l250_250397


namespace problem1_problem2_l250_250761

-- First problem
theorem problem1 :
  2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) - (-1 / 3) ^ 0 + (-1) ^ 2023 = -2 :=
by
  sorry

-- Second problem
theorem problem2 :
  abs (1 - Real.sqrt 2) - Real.sqrt 12 + (1 / 3) ^ (-1 : ℤ) - 2 * Real.cos (Real.pi / 4) = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l250_250761


namespace length_PZ_l250_250516

-- Define the given conditions
variables (CD WX : ℝ) -- segments CD and WX
variable (CW : ℝ) -- length of segment CW
variable (DP : ℝ) -- length of segment DP
variable (PX : ℝ) -- length of segment PX

-- Define the similarity condition
-- segment CD is parallel to segment WX implies that the triangles CDP and WXP are similar

-- Define what we want to prove
theorem length_PZ (hCD_WX_parallel : CD = WX)
                  (hCW : CW = 56)
                  (hDP : DP = 18)
                  (hPX : PX = 36) :
  ∃ PZ : ℝ, PZ = 4 / 3 :=
by
  -- proof steps here (omitted)
  sorry

end length_PZ_l250_250516


namespace unique_positive_real_solution_l250_250624

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l250_250624


namespace total_earnings_proof_l250_250923

-- Definitions of the given conditions
def monthly_earning : ℕ := 4000
def monthly_saving : ℕ := 500
def total_savings_needed : ℕ := 45000

-- Lean statement for the proof problem
theorem total_earnings_proof : 
  (total_savings_needed / monthly_saving) * monthly_earning = 360000 :=
by
  sorry

end total_earnings_proof_l250_250923


namespace main_inequality_l250_250073

theorem main_inequality (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ m = -4 := by
  sorry

end main_inequality_l250_250073


namespace solution_of_system_l250_250132

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l250_250132


namespace largest_x_satisfies_condition_l250_250800

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l250_250800


namespace red_tile_probability_l250_250739

def is_red_tile (n : ℕ) : Prop := n % 7 = 3

noncomputable def red_tiles_count : ℕ :=
  Nat.card {n : ℕ | n ≤ 70 ∧ is_red_tile n}

noncomputable def total_tiles_count : ℕ := 70

theorem red_tile_probability :
  (red_tiles_count : ℤ) / (total_tiles_count : ℤ) = (1 : ℤ) / 7 :=
sorry

end red_tile_probability_l250_250739


namespace minimum_area_sum_l250_250823

-- Define the coordinates and the conditions
variable {x1 y1 x2 y2 : ℝ}
variable (on_parabola_A : y1^2 = x1)
variable (on_parabola_B : y2^2 = x2)
variable (y1_pos : y1 > 0)
variable (y2_neg : y2 < 0)
variable (dot_product : x1 * x2 + y1 * y2 = 2)

-- Define the function to calculate areas
noncomputable def area_sum (y1 y2 x1 x2 : ℝ) : ℝ :=
  1/2 * 2 * (y1 - y2) + 1/2 * 1/4 * y1

theorem minimum_area_sum :
  ∃ y1 y2 x1 x2, y1^2 = x1 ∧ y2^2 = x2 ∧ y1 > 0 ∧ y2 < 0 ∧ x1 * x2 + y1 * y2 = 2 ∧
  (area_sum y1 y2 x1 x2 = 3) := sorry

end minimum_area_sum_l250_250823


namespace time_to_sweep_one_room_l250_250939

theorem time_to_sweep_one_room (x : ℕ) :
  (10 * x) = (2 * 9 + 6 * 2) → x = 3 := by
  sorry

end time_to_sweep_one_room_l250_250939


namespace largest_x_eq_48_div_7_l250_250805

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l250_250805


namespace find_simple_interest_sum_l250_250301

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * r * n / 100

theorem find_simple_interest_sum (P CIsum : ℝ)
  (simple_rate : ℝ) (simple_years : ℕ)
  (compound_rate : ℝ) (compound_years : ℕ)
  (compound_principal : ℝ)
  (hP : simple_interest P simple_rate simple_years = CIsum)
  (hCI : CIsum = (compound_interest compound_principal compound_rate compound_years - compound_principal) / 2) :
  P = 1272 :=
by
  sorry

end find_simple_interest_sum_l250_250301


namespace probability_not_win_l250_250582

theorem probability_not_win (A B : Fin 16) : 
  (256 - 16) / 256 = 15 / 16 := 
by
  sorry

end probability_not_win_l250_250582


namespace number_of_rows_seating_10_is_zero_l250_250513

theorem number_of_rows_seating_10_is_zero :
  ∀ (y : ℕ) (total_people : ℕ) (total_rows : ℕ),
    (∀ (r : ℕ), r * 9 + (total_rows - r) * 10 = total_people) →
    total_people = 54 →
    total_rows = 6 →
    y = 0 :=
by
  sorry

end number_of_rows_seating_10_is_zero_l250_250513


namespace smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l250_250361

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  -- Proof omitted
  sorry

theorem monotonically_increasing_interval :
  ∃ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
               k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤  k * Real.pi + Real.pi / 6 →
               x ≤ y → f x ≤ f y := 
by 
  -- Proof omitted
  sorry

theorem minimum_value_a_of_triangle (A B C a b c : ℝ) 
  (h₀ : f A = 1/2) 
  (h₁ : B^2 - C^2 - B * C * Real.cos A - a^2 = 4) :
  a ≥ 2 * Real.sqrt 2 :=
by 
  -- Proof omitted
  sorry

end smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l250_250361


namespace shirt_price_l250_250013

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end shirt_price_l250_250013


namespace symmetry_implies_condition_l250_250414

open Function

variable {R : Type*} [Field R]
variables (p q r s : R)

theorem symmetry_implies_condition
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0) 
  (h_symmetry : ∀ x y : R, y = (p * x + q) / (r * x - s) → 
                          -x = (p * (-y) + q) / (r * (-y) - s)) :
  r + s = 0 := 
sorry

end symmetry_implies_condition_l250_250414


namespace biography_increase_l250_250734

theorem biography_increase (B N : ℝ) (hN : N = 0.35 * (B + N) - 0.20 * B):
  (N / (0.20 * B) * 100) = 115.38 :=
by
  sorry

end biography_increase_l250_250734


namespace deposit_amount_l250_250183

theorem deposit_amount (P : ℝ) (deposit remaining : ℝ) (h1 : deposit = 0.1 * P) (h2 : remaining = P - deposit) (h3 : remaining = 1350) : 
  deposit = 150 := 
by
  sorry

end deposit_amount_l250_250183


namespace find_g7_l250_250004

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7_l250_250004


namespace problem_a_problem_b_problem_c_problem_d_l250_250410

-- a) Proof problem for \(x^2 + 5x + 6 < 0\)
theorem problem_a (x : ℝ) : x^2 + 5*x + 6 < 0 → -3 < x ∧ x < -2 := by
  sorry

-- b) Proof problem for \(-x^2 + 9x - 20 < 0\)
theorem problem_b (x : ℝ) : -x^2 + 9*x - 20 < 0 → x < 4 ∨ x > 5 := by
  sorry

-- c) Proof problem for \(x^2 + x - 56 < 0\)
theorem problem_c (x : ℝ) : x^2 + x - 56 < 0 → -8 < x ∧ x < 7 := by
  sorry

-- d) Proof problem for \(9x^2 + 4 < 12x\) (No solutions)
theorem problem_d (x : ℝ) : ¬ 9*x^2 + 4 < 12*x := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l250_250410


namespace ratio_of_x_to_y_l250_250027

theorem ratio_of_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l250_250027


namespace valid_triples_l250_250071

theorem valid_triples :
  ∀ (a b c : ℕ), 1 ≤ a → 1 ≤ b → 1 ≤ c →
  (∃ k : ℕ, 32 * a + 3 * b + 48 * c = 4 * k * a * b * c) ↔ 
  (a = 1 ∧ b = 20 ∧ c = 1) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 3 ∧ b = 4 ∧ c = 1) := 
by
  sorry

end valid_triples_l250_250071


namespace students_taking_german_l250_250848

theorem students_taking_german
  (total_students : ℕ)
  (french_students : ℕ)
  (both_courses_students : ℕ)
  (no_course_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : both_courses_students = 9)
  (h4 : no_course_students = 33)
  : ∃ german_students : ℕ, german_students = 22 := 
by
  -- proof can be filled in here
  sorry

end students_taking_german_l250_250848


namespace paper_cost_l250_250032
noncomputable section

variables (P C : ℝ)

theorem paper_cost (h : 100 * P + 200 * C = 6.00) : 
  20 * P + 40 * C = 1.20 :=
sorry

end paper_cost_l250_250032


namespace evaluate_fraction_l250_250618

theorem evaluate_fraction : (8 / 29) - (5 / 87) = (19 / 87) := sorry

end evaluate_fraction_l250_250618


namespace range_x1_x2_l250_250658

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_x1_x2 (a b c d x1 x2 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a + 2 * b + 3 * c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hx1 : f a b c x1 = 0)
  (hx2 : f a b c x2 = 0) :
  abs (x1 - x2) ∈ Set.Ico 0 (2 / 3) :=
sorry

end range_x1_x2_l250_250658


namespace distribution_ways_l250_250192

theorem distribution_ways (n_problems n_friends : ℕ) (h_problems : n_problems = 6) (h_friends : n_friends = 8) : (n_friends ^ n_problems) = 262144 :=
by
  rw [h_problems, h_friends]
  norm_num

end distribution_ways_l250_250192


namespace avg_age_of_coaches_l250_250995

theorem avg_age_of_coaches (n_girls n_boys n_coaches : ℕ)
  (avg_age_girls avg_age_boys avg_age_members : ℕ)
  (h_girls : n_girls = 30)
  (h_boys : n_boys = 15)
  (h_coaches : n_coaches = 5)
  (h_avg_age_girls : avg_age_girls = 18)
  (h_avg_age_boys : avg_age_boys = 19)
  (h_avg_age_members : avg_age_members = 20) :
  (n_girls * avg_age_girls + n_boys * avg_age_boys + n_coaches * 35) / (n_girls + n_boys + n_coaches) = avg_age_members :=
by sorry

end avg_age_of_coaches_l250_250995


namespace largest_real_solution_l250_250796

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l250_250796


namespace Z_is_all_positive_integers_l250_250450

theorem Z_is_all_positive_integers (Z : Set ℕ) (h_nonempty : Z.Nonempty)
(h1 : ∀ x ∈ Z, 4 * x ∈ Z)
(h2 : ∀ x ∈ Z, (Nat.sqrt x) ∈ Z) : 
Z = { n : ℕ | n > 0 } :=
sorry

end Z_is_all_positive_integers_l250_250450


namespace subinterval_exists_subinterval_not_exists_l250_250384

open MeasureTheory

theorem subinterval_exists {A B : set ℝ} (hA : A ⊆ (0, 1)) (hB : B ⊆ (0, 1)) (h_disjoint : disjoint A B)
  (h_muA : 0 < μ A) (h_muB : 0 < μ B) (n : ℕ) (hn : 0 < n) : 
  ∃ (c d : ℝ) (h_cd : c < d ∧ (c,d) ⊆ (0,1)), 
    μ (A ∩ set.Ioo c d) = (1 / n : ℝ) * μ A ∧ μ (B ∩ set.Ioo c d) = (1 / n : ℝ) * μ B :=
sorry

theorem subinterval_not_exists {A B : set ℝ} (hA : A ⊆ (0, 1)) (hB : B ⊆ (0, 1)) (h_disjoint : disjoint A B)
  (h_muA : 0 < μ A) (h_muB : 0 < μ B) (λ : ℝ) (hλ : ∀ n : ℕ, λ ≠ (1 / n : ℝ)) :
  ¬∃ (c d : ℝ) (h_cd : c < d ∧ (c,d) ⊆ (0,1)), 
    μ (A ∩ set.Ioo c d) = λ * μ A ∧ μ (B ∩ set.Ioo c d) = λ * μ B :=
sorry

end subinterval_exists_subinterval_not_exists_l250_250384


namespace find_ab_l250_250704

-- Define the statement to be proven
theorem find_ab (a b : ℕ) (h1 : (a + b) % 3 = 2)
                           (h2 : b % 5 = 3)
                           (h3 : (b - a) % 11 = 1) :
  10 * a + b = 23 := 
sorry

end find_ab_l250_250704


namespace find_other_integer_l250_250273

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 150) (h2 : x = 15 ∨ y = 15) : y = 30 :=
by
  sorry

end find_other_integer_l250_250273


namespace b_plus_c_is_square_l250_250566

-- Given the conditions:
variables (a b c : ℕ)
variable (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Condition 1: Positive integers
variable (h2 : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)  -- Condition 2: Pairwise relatively prime
variable (h3 : a % 2 = 1 ∧ c % 2 = 1)  -- Condition 3: a and c are odd
variable (h4 : a^2 + b^2 = c^2)  -- Condition 4: Pythagorean triple equation

-- Prove that b + c is the square of an integer
theorem b_plus_c_is_square : ∃ k : ℕ, b + c = k^2 :=
by
  sorry

end b_plus_c_is_square_l250_250566


namespace washer_cost_l250_250464

theorem washer_cost (D : ℝ) (H1 : D + (D + 220) = 1200) : D + 220 = 710 :=
by
  sorry

end washer_cost_l250_250464


namespace find_z_l250_250508

-- Condition: there exists a constant k such that z = k * w
def direct_variation (z w : ℝ): Prop := ∃ k, z = k * w

-- We set up the conditions given in the problem.
theorem find_z (k : ℝ) (hw1 : 10 = k * 5) (hw2 : w = -15) : direct_variation z w → z = -30 :=
by
  sorry

end find_z_l250_250508


namespace expression_divisible_512_l250_250990

theorem expression_divisible_512 (n : ℤ) (h : n % 2 ≠ 0) : (n^12 - n^8 - n^4 + 1) % 512 = 0 := 
by 
  sorry

end expression_divisible_512_l250_250990


namespace min_students_l250_250156

noncomputable def num_boys_min (students : ℕ) (girls : ℕ) : Prop :=
  ∃ (boys : ℕ), boys > (3 * girls / 2) ∧ students = boys + girls

theorem min_students (girls : ℕ) (h_girls : girls = 5) : ∃ n, num_boys_min n girls ∧ n = 13 :=
by
  use 13
  unfold num_boys_min
  use 8
  sorry

end min_students_l250_250156


namespace sum_coeff_eq_neg_two_l250_250270

theorem sum_coeff_eq_neg_two (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (1 - 2*x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  sorry

end sum_coeff_eq_neg_two_l250_250270


namespace icosahedron_path_count_l250_250180

-- Definitions from the conditions
def vertices := 12
def edges := 30
def top_adjacent := 5
def bottom_adjacent := 5

-- Define the total paths calculation based on the given structural conditions
theorem icosahedron_path_count (v e ta ba : ℕ) (hv : v = 12) (he : e = 30) (hta : ta = 5) (hba : ba = 5) : 
  (ta * (ta - 1) * (ba - 1)) * 2 = 810 :=
by
-- Insert calculation logic here if needed or detailed structure definitions
  sorry

end icosahedron_path_count_l250_250180


namespace largest_x_satisfies_condition_l250_250790

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l250_250790


namespace range_of_m_l250_250647

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : 
  (x + y ≥ m) → m ≤ 18 :=
sorry

end range_of_m_l250_250647


namespace chests_contents_l250_250526

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l250_250526


namespace A_serves_on_50th_week_is_Friday_l250_250552

-- Define the people involved in the rotation
inductive Person
| A | B | C | D | E | F

open Person

-- Define the function that computes the day A serves on given the number of weeks
def day_A_serves (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (days % 6 + 0) % 7 -- 0 is the offset for the initial day when A serves (Sunday)

theorem A_serves_on_50th_week_is_Friday :
  day_A_serves 50 = 5 :=
by
  -- We provide the proof here
  sorry

end A_serves_on_50th_week_is_Friday_l250_250552


namespace operation_is_addition_l250_250831

theorem operation_is_addition : (5 + (-5) = 0) :=
by
  sorry

end operation_is_addition_l250_250831


namespace chests_contents_l250_250525

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l250_250525


namespace eccentricity_range_of_hyperbola_l250_250654

open Real

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def eccentricity_range :=
  ∀ (a b c : ℝ), 
    ∃ (e : ℝ),
      hyperbola a b (-c) 0 ∧ -- condition for point F
      (a + b > 0) ∧ -- additional conditions due to hyperbola properties
      (1 < e ∧ e < 2)
      
theorem eccentricity_range_of_hyperbola :
  eccentricity_range :=
by
  sorry

end eccentricity_range_of_hyperbola_l250_250654


namespace simplified_radical_formula_l250_250061

theorem simplified_radical_formula (y : ℝ) (hy : 0 ≤ y):
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) :=
by
  sorry

end simplified_radical_formula_l250_250061


namespace sum_of_squares_l250_250722

def gcd (a b c : Nat) : Nat := (Nat.gcd (Nat.gcd a b) c)

theorem sum_of_squares {a b c : ℕ} (h1 : 3 * a + 2 * b = 4 * c)
                                   (h2 : 3 * c ^ 2 = 4 * a ^ 2 + 2 * b ^ 2)
                                   (h3 : gcd a b c = 1) :
  a^2 + b^2 + c^2 = 45 :=
by
  sorry

end sum_of_squares_l250_250722


namespace largest_real_solution_l250_250794

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l250_250794


namespace probability_P_is_1_over_3_l250_250685

-- Definitions and conditions
def A := 0
def B := 3
def C := 1
def D := 2
def length_AB := B - A
def length_CD := D - C

-- Problem statement to prove
theorem probability_P_is_1_over_3 : (length_CD / length_AB) = 1 / 3 := by
  sorry

end probability_P_is_1_over_3_l250_250685


namespace find_a_b_l250_250257

theorem find_a_b (a b : ℤ) (h : ∀ x : ℤ, (x - 2) * (x + 3) = x^2 + a * x + b) : a = 1 ∧ b = -6 :=
by
  sorry

end find_a_b_l250_250257


namespace evaluate_expression_l250_250479

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 :=
by
  sorry

end evaluate_expression_l250_250479


namespace max_product_condition_l250_250230

theorem max_product_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 12) (h3 : 0 ≤ y) (h4 : y ≤ 12) (h_eq : x * y = (12 - x) ^ 2 * (12 - y) ^ 2) : x * y ≤ 81 :=
sorry

end max_product_condition_l250_250230


namespace maximum_range_of_temperatures_l250_250911

variable (T1 T2 T3 T4 T5 : ℝ)

-- Given conditions
def average_condition : Prop := (T1 + T2 + T3 + T4 + T5) / 5 = 50
def lowest_temperature_condition : Prop := T1 = 45

-- Question to prove
def possible_maximum_range : Prop := T5 - T1 = 25

-- The final theorem statement
theorem maximum_range_of_temperatures 
  (h_avg : average_condition T1 T2 T3 T4 T5) 
  (h_lowest : lowest_temperature_condition T1) 
  : possible_maximum_range T1 T5 := by
  sorry

end maximum_range_of_temperatures_l250_250911


namespace perfect_squares_digit_4_5_6_l250_250662

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l250_250662


namespace cost_per_liter_l250_250514

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

end cost_per_liter_l250_250514


namespace area_of_triangle_ACD_l250_250322

theorem area_of_triangle_ACD (p : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : y1^2 = 2 * p * x1)
  (h2 : y2^2 = 2 * p * x2)
  (h3 : y1 + y2 = 4 * p)
  (h4 : y2 - y1 = p)
  (h5 : 2 * y1 + 2 * y2 = 8 * p^2 / (x2 - x1))
  (h6 : x2 - x1 = 2 * p)
  (h7 : 8 * p^2 = (y1 + y2) * 2 * p) :
  1 / 2 * (y1 * (x1 - (x2 + x1) / 2) + y2 * (x2 - (x2 + x1) / 2)) = 15 / 2 * p^2 :=
by
  sorry

end area_of_triangle_ACD_l250_250322


namespace quadratic_vertex_coords_l250_250881

theorem quadratic_vertex_coords :
  ∀ x : ℝ, (y = (x-2)^2 - 1) → (2, -1) = (2, -1) :=
by
  sorry

end quadratic_vertex_coords_l250_250881


namespace middle_number_of_consecutive_squares_l250_250179

theorem middle_number_of_consecutive_squares (x : ℕ ) (h : x^2 + (x+1)^2 + (x+2)^2 = 2030) : x + 1 = 26 :=
sorry

end middle_number_of_consecutive_squares_l250_250179


namespace inequality_holds_l250_250860

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 :=
by
  sorry

end inequality_holds_l250_250860


namespace smallest_multiple_of_45_and_60_not_divisible_by_18_l250_250313

noncomputable def smallest_multiple_not_18 (n : ℕ) : Prop :=
  (n % 45 = 0) ∧
  (n % 60 = 0) ∧
  (n % 18 ≠ 0) ∧
  ∀ m : ℕ, (m % 45 = 0) ∧ (m % 60 = 0) ∧ (m % 18 ≠ 0) → n ≤ m

theorem smallest_multiple_of_45_and_60_not_divisible_by_18 : ∃ n : ℕ, smallest_multiple_not_18 n ∧ n = 810 := 
by
  existsi 810
  sorry

end smallest_multiple_of_45_and_60_not_divisible_by_18_l250_250313


namespace leftovers_value_l250_250333

def quarters_in_roll : ℕ := 30
def dimes_in_roll : ℕ := 40
def james_quarters : ℕ := 77
def james_dimes : ℕ := 138
def lindsay_quarters : ℕ := 112
def lindsay_dimes : ℕ := 244
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftovers_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_in_roll
  let leftover_dimes := total_dimes % dimes_in_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 2.45 :=
by
  sorry

end leftovers_value_l250_250333


namespace find_face_value_l250_250297

-- Define the conditions as variables in Lean
variable (BD TD FV : ℝ)
variable (hBD : BD = 36)
variable (hTD : TD = 30)
variable (hRel : BD = TD + (TD * BD / FV))

-- State the theorem we want to prove
theorem find_face_value (BD TD : ℝ) (FV : ℝ) 
  (hBD : BD = 36) (hTD : TD = 30) (hRel : BD = TD + (TD * BD / FV)) : 
  FV = 180 := 
  sorry

end find_face_value_l250_250297


namespace complement_union_correct_l250_250714

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_correct :
  ((U \ A) ∪ B) = {0, 2, 3, 6} :=
by
  sorry

end complement_union_correct_l250_250714


namespace supplement_of_complementary_angle_l250_250081

theorem supplement_of_complementary_angle (α β : ℝ) 
  (h1 : α + β = 90) (h2 : α = 30) : 180 - β = 120 :=
by sorry

end supplement_of_complementary_angle_l250_250081


namespace max_value_x_y3_z4_l250_250859

theorem max_value_x_y3_z4 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  x + y^3 + z^4 ≤ 2 :=
by
  sorry

end max_value_x_y3_z4_l250_250859


namespace unique_positive_solution_eq_15_l250_250631

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l250_250631


namespace part1_l250_250742

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l250_250742


namespace merchant_marked_price_l250_250454

theorem merchant_marked_price (L : ℝ) (x : ℝ) : 
  (L = 100) →
  (L - 0.3 * L = 70) →
  (0.75 * x - 70 = 0.225 * x) →
  x = 133.33 :=
by
  intro h1 h2 h3
  sorry

end merchant_marked_price_l250_250454


namespace hockey_season_length_l250_250720

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l250_250720


namespace largest_element_lg11_l250_250968

variable (x y : ℝ)
variable (A : Set ℝ)  (B : Set ℝ)

-- Conditions
def condition1 : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)) := sorry
def condition2 : B = Set.insert 0 (Set.insert 1 ∅) := sorry
def condition3 : B ⊆ A := sorry

-- Statement
theorem largest_element_lg11 (x y : ℝ)

  (Aeq : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)))
  (Beq : B = Set.insert 0 (Set.insert 1 ∅))
  (subset : B ⊆ A) :
  ∃ M ∈ A, ∀ a ∈ A, a ≤ M ∧ M = Real.log 11 :=
sorry

end largest_element_lg11_l250_250968


namespace largest_lcm_l250_250429

theorem largest_lcm :
  max (max (max (max (max (Nat.lcm 12 2) (Nat.lcm 12 4)) 
                    (Nat.lcm 12 6)) 
                 (Nat.lcm 12 8)) 
            (Nat.lcm 12 10)) 
      (Nat.lcm 12 12) = 60 :=
by sorry

end largest_lcm_l250_250429


namespace find_m_and_e_l250_250858

theorem find_m_and_e (m e : ℕ) (hm : 0 < m) (he : e < 10) 
(h1 : 4 * m^2 + m + e = 346) 
(h2 : 4 * m^2 + m + 6 = 442 + 7 * e) : 
  m + e = 22 := by
  sorry

end find_m_and_e_l250_250858


namespace simon_age_is_10_l250_250931

-- Define the conditions
def alvin_age := 30
def half_alvin_age := alvin_age / 2
def simon_age := half_alvin_age - 5

-- State the theorem
theorem simon_age_is_10 : simon_age = 10 :=
by
  sorry

end simon_age_is_10_l250_250931


namespace B_knit_time_l250_250036

theorem B_knit_time (x : ℕ) (hA : 3 > 0) (h_combined_rate : 1/3 + 1/x = 1/2) : x = 6 := sorry

end B_knit_time_l250_250036


namespace largest_x_exists_largest_x_largest_real_number_l250_250785

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l250_250785


namespace max_int_value_of_a_real_roots_l250_250643

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l250_250643


namespace worker_allocation_correct_l250_250932

variable (x y : ℕ)
variable (H1 : x + y = 50)
variable (H2 : x = 30)
variable (H3 : y = 20)
variable (H4 : 120 * (50 - x) = 2 * 40 * x)

theorem worker_allocation_correct 
  (h₁ : x = 30) 
  (h₂ : y = 20) 
  (h₃ : x + y = 50) 
  (h₄ : 120 * (50 - x) = 2 * 40 * x) 
  : true := 
by
  sorry

end worker_allocation_correct_l250_250932


namespace number_of_ways_to_select_officers_l250_250715

-- Definitions based on conditions
def boys : ℕ := 6
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def officers_to_select : ℕ := 3

-- Number of ways to choose 3 individuals out of 10
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_choices : ℕ := choose total_people officers_to_select

-- Number of ways to choose 3 boys out of 6 (0 girls)
def all_boys_choices : ℕ := choose boys officers_to_select

-- Number of ways to choose at least 1 girl
def at_least_one_girl_choices : ℕ := total_choices - all_boys_choices

-- Theorem to prove the number of ways to select the officers
theorem number_of_ways_to_select_officers :
  at_least_one_girl_choices = 100 := by
  sorry

end number_of_ways_to_select_officers_l250_250715


namespace value_of_expression_l250_250423

theorem value_of_expression : 1 + 3^2 = 10 :=
by
  sorry

end value_of_expression_l250_250423


namespace inequality_proof_l250_250233

theorem inequality_proof (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n):
  x^n / (1 + x^2) + y^n / (1 + y^2) ≤ (x^n + y^n) / (1 + x * y) :=
by
  sorry

end inequality_proof_l250_250233


namespace factor_of_polynomial_l250_250402

theorem factor_of_polynomial (x : ℝ) : 
  (x^2 - 2*x + 2) ∣ (29 * 39 * x^4 + 4) :=
sorry

end factor_of_polynomial_l250_250402


namespace first_recipe_cups_l250_250154

-- Definitions based on the given conditions
def ounces_per_bottle : ℕ := 16
def ounces_per_cup : ℕ := 8
def cups_second_recipe : ℕ := 1
def cups_third_recipe : ℕ := 3
def total_bottles : ℕ := 3
def total_ounces : ℕ := total_bottles * ounces_per_bottle
def total_cups_needed : ℕ := total_ounces / ounces_per_cup

-- Proving the amount of cups of soy sauce needed for the first recipe
theorem first_recipe_cups : 
  total_cups_needed - (cups_second_recipe + cups_third_recipe) = 2 
:= by 
-- Proof omitted
  sorry

end first_recipe_cups_l250_250154


namespace min_groups_l250_250639

open Set

structure TwinSiblings (α : Type) := 
  (pairs : Set (Set α)) (members : ∀ s ∈ pairs, s.card = 2)

structure GroupActivities (α : Type) :=
  (groups : Set (Set α))
  (no_twins_same_group : ∀ g ∈ groups, ∀ pair ∈ TwinSiblings.pairs α, pair ∩ g ≠ pair)
  (each_non_twin_pair_once : ∀ g1 g2 ∈ groups, g1 ≠ g2 → ∀ x y, x ≠ y → x ∈ g1 ∩ g2 → y ∈ g1 ∩ g2 → False)
  (one_person_two_groups : ∃ x, (groups.filter (λ g, x ∈ g)).card = 2)

def problem_instance (α : Type) [Fintype α] :=
  (TwinSiblings α) × (GroupActivities α)

theorem min_groups {α : Type} [Fintype α] : 
  ∃ k, k = 14 → 
  ∃ inst : problem_instance α, 
  inst.2.groups.card = k := 
sorry

end min_groups_l250_250639


namespace binomial_30_3_l250_250205

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l250_250205


namespace unique_positive_solution_eq_15_l250_250628

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l250_250628


namespace correct_operation_l250_250904

theorem correct_operation : ∀ (a b : ℤ), 3 * a^2 * b - 2 * b * a^2 = a^2 * b :=
by
  sorry

end correct_operation_l250_250904


namespace expected_value_of_bernoulli_l250_250862

noncomputable def P (p : ℝ) (k : ℕ) : ℝ :=
if k = 0 then (1 - p) else if k = 1 then p else 0

def bernoulli_distribution_condition (X : ℕ → Prop) (p : ℝ) : Prop :=
(∀ k, X k ↔ k = 0 ∨ k = 1) ∧ ∀ (k : ℕ), X k → P p k ∈ set.Ioo 0 1

theorem expected_value_of_bernoulli (X : ℕ → Prop) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : bernoulli_distribution_condition X p) :
  ∑ k in {0, 1}, P p k * k = p :=
sorry

end expected_value_of_bernoulli_l250_250862


namespace oshea_bought_basil_seeds_l250_250403

-- Define the number of large and small planters and their capacities.
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30
def seeds_per_small_planter := 4

-- The theorem statement: Oshea bought 200 basil seeds
theorem oshea_bought_basil_seeds :
  large_planters * seeds_per_large_planter + small_planters * seeds_per_small_planter = 200 :=
by sorry

end oshea_bought_basil_seeds_l250_250403


namespace find_alpha_l250_250824

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end find_alpha_l250_250824


namespace find_f_minus1_plus_f_2_l250_250498

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin := ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

def f_value_at_zero := f 0 = 1

theorem find_f_minus1_plus_f_2 :
  even_function f →
  symmetric_about_origin f →
  f_value_at_zero f →
  f (-1) + f 2 = -1 :=
by
  intros
  sorry

end find_f_minus1_plus_f_2_l250_250498


namespace bus_capacity_percentage_l250_250170

theorem bus_capacity_percentage (x : ℕ) (h1 : 150 * x / 100 + 150 * 70 / 100 = 195) : x = 60 :=
by
  sorry

end bus_capacity_percentage_l250_250170


namespace derek_age_l250_250940

theorem derek_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end derek_age_l250_250940


namespace value_of_m_l250_250395

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l250_250395


namespace binomial_30_3_l250_250206

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l250_250206


namespace total_students_l250_250617

/-- Definition of the problem's conditions as Lean statements -/
def left_col := 8
def right_col := 14
def front_row := 7
def back_row := 15

/-- The total number of columns calculated from Eunji's column positions -/
def total_columns := left_col + right_col - 1
/-- The total number of rows calculated from Eunji's row positions -/
def total_rows := front_row + back_row - 1

/-- Lean statement showing the total number of students given the conditions -/
theorem total_students : total_columns * total_rows = 441 := by
  sorry

end total_students_l250_250617


namespace P_geq1_l250_250369

-- Defining the conditions
constant ξ : ℝ → ℝ
constant σ : ℝ
axiom normal_dist_ξ : ∀ x : ℝ, ξ x = (1 / (√(2 * π * (σ^2)))) * (exp (-((x + 1)^2) / (2 * σ^2)))
axiom P_neg3_to_neg1 : ∫ (x : ℝ) in (-3 : ℝ)..(-1 : ℝ), ξ x = 0.4

-- The proof problem
theorem P_geq1 : ∫ (x : ℝ) in (1 : ℝ)..(∞ : ℝ), ξ x = 0.1 := sorry

end P_geq1_l250_250369


namespace no_valid_arrangement_in_7x7_grid_l250_250853

theorem no_valid_arrangement_in_7x7_grid :
  ¬ (∃ (f : Fin 7 → Fin 7 → ℕ),
    (∀ (i j : Fin 6),
      (f i j + f i (j + 1) + f (i + 1) j + f (i + 1) (j + 1)) % 2 = 1) ∧
    (∀ (i j : Fin 5),
      (f i j + f i (j + 1) + f i (j + 2) + f (i + 1) j + f (i + 1) (j + 1) + f (i + 1) (j + 2) +
       f (i + 2) j + f (i + 2) (j + 1) + f (i + 2) (j + 2)) % 2 = 1)) := by
  sorry

end no_valid_arrangement_in_7x7_grid_l250_250853


namespace cook_weave_l250_250596

theorem cook_weave (Y C W OC CY CYW : ℕ) (hY : Y = 25) (hC : C = 15) (hW : W = 8) (hOC : OC = 2)
  (hCY : CY = 7) (hCYW : CYW = 3) : 
  ∃ (CW : ℕ), CW = 9 :=
by 
  have CW : ℕ := C - OC - (CY - CYW) 
  use CW
  sorry

end cook_weave_l250_250596


namespace temperature_difference_l250_250730

/-- The average temperature at the top of Mount Tai. -/
def T_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai. -/
def T_foot : ℝ := -1

/-- The temperature difference between the average temperature at the foot and the top of Mount Tai is 8 degrees Celsius. -/
theorem temperature_difference : T_foot - T_top = 8 := by
  sorry

end temperature_difference_l250_250730


namespace actual_average_height_correct_l250_250320

theorem actual_average_height_correct : 
  (∃ (avg_height : ℚ), avg_height = 181 ) →
  (∃ (num_boys : ℕ), num_boys = 35) →
  (∃ (incorrect_height : ℚ), incorrect_height = 166) →
  (∃ (actual_height : ℚ), actual_height = 106) →
  (179.29 : ℚ) = 
    (round ((6315 + 106 : ℚ) / 35 * 100) / 100 ) :=
by
sorry

end actual_average_height_correct_l250_250320


namespace probability_blue_face_up_l250_250727

-- Definitions of the conditions
def dodecahedron_faces : ℕ := 12
def blue_faces : ℕ := 10
def red_faces : ℕ := 2

-- Expected probability
def probability_blue_face : ℚ := 5 / 6

-- Theorem to prove the probability of rolling a blue face on a dodecahedron
theorem probability_blue_face_up (total_faces blue_count red_count : ℕ)
    (h1 : total_faces = dodecahedron_faces)
    (h2 : blue_count = blue_faces)
    (h3 : red_count = red_faces) :
  blue_count / total_faces = probability_blue_face :=
by sorry

end probability_blue_face_up_l250_250727


namespace distance_walked_east_l250_250875

-- Definitions for distances
def s1 : ℕ := 25   -- distance walked south
def s2 : ℕ := 20   -- distance walked east
def s3 : ℕ := 25   -- distance walked north
def final_distance : ℕ := 35   -- final distance from the starting point

-- Proof problem: Prove that the distance walked east in the final step is as expected
theorem distance_walked_east (d : Real) :
  d = Real.sqrt (final_distance ^ 2 - s2 ^ 2) :=
sorry

end distance_walked_east_l250_250875


namespace find_c_l250_250610

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 3 - 2 * x - 6 * x^3 + 7 * x^4

-- Define the main theorem stating that c = -5/7 makes f(x) + c*g(x) have degree 3
theorem find_c (c : ℝ) (h : ∀ x : ℝ, f x + c * g x = 0) : c = -5 / 7 := by
  sorry

end find_c_l250_250610


namespace unique_positive_solution_l250_250633

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l250_250633


namespace goat_can_circle_around_tree_l250_250603

/-- 
  Given a goat tied with a rope of length 4.7 meters (L) near an old tree with a cylindrical trunk of radius 0.5 meters (R), 
  with the shortest distance from the stake to the surface of the tree being 1 meter (d), 
  prove that the minimal required rope length to encircle the tree and return to the stake is less than 
  or equal to the given rope length of 4.7 meters (L).
-/ 
theorem goat_can_circle_around_tree (L R d : ℝ) (hR : R = 0.5) (hd : d = 1) (hL : L = 4.7) : 
  ∃ L_min, L_min ≤ L := 
by
  -- Detailed proof steps omitted.
  sorry

end goat_can_circle_around_tree_l250_250603


namespace range_of_g_l250_250812

-- Define the function g
def g (x : ℝ) := (Real.arcsin (x / 3))^2 - π * Real.arccos (x / 3) +
                  (Real.arccos (x / 3))^2 + (π^2 / 8) * (x^2 - 4 * x + 3)

-- State the theorem about the range of g over the interval [-3, 3]
theorem range_of_g : 
  (∀ x, -3 ≤ x ∧ x ≤ 3 → 
  (g x) ≥ (π^2 / 4) ∧ (g x) ≤ (33 * π^2 / 8)) := 
sorry

end range_of_g_l250_250812


namespace cone_base_circumference_l250_250190

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ)
  (volume_eq : V = 18 * Real.pi)
  (height_eq : h = 3) :
  C = 6 * Real.sqrt 2 * Real.pi :=
sorry

end cone_base_circumference_l250_250190


namespace find_m_range_l250_250496

noncomputable def proposition_p (x : ℝ) : Prop := (-2 : ℝ) ≤ x ∧ x ≤ 10
noncomputable def proposition_q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m)

theorem find_m_range (m : ℝ) (h : m > 0) : (¬ ∃ x : ℝ, proposition_p x) → (¬ ∃ x : ℝ, proposition_q x m) → (¬ (¬ (¬ ∃ x : ℝ, proposition_q x m)) → ¬ (¬ ∃ x : ℝ, proposition_p x)) → m ≥ 9 := 
sorry

end find_m_range_l250_250496


namespace mohan_cookies_l250_250541

theorem mohan_cookies :
  ∃ a : ℕ, 
    a % 4 = 3 ∧
    a % 5 = 2 ∧
    a % 7 = 4 ∧
    a = 67 :=
by
  -- The proof will be written here.
  sorry

end mohan_cookies_l250_250541


namespace product_of_primes_l250_250938

theorem product_of_primes :
  let p1 := 11
  let p2 := 13
  let p3 := 997
  p1 * p2 * p3 = 142571 :=
by
  sorry

end product_of_primes_l250_250938


namespace work_done_by_forces_l250_250085

-- Definitions of given forces and displacement
noncomputable def F1 : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F2 : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Statement of the theorem
theorem work_done_by_forces :
  let F := (F1.1 + F2.1, F1.2 + F2.2)
  let W := F.1 * S.1 + F.2 * S.2
  W = 2 :=
by
  sorry

end work_done_by_forces_l250_250085


namespace factorization_correct_l250_250315

theorem factorization_correct (a x y : ℝ) : a * x - a * y = a * (x - y) := by sorry

end factorization_correct_l250_250315


namespace monotonic_decreasing_interval_range_of_a_l250_250362

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) * ((a / x) + a + 1)

theorem monotonic_decreasing_interval (a : ℝ) (h : a ≥ -1) :
  (a = -1 → ∀ x, x < -1 → f a x < f a (x + 1)) ∧
  (a ≠ -1 → (∀ x, -1 < a ∧ x < -1 ∨ x > 1 / (a + 1) → f a x < f a (x + 1)) ∧
                (∀ x, -1 < a ∧ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 / (a + 1) → f a x < f a (x + 1)))
:= sorry

theorem range_of_a (a : ℝ) (h : a ≥ -1) :
  (∃ x1 x2, x1 > 0 ∧ x2 < 0 ∧ f a x1 < f a x2 → -1 ≤ a ∧ a < 0)
:= sorry

end monotonic_decreasing_interval_range_of_a_l250_250362


namespace value_of_y_l250_250840

variable (x y : ℤ)

-- Define the conditions
def condition1 : Prop := 3 * (x^2 + x + 1) = y - 6
def condition2 : Prop := x = -3

-- Theorem to prove
theorem value_of_y (h1 : condition1 x y) (h2 : condition2 x) : y = 27 := by
  sorry

end value_of_y_l250_250840


namespace solution_set_line_l250_250126

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l250_250126


namespace total_blossoms_l250_250540

theorem total_blossoms (first second third : ℕ) (h1 : first = 2) (h2 : second = 2 * first) (h3 : third = 4 * second) : first + second + third = 22 :=
by
  sorry

end total_blossoms_l250_250540


namespace binom_computation_l250_250209

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l250_250209


namespace sequence_problems_l250_250267
open Nat

-- Define the arithmetic sequence conditions
def arith_seq_condition_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 7 = -23

def arith_seq_condition_2 (a : ℕ → ℤ) : Prop :=
  a 3 + a 8 = -29

-- Define the geometric sequence condition
def geom_seq_condition (a b : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n, a n + b n = c^(n - 1)

-- Define the arithmetic sequence formula
def arith_seq_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = -3 * n + 2

-- Define the sum of the first n terms of the sequence b_n
def sum_b_n (b : ℕ → ℤ) (S_n : ℕ → ℤ) (c : ℤ) : Prop :=
  (c = 1 → ∀ n, S_n n = (3 * n^2 + n) / 2) ∧
  (c ≠ 1 → ∀ n, S_n n = (n * (3 * n - 1)) / 2 + ((1 - c^n) / (1 - c)))

-- Define the main theorem
theorem sequence_problems (a b : ℕ → ℤ) (c : ℤ) (S_n : ℕ → ℤ) :
  arith_seq_condition_1 a →
  arith_seq_condition_2 a →
  geom_seq_condition a b c →
  arith_seq_formula a ∧ sum_b_n b S_n c :=
by
  -- Proofs for the conditions to the formula
  sorry

end sequence_problems_l250_250267


namespace det_commutator_zero_l250_250385

open Matrix

variable {n : Type} [Fintype n] [DecidableEq n]

theorem det_commutator_zero (n : ℕ) (A B : Matrix n n ℂ)
  (h_odd : n % 2 = 1) (h_idempotent : (A - B) * (A - B) = 0) :
  det (A * B - B * A) = 0 := by
  sorry

end det_commutator_zero_l250_250385


namespace largest_x_l250_250778

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l250_250778


namespace commute_distance_l250_250928

theorem commute_distance (D : ℝ)
  (h1 : ∀ t : ℝ, t > 0 → t = D / 45)
  (h2 : ∀ t : ℝ, t > 0 → t = D / 30)
  (h3 : D / 45 + D / 30 = 1) :
  D = 18 :=
by
  sorry

end commute_distance_l250_250928


namespace expected_value_full_circles_l250_250754

-- Definition of the conditions
def num_small_triangles (n : ℕ) : ℕ :=
  n^2

def potential_full_circle_vertices (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

def prob_full_circle : ℚ :=
  1 / 729

-- The expected number of full circles formed
def expected_full_circles (n : ℕ) : ℚ :=
  potential_full_circle_vertices n * prob_full_circle

-- The mathematical equivalence to be proved
theorem expected_value_full_circles (n : ℕ) : expected_full_circles n = (n - 2) * (n - 1) / 1458 := 
  sorry

end expected_value_full_circles_l250_250754


namespace probability_sum_9_is_correct_l250_250314

def num_faces : ℕ := 6

def possible_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ := 4  -- (3,6), (6,3), (4,5), (5,4)

def probability_sum_9 : ℚ := favorable_outcomes / possible_outcomes

theorem probability_sum_9_is_correct :
  probability_sum_9 = 1/9 :=
sorry

end probability_sum_9_is_correct_l250_250314


namespace arithmetic_geometric_seq_l250_250712

open Real

theorem arithmetic_geometric_seq (a d : ℝ) (h₀ : d ≠ 0) 
  (h₁ : (a + d) * (a + 5 * d) = (a + 2 * d) ^ 2) : 
  (a + 2 * d) / (a + d) = 3 :=
sorry

end arithmetic_geometric_seq_l250_250712


namespace total_amount_shared_l250_250845

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.20 * z) (hz : z = 100) :
  x + y + z = 370 := by
  sorry

end total_amount_shared_l250_250845


namespace smallest_number_of_students_l250_250935

theorem smallest_number_of_students
  (tenth_graders eighth_graders ninth_graders : ℕ)
  (ratio1 : 7 * eighth_graders = 4 * tenth_graders)
  (ratio2 : 9 * ninth_graders = 5 * tenth_graders) :
  (∀ n, (∃ a b c, a = 7 * b ∧ b = 4 * n ∧ a = 9 * c ∧ c = 5 * n) → n = 134) :=
by {
  -- We currently just assume the result for Lean to be syntactically correct
  sorry
}

end smallest_number_of_students_l250_250935


namespace inequality_areas_l250_250892

theorem inequality_areas (a b c α β γ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  a / α + b / β + c / γ ≥ 3 / 2 :=
by
  -- Insert the AM-GM inequality application and simplifications
  sorry

end inequality_areas_l250_250892


namespace solution_of_system_l250_250135

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l250_250135


namespace inequality_count_l250_250400

theorem inequality_count
  (x y a b : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hx_lt_one : x < 1)
  (hy_lt_one : y < 1)
  (hx_lt_a : x < a)
  (hy_lt_b : y < b)
  (h_sum : x + y = a - b) :
  ({(x + y < a + b), (x - y < a - b), (x * y < a * b)}:Finset Prop).card = 3 :=
by
  sorry

end inequality_count_l250_250400


namespace solution_set_of_inequality_l250_250568

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x ≥ 0} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by
  sorry

end solution_set_of_inequality_l250_250568


namespace x_value_for_divisibility_l250_250229

theorem x_value_for_divisibility (x : ℕ) (h1 : x = 0 ∨ x = 5) (h2 : (8 * 10 + x) % 4 = 0) : x = 0 :=
by
  sorry

end x_value_for_divisibility_l250_250229


namespace find_a_l250_250091

theorem find_a (a : ℝ) (h₁ : ¬ (a = 0)) (h_perp : (∀ x y : ℝ, (a * x + 1 = 0) 
  -> (a - 2) * x + y + a = 0 -> ∀ x₁ y₁, (a * x₁ + 1 = 0) -> y = y₁)) : a = 2 := 
by 
  sorry

end find_a_l250_250091


namespace number_of_nickels_is_3_l250_250178

-- Defining the problem conditions
def total_coins := 8
def total_value := 53 -- in cents
def at_least_one_penny := 1
def at_least_one_nickel := 1
def at_least_one_dime := 1

-- Stating the proof problem
theorem number_of_nickels_is_3 : ∃ (pennies nickels dimes : Nat), 
  pennies + nickels + dimes = total_coins ∧ 
  pennies ≥ at_least_one_penny ∧ 
  nickels ≥ at_least_one_nickel ∧ 
  dimes ≥ at_least_one_dime ∧ 
  pennies + 5 * nickels + 10 * dimes = total_value ∧ 
  nickels = 3 := sorry

end number_of_nickels_is_3_l250_250178


namespace a4_value_l250_250821

axiom a_n : ℕ → ℝ
axiom S_n : ℕ → ℝ
axiom q : ℝ

-- Conditions
axiom a1_eq_1 : a_n 1 = 1
axiom S6_eq_4S3 : S_n 6 = 4 * S_n 3
axiom q_ne_1 : q ≠ 1

-- Arithmetic Sequence Sum Formula
axiom sum_formula : ∀ n, S_n n = (1 - q^n) / (1 - q)

-- nth-term Formula
axiom nth_term_formula : ∀ n, a_n n = a_n 1 * q^(n - 1)

-- Prove the value of the 4th term
theorem a4_value : a_n 4 = 3 := sorry

end a4_value_l250_250821


namespace simplify_fraction_multiplication_l250_250148

theorem simplify_fraction_multiplication :
  (15/35) * (28/45) * (75/28) = 5/7 :=
by
  sorry

end simplify_fraction_multiplication_l250_250148


namespace chest_contents_l250_250524

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l250_250524


namespace complex_number_simplification_l250_250558

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : i * (1 - i) - 1 = i := 
by
  sorry

end complex_number_simplification_l250_250558


namespace max_correct_answers_l250_250176

variable (x y z : ℕ)

theorem max_correct_answers
  (h1 : x + y + z = 100)
  (h2 : x - 3 * y - 2 * z = 50) :
  x ≤ 87 := by
    sorry

end max_correct_answers_l250_250176


namespace geometric_sequence_sum_l250_250515

-- Define the positive terms of the geometric sequence
variables {a_1 a_2 a_3 a_4 a_5 : ℝ}
-- Assume all terms are positive
variables (h1 : a_1 > 0) (h2 : a_2 > 0) (h3 : a_3 > 0) (h4 : a_4 > 0) (h5 : a_5 > 0)

-- Main condition given in the problem
variable (h_main : a_1 * a_3 + 2 * a_2 * a_4 + a_3 * a_5 = 16)

-- Goal: Prove that a_2 + a_4 = 4
theorem geometric_sequence_sum : a_2 + a_4 = 4 :=
by
  sorry

end geometric_sequence_sum_l250_250515


namespace probability_one_white_one_black_l250_250916

def white_ball_count : ℕ := 8
def black_ball_count : ℕ := 7
def total_ball_count : ℕ := white_ball_count + black_ball_count
def total_ways_to_choose_2_balls : ℕ := total_ball_count.choose 2
def favorable_ways : ℕ := white_ball_count * black_ball_count

theorem probability_one_white_one_black : 
  (favorable_ways : ℚ) / (total_ways_to_choose_2_balls : ℚ) = 8 / 15 :=
by
  sorry

end probability_one_white_one_black_l250_250916


namespace solve_x_l250_250584

theorem solve_x (x : ℝ) (h : 9 - 4 / x = 7 + 8 / x) : x = 6 := 
by 
  sorry

end solve_x_l250_250584


namespace ratio_a_b_l250_250665

theorem ratio_a_b (a b c d : ℝ) 
  (h1 : b / c = 7 / 9) 
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) : 
  a / b = 3 / 4 :=
  sorry

end ratio_a_b_l250_250665


namespace original_number_l250_250344

theorem original_number (x : ℤ) (h : (x - 5) / 4 = (x - 4) / 5) : x = 9 :=
sorry

end original_number_l250_250344


namespace smallest_n_for_square_and_cube_l250_250311

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end smallest_n_for_square_and_cube_l250_250311


namespace dividends_CEO_2018_l250_250058

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l250_250058


namespace not_divisible_67_l250_250120

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l250_250120


namespace line_equation_intersections_l250_250774

theorem line_equation_intersections (m b k : ℝ) (h1 : b ≠ 0) 
  (h2 : m * 2 + b = 7) (h3 : abs (k^2 + 8*k + 7 - (m*k + b)) = 4) :
  m = 6 ∧ b = -5 :=
by {
  sorry
}

end line_equation_intersections_l250_250774


namespace sum_of_h_and_k_l250_250052

theorem sum_of_h_and_k (foci1 foci2 : ℝ × ℝ) (pt : ℝ × ℝ) (a b h k : ℝ) 
  (h_positive : a > 0) (b_positive : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = if (x, y) = pt then 1 else sorry)
  (foci_eq : foci1 = (1, 2) ∧ foci2 = (4, 2))
  (pt_eq : pt = (-1, 5)) :
  h + k = 4.5 :=
sorry

end sum_of_h_and_k_l250_250052


namespace chest_contents_solution_l250_250523

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l250_250523


namespace ratio_largest_smallest_root_geometric_progression_l250_250887

theorem ratio_largest_smallest_root_geometric_progression (a b c d : ℤ)
  (h_poly : a * x^3 + b * x^2 + c * x + d = 0) 
  (h_in_geo_prog : ∃ r1 r2 r3 q : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 * q = r2 ∧ r2 * q = r3 ∧ q ≠ 0) : 
  ∃ R : ℝ, R = 1 := 
by
  sorry

end ratio_largest_smallest_root_geometric_progression_l250_250887


namespace number_of_new_players_l250_250711

variable (returning_players : ℕ)
variable (groups : ℕ)
variable (players_per_group : ℕ)

theorem number_of_new_players
  (h1 : returning_players = 6)
  (h2 : groups = 9)
  (h3 : players_per_group = 6) :
  (groups * players_per_group - returning_players = 48) := 
sorry

end number_of_new_players_l250_250711


namespace integral_x_minus_reciprocal_l250_250474

theorem integral_x_minus_reciprocal :
  ∫ x in (1:ℝ)..(Real.exp 1), (x - (1 / x)) = (Real.exp 2 - 3) / 2 :=
by
  sorry

end integral_x_minus_reciprocal_l250_250474


namespace alice_bob_speed_l250_250193

theorem alice_bob_speed (x : ℝ) (h : x = 3 + 2 * Real.sqrt 7) :
  x^2 - 5 * x - 14 = 8 + 2 * Real.sqrt 7 - 5 := by
sorry

end alice_bob_speed_l250_250193


namespace fruit_display_l250_250426

theorem fruit_display (bananas : ℕ) (Oranges : ℕ) (Apples : ℕ) (hBananas : bananas = 5)
  (hOranges : Oranges = 2 * bananas) (hApples : Apples = 2 * Oranges) :
  bananas + Oranges + Apples = 35 :=
by sorry

end fruit_display_l250_250426


namespace senior_students_in_sample_l250_250750

theorem senior_students_in_sample 
  (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : total_seniors = 500)
  (h3 : sample_size = 200) : 
  (total_seniors * sample_size / total_students = 50) :=
by {
  sorry
}

end senior_students_in_sample_l250_250750


namespace correct_option_l250_250902

def condition_A : Prop := abs ((-5 : ℤ)^2) = -5
def condition_B : Prop := abs (9 : ℤ) = 3 ∨ abs (9 : ℤ) = -3
def condition_C : Prop := abs (3 : ℤ) / abs (((-2)^3 : ℤ)) = -2
def condition_D : Prop := (2 * abs (3 : ℤ))^2 = 6 

theorem correct_option : ¬condition_A ∧ ¬condition_B ∧ condition_C ∧ ¬condition_D :=
by
  sorry

end correct_option_l250_250902


namespace find_point_P_l250_250843

theorem find_point_P :
  ∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 0 ∧ 
  (P.2 = P.1^4 - P.1) ∧
  (∃ m, m = 4 * P.1^3 - 1 ∧ m = 3) :=
by
  sorry

end find_point_P_l250_250843


namespace sum_of_given_geom_series_l250_250762

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l250_250762


namespace sum_of_roots_of_quadratic_l250_250952

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, x^2 + 2000*x - 2000 = 0 ->
  (∃ x1 x2 : ℝ, (x1 ≠ x2 ∧ x1^2 + 2000*x1 - 2000 = 0 ∧ x2^2 + 2000*x2 - 2000 = 0 ∧ x1 + x2 = -2000)) :=
sorry

end sum_of_roots_of_quadratic_l250_250952


namespace trigonometric_identity_proof_l250_250223

theorem trigonometric_identity_proof :
  ( (Real.cos (40 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)))
  / (Real.sin (70 * Real.pi / 180) * Real.sqrt (1 + Real.cos (40 * Real.pi / 180))) ) =
  Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_proof_l250_250223


namespace range_a_condition_l250_250538

theorem range_a_condition (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ a → x^2 ≤ 2 * x + 3) ↔ (1 / 2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_a_condition_l250_250538


namespace largest_x_exists_largest_x_largest_real_number_l250_250782

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l250_250782


namespace solve_for_a_l250_250371

noncomputable def a_value (a x : ℝ) : Prop :=
  (3 / 10) * a + (2 * x + 4) / 2 = 4 * (x - 1)

theorem solve_for_a (a : ℝ) : a_value a 3 → a = 10 :=
by
  sorry

end solve_for_a_l250_250371


namespace least_positive_integer_l250_250576

theorem least_positive_integer (n : ℕ) (h1 : n > 1) 
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 5 = 1) 
  (h5 : n % 7 = 1) (h6 : n % 11 = 1): 
  n = 2311 := 
by
  sorry

end least_positive_integer_l250_250576


namespace xiaoming_climb_stairs_five_steps_l250_250316

def count_ways_to_climb (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else count_ways_to_climb (n - 1) + count_ways_to_climb (n - 2)

theorem xiaoming_climb_stairs_five_steps :
  count_ways_to_climb 5 = 5 :=
by
  sorry

end xiaoming_climb_stairs_five_steps_l250_250316


namespace mark_gig_schedule_l250_250867

theorem mark_gig_schedule 
  (every_other_day : ∀ weeks, ∃ gigs, gigs = weeks * 7 / 2) 
  (songs_per_gig : 2 * 5 + 10 = 20) 
  (total_minutes : ∃ gigs, 280 = gigs * 20) : 
  ∃ weeks, weeks = 4 := 
by 
  sorry

end mark_gig_schedule_l250_250867


namespace nina_total_miles_l250_250399

noncomputable def totalDistance (warmUp firstHillUp firstHillDown firstRecovery 
                                 tempoRun secondHillUp secondHillDown secondRecovery 
                                 fartlek sprintsYards jogsBetweenSprints coolDown : ℝ) 
                                 (mileInYards : ℝ) : ℝ :=
  warmUp + 
  (firstHillUp + firstHillDown + firstRecovery) + 
  tempoRun + 
  (secondHillUp + secondHillDown + secondRecovery) + 
  fartlek + 
  (sprintsYards / mileInYards) + 
  jogsBetweenSprints + 
  coolDown

theorem nina_total_miles : 
  totalDistance 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 (8 * 50) (8 * 0.2) 0.3 1760 = 5.877 :=
by
  sorry

end nina_total_miles_l250_250399


namespace area_of_transformed_region_l250_250976

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![5, 3]]
def area_T : ℝ := 9

-- Theorem statement
theorem area_of_transformed_region : 
  let det_matrix := matrix.det
  (det_matrix = 9) → (area_T = 9) → (area_T * det_matrix = 81) :=
by
  intros h₁ h₂
  sorry

end area_of_transformed_region_l250_250976


namespace solution_set_unique_line_l250_250137

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l250_250137


namespace time_at_simple_interest_l250_250462

theorem time_at_simple_interest 
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * (R + 5) / 100) * T = (P * (R / 100) * T) + 150) : 
  T = 10 := 
by 
  -- Proof is omitted.
  sorry

end time_at_simple_interest_l250_250462


namespace karen_piggy_bank_total_l250_250112

theorem karen_piggy_bank_total (a r n : ℕ) (h1 : a = 2) (h2 : r = 3) (h3 : n = 7) :
  (a * ((1 - r^n) / (1 - r))) = 2186 := by
  sorry

end karen_piggy_bank_total_l250_250112


namespace find_k_for_parallel_vectors_l250_250501

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem find_k_for_parallel_vectors 
  (h_a : a = (2, -1)) 
  (h_b : b = (1, 1)) 
  (h_c : c = (-5, 1)) 
  (h_parallel : vector_parallel (a.1 + k * b.1, a.2 + k * b.2) c) : 
  k = 1 / 2 :=
by
  unfold vector_parallel at h_parallel
  simp at h_parallel
  sorry

end find_k_for_parallel_vectors_l250_250501


namespace part1_l250_250741

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l250_250741


namespace survey_support_percentage_l250_250043

theorem survey_support_percentage 
  (num_men : ℕ) (percent_men_support : ℝ)
  (num_women : ℕ) (percent_women_support : ℝ)
  (h_men : num_men = 200)
  (h_percent_men_support : percent_men_support = 0.7)
  (h_women : num_women = 500)
  (h_percent_women_support : percent_women_support = 0.75) :
  (num_men * percent_men_support + num_women * percent_women_support) / (num_men + num_women) * 100 = 74 := 
by
  sorry

end survey_support_percentage_l250_250043


namespace calculate_truck_loads_of_dirt_l250_250332

noncomputable def truck_loads_sand: ℚ := 0.16666666666666666
noncomputable def truck_loads_cement: ℚ := 0.16666666666666666
noncomputable def total_truck_loads_material: ℚ := 0.6666666666666666
noncomputable def truck_loads_dirt: ℚ := total_truck_loads_material - (truck_loads_sand + truck_loads_cement)

theorem calculate_truck_loads_of_dirt :
  truck_loads_dirt = 0.3333333333333333 := 
by
  sorry

end calculate_truck_loads_of_dirt_l250_250332


namespace sum_digits_of_three_digit_numbers_l250_250713

theorem sum_digits_of_three_digit_numbers (a c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hc : 1 ≤ c ∧ c < 10) 
  (h1 : (300 + 10 * a + 7) + 414 = 700 + 10 * c + 1)
  (h2 : ∃ k : ℤ, 700 + 10 * c + 1 = 11 * k) :
  a + c = 14 :=
by
  sorry

end sum_digits_of_three_digit_numbers_l250_250713


namespace exp_gt_one_l250_250446

theorem exp_gt_one (a x y : ℝ) (ha : 1 < a) (hxy : x > y) : a^x > a^y :=
sorry

end exp_gt_one_l250_250446


namespace complement_of_M_is_34_l250_250166

open Set

noncomputable def U : Set ℝ := univ
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}
def complement_M (U : Set ℝ) (M : Set ℝ) : Set ℝ := U \ M

theorem complement_of_M_is_34 : complement_M U M = {x | 3 ≤ x ∧ x ≤ 4} := 
by sorry

end complement_of_M_is_34_l250_250166


namespace ones_digit_of_8_pow_47_l250_250025

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l250_250025


namespace distance_is_12_l250_250520

def distance_to_Mount_Overlook (D : ℝ) : Prop :=
  let T1 := D / 4
  let T2 := D / 6
  T1 + T2 = 5

theorem distance_is_12 : ∃ D : ℝ, distance_to_Mount_Overlook D ∧ D = 12 :=
by
  use 12
  rw [distance_to_Mount_Overlook]
  sorry

end distance_is_12_l250_250520


namespace decagon_diagonals_l250_250008

-- Definition of the number of diagonals in a polygon with n sides.
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The proof problem statement
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l250_250008


namespace log_sin_decrease_interval_l250_250482

open Real

noncomputable def interval_of_decrease (x : ℝ) : Prop :=
  ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8)

theorem log_sin_decrease_interval (x : ℝ) :
  interval_of_decrease x ↔ ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8) :=
by
  sorry

end log_sin_decrease_interval_l250_250482


namespace compare_fractions_compare_integers_l250_250064

-- First comparison: Prove -4/7 > -2/3
theorem compare_fractions : - (4 : ℚ) / 7 > - (2 : ℚ) / 3 := 
by sorry

-- Second comparison: Prove -(-7) > -| -7 |
theorem compare_integers : -(-7) > -abs (-7) := 
by sorry

end compare_fractions_compare_integers_l250_250064


namespace mean_weight_participants_l250_250569

def weights_120s := [123, 125]
def weights_130s := [130, 132, 133, 135, 137, 138]
def weights_140s := [141, 145, 145, 149, 149]
def weights_150s := [150, 152, 153, 155, 158]
def weights_160s := [164, 167, 167, 169]

def total_weights := weights_120s ++ weights_130s ++ weights_140s ++ weights_150s ++ weights_160s

def total_sum : ℕ := total_weights.sum
def total_count : ℕ := total_weights.length

theorem mean_weight_participants : (total_sum : ℚ) / total_count = 3217 / 22 := by
  sorry -- Proof goes here, but we're skipping it

end mean_weight_participants_l250_250569


namespace parabola_directrix_l250_250882

theorem parabola_directrix (y : ℝ) : (∃ p : ℝ, x = (1 / (4 * p)) * y^2 ∧ p = 2) → x = -2 :=
by
  sorry

end parabola_directrix_l250_250882


namespace largest_x_satisfies_condition_l250_250787

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l250_250787


namespace vertex_of_parabola_l250_250997

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, (x - 2)^2 + 1 = (x - h)^2 + k) ∧ (h = 2 ∧ k = 1) :=
by
  use 2, 1
  split
  · intro x
    ring
  · exact ⟨rfl, rfl⟩

end vertex_of_parabola_l250_250997


namespace total_cost_for_james_l250_250271

-- Prove that James will pay a total of $250 for his new pair of glasses.

theorem total_cost_for_james
  (frame_cost : ℕ := 200)
  (lens_cost : ℕ := 500)
  (insurance_cover_percentage : ℚ := 0.80)
  (coupon_on_frames : ℕ := 50) :
  (frame_cost - coupon_on_frames + lens_cost * (1 - insurance_cover_percentage)) = 250 :=
by
  -- Declare variables for the described values
  let total_frame_cost := frame_cost - coupon_on_frames
  let insurance_cover := lens_cost * insurance_cover_percentage
  let total_lens_cost := lens_cost - insurance_cover
  let total_cost := total_frame_cost + total_lens_cost

  -- We need to show total_cost = 250
  have h1 : total_frame_cost = 150 := by sorry
  have h2 : insurance_cover = 400 := by sorry
  have h3 : total_lens_cost = 100 := by sorry
  have h4 : total_cost = 250 := by
    rw [←h1, ←h3]
    sorry

  exact h4

end total_cost_for_james_l250_250271


namespace solution_set_l250_250660

noncomputable def truncated_interval (x : ℝ) (n : ℤ) : Prop :=
n ≤ x ∧ x < n + 1

theorem solution_set (x : ℝ) (hx : ∃ n : ℤ, n > 0 ∧ truncated_interval x n) :
  2 ≤ x ∧ x < 8 :=
sorry

end solution_set_l250_250660


namespace part1_part2_part3_l250_250594

noncomputable def p1_cost (t : ℕ) : ℕ := 
  if t <= 150 then 58 else 58 + 25 * (t - 150) / 100

noncomputable def p2_cost (t : ℕ) (a : ℕ) : ℕ := 
  if t <= 350 then 88 else 88 + a * (t - 350)

-- Part 1: Prove the costs for 260 minutes
theorem part1 : p1_cost 260 = 855 / 10 ∧ p2_cost 260 30 = 88 :=
by 
  sorry

-- Part 2: Prove the existence of t for given a
theorem part2 (t : ℕ) : (a = 30) → (∃ t, p1_cost t = p2_cost t a) :=
by 
  sorry

-- Part 3: Prove a=45 and the range for which Plan 1 is cheaper
theorem part3 : 
  (a = 45) ↔ (p1_cost 450 = p2_cost 450 a) ∧ (∀ t, (0 ≤ t ∧ t < 270) ∨ (t > 450) → p1_cost t < p2_cost t 45 ) :=
by 
  sorry

end part1_part2_part3_l250_250594


namespace field_area_l250_250049

-- Define the given conditions and prove the area of the field
theorem field_area (x y : ℕ) 
  (h1 : 2*(x + 20) + 2*y = 2*(2*x + 2*y))
  (h2 : 2*x + 2*(2*y) = 2*x + 2*y + 18) : x * y = 99 := by 
{
  sorry
}

end field_area_l250_250049


namespace unique_positive_real_solution_l250_250625

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l250_250625


namespace coefficient_of_y_squared_l250_250813

/-- Given the equation ay^2 - 8y + 55 = 59 and y = 2, prove that the coefficient a is 5. -/
theorem coefficient_of_y_squared (a y : ℝ) (h_y : y = 2) (h_eq : a * y^2 - 8 * y + 55 = 59) : a = 5 := by
  sorry

end coefficient_of_y_squared_l250_250813


namespace discount_percentage_l250_250602

/-
  A retailer buys 80 pens at the market price of 36 pens from a wholesaler.
  He sells these pens giving a certain discount and his profit is 120%.
  What is the discount percentage he gave on the pens?
-/
theorem discount_percentage
  (P : ℝ)
  (CP SP D DP : ℝ) 
  (h1 : CP = 36 * P)
  (h2 : SP = 2.2 * CP)
  (h3 : D = P - (SP / 80))
  (h4 : DP = (D / P) * 100) :
  DP = 1 := 
sorry

end discount_percentage_l250_250602


namespace mr_mcpherson_needs_to_raise_840_l250_250117

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end mr_mcpherson_needs_to_raise_840_l250_250117


namespace fewest_people_to_join_CBL_l250_250424

theorem fewest_people_to_join_CBL (initial_people teamsize : ℕ) (even_teams : ℕ → Prop)
  (initial_people_eq : initial_people = 38)
  (teamsize_eq : teamsize = 9)
  (even_teams_def : ∀ n, even_teams n ↔ n % 2 = 0) :
  ∃(p : ℕ), (initial_people + p) % teamsize = 0 ∧ even_teams ((initial_people + p) / teamsize) ∧ p = 16 := by
  sorry

end fewest_people_to_join_CBL_l250_250424


namespace solution_set_l250_250131

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l250_250131


namespace distance_between_cities_l250_250018

theorem distance_between_cities (x : ℝ) (h1 : x ≥ 100) (t : ℝ)
  (A_speed : ℝ := 12) (B_speed : ℝ := 0.05 * x)
  (condition_A : 7 + A_speed * t + B_speed * t = x)
  (condition_B : t = (x - 7) / (A_speed + B_speed)) :
  x = 140 :=
sorry

end distance_between_cities_l250_250018


namespace y_intercept_exists_l250_250161

def line_eq (x y : ℝ) : Prop := x + 2 * y + 2 = 0

theorem y_intercept_exists : ∃ y : ℝ, line_eq 0 y ∧ y = -1 :=
by
  sorry

end y_intercept_exists_l250_250161


namespace Ray_wrote_35_l250_250293

theorem Ray_wrote_35 :
  ∃ (x y : ℕ), (10 * x + y = 35) ∧ (10 * x + y = 4 * (x + y) + 3) ∧ (10 * x + y + 18 = 10 * y + x) :=
by
  sorry

end Ray_wrote_35_l250_250293


namespace smallest_x_value_l250_250504

-- Definitions based on given problem conditions
def is_solution (x y : ℕ) : Prop :=
  0 < x ∧ 0 < y ∧ (3 : ℝ) / 4 = y / (252 + x)

theorem smallest_x_value : ∃ x : ℕ, ∀ y : ℕ, is_solution x y → x = 0 :=
by
  sorry

end smallest_x_value_l250_250504


namespace num_terms_arithmetic_seq_l250_250341

theorem num_terms_arithmetic_seq (a d l : ℝ) (n : ℕ)
  (h1 : a = 3.25) 
  (h2 : d = 4)
  (h3 : l = 55.25)
  (h4 : l = a + (↑n - 1) * d) :
  n = 14 :=
by
  sorry

end num_terms_arithmetic_seq_l250_250341


namespace two_pipes_fill_time_l250_250307

theorem two_pipes_fill_time (R : ℝ) (h : 3 * R = 1 / 8) : 2 * R = 1 / 12 := 
by sorry

end two_pipes_fill_time_l250_250307


namespace probability_each_delegate_next_to_another_country_l250_250149

theorem probability_each_delegate_next_to_another_country
  (total_delegates : ℕ)
  (delegates_per_country : ℕ)
  (countries : ℕ)
  (seats : ℕ)
  (h1 : total_delegates = 16)
  (h2 : delegates_per_country = 4)
  (h3 : countries = 4)
  (h4 : seats = 16)
  : ∃ m n : ℕ, m.gcd n = 1 ∧ (m + n) = ? := 
sorry

end probability_each_delegate_next_to_another_country_l250_250149


namespace exists_fraction_equal_to_d_minus_1_l250_250871

theorem exists_fraction_equal_to_d_minus_1 (n d : ℕ) (hdiv : d > 0 ∧ n % d = 0) :
  ∃ k : ℕ, k < n ∧ (n - k) / (n - (n - k)) = d - 1 :=
by
  sorry

end exists_fraction_equal_to_d_minus_1_l250_250871


namespace football_team_progress_l250_250445

theorem football_team_progress : 
  ∀ {loss gain : ℤ}, loss = 5 → gain = 11 → gain - loss = 6 :=
by
  intros loss gain h_loss h_gain
  rw [h_loss, h_gain]
  sorry

end football_team_progress_l250_250445


namespace must_hold_inequality_l250_250221

variable (f : ℝ → ℝ)

noncomputable def condition : Prop := ∀ x > 0, x * (deriv^[2] f) x < 1

theorem must_hold_inequality (h : condition f) : f (Real.exp 1) < f 1 + 1 := 
sorry

end must_hold_inequality_l250_250221


namespace increasing_on_1_to_infinity_max_and_min_on_1_to_4_l250_250834

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem increasing_on_1_to_infinity : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (1 ≤ x2) → f x1 < f x2 := by
  sorry

theorem max_and_min_on_1_to_4 : 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f x ≤ f 4) ∧ 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f 1 ≤ f x) := by
  sorry

end increasing_on_1_to_infinity_max_and_min_on_1_to_4_l250_250834


namespace solution_set_line_l250_250122

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l250_250122


namespace unique_positive_solution_l250_250632

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l250_250632


namespace max_int_value_of_a_real_roots_l250_250644

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l250_250644


namespace bruised_more_than_wormy_l250_250863

noncomputable def total_apples : ℕ := 85
noncomputable def fifth_of_apples (n : ℕ) : ℕ := n / 5
noncomputable def apples_left_to_eat_raw : ℕ := 42

noncomputable def wormy_apples : ℕ := fifth_of_apples total_apples
noncomputable def total_non_raw_eatable_apples : ℕ := total_apples - apples_left_to_eat_raw
noncomputable def bruised_apples : ℕ := total_non_raw_eatable_apples - wormy_apples

theorem bruised_more_than_wormy :
  bruised_apples - wormy_apples = 43 - 17 :=
by sorry

end bruised_more_than_wormy_l250_250863


namespace largest_x_satisfies_condition_l250_250799

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l250_250799


namespace largest_x_eq_48_div_7_l250_250802

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l250_250802


namespace convert_decimal_to_fraction_l250_250440

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l250_250440


namespace find_a_find_n_l250_250236

noncomputable def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
noncomputable def S (a d n : ℕ) : ℕ := if n = 1 then a else sum_of_first_n_terms a d n
noncomputable def arithmetic_sum_property (a d n : ℕ) : Prop :=
  ∀ n ≥ 2, (S a d n) ^ 2 = 3 * n ^ 2 * arithmetic_sequence a d n + (S a d (n - 1)) ^ 2

theorem find_a (a : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2) :
  a = 3 :=
sorry

noncomputable def c (n : ℕ) (a5 : ℕ) : ℕ := 3 ^ (n - 1) + a5
noncomputable def sum_of_first_n_terms_c (n a5 : ℕ) : ℕ := (3^n - 1) / 2 + 15 * n
noncomputable def T (n a5 : ℕ) : ℕ := sum_of_first_n_terms_c n a5

theorem find_n (a : ℕ) (a5 : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2)
  (h2 : a = 3) (h3 : a5 = 15) :
  ∃ n : ℕ, 4 * T n a5 > S a 3 10 ∧ n = 3 :=
sorry

end find_a_find_n_l250_250236


namespace intersecting_lines_l250_250563

theorem intersecting_lines (a b : ℚ) :
  (3 = (1 / 3 : ℚ) * 4 + a) → 
  (4 = (1 / 2 : ℚ) * 3 + b) → 
  a + b = 25 / 6 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l250_250563


namespace sampling_survey_suitability_l250_250174

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

end sampling_survey_suitability_l250_250174


namespace chest_contents_correct_l250_250527

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l250_250527


namespace minimum_club_members_l250_250431

theorem minimum_club_members : ∃ (b : ℕ), (b = 7) ∧ ∃ (a : ℕ), (2 : ℚ) / 5 < (a : ℚ) / b ∧ (a : ℚ) / b < 1 / 2 := 
sorry

end minimum_club_members_l250_250431


namespace solution_set_unique_line_l250_250140

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l250_250140


namespace intersection_of_A_and_B_l250_250963

-- Given sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Prove the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := 
by
  sorry

end intersection_of_A_and_B_l250_250963


namespace jacket_price_equation_l250_250329

theorem jacket_price_equation (x : ℝ) (h : 0.8 * (1 + 0.5) * x - x = 28) : 0.8 * (1 + 0.5) * x = x + 28 :=
by sorry

end jacket_price_equation_l250_250329


namespace circumscribed_circle_radius_l250_250100

noncomputable def radius_of_circumscribed_circle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
  let R := a / (2 * Real.sin A)
  R

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (hb : b = 4) (hc : c = 2) (hA : A = Real.pi / 3) :
  radius_of_circumscribed_circle b c A = 2 := by
  sorry

end circumscribed_circle_radius_l250_250100


namespace question_equals_answer_l250_250259

theorem question_equals_answer (x y : ℝ) (h : abs (x - 6) + (y + 4)^2 = 0) : x + y = 2 :=
sorry

end question_equals_answer_l250_250259


namespace youtube_dislikes_l250_250589

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l250_250589


namespace seed_total_after_trading_l250_250177

theorem seed_total_after_trading :
  ∀ (Bom Gwi Yeon Eun : ℕ),
  Yeon = 3 * Gwi →
  Gwi = Bom + 40 →
  Eun = 2 * Gwi →
  Bom = 300 →
  Yeon_gives = 20 * Yeon / 100 →
  Bom_gives = 50 →
  let Yeon_after := Yeon - Yeon_gives
  let Gwi_after := Gwi + Yeon_gives
  let Bom_after := Bom - Bom_gives
  let Eun_after := Eun + Bom_gives
  Bom_after + Gwi_after + Yeon_after + Eun_after = 2340 :=
by
  intros Bom Gwi Yeon Eun hYeon hGwi hEun hBom hYeonGives hBomGives Yeon_after Gwi_after Bom_after Eun_after
  sorry

end seed_total_after_trading_l250_250177


namespace vertex_of_parabola_l250_250705

theorem vertex_of_parabola (a b c : ℝ) :
  (∀ x y : ℝ, (x = -2 ∧ y = 5) ∨ (x = 4 ∧ y = 5) ∨ (x = 2 ∧ y = 2) →
    y = a * x^2 + b * x + c) →
  (∃ x_vertex : ℝ, x_vertex = 1) :=
by
  sorry

end vertex_of_parabola_l250_250705


namespace total_bill_l250_250616

theorem total_bill (total_friends : ℕ) (extra_payment : ℝ) (total_bill : ℝ) (paid_by_friends : ℝ) :
  total_friends = 8 → extra_payment = 2.50 →
  (7 * ((total_bill / total_friends) + extra_payment)) = total_bill →
  total_bill = 140 :=
by
  intros h1 h2 h3
  sorry

end total_bill_l250_250616


namespace sqrt_inequality_l250_250076

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
by 
  sorry

end sqrt_inequality_l250_250076


namespace largest_real_number_condition_l250_250809

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l250_250809


namespace Evelyn_bottle_caps_l250_250946

theorem Evelyn_bottle_caps (initial_caps found_caps total_caps : ℕ)
  (h1 : initial_caps = 18)
  (h2 : found_caps = 63) :
  total_caps = 81 :=
by
  sorry

end Evelyn_bottle_caps_l250_250946


namespace binom_30_3_eq_4060_l250_250203

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l250_250203


namespace student_weight_l250_250261

-- Define the weights of the student and sister
variables (S R : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := S - 5 = 1.25 * R
def condition2 : Prop := S + R = 104

-- The theorem we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 60 := 
by
  sorry

end student_weight_l250_250261


namespace tens_digit_of_8_pow_2023_l250_250578

theorem tens_digit_of_8_pow_2023 :
    ∃ d, 0 ≤ d ∧ d < 10 ∧ (8^2023 % 100) / 10 = d ∧ d = 1 :=
by
  sorry

end tens_digit_of_8_pow_2023_l250_250578


namespace geometric_series_sum_l250_250770

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l250_250770


namespace total_crosswalk_lines_l250_250905

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l250_250905


namespace triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l250_250408

noncomputable def sum_sine_3A_3B_3C (A B C : ℝ) : ℝ :=
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C)

theorem triangle_inequality_sine_three_times {A B C : ℝ} (h : A + B + C = Real.pi) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) : 
  (-2 : ℝ) ≤ sum_sine_3A_3B_3C A B C ∧ sum_sine_3A_3B_3C A B C ≤ (3 * Real.sqrt 3 / 2) :=
by
  sorry

theorem equality_sine_three_times_lower_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = 0) (h2: B = Real.pi / 2) (h3: C = Real.pi / 2) :
  sum_sine_3A_3B_3C A B C = -2 :=
by
  sorry

theorem equality_sine_three_times_upper_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = Real.pi / 3) (h2: B = Real.pi / 3) (h3: C = Real.pi / 3) :
  sum_sine_3A_3B_3C A B C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l250_250408


namespace smallest_fraction_l250_250903

theorem smallest_fraction {a b c d e : ℚ}
  (ha : a = 7/15)
  (hb : b = 5/11)
  (hc : c = 16/33)
  (hd : d = 49/101)
  (he : e = 89/183) :
  (b < a) ∧ (b < c) ∧ (b < d) ∧ (b < e) := 
sorry

end smallest_fraction_l250_250903


namespace factor_expression_l250_250343

theorem factor_expression (x : ℝ) : 
  72 * x^2 + 108 * x + 36 = 36 * (2 * x^2 + 3 * x + 1) :=
sorry

end factor_expression_l250_250343


namespace A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l250_250502

def A : Set ℝ := { x | x^2 + x - 2 < 0 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem A_union_B_when_m_neg_half : A ∪ B (-1/2) = { x | -2 < x ∧ x < 3/2 } :=
by
  sorry

theorem B_subset_A_implies_m_geq_zero (m : ℝ) : B m ⊆ A → 0 ≤ m :=
by
  sorry

end A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l250_250502


namespace value_of_expression_l250_250014

theorem value_of_expression : (0.3 : ℝ)^2 + 0.1 = 0.19 := 
by sorry

end value_of_expression_l250_250014


namespace ones_digit_of_8_pow_47_l250_250023

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l250_250023


namespace expression_never_prime_l250_250433

theorem expression_never_prime (p : ℕ) (hp : Prime p) : ¬ Prime (3 * p^2 + 15) :=
by
  sorry

end expression_never_prime_l250_250433


namespace correct_transformation_l250_250579

-- Definitions of the equations and their transformations
def optionA := (forall (x : ℝ), ((x / 5) + 1 = x / 2) -> (2 * x + 10 = 5 * x))
def optionB := (forall (x : ℝ), (5 - 2 * (x - 1) = x + 3) -> (5 - 2 * x + 2 = x + 3))
def optionC := (forall (x : ℝ), (5 * x + 3 = 8) -> (5 * x = 8 - 3))
def optionD := (forall (x : ℝ), (3 * x = -7) -> (x = -7 / 3))

-- Theorem stating that option D is the correct transformation
theorem correct_transformation : optionD := 
by 
  sorry

end correct_transformation_l250_250579


namespace arithmetic_sequence_sixth_term_l250_250889

theorem arithmetic_sequence_sixth_term (a d : ℤ) 
    (sum_first_five : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
    (fourth_term : a + 3 * d = 4) : a + 5 * d = 6 :=
by
  sorry

end arithmetic_sequence_sixth_term_l250_250889


namespace minimum_value_of_f_l250_250348

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 1/x + 1/(x^2 + 1/x)

theorem minimum_value_of_f : 
  ∃ x > 0, f x = 2.5 :=
by 
  sorry

end minimum_value_of_f_l250_250348


namespace initial_candies_count_l250_250288

-- Definitions based on conditions
def NelliesCandies : Nat := 12
def JacobsCandies : Nat := NelliesCandies / 2
def LanasCandies : Nat := JacobsCandies - 3
def TotalCandiesEaten : Nat := NelliesCandies + JacobsCandies + LanasCandies
def RemainingCandies : Nat := 3 * 3
def InitialCandies := TotalCandiesEaten + RemainingCandies

-- Theorem stating the initial candies count
theorem initial_candies_count : InitialCandies = 30 := by 
  sorry

end initial_candies_count_l250_250288


namespace ellipse_standard_equation_l250_250079

-- Definitions from the problem
def major_axis : ℝ := 8
def eccentricity : ℝ := 3 / 4

-- Statement to prove that the given conditions lead to the expected equations
theorem ellipse_standard_equation (a : ℝ) (b : ℝ) (c : ℝ) :
  2 * a = major_axis ∧ c = eccentricity * a ∧ b^2 = a^2 - c^2 →
  (4 * a^2 = 16 ∧ b^2 = 7)
  ∨ (4 * a^2 = 7 ∧ b^2 = 16) :=
sorry

end ellipse_standard_equation_l250_250079


namespace value_of_n_l250_250033

theorem value_of_n {k n : ℕ} (h1 : k = 71 * n + 11) (h2 : (k : ℝ) / (n : ℝ) = 71.2) : n = 55 :=
sorry

end value_of_n_l250_250033


namespace cricket_target_runs_l250_250675

def run_rate_first_20_overs : ℝ := 4.2
def overs_first_20 : ℝ := 20
def run_rate_remaining_30_overs : ℝ := 5.533333333333333
def overs_remaining_30 : ℝ := 30
def total_runs_first_20 : ℝ := run_rate_first_20_overs * overs_first_20
def total_runs_remaining_30 : ℝ := run_rate_remaining_30_overs * overs_remaining_30

theorem cricket_target_runs :
  (total_runs_first_20 + total_runs_remaining_30) = 250 :=
by
  sorry

end cricket_target_runs_l250_250675


namespace cross_section_equilateral_triangle_l250_250006

-- Definitions and conditions
structure Cone where
  r : ℝ -- radius of the base circle
  R : ℝ -- radius of the semicircle
  h : ℝ -- slant height

axiom lateral_surface_unfolded (c : Cone) : c.R = 2 * c.r

def CrossSectionIsEquilateral (c : Cone) : Prop :=
  (c.h ^ 2 = (c.r * c.h)) ∧ (c.h = 2 * c.r)

-- Problem statement with conditions
theorem cross_section_equilateral_triangle (c : Cone) (h_equals_diameter : c.R = 2 * c.r) : CrossSectionIsEquilateral c :=
by
  sorry

end cross_section_equilateral_triangle_l250_250006


namespace pens_distributed_evenly_l250_250564

theorem pens_distributed_evenly (S : ℕ) (P : ℕ) (pencils : ℕ) 
  (hS : S = 10) (hpencils : pencils = 920) 
  (h_pencils_distributed : pencils % S = 0) 
  (h_pens_distributed : P % S = 0) : 
  ∃ k : ℕ, P = 10 * k :=
by 
  sorry

end pens_distributed_evenly_l250_250564


namespace minimum_value_of_quadratic_l250_250416

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem minimum_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f(y) >= f(x) ∧ f(x) = -3 :=
by
  sorry

end minimum_value_of_quadratic_l250_250416


namespace probability_of_extreme_value_l250_250746

def f (a b x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * a * x^2 + b * x

def has_extreme_value (a b : ℝ) : Prop :=
  let Δ := a^2 - 4 * b
  Δ > 0

def num_satisfying_pairs : ℕ :=
  -- count all (a, b) pairs for which has_extreme_value holds
  let pairs := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6], has_extreme_value a b]
  pairs.length

def total_pairs : ℕ := 6 * 6

noncomputable def probability_extreme_value : ℝ :=
  num_satisfying_pairs / total_pairs

theorem probability_of_extreme_value :
  probability_extreme_value = 17 / 36 :=
sorry

end probability_of_extreme_value_l250_250746


namespace not_possible_acquaintance_arrangement_l250_250352

-- Definitions and conditions for the problem
def num_people : ℕ := 40
def even_people_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 0 → A ≠ B → true -- A and B have a mutual acquaintance if an even number of people sit between them

def odd_people_not_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 1 → A ≠ B → true -- A and B do not have a mutual acquaintance if an odd number of people sit between them

theorem not_possible_acquaintance_arrangement : ¬ (∀ A B : ℕ, A ≠ B →
  (∀ num_between : ℕ, (num_between % 2 = 0 → even_people_acquainted A B num_between) ∧
  (num_between % 2 = 1 → odd_people_not_acquainted A B num_between))) :=
sorry

end not_possible_acquaintance_arrangement_l250_250352


namespace soccer_players_positions_l250_250470

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l250_250470


namespace x_squared_plus_y_squared_l250_250967

theorem x_squared_plus_y_squared (x y : ℝ) (h₀ : x + y = 10) (h₁ : x * y = 15) : x^2 + y^2 = 70 :=
by
  sorry

end x_squared_plus_y_squared_l250_250967


namespace find_number_l250_250373

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 100) : N = 192 :=
sorry

end find_number_l250_250373


namespace men_absent_l250_250040

/-- 
A group of men decided to do a work in 20 days, but some of them became absent. 
The rest of the group did the work in 40 days. The original number of men was 20. 
Prove that 10 men became absent. 
--/
theorem men_absent 
    (original_men : ℕ) (absent_men : ℕ) (planned_days : ℕ) (actual_days : ℕ)
    (h1 : original_men = 20) (h2 : planned_days = 20) (h3 : actual_days = 40)
    (h_work : original_men * planned_days = (original_men - absent_men) * actual_days) : 
    absent_men = 10 :=
    by 
    rw [h1, h2, h3] at h_work
    -- Proceed to manually solve the equation, but here we add sorry
    sorry

end men_absent_l250_250040


namespace smallest_positive_integer_n_l250_250312

noncomputable def smallest_n : ℕ :=
  Inf { n : ℕ | ∃ (k m : ℕ), 3 * n = (3 * k) ^ 2 ∧ 5 * n = (5 * m) ^ 3 }

theorem smallest_positive_integer_n : smallest_n = 1875 :=
sorry

end smallest_positive_integer_n_l250_250312


namespace digit_7_appears_602_times_l250_250850

theorem digit_7_appears_602_times :
  ∑ n in Finset.range 2018, has_digit 7 n = 602 := 
sorry

end digit_7_appears_602_times_l250_250850


namespace problem_1_problem_2_l250_250254

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1 }

-- Problem 1: Prove that if A ∩ B = [1, 3], then m = 2
theorem problem_1 (m : ℝ) (h : (A ∩ B m) = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) : m = 2 :=
sorry

-- Problem 2: Prove that if A ⊆ complement ℝ B m, then m > 4 or m < -2
theorem problem_2 (m : ℝ) (h : A ⊆ { x : ℝ | x < m - 1 ∨ x > m + 1 }) : m > 4 ∨ m < -2 :=
sorry

end problem_1_problem_2_l250_250254


namespace inequality_not_hold_l250_250835

variable {f : ℝ → ℝ}
variable {x : ℝ}

theorem inequality_not_hold (h : ∀ x : ℝ, -π/2 < x ∧ x < π/2 → (deriv f x) * cos x + f x * sin x > 0) : 
  ¬(sqrt 2 * f (π / 3) < f (π / 4)) :=
sorry

end inequality_not_hold_l250_250835


namespace sufficient_condition_for_line_perpendicular_to_plane_l250_250356

variables {Plane Line : Type}
variables (α β γ : Plane) (m n l : Line)

-- Definitions of perpendicularity and inclusion
def perp (l : Line) (p : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def parallel (p₁ p₂ : Plane) : Prop := sorry -- definition of parallel planes
def incl (l : Line) (p : Plane) : Prop := sorry -- definition of a line being in a plane

-- The given conditions
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- The proof goal
theorem sufficient_condition_for_line_perpendicular_to_plane :
  perp m β :=
by
    sorry

end sufficient_condition_for_line_perpendicular_to_plane_l250_250356


namespace solution_set_l250_250127

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l250_250127


namespace determine_head_start_l250_250045

def head_start (v : ℝ) (s : ℝ) : Prop :=
  let a_speed := 2 * v
  let distance := 142
  distance / a_speed = (distance - s) / v

theorem determine_head_start (v : ℝ) : head_start v 71 :=
  by
    sorry

end determine_head_start_l250_250045


namespace original_price_of_article_l250_250051

theorem original_price_of_article (selling_price : ℝ) (loss_percent : ℝ) (P : ℝ) 
  (h1 : selling_price = 450)
  (h2 : loss_percent = 25)
  : selling_price = (1 - loss_percent / 100) * P → P = 600 :=
by
  sorry

end original_price_of_article_l250_250051


namespace sequence_length_l250_250366

theorem sequence_length :
  ∀ (a d n : ℤ), a = -6 → d = 4 → (a + (n - 1) * d = 50) → n = 15 :=
by
  intros a d n ha hd h_seq
  sorry

end sequence_length_l250_250366


namespace parabola_triangle_areas_l250_250960

-- Define necessary points and expressions
variables (x1 y1 x2 y2 x3 y3 : ℝ)
variables (m n : ℝ)
def parabola_eq (x y : ℝ) := y ^ 2 = 4 * x
def median_line (m n x y : ℝ) := m * x + n * y - m = 0
def areas_sum_sq (S1 S2 S3 : ℝ) := S1 ^ 2 + S2 ^ 2 + S3 ^ 2 = 3

-- Main statement
theorem parabola_triangle_areas :
  (parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ parabola_eq x3 y3) →
  (m ≠ 0) →
  (median_line m n 1 0) →
  (x1 + x2 + x3 = 3) →
  ∃ S1 S2 S3 : ℝ, areas_sum_sq S1 S2 S3 :=
by sorry

end parabola_triangle_areas_l250_250960


namespace probability_of_green_ball_l250_250771

-- Definitions according to the conditions.
def containerA : ℕ × ℕ := (4, 6) -- 4 red balls, 6 green balls
def containerB : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls
def containerC : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls

-- Proving the probability of selecting a green ball.
theorem probability_of_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := (containerA.2 : ℚ) / (containerA.1 + containerA.2)
  let pGreenB := (containerB.2 : ℚ) / (containerB.1 + containerB.2)
  let pGreenC := (containerC.2 : ℚ) / (containerC.1 + containerC.2)
  pA * pGreenA + pB * pGreenB + pC * pGreenC = 7 / 15
  :=
by
  -- Formal proof will be filled in here.
  sorry

end probability_of_green_ball_l250_250771


namespace total_days_to_finish_tea_and_coffee_l250_250844

-- Define the given conditions formally before expressing the theorem
def drinks_coffee_together (days : ℕ) : Prop := days = 10
def drinks_coffee_alone_A (days : ℕ) : Prop := days = 12
def drinks_tea_together (days : ℕ) : Prop := days = 12
def drinks_tea_alone_B (days : ℕ) : Prop := days = 20

-- The goal is to prove that A and B together finish a pound of tea and a can of coffee in 35 days
theorem total_days_to_finish_tea_and_coffee : 
  ∃ days : ℕ, 
    drinks_coffee_together 10 ∧ 
    drinks_coffee_alone_A 12 ∧ 
    drinks_tea_together 12 ∧ 
    drinks_tea_alone_B 20 ∧ 
    days = 35 :=
by
  sorry

end total_days_to_finish_tea_and_coffee_l250_250844


namespace quadratic_ratio_l250_250884

theorem quadratic_ratio (b c : ℤ) (h : ∀ x : ℤ, x^2 + 1400 * x + 1400 = (x + b) ^ 2 + c) : c / b = -698 :=
sorry

end quadratic_ratio_l250_250884


namespace t_n_closed_form_t_2022_last_digit_l250_250640

noncomputable def t_n (n : ℕ) : ℕ :=
  (4^n - 3 * 3^n + 3 * 2^n - 1) / 6

theorem t_n_closed_form (n : ℕ) (hn : 0 < n) :
  t_n n = (4^n - 3 * 3^n + 3 * 2^n - 1) / 6 :=
by
  sorry

theorem t_2022_last_digit :
  (t_n 2022) % 10 = 1 :=
by
  sorry

end t_n_closed_form_t_2022_last_digit_l250_250640


namespace geometric_series_sum_l250_250765

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l250_250765


namespace value_spent_more_than_l250_250738

theorem value_spent_more_than (x : ℕ) (h : 8 * 12 + (x + 8) = 117) : x = 13 :=
by
  sorry

end value_spent_more_than_l250_250738


namespace cos_three_pi_over_two_l250_250224

theorem cos_three_pi_over_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  -- Provided as correct by the solution steps role
  sorry

end cos_three_pi_over_two_l250_250224


namespace integer_side_lengths_triangle_l250_250656

theorem integer_side_lengths_triangle :
  ∃ (a b c : ℤ), (abc = 2 * (a - 1) * (b - 1) * (c - 1)) ∧
            (a = 8 ∧ b = 7 ∧ c = 3 ∨ a = 6 ∧ b = 5 ∧ c = 4) := 
by
  sorry

end integer_side_lengths_triangle_l250_250656


namespace cos_sum_sin_sum_cos_diff_sin_diff_l250_250269

section

variables (A B : ℝ)

-- Definition of cos and sin of angles
def cos (θ : ℝ) : ℝ := sorry
def sin (θ : ℝ) : ℝ := sorry

-- Cosine of the sum of angles
theorem cos_sum : cos (A + B) = cos A * cos B - sin A * sin B := sorry

-- Sine of the sum of angles
theorem sin_sum : sin (A + B) = sin A * cos B + cos A * sin B := sorry

-- Cosine of the difference of angles
theorem cos_diff : cos (A - B) = cos A * cos B + sin A * sin B := sorry

-- Sine of the difference of angles
theorem sin_diff : sin (A - B) = sin A * cos B - cos A * sin B := sorry

end

end cos_sum_sin_sum_cos_diff_sin_diff_l250_250269


namespace max_value_y_l250_250241

noncomputable def y (x : ℝ) : ℝ := 3 - 3*x - 1/x

theorem max_value_y : (∃ x > 0, ∀ x' > 0, y x' ≤ y x) ∧ (y (1 / Real.sqrt 3) = 3 - 2 * Real.sqrt 3) :=
by
  sorry

end max_value_y_l250_250241


namespace isosceles_in_27_gon_l250_250231

def vertices := {x : ℕ // x < 27}

def is_isosceles_triangle (a b c : vertices) : Prop :=
  (a.val + c.val) / 2 % 27 = b.val

def is_isosceles_trapezoid (a b c d : vertices) : Prop :=
  (a.val + d.val) / 2 % 27 = (b.val + c.val) / 2 % 27

def seven_points_form_isosceles (s : Finset vertices) : Prop :=
  ∃ (a b c : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s), is_isosceles_triangle a b c

def seven_points_form_isosceles_trapezoid (s : Finset vertices) : Prop :=
  ∃ (a b c d : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s) (h4 : d ∈ s), is_isosceles_trapezoid a b c d

theorem isosceles_in_27_gon :
  ∀ (s : Finset vertices), s.card = 7 → 
  (seven_points_form_isosceles s) ∨ (seven_points_form_isosceles_trapezoid s) :=
by sorry

end isosceles_in_27_gon_l250_250231


namespace original_price_l250_250601

theorem original_price (saving : ℝ) (percentage : ℝ) (h_saving : saving = 10) (h_percentage : percentage = 0.10) :
  ∃ OP : ℝ, OP = 100 :=
by
  sorry

end original_price_l250_250601


namespace nth_term_of_sequence_99_l250_250994

def sequence_rule (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n > 19 ∧ n % 7 ≠ 0 then n - 5
  else n + 7

noncomputable def sequence_nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.repeat sequence_rule n start

theorem nth_term_of_sequence_99 :
  sequence_nth_term 65 98 = 30 :=
sorry

end nth_term_of_sequence_99_l250_250994


namespace solution_set_l250_250143

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l250_250143


namespace copper_percentage_l250_250194

theorem copper_percentage (copperFirst copperSecond totalWeight1 totalWeight2: ℝ) 
    (h1 : copperFirst = 0.25)
    (h2 : copperSecond = 0.50) 
    (h3 : totalWeight1 = 200) 
    (h4 : totalWeight2 = 800) : 
    (copperFirst * totalWeight1 + copperSecond * totalWeight2) / (totalWeight1 + totalWeight2) * 100 = 45 := 
by 
  sorry

end copper_percentage_l250_250194


namespace inscribed_circle_radius_l250_250702

theorem inscribed_circle_radius (R : ℝ) (h : 0 < R) : 
  ∃ x : ℝ, (x = R / 3) :=
by
  -- Given conditions
  have h1 : R > 0 := h

  -- Mathematical proof statement derived from conditions
  sorry

end inscribed_circle_radius_l250_250702


namespace least_product_xy_l250_250493

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l250_250493


namespace gcd_100_450_l250_250347

theorem gcd_100_450 : Int.gcd 100 450 = 50 := 
by sorry

end gcd_100_450_l250_250347


namespace check_bag_correct_l250_250682

-- Define the conditions as variables and statements
variables (uber_to_house : ℕ) (uber_to_airport : ℕ) (check_bag : ℕ)
          (security : ℕ) (wait_for_boarding : ℕ) (wait_for_takeoff : ℕ) (total_time : ℕ)

-- Assign the given conditions
def given_conditions : Prop :=
  uber_to_house = 10 ∧
  uber_to_airport = 5 * uber_to_house ∧
  security = 3 * check_bag ∧
  wait_for_boarding = 20 ∧
  wait_for_takeoff = 2 * wait_for_boarding ∧
  total_time = 180

-- Define the question as a statement
def check_bag_time (check_bag : ℕ) : Prop :=
  check_bag = 15

-- The Lean theorem based on the problem, conditions, and answer
theorem check_bag_correct :
  given_conditions uber_to_house uber_to_airport check_bag security wait_for_boarding wait_for_takeoff total_time →
  check_bag_time check_bag :=
by
  intros h
  sorry

end check_bag_correct_l250_250682


namespace fill_time_with_leak_is_correct_l250_250920

-- Define the conditions
def time_to_fill_without_leak := 8
def time_to_empty_with_leak := 24

-- Define the rates
def fill_rate := 1 / time_to_fill_without_leak
def leak_rate := 1 / time_to_empty_with_leak
def effective_fill_rate := fill_rate - leak_rate

-- Prove the time to fill with leak
def time_to_fill_with_leak := 1 / effective_fill_rate

-- The theorem to prove that the time is 12 hours
theorem fill_time_with_leak_is_correct :
  time_to_fill_with_leak = 12 := by
  simp [time_to_fill_without_leak, time_to_empty_with_leak, fill_rate, leak_rate, effective_fill_rate, time_to_fill_with_leak]
  sorry

end fill_time_with_leak_is_correct_l250_250920


namespace birds_meeting_distance_l250_250447

theorem birds_meeting_distance 
  (D : ℝ) (S1 : ℝ) (S2 : ℝ) (t : ℝ)
  (H1 : D = 45)
  (H2 : S1 = 6)
  (H3 : S2 = 2.5)
  (H4 : t = D / (S1 + S2)) :
  S1 * t = 31.76 :=
by
  sorry

end birds_meeting_distance_l250_250447


namespace min_sum_reciprocal_l250_250827

theorem min_sum_reciprocal (a b c : ℝ) (hp0 : 0 < a) (hp1 : 0 < b) (hp2 : 0 < c) (h : a + b + c = 1) : 
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
by
  sorry

end min_sum_reciprocal_l250_250827


namespace ingrid_tax_rate_l250_250111

def john_income : ℝ := 57000
def ingrid_income : ℝ := 72000
def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_tax_rate :
  let john_tax := john_tax_rate * john_income
  let combined_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * combined_income
  let ingrid_tax := total_tax - john_tax
  let ingrid_tax_rate := ingrid_tax / ingrid_income
  ingrid_tax_rate = 0.40 :=
by
  sorry

end ingrid_tax_rate_l250_250111


namespace hockey_season_length_l250_250721

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l250_250721


namespace number_is_0_point_5_l250_250599

theorem number_is_0_point_5 (x : ℝ) (h : x = 1/6 + 0.33333333333333337) : x = 0.5 := 
by
  -- The actual proof would go here.
  sorry

end number_is_0_point_5_l250_250599


namespace ellipse_rolls_condition_l250_250053

variables {a b c : ℝ} (h_ellipse : ∀ x : ℝ, x ∈ (0..2 * Real.pi * a) → c = real.sqrt (b^2 - a^2))
  (h_roll_without_slip : b ≥ a)
  (h_curv : ∀ x : ℝ, c < a ^ 1.5 / b^2)

theorem ellipse_rolls_condition : b ≥ a ∧ c = real.sqrt (b^2 - a^2) ∧ c < a ^ 1.5 / b^2 :=
by
  apply and.intro h_roll_without_slip
  apply and.intro (h_ellipse _ _)
  apply h_curv
  done
end

end ellipse_rolls_condition_l250_250053


namespace largest_x_satisfies_condition_l250_250797

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l250_250797


namespace financial_calculations_correct_l250_250060

noncomputable def revenue : ℝ := 2500000
noncomputable def expenses : ℝ := 1576250
noncomputable def loan_payment_per_month : ℝ := 25000
noncomputable def number_of_shares : ℕ := 1600
noncomputable def ceo_share_percentage : ℝ := 0.35

theorem financial_calculations_correct :
  let net_profit := (revenue - expenses) - (revenue - expenses) * 0.2 in
  let total_loan_payment := loan_payment_per_month * 12 in
  let dividends_per_share := (net_profit - total_loan_payment) / number_of_shares in
  let ceo_dividends := dividends_per_share * ceo_share_percentage * number_of_shares in
  net_profit = 739000 ∧
  total_loan_payment = 300000 ∧
  dividends_per_share = 274 ∧
  ceo_dividends = 153440 :=
begin
  sorry
end

end financial_calculations_correct_l250_250060


namespace general_formula_arithmetic_sequence_sum_of_sequence_b_l250_250237

-- Definitions of arithmetic sequence {a_n} and geometric sequence conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

def arithmetic_sum_S3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def general_formula (a : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, a n = n + 1

def sum_first_n_terms_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2

-- The Lean theorem statements
theorem general_formula_arithmetic_sequence
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : geometric_sequence a)
  (h4 : arithmetic_sum_S3 S) :
  general_formula a :=
  sorry

theorem sum_of_sequence_b
  (a b : ℕ → ℤ) (T : ℕ → ℤ)
  (h1 : general_formula a)
  (h2 : ∀ n : ℕ, b n = (a n - 1) * 2^n)
  (h3 : sum_first_n_terms_b b T) :
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2 :=
  sorry

end general_formula_arithmetic_sequence_sum_of_sequence_b_l250_250237


namespace geometric_sequence_seventh_term_l250_250002

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term_l250_250002


namespace total_distance_traveled_is_7_75_l250_250846

open Real

def walking_time_minutes : ℝ := 30
def walking_rate : ℝ := 3.5

def running_time_minutes : ℝ := 45
def running_rate : ℝ := 8

theorem total_distance_traveled_is_7_75 :
  let walking_hours := walking_time_minutes / 60
  let distance_walked := walking_rate * walking_hours
  let running_hours := running_time_minutes / 60
  let distance_run := running_rate * running_hours
  let total_distance := distance_walked + distance_run
  total_distance = 7.75 :=
by
  sorry

end total_distance_traveled_is_7_75_l250_250846


namespace probability_three_reds_first_l250_250605

open ProbabilityTheory

/-- A hat contains 3 red chips and 3 green chips. Chips are drawn randomly one by one without replacement until either all 3 red chips or all 3 green chips are drawn. Prove that the probability of drawing all 3 red chips before all 3 green chips is 1/2. -/
theorem probability_three_reds_first : 
  let chips := {x : Fin 6 // x < 3 ∨ x ≥ 3} in
  let event := {ω : Finset (Fin 6) | ∀ x ∈ ω, ∀ y ∈ (Finset.univ \ ω), (x < 3 ↔ y ≥ 3)} in
  (|{ω ∈ (events Ω) | ∀ x ∈ ω, x < 3}| / |Ω|) = 1 / 2 :
  sorry

end probability_three_reds_first_l250_250605


namespace sequence_form_l250_250949

-- Defining the sequence a_n as a function f
def seq (f : ℕ → ℕ) : Prop :=
  ∃ c : ℝ, (0 < c) ∧ ∀ m n : ℕ, Nat.gcd (f m + n) (f n + m) > (c * (m + n))

-- Proving that if there exists such a sequence, then it is of the form n + c
theorem sequence_form (f : ℕ → ℕ) (h : seq f) :
  ∃ c : ℤ, ∀ n : ℕ, f n = n + c :=
sorry

end sequence_form_l250_250949


namespace tom_won_whack_a_mole_l250_250758

variable (W : ℕ)  -- let W be the number of tickets Tom won playing 'whack a mole'
variable (won_skee_ball : ℕ := 25)  -- Tom won 25 tickets playing 'skee ball'
variable (spent_on_hat : ℕ := 7)  -- Tom spent 7 tickets on a hat
variable (tickets_left : ℕ := 50)  -- Tom has 50 tickets left

theorem tom_won_whack_a_mole :
  W + 25 + 50 = 57 →
  W = 7 :=
by
  sorry  -- proof goes here

end tom_won_whack_a_mole_l250_250758


namespace gcd_poly_multiple_l250_250242

theorem gcd_poly_multiple {x : ℤ} (h : ∃ k : ℤ, x = 54321 * k) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 14)) x = 1 :=
sorry

end gcd_poly_multiple_l250_250242


namespace max_area_angle_A_l250_250517

open Real

theorem max_area_angle_A (A B C : ℝ) (tan_A tan_B : ℝ) :
  tan A * tan B = 1 ∧ AB = sqrt 3 → 
  (∃ A, A = π / 4 ∧ area_maximized)
  :=
by sorry

end max_area_angle_A_l250_250517


namespace nate_ratio_is_four_to_one_l250_250542

def nate_exercise : Prop :=
  ∃ (D T L : ℕ), 
    T = D + 500 ∧ 
    T = 1172 ∧ 
    L = 168 ∧ 
    D / L = 4

theorem nate_ratio_is_four_to_one : nate_exercise := 
  sorry

end nate_ratio_is_four_to_one_l250_250542


namespace ken_and_kendra_brought_home_l250_250114

-- Define the main variables
variables (ken_caught kendra_caught ken_brought_home : ℕ)

-- Define the conditions as hypothesis
def conditions :=
  kendra_caught = 30 ∧
  ken_caught = 2 * kendra_caught ∧
  ken_brought_home = ken_caught - 3

-- Define the problem to prove
theorem ken_and_kendra_brought_home :
  (ken_caught + kendra_caught = 87) :=
begin
  -- Unpacking the conditions for readability
  unfold conditions at *,
  sorry -- Proof will go here
end

end ken_and_kendra_brought_home_l250_250114


namespace final_center_coordinates_l250_250338

-- Definition of the initial condition: the center of Circle U
def center_initial : ℝ × ℝ := (3, -4)

-- Definition of the reflection function across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Definition of the translation function to translate a point 5 units up
def translate_up_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 5)

-- Defining the final coordinates after reflection and translation
def center_final : ℝ × ℝ :=
  translate_up_5 (reflect_y_axis center_initial)

-- Problem statement: Prove that the final center coordinates are (-3, 1)
theorem final_center_coordinates :
  center_final = (-3, 1) :=
by {
  -- Skipping the proof itself, but the theorem statement should be equivalent
  sorry
}

end final_center_coordinates_l250_250338


namespace amount_saved_percentage_l250_250531

variable (S : ℝ) 

-- Condition: Last year, Sandy saved 7% of her annual salary
def amount_saved_last_year (S : ℝ) : ℝ := 0.07 * S

-- Condition: This year, she made 15% more money than last year
def salary_this_year (S : ℝ) : ℝ := 1.15 * S

-- Condition: This year, she saved 10% of her salary
def amount_saved_this_year (S : ℝ) : ℝ := 0.10 * salary_this_year S

-- The statement to prove
theorem amount_saved_percentage (S : ℝ) : 
  amount_saved_this_year S = 1.642857 * amount_saved_last_year S :=
by 
  sorry

end amount_saved_percentage_l250_250531


namespace ant_return_5_moves_l250_250326

-- Define the conditions of the problem
def five_dim_hypercube := SimpleGraph.cube 5
def start_vertex := (0, 0, 0, 0, 0 : Fin 2)

theorem ant_return_5_moves :
  let moves := 5
  let distance := 1
  let total_ways := 6240
  ∃ (count : ℕ), count = total_ways :=
sorry

end ant_return_5_moves_l250_250326


namespace find_theta_l250_250958

open Real

theorem find_theta (x y : ℝ) (hx : x = (sqrt 3) / 2) (hy : y = -1 / 2) (hxy : y < 0 ∧ x > 0) (hθ : θ = Real.arctan (y / x) ∨ θ = Real.arctan (y / x) + π ∨ θ = Real.arctan (y / x) + 2*π) : θ = 11*π / 6 :=
by
  sorry

end find_theta_l250_250958


namespace max_value_of_z_l250_250430

theorem max_value_of_z : ∀ x : ℝ, (x^2 - 14 * x + 10 ≤ 0 - 39) :=
by
  sorry

end max_value_of_z_l250_250430


namespace total_journey_distance_l250_250597

-- Definitions of the conditions

def journey_time : ℝ := 40
def first_half_speed : ℝ := 20
def second_half_speed : ℝ := 30

-- Proof statement
theorem total_journey_distance : ∃ D : ℝ, (D / first_half_speed + D / second_half_speed = journey_time) ∧ (D = 960) :=
by 
  sorry

end total_journey_distance_l250_250597


namespace at_least_100_valid_pairs_l250_250015

-- Define the conditions
def boots_distribution (L41 L42 L43 R41 R42 R43 : ℕ) : Prop :=
  L41 + L42 + L43 = 300 ∧ R41 + R42 + R43 = 300 ∧
  (L41 = 200 ∨ L42 = 200 ∨ L43 = 200) ∧
  (R41 = 200 ∨ R42 = 200 ∨ R43 = 200)

-- Define the theorem to be proven
theorem at_least_100_valid_pairs (L41 L42 L43 R41 R42 R43 : ℕ) :
  boots_distribution L41 L42 L43 R41 R42 R43 → 
  (L41 ≥ 100 ∧ R41 ≥ 100 ∨ L42 ≥ 100 ∧ R42 ≥ 100 ∨ L43 ≥ 100 ∧ R43 ≥ 100) → 100 ≤ min L41 R41 ∨ 100 ≤ min L42 R42 ∨ 100 ≤ min L43 R43 :=
  sorry

end at_least_100_valid_pairs_l250_250015


namespace binom_multiplication_l250_250210

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l250_250210


namespace stationery_store_profit_l250_250460

variable (a : ℝ)

def store_cost : ℝ := 100 * a
def markup_price : ℝ := a * 1.2
def discount_price : ℝ := markup_price a * 0.8

def revenue_first_half : ℝ := 50 * markup_price a
def revenue_second_half : ℝ := 50 * discount_price a
def total_revenue : ℝ := revenue_first_half a + revenue_second_half a

def profit : ℝ := total_revenue a - store_cost a

theorem stationery_store_profit : profit a = 8 * a := 
by sorry

end stationery_store_profit_l250_250460


namespace binom_multiplication_l250_250211

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l250_250211


namespace largest_x_exists_largest_x_largest_real_number_l250_250783

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l250_250783


namespace positive_difference_1010_1000_l250_250477

-- Define the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Define the specific terms
def a_1000 := arithmetic_sequence 5 7 1000
def a_1010 := arithmetic_sequence 5 7 1010

-- Proof statement
theorem positive_difference_1010_1000 : a_1010 - a_1000 = 70 :=
by
  sorry

end positive_difference_1010_1000_l250_250477


namespace minimum_value_fraction_l250_250951

theorem minimum_value_fraction (x : ℝ) (h : x > 6) : (∃ c : ℝ, c = 12 ∧ ((x = c) → (x^2 / (x - 6) = 18)))
  ∧ (∀ y : ℝ, y > 6 → y^2 / (y - 6) ≥ 18) :=
by {
  sorry
}

end minimum_value_fraction_l250_250951


namespace find_ABC_l250_250386

noncomputable def problem (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 8 ∧ B < 8 ∧ C < 6 ∧
  (A * 8 + B + C = 8 * 2 + C) ∧
  (A * 8 + B + B * 8 + A = C * 8 + C) ∧
  (100 * A + 10 * B + C = 246)

theorem find_ABC : ∃ A B C : ℕ, problem A B C := sorry

end find_ABC_l250_250386


namespace sin_inequality_l250_250857

theorem sin_inequality (d n : ℤ) (hd : d ≥ 1) (hnsq : ∀ k : ℤ, k * k ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * |Real.sin (n * Real.pi * Real.sqrt d)| ≥ 1 := by
  sorry

end sin_inequality_l250_250857


namespace binomial_coeffs_not_arith_seq_l250_250292

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def are_pos_integer (n : ℕ) : Prop := n > 0

def is_arith_seq (a b c d : ℕ) : Prop := 
  2 * b = a + c ∧ 2 * c = b + d 

theorem binomial_coeffs_not_arith_seq (n r : ℕ) : 
  are_pos_integer n → are_pos_integer r → n ≥ r + 3 → ¬ is_arith_seq (binomial n r) (binomial n (r+1)) (binomial n (r+2)) (binomial n (r+3)) :=
by
  sorry

end binomial_coeffs_not_arith_seq_l250_250292


namespace walter_bus_time_l250_250895

/--
Walter wakes up at 6:30 a.m., leaves for the bus at 7:30 a.m., attends 7 classes that each last 45 minutes,
enjoys a 40-minute lunch, and spends 2.5 hours of additional time at school for activities.
He takes the bus home and arrives at 4:30 p.m.
Prove that Walter spends 35 minutes on the bus.
-/
theorem walter_bus_time : 
  let total_time_away := 9 * 60 -- in minutes
  let class_time := 7 * 45 -- in minutes
  let lunch_time := 40 -- in minutes
  let additional_school_time := 2.5 * 60 -- in minutes
  total_time_away - (class_time + lunch_time + additional_school_time) = 35 := 
by
  sorry

end walter_bus_time_l250_250895


namespace cost_of_adult_ticket_is_15_l250_250419

variable (A : ℕ) -- Cost of an adult ticket
variable (total_tickets : ℕ) (cost_child_ticket : ℕ) (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)

theorem cost_of_adult_ticket_is_15
  (h1 : total_tickets = 522)
  (h2 : cost_child_ticket = 8)
  (h3 : total_revenue = 5086)
  (h4 : adult_tickets_sold = 130) 
  (h5 : (total_tickets - adult_tickets_sold) * cost_child_ticket + adult_tickets_sold * A = total_revenue) :
  A = 15 :=
by
  sorry

end cost_of_adult_ticket_is_15_l250_250419


namespace third_day_opponent_l250_250588

def Person : Type := {A, B, C, D : Person}

def plays (x y : Person) : Prop :=
  (x, y) ∈ {(A, C), (C, D)} ∨ (y, x) ∈ {(A, C), (C, D)}

theorem third_day_opponent :
  ∀ (A B C D : Person), 
  (plays A C) →
  (plays C D) →
  (¬ plays A B) →
  (¬ plays B D) →
  (∀ x, (x ≠ A ∧ x ≠ B) → (plays B x))  → 
  (plays B C) := 
by {
  intros A B C D h1 h2 h3 h4 h5,
  sorry
}

end third_day_opponent_l250_250588


namespace probability_at_least_three_heads_l250_250478

noncomputable def coin_toss_pmf : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofFinite (λ i, if i ≤ 5 then (nat.succ_pmf 5).pmf i else 0)

def prob_three_or_more_heads : ℕ → ℝ
  | 5 => (∀ k, k ≥ 3 → coin_toss_pmf pmf k).sum

theorem probability_at_least_three_heads :
  prob_three_or_more_heads 5 = 1 / 2 :=
by
  sorry

end probability_at_least_three_heads_l250_250478


namespace length_of_BD_is_six_l250_250266

-- Definitions of the conditions
def AB : ℕ := 6
def BC : ℕ := 11
def CD : ℕ := 6
def DA : ℕ := 8
def BD : ℕ := 6 -- adding correct answer into definition

-- The statement we want to prove
theorem length_of_BD_is_six (hAB : AB = 6) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 8) (hBD_int : BD = 6) : 
  BD = 6 :=
by
  -- Proof placeholder
  sorry

end length_of_BD_is_six_l250_250266


namespace equivalent_operation_l250_250175

theorem equivalent_operation (x : ℚ) :
  (x / (5 / 6) * (4 / 7)) = x * (24 / 35) :=
by
  sorry

end equivalent_operation_l250_250175


namespace sum_of_perimeters_of_squares_l250_250019

theorem sum_of_perimeters_of_squares
  (x y : ℝ)
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12*Real.sqrt 26 := by
  sorry

end sum_of_perimeters_of_squares_l250_250019


namespace solve_g_eq_g_inv_l250_250219

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end solve_g_eq_g_inv_l250_250219


namespace prime_saturated_two_digit_max_is_98_l250_250042

def is_prime_saturated (z : ℕ) : Prop :=
  ∃ p, (z > 1) ∧ (Nat.factors z = p) ∧ (List.prod p < Real.sqrt z)

def greatest_prime_saturated_two_digit : ℕ :=
  98

theorem prime_saturated_two_digit_max_is_98 :
  greatest_prime_saturated_two_digit = 98 ∧ is_prime_saturated greatest_prime_saturated_two_digit :=
by
  -- We need to prove the greatest two-digit prime saturated integer is 98
  sorry

end prime_saturated_two_digit_max_is_98_l250_250042


namespace range_of_m_for_distinct_real_roots_l250_250642

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4 * x1 - m = 0 ∧ x2^2 - 4 * x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_l250_250642


namespace part_a_intersection_part_b_section_l250_250731

section ThreeFacedAngle

variable (O a b c A B : Point)
variable (Plane : Type) [has_plane : has_plane Plane]
variable (O_a_b c : Plane)
variable [has_plane_plane : has_plane_plane O_a_b c]
variable (AB : Line)
variable (A_in_O_bc : A ∈ face O b c)
variable (B_in_O_ac : B ∈ face O a c)
variable (P_in_AB : Point)

-- Part (a)
theorem part_a_intersection (O a b c A B : Point) (A_in_O_bc : A ∈ face O b c) (B_in_O_ac : B ∈ face O a c) (AB : Line) : 
  ∃ P, intersection (line_through A B) (plane_through O a b) = Some P := 
by
  sorry

variable (A B C : Point)
variable (O_a_b O_b_c O_a_c : Plane)
variable (A_in_O_bc : A ∈ face O b c)
variable (B_in_O_ac : B ∈ face O a c)
variable (C_in_O_ab : C ∈ face O a b)

-- Part (b)
theorem part_b_section (O a b c A B C: Point) (A_in_O_bc : A ∈ face O b c) (B_in_O_ac : B ∈ face O a c) (C_in_O_ab : C ∈ face O a b) : 
  ∃ Q, intersection (plane_through P C) (plane_passing_through A B C) = Some Q :=
by
  sorry

end ThreeFacedAngle

end part_a_intersection_part_b_section_l250_250731


namespace sum_first_five_terms_l250_250355

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem sum_first_five_terms (a1 q : ℝ) 
  (h1 : geometric_sequence a1 q 2 * geometric_sequence a1 q 3 = 2 * a1)
  (h2 : (geometric_sequence a1 q 4 + 2 * geometric_sequence a1 q 7) / 2 = 5 / 4)
  : sum_geometric_sequence a1 q 5 = 31 :=
sorry

end sum_first_five_terms_l250_250355


namespace area_square_diagonal_l250_250421

theorem area_square_diagonal (d : ℝ) (k : ℝ) :
  (∀ side : ℝ, d^2 = 2 * side^2 → side^2 = (d^2)/2) →
  (∀ A : ℝ, A = (d^2)/2 → A = k * d^2) →
  k = 1/2 :=
by
  intros h1 h2
  sorry

end area_square_diagonal_l250_250421


namespace expected_value_greater_than_median_l250_250595

noncomputable def density_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, x < a → f x = 0) ∧ 
  (∀ x, x ≥ b → f x = 0) ∧
  (∀ x, a ≤ x → x < b → f x > 0) ∧
  (∀ x y, a ≤ x → x ≤ y → y < b → f y ≤ f x)


theorem expected_value_greater_than_median
  (X : Type) [Probability.ZeroOneOneClassDensity X]
  (f : ℝ → ℝ) (a b : ℝ)
  (h1 : density_function f a b)
  (h2 : ∃ x, is_median X x) :
  expected_value X > median X := 
sorry

end expected_value_greater_than_median_l250_250595


namespace boy_walking_speed_l250_250740

theorem boy_walking_speed 
  (travel_rate : ℝ) 
  (total_journey_time : ℝ) 
  (distance : ℝ) 
  (post_office_time : ℝ) 
  (walking_back_time : ℝ) 
  (walking_speed : ℝ): 
  travel_rate = 12.5 ∧ 
  total_journey_time = 5 + 48/60 ∧ 
  distance = 9.999999999999998 ∧ 
  post_office_time = distance / travel_rate ∧ 
  walking_back_time = total_journey_time - post_office_time ∧ 
  walking_speed = distance / walking_back_time 
  → walking_speed = 2 := 
by 
  intros h;
  sorry

end boy_walking_speed_l250_250740


namespace correct_calculation_l250_250256

theorem correct_calculation (x : ℝ) (h : 5.46 - x = 3.97) : 5.46 + x = 6.95 := by
  sorry

end correct_calculation_l250_250256


namespace bisectors_form_inscribed_quadrilateral_l250_250691

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end bisectors_form_inscribed_quadrilateral_l250_250691


namespace solve_inequality_l250_250901

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2)
  (h3 : (x^2 + 3*x - 1) / (4 - x^2) < 1)
  (h4 : (x^2 + 3*x - 1) / (4 - x^2) ≥ -1) :
  x < -5 / 2 ∨ (-1 ≤ x ∧ x < 1) :=
by sorry

end solve_inequality_l250_250901


namespace linda_loan_interest_difference_l250_250864

theorem linda_loan_interest_difference :
  let P : ℝ := 8000
  let r : ℝ := 0.10
  let t : ℕ := 3
  let n_monthly : ℕ := 12
  let n_annual : ℕ := 1
  let A_monthly : ℝ := P * (1 + r / (n_monthly : ℝ))^(n_monthly * t)
  let A_annual : ℝ := P * (1 + r)^t
  A_monthly - A_annual = 151.07 :=
by
  sorry

end linda_loan_interest_difference_l250_250864


namespace solution_of_system_l250_250136

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l250_250136


namespace largest_x_l250_250777

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l250_250777


namespace area_of_rectangular_field_l250_250854

theorem area_of_rectangular_field (length width : ℝ) (h_length: length = 5.9) (h_width: width = 3) : 
  length * width = 17.7 := 
by
  sorry

end area_of_rectangular_field_l250_250854


namespace initially_caught_and_tagged_is_30_l250_250265

open Real

-- Define conditions
def total_second_catch : ℕ := 50
def tagged_second_catch : ℕ := 2
def total_pond_fish : ℕ := 750

-- Define ratio condition
def ratio_condition (T : ℕ) : Prop :=
  (T : ℝ) / (total_pond_fish : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ)

-- Prove the number of fish initially caught and tagged is 30
theorem initially_caught_and_tagged_is_30 :
  ∃ T : ℕ, ratio_condition T ∧ T = 30 :=
by
  -- Skipping proof
  sorry

end initially_caught_and_tagged_is_30_l250_250265


namespace probability_four_different_numbers_l250_250167

theorem probability_four_different_numbers :
  let total_balls := 10
  let total_ways_to_choose_four := nat.choose total_balls 4
  let ways_to_choose_four_unique_numbers := nat.choose 5 4
  let ways_to_choose_colors := 2 ^ 4
  let successful_outcomes := ways_to_choose_four_unique_numbers * ways_to_choose_colors
  let probability : ℚ := successful_outcomes / total_ways_to_choose_four
  probability = 8 / 21 :=
by
  sorry

end probability_four_different_numbers_l250_250167


namespace train_speed_l250_250748

noncomputable def jogger_speed : ℝ := 9 -- speed in km/hr
noncomputable def jogger_distance : ℝ := 150 / 1000 -- distance in km
noncomputable def train_length : ℝ := 100 / 1000 -- length in km
noncomputable def time_to_pass : ℝ := 25 -- time in seconds

theorem train_speed 
  (v_j : ℝ := jogger_speed)
  (d_j : ℝ := jogger_distance)
  (L : ℝ := train_length)
  (t : ℝ := time_to_pass) :
  (train_speed_in_kmh : ℝ) = 36 :=
by 
  sorry

end train_speed_l250_250748


namespace BoatsRUs_canoes_l250_250472

theorem BoatsRUs_canoes :
  let a := 6
  let r := 3
  let n := 5
  let S := a * (r^n - 1) / (r - 1)
  S = 726 := by
  -- Proof
  sorry

end BoatsRUs_canoes_l250_250472


namespace abcd_mod_7_zero_l250_250680

theorem abcd_mod_7_zero
  (a b c d : ℕ)
  (h1 : a + 2 * b + 3 * c + 4 * d ≡ 1 [MOD 7])
  (h2 : 2 * a + 3 * b + c + 2 * d ≡ 5 [MOD 7])
  (h3 : 3 * a + b + 2 * c + 3 * d ≡ 3 [MOD 7])
  (h4 : 4 * a + 2 * b + d + c ≡ 2 [MOD 7])
  (ha : a < 7) (hb : b < 7) (hc : c < 7) (hd : d < 7) :
  (a * b * c * d) % 7 = 0 :=
by sorry

end abcd_mod_7_zero_l250_250680


namespace triangle_properties_l250_250379

open Real

noncomputable def vec_m (a : ℝ) : ℝ × ℝ := (2 * sin (a / 2), sqrt 3)
noncomputable def vec_n (a : ℝ) : ℝ × ℝ := (cos a, 2 * cos (a / 4)^2 - 1)
noncomputable def area_triangle := 3 * sqrt 3 / 2

theorem triangle_properties (a b c : ℝ) (A : ℝ)
  (ha : a = sqrt 7)
  (hA : (1 / 2) * b * c * sin A = area_triangle)
  (hparallel : vec_m A = vec_n A) :
  A = π / 3 ∧ b + c = 5 :=
by
  sorry

end triangle_properties_l250_250379


namespace divisible_by_9_l250_250737

-- Definition of the sum of digits function S
def sum_of_digits (n : ℕ) : ℕ := sorry  -- Assume we have a function that sums the digits of n

theorem divisible_by_9 (a : ℕ) (h₁ : sum_of_digits a = sum_of_digits (2 * a)) 
  (h₂ : a % 9 = sum_of_digits a % 9) (h₃ : (2 * a) % 9 = sum_of_digits (2 * a) % 9) : 
  a % 9 = 0 :=
by
  sorry

end divisible_by_9_l250_250737


namespace no_grasshopper_at_fourth_vertex_l250_250169

-- Definitions based on given conditions
def is_vertex_of_square (x : ℝ) (y : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1)

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

def leapfrog_jump (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2)

-- Problem statement
theorem no_grasshopper_at_fourth_vertex (a b c : ℝ × ℝ) :
  is_vertex_of_square a.1 a.2 ∧ is_vertex_of_square b.1 b.2 ∧ is_vertex_of_square c.1 c.2 →
  ∃ d : ℝ × ℝ, is_vertex_of_square d.1 d.2 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c →
  ∀ (n : ℕ) (pos : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), (pos 0 a b = leapfrog_jump a b) ∧
    (pos n a b = leapfrog_jump (pos (n-1) a b) (pos (n-1) b c)) →
    (pos n a b).1 ≠ (d.1) ∨ (pos n a b).2 ≠ (d.2) :=
sorry

end no_grasshopper_at_fourth_vertex_l250_250169


namespace probability_both_visible_l250_250936

noncomputable def emma_lap_time : ℕ := 100
noncomputable def ethan_lap_time : ℕ := 75
noncomputable def start_time : ℕ := 0
noncomputable def photo_start_minute : ℕ := 12 * 60 -- converted to seconds
noncomputable def photo_end_minute : ℕ := 13 * 60 -- converted to seconds
noncomputable def photo_visible_angle : ℚ := 1 / 3

theorem probability_both_visible :
  ∀ start_time photo_start_minute photo_end_minute emma_lap_time ethan_lap_time photo_visible_angle,
  start_time = 0 →
  photo_start_minute = 12 * 60 →
  photo_end_minute = 13 * 60 →
  emma_lap_time = 100 →
  ethan_lap_time = 75 →
  photo_visible_angle = 1 / 3 →
  (∃ t, photo_start_minute ≤ t ∧ t < photo_end_minute ∧
        (t % emma_lap_time ≤ (photo_visible_angle * emma_lap_time) / 2 ∨
         t % emma_lap_time ≥ emma_lap_time - (photo_visible_angle * emma_lap_time) / 2) ∧
        (t % ethan_lap_time ≤ (photo_visible_angle * ethan_lap_time) / 2 ∨
         t % ethan_lap_time ≥ ethan_lap_time - (photo_visible_angle * ethan_lap_time) / 2)) ↔
  true :=
sorry

end probability_both_visible_l250_250936


namespace total_bill_l250_250556

theorem total_bill (n : ℝ) (h : 9 * (n / 10 + 3) = n) : n = 270 := 
sorry

end total_bill_l250_250556


namespace transformed_quadratic_equation_l250_250072

theorem transformed_quadratic_equation (u v: ℝ) :
  (u + v = -5 / 2) ∧ (u * v = 3 / 2) ↔ (∃ y : ℝ, y^2 - y + 6 = 0) := sorry

end transformed_quadratic_equation_l250_250072


namespace solution_of_system_l250_250133

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l250_250133


namespace stacy_days_to_finish_l250_250992

-- Definitions based on the conditions
def total_pages : ℕ := 81
def pages_per_day : ℕ := 27

-- The theorem statement
theorem stacy_days_to_finish : total_pages / pages_per_day = 3 := by
  -- the proof is omitted
  sorry

end stacy_days_to_finish_l250_250992


namespace find_E_l250_250459

variable (A H C S M N E : ℕ)
variable (x y z l : ℕ)

theorem find_E (h1 : A * x + H * y + C * z = l)
 (h2 : S * x + M * y + N * z = l)
 (h3 : E * x = l)
 (h4 : A ≠ S ∧ A ≠ H ∧ A ≠ C ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧ H ≠ C ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧ M ≠ N ∧ M ≠ E ∧ N ≠ E)
 : E = (A * M + C * N - S * H - N * H) / (M + N - H) := 
sorry

end find_E_l250_250459


namespace computation_equal_l250_250199

theorem computation_equal (a b c d : ℕ) (inv : ℚ → ℚ) (mul : ℚ → ℕ → ℚ) : 
  a = 3 → b = 1 → c = 6 → d = 2 → 
  inv ((a^b - d + c^2 + b) : ℚ) * 6 = (3 / 19) := by
  intros ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end computation_equal_l250_250199


namespace find_s_l250_250681

noncomputable def f (s : ℝ) :=
  Polynomial.monicHorner [1, -(s + 2 + s + 8), (s + 2) * (s + 8), 0, 0]

noncomputable def g (s : ℝ) :=
  Polynomial.monicHorner [1, -(s + 4 + s + 10), (s + 4) * (s + 10), 0, 0]

theorem find_s :
  ∃ s : ℝ, s = 12 ∧ ∀ x : ℝ, f s x - g s x = 2 * s :=
  by
    use 12
    sorry

end find_s_l250_250681


namespace part1_part2_l250_250744

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l250_250744


namespace estimate_students_height_at_least_165_l250_250816

theorem estimate_students_height_at_least_165 
  (sample_size : ℕ)
  (total_school_size : ℕ)
  (students_165_170 : ℕ)
  (students_170_175 : ℕ)
  (h_sample : sample_size = 100)
  (h_total_school : total_school_size = 1000)
  (h_students_165_170 : students_165_170 = 20)
  (h_students_170_175 : students_170_175 = 30)
  : (students_165_170 + students_170_175) * (total_school_size / sample_size) = 500 := 
by
  sorry

end estimate_students_height_at_least_165_l250_250816


namespace arithmetic_sequence_problem_l250_250973

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) -- condition for arithmetic sequence
  (h_condition : a 3 + a 5 + a 7 + a 9 + a 11 = 100) : 
  3 * a 9 - a 13 = 40 :=
sorry

end arithmetic_sequence_problem_l250_250973


namespace largest_x_eq_48_div_7_l250_250803

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l250_250803


namespace relationship_x_y_l250_250497

theorem relationship_x_y (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : x = Real.sqrt ((a - b) * (b - c))) (h₃ : y = (a - c) / 2) : 
  x ≤ y :=
by
  sorry

end relationship_x_y_l250_250497


namespace power_function_value_at_quarter_l250_250299

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem power_function_value_at_quarter (α : ℝ) (h : f 4 α = 1 / 2) : f (1 / 4) α = 2 := 
  sorry

end power_function_value_at_quarter_l250_250299


namespace value_of_4_Y_3_l250_250506

def Y (a b : ℕ) : ℕ := (2 * a ^ 2 - 3 * a * b + b ^ 2) ^ 2

theorem value_of_4_Y_3 : Y 4 3 = 25 := by
  sorry

end value_of_4_Y_3_l250_250506


namespace find_m_eq_zero_l250_250392

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l250_250392


namespace average_speed_correct_l250_250031

noncomputable def total_distance := 120 + 70
noncomputable def total_time := 2
noncomputable def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed = 95 := by
  sorry

end average_speed_correct_l250_250031


namespace possible_values_of_m_l250_250255

theorem possible_values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  (∀ x, x ∈ B → x ∈ A) ↔ m = 0 ∨ m = -1 ∨ m = -1 / 3 :=
by
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  sorry -- Proof needed

end possible_values_of_m_l250_250255


namespace evaluate_expression_l250_250619

theorem evaluate_expression (k : ℤ): 
  2^(-(3*k+1)) - 2^(-(3*k-2)) + 2^(-(3*k)) - 2^(-(3*k+3)) = -((21:ℚ)/(8:ℚ)) * 2^(-(3*k)) := 
by 
  sorry

end evaluate_expression_l250_250619


namespace product_of_three_consecutive_cubes_divisible_by_504_l250_250883

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℤ) : 
  ∃ k : ℤ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by
  -- Proof omitted
  sorry

end product_of_three_consecutive_cubes_divisible_by_504_l250_250883


namespace nalani_net_amount_l250_250287

-- Definitions based on the conditions
def luna_birth := 10 -- Luna gave birth to 10 puppies
def stella_birth := 14 -- Stella gave birth to 14 puppies
def luna_sold := 8 -- Nalani sold 8 puppies from Luna's litter
def stella_sold := 10 -- Nalani sold 10 puppies from Stella's litter
def luna_price := 200 -- Price per puppy for Luna's litter is $200
def stella_price := 250 -- Price per puppy for Stella's litter is $250
def luna_cost := 80 -- Cost of raising each puppy from Luna's litter is $80
def stella_cost := 90 -- Cost of raising each puppy from Stella's litter is $90

-- Theorem stating the net amount received by Nalani
theorem nalani_net_amount : 
        luna_sold * luna_price + stella_sold * stella_price - 
        (luna_birth * luna_cost + stella_birth * stella_cost) = 2040 :=
by 
  sorry

end nalani_net_amount_l250_250287


namespace axis_of_symmetry_parabola_l250_250346

theorem axis_of_symmetry_parabola (x y : ℝ) :
  y = - (1 / 8) * x^2 → y = 2 :=
sorry

end axis_of_symmetry_parabola_l250_250346


namespace fraction_of_milk_in_second_cup_l250_250380

noncomputable def ratio_mixture (V: ℝ) (x: ℝ) :=
  ((2 / 5 * V + (1 - x) * V) / (3 / 5 * V + x * V))

theorem fraction_of_milk_in_second_cup
  (V: ℝ) 
  (hV: V > 0)
  (hx: ratio_mixture V x = 3 / 7) :
  x = 4 / 5 :=
by
  sorry

end fraction_of_milk_in_second_cup_l250_250380


namespace required_barrels_of_pitch_l250_250457

def total_road_length : ℕ := 16
def bags_of_gravel_per_truckload : ℕ := 2
def barrels_of_pitch_per_truckload (bgt : ℕ) : ℚ := bgt / 5
def truckloads_per_mile : ℕ := 3

def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := (miles_paved_day1 * 2) - 1
def total_miles_paved_first_two_days : ℕ := miles_paved_day1 + miles_paved_day2
def remaining_miles_paved_day3 : ℕ := total_road_length - total_miles_paved_first_two_days

def truckloads_needed (miles : ℕ) : ℕ := miles * truckloads_per_mile
def barrels_of_pitch_needed (truckloads : ℕ) (bgt : ℕ) : ℚ := truckloads * barrels_of_pitch_per_truckload bgt

theorem required_barrels_of_pitch : 
  barrels_of_pitch_needed (truckloads_needed remaining_miles_paved_day3) bags_of_gravel_per_truckload = 6 := 
by
  sorry

end required_barrels_of_pitch_l250_250457


namespace max_notebooks_15_dollars_l250_250466

noncomputable def max_notebooks (money : ℕ) : ℕ :=
  let cost_individual   := 2
  let cost_pack_4       := 6
  let cost_pack_7       := 9
  let notebooks_budget  := 15
  if money >= 9 then 
    7 + max_notebooks (money - 9)
  else if money >= 6 then 
    4 + max_notebooks (money - 6)
  else 
    money / 2

theorem max_notebooks_15_dollars : max_notebooks 15 = 11 :=
by
  sorry

end max_notebooks_15_dollars_l250_250466


namespace helen_owes_more_l250_250092

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value_semiannually : ℝ :=
  future_value 8000 0.10 2 3

noncomputable def future_value_annually : ℝ :=
  8000 * (1 + 0.10) ^ 3

noncomputable def difference : ℝ :=
  future_value_semiannually - future_value_annually

theorem helen_owes_more : abs (difference - 72.80) < 0.01 :=
by
  sorry

end helen_owes_more_l250_250092


namespace complement_of_A_in_U_l250_250965

namespace SetTheory

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by
  sorry

end SetTheory

end complement_of_A_in_U_l250_250965


namespace polygon_area_is_12_l250_250068

def polygon_vertices := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

def area_of_polygon (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to compute the area (stub here for now)
  sorry

theorem polygon_area_is_12 :
  area_of_polygon polygon_vertices = 12 :=
by
  sorry

end polygon_area_is_12_l250_250068


namespace inequality_holds_for_positive_reals_l250_250278

theorem inequality_holds_for_positive_reals (x y : ℝ) (m n : ℤ) 
  (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (1 - x^n)^m + (1 - y^m)^n ≥ 1 :=
sorry

end inequality_holds_for_positive_reals_l250_250278


namespace hockey_season_duration_l250_250718

theorem hockey_season_duration 
  (total_games : ℕ)
  (games_per_month : ℕ)
  (h_total : total_games = 182)
  (h_monthly : games_per_month = 13) : 
  total_games / games_per_month = 14 := 
by
  sorry

end hockey_season_duration_l250_250718


namespace solution_set_l250_250130

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l250_250130


namespace profit_ratio_l250_250870

noncomputable def effective_capital (investment : ℕ) (months : ℕ) : ℕ := investment * months

theorem profit_ratio : 
  let P_investment := 4000
  let P_months := 12
  let Q_investment := 9000
  let Q_months := 8
  let P_effective := effective_capital P_investment P_months
  let Q_effective := effective_capital Q_investment Q_months
  (P_effective / Nat.gcd P_effective Q_effective) = 2 ∧ (Q_effective / Nat.gcd P_effective Q_effective) = 3 :=
sorry

end profit_ratio_l250_250870


namespace contrapositive_of_ab_eq_zero_l250_250412

theorem contrapositive_of_ab_eq_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → ab ≠ 0 :=
by
  sorry

end contrapositive_of_ab_eq_zero_l250_250412


namespace smallest_integer_base_cube_l250_250310

theorem smallest_integer_base_cube (b : ℤ) (h1 : b > 5) (h2 : ∃ k : ℤ, 1 * b + 2 = k^3) : b = 6 :=
sorry

end smallest_integer_base_cube_l250_250310


namespace not_divisible_67_l250_250119

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l250_250119


namespace number_of_boxes_initially_l250_250289

theorem number_of_boxes_initially (B : ℕ) (h1 : ∃ B, 8 * B - 17 = 15) : B = 4 :=
  by
  sorry

end number_of_boxes_initially_l250_250289


namespace largest_x_satisfies_condition_l250_250789

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l250_250789


namespace proof_intersection_l250_250252

def setA : Set ℤ := {x | abs x ≤ 2}

def setB : Set ℝ := {x | x^2 - 2 * x - 8 ≥ 0}

def complementB : Set ℝ := {x | x^2 - 2 * x - 8 < 0}

def intersectionAComplementB : Set ℤ := {x | x ∈ setA ∧ (x : ℝ) ∈ complementB}

theorem proof_intersection : intersectionAComplementB = {-1, 0, 1, 2} := by
  sorry

end proof_intersection_l250_250252


namespace triangle_isosceles_of_sin_condition_l250_250852

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_isosceles_of_sin_condition {A B C : ℝ} (h : 2 * Real.sin A * Real.cos B = Real.sin C) : 
  isosceles_triangle A B C :=
by
  sorry

end triangle_isosceles_of_sin_condition_l250_250852


namespace midpoint_of_diagonal_l250_250873

-- Definition of the points
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (14, 9)

-- Statement about the midpoint of a diagonal in a rectangle
theorem midpoint_of_diagonal : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1) = point1 → (x2, y2) = point2 →
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y) = (8, 3) :=
by
  intros
  sorry

end midpoint_of_diagonal_l250_250873


namespace leah_coins_worth_89_cents_l250_250383

variables (p n d : ℕ)

theorem leah_coins_worth_89_cents (h1 : p + n + d = 15) (h2 : d - 1 = n) : 
  1 * p + 5 * n + 10 * d = 89 := 
sorry

end leah_coins_worth_89_cents_l250_250383


namespace price_reduction_relationship_l250_250436

variable (a : ℝ) -- original price a in yuan
variable (b : ℝ) -- final price b in yuan

-- condition: price decreased by 10% first
def priceAfterFirstReduction := a * (1 - 0.10)

-- condition: price decreased by 20% on the result of the first reduction
def finalPrice := priceAfterFirstReduction a * (1 - 0.20)

-- theorem: relationship between original price a and final price b
theorem price_reduction_relationship (h : b = finalPrice a) : 
  b = a * (1 - 0.10) * (1 - 0.20) :=
by
  -- proof would go here
  sorry

end price_reduction_relationship_l250_250436


namespace probability_sum_does_not_exceed_8_l250_250580

-- Definitions for the conditions
def uniform_die : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6} -- A uniform die with faces from 1 to 6

-- Resulting probability that we need to prove
theorem probability_sum_does_not_exceed_8 :
  (∑ n1 in finset.univ.filter (λ n1 : uniform_die, ∑ n2 in finset.univ.filter (λ n2 : uniform_die, n1 ≠ n2), 
         ∑ n3 in finset.univ.filter (λ n3 : uniform_die, n1 ≠ n3 ∧ n2 ≠ n3),
            if (n1.1 + n2.1 + n3.1 ≤ 8) then 1 else 0).to_real) = (∑ _ in finset.univ, 1).to_real * (1/5 : ℝ) :=
sorry

end probability_sum_does_not_exceed_8_l250_250580


namespace sum_of_remainders_and_parity_l250_250368

theorem sum_of_remainders_and_parity 
  (n : ℤ) 
  (h₀ : n % 20 = 13) : 
  (n % 4 + n % 5 = 4) ∧ (n % 2 = 1) :=
by
  sorry

end sum_of_remainders_and_parity_l250_250368


namespace valbonne_middle_school_l250_250295

theorem valbonne_middle_school (students : Finset ℕ) (h : students.card = 367) :
  ∃ (date1 date2 : ℕ), date1 ≠ date2 ∧ date1 = date2 ∧ date1 ∈ students ∧ date2 ∈ students :=
by {
  sorry
}

end valbonne_middle_school_l250_250295


namespace oblique_projection_intuitive_diagrams_correct_l250_250894

-- Definitions based on conditions
structure ObliqueProjection :=
  (lines_parallel_x_axis_same_length : Prop)
  (lines_parallel_y_axis_halved_length : Prop)
  (perpendicular_relationship_becomes_45_angle : Prop)

-- Definitions based on statements
def intuitive_triangle_projection (P : ObliqueProjection) : Prop :=
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_parallelogram_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_square_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_rhombus_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

-- Theorem stating which intuitive diagrams are correctly represented under the oblique projection method.
theorem oblique_projection_intuitive_diagrams_correct : 
  ∀ (P : ObliqueProjection), 
    intuitive_triangle_projection P ∧ 
    intuitive_parallelogram_projection P ∧
    ¬intuitive_square_projection P ∧
    ¬intuitive_rhombus_projection P :=
by 
  sorry

end oblique_projection_intuitive_diagrams_correct_l250_250894


namespace intersection_height_correct_l250_250893

noncomputable def height_of_intersection (height1 height2 distance : ℝ) : ℝ :=
  let line1 (x : ℝ) := - (height1 / distance) * x + height1
  let line2 (x : ℝ) := - (height2 / distance) * x
  let x_intersect := - (height2 * distance) / (height1 - height2)
  line1 x_intersect

theorem intersection_height_correct :
  height_of_intersection 40 60 120 = 120 :=
by
  sorry

end intersection_height_correct_l250_250893


namespace total_crosswalk_lines_l250_250906

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l250_250906


namespace perfect_squares_ending_in_4_5_6_less_than_2000_l250_250663

theorem perfect_squares_ending_in_4_5_6_less_than_2000 :
  let squares := { n : ℕ | n * n < 2000 ∧ (n * n % 10 = 4 ∨ n * n % 10 = 5 ∨ n * n % 10 = 6) } in
  squares.card = 23 :=
by
  sorry

end perfect_squares_ending_in_4_5_6_less_than_2000_l250_250663


namespace length_of_common_chord_l250_250090

theorem length_of_common_chord (x y : ℝ) :
  (x + 1)^2 + (y - 3)^2 = 9 ∧ x^2 + y^2 - 4 * x + 2 * y - 11 = 0 → 
  ∃ l : ℝ, l = 24 / 5 :=
by
  sorry

end length_of_common_chord_l250_250090


namespace solve_equation_l250_250991

theorem solve_equation (x : ℝ) (h : (4 * x ^ 2 + 6 * x + 2) / (x + 2) = 4 * x + 7) : x = -4 / 3 :=
by
  sorry

end solve_equation_l250_250991


namespace widget_cost_reduction_l250_250984

theorem widget_cost_reduction (W R : ℝ) (h1 : 6 * W = 36) (h2 : 8 * (W - R) = 36) : R = 1.5 :=
by
  sorry

end widget_cost_reduction_l250_250984


namespace solution_set_line_l250_250125

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l250_250125


namespace slope_of_line_through_intersecting_points_of_circles_l250_250066

theorem slope_of_line_through_intersecting_points_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 6*x + 4*y - 5 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 10*x + 16*y + 24 = 0
  ∀ (C D : ℝ × ℝ), circle1 C.1 C.2 → circle2 C.1 C.2 → circle1 D.1 D.2 → circle2 D.1 D.2 → 
  let dx := D.1 - C.1
  let dy := D.2 - C.2
  dx ≠ 0 → dy / dx = 1 / 3 :=
by
  intros
  sorry

end slope_of_line_through_intersecting_points_of_circles_l250_250066


namespace percentage_boys_not_attended_college_l250_250465

/-
Define the constants and given conditions.
-/
def number_of_boys : ℕ := 300
def number_of_girls : ℕ := 240
def total_students : ℕ := number_of_boys + number_of_girls
def percentage_class_attended_college : ℝ := 0.70
def percentage_girls_not_attended_college : ℝ := 0.30

/-
The proof problem statement: 
Prove the percentage of the boys class that did not attend college.
-/
theorem percentage_boys_not_attended_college :
  let students_attended_college := percentage_class_attended_college * total_students
  let not_attended_college_students := total_students - students_attended_college
  let not_attended_college_girls := percentage_girls_not_attended_college * number_of_girls
  let not_attended_college_boys := not_attended_college_students - not_attended_college_girls
  let percentage_boys_not_attended_college := (not_attended_college_boys / number_of_boys) * 100
  percentage_boys_not_attended_college = 30 := by
  sorry

end percentage_boys_not_attended_college_l250_250465


namespace shelves_needed_number_of_shelves_l250_250868

-- Define the initial number of books
def initial_books : Float := 46.0

-- Define the number of additional books added by the librarian
def additional_books : Float := 10.0

-- Define the number of books each shelf can hold
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_books + additional_books

-- The mathematical proof statement for the number of shelves needed
theorem shelves_needed : Float := total_books / books_per_shelf

-- The required statement proving that the number of shelves needed is 14.0
theorem number_of_shelves : shelves_needed = 14.0 := by
  sorry

end shelves_needed_number_of_shelves_l250_250868


namespace initial_pants_l250_250615

theorem initial_pants (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) (total_pants : ℕ) 
  (h1 : pairs_per_year = 4) (h2 : pants_per_pair = 2) (h3 : years = 5) (h4 : total_pants = 90) : 
  ∃ (initial_pants : ℕ), initial_pants = total_pants - (pairs_per_year * pants_per_pair * years) :=
by
  use 50
  sorry

end initial_pants_l250_250615


namespace expected_balls_in_original_positions_six_l250_250550

noncomputable def expected_balls_in_original_positions :
  ℕ := 6

def probability_never_swapped :
  ℚ := (4 / 6) ^ 3

theorem expected_balls_in_original_positions_six :
  expected_balls_in_original_positions * probability_never_swapped = 48 / 27 :=
by 
  simp [expected_balls_in_original_positions, probability_never_swapped]
  norm_num
sorry

end expected_balls_in_original_positions_six_l250_250550


namespace negation_of_universal_to_existential_l250_250363

theorem negation_of_universal_to_existential :
  (¬(∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end negation_of_universal_to_existential_l250_250363


namespace range_of_m_l250_250641

open Real

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → log x ≤ x * exp (m^2 - m - 1)

theorem range_of_m : 
  {m : ℝ | satisfies_inequality m} = {m : ℝ | m ≤ 0 ∨ m ≥ 1} :=
by 
  sorry

end range_of_m_l250_250641


namespace find_siblings_l250_250420

-- Define the characteristics of each child
structure Child where
  name : String
  eyeColor : String
  hairColor : String
  age : Nat

-- List of children
def Olivia : Child := { name := "Olivia", eyeColor := "Green", hairColor := "Red", age := 12 }
def Henry  : Child := { name := "Henry", eyeColor := "Gray", hairColor := "Brown", age := 12 }
def Lucas  : Child := { name := "Lucas", eyeColor := "Green", hairColor := "Red", age := 10 }
def Emma   : Child := { name := "Emma", eyeColor := "Green", hairColor := "Brown", age := 12 }
def Mia    : Child := { name := "Mia", eyeColor := "Gray", hairColor := "Red", age := 10 }
def Noah   : Child := { name := "Noah", eyeColor := "Gray", hairColor := "Brown", age := 12 }

-- Define a family as a set of children who share at least one characteristic
def isFamily (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.eyeColor = c3.eyeColor ∨ c2.eyeColor = c3.eyeColor) ∨
  (c1.hairColor = c2.hairColor ∨ c1.hairColor = c3.hairColor ∨ c2.hairColor = c3.hairColor) ∨
  (c1.age = c2.age ∨ c1.age = c3.age ∨ c2.age = c3.age)

-- The main theorem
theorem find_siblings : isFamily Olivia Lucas Emma :=
by
  sorry

end find_siblings_l250_250420


namespace angle_part_a_angle_part_b_l250_250225

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2^2)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_part_a :
  angle_between_vectors (4, 0) (2, -2) = Real.arccos (Real.sqrt 2 / 2) :=
by
  sorry

theorem angle_part_b :
  angle_between_vectors (5, -3) (3, 5) = Real.pi / 2 :=
by
  sorry

end angle_part_a_angle_part_b_l250_250225


namespace solution_set_line_l250_250123

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l250_250123


namespace min_value_of_a_is_five_l250_250689

-- Given: a, b, c in table satisfying the conditions
-- We are to prove that the minimum value of a is 5.
theorem min_value_of_a_is_five
  {a b c: ℤ} (h_pos: 0 < a) (hx_distinct: 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
                               a*x₁^2 + b*x₁ + c = 0 ∧ 
                               a*x₂^2 + b*x₂ + c = 0) (hb_neg: b < 0) 
                               (h_disc_pos: (b^2 - 4*a*c) > 0) : a = 5 :=
sorry

end min_value_of_a_is_five_l250_250689


namespace johns_original_earnings_l250_250381

theorem johns_original_earnings (x : ℝ) (h1 : x + 0.5 * x = 90) : x = 60 := 
by
  -- sorry indicates the proof steps are omitted
  sorry

end johns_original_earnings_l250_250381


namespace volume_of_alcohol_correct_l250_250039

noncomputable def radius := 3 / 2 -- radius of the tank
noncomputable def total_height := 9 -- total height of the tank
noncomputable def full_solution_height := total_height / 3 -- height of the liquid when the tank is one-third full
noncomputable def volume := Real.pi * radius^2 * full_solution_height -- volume of liquid in the tank
noncomputable def alcohol_ratio := 1 / 6 -- ratio of alcohol to the total solution
noncomputable def volume_of_alcohol := volume * alcohol_ratio -- volume of alcohol in the tank

theorem volume_of_alcohol_correct : volume_of_alcohol = (9 / 8) * Real.pi :=
by
  -- Proof would go here
  sorry

end volume_of_alcohol_correct_l250_250039


namespace parabola_cubic_intersection_points_l250_250775

def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15

def cubic (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem parabola_cubic_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    p1 = (-1, 0) ∧ p2 = (1, -24) ∧ p3 = (9, 162) ∧
    parabola p1.1 = p1.2 ∧ cubic p1.1 = p1.2 ∧
    parabola p2.1 = p2.2 ∧ cubic p2.1 = p2.2 ∧
    parabola p3.1 = p3.2 ∧ cubic p3.1 = p3.2 :=
by {
  -- This is the statement
  sorry
}

end parabola_cubic_intersection_points_l250_250775


namespace cosine_difference_formula_l250_250083

theorem cosine_difference_formula
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < (Real.pi / 2))
  (h3 : Real.tan α = 2) :
  Real.cos (α - (Real.pi / 4)) = (3 * Real.sqrt 10) / 10 := 
by
  sorry

end cosine_difference_formula_l250_250083


namespace ones_digit_of_8_pow_47_l250_250022

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l250_250022


namespace solution_set_l250_250128

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l250_250128


namespace trapezoid_perimeter_l250_250925

noncomputable def semiCircularTrapezoidPerimeter (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2) : ℝ :=
-((x^2) / 8) + 2 * x + 32

theorem trapezoid_perimeter 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2)
  (r : ℝ) 
  (h_r : r = 8) 
  (AB : ℝ) 
  (h_AB : AB = 2 * r)
  (CD_on_circumference : true) :
  semiCircularTrapezoidPerimeter x hx = -((x^2) / 8) + 2 * x + 32 :=   
sorry

end trapezoid_perimeter_l250_250925


namespace largest_x_satisfies_condition_l250_250798

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l250_250798


namespace total_packs_l250_250686

noncomputable def robyn_packs : ℕ := 16
noncomputable def lucy_packs : ℕ := 19

theorem total_packs : robyn_packs + lucy_packs = 35 := by
  sorry

end total_packs_l250_250686


namespace chinese_character_symmetry_l250_250106

-- Definitions of the characters and their symmetry properties
def is_symmetric (ch : String) : Prop :=
  ch = "喜"

-- Hypotheses (conditions)
def option_A := "喜"
def option_B := "欢"
def option_C := "数"
def option_D := "学"

-- Lean statement to prove the symmetry
theorem chinese_character_symmetry :
  is_symmetric option_A ∧ 
  ¬ is_symmetric option_B ∧ 
  ¬ is_symmetric option_C ∧ 
  ¬ is_symmetric option_D :=
by
  sorry

end chinese_character_symmetry_l250_250106


namespace exponents_problem_l250_250913

theorem exponents_problem :
  5000 * (5000^9) * 2^(1000) = 5000^(10) * 2^(1000) := by sorry

end exponents_problem_l250_250913


namespace domain_of_composite_function_l250_250970

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → -1 ≤ x + 1) →
  (∀ x, -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3 → 0 ≤ x ∧ x ≤ 1) :=
by
  sorry

end domain_of_composite_function_l250_250970


namespace olympic_rings_area_l250_250401

theorem olympic_rings_area (d R r: ℝ) 
  (hyp_d : d = 12 * Real.sqrt 2) 
  (hyp_R : R = 11) 
  (hyp_r : r = 9) 
  (overlap_area : ∀ (i j : ℕ), i ≠ j → 592 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54): 
  592.0 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54 := 
by sorry

end olympic_rings_area_l250_250401


namespace geese_initial_formation_l250_250930

theorem geese_initial_formation (G : ℕ) 
  (h1 : G / 2 + 4 = 12) : G = 16 := 
sorry

end geese_initial_formation_l250_250930


namespace positive_difference_16_l250_250996

def avg_is_37 (y : ℤ) : Prop := (45 + y) / 2 = 37

def positive_difference (a b : ℤ) : ℤ := if a > b then a - b else b - a

theorem positive_difference_16 (y : ℤ) (h : avg_is_37 y) : positive_difference 45 y = 16 :=
by
  sorry

end positive_difference_16_l250_250996


namespace simplify_fraction_l250_250872

theorem simplify_fraction (e : ℤ) : 
  (∃ k : ℤ, e = 13 * k + 12) ↔ (∃ d : ℤ, d ≠ 1 ∧ (16 * e - 10) % d = 0 ∧ (10 * e - 3) % d = 0) :=
by
  sorry

end simplify_fraction_l250_250872


namespace find_equation_with_new_roots_l250_250961

variable {p q r s : ℝ}

theorem find_equation_with_new_roots 
  (h_eq : ∀ x, x^2 - p * x + q = 0 ↔ (x = r ∧ x = s))
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  : 
  ∀ x, (x^2 - ((q^2 + 1) * (p^2 - 2 * q) / q^2) * x + (q + 1/q)^2) = 0 ↔ 
       (x = r^2 + 1/(s^2) ∧ x = s^2 + 1/(r^2)) := 
sorry

end find_equation_with_new_roots_l250_250961


namespace least_xy_value_l250_250495

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l250_250495


namespace common_difference_is_7_l250_250103

-- Define the arithmetic sequence with common difference d
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the conditions
variables (a1 d : ℕ)

-- Define the conditions provided in the problem
def condition1 := (arithmetic_seq a1 d 3) + (arithmetic_seq a1 d 6) = 11
def condition2 := (arithmetic_seq a1 d 5) + (arithmetic_seq a1 d 8) = 39

-- Prove that the common difference d is 7
theorem common_difference_is_7 : condition1 a1 d → condition2 a1 d → d = 7 :=
by
  intros cond1 cond2
  sorry

end common_difference_is_7_l250_250103


namespace carnations_third_bouquet_l250_250723

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l250_250723


namespace scientific_notation_248000_l250_250701

theorem scientific_notation_248000 : (248000 : Float) = 2.48 * 10^5 := 
sorry

end scientific_notation_248000_l250_250701


namespace appointment_duration_l250_250933

-- Define the given conditions
def total_workday_hours : ℕ := 8
def permits_per_hour : ℕ := 50
def total_permits : ℕ := 100
def stamping_time : ℕ := total_permits / permits_per_hour
def appointment_time : ℕ := (total_workday_hours - stamping_time) / 2

-- State the theorem and ignore the proof part by adding sorry
theorem appointment_duration : appointment_time = 3 := by
  -- skipping the proof steps
  sorry

end appointment_duration_l250_250933


namespace AngiesClassGirlsCount_l250_250755

theorem AngiesClassGirlsCount (n_girls n_boys : ℕ) (total_students : ℕ)
  (h1 : n_girls = 2 * (total_students / 5))
  (h2 : n_boys = 3 * (total_students / 5))
  (h3 : n_girls + n_boys = 20)
  : n_girls = 8 :=
by
  sorry

end AngiesClassGirlsCount_l250_250755


namespace arithmetic_geometric_sequence_product_l250_250325

theorem arithmetic_geometric_sequence_product :
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a_n = a_1 + m * d) →
  (∀ n : ℕ, ∃ q : ℝ, ∀ m : ℕ, b_n = b_1 * q ^ m) →
  a_1 = 1 → a_2 = 2 →
  b_1 = 1 → b_2 = 2 →
  a_5 * b_5 = 80 :=
by
  sorry

end arithmetic_geometric_sequence_product_l250_250325


namespace bounds_of_F_and_G_l250_250080

noncomputable def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem bounds_of_F_and_G {a b c : ℝ}
  (hF0 : |F a b c 0| ≤ 1)
  (hF1 : |F a b c 1| ≤ 1)
  (hFm1 : |F a b c (-1)| ≤ 1) :
  (∀ x, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by
  sorry

end bounds_of_F_and_G_l250_250080


namespace factorize_expr1_factorize_expr2_l250_250948

theorem factorize_expr1 (x y : ℝ) : 
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2 * y) :=
by
  sorry

theorem factorize_expr2 (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) :=
by
  sorry

end factorize_expr1_factorize_expr2_l250_250948


namespace largest_real_solution_l250_250793

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l250_250793


namespace tom_total_payment_l250_250572

variable (apples_kg : ℕ := 8)
variable (apples_rate : ℕ := 70)
variable (mangoes_kg : ℕ := 9)
variable (mangoes_rate : ℕ := 65)
variable (oranges_kg : ℕ := 5)
variable (oranges_rate : ℕ := 50)
variable (bananas_kg : ℕ := 3)
variable (bananas_rate : ℕ := 30)
variable (discount_apples : ℝ := 0.10)
variable (discount_oranges : ℝ := 0.15)

def total_cost_apple : ℝ := apples_kg * apples_rate
def total_cost_mango : ℝ := mangoes_kg * mangoes_rate
def total_cost_orange : ℝ := oranges_kg * oranges_rate
def total_cost_banana : ℝ := bananas_kg * bananas_rate
def discount_apples_amount : ℝ := discount_apples * total_cost_apple
def discount_oranges_amount : ℝ := discount_oranges * total_cost_orange
def apples_after_discount : ℝ := total_cost_apple - discount_apples_amount
def oranges_after_discount : ℝ := total_cost_orange - discount_oranges_amount

theorem tom_total_payment :
  apples_after_discount + total_cost_mango + oranges_after_discount + total_cost_banana = 1391.5 := by
  sorry

end tom_total_payment_l250_250572


namespace liza_phone_bill_eq_70_l250_250985

theorem liza_phone_bill_eq_70 (initial_balance rent payment paycheck electricity internet final_balance phone_bill : ℝ)
  (h1 : initial_balance = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity = 117)
  (h5 : internet = 100)
  (h6 : final_balance = 1563)
  (h_balance_before_phone_bill : initial_balance - rent + paycheck - (electricity + internet) = 1633)
  (h_final_balance_def : 1633 - phone_bill = final_balance) :
  phone_bill = 70 := sorry

end liza_phone_bill_eq_70_l250_250985


namespace Carla_total_marbles_l250_250476

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem Carla_total_marbles : initial_marbles + bought_marbles = 321.0 := 
by 
  sorry

end Carla_total_marbles_l250_250476


namespace ratio_spaghetti_pizza_l250_250181

/-- Define the number of students who participated in the survey and their preferences --/
def students_surveyed : ℕ := 800
def lasagna_pref : ℕ := 150
def manicotti_pref : ℕ := 120
def ravioli_pref : ℕ := 180
def spaghetti_pref : ℕ := 200
def pizza_pref : ℕ := 150

/-- Prove the ratio of students who preferred spaghetti to those who preferred pizza is 4/3 --/
theorem ratio_spaghetti_pizza : (200 / 150 : ℚ) = 4 / 3 :=
by sorry

end ratio_spaghetti_pizza_l250_250181


namespace normal_dist_prob_geq_one_l250_250359

-- Given a random variable ξ following a normal distribution
-- with mean -1 and variance 6^2, with P(-3 ≤ ξ ≤ -1) = 0.4
variables {ξ : ℝ} (dist : Normal (-1) (6^2))

-- and the given condition P(-3 ≤ ξ ≤ -1) = 0.4
axiom prob_interval : dist.prob (-3 :.. -1) = 0.4

-- Prove that P(ξ ≥ 1) = 0.1
theorem normal_dist_prob_geq_one :
  dist.prob (1 :..) = 0.1 :=
sorry

end normal_dist_prob_geq_one_l250_250359


namespace yellow_chip_value_l250_250851

theorem yellow_chip_value
  (y b g : ℕ)
  (hb : b = g)
  (hchips : y^4 * (4 * b)^b * (5 * g)^g = 16000)
  (h4yellow : y = 2) :
  y = 2 :=
by {
  sorry
}

end yellow_chip_value_l250_250851


namespace gcd_1729_1768_l250_250428

theorem gcd_1729_1768 : Int.gcd 1729 1768 = 13 := by
  sorry

end gcd_1729_1768_l250_250428


namespace complex_number_on_imaginary_axis_l250_250703

theorem complex_number_on_imaginary_axis (a : ℝ) 
(h : ∃ z : ℂ, z = (a^2 - 2 * a) + (a^2 - a - 2) * Complex.I ∧ z.re = 0) : 
a = 0 ∨ a = 2 :=
by
  sorry

end complex_number_on_imaginary_axis_l250_250703


namespace largest_divisible_by_88_l250_250321

theorem largest_divisible_by_88 (n : ℕ) (h₁ : n = 9999) (h₂ : n % 88 = 55) : n - 55 = 9944 := by
  sorry

end largest_divisible_by_88_l250_250321


namespace sum_of_decimals_as_fraction_l250_250480

theorem sum_of_decimals_as_fraction :
  (0.2 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) + (0.000008 : ℝ) + (0.0000009 : ℝ) = 
  (2340087 / 10000000 : ℝ) :=
sorry

end sum_of_decimals_as_fraction_l250_250480


namespace profit_450_l250_250187

-- Define the conditions
def cost_per_garment : ℕ := 40
def wholesale_price : ℕ := 60

-- Define the piecewise function for wholesale price P
noncomputable def P (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then wholesale_price
  else if h : 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

-- Define the profit function L
noncomputable def L (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then (P x - cost_per_garment) * x
  else if h : 100 < x ∧ x ≤ 500 then (22 * x - x^2 / 50)
  else 0

-- State the theorem
theorem profit_450 : L 450 = 5850 :=
by
  sorry

end profit_450_l250_250187


namespace second_group_students_l250_250422

-- Define the number of groups and their respective sizes
def num_groups : ℕ := 4
def first_group_students : ℕ := 5
def third_group_students : ℕ := 7
def fourth_group_students : ℕ := 4
def total_students : ℕ := 24

-- Define the main theorem to prove
theorem second_group_students :
  (∃ second_group_students : ℕ,
    total_students = first_group_students + second_group_students + third_group_students + fourth_group_students ∧
    second_group_students = 8) :=
sorry

end second_group_students_l250_250422


namespace equation_solution_l250_250554

noncomputable def solveEquation (x : ℂ) : Prop :=
  -x^2 = (2*x + 4)/(x + 2)

theorem equation_solution (x : ℂ) (h : x ≠ -2) :
  solveEquation x ↔ x = -2 ∨ x = Complex.I * 2 ∨ x = - Complex.I * 2 :=
sorry

end equation_solution_l250_250554


namespace largest_x_satisfies_condition_l250_250801

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l250_250801


namespace area_of_triangle_with_given_sides_l250_250370

variable (a b c : ℝ)
variable (s : ℝ := (a + b + c) / 2)
variable (area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_of_triangle_with_given_sides (ha : a = 65) (hb : b = 60) (hc : c = 25) :
  area = 750 := by
  sorry

end area_of_triangle_with_given_sides_l250_250370


namespace intersection_point_x_value_l250_250304

theorem intersection_point_x_value :
  ∃ x y : ℚ, (y = 3 * x - 22) ∧ (3 * x + y = 100) ∧ (x = 20 + 1 / 3) := by
  sorry

end intersection_point_x_value_l250_250304


namespace carnations_third_bouquet_l250_250724

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l250_250724


namespace cooks_selection_l250_250375

theorem cooks_selection (total_people : ℕ) (specific_person : ℕ) (other_people : ℕ) 
                        (total_people_eq : total_people = 10)
                        (specific_person_inclusion : specific_person = 1)
                        (other_people_eq : other_people = 9) :
  (combinatorics.choose other_people 1) = 9 := by
  sorry

end cooks_selection_l250_250375


namespace other_divisor_l250_250483

theorem other_divisor (x : ℕ) (h1 : 266 % 33 = 2) (h2 : 266 % x = 2) : x = 132 :=
sorry

end other_divisor_l250_250483


namespace fraction_simplification_l250_250560

theorem fraction_simplification :
  (20 + 16 * 20) / (20 * 16) = 17 / 16 :=
by
  sorry

end fraction_simplification_l250_250560


namespace lesser_number_l250_250165

theorem lesser_number (x y : ℕ) (h1 : x + y = 58) (h2 : x - y = 6) : y = 26 :=
by
  sorry

end lesser_number_l250_250165


namespace count_valid_pairs_l250_250623

theorem count_valid_pairs :
  let S := {a | 10 ≤ a ∧ a ≤ 30} in
  (∃ count : ℕ, count = 11 ∧ 
   ∃ f : ℕ × ℕ → Prop, 
   (∀ (a b : ℕ), (a ∈ S) ∧ (b ∈ S) → f (a, b)) ∧
   (∀ (a b : ℕ), f (a, b) ↔ (Nat.gcd a b + Nat.lcm a b = a + b)) ∧
   (count = S.card)) :=
sorry

end count_valid_pairs_l250_250623


namespace quadratic_root_l250_250350

theorem quadratic_root (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x^2 + x - 2 = 0) → a = 1 := by
  sorry

end quadratic_root_l250_250350


namespace solve_absolute_value_equation_l250_250152

theorem solve_absolute_value_equation (x : ℝ) :
  |2 * x - 3| = x + 1 → (x = 4 ∨ x = 2 / 3) := by
  sorry

end solve_absolute_value_equation_l250_250152


namespace average_goals_increase_l250_250747

theorem average_goals_increase (A : ℚ) (h1 : 4 * A + 2 = 4) : (4 / 5 - A) = 0.3 := by
  sorry

end average_goals_increase_l250_250747


namespace largest_whole_number_less_than_100_l250_250162

theorem largest_whole_number_less_than_100 (x : ℕ) (h1 : 7 * x < 100) (h_max : ∀ y : ℕ, 7 * y < 100 → y ≤ x) :
  x = 14 := 
sorry

end largest_whole_number_less_than_100_l250_250162


namespace intersect_single_point_l250_250669

theorem intersect_single_point (k : ℝ) :
  (∃ x : ℝ, (x^2 + k * x + 1 = 0) ∧
   ∀ x y : ℝ, (x^2 + k * x + 1 = 0 → y^2 + k * y + 1 = 0 → x = y))
  ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end intersect_single_point_l250_250669


namespace solve_equation_l250_250553

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end solve_equation_l250_250553


namespace find_number_l250_250837

theorem find_number (x : ℝ) (h : (1/2) * x + 7 = 17) : x = 20 :=
sorry

end find_number_l250_250837


namespace angle_B_is_180_l250_250116

variables {l k : Line} {A B C: Point}

def parallel (l k : Line) : Prop := sorry 
def angle (A B C : Point) : ℝ := sorry

theorem angle_B_is_180 (h1 : parallel l k) (h2 : angle A = 110) (h3 : angle C = 70) :
  angle B = 180 := 
by
  sorry

end angle_B_is_180_l250_250116


namespace binom_computation_l250_250208

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l250_250208


namespace minimum_employees_needed_l250_250159

-- Conditions
def water_monitors : ℕ := 95
def air_monitors : ℕ := 80
def soil_monitors : ℕ := 45
def water_and_air : ℕ := 30
def air_and_soil : ℕ := 20
def water_and_soil : ℕ := 15
def all_three : ℕ := 10

-- Theorems/Goals
theorem minimum_employees_needed 
  (water : ℕ := water_monitors)
  (air : ℕ := air_monitors)
  (soil : ℕ := soil_monitors)
  (water_air : ℕ := water_and_air)
  (air_soil : ℕ := air_and_soil)
  (water_soil : ℕ := water_and_soil)
  (all_3 : ℕ := all_three) :
  water + air + soil - water_air - air_soil - water_soil + all_3 = 165 :=
by
  sorry

end minimum_employees_needed_l250_250159


namespace unique_positive_solution_l250_250634

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l250_250634


namespace cottonCandyToPopcornRatio_l250_250917

variable (popcornEarningsPerDay : ℕ) (netEarnings : ℕ) (rentCost : ℕ) (ingredientCost : ℕ)

theorem cottonCandyToPopcornRatio
  (h_popcorn : popcornEarningsPerDay = 50)
  (h_net : netEarnings = 895)
  (h_rent : rentCost = 30)
  (h_ingredient : ingredientCost = 75)
  (h : ∃ C : ℕ, 5 * C + 5 * popcornEarningsPerDay - rentCost - ingredientCost = netEarnings) :
  ∃ r : ℕ, r = 3 :=
by
  sorry

end cottonCandyToPopcornRatio_l250_250917


namespace least_integer_value_l250_250914

theorem least_integer_value (x : ℝ) (h : |3 * x - 4| ≤ 25) : x = -7 :=
sorry

end least_integer_value_l250_250914


namespace largest_x_exists_largest_x_largest_real_number_l250_250784

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l250_250784


namespace integer_pairs_prime_P_l250_250532

theorem integer_pairs_prime_P (P : ℕ) (hP_prime : Prime P) 
  (h_condition : ∃ a b : ℤ, |a + b| + (a - b)^2 = P) : 
  P = 2 ∧ ((∃ a b : ℤ, |a + b| = 2 ∧ a - b = 0) ∨ 
           (∃ a b : ℤ, |a + b| = 1 ∧ (a - b = 1 ∨ a - b = -1))) :=
by
  sorry

end integer_pairs_prime_P_l250_250532


namespace medicine_supply_duration_l250_250676

theorem medicine_supply_duration
  (pills_per_three_days : ℚ := 1 / 3)
  (total_pills : ℕ := 60)
  (days_per_month : ℕ := 30) :
  (((total_pills : ℚ) * ( 3 / pills_per_three_days)) / days_per_month) = 18 := sorry

end medicine_supply_duration_l250_250676


namespace unique_positive_solution_l250_250635

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l250_250635


namespace geom_arith_seq_l250_250105

theorem geom_arith_seq (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_arith : 2 * a 3 - (a 5 / 2) = (a 5 / 2) - 3 * a 1) (hq : q > 0) :
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 :=
by
  sorry

end geom_arith_seq_l250_250105


namespace more_action_figures_than_books_l250_250521

-- Definitions of initial conditions
def books : ℕ := 3
def initial_action_figures : ℕ := 4
def added_action_figures : ℕ := 2

-- Definition of final number of action figures
def final_action_figures : ℕ := initial_action_figures + added_action_figures

-- Proposition to be proved
theorem more_action_figures_than_books : final_action_figures - books = 3 := by
  -- We leave the proof empty
  sorry

end more_action_figures_than_books_l250_250521


namespace student_tickets_sold_l250_250016

theorem student_tickets_sold (S NS : ℕ) (h1 : 9 * S + 11 * NS = 20960) (h2 : S + NS = 2000) : S = 520 :=
by
  sorry

end student_tickets_sold_l250_250016


namespace smallest_xym_sum_l250_250536

def is_two_digit_integer (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

def reversed_digits (x y : ℤ) : Prop :=
  ∃ a b : ℤ, x = 10 * a + b ∧ y = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def odd_multiple_of_9 (n : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ n = 9 * k

theorem smallest_xym_sum :
  ∃ (x y m : ℤ), is_two_digit_integer x ∧ is_two_digit_integer y ∧ reversed_digits x y ∧ x^2 + y^2 = m^2 ∧ odd_multiple_of_9 (x + y) ∧ x + y + m = 169 :=
by
  sorry

end smallest_xym_sum_l250_250536


namespace arithmetic_seq_sum_mod_9_l250_250196

def sum_arithmetic_seq := 88230 + 88231 + 88232 + 88233 + 88234 + 88235 + 88236 + 88237 + 88238 + 88239 + 88240

theorem arithmetic_seq_sum_mod_9 : 
  sum_arithmetic_seq % 9 = 0 :=
by
-- proof will be provided here
sorry

end arithmetic_seq_sum_mod_9_l250_250196


namespace simplify_fraction_l250_250699

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) : 
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end simplify_fraction_l250_250699


namespace jake_weight_l250_250096

variable (J S : ℕ)

theorem jake_weight (h1 : J - 15 = 2 * S) (h2 : J + S = 132) : J = 93 := by
  sorry

end jake_weight_l250_250096


namespace find_intersection_l250_250222

def intersection_point (x y : ℚ) : Prop :=
  3 * x + 4 * y = 12 ∧ 7 * x - 2 * y = 14

theorem find_intersection :
  intersection_point (40 / 17) (21 / 17) :=
by
  sorry

end find_intersection_l250_250222


namespace perpendicular_line_l250_250000

theorem perpendicular_line 
  (a b c : ℝ) 
  (p : ℝ × ℝ) 
  (h₁ : p = (-1, 3)) 
  (h₂ : a * (-1) + b * 3 + c = 0) 
  (h₃ : a * p.fst + b * p.snd + c = 0) 
  (hp : a = 1 ∧ b = -2 ∧ c = 3) : 
  ∃ a₁ b₁ c₁ : ℝ, 
  a₁ * (-1) + b₁ * 3 + c₁ = 0 ∧ a₁ = 2 ∧ b₁ = 1 ∧ c₁ = -1 := 
by 
  sorry

end perpendicular_line_l250_250000


namespace problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l250_250841

-- Problem I2.1
theorem problem_I2_1 (a : ℕ) (h₁ : a > 0) (h₂ : a^2 - 1 = 123 * 125) : a = 124 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.2
theorem problem_I2_2 (b : ℕ) (h₁ : b = (2^3 - 16*2^2 - 9*2 + 124)) : b = 50 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.3
theorem problem_I2_3 (n : ℕ) (h₁ : (n * (n - 3)) / 2 = 54) : n = 12 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2_4
theorem problem_I2_4 (d : ℤ) (n : ℤ) (h₁ : n = 12) 
  (h₂ : (d - 1) * 2 = (1 - n) * 2) : d = -10 :=
by {
  -- This proof needs to be filled in
  sorry
}

end problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l250_250841


namespace area_ratio_is_four_l250_250279

-- Definitions based on the given conditions
variables (k a b c d : ℝ)
variables (ka kb kc kd : ℝ)

-- Equations from the conditions
def eq1 : a = k * ka := sorry
def eq2 : b = k * kb := sorry
def eq3 : c = k * kc := sorry
def eq4 : d = k * kd := sorry

-- Ratios provided in the problem
def ratio1 : ka / kc = 2 / 5 := sorry
def ratio2 : kb / kd = 2 / 5 := sorry

-- The theorem to prove the ratio of areas is 4:1
theorem area_ratio_is_four : (k * ka * k * kb) / (k * kc * k * kd) = 4 :=
by sorry

end area_ratio_is_four_l250_250279


namespace can_form_triangle_l250_250435

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end can_form_triangle_l250_250435


namespace sum_of_squares_of_roots_l250_250475

noncomputable def given_polynomial : Polynomial ℝ :=
  Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 8 * Polynomial.X - 2

theorem sum_of_squares_of_roots :
  (∀ r s t : ℝ, (r, s, t).perm (Polynomial.roots given_polynomial) → 
  r > 0 ∧ s > 0 ∧ t > 0 ∨ 
  r + s + t = 9 ∧
  r * s + s * t + r * t = 8 →
  r ^ 2 + s ^ 2 + t ^ 2 = 65) :=
sorry

end sum_of_squares_of_roots_l250_250475


namespace number_of_wickets_last_match_l250_250598

noncomputable def bowling_average : ℝ := 12.4
noncomputable def runs_taken_last_match : ℝ := 26
noncomputable def wickets_before_last_match : ℕ := 175
noncomputable def decrease_in_average : ℝ := 0.4
noncomputable def new_average : ℝ := bowling_average - decrease_in_average

theorem number_of_wickets_last_match (w : ℝ) :
  (175 + w) > 0 → 
  ((wickets_before_last_match * bowling_average + runs_taken_last_match) / (wickets_before_last_match + w) = new_average) →
  w = 8 := 
sorry

end number_of_wickets_last_match_l250_250598


namespace trapezoid_length_l250_250672

variable (W X Y Z : ℝ)
variable (WX ZY WY XY : ℝ)

theorem trapezoid_length (h1 : WX = ZY)
  (h2 : WY * ZY ≠ 0)
  (h3 : YZ = 15)
  (h4 : Real.tan Z = 4 / 3)
  (h5 : Real.tan X = 3 / 2)
  (h6 : ∃ WY WX XY, XY^2 = WY^2 + WX^2 ∧ WX = WY / (3 / 2) ∧ WY = 15 * (4 / 3) ∧ XY > 0):
  XY = 20 * Real.sqrt 13 / 3 :=
by
  sorry

end trapezoid_length_l250_250672


namespace necessary_but_not_sufficient_conditions_l250_250324

theorem necessary_but_not_sufficient_conditions (x y : ℝ) :
  (|x| ≤ 1 ∧ |y| ≤ 1) → x^2 + y^2 ≤ 1 ∨ ¬(x^2 + y^2 ≤ 1) → 
  (|x| ≤ 1 ∧ |y| ≤ 1) → (x^2 + y^2 ≤ 1 → (|x| ≤ 1 ∧ |y| ≤ 1)) :=
by
  sorry

end necessary_but_not_sufficient_conditions_l250_250324


namespace smallest_c_geometric_arithmetic_progression_l250_250274

theorem smallest_c_geometric_arithmetic_progression (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 0 < c) 
(h4 : b ^ 2 = a * c) (h5 : a + b = 2 * c) : c = 1 :=
sorry

end smallest_c_geometric_arithmetic_progression_l250_250274


namespace binom_30_3_eq_4060_l250_250204

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l250_250204


namespace expected_value_of_geometric_variance_of_geometric_l250_250407

noncomputable def expected_value (p : ℝ) : ℝ :=
  1 / p

noncomputable def variance (p : ℝ) : ℝ :=
  (1 - p) / (p ^ 2)

theorem expected_value_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, (n + 1 : ℝ) * (1 - p) ^ n * p = expected_value p := by
  sorry

theorem variance_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, ((n + 1 : ℝ) ^ 2) * (1 - p) ^ n * p - (expected_value p) ^ 2 = variance p := by
  sorry

end expected_value_of_geometric_variance_of_geometric_l250_250407


namespace probability_non_perfect_power_200_l250_250707

def is_perfect_power (x : ℕ) : Prop := 
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b = x

def count_perfect_powers_up_to (n : ℕ) : ℕ := 
  Finset.card (Finset.filter is_perfect_power (Finset.range (n + 1)))

def probability_not_perfect_power (n : ℕ) : ℚ :=
  let total := n in
  let perfect_powers := count_perfect_powers_up_to n in
  (total - perfect_powers) / total

theorem probability_non_perfect_power_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by {
  -- statement placeholder
  sorry
}

end probability_non_perfect_power_200_l250_250707


namespace M_inter_N_empty_l250_250396

def M : Set ℝ := {a : ℝ | (1 / 2 < a ∧ a < 1) ∨ (1 < a)}
def N : Set ℝ := {a : ℝ | 0 < a ∧ a ≤ 1 / 2}

theorem M_inter_N_empty : M ∩ N = ∅ :=
sorry

end M_inter_N_empty_l250_250396


namespace three_seventy_five_as_fraction_l250_250441

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l250_250441


namespace reciprocal_eq_self_is_one_or_neg_one_l250_250886

/-- If a rational number equals its own reciprocal, then the number is either 1 or -1. -/
theorem reciprocal_eq_self_is_one_or_neg_one (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 := 
by
  sorry

end reciprocal_eq_self_is_one_or_neg_one_l250_250886


namespace simplify_evaluate_l250_250548

noncomputable def a := (1 / 2) + Real.sqrt (1 / 2)

theorem simplify_evaluate (a : ℝ) (h : a = (1 / 2) + Real.sqrt (1 / 2)) :
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 :=
by sorry

end simplify_evaluate_l250_250548


namespace find_m_real_find_m_imaginary_l250_250953

-- Define the real part condition
def real_part_condition (m : ℝ) : Prop :=
  m^2 - 3 * m - 4 = 0

-- Define the imaginary part condition
def imaginary_part_condition (m : ℝ) : Prop :=
  m^2 - 2 * m - 3 = 0 ∧ m^2 - 3 * m - 4 ≠ 0

-- Theorem for the first part
theorem find_m_real : ∀ (m : ℝ), (real_part_condition m) → (m = 4 ∨ m = -1) :=
by sorry

-- Theorem for the second part
theorem find_m_imaginary : ∀ (m : ℝ), (imaginary_part_condition m) → (m = 3) :=
by sorry

end find_m_real_find_m_imaginary_l250_250953


namespace non_obtuse_triangle_range_l250_250378

noncomputable def range_of_2a_over_c (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) : Set ℝ :=
  {x | ∃ (a b c A : ℝ), x = (2 * a) / c ∧ 1 < x ∧ x ≤ 4}

theorem non_obtuse_triangle_range (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) :
  (2 * a) / c ∈ range_of_2a_over_c a b c A C h1 h2 h3 := 
sorry

end non_obtuse_triangle_range_l250_250378


namespace rectangle_area_l250_250010

theorem rectangle_area (y : ℝ) (h_rect : (5 - (-3)) * (y - (-1)) = 48) (h_pos : 0 < y) : y = 5 :=
by
  sorry

end rectangle_area_l250_250010


namespace problem_1_problem_2_problem_3_l250_250246

-- Problem 1
theorem problem_1 (a : ℝ) (h_pos : a > 0) 
  (h_increasing : ∀ x : ℝ, (1 < x) → deriv (λ x, (1 - x) / (a * x) + log x) x ≥ 0) : 
  1 ≤ a := sorry

-- Problem 2
theorem problem_2 : 
  ∃ x₀, ∀ x ∈ set.Ici (0 : ℝ), g x ≤ g x₀ ∧ g x₀ = 0 :=
sorry
  where g (x : ℝ) := log (1 + x) - x 

-- Problem 3
theorem problem_3 (a b : ℝ) (h_a : a > 1) (h_b : b > 0) :
  1 / (a + b) ≤ log ((a + b) / b) ∧ log ((a + b) / b) < a / b :=
sorry

end problem_1_problem_2_problem_3_l250_250246


namespace smallest_multiple_of_6_and_15_l250_250074

theorem smallest_multiple_of_6_and_15 : ∃ a : ℕ, a > 0 ∧ a % 6 = 0 ∧ a % 15 = 0 ∧ ∀ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 → a ≤ b :=
  sorry

end smallest_multiple_of_6_and_15_l250_250074


namespace jason_cutting_hours_l250_250975

-- Definitions derived from conditions
def time_to_cut_one_lawn : ℕ := 30  -- minutes
def lawns_per_day := 8 -- number of lawns Jason cuts each day
def days := 2 -- number of days (Saturday and Sunday)
def minutes_in_an_hour := 60 -- conversion factor from minutes to hours

-- The proof problem
theorem jason_cutting_hours : 
  (time_to_cut_one_lawn * lawns_per_day * days) / minutes_in_an_hour = 8 := sorry

end jason_cutting_hours_l250_250975


namespace radius_ratio_l250_250046

noncomputable def ratio_of_radii (V1 V2 : ℝ) (R : ℝ) : ℝ := 
  (V2 / V1)^(1/3) * R 

theorem radius_ratio (V1 V2 : ℝ) (π : ℝ) (R r : ℝ) :
  V1 = 450 * π → 
  V2 = 36 * π → 
  (4 / 3) * π * R^3 = V1 →
  (4 / 3) * π * r^3 = V2 →
  r / R = 1 / (12.5)^(1/3) :=
by {
  sorry
}

end radius_ratio_l250_250046


namespace travel_cost_is_correct_l250_250924

-- Definitions of the conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60
def road_width : ℝ := 15
def cost_per_sq_m : ℝ := 3

-- Areas of individual roads
def area_road_length := road_width * lawn_breadth
def area_road_breadth := road_width * lawn_length
def intersection_area := road_width * road_width

-- Adjusted area for roads discounting intersection area
def total_area_roads := area_road_length + area_road_breadth - intersection_area

-- Total cost of traveling the roads
def total_cost := total_area_roads * cost_per_sq_m

theorem travel_cost_is_correct : total_cost = 5625 := by
  sorry

end travel_cost_is_correct_l250_250924


namespace unique_angles_sum_l250_250268

theorem unique_angles_sum (a1 a2 a3 a4 e4 e5 e6 e7 : ℝ) 
  (h_abcd: a1 + a2 + a3 + a4 = 360) 
  (h_efgh: e4 + e5 + e6 + e7 = 360) 
  (h_shared: a4 = e4) : 
  a1 + a2 + a3 + e4 + e5 + e6 + e7 - a4 = 360 := 
by 
  sorry

end unique_angles_sum_l250_250268


namespace right_triangle_area_l250_250291

noncomputable def num_possible_locations (P Q : EuclideanSpace ℝ (Fin 2)) (hPQ : dist P Q = 10) : Nat :=
  8

theorem right_triangle_area (P Q R : EuclideanSpace ℝ (Fin 2)) 
  (hPQ : dist P Q = 10)
  (hArea : euclidean_dist P Q * height R = 32) : 
  num_possible_locations P Q hPQ = 8 :=
by
  sorry

end right_triangle_area_l250_250291


namespace locus_of_points_l250_250897

def point := (ℝ × ℝ)

variables (F_1 F_2 : point) (r k : ℝ)

def distance (P Q : point) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def on_circle (P : point) (center : point) (radius : ℝ) : Prop :=
  distance P center = radius

theorem locus_of_points
  (P : point)
  (r1 r2 PF1 PF2 : ℝ)
  (h_pF1 : r1 = distance P F_1)
  (h_pF2 : PF2 = distance P F_2)
  (h_outside_circle : PF2 = r2 + r)
  (h_inside_circle : PF2 = r - r2)
  (h_k : r1 + PF2 = k) :
  (∀ P, distance P F_1 + distance P F_2 = k →
  ( ∃ e_ellipse : Prop, on_circle P F_2 r → e_ellipse) ∨ 
  ( ∃ h_hyperbola : Prop, on_circle P F_2 r → h_hyperbola)) :=
by
  sorry

end locus_of_points_l250_250897


namespace valid_relationship_l250_250232

noncomputable def proof_statement (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : Prop :=
  b > a ∧ a > c

theorem valid_relationship (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : proof_statement a b c h_distinct h_pos h_eq :=
  sorry

end valid_relationship_l250_250232


namespace multiplication_expression_l250_250608

theorem multiplication_expression : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end multiplication_expression_l250_250608


namespace original_people_in_room_l250_250869

theorem original_people_in_room (x : ℕ) (h1 : 18 = (2 * x / 3) - (x / 6)) : x = 36 :=
by sorry

end original_people_in_room_l250_250869


namespace find_line_equation_l250_250962

open Real

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line passing through (0,2)
def LineThruPoint (x y k : ℝ) : Prop := y = k * x + 2

-- Define when line intersects parabola
def LineIntersectsParabola (x1 y1 x2 y2 k : ℝ) : Prop :=
  LineThruPoint x1 y1 k ∧ LineThruPoint x2 y2 k ∧ Parabola x1 y1 ∧ Parabola x2 y2

-- Define when circle with diameter MN passes through origin O
def CircleThroughOrigin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem find_line_equation (k : ℝ) 
    (h₀ : k ≠ 0)
    (h₁ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k)
    (h₂ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k ∧ CircleThroughOrigin x1 y1 x2 y2) :
  (∃ x y, LineThruPoint x y k ∧ y = -x + 2) :=
sorry

end find_line_equation_l250_250962


namespace sqrt_domain_l250_250017

theorem sqrt_domain (x : ℝ) : 1 - x ≥ 0 → x ≤ 1 := by
  sorry

end sqrt_domain_l250_250017


namespace angus_tokens_l250_250280

theorem angus_tokens (x : ℕ) (h1 : x = 60 - (25 / 100) * 60) : x = 45 :=
by
  sorry

end angus_tokens_l250_250280


namespace range_of_m_l250_250657

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then 1 / (Real.exp x) + m * x^2
else Real.exp x + m * x^2

theorem range_of_m {m : ℝ} : (∀ m, ∃ x y, f x m = 0 ∧ f y m = 0 ∧ x ≠ y) ↔ m < -Real.exp 2 / 4 := by
  sorry

end range_of_m_l250_250657


namespace find_ab_l250_250367

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 30) : a * b = 32 :=
by
  -- We will complete the proof in this space
  sorry

end find_ab_l250_250367


namespace hundredths_digit_of_power_l250_250172

theorem hundredths_digit_of_power (n : ℕ) (h : n % 20 = 14) : 
  (8 ^ n % 1000) / 100 = 1 :=
by sorry

lemma test_power_hundredths_digit : (8 ^ 1234 % 1000) / 100 = 1 :=
hundredths_digit_of_power 1234 (by norm_num)

end hundredths_digit_of_power_l250_250172


namespace find_other_number_l250_250411

theorem find_other_number (HCF LCM num1 num2 : ℕ) 
    (h_hcf : HCF = 14)
    (h_lcm : LCM = 396)
    (h_num1 : num1 = 36)
    (h_prod : HCF * LCM = num1 * num2)
    : num2 = 154 := by
  sorry

end find_other_number_l250_250411


namespace min_value_two_x_plus_y_l250_250534

theorem min_value_two_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y + 2 * x * y = 5 / 4) : 2 * x + y ≥ 1 :=
by
  sorry

end min_value_two_x_plus_y_l250_250534


namespace total_cost_of_digging_well_l250_250622

noncomputable def cost_of_digging (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * (radius^2) * depth
  volume * cost_per_cubic_meter

theorem total_cost_of_digging_well :
  cost_of_digging 14 3 15 = 1484.4 :=
by
  sorry

end total_cost_of_digging_well_l250_250622


namespace number_solution_l250_250097

variable (a : ℝ) (x : ℝ)

theorem number_solution :
  (a^(-x) + 25^(-2*x) + 5^(-4*x) = 11) ∧ (x = 0.25) → a = 625 / 7890481 :=
by 
  sorry

end number_solution_l250_250097


namespace solution_set_l250_250142

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l250_250142


namespace pepper_left_l250_250473

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem pepper_left (h1 : initial_pepper = 0.25) (h2 : used_pepper = 0.16) :
  initial_pepper - used_pepper = remaining_pepper :=
by
  sorry

end pepper_left_l250_250473


namespace sum_lent_is_300_l250_250189

-- Define the conditions
def interest_rate : ℕ := 4
def time_period : ℕ := 8
def interest_amounted_less : ℕ := 204

-- Prove that the sum lent P is 300 given the conditions
theorem sum_lent_is_300 (P : ℕ) : 
  (P * interest_rate * time_period / 100 = P - interest_amounted_less) -> P = 300 := by
  sorry

end sum_lent_is_300_l250_250189


namespace reassemble_square_with_hole_l250_250604

theorem reassemble_square_with_hole 
  (a b c d k1 k2 : ℝ)
  (h1 : a = b)
  (h2 : c = d)
  (h3 : k1 = k2) :
  ∃ (f gh ef gh' : ℝ), 
    f = a - c ∧
    gh = b - d ∧
    ef = f ∧
    gh' = gh := 
by sorry

end reassemble_square_with_hole_l250_250604


namespace total_books_l250_250306

noncomputable def num_books_on_shelf : ℕ := 8

theorem total_books (p h s : ℕ) (assump1 : p = 2) (assump2 : h = 6) (assump3 : s = 36) :
  p + h = num_books_on_shelf :=
by {
  -- leaving the proof construction out as per instructions
  sorry
}

end total_books_l250_250306


namespace max_integer_a_for_real_roots_l250_250646

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l250_250646


namespace value_of_phi_l250_250509

theorem value_of_phi { φ : ℝ } (hφ1 : 0 < φ) (hφ2 : φ < π)
  (symm_condition : ∃ k : ℤ, -π / 8 + φ = k * π + π / 2) : φ = 3 * π / 4 := 
by 
  sorry

end value_of_phi_l250_250509


namespace unique_positive_solution_eq_15_l250_250629

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l250_250629


namespace number_of_bead_necklaces_sold_is_3_l250_250518

-- Definitions of the given conditions
def total_earnings : ℕ := 36
def gemstone_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 6

-- Define the earnings from gemstone necklaces as a separate definition
def earnings_gemstone_necklaces : ℕ := gemstone_necklaces * cost_per_necklace

-- Define the earnings from bead necklaces based on total earnings and earnings from gemstone necklaces
def earnings_bead_necklaces : ℕ := total_earnings - earnings_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_bead_necklaces / cost_per_necklace

-- The theorem we want to prove
theorem number_of_bead_necklaces_sold_is_3 : bead_necklaces_sold = 3 :=
by
  sorry

end number_of_bead_necklaces_sold_is_3_l250_250518


namespace binom_30_3_is_4060_l250_250201

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l250_250201


namespace infinite_double_perfect_squares_l250_250455

def is_double_number (n : ℕ) : Prop :=
  ∃ k m : ℕ, m > 0 ∧ n = m * 10^k + m

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_double_perfect_squares : ∀ n : ℕ, ∃ m, n < m ∧ is_double_number m ∧ is_perfect_square m :=
  sorry

end infinite_double_perfect_squares_l250_250455


namespace angle_bisectors_form_inscribed_quadrilateral_l250_250692

-- Definitions for basic angle representation and quadrilateral
structure Quadrilateral (α : Type) :=
  (A B C D : α)

structure AngleBisector where
  quadrilateral : Quadrilateral ℝ
  bisectorA bisectorB bisectorC bisectorD : ℝ

def isConvex (Q : Quadrilateral ℝ) : Prop := sorry

def sumInternalAngles (Q : Quadrilateral ℝ) : ℝ :=
  let ⟨A, B, C, D⟩ := Q
  (D - A) + (B - A) + (C - D) + (D - C)

def isCyclic (Q : Quadrilateral ℝ) (a b c d : ℝ) : Prop :=
  let α := (a + b) 
  let β := (c + d) 
  α + β = 180

-- Given conditions based on the problem
axiom quadrilateral_convex (Q : Quadrilateral ℝ) : isConvex Q

axiom quadrilateral_internal_angle_sum (Q : Quadrilateral ℝ) : sumInternalAngles Q = 360

-- The main theorem statement we need to prove
theorem angle_bisectors_form_inscribed_quadrilateral (Q : Quadrilateral ℝ) (a b c d : ℝ) (h: AngleBisector) :
  isConvex Q →
  sumInternalAngles Q = 360 →
  isCyclic Q a b c d :=
begin
  sorry
end

end angle_bisectors_form_inscribed_quadrilateral_l250_250692


namespace gcd_polynomial_multiple_of_532_l250_250240

theorem gcd_polynomial_multiple_of_532 (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a ^ 3 + 2 * a ^ 2 + 6 * a + 76) a = 76 :=
by
  sorry

end gcd_polynomial_multiple_of_532_l250_250240


namespace calculate_expression_l250_250507

theorem calculate_expression (x y : ℚ) (hx : x = 5 / 6) (hy : y = 6 / 5) : 
  (1 / 3) * (x ^ 8) * (y ^ 9) = 2 / 5 :=
by
  sorry

end calculate_expression_l250_250507


namespace solution_set_l250_250129

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l250_250129


namespace probability_neither_red_blue_purple_l250_250327

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_neither_red_blue_purple :
  (total_balls - (red_balls + blue_balls + purple_balls)) / total_balls = 35 / 48 := 
by 
  /- Proof details are not necessary -/
  sorry

end probability_neither_red_blue_purple_l250_250327


namespace part_I_part_II_l250_250826

variable (α : ℝ)

-- The given conditions.
variable (h1 : π < α)
variable (h2 : α < (3 * π) / 2)
variable (h3 : Real.sin α = -4/5)

-- Part (I): Prove cos α = -3/5
theorem part_I : Real.cos α = -3/5 :=
sorry

-- Part (II): Prove sin 2α + 3 tan α = 24/25 + 4
theorem part_II : Real.sin (2 * α) + 3 * Real.tan α = 24/25 + 4 :=
sorry

end part_I_part_II_l250_250826


namespace unique_positive_real_solution_l250_250626

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l250_250626


namespace triangle_area_is_60_l250_250448

noncomputable def triangle_area (P r : ℝ) : ℝ :=
  (r * P) / 2

theorem triangle_area_is_60 (hP : 48 = 48) (hr : 2.5 = 2.5) : triangle_area 48 2.5 = 60 := by
  sorry

end triangle_area_is_60_l250_250448


namespace abs_fraction_inequality_l250_250555

theorem abs_fraction_inequality (x : ℝ) :
  (abs ((3 * x - 4) / (x - 2)) > 3) ↔
  (x ∈ Set.Iio (5 / 3) ∪ Set.Ioo (5 / 3) 2 ∪ Set.Ioi 2) :=
by 
  sorry

end abs_fraction_inequality_l250_250555


namespace age_difference_ratio_l250_250696

def Roy_age_condition_1 (R J K : ℕ) : Prop := R = J + 8
def Roy_age_condition_2 (R J K : ℕ) : Prop := R + 2 = 3 * (J + 2)
def Roy_age_condition_3 (R J K : ℕ) : Prop := (R + 2) * (K + 2) = 96

def ratio_of_age_differences (R J K : ℕ) : ℚ := (R - J : ℚ) / (R - K)

theorem age_difference_ratio (R J K : ℕ) :
  Roy_age_condition_1 R J K →
  Roy_age_condition_2 R J K →
  Roy_age_condition_3 R J K →
  ratio_of_age_differences R J K = 2 :=
by
  sorry

end age_difference_ratio_l250_250696


namespace find_circle_equation_l250_250357

noncomputable def center_of_parabola : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

noncomputable def equation_of_circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1

theorem find_circle_equation 
  (center_c : ℝ × ℝ := center_of_parabola)
  (tangent : ∀ x y, tangent_line x y → (x - 1) ^ 2 + (y - 0) ^ 2 = 1) :
  equation_of_circle = (fun x y => sorry) :=
sorry

end find_circle_equation_l250_250357


namespace binom_computation_l250_250207

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l250_250207


namespace image_of_center_l250_250942

def original_center : ℤ × ℤ := (3, -4)

def reflect_x (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def translate_down (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1, p.2 - d)

theorem image_of_center :
  (translate_down (reflect_y (reflect_x original_center)) 10) = (-3, -6) :=
by
  sorry

end image_of_center_l250_250942


namespace total_waiting_time_difference_l250_250305

theorem total_waiting_time_difference :
  let n_swings := 6
  let n_slide := 4 * n_swings
  let t_swings := 3.5 * 60
  let t_slide := 45
  let T_swings := n_swings * t_swings
  let T_slide := n_slide * t_slide
  let T_difference := T_swings - T_slide
  T_difference = 180 :=
by
  sorry

end total_waiting_time_difference_l250_250305


namespace modified_cube_surface_area_l250_250609

noncomputable def total_surface_area_modified_cube : ℝ :=
  let side_length := 10
  let triangle_side := 7 * Real.sqrt 2
  let tunnel_wall_area := 3 * (Real.sqrt 3 / 4 * triangle_side^2)
  let original_surface_area := 6 * side_length^2
  original_surface_area + tunnel_wall_area

theorem modified_cube_surface_area : 
  total_surface_area_modified_cube = 600 + 73.5 * Real.sqrt 3 := 
  sorry

end modified_cube_surface_area_l250_250609


namespace AcmeExtendedVowelSoup_correct_l250_250751

noncomputable def AcmeExtendedVowelSoup : Prop :=
  let vowels := 5
  let semi_vowel_y := 3
  let total_words := (vowels^5) + (5 * (vowels^4)) + (Nat.choose 5 2 * (vowels^3)) + (Nat.choose 5 3 * (vowels^2))
  total_words = 7750

theorem AcmeExtendedVowelSoup_correct : AcmeExtendedVowelSoup := by
  sorry -- Proof is omitted as requested.

end AcmeExtendedVowelSoup_correct_l250_250751


namespace largest_x_satisfies_condition_l250_250788

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l250_250788


namespace handshake_problem_l250_250729

-- Define the remainder operation
def r_mod (n : ℕ) (k : ℕ) : ℕ := n % k

-- Define the function F
def F (t : ℕ) : ℕ := r_mod (t^3) 5251

-- The lean theorem statement with the given conditions and expected results
theorem handshake_problem :
  ∃ (x y : ℕ),
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 :=
by
  sorry

end handshake_problem_l250_250729


namespace total_lives_l250_250586

/-- Suppose there are initially 4 players, then 5 more players join. Each player has 3 lives.
    Prove that the total number of lives is equal to 27. -/
theorem total_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
  (h_initial : initial_players = 4) (h_additional : additional_players = 5) (h_lives : lives_per_player = 3) : 
  initial_players + additional_players = 9 ∧ 
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l250_250586


namespace largest_real_solution_l250_250792

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l250_250792


namespace bus_speed_excluding_stoppages_l250_250070

theorem bus_speed_excluding_stoppages (s_including_stops : ℕ) (stop_time_minutes : ℕ) (s_excluding_stops : ℕ) (v : ℕ) : 
  (s_including_stops = 45) ∧ (stop_time_minutes = 24) ∧ (v = s_including_stops * 5 / 3) → s_excluding_stops = 75 := 
by {
  sorry
}

end bus_speed_excluding_stoppages_l250_250070


namespace soccer_field_solution_l250_250467

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l250_250467


namespace sufficient_but_not_necessary_condition_l250_250912

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∃ m0 : ℝ, m0 > 0 ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m0 x1 ≤ f m0 x2)) ∧ 
  ¬ (∀ m : ℝ, (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m x1 ≤ f m x2) → m > 0) :=
by sorry

end sufficient_but_not_necessary_condition_l250_250912


namespace birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l250_250733

theorem birds_percentage_not_hawks_paddyfield_warblers_kingfishers
  (total_birds : ℕ)
  (hawks_percentage : ℝ := 0.3)
  (paddyfield_warblers_percentage : ℝ := 0.4)
  (kingfishers_ratio : ℝ := 0.25) :
  (35 : ℝ) = 100 * ( total_birds - (hawks_percentage * total_birds) 
                     - (paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) 
                     - (kingfishers_ratio * paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) )
                / total_birds :=
by
  sorry

end birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l250_250733


namespace ceo_dividends_correct_l250_250057

-- Definitions of parameters
def revenue := 2500000
def expenses := 1576250
def tax_rate := 0.2
def monthly_loan_payment := 25000
def months := 12
def number_of_shares := 1600
def ceo_ownership := 0.35

-- Calculation functions based on conditions
def net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
def loan_payments := monthly_loan_payment * months
def dividends_per_share := (net_profit - loan_payments) / number_of_shares
def ceo_dividends := dividends_per_share * ceo_ownership * number_of_shares

-- Statement to prove
theorem ceo_dividends_correct : ceo_dividends = 153440 :=
by 
  -- skipping the proof
  sorry

end ceo_dividends_correct_l250_250057


namespace solve_for_x_l250_250335

-- Define the conditions
def percentage15_of_25 : ℝ := 0.15 * 25
def percentage12 (x : ℝ) : ℝ := 0.12 * x
def condition (x : ℝ) : Prop := percentage15_of_25 + percentage12 x = 9.15

-- The target statement to prove
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 45 :=
by 
  -- The proof is omitted
  sorry

end solve_for_x_l250_250335


namespace birch_tree_count_l250_250264

theorem birch_tree_count:
  let total_trees := 8000
  let spruces := 0.12 * total_trees
  let pines := 0.15 * total_trees
  let maples := 0.18 * total_trees
  let cedars := 0.09 * total_trees
  let oaks := spruces + pines
  let calculated_trees := spruces + pines + maples + cedars + oaks
  let birches := total_trees - calculated_trees
  spruces = 960 → pines = 1200 → maples = 1440 → cedars = 720 → oaks = 2160 →
  birches = 1520 :=
by
  intros
  sorry

end birch_tree_count_l250_250264


namespace passing_marks_l250_250583

theorem passing_marks
  (T P : ℝ)
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) :
  P = 160 :=
by
  sorry

end passing_marks_l250_250583


namespace thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l250_250309

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem thirtieth_triangular_number :
  triangular_number 30 = 465 :=
by
  sorry

theorem sum_thirtieth_thirtyfirst_triangular_numbers :
  triangular_number 30 + triangular_number 31 = 961 :=
by
  sorry

end thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l250_250309


namespace min_value_expression_l250_250861

theorem min_value_expression (a b : ℝ) (h : a > b) (h0 : b > 0) :
  ∃ m : ℝ, m = (a^2 + 1 / (a * b) + 1 / (a * (a - b))) ∧ m = 4 :=
sorry

end min_value_expression_l250_250861


namespace income_second_day_l250_250184

theorem income_second_day (x : ℕ) 
  (h_condition : (200 + x + 750 + 400 + 500) / 5 = 400) : x = 150 :=
by 
  -- Proof omitted.
  sorry

end income_second_day_l250_250184


namespace max_lateral_surface_area_l250_250926

theorem max_lateral_surface_area : ∀ (x y : ℝ), 6 * x + 3 * y = 12 → (3 * x * y) ≤ 6 :=
by
  intros x y h
  have xy_le_2 : x * y ≤ 2 :=
    by
      sorry
  have max_area_6 : 3 * x * y ≤ 6 :=
    by
      sorry
  exact max_area_6

end max_lateral_surface_area_l250_250926


namespace loss_percentage_first_book_l250_250094

theorem loss_percentage_first_book (C1 C2 : ℝ) 
    (total_cost : ℝ) 
    (gain_percentage : ℝ)
    (S1 S2 : ℝ)
    (cost_first_book : C1 = 175)
    (total_cost_condition : total_cost = 300)
    (gain_condition : gain_percentage = 0.19)
    (same_selling_price : S1 = S2)
    (second_book_cost : C2 = total_cost - C1)
    (selling_price_second_book : S2 = C2 * (1 + gain_percentage)) :
    (C1 - S1) / C1 * 100 = 15 :=
by
  sorry

end loss_percentage_first_book_l250_250094


namespace minimum_value_l250_250115

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
    6 ≤ (x^2 + 2*y^2) / (x + y) + (x^2 + 2*z^2) / (x + z) + (y^2 + 2*z^2) / (y + z) :=
by
  sorry

end minimum_value_l250_250115


namespace geometric_series_sum_l250_250766

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l250_250766


namespace no_such_n_l250_250340

theorem no_such_n (n : ℕ) (h_positive : n > 0) : 
  ¬ ∃ k : ℕ, (n^2 + 1) = k * (Nat.floor (Real.sqrt n))^2 + 2 := by
  sorry

end no_such_n_l250_250340


namespace find_a_of_extreme_at_1_l250_250003

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - x - Real.log x

theorem find_a_of_extreme_at_1 :
  (∃ a : ℝ, ∃ f' : ℝ -> ℝ, (f' x = 3 * a * x^2 - 1 - 1/x) ∧ f' 1 = 0) →
  ∃ a : ℝ, a = 2 / 3 :=
by
  sorry

end find_a_of_extreme_at_1_l250_250003


namespace marias_workday_end_time_l250_250281

theorem marias_workday_end_time :
  ∀ (start_time : ℕ) (lunch_time : ℕ) (work_duration : ℕ) (lunch_break : ℕ) (total_work_time : ℕ),
  start_time = 8 ∧ lunch_time = 13 ∧ work_duration = 8 ∧ lunch_break = 1 →
  (total_work_time = work_duration - (lunch_time - start_time - lunch_break)) →
  lunch_time + 1 + (work_duration - (lunch_time - start_time)) = 17 :=
by
  sorry

end marias_workday_end_time_l250_250281


namespace more_than_half_millet_on_day_three_l250_250544

-- Definition of the initial conditions
def seeds_in_feeder (n: ℕ) : ℝ :=
  1 + n

def millet_amount (n: ℕ) : ℝ :=
  0.6 * (1 - (0.5)^n)

-- The theorem we want to prove
theorem more_than_half_millet_on_day_three :
  ∀ n, n = 3 → (millet_amount n) / (seeds_in_feeder n) > 0.5 :=
by
  intros n hn
  rw [hn, seeds_in_feeder, millet_amount]
  sorry

end more_than_half_millet_on_day_three_l250_250544


namespace linear_function_passing_origin_l250_250971

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin_l250_250971


namespace yura_picture_dimensions_l250_250444

-- Definitions based on the problem conditions
variable {a b : ℕ} -- dimensions of the picture
variable (hasFrame : ℕ × ℕ → Prop) -- definition sketch

-- The main statement to prove
theorem yura_picture_dimensions (h : (a + 2) * (b + 2) - a * b = 2 * a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
  sorry

end yura_picture_dimensions_l250_250444


namespace total_steps_traveled_l250_250929

def steps_per_mile : ℕ := 2000
def walk_to_subway : ℕ := 2000
def subway_ride_miles : ℕ := 7
def walk_to_rockefeller : ℕ := 3000
def cab_ride_miles : ℕ := 3

theorem total_steps_traveled :
  walk_to_subway +
  (subway_ride_miles * steps_per_mile) +
  walk_to_rockefeller +
  (cab_ride_miles * steps_per_mile)
  = 24000 := 
by 
  sorry

end total_steps_traveled_l250_250929


namespace part1_solution_part2_solution_l250_250819

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) * x + abs (x - 2) * (x - a)

theorem part1_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a < 0} = {x : ℝ | x < 1} :=
by
  sorry

theorem part2_solution (x : ℝ) (hx : x < 1) :
  {a : ℝ | f x a < 0} = {a : ℝ | 1 ≤ a} :=
by
  sorry

end part1_solution_part2_solution_l250_250819


namespace optimal_play_winner_l250_250574

-- Definitions for the conditions
def chessboard_size (K N : ℕ) : Prop := True
def rook_initial_position (K N : ℕ) : (ℕ × ℕ) :=
  (K, N)
def move (r : ℕ × ℕ) (direction : ℕ) : (ℕ × ℕ) :=
  if direction = 0 then (r.1 - 1, r.2)
  else (r.1, r.2 - 1)
def rook_cannot_move (r : ℕ × ℕ) : Prop :=
  r.1 = 0 ∨ r.2 = 0

-- Theorem to prove the winner given the conditions
theorem optimal_play_winner (K N : ℕ) :
  (K = N → ∃ player : ℕ, player = 2) ∧ (K ≠ N → ∃ player : ℕ, player = 1) :=
by
  sorry

end optimal_play_winner_l250_250574


namespace greatest_whole_number_solution_l250_250776

theorem greatest_whole_number_solution :
  ∃ (x : ℕ), (5 * x - 4 < 3 - 2 * x) ∧ ∀ (y : ℕ), (5 * y - 4 < 3 - 2 * y) → y ≤ x ∧ x = 0 :=
by
  sorry

end greatest_whole_number_solution_l250_250776


namespace sum_of_given_geom_series_l250_250763

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l250_250763


namespace extreme_points_f_l250_250877

theorem extreme_points_f (a b : ℝ)
  (h1 : 3 * (-2)^2 + 2 * a * (-2) + b = 0)
  (h2 : 3 * 4^2 + 2 * a * 4 + b = 0) :
  a - b = 21 :=
sorry

end extreme_points_f_l250_250877


namespace tan_half_angle_product_l250_250612

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0) :
  (Real.tan (a / 2)) * (Real.tan (b / 2)) = 5 ∨ (Real.tan (a / 2)) * (Real.tan (b / 2)) = -5 :=
by 
  sorry

end tan_half_angle_product_l250_250612


namespace geometric_series_sum_l250_250767

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l250_250767


namespace hyperbola_distance_property_l250_250668

theorem hyperbola_distance_property (P : ℝ × ℝ)
  (hP_on_hyperbola : (P.1 ^ 2 / 16) - (P.2 ^ 2 / 9) = 1)
  (h_dist_15 : dist P (5, 0) = 15) :
  dist P (-5, 0) = 7 ∨ dist P (-5, 0) = 23 := 
sorry

end hyperbola_distance_property_l250_250668


namespace color_dot_figure_l250_250944

-- Definitions reflecting the problem conditions
def num_colors : ℕ := 3
def first_triangle_coloring_ways : ℕ := 6
def subsequent_triangle_coloring_ways : ℕ := 3
def additional_dot_coloring_ways : ℕ := 2

-- The theorem stating the required proof
theorem color_dot_figure : first_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           additional_dot_coloring_ways = 108 := by
sorry

end color_dot_figure_l250_250944


namespace sunzi_classic_l250_250035

noncomputable def length_of_rope : ℝ := sorry
noncomputable def length_of_wood : ℝ := sorry
axiom first_condition : length_of_rope - length_of_wood = 4.5
axiom second_condition : length_of_wood - (1 / 2) * length_of_rope = 1

theorem sunzi_classic : 
  (length_of_rope - length_of_wood = 4.5) ∧ (length_of_wood - (1 / 2) * length_of_rope = 1) := 
by 
  exact ⟨first_condition, second_condition⟩

end sunzi_classic_l250_250035


namespace roots_sum_cubes_l250_250390

theorem roots_sum_cubes (a b c d : ℝ) 
  (h_eqn : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    3 * x^4 + 6 * x^3 + 1002 * x^2 + 2005 * x + 4010 = 0) :
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by { sorry }

end roots_sum_cubes_l250_250390


namespace max_value_part1_l250_250587

theorem max_value_part1 (a : ℝ) (h : a < 3 / 2) : 2 * a + 4 / (2 * a - 3) + 3 ≤ 2 :=
sorry

end max_value_part1_l250_250587


namespace chests_content_l250_250528

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l250_250528


namespace cubic_root_identity_l250_250988

theorem cubic_root_identity (x1 x2 x3 : ℝ) (h1 : x1^3 - 3*x1 - 1 = 0) (h2 : x2^3 - 3*x2 - 1 = 0) (h3 : x3^3 - 3*x3 - 1 = 0) (h4 : x1 < x2) (h5 : x2 < x3) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_identity_l250_250988


namespace johns_new_weekly_earnings_l250_250110

-- Definition of the initial weekly earnings
def initial_weekly_earnings := 40

-- Definition of the percent increase in earnings
def percent_increase := 100

-- Definition for the final weekly earnings after the raise
def final_weekly_earnings (initial_earnings : Nat) (percentage : Nat) := 
  initial_earnings + (initial_earnings * percentage / 100)

-- Theorem stating John’s final weekly earnings after the raise
theorem johns_new_weekly_earnings : final_weekly_earnings initial_weekly_earnings percent_increase = 80 :=
  by
  sorry

end johns_new_weekly_earnings_l250_250110


namespace min_a_minus_b_when_ab_eq_156_l250_250389

theorem min_a_minus_b_when_ab_eq_156 : ∃ a b : ℤ, (a * b = 156 ∧ a - b = -155) :=
by
  sorry

end min_a_minus_b_when_ab_eq_156_l250_250389


namespace division_addition_example_l250_250197

theorem division_addition_example : 12 / (1 / 6) + 3 = 75 := by
  sorry

end division_addition_example_l250_250197


namespace find_a_l250_250218

def operation (a b : ℤ) : ℤ := 2 * a - b * b

theorem find_a (a : ℤ) : operation a 3 = 15 → a = 12 := by
  sorry

end find_a_l250_250218


namespace domain_of_f_zeros_of_f_l250_250500

-- Given condition
variable {a : ℝ} (ha : 0 < a ∧ a < 1)

-- Define the function
def f (x : ℝ) : ℝ := log a (1 - x) + log a (x + 3)

-- Domain condition
theorem domain_of_f : ∀ x, -3 < x ∧ x < 1 ↔ 1 - x > 0 ∧ x + 3 > 0 :=
by
  intro x
  split
  · intro h
    dsimp at h
    rw [←h.left, ←h.right]
    linarith
  · intro h
    linarith
  done

-- Zeros condition
theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 + sqrt 3 ∨ x = -1 - sqrt 3 :=
by
  intro x
  have hx : f x = log a (-(x ^ 2) - 2 * x + 3) := sorry  -- Simplify f(x) expression
  rw [hx]
  -- Showing that f(x) = 0 implies log expression equal to 1
  rw [log_eq_zero ha]
  split
  · intro h
    apply quadratic.zero_0 h
    {
      linarith,
    linarith
      refined ⟨le_of_lt (sqrt_pos.mpr _), le_of_lt (sqrt_pos.mpr _)⟩ sorry sorry
    }
  use [ -1 + sqrt 3, -1 - sqrt 3 ],
  done

end domain_of_f_zeros_of_f_l250_250500


namespace three_seventy_five_as_fraction_l250_250443

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l250_250443


namespace sum_of_given_geom_series_l250_250764

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l250_250764


namespace trigonometric_identity_l250_250490

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.sin (3 * Real.pi / 2 - α) = -3 / 10 :=
by
  sorry

end trigonometric_identity_l250_250490


namespace fill_tank_time_l250_250922

theorem fill_tank_time (R L E : ℝ) (fill_time : ℝ) (leak_time : ℝ) (effective_rate : ℝ) : 
  (R = 1 / fill_time) → 
  (L = 1 / leak_time) →
  (E = R - L) →
  (fill_time = 10) →
  (leak_time = 110) →
  (E = 1 / effective_rate) →
  effective_rate = 11 :=
by
  sorry

end fill_tank_time_l250_250922


namespace number_of_people_after_10_years_l250_250195

def number_of_people_after_n_years (n : ℕ) : ℕ :=
  Nat.recOn n 30 (fun k a_k => 3 * a_k - 20)

theorem number_of_people_after_10_years :
  number_of_people_after_n_years 10 = 1180990 := by
  sorry

end number_of_people_after_10_years_l250_250195


namespace vendor_has_maaza_l250_250921

theorem vendor_has_maaza (liters_pepsi : ℕ) (liters_sprite : ℕ) (total_cans : ℕ) (gcd_pepsi_sprite : ℕ) (cans_pepsi : ℕ) (cans_sprite : ℕ) (cans_maaza : ℕ) (liters_per_can : ℕ) (total_liters_maaza : ℕ) :
  liters_pepsi = 144 →
  liters_sprite = 368 →
  total_cans = 133 →
  gcd_pepsi_sprite = Nat.gcd liters_pepsi liters_sprite →
  gcd_pepsi_sprite = 16 →
  cans_pepsi = liters_pepsi / gcd_pepsi_sprite →
  cans_sprite = liters_sprite / gcd_pepsi_sprite →
  cans_maaza = total_cans - (cans_pepsi + cans_sprite) →
  liters_per_can = gcd_pepsi_sprite →
  total_liters_maaza = cans_maaza * liters_per_can →
  total_liters_maaza = 1616 :=
by
  sorry

end vendor_has_maaza_l250_250921


namespace find_number_l250_250432

theorem find_number (x : ℝ) (h : ((x / 3) * 24) - 7 = 41) : x = 6 :=
by
  sorry

end find_number_l250_250432


namespace evaluate_expression_l250_250234

noncomputable def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem evaluate_expression (A B C D : ℝ) (h1 : g A B C D 2 = 5) (h2 : g A B C D (-1) = -8) (h3 : g A B C D 0 = 2) :
  -12 * A + 6 * B - 3 * C + D = 27.5 :=
by
  sorry

end evaluate_expression_l250_250234


namespace exists_nat_sol_x9_eq_2013y10_l250_250342

theorem exists_nat_sol_x9_eq_2013y10 : ∃ (x y : ℕ), x^9 = 2013 * y^10 :=
by {
  -- Assume x and y are natural numbers, and prove that x^9 = 2013 y^10 has a solution
  sorry
}

end exists_nat_sol_x9_eq_2013y10_l250_250342


namespace quadratic_has_real_roots_iff_l250_250235

theorem quadratic_has_real_roots_iff (a : ℝ) :
  (∃ (x : ℝ), a * x^2 - 4 * x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) := by
  sorry

end quadratic_has_real_roots_iff_l250_250235


namespace solve_for_x_l250_250150

theorem solve_for_x (x : ℂ) (i : ℂ) (h : i ^ 2 = -1) (eqn : 3 + i * x = 5 - 2 * i * x) : x = i / 3 :=
sorry

end solve_for_x_l250_250150


namespace sum_of_factors_coefficients_l250_250001

theorem sum_of_factors_coefficients (a b c d e f g h i j k l m n o p : ℤ) :
  (81 * x^8 - 256 * y^8 = (a * x + b * y) *
                        (c * x^2 + d * x * y + e * y^2) *
                        (f * x^3 + g * x * y^2 + h * y^3) *
                        (i * x + j * y) *
                        (k * x^2 + l * x * y + m * y^2) *
                        (n * x^3 + o * x * y^2 + p * y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
by
  sorry

end sum_of_factors_coefficients_l250_250001


namespace largest_sphere_radius_on_torus_l250_250078

theorem largest_sphere_radius_on_torus
  (inner_radius outer_radius : ℝ)
  (torus_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (sphere_radius : ℝ)
  (sphere_center : ℝ × ℝ × ℝ) :
  inner_radius = 3 →
  outer_radius = 5 →
  torus_center = (4, 0, 1) →
  circle_radius = 1 →
  sphere_center = (0, 0, sphere_radius) →
  sphere_radius = 4 :=
by
  intros h_inner_radius h_outer_radius h_torus_center h_circle_radius h_sphere_center
  sorry

end largest_sphere_radius_on_torus_l250_250078


namespace area_of_triangle_DEF_l250_250749

theorem area_of_triangle_DEF :
  ∃ (DEF : Type) (area_u1 area_u2 area_u3 area_triangle : ℝ),
  area_u1 = 25 ∧
  area_u2 = 16 ∧
  area_u3 = 64 ∧
  area_triangle = area_u1 + area_u2 + area_u3 ∧
  area_triangle = 289 :=
by
  sorry

end area_of_triangle_DEF_l250_250749


namespace joe_marshmallow_ratio_l250_250109

theorem joe_marshmallow_ratio (J : ℕ) (h1 : 21 / 3 = 7) (h2 : 1 / 2 * J = 49 - 7) : J / 21 = 4 :=
by
  sorry

end joe_marshmallow_ratio_l250_250109


namespace monotonic_interval_a_l250_250358

theorem monotonic_interval_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (2 * x - 2 * a) * (2 * 2 - 2 * a) ≥ 0 ∧ (2 * x - 2 * a) * (2 * 3 - 2 * a) ≥ 0) →
  a ≤ 2 ∨ a ≥ 3 := sorry

end monotonic_interval_a_l250_250358


namespace largest_gcd_sum780_l250_250890

theorem largest_gcd_sum780 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 780) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 390 ∧ (∀ (d' : ℕ), d' = Nat.gcd a b → d' ≤ 390) :=
sorry

end largest_gcd_sum780_l250_250890


namespace positive_difference_two_numbers_l250_250302

theorem positive_difference_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) : |x - y| = 8 := by
  sorry

end positive_difference_two_numbers_l250_250302


namespace perfect_squares_count_2000_l250_250664

theorem perfect_squares_count_2000 : 
  let count := (1 to 44).filter (λ x, 
    let ones_digit := (x * x) % 10 in 
    ones_digit = 4 ∨ ones_digit = 5 ∨ ones_digit = 6).length
  in
  count = 22 :=
by
  sorry

end perfect_squares_count_2000_l250_250664


namespace koschei_chests_l250_250530

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l250_250530


namespace total_cost_l250_250756

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 5

theorem total_cost : (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 31 := by
  sorry

end total_cost_l250_250756


namespace geometric_sequence_a5_l250_250653

variable (a : ℕ → ℝ) (q : ℝ)

axiom pos_terms : ∀ n, a n > 0

axiom a1a3_eq : a 1 * a 3 = 16
axiom a3a4_eq : a 3 + a 4 = 24

theorem geometric_sequence_a5 :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) → a 5 = 32 :=
by
  sorry

end geometric_sequence_a5_l250_250653


namespace smallest_sum_abc_d_l250_250418

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end smallest_sum_abc_d_l250_250418


namespace sufficient_condition_for_p_l250_250650

theorem sufficient_condition_for_p (m : ℝ) (h : 1 < m) : ∀ x : ℝ, x^2 - 2 * x + m > 0 :=
sorry

end sufficient_condition_for_p_l250_250650


namespace initial_students_l250_250757

theorem initial_students {f : ℕ → ℕ} {g : ℕ → ℕ} (h_f : ∀ t, t ≥ 15 * 60 + 3 → (f t = 4 * ((t - (15 * 60 + 3)) / 3 + 1))) 
    (h_g : ∀ t, t ≥ 15 * 60 + 10 → (g t = 8 * ((t - (15 * 60 + 10)) / 10 + 1))) 
    (students_at_1544 : f 15 * 60 + 44 - g 15 * 60 + 44 + initial = 27) : 
    initial = 3 := 
sorry

end initial_students_l250_250757


namespace not_perfect_power_probability_l250_250708

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

theorem not_perfect_power_probability :
  let total := 200
  let count_perfect_powers := 19
  let count_non_perfect_powers := total - count_perfect_powers
  (count_non_perfect_powers : ℚ) / total = 181 / 200 :=
by
  sorry

end not_perfect_power_probability_l250_250708


namespace max_area_of_triangle_l250_250262

theorem max_area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : 4 * (Real.cos (A / 2))^2 -  Real.cos (2 * (B + C)) = 7 / 2)
  (h3 : A + B + C = Real.pi) :
  (Real.sqrt 3 / 2 * b * c) ≤ Real.sqrt 3 :=
sorry

end max_area_of_triangle_l250_250262


namespace largest_real_number_condition_l250_250808

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l250_250808


namespace evaluate_g_at_4_l250_250391

def g (x : ℕ) := 5 * x + 2

theorem evaluate_g_at_4 : g 4 = 22 := by
  sorry

end evaluate_g_at_4_l250_250391


namespace natasha_average_speed_l250_250983

theorem natasha_average_speed :
  ∀ (time_up time_down : ℝ) (speed_up : ℝ),
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (2 * (time_up * speed_up) / (time_up + time_down) = 3) :=
by
  intros time_up time_down speed_up h_time_up h_time_down h_speed_up
  rw [h_time_up, h_time_down, h_speed_up]
  sorry

end natasha_average_speed_l250_250983


namespace die_total_dots_l250_250290

theorem die_total_dots :
  ∀ (face1 face2 face3 face4 face5 face6 : ℕ),
    face1 < face2 ∧ face2 < face3 ∧ face3 < face4 ∧ face4 < face5 ∧ face5 < face6 ∧
    (face2 - face1 ≥ 2) ∧ (face3 - face2 ≥ 2) ∧ (face4 - face3 ≥ 2) ∧ (face5 - face4 ≥ 2) ∧ (face6 - face5 ≥ 2) ∧
    (face3 ≠ face1 + 2) ∧ (face4 ≠ face2 + 2) ∧ (face5 ≠ face3 + 2) ∧ (face6 ≠ face4 + 2)
    → face1 + face2 + face3 + face4 + face5 + face6 = 27 :=
by {
  sorry
}

end die_total_dots_l250_250290


namespace circumradius_of_triangle_l250_250463

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 14) : 
  R = (35 * Real.sqrt 2) / 3 :=
by
  sorry

end circumradius_of_triangle_l250_250463


namespace pickles_per_cucumber_l250_250688

theorem pickles_per_cucumber (jars cucumbers vinegar_initial vinegar_left pickles_per_jar vinegar_per_jar total_pickles_per_cucumber : ℕ) 
    (h1 : jars = 4) 
    (h2 : cucumbers = 10) 
    (h3 : vinegar_initial = 100) 
    (h4 : vinegar_left = 60) 
    (h5 : pickles_per_jar = 12) 
    (h6 : vinegar_per_jar = 10) 
    (h7 : total_pickles_per_cucumber = 4): 
    total_pickles_per_cucumber = (vinegar_initial - vinegar_left) / vinegar_per_jar * pickles_per_jar / cucumbers := 
by 
  sorry

end pickles_per_cucumber_l250_250688


namespace largest_x_l250_250779

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l250_250779


namespace digit_7_occurrences_in_range_1_to_2017_l250_250849

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end digit_7_occurrences_in_range_1_to_2017_l250_250849


namespace find_m_l250_250250

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_m :
  let a := (-sqrt 3, m)
  let b := (2, 1)
  (dot_product a b = 0) → m = 2 * sqrt 3 :=
by
  sorry

end find_m_l250_250250


namespace CandyGivenToJanetEmily_l250_250814

noncomputable def initial_candy : ℝ := 78.5
noncomputable def candy_left_after_janet : ℝ := 68.75
noncomputable def candy_given_to_emily : ℝ := 2.25

theorem CandyGivenToJanetEmily :
  initial_candy - candy_left_after_janet + candy_given_to_emily = 12 := 
by
  sorry

end CandyGivenToJanetEmily_l250_250814


namespace calc_1_calc_2_calc_3_calc_4_l250_250200

-- Problem 1
theorem calc_1 : 26 - 7 + (-6) + 17 = 30 := 
by
  sorry

-- Problem 2
theorem calc_2 : -81 / (9 / 4) * (-4 / 9) / (-16) = -1 := 
by
  sorry

-- Problem 3
theorem calc_3 : ((2 / 3) - (3 / 4) + (1 / 6)) * (-36) = -3 := 
by
  sorry

-- Problem 4
theorem calc_4 : -1^4 + 12 / (-2)^2 + (1 / 4) * (-8) = 0 := 
by
  sorry


end calc_1_calc_2_calc_3_calc_4_l250_250200


namespace radius_of_larger_circle_l250_250573

theorem radius_of_larger_circle
  (r : ℝ) -- radius of the smaller circle
  (R : ℝ) -- radius of the larger circle
  (ratio : R = 4 * r) -- radii ratio 1:4
  (AC : ℝ) -- diameter of the larger circle
  (BC : ℝ) -- chord of the larger circle
  (AB : ℝ := 16) -- given condition AB = 16
  (diameter_AC : AC = 2 * R) -- AC is diameter of the larger circle
  (tangent : BC^2 = AB^2 + (2 * R)^2) -- Pythagorean theorem for the right triangle ABC
  :
  R = 32 := 
sorry

end radius_of_larger_circle_l250_250573


namespace largest_minus_smallest_l250_250716

-- Define the given conditions
def A : ℕ := 10 * 2 + 9
def B : ℕ := A - 16
def C : ℕ := B * 3

-- Statement to prove
theorem largest_minus_smallest : C - B = 26 := by
  sorry

end largest_minus_smallest_l250_250716


namespace max_value_is_zero_l250_250535

noncomputable def max_value (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : ℝ :=
  x^2 - y^2

theorem max_value_is_zero (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : max_value x y h = 0 :=
sorry

end max_value_is_zero_l250_250535


namespace abs_eq_iff_mul_nonpos_l250_250559

theorem abs_eq_iff_mul_nonpos (a b : ℝ) : |a - b| = |a| + |b| ↔ a * b ≤ 0 :=
sorry

end abs_eq_iff_mul_nonpos_l250_250559


namespace min_value_l250_250353

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (a - 1) * 1 + 1 * (2 * b) = 0) :
    (2 / a) + (1 / b) = 8 :=
  sorry

end min_value_l250_250353


namespace alice_book_payment_l250_250336

/--
Alice is in the UK and wants to purchase a book priced at £25.
If one U.S. dollar is equivalent to £0.75, 
then Alice needs to pay 33.33 USD for the book.
-/
theorem alice_book_payment :
  ∀ (price_gbp : ℝ) (conversion_rate : ℝ), 
  price_gbp = 25 → conversion_rate = 0.75 → 
  (price_gbp / conversion_rate) = 33.33 :=
by
  intros price_gbp conversion_rate hprice hrate
  rw [hprice, hrate]
  sorry

end alice_book_payment_l250_250336


namespace quadratic_function_solution_l250_250828

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1/2 * x

theorem quadratic_function_solution (f : ℝ → ℝ)
  (h1 : ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c))
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x+1) = f x + x + 1) :
  ∀ x, f x = 1/2 * x^2 + 1/2 * x :=
by
  sorry

end quadratic_function_solution_l250_250828


namespace proof_part1_proof_part2_l250_250245

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 3 * x

def condition1 (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x + 3 ≥ 0

def condition2 (a : ℝ) : Prop := 3 * 3^2 - 2 * a * 3 + 3 = 0

theorem proof_part1 (a : ℝ) : condition1 a → a ≤ 3 := 
sorry

theorem proof_part2 (a : ℝ) (ha : a = 5) : 
  f 1 a = -1 ∧ f 3 a = -9 ∧ f 5 a = 15 :=
sorry

end proof_part1_proof_part2_l250_250245


namespace gcd_84_126_l250_250226

-- Conditions
def a : ℕ := 84
def b : ℕ := 126

-- Theorem to prove gcd(a, b) = 42
theorem gcd_84_126 : Nat.gcd a b = 42 := by
  sorry

end gcd_84_126_l250_250226


namespace total_nominal_income_l250_250282

noncomputable def monthly_income (principal : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  principal * ((1 + rate) ^ months - 1)

def total_income : ℝ :=
  let rate := 0.06 / 12
  let principal := 8700
  (monthly_income principal rate 6) + 
  (monthly_income principal rate 5) + 
  (monthly_income principal rate 4) + 
  (monthly_income principal rate 3) + 
  (monthly_income principal rate 2) + 
  (monthly_income principal rate 1)

theorem total_nominal_income :
  total_income = 921.15 :=
by
  sorry

end total_nominal_income_l250_250282


namespace video_has_2600_dislikes_l250_250590

def likes := 3000
def initial_dislikes := 1500 + 100
def additional_dislikes := 1000
def total_dislikes := initial_dislikes + additional_dislikes

theorem video_has_2600_dislikes:
  total_dislikes = 2600 :=
by
  unfold likes initial_dislikes additional_dislikes total_dislikes
  sorry

end video_has_2600_dislikes_l250_250590


namespace missing_digit_divisibility_l250_250011

theorem missing_digit_divisibility (x : ℕ) (h1 : x < 10) :
  3 ∣ (1 + 3 + 5 + 7 + x + 2) ↔ x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end missing_digit_divisibility_l250_250011


namespace find_k_value_l250_250977

noncomputable def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| n => a + (n - 1) * d

theorem find_k_value (a d : ℤ) (k : ℕ) 
  (h1 : arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 24)
  (h2 : (Finset.range 11).sum (λ i => arithmetic_seq a d (5 + i)) = 110)
  (h3 : arithmetic_seq a d k = 16) : 
  k = 16 :=
sorry

end find_k_value_l250_250977


namespace snow_leopards_arrangement_l250_250981

theorem snow_leopards_arrangement :
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  end_positions * factorial_six = 1440 :=
by
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  show end_positions * factorial_six = 1440
  sorry

end snow_leopards_arrangement_l250_250981


namespace abs_eq_4_reciprocal_eq_self_l250_250710

namespace RationalProofs

-- Problem 1
theorem abs_eq_4 (x : ℚ) : |x| = 4 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Problem 2
theorem reciprocal_eq_self (x : ℚ) : x ≠ 0 → x⁻¹ = x ↔ x = 1 ∨ x = -1 :=
by sorry

end RationalProofs

end abs_eq_4_reciprocal_eq_self_l250_250710


namespace solution_set_l250_250146

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l250_250146


namespace binomial_product_l250_250214

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l250_250214


namespace symmetrical_point_l250_250972

-- Definition of symmetry with respect to the x-axis
def symmetrical (x y: ℝ) : ℝ × ℝ := (x, -y)

-- Coordinates of the original point A
def A : ℝ × ℝ := (-2, 3)

-- Coordinates of the symmetrical point
def symmetrical_A : ℝ × ℝ := symmetrical (-2) 3

-- The theorem we want to prove
theorem symmetrical_point :
  symmetrical_A = (-2, -3) :=
by
  -- Provide the proof here
  sorry

end symmetrical_point_l250_250972


namespace cos_alpha_third_quadrant_l250_250084

theorem cos_alpha_third_quadrant (α : ℝ) (hα1 : π < α ∧ α < 3 * π / 2) (hα2 : Real.tan α = 4 / 3) :
  Real.cos α = -3 / 5 :=
sorry

end cos_alpha_third_quadrant_l250_250084


namespace intervals_of_decrease_l250_250561

open Real

noncomputable def func (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem intervals_of_decrease :
  {x | deriv func x < 0 ∧ 0 < x ∧ x < 2 * π} =
  {x | (π / 6 < x ∧ x < π / 2) ∨ (5 * π / 6 < x ∧ x < 3 * π / 2)} :=
by
  sorry

end intervals_of_decrease_l250_250561


namespace cyclic_sum_inequality_l250_250075

open BigOperators

theorem cyclic_sum_inequality {n : ℕ} (h : 0 < n) (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i) :
  (∑ k in Finset.range n, a k / (a (k+1) + a (k+2))) > n / 4 := by
  sorry

end cyclic_sum_inequality_l250_250075


namespace star_inequalities_not_all_true_simultaneously_l250_250690

theorem star_inequalities_not_all_true_simultaneously
  (AB BC CD DE EF FG GH HK KL LA : ℝ)
  (h1 : BC > AB)
  (h2 : DE > CD)
  (h3 : FG > EF)
  (h4 : HK > GH)
  (h5 : LA > KL) :
  False :=
  sorry

end star_inequalities_not_all_true_simultaneously_l250_250690


namespace solution_of_system_l250_250134

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l250_250134


namespace inequality_proof_l250_250547

theorem inequality_proof (a b c : ℝ) (h : a ^ 2 + b ^ 2 + c ^ 2 = 3) :
  (a ^ 2) / (2 + b + c ^ 2) + (b ^ 2) / (2 + c + a ^ 2) + (c ^ 2) / (2 + a + b ^ 2) ≥ (a + b + c) ^ 2 / 12 :=
by sorry

end inequality_proof_l250_250547


namespace gaokun_population_scientific_notation_l250_250182

theorem gaokun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (425000 = a * 10^n) ∧ (a = 4.25) ∧ (n = 5) :=
by
  sorry

end gaokun_population_scientific_notation_l250_250182


namespace solve_eq_l250_250409

theorem solve_eq : ∃ x : ℝ, 6 * x - 4 * x = 380 - 10 * (x + 2) ∧ x = 30 := 
by
  sorry

end solve_eq_l250_250409


namespace probability_succeeding_third_attempt_l250_250600

theorem probability_succeeding_third_attempt :
  let total_keys := 5
  let successful_keys := 2
  let attempts := 3
  let prob := successful_keys / total_keys * (successful_keys / (total_keys - 1)) * (successful_keys / (total_keys - 2))
  prob = 1 / 5 := by
sorry

end probability_succeeding_third_attempt_l250_250600


namespace girls_more_than_boys_l250_250999

/-- 
In a class with 42 students, where the ratio of boys to girls is 3:4, 
prove that there are 6 more girls than boys.
-/
theorem girls_more_than_boys (students total_students : ℕ) (boys girls : ℕ) (ratio_boys_girls : 3 * girls = 4 * boys)
  (total_students_count : boys + girls = total_students)
  (total_students_value : total_students = 42) : girls - boys = 6 :=
by
  sorry

end girls_more_than_boys_l250_250999


namespace lowest_temperature_at_noon_l250_250296

theorem lowest_temperature_at_noon
  (L : ℤ) -- Denote lowest temperature as L
  (avg_temp : ℤ) -- Average temperature from Monday to Friday
  (max_range : ℤ) -- Maximum possible range of the temperature
  (h1 : avg_temp = 50) -- Condition 1: average temperature is 50
  (h2 : max_range = 50) -- Condition 2: maximum range is 50
  (total_temp : ℤ) -- Sum of temperatures from Monday to Friday
  (h3 : total_temp = 250) -- Sum of temperatures equals 5 * 50
  (h4 : total_temp = L + (L + 50) + (L + 50) + (L + 50) + (L + 50)) -- Sum represented in terms of L
  : L = 10 := -- Prove that L equals 10
sorry

end lowest_temperature_at_noon_l250_250296
