import Mathlib

namespace min_inequality_l1950_195080

theorem min_inequality (r s u v : ℝ) : 
  min (min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2)))) ≤ 1 / 4 :=
by sorry

end min_inequality_l1950_195080


namespace silk_per_dress_l1950_195019

theorem silk_per_dress (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (total_dresses : ℕ)
  (h1 : initial_silk = 600)
  (h2 : friends = 5)
  (h3 : silk_per_friend = 20)
  (h4 : total_dresses = 100)
  (remaining_silk := initial_silk - friends * silk_per_friend) :
  remaining_silk / total_dresses = 5 :=
by
  -- proof goes here
  sorry

end silk_per_dress_l1950_195019


namespace find_mass_of_aluminum_l1950_195097

noncomputable def mass_of_aluminum 
  (rho_A : ℝ) (rho_M : ℝ) (delta_m : ℝ) : ℝ :=
  rho_A * delta_m / (rho_M - rho_A)

theorem find_mass_of_aluminum :
  mass_of_aluminum 2700 8900 0.06 = 26 := by
  sorry

end find_mass_of_aluminum_l1950_195097


namespace shortest_distance_l1950_195090

-- Define the line and the circle
def is_on_line (P : ℝ × ℝ) : Prop := P.snd = P.fst - 1

def is_on_circle (Q : ℝ × ℝ) : Prop := Q.fst^2 + Q.snd^2 + 4 * Q.fst - 2 * Q.snd + 4 = 0

-- Define the square of the Euclidean distance between two points
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- State the theorem regarding the shortest distance between the points on the line and the circle
theorem shortest_distance : ∃ P Q : ℝ × ℝ, is_on_line P ∧ is_on_circle Q ∧ dist_squared P Q = 1 := sorry

end shortest_distance_l1950_195090


namespace tan_theta_value_l1950_195008

open Real

theorem tan_theta_value (θ : ℝ) (h : sin (θ / 2) - 2 * cos (θ / 2) = 0) : tan θ = -4 / 3 :=
sorry

end tan_theta_value_l1950_195008


namespace product_of_consecutive_nat_is_divisible_by_2_l1950_195029

theorem product_of_consecutive_nat_is_divisible_by_2 (n : ℕ) : 2 ∣ n * (n + 1) :=
sorry

end product_of_consecutive_nat_is_divisible_by_2_l1950_195029


namespace triangle_third_side_l1950_195023

noncomputable def length_of_third_side
  (a b : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * cosθ)

theorem triangle_third_side : 
  length_of_third_side 8 15 (Real.pi / 6) (Real.cos (Real.pi / 6)) = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by
  sorry

end triangle_third_side_l1950_195023


namespace rosie_pies_l1950_195037

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l1950_195037


namespace S_12_l1950_195075

variable {S : ℕ → ℕ}

-- Given conditions
axiom S_4 : S 4 = 4
axiom S_8 : S 8 = 12

-- Goal: Prove S_12
theorem S_12 : S 12 = 24 :=
by
  sorry

end S_12_l1950_195075


namespace necessarily_negative_l1950_195062

theorem necessarily_negative (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : -2 < b ∧ b < 0) (h3 : 0 < c ∧ c < 1) : b + c < 0 :=
sorry

end necessarily_negative_l1950_195062


namespace square_binomial_constant_l1950_195007

theorem square_binomial_constant (y : ℝ) : ∃ b : ℝ, (y^2 + 12*y + 50 = (y + 6)^2 + b) ∧ b = 14 := 
by
  sorry

end square_binomial_constant_l1950_195007


namespace min_value_of_reciprocal_sum_l1950_195086

variable (a b : ℝ)
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (condition : 2 * a + b = 1)

theorem min_value_of_reciprocal_sum : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof is skipped
  sorry

end min_value_of_reciprocal_sum_l1950_195086


namespace number_of_lists_correct_l1950_195066

noncomputable def number_of_lists : Nat :=
  15 ^ 4

theorem number_of_lists_correct :
  number_of_lists = 50625 := by
  sorry

end number_of_lists_correct_l1950_195066


namespace additional_kgs_l1950_195089

variables (P R A : ℝ)
variables (h1 : R = 0.80 * P) (h2 : R = 34.2) (h3 : 684 = A * R)

theorem additional_kgs :
  A = 20 :=
by
  sorry

end additional_kgs_l1950_195089


namespace find_d_l1950_195073

-- Define the six-digit number as a function of d
def six_digit_num (d : ℕ) : ℕ := 3 * 100000 + 2 * 10000 + 5 * 1000 + 4 * 100 + 7 * 10 + d

-- Define the sum of digits of the six-digit number
def sum_of_digits (d : ℕ) : ℕ := 3 + 2 + 5 + 4 + 7 + d

-- The statement we want to prove
theorem find_d (d : ℕ) : sum_of_digits d % 3 = 0 ↔ d = 3 :=
by
  sorry

end find_d_l1950_195073


namespace power_of_negative_125_l1950_195038

theorem power_of_negative_125 : (-125 : ℝ)^(4/3) = 625 := by
  sorry

end power_of_negative_125_l1950_195038


namespace real_solution_x_condition_l1950_195026

theorem real_solution_x_condition (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
by
  sorry

end real_solution_x_condition_l1950_195026


namespace inequality_sqrt_l1950_195050

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l1950_195050


namespace quadratic_root_in_interval_l1950_195014

theorem quadratic_root_in_interval 
  (a b c : ℝ) 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_root_in_interval_l1950_195014


namespace captain_age_l1950_195059

theorem captain_age (C : ℕ) (h1 : ∀ W : ℕ, W = C + 3) 
                    (h2 : 21 * 11 = 231) 
                    (h3 : 21 - 1 = 20) 
                    (h4 : 20 * 9 = 180)
                    (h5 : 231 - 180 = 51) :
  C = 24 :=
by
  sorry

end captain_age_l1950_195059


namespace find_first_term_and_common_difference_l1950_195006

variable (a d : ℕ)
variable (S_even S_odd S_total : ℕ)

-- Conditions
axiom condition1 : S_total = 354
axiom condition2 : S_even = 192
axiom condition3 : S_odd = 162
axiom condition4 : 12*(2*a + 11*d) = 2*S_total
axiom condition5 : 6*(a + 6*d) = S_even
axiom condition6 : 6*(a + 5*d) = S_odd

-- Theorem to prove
theorem find_first_term_and_common_difference (a d S_even S_odd S_total : ℕ)
  (h1 : S_total = 354)
  (h2 : S_even = 192)
  (h3 : S_odd = 162)
  (h4 : 12*(2*a + 11*d) = 2*S_total)
  (h5 : 6*(a + 6*d) = S_even)
  (h6 : 6*(a + 5*d) = S_odd) : a = 2 ∧ d = 5 := by
  sorry

end find_first_term_and_common_difference_l1950_195006


namespace polynomial_no_ab_term_l1950_195083

theorem polynomial_no_ab_term (a b m : ℝ) :
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  ∃ (m : ℝ), (p = a^2 - 12 * b^2) → (m = -2) :=
by
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  intro h
  use -2
  sorry

end polynomial_no_ab_term_l1950_195083


namespace determine_digits_in_base_l1950_195072

theorem determine_digits_in_base (x y z b : ℕ) (h1 : 1993 = x * b^2 + y * b + z) (h2 : x + y + z = 22) :
  x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 :=
sorry

end determine_digits_in_base_l1950_195072


namespace average_of_middle_three_l1950_195061

-- Define the conditions based on the problem statement
def isPositiveWhole (n: ℕ) := n > 0
def areDifferent (a b c d e: ℕ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def isMaximumDifference (a b c d e: ℕ) := max a (max b (max c (max d e))) - min a (min b (min c (min d e)))
def isSecondSmallest (a b c d e: ℕ) := b = 3 ∧ (a < b ∧ (c < b ∨ d < b ∨ e < b) ∧ areDifferent a b c d e)
def totalSumIs30 (a b c d e: ℕ) := a + b + c + d + e = 30

-- Average of the middle three numbers calculated
theorem average_of_middle_three {a b c d e: ℕ} (cond1: isPositiveWhole a)
  (cond2: isPositiveWhole b) (cond3: isPositiveWhole c) (cond4: isPositiveWhole d)
  (cond5: isPositiveWhole e) (cond6: areDifferent a b c d e) (cond7: b = 3)
  (cond8: max a (max c (max d e)) - min a (min c (min d e)) = 16)
  (cond9: totalSumIs30 a b c d e) : (a + c + d) / 3 = 4 :=
by sorry

end average_of_middle_three_l1950_195061


namespace additional_emails_per_day_l1950_195000

theorem additional_emails_per_day
  (emails_per_day_before : ℕ)
  (half_days : ℕ)
  (total_days : ℕ)
  (total_emails : ℕ)
  (emails_received_first_half : ℕ := emails_per_day_before * half_days)
  (emails_received_second_half : ℕ := total_emails - emails_received_first_half)
  (emails_per_day_after : ℕ := emails_received_second_half / half_days) :
  emails_per_day_before = 20 → half_days = 15 → total_days = 30 → total_emails = 675 → (emails_per_day_after - emails_per_day_before = 5) :=
by
  intros
  sorry

end additional_emails_per_day_l1950_195000


namespace necessary_but_not_sufficient_l1950_195087

variable (a : ℝ)

theorem necessary_but_not_sufficient : (a > 2) → (a > 1) ∧ ¬((a > 1) → (a > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l1950_195087


namespace minimum_ratio_cone_cylinder_l1950_195077

theorem minimum_ratio_cone_cylinder (r : ℝ) (h : ℝ) (a : ℝ) :
  (h = 4 * r) →
  (a^2 = r^2 * h^2 / (h - 2 * r)) →
  (∀ h > 0, ∃ V_cone V_cylinder, 
    V_cone = (1/3) * π * a^2 * h ∧ 
    V_cylinder = π * r^2 * (2 * r) ∧ 
    V_cone / V_cylinder = (4 / 3)) := 
sorry

end minimum_ratio_cone_cylinder_l1950_195077


namespace condition_sufficient_but_not_necessary_l1950_195063

variables (p q : Prop)

theorem condition_sufficient_but_not_necessary (hpq : ∀ q, (¬p → ¬q)) (hpns : ¬ (¬p → ¬q ↔ p → q)) : (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end condition_sufficient_but_not_necessary_l1950_195063


namespace min_value_of_expr_l1950_195049

noncomputable def min_value (x y : ℝ) : ℝ :=
  (4 * x^2) / (y + 1) + (y^2) / (2*x + 2)

theorem min_value_of_expr : 
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (2 * x + y = 2) →
  min_value x y = 4 / 5 :=
by
  intros x y hx hy hxy
  sorry

end min_value_of_expr_l1950_195049


namespace isabella_paint_area_l1950_195002

-- Lean 4 statement for the proof problem based on given conditions and question:
theorem isabella_paint_area :
  let length := 15
  let width := 12
  let height := 9
  let door_and_window_area := 80
  let number_of_bedrooms := 4
  (2 * (length * height) + 2 * (width * height) - door_and_window_area) * number_of_bedrooms = 1624 :=
by
  sorry

end isabella_paint_area_l1950_195002


namespace impossible_distance_l1950_195095

noncomputable def radius_O1 : ℝ := 2
noncomputable def radius_O2 : ℝ := 5

theorem impossible_distance :
  ∀ (d : ℝ), ¬ (radius_O1 ≠ radius_O2 → ¬ (d < abs (radius_O2 - radius_O1) ∨ d > radius_O2 + radius_O1) → d = 5) :=
by
  sorry

end impossible_distance_l1950_195095


namespace max_value_of_squares_l1950_195046

theorem max_value_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18) 
  (h2 : ab + c + d = 91) 
  (h3 : ad + bc = 187) 
  (h4 : cd = 105) : 
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
sorry

end max_value_of_squares_l1950_195046


namespace shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l1950_195035

theorem shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder
  (c : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (hr : r ≥ 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y z : ℝ), (z = c) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ :=
by
  sorry

end shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l1950_195035


namespace expected_turns_formula_l1950_195011

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n +  1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1)))

theorem expected_turns_formula (n : ℕ) (h : n ≥ 1) :
  expected_turns n = n + 1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1))) :=
by
  sorry

end expected_turns_formula_l1950_195011


namespace total_volume_of_5_cubes_is_135_l1950_195022

-- Define the edge length of a single cube
def edge_length : ℕ := 3

-- Define the volume of a single cube
def volume_single_cube (s : ℕ) : ℕ := s^3

-- State the total volume for a given number of cubes
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_single_cube s

-- Prove that for 5 cubes with an edge length of 3 meters, the total volume is 135 cubic meters
theorem total_volume_of_5_cubes_is_135 :
    total_volume 5 edge_length = 135 :=
by
  sorry

end total_volume_of_5_cubes_is_135_l1950_195022


namespace compound_interest_comparison_l1950_195016

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l1950_195016


namespace infinite_series_equals_3_l1950_195068

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end infinite_series_equals_3_l1950_195068


namespace find_d_l1950_195004

-- Defining the basic points and their corresponding conditions
structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def a : Point ℝ := ⟨1, 0, 1⟩
def b : Point ℝ := ⟨0, 1, 0⟩
def c : Point ℝ := ⟨0, 1, 1⟩

-- introducing k as a positive integer
variables (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1)

def d (k : ℤ) : Point ℝ := ⟨k*d, k*d, -d⟩ where d := -(k / (k-1))

-- The proof statement
theorem find_d (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1) :
∃ d: ℝ, d = - (k / (k-1)) :=
sorry

end find_d_l1950_195004


namespace product_of_intersection_coordinates_l1950_195045

noncomputable def circle1 := {P : ℝ×ℝ | (P.1^2 - 4*P.1 + P.2^2 - 8*P.2 + 20) = 0}
noncomputable def circle2 := {P : ℝ×ℝ | (P.1^2 - 6*P.1 + P.2^2 - 8*P.2 + 25) = 0}

theorem product_of_intersection_coordinates :
  ∀ P ∈ circle1 ∩ circle2, P = (2, 4) → (P.1 * P.2) = 8 :=
by
  sorry

end product_of_intersection_coordinates_l1950_195045


namespace inequality_least_one_l1950_195070

theorem inequality_least_one {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4) :=
by
  sorry

end inequality_least_one_l1950_195070


namespace cat_catches_total_birds_l1950_195064

theorem cat_catches_total_birds :
  let morning_birds := 15
  let morning_success_rate := 0.60
  let afternoon_birds := 25
  let afternoon_success_rate := 0.80
  let night_birds := 20
  let night_success_rate := 0.90
  
  let morning_caught := morning_birds * morning_success_rate
  let afternoon_initial_caught := 2 * morning_caught
  let afternoon_caught := min (afternoon_birds * afternoon_success_rate) afternoon_initial_caught
  let night_caught := night_birds * night_success_rate

  let total_caught := morning_caught + afternoon_caught + night_caught
  total_caught = 47 := 
by
  sorry

end cat_catches_total_birds_l1950_195064


namespace total_pennies_l1950_195042

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l1950_195042


namespace min_students_with_blue_eyes_and_backpack_l1950_195079

theorem min_students_with_blue_eyes_and_backpack :
  ∀ (students : Finset ℕ), 
  (∀ s, s ∈ students → s = 1) →
  ∃ A B : Finset ℕ, 
    A.card = 18 ∧ B.card = 24 ∧ students.card = 35 ∧ 
    (A ∩ B).card ≥ 7 :=
by
  sorry

end min_students_with_blue_eyes_and_backpack_l1950_195079


namespace number_of_cards_per_page_l1950_195091

variable (packs : ℕ) (cards_per_pack : ℕ) (total_pages : ℕ)

def number_of_cards (packs cards_per_pack : ℕ) : ℕ :=
  packs * cards_per_pack

def cards_per_page (total_cards total_pages : ℕ) : ℕ :=
  total_cards / total_pages

theorem number_of_cards_per_page
  (packs := 60) (cards_per_pack := 7) (total_pages := 42)
  (total_cards := number_of_cards packs cards_per_pack)
    : cards_per_page total_cards total_pages = 10 :=
by {
  sorry
}

end number_of_cards_per_page_l1950_195091


namespace average_age_of_remaining_people_l1950_195032

theorem average_age_of_remaining_people:
  ∀ (ages : List ℕ), 
  (List.length ages = 8) →
  (List.sum ages = 224) →
  (24 ∈ ages) →
  ((List.sum ages - 24) / 7 = 28 + 4/7) := 
by
  intro ages
  intro h_len
  intro h_sum
  intro h_24
  sorry

end average_age_of_remaining_people_l1950_195032


namespace find_larger_number_l1950_195052

theorem find_larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
  sorry

end find_larger_number_l1950_195052


namespace minimum_positive_period_l1950_195088

open Real

noncomputable def function := fun x : ℝ => 3 * sin (2 * x + π / 3)

theorem minimum_positive_period : ∃ T > 0, ∀ x, function (x + T) = function x ∧ (∀ T', T' > 0 → (∀ x, function (x + T') = function x) → T ≤ T') :=
  sorry

end minimum_positive_period_l1950_195088


namespace rate_in_still_water_l1950_195047

theorem rate_in_still_water (with_stream_speed against_stream_speed : ℕ) 
  (h₁ : with_stream_speed = 16) 
  (h₂ : against_stream_speed = 12) : 
  (with_stream_speed + against_stream_speed) / 2 = 14 := 
by
  sorry

end rate_in_still_water_l1950_195047


namespace gcd_fact_8_10_l1950_195031

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end gcd_fact_8_10_l1950_195031


namespace alpha_beta_roots_l1950_195048

variable (α β : ℝ)

theorem alpha_beta_roots (h1 : α^2 - 7 * α + 3 = 0) (h2 : β^2 - 7 * β + 3 = 0) (h3 : α > β) :
  α^2 + 7 * β = 46 :=
sorry

end alpha_beta_roots_l1950_195048


namespace fraction_increase_by_three_l1950_195074

variables (a b : ℝ)

theorem fraction_increase_by_three : 
  3 * (2 * a * b / (3 * a - 4 * b)) = 2 * (3 * a * 3 * b) / (3 * (3 * a) - 4 * (3 * b)) :=
by
  sorry

end fraction_increase_by_three_l1950_195074


namespace C_pow_50_l1950_195055

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℝ :=
![![3, 1], ![-4, -1]]

theorem C_pow_50 :
  (C ^ 50) = ![![101, 50], ![-200, -99]] :=
by
  sorry

end C_pow_50_l1950_195055


namespace vasya_can_win_l1950_195082

noncomputable def initial_first : ℝ := 1 / 2009
noncomputable def initial_second : ℝ := 1 / 2008
noncomputable def increment : ℝ := 1 / (2008 * 2009)

theorem vasya_can_win :
  ∃ n : ℕ, ((2009 * n) * increment = 1) ∨ ((2008 * n) * increment = 1) :=
sorry

end vasya_can_win_l1950_195082


namespace students_interested_in_both_l1950_195084

theorem students_interested_in_both (total_students interested_in_sports interested_in_entertainment not_interested interested_in_both : ℕ)
  (h_total_students : total_students = 1400)
  (h_interested_in_sports : interested_in_sports = 1250)
  (h_interested_in_entertainment : interested_in_entertainment = 952)
  (h_not_interested : not_interested = 60)
  (h_equation : not_interested + interested_in_both + (interested_in_sports - interested_in_both) + (interested_in_entertainment - interested_in_both) = total_students) :
  interested_in_both = 862 :=
by
  sorry

end students_interested_in_both_l1950_195084


namespace number_of_full_rows_in_first_field_l1950_195028

-- Define the conditions
def total_corn_cobs : ℕ := 116
def rows_in_second_field : ℕ := 16
def cobs_per_row : ℕ := 4
def cobs_in_second_field : ℕ := rows_in_second_field * cobs_per_row
def cobs_in_first_field : ℕ := total_corn_cobs - cobs_in_second_field

-- Define the theorem to be proven
theorem number_of_full_rows_in_first_field : 
  cobs_in_first_field / cobs_per_row = 13 :=
by
  sorry

end number_of_full_rows_in_first_field_l1950_195028


namespace largest_perfect_square_factor_1800_l1950_195081

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  if n = 1800 then 900 else sorry

theorem largest_perfect_square_factor_1800 : 
  largest_perfect_square_factor 1800 = 900 :=
by
  -- Proof is not needed, so we use sorry
  sorry

end largest_perfect_square_factor_1800_l1950_195081


namespace find_theta_perpendicular_l1950_195018

theorem find_theta_perpendicular (θ : ℝ) (hθ : 0 < θ ∧ θ < π)
  (a b : ℝ × ℝ) (ha : a = (Real.sin θ, 1)) (hb : b = (2 * Real.cos θ, -1))
  (hperp : a.fst * b.fst + a.snd * b.snd = 0) : θ = π / 4 :=
by
  -- Proof would be written here
  sorry

end find_theta_perpendicular_l1950_195018


namespace find_a_l1950_195060

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l1950_195060


namespace solve_equation_l1950_195044

theorem solve_equation : ∀ (x : ℝ), -2 * x + 3 - 2 * x + 3 = 3 * x - 6 → x = 12 / 7 :=
by 
  intro x
  intro h
  sorry

end solve_equation_l1950_195044


namespace night_crew_fraction_of_day_l1950_195067

variable (D : ℕ) -- Number of workers in the day crew
variable (N : ℕ) -- Number of workers in the night crew
variable (total_boxes : ℕ) -- Total number of boxes loaded by both crews

-- Given conditions
axiom day_fraction : D > 0 ∧ N > 0 ∧ total_boxes > 0
axiom night_workers_fraction : N = (4 * D) / 5
axiom day_crew_boxes_fraction : (5 * total_boxes) / 7 = (5 * D)
axiom night_crew_boxes_fraction : (2 * total_boxes) / 7 = (2 * N)

-- To prove
theorem night_crew_fraction_of_day : 
  let F_d := (5 : ℚ) / (7 * D)
  let F_n := (2 : ℚ) / (7 * N)
  F_n = (5 / 14) * F_d :=
by
  sorry

end night_crew_fraction_of_day_l1950_195067


namespace number_of_adults_l1950_195030

-- Define the constants and conditions of the problem.
def children : ℕ := 52
def total_seats : ℕ := 95
def empty_seats : ℕ := 14

-- Define the number of adults and prove it equals 29 given the conditions.
theorem number_of_adults : total_seats - empty_seats - children = 29 :=
by {
  sorry
}

end number_of_adults_l1950_195030


namespace probability_blue_given_popped_is_18_over_53_l1950_195096

section PopcornProblem

/-- Representation of probabilities -/
def prob_white : ℚ := 1 / 2
def prob_yellow : ℚ := 1 / 4
def prob_blue : ℚ := 1 / 4

def pop_white_given_white : ℚ := 1 / 2
def pop_yellow_given_yellow : ℚ := 3 / 4
def pop_blue_given_blue : ℚ := 9 / 10

/-- Joint probabilities of kernel popping -/
def prob_white_popped : ℚ := prob_white * pop_white_given_white
def prob_yellow_popped : ℚ := prob_yellow * pop_yellow_given_yellow
def prob_blue_popped : ℚ := prob_blue * pop_blue_given_blue

/-- Total probability of popping -/
def prob_popped : ℚ := prob_white_popped + prob_yellow_popped + prob_blue_popped

/-- Conditional probability of being a blue kernel given that it popped -/
def prob_blue_given_popped : ℚ := prob_blue_popped / prob_popped

/-- The main theorem to prove the final probability -/
theorem probability_blue_given_popped_is_18_over_53 :
  prob_blue_given_popped = 18 / 53 :=
by sorry

end PopcornProblem

end probability_blue_given_popped_is_18_over_53_l1950_195096


namespace graded_worksheets_before_l1950_195041

-- Definitions based on conditions
def initial_worksheets : ℕ := 34
def additional_worksheets : ℕ := 36
def total_worksheets : ℕ := 63

-- Equivalent proof problem statement
theorem graded_worksheets_before (x : ℕ) (h₁ : initial_worksheets - x + additional_worksheets = total_worksheets) : x = 7 :=
by sorry

end graded_worksheets_before_l1950_195041


namespace correct_equation_l1950_195003

theorem correct_equation (a b : ℝ) : (a - b) ^ 3 * (b - a) ^ 4 = (a - b) ^ 7 :=
sorry

end correct_equation_l1950_195003


namespace find_number_that_satisfies_congruences_l1950_195020

theorem find_number_that_satisfies_congruences :
  ∃ m : ℕ, 
  (m % 13 = 12) ∧ 
  (m % 11 = 10) ∧ 
  (m % 7 = 6) ∧ 
  (m % 5 = 4) ∧ 
  (m % 3 = 2) ∧ 
  m = 15014 :=
by
  sorry

end find_number_that_satisfies_congruences_l1950_195020


namespace necessary_and_sufficient_condition_l1950_195040

variable (a b : ℝ)

theorem necessary_and_sufficient_condition:
  (ab + 1 ≠ a + b) ↔ (a ≠ 1 ∧ b ≠ 1) :=
sorry

end necessary_and_sufficient_condition_l1950_195040


namespace kevin_hop_distance_l1950_195043

theorem kevin_hop_distance :
  (1/4) + (3/16) + (9/64) + (27/256) + (81/1024) + (243/4096) = 3367 / 4096 := 
by
  sorry 

end kevin_hop_distance_l1950_195043


namespace age_difference_l1950_195015

variable (A B C : ℕ)

def condition1 := C = B / 2
def condition2 := A + B + C = 22
def condition3 := B = 8

theorem age_difference (h1 : condition1 C B)
                       (h2 : condition2 A B C) 
                       (h3 : condition3 B) : A - B = 2 := by
  sorry

end age_difference_l1950_195015


namespace annual_income_of_A_l1950_195057

def monthly_income_ratios (A_income B_income : ℝ) : Prop := A_income / B_income = 5 / 2
def B_income_increase (B_income C_income : ℝ) : Prop := B_income = C_income + 0.12 * C_income

theorem annual_income_of_A (A_income B_income C_income : ℝ)
  (h1 : monthly_income_ratios A_income B_income)
  (h2 : B_income_increase B_income C_income)
  (h3 : C_income = 13000) :
  12 * A_income = 436800 :=
by 
  sorry

end annual_income_of_A_l1950_195057


namespace length_more_than_breadth_by_200_percent_l1950_195001

noncomputable def length: ℝ := 19.595917942265423
noncomputable def total_cost: ℝ := 640
noncomputable def rate_per_sq_meter: ℝ := 5

theorem length_more_than_breadth_by_200_percent
  (area : ℝ := total_cost / rate_per_sq_meter)
  (breadth : ℝ := area / length) :
  ((length - breadth) / breadth) * 100 = 200 := by
  have h1 : area = 128 := by sorry
  have h2 : breadth = 128 / 19.595917942265423 := by sorry
  rw [h1, h2]
  sorry

end length_more_than_breadth_by_200_percent_l1950_195001


namespace sqrt_product_eq_l1950_195094

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l1950_195094


namespace subtraction_is_addition_of_negatives_l1950_195071

theorem subtraction_is_addition_of_negatives : (-1) - 3 = -4 := by
  sorry

end subtraction_is_addition_of_negatives_l1950_195071


namespace jonessa_total_pay_l1950_195009

theorem jonessa_total_pay (total_pay : ℝ) (take_home_pay : ℝ) (h1 : take_home_pay = 450) (h2 : 0.90 * total_pay = take_home_pay) : total_pay = 500 :=
by
  sorry

end jonessa_total_pay_l1950_195009


namespace exists_pythagorean_number_in_range_l1950_195010

def is_pythagorean_area (a : ℕ) : Prop :=
  ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ a = (x * y) / 2

theorem exists_pythagorean_number_in_range (n : ℕ) (hn : n > 12) : 
  ∃ (m : ℕ), is_pythagorean_area m ∧ n < m ∧ m < 2 * n :=
sorry

end exists_pythagorean_number_in_range_l1950_195010


namespace prob_fourth_black_ball_is_half_l1950_195013

-- Define the conditions
def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_black_balls

-- The theorem stating that the probability of drawing a black ball on the fourth draw is 1/2
theorem prob_fourth_black_ball_is_half : 
  (num_black_balls : ℚ) / (total_balls : ℚ) = 1 / 2 :=
by
  sorry

end prob_fourth_black_ball_is_half_l1950_195013


namespace quadruples_solution_l1950_195054

theorem quadruples_solution (a b c d : ℝ) :
  (a * b + c * d = 6) ∧
  (a * c + b * d = 3) ∧
  (a * d + b * c = 2) ∧
  (a + b + c + d = 6) ↔
  (a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0) :=
sorry

end quadruples_solution_l1950_195054


namespace simplified_factorial_fraction_l1950_195076

theorem simplified_factorial_fraction :
  (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 :=
by
  sorry

end simplified_factorial_fraction_l1950_195076


namespace find_p_from_parabola_and_distance_l1950_195085

theorem find_p_from_parabola_and_distance 
  (p : ℝ) (hp : p > 0) 
  (M : ℝ × ℝ) (hM : M = (8 / p, 4))
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hMF : dist M F = 4) : 
  p = 4 :=
sorry

end find_p_from_parabola_and_distance_l1950_195085


namespace employee_overtime_hours_l1950_195027

theorem employee_overtime_hours (gross_pay : ℝ) (rate_regular : ℝ) (regular_hours : ℕ) (rate_overtime : ℝ) :
  gross_pay = 622 → rate_regular = 11.25 → regular_hours = 40 → rate_overtime = 16 →
  ∃ (overtime_hours : ℕ), overtime_hours = 10 :=
by
  sorry

end employee_overtime_hours_l1950_195027


namespace parallel_lines_a_l1950_195065

-- Definitions of the lines
def l1 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 6 * x + a * y + 2 = 0

-- The main theorem to prove
theorem parallel_lines_a (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = 3) := 
sorry

end parallel_lines_a_l1950_195065


namespace sin_add_alpha_l1950_195053

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha_l1950_195053


namespace focus_of_parabola_l1950_195051

theorem focus_of_parabola :
  (∀ y : ℝ, x = (1 / 4) * y^2) → (focus = (-1, 0)) := by
  sorry

end focus_of_parabola_l1950_195051


namespace part1_part2_l1950_195078

noncomputable def f (a : ℝ) (x : ℝ) := (a * x - 1) * (x - 1)

theorem part1 (h : ∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) : a = 1/2 :=
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 1/a) ∨
  (a = 1 → ∀ x : ℝ, ¬(f a x < 0)) ∨
  (∀ x : ℝ, f a x < 0 ↔ 1/a < x ∧ x < 1) :=
  sorry

end part1_part2_l1950_195078


namespace percentage_difference_l1950_195033

variable (p : ℝ) (j : ℝ) (t : ℝ)

def condition_1 := j = 0.75 * p
def condition_2 := t = 0.9375 * p

theorem percentage_difference : (j = 0.75 * p) → (t = 0.9375 * p) → ((t - j) / t * 100 = 20) :=
by
  intros h1 h2
  rw [h1, h2]
  -- This will use the derived steps from the solution, and ultimately show 20
  sorry

end percentage_difference_l1950_195033


namespace country_x_income_l1950_195058

theorem country_x_income (I : ℝ) (h1 : I > 40000) (_ : 0.15 * 40000 + 0.20 * (I - 40000) = 8000) : I = 50000 :=
sorry

end country_x_income_l1950_195058


namespace train_speed_equals_36_0036_l1950_195092

noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_equals_36_0036 :
  train_speed 70 6.999440044796416 = 36.0036 :=
by
  unfold train_speed
  sorry

end train_speed_equals_36_0036_l1950_195092


namespace store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l1950_195017

-- Definitions and conditions
def cost_per_soccer : ℕ := 200
def cost_per_basketball : ℕ := 80
def discount_A_soccer (n : ℕ) : ℕ := n * cost_per_soccer
def discount_A_basketball (n : ℕ) : ℕ := if n > 100 then (n - 100) * cost_per_basketball else 0
def discount_B_soccer (n : ℕ) : ℕ := n * cost_per_soccer * 8 / 10
def discount_B_basketball (n : ℕ) : ℕ := n * cost_per_basketball * 8 / 10

-- For x = 100
def total_cost_A_100 : ℕ := discount_A_soccer 100 + discount_A_basketball 100
def total_cost_B_100 : ℕ := discount_B_soccer 100 + discount_B_basketball 100

-- Prove that for x = 100, Store A is more cost-effective
theorem store_A_more_cost_effective_100 : total_cost_A_100 < total_cost_B_100 :=
by sorry

-- For x > 100, express costs in terms of x
def total_cost_A (x : ℕ) : ℕ := 80 * x + 12000
def total_cost_B (x : ℕ) : ℕ := 64 * x + 16000

-- Prove the expressions for costs
theorem cost_expressions_for_x (x : ℕ) (h : x > 100) : 
  total_cost_A x = 80 * x + 12000 ∧ total_cost_B x = 64 * x + 16000 :=
by sorry

-- For x = 300, most cost-effective plan
def combined_A_100_B_200 : ℕ := (discount_A_soccer 100 + cost_per_soccer * 100) + (200 * cost_per_basketball * 8 / 10)
def only_A_300 : ℕ := discount_A_soccer 100 + (300 - 100) * cost_per_basketball
def only_B_300 : ℕ := discount_B_soccer 100 + 300 * cost_per_basketball * 8 / 10

-- Prove the most cost-effective plan for x = 300
theorem most_cost_effective_plan : combined_A_100_B_200 < only_B_300 ∧ combined_A_100_B_200 < only_A_300 :=
by sorry

end store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l1950_195017


namespace number_of_lizards_l1950_195039

theorem number_of_lizards (total_geckos : ℕ) (insects_per_gecko : ℕ) (total_insects_eaten : ℕ) (insects_per_lizard : ℕ) 
  (gecko_total_insects : total_geckos * insects_per_gecko = 5 * 6) (lizard_insects: insects_per_lizard = 2 * insects_per_gecko)
  (total_insects : total_insects_eaten = 66) : 
  (total_insects_eaten - total_geckos * insects_per_gecko) / insects_per_lizard = 3 :=
by 
  sorry

end number_of_lizards_l1950_195039


namespace cistern_emptying_time_l1950_195005

noncomputable def cistern_time_without_tap (tap_rate : ℕ) (empty_time_with_tap : ℕ) (cistern_volume : ℕ) : ℕ := 
  let tap_total := tap_rate * empty_time_with_tap
  let leaked_volume := cistern_volume - tap_total
  let leak_rate := leaked_volume / empty_time_with_tap
  cistern_volume / leak_rate

theorem cistern_emptying_time :
  cistern_time_without_tap 4 24 480 = 30 := 
by
  unfold cistern_time_without_tap
  norm_num

end cistern_emptying_time_l1950_195005


namespace necessary_but_not_sufficient_l1950_195021

noncomputable def represents_ellipse (m : ℝ) : Prop :=
  2 < m ∧ m < 6 ∧ m ≠ 4

theorem necessary_but_not_sufficient (m : ℝ) :
  represents_ellipse (m) ↔ (2 < m ∧ m < 6) :=
by
  sorry

end necessary_but_not_sufficient_l1950_195021


namespace necessary_but_not_sufficient_l1950_195093

-- Define the propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2 * a * b
def Q (a b : ℝ) : Prop := abs (a + b) < abs a + abs b

-- Define the conditions for P and Q
def condition_for_P (a b : ℝ) : Prop := a ≠ b
def condition_for_Q (a b : ℝ) : Prop := a * b < 0

-- Define the statement
theorem necessary_but_not_sufficient (a b : ℝ) :
  (P a b → Q a b) ∧ ¬ (Q a b → P a b) :=
by
  sorry

end necessary_but_not_sufficient_l1950_195093


namespace minimum_value_inequality_l1950_195034

theorem minimum_value_inequality {a b : ℝ} (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * Real.log b / Real.log a + 2 * Real.log a / Real.log b = 7) :
  a^2 + 3 / (b - 1) ≥ 2 * Real.sqrt 3 + 1 :=
sorry

end minimum_value_inequality_l1950_195034


namespace negation_of_proposition_l1950_195012

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∃ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_of_proposition_l1950_195012


namespace brownies_on_counter_l1950_195069

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l1950_195069


namespace billy_free_time_l1950_195036

theorem billy_free_time
  (play_time_percentage : ℝ := 0.75)
  (read_pages_per_hour : ℝ := 60)
  (book_pages : ℝ := 80)
  (number_of_books : ℝ := 3)
  (read_percentage : ℝ := 1 - play_time_percentage)
  (total_pages : ℝ := number_of_books * book_pages)
  (read_time_hours : ℝ := total_pages / read_pages_per_hour)
  (free_time_hours : ℝ := read_time_hours / read_percentage) :
  free_time_hours = 16 := 
sorry

end billy_free_time_l1950_195036


namespace debate_team_has_11_boys_l1950_195056

def debate_team_boys_count (num_groups : Nat) (members_per_group : Nat) (num_girls : Nat) : Nat :=
  let total_members := num_groups * members_per_group
  total_members - num_girls

theorem debate_team_has_11_boys :
  debate_team_boys_count 8 7 45 = 11 :=
by
  sorry

end debate_team_has_11_boys_l1950_195056


namespace parallel_line_slope_l1950_195025

theorem parallel_line_slope (x y : ℝ) : (∃ (c : ℝ), 3 * x - 6 * y = c) → (1 / 2) = 1 / 2 :=
by sorry

end parallel_line_slope_l1950_195025


namespace instrument_accuracy_confidence_l1950_195099

noncomputable def instrument_accuracy (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ) : ℝ × ℝ :=
  let lower := s * (1 - q)
  let upper := s * (1 + q)
  (lower, upper)

theorem instrument_accuracy_confidence :
  ∀ (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ),
    n = 12 →
    s = 0.6 →
    gamma = 0.99 →
    q = 0.9 →
    0.06 < (instrument_accuracy n s gamma q).fst ∧
    (instrument_accuracy n s gamma q).snd < 1.14 :=
by
  intros n s gamma q h_n h_s h_gamma h_q
  -- proof would go here
  sorry

end instrument_accuracy_confidence_l1950_195099


namespace merchant_installed_zucchini_l1950_195024

theorem merchant_installed_zucchini (Z : ℕ) : 
  (15 + Z + 8) / 2 = 18 → Z = 13 :=
by
 sorry

end merchant_installed_zucchini_l1950_195024


namespace hexagon_side_relation_l1950_195098

noncomputable def hexagon (a b c d e f : ℝ) :=
  ∃ (i j k l m n : ℝ), 
    i = 120 ∧ j = 120 ∧ k = 120 ∧ l = 120 ∧ m = 120 ∧ n = 120 ∧  
    a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f ∧ f = a

theorem hexagon_side_relation
  (a b c d e f : ℝ)
  (ha : hexagon a b c d e f) :
  d - a = b - e ∧ b - e = f - c :=
by
  sorry

end hexagon_side_relation_l1950_195098
