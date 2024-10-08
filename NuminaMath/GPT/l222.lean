import Mathlib

namespace a_eq_3_suff_not_nec_l222_222030

theorem a_eq_3_suff_not_nec (a : ℝ) : (a = 3 → a^2 = 9) ∧ (a^2 = 9 → ∃ b : ℝ, b = a ∧ (b = 3 ∨ b = -3)) :=
by
  sorry

end a_eq_3_suff_not_nec_l222_222030


namespace point_in_fourth_quadrant_l222_222525

theorem point_in_fourth_quadrant (θ : ℝ) (h : -1 < Real.cos θ ∧ Real.cos θ < 0) :
    ∃ (x y : ℝ), x = Real.sin (Real.cos θ) ∧ y = Real.cos (Real.cos θ) ∧ x < 0 ∧ y > 0 :=
by
  sorry

end point_in_fourth_quadrant_l222_222525


namespace problem_statement_l222_222144

noncomputable def ratio_AD_AB (AB AD : ℝ) (angle_A angle_B angle_ADE : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ angle_ADE = 45 ∧
  AD / AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2)

theorem problem_statement {AB AD : ℝ} (angle_A angle_B angle_ADE : ℝ) 
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45) : 
  ratio_AD_AB AB AD angle_A angle_B angle_ADE := by {
    sorry
}

end problem_statement_l222_222144


namespace pirates_on_schooner_l222_222129

def pirate_problem (N : ℝ) : Prop :=
  let total_pirates       := N
  let non_participants    := 10
  let participants        := total_pirates - non_participants
  let lost_arm            := 0.54 * participants
  let lost_arm_and_leg    := 0.34 * participants
  let lost_leg            := (2 / 3) * total_pirates
  -- The number of pirates who lost only a leg can be calculated.
  let lost_only_leg       := lost_leg - lost_arm_and_leg
  -- The equation that needs to be satisfied
  lost_leg = lost_arm_and_leg + lost_only_leg

theorem pirates_on_schooner : ∃ N : ℝ, N > 10 ∧ pirate_problem N :=
sorry

end pirates_on_schooner_l222_222129


namespace largest_angle_measure_l222_222813

noncomputable def measure_largest_angle (x : ℚ) : Prop :=
  let a1 := 2 * x + 2
  let a2 := 3 * x
  let a3 := 4 * x + 3
  let a4 := 5 * x
  let a5 := 6 * x - 1
  let a6 := 7 * x
  a1 + a2 + a3 + a4 + a5 + a6 = 720 ∧ a6 = 5012 / 27

theorem largest_angle_measure : ∃ x : ℚ, measure_largest_angle x := by
  sorry

end largest_angle_measure_l222_222813


namespace part1_part2_l222_222660

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 + Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp (1 - x) + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) : (∀ x > 0, f a x ≤ Real.exp 1) → a ≤ 1 := 
sorry

theorem part2 (a : ℝ) : (∃! x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3) → a = 3 :=
sorry

end part1_part2_l222_222660


namespace gum_total_l222_222852

theorem gum_total (initial_gum : ℝ) (additional_gum : ℝ) : initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 :=
by
  intros
  sorry

end gum_total_l222_222852


namespace train_pass_bridge_in_56_seconds_l222_222514

noncomputable def time_for_train_to_pass_bridge 
(length_of_train : ℕ) (speed_of_train_kmh : ℕ) (length_of_bridge : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  total_distance / speed_of_train_ms

theorem train_pass_bridge_in_56_seconds :
  time_for_train_to_pass_bridge 560 45 140 = 56 := by
  sorry

end train_pass_bridge_in_56_seconds_l222_222514


namespace least_integer_exists_l222_222927

theorem least_integer_exists (x : ℕ) (h1 : x = 10 * (x / 10) + x % 10) (h2 : (x / 10) = x / 17) : x = 17 :=
sorry

end least_integer_exists_l222_222927


namespace lcm_48_75_l222_222753

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end lcm_48_75_l222_222753


namespace articles_in_selling_price_l222_222633

theorem articles_in_selling_price (C : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * (1.25 * C)) 
  (h2 : 0.25 * C = 25 / 100 * C) :
  N = 40 :=
by
  sorry

end articles_in_selling_price_l222_222633


namespace cos_function_max_value_l222_222059

theorem cos_function_max_value (k : ℤ) : (2 * Real.cos (2 * k * Real.pi) - 1) = 1 :=
by
  -- Proof not included
  sorry

end cos_function_max_value_l222_222059


namespace boy_walking_speed_l222_222343

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

end boy_walking_speed_l222_222343


namespace range_of_square_root_l222_222716

theorem range_of_square_root (x : ℝ) : x + 4 ≥ 0 → x ≥ -4 :=
by
  intro h
  linarith

end range_of_square_root_l222_222716


namespace women_in_club_l222_222474

theorem women_in_club (total_members : ℕ) (men : ℕ) (total_members_eq : total_members = 52) (men_eq : men = 37) :
  ∃ women : ℕ, women = 15 :=
by
  sorry

end women_in_club_l222_222474


namespace find_larger_number_l222_222970

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1355) (h2 : L = 6 * S + 15) : L = 1623 :=
sorry

end find_larger_number_l222_222970


namespace independent_trials_probability_l222_222825

theorem independent_trials_probability (p : ℝ) (q : ℝ) (ε : ℝ) (desired_prob : ℝ) 
    (h_p : p = 0.7) (h_q : q = 0.3) (h_ε : ε = 0.2) (h_desired_prob : desired_prob = 0.96) :
    ∃ n : ℕ, n > (p * q) / (desired_prob * ε^2) ∧ n = 132 :=
by
  sorry

end independent_trials_probability_l222_222825


namespace find_costs_compare_options_l222_222142

-- Definitions and theorems
def cost1 (x y : ℕ) : Prop := 2 * x + 4 * y = 350
def cost2 (x y : ℕ) : Prop := 6 * x + 3 * y = 420

def optionACost (m : ℕ) : ℕ := 70 * m + 35 * (80 - 2 * m)
def optionBCost (m : ℕ) : ℕ := (8 * (35 * m + 2800)) / 10

theorem find_costs (x y : ℕ) : 
  cost1 x y ∧ cost2 x y → (x = 35 ∧ y = 70) :=
by sorry

theorem compare_options (m : ℕ) (h : m < 41) : 
  if m < 20 then optionBCost m < optionACost m else 
  if m = 20 then optionBCost m = optionACost m 
  else optionBCost m > optionACost m :=
by sorry

end find_costs_compare_options_l222_222142


namespace third_group_members_l222_222966

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l222_222966


namespace sufficient_not_necessary_l222_222188

def M : Set Int := {0, 1, 2}
def N : Set Int := {-1, 0, 1, 2}

theorem sufficient_not_necessary (a : Int) : a ∈ M → a ∈ N ∧ ¬(a ∈ N → a ∈ M) := by
  sorry

end sufficient_not_necessary_l222_222188


namespace sphere_surface_area_l222_222511

theorem sphere_surface_area (r : ℝ) (h : π * r^2 = 81 * π) : 4 * π * r^2 = 324 * π :=
  sorry

end sphere_surface_area_l222_222511


namespace product_of_two_numbers_in_ratio_l222_222770

theorem product_of_two_numbers_in_ratio (x y : ℚ) 
  (h1 : x - y = d)
  (h2 : x + y = 8 * d)
  (h3 : x * y = 15 * d) :
  x * y = 100 / 7 :=
by
  sorry

end product_of_two_numbers_in_ratio_l222_222770


namespace gcd_2_pow_2018_2_pow_2029_l222_222521

theorem gcd_2_pow_2018_2_pow_2029 : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2047 :=
by
  sorry

end gcd_2_pow_2018_2_pow_2029_l222_222521


namespace sum_of_coordinates_eq_69_l222_222092

theorem sum_of_coordinates_eq_69 {f k : ℝ → ℝ} (h₁ : f 4 = 8) (h₂ : ∀ x, k x = (f x)^2 + 1) : 4 + k 4 = 69 :=
by
  sorry

end sum_of_coordinates_eq_69_l222_222092


namespace work_completion_l222_222099

/-- 
  Let A, B, and C have work rates where:
  1. A completes the work in 4 days (work rate: 1/4 per day)
  2. C completes the work in 12 days (work rate: 1/12 per day)
  3. Together with B, they complete the work in 2 days (combined work rate: 1/2 per day)
  Prove that B alone can complete the work in 6 days.
--/
theorem work_completion (A B C : ℝ) (x : ℝ)
  (hA : A = 1/4)
  (hC : C = 1/12)
  (h_combined : A + 1/x + C = 1/2) :
  x = 6 := sorry

end work_completion_l222_222099


namespace perimeter_of_square_l222_222294

theorem perimeter_of_square
  (length_rect : ℕ) (width_rect : ℕ) (area_rect : ℕ)
  (area_square : ℕ) (side_square : ℕ) (perimeter_square : ℕ) :
  (length_rect = 32) → (width_rect = 10) → 
  (area_rect = length_rect * width_rect) →
  (area_square = 5 * area_rect) →
  (side_square * side_square = area_square) →
  (perimeter_square = 4 * side_square) →
  perimeter_square = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof would go here
  sorry

end perimeter_of_square_l222_222294


namespace remaining_amount_to_be_paid_l222_222910

theorem remaining_amount_to_be_paid (p : ℝ) (deposit : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (final_payment : ℝ) :
  deposit = 80 ∧ tax_rate = 0.07 ∧ discount_rate = 0.05 ∧ deposit = 0.1 * p ∧ 
  final_payment = (p - (discount_rate * p)) * (1 + tax_rate) - deposit → 
  final_payment = 733.20 :=
by
  sorry

end remaining_amount_to_be_paid_l222_222910


namespace complement_union_l222_222974

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {-1, 2}

def B : Set Int := {-1, 0, 1}

theorem complement_union :
  (U \ B) ∪ A = {-2, -1, 2} :=
by
  sorry

end complement_union_l222_222974


namespace sum_mnp_l222_222847

noncomputable def volume_of_parallelepiped := 2 * 3 * 4
noncomputable def volume_of_extended_parallelepipeds := 
  2 * (1 * 2 * 3 + 1 * 2 * 4 + 1 * 3 * 4)
noncomputable def volume_of_quarter_cylinders := 
  4 * (1 / 4 * Real.pi * 1^2 * (2 + 3 + 4))
noncomputable def volume_of_spherical_octants := 
  8 * (1 / 8 * (4 / 3) * Real.pi * 1^3)

noncomputable def total_volume := 
  volume_of_parallelepiped + volume_of_extended_parallelepipeds + 
  volume_of_quarter_cylinders + volume_of_spherical_octants

theorem sum_mnp : 228 + 85 + 3 = 316 := by
  sorry

end sum_mnp_l222_222847


namespace Carly_applications_l222_222371

theorem Carly_applications (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : x + 2 * x = 600) : x = 200 :=
sorry

end Carly_applications_l222_222371


namespace pythagorean_triple_solution_l222_222135

theorem pythagorean_triple_solution
  (x y z a b : ℕ)
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x y = 1)
  (h3 : 2 ∣ y)
  (h4 : a > b)
  (h5 : b > 0)
  (h6 : (Nat.gcd a b = 1))
  (h7 : ((a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) 
  : (x = a^2 - b^2 ∧ y = 2 * a * b ∧ z = a^2 + b^2) := 
sorry

end pythagorean_triple_solution_l222_222135


namespace find_m_plus_n_l222_222555

-- Define the number of ways Blair and Corey can draw the remaining cards
def num_ways_blair_and_corey_draw : ℕ := Nat.choose 50 2

-- Define the function q(a) as given in the problem
noncomputable def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / num_ways_blair_and_corey_draw

-- Define the problem statement to find the minimum value of a for which q(a) >= 1/2
noncomputable def minimum_a : ℤ :=
  if q 7 >= 1/2 then 7 else 36 -- According to the solution, these are the points of interest

-- The final statement to be proved
theorem find_m_plus_n : minimum_a = 7 ∨ minimum_a = 36 :=
  sorry

end find_m_plus_n_l222_222555


namespace circle_equation_l222_222118

def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem circle_equation : ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ∧ (y = parabola x) ∧ (x = -1 ∨ x = 3 ∨ (x = 0 ∧ y = -3)) :=
by { sorry }

end circle_equation_l222_222118


namespace place_value_diff_7669_l222_222790

theorem place_value_diff_7669 :
  let a := 6 * 10
  let b := 6 * 100
  b - a = 540 :=
by
  let a := 6 * 10
  let b := 6 * 100
  have h : b - a = 540 := by sorry
  exact h

end place_value_diff_7669_l222_222790


namespace quadratic_two_equal_real_roots_l222_222831

theorem quadratic_two_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x^2 + m * x + m = 0 ∧ ∀ (y : ℝ), x = y → x^2 + m * y + m = 0) →
  (m = 0 ∨ m = 4) :=
by {
  sorry
}

end quadratic_two_equal_real_roots_l222_222831


namespace direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l222_222333

-- Direct Proportional Function
theorem direct_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 1) → m = 1 :=
by 
  sorry

-- Inverse Proportional Function
theorem inverse_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = -1) → m = -1 :=
by 
  sorry

-- Quadratic Function
theorem quadratic_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 2) → (m = (-1 + Real.sqrt 13) / 2 ∨ m = (-1 - Real.sqrt 13) / 2) :=
by 
  sorry

-- Power Function
theorem power_function (m : ℝ) :
  (m^2 + 2 * m = 1) → (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) :=
by 
  sorry

end direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l222_222333


namespace evaluate_expression_l222_222566

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l222_222566


namespace simplify_fraction_l222_222361

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = (65 : ℚ) / 12 := 
by
  sorry

end simplify_fraction_l222_222361


namespace ants_first_group_count_l222_222801

theorem ants_first_group_count :
    ∃ x : ℕ, 
        (∀ (w1 c1 a1 t1 w2 c2 a2 t2 : ℕ),
          w1 = 10 ∧ c1 = 600 ∧ a1 = x ∧ t1 = 5 ∧
          w2 = 5 ∧ c2 = 960 ∧ a2 = 20 ∧ t2 = 3 ∧ 
          (w1 * c1) / t1 = 1200 / a1 ∧ (w2 * c2) / t2 = 1600 / 20 →
             x = 15)
:= sorry

end ants_first_group_count_l222_222801


namespace all_acute_angles_in_first_quadrant_l222_222558

def terminal_side_same (θ₁ θ₂ : ℝ) : Prop := 
  ∃ (k : ℤ), θ₁ = θ₂ + 360 * k

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def first_quadrant_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem all_acute_angles_in_first_quadrant :
  ∀ θ : ℝ, acute_angle θ → first_quadrant_angle θ :=
by
  intros θ h
  exact h

end all_acute_angles_in_first_quadrant_l222_222558


namespace thirteenth_result_is_128_l222_222518

theorem thirteenth_result_is_128 
  (avg_all : ℕ → ℕ → ℕ) (avg_first : ℕ → ℕ → ℕ) (avg_last : ℕ → ℕ → ℕ) :
  avg_all 25 20 = (avg_first 12 14) + (avg_last 12 17) + 128 :=
by
  sorry

end thirteenth_result_is_128_l222_222518


namespace cover_condition_l222_222410

theorem cover_condition (n : ℕ) :
  (∃ (f : ℕ) (h1 : f = n^2), f % 2 = 0) ↔ (n % 2 = 0) := 
sorry

end cover_condition_l222_222410


namespace ratio_of_earnings_l222_222479

theorem ratio_of_earnings (K V S : ℕ) (h1 : K + 30 = V) (h2 : V = 84) (h3 : S = 216) : S / K = 4 :=
by
  -- proof goes here
  sorry

end ratio_of_earnings_l222_222479


namespace x_power_12_l222_222089

theorem x_power_12 (x : ℝ) (h : x + 1 / x = 2) : x^12 = 1 :=
by sorry

end x_power_12_l222_222089


namespace ram_weight_increase_percentage_l222_222084

theorem ram_weight_increase_percentage :
  ∃ r s r_new: ℝ,
  r / s = 4 / 5 ∧ 
  r + s = 72 ∧ 
  s * 1.19 = 47.6 ∧
  r_new = 82.8 - 47.6 ∧ 
  (r_new - r) / r * 100 = 10 :=
by
  sorry

end ram_weight_increase_percentage_l222_222084


namespace max_area_of_triangle_MAN_l222_222251

noncomputable def maximum_area_triangle_MAN (e : ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
  if h : e = Real.sqrt 3 / 2 ∧ F = (Real.sqrt 3, 0) ∧ A = (1, 1 / 2) then
    Real.sqrt 2
  else
    0

theorem max_area_of_triangle_MAN :
  maximum_area_triangle_MAN (Real.sqrt 3 / 2) (Real.sqrt 3, 0) (1, 1 / 2) = Real.sqrt 2 :=
by
  sorry

end max_area_of_triangle_MAN_l222_222251


namespace percentage_of_liquid_X_in_solution_A_l222_222812

theorem percentage_of_liquid_X_in_solution_A (P : ℝ) :
  (0.018 * 700 / 1200 + P * 500 / 1200) = 0.0166 → P = 0.01464 :=
by 
  sorry

end percentage_of_liquid_X_in_solution_A_l222_222812


namespace friend_cutoff_fraction_l222_222057

-- Definitions based on problem conditions
def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def days_biking : ℕ := 1
def days_bus : ℕ := 3
def days_friend : ℕ := 1
def total_weekly_commuting_time : ℕ := 160

-- Lean theorem statement
theorem friend_cutoff_fraction (F : ℕ) (hF : days_biking * biking_time + days_bus * bus_time + days_friend * F = total_weekly_commuting_time) :
  (biking_time - F) / biking_time = 2 / 3 :=
by
  sorry

end friend_cutoff_fraction_l222_222057


namespace find_integer_pairs_l222_222592

def satisfies_conditions (m n : ℤ) : Prop :=
  m^2 = n^5 + n^4 + 1 ∧ ((m - 7 * n) ∣ (m - 4 * n))

theorem find_integer_pairs :
  ∀ (m n : ℤ), satisfies_conditions m n → (m, n) = (-1, 0) ∨ (m, n) = (1, 0) := by
  sorry

end find_integer_pairs_l222_222592


namespace john_father_age_difference_l222_222462

theorem john_father_age_difference (J F X : ℕ) (h1 : J + F = 77) (h2 : J = 15) (h3 : F = 2 * J + X) : X = 32 :=
by
  -- Adding the "sory" to skip the proof
  sorry

end john_father_age_difference_l222_222462


namespace total_distance_run_l222_222107

def track_meters : ℕ := 9
def laps_already_run : ℕ := 6
def laps_to_run : ℕ := 5

theorem total_distance_run :
  (laps_already_run * track_meters) + (laps_to_run * track_meters) = 99 := by
  sorry

end total_distance_run_l222_222107


namespace age_sum_proof_l222_222580

theorem age_sum_proof : 
  ∀ (Matt Fem Jake : ℕ), 
    Matt = 4 * Fem →
    Fem = 11 →
    Jake = Matt + 5 →
    (Matt + 2) + (Fem + 2) + (Jake + 2) = 110 :=
by
  intros Matt Fem Jake h1 h2 h3
  sorry

end age_sum_proof_l222_222580


namespace vasya_new_scoring_system_l222_222230

theorem vasya_new_scoring_system (a b c : ℕ) 
  (h1 : a + b + c = 52) 
  (h2 : a + b / 2 = 35) : a - c = 18 :=
by
  sorry

end vasya_new_scoring_system_l222_222230


namespace area_of_PINE_l222_222143

def PI := 6
def IN := 15
def NE := 6
def EP := 25
def sum_angles := 60 

theorem area_of_PINE : 
  (∃ (area : ℝ), area = (100 * Real.sqrt 3) / 3) := 
sorry

end area_of_PINE_l222_222143


namespace minimum_score_for_fourth_term_l222_222215

variable (score1 score2 score3 score4 : ℕ)
variable (avg_required : ℕ)

theorem minimum_score_for_fourth_term :
  score1 = 80 →
  score2 = 78 →
  score3 = 76 →
  avg_required = 85 →
  4 * avg_required - (score1 + score2 + score3) ≤ score4 :=
by
  sorry

end minimum_score_for_fourth_term_l222_222215


namespace volume_correctness_l222_222827

noncomputable def volume_of_regular_triangular_pyramid (d : ℝ) : ℝ :=
  1/3 * d^2 * d * Real.sqrt 2

theorem volume_correctness (d : ℝ) : 
  volume_of_regular_triangular_pyramid d = 1/3 * d^3 * Real.sqrt 2 :=
by
  sorry

end volume_correctness_l222_222827


namespace repaired_shoes_last_correct_l222_222178

noncomputable def repaired_shoes_last := 
  let repair_cost: ℝ := 10.50
  let new_shoes_cost: ℝ := 30.00
  let new_shoes_years: ℝ := 2.0
  let percentage_increase: ℝ := 42.857142857142854 / 100
  (T : ℝ) -> 15.00 = (repair_cost / T) * (1 + percentage_increase) → T = 1

theorem repaired_shoes_last_correct : repaired_shoes_last :=
by
  sorry

end repaired_shoes_last_correct_l222_222178


namespace lilith_caps_collection_l222_222380

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l222_222380


namespace fewest_number_of_gymnasts_l222_222272

theorem fewest_number_of_gymnasts (n : ℕ) (h : n % 2 = 0)
  (handshakes : ∀ (n : ℕ), (n * (n - 1) / 2) + n = 465) : 
  n = 30 :=
by
  sorry

end fewest_number_of_gymnasts_l222_222272


namespace mn_necessary_not_sufficient_l222_222503

variable (m n : ℝ)

def is_ellipse (m n : ℝ) : Prop := 
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem mn_necessary_not_sufficient : (mn > 0) → (is_ellipse m n) ↔ false := 
by sorry

end mn_necessary_not_sufficient_l222_222503


namespace ms_emily_inheritance_l222_222951

theorem ms_emily_inheritance :
  ∃ (y : ℝ), 
    (0.25 * y + 0.15 * (y - 0.25 * y) = 19500) ∧
    (y = 53800) :=
by
  sorry

end ms_emily_inheritance_l222_222951


namespace field_area_is_13_point854_hectares_l222_222071

noncomputable def area_of_field_in_hectares (cost_fencing: ℝ) (rate_per_meter: ℝ): ℝ :=
  let length_of_fence := cost_fencing / rate_per_meter
  let radius := length_of_fence / (2 * Real.pi)
  let area_in_square_meters := Real.pi * (radius * radius)
  area_in_square_meters / 10000

theorem field_area_is_13_point854_hectares :
  area_of_field_in_hectares 6202.75 4.70 = 13.854 :=
by
  sorry

end field_area_is_13_point854_hectares_l222_222071


namespace arithmetic_sequence_75th_term_l222_222600

theorem arithmetic_sequence_75th_term (a1 d : ℤ) (n : ℤ) (h1 : a1 = 3) (h2 : d = 5) (h3 : n = 75) :
  a1 + (n - 1) * d = 373 :=
by
  rw [h1, h2, h3]
  -- Here, we arrive at the explicitly stated elements and evaluate:
  -- 3 + (75 - 1) * 5 = 373
  sorry

end arithmetic_sequence_75th_term_l222_222600


namespace greatest_integer_leq_fraction_l222_222209

theorem greatest_integer_leq_fraction (N D : ℝ) (hN : N = 4^103 + 3^103 + 2^103) (hD : D = 4^100 + 3^100 + 2^100) :
  ⌊N / D⌋ = 64 :=
by
  sorry

end greatest_integer_leq_fraction_l222_222209


namespace minimum_z_value_l222_222097

theorem minimum_z_value (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : x^2 + y^2 ≥ 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_z_value_l222_222097


namespace min_f_value_f_achieves_min_l222_222415

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x ^ 2 + 1) + (x * (x + 3)) / (x ^ 2 + 2) + (3 * (x + 1)) / (x * (x ^ 2 + 2))

theorem min_f_value (x : ℝ) (hx : x > 0) : f x ≥ 3 :=
sorry

theorem f_achieves_min (x : ℝ) (hx : x > 0) : ∃ x, f x = 3 :=
sorry

end min_f_value_f_achieves_min_l222_222415


namespace sum_of_transformed_numbers_l222_222588

theorem sum_of_transformed_numbers (a b S : ℕ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l222_222588


namespace annie_jacob_ratio_l222_222577

theorem annie_jacob_ratio :
  ∃ (a j : ℕ), ∃ (m : ℕ), (m = 2 * a) ∧ (j = 90) ∧ (m = 60) ∧ (a / j = 1 / 3) :=
by
  sorry

end annie_jacob_ratio_l222_222577


namespace max_ab_l222_222298

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 6) : ab ≤ 9 / 2 :=
by
  sorry

end max_ab_l222_222298


namespace sum_lent_250_l222_222313

theorem sum_lent_250 (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (hR : R = 4) (hT : T = 8) (hSI1 : SI = P - 170) 
  (hSI2 : SI = (P * R * T) / 100) : 
  P = 250 := 
by 
  sorry

end sum_lent_250_l222_222313


namespace simplify_expression_l222_222626

variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 9) - (x + 6) * (3 * x - 2) = 7 * x - 24 :=
by
  sorry

end simplify_expression_l222_222626


namespace andrew_apples_l222_222318

theorem andrew_apples : ∃ (A n : ℕ), (6 * n = A) ∧ (5 * (n + 2) = A) ∧ (A = 60) :=
by 
  sorry

end andrew_apples_l222_222318


namespace number_of_three_digit_multiples_of_9_with_odd_digits_l222_222643

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def consists_only_of_odd_digits (n : ℕ) : Prop :=
  (∀ d ∈ (n.digits 10), d % 2 = 1)

theorem number_of_three_digit_multiples_of_9_with_odd_digits :
  ∃ t, t = 11 ∧
  (∀ n, is_three_digit_number n ∧ is_multiple_of_9 n ∧ consists_only_of_odd_digits n) → 1 ≤ t ∧ t ≤ 11 :=
sorry

end number_of_three_digit_multiples_of_9_with_odd_digits_l222_222643


namespace simplify_expression_l222_222216

theorem simplify_expression :
  (1024 ^ (1/5) * 125 ^ (1/3)) = 20 :=
by
  have h1 : 1024 = 2 ^ 10 := by norm_num
  have h2 : 125 = 5 ^ 3 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end simplify_expression_l222_222216


namespace line_through_point_parallel_to_given_l222_222429

open Real

theorem line_through_point_parallel_to_given (x y : ℝ) :
  (∃ (m : ℝ), (y - 0 = m * (x - 1)) ∧ x - 2*y - 1 = 0) ↔
  (x = 1 ∧ y = 0 ∧ ∃ l, x - 2*y - l = 0) :=
by sorry

end line_through_point_parallel_to_given_l222_222429


namespace bridge_length_l222_222181

noncomputable def speed_km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def distance_travelled (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_condition : train_length = 150) 
  (train_speed_condition : train_speed_kmph = 45) 
  (crossing_time_condition : crossing_time_s = 30) :
  (distance_travelled (speed_km_per_hr_to_m_per_s train_speed_kmph) crossing_time_s - train_length) = 225 :=
by 
  sorry

end bridge_length_l222_222181


namespace f_is_periodic_l222_222140

noncomputable def f (x : ℝ) : ℝ := x - ⌈x⌉

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x :=
by 
  intro x
  sorry

end f_is_periodic_l222_222140


namespace integer_roots_of_quadratic_eq_are_neg3_and_neg7_l222_222449

theorem integer_roots_of_quadratic_eq_are_neg3_and_neg7 :
  {k : ℤ | ∃ x : ℤ, k * x^2 - 2 * (3 * k - 1) * x + 9 * k - 1 = 0} = {-3, -7} :=
by
  sorry

end integer_roots_of_quadratic_eq_are_neg3_and_neg7_l222_222449


namespace a_plus_b_eq_l222_222603

-- Define the sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -3 < x ∧ x < 2 }

-- Define the intersection set A ∩ B
def A_inter_B := { x : ℝ | -1 < x ∧ x < 2 }

-- Define a condition
noncomputable def is_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (-1 < x ∧ x < 2) ↔ (x^2 + a * x + b < 0)

-- The proof statement
theorem a_plus_b_eq : ∃ a b : ℝ, is_solution_set a b ∧ a + b = -3 := by
  sorry

end a_plus_b_eq_l222_222603


namespace solve_quadratic_eq_l222_222024

theorem solve_quadratic_eq (x : ℝ) :
  (x^2 + (x - 1) * (x + 3) = 3 * x + 5) ↔ (x = -2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_eq_l222_222024


namespace imaginary_part_is_neg_two_l222_222702

open Complex

noncomputable def imaginary_part_of_square : ℂ := (1 - I)^2

theorem imaginary_part_is_neg_two : imaginary_part_of_square.im = -2 := by
  sorry

end imaginary_part_is_neg_two_l222_222702


namespace discount_percentage_l222_222004

theorem discount_percentage (p : ℝ) : 
  (1 + 0.25) * p * (1 - 0.20) = p :=
by
  sorry

end discount_percentage_l222_222004


namespace halloween_candy_l222_222737

theorem halloween_candy (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) (total_candy : ℕ) (eaten_candy : ℕ)
  (h1 : katie_candy = 10) 
  (h2 : sister_candy = 6) 
  (h3 : remaining_candy = 7) 
  (h4 : total_candy = katie_candy + sister_candy) 
  (h5 : eaten_candy = total_candy - remaining_candy) : 
  eaten_candy = 9 :=
by sorry

end halloween_candy_l222_222737


namespace arithmetic_seq_20th_term_l222_222715

variable (a : ℕ → ℤ) -- a_n is an arithmetic sequence
variable (d : ℤ) -- common difference of the arithmetic sequence

-- Condition for arithmetic sequence
variable (h_seq : ∀ n, a (n+1) = a n + d)

-- Given conditions
axiom h1 : a 1 + a 3 + a 5 = 105
axiom h2 : a 2 + a 4 + a 6 = 99

-- Goal: prove that a 20 = 1
theorem arithmetic_seq_20th_term :
  a 20 = 1 :=
sorry

end arithmetic_seq_20th_term_l222_222715


namespace largest_rectangle_area_l222_222821

theorem largest_rectangle_area (l w : ℕ) (hl : l > 0) (hw : w > 0) (hperimeter : 2 * l + 2 * w = 42)
  (harea_diff : ∃ (l1 w1 l2 w2 : ℕ), l1 > 0 ∧ w1 > 0 ∧ l2 > 0 ∧ w2 > 0 ∧ 2 * l1 + 2 * w1 = 42 
  ∧ 2 * l2 + 2 * w2 = 42 ∧ (l1 * w1) - (l2 * w2) = 90) : (l * w ≤ 110) :=
sorry

end largest_rectangle_area_l222_222821


namespace tan_alpha_eq_4_over_3_expression_value_eq_4_l222_222473

-- Conditions
variable (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 2)) (h_sin : Real.sin α = 4 / 5)

-- Prove: tan α = 4 / 3
theorem tan_alpha_eq_4_over_3 : Real.tan α = 4 / 3 :=
by
  sorry

-- Prove: the value of the given expression is 4
theorem expression_value_eq_4 : 
  (Real.sin (α + Real.pi) - 2 * Real.cos ((Real.pi / 2) + α)) / 
  (- Real.sin (-α) + Real.cos (Real.pi + α)) = 4 :=
by
  sorry

end tan_alpha_eq_4_over_3_expression_value_eq_4_l222_222473


namespace sqrt_factorial_mul_factorial_eq_l222_222235

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l222_222235


namespace num_pairs_in_arithmetic_progression_l222_222611

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end num_pairs_in_arithmetic_progression_l222_222611


namespace AM_GM_contradiction_l222_222083

open Real

theorem AM_GM_contradiction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
      ¬ (6 < a + 4 / b ∧ 6 < b + 9 / c ∧ 6 < c + 16 / a) := by
  sorry

end AM_GM_contradiction_l222_222083


namespace smallest_factor_of_36_l222_222322

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l222_222322


namespace principal_amount_l222_222352

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 5)

theorem principal_amount :
  ((P * R * T) / 100 = P - 2000) → P = 2500 :=
by
  sorry

end principal_amount_l222_222352


namespace jerry_water_usage_l222_222213

noncomputable def total_water_usage 
  (drinking_cooking : ℕ) 
  (shower_per_gallon : ℕ) 
  (length width height : ℕ) 
  (gallon_per_cubic_ft : ℕ) 
  (number_of_showers : ℕ) 
  : ℕ := 
   drinking_cooking + 
   (number_of_showers * shower_per_gallon) + 
   (length * width * height / gallon_per_cubic_ft)

theorem jerry_water_usage 
  (drinking_cooking : ℕ := 100)
  (shower_per_gallon : ℕ := 20)
  (length : ℕ := 10)
  (width : ℕ := 10)
  (height : ℕ := 6)
  (gallon_per_cubic_ft : ℕ := 1)
  (number_of_showers : ℕ := 15)
  : total_water_usage drinking_cooking shower_per_gallon length width height gallon_per_cubic_ft number_of_showers = 1400 := 
by
  sorry

end jerry_water_usage_l222_222213


namespace average_viewer_watches_two_videos_daily_l222_222332

variable (V : ℕ)
variable (video_time : ℕ := 7)
variable (ad_time : ℕ := 3)
variable (total_time : ℕ := 17)

theorem average_viewer_watches_two_videos_daily :
  7 * V + 3 = 17 → V = 2 := 
by
  intro h
  have h1 : 7 * V = 14 := by linarith
  have h2 : V = 2 := by linarith
  exact h2

end average_viewer_watches_two_videos_daily_l222_222332


namespace original_price_l222_222591

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end original_price_l222_222591


namespace minimum_games_pasha_wins_l222_222467

noncomputable def pasha_initial_money : Nat := 9 -- Pasha has a single-digit amount
noncomputable def igor_initial_money : Nat := 1000 -- Igor has a four-digit amount
noncomputable def pasha_final_money : Nat := 100 -- Pasha has a three-digit amount
noncomputable def igor_final_money : Nat := 99 -- Igor has a two-digit amount

theorem minimum_games_pasha_wins :
  ∃ (games_won_by_pasha : Nat), 
    (games_won_by_pasha >= 7) ∧
    (games_won_by_pasha <= 7) := sorry

end minimum_games_pasha_wins_l222_222467


namespace total_employees_in_company_l222_222838

-- Given facts and conditions
def ratio_A_B_C : Nat × Nat × Nat := (5, 4, 1)
def sample_size : Nat := 20
def prob_sel_A_B_from_C : ℚ := 1 / 45

-- Number of group C individuals, calculated from probability constraint
def num_persons_group_C := 10

theorem total_employees_in_company (x : Nat) :
  x = 10 * (5 + 4 + 1) :=
by
  -- Since the sample size is 20, and the ratio of sampling must be consistent with the population ratio,
  -- it can be derived that the total number of employees in the company must be 100.
  -- Adding sorry to skip the actual detailed proof.
  sorry

end total_employees_in_company_l222_222838


namespace find_line_equation_through_point_intersecting_hyperbola_l222_222625

theorem find_line_equation_through_point_intersecting_hyperbola 
  (x y : ℝ) 
  (hx : x = -2 / 3)
  (hy : (x : ℝ) = 0) : 
  ∃ k : ℝ, (∀ x y : ℝ, y = k * x - 1 → ((x^2 / 2) - (y^2 / 5) = 1)) ∧ k = 1 := 
sorry

end find_line_equation_through_point_intersecting_hyperbola_l222_222625


namespace average_test_score_l222_222773

theorem average_test_score (x : ℝ) :
  (0.45 * 95 + 0.50 * x + 0.05 * 60 = 84.75) → x = 78 :=
by
  sorry

end average_test_score_l222_222773


namespace find_a12_a14_l222_222060

noncomputable def S (n : ℕ) (a_n : ℕ → ℝ) (b : ℝ) : ℝ := a_n n ^ 2 + b * n

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ (a1 : ℝ) (c : ℝ), ∀ n : ℕ, a_n n = a1 + (n - 1) * c

theorem find_a12_a14
  (a_n : ℕ → ℝ)
  (b : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a_n n ^ 2 + b * n)
  (h2 : S 25 = 100)
  (h3 : is_arithmetic_sequence a_n) :
  a_n 12 + a_n 14 = 5 :=
sorry

end find_a12_a14_l222_222060


namespace xy_sum_proof_l222_222394

-- Define the given list of numbers
def original_list := [201, 202, 204, 205, 206, 209, 209, 210, 212]

-- Define the target new average and sum of numbers
def target_average : ℕ := 207
def sum_xy : ℕ := 417

-- Calculate the original sum
def original_sum : ℕ := 201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212

-- The new total sum calculation with x and y included
def new_total_sum := original_sum + sum_xy

-- Number of elements in the new list
def new_num_elements : ℕ := 11

-- Target new sum based on the new average and number of elements
def target_new_sum := target_average * new_num_elements

theorem xy_sum_proof : new_total_sum = target_new_sum := by
  sorry

end xy_sum_proof_l222_222394


namespace value_of_expression_l222_222353

theorem value_of_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 :=
by sorry

end value_of_expression_l222_222353


namespace intersection_complement_l222_222284

universe u

def U := Real

def M : Set Real := { x | -2 ≤ x ∧ x ≤ 2 }

def N : Set Real := { x | x * (x - 3) ≤ 0 }

def complement_U (S : Set Real) : Set Real := { x | x ∉ S }

theorem intersection_complement :
  M ∩ (complement_U N) = { x | -2 ≤ x ∧ x < 0 } := by
  sorry

end intersection_complement_l222_222284


namespace half_way_fraction_l222_222672

def half_way_between (a b : ℚ) : ℚ := (a + b) / 2

theorem half_way_fraction : 
  half_way_between (1/3) (3/4) = 13/24 :=
by 
  -- Proof follows from the calculation steps, but we leave it unproved.
  sorry

end half_way_fraction_l222_222672


namespace derivative_of_log_base_3_derivative_of_exp_base_2_l222_222292

noncomputable def log_base_3_deriv (x : ℝ) : ℝ := (Real.log x / Real.log 3)
noncomputable def exp_base_2_deriv (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem derivative_of_log_base_3 (x : ℝ) (h : x > 0) :
  (log_base_3_deriv x) = (1 / (x * Real.log 3)) :=
by
  sorry

theorem derivative_of_exp_base_2 (x : ℝ) :
  (exp_base_2_deriv x) = (Real.exp (x * Real.log 2) * Real.log 2) :=
by
  sorry

end derivative_of_log_base_3_derivative_of_exp_base_2_l222_222292


namespace f_is_monotonic_l222_222517

variable (f : ℝ → ℝ)

theorem f_is_monotonic (h : ∀ a b x : ℝ, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  (∀ x y : ℝ, x ≤ y → f x <= f y) ∨ (∀ x y : ℝ, x ≤ y → f x >= f y) :=
sorry

end f_is_monotonic_l222_222517


namespace least_possible_integer_l222_222977

theorem least_possible_integer (N : ℕ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ N) ∧
  (∀ m : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ m) → N ≤ m) →
  N = 2329089562800 :=
sorry

end least_possible_integer_l222_222977


namespace choosing_ways_president_vp_committee_l222_222549

theorem choosing_ways_president_vp_committee :
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  total_choices = 2520 := by
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  have : total_choices = 2520 := by
    sorry
  exact this

end choosing_ways_president_vp_committee_l222_222549


namespace min_value_of_expression_l222_222975

theorem min_value_of_expression {x y z : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) : 
  (x + 1 / y) * (x + 1 / z) >= Real.sqrt 2 :=
by
  sorry

end min_value_of_expression_l222_222975


namespace contest_paths_correct_l222_222663

noncomputable def count_contest_paths : Nat := sorry

theorem contest_paths_correct : count_contest_paths = 127 := sorry

end contest_paths_correct_l222_222663


namespace additional_days_use_l222_222793

variable (m a : ℝ)

theorem additional_days_use (hm : m > 0) (ha : a > 1) : 
  (m / (a - 1) - m / a) = m / (a * (a - 1)) :=
sorry

end additional_days_use_l222_222793


namespace gross_profit_percentage_without_discount_l222_222044

theorem gross_profit_percentage_without_discount (C P : ℝ)
  (discount : P * 0.9 = C * 1.2)
  (discount_profit : C * 0.2 = P * 0.9 - C) :
  (P - C) / C * 100 = 33.3 :=
by
  sorry

end gross_profit_percentage_without_discount_l222_222044


namespace solve_quadratic_solve_inequality_system_l222_222551

theorem solve_quadratic :
  ∀ x : ℝ, x^2 - 6 * x + 5 = 0 ↔ x = 1 ∨ x = 5 :=
sorry

theorem solve_inequality_system :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2 * (x + 1) < 4) ↔ (-3 < x ∧ x < 1) :=
sorry

end solve_quadratic_solve_inequality_system_l222_222551


namespace balloon_arrangement_count_l222_222967

theorem balloon_arrangement_count :
  let total_permutations := (Nat.factorial 7) / (Nat.factorial 2 * Nat.factorial 3)
  let ways_to_arrange_L_and_O := Nat.choose 4 1 * (Nat.factorial 3)
  let valid_arrangements := ways_to_arrange_L_and_O * total_permutations
  valid_arrangements = 10080 :=
by
  sorry

end balloon_arrangement_count_l222_222967


namespace simplify_expression_l222_222252

theorem simplify_expression (b : ℝ) :
  (1 * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5)) = 720 * b^15 :=
by
  sorry

end simplify_expression_l222_222252


namespace quadratic_vertex_coordinates_l222_222538

theorem quadratic_vertex_coordinates (x y : ℝ) (h : y = 2 * x^2 - 4 * x + 5) : (x, y) = (1, 3) :=
sorry

end quadratic_vertex_coordinates_l222_222538


namespace binom_n_plus_1_n_minus_1_eq_l222_222265

theorem binom_n_plus_1_n_minus_1_eq (n : ℕ) (h : 0 < n) : (Nat.choose (n + 1) (n - 1)) = n * (n + 1) / 2 := 
by sorry

end binom_n_plus_1_n_minus_1_eq_l222_222265


namespace halfway_between_one_third_and_one_fifth_l222_222647

theorem halfway_between_one_third_and_one_fifth : (1/3 + 1/5) / 2 = 4/15 := 
by 
  sorry

end halfway_between_one_third_and_one_fifth_l222_222647


namespace sqrt_expression_eq_seven_div_two_l222_222233

theorem sqrt_expression_eq_seven_div_two :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 / Real.sqrt 24) = 7 / 2 :=
by
  sorry

end sqrt_expression_eq_seven_div_two_l222_222233


namespace determine_set_A_l222_222547

-- Define the function f as described
def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n - 1)

-- Define the set A
def A (n : ℕ) : Set ℕ :=
  { x | (Nat.iterate (f n) n x) = x }

-- State the theorem
theorem determine_set_A (n : ℕ) (hn : n > 0) :
    A n = { x | 1 ≤ x ∧ x ≤ 2^n } :=
sorry

end determine_set_A_l222_222547


namespace walt_age_l222_222650

-- Conditions
variables (T W : ℕ)
axiom h1 : T = 3 * W
axiom h2 : T + 12 = 2 * (W + 12)

-- Goal: Prove W = 12
theorem walt_age : W = 12 :=
sorry

end walt_age_l222_222650


namespace initial_men_count_l222_222255

theorem initial_men_count (M : ℕ) (F : ℕ) (h1 : F = M * 22) (h2 : (M + 2280) * 5 = M * 20) : M = 760 := by
  sorry

end initial_men_count_l222_222255


namespace matrix_vector_addition_l222_222070

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -2], ![-5, 6]]
def v : Fin 2 → ℤ := ![5, -2]
def w : Fin 2 → ℤ := ![1, -1]

theorem matrix_vector_addition :
  (A.mulVec v + w) = ![25, -38] :=
by
  sorry

end matrix_vector_addition_l222_222070


namespace gcd_204_85_l222_222890

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l222_222890


namespace car_trip_cost_proof_l222_222257

def car_trip_cost 
  (d1 d2 d3 d4 : ℕ) 
  (efficiency : ℕ) 
  (cost_per_gallon : ℕ) 
  (total_distance : ℕ) 
  (gallons_used : ℕ) 
  (cost : ℕ) : Prop :=
  d1 = 8 ∧
  d2 = 6 ∧
  d3 = 12 ∧
  d4 = 2 * d3 ∧
  efficiency = 25 ∧
  cost_per_gallon = 250 ∧
  total_distance = d1 + d2 + d3 + d4 ∧
  gallons_used = total_distance / efficiency ∧
  cost = gallons_used * cost_per_gallon ∧
  cost = 500

theorem car_trip_cost_proof : car_trip_cost 8 6 12 (2 * 12) 25 250 (8 + 6 + 12 + (2 * 12)) ((8 + 6 + 12 + (2 * 12)) / 25) (((8 + 6 + 12 + (2 * 12)) / 25) * 250) :=
by 
  sorry

end car_trip_cost_proof_l222_222257


namespace icing_cubes_count_31_l222_222163

def cake_cubed (n : ℕ) := n^3

noncomputable def slabs_with_icing (n : ℕ): ℕ := 
    let num_faces := 3
    let edge_per_face := n - 1
    let edges_with_icing := num_faces * edge_per_face * (n - 2)
    edges_with_icing + (n - 2) * 4 * (n - 2)

theorem icing_cubes_count_31 : ∀ (n : ℕ), n = 5 → slabs_with_icing n = 31 :=
by
  intros n hn
  revert hn
  sorry

end icing_cubes_count_31_l222_222163


namespace xiaohong_home_to_school_distance_l222_222404

noncomputable def driving_distance : ℝ := 1000
noncomputable def total_travel_time : ℝ := 22.5
noncomputable def walking_speed : ℝ := 80
noncomputable def biking_time : ℝ := 40
noncomputable def biking_speed_offset : ℝ := 800

theorem xiaohong_home_to_school_distance (d : ℝ) (v_d : ℝ) :
    let t_w := (d - driving_distance) / walking_speed
    let t_d := driving_distance / v_d
    let v_b := v_d - biking_speed_offset
    (t_d + t_w = total_travel_time)
    → (d / v_b = biking_time)
    → d = 2720 :=
by
  sorry

end xiaohong_home_to_school_distance_l222_222404


namespace solve_parabola_l222_222907

theorem solve_parabola (a b c : ℝ) 
  (h1 : 1 = a * 1^2 + b * 1 + c)
  (h2 : 4 * a + b = 1)
  (h3 : -1 = a * 2^2 + b * 2 + c) :
  a = 3 ∧ b = -11 ∧ c = 9 :=
by {
  sorry
}

end solve_parabola_l222_222907


namespace problem1_problem2_l222_222051

variable {x : ℝ} (hx : x > 0)

theorem problem1 : (2 / (3 * x)) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x := 
by sorry

theorem problem2 : (Real.sqrt 24 + Real.sqrt 6) / Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 3 * Real.sqrt 2 + 2 := 
by sorry

end problem1_problem2_l222_222051


namespace deer_meat_distribution_l222_222557

theorem deer_meat_distribution (a d : ℕ) (H1 : a = 100) :
  ∀ (Dafu Bugeng Zanbao Shangzao Gongshe : ℕ),
    Dafu = a - 2 * d →
    Bugeng = a - d →
    Zanbao = a →
    Shangzao = a + d →
    Gongshe = a + 2 * d →
    Dafu + Bugeng + Zanbao + Shangzao + Gongshe = 500 →
    Bugeng + Zanbao + Shangzao = 300 :=
by
  intros Dafu Bugeng Zanbao Shangzao Gongshe hDafu hBugeng hZanbao hShangzao hGongshe hSum
  sorry

end deer_meat_distribution_l222_222557


namespace raisin_fraction_of_mixture_l222_222130

noncomputable def raisin_nut_cost_fraction (R : ℝ) : ℝ :=
  let raisin_cost := 3 * R
  let nut_cost := 4 * (4 * R)
  let total_cost := raisin_cost + nut_cost
  raisin_cost / total_cost

theorem raisin_fraction_of_mixture (R : ℝ) : raisin_nut_cost_fraction R = 3 / 19 :=
by
  sorry

end raisin_fraction_of_mixture_l222_222130


namespace pin_probability_l222_222804

theorem pin_probability :
  let total_pins := 9 * 10^5
  let valid_pins := 10^4
  ∃ p : ℚ, p = valid_pins / total_pins ∧ p = 1 / 90 := by
  sorry

end pin_probability_l222_222804


namespace sum_of_digits_B_l222_222132

/- 
  Let A be the natural number formed by concatenating integers from 1 to 100.
  Let B be the smallest possible natural number formed by removing 100 digits from A.
  We need to prove that the sum of the digits of B equals 486.
-/
def A : ℕ := sorry -- construct the natural number 1234567891011121314...99100

def sum_of_digits (n : ℕ) : ℕ := sorry -- function to calculate the sum of digits of a natural number

def B : ℕ := sorry -- construct the smallest possible number B by removing 100 digits from A

theorem sum_of_digits_B : sum_of_digits B = 486 := sorry

end sum_of_digits_B_l222_222132


namespace car_gas_consumption_l222_222685

theorem car_gas_consumption
  (miles_today : ℕ)
  (miles_tomorrow : ℕ)
  (total_gallons : ℕ)
  (h1 : miles_today = 400)
  (h2 : miles_tomorrow = miles_today + 200)
  (h3 : total_gallons = 4000)
  : (∃ g : ℕ, 400 * g + (400 + 200) * g = total_gallons ∧ g = 4) :=
by
  sorry

end car_gas_consumption_l222_222685


namespace multiplication_correct_l222_222498

theorem multiplication_correct (a b c d e f: ℤ) (h₁: a * b = c) (h₂: d * e = f): 
    (63 * 14 = c) → (68 * 14 = f) → c = 882 ∧ f = 952 :=
by sorry

end multiplication_correct_l222_222498


namespace dispatch_3_male_2_female_dispatch_at_least_2_male_l222_222043

-- Define the number of male and female drivers
def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def total_drivers_needed : ℕ := 5

-- Define the combination formula (binomial coefficient)
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- First part of the problem
theorem dispatch_3_male_2_female : 
  combination male_drivers 3 * combination female_drivers 2 = 60 :=
by sorry

-- Second part of the problem
theorem dispatch_at_least_2_male : 
  combination male_drivers 2 * combination female_drivers 3 + 
  combination male_drivers 3 * combination female_drivers 2 + 
  combination male_drivers 4 * combination female_drivers 1 + 
  combination male_drivers 5 * combination female_drivers 0 = 121 :=
by sorry

end dispatch_3_male_2_female_dispatch_at_least_2_male_l222_222043


namespace part_I_part_II_l222_222266

def f (x a : ℝ) := |x - a| + |x - 1|

theorem part_I {x : ℝ} : Set.Icc 0 4 = {y | f y 3 ≤ 4} := 
sorry

theorem part_II {a : ℝ} : (∀ x, ¬ (f x a < 2)) ↔ a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end part_I_part_II_l222_222266


namespace gardener_works_days_l222_222240

theorem gardener_works_days :
  let rose_bushes := 20
  let cost_per_rose_bush := 150
  let gardener_hourly_wage := 30
  let gardener_hours_per_day := 5
  let soil_volume := 100
  let cost_per_soil := 5
  let total_project_cost := 4100
  let total_gardening_days := 4
  (rose_bushes * cost_per_rose_bush + soil_volume * cost_per_soil + total_gardening_days * gardener_hours_per_day * gardener_hourly_wage = total_project_cost) →
  total_gardening_days = 4 :=
by
  intros
  sorry

end gardener_works_days_l222_222240


namespace h_h_3_eq_3568_l222_222056

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h_3_eq_3568_l222_222056


namespace co_complementary_angles_equal_l222_222237

def co_complementary (A : ℝ) : ℝ := 90 - A

theorem co_complementary_angles_equal (A B : ℝ) (h : co_complementary A = co_complementary B) : A = B :=
sorry

end co_complementary_angles_equal_l222_222237


namespace relationship_between_3a_3b_4a_l222_222996

variable (a b : ℝ)
variable (h : a > b)
variable (hb : b > 0)

theorem relationship_between_3a_3b_4a (a b : ℝ) (h : a > b) (hb : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := 
by
  sorry

end relationship_between_3a_3b_4a_l222_222996


namespace solve_equation_l222_222184

variable (x : ℝ)

theorem solve_equation (h : x * (x - 4) = x - 6) : x = 2 ∨ x = 3 := 
sorry

end solve_equation_l222_222184


namespace instantaneous_velocity_at_t2_l222_222416

def displacement (t : ℝ) : ℝ := 14 * t - t ^ 2

theorem instantaneous_velocity_at_t2 : (deriv displacement 2) = 10 := by
  sorry

end instantaneous_velocity_at_t2_l222_222416


namespace stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l222_222041

theorem stratified_sampling_number_of_boys (total_students : Nat) (num_girls : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : num_girls = 50) (h3 : selected_students = 25) :
  (total_students - num_girls) * selected_students / total_students = 15 :=
  sorry

theorem stratified_sampling_probability_of_boy (total_students : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : selected_students = 25) :
  selected_students / total_students = 1 / 5 :=
  sorry

end stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l222_222041


namespace minimal_polynomial_correct_l222_222623

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (Polynomial.X^2 - 4 * Polynomial.X + 1) * (Polynomial.X^2 - 6 * Polynomial.X + 2)

theorem minimal_polynomial_correct :
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 26 * Polynomial.X + 2 = minimal_polynomial :=
  sorry

end minimal_polynomial_correct_l222_222623


namespace class_president_is_yi_l222_222211

variable (Students : Type)
variable (Jia Yi Bing StudyCommittee SportsCommittee ClassPresident : Students)
variable (age : Students → ℕ)

-- Conditions
axiom bing_older_than_study_committee : age Bing > age StudyCommittee
axiom jia_age_different_from_sports_committee : age Jia ≠ age SportsCommittee
axiom sports_committee_younger_than_yi : age SportsCommittee < age Yi

-- Prove that Yi is the class president
theorem class_president_is_yi : ClassPresident = Yi :=
sorry

end class_president_is_yi_l222_222211


namespace Nina_money_before_tax_l222_222811

theorem Nina_money_before_tax :
  ∃ (M P : ℝ), M = 6 * P ∧ M = 8 * 0.9 * P ∧ M = 5 :=
by 
  sorry

end Nina_money_before_tax_l222_222811


namespace determine_g_x2_l222_222700

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem determine_g_x2 (x : ℝ) (h : x^2 ≠ 4) : g (x^2) = (2 * x^2 + 3) / (x^2 - 2) :=
by sorry

end determine_g_x2_l222_222700


namespace find_a_find_m_l222_222169

-- Definition of the odd function condition
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- The first proof problem
theorem find_a (a : ℝ) (h_odd : odd_function (fun x => Real.log (Real.exp x + a + 1))) : a = -1 :=
sorry

-- Definitions of the two functions involved in the second proof problem
noncomputable def f1 (x : ℝ) : ℝ :=
if x = 0 then 0 else Real.log x / x

noncomputable def f2 (x m : ℝ) : ℝ :=
x^2 - 2 * Real.exp 1 * x + m

-- The second proof problem
theorem find_m (m : ℝ) (h_root : ∃! x, f1 x = f2 x m) : m = Real.exp 2 + 1 / Real.exp 1 :=
sorry

end find_a_find_m_l222_222169


namespace perpendicular_line_sum_l222_222446

theorem perpendicular_line_sum (a b c : ℝ) 
  (h1 : -a / 4 * 2 / 5 = -1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * c + b = 0) : 
  a + b + c = -4 :=
sorry

end perpendicular_line_sum_l222_222446


namespace unique_solution_l222_222955

theorem unique_solution (x : ℝ) : (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x ↔ x = 3 := by
  sorry

end unique_solution_l222_222955


namespace find_a_if_line_passes_through_center_l222_222964

-- Define the given circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the given line equation
def line_eqn (x y a : ℝ) : Prop := 3*x + y + a = 0

-- The coordinates of the center of the circle
def center_of_circle : (ℝ × ℝ) := (-1, 2)

-- Prove that a = 1 if the line passes through the center of the circle
theorem find_a_if_line_passes_through_center (a : ℝ) :
  line_eqn (-1) 2 a → a = 1 :=
by
  sorry

end find_a_if_line_passes_through_center_l222_222964


namespace toby_initial_photos_l222_222468

-- Defining the problem conditions and proving the initial number of photos Toby had.
theorem toby_initial_photos (X : ℕ) 
  (h1 : ∃ n, X = n - 7) 
  (h2 : ∃ m, m = (n - 7) + 15) 
  (h3 : ∃ k, k = m) 
  (h4 : (k - 3) = 84) 
  : X = 79 :=
sorry

end toby_initial_photos_l222_222468


namespace transform_polynomial_l222_222293

theorem transform_polynomial (x y : ℝ) 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 - x^3 - 2 * x^2 - x + 1 = 0) : x^2 * (y^2 - y - 4) = 0 :=
sorry

end transform_polynomial_l222_222293


namespace cannot_pay_exactly_500_can_pay_exactly_600_l222_222846

-- Defining the costs and relevant equations
def price_of_bun : ℕ := 15
def price_of_croissant : ℕ := 12

-- Proving the non-existence for the 500 Ft case
theorem cannot_pay_exactly_500 : ¬ ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 500 :=
sorry

-- Proving the existence for the 600 Ft case
theorem can_pay_exactly_600 : ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 600 :=
sorry

end cannot_pay_exactly_500_can_pay_exactly_600_l222_222846


namespace rectangle_area_l222_222589

theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 14) (h2 : l^2 + w^2 = 25) : l * w = 12 :=
by
  sorry

end rectangle_area_l222_222589


namespace point_on_x_axis_l222_222569

theorem point_on_x_axis (m : ℝ) (P : ℝ × ℝ) (hP : P = (m + 3, m - 1)) (hx : P.2 = 0) :
  P = (4, 0) :=
by
  sorry

end point_on_x_axis_l222_222569


namespace steps_to_Madison_eq_991_l222_222013

variable (steps_down steps_to_Madison : ℕ)

def total_steps (steps_down steps_to_Madison : ℕ) : ℕ :=
  steps_down + steps_to_Madison

theorem steps_to_Madison_eq_991 (h1 : steps_down = 676) (h2 : steps_to_Madison = 315) :
  total_steps steps_down steps_to_Madison = 991 :=
by
  sorry

end steps_to_Madison_eq_991_l222_222013


namespace monotonically_increasing_interval_l222_222167

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonically_increasing_interval : 
  ∃ (a b : ℝ), a = -Real.pi / 3 ∧ b = Real.pi / 6 ∧ ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y :=
by
  sorry

end monotonically_increasing_interval_l222_222167


namespace greatest_c_value_l222_222223

theorem greatest_c_value (c : ℤ) : 
  (∀ (x : ℝ), x^2 + (c : ℝ) * x + 20 ≠ -7) → c = 10 :=
by
  sorry

end greatest_c_value_l222_222223


namespace common_ratio_geometric_progression_l222_222552

theorem common_ratio_geometric_progression (r : ℝ) (a : ℝ) (h : a > 0) (h_r : r > 0) (h_eq : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) : r^3 + r^2 + r - 1 = 0 := 
by sorry

end common_ratio_geometric_progression_l222_222552


namespace max_x_for_integer_fraction_l222_222341

theorem max_x_for_integer_fraction (x : ℤ) (h : ∃ k : ℤ, x^2 + 2 * x + 11 = k * (x - 3)) : x ≤ 29 :=
by {
    -- This is where the proof would be,
    -- but we skip the proof per the instructions.
    sorry
}

end max_x_for_integer_fraction_l222_222341


namespace max_c_value_l222_222636

variable {a b c : ℝ}

theorem max_c_value (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c ≤ 8 / 15 :=
sorry

end max_c_value_l222_222636


namespace gcd_euclidean_120_168_gcd_subtraction_459_357_l222_222315

theorem gcd_euclidean_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

theorem gcd_subtraction_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_euclidean_120_168_gcd_subtraction_459_357_l222_222315


namespace infinite_powers_of_two_in_sequence_l222_222074

theorem infinite_powers_of_two_in_sequence :
  ∃ᶠ n in at_top, ∃ k : ℕ, ∃ a : ℕ, (a = ⌊n * Real.sqrt 2⌋ ∧ a = 2^k) :=
sorry

end infinite_powers_of_two_in_sequence_l222_222074


namespace total_earning_l222_222221

theorem total_earning (days_a days_b days_c : ℕ) (wage_ratio_a wage_ratio_b wage_ratio_c daily_wage_c total : ℕ)
  (h_ratio : wage_ratio_a = 3 ∧ wage_ratio_b = 4 ∧ wage_ratio_c = 5)
  (h_days : days_a = 6 ∧ days_b = 9 ∧ days_c = 4)
  (h_daily_wage_c : daily_wage_c = 125)
  (h_total : total = ((wage_ratio_a * (daily_wage_c / wage_ratio_c) * days_a) +
                     (wage_ratio_b * (daily_wage_c / wage_ratio_c) * days_b) +
                     (daily_wage_c * days_c))) : total = 1850 := by
  sorry

end total_earning_l222_222221


namespace oranges_and_apples_l222_222778

theorem oranges_and_apples (O A : ℕ) (h₁ : 7 * O = 5 * A) (h₂ : O = 28) : A = 20 :=
by {
  sorry
}

end oranges_and_apples_l222_222778


namespace percentage_of_difference_is_50_l222_222667

noncomputable def percentage_of_difference (x y : ℝ) (p : ℝ) :=
  (p / 100) * (x - y) = 0.20 * (x + y)

noncomputable def y_is_percentage_of_x (x y : ℝ) :=
  y = 0.42857142857142854 * x

theorem percentage_of_difference_is_50 (x y : ℝ) (p : ℝ)
  (h1 : percentage_of_difference x y p)
  (h2 : y_is_percentage_of_x x y) :
  p = 50 :=
by
  sorry

end percentage_of_difference_is_50_l222_222667


namespace wendy_created_albums_l222_222254

theorem wendy_created_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums : ℕ) :
  phone_pics = 22 → camera_pics = 2 → pics_per_album = 6 → total_pics = phone_pics + camera_pics → albums = total_pics / pics_per_album → albums = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end wendy_created_albums_l222_222254


namespace solution_set_l222_222212

variable (x : ℝ)

noncomputable def expr := (x - 1)^2 / (x - 5)^2

theorem solution_set :
  { x : ℝ | expr x ≥ 0 } = { x | x < 5 } ∪ { x | x > 5 } :=
by
  sorry

end solution_set_l222_222212


namespace triangle_angle_relation_l222_222103

theorem triangle_angle_relation 
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : b = (a + c) / Real.sqrt 2)
  (h2 : β = (α + γ) / 2)
  (h3 : c > a)
  : γ = α + 90 :=
sorry

end triangle_angle_relation_l222_222103


namespace mixed_gender_selection_count_is_correct_l222_222197

/- Define the given constants -/
def num_male_students : ℕ := 5
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students
def selection_size : ℕ := 3

/- Define the function to compute binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

/- The Lean 4 statement -/
theorem mixed_gender_selection_count_is_correct
  (num_male_students num_female_students total_students selection_size : ℕ)
  (hc1 : num_male_students = 5)
  (hc2 : num_female_students = 3)
  (hc3 : total_students = num_male_students + num_female_students)
  (hc4 : selection_size = 3) :
  binom total_students selection_size 
  - binom num_male_students selection_size
  - binom num_female_students selection_size = 45 := 
  by 
    -- Only the statement is required
    sorry

end mixed_gender_selection_count_is_correct_l222_222197


namespace value_of_h_l222_222540

theorem value_of_h (h : ℝ) : (∃ x : ℝ, x^3 + h * x - 14 = 0 ∧ x = 3) → h = -13/3 :=
by
  sorry

end value_of_h_l222_222540


namespace negation_equiv_l222_222124

variable (f : ℝ → ℝ)

theorem negation_equiv :
  ¬ (∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
sorry

end negation_equiv_l222_222124


namespace find_a_l222_222622

theorem find_a (a : ℚ) :
  let p1 := (3, 4)
  let p2 := (-4, 1)
  let direction_vector := (a, -2)
  let vector_between_points := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ k : ℚ, direction_vector = (k * vector_between_points.1, k * vector_between_points.2) →
  a = -14 / 3 := by
    sorry

end find_a_l222_222622


namespace merchant_markup_percentage_l222_222155

theorem merchant_markup_percentage
  (CP : ℕ) (discount_percent : ℚ) (profit_percent : ℚ)
  (mp : ℚ := CP + x)
  (sp : ℚ := (1 - discount_percent) * mp)
  (final_sp : ℚ := CP * (1 + profit_percent)) :
  discount_percent = 15 / 100 ∧ profit_percent = 19 / 100 ∧ CP = 100 → 
  sp = 85 + 0.85 * x → 
  final_sp = 119 →
  x = 40 :=
by 
  sorry

end merchant_markup_percentage_l222_222155


namespace triangles_with_vertex_A_l222_222698

theorem triangles_with_vertex_A : 
  ∃ (A : Point) (remaining_points : Finset Point), 
    (remaining_points.card = 8) → 
    (∃ (n : ℕ), n = (Nat.choose 8 2) ∧ n = 28) :=
by
  sorry

end triangles_with_vertex_A_l222_222698


namespace range_of_a_l222_222786

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + 3 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4 / 9) :=
sorry

end range_of_a_l222_222786


namespace local_maximum_at_neg2_l222_222122

noncomputable def y (x : ℝ) : ℝ :=
  (1/3) * x^3 - 4 * x + 4

theorem local_maximum_at_neg2 :
  ∃ x : ℝ, x = -2 ∧ 
           y x = 28/3 ∧
           (∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 2) < δ → y z < y (-2)) := by
  sorry

end local_maximum_at_neg2_l222_222122


namespace international_sales_correct_option_l222_222826

theorem international_sales_correct_option :
  (∃ (A B C D : String),
     A = "who" ∧
     B = "what" ∧
     C = "whoever" ∧
     D = "whatever" ∧
     (∃ x, x = C → "Could I speak to " ++ x ++ " is in charge of International Sales please?" = "Could I speak to whoever is in charge of International Sales please?")) :=
sorry

end international_sales_correct_option_l222_222826


namespace area_of_trapezoid_l222_222749

variable (a d : ℝ)
variable (h b1 b2 : ℝ)

def is_arithmetic_progression (a d : ℝ) (h b1 b2 : ℝ) : Prop :=
  h = a ∧ b1 = a + d ∧ b2 = a - d

theorem area_of_trapezoid (a d : ℝ) (h b1 b2 : ℝ) (hAP : is_arithmetic_progression a d h b1 b2) :
  ∃ J : ℝ, J = a^2 ∧ ∀ x : ℝ, 0 ≤ x → (J = x → x ≥ 0) :=
by
  sorry

end area_of_trapezoid_l222_222749


namespace reciprocal_div_calculate_fraction_reciprocal_div_result_l222_222765

-- Part 1
theorem reciprocal_div {a b c : ℚ} (h : (a + b) / c = -2) : c / (a + b) = -1 / 2 :=
sorry

-- Part 2
theorem calculate_fraction : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 :=
sorry

-- Part 3
theorem reciprocal_div_result : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 →
 (-1 / 36) / (5 / 12 - 1 / 9 + 2 / 3) = -1 / 35 :=
sorry

end reciprocal_div_calculate_fraction_reciprocal_div_result_l222_222765


namespace problem_prime_square_plus_two_l222_222151

theorem problem_prime_square_plus_two (P : ℕ) (hP_prime : Prime P) (hP2_plus_2_prime : Prime (P^2 + 2)) : P^4 + 1921 = 2002 :=
by
  sorry

end problem_prime_square_plus_two_l222_222151


namespace cars_given_by_mum_and_dad_l222_222037

-- Define the conditions given in the problem
def initial_cars : ℕ := 150
def final_cars : ℕ := 196
def cars_by_auntie : ℕ := 6
def cars_more_than_uncle : ℕ := 1
def cars_given_by_family (uncle : ℕ) (grandpa : ℕ) (auntie : ℕ) : ℕ :=
  uncle + grandpa + auntie

-- Prove the required statement
theorem cars_given_by_mum_and_dad :
  ∃ (uncle grandpa : ℕ), grandpa = 2 * uncle ∧ auntie = uncle + cars_more_than_uncle ∧ 
    auntie = cars_by_auntie ∧
    final_cars - initial_cars - cars_given_by_family uncle grandpa auntie = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end cars_given_by_mum_and_dad_l222_222037


namespace pipe_Q_drain_portion_l222_222369

noncomputable def portion_liquid_drain_by_Q (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) : ℝ :=
  let rate_P := 1 / T_P
  let rate_Q := 1 / T_Q
  let rate_R := 1 / T_R
  let combined_rate := rate_P + rate_Q + rate_R
  (rate_Q / combined_rate)

theorem pipe_Q_drain_portion (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) :
  portion_liquid_drain_by_Q T_Q T_P T_R h1 h2 = 3 / 11 :=
by
  sorry

end pipe_Q_drain_portion_l222_222369


namespace angle_bisector_proportion_l222_222435

theorem angle_bisector_proportion
  (p q r : ℝ)
  (u v : ℝ)
  (h1 : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < u ∧ 0 < v)
  (h2 : u + v = p)
  (h3 : u * q = v * r) :
  u / p = r / (r + q) :=
sorry

end angle_bisector_proportion_l222_222435


namespace prob_divisible_by_5_of_digits_ending_in_7_l222_222472

theorem prob_divisible_by_5_of_digits_ending_in_7 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ N % 10 = 7) → (0 : ℚ) = 0 :=
by
  intro N
  sorry

end prob_divisible_by_5_of_digits_ending_in_7_l222_222472


namespace right_triangles_count_l222_222253

theorem right_triangles_count (b a : ℕ) (h₁: b < 150) (h₂: (a^2 + b^2 = (b + 2)^2)) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ b = n^2 - 1 :=
by
  -- This intended to state the desired number and form of the right triangles.
  sorry

def count_right_triangles : ℕ :=
  12 -- Result as a constant based on proof steps

#eval count_right_triangles -- Should output 12

end right_triangles_count_l222_222253


namespace max_min_values_l222_222630

namespace ProofPrimary

-- Define the polynomial function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

-- State the interval of interest
def interval : Set ℝ := Set.Icc 1 11

-- Main theorem asserting the minimum and maximum values
theorem max_min_values : 
  (∀ x ∈ interval, f x ≥ -43 ∧ f x ≤ 2630) ∧
  (∃ x ∈ interval, f x = -43) ∧
  (∃ x ∈ interval, f x = 2630) :=
by
  sorry

end ProofPrimary

end max_min_values_l222_222630


namespace average_of_remaining_numbers_l222_222436

theorem average_of_remaining_numbers (S : ℕ) 
  (h₁ : S = 85 * 10) 
  (S' : ℕ) 
  (h₂ : S' = S - 70 - 76) : 
  S' / 8 = 88 := 
sorry

end average_of_remaining_numbers_l222_222436


namespace john_speed_above_limit_l222_222465

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end john_speed_above_limit_l222_222465


namespace part1_part2_l222_222632

-- Part 1: Proving the inequality
theorem part1 (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := by
  sorry

-- Part 2: Maximizing 2a + b
theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_constraint : a^2 + b^2 = 5) : 
  2 * a + b ≤ 5 := by
  sorry

end part1_part2_l222_222632


namespace reciprocal_equality_l222_222402

theorem reciprocal_equality (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_equality_l222_222402


namespace solve_for_y_l222_222958

theorem solve_for_y (y : ℤ) : 
  7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y) → y = -24 :=
by
  intro h
  sorry

end solve_for_y_l222_222958


namespace part_a_part_b_l222_222490

variable {f : ℝ → ℝ} 

-- Given conditions
axiom condition1 (x y : ℝ) : f (x + y) + 1 = f x + f y
axiom condition2 : f (1/2) = 0
axiom condition3 (x : ℝ) : x > 1/2 → f x < 0

-- Part (a)
theorem part_a (x : ℝ) : f x = 1/2 + 1/2 * f (2 * x) :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n > 0) (x : ℝ) 
  (hx : 1 / 2^(n + 1) ≤ x ∧ x ≤ 1 / 2^n) : f x ≤ 1 - 1 / 2^n :=
sorry

end part_a_part_b_l222_222490


namespace average_age_union_l222_222491

theorem average_age_union
    (A B C : Set Person)
    (a b c : ℕ)
    (sum_A sum_B sum_C : ℝ)
    (h_disjoint_AB : Disjoint A B)
    (h_disjoint_AC : Disjoint A C)
    (h_disjoint_BC : Disjoint B C)
    (h_avg_A : sum_A / a = 40)
    (h_avg_B : sum_B / b = 25)
    (h_avg_C : sum_C / c = 35)
    (h_avg_AB : (sum_A + sum_B) / (a + b) = 33)
    (h_avg_AC : (sum_A + sum_C) / (a + c) = 37.5)
    (h_avg_BC : (sum_B + sum_C) / (b + c) = 30) :
  (sum_A + sum_B + sum_C) / (a + b + c) = 51.6 :=
sorry

end average_age_union_l222_222491


namespace heaviest_person_is_Vanya_l222_222303

variables (A D T V M : ℕ)

-- conditions
def condition1 : Prop := A + D = 82
def condition2 : Prop := D + T = 74
def condition3 : Prop := T + V = 75
def condition4 : Prop := V + M = 65
def condition5 : Prop := M + A = 62

theorem heaviest_person_is_Vanya (h1 : condition1 A D) (h2 : condition2 D T) (h3 : condition3 T V) (h4 : condition4 V M) (h5 : condition5 M A) :
  V = 43 :=
sorry

end heaviest_person_is_Vanya_l222_222303


namespace initial_pile_counts_l222_222291

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end initial_pile_counts_l222_222291


namespace inequality_AM_GM_l222_222906

theorem inequality_AM_GM
  (a b c : ℝ)
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (habc : a + b + c = 1) : 
  (a + 2 * a * b + 2 * a * c + b * c) ^ a * 
  (b + 2 * b * c + 2 * b * a + c * a) ^ b * 
  (c + 2 * c * a + 2 * c * b + a * b) ^ c ≤ 1 :=
by
  sorry

end inequality_AM_GM_l222_222906


namespace a_seq_correct_l222_222889

-- Define the sequence and the sum condition
def a_seq (n : ℕ) : ℚ := if n = 0 then 0 else (2 ^ n - 1) / 2 ^ (n - 1)

def S_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.sum (Finset.range n) a_seq)

axiom condition (n : ℕ) (hn : n > 0) : S_n n + a_seq n = 2 * n

theorem a_seq_correct (n : ℕ) (hn : n > 0) : 
  a_seq n = (2 ^ n - 1) / 2 ^ (n - 1) := sorry

end a_seq_correct_l222_222889


namespace equation_solutions_l222_222513

noncomputable def count_solutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a <= 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else if a > Real.exp (1 / Real.exp 1) then 0
  else 0

theorem equation_solutions (a : ℝ) (h₀ : 0 < a) :
  (∃! x : ℝ, a^x = x) ↔ count_solutions a = 1 ∨ count_solutions a = 2 ∨ count_solutions a = 0 := sorry

end equation_solutions_l222_222513


namespace expected_pairs_of_adjacent_face_cards_is_44_over_17_l222_222330
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17_l222_222330


namespace icosahedron_to_octahedron_l222_222948

theorem icosahedron_to_octahedron : 
  ∃ (f : Finset (Fin 20)), f.card = 8 ∧ 
  (∀ {o : Finset (Fin 8)}, (True ∧ True)) ∧
  (∃ n : ℕ, n = 5) := by
  sorry

end icosahedron_to_octahedron_l222_222948


namespace average_speed_is_70_kmh_l222_222234

-- Define the given conditions
def distance1 : ℕ := 90
def distance2 : ℕ := 50
def time1 : ℕ := 1
def time2 : ℕ := 1

-- We need to prove that the average speed of the car is 70 km/h
theorem average_speed_is_70_kmh :
    ((distance1 + distance2) / (time1 + time2)) = 70 := 
by 
    -- This is the proof placeholder
    sorry

end average_speed_is_70_kmh_l222_222234


namespace eric_boxes_l222_222624

def numberOfBoxes (totalPencils : Nat) (pencilsPerBox : Nat) : Nat :=
  totalPencils / pencilsPerBox

theorem eric_boxes :
  numberOfBoxes 27 9 = 3 := by
  sorry

end eric_boxes_l222_222624


namespace find_missing_number_l222_222785

theorem find_missing_number (x : ℝ) :
  ((20 + 40 + 60) / 3) = ((10 + 70 + x) / 3) + 8 → x = 16 :=
by
  intro h
  sorry

end find_missing_number_l222_222785


namespace one_half_of_scientific_notation_l222_222705

theorem one_half_of_scientific_notation :
  (1 / 2) * (1.2 * 10 ^ 30) = 6.0 * 10 ^ 29 :=
by
  sorry

end one_half_of_scientific_notation_l222_222705


namespace triangle_area_is_64_l222_222011

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l222_222011


namespace kittens_given_is_two_l222_222808

-- Definitions of the conditions
def original_kittens : Nat := 8
def current_kittens : Nat := 6

-- Statement of the proof problem
theorem kittens_given_is_two : (original_kittens - current_kittens) = 2 := 
by
  sorry

end kittens_given_is_two_l222_222808


namespace chessboard_L_T_equivalence_l222_222327

theorem chessboard_L_T_equivalence (n : ℕ) :
  ∃ L_count T_count : ℕ, 
  (L_count = T_count) ∧ -- number of L-shaped pieces is equal to number of T-shaped pieces
  (L_count + T_count = n * (n + 1)) := 
sorry

end chessboard_L_T_equivalence_l222_222327


namespace sum_of_squares_transform_l222_222866

def isSumOfThreeSquaresDivByThree (N : ℕ) : Prop := 
  ∃ (a b c : ℤ), N = a^2 + b^2 + c^2 ∧ (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c)

def isSumOfThreeSquaresNotDivByThree (N : ℕ) : Prop := 
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z)

theorem sum_of_squares_transform {N : ℕ} :
  isSumOfThreeSquaresDivByThree N → isSumOfThreeSquaresNotDivByThree N :=
sorry

end sum_of_squares_transform_l222_222866


namespace find_f_at_6_l222_222300

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2

theorem find_f_at_6 (f : ℝ → ℝ) (h : example_function f) : f 6 = 4 := 
by
  sorry

end find_f_at_6_l222_222300


namespace one_quarters_in_one_eighth_l222_222729

theorem one_quarters_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 :=
by sorry

end one_quarters_in_one_eighth_l222_222729


namespace set_intersection_example_l222_222853

def universal_set := Set ℝ

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1 ∧ -2 ≤ x ∧ x ≤ 1}

def C : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

def complement (A : Set ℝ) : Set ℝ := {x : ℝ | x ∉ A}

def difference (A B : Set ℝ) : Set ℝ := A \ B

def union (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∨ x ∈ B}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection_example :
  intersection (complement A) (union B C) = {x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 4)} :=
by
  sorry

end set_intersection_example_l222_222853


namespace envelope_addressing_equation_l222_222938

theorem envelope_addressing_equation (x : ℝ) :
  (800 / 10 + 800 / x + 800 / 5) * (3 / 800) = 1 / 3 :=
  sorry

end envelope_addressing_equation_l222_222938


namespace sufficiency_condition_a_gt_b_sq_gt_sq_l222_222677

theorem sufficiency_condition_a_gt_b_sq_gt_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^2 > b^2) ∧ (∀ (h : a^2 > b^2), ∃ c > 0, ∃ d > 0, c^2 > d^2 ∧ ¬(c > d)) :=
by
  sorry

end sufficiency_condition_a_gt_b_sq_gt_sq_l222_222677


namespace compare_neg_rational_l222_222742

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l222_222742


namespace torn_pages_count_l222_222195

theorem torn_pages_count (pages : Finset ℕ) (h1 : ∀ p ∈ pages, 1 ≤ p ∧ p ≤ 100) (h2 : pages.sum id = 4949) : 
  100 - pages.card = 3 := 
by
  sorry

end torn_pages_count_l222_222195


namespace P_lt_Q_l222_222086

variable {x : ℝ}

def P (x : ℝ) : ℝ := (x - 2) * (x - 4)
def Q (x : ℝ) : ℝ := (x - 3) ^ 2

theorem P_lt_Q : P x < Q x := by
  sorry

end P_lt_Q_l222_222086


namespace triangle_base_angles_eq_l222_222946

theorem triangle_base_angles_eq
  (A B C C1 C2 : ℝ)
  (h1 : A > B)
  (h2 : C1 = 2 * C2)
  (h3 : A + B + C = 180)
  (h4 : B + C2 = 90)
  (h5 : C = C1 + C2) :
  A = B := by
  sorry

end triangle_base_angles_eq_l222_222946


namespace at_least_one_is_zero_l222_222788

theorem at_least_one_is_zero (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : false := by sorry

end at_least_one_is_zero_l222_222788


namespace proof_C_I_M_cap_N_l222_222010

open Set

variable {𝕜 : Type _} [LinearOrderedField 𝕜]

def I : Set 𝕜 := Set.univ
def M : Set 𝕜 := {x : 𝕜 | -2 ≤ x ∧ x ≤ 2}
def N : Set 𝕜 := {x : 𝕜 | x < 1}
def C_I (A : Set 𝕜) : Set 𝕜 := I \ A

theorem proof_C_I_M_cap_N :
  C_I M ∩ N = {x : 𝕜 | x < -2} := by
  sorry

end proof_C_I_M_cap_N_l222_222010


namespace chicken_cost_l222_222925

theorem chicken_cost (total_money hummus_price hummus_count bacon_price vegetables_price apple_price apple_count chicken_price : ℕ)
  (h_total_money : total_money = 60)
  (h_hummus_price : hummus_price = 5)
  (h_hummus_count : hummus_count = 2)
  (h_bacon_price : bacon_price = 10)
  (h_vegetables_price : vegetables_price = 10)
  (h_apple_price : apple_price = 2)
  (h_apple_count : apple_count = 5)
  (h_remaining_money : chicken_price = total_money - (hummus_count * hummus_price + bacon_price + vegetables_price + apple_count * apple_price)) :
  chicken_price = 20 := 
by sorry

end chicken_cost_l222_222925


namespace gcd_f_101_102_l222_222499

def f (x : ℕ) : ℕ := x^2 + x + 2010

theorem gcd_f_101_102 : Nat.gcd (f 101) (f 102) = 12 := 
by sorry

end gcd_f_101_102_l222_222499


namespace range_of_fraction_l222_222572

variable {x y : ℝ}

-- Condition given in the problem
def equation (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- The range condition for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 3

-- The corresponding theorem statement
theorem range_of_fraction (h_eq : equation x y) (h_x_range : x_range x) :
  ∃ a b : ℝ, (a < 1 ∧ 10 < b) ∧ (a, b) = (1, 10) ∧
  ∀ k : ℝ, k = (x + 2) / (y - 1) → 1 < k ∧ k < 10 :=
sorry

end range_of_fraction_l222_222572


namespace tangent_line_to_curve_l222_222717

theorem tangent_line_to_curve (a : ℝ) : (∀ (x : ℝ), y = x → y = a + Real.log x) → a = 1 := 
sorry

end tangent_line_to_curve_l222_222717


namespace election_majority_l222_222783

theorem election_majority (total_votes : ℕ) (winning_percentage : ℝ) (losing_percentage : ℝ)
  (h_total_votes : total_votes = 700)
  (h_winning_percentage : winning_percentage = 0.70)
  (h_losing_percentage : losing_percentage = 0.30) :
  (winning_percentage * total_votes - losing_percentage * total_votes) = 280 :=
by
  sorry

end election_majority_l222_222783


namespace find_m_l222_222236

variables (m x y : ℤ)

-- Conditions
def cond1 := x = 3 * m + 1
def cond2 := y = 2 * m - 2
def cond3 := 4 * x - 3 * y = 10

theorem find_m (h1 : cond1 m x) (h2 : cond2 m y) (h3 : cond3 x y) : m = 0 :=
by sorry

end find_m_l222_222236


namespace inequality_abc_l222_222937

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) >= (a + b + c) / 3 := by
  sorry

end inequality_abc_l222_222937


namespace t_shirt_cost_l222_222269

theorem t_shirt_cost (total_amount_spent : ℝ) (number_of_t_shirts : ℕ) (cost_per_t_shirt : ℝ)
  (h0 : total_amount_spent = 201) 
  (h1 : number_of_t_shirts = 22)
  (h2 : cost_per_t_shirt = total_amount_spent / number_of_t_shirts) :
  cost_per_t_shirt = 9.14 := 
sorry

end t_shirt_cost_l222_222269


namespace smallest_number_append_l222_222872

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end smallest_number_append_l222_222872


namespace linear_equation_in_options_l222_222100

def is_linear_equation_with_one_variable (eqn : String) : Prop :=
  eqn = "3 - 2x = 5"

theorem linear_equation_in_options :
  is_linear_equation_with_one_variable "3 - 2x = 5" :=
by
  sorry

end linear_equation_in_options_l222_222100


namespace initial_bananas_per_child_l222_222662

theorem initial_bananas_per_child : 
  ∀ (B n m x : ℕ), 
  n = 740 → 
  m = 370 → 
  (B = n * x) → 
  (B = (n - m) * (x + 2)) → 
  x = 2 := 
by
  intros B n m x h1 h2 h3 h4
  sorry

end initial_bananas_per_child_l222_222662


namespace opposite_of_5_is_neg5_l222_222565

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l222_222565


namespace students_with_uncool_parents_but_cool_siblings_l222_222243

-- The total number of students in the classroom
def total_students : ℕ := 40

-- The number of students with cool dads
def students_with_cool_dads : ℕ := 18

-- The number of students with cool moms
def students_with_cool_moms : ℕ := 22

-- The number of students with both cool dads and cool moms
def students_with_both_cool_parents : ℕ := 10

-- The number of students with cool siblings
def students_with_cool_siblings : ℕ := 8

-- The theorem we want to prove
theorem students_with_uncool_parents_but_cool_siblings
  (h1 : total_students = 40)
  (h2 : students_with_cool_dads = 18)
  (h3 : students_with_cool_moms = 22)
  (h4 : students_with_both_cool_parents = 10)
  (h5 : students_with_cool_siblings = 8) :
  8 = (students_with_cool_siblings) :=
sorry

end students_with_uncool_parents_but_cool_siblings_l222_222243


namespace geometric_progression_condition_l222_222620

theorem geometric_progression_condition (a b c : ℝ) (h_b_neg : b < 0) : 
  (b^2 = a * c) ↔ (∃ (r : ℝ), a = r * b ∧ b = r * c) :=
sorry

end geometric_progression_condition_l222_222620


namespace tan_addition_formula_15_30_l222_222000

-- Define tangent function for angles in degrees.
noncomputable def tanDeg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem for the given problem
theorem tan_addition_formula_15_30 :
  tanDeg 15 + tanDeg 30 + tanDeg 15 * tanDeg 30 = 1 :=
by
  -- Here we use the given conditions and properties in solution
  sorry

end tan_addition_formula_15_30_l222_222000


namespace solve_for_x_l222_222263

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  (4 * y^2 + y + 6 = 3 * (9 * x^2 + y + 3)) ↔ (x = 1 ∨ x = -1/3) :=
by
  sorry

end solve_for_x_l222_222263


namespace coloring_methods_390_l222_222916

def numColoringMethods (colors cells : ℕ) (maxColors : ℕ) : ℕ :=
  if colors = 6 ∧ cells = 4 ∧ maxColors = 3 then 390 else 0

theorem coloring_methods_390 :
  numColoringMethods 6 4 3 = 390 :=
by 
  sorry

end coloring_methods_390_l222_222916


namespace percentage_of_students_in_band_l222_222714

theorem percentage_of_students_in_band 
  (students_in_band : ℕ)
  (total_students : ℕ)
  (students_in_band_eq : students_in_band = 168)
  (total_students_eq : total_students = 840) :
  (students_in_band / total_students : ℚ) * 100 = 20 :=
by
  sorry

end percentage_of_students_in_band_l222_222714


namespace total_shirts_sold_l222_222153

theorem total_shirts_sold (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : ℕ) (h1 : p1 = 20) (h2 : p2 = 22) (h3 : p3 = 25)
(h4 : p4 + p5 + p6 + p7 + p8 + p9 + p10 = 133) (h5 : ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10) / 10) > 20)
: p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 = 200 ∧ 10 = 10 := sorry

end total_shirts_sold_l222_222153


namespace james_monthly_earnings_l222_222105

theorem james_monthly_earnings :
  let initial_subscribers := 150
  let gifted_subscribers := 50
  let rate_per_subscriber := 9
  let total_subscribers := initial_subscribers + gifted_subscribers
  let total_earnings := total_subscribers * rate_per_subscriber
  total_earnings = 1800 := by
  sorry

end james_monthly_earnings_l222_222105


namespace certain_multiple_l222_222432

theorem certain_multiple (n m : ℤ) (h : n = 5) (eq : 7 * n - 15 = m * n + 10) : m = 2 :=
by
  sorry

end certain_multiple_l222_222432


namespace interior_points_in_divided_square_l222_222816

theorem interior_points_in_divided_square :
  ∀ (n : ℕ), 
  (n = 2016) →
  ∃ (k : ℕ), 
  (∀ (t : ℕ), t = 180 * n) → 
  k = 1007 :=
by
  intros n hn
  use 1007
  sorry

end interior_points_in_divided_square_l222_222816


namespace factorize_expression_l222_222935

theorem factorize_expression (x y : ℝ) : (y + 2 * x)^2 - (x + 2 * y)^2 = 3 * (x + y) * (x - y) :=
  sorry

end factorize_expression_l222_222935


namespace mosquito_distance_ratio_l222_222886

-- Definition of the clock problem conditions
structure ClockInsects where
  distance_from_center : ℕ
  initial_time : ℕ := 1

-- Prove the ratio of distances traveled by mosquito and fly over 12 hours
theorem mosquito_distance_ratio (c : ClockInsects) :
  let mosquito_distance := (83 : ℚ)/12
  let fly_distance := (73 : ℚ)/12
  mosquito_distance / fly_distance = 83 / 73 :=
by 
  sorry

end mosquito_distance_ratio_l222_222886


namespace compare_fractions_l222_222166

variable {a b c d : ℝ}

theorem compare_fractions (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  (b / (a - c)) < (a / (b - d)) := 
by
  sorry

end compare_fractions_l222_222166


namespace possible_values_of_ab_plus_ac_plus_bc_l222_222058

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ x ∈ Set.Iic 0, ab + ac + bc = x :=
sorry

end possible_values_of_ab_plus_ac_plus_bc_l222_222058


namespace total_units_l222_222480

theorem total_units (A B C: ℕ) (hA: A = 2 + 4 + 6 + 8 + 10 + 12) (hB: B = A) (hC: C = 3 + 5 + 7 + 9) : 
  A + B + C = 108 := 
sorry

end total_units_l222_222480


namespace present_price_after_discount_l222_222536

theorem present_price_after_discount :
  ∀ (P : ℝ), (∀ x : ℝ, (3 * x = P - 0.20 * P) ∧ (x = (P / 3) - 4)) → P = 60 → 0.80 * P = 48 :=
by
  intros P hP h60
  sorry

end present_price_after_discount_l222_222536


namespace exists_congruent_triangle_covering_with_parallel_side_l222_222321

variable {Point : Type}
variable [MetricSpace Point]
variable {Triangle : Type}
variable {Polygon : Type}

-- Definitions of triangle and polygon covering relationships.
def covers (T : Triangle) (P : Polygon) : Prop := sorry 
def congruent (T1 T2 : Triangle) : Prop := sorry
def side_parallel_or_coincident (T : Triangle) (P : Polygon) : Prop := sorry

-- Statement: Given a triangle covering a polygon, there exists a congruent triangle which covers the polygon 
-- and has one side parallel to or coincident with a side of the polygon.
theorem exists_congruent_triangle_covering_with_parallel_side 
  (ABC : Triangle) (M : Polygon) 
  (h_cover : covers ABC M) : 
  ∃ Δ : Triangle, congruent Δ ABC ∧ covers Δ M ∧ side_parallel_or_coincident Δ M := 
sorry

end exists_congruent_triangle_covering_with_parallel_side_l222_222321


namespace toby_total_time_l222_222244

def speed_unloaded := 20 -- Speed of Toby pulling unloaded sled in mph
def speed_loaded := 10   -- Speed of Toby pulling loaded sled in mph

def distance_part1 := 180 -- Distance for the first part (loaded sled) in miles
def distance_part2 := 120 -- Distance for the second part (unloaded sled) in miles
def distance_part3 := 80  -- Distance for the third part (loaded sled) in miles
def distance_part4 := 140 -- Distance for the fourth part (unloaded sled) in miles

def time_part1 := distance_part1 / speed_loaded -- Time for the first part in hours
def time_part2 := distance_part2 / speed_unloaded -- Time for the second part in hours
def time_part3 := distance_part3 / speed_loaded -- Time for the third part in hours
def time_part4 := distance_part4 / speed_unloaded -- Time for the fourth part in hours

def total_time := time_part1 + time_part2 + time_part3 + time_part4 -- Total time in hours

theorem toby_total_time : total_time = 39 :=
by 
  sorry

end toby_total_time_l222_222244


namespace percentage_markup_l222_222595

variable (W R : ℝ) -- W is the wholesale cost, R is the normal retail price

-- The condition that, at 60% discount, the sale price nets a 35% profit on the wholesale cost
variable (h : 0.4 * R = 1.35 * W)

-- The goal statement to prove
theorem percentage_markup (h : 0.4 * R = 1.35 * W) : ((R - W) / W) * 100 = 237.5 :=
by
  sorry

end percentage_markup_l222_222595


namespace Brenda_new_lead_l222_222542

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end Brenda_new_lead_l222_222542


namespace proof_of_independence_l222_222806

/-- A line passing through the plane of two parallel lines and intersecting one of them also intersects the other. -/
def independent_of_parallel_postulate (statement : String) : Prop :=
  statement = "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other."

theorem proof_of_independence :
  independent_of_parallel_postulate "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other." :=
sorry

end proof_of_independence_l222_222806


namespace evaluate_expression_l222_222608

theorem evaluate_expression : 12^2 + 2 * 12 * 5 + 5^2 = 289 := by
  sorry

end evaluate_expression_l222_222608


namespace smallest_x_value_l222_222481

theorem smallest_x_value : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (3 : ℚ) / 4 = y / (250 + x) ∧ x = 2 := by
  sorry

end smallest_x_value_l222_222481


namespace problem_statement_l222_222106

theorem problem_statement (x : ℕ) (h : 4 * (3^x) = 2187) : (x + 2) * (x - 2) = 21 := 
by
  sorry

end problem_statement_l222_222106


namespace total_revenue_is_correct_l222_222385

-- Joan decided to sell all of her old books.
-- She had 33 books in total.
-- She sold 15 books at $4 each.
-- She sold 6 books at $7 each.
-- The rest of the books were sold at $10 each.
-- We need to prove that the total revenue is $222.

def totalBooks := 33
def booksAt4 := 15
def priceAt4 := 4
def booksAt7 := 6
def priceAt7 := 7
def priceAt10 := 10
def remainingBooks := totalBooks - (booksAt4 + booksAt7)
def revenueAt4 := booksAt4 * priceAt4
def revenueAt7 := booksAt7 * priceAt7
def revenueAt10 := remainingBooks * priceAt10
def totalRevenue := revenueAt4 + revenueAt7 + revenueAt10

theorem total_revenue_is_correct : totalRevenue = 222 := by
  sorry

end total_revenue_is_correct_l222_222385


namespace remainder_b94_mod_55_eq_29_l222_222307

theorem remainder_b94_mod_55_eq_29 :
  (5^94 + 7^94) % 55 = 29 := 
by
  -- conditions: local definitions for bn, modulo, etc.
  sorry

end remainder_b94_mod_55_eq_29_l222_222307


namespace claire_apple_pies_l222_222373

theorem claire_apple_pies (N : ℤ) 
  (h1 : N % 6 = 4) 
  (h2 : N % 8 = 5) 
  (h3 : N < 30) : 
  N = 22 :=
by
  sorry

end claire_apple_pies_l222_222373


namespace tangent_line_at_P_l222_222146

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x

theorem tangent_line_at_P 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : P.1 + P.2 = 0)
  (h2 : f' P.1 a = -1) 
  (h3 : P.2 = f P.1 a) 
  : P = (1, -1) ∨ P = (-1, 1) := 
  sorry

end tangent_line_at_P_l222_222146


namespace sum_coordinates_l222_222271

variables (x y : ℝ)
def A_coord := (9, 3)
def M_coord := (3, 7)

def midpoint_condition_x : Prop := (x + 9) / 2 = 3
def midpoint_condition_y : Prop := (y + 3) / 2 = 7

theorem sum_coordinates (h1 : midpoint_condition_x x) (h2 : midpoint_condition_y y) : 
  x + y = 8 :=
by 
  sorry

end sum_coordinates_l222_222271


namespace sum_q_p_values_l222_222095

def p (x : ℤ) : ℤ := x^2 - 4
def q (x : ℤ) : ℤ := -x

def q_p_composed (x : ℤ) : ℤ := q (p x)

theorem sum_q_p_values :
  q_p_composed (-3) + q_p_composed (-2) + q_p_composed (-1) + q_p_composed 0 + 
  q_p_composed 1 + q_p_composed 2 + q_p_composed 3 = 0 := by
  sorry

end sum_q_p_values_l222_222095


namespace marys_birthday_l222_222324

theorem marys_birthday (M : ℝ) (h1 : (3 / 4) * M - (3 / 20) * M = 60) : M = 100 := by
  -- Leave the proof as sorry for now
  sorry

end marys_birthday_l222_222324


namespace ratio_of_perimeters_l222_222898

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem ratio_of_perimeters (d1 : ℝ) :
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2 
  (P2 / P1 = 1 + sqrt2) :=
by
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2
  sorry

end ratio_of_perimeters_l222_222898


namespace mrs_hilt_money_l222_222548

-- Definitions and given conditions
def cost_of_pencil := 5  -- in cents
def number_of_pencils := 10

-- The theorem we need to prove
theorem mrs_hilt_money : cost_of_pencil * number_of_pencils = 50 := by
  sorry

end mrs_hilt_money_l222_222548


namespace ellipse_eccentricity_proof_l222_222944

theorem ellipse_eccentricity_proof (a b c : ℝ) 
  (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hc_gt_zero : c > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_r : ∃ r : ℝ, r = (Real.sqrt 2 / 6) * c) :
  (Real.sqrt (1 - b^2 / a^2)) = (2 * Real.sqrt 5 / 5) := by {
  sorry
}

end ellipse_eccentricity_proof_l222_222944


namespace sum_fractions_correct_l222_222984

def sum_of_fractions : Prop :=
  (3 / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386)

theorem sum_fractions_correct : sum_of_fractions :=
by
  sorry

end sum_fractions_correct_l222_222984


namespace pairs_sold_l222_222997

theorem pairs_sold (total_sales : ℝ) (avg_price_per_pair : ℝ) (h1 : total_sales = 490) (h2 : avg_price_per_pair = 9.8) :
  total_sales / avg_price_per_pair = 50 :=
by
  rw [h1, h2]
  norm_num

end pairs_sold_l222_222997


namespace area_of_L_shape_l222_222302

theorem area_of_L_shape (a : ℝ) (h_pos : a > 0) (h_eq : 4 * ((a + 3)^2 - a^2) = 5 * a^2) : 
  (a + 3)^2 - a^2 = 45 :=
by
  sorry

end area_of_L_shape_l222_222302


namespace probability_king_of_diamonds_top_two_l222_222496

-- Definitions based on the conditions
def total_cards : ℕ := 54
def king_of_diamonds : ℕ := 1
def jokers : ℕ := 2

-- The main theorem statement proving the probability
theorem probability_king_of_diamonds_top_two :
  let prob := (king_of_diamonds / total_cards) + ((total_cards - 1) / total_cards * king_of_diamonds / (total_cards - 1))
  prob = 1 / 27 :=
by
  sorry

end probability_king_of_diamonds_top_two_l222_222496


namespace grocer_second_month_sale_l222_222668

theorem grocer_second_month_sale (sale_1 sale_3 sale_4 sale_5 sale_6 avg_sale n : ℕ) 
(h1 : sale_1 = 6435) 
(h3 : sale_3 = 6855) 
(h4 : sale_4 = 7230) 
(h5 : sale_5 = 6562) 
(h6 : sale_6 = 7391) 
(havg : avg_sale = 6900) 
(hn : n = 6) : 
  sale_2 = 6927 :=
by
  sorry

end grocer_second_month_sale_l222_222668


namespace least_possible_value_l222_222682

theorem least_possible_value (y q p : ℝ) (h1: 5 < y) (h2: y < 7)
  (hq: q = 7) (hp: p = 5) : q - p = 2 :=
by
  sorry

end least_possible_value_l222_222682


namespace total_profit_correct_l222_222028

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct_l222_222028


namespace find_ab_l222_222425

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  a = -1 ∧ b = 2 :=
by
  sorry

end find_ab_l222_222425


namespace pages_to_read_l222_222545

variable (E P_Science P_Civics P_Chinese Total : ℕ)
variable (h_Science : P_Science = 16)
variable (h_Civics : P_Civics = 8)
variable (h_Chinese : P_Chinese = 12)
variable (h_Total : Total = 14)

theorem pages_to_read :
  (E / 4) + (P_Science / 4) + (P_Civics / 4) + (P_Chinese / 4) = Total → 
  E = 20 := by
  sorry

end pages_to_read_l222_222545


namespace number_is_76_l222_222423

theorem number_is_76 (x : ℝ) (h : (3 / 4) * x = x - 19) : x = 76 :=
sorry

end number_is_76_l222_222423


namespace combined_original_price_l222_222845

def original_price_shoes (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

def original_price_dress (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

theorem combined_original_price (shoes_price : ℚ) (shoes_discount : ℚ) (dress_price : ℚ) (dress_discount : ℚ) 
  (h_shoes : shoes_discount = 0.20 ∧ shoes_price = 480) 
  (h_dress : dress_discount = 0.30 ∧ dress_price = 350) : 
  original_price_shoes shoes_price shoes_discount + original_price_dress dress_price dress_discount = 1100 := by
  sorry

end combined_original_price_l222_222845


namespace equilateral_given_inequality_l222_222764

open Real

-- Define the primary condition to be used in the theorem
def inequality (a b c : ℝ) : Prop :=
  (1 / a * sqrt (1 / b + 1 / c) + 1 / b * sqrt (1 / c + 1 / a) + 1 / c * sqrt (1 / a + 1 / b)) ≥
  (3 / 2 * sqrt ((1 / a + 1 / b) * (1 / b + 1 / c) * (1 / c + 1 / a)))

-- Define the theorem that states the sides form an equilateral triangle under the given condition
theorem equilateral_given_inequality (a b c : ℝ) (habc : inequality a b c) (htriangle : a > 0 ∧ b > 0 ∧ c > 0):
  a = b ∧ b = c ∧ c = a := 
sorry

end equilateral_given_inequality_l222_222764


namespace coin_toss_tails_count_l222_222040

theorem coin_toss_tails_count (flips : ℕ) (frequency_heads : ℝ) (h_flips : flips = 20) (h_frequency_heads : frequency_heads = 0.45) : 
  (20 : ℝ) * (1 - 0.45) = 11 := 
by
  sorry

end coin_toss_tails_count_l222_222040


namespace find_length_of_polaroid_l222_222720

theorem find_length_of_polaroid 
  (C : ℝ) (W : ℝ) (L : ℝ)
  (hC : C = 40) (hW : W = 8) 
  (hFormula : C = 2 * (L + W)) : 
  L = 12 :=
by
  sorry

end find_length_of_polaroid_l222_222720


namespace number_of_numbers_tadd_said_after_20_rounds_l222_222917

-- Define the arithmetic sequence representing the count of numbers Tadd says each round
def tadd_sequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Define the sum of the first n terms of Tadd's sequence
def sum_tadd_sequence (n : ℕ) : ℕ :=
  n * (1 + tadd_sequence n) / 2

-- The main theorem to state the problem
theorem number_of_numbers_tadd_said_after_20_rounds :
  sum_tadd_sequence 20 = 400 :=
by
  -- The actual proof should be filled in here
  sorry

end number_of_numbers_tadd_said_after_20_rounds_l222_222917


namespace min_distance_l222_222460

noncomputable def point_on_curve (x₁ y₁ : ℝ) : Prop :=
  y₁ = x₁^2 - Real.log x₁

noncomputable def point_on_line (x₂ y₂ : ℝ) : Prop :=
  x₂ - y₂ - 2 = 0

theorem min_distance 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : point_on_curve x₁ y₁)
  (h₂ : point_on_line x₂ y₂) 
  : (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2 :=
sorry

end min_distance_l222_222460


namespace right_triangle_other_side_l222_222724

theorem right_triangle_other_side (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 17) (h_a : a = 15) : b = 8 := 
by
  sorry

end right_triangle_other_side_l222_222724


namespace simplify_expression_l222_222981

theorem simplify_expression :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) *
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 :=
by 
  sorry

end simplify_expression_l222_222981


namespace product_xyz_equals_zero_l222_222131

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l222_222131


namespace remainder_of_3_pow_500_mod_17_l222_222859

theorem remainder_of_3_pow_500_mod_17 : (3 ^ 500) % 17 = 13 := 
by
  sorry

end remainder_of_3_pow_500_mod_17_l222_222859


namespace debby_weekly_jog_distance_l222_222201

theorem debby_weekly_jog_distance :
  let monday_distance := 3.0
  let tuesday_distance := 5.5
  let wednesday_distance := 9.7
  let thursday_distance := 10.8
  let friday_distance_miles := 2.0
  let miles_to_km := 1.60934
  let friday_distance := friday_distance_miles * miles_to_km
  let total_distance := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance
  total_distance = 32.21868 :=
by
  sorry

end debby_weekly_jog_distance_l222_222201


namespace vector_c_equals_combination_l222_222833

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)
def vector_c : ℝ × ℝ := (-2, 4)

theorem vector_c_equals_combination : vector_c = (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2) :=
sorry

end vector_c_equals_combination_l222_222833


namespace least_prime_factor_of_11_pow4_minus_11_pow3_l222_222803

open Nat

theorem least_prime_factor_of_11_pow4_minus_11_pow3 : 
  Nat.minFac (11^4 - 11^3) = 2 :=
  sorry

end least_prime_factor_of_11_pow4_minus_11_pow3_l222_222803


namespace relationship_between_a_and_b_l222_222441

-- Definitions based on the conditions
def point1_lies_on_line (a : ℝ) : Prop := a = (2/3 : ℝ) * (-1 : ℝ) - 3
def point2_lies_on_line (b : ℝ) : Prop := b = (2/3 : ℝ) * (1/2 : ℝ) - 3

-- The main theorem to prove the relationship between a and b
theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : point1_lies_on_line a)
  (h2 : point2_lies_on_line b) : a < b :=
by
  -- Skipping the actual proof. Including sorry to indicate it's not provided.
  sorry

end relationship_between_a_and_b_l222_222441


namespace bulbs_arrangement_l222_222196

theorem bulbs_arrangement :
  let blue_bulbs := 5
  let red_bulbs := 8
  let white_bulbs := 11
  let total_non_white_bulbs := blue_bulbs + red_bulbs
  let total_gaps := total_non_white_bulbs + 1
  (Nat.choose 13 5) * (Nat.choose total_gaps white_bulbs) = 468468 :=
by
  sorry

end bulbs_arrangement_l222_222196


namespace polynomial_determination_l222_222681

theorem polynomial_determination (P : Polynomial ℝ) :
  (∀ X : ℝ, P.eval (X^2) = (X^2 + 1) * P.eval X) →
  (∃ a : ℝ, ∀ X : ℝ, P.eval X = a * (X^2 - 1)) :=
by
  sorry

end polynomial_determination_l222_222681


namespace necessary_but_not_sufficient_condition_l222_222387

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  ((1 / a < 1 ↔ a < 0 ∨ a > 1) ∧ ¬(1 / a < 1 → a ≤ 0 ∨ a ≤ 1)) := 
by sorry

end necessary_but_not_sufficient_condition_l222_222387


namespace problem_solution_l222_222564

variable (x y : ℝ)

-- Conditions
axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : (4 * x - 3 * y) / (x + 4 * y) = 3

-- Goal
theorem problem_solution : (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 :=
by
  sorry

end problem_solution_l222_222564


namespace calculate_glass_area_l222_222735

-- Given conditions as definitions
def long_wall_length : ℕ := 30
def long_wall_height : ℕ := 12
def short_wall_length : ℕ := 20

-- Total area of glass required (what we want to prove)
def total_glass_area : ℕ := 960

-- The theorem to prove
theorem calculate_glass_area
  (a1 : long_wall_length = 30)
  (a2 : long_wall_height = 12)
  (a3 : short_wall_length = 20) :
  2 * (long_wall_length * long_wall_height) + (short_wall_length * long_wall_height) = total_glass_area :=
by
  -- The proof is omitted
  sorry

end calculate_glass_area_l222_222735


namespace sum_of_coefficients_l222_222736

noncomputable def expand_and_sum_coefficients (d : ℝ) : ℝ :=
  let poly := -2 * (4 - d) * (d + 3 * (4 - d))
  let expanded := -4 * d^2 + 40 * d - 96
  let sum_coefficients := (-4) + 40 + (-96)
  sum_coefficients

theorem sum_of_coefficients (d : ℝ) : expand_and_sum_coefficients d = -60 := by
  sorry

end sum_of_coefficients_l222_222736


namespace problem_a_problem_b_problem_c_l222_222231

noncomputable def inequality_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (0 * y + 1)) + 1 / (y * (0 * z + 1)) + 1 / (z * (0 * x + 1))) ≥ 3

noncomputable def inequality_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (1 * y + 0)) + 1 / (y * (1 * z + 0)) + 1 / (z * (1 * x + 0))) ≥ 3

noncomputable def inequality_c (x y z : ℝ) (a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : Prop :=
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b))) ≥ 3

theorem problem_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_a x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_b x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_c (x y z a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : inequality_c x y z a b h1 h2 h3 h4 h5 h6 h7 :=
  by sorry

end problem_a_problem_b_problem_c_l222_222231


namespace hydrocarbon_tree_configurations_l222_222080

theorem hydrocarbon_tree_configurations (n : ℕ) 
  (h1 : 3 * n + 2 > 0) -- Total vertices count must be positive
  (h2 : 2 * n + 2 > 0) -- Leaves count must be positive
  (h3 : n > 0) -- Internal nodes count must be positive
  : (n:ℕ) ^ (n-2) = n ^ (n-2) :=
sorry

end hydrocarbon_tree_configurations_l222_222080


namespace identity_implies_a_minus_b_l222_222922

theorem identity_implies_a_minus_b (a b : ℚ) (y : ℚ) (h : y > 0) :
  (∀ y, y > 0 → (a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5)))) → (a - b = 1) :=
by
  sorry

end identity_implies_a_minus_b_l222_222922


namespace unique_solution_xy_l222_222543

theorem unique_solution_xy
  (x y : ℕ)
  (h1 : (x^3 + y) % (x^2 + y^2) = 0)
  (h2 : (y^3 + x) % (x^2 + y^2) = 0) :
  x = 1 ∧ y = 1 := sorry

end unique_solution_xy_l222_222543


namespace factor_problem_l222_222584

theorem factor_problem (C D : ℤ) (h1 : 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) (h2 : C * D + C = 21) : C = 7 ∧ D = 2 :=
by 
  sorry

end factor_problem_l222_222584


namespace instantaneous_acceleration_at_3_l222_222787

def v (t : ℝ) : ℝ := t^2 + 3

theorem instantaneous_acceleration_at_3 :
  deriv v 3 = 6 :=
by
  sorry

end instantaneous_acceleration_at_3_l222_222787


namespace total_colors_over_two_hours_l222_222708

def colors_in_first_hour : Nat :=
  let quick_colors := 5 * 3
  let slow_colors := 2 * 3
  quick_colors + slow_colors

def colors_in_second_hour : Nat :=
  let quick_colors := (5 * 2) * 3
  let slow_colors := (2 * 2) * 3
  quick_colors + slow_colors

theorem total_colors_over_two_hours : colors_in_first_hour + colors_in_second_hour = 63 := by
  sorry

end total_colors_over_two_hours_l222_222708


namespace dice_product_probability_is_one_l222_222121

def dice_probability_product_is_one : Prop :=
  ∀ (a b c d e : ℕ), (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 → 
    (a * b * c * d * e) = 1) ∧
  ∃ (p : ℚ), p = (1/6)^5 ∧ p = 1/7776

theorem dice_product_probability_is_one (a b c d e : ℕ) :
  dice_probability_product_is_one :=
by
  sorry

end dice_product_probability_is_one_l222_222121


namespace cans_per_person_on_second_day_l222_222701

theorem cans_per_person_on_second_day :
  ∀ (initial_stock : ℕ) (people_first_day : ℕ) (cans_taken_first_day : ℕ)
    (restock_first_day : ℕ) (people_second_day : ℕ)
    (restock_second_day : ℕ) (total_cans_given : ℕ) (cans_per_person_second_day : ℚ),
    cans_taken_first_day = 1 →
    initial_stock = 2000 →
    people_first_day = 500 →
    restock_first_day = 1500 →
    people_second_day = 1000 →
    restock_second_day = 3000 →
    total_cans_given = 2500 →
    cans_per_person_second_day = total_cans_given / people_second_day →
    cans_per_person_second_day = 2.5 := by
  sorry

end cans_per_person_on_second_day_l222_222701


namespace feathers_per_crown_l222_222840

theorem feathers_per_crown (total_feathers total_crowns feathers_per_crown : ℕ) 
  (h₁ : total_feathers = 6538) 
  (h₂ : total_crowns = 934) 
  (h₃ : feathers_per_crown = total_feathers / total_crowns) : 
  feathers_per_crown = 7 := 
by 
  sorry

end feathers_per_crown_l222_222840


namespace golden_apples_per_pint_l222_222180

-- Data definitions based on given conditions and question
def farmhands : ℕ := 6
def apples_per_hour : ℕ := 240
def hours : ℕ := 5
def ratio_golden_to_pink : ℕ × ℕ := (1, 2)
def pints_of_cider : ℕ := 120
def pink_lady_per_pint : ℕ := 40

-- Total apples picked by farmhands in 5 hours
def total_apples_picked : ℕ := farmhands * apples_per_hour * hours

-- Total pink lady apples picked
def total_pink_lady_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.2) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total golden delicious apples picked
def total_golden_delicious_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.1) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total pink lady apples used for 120 pints of cider
def pink_lady_apples_used : ℕ := pints_of_cider * pink_lady_per_pint

-- Number of golden delicious apples used per pint of cider
def golden_delicious_apples_per_pint : ℕ := total_golden_delicious_apples / pints_of_cider

-- Main theorem to prove
theorem golden_apples_per_pint : golden_delicious_apples_per_pint = 20 := by
  -- Start proof (proof body is omitted)
  sorry

end golden_apples_per_pint_l222_222180


namespace division_quotient_l222_222582

-- Define conditions
def dividend : ℕ := 686
def divisor : ℕ := 36
def remainder : ℕ := 2

-- Define the quotient
def quotient : ℕ := dividend - remainder

theorem division_quotient :
  quotient = divisor * 19 :=
sorry

end division_quotient_l222_222582


namespace john_new_salary_after_raise_l222_222019

theorem john_new_salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (h1 : original_salary = 60) (h2 : percentage_increase = 0.8333333333333334) : 
  original_salary * (1 + percentage_increase) = 110 := 
sorry

end john_new_salary_after_raise_l222_222019


namespace find_g_neg6_l222_222141

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l222_222141


namespace bacterium_probability_l222_222182

noncomputable def probability_bacterium_in_small_cup
  (total_volume : ℚ) (small_cup_volume : ℚ) (contains_bacterium : Bool) : ℚ :=
if contains_bacterium then small_cup_volume / total_volume else 0

theorem bacterium_probability
  (total_volume : ℚ) (small_cup_volume : ℚ) (bacterium_present : Bool) :
  total_volume = 2 ∧ small_cup_volume = 0.1 ∧ bacterium_present = true →
  probability_bacterium_in_small_cup 2 0.1 true = 0.05 :=
by
  intros h
  sorry

end bacterium_probability_l222_222182


namespace third_player_matches_l222_222047

theorem third_player_matches (first_player second_player third_player : ℕ) (h1 : first_player = 10) (h2 : second_player = 21) :
  third_player = 11 :=
by
  sorry

end third_player_matches_l222_222047


namespace red_ballpoint_pens_count_l222_222972

theorem red_ballpoint_pens_count (R B : ℕ) (h1: R + B = 240) (h2: B = R - 2) : R = 121 :=
by
  sorry

end red_ballpoint_pens_count_l222_222972


namespace bug_visits_exactly_16_pavers_l222_222260

-- Defining the dimensions of the garden and the pavers
def garden_width : ℕ := 14
def garden_length : ℕ := 19
def paver_size : ℕ := 2

-- Calculating the number of pavers in width and length
def pavers_width : ℕ := garden_width / paver_size
def pavers_length : ℕ := (garden_length + paver_size - 1) / paver_size  -- Taking ceiling of 19/2

-- Calculating the GCD of the pavers count in width and length
def gcd_pavers : ℕ := Nat.gcd pavers_width pavers_length

-- Calculating the number of pavers the bug crosses
def pavers_crossed : ℕ := pavers_width + pavers_length - gcd_pavers

-- Theorem that states the number of pavers visited
theorem bug_visits_exactly_16_pavers :
  pavers_crossed = 16 := by
  -- Sorry is used to skip the proof steps
  sorry

end bug_visits_exactly_16_pavers_l222_222260


namespace line_tangent_to_parabola_proof_l222_222340

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end line_tangent_to_parabola_proof_l222_222340


namespace sum_of_solutions_eq_neg_six_l222_222574

theorem sum_of_solutions_eq_neg_six (x r s : ℝ) :
  (81 : ℝ) - 18 * x - 3 * x^2 = 0 →
  (r + s = -6) :=
by
  sorry

end sum_of_solutions_eq_neg_six_l222_222574


namespace cost_large_bulb_l222_222528

def small_bulbs : Nat := 3
def cost_small_bulb : Nat := 8
def total_budget : Nat := 60
def amount_left : Nat := 24

theorem cost_large_bulb (cost_large_bulb : Nat) :
  total_budget - amount_left - small_bulbs * cost_small_bulb = cost_large_bulb →
  cost_large_bulb = 12 := by
  sorry

end cost_large_bulb_l222_222528


namespace percent_gain_correct_l222_222781

theorem percent_gain_correct :
  ∀ (x : ℝ), (900 * x + 50 * (900 * x / 850) - 900 * x) / (900 * x) * 100 = 58.82 :=
by sorry

end percent_gain_correct_l222_222781


namespace bob_speed_l222_222406

theorem bob_speed (v : ℝ) : (∀ v_a : ℝ, v_a > 120 → 30 / v_a < 30 / v - 0.5) → v = 40 :=
by
  sorry

end bob_speed_l222_222406


namespace first_day_price_l222_222634

theorem first_day_price (x n: ℝ) :
  n * x = (n + 100) * (x - 1) ∧ 
  n * x = (n - 200) * (x + 2) → 
  x = 4 :=
by
  sorry

end first_day_price_l222_222634


namespace find_integer_for_prime_l222_222193

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem find_integer_for_prime (n : ℤ) :
  is_prime (4 * n^4 + 1) ↔ n = 1 :=
by
  sorry

end find_integer_for_prime_l222_222193


namespace total_oak_trees_after_planting_l222_222982

-- Definitions based on conditions
def initial_oak_trees : ℕ := 5
def new_oak_trees : ℕ := 4

-- Statement of the problem and solution
theorem total_oak_trees_after_planting : initial_oak_trees + new_oak_trees = 9 := by
  sorry

end total_oak_trees_after_planting_l222_222982


namespace reciprocal_of_neg3_l222_222451

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l222_222451


namespace cos_2alpha_value_l222_222666

noncomputable def cos_double_angle (α : ℝ) : ℝ := Real.cos (2 * α)

theorem cos_2alpha_value (α : ℝ): 
  (∃ a : ℝ, α = Real.arctan (-3) + 2 * a * Real.pi) → cos_double_angle α = -4 / 5 :=
by
  intro h
  sorry

end cos_2alpha_value_l222_222666


namespace weather_conclusion_l222_222767

variables (T C : ℝ) (visitors : ℕ)

def condition1 : Prop :=
  (T ≥ 75.0 ∧ C < 10) → visitors > 100

def condition2 : Prop :=
  visitors ≤ 100

theorem weather_conclusion (h1 : condition1 T C visitors) (h2 : condition2 visitors) : 
  T < 75.0 ∨ C ≥ 10 :=
by 
  sorry

end weather_conclusion_l222_222767


namespace final_position_3000_l222_222202

def initial_position : ℤ × ℤ := (0, 0)
def moves_up_first_minute (pos : ℤ × ℤ) : ℤ × ℤ := (pos.1, pos.2 + 1)

def next_position (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  if n % 4 = 0 then (pos.1 + n, pos.2)
  else if n % 4 = 1 then (pos.1, pos.2 + n)
  else if n % 4 = 2 then (pos.1 - n, pos.2)
  else (pos.1, pos.2 - n)

def final_position (minutes : ℕ) : ℤ × ℤ := sorry

theorem final_position_3000 : final_position 3000 = (0, 27) :=
by {
  -- logic to compute final_position
  sorry -- proof exists here
}

end final_position_3000_l222_222202


namespace strawberry_quality_meets_standard_l222_222721

def acceptable_weight_range (w : ℝ) : Prop :=
  4.97 ≤ w ∧ w ≤ 5.03

theorem strawberry_quality_meets_standard :
  acceptable_weight_range 4.98 :=
by
  sorry

end strawberry_quality_meets_standard_l222_222721


namespace find_x_l222_222675

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l222_222675


namespace wheel_radius_increase_l222_222823

theorem wheel_radius_increase :
  let r := 18
  let distance_AB := 600   -- distance from A to B in miles
  let distance_BA := 582   -- distance from B to A in miles
  let circumference_orig := 2 * Real.pi * r
  let dist_per_rotation_orig := circumference_orig / 63360
  let rotations_orig := distance_AB / dist_per_rotation_orig
  let r' := ((distance_BA * dist_per_rotation_orig * 63360) / (2 * Real.pi * rotations_orig))
  ((r' - r) : ℝ) = 0.34 := by
  sorry

end wheel_radius_increase_l222_222823


namespace capacity_of_new_vessel_is_10_l222_222015

-- Define the conditions
def first_vessel_capacity : ℕ := 2
def first_vessel_concentration : ℚ := 0.25
def second_vessel_capacity : ℕ := 6
def second_vessel_concentration : ℚ := 0.40
def total_liquid_combined : ℕ := 8
def new_mixture_concentration : ℚ := 0.29
def total_alcohol_content : ℚ := (first_vessel_capacity * first_vessel_concentration) + (second_vessel_capacity * second_vessel_concentration)
def desired_vessel_capacity : ℚ := total_alcohol_content / new_mixture_concentration

-- The theorem we want to prove
theorem capacity_of_new_vessel_is_10 : desired_vessel_capacity = 10 := by
  sorry

end capacity_of_new_vessel_is_10_l222_222015


namespace find_second_number_l222_222329

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end find_second_number_l222_222329


namespace minimum_xy_l222_222113

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  x * y ≥ 18 :=
sorry

end minimum_xy_l222_222113


namespace percent_exceed_l222_222453

theorem percent_exceed (x y : ℝ) (h : x = 0.75 * y) : ((y - x) / x) * 100 = 33.33 :=
by
  sorry

end percent_exceed_l222_222453


namespace sequence_property_implies_geometric_progression_l222_222069

theorem sequence_property_implies_geometric_progression {p : ℝ} {a : ℕ → ℝ}
  (h_p : (2 / (Real.sqrt 5 + 1) ≤ p) ∧ (p < 1))
  (h_a : ∀ (e : ℕ → ℤ), (∀ n, (e n = 0) ∨ (e n = 1) ∨ (e n = -1)) →
    (∑' n, (e n) * (p ^ n)) = 0 → (∑' n, (e n) * (a n)) = 0) :
  ∃ c : ℝ, ∀ n, a n = c * (p ^ n) := by
  sorry

end sequence_property_implies_geometric_progression_l222_222069


namespace no_infinite_harmonic_mean_sequence_l222_222354

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), (∀ n, a n = a 0 → False) ∧
                   (∀ i, 1 ≤ i → a i = (2 * a (i - 1) * a (i + 1)) / (a (i - 1) + a (i + 1))) :=
sorry

end no_infinite_harmonic_mean_sequence_l222_222354


namespace necessary_and_sufficient_condition_l222_222914

theorem necessary_and_sufficient_condition (x : ℝ) : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) := 
by
  sorry

end necessary_and_sufficient_condition_l222_222914


namespace age_of_youngest_child_l222_222270

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60) : 
  x = 6 :=
sorry

end age_of_youngest_child_l222_222270


namespace sum_difference_l222_222075

def even_sum (n : ℕ) : ℕ :=
  n * (n + 1)

def odd_sum (n : ℕ) : ℕ :=
  n^2

theorem sum_difference : even_sum 100 - odd_sum 100 = 100 := by
  sorry

end sum_difference_l222_222075


namespace total_pitches_missed_l222_222087

theorem total_pitches_missed (tokens_to_pitches : ℕ → ℕ) 
  (macy_used : ℕ) (piper_used : ℕ) 
  (macy_hits : ℕ) (piper_hits : ℕ) 
  (h1 : tokens_to_pitches 1 = 15) 
  (h_macy_used : macy_used = 11) 
  (h_piper_used : piper_used = 17) 
  (h_macy_hits : macy_hits = 50) 
  (h_piper_hits : piper_hits = 55) :
  let total_pitches := tokens_to_pitches macy_used + tokens_to_pitches piper_used
  let total_hits := macy_hits + piper_hits
  total_pitches - total_hits = 315 :=
by
  sorry

end total_pitches_missed_l222_222087


namespace hikers_rate_l222_222161

-- Define the conditions from the problem
variables (R : ℝ) (time_up time_down : ℝ) (distance_down : ℝ)

-- Conditions given in the problem
axiom condition1 : time_up = 2
axiom condition2 : time_down = 2
axiom condition3 : distance_down = 9
axiom condition4 : (distance_down / time_down) = 1.5 * R

-- The proof goal
theorem hikers_rate (h1 : time_up = 2) 
                    (h2 : time_down = 2) 
                    (h3 : distance_down = 9) 
                    (h4 : distance_down / time_down = 1.5 * R) : R = 3 := 
by 
  sorry

end hikers_rate_l222_222161


namespace symmetrical_character_is_C_l222_222396

-- Definitions of the characters and the concept of symmetry
def is_symmetrical (char: Char): Prop := 
  match char with
  | '中' => True
  | _ => False

-- The options given in the problem
def optionA := '爱'
def optionB := '我'
def optionC := '中'
def optionD := '国'

-- The problem statement: Prove that among the given options, the symmetrical character is 中.
theorem symmetrical_character_is_C : (is_symmetrical optionA = False) ∧ (is_symmetrical optionB = False) ∧ (is_symmetrical optionC = True) ∧ (is_symmetrical optionD = False) :=
by
  sorry

end symmetrical_character_is_C_l222_222396


namespace convert_to_rectangular_form_l222_222926

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end convert_to_rectangular_form_l222_222926


namespace aaron_brothers_l222_222108

theorem aaron_brothers (A : ℕ) (h1 : 6 = 2 * A - 2) : A = 4 :=
by
  sorry

end aaron_brothers_l222_222108


namespace wendy_time_correct_l222_222370

noncomputable section

def bonnie_time : ℝ := 7.80
def wendy_margin : ℝ := 0.25

theorem wendy_time_correct : (bonnie_time - wendy_margin) = 7.55 := by
  sorry

end wendy_time_correct_l222_222370


namespace range_of_g_l222_222529

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by
  sorry

end range_of_g_l222_222529


namespace quadratic_sum_roots_twice_difference_l222_222444

theorem quadratic_sum_roots_twice_difference
  (a b c x₁ x₂ : ℝ)
  (h_eq : a * x₁^2 + b * x₁ + c = 0)
  (h_eq2 : a * x₂^2 + b * x₂ + c = 0)
  (h_sum_twice_diff: x₁ + x₂ = 2 * (x₁ - x₂)) :
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_sum_roots_twice_difference_l222_222444


namespace time_for_B_is_24_days_l222_222779

noncomputable def A_work : ℝ := (1 / 2) / (3 / 4)
noncomputable def B_work : ℝ := 1 -- assume B does 1 unit of work in 1 day
noncomputable def total_work : ℝ := (A_work + B_work) * 18

theorem time_for_B_is_24_days : 
  ((A_work + B_work) * 18) / B_work = 24 := by
  sorry

end time_for_B_is_24_days_l222_222779


namespace line_perpendicular_to_plane_l222_222733

-- Define a structure for vectors in 3D
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define line l with the given direction vector
def direction_vector_l : Vector3D := ⟨1, -1, -2⟩

-- Define plane α with the given normal vector
def normal_vector_alpha : Vector3D := ⟨2, -2, -4⟩

-- Prove that line l is perpendicular to plane α
theorem line_perpendicular_to_plane :
  let a := direction_vector_l
  let b := normal_vector_alpha
  (b.x = 2 * a.x) ∧ (b.y = 2 * a.y) ∧ (b.z = 2 * a.z) → 
  (a.x * b.x + a.y * b.y + a.z * b.z = 0) :=
by
  intro a b h
  sorry

end line_perpendicular_to_plane_l222_222733


namespace area_of_rectangle_inscribed_in_triangle_l222_222703

theorem area_of_rectangle_inscribed_in_triangle :
  ∀ (E F G A B C D : ℝ) (EG altitude_ABCD : ℝ),
    E < F ∧ F < G ∧ A < B ∧ B < C ∧ C < D ∧ A < D ∧ D < G ∧ A < G ∧
    EG = 10 ∧ 
    altitude_ABCD = 7 ∧ 
    B = C ∧ 
    A + D = EG ∧ 
    A + 2 * B = EG →
    ((A * B) = (1225 / 72)) :=
by
  intros E F G A B C D EG altitude_ABCD
  intro h
  sorry

end area_of_rectangle_inscribed_in_triangle_l222_222703


namespace games_played_by_third_player_l222_222598

theorem games_played_by_third_player
    (games_first : ℕ)
    (games_second : ℕ)
    (games_first_eq : games_first = 10)
    (games_second_eq : games_second = 21) :
    ∃ (games_third : ℕ), games_third = 11 := by
  sorry

end games_played_by_third_player_l222_222598


namespace cube_volume_of_surface_area_l222_222372

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l222_222372


namespace solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l222_222873

-- Define the function f(x) and g(x)
def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Define the inequality problem when a = 2
theorem solution_set_for_f_when_a_2 : 
  { x : ℝ | f x 2 ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

-- Prove the range of values for a when f(x) + g(x) ≥ 3
theorem range_of_a_for_f_plus_g_ge_3 : 
  ∀ a : ℝ, (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a :=
by
  sorry

end solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l222_222873


namespace find_pairs_l222_222646

theorem find_pairs (p a : ℕ) (hp_prime : Nat.Prime p) (hp_ge_2 : p ≥ 2) (ha_ge_1 : a ≥ 1) (h_p_ne_a : p ≠ a) :
  (a + p) ∣ (a^2 + p^2) → (a = p ∧ p = p) ∨ (a = p^2 - p ∧ p = p) ∨ (a = 2 * p^2 - p ∧ p = p) :=
by
  sorry

end find_pairs_l222_222646


namespace ab_ac_bc_nonpositive_l222_222594

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end ab_ac_bc_nonpositive_l222_222594


namespace vector_operation_result_l222_222335

variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C O E : V)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = (A - E) :=
by
  sorry

end vector_operation_result_l222_222335


namespace area_of_DEF_l222_222198

variable (t4_area t5_area t6_area : ℝ) (a_DEF : ℝ)

def similar_triangles_area := (t4_area = 1) ∧ (t5_area = 16) ∧ (t6_area = 36)

theorem area_of_DEF 
  (h : similar_triangles_area t4_area t5_area t6_area) :
  a_DEF = 121 := sorry

end area_of_DEF_l222_222198


namespace total_profit_l222_222048

-- Define the variables for the subscriptions and profits
variables {A B C : ℕ} -- Subscription amounts
variables {profit : ℕ} -- Total profit

-- Given conditions
def conditions (A B C : ℕ) (profit : ℕ) :=
  50000 = A + B + C ∧
  A = B + 4000 ∧
  B = C + 5000 ∧
  A * profit = 29400 * 50000

-- Statement of the theorem
theorem total_profit (A B C : ℕ) (profit : ℕ) (h : conditions A B C profit) :
  profit = 70000 :=
sorry

end total_profit_l222_222048


namespace cricket_team_right_handed_players_l222_222174

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (non_throwers : ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_throwers : ℕ := throwers)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers)
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers = non_throwers / 3) :
  total_right_handed = 59 :=
by
  rw [h1, h2] at *
  -- The remaining parts of the proof here are omitted for brevity.
  sorry

end cricket_team_right_handed_players_l222_222174


namespace card_2015_in_box_3_l222_222409

-- Define the pattern function for placing cards
def card_placement (n : ℕ) : ℕ :=
  let cycle_length := 12
  let cycle_pos := (n - 1) % cycle_length + 1
  if cycle_pos ≤ 7 then cycle_pos
  else 14 - cycle_pos

-- Define the theorem to prove the position of the 2015th card
theorem card_2015_in_box_3 : card_placement 2015 = 3 := by
  -- sorry is used to skip the proof
  sorry

end card_2015_in_box_3_l222_222409


namespace water_for_bathing_per_horse_per_day_l222_222744

-- Definitions of the given conditions
def initial_horses : ℕ := 3
def additional_horses : ℕ := 5
def total_horses : ℕ := initial_horses + additional_horses
def drink_water_per_horse_per_day : ℕ := 5
def total_days : ℕ := 28
def total_water_needed : ℕ := 1568

-- The proven statement
theorem water_for_bathing_per_horse_per_day :
  ((total_water_needed - (total_horses * drink_water_per_horse_per_day * total_days)) / (total_horses * total_days)) = 2 :=
by
  sorry

end water_for_bathing_per_horse_per_day_l222_222744


namespace no_real_solution_ineq_l222_222471

theorem no_real_solution_ineq (x : ℝ) (h : x ≠ 5) : ¬ (x^3 - 125) / (x - 5) < 0 := 
by
  sorry

end no_real_solution_ineq_l222_222471


namespace question_1_question_2_l222_222312

def curve_is_ellipse (m : ℝ) : Prop :=
  (3 - m > 0) ∧ (m - 1 > 0) ∧ (3 - m > m - 1)

def domain_is_R (m : ℝ) : Prop :=
  m^2 < (9 / 4)

theorem question_1 (m : ℝ) :
  curve_is_ellipse m → 1 < m ∧ m < 2 :=
sorry

theorem question_2 (m : ℝ) :
  (curve_is_ellipse m ∧ domain_is_R m) → 1 < m ∧ m < (3 / 2) :=
sorry

end question_1_question_2_l222_222312


namespace DogHeight_is_24_l222_222154

-- Define the given conditions as Lean definitions (variables and equations)
variable (CarterHeight DogHeight BettyHeight : ℝ)

-- Assume the conditions given in the problem
axiom h1 : CarterHeight = 2 * DogHeight
axiom h2 : BettyHeight + 12 = CarterHeight
axiom h3 : BettyHeight = 36

-- State the proposition (the height of Carter's dog)
theorem DogHeight_is_24 : DogHeight = 24 :=
by
  -- Proof goes here
  sorry

end DogHeight_is_24_l222_222154


namespace find_the_number_l222_222355

theorem find_the_number (x : ℕ) (h : x * 9999 = 4691110842) : x = 469211 := by
    sorry

end find_the_number_l222_222355


namespace manager_salary_l222_222523

theorem manager_salary (avg_salary_50 : ℕ) (num_employees : ℕ) (increment_new_avg : ℕ)
  (new_avg_salary : ℕ) (total_old_salary : ℕ) (total_new_salary : ℕ) (M : ℕ) :
  avg_salary_50 = 2000 →
  num_employees = 50 →
  increment_new_avg = 250 →
  new_avg_salary = avg_salary_50 + increment_new_avg →
  total_old_salary = num_employees * avg_salary_50 →
  total_new_salary = (num_employees + 1) * new_avg_salary →
  M = total_new_salary - total_old_salary →
  M = 14750 :=
by {
  sorry
}

end manager_salary_l222_222523


namespace find_m_l222_222134

theorem find_m (m : ℝ) : (∀ x > 0, x^2 - 2 * (m^2 + m + 1) * Real.log x ≥ 1) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end find_m_l222_222134


namespace smallest_n_square_area_l222_222399

theorem smallest_n_square_area (n : ℕ) (n_positive : 0 < n) : ∃ k : ℕ, 14 * n = k^2 ↔ n = 14 := 
sorry

end smallest_n_square_area_l222_222399


namespace right_triangle_area_l222_222345

theorem right_triangle_area (a b c : ℝ) (h1 : a + b + c = 90) (h2 : a^2 + b^2 + c^2 = 3362) (h3 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 180 :=
by
  sorry

end right_triangle_area_l222_222345


namespace max_marks_obtainable_l222_222191

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end max_marks_obtainable_l222_222191


namespace solution_inequality_l222_222426

theorem solution_inequality (θ x : ℝ)
  (h : |x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) : 
  -1 ≤ x ∧ x ≤ -Real.cos (2 * θ) :=
sorry

end solution_inequality_l222_222426


namespace find_salary_J_l222_222062

variables {J F M A May : ℝ}
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (h3 : May = 6500)

theorem find_salary_J : J = 5700 :=
by
  sorry

end find_salary_J_l222_222062


namespace find_a10_l222_222414

def arith_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variables (a : ℕ → ℚ) (d : ℚ)

-- Conditions
def condition1 := a 4 + a 11 = 16  -- translates to a_5 + a_12 = 16
def condition2 := a 6 = 1  -- translates to a_7 = 1
def condition3 := arith_seq a d  -- a is an arithmetic sequence with common difference d

-- The main theorem
theorem find_a10 : condition1 a ∧ condition2 a ∧ condition3 a d → a 9 = 15 := sorry

end find_a10_l222_222414


namespace combined_resistance_parallel_l222_222671

theorem combined_resistance_parallel (R1 R2 R3 R : ℝ)
  (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6)
  (h4 : 1/R = 1/R1 + 1/R2 + 1/R3) :
  R = 15/13 := 
by
  sorry

end combined_resistance_parallel_l222_222671


namespace minimum_f_value_g_ge_f_implies_a_ge_4_l222_222443

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_f_value : (∃ x : ℝ, f x = 2 / Real.exp 1) :=
  sorry

theorem g_ge_f_implies_a_ge_4 (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ g x a) → a ≥ 4 :=
  sorry

end minimum_f_value_g_ge_f_implies_a_ge_4_l222_222443


namespace geometric_Sn_over_n_sum_first_n_terms_l222_222397

-- The first problem statement translation to Lean 4
theorem geometric_Sn_over_n (a S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n+1) = (n + 2) * S n) :
  ∃ r : ℕ, (r = 2 ∧ ∃ b : ℕ, b = 1 ∧ 
    ∀ n : ℕ, 0 < n → (S (n + 1)) / (n + 1) = r * (S n) / n) := 
sorry

-- The second problem statement translation to Lean 4
theorem sum_first_n_terms (a S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 2) * S n)
  (h3 : ∀ n : ℕ, S n = n * 2^(n - 1)) :
  ∀ n : ℕ, T n = (n - 1) * 2^n + 1 :=
sorry

end geometric_Sn_over_n_sum_first_n_terms_l222_222397


namespace rectangle_perimeter_at_least_l222_222308

theorem rectangle_perimeter_at_least (m : ℕ) (m_pos : 0 < m) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a * b ≥ 1 / (m * m) ∧ 2 * (a + b) ≥ 4 / m) := sorry

end rectangle_perimeter_at_least_l222_222308


namespace find_number_l222_222784

-- Define the conditions
def number_times_x_eq_165 (number x : ℕ) : Prop :=
  number * x = 165

def x_eq_11 (x : ℕ) : Prop :=
  x = 11

-- The proof problem statement
theorem find_number (number x : ℕ) (h1 : number_times_x_eq_165 number x) (h2 : x_eq_11 x) : number = 15 :=
by
  sorry

end find_number_l222_222784


namespace factorization_problem_l222_222502

theorem factorization_problem (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 1 := 
by
  sorry

end factorization_problem_l222_222502


namespace a4_equals_8_l222_222546

variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {n : ℕ}

-- Defining the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a (n + 1) = a n * r

-- Given conditions as hypotheses
variable (h_geometric : geometric_sequence a r)
variable (h_root_2 : a 2 * a 6 = 64)
variable (h_roots_eq : ∀ x, x^2 - 34 * x + 64 = 0 → (x = a 2 ∨ x = a 6))

-- The statement to prove
theorem a4_equals_8 : a 4 = 8 :=
by
  sorry

end a4_equals_8_l222_222546


namespace percentage_cleared_all_sections_l222_222544

def total_candidates : ℝ := 1200
def cleared_none : ℝ := 0.05 * total_candidates
def cleared_one_section : ℝ := 0.25 * total_candidates
def cleared_four_sections : ℝ := 0.20 * total_candidates
def cleared_two_sections : ℝ := 0.245 * total_candidates
def cleared_three_sections : ℝ := 300

-- Let x be the percentage of candidates who cleared all sections
def cleared_all_sections (x: ℝ) : Prop :=
  let total_cleared := (cleared_none + 
                        cleared_one_section + 
                        cleared_four_sections + 
                        cleared_two_sections + 
                        cleared_three_sections + 
                        x * total_candidates / 100)
  total_cleared = total_candidates

theorem percentage_cleared_all_sections :
  ∃ x, cleared_all_sections x ∧ x = 0.5 :=
by
  sorry

end percentage_cleared_all_sections_l222_222544


namespace initial_participants_l222_222522

theorem initial_participants (p : ℕ) (h1 : 0.6 * p = 0.6 * (p : ℝ)) (h2 : ∀ (n : ℕ), n = 4 * m → 30 = (2 / 5) * n * (1 / 4)) :
  p = 300 :=
by sorry

end initial_participants_l222_222522


namespace solution_volume_l222_222645

theorem solution_volume (concentration volume_acid volume_solution : ℝ) 
  (h_concentration : concentration = 0.25) 
  (h_acid : volume_acid = 2.5) 
  (h_formula : concentration = volume_acid / volume_solution) : 
  volume_solution = 10 := 
by
  sorry

end solution_volume_l222_222645


namespace sum_of_first_15_terms_l222_222990

variable (a d : ℕ)

def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_15_terms (h : nth_term 4 + nth_term 12 = 16) : sum_of_first_n_terms 15 = 120 :=
by
  sorry

end sum_of_first_15_terms_l222_222990


namespace percentage_of_girls_after_change_l222_222869

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change_l222_222869


namespace total_miles_driven_l222_222162

-- Given constants and conditions
def city_mpg : ℝ := 30
def highway_mpg : ℝ := 37
def total_gallons : ℝ := 11
def highway_extra_miles : ℕ := 5

-- Variable for the number of city miles
variable (x : ℝ)

-- Conditions encapsulated in a theorem statement
theorem total_miles_driven:
  (x / city_mpg) + ((x + highway_extra_miles) / highway_mpg) = total_gallons →
  x + (x + highway_extra_miles) = 365 :=
by
  sorry

end total_miles_driven_l222_222162


namespace huahuan_initial_cards_l222_222539

theorem huahuan_initial_cards
  (a b c : ℕ) -- let a, b, c be the initial number of cards Huahuan, Yingying, and Nini have
  (total : a + b + c = 2712)
  (condition_after_50_rounds : ∃ d, b = a + d ∧ c = a + 2 * d) -- after 50 rounds, form an arithmetic sequence
  : a = 754 := sorry

end huahuan_initial_cards_l222_222539


namespace mutually_exclusive_event_l222_222952

theorem mutually_exclusive_event (A B C D: Prop) 
  (h_A: ¬ (A ∧ (¬D)) ∧ ¬ ¬ D)
  (h_B: ¬ (B ∧ (¬D)) ∧ ¬ ¬ D)
  (h_C: ¬ (C ∧ (¬D)) ∧ ¬ ¬ D)
  (h_D: ¬ (D ∧ (¬D)) ∧ ¬ ¬ D) :
  D :=
sorry

end mutually_exclusive_event_l222_222952


namespace part1_part2_l222_222306

-- Define the universal set U as real numbers ℝ
def U : Set ℝ := Set.univ

-- Define Set A
def A (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1 }

-- Define Set B
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0 }

-- Part 1: Prove A ∪ B when a = 4
theorem part1 : A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} :=
sorry

-- Part 2: Prove the range of values for a given A ∩ B = A
theorem part2 (a : ℝ) (h : A a ∩ B = A a) : a ≥ 5 ∨ a ≤ 0 :=
sorry

end part1_part2_l222_222306


namespace exists_subset_sum_divisible_by_2n_l222_222310

open BigOperators

theorem exists_subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℤ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_interval : ∀ i : Fin n, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 :=
sorry

end exists_subset_sum_divisible_by_2n_l222_222310


namespace complex_quadrant_l222_222061

theorem complex_quadrant (θ : ℝ) (hθ : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) :
  let z := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l222_222061


namespace integer_solutions_of_inequality_system_l222_222495

theorem integer_solutions_of_inequality_system :
  { x : ℤ | (3 * x - 2) / 3 ≥ 1 ∧ 3 * x + 5 > 4 * x - 2 } = {2, 3, 4, 5, 6} :=
by {
  sorry
}

end integer_solutions_of_inequality_system_l222_222495


namespace intersection_A_B_l222_222820

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersection_A_B :
  { x | x^2 - 4*x - 5 < 0 } ∩ { x | -2 < x ∧ x < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  -- Here would be the proof, but we use sorry to skip it
  sorry

end intersection_A_B_l222_222820


namespace product_neg_six_l222_222274

theorem product_neg_six (m b : ℝ)
  (h1 : m = 2)
  (h2 : b = -3) : m * b < -3 := by
-- Proof skipped
sorry

end product_neg_six_l222_222274


namespace largest_y_coordinate_l222_222604

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 49) + ((y - 3)^2 / 25) = 0) : y = 3 :=
sorry

end largest_y_coordinate_l222_222604


namespace broken_glass_pieces_l222_222093

theorem broken_glass_pieces (x : ℕ) 
    (total_pieces : ℕ := 100) 
    (safe_fee : ℕ := 3) 
    (compensation : ℕ := 5) 
    (total_fee : ℕ := 260) 
    (h : safe_fee * (total_pieces - x) - compensation * x = total_fee) : x = 5 := by
  sorry

end broken_glass_pieces_l222_222093


namespace find_quotient_l222_222879

theorem find_quotient
    (dividend divisor remainder : ℕ)
    (h1 : dividend = 136)
    (h2 : divisor = 15)
    (h3 : remainder = 1)
    (h4 : dividend = divisor * quotient + remainder) :
    quotient = 9 :=
by
  sorry

end find_quotient_l222_222879


namespace range_of_a_l222_222485

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1) := 
sorry

end range_of_a_l222_222485


namespace line_parabola_midpoint_l222_222560

theorem line_parabola_midpoint (a b : ℝ) 
  (r s : ℝ) 
  (intersects_parabola : ∀ x, x = r ∨ x = s → ax + b = x^2)
  (midpoint_cond : (r + s) / 2 = 5 ∧ (r^2 + s^2) / 2 = 101) :
  a + b = -41 :=
sorry

end line_parabola_midpoint_l222_222560


namespace new_average_is_15_l222_222837

-- Definitions corresponding to the conditions
def avg_10_consecutive (seq : List ℤ) : Prop :=
  seq.length = 10 ∧ seq.sum = 200

def new_seq (seq : List ℤ) : List ℤ :=
  List.mapIdx (λ i x => x - ↑(9 - i)) seq

-- Statement of the proof problem
theorem new_average_is_15
  (seq : List ℤ)
  (h_seq : avg_10_consecutive seq) :
  (new_seq seq).sum = 150 := sorry

end new_average_is_15_l222_222837


namespace copy_pages_l222_222637

theorem copy_pages (cost_per_5_pages : ℝ) (total_dollars : ℝ) : 
  (cost_per_5_pages = 10) → (total_dollars = 15) → (15 * 100 / 10 * 5 = 750) :=
by
  intros
  sorry

end copy_pages_l222_222637


namespace probability_one_defective_l222_222757

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l222_222757


namespace abs_val_equality_l222_222965

theorem abs_val_equality (m : ℝ) (h : |m| = |(-3 : ℝ)|) : m = 3 ∨ m = -3 :=
sorry

end abs_val_equality_l222_222965


namespace slope_of_BC_l222_222064

theorem slope_of_BC
  (h₁ : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1)
  (h₂ : ∀ A : ℝ × ℝ, A = (2, 1))
  (h₃ : ∀ k₁ k₂ : ℝ, k₁ + k₂ = 0) :
  ∃ k : ℝ, k = 1 / 2 :=
by
  sorry

end slope_of_BC_l222_222064


namespace Sam_drinks_l222_222855

theorem Sam_drinks (juice_don : ℚ) (fraction_sam : ℚ) 
  (h1 : juice_don = 3 / 7) (h2 : fraction_sam = 4 / 5) : 
  (fraction_sam * juice_don = 12 / 35) :=
by
  sorry

end Sam_drinks_l222_222855


namespace perpendicular_lines_a_value_l222_222711

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (x + a * y - a = 0) ∧ (a * x - (2 * a - 3) * y - 1 = 0) → x ≠ y) →
  a = 0 ∨ a = 2 :=
sorry

end perpendicular_lines_a_value_l222_222711


namespace class_A_students_l222_222695

variable (A B : ℕ)

theorem class_A_students 
    (h1 : A = (5 * B) / 7)
    (h2 : A + 3 = (4 * (B - 3)) / 5) :
    A = 45 :=
sorry

end class_A_students_l222_222695


namespace applesauce_ratio_is_half_l222_222928

-- Define the weights and number of pies
def total_weight : ℕ := 120
def weight_per_pie : ℕ := 4
def num_pies : ℕ := 15

-- Calculate weights used for pies and applesauce
def weight_for_pies : ℕ := num_pies * weight_per_pie
def weight_for_applesauce : ℕ := total_weight - weight_for_pies

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to prove
theorem applesauce_ratio_is_half :
  ratio weight_for_applesauce total_weight = 1 / 2 :=
by
  -- The proof goes here
  sorry

end applesauce_ratio_is_half_l222_222928


namespace sufficient_condition_range_k_l222_222774

theorem sufficient_condition_range_k {x k : ℝ} (h : ∀ x, x > k → (3 / (x + 1) < 1)) : k ≥ 2 :=
sorry

end sufficient_condition_range_k_l222_222774


namespace value_of_m_l222_222139

theorem value_of_m (a a1 a2 a3 a4 a5 a6 m : ℝ) (x : ℝ)
  (h1 : (1 + m * x)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  (m = 1 ∨ m = -3) :=
sorry

end value_of_m_l222_222139


namespace particular_solution_satisfies_initial_conditions_l222_222706

noncomputable def x_solution : ℝ → ℝ := λ t => (-4/3) * Real.exp t + (7/3) * Real.exp (-2 * t)
noncomputable def y_solution : ℝ → ℝ := λ t => (-1/3) * Real.exp t + (7/3) * Real.exp (-2 * t)

def x_prime (x y : ℝ) := 2 * x - 4 * y
def y_prime (x y : ℝ) := x - 3 * y

theorem particular_solution_satisfies_initial_conditions :
  (∀ t, deriv x_solution t = x_prime (x_solution t) (y_solution t)) ∧
  (∀ t, deriv y_solution t = y_prime (x_solution t) (y_solution t)) ∧
  (x_solution 0 = 1) ∧
  (y_solution 0 = 2) := by
  sorry

end particular_solution_satisfies_initial_conditions_l222_222706


namespace f_x_plus_1_even_f_x_plus_3_odd_l222_222377

variable (R : Type) [CommRing R]

variable (f : R → R)

-- Conditions
axiom condition1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom condition2 : ∀ x : R, f (x - 2) + f (-x) = 0

-- Prove that f(x + 1) is an even function
theorem f_x_plus_1_even (x : R) : f (x + 1) = f (-(x + 1)) :=
by sorry

-- Prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd (x : R) : f (x + 3) = - f (-(x + 3)) :=
by sorry

end f_x_plus_1_even_f_x_plus_3_odd_l222_222377


namespace expression_value_l222_222305

theorem expression_value (a b : ℚ) (h : a + 2 * b = 0) : 
  abs (a / |b| - 1) + abs (|a| / b - 2) + abs (|a / b| - 3) = 4 :=
sorry

end expression_value_l222_222305


namespace g_zero_l222_222532

variable (f g h : Polynomial ℤ) -- Assume f, g, h are polynomials over the integers

-- Condition: h(x) = f(x) * g(x)
axiom h_def : h = f * g

-- Condition: The constant term of f(x) is 2
axiom f_const : f.coeff 0 = 2

-- Condition: The constant term of h(x) is -6
axiom h_const : h.coeff 0 = -6

-- Proof statement that g(0) = -3
theorem g_zero : g.coeff 0 = -3 := by
  sorry

end g_zero_l222_222532


namespace distance_between_points_A_and_B_is_240_l222_222382

noncomputable def distance_between_A_and_B (x y : ℕ) : ℕ := 6 * x * 2

theorem distance_between_points_A_and_B_is_240 (x y : ℕ)
  (h1 : 6 * x = 6 * y)
  (h2 : 5 * (x + 4) = 6 * y) :
  distance_between_A_and_B x y = 240 := by
  sorry

end distance_between_points_A_and_B_is_240_l222_222382


namespace difference_of_squares_l222_222478

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l222_222478


namespace range_of_b_l222_222679

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f x b ≥ 0) ↔ b ≤ -1 :=
by sorry

end range_of_b_l222_222679


namespace simplify_expression_l222_222854

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a)^(2 : ℝ) :=
by sorry

end simplify_expression_l222_222854


namespace total_bags_sold_l222_222680

theorem total_bags_sold (first_week second_week third_week fourth_week total : ℕ) 
  (h1 : first_week = 15) 
  (h2 : second_week = 3 * first_week) 
  (h3 : third_week = 20) 
  (h4 : fourth_week = 20) 
  (h5 : total = first_week + second_week + third_week + fourth_week) : 
  total = 100 := 
sorry

end total_bags_sold_l222_222680


namespace a_beats_b_by_7_seconds_l222_222596

/-
  Given:
  1. A's time to finish the race is 28 seconds (tA = 28).
  2. The race distance is 280 meters (d = 280).
  3. A beats B by 56 meters (dA - dB = 56).
  
  Prove:
  A beats B by 7 seconds (tB - tA = 7).
-/

theorem a_beats_b_by_7_seconds 
  (tA : ℕ) (d : ℕ) (speedA : ℕ) (dB : ℕ) (tB : ℕ) 
  (h1 : tA = 28) 
  (h2 : d = 280) 
  (h3 : d - dB = 56) 
  (h4 : speedA = d / tA) 
  (h5 : dB = speedA * tA) 
  (h6 : tB = d / speedA) :
  tB - tA = 7 := 
sorry

end a_beats_b_by_7_seconds_l222_222596


namespace g_sum_zero_l222_222049

def g (x : ℝ) : ℝ := x^2 - 2013 * x

theorem g_sum_zero (a b : ℝ) (h₁ : g a = g b) (h₂ : a ≠ b) : g (a + b) = 0 :=
sorry

end g_sum_zero_l222_222049


namespace largest_integer_base7_four_digits_l222_222455

theorem largest_integer_base7_four_digits :
  ∃ M : ℕ, (∀ m : ℕ, 7^3 ≤ m^2 ∧ m^2 < 7^4 → m ≤ M) ∧ M = 48 :=
sorry

end largest_integer_base7_four_digits_l222_222455


namespace marks_per_correct_answer_l222_222755

-- Definitions based on the conditions
def total_questions : ℕ := 60
def total_marks : ℕ := 160
def correct_questions : ℕ := 44
def wrong_mark_loss : ℕ := 1

-- The number of correct answers multiplies the marks per correct answer,
-- minus the loss from wrong answers, equals the total marks.
theorem marks_per_correct_answer (x : ℕ) :
  correct_questions * x - (total_questions - correct_questions) * wrong_mark_loss = total_marks → x = 4 := by
sorry

end marks_per_correct_answer_l222_222755


namespace integer_triangle_cosines_rational_l222_222836

theorem integer_triangle_cosines_rational (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ∃ (cos_α cos_β cos_γ : ℚ), 
    cos_γ = (a^2 + b^2 - c^2) / (2 * a * b) ∧
    cos_β = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    cos_α = (b^2 + c^2 - a^2) / (2 * b * c) :=
by
  sorry

end integer_triangle_cosines_rational_l222_222836


namespace clock_angle_at_3_45_l222_222617

theorem clock_angle_at_3_45 :
  let minute_angle_rate := 6.0 -- degrees per minute
  let hour_angle_rate := 0.5  -- degrees per minute
  let initial_angle := 90.0   -- degrees at 3:00
  let minutes_passed := 45.0  -- minutes since 3:00
  let angle_difference_rate := minute_angle_rate - hour_angle_rate
  let angle_change := angle_difference_rate * minutes_passed
  let final_angle := initial_angle - angle_change
  let smaller_angle := if final_angle < 0 then 360.0 + final_angle else final_angle
  smaller_angle = 157.5 :=
by
  sorry

end clock_angle_at_3_45_l222_222617


namespace solve_abs_inequality_l222_222484

theorem solve_abs_inequality (x : ℝ) :
  abs ((6 - 2 * x + 5) / 4) < 3 ↔ -1 / 2 < x ∧ x < 23 / 2 := 
sorry

end solve_abs_inequality_l222_222484


namespace proof_inequality_l222_222192

theorem proof_inequality (p q r : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hpr_lt_qr : p * r < q * r) : 
  p < q :=
by 
  sorry

end proof_inequality_l222_222192


namespace circle_center_radius_sum_l222_222918

-- We define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 14 * x + y^2 + 16 * y + 100 = 0

-- We need to find that the center and radius satisfy a specific relationship
theorem circle_center_radius_sum :
  let a' := 7
  let b' := -8
  let r' := Real.sqrt 13
  a' + b' + r' = -1 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l222_222918


namespace find_k_l222_222098

theorem find_k (k : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, 3 * x - k * y + c = 0) ∧ (∀ x y : ℝ, k * x + y + 1 = 0 → 3 * k + (-k) = 0) → k = 0 :=
by
  sorry

end find_k_l222_222098


namespace div_neg_21_by_3_l222_222408

theorem div_neg_21_by_3 : (-21 : ℤ) / 3 = -7 :=
by sorry

end div_neg_21_by_3_l222_222408


namespace divide_square_into_smaller_squares_l222_222018

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end divide_square_into_smaller_squares_l222_222018


namespace fabric_ratio_wednesday_tuesday_l222_222208

theorem fabric_ratio_wednesday_tuesday :
  let fabric_monday := 20
  let fabric_tuesday := 2 * fabric_monday
  let cost_per_yard := 2
  let total_earnings := 140
  let earnings_monday := fabric_monday * cost_per_yard
  let earnings_tuesday := fabric_tuesday * cost_per_yard
  let earnings_wednesday := total_earnings - (earnings_monday + earnings_tuesday)
  let fabric_wednesday := earnings_wednesday / cost_per_yard
  (fabric_wednesday / fabric_tuesday = 1 / 4) :=
by
  sorry

end fabric_ratio_wednesday_tuesday_l222_222208


namespace quadruples_positive_integers_l222_222652

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l222_222652


namespace sum_of_consecutive_even_numbers_l222_222601

theorem sum_of_consecutive_even_numbers (n : ℕ) (h : (n + 2)^2 - n^2 = 84) :
  n + (n + 2) = 42 :=
sorry

end sum_of_consecutive_even_numbers_l222_222601


namespace evaluate_expression_l222_222616

theorem evaluate_expression :
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (3 / 4) :=
sorry

end evaluate_expression_l222_222616


namespace find_b_for_continuity_at_2_l222_222960

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if h : x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity_at_2 (b : ℝ) : (∀ x, f x b = if x ≤ 2 then 4 * x^2 + 5 else b * x + 3) ∧ 
  (f 2 b = 21) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x b - f 2 b| < ε) → 
  b = 9 :=
by
  sorry

end find_b_for_continuity_at_2_l222_222960


namespace simplify_and_evaluate_expression_l222_222891

theorem simplify_and_evaluate_expression (x y : ℝ) (h_x : x = -2) (h_y : y = 1) :
  (((2 * x - (1/2) * y)^2 - ((-y + 2 * x) * (2 * x + y)) + y * (x^2 * y - (5/4) * y)) / x) = -4 :=
by
  sorry

end simplify_and_evaluate_expression_l222_222891


namespace shaded_area_ratio_l222_222438

noncomputable def ratio_of_shaded_area_to_circle_area (AB r : ℝ) : ℝ :=
  let AC := r
  let CB := 2 * r
  let radius_semicircle_AB := 3 * r / 2
  let area_semicircle_AB := (1 / 2) * (Real.pi * (radius_semicircle_AB ^ 2))
  let radius_semicircle_AC := r / 2
  let area_semicircle_AC := (1 / 2) * (Real.pi * (radius_semicircle_AC ^ 2))
  let radius_semicircle_CB := r
  let area_semicircle_CB := (1 / 2) * (Real.pi * (radius_semicircle_CB ^ 2))
  let total_area_semicircles := area_semicircle_AB + area_semicircle_AC + area_semicircle_CB
  let non_overlapping_area_semicircle_AB := area_semicircle_AB - (area_semicircle_AC + area_semicircle_CB)
  let shaded_area := non_overlapping_area_semicircle_AB
  let area_circle_CD := Real.pi * (r ^ 2)
  shaded_area / area_circle_CD

theorem shaded_area_ratio (AB r : ℝ) : ratio_of_shaded_area_to_circle_area AB r = 1 / 4 :=
by
  sorry

end shaded_area_ratio_l222_222438


namespace gondor_total_earnings_l222_222760

-- Defining the earnings from repairing a phone and a laptop
def phone_earning : ℕ := 10
def laptop_earning : ℕ := 20

-- Defining the number of repairs
def monday_phone_repairs : ℕ := 3
def tuesday_phone_repairs : ℕ := 5
def wednesday_laptop_repairs : ℕ := 2
def thursday_laptop_repairs : ℕ := 4

-- Calculating total earnings
def monday_earnings : ℕ := monday_phone_repairs * phone_earning
def tuesday_earnings : ℕ := tuesday_phone_repairs * phone_earning
def wednesday_earnings : ℕ := wednesday_laptop_repairs * laptop_earning
def thursday_earnings : ℕ := thursday_laptop_repairs * laptop_earning

def total_earnings : ℕ := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The theorem to be proven
theorem gondor_total_earnings : total_earnings = 200 := by
  sorry

end gondor_total_earnings_l222_222760


namespace estimated_height_is_644_l222_222296

noncomputable def height_of_second_building : ℝ := 100
noncomputable def height_of_first_building : ℝ := 0.8 * height_of_second_building
noncomputable def height_of_third_building : ℝ := (height_of_first_building + height_of_second_building) - 20
noncomputable def height_of_fourth_building : ℝ := 1.15 * height_of_third_building
noncomputable def height_of_fifth_building : ℝ := 2 * |height_of_second_building - height_of_third_building|
noncomputable def total_estimated_height : ℝ := height_of_first_building + height_of_second_building + height_of_third_building + height_of_fourth_building + height_of_fifth_building

theorem estimated_height_is_644 : total_estimated_height = 644 := by
  sorry

end estimated_height_is_644_l222_222296


namespace max_x1_x2_squares_l222_222842

noncomputable def x1_x2_squares_eq_max : Prop :=
  ∃ k : ℝ, (∀ x1 x2 : ℝ, (x1 + x2 = k - 2) ∧ (x1 * x2 = k^2 + 3 * k + 5) → x1^2 + x2^2 = 18)

theorem max_x1_x2_squares : x1_x2_squares_eq_max :=
by sorry

end max_x1_x2_squares_l222_222842


namespace symmetric_about_y_l222_222512

theorem symmetric_about_y (m n : ℤ) (h1 : 2 * n - m = -14) (h2 : m = 4) : (m + n) ^ 2023 = -1 := by
  sorry

end symmetric_about_y_l222_222512


namespace range_of_a_for_domain_of_f_l222_222224

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-5 / (a * x^2 + a * x - 3))

theorem range_of_a_for_domain_of_f :
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x - 3 < 0} = {a : ℝ | -12 < a ∧ a ≤ 0} :=
by
  sorry

end range_of_a_for_domain_of_f_l222_222224


namespace most_colored_pencils_l222_222759

theorem most_colored_pencils (total red blue yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - (red + blue)) :
  blue = 12 :=
by
  sorry

end most_colored_pencils_l222_222759


namespace pony_jeans_discount_rate_l222_222090

noncomputable def fox_price : ℝ := 15
noncomputable def pony_price : ℝ := 18

-- Define the conditions
def total_savings (F P : ℝ) : Prop :=
  3 * (F / 100 * fox_price) + 2 * (P / 100 * pony_price) = 9

def discount_sum (F P : ℝ) : Prop :=
  F + P = 22

-- Main statement to be proven
theorem pony_jeans_discount_rate (F P : ℝ) (h1 : total_savings F P) (h2 : discount_sum F P) : P = 10 :=
by
  -- Proof goes here
  sorry

end pony_jeans_discount_rate_l222_222090


namespace perpendicular_lines_a_eq_2_l222_222469

/-- Given two lines, ax + 2y + 2 = 0 and x - y - 2 = 0, prove that if these lines are perpendicular, then a = 2. -/
theorem perpendicular_lines_a_eq_2 {a : ℝ} :
  (∃ a, (a ≠ 0)) → (∃ x y, ((ax + 2*y + 2 = 0) ∧ (x - y - 2 = 0)) → - (a / 2) * 1 = -1) → a = 2 :=
by
  sorry

end perpendicular_lines_a_eq_2_l222_222469


namespace lines_intersect_l222_222422

-- Condition definitions
def line1 (t : ℝ) : ℝ × ℝ :=
  ⟨2 + t * -1, 3 + t * 5⟩

def line2 (u : ℝ) : ℝ × ℝ :=
  ⟨u * -1, 7 + u * 4⟩

-- Theorem statement
theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (6, -17) :=
by
  sorry

end lines_intersect_l222_222422


namespace isosceles_with_base_c_l222_222350

theorem isosceles_with_base_c (a b c: ℝ) (h: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (triangle_rel: 1/a - 1/b + 1/c = 1/(a - b + c)) : a = c ∨ b = c :=
sorry

end isosceles_with_base_c_l222_222350


namespace estimate_value_l222_222079

theorem estimate_value : 1 < (3 - Real.sqrt 3) ∧ (3 - Real.sqrt 3) < 2 :=
by
  have h₁ : Real.sqrt 18 = 3 * Real.sqrt 2 :=
    by sorry
  have h₂ : Real.sqrt 6 = Real.sqrt 3 * Real.sqrt 2 :=
    by sorry
  have h₃ : (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 = (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 :=
    by sorry
  have h₄ : (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 = 3 - Real.sqrt 3 :=
    by sorry
  have h₅ : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 :=
    by sorry
  sorry

end estimate_value_l222_222079


namespace graph_paper_squares_below_line_l222_222347

theorem graph_paper_squares_below_line
  (h : ∀ (x y : ℕ), 12 * x + 247 * y = 2976)
  (square_size : ℕ) 
  (xs : ℕ) (ys : ℕ)
  (line_eq : ∀ (x y : ℕ), y = 247 * x / 12)
  (n_squares : ℕ) :
  n_squares = 1358
  := by
    sorry

end graph_paper_squares_below_line_l222_222347


namespace pow_2023_eq_one_or_neg_one_l222_222111

theorem pow_2023_eq_one_or_neg_one (x : ℂ) (h : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) : 
  x^2023 = 1 ∨ x^2023 = -1 := 
by 
{
  sorry
}

end pow_2023_eq_one_or_neg_one_l222_222111


namespace sin_330_value_l222_222226

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l222_222226


namespace quintuplets_babies_l222_222493

theorem quintuplets_babies (t r q : ℕ) (h1 : r = 6 * q)
  (h2 : t = 2 * r)
  (h3 : 2 * t + 3 * r + 5 * q = 1500) :
  5 * q = 160 :=
by
  sorry

end quintuplets_babies_l222_222493


namespace smallestC_l222_222936

def isValidFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  f 1 = 1 ∧
  (∀ x y, 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1 → f x + f y ≤ f (x + y))

theorem smallestC (f : ℝ → ℝ) (h : isValidFunction f) : ∃ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ c * x) ∧
  (∀ d, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ d * x) → 2 ≤ d) :=
sorry

end smallestC_l222_222936


namespace three_pow_1234_mod_5_l222_222659

theorem three_pow_1234_mod_5 : (3^1234) % 5 = 4 := 
by 
  have h1 : 3^4 % 5 = 1 := by norm_num
  sorry

end three_pow_1234_mod_5_l222_222659


namespace sum_of_two_numbers_l222_222578

variables {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l222_222578


namespace pyramid_base_length_l222_222301

theorem pyramid_base_length (A s h : ℝ): A = 120 ∧ h = 40 ∧ (A = 1/2 * s * h) → s = 6 := 
by
  sorry

end pyramid_base_length_l222_222301


namespace tan_to_trig_identity_l222_222336

theorem tan_to_trig_identity (α : ℝ) (h : Real.tan α = 3) : (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
by
  sorry

end tan_to_trig_identity_l222_222336


namespace speed_of_first_boy_proof_l222_222101

noncomputable def speed_of_first_boy := 5.9

theorem speed_of_first_boy_proof :
  ∀ (x : ℝ) (t : ℝ) (d : ℝ),
    (d = x * t) → (d = (x - 5.6) * 35) →
    d = 10.5 →
    t = 35 →
    x = 5.9 := 
by
  intros x t d h1 h2 h3 h4
  sorry

end speed_of_first_boy_proof_l222_222101


namespace ekon_uma_diff_l222_222895

-- Definitions based on conditions
def total_videos := 411
def kelsey_videos := 160
def ekon_kelsey_diff := 43

-- Definitions derived from conditions
def ekon_videos := kelsey_videos - ekon_kelsey_diff
def uma_videos (E : ℕ) := total_videos - kelsey_videos - E

-- The Lean problem statement
theorem ekon_uma_diff : 
  uma_videos ekon_videos - ekon_videos = 17 := 
by 
  sorry

end ekon_uma_diff_l222_222895


namespace germs_left_percentage_l222_222287

-- Defining the conditions
def first_spray_kill_percentage : ℝ := 0.50
def second_spray_kill_percentage : ℝ := 0.25
def overlap_percentage : ℝ := 0.05
def total_kill_percentage : ℝ := first_spray_kill_percentage + second_spray_kill_percentage - overlap_percentage

-- The statement to be proved
theorem germs_left_percentage :
  1 - total_kill_percentage = 0.30 :=
by
  -- The proof would go here.
  sorry

end germs_left_percentage_l222_222287


namespace max_value_5x_minus_25x_l222_222031

open Real

theorem max_value_5x_minus_25x : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 5^x) → (y - y^2) ≤ 1 / 4 := 
by 
  sorry

end max_value_5x_minus_25x_l222_222031


namespace usable_field_area_l222_222194

open Float

def breadth_of_field (P : ℕ) (extra_length : ℕ) := (P / 2 - extra_length) / 2

def length_of_field (b : ℕ) (extra_length : ℕ) := b + extra_length

def effective_length (l : ℕ) (obstacle_length : ℕ) := l - obstacle_length

def effective_breadth (b : ℕ) (obstacle_breadth : ℕ) := b - obstacle_breadth

def field_area (length : ℕ) (breadth : ℕ) := length * breadth 

theorem usable_field_area : 
  ∀ (P extra_length obstacle_length obstacle_breadth : ℕ), 
  P = 540 -> extra_length = 30 -> obstacle_length = 10 -> obstacle_breadth = 5 -> 
  field_area (effective_length (length_of_field (breadth_of_field P extra_length) extra_length) obstacle_length) (effective_breadth (breadth_of_field P extra_length) obstacle_breadth) = 16100 := by
  sorry

end usable_field_area_l222_222194


namespace problem_divisibility_l222_222348

theorem problem_divisibility (k : ℕ) (hk : k > 1) (p : ℕ) (hp : p = 6 * k + 1) (hprime : Prime p) 
  (m : ℕ) (hm : m = 2^p - 1) : 
  127 * m ∣ 2^(m - 1) - 1 := 
sorry

end problem_divisibility_l222_222348


namespace each_client_selected_cars_l222_222957

theorem each_client_selected_cars (cars clients selections : ℕ) (h1 : cars = 16) (h2 : selections = 3 * cars) (h3 : clients = 24) :
  selections / clients = 2 :=
by
  sorry

end each_client_selected_cars_l222_222957


namespace minimum_letters_for_grid_coloring_l222_222609

theorem minimum_letters_for_grid_coloring : 
  ∀ (grid_paper : Type) 
  (is_node : grid_paper → Prop) 
  (marked : grid_paper → Prop)
  (mark_with_letter : grid_paper → ℕ) 
  (connected : grid_paper → grid_paper → Prop), 
  (∀ n₁ n₂ : grid_paper, is_node n₁ → is_node n₂ → mark_with_letter n₁ = mark_with_letter n₂ → 
  (n₁ ≠ n₂ → ∃ n₃ : grid_paper, is_node n₃ ∧ connected n₁ n₃ ∧ connected n₃ n₂ ∧ mark_with_letter n₃ ≠ mark_with_letter n₁)) → 
  ∃ (k : ℕ), k = 2 :=
by
  sorry

end minimum_letters_for_grid_coloring_l222_222609


namespace cows_total_l222_222789

theorem cows_total (M F : ℕ) 
  (h1 : F = 2 * M) 
  (h2 : F / 2 = M / 2 + 50) : 
  M + F = 300 :=
by
  sorry

end cows_total_l222_222789


namespace print_shop_cost_difference_l222_222877

theorem print_shop_cost_difference :
  let cost_per_copy_X := 1.25
  let cost_per_copy_Y := 2.75
  let num_copies := 40
  let total_cost_X := cost_per_copy_X * num_copies
  let total_cost_Y := cost_per_copy_Y * num_copies
  total_cost_Y - total_cost_X = 60 :=
by 
  dsimp only []
  sorry

end print_shop_cost_difference_l222_222877


namespace range_of_z_l222_222219

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(h₁ : x + y = x * y) (h₂ : x + y + z = x * y * z) : 1 < z ∧ z ≤ 4 / 3 :=
sorry

end range_of_z_l222_222219


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l222_222871

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l222_222871


namespace pell_infinite_solutions_l222_222730

theorem pell_infinite_solutions : ∃ m : ℕ, ∃ a b c : ℕ, 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (∀ n : ℕ, ∃ an bn cn : ℕ, 
    (1 / an + 1 / bn + 1 / cn + 1 / (an * bn * cn) = m / (an + bn + cn))) := 
sorry

end pell_infinite_solutions_l222_222730


namespace loss_percentage_first_book_l222_222374

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

end loss_percentage_first_book_l222_222374


namespace tom_average_speed_l222_222148

theorem tom_average_speed
  (total_distance : ℕ)
  (distance1 : ℕ)
  (speed1 : ℕ)
  (distance2 : ℕ)
  (speed2 : ℕ)
  (H : total_distance = distance1 + distance2)
  (H1 : distance1 = 12)
  (H2 : speed1 = 24)
  (H3 : distance2 = 48)
  (H4 : speed2 = 48) :
  (total_distance : ℚ) / ((distance1 : ℚ) / speed1 + (distance2 : ℚ) / speed2) = 40 :=
by
  sorry

end tom_average_speed_l222_222148


namespace quadratic_root_d_value_l222_222830

theorem quadratic_root_d_value :
  (∃ d : ℝ, ∀ x : ℝ, (2 * x^2 + 8 * x + d = 0) ↔ (x = (-8 + Real.sqrt 12) / 4) ∨ (x = (-8 - Real.sqrt 12) / 4)) → 
  d = 6.5 :=
by
  sorry

end quadratic_root_d_value_l222_222830


namespace math_problem_l222_222187

variable (a b c d : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
variable (h1 : a^3 + b^3 + 3 * a * b = 1)
variable (h2 : c + d = 1)

theorem math_problem :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 + (d + 1 / d)^3 ≥ 40 := sorry

end math_problem_l222_222187


namespace abs_of_neg_square_add_l222_222669

theorem abs_of_neg_square_add (a b : ℤ) : |-a^2 + b| = 10 :=
by
  sorry

end abs_of_neg_square_add_l222_222669


namespace ratio_gluten_free_l222_222756

theorem ratio_gluten_free (total_cupcakes vegan_cupcakes non_vegan_gluten cupcakes_gluten_free : ℕ)
    (H1 : total_cupcakes = 80)
    (H2 : vegan_cupcakes = 24)
    (H3 : non_vegan_gluten = 28)
    (H4 : cupcakes_gluten_free = vegan_cupcakes / 2) :
    (cupcakes_gluten_free : ℚ) / (total_cupcakes : ℚ) = 3 / 20 :=
by 
  -- Proof goes here
  sorry

end ratio_gluten_free_l222_222756


namespace correct_option_c_l222_222782

-- Definitions for the problem context
noncomputable def qualification_rate : ℝ := 0.99
noncomputable def picking_probability := qualification_rate

-- The theorem statement that needs to be proven
theorem correct_option_c : picking_probability = 0.99 :=
sorry

end correct_option_c_l222_222782


namespace ratio_of_millipedes_l222_222304

-- Define the given conditions
def total_segments_needed : ℕ := 800
def first_millipede_segments : ℕ := 60
def millipedes_segments (x : ℕ) : ℕ := x
def ten_millipedes_segments : ℕ := 10 * 50

-- State the main theorem
theorem ratio_of_millipedes (x : ℕ) : 
  total_segments_needed = 60 + 2 * x + 10 * 50 →
  2 * x / 60 = 4 :=
sorry

end ratio_of_millipedes_l222_222304


namespace value_of_business_l222_222590

variable (V : ℝ)
variable (h1 : (2 / 3) * V = S)
variable (h2 : (3 / 4) * S = 75000)

theorem value_of_business (h1 : (2 / 3) * V = S) (h2 : (3 / 4) * S = 75000) : V = 150000 :=
sorry

end value_of_business_l222_222590


namespace find_h_plus_k_l222_222932

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 14*y - 11 = 0

-- State the problem: Prove h + k = -4 given (h, k) is the center of the circle
theorem find_h_plus_k : (∃ h k, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 69) ∧ h + k = -4) :=
by {
  sorry
}

end find_h_plus_k_l222_222932


namespace arcsin_cos_eq_l222_222245

theorem arcsin_cos_eq :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  have h1 : Real.cos (2 * Real.pi / 3) = -1 / 2 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  rw [h1, h2]

end arcsin_cos_eq_l222_222245


namespace parabola_point_ordinate_l222_222535

-- The definition of the problem as a Lean 4 statement
theorem parabola_point_ordinate (a : ℝ) (x₀ y₀ : ℝ) 
  (h₀ : 0 < a)
  (h₁ : x₀^2 = (1 / a) * y₀)
  (h₂ : dist (0, 1 / (4 * a)) (0, -1 / (4 * a)) = 1)
  (h₃ : dist (x₀, y₀) (0, 1 / (4 * a)) = 5) :
  y₀ = 9 / 2 := 
sorry

end parabola_point_ordinate_l222_222535


namespace total_fault_line_movement_l222_222311

-- Define the movements in specific years.
def movement_past_year : ℝ := 1.25
def movement_year_before : ℝ := 5.25

-- Theorem stating the total movement of the fault line over the two years.
theorem total_fault_line_movement : movement_past_year + movement_year_before = 6.50 :=
by
  -- Proof is omitted.
  sorry

end total_fault_line_movement_l222_222311


namespace irrationals_among_examples_l222_222395

theorem irrationals_among_examples :
  ¬ ∃ (r : ℚ), r = π ∧
  (∃ (a b : ℚ), a * a = 4) ∧
  (∃ (r : ℚ), r = 0) ∧
  (∃ (r : ℚ), r = -22 / 7) := 
sorry

end irrationals_among_examples_l222_222395


namespace function_inequality_l222_222375

variable {f : ℕ → ℝ}
variable {a : ℝ}

theorem function_inequality (h : ∀ n : ℕ, f (n + 1) ≥ a^n * f n) :
  ∀ n : ℕ, f n = a^((n * (n - 1)) / 2) * f 1 := 
sorry

end function_inequality_l222_222375


namespace problem_proof_l222_222447

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_proof : f (1 + g 3) = 32 := by
  sorry

end problem_proof_l222_222447


namespace option_d_is_deductive_reasoning_l222_222403

-- Define the conditions of the problem
def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ c q : ℤ, c * q ≠ 0 ∧ ∀ n : ℕ, a n = c * q ^ n

-- Define the specific sequence {-2^n}
def a (n : ℕ) : ℤ := -2^n

-- State the proof problem
theorem option_d_is_deductive_reasoning :
  is_geometric_sequence a :=
sorry

end option_d_is_deductive_reasoning_l222_222403


namespace books_left_unchanged_l222_222718

theorem books_left_unchanged (initial_books : ℕ) (initial_pens : ℕ) (pens_sold : ℕ) (pens_left : ℕ) :
  initial_books = 51 → initial_pens = 106 → pens_sold = 92 → pens_left = 14 → initial_books = 51 := 
by
  intros h_books h_pens h_sold h_left
  exact h_books

end books_left_unchanged_l222_222718


namespace exists_divisible_triangle_l222_222128

theorem exists_divisible_triangle (p : ℕ) (n : ℕ) (m : ℕ) (points : Fin m → ℤ × ℤ) 
  (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_pos : 0 < n) (hm_eight : m = 8) 
  (on_circle : ∀ k : Fin m, (points k).fst ^ 2 + (points k).snd ^ 2 = (p ^ n) ^ 2) :
  ∃ (i j k : Fin m), (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ (∃ d : ℕ, (points i).fst - (points j).fst = p ^ d ∧ 
  (points i).snd - (points j).snd = p ^ d ∧ d ≥ n + 1) :=
sorry

end exists_divisible_triangle_l222_222128


namespace tom_has_9_balloons_l222_222924

-- Define Tom's and Sara's yellow balloon counts
variables (total_balloons saras_balloons toms_balloons : ℕ)

-- Given conditions
axiom total_balloons_def : total_balloons = 17
axiom saras_balloons_def : saras_balloons = 8
axiom toms_balloons_total : toms_balloons + saras_balloons = total_balloons

-- Theorem stating that Tom has 9 yellow balloons
theorem tom_has_9_balloons : toms_balloons = 9 := by
  sorry

end tom_has_9_balloons_l222_222924


namespace inequality_solution_l222_222073

theorem inequality_solution (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := 
sorry

end inequality_solution_l222_222073


namespace three_nabla_four_l222_222326

noncomputable def modified_operation (a b : ℝ) : ℝ :=
  (a + b^2) / (1 + a * b^2)

theorem three_nabla_four : modified_operation 3 4 = 19 / 49 := 
  by 
  sorry

end three_nabla_four_l222_222326


namespace real_roots_of_quadratics_l222_222940

theorem real_roots_of_quadratics {p1 p2 q1 q2 : ℝ} (h : p1 * p2 = 2 * (q1 + q2)) :
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  have D1 := p1^2 - 4 * q1
  have D2 := p2^2 - 4 * q2
  sorry

end real_roots_of_quadratics_l222_222940


namespace divisor_condition_l222_222740

def M (n : ℤ) : Set ℤ := {n, n+1, n+2, n+3, n+4}

def S (n : ℤ) : ℤ := 5*n^2 + 20*n + 30

def P (n : ℤ) : ℤ := (n * (n+1) * (n+2) * (n+3) * (n+4))^2

theorem divisor_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := 
by
  sorry

end divisor_condition_l222_222740


namespace find_m_l222_222809

/-- 
If the function y=x + m/(x-1) defined for x > 1 attains its minimum value at x = 3,
then the positive number m is 4.
-/
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x -> x + m / (x - 1) ≥ 3 + m / 2):
  m = 4 :=
sorry

end find_m_l222_222809


namespace sum_reciprocals_seven_l222_222713

variable (x y : ℝ)

theorem sum_reciprocals_seven (h : x + y = 7 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / x) + (1 / y) = 7 := 
sorry

end sum_reciprocals_seven_l222_222713


namespace James_total_passengers_l222_222277

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l222_222277


namespace y_range_l222_222418

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end y_range_l222_222418


namespace relationship_among_abc_l222_222519

noncomputable def a := Real.log 2 / Real.log (1/5)
noncomputable def b := 3 ^ (3/5)
noncomputable def c := 4 ^ (1/5)

theorem relationship_among_abc : a < c ∧ c < b := 
by
  sorry

end relationship_among_abc_l222_222519


namespace gcd_g50_g52_l222_222170

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 2 * x^2 + x + 2023

-- Define the integers n1 and n2 corresponding to g(50) and g(52)
def n1 : ℤ := g 50
def n2 : ℤ := g 52

-- Statement of the proof goal
theorem gcd_g50_g52 : Int.gcd n1 n2 = 1 := by
  sorry

end gcd_g50_g52_l222_222170


namespace brogan_total_red_apples_l222_222289

def red_apples (total_apples percentage_red : ℕ) : ℕ :=
  (total_apples * percentage_red) / 100

theorem brogan_total_red_apples :
  red_apples 20 40 + red_apples 20 50 = 18 :=
by
  sorry

end brogan_total_red_apples_l222_222289


namespace statement_A_statement_A_statement_C_statement_D_l222_222768

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end statement_A_statement_A_statement_C_statement_D_l222_222768


namespace least_positive_integer_reducible_fraction_l222_222696

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ gcd (n - 17) (7 * n + 4) > 1 ∧ (∀ m : ℕ, m > 0 ∧ gcd (m - 17) (7 * m + 4) > 1 → n ≤ m) :=
by sorry

end least_positive_integer_reducible_fraction_l222_222696


namespace ilya_arithmetic_l222_222619

theorem ilya_arithmetic (v t : ℝ) (h : v + t = v * t ∧ v + t = v / t) : False :=
by
  sorry

end ilya_arithmetic_l222_222619


namespace prove_a_lt_neg_one_l222_222454

variable {f : ℝ → ℝ} (a : ℝ)

-- Conditions:
-- 1. f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- 2. f has a period of 3
def has_period_three (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

-- 3. f(1) > 1
def f_one_gt_one (f : ℝ → ℝ) : Prop := f 1 > 1

-- 4. f(2) = a
def f_two_eq_a (f : ℝ → ℝ) (a : ℝ) : Prop := f 2 = a

-- Proof statement:
theorem prove_a_lt_neg_one (h1 : is_odd_function f) (h2 : has_period_three f)
  (h3 : f_one_gt_one f) (h4 : f_two_eq_a f a) : a < -1 :=
  sorry

end prove_a_lt_neg_one_l222_222454


namespace rhombus_area_l222_222642

/-
  We want to prove that the area of a rhombus with given diagonals' lengths is 
  equal to the computed value according to the formula Area = (d1 * d2) / 2.
-/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : 
  (d1 * d2) / 2 = 60 :=
by
  rw [h1, h2]
  sorry

end rhombus_area_l222_222642


namespace calc_305_squared_minus_295_squared_l222_222002

theorem calc_305_squared_minus_295_squared :
  305^2 - 295^2 = 6000 := 
  by
    sorry

end calc_305_squared_minus_295_squared_l222_222002


namespace cosine_range_l222_222824

theorem cosine_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.cos x ≤ 1 / 2) : 
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end cosine_range_l222_222824


namespace find_constants_l222_222102

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 + a * x
def g (x : ℝ) (b c : ℝ) : ℝ := b * x ^ 2 + c
def f' (x : ℝ) (a : ℝ) : ℝ := 6 * x ^ 2 + a
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * b * x

theorem find_constants (a b c : ℝ) :
  f 2 a = 0 ∧ g 2 b c = 0 ∧ f' 2 a = g' 2 b →
  a = -8 ∧ b = 4 ∧ c = -16 :=
by
  intro h
  sorry

end find_constants_l222_222102


namespace basketball_total_points_l222_222585

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l222_222585


namespace solve_system_of_equations_l222_222832

theorem solve_system_of_equations:
  (∀ (x y : ℝ), 2 * y - x - 2 * x * y = -1 ∧ 4 * x ^ 2 * y ^ 2 + x ^ 2 + 4 * y ^ 2 - 4 * x * y = 61 →
  (x, y) = (-6, -1/2) ∨ (x, y) = (1, 3) ∨ (x, y) = (1, -5/2) ∨ (x, y) = (5, -1/2)) :=
by
  sorry

end solve_system_of_equations_l222_222832


namespace quadratic_solution_l222_222023

theorem quadratic_solution (a c: ℝ) (h1 : a + c = 7) (h2 : a < c) (h3 : 36 - 4 * a * c = 0) : 
  a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2 :=
by
  sorry

end quadratic_solution_l222_222023


namespace value_of_a_l222_222299

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l222_222299


namespace consecutive_integers_no_two_l222_222747

theorem consecutive_integers_no_two (a n : ℕ) : 
  ¬(∃ (b : ℤ), (b : ℤ) = 2) :=
sorry

end consecutive_integers_no_two_l222_222747


namespace final_answer_for_m_l222_222488

noncomputable def proof_condition_1 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def proof_condition_2 (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

noncomputable def proof_condition_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem final_answer_for_m :
  (∀ (x y m : ℝ), proof_condition_1 x y m) →
  (∀ (x y : ℝ), proof_condition_2 x y) →
  (∀ (x1 y1 x2 y2 : ℝ), proof_condition_perpendicular x1 y1 x2 y2) →
  m = 12 / 5 :=
sorry

end final_answer_for_m_l222_222488


namespace num_students_B_l222_222378

-- Define the given conditions
variables (x : ℕ) -- The number of students who get a B

noncomputable def number_of_A := 2 * x
noncomputable def number_of_C := (12 / 10 : ℤ) * x -- Using (12 / 10) to approximate 1.2 in integers

-- Given total number of students is 42 for integer result
def total_students := 42

-- Lean statement to show number of students getting B is 10
theorem num_students_B : 4.2 * (x : ℝ) = 42 → x = 10 :=
by
  sorry

end num_students_B_l222_222378


namespace sum_of_coeffs_binomial_eq_32_l222_222686

noncomputable def sum_of_coeffs_binomial (x : ℝ) : ℝ :=
  (3 * x - 1 / Real.sqrt x)^5

theorem sum_of_coeffs_binomial_eq_32 :
  sum_of_coeffs_binomial 1 = 32 :=
by
  sorry

end sum_of_coeffs_binomial_eq_32_l222_222686


namespace simplify_expression_l222_222400

theorem simplify_expression :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end simplify_expression_l222_222400


namespace train_speed_is_100_kmph_l222_222777

noncomputable def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * 3.6

theorem train_speed_is_100_kmph :
  speed_of_train 100 3.6 = 100 :=
by
  sorry

end train_speed_is_100_kmph_l222_222777


namespace factor_x_minus_1_l222_222541

theorem factor_x_minus_1 (P Q R S : Polynomial ℂ) : 
  (P.eval 1 = 0) → 
  (P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) 
  = (x^4 + x^3 + x^2 + x + 1) * S.eval (x)) :=
sorry

end factor_x_minus_1_l222_222541


namespace prob_blue_lower_than_yellow_l222_222563

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  3^(-k : ℤ)

noncomputable def prob_same_bin : ℝ :=
  ∑' k, 3^(-2*k : ℤ)

theorem prob_blue_lower_than_yellow :
  (1 - prob_same_bin) / 2 = 7 / 16 :=
by
  -- proof goes here
  sorry

end prob_blue_lower_than_yellow_l222_222563


namespace sequence_has_max_and_min_l222_222359

noncomputable def a_n (n : ℕ) : ℝ := (4 / 9)^(n - 1) - (2 / 3)^(n - 1)

theorem sequence_has_max_and_min : 
  (∃ N, ∀ n, a_n n ≤ a_n N) ∧ 
  (∃ M, ∀ n, a_n n ≥ a_n M) :=
sorry

end sequence_has_max_and_min_l222_222359


namespace smallest_k_no_real_roots_l222_222903

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 13 ≠ 0) ∧
  (∀ n : ℤ, n < k → ∃ x : ℝ, 3 * x * (n * x - 5) - 2 * x^2 + 13 = 0) :=
by sorry

end smallest_k_no_real_roots_l222_222903


namespace watch_cost_price_l222_222911

theorem watch_cost_price (C : ℝ) (h1 : 0.85 * C = SP1) (h2 : 1.06 * C = SP2) (h3 : SP2 - SP1 = 350) : 
  C = 1666.67 := 
  sorry

end watch_cost_price_l222_222911


namespace circle_intersects_y_axis_at_one_l222_222283

theorem circle_intersects_y_axis_at_one :
  let A := (-2011, 0)
  let B := (2010, 0)
  let C := (0, (-2010) * 2011)
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧
    (∃ O : ℝ × ℝ, O = (0, 0) ∧
    (dist O A) * (dist O B) = (dist O C) * (dist O D)) :=
by
  sorry -- Proof of the theorem

end circle_intersects_y_axis_at_one_l222_222283


namespace total_food_each_day_l222_222029

-- Conditions
def num_dogs : ℕ := 2
def food_per_dog : ℝ := 0.125
def total_food : ℝ := num_dogs * food_per_dog

-- Proof statement
theorem total_food_each_day : total_food = 0.25 :=
by
  sorry

end total_food_each_day_l222_222029


namespace min_fraction_value_l222_222117

theorem min_fraction_value
    (a x y : ℕ)
    (h1 : a > 100)
    (h2 : x > 100)
    (h3 : y > 100)
    (h4 : y^2 - 1 = a^2 * (x^2 - 1))
    : a / x ≥ 2 := 
sorry

end min_fraction_value_l222_222117


namespace set_inter_complement_l222_222505

open Set

variable {α : Type*}
variable (U A B : Set α)

theorem set_inter_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {1, 4}) :
  ((U \ A) ∩ B) = {4} := 
by
  sorry

end set_inter_complement_l222_222505


namespace photograph_perimeter_l222_222339

-- Definitions of the conditions
def photograph_is_rectangular : Prop := True
def one_inch_border_area (w l m : ℕ) : Prop := (w + 2) * (l + 2) = m
def three_inch_border_area (w l m : ℕ) : Prop := (w + 6) * (l + 6) = m + 52

-- Lean statement of the problem
theorem photograph_perimeter (w l m : ℕ) 
  (h1 : photograph_is_rectangular)
  (h2 : one_inch_border_area w l m)
  (h3 : three_inch_border_area w l m) : 
  2 * (w + l) = 10 := 
by 
  sorry

end photograph_perimeter_l222_222339


namespace find_value_of_expression_l222_222629

variable {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : x + y = x * y + 1)

theorem find_value_of_expression (h : x + y = x * y + 1) : 
  (1 / x) + (1 / y) = 1 + (1 / (x * y)) :=
  sorry

end find_value_of_expression_l222_222629


namespace inequality_proof_l222_222712

theorem inequality_proof (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^2 - b * c) / (2 * a^2 + b * c) + (b^2 - c * a) / (2 * b^2 + c * a) + (c^2 - a * b) / (2 * c^2 + a * b) ≤ 0 :=
sorry

end inequality_proof_l222_222712


namespace arithmetic_sequence_common_difference_l222_222883

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions
def condition1 : Prop := ∀ n, S n = (n * (2*a 1 + (n-1) * d)) / 2
def condition2 : Prop := S 3 = 6
def condition3 : Prop := a 3 = 0

-- Question
def question : ℝ := d

-- Correct Answer
def correct_answer : ℝ := -2

-- Proof Problem Statement
theorem arithmetic_sequence_common_difference : 
  condition1 a S d ∧ condition2 S ∧ condition3 a →
  question d = correct_answer :=
sorry

end arithmetic_sequence_common_difference_l222_222883


namespace simplify_polynomial_l222_222723

theorem simplify_polynomial (q : ℚ) :
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := 
by 
  sorry

end simplify_polynomial_l222_222723


namespace combination_coins_l222_222104

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l222_222104


namespace prime_large_factor_l222_222437

theorem prime_large_factor (p : ℕ) (hp : Nat.Prime p) (hp_ge_3 : p ≥ 3) (x : ℕ) (hx_large : ∃ N, x ≥ N) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ (p + 3) / 2 ∧ (∃ q : ℕ, Nat.Prime q ∧ q > p ∧ q ∣ (x + i)) := by
  sorry

end prime_large_factor_l222_222437


namespace total_revenue_full_price_l222_222693

theorem total_revenue_full_price (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (3 * p) / 4 = 2800) : 
  f * p = 680 :=
by
  -- proof omitted
  sorry

end total_revenue_full_price_l222_222693


namespace foci_distance_of_hyperbola_l222_222383

theorem foci_distance_of_hyperbola : 
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c = 4 * Real.sqrt 10 :=
by
  -- Definitions based on conditions
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  
  -- Proof outline here (using sorry to skip proof details)
  sorry

end foci_distance_of_hyperbola_l222_222383


namespace find_f_neg_one_l222_222534

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)

theorem find_f_neg_one (k : ℝ) (h : ∀ (x : ℝ), f k (-x) = -f k x) : f k (-1) = 2 :=
sorry

end find_f_neg_one_l222_222534


namespace prism_lateral_edges_correct_cone_axial_section_equilateral_l222_222466

/-- Defining the lateral edges of a prism and its properties --/
structure Prism (r : ℝ) :=
(lateral_edges_equal : ∀ (e1 e2 : ℝ), e1 = r ∧ e2 = r)

/-- Defining the axial section of a cone with properties of base radius and generatrix length --/
structure Cone (r : ℝ) :=
(base_radius : ℝ := r)
(generatrix_length : ℝ := 2 * r)
(is_equilateral : base_radius * 2 = generatrix_length)

theorem prism_lateral_edges_correct (r : ℝ) (P : Prism r) : 
 ∃ e, e = r ∧ ∀ e', e' = r :=
by {
  sorry
}

theorem cone_axial_section_equilateral (r : ℝ) (C : Cone r) : 
 base_radius * 2 = generatrix_length :=
by {
  sorry
}

end prism_lateral_edges_correct_cone_axial_section_equilateral_l222_222466


namespace student_score_max_marks_l222_222550

theorem student_score_max_marks (M : ℝ)
  (pass_threshold : ℝ := 0.60 * M)
  (student_marks : ℝ := 80)
  (fail_by : ℝ := 40)
  (required_passing_score : ℝ := student_marks + fail_by) :
  pass_threshold = required_passing_score → M = 200 := 
by
  sorry

end student_score_max_marks_l222_222550


namespace line_circle_intersection_l222_222005

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection_l222_222005


namespace sqrt_of_4_l222_222867

theorem sqrt_of_4 :
  {x | x * x = 4} = {2, -2} :=
sorry

end sqrt_of_4_l222_222867


namespace jonathan_tax_per_hour_l222_222091

-- Given conditions
def wage : ℝ := 25          -- wage in dollars per hour
def tax_rate : ℝ := 0.024    -- tax rate in decimal

-- Prove statement
theorem jonathan_tax_per_hour :
  (wage * 100) * tax_rate = 60 :=
sorry

end jonathan_tax_per_hour_l222_222091


namespace carpet_area_in_yards_l222_222763

def main_length_feet : ℕ := 15
def main_width_feet : ℕ := 12
def extension_length_feet : ℕ := 6
def extension_width_feet : ℕ := 5
def feet_per_yard : ℕ := 3

def main_length_yards : ℕ := main_length_feet / feet_per_yard
def main_width_yards : ℕ := main_width_feet / feet_per_yard
def extension_length_yards : ℕ := extension_length_feet / feet_per_yard
def extension_width_yards : ℕ := extension_width_feet / feet_per_yard

def main_area_yards : ℕ := main_length_yards * main_width_yards
def extension_area_yards : ℕ := extension_length_yards * extension_width_yards

theorem carpet_area_in_yards : (main_area_yards : ℚ) + (extension_area_yards : ℚ) = 23.33 := 
by
  apply sorry

end carpet_area_in_yards_l222_222763


namespace incorrect_calculation_l222_222691

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l222_222691


namespace narrow_black_stripes_l222_222346

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l222_222346


namespace yellow_ball_kids_l222_222256

theorem yellow_ball_kids (total_kids white_ball_kids both_ball_kids : ℕ) :
  total_kids = 35 → white_ball_kids = 26 → both_ball_kids = 19 → 
  (total_kids = white_ball_kids + (total_kids - both_ball_kids)) → 
  (total_kids - (white_ball_kids - both_ball_kids)) = 28 :=
by
  sorry

end yellow_ball_kids_l222_222256


namespace least_positive_integer_l222_222323

theorem least_positive_integer (k : ℕ) (h : (528 + k) % 5 = 0) : k = 2 :=
sorry

end least_positive_integer_l222_222323


namespace value_of_fraction_l222_222533

theorem value_of_fraction (x y : ℝ) (h : 1 / x - 1 / y = 2) : (x + x * y - y) / (x - x * y - y) = 1 / 3 :=
by
  sorry

end value_of_fraction_l222_222533


namespace average_speed_of_bus_trip_l222_222520

theorem average_speed_of_bus_trip
  (v : ℝ)
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_increment : ℝ)
  (original_time : ℝ := distance / v)
  (faster_time : ℝ := distance / (v + speed_increment))
  (h1 : distance = 360)
  (h2 : time_difference = 1)
  (h3 : speed_increment = 5)
  (h4 : original_time - time_difference = faster_time) :
  v = 40 :=
by
  sorry

end average_speed_of_bus_trip_l222_222520


namespace smallest_integer_five_consecutive_sum_2025_l222_222810

theorem smallest_integer_five_consecutive_sum_2025 :
  ∃ n : ℤ, 5 * n + 10 = 2025 ∧ n = 403 :=
by
  sorry

end smallest_integer_five_consecutive_sum_2025_l222_222810


namespace number_of_pairs_exterior_angles_l222_222692

theorem number_of_pairs_exterior_angles (m n : ℕ) :
  (3 ≤ m ∧ 3 ≤ n ∧ 360 = m * n) ↔ 20 = 20 := 
by sorry

end number_of_pairs_exterior_angles_l222_222692


namespace total_athletes_l222_222076

theorem total_athletes (g : ℕ) (p : ℕ)
  (h₁ : g = 7)
  (h₂ : p = 5)
  (h₃ : 3 * (g + p - 1) = 33) : 
  3 * (g + p - 1) = 33 :=
sorry

end total_athletes_l222_222076


namespace middle_angle_of_triangle_l222_222036

theorem middle_angle_of_triangle (α β γ : ℝ) 
  (h1 : 0 < β) (h2 : β < 90) 
  (h3 : α ≤ β) (h4 : β ≤ γ) 
  (h5 : α + β + γ = 180) :
  True :=
by
  -- Proof would go here
  sorry

end middle_angle_of_triangle_l222_222036


namespace pants_price_100_l222_222800

-- Define the variables and conditions
variables (x y : ℕ)

-- Define the prices according to the conditions
def coat_price_pants := x + 340
def coat_price_shoes_pants := y + x + 180
def total_price := (coat_price_pants x) + x + y

-- The theorem to prove
theorem pants_price_100 (h1: coat_price_pants x = coat_price_shoes_pants x y) (h2: total_price x y = 700) : x = 100 :=
sorry

end pants_price_100_l222_222800


namespace distance_between_intersection_points_l222_222818

noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def l (t : ℝ) : ℝ × ℝ :=
  (-2 * t + 2, 3 * t)

theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (∃ θ : ℝ, C θ = A) ∧
    (∃ t : ℝ, l t = A) ∧
    (∃ θ : ℝ, C θ = B) ∧
    (∃ t : ℝ, l t = B) ∧
    dist A B = Real.sqrt 13 / 2 :=
sorry

end distance_between_intersection_points_l222_222818


namespace boxes_calculation_proof_l222_222526

variable (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_box : ℕ)
variable (total_eggs : ℕ := baskets * eggs_per_basket)
variable (boxes_needed : ℕ := total_eggs / eggs_per_box)

theorem boxes_calculation_proof :
  baskets = 21 →
  eggs_per_basket = 48 →
  eggs_per_box = 28 →
  boxes_needed = 36 :=
by
  intros
  sorry

end boxes_calculation_proof_l222_222526


namespace wholesale_cost_calc_l222_222133

theorem wholesale_cost_calc (wholesale_cost : ℝ) 
  (h_profit : 0.15 * wholesale_cost = 28 - wholesale_cost) : 
  wholesale_cost = 28 / 1.15 :=
by
  sorry

end wholesale_cost_calc_l222_222133


namespace sum_of_roots_l222_222157

-- Given condition: the equation to be solved
def equation (x : ℝ) : Prop :=
  x^2 - 7 * x + 2 = 16

-- Define the proof problem: the sum of the values of x that satisfy the equation
theorem sum_of_roots : 
  (∀ x : ℝ, equation x) → x^2 - 7*x - 14 = 0 → (root : ℝ) → (root₀ + root₁) = 7 :=
by
  sorry

end sum_of_roots_l222_222157


namespace negation_of_homework_submission_l222_222863

variable {S : Type} -- S is the set of all students in this class
variable (H : S → Prop) -- H(x) means "student x has submitted the homework"

theorem negation_of_homework_submission :
  (¬ ∀ x, H x) ↔ (∃ x, ¬ H x) :=
by
  sorry

end negation_of_homework_submission_l222_222863


namespace ones_digit_of_first_in_sequence_l222_222442

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
  
def in_arithmetic_sequence (a d : ℕ) (n : ℕ) : Prop :=
  ∃ k, a = k * d + n

theorem ones_digit_of_first_in_sequence {p q r s t : ℕ}
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (ht : is_prime t)
  (hseq : in_arithmetic_sequence p 10 q ∧ 
          in_arithmetic_sequence q 10 r ∧
          in_arithmetic_sequence r 10 s ∧
          in_arithmetic_sequence s 10 t)
  (hincr : p < q ∧ q < r ∧ r < s ∧ s < t)
  (hstart : p > 5) :
  p % 10 = 1 := sorry

end ones_digit_of_first_in_sequence_l222_222442


namespace LCM_180_504_l222_222008

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l222_222008


namespace juju_juice_bar_l222_222177

theorem juju_juice_bar (M P : ℕ) 
  (h₁ : 6 * P = 54)
  (h₂ : 5 * M + 6 * P = 94) : 
  M + P = 17 := 
sorry

end juju_juice_bar_l222_222177


namespace FindDotsOnFaces_l222_222393

-- Define the structure of a die with specific dot distribution
structure Die where
  three_dots_face : ℕ
  two_dots_faces : ℕ
  one_dot_faces : ℕ

-- Define the problem scenario of 7 identical dice forming 'П' shape
noncomputable def SevenIdenticalDiceFormP (A B C : ℕ) : Prop :=
  ∃ (d : Die), 
    d.three_dots_face = 3 ∧
    d.two_dots_faces = 2 ∧
    d.one_dot_faces = 1 ∧
    (d.three_dots_face + d.two_dots_faces + d.one_dot_faces = 6) ∧
    (A = 2) ∧
    (B = 2) ∧
    (C = 3) 

-- State the theorem to prove A = 2, B = 2, C = 3 given the conditions
theorem FindDotsOnFaces (A B C : ℕ) (h : SevenIdenticalDiceFormP A B C) : A = 2 ∧ B = 2 ∧ C = 3 :=
  by sorry

end FindDotsOnFaces_l222_222393


namespace inequality_bound_l222_222913

theorem inequality_bound (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ e^x * (x^2 - x + 1) * (a * x + 3 * a - 1) < 1) : a < 2 / 3 :=
by
  sorry

end inequality_bound_l222_222913


namespace cube_painting_problem_l222_222450

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l222_222450


namespace desks_increase_l222_222288

theorem desks_increase 
  (rows : ℕ) (first_row_desks : ℕ) (total_desks : ℕ) 
  (d : ℕ) 
  (h_rows : rows = 8) 
  (h_first_row : first_row_desks = 10) 
  (h_total_desks : total_desks = 136)
  (h_desks_sum : 10 + (10 + d) + (10 + 2 * d) + (10 + 3 * d) + (10 + 4 * d) + (10 + 5 * d) + (10 + 6 * d) + (10 + 7 * d) = total_desks) : 
  d = 2 := 
by 
  sorry

end desks_increase_l222_222288


namespace xiao_hua_spent_7_yuan_l222_222656

theorem xiao_hua_spent_7_yuan :
  ∃ (a b c d: ℕ), a + b + c + d = 30 ∧
                   ((a = 5 ∧ b = 5 ∧ c = 10 ∧ d = 10) ∨
                    (a = 5 ∧ b = 10 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 5 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 10 ∧ c = 5 ∧ d = 5) ∨
                    (a = 5 ∧ b = 10 ∧ c = 10 ∧ d = 5) ∨
                    (a = 10 ∧ b = 5 ∧ c = 10 ∧ d = 5)) ∧
                   10 * c + 15 * a + 25 * b + 40 * d = 700 :=
by {
  sorry
}

end xiao_hua_spent_7_yuan_l222_222656


namespace arithmetic_sequence_term_count_l222_222094

theorem arithmetic_sequence_term_count (a1 d an : ℤ) (h₀ : a1 = -6) (h₁ : d = 5) (h₂ : an = 59) :
  ∃ n : ℤ, an = a1 + (n - 1) * d ∧ n = 14 :=
by
  sorry

end arithmetic_sequence_term_count_l222_222094


namespace loaves_of_bread_l222_222286

-- Definitions for the given conditions
def total_flour : ℝ := 5
def flour_per_loaf : ℝ := 2.5

-- The statement of the problem
theorem loaves_of_bread (total_flour : ℝ) (flour_per_loaf : ℝ) : 
  total_flour / flour_per_loaf = 2 :=
by
  -- Proof is not required
  sorry

end loaves_of_bread_l222_222286


namespace fifteenth_term_ratio_l222_222109

noncomputable def U (n : ℕ) (c f : ℚ) := n * (2 * c + (n - 1) * f) / 2
noncomputable def V (n : ℕ) (g h : ℚ) := n * (2 * g + (n - 1) * h) / 2

theorem fifteenth_term_ratio (c f g h : ℚ)
  (h1 : ∀ n : ℕ, (n > 0) → (U n c f) / (V n g h) = (5 * (n * n) + 3 * n + 2) / (3 * (n * n) + 2 * n + 30)) :
  (c + 14 * f) / (g + 14 * h) = 125 / 99 :=
by
  sorry

end fifteenth_term_ratio_l222_222109


namespace least_number_subtracted_divisible_by_17_and_23_l222_222278

-- Conditions
def is_divisible_by_17_and_23 (n : ℕ) : Prop := 
  n % 17 = 0 ∧ n % 23 = 0

def target_number : ℕ := 7538

-- The least number to be subtracted
noncomputable def least_number_to_subtract : ℕ := 109

-- Theorem statement
theorem least_number_subtracted_divisible_by_17_and_23 : 
  is_divisible_by_17_and_23 (target_number - least_number_to_subtract) :=
by 
  -- Proof details would normally follow here.
  sorry

end least_number_subtracted_divisible_by_17_and_23_l222_222278


namespace gcd_correct_l222_222379

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l222_222379


namespace points_in_quadrants_l222_222365

theorem points_in_quadrants :
  ∀ (x y : ℝ), (y > 3 * x) → (y > 5 - 2 * x) → ((0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)) :=
by
  intros x y h1 h2
  sorry

end points_in_quadrants_l222_222365


namespace largest_value_a_plus_b_plus_c_l222_222739

open Nat
open Function

def sum_of_digits (n : ℕ) : ℕ :=
  (digits 10 n).sum

theorem largest_value_a_plus_b_plus_c :
  ∃ (a b c : ℕ),
    10 ≤ a ∧ a < 100 ∧
    100 ≤ b ∧ b < 1000 ∧
    1000 ≤ c ∧ c < 10000 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    (a + b + c = 10199) := sorry

end largest_value_a_plus_b_plus_c_l222_222739


namespace expand_subtract_equals_result_l222_222607

-- Definitions of the given expressions
def expand_and_subtract (x : ℝ) : ℝ :=
  (x + 3) * (2 * x - 5) - (2 * x + 1)

-- Expected result
def expected_result (x : ℝ) : ℝ :=
  2 * x ^ 2 - x - 16

-- The theorem stating the equivalence of the expanded and subtracted expression with the expected result
theorem expand_subtract_equals_result (x : ℝ) : expand_and_subtract x = expected_result x :=
  sorry

end expand_subtract_equals_result_l222_222607


namespace first_triangular_number_year_in_21st_century_l222_222769

theorem first_triangular_number_year_in_21st_century :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2016 ∧ 2000 ≤ 2016 ∧ 2016 < 2100 :=
by
  sorry

end first_triangular_number_year_in_21st_century_l222_222769


namespace quadratic_completion_l222_222678

theorem quadratic_completion 
    (x : ℝ) 
    (h : 16*x^2 - 32*x - 512 = 0) : 
    ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by sorry

end quadratic_completion_l222_222678


namespace students_play_both_l222_222934

-- Definitions of problem conditions
def total_students : ℕ := 1200
def play_football : ℕ := 875
def play_cricket : ℕ := 450
def play_neither : ℕ := 100
def play_either := total_students - play_neither

-- Lean statement to prove that the number of students playing both football and cricket
theorem students_play_both : play_football + play_cricket - 225 = play_either :=
by
  -- The proof is omitted
  sorry

end students_play_both_l222_222934


namespace value_of_a_minus_b_l222_222902

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : |a + b| = a + b) : a - b = 2 ∨ a - b = 14 := 
sorry

end value_of_a_minus_b_l222_222902


namespace cos_identity_arithmetic_sequence_in_triangle_l222_222586

theorem cos_identity_arithmetic_sequence_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 2 * b = a + c)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : A + B + C = Real.pi)
  : 5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 4 := 
  sorry

end cos_identity_arithmetic_sequence_in_triangle_l222_222586


namespace find_x_value_l222_222567

theorem find_x_value :
  ∃ x : ℝ, (75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734) ∧ (x = 37.03) :=
by {
  sorry
}

end find_x_value_l222_222567


namespace cricket_run_rate_l222_222206

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (total_target : ℝ) (overs_first_period : ℕ) (overs_remaining_period : ℕ)
  (h1 : run_rate_first_10_overs = 3.2)
  (h2 : total_target = 252)
  (h3 : overs_first_period = 10)
  (h4 : overs_remaining_period = 40) :
  (total_target - (run_rate_first_10_overs * overs_first_period)) / overs_remaining_period = 5.5 := 
by
  sorry

end cricket_run_rate_l222_222206


namespace equation_of_perpendicular_line_l222_222461

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), (5, 3) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  ∧ (a = 2 ∧ b = 1 ∧ c = -13)
  ∧ (a * 1 + b * (-2) = 0) :=
sorry

end equation_of_perpendicular_line_l222_222461


namespace french_fries_cost_is_10_l222_222844

-- Define the costs as given in the problem conditions
def taco_salad_cost : ℕ := 10
def daves_single_cost : ℕ := 5
def peach_lemonade_cost : ℕ := 2
def num_friends : ℕ := 5
def friend_payment : ℕ := 11

-- Define the total amount collected from friends
def total_collected : ℕ := num_friends * friend_payment

-- Define the subtotal for the known items
def subtotal : ℕ := taco_salad_cost + (num_friends * daves_single_cost) + (num_friends * peach_lemonade_cost)

-- The total cost of french fries
def total_french_fries_cost := total_collected - subtotal

-- The proof statement:
theorem french_fries_cost_is_10 : total_french_fries_cost = 10 := by
  sorry

end french_fries_cost_is_10_l222_222844


namespace max_tied_teams_round_robin_l222_222516

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l222_222516


namespace root_of_polynomial_l222_222228

theorem root_of_polynomial (k : ℝ) (h : (3 : ℝ) ^ 4 + k * (3 : ℝ) ^ 2 + 27 = 0) : k = -12 :=
by
  sorry

end root_of_polynomial_l222_222228


namespace sum_of_cubes_l222_222401

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l222_222401


namespace correct_judgment_l222_222262

open Real

def period_sin2x (T : ℝ) : Prop := ∀ x, sin (2 * x) = sin (2 * (x + T))
def smallest_positive_period_sin2x : Prop := ∃ T > 0, period_sin2x T ∧ ∀ T' > 0, period_sin2x T' → T ≤ T'
def smallest_positive_period_sin2x_is_pi : Prop := ∃ T, smallest_positive_period_sin2x ∧ T = π

def symmetry_cosx (L : ℝ) : Prop := ∀ x, cos (L - x) = cos (L + x)
def symmetry_about_line_cosx (L : ℝ) : Prop := L = π / 2

def p : Prop := smallest_positive_period_sin2x_is_pi
def q : Prop := symmetry_about_line_cosx (π / 2)

theorem correct_judgment : ¬ (p ∧ q) :=
by 
  sorry

end correct_judgment_l222_222262


namespace find_number_l222_222694

-- Define the condition given in the problem
def condition (x : ℤ) := 13 * x - 272 = 105

-- Prove that given the condition, x equals 29
theorem find_number : ∃ x : ℤ, condition x ∧ x = 29 :=
by
  use 29
  unfold condition
  sorry

end find_number_l222_222694


namespace pete_backward_speed_l222_222893

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l222_222893


namespace find_nth_number_in_s_l222_222126

def s (k : ℕ) : ℕ := 8 * k + 5

theorem find_nth_number_in_s (n : ℕ) (number_in_s : ℕ) (h : number_in_s = 573) :
  ∃ k : ℕ, s k = number_in_s ∧ n = k + 1 := 
sorry

end find_nth_number_in_s_l222_222126


namespace sum_of_integers_70_to_85_l222_222771

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end sum_of_integers_70_to_85_l222_222771


namespace average_grade_of_female_students_is_92_l222_222440

noncomputable def female_average_grade 
  (overall_avg : ℝ) (male_avg : ℝ) (num_males : ℕ) (num_females : ℕ) : ℝ :=
  let total_students := num_males + num_females
  let total_score := total_students * overall_avg
  let male_total_score := num_males * male_avg
  let female_total_score := total_score - male_total_score
  female_total_score / num_females

theorem average_grade_of_female_students_is_92 :
  female_average_grade 90 83 8 28 = 92 := 
by
  -- Proof steps to be completed
  sorry

end average_grade_of_female_students_is_92_l222_222440


namespace count_possible_pairs_l222_222849

/-- There are four distinct mystery novels, three distinct fantasy novels, and three distinct biographies.
I want to choose two books with one of them being a specific mystery novel, "Mystery Masterpiece".
Prove that the number of possible pairs that include this mystery novel and one book from a different genre
is 6. -/
theorem count_possible_pairs (mystery_novels : Fin 4)
                            (fantasy_novels : Fin 3)
                            (biographies : Fin 3)
                            (MysteryMasterpiece : Fin 4):
                            (mystery_novels ≠ MysteryMasterpiece) →
                            ∀ genre : Fin 2, genre ≠ 0 ∧ genre ≠ 1 →
                            (genre = 1 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            (genre = 2 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            ∃ total_pairs : Nat, total_pairs = 6 :=
by
  intros h_ne_genres h_genres h_counts1 h_counts2
  sorry

end count_possible_pairs_l222_222849


namespace common_tangent_line_range_a_l222_222325

open Real

theorem common_tangent_line_range_a (a : ℝ) (h_pos : 0 < a) :
  (∃ x₁ x₂ : ℝ, 2 * a * x₁ = exp x₂ ∧ (exp x₂ - a * x₁^2) / (x₂ - x₁) = 2 * a * x₁) →
  a ≥ exp 2 / 4 := 
sorry

end common_tangent_line_range_a_l222_222325


namespace total_donation_l222_222501

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l222_222501


namespace sum_of_three_integers_l222_222843

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l222_222843


namespace value_of_y_l222_222386

theorem value_of_y (y : ℕ) (hy : (1 / 8) * 2^36 = 8^y) : y = 11 :=
by
  sorry

end value_of_y_l222_222386


namespace find_int_less_than_neg3_l222_222504

theorem find_int_less_than_neg3 : 
  ∃ x ∈ ({-4, -2, 0, 3} : Set Int), x < -3 ∧ x = -4 := 
by
  -- formal proof goes here
  sorry

end find_int_less_than_neg3_l222_222504


namespace compare_x_y_l222_222559

variable (a b : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (a_ne_b : a ≠ b)

noncomputable def x : ℝ := (Real.sqrt a + Real.sqrt b) / Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt (a + b)

theorem compare_x_y : y a b > x a b := sorry

end compare_x_y_l222_222559


namespace correct_operation_l222_222878

theorem correct_operation (x : ℝ) (hx : x ≠ 0) :
  (x^3 / x^2 = x) :=
by {
  sorry
}

end correct_operation_l222_222878


namespace bubbleSort_iter_count_l222_222731

/-- Bubble sort iterates over the list repeatedly, swapping adjacent elements if they are in the wrong order. -/
def bubbleSortSteps (lst : List Int) : List (List Int) :=
sorry -- Implementation of bubble sort to capture each state after each iteration

/-- Prove that sorting [6, -3, 0, 15] in descending order using bubble sort requires exactly 3 iterations. -/
theorem bubbleSort_iter_count : 
  (bubbleSortSteps [6, -3, 0, 15]).length = 3 :=
sorry

end bubbleSort_iter_count_l222_222731


namespace boys_on_trip_l222_222279

theorem boys_on_trip (B G : ℕ) 
    (h1 : G = B + (2 / 5 : ℚ) * B) 
    (h2 : 1 + 1 + 1 + B + G = 123) : 
    B = 50 := 
by 
  -- Proof skipped 
  sorry

end boys_on_trip_l222_222279


namespace proof_problem_l222_222470

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l222_222470


namespace fraction_of_remaining_birds_left_l222_222614

theorem fraction_of_remaining_birds_left :
  ∀ (total_birds initial_fraction next_fraction x : ℚ), 
    total_birds = 60 ∧ 
    initial_fraction = 1 / 3 ∧ 
    next_fraction = 2 / 5 ∧ 
    8 = (total_birds * (1 - initial_fraction)) * (1 - next_fraction) * (1 - x) →
    x = 2 / 3 :=
by
  intros total_birds initial_fraction next_fraction x h
  obtain ⟨hb, hi, hn, he⟩ := h
  sorry

end fraction_of_remaining_birds_left_l222_222614


namespace angle_C_is_108_l222_222851

theorem angle_C_is_108
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : C < D)
  (h4 : D < E)
  (h5 : B - A = C - B)
  (h6 : C - B = D - C)
  (h7 : D - C = E - D)
  (angle_sum : A + B + C + D + E = 540) :
  C = 108 := 
sorry

end angle_C_is_108_l222_222851


namespace line_equation_l222_222025

open Real

-- Define the points A, B, and C
def A : ℝ × ℝ := ⟨1, 4⟩
def B : ℝ × ℝ := ⟨3, 2⟩
def C : ℝ × ℝ := ⟨2, -1⟩

-- Definition for a line passing through point C
-- and having equal distance to points A and B
def is_line_equation (l : ℝ → ℝ → Prop) :=
  ∀ x y, (l x y ↔ (x + y - 1 = 0 ∨ x - 2 = 0))

-- Our main statement
theorem line_equation :
  ∃ l : ℝ → ℝ → Prop, is_line_equation l ∧ (l 2 (-1)) :=
by
  sorry  -- Proof goes here.

end line_equation_l222_222025


namespace specified_time_eq_l222_222285

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end specified_time_eq_l222_222285


namespace solve_fractional_eq_l222_222894

theorem solve_fractional_eq (x: ℝ) (h1: x ≠ -11) (h2: x ≠ -8) (h3: x ≠ -12) (h4: x ≠ -7) :
  (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) → (x = -19 / 2) :=
by
  sorry

end solve_fractional_eq_l222_222894


namespace intersection_correct_l222_222176

open Set

noncomputable def A := {x : ℕ | x^2 - x - 2 ≤ 0}
noncomputable def B := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def A_cap_B := A ∩ {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_correct : A_cap_B = {0, 1} :=
sorry

end intersection_correct_l222_222176


namespace find_x_when_areas_equal_l222_222815

-- Definitions based on the problem conditions
def glass_area : ℕ := 4 * (30 * 20)
def window_area (x : ℕ) : ℕ := (60 + 3 * x) * (40 + 3 * x)
def total_area_of_glass : ℕ := glass_area
def total_area_of_wood (x : ℕ) : ℕ := window_area x - glass_area

-- Proof problem, proving x == 20 / 3 when total area of glass equals total area of wood
theorem find_x_when_areas_equal : 
  ∃ x : ℕ, (total_area_of_glass = total_area_of_wood x) ∧ x = 20 / 3 :=
sorry

end find_x_when_areas_equal_l222_222815


namespace simplify_fraction_l222_222225

theorem simplify_fraction (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := 
by
  sorry

end simplify_fraction_l222_222225


namespace simplify_trig_identity_l222_222556

open Real

theorem simplify_trig_identity (x y : ℝ) :
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = sin y ^ 2 := 
sorry

end simplify_trig_identity_l222_222556


namespace mother_age_is_correct_l222_222973

variable (D M : ℕ)

theorem mother_age_is_correct:
  (D + 3 = 26) → (M - 5 = 2 * (D - 5)) → M = 41 := by
  intros h1 h2
  sorry

end mother_age_is_correct_l222_222973


namespace sum_is_two_l222_222246

-- Define the numbers based on conditions
def a : Int := 9
def b : Int := -9 + 2

-- Theorem stating that the sum of the two numbers is 2
theorem sum_is_two : a + b = 2 :=
by
  -- proof goes here
  sorry

end sum_is_two_l222_222246


namespace bacteria_population_l222_222874

theorem bacteria_population (initial_population : ℕ) (tripling_factor : ℕ) (hours_per_tripling : ℕ) (target_population : ℕ) 
(initial_population_eq : initial_population = 300)
(tripling_factor_eq : tripling_factor = 3)
(hours_per_tripling_eq : hours_per_tripling = 5)
(target_population_eq : target_population = 87480) :
∃ n : ℕ, (hours_per_tripling * n = 30) ∧ (initial_population * (tripling_factor ^ n) ≥ target_population) := sorry

end bacteria_population_l222_222874


namespace buyers_of_cake_mix_l222_222969

/-
  A certain manufacturer of cake, muffin, and bread mixes has 100 buyers,
  of whom some purchase cake mix, 40 purchase muffin mix, and 17 purchase both cake mix and muffin mix.
  If a buyer is to be selected at random from the 100 buyers, the probability that the buyer selected will be one who purchases 
  neither cake mix nor muffin mix is 0.27.
  Prove that the number of buyers who purchase cake mix is 50.
-/

theorem buyers_of_cake_mix (C M B total : ℕ) (hM : M = 40) (hB : B = 17) (hTotal : total = 100)
    (hProb : (total - (C + M - B) : ℝ) / total = 0.27) : C = 50 :=
by
  -- Definition of the proof is required here
  sorry

end buyers_of_cake_mix_l222_222969


namespace is_composite_1010_pattern_l222_222654

theorem is_composite_1010_pattern (k : ℕ) (h : k ≥ 2) : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (1010^k + 101 = a * b)) :=
  sorry

end is_composite_1010_pattern_l222_222654


namespace number_of_student_tickets_sold_l222_222945

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end number_of_student_tickets_sold_l222_222945


namespace cube_split_odd_numbers_l222_222606

theorem cube_split_odd_numbers (m : ℕ) (h1 : 1 < m) (h2 : ∃ k, (31 = 2 * k + 1 ∧ (m - 1) * m / 2 = k)) : m = 6 := 
by
  sorry

end cube_split_odd_numbers_l222_222606


namespace most_convincing_method_for_relationship_l222_222930

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship_l222_222930


namespace bob_weight_l222_222921

noncomputable def jim_bob_equations (j b : ℝ) : Prop :=
  j + b = 200 ∧ b - 3 * j = b / 4

theorem bob_weight (j b : ℝ) (h : jim_bob_equations j b) : b = 171.43 :=
by
  sorry

end bob_weight_l222_222921


namespace value_of_expression_at_3_l222_222238

theorem value_of_expression_at_3 :
  ∀ (x : ℕ), x = 3 → (x^4 - 6 * x) = 63 :=
by
  intros x h
  sorry

end value_of_expression_at_3_l222_222238


namespace files_missing_is_15_l222_222439

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l222_222439


namespace hexagon_area_l222_222573

theorem hexagon_area (s : ℝ) (hex_area : ℝ) (p q : ℤ) :
  s = 3 ∧ hex_area = (3 * Real.sqrt 3 / 2) * s^2 ∧ hex_area = Real.sqrt p + Real.sqrt q → p + q = 545 :=
by
  sorry

end hexagon_area_l222_222573


namespace probability_two_digit_between_15_25_l222_222858

-- Define a type for standard six-sided dice rolls
def is_standard_six_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define the set of valid two-digit numbers
def valid_two_digit_number (n : ℕ) : Prop := n ≥ 15 ∧ n ≤ 25

-- Function to form a two-digit number from two dice rolls
def form_two_digit_number (d1 d2 : ℕ) : ℕ := 10 * d1 + d2

-- The main statement of the problem
theorem probability_two_digit_between_15_25 :
  (∃ (n : ℚ), n = 5/9) ∧
  (∀ (d1 d2 : ℕ), is_standard_six_sided_die d1 → is_standard_six_sided_die d2 →
  valid_two_digit_number (form_two_digit_number d1 d2)) :=
sorry

end probability_two_digit_between_15_25_l222_222858


namespace solve_line_eq_l222_222908

theorem solve_line_eq (a b x : ℝ) (h1 : (0 : ℝ) * a + b = 2) (h2 : -3 * a + b = 0) : x = -3 :=
by
  sorry

end solve_line_eq_l222_222908


namespace bread_remaining_is_26_85_l222_222388

noncomputable def bread_leftover (jimin_cm : ℕ) (taehyung_m original_length : ℝ) : ℝ :=
  original_length - (jimin_cm / 100 + taehyung_m)

theorem bread_remaining_is_26_85 :
  bread_leftover 150 1.65 30 = 26.85 :=
by
  sorry

end bread_remaining_is_26_85_l222_222388


namespace find_range_of_m_l222_222766

-- Define propositions p and q based on the problem description
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, m ≠ 0 → (x - 2 * y + 3 = 0 ∧ y * y ≠ m * x)

def q (m : ℝ) : Prop :=
  5 - 2 * m ≠ 0 ∧ m ≠ 0 ∧ (∃ x y : ℝ, (x * x) / (5 - 2 * m) + (y * y) / m = 1)

-- Given conditions
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

-- The range of m that satisfies the given problem
def valid_m (m : ℝ) : Prop :=
  (m ≥ 3) ∨ (m < 0) ∨ (0 < m ∧ m ≤ 2.5)

theorem find_range_of_m (m : ℝ) : condition1 m → condition2 m → valid_m m := 
  sorry

end find_range_of_m_l222_222766


namespace intersection_closure_M_and_N_l222_222486

noncomputable def set_M : Set ℝ :=
  { x | 2 / x < 1 }

noncomputable def closure_M : Set ℝ :=
  Set.Icc 0 2

noncomputable def set_N : Set ℝ :=
  { y | ∃ x, y = Real.sqrt (x - 1) }

theorem intersection_closure_M_and_N :
  (closure_M ∩ set_N) = Set.Icc 0 2 :=
by
  sorry

end intersection_closure_M_and_N_l222_222486


namespace range_of_k_for_real_roots_l222_222743

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 :=
by
  sorry

end range_of_k_for_real_roots_l222_222743


namespace max_bishops_on_chessboard_l222_222995

theorem max_bishops_on_chessboard (N : ℕ) (N_pos: 0 < N) : 
  ∃ max_number : ℕ, max_number = 2 * N - 2 :=
sorry

end max_bishops_on_chessboard_l222_222995


namespace temperature_on_Friday_l222_222159

variable {M T W Th F : ℝ}

theorem temperature_on_Friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (hM : M = 41) :
  F = 33 :=
by
  -- Proof goes here
  sorry

end temperature_on_Friday_l222_222159


namespace trivia_team_points_l222_222697

theorem trivia_team_points (total_members: ℕ) (total_points: ℕ) (points_per_member: ℕ) (members_showed_up: ℕ) (members_did_not_show_up: ℕ):
  total_members = 7 → 
  total_points = 20 → 
  points_per_member = 4 → 
  members_showed_up = total_points / points_per_member → 
  members_did_not_show_up = total_members - members_showed_up → 
  members_did_not_show_up = 2 := 
by 
  intros h1 h2 h3 h4 h5
  sorry

end trivia_team_points_l222_222697


namespace train_arrival_time_l222_222160

-- Define the time type
structure Time where
  hour : Nat
  minute : Nat

namespace Time

-- Define the addition of minutes to a time.
def add_minutes (t : Time) (m : Nat) : Time :=
  let new_minutes := t.minute + m
  if new_minutes < 60 then 
    { hour := t.hour, minute := new_minutes }
  else 
    { hour := t.hour + new_minutes / 60, minute := new_minutes % 60 }

-- Define the departure time
def departure_time : Time := { hour := 9, minute := 45 }

-- Define the travel time in minutes
def travel_time : Nat := 15

-- Define the expected arrival time
def expected_arrival_time : Time := { hour := 10, minute := 0 }

-- The theorem we need to prove
theorem train_arrival_time:
  add_minutes departure_time travel_time = expected_arrival_time := by
  sorry

end train_arrival_time_l222_222160


namespace melissa_points_per_game_l222_222364

variable (t g p : ℕ)

theorem melissa_points_per_game (ht : t = 36) (hg : g = 3) : p = t / g → p = 12 :=
by
  intro h
  sorry

end melissa_points_per_game_l222_222364


namespace trigonometric_identity_l222_222088

noncomputable def sin110cos40_minus_cos70sin40 : ℝ := 
  Real.sin (110 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (70 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)

theorem trigonometric_identity : 
  sin110cos40_minus_cos70sin40 = 1 / 2 := 
by sorry

end trigonometric_identity_l222_222088


namespace handshakes_7_boys_l222_222137

theorem handshakes_7_boys : Nat.choose 7 2 = 21 :=
by
  sorry

end handshakes_7_boys_l222_222137


namespace circles_intersect_l222_222942

noncomputable def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def circle2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 9}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 :=
sorry

end circles_intersect_l222_222942


namespace expand_expression_l222_222651

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := 
  sorry

end expand_expression_l222_222651


namespace inequality_AM_GM_HM_l222_222016

variable {x y k : ℝ}

-- Define the problem conditions
def is_positive (a : ℝ) : Prop := a > 0
def is_unequal (a b : ℝ) : Prop := a ≠ b
def positive_constant_lessthan_two (c : ℝ) : Prop := c > 0 ∧ c < 2

-- State the theorem to be proven
theorem inequality_AM_GM_HM (h₁ : is_positive x) 
                             (h₂ : is_positive y) 
                             (h₃ : is_unequal x y) 
                             (h₄ : positive_constant_lessthan_two k) :
  ( ( ( (x + y) / 2 )^k > ( (x * y)^(1/2) )^k ) ∧ 
    ( ( (x * y)^(1/2) )^k > ( ( 2 * x * y ) / ( x + y ) )^k ) ) :=
by
  sorry

end inequality_AM_GM_HM_l222_222016


namespace percent_psychology_majors_l222_222205

theorem percent_psychology_majors
  (total_students : ℝ)
  (pct_freshmen : ℝ)
  (pct_freshmen_liberal_arts : ℝ)
  (pct_freshmen_psychology_majors : ℝ)
  (h1 : pct_freshmen = 0.6)
  (h2 : pct_freshmen_liberal_arts = 0.4)
  (h3 : pct_freshmen_psychology_majors = 0.048)
  :
  (pct_freshmen_psychology_majors / (pct_freshmen * pct_freshmen_liberal_arts)) * 100 = 20 := 
by
  sorry

end percent_psychology_majors_l222_222205


namespace more_likely_millionaire_city_resident_l222_222259

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l222_222259


namespace max_Sn_in_arithmetic_sequence_l222_222920

theorem max_Sn_in_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ {m n p q : ℕ}, m + n = p + q → a m + a n = a p + a q)
  (h_a4 : a 4 = 1)
  (h_S5 : S 5 = 10)
  (h_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  ∃ n, n = 4 ∨ n = 5 ∧ ∀ m ≠ n, S m ≤ S n := by
  sorry

end max_Sn_in_arithmetic_sequence_l222_222920


namespace min_sum_of_squares_l222_222553

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3 * b + 5 * c + 7 * d = 14) : 
  a^2 + b^2 + c^2 + d^2 ≥ 7 / 3 :=
sorry

end min_sum_of_squares_l222_222553


namespace arithmetic_sequence_terms_count_l222_222417

theorem arithmetic_sequence_terms_count :
  ∃ n : ℕ, ∀ a d l, 
    a = 13 → 
    d = 3 → 
    l = 73 → 
    l = a + (n - 1) * d ∧ n = 21 :=
by
  sorry

end arithmetic_sequence_terms_count_l222_222417


namespace find_n_values_l222_222704

theorem find_n_values (n : ℚ) :
  ( 4 * n ^ 2 + 3 * n + 2 = 2 * n + 2 ∨ 4 * n ^ 2 + 3 * n + 2 = 5 * n + 4 ) →
  ( n = 0 ∨ n = 1 ) :=
by
  sorry

end find_n_values_l222_222704


namespace find_positive_int_sol_l222_222115

theorem find_positive_int_sol (a b c d n : ℕ) (h1 : n > 1) (h2 : a ≤ b) (h3 : b ≤ c) :
  ((n^a + n^b + n^c = n^d) ↔ 
  ((a = b ∧ b = c - 1 ∧ c = d - 1 ∧ n = 2) ∨ 
  (a = b ∧ b = c ∧ c = d - 1 ∧ n = 3))) :=
  sorry

end find_positive_int_sol_l222_222115


namespace one_set_working_communication_possible_l222_222282

variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

def P_A : ℝ := p^3
def P_B : ℝ := p^3
def P_not_A : ℝ := 1 - p^3
def P_not_B : ℝ := 1 - p^3

theorem one_set_working : 2 * P_A p - 2 * (P_A p)^2 = 2 * p^3 - 2 * p^6 :=
by 
  sorry

theorem communication_possible : 2 * P_A p - (P_A p)^2 = 2 * p^3 - p^6 :=
by 
  sorry

end one_set_working_communication_possible_l222_222282


namespace jump_difference_l222_222710

def frog_jump := 39
def grasshopper_jump := 17

theorem jump_difference :
  frog_jump - grasshopper_jump = 22 := by
  sorry

end jump_difference_l222_222710


namespace number_of_friends_l222_222884

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l222_222884


namespace not_coincidence_l222_222641

theorem not_coincidence (G : Type) [Fintype G] [DecidableEq G]
    (friend_relation : G → G → Prop)
    (h_friend : ∀ (a b : G), friend_relation a b → friend_relation b a)
    (initial_condition : ∀ (subset : Finset G), subset.card = 4 → 
         ∃ x ∈ subset, ∀ y ∈ subset, x ≠ y → friend_relation x y) :
    ∀ (subset : Finset G), subset.card = 4 → 
        ∃ x ∈ subset, ∀ y ∈ Finset.univ, x ≠ y → friend_relation x y :=
by
  intros subset h_card
  -- The proof would be constructed here
  sorry

end not_coincidence_l222_222641


namespace sapling_height_relationship_l222_222034

-- Definition to state the conditions
def initial_height : ℕ := 100
def growth_per_year : ℕ := 50
def height_after_years (years : ℕ) : ℕ := initial_height + growth_per_year * years

-- The theorem statement that should be proved
theorem sapling_height_relationship (x : ℕ) : height_after_years x = 50 * x + 100 := 
by
  sorry

end sapling_height_relationship_l222_222034


namespace smallest_possible_gcd_l222_222664

theorem smallest_possible_gcd (m n p : ℕ) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ k, k = Nat.gcd n p ∧ k = 60 := by
  sorry

end smallest_possible_gcd_l222_222664


namespace max_contribution_l222_222780

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end max_contribution_l222_222780


namespace find_a_l222_222351

theorem find_a (x y z a : ℝ) (k : ℝ) (h1 : x = 2 * k) (h2 : y = 3 * k) (h3 : z = 5 * k)
    (h4 : x + y + z = 100) (h5 : y = a * x - 10) : a = 2 :=
  sorry

end find_a_l222_222351


namespace good_numbers_count_1_to_50_l222_222754

def is_good_number (n : ℕ) : Prop :=
  ∃ (k l : ℕ), k ≠ 0 ∧ l ≠ 0 ∧ n = k * l + l - k

theorem good_numbers_count_1_to_50 : ∃ cnt, cnt = 49 ∧ (∀ n, n ∈ (Finset.range 51).erase 0 → is_good_number n) :=
  sorry

end good_numbers_count_1_to_50_l222_222754


namespace gcd_18_30_l222_222954

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l222_222954


namespace pencils_given_out_l222_222227
-- Define the problem conditions
def students : ℕ := 96
def dozens_per_student : ℕ := 7
def pencils_per_dozen : ℕ := 12

-- Define the expected total pencils
def expected_pencils : ℕ := 8064

-- Define the statement to be proven
theorem pencils_given_out : (students * (dozens_per_student * pencils_per_dozen)) = expected_pencils := 
  by
  sorry

end pencils_given_out_l222_222227


namespace mary_initial_sugar_eq_4_l222_222267

/-- Mary is baking a cake. The recipe calls for 7 cups of sugar and she needs to add 3 more cups of sugar. -/
def total_sugar : ℕ := 7
def additional_sugar : ℕ := 3

theorem mary_initial_sugar_eq_4 :
  ∃ initial_sugar : ℕ, initial_sugar + additional_sugar = total_sugar ∧ initial_sugar = 4 :=
sorry

end mary_initial_sugar_eq_4_l222_222267


namespace distribute_tourists_l222_222138

-- Define the number of ways k tourists can distribute among n cinemas
def num_ways (n k : ℕ) : ℕ := n^k

-- Theorem stating the number of distribution ways
theorem distribute_tourists (n k : ℕ) : num_ways n k = n^k :=
by sorry

end distribute_tourists_l222_222138


namespace train_crosses_platform_in_26_seconds_l222_222390

def km_per_hr_to_m_per_s (km_per_hr : ℕ) : ℕ :=
  km_per_hr * 5 / 18

def train_crossing_time
  (train_speed_km_per_hr : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) : ℕ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr
  total_distance_m / train_speed_m_per_s

theorem train_crosses_platform_in_26_seconds :
  train_crossing_time 72 300 220 = 26 :=
by
  sorry

end train_crosses_platform_in_26_seconds_l222_222390


namespace passed_boys_avg_marks_l222_222405

theorem passed_boys_avg_marks (total_boys : ℕ) (avg_marks_all_boys : ℕ) (avg_marks_failed_boys : ℕ) (passed_boys : ℕ) 
  (h1 : total_boys = 120)
  (h2 : avg_marks_all_boys = 35)
  (h3 : avg_marks_failed_boys = 15)
  (h4 : passed_boys = 100) : 
  (39 = (35 * 120 - 15 * (total_boys - passed_boys)) / passed_boys) :=
  sorry

end passed_boys_avg_marks_l222_222405


namespace right_triangle_area_hypotenuse_30_deg_l222_222673

theorem right_triangle_area_hypotenuse_30_deg
  (h : Real)
  (θ : Real)
  (A : Real)
  (H1 : θ = 30)
  (H2 : h = 12)
  : A = 18 * Real.sqrt 3 := by
  sorry

end right_triangle_area_hypotenuse_30_deg_l222_222673


namespace product_of_abcd_l222_222407

noncomputable def a (c : ℚ) : ℚ := 33 * c + 16
noncomputable def b (c : ℚ) : ℚ := 8 * c + 4
noncomputable def d (c : ℚ) : ℚ := c + 1

theorem product_of_abcd :
  (2 * a c + 3 * b c + 5 * c + 8 * d c = 45) →
  (4 * (d c + c) = b c) →
  (4 * (b c) + c = a c) →
  (c + 1 = d c) →
  a c * b c * c * d c = ((1511 : ℚ) / 103) * ((332 : ℚ) / 103) * (-(7 : ℚ) / 103) * ((96 : ℚ) / 103) :=
by
  intros
  sorry

end product_of_abcd_l222_222407


namespace jake_needs_total_hours_to_pay_off_debts_l222_222728

-- Define the conditions for the debts and payments
variable (debtA debtB debtC : ℝ)
variable (paymentA paymentB paymentC : ℝ)
variable (task1P task2P task3P task4P task5P task6P : ℝ)
variable (task2Payoff task4Payoff task6Payoff : ℝ)

-- Assume provided values
noncomputable def total_hours_needed : ℝ :=
  let remainingA := debtA - paymentA
  let remainingB := debtB - paymentB
  let remainingC := debtC - paymentC
  let hoursTask1 := (remainingA - task2Payoff) / task1P
  let hoursTask2 := task2Payoff / task2P
  let hoursTask3 := (remainingB - task4Payoff) / task3P
  let hoursTask4 := task4Payoff / task4P
  let hoursTask5 := (remainingC - task6Payoff) / task5P
  let hoursTask6 := task6Payoff / task6P
  hoursTask1 + hoursTask2 + hoursTask3 + hoursTask4 + hoursTask5 + hoursTask6

-- Given our specific problem conditions
theorem jake_needs_total_hours_to_pay_off_debts :
  total_hours_needed 150 200 250 60 80 100 15 12 20 10 25 30 30 40 60 = 20.1 :=
by
  sorry

end jake_needs_total_hours_to_pay_off_debts_l222_222728


namespace total_capacity_both_dressers_l222_222356

/-- Definition of drawers and capacity -/
def first_dresser_drawers : ℕ := 12
def first_dresser_capacity_per_drawer : ℕ := 8
def second_dresser_drawers : ℕ := 6
def second_dresser_capacity_per_drawer : ℕ := 10

/-- Theorem stating the total capacity of both dressers -/
theorem total_capacity_both_dressers :
  (first_dresser_drawers * first_dresser_capacity_per_drawer) +
  (second_dresser_drawers * second_dresser_capacity_per_drawer) = 156 :=
by sorry

end total_capacity_both_dressers_l222_222356


namespace sculpture_cost_in_chinese_yuan_l222_222657

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l222_222657


namespace smallest_n_l222_222328

theorem smallest_n (n : ℕ) : 
  (n % 6 = 4) ∧ (n % 7 = 2) ∧ (n > 20) → n = 58 :=
by
  sorry

end smallest_n_l222_222328


namespace large_A_exists_l222_222116

noncomputable def F_n (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem large_A_exists : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  ∀ a : ℕ, a ≤ 53590 → 
  F_n n6 (F_n n5 (F_n n4 (F_n n3 (F_n n2 (F_n n1 a))))) = 1 :=
by
  sorry

end large_A_exists_l222_222116


namespace OReilly_triple_8_49_x_l222_222054

def is_OReilly_triple (a b x : ℕ) : Prop :=
  (a : ℝ)^(1/3) + (b : ℝ)^(1/2) = x

theorem OReilly_triple_8_49_x (x : ℕ) (h : is_OReilly_triple 8 49 x) : x = 9 := by
  sorry

end OReilly_triple_8_49_x_l222_222054


namespace journey_distance_l222_222579

theorem journey_distance
  (total_time : ℝ)
  (speed1 speed2 : ℝ)
  (journey_time : total_time = 10)
  (speed1_val : speed1 = 21)
  (speed2_val : speed2 = 24) :
  ∃ D : ℝ, (D / 2 / speed1 + D / 2 / speed2 = total_time) ∧ D = 224 :=
by
  sorry

end journey_distance_l222_222579


namespace min_positive_value_l222_222992

theorem min_positive_value (c d : ℤ) (h : c > d) : 
  ∃ x : ℝ, x = (c + 2 * d) / (c - d) + (c - d) / (c + 2 * d) ∧ x = 2 :=
by {
  sorry
}

end min_positive_value_l222_222992


namespace max_pieces_with_three_cuts_l222_222639

def cake := Type

noncomputable def max_identical_pieces (cuts : ℕ) (max_cuts : ℕ) : ℕ :=
  if cuts = 3 ∧ max_cuts = 3 then 8 else sorry

theorem max_pieces_with_three_cuts : ∀ (c : cake), max_identical_pieces 3 3 = 8 :=
by
  intro c
  sorry

end max_pieces_with_three_cuts_l222_222639


namespace divisible_by_9_l222_222745

-- Definition of the sum of digits function S
def sum_of_digits (n : ℕ) : ℕ := sorry  -- Assume we have a function that sums the digits of n

theorem divisible_by_9 (a : ℕ) (h₁ : sum_of_digits a = sum_of_digits (2 * a)) 
  (h₂ : a % 9 = sum_of_digits a % 9) (h₃ : (2 * a) % 9 = sum_of_digits (2 * a) % 9) : 
  a % 9 = 0 :=
by
  sorry

end divisible_by_9_l222_222745


namespace general_formula_expression_of_k_l222_222772

noncomputable def sequence_a : ℕ → ℤ
| 0     => 0 
| 1     => 0 
| 2     => -6
| n + 2 => 2 * (sequence_a (n + 1)) - (sequence_a n)

theorem general_formula :
  ∀ n, sequence_a n = 2 * n - 10 := sorry

def sequence_k : ℕ → ℕ
| 0     => 0 
| 1     => 8 
| n + 1 => 3 * 2 ^ n + 5

theorem expression_of_k (n : ℕ) :
  sequence_k (n + 1) = 3 * 2 ^ n + 5 := sorry

end general_formula_expression_of_k_l222_222772


namespace isosceles_triangle_base_length_l222_222537

theorem isosceles_triangle_base_length (P B : ℕ) (hP : P = 13) (hB : B = 3) :
    ∃ S : ℕ, S ≠ 3 ∧ S = 3 :=
by
    sorry

end isosceles_triangle_base_length_l222_222537


namespace measure_of_angle_B_find_a_and_c_find_perimeter_l222_222998

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ) 
    (h : c / (b - a) = (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C)) 
    (cos_B : Real.cos B = -1 / 2) : B = 2 * Real.pi / 3 :=
by
  sorry

theorem find_a_and_c (a c A C : ℝ) (S : ℝ) 
    (h1 : Real.sin C = 2 * Real.sin A) (h2 : S = 2 * Real.sqrt 3) 
    (A' : a * c = 8) : a = 2 ∧ c = 4 :=
by
  sorry

theorem find_perimeter (a b c : ℝ) 
    (h1 : b = Real.sqrt 3) (h2 : a * c = 1) 
    (h3 : a + c = 2) : a + b + c = 2 + Real.sqrt 3 :=
by
  sorry

end measure_of_angle_B_find_a_and_c_find_perimeter_l222_222998


namespace range_of_a_l222_222797

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l222_222797


namespace find_a3_l222_222980

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_a3 {a : ℕ → ℤ} (d : ℤ) (h6 : a 6 = 6) (h9 : a 9 = 9) :
  (∃ d : ℤ, arithmetic_sequence a d) →
  a 3 = 3 :=
by
  intro h_arith_seq
  sorry

end find_a3_l222_222980


namespace sequence_properties_l222_222158

theorem sequence_properties
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2) ∧ (a 2 = 3) ∧
  (∀ n, a (2 * n - 1) = 2 * (2 * n - 1)) ∧
  (∀ n, a (2 * n) = 2 * 2 * n - 1) :=
by
  sorry

end sequence_properties_l222_222158


namespace vertical_asymptote_once_l222_222112

theorem vertical_asymptote_once (c : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + c) / (x^2 - x - 12) = (x^2 + 2*x + c) / ((x - 4) * (x + 3))) → 
  (c = -24 ∨ c = -3) :=
by 
  sorry

end vertical_asymptote_once_l222_222112


namespace vector_c_correct_l222_222078

theorem vector_c_correct (a b c : ℤ × ℤ) (h_a : a = (1, -3)) (h_b : b = (-2, 4))
    (h_condition : 4 • a + (3 • b - 2 • a) + c = (0, 0)) :
    c = (4, -6) :=
by 
  -- The proof steps go here, but we'll skip them with 'sorry' for now.
  sorry

end vector_c_correct_l222_222078


namespace angle_complementary_supplementary_l222_222381

theorem angle_complementary_supplementary (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle1 + angle3 = 180)
  (h3 : angle3 = 125) :
  angle2 = 35 :=
by 
  sorry

end angle_complementary_supplementary_l222_222381


namespace evaluate_fraction_l222_222217

theorem evaluate_fraction : ∃ p q : ℤ, gcd p q = 1 ∧ (2023 : ℤ) / (2022 : ℤ) - 2 * (2022 : ℤ) / (2023 : ℤ) = (p : ℚ) / (q : ℚ) ∧ p = -(2022^2 : ℤ) + 4045 :=
by
  sorry

end evaluate_fraction_l222_222217


namespace tank_fraction_after_adding_water_l222_222870

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (full_capacity : ℚ) 
  (added_water : ℚ) 
  (final_fraction : ℚ) 
  (h1 : initial_fraction = 3/4) 
  (h2 : full_capacity = 56) 
  (h3 : added_water = 7) 
  (h4 : final_fraction = (initial_fraction * full_capacity + added_water) / full_capacity) : 
  final_fraction = 7 / 8 := 
by 
  sorry

end tank_fraction_after_adding_water_l222_222870


namespace set_has_one_element_iff_double_root_l222_222510

theorem set_has_one_element_iff_double_root (k : ℝ) :
  (∃ x, ∀ y, y^2 - k*y + 1 = 0 ↔ y = x) ↔ k = 2 ∨ k = -2 :=
by
  sorry

end set_has_one_element_iff_double_root_l222_222510


namespace lucy_l222_222621

theorem lucy's_age 
  (L V: ℕ)
  (h1: L - 5 = 3 * (V - 5))
  (h2: L + 10 = 2 * (V + 10)) :
  L = 50 :=
by
  sorry

end lucy_l222_222621


namespace geometric_sequence_common_ratio_and_general_formula_l222_222762

variable (a : ℕ → ℝ)

theorem geometric_sequence_common_ratio_and_general_formula (h₁ : a 1 = 1) (h₃ : a 3 = 4) : 
  (∃ q : ℝ, q = 2 ∨ q = -2 ∧ (∀ n : ℕ, a n = 2^(n-1) ∨ a n = (-2)^(n-1))) := 
by
  sorry

end geometric_sequence_common_ratio_and_general_formula_l222_222762


namespace monotonically_increasing_interval_l222_222344

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem monotonically_increasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → (f x > f 0) := 
by
  sorry

end monotonically_increasing_interval_l222_222344


namespace find_n_l222_222367

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (S_odd : ℝ) (S_even : ℝ)
  (h1 : ∀ k, a (2 * k - 1) = a 0 + (2 * k - 2) * d)
  (h2 : ∀ k, a (2 * k) = a 1 + (2 * k - 1) * d)
  (h3 : 2 * n + 1 = n + (n + 1))
  (h4 : S_odd = (n + 1) * (a 0 + n * d))
  (h5 : S_even = n * (a 1 + (n - 1) * d))
  (h6 : S_odd = 4)
  (h7 : S_even = 3) : n = 3 :=
by
  sorry

end find_n_l222_222367


namespace geraldine_banana_count_l222_222214

variable (b : ℕ) -- the number of bananas Geraldine ate on June 1

theorem geraldine_banana_count 
    (h1 : (5 * b + 80 = 150)) 
    : (b + 32 = 46) :=
by
  sorry

end geraldine_banana_count_l222_222214


namespace centroid_of_quadrant_arc_l222_222320

def circle_equation (R : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = R^2
def density (ρ₀ x y : ℝ) : ℝ := ρ₀ * x * y

theorem centroid_of_quadrant_arc (R ρ₀ : ℝ) :
  (∃ x y, circle_equation R x y ∧ x ≥ 0 ∧ y ≥ 0) →
  ∃ x_c y_c, x_c = 2 * R / 3 ∧ y_c = 2 * R / 3 :=
sorry

end centroid_of_quadrant_arc_l222_222320


namespace quadratic_inequality_min_value_l222_222719

noncomputable def min_value (a b: ℝ) : ℝ := 2 * a^2 + b^2

theorem quadratic_inequality_min_value
  (a b: ℝ) (hx: ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (x0: ℝ) (hx0: a * x0^2 + 2 * x0 + b = 0) :
  a > b → min_value a b = 2 * Real.sqrt 2 := 
sorry

end quadratic_inequality_min_value_l222_222719


namespace arithmetic_mean_of_fractions_l222_222989

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l222_222989


namespace division_of_decimals_l222_222114

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l222_222114


namespace pentagon_area_l222_222956

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℕ := 
  let area_triangle := (1/2) * a * b
  let area_trapezoid := (1/2) * (c + e) * d
  area_triangle + area_trapezoid

theorem pentagon_area : area_of_pentagon 18 25 30 28 25 = 995 :=
by sorry

end pentagon_area_l222_222956


namespace number_of_B_students_l222_222962

theorem number_of_B_students (x : ℝ) (h1 : 0.8 * x + x + 1.2 * x = 40) : x = 13 :=
  sorry

end number_of_B_students_l222_222962


namespace simplify_fraction_l222_222391

-- Define factorial (or use the existing factorial definition if available in Mathlib)
def fact : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Problem statement
theorem simplify_fraction :
  (5 * fact 7 + 35 * fact 6) / fact 8 = 5 / 4 := by
  sorry

end simplify_fraction_l222_222391


namespace people_per_bus_l222_222172

def num_vans : ℝ := 6.0
def num_buses : ℝ := 8.0
def people_per_van : ℝ := 6.0
def extra_people : ℝ := 108.0

theorem people_per_bus :
  let people_vans := num_vans * people_per_van
  let people_buses := people_vans + extra_people
  let people_per_bus := people_buses / num_buses
  people_per_bus = 18.0 :=
by 
  sorry

end people_per_bus_l222_222172


namespace calculate_savings_l222_222882

theorem calculate_savings :
  let plane_cost : ℕ := 600
  let boat_cost : ℕ := 254
  plane_cost - boat_cost = 346 := by
    let plane_cost : ℕ := 600
    let boat_cost : ℕ := 254
    sorry

end calculate_savings_l222_222882


namespace number_of_triangles_l222_222819

theorem number_of_triangles (m : ℕ) (h : m > 0) :
  ∃ n : ℕ, n = (m * (m + 1)) / 2 :=
by sorry

end number_of_triangles_l222_222819


namespace radius_of_circle_l222_222085

theorem radius_of_circle (d : ℝ) (h : d = 22) : (d / 2) = 11 := by
  sorry

end radius_of_circle_l222_222085


namespace lawnmower_blades_l222_222110

theorem lawnmower_blades (B : ℤ) (h : 8 * B + 7 = 39) : B = 4 :=
by 
  sorry

end lawnmower_blades_l222_222110


namespace min_x_plus_3y_l222_222149

noncomputable def minimum_x_plus_3y (x y : ℝ) : ℝ :=
  if h : (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) then x + 3*y else 0

theorem min_x_plus_3y : ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) → x + 3*y = 6 :=
by
  intros x y h
  sorry

end min_x_plus_3y_l222_222149


namespace smallest_positive_period_of_f_l222_222368

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi / 2 :=
by
  sorry

end smallest_positive_period_of_f_l222_222368


namespace ford_younger_than_christopher_l222_222835

variable (G C F Y : ℕ)

-- Conditions
axiom h1 : G = C + 8
axiom h2 : F = C - Y
axiom h3 : G + C + F = 60
axiom h4 : C = 18

-- Target statement
theorem ford_younger_than_christopher : Y = 2 :=
sorry

end ford_younger_than_christopher_l222_222835


namespace three_a_in_S_implies_a_in_S_l222_222072

def S := {n | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : 3 * a ∈ S) : a ∈ S := 
sorry

end three_a_in_S_implies_a_in_S_l222_222072


namespace kevin_food_expenditure_l222_222125

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l222_222125


namespace distinct_painted_cubes_l222_222136

-- Define the context of the problem
def num_faces : ℕ := 6

def total_paintings : ℕ := num_faces.factorial

def num_rotations : ℕ := 24

-- Statement of the theorem
theorem distinct_painted_cubes (h1 : total_paintings = 720) (h2 : num_rotations = 24) : 
  total_paintings / num_rotations = 30 := by
  sorry

end distinct_painted_cubes_l222_222136


namespace thermos_count_l222_222605

theorem thermos_count
  (total_gallons : ℝ)
  (pints_per_gallon : ℝ)
  (thermoses_drunk_by_genevieve : ℕ)
  (pints_drunk_by_genevieve : ℝ)
  (total_pints : ℝ) :
  total_gallons * pints_per_gallon = total_pints ∧
  pints_drunk_by_genevieve / thermoses_drunk_by_genevieve = 2 →
  total_pints / 2 = 18 :=
by
  intros h
  have := h.2
  sorry

end thermos_count_l222_222605


namespace find_x_l222_222280

theorem find_x (x y z : ℝ) (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 :=
sorry

end find_x_l222_222280


namespace max_smoothie_servings_l222_222655

-- Define the constants based on the problem conditions
def servings_per_recipe := 4
def bananas_per_recipe := 3
def yogurt_per_recipe := 1 -- cup
def honey_per_recipe := 2 -- tablespoons
def strawberries_per_recipe := 2 -- cups

-- Define the total amount of ingredients Lynn has
def total_bananas := 12
def total_yogurt := 6 -- cups
def total_honey := 16 -- tablespoons (since 1 cup = 16 tablespoons)
def total_strawberries := 8 -- cups

-- Define the calculation for the number of servings each ingredient can produce
def servings_from_bananas := (total_bananas / bananas_per_recipe) * servings_per_recipe
def servings_from_yogurt := (total_yogurt / yogurt_per_recipe) * servings_per_recipe
def servings_from_honey := (total_honey / honey_per_recipe) * servings_per_recipe
def servings_from_strawberries := (total_strawberries / strawberries_per_recipe) * servings_per_recipe

-- Define the minimum number of servings that can be made based on all ingredients
def max_servings := min servings_from_bananas (min servings_from_yogurt (min servings_from_honey servings_from_strawberries))

theorem max_smoothie_servings : max_servings = 16 :=
by
  sorry

end max_smoothie_servings_l222_222655


namespace impossible_to_place_integers_35x35_l222_222841

theorem impossible_to_place_integers_35x35 (f : Fin 35 → Fin 35 → ℤ) :
  (∀ i j, abs (f i j - f (i + 1) j) ≤ 18 ∧ abs (f i j - f i (j + 1)) ≤ 18) →
  ∃ i j, i ≠ j ∧ f i j = f i j → False :=
by sorry

end impossible_to_place_integers_35x35_l222_222841


namespace rectangle_perimeter_l222_222896

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) : 2 * (L + B) = 186 :=
by
  sorry

end rectangle_perimeter_l222_222896


namespace intersection_eq_l222_222798

open Set

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B : Set ℝ := { -1, 2, 3, 6 }

-- State the proof problem
theorem intersection_eq : A ∩ B = {2, 3} := 
by 
-- placeholder for the proof steps
sorry

end intersection_eq_l222_222798


namespace problem_equivalent_l222_222732

theorem problem_equivalent :
  2^1998 - 2^1997 - 2^1996 + 2^1995 = 3 * 2^1995 :=
by
  sorry

end problem_equivalent_l222_222732


namespace possibleValuesOfSum_l222_222248

noncomputable def symmetricMatrixNonInvertible (x y z : ℝ) : Prop := 
  -(x + y + z) * ( x^2 + y^2 + z^2 - x * y - x * z - y * z ) = 0

theorem possibleValuesOfSum (x y z : ℝ) (h : symmetricMatrixNonInvertible x y z) :
  ∃ v : ℝ, v = -3 ∨ v = 3 / 2 := 
sorry

end possibleValuesOfSum_l222_222248


namespace sports_club_members_l222_222027

theorem sports_club_members (N B T : ℕ) (h_total : N = 30) (h_badminton : B = 18) (h_tennis : T = 19) (h_neither : N - (B + T - 9) = 2) : B + T - 9 = 28 :=
by
  sorry

end sports_club_members_l222_222027


namespace find_number_l222_222599

theorem find_number (x : ℝ) (h : (((18 + x) / 3 + 10) / 5 = 4)) : x = 12 :=
by
  sorry

end find_number_l222_222599


namespace Karlson_cannot_prevent_Baby_getting_one_fourth_l222_222500

theorem Karlson_cannot_prevent_Baby_getting_one_fourth 
  (a : ℝ) (h : a > 0) (K : ℝ × ℝ) (hK : 0 < K.1 ∧ K.1 < a ∧ 0 < K.2 ∧ K.2 < a) :
  ∀ (O : ℝ × ℝ) (cut1 cut2 : ℝ), 
    ((O.1 = a/2) ∧ (O.2 = a/2) ∧ (cut1 = K.1 ∧ cut1 = a ∨ cut1 = K.2 ∧ cut1 = a) ∧ 
                             (cut2 = K.1 ∧ cut2 = a ∨ cut2 = K.2 ∧ cut2 = a)) →
  ∃ (piece : ℝ), piece ≥ a^2 / 4 :=
by
  sorry

end Karlson_cannot_prevent_Baby_getting_one_fourth_l222_222500


namespace even_fn_increasing_max_val_l222_222726

variable {f : ℝ → ℝ}

theorem even_fn_increasing_max_val (h_even : ∀ x, f x = f (-x))
    (h_inc_0_5 : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 5 → f x ≤ f y)
    (h_dec_5_inf : ∀ x y, 5 ≤ x → x ≤ y → f y ≤ f x)
    (h_f5 : f 5 = 2) :
    (∀ x y, -5 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y) ∧ (∀ x, -5 ≤ x → x ≤ 0 → f x ≤ 2) :=
by
    sorry

end even_fn_increasing_max_val_l222_222726


namespace inequality_solution_set_l222_222900

theorem inequality_solution_set :
  {x : ℝ | (x - 3) / (x + 2) ≤ 0} = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by
  sorry

end inequality_solution_set_l222_222900


namespace wire_cut_perimeter_equal_l222_222463

theorem wire_cut_perimeter_equal (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 4 * (a / 4) = 8 * (b / 8)) :
  a / b = 1 :=
sorry

end wire_cut_perimeter_equal_l222_222463


namespace paving_stones_correct_l222_222985

def paving_stone_area : ℕ := 3 * 2
def courtyard_breadth : ℕ := 6
def number_of_paving_stones : ℕ := 15
def courtyard_length : ℕ := 15

theorem paving_stones_correct : 
  number_of_paving_stones * paving_stone_area = courtyard_length * courtyard_breadth :=
by
  sorry

end paving_stones_correct_l222_222985


namespace total_time_equiv_7_75_l222_222052

def acclimation_period : ℝ := 1
def learning_basics : ℝ := 2
def research_time_without_sabbatical : ℝ := learning_basics + 0.75 * learning_basics
def sabbatical : ℝ := 0.5
def research_time_with_sabbatical : ℝ := research_time_without_sabbatical + sabbatical
def dissertation_without_conference : ℝ := 0.5 * acclimation_period
def conference : ℝ := 0.25
def dissertation_with_conference : ℝ := dissertation_without_conference + conference
def total_time : ℝ := acclimation_period + learning_basics + research_time_with_sabbatical + dissertation_with_conference

theorem total_time_equiv_7_75 : total_time = 7.75 := by
  sorry

end total_time_equiv_7_75_l222_222052


namespace binomial_minus_floor_divisible_by_seven_l222_222994

theorem binomial_minus_floor_divisible_by_seven (n : ℕ) (h : n > 7) :
  ((Nat.choose n 7 : ℤ) - ⌊(n : ℤ) / 7⌋) % 7 = 0 :=
  sorry

end binomial_minus_floor_divisible_by_seven_l222_222994


namespace no_common_root_l222_222699

theorem no_common_root (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) := 
sorry

end no_common_root_l222_222699


namespace boys_camp_problem_l222_222185

noncomputable def total_boys_in_camp : ℝ :=
  let schoolA_fraction := 0.20
  let science_fraction := 0.30
  let non_science_boys := 63
  let non_science_fraction := 1 - science_fraction
  let schoolA_boys := (non_science_boys / non_science_fraction)
  schoolA_boys / schoolA_fraction

theorem boys_camp_problem : total_boys_in_camp = 450 := 
by
  sorry

end boys_camp_problem_l222_222185


namespace total_height_of_buildings_l222_222358

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l222_222358


namespace matchstick_triangles_l222_222492

theorem matchstick_triangles (perimeter : ℕ) (h_perimeter : perimeter = 30) : 
  ∃ n : ℕ, n = 17 ∧ 
  (∀ a b c : ℕ, a + b + c = perimeter → a > 0 → b > 0 → c > 0 → 
                a + b > c ∧ a + c > b ∧ b + c > a → 
                a ≤ b ∧ b ≤ c → n = 17) := 
sorry

end matchstick_triangles_l222_222492


namespace unique_rectangle_exists_l222_222674

theorem unique_rectangle_exists (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 4 :=
by
  sorry

end unique_rectangle_exists_l222_222674


namespace prime_cube_difference_l222_222003

theorem prime_cube_difference (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (eqn : p^3 - q^3 = 11 * r) : 
  (p = 13 ∧ q = 2 ∧ r = 199) :=
sorry

end prime_cube_difference_l222_222003


namespace batsman_average_after_25th_innings_l222_222419

theorem batsman_average_after_25th_innings (A : ℝ) (h_pre_avg : (25 * (A + 3)) = (24 * A + 80))
  : A + 3 = 8 := 
by
  sorry

end batsman_average_after_25th_innings_l222_222419


namespace number_of_six_digit_integers_l222_222968

-- Define the problem conditions
def digits := [1, 1, 3, 3, 7, 8]

-- State the theorem
theorem number_of_six_digit_integers : 
  (List.permutations digits).length = 180 := 
by sorry

end number_of_six_digit_integers_l222_222968


namespace cosh_le_exp_sqr_l222_222587

open Real

theorem cosh_le_exp_sqr {x k : ℝ} : (∀ x : ℝ, cosh x ≤ exp (k * x^2)) ↔ k ≥ 1/2 :=
sorry

end cosh_le_exp_sqr_l222_222587


namespace pratyya_payel_min_difference_l222_222915

theorem pratyya_payel_min_difference (n m : ℕ) (h : n > m ∧ n - m ≥ 4) :
  ∀ t : ℕ, (2^(t+1) * n - 2^(t+1)) > 2^(t+1) * m + 2^(t+1) :=
by
  sorry

end pratyya_payel_min_difference_l222_222915


namespace dress_designs_count_l222_222947

-- Define the number of colors, fabric types, and patterns
def num_colors : Nat := 3
def num_fabric_types : Nat := 4
def num_patterns : Nat := 3

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_fabric_types * num_patterns

-- Define the theorem to prove the equivalence
theorem dress_designs_count :
  total_dress_designs = 36 :=
by
  -- This is to show the theorem's structure; proof will be added here.
  sorry

end dress_designs_count_l222_222947


namespace problem_statement_l222_222020

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := { x : ℝ | abs x < 2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

theorem problem_statement :
  compl M ∪ compl N = Iic (-1) ∪ Ici 2 :=
by {
  sorry
}

end problem_statement_l222_222020


namespace point_M_coordinates_l222_222398

/- Define the conditions -/

def isInFourthQuadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distanceToXAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.2 = d

def distanceToYAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.1 = d

/- Write the Lean theorem statement -/

theorem point_M_coordinates :
  ∀ (M : ℝ × ℝ), isInFourthQuadrant M ∧ distanceToXAxis M 3 ∧ distanceToYAxis M 4 → M = (4, -3) :=
by
  intro M
  sorry

end point_M_coordinates_l222_222398


namespace range_of_x_l222_222530

theorem range_of_x (a : ℝ) (x : ℝ) (h₁ : a = 1) (h₂ : (x - a) * (x - 3 * a) < 0) (h₃ : 2 < x ∧ x ≤ 3) : 2 < x ∧ x < 3 :=
by sorry

end range_of_x_l222_222530


namespace gcd_18222_24546_66364_eq_2_l222_222067

/-- Definition of three integers a, b, c --/
def a : ℕ := 18222 
def b : ℕ := 24546
def c : ℕ := 66364

/-- Proof of the gcd of the three integers being 2 --/
theorem gcd_18222_24546_66364_eq_2 : Nat.gcd (Nat.gcd a b) c = 2 := by
  sorry

end gcd_18222_24546_66364_eq_2_l222_222067


namespace James_trout_pounds_l222_222464

def pounds_trout (T : ℝ) : Prop :=
  let salmon := 1.5 * T
  let tuna := 2 * T
  T + salmon + tuna = 1100

theorem James_trout_pounds :
  ∃ T : ℝ, pounds_trout T ∧ T = 244 :=
sorry

end James_trout_pounds_l222_222464


namespace cannon_hit_probability_l222_222939

theorem cannon_hit_probability {P2 P3 : ℝ} (hP1 : 0.5 <= P2) (hP2 : P2 = 0.2) (hP3 : P3 = 0.3) (h_none_hit : (1 - 0.5) * (1 - P2) * (1 - P3) = 0.28) :
  0.5 = 0.5 :=
by sorry

end cannon_hit_probability_l222_222939


namespace eq_has_infinite_solutions_l222_222987

theorem eq_has_infinite_solutions (b : ℝ) (x : ℝ) :
  5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 := by
sorry

end eq_has_infinite_solutions_l222_222987


namespace mode_is_37_median_is_36_l222_222583

namespace ProofProblem

def data_set : List ℕ := [34, 35, 36, 34, 36, 37, 37, 36, 37, 37]

def mode (l : List ℕ) : ℕ := sorry -- Implementing a mode function

def median (l : List ℕ) : ℕ := sorry -- Implementing a median function

theorem mode_is_37 : mode data_set = 37 := 
  by 
    sorry -- Proof of mode

theorem median_is_36 : median data_set = 36 := 
  by
    sorry -- Proof of median

end ProofProblem

end mode_is_37_median_is_36_l222_222583


namespace find_n_for_k_eq_1_l222_222515

theorem find_n_for_k_eq_1 (n : ℤ) (h : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 1)) : n = 5 := 
by 
  sorry

end find_n_for_k_eq_1_l222_222515


namespace infinite_integers_repr_l222_222986

theorem infinite_integers_repr : ∀ (k : ℕ), k > 1 →
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
by
  intros k hk
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  sorry

end infinite_integers_repr_l222_222986


namespace total_notebooks_distributed_l222_222976

/-- Define the parameters for children in Class A and Class B and the conditions given. -/
def ClassAChildren : ℕ := 64
def ClassBChildren : ℕ := 13

/-- Define the conditions as per the problem -/
def notebooksPerChildInClassA (A : ℕ) : ℕ := A / 8
def notebooksPerChildInClassB (A : ℕ) : ℕ := 2 * A
def totalChildrenClasses (A B : ℕ) : ℕ := A + B
def totalChildrenCondition (A : ℕ) : ℕ := 6 * A / 5

/-- Theorem to state the number of notebooks distributed between the two classes -/
theorem total_notebooks_distributed (A : ℕ) (B : ℕ) (H : A = 64) (H1 : B = 13) : 
  (A * (A / 8) + B * (2 * A)) = 2176 := by
  -- Conditions from the problem
  have conditionA : A = 64 := H
  have conditionB : B = 13 := H1
  have classA_notebooks : ℕ := (notebooksPerChildInClassA A) * A
  have classB_notebooks : ℕ := (notebooksPerChildInClassB A) * B
  have total_notebooks : ℕ := classA_notebooks + classB_notebooks
  -- Proof that total notebooks equals 2176
  sorry

end total_notebooks_distributed_l222_222976


namespace simplify_fraction_subtraction_l222_222258

theorem simplify_fraction_subtraction : (7 / 3) - (5 / 6) = 3 / 2 := by
  sorry

end simplify_fraction_subtraction_l222_222258


namespace median_circumradius_altitude_inequality_l222_222077

variable (h R m_a m_b m_c : ℝ)

-- Define the condition for the lengths of the medians and other related parameters
-- m_a, m_b, m_c are medians, R is the circumradius, h is the greatest altitude

theorem median_circumradius_altitude_inequality :
  m_a + m_b + m_c ≤ 3 * R + h :=
sorry

end median_circumradius_altitude_inequality_l222_222077


namespace books_bought_l222_222482

theorem books_bought (cost_crayons cost_calculators total_money cost_per_bag bags_bought cost_per_book remaining_money books_bought : ℕ) 
  (h1: cost_crayons = 5 * 5)
  (h2: cost_calculators = 3 * 5)
  (h3: total_money = 200)
  (h4: cost_per_bag = 10)
  (h5: bags_bought = 11)
  (h6: remaining_money = total_money - (cost_crayons + cost_calculators) - (bags_bought * cost_per_bag)) :
  books_bought = remaining_money / cost_per_book → books_bought = 10 :=
by
  sorry

end books_bought_l222_222482


namespace katharina_order_is_correct_l222_222242

-- Define the mixed up order around a circle starting with L
def mixedUpOrder : List Char := ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

-- Define the positions and process of Jaxon's list generation
def jaxonList : List Nat := [1, 4, 7, 3, 8, 5, 2, 6]

-- Define the resulting order from Jaxon's process
def resultingOrder (initialList : List Char) (positions : List Nat) : List Char :=
  positions.map (λ i => initialList.get! (i - 1))

-- Define the function to prove Katharina's order
theorem katharina_order_is_correct :
  resultingOrder mixedUpOrder jaxonList = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] :=
by
  -- Proof omitted
  sorry

end katharina_order_is_correct_l222_222242


namespace div_pow_eq_l222_222658

theorem div_pow_eq (n : ℕ) (h : n = 16 ^ 2023) : n / 4 = 4 ^ 4045 :=
by
  rw [h]
  sorry

end div_pow_eq_l222_222658


namespace fraction_meaningful_l222_222022

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end fraction_meaningful_l222_222022


namespace compute_expression_l222_222613
-- Start with importing math library utilities for linear algebra and dot product

-- Define vector 'a' and 'b' in Lean
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

-- Define dot product operation 
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Define the expression and the theorem
theorem compute_expression : dot_product ((2 * a.1 + b.1, 2 * a.2 + b.2)) a = 1 :=
by
  -- Insert the proof steps here
  sorry

end compute_expression_l222_222613


namespace outfit_combination_count_l222_222477

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end outfit_combination_count_l222_222477


namespace spaceship_journey_time_l222_222487

theorem spaceship_journey_time
  (initial_travel_1 : ℕ)
  (first_break : ℕ)
  (initial_travel_2 : ℕ)
  (second_break : ℕ)
  (travel_per_segment : ℕ)
  (break_per_segment : ℕ)
  (total_break_time : ℕ)
  (remaining_break_time : ℕ)
  (num_segments : ℕ)
  (total_travel_time : ℕ)
  (total_time : ℕ) :
  initial_travel_1 = 10 →
  first_break = 3 →
  initial_travel_2 = 10 →
  second_break = 1 →
  travel_per_segment = 11 →
  break_per_segment = 1 →
  total_break_time = 8 →
  remaining_break_time = total_break_time - (first_break + second_break) →
  num_segments = remaining_break_time / break_per_segment →
  total_travel_time = initial_travel_1 + initial_travel_2 + (num_segments * travel_per_segment) →
  total_time = total_travel_time + total_break_time →
  total_time = 72 :=
by
  intros
  sorry

end spaceship_journey_time_l222_222487


namespace cylinder_height_l222_222888

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h_cond : SA = 2 * π * r^2 + 2 * π * r * h) 
  (r_eq : r = 3) (SA_eq : SA = 27 * π) : h = 3 / 2 :=
by
  sorry

end cylinder_height_l222_222888


namespace area_of_right_triangle_l222_222792

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l222_222792


namespace sophie_clothes_expense_l222_222628

theorem sophie_clothes_expense :
  let initial_fund := 260
  let shirt_cost := 18.50
  let trousers_cost := 63
  let num_shirts := 2
  let num_remaining_clothes := 4
  let total_spent := num_shirts * shirt_cost + trousers_cost
  let remaining_amount := initial_fund - total_spent
  let individual_item_cost := remaining_amount / num_remaining_clothes
  individual_item_cost = 40 := 
by 
  sorry

end sophie_clothes_expense_l222_222628


namespace articles_count_l222_222123

noncomputable def cost_price_per_article : ℝ := 1
noncomputable def selling_price_per_article (x : ℝ) : ℝ := x / 16
noncomputable def profit : ℝ := 0.50

theorem articles_count (x : ℝ) (h1 : cost_price_per_article * x = selling_price_per_article x * 16)
                       (h2 : selling_price_per_article 16 = cost_price_per_article * (1 + profit)) :
  x = 24 :=
by
  sorry

end articles_count_l222_222123


namespace biggest_number_in_ratio_l222_222065

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l222_222065


namespace find_f_2002_l222_222229

-- Definitions based on conditions
variable {R : Type} [CommRing R] [NoZeroDivisors R]

-- Condition 1: f is an even function.
def even_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = f x

-- Condition 2: f(2) = 0
def f_value_at_two (f : R → R) : Prop :=
  f 2 = 0

-- Condition 3: g is an odd function.
def odd_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = -g x

-- Condition 4: g(x) = f(x-1)
def g_equals_f_shifted (f g : R → R) : Prop :=
  ∀ x : R, g x = f (x - 1)

-- The main proof problem
theorem find_f_2002 (f g : R → R)
  (hf : even_function f)
  (hf2 : f_value_at_two f)
  (hg : odd_function g)
  (hgf : g_equals_f_shifted f g) :
  f 2002 = 0 :=
sorry

end find_f_2002_l222_222229


namespace red_apples_count_l222_222334

theorem red_apples_count
  (r y g : ℕ)
  (h1 : r = y)
  (h2 : g = 2 * r)
  (h3 : r + y + g = 28) : r = 7 :=
sorry

end red_apples_count_l222_222334


namespace part_1_solution_part_2_solution_l222_222173

def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem part_1_solution (x : ℝ) : f x < 3 ↔ -4 / 3 < x ∧ x < 0 :=
by
  sorry

theorem part_2_solution (a : ℝ) : (∀ x, ¬ (f x < a)) → a ≤ 2 :=
by
  sorry

end part_1_solution_part_2_solution_l222_222173


namespace sum_mod_9_l222_222430

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9_l222_222430


namespace geometric_sequence_sum_l222_222250

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = r * a n)
  (h2 : 0 < r)
  (h3 : a 1 = 3)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l222_222250


namespace circles_intersection_distance_squared_l222_222752

open Real

-- Definitions of circles
def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 25

def circle2 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 9

-- Theorem to prove
theorem circles_intersection_distance_squared :
  ∃ A B : (ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B) ∧
  (dist A B)^2 = 675 / 49 :=
sorry

end circles_intersection_distance_squared_l222_222752


namespace intersection_complement_M_N_l222_222363

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_complement_M_N :
  (U \ M) ∩ N = {-3, -4} :=
by {
  sorry
}

end intersection_complement_M_N_l222_222363


namespace average_weight_ten_students_l222_222331

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l222_222331


namespace alice_daily_savings_l222_222690

theorem alice_daily_savings :
  ∀ (d total_days : ℕ) (dime_value : ℝ),
  d = 4 → total_days = 40 → dime_value = 0.10 →
  (d * dime_value) / total_days = 0.01 :=
by
  intros d total_days dime_value h_d h_total_days h_dime_value
  sorry

end alice_daily_savings_l222_222690


namespace negation_of_p_l222_222828

def p : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0

theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0 :=
by
  sorry

end negation_of_p_l222_222828


namespace problem_I_problem_II_l222_222725

open Set

-- Definitions of the sets A and B, and their intersections would be needed
def A := {x : ℝ | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 3 * a}

-- (I) When a = 1, find A ∩ B
theorem problem_I : A ∩ (B 1) = {x : ℝ | (2 ≤ x ∧ x ≤ 3) ∨ x = 1} := by
  sorry

-- (II) When A ∩ B = B, find the range of a
theorem problem_II : {a : ℝ | a > 0 ∧ ∀ x, x ∈ B a → x ∈ A} = {a : ℝ | (0 < a ∧ a ≤ 1 / 3) ∨ a ≥ 2} := by
  sorry

end problem_I_problem_II_l222_222725


namespace find_value_of_c_l222_222309

noncomputable def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem find_value_of_c (b c : ℝ) 
    (h1 : parabola b c 1 = 2)
    (h2 : parabola b c 5 = 2) :
    c = 7 :=
by
  sorry

end find_value_of_c_l222_222309


namespace fourth_term_sum_eq_40_l222_222758

theorem fourth_term_sum_eq_40 : 3^0 + 3^1 + 3^2 + 3^3 = 40 := by
  sorry

end fourth_term_sum_eq_40_l222_222758


namespace max_true_statements_l222_222276

theorem max_true_statements {p q : ℝ} (hp : p > 0) (hq : q < 0) :
  ∀ (s1 s2 s3 s4 s5 : Prop), 
  s1 = (1 / p > 1 / q) →
  s2 = (p^3 > q^3) →
  s3 = (p^2 < q^2) →
  s4 = (p > 0) →
  s5 = (q < 0) →
  s1 ∧ s2 ∧ s4 ∧ s5 ∧ ¬s3 → 
  ∃ m : ℕ, m = 4 := 
by {
  sorry
}

end max_true_statements_l222_222276


namespace num_solutions_3x_plus_2y_eq_806_l222_222571

theorem num_solutions_3x_plus_2y_eq_806 :
  (∃ y : ℕ, ∃ x : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 806) ∧
  ((∃ t : ℤ, x = 268 - 2 * t ∧ y = 1 + 3 * t) ∧ (∃ t : ℤ, 0 ≤ t ∧ t ≤ 133)) :=
sorry

end num_solutions_3x_plus_2y_eq_806_l222_222571


namespace prime_intersect_even_l222_222384

-- Definitions for prime numbers and even numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Sets P and Q
def P : Set ℕ := { n | is_prime n }
def Q : Set ℕ := { n | is_even n }

-- Proof statement
theorem prime_intersect_even : P ∩ Q = {2} :=
by
  sorry

end prime_intersect_even_l222_222384


namespace range_of_7a_minus_5b_l222_222689

theorem range_of_7a_minus_5b (a b : ℝ) (h1 : 5 ≤ a - b ∧ a - b ≤ 27) (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  36 ≤ 7 * a - 5 * b ∧ 7 * a - 5 * b ≤ 192 :=
sorry

end range_of_7a_minus_5b_l222_222689


namespace raptors_points_l222_222001

theorem raptors_points (x y z : ℕ) (h1 : x + y + z = 48) (h2 : x - y = 18) :
  (z = 0 → y = 15) ∧
  (z = 12 → y = 9) ∧
  (z = 18 → y = 6) ∧
  (z = 30 → y = 0) :=
by sorry

end raptors_points_l222_222001


namespace sean_has_45_whistles_l222_222063

variable (Sean Charles : ℕ)

def sean_whistles (Charles : ℕ) : ℕ :=
  Charles + 32

theorem sean_has_45_whistles
    (Charles_whistles : Charles = 13) 
    (Sean_whistles_condition : Sean = sean_whistles Charles) :
    Sean = 45 := by
  sorry

end sean_has_45_whistles_l222_222063


namespace Alyosha_result_divisible_by_S_l222_222458

variable (a b S x y : ℤ)
variable (h1 : x + y = S)
variable (h2 : S ∣ a * x + b * y)

theorem Alyosha_result_divisible_by_S :
  S ∣ b * x + a * y :=
sorry

end Alyosha_result_divisible_by_S_l222_222458


namespace car_miles_per_gallon_in_city_l222_222850

-- Define the conditions and the problem
theorem car_miles_per_gallon_in_city :
  ∃ C H T : ℝ, 
    H = 462 / T ∧ 
    C = 336 / T ∧ 
    C = H - 12 ∧ 
    C = 32 :=
by
  sorry

end car_miles_per_gallon_in_city_l222_222850


namespace monotone_increasing_intervals_exists_x0_implies_p_l222_222497

noncomputable def f (x : ℝ) := 6 * Real.log x + x ^ 2 - 8 * x
noncomputable def g (x : ℝ) (p : ℝ) := p / x + x ^ 2

theorem monotone_increasing_intervals :
  (∀ x, (0 < x ∧ x ≤ 1) → ∃ ε > 0, ∀ y, x < y → f y > f x) ∧
  (∀ x, (3 ≤ x) → ∃ ε > 0, ∀ y, x < y → f y > f x) := by
  sorry

theorem exists_x0_implies_p :
  (∃ x0, 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ f x0 > g x0 p) → p < -8 := by
  sorry

end monotone_increasing_intervals_exists_x0_implies_p_l222_222497


namespace sequence_inequality_l222_222357

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_condition : ∀ n m, a (n + m) ≤ a n + a m) :
  ∀ n m, n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := by
  sorry

end sequence_inequality_l222_222357


namespace trigonometric_fraction_value_l222_222688

theorem trigonometric_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end trigonometric_fraction_value_l222_222688


namespace border_pieces_is_75_l222_222507

-- Definitions based on conditions
def total_pieces : Nat := 500
def trevor_pieces : Nat := 105
def joe_pieces : Nat := 3 * trevor_pieces
def missing_pieces : Nat := 5

-- Number of border pieces
def border_pieces : Nat := total_pieces - missing_pieces - (trevor_pieces + joe_pieces)

-- Theorem statement
theorem border_pieces_is_75 : border_pieces = 75 :=
by
  -- Proof goes here
  sorry

end border_pieces_is_75_l222_222507


namespace don_walking_speed_l222_222475

theorem don_walking_speed 
  (distance_between_homes : ℝ)
  (cara_walking_speed : ℝ)
  (cara_distance_before_meeting : ℝ)
  (time_don_starts_after_cara : ℝ)
  (total_distance : distance_between_homes = 45)
  (cara_speed : cara_walking_speed = 6)
  (cara_distance : cara_distance_before_meeting = 30)
  (time_after_cara : time_don_starts_after_cara = 2) :
  ∃ (v : ℝ), v = 5 := by
    sorry

end don_walking_speed_l222_222475


namespace arithmetic_progression_sum_l222_222961

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  a + 10 * d = 5.25 → 
  a + 6 * d = 3.25 → 
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 56.25 → 
  n = 15 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_progression_sum_l222_222961


namespace clive_change_l222_222066

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l222_222066


namespace problem1_problem2_l222_222848

noncomputable section

theorem problem1 :
  (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 :=
  sorry

theorem problem2 :
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 :=
  sorry

end problem1_problem2_l222_222848


namespace guests_did_not_respond_l222_222434

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end guests_did_not_respond_l222_222434


namespace count_solutions_absolute_value_l222_222039

theorem count_solutions_absolute_value (x : ℤ) : 
  (|4 * x + 2| ≤ 10) ↔ (x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2) :=
by sorry

end count_solutions_absolute_value_l222_222039


namespace trigonometric_expression_evaluation_l222_222222

theorem trigonometric_expression_evaluation :
  let tan30 := (Real.sqrt 3) / 3
  let sin60 := (Real.sqrt 3) / 2
  let cot60 := 1 / (Real.sqrt 3)
  let tan60 := Real.sqrt 3
  let cos45 := (Real.sqrt 2) / 2
  (3 * tan30) / (1 - sin60) + (cot60 + Real.cos (Real.pi * 70 / 180))^0 - tan60 / (cos45^4) = 7 :=
by
  -- This is where the proof would go
  sorry

end trigonometric_expression_evaluation_l222_222222


namespace neg_number_is_A_l222_222026

def A : ℤ := -(3 ^ 2)
def B : ℤ := (-3) ^ 2
def C : ℤ := abs (-3)
def D : ℤ := -(-3)

theorem neg_number_is_A : A < 0 := 
by sorry

end neg_number_is_A_l222_222026


namespace find_h_s_pairs_l222_222618

def num_regions (h s : ℕ) : ℕ :=
  1 + h * (s + 1) + s * (s + 1) / 2

theorem find_h_s_pairs (h s : ℕ) :
  h > 0 ∧ s > 0 ∧
  num_regions h s = 1992 ↔ 
  (h, s) = (995, 1) ∨ (h, s) = (176, 10) ∨ (h, s) = (80, 21) :=
by
  sorry

end find_h_s_pairs_l222_222618


namespace negation_of_existence_statement_l222_222862

theorem negation_of_existence_statement :
  ¬ (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 ≤ 0)) ↔ ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 > 0) :=
by
  sorry

end negation_of_existence_statement_l222_222862


namespace explicit_formula_for_sequence_l222_222506

theorem explicit_formula_for_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (hSn : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end explicit_formula_for_sequence_l222_222506


namespace find_positive_integers_unique_solution_l222_222568

theorem find_positive_integers_unique_solution :
  ∃ x r p n : ℕ,  
  0 < x ∧ 0 < r ∧ 0 < n ∧  Nat.Prime p ∧ 
  r > 1 ∧ n > 1 ∧ x^r - 1 = p^n ∧ 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := 
    sorry

end find_positive_integers_unique_solution_l222_222568


namespace asha_savings_l222_222428

theorem asha_savings (brother father mother granny spending remaining total borrowed_gifted savings : ℤ) 
  (h1 : brother = 20)
  (h2 : father = 40)
  (h3 : mother = 30)
  (h4 : granny = 70)
  (h5 : spending = 3 * total / 4)
  (h6 : remaining = 65)
  (h7 : remaining = total - spending)
  (h8 : total = brother + father + mother + granny + savings)
  (h9 : borrowed_gifted = brother + father + mother + granny) :
  savings = 100 := by
    sorry

end asha_savings_l222_222428


namespace vasya_days_without_purchase_l222_222007

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l222_222007


namespace roxy_total_plants_remaining_l222_222338

def initial_flowering_plants : Nat := 7
def initial_fruiting_plants : Nat := 2 * initial_flowering_plants
def flowering_plants_bought : Nat := 3
def fruiting_plants_bought : Nat := 2
def flowering_plants_given_away : Nat := 1
def fruiting_plants_given_away : Nat := 4

def total_remaining_plants : Nat :=
  let flowering_plants_now := initial_flowering_plants + flowering_plants_bought - flowering_plants_given_away
  let fruiting_plants_now := initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given_away
  flowering_plants_now + fruiting_plants_now

theorem roxy_total_plants_remaining
  : total_remaining_plants = 21 := by
  sorry

end roxy_total_plants_remaining_l222_222338


namespace first_tree_height_l222_222232

theorem first_tree_height
  (branches_first : ℕ)
  (branches_second : ℕ)
  (height_second : ℕ)
  (branches_third : ℕ)
  (height_third : ℕ)
  (branches_fourth : ℕ)
  (height_fourth : ℕ)
  (average_branches_per_foot : ℕ) :
  branches_first = 200 →
  height_second = 40 →
  branches_second = 180 →
  height_third = 60 →
  branches_third = 180 →
  height_fourth = 34 →
  branches_fourth = 153 →
  average_branches_per_foot = 4 →
  branches_first / average_branches_per_foot = 50 :=
by
  sorry

end first_tree_height_l222_222232


namespace abs_eq_inequality_l222_222290

theorem abs_eq_inequality (m : ℝ) (h : |m - 9| = 9 - m) : m ≤ 9 :=
sorry

end abs_eq_inequality_l222_222290


namespace speed_difference_between_lucy_and_sam_l222_222389

noncomputable def average_speed (distance : ℚ) (time_minutes : ℚ) : ℚ :=
  distance / (time_minutes / 60)

theorem speed_difference_between_lucy_and_sam :
  let distance := 6
  let lucy_time := 15
  let sam_time := 45
  let lucy_speed := average_speed distance lucy_time
  let sam_speed := average_speed distance sam_time
  (lucy_speed - sam_speed) = 16 :=
by
  sorry

end speed_difference_between_lucy_and_sam_l222_222389


namespace value_of_expression_l222_222923

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l222_222923


namespace swimmer_speed_in_still_water_l222_222897

-- Define the conditions
def current_speed : ℝ := 2   -- Speed of the water current is 2 km/h
def swim_time : ℝ := 2.5     -- Time taken to swim against current is 2.5 hours
def distance : ℝ := 5        -- Distance swum against current is 5 km

-- Main theorem proving the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) (h : v - current_speed = distance / swim_time) : v = 4 :=
by {
  -- Skipping the proof steps as per the requirements
  sorry
}

end swimmer_speed_in_still_water_l222_222897


namespace total_baskets_l222_222337

theorem total_baskets (Alex_baskets Sandra_baskets Hector_baskets Jordan_baskets total_baskets : ℕ)
  (h1 : Alex_baskets = 8)
  (h2 : Sandra_baskets = 3 * Alex_baskets)
  (h3 : Hector_baskets = 2 * Sandra_baskets)
  (total_combined_baskets := Alex_baskets + Sandra_baskets + Hector_baskets)
  (h4 : Jordan_baskets = total_combined_baskets / 5)
  (h5 : total_baskets = Alex_baskets + Sandra_baskets + Hector_baskets + Jordan_baskets) :
  total_baskets = 96 := by
  sorry

end total_baskets_l222_222337


namespace prime_div_p_sq_minus_one_l222_222032

theorem prime_div_p_sq_minus_one {p : ℕ} (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 
  (p % 10 = 1 ∨ p % 10 = 9) → 40 ∣ (p^2 - 1) :=
sorry

end prime_div_p_sq_minus_one_l222_222032


namespace geometric_series_sum_l222_222042

theorem geometric_series_sum :
  let a := (1 : ℝ) / 5
  let r := -(1 : ℝ) / 5
  let n := 5
  let S_n := (a * (1 - r ^ n)) / (1 - r)
  S_n = 521 / 3125 := by
  sorry

end geometric_series_sum_l222_222042


namespace unique_positive_integers_pqr_l222_222275

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2)

lemma problem_condition (p q r : ℕ) (py : ℝ) :
  py = y^100
  ∧ py = 2 * (y^98)
  ∧ py = 16 * (y^96)
  ∧ py = 13 * (y^94)
  ∧ py = - y^50
  ∧ py = ↑p * y^46
  ∧ py = ↑q * y^44
  ∧ py = ↑r * y^40 :=
sorry

theorem unique_positive_integers_pqr : 
  ∃! (p q r : ℕ), 
    p = 37 ∧ q = 47 ∧ r = 298 ∧ 
    y^100 = 2 * y^98 + 16 * y^96 + 13 * y^94 - y^50 + ↑p * y^46 + ↑q * y^44 + ↑r * y^40 :=
sorry

end unique_positive_integers_pqr_l222_222275


namespace f_2000_equals_1499001_l222_222081

noncomputable def f (x : ℕ) : ℝ → ℝ := sorry

axiom f_initial : f 0 = 1

axiom f_recursive (x : ℕ) : f (x + 4) = f x + 3 * x + 4

theorem f_2000_equals_1499001 : f 2000 = 1499001 :=
by sorry

end f_2000_equals_1499001_l222_222081


namespace annual_growth_rate_l222_222885

theorem annual_growth_rate (p : ℝ) : 
  let S1 := (1 + p) ^ 12 - 1 / p
  let S2 := ((1 + p) ^ 12 * ((1 + p) ^ 12 - 1)) / p
  let annual_growth := (S2 - S1) / S1
  annual_growth = (1 + p) ^ 12 - 1 :=
by
  sorry

end annual_growth_rate_l222_222885


namespace quadratic_root_k_l222_222991

theorem quadratic_root_k (k : ℝ) : (∃ x : ℝ, x^2 - 2 * x + k = 0 ∧ x = 1) → k = 1 :=
by
  sorry

end quadratic_root_k_l222_222991


namespace find_y_z_l222_222953

theorem find_y_z (x y z : ℚ) (h1 : (x + y) / (z - x) = 9 / 2) (h2 : (y + z) / (y - x) = 5) (h3 : x = 43 / 4) :
  y = 12 / 17 + 17 ∧ z = 5 / 68 + 17 := 
by sorry

end find_y_z_l222_222953


namespace xy_minimization_l222_222796

theorem xy_minimization (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (1 / (x : ℝ)) + 1 / (3 * y) = 1 / 11) : x * y = 176 ∧ x + y = 30 :=
by
  sorry

end xy_minimization_l222_222796


namespace probability_not_all_dice_same_l222_222794

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l222_222794


namespace paint_per_statue_calculation_l222_222933

theorem paint_per_statue_calculation (total_paint : ℚ) (num_statues : ℕ) (expected_paint_per_statue : ℚ) :
  total_paint = 7 / 8 → num_statues = 14 → expected_paint_per_statue = 7 / 112 → 
  total_paint / num_statues = expected_paint_per_statue :=
by
  intros htotal hnum_expected hequals
  rw [htotal, hnum_expected, hequals]
  -- Using the fact that:
  -- total_paint / num_statues = (7 / 8) / 14
  -- This can be rewritten as (7 / 8) * (1 / 14) = 7 / (8 * 14) = 7 / 112
  sorry

end paint_per_statue_calculation_l222_222933


namespace prove_road_length_l222_222171

-- Define variables for days taken by team A, B, and C
variables {a b c : ℕ}

-- Define the daily completion rates for teams A, B, and C
def rateA : ℕ := 300
def rateB : ℕ := 240
def rateC : ℕ := 180

-- Define the maximum length of the road
def max_length : ℕ := 3500

-- Define the total section of the road that team A completes in a days
def total_A (a : ℕ) : ℕ := a * rateA

-- Define the total section of the road that team B completes in b days and 18 hours
def total_B (a b : ℕ) : ℕ := 240 * (a + b) + 180

-- Define the total section of the road that team C completes in c days and 8 hours
def total_C (a b c : ℕ) : ℕ := 180 * (a + b + c) + 60

-- Define the constraint on the sum of days taken: a + b + c
def total_days (a b c : ℕ) : ℕ := a + b + c

-- The proof goal: Prove that (a * 300 == 3300) given the conditions
theorem prove_road_length :
  (total_A a = 3300) ∧ (total_B a b ≤ max_length) ∧ (total_C a b c ≤ max_length) ∧ (total_days a b c ≤ 19) :=
sorry

end prove_road_length_l222_222171


namespace sqrt_9_is_rational_l222_222709

theorem sqrt_9_is_rational : ∃ q : ℚ, (q : ℝ) = 3 := by
  sorry

end sqrt_9_is_rational_l222_222709


namespace parabola_c_value_l222_222494

theorem parabola_c_value (b c : ℝ) 
  (h1 : 2 * b + c = 6) 
  (h2 : -2 * b + c = 2)
  (vertex_cond : ∃ x y : ℝ, y = x^2 + b * x + c ∧ y = -x + 4) : 
  c = 4 :=
sorry

end parabola_c_value_l222_222494


namespace farmer_brown_leg_wing_count_l222_222314

theorem farmer_brown_leg_wing_count :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let pigeons := 4
  let kangaroos := 2
  
  let chicken_legs := 2
  let chicken_wings := 2
  let sheep_legs := 4
  let grasshopper_legs := 6
  let grasshopper_wings := 2
  let spider_legs := 8
  let pigeon_legs := 2
  let pigeon_wings := 2
  let kangaroo_legs := 2

  (chickens * (chicken_legs + chicken_wings) +
  sheep * sheep_legs +
  grasshoppers * (grasshopper_legs + grasshopper_wings) +
  spiders * spider_legs +
  pigeons * (pigeon_legs + pigeon_wings) +
  kangaroos * kangaroo_legs) = 172 := 
by
  sorry

end farmer_brown_leg_wing_count_l222_222314


namespace range_of_m_l222_222164

noncomputable def quadraticExpr (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + 4 * m * x + m + 3

theorem range_of_m :
  (∀ x : ℝ, quadraticExpr m x ≥ 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l222_222164


namespace g_of_50_eq_zero_l222_222761

theorem g_of_50_eq_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - 3 * y * g x = g (x / y)) : g 50 = 0 :=
sorry

end g_of_50_eq_zero_l222_222761


namespace hyperbola_eccentricity_l222_222082

theorem hyperbola_eccentricity (m : ℤ) (h1 : -2 < m) (h2 : m < 2) : 
  let a := m
  let b := (4 - m^2).sqrt 
  let c := (a^2 + b^2).sqrt
  let e := c / a
  e = 2 := by
sorry

end hyperbola_eccentricity_l222_222082


namespace smaller_integer_is_49_l222_222983

theorem smaller_integer_is_49 (m n : ℕ) (hm : 10 ≤ m ∧ m < 100) (hn : 10 ≤ n ∧ n < 100)
  (h : (m + n) / 2 = m + n / 100) : min m n = 49 :=
by
  sorry

end smaller_integer_is_49_l222_222983


namespace area_of_given_triangle_l222_222168

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def vertex_A : ℝ × ℝ := (-1, 4)
def vertex_B : ℝ × ℝ := (7, 0)
def vertex_C : ℝ × ℝ := (11, 5)

theorem area_of_given_triangle : area_of_triangle vertex_A vertex_B vertex_C = 28 :=
by
  show 1 / 2 * |(-1) * (0 - 5) + 7 * (5 - 4) + 11 * (4 - 0)| = 28
  sorry

end area_of_given_triangle_l222_222168


namespace number_of_students_l222_222899

theorem number_of_students (pencils: ℕ) (pencils_per_student: ℕ) (total_students: ℕ) 
  (h1: pencils = 195) (h2: pencils_per_student = 3) (h3: total_students = pencils / pencils_per_student) :
  total_students = 65 := by
  -- proof would go here, but we skip it with sorry for now
  sorry

end number_of_students_l222_222899


namespace multiply_by_3_l222_222597

variable (x : ℕ)  -- Declare x as a natural number

-- Define the conditions
def condition : Prop := x + 14 = 56

-- The goal to prove
theorem multiply_by_3 (h : condition x) : 3 * x = 126 := sorry

end multiply_by_3_l222_222597


namespace solve_2x2_minus1_eq_3x_l222_222012
noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  (root1, root2)

theorem solve_2x2_minus1_eq_3x :
  solve_quadratic 2 (-3) (-1) = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4 ) :=
by
  let roots := solve_quadratic 2 (-3) (-1)
  have : roots = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4) := by sorry
  exact this

end solve_2x2_minus1_eq_3x_l222_222012


namespace circle_tangent_sum_radii_l222_222273

theorem circle_tangent_sum_radii :
  let r1 := 6 + 2 * Real.sqrt 6
  let r2 := 6 - 2 * Real.sqrt 6
  r1 + r2 = 12 :=
by
  sorry

end circle_tangent_sum_radii_l222_222273


namespace train_length_is_correct_l222_222746

noncomputable def train_length (speed_kmph : ℝ) (crossing_time_s : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * crossing_time_s
  total_distance - platform_length_m

theorem train_length_is_correct :
  train_length 60 14.998800095992321 150 = 100 := by
  sorry

end train_length_is_correct_l222_222746


namespace diophantine_infinite_solutions_l222_222281

theorem diophantine_infinite_solutions
  (l m n : ℕ) (h_l_positive : l > 0) (h_m_positive : m > 0) (h_n_positive : n > 0)
  (h_gcd_lm_n : gcd (l * m) n = 1) (h_gcd_ln_m : gcd (l * n) m = 1) (h_gcd_mn_l : gcd (m * n) l = 1)
  : ∃ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0 ∧ (x ^ l + y ^ m = z ^ n)) ∧ (∀ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a ^ l + b ^ m = c ^ n)) → ∀ d : ℕ, d > 0 → ∃ e f g : ℕ, (e > 0 ∧ f > 0 ∧ g > 0 ∧ (e ^ l + f ^ m = g ^ n))) :=
sorry

end diophantine_infinite_solutions_l222_222281


namespace distance_traveled_l222_222912

noncomputable def velocity (t : ℝ) := 2 * t - 3

theorem distance_traveled : 
  (∫ t in (0 : ℝ)..5, |velocity t|) = 29 / 2 := 
by
  sorry

end distance_traveled_l222_222912


namespace rahim_pillows_l222_222046

theorem rahim_pillows (x T : ℕ) (h1 : T = 5 * x) (h2 : (T + 10) / (x + 1) = 6) : x = 4 :=
by
  sorry

end rahim_pillows_l222_222046


namespace negation_proof_l222_222239

theorem negation_proof :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by
  sorry

end negation_proof_l222_222239


namespace jewelry_store_total_cost_l222_222476

-- Definitions for given conditions
def necklace_capacity : Nat := 12
def current_necklaces : Nat := 5
def ring_capacity : Nat := 30
def current_rings : Nat := 18
def bracelet_capacity : Nat := 15
def current_bracelets : Nat := 8

def necklace_cost : Nat := 4
def ring_cost : Nat := 10
def bracelet_cost : Nat := 5

-- Definition for number of items needed to fill displays
def needed_necklaces : Nat := necklace_capacity - current_necklaces
def needed_rings : Nat := ring_capacity - current_rings
def needed_bracelets : Nat := bracelet_capacity - current_bracelets

-- Definition for cost to fill each type of jewelry
def cost_necklaces : Nat := needed_necklaces * necklace_cost
def cost_rings : Nat := needed_rings * ring_cost
def cost_bracelets : Nat := needed_bracelets * bracelet_cost

-- Total cost to fill the displays
def total_cost : Nat := cost_necklaces + cost_rings + cost_bracelets

-- Proof statement
theorem jewelry_store_total_cost : total_cost = 183 := by
  sorry

end jewelry_store_total_cost_l222_222476


namespace polygon_has_area_144_l222_222814

noncomputable def polygonArea (n_sides : ℕ) (perimeter : ℕ) (n_squares : ℕ) : ℕ :=
  let s := perimeter / n_sides
  let square_area := s * s
  square_area * n_squares

theorem polygon_has_area_144 :
  polygonArea 32 64 36 = 144 :=
by
  sorry

end polygon_has_area_144_l222_222814


namespace optimal_production_distribution_l222_222210

noncomputable def min_production_time (unitsI_A unitsI_B unitsII_B : ℕ) : ℕ :=
let rateI_A := 30
let rateII_B := 40
let rateI_B := 50
let initial_days_B := 20
let remaining_units_I := 1500 - (rateI_A * initial_days_B)
let combined_rateI_AB := rateI_A + rateI_B
let days_remaining_I := remaining_units_I / combined_rateI_AB
initial_days_B + days_remaining_I

theorem optimal_production_distribution :
  ∃ (unitsI_A unitsI_B unitsII_B : ℕ),
    unitsI_A + unitsI_B = 1500 ∧ unitsII_B = 800 ∧
    min_production_time unitsI_A unitsI_B unitsII_B = 31 := sorry

end optimal_production_distribution_l222_222210


namespace axis_of_symmetry_exists_l222_222119

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem axis_of_symmetry_exists :
  ∃ k : ℤ, ∃ x : ℝ, (x = -5 * Real.pi / 12 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi))
  ∨ (x = Real.pi / 12 + k * Real.pi / 2 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi)) :=
sorry

end axis_of_symmetry_exists_l222_222119


namespace dave_paid_more_l222_222268

-- Definitions based on conditions in the problem statement
def total_pizza_cost : ℕ := 11  -- Total cost of the pizza in dollars
def num_slices : ℕ := 8  -- Total number of slices in the pizza
def plain_pizza_cost : ℕ := 8  -- Cost of the plain pizza in dollars
def anchovies_cost : ℕ := 2  -- Extra cost of adding anchovies in dollars
def mushrooms_cost : ℕ := 1  -- Extra cost of adding mushrooms in dollars
def dave_slices : ℕ := 7  -- Number of slices Dave ate
def doug_slices : ℕ := 1  -- Number of slices Doug ate
def doug_payment : ℕ := 1  -- Amount Doug paid in dollars
def dave_payment : ℕ := total_pizza_cost - doug_payment  -- Amount Dave paid in dollars

-- Prove that Dave paid 9 dollars more than Doug
theorem dave_paid_more : dave_payment - doug_payment = 9 := by
  -- Proof to be filled in
  sorry

end dave_paid_more_l222_222268


namespace loaves_per_hour_in_one_oven_l222_222822

-- Define the problem constants and variables
def loaves_in_3_weeks : ℕ := 1740
def ovens : ℕ := 4
def weekday_hours : ℕ := 5
def weekend_hours : ℕ := 2
def weekdays_per_week : ℕ := 5
def weekends_per_week : ℕ := 2
def weeks : ℕ := 3

-- Calculate the total hours per week
def hours_per_week : ℕ := (weekdays_per_week * weekday_hours) + (weekends_per_week * weekend_hours)

-- Calculate the total oven-hours for 3 weeks
def total_oven_hours : ℕ := hours_per_week * ovens * weeks

-- Provide the proof statement
theorem loaves_per_hour_in_one_oven : (loaves_in_3_weeks = 5 * total_oven_hours) :=
by
  sorry -- Proof omitted

end loaves_per_hour_in_one_oven_l222_222822


namespace complex_fraction_simplification_l222_222648

theorem complex_fraction_simplification : 
  ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h_imag_unit
  sorry

end complex_fraction_simplification_l222_222648


namespace fraction_sum_is_half_l222_222190

theorem fraction_sum_is_half :
  (1/5 : ℚ) + (3/10 : ℚ) = 1/2 :=
by linarith

end fraction_sum_is_half_l222_222190


namespace one_div_i_plus_i_pow_2015_eq_neg_two_i_l222_222207

def is_imaginary_unit (x : ℂ) : Prop := x * x = -1

theorem one_div_i_plus_i_pow_2015_eq_neg_two_i (i : ℂ) (h : is_imaginary_unit i) : 
  (1 / i + i ^ 2015) = -2 * i :=
sorry

end one_div_i_plus_i_pow_2015_eq_neg_two_i_l222_222207


namespace smaller_angle_measure_l222_222575

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l222_222575


namespace categorize_numbers_l222_222456

def numbers : Set (Rat) := {-16, 0.04, 1/2, -2/3, 25, 0, -3.6, -0.3, 4/3}

def is_integer (x : Rat) : Prop := ∃ z : Int, x = z
def is_fraction (x : Rat) : Prop := ∃ (p q : Int), q ≠ 0 ∧ x = p / q
def is_negative (x : Rat) : Prop := x < 0

def integers (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_integer x}
def fractions (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x}
def negative_rationals (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x ∧ is_negative x}

theorem categorize_numbers :
  integers numbers = {-16, 25, 0} ∧
  fractions numbers = {0.04, 1/2, -2/33, -3.6, -0.3, 4/3} ∧
  negative_rationals numbers = {-16, -2/3, -3.6, -0.3} :=
  sorry

end categorize_numbers_l222_222456


namespace calculate_value_l222_222186

def f (x : ℕ) : ℕ := 2 * x - 3
def g (x : ℕ) : ℕ := x^2 + 1

theorem calculate_value : f (1 + g 3) = 19 := by
  sorry

end calculate_value_l222_222186


namespace sqrt_exp_cube_l222_222366

theorem sqrt_exp_cube :
  ((Real.sqrt ((Real.sqrt 5)^4))^3 = 125) :=
by
  sorry

end sqrt_exp_cube_l222_222366


namespace calculate_polynomial_value_l222_222791

theorem calculate_polynomial_value :
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := 
by 
  sorry

end calculate_polynomial_value_l222_222791


namespace carriage_problem_l222_222738

theorem carriage_problem (x : ℕ) : 
  3 * (x - 2) = 2 * x + 9 := 
sorry

end carriage_problem_l222_222738


namespace rectangle_area_inscribed_circle_l222_222249

theorem rectangle_area_inscribed_circle (r l w : ℝ) (h_r : r = 7)
(h_ratio : l / w = 2) (h_w : w = 2 * r) :
  l * w = 392 :=
by sorry

end rectangle_area_inscribed_circle_l222_222249


namespace calculate_expression_l222_222156

theorem calculate_expression :
  500 * 1986 * 0.3972 * 100 = 20 * 1986^2 :=
by sorry

end calculate_expression_l222_222156


namespace additional_toothpicks_needed_l222_222017

def three_step_toothpicks := 18
def four_step_toothpicks := 26

theorem additional_toothpicks_needed : 
  (∃ (f : ℕ → ℕ), f 3 = three_step_toothpicks ∧ f 4 = four_step_toothpicks ∧ (f 6 - f 4) = 22) :=
by {
  -- Assume f is a function that gives the number of toothpicks for a n-step staircase
  sorry
}

end additional_toothpicks_needed_l222_222017


namespace solve_xyz_integers_l222_222199

theorem solve_xyz_integers (x y z : ℤ) : x^2 + y^2 + z^2 = 2 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end solve_xyz_integers_l222_222199


namespace movie_watching_l222_222014

theorem movie_watching :
  let total_duration := 120 
  let watched1 := 35
  let watched2 := 20
  let watched3 := 15
  let total_watched := watched1 + watched2 + watched3
  total_duration - total_watched = 50 :=
by
  sorry

end movie_watching_l222_222014


namespace total_amount_spent_l222_222905

def price_of_brand_X_pen : ℝ := 4.00
def price_of_brand_Y_pen : ℝ := 2.20
def total_pens_purchased : ℝ := 12
def brand_X_pens_purchased : ℝ := 6

theorem total_amount_spent :
  let brand_X_cost := brand_X_pens_purchased * price_of_brand_X_pen
  let brand_Y_pens_purchased := total_pens_purchased - brand_X_pens_purchased
  let brand_Y_cost := brand_Y_pens_purchased * price_of_brand_Y_pen
  brand_X_cost + brand_Y_cost = 37.20 :=
by
  sorry

end total_amount_spent_l222_222905


namespace min_fraction_value_l222_222152

theorem min_fraction_value (x : ℝ) (hx : x > 9) : ∃ y, y = 36 ∧ (∀ z, z = (x^2 / (x - 9)) → y ≤ z) :=
by
  sorry

end min_fraction_value_l222_222152


namespace problem1_problem2_l222_222868

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Define the conditions and questions as Lean statements

-- First problem: Prove that if A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem problem1 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := 
  sorry

-- Second problem: Prove that if A a ⊆ B, then a ∈ (-∞, 0] ∪ [4, ∞)
theorem problem2 (a : ℝ) (h1 : A a ⊆ B) : a ≤ 0 ∨ a ≥ 4 := 
  sorry

end problem1_problem2_l222_222868


namespace ratio_addition_l222_222988

theorem ratio_addition (x : ℝ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 :=
by
  sorry

end ratio_addition_l222_222988


namespace hyperbola_focus_l222_222683

-- Definition of the hyperbola equation and foci
def is_hyperbola (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - k * y^2 = 1

-- Definition of the hyperbola having a focus at (3, 0) and the value of k
def has_focus_at (k : ℝ) : Prop :=
  ∃ x y : ℝ, is_hyperbola x y k ∧ (x, y) = (3, 0)

theorem hyperbola_focus (k : ℝ) (h : has_focus_at k) : k = 1 / 8 :=
  sorry

end hyperbola_focus_l222_222683


namespace sum_squares_of_roots_of_polynomial_l222_222929

noncomputable def roots (n : ℕ) (p : Polynomial ℂ) : List ℂ :=
  if h : n = p.natDegree then Multiset.toList p.roots else []

theorem sum_squares_of_roots_of_polynomial :
  (roots 2018 (Polynomial.C 404 + Polynomial.C 3 * X ^ 3 + Polynomial.C 44 * X ^ 2015 + X ^ 2018)).sum = 0 :=
by
  sorry

end sum_squares_of_roots_of_polynomial_l222_222929


namespace average_remaining_two_numbers_l222_222006

theorem average_remaining_two_numbers 
  (h1 : (40.5 : ℝ) = 10 * 4.05)
  (h2 : (11.1 : ℝ) = 3 * 3.7)
  (h3 : (11.85 : ℝ) = 3 * 3.95)
  (h4 : (8.6 : ℝ) = 2 * 4.3)
  : (4.475 : ℝ) = (40.5 - (11.1 + 11.85 + 8.6)) / 2 := 
sorry

end average_remaining_two_numbers_l222_222006


namespace intersecting_lines_sum_l222_222009

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end intersecting_lines_sum_l222_222009


namespace ratio_of_height_to_radius_l222_222727

theorem ratio_of_height_to_radius (r h : ℝ)
  (h_cone : r > 0 ∧ h > 0)
  (circumference_cone_base : 20 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2))
  : h / r = Real.sqrt 399 := by
  sorry

end ratio_of_height_to_radius_l222_222727


namespace talkingBirds_count_l222_222531

-- Define the conditions
def totalBirds : ℕ := 77
def nonTalkingBirds : ℕ := 13
def talkingBirds (T : ℕ) : Prop := T + nonTalkingBirds = totalBirds

-- Statement to prove
theorem talkingBirds_count : ∃ T, talkingBirds T ∧ T = 64 :=
by
  -- Proof will go here
  sorry

end talkingBirds_count_l222_222531


namespace cost_of_paint_per_quart_l222_222807

/-- Tommy has a flag that is 5 feet wide and 4 feet tall. 
He needs to paint both sides of the flag. 
A quart of paint covers 4 square feet. 
He spends $20 on paint. 
Prove that the cost of paint per quart is $2. --/
theorem cost_of_paint_per_quart
  (width height : ℕ) (paint_area_per_quart : ℕ) (total_cost : ℕ) (total_area : ℕ) (quarts_needed : ℕ) :
  width = 5 →
  height = 4 →
  paint_area_per_quart = 4 →
  total_cost = 20 →
  total_area = 2 * (width * height) →
  quarts_needed = total_area / paint_area_per_quart →
  total_cost / quarts_needed = 2 := 
by
  intros h_w h_h h_papq h_tc h_ta h_qn
  sorry

end cost_of_paint_per_quart_l222_222807


namespace train_length_l222_222750

theorem train_length (speed_fast speed_slow : ℝ) (time_pass : ℝ)
  (L : ℝ)
  (hf : speed_fast = 46 * (1000/3600))
  (hs : speed_slow = 36 * (1000/3600))
  (ht : time_pass = 36)
  (hL : (2 * L = (speed_fast - speed_slow) * time_pass)) :
  L = 50 := by
  sorry

end train_length_l222_222750


namespace intercepts_sum_eq_eight_l222_222979

def parabola_x_y (x y : ℝ) := x = 3 * y^2 - 9 * y + 5

theorem intercepts_sum_eq_eight :
  ∃ (a b c : ℝ), parabola_x_y a 0 ∧ parabola_x_y 0 b ∧ parabola_x_y 0 c ∧ a + b + c = 8 :=
sorry

end intercepts_sum_eq_eight_l222_222979


namespace solve_missing_number_l222_222909

theorem solve_missing_number (n : ℤ) (h : 121 * n = 75625) : n = 625 :=
sorry

end solve_missing_number_l222_222909


namespace find_f_prime_at_1_l222_222865

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^2 + 3 * x * f_prime_at_1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) :
  (∀ x, deriv (λ x => f x f_prime_at_1) x = 2 * x + 3 * f_prime_at_1) → 
  deriv (λ x => f x f_prime_at_1) 1 = -1 := 
by
exact sorry

end find_f_prime_at_1_l222_222865


namespace multiplicative_inverse_of_AB_l222_222376

def A : ℕ := 222222
def B : ℕ := 476190
def N : ℕ := 189
def modulus : ℕ := 1000000

theorem multiplicative_inverse_of_AB :
  (A * B * N) % modulus = 1 % modulus :=
by
  sorry

end multiplicative_inverse_of_AB_l222_222376


namespace range_of_a_l222_222342

noncomputable def proposition_p (a : ℝ) : Prop := 
  0 < a ∧ a < 1

noncomputable def proposition_q (a : ℝ) : Prop := 
  a > 1 / 4

theorem range_of_a (a : ℝ) : 
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by sorry

end range_of_a_l222_222342


namespace percentage_change_in_receipts_l222_222802

theorem percentage_change_in_receipts
  (P S : ℝ) -- Original price and sales
  (hP : P > 0)
  (hS : S > 0)
  (new_P : ℝ := 0.70 * P) -- Price after 30% reduction
  (new_S : ℝ := 1.50 * S) -- Sales after 50% increase
  :
  (new_P * new_S - P * S) / (P * S) * 100 = 5 :=
by
  sorry

end percentage_change_in_receipts_l222_222802


namespace anne_age_ratio_l222_222602

-- Define the given conditions and prove the final ratio
theorem anne_age_ratio (A M : ℕ) (h1 : A = 4 * (A - 4 * M) + M) 
(h2 : A - M = 3 * (A - 4 * M)) : (A : ℚ) / (M : ℚ) = 5.5 := 
sorry

end anne_age_ratio_l222_222602


namespace team_t_speed_l222_222053

theorem team_t_speed (v t : ℝ) (h1 : 300 = v * t) (h2 : 300 = (v + 5) * (t - 3)) : v = 20 :=
by 
  sorry

end team_t_speed_l222_222053


namespace complement_A_complement_B_intersection_A_B_complement_union_A_B_l222_222748

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def set_U : Set ℝ := {x | true}  -- This represents U = ℝ
def set_A : Set ℝ := {x | x < -2 ∨ x > 5}
def set_B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem complement_A :
  ∀ x : ℝ, x ∈ set_U \ set_A ↔ -2 ≤ x ∧ x ≤ 5 :=
by
  intro x
  sorry

theorem complement_B :
  ∀ x : ℝ, x ∉ set_B ↔ x < 4 ∨ x > 6 :=
by
  intro x
  sorry

theorem intersection_A_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 5 < x ∧ x ≤ 6 :=
by
  intro x
  sorry

theorem complement_union_A_B :
  ∀ x : ℝ, x ∈ set_U \ (set_A ∪ set_B) ↔ -2 ≤ x ∧ x < 4 :=
by
  intro x
  sorry

end complement_A_complement_B_intersection_A_B_complement_union_A_B_l222_222748


namespace pattern_C_not_foldable_without_overlap_l222_222959

-- Define the four patterns, denoted as PatternA, PatternB, PatternC, and PatternD.
inductive Pattern
| A : Pattern
| B : Pattern
| C : Pattern
| D : Pattern

-- Define a predicate for a pattern being foldable into a cube without overlap.
def foldable_into_cube (p : Pattern) : Prop := sorry

theorem pattern_C_not_foldable_without_overlap : ¬ foldable_into_cube Pattern.C := sorry

end pattern_C_not_foldable_without_overlap_l222_222959


namespace radius_of_circle_with_center_on_line_and_passing_through_points_l222_222615

theorem radius_of_circle_with_center_on_line_and_passing_through_points : 
  (∃ a b : ℝ, 2 * a + b = 0 ∧ 
              (a - 1) ^ 2 + (b - 3) ^ 2 = r ^ 2 ∧ 
              (a - 4) ^ 2 + (b - 2) ^ 2 = r ^ 2 
              → r = 5) := 
by 
  sorry

end radius_of_circle_with_center_on_line_and_passing_through_points_l222_222615


namespace evaluate_star_l222_222165

-- Define the operation c star d
def star (c d : ℤ) : ℤ := c^2 - 2 * c * d + d^2

-- State the theorem to prove the given problem
theorem evaluate_star : (star 3 5) = 4 := by
  sorry

end evaluate_star_l222_222165


namespace total_sections_l222_222817

theorem total_sections (boys girls : ℕ) (h_boys : boys = 408) (h_girls : girls = 240) :
  let gcd_boys_girls := Nat.gcd boys girls
  let sections_boys := boys / gcd_boys_girls
  let sections_girls := girls / gcd_boys_girls
  sections_boys + sections_girls = 27 :=
by
  sorry

end total_sections_l222_222817


namespace true_statements_l222_222508

theorem true_statements :
  (5 ∣ 25) ∧ (19 ∣ 209 ∧ ¬ (19 ∣ 63)) ∧ (30 ∣ 90) ∧ (14 ∣ 28 ∧ 14 ∣ 56) ∧ (9 ∣ 180) :=
by
  have A : 5 ∣ 25 := sorry
  have B1 : 19 ∣ 209 := sorry
  have B2 : ¬ (19 ∣ 63) := sorry
  have C : 30 ∣ 90 := sorry
  have D1 : 14 ∣ 28 := sorry
  have D2 : 14 ∣ 56 := sorry
  have E : 9 ∣ 180 := sorry
  exact ⟨A, ⟨B1, B2⟩, C, ⟨D1, D2⟩, E⟩

end true_statements_l222_222508


namespace evaluate_g_at_neg_four_l222_222452

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l222_222452


namespace average_salary_increase_l222_222653

theorem average_salary_increase :
  let avg_salary := 1200
  let num_employees := 20
  let manager_salary := 3300
  let new_num_people := num_employees + 1
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / new_num_people
  let increase := new_avg_salary - avg_salary
  increase = 100 :=
by
  sorry

end average_salary_increase_l222_222653


namespace eccentricity_of_ellipse_l222_222627

theorem eccentricity_of_ellipse (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (h4 : b = Real.sqrt 3 * c) : e = 1/2 :=
by
  sorry

end eccentricity_of_ellipse_l222_222627


namespace cars_on_river_road_l222_222561

theorem cars_on_river_road (B C : ℕ) (h_ratio : B / C = 1 / 3) (h_fewer : C = B + 40) : C = 60 :=
sorry

end cars_on_river_road_l222_222561


namespace minimum_value_problem1_minimum_value_problem2_l222_222145

theorem minimum_value_problem1 (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y >= 6 := 
sorry

theorem minimum_value_problem2 (x : ℝ) (h : x > 1) : 
  ∃ y, y = (x^2 + 8) / (x - 1) ∧ y >= 8 := 
sorry

end minimum_value_problem1_minimum_value_problem2_l222_222145


namespace total_bricks_l222_222795

theorem total_bricks (n1 n2 r1 r2 : ℕ) (w1 w2 : ℕ)
  (h1 : n1 = 60) (h2 : r1 = 100) (h3 : n2 = 80) (h4 : r2 = 120)
  (h5 : w1 = 5) (h6 : w2 = 5) :
  (w1 * (n1 * r1) + w2 * (n2 * r2)) = 78000 :=
by sorry

end total_bricks_l222_222795


namespace data_set_average_l222_222562

theorem data_set_average (a : ℝ) (h : (2 + 3 + 3 + 4 + a) / 5 = 3) : a = 3 := 
sorry

end data_set_average_l222_222562


namespace three_points_in_circle_of_radius_one_seventh_l222_222264

-- Define the problem
theorem three_points_in_circle_of_radius_one_seventh (P : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ (P i).1 ∧ (P i).1 ≤ 1 ∧ 0 ≤ (P i).2 ∧ (P i).2 ≤ 1) →
  ∃ (i j k : Fin 51), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    dist (P i) (P j) ≤ 2/7 ∧ dist (P j) (P k) ≤ 2/7 ∧ dist (P k) (P i) ≤ 2/7 :=
by
  sorry

end three_points_in_circle_of_radius_one_seventh_l222_222264


namespace youngest_sibling_age_l222_222901

theorem youngest_sibling_age
    (age_youngest : ℕ)
    (first_sibling : ℕ := age_youngest + 4)
    (second_sibling : ℕ := age_youngest + 5)
    (third_sibling : ℕ := age_youngest + 7)
    (average_age : ℕ := 21)
    (sum_of_ages : ℕ := 4 * average_age)
    (total_age_check : (age_youngest + first_sibling + second_sibling + third_sibling) = sum_of_ages) :
  age_youngest = 17 :=
sorry

end youngest_sibling_age_l222_222901


namespace betty_red_beads_l222_222317

theorem betty_red_beads (r b : ℕ) (h_ratio : r / b = 3 / 2) (h_blue_beads : b = 20) : r = 30 :=
by
  sorry

end betty_red_beads_l222_222317


namespace painting_time_l222_222420

noncomputable def work_rate (t : ℕ) : ℚ := 1 / t

theorem painting_time (shawn_time karen_time alex_time total_work_rate : ℚ)
  (h_shawn : shawn_time = 18)
  (h_karen : karen_time = 12)
  (h_alex : alex_time = 15) :
  total_work_rate = 1 / (shawn_time + karen_time + alex_time) :=
by
  sorry

end painting_time_l222_222420


namespace find_x_if_perpendicular_l222_222775

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (2 * x - 1, 3)
def vec_n : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_if_perpendicular (x : ℝ) : 
  dot_product (vec_m x) vec_n = 0 ↔ x = 2 :=
by
  sorry

end find_x_if_perpendicular_l222_222775


namespace triangle_BC_length_l222_222576

noncomputable def length_of_BC (ABC : Triangle) (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) 
    (BD_squared_plus_CD_squared : ℝ) : ℝ :=
  if incircle_radius = 3 ∧ altitude_A_to_BC = 15 ∧ BD_squared_plus_CD_squared = 33 then
    3 * Real.sqrt 7
  else
    0 -- This value is arbitrary, as the conditions above are specific

theorem triangle_BC_length {ABC : Triangle}
    (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) (BD_squared_plus_CD_squared : ℝ) :
    incircle_radius = 3 →
    altitude_A_to_BC = 15 →
    BD_squared_plus_CD_squared = 33 →
    length_of_BC ABC incircle_radius altitude_A_to_BC BD_squared_plus_CD_squared = 3 * Real.sqrt 7 :=
by intros; sorry

end triangle_BC_length_l222_222576


namespace border_area_correct_l222_222707

theorem border_area_correct :
  let photo_height := 9
  let photo_width := 12
  let border_width := 3
  let photo_area := photo_height * photo_width
  let framed_height := photo_height + 2 * border_width
  let framed_width := photo_width + 2 * border_width
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  border_area = 162 :=
by sorry

end border_area_correct_l222_222707


namespace gcd_294_84_l222_222876

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end gcd_294_84_l222_222876


namespace students_selecting_water_l222_222038

-- Definitions of percentages and given values.
def p : ℝ := 0.7
def q : ℝ := 0.1
def n : ℕ := 140

-- The Lean statement to prove the number of students who selected water.
theorem students_selecting_water (p_eq : p = 0.7) (q_eq : q = 0.1) (n_eq : n = 140) :
  ∃ w : ℕ, w = (q / p) * n ∧ w = 20 :=
by sorry

end students_selecting_water_l222_222038


namespace greatest_common_multiple_of_10_and_15_lt_120_l222_222971

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l222_222971


namespace minimize_distances_is_k5_l222_222033

-- Define the coordinates of points A, B, and D
def A : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (0, 5)

-- Define C as a point vertically below D, implying the x-coordinate is the same as that of D and y = k
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Prove that the value of k that minimizes the distances over AC and BC is k = 5
theorem minimize_distances_is_k5 : ∃ k : ℝ, (C k = (0, 5)) ∧ k = 5 :=
by {
  sorry
}

end minimize_distances_is_k5_l222_222033


namespace weight_of_B_l222_222045

theorem weight_of_B (A B C : ℕ) (h1 : A + B + C = 90) (h2 : A + B = 50) (h3 : B + C = 56) : B = 16 := 
sorry

end weight_of_B_l222_222045


namespace range_of_p_nonnegative_range_of_p_all_values_range_of_p_l222_222861

def p (x : ℝ) : ℝ := x^4 - 6 * x^2 + 9

theorem range_of_p_nonnegative (x : ℝ) (hx : 0 ≤ x) : 
  ∃ y, y = p x ∧ 0 ≤ y := 
sorry

theorem range_of_p_all_values (y : ℝ) : 
  0 ≤ y → (∃ x, 0 ≤ x ∧ p x = y) :=
sorry

theorem range_of_p (x : ℝ) (hx : 0 ≤ x) : 
  ∀ y, (∃ x, 0 ≤ x ∧ p x = y) ↔ (0 ≤ y) :=
sorry

end range_of_p_nonnegative_range_of_p_all_values_range_of_p_l222_222861


namespace sufficient_but_not_necessary_l222_222433

variable {a b : ℝ}

theorem sufficient_but_not_necessary (ha : a > 0) (hb : b > 0) : 
  (ab > 1) → (a + b > 2) ∧ ¬ (a + b > 2 → ab > 1) :=
by
  sorry

end sufficient_but_not_necessary_l222_222433


namespace four_digit_integers_correct_five_digit_integers_correct_l222_222412

-- Definition for the four-digit integers problem
def num_four_digit_integers := ∃ digits : Finset (Fin 5), 4 * 24 = 96

theorem four_digit_integers_correct : num_four_digit_integers := 
by
  sorry

-- Definition for the five-digit integers problem without repetition and greater than 21000
def num_five_digit_integers := ∃ digits : Finset (Fin 5), 48 + 18 = 66

theorem five_digit_integers_correct : num_five_digit_integers := 
by
  sorry

end four_digit_integers_correct_five_digit_integers_correct_l222_222412


namespace arithmetic_progr_property_l222_222887

theorem arithmetic_progr_property (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 + a 3 = 5 / 2)
  (h2 : a 2 + a 4 = 5 / 4)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h4 : a 3 = a 1 + 2 * (a 2 - a 1))
  (h5 : a 2 = a 1 + (a 2 - a 1)) :
  S 3 / a 3 = 6 := sorry

end arithmetic_progr_property_l222_222887


namespace find_cubic_expression_l222_222734

theorem find_cubic_expression (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end find_cubic_expression_l222_222734


namespace range_of_k_l222_222799

noncomputable def circle_equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem range_of_k (k : ℝ) :
  circle_equation k →
  k ∈ (Set.Iio (-1) ∪ Set.Ioi 4) :=
sorry

end range_of_k_l222_222799


namespace vanessa_deleted_files_l222_222319

theorem vanessa_deleted_files (initial_music_files : ℕ) (initial_video_files : ℕ) (files_left : ℕ) (files_deleted : ℕ) :
  initial_music_files = 13 → initial_video_files = 30 → files_left = 33 → 
  files_deleted = (initial_music_files + initial_video_files) - files_left → files_deleted = 10 :=
by
  sorry

end vanessa_deleted_files_l222_222319


namespace base_7_units_digit_of_product_359_72_l222_222892

def base_7_units_digit (n : ℕ) : ℕ := n % 7

theorem base_7_units_digit_of_product_359_72 : base_7_units_digit (359 * 72) = 4 := 
by
  sorry

end base_7_units_digit_of_product_359_72_l222_222892


namespace more_pie_eaten_l222_222457

theorem more_pie_eaten (e f : ℝ) (h1 : e = 0.67) (h2 : f = 0.33) : e - f = 0.34 :=
by sorry

end more_pie_eaten_l222_222457


namespace opposite_of_point_one_l222_222021

theorem opposite_of_point_one : ∃ x : ℝ, 0.1 + x = 0 ∧ x = -0.1 :=
by
  sorry

end opposite_of_point_one_l222_222021


namespace solution_set_l222_222644

variable {f : ℝ → ℝ}
variable (h1 : ∀ x, x < 0 → x * deriv f x - 2 * f x > 0)
variable (h2 : ∀ x, x < 0 → f x ≠ 0)

theorem solution_set (h3 : ∀ x, -2024 < x ∧ x < -2023 → f (x + 2023) - (x + 2023)^2 * f (-1) < 0) :
    {x : ℝ | f (x + 2023) - (x + 2023)^2 * f (-1) < 0} = {x : ℝ | -2024 < x ∧ x < -2023} :=
by
  sorry

end solution_set_l222_222644


namespace min_positive_period_and_symmetry_axis_l222_222179

noncomputable def f (x : ℝ) := - (Real.sin (x + Real.pi / 6)) * (Real.sin (x - Real.pi / 3))

theorem min_positive_period_and_symmetry_axis :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ k : ℤ, ∀ x : ℝ, f x = f (x + 1 / 2 * k * Real.pi + Real.pi / 12)) := by
  sorry

end min_positive_period_and_symmetry_axis_l222_222179


namespace mart_income_percentage_l222_222776

theorem mart_income_percentage 
  (J T M : ℝ)
  (h1 : M = 1.60 * T)
  (h2 : T = 0.60 * J) :
  M = 0.96 * J :=
sorry

end mart_income_percentage_l222_222776


namespace rectangle_measurement_error_l222_222856

theorem rectangle_measurement_error
    (L W : ℝ) -- actual lengths of the sides
    (x : ℝ) -- percentage in excess for the first side
    (h1 : 0 ≤ x) -- ensuring percentage cannot be negative
    (h2 : (L * (1 + x / 100)) * (W * 0.95) = L * W * 1.045) -- given condition on areas
    : x = 10 :=
by
  sorry

end rectangle_measurement_error_l222_222856


namespace HeatherIsHeavier_l222_222220

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l222_222220


namespace quadr_pyramid_edge_sum_is_36_l222_222150

def sum_edges_quad_pyr (hex_sum_edges : ℕ) (hex_num_edges : ℕ) (quad_num_edges : ℕ) : ℕ :=
  let length_one_edge := hex_sum_edges / hex_num_edges
  length_one_edge * quad_num_edges

theorem quadr_pyramid_edge_sum_is_36 :
  sum_edges_quad_pyr 81 18 8 = 36 :=
by
  -- We defer proof
  sorry

end quadr_pyramid_edge_sum_is_36_l222_222150


namespace subcommittees_with_at_least_one_teacher_l222_222722

theorem subcommittees_with_at_least_one_teacher
  (total_members teachers : ℕ)
  (total_members_eq : total_members = 12)
  (teachers_eq : teachers = 5)
  (subcommittee_size : ℕ)
  (subcommittee_size_eq : subcommittee_size = 5) :
  ∃ (n : ℕ), n = 771 :=
by
  sorry

end subcommittees_with_at_least_one_teacher_l222_222722


namespace floor_cube_neg_seven_four_l222_222055

theorem floor_cube_neg_seven_four :
  (Int.floor ((-7 / 4 : ℚ) ^ 3) = -6) :=
by
  sorry

end floor_cube_neg_seven_four_l222_222055


namespace tan_alpha_value_l222_222261

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi + α) = 3 / 5) 
  (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l222_222261


namespace red_to_green_speed_ratio_l222_222200

-- Conditions
def blue_car_speed : Nat := 80 -- The blue car's speed is 80 miles per hour
def green_car_speed : Nat := 8 * blue_car_speed -- The green car's speed is 8 times the blue car's speed
def red_car_speed : Nat := 1280 -- The red car's speed is 1280 miles per hour

-- Theorem stating the ratio of red car's speed to green car's speed
theorem red_to_green_speed_ratio : red_car_speed / green_car_speed = 2 := by
  sorry -- proof goes here

end red_to_green_speed_ratio_l222_222200


namespace x_value_l222_222297

def x_is_75_percent_greater (x : ℝ) (y : ℝ) : Prop := x = y + 0.75 * y

theorem x_value (x : ℝ) : x_is_75_percent_greater x 150 → x = 262.5 :=
by
  intro h
  rw [x_is_75_percent_greater] at h
  sorry

end x_value_l222_222297


namespace back_seat_can_hold_8_people_l222_222295

def totalPeopleOnSides : ℕ :=
  let left_seats := 15
  let right_seats := left_seats - 3
  let people_per_seat := 3
  (left_seats + right_seats) * people_per_seat

def bus_total_capacity : ℕ := 89

def back_seat_capacity : ℕ :=
  bus_total_capacity - totalPeopleOnSides

theorem back_seat_can_hold_8_people : back_seat_capacity = 8 := by
  sorry

end back_seat_can_hold_8_people_l222_222295


namespace n_cubed_minus_9n_plus_27_not_div_by_81_l222_222421

theorem n_cubed_minus_9n_plus_27_not_div_by_81 (n : ℤ) : ¬ 81 ∣ (n^3 - 9 * n + 27) :=
sorry

end n_cubed_minus_9n_plus_27_not_div_by_81_l222_222421


namespace moles_of_KCl_formed_l222_222941

variables (NaCl KNO3 KCl NaNO3 : Type) 

-- Define the moles of each compound
variables (moles_NaCl moles_KNO3 moles_KCl moles_NaNO3 : ℕ)

-- Initial conditions
axiom initial_NaCl_condition : moles_NaCl = 2
axiom initial_KNO3_condition : moles_KNO3 = 2

-- Reaction definition
axiom reaction : moles_KCl = moles_NaCl

theorem moles_of_KCl_formed :
  moles_KCl = 2 :=
by sorry

end moles_of_KCl_formed_l222_222941


namespace length_of_train_l222_222424

theorem length_of_train
  (L : ℝ) 
  (h1 : ∀ S, S = L / 8)
  (h2 : L + 267 = (L / 8) * 20) :
  L = 178 :=
sorry

end length_of_train_l222_222424


namespace xyz_sum_eq_7x_plus_5_l222_222919

variable (x y z : ℝ)

theorem xyz_sum_eq_7x_plus_5 (h1: y = 3 * x) (h2: z = y + 5) : x + y + z = 7 * x + 5 :=
by
  sorry

end xyz_sum_eq_7x_plus_5_l222_222919


namespace mary_days_eq_11_l222_222120

variable (x : ℝ) -- Number of days Mary takes to complete the work
variable (m_eff : ℝ) -- Efficiency of Mary (work per day)
variable (r_eff : ℝ) -- Efficiency of Rosy (work per day)

-- Given conditions
axiom rosy_efficiency : r_eff = 1.1 * m_eff
axiom rosy_days : r_eff * 10 = 1

-- Define the efficiency of Mary in terms of days
axiom mary_efficiency : m_eff = 1 / x

-- The theorem to prove
theorem mary_days_eq_11 : x = 11 :=
by
  sorry

end mary_days_eq_11_l222_222120


namespace correct_inequality_l222_222581

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_increasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

theorem correct_inequality : f (-2) < f 1 ∧ f 1 < f 3 :=
by 
  sorry

end correct_inequality_l222_222581


namespace correct_statement_l222_222509

def is_accurate_to (value : ℝ) (place : ℝ) : Prop :=
  ∃ k : ℤ, value = k * place

def statement_A : Prop := is_accurate_to 51000 0.1
def statement_B : Prop := is_accurate_to 0.02 1
def statement_C : Prop := (2.8 = 2.80)
def statement_D : Prop := is_accurate_to (2.3 * 10^4) 1000

theorem correct_statement : statement_D :=
by
  sorry

end correct_statement_l222_222509


namespace second_supplier_more_cars_l222_222661

-- Define the constants and conditions given in the problem
def total_production := 5650000
def first_supplier := 1000000
def fourth_fifth_supplier := 325000

-- Define the unknown variable for the second supplier
noncomputable def second_supplier : ℕ := sorry

-- Define the equation based on the conditions
def equation := first_supplier + second_supplier + (first_supplier + second_supplier) + (4 * fourth_fifth_supplier / 2) = total_production

-- Prove that the second supplier receives 500,000 more cars than the first supplier
theorem second_supplier_more_cars : 
  ∃ X : ℕ, equation → (X = first_supplier + 500000) :=
sorry

end second_supplier_more_cars_l222_222661


namespace total_trash_pieces_l222_222881

theorem total_trash_pieces (classroom_trash : ℕ) (outside_trash : ℕ)
  (h1 : classroom_trash = 344) (h2 : outside_trash = 1232) : 
  classroom_trash + outside_trash = 1576 :=
by
  sorry

end total_trash_pieces_l222_222881


namespace dart_within_triangle_probability_l222_222316

theorem dart_within_triangle_probability (s : ℝ) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  (triangle_area / hexagon_area) = 1 / 24 :=
by sorry

end dart_within_triangle_probability_l222_222316


namespace correct_propositions_l222_222175

-- Definitions according to the given conditions
def generatrix_cylinder (p1 p2 : Point) (c : Cylinder) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def generatrix_cone (v : Point) (p : Point) (c : Cone) : Prop :=
  -- Check if the line from the vertex to a base point is a generatrix
  sorry

def generatrix_frustum (p1 p2 : Point) (f : Frustum) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def parallel_generatrices_cylinder (gen1 gen2 : Line) (c : Cylinder) : Prop :=
  -- Check if two generatrices of the cylinder are parallel
  sorry

-- The theorem stating propositions ② and ④ are correct
theorem correct_propositions :
  generatrix_cone vertex point cone ∧
  parallel_generatrices_cylinder gen1 gen2 cylinder :=
by
  sorry

end correct_propositions_l222_222175


namespace determine_angle_B_l222_222949

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)
  ∧ (a = 8)
  ∧ (b = Real.sqrt 3)

theorem determine_angle_B (A B C : ℝ) (a b c : ℝ)
  (h : problem_statement A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
by 
  sorry

end determine_angle_B_l222_222949


namespace removed_term_is_a11_l222_222631

noncomputable def sequence_a (n : ℕ) (a1 d : ℤ) := a1 + (n - 1) * d

def sequence_sum (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem removed_term_is_a11 :
  ∃ d : ℤ, ∀ a1 d : ℤ, 
            a1 = -5 ∧ 
            sequence_sum 11 a1 d = 55 ∧ 
            (sequence_sum 11 a1 d - sequence_a 11 a1 d) / 10 = 4 
          → sequence_a 11 a1 d = removed_term :=
sorry

end removed_term_is_a11_l222_222631


namespace cannot_lie_on_line_l222_222554

open Real

theorem cannot_lie_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  (0, -2023) ≠ (0, b) :=
by
  sorry

end cannot_lie_on_line_l222_222554


namespace sum_of_first_15_terms_l222_222427

-- Given an arithmetic sequence {a_n} such that a_4 + a_6 + a_8 + a_10 + a_12 = 40
-- we need to prove that the sum of the first 15 terms is 120

theorem sum_of_first_15_terms 
  (a_4 a_6 a_8 a_10 a_12 : ℤ)
  (h1 : a_4 + a_6 + a_8 + a_10 + a_12 = 40)
  (a1 d : ℤ)
  (h2 : a_4 = a1 + 3*d)
  (h3 : a_6 = a1 + 5*d)
  (h4 : a_8 = a1 + 7*d)
  (h5 : a_10 = a1 + 9*d)
  (h6 : a_12 = a1 + 11*d) :
  (15 * (a1 + 7*d) = 120) :=
by
  sorry

end sum_of_first_15_terms_l222_222427


namespace decagon_adjacent_vertices_probability_l222_222411

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l222_222411


namespace sale_record_is_negative_five_l222_222483

-- Given that a purchase of 10 items is recorded as +10
def purchase_record (items : Int) : Int := items

-- Prove that the sale of 5 items should be recorded as -5
theorem sale_record_is_negative_five : purchase_record 10 = 10 → purchase_record (-5) = -5 :=
by
  intro h
  sorry

end sale_record_is_negative_five_l222_222483


namespace coefficient_a5_l222_222183

theorem coefficient_a5 (a a1 a2 a3 a4 a5 a6 : ℝ) (h :  (∀ x : ℝ, x^6 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) :
  a5 = 6 :=
sorry

end coefficient_a5_l222_222183


namespace max_digit_d_l222_222638

theorem max_digit_d (d f : ℕ) (h₁ : d ≤ 9) (h₂ : f ≤ 9) (h₃ : (18 + d + f) % 3 = 0) (h₄ : (12 - (d + f)) % 11 = 0) : d = 1 :=
sorry

end max_digit_d_l222_222638


namespace circles_chord_length_l222_222096

theorem circles_chord_length (r1 r2 r3 : ℕ) (m n p : ℕ) (h1 : r1 = 4) (h2 : r2 = 10) (h3 : r3 = 14)
(h4 : gcd m p = 1) (h5 : ¬ (∃ (k : ℕ), k^2 ∣ n)) : m + n + p = 19 :=
by
  sorry

end circles_chord_length_l222_222096


namespace modular_inverse_addition_l222_222349

theorem modular_inverse_addition :
  (3 * 9 + 9 * 37) % 63 = 45 :=
by
  sorry

end modular_inverse_addition_l222_222349


namespace Shelby_fog_time_l222_222612

variable (x y : ℕ)

-- Conditions
def speed_sun := 7/12
def speed_rain := 5/12
def speed_fog := 1/4
def total_time := 60
def total_distance := 20

theorem Shelby_fog_time :
  ((speed_sun * (total_time - x - y)) + (speed_rain * x) + (speed_fog * y) = total_distance) → y = 45 :=
by
  sorry

end Shelby_fog_time_l222_222612


namespace hiker_total_distance_l222_222489

def hiker_distance (day1_hours day1_speed day2_speed : ℕ) : ℕ :=
  let day2_hours := day1_hours - 1
  let day3_hours := day1_hours
  (day1_hours * day1_speed) + (day2_hours * day2_speed) + (day3_hours * day2_speed)

theorem hiker_total_distance :
  hiker_distance 6 3 4 = 62 := 
by 
  sorry

end hiker_total_distance_l222_222489


namespace log_problem_l222_222676

open Real

theorem log_problem : 2 * log 5 + log 4 = 2 := by
  sorry

end log_problem_l222_222676


namespace max_int_difference_l222_222999

theorem max_int_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) : 
  y - x = 5 :=
sorry

end max_int_difference_l222_222999


namespace quadratic_equation_root_form_l222_222445

theorem quadratic_equation_root_form
  (a b c : ℤ) (m n p : ℤ)
  (ha : a = 3)
  (hb : b = -4)
  (hc : c = -7)
  (h_discriminant : b^2 - 4 * a * c = n)
  (hgcd_mn : Int.gcd m n = 1)
  (hgcd_mp : Int.gcd m p = 1)
  (hgcd_np : Int.gcd n p = 1) :
  n = 100 :=
by
  sorry

end quadratic_equation_root_form_l222_222445


namespace problem_statement_l222_222570

theorem problem_statement (x y : ℤ) (k : ℤ) (h : 4 * x - y = 3 * k) : 9 ∣ 4 * x^2 + 7 * x * y - 2 * y^2 :=
by
  sorry

end problem_statement_l222_222570


namespace ordered_pairs_squares_diff_150_l222_222035

theorem ordered_pairs_squares_diff_150 (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn : m ≥ n) (h_diff : m^2 - n^2 = 150) : false :=
by {
    sorry
}

end ordered_pairs_squares_diff_150_l222_222035


namespace dorchester_daily_pay_l222_222203

theorem dorchester_daily_pay (D : ℝ) (P : ℝ) (total_earnings : ℝ) (num_puppies : ℕ) (earn_per_puppy : ℝ) 
  (h1 : total_earnings = 76) (h2 : num_puppies = 16) (h3 : earn_per_puppy = 2.25) 
  (h4 : total_earnings = D + num_puppies * earn_per_puppy) : D = 40 :=
by
  sorry

end dorchester_daily_pay_l222_222203


namespace find_xyz_l222_222665

theorem find_xyz (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h₃ : x + y + z = 3) :
  x * y * z = 16 / 3 := 
  sorry

end find_xyz_l222_222665


namespace correct_factorization_l222_222050

theorem correct_factorization (a x m : ℝ) :
  (ax^2 - a = a * (x^2 - 1)) ∨
  (m^3 + m = m * (m^2 + 1)) ∨
  (x^2 + 2*x - 3 = x*(x+2) - 3) ∨
  (x^2 + 2*x - 3 = (x-3)*(x+1)) :=
by sorry

end correct_factorization_l222_222050


namespace angle_triple_supplement_l222_222805

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l222_222805


namespace quarter_pounder_cost_l222_222751

theorem quarter_pounder_cost :
  let fries_cost := 2 * 1.90
  let milkshakes_cost := 2 * 2.40
  let min_purchase := 18
  let current_total := fries_cost + milkshakes_cost
  let amount_needed := min_purchase - current_total
  let additional_spending := 3
  let total_cost := amount_needed + additional_spending
  total_cost = 12.40 :=
by
  sorry

end quarter_pounder_cost_l222_222751


namespace angle_coincides_with_graph_y_eq_neg_abs_x_l222_222362

noncomputable def angle_set (α : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

theorem angle_coincides_with_graph_y_eq_neg_abs_x (α : ℝ) :
  α ∈ angle_set α ↔ 
  ∃ k : ℤ, (α = k * 360 + 225 ∨ α = k * 360 + 315) :=
by
  sorry

end angle_coincides_with_graph_y_eq_neg_abs_x_l222_222362


namespace find_k_value_l222_222610

theorem find_k_value 
  (A B C k : ℤ)
  (hA : A = -3)
  (hB : B = -5)
  (hC : C = 6)
  (hSum : A + B + C + k = -A - B - C - k) : 
  k = 2 :=
sorry

end find_k_value_l222_222610


namespace ratio_not_necessarily_constant_l222_222839

theorem ratio_not_necessarily_constant (x y : ℝ) : ¬ (∃ k : ℝ, ∀ x y, x / y = k) :=
by
  sorry

end ratio_not_necessarily_constant_l222_222839


namespace last_three_digits_of_2_pow_10000_l222_222649

theorem last_three_digits_of_2_pow_10000 (h : 2^500 ≡ 1 [MOD 1250]) : (2^10000) % 1000 = 1 :=
by
  sorry

end last_three_digits_of_2_pow_10000_l222_222649


namespace steve_took_4_berries_l222_222864

theorem steve_took_4_berries (s t : ℕ) (H1 : s = 32) (H2 : t = 21) (H3 : s - 7 = t + x) :
  x = 4 :=
by
  sorry

end steve_took_4_berries_l222_222864


namespace length_of_other_parallel_side_l222_222741

theorem length_of_other_parallel_side (a b h area : ℝ) 
  (h_area : area = 190) 
  (h_parallel1 : b = 18) 
  (h_height : h = 10) : 
  a = 20 :=
by
  sorry

end length_of_other_parallel_side_l222_222741


namespace remainder_problem_l222_222413

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 :=
by
  sorry

end remainder_problem_l222_222413


namespace diversity_values_l222_222963

theorem diversity_values (k : ℕ) (h : 1 ≤ k ∧ k ≤ 4) :
  ∃ (D : ℕ), D = 1000 * (k - 1) := by
  sorry

end diversity_values_l222_222963


namespace total_number_of_students_l222_222993

theorem total_number_of_students 
    (T : ℕ)
    (h1 : ∃ a, a = T / 5) 
    (h2 : ∃ b, b = T / 4) 
    (h3 : ∃ c, c = T / 2) 
    (h4 : T - (T / 5 + T / 4 + T / 2) = 25) : 
  T = 500 := by 
  sorry

end total_number_of_students_l222_222993


namespace problem_l222_222829

def remainder_when_divided_by_20 (a b : ℕ) : ℕ := (a + b) % 20

theorem problem (a b : ℕ) (n m : ℤ) (h1 : a = 60 * n + 53) (h2 : b = 50 * m + 24) : 
  remainder_when_divided_by_20 a b = 17 := 
by
  -- Proof would go here
  sorry

end problem_l222_222829


namespace solve_for_y_l222_222904

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end solve_for_y_l222_222904


namespace find_y_l222_222978

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l222_222978


namespace sum_mod_9_l222_222527

theorem sum_mod_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  sorry

end sum_mod_9_l222_222527


namespace find_t_l222_222834

variable {x y z w t : ℝ}

theorem find_t (hx : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
               (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
               (hxy : x + 1/y = t)
               (hyz : y + 1/z = t)
               (hzw : z + 1/w = t)
               (hwx : w + 1/x = t) : 
               t = Real.sqrt 2 :=
by
  sorry

end find_t_l222_222834


namespace flowers_per_pot_l222_222524

def total_gardens : ℕ := 10
def pots_per_garden : ℕ := 544
def total_flowers : ℕ := 174080

theorem flowers_per_pot  :
  (total_flowers / (total_gardens * pots_per_garden)) = 32 :=
by
  -- Here would be the place to provide the proof, but we use sorry for now
  sorry

end flowers_per_pot_l222_222524


namespace num_valid_10_digit_sequences_l222_222247

theorem num_valid_10_digit_sequences : 
  ∃ (n : ℕ), n = 64 ∧ 
  (∀ (seq : Fin 10 → Fin 3), 
    (∀ i : Fin 9, abs (seq i.succ - seq i) = 1) → 
    (∀ i : Fin 10, seq i < 3) →
    ∃ k : Nat, k = 10 ∧ seq 0 < 10 ∧ seq 1 < 10 ∧ seq 2 < 10 ∧ seq 3 < 10 ∧ 
      seq 4 < 10 ∧ seq 5 < 10 ∧ seq 6 < 10 ∧ seq 7 < 10 ∧ 
      seq 8 < 10 ∧ seq 9 < 10 ∧ k = 10 → n = 64) :=
sorry

end num_valid_10_digit_sequences_l222_222247


namespace total_age_is_47_l222_222218

-- Define the ages of B and conditions
def B : ℕ := 18
def A : ℕ := B + 2
def C : ℕ := B / 2

-- Prove the total age of A, B, and C
theorem total_age_is_47 : A + B + C = 47 :=
by
  sorry

end total_age_is_47_l222_222218


namespace cube_sum_l222_222448

theorem cube_sum (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) : x^3 + y^3 = 836 := 
by
  sorry

end cube_sum_l222_222448


namespace sum_of_ages_l222_222635

theorem sum_of_ages (y : ℕ) 
  (h_diff : 38 - y = 2) : y + 38 = 74 := 
by {
  sorry
}

end sum_of_ages_l222_222635


namespace exists_distinct_group_and_country_selection_l222_222880

theorem exists_distinct_group_and_country_selection 
  (n m : ℕ) 
  (h_nm1 : n > m) 
  (h_m1 : m > 1) 
  (groups : Fin n → Fin m → Fin n → Prop) 
  (group_conditions : ∀ i j : Fin n, ∀ k : Fin m, ∀ l : Fin m, (i ≠ j) → (groups i k j = false)) 
  : 
  ∃ (selected : Fin n → Fin (m * n)), 
    (∀ i j: Fin n, i ≠ j → selected i ≠ selected j) ∧ 
    (∀ i j: Fin n, selected i / m ≠ selected j / m) := sorry

end exists_distinct_group_and_country_selection_l222_222880


namespace total_revenue_correct_l222_222875

-- Define the conditions
def charge_per_slice : ℕ := 5
def slices_per_pie : ℕ := 4
def pies_sold : ℕ := 9

-- Prove the question: total revenue
theorem total_revenue_correct : charge_per_slice * slices_per_pie * pies_sold = 180 :=
by
  sorry

end total_revenue_correct_l222_222875


namespace number_of_boys_in_school_l222_222241

variable (x : ℕ) (y : ℕ)

theorem number_of_boys_in_school 
    (h1 : 1200 = x + (1200 - x))
    (h2 : 200 = y + (y + 10))
    (h3 : 105 / 200 = (x : ℝ) / 1200) 
    : x = 630 := 
  by 
  sorry

end number_of_boys_in_school_l222_222241


namespace boats_left_l222_222943

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end boats_left_l222_222943


namespace sum_of_squares_of_consecutive_integers_l222_222392

theorem sum_of_squares_of_consecutive_integers :
  ∃ x : ℕ, x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) ∧ (x^2 + (x + 1)^2 + (x + 2)^2 = 77) :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l222_222392


namespace rectangle_dimension_area_l222_222204

theorem rectangle_dimension_area (x : Real) 
  (h_dim1 : x + 3 > 0) 
  (h_dim2 : 3 * x - 2 > 0) :
  ((x + 3) * (3 * x - 2) = 9 * x + 1) ↔ x = (11 + Real.sqrt 205) / 6 := 
sorry

end rectangle_dimension_area_l222_222204


namespace employed_males_percent_l222_222640

def percent_employed_population : ℝ := 96
def percent_females_among_employed : ℝ := 75

theorem employed_males_percent :
  percent_employed_population * (1 - percent_females_among_employed / 100) = 24 := by
    sorry

end employed_males_percent_l222_222640


namespace find_pairs_l222_222950

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ n : ℕ, (n > 0) ∧ (a = n ∧ b = n) ∨ (a = n ∧ b = 1)) ↔ 
  (a^3 ∣ b^2) ∧ ((b - 1) ∣ (a - 1)) :=
by {
  sorry
}

end find_pairs_l222_222950


namespace simplify_expression_l222_222670

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l222_222670


namespace ratio_problem_l222_222068

variable {a b c d : ℚ}

theorem ratio_problem (h₁ : a / b = 5) (h₂ : c / b = 3) (h₃ : c / d = 2) :
  d / a = 3 / 10 :=
sorry

end ratio_problem_l222_222068


namespace student_answered_two_questions_incorrectly_l222_222459

/-
  Defining the variables and conditions for the problem.
  x: number of questions answered correctly,
  y: number of questions not answered,
  z: number of questions answered incorrectly.
-/

theorem student_answered_two_questions_incorrectly (x y z : ℕ) 
  (h1 : x + y + z = 6) 
  (h2 : 8 * x + 2 * y = 20) : z = 2 :=
by
  /- We know the total number of questions is 6.
     And the total score is 20 with the given scoring rules.
     Thus, we need to prove that z = 2 under these conditions. -/
  sorry

end student_answered_two_questions_incorrectly_l222_222459


namespace employee_total_weekly_pay_l222_222431

-- Define the conditions
def hours_per_day_first_3_days : ℕ := 6
def hours_per_day_last_2_days : ℕ := 2 * hours_per_day_first_3_days
def first_40_hours_pay_rate : ℕ := 30
def overtime_multiplier : ℕ := 3 / 2 -- 50% more pay, i.e., 1.5 times

-- Functions to compute total hours worked and total pay
def hours_first_3_days (d : ℕ) : ℕ := d * hours_per_day_first_3_days
def hours_last_2_days (d : ℕ) : ℕ := d * hours_per_day_last_2_days
def total_hours_worked : ℕ := (hours_first_3_days 3) + (hours_last_2_days 2)
def regular_hours : ℕ := min 40 total_hours_worked
def overtime_hours : ℕ := total_hours_worked - regular_hours
def regular_pay : ℕ := regular_hours * first_40_hours_pay_rate
def overtime_pay_rate : ℕ := first_40_hours_pay_rate + (first_40_hours_pay_rate / 2) -- 50% more
def overtime_pay : ℕ := overtime_hours * overtime_pay_rate
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem employee_total_weekly_pay : total_pay = 1290 := by
  sorry

end employee_total_weekly_pay_l222_222431


namespace specific_value_correct_l222_222189

noncomputable def specific_value (x : ℝ) : ℝ :=
  (3 / 5) * (x ^ 2)

theorem specific_value_correct :
  specific_value 14.500000000000002 = 126.15000000000002 :=
by
  sorry

end specific_value_correct_l222_222189


namespace winnie_balloons_rem_l222_222147

theorem winnie_balloons_rem (r w g c : ℕ) (h_r : r = 17) (h_w : w = 33) (h_g : g = 65) (h_c : c = 83) :
  (r + w + g + c) % 8 = 6 := 
by 
  sorry

end winnie_balloons_rem_l222_222147


namespace paul_completion_time_l222_222127

theorem paul_completion_time :
  let george_rate := 1 / 15
  let remaining_work := 2 / 5
  let combined_rate (P : ℚ) := george_rate + P
  let P_work := 4 * combined_rate P = remaining_work
  let paul_rate := 13 / 90
  let total_work := 1
  let time_paul_alone := total_work / paul_rate
  P_work → time_paul_alone = (90 / 13) := by
  intros
  -- all necessary definitions and conditions are used
  sorry

end paul_completion_time_l222_222127


namespace hours_per_day_for_first_group_l222_222860

theorem hours_per_day_for_first_group (h : ℕ) :
  (39 * h * 12 = 30 * 6 * 26) → h = 10 :=
by
  sorry

end hours_per_day_for_first_group_l222_222860


namespace set_subset_of_inter_union_l222_222684

variable {α : Type} [Nonempty α]
variables {A B C : Set α}

-- The main theorem based on the problem statement
theorem set_subset_of_inter_union (h : A ∩ B = B ∪ C) : C ⊆ B :=
by
  sorry

end set_subset_of_inter_union_l222_222684


namespace evaluate_polynomial_at_6_eq_1337_l222_222593

theorem evaluate_polynomial_at_6_eq_1337 :
  (3 * 6^2 + 15 * 6 + 7) + (4 * 6^3 + 8 * 6^2 - 5 * 6 + 10) = 1337 := by
  sorry

end evaluate_polynomial_at_6_eq_1337_l222_222593


namespace initial_workers_l222_222931

theorem initial_workers (W : ℕ) (H1 : (8 * W) / 30 = W) (H2 : (6 * (2 * W - 45)) / 45 = 2 * W - 45) : W = 45 :=
sorry

end initial_workers_l222_222931


namespace cost_of_pen_l222_222687

theorem cost_of_pen 
  (total_amount_spent : ℕ)
  (total_items : ℕ)
  (number_of_pencils : ℕ)
  (cost_of_pencil : ℕ)
  (cost_of_pen : ℕ)
  (h1 : total_amount_spent = 2000)
  (h2 : total_items = 36)
  (h3 : number_of_pencils = 16)
  (h4 : cost_of_pencil = 25)
  (remaining_amount_spent : ℕ)
  (number_of_pens : ℕ)
  (h5 : remaining_amount_spent = total_amount_spent - (number_of_pencils * cost_of_pencil))
  (h6 : number_of_pens = total_items - number_of_pencils)
  (total_cost_of_pens : ℕ)
  (h7 : total_cost_of_pens = remaining_amount_spent)
  (h8 : total_cost_of_pens = number_of_pens * cost_of_pen)
  : cost_of_pen = 80 := by
  sorry

end cost_of_pen_l222_222687


namespace fruit_problem_l222_222360

variables (A O x : ℕ) -- Natural number variables for apples, oranges, and oranges put back

theorem fruit_problem :
  (A + O = 10) ∧
  (40 * A + 60 * O = 480) ∧
  (240 + 60 * (O - x) = 45 * (10 - x)) →
  A = 6 ∧ O = 4 ∧ x = 2 :=
  sorry

end fruit_problem_l222_222360


namespace Richard_walked_10_miles_third_day_l222_222857

def distance_to_NYC := 70
def day1 := 20
def day2 := (day1 / 2) - 6
def remaining_distance := 36
def day3 := 70 - (day1 + day2 + remaining_distance)

theorem Richard_walked_10_miles_third_day (h : day3 = 10) : day3 = 10 :=
by {
    sorry
}

end Richard_walked_10_miles_third_day_l222_222857
