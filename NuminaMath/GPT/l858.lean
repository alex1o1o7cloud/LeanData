import Mathlib

namespace total_blue_balloons_l858_85840

def Joan_balloons : Nat := 9
def Sally_balloons : Nat := 5
def Jessica_balloons : Nat := 2

theorem total_blue_balloons : Joan_balloons + Sally_balloons + Jessica_balloons = 16 :=
by
  sorry

end total_blue_balloons_l858_85840


namespace initial_puppies_l858_85899

-- Define the conditions
variable (a : ℕ) (t : ℕ) (p_added : ℕ) (p_total_adopted : ℕ)

-- State the theorem with the conditions and the target proof
theorem initial_puppies
  (h₁ : a = 3) 
  (h₂ : t = 2)
  (h₃ : p_added = 3)
  (h₄ : p_total_adopted = a * t) :
  (p_total_adopted - p_added) = 3 :=
sorry

end initial_puppies_l858_85899


namespace minimum_sum_of_squares_l858_85853

theorem minimum_sum_of_squares (α p q : ℝ) 
  (h1: p + q = α - 2) (h2: p * q = - (α + 1)) :
  p^2 + q^2 ≥ 5 :=
by
  sorry

end minimum_sum_of_squares_l858_85853


namespace parabola_intersects_line_exactly_once_l858_85822

theorem parabola_intersects_line_exactly_once (p q : ℚ) : 
  (∀ x : ℝ, 2 * (x - p) ^ 2 = x - 4 ↔ p = 31 / 8) ∧ 
  (∀ x : ℝ, 2 * x ^ 2 - q = x - 4 ↔ q = 31 / 8) := 
by 
  sorry

end parabola_intersects_line_exactly_once_l858_85822


namespace range_of_a_l858_85881

noncomputable def proposition_p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
noncomputable def proposition_q (x : ℝ) (a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ (∃ x, ¬ proposition_p x) → ¬ (∃ x, ¬ proposition_q x a)) →
  (¬ (¬ (∃ x, ¬ proposition_p x) ∧ ¬ (¬ (∃ x, ¬ proposition_q x a)))) →
  (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  intro h₁ h₂
  sorry

end range_of_a_l858_85881


namespace book_cost_price_l858_85843

theorem book_cost_price 
  (M : ℝ) (hM : M = 64.54) 
  (h1 : ∃ L : ℝ, 0.92 * L = M ∧ L = 1.25 * 56.12) :
  ∃ C : ℝ, C = 56.12 :=
by
  sorry

end book_cost_price_l858_85843


namespace c_share_of_rent_l858_85859

/-- 
Given the conditions:
- a puts 10 oxen for 7 months,
- b puts 12 oxen for 5 months,
- c puts 15 oxen for 3 months,
- The rent of the pasture is Rs. 210,
Prove that C should pay Rs. 54 as his share of rent.
-/
noncomputable def total_rent : ℝ := 210
noncomputable def oxen_months_a : ℝ := 10 * 7
noncomputable def oxen_months_b : ℝ := 12 * 5
noncomputable def oxen_months_c : ℝ := 15 * 3
noncomputable def total_oxen_months : ℝ := oxen_months_a + oxen_months_b + oxen_months_c

theorem c_share_of_rent : (total_rent / total_oxen_months) * oxen_months_c = 54 :=
by
  sorry

end c_share_of_rent_l858_85859


namespace shaded_region_area_and_circle_centers_l858_85826

theorem shaded_region_area_and_circle_centers :
  ∃ (R : ℝ) (center_big center_small1 center_small2 : ℝ × ℝ),
    R = 10 ∧ 
    center_small1 = (4, 0) ∧
    center_small2 = (10, 0) ∧
    center_big = (7, 0) ∧
    (π * R^2) - (π * 4^2 + π * 6^2) = 48 * π :=
by 
  sorry

end shaded_region_area_and_circle_centers_l858_85826


namespace eq_zero_or_one_if_square_eq_self_l858_85839

theorem eq_zero_or_one_if_square_eq_self (a : ℝ) (h : a^2 = a) : a = 0 ∨ a = 1 :=
sorry

end eq_zero_or_one_if_square_eq_self_l858_85839


namespace mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l858_85815

-- Definition of Mr. Wang's movements
def movements : List Int := [6, -3, 10, -8, 12, -7, -10]

-- Definitions of given conditions
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.3

theorem mr_wang_returned_to_1st_floor :
  (List.sum movements = 0) :=
by
  sorry

theorem mr_wang_electricity_consumption :
  (List.sum (movements.map Int.natAbs) * floor_height * electricity_per_meter = 50.4) :=
by
  sorry

end mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l858_85815


namespace shirt_price_percentage_l858_85866

variable (original_price : ℝ) (final_price : ℝ)

def calculate_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_new_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_final_price (p : ℝ) : ℝ := 0.85 * p

theorem shirt_price_percentage :
  (original_price = 60) →
  (final_price = calculate_final_price (calculate_new_sale_price (calculate_sale_price original_price))) →
  (final_price / original_price) * 100 = 54.4 :=
by
  intros h₁ h₂
  sorry

end shirt_price_percentage_l858_85866


namespace inequality_proof_l858_85880

noncomputable def inequality (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : Prop :=
  (a * b) / (c - 1) + (b * c) / (a - 1) + (c * a) / (b - 1) >= 12

theorem inequality_proof (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : inequality a b c ha hb hc :=
by
  sorry

end inequality_proof_l858_85880


namespace michael_total_time_l858_85817

def time_for_200_meters (distance speed : ℕ) : ℚ :=
  distance / speed

def total_time_per_lap : ℚ :=
  (time_for_200_meters 200 6) + (time_for_200_meters 200 3)

def total_time_8_laps : ℚ :=
  8 * total_time_per_lap

theorem michael_total_time : total_time_8_laps = 800 :=
by
  -- The proof would go here
  sorry

end michael_total_time_l858_85817


namespace min_g_l858_85809

noncomputable def f (a m x : ℝ) := m + Real.log x / Real.log a -- definition of f(x) = m + logₐ(x)

-- Given conditions
variables (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
variables (m : ℝ)
axiom h_f8 : f a m 8 = 2
axiom h_f1 : f a m 1 = -1

-- Derived expressions
noncomputable def g (x : ℝ) := 2 * f a m x - f a m (x - 1)

-- Theorem statement
theorem min_g : ∃ (x : ℝ), x > 1 ∧ g a m x = 1 ∧ ∀ x' > 1, g a m x' ≥ 1 :=
sorry

end min_g_l858_85809


namespace brick_length_l858_85873

theorem brick_length 
  (width : ℝ) (height : ℝ) (num_bricks : ℕ)
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_vol : ℝ) :
  width = 10 →
  height = 7.5 →
  num_bricks = 27000 →
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_vol = width * height * (20:ℝ) →
  wall_length * wall_width * wall_height * 1000000 = num_bricks * brick_vol :=
by
  intros
  sorry

end brick_length_l858_85873


namespace lcm_prime_factors_l858_85856

-- Conditions
def n1 := 48
def n2 := 180
def n3 := 250

-- The equivalent proof problem
theorem lcm_prime_factors (l : ℕ) (h1: l = Nat.lcm n1 (Nat.lcm n2 n3)) :
  l = 18000 ∧ (∀ a : ℕ, a ∣ l ↔ a ∣ 2^4 * 3^2 * 5^3) :=
by
  sorry

end lcm_prime_factors_l858_85856


namespace inflated_cost_per_person_l858_85830

def estimated_cost : ℝ := 30e9
def people_sharing : ℝ := 200e6
def inflation_rate : ℝ := 0.05

theorem inflated_cost_per_person :
  (estimated_cost * (1 + inflation_rate)) / people_sharing = 157.5 := by
  sorry

end inflated_cost_per_person_l858_85830


namespace vector_coordinates_l858_85811

theorem vector_coordinates (A B : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (-1, 2)) :
  B - A = (-1, 1) :=
sorry

end vector_coordinates_l858_85811


namespace sum_of_values_for_one_solution_l858_85831

noncomputable def sum_of_a_values (a1 a2 : ℝ) : ℝ :=
  a1 + a2

theorem sum_of_values_for_one_solution :
  ∃ a1 a2 : ℝ, 
  (∀ x : ℝ, 4 * x^2 + (a1 + 8) * x + 9 = 0 ∨ 4 * x^2 + (a2 + 8) * x + 9 = 0) ∧
  ((a1 + 8)^2 - 144 = 0) ∧ ((a2 + 8)^2 - 144 = 0) ∧
  sum_of_a_values a1 a2 = -16 :=
by
  sorry

end sum_of_values_for_one_solution_l858_85831


namespace pancake_fundraiser_l858_85879

-- Define the constants and conditions
def cost_per_stack_of_pancakes : ℕ := 4
def cost_per_slice_of_bacon : ℕ := 2
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90
def total_raised : ℕ := 420

-- Define a theorem that states what we want to prove
theorem pancake_fundraiser : 
  (stacks_sold * cost_per_stack_of_pancakes + slices_sold * cost_per_slice_of_bacon) = total_raised :=
by
  sorry -- We place a sorry here to skip the proof, as instructed.

end pancake_fundraiser_l858_85879


namespace mark_first_vaccine_wait_time_l858_85829

-- Define the variables and conditions
variable (x : ℕ)
variable (total_wait_time : ℕ)
variable (second_appointment_wait : ℕ)
variable (effectiveness_wait : ℕ)

-- Given conditions
axiom h1 : second_appointment_wait = 20
axiom h2 : effectiveness_wait = 14
axiom h3 : total_wait_time = 38

-- The statement to be proven
theorem mark_first_vaccine_wait_time
  (h4 : x + second_appointment_wait + effectiveness_wait = total_wait_time) :
  x = 4 := by
  sorry

end mark_first_vaccine_wait_time_l858_85829


namespace average_speed_l858_85861

theorem average_speed (d1 d2 d3 v1 v2 v3 total_distance total_time avg_speed : ℝ)
    (h1 : d1 = 40) (h2 : d2 = 20) (h3 : d3 = 10) 
    (h4 : v1 = 8) (h5 : v2 = 40) (h6 : v3 = 20) 
    (h7 : total_distance = d1 + d2 + d3)
    (h8 : total_time = d1 / v1 + d2 / v2 + d3 / v3) 
    (h9 : avg_speed = total_distance / total_time) : avg_speed = 11.67 :=
by 
  sorry

end average_speed_l858_85861


namespace Adeline_hourly_wage_l858_85849

theorem Adeline_hourly_wage
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 9) 
  (h2 : days_per_week = 5) 
  (h3 : weeks = 7) 
  (h4 : total_earnings = 3780) :
  total_earnings = 12 * (hours_per_day * days_per_week * weeks) :=
by
  sorry

end Adeline_hourly_wage_l858_85849


namespace cylinder_base_ratio_l858_85870

variable (O : Point) -- origin
variable (a b c : ℝ) -- fixed point
variable (p q : ℝ) -- center of circular base
variable (α β : ℝ) -- intersection points with axis

-- Let O be the origin
-- Let (a, b, c) be the fixed point through which the cylinder passes
-- The cylinder's axis is parallel to the z-axis and the center of its base is (p, q)
-- The cylinder intersects the x-axis at (α, 0, 0) and the y-axis at (0, β, 0)
-- Let α = 2p and β = 2q

theorem cylinder_base_ratio : 
  α = 2 * p ∧ β = 2 * q → (a / p + b / q = 4) := by
  sorry

end cylinder_base_ratio_l858_85870


namespace continuous_function_solution_l858_85857

theorem continuous_function_solution (f : ℝ → ℝ) (a : ℝ) (h_continuous : Continuous f) (h_pos : 0 < a)
    (h_equation : ∀ x, f x = a^x * f (x / 2)) :
    ∃ C : ℝ, ∀ x, f x = C * a^(2 * x) := 
sorry

end continuous_function_solution_l858_85857


namespace mask_production_l858_85895

theorem mask_production (M : ℕ) (h : 16 * M = 48000) : M = 3000 :=
by
  sorry

end mask_production_l858_85895


namespace ned_long_sleeve_shirts_l858_85835

-- Define the conditions
def total_shirts_washed_before_school : ℕ := 29
def short_sleeve_shirts : ℕ := 9
def unwashed_shirts : ℕ := 1

-- Define the proof problem
theorem ned_long_sleeve_shirts (total_shirts_washed_before_school short_sleeve_shirts unwashed_shirts: ℕ) : 
(total_shirts_washed_before_school - unwashed_shirts - short_sleeve_shirts) = 19 :=
by
  -- It is given: 29 total shirts - 1 unwashed shirt = 28 washed shirts
  -- Out of the 28 washed shirts, 9 are short sleeve shirts
  -- Therefore, Ned washed 28 - 9 = 19 long sleeve shirts
  sorry

end ned_long_sleeve_shirts_l858_85835


namespace area_of_shaded_region_l858_85801

-- Definitions of given conditions
def octagon_side_length : ℝ := 5
def arc_radius : ℝ := 4

-- Theorem statement
theorem area_of_shaded_region : 
  let octagon_area := 50
  let sectors_area := 16 * Real.pi
  octagon_area - sectors_area = 50 - 16 * Real.pi :=
by
  sorry

end area_of_shaded_region_l858_85801


namespace area_of_region_l858_85888

-- The problem definition
def condition_1 (z : ℂ) : Prop := 
  0 < z.re / 20 ∧ z.re / 20 < 1 ∧
  0 < z.im / 20 ∧ z.im / 20 < 1 ∧
  0 < (20 / z).re ∧ (20 / z).re < 1 ∧
  0 < (20 / z).im ∧ (20 / z).im < 1

-- The proof statement
theorem area_of_region {z : ℂ} (h : condition_1 z) : 
  ∃ s : ℝ, s = 300 - 50 * Real.pi := sorry

end area_of_region_l858_85888


namespace percentage_of_students_passed_l858_85883

def total_students : ℕ := 740
def failed_students : ℕ := 481
def passed_students : ℕ := total_students - failed_students
def pass_percentage : ℚ := (passed_students / total_students) * 100

theorem percentage_of_students_passed : pass_percentage = 35 := by
  sorry

end percentage_of_students_passed_l858_85883


namespace find_k_in_geometric_sequence_l858_85894

theorem find_k_in_geometric_sequence (c k : ℝ) (h1_nonzero : c ≠ 0)
  (S : ℕ → ℝ) (a : ℕ → ℝ) (h2 : ∀ n, a (n + 1) = c * a n)
  (h3 : ∀ n, S n = 3^n + k)
  (h4 : a 1 = 3 + k)
  (h5 : a 2 = S 2 - S 1)
  (h6 : a 3 = S 3 - S 2) : k = -1 := by
  sorry

end find_k_in_geometric_sequence_l858_85894


namespace midpoint_sum_of_coordinates_l858_85892

theorem midpoint_sum_of_coordinates : 
  let p1 := (8, 10)
  let p2 := (-4, -10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 2 :=
by
  sorry

end midpoint_sum_of_coordinates_l858_85892


namespace domain_of_sqrt_ln_l858_85889

def domain_function (x : ℝ) : Prop := x - 1 ≥ 0 ∧ 2 - x > 0

theorem domain_of_sqrt_ln (x : ℝ) : domain_function x ↔ 1 ≤ x ∧ x < 2 := by
  sorry

end domain_of_sqrt_ln_l858_85889


namespace inequality_holds_for_all_x_l858_85824

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end inequality_holds_for_all_x_l858_85824


namespace find_p_l858_85897

theorem find_p (p : ℕ) (h : 81^6 = 3^p) : p = 24 :=
sorry

end find_p_l858_85897


namespace xyz_squared_sum_l858_85800

theorem xyz_squared_sum (x y z : ℝ) 
  (h1 : x^2 + 4 * y^2 + 16 * z^2 = 48)
  (h2 : x * y + 4 * y * z + 2 * z * x = 24) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end xyz_squared_sum_l858_85800


namespace sunzi_wood_problem_l858_85828

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l858_85828


namespace fg_neg_one_eq_neg_eight_l858_85805

def f (x : ℤ) : ℤ := x - 4
def g (x : ℤ) : ℤ := x^2 + 2*x - 3

theorem fg_neg_one_eq_neg_eight : f (g (-1)) = -8 := by
  sorry

end fg_neg_one_eq_neg_eight_l858_85805


namespace value_of_m_l858_85854

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end value_of_m_l858_85854


namespace train_speed_correct_l858_85812

-- Definitions based on the conditions in a)
def train_length_meters : ℝ := 160
def time_seconds : ℝ := 4

-- Correct answer identified in b)
def expected_speed_kmh : ℝ := 144

-- Proof statement verifying that speed computed from the conditions equals the expected speed
theorem train_speed_correct :
  train_length_meters / 1000 / (time_seconds / 3600) = expected_speed_kmh :=
by
  sorry

end train_speed_correct_l858_85812


namespace carA_catches_up_with_carB_at_150_km_l858_85872

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l858_85872


namespace total_balloons_is_18_l858_85842

-- Define the number of balloons each person has
def Fred_balloons : Nat := 5
def Sam_balloons : Nat := 6
def Mary_balloons : Nat := 7

-- Define the total number of balloons
def total_balloons : Nat := Fred_balloons + Sam_balloons + Mary_balloons

-- The theorem statement to prove
theorem total_balloons_is_18 : total_balloons = 18 := sorry

end total_balloons_is_18_l858_85842


namespace skew_lines_angle_range_l858_85896

theorem skew_lines_angle_range (θ : ℝ) (h_skew : θ > 0 ∧ θ ≤ 90) :
  0 < θ ∧ θ ≤ 90 :=
sorry

end skew_lines_angle_range_l858_85896


namespace hcf_of_two_numbers_900_l858_85846

theorem hcf_of_two_numbers_900 (A B H : ℕ) (h_lcm : lcm A B = H * 11 * 15) (h_A : A = 900) : gcd A B = 165 :=
by
  sorry

end hcf_of_two_numbers_900_l858_85846


namespace joan_gave_sam_43_seashells_l858_85804

def joan_original_seashells : ℕ := 70
def joan_seashells_left : ℕ := 27
def seashells_given_to_sam : ℕ := 43

theorem joan_gave_sam_43_seashells :
  joan_original_seashells - joan_seashells_left = seashells_given_to_sam :=
by
  sorry

end joan_gave_sam_43_seashells_l858_85804


namespace rainfall_ratio_l858_85858

theorem rainfall_ratio (r_wed tuesday_rate : ℝ)
    (h_monday : 7 * 1 = 7)
    (h_tuesday : 4 * 2 = 8)
    (h_total : 7 + 8 + 2 * r_wed = 23)
    (h_wed_eq: r_wed = 8 / 2)
    (h_tuesday_rate: tuesday_rate = 2) 
    : r_wed / tuesday_rate = 2 :=
by
  sorry

end rainfall_ratio_l858_85858


namespace wrong_mark_is_43_l858_85825

theorem wrong_mark_is_43
  (correct_mark : ℕ)
  (wrong_mark : ℕ)
  (num_students : ℕ)
  (avg_increase : ℕ)
  (h_correct : correct_mark = 63)
  (h_num_students : num_students = 40)
  (h_avg_increase : avg_increase = 40 / 2) 
  (h_wrong_avg : (num_students - 1) * (correct_mark + avg_increase) / num_students = (num_students - 1) * (wrong_mark + avg_increase + correct_mark) / num_students) :
  wrong_mark = 43 :=
sorry

end wrong_mark_is_43_l858_85825


namespace standard_deviation_is_two_l858_85818

def weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
  ((l.map (λ x => (x - mean l)^2)).sum / l.length)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_two : standard_deviation weights = 2 := 
by
  sorry

end standard_deviation_is_two_l858_85818


namespace non_prime_in_sequence_l858_85887

theorem non_prime_in_sequence : ∃ n : ℕ, ¬Prime (41 + n * (n - 1)) :=
by {
  use 41,
  sorry
}

end non_prime_in_sequence_l858_85887


namespace intersecting_lines_implies_a_eq_c_l858_85852

theorem intersecting_lines_implies_a_eq_c
  (k b a c : ℝ)
  (h_kb : k ≠ b)
  (exists_point : ∃ (x y : ℝ), (y = k * x + k) ∧ (y = b * x + b) ∧ (y = a * x + c)) :
  a = c := 
sorry

end intersecting_lines_implies_a_eq_c_l858_85852


namespace combined_supply_duration_l858_85851

variable (third_of_pill_per_third_day : ℕ → Prop)
variable (alternate_days : ℕ → ℕ → Prop)
variable (supply : ℕ)
variable (days_in_month : ℕ)

-- Conditions:
def one_third_per_third_day (p: ℕ) (d: ℕ) : Prop := 
  third_of_pill_per_third_day d ∧ alternate_days d (d + 3)
def total_supply (s: ℕ) := s = 60
def duration_per_pill (d: ℕ) := d = 9
def month_days (m: ℕ) := m = 30

-- Proof Problem Statement:
theorem combined_supply_duration :
  ∀ (s t: ℕ), total_supply s ∧ duration_per_pill t ∧ month_days 30 → 
  (s * t / 30) = 18 :=
by
  intros s t h
  sorry

end combined_supply_duration_l858_85851


namespace minimum_value_inequality_l858_85837

theorem minimum_value_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3 / 2) :=
sorry

end minimum_value_inequality_l858_85837


namespace total_volume_of_all_cubes_l858_85891

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end total_volume_of_all_cubes_l858_85891


namespace cubic_polynomial_roots_value_l858_85886

theorem cubic_polynomial_roots_value
  (a b c d : ℝ) 
  (h_cond : a ≠ 0 ∧ d ≠ 0)
  (h_equiv : (a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + d) + (a * (-1/2)^3 + b * (-1/2)^2 + c * (-1/2) + d) = 1000 * d)
  (h_roots : ∃ (x1 x2 x3 : ℝ), a * x1^3 + b * x1^2 + c * x1 + d = 0 ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
  : (∃ (x1 x2 x3 : ℝ), (1 / (x1 * x2) + 1 / (x2 * x3) + 1 / (x1 * x3) = 1996)) :=
by
  sorry

end cubic_polynomial_roots_value_l858_85886


namespace solve_for_a_and_b_l858_85820

theorem solve_for_a_and_b (a b : ℤ) :
  (∀ x : ℤ, (x + a) * (x - 2) = x^2 + b * x - 6) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_for_a_and_b_l858_85820


namespace cake_eating_contest_l858_85876

-- Define the fractions representing the amounts of cake eaten by the two students.
def first_student : ℚ := 7 / 8
def second_student : ℚ := 5 / 6

-- The statement of our proof problem
theorem cake_eating_contest : first_student - second_student = 1 / 24 := by
  sorry

end cake_eating_contest_l858_85876


namespace houses_with_neither_l858_85833

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither_l858_85833


namespace width_of_boxes_l858_85885

theorem width_of_boxes
  (total_volume : ℝ)
  (total_payment : ℝ)
  (cost_per_box : ℝ)
  (h1 : total_volume = 1.08 * 10^6)
  (h2 : total_payment = 120)
  (h3 : cost_per_box = 0.2) :
  (∃ w : ℝ, w = (total_volume / (total_payment / cost_per_box))^(1/3)) :=
by {
  sorry
}

end width_of_boxes_l858_85885


namespace sum_of_digits_l858_85819

theorem sum_of_digits (P Q R : ℕ) (hP : P < 10) (hQ : Q < 10) (hR : R < 10)
 (h_sum : P * 1000 + Q * 100 + Q * 10 + R = 2009) : P + Q + R = 10 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l858_85819


namespace student_percentage_l858_85807

theorem student_percentage (s1 s3 overall : ℕ) (percentage_second_subject : ℕ) :
  s1 = 60 →
  s3 = 85 →
  overall = 75 →
  (s1 + percentage_second_subject + s3) / 3 = overall →
  percentage_second_subject = 80 := by
  intros h1 h2 h3 h4
  sorry

end student_percentage_l858_85807


namespace angle_equivalence_modulo_l858_85841

-- Defining the given angles
def theta1 : ℤ := -510
def theta2 : ℤ := 210

-- Proving that the angles are equivalent modulo 360
theorem angle_equivalence_modulo : theta1 % 360 = theta2 % 360 :=
by sorry

end angle_equivalence_modulo_l858_85841


namespace sufficiency_condition_l858_85874

-- Definitions of p and q
def p (a b : ℝ) : Prop := a > |b|
def q (a b : ℝ) : Prop := a^2 > b^2

-- Main theorem statement
theorem sufficiency_condition (a b : ℝ) : (p a b → q a b) ∧ (¬(q a b → p a b)) := 
by
  sorry

end sufficiency_condition_l858_85874


namespace smallest_number_divisible_1_through_12_and_15_l858_85884

theorem smallest_number_divisible_1_through_12_and_15 :
  ∃ n, (∀ i, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ∧ 15 ∣ n ∧ n = 27720 :=
by {
  sorry
}

end smallest_number_divisible_1_through_12_and_15_l858_85884


namespace monotonicity_of_g_l858_85838

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb a (|x + 1|)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.logb a (- (3 / 2) * x^2 + a * x)

theorem monotonicity_of_g (a : ℝ) (h : 0 < a ∧ a ≠ 1) (h0 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x a < 0) :
  ∀ x : ℝ, 0 < x ∧ x ≤ a / 3 → (g x a) < (g (x + ε) a) := 
sorry


end monotonicity_of_g_l858_85838


namespace remainder_range_l858_85898

theorem remainder_range (x y z a b c d e : ℕ)
(h1 : x % 211 = a) (h2 : y % 211 = b) (h3 : z % 211 = c)
(h4 : x % 251 = c) (h5 : y % 251 = d) (h6 : z % 251 = e)
(h7 : a < 211) (h8 : b < 211) (h9 : c < 211)
(h10 : c < 251) (h11 : d < 251) (h12 : e < 251) :
0 ≤ (2 * x - y + 3 * z + 47) % (211 * 251) ∧
(2 * x - y + 3 * z + 47) % (211 * 251) < (211 * 251) :=
by
  sorry

end remainder_range_l858_85898


namespace MrKozelGarden_l858_85836

theorem MrKozelGarden :
  ∀ (x y : ℕ), 
  (y = 3 * x + 1) ∧ (y = 4 * (x - 1)) → (x = 5 ∧ y = 16) := 
by
  intros x y h
  sorry

end MrKozelGarden_l858_85836


namespace find_base_l858_85867

def distinct_three_digit_numbers (b : ℕ) : ℕ :=
    (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)

theorem find_base (b : ℕ) (h : distinct_three_digit_numbers b = 144) : b = 9 :=
by 
  sorry

end find_base_l858_85867


namespace find_x_l858_85890

theorem find_x (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := 
sorry

end find_x_l858_85890


namespace quadratic_monotonic_range_l858_85878

theorem quadratic_monotonic_range {a : ℝ} :
  (∀ x1 x2 : ℝ, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → (x1^2 - 2*a*x1 + 1) ≤ (x2^2 - 2*a*x2 + 1) ∨ (x1^2 - 2*a*x1 + 1) ≥ (x2^2 - 2*a*x2 + 1)) → (a ≤ 2 ∨ a ≥ 3) := 
sorry

end quadratic_monotonic_range_l858_85878


namespace probability_correct_l858_85832

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l858_85832


namespace limiting_reactant_and_products_l858_85847

def balanced_reaction 
  (al_moles : ℕ) (h2so4_moles : ℕ) 
  (al2_so4_3_moles : ℕ) (h2_moles : ℕ) : Prop :=
  2 * al_moles >= 0 ∧ 3 * h2so4_moles >= 0 ∧ 
  al_moles = 2 ∧ h2so4_moles = 3 ∧ 
  al2_so4_3_moles = 1 ∧ h2_moles = 3 ∧ 
  (2 : ℕ) * al_moles + (3 : ℕ) * h2so4_moles = 2 * 2 + 3 * 3

theorem limiting_reactant_and_products :
  balanced_reaction 2 3 1 3 :=
by {
  -- Here we would provide the proof based on the conditions and balances provided in the problem statement.
  sorry
}

end limiting_reactant_and_products_l858_85847


namespace range_of_a_l858_85834

noncomputable def f (a : ℝ) (x : ℝ) := x * Real.log x - a * x^2

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l858_85834


namespace quadratic_vertex_properties_l858_85860

theorem quadratic_vertex_properties (a : ℝ) (x1 x2 y1 y2 : ℝ) (h_ax : a ≠ 0) (h_sum : x1 + x2 = 2) (h_order : x1 < x2) (h_value : y1 > y2) :
  a < -2 / 5 :=
sorry

end quadratic_vertex_properties_l858_85860


namespace solve_quadratic_1_solve_quadratic_2_l858_85863

theorem solve_quadratic_1 (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l858_85863


namespace Bill_donut_combinations_correct_l858_85827

/-- Bill has to purchase exactly six donuts from a shop with four kinds of donuts, ensuring he gets at least one of each kind. -/
def Bill_donut_combinations : ℕ :=
  let k := 4  -- number of kinds of donuts
  let n := 6  -- total number of donuts Bill needs to buy
  let m := 2  -- remaining donuts after buying one of each kind
  let same_kind := k          -- ways to choose 2 donuts of the same kind
  let different_kind := (k * (k - 1)) / 2  -- ways to choose 2 donuts of different kinds
  same_kind + different_kind

theorem Bill_donut_combinations_correct : Bill_donut_combinations = 10 :=
  by
    sorry  -- Proof is omitted; we assert this statement is true

end Bill_donut_combinations_correct_l858_85827


namespace sqrt_range_l858_85882

theorem sqrt_range (x : ℝ) (hx : 0 ≤ x - 1) : 1 ≤ x :=
by sorry

end sqrt_range_l858_85882


namespace stanleyRanMore_l858_85813

def distanceStanleyRan : ℝ := 0.4
def distanceStanleyWalked : ℝ := 0.2

theorem stanleyRanMore : distanceStanleyRan - distanceStanleyWalked = 0.2 := by
  sorry

end stanleyRanMore_l858_85813


namespace city_A_fare_higher_than_city_B_l858_85845

def fare_in_city_A (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

def fare_in_city_B (x : ℝ) : ℝ :=
  8 + 2.5 * (x - 3)

theorem city_A_fare_higher_than_city_B (x : ℝ) (h : x > 3) :
  fare_in_city_A x > fare_in_city_B x → 3 < x ∧ x < 7 :=
by
  sorry

end city_A_fare_higher_than_city_B_l858_85845


namespace woman_year_of_birth_l858_85803

def year_of_birth (x : ℕ) : ℕ := x^2 - x

theorem woman_year_of_birth : ∃ (x : ℕ), 1850 ≤ year_of_birth x ∧ year_of_birth x < 1900 ∧ year_of_birth x = 1892 :=
by
  sorry

end woman_year_of_birth_l858_85803


namespace problem_a51_l858_85875

-- Definitions of given conditions
variable {a : ℕ → ℤ}
variable (h1 : ∀ n : ℕ, a (n + 2) - 2 * a (n + 1) + a n = 16)
variable (h2 : a 63 = 10)
variable (h3 : a 89 = 10)

-- Proof problem statement
theorem problem_a51 :
  a 51 = 3658 :=
by
  sorry

end problem_a51_l858_85875


namespace average_star_rating_is_four_l858_85816

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l858_85816


namespace people_got_off_train_l858_85823

theorem people_got_off_train (initial_people : ℕ) (people_left : ℕ) (people_got_off : ℕ) 
  (h1 : initial_people = 48) 
  (h2 : people_left = 31) 
  : people_got_off = 17 := by
  sorry

end people_got_off_train_l858_85823


namespace minimum_positive_temperature_announcement_l858_85877

-- Problem conditions translated into Lean
def num_interactions (x : ℕ) : ℕ := x * (x - 1)
def total_interactions := 132
def total_positive := 78
def total_negative := 54
def positive_temperature_count (x y : ℕ) : ℕ := y * (y - 1)
def negative_temperature_count (x y : ℕ) : ℕ := (x - y) * (x - 1 - y)
def minimum_positive_temperature (x y : ℕ) := 
  x = 12 → 
  total_interactions = total_positive + total_negative →
  total_positive + total_negative = num_interactions x →
  total_positive = positive_temperature_count x y →
  sorry -- proof goes here

theorem minimum_positive_temperature_announcement : ∃ y, 
  minimum_positive_temperature 12 y ∧ y = 3 :=
by {
  sorry -- proof goes here
}

end minimum_positive_temperature_announcement_l858_85877


namespace power_of_negative_fraction_l858_85869

theorem power_of_negative_fraction :
  (- (1/3))^2 = 1/9 := 
by 
  sorry

end power_of_negative_fraction_l858_85869


namespace solve_quadratic_inequality_l858_85871

theorem solve_quadratic_inequality :
  ∀ x : ℝ, ((x - 1) * (x - 3) < 0) ↔ (1 < x ∧ x < 3) :=
by
  intro x
  sorry

end solve_quadratic_inequality_l858_85871


namespace basketball_club_boys_l858_85848

theorem basketball_club_boys (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : B + (1 / 3) * G = 18) : B = 12 := 
by
  sorry

end basketball_club_boys_l858_85848


namespace ratio_suspension_to_fingers_toes_l858_85868

-- Definition of conditions
def suspension_days_per_instance : Nat := 3
def bullying_instances : Nat := 20
def fingers_and_toes : Nat := 20

-- Theorem statement
theorem ratio_suspension_to_fingers_toes :
  (suspension_days_per_instance * bullying_instances) / fingers_and_toes = 3 :=
by
  sorry

end ratio_suspension_to_fingers_toes_l858_85868


namespace total_coins_received_l858_85865

theorem total_coins_received (coins_first_day coins_second_day : ℕ) 
  (h_first_day : coins_first_day = 22) 
  (h_second_day : coins_second_day = 12) : 
  coins_first_day + coins_second_day = 34 := 
by 
  sorry

end total_coins_received_l858_85865


namespace fair_eight_sided_die_probability_l858_85806

def prob_at_least_seven_at_least_four_times (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem fair_eight_sided_die_probability : prob_at_least_seven_at_least_four_times 5 4 (1 / 4) + (1 / 4) ^ 5 = 1 / 64 :=
by
  sorry

end fair_eight_sided_die_probability_l858_85806


namespace zero_is_multiple_of_every_integer_l858_85808

theorem zero_is_multiple_of_every_integer (x : ℤ) : ∃ n : ℤ, 0 = n * x := by
  use 0
  exact (zero_mul x).symm

end zero_is_multiple_of_every_integer_l858_85808


namespace math_olympiad_proof_l858_85844

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l858_85844


namespace total_combinations_meals_l858_85864

-- Define the total number of menu items
def menu_items : ℕ := 12

-- Define the function for computing the number of combinations of meals ordered by three people
def combinations_of_meals (n : ℕ) : ℕ := n * n * n

-- Theorem stating the total number of different combinations of meals is 1728
theorem total_combinations_meals : combinations_of_meals menu_items = 1728 :=
by
  -- Placeholder for actual proof
  sorry

end total_combinations_meals_l858_85864


namespace Kates_hair_length_l858_85810

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l858_85810


namespace average_of_rest_l858_85893

theorem average_of_rest 
  (total_students : ℕ)
  (marks_5_students : ℕ)
  (marks_3_students : ℕ)
  (marks_others : ℕ)
  (average_class : ℚ)
  (remaining_students : ℕ)
  (expected_average : ℚ) 
  (h1 : total_students = 27) 
  (h2 : marks_5_students = 5 * 95) 
  (h3 : marks_3_students = 3 * 0) 
  (h4 : average_class = 49.25925925925926) 
  (h5 : remaining_students = 27 - 5 - 3) 
  (h6 : (marks_5_students + marks_3_students + marks_others) = total_students * average_class)
  : marks_others / remaining_students = expected_average :=
sorry

end average_of_rest_l858_85893


namespace roberto_starting_salary_l858_85814

-- Given conditions as Lean definitions
def current_salary : ℝ := 134400
def previous_salary (S : ℝ) : ℝ := 1.40 * S

-- The proof problem statement
theorem roberto_starting_salary (S : ℝ) 
    (h1 : current_salary = 1.20 * previous_salary S) : 
    S = 80000 :=
by
  -- We will insert the proof here
  sorry

end roberto_starting_salary_l858_85814


namespace girl_speed_l858_85821

theorem girl_speed (distance time : ℝ) (h₁ : distance = 128) (h₂ : time = 32) : distance / time = 4 := 
by 
  rw [h₁, h₂]
  norm_num

end girl_speed_l858_85821


namespace age_ratio_in_future_l858_85850

variables (t j x : ℕ)

theorem age_ratio_in_future:
  (t - 4 = 5 * (j - 4)) → 
  (t - 10 = 6 * (j - 10)) →
  (t + x = 3 * (j + x)) →
  x = 26 := 
by {
  sorry
}

end age_ratio_in_future_l858_85850


namespace avg_of_8_numbers_l858_85855

theorem avg_of_8_numbers
  (n : ℕ)
  (h₁ : n = 8)
  (sum_first_half : ℝ)
  (h₂ : sum_first_half = 158.4)
  (avg_second_half : ℝ)
  (h₃ : avg_second_half = 46.6) :
  ((sum_first_half + avg_second_half * (n / 2)) / n) = 43.1 :=
by
  sorry

end avg_of_8_numbers_l858_85855


namespace problem_statement_l858_85862

theorem problem_statement : ∃ n : ℤ, 0 < n ∧ (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).den = 1 ∧ ¬ n > 96 := 
by 
  sorry

end problem_statement_l858_85862


namespace calculate_paving_cost_l858_85802

theorem calculate_paving_cost
  (length : ℝ) (width : ℝ) (rate_per_sq_meter : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_rate : rate_per_sq_meter = 1200) :
  (length * width * rate_per_sq_meter = 24750) :=
by
  sorry

end calculate_paving_cost_l858_85802
