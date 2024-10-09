import Mathlib

namespace percentage_increase_l491_49125

variables {a b : ℝ} -- Assuming a and b are real numbers

-- Define the conditions explicitly
def initial_workers := a
def workers_left := b
def remaining_workers := a - b

-- Define the theorem for percentage increase in daily performance
theorem percentage_increase (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (100 * b) / (a - b) = (100 * a * b) / (a * (a - b)) :=
by
  sorry -- Proof will be filled in as needed

end percentage_increase_l491_49125


namespace find_parabola_eq_l491_49171

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end find_parabola_eq_l491_49171


namespace words_per_page_l491_49131

/-- 
  Let p denote the number of words per page.
  Given conditions:
  - A book contains 154 pages.
  - Each page has the same number of words, p, and no page contains more than 120 words.
  - The total number of words in the book (154p) is congruent to 250 modulo 227.
  Prove that the number of words in each page p is congruent to 49 modulo 227.
 -/
theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p ≡ 250 [MOD 227]) : p ≡ 49 [MOD 227] :=
sorry

end words_per_page_l491_49131


namespace single_solution_inequality_l491_49136

theorem single_solution_inequality (a : ℝ) :
  (∃! (x : ℝ), abs (x^2 + 2 * a * x + 3 * a) ≤ 2) ↔ a = 1 ∨ a = 2 := 
sorry

end single_solution_inequality_l491_49136


namespace probability_blue_is_approx_50_42_l491_49198

noncomputable def probability_blue_second_pick : ℚ :=
  let yellow := 30
  let green := yellow / 3
  let red := 2 * green
  let total_marbles := 120
  let blue := total_marbles - (yellow + green + red)
  let total_after_first_pick := total_marbles - 1
  let blue_probability := (blue : ℚ) / total_after_first_pick
  blue_probability * 100

theorem probability_blue_is_approx_50_42 :
  abs (probability_blue_second_pick - 50.42) < 0.005 := -- Approximately checking for equality due to possible floating-point precision issues
sorry

end probability_blue_is_approx_50_42_l491_49198


namespace original_mixture_litres_l491_49177

theorem original_mixture_litres 
  (x : ℝ)
  (h1 : 0.20 * x = 0.15 * (x + 5)) :
  x = 15 :=
sorry

end original_mixture_litres_l491_49177


namespace find_slope_angle_l491_49157

theorem find_slope_angle (α : ℝ) :
    (∃ x y : ℝ, x * Real.sin (2 * Real.pi / 5) + y * Real.cos (2 * Real.pi / 5) = 0) →
    α = 3 * Real.pi / 5 :=
by
  intro h
  sorry

end find_slope_angle_l491_49157


namespace least_number_to_subtract_from_724946_l491_49145

def divisible_by_10 (n : ℕ) : Prop :=
  n % 10 = 0

theorem least_number_to_subtract_from_724946 :
  ∃ k : ℕ, k = 6 ∧ divisible_by_10 (724946 - k) :=
by
  sorry

end least_number_to_subtract_from_724946_l491_49145


namespace wine_cost_is_3_60_l491_49154

noncomputable def appetizer_cost : ℕ := 8
noncomputable def steak_cost : ℕ := 20
noncomputable def dessert_cost : ℕ := 6
noncomputable def total_spent : ℝ := 38
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def number_of_wines : ℕ := 2

noncomputable def discounted_steak_cost : ℝ := steak_cost / 2
noncomputable def full_meal_cost : ℝ := appetizer_cost + steak_cost + dessert_cost
noncomputable def meal_cost_after_discount : ℝ := appetizer_cost + discounted_steak_cost + dessert_cost
noncomputable def full_meal_tip := tip_percentage * full_meal_cost
noncomputable def meal_cost_with_tip := meal_cost_after_discount + full_meal_tip
noncomputable def total_wine_cost := total_spent - meal_cost_with_tip
noncomputable def cost_per_wine := total_wine_cost / number_of_wines

theorem wine_cost_is_3_60 : cost_per_wine = 3.60 := by
  sorry

end wine_cost_is_3_60_l491_49154


namespace maximize_x5y3_l491_49158

theorem maximize_x5y3 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x = 18.75 ∧ y = 11.25 → (x^5 * y^3) = (18.75^5 * 11.25^3) :=
sorry

end maximize_x5y3_l491_49158


namespace exist_n_exactly_3_rainy_days_l491_49189

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exist_n_exactly_3_rainy_days (p : ℝ) (k : ℕ) (prob : ℝ) :
  p = 0.5 → k = 3 → prob = 0.25 →
  ∃ n : ℕ, binomial_prob n k p = prob :=
by
  intros h1 h2 h3
  sorry

end exist_n_exactly_3_rainy_days_l491_49189


namespace true_proposition_l491_49192

def proposition_p := ∀ (x : ℤ), x^2 > x
def proposition_q := ∃ (x : ℝ) (hx : x > 0), x + (2 / x) > 4

theorem true_proposition :
  (¬ proposition_p) ∨ proposition_q :=
by
  sorry

end true_proposition_l491_49192


namespace farmer_goats_l491_49104

theorem farmer_goats (G C P : ℕ) (h1 : P = 2 * C) (h2 : C = G + 4) (h3 : G + C + P = 56) : G = 11 :=
by
  sorry

end farmer_goats_l491_49104


namespace find_square_tiles_l491_49121

theorem find_square_tiles (t s p : ℕ) (h1 : t + s + p = 35) (h2 : 3 * t + 4 * s + 5 * p = 140) (hp0 : p = 0) : s = 35 := by
  sorry

end find_square_tiles_l491_49121


namespace lead_points_l491_49111

-- Define final scores
def final_score_team : ℕ := 68
def final_score_green : ℕ := 39

-- Prove the lead
theorem lead_points : final_score_team - final_score_green = 29 :=
by
  sorry

end lead_points_l491_49111


namespace aubrey_distance_from_school_l491_49133

-- Define average speed and travel time
def average_speed : ℝ := 22 -- in miles per hour
def travel_time : ℝ := 4 -- in hours

-- Define the distance function
def calc_distance (speed time : ℝ) : ℝ := speed * time

-- State the theorem
theorem aubrey_distance_from_school : calc_distance average_speed travel_time = 88 := 
by
  sorry

end aubrey_distance_from_school_l491_49133


namespace shaded_areas_total_l491_49149

theorem shaded_areas_total (r R : ℝ) (h_divides : ∀ (A : ℝ), ∃ (B : ℝ), B = A / 3)
  (h_center : True) (h_area : π * R^2 = 81 * π) :
  (π * R^2 / 3) + (π * (R / 2)^2 / 3) = 33.75 * π :=
by
  -- The proof here will be added.
  sorry

end shaded_areas_total_l491_49149


namespace estimated_germination_probability_stable_l491_49191

structure ExperimentData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations

def experimentalData : List ExperimentData := [
  ⟨50, 47⟩,
  ⟨100, 89⟩,
  ⟨200, 188⟩,
  ⟨500, 461⟩,
  ⟨1000, 892⟩,
  ⟨2000, 1826⟩,
  ⟨3000, 2733⟩
]

def germinationFrequency (data : ExperimentData) : ℚ :=
  data.m / data.n

def closeTo (x y : ℚ) (ε : ℚ) : Prop :=
  |x - y| < ε

theorem estimated_germination_probability_stable :
  ∃ ε > 0, ∀ data ∈ experimentalData, closeTo (germinationFrequency data) 0.91 ε :=
by
  sorry

end estimated_germination_probability_stable_l491_49191


namespace distance_to_first_sign_l491_49114

-- Definitions based on conditions
def total_distance : ℕ := 1000
def after_second_sign : ℕ := 275
def between_signs : ℕ := 375

-- Problem statement
theorem distance_to_first_sign 
  (D : ℕ := total_distance) 
  (a : ℕ := after_second_sign) 
  (d : ℕ := between_signs) : 
  (D - a - d = 350) :=
by
  sorry

end distance_to_first_sign_l491_49114


namespace different_suits_card_combinations_l491_49168

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l491_49168


namespace cone_volume_and_surface_area_l491_49129

noncomputable def cone_volume (slant_height height : ℝ) : ℝ := 
  1 / 3 * Real.pi * (Real.sqrt (slant_height^2 - height^2))^2 * height

noncomputable def cone_surface_area (slant_height height : ℝ) : ℝ :=
  Real.pi * (Real.sqrt (slant_height^2 - height^2)) * (Real.sqrt (slant_height^2 - height^2) + slant_height)

theorem cone_volume_and_surface_area :
  (cone_volume 15 9 = 432 * Real.pi) ∧ (cone_surface_area 15 9 = 324 * Real.pi) :=
by
  sorry

end cone_volume_and_surface_area_l491_49129


namespace max_area_cross_section_rect_prism_l491_49107

/-- The maximum area of the cross-sectional cut of a rectangular prism 
having its vertical edges parallel to the z-axis, with cross-section 
rectangle of sides 8 and 12, whose bottom side lies in the xy-plane 
centered at the origin (0,0,0), cut by the plane 3x + 5y - 2z = 30 
is approximately 118.34. --/
theorem max_area_cross_section_rect_prism :
  ∃ A : ℝ, abs (A - 118.34) < 0.01 :=
sorry

end max_area_cross_section_rect_prism_l491_49107


namespace linear_function_equality_l491_49103

theorem linear_function_equality (f : ℝ → ℝ) (hf : ∀ x, f (3 * (f x)⁻¹ + 5) = f x)
  (hf1 : f 1 = 5) : f 2 = 3 :=
sorry

end linear_function_equality_l491_49103


namespace raw_materials_amount_true_l491_49165

def machinery_cost : ℝ := 2000
def total_amount : ℝ := 5555.56
def cash (T : ℝ) : ℝ := 0.10 * T
def raw_materials_cost (T : ℝ) : ℝ := T - machinery_cost - cash T

theorem raw_materials_amount_true :
  raw_materials_cost total_amount = 3000 := 
  by
  sorry

end raw_materials_amount_true_l491_49165


namespace ratio_of_areas_l491_49195

theorem ratio_of_areas (C1 C2 : ℝ) (h1 : (60 : ℝ) / 360 * C1 = (48 : ℝ) / 360 * C2) : 
  (C1 / C2) ^ 2 = 16 / 25 := 
by
  sorry

end ratio_of_areas_l491_49195


namespace farmer_plant_beds_l491_49119

theorem farmer_plant_beds :
  ∀ (bean_seedlings pumpkin_seeds radishes seedlings_per_row_pumpkin seedlings_per_row_radish radish_rows_per_bed : ℕ),
    bean_seedlings = 64 →
    seedlings_per_row_pumpkin = 7 →
    pumpkin_seeds = 84 →
    seedlings_per_row_radish = 6 →
    radish_rows_per_bed = 2 →
    (bean_seedlings / 8 + pumpkin_seeds / seedlings_per_row_pumpkin + radishes / seedlings_per_row_radish) / radish_rows_per_bed = 14 :=
by
  -- sorry to skip the proof
  sorry

end farmer_plant_beds_l491_49119


namespace walk_time_is_correct_l491_49151

noncomputable def time_to_walk_one_block := 
  let blocks := 18
  let bike_time_per_block := 20 -- seconds
  let additional_walk_time := 12 * 60 -- 12 minutes in seconds
  let walk_time := blocks * bike_time_per_block + additional_walk_time
  walk_time / blocks

theorem walk_time_is_correct : 
  let W := time_to_walk_one_block
  W = 60 := by
    sorry -- proof goes here

end walk_time_is_correct_l491_49151


namespace Ashis_height_more_than_Babji_height_l491_49172

-- Definitions based on conditions
variables {A B : ℝ}
-- Condition expressing the relationship between Ashis's and Babji's height
def Babji_height (A : ℝ) : ℝ := 0.80 * A

-- The proof problem to show the percentage increase
theorem Ashis_height_more_than_Babji_height :
  B = Babji_height A → (A - B) / B * 100 = 25 :=
sorry

end Ashis_height_more_than_Babji_height_l491_49172


namespace spherical_to_rectangular_coordinates_l491_49183

noncomputable
def convert_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  let ρ1 := 10
  let θ1 := Real.pi / 4
  let φ1 := Real.pi / 6
  let ρ2 := 15
  let θ2 := 5 * Real.pi / 4
  let φ2 := Real.pi / 3
  convert_to_cartesian ρ1 θ1 φ1 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  ∧ convert_to_cartesian ρ2 θ2 φ2 = (-15 * Real.sqrt 6 / 4, -15 * Real.sqrt 6 / 4, 7.5) := 
by
  sorry

end spherical_to_rectangular_coordinates_l491_49183


namespace poem_lines_added_l491_49167

theorem poem_lines_added (x : ℕ) 
  (initial_lines : ℕ)
  (months : ℕ)
  (final_lines : ℕ)
  (h_init : initial_lines = 24)
  (h_months : months = 22)
  (h_final : final_lines = 90)
  (h_equation : initial_lines + months * x = final_lines) :
  x = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end poem_lines_added_l491_49167


namespace soccer_claim_fraction_l491_49181

theorem soccer_claim_fraction 
  (total_students enjoy_soccer do_not_enjoy_soccer claim_do_not_enjoy honesty fraction_3_over_11 : ℕ)
  (h1 : enjoy_soccer = total_students / 2)
  (h2 : do_not_enjoy_soccer = total_students / 2)
  (h3 : claim_do_not_enjoy = enjoy_soccer * 3 / 10)
  (h4 : honesty = do_not_enjoy_soccer * 8 / 10)
  (h5 : fraction_3_over_11 = enjoy_soccer * 3 / (10 * (enjoy_soccer * 3 / 10 + do_not_enjoy_soccer * 2 / 10)))
  : fraction_3_over_11 = 3 / 11 :=
sorry

end soccer_claim_fraction_l491_49181


namespace tan_sum_formula_l491_49117

open Real

theorem tan_sum_formula (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_cos_2α : cos (2 * α) = -3 / 5) :
  tan (π / 4 + 2 * α) = -1 / 7 :=
by
  -- Insert the proof here
  sorry

end tan_sum_formula_l491_49117


namespace talia_mom_age_to_talia_age_ratio_l491_49185

-- Definitions for the problem
def Talia_current_age : ℕ := 13
def Talia_mom_current_age : ℕ := 39
def Talia_father_current_age : ℕ := 36

-- These definitions match the conditions in the math problem
def condition1 : Prop := Talia_current_age + 7 = 20
def condition2 : Prop := Talia_father_current_age + 3 = Talia_mom_current_age
def condition3 : Prop := Talia_father_current_age = 36

-- The ratio calculation
def ratio := Talia_mom_current_age / Talia_current_age

-- The main theorem to prove
theorem talia_mom_age_to_talia_age_ratio :
  condition1 ∧ condition2 ∧ condition3 → ratio = 3 := by
  sorry

end talia_mom_age_to_talia_age_ratio_l491_49185


namespace evaluate_g_at_4_l491_49156

def g (x : ℕ) : ℕ := 5 * x - 2

theorem evaluate_g_at_4 : g 4 = 18 := by
  sorry

end evaluate_g_at_4_l491_49156


namespace village_population_l491_49174

theorem village_population (P : ℝ) (h1 : 0.08 * P = 4554) : P = 6325 :=
by
  sorry

end village_population_l491_49174


namespace find_f_0_abs_l491_49188

noncomputable def f : ℝ → ℝ := sorry -- f is a second-degree polynomial with real coefficients

axiom h1 : ∀ (x : ℝ), x = 1 → |f x| = 9
axiom h2 : ∀ (x : ℝ), x = 2 → |f x| = 9
axiom h3 : ∀ (x : ℝ), x = 3 → |f x| = 9

theorem find_f_0_abs : |f 0| = 9 := sorry

end find_f_0_abs_l491_49188


namespace inequality_1_system_of_inequalities_l491_49140

-- Statement for inequality (1)
theorem inequality_1 (x : ℝ) : 2 - x ≥ (x - 1) / 3 - 1 → x ≤ 2.5 := 
sorry

-- Statement for system of inequalities (2)
theorem system_of_inequalities (x : ℝ) : 
  (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) → false := 
sorry

end inequality_1_system_of_inequalities_l491_49140


namespace interval_where_f_decreasing_minimum_value_of_a_l491_49161

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x
noncomputable def h (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * a * x - 1

theorem interval_where_f_decreasing :
  {x : ℝ | 1 < x} = {x : ℝ | deriv f x < 0} :=
by sorry

theorem minimum_value_of_a (a : ℤ) (ha : ∀ x : ℝ, 0 < x → (a - 1) * x^2 + 2 * a * x - 1 ≥ log x - x^2 + x) :
  a ≥ 1 :=
by sorry

end interval_where_f_decreasing_minimum_value_of_a_l491_49161


namespace length_cd_l491_49163

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end length_cd_l491_49163


namespace solution_l491_49187

-- Define the problem.
def problem (CD : ℝ) (hexagon_side : ℝ) (CY : ℝ) (BY : ℝ) : Prop :=
  CD = 2 ∧ hexagon_side = 2 ∧ CY = 4 * CD ∧ BY = 9 * Real.sqrt 2 → BY = 9 * Real.sqrt 2

theorem solution : problem 2 2 8 (9 * Real.sqrt 2) :=
by
  -- Contextualize the given conditions and directly link to the desired proof.
  intro h
  sorry

end solution_l491_49187


namespace find_S_l491_49112

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S) ^ 2) (h2 : S > 0) :
  S = 333332 := 
sorry

end find_S_l491_49112


namespace simplify_and_evaluate_expression_l491_49134

theorem simplify_and_evaluate_expression : 
  ∀ x : ℝ, x = 1 → ( (x^2 - 5) / (x - 3) - 4 / (x - 3) ) = 4 :=
by
  intros x hx
  simp [hx]
  have eq : (1 * 1 - 5) = -4 := by norm_num -- Verify that the expression simplifies correctly
  sorry -- Skip the actual complex proof steps

end simplify_and_evaluate_expression_l491_49134


namespace contractor_total_received_l491_49199

-- Define the conditions
def days_engaged : ℕ := 30
def daily_earnings : ℝ := 25
def fine_per_absence_day : ℝ := 7.50
def days_absent : ℕ := 4

-- Define the days worked based on conditions
def days_worked : ℕ := days_engaged - days_absent

-- Define the total earnings and total fines
def total_earnings : ℝ := days_worked * daily_earnings
def total_fines : ℝ := days_absent * fine_per_absence_day

-- Define the total amount received
def total_amount_received : ℝ := total_earnings - total_fines

-- State the theorem
theorem contractor_total_received :
  total_amount_received = 620 := 
by
  sorry

end contractor_total_received_l491_49199


namespace find_d_l491_49106

theorem find_d (d x y : ℝ) (H1 : x - 2 * y = 5) (H2 : d * x + y = 6) (H3 : x > 0) (H4 : y > 0) :
  -1 / 2 < d ∧ d < 6 / 5 :=
by
  sorry

end find_d_l491_49106


namespace range_of_m_l491_49160

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 - m * x + 1 > 0 → -2 < m ∧ m < 2)) ∧
  (∃ x : ℝ, x^2 < 9 - m^2) ∧
  (-3 < m ∧ m < 3) →
  ((-3 < m ∧ m ≤ -2) ∨ (2 ≤ m ∧ m < 3)) :=
by sorry

end range_of_m_l491_49160


namespace a_75_eq_24_l491_49169

variable {a : ℕ → ℤ}

-- Conditions for the problem
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_15_eq_8 : a 15 = 8 := sorry

def a_60_eq_20 : a 60 = 20 := sorry

-- The theorem we want to prove
theorem a_75_eq_24 (d : ℤ) (h_seq : is_arithmetic_sequence a d) (h15 : a 15 = 8) (h60 : a 60 = 20) : a 75 = 24 :=
  by
    sorry

end a_75_eq_24_l491_49169


namespace wrongly_copied_value_l491_49184

theorem wrongly_copied_value (mean_initial mean_correct : ℕ) (n : ℕ) 
  (wrong_copied_value : ℕ) (total_sum_initial total_sum_correct : ℕ) : 
  (mean_initial = 150) ∧ (mean_correct = 151) ∧ (n = 30) ∧ 
  (wrong_copied_value = 135) ∧ (total_sum_initial = n * mean_initial) ∧ 
  (total_sum_correct = n * mean_correct) → 
  (total_sum_correct - (total_sum_initial - wrong_copied_value) + wrong_copied_value = 300) :=
by
  intros h
  have h1 : mean_initial = 150 := by sorry
  have h2 : mean_correct = 151 := by sorry
  have h3 : n = 30 := by sorry
  have h4 : wrong_copied_value = 135 := by sorry
  have h5 : total_sum_initial = n * mean_initial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by sorry
  sorry -- This is where the proof would go, but is not required per instructions.

end wrongly_copied_value_l491_49184


namespace a_profit_share_l491_49118

/-- Definitions for the shares of capital -/
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)
def total_profit : ℚ := 2415

/-- The profit share for A, given the conditions on capital subscriptions -/
theorem a_profit_share : a_share * total_profit = 805 := by
  sorry

end a_profit_share_l491_49118


namespace whitney_money_left_over_l491_49196

def total_cost (posters_cost : ℝ) (notebooks_cost : ℝ) (bookmarks_cost : ℝ) (pencils_cost : ℝ) (tax_rate : ℝ) :=
  let pre_tax := (3 * posters_cost) + (4 * notebooks_cost) + (5 * bookmarks_cost) + (2 * pencils_cost)
  let tax := pre_tax * tax_rate
  pre_tax + tax

def money_left_over (initial_money : ℝ) (total_cost : ℝ) :=
  initial_money - total_cost

theorem whitney_money_left_over :
  let initial_money := 40
  let posters_cost := 7.50
  let notebooks_cost := 5.25
  let bookmarks_cost := 3.10
  let pencils_cost := 1.15
  let tax_rate := 0.08
  money_left_over initial_money (total_cost posters_cost notebooks_cost bookmarks_cost pencils_cost tax_rate) = -26.20 :=
by
  sorry

end whitney_money_left_over_l491_49196


namespace expected_absolute_deviation_greater_in_10_tosses_l491_49128

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l491_49128


namespace maximize_profits_l491_49176

variable (m : ℝ) (x : ℝ)

def w1 (m x : ℝ) := (8 - m) * x - 30
def w2 (x : ℝ) := -0.01 * x^2 + 8 * x - 80

theorem maximize_profits : 
  (4 ≤ m ∧ m < 5.1 → ∀ x, 0 ≤ x ∧ x ≤ 500 → w1 m x ≥ w2 x) ∧
  (m = 5.1 → ∀ x ≤ 300, w1 m 500 = w2 300) ∧
  (m > 5.1 ∧ m ≤ 6 → ∀ x, 0 ≤ x ∧ x ≤ 300 → w2 x ≥ w1 m x) :=
  sorry

end maximize_profits_l491_49176


namespace wall_length_is_800_l491_49108

def brick_volume : ℝ := 50 * 11.25 * 6
def total_brick_volume : ℝ := 3200 * brick_volume
def wall_volume (x : ℝ) : ℝ := x * 600 * 22.5

theorem wall_length_is_800 :
  ∀ (x : ℝ), total_brick_volume = wall_volume x → x = 800 :=
by
  intros x h
  sorry

end wall_length_is_800_l491_49108


namespace smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l491_49146

theorem smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday :
  ∃ d : ℕ, d = 17 :=
by
  -- Assuming the starting condition that the month starts such that the second Thursday is on the 8th
  let second_thursday := 8

  -- Calculate second Monday after the second Thursday
  let second_monday := second_thursday + 4
  
  -- Calculate first Saturday after the second Monday
  let first_saturday := second_monday + 5

  have smallest_date : first_saturday = 17 := rfl
  
  exact ⟨first_saturday, smallest_date⟩

end smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l491_49146


namespace symmetric_points_origin_l491_49120

theorem symmetric_points_origin (a b : ℝ) 
  (h1 : (-2, b) = (-a, -3)) : a - b = 5 := 
by
  -- solution steps are not included in the statement
  sorry

end symmetric_points_origin_l491_49120


namespace greatest_prime_factor_5pow8_plus_10pow7_l491_49190

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_5pow8_plus_10pow7 : greatest_prime_factor (5^8 + 10^7) = 19 := by
  sorry

end greatest_prime_factor_5pow8_plus_10pow7_l491_49190


namespace minimum_width_for_fence_l491_49122

theorem minimum_width_for_fence (w : ℝ) (h : 0 ≤ 20) : 
  (w * (w + 20) ≥ 150) → w ≥ 10 :=
by
  sorry

end minimum_width_for_fence_l491_49122


namespace radius_of_circle_zero_l491_49100

theorem radius_of_circle_zero (x y : ℝ) :
    (x^2 + 4*x + y^2 - 2*y + 5 = 0) → 0 = 0 :=
by
  sorry

end radius_of_circle_zero_l491_49100


namespace gcd_10010_15015_l491_49153

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l491_49153


namespace bean_seedlings_l491_49175

theorem bean_seedlings
  (beans_per_row : ℕ)
  (pumpkins : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (radishes_per_row : ℕ)
  (rows_per_bed : ℕ) (beds : ℕ)
  (H_beans_per_row : beans_per_row = 8)
  (H_pumpkins : pumpkins = 84)
  (H_pumpkins_per_row : pumpkins_per_row = 7)
  (H_radishes : radishes = 48)
  (H_radishes_per_row : radishes_per_row = 6)
  (H_rows_per_bed : rows_per_bed = 2)
  (H_beds : beds = 14) :
  (beans_per_row * ((beds * rows_per_bed) - (pumpkins / pumpkins_per_row) - (radishes / radishes_per_row)) = 64) :=
by
  sorry

end bean_seedlings_l491_49175


namespace geometric_sequence_expression_l491_49115

variable {a : ℕ → ℝ}

-- Define the geometric sequence property
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_expression :
  is_geometric a q →
  a 3 = 2 →
  a 6 = 16 →
  ∀ n, a n = 2^(n-2) := by
  intros h_geom h_a3 h_a6
  sorry

end geometric_sequence_expression_l491_49115


namespace f_4_1981_eq_l491_49138

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x+1), 0 => f x 1
| (x+1), (y+1) => f x (f (x+1) y)

theorem f_4_1981_eq : f 4 1981 = 2^1984 - 3 := 
by
  sorry

end f_4_1981_eq_l491_49138


namespace coordinate_sum_condition_l491_49193

open Function

theorem coordinate_sum_condition :
  (∃ (g : ℝ → ℝ), g 6 = 5 ∧
    (∃ y : ℝ, 4 * y = g (3 * 2) + 4 ∧ y = 9 / 4 ∧ 2 + y = 17 / 4)) :=
by
  sorry

end coordinate_sum_condition_l491_49193


namespace cube_sum_l491_49197

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l491_49197


namespace marthas_bedroom_size_l491_49126

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l491_49126


namespace avg_length_one_third_wires_l491_49110

theorem avg_length_one_third_wires (x : ℝ) (L1 L2 L3 L4 L5 L6 : ℝ) 
  (h_total_wires : L1 + L2 + L3 + L4 + L5 + L6 = 6 * 80) 
  (h_avg_other_wires : (L3 + L4 + L5 + L6) / 4 = 85) 
  (h_avg_all_wires : (L1 + L2 + L3 + L4 + L5 + L6) / 6 = 80) :
  (L1 + L2) / 2 = 70 :=
by
  sorry

end avg_length_one_third_wires_l491_49110


namespace value_x2012_l491_49109

def f (x : ℝ) : ℝ := sorry

noncomputable def x (n : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom increasing_f : ∀ x y : ℝ, x < y → f x < f y
axiom arithmetic_seq : ∀ n : ℕ, x (n) = x (1) + (n-1) * 2
axiom condition : f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

theorem value_x2012 : x 2012 = 4005 := 
by sorry

end value_x2012_l491_49109


namespace loan_difference_eq_1896_l491_49155

/-- 
  Samantha borrows $12,000 with two repayment schemes:
  1. A twelve-year loan with an annual interest rate of 8% compounded semi-annually. 
     At the end of 6 years, she must make a payment equal to half of what she owes, 
     and the remaining balance accrues interest until the end of 12 years.
  2. A twelve-year loan with a simple annual interest rate of 10%, paid as a lump-sum at the end.

  Prove that the positive difference between the total amounts to be paid back 
  under the two schemes is $1,896, rounded to the nearest dollar.
-/
theorem loan_difference_eq_1896 :
  let P := 12000
  let r1 := 0.08
  let r2 := 0.10
  let n := 2
  let t := 12
  let t1 := 6
  let A1 := P * (1 + r1 / n) ^ (n * t1)
  let payment_after_6_years := A1 / 2
  let remaining_balance := A1 / 2
  let compounded_remaining := remaining_balance * (1 + r1 / n) ^ (n * t1)
  let total_compound := payment_after_6_years + compounded_remaining
  let total_simple := P * (1 + r2 * t)
  (total_simple - total_compound).round = 1896 := 
by
  sorry

end loan_difference_eq_1896_l491_49155


namespace basin_capacity_l491_49186

-- Defining the flow rate of water into the basin
def inflow_rate : ℕ := 24

-- Defining the leak rate of the basin
def leak_rate : ℕ := 4

-- Defining the time taken to fill the basin in seconds
def fill_time : ℕ := 13

-- Net rate of filling the basin
def net_rate : ℕ := inflow_rate - leak_rate

-- Volume of the basin
def basin_volume : ℕ := net_rate * fill_time

-- The goal is to prove that the volume of the basin is 260 gallons
theorem basin_capacity : basin_volume = 260 := by
  sorry

end basin_capacity_l491_49186


namespace greatest_integer_less_than_PS_l491_49164

noncomputable def PS := (150 * Real.sqrt 2)

theorem greatest_integer_less_than_PS
  (PQ RS : ℝ)
  (PS : ℝ := PQ * Real.sqrt 2)
  (h₁ : PQ = 150)
  (h_midpoint : PS / 2 = PQ) :
  ∀ n : ℤ, n < PS → n = 212 :=
by
  -- Proof to be completed later
  sorry

end greatest_integer_less_than_PS_l491_49164


namespace rose_clothing_tax_l491_49143

theorem rose_clothing_tax {total_spent total_tax tax_other tax_clothing amount_clothing amount_food amount_other clothing_tax_rate : ℝ} 
  (h_total_spent : total_spent = 100)
  (h_amount_clothing : amount_clothing = 0.5 * total_spent)
  (h_amount_food : amount_food = 0.2 * total_spent)
  (h_amount_other : amount_other = 0.3 * total_spent)
  (h_no_tax_food : True)
  (h_tax_other_rate : tax_other = 0.08 * amount_other)
  (h_total_tax_rate : total_tax = 0.044 * total_spent)
  (h_calculate_tax_clothing : tax_clothing = total_tax - tax_other) :
  clothing_tax_rate = (tax_clothing / amount_clothing) * 100 → 
  clothing_tax_rate = 4 := 
by
  sorry

end rose_clothing_tax_l491_49143


namespace correct_calculation_of_mistake_l491_49139

theorem correct_calculation_of_mistake (x : ℝ) (h : x - 48 = 52) : x + 48 = 148 :=
by
  sorry

end correct_calculation_of_mistake_l491_49139


namespace total_pages_correct_l491_49170

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l491_49170


namespace arithmetic_sequence_problem_l491_49130

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h1 : (a 1 - 3) ^ 3 + 3 * (a 1 - 3) = -3)
  (h12 : (a 12 - 3) ^ 3 + 3 * (a 12 - 3) = 3) :
  a 1 < a 12 ∧ (12 * (a 1 + a 12)) / 2 = 36 :=
by
  sorry

end arithmetic_sequence_problem_l491_49130


namespace dice_faces_l491_49123

theorem dice_faces (n : ℕ) (h : (1 / (n : ℝ)) ^ 5 = 0.0007716049382716049) : n = 10 := sorry

end dice_faces_l491_49123


namespace find_n_value_l491_49113

theorem find_n_value : (15 * 25 + 20 * 5) = (10 * 25 + 45 * 5) := 
  sorry

end find_n_value_l491_49113


namespace min_distance_ellipse_line_l491_49179

theorem min_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x ^ 2) / 16 + (y ^ 2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
             (∀ (x y : ℝ), ellipse x y → ∃ (d' : ℝ), line x y → d' ≥ d) :=
  sorry

end min_distance_ellipse_line_l491_49179


namespace scientific_notation_320000_l491_49141

theorem scientific_notation_320000 : 320000 = 3.2 * 10^5 :=
  by sorry

end scientific_notation_320000_l491_49141


namespace students_participated_l491_49127

theorem students_participated (like_dislike_sum : 383 + 431 = 814) : 
  383 + 431 = 814 := 
by exact like_dislike_sum

end students_participated_l491_49127


namespace part1_solution_set_a_eq_2_part2_range_of_a_l491_49166

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + abs (2 * x - 2)

theorem part1_solution_set_a_eq_2 :
  { x : ℝ | f 2 x > 2 } = { x | x < (2 / 3) } ∪ { x | x > 2 } :=
by
  sorry

theorem part2_range_of_a :
  { a : ℝ | ∀ x : ℝ, f a x ≥ 2 } = { a | a ≤ -1 } ∪ { a | a ≥ 3 } :=
by
  sorry

end part1_solution_set_a_eq_2_part2_range_of_a_l491_49166


namespace value_of_fraction_l491_49101

variable (m n : ℚ)

theorem value_of_fraction (h₁ : 3 * m + 2 * n = 0) (h₂ : m ≠ 0 ∧ n ≠ 0) :
  (m / n - n / m) = 5 / 6 := 
sorry

end value_of_fraction_l491_49101


namespace fill_cistern_time_l491_49150

-- Definitions based on conditions
def rate_A : ℚ := 1 / 8
def rate_B : ℚ := 1 / 16
def rate_C : ℚ := -1 / 12

-- Combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Time to fill the cistern
def time_to_fill := 1 / combined_rate

-- Lean statement of the proof
theorem fill_cistern_time : time_to_fill = 9.6 := by
  sorry

end fill_cistern_time_l491_49150


namespace largest_sampled_number_l491_49142

theorem largest_sampled_number (N : ℕ) (a₁ a₂ : ℕ) (k : ℕ) (H_N : N = 1500)
  (H_a₁ : a₁ = 18) (H_a₂ : a₂ = 68) (H_k : k = a₂ - a₁) :
  ∃ m, m ≤ N ∧ (m % k = 18 % k) ∧ ∀ n, (n % k = 18 % k) → n ≤ N → n ≤ m :=
by {
  -- sorry
  sorry
}

end largest_sampled_number_l491_49142


namespace planes_perpendicular_l491_49132

variables {m n : Type} -- lines
variables {α β : Type} -- planes

axiom lines_different : m ≠ n
axiom planes_different : α ≠ β
axiom parallel_lines : ∀ (m n : Type), Prop -- m ∥ n
axiom parallel_plane_line : ∀ (m α : Type), Prop -- m ∥ α
axiom perp_plane_line : ∀ (n β : Type), Prop -- n ⊥ β
axiom perp_planes : ∀ (α β : Type), Prop -- α ⊥ β

theorem planes_perpendicular 
  (h1 : parallel_lines m n) 
  (h2 : parallel_plane_line m α) 
  (h3 : perp_plane_line n β) 
: perp_planes α β := 
sorry

end planes_perpendicular_l491_49132


namespace geometric_sequence_a3_l491_49178

noncomputable def a_1 (S_4 : ℕ) (q : ℕ) : ℕ :=
  S_4 * (q - 1) / (1 - q^4)

noncomputable def a_3 (a_1 : ℕ) (q : ℕ) : ℕ :=
  a_1 * q^(3 - 1)

theorem geometric_sequence_a3 (a_n : ℕ → ℕ) (S_4 : ℕ) (q : ℕ) :
  (q = 2) →
  (S_4 = 60) →
  a_3 (a_1 S_4 q) q = 16 :=
by
  intro hq hS4
  rw [hq, hS4]
  sorry

end geometric_sequence_a3_l491_49178


namespace compressor_stations_l491_49116

/-- 
Problem: Given three compressor stations connected by straight roads and not on the same line,
with distances satisfying:
1. x + y = 4z
2. x + z + y = x + a
3. z + y + x = 85

Prove:
- The range of values for 'a' such that the described configuration of compressor stations is 
  possible is 60.71 < a < 68.
- The distances between the compressor stations for a = 5 are x = 70, y = 0, z = 15.
--/
theorem compressor_stations (x y z a : ℝ) 
  (h1 : x + y = 4 * z)
  (h2 : x + z + y = x + a)
  (h3 : z + y + x = 85) :
  (60.71 < a ∧ a < 68) ∧ (a = 5 → x = 70 ∧ y = 0 ∧ z = 15) :=
  sorry

end compressor_stations_l491_49116


namespace ratio_of_sum_of_terms_l491_49105

variable {α : Type*}
variable [Field α]

def geometric_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) := S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem ratio_of_sum_of_terms (a : ℕ → α) (S : ℕ → α)
  (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 :=
by
  sorry

end ratio_of_sum_of_terms_l491_49105


namespace gerry_bananas_eaten_l491_49182

theorem gerry_bananas_eaten (b : ℝ) : 
  (b + (b + 8) + (b + 16) + 0 + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 220) →
  b + 48 = 56.67 :=
by
  sorry

end gerry_bananas_eaten_l491_49182


namespace log_function_domain_l491_49148

theorem log_function_domain {x : ℝ} (h : 1 / x - 1 > 0) : 0 < x ∧ x < 1 :=
sorry

end log_function_domain_l491_49148


namespace scientific_notation_of_1_656_million_l491_49159

theorem scientific_notation_of_1_656_million :
  (1.656 * 10^6 = 1656000) := by
sorry

end scientific_notation_of_1_656_million_l491_49159


namespace mike_total_spent_on_toys_l491_49102

theorem mike_total_spent_on_toys :
  let marbles := 9.05
  let football := 4.95
  let baseball := 6.52
  marbles + football + baseball = 20.52 :=
by
  sorry

end mike_total_spent_on_toys_l491_49102


namespace factorization_correct_l491_49194

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l491_49194


namespace problem_1_problem_2_l491_49152

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -4 ≤ x ∧ x < 2 }
def B : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def P : Set ℝ := { x | x ≤ 0 ∨ x ≥ 5 / 2 }

theorem problem_1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem problem_2 : (U \ B) ∪ P = { x | x ≤ 0 ∨ x ≥ 5 / 2 } :=
sorry

end problem_1_problem_2_l491_49152


namespace range_of_alpha_minus_beta_l491_49173

open Real

theorem range_of_alpha_minus_beta (
    α β : ℝ) 
    (h1 : -π / 2 < α) 
    (h2 : α < 0)
    (h3 : 0 < β)
    (h4 : β < π / 3)
  : -5 * π / 6 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l491_49173


namespace common_chord_properties_l491_49124

noncomputable def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 1 = 0

noncomputable def length_common_chord : ℝ := 2 * Real.sqrt 5

theorem common_chord_properties :
  (∀ x y : ℝ, 
    x^2 + y^2 + 2 * x + 8 * y - 8 = 0 ∧
    x^2 + y^2 - 4 * x - 4 * y - 2 = 0 →
    line_equation x y) ∧ 
  length_common_chord = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_properties_l491_49124


namespace garden_length_l491_49147

def PerimeterLength (P : ℕ) (length : ℕ) (breadth : ℕ) : Prop :=
  P = 2 * (length + breadth)

theorem garden_length
  (P : ℕ)
  (breadth : ℕ)
  (h1 : P = 480)
  (h2 : breadth = 100):
  ∃ length : ℕ, PerimeterLength P length breadth ∧ length = 140 :=
by
  use 140
  sorry

end garden_length_l491_49147


namespace least_possible_value_l491_49135

theorem least_possible_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 4 * x = 5 * y ∧ 5 * y = 6 * z) : x + y + z = 37 :=
by
  sorry

end least_possible_value_l491_49135


namespace serena_mother_age_l491_49144

theorem serena_mother_age {x : ℕ} (h : 39 + x = 3 * (9 + x)) : x = 6 := 
by
  sorry

end serena_mother_age_l491_49144


namespace inverse_110_mod_667_l491_49180

theorem inverse_110_mod_667 :
  (∃ (a b c : ℕ), a = 65 ∧ b = 156 ∧ c = 169 ∧ c^2 = a^2 + b^2) →
  (∃ n : ℕ, 110 * n % 667 = 1 ∧ 0 ≤ n ∧ n < 667 ∧ n = 608) :=
by
  sorry

end inverse_110_mod_667_l491_49180


namespace phase_shift_sin_l491_49162

theorem phase_shift_sin (x : ℝ) : 
  let B := 4
  let C := - (π / 2)
  let φ := - C / B
  φ = π / 8 := 
by 
  sorry

end phase_shift_sin_l491_49162


namespace evaluate_operation_l491_49137

def operation (x : ℝ) : ℝ := 9 - x

theorem evaluate_operation : operation (operation 15) = 15 :=
by
  -- Proof would go here
  sorry

end evaluate_operation_l491_49137
