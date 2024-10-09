import Mathlib

namespace cost_of_producing_one_component_l1141_114197

-- Define the conditions as constants
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_cost : ℕ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℕ := 195

-- Define the cost of producing one component as a variable
variable (C : ℕ)

/-- Prove that C must be less than or equal to 80 given the conditions -/
theorem cost_of_producing_one_component : 
  150 * C + 150 * shipping_cost_per_unit + fixed_monthly_cost ≤ 150 * selling_price_per_component → C ≤ 80 :=
by
  sorry

end cost_of_producing_one_component_l1141_114197


namespace number_of_games_is_15_l1141_114130

-- Definition of the given conditions
def total_points : ℕ := 345
def avg_points_per_game : ℕ := 4 + 10 + 9
def number_of_games (total_points : ℕ) (avg_points_per_game : ℕ) := total_points / avg_points_per_game

-- The theorem stating the proof problem
theorem number_of_games_is_15 : number_of_games total_points avg_points_per_game = 15 :=
by
  -- Skipping the proof as only the statement is required
  sorry

end number_of_games_is_15_l1141_114130


namespace determine_a_l1141_114132

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs (x - 1)) + 1

theorem determine_a (a : ℝ) (h : f a = 2) : a = 1 :=
by
  sorry

end determine_a_l1141_114132


namespace div_by_19_l1141_114147

theorem div_by_19 (n : ℕ) (h : n > 0) : (3^(3*n+2) + 5 * 2^(3*n+1)) % 19 = 0 := by
  sorry

end div_by_19_l1141_114147


namespace candy_problem_l1141_114193

theorem candy_problem (n : ℕ) (h : n ∈ [2, 5, 9, 11, 14]) : ¬(23 - n) % 3 ≠ 0 → n = 9 := by
  sorry

end candy_problem_l1141_114193


namespace sqrt_a_plus_sqrt_b_eq_3_l1141_114159

theorem sqrt_a_plus_sqrt_b_eq_3 (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) : Real.sqrt a + Real.sqrt b = 3 :=
sorry

end sqrt_a_plus_sqrt_b_eq_3_l1141_114159


namespace point_P_location_l1141_114138

theorem point_P_location (a b : ℝ) : (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → a^2 + b^2 > 1 :=
by sorry

end point_P_location_l1141_114138


namespace find_a_l1141_114149

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) / Real.log a

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) : a = 2 :=
sorry

end find_a_l1141_114149


namespace problem_solution_l1141_114104

noncomputable def p (x : ℝ) : ℝ := 
  (x - (Real.sin 1)^2) * (x - (Real.sin 3)^2) * (x - (Real.sin 9)^2)

theorem problem_solution : ∃ a b n : ℕ, 
  p (1 / 4) = Real.sin (a * Real.pi / 180) / (n * Real.sin (b * Real.pi / 180)) ∧
  a > 0 ∧ b > 0 ∧ a ≤ 90 ∧ b ≤ 90 ∧ a + b + n = 216 :=
sorry

end problem_solution_l1141_114104


namespace number_of_blocks_l1141_114181

theorem number_of_blocks (children_per_block : ℕ) (total_children : ℕ) (h1: children_per_block = 6) (h2: total_children = 54) : (total_children / children_per_block) = 9 :=
by {
  sorry
}

end number_of_blocks_l1141_114181


namespace number_of_restaurants_l1141_114150

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

end number_of_restaurants_l1141_114150


namespace max_value_of_g_l1141_114145

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l1141_114145


namespace carl_wins_in_4950_configurations_l1141_114174

noncomputable def num_distinct_configurations_at_Carl_win : ℕ :=
  sorry
  
theorem carl_wins_in_4950_configurations :
  num_distinct_configurations_at_Carl_win = 4950 :=
sorry

end carl_wins_in_4950_configurations_l1141_114174


namespace clothing_store_gross_profit_l1141_114164

theorem clothing_store_gross_profit :
  ∃ S : ℝ, S = 81 + 0.25 * S ∧
  ∃ new_price : ℝ,
    new_price = S - 0.20 * S ∧
    ∃ profit : ℝ,
      profit = new_price - 81 ∧
      profit = 5.40 :=
by
  sorry

end clothing_store_gross_profit_l1141_114164


namespace root_product_identity_l1141_114166

theorem root_product_identity (a b c : ℝ) (h1 : a * b * c = -8) (h2 : a * b + b * c + c * a = 20) (h3 : a + b + c = 15) :
    (1 + a) * (1 + b) * (1 + c) = 28 :=
by
  sorry

end root_product_identity_l1141_114166


namespace compute_expression_l1141_114187

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l1141_114187


namespace son_l1141_114182

theorem son's_age (S F : ℕ) (h₁ : F = 7 * (S - 8)) (h₂ : F / 4 = 14) : S = 16 :=
by {
  sorry
}

end son_l1141_114182


namespace sum_geometric_series_l1141_114111

theorem sum_geometric_series :
  ∑' n : ℕ+, (3 : ℝ)⁻¹ ^ (n : ℕ) = (1 / 2 : ℝ) := by
  sorry

end sum_geometric_series_l1141_114111


namespace billed_minutes_l1141_114189

noncomputable def John_bill (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - monthly_fee) / cost_per_minute

theorem billed_minutes : ∀ (monthly_fee cost_per_minute total_bill : ℝ), 
  monthly_fee = 5 → 
  cost_per_minute = 0.25 → 
  total_bill = 12.02 → 
  John_bill monthly_fee cost_per_minute total_bill = 28 :=
by
  intros monthly_fee cost_per_minute total_bill hf hm hb
  rw [hf, hm, hb, John_bill]
  norm_num
  sorry

end billed_minutes_l1141_114189


namespace radius_of_isosceles_tangent_circle_l1141_114129

noncomputable def R : ℝ := 2 * Real.sqrt 3

variables (x : ℝ) (AB AC BD AD DC r : ℝ)

def is_isosceles (AB BC : ℝ) : Prop := AB = BC
def is_tangent (r : ℝ) (x : ℝ) : Prop := r = 2.4 * x

theorem radius_of_isosceles_tangent_circle
  (h_isosceles: is_isosceles AB BC)
  (h_area: 1/2 * AC * BD = 25)
  (h_height_ratio: BD / AC = 3 / 8)
  (h_AD_DC: AD = DC)
  (h_AC: AC = 8 * x)
  (h_BD: BD = 3 * x)
  (h_radius: is_tangent r x):
  r = R :=
sorry

end radius_of_isosceles_tangent_circle_l1141_114129


namespace abs_ineq_sol_set_l1141_114123

theorem abs_ineq_sol_set (x : ℝ) : (|x - 2| + |x - 1| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end abs_ineq_sol_set_l1141_114123


namespace average_shifted_samples_l1141_114171

variables (x1 x2 x3 x4 : ℝ)

theorem average_shifted_samples (h : (x1 + x2 + x3 + x4) / 4 = 2) :
  ((x1 + 3) + (x2 + 3) + (x3 + 3) + (x4 + 3)) / 4 = 5 :=
by
  sorry

end average_shifted_samples_l1141_114171


namespace all_ones_l1141_114172

theorem all_ones (k : ℕ) (h₁ : k ≥ 2) (n : ℕ → ℕ) (h₂ : ∀ i, 1 ≤ i → i < k → n (i + 1) ∣ (2 ^ n i - 1))
(h₃ : n 1 ∣ (2 ^ n k - 1)) : (∀ i, 1 ≤ i → i ≤ k → n i = 1) :=
by
  sorry

end all_ones_l1141_114172


namespace differential_of_y_l1141_114146

variable (x : ℝ) (dx : ℝ)

noncomputable def y := x * (Real.sin (Real.log x) - Real.cos (Real.log x))

theorem differential_of_y : (deriv y x * dx) = 2 * Real.sin (Real.log x) * dx := by
  sorry

end differential_of_y_l1141_114146


namespace find_line_eq_l1141_114136

theorem find_line_eq (x y : ℝ) (h : x^2 + y^2 - 4 * x - 5 = 0) 
(mid_x mid_y : ℝ) (mid_point : mid_x = 3 ∧ mid_y = 1) : 
x + y - 4 = 0 := 
sorry

end find_line_eq_l1141_114136


namespace certain_number_is_50_l1141_114115

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50_l1141_114115


namespace find_f2_l1141_114139

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by sorry

end find_f2_l1141_114139


namespace sin_810_cos_neg60_l1141_114116

theorem sin_810_cos_neg60 :
  Real.sin (810 * Real.pi / 180) + Real.cos (-60 * Real.pi / 180) = 3 / 2 :=
by
  sorry

end sin_810_cos_neg60_l1141_114116


namespace total_wheels_correct_l1141_114179

-- Define the initial state of the garage
def initial_bicycles := 20
def initial_cars := 10
def initial_motorcycles := 5
def initial_tricycles := 3
def initial_quads := 2

-- Define the changes in the next hour
def bicycles_leaving := 7
def cars_arriving := 4
def motorcycles_arriving := 3
def motorcycles_leaving := 2

-- Define the damaged vehicles
def damaged_bicycles := 5  -- each missing 1 wheel
def damaged_cars := 2      -- each missing 1 wheel
def damaged_motorcycle := 1 -- missing 2 wheels

-- Define the number of wheels per type of vehicle
def bicycle_wheels := 2
def car_wheels := 4
def motorcycle_wheels := 2
def tricycle_wheels := 3
def quad_wheels := 4

-- Calculate the state of vehicles at the end of the hour
def final_bicycles := initial_bicycles - bicycles_leaving
def final_cars := initial_cars + cars_arriving
def final_motorcycles := initial_motorcycles + motorcycles_arriving - motorcycles_leaving

-- Calculate the total wheels in the garage at the end of the hour
def total_wheels : Nat := 
  (final_bicycles - damaged_bicycles) * bicycle_wheels + damaged_bicycles +
  (final_cars - damaged_cars) * car_wheels + damaged_cars * 3 +
  (final_motorcycles - damaged_motorcycle) * motorcycle_wheels +
  initial_tricycles * tricycle_wheels +
  initial_quads * quad_wheels

-- The goal is to prove that the total number of wheels in the garage is 102 at the end of the hour
theorem total_wheels_correct : total_wheels = 102 := 
  by
    sorry

end total_wheels_correct_l1141_114179


namespace ordered_pairs_count_l1141_114109

theorem ordered_pairs_count : 
  ∃ n : ℕ, n = 6 ∧ ∀ A B : ℕ, (0 < A ∧ 0 < B) → (A * B = 32 ↔ A = 1 ∧ B = 32 ∨ A = 32 ∧ B = 1 ∨ A = 2 ∧ B = 16 ∨ A = 16 ∧ B = 2 ∨ A = 4 ∧ B = 8 ∨ A = 8 ∧ B = 4) := 
sorry

end ordered_pairs_count_l1141_114109


namespace no_integer_solutions_l1141_114158

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100 →
  false :=
by
  sorry

end no_integer_solutions_l1141_114158


namespace a0_a2_a4_sum_l1141_114120

theorem a0_a2_a4_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 5 = a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5) →
  a0 + a2 + a4 = -121 :=
by
  intros h
  sorry

end a0_a2_a4_sum_l1141_114120


namespace trees_left_after_typhoon_l1141_114186

variable (initial_trees : ℕ)
variable (died_trees : ℕ)
variable (remaining_trees : ℕ)

theorem trees_left_after_typhoon :
  initial_trees = 20 →
  died_trees = 16 →
  remaining_trees = initial_trees - died_trees →
  remaining_trees = 4 :=
by
  intros h_initial h_died h_remaining
  rw [h_initial, h_died] at h_remaining
  exact h_remaining

end trees_left_after_typhoon_l1141_114186


namespace lighting_candles_correct_l1141_114117

noncomputable def time_to_light_candles (initial_length : ℝ) : ℝ :=
  let burn_rate_1 := initial_length / 300
  let burn_rate_2 := initial_length / 240
  let t := (5 * 60 + 43) - (5 * 60) -- 11:17 AM is 342.857 minutes before 5 PM
  if ((initial_length - burn_rate_2 * t) = 3 * (initial_length - burn_rate_1 * t)) then 11 + 17 / 60 else 0 -- Check if the condition is met

theorem lighting_candles_correct :
  ∀ (initial_length : ℝ), time_to_light_candles initial_length = 11 + 17 / 60 :=
by
  intros initial_length
  sorry  -- Proof goes here

end lighting_candles_correct_l1141_114117


namespace evaluate_power_l1141_114119

theorem evaluate_power :
  (64 : ℝ) = 2^6 →
  64^(3/4 : ℝ) = 16 * Real.sqrt 2 :=
by
  intro h₁
  rw [h₁]
  sorry

end evaluate_power_l1141_114119


namespace find_angle_between_altitude_and_median_l1141_114118

noncomputable def angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : ℝ :=
  Real.arctan ((a^2 - b^2) / (4 * S))

theorem find_angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : 
  angle_between_altitude_and_median a b S h1 h2 = 
    Real.arctan ((a^2 - b^2) / (4 * S)) := 
  sorry

end find_angle_between_altitude_and_median_l1141_114118


namespace quadratic_roots_sum_cubes_l1141_114102

theorem quadratic_roots_sum_cubes (k : ℚ) (a b : ℚ) 
  (h1 : 4 * a^2 + 5 * a + k = 0) 
  (h2 : 4 * b^2 + 5 * b + k = 0) 
  (h3 : a^3 + b^3 = a + b) :
  k = 9 / 4 :=
by {
  -- Lean code requires the proof, here we use sorry to skip it
  sorry
}

end quadratic_roots_sum_cubes_l1141_114102


namespace man_speed_km_per_hr_l1141_114162

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 82
noncomputable def time_to_pass_man_sec : ℝ := 4.499640028797696

theorem man_speed_km_per_hr :
  ∃ (Vm_km_per_hr : ℝ), Vm_km_per_hr = 6.0084 :=
sorry

end man_speed_km_per_hr_l1141_114162


namespace modulus_of_complex_l1141_114135

noncomputable def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_complex :
  ∀ (i : Complex) (z : Complex), i = Complex.I → z = i * (2 - i) → modulus z = Real.sqrt 5 :=
by
  intros i z hi hz
  -- Proof omitted
  sorry

end modulus_of_complex_l1141_114135


namespace larger_solution_quadratic_l1141_114176

theorem larger_solution_quadratic :
  ∃ x : ℝ, x^2 - 13 * x + 30 = 0 ∧ (∀ y : ℝ, y^2 - 13 * y + 30 = 0 → y ≤ x) ∧ x = 10 := 
by
  sorry

end larger_solution_quadratic_l1141_114176


namespace x_divisible_by_5_l1141_114190

theorem x_divisible_by_5
  (x y : ℕ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_gt_1 : 1 < x)
  (h_eq : 2 * x^2 - 1 = y^15) : x % 5 = 0 :=
sorry

end x_divisible_by_5_l1141_114190


namespace right_triangle_properties_l1141_114157

theorem right_triangle_properties (a b c : ℕ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) :
  ∃ (area perimeter : ℕ), area = 30 ∧ perimeter = 30 ∧ (a < c ∧ b < c) :=
by
  let area := 1 / 2 * a * b
  let perimeter := a + b + c
  have acute_angles : a < c ∧ b < c := by sorry
  exact ⟨area, perimeter, ⟨sorry, sorry, acute_angles⟩⟩

end right_triangle_properties_l1141_114157


namespace villages_population_equal_l1141_114195

def population_x (initial_population rate_decrease : Int) (n : Int) := initial_population - rate_decrease * n
def population_y (initial_population rate_increase : Int) (n : Int) := initial_population + rate_increase * n

theorem villages_population_equal
    (initial_population_x : Int) (rate_decrease_x : Int)
    (initial_population_y : Int) (rate_increase_y : Int)
    (h₁ : initial_population_x = 76000) (h₂ : rate_decrease_x = 1200)
    (h₃ : initial_population_y = 42000) (h₄ : rate_increase_y = 800) :
    ∃ n : Int, population_x initial_population_x rate_decrease_x n = population_y initial_population_y rate_increase_y n ∧ n = 17 :=
by
    sorry

end villages_population_equal_l1141_114195


namespace four_fives_to_hundred_case1_four_fives_to_hundred_case2_l1141_114152

theorem four_fives_to_hundred_case1 : (5 + 5) * (5 + 5) = 100 :=
by sorry

theorem four_fives_to_hundred_case2 : (5 * 5 - 5) * 5 = 100 :=
by sorry

end four_fives_to_hundred_case1_four_fives_to_hundred_case2_l1141_114152


namespace find_dividend_l1141_114199

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 14) (h2 : quotient = 4) (h3 : dividend = quotient * k) : dividend = 56 :=
by
  sorry

end find_dividend_l1141_114199


namespace locomotive_distance_l1141_114163

theorem locomotive_distance 
  (speed_train : ℝ) (speed_sound : ℝ) (time_diff : ℝ)
  (h_train : speed_train = 20) 
  (h_sound : speed_sound = 340) 
  (h_time : time_diff = 4) : 
  ∃ x : ℝ, x = 85 := 
by 
  sorry

end locomotive_distance_l1141_114163


namespace ellipse_equation_l1141_114100

def major_axis_length (a : ℝ) := 2 * a = 8
def eccentricity (c a : ℝ) := c / a = 3 / 4

theorem ellipse_equation (a b c x y : ℝ) (h1 : major_axis_length a)
    (h2 : eccentricity c a) (h3 : b^2 = a^2 - c^2) :
    (x^2 / 16 + y^2 / 7 = 1 ∨ x^2 / 7 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l1141_114100


namespace total_students_in_school_l1141_114106

theorem total_students_in_school (s : ℕ) (below_8 above_8 : ℕ) (students_8 : ℕ)
  (h1 : below_8 = 20 * s / 100) 
  (h2 : above_8 = 2 * students_8 / 3) 
  (h3 : students_8 = 48) 
  (h4 : s = students_8 + above_8 + below_8) : 
  s = 100 := 
by 
  sorry 

end total_students_in_school_l1141_114106


namespace fraction_decomposition_l1141_114140

theorem fraction_decomposition (P Q : ℚ) :
  (∀ x : ℚ, 4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24 = (2 * x ^ 2 - 5 * x + 3) * (2 * x - 3))
  → P / (2 * x ^ 2 - 5 * x + 3) + Q / (2 * x - 3) = (8 * x ^ 2 - 9 * x + 20) / (4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24)
  → P = 4 / 9 ∧ Q = 68 / 9 := by 
  sorry

end fraction_decomposition_l1141_114140


namespace color_films_count_l1141_114192

variables (x y C : ℕ)
variables (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ)))

theorem color_films_count (x y : ℕ) (C : ℕ) (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ))) :
  C = 10 * y :=
sorry

end color_films_count_l1141_114192


namespace intersection_P_Q_l1141_114107

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

theorem intersection_P_Q :
  {x : ℤ | (x : ℝ) ∈ P} ∩ Q = {-1, 0, 1, 2} := 
by
  sorry

end intersection_P_Q_l1141_114107


namespace garden_length_l1141_114194

theorem garden_length (P B : ℕ) (h₁ : P = 600) (h₂ : B = 95) : (∃ L : ℕ, 2 * (L + B) = P ∧ L = 205) :=
by
  sorry

end garden_length_l1141_114194


namespace part1_part2_l1141_114191

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then -3 * x + (1/2)^x - 1 else sorry -- Placeholder: function definition incomplete for x ≤ 0

def odd (f : ℝ → ℝ) :=
∀ x, f (-x) = - f x

def monotonic_decreasing (f : ℝ → ℝ) :=
∀ x y, x < y → f x > f y

axiom f_conditions :
  monotonic_decreasing f ∧
  odd f ∧
  (∀ x, x > 0 → f x = -3 * x + (1/2)^x - 1)

theorem part1 : f (-1) = 3.5 :=
by
  sorry

theorem part2 (t : ℝ) (k : ℝ) :
  (∀ t, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1/3 :=
by
  sorry

end part1_part2_l1141_114191


namespace no_solution_for_inequalities_l1141_114133

theorem no_solution_for_inequalities (m : ℝ) :
  (∀ x : ℝ, x - m ≤ 2 * m + 3 ∧ (x - 1) / 2 ≥ m → false) ↔ m < -2 :=
by
  sorry

end no_solution_for_inequalities_l1141_114133


namespace fraction_of_credit_extended_l1141_114125

noncomputable def C_total : ℝ := 342.857
noncomputable def P_auto : ℝ := 0.35
noncomputable def C_company : ℝ := 40

theorem fraction_of_credit_extended :
  (C_company / (C_total * P_auto)) = (1 / 3) :=
  by
    sorry

end fraction_of_credit_extended_l1141_114125


namespace each_persons_final_share_l1141_114142

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

end each_persons_final_share_l1141_114142


namespace divisor_exists_l1141_114196

theorem divisor_exists (n : ℕ) : (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) →
                                (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) ∧
                                (n = 3) :=
by
  sorry

end divisor_exists_l1141_114196


namespace find_other_number_l1141_114185

def smallest_multiple_of_711 (n : ℕ) : ℕ := Nat.lcm n 711

theorem find_other_number (n : ℕ) : smallest_multiple_of_711 n = 3555 → n = 5 := by
  sorry

end find_other_number_l1141_114185


namespace factor_x12_minus_729_l1141_114110

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end factor_x12_minus_729_l1141_114110


namespace amy_uploaded_photos_l1141_114156

theorem amy_uploaded_photos (albums photos_per_album : ℕ) (h1 : albums = 9) (h2 : photos_per_album = 20) :
  albums * photos_per_album = 180 :=
by {
  sorry
}

end amy_uploaded_photos_l1141_114156


namespace num_students_third_school_l1141_114148

variable (x : ℕ)

def num_students_condition := (2 * (x + 40) + (x + 40) + x = 920)

theorem num_students_third_school (h : num_students_condition x) : x = 200 :=
sorry

end num_students_third_school_l1141_114148


namespace time_to_eat_quarter_l1141_114183

noncomputable def total_nuts : ℕ := sorry

def rate_first_crow (N : ℕ) := N / 40
def rate_second_crow (N : ℕ) := N / 36

theorem time_to_eat_quarter (N : ℕ) (T : ℝ) :
  (rate_first_crow N + rate_second_crow N) * T = (1 / 4 : ℝ) * N → 
  T = (90 / 19 : ℝ) :=
by
  intros h
  sorry

end time_to_eat_quarter_l1141_114183


namespace robyn_packs_l1141_114198

-- Define the problem conditions
def total_packs : ℕ := 76
def lucy_packs : ℕ := 29

-- Define the goal to be proven
theorem robyn_packs : total_packs - lucy_packs = 47 := 
by
  sorry

end robyn_packs_l1141_114198


namespace first_divisibility_second_divisibility_l1141_114112

variable {n : ℕ}
variable (h : n > 0)

theorem first_divisibility :
  17 ∣ (5 * 3^(4*n+1) + 2^(6*n+1)) :=
sorry

theorem second_divisibility :
  32 ∣ (25 * 7^(2*n+1) + 3^(4*n)) :=
sorry

end first_divisibility_second_divisibility_l1141_114112


namespace sum_of_ages_is_22_l1141_114173

noncomputable def Ashley_Age := 8
def Mary_Age (M : ℕ) := 7 * Ashley_Age = 4 * M

theorem sum_of_ages_is_22 (M : ℕ) (h : Mary_Age M):
  Ashley_Age + M = 22 :=
by
  -- skipping proof details
  sorry

end sum_of_ages_is_22_l1141_114173


namespace ratio_of_areas_l1141_114131

variable (A B : ℝ)

-- Conditions
def total_area := A + B = 700
def smaller_part_area := B = 315

-- Problem Statement
theorem ratio_of_areas (h_total : total_area A B) (h_small : smaller_part_area B) :
  (A - B) / ((A + B) / 2) = 1 / 5 := by
sorry

end ratio_of_areas_l1141_114131


namespace exchange_rate_l1141_114169

def jackPounds : ℕ := 42
def jackEuros : ℕ := 11
def jackYen : ℕ := 3000
def poundsPerYen : ℕ := 100
def totalYen : ℕ := 9400

theorem exchange_rate :
  ∃ (x : ℕ), 100 * jackPounds + 100 * jackEuros * x + jackYen = totalYen ∧ x = 2 :=
by
  sorry

end exchange_rate_l1141_114169


namespace combined_average_l1141_114165

-- Given Conditions
def num_results_1 : ℕ := 30
def avg_results_1 : ℝ := 20
def num_results_2 : ℕ := 20
def avg_results_2 : ℝ := 30
def num_results_3 : ℕ := 25
def avg_results_3 : ℝ := 40

-- Helper Definitions
def total_sum_1 : ℝ := num_results_1 * avg_results_1
def total_sum_2 : ℝ := num_results_2 * avg_results_2
def total_sum_3 : ℝ := num_results_3 * avg_results_3
def total_sum_all : ℝ := total_sum_1 + total_sum_2 + total_sum_3
def total_number_results : ℕ := num_results_1 + num_results_2 + num_results_3

-- Problem Statement
theorem combined_average : 
  (total_sum_all / (total_number_results:ℝ)) = 29.33 := 
by 
  sorry

end combined_average_l1141_114165


namespace dog_speed_is_16_kmh_l1141_114134

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

end dog_speed_is_16_kmh_l1141_114134


namespace dryer_cost_l1141_114137

theorem dryer_cost (W D : ℕ) (h1 : W + D = 600) (h2 : W = 3 * D) : D = 150 :=
by
  sorry

end dryer_cost_l1141_114137


namespace first_book_cost_correct_l1141_114126

noncomputable def cost_of_first_book (x : ℝ) : Prop :=
  let total_cost := x + 6.5
  let given_amount := 20
  let change_received := 8
  total_cost = given_amount - change_received → x = 5.5

theorem first_book_cost_correct : cost_of_first_book 5.5 :=
by
  sorry

end first_book_cost_correct_l1141_114126


namespace unique_solution_l1141_114175

theorem unique_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 :=
by {
  sorry
}

end unique_solution_l1141_114175


namespace find_denominator_l1141_114101

-- Define the conditions given in the problem
variables (p q : ℚ)
variable (denominator : ℚ)

-- Assuming the conditions
variables (h1 : p / q = 4 / 5)
variables (h2 : 11 / 7 + (2 * q - p) / denominator = 2)

-- State the theorem we want to prove
theorem find_denominator : denominator = 14 :=
by
  -- The proof will be constructed later
  sorry

end find_denominator_l1141_114101


namespace meadow_total_revenue_correct_l1141_114180

-- Define the given quantities and conditions as Lean definitions
def total_diapers : ℕ := 192000
def price_per_diaper : ℝ := 4.0
def bundle_discount : ℝ := 0.05
def purchase_discount : ℝ := 0.05
def tax_rate : ℝ := 0.10

-- Define a function that calculates the revenue from selling all the diapers
def calculate_revenue (total_diapers : ℕ) (price_per_diaper : ℝ) (bundle_discount : ℝ) 
    (purchase_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let gross_revenue := total_diapers * price_per_diaper
  let bundle_discounted_revenue := gross_revenue * (1 - bundle_discount)
  let purchase_discounted_revenue := bundle_discounted_revenue * (1 - purchase_discount)
  let taxed_revenue := purchase_discounted_revenue * (1 + tax_rate)
  taxed_revenue

-- The main theorem to prove that the calculated revenue matches the expected value
theorem meadow_total_revenue_correct : 
  calculate_revenue total_diapers price_per_diaper bundle_discount purchase_discount tax_rate = 762432 := 
by
  sorry

end meadow_total_revenue_correct_l1141_114180


namespace jack_total_damage_costs_l1141_114128

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l1141_114128


namespace exponential_quotient_l1141_114178

variable {x a b : ℝ}

theorem exponential_quotient (h1 : x^a = 3) (h2 : x^b = 5) : x^(a-b) = 3 / 5 :=
sorry

end exponential_quotient_l1141_114178


namespace circle_through_two_points_on_y_axis_l1141_114170

theorem circle_through_two_points_on_y_axis :
  ∃ (b : ℝ), (∀ (x y : ℝ), (x + 1)^2 + (y - 4)^2 = (x - 3)^2 + (y - 2)^2 → b = 1) ∧ 
  (∀ (x y : ℝ), (x - 0)^2 + (y - b)^2 = 10) := 
sorry

end circle_through_two_points_on_y_axis_l1141_114170


namespace no_contradiction_to_thermodynamics_l1141_114167

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l1141_114167


namespace bacteria_growth_l1141_114177

-- Define the original and current number of bacteria
def original_bacteria := 600
def current_bacteria := 8917

-- Define the increase in bacteria count
def additional_bacteria := 8317

-- Prove the statement
theorem bacteria_growth : current_bacteria - original_bacteria = additional_bacteria :=
by {
    -- Lean will require the proof here, so we use sorry for now 
    sorry
}

end bacteria_growth_l1141_114177


namespace value_of_expression_l1141_114144

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by sorry

end value_of_expression_l1141_114144


namespace jenny_best_neighborhood_earnings_l1141_114168

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end jenny_best_neighborhood_earnings_l1141_114168


namespace total_balloons_l1141_114122

theorem total_balloons (A_initial : Nat) (A_additional : Nat) (J_initial : Nat) 
  (h1 : A_initial = 3) (h2 : J_initial = 5) (h3 : A_additional = 2) : 
  A_initial + A_additional + J_initial = 10 := by
  sorry

end total_balloons_l1141_114122


namespace sum_first_4_terms_l1141_114121

theorem sum_first_4_terms 
  (a_1 : ℚ) 
  (q : ℚ) 
  (h1 : a_1 * q - a_1 * q^2 = -2) 
  (h2 : a_1 + a_1 * q^2 = 10 / 3) 
  : a_1 * (1 + q + q^2 + q^3) = 40 / 3 := sorry

end sum_first_4_terms_l1141_114121


namespace price_per_cake_l1141_114113

def number_of_cakes_per_day := 4
def number_of_working_days_per_week := 5
def total_amount_collected := 640
def number_of_weeks := 4

theorem price_per_cake :
  let total_cakes_per_week := number_of_cakes_per_day * number_of_working_days_per_week
  let total_cakes_in_four_weeks := total_cakes_per_week * number_of_weeks
  let price_per_cake := total_amount_collected / total_cakes_in_four_weeks
  price_per_cake = 8 := by
sorry

end price_per_cake_l1141_114113


namespace greta_hours_worked_l1141_114124

-- Define the problem conditions
def greta_hourly_rate := 12
def lisa_hourly_rate := 15
def lisa_hours_to_equal_greta_earnings := 32
def greta_earnings (hours_worked : ℕ) := greta_hourly_rate * hours_worked
def lisa_earnings := lisa_hourly_rate * lisa_hours_to_equal_greta_earnings

-- Problem statement
theorem greta_hours_worked (G : ℕ) (H : greta_earnings G = lisa_earnings) : G = 40 := by
  sorry

end greta_hours_worked_l1141_114124


namespace highest_more_than_lowest_by_37_5_percent_l1141_114108

variables (highest_price lowest_price : ℝ)

theorem highest_more_than_lowest_by_37_5_percent
  (h_highest : highest_price = 22)
  (h_lowest : lowest_price = 16) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 37.5 :=
by
  sorry

end highest_more_than_lowest_by_37_5_percent_l1141_114108


namespace solution_set_of_inequality_l1141_114114

theorem solution_set_of_inequality :
  { x : ℝ | x ^ 2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
by 
  sorry

end solution_set_of_inequality_l1141_114114


namespace left_handed_ratio_l1141_114127

-- Given the conditions:
-- total number of players
def total_players : ℕ := 70
-- number of throwers who are all right-handed 
def throwers : ℕ := 37 
-- total number of right-handed players
def right_handed : ℕ := 59

-- Define the necessary variables based on the given conditions.
def non_throwers : ℕ := total_players - throwers
def non_throwing_right_handed : ℕ := right_handed - throwers
def left_handed_non_throwers : ℕ := non_throwers - non_throwing_right_handed

-- State the theorem to prove that the ratio of 
-- left-handed non-throwers to the rest of the team (excluding throwers) is 1:3
theorem left_handed_ratio : 
  (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 := by
    sorry

end left_handed_ratio_l1141_114127


namespace efficacy_rate_is_80_percent_l1141_114155

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

end efficacy_rate_is_80_percent_l1141_114155


namespace total_sonnets_written_l1141_114103

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l1141_114103


namespace right_triangular_pyramid_property_l1141_114151

theorem right_triangular_pyramid_property
  (S1 S2 S3 S : ℝ)
  (right_angle_face1_area : S1 = S1) 
  (right_angle_face2_area : S2 = S2) 
  (right_angle_face3_area : S3 = S3) 
  (oblique_face_area : S = S) :
  S1^2 + S2^2 + S3^2 = S^2 := 
sorry

end right_triangular_pyramid_property_l1141_114151


namespace seats_filled_percentage_l1141_114143

theorem seats_filled_percentage (total_seats vacant_seats : ℕ) (h1 : total_seats = 600) (h2 : vacant_seats = 228) :
  ((total_seats - vacant_seats) / total_seats * 100 : ℝ) = 62 := by
  sorry

end seats_filled_percentage_l1141_114143


namespace sum_ad_eq_two_l1141_114184

theorem sum_ad_eq_two (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 2 :=
by
  sorry

end sum_ad_eq_two_l1141_114184


namespace quadratic_equation_completing_square_l1141_114153

theorem quadratic_equation_completing_square :
  (∃ m n : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 45 = 15 * ((x + m)^2 - m^2 - 3) + 45 ∧ (m + n = 3))) :=
sorry

end quadratic_equation_completing_square_l1141_114153


namespace five_wednesdays_implies_five_saturdays_in_august_l1141_114105

theorem five_wednesdays_implies_five_saturdays_in_august (N : ℕ) (H1 : ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ w ∈ ws, w < 32 ∧ (w % 7 = 3)) (H2 : July_days = 31) (H3 : August_days = 31):
  ∀ w : ℕ, w < 7 → ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ sat ∈ ws, (sat % 7 = 6) :=
by
  sorry

end five_wednesdays_implies_five_saturdays_in_august_l1141_114105


namespace simplify_fraction_result_l1141_114188

theorem simplify_fraction_result :
  (144: ℝ) / 1296 * 72 = 8 :=
by
  sorry

end simplify_fraction_result_l1141_114188


namespace num_sides_polygon_l1141_114141

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l1141_114141


namespace overall_weighted_defective_shipped_percentage_l1141_114161

theorem overall_weighted_defective_shipped_percentage
  (defective_A : ℝ := 0.06) (shipped_A : ℝ := 0.04) (prod_A : ℝ := 0.30)
  (defective_B : ℝ := 0.09) (shipped_B : ℝ := 0.06) (prod_B : ℝ := 0.50)
  (defective_C : ℝ := 0.12) (shipped_C : ℝ := 0.07) (prod_C : ℝ := 0.20) :
  prod_A * defective_A * shipped_A + prod_B * defective_B * shipped_B + prod_C * defective_C * shipped_C = 0.00510 :=
by
  sorry

end overall_weighted_defective_shipped_percentage_l1141_114161


namespace clients_select_two_cars_l1141_114160

theorem clients_select_two_cars (cars clients selections : ℕ) (total_selections : ℕ)
  (h1 : cars = 10) (h2 : clients = 15) (h3 : total_selections = cars * 3) (h4 : total_selections = clients * selections) :
  selections = 2 :=
by 
  sorry

end clients_select_two_cars_l1141_114160


namespace find_points_per_enemy_l1141_114154

def points_per_enemy (x : ℕ) : Prop :=
  let points_from_enemies := 6 * x
  let additional_points := 8
  let total_points := points_from_enemies + additional_points
  total_points = 62

theorem find_points_per_enemy (x : ℕ) (h : points_per_enemy x) : x = 9 :=
  by sorry

end find_points_per_enemy_l1141_114154
