import Mathlib

namespace NUMINAMATH_GPT_sawyer_joined_coaching_l367_36721

variable (daily_fees total_fees : ℕ)
variable (year_not_leap : Prop)
variable (discontinue_day : ℕ)

theorem sawyer_joined_coaching :
  daily_fees = 39 → 
  total_fees = 11895 → 
  year_not_leap → 
  discontinue_day = 307 → 
  ∃ start_day, start_day = 30 := 
by
  intros h_daily_fees h_total_fees h_year_not_leap h_discontinue_day
  sorry

end NUMINAMATH_GPT_sawyer_joined_coaching_l367_36721


namespace NUMINAMATH_GPT_one_third_sugar_l367_36785

theorem one_third_sugar (sugar : ℚ) (h : sugar = 3 + 3 / 4) : sugar / 3 = 1 + 1 / 4 :=
by sorry

end NUMINAMATH_GPT_one_third_sugar_l367_36785


namespace NUMINAMATH_GPT_sum_of_xyz_l367_36702

theorem sum_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y + z)^3 - x^3 - y^3 - z^3 = 504) : x + y + z = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_xyz_l367_36702


namespace NUMINAMATH_GPT_Martha_cards_l367_36715

theorem Martha_cards :
  let initial_cards := 76.0
  let given_away_cards := 3.0
  initial_cards - given_away_cards = 73.0 :=
by 
  let initial_cards := 76.0
  let given_away_cards := 3.0
  have h : initial_cards - given_away_cards = 73.0 := by sorry
  exact h

end NUMINAMATH_GPT_Martha_cards_l367_36715


namespace NUMINAMATH_GPT_valid_three_digit_numbers_count_l367_36753

noncomputable def count_valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let excluded_numbers := 81 + 72
  total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 747 :=
by
  sorry

end NUMINAMATH_GPT_valid_three_digit_numbers_count_l367_36753


namespace NUMINAMATH_GPT_speed_of_jogger_l367_36714

noncomputable def jogger_speed_problem (jogger_distance_ahead train_length train_speed_kmh time_to_pass : ℕ) :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := jogger_distance_ahead + train_length
  let relative_speed := total_distance / time_to_pass
  let jogger_speed_ms := train_speed_ms - relative_speed
  let jogger_speed_kmh := jogger_speed_ms * 3600 / 1000
  jogger_speed_kmh

theorem speed_of_jogger :
  jogger_speed_problem 240 210 45 45 = 9 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_jogger_l367_36714


namespace NUMINAMATH_GPT_exists_k_consecutive_squareful_numbers_l367_36745

-- Define what it means for a number to be squareful
def is_squareful (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

-- State the theorem
theorem exists_k_consecutive_squareful_numbers (k : ℕ) : 
  ∃ (a : ℕ), ∀ i, i < k → is_squareful (a + i) :=
sorry

end NUMINAMATH_GPT_exists_k_consecutive_squareful_numbers_l367_36745


namespace NUMINAMATH_GPT_sin_150_eq_half_l367_36748

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_150_eq_half_l367_36748


namespace NUMINAMATH_GPT_ben_david_bagel_cost_l367_36784

theorem ben_david_bagel_cost (B D : ℝ)
  (h1 : D = 0.5 * B)
  (h2 : B = D + 16) :
  B + D = 48 := 
sorry

end NUMINAMATH_GPT_ben_david_bagel_cost_l367_36784


namespace NUMINAMATH_GPT_solve_system_of_equations_l367_36708

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x + 2 * y = 5 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l367_36708


namespace NUMINAMATH_GPT_find_r_l367_36740

noncomputable def g (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

theorem find_r 
  (p q r : ℝ) 
  (h1 : ∀ x : ℝ, g x p q r = (x + 100) * (x + 0) * (x + 0))
  (h2 : p + q + r = 100) : 
  r = 0 := 
by
  sorry

end NUMINAMATH_GPT_find_r_l367_36740


namespace NUMINAMATH_GPT_find_b_l367_36700

open Real

noncomputable def triangle_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) : Prop :=
  B < π / 2 ∧
  sin_B = sqrt 7 / 4 ∧
  area = 5 * sqrt 7 / 4 ∧
  sin_A / sin_B = 5 * c / (2 * b) ∧
  a = 5 / 2 * c ∧
  area = 1 / 2 * a * c * sin_B

theorem find_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) :
  triangle_b a b c A B C sin_A sin_B area → b = sqrt 14 := by
  sorry

end NUMINAMATH_GPT_find_b_l367_36700


namespace NUMINAMATH_GPT_train_crosses_pole_in_1_5_seconds_l367_36791

noncomputable def time_to_cross_pole (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  length / speed_m_s

theorem train_crosses_pole_in_1_5_seconds :
  time_to_cross_pole 60 144 = 1.5 :=
by
  unfold time_to_cross_pole
  -- simplified proof would be here
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_1_5_seconds_l367_36791


namespace NUMINAMATH_GPT_prob_A_exactly_once_l367_36778

theorem prob_A_exactly_once (P : ℚ) (h : 1 - (1 - P)^3 = 63 / 64) : 
  (3 * P * (1 - P)^2 = 9 / 64) :=
by
  sorry

end NUMINAMATH_GPT_prob_A_exactly_once_l367_36778


namespace NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l367_36741

theorem find_n_in_arithmetic_sequence 
  (a : ℕ → ℕ)
  (a_1 : ℕ)
  (d : ℕ) 
  (a_n : ℕ) 
  (n : ℕ)
  (h₀ : a_1 = 11)
  (h₁ : d = 2)
  (h₂ : a n = a_1 + (n - 1) * d)
  (h₃ : a n = 2009) :
  n = 1000 := 
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l367_36741


namespace NUMINAMATH_GPT_max_parts_three_planes_divide_space_l367_36710

-- Define the conditions given in the problem.
-- Condition 1: A plane divides the space into two parts.
def plane_divides_space (n : ℕ) : ℕ := 2

-- Condition 2: Two planes can divide the space into either three or four parts.
def two_planes_divide_space (n : ℕ) : ℕ := if n = 2 then 3 else 4

-- Condition 3: Three planes can divide the space into four, six, seven, or eight parts.
def three_planes_divide_space (n : ℕ) : ℕ := if n = 4 then 8 else sorry

-- The statement to be proved.
theorem max_parts_three_planes_divide_space : 
  ∃ n, three_planes_divide_space n = 8 := by
  use 4
  sorry

end NUMINAMATH_GPT_max_parts_three_planes_divide_space_l367_36710


namespace NUMINAMATH_GPT_actual_distance_traveled_l367_36762

theorem actual_distance_traveled
  (D : ℝ) 
  (H : ∃ T : ℝ, D = 5 * T ∧ D + 20 = 15 * T) : 
  D = 10 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l367_36762


namespace NUMINAMATH_GPT_parametric_eqn_and_max_sum_l367_36724

noncomputable def polar_eq (ρ θ : ℝ) := ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

theorem parametric_eqn_and_max_sum (θ : ℝ):
  (∃ (x y : ℝ), (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ) = (x, y)) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4 → (3, 3) = (3, 3) ∧ 6 = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_parametric_eqn_and_max_sum_l367_36724


namespace NUMINAMATH_GPT_solve_abs_eq_l367_36713

theorem solve_abs_eq : ∀ x : ℚ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) := by
  intros x
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l367_36713


namespace NUMINAMATH_GPT_matchstick_triangles_l367_36770

/-- Using 12 equal-length matchsticks, it is possible to form an isosceles triangle, an equilateral triangle, and a right-angled triangle without breaking or overlapping the matchsticks. --/
theorem matchstick_triangles :
  ∃ a b c : ℕ, a + b + c = 12 ∧ (a = b ∨ b = c ∨ a = c) ∧ (a * a + b * b = c * c ∨ a = b ∧ b = c) :=
by
  sorry

end NUMINAMATH_GPT_matchstick_triangles_l367_36770


namespace NUMINAMATH_GPT_avg_rate_of_change_l367_36793

def f (x : ℝ) := 2 * x + 1

theorem avg_rate_of_change : (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end NUMINAMATH_GPT_avg_rate_of_change_l367_36793


namespace NUMINAMATH_GPT_min_value_geq_9div2_l367_36716

noncomputable def min_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) : ℝ := 
  (x + y + z : ℝ) * ((1 : ℝ) / (x + y) + (1 : ℝ) / (x + z) + (1 : ℝ) / (y + z))

theorem min_value_geq_9div2 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) :
  min_value x y z hx hy hz h_sum ≥ 9 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_geq_9div2_l367_36716


namespace NUMINAMATH_GPT_complement_of_A_l367_36798

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (C_UA : Set ℕ) :
  U = {2, 3, 4} →
  A = {x | (x - 1) * (x - 4) < 0 ∧ x ∈ Set.univ} →
  C_UA = {x ∈ U | x ∉ A} →
  C_UA = {4} :=
by
  intros hU hA hCUA
  -- proof omitted, sorry placeholder
  sorry

end NUMINAMATH_GPT_complement_of_A_l367_36798


namespace NUMINAMATH_GPT_weight_of_NH4I_H2O_l367_36751

noncomputable def total_weight (moles_NH4I : ℕ) (molar_mass_NH4I : ℝ) 
                             (moles_H2O : ℕ) (molar_mass_H2O : ℝ) : ℝ :=
  (moles_NH4I * molar_mass_NH4I) + (moles_H2O * molar_mass_H2O)

theorem weight_of_NH4I_H2O :
  total_weight 15 144.95 7 18.02 = 2300.39 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_NH4I_H2O_l367_36751


namespace NUMINAMATH_GPT_josh_money_remaining_l367_36794

theorem josh_money_remaining :
  let initial := 50.00
  let shirt := 7.85
  let meal := 15.49
  let magazine := 6.13
  let friends_debt := 3.27
  let cd := 11.75
  initial - shirt - meal - magazine - friends_debt - cd = 5.51 :=
by
  sorry

end NUMINAMATH_GPT_josh_money_remaining_l367_36794


namespace NUMINAMATH_GPT_no_real_a_l367_36792

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}

theorem no_real_a (a : ℝ) : ¬ ((A a ≠ B) ∧ (A a ∪ B = B) ∧ (∅ ⊂ (A a ∩ B))) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_real_a_l367_36792


namespace NUMINAMATH_GPT_train_crossing_tree_time_l367_36761

noncomputable def time_to_cross_platform (train_length : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) : ℕ :=
  (train_length + platform_length) / time_to_cross_platform

noncomputable def time_to_cross_tree (train_length : ℕ) (speed : ℕ) : ℕ :=
  train_length / speed

theorem train_crossing_tree_time :
  ∀ (train_length platform_length time platform_time speed : ℕ),
  train_length = 1200 →
  platform_length = 900 →
  platform_time = 210 →
  speed = (train_length + platform_length) / platform_time →
  time = train_length / speed →
  time = 120 :=
by
  intros train_length platform_length time platform_time speed h_train_length h_platform_length h_platform_time h_speed h_time
  sorry

end NUMINAMATH_GPT_train_crossing_tree_time_l367_36761


namespace NUMINAMATH_GPT_nine_distinct_numbers_product_l367_36719

variable (a b c d e f g h i : ℕ)

theorem nine_distinct_numbers_product (ha : a = 12) (hb : b = 9) (hc : c = 2)
                                      (hd : d = 1) (he : e = 6) (hf : f = 36)
                                      (hg : g = 18) (hh : h = 4) (hi : i = 3) :
  (a * b * c = 216) ∧ (d * e * f = 216) ∧ (g * h * i = 216) ∧
  (a * d * g = 216) ∧ (b * e * h = 216) ∧ (c * f * i = 216) ∧
  (a * e * i = 216) ∧ (c * e * g = 216) :=
by
  sorry

end NUMINAMATH_GPT_nine_distinct_numbers_product_l367_36719


namespace NUMINAMATH_GPT_loss_per_meter_is_5_l367_36756

-- Define the conditions
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 50
def quantity : ℕ := 400

-- Define the statement to prove (question == answer given conditions)
theorem loss_per_meter_is_5 : 
  ((cost_price_per_meter * quantity - selling_price) / quantity) = 5 := 
by
  sorry

end NUMINAMATH_GPT_loss_per_meter_is_5_l367_36756


namespace NUMINAMATH_GPT_train_speed_approx_l367_36742

noncomputable def man_speed_kmh : ℝ := 3
noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600
noncomputable def train_length : ℝ := 900
noncomputable def time_to_cross : ℝ := 53.99568034557235
noncomputable def train_speed_ms := (train_length / time_to_cross) + man_speed_ms
noncomputable def train_speed_kmh := (train_speed_ms * 3600) / 1000

theorem train_speed_approx :
  abs (train_speed_kmh - 63.009972) < 1e-5 := sorry

end NUMINAMATH_GPT_train_speed_approx_l367_36742


namespace NUMINAMATH_GPT_length_of_train_l367_36796

theorem length_of_train (v : ℝ) (t : ℝ) (L : ℝ) 
  (h₁ : v = 36) 
  (h₂ : t = 1) 
  (h_eq_lengths : true) -- assuming the equality of lengths tacitly without naming
  : L = 300 := 
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_length_of_train_l367_36796


namespace NUMINAMATH_GPT_rectangle_inscribed_circle_circumference_l367_36769

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end NUMINAMATH_GPT_rectangle_inscribed_circle_circumference_l367_36769


namespace NUMINAMATH_GPT_number_of_animals_per_aquarium_l367_36780

variable (aq : ℕ) (ani : ℕ) (a : ℕ)

axiom condition1 : aq = 26
axiom condition2 : ani = 52
axiom condition3 : ani = aq * a

theorem number_of_animals_per_aquarium : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_animals_per_aquarium_l367_36780


namespace NUMINAMATH_GPT_copper_needed_l367_36752

theorem copper_needed (T : ℝ) (lead_percentage : ℝ) (lead_weight : ℝ) (copper_percentage : ℝ) 
  (h_lead_percentage : lead_percentage = 0.25)
  (h_lead_weight : lead_weight = 5)
  (h_copper_percentage : copper_percentage = 0.60)
  (h_total_weight : T = lead_weight / lead_percentage) :
  copper_percentage * T = 12 := 
by
  sorry

end NUMINAMATH_GPT_copper_needed_l367_36752


namespace NUMINAMATH_GPT_penguins_seals_ratio_l367_36779

theorem penguins_seals_ratio (t_total t_seals t_elephants t_penguins : ℕ) 
    (h1 : t_total = 130) 
    (h2 : t_seals = 13) 
    (h3 : t_elephants = 13) 
    (h4 : t_penguins = t_total - t_seals - t_elephants) : 
    (t_penguins / t_seals = 8) := by
  sorry

end NUMINAMATH_GPT_penguins_seals_ratio_l367_36779


namespace NUMINAMATH_GPT_fixed_points_l367_36718

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end NUMINAMATH_GPT_fixed_points_l367_36718


namespace NUMINAMATH_GPT_cost_of_pumpkin_seeds_l367_36703

theorem cost_of_pumpkin_seeds (P : ℝ)
    (h1 : ∃(P_tomato P_chili : ℝ), P_tomato = 1.5 ∧ P_chili = 0.9) 
    (h2 : 3 * P + 4 * 1.5 + 5 * 0.9 = 18) 
    : P = 2.5 :=
by sorry

end NUMINAMATH_GPT_cost_of_pumpkin_seeds_l367_36703


namespace NUMINAMATH_GPT_necessary_condition_l367_36704

theorem necessary_condition (a b : ℝ) (h : b ≠ 0) (h2 : a > b) (h3 : b > 0) : (1 / a < 1 / b) :=
sorry

end NUMINAMATH_GPT_necessary_condition_l367_36704


namespace NUMINAMATH_GPT_bobby_weekly_salary_l367_36790

variable (S : ℝ)
variables (federal_tax : ℝ) (state_tax : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) (city_fee : ℝ) (net_paycheck : ℝ)

def bobby_salary_equation := 
  S - (federal_tax * S) - (state_tax * S) - health_insurance - life_insurance - city_fee = net_paycheck

theorem bobby_weekly_salary 
  (S : ℝ) 
  (federal_tax : ℝ := 1/3) 
  (state_tax : ℝ := 0.08) 
  (health_insurance : ℝ := 50) 
  (life_insurance : ℝ := 20) 
  (city_fee : ℝ := 10) 
  (net_paycheck : ℝ := 184) 
  (valid_solution : bobby_salary_equation S (1/3) 0.08 50 20 10 184) : 
  S = 450.03 := 
  sorry

end NUMINAMATH_GPT_bobby_weekly_salary_l367_36790


namespace NUMINAMATH_GPT_line_perpendicular_exists_k_line_intersects_circle_l367_36726

theorem line_perpendicular_exists_k (k : ℝ) :
  ∃ k, (k * (1 / 2)) = -1 :=
sorry

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (k * x - y + 2 * k = 0) ∧ (x^2 + y^2 = 8) :=
sorry

end NUMINAMATH_GPT_line_perpendicular_exists_k_line_intersects_circle_l367_36726


namespace NUMINAMATH_GPT_tv_cost_l367_36743

theorem tv_cost (savings : ℕ) (fraction_spent_on_furniture : ℚ) (amount_spent_on_furniture : ℚ) (remaining_savings : ℚ) :
  savings = 1000 →
  fraction_spent_on_furniture = 3/5 →
  amount_spent_on_furniture = fraction_spent_on_furniture * savings →
  remaining_savings = savings - amount_spent_on_furniture →
  remaining_savings = 400 :=
by
  sorry

end NUMINAMATH_GPT_tv_cost_l367_36743


namespace NUMINAMATH_GPT_x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l367_36728

variable {x : ℝ}

theorem x_cubed_lt_one_of_x_lt_one (hx : x < 1) : x^3 < 1 :=
sorry

theorem abs_x_lt_one_of_x_lt_one (hx : x < 1) : |x| < 1 :=
sorry

end NUMINAMATH_GPT_x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l367_36728


namespace NUMINAMATH_GPT_jar_ratios_l367_36755

theorem jar_ratios (C_X C_Y : ℝ) 
  (h1 : 0 < C_X) 
  (h2 : 0 < C_Y)
  (h3 : (1/2) * C_X + (1/2) * C_Y = (3/4) * C_X) : 
  C_Y = (1/2) * C_X := 
sorry

end NUMINAMATH_GPT_jar_ratios_l367_36755


namespace NUMINAMATH_GPT_compute_65_sq_minus_55_sq_l367_36765

theorem compute_65_sq_minus_55_sq : 65^2 - 55^2 = 1200 :=
by
  -- We'll skip the proof here for simplicity
  sorry

end NUMINAMATH_GPT_compute_65_sq_minus_55_sq_l367_36765


namespace NUMINAMATH_GPT_triangle_angle_and_perimeter_l367_36747

/-
In a triangle ABC, given c * sin B = sqrt 3 * cos C,
prove that angle C equals pi / 3,
and given a + b = 6, find the minimum perimeter of triangle ABC.
-/
theorem triangle_angle_and_perimeter (A B C : ℝ) (a b c : ℝ) 
  (h1 : c * Real.sin B = Real.sqrt 3 * Real.cos C)
  (h2 : a + b = 6) :
  C = Real.pi / 3 ∧ a + b + (Real.sqrt (36 - a * b)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_and_perimeter_l367_36747


namespace NUMINAMATH_GPT_natasha_avg_speed_climbing_l367_36733

-- Natasha climbs up a hill in 4 hours and descends in 2 hours.
-- Her average speed along the whole journey is 1.5 km/h.
-- Prove that her average speed while climbing to the top is 1.125 km/h.

theorem natasha_avg_speed_climbing (v_up v_down : ℝ) :
  (4 * v_up = 2 * v_down) ∧ (1.5 = (2 * (4 * v_up) / 6)) → v_up = 1.125 :=
by
  -- We provide no proof here; this is just the statement.
  sorry

end NUMINAMATH_GPT_natasha_avg_speed_climbing_l367_36733


namespace NUMINAMATH_GPT_game_is_unfair_l367_36754

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

end NUMINAMATH_GPT_game_is_unfair_l367_36754


namespace NUMINAMATH_GPT_kendra_and_tony_keep_two_each_l367_36707

-- Define the conditions
def kendra_packs : Nat := 4
def tony_packs : Nat := 2
def pens_per_pack : Nat := 3
def pens_given_to_friends : Nat := 14

-- Define the total pens each has
def kendra_pens : Nat := kendra_packs * pens_per_pack
def tony_pens : Nat := tony_packs * pens_per_pack

-- Define the total pens
def total_pens : Nat := kendra_pens + tony_pens

-- Define the pens left after distribution
def pens_left : Nat := total_pens - pens_given_to_friends

-- Define the number of pens each keeps
def pens_each_kept : Nat := pens_left / 2

-- Prove the final statement
theorem kendra_and_tony_keep_two_each :
  pens_each_kept = 2 :=
by
  sorry

end NUMINAMATH_GPT_kendra_and_tony_keep_two_each_l367_36707


namespace NUMINAMATH_GPT_interest_rate_proof_l367_36771

variable (P : ℝ) (n : ℕ) (CI SI : ℝ → ℝ → ℕ → ℝ) (diff : ℝ → ℝ → ℝ)

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n
def simple_interest (P r : ℝ) (n : ℕ) : ℝ := P * r * n

theorem interest_rate_proof (r : ℝ) :
  diff (compound_interest 5400 r 2) (simple_interest 5400 r 2) = 216 → r = 0.2 :=
by sorry

end NUMINAMATH_GPT_interest_rate_proof_l367_36771


namespace NUMINAMATH_GPT_smallest_n_l367_36772

theorem smallest_n (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3)
  (h4 : x * y * z ∣ (x + y + z)^n) : n = 13 :=
sorry

end NUMINAMATH_GPT_smallest_n_l367_36772


namespace NUMINAMATH_GPT_triangle_construction_possible_l367_36789

theorem triangle_construction_possible (r l_alpha k_alpha : ℝ) (h1 : r > 0) (h2 : l_alpha > 0) (h3 : k_alpha > 0) :
  l_alpha^2 < (4 * k_alpha^2 * r^2) / (k_alpha^2 + r^2) :=
sorry

end NUMINAMATH_GPT_triangle_construction_possible_l367_36789


namespace NUMINAMATH_GPT_bc_guilty_l367_36783

-- Definition of guilty status of defendants
variables (A B C : Prop)

-- Conditions
axiom condition1 : A ∨ B ∨ C
axiom condition2 : A → ¬B → ¬C

-- Theorem stating that one of B or C is guilty
theorem bc_guilty : B ∨ C :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_bc_guilty_l367_36783


namespace NUMINAMATH_GPT_arithmetic_sequence_value_y_l367_36781

theorem arithmetic_sequence_value_y :
  ∀ (a₁ a₃ y : ℤ), 
  a₁ = 3 ^ 3 →
  a₃ = 5 ^ 3 →
  y = (a₁ + a₃) / 2 →
  y = 76 :=
by 
  intros a₁ a₃ y h₁ h₃ hy 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_y_l367_36781


namespace NUMINAMATH_GPT_sum_coordinates_D_is_13_l367_36709

theorem sum_coordinates_D_is_13 
  (A B C D : ℝ × ℝ) 
  (hA : A = (4, 8))
  (hB : B = (2, 2))
  (hC : C = (6, 4))
  (hD : D = (8, 5))
  (h_mid1 : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 5)
  (h_mid2 : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 3)
  (h_mid3 : (C.1 + D.1) / 2 = 7 ∧ (C.2 + D.2) / 2 = 4.5)
  (h_mid4 : (D.1 + A.1) / 2 = 6 ∧ (D.2 + A.2) / 2 = 6.5)
  (h_square : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 5) ∧
               ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = (4, 3) ∧
               ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 4.5) ∧
               ((D.1 + A.1) / 2, (D.2 + A.2) / 2) = (6, 6.5))
  : (8 + 5) = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_coordinates_D_is_13_l367_36709


namespace NUMINAMATH_GPT_motorcycles_meet_after_54_minutes_l367_36782

noncomputable def motorcycles_meet_time : ℕ := sorry

theorem motorcycles_meet_after_54_minutes :
  motorcycles_meet_time = 54 := sorry

end NUMINAMATH_GPT_motorcycles_meet_after_54_minutes_l367_36782


namespace NUMINAMATH_GPT_sum_of_integers_is_18_l367_36737

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_18_l367_36737


namespace NUMINAMATH_GPT_simplify_expression_l367_36705

theorem simplify_expression :
  (360 / 24) * (10 / 240) * (6 / 3) * (9 / 18) = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l367_36705


namespace NUMINAMATH_GPT_variance_of_heights_l367_36768
-- Importing all necessary libraries

-- Define a list of heights
def heights : List ℕ := [160, 162, 159, 160, 159]

-- Define the function to calculate the mean of a list of natural numbers
def mean (list : List ℕ) : ℚ :=
  list.sum / list.length

-- Define the function to calculate the variance of a list of natural numbers
def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (λ x => (x - μ) ^ 2)).sum / list.length

-- The theorem statement that proves the variance is 6/5
theorem variance_of_heights : variance heights = 6 / 5 :=
  sorry

end NUMINAMATH_GPT_variance_of_heights_l367_36768


namespace NUMINAMATH_GPT_no_n_satisfies_l367_36720

def sum_first_n_terms_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem no_n_satisfies (n : ℕ) (h_n : n ≠ 0) :
  let s1 := sum_first_n_terms_arith_seq 5 6 n
  let s2 := sum_first_n_terms_arith_seq 12 4 n
  (s1 * s2 = 24 * n^2) → False :=
by
  sorry

end NUMINAMATH_GPT_no_n_satisfies_l367_36720


namespace NUMINAMATH_GPT_gcd_possible_values_count_l367_36732

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end NUMINAMATH_GPT_gcd_possible_values_count_l367_36732


namespace NUMINAMATH_GPT_polynomial_value_l367_36734

theorem polynomial_value 
  (x : ℝ) 
  (h1 : x = (1 + (1994 : ℝ).sqrt) / 2) : 
  (4 * x ^ 3 - 1997 * x - 1994) ^ 20001 = -1 := 
  sorry

end NUMINAMATH_GPT_polynomial_value_l367_36734


namespace NUMINAMATH_GPT_total_pieces_of_clothing_l367_36727

-- Define Kaleb's conditions
def pieces_in_one_load : ℕ := 19
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- The total pieces of clothing Kaleb has
theorem total_pieces_of_clothing : pieces_in_one_load + num_equal_loads * pieces_per_load = 39 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_clothing_l367_36727


namespace NUMINAMATH_GPT_find_c_plus_d_l367_36763

theorem find_c_plus_d (c d : ℝ) (h1 : 2 * c = 6) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l367_36763


namespace NUMINAMATH_GPT_greatest_third_side_l367_36759

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end NUMINAMATH_GPT_greatest_third_side_l367_36759


namespace NUMINAMATH_GPT_inequality_solution_sum_of_m_and_2n_l367_36725

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := abs (x - a)

-- Part (1): The inequality problem for a = 2
theorem inequality_solution (x : ℝ) :
  f x 2 ≥ 4 - abs (x - 1) → x ≤ 2 / 3 := sorry

-- Part (2): Given conditions with solution set [0, 2] and condition on m and n
theorem sum_of_m_and_2n (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : ∀ x, f x 1 ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) (h₄ : 1 / m + 1 / (2 * n) = 1) :
  m + 2 * n ≥ 4 := sorry

end NUMINAMATH_GPT_inequality_solution_sum_of_m_and_2n_l367_36725


namespace NUMINAMATH_GPT_triangle_is_right_l367_36766

-- Definitions based on the conditions given in the problem
variables {a b c A B C : ℝ}

-- Introduction of the conditions in Lean
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

def given_condition (A b c : ℝ) : Prop :=
  (Real.cos (A / 2))^2 = (b + c) / (2 * c)

-- Theorem statement to prove the conclusion based on given conditions
theorem triangle_is_right (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_given : given_condition A b c) :
  A = 90 := sorry

end NUMINAMATH_GPT_triangle_is_right_l367_36766


namespace NUMINAMATH_GPT_fraction_power_multiplication_l367_36767

theorem fraction_power_multiplication :
  ( (8 / 9)^3 * (5 / 3)^3 ) = (64000 / 19683) :=
by
  sorry

end NUMINAMATH_GPT_fraction_power_multiplication_l367_36767


namespace NUMINAMATH_GPT_solve_equation_l367_36788

theorem solve_equation : ∃ x : ℝ, (x^3 - ⌊x⌋ = 3) := 
sorry

end NUMINAMATH_GPT_solve_equation_l367_36788


namespace NUMINAMATH_GPT_solve_logarithmic_equation_l367_36722

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solve_logarithmic_equation (x : ℝ) (h_pos : x > 0) :
  log_base 8 x + log_base 4 (x^2) + log_base 2 (x^3) = 15 ↔ x = 2 ^ (45 / 13) :=
by
  have h1 : log_base 8 x = (1 / 3) * log_base 2 x :=
    by { sorry }
  have h2 : log_base 4 (x^2) = log_base 2 x :=
    by { sorry }
  have h3 : log_base 2 (x^3) = 3 * log_base 2 x :=
    by { sorry }
  have h4 : (1 / 3) * log_base 2 x + log_base 2 x + 3 * log_base 2 x = 15 ↔ log_base 2 x = 45 / 13 :=
    by { sorry }
  exact sorry

end NUMINAMATH_GPT_solve_logarithmic_equation_l367_36722


namespace NUMINAMATH_GPT_packages_eq_nine_l367_36775

-- Definitions of the given conditions
def x : ℕ := 50
def y : ℕ := 5
def z : ℕ := 5

-- Statement: Prove that the number of packages Amy could make equals 9
theorem packages_eq_nine : (x - y) / z = 9 :=
by
  sorry

end NUMINAMATH_GPT_packages_eq_nine_l367_36775


namespace NUMINAMATH_GPT_proposition_B_correct_l367_36764

theorem proposition_B_correct (a b c : ℝ) (hc : c ≠ 0) : ac^2 > b * c^2 → a > b := sorry

end NUMINAMATH_GPT_proposition_B_correct_l367_36764


namespace NUMINAMATH_GPT_general_term_of_arithmetic_seq_l367_36706

variable {a : ℕ → ℤ}

def arithmetic_seq (a : ℕ → ℤ) := ∃ d, ∀ n, a n = a 0 + n * d

theorem general_term_of_arithmetic_seq :
  arithmetic_seq a →
  a 2 = 9 →
  (∃ x y, (x ^ 2 - 16 * x + 60 = 0) ∧ (a 0 = x) ∧ (a 4 = y)) →
  ∀ n, a n = -n + 11 :=
by
  intros h_arith h_a2 h_root
  sorry

end NUMINAMATH_GPT_general_term_of_arithmetic_seq_l367_36706


namespace NUMINAMATH_GPT_sum_of_even_sequence_is_194_l367_36739

theorem sum_of_even_sequence_is_194
  (a b c d : ℕ) 
  (even_a : a % 2 = 0) 
  (even_b : b % 2 = 0) 
  (even_c : c % 2 = 0) 
  (even_d : d % 2 = 0)
  (a_lt_b : a < b) 
  (b_lt_c : b < c) 
  (c_lt_d : c < d)
  (diff_da : d - a = 90)
  (arith_ab_c : 2 * b = a + c)
  (geo_bc_d : c^2 = b * d)
  : a + b + c + d = 194 := 
sorry

end NUMINAMATH_GPT_sum_of_even_sequence_is_194_l367_36739


namespace NUMINAMATH_GPT_true_propositions_l367_36730

theorem true_propositions : 
  (∀ x : ℝ, x^3 < 1 → x^2 + 1 > 0) ∧ (∀ x : ℚ, x^2 = 2 → false) ∧ 
  (∀ x : ℕ, x^3 > x^2 → false) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_true_propositions_l367_36730


namespace NUMINAMATH_GPT_trapezoid_fraction_l367_36711

theorem trapezoid_fraction 
  (shorter_base longer_base side_length : ℝ)
  (angle_adjacent : ℝ)
  (h1 : shorter_base = 120)
  (h2 : longer_base = 180)
  (h3 : side_length = 130)
  (h4 : angle_adjacent = 60) :
  ∃ fraction : ℝ, fraction = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_fraction_l367_36711


namespace NUMINAMATH_GPT_part_I_intersection_part_I_union_complements_part_II_range_l367_36729

namespace MathProof

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Prove that the intersection of A and B is {x | 3 < x ∧ x < 6}
theorem part_I_intersection : A ∩ B = {x | 3 < x ∧ x < 6} := sorry

-- Prove that the union of the complements of A and B is {x | x ≤ 3 ∨ x ≥ 6}
theorem part_I_union_complements : (Aᶜ ∪ Bᶜ) = {x | x ≤ 3 ∨ x ≥ 6} := sorry

-- Prove the range of a such that C is a subset of B and B union C equals B
theorem part_II_range (a : ℝ) : B ∪ C a = B → (a ≤ 1 ∨ 2 ≤ a ∧ a ≤ 5) := sorry

end MathProof

end NUMINAMATH_GPT_part_I_intersection_part_I_union_complements_part_II_range_l367_36729


namespace NUMINAMATH_GPT_sam_possible_lunches_without_violation_l367_36736

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end NUMINAMATH_GPT_sam_possible_lunches_without_violation_l367_36736


namespace NUMINAMATH_GPT_train_speed_84_kmph_l367_36795

theorem train_speed_84_kmph (length : ℕ) (time : ℕ) (conversion_factor : ℚ)
  (h_length : length = 140) (h_time : time = 6) (h_conversion_factor : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 84 :=
  sorry

end NUMINAMATH_GPT_train_speed_84_kmph_l367_36795


namespace NUMINAMATH_GPT_jerome_family_members_l367_36777

-- Define the conditions of the problem
variables (C F M T : ℕ)
variables (hC : C = 20) (hF : F = C / 2) (hT : T = 33)

-- Formulate the theorem to prove
theorem jerome_family_members :
  M = T - (C + F) :=
sorry

end NUMINAMATH_GPT_jerome_family_members_l367_36777


namespace NUMINAMATH_GPT_quadrilateral_angle_l367_36738

theorem quadrilateral_angle (x y : ℝ) (h1 : 3 * x ^ 2 - x + 4 = 5) (h2 : x ^ 2 + y ^ 2 = 9) :
  x = (1 + Real.sqrt 13) / 6 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_angle_l367_36738


namespace NUMINAMATH_GPT_min_value_l367_36773

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 10 * x - 6 * y + 3

theorem min_value : ∃ x y : ℝ, (x + y = 2) ∧ (f x y = -(1/7)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_l367_36773


namespace NUMINAMATH_GPT_spending_on_gifts_l367_36731

-- Defining the conditions as Lean statements
def num_sons_teachers : ℕ := 3
def num_daughters_teachers : ℕ := 4
def cost_per_gift : ℕ := 10

-- The total number of teachers
def total_teachers : ℕ := num_sons_teachers + num_daughters_teachers

-- Proving that the total spending on gifts is $70
theorem spending_on_gifts : total_teachers * cost_per_gift = 70 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_spending_on_gifts_l367_36731


namespace NUMINAMATH_GPT_least_positive_integer_to_add_l367_36735

theorem least_positive_integer_to_add (n : ℕ) (h1 : n > 0) (h2 : (624 + n) % 5 = 0) : n = 1 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_to_add_l367_36735


namespace NUMINAMATH_GPT_find_k_l367_36717

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l367_36717


namespace NUMINAMATH_GPT_number_of_diet_soda_bottles_l367_36746

theorem number_of_diet_soda_bottles (apples regular_soda total_bottles diet_soda : ℕ)
    (h_apples : apples = 36)
    (h_regular_soda : regular_soda = 80)
    (h_total_bottles : total_bottles = apples + 98)
    (h_diet_soda_eq : total_bottles = regular_soda + diet_soda) :
    diet_soda = 54 := by
  sorry

end NUMINAMATH_GPT_number_of_diet_soda_bottles_l367_36746


namespace NUMINAMATH_GPT_transform_equation_to_polynomial_l367_36774

variable (x y : ℝ)

theorem transform_equation_to_polynomial (h : (x^2 + 2) / (x + 1) = y) :
    (x^2 + 2) / (x + 1) + (5 * (x + 1)) / (x^2 + 2) = 6 → y^2 - 6 * y + 5 = 0 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_transform_equation_to_polynomial_l367_36774


namespace NUMINAMATH_GPT_problem_statement_l367_36712

-- Define the statement for positive integers m and n
def div_equiv (m n : ℕ) : Prop :=
  19 ∣ (11 * m + 2 * n) ↔ 19 ∣ (18 * m + 5 * n)

-- The final theorem statement
theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : div_equiv m n :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l367_36712


namespace NUMINAMATH_GPT_correct_system_of_equations_l367_36723

theorem correct_system_of_equations
  (x y : ℝ)
  (h1 : x + (1 / 2) * y = 50)
  (h2 : y + (2 / 3) * x = 50) :
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_correct_system_of_equations_l367_36723


namespace NUMINAMATH_GPT_largest_divisor_n4_n2_l367_36757

theorem largest_divisor_n4_n2 (n : ℤ) : (6 : ℤ) ∣ (n^4 - n^2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_n4_n2_l367_36757


namespace NUMINAMATH_GPT_area_of_30_60_90_triangle_l367_36786

theorem area_of_30_60_90_triangle (altitude : ℝ) (h : altitude = 3) : 
  ∃ (area : ℝ), area = 6 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_area_of_30_60_90_triangle_l367_36786


namespace NUMINAMATH_GPT_angle_A_in_triangle_l367_36776

theorem angle_A_in_triangle (a b c : ℝ) (h : a^2 = b^2 + b * c + c^2) : A = 120 :=
sorry

end NUMINAMATH_GPT_angle_A_in_triangle_l367_36776


namespace NUMINAMATH_GPT_sum_of_digits_0_to_999_l367_36787

-- Sum of digits from 0 to 9
def sum_of_digits : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Sum of digits from 1 to 9
def sum_of_digits_without_zero : ℕ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Units place sum
def units_sum : ℕ := sum_of_digits * 100

-- Tens place sum
def tens_sum : ℕ := sum_of_digits * 100

-- Hundreds place sum
def hundreds_sum : ℕ := sum_of_digits_without_zero * 100

-- Total sum
def total_sum : ℕ := units_sum + tens_sum + hundreds_sum

theorem sum_of_digits_0_to_999 : total_sum = 13500 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_0_to_999_l367_36787


namespace NUMINAMATH_GPT_parallel_lines_l367_36749

theorem parallel_lines (a : ℝ) :
  ((3 * a + 2) * x + a * y + 6 = 0) ↔
  (a * x - y + 3 = 0) →
  a = -1 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_l367_36749


namespace NUMINAMATH_GPT_linda_total_miles_l367_36758

def calculate_total_miles (x : ℕ) : ℕ :=
  (60 / x) + (60 / (x + 4)) + (60 / (x + 8)) + (60 / (x + 12)) + (60 / (x + 16))

theorem linda_total_miles (x : ℕ) (hx1 : x > 0)
(hdx2 : 60 % x = 0)
(hdx3 : 60 % (x + 4) = 0) 
(hdx4 : 60 % (x + 8) = 0) 
(hdx5 : 60 % (x + 12) = 0) 
(hdx6 : 60 % (x + 16) = 0) :
  calculate_total_miles x = 33 := by
  sorry

end NUMINAMATH_GPT_linda_total_miles_l367_36758


namespace NUMINAMATH_GPT_tail_count_likelihood_draw_and_rainy_l367_36799

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draws_when_heads : ℕ := 7
def rainy_when_tails : ℕ := 4

theorem tail_count :
  coin_tosses - heads_count = 14 :=
sorry

theorem likelihood_draw_and_rainy :
  0 = 0 :=
sorry

end NUMINAMATH_GPT_tail_count_likelihood_draw_and_rainy_l367_36799


namespace NUMINAMATH_GPT_event_A_probability_l367_36760

theorem event_A_probability (n : ℕ) (m₀ : ℕ) (H_n : n = 120) (H_m₀ : m₀ = 32) (p : ℝ) :
  (n * p - (1 - p) ≤ m₀) ∧ (n * p + p ≥ m₀) → 
  (32 / 121 : ℝ) ≤ p ∧ p ≤ (33 / 121 : ℝ) :=
sorry

end NUMINAMATH_GPT_event_A_probability_l367_36760


namespace NUMINAMATH_GPT_average_minutes_per_day_l367_36744

theorem average_minutes_per_day (e : ℕ) (h_e_pos : 0 < e) : 
  let sixth_grade_minutes := 20
  let seventh_grade_minutes := 18
  let eighth_grade_minutes := 12
  
  let sixth_graders := 3 * e
  let seventh_graders := 4 * e
  let eighth_graders := e
  
  let total_minutes := sixth_grade_minutes * sixth_graders + seventh_grade_minutes * seventh_graders + eighth_grade_minutes * eighth_graders
  let total_students := sixth_graders + seventh_graders + eighth_graders
  
  (total_minutes / total_students) = 18 := by
sorry

end NUMINAMATH_GPT_average_minutes_per_day_l367_36744


namespace NUMINAMATH_GPT_perpendicular_vectors_vector_sum_norm_min_value_f_l367_36701

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3*x/2), Real.sin (3*x/2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem perpendicular_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 ↔ x = Real.pi / 4 := sorry

theorem vector_sum_norm (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ 1 ↔ 0 ≤ x ∧ x ≤ Real.pi / 3 := sorry

theorem min_value_f (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≥ -2) ↔ m = Real.sqrt 2 / 2 := sorry

end NUMINAMATH_GPT_perpendicular_vectors_vector_sum_norm_min_value_f_l367_36701


namespace NUMINAMATH_GPT_probability_of_same_team_is_one_third_l367_36797

noncomputable def probability_same_team : ℚ :=
  let teams := 3
  let total_combinations := teams * teams
  let successful_outcomes := teams
  successful_outcomes / total_combinations

theorem probability_of_same_team_is_one_third :
  probability_same_team = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_same_team_is_one_third_l367_36797


namespace NUMINAMATH_GPT_totalWatermelons_l367_36750

def initialWatermelons : ℕ := 4
def additionalWatermelons : ℕ := 3

theorem totalWatermelons : initialWatermelons + additionalWatermelons = 7 := by
  sorry

end NUMINAMATH_GPT_totalWatermelons_l367_36750
