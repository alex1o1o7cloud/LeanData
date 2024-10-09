import Mathlib

namespace long_show_episodes_correct_l1402_140213

variable {short_show_episodes : ℕ} {short_show_duration : ℕ} {total_watched_time : ℕ} {long_show_episode_duration : ℕ}

def episodes_long_show (short_episodes_duration total_duration long_episode_duration : ℕ) : ℕ :=
  (total_duration - short_episodes_duration) / long_episode_duration

theorem long_show_episodes_correct :
  ∀ (short_show_episodes short_show_duration total_watched_time long_show_episode_duration : ℕ),
  short_show_episodes = 24 →
  short_show_duration = 1 / 2 →
  total_watched_time = 24 →
  long_show_episode_duration = 1 →
  episodes_long_show (short_show_episodes * short_show_duration) total_watched_time long_show_episode_duration = 12 := by
  intros
  sorry

end long_show_episodes_correct_l1402_140213


namespace connections_required_l1402_140238

theorem connections_required (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end connections_required_l1402_140238


namespace ratio_spaghetti_pizza_l1402_140257

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

end ratio_spaghetti_pizza_l1402_140257


namespace smallest_angle_of_triangle_l1402_140201

theorem smallest_angle_of_triangle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) : 3 * x = 45 :=
by
  sorry

end smallest_angle_of_triangle_l1402_140201


namespace car_rental_daily_rate_l1402_140265

theorem car_rental_daily_rate (x : ℝ) : 
  (x + 0.18 * 48 = 18.95 + 0.16 * 48) -> 
  x = 17.99 :=
by 
  sorry

end car_rental_daily_rate_l1402_140265


namespace find_phi_l1402_140220

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem find_phi (phi : ℝ) (h_shift : ∀ x : ℝ, f (x + phi) = f (-x - phi)) : 
  phi = Real.pi / 8 :=
  sorry

end find_phi_l1402_140220


namespace tetrahedron_volume_correct_l1402_140298

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

theorem tetrahedron_volume_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 = c^2) :
  tetrahedron_volume a b c = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by
  sorry

end tetrahedron_volume_correct_l1402_140298


namespace complex_number_z_satisfies_l1402_140215

theorem complex_number_z_satisfies (z : ℂ) : 
  (z * (1 + I) + (-I) * (1 - I) = 0) → z = -1 := 
by {
  sorry
}

end complex_number_z_satisfies_l1402_140215


namespace quincy_sold_more_than_jake_l1402_140252

theorem quincy_sold_more_than_jake :
  ∀ (T Jake : ℕ), Jake = 2 * T + 15 → 4000 = 100 * (T + Jake) → 4000 - Jake = 3969 :=
by
  intros T Jake hJake hQuincy
  sorry

end quincy_sold_more_than_jake_l1402_140252


namespace factor_expression_l1402_140219

theorem factor_expression (x : ℝ) : 
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l1402_140219


namespace radius_of_circle_from_chord_and_line_l1402_140203

theorem radius_of_circle_from_chord_and_line (r : ℝ) (t θ : ℝ) 
    (param_line : ℝ × ℝ) (param_circle : ℝ × ℝ)
    (chord_length : ℝ) 
    (h1 : param_line = (3 + 3 * t, 1 - 4 * t))
    (h2 : param_circle = (r * Real.cos θ, r * Real.sin θ))
    (h3 : chord_length = 4) 
    : r = Real.sqrt 13 :=
sorry

end radius_of_circle_from_chord_and_line_l1402_140203


namespace find_a_in_subset_l1402_140270

theorem find_a_in_subset 
  (A : Set ℝ)
  (B : Set ℝ)
  (hA : A = { x | x^2 ≠ 1 })
  (hB : ∃ a : ℝ, B = { x | a * x = 1 })
  (h_subset : B ⊆ A) : 
  ∃ a : ℝ, a = 0 ∨ a = 1 ∨ a = -1 := 
by
  sorry

end find_a_in_subset_l1402_140270


namespace solution_interval_l1402_140242

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ (5 / 2 : ℝ) < x ∧ x ≤ (14 / 5 : ℝ) := 
by
  sorry

end solution_interval_l1402_140242


namespace initial_population_l1402_140225

variable (P : ℕ)

theorem initial_population
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℚ := 1.2) :
  (P = 3000) :=
by
  sorry

end initial_population_l1402_140225


namespace tan_A_plus_C_eq_neg_sqrt3_l1402_140236

theorem tan_A_plus_C_eq_neg_sqrt3
  (A B C : Real)
  (hSum : A + B + C = Real.pi)
  (hArithSeq : 2 * B = A + C)
  (hTriangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 := by
  sorry

end tan_A_plus_C_eq_neg_sqrt3_l1402_140236


namespace prob_exactly_M_laws_expected_laws_included_l1402_140237

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l1402_140237


namespace average_production_is_correct_l1402_140299

noncomputable def average_tv_production_last_5_days
  (daily_production : ℕ)
  (ill_workers : List ℕ)
  (decrease_rate : ℕ) : ℚ :=
  let productivity_decrease (n : ℕ) : ℚ := (1 - (decrease_rate * n) / 100 : ℚ) * daily_production
  let total_production := (ill_workers.map productivity_decrease).sum
  total_production / ill_workers.length

theorem average_production_is_correct :
  average_tv_production_last_5_days 50 [3, 5, 2, 4, 3] 2 = 46.6 :=
by
  -- proof needed here
  sorry

end average_production_is_correct_l1402_140299


namespace intersection_union_complement_l1402_140227

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def universal_set := U = univ
def set_A := A = {x : ℝ | -1 ≤ x ∧ x < 2}
def set_B := B = {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

theorem union (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry

theorem complement (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) :
  U \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := sorry

end intersection_union_complement_l1402_140227


namespace complementary_event_target_l1402_140233

theorem complementary_event_target (S : Type) (hit miss : S) (shoots : ℕ → S) :
  (∀ n : ℕ, (shoots n = hit ∨ shoots n = miss)) →
  (∃ n : ℕ, shoots n = hit) ↔ (∀ n : ℕ, shoots n ≠ hit) :=
by
sorry

end complementary_event_target_l1402_140233


namespace find_a_l1402_140247

def F (a b c : ℝ) : ℝ := a * b^3 + c

theorem find_a (a : ℝ) (h : F a 3 8 = F a 5 12) : a = -2 / 49 := by
  sorry

end find_a_l1402_140247


namespace minimum_trucks_needed_l1402_140226

theorem minimum_trucks_needed (total_weight : ℝ) (box_weight : ℕ → ℝ) 
  (n : ℕ) (H_total_weight : total_weight = 10) 
  (H_box_weight : ∀ i, box_weight i ≤ 1) 
  (truck_capacity : ℝ) 
  (H_truck_capacity : truck_capacity = 3) : 
  n = 5 :=
by {
  sorry
}

end minimum_trucks_needed_l1402_140226


namespace second_largest_is_D_l1402_140286

noncomputable def A := 3 * 3
noncomputable def C := 4 * A
noncomputable def B := C - 15
noncomputable def D := A + 19

theorem second_largest_is_D : 
    ∀ (A B C D : ℕ), 
      A = 9 → 
      B = 21 →
      C = 36 →
      D = 28 →
      D = 28 :=
by
  intros A B C D hA hB hC hD
  have h1 : A = 9 := by assumption
  have h2 : B = 21 := by assumption
  have h3 : C = 36 := by assumption
  have h4 : D = 28 := by assumption
  exact h4

end second_largest_is_D_l1402_140286


namespace find_x_l1402_140210

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_x (x : ℝ) (hx : x > 0) :
  distance (1, 3) (x, -4) = 15 → x = 1 + Real.sqrt 176 :=
by
  sorry

end find_x_l1402_140210


namespace tony_gas_expense_in_4_weeks_l1402_140290

theorem tony_gas_expense_in_4_weeks :
  let miles_per_gallon := 25
  let miles_per_round_trip_per_day := 50
  let travel_days_per_week := 5
  let tank_capacity_in_gallons := 10
  let cost_per_gallon := 2
  let weeks := 4
  let total_miles_per_week := miles_per_round_trip_per_day * travel_days_per_week
  let total_miles := total_miles_per_week * weeks
  let miles_per_tank := miles_per_gallon * tank_capacity_in_gallons
  let fill_ups_needed := total_miles / miles_per_tank
  let total_gallons_needed := fill_ups_needed * tank_capacity_in_gallons
  let total_cost := total_gallons_needed * cost_per_gallon
  total_cost = 80 :=
by
  sorry

end tony_gas_expense_in_4_weeks_l1402_140290


namespace find_4a_add_c_find_2a_sub_2b_sub_c_l1402_140278

variables {R : Type*} [CommRing R]

theorem find_4a_add_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  4 * a + c = 12 :=
sorry

theorem find_2a_sub_2b_sub_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  2 * a - 2 * b - c = 14 :=
sorry

end find_4a_add_c_find_2a_sub_2b_sub_c_l1402_140278


namespace isosceles_triangle_interior_angles_l1402_140251

theorem isosceles_triangle_interior_angles (a b c : ℝ) 
  (h1 : b = c) (h2 : a + b + c = 180) (exterior : a + 40 = 180 ∨ b + 40 = 140) :
  (a = 40 ∧ b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40 ∧ c = 40) :=
by
  sorry

end isosceles_triangle_interior_angles_l1402_140251


namespace bobby_finishes_candies_in_weeks_l1402_140296

def total_candies (packets: Nat) (candies_per_packet: Nat) : Nat := packets * candies_per_packet

def candies_eaten_per_week (candies_per_day_mon_fri: Nat) (days_mon_fri: Nat) (candies_per_day_weekend: Nat) (days_weekend: Nat) : Nat :=
  (candies_per_day_mon_fri * days_mon_fri) + (candies_per_day_weekend * days_weekend)

theorem bobby_finishes_candies_in_weeks :
  let packets := 2
  let candies_per_packet := 18
  let candies_per_day_mon_fri := 2
  let days_mon_fri := 5
  let candies_per_day_weekend := 1
  let days_weekend := 2

  total_candies packets candies_per_packet / candies_eaten_per_week candies_per_day_mon_fri days_mon_fri candies_per_day_weekend days_weekend = 3 :=
by
  sorry

end bobby_finishes_candies_in_weeks_l1402_140296


namespace number_of_lamps_bought_l1402_140239

-- Define the given conditions
def price_of_lamp : ℕ := 7
def price_of_bulb : ℕ := price_of_lamp - 4
def bulbs_bought : ℕ := 6
def total_spent : ℕ := 32

-- Define the statement to prove
theorem number_of_lamps_bought : 
  ∃ (L : ℕ), (price_of_lamp * L + price_of_bulb * bulbs_bought = total_spent) ∧ (L = 2) :=
sorry

end number_of_lamps_bought_l1402_140239


namespace find_other_leg_length_l1402_140202

theorem find_other_leg_length (a b c : ℝ) (h1 : a = 15) (h2 : b = 5 * Real.sqrt 3) (h3 : c = 2 * (5 * Real.sqrt 3)) (h4 : a^2 + b^2 = c^2)
  (angle_A : ℝ) (h5 : angle_A = Real.pi / 3) (h6 : angle_A ≠ Real.pi / 2) :
  b = 5 * Real.sqrt 3 :=
by
  sorry

end find_other_leg_length_l1402_140202


namespace line_eq_x_1_parallel_y_axis_l1402_140284

theorem line_eq_x_1_parallel_y_axis (P : ℝ × ℝ) (hP : P = (1, 0)) (h_parallel : ∀ y : ℝ, (1, y) = P ∨ P = (1, y)) :
  ∃ x : ℝ, (∀ y : ℝ, P = (x, y)) → x = 1 := 
by 
  sorry

end line_eq_x_1_parallel_y_axis_l1402_140284


namespace total_flowers_in_vase_l1402_140280

-- Conditions as definitions
def num_roses : ℕ := 5
def num_lilies : ℕ := 2

-- Theorem statement
theorem total_flowers_in_vase : num_roses + num_lilies = 7 :=
by
  sorry

end total_flowers_in_vase_l1402_140280


namespace sufficient_but_not_necessary_condition_l1402_140205

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ ¬ (|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1402_140205


namespace car_a_speed_l1402_140269

theorem car_a_speed (d_A d_B v_B t v_A : ℝ)
  (h1 : d_A = 10)
  (h2 : v_B = 50)
  (h3 : t = 2.25)
  (h4 : d_A + 8 - d_B = v_A * t)
  (h5 : d_B = v_B * t) :
  v_A = 58 :=
by
  -- Work on the proof here
  sorry

end car_a_speed_l1402_140269


namespace child_sold_apples_correct_l1402_140254

-- Definitions based on conditions
def initial_apples (children : ℕ) (apples_per_child : ℕ) : ℕ := children * apples_per_child
def eaten_apples (children_eating : ℕ) (apples_eaten_per_child : ℕ) : ℕ := children_eating * apples_eaten_per_child
def remaining_apples (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten
def sold_apples (remaining : ℕ) (final : ℕ) : ℕ := remaining - final

-- Given conditions
variable (children : ℕ := 5)
variable (apples_per_child : ℕ := 15)
variable (children_eating : ℕ := 2)
variable (apples_eaten_per_child : ℕ := 4)
variable (final_apples : ℕ := 60)

-- Theorem statement
theorem child_sold_apples_correct :
  sold_apples (remaining_apples (initial_apples children apples_per_child) (eaten_apples children_eating apples_eaten_per_child)) final_apples = 7 :=
by
  sorry -- Proof is omitted

end child_sold_apples_correct_l1402_140254


namespace percentage_rotten_apples_l1402_140200

theorem percentage_rotten_apples
  (total_apples : ℕ)
  (smell_pct : ℚ)
  (non_smelling_rotten_apples : ℕ)
  (R : ℚ) :
  total_apples = 200 →
  smell_pct = 0.70 →
  non_smelling_rotten_apples = 24 →
  0.30 * (R / 100 * total_apples) = non_smelling_rotten_apples →
  R = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_rotten_apples_l1402_140200


namespace cube_volume_of_surface_area_l1402_140295

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l1402_140295


namespace max_profit_at_9_l1402_140241

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
else if h : x > 10 then 108 / x - 1000 / (3 * x^2)
else 0

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
else if h : x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
else 0

theorem max_profit_at_9 : W 9 = 38.6 :=
sorry

end max_profit_at_9_l1402_140241


namespace roots_quadratic_expression_value_l1402_140231

theorem roots_quadratic_expression_value (m n : ℝ) 
  (h1 : m^2 + 2 * m - 2027 = 0)
  (h2 : n^2 + 2 * n - 2027 = 0) :
  (2 * m - m * n + 2 * n) = 2023 :=
by
  sorry

end roots_quadratic_expression_value_l1402_140231


namespace circle_form_eq_standard_form_l1402_140272

theorem circle_form_eq_standard_form :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 11 := 
by
  intro x y
  sorry

end circle_form_eq_standard_form_l1402_140272


namespace kanul_spent_on_machinery_l1402_140243

theorem kanul_spent_on_machinery (total raw_materials cash M : ℝ) 
  (h_total : total = 7428.57) 
  (h_raw_materials : raw_materials = 5000) 
  (h_cash : cash = 0.30 * total) 
  (h_expenditure : total = raw_materials + M + cash) :
  M = 200 := 
by
  sorry

end kanul_spent_on_machinery_l1402_140243


namespace find_areas_after_shortening_l1402_140244

-- Define initial dimensions
def initial_length : ℤ := 5
def initial_width : ℤ := 7
def shortened_by : ℤ := 2

-- Define initial area condition
def initial_area_condition : Prop := 
  initial_length * (initial_width - shortened_by) = 15 ∨ (initial_length - shortened_by) * initial_width = 15

-- Define the resulting areas for shortening each dimension
def area_shortening_length : ℤ := (initial_length - shortened_by) * initial_width
def area_shortening_width : ℤ := initial_length * (initial_width - shortened_by)

-- Statement for proof
theorem find_areas_after_shortening
  (h : initial_area_condition) :
  area_shortening_length = 21 ∧ area_shortening_width = 25 :=
sorry

end find_areas_after_shortening_l1402_140244


namespace sum_of_squares_iff_double_sum_of_squares_l1402_140248

theorem sum_of_squares_iff_double_sum_of_squares (n : ℕ) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ a b : ℤ, 2 * n = a^2 + b^2) :=
sorry

end sum_of_squares_iff_double_sum_of_squares_l1402_140248


namespace three_digit_integers_product_30_l1402_140204

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end three_digit_integers_product_30_l1402_140204


namespace mean_transformation_l1402_140291

variable {x1 x2 x3 : ℝ}
variable (s : ℝ)
variable (h_var : s^2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12))

theorem mean_transformation :
  (x1 + 1 + x2 + 1 + x3 + 1) / 3 = 3 :=
by
  sorry

end mean_transformation_l1402_140291


namespace total_students_standing_committee_ways_different_grade_pairs_ways_l1402_140268

-- Given conditions
def freshmen : ℕ := 5
def sophomores : ℕ := 6
def juniors : ℕ := 4

-- Proofs (statements only, no proofs provided)
theorem total_students : freshmen + sophomores + juniors = 15 :=
by sorry

theorem standing_committee_ways : freshmen * sophomores * juniors = 120 :=
by sorry

theorem different_grade_pairs_ways :
  freshmen * sophomores + sophomores * juniors + juniors * freshmen = 74 :=
by sorry

end total_students_standing_committee_ways_different_grade_pairs_ways_l1402_140268


namespace symmetric_line_eq_x_axis_l1402_140259

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_eq_x_axis_l1402_140259


namespace min_possible_value_box_l1402_140293

theorem min_possible_value_box :
  ∃ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15 ∧ a^2 + b^2 = 61) ∧
  ∀ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15) → (a^2 + b^2 ≥ 61) :=
by {
  sorry
}

end min_possible_value_box_l1402_140293


namespace a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l1402_140256

theorem a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4 (a : ℝ) :
  (a < 2 → a^2 < 4) ∧ (a^2 < 4 → a < 2) :=
by
  -- Proof skipped
  sorry

end a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l1402_140256


namespace original_price_of_petrol_l1402_140282

theorem original_price_of_petrol (P : ℝ) (h : 0.9 * P * 190 / (0.9 * P) = 190 / P + 5) : P = 4.22 :=
by
  -- The proof goes here
  sorry

end original_price_of_petrol_l1402_140282


namespace rectangle_perimeter_l1402_140240

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l1402_140240


namespace total_value_of_gold_is_l1402_140230

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l1402_140230


namespace cost_of_each_candy_bar_l1402_140292

-- Definitions of the conditions
def initial_amount : ℕ := 20
def final_amount : ℕ := 12
def number_of_candy_bars : ℕ := 4

-- Statement of the proof problem: prove the cost of each candy bar
theorem cost_of_each_candy_bar :
  (initial_amount - final_amount) / number_of_candy_bars = 2 := by
  sorry

end cost_of_each_candy_bar_l1402_140292


namespace no_integers_abc_for_polynomial_divisible_by_9_l1402_140294

theorem no_integers_abc_for_polynomial_divisible_by_9 :
  ¬ ∃ (a b c : ℤ), ∀ x : ℤ, 9 ∣ (x + a) * (x + b) * (x + c) - x ^ 3 - 1 :=
by
  sorry

end no_integers_abc_for_polynomial_divisible_by_9_l1402_140294


namespace mrs_sheridan_fish_count_l1402_140232

/-
  Problem statement: 
  Prove that the total number of fish Mrs. Sheridan has now is 69, 
  given that she initially had 22 fish and she received 47 more from her sister.
-/

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let additional_fish : ℕ := 47
  initial_fish + additional_fish = 69 := by
sorry

end mrs_sheridan_fish_count_l1402_140232


namespace log21_requires_additional_information_l1402_140212

noncomputable def log3 : ℝ := 0.4771
noncomputable def log5 : ℝ := 0.6990

theorem log21_requires_additional_information
  (log3_given : log3 = 0.4771)
  (log5_given : log5 = 0.6990) :
  ¬ (∃ c₁ c₂ : ℝ, log21 = c₁ * log3 + c₂ * log5) :=
sorry

end log21_requires_additional_information_l1402_140212


namespace average_age_of_9_students_l1402_140216

theorem average_age_of_9_students
  (avg_20_students : ℝ)
  (n_20_students : ℕ)
  (avg_10_students : ℝ)
  (n_10_students : ℕ)
  (age_20th_student : ℝ)
  (total_age_20_students : ℝ := avg_20_students * n_20_students)
  (total_age_10_students : ℝ := avg_10_students * n_10_students)
  (total_age_9_students : ℝ := total_age_20_students - total_age_10_students - age_20th_student)
  (n_9_students : ℕ)
  (expected_avg_9_students : ℝ := total_age_9_students / n_9_students)
  (H1 : avg_20_students = 20)
  (H2 : n_20_students = 20)
  (H3 : avg_10_students = 24)
  (H4 : n_10_students = 10)
  (H5 : age_20th_student = 61)
  (H6 : n_9_students = 9) :
  expected_avg_9_students = 11 :=
sorry

end average_age_of_9_students_l1402_140216


namespace sum_coeff_eq_neg_two_l1402_140274

theorem sum_coeff_eq_neg_two (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (1 - 2*x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  sorry

end sum_coeff_eq_neg_two_l1402_140274


namespace ratio_is_correct_l1402_140258

-- Define the constants
def total_students : ℕ := 47
def current_students : ℕ := 6 * 3
def girls_bathroom : ℕ := 3
def new_groups : ℕ := 2 * 4
def foreign_exchange_students : ℕ := 3 * 3

-- The total number of missing students
def missing_students : ℕ := girls_bathroom + new_groups + foreign_exchange_students

-- The number of students who went to the canteen
def students_canteen : ℕ := total_students - current_students - missing_students

-- The ratio of students who went to the canteen to girls who went to the bathroom
def canteen_to_bathroom_ratio : ℕ × ℕ := (students_canteen, girls_bathroom)

theorem ratio_is_correct : canteen_to_bathroom_ratio = (3, 1) :=
by
  -- Proof goes here
  sorry

end ratio_is_correct_l1402_140258


namespace taxi_faster_than_truck_l1402_140250

noncomputable def truck_speed : ℝ := 2.1 / 1
noncomputable def taxi_speed : ℝ := 10.5 / 4

theorem taxi_faster_than_truck :
  taxi_speed / truck_speed = 1.25 :=
by
  sorry

end taxi_faster_than_truck_l1402_140250


namespace shaded_area_is_28_l1402_140222

theorem shaded_area_is_28 (A B : ℕ) (h1 : A = 64) (h2 : B = 28) : B = 28 := by
  sorry

end shaded_area_is_28_l1402_140222


namespace rectangle_area_l1402_140271

theorem rectangle_area (p q : ℝ) (x : ℝ) (h1 : x^2 + (2 * x)^2 = (p + q)^2) : 
    2 * x^2 = (2 * (p + q)^2) / 5 := 
sorry

end rectangle_area_l1402_140271


namespace kelly_initial_sony_games_l1402_140262

def nintendo_games : ℕ := 46
def sony_games_given_away : ℕ := 101
def sony_games_left : ℕ := 31

theorem kelly_initial_sony_games :
  sony_games_given_away + sony_games_left = 132 :=
by
  sorry

end kelly_initial_sony_games_l1402_140262


namespace original_balance_l1402_140246

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

theorem original_balance (decrease_percentage : ℝ) (current_balance : ℝ) (original_balance : ℝ) :
  decrease_percentage = 0.10 → current_balance = 90000 → 
  current_balance = (1 - decrease_percentage) * original_balance → 
  original_balance = 100000 := by
  sorry

end original_balance_l1402_140246


namespace factorize_negative_quadratic_l1402_140285

theorem factorize_negative_quadratic (x y : ℝ) : 
  -4 * x^2 + y^2 = (y - 2 * x) * (y + 2 * x) :=
by 
  sorry

end factorize_negative_quadratic_l1402_140285


namespace circle_area_conversion_l1402_140281

-- Define the given diameter
def diameter (d : ℝ) := d = 8

-- Define the radius calculation
def radius (r : ℝ) := r = 4

-- Define the formula for the area of the circle in square meters
def area_sq_m (A : ℝ) := A = 16 * Real.pi

-- Define the conversion factor from square meters to square centimeters
def conversion_factor := 10000

-- Define the expected area in square centimeters
def area_sq_cm (A : ℝ) := A = 160000 * Real.pi

-- The theorem to prove
theorem circle_area_conversion (d r A_cm : ℝ) (h1 : diameter d) (h2 : radius r) (h3 : area_sq_cm A_cm) :
  A_cm = 160000 * Real.pi :=
by
  sorry

end circle_area_conversion_l1402_140281


namespace fraction_distance_traveled_by_bus_l1402_140266

theorem fraction_distance_traveled_by_bus (D : ℝ) (hD : D = 105.00000000000003)
    (distance_by_foot : ℝ) (h_foot : distance_by_foot = (1 / 5) * D)
    (distance_by_car : ℝ) (h_car : distance_by_car = 14) :
    (D - (distance_by_foot + distance_by_car)) / D = 2 / 3 := by
  sorry

end fraction_distance_traveled_by_bus_l1402_140266


namespace a_4_eq_15_l1402_140218

noncomputable def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n + 1

theorem a_4_eq_15 : a 3 = 15 :=
by
  sorry

end a_4_eq_15_l1402_140218


namespace possible_values_of_a_l1402_140207

theorem possible_values_of_a (x y a : ℝ) (h1 : x + y = a) (h2 : x^3 + y^3 = a) (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 :=
by sorry

end possible_values_of_a_l1402_140207


namespace john_small_planks_l1402_140275

theorem john_small_planks (L S : ℕ) (h1 : L = 12) (h2 : L + S = 29) : S = 17 :=
by {
  sorry
}

end john_small_planks_l1402_140275


namespace problem_statement_l1402_140287

theorem problem_statement (M N : ℕ) 
  (hM : M = 2020 / 5) 
  (hN : N = 2020 / 20) : 10 * M / N = 40 := 
by
  sorry

end problem_statement_l1402_140287


namespace ball_third_bounce_distance_is_correct_l1402_140235

noncomputable def total_distance_third_bounce (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + 2 * (initial_height * rebound_ratio) + 2 * (initial_height * rebound_ratio^2)

theorem ball_third_bounce_distance_is_correct : 
  total_distance_third_bounce 80 (2/3) = 257.78 := 
by
  sorry

end ball_third_bounce_distance_is_correct_l1402_140235


namespace polynomial_roots_bounds_l1402_140214

theorem polynomial_roots_bounds (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ (x1^4 + 3*p*x1^3 + x1^2 + 3*p*x1 + 1 = 0) ∧ (x2^4 + 3*p*x2^3 + x2^2 + 3*p*x2 + 1 = 0)) ↔ p ∈ Set.Iio (1 / 4) := by
sorry

end polynomial_roots_bounds_l1402_140214


namespace range_of_a_l1402_140223

open Real

theorem range_of_a (k a : ℝ) : 
  (∀ k : ℝ, ∀ x y : ℝ, k * x - y - k + 2 = 0 → x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-7 : ℝ) (-2) ∪ Set.Ioi 1) := 
sorry

end range_of_a_l1402_140223


namespace SomeAthletesNotHonorSociety_l1402_140206

variable (Athletes HonorSociety : Type)
variable (Discipline : Athletes → Prop)
variable (isMember : Athletes → HonorSociety → Prop)

-- Some athletes are not disciplined
axiom AthletesNotDisciplined : ∃ a : Athletes, ¬Discipline a

-- All members of the honor society are disciplined
axiom AllHonorSocietyDisciplined : ∀ h : HonorSociety, ∀ a : Athletes, isMember a h → Discipline a

-- The theorem to be proved
theorem SomeAthletesNotHonorSociety : ∃ a : Athletes, ∀ h : HonorSociety, ¬isMember a h :=
  sorry

end SomeAthletesNotHonorSociety_l1402_140206


namespace total_alphabets_written_l1402_140264

-- Define the number of vowels and the number of times each is written
def num_vowels : ℕ := 5
def repetitions : ℕ := 4

-- The theorem stating the total number of alphabets written on the board
theorem total_alphabets_written : num_vowels * repetitions = 20 := by
  sorry

end total_alphabets_written_l1402_140264


namespace find_circle_radius_l1402_140253

noncomputable def circle_radius (x y : ℝ) : ℝ :=
  (x - 1) ^ 2 + (y + 2) ^ 2

theorem find_circle_radius :
  (∀ x y : ℝ, 25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 → circle_radius x y = 0) → radius = 0 :=
sorry

end find_circle_radius_l1402_140253


namespace player_weekly_earnings_l1402_140288

structure Performance :=
  (points assists rebounds steals : ℕ)

def base_pay (avg_points : ℕ) : ℕ :=
  if avg_points >= 30 then 10000 else 8000

def assists_bonus (total_assists : ℕ) : ℕ :=
  if total_assists >= 20 then 5000
  else if total_assists >= 10 then 3000
  else 1000

def rebounds_bonus (total_rebounds : ℕ) : ℕ :=
  if total_rebounds >= 40 then 5000
  else if total_rebounds >= 20 then 3000
  else 1000

def steals_bonus (total_steals : ℕ) : ℕ :=
  if total_steals >= 15 then 5000
  else if total_steals >= 5 then 3000
  else 1000

def total_payment (performances : List Performance) : ℕ :=
  let total_points := performances.foldl (λ acc p => acc + p.points) 0
  let total_assists := performances.foldl (λ acc p => acc + p.assists) 0
  let total_rebounds := performances.foldl (λ acc p => acc + p.rebounds) 0
  let total_steals := performances.foldl (λ acc p => acc + p.steals) 0
  let avg_points := total_points / performances.length
  base_pay avg_points + assists_bonus total_assists + rebounds_bonus total_rebounds + steals_bonus total_steals
  
theorem player_weekly_earnings :
  let performances := [
    Performance.mk 30 5 7 3,
    Performance.mk 28 6 5 2,
    Performance.mk 32 4 9 1,
    Performance.mk 34 3 11 2,
    Performance.mk 26 2 8 3
  ]
  total_payment performances = 23000 := by 
    sorry

end player_weekly_earnings_l1402_140288


namespace find_divisor_l1402_140276

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 34) / y = 2) : y = 10 :=
by
  sorry

end find_divisor_l1402_140276


namespace bank_robbery_participants_l1402_140211

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l1402_140211


namespace det_condition_l1402_140277

theorem det_condition (a b c d : ℤ) 
    (h_exists : ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n) :
    |a * d - b * c| = 1 :=
sorry

end det_condition_l1402_140277


namespace original_denominator_l1402_140228

theorem original_denominator (d : ℤ) (h1 : 5 = d + 3) : d = 12 := 
by 
  sorry

end original_denominator_l1402_140228


namespace equation_1_solution_equation_2_solution_l1402_140260

theorem equation_1_solution (x : ℝ) :
  6 * (x - 2 / 3) - (x + 7) = 11 → x = 22 / 5 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

theorem equation_2_solution (x : ℝ) :
  (2 * x - 1) / 3 = (2 * x + 1) / 6 - 2 → x = -9 / 2 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

end equation_1_solution_equation_2_solution_l1402_140260


namespace regular_octahedron_vertices_count_l1402_140267

def regular_octahedron_faces := 8
def regular_octahedron_edges := 12
def regular_octahedron_faces_shape := "equilateral triangle"
def regular_octahedron_vertices_meet := 4

theorem regular_octahedron_vertices_count :
  ∀ (F E V : ℕ),
    F = regular_octahedron_faces →
    E = regular_octahedron_edges →
    (∀ (v : ℕ), v = regular_octahedron_vertices_meet) →
    V = 6 :=
by
  intros F E V hF hE hV
  sorry

end regular_octahedron_vertices_count_l1402_140267


namespace problem1_problem2_problem3_problem4_l1402_140221

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2023 + (π - 3.14) ^ 0 - ((-1 / 2 : ℚ) ^ (-2 : ℤ)) = -4 := by
  sorry

-- Problem 2
theorem problem2 (x : ℚ) : 
  ((1 / 4 * x^4 + 2 * x^3 - 4 * x^2) / (-(2 * x))^2) = (1 / 16 * x^2 + 1 / 2 * x - 1) := by
  sorry

-- Problem 3
theorem problem3 (x y : ℚ) : 
  (2 * x + y + 1) * (2 * x + y - 1) = 4 * x^2 + 4 * x * y + y^2 - 1 := by
  sorry

-- Problem 4
theorem problem4 (x : ℚ) : 
  (2 * x + 3) * (2 * x - 3) - (2 * x - 1)^2 = 4 * x - 10 := by
  sorry

end problem1_problem2_problem3_problem4_l1402_140221


namespace turtles_still_on_sand_l1402_140209

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l1402_140209


namespace players_started_first_half_l1402_140224

variable (total_players : Nat)
variable (first_half_substitutions : Nat)
variable (second_half_substitutions : Nat)
variable (players_not_playing : Nat)

theorem players_started_first_half :
  total_players = 24 →
  first_half_substitutions = 2 →
  second_half_substitutions = 2 * first_half_substitutions →
  players_not_playing = 7 →
  let total_substitutions := first_half_substitutions + second_half_substitutions 
  let players_played := total_players - players_not_playing
  ∃ S, S + total_substitutions = players_played ∧ S = 11 := 
by
  sorry

end players_started_first_half_l1402_140224


namespace total_cost_is_46_8_l1402_140255

def price_pork : ℝ := 6
def price_chicken : ℝ := price_pork - 2
def price_beef : ℝ := price_chicken + 4
def price_lamb : ℝ := price_pork + 3

def quantity_chicken : ℝ := 3.5
def quantity_pork : ℝ := 1.2
def quantity_beef : ℝ := 2.3
def quantity_lamb : ℝ := 0.8

def total_cost : ℝ :=
    (quantity_chicken * price_chicken) +
    (quantity_pork * price_pork) +
    (quantity_beef * price_beef) +
    (quantity_lamb * price_lamb)

theorem total_cost_is_46_8 : total_cost = 46.8 :=
by
  sorry

end total_cost_is_46_8_l1402_140255


namespace smallest_white_erasers_l1402_140217

def total_erasers (n : ℕ) (pink : ℕ) (orange : ℕ) (purple : ℕ) (white : ℕ) : Prop :=
  pink = n / 5 ∧ orange = n / 6 ∧ purple = 10 ∧ white = n - (pink + orange + purple)

theorem smallest_white_erasers : ∃ n : ℕ, ∃ pink : ℕ, ∃ orange : ℕ, ∃ purple : ℕ, ∃ white : ℕ,
  total_erasers n pink orange purple white ∧ white = 9 := sorry

end smallest_white_erasers_l1402_140217


namespace buffy_less_brittany_by_40_seconds_l1402_140229

/-
The following statement proves that Buffy's breath-holding time was 40 seconds less than Brittany's, 
given the initial conditions about their breath-holding times.
-/
theorem buffy_less_brittany_by_40_seconds 
  (kelly_time : ℕ) 
  (brittany_time : ℕ) 
  (buffy_time : ℕ) 
  (h_kelly : kelly_time = 180) 
  (h_brittany : brittany_time = kelly_time - 20) 
  (h_buffy : buffy_time = 120)
  :
  brittany_time - buffy_time = 40 :=
sorry

end buffy_less_brittany_by_40_seconds_l1402_140229


namespace abs_eq_four_l1402_140234

theorem abs_eq_four (x : ℝ) (h : |x| = 4) : x = 4 ∨ x = -4 :=
by
  sorry

end abs_eq_four_l1402_140234


namespace solution_unique_l1402_140245

def satisfies_equation (x y : ℝ) : Prop :=
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1 / 3

theorem solution_unique (x y : ℝ) :
  satisfies_equation x y ↔ x = 7 + 1/3 ∧ y = 8 - 1/3 :=
by {
  sorry
}

end solution_unique_l1402_140245


namespace max_x_plus_y_l1402_140289

theorem max_x_plus_y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 9) (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7 / 3 :=
sorry

end max_x_plus_y_l1402_140289


namespace minimum_value_of_E_l1402_140261

theorem minimum_value_of_E (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 12) : |E| = 11 :=
sorry

end minimum_value_of_E_l1402_140261


namespace buyers_cake_and_muffin_l1402_140249

theorem buyers_cake_and_muffin (total_buyers cake_buyers muffin_buyers neither_prob : ℕ) :
  total_buyers = 100 →
  cake_buyers = 50 →
  muffin_buyers = 40 →
  neither_prob = 26 →
  (cake_buyers + muffin_buyers - neither_prob) = 74 →
  90 - cake_buyers - muffin_buyers = neither_prob :=
by
  sorry

end buyers_cake_and_muffin_l1402_140249


namespace terminal_side_angle_is_in_fourth_quadrant_l1402_140279

variable (α : ℝ)
variable (tan_alpha cos_alpha : ℝ)

-- Given conditions
def in_second_quadrant := tan_alpha < 0 ∧ cos_alpha > 0

-- Conclusion to prove
theorem terminal_side_angle_is_in_fourth_quadrant 
  (h : in_second_quadrant tan_alpha cos_alpha) : 
  -- Here we model the "fourth quadrant" in a proof-statement context:
  true := sorry

end terminal_side_angle_is_in_fourth_quadrant_l1402_140279


namespace abs_sum_inequality_l1402_140283

theorem abs_sum_inequality (x : ℝ) : (|x - 2| + |x + 3| < 7) ↔ (-6 < x ∧ x < 3) :=
sorry

end abs_sum_inequality_l1402_140283


namespace x_intercept_of_line_l1402_140273

theorem x_intercept_of_line : ∃ x : ℚ, (6 * x, 0) = (35 / 6, 0) :=
by
  use 35 / 6
  sorry

end x_intercept_of_line_l1402_140273


namespace distance_ran_each_morning_l1402_140208

-- Definitions based on conditions
def days_ran : ℕ := 3
def total_distance : ℕ := 2700

-- The goal is to prove the distance ran each morning
theorem distance_ran_each_morning : total_distance / days_ran = 900 :=
by
  sorry

end distance_ran_each_morning_l1402_140208


namespace squared_expression_l1402_140263

variable {x y : ℝ}

theorem squared_expression (x y : ℝ) : (-3 * x^2 * y)^2 = 9 * x^4 * y^2 :=
  by
  sorry

end squared_expression_l1402_140263


namespace find_room_height_l1402_140297

theorem find_room_height (l b d : ℕ) (h : ℕ) (hl : l = 12) (hb : b = 8) (hd : d = 17) :
  d = Int.sqrt (l^2 + b^2 + h^2) → h = 9 :=
by
  sorry

end find_room_height_l1402_140297
