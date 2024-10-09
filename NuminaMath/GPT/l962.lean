import Mathlib

namespace kite_area_l962_96203

theorem kite_area {length height : ℕ} (h_length : length = 8) (h_height : height = 10): 
  2 * (1/2 * (length * 2) * (height * 2 / 2)) = 160 :=
by
  rw [h_length, h_height]
  norm_num
  sorry

end kite_area_l962_96203


namespace value_of_expression_l962_96238

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l962_96238


namespace part_a_l962_96206

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l962_96206


namespace smallest_digit_divisible_by_11_l962_96296

theorem smallest_digit_divisible_by_11 :
  ∃ (d : ℕ), d < 10 ∧ ∀ n : ℕ, (n + 45000 + 1000 + 457 + d) % 11 = 0 → d = 5 :=
by {
  sorry
}

end smallest_digit_divisible_by_11_l962_96296


namespace animal_stickers_l962_96260

theorem animal_stickers {flower stickers total_stickers animal_stickers : ℕ} 
  (h_flower_stickers : flower = 8) 
  (h_total_stickers : total_stickers = 14)
  (h_total_eq : total_stickers = flower + animal_stickers) : 
  animal_stickers = 6 :=
by
  sorry

end animal_stickers_l962_96260


namespace julia_average_speed_l962_96288

-- Define the conditions as constants
def total_distance : ℝ := 28
def total_time : ℝ := 4

-- Define the theorem stating Julia's average speed
theorem julia_average_speed : total_distance / total_time = 7 := by
  sorry

end julia_average_speed_l962_96288


namespace arithmetic_sequence_eighth_term_l962_96261

theorem arithmetic_sequence_eighth_term (a d : ℚ) 
  (h1 : 6 * a + 15 * d = 21) 
  (h2 : a + 6 * d = 8) : 
  a + 7 * d = 9 + 2/7 :=
by
  sorry

end arithmetic_sequence_eighth_term_l962_96261


namespace smallest_mu_real_number_l962_96232

theorem smallest_mu_real_number (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≤ ab + (3/2) * bc + cd :=
sorry

end smallest_mu_real_number_l962_96232


namespace min_value_inequality_l962_96230

theorem min_value_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 9)
  : (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 :=
by
  sorry

end min_value_inequality_l962_96230


namespace measure_of_C_and_max_perimeter_l962_96271

noncomputable def triangle_C_and_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (2 * Real.sin A + 2 * Real.sin B + c ≤ 2 + Real.sqrt 3)

-- Now the Lean theorem statement
theorem measure_of_C_and_max_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) :
  triangle_C_and_perimeter a b c A B C hABC hc :=
by 
  sorry

end measure_of_C_and_max_perimeter_l962_96271


namespace second_neighbor_brought_less_l962_96286

theorem second_neighbor_brought_less (n1 n2 : ℕ) (htotal : ℕ) (h1 : n1 = 75) (h_total : n1 + n2 = 125) :
  n1 - n2 = 25 :=
by
  sorry

end second_neighbor_brought_less_l962_96286


namespace region_area_correct_l962_96219

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l962_96219


namespace speed_down_l962_96269

theorem speed_down {u avg_speed d v : ℝ} (hu : u = 18) (havg : avg_speed = 20.571428571428573) (hv : 2 * d / ((d / u) + (d / v)) = avg_speed) : v = 24 :=
by
  have h1 : 20.571428571428573 = 20.571428571428573 := rfl
  have h2 : 18 = 18 := rfl
  sorry

end speed_down_l962_96269


namespace central_angle_probability_l962_96278

theorem central_angle_probability (A : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : (x / 360) * A / A = 1 / 8) : 
  x = 45 := 
by
  sorry

end central_angle_probability_l962_96278


namespace bus_fare_max_profit_passenger_count_change_l962_96281

noncomputable def demand (p : ℝ) : ℝ := 3000 - 20 * p
noncomputable def train_fare : ℝ := 10
noncomputable def train_capacity : ℝ := 1000
noncomputable def bus_cost (y : ℝ) : ℝ := y + 5

theorem bus_fare_max_profit : 
  ∃ (p_bus : ℝ), 
  p_bus = 50.5 ∧ 
  p_bus * (demand p_bus - train_capacity) - bus_cost (demand p_bus - train_capacity) = 
  p_bus * (demand p_bus - train_capacity) - (demand p_bus - train_capacity + 5) := 
sorry

theorem passenger_count_change :
  (demand train_fare - train_capacity) + train_capacity - demand 75.5 = 500 :=
sorry

end bus_fare_max_profit_passenger_count_change_l962_96281


namespace exists_fg_pairs_l962_96248

theorem exists_fg_pairs (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), (∀ x : ℤ, f (g x) = x + a) ∧ (∀ x : ℤ, g (f x) = x + b)) ↔ (a = b ∨ a = -b) := 
sorry

end exists_fg_pairs_l962_96248


namespace determine_l_l962_96208

theorem determine_l :
  ∃ l : ℤ, (2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997) ∧ l = -1 :=
by
  sorry

end determine_l_l962_96208


namespace sqrt_floor_eq_sqrt_sqrt_floor_l962_96211

theorem sqrt_floor_eq_sqrt_sqrt_floor (a : ℝ) (h : a > 1) :
  Int.floor (Real.sqrt (Int.floor (Real.sqrt a))) = Int.floor (Real.sqrt (Real.sqrt a)) :=
sorry

end sqrt_floor_eq_sqrt_sqrt_floor_l962_96211


namespace cuboid_volume_l962_96274

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by
  sorry

end cuboid_volume_l962_96274


namespace find_parallelogram_height_l962_96212

def parallelogram_height (base area : ℕ) : ℕ := area / base

theorem find_parallelogram_height :
  parallelogram_height 32 448 = 14 :=
by {
  sorry
}

end find_parallelogram_height_l962_96212


namespace task1_task2_task3_task4_l962_96217

-- Definitions of the given conditions
def cost_price : ℝ := 16
def selling_price_range (x : ℝ) : Prop := 16 ≤ x ∧ x ≤ 48
def init_selling_price : ℝ := 20
def init_sales_volume : ℝ := 360
def decreasing_sales_rate : ℝ := 10
def daily_sales_vol (x : ℝ) : ℝ := 360 - 10 * (x - 20)
def daily_total_profit (x : ℝ) (y : ℝ) : ℝ := y * (x - cost_price)

-- Proof task (1)
theorem task1 : daily_sales_vol 25 = 310 ∧ daily_total_profit 25 (daily_sales_vol 25) = 2790 := 
by 
    -- Your code here
    sorry

-- Proof task (2)
theorem task2 : ∀ x, daily_sales_vol x = -10 * x + 560 := 
by 
    -- Your code here
    sorry

-- Proof task (3)
theorem task3 : ∀ x, 
    W = (x - 16) * (daily_sales_vol x) 
    ∧ W = -10 * x ^ 2 + 720 * x - 8960 
    ∧ (∃ x, -10 * x ^ 2 + 720 * x - 8960 = 4000 ∧ selling_price_range x) := 
by 
    -- Your code here 
    sorry

-- Proof task (4)
theorem task4 : ∃ x, 
    -10 * (x - 36) ^ 2 + 4000 = 3000 
    ∧ selling_price_range x 
    ∧ (x = 26 ∨ x = 46) := 
by 
    -- Your code here 
    sorry

end task1_task2_task3_task4_l962_96217


namespace triangle_inequality_proof_l962_96284

variable (a b c : ℝ)

-- Condition that a, b, c are side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Theorem stating the required inequality and the condition for equality
theorem triangle_inequality_proof :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c ∧ c = a) :=
sorry

end triangle_inequality_proof_l962_96284


namespace positive_real_triangle_inequality_l962_96294

theorem positive_real_triangle_inequality
    (a b c : ℝ)
    (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h : 5 * a * b * c > a^3 + b^3 + c^3) :
    a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end positive_real_triangle_inequality_l962_96294


namespace bus_stops_for_45_minutes_per_hour_l962_96244

-- Define the conditions
def speed_excluding_stoppages : ℝ := 48 -- in km/hr
def speed_including_stoppages : ℝ := 12 -- in km/hr

-- Define the statement to be proven
theorem bus_stops_for_45_minutes_per_hour :
  let speed_reduction := speed_excluding_stoppages - speed_including_stoppages
  let time_stopped : ℝ := (speed_reduction / speed_excluding_stoppages) * 60
  time_stopped = 45 :=
by
  sorry

end bus_stops_for_45_minutes_per_hour_l962_96244


namespace average_math_score_of_class_l962_96240

theorem average_math_score_of_class (n : ℕ) (jimin_score jung_score avg_others : ℕ) 
  (h1 : n = 40) 
  (h2 : jimin_score = 98) 
  (h3 : jung_score = 100) 
  (h4 : avg_others = 79) : 
  (38 * avg_others + jimin_score + jung_score) / n = 80 :=
by sorry

end average_math_score_of_class_l962_96240


namespace tangent_line_eqn_c_range_l962_96291

noncomputable def f (x : ℝ) := 3 * x * Real.log x + 2

theorem tangent_line_eqn :
  let k := 3 
  let x₀ := 1 
  let y₀ := f x₀ 
  y = k * (x - x₀) + y₀ ↔ 3*x - y - 1 = 0 :=
by sorry

theorem c_range (x : ℝ) (hx : 1 < x) (c : ℝ) :
  f x ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 :=
by sorry

end tangent_line_eqn_c_range_l962_96291


namespace line_slope_angle_y_intercept_l962_96224

theorem line_slope_angle_y_intercept :
  ∀ (x y : ℝ), x - y - 1 = 0 → 
    (∃ k b : ℝ, y = x - 1 ∧ k = 1 ∧ b = -1 ∧ θ = 45 ∧ θ = Real.arctan k) := 
    by
      sorry

end line_slope_angle_y_intercept_l962_96224


namespace sara_total_cents_l962_96246

def number_of_quarters : ℕ := 11
def value_per_quarter : ℕ := 25

theorem sara_total_cents : number_of_quarters * value_per_quarter = 275 := by
  sorry

end sara_total_cents_l962_96246


namespace distance_AC_l962_96276

theorem distance_AC (t_Eddy t_Freddy : ℕ) (d_AB : ℝ) (speed_ratio : ℝ) : 
  t_Eddy = 3 ∧ t_Freddy = 4 ∧ d_AB = 510 ∧ speed_ratio = 2.2666666666666666 → 
  ∃ d_AC : ℝ, d_AC = 300 :=
by 
  intros h
  obtain ⟨hE, hF, hD, hR⟩ := h
  -- Declare velocities
  let v_Eddy : ℝ := d_AB / t_Eddy
  let v_Freddy : ℝ := v_Eddy / speed_ratio
  let d_AC : ℝ := v_Freddy * t_Freddy
  -- Prove the distance
  use d_AC
  sorry

end distance_AC_l962_96276


namespace complex_multiplication_l962_96221

-- Define i such that i^2 = -1
def i : ℂ := Complex.I

theorem complex_multiplication : (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i := by
  sorry

end complex_multiplication_l962_96221


namespace pete_and_ray_spent_200_cents_l962_96257

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l962_96257


namespace ferris_break_length_l962_96293

-- Definitions of the given conditions
def audrey_work_rate := 1 / 4  -- Audrey completes 1/4 of the job per hour
def ferris_work_rate := 1 / 3  -- Ferris completes 1/3 of the job per hour
def total_work_time := 2       -- They worked together for 2 hours
def num_breaks := 6            -- Ferris took 6 breaks during the work period

-- The theorem to prove the length of each break Ferris took
theorem ferris_break_length (break_length : ℝ) :
  (audrey_work_rate * total_work_time) + 
  (ferris_work_rate * (total_work_time - (break_length / 60) * num_breaks)) = 1 →
  break_length = 2.5 :=
by
  sorry

end ferris_break_length_l962_96293


namespace quadratic_nonneg_for_all_t_l962_96222

theorem quadratic_nonneg_for_all_t (x y : ℝ) : 
  (y ≤ x + 1) → (y ≥ -x - 1) → (x ≥ y^2 / 4) → (∀ (t : ℝ), (|t| ≤ 1) → t^2 + y * t + x ≥ 0) :=
by
  intro h1 h2 h3 t ht
  sorry

end quadratic_nonneg_for_all_t_l962_96222


namespace triangleProblem_correct_l962_96265

noncomputable def triangleProblem : Prop :=
  ∃ (a b c A B C : ℝ),
    A = 60 * Real.pi / 180 ∧
    b = 1 ∧
    (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    Real.cos A = 1 / 2 ∧
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    (a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧
    (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3

theorem triangleProblem_correct : triangleProblem :=
  sorry

end triangleProblem_correct_l962_96265


namespace quadratic_increasing_l962_96268

theorem quadratic_increasing (x : ℝ) (hx : x > 1) : ∃ y : ℝ, y = (x-1)^2 + 1 ∧ ∀ (x₁ x₂ : ℝ), x₁ > x ∧ x₂ > x₁ → (x₁ - 1)^2 + 1 < (x₂ - 1)^2 + 1 := by
  sorry

end quadratic_increasing_l962_96268


namespace exists_integers_m_n_l962_96216

theorem exists_integers_m_n (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (m n : ℤ), (m * x + n * y > 0) ∧ (n * x + m * y < 0) :=
sorry

end exists_integers_m_n_l962_96216


namespace students_suggested_pasta_l962_96251

-- Define the conditions as variables in Lean
variable (total_students : ℕ := 470)
variable (suggested_mashed_potatoes : ℕ := 230)
variable (suggested_bacon : ℕ := 140)

-- The problem statement to prove
theorem students_suggested_pasta : 
  total_students - (suggested_mashed_potatoes + suggested_bacon) = 100 := by
  sorry

end students_suggested_pasta_l962_96251


namespace unique_function_satisfying_condition_l962_96215

theorem unique_function_satisfying_condition :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) ↔ f = id :=
sorry

end unique_function_satisfying_condition_l962_96215


namespace find_toonies_l962_96277

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l962_96277


namespace tan_sum_property_l962_96204

theorem tan_sum_property (t23 t37 : ℝ) (h1 : 23 + 37 = 60) (h2 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3) :
  Real.tan (23 * Real.pi / 180) + Real.tan (37 * Real.pi / 180) + Real.sqrt 3 * Real.tan (23 * Real.pi / 180) * Real.tan (37 * Real.pi / 180) = Real.sqrt 3 :=
sorry

end tan_sum_property_l962_96204


namespace inequality_b_does_not_hold_l962_96220

theorem inequality_b_does_not_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬(a + d > b + c) ↔ a + d ≤ b + c :=
by
  -- We only need the statement, so we add sorry at the end
  sorry

end inequality_b_does_not_hold_l962_96220


namespace find_replaced_man_weight_l962_96280

variable (n : ℕ) (new_weight old_avg_weight : ℝ) (weight_inc : ℝ) (W : ℝ)

theorem find_replaced_man_weight 
  (h1 : n = 8) 
  (h2 : new_weight = 68) 
  (h3 : weight_inc = 1) 
  (h4 : 8 * (old_avg_weight + 1) = 8 * old_avg_weight + (new_weight - W)) 
  : W = 60 :=
by
  sorry

end find_replaced_man_weight_l962_96280


namespace equation_1_solution_set_equation_2_solution_set_l962_96254

open Real

theorem equation_1_solution_set (x : ℝ) : x^2 - 4 * x - 8 = 0 ↔ (x = 2 * sqrt 3 + 2 ∨ x = -2 * sqrt 3 + 2) :=
by sorry

theorem equation_2_solution_set (x : ℝ) : 3 * x - 6 = x * (x - 2) ↔ (x = 2 ∨ x = 3) :=
by sorry

end equation_1_solution_set_equation_2_solution_set_l962_96254


namespace range_of_a_l962_96298

/-- Proposition p: ∀ x ∈ [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

/-- Proposition q: ∃ x₀ ∈ ℝ, x + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop := 
  ∃ x₀ : ℝ, ∃ x : ℝ, x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : prop_p a ∧ prop_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l962_96298


namespace eval_expression_l962_96283

theorem eval_expression : 3 ^ 2 - (4 * 2) = 1 :=
by
  sorry

end eval_expression_l962_96283


namespace trout_split_equally_l962_96263

-- Conditions: Nancy and Joan caught 18 trout and split them equally
def total_trout : ℕ := 18
def equal_split (n : ℕ) : ℕ := n / 2

-- Theorem: Prove that if they equally split the trout, each person will get 9 trout.
theorem trout_split_equally : equal_split total_trout = 9 :=
by 
  -- Placeholder for the actual proof
  sorry

end trout_split_equally_l962_96263


namespace candy_store_price_per_pound_fudge_l962_96202

theorem candy_store_price_per_pound_fudge 
  (fudge_pounds : ℕ)
  (truffles_dozen : ℕ)
  (truffles_price_each : ℝ)
  (pretzels_dozen : ℕ)
  (pretzels_price_each : ℝ)
  (total_revenue : ℝ) 
  (truffles_total : ℕ := truffles_dozen * 12)
  (pretzels_total : ℕ := pretzels_dozen * 12)
  (truffles_revenue : ℝ := truffles_total * truffles_price_each)
  (pretzels_revenue : ℝ := pretzels_total * pretzels_price_each)
  (fudge_revenue : ℝ := total_revenue - (truffles_revenue + pretzels_revenue))
  (fudge_price_per_pound : ℝ := fudge_revenue / fudge_pounds) :
  fudge_pounds = 20 →
  truffles_dozen = 5 →
  truffles_price_each = 1.50 →
  pretzels_dozen = 3 →
  pretzels_price_each = 2.00 →
  total_revenue = 212 →
  fudge_price_per_pound = 2.5 :=
by 
  sorry

end candy_store_price_per_pound_fudge_l962_96202


namespace necklace_cost_l962_96262

theorem necklace_cost (total_savings earrings_cost remaining_savings: ℕ) 
                      (h1: total_savings = 80) 
                      (h2: earrings_cost = 23) 
                      (h3: remaining_savings = 9) : 
   total_savings - earrings_cost - remaining_savings = 48 :=
by
  sorry

end necklace_cost_l962_96262


namespace radiator_water_fraction_l962_96242

noncomputable def fraction_of_water_after_replacements (initial_water : ℚ) (initial_antifreeze : ℚ) (removal_fraction : ℚ)
  (num_replacements : ℕ) : ℚ :=
  initial_water * (removal_fraction ^ num_replacements)

theorem radiator_water_fraction :
  let initial_water := 10
  let initial_antifreeze := 10
  let total_volume := 20
  let removal_volume := 5
  let removal_fraction := 3 / 4
  let num_replacements := 4
  fraction_of_water_after_replacements initial_water initial_antifreeze removal_fraction num_replacements / total_volume = 0.158 := 
sorry

end radiator_water_fraction_l962_96242


namespace total_surface_area_correct_l962_96256

-- Defining the dimensions of the rectangular solid
def length := 10
def width := 9
def depth := 6

-- Definition of the total surface area of a rectangular solid
def surface_area (l w d : ℕ) := 2 * (l * w + w * d + l * d)

-- Proposition that the total surface area for the given dimensions is 408 square meters
theorem total_surface_area_correct : surface_area length width depth = 408 := 
by
  sorry

end total_surface_area_correct_l962_96256


namespace compute_value_of_fractions_l962_96210

theorem compute_value_of_fractions (a b c : ℝ) 
  (h1 : (ac / (a + b)) + (ba / (b + c)) + (cb / (c + a)) = 0)
  (h2 : (bc / (a + b)) + (ca / (b + c)) + (ab / (c + a)) = 1) :
  (b / (a + b)) + (c / (b + c)) + (a / (c + a)) = 5 / 2 :=
sorry

end compute_value_of_fractions_l962_96210


namespace trader_loss_percent_l962_96279

noncomputable def CP1 : ℝ := 325475 / 1.13
noncomputable def CP2 : ℝ := 325475 / 0.87
noncomputable def TCP : ℝ := CP1 + CP2
noncomputable def TSP : ℝ := 325475 * 2
noncomputable def profit_or_loss : ℝ := TSP - TCP
noncomputable def profit_or_loss_percent : ℝ := (profit_or_loss / TCP) * 100

theorem trader_loss_percent : profit_or_loss_percent = -1.684 := by 
  sorry

end trader_loss_percent_l962_96279


namespace suzanna_history_book_pages_l962_96250

theorem suzanna_history_book_pages (H G M S : ℕ) 
  (h_geography : G = H + 70)
  (h_math : M = (1 / 2) * (H + H + 70))
  (h_science : S = 2 * H)
  (h_total : H + G + M + S = 905) : 
  H = 160 := 
by
  sorry

end suzanna_history_book_pages_l962_96250


namespace bottles_needed_to_fill_large_bottle_l962_96207

def medium_bottle_ml : ℕ := 150
def large_bottle_ml : ℕ := 1200

theorem bottles_needed_to_fill_large_bottle : large_bottle_ml / medium_bottle_ml = 8 :=
by
  sorry

end bottles_needed_to_fill_large_bottle_l962_96207


namespace y_intercept_is_2_l962_96245

def equation_of_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def point_P : ℝ × ℝ := (-1, 1)

def y_intercept_of_tangent_line (m c x y : ℝ) : Prop :=
  equation_of_circle x y ∧
  ((y = m * x + c) ∧ (point_P.1, point_P.2) ∈ {(x, y) | y = m * x + c})

theorem y_intercept_is_2 :
  ∃ m c : ℝ, y_intercept_of_tangent_line m c 0 2 :=
sorry

end y_intercept_is_2_l962_96245


namespace solution_exists_l962_96253

def operation (a b : ℚ) : ℚ :=
if a ≥ b then a^2 * b else a * b^2

theorem solution_exists (m : ℚ) (h : operation 3 m = 48) : m = 4 := by
  sorry

end solution_exists_l962_96253


namespace symmetric_line_eq_l962_96297

theorem symmetric_line_eq (x y : ℝ) (h : 3 * x + 4 * y + 5 = 0) : 3 * x - 4 * y + 5 = 0 :=
sorry

end symmetric_line_eq_l962_96297


namespace preimage_of_4_neg_2_eq_1_3_l962_96227

def mapping (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem preimage_of_4_neg_2_eq_1_3 : ∃ x y : ℝ, mapping x y = (4, -2) ∧ (x = 1) ∧ (y = 3) :=
by 
  sorry

end preimage_of_4_neg_2_eq_1_3_l962_96227


namespace unique_solution_m_l962_96287

theorem unique_solution_m (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ (∀ y₁ y₂ : ℝ, 3 * y₁^2 - 6 * y₂ + m = 0 → y₁ = y₂)) → m = 3 :=
by
  sorry

end unique_solution_m_l962_96287


namespace factorization_x12_minus_729_l962_96295

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l962_96295


namespace page_number_counted_twice_l962_96231

theorem page_number_counted_twice {n x : ℕ} (h₁ : n = 70) (h₂ : x > 0) (h₃ : x ≤ n) (h₄ : 2550 = n * (n + 1) / 2 + x) : x = 65 :=
by {
  sorry
}

end page_number_counted_twice_l962_96231


namespace money_leftover_is_90_l962_96275

-- Define constants and given conditions.
def jars_quarters : ℕ := 4
def quarters_per_jar : ℕ := 160
def jars_dimes : ℕ := 4
def dimes_per_jar : ℕ := 300
def jars_nickels : ℕ := 2
def nickels_per_jar : ℕ := 500

def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05

def bike_cost : ℝ := 240
def total_quarters := jars_quarters * quarters_per_jar
def total_dimes := jars_dimes * dimes_per_jar
def total_nickels := jars_nickels * nickels_per_jar

-- Calculate the total money Jenn has in quarters, dimes, and nickels.
def total_value_quarters : ℝ := total_quarters * value_per_quarter
def total_value_dimes : ℝ := total_dimes * value_per_dime
def total_value_nickels : ℝ := total_nickels * value_per_nickel

def total_money : ℝ := total_value_quarters + total_value_dimes + total_value_nickels

-- Calculate the money left after buying the bike.
def money_left : ℝ := total_money - bike_cost

-- Prove that the amount of money left is precisely $90.
theorem money_leftover_is_90 : money_left = 90 :=
by
  -- Placeholder for the proof
  sorry

end money_leftover_is_90_l962_96275


namespace inverse_proportion_order_l962_96264

theorem inverse_proportion_order (k : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : k > 0) 
  (ha : y1 = k / (-3)) 
  (hb : y2 = k / (-2)) 
  (hc : y3 = k / 2) : 
  y2 < y1 ∧ y1 < y3 := 
sorry

end inverse_proportion_order_l962_96264


namespace intersection_compl_A_compl_B_l962_96282

open Set

variable (x y : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}

theorem intersection_compl_A_compl_B (U A B : Set ℝ) (hU : U = univ) (hA : A = {x | -1 < x ∧ x < 4}) (hB : B = {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}):
  (Aᶜ ∩ Bᶜ) = (Iic (-1) ∪ Ici 5) :=
by
  sorry

end intersection_compl_A_compl_B_l962_96282


namespace N_subset_M_values_l962_96228

def M : Set ℝ := { x | 2 * x^2 - 3 * x - 2 = 0 }
def N (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem N_subset_M_values (a : ℝ) (h : N a ⊆ M) : a = 0 ∨ a = -2 ∨ a = 1/2 := 
by
  sorry

end N_subset_M_values_l962_96228


namespace intersection_A_B_l962_96272

-- Define the sets A and B
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

-- Prove that A ∩ B = {0, 1}
theorem intersection_A_B :
  A ∩ B = {0, 1} :=
by
  -- Proof goes here
  sorry

end intersection_A_B_l962_96272


namespace evaluate_f_i_l962_96285

noncomputable def f (x : ℂ) : ℂ :=
  (x^5 + 2 * x^3 + x) / (x + 1)

theorem evaluate_f_i : f (Complex.I) = 0 := 
  sorry

end evaluate_f_i_l962_96285


namespace solve_for_x_l962_96225

theorem solve_for_x (x : ℝ) (h : 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l962_96225


namespace initial_investors_and_contribution_l962_96205

theorem initial_investors_and_contribution :
  ∃ (x y : ℕ), 
    (x - 10) * (y + 1) = x * y ∧
    (x - 25) * (y + 3) = x * y ∧
    x = 100 ∧ 
    y = 9 :=
by
  sorry

end initial_investors_and_contribution_l962_96205


namespace height_of_pyramid_l962_96229

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end height_of_pyramid_l962_96229


namespace upper_limit_for_y_l962_96201

theorem upper_limit_for_y (x y : ℝ) (hx : 5 < x) (hx' : x < 8) (hy : 8 < y) (h_diff : y - x = 7) : y ≤ 14 :=
by
  sorry

end upper_limit_for_y_l962_96201


namespace number_of_squares_l962_96247

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end number_of_squares_l962_96247


namespace find_ratio_l962_96226

variable {R : Type} [LinearOrderedField R]

def f (x a b : R) : R := x^3 + a*x^2 + b*x - a^2 - 7*a

def condition1 (a b : R) : Prop := f 1 a b = 10

def condition2 (a b : R) : Prop :=
  let f' := fun x => 3*x^2 + 2*a*x + b
  f' 1 = 0

theorem find_ratio (a b : R) (h1 : condition1 a b) (h2 : condition2 a b) :
  a / b = -2 / 3 :=
  sorry

end find_ratio_l962_96226


namespace quadratic_passing_origin_l962_96223

theorem quadratic_passing_origin (a b c : ℝ) (h : a ≠ 0) :
  ((∀ x y : ℝ, x = 0 → y = 0 → y = a * x^2 + b * x + c) ↔ c = 0) := 
by
  sorry

end quadratic_passing_origin_l962_96223


namespace balloon_descent_rate_l962_96233

theorem balloon_descent_rate (D : ℕ) 
    (rate_of_ascent : ℕ := 50) 
    (time_chain_pulled_1 : ℕ := 15) 
    (time_chain_pulled_2 : ℕ := 15) 
    (time_chain_released_1 : ℕ := 10) 
    (highest_elevation : ℕ := 1400) :
    (time_chain_pulled_1 + time_chain_pulled_2) * rate_of_ascent - time_chain_released_1 * D = highest_elevation 
    → D = 10 := 
by 
  intro h
  sorry

end balloon_descent_rate_l962_96233


namespace gcd_840_1764_l962_96292

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by
  -- proof steps will go here
  sorry

end gcd_840_1764_l962_96292


namespace typing_time_l962_96234

def original_speed : ℕ := 212
def reduction : ℕ := 40
def new_speed : ℕ := original_speed - reduction
def document_length : ℕ := 3440
def required_time : ℕ := 20

theorem typing_time :
  document_length / new_speed = required_time :=
by
  sorry

end typing_time_l962_96234


namespace square_sum_zero_real_variables_l962_96243

theorem square_sum_zero_real_variables (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end square_sum_zero_real_variables_l962_96243


namespace karen_start_time_late_l962_96267

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l962_96267


namespace no_minus_three_in_range_l962_96237

theorem no_minus_three_in_range (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b^2 < 24 :=
by
  sorry

end no_minus_three_in_range_l962_96237


namespace multiple_of_a_l962_96258

theorem multiple_of_a's_share (A B : ℝ) (x : ℝ) (h₁ : A + B + 260 = 585) (h₂ : x * A = 780) (h₃ : 6 * B = 780) : x = 4 :=
sorry

end multiple_of_a_l962_96258


namespace tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l962_96259

variable (α : ℝ)
variable (h1 : π / 2 < α)
variable (h2 : α < π)
variable (h3 : Real.sin α = 4 / 5)

theorem tan_alpha_neg_four_thirds (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : Real.tan α = -4 / 3 := 
by sorry

theorem cos2alpha_plus_cos_alpha_add_pi_over_2 (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (2 * α) + Real.cos (α + π / 2) = -27 / 25 := 
by sorry

end tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l962_96259


namespace new_average_weight_calculation_l962_96266

noncomputable def new_average_weight (total_weight : ℝ) (number_of_people : ℝ) : ℝ :=
  total_weight / number_of_people

theorem new_average_weight_calculation :
  let initial_people := 6
  let initial_avg_weight := 156
  let new_person_weight := 121
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

end new_average_weight_calculation_l962_96266


namespace minimum_value_l962_96252

open Real

theorem minimum_value (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z > 0, 3 * sqrt z + 2 / z ≥ y) ∧ y = 5 := by
  sorry

end minimum_value_l962_96252


namespace polygon_sidedness_l962_96200

-- Define the condition: the sum of the interior angles of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Given condition
def given_condition : ℝ := 1260

-- Target proposition to prove
theorem polygon_sidedness (n : ℕ) (h : sum_of_interior_angles n = given_condition) : n = 9 :=
by
  sorry

end polygon_sidedness_l962_96200


namespace smallest_k_for_sixty_four_gt_four_nineteen_l962_96270

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen_l962_96270


namespace ernie_can_make_circles_l962_96273

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l962_96273


namespace diamond_example_l962_96214

def diamond (a b : ℕ) : ℤ := 4 * a + 5 * b - a^2 * b

theorem diamond_example : diamond 3 4 = -4 :=
by
  rw [diamond]
  calc
    4 * 3 + 5 * 4 - 3^2 * 4 = 12 + 20 - 36 := by norm_num
                           _              = -4 := by norm_num

end diamond_example_l962_96214


namespace sequence_formula_l962_96209

theorem sequence_formula (u : ℕ → ℤ) (h0 : u 0 = 1) (h1 : u 1 = 4)
  (h_rec : ∀ n : ℕ, u (n + 2) = 5 * u (n + 1) - 6 * u n) :
  ∀ n : ℕ, u n = 2 * 3^n - 2^n :=
by 
  sorry

end sequence_formula_l962_96209


namespace intersection_of_M_and_N_l962_96218

-- Define set M
def M : Set ℝ := {x | Real.log x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the target set
def target : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N :
  M ∩ N = target :=
sorry

end intersection_of_M_and_N_l962_96218


namespace probability_ace_then_king_l962_96213

-- Definitions of the conditions
def custom_deck := 65
def extra_spades := 14
def total_aces := 4
def total_kings := 4

-- Probability calculations
noncomputable def P_ace_first : ℚ := total_aces / custom_deck
noncomputable def P_king_second : ℚ := total_kings / (custom_deck - 1)

theorem probability_ace_then_king :
  (P_ace_first * P_king_second) = 1 / 260 :=
by
  sorry

end probability_ace_then_king_l962_96213


namespace triangle_perimeter_correct_l962_96289

noncomputable def triangle_perimeter (a b x : ℕ) : ℕ := a + b + x

theorem triangle_perimeter_correct :
  ∀ (x : ℕ), (2 + 4 + x = 10) → 2 < x → x < 6 → (∀ k : ℕ, k = x → k % 2 = 0) → triangle_perimeter 2 4 x = 10 :=
by
  intros x h1 h2 h3
  rw [triangle_perimeter, h1]
  sorry

end triangle_perimeter_correct_l962_96289


namespace bicycle_wheels_l962_96239

theorem bicycle_wheels :
  ∀ (b : ℕ),
  let bicycles := 24
  let tricycles := 14
  let wheels_per_tricycle := 3
  let total_wheels := 90
  ((bicycles * b) + (tricycles * wheels_per_tricycle) = total_wheels) → b = 2 :=
by {
  sorry
}

end bicycle_wheels_l962_96239


namespace seq_proof_l962_96249
noncomputable def seq1_arithmetic (a1 a2 : ℝ) : Prop :=
  ∃ d : ℝ, a1 = -2 + d ∧ a2 = a1 + d ∧ -8 = a2 + d

noncomputable def seq2_geometric (b1 b2 b3 : ℝ) : Prop :=
  ∃ r : ℝ, b1 = -2 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -8 = b3 * r

theorem seq_proof (a1 a2 b1 b2 b3: ℝ) (h1 : seq1_arithmetic a1 a2) (h2 : seq2_geometric b1 b2 b3) :
  (a2 - a1) / b2 = 1 / 2 :=
sorry

end seq_proof_l962_96249


namespace axis_of_symmetry_values_ge_one_range_m_l962_96235

open Real

-- Definitions for vectors and the function f(x)
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Part I: Prove the equation of the axis of symmetry of f(x)
theorem axis_of_symmetry {k : ℤ} : f x = (sqrt 2 / 2) * sin (2 * x - π / 4) + 1 / 2 → 
                                    x = k * π / 2 + 3 * π / 8 := 
sorry

-- Part II: Prove the set of values x for which f(x) ≥ 1
theorem values_ge_one : (f x ≥ 1) ↔ (∃ (k : ℤ), π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) := 
sorry

-- Part III: Prove the range of m given the inequality
theorem range_m (m : ℝ) : (∀ x, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
                            m > (sqrt 3 - 5) / 4 := 
sorry

end axis_of_symmetry_values_ge_one_range_m_l962_96235


namespace stream_speed_l962_96241

/-- The speed of the stream problem -/
theorem stream_speed 
    (b s : ℝ) 
    (downstream_time : ℝ := 3)
    (upstream_time : ℝ := 3)
    (downstream_distance : ℝ := 60)
    (upstream_distance : ℝ := 30)
    (h1 : downstream_distance = (b + s) * downstream_time)
    (h2 : upstream_distance = (b - s) * upstream_time) : 
    s = 5 := 
by {
  -- The proof can be filled here
  sorry
}

end stream_speed_l962_96241


namespace volume_proportionality_l962_96299

variable (W V : ℕ)
variable (k : ℚ)

-- Given conditions
theorem volume_proportionality (h1 : V = k * W) (h2 : W = 112) (h3 : k = 3 / 7) :
  V = 48 := by
  sorry

end volume_proportionality_l962_96299


namespace range_of_a_l962_96236

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 - x + (a - 4) = 0 ∧ y^2 - y + (a - 4) = 0 ∧ x > 0 ∧ y < 0) → a < 4 :=
by
  sorry

end range_of_a_l962_96236


namespace find_other_number_l962_96290

theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 385) 
  (h2 : Nat.lcm A B = 2310) 
  (h3 : Nat.gcd A B = 30) : 
  B = 180 := 
by
  sorry

end find_other_number_l962_96290


namespace tractors_moved_l962_96255

-- Define initial conditions
def total_area (tractors: ℕ) (days: ℕ) (hectares_per_day: ℕ) := tractors * days * hectares_per_day

theorem tractors_moved (original_tractors remaining_tractors: ℕ)
  (days_original: ℕ) (hectares_per_day_original: ℕ)
  (days_remaining: ℕ) (hectares_per_day_remaining: ℕ)
  (total_area_original: ℕ) 
  (h1: total_area original_tractors days_original hectares_per_day_original = total_area_original)
  (h2: total_area remaining_tractors days_remaining hectares_per_day_remaining = total_area_original) :
  original_tractors - remaining_tractors = 2 :=
by
  sorry

end tractors_moved_l962_96255
