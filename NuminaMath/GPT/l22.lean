import Mathlib

namespace avg_height_first_30_girls_l22_2214

theorem avg_height_first_30_girls (H : ℝ)
  (h1 : ∀ x : ℝ, 30 * x + 10 * 156 = 40 * 159) :
  H = 160 :=
by sorry

end avg_height_first_30_girls_l22_2214


namespace min_value_expression_l22_2285

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (y : ℝ), y = x * sqrt 2 ∧ ∀ (u : ℝ), ∀ (hu : u > 0), 
     sqrt ((x^2 + u^2) * (4 * x^2 + u^2)) / (x * u) ≥ 3 * sqrt 2) := 
sorry

end min_value_expression_l22_2285


namespace polynomial_divisible_by_squared_root_l22_2269

noncomputable def f (a1 a2 a3 a4 x : ℝ) : ℝ := 
  x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4

noncomputable def f_prime (a1 a2 a3 a4 x : ℝ) : ℝ := 
  4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_squared_root 
  (a1 a2 a3 a4 x0 : ℝ) 
  (h1 : f a1 a2 a3 a4 x0 = 0) 
  (h2 : f_prime a1 a2 a3 a4 x0 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x, f a1 a2 a3 a4 x = (x - x0)^2 * g x := 
sorry

end polynomial_divisible_by_squared_root_l22_2269


namespace joey_average_speed_l22_2228

noncomputable def average_speed_of_round_trip (distance_out : ℝ) (time_out : ℝ) (speed_return : ℝ) : ℝ :=
  let distance_return := distance_out
  let total_distance := distance_out + distance_return
  let time_return := distance_return / speed_return
  let total_time := time_out + time_return
  total_distance / total_time

theorem joey_average_speed :
  average_speed_of_round_trip 2 1 6.000000000000002 = 3 := by
  sorry

end joey_average_speed_l22_2228


namespace sally_earnings_proof_l22_2250

def sally_last_month_earnings : ℝ := 1000
def raise_percentage : ℝ := 0.10
def sally_this_month_earnings := sally_last_month_earnings * (1 + raise_percentage)
def sally_total_two_months_earnings := sally_last_month_earnings + sally_this_month_earnings

theorem sally_earnings_proof :
  sally_total_two_months_earnings = 2100 :=
by
  sorry

end sally_earnings_proof_l22_2250


namespace find_number_l22_2288

theorem find_number :
  ∃ (x : ℤ), 
  x * (x + 6) = -8 ∧ 
  x^4 + (x + 6)^4 = 272 :=
by
  sorry

end find_number_l22_2288


namespace point_P_coordinates_l22_2292

noncomputable def P_coordinates (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 4 * Real.sin θ)

theorem point_P_coordinates : 
  ∀ θ, (0 ≤ θ ∧ θ ≤ Real.pi ∧ 1 = (4 / 3) * Real.tan θ) →
  P_coordinates θ = (12 / 5, 12 / 5) :=
by
  intro θ h
  sorry

end point_P_coordinates_l22_2292


namespace solution_set_for_inequality_l22_2221

theorem solution_set_for_inequality :
  {x : ℝ | (1 / (x - 1) ≥ -1)} = {x : ℝ | x ≤ 0 ∨ x > 1} :=
by
  sorry

end solution_set_for_inequality_l22_2221


namespace M_identically_zero_l22_2209

noncomputable def M (x y : ℝ) : ℝ := sorry

theorem M_identically_zero (a : ℝ) (h1 : a > 1) (h2 : ∀ x, M x (a^x) = 0) : ∀ x y, M x y = 0 :=
sorry

end M_identically_zero_l22_2209


namespace rotation_test_l22_2267

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℝ) : Point ℝ :=
  Point.mk p.y (-p.x)

def A : Point ℝ := ⟨2, 3⟩
def B : Point ℝ := ⟨3, -2⟩

theorem rotation_test : rotate_90_clockwise A = B :=
by
  sorry

end rotation_test_l22_2267


namespace net_change_is_12_l22_2247

-- Definitions based on the conditions of the problem

def initial_investment : ℝ := 100
def first_year_increase_percentage : ℝ := 0.60
def second_year_decrease_percentage : ℝ := 0.30

-- Calculate the wealth at the end of the first year
def end_of_first_year_wealth : ℝ := initial_investment * (1 + first_year_increase_percentage)

-- Calculate the wealth at the end of the second year
def end_of_second_year_wealth : ℝ := end_of_first_year_wealth * (1 - second_year_decrease_percentage)

-- Calculate the net change
def net_change : ℝ := end_of_second_year_wealth - initial_investment

-- The target theorem to prove
theorem net_change_is_12 : net_change = 12 := by
  sorry

end net_change_is_12_l22_2247


namespace correct_pythagorean_triple_l22_2237

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end correct_pythagorean_triple_l22_2237


namespace max_divisors_with_remainder_10_l22_2222

theorem max_divisors_with_remainder_10 (m : ℕ) :
  (m > 0) → (∀ k, (2008 % k = 10) ↔ k < m) → m = 11 :=
by
  sorry

end max_divisors_with_remainder_10_l22_2222


namespace emily_annual_income_l22_2276

variables {q I : ℝ}

theorem emily_annual_income (h1 : (0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) = ((q + 0.75) * 0.01 * I)) : 
  I = 40000 := 
by
  sorry

end emily_annual_income_l22_2276


namespace eccentricity_of_ellipse_l22_2217

open Real

theorem eccentricity_of_ellipse 
  (O B F : ℝ × ℝ)
  (a b : ℝ) 
  (h_a_gt_b: a > b)
  (h_b_gt_0: b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_OB_eq_OF : dist O B = dist O F)
  (O_is_origin : O = (0,0))
  (B_is_upper_vertex : B = (0, b))
  (F_is_right_focus : F = (c, 0) ∧ c = Real.sqrt (a^2 - b^2)) :
 (c / a = sqrt 2 / 2)
:=
sorry

end eccentricity_of_ellipse_l22_2217


namespace cost_of_building_fence_l22_2232

-- Define the conditions
def area_of_circle := 289 -- Area in square feet
def price_per_foot := 58  -- Price in rupees per foot

-- Define the equations used in the problem
noncomputable def radius := Real.sqrt (area_of_circle / Real.pi)
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def cost := circumference * price_per_foot

-- The statement to prove
theorem cost_of_building_fence : cost = 1972 :=
  sorry

end cost_of_building_fence_l22_2232


namespace rate_per_meter_l22_2261

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (rate_per_meter : ℝ) (h_d : d = 30)
    (h_total_cost : total_cost = 188.49555921538757) :
    rate_per_meter = 2 :=
by
  sorry

end rate_per_meter_l22_2261


namespace common_remainder_proof_l22_2211

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end common_remainder_proof_l22_2211


namespace reduced_price_per_dozen_apples_l22_2226

variables (P R : ℝ) 

theorem reduced_price_per_dozen_apples (h₁ : R = 0.70 * P) 
  (h₂ : (30 / P + 54) * R = 30) :
  12 * R = 2 := 
sorry

end reduced_price_per_dozen_apples_l22_2226


namespace Teresa_current_age_l22_2251

-- Definitions of the conditions
def Morio_current_age := 71
def Morio_age_when_Michiko_born := 38
def Teresa_age_when_Michiko_born := 26

-- Definition of Michiko's current age
def Michiko_current_age := Morio_current_age - Morio_age_when_Michiko_born

-- The Theorem statement
theorem Teresa_current_age : Teresa_age_when_Michiko_born + Michiko_current_age = 59 :=
by
  -- Skip the proof
  sorry

end Teresa_current_age_l22_2251


namespace point_in_fourth_quadrant_l22_2296

def lies_in_fourth_quadrant (P : ℤ × ℤ) : Prop :=
  P.fst > 0 ∧ P.snd < 0

theorem point_in_fourth_quadrant : lies_in_fourth_quadrant (2023, -2024) :=
by
  -- Here is where the proof steps would go
  sorry

end point_in_fourth_quadrant_l22_2296


namespace inequality_abc_equality_condition_l22_2233

theorem inequality_abc (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) ≥ 12 :=
sorry

theorem equality_condition (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_abc_equality_condition_l22_2233


namespace percent_of_x_l22_2249

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25 - x / 10 + x / 5) = (16 / 100) * x := by
  sorry

end percent_of_x_l22_2249


namespace angles_on_x_axis_l22_2280

theorem angles_on_x_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi) ∨ (∃ k : ℤ, α = (2 * k + 1) * Real.pi) ↔ 
  ∃ k : ℤ, α = k * Real.pi :=
by
  sorry

end angles_on_x_axis_l22_2280


namespace algebraic_expression_value_l22_2234

theorem algebraic_expression_value (x : ℝ) 
  (h : 2 * x^2 + 3 * x + 7 = 8) : 
  4 * x^2 + 6 * x - 9 = -7 := 
by 
  sorry

end algebraic_expression_value_l22_2234


namespace manufacturing_department_percentage_l22_2210

theorem manufacturing_department_percentage (total_degrees mfg_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : mfg_degrees = 162) : (mfg_degrees / total_degrees) * 100 = 45 :=
by 
  sorry

end manufacturing_department_percentage_l22_2210


namespace min_value_expr_least_is_nine_l22_2219

noncomputable def minimum_value_expression (a b c d : ℝ) : ℝ :=
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2

theorem min_value_expr_least_is_nine (a b c d : ℝ)
  (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  minimum_value_expression a b c d = 9 := 
sorry

end min_value_expr_least_is_nine_l22_2219


namespace journey_time_difference_l22_2252

theorem journey_time_difference :
  let t1 := (100:ℝ) / 60
  let t2 := (400:ℝ) / 40
  let T1 := t1 + t2
  let T2 := (500:ℝ) / 50
  let difference := (T1 - T2) * 60
  abs (difference - 100) < 0.01 :=
by
  sorry

end journey_time_difference_l22_2252


namespace peanut_price_is_correct_l22_2216

noncomputable def price_per_pound_of_peanuts : ℝ := 
  let total_weight := 100
  let mixed_price_per_pound := 2.5
  let cashew_weight := 60
  let cashew_price_per_pound := 4
  let peanut_weight := total_weight - cashew_weight
  let total_revenue := total_weight * mixed_price_per_pound
  let cashew_cost := cashew_weight * cashew_price_per_pound
  let peanut_cost := total_revenue - cashew_cost
  peanut_cost / peanut_weight

theorem peanut_price_is_correct :
  price_per_pound_of_peanuts = 0.25 := 
by sorry

end peanut_price_is_correct_l22_2216


namespace bakery_batches_per_day_l22_2201

-- Definitions for the given problem's conditions
def baguettes_per_batch := 48
def baguettes_sold_batch1 := 37
def baguettes_sold_batch2 := 52
def baguettes_sold_batch3 := 49
def baguettes_left := 6

-- Theorem stating the number of batches made
theorem bakery_batches_per_day : 
  (baguettes_sold_batch1 + baguettes_sold_batch2 + baguettes_sold_batch3 + baguettes_left) / baguettes_per_batch = 3 :=
by 
  sorry

end bakery_batches_per_day_l22_2201


namespace grade_point_average_one_third_l22_2207

theorem grade_point_average_one_third :
  ∃ (x : ℝ), 55 = (1/3) * x + (2/3) * 60 ∧ x = 45 :=
by
  sorry

end grade_point_average_one_third_l22_2207


namespace max_sigma_squared_l22_2224

theorem max_sigma_squared (c d : ℝ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_c_ge_d : c ≥ d)
    (h : ∃ x y : ℝ, 0 ≤ x ∧ x < c ∧ 0 ≤ y ∧ y < d ∧ 
      c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x) ^ 2 + (d - y) ^ 2) : 
    σ^2 = 4 / 3 := by
  sorry

end max_sigma_squared_l22_2224


namespace trig_identity_sin_cos_l22_2287

theorem trig_identity_sin_cos
  (a : ℝ)
  (h : Real.sin (Real.pi / 3 - a) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - a) = -1 / 3 :=
by
  sorry

end trig_identity_sin_cos_l22_2287


namespace relationship_y1_y2_y3_l22_2258

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l22_2258


namespace domain_g_l22_2242

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 14 * x - 3)

theorem domain_g :
  {x : ℝ | -8 * x^2 + 14 * x - 3 ≥ 0} = { x : ℝ | x ≤ 1 / 4 ∨ x ≥ 3 / 2 } :=
by
  sorry

end domain_g_l22_2242


namespace question_I_question_II_l22_2283

def f (x a : ℝ) : ℝ := |x - a| + 3 * x

theorem question_I (a : ℝ) (h_pos : a > 0) : 
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) := by sorry

theorem question_II (a : ℝ) (h_pos : a > 0) : 
  (- (a / 2) = -1) ↔ (a = 2) := by sorry

end question_I_question_II_l22_2283


namespace magic_shop_purchase_l22_2278

theorem magic_shop_purchase :
  let deck_price := 7
  let frank_decks := 3
  let friend_decks := 2
  let discount_rate := 0.1
  let tax_rate := 0.05
  let total_cost := (frank_decks + friend_decks) * deck_price
  let discount := discount_rate * total_cost
  let discounted_total := total_cost - discount
  let sales_tax := tax_rate * discounted_total
  let rounded_sales_tax := (sales_tax * 100).round / 100
  let final_amount := discounted_total + rounded_sales_tax
  final_amount = 33.08 :=
by
  sorry

end magic_shop_purchase_l22_2278


namespace prism_coloring_1995_prism_coloring_1996_l22_2238

def prism_coloring_possible (n : ℕ) : Prop :=
  ∃ (color : ℕ → ℕ),
    (∀ i, 1 ≤ color i ∧ color i ≤ 3) ∧ -- Each color is within bounds
    (∀ i, color i ≠ color ((i + 1) % n)) ∧ -- Colors on each face must be different
    (n % 3 = 0 ∨ n ≠ 1996) -- Condition for coloring

theorem prism_coloring_1995 : prism_coloring_possible 1995 :=
sorry

theorem prism_coloring_1996 : ¬prism_coloring_possible 1996 :=
sorry

end prism_coloring_1995_prism_coloring_1996_l22_2238


namespace area_enclosed_by_region_l22_2275

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l22_2275


namespace right_triangle_integral_sides_parity_l22_2244

theorem right_triangle_integral_sides_parity 
  (a b c : ℕ) 
  (h : a^2 + b^2 = c^2) 
  (ha : a % 2 = 1 ∨ a % 2 = 0) 
  (hb : b % 2 = 1 ∨ b % 2 = 0) 
  (hc : c % 2 = 1 ∨ c % 2 = 0) : 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) := 
sorry

end right_triangle_integral_sides_parity_l22_2244


namespace age_of_B_l22_2295

variable (a b : ℕ)

-- Conditions
def condition1 := a + 10 = 2 * (b - 10)
def condition2 := a = b + 5

-- The proof goal
theorem age_of_B (h1 : condition1 a b) (h2 : condition2 a b) : b = 35 := by
  sorry

end age_of_B_l22_2295


namespace odd_function_properties_l22_2272

def f : ℝ → ℝ := sorry

theorem odd_function_properties 
  (H1 : ∀ x, f (-x) = -f x) -- f is odd
  (H2 : ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) -- f is increasing on [1, 3]
  (H3 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 7) -- f has a minimum value of 7 on [1, 3]
  : (∀ x y, -3 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) -- f is increasing on [-3, -1]
    ∧ (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) -- f has a maximum value of -7 on [-3, -1]
:= sorry

end odd_function_properties_l22_2272


namespace total_cost_correct_l22_2218

def cost_first_day : Nat := 4 + 5 + 3 + 2
def cost_second_day : Nat := 5 + 6 + 4
def total_cost : Nat := cost_first_day + cost_second_day

theorem total_cost_correct : total_cost = 29 := by
  sorry

end total_cost_correct_l22_2218


namespace stanley_total_cost_l22_2284

theorem stanley_total_cost (n_tires : ℕ) (price_per_tire : ℝ) (h_n : n_tires = 4) (h_price : price_per_tire = 60) : n_tires * price_per_tire = 240 := by
  sorry

end stanley_total_cost_l22_2284


namespace number_of_girls_is_4_l22_2294

variable (x : ℕ)

def number_of_boys : ℕ := 12

def average_score_boys : ℕ := 84

def average_score_girls : ℕ := 92

def average_score_class : ℕ := 86

theorem number_of_girls_is_4 
  (h : average_score_class = 
    (average_score_boys * number_of_boys + average_score_girls * x) / (number_of_boys + x))
  : x = 4 := 
sorry

end number_of_girls_is_4_l22_2294


namespace min_value_arithmetic_seq_l22_2240

theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h_arith_seq : ∀ n, a n ≤ a (n + 1)) (h_pos : ∀ n, a n > 0) (h_cond : a 1 + a 2017 = 2) :
  ∃ (min_value : ℝ), min_value = 2 ∧ (∀ (x y : ℝ), x + y = 2 → x > 0 → y > 0 → x + y / (x * y) = 2) :=
  sorry

end min_value_arithmetic_seq_l22_2240


namespace find_initial_nickels_l22_2299

variable (initial_nickels current_nickels borrowed_nickels : ℕ)

def initial_nickels_equation (initial_nickels current_nickels borrowed_nickels : ℕ) : Prop :=
  initial_nickels - borrowed_nickels = current_nickels

theorem find_initial_nickels (h : initial_nickels_equation initial_nickels current_nickels borrowed_nickels) 
                             (h_current : current_nickels = 11) 
                             (h_borrowed : borrowed_nickels = 20) : 
                             initial_nickels = 31 :=
by
  sorry

end find_initial_nickels_l22_2299


namespace mushrooms_collected_l22_2213

theorem mushrooms_collected (x1 x2 x3 x4 : ℕ) 
  (h1 : x1 + x2 = 7) 
  (h2 : x1 + x3 = 9)
  (h3 : x2 + x3 = 10) : x1 = 3 ∧ x2 = 4 ∧ x3 = 6 ∧ x4 = 7 :=
by
  sorry

end mushrooms_collected_l22_2213


namespace max_profit_l22_2271

variables (x y : ℕ)

def steel_constraint := 10 * x + 70 * y ≤ 700
def non_ferrous_constraint := 23 * x + 40 * y ≤ 642
def non_negativity := x ≥ 0 ∧ y ≥ 0
def profit := 80 * x + 100 * y

theorem max_profit (h₁ : steel_constraint x y)
                   (h₂ : non_ferrous_constraint x y)
                   (h₃ : non_negativity x y):
  profit x y = 2180 := 
sorry

end max_profit_l22_2271


namespace simplify_exponent_expression_l22_2256

theorem simplify_exponent_expression (n : ℕ) :
  (3^(n+4) - 3 * 3^n) / (3 * 3^(n+3)) = 26 / 9 := by
  sorry

end simplify_exponent_expression_l22_2256


namespace power_of_same_base_power_of_different_base_l22_2243

theorem power_of_same_base (a n : ℕ) (h : ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m) :
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ a^n = (a^k)^m :=
  sorry

theorem power_of_different_base (a n : ℕ) : ∃ (b m : ℕ), a^n = b^m :=
  sorry

end power_of_same_base_power_of_different_base_l22_2243


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l22_2281

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l22_2281


namespace smallest_repeating_block_length_of_7_over_13_l22_2259

theorem smallest_repeating_block_length_of_7_over_13 : 
  ∀ k, (∃ a b, 7 / 13 = a + (b / 10^k)) → k = 6 := 
sorry

end smallest_repeating_block_length_of_7_over_13_l22_2259


namespace find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l22_2289

theorem find_counterfeit_80_coins_in_4_weighings :
  ∃ f : Fin 80 → Bool, (∃ i, f i = true) ∧ (∃ i j, f i ≠ f j) := sorry

theorem min_weighings_for_n_coins (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, 3^(k-1) < n ∧ n ≤ 3^k := sorry

end find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l22_2289


namespace max_sum_of_four_numbers_l22_2208

theorem max_sum_of_four_numbers : 
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ (2 * a + 3 * b + 2 * c + 3 * d = 2017) ∧ 
    (a + b + c + d = 806) :=
by
  sorry

end max_sum_of_four_numbers_l22_2208


namespace eval_expr_l22_2204

variable {x y : ℝ}

theorem eval_expr (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) - ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = (2 * x^2) / (y^2) + (2 * y^2) / (x^2) := by
  sorry

end eval_expr_l22_2204


namespace area_of_smaller_circle_l22_2239

noncomputable def radius_large_circle (x : ℝ) : ℝ := 2 * x
noncomputable def radius_small_circle (y : ℝ) : ℝ := y

theorem area_of_smaller_circle 
(pa ab : ℝ)
(r : ℝ)
(area : ℝ) 
(h1 : pa = 5) 
(h2 : ab = 5) 
(h3 : radius_large_circle r = 2 * radius_small_circle r)
(h4 : 2 * radius_small_circle r + radius_large_circle r = 10)
(h5 : area = Real.pi * (radius_small_circle r)^2) 
: area = 6.25 * Real.pi :=
by
  sorry

end area_of_smaller_circle_l22_2239


namespace area_shaded_region_l22_2257

theorem area_shaded_region :
  let r_s := 3   -- Radius of the smaller circle
  let r_l := 3 * r_s  -- Radius of the larger circle
  let A_l := π * r_l^2  -- Area of the larger circle
  let A_s := π * r_s^2  -- Area of the smaller circle
  A_l - A_s = 72 * π := 
by
  sorry

end area_shaded_region_l22_2257


namespace num_of_valid_m_vals_l22_2246

theorem num_of_valid_m_vals : 
  (∀ m x : ℤ, (x + m ≤ 4 ∧ (x / 2 - (x - 1) / 4 > 1 → x > 3 → ∃ (c : ℚ), (x + 1)/4 > 1 )) ∧
  (∃ (x : ℤ), (x + m ≤ 4 ∧ (x > 3) ∧ (m < 1 ∧ m > -4)) ∧ 
  ∃ a b : ℚ, x^2 + a * x + b = 0) → 
  (∃ (count m : ℤ), count = 2)) :=
sorry

end num_of_valid_m_vals_l22_2246


namespace integer_roots_of_polynomial_l22_2227

theorem integer_roots_of_polynomial :
  {x : ℤ | x^3 - 4*x^2 - 14*x + 24 = 0} = {-4, -3, 3} := by
  sorry

end integer_roots_of_polynomial_l22_2227


namespace olivia_insurance_premium_l22_2293

theorem olivia_insurance_premium :
  ∀ (P : ℕ) (base_premium accident_percentage ticket_cost : ℤ) (tickets accidents : ℕ),
    base_premium = 50 →
    accident_percentage = P →
    ticket_cost = 5 →
    tickets = 3 →
    accidents = 1 →
    (base_premium + (accidents * base_premium * P / 100) + (tickets * ticket_cost) = 70) →
    P = 10 :=
by
  intros P base_premium accident_percentage ticket_cost tickets accidents
  intro h1 h2 h3 h4 h5 h6
  sorry

end olivia_insurance_premium_l22_2293


namespace car_speed_problem_l22_2279

theorem car_speed_problem (S1 S2 : ℝ) (T : ℝ) (avg_speed : ℝ) (H1 : S1 = 70) (H2 : T = 2) (H3 : avg_speed = 80) :
  S2 = 90 :=
by
  have avg_speed_eq : avg_speed = (S1 + S2) / T := sorry
  have h : S2 = 90 := sorry
  exact h

end car_speed_problem_l22_2279


namespace positive_real_solutions_unique_l22_2268

theorem positive_real_solutions_unique (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
(h : (a^2 - b * d) / (b + 2 * c + d) + (b^2 - c * a) / (c + 2 * d + a) + (c^2 - d * b) / (d + 2 * a + b) + (d^2 - a * c) / (a + 2 * b + c) = 0) : 
a = b ∧ b = c ∧ c = d :=
sorry

end positive_real_solutions_unique_l22_2268


namespace relationship_between_y_values_l22_2248

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l22_2248


namespace percentage_less_than_l22_2264

theorem percentage_less_than (x y : ℝ) (h : x = 8 * y) : ((x - y) / x) * 100 = 87.5 := 
by sorry

end percentage_less_than_l22_2264


namespace hyperbola_eq_l22_2286

/-- Given a hyperbola with center at the origin, 
    one focus at (-√5, 0), and a point P on the hyperbola such that 
    the midpoint of segment PF₁ has coordinates (0, 2), 
    then the equation of the hyperbola is x² - y²/4 = 1. --/
theorem hyperbola_eq (x y : ℝ) (P F1 : ℝ × ℝ) 
  (hF1 : F1 = (-Real.sqrt 5, 0)) 
  (hMidPoint : (P.1 + -Real.sqrt 5) / 2 = 0 ∧ (P.2 + 0) / 2 = 2) 
  : x^2 - y^2 / 4 = 1 := 
sorry

end hyperbola_eq_l22_2286


namespace correct_sunset_time_proof_l22_2298

def Time := ℕ × ℕ  -- hours and minutes

def sunrise_time : Time := (7, 12)  -- 7:12 AM
def incorrect_daylight_duration : Time := (11, 15)  -- 11 hours 15 minutes as per newspaper

def add_time (t1 t2 : Time) : Time :=
  let (h1, m1) := t1
  let (h2, m2) := t2
  let minutes := m1 + m2
  let hours := h1 + h2 + minutes / 60
  (hours % 24, minutes % 60)

def correct_sunset_time : Time := (18, 27)  -- 18:27 in 24-hour format equivalent to 6:27 PM in 12-hour format

theorem correct_sunset_time_proof :
  add_time sunrise_time incorrect_daylight_duration = correct_sunset_time :=
by
  -- skipping the detailed proof for now
  sorry

end correct_sunset_time_proof_l22_2298


namespace maria_savings_l22_2297

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l22_2297


namespace f_at_neg_one_l22_2253

def f : ℝ → ℝ := sorry

theorem f_at_neg_one :
  (∀ x : ℝ, f (x / (1 + x)) = x) →
  f (-1) = -1 / 2 :=
by
  intro h
  -- proof omitted for clarity
  sorry

end f_at_neg_one_l22_2253


namespace single_rooms_booked_l22_2235

noncomputable def hotel_problem (S D : ℕ) : Prop :=
  S + D = 260 ∧ 35 * S + 60 * D = 14000

theorem single_rooms_booked (S D : ℕ) (h : hotel_problem S D) : S = 64 :=
by
  sorry

end single_rooms_booked_l22_2235


namespace Carlos_earnings_l22_2225

theorem Carlos_earnings :
  ∃ (wage : ℝ), 
  (18 * wage) = (12 * wage + 36) ∧ 
  wage = 36 / 6 ∧ 
  (12 * wage + 18 * wage) = 180 :=
by
  sorry

end Carlos_earnings_l22_2225


namespace exists_nat_numbers_satisfying_sum_l22_2265

theorem exists_nat_numbers_satisfying_sum :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 :=
sorry

end exists_nat_numbers_satisfying_sum_l22_2265


namespace product_of_fractions_l22_2205

theorem product_of_fractions :
  (2 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) * (5 / 6 : ℚ) * (6 / 7 : ℚ) * (7 / 8 : ℚ) = 1 / 4 :=
by
  sorry

end product_of_fractions_l22_2205


namespace number_of_hens_l22_2254

theorem number_of_hens (H C G : ℕ) 
  (h1 : H + C + G = 120) 
  (h2 : 2 * H + 4 * C + 4 * G = 348) : 
  H = 66 := 
by 
  sorry

end number_of_hens_l22_2254


namespace max_expression_value_l22_2212

open Real

theorem max_expression_value (a b d x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : (x₁^4 - a * x₁^3 + b * x₁^2 - a * x₁ + d = 0))
  (h2 : (x₂^4 - a * x₂^3 + b * x₂^2 - a * x₂ + d = 0))
  (h3 : (x₃^4 - a * x₃^3 + b * x₃^2 - a * x₃ + d = 0))
  (h4 : (x₄^4 - a * x₄^3 + b * x₄^2 - a * x₄ + d = 0))
  (h5 : (1 / 2 ≤ x₁ ∧ x₁ ≤ 2))
  (h6 : (1 / 2 ≤ x₂ ∧ x₂ ≤ 2))
  (h7 : (1 / 2 ≤ x₃ ∧ x₃ ≤ 2))
  (h8 : (1 / 2 ≤ x₄ ∧ x₄ ≤ 2)) :
  ∃ (M : ℝ), M = 5 / 4 ∧
  (∀ (y₁ y₂ y₃ y₄ : ℝ),
    (y₁^4 - a * y₁^3 + b * y₁^2 - a * y₁ + d = 0) →
    (y₂^4 - a * y₂^3 + b * y₂^2 - a * y₂ + d = 0) →
    (y₃^4 - a * y₃^3 + b * y₃^2 - a * y₃ + d = 0) →
    (y₄^4 - a * y₄^3 + b * y₄^2 - a * y₄ + d = 0) →
    (1 / 2 ≤ y₁ ∧ y₁ ≤ 2) →
    (1 / 2 ≤ y₂ ∧ y₂ ≤ 2) →
    (1 / 2 ≤ y₃ ∧ y₃ ≤ 2) →
    (1 / 2 ≤ y₄ ∧ y₄ ≤ 2) →
    (y = (y₁ + y₂) * (y₁ + y₃) * y₄ / ((y₄ + y₂) * (y₄ + y₃) * y₁)) →
    y ≤ M) := 
sorry

end max_expression_value_l22_2212


namespace find_S20_l22_2203

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

axiom a_nonzero (n : ℕ) : a_seq n ≠ 0
axiom a1_eq : a_seq 1 = 1
axiom Sn_eq (n : ℕ) : S n = (a_seq n * a_seq (n + 1)) / 2

theorem find_S20 : S 20 = 210 := sorry

end find_S20_l22_2203


namespace range_of_a_l22_2200

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → ((x - a) ^ 2 < 1)) ↔ (1 ≤ a ∧ a ≤ 2) :=
by 
  sorry

end range_of_a_l22_2200


namespace abs_eq_k_solution_l22_2231

theorem abs_eq_k_solution (k : ℝ) (h : k > 4014) :
  {x : ℝ | |x - 2007| + |x + 2007| = k} = (Set.Iio (-2007)) ∪ (Set.Ioi (2007)) :=
by
  sorry

end abs_eq_k_solution_l22_2231


namespace average_speed_entire_journey_l22_2245

-- Define the average speed for the journey from x to y
def speed_xy := 60

-- Define the average speed for the journey from y to x
def speed_yx := 30

-- Definition for the distance (D) (it's an abstract value, so we don't need to specify)
variable (D : ℝ) (hD : D > 0)

-- Theorem stating that the average speed for the entire journey is 40 km/hr
theorem average_speed_entire_journey : 
  2 * D / ((D / speed_xy) + (D / speed_yx)) = 40 := 
by 
  sorry

end average_speed_entire_journey_l22_2245


namespace function_values_l22_2223

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem function_values (a b : ℝ) (h1 : f 1 a b = 2) (h2 : a = 2) : f 2 a b = 4 := by
  sorry

end function_values_l22_2223


namespace percentage_error_in_side_l22_2229

theorem percentage_error_in_side {S S' : ℝ}
  (hs : S > 0)
  (hs' : S' > S)
  (h_area_error : (S'^2 - S^2) / S^2 * 100 = 90.44) :
  ((S' - S) / S * 100) = 38 :=
by
  sorry

end percentage_error_in_side_l22_2229


namespace exists_infinite_diff_but_not_sum_of_kth_powers_l22_2277

theorem exists_infinite_diff_but_not_sum_of_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (infinitely_many x : ℕ), (∃ (a b : ℕ), x = a^k - b^k) ∧ ¬ (∃ (c d : ℕ), x = c^k + d^k) :=
  sorry

end exists_infinite_diff_but_not_sum_of_kth_powers_l22_2277


namespace negation_of_P_is_non_P_l22_2260

open Real

/-- Proposition P: For any x in the real numbers, sin(x) <= 1 -/
def P : Prop := ∀ x : ℝ, sin x ≤ 1

/-- Negation of P: There exists x in the real numbers such that sin(x) >= 1 -/
def non_P : Prop := ∃ x : ℝ, sin x ≥ 1

theorem negation_of_P_is_non_P : ¬P ↔ non_P :=
by 
  sorry

end negation_of_P_is_non_P_l22_2260


namespace simplify_expression_l22_2262

variable (x : ℝ)

theorem simplify_expression : 2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := 
  sorry

end simplify_expression_l22_2262


namespace calculate_tan_product_l22_2290

theorem calculate_tan_product :
  let A := 30
  let B := 40
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2.9 :=
by
  sorry

end calculate_tan_product_l22_2290


namespace problem_statement_l22_2206

noncomputable def solveProblem : ℝ :=
  let a := 2
  let b := -3
  let c := 1
  a + b + c

-- The theorem statement to ensure a + b + c equals 0
theorem problem_statement : solveProblem = 0 := by
  sorry

end problem_statement_l22_2206


namespace minimum_work_to_remove_cube_l22_2241

namespace CubeBuoyancy

def edge_length (ℓ : ℝ) := ℓ = 0.30 -- in meters
def wood_density (ρ : ℝ) := ρ = 750  -- in kg/m^3
def water_density (ρ₀ : ℝ) := ρ₀ = 1000 -- in kg/m^3

theorem minimum_work_to_remove_cube 
  {ℓ ρ ρ₀ : ℝ} 
  (h₁ : edge_length ℓ)
  (h₂ : wood_density ρ)
  (h₃ : water_density ρ₀) : 
  ∃ W : ℝ, W = 22.8 := 
sorry

end CubeBuoyancy

end minimum_work_to_remove_cube_l22_2241


namespace max_cube_sum_l22_2282

theorem max_cube_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : x^3 + y^3 + z^3 ≤ 27 :=
sorry

end max_cube_sum_l22_2282


namespace common_ratio_geometric_sequence_l22_2263

theorem common_ratio_geometric_sequence (n : ℕ) :
  ∃ q : ℕ, (∀ k : ℕ, q = 4^(2*k+3) / 4^(2*k+1)) ∧ q = 16 :=
by
  use 16
  sorry

end common_ratio_geometric_sequence_l22_2263


namespace part_a_l22_2270

theorem part_a (x y : ℝ) : x^2 - 2*y^2 = -((x + 2*y)^2 - 2*(x + y)^2) :=
sorry

end part_a_l22_2270


namespace solve_for_x_l22_2266

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 8) * x = 12) : x = 168 := by
  sorry

end solve_for_x_l22_2266


namespace Anton_thought_of_729_l22_2255

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l22_2255


namespace butterfat_milk_mixing_l22_2202

theorem butterfat_milk_mixing :
  ∀ (x : ℝ), 
  (0.35 * x + 0.10 * 12 = 0.20 * (x + 12)) → x = 8 :=
by
  intro x
  intro h
  sorry

end butterfat_milk_mixing_l22_2202


namespace parallel_vectors_solution_l22_2220

theorem parallel_vectors_solution 
  (x : ℝ) 
  (a : ℝ × ℝ := (-1, 3)) 
  (b : ℝ × ℝ := (x, 1)) 
  (h : ∃ k : ℝ, a = k • b) :
  x = -1 / 3 :=
by
  sorry

end parallel_vectors_solution_l22_2220


namespace height_of_fifth_tree_l22_2274

theorem height_of_fifth_tree 
  (h₁ : tallest_tree = 108) 
  (h₂ : second_tallest_tree = 54 - 6) 
  (h₃ : third_tallest_tree = second_tallest_tree / 4) 
  (h₄ : fourth_shortest_tree = (second_tallest_tree + third_tallest_tree) - 2) 
  (h₅ : fifth_tree = 0.75 * (tallest_tree + second_tallest_tree + third_tallest_tree + fourth_shortest_tree)) : 
  fifth_tree = 169.5 :=
by
  sorry

end height_of_fifth_tree_l22_2274


namespace no_valid_pairs_l22_2291

theorem no_valid_pairs : ∀ (a b : ℕ), (a > 0) → (b > 0) → (a ≥ b) → 
  a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b → 
  false := by
  sorry

end no_valid_pairs_l22_2291


namespace find_m_l22_2273

-- Define the points M and N and the normal vector n
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M (m : ℝ) : Point3D := { x := m, y := -2, z := 1 }
def N (m : ℝ) : Point3D := { x := 0, y := m, z := 3 }
def n : Point3D := { x := 3, y := 1, z := 2 }

-- Define the dot product
def dot_product (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

-- Define the vector MN
def MN (m : ℝ) : Point3D := { x := -(m), y := m + 2, z := 2 }

-- Prove the dot product condition is zero implies m = 3
theorem find_m (m : ℝ) (h : dot_product n (MN m) = 0) : m = 3 :=
by
  sorry

end find_m_l22_2273


namespace find_fraction_l22_2236

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction_l22_2236


namespace probability_is_five_eleven_l22_2215

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := n.choose k

-- Define the number of favorable outcomes for same letter and same color
def favorable_same_letter : ℕ := 4 * comb 3 2
def favorable_same_color : ℕ := 3 * comb 4 2

-- Total number of favorable outcomes
def total_favorable : ℕ := favorable_same_letter + favorable_same_color

-- Total number of ways to draw 2 cards from 12
def total_ways : ℕ := comb total_cards 2

-- Probability of drawing a winning pair
def probability_winning_pair : ℚ := total_favorable / total_ways

theorem probability_is_five_eleven : probability_winning_pair = 5 / 11 :=
by
  sorry

end probability_is_five_eleven_l22_2215


namespace polynomial_condition_l22_2230

theorem polynomial_condition {P : Polynomial ℝ} :
  (∀ (a b c : ℝ), a * b + b * c + c * a = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
    ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X^4 + Polynomial.C β * Polynomial.X^2 :=
by
  intro h
  sorry

end polynomial_condition_l22_2230
