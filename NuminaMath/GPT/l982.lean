import Mathlib

namespace hyperbola_equation_l982_98277

-- Definitions based on problem conditions
def asymptotes (x y : ℝ) : Prop :=
  y = (1/3) * x ∨ y = -(1/3) * x

def focus (p : ℝ × ℝ) : Prop :=
  p = (Real.sqrt 10, 0)

-- The main statement to prove
theorem hyperbola_equation :
  (∃ p, focus p) ∧ (∀ (x y : ℝ), asymptotes x y) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 = 1)) :=
sorry

end hyperbola_equation_l982_98277


namespace dogs_with_flea_collars_l982_98215

-- Conditions
def T : ℕ := 80
def Tg : ℕ := 45
def B : ℕ := 6
def N : ℕ := 1

-- Goal: prove the number of dogs with flea collars is 40 given the above conditions
theorem dogs_with_flea_collars : ∃ F : ℕ, F = 40 ∧ T = Tg + F - B + N := 
by
  use 40
  sorry

end dogs_with_flea_collars_l982_98215


namespace haley_total_lives_l982_98292

-- Define initial conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def lives_gained : ℕ := 36

-- Definition to calculate total lives
def total_lives (initial_lives lives_lost lives_gained : ℕ) : ℕ :=
  initial_lives - lives_lost + lives_gained

-- The theorem statement we want to prove
theorem haley_total_lives : total_lives initial_lives lives_lost lives_gained = 46 :=
by 
  sorry

end haley_total_lives_l982_98292


namespace positive_number_property_l982_98241

theorem positive_number_property (y : ℝ) (hy : 0 < y) : 
  (y^2 / 100) + 6 = 10 → y = 20 := by
  sorry

end positive_number_property_l982_98241


namespace carla_zoo_l982_98280

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l982_98280


namespace white_tshirts_per_pack_l982_98293

-- Define the given conditions
def packs_white := 5
def packs_blue := 3
def t_shirts_per_blue_pack := 9
def total_t_shirts := 57

-- Define the total number of blue t-shirts
def total_blue_t_shirts := packs_blue * t_shirts_per_blue_pack

-- Define the variable W for the number of white t-shirts per pack
variable (W : ℕ)

-- Define the total number of white t-shirts
def total_white_t_shirts := packs_white * W

-- State the theorem to prove
theorem white_tshirts_per_pack :
    total_white_t_shirts + total_blue_t_shirts = total_t_shirts → W = 6 :=
by
  sorry

end white_tshirts_per_pack_l982_98293


namespace joe_eggs_town_hall_l982_98223

-- Define the conditions.
def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_total : ℕ := 20

-- Define the desired result.
def eggs_town_hall : ℕ := eggs_total - eggs_club_house - eggs_park

-- The statement that needs to be proved.
theorem joe_eggs_town_hall : eggs_town_hall = 3 :=
by
  sorry

end joe_eggs_town_hall_l982_98223


namespace g_at_neg10_l982_98260

def g (x : ℤ) : ℤ := 
  if x < -3 then 3 * x + 7 else 4 - x

theorem g_at_neg10 : g (-10) = -23 := by
  -- The proof goes here
  sorry

end g_at_neg10_l982_98260


namespace second_alloy_amount_l982_98235

theorem second_alloy_amount (x : ℝ) :
  let chromium_first_alloy := 0.12 * 15
  let chromium_second_alloy := 0.08 * x
  let total_weight := 15 + x
  let chromium_percentage_new_alloy := (0.12 * 15 + 0.08 * x) / (15 + x)
  chromium_percentage_new_alloy = (28 / 300) →
  x = 30 := sorry

end second_alloy_amount_l982_98235


namespace cubic_yard_to_cubic_meter_l982_98242

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter_l982_98242


namespace percent_decrease_is_30_l982_98232

def original_price : ℝ := 100
def sale_price : ℝ := 70
def decrease_in_price : ℝ := original_price - sale_price

theorem percent_decrease_is_30 : (decrease_in_price / original_price) * 100 = 30 :=
by
  sorry

end percent_decrease_is_30_l982_98232


namespace find_k_l982_98230

theorem find_k (k : ℝ) : 
  (1 / 2) * |k| * |k / 2| = 4 → (k = 4 ∨ k = -4) := 
sorry

end find_k_l982_98230


namespace complex_number_purely_imaginary_l982_98220

theorem complex_number_purely_imaginary (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 1 ≠ 0) → m = 3 :=
by
  intros h
  sorry

end complex_number_purely_imaginary_l982_98220


namespace evaluate_expression_at_y_minus3_l982_98200

theorem evaluate_expression_at_y_minus3 :
  let y := -3
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 :=
by
  let y := -3
  sorry

end evaluate_expression_at_y_minus3_l982_98200


namespace CanVolume_l982_98297

variable (X Y : Type) [Field X] [Field Y] (V W : X)

theorem CanVolume (mix_ratioX mix_ratioY drawn_volume new_ratioX new_ratioY : ℤ)
  (h1 : mix_ratioX = 5) (h2 : mix_ratioY = 7) (h3 : drawn_volume = 12) 
  (h4 : new_ratioX = 4) (h5 : new_ratioY = 7) :
  V = 72 ∧ W = 72 := 
sorry

end CanVolume_l982_98297


namespace parabola_min_value_l982_98254

variable {x0 y0 : ℝ}

def isOnParabola (x0 y0 : ℝ) : Prop := x0^2 = y0

noncomputable def expression (y0 x0 : ℝ) : ℝ :=
  Real.sqrt 2 * y0 + |x0 - y0 - 2|

theorem parabola_min_value :
  isOnParabola x0 y0 → ∃ (m : ℝ), m = (9 / 4 : ℝ) - (Real.sqrt 2 / 4) ∧ 
  ∀ y0 x0, expression y0 x0 ≥ (9 / 4 : ℝ) - (Real.sqrt 2 / 4) := 
by
  sorry

end parabola_min_value_l982_98254


namespace circumscribed_center_on_Ox_axis_l982_98257

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end circumscribed_center_on_Ox_axis_l982_98257


namespace opposite_of_2023_l982_98264

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l982_98264


namespace find_m_from_hyperbola_and_parabola_l982_98289

theorem find_m_from_hyperbola_and_parabola (a m : ℝ) 
  (h_eccentricity : (Real.sqrt (a^2 + 4)) / a = 3 * Real.sqrt 5 / 5) 
  (h_focus_coincide : (m / 4) = -3) : m = -12 := 
  sorry

end find_m_from_hyperbola_and_parabola_l982_98289


namespace quadratic_has_one_solution_l982_98210

theorem quadratic_has_one_solution (m : ℝ) : 3 * (49 / 12) - 7 * (49 / 12) + m = 0 → m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_l982_98210


namespace no_30_cents_l982_98239

/-- Given six coins selected from nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total value of the six coins cannot be 30 cents or less. -/
theorem no_30_cents {n d q : ℕ} (h : n + d + q = 6) (hn : n * 5 + d * 10 + q * 25 <= 30) : false :=
by
  sorry

end no_30_cents_l982_98239


namespace commute_time_l982_98216

theorem commute_time (d s1 s2 : ℝ) (h1 : s1 = 45) (h2 : s2 = 30) (h3 : d = 18) : (d / s1 + d / s2 = 1) :=
by
  -- Definitions and assumptions
  rw [h1, h2, h3]
  -- Total time calculation
  exact sorry

end commute_time_l982_98216


namespace solution_set_of_inequality_l982_98294

theorem solution_set_of_inequality :
  { x : ℝ | (x - 3) * (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l982_98294


namespace find_y_l982_98238

theorem find_y (y : ℝ) (a b : ℝ × ℝ) (h_a : a = (4, 2)) (h_b : b = (6, y)) (h_parallel : 4 * y - 2 * 6 = 0) :
  y = 3 :=
sorry

end find_y_l982_98238


namespace cost_milk_is_5_l982_98209

-- Define the total cost the baker paid
def total_cost : ℕ := 80

-- Define the cost components
def cost_flour : ℕ := 3 * 3
def cost_eggs : ℕ := 3 * 10
def cost_baking_soda : ℕ := 2 * 3

-- Define the number of liters of milk
def liters_milk : ℕ := 7

-- Define the unknown cost per liter of milk
noncomputable def cost_per_liter_milk (c : ℕ) : Prop :=
  c * liters_milk = total_cost - (cost_flour + cost_eggs + cost_baking_soda)

-- State the theorem we want to prove
theorem cost_milk_is_5 : cost_per_liter_milk 5 := 
by
  sorry

end cost_milk_is_5_l982_98209


namespace find_f_three_l982_98245

noncomputable def f : ℝ → ℝ := sorry -- f(x) is a linear function

axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

axiom equation : ∀ x, f x = 3 * (f⁻¹ x) + 9

axiom f_zero : f 0 = 3

axiom f_inv_three : f⁻¹ 3 = 0

theorem find_f_three : f 3 = 6 * Real.sqrt 3 := 
by sorry

end find_f_three_l982_98245


namespace min_angle_for_quadrilateral_l982_98284

theorem min_angle_for_quadrilateral (d : ℝ) (h : ∀ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a + b + c + d = 360 → (a < d ∨ b < d)) :
  d = 120 :=
by
  sorry

end min_angle_for_quadrilateral_l982_98284


namespace evaluate_expression_l982_98221

theorem evaluate_expression :
  (Int.floor ((Int.ceil ((11/5:ℚ)^2)) * (19/3:ℚ))) = 31 :=
by
  sorry

end evaluate_expression_l982_98221


namespace geometric_sequence_a7_l982_98243

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

-- Given condition
axiom geom_seq_condition : a 4 * a 10 = 9

-- proving the required result
theorem geometric_sequence_a7 (h : is_geometric_sequence a r) : a 7 = 3 ∨ a 7 = -3 :=
by
  sorry

end geometric_sequence_a7_l982_98243


namespace problem_ABCD_cos_l982_98283

/-- In convex quadrilateral ABCD, angle A = 2 * angle C, AB = 200, CD = 200, the perimeter of 
ABCD is 720, and AD ≠ BC. Find the floor of 1000 * cos A. -/
theorem problem_ABCD_cos (A C : ℝ) (AB CD AD BC : ℝ) (h1 : AB = 200)
  (h2 : CD = 200) (h3 : AD + BC = 320) (h4 : A = 2 * C)
  (h5 : AD ≠ BC) : ⌊1000 * Real.cos A⌋ = 233 := 
sorry

end problem_ABCD_cos_l982_98283


namespace calendar_reuse_initial_year_l982_98228

theorem calendar_reuse_initial_year (y k : ℕ)
    (h2064 : 2052 % 4 = 0)
    (h_y: y + 28 * k = 2052) :
    y = 1912 := by
  sorry

end calendar_reuse_initial_year_l982_98228


namespace cupcake_ratio_l982_98237

theorem cupcake_ratio (C B : ℕ) (hC : C = 4) (hTotal : C + B = 12) : B / C = 2 :=
by
  sorry

end cupcake_ratio_l982_98237


namespace digit_sum_solution_l982_98204

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l982_98204


namespace minimum_value_of_x_plus_y_l982_98256

-- Define the conditions as a hypothesis and the goal theorem statement.
theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 9 / y = 1) :
  x + y = 16 :=
by
  sorry

end minimum_value_of_x_plus_y_l982_98256


namespace distinct_digit_sum_l982_98246

theorem distinct_digit_sum (a b c d : ℕ) (h1 : a + c = 10) (h2 : b + c = 9) (h3 : a + d = 1)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : a ≠ d) (h7 : b ≠ c) (h8 : b ≠ d) (h9 : c ≠ d)
  (h10 : a < 10) (h11 : b < 10) (h12 : c < 10) (h13 : d < 10)
  (h14 : 0 ≤ a) (h15 : 0 ≤ b) (h16 : 0 ≤ c) (h17 : 0 ≤ d) :
  a + b + c + d = 18 :=
sorry

end distinct_digit_sum_l982_98246


namespace correct_expression_l982_98259

theorem correct_expression (a b c : ℝ) : 3 * a - (2 * b - c) = 3 * a - 2 * b + c :=
sorry

end correct_expression_l982_98259


namespace find_smallest_m_l982_98270

def is_in_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), ((1 / 2 : ℝ) ≤ x) ∧ (x ≤ Real.sqrt 2 / 2) ∧ (z = (x : ℂ) + (y : ℂ) * Complex.I)

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def smallest_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_S z ∧ is_nth_root_of_unity z n

theorem find_smallest_m : smallest_m 24 :=
  sorry

end find_smallest_m_l982_98270


namespace find_a_maximize_profit_l982_98214

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end find_a_maximize_profit_l982_98214


namespace number_of_bowls_l982_98296

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l982_98296


namespace Rachel_total_score_l982_98251

theorem Rachel_total_score
    (points_per_treasure : ℕ)
    (treasures_first_level : ℕ)
    (treasures_second_level : ℕ)
    (h1 : points_per_treasure = 9)
    (h2 : treasures_first_level = 5)
    (h3 : treasures_second_level = 2) : 
    (points_per_treasure * treasures_first_level + points_per_treasure * treasures_second_level = 63) :=
by
    sorry

end Rachel_total_score_l982_98251


namespace servings_per_guest_l982_98208

-- Definitions based on conditions
def num_guests : ℕ := 120
def servings_per_bottle : ℕ := 6
def num_bottles : ℕ := 40

-- Theorem statement
theorem servings_per_guest : (num_bottles * servings_per_bottle) / num_guests = 2 := by
  sorry

end servings_per_guest_l982_98208


namespace solve_weights_problem_l982_98234

variable (a b c d : ℕ) 

def weights_problem := 
  a + b = 280 ∧ 
  a + d = 300 ∧ 
  c + d = 290 → 
  b + c = 270

theorem solve_weights_problem (a b c d : ℕ) : weights_problem a b c d :=
 by
  sorry

end solve_weights_problem_l982_98234


namespace solve_for_n_l982_98285

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end solve_for_n_l982_98285


namespace cone_lateral_surface_area_l982_98205

theorem cone_lateral_surface_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 90) : 
  let base_circumference := 2 * Real.pi * r
  let R := 12
  let lateral_surface_area := (1 / 2) * base_circumference * R 
  lateral_surface_area = 36 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l982_98205


namespace equation_solution_unique_l982_98269

theorem equation_solution_unique (m a b : ℕ) (hm : 1 < m) (ha : 1 < a) (hb : 1 < b) :
  ((m + 1) * a = m * b + 1) ↔ m = 2 :=
sorry

end equation_solution_unique_l982_98269


namespace quadratic_expression_rewrite_l982_98291

theorem quadratic_expression_rewrite :
  ∃ a b c : ℚ, (∀ k : ℚ, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧ c + 3 * b = -49/3 :=
sorry

end quadratic_expression_rewrite_l982_98291


namespace max_parrots_l982_98248

-- Define the parameters and conditions for the problem
def N : ℕ := 2018
def Y : ℕ := 1009
def number_of_islanders (R L P : ℕ) := R + L + P = N

-- Define the main theorem
theorem max_parrots (R L P : ℕ) (h : number_of_islanders R L P) (hY : Y = 1009) :
  P = 1009 :=
sorry

end max_parrots_l982_98248


namespace max_k_value_l982_98247

noncomputable def max_k : ℝ := sorry 

theorem max_k_value :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧  (x - 4)^2 + y^2 ≤ 4) ↔ 
  k ≤ 4 / 3 := sorry

end max_k_value_l982_98247


namespace find_total_amount_l982_98272

noncomputable def total_amount (A T yearly_income : ℝ) : Prop :=
  0.05 * A + 0.06 * (T - A) = yearly_income

theorem find_total_amount :
  ∃ T : ℝ, total_amount 1600 T 140 ∧ T = 2600 :=
sorry

end find_total_amount_l982_98272


namespace negation_example_l982_98218

theorem negation_example : 
  (¬ ∃ x_0 : ℚ, x_0 - 2 = 0) = (∀ x : ℚ, x - 2 ≠ 0) :=
by 
  sorry

end negation_example_l982_98218


namespace tensor_calculation_jiaqi_statement_l982_98226

def my_tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem tensor_calculation :
  my_tensor (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := 
by
  sorry

theorem jiaqi_statement (a b : ℝ) (h : a + b = 0) :
  my_tensor a a + my_tensor b b = 2 * a * b := 
by
  sorry

end tensor_calculation_jiaqi_statement_l982_98226


namespace solve_for_C_l982_98206

-- Given constants and assumptions
def SumOfDigitsFirst (A B : ℕ) := 8 + 4 + A + 5 + 3 + B + 2 + 1
def SumOfDigitsSecond (A B C : ℕ) := 5 + 2 + 7 + A + B + 6 + 0 + C

theorem solve_for_C (A B C : ℕ) 
  (h1 : (SumOfDigitsFirst A B % 9) = 0)
  (h2 : (SumOfDigitsSecond A B C % 9) = 0) 
  : C = 3 :=
sorry

end solve_for_C_l982_98206


namespace concentration_of_first_solution_l982_98261

theorem concentration_of_first_solution
  (C : ℝ)
  (h : 4 * (C / 100) + 0.2 = 0.36) :
  C = 4 :=
by
  sorry

end concentration_of_first_solution_l982_98261


namespace min_value_l982_98268

theorem min_value (x : ℝ) (h : x > 1) : ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ y : ℝ, y = Real.sqrt (x - 1) → (x = y^2 + 1) → (x + 4) / y = m :=
by
  sorry

end min_value_l982_98268


namespace range_of_real_number_a_l982_98265

theorem range_of_real_number_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 1 = 0 → x = a) ↔ (a = 0 ∨ a ≥ 9/4) :=
sorry

end range_of_real_number_a_l982_98265


namespace curve_touches_all_Ca_l982_98229

theorem curve_touches_all_Ca (a : ℝ) (h : a > 0) : ∃ C : ℝ → ℝ, ∀ x y, (y - a^2)^2 = x^2 * (a^2 - x^2) → y = C x ∧ C x = 3 * x^2 / 4 :=
sorry

end curve_touches_all_Ca_l982_98229


namespace total_floor_area_l982_98298

theorem total_floor_area
    (n : ℕ) (a_cm : ℕ)
    (num_of_slabs : n = 30)
    (length_of_slab_cm : a_cm = 130) :
    (30 * ((130 * 130) / 10000)) = 50.7 :=
by
  sorry

end total_floor_area_l982_98298


namespace subset_P_Q_l982_98201

def P := {x : ℝ | x > 1}
def Q := {x : ℝ | x^2 - x > 0}

theorem subset_P_Q : P ⊆ Q :=
by
  sorry

end subset_P_Q_l982_98201


namespace compute_zeta_seventh_power_sum_l982_98271

noncomputable def complex_seventh_power_sum : Prop :=
  ∀ (ζ₁ ζ₂ ζ₃ : ℂ), 
    (ζ₁ + ζ₂ + ζ₃ = 1) ∧ 
    (ζ₁^2 + ζ₂^2 + ζ₃^2 = 3) ∧
    (ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) →
    (ζ₁^7 + ζ₂^7 + ζ₃^7 = 71)

theorem compute_zeta_seventh_power_sum : complex_seventh_power_sum :=
by
  sorry

end compute_zeta_seventh_power_sum_l982_98271


namespace gcd_polynomials_l982_98258

theorem gcd_polynomials (b : ℕ) (hb : ∃ k : ℕ, b = 2 * 7771 * k) :
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 19) = 8 :=
by sorry

end gcd_polynomials_l982_98258


namespace money_left_after_purchase_l982_98267

noncomputable def initial_money : ℝ := 200
noncomputable def candy_bars : ℝ := 25
noncomputable def bags_of_chips : ℝ := 10
noncomputable def soft_drinks : ℝ := 15

noncomputable def cost_per_candy_bar : ℝ := 3
noncomputable def cost_per_bag_of_chips : ℝ := 2.5
noncomputable def cost_per_soft_drink : ℝ := 1.75

noncomputable def discount_candy_bars : ℝ := 0.10
noncomputable def discount_bags_of_chips : ℝ := 0.05
noncomputable def sales_tax : ℝ := 0.06

theorem money_left_after_purchase : initial_money - 
  ( ((candy_bars * cost_per_candy_bar * (1 - discount_candy_bars)) + 
    (bags_of_chips * cost_per_bag_of_chips * (1 - discount_bags_of_chips)) + 
    (soft_drinks * cost_per_soft_drink)) * 
    (1 + sales_tax)) = 75.45 := by
  sorry

end money_left_after_purchase_l982_98267


namespace a2016_value_l982_98217

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 1 - (1 / a n)

theorem a2016_value : ∃ a : ℕ → ℚ, seq a ∧ a 2016 = 1 / 3 :=
by
  sorry

end a2016_value_l982_98217


namespace expression_for_x_l982_98211

variable (A B C x y : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := C > B
def condition3 := B > 0
def condition4 := C = (1 + y / 100) * B
def condition5 := A = (1 + x / 100) * C

-- The theorem
theorem expression_for_x (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 B) (h4 : condition4 B C y) (h5 : condition5 A C x) :
    x = 100 * ((100 * (A - B)) / (100 + y)) :=
sorry

end expression_for_x_l982_98211


namespace statement_B_l982_98287

variable (Student : Type)
variable (nora : Student)
variable (correctly_answered_all_math_questions : Student → Prop)
variable (received_at_least_B : Student → Prop)

theorem statement_B :
  (∀ s : Student, correctly_answered_all_math_questions s → received_at_least_B s) →
  (¬ received_at_least_B nora → ∃ q : Student, ¬ correctly_answered_all_math_questions q) :=
by
  intros h hn
  sorry

end statement_B_l982_98287


namespace answer_to_rarely_infrequently_word_l982_98231

-- Declare variables and definitions based on given conditions
-- In this context, we'll introduce a basic definition for the word "seldom".

noncomputable def is_word_meaning_rarely (w : String) : Prop :=
  w = "seldom"

-- Now state the problem in the form of a Lean theorem
theorem answer_to_rarely_infrequently_word : ∃ w, is_word_meaning_rarely w :=
by
  use "seldom"
  unfold is_word_meaning_rarely
  rfl

end answer_to_rarely_infrequently_word_l982_98231


namespace lemonade_total_difference_is_1860_l982_98233

-- Define the conditions
def stanley_rate : Nat := 4
def stanley_price : Real := 1.50

def carl_rate : Nat := 7
def carl_price : Real := 1.30

def lucy_rate : Nat := 5
def lucy_price : Real := 1.80

def hours : Nat := 3

-- Compute the total amounts for each sibling
def stanley_total : Real := stanley_rate * hours * stanley_price
def carl_total : Real := carl_rate * hours * carl_price
def lucy_total : Real := lucy_rate * hours * lucy_price

-- Compute the individual differences
def diff_stanley_carl : Real := carl_total - stanley_total
def diff_stanley_lucy : Real := lucy_total - stanley_total
def diff_carl_lucy : Real := carl_total - lucy_total

-- Sum the differences
def total_difference : Real := diff_stanley_carl + diff_stanley_lucy + diff_carl_lucy

-- The proof statement
theorem lemonade_total_difference_is_1860 :
  total_difference = 18.60 :=
by
  sorry

end lemonade_total_difference_is_1860_l982_98233


namespace minimum_questionnaires_l982_98266

theorem minimum_questionnaires (p : ℝ) (r : ℝ) (n_min : ℕ) (h1 : p = 0.65) (h2 : r = 300) :
  n_min = ⌈r / p⌉ ∧ n_min = 462 := 
by
  sorry

end minimum_questionnaires_l982_98266


namespace boxes_needed_l982_98275

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l982_98275


namespace initial_capacity_of_bottle_l982_98249

theorem initial_capacity_of_bottle 
  (C : ℝ)
  (h1 : 1/3 * 3/4 * C = 1) : 
  C = 4 :=
by
  sorry

end initial_capacity_of_bottle_l982_98249


namespace megan_works_per_day_hours_l982_98219

theorem megan_works_per_day_hours
  (h : ℝ)
  (earnings_per_hour : ℝ)
  (days_per_month : ℝ)
  (total_earnings_two_months : ℝ) :
  earnings_per_hour = 7.50 →
  days_per_month = 20 →
  total_earnings_two_months = 2400 →
  2 * days_per_month * earnings_per_hour * h = total_earnings_two_months →
  h = 8 :=
by {
  sorry
}

end megan_works_per_day_hours_l982_98219


namespace number_of_possible_digits_to_make_divisible_by_4_l982_98262

def four_digit_number_divisible_by_4 (N : ℕ) : Prop :=
  let number := N * 1000 + 264
  number % 4 = 0

theorem number_of_possible_digits_to_make_divisible_by_4 :
  ∃ (count : ℕ), count = 10 ∧ (∀ (N : ℕ), N < 10 → four_digit_number_divisible_by_4 N) :=
by {
  sorry
}

end number_of_possible_digits_to_make_divisible_by_4_l982_98262


namespace shortest_chord_through_point_l982_98286

theorem shortest_chord_through_point
  (correct_length : ℝ)
  (h1 : correct_length = 2 * Real.sqrt 2)
  (circle_eq : ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 4)
  (passes_point : ∀ (p : ℝ × ℝ), p = (3, 1)) :
  correct_length = 2 * Real.sqrt 2 :=
by {
  -- the proof steps would go here
  sorry
}

end shortest_chord_through_point_l982_98286


namespace trigonometric_expression_identity_l982_98244

open Real

theorem trigonometric_expression_identity :
  (1 - 1 / cos (35 * (pi / 180))) * 
  (1 + 1 / sin (55 * (pi / 180))) * 
  (1 - 1 / sin (35 * (pi / 180))) * 
  (1 + 1 / cos (55 * (pi / 180))) = 1 := by
  sorry

end trigonometric_expression_identity_l982_98244


namespace domain_f_l982_98207

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) + 1 / (x - 3)

theorem domain_f :
  {x : ℝ | x ≥ 2 ∧ x ≠ 3 } = {x : ℝ | (2 ≤ x ∧ x < 3) ∨ (3 < x)} :=
by
  sorry

end domain_f_l982_98207


namespace find_median_of_first_twelve_positive_integers_l982_98252

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l982_98252


namespace first_car_departure_time_l982_98225

variable (leave_time : Nat) -- in minutes past 8:00 am

def speed : Nat := 60 -- km/h
def firstCarTimeAt32 : Nat := 32 -- minutes since 8:00 am
def secondCarFactorAt32 : Nat := 3
def firstCarTimeAt39 : Nat := 39 -- minutes since 8:00 am
def secondCarFactorAt39 : Nat := 2

theorem first_car_departure_time :
  let firstCarSpeed := (60 / 60 : Nat) -- km/min
  let d1_32 := firstCarSpeed * firstCarTimeAt32
  let d2_32 := firstCarSpeed * (firstCarTimeAt32 - leave_time)
  let d1_39 := firstCarSpeed * firstCarTimeAt39
  let d2_39 := firstCarSpeed * (firstCarTimeAt39 - leave_time)
  d1_32 = secondCarFactorAt32 * d2_32 →
  d1_39 = secondCarFactorAt39 * d2_39 →
  leave_time = 11 :=
by
  intros h1 h2
  sorry

end first_car_departure_time_l982_98225


namespace find_abc_l982_98250

theorem find_abc (a b c : ℝ) 
  (h1 : a = 0.8 * b) 
  (h2 : c = 1.4 * b) 
  (h3 : c - a = 72) : 
  a = 96 ∧ b = 120 ∧ c = 168 := 
by
  sorry

end find_abc_l982_98250


namespace exist_projections_l982_98202

-- Define types for lines and points
variable {Point : Type} [MetricSpace Point]

-- Define the projection operator
def projection (t_i t_j : Set Point) (p : Point) : Point := 
  sorry -- projection definition will go here

-- Define t1, t2, ..., tk
variables (t : ℕ → Set Point) (k : ℕ)
  (hk : k > 1)  -- condition: k > 1
  (ht_distinct : ∀ i j, i ≠ j → t i ≠ t j)  -- condition: different lines

-- Define the proposition
theorem exist_projections : 
  ∃ (P : ℕ → Point), 
    (∀ i, 1 ≤ i ∧ i < k → P (i + 1) = projection (t i) (t (i + 1)) (P i)) ∧ 
    P 1 = projection (t k) (t 1) (P k) :=
sorry

end exist_projections_l982_98202


namespace difference_is_2395_l982_98290

def S : ℕ := 476
def L : ℕ := 6 * S + 15
def difference : ℕ := L - S

theorem difference_is_2395 : difference = 2395 :=
by
  sorry

end difference_is_2395_l982_98290


namespace second_number_less_than_twice_first_l982_98212

theorem second_number_less_than_twice_first (x y z : ℤ) (h1 : y = 37) (h2 : x + y = 57) (h3 : y = 2 * x - z) : z = 3 :=
by
  sorry

end second_number_less_than_twice_first_l982_98212


namespace second_odd_integer_is_72_l982_98299

def consecutive_odd_integers (n : ℤ) : ℤ × ℤ × ℤ :=
  (n - 2, n, n + 2)

theorem second_odd_integer_is_72 (n : ℤ) (h : (n - 2) + (n + 2) = 144) : n = 72 :=
by {
  sorry
}

end second_odd_integer_is_72_l982_98299


namespace triangle_side_range_a_l982_98274

theorem triangle_side_range_a {a : ℝ} : 2 < a ∧ a < 5 ↔
  3 + (2 * a + 1) > 8 ∧ 
  8 - 3 < 2 * a + 1 ∧ 
  8 - (2 * a + 1) < 3 :=
by
  sorry

end triangle_side_range_a_l982_98274


namespace sum_of_coefficients_condition_l982_98222

theorem sum_of_coefficients_condition 
  (t : ℕ → ℤ) 
  (d e f : ℤ) 
  (h0 : t 0 = 3) 
  (h1 : t 1 = 7) 
  (h2 : t 2 = 17) 
  (h3 : t 3 = 86)
  (rec_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2)) : 
  d + e + f = 14 :=
by
  sorry

end sum_of_coefficients_condition_l982_98222


namespace find_angle_A_range_of_bc_l982_98263

-- Define the necessary conditions and prove the size of angle A
theorem find_angle_A 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : b * (Real.sin B + Real.sin C) = (a - c) * (Real.sin A + Real.sin C))
  (h₂ : B > Real.pi / 2)
  (h₃ : A + B + C = Real.pi)
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0): 
  A = 2 * Real.pi / 3 :=
sorry

-- Define the necessary conditions and prove the range for b+c when a = sqrt(3)/2
theorem range_of_bc 
  (a b c : ℝ)
  (A : ℝ)
  (h₁ : A = 2 * Real.pi / 3)
  (h₂ : a = Real.sqrt 3 / 2)
  (h₃ : a > 0) (h₄ : b > 0) (h₅ : c > 0)
  (h₆ : A + B + C = Real.pi)
  (h₇ : B + C = Real.pi / 3) : 
  Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1 :=
sorry

end find_angle_A_range_of_bc_l982_98263


namespace add_2001_1015_l982_98295

theorem add_2001_1015 : 2001 + 1015 = 3016 := 
by
  sorry

end add_2001_1015_l982_98295


namespace cars_travel_same_distance_l982_98224

-- Define all the variables and conditions
def TimeR : ℝ := sorry -- the time taken by car R
def TimeP : ℝ := TimeR - 2
def SpeedR : ℝ := 58.4428877022476
def SpeedP : ℝ := SpeedR + 10

-- state the distance travelled by both cars
def DistanceR : ℝ := SpeedR * TimeR
def DistanceP : ℝ := SpeedP * TimeP

-- Prove that both distances are the same and equal to 800
theorem cars_travel_same_distance : DistanceR = 800 := by
  sorry

end cars_travel_same_distance_l982_98224


namespace total_pieces_of_art_l982_98273

variable (A : ℕ) (displayed : ℕ) (sculptures_on_display : ℕ) (not_on_display : ℕ) (paintings_not_on_display : ℕ) (sculptures_not_on_display : ℕ)

-- Constants and conditions from the problem
axiom H1 : displayed = 1 / 3 * A
axiom H2 : sculptures_on_display = 1 / 6 * displayed
axiom H3 : not_on_display = 2 / 3 * A
axiom H4 : paintings_not_on_display = 1 / 3 * not_on_display
axiom H5 : sculptures_not_on_display = 800
axiom H6 : sculptures_not_on_display = 2 / 3 * not_on_display

-- Prove that the total number of pieces of art is 1800
theorem total_pieces_of_art : A = 1800 :=
by
  sorry

end total_pieces_of_art_l982_98273


namespace necessary_condition_range_l982_98236

variables {x m : ℝ}

def p (x : ℝ) : Prop := x^2 - x - 2 < 0
def q (x m : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

theorem necessary_condition_range (H : ∀ x, q x m → p x) : -1 < m ∧ m < 1 :=
by {
  sorry
}

end necessary_condition_range_l982_98236


namespace minimum_value_problem_l982_98227

theorem minimum_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 + 4 * x^2 + 2 * x + 1) * (y^3 + 4 * y^2 + 2 * y + 1) * (z^3 + 4 * z^2 + 2 * z + 1) / (x * y * z) ≥ 1331 :=
sorry

end minimum_value_problem_l982_98227


namespace meaningful_domain_l982_98276

def is_meaningful (x : ℝ) : Prop :=
  (x - 1) ≠ 0

theorem meaningful_domain (x : ℝ) : is_meaningful x ↔ (x ≠ 1) :=
  sorry

end meaningful_domain_l982_98276


namespace factorial_ratio_integer_l982_98253

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end factorial_ratio_integer_l982_98253


namespace problem1_problem2_l982_98240

open Real

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 3| - |2 * x - a|

-- Problem (1)
theorem problem1 {a : ℝ} (h : ∃ x, f x a ≤ -5) : a ≤ -8 ∨ a ≥ 2 :=
sorry

-- Problem (2)
theorem problem2 {a : ℝ} (h : ∀ x, f (x - 1/2) a + f (-x - 1/2) a = 0) : a = 1 :=
sorry

end problem1_problem2_l982_98240


namespace IntervalForKTriangleLengths_l982_98213

noncomputable def f (x k : ℝ) := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem IntervalForKTriangleLengths (k : ℝ) :
  (∀ (x : ℝ), 1 ≤ f x k ∧
              (k ≥ 1 → f x k ≤ (k + 2) / 3) ∧ 
              (k < 1 → f x k ≥ (k + 2) / 3)) →
  (∀ (a b c : ℝ), (f a k < f b k + f c k) ∧ 
                  (f b k < f a k + f c k) ∧ 
                  (f c k < f a k + f b k)) ↔ (-1/2 < k ∧ k < 4) :=
by sorry

#check f
#check IntervalForKTriangleLengths

end IntervalForKTriangleLengths_l982_98213


namespace baseball_card_ratio_l982_98203

-- Define the conditions
variable (T : ℤ) -- Number of baseball cards on Tuesday

-- Given conditions
-- On Monday, Buddy has 30 baseball cards
def monday_cards : ℤ := 30

-- On Wednesday, Buddy has T + 12 baseball cards
def wednesday_cards : ℤ := T + 12

-- On Thursday, Buddy buys a third of what he had on Tuesday
def thursday_additional_cards : ℤ := T / 3

-- Total number of cards on Thursday is 32
def thursday_cards (T : ℤ) : ℤ := T + 12 + T / 3

-- We are given that Buddy has 32 baseball cards on Thursday
axiom thursday_total : thursday_cards T = 32

-- The theorem we want to prove: the ratio of Tuesday's to Monday's cards is 1:2
theorem baseball_card_ratio
  (T : ℤ)
  (htotal : thursday_cards T = 32)
  (hmon : monday_cards = 30) :
  T = 15 ∧ (T : ℚ) / monday_cards = 1 / 2 := by
  -- Proof goes here
  sorry

end baseball_card_ratio_l982_98203


namespace remainder_of_cake_l982_98281

theorem remainder_of_cake (John Emily : ℝ) (h1 : 0.60 ≤ John) (h2 : Emily = 0.50 * (1 - John)) :
  1 - John - Emily = 0.20 :=
by
  sorry

end remainder_of_cake_l982_98281


namespace length_of_train_l982_98279

theorem length_of_train (speed_kmh : ℕ) (time_seconds : ℕ) (h_speed : speed_kmh = 60) (h_time : time_seconds = 36) :
  let time_hours := (time_seconds : ℚ) / 3600
  let distance_km := (speed_kmh : ℚ) * time_hours
  let distance_m := distance_km * 1000
  distance_m = 600 :=
by
  sorry

end length_of_train_l982_98279


namespace weight_loss_clothes_percentage_l982_98282

theorem weight_loss_clothes_percentage (W : ℝ) : 
  let initial_weight := W
  let weight_after_loss := 0.89 * initial_weight
  let final_weight_with_clothes := 0.9078 * initial_weight
  let added_weight_percentage := (final_weight_with_clothes / weight_after_loss - 1) * 100
  added_weight_percentage = 2 :=
by
  sorry

end weight_loss_clothes_percentage_l982_98282


namespace goods_train_speed_l982_98288

def train_speed_km_per_hr (length_of_train length_of_platform time_to_cross : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_platform
  let speed_m_s := total_distance / time_to_cross
  speed_m_s * 36 / 10

-- Define the conditions given in the problem
def length_of_train : ℕ := 310
def length_of_platform : ℕ := 210
def time_to_cross : ℕ := 26

-- Define the target speed
def target_speed : ℕ := 72

-- The theorem proving the conclusion
theorem goods_train_speed :
  train_speed_km_per_hr length_of_train length_of_platform time_to_cross = target_speed := by
  sorry

end goods_train_speed_l982_98288


namespace inequality_solution_l982_98278

-- We define the problem
def interval_of_inequality : Set ℝ := { x : ℝ | (x + 1) * (2 - x) > 0 }

-- We define the expected solution set
def expected_solution_set : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

-- The theorem to be proved
theorem inequality_solution :
  interval_of_inequality = expected_solution_set := by 
  sorry

end inequality_solution_l982_98278


namespace circle_range_k_l982_98255

theorem circle_range_k (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y + 10 - k = 0) → k > 2 :=
by
  sorry

end circle_range_k_l982_98255
