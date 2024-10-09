import Mathlib

namespace range_of_m_l432_43291

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x >= (4 + m)) ∧ (x <= 3 * (x - 2) + 4) → (x ≥ 2)) →
  (-3 < m ∧ m <= -2) :=
sorry

end range_of_m_l432_43291


namespace intersection_P_Q_l432_43266

open Set

noncomputable def P : Set ℝ := {1, 2, 3, 4}

noncomputable def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := 
by {
  sorry
}

end intersection_P_Q_l432_43266


namespace olive_charged_10_hours_l432_43288

/-- If Olive charges her phone for 3/5 of the time she charged last night, and that results
    in 12 hours of use, where each hour of charge results in 2 hours of phone usage,
    then the time Olive charged her phone last night was 10 hours. -/
theorem olive_charged_10_hours (x : ℝ) 
  (h1 : 2 * (3 / 5) * x = 12) : 
  x = 10 :=
by
  sorry

end olive_charged_10_hours_l432_43288


namespace lukas_points_in_5_games_l432_43253

theorem lukas_points_in_5_games (avg_points_per_game : ℕ) (games_played : ℕ) (total_points : ℕ)
  (h_avg : avg_points_per_game = 12) (h_games : games_played = 5) : total_points = 60 :=
by
  sorry

end lukas_points_in_5_games_l432_43253


namespace tom_charges_per_lawn_l432_43272

theorem tom_charges_per_lawn (gas_cost earnings_from_weeding total_profit lawns_mowed : ℕ) (charge_per_lawn : ℤ) 
  (h1 : gas_cost = 17)
  (h2 : earnings_from_weeding = 10)
  (h3 : total_profit = 29)
  (h4 : lawns_mowed = 3)
  (h5 : total_profit = ((lawns_mowed * charge_per_lawn) + earnings_from_weeding) - gas_cost) :
  charge_per_lawn = 12 := 
by
  sorry

end tom_charges_per_lawn_l432_43272


namespace first_cube_weight_l432_43214

-- Given definitions of cubes and their relationships
def weight_of_cube (s : ℝ) (weight : ℝ) : Prop :=
  ∃ v : ℝ, v = s^3 ∧ weight = v

def cube_relationship (s1 s2 weight2 : ℝ) : Prop :=
  s2 = 2 * s1 ∧ weight2 = 32

-- The proof problem
theorem first_cube_weight (s1 s2 weight1 weight2 : ℝ) (h1 : cube_relationship s1 s2 weight2) : weight1 = 4 :=
  sorry

end first_cube_weight_l432_43214


namespace circles_externally_tangent_l432_43268

theorem circles_externally_tangent
  (r1 r2 d : ℝ)
  (hr1 : r1 = 2) (hr2 : r2 = 3)
  (hd : d = 5) :
  r1 + r2 = d :=
by
  sorry

end circles_externally_tangent_l432_43268


namespace inequality_implication_l432_43212

theorem inequality_implication (x : ℝ) : 3 * x + 4 < 5 * x - 6 → x > 5 := 
by {
  sorry
}

end inequality_implication_l432_43212


namespace nicole_answers_correctly_l432_43251

theorem nicole_answers_correctly :
  ∀ (C K N : ℕ), C = 17 → K = C + 8 → N = K - 3 → N = 22 :=
by
  intros C K N hC hK hN
  sorry

end nicole_answers_correctly_l432_43251


namespace problem1_l432_43279

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l432_43279


namespace geometric_sequence_a3_l432_43287

theorem geometric_sequence_a3 (a : ℕ → ℝ)
  (h : ∀ n m : ℕ, a (n + m) = a n * a m)
  (pos : ∀ n, 0 < a n)
  (a1 : a 1 = 1)
  (a5 : a 5 = 9) :
  a 3 = 3 := by
  sorry

end geometric_sequence_a3_l432_43287


namespace remainder_when_13_add_x_div_31_eq_22_l432_43218

open BigOperators

theorem remainder_when_13_add_x_div_31_eq_22
  (x : ℕ) (hx : x > 0) (hmod : 7 * x ≡ 1 [MOD 31]) :
  (13 + x) % 31 = 22 := 
  sorry

end remainder_when_13_add_x_div_31_eq_22_l432_43218


namespace average_of_remaining_two_l432_43283

theorem average_of_remaining_two
  (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg_2_1 : (a + b) / 2 = 4.2)
  (h_avg_2_2 : (c + d) / 2 = 3.85) : 
  ((e + f) / 2) = 3.8 :=
by
  sorry

end average_of_remaining_two_l432_43283


namespace four_number_theorem_l432_43211

theorem four_number_theorem (a b c d : ℕ) (H : a * b = c * d) (Ha : 0 < a) (Hb : 0 < b) (Hc : 0 < c) (Hd : 0 < d) : 
  ∃ (p q r s : ℕ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r :=
by
  sorry

end four_number_theorem_l432_43211


namespace total_amount_received_l432_43294

theorem total_amount_received (P R CI: ℝ) (T: ℕ) 
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (P_eq: P = 2828.80 / 0.1664) 
  (R_eq: R = 8) 
  (T_eq: T = 2) : 
  P + CI = 19828.80 := 
by 
  sorry

end total_amount_received_l432_43294


namespace evaluate_expression_l432_43280

variable (x y : ℝ)

theorem evaluate_expression
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hsum_sq : x^2 + y^2 ≠ 0)
  (hsum : x + y ≠ 0) :
    (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) :=
sorry

end evaluate_expression_l432_43280


namespace volume_of_cylinder_cut_l432_43275

open Real

noncomputable def cylinder_cut_volume (R α : ℝ) : ℝ :=
  (2 / 3) * R^3 * tan α

theorem volume_of_cylinder_cut (R α : ℝ) :
  cylinder_cut_volume R α = (2 / 3) * R^3 * tan α :=
by
  sorry

end volume_of_cylinder_cut_l432_43275


namespace fraction_of_integer_l432_43255

theorem fraction_of_integer :
  (5 / 6) * 30 = 25 :=
by
  sorry

end fraction_of_integer_l432_43255


namespace unique_solution_implies_relation_l432_43260

theorem unique_solution_implies_relation (a b : ℝ)
    (h : ∃! (x y : ℝ), y = x^2 + a * x + b ∧ x = y^2 + a * y + b) : 
    a^2 = 2 * (a + 2 * b) - 1 :=
by
  sorry

end unique_solution_implies_relation_l432_43260


namespace statues_ratio_l432_43204

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio_l432_43204


namespace second_horse_revolutions_l432_43249

-- Define the parameters and conditions:
def r₁ : ℝ := 30  -- Distance of the first horse from the center
def revolutions₁ : ℕ := 15  -- Number of revolutions by the first horse
def r₂ : ℝ := 5  -- Distance of the second horse from the center

-- Define the statement to prove:
theorem second_horse_revolutions : r₂ * (↑revolutions₁ * r₁⁻¹) * (↑revolutions₁) = 90 := 
by sorry

end second_horse_revolutions_l432_43249


namespace find_minimum_value_l432_43224

-- This definition captures the condition that a, b, c are positive real numbers
def pos_reals := { x : ℝ // 0 < x }

-- The main theorem statement
theorem find_minimum_value (a b c : pos_reals) :
  4 * (a.1 ^ 4) + 8 * (b.1 ^ 4) + 16 * (c.1 ^ 4) + 1 / (a.1 * b.1 * c.1) ≥ 10 :=
by
  -- This is where the proof will go
  sorry

end find_minimum_value_l432_43224


namespace charge_difference_is_51_l432_43244

-- Define the charges and calculations for print shop X
def print_shop_x_cost (n : ℕ) : ℝ :=
  if n ≤ 50 then n * 1.20 else 50 * 1.20 + (n - 50) * 0.90

-- Define the charges and calculations for print shop Y
def print_shop_y_cost (n : ℕ) : ℝ :=
  10 + n * 1.70

-- Define the difference in charges for 70 copies
def charge_difference : ℝ :=
  print_shop_y_cost 70 - print_shop_x_cost 70

-- The proof statement
theorem charge_difference_is_51 : charge_difference = 51 :=
by
  sorry

end charge_difference_is_51_l432_43244


namespace S_2012_value_l432_43297

-- Define the first term of the arithmetic sequence
def a1 : ℤ := -2012

-- Define the common difference
def d : ℤ := 2

-- Define the sequence a_n
def a (n : ℕ) : ℤ := a1 + d * (n - 1)

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℤ := n * (a1 + a n) / 2

-- Formalize the given problem as a Lean statement
theorem S_2012_value : S 2012 = -2012 :=
by 
{
  -- The proof is omitted as requested
  sorry
}

end S_2012_value_l432_43297


namespace cos_A_minus_B_l432_43267

variable {A B : ℝ}

-- Conditions
def cos_conditions (A B : ℝ) : Prop :=
  (Real.cos A + Real.cos B = 1 / 2)

def sin_conditions (A B : ℝ) : Prop :=
  (Real.sin A + Real.sin B = 3 / 2)

-- Mathematically equivalent proof problem
theorem cos_A_minus_B (h1 : cos_conditions A B) (h2 : sin_conditions A B) :
  Real.cos (A - B) = 1 / 4 := 
sorry

end cos_A_minus_B_l432_43267


namespace simplified_expression_l432_43262

-- Non-computable context since we are dealing with square roots and division
noncomputable def expr (x : ℝ) : ℝ := ((x / (x - 1)) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem simplified_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : expr x = Real.sqrt 2 / 2 := by
  sorry

end simplified_expression_l432_43262


namespace sufficient_not_necessary_condition_l432_43232

theorem sufficient_not_necessary_condition (a : ℝ)
  : (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, a ≥ 0 ∨ a * x^2 + x + 1 ≥ 0)
:= sorry

end sufficient_not_necessary_condition_l432_43232


namespace system1_solution_system2_solution_l432_43229

-- System (1)
theorem system1_solution {x y : ℝ} : 
  x + y = 3 → 
  x - y = 1 → 
  (x = 2 ∧ y = 1) :=
by
  intros h1 h2
  -- proof goes here
  sorry

-- System (2)
theorem system2_solution {x y : ℝ} :
  2 * x + y = 3 →
  x - 2 * y = 1 →
  (x = 7 / 5 ∧ y = 1 / 5) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end system1_solution_system2_solution_l432_43229


namespace John_leftover_money_l432_43209

variables (q : ℝ)

def drinks_price (q : ℝ) : ℝ := 4 * q
def small_pizza_price (q : ℝ) : ℝ := q
def large_pizza_price (q : ℝ) : ℝ := 4 * q
def total_cost (q : ℝ) : ℝ := drinks_price q + small_pizza_price q + 2 * large_pizza_price q
def John_initial_money : ℝ := 50
def John_money_left (q : ℝ) : ℝ := John_initial_money - total_cost q

theorem John_leftover_money : John_money_left q = 50 - 13 * q :=
by
  sorry

end John_leftover_money_l432_43209


namespace equal_cookies_per_person_l432_43259

theorem equal_cookies_per_person 
  (boxes : ℕ) (cookies_per_box : ℕ) (people : ℕ)
  (h1 : boxes = 7) (h2 : cookies_per_box = 10) (h3 : people = 5) :
  (boxes * cookies_per_box) / people = 14 :=
by sorry

end equal_cookies_per_person_l432_43259


namespace percentage_of_acid_is_18_18_percent_l432_43239

noncomputable def percentage_of_acid_in_original_mixture
  (a w : ℝ) (h1 : (a + 1) / (a + w + 1) = 1 / 4) (h2 : (a + 1) / (a + w + 2) = 1 / 5) : ℝ :=
  a / (a + w) 

theorem percentage_of_acid_is_18_18_percent :
  ∃ (a w : ℝ), (a + 1) / (a + w + 1) = 1 / 4 ∧ (a + 1) / (a + w + 2) = 1 / 5 ∧ percentage_of_acid_in_original_mixture a w (by sorry) (by sorry) = 18.18 := by
  sorry

end percentage_of_acid_is_18_18_percent_l432_43239


namespace main_line_train_probability_l432_43265

noncomputable def probability_catching_main_line (start_main_line start_harbor_line : Nat) (frequency : Nat) : ℝ :=
  if start_main_line % frequency = 0 ∧ start_harbor_line % frequency = 2 then 1 / 2 else 0

theorem main_line_train_probability :
  probability_catching_main_line 0 2 10 = 1 / 2 :=
by
  sorry

end main_line_train_probability_l432_43265


namespace find_x_minus_y_l432_43290

theorem find_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : y^2 = 16) (h3 : x + y > 0) : x - y = 1 ∨ x - y = 9 := 
by sorry

end find_x_minus_y_l432_43290


namespace find_missing_number_l432_43200

theorem find_missing_number (x : ℕ) (h : 10010 - 12 * 3 * x = 9938) : x = 2 :=
by {
  sorry
}

end find_missing_number_l432_43200


namespace roots_of_cubic_l432_43274

-- Define the cubic equation having roots 3 and -2
def cubic_eq (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The proof problem statement
theorem roots_of_cubic (a b c d : ℝ) (h₁ : a ≠ 0)
  (h₂ : cubic_eq a b c d 3)
  (h₃ : cubic_eq a b c d (-2)) : 
  (b + c) / a = -7 := 
sorry

end roots_of_cubic_l432_43274


namespace simplify_expression_l432_43206

def is_real (x : ℂ) : Prop := ∃ (y : ℝ), x = y

theorem simplify_expression 
  (x y c : ℝ) 
  (i : ℂ) 
  (hi : i^2 = -1) :
  (x + i*y + c)^2 = (x^2 + c^2 - y^2 + 2 * c * x + (2 * x * y + 2 * c * y) * i) :=
by
  sorry

end simplify_expression_l432_43206


namespace trig_expression_equality_l432_43289

theorem trig_expression_equality :
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  2 * tan_60 + tan_45 - 4 * cos_30 = 1 := by
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  sorry

end trig_expression_equality_l432_43289


namespace least_number_of_tiles_l432_43220

/-- A room of 544 cm long and 374 cm broad is to be paved with square tiles. 
    Prove that the least number of square tiles required to cover the floor is 176. -/
theorem least_number_of_tiles (length breadth : ℕ) (h1 : length = 544) (h2 : breadth = 374) :
  let gcd_length_breadth := Nat.gcd length breadth
  let num_tiles_length := length / gcd_length_breadth
  let num_tiles_breadth := breadth / gcd_length_breadth
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end least_number_of_tiles_l432_43220


namespace order_of_xyz_l432_43276

variable (a b c d : ℝ)

noncomputable def x : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
noncomputable def y : ℝ := Real.sqrt (a * c) + Real.sqrt (b * d)
noncomputable def z : ℝ := Real.sqrt (a * d) + Real.sqrt (b * c)

theorem order_of_xyz (h₁ : a > b) (h₂ : b > c) (h₃ : c > d) (h₄ : d > 0) : x a b c d > y a b c d ∧ y a b c d > z a b c d :=
by
  sorry

end order_of_xyz_l432_43276


namespace express_y_in_terms_of_x_l432_43221

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 5) : y = 2 * x + 5 :=
by
  sorry

end express_y_in_terms_of_x_l432_43221


namespace geometric_series_sum_l432_43273

theorem geometric_series_sum : 
  let a := 6
  let r := - (2 / 5)
  let s := a / (1 - r)
  s = 30 / 7 :=
by
  let a := 6
  let r := -(2 / 5)
  let s := a / (1 - r)
  show s = 30 / 7
  sorry

end geometric_series_sum_l432_43273


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l432_43292

-- Define the solutions to the given quadratic equations

theorem solve_eq1 (x : ℝ) : 2 * x ^ 2 - 8 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : x ^ 2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1 :=
by sorry

theorem solve_eq3 (x : ℝ) : 5 * x ^ 2 - 4 * x - 1 = 0 ↔ x = -1 / 5 ∨ x = 1 :=
by sorry

theorem solve_eq4 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l432_43292


namespace tangent_line_at_origin_l432_43248

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 1

def tangent_line (x₀ y₀ : ℝ) (k : ℝ) (x : ℝ) := y₀ + k * (x - x₀)

theorem tangent_line_at_origin : 
  tangent_line 0 (-1) 3 = λ x => 3 * x - 1 :=
by
  sorry

end tangent_line_at_origin_l432_43248


namespace degree_le_three_l432_43227

theorem degree_le_three
  (d : ℕ)
  (P : Polynomial ℤ)
  (hdeg : P.degree = d)
  (hP : ∃ (S : Finset ℤ), (S.card ≥ d + 1) ∧ ∀ m ∈ S, |P.eval m| = 1) :
  d ≤ 3 := 
sorry

end degree_le_three_l432_43227


namespace speed_of_stream_l432_43235

theorem speed_of_stream
  (b s : ℝ)
  (H1 : 120 = 2 * (b + s))
  (H2 : 60 = 2 * (b - s)) :
  s = 15 :=
by
  sorry

end speed_of_stream_l432_43235


namespace area_enclosed_by_graph_eq_2pi_l432_43254

theorem area_enclosed_by_graph_eq_2pi :
  (∃ (x y : ℝ), x^2 + y^2 = 2 * |x| + 2 * |y| ) →
  ∀ (A : ℝ), A = 2 * Real.pi :=
sorry

end area_enclosed_by_graph_eq_2pi_l432_43254


namespace julia_paint_area_l432_43205

noncomputable def area_to_paint (bedroom_length: ℕ) (bedroom_width: ℕ) (bedroom_height: ℕ) (non_paint_area: ℕ) (num_bedrooms: ℕ) : ℕ :=
  let wall_area_one_bedroom := 2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)
  let paintable_area_one_bedroom := wall_area_one_bedroom - non_paint_area
  num_bedrooms * paintable_area_one_bedroom

theorem julia_paint_area :
  area_to_paint 14 11 9 70 4 = 1520 :=
by
  sorry

end julia_paint_area_l432_43205


namespace coefficient_x3_l432_43228

-- Define the binomial coefficient
def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

noncomputable def coefficient_x3_term : Nat :=
  binomial_coefficient 25 3

theorem coefficient_x3 : coefficient_x3_term = 2300 :=
by
  unfold coefficient_x3_term
  unfold binomial_coefficient
  -- Here, one would normally provide the proof steps, but we're adding sorry to skip
  sorry

end coefficient_x3_l432_43228


namespace instantaneous_velocity_at_2_l432_43238

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- State the problem: Prove the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : (deriv s) 2 = 4 := by
  sorry

end instantaneous_velocity_at_2_l432_43238


namespace elements_of_set_A_l432_43242

theorem elements_of_set_A (A : Set ℝ) (h₁ : ∀ a : ℝ, a ∈ A → (1 + a) / (1 - a) ∈ A)
(h₂ : -3 ∈ A) : A = {-3, -1/2, 1/3, 2} := by
  sorry

end elements_of_set_A_l432_43242


namespace f_monotonic_intervals_g_greater_than_4_3_l432_43256

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem f_monotonic_intervals :
  (∀ x < -1, ∀ y < -1, x < y → f x > f y) ∧ 
  (∀ x > -1, ∀ y > -1, x < y → f x < f y) :=
sorry

theorem g_greater_than_4_3 (x : ℝ) (h : x > 0) : g x > (4 / 3) :=
sorry

end f_monotonic_intervals_g_greater_than_4_3_l432_43256


namespace danielle_rooms_is_6_l432_43277

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l432_43277


namespace distance_equals_absolute_value_l432_43278

def distance_from_origin (x : ℝ) : ℝ := abs x

theorem distance_equals_absolute_value (x : ℝ) : distance_from_origin x = abs x :=
by
  sorry

end distance_equals_absolute_value_l432_43278


namespace set_equality_l432_43298

theorem set_equality (a : ℤ) : 
  {z : ℤ | ∃ x : ℤ, (x - a = z ∧ a - 1 ≤ x ∧ x ≤ a + 1)} = {-1, 0, 1} :=
by {
  sorry
}

end set_equality_l432_43298


namespace range_of_a_l432_43213

-- Define the function f
def f (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by {
  sorry
}

end range_of_a_l432_43213


namespace total_points_scored_l432_43264

theorem total_points_scored :
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  a + b + c + d + e + f + g + h = 54 :=
by
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  sorry

end total_points_scored_l432_43264


namespace cupcakes_left_l432_43217

theorem cupcakes_left (initial_cupcakes : ℕ)
  (students_delmont : ℕ) (ms_delmont : ℕ)
  (students_donnelly : ℕ) (mrs_donnelly : ℕ)
  (school_nurse : ℕ) (school_principal : ℕ) (school_custodians : ℕ)
  (favorite_teachers : ℕ) (cupcakes_per_favorite_teacher : ℕ)
  (other_classmates : ℕ) :
  initial_cupcakes = 80 →
  students_delmont = 18 → ms_delmont = 1 →
  students_donnelly = 16 → mrs_donnelly = 1 →
  school_nurse = 1 → school_principal = 1 → school_custodians = 3 →
  favorite_teachers = 5 → cupcakes_per_favorite_teacher = 2 → 
  other_classmates = 10 →
  initial_cupcakes - (students_delmont + ms_delmont +
                      students_donnelly + mrs_donnelly +
                      school_nurse + school_principal + school_custodians +
                      favorite_teachers * cupcakes_per_favorite_teacher +
                      other_classmates) = 19 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end cupcakes_left_l432_43217


namespace length_of_diagonal_l432_43286

open Real

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, -a^2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (-a, -a^2)
noncomputable def C (a : ℝ) : ℝ × ℝ := (a, -a^2)
def O : ℝ × ℝ := (0, 0)

noncomputable def is_square (A B O C : ℝ × ℝ) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O C ∧ dist O C = dist C A

theorem length_of_diagonal (a : ℝ) (h_square : is_square (A a) (B a) O (C a)) : 
  dist (A a) (C a) = 2 * abs a :=
sorry

end length_of_diagonal_l432_43286


namespace number_of_pairs_l432_43201

theorem number_of_pairs (x y : ℤ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000) :
  (x^2 + y^2) % 7 = 0 → (∃ n : ℕ, n = 20164) :=
by {
  sorry
}

end number_of_pairs_l432_43201


namespace cuboid_third_edge_l432_43215

theorem cuboid_third_edge (a b V h : ℝ) (ha : a = 4) (hb : b = 4) (hV : V = 96) (volume_formula : V = a * b * h) : h = 6 :=
by
  sorry

end cuboid_third_edge_l432_43215


namespace speed_in_m_per_s_eq_l432_43263

theorem speed_in_m_per_s_eq : (1 : ℝ) / 3.6 = (0.27777 : ℝ) :=
by sorry

end speed_in_m_per_s_eq_l432_43263


namespace solve_for_x_l432_43246

theorem solve_for_x (x : ℤ) (h : 24 - 6 = 3 + x) : x = 15 :=
by {
  sorry
}

end solve_for_x_l432_43246


namespace propA_propB_relation_l432_43241

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation_l432_43241


namespace travel_time_l432_43261

noncomputable def distance (time: ℝ) (rate: ℝ) : ℝ := time * rate

theorem travel_time
  (initial_time: ℝ)
  (initial_speed: ℝ)
  (reduced_speed: ℝ)
  (stopover: ℝ)
  (h1: initial_time = 4)
  (h2: initial_speed = 80)
  (h3: reduced_speed = 50)
  (h4: stopover = 0.5) :
  (distance initial_time initial_speed) / reduced_speed + stopover = 6.9 := 
by
  sorry

end travel_time_l432_43261


namespace carl_max_rocks_value_l432_43202

/-- 
Carl finds rocks of three different types:
  - 6-pound rocks worth $18 each.
  - 3-pound rocks worth $9 each.
  - 2-pound rocks worth $3 each.
There are at least 15 rocks available for each type.
Carl can carry at most 20 pounds.

Prove that the maximum value, in dollars, of the rocks Carl can carry out of the cave is $57.
-/
theorem carl_max_rocks_value : 
  (∃ x y z : ℕ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 6 * x + 3 * y + 2 * z ≤ 20 ∧ 18 * x + 9 * y + 3 * z = 57) :=
sorry

end carl_max_rocks_value_l432_43202


namespace not_exists_odd_product_sum_l432_43222

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l432_43222


namespace sin_C_eq_sin_A_minus_B_eq_l432_43243

open Real

-- Problem 1
theorem sin_C_eq (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) :
  sin C = (sqrt 3 + 3 * sqrt 2) / 6 :=
sorry

-- Problem 2
theorem sin_A_minus_B_eq (A B C : ℝ) (a b c : ℝ)
  (h_cosC : cos C = 2 / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) 
  (hA_acute : 0 < A ∧ A < π / 2)
  (hB_acute : 0 < B ∧ B < π / 2) :
  sin (A - B) = -sqrt 5 / 3 :=
sorry

end sin_C_eq_sin_A_minus_B_eq_l432_43243


namespace final_price_of_hat_is_correct_l432_43225

-- Definitions capturing the conditions.
def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

-- Calculations for the intermediate prices.
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

-- The theorem we need to prove.
theorem final_price_of_hat_is_correct : final_price = 9 := by
  sorry

end final_price_of_hat_is_correct_l432_43225


namespace fraction_area_of_triangles_l432_43231

theorem fraction_area_of_triangles 
  (base_PQR : ℝ) (height_PQR : ℝ)
  (base_XYZ : ℝ) (height_XYZ : ℝ)
  (h_base_PQR : base_PQR = 3)
  (h_height_PQR : height_PQR = 2)
  (h_base_XYZ : base_XYZ = 6)
  (h_height_XYZ : height_XYZ = 3) :
  (1/2 * base_PQR * height_PQR) / (1/2 * base_XYZ * height_XYZ) = 1 / 3 :=
by
  sorry

end fraction_area_of_triangles_l432_43231


namespace positive_inequality_l432_43234

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * (a^2 * b^2) / 4 + 9 * a * b > 18 := 
  sorry

end positive_inequality_l432_43234


namespace solve_for_b_l432_43252

theorem solve_for_b (a b c : ℝ) (cosC : ℝ) (h_a : a = 3) (h_c : c = 4) (h_cosC : cosC = -1/4) :
    c^2 = a^2 + b^2 - 2 * a * b * cosC → b = 7 / 2 :=
by 
  intro h_cosine_theorem
  sorry

end solve_for_b_l432_43252


namespace find_relationship_l432_43281

theorem find_relationship (n m : ℕ) (a : ℚ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (n > m ↔ (1 / n < a)) → m = ⌊1 / a⌋ :=
sorry

end find_relationship_l432_43281


namespace range_of_a_l432_43223

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- The theorem we need to prove
theorem range_of_a {a : ℝ} (h : A a ⊆ B) : 1 ≤ a ∧ a < 5 := 
sorry

end range_of_a_l432_43223


namespace min_f_triangle_sides_l432_43258

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x ^ 2, Real.sqrt 3)
  let b := (1, Real.sin (2 * x))
  (a.1 * b.1 + a.2 * b.2) - 2

theorem min_f (x : ℝ) (h1 : -Real.pi / 6 ≤ x) (h2 : x ≤ Real.pi / 3) :
  ∃ x₀, f x₀ = -2 ∧ ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x ≥ -2 :=
  sorry

theorem triangle_sides (a b C : ℝ) (h1 : f C = 1) (h2 : C = Real.pi / 6)
  (h3 : 1 = 1) (h4 : a * b = 2 * Real.sqrt 3) (h5 : a > b) :
  a = 2 ∧ b = Real.sqrt 3 :=
  sorry

end min_f_triangle_sides_l432_43258


namespace pizza_toppings_l432_43237

theorem pizza_toppings :
  ∀ (F V T : ℕ), F = 4 → V = 16 → F * (1 + T) = V → T = 3 :=
by
  intros F V T hF hV h
  sorry

end pizza_toppings_l432_43237


namespace sum_of_squares_remainder_l432_43296

theorem sum_of_squares_remainder (n : ℕ) : 
  ((n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2) % 3 = 2 :=
by
  sorry

end sum_of_squares_remainder_l432_43296


namespace part_a_area_of_square_l432_43216

theorem part_a_area_of_square {s : ℝ} (h : s = 9) : s ^ 2 = 81 := 
sorry

end part_a_area_of_square_l432_43216


namespace norm_of_w_l432_43208

variable (u v : EuclideanSpace ℝ (Fin 2)) 
variable (hu : ‖u‖ = 3) (hv : ‖v‖ = 5) 
variable (h_orthogonal : inner u v = 0)

theorem norm_of_w :
  ‖4 • u - 2 • v‖ = 2 * Real.sqrt 61 := by
  sorry

end norm_of_w_l432_43208


namespace average_speeds_l432_43250

theorem average_speeds (x y : ℝ) (h1 : 4 * x + 5 * y = 98) (h2 : 4 * x = 5 * y - 2) : 
  x = 12 ∧ y = 10 :=
by sorry

end average_speeds_l432_43250


namespace oranges_apples_ratio_l432_43245

variable (A O P : ℕ)
variable (n : ℚ)
variable (h1 : O = n * A)
variable (h2 : P = 4 * O)
variable (h3 : A = (0.08333333333333333 : ℚ) * P)

theorem oranges_apples_ratio (A O P : ℕ) (n : ℚ) 
  (h1 : O = n * A) (h2 : P = 4 * O) (h3 : A = (0.08333333333333333 : ℚ) * P) : n = 3 := 
by
  sorry

end oranges_apples_ratio_l432_43245


namespace saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l432_43284

noncomputable def bread_saving (n_days : ℕ) : ℕ :=
  (1 / 2) * n_days

theorem saving_20_days :
  bread_saving 20 = 10 :=
by
  -- proof steps for bread_saving 20 = 10
  sorry

theorem cost_saving_20_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 20 * cost_per_loaf) = 350 :=
by
  -- proof steps for cost_saving_20_days
  sorry

theorem saving_60_days :
  bread_saving 60 = 30 :=
by
  -- proof steps for bread_saving 60 = 30
  sorry

theorem cost_saving_60_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 60 * cost_per_loaf) = 1050 :=
by
  -- proof steps for cost_saving_60_days
  sorry

end saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l432_43284


namespace seatingArrangementsAreSix_l432_43210

-- Define the number of seating arrangements for 4 people around a round table
def numSeatingArrangements : ℕ :=
  3 * 2 * 1 -- Following the condition that the narrator's position is fixed

-- The main theorem stating the number of different seating arrangements
theorem seatingArrangementsAreSix : numSeatingArrangements = 6 :=
  by
    -- This is equivalent to following the explanation of solution which is just multiplying the numbers
    sorry

end seatingArrangementsAreSix_l432_43210


namespace digits_in_2_pow_120_l432_43285

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l432_43285


namespace abc_sum_eq_11_sqrt_6_l432_43226

variable {a b c : ℝ}

theorem abc_sum_eq_11_sqrt_6 : 
  0 < a → 0 < b → 0 < c → 
  a * b = 36 → 
  a * c = 72 → 
  b * c = 108 → 
  a + b + c = 11 * Real.sqrt 6 :=
by sorry

end abc_sum_eq_11_sqrt_6_l432_43226


namespace parabola_translation_l432_43282

theorem parabola_translation :
  ∀ f g : ℝ → ℝ,
    (∀ x, f x = - (x - 1) ^ 2) →
    (∀ x, g x = f (x - 1) + 2) →
    ∀ x, g x = - (x - 2) ^ 2 + 2 :=
by
  -- Add the proof steps here if needed
  sorry

end parabola_translation_l432_43282


namespace given_conditions_implies_correct_answer_l432_43240

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer_l432_43240


namespace probability_four_vertices_same_plane_proof_l432_43269

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end probability_four_vertices_same_plane_proof_l432_43269


namespace students_not_examined_l432_43270

theorem students_not_examined (boys girls examined : ℕ) (h1 : boys = 121) (h2 : girls = 83) (h3 : examined = 150) : 
  (boys + girls - examined = 54) := by
  sorry

end students_not_examined_l432_43270


namespace sum_of_remainders_correct_l432_43247

def sum_of_remainders : ℕ :=
  let remainders := [43210 % 37, 54321 % 37, 65432 % 37, 76543 % 37, 87654 % 37, 98765 % 37]
  remainders.sum

theorem sum_of_remainders_correct : sum_of_remainders = 36 :=
by sorry

end sum_of_remainders_correct_l432_43247


namespace largest_y_coordinate_l432_43299

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l432_43299


namespace geometry_problem_l432_43293

-- Definitions for points and segments based on given conditions
variables {O A B C D E F G : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G]

-- Lengths of segments based on given conditions
variables (DE EG : ℝ)
variable (BG : ℝ)

-- Given lengths
def given_lengths : Prop :=
  DE = 5 ∧ EG = 3

-- Goal to prove
def goal : Prop :=
  BG = 12

-- The theorem combining conditions and the goal
theorem geometry_problem (h : given_lengths DE EG) : goal BG :=
  sorry

end geometry_problem_l432_43293


namespace trig_identity_problem_l432_43271

theorem trig_identity_problem 
  (t m n k : ℕ) 
  (h_rel_prime : Nat.gcd m n = 1) 
  (h_condition1 : (1 + Real.sin t) * (1 + Real.cos t) = 8 / 9) 
  (h_condition2 : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k) 
  (h_pos_int_m : 0 < m) 
  (h_pos_int_n : 0 < n) 
  (h_pos_int_k : 0 < k) :
  k + m + n = 15 := 
sorry

end trig_identity_problem_l432_43271


namespace decompose_one_into_five_unit_fractions_l432_43230

theorem decompose_one_into_five_unit_fractions :
  1 = (1/2) + (1/3) + (1/7) + (1/43) + (1/1806) :=
by
  sorry

end decompose_one_into_five_unit_fractions_l432_43230


namespace math_problem_l432_43233

-- Definitions for the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 0
def condition2 (a b c : ℝ) : Prop := |a| > |b| ∧ |b| > |c|

-- Theorem statement
theorem math_problem (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : c > 0 ∧ a < 0 :=
by
  sorry

end math_problem_l432_43233


namespace smallest_positive_integer_cube_ends_544_l432_43203

theorem smallest_positive_integer_cube_ends_544 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 544 → m ≥ n :=
by
  sorry

end smallest_positive_integer_cube_ends_544_l432_43203


namespace total_revenue_correct_l432_43207

-- Define the conditions
def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sold_sneakers : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sold_sandals : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.4
def pairs_sold_boots : ℕ := 11

-- Compute discounted prices
def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (original_price * discount)

-- Compute revenue from each type of shoe
def revenue (price : ℝ) (pairs_sold : ℕ) : ℝ :=
  price * (pairs_sold : ℝ)

open Real

-- Main statement to prove
theorem total_revenue_correct : 
  revenue (discounted_price original_price_sneakers discount_sneakers) pairs_sold_sneakers + 
  revenue (discounted_price original_price_sandals discount_sandals) pairs_sold_sandals + 
  revenue (discounted_price original_price_boots discount_boots) pairs_sold_boots = 1068 := 
by
  sorry

end total_revenue_correct_l432_43207


namespace female_students_selected_l432_43236

theorem female_students_selected (males females : ℕ) (p : ℚ) (h_males : males = 28)
  (h_females : females = 21) (h_p : p = 1 / 7) : females * p = 3 := by 
  sorry

end female_students_selected_l432_43236


namespace count_players_studying_chemistry_l432_43219

theorem count_players_studying_chemistry :
  ∀ 
    (total_players : ℕ)
    (math_players : ℕ)
    (physics_players : ℕ)
    (math_and_physics_players : ℕ)
    (all_three_subjects_players : ℕ),
    total_players = 18 →
    math_players = 10 →
    physics_players = 6 →
    math_and_physics_players = 3 →
    all_three_subjects_players = 2 →
    (total_players - (math_players + physics_players - math_and_physics_players)) + all_three_subjects_players = 7 :=
by
  intros total_players math_players physics_players math_and_physics_players all_three_subjects_players
  sorry

end count_players_studying_chemistry_l432_43219


namespace pairs_of_different_positives_l432_43295

def W (x : ℕ) : ℕ := x^4 - 3 * x^3 + 5 * x^2 - 9 * x

theorem pairs_of_different_positives (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (hW : W a = W b) : (a, b) = (1, 2) ∨ (a, b) = (2, 1) := 
sorry

end pairs_of_different_positives_l432_43295


namespace abc_zero_iff_quadratic_identities_l432_43257

variable {a b c : ℝ}

theorem abc_zero_iff_quadratic_identities (h : ¬(a = b ∧ b = c ∧ c = a)) : 
  a + b + c = 0 ↔ a^2 + ab + b^2 = b^2 + bc + c^2 ∧ b^2 + bc + c^2 = c^2 + ca + a^2 :=
by
  sorry

end abc_zero_iff_quadratic_identities_l432_43257
