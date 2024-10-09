import Mathlib

namespace coordinates_of_P_l1862_186209

structure Point (α : Type) [LinearOrderedField α] :=
  (x : α)
  (y : α)

def in_fourth_quadrant {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  P.x > 0 ∧ P.y < 0

def distance_to_axes_is_4 {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  abs P.x = 4 ∧ abs P.y = 4

theorem coordinates_of_P {α : Type} [LinearOrderedField α] (P : Point α) :
  in_fourth_quadrant P ∧ distance_to_axes_is_4 P → P = ⟨4, -4⟩ :=
by
  sorry

end coordinates_of_P_l1862_186209


namespace find_hourly_charge_l1862_186234

variable {x : ℕ}

--Assumptions and conditions
def fixed_charge := 17
def total_paid := 80
def rental_hours := 9

-- Proof problem
theorem find_hourly_charge (h : fixed_charge + rental_hours * x = total_paid) : x = 7 :=
sorry

end find_hourly_charge_l1862_186234


namespace distance_equal_axes_l1862_186217

theorem distance_equal_axes (m : ℝ) :
  (abs (3 * m + 1) = abs (2 * m - 5)) ↔ (m = -6 ∨ m = 4 / 5) :=
by 
  sorry

end distance_equal_axes_l1862_186217


namespace max_value_fraction_l1862_186290

theorem max_value_fraction (a b : ℝ)
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  (a ≠ 0) → (b ≠ 0) →
  ∃ m, m = (a + 2 * b) / (2 * a + b) ∧ m ≤ 7 / 5 :=
by
  sorry

end max_value_fraction_l1862_186290


namespace twelfth_term_arithmetic_sequence_l1862_186297

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l1862_186297


namespace problem_solution_l1862_186243

theorem problem_solution
  (x y : ℝ)
  (h : 5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0) :
  (x - y) ^ 2007 = -1 := by
  sorry

end problem_solution_l1862_186243


namespace time_to_save_for_vehicle_l1862_186296

def monthly_earnings : ℕ := 4000
def saving_factor : ℚ := 1 / 2
def vehicle_cost : ℕ := 16000

theorem time_to_save_for_vehicle : (vehicle_cost / (monthly_earnings * saving_factor)) = 8 := by
  sorry

end time_to_save_for_vehicle_l1862_186296


namespace circle_radius_d_l1862_186256

theorem circle_radius_d (d : ℝ) : ∀ (x y : ℝ), (x^2 + 8 * x + y^2 + 2 * y + d = 0) → (∃ r : ℝ, r = 5) → d = -8 :=
by
  sorry

end circle_radius_d_l1862_186256


namespace total_songs_l1862_186238

theorem total_songs (h : ℕ) (m : ℕ) (a : ℕ) (t : ℕ) (P : ℕ)
  (Hh : h = 6) (Hm : m = 3) (Ha : a = 5) 
  (Htotal : P = (h + m + a + t) / 3) 
  (Hdiv : (h + m + a + t) % 3 = 0) : P = 6 := by
  sorry

end total_songs_l1862_186238


namespace cloth_sales_worth_l1862_186279

theorem cloth_sales_worth 
  (commission : ℝ) 
  (commission_rate : ℝ) 
  (commission_received : ℝ) 
  (commission_rate_of_sales : commission_rate = 2.5)
  (commission_received_rs : commission_received = 21) 
  : (commission_received / (commission_rate / 100)) = 840 :=
by
  sorry

end cloth_sales_worth_l1862_186279


namespace continuity_at_x_2_l1862_186270

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem continuity_at_x_2 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 0 :=
by
  sorry

end continuity_at_x_2_l1862_186270


namespace equivalent_problem_l1862_186298

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem equivalent_problem
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h2 : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 := 
sorry

end equivalent_problem_l1862_186298


namespace max_sin_a_l1862_186277

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l1862_186277


namespace more_girls_than_boys_l1862_186231

theorem more_girls_than_boys (girls boys total_pupils : ℕ) (h1 : girls = 692) (h2 : total_pupils = 926) (h3 : boys = total_pupils - girls) : girls - boys = 458 :=
by
  sorry

end more_girls_than_boys_l1862_186231


namespace cosine_of_angle_in_second_quadrant_l1862_186229

theorem cosine_of_angle_in_second_quadrant
  (α : ℝ)
  (h1 : Real.sin α = 1 / 3)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end cosine_of_angle_in_second_quadrant_l1862_186229


namespace prove_p_l1862_186287

variables {m n p : ℝ}

/-- Given points (m, n) and (m + p, n + 4) lie on the line 
   x = y / 2 - 2 / 5, prove p = 2.
-/
theorem prove_p (hmn : m = n / 2 - 2 / 5)
                (hmpn4 : m + p = (n + 4) / 2 - 2 / 5) : p = 2 := 
by
  sorry

end prove_p_l1862_186287


namespace bricklayer_hours_l1862_186215

theorem bricklayer_hours
  (B E : ℝ)
  (h1 : B + E = 90)
  (h2 : 12 * B + 16 * E = 1350) :
  B = 22.5 :=
by
  sorry

end bricklayer_hours_l1862_186215


namespace simple_interest_rate_l1862_186281

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (hSI : SI = 250) (hP : P = 1500) (hT : T = 5)
  (hSIFormula : SI = (P * R * T) / 100) :
  R = 3.33 := 
by 
  sorry

end simple_interest_rate_l1862_186281


namespace simplify_expr1_simplify_expr2_l1862_186268

noncomputable section

-- Problem 1: Simplify the given expression
theorem simplify_expr1 (a b : ℝ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := 
by sorry

-- Problem 2: Simplify the given expression
theorem simplify_expr2 (x y : ℝ) : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y :=
by sorry

end simplify_expr1_simplify_expr2_l1862_186268


namespace how_many_did_not_play_l1862_186257

def initial_players : ℕ := 40
def first_half_starters : ℕ := 11
def first_half_substitutions : ℕ := 4
def second_half_extra_substitutions : ℕ := (first_half_substitutions * 3) / 4 -- 75% more substitutions
def injury_substitution : ℕ := 1
def total_second_half_substitutions : ℕ := first_half_substitutions + second_half_extra_substitutions + injury_substitution
def total_players_played : ℕ := first_half_starters + first_half_substitutions + total_second_half_substitutions
def players_did_not_play : ℕ := initial_players - total_players_played

theorem how_many_did_not_play : players_did_not_play = 17 := by
  sorry

end how_many_did_not_play_l1862_186257


namespace stormi_additional_money_needed_l1862_186204

noncomputable def earnings_from_jobs : ℝ :=
  let washing_cars := 5 * 8.50
  let walking_dogs := 4 * 6.75
  let mowing_lawns := 3 * 12.25
  let gardening := 2 * 7.40
  washing_cars + walking_dogs + mowing_lawns + gardening

noncomputable def discounted_prices : ℝ :=
  let bicycle := 150.25 * (1 - 0.15)
  let helmet := 35.75 - 5.00
  let lock := 24.50
  bicycle + helmet + lock

noncomputable def total_cost_after_tax : ℝ :=
  let cost_before_tax := discounted_prices
  cost_before_tax * 1.05

noncomputable def amount_needed : ℝ :=
  total_cost_after_tax - earnings_from_jobs

theorem stormi_additional_money_needed : amount_needed = 71.06 := by
  sorry

end stormi_additional_money_needed_l1862_186204


namespace equation_has_real_roots_for_all_K_l1862_186265

open Real

noncomputable def original_equation (K x : ℝ) : ℝ :=
  x - K^3 * (x - 1) * (x - 3)

theorem equation_has_real_roots_for_all_K :
  ∀ K : ℝ, ∃ x : ℝ, original_equation K x = 0 :=
sorry

end equation_has_real_roots_for_all_K_l1862_186265


namespace geom_sequence_a_n_l1862_186206

variable {a : ℕ → ℝ}

-- Given conditions
def is_geom_seq (a : ℕ → ℝ) : Prop :=
  |a 1| = 1 ∧ a 5 = -8 * a 2 ∧ a 5 > a 2

-- Statement to prove
theorem geom_sequence_a_n (h : is_geom_seq a) : ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end geom_sequence_a_n_l1862_186206


namespace factorize_expression_l1862_186212

theorem factorize_expression (x y : ℝ) : 25 * x - x * y ^ 2 = x * (5 + y) * (5 - y) := by
  sorry

end factorize_expression_l1862_186212


namespace inequality_transitive_l1862_186283

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by
  sorry

end inequality_transitive_l1862_186283


namespace parallel_lines_condition_l1862_186269

variable {a : ℝ}

theorem parallel_lines_condition (a_is_2 : a = 2) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → x + y = 1) ∧ (∀ x y : ℝ, x + y = 1 → a * x + 2 * y = 0) :=
by
  sorry

end parallel_lines_condition_l1862_186269


namespace find_P_l1862_186221

noncomputable def P (x : ℝ) : ℝ :=
  4 * x^3 - 6 * x^2 - 12 * x

theorem find_P (a b c : ℝ) (h_root : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_roots : ∀ x, x^3 - 2 * x^2 - 4 * x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c)
  (h_Pa : P a = b + 2 * c)
  (h_Pb : P b = 2 * a + c)
  (h_Pc : P c = a + 2 * b)
  (h_Psum : P (a + b + c) = -20) :
  ∀ x, P x = 4 * x^3 - 6 * x^2 - 12 * x :=
by
  sorry

end find_P_l1862_186221


namespace minimum_filtrations_needed_l1862_186202

theorem minimum_filtrations_needed (I₀ I_n : ℝ) (n : ℕ) (h1 : I₀ = 0.02) (h2 : I_n ≤ 0.001) (h3 : I_n = I₀ * 0.5 ^ n) :
  n = 8 := by
sorry

end minimum_filtrations_needed_l1862_186202


namespace marble_distribution_l1862_186250

theorem marble_distribution (a b c : ℚ) (h1 : a + b + c = 78) (h2 : a = 3 * b + 2) (h3 : b = c / 2) : 
  a = 40 ∧ b = 38 / 3 ∧ c = 76 / 3 :=
by
  sorry

end marble_distribution_l1862_186250


namespace arithmetic_progression_common_difference_and_first_terms_l1862_186253

def sum (n : ℕ) : ℕ := 5 * n ^ 2
def Sn (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_common_difference_and_first_terms:
  ∀ n : ℕ, Sn 5 10 n = sum n :=
by
  sorry

end arithmetic_progression_common_difference_and_first_terms_l1862_186253


namespace function_property_l1862_186251

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l1862_186251


namespace determine_x_l1862_186299

theorem determine_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^3) (h3 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 :=
by
  sorry

end determine_x_l1862_186299


namespace problem_solution_l1862_186201

theorem problem_solution {n : ℕ} :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 :=
by
  sorry

end problem_solution_l1862_186201


namespace total_dogs_l1862_186249

-- Definitions of conditions
def brown_dogs : Nat := 20
def white_dogs : Nat := 10
def black_dogs : Nat := 15

-- Theorem to prove the total number of dogs
theorem total_dogs : brown_dogs + white_dogs + black_dogs = 45 := by
  -- Placeholder for proof
  sorry

end total_dogs_l1862_186249


namespace fair_game_x_value_l1862_186210

theorem fair_game_x_value (x : ℕ) (h : x + 2 * x + 2 * x = 15) : x = 3 := 
by sorry

end fair_game_x_value_l1862_186210


namespace alicia_taxes_l1862_186239

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end alicia_taxes_l1862_186239


namespace batsman_average_46_innings_l1862_186275

variable (A : ℕ) (highest_score : ℕ) (lowest_score : ℕ) (average_excl : ℕ)
variable (n_innings n_without_highest_lowest : ℕ)

theorem batsman_average_46_innings
  (h_diff: highest_score - lowest_score = 190)
  (h_avg_excl: average_excl = 58)
  (h_highest: highest_score = 199)
  (h_innings: n_innings = 46)
  (h_innings_excl: n_without_highest_lowest = 44) :
  A = (44 * 58 + 199 + 9) / 46 := by
  sorry

end batsman_average_46_innings_l1862_186275


namespace fresh_water_needed_l1862_186267

noncomputable def mass_of_seawater : ℝ := 30
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def desired_salt_concentration : ℝ := 0.015

theorem fresh_water_needed :
  ∃ (fresh_water_mass : ℝ), 
    fresh_water_mass = 70 ∧ 
    (mass_of_seawater * initial_salt_concentration) / (mass_of_seawater + fresh_water_mass) = desired_salt_concentration :=
by
  sorry

end fresh_water_needed_l1862_186267


namespace find_a_l1862_186230

theorem find_a (x y a : ℝ) (h1 : x + 2 * y = 2) (h2 : 2 * x + y = a) (h3 : x + y = 5) : a = 13 := by
  sorry

end find_a_l1862_186230


namespace exponent_on_right_side_l1862_186289

theorem exponent_on_right_side (n : ℕ) (k : ℕ) (h : n = 25) :
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^k → k = 26 :=
by
  sorry

end exponent_on_right_side_l1862_186289


namespace orange_juice_production_correct_l1862_186211

noncomputable def orangeJuiceProduction (total_oranges : Float) (export_percent : Float) (juice_percent : Float) : Float :=
  let remaining_oranges := total_oranges * (1 - export_percent / 100)
  let juice_oranges := remaining_oranges * (juice_percent / 100)
  Float.round (juice_oranges * 10) / 10

theorem orange_juice_production_correct :
  orangeJuiceProduction 8.2 30 40 = 2.3 := by
  sorry

end orange_juice_production_correct_l1862_186211


namespace part_I_extreme_value_part_II_range_of_a_l1862_186291

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + Real.log x + 1

theorem part_I_extreme_value (a : ℝ) (h1 : a = -1/4) :
  (∀ x > 0, f a x ≤ f a 2) ∧ f a 2 = 3/4 + Real.log 2 :=
sorry

theorem part_II_range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ x) ↔ a ≤ 0 :=
sorry

end part_I_extreme_value_part_II_range_of_a_l1862_186291


namespace value_of_expression_l1862_186235

theorem value_of_expression : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end value_of_expression_l1862_186235


namespace exists_not_odd_l1862_186222

variable (f : ℝ → ℝ)

-- Define the condition that f is not an odd function
def not_odd_function := ¬ (∀ x : ℝ, f (-x) = -f x)

-- Lean statement to prove the correct answer
theorem exists_not_odd (h : not_odd_function f) : ∃ x : ℝ, f (-x) ≠ -f x :=
sorry

end exists_not_odd_l1862_186222


namespace final_quantity_of_milk_l1862_186225

-- Initially, a vessel is filled with 45 litres of pure milk
def initial_milk : Nat := 45

-- First operation: removing 9 litres of milk and replacing with water
def first_operation_milk(initial_milk : Nat) : Nat := initial_milk - 9
def first_operation_water : Nat := 9

-- Second operation: removing 9 litres of the mixture and replacing with water
def milk_fraction_mixture(milk : Nat) (total : Nat) : Rat := milk / total
def water_fraction_mixture(water : Nat) (total : Nat) : Rat := water / total

def second_operation_milk(milk : Nat) (total : Nat) (removed : Nat) : Rat := 
  milk - (milk_fraction_mixture milk total) * removed
def second_operation_water(water : Nat) (total : Nat) (removed : Nat) : Rat := 
  water - (water_fraction_mixture water total) * removed + removed

-- Prove the final quantity of milk
theorem final_quantity_of_milk : second_operation_milk 36 45 9 = 28.8 := by
  sorry

end final_quantity_of_milk_l1862_186225


namespace solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l1862_186261

-- Definitions as conditions
def is_cone (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_cylinder (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_triangular_pyramid (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_rectangular_prism (solid : Type) : Prop := -- Definition placeholder
sorry 

-- Predicate to check if the front view of a solid is a quadrilateral
def front_view_is_quadrilateral (solid : Type) : Prop :=
  (is_cylinder solid ∨ is_rectangular_prism solid)

-- Theorem stating the problem
theorem solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism
    (s : Type) :
  front_view_is_quadrilateral s ↔ is_cylinder s ∨ is_rectangular_prism s :=
by
  sorry

end solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l1862_186261


namespace eddys_climbing_rate_l1862_186242

def base_camp_ft := 5000
def departure_time := 6 -- in hours: 6:00 AM
def hillary_climbing_rate := 800 -- ft/hr
def stopping_distance_ft := 1000 -- ft short of summit
def hillary_descending_rate := 1000 -- ft/hr
def passing_time := 12 -- in hours: 12:00 PM

theorem eddys_climbing_rate :
  ∀ (base_ft departure hillary_rate stop_dist descend_rate pass_time : ℕ),
    base_ft = base_camp_ft →
    departure = departure_time →
    hillary_rate = hillary_climbing_rate →
    stop_dist = stopping_distance_ft →
    descend_rate = hillary_descending_rate →
    pass_time = passing_time →
    (pass_time - departure) * hillary_rate - descend_rate * (pass_time - (departure + (base_ft - stop_dist) / hillary_rate)) = 6 * 500 :=
by
  intros
  sorry

end eddys_climbing_rate_l1862_186242


namespace cos_product_inequality_l1862_186285

theorem cos_product_inequality : (1 / 8 : ℝ) < (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) ∧
    (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
by
  sorry

end cos_product_inequality_l1862_186285


namespace books_left_after_sale_l1862_186207

theorem books_left_after_sale (initial_books sold_books books_left : ℕ)
    (h1 : initial_books = 33)
    (h2 : sold_books = 26)
    (h3 : books_left = initial_books - sold_books) :
    books_left = 7 := by
  sorry

end books_left_after_sale_l1862_186207


namespace average_value_of_series_l1862_186200

theorem average_value_of_series (z : ℤ) :
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sum_series / n = 21 * z^2 :=
by
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sorry

end average_value_of_series_l1862_186200


namespace sqrt_arithmetic_identity_l1862_186224

theorem sqrt_arithmetic_identity : 4 * (Real.sqrt 2) * (Real.sqrt 3) - (Real.sqrt 12) / (Real.sqrt 2) + (Real.sqrt 24) = 5 * (Real.sqrt 6) := by
  sorry

end sqrt_arithmetic_identity_l1862_186224


namespace sum_le_xyz_plus_two_l1862_186245

theorem sum_le_xyz_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ xyz + 2 := 
sorry

end sum_le_xyz_plus_two_l1862_186245


namespace parallelogram_base_length_l1862_186295

theorem parallelogram_base_length (A : ℕ) (h b : ℕ) (h1 : A = b * h) (h2 : h = 2 * b) (h3 : A = 200) : b = 10 :=
by {
  sorry
}

end parallelogram_base_length_l1862_186295


namespace total_peaches_l1862_186282

variable {n m : ℕ}

-- conditions
def equal_subgroups (n : ℕ) := (n % 3 = 0)

def condition_1 (n m : ℕ) := (m - 27) % n = 0 ∧ (m - 27) / n = 5

def condition_2 (n m : ℕ) : Prop := 
  ∃ x : ℕ, 0 < x ∧ x < 7 ∧ (m - x) % n = 0 ∧ ((m - x) / n = 7) 

-- theorem to be proved
theorem total_peaches (n m : ℕ) (h1 : equal_subgroups n) (h2 : condition_1 n m) (h3 : condition_2 n m) : m = 102 := 
sorry

end total_peaches_l1862_186282


namespace orange_shells_correct_l1862_186286

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l1862_186286


namespace man_owns_fraction_of_business_l1862_186216

theorem man_owns_fraction_of_business
  (x : ℚ)
  (H1 : (3 / 4) * (x * 90000) = 45000)
  (H2 : x * 90000 = y) : 
  x = 2 / 3 := 
by
  sorry

end man_owns_fraction_of_business_l1862_186216


namespace area_of_right_triangle_l1862_186254

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 45) :
  (1 / 2) * (5 * Real.sqrt 2) * (5 * Real.sqrt 2) = 25 :=
by
  -- Proof goes here
  sorry

end area_of_right_triangle_l1862_186254


namespace find_max_m_l1862_186205

-- We define real numbers a, b, c that satisfy the given conditions
variable (a b c m : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 12)
variable (h_prod_sum : a * b + b * c + c * a = 30)
variable (m_def : m = min (a * b) (min (b * c) (c * a)))

-- We state the main theorem to be proved
theorem find_max_m : m ≤ 2 :=
by
  sorry

end find_max_m_l1862_186205


namespace greatest_possible_xy_value_l1862_186246

-- Define the conditions
variables (a b c d x y : ℕ)
variables (h1 : a < b) (h2 : b < c) (h3 : c < d)
variables (sums : Finset ℕ) (hsums : sums = {189, 320, 287, 234, x, y})

-- Define the goal statement to prove
theorem greatest_possible_xy_value : x + y = 791 :=
sorry

end greatest_possible_xy_value_l1862_186246


namespace zero_in_set_l1862_186263

theorem zero_in_set : 0 ∈ ({0, 1, 2} : Set Nat) := 
sorry

end zero_in_set_l1862_186263


namespace sum_of_largest_and_smallest_odd_numbers_is_16_l1862_186255

-- Define odd numbers between 5 and 12
def odd_numbers_set := {n | 5 ≤ n ∧ n ≤ 12 ∧ n % 2 = 1}

-- Define smallest odd number from the set
def min_odd := 5

-- Define largest odd number from the set
def max_odd := 11

-- The main theorem stating that the sum of the smallest and largest odd numbers is 16
theorem sum_of_largest_and_smallest_odd_numbers_is_16 :
  min_odd + max_odd = 16 := by
  sorry

end sum_of_largest_and_smallest_odd_numbers_is_16_l1862_186255


namespace sin_theta_plus_2cos_theta_eq_zero_l1862_186272

theorem sin_theta_plus_2cos_theta_eq_zero (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (1 + Real.sin (2 * θ)) / (Real.cos θ)^2 = 1 :=
  sorry

end sin_theta_plus_2cos_theta_eq_zero_l1862_186272


namespace max_profit_jars_max_tax_value_l1862_186262

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end max_profit_jars_max_tax_value_l1862_186262


namespace number_of_girls_in_school_l1862_186244

-- Variables representing the population and the sample.
variables (total_students sample_size boys_sample girls_sample : ℕ)

-- Initial conditions.
def initial_conditions := 
  total_students = 1600 ∧ 
  sample_size = 200 ∧
  girls_sample = 90 ∧
  boys_sample = 110 ∧
  (girls_sample + 20 = boys_sample)

-- Statement to prove.
theorem number_of_girls_in_school (x: ℕ) 
  (h : initial_conditions total_students sample_size boys_sample girls_sample) :
  x = 720 :=
by {
  -- Obligatory proof omitted.
  sorry
}

end number_of_girls_in_school_l1862_186244


namespace volume_tetrahedral_region_is_correct_l1862_186284

noncomputable def volume_of_tetrahedral_region (a : ℝ) : ℝ :=
  (81 - 8 * Real.pi) * a^3 / 486

theorem volume_tetrahedral_region_is_correct (a : ℝ) :
  volume_of_tetrahedral_region a = (81 - 8 * Real.pi) * a^3 / 486 :=
by
  sorry

end volume_tetrahedral_region_is_correct_l1862_186284


namespace symmetric_point_to_origin_l1862_186237

theorem symmetric_point_to_origin (a b : ℝ) :
  (∃ (a b : ℝ), (a / 2) - 2 * (b / 2) + 2 = 0 ∧ (b / a) * (1 / 2) = -1) →
  (a = -4 / 5 ∧ b = 8 / 5) :=
sorry

end symmetric_point_to_origin_l1862_186237


namespace min_value_frac_inv_l1862_186252

theorem min_value_frac_inv {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (∃ m, (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 2 → m ≤ (1 / x + 1 / y)) ∧ (m = 2)) :=
by
  sorry

end min_value_frac_inv_l1862_186252


namespace trigonometric_identity_l1862_186280

theorem trigonometric_identity :
  (1 / Real.cos (80 * (Real.pi / 180)) - Real.sqrt 3 / Real.sin (80 * (Real.pi / 180)) = 4) :=
by
  sorry

end trigonometric_identity_l1862_186280


namespace range_of_x_l1862_186274

theorem range_of_x (x : ℝ) : (2 : ℝ)^(3 - 2 * x) < (2 : ℝ)^(3 * x - 4) → x > 7 / 5 := by
  sorry

end range_of_x_l1862_186274


namespace santiago_stay_in_australia_l1862_186219

/-- Santiago leaves his home country in the month of January,
    stays in Australia for a few months,
    and returns on the same date in the month of December.
    Prove that Santiago stayed in Australia for 11 months. -/
theorem santiago_stay_in_australia :
  ∃ (months : ℕ), months = 11 ∧
  (months = if (departure_month = 1) ∧ (return_month = 12) then 11 else 0) :=
by sorry

end santiago_stay_in_australia_l1862_186219


namespace simplify_expression_l1862_186232

variable (x y : ℝ)

theorem simplify_expression : 2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 :=
by
  sorry

end simplify_expression_l1862_186232


namespace solve_for_x_l1862_186218

theorem solve_for_x (x : ℝ) (h : 10 - x = 15) : x = -5 :=
by
  sorry

end solve_for_x_l1862_186218


namespace philip_paints_2_per_day_l1862_186264

def paintings_per_day (initial_paintings total_paintings days : ℕ) : ℕ :=
  (total_paintings - initial_paintings) / days

theorem philip_paints_2_per_day :
  paintings_per_day 20 80 30 = 2 :=
by
  sorry

end philip_paints_2_per_day_l1862_186264


namespace find_breadth_of_rectangle_l1862_186293

theorem find_breadth_of_rectangle
  (L R S : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S ^ 2 = 625)
  (A : ℝ := 100)
  (h4 : A = L * B) :
  B = 10 := sorry

end find_breadth_of_rectangle_l1862_186293


namespace unique_pair_exists_l1862_186240

theorem unique_pair_exists (n : ℕ) (hn : n > 0) : 
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ 0 ≤ l ∧ l < k :=
sorry

end unique_pair_exists_l1862_186240


namespace triangle_hypotenuse_and_area_l1862_186233

theorem triangle_hypotenuse_and_area 
  (A B C D : Type) 
  (CD : ℝ) 
  (angle_A : ℝ) 
  (hypotenuse_AC : ℝ) 
  (area_ABC : ℝ) 
  (h1 : CD = 1) 
  (h2 : angle_A = 45) : 
  hypotenuse_AC = Real.sqrt 2 
  ∧ 
  area_ABC = 1 / 2 := 
by
  sorry

end triangle_hypotenuse_and_area_l1862_186233


namespace total_coins_l1862_186292

theorem total_coins (q1 q2 q3 q4 : Nat) (d1 d2 d3 : Nat) (n1 n2 : Nat) (p1 p2 p3 p4 p5 : Nat) :
  q1 = 8 → q2 = 6 → q3 = 7 → q4 = 5 →
  d1 = 7 → d2 = 5 → d3 = 9 →
  n1 = 4 → n2 = 6 →
  p1 = 10 → p2 = 3 → p3 = 8 → p4 = 2 → p5 = 13 →
  q1 + q2 + q3 + q4 + d1 + d2 + d3 + n1 + n2 + p1 + p2 + p3 + p4 + p5 = 93 :=
by
  intros
  sorry

end total_coins_l1862_186292


namespace different_books_l1862_186248

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l1862_186248


namespace variance_of_remaining_scores_l1862_186294

def scores : List ℕ := [91, 89, 91, 96, 94, 95, 94]

def remaining_scores : List ℕ := [91, 91, 94, 95, 94]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by
  sorry

end variance_of_remaining_scores_l1862_186294


namespace value_range_of_f_l1862_186236

noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

theorem value_range_of_f : Set.range (fun x => f x) ∩ Set.Icc 3 6 = Set.Icc 1 4 :=
by
  sorry

end value_range_of_f_l1862_186236


namespace div_fractions_eq_l1862_186227

theorem div_fractions_eq : (3/7) / (5/2) = 6/35 := 
by sorry

end div_fractions_eq_l1862_186227


namespace element_in_set_l1862_186203

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def complement_U_M : Set ℕ := {1, 2}

-- The main statement to prove
theorem element_in_set (M : Set ℕ) (h1 : U = {1, 2, 3, 4, 5}) (h2 : U \ M = complement_U_M) : 3 ∈ M := 
sorry

end element_in_set_l1862_186203


namespace total_number_of_workers_l1862_186271

theorem total_number_of_workers (W N : ℕ) 
    (avg_all : ℝ) 
    (avg_technicians : ℝ) 
    (avg_non_technicians : ℝ)
    (h1 : avg_all = 8000)
    (h2 : avg_technicians = 20000)
    (h3 : avg_non_technicians = 6000)
    (h4 : 7 * avg_technicians + N * avg_non_technicians = (7 + N) * avg_all) :
  W = 49 := by
  sorry

end total_number_of_workers_l1862_186271


namespace tiles_cover_the_floor_l1862_186288

theorem tiles_cover_the_floor
  (n : ℕ)
  (h : 2 * n - 1 = 101)
  : n ^ 2 = 2601 := sorry

end tiles_cover_the_floor_l1862_186288


namespace expected_value_coin_flip_l1862_186213

-- Definitions based on conditions
def P_heads : ℚ := 2 / 3
def P_tails : ℚ := 1 / 3
def win_heads : ℚ := 4
def lose_tails : ℚ := -9

-- Expected value calculation
def expected_value : ℚ :=
  P_heads * win_heads + P_tails * lose_tails

-- Theorem statement to be proven
theorem expected_value_coin_flip : expected_value = -1 / 3 :=
by sorry

end expected_value_coin_flip_l1862_186213


namespace total_amount_l1862_186273

theorem total_amount (A N J : ℕ) (h1 : A = N - 5) (h2 : J = 4 * N) (h3 : J = 48) : A + N + J = 67 :=
by
  -- Proof will be constructed here
  sorry

end total_amount_l1862_186273


namespace possible_last_three_digits_product_l1862_186258

def lastThreeDigits (n : ℕ) : ℕ := n % 1000

theorem possible_last_three_digits_product (a b c : ℕ) (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (a + c) % 10 = b % 10)
  (h3 : (b + c) % 10 = a % 10) :
  lastThreeDigits (a * b * c) = 0 ∨ lastThreeDigits (a * b * c) = 250 ∨ lastThreeDigits (a * b * c) = 500 ∨ lastThreeDigits (a * b * c) = 750 := 
sorry

end possible_last_three_digits_product_l1862_186258


namespace problem_statement_l1862_186223

theorem problem_statement :
  ∃ p q r : ℤ,
    (∀ x : ℝ, (x^2 + 19*x + 88 = (x + p) * (x + q)) ∧ (x^2 - 23*x + 132 = (x - q) * (x - r))) →
      p + q + r = 31 :=
sorry

end problem_statement_l1862_186223


namespace projection_v_w_l1862_186276

noncomputable def vector_v : ℝ × ℝ := (3, 4)
noncomputable def vector_w : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := dot_product u v / dot_product v v
  (scalar * v.1, scalar * v.2)

theorem projection_v_w :
  proj vector_v vector_w = (4/5, -2/5) :=
sorry

end projection_v_w_l1862_186276


namespace arithmetic_sequence_nth_term_l1862_186220

theorem arithmetic_sequence_nth_term (a b c n : ℕ) (x: ℕ)
  (h1: a = 3*x - 4)
  (h2: b = 6*x - 17)
  (h3: c = 4*x + 5)
  (h4: b - a = c - b)
  (h5: a + (n - 1) * (b - a) = 4021) : 
  n = 502 :=
by 
  sorry

end arithmetic_sequence_nth_term_l1862_186220


namespace incorrect_games_less_than_three_fourths_l1862_186208

/-- In a round-robin chess tournament, each participant plays against every other participant exactly once.
A win earns one point, a draw earns half a point, and a loss earns zero points.
We will call a game incorrect if the player who won the game ends up with fewer total points than the player who lost.

1. Prove that incorrect games make up less than 3/4 of the total number of games in the tournament.
2. Prove that in part (1), the number 3/4 cannot be replaced with a smaller number.
--/
theorem incorrect_games_less_than_three_fourths {n : ℕ} (h : n > 1) :
  ∃ m, (∃ (incorrect_games total_games : ℕ), m = incorrect_games ∧ total_games = (n * (n - 1)) / 2 
    ∧ (incorrect_games : ℚ) / total_games < 3 / 4) 
    ∧ (∀ m' : ℚ, m' ≥ 0 → m = incorrect_games ∧ (incorrect_games : ℚ) / total_games < m' → m' ≥ 3 / 4) :=
sorry

end incorrect_games_less_than_three_fourths_l1862_186208


namespace percentage_of_boy_scouts_with_signed_permission_slips_l1862_186278

noncomputable def total_scouts : ℕ := 100 -- assume 100 scouts
noncomputable def total_signed_permission_slips : ℕ := 70 -- 70% of 100
noncomputable def boy_scouts : ℕ := 60 -- 60% of 100
noncomputable def girl_scouts : ℕ := 40 -- total_scouts - boy_scouts 

noncomputable def girl_scouts_signed_permission_slips : ℕ := girl_scouts * 625 / 1000 

theorem percentage_of_boy_scouts_with_signed_permission_slips :
  (boy_scouts * 75 / 100) = (total_signed_permission_slips - girl_scouts_signed_permission_slips) :=
by
  sorry

end percentage_of_boy_scouts_with_signed_permission_slips_l1862_186278


namespace janet_saves_minutes_l1862_186247

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l1862_186247


namespace k5_possibility_l1862_186260

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end k5_possibility_l1862_186260


namespace no_base_450_odd_last_digit_l1862_186266

theorem no_base_450_odd_last_digit :
  ¬ ∃ b : ℕ, b^3 ≤ 450 ∧ 450 < b^4 ∧ (450 % b) % 2 = 1 :=
sorry

end no_base_450_odd_last_digit_l1862_186266


namespace sum_of_triangle_ops_l1862_186228

def triangle_op (a b c : ℕ) : ℕ := 2 * a + b - c 

theorem sum_of_triangle_ops : 
  triangle_op 1 2 3 + triangle_op 4 6 5 + triangle_op 2 7 1 = 20 :=
by
  sorry

end sum_of_triangle_ops_l1862_186228


namespace polygon_interior_angle_sum_l1862_186226

theorem polygon_interior_angle_sum (n : ℕ) (h : (n-1) * 180 = 2400 + 120) : n = 16 :=
by
  sorry

end polygon_interior_angle_sum_l1862_186226


namespace point_P_position_l1862_186241

variable {a b c d : ℝ}
variable (h1: a ≠ b) (h2: a ≠ c) (h3: a ≠ d) (h4: b ≠ c) (h5: b ≠ d) (h6: c ≠ d)

theorem point_P_position (P : ℝ) (hP: b < P ∧ P < c) (hRatio: (|a - P| / |P - d|) = (|b - P| / |P - c|)) : 
  P = (a * c - b * d) / (a - b + c - d) := 
by
  sorry

end point_P_position_l1862_186241


namespace max_wx_plus_xy_plus_yz_l1862_186259

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) (h_sum : w + x + y + z = 200) : wx + xy + yz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_l1862_186259


namespace sequence_sum_l1862_186214

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l1862_186214
