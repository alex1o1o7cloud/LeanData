import Mathlib

namespace NUMINAMATH_GPT_fixed_point_parabola_l1492_149231

theorem fixed_point_parabola (t : ℝ) : 4 * 3^2 + t * 3 - t^2 - 3 * t = 36 := by
  sorry

end NUMINAMATH_GPT_fixed_point_parabola_l1492_149231


namespace NUMINAMATH_GPT_count_unique_lists_of_five_l1492_149232

theorem count_unique_lists_of_five :
  (∃ (f : ℕ → ℕ), ∀ (i j : ℕ), i < j → f (i + 1) - f i = 3 ∧ j = 5 → f 5 % f 1 = 0) →
  (∃ (n : ℕ), n = 6) :=
by
  sorry

end NUMINAMATH_GPT_count_unique_lists_of_five_l1492_149232


namespace NUMINAMATH_GPT_factorization_correct_l1492_149237

noncomputable def factorize_diff_of_squares (a b : ℝ) : ℝ :=
  36 * a * a - 4 * b * b

theorem factorization_correct (a b : ℝ) : factorize_diff_of_squares a b = 4 * (3 * a + b) * (3 * a - b) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1492_149237


namespace NUMINAMATH_GPT_smaller_number_is_25_l1492_149214

theorem smaller_number_is_25 (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_25_l1492_149214


namespace NUMINAMATH_GPT_box_surface_area_l1492_149230

theorem box_surface_area (w l s tab : ℕ):
  w = 40 → l = 60 → s = 8 → tab = 2 →
  (40 * 60 - 4 * 8 * 8 + 2 * (2 * (60 - 2 * 8) + 2 * (40 - 2 * 8))) = 2416 :=
by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_box_surface_area_l1492_149230


namespace NUMINAMATH_GPT_terminal_angle_quadrant_l1492_149251

theorem terminal_angle_quadrant : 
  let angle := -558
  let reduced_angle := angle % 360
  90 < reduced_angle ∧ reduced_angle < 180 →
  SecondQuadrant := 
by 
  intro angle reduced_angle h 
  sorry

end NUMINAMATH_GPT_terminal_angle_quadrant_l1492_149251


namespace NUMINAMATH_GPT_flammable_ice_storage_capacity_l1492_149297

theorem flammable_ice_storage_capacity (billion : ℕ) (h : billion = 10^9) : (800 * billion = 8 * 10^11) :=
by
  sorry

end NUMINAMATH_GPT_flammable_ice_storage_capacity_l1492_149297


namespace NUMINAMATH_GPT_average_score_girls_proof_l1492_149221

noncomputable def average_score_girls_all_schools (A a B b C c : ℕ)
  (adams_boys : ℕ) (adams_girls : ℕ) (adams_comb : ℕ)
  (baker_boys : ℕ) (baker_girls : ℕ) (baker_comb : ℕ)
  (carter_boys : ℕ) (carter_girls : ℕ) (carter_comb : ℕ)
  (all_boys_comb : ℕ) : ℕ :=
  -- Assume number of boys and girls per school A, B, C (boys) and a, b, c (girls)
  if (adams_boys * A + adams_girls * a) / (A + a) = adams_comb ∧
     (baker_boys * B + baker_girls * b) / (B + b) = baker_comb ∧
     (carter_boys * C + carter_girls * c) / (C + c) = carter_comb ∧
     (adams_boys * A + baker_boys * B + carter_boys * C) / (A + B + C) = all_boys_comb
  then (85 * a + 92 * b + 80 * c) / (a + b + c) else 0

theorem average_score_girls_proof (A a B b C c : ℕ)
  (adams_boys : ℕ := 82) (adams_girls : ℕ := 85) (adams_comb : ℕ := 83)
  (baker_boys : ℕ := 87) (baker_girls : ℕ := 92) (baker_comb : ℕ := 91)
  (carter_boys : ℕ := 78) (carter_girls : ℕ := 80) (carter_comb : ℕ := 80)
  (all_boys_comb : ℕ := 84) :
  average_score_girls_all_schools A a B b C c adams_boys adams_girls adams_comb baker_boys baker_girls baker_comb carter_boys carter_girls carter_comb all_boys_comb = 85 :=
by
  sorry

end NUMINAMATH_GPT_average_score_girls_proof_l1492_149221


namespace NUMINAMATH_GPT_average_age_of_John_Mary_Tonya_is_35_l1492_149273

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_John_Mary_Tonya_is_35_l1492_149273


namespace NUMINAMATH_GPT_part_1_part_2_equality_case_l1492_149205

variables {m n : ℝ}

-- Definition of positive real numbers and given condition m > n and n > 1
def conditions_1 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m > n ∧ n > 1

-- Prove that given conditions, m^2 + n > mn + m
theorem part_1 (m n : ℝ) (h : conditions_1 m n) : m^2 + n > m * n + m :=
  by sorry

-- Definition of the condition m + 2n = 1
def conditions_2 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m + 2 * n = 1

-- Prove that given conditions, (2/m) + (1/n) ≥ 8
theorem part_2 (m n : ℝ) (h : conditions_2 m n) : (2 / m) + (1 / n) ≥ 8 :=
  by sorry

-- Prove that the minimum value is obtained when m = 2n = 1/2
theorem equality_case (m n : ℝ) (h : conditions_2 m n) : 
  (2 / m) + (1 / n) = 8 ↔ m = 1/2 ∧ n = 1/4 :=
  by sorry

end NUMINAMATH_GPT_part_1_part_2_equality_case_l1492_149205


namespace NUMINAMATH_GPT_min_value_frac_f1_f_l1492_149212

theorem min_value_frac_f1_f'0 (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_discriminant : b^2 ≤ 4 * a * c) :
  (a + b + c) / b ≥ 2 := 
by
  -- Here goes the proof
  sorry

end NUMINAMATH_GPT_min_value_frac_f1_f_l1492_149212


namespace NUMINAMATH_GPT_tom_paid_correct_amount_l1492_149243

def quantity_of_apples : ℕ := 8
def rate_per_kg_apples : ℕ := 70
def quantity_of_mangoes : ℕ := 9
def rate_per_kg_mangoes : ℕ := 45

def cost_of_apples : ℕ := quantity_of_apples * rate_per_kg_apples
def cost_of_mangoes : ℕ := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid : ℕ := cost_of_apples + cost_of_mangoes

theorem tom_paid_correct_amount :
  total_amount_paid = 965 :=
sorry

end NUMINAMATH_GPT_tom_paid_correct_amount_l1492_149243


namespace NUMINAMATH_GPT_chimney_bricks_l1492_149241

theorem chimney_bricks (x : ℕ) 
  (h1 : Brenda_rate = x / 8) 
  (h2 : Brandon_rate = x / 12) 
  (h3 : Brian_rate = x / 16) 
  (h4 : effective_combined_rate = (Brenda_rate + Brandon_rate + Brian_rate) - 15) 
  (h5 : total_time = 4) :
  (4 * effective_combined_rate) = x := 
  sorry

end NUMINAMATH_GPT_chimney_bricks_l1492_149241


namespace NUMINAMATH_GPT_points_opposite_sides_l1492_149272

theorem points_opposite_sides (x y : ℝ) (h : (3 * x + 2 * y - 8) * (-1) < 0) : 3 * x + 2 * y > 8 := 
by
  sorry

end NUMINAMATH_GPT_points_opposite_sides_l1492_149272


namespace NUMINAMATH_GPT_Xiaoxi_has_largest_final_answer_l1492_149219

def Laura_final : ℕ := 8 - 2 * 3 + 3
def Navin_final : ℕ := (8 * 3) - 2 + 3
def Xiaoxi_final : ℕ := (8 - 2 + 3) * 3

theorem Xiaoxi_has_largest_final_answer : 
  Xiaoxi_final > Laura_final ∧ Xiaoxi_final > Navin_final :=
by
  unfold Laura_final Navin_final Xiaoxi_final
  -- Proof steps would go here, but we skip them as per instructions
  sorry

end NUMINAMATH_GPT_Xiaoxi_has_largest_final_answer_l1492_149219


namespace NUMINAMATH_GPT_baking_trays_used_l1492_149201

-- Let T be the number of baking trays Anna used.
variable (T : ℕ)

-- Condition: Each tray has 20 cupcakes.
def cupcakes_per_tray : ℕ := 20

-- Condition: Each cupcake was sold for $2.
def cupcake_price : ℕ := 2

-- Condition: Only 3/5 of the cupcakes were sold.
def fraction_sold : ℚ := 3 / 5

-- Condition: Anna earned $96 from sold cupcakes.
def earnings : ℕ := 96

-- Derived expressions:
def total_cupcakes (T : ℕ) : ℕ := cupcakes_per_tray * T

def sold_cupcakes (T : ℕ) : ℚ := fraction_sold * total_cupcakes T

def total_earnings (T : ℕ) : ℚ := cupcake_price * sold_cupcakes T

-- The statement to be proved: Given the conditions, the number of trays T must be 4.
theorem baking_trays_used (h : total_earnings T = earnings) : T = 4 := by
  sorry

end NUMINAMATH_GPT_baking_trays_used_l1492_149201


namespace NUMINAMATH_GPT_shara_monthly_payment_l1492_149284

theorem shara_monthly_payment : 
  ∀ (T M : ℕ), 
  (T / 2 = 6 * M) → 
  (T / 2 - 4 * M = 20) → 
  M = 10 :=
by
  intros T M h1 h2
  sorry

end NUMINAMATH_GPT_shara_monthly_payment_l1492_149284


namespace NUMINAMATH_GPT_opposite_of_number_l1492_149293

-- Define the original number
def original_number : ℚ := -1 / 6

-- Statement to prove
theorem opposite_of_number : -original_number = 1 / 6 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_opposite_of_number_l1492_149293


namespace NUMINAMATH_GPT_valid_lineups_count_l1492_149283

-- Definitions of the problem conditions
def num_players : ℕ := 18
def quadruplets : Finset ℕ := {0, 1, 2, 3} -- Indices of Benjamin, Brenda, Brittany, Bryan
def total_starters : ℕ := 8

-- Function to count lineups based on given constraints
noncomputable def count_valid_lineups : ℕ :=
  let others := num_players - quadruplets.card
  Nat.choose others total_starters + quadruplets.card * Nat.choose others (total_starters - 1)

-- The theorem to prove the count of valid lineups
theorem valid_lineups_count : count_valid_lineups = 16731 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_valid_lineups_count_l1492_149283


namespace NUMINAMATH_GPT_rods_in_one_mile_l1492_149259

theorem rods_in_one_mile :
  (1 * 80 * 4 = 320) :=
sorry

end NUMINAMATH_GPT_rods_in_one_mile_l1492_149259


namespace NUMINAMATH_GPT_find_optimal_price_and_units_l1492_149291

noncomputable def price_and_units (x : ℝ) : Prop := 
  let cost_price := 40
  let initial_units := 500
  let profit_goal := 8000
  50 ≤ x ∧ x ≤ 70 ∧ (x - cost_price) * (initial_units - 10 * (x - 50)) = profit_goal

theorem find_optimal_price_and_units : 
  ∃ x units, price_and_units x ∧ units = 500 - 10 * (x - 50) ∧ x = 60 ∧ units = 400 := 
sorry

end NUMINAMATH_GPT_find_optimal_price_and_units_l1492_149291


namespace NUMINAMATH_GPT_part_a_l1492_149246

-- Power tower with 100 twos
def power_tower_100_t2 : ℕ := sorry

theorem part_a : power_tower_100_t2 > 3 := sorry

end NUMINAMATH_GPT_part_a_l1492_149246


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1492_149208

-- Train length in meters
def train_length : ℕ := 130

-- Train speed in km/hr
def train_speed_kmh : ℕ := 45

-- Time to cross bridge in seconds
def time_to_cross_bridge : ℕ := 30

-- Conversion factor from km/hr to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_mps := kmh_to_mps train_speed_kmh

-- Total distance covered by the train in 30 seconds
def total_distance := train_speed_mps * time_to_cross_bridge

-- Length of the bridge
def bridge_length := total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 245 := by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1492_149208


namespace NUMINAMATH_GPT_same_cost_duration_l1492_149206

-- Define the cost function for Plan A
def cost_plan_a (x : ℕ) : ℚ :=
 if x ≤ 8 then 0.60 else 0.60 + 0.06 * (x - 8)

-- Define the cost function for Plan B
def cost_plan_b (x : ℕ) : ℚ :=
 0.08 * x

-- The duration of a call for which the company charges the same under Plan A and Plan B is 14 minutes
theorem same_cost_duration (x : ℕ) : cost_plan_a x = cost_plan_b x ↔ x = 14 :=
by
  -- The proof is not required, using sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_same_cost_duration_l1492_149206


namespace NUMINAMATH_GPT_total_amount_l1492_149238

noncomputable def initial_amounts (a j t : ℕ) := (t = 24)
noncomputable def redistribution_amounts (a j t a' j' t' : ℕ) :=
  a' = 3 * (2 * (a - 2 * j - 24)) ∧
  j' = 3 * (3 * j - (a - 2 * j - 24 + 48)) ∧
  t' = 144 - (6 * (a - 2 * j - 24) + 9 * j - 3 * (a - 2 * j - 24 + 48))

theorem total_amount (a j t a' j' t' : ℕ) (h1 : t = 24)
  (h2 : redistribution_amounts a j t a' j' t')
  (h3 : t' = 24) : 
  a + j + t = 72 :=
sorry

end NUMINAMATH_GPT_total_amount_l1492_149238


namespace NUMINAMATH_GPT_sin_half_angle_identity_l1492_149254

theorem sin_half_angle_identity (theta : ℝ) (h : Real.sin (Real.pi / 2 + theta) = - 1 / 2) :
  2 * Real.sin (theta / 2) ^ 2 - 1 = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_half_angle_identity_l1492_149254


namespace NUMINAMATH_GPT_average_last_three_l1492_149277

noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_last_three (l : List ℝ) (h₁ : l.length = 7) (h₂ : average l = 62) 
  (h₃ : average (l.take 4) = 58) :
  average (l.drop 4) = 202 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_average_last_three_l1492_149277


namespace NUMINAMATH_GPT_find_triangle_l1492_149299

theorem find_triangle (q : ℝ) (triangle : ℝ) (h1 : 3 * triangle * q = 63) (h2 : 7 * (triangle + q) = 161) : triangle = 1 :=
sorry

end NUMINAMATH_GPT_find_triangle_l1492_149299


namespace NUMINAMATH_GPT_difference_in_ages_is_54_l1492_149279

theorem difference_in_ages_is_54 (c d : ℕ) (h1 : 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100) 
    (h2 : 10 * c + d - (10 * d + c) = 9 * (c - d)) 
    (h3 : 10 * c + d + 10 = 3 * (10 * d + c + 10)) : 
    10 * c + d - (10 * d + c) = 54 :=
by
sorry

end NUMINAMATH_GPT_difference_in_ages_is_54_l1492_149279


namespace NUMINAMATH_GPT_find_f_neg1_l1492_149224

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 :
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 < x) → f x = 2 * x * (x + 1)) →
  f (-1) = -4 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_f_neg1_l1492_149224


namespace NUMINAMATH_GPT_max_popsicles_l1492_149294

theorem max_popsicles (total_money : ℝ) (cost_per_popsicle : ℝ) (h_money : total_money = 19.23) (h_cost : cost_per_popsicle = 1.60) : 
  ∃ (x : ℕ), x = ⌊total_money / cost_per_popsicle⌋ ∧ x = 12 :=
by
    sorry

end NUMINAMATH_GPT_max_popsicles_l1492_149294


namespace NUMINAMATH_GPT_B_work_rate_l1492_149213

theorem B_work_rate :
  let A := (1 : ℝ) / 8
  let C := (1 : ℝ) / 4.8
  (A + B + C = 1 / 2) → (B = 1 / 6) :=
by
  intro h
  let A : ℝ := 1 / 8
  let C : ℝ := 1 / 4.8
  let B : ℝ := 1 / 6
  sorry

end NUMINAMATH_GPT_B_work_rate_l1492_149213


namespace NUMINAMATH_GPT_k_value_l1492_149258

theorem k_value (k m : ℤ) (h : (m - 8) ∣ (m^2 - k * m - 24)) : k = 5 := by
  have : (m - 8) ∣ (m^2 - 8 * m - 24) := sorry
  sorry

end NUMINAMATH_GPT_k_value_l1492_149258


namespace NUMINAMATH_GPT_initial_tickets_l1492_149266

theorem initial_tickets (X : ℕ) (h : (X - 22) + 15 = 18) : X = 25 :=
by
  sorry

end NUMINAMATH_GPT_initial_tickets_l1492_149266


namespace NUMINAMATH_GPT_heptagon_diagonals_l1492_149281

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_GPT_heptagon_diagonals_l1492_149281


namespace NUMINAMATH_GPT_new_function_expression_l1492_149260

def initial_function (x : ℝ) : ℝ := -2 * x ^ 2

def shifted_function (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 3

theorem new_function_expression :
  (∀ x : ℝ, (initial_function (x + 1) - 3) = shifted_function x) :=
by
  sorry

end NUMINAMATH_GPT_new_function_expression_l1492_149260


namespace NUMINAMATH_GPT_value_range_of_log_function_l1492_149275

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 4

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

theorem value_range_of_log_function :
  ∀ x : ℝ, log_base_3 (quadratic_function x) ≥ 1 := by
  sorry

end NUMINAMATH_GPT_value_range_of_log_function_l1492_149275


namespace NUMINAMATH_GPT_polygonal_chain_segments_l1492_149211

theorem polygonal_chain_segments (n : ℕ) :
  (∃ (S : Type) (chain : S → Prop), (∃ (closed_non_self_intersecting : S → Prop), 
  (∀ s : S, chain s → closed_non_self_intersecting s) ∧
  ∀ line_segment : S, chain line_segment → 
  (∃ other_segment : S, chain other_segment ∧ line_segment ≠ other_segment))) ↔ 
  (∃ k : ℕ, (n = 2 * k ∧ 5 ≤ k) ∨ (n = 2 * k + 1 ∧ 7 ≤ k)) :=
by sorry

end NUMINAMATH_GPT_polygonal_chain_segments_l1492_149211


namespace NUMINAMATH_GPT_pyramid_height_is_6_l1492_149268

-- Define the conditions for the problem
def square_side_length : ℝ := 18
def pyramid_base_side_length (s : ℝ) : Prop := s * s = (square_side_length / 2) * (square_side_length / 2)
def pyramid_slant_height (s l : ℝ) : Prop := 2 * s * l = square_side_length * square_side_length

-- State the main theorem
theorem pyramid_height_is_6 (s l h : ℝ) (hs : pyramid_base_side_length s) (hl : pyramid_slant_height s l) : h = 6 := 
sorry

end NUMINAMATH_GPT_pyramid_height_is_6_l1492_149268


namespace NUMINAMATH_GPT_farm_problem_l1492_149240

theorem farm_problem
    (initial_cows : ℕ := 12)
    (initial_pigs : ℕ := 34)
    (remaining_animals : ℕ := 30)
    (C : ℕ)
    (P : ℕ)
    (h1 : P = 3 * C)
    (h2 : initial_cows - C + (initial_pigs - P) = remaining_animals) :
    C = 4 :=
by
  sorry

end NUMINAMATH_GPT_farm_problem_l1492_149240


namespace NUMINAMATH_GPT_city_population_distribution_l1492_149289

theorem city_population_distribution :
  (20 + 35) = 55 :=
by
  sorry

end NUMINAMATH_GPT_city_population_distribution_l1492_149289


namespace NUMINAMATH_GPT_minimum_excellence_percentage_l1492_149234

theorem minimum_excellence_percentage (n : ℕ) (h : n = 100)
    (m c b : ℕ) 
    (h_math : m = 70)
    (h_chinese : c = 75) 
    (h_min_both : b = c - (n - m))
    (h_percent : b = 45) :
    b = 45 :=
    sorry

end NUMINAMATH_GPT_minimum_excellence_percentage_l1492_149234


namespace NUMINAMATH_GPT_ring_matching_possible_iff_odd_l1492_149235

theorem ring_matching_possible_iff_odd (n : ℕ) (hn : n ≥ 3) :
  (∃ f : ℕ → ℕ, (∀ k : ℕ, k < n → ∃ j : ℕ, j < n ∧ f (j + k) % n = k % n) ↔ Odd n) :=
sorry

end NUMINAMATH_GPT_ring_matching_possible_iff_odd_l1492_149235


namespace NUMINAMATH_GPT_accounting_majors_count_l1492_149249

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end NUMINAMATH_GPT_accounting_majors_count_l1492_149249


namespace NUMINAMATH_GPT_prob_red_or_blue_l1492_149286

-- Total marbles and given probabilities
def total_marbles : ℕ := 120
def prob_white : ℚ := 1 / 4
def prob_green : ℚ := 1 / 3

-- Problem statement
theorem prob_red_or_blue : (1 - (prob_white + prob_green)) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_prob_red_or_blue_l1492_149286


namespace NUMINAMATH_GPT_num_five_digit_numbers_is_correct_l1492_149298

-- Define the set of digits and their repetition as given in the conditions
def digits : Multiset ℕ := {1, 3, 3, 5, 8}

-- Calculate the permutation with repetitions
noncomputable def num_five_digit_numbers : ℕ := (digits.card.factorial) / 
  (Multiset.count 1 digits).factorial / 
  (Multiset.count 3 digits).factorial / 
  (Multiset.count 5 digits).factorial / 
  (Multiset.count 8 digits).factorial

-- Theorem stating the final result
theorem num_five_digit_numbers_is_correct : num_five_digit_numbers = 60 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_num_five_digit_numbers_is_correct_l1492_149298


namespace NUMINAMATH_GPT_ratio_when_volume_maximized_l1492_149218

-- Definitions based on conditions
def cylinder_perimeter := 24

-- Definition of properties derived from maximizing the volume
def max_volume_height := 4

def max_volume_circumference := 12 - max_volume_height

-- The ratio of the circumference of the cylinder's base to its height when the volume is maximized
def max_volume_ratio := max_volume_circumference / max_volume_height

-- The theorem to be proved
theorem ratio_when_volume_maximized :
  max_volume_ratio = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_when_volume_maximized_l1492_149218


namespace NUMINAMATH_GPT_shortest_distance_between_circles_is_zero_l1492_149203

open Real

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 - 12 * x + y^2 - 8 * y - 12 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + 10 * x + y^2 - 10 * y + 34 = 0

-- Statement of the proof problem: 
-- Prove the shortest distance between the two circles defined by circle1 and circle2 is 0.
theorem shortest_distance_between_circles_is_zero :
    ∀ (x1 y1 x2 y2 : ℝ),
      circle1 x1 y1 →
      circle2 x2 y2 →
      0 = 0 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_is_zero_l1492_149203


namespace NUMINAMATH_GPT_range_of_a_l1492_149223

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ 2 * a - (1 / 2) * a^2) ↔ 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1492_149223


namespace NUMINAMATH_GPT_parabola_standard_equation_l1492_149250

variable {a : ℝ} (h : a < 0)

theorem parabola_standard_equation (h : a < 0) :
  ∃ (p : ℝ), p = -2 * a ∧ (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = 4 * a * x) :=
sorry

end NUMINAMATH_GPT_parabola_standard_equation_l1492_149250


namespace NUMINAMATH_GPT_sum_derivatives_positive_l1492_149265

noncomputable def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6
noncomputable def f' (x : ℝ) : ℝ := -2*x - 4*x^3 - 6*x^5

theorem sum_derivatives_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 < 0) (h2 : x2 + x3 < 0) (h3 : x3 + x1 < 0) :
  f' x1 + f' x2 + f' x3 > 0 := 
sorry

end NUMINAMATH_GPT_sum_derivatives_positive_l1492_149265


namespace NUMINAMATH_GPT_problem_solution_l1492_149285

theorem problem_solution : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1492_149285


namespace NUMINAMATH_GPT_nickel_ate_3_chocolates_l1492_149233

theorem nickel_ate_3_chocolates (R N : ℕ) (h1 : R = 7) (h2 : R = N + 4) : N = 3 := by
  sorry

end NUMINAMATH_GPT_nickel_ate_3_chocolates_l1492_149233


namespace NUMINAMATH_GPT_jessica_monthly_car_insurance_payment_l1492_149244

theorem jessica_monthly_car_insurance_payment
  (rent_last_year : ℤ := 1000)
  (food_last_year : ℤ := 200)
  (car_insurance_last_year : ℤ)
  (rent_increase_rate : ℕ := 3 / 10)
  (food_increase_rate : ℕ := 1 / 2)
  (car_insurance_increase_rate : ℕ := 3)
  (additional_expenses_this_year : ℤ := 7200) :
  car_insurance_last_year = 300 :=
by
  sorry

end NUMINAMATH_GPT_jessica_monthly_car_insurance_payment_l1492_149244


namespace NUMINAMATH_GPT_dogs_in_pet_shop_l1492_149239

variable (D C B : ℕ) (x : ℕ)

theorem dogs_in_pet_shop
  (h1 : D = 3 * x)
  (h2 : C = 7 * x)
  (h3 : B = 12 * x)
  (h4 : D + B = 375) :
  D = 75 :=
by
  sorry

end NUMINAMATH_GPT_dogs_in_pet_shop_l1492_149239


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_parallelism_l1492_149255

-- Define the two lines
def line1 (x y : ℝ) (m : ℝ) : Prop := 2 * x - m * y = 1
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 1) * x - y = 1

-- Define the parallel condition for the two lines
def parallel (m : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 m ∧ line2 x2 y2 m ∧ (2 * m + 1 = 0 ∧ m^2 - m - 2 = 0)) ∨ 
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 2 ∧ line2 x2 y2 2)

theorem sufficient_but_not_necessary_condition_for_parallelism :
  ∀ m, (parallel m) ↔ (m = 2) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_parallelism_l1492_149255


namespace NUMINAMATH_GPT_negation_of_exists_l1492_149292

theorem negation_of_exists (x : ℝ) : ¬(∃ x_0 : ℝ, |x_0| + x_0^2 < 0) ↔ ∀ x : ℝ, |x| + x^2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1492_149292


namespace NUMINAMATH_GPT_unique_int_pair_exists_l1492_149210

theorem unique_int_pair_exists (a b : ℤ) : 
  ∃! (x y : ℤ), (x + 2 * y - a)^2 + (2 * x - y - b)^2 ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_int_pair_exists_l1492_149210


namespace NUMINAMATH_GPT_people_in_the_theater_l1492_149257

theorem people_in_the_theater : ∃ P : ℕ, P = 100 ∧ 
  P = 19 + (1/2 : ℚ) * P + (1/4 : ℚ) * P + 6 := by
  sorry

end NUMINAMATH_GPT_people_in_the_theater_l1492_149257


namespace NUMINAMATH_GPT_calculate_difference_l1492_149228

theorem calculate_difference :
  let a := 3.56
  let b := 2.1
  let c := 1.5
  a - (b * c) = 0.41 :=
by
  let a := 3.56
  let b := 2.1
  let c := 1.5
  show a - (b * c) = 0.41
  sorry

end NUMINAMATH_GPT_calculate_difference_l1492_149228


namespace NUMINAMATH_GPT_trapezoid_proof_l1492_149267

variables {Point : Type} [MetricSpace Point]

-- Definitions of the points and segments as given conditions.
variables (A B C D E : Point)

-- Definitions representing the trapezoid and point E's property.
def is_trapezoid (ABCD : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)) : Prop :=
  (A ≠ B) ∧ (C ≠ D)

def on_segment (E : Point) (A D : Point) : Prop :=
  -- This definition will encompass the fact that E is on segment AD.
  -- Representing the notion that E lies between A and D.
  dist A E + dist E D = dist A D

def equal_perimeters (E : Point) (A B C D : Point) : Prop :=
  let p1 := (dist A B + dist B E + dist E A)
  let p2 := (dist B C + dist C E + dist E B)
  let p3 := (dist C D + dist D E + dist E C)
  p1 = p2 ∧ p2 = p3

-- The theorem we need to prove.
theorem trapezoid_proof (ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (onSeg : on_segment E A D) (eqPerim : equal_perimeters E A B C D) : 
  dist B C = dist A D / 2 :=
sorry

end NUMINAMATH_GPT_trapezoid_proof_l1492_149267


namespace NUMINAMATH_GPT_nth_derivative_ln_correct_l1492_149271

noncomputable def nth_derivative_ln (n : ℕ) : ℝ → ℝ
| x => (-1)^(n-1) * (Nat.factorial (n-1)) / (1 + x) ^ n

theorem nth_derivative_ln_correct (n : ℕ) (x : ℝ) :
  deriv^[n] (λ x => Real.log (1 + x)) x = nth_derivative_ln n x := 
by
  sorry

end NUMINAMATH_GPT_nth_derivative_ln_correct_l1492_149271


namespace NUMINAMATH_GPT_grape_juice_percentage_l1492_149236

theorem grape_juice_percentage
  (initial_volume : ℝ) (initial_percentage : ℝ) (added_juice : ℝ)
  (h_initial_volume : initial_volume = 50)
  (h_initial_percentage : initial_percentage = 0.10)
  (h_added_juice : added_juice = 10) :
  ((initial_percentage * initial_volume + added_juice) / (initial_volume + added_juice) * 100) = 25 := 
by
  sorry

end NUMINAMATH_GPT_grape_juice_percentage_l1492_149236


namespace NUMINAMATH_GPT_number_of_friends_gave_money_l1492_149262

-- Definition of given data in conditions
def amount_per_friend : ℕ := 6
def total_amount : ℕ := 30

-- Theorem to be proved
theorem number_of_friends_gave_money : total_amount / amount_per_friend = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_gave_money_l1492_149262


namespace NUMINAMATH_GPT_average_remaining_two_l1492_149245

theorem average_remaining_two (a b c d e : ℝ) 
  (h1 : (a + b + c + d + e) / 5 = 12) 
  (h2 : (a + b + c) / 3 = 4) : 
  (d + e) / 2 = 24 :=
by 
  sorry

end NUMINAMATH_GPT_average_remaining_two_l1492_149245


namespace NUMINAMATH_GPT_volume_to_surface_area_ratio_l1492_149248

-- Define the structure of the object consisting of unit cubes
structure CubicObject where
  volume : ℕ
  surface_area : ℕ

-- Define a specific cubic object based on given conditions
def specialCubicObject : CubicObject := {
  volume := 8,
  surface_area := 29
}

-- Statement to prove the ratio of the volume to the surface area
theorem volume_to_surface_area_ratio :
  (specialCubicObject.volume : ℚ) / (specialCubicObject.surface_area : ℚ) = 8 / 29 := by
  sorry

end NUMINAMATH_GPT_volume_to_surface_area_ratio_l1492_149248


namespace NUMINAMATH_GPT_num_passed_candidates_l1492_149296

theorem num_passed_candidates
  (total_candidates : ℕ)
  (avg_passed_marks : ℕ)
  (avg_failed_marks : ℕ)
  (overall_avg_marks : ℕ)
  (h1 : total_candidates = 120)
  (h2 : avg_passed_marks = 39)
  (h3 : avg_failed_marks = 15)
  (h4 : overall_avg_marks = 35) :
  ∃ (P : ℕ), P = 100 :=
by
  sorry

end NUMINAMATH_GPT_num_passed_candidates_l1492_149296


namespace NUMINAMATH_GPT_comparison_abc_l1492_149204

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_comparison_abc_l1492_149204


namespace NUMINAMATH_GPT_probability_heads_equals_7_over_11_l1492_149247

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_heads_equals_7_over_11_l1492_149247


namespace NUMINAMATH_GPT_width_of_box_l1492_149252

theorem width_of_box (w : ℝ) (h1 : w > 0) 
    (length : ℝ) (h2 : length = 60) 
    (area_lawn : ℝ) (h3 : area_lawn = 2109) 
    (width_road : ℝ) (h4 : width_road = 3) 
    (crossroads : ℝ) (h5 : crossroads = 2 * (60 / 3 * 3)) :
    60 * w - 120 = 2109 → w = 37.15 := 
by 
  intro h6
  sorry

end NUMINAMATH_GPT_width_of_box_l1492_149252


namespace NUMINAMATH_GPT_quadratic_complete_square_l1492_149220

theorem quadratic_complete_square : ∀ x : ℝ, (x^2 - 8*x - 1) = (x - 4)^2 - 17 :=
by sorry

end NUMINAMATH_GPT_quadratic_complete_square_l1492_149220


namespace NUMINAMATH_GPT_dog_food_duration_l1492_149278

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end NUMINAMATH_GPT_dog_food_duration_l1492_149278


namespace NUMINAMATH_GPT_Alec_goal_ratio_l1492_149209

theorem Alec_goal_ratio (total_students half_votes thinking_votes more_needed fifth_votes : ℕ)
  (h_class : total_students = 60)
  (h_half : half_votes = total_students / 2)
  (remaining_students : ℕ := total_students - half_votes)
  (h_thinking : thinking_votes = 5)
  (h_fifth : fifth_votes = (remaining_students - thinking_votes) / 5)
  (h_current_votes : half_votes + thinking_votes + fifth_votes = 40)
  (h_needed : more_needed = 5)
  :
  (half_votes + thinking_votes + fifth_votes + more_needed) / total_students = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_Alec_goal_ratio_l1492_149209


namespace NUMINAMATH_GPT_simplify_fraction_l1492_149295

-- Define the fraction and the GCD condition
def fraction_numerator : ℕ := 66
def fraction_denominator : ℕ := 4356
def gcd_condition : ℕ := Nat.gcd fraction_numerator fraction_denominator

-- State the theorem that the fraction simplifies to 1/66 given the GCD condition
theorem simplify_fraction (h : gcd_condition = 66) : (fraction_numerator / fraction_denominator = 1 / 66) :=
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1492_149295


namespace NUMINAMATH_GPT_convert_decimal_to_vulgar_fraction_l1492_149269

theorem convert_decimal_to_vulgar_fraction : (32 : ℝ) / 100 = (8 : ℝ) / 25 :=
by
  sorry

end NUMINAMATH_GPT_convert_decimal_to_vulgar_fraction_l1492_149269


namespace NUMINAMATH_GPT_remainder_division_l1492_149270

theorem remainder_division (β : ℂ) 
  (h1 : β^6 + β^5 + β^4 + β^3 + β^2 + β + 1 = 0) 
  (h2 : β^7 = 1) : (β^100 + β^75 + β^50 + β^25 + 1) % (β^6 + β^5 + β^4 + β^3 + β^2 + β + 1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_l1492_149270


namespace NUMINAMATH_GPT_sin_double_angle_l1492_149202

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = -3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1492_149202


namespace NUMINAMATH_GPT_max_height_l1492_149207

-- Given definitions
def height_eq (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 10

def max_height_problem : Prop :=
  ∃ t : ℝ, height_eq t = 74 ∧ ∀ t' : ℝ, height_eq t' ≤ height_eq t

-- Statement of the proof
theorem max_height : max_height_problem := sorry

end NUMINAMATH_GPT_max_height_l1492_149207


namespace NUMINAMATH_GPT_jeans_original_price_l1492_149274

theorem jeans_original_price 
  (discount : ℝ -> ℝ)
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (final_price : ℝ) 
  (customer_payment : ℝ) : 
  discount_percentage = 0.10 -> 
  discount x = x * (1 - discount_percentage) -> 
  final_price = discount (2 * original_price) + original_price -> 
  customer_payment = 112 -> 
  final_price = 112 -> 
  original_price = 40 := 
by
  intros
  sorry

end NUMINAMATH_GPT_jeans_original_price_l1492_149274


namespace NUMINAMATH_GPT_intersection_of_sets_l1492_149261

-- Defining the sets as given in the conditions
def setM : Set ℝ := { x | (x + 1) * (x - 3) ≤ 0 }
def setN : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Statement to prove
theorem intersection_of_sets :
  { x | (x + 1) * (x - 3) ≤ 0 } ∩ { x | 1 < x ∧ x < 4 } = { x | 1 < x ∧ x ≤ 3 } := by
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1492_149261


namespace NUMINAMATH_GPT_regular_decagon_triangle_probability_l1492_149263

theorem regular_decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_triangles := 10
  let probability := favorable_triangles / total_triangles
  probability = (1 : ℚ) / 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_decagon_triangle_probability_l1492_149263


namespace NUMINAMATH_GPT_maximize_profit_l1492_149256

variable {k : ℝ} (hk : k > 0)
variable {x : ℝ} (hx : 0 < x ∧ x < 0.06)

def deposit_volume (x : ℝ) : ℝ := k * x
def interest_paid (x : ℝ) : ℝ := k * x ^ 2
def profit (x : ℝ) : ℝ := (0.06 * k^2 * x) - (k * x^2)

theorem maximize_profit : 0.03 = x :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1492_149256


namespace NUMINAMATH_GPT_find_x_value_l1492_149215

-- Definitions based on the conditions
def varies_inversely_as_square (k : ℝ) (x y : ℝ) : Prop := x = k / y^2

def given_condition (k : ℝ) : Prop := 1 = k / 3^2

-- The main proof problem to solve
theorem find_x_value (k : ℝ) (y : ℝ) (h1 : varies_inversely_as_square k 1 3) (h2 : y = 9) : 
  varies_inversely_as_square k (1/9) y :=
sorry

end NUMINAMATH_GPT_find_x_value_l1492_149215


namespace NUMINAMATH_GPT_value_of_a_plus_b_minus_c_l1492_149229

def a : ℤ := 1 -- smallest positive integer
def b : ℤ := 0 -- number with the smallest absolute value
def c : ℤ := -1 -- largest negative integer

theorem value_of_a_plus_b_minus_c : a + b - c = 2 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_minus_c_l1492_149229


namespace NUMINAMATH_GPT_initial_cakes_count_l1492_149242

theorem initial_cakes_count (f : ℕ) (a b : ℕ) 
  (condition1 : f = 5)
  (condition2 : ∀ i, i ∈ Finset.range f → a = 4)
  (condition3 : ∀ i, i ∈ Finset.range f → b = 20 / 2)
  (condition4 : f * a = 2 * b) : 
  b = 40 := 
by
  sorry

end NUMINAMATH_GPT_initial_cakes_count_l1492_149242


namespace NUMINAMATH_GPT_school_club_profit_l1492_149216

theorem school_club_profit :
  let pencils := 1200
  let buy_rate := 4 / 3 -- pencils per dollar
  let sell_rate := 5 / 4 -- pencils per dollar
  let cost_per_pencil := 3 / 4 -- dollars per pencil
  let sell_per_pencil := 4 / 5 -- dollars per pencil
  let cost := pencils * cost_per_pencil
  let revenue := pencils * sell_per_pencil
  let profit := revenue - cost
  profit = 60 := 
by
  sorry

end NUMINAMATH_GPT_school_club_profit_l1492_149216


namespace NUMINAMATH_GPT_car_gas_cost_l1492_149253

def car_mpg_city : ℝ := 30
def car_mpg_highway : ℝ := 40
def city_distance_one_way : ℝ := 60
def highway_distance_one_way : ℝ := 200
def gas_cost_per_gallon : ℝ := 3
def total_gas_cost : ℝ := 42

theorem car_gas_cost :
  (city_distance_one_way / car_mpg_city * 2 + highway_distance_one_way / car_mpg_highway * 2) * gas_cost_per_gallon = total_gas_cost := 
  sorry

end NUMINAMATH_GPT_car_gas_cost_l1492_149253


namespace NUMINAMATH_GPT_addends_are_negative_l1492_149217

theorem addends_are_negative (a b : ℤ) (h1 : a + b < a) (h2 : a + b < b) : a < 0 ∧ b < 0 := 
sorry

end NUMINAMATH_GPT_addends_are_negative_l1492_149217


namespace NUMINAMATH_GPT_sin_A_is_eight_ninths_l1492_149280

variable (AB AC : ℝ) (A : ℝ)

-- Given conditions
def area_triangle := 1 / 2 * AB * AC * Real.sin A = 100
def geometric_mean := Real.sqrt (AB * AC) = 15

-- Proof statement
theorem sin_A_is_eight_ninths (h1 : area_triangle AB AC A) (h2 : geometric_mean AB AC) :
  Real.sin A = 8 / 9 := sorry

end NUMINAMATH_GPT_sin_A_is_eight_ninths_l1492_149280


namespace NUMINAMATH_GPT_chris_and_fiona_weight_l1492_149264

theorem chris_and_fiona_weight (c d e f : ℕ) (h1 : c + d = 330) (h2 : d + e = 290) (h3 : e + f = 310) : c + f = 350 :=
by
  sorry

end NUMINAMATH_GPT_chris_and_fiona_weight_l1492_149264


namespace NUMINAMATH_GPT_sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l1492_149288

theorem sqrt5_times_sqrt6_minus_1_over_sqrt5_bound :
  4 < (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) ∧ (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) < 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l1492_149288


namespace NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_l1492_149222

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + m = 0 ∧ x₂^2 + 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_l1492_149222


namespace NUMINAMATH_GPT_xiao_ming_total_score_l1492_149287

-- Definitions for the given conditions
def score_regular : ℝ := 70
def score_midterm : ℝ := 80
def score_final : ℝ := 85

def weight_regular : ℝ := 0.3
def weight_midterm : ℝ := 0.3
def weight_final : ℝ := 0.4

-- The statement that we need to prove
theorem xiao_ming_total_score : 
  (score_regular * weight_regular) + (score_midterm * weight_midterm) + (score_final * weight_final) = 79 := 
by
  sorry

end NUMINAMATH_GPT_xiao_ming_total_score_l1492_149287


namespace NUMINAMATH_GPT_problem_statement_l1492_149225

open Set

noncomputable def U := ℝ

def A : Set ℝ := { x | 0 < 2 * x + 4 ∧ 2 * x + 4 < 10 }
def B : Set ℝ := { x | x < -4 ∨ x > 2 }
def C (a : ℝ) (h : a < 0) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem problem_statement (a : ℝ) (ha : a < 0) :
    A ∪ B = { x | x < -4 ∨ x > -2 } ∧
    compl (A ∪ B) ⊆ C a ha → -2 < a ∧ a < -4 / 3 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1492_149225


namespace NUMINAMATH_GPT_count_jianzhan_count_gift_boxes_l1492_149227

-- Definitions based on given conditions
def firewood_red_clay : Int := 90
def firewood_white_clay : Int := 60
def electric_red_clay : Int := 75
def electric_white_clay : Int := 75
def total_red_clay : Int := 1530
def total_white_clay : Int := 1170

-- Proof problem 1: Number of "firewood firing" and "electric firing" Jianzhan produced
theorem count_jianzhan (x y : Int) (hx : firewood_red_clay * x + electric_red_clay * y = total_red_clay)
  (hy : firewood_white_clay * x + electric_white_clay * y = total_white_clay) : 
  x = 12 ∧ y = 6 :=
sorry

-- Definitions based on given conditions for Part 2
def total_jianzhan : Int := 18
def box_a_capacity : Int := 2
def box_b_capacity : Int := 6

-- Proof problem 2: Number of purchasing plans for gift boxes
theorem count_gift_boxes (m n : Int) (h : box_a_capacity * m + box_b_capacity * n = total_jianzhan) : 
  ∃ s : Finset (Int × Int), s.card = 4 ∧ ∀ (p : Int × Int), p ∈ s ↔ (p = (9, 0) ∨ p = (6, 1) ∨ p = (3, 2) ∨ p = (0, 3)) :=
sorry

end NUMINAMATH_GPT_count_jianzhan_count_gift_boxes_l1492_149227


namespace NUMINAMATH_GPT_arithmetic_sequence_a18_value_l1492_149290

theorem arithmetic_sequence_a18_value 
  (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom : (a 5 + 1) ^ 2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a18_value_l1492_149290


namespace NUMINAMATH_GPT_ann_susan_age_sum_l1492_149200

theorem ann_susan_age_sum (ann_age : ℕ) (susan_age : ℕ) (h1 : ann_age = 16) (h2 : ann_age = susan_age + 5) : ann_age + susan_age = 27 :=
by
  sorry

end NUMINAMATH_GPT_ann_susan_age_sum_l1492_149200


namespace NUMINAMATH_GPT_campers_morning_count_l1492_149276

theorem campers_morning_count (afternoon_count : ℕ) (additional_morning : ℕ) (h1 : afternoon_count = 39) (h2 : additional_morning = 5) :
  afternoon_count + additional_morning = 44 :=
by
  sorry

end NUMINAMATH_GPT_campers_morning_count_l1492_149276


namespace NUMINAMATH_GPT_rectangle_length_fraction_of_circle_radius_l1492_149226

noncomputable def square_side (area : ℕ) : ℕ :=
  Nat.sqrt area

noncomputable def rectangle_length (breadth area : ℕ) : ℕ :=
  area / breadth

theorem rectangle_length_fraction_of_circle_radius
  (square_area : ℕ)
  (rectangle_breadth : ℕ)
  (rectangle_area : ℕ)
  (side := square_side square_area)
  (radius := side)
  (length := rectangle_length rectangle_breadth rectangle_area) :
  square_area = 4761 →
  rectangle_breadth = 13 →
  rectangle_area = 598 →
  length / radius = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_rectangle_length_fraction_of_circle_radius_l1492_149226


namespace NUMINAMATH_GPT_cone_base_area_l1492_149282

theorem cone_base_area (r l : ℝ) (h1 : (1/2) * π * l^2 = 2 * π) (h2 : 2 * π * r = 2 * π) :
  π * r^2 = π :=
by 
  sorry

end NUMINAMATH_GPT_cone_base_area_l1492_149282
