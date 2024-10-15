import Mathlib

namespace NUMINAMATH_GPT_jiwon_walk_distance_l1748_174802

theorem jiwon_walk_distance : 
  (13 * 90) * 0.45 = 526.5 := by
  sorry

end NUMINAMATH_GPT_jiwon_walk_distance_l1748_174802


namespace NUMINAMATH_GPT_curtain_price_l1748_174878

theorem curtain_price
  (C : ℝ)
  (h1 : 2 * C + 9 * 15 + 50 = 245) :
  C = 30 :=
sorry

end NUMINAMATH_GPT_curtain_price_l1748_174878


namespace NUMINAMATH_GPT_train_length_l1748_174827

theorem train_length (L V : ℝ) (h1 : L = V * 26) (h2 : L + 150 = V * 39) : L = 300 := by
  sorry

end NUMINAMATH_GPT_train_length_l1748_174827


namespace NUMINAMATH_GPT_remaining_laps_l1748_174896

theorem remaining_laps (total_laps_friday : ℕ)
                       (total_laps_saturday : ℕ)
                       (laps_sunday_morning : ℕ)
                       (total_required_laps : ℕ)
                       (total_laps_weekend : ℕ)
                       (remaining_laps : ℕ) :
  total_laps_friday = 63 →
  total_laps_saturday = 62 →
  laps_sunday_morning = 15 →
  total_required_laps = 198 →
  total_laps_weekend = total_laps_friday + total_laps_saturday + laps_sunday_morning →
  remaining_laps = total_required_laps - total_laps_weekend →
  remaining_laps = 58 := by
  intros
  sorry

end NUMINAMATH_GPT_remaining_laps_l1748_174896


namespace NUMINAMATH_GPT_profit_eqn_65_to_75_maximize_profit_with_discount_l1748_174838

-- Definitions for the conditions
def total_pieces (x y : ℕ) : Prop := x + y = 100

def total_cost (x y : ℕ) : Prop := 80 * x + 60 * y ≤ 7500

def min_pieces_A (x : ℕ) : Prop := x ≥ 65

def profit_without_discount (x : ℕ) : ℕ := 10 * x + 3000

def profit_with_discount (x a : ℕ) (h1 : 0 < a) (h2 : a < 20): ℕ := (10 - a) * x + 3000

-- Proof statement
theorem profit_eqn_65_to_75 (x: ℕ) (h1: total_pieces x (100 - x)) (h2: total_cost x (100 - x)) (h3: min_pieces_A x) :
  65 ≤ x ∧ x ≤ 75 → profit_without_discount x = 10 * x + 3000 :=
by
  sorry

theorem maximize_profit_with_discount (x a : ℕ) (h1 : total_pieces x (100 - x)) (h2 : total_cost x (100 - x)) (h3 : min_pieces_A x) (h4 : 0 < a) (h5 : a < 20) :
  if a < 10 then x = 75 ∧ profit_with_discount 75 a h4 h5 = (10 - a) * 75 + 3000
  else if a = 10 then 65 ≤ x ∧ x ≤ 75 ∧ profit_with_discount x a h4 h5 = 3000
  else x = 65 ∧ profit_with_discount 65 a h4 h5 = (10 - a) * 65 + 3000 :=
by
  sorry

end NUMINAMATH_GPT_profit_eqn_65_to_75_maximize_profit_with_discount_l1748_174838


namespace NUMINAMATH_GPT_total_bees_approx_l1748_174816

-- Define a rectangular garden with given width and length
def garden_width : ℝ := 450
def garden_length : ℝ := 550

-- Define the average density of bees per square foot
def bee_density : ℝ := 2.5

-- Define the area of the garden in square feet
def garden_area : ℝ := garden_width * garden_length

-- Define the total number of bees in the garden
def total_bees : ℝ := bee_density * garden_area

-- Prove that the total number of bees approximately equals 620,000
theorem total_bees_approx : abs (total_bees - 620000) < 1000 :=
by
  sorry

end NUMINAMATH_GPT_total_bees_approx_l1748_174816


namespace NUMINAMATH_GPT_lcm_gcd_product_l1748_174893

theorem lcm_gcd_product (n m : ℕ) (h1 : n = 9) (h2 : m = 10) : 
  Nat.lcm n m * Nat.gcd n m = 90 := by
  sorry

end NUMINAMATH_GPT_lcm_gcd_product_l1748_174893


namespace NUMINAMATH_GPT_fresh_grapes_weight_l1748_174803

/-- Given fresh grapes containing 90% water by weight, 
    and dried grapes containing 20% water by weight,
    if the weight of dried grapes obtained from a certain amount of fresh grapes is 2.5 kg,
    then the weight of the fresh grapes used is 20 kg.
-/
theorem fresh_grapes_weight (F D : ℝ)
  (hD : D = 2.5)
  (fresh_water_content : ℝ := 0.90)
  (dried_water_content : ℝ := 0.20)
  (fresh_solid_content : ℝ := 1 - fresh_water_content)
  (dried_solid_content : ℝ := 1 - dried_water_content)
  (solid_mass_constancy : fresh_solid_content * F = dried_solid_content * D) : 
  F = 20 := 
  sorry

end NUMINAMATH_GPT_fresh_grapes_weight_l1748_174803


namespace NUMINAMATH_GPT_minimum_value_l1748_174886

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ min_value_expr x y :=
  sorry

end NUMINAMATH_GPT_minimum_value_l1748_174886


namespace NUMINAMATH_GPT_relationship_of_y_values_l1748_174881

theorem relationship_of_y_values 
  (k : ℝ) (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_pos : k > 0) 
  (hA : y1 = k / x1) 
  (hB : y2 = k / x2) 
  (hC : y3 = k / x3) 
  (h_order : x1 < 0 ∧ 0 < x2 ∧ x2 < x3) : y1 < y3 ∧ y3 < y2 := 
by
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l1748_174881


namespace NUMINAMATH_GPT_sum_of_ages_l1748_174854

theorem sum_of_ages (Petra_age : ℕ) (Mother_age : ℕ)
  (h_petra : Petra_age = 11)
  (h_mother : Mother_age = 36) :
  Petra_age + Mother_age = 47 :=
by
  -- Using the given conditions:
  -- Petra_age = 11
  -- Mother_age = 36
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1748_174854


namespace NUMINAMATH_GPT_part1_part2_l1748_174889

def setA (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def setB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem part1 (m : ℝ) (h : m = 1) : 
  {x | x ∈ setA m} ∩ {x | x ∈ setB} = {x | 3 ≤ x ∧ x < 4} :=
by {
  sorry
}

theorem part2 (m : ℝ): 
  ({x | x ∈ setA m} ∪ {x | x ∈ setB} = {x | x ∈ setB}) ↔ (m ≥ 3 ∨ m ≤ -3) :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1748_174889


namespace NUMINAMATH_GPT_train_crossing_time_l1748_174830

theorem train_crossing_time
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ)
  (train_length_eq : length_train = 720)
  (bridge_length_eq : length_bridge = 320)
  (speed_eq : speed_kmh = 90) :
  (length_train + length_bridge) / (speed_kmh * (1000 / 3600)) = 41.6 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1748_174830


namespace NUMINAMATH_GPT_find_a4_l1748_174835

def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem find_a4 (a₁ d : ℤ) (S₅ S₉ : ℤ) 
  (h₁ : arithmetic_sequence_sum 5 a₁ d = 35)
  (h₂ : arithmetic_sequence_sum 9 a₁ d = 117) :
  (a₁ + 3 * d) = 20 := 
sorry

end NUMINAMATH_GPT_find_a4_l1748_174835


namespace NUMINAMATH_GPT_cistern_capacity_l1748_174829

theorem cistern_capacity (C : ℝ) (h1 : C / 20 > 0) (h2 : C / 24 > 0) (h3 : 4 - C / 20 = C / 24) : C = 480 / 11 :=
by sorry

end NUMINAMATH_GPT_cistern_capacity_l1748_174829


namespace NUMINAMATH_GPT_log_abs_is_even_l1748_174864

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

noncomputable def f (x : ℝ) : ℝ := log (abs x)

theorem log_abs_is_even : is_even_function f :=
by
  sorry

end NUMINAMATH_GPT_log_abs_is_even_l1748_174864


namespace NUMINAMATH_GPT_inequality_neg_multiply_l1748_174809

theorem inequality_neg_multiply {a b : ℝ} (h : a > b) : -2 * a < -2 * b :=
sorry

end NUMINAMATH_GPT_inequality_neg_multiply_l1748_174809


namespace NUMINAMATH_GPT_systematic_sampling_eighth_group_number_l1748_174899

theorem systematic_sampling_eighth_group_number (total_students groups students_per_group draw_lots_first : ℕ) 
  (h_total : total_students = 480)
  (h_groups : groups = 30)
  (h_students_per_group : students_per_group = 16)
  (h_draw_lots_first : draw_lots_first = 5) : 
  (8 - 1) * students_per_group + draw_lots_first = 117 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_eighth_group_number_l1748_174899


namespace NUMINAMATH_GPT_range_of_a_l1748_174877

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := -(x + 1)^2 + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f x2 ≤ g x1 a) ↔ a ≥ -1 / Real.exp 1 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_range_of_a_l1748_174877


namespace NUMINAMATH_GPT_profit_ratio_l1748_174837

theorem profit_ratio (SP CP : ℝ) (h : SP / CP = 3) : (SP - CP) / CP = 2 :=
by
  sorry

end NUMINAMATH_GPT_profit_ratio_l1748_174837


namespace NUMINAMATH_GPT_determine_ratio_l1748_174817

-- Definition of the given conditions.
def total_length : ℕ := 69
def longer_length : ℕ := 46
def ratio_of_lengths (shorter_length longer_length : ℕ) : ℕ := longer_length / shorter_length

-- The theorem we need to prove.
theorem determine_ratio (x : ℕ) (m : ℕ) (h1 : longer_length = m * x) (h2 : x + longer_length = total_length) : 
  ratio_of_lengths x longer_length = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_ratio_l1748_174817


namespace NUMINAMATH_GPT_sector_area_l1748_174857

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = π / 3) (hr : r = 6) : 
  1/2 * r^2 * α = 6 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_sector_area_l1748_174857


namespace NUMINAMATH_GPT_sum_of_integers_l1748_174805

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1748_174805


namespace NUMINAMATH_GPT_monkey_climb_time_l1748_174892

theorem monkey_climb_time : 
  ∀ (height hop slip : ℕ), 
    height = 22 ∧ hop = 3 ∧ slip = 2 → 
    ∃ (time : ℕ), time = 20 := 
by
  intros height hop slip h
  rcases h with ⟨h_height, ⟨h_hop, h_slip⟩⟩
  sorry

end NUMINAMATH_GPT_monkey_climb_time_l1748_174892


namespace NUMINAMATH_GPT_original_contribution_amount_l1748_174862

theorem original_contribution_amount (F : ℕ) (N : ℕ) (C : ℕ) (A : ℕ) 
  (hF : F = 14) (hN : N = 19) (hC : C = 4) : A = 90 :=
by 
  sorry

end NUMINAMATH_GPT_original_contribution_amount_l1748_174862


namespace NUMINAMATH_GPT_bird_families_flew_away_to_Asia_l1748_174863

-- Defining the given conditions
def Total_bird_families_flew_away_for_winter : ℕ := 118
def Bird_families_flew_away_to_Africa : ℕ := 38

-- Proving the main statement
theorem bird_families_flew_away_to_Asia : 
  (Total_bird_families_flew_away_for_winter - Bird_families_flew_away_to_Africa) = 80 :=
by
  sorry

end NUMINAMATH_GPT_bird_families_flew_away_to_Asia_l1748_174863


namespace NUMINAMATH_GPT_fixed_point_through_ellipse_l1748_174808

-- Define the ellipse and the points
def C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def P2 : ℝ × ℝ := (0, 1)

-- Define the condition for a line not passing through P2 and intersecting the ellipse
def line_l_intersects_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ (x1 x2 b k : ℝ), l (x1, k * x1 + b) ∧ l (x2, k * x2 + b) ∧
  (C x1 (k * x1 + b)) ∧ (C x2 (k * x2 + b)) ∧
  ((x1, k * x1 + b) ≠ P2 ∧ (x2, k * x2 + b) ≠ P2) ∧
  ((k * x1 + b ≠ 1) ∧ (k * x2 + b ≠ 1)) ∧ 
  (∃ (kA kB : ℝ), kA = (k * x1 + b - 1) / x1 ∧ kB = (k * x2 + b - 1) / x2 ∧ kA + kB = -1)

-- Prove there exists a fixed point (2, -1) through which all such lines must pass
theorem fixed_point_through_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_l_intersects_ellipse A B l → l (2, -1) :=
sorry

end NUMINAMATH_GPT_fixed_point_through_ellipse_l1748_174808


namespace NUMINAMATH_GPT_roy_age_product_l1748_174855

theorem roy_age_product (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R = K + (R - J) / 2)
  (h3 : R + 2 = 3 * (J + 2)) :
  (R + 2) * (K + 2) = 96 :=
by
  sorry

end NUMINAMATH_GPT_roy_age_product_l1748_174855


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1748_174810

theorem simplify_and_evaluate (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2 * x^2 = -6 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_and_evaluate_l1748_174810


namespace NUMINAMATH_GPT_balloon_difference_l1748_174823

theorem balloon_difference (your_balloons : ℕ) (friend_balloons : ℕ) (h1 : your_balloons = 7) (h2 : friend_balloons = 5) : your_balloons - friend_balloons = 2 :=
by
  sorry

end NUMINAMATH_GPT_balloon_difference_l1748_174823


namespace NUMINAMATH_GPT_rope_in_two_months_period_l1748_174866

theorem rope_in_two_months_period :
  let week1 := 6
  let week2 := 3 * week1
  let week3 := week2 - 4
  let week4 := - (week2 / 2)
  let week5 := week1 + 2
  let week6 := - (2 / 2)
  let week7 := 3 * (2 / 2)
  let week8 := - 10
  let total_length := (week1 + week2 + week3 + week4 + week5 + week6 + week7 + week8)
  total_length * 12 = 348
:= sorry

end NUMINAMATH_GPT_rope_in_two_months_period_l1748_174866


namespace NUMINAMATH_GPT_steven_apples_minus_peaches_l1748_174811

-- Define the number of apples and peaches Steven has.
def steven_apples : ℕ := 19
def steven_peaches : ℕ := 15

-- Problem statement: Prove that the number of apples minus the number of peaches is 4.
theorem steven_apples_minus_peaches : steven_apples - steven_peaches = 4 := by
  sorry

end NUMINAMATH_GPT_steven_apples_minus_peaches_l1748_174811


namespace NUMINAMATH_GPT_find_fraction_l1748_174869

theorem find_fraction
  (x : ℝ)
  (h : (x)^35 * (1/4)^18 = 1 / (2 * 10^35)) : x = 1/5 :=
by 
  sorry

end NUMINAMATH_GPT_find_fraction_l1748_174869


namespace NUMINAMATH_GPT_new_person_weight_l1748_174825

theorem new_person_weight (weights : List ℝ) (len_weights : weights.length = 8) (replace_weight : ℝ) (new_weight : ℝ)
  (weight_diff :  (weights.sum - replace_weight + new_weight) / 8 = (weights.sum / 8) + 3) 
  (replace_weight_eq : replace_weight = 70):
  new_weight = 94 :=
sorry

end NUMINAMATH_GPT_new_person_weight_l1748_174825


namespace NUMINAMATH_GPT_original_price_is_975_l1748_174880

variable (x : ℝ)
variable (discounted_price : ℝ := 780)
variable (discount : ℝ := 0.20)

-- The condition that Smith bought the shirt for Rs. 780 after a 20% discount
def original_price_calculation (x : ℝ) (discounted_price : ℝ) (discount : ℝ) : Prop :=
  (1 - discount) * x = discounted_price

theorem original_price_is_975 : ∃ x : ℝ, original_price_calculation x 780 0.20 ∧ x = 975 := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_original_price_is_975_l1748_174880


namespace NUMINAMATH_GPT_percentage_commute_l1748_174839

variable (x : Real)
variable (h : 0.20 * 0.10 * x = 12)

theorem percentage_commute :
  0.10 * 0.20 * x = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_commute_l1748_174839


namespace NUMINAMATH_GPT_true_discount_l1748_174807

theorem true_discount (BD PV TD : ℝ) (h1 : BD = 36) (h2 : PV = 180) :
  TD = 30 :=
by
  sorry

end NUMINAMATH_GPT_true_discount_l1748_174807


namespace NUMINAMATH_GPT_solve_cubic_equation_l1748_174876

variable (t : ℝ)

theorem solve_cubic_equation (x : ℝ) :
  x^3 - 2 * t * x^2 + t^3 = 0 ↔ 
  x = t ∨ x = t * (1 + Real.sqrt 5) / 2 ∨ x = t * (1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_solve_cubic_equation_l1748_174876


namespace NUMINAMATH_GPT_speed_of_stream_l1748_174871

theorem speed_of_stream
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time_eq_upstream_time : downstream_distance / (boat_speed + v) = upstream_distance / (boat_speed - v)) :
  v = 8 :=
by
  let v := 8
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1748_174871


namespace NUMINAMATH_GPT_function_properties_l1748_174898

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x > 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  f 1 = 3 →
  Set.range f = Set.Icc (-6) 6 :=
sorry

end NUMINAMATH_GPT_function_properties_l1748_174898


namespace NUMINAMATH_GPT_Jose_Raju_Work_Together_l1748_174867

-- Definitions for the conditions
def JoseWorkRate : ℚ := 1 / 10
def RajuWorkRate : ℚ := 1 / 40
def CombinedWorkRate : ℚ := JoseWorkRate + RajuWorkRate

-- Theorem statement
theorem Jose_Raju_Work_Together :
  1 / CombinedWorkRate = 8 := by
    sorry

end NUMINAMATH_GPT_Jose_Raju_Work_Together_l1748_174867


namespace NUMINAMATH_GPT_time_for_pipe_a_to_fill_l1748_174828

noncomputable def pipe_filling_time (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) : ℝ := 
  (1 / a_rate)

theorem time_for_pipe_a_to_fill (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) 
  (h1 : b_rate = 2 * a_rate) 
  (h2 : c_rate = 2 * b_rate) 
  (h3 : (a_rate + b_rate + c_rate) * fill_time_together = 1) : 
  pipe_filling_time a_rate b_rate c_rate fill_time_together = 42 :=
sorry

end NUMINAMATH_GPT_time_for_pipe_a_to_fill_l1748_174828


namespace NUMINAMATH_GPT_age_of_youngest_child_l1748_174890

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l1748_174890


namespace NUMINAMATH_GPT_simplify_expression_l1748_174800

open Real

theorem simplify_expression (α : ℝ) : 
  (cos (4 * α - π / 2) * sin (5 * π / 2 + 2 * α)) / ((1 + cos (2 * α)) * (1 + cos (4 * α))) = tan α :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1748_174800


namespace NUMINAMATH_GPT_largest_number_l1748_174885

theorem largest_number (a b c : ℕ) (h1 : c = a + 6) (h2 : b = (a + c) / 2) (h3 : a * b * c = 46332) : 
  c = 39 := 
sorry

end NUMINAMATH_GPT_largest_number_l1748_174885


namespace NUMINAMATH_GPT_plywood_long_side_length_l1748_174887

theorem plywood_long_side_length (L : ℕ) (h1 : 2 * (L + 5) = 22) : L = 6 :=
by
  sorry

end NUMINAMATH_GPT_plywood_long_side_length_l1748_174887


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l1748_174858

-- Define the sum of the first 15 terms of the arithmetic progression
theorem arithmetic_progression_sum (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 16) :
  (15 / 2) * (2 * a + 14 * d) = 120 := by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l1748_174858


namespace NUMINAMATH_GPT_width_of_river_l1748_174874

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end NUMINAMATH_GPT_width_of_river_l1748_174874


namespace NUMINAMATH_GPT_problem1_problem2_l1748_174845

def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1
def f (a b : ℝ) : ℝ := 3 * A a b + 6 * B a b

theorem problem1 (a b : ℝ) : f a b = 15 * a * b - 6 * a - 9 :=
by 
  sorry

theorem problem2 (b : ℝ) : (∀ a : ℝ, f a b = -9) → b = 2 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1748_174845


namespace NUMINAMATH_GPT_geometric_sequence_fraction_l1748_174814

noncomputable def a_n : ℕ → ℝ := sorry -- geometric sequence {a_n}
noncomputable def S : ℕ → ℝ := sorry   -- sequence sum S_n
def q : ℝ := sorry                     -- common ratio

theorem geometric_sequence_fraction (h_sequence: ∀ n, 2 * S (n - 1) = S n + S (n + 1))
  (h_q: ∀ n, a_n (n + 1) = q * a_n n)
  (h_q_neg2: q = -2) :
  (a_n 5 + a_n 7) / (a_n 3 + a_n 5) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_fraction_l1748_174814


namespace NUMINAMATH_GPT_radius_of_base_of_cone_l1748_174884

theorem radius_of_base_of_cone (S : ℝ) (hS : S = 9 * Real.pi)
  (H : ∃ (l r : ℝ), (Real.pi * l = 2 * Real.pi * r) ∧ S = Real.pi * r^2 + Real.pi * r * l) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_base_of_cone_l1748_174884


namespace NUMINAMATH_GPT_kho_kho_only_l1748_174888

theorem kho_kho_only (kabaddi_total : ℕ) (both_games : ℕ) (total_players : ℕ) (kabaddi_only : ℕ) (kho_kho_only : ℕ) 
  (h1 : kabaddi_total = 10)
  (h2 : both_games = 5)
  (h3 : total_players = 50)
  (h4 : kabaddi_only = 10 - both_games)
  (h5 : kabaddi_only + kho_kho_only + both_games = total_players) :
  kho_kho_only = 40 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_kho_kho_only_l1748_174888


namespace NUMINAMATH_GPT_number_of_gummies_l1748_174813

-- Define the necessary conditions
def lollipop_cost : ℝ := 1.5
def lollipop_count : ℕ := 4
def gummy_cost : ℝ := 2.0
def initial_money : ℝ := 15.0
def money_left : ℝ := 5.0

-- Total cost of lollipops and total amount spent on candies
noncomputable def total_lollipop_cost := lollipop_count * lollipop_cost
noncomputable def total_spent := initial_money - money_left
noncomputable def total_gummy_cost := total_spent - total_lollipop_cost
noncomputable def gummy_count := total_gummy_cost / gummy_cost

-- Main theorem statement
theorem number_of_gummies : gummy_count = 2 := 
by
  sorry -- Proof to be added

end NUMINAMATH_GPT_number_of_gummies_l1748_174813


namespace NUMINAMATH_GPT_angle_C_eq_pi_div_3_side_c_eq_7_l1748_174819

theorem angle_C_eq_pi_div_3 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
  C = Real.pi / 3 :=
sorry

theorem side_c_eq_7 
  (a b c : ℝ) 
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h1a : a = 5) 
  (h1b : b = 8) 
  (h2 : C = Real.pi / 3) :
  c = 7 :=
sorry

end NUMINAMATH_GPT_angle_C_eq_pi_div_3_side_c_eq_7_l1748_174819


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l1748_174894

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_standard_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (focus_distance_condition : ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * (c a b))
  (circle_intersects_asymptote : ∃ (x y : ℝ), (x, y) = (1, 2) ∧ y = (b/a) * x + 2): 
  (a = 1) ∧ (b = 2) → (x^2 - (y^2 / 4) = 1) := 
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l1748_174894


namespace NUMINAMATH_GPT_average_birds_monday_l1748_174891

variable (M : ℕ)

def avg_birds_monday (M : ℕ) : Prop :=
  let total_sites := 5 + 5 + 10
  let total_birds := 5 * M + 5 * 5 + 10 * 8
  (total_birds = total_sites * 7)

theorem average_birds_monday (M : ℕ) (h : avg_birds_monday M) : M = 7 := by
  sorry

end NUMINAMATH_GPT_average_birds_monday_l1748_174891


namespace NUMINAMATH_GPT_cost_of_carrots_and_cauliflower_l1748_174843

variable {p c f o : ℝ}

theorem cost_of_carrots_and_cauliflower
  (h1 : p + c + f + o = 30)
  (h2 : o = 3 * p)
  (h3 : f = p + c) : 
  c + f = 14 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_carrots_and_cauliflower_l1748_174843


namespace NUMINAMATH_GPT_two_digit_multiples_of_4_and_9_l1748_174812

theorem two_digit_multiples_of_4_and_9 :
  ∃ (count : ℕ), 
    (∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → (n % 4 = 0 ∧ n % 9 = 0) → (n = 36 ∨ n = 72)) ∧ count = 2 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_multiples_of_4_and_9_l1748_174812


namespace NUMINAMATH_GPT_total_pears_picked_l1748_174820

variables (jason_keith_mike_morning : ℕ)
variables (alicia_tina_nicola_afternoon : ℕ)
variables (days : ℕ)
variables (total_pears : ℕ)

def one_day_total (jason_keith_mike_morning alicia_tina_nicola_afternoon : ℕ) : ℕ :=
  jason_keith_mike_morning + alicia_tina_nicola_afternoon

theorem total_pears_picked (hjkm: jason_keith_mike_morning = 46 + 47 + 12)
                           (hatn: alicia_tina_nicola_afternoon = 28 + 33 + 52)
                           (hdays: days = 3)
                           (htotal: total_pears = 654):
  total_pears = (one_day_total  (46 + 47 + 12)  (28 + 33 + 52)) * 3 := 
sorry

end NUMINAMATH_GPT_total_pears_picked_l1748_174820


namespace NUMINAMATH_GPT_problem_statement_l1748_174847

theorem problem_statement
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
                 (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + 2*b3*x + c3)) :
  b1 * c1 + b2 * c2 + 2 * b3 * c3 = 0 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1748_174847


namespace NUMINAMATH_GPT_bus_problem_l1748_174852

theorem bus_problem : ∀ before_stop after_stop : ℕ, before_stop = 41 → after_stop = 18 → before_stop - after_stop = 23 :=
by
  intros before_stop after_stop h_before h_after
  sorry

end NUMINAMATH_GPT_bus_problem_l1748_174852


namespace NUMINAMATH_GPT_diane_bakes_gingerbreads_l1748_174865

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end NUMINAMATH_GPT_diane_bakes_gingerbreads_l1748_174865


namespace NUMINAMATH_GPT_find_x_for_g_l1748_174879

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5)/6))^(1/3)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -65 / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_g_l1748_174879


namespace NUMINAMATH_GPT_min_sum_a_b2_l1748_174897

theorem min_sum_a_b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) : a + b ≥ 2 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_a_b2_l1748_174897


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l1748_174856

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h₀ : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
  (h₁ : ∃ x y : ℝ, x = a 4 ∧ y = a 8 ∧ (x^2 - 4 * x - 1 = 0) ∧ (y^2 - 4 * y - 1 = 0) ∧ (x + y = 4)) :
  a 6 = 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l1748_174856


namespace NUMINAMATH_GPT_not_perfect_square_l1748_174822

open Nat

theorem not_perfect_square (m n : ℕ) : ¬∃ k : ℕ, k^2 = 1 + 3^m + 3^n :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1748_174822


namespace NUMINAMATH_GPT_exists_abcd_for_n_gt_one_l1748_174836

theorem exists_abcd_for_n_gt_one (n : Nat) (h : n > 1) :
  ∃ a b c d : Nat, a + b = 4 * n ∧ c + d = 4 * n ∧ a * b - c * d = 4 * n := 
by
  sorry

end NUMINAMATH_GPT_exists_abcd_for_n_gt_one_l1748_174836


namespace NUMINAMATH_GPT_common_chord_equation_l1748_174842

-- Definition of the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Definition of the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Proposition stating we need to prove the line equation
theorem common_chord_equation (x y : ℝ) : circle1 x y → circle2 x y → x - y = 0 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_equation_l1748_174842


namespace NUMINAMATH_GPT_anthony_pencils_total_l1748_174895

def pencils_initial : Nat := 9
def pencils_kathryn : Nat := 56
def pencils_greg : Nat := 84
def pencils_maria : Nat := 138

theorem anthony_pencils_total : 
  pencils_initial + pencils_kathryn + pencils_greg + pencils_maria = 287 := 
by
  sorry

end NUMINAMATH_GPT_anthony_pencils_total_l1748_174895


namespace NUMINAMATH_GPT_expression_numerator_l1748_174804

theorem expression_numerator (p q : ℕ) (E : ℕ) 
  (h1 : p * 5 = q * 4)
  (h2 : (18 / 7) + (E / (2 * q + p)) = 3) : E = 6 := 
by 
  sorry

end NUMINAMATH_GPT_expression_numerator_l1748_174804


namespace NUMINAMATH_GPT_inversely_proportional_l1748_174840

theorem inversely_proportional (x y : ℕ) (c : ℕ) 
  (h1 : x * y = c)
  (hx1 : x = 40) 
  (hy1 : y = 5) 
  (hy2 : y = 10) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_l1748_174840


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1748_174848

theorem value_of_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x / 3 = y^2) (h2 : x / 9 = 9 * y) : 
  x + y = 2214 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1748_174848


namespace NUMINAMATH_GPT_find_interest_rate_l1748_174853

noncomputable def annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem find_interest_rate :
  annual_interest_rate 5000 5100.50 4 0.5 0.04 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1748_174853


namespace NUMINAMATH_GPT_pairs_satisfy_condition_l1748_174850

theorem pairs_satisfy_condition (a b : ℝ) :
  (∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a_int b_int : ℤ, a = a_int ∧ b = b_int)) :=
by
  sorry

end NUMINAMATH_GPT_pairs_satisfy_condition_l1748_174850


namespace NUMINAMATH_GPT_compute_ratio_d_e_l1748_174882

open Polynomial

noncomputable def quartic_polynomial (a b c d e : ℚ) : Polynomial ℚ := 
  C a * X^4 + C b * X^3 + C c * X^2 + C d * X + C e

def roots_of_quartic (a b c d e: ℚ) : Prop :=
  (quartic_polynomial a b c d e).roots = {1, 2, 3, 5}

theorem compute_ratio_d_e (a b c d e : ℚ) 
    (h : roots_of_quartic a b c d e) :
    d / e = -61 / 30 :=
  sorry

end NUMINAMATH_GPT_compute_ratio_d_e_l1748_174882


namespace NUMINAMATH_GPT_max_term_in_sequence_l1748_174846

theorem max_term_in_sequence (a : ℕ → ℝ)
  (h : ∀ n, a n = (n+1) * (7/8)^n) :
  (∀ n, a n ≤ a 6 ∨ a n ≤ a 7) ∧ (a 6 = max (a 6) (a 7)) ∧ (a 7 = max (a 6) (a 7)) :=
sorry

end NUMINAMATH_GPT_max_term_in_sequence_l1748_174846


namespace NUMINAMATH_GPT_yard_length_l1748_174821

theorem yard_length :
  let num_trees := 11
  let distance_between_trees := 18
  (num_trees - 1) * distance_between_trees = 180 :=
by
  let num_trees := 11
  let distance_between_trees := 18
  sorry

end NUMINAMATH_GPT_yard_length_l1748_174821


namespace NUMINAMATH_GPT_analytical_expression_f_l1748_174841

def f : ℝ → ℝ := sorry

theorem analytical_expression_f :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  (∀ y : ℝ, f y = y^2 - 5*y + 7) :=
by
  sorry

end NUMINAMATH_GPT_analytical_expression_f_l1748_174841


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1748_174861

variable {a b : ℕ → ℕ}
variable (S T : ℕ → ℕ)

-- Conditions
def condition (n : ℕ) : Prop :=
  S n / T n = (2 * n + 1) / (3 * n + 2)

-- Conjecture to prove
theorem arithmetic_sequence_problem (h : ∀ n, condition S T n) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1748_174861


namespace NUMINAMATH_GPT_minimize_frac_inv_l1748_174834

theorem minimize_frac_inv (a b : ℕ) (h1: 4 * a + b = 30) (h2: a > 0) (h3: b > 0) :
  (a, b) = (5, 10) :=
sorry

end NUMINAMATH_GPT_minimize_frac_inv_l1748_174834


namespace NUMINAMATH_GPT_proof_problem_l1748_174868

theorem proof_problem (x y : ℚ) : 
  (x ^ 2 - 9 * y ^ 2 = 0) ∧ 
  (x + y = 1) ↔ 
  ((x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2)) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1748_174868


namespace NUMINAMATH_GPT_solve_inequality_l1748_174818

theorem solve_inequality :
  {x : ℝ | (x^2 - 9) / (x - 3) > 0} = { x : ℝ | (-3 < x ∧ x < 3) ∨ (x > 3)} :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l1748_174818


namespace NUMINAMATH_GPT_range_of_b_l1748_174844

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := x^2 + b * x + c

def A (b c : ℝ) := {x : ℝ | f x b c = 0}
def B (b c : ℝ) := {x : ℝ | f (f x b c) b c = 0}

theorem range_of_b (b c : ℝ) (h : ∃ x₀ : ℝ, x₀ ∈ B b c ∧ x₀ ∉ A b c) :
  b < 0 ∨ b ≥ 4 := 
sorry

end NUMINAMATH_GPT_range_of_b_l1748_174844


namespace NUMINAMATH_GPT_smallest_c_is_52_l1748_174883

def seq (n : ℕ) : ℤ := -103 + (n:ℤ) * 2

theorem smallest_c_is_52 :
  ∃ c : ℕ, 
  (∀ n : ℕ, n < c → (∀ m : ℕ, m < n → seq m < 0) ∧ seq n = 0) ∧
  seq c > 0 ∧
  c = 52 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_is_52_l1748_174883


namespace NUMINAMATH_GPT_alex_lost_fish_l1748_174875

theorem alex_lost_fish (jacob_initial : ℕ) (alex_catch_ratio : ℕ) (jacob_additional : ℕ) (alex_initial : ℕ) (alex_final : ℕ) : 
  (jacob_initial = 8) → 
  (alex_catch_ratio = 7) → 
  (jacob_additional = 26) →
  (alex_initial = alex_catch_ratio * jacob_initial) →
  (alex_final = (jacob_initial + jacob_additional) - 1) → 
  alex_initial - alex_final = 23 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alex_lost_fish_l1748_174875


namespace NUMINAMATH_GPT_rectangle_square_division_l1748_174815

theorem rectangle_square_division (n : ℕ) 
  (a b c d : ℕ) 
  (h1 : a * b = n) 
  (h2 : c * d = n + 76)
  (h3 : ∃ u v : ℕ, gcd a c = u ∧ gcd b d = v ∧ u * v * a^2 = u * v * c^2 ∧ u * v * b^2 = u * v * d^2) : 
  n = 324 := sorry

end NUMINAMATH_GPT_rectangle_square_division_l1748_174815


namespace NUMINAMATH_GPT_sequence_solution_l1748_174860

-- Defining the sequence and the condition
def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 1

-- Defining the sequence formula we need to prove
def sequence_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1)

theorem sequence_solution (a S : ℕ → ℝ) (h : sequence_condition a S) :
  sequence_formula a :=
by 
  sorry

end NUMINAMATH_GPT_sequence_solution_l1748_174860


namespace NUMINAMATH_GPT_find_f_zero_l1748_174849

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem find_f_zero (a b : ℝ)
  (h1 : f 3 a b = 7)
  (h2 : f 5 a b = -1) : f 0 a b = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_f_zero_l1748_174849


namespace NUMINAMATH_GPT_first_runner_meets_conditions_l1748_174826

noncomputable def first_runner_time := 11

theorem first_runner_meets_conditions (T : ℕ) (second_runner_time third_runner_time : ℕ) (meet_time : ℕ)
  (h1 : second_runner_time = 4)
  (h2 : third_runner_time = 11 / 2)
  (h3 : meet_time = 44)
  (h4 : meet_time % T = 0)
  (h5 : meet_time % second_runner_time = 0)
  (h6 : meet_time % third_runner_time = 0) : 
  T = first_runner_time :=
by
  sorry

end NUMINAMATH_GPT_first_runner_meets_conditions_l1748_174826


namespace NUMINAMATH_GPT_max_value_of_expression_l1748_174851

variables (a x1 x2 : ℝ)

theorem max_value_of_expression :
  (x1 < 0) → (0 < x2) → (∀ x, x^2 - a * x + a - 2 > 0 ↔ (x < x1) ∨ (x > x2)) →
  (x1 * x2 = a - 2) → 
  x1 + x2 + 2 / x1 + 2 / x2 ≤ 0 :=
by
  intros h1 h2 h3 h4
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l1748_174851


namespace NUMINAMATH_GPT_find_p_l1748_174870

-- Definitions
variables {n : ℕ} {p : ℝ}
def X : Type := ℕ -- Assume X is ℕ-valued

-- Conditions
axiom binomial_expectation : n * p = 6
axiom binomial_variance : n * p * (1 - p) = 3

-- Question to prove
theorem find_p : p = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1748_174870


namespace NUMINAMATH_GPT_ray_climbing_stairs_l1748_174832

theorem ray_climbing_stairs (n : ℕ) (h1 : n % 4 = 3) (h2 : n % 5 = 2) (h3 : 10 < n) : n = 27 :=
sorry

end NUMINAMATH_GPT_ray_climbing_stairs_l1748_174832


namespace NUMINAMATH_GPT_students_arrangement_l1748_174806

def num_students := 5
def num_females := 2
def num_males := 3
def female_A_cannot_end := true
def only_two_males_next_to_each_other := true

theorem students_arrangement (h1: num_students = 5)
                             (h2: num_females = 2)
                             (h3: num_males = 3)
                             (h4: female_A_cannot_end = true)
                             (h5: only_two_males_next_to_each_other = true) :
    ∃ n, n = 48 :=
by
  sorry

end NUMINAMATH_GPT_students_arrangement_l1748_174806


namespace NUMINAMATH_GPT_solve_equation_l1748_174831

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1748_174831


namespace NUMINAMATH_GPT_percentage_of_copper_in_second_alloy_l1748_174859

theorem percentage_of_copper_in_second_alloy
  (w₁ w₂ w_total : ℝ)
  (p₁ p_total : ℝ)
  (h₁ : w₁ = 66)
  (h₂ : p₁ = 0.10)
  (h₃ : w_total = 121)
  (h₄ : p_total = 0.15) :
  (w_total - w₁) * 0.21 = w_total * p_total - w₁ * p₁ := 
  sorry

end NUMINAMATH_GPT_percentage_of_copper_in_second_alloy_l1748_174859


namespace NUMINAMATH_GPT_hall_mat_expenditure_l1748_174833

theorem hall_mat_expenditure
  (length width height cost_per_sq_meter : ℕ)
  (H_length : length = 20)
  (H_width : width = 15)
  (H_height : height = 5)
  (H_cost_per_sq_meter : cost_per_sq_meter = 50) :
  (2 * (length * width) + 2 * (length * height) + 2 * (width * height)) * cost_per_sq_meter = 47500 :=
by
  sorry

end NUMINAMATH_GPT_hall_mat_expenditure_l1748_174833


namespace NUMINAMATH_GPT_max_gold_coins_l1748_174801

variables (planks : ℕ)
          (windmill_planks windmill_gold : ℕ)
          (steamboat_planks steamboat_gold : ℕ)
          (airplane_planks airplane_gold : ℕ)

theorem max_gold_coins (h_planks: planks = 130)
                       (h_windmill: windmill_planks = 5 ∧ windmill_gold = 6)
                       (h_steamboat: steamboat_planks = 7 ∧ steamboat_gold = 8)
                       (h_airplane: airplane_planks = 14 ∧ airplane_gold = 19) :
  ∃ (gold : ℕ), gold = 172 :=
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l1748_174801


namespace NUMINAMATH_GPT_solved_work_problem_l1748_174824

noncomputable def work_problem : Prop :=
  ∃ (m w x : ℝ), 
  (3 * m + 8 * w = 6 * m + x * w) ∧ 
  (4 * m + 5 * w = 0.9285714285714286 * (3 * m + 8 * w)) ∧
  (x = 14)

theorem solved_work_problem : work_problem := sorry

end NUMINAMATH_GPT_solved_work_problem_l1748_174824


namespace NUMINAMATH_GPT_value_range_abs_function_l1748_174872

theorem value_range_abs_function : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 9 → 1 ≤ (abs (x - 3) + 1) ∧ (abs (x - 3) + 1) ≤ 7 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_value_range_abs_function_l1748_174872


namespace NUMINAMATH_GPT_shooting_competition_l1748_174873

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end NUMINAMATH_GPT_shooting_competition_l1748_174873
