import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_value_l1750_175000

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1750_175000


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l1750_175069

theorem fundraiser_group_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (initial_total * 3) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 3
  (initial_girls : ℚ) / initial_total = 3 / 10 →
  (final_girls : ℚ) / final_total = 1 / 4 →
  initial_girls = 18 :=
by
  sorry

#check fundraiser_group_composition

end NUMINAMATH_CALUDE_fundraiser_group_composition_l1750_175069


namespace NUMINAMATH_CALUDE_average_after_addition_l1750_175028

theorem average_after_addition (numbers : List ℝ) (target_avg : ℝ) : 
  numbers = [6, 16, 8, 12, 21] → target_avg = 17 →
  ∃ x : ℝ, (numbers.sum + x) / (numbers.length + 1 : ℝ) = target_avg ∧ x = 39 := by
sorry

end NUMINAMATH_CALUDE_average_after_addition_l1750_175028


namespace NUMINAMATH_CALUDE_equation_solution_l1750_175064

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1750_175064


namespace NUMINAMATH_CALUDE_stock_investment_net_increase_l1750_175083

theorem stock_investment_net_increase (x : ℝ) (x_pos : x > 0) :
  x * 1.5 * 0.7 = 1.05 * x := by
  sorry

end NUMINAMATH_CALUDE_stock_investment_net_increase_l1750_175083


namespace NUMINAMATH_CALUDE_quadratic_root_relationship_l1750_175086

/-- Given two quadratic equations and their relationship, prove the ratio of their coefficients -/
theorem quadratic_root_relationship (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) →
  n/p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relationship_l1750_175086


namespace NUMINAMATH_CALUDE_gray_sections_total_seeds_l1750_175092

theorem gray_sections_total_seeds (circle1_total : ℕ) (circle2_total : ℕ) (white_section : ℕ)
  (h1 : circle1_total = 87)
  (h2 : circle2_total = 110)
  (h3 : white_section = 68) :
  (circle1_total - white_section) + (circle2_total - white_section) = 61 := by
  sorry

end NUMINAMATH_CALUDE_gray_sections_total_seeds_l1750_175092


namespace NUMINAMATH_CALUDE_half_dollar_and_dollar_heads_probability_l1750_175002

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the result of flipping four coins -/
structure FourCoinFlip :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The set of all possible outcomes when flipping four coins -/
def allOutcomes : Finset FourCoinFlip := sorry

/-- The set of favorable outcomes (half-dollar and one-dollar are both heads) -/
def favorableOutcomes : Finset FourCoinFlip := sorry

/-- The probability of an event occurring -/
def probability (event : Finset FourCoinFlip) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem half_dollar_and_dollar_heads_probability :
  probability favorableOutcomes = 1/4 := by sorry

end NUMINAMATH_CALUDE_half_dollar_and_dollar_heads_probability_l1750_175002


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l1750_175024

def f (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -1) ∧ (∀ x : ℝ, f x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l1750_175024


namespace NUMINAMATH_CALUDE_expression_value_l1750_175098

theorem expression_value : 
  let c : ℚ := 1
  (1 + c + 1/1) * (1 + c + 1/2) * (1 + c + 1/3) * (1 + c + 1/4) * (1 + c + 1/5) = 133/20 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1750_175098


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1750_175097

/-- Given that the solution set of ax² + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1750_175097


namespace NUMINAMATH_CALUDE_y_minimizer_l1750_175066

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3*x + 5

/-- The theorem stating that (2a + 2b - 3) / 4 minimizes y -/
theorem y_minimizer (a b : ℝ) :
  ∃ (x_min : ℝ), x_min = (2*a + 2*b - 3) / 4 ∧
    ∀ (x : ℝ), y x a b ≥ y x_min a b :=
by
  sorry

end NUMINAMATH_CALUDE_y_minimizer_l1750_175066


namespace NUMINAMATH_CALUDE_expression_simplification_l1750_175043

theorem expression_simplification (a : ℝ) (h : a/2 - 2/a = 3) :
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1750_175043


namespace NUMINAMATH_CALUDE_seventh_root_of_137858491849_l1750_175030

theorem seventh_root_of_137858491849 : 
  (137858491849 : ℝ) ^ (1/7 : ℝ) = 11 := by sorry

end NUMINAMATH_CALUDE_seventh_root_of_137858491849_l1750_175030


namespace NUMINAMATH_CALUDE_two_cos_thirty_degrees_equals_sqrt_three_l1750_175029

theorem two_cos_thirty_degrees_equals_sqrt_three :
  2 * Real.cos (30 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_thirty_degrees_equals_sqrt_three_l1750_175029


namespace NUMINAMATH_CALUDE_fair_wall_painting_l1750_175010

theorem fair_wall_painting (people : ℕ) (rooms_type1 rooms_type2 : ℕ) 
  (walls_per_room_type1 walls_per_room_type2 : ℕ) :
  people = 5 →
  rooms_type1 = 5 →
  rooms_type2 = 4 →
  walls_per_room_type1 = 4 →
  walls_per_room_type2 = 5 →
  (rooms_type1 * walls_per_room_type1 + rooms_type2 * walls_per_room_type2) / people = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_wall_painting_l1750_175010


namespace NUMINAMATH_CALUDE_dm_length_l1750_175048

/-- A square with side length 3 and two points that divide it into three equal areas -/
structure EqualAreaSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The point M on side AD -/
  m : ℝ
  /-- The point N on side AB -/
  n : ℝ
  /-- The side length is 3 -/
  side_eq : side = 3
  /-- The point M is between 0 and the side length -/
  m_range : 0 ≤ m ∧ m ≤ side
  /-- The point N is between 0 and the side length -/
  n_range : 0 ≤ n ∧ n ≤ side
  /-- CM and CN divide the square into three equal areas -/
  equal_areas : (1/2 * m * side) = (1/2 * n * side) ∧ (1/2 * m * side) = (1/3 * side^2)

/-- The length of DM in an EqualAreaSquare is 2 -/
theorem dm_length (s : EqualAreaSquare) : s.m = 2 := by
  sorry

end NUMINAMATH_CALUDE_dm_length_l1750_175048


namespace NUMINAMATH_CALUDE_constant_value_proof_l1750_175025

/-- Given consecutive integers x, y, and z where x > y > z, z = 2, 
    and 2x + 3y + 3z = 5y + C, prove that C = 8 -/
theorem constant_value_proof (x y z : ℤ) (C : ℤ) 
    (h1 : x = z + 2)
    (h2 : y = z + 1)
    (h3 : x > y ∧ y > z)
    (h4 : z = 2)
    (h5 : 2*x + 3*y + 3*z = 5*y + C) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l1750_175025


namespace NUMINAMATH_CALUDE_projectile_max_height_l1750_175011

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 155

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∃ t, h t = max_height ∧ ∀ s, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1750_175011


namespace NUMINAMATH_CALUDE_systematic_sampling_l1750_175020

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : Nat) 
  (num_parts : Nat) 
  (first_part_end : Nat) 
  (first_drawn : Nat) 
  (nth_draw : Nat) :
  total_students = 1000 →
  num_parts = 50 →
  first_part_end = 20 →
  first_drawn = 15 →
  nth_draw = 40 →
  (nth_draw - 1) * (total_students / num_parts) + first_drawn = 795 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1750_175020


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l1750_175089

theorem equal_ratios_sum (M N : ℚ) : 
  (5 : ℚ) / 7 = M / 63 ∧ (5 : ℚ) / 7 = 70 / N → M + N = 143 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l1750_175089


namespace NUMINAMATH_CALUDE_baker_initial_cakes_l1750_175001

theorem baker_initial_cakes (total_cakes : ℕ) (extra_cakes : ℕ) (initial_cakes : ℕ) : 
  total_cakes = 87 → extra_cakes = 9 → initial_cakes = total_cakes - extra_cakes → initial_cakes = 78 := by
  sorry

end NUMINAMATH_CALUDE_baker_initial_cakes_l1750_175001


namespace NUMINAMATH_CALUDE_square_side_length_l1750_175072

/-- Given a square with diagonal length 4, prove that its side length is 2√2 -/
theorem square_side_length (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s^2 + s^2 = d^2 ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1750_175072


namespace NUMINAMATH_CALUDE_election_result_l1750_175081

/-- Represents an election with five candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (votes_D : ℕ)
  (votes_E : ℕ)

/-- Conditions for the election -/
def ElectionConditions (e : Election) : Prop :=
  e.votes_A = (30 * e.total_votes) / 100 ∧
  e.votes_B = (25 * e.total_votes) / 100 ∧
  e.votes_C = (20 * e.total_votes) / 100 ∧
  e.votes_D = (15 * e.total_votes) / 100 ∧
  e.votes_E = e.total_votes - (e.votes_A + e.votes_B + e.votes_C + e.votes_D) ∧
  e.votes_A = e.votes_B + 1200

theorem election_result (e : Election) (h : ElectionConditions e) :
  e.total_votes = 24000 ∧ e.votes_E = 2400 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l1750_175081


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1750_175074

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 20)
  (h2 : added_water = 2)
  (h3 : final_percentage = 20)
  : ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_water =
    final_percentage * (initial_volume + added_water) / 100 ∧
    initial_percentage = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l1750_175074


namespace NUMINAMATH_CALUDE_problem_statement_l1750_175042

/-- Given xw + yz = 8 and (2x + y)(2z + w) = 20, prove that xz + yw = 1 -/
theorem problem_statement (x y z w : ℝ) 
  (h1 : x * w + y * z = 8)
  (h2 : (2 * x + y) * (2 * z + w) = 20) :
  x * z + y * w = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1750_175042


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1750_175095

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.4 → p_black = 0.25 → p_red + p_black + p_white = 1 → p_white = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1750_175095


namespace NUMINAMATH_CALUDE_worker_travel_time_l1750_175090

/-- If a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, 
    then her usual time to reach the office is 60 minutes. -/
theorem worker_travel_time (S : ℝ) (T : ℝ) (h1 : S > 0) (h2 : T > 0) : 
  S * T = (5/6 * S) * (T + 12) → T = 60 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1750_175090


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1750_175076

-- Problem 1
theorem problem_one (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.cos α = -3 * a / (5 * (-a))) 
  (h3 : Real.sin α = 4 * a / (5 * (-a))) : 
  Real.sin α + 2 * Real.cos α = 2/5 := by sorry

-- Problem 2
theorem problem_two (β : ℝ) (h : Real.tan β = 2) : 
  Real.sin β ^ 2 + 2 * Real.sin β * Real.cos β = 8/5 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1750_175076


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_4032_l1750_175022

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ - 2016) ∧
  (∀ x : ℝ, x > 0 → f x > 2016)

/-- The theorem to be proved -/
theorem sum_of_max_min_is_4032 (f : ℝ → ℝ) (h : special_function f) :
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  let N := ⨅ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  M + N = 4032 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_4032_l1750_175022


namespace NUMINAMATH_CALUDE_road_repair_time_l1750_175051

/-- 
Theorem: Time to repair a road with two teams working simultaneously
Given:
- Team A can repair the entire road in 3 hours
- Team B can repair the entire road in 6 hours
- Both teams work simultaneously from opposite ends
Prove: The time taken to complete the repair is 2 hours
-/
theorem road_repair_time (team_a_time team_b_time : ℝ) 
  (ha : team_a_time = 3)
  (hb : team_b_time = 6) :
  (1 / team_a_time + 1 / team_b_time) * 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_time_l1750_175051


namespace NUMINAMATH_CALUDE_windows_preference_l1750_175016

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  no_pref = 90 →
  ∃ (windows_pref : ℕ),
    windows_pref = total - mac_pref - (mac_pref / 3) - no_pref ∧
    windows_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l1750_175016


namespace NUMINAMATH_CALUDE_max_value_constrained_sum_l1750_175008

theorem max_value_constrained_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → 2*x + y + 2*z ≤ 2*a + b + 2*c) →
  2*a + b + 2*c = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_sum_l1750_175008


namespace NUMINAMATH_CALUDE_series_sum_l1750_175060

theorem series_sum : ∑' n, (3 * n - 1 : ℝ) / 2^n = 5 := by sorry

end NUMINAMATH_CALUDE_series_sum_l1750_175060


namespace NUMINAMATH_CALUDE_quadratic_radical_always_nonnegative_l1750_175094

theorem quadratic_radical_always_nonnegative (x : ℝ) : x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_always_nonnegative_l1750_175094


namespace NUMINAMATH_CALUDE_min_of_four_expressions_bound_l1750_175044

theorem min_of_four_expressions_bound (r s u v : ℝ) :
  min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2))) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_of_four_expressions_bound_l1750_175044


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1750_175006

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1750_175006


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_twelve_l1750_175033

theorem logarithm_expression_equals_twelve :
  let x := (4 - (Real.log 4 / Real.log 36) - (Real.log 18 / Real.log 6)) / (Real.log 3 / Real.log 4)
  let y := (Real.log 27 / Real.log 8) + (Real.log 9 / Real.log 2)
  x * y = 12 := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_twelve_l1750_175033


namespace NUMINAMATH_CALUDE_wall_length_calculation_l1750_175047

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the wall's length is approximately 86 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side) * 2 = wall_width * (round ((mirror_side * mirror_side) * 2 / wall_width)) :=
by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l1750_175047


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1750_175040

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1750_175040


namespace NUMINAMATH_CALUDE_weighted_average_increase_matches_reduction_l1750_175034

/-- Represents the consumption proportion and price increases for a gas type -/
structure GasInfo where
  proportion : Real
  first_increase : Real
  second_increase : Real

/-- Calculates the final price multiplier for a gas type -/
def final_price_multiplier (gas : GasInfo) : Real :=
  (1 + gas.first_increase) * (1 + gas.second_increase)

/-- Calculates the weighted average price increase -/
def weighted_average_increase (gases : List GasInfo) : Real :=
  gases.foldl (fun acc gas => acc + gas.proportion * final_price_multiplier gas) 0

theorem weighted_average_increase_matches_reduction :
  let gases : List GasInfo := [
    { proportion := 0.40, first_increase := 0.30, second_increase := 0.15 },
    { proportion := 0.35, first_increase := 0.25, second_increase := 0.10 },
    { proportion := 0.25, first_increase := 0.20, second_increase := 0.05 }
  ]
  weighted_average_increase gases = 1.39425 := by sorry

end NUMINAMATH_CALUDE_weighted_average_increase_matches_reduction_l1750_175034


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l1750_175012

/-- The actual distance between two cities given their distance on a map and the map's scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The actual distance between Stockholm and Uppsala is 450 km. -/
theorem stockholm_uppsala_distance : 
  let map_distance : ℝ := 45
  let scale : ℝ := 10
  actual_distance map_distance scale = 450 :=
by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l1750_175012


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1750_175088

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1750_175088


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1750_175009

theorem system_solution_ratio (a b x y : ℝ) (h1 : b ≠ 0) (h2 : 4*x - y = a) (h3 : 5*y - 20*x = b) : a / b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1750_175009


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1750_175023

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the condition for OA and OB to be perpendicular
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem ellipse_intersection_theorem :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  C x₁ y₁ ∧ C x₂ y₂ ∧
  Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
  Perpendicular x₁ y₁ x₂ y₂ →
  (k = 1/2 ∨ k = -1/2) ∧
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (4*Real.sqrt 65/17)^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1750_175023


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1750_175070

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 5 ∧ x^2 + a*x - 2 > 0) → a > -23/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1750_175070


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1750_175093

def S : Set ℝ := {x | -2 < x ∧ x < 5}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem subset_implies_a_range (a : ℝ) (h : S ⊆ P a) : -5 ≤ a ∧ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1750_175093


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1750_175059

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1750_175059


namespace NUMINAMATH_CALUDE_proposition_implication_l1750_175077

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
sorry

end NUMINAMATH_CALUDE_proposition_implication_l1750_175077


namespace NUMINAMATH_CALUDE_complementary_and_supplementary_angles_l1750_175026

/-- Given an angle of 46 degrees, its complementary angle is 44 degrees and its supplementary angle is 134 degrees. -/
theorem complementary_and_supplementary_angles (angle : ℝ) : 
  angle = 46 → 
  (90 - angle = 44) ∧ (180 - angle = 134) := by
  sorry

end NUMINAMATH_CALUDE_complementary_and_supplementary_angles_l1750_175026


namespace NUMINAMATH_CALUDE_map_distance_calculation_l1750_175019

/-- Given a map scale and actual distances, calculate the map distance --/
theorem map_distance_calculation 
  (map_distance_mountains : ℝ) 
  (actual_distance_mountains : ℝ) 
  (actual_distance_ram : ℝ) :
  let scale := actual_distance_mountains / map_distance_mountains
  actual_distance_ram / scale = map_distance_mountains * (actual_distance_ram / actual_distance_mountains) :=
by sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l1750_175019


namespace NUMINAMATH_CALUDE_result_is_fifty_l1750_175046

-- Define the original number
def x : ℝ := 150

-- Define the percentage
def percentage : ℝ := 0.60

-- Define the subtracted value
def subtracted : ℝ := 40

-- Theorem to prove
theorem result_is_fifty : percentage * x - subtracted = 50 := by
  sorry

end NUMINAMATH_CALUDE_result_is_fifty_l1750_175046


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l1750_175005

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (z - Complex.I)).re = z.re / (2 - 2 * z.im) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l1750_175005


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l1750_175079

theorem sum_greater_than_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y > x + y) : x + y > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l1750_175079


namespace NUMINAMATH_CALUDE_unreachable_141_l1750_175014

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable (start : ℕ) : Set ℕ :=
  sorry

theorem unreachable_141 :
  141 ∉ reachable 141 \ {141} :=
sorry

end NUMINAMATH_CALUDE_unreachable_141_l1750_175014


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l1750_175075

def calculate_remaining_plants (initial_flowering : ℕ) (saturday_flowering : ℕ) (saturday_fruiting : ℕ) (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let saturday_total := initial_flowering + initial_fruiting + saturday_flowering + saturday_fruiting
  let sunday_total := saturday_total - sunday_flowering - sunday_fruiting
  sunday_total

theorem roxy_garden_plants : 
  calculate_remaining_plants 7 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l1750_175075


namespace NUMINAMATH_CALUDE_find_first_number_l1750_175080

theorem find_first_number (x : ℝ) : 
  let set1 := [10, 70, 19]
  let set2 := [x, 40, 60]
  (List.sum set2 / 3 : ℝ) = (List.sum set1 / 3 : ℝ) + 7 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_first_number_l1750_175080


namespace NUMINAMATH_CALUDE_bird_families_count_l1750_175087

/-- The number of bird families that flew to Africa -/
def families_to_africa : ℕ := 47

/-- The number of bird families that flew to Asia -/
def families_to_asia : ℕ := 94

/-- The difference between families that flew to Asia and Africa -/
def difference : ℕ := 47

/-- The total number of bird families before migration -/
def total_families : ℕ := families_to_africa + families_to_asia

theorem bird_families_count : 
  (families_to_asia = families_to_africa + difference) → 
  (total_families = 141) := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l1750_175087


namespace NUMINAMATH_CALUDE_shares_owned_shares_owned_example_l1750_175084

/-- Calculates the number of shares owned based on dividend payment and earnings -/
theorem shares_owned (expected_earnings dividend_ratio additional_dividend_rate actual_earnings total_dividend : ℚ) : ℚ :=
  let base_dividend := expected_earnings * dividend_ratio
  let additional_earnings := actual_earnings - expected_earnings
  let additional_dividend := (additional_earnings / 0.1) * additional_dividend_rate
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend / total_dividend_per_share

/-- Proves that the number of shares owned is 600 given the specific conditions -/
theorem shares_owned_example : shares_owned 0.8 0.5 0.04 1.1 312 = 600 := by
  sorry

end NUMINAMATH_CALUDE_shares_owned_shares_owned_example_l1750_175084


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1750_175068

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : FunctionType) 
  (h1 : ∀ a b : ℝ, f (a + b) + f (a - b) = 3 * f a + f b) 
  (h2 : f 1 = 1) : 
  ∀ x : ℝ, f x = if x = 1 then 1 else 0 := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l1750_175068


namespace NUMINAMATH_CALUDE_all_signs_used_l1750_175055

/-- Proves that all signs are used in the area code system --/
theorem all_signs_used (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) 
  (h1 : total_signs = 224)
  (h2 : used_signs = 222)
  (h3 : additional_codes = 888)
  (h4 : ∀ (sign : Nat), sign ≤ total_signs → (additional_codes / used_signs) * sign ≤ additional_codes) :
  total_signs - used_signs = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_signs_used_l1750_175055


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l1750_175027

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ initial_avg = 100 ∧ wrong_mark = 60 ∧ correct_avg = 95 →
  ∃ x : ℚ, n * initial_avg - wrong_mark + x = n * correct_avg ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l1750_175027


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l1750_175041

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The equation x² - x = 0 -/
def equation_D : QuadraticEquation where
  a := 1
  b := -1
  c := 0
  a_nonzero := by sorry

theorem equation_D_is_quadratic : equation_D.a ≠ 0 ∧ 
  equation_D.a * X^2 + equation_D.b * X + equation_D.c = X^2 - X := by sorry


end NUMINAMATH_CALUDE_equation_D_is_quadratic_l1750_175041


namespace NUMINAMATH_CALUDE_well_depth_is_784_l1750_175035

/-- The depth of the well in feet -/
def well_depth : ℝ := 784

/-- The total time in seconds for the stone to fall and the sound to return -/
def total_time : ℝ := 7.7

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1120

/-- The function describing the distance fallen by the stone after t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

/-- Theorem stating that the well depth is 784 feet given the conditions -/
theorem well_depth_is_784 :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧
    stone_fall t_fall = well_depth ∧
    t_fall + well_depth / sound_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_well_depth_is_784_l1750_175035


namespace NUMINAMATH_CALUDE_stones_for_hall_l1750_175003

/-- Calculates the number of stones required to pave a rectangular hall --/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 4,500 stones are required to pave the given hall --/
theorem stones_for_hall : stones_required 72 30 6 8 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l1750_175003


namespace NUMINAMATH_CALUDE_larry_cards_l1750_175078

theorem larry_cards (initial_cards final_cards taken_cards : ℕ) : 
  final_cards = initial_cards - taken_cards → 
  taken_cards = 9 → 
  final_cards = 58 → 
  initial_cards = 67 := by
sorry

end NUMINAMATH_CALUDE_larry_cards_l1750_175078


namespace NUMINAMATH_CALUDE_dance_class_theorem_l1750_175045

theorem dance_class_theorem (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 40 →
  Finset.card A = 18 →
  Finset.card B = 22 →
  Finset.card (A ∩ B) = 10 →
  Finset.card (U \ (A ∪ B)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_dance_class_theorem_l1750_175045


namespace NUMINAMATH_CALUDE_square_difference_l1750_175017

theorem square_difference (a b : ℝ) 
  (h1 : a^2 + a*b = 8) 
  (h2 : a*b + b^2 = 9) : 
  a^2 - b^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1750_175017


namespace NUMINAMATH_CALUDE_sanctuary_swamps_count_l1750_175056

/-- The number of different reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles living in the swamp areas -/
def total_reptiles : ℕ := 1424

/-- The number of swamps in the sanctuary -/
def number_of_swamps : ℕ := total_reptiles / reptiles_per_swamp

theorem sanctuary_swamps_count :
  number_of_swamps = 4 :=
by sorry

end NUMINAMATH_CALUDE_sanctuary_swamps_count_l1750_175056


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l1750_175052

theorem distance_between_complex_points : 
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l1750_175052


namespace NUMINAMATH_CALUDE_one_km_equals_500_chains_l1750_175013

-- Define the units
def kilometer : ℕ → ℕ := id
def hectometer : ℕ → ℕ := id
def chain : ℕ → ℕ := id

-- Define the conversion factors
axiom km_to_hm : ∀ x : ℕ, kilometer x = hectometer (10 * x)
axiom hm_to_chain : ∀ x : ℕ, hectometer x = chain (50 * x)

-- Theorem to prove
theorem one_km_equals_500_chains : kilometer 1 = chain 500 := by
  sorry

end NUMINAMATH_CALUDE_one_km_equals_500_chains_l1750_175013


namespace NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l1750_175049

open Real

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  area : ℝ  -- Area of the triangle

/-- The area of the circumcircle of a triangle --/
def circumcircle_area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of the circumcircle for the given triangle --/
theorem circumcircle_area_of_special_triangle :
  let t : Triangle := {
    A := π/4,  -- 45° in radians
    b := 2 * sqrt 2,
    area := 1
  }
  circumcircle_area t = 5*π/2 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l1750_175049


namespace NUMINAMATH_CALUDE_six_ronna_scientific_notation_l1750_175054

/-- Represents the number of zeros after a number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to its scientific notation -/
def ronna_to_scientific (n : ℕ) : ℝ := n * (10 : ℝ) ^ ronna_zeros

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 : ℝ) ^ 27 := by
  sorry

end NUMINAMATH_CALUDE_six_ronna_scientific_notation_l1750_175054


namespace NUMINAMATH_CALUDE_lizette_stamp_count_l1750_175065

/-- The number of stamps Minerva has -/
def minerva_stamps : ℕ := 688

/-- The number of additional stamps Lizette has compared to Minerva -/
def additional_stamps : ℕ := 125

/-- The total number of stamps Lizette has -/
def lizette_stamps : ℕ := minerva_stamps + additional_stamps

theorem lizette_stamp_count : lizette_stamps = 813 := by
  sorry

end NUMINAMATH_CALUDE_lizette_stamp_count_l1750_175065


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1750_175073

theorem fraction_to_decimal : (58 : ℚ) / 125 = 464 / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1750_175073


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l1750_175096

/-- Proves the number of girls in a class given a specific ratio and total number of students -/
theorem number_of_girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h_total : total = 260) (h_ratio : boy_ratio = 5 ∧ girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l1750_175096


namespace NUMINAMATH_CALUDE_function_increasing_intervals_l1750_175067

noncomputable def f (A ω φ x : ℝ) : ℝ := 2 * A * (Real.cos (ω * x + φ))^2 - A

theorem function_increasing_intervals
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_symmetry_axis : ∀ x, f A ω φ (π/3 - x) = f A ω φ (π/3 + x))
  (h_symmetry_center : ∀ x, f A ω φ (π/12 - x) = f A ω φ (π/12 + x))
  : ∀ k : ℤ, StrictMonoOn (f A ω φ) (Set.Icc (k * π - 2*π/3) (k * π - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_intervals_l1750_175067


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_diff_l1750_175004

theorem unique_prime_sum_and_diff : 
  ∃! p : ℕ, Prime p ∧ 
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ p = a + b) ∧ 
  (∃ c d : ℕ, Prime c ∧ Prime d ∧ p = c - d) ∧ 
  p = 5 := by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_diff_l1750_175004


namespace NUMINAMATH_CALUDE_breakfast_dessert_l1750_175021

-- Define the possible breakfast items
inductive BreakfastItem
  | Whiskey
  | Duck
  | Oranges
  | Pie
  | BelleHelenePear
  | StrawberrySherbet
  | Coffee

-- Define the structure of a journalist's statement
structure JournalistStatement where
  items : List BreakfastItem

-- Define the honesty levels of journalists
inductive JournalistHonesty
  | AlwaysTruthful
  | OneFalseStatement
  | AlwaysLies

-- Define the breakfast observation
structure BreakfastObservation where
  jules : JournalistStatement
  jacques : JournalistStatement
  jim : JournalistStatement
  julesHonesty : JournalistHonesty
  jacquesHonesty : JournalistHonesty
  jimHonesty : JournalistHonesty

def breakfast : BreakfastObservation := {
  jules := { items := [BreakfastItem.Whiskey, BreakfastItem.Duck, BreakfastItem.Oranges, BreakfastItem.Coffee] },
  jacques := { items := [BreakfastItem.Pie, BreakfastItem.BelleHelenePear] },
  jim := { items := [BreakfastItem.Whiskey, BreakfastItem.Pie, BreakfastItem.StrawberrySherbet, BreakfastItem.Coffee] },
  julesHonesty := JournalistHonesty.AlwaysTruthful,
  jacquesHonesty := JournalistHonesty.AlwaysLies,
  jimHonesty := JournalistHonesty.OneFalseStatement
}

theorem breakfast_dessert :
  ∃ (dessert : BreakfastItem), dessert = BreakfastItem.StrawberrySherbet :=
by sorry

end NUMINAMATH_CALUDE_breakfast_dessert_l1750_175021


namespace NUMINAMATH_CALUDE_distance_to_work_l1750_175036

/-- Proves that the distance from home to work is 10 km given the conditions --/
theorem distance_to_work (outbound_speed return_speed : ℝ) (distance : ℝ) : 
  return_speed = 2 * outbound_speed →
  distance / outbound_speed + distance / return_speed = 6 →
  return_speed = 5 →
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_to_work_l1750_175036


namespace NUMINAMATH_CALUDE_complex_simplification_l1750_175007

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) - 2*i * (3 - 4*i) = 20 - 20*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1750_175007


namespace NUMINAMATH_CALUDE_harolds_rent_l1750_175062

/-- Harold's monthly finances --/
def harolds_finances (rent : ℝ) : Prop :=
  let income : ℝ := 2500
  let car_payment : ℝ := 300
  let utilities : ℝ := car_payment / 2
  let groceries : ℝ := 50
  let remaining : ℝ := income - rent - car_payment - utilities - groceries
  let retirement_savings : ℝ := remaining / 2
  let final_balance : ℝ := remaining - retirement_savings
  final_balance = 650

/-- Theorem: Harold's rent is $700.00 --/
theorem harolds_rent : ∃ (rent : ℝ), harolds_finances rent ∧ rent = 700 := by
  sorry

end NUMINAMATH_CALUDE_harolds_rent_l1750_175062


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l1750_175099

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (left : ℕ) 
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332) :
  initial - left - lost = 563 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l1750_175099


namespace NUMINAMATH_CALUDE_election_results_l1750_175039

/-- Election results theorem -/
theorem election_results 
  (total_students : ℕ) 
  (voter_turnout : ℚ) 
  (vote_percent_A vote_percent_B vote_percent_C vote_percent_D vote_percent_E : ℚ) : 
  total_students = 5000 →
  voter_turnout = 3/5 →
  vote_percent_A = 2/5 →
  vote_percent_B = 1/4 →
  vote_percent_C = 1/5 →
  vote_percent_D = 1/10 →
  vote_percent_E = 1/20 →
  (↑total_students * voter_turnout * vote_percent_A - ↑total_students * voter_turnout * vote_percent_B : ℚ) = 450 ∧
  (↑total_students * voter_turnout * (vote_percent_C + vote_percent_D + vote_percent_E) : ℚ) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_election_results_l1750_175039


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1750_175082

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1750_175082


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1750_175071

theorem simplify_complex_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 2)))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1750_175071


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1750_175091

theorem triangle_abc_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  4 * Real.sin A * Real.sin B - 4 * (Real.cos ((A - B) / 2))^2 = Real.sqrt 2 - 2 ∧
  a * Real.sin B / Real.sin A = 4 ∧
  1/2 * a * b * Real.sin C = 8 →
  C = π/4 ∧ c = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1750_175091


namespace NUMINAMATH_CALUDE_area_of_rectangle_l1750_175061

/-- The area of rectangle ABCD given the described configuration of squares and triangle -/
theorem area_of_rectangle (
  shaded_square_area : ℝ) 
  (h1 : shaded_square_area = 4) 
  (h2 : ∃ (side : ℝ), side^2 = shaded_square_area) 
  (h3 : ∃ (triangle_height : ℝ), triangle_height = Real.sqrt shaded_square_area) : 
  shaded_square_area + shaded_square_area + (2 * Real.sqrt shaded_square_area * Real.sqrt shaded_square_area / 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l1750_175061


namespace NUMINAMATH_CALUDE_periodic_function_smallest_period_l1750_175032

/-- A function satisfying the given periodic property -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

/-- The smallest positive period of a function -/
def SmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriod f T ∧ T > 0 ∧ ∀ S : ℝ, IsPeriod f S ∧ S > 0 → T ≤ S

/-- The main theorem stating that functions satisfying the given condition have a smallest period of 24 -/
theorem periodic_function_smallest_period (f : ℝ → ℝ) (h : PeriodicFunction f) :
    SmallestPeriod f 24 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_smallest_period_l1750_175032


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1750_175063

theorem collinear_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt (1 + Real.sin (40 * π / 180))]
  let b : Fin 2 → ℝ := ![1 / Real.sin (65 * π / 180), x]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) → x = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1750_175063


namespace NUMINAMATH_CALUDE_solve_for_y_l1750_175050

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1750_175050


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l1750_175038

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  x = 2 ∧ y = 2 * Real.sqrt 3 ∧ z = 4 →
  ∃ (r θ : ℝ),
    r = 4 ∧
    θ = π / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = z :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l1750_175038


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1750_175018

/-- The equation has exactly three distinct roots if and only if a is in the set {-1.5, -0.75, 0, 1/4} -/
theorem three_distinct_roots (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ w : ℝ, (w^2 + (2*a - 1)*w - 4*a - 2) * (w^2 + w + a) = 0 ↔ w = x ∨ w = y ∨ w = z)) ↔
  a = -1.5 ∨ a = -0.75 ∨ a = 0 ∨ a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1750_175018


namespace NUMINAMATH_CALUDE_gcd_360_504_l1750_175031

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l1750_175031


namespace NUMINAMATH_CALUDE_original_number_l1750_175037

theorem original_number (x : ℝ) : x * 1.5 = 150 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1750_175037


namespace NUMINAMATH_CALUDE_painting_time_ratio_l1750_175015

-- Define the painting times for each person
def matt_time : ℝ := 12
def rachel_time : ℝ := 13

-- Define Patty's time in terms of a variable
def patty_time : ℝ → ℝ := λ p => p

-- Define Rachel's time in terms of Patty's time
def rachel_time_calc : ℝ → ℝ := λ p => 2 * p + 5

-- Theorem statement
theorem painting_time_ratio :
  ∃ p : ℝ, 
    rachel_time_calc p = rachel_time ∧ 
    (patty_time p) / matt_time = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_ratio_l1750_175015


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l1750_175053

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / ((10 ^ d.repeatLength - 1) : ℚ)

/-- The repeating decimal 0.3045045045... -/
def decimal : RepeatingDecimal :=
  { integerPart := 0
  , repeatingPart := 3045
  , repeatLength := 4 }

theorem decimal_equals_fraction : toRational decimal = 383 / 1110 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l1750_175053


namespace NUMINAMATH_CALUDE_dianas_age_dianas_age_is_eight_l1750_175085

/-- Diana's age today, given that she is twice as old as Grace and Grace turned 3 a year ago -/
theorem dianas_age : ℕ :=
  let graces_age_last_year : ℕ := 3
  let graces_age_today : ℕ := graces_age_last_year + 1
  let dianas_age : ℕ := 2 * graces_age_today
  8

/-- Proof that Diana's age is 8 years old today -/
theorem dianas_age_is_eight : dianas_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_dianas_age_dianas_age_is_eight_l1750_175085


namespace NUMINAMATH_CALUDE_systematic_sampling_50_5_l1750_175057

/-- Represents a list of product numbers selected using systematic sampling. -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that systematic sampling of 5 products from 50 products
    results in the selection of products numbered 10, 20, 30, 40, and 50. -/
theorem systematic_sampling_50_5 :
  systematicSample 50 5 = [10, 20, 30, 40, 50] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_50_5_l1750_175057


namespace NUMINAMATH_CALUDE_bus_speed_with_stops_l1750_175058

/-- The speed of a bus including stoppages, given its speed excluding stoppages and stop time -/
theorem bus_speed_with_stops (speed_without_stops : ℝ) (stop_time : ℝ) :
  speed_without_stops = 54 →
  stop_time = 20 →
  let speed_with_stops := speed_without_stops * (60 - stop_time) / 60
  speed_with_stops = 36 := by
  sorry

#check bus_speed_with_stops

end NUMINAMATH_CALUDE_bus_speed_with_stops_l1750_175058
