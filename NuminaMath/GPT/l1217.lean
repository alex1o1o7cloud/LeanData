import Mathlib

namespace NUMINAMATH_GPT_factorize_expression_l1217_121703

theorem factorize_expression (a x y : ℝ) : 2 * x * (a - 2) - y * (2 - a) = (a - 2) * (2 * x + y) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1217_121703


namespace NUMINAMATH_GPT_equivalent_annual_rate_l1217_121716

theorem equivalent_annual_rate :
  ∀ (annual_rate compounding_periods: ℝ), annual_rate = 0.08 → compounding_periods = 4 → 
  ((1 + (annual_rate / compounding_periods)) ^ compounding_periods - 1) * 100 = 8.24 :=
by
  intros annual_rate compounding_periods h_rate h_periods
  sorry

end NUMINAMATH_GPT_equivalent_annual_rate_l1217_121716


namespace NUMINAMATH_GPT_ellipse_and_triangle_properties_l1217_121705

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  1/2 * a * b

theorem ellipse_and_triangle_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y ↔ (x, y) = (1, 3/2) ∨ (x, y) = (1, -3/2)) ∧
  area_triangle 2 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_and_triangle_properties_l1217_121705


namespace NUMINAMATH_GPT_initial_markers_count_l1217_121767

   -- Let x be the initial number of markers Megan had.
   variable (x : ℕ)

   -- Conditions:
   def robert_gave_109_markers : Prop := true
   def total_markers_after_adding : ℕ := 326
   def markers_added_by_robert : ℕ := 109

   -- The total number of markers Megan has now is 326.
   def total_markers_eq (x : ℕ) : Prop := x + markers_added_by_robert = total_markers_after_adding

   -- Prove that initially Megan had 217 markers.
   theorem initial_markers_count : total_markers_eq 217 := by
     sorry
   
end NUMINAMATH_GPT_initial_markers_count_l1217_121767


namespace NUMINAMATH_GPT_function_minimum_value_no_maximum_l1217_121795

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.sin x + a) / Real.sin x

theorem function_minimum_value_no_maximum (a : ℝ) (h_a : 0 < a) : 
  ∃ x_min, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≥ x_min ∧ 
           (∀ x ∈ Set.Ioo 0 Real.pi, f a x ≠ x_min) ∧ 
           ¬ (∃ x_max, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≤ x_max) :=
by
  let t := Real.sin
  have h : ∀ x ∈ Set.Ioo 0 Real.pi, t x ∈ Set.Ioo 0 1 := sorry -- Simple property of sine function in (0, π)
  -- Exact details skipped to align with the conditions from the problem, leveraging the property
  sorry -- Full proof not required as per instructions

end NUMINAMATH_GPT_function_minimum_value_no_maximum_l1217_121795


namespace NUMINAMATH_GPT_maciek_total_cost_l1217_121722

-- Define the cost of pretzels without discount
def pretzel_price : ℝ := 4.0

-- Define the discounted price of pretzels when buying 3 or more packs
def pretzel_discount_price : ℝ := 3.5

-- Define the cost of chips without discount
def chips_price : ℝ := 7.0

-- Define the discounted price of chips when buying 2 or more packs
def chips_discount_price : ℝ := 6.0

-- Define the number of pretzels Maciek buys
def pretzels_bought : ℕ := 3

-- Define the number of chips Maciek buys
def chips_bought : ℕ := 4

-- Calculate the total cost of pretzels
def pretzel_cost : ℝ :=
  if pretzels_bought >= 3 then pretzels_bought * pretzel_discount_price else pretzels_bought * pretzel_price

-- Calculate the total cost of chips
def chips_cost : ℝ :=
  if chips_bought >= 2 then chips_bought * chips_discount_price else chips_bought * chips_price

-- Calculate the total amount Maciek needs to pay
def total_cost : ℝ :=
  pretzel_cost + chips_cost

theorem maciek_total_cost :
  total_cost = 34.5 :=
by 
  sorry

end NUMINAMATH_GPT_maciek_total_cost_l1217_121722


namespace NUMINAMATH_GPT_expression_value_l1217_121710

theorem expression_value : 
  ∀ (x y z: ℤ), x = 2 ∧ y = -3 ∧ z = 1 → x^2 + y^2 - z^2 - 2*x*y = 24 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1217_121710


namespace NUMINAMATH_GPT_max_airlines_l1217_121732

-- Definitions for the conditions
-- There are 200 cities
def num_cities : ℕ := 200

-- Calculate the total number of city pairs
def num_city_pairs (n : ℕ) : ℕ := (n * (n - 1)) / 2

def total_city_pairs : ℕ := num_city_pairs num_cities

-- Minimum spanning tree concept
def min_flights_per_airline (n : ℕ) : ℕ := n - 1

def total_flights_required : ℕ := num_cities * min_flights_per_airline num_cities

-- Claim: Maximum number of airlines
theorem max_airlines (n : ℕ) (h : n = 200) : ∃ m : ℕ, m = (total_city_pairs / (min_flights_per_airline n)) ∧ m = 100 :=
by sorry

end NUMINAMATH_GPT_max_airlines_l1217_121732


namespace NUMINAMATH_GPT_smallest_pencils_l1217_121755

theorem smallest_pencils (P : ℕ) :
  (P > 2) ∧
  (P % 5 = 2) ∧
  (P % 9 = 2) ∧
  (P % 11 = 2) →
  P = 497 := by
  sorry

end NUMINAMATH_GPT_smallest_pencils_l1217_121755


namespace NUMINAMATH_GPT_remainder_of_9_pow_1995_mod_7_l1217_121739

theorem remainder_of_9_pow_1995_mod_7 : (9^1995) % 7 = 1 := 
by 
sorry

end NUMINAMATH_GPT_remainder_of_9_pow_1995_mod_7_l1217_121739


namespace NUMINAMATH_GPT_functional_equation_solution_l1217_121702

noncomputable def f : ℚ → ℚ := sorry

theorem functional_equation_solution :
  (∀ x y : ℚ, f (f x + x * f y) = x + f x * y) →
  (∀ x : ℚ, f x = x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1217_121702


namespace NUMINAMATH_GPT_line_equation_through_P_and_equidistant_from_A_B_l1217_121718

theorem line_equation_through_P_and_equidistant_from_A_B (P A B : ℝ × ℝ) (hP : P = (1, 2)) (hA : A = (2, 3)) (hB : B = (4, -5)) :
  (∃ l : ℝ × ℝ → Prop, ∀ x y, l (x, y) ↔ 4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_through_P_and_equidistant_from_A_B_l1217_121718


namespace NUMINAMATH_GPT_lassis_from_mangoes_l1217_121713

def ratio (lassis mangoes : ℕ) : Prop := lassis = 11 * mangoes / 2

theorem lassis_from_mangoes (mangoes : ℕ) (h : mangoes = 10) : ratio 55 mangoes :=
by
  rw [h]
  unfold ratio
  sorry

end NUMINAMATH_GPT_lassis_from_mangoes_l1217_121713


namespace NUMINAMATH_GPT_number_of_numbers_in_last_group_l1217_121770

theorem number_of_numbers_in_last_group :
  ∃ n : ℕ, (60 * 13) = (57 * 6) + 50 + (61 * n) ∧ n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_numbers_in_last_group_l1217_121770


namespace NUMINAMATH_GPT_log_product_eq_3_div_4_l1217_121747

theorem log_product_eq_3_div_4 : (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 9) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_log_product_eq_3_div_4_l1217_121747


namespace NUMINAMATH_GPT_number_is_square_plus_opposite_l1217_121723

theorem number_is_square_plus_opposite (x : ℝ) (hx : x = x^2 + -x) : x = 0 ∨ x = 2 :=
by sorry

end NUMINAMATH_GPT_number_is_square_plus_opposite_l1217_121723


namespace NUMINAMATH_GPT_exists_n_good_not_n_add_1_good_l1217_121754

-- Define the sum of digits function S
def S (k : ℕ) : ℕ := (k.digits 10).sum

-- Define what it means for a number to be n-good
def n_good (a n : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), (a_seq 0 = a) ∧ (∀ i : Fin n, a_seq i.succ = a_seq i - S (a_seq i))

-- Define the main theorem
theorem exists_n_good_not_n_add_1_good : ∀ n : ℕ, ∃ a : ℕ, n_good a n ∧ ¬n_good a (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_good_not_n_add_1_good_l1217_121754


namespace NUMINAMATH_GPT_find_inverse_mod_36_l1217_121709

-- Given condition
def inverse_mod_17 := (17 * 23) % 53 = 1

-- Definition for the problem statement
def inverse_mod_36 : Prop := (36 * 30) % 53 = 1

theorem find_inverse_mod_36 (h : inverse_mod_17) : inverse_mod_36 :=
sorry

end NUMINAMATH_GPT_find_inverse_mod_36_l1217_121709


namespace NUMINAMATH_GPT_fraction_proof_l1217_121779

theorem fraction_proof (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) : 
  (x + y) / (y + z) = 26 / 53 := 
by
  sorry

end NUMINAMATH_GPT_fraction_proof_l1217_121779


namespace NUMINAMATH_GPT_max_sum_of_factors_l1217_121781

theorem max_sum_of_factors (heartsuit spadesuit : ℕ) (h : heartsuit * spadesuit = 24) :
  heartsuit + spadesuit ≤ 25 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l1217_121781


namespace NUMINAMATH_GPT_percentage_decrease_10_l1217_121756

def stocks_decrease (F J M : ℝ) (X : ℝ) : Prop :=
  J = F * (1 - X / 100) ∧
  J = M * 1.20 ∧
  M = F * 0.7500000000000007

theorem percentage_decrease_10 {F J M X : ℝ} (h : stocks_decrease F J M X) :
  X = 9.99999999999992 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_10_l1217_121756


namespace NUMINAMATH_GPT_radius_of_circle_l1217_121762

/-- Given the equation of a circle x^2 + y^2 - 8 = 2x + 4y,
    we need to prove that the radius of the circle is sqrt 13. -/
theorem radius_of_circle : 
    ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8 = 2*x + 4*y → r = Real.sqrt 13) :=
by
    sorry

end NUMINAMATH_GPT_radius_of_circle_l1217_121762


namespace NUMINAMATH_GPT_part_one_part_two_l1217_121757

theorem part_one (g : ℝ → ℝ) (h : ∀ x, g x = |x - 1| + 2) : {x : ℝ | |g x| < 5} = {x : ℝ | -2 < x ∧ x < 4} :=
sorry

theorem part_two (f g : ℝ → ℝ) (h1 : ∀ x, f x = |2 * x - a| + |2 * x + 3|) (h2 : ∀ x, g x = |x - 1| + 2) 
(h3 : ∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g x2) : {a : ℝ | a ≥ -1 ∨ a ≤ -5} :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1217_121757


namespace NUMINAMATH_GPT_woman_born_1892_l1217_121792

theorem woman_born_1892 (y : ℕ) (hy : 1850 ≤ y^2 - y ∧ y^2 - y < 1900) : y = 44 :=
by
  sorry

end NUMINAMATH_GPT_woman_born_1892_l1217_121792


namespace NUMINAMATH_GPT_constant_is_5_variables_are_n_and_S_l1217_121730

-- Define the conditions
def cost_per_box : ℕ := 5
def total_cost (n : ℕ) : ℕ := n * cost_per_box

-- Define the statement to be proved
-- constant is 5
theorem constant_is_5 : cost_per_box = 5 := 
by sorry

-- variables are n and S, where S is total_cost n
theorem variables_are_n_and_S (n : ℕ) : 
    ∃ S : ℕ, S = total_cost n :=
by sorry

end NUMINAMATH_GPT_constant_is_5_variables_are_n_and_S_l1217_121730


namespace NUMINAMATH_GPT_car_speed_proof_l1217_121724

noncomputable def car_speed_second_hour 
  (speed_first_hour: ℕ) (average_speed: ℕ) (total_time: ℕ) 
  (speed_second_hour: ℕ) : Prop :=
  (speed_first_hour = 80) ∧ (average_speed = 70) ∧ (total_time = 2) → speed_second_hour = 60

theorem car_speed_proof : 
  car_speed_second_hour 80 70 2 60 := by
  sorry

end NUMINAMATH_GPT_car_speed_proof_l1217_121724


namespace NUMINAMATH_GPT_bobbo_minimum_speed_increase_l1217_121771

theorem bobbo_minimum_speed_increase
  (initial_speed: ℝ)
  (river_width : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (midpoint_distance : ℝ)
  (required_increase: ℝ) :
  initial_speed = 2 ∧ river_width = 100 ∧ current_speed = 5 ∧ waterfall_distance = 175 ∧ midpoint_distance = 50 ∧ required_increase = 3 → 
  (required_increase = (50 / (50 / current_speed)) - initial_speed) := 
by
  sorry

end NUMINAMATH_GPT_bobbo_minimum_speed_increase_l1217_121771


namespace NUMINAMATH_GPT_cot_30_plus_cot_75_eq_2_l1217_121706

noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cot_30_plus_cot_75_eq_2 : cot 30 + cot 75 = 2 := by sorry

end NUMINAMATH_GPT_cot_30_plus_cot_75_eq_2_l1217_121706


namespace NUMINAMATH_GPT_base_conversion_equivalence_l1217_121733

theorem base_conversion_equivalence :
  ∃ (n : ℕ), (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 9 * C + B) ∧
             (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 6 * B + C) ∧
             n = 0 := 
by 
  sorry

end NUMINAMATH_GPT_base_conversion_equivalence_l1217_121733


namespace NUMINAMATH_GPT_distance_from_dormitory_to_city_l1217_121784

theorem distance_from_dormitory_to_city (D : ℝ) :
  (1 / 4) * D + (1 / 2) * D + 10 = D → D = 40 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_from_dormitory_to_city_l1217_121784


namespace NUMINAMATH_GPT_river_flow_rate_l1217_121700

-- Define the conditions
def depth : ℝ := 8
def width : ℝ := 25
def volume_per_min : ℝ := 26666.666666666668

-- The main theorem proving the rate at which the river is flowing
theorem river_flow_rate : (volume_per_min / (depth * width)) = 133.33333333333334 := by
  -- Express the area of the river's cross-section
  let area := depth * width
  -- Define the velocity based on the given volume and calculated area
  let velocity := volume_per_min / area
  -- Simplify and derive the result
  show velocity = 133.33333333333334
  sorry

end NUMINAMATH_GPT_river_flow_rate_l1217_121700


namespace NUMINAMATH_GPT_negate_even_condition_l1217_121728

theorem negate_even_condition (a b c : ℤ) :
  (¬(∀ a b c : ℤ, ∃ x : ℚ, a * x^2 + b * x + c = 0 → Even a ∧ Even b ∧ Even c)) →
  (¬Even a ∨ ¬Even b ∨ ¬Even c) :=
by
  sorry

end NUMINAMATH_GPT_negate_even_condition_l1217_121728


namespace NUMINAMATH_GPT_total_children_estimate_l1217_121720

theorem total_children_estimate (k m n : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) 
(h4 : n ≤ m) (h5 : n ≤ k) (h6 : m ≤ k) :
  (∃ (total : ℕ), total = k * m / n) :=
sorry

end NUMINAMATH_GPT_total_children_estimate_l1217_121720


namespace NUMINAMATH_GPT_line_parallel_eq_l1217_121763

theorem line_parallel_eq (x y : ℝ) (h1 : 3 * x - y = 6) (h2 : x = -2 ∧ y = 3) :
  ∃ m b, m = 3 ∧ b = 9 ∧ y = m * x + b :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_eq_l1217_121763


namespace NUMINAMATH_GPT_intersection_points_l1217_121751

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem intersection_points :
  ∃ p q : ℝ × ℝ, 
    (p = (0, c) ∨ p = (-1, a - b + c)) ∧ 
    (q = (0, c) ∨ q = (-1, a - b + c)) ∧
    p ≠ q ∧
    (∃ x : ℝ, (x, ax^2 + bx + c) = p) ∧
    (∃ x : ℝ, (x, -ax^3 + bx + c) = q) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1217_121751


namespace NUMINAMATH_GPT_students_play_both_l1217_121711

def students_total : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def neither_players : ℕ := 4

theorem students_play_both : 
  (students_total - neither_players) + (hockey_players + basketball_players - students_total + neither_players - students_total) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_students_play_both_l1217_121711


namespace NUMINAMATH_GPT_range_of_m_solve_inequality_l1217_121799

open Real Set

noncomputable def f (x: ℝ) := -abs (x - 2)
noncomputable def g (x: ℝ) (m: ℝ) := -abs (x - 3) + m

-- Problem 1: Prove the range of m given the condition
theorem range_of_m (h : ∀ x : ℝ, f x > g x m) : m < 1 :=
  sorry

-- Problem 2: Prove the set of solutions for f(x) + a - 1 > 0
theorem solve_inequality (a : ℝ) :
  (if a = 1 then {x : ℝ | x ≠ 2}
   else if a > 1 then univ
   else {x : ℝ | x < 1 + a} ∪ {x : ℝ | x > 3 - a}) = {x : ℝ | f x + a - 1 > 0} :=
  sorry

end NUMINAMATH_GPT_range_of_m_solve_inequality_l1217_121799


namespace NUMINAMATH_GPT_max_M_range_a_l1217_121750

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem max_M (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) : 
  4 ≤ g x1 - g x2 :=
sorry

theorem range_a (a : ℝ) (s t : ℝ) (h1 : 1 / 2 ≤ s) (h2 : s ≤ 2) (h3 : 1 / 2 ≤ t) (h4 : t ≤ 2) : 
  1 ≤ a ∧ f s a ≥ g t :=
sorry

end NUMINAMATH_GPT_max_M_range_a_l1217_121750


namespace NUMINAMATH_GPT_prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l1217_121735

section card_draws

/-- A card draw experiment with 10 cards: 5 red, 3 white, 2 blue. --/
inductive CardColor
| red
| white
| blue

def bag : List CardColor := List.replicate 5 CardColor.red ++ List.replicate 3 CardColor.white ++ List.replicate 2 CardColor.blue

/-- Probability of drawing exactly 2 red cards in up to 3 draws with the given conditions. --/
def prob_two_reds : ℚ :=
  (5 / 10) * (5 / 10) + 
  (5 / 10) * (2 / 10) * (5 / 10) + 
  (2 / 10) * (5 / 10) * (5 / 10)

theorem prob_of_two_reds_is_7_over_20 : prob_two_reds = 7 / 20 :=
  sorry

/-- Probability distribution of the number of draws necessary. --/
def prob_ξ_1 : ℚ := 3 / 10
def prob_ξ_2 : ℚ := 21 / 100
def prob_ξ_3 : ℚ := 49 / 100
def expected_value_ξ : ℚ :=
  1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3

theorem expected_value_is_2_19 : expected_value_ξ = 219 / 100 :=
  sorry

end card_draws

end NUMINAMATH_GPT_prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l1217_121735


namespace NUMINAMATH_GPT_p_of_neg3_equals_14_l1217_121794

-- Functions definitions
def u (x : ℝ) : ℝ := 4 * x + 5
def p (y : ℝ) : ℝ := y^2 - 2 * y + 6

-- Theorem statement
theorem p_of_neg3_equals_14 : p (-3) = 14 := by
  sorry

end NUMINAMATH_GPT_p_of_neg3_equals_14_l1217_121794


namespace NUMINAMATH_GPT_largest_sum_is_1173_l1217_121785

def largest_sum_of_two_3digit_numbers : Prop :=
  ∃ a b c d e f : ℕ, 
  (a = 6 ∧ b = 5 ∧ c = 4 ∧ d = 3 ∧ e = 2 ∧ f = 1) ∧
  100 * (a + b) + 10 * (c + d) + (e + f) = 1173

theorem largest_sum_is_1173 : largest_sum_of_two_3digit_numbers :=
  by
  sorry

end NUMINAMATH_GPT_largest_sum_is_1173_l1217_121785


namespace NUMINAMATH_GPT_average_eq_one_half_l1217_121753

variable (w x y : ℝ)

-- Conditions
variables (h1 : 2 / w + 2 / x = 2 / y)
variables (h2 : w * x = y)

theorem average_eq_one_half : (w + x) / 2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_average_eq_one_half_l1217_121753


namespace NUMINAMATH_GPT_option_B_correct_l1217_121715

theorem option_B_correct (x y : ℝ) : 
  x * y^2 - y^2 * x = 0 :=
by sorry

end NUMINAMATH_GPT_option_B_correct_l1217_121715


namespace NUMINAMATH_GPT_complex_fraction_l1217_121721

open Complex

theorem complex_fraction
  (a b : ℂ)
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 := 
by
  sorry

end NUMINAMATH_GPT_complex_fraction_l1217_121721


namespace NUMINAMATH_GPT_bacon_needed_l1217_121712

def eggs_per_plate : ℕ := 2
def bacon_per_plate : ℕ := 2 * eggs_per_plate
def customers : ℕ := 14
def bacon_total (eggs_per_plate bacon_per_plate customers : ℕ) : ℕ := customers * bacon_per_plate

theorem bacon_needed : bacon_total eggs_per_plate bacon_per_plate customers = 56 :=
by
  sorry

end NUMINAMATH_GPT_bacon_needed_l1217_121712


namespace NUMINAMATH_GPT_total_savings_correct_l1217_121789

-- Define the savings of Sam, Victory and Alex according to the given conditions
def sam_savings : ℕ := 1200
def victory_savings : ℕ := sam_savings - 200
def alex_savings : ℕ := 2 * victory_savings

-- Define the total savings
def total_savings : ℕ := sam_savings + victory_savings + alex_savings

-- The theorem to prove the total savings
theorem total_savings_correct : total_savings = 4200 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_correct_l1217_121789


namespace NUMINAMATH_GPT_travel_time_l1217_121786

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end NUMINAMATH_GPT_travel_time_l1217_121786


namespace NUMINAMATH_GPT_tiles_difference_ninth_eighth_rectangle_l1217_121772

theorem tiles_difference_ninth_eighth_rectangle : 
  let width (n : Nat) := 2 * n
  let height (n : Nat) := n
  let tiles (n : Nat) := width n * height n
  tiles 9 - tiles 8 = 34 :=
by
  intro width height tiles
  sorry

end NUMINAMATH_GPT_tiles_difference_ninth_eighth_rectangle_l1217_121772


namespace NUMINAMATH_GPT_intersection_PQ_l1217_121748

def setP  := {x : ℝ | x * (x - 1) ≥ 0}
def setQ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_PQ : {x : ℝ | x > 1} = {z : ℝ | z ∈ setP ∧ z ∈ setQ} :=
by
  sorry

end NUMINAMATH_GPT_intersection_PQ_l1217_121748


namespace NUMINAMATH_GPT_minimum_value_l1217_121760

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 1) + 9 / y = 1) : 4 * x + y ≥ 21 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1217_121760


namespace NUMINAMATH_GPT_find_D_l1217_121717

-- We define the points E and F
def E : ℝ × ℝ := (-3, -2)
def F : ℝ × ℝ := (5, 10)

-- Definition of point D with the given conditions
def D : ℝ × ℝ := (3, 7)

-- We state the main theorem to prove that D is such that ED = 2 * DF given E and F
theorem find_D (D : ℝ × ℝ) (ED_DF_relation : dist E D = 2 * dist D F) : D = (3, 7) :=
sorry

end NUMINAMATH_GPT_find_D_l1217_121717


namespace NUMINAMATH_GPT_greatest_integer_l1217_121777

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end NUMINAMATH_GPT_greatest_integer_l1217_121777


namespace NUMINAMATH_GPT_handshakes_at_meetup_l1217_121729

theorem handshakes_at_meetup :
  let gremlins := 25
  let imps := 20
  let sprites := 10
  ∃ (total_handshakes : ℕ), total_handshakes = 1095 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_at_meetup_l1217_121729


namespace NUMINAMATH_GPT_part1_solution_part2_solution_part3_solution_l1217_121798

-- Define the basic conditions
variables (x y m : ℕ)

-- Part 1: Number of pieces of each type purchased (Proof for 10 pieces of A, 20 pieces of B)
theorem part1_solution (h1 : x + y = 30) (h2 : 28 * x + 22 * y = 720) :
  (x = 10) ∧ (y = 20) :=
sorry

-- Part 2: Maximize sales profit for the second purchase
theorem part2_solution (h1 : 28 * m + 22 * (80 - m) ≤ 2000) :
  m = 40 ∧ (max_profit = 1040) :=
sorry

-- Variables for Part 3
variables (a : ℕ)
-- Profit equation for type B apples with adjusted selling price
theorem part3_solution (h : (4 + 2 * a) * (34 - a - 22) = 90) :
  (a = 7) ∧ (selling_price = 27) :=
sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_part3_solution_l1217_121798


namespace NUMINAMATH_GPT_pairs_of_natural_numbers_l1217_121737

theorem pairs_of_natural_numbers (a b : ℕ) (h₁ : b ∣ a + 1) (h₂ : a ∣ b + 1) :
    (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_pairs_of_natural_numbers_l1217_121737


namespace NUMINAMATH_GPT_gideon_fraction_of_marbles_l1217_121759

variable (f : ℝ)

theorem gideon_fraction_of_marbles (marbles : ℝ) (age_now : ℝ) (age_future : ℝ) (remaining_marbles : ℝ) (future_age_with_remaining_marbles : Bool)
  (h1 : marbles = 100)
  (h2 : age_now = 45)
  (h3 : age_future = age_now + 5)
  (h4 : remaining_marbles = 2 * (1 - f) * marbles)
  (h5 : remaining_marbles = age_future)
  (h6 : future_age_with_remaining_marbles = (age_future = 50)) :
  f = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_gideon_fraction_of_marbles_l1217_121759


namespace NUMINAMATH_GPT_find_m_l1217_121773

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end NUMINAMATH_GPT_find_m_l1217_121773


namespace NUMINAMATH_GPT_hired_year_l1217_121714

theorem hired_year (A W : ℕ) (Y : ℕ) (retire_year : ℕ) 
    (hA : A = 30) 
    (h_rule : A + W = 70) 
    (h_retire : retire_year = 2006) 
    (h_employment : retire_year - Y = W) 
    : Y = 1966 := 
by 
  -- proofs are skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_hired_year_l1217_121714


namespace NUMINAMATH_GPT_list_price_proof_l1217_121734

theorem list_price_proof (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_list_price_proof_l1217_121734


namespace NUMINAMATH_GPT_teagan_total_cost_l1217_121744

theorem teagan_total_cost :
  let reduction_percentage := 20
  let original_price_shirt := 60
  let original_price_jacket := 90
  let reduced_price_shirt := original_price_shirt * (100 - reduction_percentage) / 100
  let reduced_price_jacket := original_price_jacket * (100 - reduction_percentage) / 100
  let cost_5_shirts := 5 * reduced_price_shirt
  let cost_10_jackets := 10 * reduced_price_jacket
  let total_cost := cost_5_shirts + cost_10_jackets
  total_cost = 960 := by
  sorry

end NUMINAMATH_GPT_teagan_total_cost_l1217_121744


namespace NUMINAMATH_GPT_max_value_l1217_121701

theorem max_value (y : ℝ) (h : y ≠ 0) : 
  ∃ M, M = 1 / 25 ∧ 
       ∀ y ≠ 0,  ∀ value, value = y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) 
       → value ≤ M :=
sorry

end NUMINAMATH_GPT_max_value_l1217_121701


namespace NUMINAMATH_GPT_min_jellybeans_l1217_121746

theorem min_jellybeans (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 15) : n = 151 :=
by { sorry }

end NUMINAMATH_GPT_min_jellybeans_l1217_121746


namespace NUMINAMATH_GPT_evaluate_expression_l1217_121783

-- Define the integers a and b
def a := 2019
def b := 2020

-- The main theorem stating the equivalence
theorem evaluate_expression :
  (a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3 + 6) / (a * b) = 5 / (a * b) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1217_121783


namespace NUMINAMATH_GPT_Peggy_dolls_l1217_121749

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end NUMINAMATH_GPT_Peggy_dolls_l1217_121749


namespace NUMINAMATH_GPT_alex_pen_difference_l1217_121797

theorem alex_pen_difference 
  (alex_initial_pens : Nat) 
  (doubling_rate : Nat) 
  (weeks : Nat) 
  (jane_pens_month : Nat) :
  alex_initial_pens = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  jane_pens_month = 16 →
  (alex_initial_pens * doubling_rate ^ weeks) - jane_pens_month = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end NUMINAMATH_GPT_alex_pen_difference_l1217_121797


namespace NUMINAMATH_GPT_Mike_can_play_300_minutes_l1217_121790

-- Define the weekly earnings, spending, and costs as conditions
def weekly_earnings : ℕ := 100
def half_spent_at_arcade : ℕ := weekly_earnings / 2
def food_cost : ℕ := 10
def token_cost_per_hour : ℕ := 8
def hour_in_minutes : ℕ := 60

-- Define the remaining money after buying food
def money_for_tokens : ℕ := half_spent_at_arcade - food_cost

-- Define the hours he can play
def hours_playable : ℕ := money_for_tokens / token_cost_per_hour

-- Define the total minutes he can play
def total_minutes_playable : ℕ := hours_playable * hour_in_minutes

-- Prove that with his expenditure, Mike can play for 300 minutes
theorem Mike_can_play_300_minutes : total_minutes_playable = 300 := 
by
  sorry -- Proof will be filled here

end NUMINAMATH_GPT_Mike_can_play_300_minutes_l1217_121790


namespace NUMINAMATH_GPT_problem_to_prove_l1217_121752

theorem problem_to_prove
  (α : ℝ)
  (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = -7 / 9 :=
by
  sorry -- proof required

end NUMINAMATH_GPT_problem_to_prove_l1217_121752


namespace NUMINAMATH_GPT_chinese_mathematical_system_l1217_121742

noncomputable def problem_statement : Prop :=
  ∃ (x : ℕ) (y : ℕ),
    7 * x + 7 = y ∧ 
    9 * (x - 1) = y

theorem chinese_mathematical_system :
  problem_statement := by
  sorry

end NUMINAMATH_GPT_chinese_mathematical_system_l1217_121742


namespace NUMINAMATH_GPT_simplify_expression_l1217_121726

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 2) (h₂ : a ≠ -2) : 
  (2 * a / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2)) :=
by
  -- proof to be added
  sorry

end NUMINAMATH_GPT_simplify_expression_l1217_121726


namespace NUMINAMATH_GPT_possible_values_a1_l1217_121704

theorem possible_values_a1 (m : ℕ) (h_m_pos : 0 < m)
    (a : ℕ → ℕ) (h_seq : ∀ n, a n.succ = if a n < 2^m then a n ^ 2 + 2^m else a n / 2)
    (h1 : ∀ n, a n > 0) :
    (∀ n, ∃ k : ℕ, a n = 2^k) ↔ (m = 2 ∧ ∃ ℓ : ℕ, a 0 = 2 ^ ℓ ∧ 0 < ℓ) :=
by sorry

end NUMINAMATH_GPT_possible_values_a1_l1217_121704


namespace NUMINAMATH_GPT_worst_player_is_niece_l1217_121764

structure Player where
  name : String
  sex : String
  generation : Nat

def grandmother := Player.mk "Grandmother" "Female" 1
def niece := Player.mk "Niece" "Female" 2
def grandson := Player.mk "Grandson" "Male" 3
def son_in_law := Player.mk "Son-in-law" "Male" 2

def worst_player : Player := niece
def best_player : Player := grandmother

-- Conditions
def cousin_check : worst_player ≠ best_player ∧
                   worst_player.generation ≠ best_player.generation ∧ 
                   worst_player.sex ≠ best_player.sex := 
  by sorry

-- Prove that the worst player is the niece
theorem worst_player_is_niece : worst_player = niece :=
  by sorry

end NUMINAMATH_GPT_worst_player_is_niece_l1217_121764


namespace NUMINAMATH_GPT_polynomial_sum_evaluation_l1217_121738

noncomputable def q1 : Polynomial ℤ := Polynomial.X^3
noncomputable def q2 : Polynomial ℤ := Polynomial.X^2 + Polynomial.X + 1
noncomputable def q3 : Polynomial ℤ := Polynomial.X - 1
noncomputable def q4 : Polynomial ℤ := Polynomial.X^2 + 1

theorem polynomial_sum_evaluation :
  q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 52 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_evaluation_l1217_121738


namespace NUMINAMATH_GPT_minimum_value_2x_plus_y_l1217_121731

theorem minimum_value_2x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y + 6 = x * y) : 
  2 * x + y ≥ 12 := 
sorry

end NUMINAMATH_GPT_minimum_value_2x_plus_y_l1217_121731


namespace NUMINAMATH_GPT_anthony_initial_pencils_l1217_121778

def initial_pencils (given_pencils : ℝ) (remaining_pencils : ℝ) : ℝ :=
  given_pencils + remaining_pencils

theorem anthony_initial_pencils :
  initial_pencils 9.0 47.0 = 56.0 :=
by
  sorry

end NUMINAMATH_GPT_anthony_initial_pencils_l1217_121778


namespace NUMINAMATH_GPT_largest_last_digit_in_string_l1217_121793

theorem largest_last_digit_in_string :
  ∃ (s : Nat → Fin 10), 
    (s 0 = 1) ∧ 
    (∀ k, k < 99 → (∃ m, (s k * 10 + s (k + 1)) = 17 * m ∨ (s k * 10 + s (k + 1)) = 23 * m)) ∧
    (∃ l, l < 10 ∧ (s 99 = l)) ∧
    (forall last, (last < 10 ∧ (s 99 = last))) ∧
    (∀ m n, s 99 = m → s 99 = n → m ≤ n → n = 9) :=
sorry

end NUMINAMATH_GPT_largest_last_digit_in_string_l1217_121793


namespace NUMINAMATH_GPT_problem_a1_value_l1217_121743

theorem problem_a1_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (h : ∀ x : ℝ, x^10 = a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7 + a₈ * (x - 1)^8 + a₉ * (x - 1)^9 + a₁₀ * (x - 1)^10) :
  a₁ = 10 :=
sorry

end NUMINAMATH_GPT_problem_a1_value_l1217_121743


namespace NUMINAMATH_GPT_marbles_leftover_l1217_121725

theorem marbles_leftover (g j : ℕ) (hg : g % 8 = 5) (hj : j % 8 = 6) :
  ((g + 5 + j) % 8) = 0 :=
by
  sorry

end NUMINAMATH_GPT_marbles_leftover_l1217_121725


namespace NUMINAMATH_GPT_find_particular_number_l1217_121775

theorem find_particular_number (A B : ℤ) (x : ℤ) (hA : A = 14) (hB : B = 24)
  (h : (((A + x) * A - B) / B = 13)) : x = 10 :=
by {
  -- You can add an appropriate lemma or proof here if necessary
  sorry
}

end NUMINAMATH_GPT_find_particular_number_l1217_121775


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1217_121741

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 7636) (h3 : v3 = 11628) :
  ((v3: ℝ) / (v1 + v2 + v3)) * 100 = 57 := by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1217_121741


namespace NUMINAMATH_GPT_circle_area_l1217_121766

theorem circle_area (r : ℝ) (h1 : 5 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_circle_area_l1217_121766


namespace NUMINAMATH_GPT_gold_bars_distribution_l1217_121736

theorem gold_bars_distribution 
  (initial_gold : ℕ) 
  (lost_gold : ℕ) 
  (num_friends : ℕ) 
  (remaining_gold : ℕ)
  (each_friend_gets : ℕ) :
  initial_gold = 100 →
  lost_gold = 20 →
  num_friends = 4 →
  remaining_gold = initial_gold - lost_gold →
  each_friend_gets = remaining_gold / num_friends →
  each_friend_gets = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gold_bars_distribution_l1217_121736


namespace NUMINAMATH_GPT_pyramid_edge_length_correct_l1217_121791

-- Definitions for the conditions
def total_length (sum_of_edges : ℝ) := sum_of_edges = 14.8
def edges_count (num_of_edges : ℕ) := num_of_edges = 8

-- Definition for the question and corresponding answer to prove
def length_of_one_edge (sum_of_edges : ℝ) (num_of_edges : ℕ) (one_edge_length : ℝ) :=
  sum_of_edges / num_of_edges = one_edge_length

-- The statement that needs to be proven
theorem pyramid_edge_length_correct : total_length 14.8 → edges_count 8 → length_of_one_edge 14.8 8 1.85 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_pyramid_edge_length_correct_l1217_121791


namespace NUMINAMATH_GPT_determine_a_range_l1217_121769

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem determine_a_range (e : ℝ) (he : e = Real.exp 1) :
  ∃ a_range : Set ℝ, a_range = Set.Icc 1 (e + 1 / e) :=
by 
  sorry

end NUMINAMATH_GPT_determine_a_range_l1217_121769


namespace NUMINAMATH_GPT_both_solve_prob_l1217_121708

variable (a b : ℝ) -- Define a and b as real numbers

-- Define the conditions
def not_solve_prob_A := (0 ≤ a) ∧ (a ≤ 1)
def not_solve_prob_B := (0 ≤ b) ∧ (b ≤ 1)
def independent := true -- independence is implicit by the question

-- Define the statement of the proof
theorem both_solve_prob (h1 : not_solve_prob_A a) (h2 : not_solve_prob_B b) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by sorry

end NUMINAMATH_GPT_both_solve_prob_l1217_121708


namespace NUMINAMATH_GPT_proof_problem_l1217_121765

-- Condition for the first part: a quadratic inequality having a solution set
def quadratic_inequality (a : ℝ) :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * x^2 - 3 * x + 2 ≤ 0

-- Condition for the second part: the solution set of a rational inequality
def rational_inequality_solution (a : ℝ) (b : ℝ) :=
  ∀ x : ℝ, (x + 3) / (a * x - b) > 0 ↔ (x < -3 ∨ x > 2)

theorem proof_problem {a : ℝ} {b : ℝ} :
  (quadratic_inequality a → a = 1 ∧ b = 2) ∧ 
  (rational_inequality_solution 1 2) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1217_121765


namespace NUMINAMATH_GPT_two_digit_divisors_1995_l1217_121707

theorem two_digit_divisors_1995 :
  (∃ (n : Finset ℕ), (∀ x ∈ n, 10 ≤ x ∧ x < 100 ∧ 1995 % x = 0) ∧ n.card = 6 ∧ ∃ y ∈ n, y = 95) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_divisors_1995_l1217_121707


namespace NUMINAMATH_GPT_area_of_bounded_region_l1217_121776

theorem area_of_bounded_region (x y : ℝ) (h : y^2 + 2 * x * y + 50 * abs x = 500) : 
  ∃ A, A = 1250 :=
sorry

end NUMINAMATH_GPT_area_of_bounded_region_l1217_121776


namespace NUMINAMATH_GPT_cube_positive_integers_solution_l1217_121740

theorem cube_positive_integers_solution (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∃ k : ℕ, 2^(Nat.factorial a) + 2^(Nat.factorial b) + 2^(Nat.factorial c) = k^3) ↔ 
    ( (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
      (a = 1 ∧ b = 2 ∧ c = 1) ∨ 
      (a = 2 ∧ b = 1 ∧ c = 1) ) :=
by
  sorry

end NUMINAMATH_GPT_cube_positive_integers_solution_l1217_121740


namespace NUMINAMATH_GPT_inequality_holds_l1217_121780

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
axiom symmetric_property : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom increasing_property : ∀ x y : ℝ, (1 ≤ x) → (x ≤ y) → f x ≤ f y

-- The statement of the theorem
theorem inequality_holds (m : ℝ) (h : m < 1 / 2) : f (1 - m) < f m :=
by sorry

end NUMINAMATH_GPT_inequality_holds_l1217_121780


namespace NUMINAMATH_GPT_draw_9_cards_ensure_even_product_l1217_121768

theorem draw_9_cards_ensure_even_product :
  ∀ (cards : Finset ℕ), (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 16) →
  (cards.card = 9) →
  (∃ (subset : Finset ℕ), subset ⊆ cards ∧ ∃ k ∈ subset, k % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_draw_9_cards_ensure_even_product_l1217_121768


namespace NUMINAMATH_GPT_not_a_solution_set4_l1217_121782

def set1 : ℝ × ℝ := (1, 2)
def set2 : ℝ × ℝ := (2, 0)
def set3 : ℝ × ℝ := (0.5, 3)
def set4 : ℝ × ℝ := (-2, 4)

noncomputable def is_solution (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 = 4

theorem not_a_solution_set4 : ¬ is_solution set4 := 
by 
  sorry

end NUMINAMATH_GPT_not_a_solution_set4_l1217_121782


namespace NUMINAMATH_GPT_men_in_first_group_l1217_121719

theorem men_in_first_group (x : ℕ) :
  (20 * 48 = x * 80) → x = 12 :=
by
  intro h_eq
  have : x = (20 * 48) / 80 := sorry
  exact this

end NUMINAMATH_GPT_men_in_first_group_l1217_121719


namespace NUMINAMATH_GPT_path_count_l1217_121758

theorem path_count (f : ℕ → (ℤ × ℤ)) :
  (∀ n, (f (n + 1)).1 = (f n).1 + 1 ∨ (f (n + 1)).2 = (f n).2 + 1) ∧
  f 0 = (-6, -6) ∧ f 24 = (6, 6) ∧
  (∀ n, ¬(-3 ≤ (f n).1 ∧ (f n).1 ≤ 3 ∧ -3 ≤ (f n).2 ∧ (f n).2 ≤ 3)) →
  ∃ N, N = 2243554 :=
by {
  sorry
}

end NUMINAMATH_GPT_path_count_l1217_121758


namespace NUMINAMATH_GPT_initial_population_l1217_121796

theorem initial_population (P : ℝ) : 
  (P * 1.2 * 0.8 = 9600) → P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l1217_121796


namespace NUMINAMATH_GPT_range_of_m_l1217_121761

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x^4) / Real.log 3
noncomputable def g (x : ℝ) : ℝ := x * f x

theorem range_of_m (m : ℝ) : g (1 - m) < g (2 * m) → m > 1 / 3 :=
  by
  sorry

end NUMINAMATH_GPT_range_of_m_l1217_121761


namespace NUMINAMATH_GPT_proof_problem_l1217_121727

noncomputable def f (x : ℝ) := 3 * Real.sin x + 2 * Real.cos x + 1

theorem proof_problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  (b * Real.cos c / a) = -1 :=
sorry

end NUMINAMATH_GPT_proof_problem_l1217_121727


namespace NUMINAMATH_GPT_find_complementary_angle_l1217_121788

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end NUMINAMATH_GPT_find_complementary_angle_l1217_121788


namespace NUMINAMATH_GPT_inequality_solution_min_value_of_a2_b2_c2_min_achieved_l1217_121787

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

theorem inequality_solution :
  ∀ x : ℝ, (f x ≥ 3) ↔ (x ≤ -1 ∨ x ≥ 1) :=
by sorry

theorem min_value_of_a2_b2_c2 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  a^2 + b^2 + c^2 ≥ 3/7 :=
by sorry

theorem min_achieved (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  (2*a = b) ∧ (b = c/2) ∧ (a^2 + b^2 + c^2 = 3/7) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_min_value_of_a2_b2_c2_min_achieved_l1217_121787


namespace NUMINAMATH_GPT_find_k_l1217_121745

open Classical

theorem find_k 
    (z x y k : ℝ) 
    (k_pos_int : k > 0 ∧ ∃ n : ℕ, k = n)
    (prop1 : z - y = k * x)
    (prop2 : x - z = k * y)
    (cond : z = (5 / 3) * (x - y)) :
    k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1217_121745


namespace NUMINAMATH_GPT_sufficient_balance_after_29_months_l1217_121774

noncomputable def accumulated_sum (S0 : ℕ) (D : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  S0 * (1 + r)^n + D * ((1 + r)^n - 1) / r

theorem sufficient_balance_after_29_months :
  let S0 := 300000
  let D := 15000
  let r := (1 / 100 : ℚ) -- interest rate of 1%
  accumulated_sum S0 D r 29 ≥ 900000 :=
by
  sorry -- The proof will be elaborated later

end NUMINAMATH_GPT_sufficient_balance_after_29_months_l1217_121774
