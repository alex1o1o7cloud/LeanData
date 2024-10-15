import Mathlib

namespace NUMINAMATH_GPT_miles_left_to_reach_E_l1978_197852

-- Given conditions as definitions
def total_journey : ℕ := 2500
def miles_driven : ℕ := 642
def miles_B_to_C : ℕ := 400
def miles_C_to_D : ℕ := 550
def detour_D_to_E : ℕ := 200

-- Proof statement
theorem miles_left_to_reach_E : 
  (miles_B_to_C + miles_C_to_D + detour_D_to_E) = 1150 :=
by
  sorry

end NUMINAMATH_GPT_miles_left_to_reach_E_l1978_197852


namespace NUMINAMATH_GPT_vanessas_mother_picked_14_carrots_l1978_197881

-- Define the problem parameters
variable (V : Nat := 17)  -- Vanessa picked 17 carrots
variable (G : Nat := 24)  -- Total good carrots
variable (B : Nat := 7)   -- Total bad carrots

-- Define the proof goal: Vanessa's mother picked 14 carrots
theorem vanessas_mother_picked_14_carrots : (G + B) - V = 14 := by
  sorry

end NUMINAMATH_GPT_vanessas_mother_picked_14_carrots_l1978_197881


namespace NUMINAMATH_GPT_cookies_last_days_l1978_197840

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end NUMINAMATH_GPT_cookies_last_days_l1978_197840


namespace NUMINAMATH_GPT_sequence_comparison_l1978_197872

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define geometric sequence
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ (∀ n, b (n + 1) = b n * q) ∧ (∀ i, i ≥ 1 → b i > 0)

-- Main theorem to prove
theorem sequence_comparison {a b : ℕ → ℝ} (q : ℝ) (h_a_arith : arithmetic_sequence a) 
  (h_b_geom : geometric_sequence b q) (h_eq_1 : a 1 = b 1) (h_eq_11 : a 11 = b 11) :
  a 6 > b 6 :=
sorry

end NUMINAMATH_GPT_sequence_comparison_l1978_197872


namespace NUMINAMATH_GPT_mixed_oil_rate_l1978_197823

/-- Given quantities and prices of three types of oils, any combination
that satisfies the volume and price conditions will achieve a final mixture rate of Rs. 65 per litre. -/
theorem mixed_oil_rate (x y z : ℝ) : 
  12.5 * 55 + 7.75 * 70 + 3.25 * 82 = 1496.5 ∧ 12.5 + 7.75 + 3.25 = 23.5 →
  x + y + z = 23.5 ∧ 55 * x + 70 * y + 82 * z = 65 * 23.5 →
  true :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_l1978_197823


namespace NUMINAMATH_GPT_slope_of_line_between_intersections_of_circles_l1978_197805

theorem slope_of_line_between_intersections_of_circles :
  ∀ C D : ℝ × ℝ, 
    -- Conditions: equations of the circles
    (C.1^2 + C.2^2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ (C.1^2 + C.2^2 - 8 * C.1 - 2 * C.2 + 10 = 0) →
    (D.1^2 + D.2^2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ (D.1^2 + D.2^2 - 8 * D.1 - 2 * D.2 + 10 = 0) →
    -- Question: slope of line CD
    ((C.2 - D.2) / (C.1 - D.1) = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_between_intersections_of_circles_l1978_197805


namespace NUMINAMATH_GPT_angle_733_in_first_quadrant_l1978_197882

def in_first_quadrant (θ : ℝ) : Prop := 
  0 < θ ∧ θ < 90

theorem angle_733_in_first_quadrant :
  in_first_quadrant (733 % 360 : ℝ) :=
sorry

end NUMINAMATH_GPT_angle_733_in_first_quadrant_l1978_197882


namespace NUMINAMATH_GPT_angle_B_sum_a_c_l1978_197870

theorem angle_B (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B) :
  B = π / 3 :=
  sorry

theorem sum_a_c (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B)
  (hB : B = π / 3) :
  a + c = Real.sqrt 15 :=
  sorry

end NUMINAMATH_GPT_angle_B_sum_a_c_l1978_197870


namespace NUMINAMATH_GPT_probability_top_card_special_l1978_197858

-- Definition of the problem conditions
def deck_size : ℕ := 52
def special_card_count : ℕ := 16

-- The statement we need to prove
theorem probability_top_card_special : 
  (special_card_count : ℚ) / deck_size = 4 / 13 := 
  by sorry

end NUMINAMATH_GPT_probability_top_card_special_l1978_197858


namespace NUMINAMATH_GPT_abs_AB_l1978_197863

noncomputable def ellipse_foci (A B : ℝ) : Prop :=
  B^2 - A^2 = 25

noncomputable def hyperbola_foci (A B : ℝ) : Prop :=
  A^2 + B^2 = 64

theorem abs_AB (A B : ℝ) (h1 : ellipse_foci A B) (h2 : hyperbola_foci A B) :
  |A * B| = Real.sqrt 867.75 := 
sorry

end NUMINAMATH_GPT_abs_AB_l1978_197863


namespace NUMINAMATH_GPT_calculation_l1978_197819

theorem calculation :
  12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l1978_197819


namespace NUMINAMATH_GPT_minimum_pipe_length_l1978_197893

theorem minimum_pipe_length 
  (M S : ℝ × ℝ) 
  (horiz_dist : abs (M.1 - S.1) = 160)
  (vert_dist : abs (M.2 - S.2) = 120) :
  dist M S = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_pipe_length_l1978_197893


namespace NUMINAMATH_GPT_find_removed_number_l1978_197887

def list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def target_average : ℝ := 8.2

theorem find_removed_number (n : ℕ) (h : n ∈ list) :
  (list.sum - n) / (list.length - 1) = target_average -> n = 5 := by
  sorry

end NUMINAMATH_GPT_find_removed_number_l1978_197887


namespace NUMINAMATH_GPT_ellipse_foci_distance_l1978_197885

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l1978_197885


namespace NUMINAMATH_GPT_doughnuts_remaining_l1978_197867

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_doughnuts_remaining_l1978_197867


namespace NUMINAMATH_GPT_ratio_Lisa_Charlotte_l1978_197899

def P_tot : ℕ := 100
def Pat_money : ℕ := 6
def Lisa_money : ℕ := 5 * Pat_money
def additional_required : ℕ := 49
def current_total_money : ℕ := P_tot - additional_required
def Pat_Lisa_total : ℕ := Pat_money + Lisa_money
def Charlotte_money : ℕ := current_total_money - Pat_Lisa_total

theorem ratio_Lisa_Charlotte : (Lisa_money : ℕ) / Charlotte_money = 2 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_ratio_Lisa_Charlotte_l1978_197899


namespace NUMINAMATH_GPT_factorize_expression_l1978_197851

variable {R : Type} [Ring R]
variables (a b x y : R)

theorem factorize_expression :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1978_197851


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1978_197880

theorem boat_speed_in_still_water
  (V_s : ℝ) (t : ℝ) (d : ℝ) (V_b : ℝ)
  (h_stream_speed : V_s = 4)
  (h_travel_time : t = 7)
  (h_distance : d = 196)
  (h_downstream_speed : d / t = V_b + V_s) :
  V_b = 24 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1978_197880


namespace NUMINAMATH_GPT_ted_candy_bars_l1978_197816

theorem ted_candy_bars (b : ℕ) (n : ℕ) (h : b = 5) (h2 : n = 3) : b * n = 15 :=
by
  sorry

end NUMINAMATH_GPT_ted_candy_bars_l1978_197816


namespace NUMINAMATH_GPT_total_apples_picked_l1978_197804

def number_of_children : Nat := 33
def apples_per_child : Nat := 10
def number_of_adults : Nat := 40
def apples_per_adult : Nat := 3

theorem total_apples_picked :
  (number_of_children * apples_per_child) + (number_of_adults * apples_per_adult) = 450 := by
  -- You need to provide proof here
  sorry

end NUMINAMATH_GPT_total_apples_picked_l1978_197804


namespace NUMINAMATH_GPT_fruit_salad_cherries_l1978_197834

theorem fruit_salad_cherries (b r g c : ℕ) 
  (h1 : b + r + g + c = 390)
  (h2 : r = 3 * b)
  (h3 : g = 2 * c)
  (h4 : c = 5 * r) :
  c = 119 :=
by
  sorry

end NUMINAMATH_GPT_fruit_salad_cherries_l1978_197834


namespace NUMINAMATH_GPT_population_net_increase_l1978_197897

-- Definitions for birth and death rate, and the number of seconds in a day
def birth_rate : ℕ := 10
def death_rate : ℕ := 2
def seconds_in_day : ℕ := 86400

-- Calculate the population net increase in one day
theorem population_net_increase (birth_rate death_rate seconds_in_day : ℕ) :
  (seconds_in_day / 2) * birth_rate - (seconds_in_day / 2) * death_rate = 345600 :=
by
  sorry

end NUMINAMATH_GPT_population_net_increase_l1978_197897


namespace NUMINAMATH_GPT_chess_tournament_games_l1978_197890

theorem chess_tournament_games (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end NUMINAMATH_GPT_chess_tournament_games_l1978_197890


namespace NUMINAMATH_GPT_ratio_is_one_half_l1978_197892

noncomputable def ratio_dresses_with_pockets (D : ℕ) (total_pockets : ℕ) (pockets_two : ℕ) (pockets_three : ℕ) :=
  ∃ (P : ℕ), D = 24 ∧ total_pockets = 32 ∧
  (P / 3) * 2 + (2 * P / 3) * 3 = total_pockets ∧ 
  P / D = 1 / 2

theorem ratio_is_one_half :
  ratio_dresses_with_pockets 24 32 2 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_is_one_half_l1978_197892


namespace NUMINAMATH_GPT_percentage_greater_than_l1978_197813

-- Definitions of the variables involved
variables (X Y Z : ℝ)

-- Lean statement to prove the formula
theorem percentage_greater_than (X Y Z : ℝ) : 
  (100 * (X - Y)) / (Y + Z) = (100 * (X - Y)) / (Y + Z) :=
by
  -- skipping the actual proof
  sorry

end NUMINAMATH_GPT_percentage_greater_than_l1978_197813


namespace NUMINAMATH_GPT_length_of_segment_cutoff_l1978_197869

-- Define the parabola equation
def parabola (x y : ℝ) := y^2 = 4 * (x + 1)

-- Define the line passing through the focus and perpendicular to the x-axis
def line_through_focus_perp_x_axis (x y : ℝ) := x = 0

-- The actual length calculation lemma
lemma segment_length : 
  ∀ (x y : ℝ), parabola x y → line_through_focus_perp_x_axis x y → y = 2 ∨ y = -2 :=
by sorry

-- The final theorem which gives the length of the segment
theorem length_of_segment_cutoff (y1 y2 : ℝ) :
  ∀ (x : ℝ), parabola x y1 → parabola x y2 → line_through_focus_perp_x_axis x y1 → line_through_focus_perp_x_axis x y2 → (y1 = 2 ∨ y1 = -2) ∧ (y2 = 2 ∨ y2 = -2) → abs (y2 - y1) = 4 :=
by sorry

end NUMINAMATH_GPT_length_of_segment_cutoff_l1978_197869


namespace NUMINAMATH_GPT_cube_colorings_distinguishable_l1978_197888

-- Define the problem
def cube_construction_distinguishable_ways : Nat :=
  30

-- The theorem we need to prove
theorem cube_colorings_distinguishable :
  ∃ (ways : Nat), ways = cube_construction_distinguishable_ways :=
by
  sorry

end NUMINAMATH_GPT_cube_colorings_distinguishable_l1978_197888


namespace NUMINAMATH_GPT_factorize_x_squared_minus_121_l1978_197835

theorem factorize_x_squared_minus_121 (x : ℝ) : (x^2 - 121) = (x + 11) * (x - 11) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_121_l1978_197835


namespace NUMINAMATH_GPT_ratio_of_goals_l1978_197898

-- The conditions
def first_period_goals_kickers : ℕ := 2
def second_period_goals_kickers := 4
def first_period_goals_spiders := first_period_goals_kickers / 2
def second_period_goals_spiders := 2 * second_period_goals_kickers
def total_goals := first_period_goals_kickers + second_period_goals_kickers + first_period_goals_spiders + second_period_goals_spiders

-- The ratio to prove
def ratio_goals : ℕ := second_period_goals_kickers / first_period_goals_kickers

theorem ratio_of_goals : total_goals = 15 → ratio_goals = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_of_goals_l1978_197898


namespace NUMINAMATH_GPT_solution_set_for_inequality_l1978_197841

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

theorem solution_set_for_inequality :
  {x : ℝ | f x >= -2} = {x | -2 <= x ∧ x < 1 ∨ x >= 3} := sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l1978_197841


namespace NUMINAMATH_GPT_algebraic_identity_l1978_197855

theorem algebraic_identity (a b c d : ℝ) : a - b + c - d = a + c - (b + d) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l1978_197855


namespace NUMINAMATH_GPT_trigonometric_identity_l1978_197827

theorem trigonometric_identity :
  (Real.cos (17 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) + 
   Real.sin (163 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1978_197827


namespace NUMINAMATH_GPT_min_fraction_value_l1978_197838

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem min_fraction_value : ∀ x ∈ (Set.Ici (7 / 4)), (f x)^2 + 2 / (f x) ≥ 81 / 28 :=
by
  sorry

end NUMINAMATH_GPT_min_fraction_value_l1978_197838


namespace NUMINAMATH_GPT_two_pow_geq_n_cubed_for_n_geq_ten_l1978_197820

theorem two_pow_geq_n_cubed_for_n_geq_ten (n : ℕ) (hn : n ≥ 10) : 2^n ≥ n^3 := 
sorry

end NUMINAMATH_GPT_two_pow_geq_n_cubed_for_n_geq_ten_l1978_197820


namespace NUMINAMATH_GPT_range_of_x_sqrt_4_2x_l1978_197832

theorem range_of_x_sqrt_4_2x (x : ℝ) : (4 - 2 * x ≥ 0) ↔ (x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_sqrt_4_2x_l1978_197832


namespace NUMINAMATH_GPT_find_m_value_l1978_197876

theorem find_m_value (m : ℤ) : (∃ a : ℤ, x^2 + 2 * (m + 1) * x + 25 = (x + a)^2) ↔ (m = 4 ∨ m = -6) := 
sorry

end NUMINAMATH_GPT_find_m_value_l1978_197876


namespace NUMINAMATH_GPT_quadratic_roots_l1978_197853

theorem quadratic_roots (a b: ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (root_condition1 : a * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (root_condition2 : a * (1/3)^2 + b * (1/3) + 2 = 0) 
  : a - b = -10 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_l1978_197853


namespace NUMINAMATH_GPT_distance_AB_polar_l1978_197861

open Real

/-- The distance between points A and B in polar coordinates, given that θ₁ - θ₂ = π. -/
theorem distance_AB_polar (A B : ℝ × ℝ) (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hA : A = (r1, θ1)) (hB : B = (r2, θ2)) (hθ : θ1 - θ2 = π) :
  dist (r1 * cos θ1, r1 * sin θ1) (r2 * cos θ2, r2 * sin θ2) = r1 + r2 :=
sorry

end NUMINAMATH_GPT_distance_AB_polar_l1978_197861


namespace NUMINAMATH_GPT_Alan_ate_1_fewer_pretzel_than_John_l1978_197859

/-- Given that there are 95 pretzels in a bowl, John ate 28 pretzels, 
Marcus ate 12 more pretzels than John, and Marcus ate 40 pretzels,
prove that Alan ate 1 fewer pretzel than John. -/
theorem Alan_ate_1_fewer_pretzel_than_John 
  (h95 : 95 = 95)
  (John_ate : 28 = 28)
  (Marcus_ate_more : ∀ (x : ℕ), 40 = x + 12 → x = 28)
  (Marcus_ate : 40 = 40) :
  ∃ (Alan : ℕ), Alan = 27 ∧ 28 - Alan = 1 :=
by
  sorry

end NUMINAMATH_GPT_Alan_ate_1_fewer_pretzel_than_John_l1978_197859


namespace NUMINAMATH_GPT_money_left_after_purchase_l1978_197864

def initial_money : ℕ := 7
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := 3

def total_spent : ℕ := cost_candy_bar + cost_chocolate
def money_left : ℕ := initial_money - total_spent

theorem money_left_after_purchase : 
  money_left = 2 := by
  sorry

end NUMINAMATH_GPT_money_left_after_purchase_l1978_197864


namespace NUMINAMATH_GPT_solve_for_x_l1978_197844

theorem solve_for_x (x : ℝ) (h : |x - 2| = |x - 3| + 1) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1978_197844


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1978_197802

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1978_197802


namespace NUMINAMATH_GPT_apples_preference_count_l1978_197865

theorem apples_preference_count (total_people : ℕ) (total_angle : ℝ) (apple_angle : ℝ) 
  (h_total_people : total_people = 530) 
  (h_total_angle : total_angle = 360) 
  (h_apple_angle : apple_angle = 285) : 
  round ((total_people : ℝ) * (apple_angle / total_angle)) = 419 := 
by 
  sorry

end NUMINAMATH_GPT_apples_preference_count_l1978_197865


namespace NUMINAMATH_GPT_total_items_18_l1978_197824

-- Define the number of dogs, biscuits per dog, and boots per set
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 5
def boots_per_set : ℕ := 4

-- Calculate the total number of items
def total_items (num_dogs biscuits_per_dog boots_per_set : ℕ) : ℕ :=
  (num_dogs * biscuits_per_dog) + (num_dogs * boots_per_set)

-- Prove that the total number of items is 18
theorem total_items_18 : total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  -- Proof is not provided
  sorry

end NUMINAMATH_GPT_total_items_18_l1978_197824


namespace NUMINAMATH_GPT_total_spent_on_pens_l1978_197850

/-- Dorothy, Julia, and Robert go to the store to buy school supplies.
    Dorothy buys half as many pens as Julia.
    Julia buys three times as many pens as Robert.
    Robert buys 4 pens.
    The cost of one pen is $1.50.
    Prove that the total amount of money spent on pens by the three friends is $33. 
-/
theorem total_spent_on_pens :
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen = 33 := 
by
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  sorry

end NUMINAMATH_GPT_total_spent_on_pens_l1978_197850


namespace NUMINAMATH_GPT_sum_of_digits_is_twenty_l1978_197811

theorem sum_of_digits_is_twenty (a b c d : ℕ) (h1 : c + b = 9) (h2 : a + d = 10) 
  (H1 : a ≠ b) (H2 : a ≠ c) (H3 : a ≠ d) 
  (H4 : b ≠ c) (H5 : b ≠ d) (H6 : c ≠ d) :
  a + b + c + d = 20 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_is_twenty_l1978_197811


namespace NUMINAMATH_GPT_boxes_of_apples_with_cherries_l1978_197810

-- Define everything in the conditions
variable (A P Sp Sa : ℕ)
variable (box_cherries box_apples : ℕ)

-- Given conditions
axiom price_relation : 2 * P = 3 * A
axiom size_relation  : Sa = 12 * Sp
axiom cherries_per_box : box_cherries = 12

-- The problem statement (to be proved)
theorem boxes_of_apples_with_cherries : box_apples * A = box_cherries * P → box_apples = 18 :=
by
  sorry

end NUMINAMATH_GPT_boxes_of_apples_with_cherries_l1978_197810


namespace NUMINAMATH_GPT_hiker_total_distance_l1978_197815

-- Define conditions based on the problem description
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed : ℕ := day1_speed + 1
def day1_time : ℕ := day1_distance / day1_speed
def day2_time : ℕ := day1_time - 1
def day3_speed : ℕ := 5
def day3_time : ℕ := 3

-- Define the total distance walked based on the conditions
def total_distance : ℕ :=
  day1_distance + (day2_speed * day2_time) + (day3_speed * day3_time)

-- The theorem stating the hiker walked a total of 53 miles
theorem hiker_total_distance : total_distance = 53 := by
  sorry

end NUMINAMATH_GPT_hiker_total_distance_l1978_197815


namespace NUMINAMATH_GPT_vertical_asymptote_c_values_l1978_197879

theorem vertical_asymptote_c_values (c : ℝ) :
  (∃ x : ℝ, (x^2 - x - 6) = 0 ∧ (x^2 - 2*x + c) ≠ 0 ∧ ∀ y : ℝ, ((y ≠ x) → (x ≠ 3) ∧ (x ≠ -2)))
  → (c = -3 ∨ c = -8) :=
by sorry

end NUMINAMATH_GPT_vertical_asymptote_c_values_l1978_197879


namespace NUMINAMATH_GPT_one_fourth_of_56_equals_75_l1978_197809

theorem one_fourth_of_56_equals_75 : (5.6 / 4) = 7 / 5 := 
by
  -- Temporarily omitting the actual proof
  sorry

end NUMINAMATH_GPT_one_fourth_of_56_equals_75_l1978_197809


namespace NUMINAMATH_GPT_boat_upstream_time_is_1_5_hours_l1978_197803

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_boat_upstream_time_is_1_5_hours_l1978_197803


namespace NUMINAMATH_GPT_returning_players_l1978_197842

-- Definitions of conditions
def num_groups : Nat := 9
def players_per_group : Nat := 6
def new_players : Nat := 48

-- Definition of total number of players
def total_players : Nat := num_groups * players_per_group

-- Theorem: Find the number of returning players
theorem returning_players :
  total_players - new_players = 6 :=
by
  sorry

end NUMINAMATH_GPT_returning_players_l1978_197842


namespace NUMINAMATH_GPT_kiran_currency_notes_l1978_197808

theorem kiran_currency_notes :
  ∀ (n50_amount n100_amount total50 total100 : ℝ),
    n50_amount = 3500 →
    total50 = 5000 →
    total100 = 5000 - 3500 →
    n100_amount = total100 →
    (n50_amount / 50 + total100 / 100) = 85 :=
by
  intros n50_amount n100_amount total50 total100 n50_amount_eq total50_eq total100_eq n100_amount_eq
  sorry

end NUMINAMATH_GPT_kiran_currency_notes_l1978_197808


namespace NUMINAMATH_GPT_arctan_sum_l1978_197857

theorem arctan_sum : 
  let x := (3 : ℝ) / 7
  let y := 7 / 3
  x * y = 1 → (Real.arctan x + Real.arctan y = Real.pi / 2) :=
by
  intros x y h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arctan_sum_l1978_197857


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_l1978_197878

-- Define the two points the line passes through
structure Point (α : Type) :=
(x : α)
(y : α)

def p1 : Point ℤ := Point.mk 2 9
def p2 : Point ℤ := Point.mk 4 13

-- Define the function that describes the point where the line intersects the y-axis
def y_intercept : Point ℤ :=
  -- We are proving that the line intersects the y-axis at the point (0, 5)
  Point.mk 0 5

-- State the theorem to be proven
theorem line_intersects_y_axis_at (p1 p2 : Point ℤ) (yi : Point ℤ) :
  p1.x = 2 ∧ p1.y = 9 ∧ p2.x = 4 ∧ p2.y = 13 → yi = Point.mk 0 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_l1978_197878


namespace NUMINAMATH_GPT_percent_increase_l1978_197837

-- Definitions based on conditions
def initial_price : ℝ := 10
def final_price : ℝ := 15

-- Goal: Prove that the percent increase in the price per share is 50%
theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 50 := 
by
  sorry  -- Proof is not required, so we skip it with sorry.

end NUMINAMATH_GPT_percent_increase_l1978_197837


namespace NUMINAMATH_GPT_min_value_proof_l1978_197839

noncomputable def min_value (t c : ℝ) :=
  (t^2 + c^2 - 2 * t * c + 2 * c^2) / 2

theorem min_value_proof (a b t c : ℝ) (h : a + b = t) :
  (a^2 + (b + c)^2) ≥ min_value t c :=
by
  sorry

end NUMINAMATH_GPT_min_value_proof_l1978_197839


namespace NUMINAMATH_GPT_unique_solution_l1978_197873

theorem unique_solution (a b : ℤ) : 
  (a^6 + 1 ∣ b^11 - 2023 * b^3 + 40 * b) ∧ (a^4 - 1 ∣ b^10 - 2023 * b^2 - 41) 
  ↔ (a = 0 ∧ ∃ c : ℤ, b = c) := 
by 
  sorry

end NUMINAMATH_GPT_unique_solution_l1978_197873


namespace NUMINAMATH_GPT_algebraic_expression_zero_iff_x_eq_2_l1978_197849

theorem algebraic_expression_zero_iff_x_eq_2 (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (1 / (x - 1) + 3 / (1 - x^2) = 0) ↔ (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_zero_iff_x_eq_2_l1978_197849


namespace NUMINAMATH_GPT_initial_percentage_of_milk_l1978_197843

theorem initial_percentage_of_milk (M : ℝ) (H1 : M / 100 * 60 = 0.58 * 86.9) : M = 83.99 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_milk_l1978_197843


namespace NUMINAMATH_GPT_quadratic_coefficients_l1978_197807

theorem quadratic_coefficients : 
  ∀ (b k : ℝ), (∀ x : ℝ, x^2 + b * x + 5 = (x - 2)^2 + k) → b = -4 ∧ k = 1 :=
by
  intro b k h
  have h1 := h 0
  have h2 := h 1
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1978_197807


namespace NUMINAMATH_GPT_lines_intersection_l1978_197860

theorem lines_intersection :
  ∃ (x y : ℝ), 
  (3 * y = -2 * x + 6) ∧ 
  (-4 * y = 3 * x + 4) ∧ 
  (x = -36) ∧ 
  (y = 26) :=
sorry

end NUMINAMATH_GPT_lines_intersection_l1978_197860


namespace NUMINAMATH_GPT_total_amount_paid_l1978_197856

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1978_197856


namespace NUMINAMATH_GPT_bakery_flour_total_l1978_197826

theorem bakery_flour_total :
  (0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6) :=
by {
  sorry
}

end NUMINAMATH_GPT_bakery_flour_total_l1978_197826


namespace NUMINAMATH_GPT_cylinder_original_radius_l1978_197830

theorem cylinder_original_radius
    (r h: ℝ)
    (h₀: h = 4)
    (h₁: π * (r + 8)^2 * 4 = π * r^2 * 12) :
    r = 12 :=
by
  -- Insert your proof here
  sorry

end NUMINAMATH_GPT_cylinder_original_radius_l1978_197830


namespace NUMINAMATH_GPT_luke_piles_coins_l1978_197814

theorem luke_piles_coins (x : ℕ) (h_total_piles : 10 = 5 + 5) (h_total_coins : 10 * x = 30) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_luke_piles_coins_l1978_197814


namespace NUMINAMATH_GPT_cement_used_tess_street_l1978_197862

-- Define the given conditions
def cement_used_lexi_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Define the statement to prove the amount of cement used for Tess's street
theorem cement_used_tess_street : total_cement_used - cement_used_lexi_street = 5.1 :=
by
  sorry

end NUMINAMATH_GPT_cement_used_tess_street_l1978_197862


namespace NUMINAMATH_GPT_ice_cream_ratio_l1978_197866

theorem ice_cream_ratio
    (T : ℕ)
    (W : ℕ)
    (hT : T = 12000)
    (hMultiple : ∃ k : ℕ, W = k * T)
    (hTotal : T + W = 36000) :
    W / T = 2 :=
by
  -- Proof is omitted, so sorry is used
  sorry

end NUMINAMATH_GPT_ice_cream_ratio_l1978_197866


namespace NUMINAMATH_GPT_num_factors_180_l1978_197817

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end NUMINAMATH_GPT_num_factors_180_l1978_197817


namespace NUMINAMATH_GPT_algebraic_expr_value_l1978_197812

theorem algebraic_expr_value {a b : ℝ} (h: a + b = 1) : a^2 - b^2 + 2 * b + 9 = 10 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expr_value_l1978_197812


namespace NUMINAMATH_GPT_factorial_expression_l1978_197822

open Nat

theorem factorial_expression : ((sqrt (5! * 4!)) ^ 2 + 3!) = 2886 := by
  sorry

end NUMINAMATH_GPT_factorial_expression_l1978_197822


namespace NUMINAMATH_GPT_sum_of_eighth_powers_of_roots_l1978_197891

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root_disc := Real.sqrt discriminant
  ((-b + root_disc) / (2 * a), (-b - root_disc) / (2 * a))

theorem sum_of_eighth_powers_of_roots :
  let (p, q) := quadratic_roots 1 (-Real.sqrt 7) 1
  p^2 + q^2 = 5 ∧ p^4 + q^4 = 23 ∧ p^8 + q^8 = 527 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eighth_powers_of_roots_l1978_197891


namespace NUMINAMATH_GPT_VincentLearnedAtCamp_l1978_197886

def VincentSongsBeforeSummerCamp : ℕ := 56
def VincentSongsAfterSummerCamp : ℕ := 74

theorem VincentLearnedAtCamp :
  VincentSongsAfterSummerCamp - VincentSongsBeforeSummerCamp = 18 := by
  sorry

end NUMINAMATH_GPT_VincentLearnedAtCamp_l1978_197886


namespace NUMINAMATH_GPT_bounded_harmonic_is_constant_l1978_197806

noncomputable def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ (x y : ℤ), f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1) = 4 * f (x, y)

theorem bounded_harmonic_is_constant (f : ℤ × ℤ → ℝ) (M : ℝ) 
  (h_bound : ∀ (x y : ℤ), |f (x, y)| ≤ M)
  (h_harmonic : is_harmonic f) :
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c :=
sorry

end NUMINAMATH_GPT_bounded_harmonic_is_constant_l1978_197806


namespace NUMINAMATH_GPT_tangent_line_y_intercept_l1978_197847

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end NUMINAMATH_GPT_tangent_line_y_intercept_l1978_197847


namespace NUMINAMATH_GPT_alan_carla_weight_l1978_197846

variable (a b c d : ℝ)

theorem alan_carla_weight (h1 : a + b = 280) (h2 : b + c = 230) (h3 : c + d = 250) (h4 : a + d = 300) :
  a + c = 250 := by
sorry

end NUMINAMATH_GPT_alan_carla_weight_l1978_197846


namespace NUMINAMATH_GPT_log_relation_l1978_197874

theorem log_relation (a b : ℝ) (log_7 : ℝ → ℝ) (log_6 : ℝ → ℝ) (log_6_343 : log_6 343 = a) (log_7_18 : log_7 18 = b) :
  a = 6 / (b + 2 * log_7 2) :=
by
  sorry

end NUMINAMATH_GPT_log_relation_l1978_197874


namespace NUMINAMATH_GPT_original_price_of_shoes_l1978_197845

theorem original_price_of_shoes (P : ℝ) (h1 : 2 * 0.60 * P + 0.80 * 100 = 140) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l1978_197845


namespace NUMINAMATH_GPT_largest_square_area_l1978_197828

theorem largest_square_area (XY YZ XZ : ℝ)
  (h1 : XZ^2 = XY^2 + YZ^2)
  (h2 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_area_l1978_197828


namespace NUMINAMATH_GPT_problem_statement_l1978_197889

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1978_197889


namespace NUMINAMATH_GPT_sum_denominators_l1978_197896

theorem sum_denominators (a b: ℕ) (h_coprime : Nat.gcd a b = 1) :
  (3:ℚ) / (5 * b) + (2:ℚ) / (9 * b) + (4:ℚ) / (15 * b) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 :=
by
  sorry

end NUMINAMATH_GPT_sum_denominators_l1978_197896


namespace NUMINAMATH_GPT_min_cost_at_100_l1978_197884

noncomputable def cost_function (v : ℝ) : ℝ :=
if (0 < v ∧ v ≤ 50) then (123000 / v + 690)
else if (v > 50) then (3 * v^2 / 50 + 120000 / v + 600)
else 0

theorem min_cost_at_100 : ∃ v : ℝ, v = 100 ∧ cost_function v = 2400 :=
by
  -- We are not proving but stating the theorem here
  sorry

end NUMINAMATH_GPT_min_cost_at_100_l1978_197884


namespace NUMINAMATH_GPT_range_of_m_l1978_197821

def P (x : ℝ) : Prop := |(4 - x) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) (h : m > 0) : (∀ x, ¬P x → ¬q x m) → m ≥ 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_m_l1978_197821


namespace NUMINAMATH_GPT_k_value_for_inequality_l1978_197825

theorem k_value_for_inequality :
    (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d)) ∧
    (∀ k : ℝ, (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) → k = 3/4) :=
sorry

end NUMINAMATH_GPT_k_value_for_inequality_l1978_197825


namespace NUMINAMATH_GPT_june_earnings_l1978_197833

theorem june_earnings 
    (total_clovers : ℕ := 300)
    (pct_3_petals : ℕ := 70)
    (pct_2_petals : ℕ := 20)
    (pct_4_petals : ℕ := 8)
    (pct_5_petals : ℕ := 2)
    (earn_3_petals : ℕ := 1)
    (earn_2_petals : ℕ := 2)
    (earn_4_petals : ℕ := 5)
    (earn_5_petals : ℕ := 10)
    (earn_total : ℕ := 510) : 
  (pct_3_petals * total_clovers) / 100 * earn_3_petals + 
  (pct_2_petals * total_clovers) / 100 * earn_2_petals + 
  (pct_4_petals * total_clovers) / 100 * earn_4_petals + 
  (pct_5_petals * total_clovers) / 100 * earn_5_petals = earn_total := 
by
  -- Proof of this theorem involves calculating each part and summing them. Skipping detailed steps with sorry.
  sorry

end NUMINAMATH_GPT_june_earnings_l1978_197833


namespace NUMINAMATH_GPT_terminating_decimal_count_l1978_197800

def count_terminating_decimals (n: ℕ): ℕ :=
  (n / 17)

theorem terminating_decimal_count : count_terminating_decimals 493 = 29 := by
  sorry

end NUMINAMATH_GPT_terminating_decimal_count_l1978_197800


namespace NUMINAMATH_GPT_possible_lengths_of_c_l1978_197831

-- Definitions of the given conditions
variables (a b c : ℝ) (S : ℝ)
variables (h₁ : a = 4)
variables (h₂ : b = 5)
variables (h₃ : S = 5 * Real.sqrt 3)

-- The main theorem stating the possible lengths of c
theorem possible_lengths_of_c : c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
  sorry

end NUMINAMATH_GPT_possible_lengths_of_c_l1978_197831


namespace NUMINAMATH_GPT_number_of_customers_who_did_not_want_tires_change_l1978_197894

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end NUMINAMATH_GPT_number_of_customers_who_did_not_want_tires_change_l1978_197894


namespace NUMINAMATH_GPT_jellybean_total_l1978_197895

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end NUMINAMATH_GPT_jellybean_total_l1978_197895


namespace NUMINAMATH_GPT_additional_investment_interest_rate_l1978_197871

theorem additional_investment_interest_rate :
  let initial_investment := 2400
  let initial_rate := 0.05
  let additional_investment := 600
  let total_investment := initial_investment + additional_investment
  let desired_total_income := 0.06 * total_investment
  let income_from_initial := initial_rate * initial_investment
  let additional_income_needed := desired_total_income - income_from_initial
  let additional_rate := additional_income_needed / additional_investment
  additional_rate * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_additional_investment_interest_rate_l1978_197871


namespace NUMINAMATH_GPT_veranda_area_l1978_197875

theorem veranda_area (room_length room_width veranda_length_width veranda_width_width : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_width = 2.5)
  (h4 : veranda_width_width = 3)
  : (room_length + 2 * veranda_length_width) * (room_width + 2 * veranda_width_width) - room_length * room_width = 204 :=
by
  simp [h1, h2, h3, h4]
  norm_num
  done

end NUMINAMATH_GPT_veranda_area_l1978_197875


namespace NUMINAMATH_GPT_parabola_shift_l1978_197877

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end NUMINAMATH_GPT_parabola_shift_l1978_197877


namespace NUMINAMATH_GPT_juice_expense_l1978_197829

theorem juice_expense (M P : ℕ) 
  (h1 : M + P = 17) 
  (h2 : 5 * M + 6 * P = 94) : 6 * P = 54 :=
by 
  sorry

end NUMINAMATH_GPT_juice_expense_l1978_197829


namespace NUMINAMATH_GPT_sum_of_squares_not_perfect_square_l1978_197848

theorem sum_of_squares_not_perfect_square (n : ℤ) : ¬ (∃ k : ℤ, k^2 = (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_not_perfect_square_l1978_197848


namespace NUMINAMATH_GPT_roots_of_equation_l1978_197836

theorem roots_of_equation (
  x y: ℝ
) (h1: x + y = 10) (h2: |x - y| = 12):
  (x = 11 ∧ y = -1) ∨ (x = -1 ∧ y = 11) ↔ ∃ (a b: ℝ), a = 11 ∧ b = -1 ∨ a = -1 ∧ b = 11 ∧ a^2 - 10*a - 22 = 0 ∧ b^2 - 10*b - 22 = 0 := 
by sorry

end NUMINAMATH_GPT_roots_of_equation_l1978_197836


namespace NUMINAMATH_GPT_sequence_term_2023_l1978_197854

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end NUMINAMATH_GPT_sequence_term_2023_l1978_197854


namespace NUMINAMATH_GPT_reciprocal_of_neg_1_point_5_l1978_197868

theorem reciprocal_of_neg_1_point_5 : (1 / (-1.5) = -2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_1_point_5_l1978_197868


namespace NUMINAMATH_GPT_value_of_expression_l1978_197818

variables {a b c d e f : ℝ}

theorem value_of_expression
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1978_197818


namespace NUMINAMATH_GPT_exact_days_two_friends_visit_l1978_197801

-- Define the periodicities of Alice, Beatrix, and Claire
def periodicity_alice : ℕ := 1
def periodicity_beatrix : ℕ := 5
def periodicity_claire : ℕ := 7

-- Define the total days to be considered
def total_days : ℕ := 180

-- Define the number of days three friends visit together
def lcm_ab := Nat.lcm periodicity_alice periodicity_beatrix
def lcm_ac := Nat.lcm periodicity_alice periodicity_claire
def lcm_bc := Nat.lcm periodicity_beatrix periodicity_claire
def lcm_abc := Nat.lcm lcm_ab periodicity_claire

-- Define the counts of visitations
def count_ab := total_days / lcm_ab - total_days / lcm_abc
def count_ac := total_days / lcm_ac - total_days / lcm_abc
def count_bc := total_days / lcm_bc - total_days / lcm_abc

-- Finally calculate the number of days exactly two friends visit together
def days_two_friends_visit : ℕ := count_ab + count_ac + count_bc

-- The theorem to prove
theorem exact_days_two_friends_visit : days_two_friends_visit = 51 :=
by 
  -- This is where the actual proof would go
  sorry

end NUMINAMATH_GPT_exact_days_two_friends_visit_l1978_197801


namespace NUMINAMATH_GPT_find_number_l1978_197883

theorem find_number (x : ℝ) :
  (10 + 30 + 50) / 3 = 30 →
  ((x + 40 + 6) / 3 = (10 + 30 + 50) / 3 - 8) →
  x = 20 :=
by
  intros h_avg1 h_avg2
  sorry

end NUMINAMATH_GPT_find_number_l1978_197883
