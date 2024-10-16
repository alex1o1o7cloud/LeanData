import Mathlib

namespace NUMINAMATH_CALUDE_senior_field_trip_l107_10791

theorem senior_field_trip :
  ∃! n : ℕ, n < 300 ∧ n % 17 = 15 ∧ n % 19 = 12 ∧ n = 202 := by
  sorry

end NUMINAMATH_CALUDE_senior_field_trip_l107_10791


namespace NUMINAMATH_CALUDE_group_division_ways_l107_10772

theorem group_division_ways (n : ℕ) (g₁ g₂ g₃ : ℕ) (h₁ : n = 8) (h₂ : g₁ = 2) (h₃ : g₂ = 3) (h₄ : g₃ = 3) :
  (Nat.choose n g₂ * Nat.choose (n - g₂) g₃) / 2 = 280 :=
by sorry

end NUMINAMATH_CALUDE_group_division_ways_l107_10772


namespace NUMINAMATH_CALUDE_trajectory_is_hyperbola_l107_10729

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the condition for the trajectory
def TrajectoryCondition (z : ℂ) : Prop :=
  Complex.abs (Complex.abs (z - 1) - Complex.abs (z + Complex.I)) = 1

-- Define a hyperbola in the complex plane
def IsHyperbola (S : Set ℂ) : Prop :=
  ∃ (F₁ F₂ : ℂ) (a : ℝ), a > 0 ∧ Complex.abs (F₁ - F₂) > 2 * a ∧
    S = {z : ℂ | Complex.abs (Complex.abs (z - F₁) - Complex.abs (z - F₂)) = 2 * a}

-- Theorem statement
theorem trajectory_is_hyperbola :
  IsHyperbola {z : ℂ | TrajectoryCondition z} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_hyperbola_l107_10729


namespace NUMINAMATH_CALUDE_final_amount_in_euros_l107_10787

/-- Represents the number of coins of each type -/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  half_dollars : ℕ
  one_dollar_coins : ℕ

/-- Calculates the total value of a coin collection in dollars -/
def collection_value (c : CoinCollection) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20) + 
  c.pennies * (1/100) + c.half_dollars * (1/2) + c.one_dollar_coins

/-- Rob's initial coin collection -/
def initial_collection : CoinCollection := {
  quarters := 7,
  dimes := 3,
  nickels := 5,
  pennies := 12,
  half_dollars := 3,
  one_dollar_coins := 2
}

/-- Removes one coin of each type from the collection -/
def remove_one_each (c : CoinCollection) : CoinCollection := {
  quarters := c.quarters - 1,
  dimes := c.dimes - 1,
  nickels := c.nickels - 1,
  pennies := c.pennies - 1,
  half_dollars := c.half_dollars - 1,
  one_dollar_coins := c.one_dollar_coins - 1
}

/-- Exchanges three nickels for two dimes -/
def exchange_nickels_for_dimes (c : CoinCollection) : CoinCollection := {
  c with
  nickels := c.nickels - 3,
  dimes := c.dimes + 2
}

/-- Exchanges a half-dollar for a quarter and two dimes -/
def exchange_half_dollar (c : CoinCollection) : CoinCollection := {
  c with
  half_dollars := c.half_dollars - 1,
  quarters := c.quarters + 1,
  dimes := c.dimes + 2
}

/-- Exchanges a one-dollar coin for fifty pennies -/
def exchange_dollar_for_pennies (c : CoinCollection) : CoinCollection := {
  c with
  one_dollar_coins := c.one_dollar_coins - 1,
  pennies := c.pennies + 50
}

/-- Converts dollars to euros -/
def dollars_to_euros (dollars : ℚ) : ℚ :=
  dollars * (85/100)

/-- The main theorem stating the final amount in euros -/
theorem final_amount_in_euros : 
  dollars_to_euros (collection_value (
    exchange_dollar_for_pennies (
      exchange_half_dollar (
        exchange_nickels_for_dimes (
          remove_one_each initial_collection
        )
      )
    )
  )) = 2.9835 := by
  sorry


end NUMINAMATH_CALUDE_final_amount_in_euros_l107_10787


namespace NUMINAMATH_CALUDE_intersection_equality_l107_10763

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.cos (Real.arccos p.1)}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.arccos (Real.cos p.1)}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}

-- Theorem statement
theorem intersection_equality : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l107_10763


namespace NUMINAMATH_CALUDE_similar_triangles_corresponding_side_length_l107_10777

/-- Given two similar right triangles, where the first triangle has a leg of 15 inches and a hypotenuse
    of 17 inches, and the second triangle has a hypotenuse of 34 inches, the length of the side in the
    second triangle corresponding to the 15-inch leg is 30 inches. -/
theorem similar_triangles_corresponding_side_length (a b c d : ℝ) : 
  a = 15 →  -- First leg of the first triangle
  c = 17 →  -- Hypotenuse of the first triangle
  d = 34 →  -- Hypotenuse of the second triangle
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the first triangle
  ∃ (k : ℝ), k > 0 ∧ d = k * c ∧ k * a = 30  -- The corresponding side in the second triangle is 30 inches
  := by sorry

end NUMINAMATH_CALUDE_similar_triangles_corresponding_side_length_l107_10777


namespace NUMINAMATH_CALUDE_hyperbola_properties_l107_10778

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptote equation
def is_asymptote (m : ℝ) : Prop := ∀ x y : ℝ, hyperbola x y → (y = m * x ∨ y = -m * x)

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := ∃ a c : ℝ, a > 0 ∧ c > 0 ∧ e = c / a ∧ ∀ x y : ℝ, hyperbola x y → x^2 / a^2 - y^2 / (c^2 - a^2) = 1

-- Theorem statement
theorem hyperbola_properties : is_asymptote (Real.sqrt 2) ∧ eccentricity (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l107_10778


namespace NUMINAMATH_CALUDE_circle_distance_range_l107_10768

theorem circle_distance_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  3 - 2 * Real.sqrt 2 ≤ x^2 - 2*x + y^2 + 2*y + 2 ∧ 
  x^2 - 2*x + y^2 + 2*y + 2 ≤ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_range_l107_10768


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l107_10764

theorem cosine_sine_identity : 
  (Real.cos (10 * π / 180)) / (2 * Real.sin (10 * π / 180)) - 2 * Real.cos (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l107_10764


namespace NUMINAMATH_CALUDE_motorboat_travel_theorem_l107_10726

noncomputable def motorboat_travel_fraction (S : ℝ) (v : ℝ) : Set ℝ :=
  let u₁ := (2 / 3) * v
  let u₂ := (1 / 3) * v
  let V_m₁ := 2 * v + u₁
  let V_m₂ := 2 * v + u₂
  let V_b₁ := 3 * v - u₁
  let V_b₂ := 3 * v - u₂
  let t₁ := S / (5 * v)
  let d := (56 / 225) * S
  { x | x = (V_m₁ * t₁ + d) / S ∨ x = (V_m₂ * t₁ + d) / S }

theorem motorboat_travel_theorem (S : ℝ) (v : ℝ) (h_S : S > 0) (h_v : v > 0) :
  motorboat_travel_fraction S v = {161 / 225, 176 / 225} := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_theorem_l107_10726


namespace NUMINAMATH_CALUDE_burger_cost_l107_10771

/-- Given Alice's and Charlie's purchases, prove the cost of a burger -/
theorem burger_cost :
  ∀ (burger_cost soda_cost : ℕ),
  5 * burger_cost + 3 * soda_cost = 500 →
  3 * burger_cost + 2 * soda_cost = 310 →
  burger_cost = 70 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l107_10771


namespace NUMINAMATH_CALUDE_solution_characterization_l107_10758

theorem solution_characterization (x y : ℝ) :
  (|x| + |y| = 1340) ∧ (x^3 + y^3 + 2010*x*y = 670^3) →
  (x + y = 670) ∧ (x * y = -673350) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l107_10758


namespace NUMINAMATH_CALUDE_work_completion_time_l107_10703

theorem work_completion_time (x_days y_days : ℕ) (x_remaining : ℕ) (y_worked : ℕ) : 
  x_days = 24 →
  y_worked = 10 →
  x_remaining = 9 →
  (y_worked : ℚ) / y_days + (x_remaining : ℚ) / x_days = 1 →
  y_days = 16 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l107_10703


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l107_10797

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l107_10797


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l107_10773

theorem nested_fraction_evaluation :
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l107_10773


namespace NUMINAMATH_CALUDE_power_mod_eight_l107_10740

theorem power_mod_eight : 3^23 ≡ 3 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_power_mod_eight_l107_10740


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_intersection_equals_A_iff_l107_10715

def A (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 3 }
def B := { x : ℝ | x < -1 ∨ x > 5 }

theorem intersection_when_a_is_3 :
  A 3 ∩ B = { x : ℝ | 5 < x ∧ x ≤ 6 } := by sorry

theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_intersection_equals_A_iff_l107_10715


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l107_10738

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 4 - 2*p.1}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l107_10738


namespace NUMINAMATH_CALUDE_gold_value_calculation_l107_10745

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value (legacy_bars : ℕ) (aleena_difference : ℕ) (bar_value : ℕ) : ℕ :=
  (legacy_bars + (legacy_bars - aleena_difference)) * bar_value

/-- Theorem stating the total value of gold for Legacy and Aleena -/
theorem gold_value_calculation :
  total_gold_value 12 4 3500 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_gold_value_calculation_l107_10745


namespace NUMINAMATH_CALUDE_panda_survival_probability_l107_10728

theorem panda_survival_probability (p_10 p_15 : ℝ) 
  (h1 : p_10 = 0.8) 
  (h2 : p_15 = 0.6) : 
  p_15 / p_10 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_panda_survival_probability_l107_10728


namespace NUMINAMATH_CALUDE_lillian_initial_candies_l107_10765

/-- The number of candies Lillian's father gave her -/
def candies_from_father : ℕ := 5

/-- The total number of candies Lillian has after receiving candies from her father -/
def total_candies : ℕ := 93

/-- The number of candies Lillian collected initially -/
def initial_candies : ℕ := total_candies - candies_from_father

theorem lillian_initial_candies :
  initial_candies = 88 :=
by sorry

end NUMINAMATH_CALUDE_lillian_initial_candies_l107_10765


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_is_28_7_l107_10744

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_light_rain : ℝ
  prob_heavy_rain : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate the expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_light_rain * f.light_rain_amount + f.prob_heavy_rain * f.heavy_rain_amount

/-- Calculate the expected total rainfall for a week -/
def expected_weekly_rainfall (f : DailyForecast) : ℝ :=
  7 * expected_daily_rainfall f

/-- The weather forecast for each day of the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
  , prob_light_rain := 0.3
  , prob_heavy_rain := 0.4
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

theorem expected_weekly_rainfall_is_28_7 :
  expected_weekly_rainfall weekly_forecast = 28.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_is_28_7_l107_10744


namespace NUMINAMATH_CALUDE_hyperbola_equation_l107_10756

def ellipse_equation (x y : ℝ) : Prop := x^2 + y^2/2 = 1

def hyperbola_vertices (h_vertices : ℝ × ℝ) (e_vertices : ℝ × ℝ) : Prop :=
  h_vertices = e_vertices

def eccentricity_product (e_hyperbola e_ellipse : ℝ) : Prop :=
  e_hyperbola * e_ellipse = 1

theorem hyperbola_equation 
  (h_vertices : ℝ × ℝ) 
  (e_vertices : ℝ × ℝ) 
  (e_hyperbola e_ellipse : ℝ) :
  hyperbola_vertices h_vertices e_vertices →
  eccentricity_product e_hyperbola e_ellipse →
  ∃ (x y : ℝ), y^2 - x^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l107_10756


namespace NUMINAMATH_CALUDE_triangle_longest_side_l107_10722

theorem triangle_longest_side (x : ℝ) : 
  9 + (2 * x + 3) + (3 * x - 2) = 45 →
  max 9 (max (2 * x + 3) (3 * x - 2)) = 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l107_10722


namespace NUMINAMATH_CALUDE_unique_intersection_point_l107_10750

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 28*x + 24

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l107_10750


namespace NUMINAMATH_CALUDE_third_offense_sentence_extension_l107_10746

theorem third_offense_sentence_extension (original_sentence total_time : ℕ) 
  (h1 : original_sentence = 27)
  (h2 : total_time = 36) :
  (total_time - original_sentence) / original_sentence = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_third_offense_sentence_extension_l107_10746


namespace NUMINAMATH_CALUDE_women_count_l107_10769

/-- Represents the work done by one woman in one day -/
def W : ℝ := sorry

/-- Represents the work done by one child in one day -/
def C : ℝ := sorry

/-- Represents the number of women working initially -/
def x : ℝ := sorry

/-- The total work to be completed -/
def total_work : ℝ := sorry

theorem women_count : x = 10 := by
  have h1 : 5 * x * W = total_work := sorry
  have h2 : 100 * C = total_work := sorry
  have h3 : 5 * (5 * W + 10 * C) = total_work := sorry
  sorry

end NUMINAMATH_CALUDE_women_count_l107_10769


namespace NUMINAMATH_CALUDE_joshes_investment_l107_10788

/-- Proves that the initial investment is $2000 given the conditions of Josh's investment scenario -/
theorem joshes_investment
  (initial_wallet : ℝ)
  (final_wallet : ℝ)
  (stock_increase : ℝ)
  (h1 : initial_wallet = 300)
  (h2 : final_wallet = 2900)
  (h3 : stock_increase = 0.3)
  : ∃ (investment : ℝ), 
    investment = 2000 ∧ 
    final_wallet = initial_wallet + investment * (1 + stock_increase) :=
by sorry

end NUMINAMATH_CALUDE_joshes_investment_l107_10788


namespace NUMINAMATH_CALUDE_polygon_diagonals_integer_l107_10795

theorem polygon_diagonals_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, (n * (n - 3) : ℤ) / 2 = k := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_integer_l107_10795


namespace NUMINAMATH_CALUDE_bird_migration_distance_l107_10705

/-- Calculates the total distance traveled by migrating birds over two seasons -/
theorem bird_migration_distance (num_birds : ℕ) (dist_jim_disney : ℝ) (dist_disney_london : ℝ) :
  num_birds = 20 →
  dist_jim_disney = 50 →
  dist_disney_london = 60 →
  num_birds * (dist_jim_disney + dist_disney_london) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l107_10705


namespace NUMINAMATH_CALUDE_largest_number_is_312_base_4_l107_10708

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_number_is_312_base_4 :
  let binary := [1, 1, 1, 1, 1]
  let ternary := [1, 2, 2, 1]
  let quaternary := [3, 1, 2]
  let octal := [5, 6]
  
  (base_to_decimal quaternary 4) = 54 ∧
  (base_to_decimal quaternary 4) > (base_to_decimal binary 2) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal ternary 3) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal octal 8) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_312_base_4_l107_10708


namespace NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l107_10716

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l107_10716


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l107_10710

/-- Represents the number of products in the sample -/
def sample_size : ℕ := 10

/-- Represents the event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Represents the complementary event of A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is "at most one defective product" -/
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ sample_size →
    (¬ event_A defective) ↔ complement_A defective := by
  sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l107_10710


namespace NUMINAMATH_CALUDE_max_difference_of_five_integers_l107_10779

theorem max_difference_of_five_integers (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 50 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e ≤ 58 →
  e - a ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_of_five_integers_l107_10779


namespace NUMINAMATH_CALUDE_sin_22_5_deg_identity_l107_10755

theorem sin_22_5_deg_identity : 1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_22_5_deg_identity_l107_10755


namespace NUMINAMATH_CALUDE_suji_age_l107_10770

theorem suji_age (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  ((abi_age + 3) : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end NUMINAMATH_CALUDE_suji_age_l107_10770


namespace NUMINAMATH_CALUDE_first_year_steve_exceeds_wayne_l107_10796

def steve_money (year : Nat) : Nat :=
  100 * 2^(year - 2000)

def wayne_money (year : Nat) : Nat :=
  10000 / 2^(year - 2000)

theorem first_year_steve_exceeds_wayne :
  (∀ y : Nat, y < 2004 → steve_money y ≤ wayne_money y) ∧
  steve_money 2004 > wayne_money 2004 := by
  sorry

end NUMINAMATH_CALUDE_first_year_steve_exceeds_wayne_l107_10796


namespace NUMINAMATH_CALUDE_product_three_reciprocal_squares_sum_l107_10717

theorem product_three_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 3 →
  (1 : ℚ) / (a : ℚ)^2 + (1 : ℚ) / (b : ℚ)^2 = 10 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_three_reciprocal_squares_sum_l107_10717


namespace NUMINAMATH_CALUDE_simplified_expression_equals_half_l107_10737

theorem simplified_expression_equals_half :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_half_l107_10737


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l107_10784

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/5 → b = -3/2 → -1 < m * b ∧ m * b < 0 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l107_10784


namespace NUMINAMATH_CALUDE_corner_stationery_sales_proportion_l107_10774

theorem corner_stationery_sales_proportion :
  let total_sales_percent : ℝ := 100
  let markers_percent : ℝ := 25
  let notebooks_percent : ℝ := 47
  total_sales_percent - (markers_percent + notebooks_percent) = 28 := by
sorry

end NUMINAMATH_CALUDE_corner_stationery_sales_proportion_l107_10774


namespace NUMINAMATH_CALUDE_sandbox_area_l107_10781

theorem sandbox_area (length width : ℕ) (h1 : length = 312) (h2 : width = 146) :
  length * width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l107_10781


namespace NUMINAMATH_CALUDE_total_carrots_l107_10713

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
sorry

end NUMINAMATH_CALUDE_total_carrots_l107_10713


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l107_10775

theorem initial_speed_calculation (distance : ℝ) (fast_speed : ℝ) (time_diff : ℝ) 
  (h1 : distance = 24)
  (h2 : fast_speed = 12)
  (h3 : time_diff = 2/3) : 
  ∃ v : ℝ, v > 0 ∧ distance / v - distance / fast_speed = time_diff ∧ v = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l107_10775


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l107_10739

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + 2*i) / i
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l107_10739


namespace NUMINAMATH_CALUDE_second_year_cost_difference_l107_10766

/-- Proves that the difference between second and first year payments is 2 --/
theorem second_year_cost_difference (total_payments : ℕ) (first_year : ℕ) (x : ℕ) :
  total_payments = 96 →
  first_year = 20 →
  total_payments = first_year + (first_year + x) + (first_year + x + 3) + (first_year + x + 3 + 4) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_year_cost_difference_l107_10766


namespace NUMINAMATH_CALUDE_base_number_irrelevant_l107_10747

def decimal_places (n : ℝ) : ℕ := sorry

theorem base_number_irrelevant (x : ℤ) :
  decimal_places ((x^4 * 3.456789)^14) = decimal_places (3.456789^14) := by sorry

end NUMINAMATH_CALUDE_base_number_irrelevant_l107_10747


namespace NUMINAMATH_CALUDE_repeating_decimal_457_proof_l107_10757

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c : ℚ) / 999

theorem repeating_decimal_457_proof :
  let x := RepeatingDecimal 4 5 7
  x = 457 / 999 ∧ 457 + 999 = 1456 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_457_proof_l107_10757


namespace NUMINAMATH_CALUDE_inequality_proof_l107_10798

theorem inequality_proof (x : ℝ) (h : x ≥ 5) :
  Real.sqrt (x - 2) - Real.sqrt (x - 3) < Real.sqrt (x - 4) - Real.sqrt (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l107_10798


namespace NUMINAMATH_CALUDE_revision_cost_per_page_revision_cost_is_four_l107_10783

/-- The cost per page for revision in a manuscript typing service --/
theorem revision_cost_per_page : ℝ → Prop :=
  fun x =>
    let total_pages : ℕ := 100
    let pages_revised_once : ℕ := 35
    let pages_revised_twice : ℕ := 15
    let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
    let initial_typing_cost_per_page : ℝ := 6
    let total_cost : ℝ := 860
    (initial_typing_cost_per_page * total_pages + x * pages_revised_once + 2 * x * pages_revised_twice = total_cost) →
    x = 4

theorem revision_cost_is_four : revision_cost_per_page 4 := by
  sorry

end NUMINAMATH_CALUDE_revision_cost_per_page_revision_cost_is_four_l107_10783


namespace NUMINAMATH_CALUDE_saree_stripe_ratio_l107_10724

theorem saree_stripe_ratio (brown_stripes : ℕ) (blue_stripes : ℕ) (gold_stripes : ℕ) :
  brown_stripes = 4 →
  gold_stripes = 3 * brown_stripes →
  blue_stripes = 60 →
  blue_stripes = gold_stripes →
  blue_stripes / gold_stripes = 5 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_saree_stripe_ratio_l107_10724


namespace NUMINAMATH_CALUDE_min_transportation_cost_l107_10733

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  inventory_A : ℕ := 12
  inventory_B : ℕ := 8
  demand_C : ℕ := 10
  demand_D : ℕ := 10
  cost_A_to_C : ℕ := 300
  cost_A_to_D : ℕ := 500
  cost_B_to_C : ℕ := 400
  cost_B_to_D : ℕ := 800

/-- The total cost function for the transportation problem. -/
def total_cost (tp : TransportationProblem) (x : ℕ) : ℕ :=
  200 * x + 8400

/-- The theorem stating that the minimum total transportation cost is 8800 yuan. -/
theorem min_transportation_cost (tp : TransportationProblem) :
  ∃ (x : ℕ), 2 ≤ x ∧ x ≤ 10 ∧ (∀ (y : ℕ), 2 ≤ y ∧ y ≤ 10 → total_cost tp x ≤ total_cost tp y) ∧
  total_cost tp x = 8800 :=
sorry

#check min_transportation_cost

end NUMINAMATH_CALUDE_min_transportation_cost_l107_10733


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_asymptotes_and_point_l107_10785

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 3 - y^2 / 12 = 1

/-- Theorem: Given a hyperbola with asymptotes y = ±2x and passing through (2, 2),
    its equation is x²/3 - y²/12 = 1 -/
theorem hyperbola_equation_from_asymptotes_and_point :
  ∀ (h : Hyperbola), h.asymptote_slope = 2 → h.point = (2, 2) →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_asymptotes_and_point_l107_10785


namespace NUMINAMATH_CALUDE_linear_function_shift_l107_10767

/-- Given a linear function y = mx - 1, prove that if the graph is shifted down by 2 units
    and passes through the point (-2, 1), then m = -2. -/
theorem linear_function_shift (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 3 → (x = -2 ∧ y = 1)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_shift_l107_10767


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l107_10727

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (8 * bowling_ball_weight = 4 * canoe_weight) →
    (3 * canoe_weight = 108) →
    bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l107_10727


namespace NUMINAMATH_CALUDE_mans_age_ratio_l107_10762

theorem mans_age_ratio (mans_age father_age : ℕ) : 
  father_age = 60 →
  mans_age + 12 = (father_age + 12) / 2 →
  mans_age * 5 = father_age * 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_ratio_l107_10762


namespace NUMINAMATH_CALUDE_money_distribution_l107_10759

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 686, prove that Krishan has Rs. 4046. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 686 →
  krishan = 4046 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l107_10759


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l107_10719

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

/-- Theorem: If a batsman's average increases by 2 after scoring 80 in the 17th innings,
    then the new average is 48 -/
theorem batsman_average_theorem (b : Batsman) :
  b.innings = 16 →
  newAverage b 80 = b.average + 2 →
  newAverage b 80 = 48 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l107_10719


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l107_10780

theorem flu_transmission_rate : 
  ∃ x : ℝ, 
    x > 0 ∧ 
    (1 + x) + x * (1 + x) = 100 ∧ 
    x = 9 := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l107_10780


namespace NUMINAMATH_CALUDE_problem_statement_l107_10720

theorem problem_statement (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 39 ∨ y = 6 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l107_10720


namespace NUMINAMATH_CALUDE_ratio_of_sequences_l107_10749

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sequence : ℚ := arithmetic_sum 2 2 17
def denominator_sequence : ℚ := arithmetic_sum 3 3 17

theorem ratio_of_sequences : numerator_sequence / denominator_sequence = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sequences_l107_10749


namespace NUMINAMATH_CALUDE_carnival_tickets_l107_10718

/-- The number of additional tickets needed for even distribution -/
def additional_tickets (friends : ℕ) (total_tickets : ℕ) : ℕ :=
  (friends - (total_tickets % friends)) % friends

/-- Proof that 9 friends need 8 more tickets to evenly split 865 tickets -/
theorem carnival_tickets : additional_tickets 9 865 = 8 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l107_10718


namespace NUMINAMATH_CALUDE_circle_equation_correct_l107_10714

-- Define the center and radius of the circle
def center : ℝ × ℝ := (1, -2)
def radius : ℝ := 3

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem to prove
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + (y + 2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l107_10714


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l107_10730

theorem partial_fraction_decomposition_product (A B C : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 3 → 
    (x^2 - 19) / (x^3 - 2*x^2 - 5*x + 6) = A / (x - 1) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l107_10730


namespace NUMINAMATH_CALUDE_hanyoung_weight_l107_10748

theorem hanyoung_weight (hanyoung joohyung : ℝ) 
  (h1 : hanyoung = joohyung - 4)
  (h2 : hanyoung + joohyung = 88) : 
  hanyoung = 42 := by
sorry

end NUMINAMATH_CALUDE_hanyoung_weight_l107_10748


namespace NUMINAMATH_CALUDE_even_function_property_l107_10731

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_property 
  (h1 : isEven f) 
  (h2 : ∀ x ∈ Set.Icc (-5 : ℝ) 5, ∃ y, f x = y)
  (h3 : f 3 > f 1) : 
  f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l107_10731


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l107_10734

/-- Two lines are parallel if their slopes are equal (when they exist) -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ (a₁ / b₁ = a₂ / b₂)

theorem parallel_lines_m_value :
  ∀ m : ℝ,
  parallel 3 (m + 1) (-(m - 7)) m 2 (-3 * m) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l107_10734


namespace NUMINAMATH_CALUDE_problem_statement_l107_10760

theorem problem_statement (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z^2) :
  (x - 1/x) * (z^2 + 1/z^2) = x^2 - z^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l107_10760


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l107_10736

theorem ceiling_sum_sqrt : ⌈Real.sqrt 8⌉ + ⌈Real.sqrt 48⌉ + ⌈Real.sqrt 288⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l107_10736


namespace NUMINAMATH_CALUDE_total_stripes_eq_22_l107_10761

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := olga_stripes_per_shoe * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all pairs of tennis shoes -/
def total_stripes : ℕ := 
  (olga_stripes_per_shoe * shoes_per_person) + 
  (rick_stripes_per_shoe * shoes_per_person) + 
  (hortense_stripes_per_shoe * shoes_per_person)

theorem total_stripes_eq_22 : total_stripes = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_eq_22_l107_10761


namespace NUMINAMATH_CALUDE_benny_crayons_l107_10702

theorem benny_crayons (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by
  sorry

end NUMINAMATH_CALUDE_benny_crayons_l107_10702


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l107_10700

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate for three points being collinear -/
def threeCollinear (fp : FourPoints) : Prop :=
  sorry

/-- Predicate for four points being coplanar -/
def fourCoplanar (fp : FourPoints) : Prop :=
  sorry

/-- Theorem stating that three collinear points is a sufficient but not necessary condition for four coplanar points -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ fp : FourPoints, threeCollinear fp → fourCoplanar fp) ∧
  (∃ fp : FourPoints, fourCoplanar fp ∧ ¬threeCollinear fp) :=
sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l107_10700


namespace NUMINAMATH_CALUDE_water_volume_calculation_l107_10793

/-- Given a volume of water that can be transferred into small hemisphere containers,
    this theorem proves the total volume of water. -/
theorem water_volume_calculation
  (hemisphere_volume : ℝ)
  (num_hemispheres : ℕ)
  (hemisphere_volume_is_4 : hemisphere_volume = 4)
  (num_hemispheres_is_2945 : num_hemispheres = 2945) :
  hemisphere_volume * num_hemispheres = 11780 :=
by sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l107_10793


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l107_10753

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l107_10753


namespace NUMINAMATH_CALUDE_one_positive_integer_solution_l107_10776

theorem one_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 24 - 6 * x > 12 :=
by sorry

end NUMINAMATH_CALUDE_one_positive_integer_solution_l107_10776


namespace NUMINAMATH_CALUDE_volleyball_matches_l107_10789

theorem volleyball_matches (a : ℕ) : 
  (3 / 5 : ℚ) * a = (11 / 20 : ℚ) * ((7 / 6 : ℚ) * a) → a = 24 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_matches_l107_10789


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l107_10782

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 2*43*(3*y) + 43^2))) ∧
  (53 ∣ ((3*x)^2 + 2*43*(3*x) + 43^2)) ∧
  x = 21 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l107_10782


namespace NUMINAMATH_CALUDE_hash_difference_equals_two_l107_10754

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - x - 2 * y

-- State the theorem
theorem hash_difference_equals_two : (hash 6 4) - (hash 4 6) = 2 := by sorry

end NUMINAMATH_CALUDE_hash_difference_equals_two_l107_10754


namespace NUMINAMATH_CALUDE_article_a_profit_percentage_l107_10794

/-- Profit percentage calculation for Article A -/
theorem article_a_profit_percentage 
  (x : ℝ) -- selling price of Article A
  (y : ℝ) -- selling price of Article B
  (h1 : 0.5 * x = 0.8 * (x / 1.6)) -- condition for 20% loss at half price
  (h2 : 1.05 * y = 0.9 * x) -- condition for price equality after changes
  : (0.972 * x - (x / 1.6)) / (x / 1.6) * 100 = 55.52 := by sorry

end NUMINAMATH_CALUDE_article_a_profit_percentage_l107_10794


namespace NUMINAMATH_CALUDE_product_of_recurring_decimal_and_nine_l107_10743

theorem product_of_recurring_decimal_and_nine (x : ℚ) : 
  x = 1/3 → x * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_recurring_decimal_and_nine_l107_10743


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l107_10721

theorem right_triangle_third_side_square (a b c : ℝ) : 
  a = 6 → b = 8 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c^2 = 28 ∨ c^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l107_10721


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l107_10742

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (violet : ℕ)
  (orange : ℕ)
  (lime : ℕ)
  (total : ℕ)
  (h1 : violet + orange + lime = total)
  (h2 : total = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.violet^2 + d.orange^2 + d.lime^2) / d.total^2

/-- Theorem statement for the probability problem -/
theorem dice_probability_theorem (d : ColoredDie) 
  (hv : d.violet = 3) (ho : d.orange = 4) (hl : d.lime = 5) : 
  same_color_probability d = 25 / 72 := by
  sorry

#eval same_color_probability ⟨3, 4, 5, 12, by norm_num, rfl⟩

end NUMINAMATH_CALUDE_dice_probability_theorem_l107_10742


namespace NUMINAMATH_CALUDE_tunnel_length_l107_10751

/-- Calculates the length of a tunnel given train parameters and transit time -/
theorem tunnel_length
  (train_length : Real)
  (train_speed_kmh : Real)
  (transit_time_min : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 72)
  (h3 : transit_time_min = 2.5) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let transit_time_s : Real := transit_time_min * 60
  let total_distance : Real := train_speed_ms * transit_time_s
  let tunnel_length_m : Real := total_distance - train_length
  let tunnel_length_km : Real := tunnel_length_m / 1000
  tunnel_length_km = 2.9 := by
sorry

end NUMINAMATH_CALUDE_tunnel_length_l107_10751


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l107_10707

/-- 
Given a forest with an initial number of trees, and a forester who plants trees over two days,
this theorem proves that the ratio of trees planted on the second day to the first day is 1/3,
given specific conditions about the planting process.
-/
theorem tree_planting_ratio 
  (initial_trees : ℕ) 
  (trees_after_monday : ℕ) 
  (total_planted : ℕ) 
  (h1 : initial_trees = 30)
  (h2 : trees_after_monday = initial_trees * 3)
  (h3 : total_planted = 80) :
  (total_planted - (trees_after_monday - initial_trees)) / (trees_after_monday - initial_trees) = 1 / 3 := by
  sorry

#check tree_planting_ratio

end NUMINAMATH_CALUDE_tree_planting_ratio_l107_10707


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l107_10725

theorem triangle_angle_proof (A B C : ℝ) (m n : ℝ × ℝ) :
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l107_10725


namespace NUMINAMATH_CALUDE_cross_symmetry_l107_10711

/-- Represents a square in the cross shape --/
inductive Square
| TopLeft
| TopRight
| Center
| BottomLeft
| BottomRight

/-- Represents a cross shape made of 5 squares --/
def CrossShape := Square → Square

/-- Defines the diagonal reflection operation --/
def diagonalReflection (c : CrossShape) : CrossShape :=
  fun s => match s with
  | Square.TopLeft => c Square.BottomRight
  | Square.TopRight => c Square.BottomLeft
  | Square.Center => c Square.Center
  | Square.BottomLeft => c Square.TopRight
  | Square.BottomRight => c Square.TopLeft

/-- Theorem: A cross shape is symmetric with respect to diagonal reflection
    if and only if it satisfies the specified swap conditions --/
theorem cross_symmetry (c : CrossShape) :
  (∀ s : Square, diagonalReflection c s = c s) ↔
  (c Square.TopRight = Square.BottomLeft ∧
   c Square.BottomLeft = Square.TopRight ∧
   c Square.TopLeft = Square.BottomRight ∧
   c Square.BottomRight = Square.TopLeft ∧
   c Square.Center = Square.Center) :=
by sorry


end NUMINAMATH_CALUDE_cross_symmetry_l107_10711


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l107_10732

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 2*y + 5 < 3*y - 10 → y ≥ 16) ∧ (2*16 + 5 < 3*16 - 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l107_10732


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l107_10799

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4 ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 5 → k = 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l107_10799


namespace NUMINAMATH_CALUDE_min_value_of_absolute_sum_l107_10741

theorem min_value_of_absolute_sum (x : ℚ) : 
  ∀ x : ℚ, |3 - x| + |x - 2| + |-1 + x| ≥ 2 ∧ 
  ∃ x : ℚ, |3 - x| + |x - 2| + |-1 + x| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_absolute_sum_l107_10741


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l107_10709

theorem divisible_by_twelve (n : ℤ) (h : n > 1) : ∃ k : ℤ, n^4 - n^2 = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l107_10709


namespace NUMINAMATH_CALUDE_village_population_l107_10735

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.8 = 4554 → P = 6325 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l107_10735


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l107_10706

/-- Given the equation |9 - x(3 - 12)| - |5 - 11| = 75, prove that x = 8 is the solution. -/
theorem missing_number_is_eight : ∃ x : ℝ, 
  (|9 - x * (3 - 12)| - |5 - 11| = 75) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l107_10706


namespace NUMINAMATH_CALUDE_stevens_peaches_l107_10792

theorem stevens_peaches (jake steven jill : ℕ) 
  (h1 : jake = steven - 18)
  (h2 : steven = jill + 13)
  (h3 : jill = 6) : 
  steven = 19 := by
sorry

end NUMINAMATH_CALUDE_stevens_peaches_l107_10792


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l107_10723

theorem smallest_x_absolute_value (x : ℝ) : 
  (∀ y, |y + 4| = 15 → x ≤ y) ↔ x = -19 := by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l107_10723


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l107_10712

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l107_10712


namespace NUMINAMATH_CALUDE_shoe_pairs_l107_10786

theorem shoe_pairs (ellie riley : ℕ) : 
  ellie = riley + 3 →
  ellie + riley = 13 →
  ellie = 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_pairs_l107_10786


namespace NUMINAMATH_CALUDE_smallest_x_value_l107_10704

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l107_10704


namespace NUMINAMATH_CALUDE_parent_selection_theorem_l107_10701

def total_parents : ℕ := 12
def num_couples : ℕ := 6
def parents_to_select : ℕ := 4

theorem parent_selection_theorem :
  let ways_to_select_couple := num_couples
  let remaining_parents := total_parents - 2
  let ways_to_select_others := (remaining_parents.choose (parents_to_select - 2))
  ways_to_select_couple * ways_to_select_others = 240 := by
  sorry

end NUMINAMATH_CALUDE_parent_selection_theorem_l107_10701


namespace NUMINAMATH_CALUDE_joan_change_l107_10752

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost bill_amount : ℚ) : 
  cat_toy_cost = 8.77 →
  cage_cost = 10.97 →
  bill_amount = 20 →
  bill_amount - (cat_toy_cost + cage_cost) = 0.26 := by
sorry

end NUMINAMATH_CALUDE_joan_change_l107_10752


namespace NUMINAMATH_CALUDE_triangle_properties_l107_10790

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of an acute triangle -/
theorem triangle_properties (t : Triangle) 
  (acute : t.A > 0 ∧ t.A < π ∧ t.B > 0 ∧ t.B < π ∧ t.C > 0 ∧ t.C < π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (h_m : m = (Real.sqrt 3, 2 * Real.sin t.A))
  (h_n : n = (t.c, t.a))
  (h_parallel : ∃ (k : ℝ), m.1 * n.2 = k * m.2 * n.1)
  (h_c : t.c = Real.sqrt 7)
  (h_area : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l107_10790
