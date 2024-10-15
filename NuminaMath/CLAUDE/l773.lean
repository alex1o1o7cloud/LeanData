import Mathlib

namespace NUMINAMATH_CALUDE_complex_arithmetic_problem_l773_77378

theorem complex_arithmetic_problem : 
  (Complex.mk 2 5 + Complex.mk (-1) (-3)) * Complex.mk 3 1 = Complex.mk 1 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_problem_l773_77378


namespace NUMINAMATH_CALUDE_sum_of_xy_l773_77373

theorem sum_of_xy (x y : ℕ) (hx : 0 < x ∧ x < 20) (hy : 0 < y ∧ y < 20) 
  (h_eq : x + y + x * y = 76) : x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l773_77373


namespace NUMINAMATH_CALUDE_megan_total_songs_l773_77318

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem megan_total_songs : total_songs = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l773_77318


namespace NUMINAMATH_CALUDE_scooter_initial_cost_l773_77362

/-- Proves that the initial cost of a scooter is $900 given the conditions of the problem -/
theorem scooter_initial_cost (initial_cost : ℝ) : 
  (∃ (total_cost : ℝ), 
    total_cost = initial_cost + 300 ∧ 
    1500 = 1.25 * total_cost) → 
  initial_cost = 900 :=
by sorry

end NUMINAMATH_CALUDE_scooter_initial_cost_l773_77362


namespace NUMINAMATH_CALUDE_new_average_production_l773_77314

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) 
  (h1 : n = 10)
  (h2 : past_avg = 50)
  (h3 : today_prod = 105) :
  (n * past_avg + today_prod) / (n + 1) = 55 := by
sorry

end NUMINAMATH_CALUDE_new_average_production_l773_77314


namespace NUMINAMATH_CALUDE_election_invalid_votes_percentage_l773_77379

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (b_votes : ℕ) 
  (h1 : total_votes = 8720)
  (h2 : b_votes = 2834)
  (h3 : ∃ (a_votes : ℕ), a_votes = b_votes + (15 * total_votes) / 100) :
  (total_votes - (b_votes + (b_votes + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
  sorry

end NUMINAMATH_CALUDE_election_invalid_votes_percentage_l773_77379


namespace NUMINAMATH_CALUDE_hamburger_problem_l773_77365

theorem hamburger_problem (total_spent : ℚ) (total_burgers : ℕ) 
  (single_cost : ℚ) (double_cost : ℚ) (h1 : total_spent = 68.5) 
  (h2 : total_burgers = 50) (h3 : single_cost = 1) (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_count * single_cost + double_count * double_cost = total_spent ∧
    double_count = 37 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_problem_l773_77365


namespace NUMINAMATH_CALUDE_garden_length_l773_77323

/-- Proves that a rectangular garden with length twice its width and perimeter 240 yards has a length of 80 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width → -- length is twice the width
  2 * length + 2 * width = 240 → -- perimeter is 240 yards
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l773_77323


namespace NUMINAMATH_CALUDE_lcm_problem_l773_77394

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l773_77394


namespace NUMINAMATH_CALUDE_slower_truck_speed_calculation_l773_77342

-- Define the length of each truck
def truck_length : ℝ := 250

-- Define the speed of the faster truck
def faster_truck_speed : ℝ := 30

-- Define the time taken for the slower truck to pass the faster one
def passing_time : ℝ := 35.997120230381576

-- Define the speed of the slower truck
def slower_truck_speed : ℝ := 20

-- Theorem statement
theorem slower_truck_speed_calculation :
  let total_length := 2 * truck_length
  let faster_speed_ms := faster_truck_speed * (1000 / 3600)
  let slower_speed_ms := slower_truck_speed * (1000 / 3600)
  let relative_speed := faster_speed_ms + slower_speed_ms
  total_length = relative_speed * passing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_truck_speed_calculation_l773_77342


namespace NUMINAMATH_CALUDE_alyssa_total_spending_l773_77354

/-- Calculates the total cost of Alyssa's toy shopping, including discount and tax --/
def total_cost (football_price teddy_bear_price crayons_price puzzle_price doll_price : ℚ)
  (teddy_bear_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_teddy_bear := teddy_bear_price * (1 - teddy_bear_discount)
  let subtotal := football_price + discounted_teddy_bear + crayons_price + puzzle_price + doll_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Theorem stating that Alyssa's total spending matches the calculated amount --/
theorem alyssa_total_spending :
  total_cost 12.99 15.35 4.65 7.85 14.50 0.15 0.08 = 57.23 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_total_spending_l773_77354


namespace NUMINAMATH_CALUDE_identical_solutions_condition_l773_77358

theorem identical_solutions_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k) ↔ k = -4 := by
sorry

end NUMINAMATH_CALUDE_identical_solutions_condition_l773_77358


namespace NUMINAMATH_CALUDE_susan_weather_probability_l773_77345

/-- The probability of having exactly 1 or 2 sunny days in a 3-day period -/
def prob_1_or_2_sunny (p : ℚ) : ℚ :=
  (3 : ℚ) * p * (1 - p)^2 + (3 : ℚ) * p^2 * (1 - p)

/-- The theorem stating the probability of Susan getting her desired weather -/
theorem susan_weather_probability :
  prob_1_or_2_sunny (2/5) = 18/25 := by
  sorry


end NUMINAMATH_CALUDE_susan_weather_probability_l773_77345


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l773_77346

theorem rectangular_box_surface_area
  (x y z : ℝ)
  (h1 : 4 * x + 4 * y + 4 * z = 240)
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 31) :
  2 * (x * y + y * z + z * x) = 2639 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l773_77346


namespace NUMINAMATH_CALUDE_special_function_properties_l773_77301

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧
  (∀ x y : ℝ, f (x + y) = f x + f y + f x * f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 0) ∧
  (∀ n : ℕ, f (n + 1) + 1 = 2 * (f n + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x) :=
by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l773_77301


namespace NUMINAMATH_CALUDE_total_trade_scientific_notation_l773_77321

/-- Represents the total bilateral trade in goods in yuan -/
def total_trade : ℝ := 1653 * 1000000000

/-- Represents the scientific notation of the total trade -/
def scientific_notation : ℝ := 1.6553 * (10 ^ 12)

/-- Theorem stating that the total trade is equal to its scientific notation representation -/
theorem total_trade_scientific_notation : total_trade = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_total_trade_scientific_notation_l773_77321


namespace NUMINAMATH_CALUDE_sally_remaining_cards_l773_77351

def initial_cards : ℕ := 39
def cards_sold : ℕ := 24

theorem sally_remaining_cards : initial_cards - cards_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_sally_remaining_cards_l773_77351


namespace NUMINAMATH_CALUDE_tangent_line_reciprocal_function_l773_77326

/-- The equation of the tangent line to y = 1/x at (1,1) is x + y - 2 = 0 -/
theorem tangent_line_reciprocal_function (x y : ℝ) : 
  (∀ t, t ≠ 0 → y = 1 / t) →  -- Condition: the curve is y = 1/x
  (x = 1 ∧ y = 1) →           -- Condition: the point of tangency is (1,1)
  x + y - 2 = 0               -- Conclusion: equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_reciprocal_function_l773_77326


namespace NUMINAMATH_CALUDE_pattern_paths_count_l773_77364

/-- Represents a position in the diagram -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents a letter in the diagram -/
inductive Letter
  | P | A | T | E | R | N | C | O

/-- The diagram of letters -/
def diagram : List (List Letter) := sorry

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Checks if a path spells "PATTERN" -/
def spells_pattern (path : List Position) : Prop := sorry

/-- Counts the number of valid paths spelling "PATTERN" -/
def count_pattern_paths : ℕ := sorry

/-- The main theorem to prove -/
theorem pattern_paths_count :
  count_pattern_paths = 18 := by sorry

end NUMINAMATH_CALUDE_pattern_paths_count_l773_77364


namespace NUMINAMATH_CALUDE_tessa_final_debt_l773_77366

/-- Calculates the final debt after a partial repayment and additional borrowing --/
def calculateFinalDebt (initialDebt : ℚ) (repaymentFraction : ℚ) (additionalBorrowing : ℚ) : ℚ :=
  initialDebt - (repaymentFraction * initialDebt) + additionalBorrowing

/-- Proves that Tessa's final debt to Greg is $30 --/
theorem tessa_final_debt :
  calculateFinalDebt 40 (1/2) 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tessa_final_debt_l773_77366


namespace NUMINAMATH_CALUDE_area_two_quarter_circles_l773_77386

/-- The area of a figure formed by two 90° sectors of a circle with radius 10 -/
theorem area_two_quarter_circles (r : ℝ) (h : r = 10) : 
  2 * (π * r^2 / 4) = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_area_two_quarter_circles_l773_77386


namespace NUMINAMATH_CALUDE_measles_cases_1990_l773_77390

/-- Calculates the number of measles cases in a given year, assuming a linear decrease from 1970 to 2000 -/
def measlesCases (year : ℕ) : ℕ :=
  let initialYear : ℕ := 1970
  let finalYear : ℕ := 2000
  let initialCases : ℕ := 480000
  let finalCases : ℕ := 600
  let yearsPassed : ℕ := year - initialYear
  let totalYears : ℕ := finalYear - initialYear
  let totalDecrease : ℕ := initialCases - finalCases
  let yearlyDecrease : ℕ := totalDecrease / totalYears
  initialCases - (yearsPassed * yearlyDecrease)

theorem measles_cases_1990 : measlesCases 1990 = 160400 := by
  sorry

end NUMINAMATH_CALUDE_measles_cases_1990_l773_77390


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l773_77381

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → -4 ≤ m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l773_77381


namespace NUMINAMATH_CALUDE_parabola_vertex_l773_77375

/-- The parabola defined by y = (x-1)^2 + 3 has its vertex at (1,3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 3 → (∃ (a : ℝ), y = a * (x - 1)^2 + 3 ∧ a ≠ 0) → 
  (1, 3) = (x, y) ∧ (∀ (x' y' : ℝ), y' = (x' - 1)^2 + 3 → y' ≥ y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l773_77375


namespace NUMINAMATH_CALUDE_intersection_of_lines_l773_77370

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Check if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem intersection_of_lines (line1 line2 : Line)
  (h1 : line1 = ⟨6, -9, 18⟩)
  (h2 : line2 = ⟨8, 2, 20⟩) :
  ∃! p : IntersectionPoint, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = ⟨18/7, -2/7⟩ := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l773_77370


namespace NUMINAMATH_CALUDE_system_solution_l773_77391

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 7 * x + y = 19
def equation2 (x y : ℝ) : Prop := x + 3 * y = 1
def equation3 (x y z : ℝ) : Prop := 2 * x + y - 4 * z = 10

-- Theorem statement
theorem system_solution (x y z : ℝ) :
  equation1 x y ∧ equation2 x y ∧ equation3 x y z →
  2 * x + y + 3 * z = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l773_77391


namespace NUMINAMATH_CALUDE_magnitude_product_complex_l773_77332

theorem magnitude_product_complex : Complex.abs ((7 - 4 * Complex.I) * (3 + 10 * Complex.I)) = Real.sqrt 7085 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_product_complex_l773_77332


namespace NUMINAMATH_CALUDE_arbitrarily_large_solution_exists_l773_77371

theorem arbitrarily_large_solution_exists (N : ℕ) : 
  ∃ (a b c d : ℤ), 
    (a * a + b * b + c * c + d * d = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    (min a (min b (min c d)) ≥ N) := by
  sorry

end NUMINAMATH_CALUDE_arbitrarily_large_solution_exists_l773_77371


namespace NUMINAMATH_CALUDE_mark_change_factor_l773_77343

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℚ) (h1 : n = 10) (h2 : original_avg = 80) (h3 : new_avg = 160) :
  ∃ (factor : ℚ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l773_77343


namespace NUMINAMATH_CALUDE_caterer_order_l773_77380

/-- The number of ice-cream bars ordered by a caterer -/
def num_ice_cream_bars : ℕ := 225

/-- The total price of the order in cents -/
def total_price : ℕ := 20000

/-- The price of each ice-cream bar in cents -/
def price_ice_cream_bar : ℕ := 60

/-- The price of each sundae in cents -/
def price_sundae : ℕ := 52

/-- The number of sundaes ordered -/
def num_sundaes : ℕ := 125

theorem caterer_order :
  num_ice_cream_bars * price_ice_cream_bar + num_sundaes * price_sundae = total_price :=
by sorry

end NUMINAMATH_CALUDE_caterer_order_l773_77380


namespace NUMINAMATH_CALUDE_arrangement_theorem_l773_77348

/-- The number of ways to arrange n elements with k special positions --/
def special_arrangements (n : ℕ) (k : ℕ) : ℕ := k * (n - 1).factorial

/-- The number of ways to arrange n elements with two restrictions --/
def restricted_arrangements (n : ℕ) : ℕ := 
  n.factorial - 2 * (n - 1).factorial + (n - 2).factorial

theorem arrangement_theorem (total_students : ℕ) 
  (h_total : total_students = 7) :
  (special_arrangements total_students 3 = 2160) ∧
  (restricted_arrangements total_students = 3720) := by
  sorry

#eval special_arrangements 7 3
#eval restricted_arrangements 7

end NUMINAMATH_CALUDE_arrangement_theorem_l773_77348


namespace NUMINAMATH_CALUDE_tennis_balls_cost_l773_77330

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := 4

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each tennis ball in dollars -/
def cost_per_ball : ℕ := 2

/-- The total cost of the tennis balls -/
def total_cost : ℕ := num_packs * balls_per_pack * cost_per_ball

theorem tennis_balls_cost : total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_cost_l773_77330


namespace NUMINAMATH_CALUDE_x_value_proof_l773_77347

theorem x_value_proof (x : ℚ) (h : (1/2 : ℚ) - (1/4 : ℚ) + (1/8 : ℚ) = 8/x) : x = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l773_77347


namespace NUMINAMATH_CALUDE_linear_function_property_l773_77350

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the inverse function
def InverseFunction (f g : ℝ → ℝ) : Prop := ∀ x, f (g x) = x ∧ g (f x) = x

theorem linear_function_property (f : ℝ → ℝ) 
  (h1 : LinearFunction f) 
  (h2 : ∃ g : ℝ → ℝ, InverseFunction f g ∧ ∀ x, f x = 5 * g x + 8) 
  (h3 : f 1 = 5) : 
  f 3 = 2 * Real.sqrt 5 + 5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l773_77350


namespace NUMINAMATH_CALUDE_arithmetic_operations_l773_77337

theorem arithmetic_operations : 
  (400 / 5 = 80) ∧ (3 * 230 = 690) := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l773_77337


namespace NUMINAMATH_CALUDE_add_negative_two_l773_77324

theorem add_negative_two : 3 + (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_two_l773_77324


namespace NUMINAMATH_CALUDE_additional_bottles_l773_77308

theorem additional_bottles (initial_bottles : ℕ) (capacity_per_bottle : ℕ) (total_stars : ℕ) : 
  initial_bottles = 2 → capacity_per_bottle = 15 → total_stars = 75 →
  (total_stars - initial_bottles * capacity_per_bottle) / capacity_per_bottle = 3 := by
sorry

end NUMINAMATH_CALUDE_additional_bottles_l773_77308


namespace NUMINAMATH_CALUDE_jills_income_ratio_l773_77306

/-- Proves that the ratio of Jill's discretionary income to her net monthly salary is 1/5 -/
theorem jills_income_ratio :
  let net_salary : ℚ := 3500
  let discretionary_income : ℚ := 105 / (15/100)
  discretionary_income / net_salary = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jills_income_ratio_l773_77306


namespace NUMINAMATH_CALUDE_range_of_a_l773_77356

-- Define the propositions p and q as functions of x and a
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that not p is necessary but not sufficient for not q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x a)

-- State the theorem
theorem range_of_a :
  {a : ℝ | necessary_not_sufficient a} = {a : ℝ | a < 0 ∨ a > 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l773_77356


namespace NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_two_l773_77357

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ -2) :
  ((3 * x + 4) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = (x - 1) / (x + 1) :=
by sorry

theorem fraction_value_at_two :
  ((3 * 2 + 4) / (2^2 - 1) - 2 / (2 - 1)) / ((2 + 2) / (2^2 - 2*2 + 1)) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_two_l773_77357


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l773_77361

theorem nested_fraction_equality : 
  (1 : ℝ) / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l773_77361


namespace NUMINAMATH_CALUDE_gasoline_added_l773_77387

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 29.999999999999996 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 4.499999999999999 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_added_l773_77387


namespace NUMINAMATH_CALUDE_triangle_ambiguous_case_l773_77320

theorem triangle_ambiguous_case (a b : ℝ) (A : ℝ) : 
  a = 12 → A = π / 3 → (b * Real.sin A < a ∧ a < b) ↔ (12 < b ∧ b < 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ambiguous_case_l773_77320


namespace NUMINAMATH_CALUDE_nim_max_product_l773_77316

/-- Nim-sum (bitwise XOR) of two natural numbers -/
def nim_sum (a b : ℕ) : ℕ := a ^^^ b

/-- Check if a given configuration is a losing position in 3-player Nim -/
def is_losing_position (a b c d : ℕ) : Prop :=
  nim_sum (nim_sum (nim_sum a b) c) d = 0

/-- The maximum product of x and y satisfying the game conditions -/
def max_product : ℕ := 7704

/-- The theorem stating the maximum product of x and y in the given Nim game -/
theorem nim_max_product :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  is_losing_position 43 99 x y ∧
  x * y = max_product ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → is_losing_position 43 99 a b → a * b ≤ max_product :=
sorry

end NUMINAMATH_CALUDE_nim_max_product_l773_77316


namespace NUMINAMATH_CALUDE_max_n_is_81_l773_77360

/-- The maximum value of n given the conditions -/
def max_n : ℕ := 81

/-- The set of numbers from 1 to 500 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

/-- The probability of selecting a divisor of n from S -/
def prob_divisor (n : ℕ) : ℚ := (Finset.filter (· ∣ n) (Finset.range 500)).card / 500

/-- The theorem stating that 81 is the maximum value satisfying the conditions -/
theorem max_n_is_81 :
  ∀ n : ℕ, n ∈ S → prob_divisor n = 1/100 → n ≤ max_n :=
sorry

end NUMINAMATH_CALUDE_max_n_is_81_l773_77360


namespace NUMINAMATH_CALUDE_book_discount_percentage_l773_77313

theorem book_discount_percentage (marked_price : ℝ) (cost_price : ℝ) (selling_price : ℝ) :
  cost_price = 0.64 * marked_price →
  (selling_price - cost_price) / cost_price = 0.375 →
  (marked_price - selling_price) / marked_price = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_percentage_l773_77313


namespace NUMINAMATH_CALUDE_ice_cream_arrangements_l773_77319

theorem ice_cream_arrangements : (Nat.factorial 5) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangements_l773_77319


namespace NUMINAMATH_CALUDE_shopping_change_calculation_l773_77310

def book_price : ℝ := 25
def pen_price : ℝ := 4
def ruler_price : ℝ := 1
def notebook_price : ℝ := 8
def pencil_case_price : ℝ := 6
def book_discount : ℝ := 0.1
def pen_discount : ℝ := 0.05
def sales_tax_rate : ℝ := 0.06
def payment : ℝ := 100

theorem shopping_change_calculation :
  let discounted_book_price := book_price * (1 - book_discount)
  let discounted_pen_price := pen_price * (1 - pen_discount)
  let total_before_tax := discounted_book_price + discounted_pen_price + ruler_price + notebook_price + pencil_case_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_with_tax := total_before_tax + tax_amount
  let change := payment - total_with_tax
  change = 56.22 := by sorry

end NUMINAMATH_CALUDE_shopping_change_calculation_l773_77310


namespace NUMINAMATH_CALUDE_range_of_x_l773_77374

-- Define the set of real numbers that satisfy the given condition
def S : Set ℝ := {x | ¬(x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)}

-- Theorem stating that S is equal to the interval [1,2)
theorem range_of_x : S = Set.Ico 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l773_77374


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l773_77359

def car_cost : ℕ := 6500
def net_profit_per_pizza : ℕ := 7

theorem min_pizzas_to_break_even :
  ∀ n : ℕ, (n * net_profit_per_pizza ≥ car_cost) ∧ 
           (∀ m : ℕ, m < n → m * net_profit_per_pizza < car_cost) →
  n = 929 := by
  sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l773_77359


namespace NUMINAMATH_CALUDE_third_month_sale_calculation_l773_77302

/-- Calculates the sale in the third month given the sales of other months and the average -/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (first_month + second_month + fourth_month)

/-- Theorem stating the sale in the third month given the problem conditions -/
theorem third_month_sale_calculation :
  third_month_sale 2500 4000 1520 2890 = 3540 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_calculation_l773_77302


namespace NUMINAMATH_CALUDE_combined_figure_area_l773_77338

/-- Regular pentagon with side length 3 -/
structure RegularPentagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_square : side_length = 3)

/-- Combined figure of a regular pentagon and a square -/
structure CombinedFigure :=
  (pentagon : RegularPentagon)
  (square : Square)
  (shared_side : pentagon.side_length = square.side_length)

/-- Area of the combined figure -/
def area (figure : CombinedFigure) : ℝ := sorry

/-- Theorem stating the area of the combined figure -/
theorem combined_figure_area (figure : CombinedFigure) :
  area figure = Real.sqrt 81 + Real.sqrt 27 := by sorry

end NUMINAMATH_CALUDE_combined_figure_area_l773_77338


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_18_minus_1_l773_77363

theorem divisors_of_2_pow_18_minus_1 :
  ∃! (a b : ℕ), 20 < a ∧ a < 30 ∧ 20 < b ∧ b < 30 ∧
  (2^18 - 1) % a = 0 ∧ (2^18 - 1) % b = 0 ∧ a ≠ b ∧
  a = 19 ∧ b = 27 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_18_minus_1_l773_77363


namespace NUMINAMATH_CALUDE_tree_height_difference_l773_77336

theorem tree_height_difference :
  let apple_tree_height : ℚ := 53 / 4
  let cherry_tree_height : ℚ := 147 / 8
  cherry_tree_height - apple_tree_height = 41 / 8 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l773_77336


namespace NUMINAMATH_CALUDE_babblian_word_count_l773_77397

def alphabet_size : ℕ := 6
def max_word_length : ℕ := 3

def count_words (alphabet_size : ℕ) (max_word_length : ℕ) : ℕ :=
  (alphabet_size^1 + alphabet_size^2 + alphabet_size^3)

theorem babblian_word_count :
  count_words alphabet_size max_word_length = 258 := by
  sorry

end NUMINAMATH_CALUDE_babblian_word_count_l773_77397


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l773_77376

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l773_77376


namespace NUMINAMATH_CALUDE_equation_solution_l773_77331

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (3 / x + (4 / x) / (8 / x) = 1.5) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l773_77331


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l773_77389

/-- Given two polynomials (7x^2 + 5) and (3x^3 + 2x + 1), their product is equal to 21x^5 + 29x^3 + 7x^2 + 10x + 5 -/
theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5) * (3 * x^3 + 2 * x + 1) = 21 * x^5 + 29 * x^3 + 7 * x^2 + 10 * x + 5 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_product_expansion_l773_77389


namespace NUMINAMATH_CALUDE_rebeccas_salon_l773_77317

/-- Rebecca's hair salon problem -/
theorem rebeccas_salon (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_perms num_dye_jobs : ℕ) (tips total_revenue : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  total_revenue = 310 →
  ∃ (num_haircuts : ℕ),
    num_haircuts * haircut_price +
    num_perms * perm_price +
    num_dye_jobs * dye_job_price -
    num_dye_jobs * dye_cost +
    tips = total_revenue ∧
    num_haircuts = 4 :=
by sorry

end NUMINAMATH_CALUDE_rebeccas_salon_l773_77317


namespace NUMINAMATH_CALUDE_fraction_comparison_l773_77327

theorem fraction_comparison (x y : ℕ+) (h : y > x) : (x + 1 : ℚ) / (y + 1) > (x : ℚ) / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l773_77327


namespace NUMINAMATH_CALUDE_correct_statement_l773_77382

def p : Prop := 2017 % 2 = 1
def q : Prop := 2016 % 2 = 0

theorem correct_statement : p ∨ q := by sorry

end NUMINAMATH_CALUDE_correct_statement_l773_77382


namespace NUMINAMATH_CALUDE_dara_waiting_time_l773_77368

/-- Represents a person's age and employment status -/
structure Person where
  age : ℕ
  employed : Bool

/-- The minimum age required for employment -/
def min_employment_age : ℕ := 25

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Jane's current state -/
def jane : Person := { age := 28, employed := true }

/-- Dara's current age -/
def dara_age : ℕ := jane.age + 6 - 2 * (jane.age + 6 - min_employment_age)

/-- Time Dara needs to wait to reach the minimum employment age -/
def waiting_time : ℕ := min_employment_age - dara_age

theorem dara_waiting_time :
  waiting_time = 14 := by sorry

end NUMINAMATH_CALUDE_dara_waiting_time_l773_77368


namespace NUMINAMATH_CALUDE_stone_123_is_3_l773_77311

/-- The number of stones in the sequence -/
def num_stones : ℕ := 12

/-- The length of the counting pattern before it repeats -/
def pattern_length : ℕ := 22

/-- The target count we're interested in -/
def target_count : ℕ := 123

/-- The original stone number we claim is counted as the target_count -/
def claimed_stone : ℕ := 3

/-- Function to determine which stone is counted as a given number -/
def stone_at_count (count : ℕ) : ℕ :=
  (count - 1) % pattern_length + 1

theorem stone_123_is_3 : 
  stone_at_count target_count = claimed_stone := by
  sorry

end NUMINAMATH_CALUDE_stone_123_is_3_l773_77311


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l773_77355

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l773_77355


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l773_77383

theorem quadratic_solution_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 3 * a * x + b
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_average_l773_77383


namespace NUMINAMATH_CALUDE_blanket_thickness_after_four_foldings_l773_77369

/-- Represents the thickness of a blanket after a certain number of foldings -/
def blanketThickness (initialThickness : ℕ) (numFoldings : ℕ) : ℕ :=
  initialThickness * 2^numFoldings

/-- Proves that a blanket with initial thickness 3 inches will be 48 inches thick after 4 foldings -/
theorem blanket_thickness_after_four_foldings :
  blanketThickness 3 4 = 48 := by
  sorry

#eval blanketThickness 3 4

end NUMINAMATH_CALUDE_blanket_thickness_after_four_foldings_l773_77369


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l773_77367

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem necessary_not_sufficient_condition :
  (∀ a : ℕ → ℝ, IsIncreasing a → ∀ n, |a (n + 1)| > a n) ∧
  (∃ a : ℕ → ℝ, (∀ n, |a (n + 1)| > a n) ∧ ¬IsIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l773_77367


namespace NUMINAMATH_CALUDE_radical_simplification_l773_77329

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^2) * Real.sqrt (8 * p) * Real.sqrt (27 * p^5) = 18 * p^4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l773_77329


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l773_77315

/-- Represents a tetrahedron with a triangular base and square lateral faces -/
structure Tetrahedron where
  base_side_length : ℝ
  has_square_lateral_faces : Bool

/-- Represents a tetrahedron inscribed within another tetrahedron -/
structure InscribedTetrahedron where
  outer : Tetrahedron
  vertices_touch_midpoints : Bool
  base_parallel : Bool

/-- Calculates the volume of an inscribed tetrahedron -/
def volume_inscribed_tetrahedron (t : InscribedTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the inscribed tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.outer.base_side_length = 2) 
  (h2 : t.outer.has_square_lateral_faces = true)
  (h3 : t.vertices_touch_midpoints = true)
  (h4 : t.base_parallel = true) : 
  volume_inscribed_tetrahedron t = Real.sqrt 2 / 12 := by sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l773_77315


namespace NUMINAMATH_CALUDE_inequality_proof_l773_77307

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4) 
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l773_77307


namespace NUMINAMATH_CALUDE_married_men_fraction_l773_77334

theorem married_men_fraction (total_women : ℕ) (h_pos : 0 < total_women) :
  let single_women := (3 * total_women : ℕ) / 5
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (married_men : ℚ) / total_people = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l773_77334


namespace NUMINAMATH_CALUDE_quadratic_solutions_parabola_vertex_l773_77339

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 4*x - 2 = 0

theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6 ∧
  quadratic_equation x1 ∧ quadratic_equation x2 :=
sorry

-- Part 2: Parabola vertex
def parabola (x y : ℝ) : Prop :=
  y = 2*x^2 - 4*x + 6

theorem parabola_vertex :
  ∃ x y : ℝ, x = 1 ∧ y = 4 ∧ parabola x y ∧
  ∀ x' y' : ℝ, parabola x' y' → y ≤ y' :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_parabola_vertex_l773_77339


namespace NUMINAMATH_CALUDE_hyperbola_intersection_l773_77372

/-- Given a triangle AOB with A on the positive y-axis, B on the positive x-axis, and area 9,
    and a hyperbolic function y = k/x intersecting AB at C and D such that CD = 1/3 AB and AC = BD,
    prove that k = 4 -/
theorem hyperbola_intersection (y_A x_B : ℝ) (k : ℝ) : 
  y_A > 0 → x_B > 0 → -- A and B are on positive axes
  1/2 * x_B * y_A = 9 → -- Area of triangle AOB is 9
  ∃ (x_C y_C : ℝ), -- C exists on the line AB and the hyperbola
    0 < x_C ∧ x_C < x_B ∧
    y_C = (y_A / x_B) * (x_B - x_C) ∧ -- C is on line AB
    y_C = k / x_C ∧ -- C is on the hyperbola
    x_C = 1/3 * x_B ∧ -- C is a trisection point
    y_C = 2/3 * y_A → -- C is a trisection point
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_l773_77372


namespace NUMINAMATH_CALUDE_circle_center_transformation_l773_77352

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem stating the transformation of the circle's center -/
theorem circle_center_transformation :
  let original_center : ℝ × ℝ := (-3, 4)
  let reflected_center := reflect_x original_center
  let final_center := translate_right reflected_center 5
  final_center = (2, -4) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l773_77352


namespace NUMINAMATH_CALUDE_wall_length_height_ratio_l773_77344

/-- Represents the dimensions and volume of a rectangular wall. -/
structure Wall where
  breadth : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem stating the ratio of length to height for a specific wall. -/
theorem wall_length_height_ratio (w : Wall) 
  (h_volume : w.volume = 12.8)
  (h_breadth : w.breadth = 0.4)
  (h_height : w.height = 5 * w.breadth)
  (h_volume_calc : w.volume = w.breadth * w.height * w.length) :
  w.length / w.height = 4 := by
  sorry

#check wall_length_height_ratio

end NUMINAMATH_CALUDE_wall_length_height_ratio_l773_77344


namespace NUMINAMATH_CALUDE_fraction_subtraction_l773_77385

theorem fraction_subtraction : (7 : ℚ) / 3 - 5 / 6 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l773_77385


namespace NUMINAMATH_CALUDE_sum_of_fractions_sum_equals_14_1_l773_77395

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  (141 : ℚ) / 10 := by
  sorry

theorem sum_equals_14_1 : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  14.1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_sum_equals_14_1_l773_77395


namespace NUMINAMATH_CALUDE_max_probability_at_20_red_balls_l773_77328

/-- The probability of winning in one draw -/
def p (n : ℕ) : ℚ := 10 * n / ((n + 5) * (n + 4))

/-- The probability of winning exactly once in three draws -/
def P (n : ℕ) : ℚ := 3 * p n * (1 - p n)^2

theorem max_probability_at_20_red_balls (n : ℕ) (h : n ≥ 5) :
  P n ≤ P 20 ∧ ∃ (m : ℕ), m ≥ 5 ∧ P m = P 20 → m = 20 :=
sorry

end NUMINAMATH_CALUDE_max_probability_at_20_red_balls_l773_77328


namespace NUMINAMATH_CALUDE_expression_evaluation_l773_77388

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 8)
  (h2 : b = a + 3)
  (h3 : a = 2)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 5 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 1) / (b - 3)) * ((c + 7) / (c + 5)) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l773_77388


namespace NUMINAMATH_CALUDE_work_days_calculation_l773_77349

/-- Represents the number of days worked by each person -/
structure DaysWorked where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The problem statement -/
theorem work_days_calculation (days : DaysWorked) (wages : DailyWages) 
    (h1 : days.a = 6)
    (h2 : days.c = 4)
    (h3 : wages.a * 5 = wages.c * 3)
    (h4 : wages.b * 5 = wages.c * 4)
    (h5 : wages.c = 100)
    (h6 : days.a * wages.a + days.b * wages.b + days.c * wages.c = 1480) :
  days.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_days_calculation_l773_77349


namespace NUMINAMATH_CALUDE_arrangement_theorem_l773_77333

/-- Represents the number of ways to arrange people in two rows -/
def arrangement_count (total_people : ℕ) (front_row : ℕ) (back_row : ℕ) : ℕ := sorry

/-- Represents whether two people are standing next to each other -/
def standing_next_to (person1 : ℕ) (person2 : ℕ) : Prop := sorry

/-- Represents whether two people are standing apart -/
def standing_apart (person1 : ℕ) (person2 : ℕ) : Prop := sorry

theorem arrangement_theorem :
  ∀ (total_people front_row back_row : ℕ) 
    (person_a person_b person_c : ℕ),
  total_people = 7 →
  front_row = 3 →
  back_row = 4 →
  standing_next_to person_a person_b →
  standing_apart person_a person_c →
  arrangement_count total_people front_row back_row = 1056 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l773_77333


namespace NUMINAMATH_CALUDE_kids_staying_home_l773_77322

theorem kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) 
  (h1 : total_kids = 898051)
  (h2 : kids_at_camp = 629424) :
  total_kids - kids_at_camp = 268627 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l773_77322


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l773_77312

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, x) (-2, 3) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l773_77312


namespace NUMINAMATH_CALUDE_simplify_radical_product_l773_77396

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l773_77396


namespace NUMINAMATH_CALUDE_equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l773_77384

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_all_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define special quadrilaterals
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_diagonals_bisect_implies_rectangle (q : Quadrilateral) :
  has_equal_diagonals q ∧ diagonals_bisect_each_other q → is_rectangle q :=
sorry

-- Theorem 2
theorem all_sides_equal_implies_rhombus (q : Quadrilateral) :
  has_all_sides_equal q → is_rhombus q :=
sorry

-- Theorem 3
theorem perpendicular_diagonals_not_imply_rhombus :
  ∃ q : Quadrilateral, has_perpendicular_diagonals q ∧ ¬is_rhombus q :=
sorry

-- Theorem 4
theorem all_sides_equal_not_imply_square :
  ∃ q : Quadrilateral, has_all_sides_equal q ∧ ¬is_square q :=
sorry

end NUMINAMATH_CALUDE_equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l773_77384


namespace NUMINAMATH_CALUDE_figure_y_value_l773_77399

/-- Given a figure with a right triangle and two squares, prove the value of y -/
theorem figure_y_value (y : ℝ) (total_area : ℝ) : 
  total_area = 980 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = 70/9 := by
sorry

end NUMINAMATH_CALUDE_figure_y_value_l773_77399


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l773_77304

def factor_tree (X Y Z Q R : ℕ) : Prop :=
  Y = 2 * Q ∧
  Z = 7 * R ∧
  Q = 5 * 3 ∧
  R = 11 * 2 ∧
  X = Y * Z

theorem factor_tree_X_value :
  ∀ X Y Z Q R : ℕ, factor_tree X Y Z Q R → X = 4620 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l773_77304


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l773_77393

theorem sum_of_squares_theorem (x y z a b c k : ℝ) 
  (h1 : x * y = k * a) 
  (h2 : x * z = k * b) 
  (h3 : y * z = k * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hk : k ≠ 0) : 
  x^2 + y^2 + z^2 = k * (a * b / c + a * c / b + b * c / a) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l773_77393


namespace NUMINAMATH_CALUDE_elvis_matchsticks_l773_77353

theorem elvis_matchsticks (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (ralph_per_square : ℕ) (leftover : ℕ) :
  total = 50 →
  elvis_squares = 5 →
  ralph_squares = 3 →
  ralph_per_square = 8 →
  leftover = 6 →
  ∃ (elvis_per_square : ℕ), 
    elvis_per_square * elvis_squares + ralph_per_square * ralph_squares + leftover = total ∧
    elvis_per_square = 4 :=
by sorry

end NUMINAMATH_CALUDE_elvis_matchsticks_l773_77353


namespace NUMINAMATH_CALUDE_coin_flip_frequency_l773_77335

/-- The frequency of an event is the ratio of the number of times the event occurs to the total number of trials. -/
def frequency (occurrences : ℕ) (trials : ℕ) : ℚ :=
  occurrences / trials

/-- In an experiment of flipping a coin 100 times, the frequency of getting "heads" is 49. -/
theorem coin_flip_frequency :
  frequency 49 100 = 49/100 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_frequency_l773_77335


namespace NUMINAMATH_CALUDE_episodes_watched_per_day_l773_77303

theorem episodes_watched_per_day 
  (total_episodes : ℕ) 
  (total_days : ℕ) 
  (h1 : total_episodes = 50) 
  (h2 : total_days = 10) 
  (h3 : total_episodes > 0) 
  (h4 : total_days > 0) : 
  (total_episodes : ℚ) / total_days = 1 / 10 := by
  sorry

#check episodes_watched_per_day

end NUMINAMATH_CALUDE_episodes_watched_per_day_l773_77303


namespace NUMINAMATH_CALUDE_percent_of_number_l773_77340

theorem percent_of_number (N M : ℝ) (h : M ≠ 0) : (N / M) * 100 = (100 * N) / M := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l773_77340


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l773_77300

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, -3)
  let p₂ : ℝ × ℝ := (-4, 7)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l773_77300


namespace NUMINAMATH_CALUDE_total_distance_is_250_l773_77398

/-- Represents a cyclist's journey with specific conditions -/
structure CyclistJourney where
  speed : ℝ
  time_store_to_friend : ℝ
  distance_store_to_friend : ℝ
  h_speed_positive : speed > 0
  h_time_positive : time_store_to_friend > 0
  h_distance_positive : distance_store_to_friend > 0
  h_distance_store_to_friend : distance_store_to_friend = 50
  h_time_relation : 2 * time_store_to_friend = speed * distance_store_to_friend

/-- The total distance cycled in the journey -/
def total_distance (j : CyclistJourney) : ℝ :=
  3 * j.distance_store_to_friend + j.distance_store_to_friend

/-- Theorem stating that the total distance cycled is 250 miles -/
theorem total_distance_is_250 (j : CyclistJourney) : total_distance j = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_250_l773_77398


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l773_77325

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Proves that a 90m by 60m plot with poles 5m apart needs 60 poles -/
theorem rectangular_plot_poles : fence_poles 90 60 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l773_77325


namespace NUMINAMATH_CALUDE_unique_solution_equation_l773_77377

theorem unique_solution_equation :
  ∃! x : ℝ, 2017 * x^2017 - 2017 + x = (2018 - 2017*x)^(1/2017) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l773_77377


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l773_77309

/-- Given a hyperbola and an ellipse that share the same foci, 
    prove that the parameter m in the hyperbola equation is 7. -/
theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 8 ∧ c^2 = m + 1) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l773_77309


namespace NUMINAMATH_CALUDE_probability_20th_to_30th_l773_77392

/-- A sequence of 40 distinct real numbers -/
def Sequence := Fin 40 → ℝ

/-- Predicate to check if a sequence contains distinct elements -/
def IsDistinct (s : Sequence) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The probability that the 20th number ends up in the 30th position after one bubble pass -/
def ProbabilityOf20thTo30th (s : Sequence) : ℚ :=
  1 / 930

/-- Theorem stating the probability of the 20th number ending up in the 30th position -/
theorem probability_20th_to_30th (s : Sequence) (h : IsDistinct s) :
    ProbabilityOf20thTo30th s = 1 / 930 := by
  sorry

end NUMINAMATH_CALUDE_probability_20th_to_30th_l773_77392


namespace NUMINAMATH_CALUDE_biscuit_count_l773_77305

-- Define the dimensions of the dough sheet
def dough_side : ℕ := 12

-- Define the dimensions of each biscuit
def biscuit_side : ℕ := 3

-- Theorem to prove
theorem biscuit_count : (dough_side * dough_side) / (biscuit_side * biscuit_side) = 16 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_count_l773_77305


namespace NUMINAMATH_CALUDE_min_value_quadratic_l773_77341

theorem min_value_quadratic (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - a * x + (a + 1000) ≥ 1 + 999 / a) ∧
  (∃ x : ℝ, a * x^2 - a * x + (a + 1000) = 1 + 999 / a) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l773_77341
