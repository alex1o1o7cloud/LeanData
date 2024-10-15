import Mathlib

namespace NUMINAMATH_CALUDE_actual_number_is_two_l1901_190158

-- Define the set of people
inductive Person
| Natasha
| Boy1
| Boy2
| Girl1
| Girl2

-- Define a function to represent claims about the number
def claim (p : Person) (n : Nat) : Prop :=
  match p with
  | Person.Natasha => n % 15 = 0
  | _ => true  -- We don't have specific information about other claims

-- Define the conditions of the problem
axiom one_boy_correct : ∃ (b : Person), b = Person.Boy1 ∨ b = Person.Boy2
axiom one_girl_correct : ∃ (g : Person), g = Person.Girl1 ∨ g = Person.Girl2
axiom two_wrong : ∃ (p1 p2 : Person), p1 ≠ p2 ∧ ¬(claim p1 2) ∧ ¬(claim p2 2)

-- The theorem to prove
theorem actual_number_is_two : 
  ∃ (n : Nat), (claim Person.Natasha n = false) ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_actual_number_is_two_l1901_190158


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1901_190151

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The asymptote equation -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The positive slope of the asymptotes of the given hyperbola is 3/2 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), hyperbola x y → (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → asymptote m (x + δ) (y + δ))) ∧
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1901_190151


namespace NUMINAMATH_CALUDE_sin_45_75_plus_sin_45_15_l1901_190143

theorem sin_45_75_plus_sin_45_15 :
  Real.sin (45 * π / 180) * Real.sin (75 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_75_plus_sin_45_15_l1901_190143


namespace NUMINAMATH_CALUDE_annie_total_blocks_l1901_190156

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ :=
  let house_to_bus := 5
  let bus_to_train := 7
  let train_to_friend := 10
  let friend_to_coffee := 4
  2 * (house_to_bus + bus_to_train + train_to_friend) + 2 * friend_to_coffee

/-- Theorem stating that Annie traveled 52 blocks in total -/
theorem annie_total_blocks : total_blocks = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_total_blocks_l1901_190156


namespace NUMINAMATH_CALUDE_ajax_final_weight_l1901_190103

/-- Calculates the final weight in pounds after a weight loss program -/
def final_weight (initial_weight_kg : ℝ) (weight_loss_per_hour : ℝ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  let kg_to_pounds : ℝ := 2.2
  let initial_weight_pounds : ℝ := initial_weight_kg * kg_to_pounds
  let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
  initial_weight_pounds - total_weight_loss

/-- Theorem: Ajax's weight after the exercise program -/
theorem ajax_final_weight :
  final_weight 80 1.5 2 14 = 134 := by
sorry


end NUMINAMATH_CALUDE_ajax_final_weight_l1901_190103


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l1901_190114

theorem baker_cakes_sold (pastries_made : ℕ) (cakes_made : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) 
  (h1 : pastries_made = 61)
  (h2 : cakes_made = 167)
  (h3 : pastries_sold = 44)
  (h4 : cakes_left = 59) :
  cakes_made - cakes_left = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l1901_190114


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersections_l1901_190105

theorem rectangle_diagonal_intersections (ℓ b : ℕ) (hℓ : ℓ > 0) (hb : b > 0) : 
  let V := ℓ + b - Nat.gcd ℓ b
  ℓ = 6 → b = 4 → V = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersections_l1901_190105


namespace NUMINAMATH_CALUDE_girls_at_game_l1901_190131

theorem girls_at_game (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = girls + 18 → 
  girls = 30 := by
sorry

end NUMINAMATH_CALUDE_girls_at_game_l1901_190131


namespace NUMINAMATH_CALUDE_max_value_abcd_l1901_190136

theorem max_value_abcd (a b c d : ℤ) (hb : b > 0) 
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  (∀ a' b' c' d' : ℤ, b' > 0 → a' + b' = c' → b' + c' = d' → c' + d' = a' → 
    a' - 2*b' + 3*c' - 4*d' ≤ a - 2*b + 3*c - 4*d) ∧ 
  (a - 2*b + 3*c - 4*d = -7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_abcd_l1901_190136


namespace NUMINAMATH_CALUDE_milk_ratio_l1901_190186

def weekday_boxes : ℕ := 3
def saturday_boxes : ℕ := 2 * weekday_boxes
def total_boxes : ℕ := 30

def weekdays : ℕ := 5
def saturdays : ℕ := 1

def sunday_boxes : ℕ := total_boxes - (weekday_boxes * weekdays + saturday_boxes * saturdays)

theorem milk_ratio :
  (sunday_boxes : ℚ) / (weekday_boxes * weekdays : ℚ) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_milk_ratio_l1901_190186


namespace NUMINAMATH_CALUDE_total_price_two_corgis_is_2507_l1901_190169

/-- Calculates the total price for two Corgi dogs with given conditions -/
def total_price_two_corgis (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) (shipping_fee : ℝ) : ℝ :=
  let selling_price := cost * (1 + profit_percent)
  let total_before_discount := 2 * selling_price
  let discounted_price := total_before_discount * (1 - discount_percent)
  let price_with_tax := discounted_price * (1 + tax_percent)
  price_with_tax + shipping_fee

/-- Theorem stating the total price for two Corgi dogs is $2507 -/
theorem total_price_two_corgis_is_2507 :
  total_price_two_corgis 1000 0.30 0.10 0.05 50 = 2507 := by
  sorry

end NUMINAMATH_CALUDE_total_price_two_corgis_is_2507_l1901_190169


namespace NUMINAMATH_CALUDE_pizza_sales_l1901_190188

theorem pizza_sales (pepperoni cheese total : ℕ) (h1 : pepperoni = 2) (h2 : cheese = 6) (h3 : total = 14) :
  total - (pepperoni + cheese) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sales_l1901_190188


namespace NUMINAMATH_CALUDE_price_change_difference_l1901_190183

/-- 
Given that a price is increased by x percent and then decreased by y percent, 
resulting in the same price as the initial price, prove that 1/x - 1/y = -1/100.
-/
theorem price_change_difference (x y : ℝ) 
  (h : (1 + x/100) * (1 - y/100) = 1) : 
  1/x - 1/y = -1/100 :=
sorry

end NUMINAMATH_CALUDE_price_change_difference_l1901_190183


namespace NUMINAMATH_CALUDE_inequality_proof_l1901_190177

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  b / Real.sqrt (a + 2 * c) + c / Real.sqrt (b + 2 * d) + 
  d / Real.sqrt (c + 2 * a) + a / Real.sqrt (d + 2 * b) ≥ 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1901_190177


namespace NUMINAMATH_CALUDE_brother_is_tweedledee_l1901_190118

-- Define the two brothers
inductive Brother
| tweedledee
| tweedledum

-- Define a proposition for "lying today"
def lying_today (b : Brother) : Prop := sorry

-- Define the statement made by the brother
def brother_statement (b : Brother) : Prop :=
  lying_today b ∨ b = Brother.tweedledee

-- Theorem stating that the brother must be Tweedledee
theorem brother_is_tweedledee (b : Brother) : 
  brother_statement b → b = Brother.tweedledee :=
by sorry

end NUMINAMATH_CALUDE_brother_is_tweedledee_l1901_190118


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_achieved_l1901_190128

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 -/
theorem max_area_rectangle (x y : ℕ) (h : x + y = 80) : x * y ≤ 1600 :=
sorry

/-- The maximum area is achieved when both sides are 40 -/
theorem max_area_achieved (x y : ℕ) (h : x + y = 80) : x * y = 1600 ↔ x = 40 ∧ y = 40 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_achieved_l1901_190128


namespace NUMINAMATH_CALUDE_functional_inequality_l1901_190109

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) :
  let f : ℝ → ℝ := λ y => y^2 - y + 1
  2 * f x + x^2 * f (1/x) ≥ (3*x^3 - x^2 + 4*x + 3) / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_inequality_l1901_190109


namespace NUMINAMATH_CALUDE_log_inequality_implication_l1901_190137

theorem log_inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.log 3 / Real.log a < Real.log 3 / Real.log b) ∧
  (Real.log 3 / Real.log b < Real.log 3 / Real.log c) →
  ¬(a < b ∧ b < c) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implication_l1901_190137


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_l1901_190189

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ) (tea1_cost : ℝ)
  (tea2_weight : ℝ) (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 19.2) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_tea_trader_profit_percentage_l1901_190189


namespace NUMINAMATH_CALUDE_cotton_planting_solution_l1901_190190

/-- Represents the cotton planting problem with given parameters -/
structure CottonPlanting where
  total_area : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the required acres per tractor per day -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_area / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for the given parameters, each tractor needs to plant 68 acres per day -/
theorem cotton_planting_solution (cp : CottonPlanting) 
  (h1 : cp.total_area = 1700)
  (h2 : cp.total_days = 5)
  (h3 : cp.first_crew_tractors = 2)
  (h4 : cp.first_crew_days = 2)
  (h5 : cp.second_crew_tractors = 7)
  (h6 : cp.second_crew_days = 3) :
  acres_per_tractor_per_day cp = 68 := by
  sorry

end NUMINAMATH_CALUDE_cotton_planting_solution_l1901_190190


namespace NUMINAMATH_CALUDE_max_value_inequality_l1901_190127

theorem max_value_inequality (x y : ℝ) (hx : x > 1/2) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) → m ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1901_190127


namespace NUMINAMATH_CALUDE_impossible_30_cents_with_5_coins_l1901_190134

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_30_cents_with_5_coins :
  ¬ ∃ (coins : List ℕ), 
    coins.length = 5 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 30 :=
by sorry

end NUMINAMATH_CALUDE_impossible_30_cents_with_5_coins_l1901_190134


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_system_l1901_190174

theorem quadratic_and_inequality_system :
  -- Part 1: Quadratic equation
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  -- Part 2: Inequality system
  (∀ x : ℝ, x - 2*(x-1) ≤ 1 ∧ (1+x)/3 > x-1 ↔ -1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_system_l1901_190174


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1901_190124

/-- The intersection point of two lines is the solution to a system of equations -/
theorem intersection_point_is_solution (x y : ℝ) :
  (y = 2*x + 1) ∧ (y = -x + 4) →  -- Given intersection point equations
  (x = 1 ∧ y = 3) →               -- Given intersection point
  (2*x - y = -1) ∧ (x + y = 4)    -- System of equations to prove
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1901_190124


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1901_190197

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = (4/3) * x ∨ y = -(4/3) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 0) ↔ (y = (4/3) * x ∨ y = -(4/3) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1901_190197


namespace NUMINAMATH_CALUDE_factorial_6_equals_720_l1901_190130

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_equals_720 : factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factorial_6_equals_720_l1901_190130


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l1901_190107

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line ax + 2y = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 2, c := 0 }

/-- The second line x + y = 1 -/
def line2 : Line :=
  { a := 1, b := 1, c := -1 }

/-- Theorem: a = 2 is necessary and sufficient for the lines to be parallel -/
theorem parallel_iff_a_eq_two :
  ∀ a : ℝ, parallel (line1 a) line2 ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l1901_190107


namespace NUMINAMATH_CALUDE_complex_equality_l1901_190113

theorem complex_equality (a b : ℂ) : a - b = 0 → a = b := by sorry

end NUMINAMATH_CALUDE_complex_equality_l1901_190113


namespace NUMINAMATH_CALUDE_jesses_room_length_l1901_190100

theorem jesses_room_length (width : ℝ) (total_area : ℝ) (h1 : width = 8) (h2 : total_area = 96) :
  total_area / width = 12 := by
sorry

end NUMINAMATH_CALUDE_jesses_room_length_l1901_190100


namespace NUMINAMATH_CALUDE_total_birds_caught_l1901_190142

def bird_hunting (day_catch : ℕ) (night_multiplier : ℕ) : ℕ :=
  day_catch + night_multiplier * day_catch

theorem total_birds_caught : bird_hunting 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_caught_l1901_190142


namespace NUMINAMATH_CALUDE_expand_product_l1901_190138

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1901_190138


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1901_190182

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1901_190182


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l1901_190111

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_flavoring_water_ratio : ℚ := 1/2

theorem sport_to_standard_ratio :
  let sport_flavoring := sport_water * (1 / (2 * standard_ratio 2))
  let sport_ratio := sport_flavoring / sport_corn_syrup
  let standard_ratio := (standard_ratio 0) / (standard_ratio 1)
  sport_ratio / standard_ratio = 1/3 := by sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l1901_190111


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1901_190159

theorem rationalize_denominator :
  (2 * Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 5 + Real.sqrt 3) = (3 * Real.sqrt 15 - 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1901_190159


namespace NUMINAMATH_CALUDE_piggy_bank_compartments_l1901_190116

/-- Given a piggy bank with an unknown number of compartments, prove that the number of compartments is 12 based on the given conditions. -/
theorem piggy_bank_compartments :
  ∀ (c : ℕ), -- c represents the number of compartments
  (∀ (i : ℕ), i < c → 2 = 2) → -- Each compartment initially has 2 pennies (this is a trivial condition in Lean)
  (∀ (i : ℕ), i < c → 6 = 6) → -- 6 pennies are added to each compartment (also trivial in Lean)
  (c * (2 + 6) = 96) →         -- Total pennies after adding is 96
  c = 12 := by
sorry


end NUMINAMATH_CALUDE_piggy_bank_compartments_l1901_190116


namespace NUMINAMATH_CALUDE_planet_colonization_combinations_l1901_190192

/-- Represents the number of habitable planets discovered -/
def total_planets : ℕ := 13

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := total_planets - earth_like_planets

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total units available for colonization -/
def available_units : ℕ := 15

/-- Calculates the number of unique combinations of planets that can be occupied -/
def count_combinations : ℕ :=
  (Nat.choose earth_like_planets earth_like_planets * Nat.choose mars_like_planets 5) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 7)

theorem planet_colonization_combinations :
  count_combinations = 96 :=
sorry

end NUMINAMATH_CALUDE_planet_colonization_combinations_l1901_190192


namespace NUMINAMATH_CALUDE_perpendicular_lines_main_theorem_l1901_190122

/-- Two lines are perpendicular if their slopes multiply to -1 or if one of them is vertical --/
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1 ∨ m1 = 0 ∨ m2 = 0

theorem perpendicular_lines (a : ℝ) :
  perpendicular (-a/2) (-1/(a*(a+1))) → a = -3/2 ∨ a = 0 := by
  sorry

/-- The main theorem stating the conditions for perpendicularity of the given lines --/
theorem main_theorem :
  ∀ a : ℝ, (∃ x y : ℝ, a*x + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) →
  perpendicular (-a/2) (-1/(a*(a+1))) →
  a = -3/2 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_main_theorem_l1901_190122


namespace NUMINAMATH_CALUDE_max_profit_is_900_l1901_190171

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℕ) : ℤ := -10 * x + 300

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℕ) : ℤ := (x - 11) * sales_volume x

/-- The selling price that maximizes profit. -/
def optimal_price : ℕ := 20

theorem max_profit_is_900 :
  ∀ x : ℕ, x > 0 → profit x ≤ 900 ∧ profit optimal_price = 900 := by
  sorry

#eval profit optimal_price

end NUMINAMATH_CALUDE_max_profit_is_900_l1901_190171


namespace NUMINAMATH_CALUDE_parallelogram_vertices_l1901_190157

/-- A parallelogram with two known vertices and one side parallel to x-axis -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  parallel_to_x_axis : Bool

/-- The other pair of opposite vertices of the parallelogram -/
def other_vertices (p : Parallelogram) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

theorem parallelogram_vertices (p : Parallelogram) 
  (h1 : p.v1 = (2, -3)) 
  (h2 : p.v2 = (8, 9)) 
  (h3 : p.parallel_to_x_axis = true) : 
  other_vertices p = ((5, -3), (5, 9)) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertices_l1901_190157


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l1901_190110

theorem inverse_proportion_order : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = -6 / (-3) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 2 →
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l1901_190110


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1901_190152

def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence b →
  a 1 = 15 →
  b 1 = 35 →
  a 2 + b 2 = 60 →
  a 36 + b 36 = 400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1901_190152


namespace NUMINAMATH_CALUDE_min_abs_z_l1901_190184

/-- Given a complex number z satisfying |z - 16| + |z + 3i| = 17, 
    the smallest possible value of |z| is 768/265 -/
theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 16) + Complex.abs (z + 3*I) = 17 ∧ 
             Complex.abs w ≤ Complex.abs z ∧
             Complex.abs w = 768 / 265 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l1901_190184


namespace NUMINAMATH_CALUDE_distinct_paths_count_l1901_190162

/-- Represents the number of purple arrows from point A -/
def purple_arrows : Nat := 2

/-- Represents the number of gray arrows each purple arrow leads to -/
def gray_arrows_per_purple : Nat := 2

/-- Represents the number of teal arrows each gray arrow leads to -/
def teal_arrows_per_gray : Nat := 3

/-- Represents the number of yellow arrows each teal arrow leads to -/
def yellow_arrows_per_teal : Nat := 2

/-- Represents the number of yellow arrows that lead to point B -/
def yellow_arrows_to_B : Nat := 4

/-- Theorem stating that the number of distinct paths from A to B is 96 -/
theorem distinct_paths_count : 
  purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B = 96 := by
  sorry

#eval purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B

end NUMINAMATH_CALUDE_distinct_paths_count_l1901_190162


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1901_190120

/-- Given an examination with the following conditions:
  * Total number of questions is 120
  * Each correct answer scores 3 marks
  * Each wrong answer loses 1 mark
  * The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1901_190120


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l1901_190164

-- Define the line equation
def line (k : ℝ) (x y : ℝ) : Prop := y - k * x - 1 = 0

-- Define the ellipse equation
def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / b = 1

-- Define the condition that the line always intersects the ellipse
def always_intersects (b : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, line k x y ∧ ellipse b x y

-- Theorem statement
theorem ellipse_intersection_range :
  ∀ b : ℝ, (always_intersects b) ↔ (b ∈ Set.Icc 1 5 ∪ Set.Ioi 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l1901_190164


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1901_190191

theorem binomial_expansion_coefficient (a b : ℝ) : 
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^5*x^5) → b = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1901_190191


namespace NUMINAMATH_CALUDE_steel_rod_length_l1901_190168

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  /-- The weight of the rod in kilograms -/
  weight : ℝ
  /-- The length of the rod in meters -/
  length : ℝ
  /-- The rod is uniform, so weight per unit length is constant -/
  uniform : weight / length = 19 / 5

/-- Theorem stating that a steel rod weighing 42.75 kg has a length of 11.25 meters -/
theorem steel_rod_length (rod : SteelRod) (h : rod.weight = 42.75) : rod.length = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_steel_rod_length_l1901_190168


namespace NUMINAMATH_CALUDE_points_per_enemy_l1901_190180

theorem points_per_enemy (num_enemies : ℕ) (completion_bonus : ℕ) (total_points : ℕ) 
  (h1 : num_enemies = 6)
  (h2 : completion_bonus = 8)
  (h3 : total_points = 62) :
  (total_points - completion_bonus) / num_enemies = 9 := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_l1901_190180


namespace NUMINAMATH_CALUDE_golus_journey_l1901_190170

theorem golus_journey (a b c : ℝ) (h1 : a = 8) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_golus_journey_l1901_190170


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1901_190132

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^5 = k) 
  (h2 : x = 5 ∧ y = 2 → k = 800) :
  y = 4 → x^2 = 25/32 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1901_190132


namespace NUMINAMATH_CALUDE_right_triangle_area_l1901_190153

/-- A right triangle with one leg of length 15 and an inscribed circle of radius 3 has an area of 60. -/
theorem right_triangle_area (a b c r : ℝ) : 
  a = 15 → -- One leg is 15
  r = 3 → -- Radius of inscribed circle is 3
  a^2 + b^2 = c^2 → -- Right triangle (Pythagorean theorem)
  r * (a + b + c) / 2 = r * b → -- Area formula using semiperimeter and inradius
  a * b / 2 = 60 := by -- Area of the triangle is 60
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1901_190153


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1901_190150

theorem quadratic_inequality_solution (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (0 < c ∧ c < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1901_190150


namespace NUMINAMATH_CALUDE_weekly_payment_problem_l1901_190129

/-- The weekly payment problem -/
theorem weekly_payment_problem (n_pay m_pay total_pay : ℕ) : 
  n_pay = 250 →
  m_pay = (120 * n_pay) / 100 →
  total_pay = m_pay + n_pay →
  total_pay = 550 := by
  sorry

end NUMINAMATH_CALUDE_weekly_payment_problem_l1901_190129


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l1901_190160

/-- The constant term in the expansion of (x^2 - 2/x)^6 -/
def constantTerm : ℤ := 240

/-- The binomial expansion of (x^2 - 2/x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2/x)^6

theorem constant_term_of_expansion :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expansion x) ∧ 
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c = constantTerm) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l1901_190160


namespace NUMINAMATH_CALUDE_scientists_from_usa_l1901_190119

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h_total : total = 70)
  (h_europe : europe = total / 2)
  (h_canada : canada = total / 5)
  (h_sum : total = europe + canada + usa) :
  usa = 21 := by
  sorry

end NUMINAMATH_CALUDE_scientists_from_usa_l1901_190119


namespace NUMINAMATH_CALUDE_jane_rejection_calculation_l1901_190198

/-- The percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.007

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.0075

/-- The fraction of products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.5

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.001

theorem jane_rejection_calculation :
  john_rejection_rate + jane_rejection_rate * jane_inspection_fraction = total_rejection_rate :=
sorry

end NUMINAMATH_CALUDE_jane_rejection_calculation_l1901_190198


namespace NUMINAMATH_CALUDE_side_c_length_l1901_190135

/-- Given a triangle ABC with side lengths a, b, and c, and angle C opposite side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ

/-- The Law of Cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

theorem side_c_length (t : Triangle) 
  (ha : t.a = 2) 
  (hb : t.b = 1) 
  (hC : t.C = π / 3) -- 60° in radians
  (hlawCosines : lawOfCosines t) :
  t.c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_side_c_length_l1901_190135


namespace NUMINAMATH_CALUDE_prime_root_pairs_classification_l1901_190147

/-- A pair of positive primes (p,q) such that 3x^2 - px + q = 0 has two distinct rational roots -/
structure PrimeRootPair where
  p : ℕ
  q : ℕ
  p_prime : Nat.Prime p
  q_prime : Nat.Prime q
  has_distinct_rational_roots : ∃ (x y : ℚ), x ≠ y ∧ 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0

/-- The theorem stating that there are only two pairs of primes satisfying the condition -/
theorem prime_root_pairs_classification : 
  {pair : PrimeRootPair | True} = {⟨5, 2, sorry, sorry, sorry⟩, ⟨7, 2, sorry, sorry, sorry⟩} :=
by sorry

end NUMINAMATH_CALUDE_prime_root_pairs_classification_l1901_190147


namespace NUMINAMATH_CALUDE_survey_methods_correct_l1901_190193

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a school survey --/
structure SchoolSurvey where
  totalStudents : Nat
  method1 : SamplingMethod
  method2 : SamplingMethod

/-- Defines the specific survey conducted by the school --/
def surveyConducted : SchoolSurvey := {
  totalStudents := 240,
  method1 := SamplingMethod.SimpleRandom,
  method2 := SamplingMethod.Systematic
}

/-- Theorem stating that the survey methods are correctly identified --/
theorem survey_methods_correct : 
  surveyConducted.method1 = SamplingMethod.SimpleRandom ∧
  surveyConducted.method2 = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_survey_methods_correct_l1901_190193


namespace NUMINAMATH_CALUDE_miranda_rearrangement_time_l1901_190146

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (num_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Proves that writing all rearrangements of a 6-letter name at 15 per minute takes 0.8 hours -/
theorem miranda_rearrangement_time :
  time_to_write_rearrangements 6 15 = 4/5 := by
  sorry

#eval time_to_write_rearrangements 6 15

end NUMINAMATH_CALUDE_miranda_rearrangement_time_l1901_190146


namespace NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1901_190125

/-- Represents a hexagon that can be part of a rectangle and repositioned to form a square --/
structure Hexagon where
  area : ℝ
  significantDimension : ℝ

/-- Represents a rectangle that can be divided into two congruent hexagons --/
structure Rectangle where
  width : ℝ
  height : ℝ
  hexagons : Fin 2 → Hexagon
  isCongruent : hexagons 0 = hexagons 1

/-- Represents a square formed by repositioning two hexagons --/
structure Square where
  sideLength : ℝ

/-- Theorem stating the relationship between the rectangle, hexagons, and resulting square --/
theorem hexagon_dimension_theorem (rect : Rectangle) (sq : Square) : 
  rect.width = 9 ∧ 
  rect.height = 16 ∧ 
  (rect.width * rect.height = sq.sideLength * sq.sideLength) ∧
  (rect.hexagons 0).significantDimension = 6 := by
  sorry

#check hexagon_dimension_theorem

end NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1901_190125


namespace NUMINAMATH_CALUDE_problem_statement_l1901_190112

theorem problem_statement (x y : ℝ) (h : Real.sqrt (x - 1) + (y + 2)^2 = 0) :
  (x + y)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1901_190112


namespace NUMINAMATH_CALUDE_fraction_sum_l1901_190176

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1901_190176


namespace NUMINAMATH_CALUDE_tan_a_values_l1901_190148

theorem tan_a_values (a : Real) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_values_l1901_190148


namespace NUMINAMATH_CALUDE_problem_solution_l1901_190195

def p (x : ℝ) : Prop := x^2 ≤ 5*x - 4

def q (x a : ℝ) : Prop := x^2 - (a + 2)*x + 2*a ≤ 0

theorem problem_solution :
  (∀ x : ℝ, ¬(p x) ↔ (x < 1 ∨ x > 4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a)) ↔ (1 ≤ a ∧ a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1901_190195


namespace NUMINAMATH_CALUDE_polynomial_equality_main_result_l1901_190161

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_equality (x : ℝ) (a₁ a₂ a₃ a₄ : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄ →
  f a₁ a₂ a₃ a₄ = (b₁, b₂, b₃, b₄) :=
by sorry

theorem main_result : f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_main_result_l1901_190161


namespace NUMINAMATH_CALUDE_accounting_majors_count_l1901_190155

theorem accounting_majors_count 
  (p q r s : ℕ+) 
  (h1 : p * q * r * s = 1365)
  (h2 : 1 < p) (h3 : p < q) (h4 : q < r) (h5 : r < s) :
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_accounting_majors_count_l1901_190155


namespace NUMINAMATH_CALUDE_expression_equality_l1901_190149

theorem expression_equality : 
  Real.sqrt 3 * Real.tan (30 * π / 180) - (1 / 2)⁻¹ + Real.sqrt 8 - |1 - Real.sqrt 2| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1901_190149


namespace NUMINAMATH_CALUDE_power_of_product_cubed_l1901_190121

theorem power_of_product_cubed (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_cubed_l1901_190121


namespace NUMINAMATH_CALUDE_bart_tree_cutting_l1901_190165

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of pieces of firewood Bart burns daily -/
def daily_burn_rate : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := (daily_burn_rate * total_days) / pieces_per_tree

theorem bart_tree_cutting :
  trees_needed = 8 :=
sorry

end NUMINAMATH_CALUDE_bart_tree_cutting_l1901_190165


namespace NUMINAMATH_CALUDE_max_figures_in_cube_l1901_190123

/-- The volume of a rectangular cuboid -/
def volume (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the cube -/
def cube_dim : ℕ := 3

/-- The dimensions of the figure -/
def figure_dim : Vector ℕ 3 := ⟨[2, 2, 1], by simp⟩

/-- The maximum number of figures that can fit in the cube -/
def max_figures : ℕ := 6

theorem max_figures_in_cube :
  (volume cube_dim cube_dim cube_dim) ≥ max_figures * (volume figure_dim[0] figure_dim[1] figure_dim[2]) ∧
  ∀ n : ℕ, n > max_figures → (volume cube_dim cube_dim cube_dim) < n * (volume figure_dim[0] figure_dim[1] figure_dim[2]) :=
by sorry

end NUMINAMATH_CALUDE_max_figures_in_cube_l1901_190123


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1901_190144

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1901_190144


namespace NUMINAMATH_CALUDE_special_line_equation_l1901_190196

/-- A line passing through a point and at a fixed distance from the origin -/
structure SpecialLine where
  a : ℝ  -- x-coordinate of the point
  b : ℝ  -- y-coordinate of the point
  d : ℝ  -- distance from the origin

/-- The equation of the special line -/
def lineEquation (l : SpecialLine) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = l.a ∨ 3 * p.1 + 4 * p.2 - 5 = 0}

/-- Theorem: The equation of the line passing through (-1, 2) and at a distance of 1 from the origin -/
theorem special_line_equation :
  let l : SpecialLine := ⟨-1, 2, 1⟩
  lineEquation l = {p : ℝ × ℝ | p.1 = -1 ∨ 3 * p.1 + 4 * p.2 - 5 = 0} := by
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l1901_190196


namespace NUMINAMATH_CALUDE_greatest_among_five_l1901_190181

theorem greatest_among_five : ∀ (a b c d e : ℕ), 
  a = 5 → b = 8 → c = 4 → d = 3 → e = 2 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_greatest_among_five_l1901_190181


namespace NUMINAMATH_CALUDE_solve_paint_problem_l1901_190108

def paint_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) : Prop :=
  ∃ (cans_per_room : ℚ) (total_cans : ℕ),
    cans_per_room > 0 ∧
    total_cans * cans_per_room = original_rooms ∧
    (total_cans - lost_cans) * cans_per_room = remaining_rooms ∧
    remaining_rooms / cans_per_room = 17

theorem solve_paint_problem :
  paint_problem 42 4 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_paint_problem_l1901_190108


namespace NUMINAMATH_CALUDE_special_elements_in_100_l1901_190115

/-- Represents the number of elements in the nth group -/
def group_size (n : ℕ) : ℕ := n + 1

/-- Calculates the total number of elements up to and including the nth group -/
def total_elements (n : ℕ) : ℕ := n * (n + 3) / 2

/-- Represents the number of special elements in the first n groups -/
def special_elements (n : ℕ) : ℕ := n

theorem special_elements_in_100 :
  ∃ n : ℕ, total_elements n ≤ 100 ∧ total_elements (n + 1) > 100 ∧ special_elements n = 12 :=
sorry

end NUMINAMATH_CALUDE_special_elements_in_100_l1901_190115


namespace NUMINAMATH_CALUDE_square_sum_of_three_reals_l1901_190194

theorem square_sum_of_three_reals (x y z : ℝ) 
  (h1 : (x + y + z)^2 = 25)
  (h2 : x*y + x*z + y*z = 8) :
  x^2 + y^2 + z^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_three_reals_l1901_190194


namespace NUMINAMATH_CALUDE_village_population_equality_l1901_190117

/-- Represents the population change in a village over time. -/
structure VillagePopulation where
  initial : ℕ  -- Initial population
  rate : ℤ     -- Annual rate of change (positive for increase, negative for decrease)

/-- Calculates the population after a given number of years. -/
def population_after (v : VillagePopulation) (years : ℕ) : ℤ :=
  v.initial + v.rate * years

theorem village_population_equality (village_x village_y : VillagePopulation) 
  (h1 : village_x.initial = 78000)
  (h2 : village_x.rate = -1200)
  (h3 : village_y.initial = 42000)
  (h4 : population_after village_x 18 = population_after village_y 18) :
  village_y.rate = 800 := by
  sorry

#check village_population_equality

end NUMINAMATH_CALUDE_village_population_equality_l1901_190117


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l1901_190173

-- Define the initial solution parameters
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.35
def target_concentration : ℝ := 0.50

-- Define the amount of pure alcohol to be added
def added_alcohol : ℝ := 1.8

-- Theorem statement
theorem alcohol_concentration_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_concentration := by
  sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l1901_190173


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1901_190175

-- Define the square PQRS
def square_side : ℝ := 7

-- Define the shaded areas
def shaded_area_1 : ℝ := 2^2
def shaded_area_2 : ℝ := 5^2 - 3^2
def shaded_area_3 : ℝ := square_side^2 - 6^2

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2 + shaded_area_3

-- Total area of square PQRS
def total_area : ℝ := square_side^2

-- Theorem statement
theorem shaded_area_percentage :
  total_shaded_area = 33 ∧ (total_shaded_area / total_area) = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1901_190175


namespace NUMINAMATH_CALUDE_original_typing_speed_l1901_190187

theorem original_typing_speed 
  (original_speed : ℕ) 
  (speed_decrease : ℕ) 
  (words_typed : ℕ) 
  (time_taken : ℕ) :
  speed_decrease = 40 →
  words_typed = 3440 →
  time_taken = 20 →
  (original_speed - speed_decrease) * time_taken = words_typed →
  original_speed = 212 := by
sorry

end NUMINAMATH_CALUDE_original_typing_speed_l1901_190187


namespace NUMINAMATH_CALUDE_number_of_teams_in_league_l1901_190102

/-- The number of teams in the league -/
def n : ℕ := 20

/-- The number of games each team plays against every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := 760

/-- Theorem stating that n is the correct number of teams in the league -/
theorem number_of_teams_in_league :
  n * (n - 1) * games_per_pair / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_number_of_teams_in_league_l1901_190102


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1901_190163

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the theorem
theorem min_value_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧ (∃ x ∈ Set.Icc a (a + 6), f x = 9) ↔ a = 2 ∨ a = -10 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1901_190163


namespace NUMINAMATH_CALUDE_vertical_line_intercept_difference_l1901_190145

/-- A vertical line passing through two points -/
structure VerticalLine where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The x-intercept of a vertical line -/
def x_intercept (l : VerticalLine) : ℝ := l.x

/-- Theorem: For a vertical line passing through points C(7, 5) and D(7, -3),
    the difference between the x-intercept of the line and the y-coordinate of point C is 2 -/
theorem vertical_line_intercept_difference (l : VerticalLine) 
    (h1 : l.x = 7) 
    (h2 : l.y1 = 5) 
    (h3 : l.y2 = -3) : 
  x_intercept l - l.y1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_intercept_difference_l1901_190145


namespace NUMINAMATH_CALUDE_perpendicular_parallel_lines_to_plane_l1901_190126

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- The theorem to be proved -/
theorem perpendicular_parallel_lines_to_plane 
  (a b : Line) (α : Plane) 
  (h1 : a ≠ b) 
  (h2 : parallel_lines a b) 
  (h3 : perpendicular_line_plane a α) : 
  perpendicular_line_plane b α := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_lines_to_plane_l1901_190126


namespace NUMINAMATH_CALUDE_sin_max_at_neg_pi_fourth_l1901_190185

/-- The smallest positive constant c such that y = 3 sin(2x + c) reaches a maximum at x = -π/4 is π -/
theorem sin_max_at_neg_pi_fourth (c : ℝ) :
  c > 0 ∧ 
  (∀ x : ℝ, 3 * Real.sin (2 * x + c) ≤ 3 * Real.sin (2 * (-π/4) + c)) →
  c = π :=
sorry

end NUMINAMATH_CALUDE_sin_max_at_neg_pi_fourth_l1901_190185


namespace NUMINAMATH_CALUDE_min_value_fraction_l1901_190104

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  (x + y) / (x^2) ≥ -1/12 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1901_190104


namespace NUMINAMATH_CALUDE_calculate_savings_savings_calculation_l1901_190166

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * income) / income_ratio
  income - expenditure

/-- Prove that given a person's income and expenditure ratio of 10:7 and an income of Rs. 10000, the person's savings are Rs. 3000. -/
theorem savings_calculation : calculate_savings 10 7 10000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_savings_calculation_l1901_190166


namespace NUMINAMATH_CALUDE_jeonghoons_math_score_l1901_190172

theorem jeonghoons_math_score 
  (ethics : ℕ) (korean : ℕ) (science : ℕ) (social : ℕ) (average : ℕ) :
  ethics = 82 →
  korean = 90 →
  science = 88 →
  social = 84 →
  average = 88 →
  (ethics + korean + science + social + (average * 5 - (ethics + korean + science + social))) / 5 = average →
  average * 5 - (ethics + korean + science + social) = 96 :=
by sorry

end NUMINAMATH_CALUDE_jeonghoons_math_score_l1901_190172


namespace NUMINAMATH_CALUDE_weighted_mean_calculation_l1901_190140

def numbers : List ℝ := [16, 28, 45]
def weights : List ℝ := [2, 3, 5]

theorem weighted_mean_calculation :
  (List.sum (List.zipWith (· * ·) numbers weights)) / (List.sum weights) = 34.1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_mean_calculation_l1901_190140


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_l1901_190167

theorem mean_equality_implies_y (y : ℝ) : 
  (7 + 9 + 14 + 23) / 4 = (18 + y) / 2 → y = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_l1901_190167


namespace NUMINAMATH_CALUDE_remainder_problem_l1901_190141

theorem remainder_problem :
  (85^70 + 19^32)^16 ≡ 16 [ZMOD 21] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1901_190141


namespace NUMINAMATH_CALUDE_total_placards_taken_l1901_190133

/-- The number of placards taken by people entering a stadium -/
def placards_taken (people : ℕ) (placards_per_person : ℕ) : ℕ :=
  people * placards_per_person

/-- Theorem stating the total number of placards taken by 2841 people -/
theorem total_placards_taken :
  placards_taken 2841 2 = 5682 := by
  sorry

end NUMINAMATH_CALUDE_total_placards_taken_l1901_190133


namespace NUMINAMATH_CALUDE_ratio_comparison_is_three_l1901_190178

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
def sport_water_ratio : ℚ := standard_ratio 2 * 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 75

/-- The ratio of flavoring to corn syrup in the sport formulation compared to the standard formulation -/
def ratio_comparison : ℚ :=
  (sport_water / sport_water_ratio) / sport_corn_syrup /
  (standard_ratio 0 / standard_ratio 1)

theorem ratio_comparison_is_three : ratio_comparison = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_is_three_l1901_190178


namespace NUMINAMATH_CALUDE_M_intersect_N_l1901_190106

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1901_190106


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l1901_190179

/-- Given that 5 pages cost 10 cents, prove that $15 can copy 750 pages. -/
theorem pages_copied_for_fifteen_dollars : 
  let cost_per_five_pages : ℚ := 10 / 100  -- 10 cents in dollars
  let total_amount : ℚ := 15  -- $15
  let pages_per_dollar : ℚ := 5 / cost_per_five_pages
  ⌊total_amount * pages_per_dollar⌋ = 750 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l1901_190179


namespace NUMINAMATH_CALUDE_fraction_of_120_l1901_190199

theorem fraction_of_120 : (1 / 6 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) * 120 = 1 ∧ 1 = 2 * (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l1901_190199


namespace NUMINAMATH_CALUDE_lcm_12_15_18_l1901_190154

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_15_18_l1901_190154


namespace NUMINAMATH_CALUDE_equal_remainders_implies_m_zero_l1901_190139

-- Define the polynomials
def P₁ (m : ℝ) (y : ℝ) : ℝ := 29 * 42 * y^2 + m * y + 2
def P₂ (m : ℝ) (y : ℝ) : ℝ := y^2 + m * y + 2

-- Define the remainder functions
def R₁ (m : ℝ) : ℝ := P₁ m 1
def R₂ (m : ℝ) : ℝ := P₂ m (-1)

-- Theorem statement
theorem equal_remainders_implies_m_zero :
  ∀ m : ℝ, R₁ m = R₂ m → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_remainders_implies_m_zero_l1901_190139


namespace NUMINAMATH_CALUDE_correct_evaluation_l1901_190101

/-- Evaluates an expression according to right-to-left rules -/
noncomputable def evaluate (a b c d e : ℝ) : ℝ :=
  a * (b^c - (d + e))

/-- Theorem stating that the evaluation is correct -/
theorem correct_evaluation (a b c d e : ℝ) :
  evaluate a b c d e = a * (b^c - (d + e)) := by sorry

end NUMINAMATH_CALUDE_correct_evaluation_l1901_190101
