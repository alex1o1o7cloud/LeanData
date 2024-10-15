import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l532_53299

theorem complex_equation_solution (z : ℂ) (h : (3 - 4 * Complex.I) * z = 25) : z = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l532_53299


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l532_53285

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- Define the theorem
theorem range_of_a_when_p_is_false :
  (∀ a : ℝ, ¬(p a) ↔ a > 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l532_53285


namespace NUMINAMATH_CALUDE_min_value_of_xy_l532_53210

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_xy_l532_53210


namespace NUMINAMATH_CALUDE_total_sequences_is_288_l532_53252

/-- Represents a team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  day1_matches : List Match
  no_ties : Bool

/-- Calculates the number of possible outcomes for a given number of matches -/
def possible_outcomes (num_matches : Nat) : Nat :=
  2^num_matches

/-- Calculates the number of possible arrangements for the winners' group on day 2 -/
def winners_arrangements (num_winners : Nat) : Nat :=
  Nat.factorial num_winners

/-- Calculates the number of possible outcomes for the losers' match on day 2 -/
def losers_match_outcomes (num_losers : Nat) : Nat :=
  num_losers * 2

/-- Calculates the total number of possible ranking sequences -/
def total_sequences (t : Tournament) : Nat :=
  possible_outcomes t.day1_matches.length *
  winners_arrangements 3 *
  losers_match_outcomes 3 *
  possible_outcomes 1

/-- The theorem stating that the total number of possible ranking sequences is 288 -/
theorem total_sequences_is_288 (t : Tournament) 
  (h1 : t.day1_matches.length = 3)
  (h2 : t.no_ties = true) :
  total_sequences t = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_sequences_is_288_l532_53252


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l532_53235

theorem sum_of_solutions_is_zero :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (9 * x₁) / 27 = 6 / x₁ ∧
  (9 * x₂) / 27 = 6 / x₂ ∧
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l532_53235


namespace NUMINAMATH_CALUDE_pairwise_ratio_sum_geq_three_halves_l532_53280

theorem pairwise_ratio_sum_geq_three_halves
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pairwise_ratio_sum_geq_three_halves_l532_53280


namespace NUMINAMATH_CALUDE_g_of_4_l532_53205

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_of_4 : g 4 = 22 := by sorry

end NUMINAMATH_CALUDE_g_of_4_l532_53205


namespace NUMINAMATH_CALUDE_unique_solution_l532_53272

theorem unique_solution : ∃! (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ 14 * m * n = 55 - 7 * m - 2 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l532_53272


namespace NUMINAMATH_CALUDE_temperature_drop_per_tree_l532_53278

/-- Proves that the temperature drop per tree is 0.1 degrees -/
theorem temperature_drop_per_tree 
  (cost_per_tree : ℝ) 
  (initial_temp : ℝ) 
  (final_temp : ℝ) 
  (total_cost : ℝ) 
  (h1 : cost_per_tree = 6)
  (h2 : initial_temp = 80)
  (h3 : final_temp = 78.2)
  (h4 : total_cost = 108) :
  (initial_temp - final_temp) / (total_cost / cost_per_tree) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_per_tree_l532_53278


namespace NUMINAMATH_CALUDE_percentage_to_full_amount_l532_53260

theorem percentage_to_full_amount (amount : ℝ) : 
  (25 / 100) * amount = 200 → amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_full_amount_l532_53260


namespace NUMINAMATH_CALUDE_box_two_three_neg_two_l532_53271

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

-- Theorem statement
theorem box_two_three_neg_two :
  box 2 3 (-2) = 107 / 9 := by sorry

end NUMINAMATH_CALUDE_box_two_three_neg_two_l532_53271


namespace NUMINAMATH_CALUDE_fraction_meaningful_l532_53208

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l532_53208


namespace NUMINAMATH_CALUDE_three_zeros_range_l532_53290

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Define the property of having 3 zeros
def has_three_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

-- Theorem statement
theorem three_zeros_range :
  ∀ a : ℝ, has_three_zeros a ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_range_l532_53290


namespace NUMINAMATH_CALUDE_petes_flag_total_shapes_l532_53221

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares : ℕ := us_stripes * 2 + 6

/-- Theorem stating the total number of shapes on Pete's flag -/
theorem petes_flag_total_shapes :
  petes_circles + petes_squares = 54 := by sorry

end NUMINAMATH_CALUDE_petes_flag_total_shapes_l532_53221


namespace NUMINAMATH_CALUDE_base_representation_of_200_l532_53211

theorem base_representation_of_200 :
  ∃! b : ℕ, b > 1 ∧ b^5 ≤ 200 ∧ 200 < b^6 := by sorry

end NUMINAMATH_CALUDE_base_representation_of_200_l532_53211


namespace NUMINAMATH_CALUDE_calculate_net_profit_l532_53281

/-- Given a purchase price, overhead percentage, and markup, calculate the net profit -/
theorem calculate_net_profit (purchase_price overhead_percentage markup : ℝ) :
  purchase_price = 48 →
  overhead_percentage = 0.20 →
  markup = 45 →
  let overhead := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead
  let selling_price := total_cost + markup
  let net_profit := selling_price - total_cost
  net_profit = 45 := by
  sorry

end NUMINAMATH_CALUDE_calculate_net_profit_l532_53281


namespace NUMINAMATH_CALUDE_tangent_point_on_parabola_l532_53217

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the derivative of the parabola function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_point_on_parabola :
  let M : ℝ × ℝ := (1, 0)
  f M.1 = M.2 ∧ f' M.1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_point_on_parabola_l532_53217


namespace NUMINAMATH_CALUDE_prob_rain_all_days_l532_53256

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.2

theorem prob_rain_all_days :
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_all_days_l532_53256


namespace NUMINAMATH_CALUDE_square_ratios_l532_53223

theorem square_ratios (a b : ℝ) (h : b = 3 * a) :
  (4 * b) / (4 * a) = 3 ∧ (b * b) / (a * a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_ratios_l532_53223


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l532_53233

def total_figures : ℕ := 12
def num_triangles : ℕ := 4
def num_circles : ℕ := 3
def num_squares : ℕ := 5

theorem probability_triangle_or_circle :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l532_53233


namespace NUMINAMATH_CALUDE_assembled_figure_surface_area_l532_53267

/-- The surface area of a figure assembled from four identical blocks -/
def figureSurfaceArea (blockSurfaceArea : ℝ) (lostAreaPerBlock : ℝ) : ℝ :=
  4 * (blockSurfaceArea - lostAreaPerBlock)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem assembled_figure_surface_area :
  figureSurfaceArea 18 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_assembled_figure_surface_area_l532_53267


namespace NUMINAMATH_CALUDE_degree_of_specific_polynomial_l532_53253

/-- The degree of a polynomial of the form (aᵏ * bⁿ) where a and b are polynomials -/
def degree_product_power (deg_a deg_b k n : ℕ) : ℕ := k * deg_a + n * deg_b

/-- The degree of the polynomial (x³ + x + 1)⁵ * (x⁴ + x² + 1)² -/
def degree_specific_polynomial : ℕ :=
  degree_product_power 3 4 5 2

theorem degree_of_specific_polynomial :
  degree_specific_polynomial = 23 := by sorry

end NUMINAMATH_CALUDE_degree_of_specific_polynomial_l532_53253


namespace NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_5_l532_53265

open Complex

theorem abs_z_equals_2_sqrt_5 (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  abs z = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_5_l532_53265


namespace NUMINAMATH_CALUDE_train_speed_l532_53289

/-- Proves that the speed of a train is 36 km/hr given specific conditions -/
theorem train_speed (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 240 →
  train_length = 120 →
  passing_time = 36 →
  (initial_distance + train_length) / passing_time * 3.6 = 36 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l532_53289


namespace NUMINAMATH_CALUDE_set_M_equals_one_two_three_l532_53249

def M : Set ℤ := {a | 0 < 2*a - 1 ∧ 2*a - 1 ≤ 5}

theorem set_M_equals_one_two_three : M = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_M_equals_one_two_three_l532_53249


namespace NUMINAMATH_CALUDE_classroom_writing_instruments_l532_53227

theorem classroom_writing_instruments :
  let total_bags : ℕ := 16
  let compartments_per_bag : ℕ := 6
  let max_instruments_per_compartment : ℕ := 8
  let empty_compartments : ℕ := 5
  let partially_filled_compartment : ℕ := 1
  let instruments_in_partially_filled : ℕ := 6
  
  let total_compartments : ℕ := total_bags * compartments_per_bag
  let filled_compartments : ℕ := total_compartments - empty_compartments - partially_filled_compartment
  
  let total_instruments : ℕ := 
    filled_compartments * max_instruments_per_compartment + 
    partially_filled_compartment * instruments_in_partially_filled
  
  total_instruments = 726 := by
  sorry

end NUMINAMATH_CALUDE_classroom_writing_instruments_l532_53227


namespace NUMINAMATH_CALUDE_mean_temperature_l532_53266

def temperatures : List Int := [-8, -6, -3, -3, 0, 4, -1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -17 / 7 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l532_53266


namespace NUMINAMATH_CALUDE_stating_safe_zone_condition_l532_53202

/-- Represents the fuse burning speed in cm/s -/
def fuse_speed : ℝ := 0.5

/-- Represents the person's running speed in m/s -/
def person_speed : ℝ := 4

/-- Represents the safe zone distance in meters -/
def safe_distance : ℝ := 150

/-- 
Theorem stating the condition for a person to reach the safe zone before the fuse burns out.
x represents the fuse length in cm.
-/
theorem safe_zone_condition (x : ℝ) :
  (x ≥ 0) →
  (person_speed * (x / fuse_speed) ≥ safe_distance) ↔
  (4 * (x / 0.5) ≥ 150) :=
sorry

end NUMINAMATH_CALUDE_stating_safe_zone_condition_l532_53202


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l532_53250

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_condition (x : ℝ) : 
  let a : Vector2D := (1, 2)
  let b : Vector2D := (-2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l532_53250


namespace NUMINAMATH_CALUDE_complex_cube_root_l532_53291

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a - ↑b * Complex.I) ^ 3 = 27 - 27 * Complex.I →
  ↑a - ↑b * Complex.I = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l532_53291


namespace NUMINAMATH_CALUDE_school_children_count_l532_53259

theorem school_children_count :
  let absent_children : ℕ := 160
  let total_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present + 2 * absent
  let extra_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present
  let boys_bananas : ℕ → ℕ := λ total => 3 * (total / 4)
  let girls_bananas : ℕ → ℕ := λ total => total / 4
  ∃ (present_children : ℕ),
    total_bananas present_children absent_children = 
      total_bananas present_children present_children + extra_bananas present_children absent_children ∧
    boys_bananas (total_bananas present_children absent_children) + 
      girls_bananas (total_bananas present_children absent_children) = 
      total_bananas present_children absent_children ∧
    present_children + absent_children = 6560 :=
by sorry

end NUMINAMATH_CALUDE_school_children_count_l532_53259


namespace NUMINAMATH_CALUDE_trajectory_equation_l532_53206

/-- The equation of the trajectory of the center of a circle that passes through point A (2, 0) and is tangent to the circle x^2 + 4x + y^2 - 32 = 0 is x^2/9 + y^2/5 = 1 -/
theorem trajectory_equation : ∃ (f : ℝ × ℝ → ℝ), 
  (∀ (x y : ℝ), f (x, y) = 0 ↔ x^2/9 + y^2/5 = 1) ∧
  (∀ (x y : ℝ), f (x, y) = 0 → 
    ∃ (r : ℝ), r > 0 ∧
    (∀ (u v : ℝ), (u - x)^2 + (v - y)^2 = r^2 → 
      ((u - 2)^2 + v^2 = 0 ∨ u^2 + 4*u + v^2 - 32 = 0))) :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l532_53206


namespace NUMINAMATH_CALUDE_M_subset_range_l532_53236

def M (a : ℝ) := {x : ℝ | x^2 + 2*(1-a)*x + 3-a ≤ 0}

theorem M_subset_range (a : ℝ) : M a ⊆ Set.Icc 0 3 ↔ -1 ≤ a ∧ a ≤ 18/7 := by sorry

end NUMINAMATH_CALUDE_M_subset_range_l532_53236


namespace NUMINAMATH_CALUDE_polynomial_ratio_theorem_l532_53238

/-- The polynomial f(x) = x^2007 + 17x^2006 + 1 -/
def f (x : ℂ) : ℂ := x^2007 + 17*x^2006 + 1

/-- The set of distinct zeros of f -/
def zeros : Finset ℂ := sorry

/-- The polynomial P of degree 2007 -/
noncomputable def P : Polynomial ℂ := sorry

theorem polynomial_ratio_theorem :
  (∀ r ∈ zeros, f r = 0) →
  (Finset.card zeros = 2007) →
  (∀ r ∈ zeros, P.eval (r + 1/r) = 0) →
  (Polynomial.degree P = 2007) →
  P.eval 1 / P.eval (-1) = 289 / 259 := by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_theorem_l532_53238


namespace NUMINAMATH_CALUDE_production_equation_l532_53294

/-- Represents the production of machines in a factory --/
structure MachineProduction where
  x : ℝ  -- Actual number of machines produced per day
  original_plan : ℝ  -- Original planned production per day
  increased_production : ℝ  -- Increase in production per day
  time_500 : ℝ  -- Time to produce 500 machines at current rate
  time_300 : ℝ  -- Time to produce 300 machines at original rate

/-- Theorem stating the relationship between production rates and times --/
theorem production_equation (mp : MachineProduction) 
  (h1 : mp.x = mp.original_plan + mp.increased_production)
  (h2 : mp.increased_production = 20)
  (h3 : mp.time_500 = 500 / mp.x)
  (h4 : mp.time_300 = 300 / mp.original_plan)
  (h5 : mp.time_500 = mp.time_300) :
  500 / mp.x = 300 / (mp.x - 20) := by
  sorry

end NUMINAMATH_CALUDE_production_equation_l532_53294


namespace NUMINAMATH_CALUDE_prob_two_queens_or_two_aces_value_l532_53274

-- Define the deck
def total_cards : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

-- Define the probability function
noncomputable def prob_two_queens_or_two_aces : ℚ :=
  let two_queens := (num_queens.choose 2) * ((total_cards - num_queens).choose 1)
  let two_aces := (num_aces.choose 2) * ((total_cards - num_aces).choose 1)
  let three_aces := num_aces.choose 3
  (two_queens + two_aces + three_aces) / (total_cards.choose 3)

-- State the theorem
theorem prob_two_queens_or_two_aces_value : 
  prob_two_queens_or_two_aces = 29 / 1105 := by sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_two_aces_value_l532_53274


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l532_53247

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l532_53247


namespace NUMINAMATH_CALUDE_average_income_proof_l532_53218

def income_days : Nat := 5

def daily_incomes : List ℝ := [400, 250, 650, 400, 500]

theorem average_income_proof :
  (daily_incomes.sum / income_days : ℝ) = 440 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l532_53218


namespace NUMINAMATH_CALUDE_function_property_l532_53268

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * x^2 - x + 1
  f 2 = 9 → f (-2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l532_53268


namespace NUMINAMATH_CALUDE_number_is_two_l532_53222

theorem number_is_two (x y : ℝ) (n : ℝ) 
  (h1 : n * (x - y) = 4)
  (h2 : 6 * x - 3 * y = 12) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_is_two_l532_53222


namespace NUMINAMATH_CALUDE_exposed_sides_is_21_l532_53228

/-- Represents a polygon with a specific number of sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the configuration of polygons -/
structure PolygonConfiguration where
  triangle : Polygon
  square : Polygon
  pentagon : Polygon
  hexagon : Polygon
  heptagon : Polygon
  triangle_is_equilateral : triangle.sides = 3
  square_is_square : square.sides = 4
  pentagon_is_pentagon : pentagon.sides = 5
  hexagon_is_hexagon : hexagon.sides = 6
  heptagon_is_heptagon : heptagon.sides = 7

/-- The number of shared sides in the configuration -/
def shared_sides : ℕ := 4

/-- Theorem stating that the number of exposed sides in the configuration is 21 -/
theorem exposed_sides_is_21 (config : PolygonConfiguration) : 
  config.triangle.sides + config.square.sides + config.pentagon.sides + 
  config.hexagon.sides + config.heptagon.sides - shared_sides = 21 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_is_21_l532_53228


namespace NUMINAMATH_CALUDE_first_discount_percentage_l532_53287

/-- Given an initial price of 400, a final price of 240 after two discounts,
    where the second discount is 20%, prove that the first discount is 25%. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  initial_price = 400 →
  final_price = 240 →
  second_discount = 20 →
  ∃ (first_discount : ℝ),
    first_discount = 25 ∧ 
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l532_53287


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l532_53255

/-- The function f(x) = 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x + m

/-- m is not in the open interval (-3, -1) -/
def not_in_interval (m : ℝ) : Prop := m ≤ -3 ∨ m ≥ -1

/-- f has no zero in the interval [0, 1] -/
def no_zero_in_interval (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, f m x ≠ 0

theorem necessary_not_sufficient :
  (∀ m : ℝ, no_zero_in_interval m → not_in_interval m) ∧
  (∃ m : ℝ, not_in_interval m ∧ ¬(no_zero_in_interval m)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l532_53255


namespace NUMINAMATH_CALUDE_num_pupils_correct_l532_53214

/-- The number of pupils sent up for examination -/
def num_pupils : ℕ := 21

/-- The average marks of all pupils -/
def average_marks : ℚ := 39

/-- The marks of the 4 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19]

/-- The average marks if the 4 specific pupils were removed -/
def average_without_specific : ℚ := 44

/-- Theorem stating that the number of pupils is correct given the conditions -/
theorem num_pupils_correct :
  (average_marks * num_pupils : ℚ) =
  (average_without_specific * (num_pupils - 4) : ℚ) + (specific_pupils_marks.sum : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_num_pupils_correct_l532_53214


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l532_53207

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l532_53207


namespace NUMINAMATH_CALUDE_car_speed_proof_l532_53239

/-- 
Proves that a car traveling at a constant speed v km/h takes 2 seconds longer 
to travel 1 kilometer than it would at 450 km/h if and only if v = 360 km/h.
-/
theorem car_speed_proof (v : ℝ) : v > 0 → (
  (1 / v) * 3600 = (1 / 450) * 3600 + 2 ↔ v = 360
) := by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l532_53239


namespace NUMINAMATH_CALUDE_spade_calculation_l532_53276

/-- The ⋆ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : 
  let z : ℝ := 2
  spade 2 (spade 3 (1 + z)) = 4 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l532_53276


namespace NUMINAMATH_CALUDE_don_tiles_per_minute_l532_53248

/-- The number of tiles Don can paint per minute -/
def D : ℕ := sorry

/-- The number of tiles Ken can paint per minute -/
def ken_tiles : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def laura_tiles : ℕ := 2 * (D + 2)

/-- The number of tiles Kim can paint per minute -/
def kim_tiles : ℕ := 2 * (D + 2) - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

theorem don_tiles_per_minute :
  D + ken_tiles + laura_tiles + kim_tiles = total_tiles / 15 ∧ D = 3 := by sorry

end NUMINAMATH_CALUDE_don_tiles_per_minute_l532_53248


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_geometric_progression_l532_53219

theorem smallest_b_in_arithmetic_geometric_progression (a b c : ℤ) : 
  a < c → c < b → 
  (2 * c = a + b) →  -- arithmetic progression condition
  (b * b = a * c) →  -- geometric progression condition
  (∀ b' : ℤ, (∃ a' c' : ℤ, a' < c' ∧ c' < b' ∧ 
    (2 * c' = a' + b') ∧ 
    (b' * b' = a' * c')) → b' ≥ 2) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_geometric_progression_l532_53219


namespace NUMINAMATH_CALUDE_remainder_8354_mod_11_l532_53200

theorem remainder_8354_mod_11 : 8354 % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8354_mod_11_l532_53200


namespace NUMINAMATH_CALUDE_shorter_tank_radius_l532_53258

/-- Given two cylindrical tanks with equal volumes, where one tank is twice as tall as the other,
    and the radius of the taller tank is 10 units, the radius of the shorter tank is 10√2 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  let v := π * (10^2) * (2*h)  -- Volume of the taller tank
  let r := Real.sqrt 200       -- Radius of the shorter tank
  v = π * r^2 * h              -- Volumes are equal
  → r = 10 * Real.sqrt 2       -- Radius of the shorter tank is 10√2
  := by sorry

end NUMINAMATH_CALUDE_shorter_tank_radius_l532_53258


namespace NUMINAMATH_CALUDE_tribe_leadership_combinations_l532_53263

theorem tribe_leadership_combinations (n : ℕ) (h : n = 15) : 
  (n) *                             -- Choose the chief
  (Nat.choose (n - 1) 2) *          -- Choose 2 supporting chiefs
  (Nat.choose (n - 3) 2) *          -- Choose 2 inferior officers for chief A
  (Nat.choose (n - 5) 2) *          -- Choose 2 assistants for A's officers
  (Nat.choose (n - 7) 2) *          -- Choose 2 inferior officers for chief B
  (Nat.choose (n - 9) 2) *          -- Choose 2 assistants for B's officers
  (Nat.choose (n - 11) 2) *         -- Choose 2 assistants for B's officers
  (Nat.choose (n - 13) 2) = 400762320000 := by
sorry

end NUMINAMATH_CALUDE_tribe_leadership_combinations_l532_53263


namespace NUMINAMATH_CALUDE_smallest_multiple_36_with_digit_sum_multiple_9_l532_53262

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem smallest_multiple_36_with_digit_sum_multiple_9 :
  ∃ (k : ℕ), k > 0 ∧ 36 * k = 36 ∧
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(∃ n : ℕ, 36 * m = 36 * n ∧ 9 ∣ sumOfDigits (36 * n))) ∧
  (9 ∣ sumOfDigits 36) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_36_with_digit_sum_multiple_9_l532_53262


namespace NUMINAMATH_CALUDE_correct_calculation_l532_53204

theorem correct_calculation (x : ℝ) : 
  x / 3.6 = 2.5 → (x * 3.6) / 2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l532_53204


namespace NUMINAMATH_CALUDE_last_day_sales_l532_53261

/-- The number of packs sold by Lucy and Robyn on their last day -/
def total_packs_sold (lucy_packs robyn_packs : ℕ) : ℕ :=
  lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold by Lucy and Robyn is 35 -/
theorem last_day_sales : total_packs_sold 19 16 = 35 := by
  sorry

end NUMINAMATH_CALUDE_last_day_sales_l532_53261


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l532_53251

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : 
  (2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l532_53251


namespace NUMINAMATH_CALUDE_smallest_n_with_right_triangle_l532_53264

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The set S containing numbers from 1 to 50 --/
def S : Finset ℕ := Finset.range 50

/-- A property that checks if a subset of size n always contains a right triangle --/
def hasRightTriangle (n : ℕ) : Prop :=
  ∀ (T : Finset ℕ), T ⊆ S → T.card = n →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ isRightTriangle a b c

/-- The main theorem stating that 42 is the smallest n satisfying the property --/
theorem smallest_n_with_right_triangle :
  hasRightTriangle 42 ∧ ∀ m < 42, ¬(hasRightTriangle m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_right_triangle_l532_53264


namespace NUMINAMATH_CALUDE_total_profit_is_63000_l532_53237

/-- Calculates the total profit earned by two partners based on their investments and one partner's share of the profit. -/
def calculateTotalProfit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and Jose's profit share, the total profit is 63000. -/
theorem total_profit_is_63000 :
  calculateTotalProfit 30000 12 45000 10 35000 = 63000 :=
sorry

end NUMINAMATH_CALUDE_total_profit_is_63000_l532_53237


namespace NUMINAMATH_CALUDE_number_problem_l532_53242

theorem number_problem (x : ℝ) : 0.60 * x - 40 = 50 → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l532_53242


namespace NUMINAMATH_CALUDE_hundredth_count_is_twelve_l532_53230

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle :=
  { n : ℕ // n > 0 ∧ n ≤ 12 }

/-- The label assigned to a stone after a certain number of counts. -/
def label (count : ℕ) : StoneCircle → ℕ :=
  sorry

/-- The original stone number corresponding to a given label. -/
def originalStone (label : ℕ) : StoneCircle :=
  sorry

/-- Theorem stating that the 100th count corresponds to the original stone number 12. -/
theorem hundredth_count_is_twelve :
  originalStone 100 = ⟨12, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_hundredth_count_is_twelve_l532_53230


namespace NUMINAMATH_CALUDE_distance_after_12_hours_l532_53243

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: Two people walking in opposite directions for 12 hours
    at speeds of 7 km/hr and 3 km/hr will be 120 km apart -/
theorem distance_after_12_hours :
  distance_between 7 3 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_12_hours_l532_53243


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l532_53295

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x*y = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), z = x + 2*y → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l532_53295


namespace NUMINAMATH_CALUDE_max_min_f_l532_53216

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def a : ℝ := 0
def b : ℝ := 3

-- Theorem statement
theorem max_min_f :
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≤ 5) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≥ -15) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = 5) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = -15) :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_l532_53216


namespace NUMINAMATH_CALUDE_sequence_term_correct_l532_53225

def sequence_sum (n : ℕ) : ℕ := 3 + 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 →
  sequence_sum n - sequence_sum (n-1) = sequence_term n :=
sorry

end NUMINAMATH_CALUDE_sequence_term_correct_l532_53225


namespace NUMINAMATH_CALUDE_negation_equivalence_l532_53231

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l532_53231


namespace NUMINAMATH_CALUDE_exponent_multiplication_l532_53273

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l532_53273


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l532_53246

/-- The total cost of circus tickets for a group of kids and adults -/
def total_ticket_cost (num_kids : ℕ) (num_adults : ℕ) (kid_ticket_price : ℚ) : ℚ :=
  let adult_ticket_price := 2 * kid_ticket_price
  num_kids * kid_ticket_price + num_adults * adult_ticket_price

/-- Theorem stating the total cost of circus tickets for a specific group -/
theorem circus_ticket_cost :
  total_ticket_cost 6 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l532_53246


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l532_53257

def line_equation (x y : ℝ) : Prop :=
  2 * (x - 3) + (-1) * (y - (-4)) = 6

theorem line_equation_equivalence :
  ∀ x y : ℝ, line_equation x y ↔ y = 2 * x - 16 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l532_53257


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l532_53298

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l532_53298


namespace NUMINAMATH_CALUDE_unknown_number_value_l532_53277

theorem unknown_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l532_53277


namespace NUMINAMATH_CALUDE_doughnuts_served_l532_53288

theorem doughnuts_served (staff : ℕ) (doughnuts_per_staff : ℕ) (doughnuts_left : ℕ) : 
  staff = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  staff * doughnuts_per_staff + doughnuts_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_served_l532_53288


namespace NUMINAMATH_CALUDE_amy_book_count_l532_53224

theorem amy_book_count (maddie_books luisa_books : ℕ) 
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : luisa_books + amy_books = maddie_books + 9) : 
  amy_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_book_count_l532_53224


namespace NUMINAMATH_CALUDE_factorial_product_not_square_l532_53269

theorem factorial_product_not_square (n : ℕ) : 
  ∃ (m : ℕ), (n.factorial ^ 2 * (n + 1).factorial * (2 * n + 9).factorial * (2 * n + 10).factorial) ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_not_square_l532_53269


namespace NUMINAMATH_CALUDE_even_times_odd_is_odd_l532_53275

variable (f g : ℝ → ℝ)

-- Define even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem even_times_odd_is_odd (hf : IsEven f) (hg : IsOdd g) : IsOdd (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_even_times_odd_is_odd_l532_53275


namespace NUMINAMATH_CALUDE_water_tower_theorem_l532_53286

def water_tower_problem (total_capacity : ℕ) (first_neighborhood : ℕ) : Prop :=
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let used_water := first_neighborhood + second_neighborhood + third_neighborhood
  total_capacity - used_water = 350

theorem water_tower_theorem : water_tower_problem 1200 150 := by
  sorry

end NUMINAMATH_CALUDE_water_tower_theorem_l532_53286


namespace NUMINAMATH_CALUDE_product_of_square_roots_l532_53292

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (25 * p^2) * Real.sqrt (2 * p^5) = 25 * p^5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l532_53292


namespace NUMINAMATH_CALUDE_unique_solution_system_l532_53282

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 3 * x + 4) ∧ 
  (2 * y + 3 + z = 6 * y + 6) ∧ 
  (3 * z + 3 + x = 9 * z + 8) ↔ 
  (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l532_53282


namespace NUMINAMATH_CALUDE_fifth_place_votes_l532_53279

theorem fifth_place_votes (total_votes : ℕ) (num_candidates : ℕ) 
  (diff1 diff2 diff3 diff4 : ℕ) :
  total_votes = 3567 →
  num_candidates = 5 →
  diff1 = 143 →
  diff2 = 273 →
  diff3 = 329 →
  diff4 = 503 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - diff1) + (winner_votes - diff2) + 
    (winner_votes - diff3) + (winner_votes - diff4) = total_votes ∧
    winner_votes - diff4 = 700 :=
by sorry

end NUMINAMATH_CALUDE_fifth_place_votes_l532_53279


namespace NUMINAMATH_CALUDE_math_book_cost_l532_53244

/-- Proves that the cost of each math book is $4 given the conditions of the book purchase problem. -/
theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_cost = 390 →
  math_books = 10 →
  (total_books - math_books) * history_book_cost + math_books * 4 = total_cost :=
by
  sorry

#check math_book_cost

end NUMINAMATH_CALUDE_math_book_cost_l532_53244


namespace NUMINAMATH_CALUDE_geometric_sequence_max_point_l532_53212

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and (b, c) being the coordinates of the maximum point of the curve y = 3x - x^3,
    prove that ad = 2. -/
theorem geometric_sequence_max_point (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →  -- maximum point condition
  c = 3 * b - b^3 →  -- y-coordinate of maximum point
  a * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_point_l532_53212


namespace NUMINAMATH_CALUDE_students_in_row_l532_53241

theorem students_in_row (S R : ℕ) : 
  S = 5 * R + 6 →
  S = 6 * (R - 3) →
  6 = S / R - 18 := by
sorry

end NUMINAMATH_CALUDE_students_in_row_l532_53241


namespace NUMINAMATH_CALUDE_carols_blocks_l532_53284

/-- Carol's block problem -/
theorem carols_blocks (initial_blocks lost_blocks : ℕ) :
  initial_blocks = 42 →
  lost_blocks = 25 →
  initial_blocks - lost_blocks = 17 :=
by sorry

end NUMINAMATH_CALUDE_carols_blocks_l532_53284


namespace NUMINAMATH_CALUDE_elastic_collision_mass_and_velocity_ratios_l532_53220

/-- Represents the masses and velocities in an elastic collision -/
structure CollisionSystem where
  m₁ : ℝ
  m₂ : ℝ
  v₀ : ℝ
  v₁ : ℝ
  v₂ : ℝ

/-- Conditions for the elastic collision system -/
def ElasticCollision (s : CollisionSystem) : Prop :=
  s.m₁ > 0 ∧ s.m₂ > 0 ∧ s.v₀ > 0 ∧ s.v₁ > 0 ∧ s.v₂ > 0 ∧
  s.v₂ = 4 * s.v₁ ∧
  s.m₁ * s.v₀ = s.m₁ * s.v₁ + s.m₂ * s.v₂ ∧
  s.m₁ * s.v₀^2 = s.m₁ * s.v₁^2 + s.m₂ * s.v₂^2

theorem elastic_collision_mass_and_velocity_ratios (s : CollisionSystem) 
  (h : ElasticCollision s) : s.m₂ / s.m₁ = 1/2 ∧ s.v₀ / s.v₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_elastic_collision_mass_and_velocity_ratios_l532_53220


namespace NUMINAMATH_CALUDE_flag_arrangements_count_l532_53232

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  /- Number of ways to choose 11 positions out of 13 for red flags -/
  let red_positions := Nat.choose 13 11
  /- Number of ways to place the divider between flagpoles -/
  let divider_positions := 13
  /- Total number of arrangements -/
  let total_arrangements := red_positions * divider_positions
  /- Number of invalid arrangements (where one pole gets no flag) -/
  let invalid_arrangements := 2 * red_positions
  /- Final number of valid arrangements -/
  total_arrangements - invalid_arrangements

/-- Theorem stating that M is equal to 858 -/
theorem flag_arrangements_count : M = 858 := by sorry

end NUMINAMATH_CALUDE_flag_arrangements_count_l532_53232


namespace NUMINAMATH_CALUDE_arrangement_remainder_l532_53254

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of red marbles that satisfies the condition -/
def max_red_marbles : ℕ := 16

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def num_arrangements : ℕ := Nat.choose (green_marbles + max_red_marbles) green_marbles

/-- The theorem to be proved -/
theorem arrangement_remainder : num_arrangements % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l532_53254


namespace NUMINAMATH_CALUDE_vinnie_tips_l532_53270

theorem vinnie_tips (paul_tips : ℕ) (vinnie_more : ℕ) : 
  paul_tips = 14 → vinnie_more = 16 → paul_tips + vinnie_more = 30 := by
  sorry

end NUMINAMATH_CALUDE_vinnie_tips_l532_53270


namespace NUMINAMATH_CALUDE_marbles_given_to_joan_l532_53209

theorem marbles_given_to_joan (initial_marbles : ℝ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 9.0) 
  (h2 : remaining_marbles = 6) :
  initial_marbles - remaining_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_joan_l532_53209


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l532_53229

/-- A point in the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid dimensions --/
def gridWidth : Nat := 4
def gridHeight : Nat := 5

/-- The initially colored squares --/
def initialColoredSquares : List Point := [
  { x := 1, y := 4 },
  { x := 2, y := 1 },
  { x := 4, y := 2 }
]

/-- A function to check if a point is within the grid --/
def isInGrid (p : Point) : Prop :=
  1 ≤ p.x ∧ p.x ≤ gridWidth ∧ 1 ≤ p.y ∧ p.y ≤ gridHeight

/-- A function to check if two points are symmetrical about the vertical line --/
def isVerticallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y = p2.y

/-- A function to check if two points are symmetrical about the horizontal line --/
def isHorizontallySymmetric (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y + p2.y = gridHeight + 1

/-- A function to check if two points are rotationally symmetric --/
def isRotationallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y + p2.y = gridHeight + 1

/-- The main theorem --/
theorem min_additional_squares_for_symmetry :
  ∃ (additionalSquares : List Point),
    (∀ p ∈ additionalSquares, isInGrid p) ∧
    (∀ p ∈ initialColoredSquares ++ additionalSquares,
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isVerticallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isHorizontallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isRotationallySymmetric p q)) ∧
    additionalSquares.length = 9 ∧
    (∀ (otherSquares : List Point),
      (∀ p ∈ otherSquares, isInGrid p) →
      (∀ p ∈ initialColoredSquares ++ otherSquares,
        (∃ q ∈ initialColoredSquares ++ otherSquares, isVerticallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isHorizontallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isRotationallySymmetric p q)) →
      otherSquares.length ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l532_53229


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_mean_l532_53201

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def point_on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Theorem: Two different regression lines for the same dataset intersect at the sample mean -/
theorem regression_lines_intersect_at_mean 
  (l₁ l₂ : RegressionLine) 
  (x_mean y_mean : ℝ) 
  (h_different : l₁ ≠ l₂) 
  (h_on_l₁ : point_on_line l₁ x_mean y_mean)
  (h_on_l₂ : point_on_line l₂ x_mean y_mean) : 
  ∃ (x y : ℝ), x = x_mean ∧ y = y_mean ∧ point_on_line l₁ x y ∧ point_on_line l₂ x y :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_at_mean_l532_53201


namespace NUMINAMATH_CALUDE_sequence_sum_product_l532_53245

def sequence_property (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

theorem sequence_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : sequence_property α β γ a b) :
  ∀ m n, a (m + n) + b (m + n) = a m * a n + b m * b n :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_product_l532_53245


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l532_53215

theorem quadratic_equation_1 (x : ℝ) : x^2 + 16 = 8*x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l532_53215


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l532_53234

/-- The number of ways to arrange n unique objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two groups of books, where each group stays together --/
def arrange_book_groups : ℕ := permutations 2

/-- The number of ways to arrange 4 unique math books within their group --/
def arrange_math_books : ℕ := permutations 4

/-- The number of ways to arrange 4 unique English books within their group --/
def arrange_english_books : ℕ := permutations 4

/-- The total number of ways to arrange 4 unique math books and 4 unique English books on a shelf,
    with all math books staying together and all English books staying together --/
def total_arrangements : ℕ := arrange_book_groups * arrange_math_books * arrange_english_books

theorem book_arrangement_theorem : total_arrangements = 1152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l532_53234


namespace NUMINAMATH_CALUDE_savings_double_l532_53240

/-- Represents the financial situation of a man over two years -/
structure FinancialSituation where
  first_year_income : ℝ
  first_year_savings_rate : ℝ
  income_increase_rate : ℝ
  expenditure_ratio : ℝ

/-- Calculates the percentage increase in savings -/
def savings_increase_percentage (fs : FinancialSituation) : ℝ :=
  -- The actual calculation will be implemented in the proof
  sorry

/-- Theorem stating that the savings increase by 100% -/
theorem savings_double (fs : FinancialSituation) 
  (h1 : fs.first_year_savings_rate = 0.35)
  (h2 : fs.income_increase_rate = 0.35)
  (h3 : fs.expenditure_ratio = 2)
  : savings_increase_percentage fs = 100 := by
  sorry

end NUMINAMATH_CALUDE_savings_double_l532_53240


namespace NUMINAMATH_CALUDE_total_people_in_program_l532_53283

theorem total_people_in_program (parents : ℕ) (pupils : ℕ) 
  (h1 : parents = 22) (h2 : pupils = 654) : 
  parents + pupils = 676 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l532_53283


namespace NUMINAMATH_CALUDE_equation_solution_l532_53226

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 105) / 24 ∧ 
     x₂ = (3 - Real.sqrt 105) / 24) ∧ 
    (∀ x : ℝ, 4 * (3 * x)^2 + 2 * (3 * x) + 7 = 3 * (8 * x^2 + 3 * x + 3) ↔ 
      x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l532_53226


namespace NUMINAMATH_CALUDE_max_advancing_players_16_10_l532_53297

/-- Represents a chess tournament -/
structure ChessTournament where
  players : ℕ
  points_to_advance : ℕ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.players * (t.players - 1) / 2

/-- Calculates the total points awarded in the tournament -/
def total_points (t : ChessTournament) : ℕ :=
  total_games t

/-- Defines the maximum number of players that can advance -/
def max_advancing_players (t : ChessTournament) : ℕ :=
  11

/-- Theorem: In a 16-player tournament where players need at least 10 points to advance,
    the maximum number of players who can advance is 11 -/
theorem max_advancing_players_16_10 :
  ∀ t : ChessTournament,
    t.players = 16 →
    t.points_to_advance = 10 →
    max_advancing_players t = 11 :=
by sorry


end NUMINAMATH_CALUDE_max_advancing_players_16_10_l532_53297


namespace NUMINAMATH_CALUDE_system_solution_l532_53203

theorem system_solution : ∃ (x y : ℝ), (3 * x = -9 - 3 * y) ∧ (2 * x = 3 * y - 22) := by
  use -5, 2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l532_53203


namespace NUMINAMATH_CALUDE_student_attendance_probability_l532_53213

theorem student_attendance_probability :
  let p_absent : ℝ := 1 / 20
  let p_present : ℝ := 1 - p_absent
  let p_one_absent_one_present : ℝ := p_absent * p_present + p_present * p_absent
  p_one_absent_one_present = 0.095 := by
  sorry

end NUMINAMATH_CALUDE_student_attendance_probability_l532_53213


namespace NUMINAMATH_CALUDE_inequality_proof_l532_53293

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≥ a₂) (h2 : a₂ ≥ a₃) (h3 : a₃ > 0)
  (h4 : b₁ ≥ b₂) (h5 : b₂ ≥ b₃) (h6 : b₃ > 0)
  (h7 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (h8 : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l532_53293


namespace NUMINAMATH_CALUDE_polygon_sides_count_l532_53296

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l532_53296
