import Mathlib

namespace NUMINAMATH_CALUDE_exhibition_average_l2744_274483

theorem exhibition_average : 
  let works : List ℕ := [58, 52, 58, 60]
  (works.sum / works.length : ℚ) = 57 := by sorry

end NUMINAMATH_CALUDE_exhibition_average_l2744_274483


namespace NUMINAMATH_CALUDE_fruit_distribution_ways_l2744_274454

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of fruits to buy -/
def total_fruits : ℕ := 17

/-- The number of types of fruit -/
def fruit_types : ℕ := 5

/-- The number of fruits remaining after placing one in each type -/
def remaining_fruits : ℕ := total_fruits - fruit_types

theorem fruit_distribution_ways :
  distribute remaining_fruits fruit_types = 1820 :=
sorry

end NUMINAMATH_CALUDE_fruit_distribution_ways_l2744_274454


namespace NUMINAMATH_CALUDE_triangle_side_length_l2744_274471

theorem triangle_side_length (BC AC : ℝ) (A : ℝ) :
  BC = Real.sqrt 7 →
  AC = 2 * Real.sqrt 3 →
  A = π / 6 →
  ∃ AB : ℝ, (AB = 5 ∨ AB = 1) ∧
    AB^2 + AC^2 - BC^2 = 2 * AB * AC * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2744_274471


namespace NUMINAMATH_CALUDE_power_of_256_l2744_274411

theorem power_of_256 : (256 : ℝ) ^ (5/4 : ℝ) = 1024 :=
by
  have h : 256 = 2^8 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_256_l2744_274411


namespace NUMINAMATH_CALUDE_isabels_bouquets_l2744_274408

theorem isabels_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  initial_flowers = 66 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  (initial_flowers - wilted_flowers) / flowers_per_bouquet = 7 :=
by sorry

end NUMINAMATH_CALUDE_isabels_bouquets_l2744_274408


namespace NUMINAMATH_CALUDE_lesser_fraction_l2744_274414

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 9/10)
  (prod_eq : x * y = 1/15) :
  min x y = 1/5 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2744_274414


namespace NUMINAMATH_CALUDE_inner_cube_surface_area_l2744_274421

/-- Given a cube with surface area 54 square meters containing an inscribed sphere,
    which in turn contains an inscribed cube, the surface area of the inner cube
    is 18 square meters. -/
theorem inner_cube_surface_area (outer_cube : Real) (sphere : Real) (inner_cube : Real) :
  outer_cube = 54 →  -- Surface area of outer cube
  sphere ^ 2 = 3 * inner_cube ^ 2 →  -- Relation between sphere and inner cube
  inner_cube ^ 2 = 3 →  -- Side length of inner cube
  6 * inner_cube ^ 2 = 18  -- Surface area of inner cube
  := by sorry

end NUMINAMATH_CALUDE_inner_cube_surface_area_l2744_274421


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2744_274425

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z) = 140)
  (h3 : x^2 + y^2 + z^2 = 21^2) :
  2 * (x*y + x*z + y*z) = 784 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2744_274425


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l2744_274415

/-- Represents the price reduction model for a sportswear item -/
def PriceReductionModel (initial_price final_price : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 < x ∧ 
    x < 1 ∧ 
    initial_price * (1 - x)^2 = final_price

/-- Theorem stating that the given equation correctly models the price reduction -/
theorem correct_price_reduction_equation :
  PriceReductionModel 560 315 :=
sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l2744_274415


namespace NUMINAMATH_CALUDE_divisor_problem_l2744_274410

theorem divisor_problem (x : ℕ) : x > 0 ∧ 83 = 9 * x + 2 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2744_274410


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2744_274495

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2744_274495


namespace NUMINAMATH_CALUDE_odd_integers_square_l2744_274474

theorem odd_integers_square (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : ∃ k : ℕ, a^b * b^a = k^2) :
  ∃ m : ℕ, a * b = m^2 := by
sorry

end NUMINAMATH_CALUDE_odd_integers_square_l2744_274474


namespace NUMINAMATH_CALUDE_statement_falsity_l2744_274456

theorem statement_falsity (x : ℝ) : x = -4 ∨ x = -2 → x ∈ Set.Iio 2 ∧ ¬(x^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_statement_falsity_l2744_274456


namespace NUMINAMATH_CALUDE_bunny_count_l2744_274485

/-- The number of bunnies coming out of their burrows -/
def num_bunnies : ℕ := 
  let times_per_minute : ℕ := 3
  let hours : ℕ := 10
  let minutes_per_hour : ℕ := 60
  let total_times : ℕ := 36000
  total_times / (times_per_minute * hours * minutes_per_hour)

theorem bunny_count : num_bunnies = 20 := by
  sorry

end NUMINAMATH_CALUDE_bunny_count_l2744_274485


namespace NUMINAMATH_CALUDE_floor_sqrt_26_squared_l2744_274422

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_26_squared_l2744_274422


namespace NUMINAMATH_CALUDE_barrel_tank_ratio_l2744_274491

theorem barrel_tank_ratio : 
  ∀ (barrel_volume tank_volume : ℝ),
  barrel_volume > 0 → tank_volume > 0 →
  (3/4 : ℝ) * barrel_volume = (5/8 : ℝ) * tank_volume →
  barrel_volume / tank_volume = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_barrel_tank_ratio_l2744_274491


namespace NUMINAMATH_CALUDE_income_calculation_l2744_274463

def original_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def differential_savings : ℝ := 5040

theorem income_calculation (income : ℝ) :
  (original_tax_rate - new_tax_rate) * income = differential_savings →
  income = 36000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2744_274463


namespace NUMINAMATH_CALUDE_descending_order_proof_l2744_274451

theorem descending_order_proof :
  (1909 > 1100 ∧ 1100 > 1090 ∧ 1090 > 1009) ∧
  (10000 > 9999 ∧ 9999 > 9990 ∧ 9990 > 8909 ∧ 8909 > 8900) := by
  sorry

end NUMINAMATH_CALUDE_descending_order_proof_l2744_274451


namespace NUMINAMATH_CALUDE_spinner_probability_l2744_274409

/- Define an isosceles triangle with the given angle property -/
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle = 2 * vertexAngle

/- Define the division of the triangle into regions by altitudes -/
def triangleRegions : ℕ := 6

/- Define the number of shaded regions -/
def shadedRegions : ℕ := 4

/- Define the probability of landing in a shaded region -/
def shadedProbability (t : IsoscelesTriangle) : ℚ :=
  shadedRegions / triangleRegions

/- Theorem statement -/
theorem spinner_probability (t : IsoscelesTriangle) :
  shadedProbability t = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2744_274409


namespace NUMINAMATH_CALUDE_percentage_problem_l2744_274497

/-- Given that (P/100 * 1265) / 7 = 271.07142857142856, prove that P = 150 -/
theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 7 = 271.07142857142856 → P = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2744_274497


namespace NUMINAMATH_CALUDE_buses_in_five_days_l2744_274438

/-- Represents the number of buses leaving a station over multiple days -/
def buses_over_days (buses_per_half_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  buses_per_half_hour * 2 * hours_per_day * days

/-- Theorem stating that 120 buses leave the station over 5 days -/
theorem buses_in_five_days :
  buses_over_days 1 12 5 = 120 := by
  sorry

#eval buses_over_days 1 12 5

end NUMINAMATH_CALUDE_buses_in_five_days_l2744_274438


namespace NUMINAMATH_CALUDE_no_four_integers_l2744_274441

theorem no_four_integers (n : ℕ) (hn : n ≥ 1) :
  ¬ ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n^2 ≤ a ∧ a < (n+1)^2 ∧
    n^2 ≤ b ∧ b < (n+1)^2 ∧
    n^2 ≤ c ∧ c < (n+1)^2 ∧
    n^2 ≤ d ∧ d < (n+1)^2 ∧
    a * d = b * c :=
by sorry

end NUMINAMATH_CALUDE_no_four_integers_l2744_274441


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l2744_274475

theorem fourth_term_of_geometric_progression :
  let a₁ : ℝ := Real.sqrt 4
  let a₂ : ℝ := (4 : ℝ) ^ (1/4)
  let a₃ : ℝ := (4 : ℝ) ^ (1/8)
  let r : ℝ := a₂ / a₁
  let a₄ : ℝ := a₃ * r
  a₄ = (1/4 : ℝ) ^ (1/8) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l2744_274475


namespace NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l2744_274450

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance function
def verticalDistance (x : ℝ) := f x - g x

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, verticalDistance x = 0 ∧ ∀ y : ℝ, verticalDistance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l2744_274450


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2744_274433

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2744_274433


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l2744_274478

-- Define a type for colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Check if all vertices of a triangle have the same color
def monochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- Main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ monochromatic t coloring := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l2744_274478


namespace NUMINAMATH_CALUDE_school_distance_is_two_point_five_l2744_274467

/-- The distance from Philip's house to the school in miles -/
def school_distance : ℝ := sorry

/-- The round trip distance to the market in miles -/
def market_round_trip : ℝ := 4

/-- The number of round trips to school per week -/
def school_trips_per_week : ℕ := 8

/-- The number of round trips to the market per week -/
def market_trips_per_week : ℕ := 1

/-- The total mileage for a typical week in miles -/
def total_weekly_mileage : ℝ := 44

theorem school_distance_is_two_point_five : 
  school_distance = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_is_two_point_five_l2744_274467


namespace NUMINAMATH_CALUDE_red_numbers_structure_l2744_274479

-- Define the color type
inductive Color
| White
| Red

-- Define the coloring function
def coloring : ℕ → Color := sorry

-- Define properties of the coloring
axiom exists_white : ∃ n : ℕ, coloring n = Color.White
axiom exists_red : ∃ n : ℕ, coloring n = Color.Red
axiom sum_white_red_is_white :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w + r) = Color.White
axiom product_white_red_is_red :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w * r) = Color.Red

-- Define the set of red numbers
def RedNumbers : Set ℕ := {n : ℕ | coloring n = Color.Red}

-- State the theorem
theorem red_numbers_structure :
  ∃ r₀ : ℕ, r₀ > 0 ∧ r₀ ∈ RedNumbers ∧
  ∀ n : ℕ, n ∈ RedNumbers ↔ ∃ k : ℕ, n = k * r₀ :=
sorry

end NUMINAMATH_CALUDE_red_numbers_structure_l2744_274479


namespace NUMINAMATH_CALUDE_sturgeon_books_problem_l2744_274465

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_price hardcover_price total_cost : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 15)
  (hh : hardcover_price = 25)
  (ht : total_cost = 240) :
  ∃ (hardcovers : ℕ), 
    hardcovers * hardcover_price + (total_volumes - hardcovers) * paperback_price = total_cost ∧ 
    hardcovers = 6 := by
  sorry

end NUMINAMATH_CALUDE_sturgeon_books_problem_l2744_274465


namespace NUMINAMATH_CALUDE_xyz_value_l2744_274455

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h3 : x + y + z = 3) : 
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2744_274455


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l2744_274406

theorem least_positive_linear_combination (x y : ℤ) : 
  ∃ (a b : ℤ), 24 * a + 18 * b = 6 ∧ 
  ∀ (c d : ℤ), 24 * c + 18 * d > 0 → 24 * c + 18 * d ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l2744_274406


namespace NUMINAMATH_CALUDE_sample_size_is_sampled_athletes_l2744_274481

/-- A structure representing a statistical study of athletes' ages -/
structure AthleteStudy where
  total_athletes : ℕ
  sampled_athletes : ℕ
  h_total : total_athletes = 1000
  h_sampled : sampled_athletes = 100

/-- The sample size of an athlete study is equal to the number of sampled athletes -/
theorem sample_size_is_sampled_athletes (study : AthleteStudy) : 
  study.sampled_athletes = 100 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_sampled_athletes_l2744_274481


namespace NUMINAMATH_CALUDE_drop_recording_l2744_274424

/-- Represents the change in water level in meters -/
def WaterLevelChange : Type := ℝ

/-- Records a rise in water level -/
def recordRise (meters : ℝ) : WaterLevelChange := meters

/-- Records a drop in water level -/
def recordDrop (meters : ℝ) : WaterLevelChange := -meters

/-- The theorem stating how a drop in water level should be recorded -/
theorem drop_recording (rise : ℝ) (drop : ℝ) :
  recordRise rise = rise → recordDrop drop = -drop :=
by sorry

end NUMINAMATH_CALUDE_drop_recording_l2744_274424


namespace NUMINAMATH_CALUDE_equation_solution_l2744_274423

theorem equation_solution :
  ∃ x : ℝ, (3639 + 11.95 - x^2 = 3054) ∧ (abs (x - 24.43) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2744_274423


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2744_274436

theorem inverse_proportionality (α β : ℚ) (h : α ≠ 0 ∧ β ≠ 0) :
  (∃ k : ℚ, k ≠ 0 ∧ α * β = k) →
  (α = -4 ∧ β = -8) →
  (β = 12 → α = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2744_274436


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2744_274439

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The condition p: f(x) = x³ + 2x² + mx + 1 is monotonically increasing -/
def p (m : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => x^3 + 2*x^2 + m*x + 1)

/-- The condition q: m ≥ 8x / (x² + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop :=
  ∀ x, x > 0 → m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2744_274439


namespace NUMINAMATH_CALUDE_min_value_theorem_l2744_274417

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → (x + y + 1) / (x * y) ≥ (a + b + 1) / (a * b)) →
  (a + b + 1) / (a * b) = 4 * Real.sqrt 3 + 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2744_274417


namespace NUMINAMATH_CALUDE_expression_simplification_inequality_system_equivalence_l2744_274469

-- Part 1
theorem expression_simplification (a : ℝ) :
  (a - 3)^2 + a*(4 - a) = -2*a + 9 := by sorry

-- Part 2
theorem inequality_system_equivalence (x : ℝ) :
  -2 ≤ x ∧ x < 3 ↔ 3*x - 5 < x + 1 ∧ 2*(2*x - 1) ≥ 3*x - 4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_inequality_system_equivalence_l2744_274469


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2744_274458

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2744_274458


namespace NUMINAMATH_CALUDE_curve_is_ellipse_iff_k_in_range_l2744_274472

/-- The curve equation: x^2 / (4 + k) + y^2 / (1 - k) = 1 -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 + k) + y^2 / (1 - k) = 1

/-- The range of k values for which the curve represents an ellipse -/
def ellipse_k_range (k : ℝ) : Prop :=
  (k > -4 ∧ k < -3/2) ∨ (k > -3/2 ∧ k < 1)

/-- Theorem stating that the curve represents an ellipse if and only if k is in the specified range -/
theorem curve_is_ellipse_iff_k_in_range :
  ∀ k : ℝ, (∃ x y : ℝ, curve_equation x y k) ↔ ellipse_k_range k :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_iff_k_in_range_l2744_274472


namespace NUMINAMATH_CALUDE_roberto_healthcare_contribution_l2744_274448

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * contribution_rate

/-- Proves that Roberto's healthcare contribution is 50 cents per hour. -/
theorem roberto_healthcare_contribution :
  healthcare_contribution 25 (2/100) = 50 := by
  sorry

#eval healthcare_contribution 25 (2/100)

end NUMINAMATH_CALUDE_roberto_healthcare_contribution_l2744_274448


namespace NUMINAMATH_CALUDE_at_least_100_triangles_l2744_274453

/-- Represents a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Counts the number of triangular regions formed by a set of lines -/
def count_triangles (lines : LineSet) : ℕ := sorry

/-- Theorem: 300 lines with given conditions form at least 100 triangles -/
theorem at_least_100_triangles (lines : LineSet) 
  (h1 : lines.num_lines = 300)
  (h2 : lines.no_parallel = true)
  (h3 : lines.no_triple_intersection = true) :
  count_triangles lines ≥ 100 := by sorry

end NUMINAMATH_CALUDE_at_least_100_triangles_l2744_274453


namespace NUMINAMATH_CALUDE_socks_theorem_l2744_274437

/-- The number of pairs of socks Niko bought -/
def total_socks : ℕ := 9

/-- The cost of each pair of socks in dollars -/
def cost_per_pair : ℚ := 2

/-- The number of pairs with 25% profit -/
def pairs_with_25_percent : ℕ := 4

/-- The number of pairs with $0.2 profit -/
def pairs_with_20_cents : ℕ := 5

/-- The total profit in dollars -/
def total_profit : ℚ := 3

/-- The profit percentage for the first group of socks -/
def profit_percentage : ℚ := 25 / 100

/-- The profit amount for the second group of socks in dollars -/
def profit_amount : ℚ := 1 / 5

theorem socks_theorem :
  total_socks = pairs_with_25_percent + pairs_with_20_cents ∧
  total_profit = pairs_with_25_percent * (cost_per_pair * profit_percentage) +
                 pairs_with_20_cents * profit_amount :=
by sorry

end NUMINAMATH_CALUDE_socks_theorem_l2744_274437


namespace NUMINAMATH_CALUDE_number_of_workers_l2744_274482

theorem number_of_workers (total_contribution : ℕ) (extra_contribution : ℕ) (new_total : ℕ) : 
  total_contribution = 300000 →
  extra_contribution = 50 →
  new_total = 320000 →
  ∃ (workers : ℕ), 
    workers * (total_contribution / workers + extra_contribution) = new_total ∧
    workers = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_of_workers_l2744_274482


namespace NUMINAMATH_CALUDE_submarine_hit_guaranteed_l2744_274460

/-- Represents the position of a submarine at time t -/
def submarinePosition (v : ℕ+) (t : ℕ) : ℕ := v.val * t

/-- Represents the position of a missile fired at time n -/
def missilePosition (n : ℕ) : ℕ := n ^ 2

/-- Theorem stating that there exists a firing sequence that will hit the submarine -/
theorem submarine_hit_guaranteed :
  ∀ (v : ℕ+), ∃ (t : ℕ), submarinePosition v t = missilePosition t := by
  sorry


end NUMINAMATH_CALUDE_submarine_hit_guaranteed_l2744_274460


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l2744_274413

/-- The total cost to replace a floor given the room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l2744_274413


namespace NUMINAMATH_CALUDE_room_width_calculation_l2744_274464

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 950)
  (h3 : total_cost = 20900) :
  total_cost / cost_per_sqm / length = 4 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l2744_274464


namespace NUMINAMATH_CALUDE_max_expected_score_l2744_274412

/-- Xiao Zhang's box configuration -/
structure BoxConfig where
  red : ℕ
  yellow : ℕ
  white : ℕ
  sum_six : red + yellow + white = 6

/-- Expected score for a given box configuration -/
def expectedScore (config : BoxConfig) : ℚ :=
  (3 * config.red + 4 * config.yellow + 3 * config.white) / 36

/-- Theorem stating the maximum expected score and optimal configuration -/
theorem max_expected_score :
  ∃ (config : BoxConfig),
    expectedScore config = 2/3 ∧
    ∀ (other : BoxConfig), expectedScore other ≤ expectedScore config ∧
    config.red = 0 ∧ config.yellow = 6 ∧ config.white = 0 := by
  sorry


end NUMINAMATH_CALUDE_max_expected_score_l2744_274412


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2744_274444

/-- A polynomial with integer coefficients -/
def IntPolynomial : Type := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def evaluate (f : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is divisible by another if their remainder is zero -/
def divisible (a b : ℤ) : Prop := a % b = 0

theorem polynomial_divisibility (f : IntPolynomial) :
  divisible (evaluate f 2) 6 →
  divisible (evaluate f 3) 6 →
  divisible (evaluate f 5) 6 :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2744_274444


namespace NUMINAMATH_CALUDE_change_received_correct_l2744_274496

/-- Calculates the change received when buying steak -/
def change_received (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) : ℝ :=
  amount_paid - (cost_per_pound * pounds_bought)

/-- Theorem: The change received when buying steak is correct -/
theorem change_received_correct (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) :
  change_received cost_per_pound pounds_bought amount_paid =
  amount_paid - (cost_per_pound * pounds_bought) := by
  sorry

#eval change_received 7 2 20

end NUMINAMATH_CALUDE_change_received_correct_l2744_274496


namespace NUMINAMATH_CALUDE_ball_arrangement_theorem_l2744_274462

/-- The number of ways to arrange 8 balls in a row, with 5 red and 3 white,
    such that exactly 3 consecutive balls are red -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem ball_arrangement_theorem :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangement_theorem_l2744_274462


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2744_274499

theorem triangle_area_from_squares (a b : ℝ) (ha : a^2 = 25) (hb : b^2 = 144) : 
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2744_274499


namespace NUMINAMATH_CALUDE_equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l2744_274419

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem equal_sets_imply_a_eq_5 :
  ∀ a : ℝ, A a = B → a = 5 := by sorry

-- Theorem 2
theorem intersection_conditions_imply_a_eq_neg_2 :
  ∀ a : ℝ, (B ∩ A a ≠ ∅) ∧ (C ∩ A a = ∅) → a = -2 := by sorry

end NUMINAMATH_CALUDE_equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l2744_274419


namespace NUMINAMATH_CALUDE_change_calculation_l2744_274468

def initial_amount : ℕ := 20
def num_items : ℕ := 3
def cost_per_item : ℕ := 2

theorem change_calculation :
  initial_amount - (num_items * cost_per_item) = 14 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l2744_274468


namespace NUMINAMATH_CALUDE_log_equation_solution_l2744_274492

theorem log_equation_solution (s : ℝ) (h : s > 0) :
  (4 * Real.log s / Real.log 3 = Real.log (4 * s^2) / Real.log 3) → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2744_274492


namespace NUMINAMATH_CALUDE_junior_fraction_l2744_274461

/-- Represents the number of students in each category -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCounts) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = 120 ∧
  s.freshmen > 0 ∧ s.sophomores > 0 ∧ s.juniors > 0 ∧ s.seniors > 0 ∧
  s.freshmen = 2 * s.sophomores ∧
  s.juniors = 4 * s.seniors ∧
  (s.freshmen : ℚ) / 2 + (s.sophomores : ℚ) / 3 = (s.juniors : ℚ) * 2 / 3 - (s.seniors : ℚ) / 4

/-- The theorem to be proved -/
theorem junior_fraction (s : StudentCounts) (h : satisfiesConditions s) :
    (s.juniors : ℚ) / (s.freshmen + s.sophomores + s.juniors + s.seniors) = 32 / 167 := by
  sorry

end NUMINAMATH_CALUDE_junior_fraction_l2744_274461


namespace NUMINAMATH_CALUDE_brownies_remaining_l2744_274420

/-- Calculates the number of brownies left after consumption -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Proves that given the specific consumption pattern, 5 brownies are left -/
theorem brownies_remaining :
  brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

#eval brownies_left 24 2 5 1 5 4

end NUMINAMATH_CALUDE_brownies_remaining_l2744_274420


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2744_274489

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b)) →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2744_274489


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2744_274429

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  ∃ (d : ℝ), a = b - d ∧ c = b + d →  -- Terms form an arithmetic sequence
  a * b * c = 125 →  -- Product is 125
  ∀ x : ℝ, (x > 0 ∧ 
    (∃ (y z d : ℝ), y > 0 ∧ z > 0 ∧ 
      y = x - d ∧ z = x + d ∧ 
      y * x * z = 125)) → 
    x ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2744_274429


namespace NUMINAMATH_CALUDE_vitamin_c_content_l2744_274447

/-- The amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℕ := 103

/-- The total amount of vitamin C (in mg) in one 8-oz glass each of apple juice and orange juice -/
def total_vc : ℕ := 185

/-- The amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℕ := total_vc - apple_juice_vc

/-- Theorem: Two 8-oz glasses of apple juice and three 8-oz glasses of orange juice contain 452 mg of vitamin C -/
theorem vitamin_c_content : 2 * apple_juice_vc + 3 * orange_juice_vc = 452 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_c_content_l2744_274447


namespace NUMINAMATH_CALUDE_problem_solution_l2744_274480

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^2) / 7 = 75/7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2744_274480


namespace NUMINAMATH_CALUDE_triangle_property_l2744_274403

theorem triangle_property (A B C : Real) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos A = Real.sin B * Real.cos B) :
  (A = B ∨ C = Real.pi / 2) ∨ (B = C ∨ A = Real.pi / 2) ∨ (C = A ∨ B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l2744_274403


namespace NUMINAMATH_CALUDE_temperature_conversion_l2744_274487

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 221 → t = 105 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2744_274487


namespace NUMINAMATH_CALUDE_shelbys_driving_time_l2744_274434

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_no_rain speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_no_rain = 40)
  (h2 : speed_rain = 25)
  (h3 : total_time = 3)
  (h4 : total_distance = 85) :
  let rain_time := (total_distance - speed_no_rain * total_time) / (speed_rain - speed_no_rain)
  rain_time * 60 = 140 := by sorry

end NUMINAMATH_CALUDE_shelbys_driving_time_l2744_274434


namespace NUMINAMATH_CALUDE_cube_root_of_3x_plus_4y_is_3_l2744_274486

theorem cube_root_of_3x_plus_4y_is_3 (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 3) :
  (3 * x + 4 * y) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_3x_plus_4y_is_3_l2744_274486


namespace NUMINAMATH_CALUDE_largest_non_sum_42multiple_composite_l2744_274418

def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_sum_of_42multiple_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 42 * a + b ∧ a > 0 ∧ is_composite b

theorem largest_non_sum_42multiple_composite :
  (∀ n : ℕ, n > 215 → is_sum_of_42multiple_and_composite n) ∧
  ¬is_sum_of_42multiple_and_composite 215 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_42multiple_composite_l2744_274418


namespace NUMINAMATH_CALUDE_complex_number_computations_l2744_274427

theorem complex_number_computations :
  let z₁ : ℂ := 1 + 2*I
  let z₂ : ℂ := (1 + I) / (1 - I)
  let z₃ : ℂ := (Real.sqrt 2 + Real.sqrt 3 * I) / (Real.sqrt 3 - Real.sqrt 2 * I)
  (z₁^2 = -3 + 4*I) ∧
  (z₂^6 + z₃ = -1 + Real.sqrt 6 / 5 + ((Real.sqrt 3 + Real.sqrt 2) / 5) * I) := by
sorry

end NUMINAMATH_CALUDE_complex_number_computations_l2744_274427


namespace NUMINAMATH_CALUDE_robin_cupcake_ratio_l2744_274430

/-- Given that Robin ate 4 cupcakes with chocolate sauce and 12 cupcakes in total,
    prove that the ratio of cupcakes with buttercream frosting to cupcakes with chocolate sauce is 2:1 -/
theorem robin_cupcake_ratio :
  let chocolate_cupcakes : ℕ := 4
  let total_cupcakes : ℕ := 12
  let buttercream_cupcakes : ℕ := total_cupcakes - chocolate_cupcakes
  (buttercream_cupcakes : ℚ) / chocolate_cupcakes = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcake_ratio_l2744_274430


namespace NUMINAMATH_CALUDE_single_color_bound_l2744_274466

/-- A polygon on a checkered plane --/
structure CheckeredPolygon where
  /-- The area of the polygon --/
  area : ℕ
  /-- The perimeter of the polygon --/
  perimeter : ℕ

/-- The number of squares of a single color in the polygon --/
def singleColorCount (p : CheckeredPolygon) : ℕ := sorry

/-- Theorem: The number of squares of a single color is bounded --/
theorem single_color_bound (p : CheckeredPolygon) :
  singleColorCount p ≥ p.area / 2 - p.perimeter / 8 ∧
  singleColorCount p ≤ p.area / 2 + p.perimeter / 8 := by
  sorry

end NUMINAMATH_CALUDE_single_color_bound_l2744_274466


namespace NUMINAMATH_CALUDE_system_unique_solution_l2744_274490

/-- The system of equations has a unique solution (1, 2) -/
theorem system_unique_solution :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x - y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_system_unique_solution_l2744_274490


namespace NUMINAMATH_CALUDE_initial_ducks_l2744_274473

theorem initial_ducks (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 20 → total = 33 → initial + joined = total → initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l2744_274473


namespace NUMINAMATH_CALUDE_unique_triple_l2744_274432

theorem unique_triple : ∃! (a b c : ℤ), 
  a > 0 ∧ 0 > b ∧ b > c ∧ 
  a + b + c = 0 ∧ 
  ∃ (k : ℤ), 2017 - a^3*b - b^3*c - c^3*a = k^2 ∧
  a = 36 ∧ b = -12 ∧ c = -24 :=
sorry

end NUMINAMATH_CALUDE_unique_triple_l2744_274432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2744_274445

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  d : ℚ
  seq_def : ∀ n : ℕ+, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) 
  (h : ∀ n : ℕ+, (sum_n_terms a n) / (sum_n_terms b n) = (7 * n + 1) / (4 * n + 27)) :
  (a.a 7) / (b.a 7) = 92 / 79 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2744_274445


namespace NUMINAMATH_CALUDE_complex_rational_equation_root_l2744_274402

theorem complex_rational_equation_root :
  ∃! x : ℚ, (3*x^2 + 5)/(x-2) - (3*x + 10)/4 + (5 - 9*x)/(x-2) + 2 = 0 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_rational_equation_root_l2744_274402


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l2744_274426

/-- Hyperbola C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 2 - y^2 / 8 = 1

/-- Hyperbola C₂ -/
def C₂ (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptote of C₁ -/
def asymptote_C₁ (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Asymptote of C₂ -/
def asymptote_C₂ (x y a b : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_b_value (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_same_asymptotes : ∀ x y, asymptote_C₁ x y ↔ asymptote_C₂ x y a b)
  (h_focal_length : 4 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) :
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l2744_274426


namespace NUMINAMATH_CALUDE_population_is_all_scores_l2744_274477

/-- Represents a participant in the math test. -/
structure Participant where
  id : Nat
  score : ℝ

/-- Represents the entire set of participants in the math test. -/
def AllParticipants : Set Participant :=
  { p : Participant | p.id ≤ 1000 }

/-- Represents the sample of participants whose scores are analyzed. -/
def SampleParticipants : Set Participant :=
  { p : Participant | p.id ≤ 100 }

/-- The population in the context of this statistical analysis. -/
def Population : Set ℝ :=
  { score | ∃ p ∈ AllParticipants, p.score = score }

/-- Theorem stating that the population refers to the math scores of all 1000 participants. -/
theorem population_is_all_scores :
  Population = { score | ∃ p ∈ AllParticipants, p.score = score } :=
by sorry

end NUMINAMATH_CALUDE_population_is_all_scores_l2744_274477


namespace NUMINAMATH_CALUDE_solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l2744_274416

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Theorem for part I
theorem solution_set_f_x_minus_one_gt_two :
  {x : ℝ | f (x - 1) > 2} = {x : ℝ | x < -1 ∨ x > 3} :=
sorry

-- Theorem for part II
theorem min_value_x_plus_2y_plus_2z (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) :
  ∃ (m : ℝ), m = -9 ∧ ∀ (x' y' z' : ℝ), f x' ^ 2 + y' ^ 2 + z' ^ 2 = 9 → x' + 2 * y' + 2 * z' ≥ m :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l2744_274416


namespace NUMINAMATH_CALUDE_M_subset_N_l2744_274401

-- Define the sets M and N
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 8 - 1 / 4}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2744_274401


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2744_274428

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) (marks_per_wrong : ℤ) :
  total_questions = 80 →
  total_marks = 130 →
  correct_answers = 42 →
  marks_per_wrong = -1 →
  ∃ (marks_per_correct : ℤ),
    marks_per_correct * correct_answers + marks_per_wrong * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2744_274428


namespace NUMINAMATH_CALUDE_zeros_of_quadratic_function_l2744_274449

theorem zeros_of_quadratic_function (f : ℝ → ℝ) :
  (f = λ x => x^2 - x - 2) →
  (∀ x, f x = 0 ↔ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_quadratic_function_l2744_274449


namespace NUMINAMATH_CALUDE_jenna_blouses_count_l2744_274488

/-- The number of blouses Jenna needs to dye -/
def num_blouses : ℕ := 100

/-- The number of dots per blouse -/
def dots_per_blouse : ℕ := 20

/-- The amount of dye (in ml) needed per dot -/
def dye_per_dot : ℕ := 10

/-- The number of bottles of dye Jenna needs to buy -/
def num_bottles : ℕ := 50

/-- The volume (in ml) of each bottle of dye -/
def bottle_volume : ℕ := 400

/-- Theorem stating that the number of blouses Jenna needs to dye is correct -/
theorem jenna_blouses_count : 
  num_blouses * (dots_per_blouse * dye_per_dot) = num_bottles * bottle_volume :=
sorry

end NUMINAMATH_CALUDE_jenna_blouses_count_l2744_274488


namespace NUMINAMATH_CALUDE_adjacent_smaller_perfect_square_l2744_274452

theorem adjacent_smaller_perfect_square (m : ℕ) (h : ∃ k : ℕ, m = k^2) :
  ∃ n : ℕ, n^2 = m - 2*Int.sqrt m + 1 ∧
    n^2 < m ∧
    ∀ k : ℕ, k^2 < m → k^2 ≤ n^2 :=
sorry

end NUMINAMATH_CALUDE_adjacent_smaller_perfect_square_l2744_274452


namespace NUMINAMATH_CALUDE_remaining_steps_l2744_274446

/-- Given a total of 96 stair steps and 74 steps already climbed, 
    prove that the remaining steps to climb is 22. -/
theorem remaining_steps (total : Nat) (climbed : Nat) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_steps_l2744_274446


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2744_274498

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2744_274498


namespace NUMINAMATH_CALUDE_sum_of_abc_is_zero_l2744_274404

theorem sum_of_abc_is_zero 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c))
  (heq : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + c*a) + c^2 / (2*c^2 + a*b) = 1) :
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_is_zero_l2744_274404


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l2744_274493

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distributionWays (n : ℕ) : ℕ :=
  (2^n) / 2 - 1

/-- Theorem: There are 31 ways to distribute 6 distinguishable balls into 2 indistinguishable boxes -/
theorem six_balls_two_boxes : distributionWays 6 = 31 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l2744_274493


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l2744_274484

theorem max_value_sum_of_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 5) :
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 39 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 5 ∧
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l2744_274484


namespace NUMINAMATH_CALUDE_billy_lemon_heads_l2744_274494

theorem billy_lemon_heads (total_lemon_heads : ℕ) (lemon_heads_per_friend : ℕ) (h1 : total_lemon_heads = 72) (h2 : lemon_heads_per_friend = 12) :
  total_lemon_heads / lemon_heads_per_friend = 6 := by
sorry

end NUMINAMATH_CALUDE_billy_lemon_heads_l2744_274494


namespace NUMINAMATH_CALUDE_max_distance_from_point_to_unit_circle_l2744_274457

theorem max_distance_from_point_to_unit_circle :
  ∃ (M : ℝ), M = 6 ∧ ∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z - (3 - 4*I)) ≤ M ∧
    ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - (3 - 4*I)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_point_to_unit_circle_l2744_274457


namespace NUMINAMATH_CALUDE_value_of_2x_minus_y_l2744_274435

theorem value_of_2x_minus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x > y) :
  2 * x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_2x_minus_y_l2744_274435


namespace NUMINAMATH_CALUDE_car_speed_comparison_l2744_274407

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (u + 2 * v) / 3
  x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l2744_274407


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2744_274442

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2744_274442


namespace NUMINAMATH_CALUDE_probability_is_three_eighths_l2744_274405

/-- Represents a circular field with 8 roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ := 8)

/-- Represents a geologist on the field -/
structure Geologist :=
  (speed : ℝ)
  (time : ℝ)
  (road : ℕ)

/-- Calculates the distance between two geologists -/
def distance_between (g1 g2 : Geologist) (field : CircularField) : ℝ :=
  sorry

/-- Determines if two geologists are more than 8 km apart -/
def more_than_8km_apart (g1 g2 : Geologist) (field : CircularField) : Prop :=
  distance_between g1 g2 field > 8

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km_apart (field : CircularField) : ℝ :=
  sorry

theorem probability_is_three_eighths (field : CircularField) 
  (g1 g2 : Geologist) 
  (h1 : field.num_roads = 8) 
  (h2 : g1.speed = 5) 
  (h3 : g2.speed = 5) 
  (h4 : g1.time = 1) 
  (h5 : g2.time = 1) :
  probability_more_than_8km_apart field = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_is_three_eighths_l2744_274405


namespace NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_integers_l2744_274443

/-- Given an even number of consecutive integers with arithmetic mean z,
    the sum of the smallest and largest integers is equal to 2z. -/
theorem sum_smallest_largest_consecutive_integers (m : ℕ) (z : ℚ) (h_even : Even m) (h_pos : 0 < m) :
  let b : ℚ := (2 * z * m - m^2 + m) / (2 * m)
  (b + (b + m - 1)) = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_integers_l2744_274443


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2744_274459

theorem opposite_of_negative_2023 : 
  -((-2023) : ℤ) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2744_274459


namespace NUMINAMATH_CALUDE_total_share_l2744_274476

theorem total_share (z y x : ℝ) : 
  z = 300 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 1110 := by
sorry

end NUMINAMATH_CALUDE_total_share_l2744_274476


namespace NUMINAMATH_CALUDE_R_has_smallest_d_l2744_274431

/-- Represents a square with four labeled sides --/
structure Square where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The four squares given in the problem --/
def P : Square := { a := 2, b := 3, c := 10, d := 8 }
def Q : Square := { a := 8, b := 1, c := 2, d := 6 }
def R : Square := { a := 4, b := 5, c := 7, d := 1 }
def S : Square := { a := 7, b := 6, c := 5, d := 3 }

/-- Theorem stating that R has the smallest d value among the squares --/
theorem R_has_smallest_d : 
  R.d ≤ P.d ∧ R.d ≤ Q.d ∧ R.d ≤ S.d ∧ 
  (R.d < P.d ∨ R.d < Q.d ∨ R.d < S.d) := by
  sorry

end NUMINAMATH_CALUDE_R_has_smallest_d_l2744_274431


namespace NUMINAMATH_CALUDE_bob_distance_at_meeting_l2744_274440

/-- The distance between point X and point Y in miles -/
def total_distance : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 28 miles when they met -/
theorem bob_distance_at_meeting : 
  ∃ (t : ℝ), t > 0 ∧ yolanda_speed * (t + time_difference) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 28 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_at_meeting_l2744_274440


namespace NUMINAMATH_CALUDE_number_problem_l2744_274470

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2744_274470


namespace NUMINAMATH_CALUDE_perimeter_of_specific_shape_l2744_274400

/-- A shape with three sides of equal length -/
structure ThreeSidedShape where
  side_length : ℝ
  num_sides : ℕ
  h_num_sides : num_sides = 3

/-- The perimeter of a three-sided shape -/
def perimeter (shape : ThreeSidedShape) : ℝ :=
  shape.side_length * shape.num_sides

/-- Theorem: The perimeter of a shape with 3 sides, each of length 7 cm, is 21 cm -/
theorem perimeter_of_specific_shape :
  ∃ (shape : ThreeSidedShape), shape.side_length = 7 ∧ perimeter shape = 21 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_shape_l2744_274400
