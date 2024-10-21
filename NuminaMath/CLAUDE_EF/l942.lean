import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_even_integers_l942_94270

theorem min_even_integers (x y z a b m : ℤ) : 
  x + y + z = 33 →
  (a ∈ ({8, 9, 10} : Set ℤ) ∧ b ∈ ({8, 9, 10} : Set ℤ)) →
  x + y + z + a + b = 52 →
  m ∈ ({13, 14, 15} : Set ℤ) →
  x + y + z + a + b + m = 67 →
  ∃ (evens : Finset ℤ), evens.card = 1 ∧ 
    (∀ i ∈ evens, Even i) ∧
    (∀ i ∈ ({x, y, z, a, b, m} : Finset ℤ), Even i → i ∈ evens) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_even_integers_l942_94270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_equation_solution_l942_94243

theorem symmetric_points_equation_solution (a : ℝ) : 
  (Real.sin a = -Real.cos (3 * a) ∧ Real.sin (3 * a) = -Real.cos a) ↔ 
  ∃ n : ℤ, a = -Real.pi/8 + Real.pi/2 * ↑n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_equation_solution_l942_94243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l942_94282

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem min_value_f_on_interval :
  ∃ (min : ℝ), min = -4/3 ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x ≥ min) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f x = min) := by
  sorry

#check min_value_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l942_94282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_implies_a_geq_one_l942_94241

/-- A quadratic function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- The theorem states that if f(x) is decreasing on (-∞, 1], then a ≥ 1 -/
theorem quadratic_decreasing_implies_a_geq_one (a : ℝ) :
  (∀ x y, x ≤ y → y ≤ 1 → f a x ≥ f a y) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_decreasing_implies_a_geq_one_l942_94241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l942_94266

theorem divisor_problem : ∃ d : Nat, 
  (d > 0) ∧
  (50 % d = 0) ∧ 
  (50 / d + 50 + d = 65) ∧ 
  (d = 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l942_94266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l942_94208

/-- The rectangular region from which point P is selected -/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The area of the rectangular region -/
noncomputable def rectangleArea : ℝ := 6

/-- The region where points are closer to (0,0) than to (4,2) -/
def closerToOrigin : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 < (p.1 - 4)^2 + (p.2 - 2)^2}

/-- The area of the intersection of the rectangle and the region closer to the origin -/
noncomputable def intersectionArea : ℝ := 1

/-- The probability of a randomly selected point being closer to (0,0) than to (4,2) -/
noncomputable def probability : ℝ := intersectionArea / rectangleArea

theorem probability_is_one_sixth :
  probability = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l942_94208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l942_94204

theorem smallest_value_of_expression (a b : ℕ) (ha : a < 6) (hb : b < 8) :
  (∀ x y : ℕ, x < 6 → y < 8 → (2 : ℤ) * x - x * y ≥ (2 : ℤ) * a - a * b) →
  (2 : ℤ) * a - a * b = -25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l942_94204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_amount_calculation_l942_94213

/-- The original agreed-upon amount for one year of service -/
def original_amount : ℕ → ℕ := sorry

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The amount received in cash for the actual work period -/
def cash_received : ℕ := 300

/-- The price of the uniform -/
def uniform_price : ℕ := 300

theorem original_amount_calculation :
  original_amount months_worked = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_amount_calculation_l942_94213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l942_94248

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z₁
noncomputable def z₁ : ℂ := (3 - i) / (1 + i)

-- Define the property for z₂
def z₂_property (z : ℂ) : Prop := z.im = 2

-- Define the property for z₁z₂
def z₁z₂_property (z : ℂ) : Prop := (z₁ * z).im = 0

theorem complex_problem :
  ∃ (z₂ : ℂ), z₂_property z₂ ∧ z₁z₂_property z₂ ∧ Complex.abs z₁ = Real.sqrt 5 ∧ z₂ = 1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l942_94248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_is_three_l942_94237

/-- Represents the market scenario with Peter's purchases -/
structure MarketScenario where
  initial_money : ℚ
  potatoes_kg : ℚ
  potatoes_price : ℚ
  tomatoes_kg : ℚ
  cucumbers_kg : ℚ
  cucumbers_price : ℚ
  bananas_kg : ℚ
  bananas_price : ℚ
  remaining_money : ℚ

/-- Calculates the price per kilo of tomatoes -/
def tomato_price_per_kg (scenario : MarketScenario) : ℚ :=
  let total_spent := scenario.initial_money - scenario.remaining_money
  let other_items_cost := scenario.potatoes_kg * scenario.potatoes_price +
                          scenario.cucumbers_kg * scenario.cucumbers_price +
                          scenario.bananas_kg * scenario.bananas_price
  (total_spent - other_items_cost) / scenario.tomatoes_kg

/-- Theorem stating that the price per kilo of tomatoes is $3 -/
theorem tomato_price_is_three (scenario : MarketScenario)
  (h1 : scenario.initial_money = 500)
  (h2 : scenario.potatoes_kg = 6)
  (h3 : scenario.potatoes_price = 2)
  (h4 : scenario.tomatoes_kg = 9)
  (h5 : scenario.cucumbers_kg = 5)
  (h6 : scenario.cucumbers_price = 4)
  (h7 : scenario.bananas_kg = 3)
  (h8 : scenario.bananas_price = 5)
  (h9 : scenario.remaining_money = 426) :
  tomato_price_per_kg scenario = 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_is_three_l942_94237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_two_l942_94229

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

-- State the theorem
theorem unique_solution_is_negative_two :
  ∃! x, f x = 5 ∧ x = -2 := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_two_l942_94229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_count_l942_94216

/-- The number of bees in a hive -/
structure BeeHive where
  worker : ℕ
  drone : ℕ

/-- The ratio of worker bees to drone bees -/
def worker_drone_ratio : ℕ := 16

/-- The initial number of worker bees -/
def initial_workers : ℕ := 128

/-- The number of additional drones that fly in -/
def additional_drones : ℕ := 8

/-- Calculate the total number of bees in the hive -/
def total_bees (hive : BeeHive) : ℕ :=
  hive.worker + hive.drone

/-- Theorem stating the total number of bees in the hive -/
theorem bee_count :
  ∃ (hive : BeeHive),
    hive.worker = initial_workers ∧
    hive.drone = initial_workers / worker_drone_ratio + additional_drones ∧
    total_bees hive = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_count_l942_94216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l942_94283

/-- The function f(x) is defined as the minimum of 3x+1, x+1, and -x+10 for all real x. -/
noncomputable def f (x : ℝ) : ℝ := min (min (3 * x + 1) (x + 1)) (-x + 10)

/-- The maximum value of f(x) is 11/2. -/
theorem max_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 11 / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l942_94283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l942_94253

theorem equation_solution :
  ∃! x : ℝ, (128 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2) = (256 : ℝ) ^ x ∧ x = -2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l942_94253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shirt_cost_is_three_l942_94232

/-- The cost of each T-shirt given the conditions of Maddie's purchase -/
def t_shirt_cost : ℚ :=
  let white_packs : ℕ := 2
  let blue_packs : ℕ := 4
  let white_per_pack : ℕ := 5
  let blue_per_pack : ℕ := 3
  let total_cost : ℚ := 66
  let total_shirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack
  total_cost / total_shirts

#eval t_shirt_cost -- This will evaluate to 3

/-- Proof that the cost of each T-shirt is $3 -/
theorem t_shirt_cost_is_three : t_shirt_cost = 3 := by
  -- Unfold the definition of t_shirt_cost
  unfold t_shirt_cost
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#check t_shirt_cost_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shirt_cost_is_three_l942_94232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_l942_94227

theorem largest_fraction :
  let a : ℚ := 10 / 21
  let b : ℚ := 75 / 151
  let c : ℚ := 29 / 59
  let d : ℚ := 201 / 403
  let e : ℚ := 301 / 601
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_l942_94227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_capacity_l942_94209

/-- Calculates the maximum number of people who can ride a Ferris wheel with safety measures -/
def max_riders (num_seats : ℕ) (people_per_seat : ℕ) (safety_percentage : ℚ) : ℕ :=
  (((num_seats : ℚ) * (people_per_seat : ℚ)) * safety_percentage).floor.toNat

/-- Theorem stating the maximum number of riders for the given Ferris wheel -/
theorem ferris_wheel_capacity :
  max_riders 14 6 (4/5 : ℚ) = 67 := by
  rfl

#eval max_riders 14 6 (4/5 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_capacity_l942_94209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_intersections_l942_94221

/-- Represents an infinite chessboard with blue cells -/
def InfiniteChessboard := ℕ → ℕ → Prop

/-- Predicate to check if a cell is blue -/
def is_blue (board : InfiniteChessboard) (row col : ℕ) : Prop := board row col

/-- Predicate to check if any 10x10 grid contains at least one blue cell -/
def has_blue_in_10x10 (board : InfiniteChessboard) : Prop :=
  ∀ (r c : ℕ), ∃ (i j : ℕ), i < 10 ∧ j < 10 ∧ is_blue board (r + i) (c + j)

/-- Main theorem: For any positive n, there exist n rows and n columns with all intersections blue -/
theorem blue_intersections (board : InfiniteChessboard) (h : has_blue_in_10x10 board) :
  ∀ (n : ℕ), n > 0 → ∃ (rows cols : Finset ℕ),
    Finset.card rows = n ∧ Finset.card cols = n ∧
    ∀ (r c : ℕ), r ∈ rows → c ∈ cols → is_blue board r c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_intersections_l942_94221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_hexagon_area_ratio_dart_probability_inner_hexagon_l942_94280

/-- Represents a regular hexagon -/
structure RegularHexagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a regular hexagon -/
noncomputable def areaHexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.sideLength ^ 2

/-- The area of the inner hexagon formed by connecting midpoints of sides -/
noncomputable def areaInnerHexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * (h.sideLength / 2) ^ 2

/-- The theorem stating that the ratio of inner hexagon area to total area is 1/4 -/
theorem inner_hexagon_area_ratio (h : RegularHexagon) :
  areaInnerHexagon h / areaHexagon h = 1 / 4 := by
  sorry

/-- The probability of a dart landing in the inner hexagon -/
noncomputable def probabilityInnerHexagon (h : RegularHexagon) : ℝ :=
  areaInnerHexagon h / areaHexagon h

/-- The theorem stating that the probability of a dart landing in the inner hexagon is 1/4 -/
theorem dart_probability_inner_hexagon (h : RegularHexagon) :
  probabilityInnerHexagon h = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_hexagon_area_ratio_dart_probability_inner_hexagon_l942_94280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l942_94245

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_derivative (f f' : ℝ → ℝ) : Prop := ∀ x, deriv f x = f' x

-- State the theorem
theorem function_properties 
  (h_even : is_even f)
  (h_deriv : is_derivative f f')
  (h_eq : ∀ x, f (2 + x) = 4 - f (-x))
  (h_cond : 2 * f' 1 = 2 * f 0 ∧ 2 * f 0 = f 1) :
  f (-1) = 2 * f' 1 ∧ (Finset.range 50).sum (fun i => f (i + 1 : ℝ)) = 101 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l942_94245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_l942_94272

theorem angle_sum_is_pi (a b : ℝ) (h_acute_a : 0 < a ∧ a < Real.pi / 2) (h_acute_b : 0 < b ∧ b < Real.pi / 2)
  (h_cos : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2) (h_sin : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + 3 * b = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_l942_94272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l942_94265

-- Define the function f(x) = |tan x|
noncomputable def f (x : ℝ) := abs (Real.tan x)

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Theorem statement
theorem f_increasing_intervals (k : ℤ) :
  IncreasingOn f (k * π) (k * π + π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l942_94265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cappuccino_price_calculation_l942_94263

/-- Represents the cost of a coffee order -/
structure CoffeeOrder where
  total : ℚ
  drip_coffee_price : ℚ
  drip_coffee_count : ℕ
  espresso_price : ℚ
  latte_price : ℚ
  latte_count : ℕ
  vanilla_syrup_price : ℚ
  cold_brew_price : ℚ
  cold_brew_count : ℕ
  cappuccino_price : ℚ

/-- The theorem states that given the specific coffee order details,
    the price of the cappuccino is $3.50 -/
theorem cappuccino_price_calculation (order : CoffeeOrder) :
  order.total = 25 ∧
  order.drip_coffee_price = 2.25 ∧
  order.drip_coffee_count = 2 ∧
  order.espresso_price = 3.5 ∧
  order.latte_price = 4 ∧
  order.latte_count = 2 ∧
  order.vanilla_syrup_price = 0.5 ∧
  order.cold_brew_price = 2.5 ∧
  order.cold_brew_count = 2 →
  order.cappuccino_price = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cappuccino_price_calculation_l942_94263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proportion_theorem_l942_94206

/-- Two triangles FGH and FJK with given properties -/
structure Triangles where
  FG : ℝ
  GH : ℝ
  FJ : ℝ
  JK : ℝ  -- Add JK as a field in the structure
  angle_equal : ℝ  -- Represents the equal angle ∠FGH = ∠FJK
  prop_cond : ℝ → ℝ  -- Function to represent the proportionality condition

/-- Theorem stating that under given conditions, JK = 2.25 -/
theorem triangle_proportion_theorem (t : Triangles) 
  (h1 : t.FG = 4.5)
  (h2 : t.GH = 6)
  (h3 : t.FJ = 3)
  (h4 : t.GH / t.FJ = t.FG / t.JK) :
  t.JK = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proportion_theorem_l942_94206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_is_8_l942_94290

-- Define the parameters
noncomputable def cylinder_diameter : ℝ := 16
noncomputable def cylinder_height : ℝ := 16
def num_spheres : ℕ := 12

-- Define the volume of the cylinder
noncomputable def cylinder_volume : ℝ := Real.pi * (cylinder_diameter / 2) ^ 2 * cylinder_height

-- Define the volume of one sphere
noncomputable def sphere_volume : ℝ := cylinder_volume / num_spheres

-- State the theorem
theorem sphere_diameter_is_8 :
  ∃ (d : ℝ), d = 8 ∧ (4 / 3) * Real.pi * (d / 2) ^ 3 = sphere_volume := by
  -- The proof goes here
  sorry

#eval num_spheres -- This will compile and evaluate to 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_is_8_l942_94290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_friendly_triplets_l942_94202

/-- A group of people where each person has exactly three friends -/
structure FriendGroup where
  people : Finset ℕ
  friends : ℕ → Finset ℕ
  friend_count : ∀ p, p ∈ people → (friends p).card = 3
  symmetry : ∀ {p q}, p ∈ people → q ∈ friends p → p ∈ friends q

/-- A triplet of mutually friendly people -/
def FriendlyTriplet (g : FriendGroup) : Type :=
  { t : Finset ℕ // t.card = 3 ∧ ∀ p q, p ∈ t → q ∈ t → p ≠ q → q ∈ g.friends p }

theorem hundred_friendly_triplets (g : FriendGroup) 
  (h_size : g.people.card = 100)
  (h_99_triplets : ∃ triplets : Finset (FriendlyTriplet g), 
    triplets.card = 99 ∧ (∀ t1 t2, t1 ∈ triplets → t2 ∈ triplets → t1 ≠ t2 → t1.val ≠ t2.val)) :
  ∃ t : FriendlyTriplet g, t ∉ Classical.choose h_99_triplets :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_friendly_triplets_l942_94202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_556_forms_triangle_l942_94288

/-- Check if three lengths can form a triangle --/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of stick lengths --/
def stickSets : List (ℝ × ℝ × ℝ) :=
  [(7, 4, 2), (5, 5, 6), (3, 4, 8), (2, 3, 5)]

/-- Theorem: Only the set (5, 5, 6) can form a triangle --/
theorem only_556_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ stickSets ∧
    let (a, b, c) := set
    canFormTriangle a b c ∧ set = (5, 5, 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_556_forms_triangle_l942_94288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_170_negative_l942_94271

theorem tan_170_negative : Real.tan (170 * π / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_170_negative_l942_94271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_reuleaux_min_area_l942_94246

/-- A curve of constant width -/
class ConstantWidthCurve (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  width : ℝ
  is_constant_width : ∀ (p q : α), ∃ (r : α), ‖p - r‖ = width ∧ ‖q - r‖ = width

/-- Circle of diameter h -/
def Circle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] (h : ℝ) : Set α :=
  {p : α | ∃ (c : α), ‖p - c‖ ≤ h / 2}

/-- Reuleaux triangle of width h -/
def ReuleauxTriangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] (h : ℝ) : Set α :=
  sorry  -- Definition of Reuleaux triangle

/-- Area of a set -/
noncomputable def area {α : Type*} [MeasurableSpace α] (s : Set α) : ℝ :=
  sorry  -- Definition of area

/-- Theorem: Circle has largest area, Reuleaux triangle has smallest area -/
theorem circle_max_reuleaux_min_area
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] [MeasurableSpace α]
  (h : ℝ) (K : Set α) [ConstantWidthCurve α] :
  area (Circle α h) ≥ area K ∧ area K ≥ area (ReuleauxTriangle α h) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_reuleaux_min_area_l942_94246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l942_94239

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

def arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

theorem triangle_properties (t : Triangle) (h : arithmetic_sequence t) :
  t.b^2 ≥ t.a * t.c ∧
  1 / t.a + 1 / t.c ≥ 2 / t.b ∧
  t.b^2 ≤ (t.a^2 + t.c^2) / 2 ∧
  t.B ≤ π / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l942_94239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l942_94217

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the problem
theorem triangle_problem (t : Triangle) (m n : ℝ × ℝ) :
  -- Given conditions
  m = (Real.cos t.B, Real.sin t.C) →
  n = (Real.cos t.C, -Real.sin t.B) →
  m.1 * n.1 + m.2 * n.2 = 1/2 →
  t.a = 2 * Real.sqrt 3 →
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 →
  -- Conclusions
  t.A = 2 * Real.pi / 3 ∧ t.b + t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l942_94217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangency_l942_94278

/-- Given a circle and a parabola whose axes of symmetry are tangent to each other, 
    prove that the parameter m in the circle equation has a specific value. -/
theorem circle_parabola_tangency (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + m*x - 1/4 = 0 → 
    ∃ c : ℝ, (x + c)^2 + y^2 = (1 + c^2)/4) ∧ 
  (∀ x y : ℝ, y = 1/4 * x^2) ∧
  (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 + m*x - 1/4 = 0 → y = k) ∧
  (∀ x : ℝ, 1/4 * x^2 = -1) →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangency_l942_94278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l942_94294

theorem problem_statement (t : ℝ) (k m n : ℕ) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 5/4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = m/n - Real.sqrt k)
  (h3 : k > 0 ∧ m > 0 ∧ n > 0)
  (h4 : Nat.Coprime m n) :
  k + m + n = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l942_94294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l942_94225

theorem rationalize_denominator :
  ∃ (A B C D : ℕ), 
    (A = 49 ∧ B = 35 ∧ C = 25 ∧ D = 2) ∧
    (1 / ((5 : ℝ)^(1/3) - (7 : ℝ)^(1/3)) = ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / D) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l942_94225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l942_94261

theorem triangle_tan_a (A B C : ℝ) (a b c : ℝ) : 
  C = 2 * Real.pi / 3 →  -- 120° in radians
  a = 2 * b → 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  A + B + C = Real.pi →  -- sum of angles in a triangle
  Real.tan A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l942_94261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_center_of_mass_l942_94214

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a mass point in 2D space -/
structure MassPoint where
  point : Point2D
  mass : ℝ

/-- Calculates the center of mass of two mass points -/
noncomputable def centerOfMass (p1 p2 : MassPoint) : Point2D :=
  { x := (p1.mass * p1.point.x + p2.mass * p2.point.x) / (p1.mass + p2.mass)
  , y := (p1.mass * p1.point.y + p2.mass * p2.point.y) / (p1.mass + p2.mass) }

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Main theorem statement -/
theorem triangle_center_of_mass 
  (A B C : Point2D) (m1 m2 m3 : ℝ) 
  (h1 : m1 > 0) (h2 : m2 > 0) (h3 : m3 > 0) : 
  let mp1 : MassPoint := { point := A, mass := m1 }
  let mp2 : MassPoint := { point := B, mass := m2 }
  let mp3 : MassPoint := { point := C, mass := m3 }
  let C1 := centerOfMass mp1 mp2
  let O := centerOfMass { point := C1, mass := m1 + m2 } mp3
  areCollinear C O C1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_center_of_mass_l942_94214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_ratio_l942_94257

theorem chocolate_ratio (n a m : ℕ) : 
  n = 10 →
  m = 5 →
  a = n + 15 →
  (a + m) / n = 3 := by
  intros hn hm ha
  rw [hn, hm, ha]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_ratio_l942_94257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_l942_94293

theorem number_of_sets (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  Finset.card (Finset.filter (fun A => {a, b} ∪ A = {a, b, c}) (Finset.powerset {a, b, c})) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_l942_94293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l942_94244

/-- A regular polygon with an exterior angle of 45° has 8 sides -/
theorem regular_polygon_with_45_degree_exterior_angle_has_8_sides 
  (n : ℕ) -- number of sides
  (is_regular : n ≥ 3) -- the polygon is regular (at least 3 sides)
  (exterior_angle : ℝ) -- measure of the exterior angle
  (h_exterior : exterior_angle = 45) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l942_94244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_difference_l942_94273

/-- The number of red points on the circle -/
def num_red_points : ℕ := 60

/-- The number of blue points on the circle -/
def num_blue_points : ℕ := 1

/-- The total number of points on the circle -/
def total_points : ℕ := num_red_points + num_blue_points

/-- The number of ways to choose 2 points from n points -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem polygon_difference :
  choose_two num_red_points = 
    (choose_two total_points) - (choose_two num_red_points) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_difference_l942_94273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_saved_l942_94254

-- Define the train's speed and time
noncomputable def normal_speed : ℝ := 1
noncomputable def half_speed : ℝ := normal_speed / 2
noncomputable def time_at_half_speed : ℝ := 8

-- Theorem statement
theorem train_time_saved : 
  let time_at_normal_speed := time_at_half_speed * half_speed / normal_speed
  time_at_half_speed - time_at_normal_speed = 4 := by
  -- Unfold the definitions
  unfold time_at_half_speed half_speed normal_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_saved_l942_94254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l942_94297

-- Define the function (marked as noncomputable due to use of real exponential)
noncomputable def f (x : ℝ) : ℝ := (2^x - 3) / (2^x + 1)

-- State the theorem
theorem range_of_f : Set.range f = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l942_94297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_sum_equality_l942_94233

theorem mod_sum_equality (a p : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < p) :
  (a % p + a % (2 * p) + a % (3 * p) + a % (4 * p) = a + p) ↔
  ((a = 3 * p ∧ p ≠ 1) ∨ (a = 3 ∧ p = 1) ∨ (a = 3 ∧ p = 17)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_sum_equality_l942_94233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_implies_defined_not_reverse_l942_94238

variable {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
variable (f : X → Y) (a : X)

theorem continuity_implies_defined_not_reverse :
  (Continuous f → ∃ y, f a = y) ∧
  ∃ g : X → Y, ∃ b : X, (∃ y, g b = y) ∧ ¬Continuous g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_implies_defined_not_reverse_l942_94238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l942_94298

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def system (x y : ℝ) : Prop :=
  floor (x + y - 3) = 2 - floor x ∧
  floor (x + 1) + floor (y - 7) + floor x = floor y

theorem unique_solution :
  ∃! (x y : ℝ), system x y ∧ x = 3 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l942_94298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_no_solution_l942_94247

theorem smallest_m_no_solution : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → 2^x + 3^y - 5^z ≠ 2*m) ∧
  (∀ (k : ℕ), 0 < k ∧ k < m → 
    ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2^x + 3^y - 5^z = 2*k) ∧
  m = 11 := by
  sorry

#check smallest_m_no_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_no_solution_l942_94247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_care_cost_calculation_l942_94256

/-- Calculate the total cost of lawn care supplies including discount and tax -/
theorem lawn_care_cost_calculation :
  let blade_cost : ℚ := 4 * 8
  let string_cost : ℚ := 2 * 7
  let fuel_cost : ℚ := 4
  let bag_cost : ℚ := 5
  let total_before_discount : ℚ := blade_cost + string_cost + fuel_cost + bag_cost
  let discount_rate : ℚ := 1 / 10
  let discount_amount : ℚ := discount_rate * total_before_discount
  let total_after_discount : ℚ := total_before_discount - discount_amount
  let tax_rate : ℚ := 5 / 100
  let tax_amount : ℚ := tax_rate * total_after_discount
  let total_cost : ℚ := total_after_discount + (tax_amount * 100).floor / 100
  total_cost = 5198 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_care_cost_calculation_l942_94256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variable_cost_is_11_50_l942_94276

/-- A small publishing company's book production model -/
structure BookProduction where
  fixedCosts : ℚ
  sellingPrice : ℚ
  breakEvenQuantity : ℚ

/-- Calculate the variable cost per book -/
def variableCostPerBook (bp : BookProduction) : ℚ :=
  (bp.sellingPrice * bp.breakEvenQuantity - bp.fixedCosts) / bp.breakEvenQuantity

/-- Theorem: The variable cost per book for the given scenario is $11.50 -/
theorem variable_cost_is_11_50 (bp : BookProduction) 
  (h1 : bp.fixedCosts = 35630)
  (h2 : bp.sellingPrice = 20.25)
  (h3 : bp.breakEvenQuantity = 4072) : 
  variableCostPerBook bp = 11.50 := by
  -- Unfold the definition of variableCostPerBook
  unfold variableCostPerBook
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variable_cost_is_11_50_l942_94276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_4_5_l942_94211

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x + 3| - m)

-- State the theorem
theorem exercise_4_5 :
  (∀ x, f 4 x ≥ 0) ∧ 
  (∀ m, m > 4 → ∃ x, f m x < 0) ∧
  (∀ a b, a > 0 → b > 0 → 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 → 7 * a + 4 * b ≥ 9 / 4) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 ∧ 7 * a + 4 * b = 9 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_4_5_l942_94211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_3000_pounds_l942_94228

/-- Represents the exchange rate between Euros and pounds -/
structure ExchangeRate where
  euros : ℚ
  pounds : ℚ

/-- Calculates the amount of Euros received for a given amount of pounds -/
def exchange (rate : ExchangeRate) (poundsToExchange : ℚ) : ℚ :=
  (rate.euros / rate.pounds) * poundsToExchange

/-- Theorem stating that exchanging 3000 pounds results in approximately 3461.54 Euros -/
theorem exchange_3000_pounds (rate : ExchangeRate) 
  (h1 : rate.euros = 4500)
  (h2 : rate.pounds = 3900) :
  ∃ ε > 0, |exchange rate 3000 - 3461.54| < ε := by
  sorry

#eval exchange ⟨4500, 3900⟩ 3000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_3000_pounds_l942_94228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l942_94236

theorem ticket_price_possibilities : 
  let possible_prices := {x : ℕ | x > 0 ∧ 48 % x = 0 ∧ 64 % x = 0}
  Finset.card (Finset.filter (λ x => x ∈ possible_prices) (Finset.range 65)) = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l942_94236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_crayons_count_l942_94242

/-- Given a crayon box with specific conditions, prove the number of blue crayons. -/
theorem blue_crayons_count (total : ℕ) (red : ℕ) (pink : ℕ) (blue : ℕ) :
  total = 24 →
  red = 8 →
  pink = 6 →
  total = red + blue + (2 * blue / 3) + pink →
  blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_crayons_count_l942_94242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l942_94212

open Set

-- Define the variables and functions
variable (a b : ℝ)
variable (x : ℝ)

-- Define the solution set of ax - b > 0
def S : Set ℝ := Iio (-1 : ℝ)

-- Define the theorem
theorem problem_solution (h : S = {x | a * x - b > 0}) :
  {x | (x - 2) * (a * x + b) < 0} = Iio 1 ∪ Ioi 2 :=
by
  -- The proof goes here
  sorry

-- The main result
#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l942_94212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l942_94264

/-- The function f(x) = ln x + x^2 - 2ax + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧
    ∀ a ∈ Set.Icc (-2) 0,
      2*m*Real.exp a*(a+1) + f a x₀ > a^2 + 2*a + 4) →
  m ∈ Set.Ioo 1 (Real.exp 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l942_94264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_heads_one_tails_is_half_l942_94284

-- Define the sample space for tossing a fair coin twice
inductive CoinToss
| HH -- Heads, Heads
| HT -- Heads, Tails
| TH -- Tails, Heads
| TT -- Tails, Tails

-- Define a function to count the number of outcomes with one heads and one tails
def countOneTailsOneHeads (outcome : CoinToss) : Nat :=
  match outcome with
  | CoinToss.HT => 1
  | CoinToss.TH => 1
  | _ => 0

-- Define the total number of possible outcomes
def totalOutcomes : Nat := 4

-- Define the probability of getting one heads and one tails
def probOneHeadsOneTails : Rat :=
  (List.sum (List.map countOneTailsOneHeads [CoinToss.HH, CoinToss.HT, CoinToss.TH, CoinToss.TT])) / totalOutcomes

-- State the theorem
theorem prob_one_heads_one_tails_is_half :
  probOneHeadsOneTails = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_heads_one_tails_is_half_l942_94284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_polar_l942_94295

/-- The polar equation of a circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos (θ - Real.pi/4)

/-- The center of the circle in rectangular coordinates -/
noncomputable def circle_center_rect : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

/-- Convert rectangular coordinates to polar coordinates -/
noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2), Real.arctan (y/x))

theorem circle_center_polar (ρ θ : ℝ) :
  circle_equation ρ θ → rect_to_polar circle_center_rect.1 circle_center_rect.2 = (1, Real.pi/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_polar_l942_94295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_equality_l942_94250

/-- Represents the problem of Maria and John traveling to a landmark -/
structure TravelProblem where
  totalDistance : ℝ
  mariaInitialSpeed : ℝ
  mariaReducedSpeed : ℝ
  johnSpeed : ℝ
  johnDelayStart : ℝ
  johnWaitTime : ℝ

/-- Calculates the time taken by Maria to reach the landmark -/
noncomputable def mariaTravelTime (p : TravelProblem) (restStopDistance : ℝ) : ℝ :=
  restStopDistance / p.mariaInitialSpeed + (p.totalDistance - restStopDistance) / p.mariaReducedSpeed

/-- Calculates the time taken by John to reach the landmark, including delays -/
noncomputable def johnTravelTime (p : TravelProblem) : ℝ :=
  p.totalDistance / p.johnSpeed + p.johnDelayStart + p.johnWaitTime

/-- Theorem stating that Maria and John reach the landmark at the same time, which is 5 hours -/
theorem travel_time_equality (p : TravelProblem) 
    (h1 : p.totalDistance = 140)
    (h2 : p.mariaInitialSpeed = 30)
    (h3 : p.mariaReducedSpeed = 10)
    (h4 : p.johnSpeed = 40)
    (h5 : p.johnDelayStart = 1)
    (h6 : p.johnWaitTime = 0.5)
    : ∃ (restStopDistance : ℝ), 
      mariaTravelTime p restStopDistance = johnTravelTime p ∧ 
      mariaTravelTime p restStopDistance = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_equality_l942_94250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_value_l942_94275

theorem trigonometric_sum_value : 
  36 * (Real.sin (π/8) ^ 4 + Real.cos (π/8) ^ 4 + Real.sin (7*π/8) ^ 4 + Real.cos (7*π/8) ^ 4) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_value_l942_94275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_theorem_l942_94260

/-- A trapezoid with perpendicular diagonals and circumscribed circle -/
structure SpecialTrapezoid (a c : ℝ) where
  -- AB and CD are parallel sides with lengths a and c
  ab_length : ℝ := a
  cd_length : ℝ := c
  -- Diagonals are perpendicular
  diagonals_perpendicular : True
  -- K is the circumscribed circle
  has_circumscribed_circle : True
  -- AB and CD are diameters of circles K_a and K_b
  ab_cd_are_diameters : True

/-- The area and perimeter of the part inside the circumscribed circle but outside the inscribed circles -/
noncomputable def area_and_perimeter (a c : ℝ) (t : SpecialTrapezoid a c) : ℝ × ℝ :=
  (a * c / 2, Real.pi * (a / 2 + c / 2 + Real.sqrt (a^2 + c^2) / 2))

/-- The main theorem about the area and perimeter of the special region in the trapezoid -/
theorem special_trapezoid_theorem (a c : ℝ) (t : SpecialTrapezoid a c) :
  area_and_perimeter a c t = (a * c / 2, Real.pi * (a / 2 + c / 2 + Real.sqrt (a^2 + c^2) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_theorem_l942_94260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_range_l942_94205

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 
  -(1/x^2) * Real.log (1 + x) + (1/x + a) * (1 / (1 + x))

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  (Real.log 2) * 1 + f (-1) 1 - Real.log 2 = 0 ∧
  f' (-1) 1 = -Real.log 2 := by sorry

-- Theorem for the range of a
theorem monotonic_increasing_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_range_l942_94205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_given_l942_94286

theorem marbles_given (initial_marbles : ℝ) (final_marbles : ℕ) 
  (h1 : initial_marbles = 87.0) 
  (h2 : final_marbles = 95) : 
  (final_marbles : ℝ) - initial_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_given_l942_94286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l942_94269

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * abs x + 2 * abs y

-- Define the set of points enclosed by the curve
def enclosed_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 ≤ 2 * abs x + 2 * abs y}

-- State the theorem
theorem area_of_curve : MeasureTheory.volume enclosed_set = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l942_94269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l942_94200

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2)

theorem phase_shift_and_amplitude :
  (∃ (shift : ℝ), ∀ (x : ℝ), f (x + shift) = 2 * Real.sin (4 * x)) ∧
  (∀ (x : ℝ), |f x| ≤ 2 ∧ ∃ (x₀ : ℝ), |f x₀| = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l942_94200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_iff_equal_sides_l942_94222

/-- A quadrilateral with four vertices in a 2D plane. -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The length of a line segment between two points in a 2D plane. -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A quadrilateral is a rhombus if all its sides are equal in length. -/
def is_rhombus (q : Quadrilateral) : Prop :=
  distance q.A q.B = distance q.B q.C ∧
  distance q.B q.C = distance q.C q.D ∧
  distance q.C q.D = distance q.D q.A

/-- Theorem: A quadrilateral is a rhombus if and only if all four sides are equal in length. -/
theorem rhombus_iff_equal_sides (q : Quadrilateral) :
  is_rhombus q ↔ 
    distance q.A q.B = distance q.B q.C ∧
    distance q.B q.C = distance q.C q.D ∧
    distance q.C q.D = distance q.D q.A ∧
    distance q.D q.A = distance q.A q.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_iff_equal_sides_l942_94222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_length_approx_250_l942_94224

-- Define the constants
noncomputable def train_a_length : ℝ := 300
noncomputable def train_a_speed : ℝ := 120
noncomputable def train_b_speed : ℝ := 100
noncomputable def crossing_time : ℝ := 9
noncomputable def track_incline : ℝ := 3

-- Define the function to convert km/h to m/s
noncomputable def km_per_hour_to_m_per_sec (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

-- Define the theorem
theorem train_b_length_approx_250 :
  ∃ (train_b_length : ℝ),
    (train_b_length ≥ 249.5 ∧ train_b_length < 250.5) ∧
    (train_a_length + train_b_length ≥ 
      (km_per_hour_to_m_per_sec train_a_speed + km_per_hour_to_m_per_sec train_b_speed) * crossing_time - 0.5 ∧
     train_a_length + train_b_length < 
      (km_per_hour_to_m_per_sec train_a_speed + km_per_hour_to_m_per_sec train_b_speed) * crossing_time + 0.5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_length_approx_250_l942_94224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l942_94281

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 5
  | (n + 3) => 5 * sequence_a (n + 2) - 6 * sequence_a (n + 1)

def geometric_sequence (n : ℕ) : ℤ := 2 * 2^(n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) - 3 * sequence_a n = geometric_sequence n) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 3^n - 2^n) ∧
  (sequence_a 1 < 2 * 1^2 + 1) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n ≥ 2 * n^2 + 1) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l942_94281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l942_94268

noncomputable def f (x : ℝ) : ℝ := (x^2 - 9) / (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 2) ∨ 2 < x} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l942_94268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_rectangular_parallelepiped_l942_94207

-- Define the properties of the rectangular parallelepiped
def rectangular_parallelepiped (V α β H : ℝ) : Prop :=
  V > 0 ∧ 0 < α ∧ α < Real.pi ∧ 0 < β ∧ β < Real.pi/2 ∧ H > 0 ∧
  H = (2 * V * Real.tan β ^ 2 / Real.sin α) ^ (1/3)

-- State the theorem
theorem height_of_rectangular_parallelepiped (V α β : ℝ) :
  ∃ H, rectangular_parallelepiped V α β H :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_rectangular_parallelepiped_l942_94207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sum_with_reverse_is_1111_l942_94285

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define a structure for four consecutive digits
structure ConsecutiveDigits where
  a : Digit
  b : Digit
  c : Digit
  d : Digit
  consecutive : b.val = a.val + 1 ∧ c.val = a.val + 2 ∧ d.val = a.val + 3

-- Define the function to create a four-digit number from digits
def makeNumber (digits : ConsecutiveDigits) : Nat :=
  1000 * digits.a.val + 100 * digits.b.val + 10 * digits.c.val + digits.d.val

-- Define the function to reverse a four-digit number
def reverseNumber (digits : ConsecutiveDigits) : Nat :=
  1000 * digits.d.val + 100 * digits.c.val + 10 * digits.b.val + digits.a.val

-- Define the sum of a number and its reverse
def sumWithReverse (digits : ConsecutiveDigits) : Nat :=
  makeNumber digits + reverseNumber digits

-- Theorem statement
theorem gcd_of_sum_with_reverse_is_1111 :
  ∀ (digits : ConsecutiveDigits),
    (∃ (k : Nat), sumWithReverse digits = 1111 * k) ∧
    (∀ (d : Nat), d > 1111 → ¬(∀ (digits : ConsecutiveDigits), d ∣ sumWithReverse digits)) := by
  sorry

#check gcd_of_sum_with_reverse_is_1111

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sum_with_reverse_is_1111_l942_94285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_cannot_tessellate_l942_94279

/-- A shape can tessellate if its internal angles can sum up to 360°. -/
noncomputable def canTessellate (internalAngle : ℝ) : Prop :=
  ∃ k : ℕ, k • internalAngle = 360

/-- The internal angle of a regular polygon with n sides. -/
noncomputable def internalAngle (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- Theorem: A regular pentagon cannot be tessellated. -/
theorem pentagon_cannot_tessellate :
  ¬ canTessellate (internalAngle 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_cannot_tessellate_l942_94279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_winning_strategy_l942_94215

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the game state -/
structure GameState where
  n : ℕ  -- Number of points Bogdan can choose
  k : ℕ  -- Number of distances Monica can write
  monica_numbers : List ℝ  -- Distances written by Monica
  bogdan_points : List Point  -- Points chosen by Bogdan

/-- Checks if Bogdan wins given the current game state -/
def bogdan_wins (state : GameState) : Prop :=
  ∀ m, m ∈ state.monica_numbers → ∃ p1 p2, p1 ∈ state.bogdan_points ∧ p2 ∈ state.bogdan_points ∧ distance p1 p2 = m

/-- The main theorem stating the winning condition for Monica -/
theorem monica_winning_strategy (n k : ℕ) :
  (∀ state : GameState, state.n = n ∧ state.k = k ∧ state.monica_numbers.length = k →
    ¬bogdan_wins state) ↔ n ≤ k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_winning_strategy_l942_94215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l942_94226

noncomputable def f (x : ℝ) : ℝ := Real.exp (1 - x)

noncomputable def f_derivative (x : ℝ) : ℝ := -Real.exp (1 - x)

noncomputable def x₀ : ℝ := -1
noncomputable def y₀ : ℝ := f x₀

theorem tangent_line_equation :
  ∃ (m : ℝ), m = f_derivative x₀ ∧
  ∀ (x y : ℝ), y = m * x ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (y - y₀ = m * (x - x₀) ∧ f x₀ = y₀ ∧ f_derivative x₀ = m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l942_94226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l942_94259

/-- The cost in dollars to transport 1 kilogram of the sensor unit -/
noncomputable def sensor_cost_per_kg : ℚ := 25000

/-- The cost in dollars to transport 1 kilogram of the communication module -/
noncomputable def comm_cost_per_kg : ℚ := 20000

/-- The weight of the sensor unit in grams -/
noncomputable def sensor_weight_g : ℚ := 500

/-- The weight of the communication module in grams -/
noncomputable def comm_weight_g : ℚ := 1500

/-- Conversion factor from grams to kilograms -/
noncomputable def g_to_kg : ℚ := 1000

/-- The total cost of transporting both the sensor unit and the communication module -/
noncomputable def total_transport_cost : ℚ :=
  (sensor_weight_g / g_to_kg) * sensor_cost_per_kg +
  (comm_weight_g / g_to_kg) * comm_cost_per_kg

theorem transport_cost_calculation :
  total_transport_cost = 42500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l942_94259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximized_at_288_l942_94218

-- Define the divisor function d(n)
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

-- Define the function g(n)
noncomputable def g (n : ℕ+) : ℝ := (d n)^2 / (n.val : ℝ)^(1/4 : ℝ)

-- Theorem statement
theorem g_maximized_at_288 :
  ∀ m : ℕ+, m ≠ 288 → g ⟨288, by norm_num⟩ > g m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximized_at_288_l942_94218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l942_94274

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of the parabola y² = 4x -/
def is_on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Definition of the focus F(1, 0) -/
def focus : Point :=
  ⟨1, 0⟩

/-- Definition of a point being on a line -/
def is_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Definition of distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem parabola_line_intersection
  (l : Line)
  (A B P : Point)
  (hl : is_on_line focus l)
  (hA : is_on_parabola A ∧ is_on_line A l)
  (hB : is_on_parabola B ∧ is_on_line B l)
  (hP : is_on_parabola P)
  (hPq : P.x ≥ 0 ∧ P.y ≥ 0)  -- P is in the first quadrant
  (hPF : distance P focus = 3/2)
  (hM : P.x = (A.x + B.x) / 2)  -- P has same x-coordinate as midpoint of AB
  : l = ⟨Real.sqrt 2, -1, -Real.sqrt 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l942_94274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_result_l942_94287

/-- The equation function that calculates x given the parameters a, b, c, d, e, and f -/
noncomputable def equation (a b c d e f : ℝ) : ℝ :=
  ((0.47 * (1442 + a^2)) - (0.36 * (1412 - b^3))) + (65 + c * Real.log d) + e * Real.sin f

/-- The theorem stating that the equation results in approximately 261.56138 for the given values -/
theorem equation_result :
  ∃ (x : ℝ), abs (x - equation 3 2 8 7 4 (Real.pi / 3)) < 0.00001 ∧ abs (x - 261.56138) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_result_l942_94287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l942_94235

theorem inequality_proof (a b : ℕ+) :
  let c : ℚ := (a^(a.val+1) + b^(b.val+1)) / (a^a.val + b^b.val)
  c^a.val + c^b.val ≥ a^a.val + b^b.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l942_94235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l942_94277

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x)/a + a/(2^x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : ∀ x, f a x = f a (-x)) :
  a = 1 ∧ ∀ x > 0, Monotone (f 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l942_94277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_f_l942_94249

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + x))

theorem smallest_value_of_f :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 2) ∧ (∃ x : ℝ, x ≥ 0 ∧ f x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_f_l942_94249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_angle_is_arctan_two_minus_sqrt_three_l942_94291

/-- A regular quadrilateral pyramid with side length of base equal to slant height -/
structure RegularQuadPyramid where
  side_length : ℝ
  slant_height : ℝ
  slant_height_eq_side : slant_height = side_length

/-- A cross-section of the pyramid that divides the surface in half -/
structure CrossSection (p : RegularQuadPyramid) where
  divides_surface_in_half : Bool

/-- The angle between the plane of the cross-section and the base of the pyramid -/
noncomputable def cross_section_angle (p : RegularQuadPyramid) (cs : CrossSection p) : ℝ := 
  Real.arctan (2 - Real.sqrt 3)

/-- Theorem stating that the angle between the cross-section and base is arctan(2 - √3) -/
theorem cross_section_angle_is_arctan_two_minus_sqrt_three 
  (p : RegularQuadPyramid) (cs : CrossSection p) :
  cross_section_angle p cs = Real.arctan (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_angle_is_arctan_two_minus_sqrt_three_l942_94291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l942_94262

theorem sin_double_angle (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l942_94262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l942_94252

-- Define the isosceles trapezoid with arithmetic sequence of angles
structure IsoscelesTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference in arithmetic sequence

-- Define properties of the isosceles trapezoid
def IsoscelesTrapezoid.properties (t : IsoscelesTrapezoid) : Prop :=
  -- Sum of angles in a quadrilateral is 360°
  t.a + (t.a + t.d) + (t.a + t.d) + (t.a + 3*t.d) = 360 ∧
  -- Largest angle is 150°
  t.a + 3*t.d = 150 ∧
  -- All angles are positive
  t.a > 0 ∧ t.d > 0

-- Theorem statement
theorem smallest_angle_measure (t : IsoscelesTrapezoid) 
  (h : t.properties) : ∃ (ε : ℝ), abs (t.a - 47) < ε ∧ ε > 0 := by
  sorry

#check smallest_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l942_94252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_theorem_l942_94258

/-- The volume of water flowing per minute in a river -/
noncomputable def water_flow_per_minute (depth width flow_rate : ℝ) : ℝ :=
  depth * width * (flow_rate * 1000 / 60)

/-- Theorem: Given a river with depth 3 m, width 36 m, and flow rate 2 kmph,
    the volume of water flowing per minute is approximately 3,599.64 cubic meters. -/
theorem river_flow_theorem :
  let depth : ℝ := 3
  let width : ℝ := 36
  let flow_rate : ℝ := 2
  ∃ ε > 0, |water_flow_per_minute depth width flow_rate - 3599.64| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check water_flow_per_minute 3 36 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_theorem_l942_94258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l942_94219

open Real

-- Define the line l₁
noncomputable def l₁ (α : ℝ) (x : ℝ) : ℝ :=
  tan α * x

-- Define the parabola C
def C (t : ℝ) : ℝ × ℝ :=
  (t^2, -2*t)

-- Define the area of triangle OAB
noncomputable def triangle_area (α : ℝ) : ℝ :=
  16 / |sin (2*α)|

theorem min_triangle_area :
  ∀ α : ℝ, 0 ≤ α ∧ α < π ∧ α ≠ π/2 →
  ∃ A B : ℝ × ℝ,
    A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    (∃ t : ℝ, C t = A) ∧
    (∃ t : ℝ, C t = B) ∧
    (∃ x : ℝ, (x, l₁ α x) = A) ∧
    (∃ β : ℝ, β = α + π/2 ∧ (∃ x : ℝ, (x, l₁ β x) = B)) →
  ∀ S : ℝ, S = triangle_area α → S ≥ 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l942_94219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l942_94255

noncomputable def f (x : ℝ) := x - Real.sin x

theorem max_value_of_f (x : ℝ) (h : x ∈ Set.Icc (π / 2) (3 * π / 2)) :
  f x ≤ f (3 * π / 2) ∧ f (3 * π / 2) = 3 * π / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l942_94255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l942_94231

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let angle := 120 * Real.pi / 180  -- Convert 120° to radians
  a = (1, 0) ∧
  Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 2 ∧
  a.1 * b.1 + a.2 * b.2 = Real.cos angle * Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2)) →
  Real.sqrt ((2 * a.1 + b.1) ^ 2 + (2 * a.2 + b.2) ^ 2) = 2

theorem vector_problem_theorem :
  ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

#check vector_problem_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l942_94231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l942_94251

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = Real.sqrt 2 / 10)
  (h2 : -π < α ∧ α < 0) : 
  Real.cos (α - π/4) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l942_94251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l942_94289

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = 3/5) (h4 : Real.cos (α + β) = -5/13) : Real.cos β = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l942_94289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_open_pos_reals_l942_94203

open Set
open Function
open Real

/-- The function f(x) = (1/2)ˣ -/
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

/-- The range of f is (0, +∞) -/
theorem range_of_f_is_open_pos_reals :
  range f = {y : ℝ | 0 < y} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_open_pos_reals_l942_94203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_area_l942_94267

/-- The area of the region consisting of all line segments of length 6 that are tangent to a circle of radius 3 at their midpoints -/
theorem tangent_segment_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 6) : 
  π * ((r * Real.sqrt 2)^2 - r^2) = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_area_l942_94267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017th_term_l942_94292

/-- Define the sequence term for a given position n -/
noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1 : ℝ)^(Int.floor ((n + 1 : ℝ) / 3)) * Real.sqrt (n + 1 : ℝ) / (2 : ℝ)^n

/-- The theorem stating that the 2017th term of the sequence is as expected -/
theorem sequence_2017th_term :
  sequence_term 2017 = -(Real.sqrt 2018) / 2^2017 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval sequence_term 2017

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017th_term_l942_94292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_line_equation_l942_94234

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the slope of a line --/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Calculates the y-intercept of a line --/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

/-- The line we're considering --/
def givenLine : Line := { a := 3, b := -1, c := 4 }

/-- Theorem: The given line has slope 3 and y-intercept 4 --/
theorem line_properties :
  givenLine.slope = 3 ∧ givenLine.yIntercept = 4 := by
  sorry

/-- Theorem: The equation 3x - y + 4 = 0 represents the given line --/
theorem line_equation (x y : ℝ) :
  3 * x - y + 4 = 0 ↔ givenLine.a * x + givenLine.b * y + givenLine.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_line_equation_l942_94234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_16_over_3_l942_94201

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 1
def g (x : ℝ) : ℝ := x - 3

-- Define the area of the figure
noncomputable def area : ℝ := ∫ x in (0)..(4), (f x - g x)

-- Theorem statement
theorem enclosed_area_is_16_over_3 : area = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_16_over_3_l942_94201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ali_possible_scores_l942_94296

/-- Represents the score for a single category -/
structure CategoryScore :=
  (correct : Nat)
  (bonus : Bool)

/-- Calculates the total score for a given list of category scores -/
def totalScore (scores : List CategoryScore) : Nat :=
  scores.foldl (fun acc s => acc + s.correct + (if s.bonus then 1 else 0)) 0

/-- Theorem stating the possible total scores for Ali's game -/
theorem ali_possible_scores :
  ∀ (scores : List CategoryScore),
    scores.length = 5 ∧
    scores.all (fun s => s.correct ≤ 3) ∧
    scores.foldl (fun acc s => acc + s.correct) 0 = 12 ∧
    scores.all (fun s => s.bonus ↔ s.correct = 3) →
    totalScore scores ∈ ({14, 15, 16} : Set Nat) :=
by
  sorry

#check ali_possible_scores

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ali_possible_scores_l942_94296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_bound_l942_94210

/-- Operation that replaces two numbers with their average divided by 4 -/
noncomputable def replace (a b : ℝ) : ℝ := (a + b) / 4

/-- The blackboard operation process -/
noncomputable def blackboard_process (n : ℕ) (initial_value : ℝ) : ℝ :=
  sorry

/-- Theorem: The final number on the blackboard is not less than 1/n -/
theorem final_number_bound (n : ℕ) (h : n > 0) : 
  blackboard_process n 1 ≥ (1 : ℝ) / n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_bound_l942_94210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l942_94220

def x : ℕ → ℚ
  | 0 => 2
  | n + 1 => (x n ^ 2 + 3 * x n + 6) / (x n + 4)

theorem sequence_convergence :
  ∃ m : ℕ, m ∈ Set.Icc 51 200 ∧ x m ≤ 2 + 1 / 2^10 ∧
  ∀ k : ℕ, k > 0 ∧ k < 51 → x k > 2 + 1 / 2^10 := by
  sorry

#check sequence_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l942_94220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_l942_94230

/-- Given a rhombus with area 21.46 cm² and one diagonal 7.4 cm, 
    the other diagonal is approximately 5.8 cm -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) :
  area = 21.46 →
  d1 = 7.4 →
  area = (d1 * d2) / 2 →
  |d2 - 5.8| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_l942_94230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l942_94299

def is_parabola (a b c d e f : ℤ) : Prop :=
  ∃ (k : ℚ) (v : ℤ), ∀ x y : ℚ, 
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 ↔ 
    x = k * (y - v)^2

theorem parabola_equation : 
  ∃ (a b c d e f : ℤ),
    -- The parabola passes through (2, 8)
    2 * 2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0 ∧ 
    -- The y-coordinate of the focus is 5
    ∃ (focus_x : ℚ), is_parabola a b c d e f ∧ 
    -- The axis of symmetry is parallel to the x-axis
    b = 0 ∧ 
    -- The vertex lies on the y-axis
    ∃ (vertex_y : ℤ), a * 0^2 + b * 0 * vertex_y + c * vertex_y^2 + d * 0 + e * vertex_y + f = 0 ∧
    -- a, b, c, d, e, f are integers (implied by their type ℤ)
    -- c is a positive integer
    c > 0 ∧ 
    -- gcd(|a|, |b|, |c|, |d|, |e|, |f|) = 1
    Int.gcd (a.natAbs) (Int.gcd (b.natAbs) (Int.gcd (c.natAbs) (Int.gcd (d.natAbs) (Int.gcd (e.natAbs) (f.natAbs))))) = 1 ∧
    -- The equation is 2x - 2y^2 + 20y - 50 = 0
    a = 0 ∧ b = 0 ∧ c = 2 ∧ d = 2 ∧ e = 20 ∧ f = -50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l942_94299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_value_l942_94240

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 + 1/x else -(((-x)^2) + 1/(-x))

-- State the theorem
theorem f_neg_two_value :
  (∀ x : ℝ, f (-x) = -(f x)) →  -- f is odd
  (∀ x > 0, f x = x^2 + 1/x) →  -- definition for positive x
  f (-2) = -9/2 := by
  intro h_odd h_pos
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_value_l942_94240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilized_egg_genetic_material_composition_l942_94223

-- Define the basic concepts
structure Organism : Type := (id : Nat)
inductive ReproductionMethod : Type
  | Sexual
  | Asexual

structure GeneticMaterial : Type := (amount : Nat)
structure Chromosome : Type := (count : Nat)

-- Define the properties
def meiosis (o : Organism) (r : ReproductionMethod) : Prop := 
  match r with
  | ReproductionMethod.Sexual => true
  | ReproductionMethod.Asexual => false

def recombination_occurs (o : Organism) : Prop := true
def constant_chromosome_number (o : Organism) : Prop := true

-- Define the components of a fertilized egg
def sperm_genetic_material : GeneticMaterial := ⟨50⟩
def egg_cell_genetic_material : GeneticMaterial := ⟨50⟩
def fertilized_egg_genetic_material : GeneticMaterial := ⟨100⟩

-- Define addition for GeneticMaterial
instance : Add GeneticMaterial where
  add a b := ⟨a.amount + b.amount⟩

-- Define the theorem
theorem fertilized_egg_genetic_material_composition :
  ∀ (o : Organism),
    meiosis o ReproductionMethod.Sexual →
    recombination_occurs o →
    constant_chromosome_number o →
    fertilized_egg_genetic_material ≠ sperm_genetic_material + egg_cell_genetic_material :=
by
  intro o meiosis_occurs recombination constant_chromosomes
  -- The proof is omitted for now
  sorry

#check fertilized_egg_genetic_material_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilized_egg_genetic_material_composition_l942_94223
