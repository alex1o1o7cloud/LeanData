import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_divisible_by_45_l422_42217

-- Define a function to represent a four-digit number of the form a43b
def number (a b : Nat) : Nat := a * 1000 + 430 + b

-- Define the divisibility condition
def isDivisibleBy45 (n : Nat) : Prop := n % 45 = 0

-- State the theorem
theorem four_digit_divisible_by_45 :
  ∀ a b : Nat, a < 10 ∧ b < 10 →
    (isDivisibleBy45 (number a b) ↔ (a = 2 ∧ b = 0) ∨ (a = 6 ∧ b = 5)) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_45_l422_42217


namespace NUMINAMATH_CALUDE_base_of_negative_four_cubed_l422_42251

def power_expression : ℤ → ℕ → ℤ := (·^·)

theorem base_of_negative_four_cubed :
  ∃ (base : ℤ), power_expression base 3 = power_expression (-4) 3 ∧ base = -4 :=
sorry

end NUMINAMATH_CALUDE_base_of_negative_four_cubed_l422_42251


namespace NUMINAMATH_CALUDE_units_digit_sum_base9_l422_42291

-- Define a function to convert a base-9 number to base-10
def base9ToBase10 (n : ℕ) : ℕ := 
  (n / 10) * 9 + (n % 10)

-- Define a function to get the units digit in base-9
def unitsDigitBase9 (n : ℕ) : ℕ := 
  n % 9

-- Theorem statement
theorem units_digit_sum_base9 :
  unitsDigitBase9 (base9ToBase10 35 + base9ToBase10 47) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base9_l422_42291


namespace NUMINAMATH_CALUDE_problem_structure_surface_area_l422_42290

/-- Represents the 3D structure composed of unit cubes -/
structure CubeStructure where
  base : Nat
  secondLayer : Nat
  column : Nat
  sideOne : Nat
  sideTwo : Nat

/-- Calculates the surface area of the given cube structure -/
def surfaceArea (s : CubeStructure) : Nat :=
  let frontBack := s.base + s.secondLayer + s.column + s.sideOne + s.sideTwo
  let top := (s.base - s.secondLayer) + s.secondLayer + 1 + s.sideOne + s.sideTwo
  let bottom := s.base
  2 * frontBack + top + bottom

/-- The specific cube structure described in the problem -/
def problemStructure : CubeStructure :=
  { base := 5
  , secondLayer := 3
  , column := 2
  , sideOne := 3
  , sideTwo := 2 }

/-- Theorem stating that the surface area of the problem structure is 62 -/
theorem problem_structure_surface_area :
  surfaceArea problemStructure = 62 := by sorry

end NUMINAMATH_CALUDE_problem_structure_surface_area_l422_42290


namespace NUMINAMATH_CALUDE_triangle_inequality_l422_42268

theorem triangle_inequality (a b x : ℝ) : 
  (a = 3 ∧ b = 5) → (2 < x ∧ x < 8) → 
  (a + b > x ∧ b + x > a ∧ x + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l422_42268


namespace NUMINAMATH_CALUDE_simplify_expression_l422_42246

theorem simplify_expression (a b c : ℝ) (ha : a = 37/5) (hb : b = 5/37) :
  1.6 * (((1/a + 1/b - 2*c/(a*b)) * (a + b + 2*c)) / (1/a^2 + 1/b^2 + 2/(a*b) - 4*c^2/(a^2*b^2))) = 1.6 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l422_42246


namespace NUMINAMATH_CALUDE_intersection_implies_values_l422_42278

/-- Sets T and S in the xy-plane -/
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

/-- The main theorem -/
theorem intersection_implies_values (a b : ℝ) :
  T a ∩ S b = {(2, 1)} → a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_values_l422_42278


namespace NUMINAMATH_CALUDE_equation_solution_l422_42240

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7))
  ∀ x : ℝ, f x = 1/8 ↔ x = 4 + Real.sqrt 57 ∨ x = 4 - Real.sqrt 57 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l422_42240


namespace NUMINAMATH_CALUDE_parallelogram_point_D_l422_42260

/-- A point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- A parallelogram in the complex plane -/
structure Parallelogram where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- The given parallelogram ABCD -/
def givenParallelogram : Parallelogram where
  A := { re := 4, im := 1 }
  B := { re := 3, im := 4 }
  C := { re := 5, im := 2 }
  D := { re := 6, im := -1 }

theorem parallelogram_point_D (p : Parallelogram) (h : p = givenParallelogram) :
  p.D.re = 6 ∧ p.D.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_point_D_l422_42260


namespace NUMINAMATH_CALUDE_craft_item_pricing_problem_l422_42248

/-- Represents the daily profit function for a craft item store -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem craft_item_pricing_problem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (price_reduction_1050 : ℝ) 
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : price_reduction_1050 < 40)
  (h4 : daily_profit initial_sales initial_profit price_reduction_1050 = 1050) :
  price_reduction_1050 = 25 ∧ 
  ∀ (price_reduction : ℝ), price_reduction < 40 → 
    daily_profit initial_sales initial_profit price_reduction ≠ 1600 := by
  sorry


end NUMINAMATH_CALUDE_craft_item_pricing_problem_l422_42248


namespace NUMINAMATH_CALUDE_three_operations_to_one_tile_l422_42288

/-- Represents the set of tiles -/
def TileSet := Finset Nat

/-- The operation of removing perfect squares and renumbering -/
def remove_squares_and_renumber (s : TileSet) : TileSet :=
  sorry

/-- The initial set of tiles from 1 to 49 -/
def initial_set : TileSet :=
  sorry

/-- Applies the operation n times -/
def apply_n_times (n : Nat) (s : TileSet) : TileSet :=
  sorry

theorem three_operations_to_one_tile :
  ∃ (n : Nat), n = 3 ∧ (apply_n_times n initial_set).card = 1 ∧
  ∀ (m : Nat), m < n → (apply_n_times m initial_set).card > 1 :=
sorry

end NUMINAMATH_CALUDE_three_operations_to_one_tile_l422_42288


namespace NUMINAMATH_CALUDE_robot_cost_calculation_l422_42202

def number_of_robots : ℕ := 7
def total_tax : ℚ := 7.22
def remaining_change : ℚ := 11.53
def initial_amount : ℚ := 80

theorem robot_cost_calculation (number_of_robots : ℕ) (total_tax : ℚ) (remaining_change : ℚ) (initial_amount : ℚ) :
  let total_spent : ℚ := initial_amount - remaining_change
  let robots_cost : ℚ := total_spent - total_tax
  let cost_per_robot : ℚ := robots_cost / number_of_robots
  cost_per_robot = 8.75 :=
by sorry

end NUMINAMATH_CALUDE_robot_cost_calculation_l422_42202


namespace NUMINAMATH_CALUDE_line_intersection_proof_l422_42261

theorem line_intersection_proof :
  ∃! (x y : ℚ), 5 * x - 3 * y = 17 ∧ 8 * x + 2 * y = 22 ∧ x = 50 / 17 ∧ y = -13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_proof_l422_42261


namespace NUMINAMATH_CALUDE_final_number_is_81_l422_42262

/-- Represents the elimination process on a list of numbers -/
def elimination_process (n : ℕ) : ℕ :=
  if n ≤ 3 then n else
  let m := elimination_process ((2 * n + 3) / 3)
  if m * 3 > n then m else m + 1

/-- The theorem stating that 81 is the final remaining number -/
theorem final_number_is_81 : elimination_process 200 = 81 := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_81_l422_42262


namespace NUMINAMATH_CALUDE_inequalities_proof_l422_42225

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l422_42225


namespace NUMINAMATH_CALUDE_lcm_812_3214_l422_42275

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1303402 := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_3214_l422_42275


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l422_42212

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l422_42212


namespace NUMINAMATH_CALUDE_min_sum_xy_l422_42218

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x - y - 10 = 0) :
  x + y ≥ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + x₀ - y₀ - 10 = 0 ∧ x₀ + y₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xy_l422_42218


namespace NUMINAMATH_CALUDE_lcm_72_98_l422_42242

theorem lcm_72_98 : Nat.lcm 72 98 = 3528 := by
  sorry

end NUMINAMATH_CALUDE_lcm_72_98_l422_42242


namespace NUMINAMATH_CALUDE_sarah_initial_cupcakes_l422_42257

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten : ℕ := 14

/-- The number of packages Sarah could make after Todd ate some cupcakes -/
def packages : ℕ := 3

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 8

/-- The initial number of cupcakes Sarah baked -/
def initial_cupcakes : ℕ := cupcakes_eaten + packages * cupcakes_per_package

theorem sarah_initial_cupcakes : initial_cupcakes = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_cupcakes_l422_42257


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l422_42237

/-- Represents the number of eggs in a container --/
def EggsPerContainer := 12

/-- Represents the number of containers with fewer eggs --/
def FewerEggsContainers := 3

/-- Represents the number of eggs in containers with fewer eggs --/
def EggsInFewerEggsContainers := 10

/-- Calculates the total number of eggs given the number of containers --/
def totalEggs (numContainers : ℕ) : ℕ :=
  numContainers * EggsPerContainer - FewerEggsContainers * (EggsPerContainer - EggsInFewerEggsContainers)

theorem smallest_number_of_eggs :
  ∃ (n : ℕ), (n > 100 ∧ totalEggs n = 102 ∧ ∀ m, m > 100 → totalEggs m ≥ 102) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l422_42237


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l422_42253

-- Define the square
def square_side_length : ℝ := 2

-- Define the number of regions the square is divided into
def num_regions : ℕ := 3

-- Define the theorem
theorem trapezoid_longer_side_length :
  ∀ (trapezoid_area pentagon_area : ℝ),
  trapezoid_area > 0 →
  pentagon_area > 0 →
  trapezoid_area = pentagon_area →
  trapezoid_area = (square_side_length ^ 2) / num_regions →
  ∃ (y : ℝ),
    y = 5 / 3 ∧
    trapezoid_area = (y + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l422_42253


namespace NUMINAMATH_CALUDE_correct_ingredients_l422_42263

/-- Recipe proportions and banana usage --/
structure RecipeData where
  flour_to_mush : ℚ  -- ratio of flour to banana mush
  sugar_to_mush : ℚ  -- ratio of sugar to banana mush
  milk_to_flour : ℚ  -- ratio of milk to flour
  bananas_per_mush : ℕ  -- number of bananas per cup of mush
  total_bananas : ℕ  -- total number of bananas used

/-- Calculated ingredients based on recipe data --/
def calculate_ingredients (r : RecipeData) : ℚ × ℚ × ℚ :=
  let mush := r.total_bananas / r.bananas_per_mush
  let flour := mush * r.flour_to_mush
  let sugar := mush * r.sugar_to_mush
  let milk := flour * r.milk_to_flour
  (flour, sugar, milk)

/-- Theorem stating the correct amounts of ingredients --/
theorem correct_ingredients (r : RecipeData) 
  (h1 : r.flour_to_mush = 3)
  (h2 : r.sugar_to_mush = 2/3)
  (h3 : r.milk_to_flour = 1/6)
  (h4 : r.bananas_per_mush = 4)
  (h5 : r.total_bananas = 32) :
  calculate_ingredients r = (24, 16/3, 4) := by
  sorry

#eval calculate_ingredients {
  flour_to_mush := 3,
  sugar_to_mush := 2/3,
  milk_to_flour := 1/6,
  bananas_per_mush := 4,
  total_bananas := 32
}

end NUMINAMATH_CALUDE_correct_ingredients_l422_42263


namespace NUMINAMATH_CALUDE_complex_expression_equality_l422_42255

theorem complex_expression_equality : (8 * 5.4 - 0.6 * 10 / 1.2) ^ 2 = 1459.24 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l422_42255


namespace NUMINAMATH_CALUDE_darias_current_money_l422_42235

/-- Calculates Daria's current money for concert tickets -/
theorem darias_current_money
  (ticket_cost : ℕ)  -- Cost of one ticket
  (num_tickets : ℕ)  -- Number of tickets Daria needs to buy
  (money_needed : ℕ) -- Additional money Daria needs to earn
  (h1 : ticket_cost = 90)
  (h2 : num_tickets = 4)
  (h3 : money_needed = 171) :
  ticket_cost * num_tickets - money_needed = 189 :=
by sorry

end NUMINAMATH_CALUDE_darias_current_money_l422_42235


namespace NUMINAMATH_CALUDE_solve_for_a_l422_42205

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), equation a ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l422_42205


namespace NUMINAMATH_CALUDE_tangency_implies_n_equals_two_l422_42252

/-- The value of n for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - n(y - 1)^2 = 1 -/
def tangency_value : ℝ := 2

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := x^2 - n*(y - 1)^2 = 1

/-- The ellipse and hyperbola are tangent -/
def are_tangent (n : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_ellipse x y ∧ is_on_hyperbola x y n ∧
  ∀ x' y' : ℝ, is_on_ellipse x' y' ∧ is_on_hyperbola x' y' n → (x', y') = (x, y)

theorem tangency_implies_n_equals_two :
  are_tangent tangency_value := by sorry

end NUMINAMATH_CALUDE_tangency_implies_n_equals_two_l422_42252


namespace NUMINAMATH_CALUDE_two_correct_relations_l422_42204

theorem two_correct_relations : 
  (0 ∈ ({0} : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {0}) ∧ 
  ¬({0, 1} ⊆ ({(0, 1)} : Set (ℕ × ℕ))) ∧ 
  ∀ a b : ℕ, {(a, b)} ≠ ({(b, a)} : Set (ℕ × ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_two_correct_relations_l422_42204


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l422_42282

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ (∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l422_42282


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l422_42221

theorem smallest_gcd_yz (x y z : ℕ+) 
  (hxy : Nat.gcd x.val y.val = 270)
  (hxz : Nat.gcd x.val z.val = 105) :
  ∃ (y' z' : ℕ+), 
    Nat.gcd y'.val z'.val = 15 ∧
    (∀ (y'' z'' : ℕ+), 
      Nat.gcd x.val y''.val = 270 → 
      Nat.gcd x.val z''.val = 105 → 
      Nat.gcd y''.val z''.val ≥ 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l422_42221


namespace NUMINAMATH_CALUDE_total_rope_length_l422_42283

/-- The original length of each rope -/
def rope_length : ℝ := 52

/-- The length used from the first rope -/
def used_first : ℝ := 42

/-- The length used from the second rope -/
def used_second : ℝ := 12

theorem total_rope_length :
  (rope_length - used_first) * 4 = rope_length - used_second →
  2 * rope_length = 104 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_length_l422_42283


namespace NUMINAMATH_CALUDE_choir_size_choir_size_is_30_l422_42203

/-- The number of singers in a school choir, given the initial number of robes,
    the cost per new robe, and the total amount spent on new robes. -/
theorem choir_size (initial_robes : ℕ) (cost_per_robe : ℕ) (total_spent : ℕ) : ℕ :=
  initial_robes + total_spent / cost_per_robe

/-- Proof that the number of singers in the choir is 30. -/
theorem choir_size_is_30 :
  choir_size 12 2 36 = 30 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_choir_size_is_30_l422_42203


namespace NUMINAMATH_CALUDE_num_plane_determining_pairs_eq_66_l422_42266

/-- A rectangular prism with distinct dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of edges in a rectangular prism -/
def num_edges : ℕ := 12

/-- The number of unordered pairs of parallel edges -/
def num_parallel_pairs : ℕ := 18

/-- The total number of unordered pairs of edges -/
def total_edge_pairs : ℕ := num_edges * (num_edges - 1) / 2

/-- The number of unordered pairs of edges that determine a plane -/
def num_plane_determining_pairs (prism : RectangularPrism) : ℕ :=
  total_edge_pairs

/-- Theorem: The number of unordered pairs of edges in a rectangular prism
    with distinct dimensions that determine a plane is 66 -/
theorem num_plane_determining_pairs_eq_66 (prism : RectangularPrism) :
  num_plane_determining_pairs prism = 66 := by
  sorry

end NUMINAMATH_CALUDE_num_plane_determining_pairs_eq_66_l422_42266


namespace NUMINAMATH_CALUDE_sqrt_equation_average_zero_l422_42267

theorem sqrt_equation_average_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 40}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  ∀ x ∈ solutions, x = x₁ ∨ x = x₂ ∧
  (x₁ + x₂) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_average_zero_l422_42267


namespace NUMINAMATH_CALUDE_zoe_recycled_pounds_l422_42215

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 8

/-- The number of pounds Zoe's friends recycled -/
def friends_pounds : ℕ := 23

/-- The total number of points earned -/
def total_points : ℕ := 6

/-- The number of pounds Zoe recycled -/
def zoe_pounds : ℕ := 25

theorem zoe_recycled_pounds :
  zoe_pounds + friends_pounds = pounds_per_point * total_points :=
sorry

end NUMINAMATH_CALUDE_zoe_recycled_pounds_l422_42215


namespace NUMINAMATH_CALUDE_circle_area_from_points_l422_42293

/-- The area of a circle with diameter endpoints at A(-1,3) and B'(13,12) is 277π/4 square units. -/
theorem circle_area_from_points :
  let A : ℝ × ℝ := (-1, 3)
  let B' : ℝ × ℝ := (13, 12)
  let diameter := Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  area = 277 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_points_l422_42293


namespace NUMINAMATH_CALUDE_cost_of_ingredients_for_two_cakes_l422_42247

/-- The cost of ingredients for two cakes given selling price, profit, and packaging cost -/
theorem cost_of_ingredients_for_two_cakes 
  (selling_price : ℝ) 
  (profit_per_cake : ℝ) 
  (packaging_cost : ℝ) : 
  selling_price = 15 → 
  profit_per_cake = 8 → 
  packaging_cost = 1 → 
  2 * selling_price - 2 * profit_per_cake - 2 * packaging_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_ingredients_for_two_cakes_l422_42247


namespace NUMINAMATH_CALUDE_library_problem_l422_42274

/-- Calculates the number of students helped on the first day given the total books,
    books per student, and students helped on subsequent days. -/
def students_helped_first_day (total_books : ℕ) (books_per_student : ℕ) 
    (students_day2 : ℕ) (students_day3 : ℕ) (students_day4 : ℕ) : ℕ :=
  (total_books - (students_day2 + students_day3 + students_day4) * books_per_student) / books_per_student

/-- Theorem stating that given the conditions in the problem, 
    the number of students helped on the first day is 4. -/
theorem library_problem : 
  students_helped_first_day 120 5 5 6 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_library_problem_l422_42274


namespace NUMINAMATH_CALUDE_sum_even_102_to_200_proof_l422_42294

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

/-- The first even integer in the range 102 to 200 -/
def first_even_102_to_200 : ℕ := 102

/-- The last even integer in the range 102 to 200 -/
def last_even_102_to_200 : ℕ := 200

theorem sum_even_102_to_200_proof :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (first_even_102_to_200 + last_even_102_to_200) :=
by sorry

end NUMINAMATH_CALUDE_sum_even_102_to_200_proof_l422_42294


namespace NUMINAMATH_CALUDE_rectangle_width_l422_42270

/-- Proves that the width of a rectangle is 5 cm, given the specified conditions -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →           -- Width is 2 cm longer than length
  width = 5 := by
sorry


end NUMINAMATH_CALUDE_rectangle_width_l422_42270


namespace NUMINAMATH_CALUDE_pyramid_min_faces_l422_42239

/-- A pyramid is a three-dimensional polyhedron with a polygonal base and triangular faces meeting at a common point (apex). -/
structure Pyramid where
  faces : ℕ

/-- Theorem: The number of faces in any pyramid is at least 4. -/
theorem pyramid_min_faces (p : Pyramid) : p.faces ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_min_faces_l422_42239


namespace NUMINAMATH_CALUDE_equation_equivalence_l422_42259

theorem equation_equivalence (x y : ℝ) : 2 * x - y = 3 ↔ y = 2 * x - 3 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l422_42259


namespace NUMINAMATH_CALUDE_sock_selection_l422_42229

theorem sock_selection (n k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_l422_42229


namespace NUMINAMATH_CALUDE_acetone_molecular_weight_proof_l422_42276

-- Define the isotopes and their properties
structure Isotope where
  mass : Float
  abundance : Float

-- Define the elements and their isotopes
def carbon_isotopes : List Isotope := [
  { mass := 12, abundance := 0.9893 },
  { mass := 13.003355, abundance := 0.0107 }
]

def hydrogen_isotopes : List Isotope := [
  { mass := 1.007825, abundance := 0.999885 },
  { mass := 2.014102, abundance := 0.000115 }
]

def oxygen_isotopes : List Isotope := [
  { mass := 15.994915, abundance := 0.99757 },
  { mass := 16.999132, abundance := 0.00038 },
  { mass := 17.999159, abundance := 0.00205 }
]

-- Function to calculate average atomic mass
def average_atomic_mass (isotopes : List Isotope) : Float :=
  isotopes.foldl (fun acc isotope => acc + isotope.mass * isotope.abundance) 0

-- Define the molecular formula of Acetone
def acetone_formula : List (Nat × List Isotope) := [
  (3, carbon_isotopes),
  (6, hydrogen_isotopes),
  (1, oxygen_isotopes)
]

-- Calculate the molecular weight of Acetone
def acetone_molecular_weight : Float :=
  acetone_formula.foldl (fun acc (n, isotopes) => acc + n.toFloat * average_atomic_mass isotopes) 0

-- Theorem statement
theorem acetone_molecular_weight_proof :
  (acetone_molecular_weight - 58.107055).abs < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_acetone_molecular_weight_proof_l422_42276


namespace NUMINAMATH_CALUDE_uncle_omar_parking_probability_l422_42226

/-- The number of parking spaces -/
def total_spaces : ℕ := 18

/-- The number of cars already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent empty spaces needed -/
def needed_spaces : ℕ := 2

/-- The probability of finding the required adjacent empty spaces -/
def parking_probability : ℚ := 16/51

theorem uncle_omar_parking_probability :
  (1 : ℚ) - (Nat.choose (total_spaces - needed_spaces + 1) parked_cars : ℚ) / 
  (Nat.choose total_spaces parked_cars : ℚ) = parking_probability := by
  sorry

end NUMINAMATH_CALUDE_uncle_omar_parking_probability_l422_42226


namespace NUMINAMATH_CALUDE_min_radius_point_l422_42216

/-- The point that minimizes the radius of a circle centered at the origin -/
theorem min_radius_point (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_radius_point_l422_42216


namespace NUMINAMATH_CALUDE_exactly_two_good_probability_l422_42241

def total_screws : ℕ := 10
def defective_screws : ℕ := 3
def drawn_screws : ℕ := 4

def probability_exactly_two_good : ℚ :=
  (Nat.choose (total_screws - defective_screws) 2 * Nat.choose defective_screws 2) /
  Nat.choose total_screws drawn_screws

theorem exactly_two_good_probability :
  probability_exactly_two_good = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_good_probability_l422_42241


namespace NUMINAMATH_CALUDE_sqrt_negative_undefined_l422_42254

theorem sqrt_negative_undefined : ¬ ∃ (x : ℝ), x^2 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_undefined_l422_42254


namespace NUMINAMATH_CALUDE_chicken_katsu_cost_is_25_l422_42243

/-- The cost of the chicken katsu given the following conditions:
  - The family ordered a smoky salmon for $40, a black burger for $15, and a chicken katsu.
  - The bill includes a 10% service charge and 5% tip.
  - Mr. Arevalo paid with $100 and received $8 in change.
-/
def chicken_katsu_cost : ℝ :=
  let salmon_cost : ℝ := 40
  let burger_cost : ℝ := 15
  let service_charge_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.05
  let total_paid : ℝ := 100
  let change_received : ℝ := 8
  let total_bill : ℝ := total_paid - change_received
  25

theorem chicken_katsu_cost_is_25 :
  chicken_katsu_cost = 25 := by sorry

end NUMINAMATH_CALUDE_chicken_katsu_cost_is_25_l422_42243


namespace NUMINAMATH_CALUDE_square_inequality_not_sufficient_nor_necessary_l422_42233

theorem square_inequality_not_sufficient_nor_necessary (x y : ℝ) :
  ¬(∀ x y : ℝ, x^2 > y^2 → x > y) ∧ ¬(∀ x y : ℝ, x > y → x^2 > y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_not_sufficient_nor_necessary_l422_42233


namespace NUMINAMATH_CALUDE_quartic_root_sum_l422_42271

/-- Given a quartic equation px^4 + qx^3 + rx^2 + sx + t = 0 with roots 4, -3, and 0, 
    and p ≠ 0, prove that (q+r)/p = -13 -/
theorem quartic_root_sum (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 4^4 + q * 4^3 + r * 4^2 + s * 4 + t = 0)
  (h2 : p * (-3)^4 + q * (-3)^3 + r * (-3)^2 + s * (-3) + t = 0)
  (h3 : t = 0) : 
  (q + r) / p = -13 := by
  sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l422_42271


namespace NUMINAMATH_CALUDE_bakery_tart_flour_calculation_l422_42269

theorem bakery_tart_flour_calculation 
  (initial_tarts : ℕ) 
  (new_tarts : ℕ) 
  (initial_flour_per_tart : ℚ) 
  (h1 : initial_tarts = 36)
  (h2 : new_tarts = 18)
  (h3 : initial_flour_per_tart = 1 / 12)
  : (initial_tarts : ℚ) * initial_flour_per_tart / new_tarts = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_tart_flour_calculation_l422_42269


namespace NUMINAMATH_CALUDE_ages_sum_l422_42264

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l422_42264


namespace NUMINAMATH_CALUDE_coin_flip_probability_l422_42272

theorem coin_flip_probability (n : ℕ) (k : ℕ) (h : n = 4 ∧ k = 3) :
  (2 : ℚ) / (2^n : ℚ) = 1/8 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l422_42272


namespace NUMINAMATH_CALUDE_equation_solution_l422_42224

theorem equation_solution : 
  let x : ℚ := -7/6
  (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l422_42224


namespace NUMINAMATH_CALUDE_whale_first_hour_consumption_l422_42238

/-- Represents the whale's plankton consumption pattern --/
structure WhaleConsumption where
  duration : Nat
  hourlyIncrease : Nat
  totalConsumption : Nat
  sixthHourConsumption : Nat

/-- Calculates the first hour's consumption given the whale's consumption pattern --/
def firstHourConsumption (w : WhaleConsumption) : Nat :=
  w.sixthHourConsumption - (w.hourlyIncrease * 5)

/-- Theorem stating that for the given whale consumption pattern, 
    the first hour's consumption is 38 kilos --/
theorem whale_first_hour_consumption 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.hourlyIncrease = 3)
  (h3 : w.totalConsumption = 450)
  (h4 : w.sixthHourConsumption = 53) : 
  firstHourConsumption w = 38 := by
  sorry

#eval firstHourConsumption ⟨9, 3, 450, 53⟩

end NUMINAMATH_CALUDE_whale_first_hour_consumption_l422_42238


namespace NUMINAMATH_CALUDE_not_all_fractions_integer_l422_42219

theorem not_all_fractions_integer 
  (a b c r s t : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : a * b + 1 = r^2)
  (eq2 : a * c + 1 = s^2)
  (eq3 : b * c + 1 = t^2) :
  ¬(∃ (x y z : ℕ), (r * t : ℚ) / s = x ∧ (r * s : ℚ) / t = y ∧ (s * t : ℚ) / r = z) :=
sorry

end NUMINAMATH_CALUDE_not_all_fractions_integer_l422_42219


namespace NUMINAMATH_CALUDE_point_transformation_l422_42213

-- Define the point type
def Point := ℝ × ℝ × ℝ

-- Define the transformations
def reflect_yz (p : Point) : Point :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_z_90 (p : Point) : Point :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : Point) : Point :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_180 (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, -z)

def reflect_xz (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, z)

-- Define the composition of all transformations
def transform (p : Point) : Point :=
  p |> reflect_yz
    |> rotate_z_90
    |> reflect_xy
    |> rotate_x_180
    |> reflect_xz
    |> rotate_z_90

-- Theorem statement
theorem point_transformation :
  transform (2, 2, 2) = (2, 2, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l422_42213


namespace NUMINAMATH_CALUDE_total_distance_is_55km_l422_42214

/-- Represents the distances Ivan ran on each day of the week -/
structure RunningDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validRunningSchedule (d : RunningDistances) : Prop :=
  d.tuesday = 2 * d.monday ∧
  d.wednesday = d.tuesday / 2 ∧
  d.thursday = d.wednesday / 2 ∧
  d.friday = 2 * d.thursday ∧
  d.thursday = 5 -- The shortest distance is 5 km, which occurs on Thursday

/-- The theorem to prove -/
theorem total_distance_is_55km (d : RunningDistances) 
  (h : validRunningSchedule d) : 
  d.monday + d.tuesday + d.wednesday + d.thursday + d.friday = 55 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_is_55km_l422_42214


namespace NUMINAMATH_CALUDE_cylinder_volume_l422_42201

/-- Given a cylinder with height 2 and lateral surface area 4π, its volume is 2π -/
theorem cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) : 
  h = 2 → lateral_area = 4 * Real.pi → volume = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l422_42201


namespace NUMINAMATH_CALUDE_alice_win_condition_l422_42256

/-- The game state represents the positions of the red and blue pieces -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- Alice's move function -/
def move (r : ℚ) (state : GameState) (k : ℤ) : GameState :=
  { red := state.red,
    blue := state.red + r^k * (state.blue - state.red) }

/-- Alice can win the game -/
def can_win (r : ℚ) : Prop :=
  ∃ (moves : List ℤ), moves.length ≤ 2021 ∧
    (moves.foldl (move r) { red := 0, blue := 1 }).red = 1

/-- The main theorem stating the condition for Alice to win -/
theorem alice_win_condition (r : ℚ) : 
  (r > 1 ∧ can_win r) ↔ (∃ d : ℕ, d ≥ 1 ∧ d ≤ 1010 ∧ r = 1 + 1 / d) := by
  sorry


end NUMINAMATH_CALUDE_alice_win_condition_l422_42256


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l422_42200

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 1 →  -- a_1 = 1
  (a 2) * (a 5) = (a 4)^2 →  -- a_2, a_4, and a_5 form a geometric sequence
  d = 1/5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l422_42200


namespace NUMINAMATH_CALUDE_money_sharing_l422_42295

theorem money_sharing (emani howard jamal : ℕ) (h1 : emani = 150) (h2 : emani = howard + 30) (h3 : jamal = 75) :
  (emani + howard + jamal) / 3 = 115 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l422_42295


namespace NUMINAMATH_CALUDE_negative_sum_l422_42250

theorem negative_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + c < 0 := by
sorry

end NUMINAMATH_CALUDE_negative_sum_l422_42250


namespace NUMINAMATH_CALUDE_chef_michel_pies_l422_42220

/-- Calculates the total number of pies sold given the number of slices per pie and the number of slices ordered --/
def total_pies_sold (shepherds_pie_slices : ℕ) (chicken_pot_pie_slices : ℕ) 
                    (shepherds_pie_ordered : ℕ) (chicken_pot_pie_ordered : ℕ) : ℕ :=
  (shepherds_pie_ordered / shepherds_pie_slices) + (chicken_pot_pie_ordered / chicken_pot_pie_slices)

/-- Proves that Chef Michel sold 29 pies in total --/
theorem chef_michel_pies : 
  total_pies_sold 4 5 52 80 = 29 := by
  sorry

end NUMINAMATH_CALUDE_chef_michel_pies_l422_42220


namespace NUMINAMATH_CALUDE_product_of_distinct_numbers_l422_42207

theorem product_of_distinct_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_numbers_l422_42207


namespace NUMINAMATH_CALUDE_cauchy_schwarz_two_terms_l422_42297

theorem cauchy_schwarz_two_terms
  (a b x y : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hx : 0 < x)
  (hy : 0 < y) :
  a * x + b * y ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_two_terms_l422_42297


namespace NUMINAMATH_CALUDE_old_toilet_water_usage_l422_42211

/-- The amount of water saved by switching to a new toilet in June -/
def water_saved : ℝ := 1800

/-- The number of times the toilet is flushed per day -/
def flushes_per_day : ℕ := 15

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The percentage of water saved by the new toilet compared to the old one -/
def water_saving_percentage : ℝ := 0.8

theorem old_toilet_water_usage : ℝ :=
  let total_flushes : ℕ := flushes_per_day * days_in_june
  let water_saved_per_flush : ℝ := water_saved / total_flushes
  water_saved_per_flush / water_saving_percentage

#check @old_toilet_water_usage

end NUMINAMATH_CALUDE_old_toilet_water_usage_l422_42211


namespace NUMINAMATH_CALUDE_fish_value_is_three_and_three_quarters_l422_42227

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish_to_bread : ℚ) (bread_to_rice : ℚ) : ℚ :=
  (fish_to_bread * bread_to_rice)⁻¹

/-- Theorem stating that one fish is worth 3¾ bags of rice given the trade rates -/
theorem fish_value_is_three_and_three_quarters :
  fish_value (4/5) 3 = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_is_three_and_three_quarters_l422_42227


namespace NUMINAMATH_CALUDE_liam_paid_more_than_ellen_l422_42284

-- Define the pizza characteristics
def total_slices : ℕ := 12
def plain_pizza_cost : ℚ := 12
def extra_cheese_cost : ℚ := 3
def extra_cheese_slices : ℕ := total_slices / 3

-- Define what Liam and Ellen ate
def liam_extra_cheese_slices : ℕ := extra_cheese_slices
def liam_plain_slices : ℕ := 4
def ellen_plain_slices : ℕ := total_slices - liam_extra_cheese_slices - liam_plain_slices

-- Calculate total pizza cost
def total_pizza_cost : ℚ := plain_pizza_cost + extra_cheese_cost

-- Calculate cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate what Liam and Ellen paid
def liam_payment : ℚ := cost_per_slice * (liam_extra_cheese_slices + liam_plain_slices)
def ellen_payment : ℚ := (plain_pizza_cost / total_slices) * ellen_plain_slices

-- Theorem to prove
theorem liam_paid_more_than_ellen : liam_payment - ellen_payment = 6 := by
  sorry

end NUMINAMATH_CALUDE_liam_paid_more_than_ellen_l422_42284


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l422_42222

theorem carpet_area_calculation (room_length room_width wardrobe_side feet_per_yard : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_width = 12)
  (h3 : wardrobe_side = 3)
  (h4 : feet_per_yard = 3) : 
  (room_length * room_width - wardrobe_side * wardrobe_side) / (feet_per_yard * feet_per_yard) = 23 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l422_42222


namespace NUMINAMATH_CALUDE_equation_roots_l422_42232

theorem equation_roots (a b : ℝ) :
  -- Part 1
  (∃ x : ℂ, x = 1 - Complex.I * Real.sqrt 3 ∧ x / a + b / x = 1) →
  a = 2 ∧ b = 2
  ∧
  -- Part 2
  (b / a > 1 / 4 ∧ a > 0) →
  ¬∃ x : ℝ, x / a + b / x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l422_42232


namespace NUMINAMATH_CALUDE_complex_number_location_l422_42280

theorem complex_number_location : 
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) + i ^ 2012 = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l422_42280


namespace NUMINAMATH_CALUDE_mike_peaches_picked_l422_42287

/-- The number of peaches Mike picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem mike_peaches_picked : 
  peaches_picked 34 86 = 52 := by sorry

end NUMINAMATH_CALUDE_mike_peaches_picked_l422_42287


namespace NUMINAMATH_CALUDE_unit_segments_bound_l422_42228

/-- 
Given n distinct points in a plane, τ(n) represents the number of 
unit-length segments joining pairs of these points.
-/
def τ (n : ℕ) : ℕ := sorry

/-- 
Theorem: The number of unit-length segments joining pairs of n distinct 
points in a plane is at most n²/3.
-/
theorem unit_segments_bound (n : ℕ) : τ n ≤ n^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_segments_bound_l422_42228


namespace NUMINAMATH_CALUDE_largest_divisible_by_six_under_9000_l422_42245

theorem largest_divisible_by_six_under_9000 : 
  ∀ n : ℕ, n < 9000 ∧ 6 ∣ n → n ≤ 8994 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_by_six_under_9000_l422_42245


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l422_42236

theorem triangle_arctan_sum (a b c : ℝ) (h : c = a + b) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.arctan (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l422_42236


namespace NUMINAMATH_CALUDE_candy_distribution_l422_42299

theorem candy_distribution (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, n * k + 1 = 120) → 
  n = 7 ∨ n = 17 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l422_42299


namespace NUMINAMATH_CALUDE_hexagonal_table_dice_probability_l422_42249

/-- The number of people seated around the hexagonal table -/
def num_people : ℕ := 6

/-- The number of sides on the standard die -/
def die_sides : ℕ := 6

/-- A function to calculate the number of valid options for each person's roll -/
def valid_options (person : ℕ) : ℕ :=
  match person with
  | 1 => 6  -- Person A
  | 2 => 5  -- Person B
  | 3 => 4  -- Person C
  | 4 => 5  -- Person D
  | 5 => 3  -- Person E
  | 6 => 3  -- Person F
  | _ => 0  -- Invalid person number

/-- The probability of no two adjacent or opposite people rolling the same number -/
def probability : ℚ :=
  (valid_options 1 * valid_options 2 * valid_options 3 * valid_options 4 * valid_options 5 * valid_options 6) /
  (die_sides ^ num_people)

theorem hexagonal_table_dice_probability :
  probability = 25 / 648 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_table_dice_probability_l422_42249


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l422_42279

theorem quadratic_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 ∧
                (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 ∧
                x₁ * x₂ < 0 ∧
                x₁ < 0 ∧ x₂ > 0 ∧
                abs x₁ > x₂) →
  m > -3 ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l422_42279


namespace NUMINAMATH_CALUDE_basketball_lineups_l422_42231

/-- The number of players in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of different starting lineups that can be chosen -/
def num_lineups : ℕ := 95040

/-- Theorem: The number of different starting lineups that can be chosen
    from a team of 12 players for 5 distinct positions is 95,040 -/
theorem basketball_lineups :
  (team_size.factorial) / ((team_size - lineup_size).factorial) = num_lineups := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineups_l422_42231


namespace NUMINAMATH_CALUDE_house_transactions_result_l422_42258

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  hasHouse : Bool

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  (FinancialState.mk (buyer.cash - price) true, FinancialState.mk (seller.cash + price) false)

/-- The main theorem to prove -/
theorem house_transactions_result :
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 16000 false
  let (a1, b1) := houseTransaction initialB initialA 16000
  let (a2, b2) := houseTransaction a1 b1 14000
  let (a3, b3) := houseTransaction b2 a2 17000
  a3.cash = 34000 ∧ b3.cash = -3000 := by
  sorry

#check house_transactions_result

end NUMINAMATH_CALUDE_house_transactions_result_l422_42258


namespace NUMINAMATH_CALUDE_time_to_paint_one_room_l422_42223

/-- Given a painting job with a total number of rooms, rooms already painted,
    and time to paint the remaining rooms, calculate the time to paint one room. -/
theorem time_to_paint_one_room
  (total_rooms : ℕ)
  (painted_rooms : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_rooms = 10)
  (h2 : painted_rooms = 8)
  (h3 : time_for_remaining = 16)
  (h4 : painted_rooms < total_rooms) :
  (time_for_remaining : ℚ) / (total_rooms - painted_rooms : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_room_l422_42223


namespace NUMINAMATH_CALUDE_danny_steve_time_ratio_l422_42292

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 29

/-- The time it takes Steve to reach Danny's house -/
def steve_time : ℝ := 58

/-- The difference in time it takes Steve and Danny to reach the halfway point -/
def halfway_time_difference : ℝ := 14.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + halfway_time_difference :=
by sorry

end NUMINAMATH_CALUDE_danny_steve_time_ratio_l422_42292


namespace NUMINAMATH_CALUDE_division_multiplication_result_l422_42265

theorem division_multiplication_result : (-6) / (-6) * (-1/6 : ℚ) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l422_42265


namespace NUMINAMATH_CALUDE_track_extension_calculation_l422_42230

/-- Theorem: Track Extension Calculation
Given a train track with an elevation gain of 600 meters,
changing the gradient from 3% to 2% results in a track extension of 10 km. -/
theorem track_extension_calculation (elevation_gain : ℝ) (initial_gradient : ℝ) (final_gradient : ℝ) :
  elevation_gain = 600 →
  initial_gradient = 0.03 →
  final_gradient = 0.02 →
  (elevation_gain / final_gradient - elevation_gain / initial_gradient) / 1000 = 10 := by
  sorry

#check track_extension_calculation

end NUMINAMATH_CALUDE_track_extension_calculation_l422_42230


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l422_42234

/-- The volume of snow to be shoveled from a walkway -/
def snow_volume (total_length width depth no_shovel_length : ℝ) : ℝ :=
  (total_length - no_shovel_length) * width * depth

/-- Proof that the volume of snow to be shoveled is 46.875 cubic feet -/
theorem snow_volume_calculation : 
  snow_volume 30 2.5 0.75 5 = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l422_42234


namespace NUMINAMATH_CALUDE_unique_solution_system_l422_42296

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1 ∧
  2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1 ∧
  2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1 →
  x = (1 + Real.sqrt 5) / 2 ∧
  y = (1 + Real.sqrt 5) / 2 ∧
  z = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l422_42296


namespace NUMINAMATH_CALUDE_trig_identity_l422_42210

theorem trig_identity (θ : Real) 
  (h : (1 - Real.cos θ) / (4 + Real.sin θ ^ 2) = 1 / 2) : 
  (4 + Real.cos θ ^ 3) * (3 + Real.sin θ ^ 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l422_42210


namespace NUMINAMATH_CALUDE_min_distance_to_line_l422_42277

theorem min_distance_to_line : 
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ a b : ℝ, a + 2*b = Real.sqrt 5 → Real.sqrt (a^2 + b^2) ≥ d) ∧
  (∃ a b : ℝ, a + 2*b = Real.sqrt 5 ∧ Real.sqrt (a^2 + b^2) = d) ∧
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l422_42277


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l422_42206

/-- The equation 3(5 + dx) = 15x + 15 has infinitely many solutions for x if and only if d = 5 -/
theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l422_42206


namespace NUMINAMATH_CALUDE_equation_solution_l422_42285

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l422_42285


namespace NUMINAMATH_CALUDE_candy_ratio_l422_42244

theorem candy_ratio (adam james rubert : ℕ) : 
  adam = 6 →
  james = 3 * adam →
  adam + james + rubert = 96 →
  rubert = 4 * james :=
by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l422_42244


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l422_42273

theorem same_solution_implies_k_equals_one :
  ∀ k : ℝ,
  (∀ x : ℝ, 4*x + 3*k = 2*x + 2 ↔ 2*x + k = 5*x + 2.5) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l422_42273


namespace NUMINAMATH_CALUDE_repair_easier_than_thermometer_l422_42209

def word1 : String := "термометр"
def word2 : String := "ремонт"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem repair_easier_than_thermometer :
  (uniqueLetters word2).card > (uniqueLetters word1).card := by
  sorry

end NUMINAMATH_CALUDE_repair_easier_than_thermometer_l422_42209


namespace NUMINAMATH_CALUDE_max_value_abc_max_value_abc_achievable_l422_42208

theorem max_value_abc (A B C : ℕ) (h : A + B + C = 15) :
  (A * B * C + A * B + B * C + C * A) ≤ 200 :=
by sorry

theorem max_value_abc_achievable :
  ∃ (A B C : ℕ), A + B + C = 15 ∧ A * B * C + A * B + B * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_max_value_abc_achievable_l422_42208


namespace NUMINAMATH_CALUDE_matrix_value_equation_l422_42286

theorem matrix_value_equation (x : ℝ) : 
  (3 * x) * (4 * x) - 2 * (2 * x) = 6 ↔ x = -1/3 ∨ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_value_equation_l422_42286


namespace NUMINAMATH_CALUDE_f_max_value_l422_42298

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℕ) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1 / 50) ∧ (∃ n : ℕ, f n = 1 / 50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l422_42298


namespace NUMINAMATH_CALUDE_environmental_policy_support_percentage_l422_42281

theorem environmental_policy_support_percentage : 
  let total_surveyed : ℕ := 150 + 850
  let men_surveyed : ℕ := 150
  let women_surveyed : ℕ := 850
  let men_support_percentage : ℚ := 70 / 100
  let women_support_percentage : ℚ := 75 / 100
  let men_supporters : ℚ := men_surveyed * men_support_percentage
  let women_supporters : ℚ := women_surveyed * women_support_percentage
  let total_supporters : ℚ := men_supporters + women_supporters
  let overall_support_percentage : ℚ := total_supporters / total_surveyed * 100
  overall_support_percentage = 743 / 10 := by sorry

end NUMINAMATH_CALUDE_environmental_policy_support_percentage_l422_42281


namespace NUMINAMATH_CALUDE_trig_equation_roots_l422_42289

open Real

theorem trig_equation_roots (α β : ℝ) : 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π →
  (∃ x y : ℝ, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ x = tan α ∧ y = tan β) →
  tan (α + β) = -1 ∧ cos (α - β) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_roots_l422_42289
