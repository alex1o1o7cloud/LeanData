import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l1382_138293

theorem divisibility_implies_gcd_greater_than_one
  (a b c d : ℕ+)
  (h : (a.val * c.val + b.val * d.val) % (a.val^2 + b.val^2) = 0) :
  Nat.gcd (c.val^2 + d.val^2) (a.val^2 + b.val^2) > 1 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l1382_138293


namespace NUMINAMATH_CALUDE_unscreened_percentage_l1382_138240

def tv_width : ℝ := 6
def tv_height : ℝ := 5
def screen_width : ℝ := 5
def screen_height : ℝ := 4

theorem unscreened_percentage :
  (tv_width * tv_height - screen_width * screen_height) / (tv_width * tv_height) * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unscreened_percentage_l1382_138240


namespace NUMINAMATH_CALUDE_log_equation_solution_l1382_138264

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 7 →
  x = 3 ^ (14 / 3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1382_138264


namespace NUMINAMATH_CALUDE_trapezoid_to_square_l1382_138286

/-- An isosceles trapezoid with given dimensions can be rearranged into a square -/
theorem trapezoid_to_square (b₁ b₂ h : ℝ) (h₁ : b₁ = 4) (h₂ : b₂ = 12) (h₃ : h = 4) :
  ∃ (s : ℝ), (b₁ + b₂) * h / 2 = s^2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_to_square_l1382_138286


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l1382_138292

theorem angle_measure_theorem (x : ℝ) : 
  (90 - x) = (180 - x) - 4 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l1382_138292


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1382_138241

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x*(x+1)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1382_138241


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l1382_138206

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (a : ℤ) + b ≥ 45) ∧
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l1382_138206


namespace NUMINAMATH_CALUDE_equation_is_linear_l1382_138250

/-- A linear equation in two variables is of the form Ax + By = C, where A, B, and C are constants, and A and B are not both zero. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ), (A ≠ 0 ∨ B ≠ 0) ∧ ∀ x y, f x y ↔ A * x + B * y = C

/-- The equation 3x - 1 = 2 - 5y is a linear equation in two variables. -/
theorem equation_is_linear : IsLinearEquationInTwoVariables (fun x y ↦ 3 * x - 1 = 2 - 5 * y) := by
  sorry

#check equation_is_linear

end NUMINAMATH_CALUDE_equation_is_linear_l1382_138250


namespace NUMINAMATH_CALUDE_right_triangle_equivalence_l1382_138247

theorem right_triangle_equivalence (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a^3 + b^3 + c^3 = a*b*(a+b) - b*c*(b+c) + a*c*(a+c)) ↔
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_equivalence_l1382_138247


namespace NUMINAMATH_CALUDE_total_ants_l1382_138228

theorem total_ants (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  duke = abe / 2 →
  abe + beth + cece + duke = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_l1382_138228


namespace NUMINAMATH_CALUDE_contractor_problem_l1382_138289

/-- Represents the efficiency of a worker --/
structure WorkerEfficiency where
  value : ℝ
  pos : value > 0

/-- Represents a group of workers with the same efficiency --/
structure WorkerGroup where
  count : ℕ
  efficiency : WorkerEfficiency

/-- Calculates the total work done by a group of workers in a day --/
def dailyWork (group : WorkerGroup) : ℝ :=
  group.count * group.efficiency.value

/-- Calculates the total work done by multiple groups of workers in a day --/
def totalDailyWork (groups : List WorkerGroup) : ℝ :=
  groups.map dailyWork |>.sum

/-- The contractor problem --/
theorem contractor_problem 
  (initialGroups : List WorkerGroup)
  (initialDays : ℕ)
  (totalDays : ℕ)
  (firedLessEfficient : ℕ)
  (firedMoreEfficient : ℕ)
  (h_initial_groups : initialGroups = [
    { count := 15, efficiency := { value := 1, pos := by sorry } },
    { count := 10, efficiency := { value := 1.5, pos := by sorry } }
  ])
  (h_initial_days : initialDays = 40)
  (h_total_days : totalDays = 150)
  (h_fired_less : firedLessEfficient = 4)
  (h_fired_more : firedMoreEfficient = 3)
  (h_one_third_complete : totalDailyWork initialGroups * initialDays = (1/3) * (totalDailyWork initialGroups * totalDays))
  : ∃ (remainingDays : ℕ), remainingDays = 112 ∧ 
    (totalDailyWork initialGroups * totalDays) = 
    (totalDailyWork initialGroups * initialDays + 
     totalDailyWork [
       { count := 15 - firedLessEfficient, efficiency := { value := 1, pos := by sorry } },
       { count := 10 - firedMoreEfficient, efficiency := { value := 1.5, pos := by sorry } }
     ] * remainingDays) := by
  sorry


end NUMINAMATH_CALUDE_contractor_problem_l1382_138289


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_zero_l1382_138291

/-- The slope angle of the tangent line to y = x^2 - x at (1, 0) is 45° -/
theorem tangent_slope_angle_at_one_zero :
  let f : ℝ → ℝ := λ x => x^2 - x
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let slope : ℝ := deriv f x₀
  Real.arctan slope * (180 / Real.pi) = 45 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_zero_l1382_138291


namespace NUMINAMATH_CALUDE_food_waste_scientific_notation_l1382_138208

theorem food_waste_scientific_notation :
  (500 : ℝ) * 1000000000 = 5 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_food_waste_scientific_notation_l1382_138208


namespace NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l1382_138215

-- Define a geometric figure
def GeometricFigure := Type

-- Define a translation operation
def translate (F : GeometricFigure) (v : ℝ × ℝ) : GeometricFigure := sorry

-- Define properties of a figure
def shape (F : GeometricFigure) : Type := sorry
def size (F : GeometricFigure) : ℝ := sorry

-- Theorem: Translation preserves shape and size
theorem translation_preserves_shape_and_size (F : GeometricFigure) (v : ℝ × ℝ) :
  (shape (translate F v) = shape F) ∧ (size (translate F v) = size F) := by sorry

end NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l1382_138215


namespace NUMINAMATH_CALUDE_copper_percentage_bounds_l1382_138290

/-- Represents an alloy composition -/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three given alloys -/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The theorem stating the bounds on copper percentage in the new alloy -/
theorem copper_percentage_bounds (x₁ x₂ x₃ : ℝ) 
  (sum_to_one : x₁ + x₂ + x₃ = 1)
  (manganese_constraint : 0.9 * x₂ + 0.6 * x₃ = 0.4) :
  let copper_percentage := 0.7 * x₁ + 0.1 * x₂ + 0.25 * x₃
  0.4 ≤ copper_percentage ∧ copper_percentage ≤ 13/30 := by
  sorry


end NUMINAMATH_CALUDE_copper_percentage_bounds_l1382_138290


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1382_138235

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1382_138235


namespace NUMINAMATH_CALUDE_balanced_colorings_count_l1382_138223

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
def Grid := List Cell

/-- Checks if a 2x2 subgrid is balanced -/
def isBalanced2x2 (grid : Grid) (startRow startCol : Nat) : Bool :=
  sorry

/-- Checks if the entire grid is balanced -/
def isBalancedGrid (grid : Grid) : Bool :=
  sorry

/-- Counts the number of balanced colorings for an 8x6 grid -/
def countBalancedColorings : Nat :=
  sorry

/-- The main theorem stating the number of balanced colorings -/
theorem balanced_colorings_count :
  countBalancedColorings = 1896 :=
sorry

end NUMINAMATH_CALUDE_balanced_colorings_count_l1382_138223


namespace NUMINAMATH_CALUDE_sphere_radius_equals_cone_lateral_area_l1382_138238

theorem sphere_radius_equals_cone_lateral_area 
  (cone_height : ℝ) 
  (cone_base_radius : ℝ) 
  (sphere_radius : ℝ) :
  cone_height = 3 →
  cone_base_radius = 4 →
  (4 * Real.pi * sphere_radius^2) = (Real.pi * cone_base_radius * (cone_height^2 + cone_base_radius^2).sqrt) →
  sphere_radius = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_equals_cone_lateral_area_l1382_138238


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l1382_138259

/-- The total number of pencils Anthony has after receiving more from Kathryn -/
theorem anthony_total_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 9 → received = 56 → total = initial + received → total = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l1382_138259


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1382_138219

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in an m x n grid -/
def total_rectangles (m n : ℕ) : ℕ :=
  m * rectangles_in_row n + n * rectangles_in_row m - m * n

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1382_138219


namespace NUMINAMATH_CALUDE_min_value_of_S_l1382_138243

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve C representing the trajectory of the circle's center -/
def C (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The dot product of two vectors represented by points -/
def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The function S to be minimized -/
noncomputable def S (a b : Point) : ℝ :=
  let o : Point := ⟨0, 0⟩
  let f : Point := ⟨1, 0⟩
  triangle_area o f a + triangle_area o a b

/-- The main theorem stating the minimum value of S -/
theorem min_value_of_S :
  ∀ a b : Point,
  C a → C b →
  dot_product a b = -4 →
  S a b ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_S_l1382_138243


namespace NUMINAMATH_CALUDE_time_calculation_l1382_138234

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define the initial time
def initial_time : Time := { hours := 8, minutes := 45, seconds := 0 }

-- Define the number of seconds to add
def seconds_to_add : Nat := 9876

-- Define the expected final time
def expected_final_time : Time := { hours := 11, minutes := 29, seconds := 36 }

-- Function to add seconds to a given time
def add_seconds (t : Time) (s : Nat) : Time :=
  sorry

-- Theorem to prove
theorem time_calculation :
  add_seconds initial_time seconds_to_add = expected_final_time :=
sorry

end NUMINAMATH_CALUDE_time_calculation_l1382_138234


namespace NUMINAMATH_CALUDE_roberto_outfits_l1382_138239

/-- The number of trousers Roberto has -/
def num_trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def num_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def num_jackets : ℕ := 2

/-- An outfit consists of a pair of trousers, a shirt, and a jacket -/
def outfit := ℕ × ℕ × ℕ

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets

theorem roberto_outfits : total_outfits = 80 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1382_138239


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l1382_138202

/-- Given a trapezium with the following properties:
  - One parallel side is 20 cm long
  - The distance between parallel sides is 17 cm
  - The area is 323 square centimeters
  Prove that the length of the other parallel side is 18 cm -/
theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 17 → area = 323 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l1382_138202


namespace NUMINAMATH_CALUDE_difference_of_squares_l1382_138275

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (product_eq : x * y = 99) :
  x^2 - y^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1382_138275


namespace NUMINAMATH_CALUDE_max_score_is_94_l1382_138268

/-- Represents an operation that can be applied to a number -/
inductive Operation
  | Add : Operation
  | Square : Operation

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Add => n + 1
  | Operation.Square => n * n

/-- Applies a sequence of operations to a starting number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Calculates the minimum distance from a number to any perfect square -/
def minDistanceToPerfectSquare (n : ℕ) : ℕ :=
  let sqrtFloor := (n.sqrt : ℕ)
  let sqrtCeil := sqrtFloor + 1
  min (n - sqrtFloor * sqrtFloor) (sqrtCeil * sqrtCeil - n)

/-- The main theorem -/
theorem max_score_is_94 :
  (∃ (ops : List Operation),
    ops.length = 100 ∧
    minDistanceToPerfectSquare (applyOperations 0 ops) = 94) ∧
  (∀ (ops : List Operation),
    ops.length = 100 →
    minDistanceToPerfectSquare (applyOperations 0 ops) ≤ 94) :=
  sorry


end NUMINAMATH_CALUDE_max_score_is_94_l1382_138268


namespace NUMINAMATH_CALUDE_chicken_problem_l1382_138229

/-- The number of chickens Colten has -/
def colten : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar : ℕ := 3 * colten - 4

/-- The number of chickens Quentin has -/
def quentin : ℕ := 2 * skylar + 25

/-- The total number of chickens -/
def total : ℕ := 383

theorem chicken_problem :
  colten + skylar + quentin = total :=
sorry

end NUMINAMATH_CALUDE_chicken_problem_l1382_138229


namespace NUMINAMATH_CALUDE_largest_in_set_l1382_138207

def a : ℝ := -4

def S : Set ℝ := {-3 * a, 4 * a, 24 / a, a^2, 2 * a + 1, 1}

theorem largest_in_set : ∀ x ∈ S, x ≤ a^2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l1382_138207


namespace NUMINAMATH_CALUDE_banana_arrangements_l1382_138269

/-- The number of ways to arrange letters in a word -/
def arrange_letters (total : ℕ) (freq1 freq2 freq3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial freq1 * Nat.factorial freq2 * Nat.factorial freq3)

/-- Theorem: The number of arrangements of BANANA is 60 -/
theorem banana_arrangements :
  arrange_letters 6 1 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1382_138269


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1382_138242

theorem quadratic_roots_problem (α β k : ℝ) : 
  (α^2 - α + k - 1 = 0) →
  (β^2 - β + k - 1 = 0) →
  (α^2 - 2*α - β = 4) →
  (k = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1382_138242


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1382_138218

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1382_138218


namespace NUMINAMATH_CALUDE_stone_length_calculation_l1382_138227

/-- Calculates the length of stones used to pave a hall -/
theorem stone_length_calculation (hall_length hall_width : ℕ) 
  (stone_width num_stones : ℕ) (stone_length : ℚ) : 
  hall_length = 36 ∧ 
  hall_width = 15 ∧ 
  stone_width = 5 ∧ 
  num_stones = 5400 ∧
  (hall_length * 10 * hall_width * 10 : ℚ) = stone_length * stone_width * num_stones →
  stone_length = 2 := by
sorry

end NUMINAMATH_CALUDE_stone_length_calculation_l1382_138227


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1382_138270

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) : 
  x^2 - 6*x*y + 9*y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1382_138270


namespace NUMINAMATH_CALUDE_quadratic_root_sum_minus_product_l1382_138299

theorem quadratic_root_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 5 = 0 → 
  x₂^2 - 3*x₂ - 5 = 0 → 
  x₁ + x₂ - x₁ * x₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_minus_product_l1382_138299


namespace NUMINAMATH_CALUDE_f_simplification_f_specific_value_l1382_138254

noncomputable def f (α : Real) : Real :=
  (Real.sin (4 * Real.pi - α) * Real.cos (Real.pi - α) * Real.cos ((3 * Real.pi) / 2 + α) * Real.cos ((7 * Real.pi) / 2 - α)) /
  (Real.cos (Real.pi + α) * Real.sin (2 * Real.pi - α) * Real.sin (Real.pi + α) * Real.sin ((9 * Real.pi) / 2 - α))

theorem f_simplification (α : Real) : f α = Real.tan α := by sorry

theorem f_specific_value : f (-(31 / 6) * Real.pi) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_specific_value_l1382_138254


namespace NUMINAMATH_CALUDE_max_ticket_types_for_specific_car_l1382_138274

/-- Represents a one-way traveling car with stations and capacity. -/
structure TravelingCar where
  num_stations : Nat
  capacity : Nat

/-- Calculates the maximum number of different ticket types that can be sold. -/
def max_ticket_types (car : TravelingCar) : Nat :=
  let total_possible_tickets := (car.num_stations - 1) * car.num_stations / 2
  let max_non_overlapping_tickets := ((car.num_stations + 1) / 2) ^ 2
  let unsellable_tickets := max_non_overlapping_tickets - car.capacity
  total_possible_tickets - unsellable_tickets

/-- Theorem stating the maximum number of different ticket types for a specific car configuration. -/
theorem max_ticket_types_for_specific_car :
  let car := TravelingCar.mk 14 25
  max_ticket_types car = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_ticket_types_for_specific_car_l1382_138274


namespace NUMINAMATH_CALUDE_probability_of_specific_tile_arrangement_l1382_138263

theorem probability_of_specific_tile_arrangement :
  let total_tiles : ℕ := 6
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 2
  let specific_arrangement := [true, true, false, true, false, true]
  
  (x_tiles + o_tiles = total_tiles) →
  (List.length specific_arrangement = total_tiles) →
  
  (probability_of_arrangement : ℚ) =
    (x_tiles.choose 2 * o_tiles.choose 1 * x_tiles.choose 1 * o_tiles.choose 1 * x_tiles.choose 1) /
    total_tiles.factorial →
  
  probability_of_arrangement = 1 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_tile_arrangement_l1382_138263


namespace NUMINAMATH_CALUDE_least_cookies_l1382_138295

theorem least_cookies (n : ℕ) : n = 59 ↔ 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 6 ∧
  ∀ m : ℕ, m > 0 → m % 6 = 5 → m % 8 = 3 → m % 9 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_cookies_l1382_138295


namespace NUMINAMATH_CALUDE_divisible_by_three_l1382_138267

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (∃ k : ℕ, n = 6*k + 1 ∨ n = 6*k + 2) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_three_l1382_138267


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_l1382_138237

/-- A function that returns true if a number is a two-digit whole number --/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that returns the product of digits of a two-digit number --/
def digitProduct (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem stating that 62 is the greatest two-digit number whose digits have a product of 12 --/
theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, isTwoDigit n → digitProduct n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_l1382_138237


namespace NUMINAMATH_CALUDE_gear_system_rotation_l1382_138284

/-- Represents the rotation direction of a gear -/
inductive Direction
| Clockwise
| Counterclockwise

/-- Represents a system of gears -/
structure GearSystem :=
  (n : ℕ)  -- number of gears

/-- Returns the direction of the i-th gear in the system -/
def gear_direction (sys : GearSystem) (i : ℕ) : Direction :=
  if i % 2 = 0 then Direction.Counterclockwise else Direction.Clockwise

/-- Checks if the gear system can rotate -/
def can_rotate (sys : GearSystem) : Prop :=
  sys.n % 2 = 0

theorem gear_system_rotation (sys : GearSystem) :
  can_rotate sys ↔ sys.n % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_gear_system_rotation_l1382_138284


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1382_138296

theorem fractional_equation_solution_range (a x : ℝ) : 
  ((a + 2) / (x + 1) = 1) ∧ 
  (x ≤ 0) ∧ 
  (x + 1 ≠ 0) → 
  (a ≤ -1) ∧ (a ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1382_138296


namespace NUMINAMATH_CALUDE_quadratic_inequality_transformation_l1382_138212

-- Define the quadratic function and its solution set
def quadratic_inequality (a b : ℝ) := {x : ℝ | x^2 + a*x + b > 0}

-- Define the given solution set
def given_solution_set := {x : ℝ | x < -3 ∨ x > 1}

-- Define the transformed quadratic inequality
def transformed_inequality (a b : ℝ) := {x : ℝ | a*x^2 + b*x - 2 < 0}

-- Define the expected solution set
def expected_solution_set := {x : ℝ | -1/2 < x ∧ x < 2}

-- Theorem statement
theorem quadratic_inequality_transformation 
  (h : quadratic_inequality a b = given_solution_set) :
  transformed_inequality a b = expected_solution_set := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_transformation_l1382_138212


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_three_lines_l1382_138281

/-- A line in a plane --/
structure Line where
  -- Add necessary fields to represent a line

/-- An equilateral triangle --/
structure EquilateralTriangle where
  -- Add necessary fields to represent an equilateral triangle

/-- A point in a plane --/
structure Point where
  -- Add necessary fields to represent a point

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Checks if a triangle is equilateral --/
def isEquilateralTriangle (t : EquilateralTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem equilateral_triangle_on_three_lines 
  (d₁ d₂ d₃ : Line) : 
  ∃ (t : EquilateralTriangle), 
    isEquilateralTriangle t ∧ 
    (∃ (p₁ p₂ p₃ : Point), 
      pointOnLine p₁ d₁ ∧ 
      pointOnLine p₂ d₂ ∧ 
      pointOnLine p₃ d₃ ∧ 
      -- Add conditions to relate p₁, p₂, p₃ to the vertices of t
      sorry) :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_on_three_lines_l1382_138281


namespace NUMINAMATH_CALUDE_monotonic_intervals_and_comparison_l1382_138216

noncomputable def f (x : ℝ) : ℝ := 3 * Real.exp x + x^2
noncomputable def g (x : ℝ) : ℝ := 9*x - 1
noncomputable def φ (x : ℝ) : ℝ := x * Real.exp x + 4*x - f x

theorem monotonic_intervals_and_comparison :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < Real.log 2 → φ x₁ < φ x₂) ∧
  (∀ x₁ x₂, Real.log 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → φ x₁ > φ x₂) ∧
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → φ x₁ < φ x₂) ∧
  (∀ x, f x > g x) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_intervals_and_comparison_l1382_138216


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l1382_138245

theorem subtraction_of_negatives : -5 - (-2) = -3 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l1382_138245


namespace NUMINAMATH_CALUDE_data_median_and_mode_l1382_138294

def data : List Int := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

def median (l : List Int) : ℚ := sorry

def mode (l : List Int) : Int := sorry

theorem data_median_and_mode :
  median data = 14.5 ∧ mode data = 17 := by sorry

end NUMINAMATH_CALUDE_data_median_and_mode_l1382_138294


namespace NUMINAMATH_CALUDE_abc_value_l1382_138272

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 15 * Real.sqrt 3)
  (hbc : b * c = 21 * Real.sqrt 3)
  (hac : a * c = 10 * Real.sqrt 3) :
  a * b * c = 15 * Real.sqrt 42 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1382_138272


namespace NUMINAMATH_CALUDE_xe_exp_increasing_l1382_138265

/-- The function f(x) = xe^x is increasing for all x > 0 -/
theorem xe_exp_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => x * Real.exp x) := by sorry

end NUMINAMATH_CALUDE_xe_exp_increasing_l1382_138265


namespace NUMINAMATH_CALUDE_gourmet_smores_night_cost_l1382_138203

/-- The cost of supplies for a gourmet S'mores night -/
def cost_of_smores_night (num_people : ℕ) (smores_per_person : ℕ) 
  (graham_cracker_cost : ℚ) (marshmallow_cost : ℚ) (chocolate_cost : ℚ)
  (caramel_cost : ℚ) (toffee_cost : ℚ) : ℚ :=
  let total_smores := num_people * smores_per_person
  let cost_per_smore := graham_cracker_cost + marshmallow_cost + chocolate_cost + 
                        2 * caramel_cost + 4 * toffee_cost
  total_smores * cost_per_smore

/-- Theorem: The cost of supplies for the gourmet S'mores night is $26.40 -/
theorem gourmet_smores_night_cost :
  cost_of_smores_night 8 3 (10/100) (15/100) (25/100) (20/100) (5/100) = 2640/100 :=
by sorry

end NUMINAMATH_CALUDE_gourmet_smores_night_cost_l1382_138203


namespace NUMINAMATH_CALUDE_odd_not_divides_power_plus_one_l1382_138220

theorem odd_not_divides_power_plus_one (n m : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ¬(n ∣ (m^(n-1) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_odd_not_divides_power_plus_one_l1382_138220


namespace NUMINAMATH_CALUDE_gcd_2024_2048_l1382_138224

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2024_2048_l1382_138224


namespace NUMINAMATH_CALUDE_printing_presses_theorem_l1382_138277

/-- The number of printing presses used in the first scenario -/
def P : ℕ := 35

/-- The time taken (in hours) by P presses to print 500,000 papers -/
def time_P : ℕ := 15

/-- The number of presses used in the second scenario -/
def presses_2 : ℕ := 25

/-- The time taken (in hours) by presses_2 to print 500,000 papers -/
def time_2 : ℕ := 21

theorem printing_presses_theorem :
  P * time_P = presses_2 * time_2 :=
sorry

end NUMINAMATH_CALUDE_printing_presses_theorem_l1382_138277


namespace NUMINAMATH_CALUDE_BC_time_is_three_hours_l1382_138214

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 4
def work_rate_C : ℚ := 1 / 12

-- Define the combined work rate of A and C
def work_rate_AC : ℚ := 1 / 3

-- Define the time taken by B and C together
def time_BC : ℚ := 1 / (work_rate_B + work_rate_C)

-- Theorem statement
theorem BC_time_is_three_hours :
  work_rate_A = 1 / 4 →
  work_rate_B = 1 / 4 →
  work_rate_AC = 1 / 3 →
  work_rate_C = work_rate_AC - work_rate_A →
  time_BC = 3 := by
  sorry


end NUMINAMATH_CALUDE_BC_time_is_three_hours_l1382_138214


namespace NUMINAMATH_CALUDE_john_received_120_l1382_138266

/-- The amount of money John received from his grandpa -/
def grandpa_amount : ℕ := 30

/-- The amount of money John received from his grandma -/
def grandma_amount : ℕ := 3 * grandpa_amount

/-- The total amount of money John received from both grandparents -/
def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_received_120_l1382_138266


namespace NUMINAMATH_CALUDE_football_game_spectators_l1382_138209

theorem football_game_spectators (total_wristbands : ℕ) (wristbands_per_person : ℕ) 
  (h1 : total_wristbands = 290)
  (h2 : wristbands_per_person = 2)
  : total_wristbands / wristbands_per_person = 145 := by
  sorry

end NUMINAMATH_CALUDE_football_game_spectators_l1382_138209


namespace NUMINAMATH_CALUDE_range_of_a_l1382_138205

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |2*a - 1| ≤ |x + 1/x|) ↔ -1/2 ≤ a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1382_138205


namespace NUMINAMATH_CALUDE_equipment_percentage_transportation_degrees_l1382_138221

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℝ
  research_development : ℝ
  utilities : ℝ
  supplies : ℝ
  transportation : ℝ
  equipment : ℝ

/-- The theorem stating the correct percentage for equipment in the budget -/
theorem equipment_percentage (b : BudgetAllocation) : b.equipment = 4 :=
  by
    have h1 : b.salaries = 60 := by sorry
    have h2 : b.research_development = 9 := by sorry
    have h3 : b.utilities = 5 := by sorry
    have h4 : b.supplies = 2 := by sorry
    have h5 : b.transportation = 20 := by sorry
    have h6 : b.salaries + b.research_development + b.utilities + b.supplies + b.transportation + b.equipment = 100 := by sorry
    sorry

/-- The function to calculate the degrees in a circle graph for a given percentage -/
def percentToDegrees (percent : ℝ) : ℝ := 3.6 * percent

/-- The theorem stating that 72 degrees represent the transportation budget -/
theorem transportation_degrees (b : BudgetAllocation) : percentToDegrees b.transportation = 72 :=
  by sorry

end NUMINAMATH_CALUDE_equipment_percentage_transportation_degrees_l1382_138221


namespace NUMINAMATH_CALUDE_avg_people_moving_rounded_l1382_138248

/-- The number of people moving to Texas in two days -/
def people_moving : ℕ := 1500

/-- The number of days -/
def days : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculate the average number of people moving to Texas per minute -/
def avg_people_per_minute : ℚ :=
  people_moving / (days * hours_per_day * minutes_per_hour)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_moving_rounded :
  round_to_nearest avg_people_per_minute = 1 := by sorry

end NUMINAMATH_CALUDE_avg_people_moving_rounded_l1382_138248


namespace NUMINAMATH_CALUDE_basketball_free_throw_probability_l1382_138273

theorem basketball_free_throw_probability (player_A_prob player_B_prob : ℝ) 
  (h1 : player_A_prob = 0.7)
  (h2 : player_B_prob = 0.6)
  (h3 : 0 ≤ player_A_prob ∧ player_A_prob ≤ 1)
  (h4 : 0 ≤ player_B_prob ∧ player_B_prob ≤ 1) :
  1 - (1 - player_A_prob) * (1 - player_B_prob) = 0.88 := by
  sorry


end NUMINAMATH_CALUDE_basketball_free_throw_probability_l1382_138273


namespace NUMINAMATH_CALUDE_log_sum_equality_l1382_138210

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 * Real.log 2 / Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1382_138210


namespace NUMINAMATH_CALUDE_additional_tickets_needed_l1382_138287

def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def current_tickets : ℕ := 5

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

theorem additional_tickets_needed : 
  (total_cost - current_tickets : ℕ) = 8 := by sorry

end NUMINAMATH_CALUDE_additional_tickets_needed_l1382_138287


namespace NUMINAMATH_CALUDE_price_per_dozen_eggs_l1382_138257

/-- Calculates the price per dozen eggs given the number of chickens, eggs per chicken per week,
    eggs per dozen, total revenue, and number of weeks. -/
theorem price_per_dozen_eggs 
  (num_chickens : ℕ) 
  (eggs_per_chicken_per_week : ℕ) 
  (eggs_per_dozen : ℕ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : eggs_per_chicken_per_week = 6)
  (h3 : eggs_per_dozen = 12)
  (h4 : total_revenue = 552)
  (h5 : num_weeks = 8) :
  total_revenue / (num_chickens * eggs_per_chicken_per_week * num_weeks / eggs_per_dozen) = 3 := by
  sorry

end NUMINAMATH_CALUDE_price_per_dozen_eggs_l1382_138257


namespace NUMINAMATH_CALUDE_three_is_primitive_root_l1382_138233

theorem three_is_primitive_root (n : ℕ) (p : ℕ) (h1 : n > 1) (h2 : p = 2^n + 1) (h3 : Nat.Prime p) :
  IsPrimitiveRoot 3 p := by
  sorry

end NUMINAMATH_CALUDE_three_is_primitive_root_l1382_138233


namespace NUMINAMATH_CALUDE_range_of_a_l1382_138204

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * Real.exp x + a * Real.exp (-x)

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * |x - 2|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 6 ∧ ∀ x ∈ s, f a x = g a x) →
  1 < a ∧ a < Real.exp 2 / (2 * Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1382_138204


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l1382_138230

/-- Given a cyclist who travels two segments with different distances and speeds, 
    this theorem proves that the average speed for the entire trip is 18 miles per hour. -/
theorem cyclist_average_speed : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
    d₁ = 45 ∧ d₂ = 15 ∧ v₁ = 15 ∧ v₂ = 45 →
    (d₁ + d₂) / ((d₁ / v₁) + (d₂ / v₂)) = 18 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_average_speed_l1382_138230


namespace NUMINAMATH_CALUDE_biology_score_calculation_l1382_138261

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 75
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_scores_sum := math_score + science_score + social_studies_score + english_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 85 := by
  sorry

#check biology_score_calculation

end NUMINAMATH_CALUDE_biology_score_calculation_l1382_138261


namespace NUMINAMATH_CALUDE_min_value_sum_l1382_138252

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  a + 2*b ≥ 3 + 2*Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + b₀) / (a₀ * b₀) = 1 ∧ a₀ + 2*b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1382_138252


namespace NUMINAMATH_CALUDE_sign_of_f_m_plus_one_indeterminate_l1382_138211

/-- Given a quadratic function f(x) = x^2 - x + a and the condition that f(-m) < 0,
    prove that the sign of f(m+1) cannot be determined without additional information about m. -/
theorem sign_of_f_m_plus_one_indeterminate 
  (f : ℝ → ℝ) (a m : ℝ) 
  (h1 : ∀ x, f x = x^2 - x + a) 
  (h2 : f (-m) < 0) : 
  ∃ m₁ m₂, f (m₁ + 1) > 0 ∧ f (m₂ + 1) < 0 :=
sorry

end NUMINAMATH_CALUDE_sign_of_f_m_plus_one_indeterminate_l1382_138211


namespace NUMINAMATH_CALUDE_existence_of_inverse_solvable_problems_l1382_138226

/-- A mathematical problem that can be solved by first considering its inverse -/
structure InverseSolvableProblem where
  problem : Type
  inverse_problem : Type
  solve : inverse_problem → problem

/-- Theorem stating that there exist problems solvable by first solving their inverse -/
theorem existence_of_inverse_solvable_problems :
  ∃ (P : InverseSolvableProblem), True :=
  sorry

end NUMINAMATH_CALUDE_existence_of_inverse_solvable_problems_l1382_138226


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1382_138260

theorem quadratic_real_roots (k d : ℝ) (h : k ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + k^2 + d = 0) ↔ d ≤ -3/4 * k^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1382_138260


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1382_138280

/-- The hyperbola and parabola intersection problem -/
theorem hyperbola_parabola_intersection
  (a : ℝ) (P F₁ F₂ : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_hyperbola : 3 * P.1^2 - P.2^2 = 3 * a^2)
  (h_parabola : P.2^2 = 8 * a * P.1)
  (h_F₁ : F₁ = (-2*a, 0))
  (h_F₂ : F₂ = (2*a, 0))
  (h_distance : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
                Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12) :
  ∃ (x : ℝ), x = -2 ∧ x = -a/2 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1382_138280


namespace NUMINAMATH_CALUDE_race_length_l1382_138246

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The race satisfies the given conditions -/
def validRace (r : Race) : Prop :=
  r.samTime = 17 ∧
  r.johnTime = r.samTime + 5 ∧
  r.headStart = 15 ∧
  r.length / r.samTime = (r.length - r.headStart) / r.johnTime

/-- The theorem to be proved -/
theorem race_length (r : Race) (h : validRace r) : r.length = 66 := by
  sorry

end NUMINAMATH_CALUDE_race_length_l1382_138246


namespace NUMINAMATH_CALUDE_y_derivative_l1382_138251

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt ((Real.tan x + Real.sqrt (2 * Real.tan x) + 1) / (Real.tan x - Real.sqrt (2 * Real.tan x) + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = 0 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1382_138251


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1382_138225

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  monotone_increasing : Monotone a
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  condition1 : a 2 * a 5 = 6
  condition2 : a 3 + a 4 = 5

/-- The common ratio of the geometric sequence is 3/2 -/
theorem geometric_sequence_common_ratio (seq : GeometricSequence) :
  seq.a 2 / seq.a 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1382_138225


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l1382_138255

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- Definition of point M -/
def M : ℝ × ℝ := (0, 2)

/-- Definition of the dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Statement of the theorem -/
theorem ellipse_dot_product_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  ∃ (k : ℝ), P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ∧
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l1382_138255


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1382_138213

theorem min_value_trigonometric_expression (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) :
  (1 / (Real.sin A ^ 2 * Real.cos B ^ 4) + 
   1 / (Real.sin B ^ 2 * Real.cos C ^ 4) + 
   1 / (Real.sin C ^ 2 * Real.cos A ^ 4)) ≥ 81/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1382_138213


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1382_138298

-- Define a line type
structure Line where
  slope : ℝ
  yIntercept : ℝ

-- Define the line from the problem
def problemLine : Line := { slope := -1, yIntercept := 1 }

-- Define the third quadrant
def thirdQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 < 0 ∧ p.2 < 0}

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ (x y : ℝ), (y = problemLine.slope * x + problemLine.yIntercept) →
  (x, y) ∉ thirdQuadrant :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1382_138298


namespace NUMINAMATH_CALUDE_not_perfect_square_floor_theorem_l1382_138278

theorem not_perfect_square_floor_theorem (A : ℕ) (h : ¬ ∃ k : ℕ, A = k ^ 2) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_floor_theorem_l1382_138278


namespace NUMINAMATH_CALUDE_sqrt_equation_l1382_138244

theorem sqrt_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l1382_138244


namespace NUMINAMATH_CALUDE_consecutive_even_squares_l1382_138279

theorem consecutive_even_squares (x : ℕ) : 
  (x % 2 = 0) → (x^2 - (x-2)^2 = 2012) → x = 504 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_squares_l1382_138279


namespace NUMINAMATH_CALUDE_austins_change_l1382_138297

/-- The amount of change Austin had left after buying robots --/
def change_left (num_robots : ℕ) (robot_cost tax initial_amount : ℚ) : ℚ :=
  initial_amount - (num_robots * robot_cost + tax)

/-- Theorem stating that Austin's change is $11.53 --/
theorem austins_change :
  change_left 7 8.75 7.22 80 = 11.53 := by
  sorry

end NUMINAMATH_CALUDE_austins_change_l1382_138297


namespace NUMINAMATH_CALUDE_toys_sold_l1382_138276

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 16800 →
  gain = 3 * cost_price →
  cost_price = 800 →
  (selling_price - gain) / cost_price = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l1382_138276


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1382_138217

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b + 2*a*b = 8) :
  ∀ x y, x > 0 → y > 0 → x + 2*y + 2*x*y = 8 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1382_138217


namespace NUMINAMATH_CALUDE_lg_sum_five_two_l1382_138253

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_five_two : lg 5 + lg 2 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_sum_five_two_l1382_138253


namespace NUMINAMATH_CALUDE_library_visitors_proof_l1382_138262

/-- The total number of visitors to a library in a week -/
def total_visitors (monday : ℕ) (tuesday_multiplier : ℕ) (remaining_days : ℕ) (avg_remaining : ℕ) : ℕ :=
  monday + (tuesday_multiplier * monday) + (remaining_days * avg_remaining)

/-- Theorem stating that the total number of visitors to the library in a week is 250 -/
theorem library_visitors_proof : 
  total_visitors 50 2 5 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_proof_l1382_138262


namespace NUMINAMATH_CALUDE_wedge_volume_of_sphere_l1382_138285

/-- The volume of a wedge of a sphere -/
theorem wedge_volume_of_sphere (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → 
  num_wedges = 6 → 
  (1 / num_wedges : ℝ) * (4 / 3 : ℝ) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_of_sphere_l1382_138285


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l1382_138258

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the problem of fitting small blocks into a larger box -/
structure BlockFittingProblem where
  box : Dimensions
  block : Dimensions

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (volume p.box) / (volume p.block)

/-- Determines if the arrangement of blocks is physically possible -/
def isPhysicallyPossible (p : BlockFittingProblem) (n : ℕ) : Prop :=
  (p.block.width = p.box.width) ∧
  (2 * p.block.length ≤ p.box.length) ∧
  (((n / 4) * p.block.height) ≤ p.box.height) ∧
  ((n % 4) * p.block.height ≤ p.box.height - ((n / 4) * p.block.height))

theorem max_blocks_in_box (p : BlockFittingProblem) 
  (h1 : p.box = Dimensions.mk 4 3 5)
  (h2 : p.block = Dimensions.mk 1 3 2) :
  ∃ (n : ℕ), n = 10 ∧ 
    (maxBlocksByVolume p = n) ∧ 
    (isPhysicallyPossible p n) ∧
    (∀ m : ℕ, m > n → ¬(isPhysicallyPossible p m)) := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l1382_138258


namespace NUMINAMATH_CALUDE_square_difference_503_496_l1382_138256

theorem square_difference_503_496 : 503^2 - 496^2 = 6993 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_503_496_l1382_138256


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1382_138271

/-- Parabola type representing y^2 = ax -/
structure Parabola where
  a : ℝ
  hpos : a > 0

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line with slope k -/
structure Line where
  k : ℝ

def intersect_parabola_line (p : Parabola) (l : Line) : Point × Point := sorry

def extend_line (p1 p2 : Point) : Line := sorry

def slope_of_line (p1 p2 : Point) : ℝ := sorry

theorem parabola_intersection_theorem (p : Parabola) (m : Point) 
  (h_m : m.x = 4 ∧ m.y = 0) (l : Line) (k2 : ℝ) 
  (h_k : l.k = Real.sqrt 2 * k2) :
  let f := Point.mk (p.a / 4) 0
  let (a, b) := intersect_parabola_line p l
  let c := intersect_parabola_line p (extend_line a m)
  let d := intersect_parabola_line p (extend_line b m)
  slope_of_line c.1 d.1 = k2 → p.a = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1382_138271


namespace NUMINAMATH_CALUDE_mikes_tire_spending_l1382_138249

/-- The problem of calculating Mike's spending on new tires -/
theorem mikes_tire_spending (total_spent : ℚ) (speaker_cost : ℚ) (tire_cost : ℚ) :
  total_spent = 224.87 →
  speaker_cost = 118.54 →
  tire_cost = total_spent - speaker_cost →
  tire_cost = 106.33 := by
  sorry

end NUMINAMATH_CALUDE_mikes_tire_spending_l1382_138249


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1382_138288

/-- Represents a seating arrangement in an examination room --/
structure ExamRoom :=
  (rows : Nat)
  (columns : Nat)

/-- Calculates the number of seating arrangements for two students
    who cannot be seated adjacent to each other --/
def countSeatingArrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating the correct number of seating arrangements --/
theorem correct_seating_arrangements :
  let room : ExamRoom := { rows := 5, columns := 6 }
  countSeatingArrangements room = 772 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1382_138288


namespace NUMINAMATH_CALUDE_fold_reflection_sum_l1382_138283

/-- The fold line passing through the midpoint of (0,3) and (5,0) -/
def fold_line (x y : ℝ) : Prop := y = (5/3) * x - 1

/-- The property that (m,n) is the reflection of (8,4) across the fold line -/
def reflection_property (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    fold_line x y ∧ 
    (x = (8 + m) / 2 ∧ y = (4 + n) / 2) ∧
    (n - 4) / (m - 8) = -3/5

theorem fold_reflection_sum (m n : ℝ) 
  (h1 : fold_line 0 3)
  (h2 : fold_line 5 0)
  (h3 : reflection_property m n) :
  m + n = 9.75 := by sorry

end NUMINAMATH_CALUDE_fold_reflection_sum_l1382_138283


namespace NUMINAMATH_CALUDE_lilly_fish_count_l1382_138282

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l1382_138282


namespace NUMINAMATH_CALUDE_inequality_contradiction_l1382_138236

theorem inequality_contradiction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l1382_138236


namespace NUMINAMATH_CALUDE_pies_cost_l1382_138232

theorem pies_cost (a b c d : ℕ) : 
  c = 2 * a →                           -- cherry pie costs the same as two apple pies
  b = 2 * d →                           -- blueberry pie costs the same as two damson pies
  c + 2 * d = a + 2 * b →               -- cherry pie and two damson pies cost the same as an apple pie and two blueberry pies
  a + b + c + d = 18                    -- total cost is £18
  := by sorry

end NUMINAMATH_CALUDE_pies_cost_l1382_138232


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1382_138200

theorem cubic_inequality_solution (x : ℝ) : x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (x > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1382_138200


namespace NUMINAMATH_CALUDE_latin_essay_scores_l1382_138222

/-- The maximum score for the Latin essay --/
def max_score : ℕ := 20

/-- Michel's score --/
def michel_score : ℕ := 14

/-- Claude's score --/
def claude_score : ℕ := 6

/-- The average score --/
def average_score : ℚ := (michel_score + claude_score) / 2

theorem latin_essay_scores :
  michel_score > 0 ∧
  michel_score ≤ max_score ∧
  claude_score > 0 ∧
  claude_score ≤ max_score ∧
  michel_score > average_score ∧
  claude_score < average_score ∧
  michel_score - michel_score / 3 = 3 * (claude_score - claude_score / 3) :=
by sorry

end NUMINAMATH_CALUDE_latin_essay_scores_l1382_138222


namespace NUMINAMATH_CALUDE_external_diagonals_condition_l1382_138231

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the given lengths could be valid external diagonals of a right regular prism -/
def isValidExternalDiagonals (d : ExternalDiagonals) : Prop :=
  d.x^2 + d.y^2 > d.z^2 ∧ d.y^2 + d.z^2 > d.x^2 ∧ d.x^2 + d.z^2 > d.y^2

theorem external_diagonals_condition (d : ExternalDiagonals) :
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 → isValidExternalDiagonals d :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_condition_l1382_138231


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1382_138201

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 4) ∧
  (∀ x : ℝ, (x = -1 ∨ x = 1 ∨ x = 4) → (deriv p) x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1382_138201
