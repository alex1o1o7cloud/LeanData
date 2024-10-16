import Mathlib

namespace NUMINAMATH_CALUDE_zachary_bus_ride_length_l4151_415155

theorem zachary_bus_ride_length : 
  let vince_ride : ℚ := 0.625
  let difference : ℚ := 0.125
  let zachary_ride : ℚ := vince_ride - difference
  zachary_ride = 0.500 := by sorry

end NUMINAMATH_CALUDE_zachary_bus_ride_length_l4151_415155


namespace NUMINAMATH_CALUDE_value_difference_l4151_415143

theorem value_difference (n : ℝ) (h : n = 40) : 
  (n * 1.25) - (n * 0.7) = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_difference_l4151_415143


namespace NUMINAMATH_CALUDE_periodic_exponential_function_l4151_415187

theorem periodic_exponential_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x ∈ Set.Icc (-1) 1, f x = 2^(x + a)) →
  f 2017 = 8 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_exponential_function_l4151_415187


namespace NUMINAMATH_CALUDE_base_neg_two_2019_has_six_nonzero_digits_l4151_415103

/-- Represents a number in base -2 as a list of binary digits -/
def BaseNegTwo := List Bool

/-- Converts a natural number to its base -2 representation -/
def toBaseNegTwo (n : ℕ) : BaseNegTwo :=
  sorry

/-- Counts the number of non-zero digits in a base -2 representation -/
def countNonZeroDigits (b : BaseNegTwo) : ℕ :=
  sorry

/-- Theorem: 2019 in base -2 has exactly 6 non-zero digits -/
theorem base_neg_two_2019_has_six_nonzero_digits :
  countNonZeroDigits (toBaseNegTwo 2019) = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_neg_two_2019_has_six_nonzero_digits_l4151_415103


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l4151_415180

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > (1 / 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l4151_415180


namespace NUMINAMATH_CALUDE_tim_has_twelve_nickels_l4151_415102

/-- Represents the number of coins Tim has -/
structure TimsCoins where
  quarters : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total number of nickels Tim has after receiving coins from his dad -/
def total_nickels (initial : TimsCoins) (from_dad : TimsCoins) : ℕ :=
  initial.nickels + from_dad.nickels

/-- Theorem stating that Tim has 12 nickels after receiving coins from his dad -/
theorem tim_has_twelve_nickels :
  let initial := TimsCoins.mk 7 9 0
  let from_dad := TimsCoins.mk 0 3 5
  total_nickels initial from_dad = 12 := by
  sorry


end NUMINAMATH_CALUDE_tim_has_twelve_nickels_l4151_415102


namespace NUMINAMATH_CALUDE_lisa_photos_l4151_415114

def photo_problem (animal_photos flower_photos scenery_photos this_weekend last_weekend : ℕ) : Prop :=
  animal_photos = 10 ∧
  flower_photos = 3 * animal_photos ∧
  scenery_photos = flower_photos - 10 ∧
  this_weekend = animal_photos + flower_photos + scenery_photos ∧
  last_weekend = this_weekend - 15

theorem lisa_photos :
  ∀ animal_photos flower_photos scenery_photos this_weekend last_weekend,
  photo_problem animal_photos flower_photos scenery_photos this_weekend last_weekend →
  last_weekend = 45 := by
sorry

end NUMINAMATH_CALUDE_lisa_photos_l4151_415114


namespace NUMINAMATH_CALUDE_only_one_divides_power_plus_one_l4151_415115

theorem only_one_divides_power_plus_one :
  ∀ n : ℕ+, n.val % 2 = 1 ∧ (n.val ∣ 3^n.val + 1) → n = 1 := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_plus_one_l4151_415115


namespace NUMINAMATH_CALUDE_pierced_square_theorem_l4151_415133

/-- Represents a square pierced at n points and cut into triangles --/
structure PiercedSquare where
  n : ℕ  -- number of pierced points
  no_collinear_triples : True  -- represents the condition that no three points are collinear
  no_internal_piercings : True  -- represents the condition that there are no piercings inside triangles

/-- Calculates the number of triangles formed in a pierced square --/
def num_triangles (ps : PiercedSquare) : ℕ :=
  2 * (ps.n + 1)

/-- Calculates the number of cuts made in a pierced square --/
def num_cuts (ps : PiercedSquare) : ℕ :=
  (3 * num_triangles ps - 4) / 2

/-- Theorem stating the relationship between pierced points, triangles, and cuts --/
theorem pierced_square_theorem (ps : PiercedSquare) :
  (num_triangles ps = 2 * (ps.n + 1)) ∧
  (num_cuts ps = (3 * num_triangles ps - 4) / 2) := by
  sorry

end NUMINAMATH_CALUDE_pierced_square_theorem_l4151_415133


namespace NUMINAMATH_CALUDE_comprehensive_formula_l4151_415111

theorem comprehensive_formula (h1 : 12 * 5 = 60) (h2 : 60 - 42 = 18) :
  12 * 5 - 42 = 18 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_formula_l4151_415111


namespace NUMINAMATH_CALUDE_triangleAreaSum_form_l4151_415123

/-- The sum of areas of all triangles with vertices on a 2 by 3 by 4 rectangular box -/
def triangleAreaSum : ℝ := sorry

/-- The number of vertices of a rectangular box -/
def vertexCount : ℕ := 8

/-- The dimensions of the rectangular box -/
def boxDimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is just to satisfy Lean's totality requirement

/-- Theorem stating the form of the sum of triangle areas -/
theorem triangleAreaSum_form :
  ∃ (k p : ℝ), triangleAreaSum = 168 + k * Real.sqrt p :=
sorry

end NUMINAMATH_CALUDE_triangleAreaSum_form_l4151_415123


namespace NUMINAMATH_CALUDE_toy_truck_cost_l4151_415195

/-- The amount spent on a toy truck given initial amount, pencil case cost, and remaining amount -/
theorem toy_truck_cost (initial : ℝ) (pencil_case : ℝ) (remaining : ℝ) :
  initial = 10 → pencil_case = 2 → remaining = 5 →
  initial - pencil_case - remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_toy_truck_cost_l4151_415195


namespace NUMINAMATH_CALUDE_sin_cos_range_l4151_415104

theorem sin_cos_range (x y : ℝ) (h : 2 * (Real.sin x)^2 + (Real.cos y)^2 = 1) :
  ∃ (z : ℝ), (Real.sin x)^2 + (Real.cos y)^2 = z ∧ 1/2 ≤ z ∧ z ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_range_l4151_415104


namespace NUMINAMATH_CALUDE_garden_area_increase_l4151_415116

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that reshaping it into a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let square_side : ℝ := rect_perimeter / 4
  let rect_area : ℝ := rect_length * rect_width
  let square_area : ℝ := square_side * square_side
  square_area - rect_area = 400 := by
sorry


end NUMINAMATH_CALUDE_garden_area_increase_l4151_415116


namespace NUMINAMATH_CALUDE_exponent_division_l4151_415173

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^4 / x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l4151_415173


namespace NUMINAMATH_CALUDE_loggerhead_turtle_eggs_per_nest_l4151_415139

/-- The average number of eggs per nest for loggerhead turtles -/
def average_eggs_per_nest (total_eggs : ℕ) (total_nests : ℕ) : ℚ :=
  total_eggs / total_nests

/-- Theorem: The average number of eggs per nest is 150 -/
theorem loggerhead_turtle_eggs_per_nest :
  average_eggs_per_nest 3000000 20000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_loggerhead_turtle_eggs_per_nest_l4151_415139


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l4151_415121

theorem new_supervisor_salary 
  (num_workers : ℕ) 
  (num_supervisors : ℕ) 
  (initial_avg_salary : ℚ) 
  (retiring_supervisor_salary : ℚ) 
  (new_avg_salary : ℚ) 
  (h1 : num_workers = 12) 
  (h2 : num_supervisors = 3) 
  (h3 : initial_avg_salary = 650) 
  (h4 : retiring_supervisor_salary = 1200) 
  (h5 : new_avg_salary = 675) : 
  (num_workers + num_supervisors) * new_avg_salary - 
  ((num_workers + num_supervisors) * initial_avg_salary - retiring_supervisor_salary) = 1575 :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l4151_415121


namespace NUMINAMATH_CALUDE_intersection_of_sets_l4151_415140

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define set A
def SetA (x y m : ℝ) : Prop := (m + 3) * x + (m - 2) * y - 1 - 2 * m = 0

-- Define set B (tangent lines to the circle)
def SetB (x y : ℝ) : Prop := ∃ (a b : ℝ), Circle a b ∧ (x - a) * a + (y - b) * b = 0

-- Define the intersection set
def IntersectionSet (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_of_sets :
  ∀ (x y : ℝ), (∃ (m : ℝ), SetA x y m) ∧ SetB x y ↔ IntersectionSet x y :=
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l4151_415140


namespace NUMINAMATH_CALUDE_cafeteria_pies_l4151_415119

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 250 →
  handed_out = 33 →
  apples_per_pie = 7 →
  (initial_apples - handed_out) / apples_per_pie = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l4151_415119


namespace NUMINAMATH_CALUDE_circle_line_intersection_l4151_415127

def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

theorem circle_line_intersection (r : ℝ) (hr : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ A ∈ Line ∧
    B ∈ Circle r ∧ B ∈ Line ∧
    C ∈ Circle r ∧
    C.1 = (5/4 * A.1 + 3/4 * B.1) ∧
    C.2 = (5/4 * A.2 + 3/4 * B.2) →
  r = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l4151_415127


namespace NUMINAMATH_CALUDE_sum_of_numbers_l4151_415170

theorem sum_of_numbers (a b : ℕ) : 
  100 ≤ a ∧ a ≤ 999 →   -- a is a three-digit number
  10 ≤ b ∧ b ≤ 99 →     -- b is a two-digit number
  a - b = 989 →         -- their difference is 989
  a + b = 1009 :=       -- prove their sum is 1009
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l4151_415170


namespace NUMINAMATH_CALUDE_intersection_of_four_convex_sets_l4151_415168

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for convex sets in a plane
variable {ConvexSet : Type}

-- Define a function to check if a point is in a convex set
variable (in_set : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (is_convex : ConvexSet → Prop)

-- Define a function to represent the intersection of sets
variable (intersection : List ConvexSet → Set Point)

-- Theorem statement
theorem intersection_of_four_convex_sets
  (C1 C2 C3 C4 : ConvexSet)
  (convex1 : is_convex C1)
  (convex2 : is_convex C2)
  (convex3 : is_convex C3)
  (convex4 : is_convex C4)
  (intersect_three1 : (intersection [C1, C2, C3]).Nonempty)
  (intersect_three2 : (intersection [C1, C2, C4]).Nonempty)
  (intersect_three3 : (intersection [C1, C3, C4]).Nonempty)
  (intersect_three4 : (intersection [C2, C3, C4]).Nonempty) :
  (intersection [C1, C2, C3, C4]).Nonempty :=
sorry

end NUMINAMATH_CALUDE_intersection_of_four_convex_sets_l4151_415168


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l4151_415101

/-- Given two points C and D on a Cartesian plane, this theorem proves that
    the sum of the slope and y-intercept of the line passing through these points is 1. -/
theorem slope_intercept_sum (C D : ℝ × ℝ) : C = (2, 3) → D = (5, 9) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l4151_415101


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l4151_415125

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l4151_415125


namespace NUMINAMATH_CALUDE_winning_lines_8_cube_l4151_415137

/-- The number of straight lines containing 8 points in a 3D cubic grid --/
def winning_lines (n : ℕ) : ℕ :=
  ((n + 2)^3 - n^3) / 2

/-- Theorem: In an 8×8×8 cubic grid, the number of straight lines containing 8 points is 244 --/
theorem winning_lines_8_cube : winning_lines 8 = 244 := by
  sorry

end NUMINAMATH_CALUDE_winning_lines_8_cube_l4151_415137


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l4151_415192

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^2

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x y : ℝ, x < y ∧ x < (3/2) ∧ y < (3/2) → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l4151_415192


namespace NUMINAMATH_CALUDE_tom_games_owned_before_l4151_415194

/-- The number of games Tom owned before purchasing new games -/
def games_owned_before : ℕ := 0

/-- The cost of the Batman game in dollars -/
def batman_game_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games in dollars -/
def total_spent : ℚ := 18.66

theorem tom_games_owned_before :
  games_owned_before = 0 ∧
  batman_game_cost + superman_game_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_tom_games_owned_before_l4151_415194


namespace NUMINAMATH_CALUDE_percentage_problem_l4151_415132

theorem percentage_problem (P : ℝ) : P = (354.2 * 6 * 100) / 1265 ↔ (P / 100) * 1265 / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4151_415132


namespace NUMINAMATH_CALUDE_irrational_approximation_l4151_415160

theorem irrational_approximation (x : ℝ) (h_pos : x > 0) (h_irr : Irrational x) :
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l4151_415160


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l4151_415107

theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π / 6)
  let g (x : ℝ) := f (x + π / 4)
  let h (x : ℝ) := g (x / 2)
  h x = Real.sin (x / 2 + 5 * π / 12) := by sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l4151_415107


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4151_415100

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    if a₁ + a₃ = 10 and a₄ + a₆ = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 1 + a 3 = 10) 
  (h3 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4151_415100


namespace NUMINAMATH_CALUDE_whitney_cant_afford_l4151_415161

def poster_price : ℚ := 7.5
def notebook_price : ℚ := 5.25
def bookmark_price : ℚ := 3.1
def pencil_price : ℚ := 1.15
def sales_tax_rate : ℚ := 0.08
def initial_money : ℚ := 40

def total_cost (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) : ℚ :=
  let subtotal := poster_price * poster_qty + notebook_price * notebook_qty + 
                  bookmark_price * bookmark_qty + pencil_price * pencil_qty
  subtotal * (1 + sales_tax_rate)

theorem whitney_cant_afford (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) 
  (h_poster : poster_qty = 3)
  (h_notebook : notebook_qty = 4)
  (h_bookmark : bookmark_qty = 5)
  (h_pencil : pencil_qty = 2) :
  total_cost poster_qty notebook_qty bookmark_qty pencil_qty > initial_money :=
by sorry

end NUMINAMATH_CALUDE_whitney_cant_afford_l4151_415161


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l4151_415198

theorem geometric_arithmetic_sequence_sum (x y : ℝ) : 
  5 < x ∧ x < y ∧ y < 15 →
  (∃ r : ℝ, r > 0 ∧ x = 5 * r ∧ y = 5 * r^2) →
  (∃ d : ℝ, y = x + d ∧ 15 = y + d) →
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l4151_415198


namespace NUMINAMATH_CALUDE_family_income_theorem_l4151_415153

theorem family_income_theorem (initial_members : ℕ) (new_average : ℝ) (deceased_income : ℝ) :
  initial_members = 4 →
  new_average = 650 →
  deceased_income = 990 →
  (initial_members - 1) * new_average + deceased_income = initial_members * 735 :=
by sorry

end NUMINAMATH_CALUDE_family_income_theorem_l4151_415153


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l4151_415124

theorem min_value_cos_sin (x : ℝ) : 
  3 * Real.cos x - 4 * Real.sin x ≥ -5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y - 4 * Real.sin y = -5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l4151_415124


namespace NUMINAMATH_CALUDE_smallest_sum_is_381_l4151_415197

/-- A permutation of the digits 1 to 6 -/
def Digit6Perm := Fin 6 → Fin 6

/-- Checks if a permutation is valid (bijective) -/
def isValidPerm (p : Digit6Perm) : Prop :=
  Function.Bijective p

/-- Converts a permutation to two 3-digit numbers -/
def permToNumbers (p : Digit6Perm) : ℕ × ℕ :=
  ((p 0 + 1) * 100 + (p 1 + 1) * 10 + (p 2 + 1),
   (p 3 + 1) * 100 + (p 4 + 1) * 10 + (p 5 + 1))

/-- Sums the two numbers obtained from a permutation -/
def sumFromPerm (p : Digit6Perm) : ℕ :=
  let (n1, n2) := permToNumbers p
  n1 + n2

/-- The main theorem stating that 381 is the smallest possible sum -/
theorem smallest_sum_is_381 :
  ∀ p : Digit6Perm, isValidPerm p → sumFromPerm p ≥ 381 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_381_l4151_415197


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l4151_415190

/-- A shape made of unit cubes -/
structure CubeShape where
  /-- The number of cubes in the base -/
  base_cubes : ℕ
  /-- The number of layers -/
  layers : ℕ
  /-- The total number of cubes -/
  total_cubes : ℕ
  /-- Condition: The base is a square -/
  base_is_square : base_cubes = 4
  /-- Condition: There are two layers -/
  two_layers : layers = 2
  /-- Condition: Total cubes is the product of base cubes and layers -/
  total_cubes_eq : total_cubes = base_cubes * layers

/-- The volume of the shape in cubic units -/
def volume (shape : CubeShape) : ℕ := shape.total_cubes

/-- The surface area of the shape in square units -/
def surface_area (shape : CubeShape) : ℕ :=
  6 * shape.total_cubes - 2 * shape.base_cubes

/-- The theorem stating the ratio of volume to surface area -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  2 * (volume shape) = surface_area shape := by
  sorry

#check volume_to_surface_area_ratio

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l4151_415190


namespace NUMINAMATH_CALUDE_sugar_for_muffins_l4151_415177

/-- Given a recipe that requires 3 cups of sugar for 24 muffins,
    calculate the number of cups of sugar needed for 72 muffins. -/
theorem sugar_for_muffins (recipe_muffins : ℕ) (recipe_sugar : ℕ) (target_muffins : ℕ) :
  recipe_muffins = 24 →
  recipe_sugar = 3 →
  target_muffins = 72 →
  (target_muffins * recipe_sugar) / recipe_muffins = 9 :=
by
  sorry

#check sugar_for_muffins

end NUMINAMATH_CALUDE_sugar_for_muffins_l4151_415177


namespace NUMINAMATH_CALUDE_max_consecutive_sum_is_six_l4151_415172

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The target sum -/
def target_sum : ℕ := 21

/-- The property that n consecutive integers sum to the target -/
def sum_to_target (n : ℕ) : Prop :=
  sum_first_n n = target_sum

/-- The maximum number of consecutive positive integers that sum to the target -/
def max_consecutive_sum : ℕ := 6

theorem max_consecutive_sum_is_six :
  (sum_to_target max_consecutive_sum) ∧
  (∀ k : ℕ, k > max_consecutive_sum → ¬(sum_to_target k)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_is_six_l4151_415172


namespace NUMINAMATH_CALUDE_intersection_complement_A_with_B_l4151_415150

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_complement_A_with_B :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_with_B_l4151_415150


namespace NUMINAMATH_CALUDE_unique_student_count_l4151_415151

theorem unique_student_count :
  ∃! n : ℕ, n < 400 ∧ n % 17 = 15 ∧ n % 19 = 10 ∧ n = 219 :=
by sorry

end NUMINAMATH_CALUDE_unique_student_count_l4151_415151


namespace NUMINAMATH_CALUDE_evaluate_expression_l4151_415154

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4151_415154


namespace NUMINAMATH_CALUDE_baker_april_earnings_l4151_415176

def baker_earnings (cake_price cake_sold pie_price pie_sold bread_price bread_sold cookie_price cookie_sold pie_discount tax_rate : ℚ) : ℚ :=
  let cake_revenue := cake_price * cake_sold
  let pie_revenue := pie_price * pie_sold * (1 - pie_discount)
  let bread_revenue := bread_price * bread_sold
  let cookie_revenue := cookie_price * cookie_sold
  let total_revenue := cake_revenue + pie_revenue + bread_revenue + cookie_revenue
  total_revenue * (1 + tax_rate)

theorem baker_april_earnings :
  baker_earnings 12 453 7 126 3.5 95 1.5 320 0.1 0.05 = 7394.42 := by
  sorry

end NUMINAMATH_CALUDE_baker_april_earnings_l4151_415176


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l4151_415113

def miles_to_school : ℝ := 15
def miles_to_softball : ℝ := 6
def miles_to_restaurant : ℝ := 2
def miles_to_friend : ℝ := 4
def miles_to_home : ℝ := 11
def initial_gas : ℝ := 2

def total_miles : ℝ := miles_to_school + miles_to_softball + miles_to_restaurant + miles_to_friend + miles_to_home

theorem car_fuel_efficiency :
  total_miles / initial_gas = 19 := by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l4151_415113


namespace NUMINAMATH_CALUDE_single_burger_cost_l4151_415109

theorem single_burger_cost 
  (total_spent : ℚ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℚ)
  (h1 : total_spent = 64.5)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 29)
  (h4 : double_burger_cost = 1.5)
  : ∃ (single_burger_cost : ℚ), single_burger_cost = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_l4151_415109


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l4151_415157

theorem fly_distance_from_ceiling (x y z : ℝ) : 
  x = 3 → y = 4 → (x^2 + y^2 + z^2 = 5^2) → z = 0 := by sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l4151_415157


namespace NUMINAMATH_CALUDE_sum_distances_foci_to_line_l4151_415175

/-- The ellipse C in the xy-plane -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- The left focus of ellipse C -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of ellipse C -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def dist_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- Theorem: The sum of distances from the foci of ellipse C to line l is 2√2 -/
theorem sum_distances_foci_to_line :
  dist_point_to_line F₁ line_l + dist_point_to_line F₂ line_l = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_distances_foci_to_line_l4151_415175


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l4151_415163

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 1) < 0}
def B : Set ℝ := {x | Real.exp x > 1}

-- Define the interval [1, +∞)
def interval : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem complement_A_intersect_B : (Aᶜ ∩ B) = interval := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l4151_415163


namespace NUMINAMATH_CALUDE_exam_average_l4151_415179

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℝ) (failed_avg : ℝ) :
  total_candidates = 120 →
  passed_candidates = 100 →
  passed_avg = 39 →
  failed_avg = 15 →
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_candidates * passed_avg + failed_candidates * failed_avg
  let overall_avg := total_marks / total_candidates
  overall_avg = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l4151_415179


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l4151_415167

theorem stratified_sampling_theorem (teachers male_students female_students : ℕ) 
  (female_sample : ℕ) (n : ℕ) : 
  teachers = 160 → 
  male_students = 960 → 
  female_students = 800 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) = 
    (female_sample : ℚ) / (n : ℚ) → 
  n = 192 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l4151_415167


namespace NUMINAMATH_CALUDE_max_fruits_is_34_l4151_415174

/-- Represents the weight of an apple in grams -/
def apple_weight : ℕ := 300

/-- Represents the weight of a pear in grams -/
def pear_weight : ℕ := 200

/-- Represents the maximum weight Ana's bag can hold in grams -/
def bag_capacity : ℕ := 7000

/-- Represents the constraint on the number of apples and pears -/
def weight_constraint (m p : ℕ) : Prop :=
  m * apple_weight + p * pear_weight ≤ bag_capacity

/-- Represents the total number of fruits -/
def total_fruits (m p : ℕ) : ℕ := m + p

/-- Theorem stating that the maximum number of fruits Ana can buy is 34 -/
theorem max_fruits_is_34 : 
  ∃ (m p : ℕ), weight_constraint m p ∧ m > 0 ∧ p > 0 ∧
  total_fruits m p = 34 ∧
  ∀ (m' p' : ℕ), weight_constraint m' p' ∧ m' > 0 ∧ p' > 0 → 
    total_fruits m' p' ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_max_fruits_is_34_l4151_415174


namespace NUMINAMATH_CALUDE_propositions_truth_values_l4151_415134

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the theorem
theorem propositions_truth_values :
  -- Proposition ① is false
  ¬(∀ m n α β, parallelLP m α → parallelLP n β → parallelPP α β → parallel m n) ∧
  -- Proposition ② is true
  (∀ m n α β, parallel m n → contains α m → perpendicularLP n β → perpendicularPP α β) ∧
  -- Proposition ③ is false
  ¬(∀ m n α β, intersect α β m → parallel m n → parallelLP n α ∧ parallelLP n β) ∧
  -- Proposition ④ is true
  (∀ m n α β, perpendicular m n → intersect α β m → perpendicularLP n α ∨ perpendicularLP n β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_values_l4151_415134


namespace NUMINAMATH_CALUDE_point_positions_l4151_415148

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x + 4*y - 4

def point_M : ℝ × ℝ := (2, -4)
def point_N : ℝ × ℝ := (-2, 1)

theorem point_positions :
  circle_equation point_M.1 point_M.2 < 0 ∧ 
  circle_equation point_N.1 point_N.2 > 0 := by
sorry

end NUMINAMATH_CALUDE_point_positions_l4151_415148


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4151_415182

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 ↔ (2*x - a < 1 ∧ x - 2*b > 3)) → 
  (a + 1) * (b - 1) = -6 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4151_415182


namespace NUMINAMATH_CALUDE_eric_pencils_l4151_415131

theorem eric_pencils (containers : ℕ) (additional_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : containers = 5)
  (h2 : additional_pencils = 30)
  (h3 : total_pencils = 36)
  (h4 : total_pencils % containers = 0) :
  total_pencils - additional_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_eric_pencils_l4151_415131


namespace NUMINAMATH_CALUDE_number_equation_l4151_415141

theorem number_equation : ∃ x : ℝ, x * (37 - 15) - 25 = 327 :=
by
  sorry

end NUMINAMATH_CALUDE_number_equation_l4151_415141


namespace NUMINAMATH_CALUDE_total_earnings_after_seven_days_l4151_415159

/- Define the prices of books -/
def fantasy_price : ℕ := 6
def literature_price : ℕ := fantasy_price / 2
def mystery_price : ℕ := 4

/- Define the daily sales quantities -/
def fantasy_sales : ℕ := 5
def literature_sales : ℕ := 8
def mystery_sales : ℕ := 3

/- Define the number of days -/
def days : ℕ := 7

/- Calculate daily earnings -/
def daily_earnings : ℕ := 
  fantasy_sales * fantasy_price + 
  literature_sales * literature_price + 
  mystery_sales * mystery_price

/- Theorem to prove -/
theorem total_earnings_after_seven_days : 
  daily_earnings * days = 462 := by sorry

end NUMINAMATH_CALUDE_total_earnings_after_seven_days_l4151_415159


namespace NUMINAMATH_CALUDE_initial_dumbbell_count_l4151_415142

theorem initial_dumbbell_count (dumbbell_weight : ℕ) (added_dumbbells : ℕ) (total_weight : ℕ) : 
  dumbbell_weight = 20 →
  added_dumbbells = 2 →
  total_weight = 120 →
  (total_weight / dumbbell_weight) - added_dumbbells = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_dumbbell_count_l4151_415142


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4151_415117

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4151_415117


namespace NUMINAMATH_CALUDE_infinitely_many_good_pairs_l4151_415128

/-- A natural number is 'good' if every prime factor in its prime factorization appears with at least the power of 2. -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * a n * (a n + 1)

/-- The main theorem stating that there are infinitely many pairs of consecutive 'good' numbers -/
theorem infinitely_many_good_pairs :
  ∀ n : ℕ, is_good (a n) ∧ is_good (a n + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_good_pairs_l4151_415128


namespace NUMINAMATH_CALUDE_andrews_balloons_l4151_415130

/-- Given a number of blue and purple balloons, calculates how many balloons are left after sharing half of the total. -/
def balloons_left (blue : ℕ) (purple : ℕ) : ℕ :=
  (blue + purple) / 2

/-- Theorem stating that given 303 blue balloons and 453 purple balloons, 
    the number of balloons left after sharing half is 378. -/
theorem andrews_balloons : balloons_left 303 453 = 378 := by
  sorry

end NUMINAMATH_CALUDE_andrews_balloons_l4151_415130


namespace NUMINAMATH_CALUDE_gcf_75_45_l4151_415178

theorem gcf_75_45 : Nat.gcd 75 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_45_l4151_415178


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4151_415199

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4151_415199


namespace NUMINAMATH_CALUDE_pen_pencil_difference_is_1500_l4151_415122

/-- Represents the stationery order problem --/
structure StationeryOrder where
  pencilBoxes : ℕ
  pencilsPerBox : ℕ
  penCost : ℕ
  pencilCost : ℕ
  totalCost : ℕ

/-- Calculates the difference between pens and pencils ordered --/
def penPencilDifference (order : StationeryOrder) : ℕ :=
  let totalPencils := order.pencilBoxes * order.pencilsPerBox
  let totalPenCost := order.totalCost - order.pencilCost * totalPencils
  let totalPens := totalPenCost / order.penCost
  totalPens - totalPencils

/-- Theorem stating the difference between pens and pencils ordered --/
theorem pen_pencil_difference_is_1500 (order : StationeryOrder) 
  (h1 : order.pencilBoxes = 15)
  (h2 : order.pencilsPerBox = 80)
  (h3 : order.penCost = 5)
  (h4 : order.pencilCost = 4)
  (h5 : order.totalCost = 18300)
  (h6 : order.penCost * (penPencilDifference order + order.pencilBoxes * order.pencilsPerBox) > 
        2 * order.pencilCost * (order.pencilBoxes * order.pencilsPerBox)) :
  penPencilDifference order = 1500 := by
  sorry

#eval penPencilDifference { pencilBoxes := 15, pencilsPerBox := 80, penCost := 5, pencilCost := 4, totalCost := 18300 }

end NUMINAMATH_CALUDE_pen_pencil_difference_is_1500_l4151_415122


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l4151_415120

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > v₂ → v₁ > 0 → v₂ > 0 →
  (v₁ + v₂ = 25) →
  (v₁ - v₂ = 10 / 3) →
  (v₁ / v₂ = 17 / 13) := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l4151_415120


namespace NUMINAMATH_CALUDE_committee_selection_l4151_415162

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l4151_415162


namespace NUMINAMATH_CALUDE_correct_factorization_l4151_415112

theorem correct_factorization (x y : ℝ) : x^3 + 4*x^2*y + 4*x*y^2 = x * (x + 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l4151_415112


namespace NUMINAMATH_CALUDE_slope_of_cutting_line_l4151_415185

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℚ

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { v1 := ⟨4, 20⟩
  , v2 := ⟨4, 56⟩
  , v3 := ⟨13, 81⟩
  , v4 := ⟨13, 45⟩ }

/-- The theorem to be proved -/
theorem slope_of_cutting_line :
  ∃ (l : Line), cutsIntoCongruentPolygons specificParallelogram l ∧ l.slope = 53 / 9 :=
sorry

end NUMINAMATH_CALUDE_slope_of_cutting_line_l4151_415185


namespace NUMINAMATH_CALUDE_inequality_proof_l4151_415164

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4151_415164


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l4151_415165

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 18593/32768 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l4151_415165


namespace NUMINAMATH_CALUDE_red_balls_count_l4151_415193

theorem red_balls_count (yellow_balls : ℕ) (total_balls : ℕ) 
  (yellow_prob : ℚ) (h1 : yellow_balls = 4) 
  (h2 : yellow_prob = 1 / 5) 
  (h3 : yellow_prob = yellow_balls / total_balls) : 
  total_balls - yellow_balls = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l4151_415193


namespace NUMINAMATH_CALUDE_triangle_inequality_l4151_415146

/-- Given a triangle ABC with side lengths a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove that (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3R / r,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c R r p : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hR : R > 0) (hr : r > 0) (hp : p > 0) (h_semi : p = (a + b + c) / 2)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
    (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3 * R / r ∧
    ((a / (p - a)) + (b / (p - b)) + (c / (p - c)) = 3 * R / r ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4151_415146


namespace NUMINAMATH_CALUDE_nikki_to_michael_ratio_l4151_415169

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn. -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie length problem. -/
def movie_conditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.ryn = (4/5) * m.nikki ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating that the ratio of Nikki's movie length to Michael's is 3:1. -/
theorem nikki_to_michael_ratio (m : MovieLengths) 
  (h : movie_conditions m) : m.nikki / m.michael = 3 := by
  sorry

end NUMINAMATH_CALUDE_nikki_to_michael_ratio_l4151_415169


namespace NUMINAMATH_CALUDE_exists_line_intersecting_four_circles_l4151_415108

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- Represents a configuration of circles in a unit square -/
structure CircleConfiguration where
  circles : List Circle
  sum_of_circumferences_eq_10 : (circles.map (fun c => c.diameter * Real.pi)).sum = 10

/-- Main theorem: If the sum of circumferences of circles in a unit square is 10,
    then there exists a line intersecting at least 4 of these circles -/
theorem exists_line_intersecting_four_circles (config : CircleConfiguration) :
  ∃ (line : ℝ → ℝ → Prop), (∃ (intersected_circles : List Circle),
    intersected_circles.length ≥ 4 ∧
    ∀ c ∈ intersected_circles, c ∈ config.circles ∧
    ∃ (x y : ℝ), x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ line x y) :=
sorry

end NUMINAMATH_CALUDE_exists_line_intersecting_four_circles_l4151_415108


namespace NUMINAMATH_CALUDE_equation_solutions_l4151_415189

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - (x - 2) = 0 ↔ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - x = x + 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4151_415189


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l4151_415138

theorem fraction_equation_solution :
  ∃ x : ℝ, x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l4151_415138


namespace NUMINAMATH_CALUDE_max_height_foldable_triangle_l4151_415188

/-- The maximum height of a foldable table constructed from a triangle --/
theorem max_height_foldable_triangle (PQ QR PR : ℝ) (h_PQ : PQ = 24) (h_QR : QR = 32) (h_PR : PR = 40) :
  let s := (PQ + QR + PR) / 2
  let A := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  let h_p := 2 * A / QR
  let h_q := 2 * A / PR
  let h_r := 2 * A / PQ
  let h' := min (h_p * h_q / (h_p + h_q)) (min (h_q * h_r / (h_q + h_r)) (h_r * h_p / (h_r + h_p)))
  h' = 48 * Real.sqrt 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_max_height_foldable_triangle_l4151_415188


namespace NUMINAMATH_CALUDE_basketball_team_selection_l4151_415106

theorem basketball_team_selection (total_players : Nat) (twins : Nat) (lineup_size : Nat) : 
  total_players = 12 →
  twins = 2 →
  lineup_size = 5 →
  (twins * (total_players - twins).choose (lineup_size - 1)) = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l4151_415106


namespace NUMINAMATH_CALUDE_toms_lawn_mowing_l4151_415145

/-- Proves the number of lawns Tom mowed given his earnings and expenses -/
theorem toms_lawn_mowing (charge_per_lawn : ℕ) (gas_expense : ℕ) (weed_income : ℕ) (total_profit : ℕ) 
  (h1 : charge_per_lawn = 12)
  (h2 : gas_expense = 17)
  (h3 : weed_income = 10)
  (h4 : total_profit = 29) :
  ∃ (lawns_mowed : ℕ), 
    lawns_mowed * charge_per_lawn + weed_income - gas_expense = total_profit ∧ 
    lawns_mowed = 3 := by
  sorry


end NUMINAMATH_CALUDE_toms_lawn_mowing_l4151_415145


namespace NUMINAMATH_CALUDE_basketball_selection_probabilities_l4151_415152

def shot_probability : ℚ := 2/3

def second_level_after_three_shots : ℚ := 8/27

def selected_probability : ℚ := 64/81

def selected_after_five_shots : ℚ := 16/81

theorem basketball_selection_probabilities :
  (2 * shot_probability * (1 - shot_probability) * shot_probability = second_level_after_three_shots) ∧
  (selected_after_five_shots / selected_probability = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_basketball_selection_probabilities_l4151_415152


namespace NUMINAMATH_CALUDE_polynomial_equality_l4151_415171

theorem polynomial_equality (x : ℝ) : let p : ℝ → ℝ := λ x => -7*x^4 - 5*x^3 - 8*x^2 + 8*x - 9
  4*x^4 + 7*x^3 - 2*x + 5 + p x = -3*x^4 + 2*x^3 - 8*x^2 + 6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4151_415171


namespace NUMINAMATH_CALUDE_wrong_observation_value_l4151_415149

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean true_value : ℝ) : 
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.54 →
  true_value = 48 →
  (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean = true_value - (n : ℝ) * initial_mean + (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean →
  true_value - ((n : ℝ) * corrected_mean - (n : ℝ) * initial_mean) = 21 := by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l4151_415149


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l4151_415156

/-- The radius of an inscribed circle in a quarter circular sector --/
theorem inscribed_circle_radius_in_quarter_sector (R : ℝ) (h : R > 0) :
  let r := R * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = R :=
by
  sorry

/-- The specific case where the outer radius is 3 cm --/
theorem inscribed_circle_radius_3cm :
  let r := 3 * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l4151_415156


namespace NUMINAMATH_CALUDE_triangle_problem_l4151_415144

theorem triangle_problem (a b c A B C : Real) (h1 : (2*b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13) (h3 : b + c = 5) :
  A = π/3 ∧ (1/2 * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l4151_415144


namespace NUMINAMATH_CALUDE_max_value_theorem_l4151_415191

theorem max_value_theorem (m n : ℝ) 
  (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) 
  (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) : 
  (∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 ∧ 2 ≤ x + y ∧ x + y ≤ 4 → m - 2*n ≥ x - 2*y) →
  2019*m + 2020*n = 2019 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4151_415191


namespace NUMINAMATH_CALUDE_isosceles_not_equilateral_l4151_415136

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a > 0 ∧ b > 0 ∧ c > 0)

-- Define an equilateral triangle
def IsEquilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c ∧ a > 0

-- Theorem: There exists an isosceles triangle that is not equilateral
theorem isosceles_not_equilateral : ∃ a b c : ℝ, IsIsosceles a b c ∧ ¬IsEquilateral a b c := by
  sorry


end NUMINAMATH_CALUDE_isosceles_not_equilateral_l4151_415136


namespace NUMINAMATH_CALUDE_no_intersection_l4151_415126

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- Define what it means for two functions to intersect at a point
def intersect_at (f g : ℝ → ℝ) (x : ℝ) : Prop := f x = g x

-- Theorem statement
theorem no_intersection :
  ¬ ∃ x : ℝ, intersect_at f g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l4151_415126


namespace NUMINAMATH_CALUDE_flea_treatment_l4151_415105

theorem flea_treatment (initial_fleas : ℕ) : 
  (initial_fleas / 2 / 2 / 2 / 2 = 14) → (initial_fleas - 14 = 210) := by
  sorry

end NUMINAMATH_CALUDE_flea_treatment_l4151_415105


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l4151_415166

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l4151_415166


namespace NUMINAMATH_CALUDE_impossible_sum_of_squares_l4151_415186

theorem impossible_sum_of_squares : ¬ ∃ x : ℝ,
  1000^2 + 1001^2 + 1002^2 + x^2 + 1004^2 = 6 :=
by
  sorry

#check impossible_sum_of_squares

end NUMINAMATH_CALUDE_impossible_sum_of_squares_l4151_415186


namespace NUMINAMATH_CALUDE_fermat_min_l4151_415129

theorem fermat_min (n : ℕ) (x y z : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n := by
  sorry

end NUMINAMATH_CALUDE_fermat_min_l4151_415129


namespace NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l4151_415147

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l4151_415147


namespace NUMINAMATH_CALUDE_trivia_team_size_l4151_415184

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : absent_members = 2)
  (h2 : points_per_member = 6)
  (h3 : total_points = 18) :
  ∃ (original_size : ℕ), 
    original_size * points_per_member - absent_members * points_per_member = total_points ∧ 
    original_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l4151_415184


namespace NUMINAMATH_CALUDE_jessica_red_marbles_l4151_415158

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_times_more : ℕ) :
  sandy_marbles = 144 →
  sandy_times_more = 4 →
  (sandy_marbles / sandy_times_more) / 12 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_red_marbles_l4151_415158


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4151_415181

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4151_415181


namespace NUMINAMATH_CALUDE_nicky_trade_loss_l4151_415183

/-- Calculates the profit or loss in a baseball card trade with tax --/
def trade_profit_loss (cards_given_value1 cards_given_value2 cards_given_count1 cards_given_count2
                       cards_received_value1 cards_received_value2 cards_received_count1 cards_received_count2
                       tax_rate : ℚ) : ℚ :=
  let total_given := cards_given_value1 * cards_given_count1 + cards_given_value2 * cards_given_count2
  let total_received := cards_received_value1 * cards_received_count1 + cards_received_value2 * cards_received_count2
  let total_traded := total_given + total_received
  let tax := total_traded * tax_rate
  total_received - total_given - tax

theorem nicky_trade_loss :
  trade_profit_loss 8 5 2 3 21 6 1 2 (5/100) = -(6/5) :=
by sorry

end NUMINAMATH_CALUDE_nicky_trade_loss_l4151_415183


namespace NUMINAMATH_CALUDE_non_adjacent_selections_of_seven_l4151_415196

/-- A function that calculates the number of ways to select 3 non-adjacent elements from a set of n elements -/
def non_adjacent_selections (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that for a set of 7 elements, there are 60 ways to select 3 non-adjacent elements -/
theorem non_adjacent_selections_of_seven :
  non_adjacent_selections 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_non_adjacent_selections_of_seven_l4151_415196


namespace NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_expression_second_quadrant_l4151_415135

-- Problem 1
theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) / 
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by sorry

-- Problem 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.sin α - Real.cos α := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_expression_second_quadrant_l4151_415135


namespace NUMINAMATH_CALUDE_no_real_solutions_l4151_415110

theorem no_real_solutions (k d : ℝ) (hk : k = -1) (hd : d < 0 ∨ d > 2) :
  ¬∃ (x y : ℝ), x^3 + y^3 = 2 ∧ y = k * x + d :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4151_415110


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4151_415118

theorem quadratic_equation_roots (m : ℝ) :
  (2 * (2 : ℝ)^2 - 5 * 2 - m = 0) →
  (m = -2 ∧ ∃ (x : ℝ), x ≠ 2 ∧ 2 * x^2 - 5 * x - m = 0 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4151_415118
