import Mathlib

namespace NUMINAMATH_CALUDE_new_person_weight_l628_62819

/-- Proves that the weight of a new person is 380 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) :
  initial_count = 20 →
  replaced_weight = 80 →
  average_increase = 15 →
  (initial_count : ℝ) * average_increase + replaced_weight = 380 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l628_62819


namespace NUMINAMATH_CALUDE_max_ones_on_board_l628_62804

/-- The operation that replaces two numbers with their GCD and LCM -/
def replace_with_gcd_lcm (a b : ℕ) : List ℕ :=
  [Nat.gcd a b, Nat.lcm a b]

/-- The set of numbers on the board -/
def board : Set ℕ := Finset.range 2014

/-- A sequence of operations on the board -/
def operation_sequence := List (ℕ × ℕ)

/-- Apply a sequence of operations to the board -/
def apply_operations (ops : operation_sequence) (b : Set ℕ) : Set ℕ :=
  sorry

/-- Count the number of 1's in a set of natural numbers -/
def count_ones (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 1's obtainable -/
theorem max_ones_on_board :
  ∃ (ops : operation_sequence),
    ∀ (ops' : operation_sequence),
      count_ones (apply_operations ops board) ≥ count_ones (apply_operations ops' board) ∧
      count_ones (apply_operations ops board) = 1007 :=
  sorry

end NUMINAMATH_CALUDE_max_ones_on_board_l628_62804


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l628_62837

/-- The equation of the circle is x^2 + y^2 - 2x + 6y + 6 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 6 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -3)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l628_62837


namespace NUMINAMATH_CALUDE_triangle_problem_l628_62826

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  C = π / 3 →
  Real.tan A = 3 / 4 →
  (Real.sin A = 3 / 5 ∧ b = 4 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l628_62826


namespace NUMINAMATH_CALUDE_cottonwood_fiber_diameter_scientific_notation_l628_62864

theorem cottonwood_fiber_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000108 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.08 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_cottonwood_fiber_diameter_scientific_notation_l628_62864


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l628_62810

def set_A : Set ℝ := {x | 2 * x < 2 + x}
def set_B : Set ℝ := {x | 5 - x > 8 - 4 * x}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l628_62810


namespace NUMINAMATH_CALUDE_point_on_bisector_l628_62822

/-- If A(a, b) and B(b, a) represent the same point, then this point lies on the line y = x. -/
theorem point_on_bisector (a b : ℝ) : (a, b) = (b, a) → a = b :=
by sorry

end NUMINAMATH_CALUDE_point_on_bisector_l628_62822


namespace NUMINAMATH_CALUDE_base_eight_addition_l628_62845

/-- Given a base-8 addition where 5XY₈ + 32₈ = 62X₈, prove that X + Y = 12 in base 10 --/
theorem base_eight_addition (X Y : ℕ) : 
  (5 * 8^2 + X * 8 + Y) + 32 = 6 * 8^2 + 2 * 8 + X → X + Y = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_l628_62845


namespace NUMINAMATH_CALUDE_problem_solution_l628_62833

theorem problem_solution (x y z : ℝ) 
  (h1 : 3 = 0.15 * x)
  (h2 : 3 = 0.25 * y)
  (h3 : z = 0.30 * y) :
  x - y + z = 11.6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l628_62833


namespace NUMINAMATH_CALUDE_kids_difference_l628_62853

theorem kids_difference (monday : ℕ) (tuesday : ℕ) 
  (h1 : monday = 6) (h2 : tuesday = 5) : monday - tuesday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l628_62853


namespace NUMINAMATH_CALUDE_positive_integer_problem_l628_62865

theorem positive_integer_problem (n : ℕ+) (h : (12 : ℝ) * n.val = n.val ^ 2 + 36) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_problem_l628_62865


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l628_62895

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    if the quadratic ax^2 + bx + c has exactly one root, then this root is -2 + √3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b = a - d ∧ c = a - 2*d) →  -- arithmetic sequence
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- ordering condition
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →  -- exactly one root
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l628_62895


namespace NUMINAMATH_CALUDE_car_wash_earnings_l628_62860

def weekly_allowance : ℝ := 8
def final_amount : ℝ := 12

theorem car_wash_earnings :
  final_amount - (weekly_allowance / 2) = 8 := by sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l628_62860


namespace NUMINAMATH_CALUDE_incorrect_conclusions_l628_62874

structure Conclusion where
  correlation : Bool  -- true for positive, false for negative
  coefficient : Real
  constant : Real

def is_correct (c : Conclusion) : Prop :=
  (c.correlation ↔ c.coefficient > 0)

theorem incorrect_conclusions 
  (c1 : Conclusion)
  (c2 : Conclusion)
  (c3 : Conclusion)
  (c4 : Conclusion)
  (h1 : c1 = { correlation := false, coefficient := 2.347, constant := -6.423 })
  (h2 : c2 = { correlation := false, coefficient := -3.476, constant := 5.648 })
  (h3 : c3 = { correlation := true, coefficient := 5.437, constant := 8.493 })
  (h4 : c4 = { correlation := true, coefficient := -4.326, constant := -4.578 }) :
  ¬(is_correct c1) ∧ ¬(is_correct c4) :=
sorry

end NUMINAMATH_CALUDE_incorrect_conclusions_l628_62874


namespace NUMINAMATH_CALUDE_f_difference_l628_62872

/-- Given a function f defined as f(n) = 1/3 * n * (n+1) * (n+2),
    prove that f(r) - f(r-1) = r * (r+1) for any real number r. -/
theorem f_difference (r : ℝ) : 
  let f (n : ℝ) := (1/3) * n * (n+1) * (n+2)
  f r - f (r-1) = r * (r+1) := by
sorry

end NUMINAMATH_CALUDE_f_difference_l628_62872


namespace NUMINAMATH_CALUDE_percentage_of_males_l628_62838

theorem percentage_of_males (total_employees : ℕ) (males_below_50 : ℕ) 
  (h1 : total_employees = 2200)
  (h2 : males_below_50 = 616)
  (h3 : (70 : ℚ) / 100 * (males_below_50 / ((70 : ℚ) / 100)) = males_below_50) :
  (males_below_50 / ((70 : ℚ) / 100)) / total_employees = (40 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_males_l628_62838


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l628_62873

/-- The original price of a concert ticket -/
def original_price : ℝ := 20

/-- The number of people who received a 40% discount -/
def discount_40_count : ℕ := 10

/-- The number of people who received a 15% discount -/
def discount_15_count : ℕ := 20

/-- The total number of people who bought tickets -/
def total_buyers : ℕ := 45

/-- The total revenue from ticket sales -/
def total_revenue : ℝ := 760

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  discount_40_count * (original_price * 0.6) + 
  discount_15_count * (original_price * 0.85) + 
  (total_buyers - discount_40_count - discount_15_count) * original_price = 
  total_revenue := by sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l628_62873


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l628_62820

theorem arithmetic_mean_fractions : 
  let a := 8 / 11
  let b := 9 / 11
  let c := 5 / 6
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l628_62820


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l628_62840

/-- The point symmetric to A(3, 4) with respect to the x-axis -/
def symmetric_point : ℝ × ℝ := (3, -4)

/-- The original point A -/
def point_A : ℝ × ℝ := (3, 4)

/-- Theorem stating that symmetric_point is indeed symmetric to point_A with respect to the x-axis -/
theorem symmetric_point_correct :
  symmetric_point.1 = point_A.1 ∧
  symmetric_point.2 = -point_A.2 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l628_62840


namespace NUMINAMATH_CALUDE_factorization_proof_l628_62856

theorem factorization_proof (x : ℝ) : 15 * x^2 + 10 * x - 5 = 5 * (3 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l628_62856


namespace NUMINAMATH_CALUDE_original_time_calculation_l628_62834

theorem original_time_calculation (original_speed : ℝ) (original_time : ℝ) 
  (h1 : original_speed > 0) (h2 : original_time > 0) : 
  (original_time / 0.8 = original_time + 10) → original_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_time_calculation_l628_62834


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l628_62847

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ x y : ℝ, x + y + m = 0 ∧ x^2 + y^2 = m ∧ 
   ∀ x' y' : ℝ, x' + y' + m = 0 → x'^2 + y'^2 ≥ m) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l628_62847


namespace NUMINAMATH_CALUDE_house_expansion_l628_62890

/-- Given two houses with areas 5200 and 7300 square feet, if their total area
    after expanding the smaller house is 16000 square feet, then the expansion
    size is 3500 square feet. -/
theorem house_expansion (small_house large_house expanded_total : ℕ)
    (h1 : small_house = 5200)
    (h2 : large_house = 7300)
    (h3 : expanded_total = 16000)
    (h4 : expanded_total = small_house + large_house + expansion_size) :
    expansion_size = 3500 := by
  sorry

end NUMINAMATH_CALUDE_house_expansion_l628_62890


namespace NUMINAMATH_CALUDE_range_of_a_l628_62811

-- Define the open interval (1, 2)
def open_interval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ open_interval, (x - 1)^2 < Real.log x / Real.log a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, inequality_holds a ↔ a ∈ {a : ℝ | 1 < a ∧ a ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l628_62811


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l628_62812

-- Define an even function that is decreasing on (0,+∞)
def is_even_and_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_even_and_decreasing f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l628_62812


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l628_62839

def apples_per_pie (total_apples : ℕ) (pies : ℕ) : ℕ := total_apples / pies

def pies_from_apples (available_apples : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples / apples_per_pie

def leftover_apples (available_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples - pies * apples_per_pie

theorem rosie_pies_theorem (available_apples : ℕ) (base_apples : ℕ) (base_pies : ℕ) :
  available_apples = 55 →
  base_apples = 15 →
  base_pies = 3 →
  let apples_per_pie := apples_per_pie base_apples base_pies
  let pies := pies_from_apples available_apples apples_per_pie
  let leftovers := leftover_apples available_apples pies apples_per_pie
  pies = 11 ∧ leftovers = 0 := by sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l628_62839


namespace NUMINAMATH_CALUDE_election_percentage_l628_62844

/-- Given an election with 700 total votes where the winning candidate has a majority of 476 votes,
    prove that the winning candidate received 84% of the votes. -/
theorem election_percentage (total_votes : ℕ) (winning_majority : ℕ) (winning_percentage : ℚ) :
  total_votes = 700 →
  winning_majority = 476 →
  winning_percentage = 84 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_majority :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l628_62844


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l628_62882

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x ≠ 0 ∧ x + 1 ≠ 0) ∧ (1 / x = 2 / (x + 1)) ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l628_62882


namespace NUMINAMATH_CALUDE_vector_problem_l628_62803

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-1, -1)

theorem vector_problem :
  let magnitude := Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2)
  magnitude = 3 * Real.sqrt 2 ∧
  (let angle := Real.arccos ((a.1 + b.1) * (2 * a.1 - b.1) + (a.2 + b.2) * (2 * a.2 - b.2)) /
    (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) * Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2))
   angle = π / 4 → a.1 * b.1 + a.2 * b.2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l628_62803


namespace NUMINAMATH_CALUDE_robin_cupcakes_l628_62816

/-- The number of cupcakes with chocolate sauce Robin ate -/
def chocolate_cupcakes : ℕ := sorry

/-- The number of cupcakes with buttercream frosting Robin ate -/
def buttercream_cupcakes : ℕ := sorry

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := 12

theorem robin_cupcakes :
  chocolate_cupcakes + buttercream_cupcakes = total_cupcakes ∧
  buttercream_cupcakes = 2 * chocolate_cupcakes →
  chocolate_cupcakes = 4 :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_l628_62816


namespace NUMINAMATH_CALUDE_lillian_mushroom_foraging_l628_62851

/-- Calculates the number of uncertain mushrooms given the total, safe, and poisonous counts. -/
def uncertain_mushrooms (total safe : ℕ) : ℕ :=
  total - (safe + 2 * safe)

/-- Proves that the number of uncertain mushrooms is 5 given the problem conditions. -/
theorem lillian_mushroom_foraging :
  uncertain_mushrooms 32 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillian_mushroom_foraging_l628_62851


namespace NUMINAMATH_CALUDE_count_less_than_10000_l628_62877

def count_numbers_with_at_most_three_digits (n : ℕ) : ℕ :=
  sorry

theorem count_less_than_10000 : 
  count_numbers_with_at_most_three_digits 10000 = 3231 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_10000_l628_62877


namespace NUMINAMATH_CALUDE_sum_of_cubes_plus_linear_positive_l628_62815

theorem sum_of_cubes_plus_linear_positive
  (a b c : ℝ)
  (hab : a + b > 0)
  (hac : a + c > 0)
  (hbc : b + c > 0) :
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_plus_linear_positive_l628_62815


namespace NUMINAMATH_CALUDE_two_car_speeds_l628_62866

/-- Represents the speed of two cars traveling in opposite directions -/
structure TwoCarSpeeds where
  slower : ℝ
  faster : ℝ
  speed_difference : faster = slower + 10
  total_distance : 5 * slower + 5 * faster = 500

/-- Theorem stating the speeds of the two cars -/
theorem two_car_speeds : ∃ (s : TwoCarSpeeds), s.slower = 45 ∧ s.faster = 55 := by
  sorry

end NUMINAMATH_CALUDE_two_car_speeds_l628_62866


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l628_62842

/-- Define a Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ (∀ n ≥ 2, F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l628_62842


namespace NUMINAMATH_CALUDE_similar_triangles_height_l628_62809

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 25 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l628_62809


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l628_62827

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l628_62827


namespace NUMINAMATH_CALUDE_gcd_108_45_l628_62891

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_108_45_l628_62891


namespace NUMINAMATH_CALUDE_geometric_series_sum_l628_62859

/-- The sum of a geometric series with first term 2, common ratio -2, and last term 1024 -/
def geometricSeriesSum : ℤ := -682

/-- The first term of the geometric series -/
def firstTerm : ℤ := 2

/-- The common ratio of the geometric series -/
def commonRatio : ℤ := -2

/-- The last term of the geometric series -/
def lastTerm : ℤ := 1024

theorem geometric_series_sum :
  ∃ (n : ℕ), n > 0 ∧ firstTerm * commonRatio^(n - 1) = lastTerm ∧
  geometricSeriesSum = firstTerm * (commonRatio^n - 1) / (commonRatio - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l628_62859


namespace NUMINAMATH_CALUDE_box_surface_area_l628_62828

/-- Calculates the surface area of the interior of an open box formed from a rectangular cardboard with square corners removed. -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Proves that the surface area of the interior of the specified open box is 731 square units. -/
theorem box_surface_area : interior_surface_area 25 35 6 = 731 := by
  sorry

#eval interior_surface_area 25 35 6

end NUMINAMATH_CALUDE_box_surface_area_l628_62828


namespace NUMINAMATH_CALUDE_gcd_45_75_l628_62846

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l628_62846


namespace NUMINAMATH_CALUDE_rectangle_division_count_l628_62824

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a division of a large rectangle into smaller rectangles --/
structure RectangleDivision where
  large : Rectangle
  small : Rectangle
  divisions : List (List ℕ)

/-- Counts the number of ways to divide a rectangle --/
def countDivisions (r : RectangleDivision) : ℕ :=
  r.divisions.length

/-- The main rectangle --/
def mainRectangle : Rectangle :=
  { width := 24, height := 20 }

/-- The sub-rectangle --/
def subRectangle : Rectangle :=
  { width := 5, height := 4 }

/-- The division of the main rectangle into sub-rectangles --/
def rectangleDivision : RectangleDivision :=
  { large := mainRectangle
    small := subRectangle
    divisions := [[4, 4, 4, 4, 4, 4], [4, 5, 5, 5, 5], [5, 4, 5, 5, 5], [5, 5, 4, 5, 5], [5, 5, 5, 4, 5], [5, 5, 5, 5, 4]] }

/-- Theorem stating that the number of ways to divide the rectangle is 6 --/
theorem rectangle_division_count : countDivisions rectangleDivision = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_count_l628_62824


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l628_62875

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: f(x) ≥ 2 for all x and a
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l628_62875


namespace NUMINAMATH_CALUDE_sample_capacity_l628_62889

/-- Given a sample divided into groups, prove that the sample capacity is 160
    when a certain group has a frequency of 20 and a frequency rate of 0.125. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ)
  (h1 : frequency = 20)
  (h2 : frequency_rate = 1/8)
  (h3 : (frequency : ℚ) / n = frequency_rate) :
  n = 160 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l628_62889


namespace NUMINAMATH_CALUDE_larger_number_proof_l628_62867

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 13 * 15) (h3 : a > b) : a = 345 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l628_62867


namespace NUMINAMATH_CALUDE_smallest_with_properties_l628_62880

def is_smallest_with_properties (n : ℕ) : Prop :=
  (∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, n % d = 0)) ∧
  (∃ (start : ℕ), ∀ i ∈ Finset.range 10, n % (start + i) = 0) ∧
  (∀ m < n, ¬(∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, m % d = 0)) ∨
           ¬(∃ (start : ℕ), ∀ i ∈ Finset.range 10, m % (start + i) = 0))

theorem smallest_with_properties : is_smallest_with_properties 110880 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_properties_l628_62880


namespace NUMINAMATH_CALUDE_yujeong_drank_most_l628_62871

/-- Represents the amount of water drunk by each person in liters. -/
structure WaterConsumption where
  eunji : ℚ
  yujeong : ℚ
  yuna : ℚ

/-- Determines who drank the most water given the water consumption of three people. -/
def who_drank_most (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

/-- Theorem stating that Yujeong drank the most water given the specific amounts. -/
theorem yujeong_drank_most :
  who_drank_most ⟨(1/2), (7/10), (6/10)⟩ = "Yujeong" := by
  sorry

#eval who_drank_most ⟨(1/2), (7/10), (6/10)⟩

end NUMINAMATH_CALUDE_yujeong_drank_most_l628_62871


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l628_62852

/-- The area of a right-angled isosceles triangle with hypotenuse length 1 is 1/4 -/
theorem isosceles_right_triangle_area (A B C : ℝ × ℝ) : 
  (A.1 = 0 ∧ A.2 = 0) →  -- A is at origin
  (B.1 = 1 ∧ B.2 = 0) →  -- B is at (1, 0)
  (C.1 = 0 ∧ C.2 = 1) →  -- C is at (0, 1)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- AB = AC (isosceles)
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →  -- right angle at A
  (1/2) * (B.1 - A.1) * (C.2 - A.2) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l628_62852


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l628_62829

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hoursMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hoursTT : ℕ   -- Hours worked on Tuesday, Thursday
  daysLong : ℕ  -- Number of days working long hours (MWF)
  daysShort : ℕ -- Number of days working short hours (TT)
  hourlyRate : ℕ -- Hourly rate in dollars

/-- Calculates weekly earnings based on work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℕ :=
  (schedule.hoursMWF * schedule.daysLong + schedule.hoursTT * schedule.daysShort) * schedule.hourlyRate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hoursMWF := 8,
    hoursTT := 6,
    daysLong := 3,
    daysShort := 2,
    hourlyRate := 11
  }
  weeklyEarnings schedule = 396 := by sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l628_62829


namespace NUMINAMATH_CALUDE_convex_ngon_regions_l628_62830

/-- The number of regions into which the diagonals of a convex n-gon divide it -/
def f (n : ℕ) : ℕ := (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24

/-- A convex n-gon is divided into f(n) regions by its diagonals, 
    given that no three diagonals intersect at a single point -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 3) : 
  f n = (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24 := by
  sorry

end NUMINAMATH_CALUDE_convex_ngon_regions_l628_62830


namespace NUMINAMATH_CALUDE_remi_seedlings_proof_l628_62888

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 400

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred over the two days -/
def total_seedlings : ℕ := 1200

theorem remi_seedlings_proof :
  first_day_seedlings + second_day_seedlings = total_seedlings ∧
  first_day_seedlings = 400 := by
  sorry

end NUMINAMATH_CALUDE_remi_seedlings_proof_l628_62888


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l628_62818

/-- Proves that the number of sheep is 32 given the farm conditions --/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  (sheep : ℚ) / (horses : ℚ) = 4 / 7 →
  horses * 230 = 12880 →
  sheep = 32 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l628_62818


namespace NUMINAMATH_CALUDE_cube_greater_than_one_iff_l628_62835

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_one_iff_l628_62835


namespace NUMINAMATH_CALUDE_step_waddle_difference_is_six_l628_62876

/-- The number of steps Gerald takes between consecutive lamp posts -/
def gerald_steps : ℕ := 55

/-- The number of waddles Patricia takes between consecutive lamp posts -/
def patricia_waddles : ℕ := 15

/-- The number of lamp posts -/
def num_posts : ℕ := 31

/-- The total distance between the first and last lamp post in feet -/
def total_distance : ℕ := 3720

/-- Gerald's step length in feet -/
def gerald_step_length : ℚ := total_distance / (gerald_steps * (num_posts - 1))

/-- Patricia's waddle length in feet -/
def patricia_waddle_length : ℚ := total_distance / (patricia_waddles * (num_posts - 1))

/-- The difference between Gerald's step length and Patricia's waddle length -/
def step_waddle_difference : ℚ := patricia_waddle_length - gerald_step_length

theorem step_waddle_difference_is_six :
  step_waddle_difference = 6 := by sorry

end NUMINAMATH_CALUDE_step_waddle_difference_is_six_l628_62876


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l628_62863

theorem framed_painting_ratio :
  ∀ (x : ℝ),
    x > 0 →
    (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 * (3/4) →
    (min (20 + 2*x) (30 + 6*x)) / (max (20 + 2*x) (30 + 6*x)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l628_62863


namespace NUMINAMATH_CALUDE_cost_of_corn_seeds_l628_62893

/-- The cost of corn seeds for a farmer's harvest -/
theorem cost_of_corn_seeds
  (fertilizer_pesticide_cost : ℕ)
  (labor_cost : ℕ)
  (bags_of_corn : ℕ)
  (profit_percentage : ℚ)
  (price_per_bag : ℕ)
  (h1 : fertilizer_pesticide_cost = 35)
  (h2 : labor_cost = 15)
  (h3 : bags_of_corn = 10)
  (h4 : profit_percentage = 1/10)
  (h5 : price_per_bag = 11) :
  ∃ (corn_seed_cost : ℕ),
    corn_seed_cost = 49 ∧
    (corn_seed_cost : ℚ) + fertilizer_pesticide_cost + labor_cost +
      (profit_percentage * (bags_of_corn * price_per_bag)) =
    bags_of_corn * price_per_bag :=
by sorry

end NUMINAMATH_CALUDE_cost_of_corn_seeds_l628_62893


namespace NUMINAMATH_CALUDE_conference_room_capacity_l628_62836

theorem conference_room_capacity 
  (num_rooms : ℕ) 
  (current_occupancy : ℕ) 
  (occupancy_ratio : ℚ) :
  num_rooms = 6 →
  current_occupancy = 320 →
  occupancy_ratio = 2/3 →
  (current_occupancy : ℚ) / occupancy_ratio / num_rooms = 80 := by
  sorry

end NUMINAMATH_CALUDE_conference_room_capacity_l628_62836


namespace NUMINAMATH_CALUDE_inequalities_proof_l628_62892

theorem inequalities_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) : 
  (a * b > a * c) ∧ (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l628_62892


namespace NUMINAMATH_CALUDE_angle_measure_l628_62850

theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 90) ∧ (90 - x = 2 * x + 15) ∧ (x = 25) :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_l628_62850


namespace NUMINAMATH_CALUDE_percentage_relation_l628_62807

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l628_62807


namespace NUMINAMATH_CALUDE_max_ab_value_l628_62898

/-- Given a > 0, b > 0, and f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (let g := fun x => 4 * x^3 - c * x^2 - 2 * d * x + 2
     ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), g x ≤ g 1 ∨ g x ≥ g 1) →
    a * b ≥ c * d) ∧ a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l628_62898


namespace NUMINAMATH_CALUDE_white_squares_42nd_row_l628_62885

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n + 1) / 2

/-- Theorem stating the number of white squares in the 42nd row -/
theorem white_squares_42nd_row :
  white_squares_in_row 42 = 42 := by
  sorry

end NUMINAMATH_CALUDE_white_squares_42nd_row_l628_62885


namespace NUMINAMATH_CALUDE_graph_composition_l628_62832

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 * (x + y + 1) = y^3 * (x + y + 1)

-- Define the components of the graph
def parabola_component (x y : ℝ) : Prop := x^2 = y^3 ∧ x + y + 1 ≠ 0
def line_component (x y : ℝ) : Prop := y = -x - 1

-- Theorem stating that the graph consists of a parabola and a line
theorem graph_composition :
  ∀ x y : ℝ, equation x y ↔ parabola_component x y ∨ line_component x y :=
sorry

end NUMINAMATH_CALUDE_graph_composition_l628_62832


namespace NUMINAMATH_CALUDE_max_volume_parallelepiped_l628_62814

/-- The volume of a rectangular parallelepiped with square base of side length x
    and lateral faces with perimeter 6 -/
def volume (x : ℝ) : ℝ := x^2 * (3 - x)

/-- The maximum volume of a rectangular parallelepiped with square base
    and lateral faces with perimeter 6 is 4 -/
theorem max_volume_parallelepiped :
  ∃ (x : ℝ), x > 0 ∧ x < 3 ∧
  (∀ (y : ℝ), y > 0 → y < 3 → volume y ≤ volume x) ∧
  volume x = 4 := by sorry

end NUMINAMATH_CALUDE_max_volume_parallelepiped_l628_62814


namespace NUMINAMATH_CALUDE_total_apples_picked_l628_62862

/-- The total number of apples picked by Mike, Nancy, and Keith is 16. -/
theorem total_apples_picked (mike_apples nancy_apples keith_apples : ℕ)
  (h1 : mike_apples = 7)
  (h2 : nancy_apples = 3)
  (h3 : keith_apples = 6) :
  mike_apples + nancy_apples + keith_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l628_62862


namespace NUMINAMATH_CALUDE_chloe_carrots_initial_count_l628_62806

/-- Proves that the initial number of carrots Chloe picked is 48, given the conditions of the problem. -/
theorem chloe_carrots_initial_count : ∃ x : ℕ, 
  (x - 45 + 42 = 45) ∧ 
  (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_chloe_carrots_initial_count_l628_62806


namespace NUMINAMATH_CALUDE_joanne_first_hour_coins_l628_62868

/-- Represents the number of coins Joanne collected in the first hour -/
def first_hour_coins : ℕ := sorry

/-- Represents the total number of coins collected in the second and third hours -/
def second_third_hour_coins : ℕ := 35

/-- Represents the number of coins collected in the fourth hour -/
def fourth_hour_coins : ℕ := 50

/-- Represents the number of coins given to the coworker -/
def coins_given_away : ℕ := 15

/-- Represents the total number of coins after the fourth hour -/
def total_coins : ℕ := 120

/-- Theorem stating that Joanne collected 15 coins in the first hour -/
theorem joanne_first_hour_coins : 
  first_hour_coins = 15 :=
by
  sorry

#check joanne_first_hour_coins

end NUMINAMATH_CALUDE_joanne_first_hour_coins_l628_62868


namespace NUMINAMATH_CALUDE_lineArrangements_eq_36_l628_62857

/-- The number of ways to arrange 3 students (who must stand together) and 2 teachers in a line -/
def lineArrangements : ℕ :=
  let studentsCount : ℕ := 3
  let teachersCount : ℕ := 2
  let unitsCount : ℕ := teachersCount + 1  -- Students count as one unit
  (Nat.factorial unitsCount) * (Nat.factorial studentsCount)

theorem lineArrangements_eq_36 : lineArrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_lineArrangements_eq_36_l628_62857


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l628_62855

theorem sqrt_nested_expression : 
  Real.sqrt (144 * Real.sqrt (64 * Real.sqrt 36)) = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l628_62855


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_sequence_sum_ratio_l628_62848

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) :
  geometric_sequence a₁ q (n + 1) = q * geometric_sequence a₁ q n := by sorry

theorem geometric_sequence_sum_ratio (a₁ : ℝ) :
  let q := -1/3
  (geometric_sequence a₁ q 1 + geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7) /
  (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 + geometric_sequence a₁ q 6 + geometric_sequence a₁ q 8) = -3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_sequence_sum_ratio_l628_62848


namespace NUMINAMATH_CALUDE_aisha_age_l628_62884

/-- Given the ages of Ali, Yusaf, and Umar, prove Aisha's age --/
theorem aisha_age (ali_age : ℕ) (yusaf_age : ℕ) (umar_age : ℕ) 
  (h1 : ali_age = 8)
  (h2 : ali_age = yusaf_age + 3)
  (h3 : umar_age = 2 * yusaf_age)
  (h4 : ∃ (aisha_age : ℕ), aisha_age = (ali_age + umar_age) / 2) :
  ∃ (aisha_age : ℕ), aisha_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_aisha_age_l628_62884


namespace NUMINAMATH_CALUDE_remainder_of_4n_mod_4_l628_62878

theorem remainder_of_4n_mod_4 (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_4n_mod_4_l628_62878


namespace NUMINAMATH_CALUDE_pears_left_l628_62861

theorem pears_left (jason_pears keith_pears mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l628_62861


namespace NUMINAMATH_CALUDE_kongming_total_score_l628_62813

/-- Represents a recruitment exam with a written test and an interview -/
structure RecruitmentExam where
  writtenTestWeight : Real
  interviewWeight : Real
  writtenTestScore : Real
  interviewScore : Real

/-- Calculates the total score for a recruitment exam -/
def totalScore (exam : RecruitmentExam) : Real :=
  exam.writtenTestScore * exam.writtenTestWeight + exam.interviewScore * exam.interviewWeight

theorem kongming_total_score :
  let exam : RecruitmentExam := {
    writtenTestWeight := 0.6,
    interviewWeight := 0.4,
    writtenTestScore := 90,
    interviewScore := 85
  }
  totalScore exam = 88 := by sorry

end NUMINAMATH_CALUDE_kongming_total_score_l628_62813


namespace NUMINAMATH_CALUDE_subset_of_A_l628_62825

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_of_A_l628_62825


namespace NUMINAMATH_CALUDE_seven_lines_twenty_two_regions_l628_62801

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  total_lines : ℕ
  parallel_lines : ℕ
  non_parallel_lines : ℕ
  no_concurrency : Prop
  no_other_parallel : Prop

/-- Calculate the number of regions formed by a given line configuration -/
def number_of_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- The theorem stating that the specific configuration of 7 lines creates 22 regions -/
theorem seven_lines_twenty_two_regions :
  ∀ (config : LineConfiguration),
    config.total_lines = 7 ∧
    config.parallel_lines = 2 ∧
    config.non_parallel_lines = 5 ∧
    config.no_concurrency ∧
    config.no_other_parallel →
    number_of_regions config = 22 :=
by sorry

end NUMINAMATH_CALUDE_seven_lines_twenty_two_regions_l628_62801


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l628_62869

theorem cubic_roots_sum_squares (p q r : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - r = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  x₁^2 + x₂^2 + x₃^2 = p^2 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l628_62869


namespace NUMINAMATH_CALUDE_monotonic_cubic_implies_a_geq_one_l628_62849

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = (1/3)x³ + x² + ax - 5 is monotonic for all real x, then a ≥ 1 -/
theorem monotonic_cubic_implies_a_geq_one (a : ℝ) :
  Monotonic (fun x => (1/3) * x^3 + x^2 + a*x - 5) → a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_monotonic_cubic_implies_a_geq_one_l628_62849


namespace NUMINAMATH_CALUDE_intersection_dot_product_l628_62858

/-- Given a line ax + by + c = 0 intersecting a circle x^2 + y^2 = 4 at points A and B,
    prove that the dot product of OA and OB is -2 when c^2 = a^2 + b^2 -/
theorem intersection_dot_product
  (a b c : ℝ) 
  (A B : ℝ × ℝ)
  (h1 : ∀ (x y : ℝ), a * x + b * y + c = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B)
  (h2 : c^2 = a^2 + b^2) :
  A.1 * B.1 + A.2 * B.2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l628_62858


namespace NUMINAMATH_CALUDE_twelve_star_x_multiple_of_144_l628_62886

def star (a b : ℤ) : ℤ := a^2 * b

theorem twelve_star_x_multiple_of_144 (x : ℤ) : ∃ k : ℤ, star 12 x = 144 * k := by
  sorry

end NUMINAMATH_CALUDE_twelve_star_x_multiple_of_144_l628_62886


namespace NUMINAMATH_CALUDE_third_number_divisible_by_seven_l628_62800

theorem third_number_divisible_by_seven (n : ℕ) : 
  (Nat.gcd 35 91 = 7) → (Nat.gcd (Nat.gcd 35 91) n = 7) → (n % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_third_number_divisible_by_seven_l628_62800


namespace NUMINAMATH_CALUDE_unique_solution_condition_l628_62843

theorem unique_solution_condition (a b : ℤ) : 
  (∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b) ↔ 4 * b = a^2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l628_62843


namespace NUMINAMATH_CALUDE_additive_inverse_of_2023_l628_62802

theorem additive_inverse_of_2023 : ∃! x : ℤ, 2023 + x = 0 ∧ x = -2023 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_2023_l628_62802


namespace NUMINAMATH_CALUDE_line_through_point_l628_62854

/-- Given a line equation 2 - 3kx = -4y that passes through the point (3, -2),
    prove that k = -2/3 is the unique value that satisfies the equation. -/
theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * 3 = -4 * (-2)) ↔ k = -2/3 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l628_62854


namespace NUMINAMATH_CALUDE_fish_moved_l628_62897

theorem fish_moved (initial_fish : ℝ) (remaining_fish : ℕ) 
  (h1 : initial_fish = 212.0) 
  (h2 : remaining_fish = 144) : 
  initial_fish - remaining_fish = 68 := by
  sorry

end NUMINAMATH_CALUDE_fish_moved_l628_62897


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l628_62887

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ Real.log (a₀ + b₀) = 0 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l628_62887


namespace NUMINAMATH_CALUDE_jessica_age_l628_62805

theorem jessica_age :
  (∀ (jessica_age claire_age : ℕ),
    jessica_age = claire_age + 6 →
    claire_age + 2 = 20 →
    jessica_age = 24) :=
by sorry

end NUMINAMATH_CALUDE_jessica_age_l628_62805


namespace NUMINAMATH_CALUDE_expand_and_simplify_l628_62823

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l628_62823


namespace NUMINAMATH_CALUDE_only_two_reduces_to_zero_l628_62841

/-- A move on a table is either subtracting n from a column or multiplying a row by n -/
inductive Move (n : ℕ+)
  | subtract_column : Move n
  | multiply_row : Move n

/-- A table is a rectangular array of positive integers -/
def Table := Array (Array ℕ+)

/-- Apply a move to a table -/
def apply_move (t : Table) (m : Move n) : Table :=
  sorry

/-- A table is reducible to zero if there exists a sequence of moves that makes all entries zero -/
def reducible_to_zero (t : Table) (n : ℕ+) : Prop :=
  sorry

/-- The main theorem: n = 2 is the only value that allows any table to be reduced to zero -/
theorem only_two_reduces_to_zero :
  ∀ n : ℕ+, (∀ t : Table, reducible_to_zero t n) ↔ n = 2 :=
  sorry

end NUMINAMATH_CALUDE_only_two_reduces_to_zero_l628_62841


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l628_62870

-- Given identity
axiom identity (a b : ℝ) : (a - b) * (a + b) = a^2 - b^2

-- Theorem 1
theorem problem_1 : (2 - 1) * (2 + 1) = 3 := by sorry

-- Theorem 2
theorem problem_2 : (2 + 1) * (2^2 + 1) = 15 := by sorry

-- Helper function to generate the product series
def product_series (n : ℕ) : ℝ :=
  if n = 0 then 2 + 1
  else (2^(2^n) + 1) * product_series (n-1)

-- Theorem 3
theorem problem_3 : product_series 5 = 2^64 - 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l628_62870


namespace NUMINAMATH_CALUDE_max_taxiing_time_l628_62896

/-- The function representing the distance traveled by the plane after landing -/
def y (t : ℝ) : ℝ := 60 * t - 2 * t^2

/-- The maximum time the plane uses for taxiing -/
def s : ℝ := 15

theorem max_taxiing_time :
  ∀ t : ℝ, y t ≤ y s :=
by sorry

end NUMINAMATH_CALUDE_max_taxiing_time_l628_62896


namespace NUMINAMATH_CALUDE_simplified_expression_l628_62883

theorem simplified_expression (a : ℤ) 
  (h1 : (a - 1) / 2 < 2)
  (h2 : (a + 1) / 2 ≥ (4 - a) / 3)
  (h3 : a ≠ 2)
  (h4 : a ≠ 4) :
  (16 - a^2) / (a^2 + 8*a + 16) / ((1 / 2) - (4 / (a + 4))) * (1 / (2*a - 4)) = -1 / (a - 2) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l628_62883


namespace NUMINAMATH_CALUDE_train_speed_l628_62879

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) (h1 : length = 200) (h2 : time = 12) :
  (length / 1000) / (time / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l628_62879


namespace NUMINAMATH_CALUDE_ceramic_cup_price_l628_62831

theorem ceramic_cup_price 
  (total_cups : ℕ) 
  (total_revenue : ℚ) 
  (plastic_cup_price : ℚ) 
  (ceramic_cups_sold : ℕ) 
  (plastic_cups_sold : ℕ) :
  total_cups = 400 →
  total_revenue = 1458 →
  plastic_cup_price = (7/2) →
  ceramic_cups_sold = 284 →
  plastic_cups_sold = 116 →
  (total_revenue - (plastic_cup_price * plastic_cups_sold)) / ceramic_cups_sold = (37/10) := by
  sorry

end NUMINAMATH_CALUDE_ceramic_cup_price_l628_62831


namespace NUMINAMATH_CALUDE_larger_number_proof_l628_62808

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 28)
  (lcm_eq : Nat.lcm a b = 28 * 12 * 15) :
  max a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l628_62808


namespace NUMINAMATH_CALUDE_three_numbers_problem_l628_62899

theorem three_numbers_problem (a b c : ℝ) :
  ((a + 1) * (b + 1) * (c + 1) = a * b * c + 1) →
  ((a + 2) * (b + 2) * (c + 2) = a * b * c + 2) →
  (a = -1 ∧ b = -1 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l628_62899


namespace NUMINAMATH_CALUDE_gargamel_tire_purchase_l628_62894

def sale_price : ℕ := 75
def total_savings : ℕ := 36
def original_price : ℕ := 84

theorem gargamel_tire_purchase :
  (total_savings / (original_price - sale_price) : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gargamel_tire_purchase_l628_62894


namespace NUMINAMATH_CALUDE_morning_orange_sales_l628_62817

/-- Proves the number of oranges sold in the morning given fruit prices and sales data --/
theorem morning_orange_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ morning_oranges : ℕ,
    morning_oranges = 30 ∧
    total_sales = apple_price * (morning_apples + afternoon_apples : ℚ) +
                  orange_price * (morning_oranges + afternoon_oranges : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_morning_orange_sales_l628_62817


namespace NUMINAMATH_CALUDE_beef_weight_after_processing_l628_62821

theorem beef_weight_after_processing (initial_weight : ℝ) (loss_percentage : ℝ) 
  (processed_weight : ℝ) (h1 : initial_weight = 840) (h2 : loss_percentage = 35) :
  processed_weight = initial_weight * (1 - loss_percentage / 100) → 
  processed_weight = 546 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_after_processing_l628_62821


namespace NUMINAMATH_CALUDE_tuesday_temperature_l628_62881

def sunday_temp : ℝ := 40
def monday_temp : ℝ := 50
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem tuesday_temperature :
  ∃ tuesday_temp : ℝ,
    (sunday_temp + monday_temp + tuesday_temp + wednesday_temp +
     thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp ∧
    tuesday_temp = 65 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l628_62881
