import Mathlib

namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_two_l1470_147029

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

theorem odd_function_implies_a_equals_two (a : ℝ) :
  (∀ x, f a (x + 1) = -f a (-x + 1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_two_l1470_147029


namespace NUMINAMATH_CALUDE_bounce_height_theorem_l1470_147089

/-- The number of bounces required for a ball to reach a height less than 3 meters -/
def number_of_bounces : ℕ := 22

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The bounce ratio (percentage of height retained after each bounce) -/
def bounce_ratio : ℝ := 0.6

/-- The target height in meters -/
def target_height : ℝ := 3

/-- Theorem stating that the number of bounces is correct -/
theorem bounce_height_theorem :
  (∀ k : ℕ, k < number_of_bounces → initial_height * bounce_ratio ^ k ≥ target_height) ∧
  (initial_height * bounce_ratio ^ number_of_bounces < target_height) :=
sorry

end NUMINAMATH_CALUDE_bounce_height_theorem_l1470_147089


namespace NUMINAMATH_CALUDE_no_order_for_seven_l1470_147082

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem no_order_for_seven :
  ¬ ∃ n : ℕ, n > 0 ∧ iterate_f n 7 = 7 :=
sorry

end NUMINAMATH_CALUDE_no_order_for_seven_l1470_147082


namespace NUMINAMATH_CALUDE_point_on_line_l1470_147054

/-- Given a line defined by x = (y / 2) - (2 / 5), if (m, n) and (m + p, n + 4) both lie on this line, then p = 2 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 2 - 2 / 5) →
  (m + p = (n + 4) / 2 - 2 / 5) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1470_147054


namespace NUMINAMATH_CALUDE_four_row_triangle_count_l1470_147093

/-- Calculates the total number of triangles in a triangular grid with n rows -/
def triangleCount (n : ℕ) : ℕ :=
  let smallTriangles := n * (n + 1) / 2
  let mediumTriangles := (n - 1) * (n - 2) / 2
  let largeTriangles := n - 2
  smallTriangles + mediumTriangles + largeTriangles

/-- Theorem stating that a triangular grid with 4 rows contains 14 triangles in total -/
theorem four_row_triangle_count : triangleCount 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_row_triangle_count_l1470_147093


namespace NUMINAMATH_CALUDE_equation_solutions_l1470_147006

theorem equation_solutions (n : ℕ+) :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 15 ∧ 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + z = n)) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1470_147006


namespace NUMINAMATH_CALUDE_certain_number_proof_l1470_147026

theorem certain_number_proof : ∃ x : ℕ, 9873 + x = 13200 ∧ x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1470_147026


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1470_147055

/-- The cost price of a bicycle for seller A, given the selling conditions and final price. -/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1470_147055


namespace NUMINAMATH_CALUDE_lucy_shell_count_l1470_147044

/-- Lucy's shell counting problem -/
theorem lucy_shell_count (initial_shells final_shells : ℕ) 
  (h1 : initial_shells = 68) 
  (h2 : final_shells = 89) : 
  final_shells - initial_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shell_count_l1470_147044


namespace NUMINAMATH_CALUDE_system_solution_l1470_147084

theorem system_solution (x y k : ℝ) : 
  (2 * x - y = 5 * k + 6) → 
  (4 * x + 7 * y = k) → 
  (x + y = 2023) → 
  (k = 2022) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1470_147084


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1470_147030

def A : ℝ × ℝ := (5, 1)
def B : ℝ × ℝ := (7, -3)
def C : ℝ × ℝ := (2, -8)

def circumcircle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 25

theorem circumcircle_of_triangle_ABC :
  circumcircle_equation A.1 A.2 ∧
  circumcircle_equation B.1 B.2 ∧
  circumcircle_equation C.1 C.2 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1470_147030


namespace NUMINAMATH_CALUDE_min_value_of_f_l1470_147041

/-- Given positive numbers a, b, c, x, y, z satisfying the conditions,
    the function f(x, y, z) has a minimum value of 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  ∀ (x' y' z' : ℝ), 0 < x' → 0 < y' → 0 < z' →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1470_147041


namespace NUMINAMATH_CALUDE_not_increasing_on_interval_l1470_147069

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem not_increasing_on_interval : ¬ IsIncreasing f 0 2 := by
  sorry

end NUMINAMATH_CALUDE_not_increasing_on_interval_l1470_147069


namespace NUMINAMATH_CALUDE_problem_statement_l1470_147027

theorem problem_statement :
  ∀ x y : ℝ,
  x = 98 * 1.2 →
  y = (x + 35) * 0.9 →
  2 * y - 3 * x = -78.12 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1470_147027


namespace NUMINAMATH_CALUDE_choir_average_age_l1470_147092

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 8)
  (h2 : avg_age_females = 25)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 40)
  (h5 : num_females + num_males = 20) :
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l1470_147092


namespace NUMINAMATH_CALUDE_quartic_real_root_l1470_147043

theorem quartic_real_root 
  (A B C D E : ℝ) 
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
sorry

end NUMINAMATH_CALUDE_quartic_real_root_l1470_147043


namespace NUMINAMATH_CALUDE_number_puzzle_l1470_147021

theorem number_puzzle :
  ∀ (a b : ℤ),
  a + b = 72 →
  a = b + 12 →
  (a = 30 ∨ b = 30) →
  (a = 18 ∨ b = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1470_147021


namespace NUMINAMATH_CALUDE_height_equals_base_l1470_147037

/-- An isosceles triangle with constant perimeter for inscribed rectangles -/
structure ConstantPerimeterTriangle where
  -- The base of the triangle
  base : ℝ
  -- The height of the triangle
  height : ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The perimeter of any inscribed rectangle is constant
  constantPerimeter : True

/-- Theorem: In a ConstantPerimeterTriangle, the height equals the base -/
theorem height_equals_base (t : ConstantPerimeterTriangle) : t.height = t.base := by
  sorry

end NUMINAMATH_CALUDE_height_equals_base_l1470_147037


namespace NUMINAMATH_CALUDE_video_games_spending_l1470_147047

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/7
def video_games_fraction : ℚ := 2/7
def snacks_fraction : ℚ := 1/2
def clothes_fraction : ℚ := 3/14

def video_games_spent : ℚ := total_allowance * video_games_fraction

theorem video_games_spending :
  video_games_spent = 7.15 := by sorry

end NUMINAMATH_CALUDE_video_games_spending_l1470_147047


namespace NUMINAMATH_CALUDE_parallel_lines_iff_coplanar_l1470_147031

-- Define the types for points and planes
variable (Point Plane : Type*)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the "on_plane" relation for points and planes
variable (on_plane : Point → Plane → Prop)

-- Define the parallel relation for lines (represented by two points each)
variable (parallel_lines : (Point × Point) → (Point × Point) → Prop)

-- Define the coplanar relation for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallel_lines_iff_coplanar
  (α β : Plane) (A B C D : Point)
  (h_planes_parallel : parallel_planes α β)
  (h_A_on_α : on_plane A α)
  (h_C_on_α : on_plane C α)
  (h_B_on_β : on_plane B β)
  (h_D_on_β : on_plane D β) :
  parallel_lines (A, C) (B, D) ↔ coplanar A B C D :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_coplanar_l1470_147031


namespace NUMINAMATH_CALUDE_total_spent_calculation_l1470_147083

/-- Calculates the total amount spent on t-shirts given the prices, quantities, discount, and tax rate -/
def total_spent (price_a price_b price_c : ℚ) (qty_a qty_b qty_c : ℕ) (discount_b tax_rate : ℚ) : ℚ :=
  let subtotal_a := price_a * qty_a
  let subtotal_b := price_b * qty_b * (1 - discount_b)
  let subtotal_c := price_c * qty_c
  let total_before_tax := subtotal_a + subtotal_b + subtotal_c
  total_before_tax * (1 + tax_rate)

/-- Theorem stating that given the specific conditions, the total amount spent is $695.21 -/
theorem total_spent_calculation :
  total_spent 9.95 12.50 14.95 18 23 15 0.1 0.05 = 695.21 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l1470_147083


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1470_147023

/-- The area of a circle with diameter 8 meters is 16π square meters -/
theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1470_147023


namespace NUMINAMATH_CALUDE_f_2_equals_5_l1470_147049

def f (x : ℝ) : ℝ := 2 * (x - 1) + 3

theorem f_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_5_l1470_147049


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_correct_l1470_147038

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_semi_prime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

def max_consecutive_semi_primes : ℕ := 5

theorem max_consecutive_semi_primes_correct :
  (∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧
    (∀ k : ℕ, k < max_consecutive_semi_primes → is_semi_prime (m + k))) ∧
  (∀ n : ℕ, ¬∃ m : ℕ, 
    (∀ k : ℕ, k < max_consecutive_semi_primes + 1 → is_semi_prime (m + k))) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_correct_l1470_147038


namespace NUMINAMATH_CALUDE_crayons_remaining_l1470_147051

/-- Given a drawer with 7 crayons initially, prove that after removing 3 crayons, 4 crayons remain. -/
theorem crayons_remaining (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 7 → removed = 3 → remaining = initial - removed → remaining = 4 := by sorry

end NUMINAMATH_CALUDE_crayons_remaining_l1470_147051


namespace NUMINAMATH_CALUDE_problem_solution_l1470_147042

def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (1 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1470_147042


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1470_147000

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem infinitely_many_divisible_pairs :
  ∀ n : ℕ, ∃ a b : ℕ,
    a = fib (2 * n + 1) ∧
    b = fib (2 * n + 3) ∧
    a > 0 ∧
    b > 0 ∧
    a ∣ (b^2 + 1) ∧
    b ∣ (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1470_147000


namespace NUMINAMATH_CALUDE_words_removed_during_editing_l1470_147066

theorem words_removed_during_editing 
  (yvonne_words : ℕ)
  (janna_words : ℕ)
  (words_removed : ℕ)
  (words_added : ℕ)
  (h1 : yvonne_words = 400)
  (h2 : janna_words = yvonne_words + 150)
  (h3 : words_added = 2 * words_removed)
  (h4 : yvonne_words + janna_words - words_removed + words_added + 30 = 1000) :
  words_removed = 20 := by
  sorry

end NUMINAMATH_CALUDE_words_removed_during_editing_l1470_147066


namespace NUMINAMATH_CALUDE_absolute_value_difference_l1470_147091

theorem absolute_value_difference (m n : ℝ) (hm : m < 0) (hmn : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l1470_147091


namespace NUMINAMATH_CALUDE_ball_count_equals_hex_sum_ball_count_2010_l1470_147088

/-- Converts a natural number to its hexadecimal representation -/
def toHex (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball-placing process for n steps -/
def ballCount (n : ℕ) : ℕ :=
  sorry

theorem ball_count_equals_hex_sum (n : ℕ) : 
  ballCount n = sumDigits (toHex n) := by
  sorry

theorem ball_count_2010 : 
  ballCount 2010 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_equals_hex_sum_ball_count_2010_l1470_147088


namespace NUMINAMATH_CALUDE_min_value_inequality_l1470_147032

theorem min_value_inequality (a b m n : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → 
  a + b = 1 → m * n = 2 → 
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1470_147032


namespace NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1470_147045

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : n > 2) :
  (180 * (n - 2) = 720) → (360 / n = 60) := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1470_147045


namespace NUMINAMATH_CALUDE_boat_distance_upstream_l1470_147099

/-- Proves that the distance travelled upstream is 10 km given the conditions of the boat problem -/
theorem boat_distance_upstream 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : upstream_time = 1) 
  (h3 : downstream_time = 0.25) : 
  (boat_speed - ((boat_speed * upstream_time - boat_speed * downstream_time) / (upstream_time + downstream_time))) * upstream_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_upstream_l1470_147099


namespace NUMINAMATH_CALUDE_distance_between_points_l1470_147087

/-- The distance between two points on a plane is the square root of the sum of squares of differences in their coordinates. -/
theorem distance_between_points (A B : ℝ × ℝ) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 26 :=
by
  -- Given points A(2,1) and B(-3,2)
  have hA : A = (2, 1) := by sorry
  have hB : B = (-3, 2) := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1470_147087


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1470_147097

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shaded quadrilateral -/
structure ShadedQuadrilateral where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The side length of the large square -/
def largeSideLength : ℝ := 10

/-- The side length of each small square in the grid -/
def smallSideLength : ℝ := 2

/-- The number of squares in each row/column of the grid -/
def gridSize : ℕ := 5

/-- Function to calculate the area of the shaded quadrilateral -/
def shadedArea (quad : ShadedQuadrilateral) : ℝ := sorry

/-- Function to create the shaded quadrilateral based on the problem description -/
def createShadedQuadrilateral : ShadedQuadrilateral := sorry

/-- Theorem stating the ratio of shaded area to large square area -/
theorem shaded_area_ratio :
  let quad := createShadedQuadrilateral
  shadedArea quad / (largeSideLength ^ 2) = 1 / 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1470_147097


namespace NUMINAMATH_CALUDE_set_intersection_proof_l1470_147068

def A : Set ℝ := {x : ℝ | |2*x - 1| < 6}
def B : Set ℝ := {-3, 0, 1, 2, 3, 4}

theorem set_intersection_proof : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l1470_147068


namespace NUMINAMATH_CALUDE_min_value_E_l1470_147081

theorem min_value_E (E : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, |E| + |y + 7| + |y - 5| ≥ |E| + |x + 7| + |x - 5| ∧ |E| + |x + 7| + |x - 5| = 12) →
  |E| ≥ 0 ∧ ∀ δ > 0, ∃ x : ℝ, |E| + |x + 7| + |x - 5| < 12 + δ :=
by sorry

end NUMINAMATH_CALUDE_min_value_E_l1470_147081


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1470_147095

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1470_147095


namespace NUMINAMATH_CALUDE_kaleb_books_l1470_147073

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books ≥ sold_books →
  initial_books - sold_books + new_books = initial_books + new_books - sold_books :=
by
  sorry

#check kaleb_books 34 17 7

end NUMINAMATH_CALUDE_kaleb_books_l1470_147073


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1470_147035

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) →  -- f(x) has value 0 at x = -1
  (f_deriv a b (-1) = 0) →  -- f'(x) = 0 at x = -1 (condition for extreme value)
  (a - b = -7) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1470_147035


namespace NUMINAMATH_CALUDE_stratified_sampling_is_most_suitable_l1470_147079

structure Population where
  male : ℕ
  female : ℕ

structure Sample where
  male : ℕ
  female : ℕ

def isStratifiedSampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male : ℚ) / (pop.female : ℚ) = (samp.male : ℚ) / (samp.female : ℚ)

def isMostSuitableMethod (method : String) (pop : Population) (samp : Sample) : Prop :=
  method = "Stratified sampling" ∧ isStratifiedSampling pop samp

theorem stratified_sampling_is_most_suitable :
  let pop : Population := { male := 500, female := 400 }
  let samp : Sample := { male := 25, female := 20 }
  isMostSuitableMethod "Stratified sampling" pop samp :=
by
  sorry

#check stratified_sampling_is_most_suitable

end NUMINAMATH_CALUDE_stratified_sampling_is_most_suitable_l1470_147079


namespace NUMINAMATH_CALUDE_luke_coin_piles_l1470_147013

theorem luke_coin_piles (piles_quarters piles_dimes : ℕ) 
  (h1 : piles_quarters = piles_dimes)
  (h2 : 3 * piles_quarters + 3 * piles_dimes = 30) : 
  piles_quarters = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_piles_l1470_147013


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1470_147075

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1470_147075


namespace NUMINAMATH_CALUDE_probability_mean_greater_than_median_l1470_147005

/-- A fair six-sided die --/
def Die : Type := Fin 6

/-- The result of rolling three dice --/
structure ThreeDiceRoll :=
  (d1 d2 d3 : Die)

/-- The sample space of all possible outcomes when rolling three dice --/
def sampleSpace : Finset ThreeDiceRoll := sorry

/-- The mean of a three dice roll --/
def mean (roll : ThreeDiceRoll) : ℚ := sorry

/-- The median of a three dice roll --/
def median (roll : ThreeDiceRoll) : ℚ := sorry

/-- The event where the mean is greater than the median --/
def meanGreaterThanMedian : Finset ThreeDiceRoll := sorry

theorem probability_mean_greater_than_median :
  (meanGreaterThanMedian.card : ℚ) / sampleSpace.card = 29 / 72 := by sorry

end NUMINAMATH_CALUDE_probability_mean_greater_than_median_l1470_147005


namespace NUMINAMATH_CALUDE_floor_divisibility_l1470_147018

theorem floor_divisibility (n : ℕ) : 
  ∃ k : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+1) * k) ∧ 
           ¬∃ m : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+2) * m) :=
by sorry

end NUMINAMATH_CALUDE_floor_divisibility_l1470_147018


namespace NUMINAMATH_CALUDE_additional_push_ups_l1470_147056

def push_ups (x : ℕ) : ℕ → ℕ
  | 1 => 10
  | 2 => 10 + x
  | 3 => 10 + 2*x
  | _ => 0

theorem additional_push_ups :
  ∃ x : ℕ, (push_ups x 1 + push_ups x 2 + push_ups x 3 = 45) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_push_ups_l1470_147056


namespace NUMINAMATH_CALUDE_min_hits_in_square_l1470_147015

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a square -/
def SquareConfiguration := List Point

/-- Function to determine if a point is hit -/
def isHit (config : SquareConfiguration) (p : Point) : Bool := sorry

/-- Function to count the number of hits in a configuration -/
def countHits (config : SquareConfiguration) : Nat :=
  (config.filter (isHit config)).length

/-- Theorem stating the existence of a configuration with minimum 10 hits -/
theorem min_hits_in_square (n : Nat) (h : n = 50) :
  ∃ (config : SquareConfiguration),
    config.length = n ∧
    countHits config = 10 ∧
    ∀ (other_config : SquareConfiguration),
      other_config.length = n →
      countHits other_config ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_hits_in_square_l1470_147015


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1470_147028

/-- The number of ways to distribute volunteers to pavilions -/
def distribute_volunteers (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Two specific volunteers cannot be in the same pavilion -/
def separate_volunteers (n : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 5 3 2 - separate_volunteers 3 = 114 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1470_147028


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1470_147086

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1470_147086


namespace NUMINAMATH_CALUDE_daniela_shopping_cost_l1470_147053

-- Define the original prices and discount rates
def shoe_price : ℝ := 50
def dress_price : ℝ := 100
def shoe_discount : ℝ := 0.4
def dress_discount : ℝ := 0.2

-- Define the number of items purchased
def num_shoes : ℕ := 2
def num_dresses : ℕ := 1

-- Calculate the discounted prices
def discounted_shoe_price : ℝ := shoe_price * (1 - shoe_discount)
def discounted_dress_price : ℝ := dress_price * (1 - dress_discount)

-- Calculate the total cost
def total_cost : ℝ := num_shoes * discounted_shoe_price + num_dresses * discounted_dress_price

-- Theorem statement
theorem daniela_shopping_cost : total_cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_daniela_shopping_cost_l1470_147053


namespace NUMINAMATH_CALUDE_min_detectors_for_specific_board_and_ship_l1470_147052

/-- Represents a grid board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a detector placement strategy -/
def DetectorStrategy := ℕ → ℕ → Bool

/-- Checks if a detector strategy can determine the ship's position -/
def can_determine_position (b : Board) (s : Ship) (strategy : DetectorStrategy) : Prop :=
  sorry

/-- The minimum number of detectors required -/
def min_detectors (b : Board) (s : Ship) : ℕ :=
  sorry

theorem min_detectors_for_specific_board_and_ship :
  let b : Board := ⟨2015, 2015⟩
  let s : Ship := ⟨1500, 1500⟩
  min_detectors b s = 1030 :=
sorry

end NUMINAMATH_CALUDE_min_detectors_for_specific_board_and_ship_l1470_147052


namespace NUMINAMATH_CALUDE_parabola_rotation_l1470_147014

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point (x, y) by 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola y = x^2 - 6x -/
def original_parabola : Parabola := { a := 1, b := -6, c := 0 }

/-- The rotated parabola y = -(x+3)^2 + 9 -/
def rotated_parabola : Parabola := { a := -1, b := -6, c := 9 }

theorem parabola_rotation :
  ∀ x y : ℝ,
  y = original_parabola.a * x^2 + original_parabola.b * x + original_parabola.c →
  let (x', y') := rotate180 x y
  y' = rotated_parabola.a * x'^2 + rotated_parabola.b * x' + rotated_parabola.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_rotation_l1470_147014


namespace NUMINAMATH_CALUDE_complex_fraction_eq_i_l1470_147034

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_fraction_eq_i (a b : ℝ) (h : complex a b = Complex.I * (2 - Complex.I)) :
  (complex b a) / (complex a (-b)) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_eq_i_l1470_147034


namespace NUMINAMATH_CALUDE_joan_pinball_spending_l1470_147001

def half_dollar_value : ℚ := 0.5

theorem joan_pinball_spending (wednesday_spent : ℕ) (total_spent : ℚ) 
  (h1 : wednesday_spent = 4)
  (h2 : total_spent = 9)
  : ℕ := by
  sorry

#check joan_pinball_spending

end NUMINAMATH_CALUDE_joan_pinball_spending_l1470_147001


namespace NUMINAMATH_CALUDE_abs_z_equals_one_l1470_147050

theorem abs_z_equals_one (r : ℝ) (z : ℂ) (h1 : |r| < Real.sqrt 8) (h2 : z + 1/z = r) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_z_equals_one_l1470_147050


namespace NUMINAMATH_CALUDE_michael_candy_distribution_l1470_147002

def minimum_additional_candies (initial_candies : Nat) (num_friends : Nat) : Nat :=
  (num_friends - initial_candies % num_friends) % num_friends

theorem michael_candy_distribution :
  minimum_additional_candies 25 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_michael_candy_distribution_l1470_147002


namespace NUMINAMATH_CALUDE_rhombus_count_in_divided_triangle_l1470_147009

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents a rhombus made of smaller equilateral triangles --/
structure Rhombus where
  smallTriangles : ℕ

/-- Counts the number of rhombuses in an equilateral triangle --/
def countRhombuses (triangle : EquilateralTriangle) (rhombus : Rhombus) : ℕ :=
  sorry

/-- Theorem statement --/
theorem rhombus_count_in_divided_triangle 
  (largeTriangle : EquilateralTriangle) 
  (smallTriangle : EquilateralTriangle) 
  (rhombus : Rhombus) :
  largeTriangle.sideLength = 10 →
  smallTriangle.sideLength = 1 →
  rhombus.smallTriangles = 8 →
  (largeTriangle.sideLength / smallTriangle.sideLength) ^ 2 = 100 →
  countRhombuses largeTriangle rhombus = 84 :=
sorry

end NUMINAMATH_CALUDE_rhombus_count_in_divided_triangle_l1470_147009


namespace NUMINAMATH_CALUDE_max_interesting_in_five_l1470_147007

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for interesting numbers -/
def is_interesting (n : ℕ) : Prop := is_prime (sum_of_digits n)

/-- Theorem: At most 4 out of 5 consecutive natural numbers can be interesting -/
theorem max_interesting_in_five (n : ℕ) : 
  ∃ (k : Fin 5), ¬is_interesting (n + k) :=
sorry

end NUMINAMATH_CALUDE_max_interesting_in_five_l1470_147007


namespace NUMINAMATH_CALUDE_complex_fourth_power_l1470_147004

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l1470_147004


namespace NUMINAMATH_CALUDE_candy_distribution_l1470_147003

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_heart_bags = 2)
  (h4 : chocolate_kiss_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_heart_bags + chocolate_kiss_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1470_147003


namespace NUMINAMATH_CALUDE_parallel_neither_sufficient_nor_necessary_l1470_147008

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_neither_sufficient_nor_necessary 
  (a b : Line) (α : Plane) 
  (h : line_in_plane b α) :
  ¬(∀ a b α, parallel_lines a b → parallel_line_plane a α) ∧ 
  ¬(∀ a b α, parallel_line_plane a α → parallel_lines a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_neither_sufficient_nor_necessary_l1470_147008


namespace NUMINAMATH_CALUDE_new_rectangle_perimeter_l1470_147094

/-- Given a rectangle ABCD composed of four congruent triangles -/
structure Rectangle :=
  (AB BC : ℝ)
  (AK : ℝ)
  (perimeter : ℝ)
  (h1 : perimeter = 4 * (AB + BC / 2 + AK))
  (h2 : AK = 17)
  (h3 : perimeter = 180)

/-- The perimeter of a new rectangle with sides 2*AB and BC -/
def new_perimeter (r : Rectangle) : ℝ :=
  2 * (2 * r.AB + r.BC)

/-- Theorem stating the perimeter of the new rectangle is 112 cm -/
theorem new_rectangle_perimeter (r : Rectangle) : new_perimeter r = 112 :=
sorry

end NUMINAMATH_CALUDE_new_rectangle_perimeter_l1470_147094


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1470_147010

/-- Given an ellipse with minor axis length 8 and eccentricity 3/5,
    prove that the perimeter of a triangle formed by two points where a line
    through one focus intersects the ellipse and the other focus is 20. -/
theorem ellipse_triangle_perimeter (b : ℝ) (e : ℝ) (a : ℝ) (c : ℝ) 
    (h1 : b = 4)  -- Half of the minor axis length
    (h2 : e = 3/5)  -- Eccentricity
    (h3 : e = c/a)  -- Definition of eccentricity
    (h4 : a^2 = b^2 + c^2)  -- Ellipse equation
    : 4 * a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1470_147010


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1470_147070

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1470_147070


namespace NUMINAMATH_CALUDE_triangle_angle_b_triangle_sides_l1470_147098

/-- Theorem: In an acute triangle ABC, if b*cos(C) + √3*b*sin(C) = a + c, then B = π/3 -/
theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c →
  B = π/3 := by
  sorry

/-- Corollary: If b = 2 and the area of triangle ABC is √3, then a = 2 and c = 2 -/
theorem triangle_sides (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  a = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_triangle_sides_l1470_147098


namespace NUMINAMATH_CALUDE_negative_square_nonpositive_l1470_147048

theorem negative_square_nonpositive (a : ℚ) : -a^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_nonpositive_l1470_147048


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1470_147085

theorem max_value_of_expression (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (h1 : x^2 + y^2 - x*y/2 = 36)
  (h2 : w^2 + z^2 + w*z/2 = 36)
  (h3 : x*z + y*w = 30) :
  (x*y + w*z)^2 ≤ 960 ∧ ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^2 + b^2 - a*b/2 = 36 ∧
    d^2 + c^2 + d*c/2 = 36 ∧
    a*c + b*d = 30 ∧
    (a*b + d*c)^2 = 960 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1470_147085


namespace NUMINAMATH_CALUDE_sally_and_jolly_money_l1470_147062

theorem sally_and_jolly_money (total : ℕ) (jolly_plus_20 : ℕ) :
  total = 150 →
  jolly_plus_20 = 70 →
  ∃ (sally : ℕ) (jolly : ℕ),
    sally + jolly = total ∧
    jolly + 20 = jolly_plus_20 ∧
    sally = 100 ∧
    jolly = 50 :=
by sorry

end NUMINAMATH_CALUDE_sally_and_jolly_money_l1470_147062


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l1470_147067

/-- The depth of Northton's time capsule given Southton's depth and the relationship between them. -/
theorem northton_capsule_depth (southton_depth : ℕ) (h1 : southton_depth = 15) :
  southton_depth * 4 - 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l1470_147067


namespace NUMINAMATH_CALUDE_fourth_vertex_not_in_third_quadrant_l1470_147020

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem fourth_vertex_not_in_third_quadrant :
  ∀ (p : Parallelogram),
    p.A = ⟨2, 0⟩ →
    p.B = ⟨-1/2, 0⟩ →
    p.C = ⟨0, 1⟩ →
    ¬(isInThirdQuadrant p.D) :=
by sorry

end NUMINAMATH_CALUDE_fourth_vertex_not_in_third_quadrant_l1470_147020


namespace NUMINAMATH_CALUDE_double_reflection_of_F_l1470_147046

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem double_reflection_of_F (F : ℝ × ℝ) (h : F = (-2, 1)) :
  (reflect_over_x_axis ∘ reflect_over_y_axis) F = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_F_l1470_147046


namespace NUMINAMATH_CALUDE_age_difference_is_nine_l1470_147022

/-- Represents a year in the 19th or 20th century -/
structure Year where
  century : Nat
  tens : Nat
  ones : Nat
  h : century ∈ [18, 19] ∧ tens < 10 ∧ ones < 10

/-- The age of a person at a given meeting year -/
def age (birth : Year) (meetingYear : Nat) : Nat :=
  meetingYear - (birth.century * 100 + birth.tens * 10 + birth.ones)

theorem age_difference_is_nine :
  ∀ (peterBirth : Year) (paulBirth : Year) (meetingYear : Nat),
    peterBirth.century = 18 →
    paulBirth.century = 19 →
    age peterBirth meetingYear = peterBirth.century + peterBirth.tens + peterBirth.ones + 9 →
    age paulBirth meetingYear = paulBirth.century + paulBirth.tens + paulBirth.ones + 10 →
    age peterBirth meetingYear - age paulBirth meetingYear = 9 := by
  sorry

#check age_difference_is_nine

end NUMINAMATH_CALUDE_age_difference_is_nine_l1470_147022


namespace NUMINAMATH_CALUDE_binomial_sum_equals_240_l1470_147090

theorem binomial_sum_equals_240 : Nat.choose 10 3 + Nat.choose 10 7 = 240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_240_l1470_147090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1470_147024

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + 3 * a 6 + a 11 = 120) : 
  2 * a 7 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1470_147024


namespace NUMINAMATH_CALUDE_sample_for_x_24_possible_x_for_87_l1470_147065

/-- Represents a systematic sampling method for a population of 1000 individuals. -/
def systematicSample (x : Nat) : List Nat :=
  List.range 10
    |>.map (fun k => (x + 33 * k) % 1000)

/-- Checks if a number ends with given digits. -/
def endsWithDigits (n : Nat) (digits : Nat) : Bool :=
  n % 100 = digits

/-- Theorem for the first part of the problem. -/
theorem sample_for_x_24 :
    systematicSample 24 = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921] := by
  sorry

/-- Theorem for the second part of the problem. -/
theorem possible_x_for_87 :
    {x : Nat | ∃ n ∈ systematicSample x, endsWithDigits n 87} =
    {87, 54, 21, 88, 55, 22, 89, 56, 23, 90} := by
  sorry

end NUMINAMATH_CALUDE_sample_for_x_24_possible_x_for_87_l1470_147065


namespace NUMINAMATH_CALUDE_cone_base_radius_l1470_147059

/-- Given a cone with slant height 6 cm and central angle of unfolded lateral surface 120°,
    prove that the radius of its base is 2 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) :
  slant_height = 6 →
  central_angle = 120 * π / 180 →
  2 * π * slant_height * (central_angle / (2 * π)) = 2 * π * 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1470_147059


namespace NUMINAMATH_CALUDE_inequality_proof_l1470_147019

open Real

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : exp a = 2 * a * exp (1/2))
  (eq_b : exp b = 3 * b * exp (1/3))
  (eq_c : exp c = 5 * c * exp (1/5)) :
  b * c * exp a < c * a * exp b ∧ c * a * exp b < a * b * exp c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1470_147019


namespace NUMINAMATH_CALUDE_hallie_tuesday_tips_l1470_147076

/-- Represents Hallie's earnings over three days --/
structure EarningsData :=
  (hourly_rate : ℝ)
  (monday_hours : ℝ)
  (monday_tips : ℝ)
  (tuesday_hours : ℝ)
  (wednesday_hours : ℝ)
  (wednesday_tips : ℝ)
  (total_earnings : ℝ)

/-- Calculates Hallie's tips on Tuesday given her earnings data --/
def tuesday_tips (data : EarningsData) : ℝ :=
  data.total_earnings - 
  (data.hourly_rate * (data.monday_hours + data.tuesday_hours + data.wednesday_hours)) -
  (data.monday_tips + data.wednesday_tips)

/-- Theorem stating that Hallie's tips on Tuesday were $12 --/
theorem hallie_tuesday_tips :
  let data : EarningsData := {
    hourly_rate := 10,
    monday_hours := 7,
    monday_tips := 18,
    tuesday_hours := 5,
    wednesday_hours := 7,
    wednesday_tips := 20,
    total_earnings := 240
  }
  tuesday_tips data = 12 := by sorry

end NUMINAMATH_CALUDE_hallie_tuesday_tips_l1470_147076


namespace NUMINAMATH_CALUDE_third_term_is_four_l1470_147011

/-- A geometric sequence with specific terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sixth_term : a 6 = 6
  ninth_term : a 9 = 9

/-- The third term of the geometric sequence is 4 -/
theorem third_term_is_four (seq : GeometricSequence) : seq.a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_four_l1470_147011


namespace NUMINAMATH_CALUDE_subcommittees_with_experts_count_l1470_147060

def committee_size : ℕ := 12
def expert_count : ℕ := 5
def subcommittee_size : ℕ := 5

theorem subcommittees_with_experts_count : 
  (Nat.choose committee_size subcommittee_size) - 
  (Nat.choose (committee_size - expert_count) subcommittee_size) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_with_experts_count_l1470_147060


namespace NUMINAMATH_CALUDE_distinct_arrangements_l1470_147016

/-- The number of members in the committee -/
def total_members : ℕ := 10

/-- The number of women (rocking chairs) -/
def num_women : ℕ := 7

/-- The number of men (stools) -/
def num_men : ℕ := 2

/-- The number of children (benches) -/
def num_children : ℕ := 1

/-- The number of distinct arrangements of seats -/
def num_arrangements : ℕ := total_members * (total_members - 1) * (total_members - 2) / 2

theorem distinct_arrangements :
  num_arrangements = 360 :=
sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l1470_147016


namespace NUMINAMATH_CALUDE_younger_brother_age_l1470_147064

theorem younger_brother_age 
  (older younger : ℕ) 
  (sum_condition : older + younger = 46)
  (age_relation : younger = older / 3 + 10) :
  younger = 19 := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_age_l1470_147064


namespace NUMINAMATH_CALUDE_builder_purchase_cost_l1470_147039

/-- Calculates the total cost of a builder's purchase with specific items, taxes, and discounts --/
theorem builder_purchase_cost : 
  let drill_bits_cost : ℚ := 5 * 6
  let hammers_cost : ℚ := 3 * 8
  let toolbox_cost : ℚ := 25
  let nails_cost : ℚ := (50 / 2) * 0.1
  let drill_bits_tax : ℚ := drill_bits_cost * 0.1
  let toolbox_tax : ℚ := toolbox_cost * 0.15
  let hammers_discount : ℚ := hammers_cost * 0.05
  let total_before_discount : ℚ := drill_bits_cost + drill_bits_tax + hammers_cost - hammers_discount + toolbox_cost + toolbox_tax + nails_cost
  let overall_discount : ℚ := if total_before_discount > 60 then total_before_discount * 0.05 else 0
  let final_total : ℚ := total_before_discount - overall_discount
  ∃ (rounded_total : ℚ), (rounded_total ≥ final_total) ∧ (rounded_total < final_total + 0.005) ∧ (rounded_total = 82.70) :=
by sorry


end NUMINAMATH_CALUDE_builder_purchase_cost_l1470_147039


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1470_147036

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y - 6 = 0 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1470_147036


namespace NUMINAMATH_CALUDE_math_homework_pages_l1470_147057

theorem math_homework_pages (reading : ℕ) (math : ℕ) : 
  math = reading + 3 →
  reading + math = 13 →
  math = 8 := by
sorry

end NUMINAMATH_CALUDE_math_homework_pages_l1470_147057


namespace NUMINAMATH_CALUDE_composite_transformation_matrix_l1470_147072

/-- The dilation matrix with scale factor 2 -/
def dilationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0],
    ![0, 2]]

/-- The rotation matrix for 90 degrees counterclockwise rotation -/
def rotationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

/-- The expected result matrix -/
def resultMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -2],
    ![2,  0]]

theorem composite_transformation_matrix :
  rotationMatrix * dilationMatrix = resultMatrix := by
  sorry

end NUMINAMATH_CALUDE_composite_transformation_matrix_l1470_147072


namespace NUMINAMATH_CALUDE_bus_journey_fraction_l1470_147040

theorem bus_journey_fraction (total_journey : ℝ) (rail_fraction : ℝ) (foot_distance : ℝ) :
  total_journey = 130 →
  rail_fraction = 3/5 →
  foot_distance = 6.5 →
  (total_journey - (rail_fraction * total_journey + foot_distance)) / total_journey = 45.5 / 130 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_fraction_l1470_147040


namespace NUMINAMATH_CALUDE_polynomial_factor_l1470_147077

theorem polynomial_factor (x : ℝ) : 
  ∃ (p : ℝ → ℝ), (x^4 - 4*x^2 + 4) = (x^2 - 2) * p x :=
sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1470_147077


namespace NUMINAMATH_CALUDE_douglas_vote_county_y_l1470_147058

theorem douglas_vote_county_y (total_vote_percent : ℝ) (county_x_percent : ℝ) (ratio_x_to_y : ℝ) :
  total_vote_percent = 60 ∧ 
  county_x_percent = 72 ∧ 
  ratio_x_to_y = 2 →
  let county_y_percent := (3 * total_vote_percent - 2 * county_x_percent) / 1
  county_y_percent = 36 := by
sorry

end NUMINAMATH_CALUDE_douglas_vote_county_y_l1470_147058


namespace NUMINAMATH_CALUDE_division_simplification_l1470_147017

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l1470_147017


namespace NUMINAMATH_CALUDE_simplify_fraction_l1470_147033

theorem simplify_fraction : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1470_147033


namespace NUMINAMATH_CALUDE_injury_point_is_20_l1470_147074

/-- Represents the runner's journey from Marathon to Athens -/
structure RunnerJourney where
  totalDistance : ℝ
  injuryPoint : ℝ
  initialSpeed : ℝ
  secondPartTime : ℝ
  timeDifference : ℝ

/-- The conditions of the runner's journey -/
def journeyConditions (j : RunnerJourney) : Prop :=
  j.totalDistance = 40 ∧
  j.secondPartTime = 22 ∧
  j.timeDifference = 11 ∧
  j.initialSpeed > 0 ∧
  j.injuryPoint > 0 ∧
  j.injuryPoint < j.totalDistance ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.secondPartTime ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.injuryPoint / j.initialSpeed + j.timeDifference

/-- Theorem stating that given the journey conditions, the injury point is at 20 miles -/
theorem injury_point_is_20 (j : RunnerJourney) (h : journeyConditions j) : j.injuryPoint = 20 := by
  sorry

#check injury_point_is_20

end NUMINAMATH_CALUDE_injury_point_is_20_l1470_147074


namespace NUMINAMATH_CALUDE_problem_solution_l1470_147080

theorem problem_solution (x y : ℝ) (h1 : x^2 + 4 = y - 2) (h2 : x = 6) : y = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1470_147080


namespace NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l1470_147078

-- Define the properties of triangles
def IsScalene (triangle : Type) : Prop := sorry
def IsEquilateral (triangle : Type) : Prop := sorry
def IsRight (triangle : Type) : Prop := sorry

-- Theorem stating that scalene equilateral triangles cannot exist
theorem no_scalene_equilateral_triangle (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) := by sorry

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle (triangle : Type) :
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

-- Main theorem combining both impossible triangle types
theorem impossible_triangles (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) ∧
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

end NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l1470_147078


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1470_147096

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1470_147096


namespace NUMINAMATH_CALUDE_distinctNumbers_eq_2001_l1470_147063

/-- The number of distinct numbers in the list [⌊1²/500⌋, ⌊2²/500⌋, ⌊3²/500⌋, ..., ⌊1000²/500⌋] -/
def distinctNumbers : ℕ :=
  let list := List.range 1000
  let floorList := list.map (fun n => Int.floor ((n + 1)^2 / 500 : ℚ))
  floorList.eraseDups.length

/-- The theorem stating that the number of distinct numbers in the list is 2001 -/
theorem distinctNumbers_eq_2001 : distinctNumbers = 2001 := by
  sorry

end NUMINAMATH_CALUDE_distinctNumbers_eq_2001_l1470_147063


namespace NUMINAMATH_CALUDE_apple_pie_division_l1470_147025

/-- The number of apple pies Sedrach has -/
def total_pies : ℕ := 13

/-- The number of bite-size samples each part of an apple pie can be split into -/
def samples_per_part : ℕ := 5

/-- The total number of people who can taste the pies -/
def total_tasters : ℕ := 130

/-- The number of parts each apple pie is divided into -/
def parts_per_pie : ℕ := 2

theorem apple_pie_division :
  total_pies * parts_per_pie * samples_per_part = total_tasters := by sorry

end NUMINAMATH_CALUDE_apple_pie_division_l1470_147025


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1470_147061

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 5 = 0 → 
  x₂^2 - 4*x₂ - 5 = 0 → 
  (x₁ - 1) * (x₂ - 1) = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1470_147061


namespace NUMINAMATH_CALUDE_rectangle_area_l1470_147071

/-- The area of a rectangle with width w, length 3w, and diagonal d is (3/10)d^2 -/
theorem rectangle_area (w d : ℝ) (h1 : w > 0) (h2 : d > 0) (h3 : w^2 + (3*w)^2 = d^2) :
  w * (3*w) = (3/10) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1470_147071


namespace NUMINAMATH_CALUDE_nonright_angle_is_45_l1470_147012

/-- A right isosceles triangle with specific properties -/
structure RightIsoscelesTriangle where
  -- The length of the hypotenuse
  h : ℝ
  -- The height from the right angle to the hypotenuse
  a : ℝ
  -- The product of the hypotenuse and the square of the height is 90
  hyp_height_product : h * a^2 = 90
  -- The triangle is right-angled (implied by being right isosceles)
  right_angled : True
  -- The triangle is isosceles
  isosceles : True

/-- The measure of one of the non-right angles in the triangle -/
def nonRightAngle (t : RightIsoscelesTriangle) : ℝ := 45

/-- Theorem: In a right isosceles triangle where the product of the hypotenuse
    and the square of the height is 90, one of the non-right angles is 45° -/
theorem nonright_angle_is_45 (t : RightIsoscelesTriangle) :
  nonRightAngle t = 45 := by sorry

end NUMINAMATH_CALUDE_nonright_angle_is_45_l1470_147012
