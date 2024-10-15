import Mathlib

namespace NUMINAMATH_CALUDE_correct_selection_methods_l291_29144

/-- The number of ways to select students for health checks -/
def select_students (total_students : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods -/
theorem correct_selection_methods :
  select_students 23 10 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l291_29144


namespace NUMINAMATH_CALUDE_not_enough_money_l291_29104

/-- The cost of a single storybook in yuan -/
def storybook_cost : ℕ := 18

/-- The number of storybooks to be purchased -/
def num_books : ℕ := 12

/-- The available money in yuan -/
def available_money : ℕ := 200

/-- Theorem stating that the available money is not enough to buy the desired number of storybooks -/
theorem not_enough_money : storybook_cost * num_books > available_money := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l291_29104


namespace NUMINAMATH_CALUDE_cement_total_l291_29192

theorem cement_total (bought : ℕ) (brought : ℕ) (total : ℕ) : 
  bought = 215 → brought = 137 → total = bought + brought → total = 352 := by
  sorry

end NUMINAMATH_CALUDE_cement_total_l291_29192


namespace NUMINAMATH_CALUDE_equation_solution_l291_29123

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (30 + x) = 14.7 ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l291_29123


namespace NUMINAMATH_CALUDE_lunch_cakes_count_cakes_sum_equals_total_l291_29106

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

/-- Theorem proving that the sum of cakes served equals the total -/
theorem cakes_sum_equals_total : lunch_cakes + dinner_cakes + yesterday_cakes = total_cakes := by
  sorry

end NUMINAMATH_CALUDE_lunch_cakes_count_cakes_sum_equals_total_l291_29106


namespace NUMINAMATH_CALUDE_count_congruent_is_77_l291_29162

/-- The number of positive integers less than 1000 that are congruent to 7 (mod 13) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n > 0 ∧ n < 1000 ∧ n % 13 = 7) (Finset.range 1000)).card

/-- Theorem stating that the count of such integers is 77 -/
theorem count_congruent_is_77 : count_congruent = 77 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_is_77_l291_29162


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l291_29150

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  (∀ x, tangent_line x = -3 * x + 2) ∧ y₀ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l291_29150


namespace NUMINAMATH_CALUDE_statue_original_cost_l291_29113

/-- Proves that if a statue is sold for $540 with a 35% profit, then its original cost was $400. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 540 →
  profit_percentage = 0.35 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 400 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l291_29113


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l291_29181

theorem circle_equation_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0 ∧ 
    ∃ h r : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - r)^2 = (x^2 + y^2 + 2*k*x + 4*y + 3*k + 8)) 
  ↔ (k > 4 ∨ k < -1) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l291_29181


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l291_29121

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : (i - 1) * z = i) : 
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l291_29121


namespace NUMINAMATH_CALUDE_bouquet_carnations_fraction_l291_29120

theorem bouquet_carnations_fraction (total_flowers : ℕ) 
  (blue_flowers red_flowers blue_roses red_roses blue_carnations red_carnations : ℕ) :
  (blue_flowers = red_flowers) →  -- Half of the flowers are blue
  (blue_flowers + red_flowers = total_flowers) →
  (blue_roses = 2 * blue_flowers / 5) →  -- Two-fifths of blue flowers are roses
  (red_carnations = 2 * red_flowers / 3) →  -- Two-thirds of red flowers are carnations
  (blue_carnations = blue_flowers - blue_roses) →
  (red_roses = red_flowers - red_carnations) →
  ((blue_carnations + red_carnations : ℚ) / total_flowers = 19 / 30) := by
sorry

end NUMINAMATH_CALUDE_bouquet_carnations_fraction_l291_29120


namespace NUMINAMATH_CALUDE_inequality_proof_l291_29195

theorem inequality_proof (x y z : ℝ) 
  (h1 : x + 2*y + 4*z ≥ 3) 
  (h2 : y - 3*x + 2*z ≥ 5) : 
  y - x + 2*z ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l291_29195


namespace NUMINAMATH_CALUDE_find_divisor_l291_29168

theorem find_divisor (divisor : ℕ) : divisor = 2 := by
  have h1 : 2 = (433126 : ℕ) - 433124 := by sorry
  have h2 : (433126 : ℕ) % divisor = 0 := by sorry
  have h3 : ∀ n : ℕ, n < 2 → (433124 + n) % divisor ≠ 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_find_divisor_l291_29168


namespace NUMINAMATH_CALUDE_never_exceeds_100_l291_29145

def repeated_square (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | m + 1 => (repeated_square m) ^ 2

theorem never_exceeds_100 (n : ℕ) : repeated_square n ≤ 100 := by
  sorry

#check never_exceeds_100

end NUMINAMATH_CALUDE_never_exceeds_100_l291_29145


namespace NUMINAMATH_CALUDE_two_divisors_of_ten_billion_sum_to_157_l291_29156

theorem two_divisors_of_ten_billion_sum_to_157 :
  ∃ (a b : ℕ), 
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    (10^10 % a = 0) ∧
    (10^10 % b = 0) ∧
    a + b = 157 ∧
    a = 32 ∧
    b = 125 := by
  sorry

end NUMINAMATH_CALUDE_two_divisors_of_ten_billion_sum_to_157_l291_29156


namespace NUMINAMATH_CALUDE_train_crossing_time_l291_29191

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 40 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * (5/18))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l291_29191


namespace NUMINAMATH_CALUDE_line_through_points_l291_29119

/-- Given a line y = ax + b passing through points (3, -2) and (7, 14), prove that a + b = -10 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (-2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  a + b = -10 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l291_29119


namespace NUMINAMATH_CALUDE_largest_prime_to_check_for_primality_l291_29182

theorem largest_prime_to_check_for_primality (n : ℕ) :
  2500 ≤ n → n ≤ 2600 →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 47) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_to_check_for_primality_l291_29182


namespace NUMINAMATH_CALUDE_tenth_fibonacci_is_55_l291_29198

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_55 : fibonacci 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fibonacci_is_55_l291_29198


namespace NUMINAMATH_CALUDE_smallest_n_with_unique_k_l291_29155

theorem smallest_n_with_unique_k : ∃ (k : ℤ),
  (7 : ℚ) / 16 < (63 : ℚ) / (63 + k) ∧ (63 : ℚ) / (63 + k) < 9 / 20 ∧
  (∀ (k' : ℤ), k' ≠ k →
    ((7 : ℚ) / 16 ≥ (63 : ℚ) / (63 + k') ∨ (63 : ℚ) / (63 + k') ≥ 9 / 20)) ∧
  (∀ (n : ℕ), 0 < n → n < 63 →
    ¬(∃! (k : ℤ), (7 : ℚ) / 16 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 9 / 20)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_unique_k_l291_29155


namespace NUMINAMATH_CALUDE_green_bows_count_l291_29130

theorem green_bows_count (total : ℕ) : 
  (3 : ℚ) / 20 * total + 3 / 10 * total + 1 / 5 * total + 1 / 20 * total + 24 = total →
  1 / 5 * total = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_bows_count_l291_29130


namespace NUMINAMATH_CALUDE_quadratic_vertex_property_l291_29110

/-- Given a quadratic function y = x^2 - 2x + n with vertex (m, 1), prove that m - n = -1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, x^2 - 2*x + n = (x - m)^2 + 1) → m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_property_l291_29110


namespace NUMINAMATH_CALUDE_quadrilateral_area_at_least_30_l291_29108

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 5
  (ex - fx)^2 + (ey - fy)^2 = 25 ∧
  -- FG = 12
  (fx - gx)^2 + (fy - gy)^2 = 144 ∧
  -- GH = 5
  (gx - hx)^2 + (gy - hy)^2 = 25 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 169 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Define the area function
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_at_least_30 (q : Quadrilateral) :
  is_valid_quadrilateral q → area q ≥ 30 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_at_least_30_l291_29108


namespace NUMINAMATH_CALUDE_symmetrical_point_not_in_third_quadrant_l291_29111

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the symmetrical point with respect to the y-axis -/
def symmetricalPointY (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem to prove -/
theorem symmetrical_point_not_in_third_quadrant :
  let p := Point.mk (-3) 4
  let symmetricalP := symmetricalPointY p
  ¬(isInThirdQuadrant symmetricalP) := by
  sorry


end NUMINAMATH_CALUDE_symmetrical_point_not_in_third_quadrant_l291_29111


namespace NUMINAMATH_CALUDE_max_triangles_correct_l291_29151

/-- The number of points on the hypotenuse of the right triangle -/
def num_points : ℕ := 8

/-- The maximum number of triangles that can be formed -/
def max_triangles : ℕ := 28

/-- Theorem stating that the maximum number of triangles is correct -/
theorem max_triangles_correct :
  (num_points.choose 2) = max_triangles := by sorry

end NUMINAMATH_CALUDE_max_triangles_correct_l291_29151


namespace NUMINAMATH_CALUDE_two_thousand_seventeenth_number_l291_29154

def is_divisible_by_2_or_3 (n : ℕ) : Prop := 2 ∣ n ∨ 3 ∣ n

def sequence_2_or_3 : ℕ → ℕ := sorry

theorem two_thousand_seventeenth_number :
  sequence_2_or_3 2017 = 3026 := by sorry

end NUMINAMATH_CALUDE_two_thousand_seventeenth_number_l291_29154


namespace NUMINAMATH_CALUDE_line_touches_x_axis_twice_l291_29152

-- Define the function representing the line
def f (x : ℝ) : ℝ := x^2 - x^3

-- Theorem statement
theorem line_touches_x_axis_twice :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_line_touches_x_axis_twice_l291_29152


namespace NUMINAMATH_CALUDE_fraction_value_l291_29112

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l291_29112


namespace NUMINAMATH_CALUDE_points_earned_is_75_l291_29189

/-- Represents the point system and enemy counts in the video game level -/
structure GameLevel where
  goblin_points : ℕ
  orc_points : ℕ
  dragon_points : ℕ
  goblins_defeated : ℕ
  orcs_defeated : ℕ
  dragons_defeated : ℕ

/-- Calculates the total points earned in a game level -/
def total_points (level : GameLevel) : ℕ :=
  level.goblin_points * level.goblins_defeated +
  level.orc_points * level.orcs_defeated +
  level.dragon_points * level.dragons_defeated

/-- Theorem stating that the total points earned in the given scenario is 75 -/
theorem points_earned_is_75 (level : GameLevel) 
  (h1 : level.goblin_points = 3)
  (h2 : level.orc_points = 5)
  (h3 : level.dragon_points = 10)
  (h4 : level.goblins_defeated = 10)
  (h5 : level.orcs_defeated = 7)
  (h6 : level.dragons_defeated = 1) :
  total_points level = 75 := by
  sorry


end NUMINAMATH_CALUDE_points_earned_is_75_l291_29189


namespace NUMINAMATH_CALUDE_total_material_ordered_l291_29143

def concrete : ℝ := 0.17
def bricks : ℝ := 0.17
def stone : ℝ := 0.5

theorem total_material_ordered : concrete + bricks + stone = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_total_material_ordered_l291_29143


namespace NUMINAMATH_CALUDE_smallest_a_value_l291_29115

theorem smallest_a_value (a : ℝ) (h_a_pos : a > 0) : 
  (∀ x > 0, x + a / x ≥ 4) ↔ a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l291_29115


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l291_29163

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  -- Length of side EF
  ef : ℝ
  -- Length of side GH
  gh : ℝ
  -- Height of the trapezoid
  height : ℝ
  -- Area of the trapezoid
  area : ℝ
  -- EF is half the length of GH
  ef_half_gh : ef = gh / 2
  -- Height is 6 units
  height_is_6 : height = 6
  -- Area is 90 square units
  area_is_90 : area = 90

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the perimeter of the trapezoid is 30 + 2√61 -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 30 + 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l291_29163


namespace NUMINAMATH_CALUDE_car_speed_problem_l291_29116

/-- Proves that if a car traveling at speed v km/h takes 15 seconds longer to travel 1 km
    than it would at 48 km/h, then v = 40 km/h. -/
theorem car_speed_problem (v : ℝ) :
  (v > 0) →  -- Ensure speed is positive
  (3600 / v = 3600 / 48 + 15) →  -- Time difference equation
  v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l291_29116


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l291_29188

theorem complex_exponential_to_rectangular : 
  Complex.exp (Complex.I * (13 * Real.pi / 6)) * (Real.sqrt 3 : ℂ) = (3 / 2 : ℂ) + Complex.I * ((Real.sqrt 3) / 2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l291_29188


namespace NUMINAMATH_CALUDE_u_converges_immediately_l291_29137

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_converges_immediately :
  ∀ k : ℕ, |u k - 1/2| ≤ 1/2^20 := by
  sorry

end NUMINAMATH_CALUDE_u_converges_immediately_l291_29137


namespace NUMINAMATH_CALUDE_tangent_point_value_l291_29165

/-- The value of 'a' for which the line y = x + 1 is tangent to the curve y = ln(x + a) --/
def tangent_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    -- The y-coordinate of the line and curve are equal at the point of tangency
    x + 1 = Real.log (x + a) ∧ 
    -- The slope of the line (which is 1) equals the derivative of ln(x + a) at the point of tangency
    1 = 1 / (x + a)

/-- Theorem stating that 'a' must equal 2 for the tangency condition to be satisfied --/
theorem tangent_point_value : 
  ∃ a : ℝ, tangent_point a ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_value_l291_29165


namespace NUMINAMATH_CALUDE_square_rectangle_area_difference_l291_29140

theorem square_rectangle_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_difference_l291_29140


namespace NUMINAMATH_CALUDE_min_neighbors_2005_points_l291_29138

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for its endpoints to be considered neighbors -/
def max_neighbor_angle : ℝ := 10

/-- A function that calculates the minimum number of neighbor pairs given the number of points and maximum neighbor angle -/
noncomputable def min_neighbor_pairs (n : ℕ) (max_angle : ℝ) : ℕ := sorry

/-- Theorem stating that the minimum number of neighbor pairs for 2005 points with a 10° maximum angle is 56430 -/
theorem min_neighbors_2005_points :
  min_neighbor_pairs num_points max_neighbor_angle = 56430 := by sorry

end NUMINAMATH_CALUDE_min_neighbors_2005_points_l291_29138


namespace NUMINAMATH_CALUDE_rotation_of_negative_six_minus_three_i_l291_29131

def rotate90Clockwise (z : ℂ) : ℂ := -Complex.I * z

theorem rotation_of_negative_six_minus_three_i :
  rotate90Clockwise (-6 - 3*Complex.I) = -3 + 6*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotation_of_negative_six_minus_three_i_l291_29131


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l291_29100

theorem polynomial_roots_problem (r s t : ℝ) 
  (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + r*x + s = 0 ↔ x = s ∨ x = t)
  (h2 : (5 : ℝ)^2 + t*5 + r = 0) : 
  s = 29 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_problem_l291_29100


namespace NUMINAMATH_CALUDE_trig_roots_theorem_l291_29139

theorem trig_roots_theorem (θ : ℝ) (m : ℝ) 
  (h1 : (Real.sin θ)^2 - (Real.sqrt 3 - 1) * (Real.sin θ) + m = 0)
  (h2 : (Real.cos θ)^2 - (Real.sqrt 3 - 1) * (Real.cos θ) + m = 0) :
  (m = (3 - 2 * Real.sqrt 3) / 2) ∧
  ((Real.cos θ - Real.sin θ * Real.tan θ) / (1 - Real.tan θ) = Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_trig_roots_theorem_l291_29139


namespace NUMINAMATH_CALUDE_solve_inequality_minimum_a_l291_29127

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part 1
theorem solve_inequality (x : ℝ) : 
  f (-2) x > 5 ↔ x < -4/3 ∨ x > 2 := by sorry

-- Part 2
theorem minimum_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), f a x ≤ a * |x + 3|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), f a x ≤ a * |x + 3|) → a ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_minimum_a_l291_29127


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l291_29109

/-- Given a polynomial g(x) = ax^7 + bx^6 + cx - 3 where g(-5) = -3, prove that g(5) = 31250b - 3 -/
theorem polynomial_value_at_five (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^7 + b * x^6 + c * x - 3
  g (-5) = -3 →
  g 5 = 31250 * b - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l291_29109


namespace NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l291_29126

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 1000
  | Coin.Nickel => 500
  | Coin.Penny => 200

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of selecting a nickel -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l291_29126


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l291_29167

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 100)
  (h2 : final_price = 81)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l291_29167


namespace NUMINAMATH_CALUDE_males_with_college_degree_only_count_l291_29125

/-- Represents the employee demographics of a company -/
structure CompanyDemographics where
  total_employees : Nat
  total_females : Nat
  employees_with_advanced_degrees : Nat
  females_with_advanced_degrees : Nat

/-- Calculates the number of males with a college degree only -/
def males_with_college_degree_only (demo : CompanyDemographics) : Nat :=
  let total_males := demo.total_employees - demo.total_females
  let males_with_advanced_degrees := demo.employees_with_advanced_degrees - demo.females_with_advanced_degrees
  total_males - males_with_advanced_degrees

/-- Theorem stating the number of males with a college degree only -/
theorem males_with_college_degree_only_count 
  (demo : CompanyDemographics) 
  (h1 : demo.total_employees = 180)
  (h2 : demo.total_females = 110)
  (h3 : demo.employees_with_advanced_degrees = 90)
  (h4 : demo.females_with_advanced_degrees = 55) :
  males_with_college_degree_only demo = 35 := by
  sorry

#eval males_with_college_degree_only { 
  total_employees := 180, 
  total_females := 110, 
  employees_with_advanced_degrees := 90, 
  females_with_advanced_degrees := 55 
}

end NUMINAMATH_CALUDE_males_with_college_degree_only_count_l291_29125


namespace NUMINAMATH_CALUDE_square_graph_triangles_l291_29183

/-- Represents a planar graph formed by connecting points in a square --/
structure SquareGraph where
  /-- The number of internal points marked in the square --/
  internalPoints : ℕ
  /-- The total number of vertices (internal points + 4 square vertices) --/
  totalVertices : ℕ
  /-- The number of edges in the graph --/
  edges : ℕ
  /-- The number of faces (regions) formed, including the external face --/
  faces : ℕ
  /-- Condition: The total vertices is the sum of internal points and square vertices --/
  vertexCount : totalVertices = internalPoints + 4
  /-- Condition: Euler's formula for planar graphs --/
  eulerFormula : totalVertices - edges + faces = 2
  /-- Condition: Relationship between edges and faces --/
  edgeFaceRelation : 2 * edges = 3 * (faces - 1) + 4

/-- Theorem: In a square with 20 internal points connected as described, 42 triangles are formed --/
theorem square_graph_triangles (g : SquareGraph) (h : g.internalPoints = 20) : g.faces - 1 = 42 := by
  sorry


end NUMINAMATH_CALUDE_square_graph_triangles_l291_29183


namespace NUMINAMATH_CALUDE_distance_between_points_l291_29166

/-- The distance between two points (3, 0) and (7, 7) on a Cartesian coordinate plane is √65. -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((7 - 3)^2 + (7 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l291_29166


namespace NUMINAMATH_CALUDE_solve_proportion_l291_29158

theorem solve_proportion (x : ℚ) (h : x / 6 = 15 / 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l291_29158


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_36_seconds_l291_29148

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time (jogger_speed train_speed : Real) 
  (train_length initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the time for the train to pass the jogger is 36 seconds -/
theorem train_passes_jogger_in_36_seconds :
  train_passing_jogger_time 9 45 120 240 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_36_seconds_l291_29148


namespace NUMINAMATH_CALUDE_oldest_sibling_age_difference_l291_29194

theorem oldest_sibling_age_difference (siblings : Fin 4 → ℝ) 
  (avg_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 30)
  (youngest_age : siblings 3 = 25.75)
  : ∃ i : Fin 4, siblings i - siblings 3 ≥ 17 :=
by
  sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_difference_l291_29194


namespace NUMINAMATH_CALUDE_digital_earth_correct_application_l291_29147

/-- Represents the capabilities of the digital Earth -/
structure DigitalEarthCapabilities where
  resourceOptimization : Bool
  informationAccess : Bool

/-- Represents possible applications of the digital Earth -/
inductive DigitalEarthApplication
  | crimeControl
  | projectDecisionSupport
  | precipitationControl
  | disasterControl

/-- Determines if an application is correct given the capabilities of the digital Earth -/
def isCorrectApplication (capabilities : DigitalEarthCapabilities) (application : DigitalEarthApplication) : Prop :=
  capabilities.resourceOptimization ∧ 
  capabilities.informationAccess ∧ 
  application = DigitalEarthApplication.projectDecisionSupport

theorem digital_earth_correct_application (capabilities : DigitalEarthCapabilities) 
  (h1 : capabilities.resourceOptimization = true) 
  (h2 : capabilities.informationAccess = true) :
  isCorrectApplication capabilities DigitalEarthApplication.projectDecisionSupport :=
sorry

end NUMINAMATH_CALUDE_digital_earth_correct_application_l291_29147


namespace NUMINAMATH_CALUDE_sum_of_triangles_l291_29157

/-- The triangle operation defined as a × b - c -/
def triangle (a b c : ℝ) : ℝ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals -2 -/
theorem sum_of_triangles : triangle 2 3 5 + triangle 1 4 7 = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangles_l291_29157


namespace NUMINAMATH_CALUDE_problem_solution_l291_29114

theorem problem_solution (x y z : ℝ) 
  (hx : x ≠ 0)
  (hz : z ≠ 0)
  (eq1 : x/2 = y^2 + z)
  (eq2 : x/4 = 4*y + 2*z) :
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l291_29114


namespace NUMINAMATH_CALUDE_gcd_360_210_l291_29164

theorem gcd_360_210 : Nat.gcd 360 210 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_210_l291_29164


namespace NUMINAMATH_CALUDE_undecided_voters_percentage_l291_29101

theorem undecided_voters_percentage
  (total_polled : ℕ)
  (biff_percentage : ℚ)
  (marty_voters : ℕ)
  (h1 : total_polled = 200)
  (h2 : biff_percentage = 45 / 100)
  (h3 : marty_voters = 94) :
  (total_polled - (marty_voters + (biff_percentage * total_polled).floor)) / total_polled = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_undecided_voters_percentage_l291_29101


namespace NUMINAMATH_CALUDE_circle_reduction_l291_29175

/-- Represents a letter in the circle -/
inductive Letter
| A
| B

/-- Represents the circle of letters -/
def Circle := List Letter

/-- Represents a transformation rule -/
inductive Transform
| ABA_to_B
| B_to_ABA
| VAV_to_A
| A_to_VAV

/-- Applies a single transformation to the circle -/
def applyTransform (c : Circle) (t : Transform) : Circle :=
  sorry

/-- Checks if the circle contains exactly one letter -/
def isSingleLetter (c : Circle) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem circle_reduction (initial : Circle) :
  initial.length = 41 →
  ∃ (final : Circle), (∃ (transforms : List Transform),
    (List.foldl applyTransform initial transforms = final) ∧
    isSingleLetter final) :=
  sorry

end NUMINAMATH_CALUDE_circle_reduction_l291_29175


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l291_29118

theorem baker_cakes_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (extra_sold : ℕ) :
  initial_cakes = 8 →
  sold_cakes = 145 →
  extra_sold = 6 →
  ∃ (new_cakes : ℕ), 
    new_cakes + initial_cakes = sold_cakes + extra_sold ∧
    new_cakes = 131 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l291_29118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l291_29136

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h1 : a 4 = -15
  h2 : d = 3
  h3 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 27) ∧
  ∃ min : ℤ, min = -108 ∧ ∀ n : ℕ, SumOfTerms seq n ≥ min :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l291_29136


namespace NUMINAMATH_CALUDE_largest_angle_in_consecutive_angle_hexagon_l291_29122

/-- The largest angle in a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A convex hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  angles : Fin 6 → ℤ
  is_consecutive : ∀ i : Fin 5, angles i.succ = angles i + 1
  is_convex : ∀ i : Fin 6, 0 < angles i ∧ angles i < 180

theorem largest_angle_in_consecutive_angle_hexagon (h : ConsecutiveAngleHexagon) :
  (h.angles 5 : ℝ) = largest_hexagon_angle :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_consecutive_angle_hexagon_l291_29122


namespace NUMINAMATH_CALUDE_beach_trip_time_l291_29185

/-- Calculates the total trip time given the one-way drive time and the ratio of destination time to total drive time -/
def totalTripTime (oneWayDriveTime : ℝ) (destinationTimeRatio : ℝ) : ℝ :=
  let totalDriveTime := 2 * oneWayDriveTime
  let destinationTime := destinationTimeRatio * totalDriveTime
  totalDriveTime + destinationTime

/-- Proves that for a trip with 2 hours one-way drive time and 2.5 ratio of destination time to total drive time, the total trip time is 14 hours -/
theorem beach_trip_time : totalTripTime 2 2.5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_l291_29185


namespace NUMINAMATH_CALUDE_range_of_r_l291_29107

theorem range_of_r (a b c r : ℝ) 
  (h1 : b + c ≤ 4 * a)
  (h2 : c - b ≥ 0)
  (h3 : b ≥ a)
  (h4 : a > 0)
  (h5 : r > 0)
  (h6 : (a + b)^2 + (a + c)^2 ≠ (a * r)^2) :
  r ∈ Set.Ioo 0 (2 * Real.sqrt 2) ∪ Set.Ioi (3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_r_l291_29107


namespace NUMINAMATH_CALUDE_diamond_symmetry_points_l291_29149

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) satisfying x ⋄ y = y ⋄ x -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- Two lines in ℝ² -/
def two_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_symmetry_points :
  S = two_lines ∪ ({0} : Set ℝ).prod Set.univ ∪ Set.univ.prod ({0} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_diamond_symmetry_points_l291_29149


namespace NUMINAMATH_CALUDE_symmetric_difference_of_M_and_N_l291_29133

-- Define the symmetric difference operation
def symmetricDifference (A B : Set ℝ) : Set ℝ :=
  (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {y | ∃ x, 0 < x ∧ x < 2 ∧ y = -x^2 + 2*x}
def N : Set ℝ := {y | ∃ x, x > 0 ∧ y = 2^(x-1)}

-- State the theorem
theorem symmetric_difference_of_M_and_N :
  symmetricDifference M N = {y | (0 < y ∧ y ≤ 1/2) ∨ (1 < y)} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_M_and_N_l291_29133


namespace NUMINAMATH_CALUDE_edward_total_spent_l291_29134

/-- The total amount Edward spent on a board game and action figures -/
def total_spent (board_game_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  board_game_cost + num_figures * figure_cost

/-- Theorem stating that Edward spent $30 in total -/
theorem edward_total_spent :
  total_spent 2 4 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_edward_total_spent_l291_29134


namespace NUMINAMATH_CALUDE_work_completion_theorem_l291_29190

/-- Calculates the number of men needed to complete a job in a given number of days,
    given the initial number of men and days required. -/
def men_needed (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  (initial_men * initial_days) / new_days

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 25 → initial_days = 96 → new_days = 60 →
  men_needed initial_men initial_days new_days = 40 := by
  sorry

#eval men_needed 25 96 60

end NUMINAMATH_CALUDE_work_completion_theorem_l291_29190


namespace NUMINAMATH_CALUDE_certain_number_proof_l291_29142

theorem certain_number_proof : ∃ x : ℝ, (3889 + 12.952 - x = 3854.002) ∧ (x = 47.95) := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l291_29142


namespace NUMINAMATH_CALUDE_expression_value_l291_29180

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -a - b^2 + a*b = -17 := by sorry

end NUMINAMATH_CALUDE_expression_value_l291_29180


namespace NUMINAMATH_CALUDE_fred_balloons_l291_29186

theorem fred_balloons (total sam mary : ℕ) (h1 : total = 18) (h2 : sam = 6) (h3 : mary = 7) :
  total - (sam + mary) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l291_29186


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l291_29199

def Point := ℝ × ℝ

def symmetric_origin (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

def symmetric_y_axis (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_point_coordinates :
  ∀ (P P1 P2 : Point),
    symmetric_origin P1 P →
    P1 = (-2, 3) →
    symmetric_y_axis P2 P →
    P2 = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l291_29199


namespace NUMINAMATH_CALUDE_locus_and_slope_theorem_l291_29187

noncomputable def A : ℝ × ℝ := (0, 4/3)
noncomputable def B : ℝ × ℝ := (-1, 0)
noncomputable def C : ℝ × ℝ := (1, 0)

def distance_to_line (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

def line_BC : ℝ × ℝ → Prop := sorry
def line_AB : ℝ × ℝ → Prop := sorry
def line_AC : ℝ × ℝ → Prop := sorry

def locus_equation_1 (P : ℝ × ℝ) : Prop :=
  (P.1^2 + P.2^2 + 3/2 * P.2 - 1 = 0)

def locus_equation_2 (P : ℝ × ℝ) : Prop :=
  (8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0)

def incenter (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

def line_intersects_locus_at_3_points (l : ℝ → ℝ) : Prop := sorry

def slope_set : Set ℝ := {0, 1/2, -1/2, 2 * Real.sqrt 34 / 17, -2 * Real.sqrt 34 / 17, Real.sqrt 2 / 2, -Real.sqrt 2 / 2}

theorem locus_and_slope_theorem :
  ∀ P : ℝ × ℝ,
  (distance_to_line P line_BC)^2 = (distance_to_line P line_AB) * (distance_to_line P line_AC) →
  (locus_equation_1 P ∨ locus_equation_2 P) ∧
  ∀ l : ℝ → ℝ,
  (∃ x : ℝ, l x = (incenter (A, B, C)).2) →
  line_intersects_locus_at_3_points l →
  ∃ k : ℝ, k ∈ slope_set ∧ ∀ x : ℝ, l x = k * x + (incenter (A, B, C)).2 :=
sorry

end NUMINAMATH_CALUDE_locus_and_slope_theorem_l291_29187


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l291_29102

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ :=
  -l.y_intercept / l.slope

/-- The original line 4x + 5y = 10 -/
def original_line : Line :=
  { slope := -4/5, y_intercept := 2 }

/-- The perpendicular line we're interested in -/
def perpendicular_line : Line :=
  { slope := 5/4, y_intercept := -3 }

theorem x_intercept_of_perpendicular_line :
  perpendicular original_line perpendicular_line ∧
  perpendicular_line.y_intercept = -3 →
  x_intercept perpendicular_line = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l291_29102


namespace NUMINAMATH_CALUDE_marcus_savings_l291_29171

-- Define the given values
def max_budget : ℚ := 250
def shoe_price : ℚ := 120
def shoe_discount : ℚ := 0.3
def shoe_cashback : ℚ := 10
def shoe_tax : ℚ := 0.08
def sock_price : ℚ := 25
def sock_tax : ℚ := 0.06
def shirt_price : ℚ := 55
def shirt_discount : ℚ := 0.1
def shirt_tax : ℚ := 0.07

-- Define the calculation functions
def calculate_shoe_cost : ℚ := 
  (shoe_price * (1 - shoe_discount) - shoe_cashback) * (1 + shoe_tax)

def calculate_sock_cost : ℚ := 
  sock_price * (1 + sock_tax) / 2

def calculate_shirt_cost : ℚ := 
  shirt_price * (1 - shirt_discount) * (1 + shirt_tax)

def total_cost : ℚ := 
  calculate_shoe_cost + calculate_sock_cost + calculate_shirt_cost

-- Theorem statement
theorem marcus_savings : 
  max_budget - total_cost = 103.86 := by sorry

end NUMINAMATH_CALUDE_marcus_savings_l291_29171


namespace NUMINAMATH_CALUDE_thomas_drawings_l291_29153

theorem thomas_drawings (colored_pencil : ℕ) (blending_markers : ℕ) (charcoal : ℕ)
  (h1 : colored_pencil = 14)
  (h2 : blending_markers = 7)
  (h3 : charcoal = 4) :
  colored_pencil + blending_markers + charcoal = 25 :=
by sorry

end NUMINAMATH_CALUDE_thomas_drawings_l291_29153


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l291_29184

theorem complex_fraction_simplification : (Complex.I + 2) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l291_29184


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l291_29160

/-- Given a boat traveling downstream with a current of 5 km/hr,
    if it covers a distance of 7.5 km in 18 minutes,
    then its speed in still water is 20 km/hr. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (boat_speed : ℝ) :
  current_speed = 5 →
  distance_downstream = 7.5 →
  time_minutes = 18 →
  distance_downstream = (boat_speed + current_speed) * (time_minutes / 60) →
  boat_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l291_29160


namespace NUMINAMATH_CALUDE_initial_meals_proof_l291_29117

/-- The number of meals initially available for adults -/
def initial_meals : ℕ := 70

/-- The number of children that can be fed with one adult meal -/
def children_per_adult_meal : ℚ := 90 / initial_meals

theorem initial_meals_proof :
  (initial_meals - 21) * children_per_adult_meal = 63 :=
by sorry

end NUMINAMATH_CALUDE_initial_meals_proof_l291_29117


namespace NUMINAMATH_CALUDE_school_club_members_l291_29174

theorem school_club_members :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 6 = 4 ∧ n % 5 = 2 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_school_club_members_l291_29174


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l291_29179

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(31 ∣ (42739 - m))) ∧ (31 ∣ (42739 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l291_29179


namespace NUMINAMATH_CALUDE_processing_block_performs_assignment_calculation_l291_29193

-- Define the types of program blocks
inductive ProgramBlock
  | Terminal
  | InputOutput
  | Processing
  | Decision

-- Define the functions that a block can perform
inductive BlockFunction
  | StartStop
  | InformationIO
  | AssignmentCalculation
  | ConditionCheck

-- Define a function that maps a block to its primary function
def blockPrimaryFunction : ProgramBlock → BlockFunction
  | ProgramBlock.Terminal => BlockFunction.StartStop
  | ProgramBlock.InputOutput => BlockFunction.InformationIO
  | ProgramBlock.Processing => BlockFunction.AssignmentCalculation
  | ProgramBlock.Decision => BlockFunction.ConditionCheck

-- Theorem statement
theorem processing_block_performs_assignment_calculation :
  ∀ (block : ProgramBlock),
    blockPrimaryFunction block = BlockFunction.AssignmentCalculation
    ↔ block = ProgramBlock.Processing :=
by sorry

end NUMINAMATH_CALUDE_processing_block_performs_assignment_calculation_l291_29193


namespace NUMINAMATH_CALUDE_sqrt_one_hundredth_l291_29146

theorem sqrt_one_hundredth : Real.sqrt (1 / 100) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_hundredth_l291_29146


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l291_29178

theorem sum_of_coefficients : 
  let p (x : ℝ) := (3*x^8 - 2*x^7 + 4*x^6 - x^4 + 6*x^2 - 7) - 
                   5*(x^5 - 2*x^3 + 2*x - 8) + 
                   6*(x^6 + x^4 - 3)
  p 1 = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l291_29178


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l291_29128

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum % 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l291_29128


namespace NUMINAMATH_CALUDE_tetrahedron_vertices_tetrahedron_has_four_vertices_l291_29135

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- No specific fields needed for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat := 4

/-- Proof that a tetrahedron has 4 vertices. -/
theorem tetrahedron_has_four_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_vertices_tetrahedron_has_four_vertices_l291_29135


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l291_29197

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (1 + m) 3 ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l291_29197


namespace NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l291_29132

/-- Represents a 12-hour digital clock with a faulty display of '2' as '5' -/
structure FaultyClock where
  /-- The number of hours that display correctly -/
  correct_hours : ℕ
  /-- The number of minutes per hour that display correctly -/
  correct_minutes : ℕ

/-- The fraction of the day during which the faulty clock displays the correct time -/
def correct_time_fraction (clock : FaultyClock) : ℚ :=
  (clock.correct_hours : ℚ) / 12 * (clock.correct_minutes : ℚ) / 60

/-- The specific faulty clock described in the problem -/
def problem_clock : FaultyClock := {
  correct_hours := 10,
  correct_minutes := 44
}

theorem faulty_clock_correct_time_fraction :
  correct_time_fraction problem_clock = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l291_29132


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l291_29173

def is_mersenne_prime (p : Nat) : Prop :=
  ∃ n : Nat, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ q : Nat, is_mersenne_prime q ∧ q < 1000 → q ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l291_29173


namespace NUMINAMATH_CALUDE_equation_represents_ellipse_and_hyperbola_l291_29141

-- Define the equation
def equation (x y : ℝ) : Prop := y^4 - 6*x^4 = 3*y^2 - 2

-- Define what constitutes an ellipse in this context
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Define what constitutes a hyperbola in this context
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Theorem statement
theorem equation_represents_ellipse_and_hyperbola :
  (is_ellipse equation) ∧ (is_hyperbola equation) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_ellipse_and_hyperbola_l291_29141


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l291_29103

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧
  (∃ a b, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l291_29103


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l291_29169

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l291_29169


namespace NUMINAMATH_CALUDE_T_divisibility_l291_29172

-- Define the set T
def T : Set ℕ := {x | ∃ n : ℕ, x = (2*n - 2)^2 + (2*n)^2 + (2*n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, 4 ∣ x) ∧ (∃ x ∈ T, 5 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l291_29172


namespace NUMINAMATH_CALUDE_lagrange_interpolation_uniqueness_existence_l291_29177

theorem lagrange_interpolation_uniqueness_existence
  (n : ℕ) 
  (x : Fin (n + 1) → ℝ) 
  (a : Fin (n + 1) → ℝ) 
  (h_distinct : ∀ (i j : Fin (n + 1)), i ≠ j → x i ≠ x j) :
  ∃! P : Polynomial ℝ, 
    (Polynomial.degree P ≤ n) ∧ 
    (∀ i : Fin (n + 1), P.eval (x i) = a i) :=
sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_uniqueness_existence_l291_29177


namespace NUMINAMATH_CALUDE_popped_kernels_in_final_bag_l291_29176

theorem popped_kernels_in_final_bag 
  (bag1_popped bag1_total bag2_popped bag2_total bag3_total : ℕ)
  (average_percent : ℚ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_total = 100)
  (h6 : average_percent = 82/100)
  (h7 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + 
        (bag3_popped : ℚ) / bag3_total = 3 * average_percent) :
  bag3_popped = 82 := by
  sorry

#check popped_kernels_in_final_bag

end NUMINAMATH_CALUDE_popped_kernels_in_final_bag_l291_29176


namespace NUMINAMATH_CALUDE_problem_statement_l291_29196

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l291_29196


namespace NUMINAMATH_CALUDE_three_digit_automorphic_numbers_l291_29105

theorem three_digit_automorphic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n^2 % 1000 = n} = {625, 376} := by sorry

end NUMINAMATH_CALUDE_three_digit_automorphic_numbers_l291_29105


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l291_29170

theorem square_sum_equals_three (a b : ℝ) 
  (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l291_29170


namespace NUMINAMATH_CALUDE_right_triangle_equality_l291_29129

theorem right_triangle_equality (a b c p S : ℝ) : 
  a ≤ b → b ≤ c → 
  2 * p = a + b + c → 
  S = (1/2) * a * b → 
  a^2 + b^2 = c^2 → 
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l291_29129


namespace NUMINAMATH_CALUDE_doug_money_l291_29161

theorem doug_money (j d b s : ℚ) : 
  j + d + b + s = 150 →
  j = 2 * b →
  j = (3/4) * d →
  s = (1/2) * (j + d + b) →
  d = (4/3) * (150 * 12/41) := by
sorry

end NUMINAMATH_CALUDE_doug_money_l291_29161


namespace NUMINAMATH_CALUDE_expression_evaluation_l291_29124

theorem expression_evaluation : (3^(2+3+4) - (3^2 * 3^3 + 3^4)) = 19359 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l291_29124


namespace NUMINAMATH_CALUDE_third_number_is_58_l291_29159

def number_list : List ℕ := [54, 55, 58, 59, 62, 62, 63, 65, 65]

theorem third_number_is_58 : 
  number_list[2] = 58 := by sorry

end NUMINAMATH_CALUDE_third_number_is_58_l291_29159
