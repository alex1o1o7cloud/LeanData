import Mathlib

namespace NUMINAMATH_CALUDE_round_trip_average_speed_l965_96591

/-- The average speed of a round trip, given the outbound speed and relative duration of return trip -/
theorem round_trip_average_speed 
  (outbound_speed : ℝ) 
  (return_time_factor : ℝ) 
  (h1 : outbound_speed = 48) 
  (h2 : return_time_factor = 2) : 
  (2 * outbound_speed) / (1 + return_time_factor) = 32 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l965_96591


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocals_l965_96593

theorem cubic_roots_sum_of_cubes_reciprocals 
  (a b c d : ℂ) 
  (r s t : ℂ) 
  (h₁ : a ≠ 0) 
  (h₂ : d ≠ 0) 
  (h₃ : a * r^3 + b * r^2 + c * r + d = 0) 
  (h₄ : a * s^3 + b * s^2 + c * s + d = 0) 
  (h₅ : a * t^3 + b * t^2 + c * t + d = 0) : 
  1 / r^3 + 1 / s^3 + 1 / t^3 = c^3 / d^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocals_l965_96593


namespace NUMINAMATH_CALUDE_push_mower_rate_l965_96535

/-- Proves that the push mower's cutting rate is 1 acre per hour given the conditions of Jerry's lawn mowing scenario. -/
theorem push_mower_rate (total_acres : ℝ) (riding_mower_fraction : ℝ) (riding_mower_rate : ℝ) (total_mowing_time : ℝ) : 
  total_acres = 8 ∧ 
  riding_mower_fraction = 3/4 ∧ 
  riding_mower_rate = 2 ∧ 
  total_mowing_time = 5 → 
  (total_acres * (1 - riding_mower_fraction)) / (total_mowing_time - (total_acres * riding_mower_fraction) / riding_mower_rate) = 1 := by
  sorry

end NUMINAMATH_CALUDE_push_mower_rate_l965_96535


namespace NUMINAMATH_CALUDE_original_cube_volume_l965_96507

/-- Given two similar cubes where one has twice the side length of the other,
    if the larger cube has a volume of 216 cubic feet,
    then the smaller cube has a volume of 27 cubic feet. -/
theorem original_cube_volume
  (s : ℝ)  -- side length of the original cube
  (h1 : (2 * s) ^ 3 = 216)  -- volume of the larger cube is 216 cubic feet
  : s ^ 3 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cube_volume_l965_96507


namespace NUMINAMATH_CALUDE_teresa_bought_six_hardcovers_l965_96567

/-- The number of hardcover books Teresa bought -/
def num_hardcovers : ℕ := sorry

/-- The number of paperback books Teresa bought -/
def num_paperbacks : ℕ := sorry

/-- The total number of books Teresa bought -/
def total_books : ℕ := 12

/-- The cost of a hardcover book -/
def hardcover_cost : ℕ := 30

/-- The cost of a paperback book -/
def paperback_cost : ℕ := 18

/-- The total amount Teresa spent -/
def total_spent : ℕ := 288

/-- Theorem stating that Teresa bought 6 hardcover books -/
theorem teresa_bought_six_hardcovers :
  num_hardcovers = 6 ∧
  num_hardcovers ≥ 4 ∧
  num_hardcovers + num_paperbacks = total_books ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_teresa_bought_six_hardcovers_l965_96567


namespace NUMINAMATH_CALUDE_tangent_line_equation_l965_96566

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The x-coordinate of the point of tangency -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of tangency -/
def y₀ : ℝ := -3

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -3*x + 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l965_96566


namespace NUMINAMATH_CALUDE_betty_age_l965_96588

theorem betty_age (carol alice betty : ℝ) 
  (h1 : carol = 5 * alice)
  (h2 : carol = 2 * betty)
  (h3 : alice = carol - 12) :
  betty = 7.5 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l965_96588


namespace NUMINAMATH_CALUDE_percentage_problem_l965_96536

theorem percentage_problem : ∃ p : ℝ, (p / 100) * 16 = 0.04 ∧ p = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l965_96536


namespace NUMINAMATH_CALUDE_equation_solutions_l965_96527

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 11 ∧ x₂ = 2 + Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧
    (x₁ - 3)^2 + 2*(x₁ - 3) = 0 ∧ (x₂ - 3)^2 + 2*(x₂ - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l965_96527


namespace NUMINAMATH_CALUDE_marked_circle_triangles_l965_96558

/-- A circle with n equally spaced points on its circumference -/
structure MarkedCircle (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Number of triangles that can be formed with n points -/
def num_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of equilateral triangles that can be formed with n points -/
def num_equilateral_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of right triangles that can be formed with n points -/
def num_right_triangles (c : MarkedCircle n) : ℕ := sorry

theorem marked_circle_triangles 
  (c4 : MarkedCircle 4) 
  (c5 : MarkedCircle 5) 
  (c6 : MarkedCircle 6) : 
  (num_triangles c4 = 4) ∧ 
  (num_equilateral_triangles c5 = 0) ∧ 
  (num_right_triangles c6 = 12) := by sorry

end NUMINAMATH_CALUDE_marked_circle_triangles_l965_96558


namespace NUMINAMATH_CALUDE_remainder_8347_mod_9_l965_96590

theorem remainder_8347_mod_9 : 8347 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8347_mod_9_l965_96590


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l965_96509

/-- Q(N, k) represents the probability that no blue ball is adjacent to the red ball -/
def Q (N k : ℕ) : ℚ := (N + 1 : ℚ) / (N + k + 1 : ℚ)

/-- The smallest sum of N and k satisfying the conditions -/
def smallest_sum : ℕ := 4

theorem smallest_sum_proof :
  ∀ N k : ℕ,
    (N + k) % 4 = 0 →
    Q N k < 7/9 →
    N + k ≥ smallest_sum :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l965_96509


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l965_96548

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x - h)^2 + k -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally and vertically -/
def Parabola.shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

/-- The original parabola y = -2x^2 -/
def original_parabola : Parabola :=
  { a := -2, h := 0, k := 0 }

/-- Theorem stating that shifting the original parabola down 1 unit and right 3 units
    results in the equation y = -2(x - 3)^2 - 1 -/
theorem parabola_shift_theorem (x : ℝ) :
  (original_parabola.shift 3 (-1)).equation x = -2 * (x - 3)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l965_96548


namespace NUMINAMATH_CALUDE_simplify_expression_l965_96510

theorem simplify_expression (m n : ℤ) (h : m * n = m + 3) :
  2 * m * n + 3 * m - 5 * m * n - 10 = -19 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l965_96510


namespace NUMINAMATH_CALUDE_constant_k_value_l965_96529

/-- Given that -x^2 - (k + 10)x - 8 = -(x - 2)(x - 4) for all real x, prove that k = -16 -/
theorem constant_k_value (k : ℝ) 
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 := by sorry

end NUMINAMATH_CALUDE_constant_k_value_l965_96529


namespace NUMINAMATH_CALUDE_range_of_f_l965_96585

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 else 2^x

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l965_96585


namespace NUMINAMATH_CALUDE_second_smallest_palindrome_l965_96596

/-- Checks if a number is a palindrome in a given base -/
def is_palindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from one base to another -/
def base_conversion (n : ℕ) (from_base to_base : ℕ) : ℕ := sorry

/-- Counts the number of digits of a number in a given base -/
def digit_count (n : ℕ) (base : ℕ) : ℕ := sorry

/-- The second smallest 5-digit palindrome in base 2 -/
def target_number : ℕ := 21

theorem second_smallest_palindrome :
  (is_palindrome target_number 2 = true) ∧
  (digit_count target_number 2 = 5) ∧
  (∃ (b : ℕ), b > 2 ∧ is_palindrome (base_conversion target_number 2 b) b = true ∧ 
              digit_count (base_conversion target_number 2 b) b = 3) ∧
  (∀ (n : ℕ), n < target_number →
    ¬(is_palindrome n 2 = true ∧
      digit_count n 2 = 5 ∧
      (∃ (b : ℕ), b > 2 ∧ is_palindrome (base_conversion n 2 b) b = true ∧ 
                  digit_count (base_conversion n 2 b) b = 3)) ∨
    n = 17) := by sorry

end NUMINAMATH_CALUDE_second_smallest_palindrome_l965_96596


namespace NUMINAMATH_CALUDE_vanya_climb_ratio_l965_96595

-- Define the floors for Anya and Vanya
def anya_floor : ℕ := 2
def vanya_floor : ℕ := 6
def start_floor : ℕ := 1

-- Define the climbs for Anya and Vanya
def anya_climb : ℕ := anya_floor - start_floor
def vanya_climb : ℕ := vanya_floor - start_floor

-- Theorem statement
theorem vanya_climb_ratio :
  (vanya_climb : ℚ) / (anya_climb : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_vanya_climb_ratio_l965_96595


namespace NUMINAMATH_CALUDE_g_of_3_l965_96571

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem g_of_3 : g 3 = 126 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l965_96571


namespace NUMINAMATH_CALUDE_running_gender_related_l965_96579

structure RunningData where
  total_students : Nat
  male_students : Nat
  female_like_running : Nat
  male_dislike_running : Nat

def chi_square (data : RunningData) : Rat :=
  let female_students := data.total_students - data.male_students
  let male_like_running := data.male_students - data.male_dislike_running
  let female_dislike_running := female_students - data.female_like_running
  let n := data.total_students
  let a := male_like_running
  let b := data.male_dislike_running
  let c := data.female_like_running
  let d := female_dislike_running
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def is_gender_related (data : RunningData) : Prop :=
  chi_square data > 6635 / 1000

theorem running_gender_related (data : RunningData) 
  (h1 : data.total_students = 200)
  (h2 : data.male_students = 120)
  (h3 : data.female_like_running = 30)
  (h4 : data.male_dislike_running = 50) :
  is_gender_related data := by
  sorry

#eval chi_square { total_students := 200, male_students := 120, female_like_running := 30, male_dislike_running := 50 }

end NUMINAMATH_CALUDE_running_gender_related_l965_96579


namespace NUMINAMATH_CALUDE_intersection_of_sets_l965_96546

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - x < 0} →
  B = {x : ℝ | -2 < x ∧ x < 2} →
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l965_96546


namespace NUMINAMATH_CALUDE_z_value_when_x_is_one_l965_96597

-- Define the variables and their properties
variable (x y z : ℝ)

-- Define the conditions
axiom positive : x > 0 ∧ y > 0 ∧ z > 0
axiom inverse_relation : ∃ (k : ℝ), ∀ x y, x^2 * y = k
axiom direct_relation : ∃ (c : ℝ), ∀ y z, y / z = c
axiom initial_condition1 : x = 4 → y = 8
axiom initial_condition2 : x = 4 → z = 32

-- State the theorem
theorem z_value_when_x_is_one :
  x = 1 → z = 512 := by sorry

end NUMINAMATH_CALUDE_z_value_when_x_is_one_l965_96597


namespace NUMINAMATH_CALUDE_chamber_boundary_area_l965_96524

/-- The area of the boundary of a chamber formed by three intersecting pipes -/
theorem chamber_boundary_area (pipe_circumference : ℝ) (h1 : pipe_circumference = 4) :
  let pipe_diameter := pipe_circumference / Real.pi
  let cross_section_area := Real.pi * (pipe_diameter / 2) ^ 2
  let chamber_boundary_area := 2 * (1 / 4) * Real.pi * pipe_diameter ^ 2
  chamber_boundary_area = 8 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_chamber_boundary_area_l965_96524


namespace NUMINAMATH_CALUDE_investment_problem_l965_96560

theorem investment_problem (x : ℝ) : 
  (0.07 * x + 0.19 * 1500 = 0.16 * (x + 1500)) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l965_96560


namespace NUMINAMATH_CALUDE_smallest_triangle_leg_l965_96533

-- Define the properties of a 30-60-90 triangle
def thirty_sixty_ninety_triangle (short_leg long_leg hypotenuse : ℝ) : Prop :=
  short_leg = hypotenuse / 2 ∧ long_leg = short_leg * Real.sqrt 3

-- Define the sequence of four connected triangles
def connected_triangles (h1 h2 h3 h4 : ℝ) : Prop :=
  ∃ (s1 l1 s2 l2 s3 l3 s4 l4 : ℝ),
    thirty_sixty_ninety_triangle s1 l1 h1 ∧
    thirty_sixty_ninety_triangle s2 l2 h2 ∧
    thirty_sixty_ninety_triangle s3 l3 h3 ∧
    thirty_sixty_ninety_triangle s4 l4 h4 ∧
    l1 = h2 ∧ l2 = h3 ∧ l3 = h4

theorem smallest_triangle_leg (h1 h2 h3 h4 : ℝ) :
  h1 = 10 → connected_triangles h1 h2 h3 h4 → l4 = 45 / 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_leg_l965_96533


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l965_96540

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l965_96540


namespace NUMINAMATH_CALUDE_all_points_on_single_line_l965_96521

/-- A point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear. -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, pointOnLine p l ∧ pointOnLine q l ∧ pointOnLine r l

/-- The main theorem. -/
theorem all_points_on_single_line (k : ℕ) (points : Fin k → Point)
    (h : ∀ i j : Fin k, i ≠ j → ∃ m : Fin k, m ≠ i ∧ m ≠ j ∧ collinear (points i) (points j) (points m)) :
    ∃ l : Line, ∀ i : Fin k, pointOnLine (points i) l := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_single_line_l965_96521


namespace NUMINAMATH_CALUDE_max_rectangles_correct_l965_96580

/-- The maximum number of 1 × (n + 1) rectangles that can be cut from a 2n × 2n square -/
def max_rectangles (n : ℕ) : ℕ :=
  if n ≥ 4 then 4 * (n - 1)
  else if n = 1 then 2
  else if n = 2 then 5
  else 8

theorem max_rectangles_correct (n : ℕ) :
  max_rectangles n = 
    if n ≥ 4 then 4 * (n - 1)
    else if n = 1 then 2
    else if n = 2 then 5
    else 8 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_correct_l965_96580


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_eight_l965_96578

theorem three_digit_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ (n / 100) % 10 = 5 ∧ n % 8 = 0 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_eight_l965_96578


namespace NUMINAMATH_CALUDE_calculation_proof_l965_96539

theorem calculation_proof :
  (4 + (-2)^3 * 5 - (-0.28) / 4 = -35.93) ∧
  (-1^4 - 1/6 * (2 - (-3)^2) = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l965_96539


namespace NUMINAMATH_CALUDE_negative_quartic_count_l965_96554

theorem negative_quartic_count :
  (∃! (s : Finset ℤ), (∀ x ∈ s, x^4 - 51*x^2 + 100 < 0) ∧ s.card = 10) := by
  sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l965_96554


namespace NUMINAMATH_CALUDE_sin_2a_value_l965_96559

theorem sin_2a_value (a : Real) (h1 : a ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin (π / 4 - a)) : 
  Real.sin (2 * a) = -8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2a_value_l965_96559


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l965_96564

/-- Given a circle with radius 5 cm tangent to three sides of a rectangle, 
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 7.5π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_radius = 5)
  (h2 : rectangle_area = 3 * π * circle_radius^2) : 
  rectangle_area / (2 * circle_radius) = 7.5 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l965_96564


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l965_96576

theorem number_of_divisors_36 : Nat.card {d : ℕ | d ∣ 36} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l965_96576


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l965_96572

theorem projectile_trajectory_area 
  (u : ℝ) 
  (k : ℝ) 
  (φ : ℝ) 
  (h_φ_range : 30 * π / 180 ≤ φ ∧ φ ≤ 150 * π / 180) 
  (h_u_pos : u > 0) 
  (h_k_pos : k > 0) : 
  ∃ d : ℝ, d = π / 8 ∧ 
    (∀ x y : ℝ, (x^2 / (u^2 / (2 * k))^2 + (y - u^2 / (4 * k))^2 / (u^2 / (4 * k))^2 = 1) → 
      π * (u^2 / (2 * k)) * (u^2 / (4 * k)) = d * u^4 / k^2) := by
sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l965_96572


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l965_96563

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 is 48π -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (area : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  area = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  area = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l965_96563


namespace NUMINAMATH_CALUDE_donny_spending_friday_sunday_l965_96531

def monday_savings : ℝ := 15

def savings_increase_rate : ℝ := 0.1

def friday_spending_rate : ℝ := 0.5

def saturday_savings_decrease : ℝ := 0.2

def sunday_spending_rate : ℝ := 0.4

def tuesday_savings (monday : ℝ) : ℝ := monday * (1 + savings_increase_rate)

def wednesday_savings (tuesday : ℝ) : ℝ := tuesday * (1 + savings_increase_rate)

def thursday_savings (wednesday : ℝ) : ℝ := wednesday * (1 + savings_increase_rate)

def total_savings_thursday (mon tue wed thu : ℝ) : ℝ := mon + tue + wed + thu

def friday_spending (total : ℝ) : ℝ := total * friday_spending_rate

def saturday_savings (thursday : ℝ) : ℝ := thursday * (1 - saturday_savings_decrease)

def total_savings_saturday (friday_remaining saturday : ℝ) : ℝ := friday_remaining + saturday

def sunday_spending (total : ℝ) : ℝ := total * sunday_spending_rate

theorem donny_spending_friday_sunday : 
  let tue := tuesday_savings monday_savings
  let wed := wednesday_savings tue
  let thu := thursday_savings wed
  let total_thu := total_savings_thursday monday_savings tue wed thu
  let fri_spend := friday_spending total_thu
  let fri_remaining := total_thu - fri_spend
  let sat := saturday_savings thu
  let total_sat := total_savings_saturday fri_remaining sat
  let sun_spend := sunday_spending total_sat
  fri_spend + sun_spend = 55.13 := by sorry

end NUMINAMATH_CALUDE_donny_spending_friday_sunday_l965_96531


namespace NUMINAMATH_CALUDE_toilet_paper_cost_l965_96555

/-- Prove that the cost of one roll of toilet paper is $1.50 -/
theorem toilet_paper_cost 
  (total_toilet_paper : ℕ) 
  (total_paper_towels : ℕ) 
  (total_tissues : ℕ) 
  (total_cost : ℚ) 
  (paper_towel_cost : ℚ) 
  (tissue_cost : ℚ) 
  (h1 : total_toilet_paper = 10)
  (h2 : total_paper_towels = 7)
  (h3 : total_tissues = 3)
  (h4 : total_cost = 35)
  (h5 : paper_towel_cost = 2)
  (h6 : tissue_cost = 2) :
  (total_cost - (total_paper_towels * paper_towel_cost + total_tissues * tissue_cost)) / total_toilet_paper = (3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_toilet_paper_cost_l965_96555


namespace NUMINAMATH_CALUDE_ellipse_theorem_l965_96581

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Main theorem about the ellipse and the maximum product -/
theorem ellipse_theorem (C : Ellipse) (P Q : Point) (l : Line) (F : Point) (circle : Circle) :
  (P.x = 1 ∧ P.y = Real.sqrt 2 / 2) →
  (Q.x = -Real.sqrt 2 ∧ Q.y = 0) →
  (C.a^2 * P.y^2 + C.b^2 * P.x^2 = C.a^2 * C.b^2) →
  (C.a^2 * Q.y^2 + C.b^2 * Q.x^2 = C.a^2 * C.b^2) →
  (∃ (A B E : Point), A ≠ B ∧ E ≠ F ∧
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) ∧
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) ∧
    (l.p1 = F ∧ l.p2 = A) ∧
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2)) →
  (C.a = Real.sqrt 2 ∧ C.b = 1) ∧
  (∀ (A B E : Point), 
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) →
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) →
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2) →
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) * Real.sqrt ((F.x - E.x)^2 + (F.y - E.y)^2) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l965_96581


namespace NUMINAMATH_CALUDE_twenty_factorial_digits_sum_l965_96518

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twenty_factorial_digits_sum (B A : ℕ) : 
  B < 10 → A < 10 → 
  ∃ k : ℕ, factorial 20 = k * 10000 + B * 100 + A * 10 → 
  B + A = 10 := by
  sorry

end NUMINAMATH_CALUDE_twenty_factorial_digits_sum_l965_96518


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l965_96502

theorem repeating_decimal_subtraction :
  ∃ (a b c : ℚ),
    (1000 * a - a = 567) ∧
    (1000 * b - b = 234) ∧
    (1000 * c - c = 345) ∧
    (a - b - c = -4 / 333) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l965_96502


namespace NUMINAMATH_CALUDE_simplify_polynomial_l965_96534

theorem simplify_polynomial (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l965_96534


namespace NUMINAMATH_CALUDE_discount_percentage_l965_96573

theorem discount_percentage (cupcake_price cookie_price : ℝ) 
  (cupcakes_sold cookies_sold : ℕ) (total_revenue : ℝ) :
  cupcake_price = 3 →
  cookie_price = 2 →
  cupcakes_sold = 16 →
  cookies_sold = 8 →
  total_revenue = 32 →
  ∃ (x : ℝ), 
    (cupcakes_sold : ℝ) * (cupcake_price * (100 - x) / 100) + 
    (cookies_sold : ℝ) * (cookie_price * (100 - x) / 100) = total_revenue ∧
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l965_96573


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l965_96505

theorem arithmetic_mean_of_fractions : 
  (3 / 7 + 5 / 8) / 2 = 59 / 112 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l965_96505


namespace NUMINAMATH_CALUDE_quadratic_function_zero_equivalence_l965_96592

/-- A quadratic function f(x) = ax² + bx + c where a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value for a given x -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The set of zeros of the function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.value x = 0}

/-- The composition of the function with itself -/
def QuadraticFunction.compose_self (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.value (f.value x)

theorem quadratic_function_zero_equivalence (f : QuadraticFunction) :
  (f.zeros = {x : ℝ | f.compose_self x = 0}) ↔ f.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_zero_equivalence_l965_96592


namespace NUMINAMATH_CALUDE_angle_sets_relation_l965_96513

-- Define the sets A, B, and C
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- State the theorem
theorem angle_sets_relation : B ∪ C = C := by
  sorry

end NUMINAMATH_CALUDE_angle_sets_relation_l965_96513


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l965_96574

theorem probability_of_white_ball
  (P_red P_black P_yellow P_white : ℝ)
  (h_red : P_red = 1/3)
  (h_black_yellow : P_black + P_yellow = 5/12)
  (h_yellow_white : P_yellow + P_white = 5/12)
  (h_sum : P_red + P_black + P_yellow + P_white = 1)
  (h_nonneg : P_red ≥ 0 ∧ P_black ≥ 0 ∧ P_yellow ≥ 0 ∧ P_white ≥ 0) :
  P_white = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l965_96574


namespace NUMINAMATH_CALUDE_bill_face_value_l965_96547

/-- Face value of a bill given true discount and banker's discount -/
def face_value (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount * bankers_discount / (bankers_discount - true_discount)

/-- Theorem: Given the true discount and banker's discount, prove the face value is 2460 -/
theorem bill_face_value :
  face_value 360 421.7142857142857 = 2460 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l965_96547


namespace NUMINAMATH_CALUDE_solution_set_f_g_has_zero_condition_l965_96528

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem solution_set_f (x : ℝ) : 
  f 3 x > 6 ↔ x < -4 ∨ x > 2 := by sorry

-- Part II
theorem g_has_zero_condition (a : ℝ) :
  (∃ x, g a x = 0) → a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_g_has_zero_condition_l965_96528


namespace NUMINAMATH_CALUDE_equal_playing_time_l965_96514

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_playing_time_l965_96514


namespace NUMINAMATH_CALUDE_sandy_has_144_marbles_l965_96561

-- Define the number of red marbles Jessica has
def jessica_marbles : ℕ := 3 * 12

-- Define the relationship between Sandy's and Jessica's marbles
def sandy_marbles : ℕ := 4 * jessica_marbles

-- Theorem to prove
theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_144_marbles_l965_96561


namespace NUMINAMATH_CALUDE_remainder_scaling_l965_96568

theorem remainder_scaling (a b c r : ℤ) : 
  (a = b * c + r) → (0 ≤ r) → (r < b) → 
  ∃ (q : ℤ), (3 * a = 3 * b * q + 3 * r) ∧ (0 ≤ 3 * r) ∧ (3 * r < 3 * b) :=
by sorry

end NUMINAMATH_CALUDE_remainder_scaling_l965_96568


namespace NUMINAMATH_CALUDE_harry_seashells_count_l965_96519

theorem harry_seashells_count :
  ∀ (seashells : ℕ),
    -- Initial collection
    34 + seashells + 29 = 34 + seashells + 29 →
    -- Total items lost
    25 = 25 →
    -- Items left at the end
    59 = 59 →
    -- Proof that seashells = 21
    seashells = 21 := by
  sorry

end NUMINAMATH_CALUDE_harry_seashells_count_l965_96519


namespace NUMINAMATH_CALUDE_not_divides_power_plus_one_l965_96532

theorem not_divides_power_plus_one (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : Odd k) :
  ¬(n ∣ 2^(n - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_plus_one_l965_96532


namespace NUMINAMATH_CALUDE_largest_ball_radius_is_four_l965_96512

/-- Represents a torus in 3D space --/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  inner_radius : ℝ
  outer_radius : ℝ

/-- Represents a spherical ball in 3D space --/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The torus described in the problem --/
def problem_torus : Torus :=
  { center := (4, 0, 1)
    radius := 1
    inner_radius := 3
    outer_radius := 5 }

/-- 
  Given a torus sitting on the xy-plane, returns the radius of the largest
  spherical ball that can be placed on top of the center of the torus and
  still touch the horizontal plane
--/
def largest_ball_radius (t : Torus) : ℝ :=
  sorry

theorem largest_ball_radius_is_four :
  largest_ball_radius problem_torus = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_radius_is_four_l965_96512


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l965_96506

/-- Given a principal sum and conditions on simple interest, prove the annual interest rate -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * (20 / 7) * 7) / 100 = P / 5 → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l965_96506


namespace NUMINAMATH_CALUDE_angle_sum_bound_l965_96508

theorem angle_sum_bound (A B : Real) (h_triangle : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_inequality : ∀ x > 0, (Real.sin B / Real.cos A)^x + (Real.sin A / Real.cos B)^x < 2) :
  0 < A + B ∧ A + B < π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_bound_l965_96508


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l965_96582

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let circle_radius : ℝ := 7
  let length_to_width_ratio : ℝ := 3
  let rectangle_width : ℝ := 2 * circle_radius
  let rectangle_length : ℝ := length_to_width_ratio * rectangle_width
  rectangle_length * rectangle_width

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l965_96582


namespace NUMINAMATH_CALUDE_binomial_13_11_l965_96500

theorem binomial_13_11 : Nat.choose 13 11 = 78 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_11_l965_96500


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l965_96587

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 126 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l965_96587


namespace NUMINAMATH_CALUDE_lisa_marble_difference_l965_96562

/-- Proves that Lisa has 19 more marbles than Cindy after the marble exchange -/
theorem lisa_marble_difference (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) : 
  cindy_initial = 20 →
  cindy_initial = lisa_initial + 5 →
  marbles_given = 12 →
  (lisa_initial + marbles_given) - (cindy_initial - marbles_given) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_difference_l965_96562


namespace NUMINAMATH_CALUDE_total_painting_time_l965_96541

/-- Time to paint each type of flower in minutes -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def sunflower_time : ℕ := 10
def tulip_time : ℕ := 4
def vine_time : ℕ := 2
def peony_time : ℕ := 8

/-- Number of each type of flower to paint -/
def lily_count : ℕ := 23
def rose_count : ℕ := 15
def orchid_count : ℕ := 9
def sunflower_count : ℕ := 12
def tulip_count : ℕ := 18
def vine_count : ℕ := 30
def peony_count : ℕ := 27

/-- Theorem stating the total time to paint all flowers -/
theorem total_painting_time : 
  lily_time * lily_count + 
  rose_time * rose_count + 
  orchid_time * orchid_count + 
  sunflower_time * sunflower_count + 
  tulip_time * tulip_count + 
  vine_time * vine_count + 
  peony_time * peony_count = 715 := by
  sorry

end NUMINAMATH_CALUDE_total_painting_time_l965_96541


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l965_96544

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₅ = -1 and a₈ = 2,
    prove that the common difference is 1 and the first term is -5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = -1)
    (h_a8 : a 8 = 2) :
    (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 ∧ a 1 = -5 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l965_96544


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l965_96552

/-- Given a triangle with sides a, b, and c, prove that (a^2 + b^2 + ab) / c^2 < 1 --/
theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < c) (triangle : c < a + b) :
  (a^2 + b^2 + a*b) / c^2 < 1 := by
  sorry

#check triangle_inequality_squared

end NUMINAMATH_CALUDE_triangle_inequality_squared_l965_96552


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l965_96503

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x + 2

-- Define the points on the parabola
def point_a : ℝ × ℝ := (2, f 2)
def point_b : ℝ × ℝ := (-1, f (-1))
def point_c : ℝ × ℝ := (3, f 3)

-- Theorem stating the relationship between a, b, and c
theorem parabola_point_relationship :
  point_c.2 > point_a.2 ∧ point_a.2 > point_b.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l965_96503


namespace NUMINAMATH_CALUDE_extreme_values_of_f_range_of_a_for_f_greater_than_g_l965_96550

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - a * x + (a + 3) / x
def g (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 4 * x + 2 * a

-- Theorem for part 1
theorem extreme_values_of_f (x : ℝ) (hx : x > 0) :
  let f_half := f (1/2)
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ y, y > 0 → f_half y ≥ f_half x_min) ∧
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ y, y > 0 → f_half y ≤ f_half x_max) ∧
  (∀ y, y > 0 → f_half y ≥ 3) ∧
  (∀ y, y > 0 → f_half y ≤ 4 * Real.log 7 - 3) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_f_greater_than_g (a : ℝ) (ha : a ≥ 1) :
  (∃ (x₁ x₂ : ℝ), 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 ∧ f a x₁ > g a x₂) ↔
  (1 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_range_of_a_for_f_greater_than_g_l965_96550


namespace NUMINAMATH_CALUDE_min_sum_x_y_l965_96538

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) :
  x + y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2 * x₀ + Real.sqrt (4 * x₀^2 + 1)) * (Real.sqrt (y₀^2 + 4) - 2) ≥ y₀ ∧
    x₀ + y₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_x_y_l965_96538


namespace NUMINAMATH_CALUDE_total_spent_is_thirteen_l965_96570

-- Define the cost of items
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := 6

-- Define the total spent
def total_spent : ℕ := candy_bar_cost + chocolate_cost

-- Theorem to prove
theorem total_spent_is_thirteen : total_spent = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_thirteen_l965_96570


namespace NUMINAMATH_CALUDE_pancake_breakfast_fundraiser_l965_96589

def pancake_price : ℚ := 4
def bacon_price : ℚ := 2
def pancake_stacks_sold : ℕ := 60
def bacon_slices_sold : ℕ := 90

theorem pancake_breakfast_fundraiser :
  (pancake_price * pancake_stacks_sold + bacon_price * bacon_slices_sold : ℚ) = 420 := by
  sorry

end NUMINAMATH_CALUDE_pancake_breakfast_fundraiser_l965_96589


namespace NUMINAMATH_CALUDE_intersection_point_property_l965_96553

theorem intersection_point_property (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  b = -2 / a ∧ b = a + 3 → 1 / a - 1 / b = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l965_96553


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l965_96565

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 5, 6}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l965_96565


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l965_96520

theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_sphere := (4/3) * π * r_sphere^3
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372/3) - 16 * Real.sqrt 132) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l965_96520


namespace NUMINAMATH_CALUDE_mary_fruits_left_l965_96530

/-- The number of fruits Mary has left after buying, using for salad, eating, and giving away -/
def fruits_left (apples oranges blueberries grapes kiwis : ℕ) 
  (apples_salad oranges_salad blueberries_salad : ℕ)
  (apples_eaten oranges_eaten kiwis_eaten : ℕ)
  (apples_given oranges_given blueberries_given grapes_given kiwis_given : ℕ) : ℕ :=
  (apples - apples_salad - apples_eaten - apples_given) +
  (oranges - oranges_salad - oranges_eaten - oranges_given) +
  (blueberries - blueberries_salad - blueberries_given) +
  (grapes - grapes_given) +
  (kiwis - kiwis_eaten - kiwis_given)

/-- Theorem stating that Mary has 61 fruits left -/
theorem mary_fruits_left : 
  fruits_left 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l965_96530


namespace NUMINAMATH_CALUDE_student_count_prove_student_count_l965_96511

theorem student_count (weight_difference : ℝ) (average_decrease : ℝ) : ℝ :=
  weight_difference / average_decrease

theorem prove_student_count :
  let weight_difference : ℝ := 120 - 60
  let average_decrease : ℝ := 6
  student_count weight_difference average_decrease = 10 := by
    sorry

end NUMINAMATH_CALUDE_student_count_prove_student_count_l965_96511


namespace NUMINAMATH_CALUDE_intersection_difference_l965_96598

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 5

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

-- Theorem statement
theorem intersection_difference : 
  ∃ (p r : ℝ), p ∈ intersection_points ∧ r ∈ intersection_points ∧ 
  r ≥ p ∧ ∀ x ∈ intersection_points, (x = p ∨ x = r) ∧ 
  r - p = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l965_96598


namespace NUMINAMATH_CALUDE_five_million_times_eight_million_l965_96586

theorem five_million_times_eight_million :
  (5000000 : ℕ) * 8000000 = 40000000000000 := by
  sorry

end NUMINAMATH_CALUDE_five_million_times_eight_million_l965_96586


namespace NUMINAMATH_CALUDE_square_of_sum_equality_l965_96599

theorem square_of_sum_equality : 31^2 + 2*(31)*(5 + 3) + (5 + 3)^2 = 1521 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_equality_l965_96599


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l965_96543

theorem first_candidate_percentage (P : ℝ) (total_marks : ℝ) : 
  P = 199.99999999999997 →
  0.45 * total_marks = P + 25 →
  (P - 50) / total_marks * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l965_96543


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l965_96523

-- Define the ellipse parameters
def a : ℝ := 3
def b_squared : ℝ := 8

-- Define the focal length
def focal_length : ℝ := 2

-- Theorem statement
theorem ellipse_focal_length :
  focal_length = 2 * Real.sqrt (a^2 - b_squared) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l965_96523


namespace NUMINAMATH_CALUDE_tangent_line_m_value_l965_96594

-- Define the curve
def curve (x m n : ℝ) : ℝ := x^3 + m*x + n

-- Define the line
def line (x : ℝ) : ℝ := 3*x + 1

-- State the theorem
theorem tangent_line_m_value :
  ∀ (m n : ℝ),
  (curve 1 m n = 4) →  -- The point (1, 4) lies on the curve
  (line 1 = 4) →       -- The point (1, 4) lies on the line
  (∀ x : ℝ, curve x m n ≤ line x) →  -- The line is tangent to the curve (no other intersection)
  (m = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_m_value_l965_96594


namespace NUMINAMATH_CALUDE_equation_one_real_root_l965_96583

theorem equation_one_real_root :
  ∃! x : ℝ, (Real.sqrt (x^2 + 2*x - 63) + Real.sqrt (x + 9) - Real.sqrt (7 - x) + x + 13 = 0) ∧
             (x^2 + 2*x - 63 ≥ 0) ∧
             (x + 9 ≥ 0) ∧
             (7 - x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l965_96583


namespace NUMINAMATH_CALUDE_z_percentage_of_1000_l965_96522

theorem z_percentage_of_1000 (x y z : ℝ) : 
  x = (3/5) * 4864 →
  y = (2/3) * 9720 →
  z = (1/4) * 800 →
  (z / 1000) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_z_percentage_of_1000_l965_96522


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l965_96545

/-- The decimal representation of 0.1⁻35 as a real number -/
def decimal_rep : ℚ := 0.1 + (35 / 990)

/-- The fraction 67/495 as a rational number -/
def fraction : ℚ := 67 / 495

/-- Assertion that 67 and 495 are coprime -/
axiom coprime_67_495 : Nat.Coprime 67 495

theorem decimal_equals_fraction : decimal_rep = fraction := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l965_96545


namespace NUMINAMATH_CALUDE_line_y_intercept_l965_96584

/-- Given a line passing through the points (2, -3) and (6, 5), its y-intercept is -7 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -3) → 
  (f 6 = 5) → 
  (∀ x y, f x = y ↔ ∃ m b, y = m * x + b) →
  (∃ b, f 0 = b) →
  f 0 = -7 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l965_96584


namespace NUMINAMATH_CALUDE_decode_sequence_is_palindrome_l965_96569

/-- Represents the mapping from indices to letters -/
def letter_mapping : Nat → Char
| 1 => 'A'
| 2 => 'E'
| 3 => 'B'
| 4 => 'Γ'
| 5 => 'Δ'
| 6 => 'E'
| 7 => 'E'
| 8 => 'E'
| 9 => '3'
| 10 => 'V'
| 11 => 'U'
| 12 => 'K'
| 13 => 'J'
| 14 => 'M'
| 15 => 'H'
| 16 => 'O'
| 17 => '4'
| 18 => 'P'
| 19 => 'C'
| 20 => 'T'
| 21 => 'y'
| 22 => 'Φ'
| 23 => 'X'
| 24 => '4'
| 25 => '4'
| 26 => 'W'
| 27 => 'M'
| 28 => 'b'
| 29 => 'b'
| 30 => 'b'
| 31 => '3'
| 32 => 'O'
| 33 => '夕'
| _ => ' '  -- Default case

/-- The sequence of numbers to be decoded -/
def encoded_sequence : List Nat := [1, 1, 3, 0, 1, 1, 1, 7, 1, 5, 3, 1, 5, 1, 3, 2, 3, 2, 1, 5, 3, 1, 1, 2, 3, 2, 6, 2, 6, 1, 4, 1, 1, 2, 7, 3, 1, 4, 1, 1, 9, 1, 5, 0, 4, 1, 4, 9]

/-- Function to decode the sequence -/
def decode (seq : List Nat) : String := sorry

/-- The expected decoded palindrome -/
def expected_palindrome : String := "голоден носитель лет и сон не долг"

/-- Theorem stating that decoding the sequence results in the expected palindrome -/
theorem decode_sequence_is_palindrome : decode encoded_sequence = expected_palindrome := by sorry

end NUMINAMATH_CALUDE_decode_sequence_is_palindrome_l965_96569


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l965_96556

theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l965_96556


namespace NUMINAMATH_CALUDE_base_conversion_equality_l965_96577

-- Define the base-5 number 132₅
def base_5_num : ℕ := 1 * 5^2 + 3 * 5^1 + 2 * 5^0

-- Define the base-b number 221ᵦ as a function of b
def base_b_num (b : ℝ) : ℝ := 2 * b^2 + 2 * b + 1

-- Theorem statement
theorem base_conversion_equality :
  ∃ b : ℝ, b > 0 ∧ base_5_num = base_b_num b ∧ b = (-1 + Real.sqrt 83) / 2 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l965_96577


namespace NUMINAMATH_CALUDE_no_four_consecutive_power_numbers_l965_96516

theorem no_four_consecutive_power_numbers : 
  ¬ ∃ (n : ℕ), 
    (∃ (a b : ℕ) (k : ℕ), k > 1 ∧ n = a^k) ∧
    (∃ (c d : ℕ) (l : ℕ), l > 1 ∧ n + 1 = c^l) ∧
    (∃ (e f : ℕ) (m : ℕ), m > 1 ∧ n + 2 = e^m) ∧
    (∃ (g h : ℕ) (p : ℕ), p > 1 ∧ n + 3 = g^p) :=
by
  sorry


end NUMINAMATH_CALUDE_no_four_consecutive_power_numbers_l965_96516


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l965_96537

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l965_96537


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l965_96501

/-- Calculates the cost of paving a rectangular floor given its dimensions and the paving rate. -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m by 4m room at Rs. 950 per square meter is Rs. 20,900. -/
theorem paving_cost_calculation :
  paving_cost 5.5 4 950 = 20900 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l965_96501


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l965_96504

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ x = -3/2 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l965_96504


namespace NUMINAMATH_CALUDE_gathering_attendance_l965_96517

/-- The number of people who took wine -/
def W : ℕ := 26

/-- The number of people who took soda -/
def S : ℕ := 22

/-- The number of people who took juice -/
def J : ℕ := 18

/-- The number of people who took both wine and soda -/
def WS : ℕ := 17

/-- The number of people who took both wine and juice -/
def WJ : ℕ := 12

/-- The number of people who took both soda and juice -/
def SJ : ℕ := 10

/-- The number of people who took all three drinks -/
def WSJ : ℕ := 8

/-- The total number of people at the gathering -/
def total_people : ℕ := W + S + J - WS - WJ - SJ + WSJ

theorem gathering_attendance : total_people = 35 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendance_l965_96517


namespace NUMINAMATH_CALUDE_polynomial_roots_l965_96542

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => x^3 + 2*x^2 - 5*x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l965_96542


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l965_96557

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least one white ball when randomly selecting 3 balls from a bag 
    containing 5 white balls and 4 red balls -/
theorem prob_at_least_one_white : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l965_96557


namespace NUMINAMATH_CALUDE_possible_values_of_a_l965_96526

def A (a : ℕ) : Set ℕ := {2, 4, a}
def B : Set ℕ := {1, 2, 3}

theorem possible_values_of_a (a : ℕ) :
  A a ∪ B = {1, 2, 3, 4} → a = 1 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l965_96526


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l965_96549

def N : Matrix (Fin 2) (Fin 2) ℚ := !![4, 0; 2, -6]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/24 ∧ d = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l965_96549


namespace NUMINAMATH_CALUDE_six_regions_three_colors_l965_96575

/-- The number of ways to color n regions using k colors --/
def colorings (n k : ℕ) : ℕ := k^n

/-- The number of ways to color n regions using exactly k colors --/
def exactColorings (n k : ℕ) : ℕ :=
  (Nat.choose k k) * k^n - (Nat.choose k (k-1)) * (k-1)^n + (Nat.choose k (k-2)) * (k-2)^n

theorem six_regions_three_colors :
  exactColorings 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_six_regions_three_colors_l965_96575


namespace NUMINAMATH_CALUDE_divisor_problem_l965_96551

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (10154 - 14) % d = 0 ∧ d = 10140 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l965_96551


namespace NUMINAMATH_CALUDE_fraction_equality_l965_96525

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l965_96525


namespace NUMINAMATH_CALUDE_tempo_original_value_l965_96515

/-- The original value of the tempo -/
def original_value : ℝ := 11083.33

/-- The insurance coverage percentage for all three years -/
def coverage_percentage : ℚ := 5/7

/-- The premium rate for the first year -/
def premium_rate_year1 : ℚ := 3/100

/-- The premium rate for the second year -/
def premium_rate_year2 : ℚ := 4/100

/-- The premium rate for the third year -/
def premium_rate_year3 : ℚ := 5/100

/-- The total premium paid for all three years -/
def total_premium : ℝ := 950

/-- Theorem stating that the original value of the tempo satisfies the given conditions -/
theorem tempo_original_value :
  (coverage_percentage * premium_rate_year1 * original_value +
   coverage_percentage * premium_rate_year2 * original_value +
   coverage_percentage * premium_rate_year3 * original_value) = total_premium := by
  sorry

end NUMINAMATH_CALUDE_tempo_original_value_l965_96515
