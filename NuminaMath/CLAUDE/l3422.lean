import Mathlib

namespace NUMINAMATH_CALUDE_chord_length_theorem_l3422_342235

theorem chord_length_theorem (R AB BC : ℝ) (h_R : R = 12) (h_AB : AB = 6) (h_BC : BC = 4) :
  ∃ (AC : ℝ), (AC = Real.sqrt 35 + Real.sqrt 15) ∨ (AC = Real.sqrt 35 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l3422_342235


namespace NUMINAMATH_CALUDE_school_costume_problem_l3422_342275

/-- Represents the price of a costume set based on the quantity purchased -/
def price (n : ℕ) : ℕ :=
  if n ≤ 45 then 60
  else if n ≤ 90 then 50
  else 40

/-- The problem statement -/
theorem school_costume_problem :
  ∃ (a b : ℕ),
    a + b = 92 ∧
    a > b ∧
    a < 90 ∧
    a * price a + b * price b = 5020 ∧
    a = 50 ∧
    b = 42 ∧
    92 * 40 = a * price a + b * price b - 480 :=
by
  sorry


end NUMINAMATH_CALUDE_school_costume_problem_l3422_342275


namespace NUMINAMATH_CALUDE_child_tickets_sold_l3422_342232

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l3422_342232


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l3422_342230

/-- A regular polygon with exterior angles measuring 60° has 9 diagonals -/
theorem regular_polygon_diagonals :
  ∀ (n : ℕ),
  (360 / n = 60) →  -- Each exterior angle measures 60°
  (n * (n - 3)) / 2 = 9  -- Number of diagonals
  := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l3422_342230


namespace NUMINAMATH_CALUDE_function_properties_l3422_342250

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p ≠ 0 ∧ ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_shift : ∀ x, f (x + 1) = -f x)
    (h_incr : IsIncreasingOn f (-1) 0) :
    (IsPeriodic f 2) ∧ 
    (∀ x, f (2 - x) = f x) ∧
    (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3422_342250


namespace NUMINAMATH_CALUDE_remainder_theorem_l3422_342272

/-- The dividend polynomial -/
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x^2 - 1

/-- The proposed remainder -/
def R (x : ℝ) : ℝ := -2*x + 2

/-- Theorem stating that R is the remainder of P divided by D -/
theorem remainder_theorem : 
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = D x * Q x + R x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3422_342272


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3422_342251

theorem polynomial_expansion :
  ∀ t : ℝ, (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 
    6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3422_342251


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3422_342273

theorem discount_percentage_proof (couch_cost sectional_cost other_cost paid_amount : ℝ) 
  (h1 : couch_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000)
  (h4 : paid_amount = 7200) :
  let total_cost := couch_cost + sectional_cost + other_cost
  let discount := total_cost - paid_amount
  let discount_percentage := (discount / total_cost) * 100
  discount_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3422_342273


namespace NUMINAMATH_CALUDE_smallest_w_l3422_342233

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^5) (936 * w))
  (h2 : is_factor (3^3) (936 * w))
  (h3 : is_factor (13^2) (936 * w)) :
  w ≥ 156 ∧ ∃ w', w' = 156 ∧ w' > 0 ∧ 
    is_factor (2^5) (936 * w') ∧ 
    is_factor (3^3) (936 * w') ∧ 
    is_factor (13^2) (936 * w') :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l3422_342233


namespace NUMINAMATH_CALUDE_baseball_cost_calculation_l3422_342299

/-- The amount spent on marbles in dollars -/
def marbles_cost : ℚ := 9.05

/-- The amount spent on the football in dollars -/
def football_cost : ℚ := 4.95

/-- The total amount spent on toys in dollars -/
def total_cost : ℚ := 20.52

/-- The amount spent on the baseball in dollars -/
def baseball_cost : ℚ := total_cost - (marbles_cost + football_cost)

theorem baseball_cost_calculation :
  baseball_cost = 6.52 := by sorry

end NUMINAMATH_CALUDE_baseball_cost_calculation_l3422_342299


namespace NUMINAMATH_CALUDE_largest_three_digit_with_digit_product_8_l3422_342202

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem largest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → n ≤ 811 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_with_digit_product_8_l3422_342202


namespace NUMINAMATH_CALUDE_isosceles_triangles_remainder_l3422_342234

/-- The number of vertices in the regular polygon --/
def n : ℕ := 2019

/-- The number of isosceles triangles in a regular n-gon --/
def num_isosceles (n : ℕ) : ℕ := (n * (n - 1) / 2 : ℕ) - (2 * n / 3 : ℕ)

/-- The theorem stating that the remainder when the number of isosceles triangles
    in a regular 2019-gon is divided by 100 is equal to 25 --/
theorem isosceles_triangles_remainder :
  num_isosceles n % 100 = 25 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_remainder_l3422_342234


namespace NUMINAMATH_CALUDE_range_of_a_equiv_l3422_342296

/-- Proposition p: The equation x² + 2ax + 1 = 0 has two real roots greater than -1 -/
def prop_p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > -1 ∧ y > -1 ∧ x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0

/-- Proposition q: The solution set of the inequality ax² - ax + 1 > 0 with respect to x is ℝ -/
def prop_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The main theorem stating the equivalence of the conditions and the range of a -/
theorem range_of_a_equiv (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬prop_q a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_equiv_l3422_342296


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3422_342220

def p (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

theorem integer_roots_of_cubic :
  {x : ℤ | p x = 0} = {-1, -2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3422_342220


namespace NUMINAMATH_CALUDE_system_solution_l3422_342284

theorem system_solution (c₁ c₂ c₃ : ℝ) :
  let x₁ := -2 * c₁ - c₂ + 2
  let x₂ := c₁ + 1
  let x₃ := c₂ + 3
  let x₄ := 2 * c₂ + 2 * c₃ - 2
  let x₅ := c₃ + 1
  (x₁ + 2 * x₂ - x₃ + x₄ - 2 * x₅ = -3) ∧
  (x₁ + 2 * x₂ + 3 * x₃ - x₄ + 2 * x₅ = 17) ∧
  (2 * x₁ + 4 * x₂ + 2 * x₃ = 14) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3422_342284


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3422_342255

theorem inequality_solution_set (x : ℝ) :
  (3 * x^2 - 1 > 13 - 5 * x) ↔ (x < -7 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3422_342255


namespace NUMINAMATH_CALUDE_sons_age_l3422_342219

theorem sons_age (son_age man_age : ℕ) : 
  (man_age = son_age + 20) →
  (man_age + 2 = 2 * (son_age + 2)) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3422_342219


namespace NUMINAMATH_CALUDE_hexagon_ring_area_l3422_342205

/-- The area of the ring between the inscribed and circumscribed circles of a regular hexagon -/
theorem hexagon_ring_area (a : ℝ) (h : a > 0) : 
  let r_inscribed := (Real.sqrt 3 / 2) * a
  let r_circumscribed := a
  let area_inscribed := π * r_inscribed ^ 2
  let area_circumscribed := π * r_circumscribed ^ 2
  area_circumscribed - area_inscribed = π * a^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ring_area_l3422_342205


namespace NUMINAMATH_CALUDE_three_number_sum_l3422_342267

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by sorry

end NUMINAMATH_CALUDE_three_number_sum_l3422_342267


namespace NUMINAMATH_CALUDE_hardwood_flooring_area_l3422_342278

/-- Represents the dimensions of a rectangular area -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular area -/
def area (r : RectangularArea) : ℝ := r.length * r.width

/-- Represents Nancy's bathroom -/
structure Bathroom where
  centralArea : RectangularArea
  hallway : RectangularArea

/-- The actual bathroom dimensions -/
def nancysBathroom : Bathroom :=
  { centralArea := { length := 10, width := 10 }
  , hallway := { length := 6, width := 4 } }

/-- Theorem: The total area of hardwood flooring in Nancy's bathroom is 124 square feet -/
theorem hardwood_flooring_area :
  area nancysBathroom.centralArea + area nancysBathroom.hallway = 124 := by
  sorry

end NUMINAMATH_CALUDE_hardwood_flooring_area_l3422_342278


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3422_342203

theorem wire_ratio_proof (total_length : ℝ) (short_length : ℝ) :
  total_length = 70 →
  short_length = 20 →
  short_length / (total_length - short_length) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3422_342203


namespace NUMINAMATH_CALUDE_planes_and_perpendicular_lines_l3422_342231

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_and_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  parallel α β → 
  perpendicular n α → 
  perpendicular m β → 
  line_parallel m n :=
by sorry

end NUMINAMATH_CALUDE_planes_and_perpendicular_lines_l3422_342231


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_137_l3422_342215

theorem first_nonzero_digit_of_1_over_137 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 137 = n / 10^(n.succ) + r ∧ 0 < r ∧ r < 1 / 10^n ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_137_l3422_342215


namespace NUMINAMATH_CALUDE_bc_is_one_sixth_of_ad_l3422_342206

/-- Given a line segment AD with points E and B on it, prove that BC is 1/6 of AD -/
theorem bc_is_one_sixth_of_ad (A B C D E : ℝ) : 
  A < E ∧ E < D ∧   -- E is on AD
  A < B ∧ B < D ∧   -- B is on AD
  E - A = 3 * (D - E) ∧   -- AE is 3 times ED
  B - A = 5 * (D - B) ∧   -- AB is 5 times BD
  C = (B + E) / 2   -- C is midpoint of BE
  → 
  (C - B) / (D - A) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_bc_is_one_sixth_of_ad_l3422_342206


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l3422_342204

/-- The number of hotdogs Matthew needs to cook for his children -/
def total_hotdogs : ℕ :=
  let ella_emma_hotdogs := 2 + 2
  let luke_hotdogs := 2 * ella_emma_hotdogs
  let hunter_hotdogs := (3 * ella_emma_hotdogs) / 2
  ella_emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l3422_342204


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3422_342229

/-- A color representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A point with integer coordinates on a plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring of points on a plane -/
def Coloring := Point → Color

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

theorem monochromatic_unit_area_triangle_exists (c : Coloring) :
  ∃ (p1 p2 p3 : Point), c p1 = c p2 ∧ c p2 = c p3 ∧ triangleArea p1 p2 p3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3422_342229


namespace NUMINAMATH_CALUDE_power_sum_equality_l3422_342256

theorem power_sum_equality : (-1 : ℤ) ^ (6^2) + (1 : ℤ) ^ (3^3) = 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3422_342256


namespace NUMINAMATH_CALUDE_stating_circle_symmetry_l3422_342277

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y + 5 = 0

/-- Symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 7)^2 + (y + 1)^2 = 1

/-- 
Theorem stating that the symmetric_circle is indeed symmetric to the given_circle
with respect to the symmetry_line
-/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  given_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  (∃ (x_mid y_mid : ℝ),
    symmetry_line x_mid y_mid ∧
    x_mid = (x₁ + x₂) / 2 ∧
    y_mid = (y₁ + y₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_stating_circle_symmetry_l3422_342277


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3422_342213

/-- If a point P(a-1, a^2-9) lies on the y-axis, then its coordinates are (0, -8). -/
theorem point_on_y_axis (a : ℝ) :
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3422_342213


namespace NUMINAMATH_CALUDE_P_on_y_axis_after_move_l3422_342201

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of P -/
def P : Point := ⟨3, 4⟩

/-- Function to move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Predicate to check if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating that P lands on the y-axis after moving 3 units left -/
theorem P_on_y_axis_after_move : isOnYAxis (moveLeft P 3) := by
  sorry

end NUMINAMATH_CALUDE_P_on_y_axis_after_move_l3422_342201


namespace NUMINAMATH_CALUDE_archie_marbles_problem_l3422_342288

/-- The number of marbles Archie started with. -/
def initial_marbles : ℕ := 100

/-- The fraction of marbles Archie keeps after losing some in the street. -/
def street_loss_fraction : ℚ := 2/5

/-- The fraction of remaining marbles Archie keeps after losing some in the sewer. -/
def sewer_loss_fraction : ℚ := 1/2

/-- The number of marbles Archie has left at the end. -/
def final_marbles : ℕ := 20

theorem archie_marbles_problem :
  (↑final_marbles : ℚ) = ↑initial_marbles * street_loss_fraction * sewer_loss_fraction :=
by sorry

end NUMINAMATH_CALUDE_archie_marbles_problem_l3422_342288


namespace NUMINAMATH_CALUDE_exists_acute_triangle_with_large_intersection_area_l3422_342241

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle. -/
def area (t : Triangle) : ℝ := sorry

/-- A point is the median of a triangle if it is the midpoint of a side. -/
def is_median (M : Point) (t : Triangle) : Prop := sorry

/-- A point is on the angle bisector if it is equidistant from the two sides forming the angle. -/
def is_angle_bisector (K : Point) (t : Triangle) : Prop := sorry

/-- A point is on the altitude if it forms a right angle with the base of the triangle. -/
def is_altitude (H : Point) (t : Triangle) : Prop := sorry

/-- A triangle is acute if all its angles are less than 90 degrees. -/
def is_acute (t : Triangle) : Prop := sorry

/-- The area of the triangle formed by the intersection points of the median, angle bisector, and altitude. -/
def area_intersection (t : Triangle) (M K H : Point) : ℝ := sorry

/-- There exists an acute triangle where the area of the triangle formed by the intersection points
    of its median, angle bisector, and altitude is greater than 0.499 times the area of the original triangle. -/
theorem exists_acute_triangle_with_large_intersection_area :
  ∃ (t : Triangle) (M K H : Point),
    is_acute t ∧
    is_median M t ∧
    is_angle_bisector K t ∧
    is_altitude H t ∧
    area_intersection t M K H > 0.499 * area t :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_with_large_intersection_area_l3422_342241


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3422_342287

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 4) : ∑' n, a / (a + b)^n = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3422_342287


namespace NUMINAMATH_CALUDE_cricket_average_increase_l3422_342247

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the increase in average runs per innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let oldAverage : ℚ := player.totalRuns / player.innings
  let newTotal : ℕ := player.totalRuns + player.newInningsRuns
  let newAverage : ℚ := newTotal / (player.innings + 1)
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given scenario -/
theorem cricket_average_increase :
  ∀ (player : CricketPlayer),
  player.innings = 10 →
  player.totalRuns = 370 →
  player.newInningsRuns = 81 →
  averageIncrease player = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_increase_l3422_342247


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3422_342246

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Real.sqrt 2 + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3422_342246


namespace NUMINAMATH_CALUDE_distance_between_points_l3422_342211

theorem distance_between_points (a b c : ℝ) : 
  Real.sqrt ((a - (a + 3))^2 + (b - (b + 7))^2 + (c - (c + 1))^2) = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3422_342211


namespace NUMINAMATH_CALUDE_special_quadratic_property_l3422_342216

/-- A quadratic function f(x) = x^2 + ax + b satisfying specific conditions -/
def special_quadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- Theorem: If f(f(0)) = f(f(1)) = 0 and f(0) ≠ f(1), then f(2) = 3 -/
theorem special_quadratic_property (a b : ℝ) :
  let f := special_quadratic a b
  (f (f 0) = 0) → (f (f 1) = 0) → (f 0 ≠ f 1) → (f 2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_property_l3422_342216


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l3422_342249

/-- A function that checks if a natural number n satisfies the condition for forming an equilateral triangle with sticks of lengths 1 to n -/
def canFormEquilateralTriangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers -/
def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient condition for forming an equilateral triangle -/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = sumFirstN n ∧ a = b ∧ b = c) ↔ canFormEquilateralTriangle n :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l3422_342249


namespace NUMINAMATH_CALUDE_center_coordinates_sum_l3422_342266

/-- The sum of the coordinates of the center of the circle given by x^2 + y^2 = 6x - 8y + 18 is -1 -/
theorem center_coordinates_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 8*y + 18) → (∃ a b : ℝ, (x - a)^2 + (y - b)^2 = (x^2 + y^2 - 6*x + 8*y - 18) ∧ a + b = -1) :=
by sorry

end NUMINAMATH_CALUDE_center_coordinates_sum_l3422_342266


namespace NUMINAMATH_CALUDE_jack_head_circumference_l3422_342264

theorem jack_head_circumference :
  ∀ (J C B : ℝ),
  C = J / 2 + 9 →
  B = 2 / 3 * C →
  B = 10 →
  J = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_head_circumference_l3422_342264


namespace NUMINAMATH_CALUDE_inequality_proof_l3422_342253

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3422_342253


namespace NUMINAMATH_CALUDE_M_simplification_M_specific_value_l3422_342281

/-- Given expressions for A and B -/
def A (x y : ℝ) : ℝ := x^2 - 3*x*y - y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y - 3*y^2

/-- The expression M defined as 2A - B -/
def M (x y : ℝ) : ℝ := 2 * A x y - B x y

/-- Theorem stating that M simplifies to x^2 - 3xy + y^2 -/
theorem M_simplification (x y : ℝ) : M x y = x^2 - 3*x*y + y^2 := by
  sorry

/-- Theorem stating that M equals 11 when x = -2 and y = 1 -/
theorem M_specific_value : M (-2) 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_M_simplification_M_specific_value_l3422_342281


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3422_342274

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance_miles (blocks_east : ℕ) (blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Theorem stating that walking 6 blocks east and 12 blocks north, with each block being one-third of a mile, results in a total distance of 6 miles. -/
theorem arthur_walk_distance :
  total_distance_miles 6 12 (1/3) = 6 := by sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l3422_342274


namespace NUMINAMATH_CALUDE_uncommon_card_cost_is_half_dollar_l3422_342207

/-- The cost of an uncommon card in Tom's deck -/
def uncommon_card_cost : ℚ :=
  let rare_cards : ℕ := 19
  let uncommon_cards : ℕ := 11
  let common_cards : ℕ := 30
  let rare_card_cost : ℚ := 1
  let common_card_cost : ℚ := 1/4
  let total_deck_cost : ℚ := 32
  (total_deck_cost - rare_cards * rare_card_cost - common_cards * common_card_cost) / uncommon_cards

theorem uncommon_card_cost_is_half_dollar : uncommon_card_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_card_cost_is_half_dollar_l3422_342207


namespace NUMINAMATH_CALUDE_fruit_tree_ratio_l3422_342217

theorem fruit_tree_ratio (total_streets : ℕ) (plum_trees pear_trees apricot_trees : ℕ) : 
  total_streets = 18 →
  plum_trees = 3 →
  pear_trees = 3 →
  apricot_trees = 3 →
  (plum_trees + pear_trees + apricot_trees : ℚ) / total_streets = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_ratio_l3422_342217


namespace NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_four_l3422_342254

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - 2*c, b + 2*d)

-- Theorem statement
theorem identical_pairs_imply_x_equals_four :
  ∀ x y : ℤ, star 2 (-4) 1 (-3) = star x y 2 1 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_four_l3422_342254


namespace NUMINAMATH_CALUDE_calculation_proof_l3422_342200

theorem calculation_proof : 
  47 * ((4 + 3/7) - (5 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 119/171) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3422_342200


namespace NUMINAMATH_CALUDE_f_property_implies_n_times_s_eq_14_l3422_342214

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the main property of f
axiom f_property (x y z : ℝ) : f (x^2 + y * f z) = x * f x + z * f y + y^2

-- Define n as the number of possible values of f(5)
def n : ℕ := sorry

-- Define s as the sum of all possible values of f(5)
def s : ℝ := sorry

-- State the theorem to be proved
theorem f_property_implies_n_times_s_eq_14 : n * s = 14 := by sorry

end NUMINAMATH_CALUDE_f_property_implies_n_times_s_eq_14_l3422_342214


namespace NUMINAMATH_CALUDE_max_probability_dice_difference_l3422_342293

def roll_dice : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

def difference (roll : ℕ × ℕ) : ℤ := (roll.1 : ℤ) - (roll.2 : ℤ)

def target_differences : Finset ℤ := {-2, -1, 0, 1, 2}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  roll_dice.filter (λ roll => difference roll ∈ target_differences)

theorem max_probability_dice_difference :
  (favorable_outcomes.card : ℚ) / roll_dice.card = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_max_probability_dice_difference_l3422_342293


namespace NUMINAMATH_CALUDE_divisibility_of_power_minus_one_l3422_342242

theorem divisibility_of_power_minus_one (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n ^ (n - 1) : ℤ) - 1 = k * ((n - 1) ^ 2 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_minus_one_l3422_342242


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l3422_342289

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ y = -3/4 * x

-- Define the point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

theorem circle_and_line_theorem :
  -- Circle C passes through point A
  circle_C point_A.1 point_A.2 →
  -- Circle C is tangent to the line x+y=1
  ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y →
  -- The center of the circle lies on the line y=-2x
  ∃ (x y : ℝ), circle_C x y ∧ center_line x y →
  -- Line l passes through the origin
  ∃ (x y : ℝ), line_l x y ∧ (x, y) = origin →
  -- The chord intercepted by circle C on line l has a length of 2
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 →
  -- Conclusion: The equations of circle C and line l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧
  (∀ (x y : ℝ), line_l x y ↔ (x = 0 ∨ y = -3/4 * x)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_theorem_l3422_342289


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l3422_342282

theorem ball_ratio_proof (a b x : ℕ) : 
  (a / (a + b + x) = 1/4) →
  ((a + x) / (b + x) = 2/3) →
  (3*a - b = x) →
  (2*b - 3*a = x) →
  (a / b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l3422_342282


namespace NUMINAMATH_CALUDE_zombies_less_than_threshold_days_l3422_342252

/-- The number of zombies in the mall today -/
def current_zombies : ℕ := 480

/-- The threshold number of zombies -/
def threshold : ℕ := 50

/-- The function that calculates the number of zombies n days ago -/
def zombies_n_days_ago (n : ℕ) : ℚ :=
  current_zombies / (2 ^ n : ℚ)

/-- The theorem stating that 4 days ago is when there were less than 50 zombies -/
theorem zombies_less_than_threshold_days : 
  (∃ (n : ℕ), zombies_n_days_ago n < threshold) ∧ 
  (∀ (m : ℕ), m < 4 → zombies_n_days_ago m ≥ threshold) ∧
  zombies_n_days_ago 4 < threshold :=
sorry

end NUMINAMATH_CALUDE_zombies_less_than_threshold_days_l3422_342252


namespace NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l3422_342248

theorem polynomial_factors_imply_absolute_value (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (3 * x^3 - h * x + k)) →
  |3 * h - 2 * k| = 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l3422_342248


namespace NUMINAMATH_CALUDE_total_garden_area_l3422_342294

-- Define the garden dimensions and counts for each person
def mancino_gardens : ℕ := 4
def mancino_length : ℕ := 16
def mancino_width : ℕ := 5

def marquita_gardens : ℕ := 3
def marquita_length : ℕ := 8
def marquita_width : ℕ := 4

def matteo_gardens : ℕ := 2
def matteo_length : ℕ := 12
def matteo_width : ℕ := 6

def martina_gardens : ℕ := 5
def martina_length : ℕ := 10
def martina_width : ℕ := 3

-- Theorem stating the total square footage of all gardens
theorem total_garden_area :
  mancino_gardens * mancino_length * mancino_width +
  marquita_gardens * marquita_length * marquita_width +
  matteo_gardens * matteo_length * matteo_width +
  martina_gardens * martina_length * martina_width = 710 := by
  sorry

end NUMINAMATH_CALUDE_total_garden_area_l3422_342294


namespace NUMINAMATH_CALUDE_intersection_points_roots_l3422_342244

theorem intersection_points_roots (x y : ℝ) : 
  (∃ x, x^2 - 3*x = 0 ∧ x ≠ 0 ∧ x ≠ 3) ∨
  (∀ x, x = x - 3 → x^2 - 3*x ≠ 0) :=
by sorry

#check intersection_points_roots

end NUMINAMATH_CALUDE_intersection_points_roots_l3422_342244


namespace NUMINAMATH_CALUDE_integer_solution_less_than_one_l3422_342240

theorem integer_solution_less_than_one :
  ∃ (x : ℤ), x - 1 < 0 :=
by
  use 0
  sorry

end NUMINAMATH_CALUDE_integer_solution_less_than_one_l3422_342240


namespace NUMINAMATH_CALUDE_line_length_difference_l3422_342236

theorem line_length_difference (white_line blue_line : ℝ) 
  (h1 : white_line = 7.67) 
  (h2 : blue_line = 3.33) : 
  white_line - blue_line = 4.34 := by
sorry

end NUMINAMATH_CALUDE_line_length_difference_l3422_342236


namespace NUMINAMATH_CALUDE_area_of_ABCD_l3422_342209

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The problem statement -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) : 
  r1.area + r2.area + r3.area = 8 ∧ r1.area = 2 → 
  ∃ (ABCD : Rectangle), ABCD.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABCD_l3422_342209


namespace NUMINAMATH_CALUDE_solution_range_l3422_342261

theorem solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1 ∧ 2*x + y < m^2 - 8*m) ↔ 
  (m < -1 ∨ m > 9) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l3422_342261


namespace NUMINAMATH_CALUDE_type_T_machine_time_l3422_342245

-- Define the time for a type B machine to complete the job
def time_B : ℝ := 7

-- Define the time for 2 type T machines and 3 type B machines to complete the job together
def time_combined : ℝ := 1.2068965517241381

-- Define the time for a type T machine to complete the job
def time_T : ℝ := 5

-- Theorem statement
theorem type_T_machine_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_T - (1 / ((1 / time_combined) - (3 / (2 * time_B))))| < ε :=
sorry

end NUMINAMATH_CALUDE_type_T_machine_time_l3422_342245


namespace NUMINAMATH_CALUDE_tan_70_cos_10_identity_l3422_342212

theorem tan_70_cos_10_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_identity_l3422_342212


namespace NUMINAMATH_CALUDE_travelers_checks_denomination_l3422_342283

theorem travelers_checks_denomination 
  (total_checks : ℕ) 
  (total_worth : ℚ) 
  (spent_checks : ℕ) 
  (remaining_average : ℚ) 
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : spent_checks = 6)
  (h4 : remaining_average = 62.5)
  (h5 : (total_checks - spent_checks : ℚ) * remaining_average + spent_checks * x = total_worth) :
  x = 50 := by
  sorry

end NUMINAMATH_CALUDE_travelers_checks_denomination_l3422_342283


namespace NUMINAMATH_CALUDE_digital_earth_not_equal_gis_l3422_342271

-- Define the concept of Digital Earth
def DigitalEarth : Type := Unit

-- Define Geographic Information Technology
def GeographicInformationTechnology : Type := Unit

-- Define other related technologies
def RemoteSensing : Type := Unit
def GPS : Type := Unit
def VirtualTechnology : Type := Unit
def NetworkTechnology : Type := Unit

-- Define the correct properties of Digital Earth
axiom digital_earth_properties : 
  DigitalEarth → 
  (GeographicInformationTechnology × VirtualTechnology × NetworkTechnology)

-- Define the incorrect statement
def incorrect_statement : Prop :=
  DigitalEarth = GeographicInformationTechnology

-- Theorem to prove
theorem digital_earth_not_equal_gis : ¬incorrect_statement :=
sorry

end NUMINAMATH_CALUDE_digital_earth_not_equal_gis_l3422_342271


namespace NUMINAMATH_CALUDE_unique_paintable_number_l3422_342290

def isPaintable (s b a : ℕ+) : Prop :=
  -- Sarah's sequence doesn't overlap with Bob's or Alice's
  ∀ k l : ℕ, k * s.val ≠ l * b.val ∧ k * s.val ≠ 4 + l * a.val
  -- Bob's sequence doesn't overlap with Sarah's or Alice's
  ∧ ∀ k l : ℕ, 2 + k * b.val ≠ l * s.val ∧ 2 + k * b.val ≠ 4 + l * a.val
  -- Alice's sequence doesn't overlap with Sarah's or Bob's
  ∧ ∀ k l : ℕ, 4 + k * a.val ≠ l * s.val ∧ 4 + k * a.val ≠ 2 + l * b.val
  -- Every picket is painted
  ∧ ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k * s.val ∨ n = 2 + k * b.val ∨ n = 4 + k * a.val)

theorem unique_paintable_number :
  ∃! n : ℕ, ∃ s b a : ℕ+, isPaintable s b a ∧ n = 1000 * s.val + 100 * b.val + 10 * a.val :=
by sorry

end NUMINAMATH_CALUDE_unique_paintable_number_l3422_342290


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3422_342239

theorem binary_digit_difference : ∃ (n m : ℕ), n = 400 ∧ m = 1600 ∧ 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l3422_342239


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3422_342210

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3422_342210


namespace NUMINAMATH_CALUDE_triangle_area_l3422_342223

def a : ℝ × ℝ := (4, -3)
def b : ℝ × ℝ := (-6, 5)
def c : ℝ × ℝ := (-12, 10)

theorem triangle_area : 
  let det := a.1 * c.2 - a.2 * c.1
  (1/2 : ℝ) * |det| = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3422_342223


namespace NUMINAMATH_CALUDE_ball_drawing_problem_l3422_342226

-- Define the sample space
def Ω : Type := Fin 4

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def A : Set Ω := sorry -- Both balls are the same color
def B : Set Ω := sorry -- Both balls are different colors
def C : Set Ω := sorry -- The first ball drawn is red
def D : Set Ω := sorry -- The second ball drawn is red

-- State the theorem
theorem ball_drawing_problem :
  (P (A ∩ B) = 0) ∧
  (P (A ∩ C) = P A * P C) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_problem_l3422_342226


namespace NUMINAMATH_CALUDE_max_angle_point_is_tangency_point_l3422_342222

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A structure representing a circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Function to calculate the angle between three points -/
def angle (A B M : Point) : ℝ := sorry

/-- Function to check if a point is on a line -/
def pointOnLine (P : Point) (l : Line) : Prop := sorry

/-- Function to check if a line intersects a segment -/
def lineIntersectsSegment (l : Line) (A B : Point) : Prop := sorry

/-- Function to check if a circle passes through two points -/
def circlePassesThroughPoints (C : Circle) (A B : Point) : Prop := sorry

/-- Function to check if a circle is tangent to a line -/
def circleTangentToLine (C : Circle) (l : Line) : Prop := sorry

/-- Theorem stating that the point M on line (d) that maximizes the angle ∠AMB
    is the point of tangency of the smallest circumcircle passing through A and B
    with the line (d) -/
theorem max_angle_point_is_tangency_point
  (A B : Point) (d : Line) 
  (h : ¬ lineIntersectsSegment d A B) :
  ∃ (M : Point) (C : Circle),
    pointOnLine M d ∧
    circlePassesThroughPoints C A B ∧
    circleTangentToLine C d ∧
    (∀ (M' : Point), pointOnLine M' d → angle A M' B ≤ angle A M B) :=
sorry

end NUMINAMATH_CALUDE_max_angle_point_is_tangency_point_l3422_342222


namespace NUMINAMATH_CALUDE_treasure_chest_problem_l3422_342291

theorem treasure_chest_problem (n : ℕ) : 
  (n > 0 ∧ n % 8 = 6 ∧ n % 9 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 9 = 5 → n ≤ m) → 
  (n = 14 ∧ n % 11 = 3) := by
sorry

end NUMINAMATH_CALUDE_treasure_chest_problem_l3422_342291


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3422_342280

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 6*x

-- Define the general form of a quadratic expression
def general_form (a h k x : ℝ) : ℝ := a*(x - h)^2 + k

-- Theorem statement
theorem quadratic_completion_of_square :
  ∃ (a h k : ℝ), ∀ x, quadratic_expr x = general_form a h k x → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3422_342280


namespace NUMINAMATH_CALUDE_two_tails_probability_l3422_342285

theorem two_tails_probability (n : ℕ) (h : n = 5) : 
  (Nat.choose n 2 : ℚ) / (2^n : ℚ) = 10/32 := by
  sorry

end NUMINAMATH_CALUDE_two_tails_probability_l3422_342285


namespace NUMINAMATH_CALUDE_game_prep_time_calculation_l3422_342224

/-- Calculates the total time before playing the main game --/
def totalGamePrepTime (downloadTime installTime updateTime accountTime issuesTime tutorialTime : ℕ) : ℕ :=
  downloadTime + installTime + updateTime + accountTime + issuesTime + tutorialTime

theorem game_prep_time_calculation :
  let downloadTime : ℕ := 10
  let installTime : ℕ := downloadTime / 2
  let updateTime : ℕ := downloadTime * 2
  let accountTime : ℕ := 5
  let issuesTime : ℕ := 15
  let preGameTime : ℕ := downloadTime + installTime + updateTime + accountTime + issuesTime
  let tutorialTime : ℕ := preGameTime * 3
  totalGamePrepTime downloadTime installTime updateTime accountTime issuesTime tutorialTime = 220 := by
  sorry

#eval totalGamePrepTime 10 5 20 5 15 165

end NUMINAMATH_CALUDE_game_prep_time_calculation_l3422_342224


namespace NUMINAMATH_CALUDE_softball_team_size_l3422_342295

/-- Proves that a co-ed softball team with given conditions has 20 total players -/
theorem softball_team_size : ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 2/3 →
  men + women = 20 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l3422_342295


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3422_342270

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | (x^2 - y^2)^2 = 16*y + 1} =
  {(1, 0), (-1, 0), (4, 3), (-4, 3), (4, 5), (-4, 5)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3422_342270


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3422_342238

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    (m + n ≤ p + q) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3422_342238


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3422_342262

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧ 
  Real.tan (150 * π / 180 - x * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3422_342262


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3422_342243

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (k, -1),
    if a is perpendicular to b, then k = 3/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![k, -1]
  (∀ i, i < 2 → a i * b i = 0) →
  k = 3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3422_342243


namespace NUMINAMATH_CALUDE_simplify_expression_l3422_342257

theorem simplify_expression : 1 + 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3422_342257


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l3422_342227

/-- The area of a right triangle with legs of length 36 and 48 is 864 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun leg1 leg2 area =>
    leg1 = 36 ∧ leg2 = 48 → area = (1 / 2) * leg1 * leg2 → area = 864

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 36 48 864 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l3422_342227


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3422_342228

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, x^2 + 1 > 1) ↔ ¬(∃ x₀ : ℝ, x₀^2 + 1 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3422_342228


namespace NUMINAMATH_CALUDE_min_value_of_f_l3422_342269

/-- The quadratic function f(x) = x^2 + 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 8

/-- The minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3422_342269


namespace NUMINAMATH_CALUDE_sqrt_square_fourteen_l3422_342259

theorem sqrt_square_fourteen : Real.sqrt (14^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_fourteen_l3422_342259


namespace NUMINAMATH_CALUDE_number_accuracy_l3422_342237

-- Define a function to represent the accuracy of a number
def accuracy_place (n : ℝ) : ℕ :=
  sorry

-- Define the number in scientific notation
def number : ℝ := 2.3 * (10 ^ 4)

-- Theorem stating that the number is accurate to the thousands place
theorem number_accuracy :
  accuracy_place number = 3 :=
sorry

end NUMINAMATH_CALUDE_number_accuracy_l3422_342237


namespace NUMINAMATH_CALUDE_min_value_problem_l3422_342279

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3422_342279


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3422_342260

/-- Given that 3 + √5 is a root of x³ + cx² + dx + 15 = 0 where c and d are rational,
    prove that d = -18.5 -/
theorem cubic_equation_root (c d : ℚ) 
  (h : (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0) :
  d = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3422_342260


namespace NUMINAMATH_CALUDE_solve_for_y_l3422_342263

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3422_342263


namespace NUMINAMATH_CALUDE_min_value_ab_l3422_342297

theorem min_value_ab (b a : ℝ) (h1 : b > 0)
  (h2 : (b^2 + 1) * (-1 / a) = 1 / b^2) : 
  ∀ x : ℝ, a * b ≥ 2 ∧ (a * b = 2 ↔ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l3422_342297


namespace NUMINAMATH_CALUDE_nitin_rank_last_l3422_342208

def class_size : ℕ := 58
def nitin_rank_start : ℕ := 24

theorem nitin_rank_last : class_size - nitin_rank_start + 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_nitin_rank_last_l3422_342208


namespace NUMINAMATH_CALUDE_sarah_candy_theorem_l3422_342218

/-- The number of candy pieces Sarah received from her neighbors -/
def candy_from_neighbors : ℕ := sorry

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_lasted : ℕ := 9

/-- The total number of candy pieces Sarah had -/
def total_candy : ℕ := candy_per_day * days_lasted

theorem sarah_candy_theorem : candy_from_neighbors = 66 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_theorem_l3422_342218


namespace NUMINAMATH_CALUDE_emmanuel_december_charges_l3422_342298

/-- Emmanuel's total charges for December -/
def total_charges (regular_plan_cost : ℝ) (days_in_guam : ℕ) (international_data_cost : ℝ) : ℝ :=
  regular_plan_cost + (days_in_guam : ℝ) * international_data_cost

/-- Theorem: Emmanuel's total charges for December are $210 -/
theorem emmanuel_december_charges :
  total_charges 175 10 3.5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_emmanuel_december_charges_l3422_342298


namespace NUMINAMATH_CALUDE_vector_magnitude_l3422_342276

def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = k • w

theorem vector_magnitude (m : ℝ) :
  parallel (2 • (a m) + b m) (b m) → ‖a m‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3422_342276


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3422_342258

/-- The vertex coordinates of the parabola y = -2x^2 + 8x - 3 are (2, 5) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -2 * x^2 + 8 * x - 3
  ∃ (x y : ℝ), (x, y) = (2, 5) ∧ 
    (∀ t : ℝ, f t ≤ f x) ∧
    y = f x :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3422_342258


namespace NUMINAMATH_CALUDE_area_invariant_under_opposite_vertex_translation_l3422_342292

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Moves a point by a given vector -/
def movePoint (p : Point2D) (v : Vector2D) : Point2D :=
  { x := p.x + v.x, y := p.y + v.y }

/-- Theorem: The area of a quadrilateral remains unchanged when two opposite vertices
    are moved by the same vector -/
theorem area_invariant_under_opposite_vertex_translation (q : Quadrilateral) (v : Vector2D) :
  let q' := { q with
    A := movePoint q.A v,
    C := movePoint q.C v
  }
  area q = area q' :=
sorry

end NUMINAMATH_CALUDE_area_invariant_under_opposite_vertex_translation_l3422_342292


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l3422_342265

theorem min_n_for_constant_term (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), (n.choose k) * (-1)^k * x^(n - 8*k) = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬(∃ (k : ℕ), (m.choose k) * (-1)^k * x^(m - 8*k) = 1)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l3422_342265


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3422_342221

theorem cubic_equation_roots (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3422_342221


namespace NUMINAMATH_CALUDE_lemonade_stand_lemons_cost_l3422_342268

/-- Proves that the amount spent on lemons is $10 given the lemonade stand conditions --/
theorem lemonade_stand_lemons_cost (sugar_cost cups_cost : ℕ) 
  (cups_sold price_per_cup : ℕ) (profit : ℕ) :
  sugar_cost = 5 →
  cups_cost = 3 →
  cups_sold = 21 →
  price_per_cup = 4 →
  profit = 66 →
  ∃ (lemons_cost : ℕ),
    lemons_cost = 10 ∧
    profit = cups_sold * price_per_cup - (lemons_cost + sugar_cost + cups_cost) :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_lemons_cost_l3422_342268


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3422_342286

theorem sum_of_angles_two_triangles (A B C D E F : ℝ) :
  (A + B + C = 180) → (D + E + F = 180) → (A + B + C + D + E + F = 360) := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3422_342286


namespace NUMINAMATH_CALUDE_log_function_unique_parameters_l3422_342225

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+b)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := log_base a (x + b)

-- State the theorem
theorem log_function_unique_parameters :
  ∀ a b : ℝ, a > 0 → a ≠ 1 →
  (f a b (-1) = 0 ∧ f a b 0 = 1) →
  (a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_log_function_unique_parameters_l3422_342225
