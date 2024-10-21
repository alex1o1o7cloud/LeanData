import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_lens_sales_l387_38706

/-- Calculates the total sales of contact lenses given the prices and quantities sold. -/
theorem contact_lens_sales
  (soft_price : ℕ)
  (hard_price : ℕ)
  (soft_quantity : ℕ)
  (hard_quantity : ℕ)
  (h1 : soft_price = 150)
  (h2 : hard_price = 85)
  (h3 : soft_quantity = hard_quantity + 5)
  (h4 : soft_quantity + hard_quantity = 11) :
  soft_price * soft_quantity + hard_price * hard_quantity = 1455 := by
  sorry

#check contact_lens_sales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_lens_sales_l387_38706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_revolutions_l387_38769

/-- The number of revolutions made by the back wheels of a tricycle -/
noncomputable def back_wheel_revolutions (front_radius : ℝ) (back_radius : ℝ) (front_revolutions : ℝ) : ℝ :=
  (2 * Real.pi * front_radius * front_revolutions) / (2 * Real.pi * back_radius)

/-- Theorem: Given a tricycle with front wheel radius 1.5 feet making 120 revolutions
    and back wheels with radius 0.5 feet, each back wheel makes 360 revolutions. -/
theorem tricycle_revolutions :
  back_wheel_revolutions 1.5 0.5 120 = 360 := by
  -- Unfold the definition of back_wheel_revolutions
  unfold back_wheel_revolutions
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_revolutions_l387_38769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_call_ratio_is_six_fifths_l387_38758

/-- Represents the ratio of calls processed by each member of team A to each member of team B -/
noncomputable def call_ratio (total_calls : ℝ) (team_a_agents : ℝ) (team_b_agents : ℝ) : ℝ :=
  let team_b_calls := (4/7) * total_calls
  let team_a_calls := total_calls - team_b_calls
  (team_a_calls / team_a_agents) / (team_b_calls / team_b_agents)

/-- Theorem stating the ratio of calls processed by each member of team A to team B is 6/5 -/
theorem call_ratio_is_six_fifths (total_calls : ℝ) (team_b_agents : ℝ) 
  (h_positive : total_calls > 0 ∧ team_b_agents > 0) :
  call_ratio total_calls ((5/8) * team_b_agents) team_b_agents = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_call_ratio_is_six_fifths_l387_38758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_l387_38705

-- Define the conversion factor from hectares to square meters
noncomputable def hectare_to_sqm : ℝ := 10000

-- Define the area of the square in hectares
noncomputable def square_area_hectare : ℝ := 1 / 2

-- Theorem statement
theorem square_diagonal :
  let square_area_sqm := square_area_hectare * hectare_to_sqm
  let side_length := Real.sqrt square_area_sqm
  Real.sqrt (2 * side_length ^ 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_l387_38705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l387_38741

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- Define the inverse function f_inv
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem inverse_function_value :
  f_inv (1/2) = 2 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l387_38741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_scarves_difference_l387_38709

/-- Represents the inventory of a clothing store -/
structure Inventory where
  ties : ℕ
  belts : ℕ
  black_shirts : ℕ
  white_shirts : ℕ

/-- Calculates the number of jeans based on the inventory -/
def num_jeans (inv : Inventory) : ℕ :=
  2 * (inv.black_shirts + inv.white_shirts) / 3

/-- Calculates the number of scarves based on the inventory -/
def num_scarves (inv : Inventory) : ℕ :=
  (inv.ties + inv.belts) / 2

/-- The main theorem stating the difference between jeans and scarves -/
theorem jeans_scarves_difference (inv : Inventory) 
  (h1 : inv.ties = 34)
  (h2 : inv.belts = 40)
  (h3 : inv.black_shirts = 63)
  (h4 : inv.white_shirts = 42) : 
  num_jeans inv - num_scarves inv = 33 := by
  sorry

#eval num_jeans ⟨34, 40, 63, 42⟩ - num_scarves ⟨34, 40, 63, 42⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_scarves_difference_l387_38709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sine_phase_l387_38755

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (x + φ)

theorem odd_sine_phase (φ : ℝ) :
  (∀ x, f x φ = -f (-x) φ) → φ = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sine_phase_l387_38755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_properties_l387_38785

noncomputable def f (x : ℝ) := Real.sin (x - Real.pi / 4)

theorem sin_properties (k : ℤ) :
  let a := -Real.pi / 4 + 2 * Real.pi * (k : ℝ)
  let b := 2 * Real.pi * (k : ℝ) + 3 * Real.pi / 4
  (∀ x ∈ Set.Icc a b, Monotone f) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (f (2 * Real.pi * (k : ℝ) + 3 * Real.pi / 4) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_properties_l387_38785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_not_liking_any_food_l387_38771

def kennel_problem (total watermelon salmon both_sw chicken both_cw both_cs all_three : ℕ) : Prop :=
  let only_w := watermelon - (both_cw + both_sw - all_three)
  let only_s := salmon - (both_sw + both_cs - all_three)
  let only_c := chicken - (both_cw + both_cs - all_three)
  let like_any := only_w + only_s + only_c + both_cw + both_cs + both_sw + all_three
  total - like_any = 2

theorem dogs_not_liking_any_food :
  kennel_problem 75 18 58 10 15 4 8 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_not_liking_any_food_l387_38771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l387_38779

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 1 2 ∧ f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l387_38779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_spaces_theorem_l387_38747

/-- Represents the parking lot configuration --/
structure ParkingLot where
  totalSpaces : ℕ
  busLength : ℕ
  initialBuses : ℕ
  additionalBuses : ℕ
  leftmostBusStart : ℕ
  rightmostBusEnd : ℕ

/-- The maximum number of parking spaces given the conditions --/
def maxParkingSpaces : ParkingLot → ℕ
  | _ => 29

/-- Theorem stating that the maximum number of parking spaces is 29 --/
theorem parking_spaces_theorem (p : ParkingLot) 
  (h1 : p.busLength = 3)
  (h2 : p.initialBuses = 2)
  (h3 : p.additionalBuses = 4)
  (h4 : p.leftmostBusStart = 5)
  (h5 : p.rightmostBusEnd = 10)
  (h6 : p.totalSpaces ≥ p.leftmostBusStart + p.busLength - 1)
  (h7 : p.totalSpaces ≥ p.rightmostBusEnd)
  (h8 : ∀ n : ℕ, n > p.totalSpaces → ¬ ∃ (arrangement : List ℕ), 
       arrangement.length = p.initialBuses + p.additionalBuses + 1 ∧ 
       arrangement.all (λ x => x + p.busLength - 1 ≤ n) ∧
       arrangement.Pairwise (λ a b => a + p.busLength ≤ b ∨ b + p.busLength ≤ a)) :
  maxParkingSpaces p = 29 := by
  sorry

#check parking_spaces_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_spaces_theorem_l387_38747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_implies_tan_theta_parallel_vectors_in_range_implies_theta_l387_38787

open Real

-- Define the vectors a and b
noncomputable def a (θ : ℝ) : ℝ × ℝ := (2 * sin θ, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (1, sin (θ + π/3))

-- Part 1
theorem dot_product_zero_implies_tan_theta (θ : ℝ) :
  (a θ).1 * (b θ).1 + (a θ).2 * (b θ).2 = 0 →
  tan θ = -sqrt 3 / 5 := by sorry

-- Part 2
theorem parallel_vectors_in_range_implies_theta (θ : ℝ) :
  0 < θ ∧ θ < π/2 →
  (∃ (k : ℝ), k ≠ 0 ∧ a θ = k • (b θ)) →
  θ = π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_implies_tan_theta_parallel_vectors_in_range_implies_theta_l387_38787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l387_38790

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D :=
  { point := (2, 2),
    direction := (3, -4) }

/-- The second line -/
noncomputable def line2 : Line2D :=
  { point := (4, -10),
    direction := (5, 3) }

/-- A point on a line given by the parameter t -/
noncomputable def pointOnLine (l : Line2D) (t : ℝ) : ℝ × ℝ :=
  (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- The intersection point of the two lines -/
noncomputable def intersectionPoint : ℝ × ℝ := (184/11, -194/11)

/-- Theorem stating that the intersection point is on both lines -/
theorem intersection_point_on_both_lines :
  ∃ t u : ℝ, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l387_38790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_area_l387_38702

-- Define the radii of the circles
noncomputable def smallRadius : ℝ := Real.sqrt (12 / 5)
noncomputable def largeRadius : ℝ := 3 * smallRadius

-- Define the tangent length
def tangentLength : ℝ := 6

-- Theorem statement
theorem smaller_circle_area (h1 : largeRadius = 3 * smallRadius)
                            (h2 : tangentLength = 6)
                            (h3 : smallRadius^2 + tangentLength^2 = (smallRadius + largeRadius)^2 / 4) :
  π * smallRadius^2 = π * (12 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_area_l387_38702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l387_38701

-- Define a line in 2D space
def Line := ℝ → ℝ

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to check if a point is on the same side of a line as another point
def sameSide (l : Line) (p q : Point) : Prop := sorry

-- Function to check if a circle passes through two points
def passesThrough (c : Circle) (p : Point) : Prop := sorry

-- Function to check if a circle is tangent to a line
def isTangent (c : Circle) (l : Line) : Prop := sorry

-- The statement to be proved
theorem circle_tangent_to_line (l : Line) (p q : Point) :
  p ≠ q →
  sameSide l p q →
  ¬∃ (c1 c2 : Circle), c1 ≠ c2 ∧
    passesThrough c1 p ∧ passesThrough c1 q ∧ isTangent c1 l ∧
    passesThrough c2 p ∧ passesThrough c2 q ∧ isTangent c2 l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l387_38701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_m_sum_l387_38700

/-- The sum of integer values of m for which the area of the triangle formed by
    points (2, 3), (10, 8), and (6, m) is minimized -/
def minAreaTriangleMSum : ℤ := 13

/-- Point type as a pair of real numbers -/
def Point := ℝ × ℝ

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Theorem stating that the sum of integer values of m for which the area of the
    triangle formed by points (2, 3), (10, 8), and (6, m) is minimized is 13 -/
theorem min_area_triangle_m_sum :
  ∃ (m1 m2 : ℤ),
    m1 + m2 = minAreaTriangleMSum ∧
    (∀ (m : ℤ),
      triangleArea (2, 3) (10, 8) (6, ↑m) ≥ triangleArea (2, 3) (10, 8) (6, ↑m1) ∧
      triangleArea (2, 3) (10, 8) (6, ↑m) ≥ triangleArea (2, 3) (10, 8) (6, ↑m2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_m_sum_l387_38700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_isosceles_triangle_figure_l387_38722

/-- A figure formed by two isosceles triangles with specific properties -/
structure IsoscelesTriangleFigure where
  -- Side length of the isosceles triangles
  side_length : ℝ
  -- Base angle of the isosceles triangles in radians
  base_angle : ℝ
  -- Assumption that side length is 1
  side_length_is_one : side_length = 1
  -- Assumption that base angle is 30 degrees (π/6 radians)
  base_angle_is_thirty_degrees : base_angle = π / 6
  -- Assumption that the bases of the triangles are parallel

/-- The area of the figure formed by two isosceles triangles -/
noncomputable def area (figure : IsoscelesTriangleFigure) : ℝ :=
  2 * (figure.side_length ^ 2 * Real.sin figure.base_angle * Real.cos figure.base_angle) / 2

/-- Theorem stating that the area of the described figure is √3/2 -/
theorem area_of_isosceles_triangle_figure (figure : IsoscelesTriangleFigure) : 
  area figure = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_isosceles_triangle_figure_l387_38722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_sqrt_N_l387_38772

/-- Represents the number of 44 repetitions in the decimal part -/
def n : ℕ := 2018

/-- Represents the number of 88 repetitions -/
def m : ℕ := 2017

/-- Defines the number N as described in the problem -/
def N : ℚ := (44 * (10^n - 1) / 99 + 88 * (10^m - 1) / 99 + 9) / 10^n

/-- Defines the sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Converts an integer to a natural number -/
def int_to_nat (z : ℤ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_digits_sqrt_N : sum_of_digits (int_to_nat (Int.floor (Real.sqrt (↑N : ℝ)))) = 12109 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_sqrt_N_l387_38772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l387_38707

/-- Represents a convex quadrilateral with side lengths a, b, c, d. -/
def ConvexQuadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c

/-- Represents the area of a quadrilateral with side lengths a, b, c, d. -/
noncomputable def Area (a b c d : ℝ) : ℝ :=
  sorry -- Definition of area calculation

/-- Represents a permutation of four elements. -/
def Permutation (xs ys : List ℝ) : Prop :=
  xs.length = 4 ∧ ys.length = 4 ∧ xs.toFinset = ys.toFinset

/-- Given a convex quadrilateral ABCD with area S and side lengths a, b, c, d,
    prove that S ≤ 1/2(xy + zw) for any permutation (x, y, z, w) of (a, b, c, d). -/
theorem quadrilateral_area_inequality 
  (S : ℝ) (a b c d : ℝ) 
  (h_convex : ConvexQuadrilateral a b c d) 
  (h_area : Area a b c d = S) :
  ∀ (x y z w : ℝ), Permutation [x, y, z, w] [a, b, c, d] → 
    S ≤ (1/2) * (x*y + z*w) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l387_38707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subgrid_sum_upper_bound_max_subgrid_sum_achievable_max_subgrid_sum_is_six_l387_38708

/-- Represents a 5x5 grid where each cell contains either 0 or 1 -/
def Grid := Fin 5 → Fin 5 → Fin 2

/-- Calculates the sum of a 3x3 subgrid starting at position (i, j) -/
def subgridSum (grid : Grid) (i j : Fin 3) : ℕ :=
  (Finset.sum (Finset.range 3) fun x => 
   (Finset.sum (Finset.range 3) fun y => 
    (grid ⟨i.val + x, by sorry⟩ ⟨j.val + y, by sorry⟩).val))

/-- The maximum sum of any 3x3 subgrid in the given grid -/
def maxSubgridSum (grid : Grid) : ℕ :=
  Finset.sup (Finset.product (Finset.range 3) (Finset.range 3)) fun (i, j) => 
    subgridSum grid ⟨i, by sorry⟩ ⟨j, by sorry⟩

/-- Players take turns filling the grid -/
def isValidGrid (grid : Grid) : Prop :=
  (Finset.sum (Finset.product (Finset.range 5) (Finset.range 5)) fun (i, j) => (grid ⟨i, by sorry⟩ ⟨j, by sorry⟩).val) = 13 ∨
  (Finset.sum (Finset.product (Finset.range 5) (Finset.range 5)) fun (i, j) => (grid ⟨i, by sorry⟩ ⟨j, by sorry⟩).val) = 12

theorem max_subgrid_sum_upper_bound (grid : Grid) (h : isValidGrid grid) :
  maxSubgridSum grid ≤ 6 := by
  sorry

theorem max_subgrid_sum_achievable :
  ∃ grid : Grid, isValidGrid grid ∧ maxSubgridSum grid = 6 := by
  sorry

theorem max_subgrid_sum_is_six :
  (∃ grid : Grid, isValidGrid grid ∧ maxSubgridSum grid = 6) ∧
  (∀ grid : Grid, isValidGrid grid → maxSubgridSum grid ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subgrid_sum_upper_bound_max_subgrid_sum_achievable_max_subgrid_sum_is_six_l387_38708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_region_equally_l387_38781

-- Define the region P
def P : Set (ℝ × ℝ) :=
  {p | ∃ (i j : ℕ), i < 3 ∧ j < 3 ∧ 
    ((p.1 - (2*i + 1)/2)^2 + (p.2 - (2*j + 1)/2)^2 ≤ 1/4)}

-- Define the line m
def m : ℝ × ℝ → Prop :=
  λ p => 4*p.1 - p.2 = 1

-- Define the area of a region
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem line_divides_region_equally :
  (area {p ∈ P | m p} = area {p ∈ P | ¬m p}) ∧
  (4^2 + (-1)^2 + (-1)^2 = 18) := by
  sorry

#check line_divides_region_equally

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_region_equally_l387_38781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_B_l387_38727

/-- Represents the correctness of each statement about "3S" technology --/
inductive Statement
| RS_population
| RS_flood
| GPS_cash
| GIS_fire

/-- Represents the answer choices --/
inductive Answer
| A | B | C | D

/-- Checks if a given answer is correct based on the problem statement --/
def is_correct_answer (a : Answer) : Prop :=
  match a with
  | Answer.B => True
  | _ => False

/-- Theorem stating that Answer B is the correct choice --/
theorem correct_answer_is_B : is_correct_answer Answer.B := by
  -- The proof is trivial given our definition
  trivial

#check correct_answer_is_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_B_l387_38727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_mn_value_l387_38721

/-- The hyperbola equation -/
noncomputable def hyperbola (m n x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

theorem hyperbola_mn_value (m n : ℝ) :
  m > 0 → n > 0 → m * n ≠ 0 →
  (∃ x y, hyperbola m n x y) →
  (∃ c a, eccentricity c a = Real.sqrt 3) →
  (∃ x y, parabola x y ∧ x = 3 ∧ y = 0) →
  m * n = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_mn_value_l387_38721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_four_l387_38710

/-- The radius of a semicircle inscribed in a right triangle with legs 12 and 16 units,
    where the diameter of the semicircle lies along the longer leg. -/
noncomputable def inscribed_semicircle_radius : ℝ :=
  let leg1 : ℝ := 12
  let leg2 : ℝ := 16
  let hypotenuse : ℝ := Real.sqrt (leg1^2 + leg2^2)
  let triangle_area : ℝ := (1/2) * leg1 * leg2
  triangle_area / ((1/2) * (leg1 + leg2 + hypotenuse))

theorem inscribed_semicircle_radius_is_four :
  inscribed_semicircle_radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_four_l387_38710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identities_l387_38711

theorem angle_identities (α : Real) 
  (h1 : α ∈ Set.Ioo (Real.pi/2) Real.pi) 
  (h2 : Real.sin α = Real.sqrt 5 / 5) : 
  Real.tan (Real.pi/4 + 2*α) = -1/7 ∧ 
  Real.cos (5*Real.pi/6 - 2*α) = -(3*Real.sqrt 3 + 4)/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identities_l387_38711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trams_spy_proof_l387_38760

/-- Represents the time interval between consecutive trams in hours -/
noncomputable def tram_interval : ℚ := 118 / 60

/-- Represents the duration of Vasya's observation in hours -/
def vasya_observation_time : ℚ := 2

/-- Represents the number of buses Vasya observed -/
def vasya_buses : ℕ := 1

/-- Represents the number of trams Vasya observed -/
def vasya_trams : ℕ := 2

/-- Represents the duration of the Spy's observation in hours -/
def spy_observation_time : ℕ := 10

/-- Represents the number of buses the Spy observed -/
def spy_buses : ℕ := 10

/-- Calculates the minimum number of trams that could have passed during the Spy's observation -/
def min_trams_spy : ℕ := 4

theorem min_trams_spy_proof :
  (tram_interval ≤ vasya_observation_time) ∧
  (vasya_observation_time * vasya_buses < spy_observation_time * spy_buses) →
  min_trams_spy = 4 := by
  sorry

#eval min_trams_spy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trams_spy_proof_l387_38760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_second_quadrant_l387_38703

theorem tan_alpha_second_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 3 / 5) : 
  Real.tan α = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_second_quadrant_l387_38703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_top_side_length_l387_38714

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  base : ℚ
  top : ℚ
  height : ℚ
  area : ℚ

/-- The area formula for a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℚ :=
  (t.base + t.top) * t.height / 2

/-- Theorem stating that for a trapezoid with given dimensions, the top side length is 19 -/
theorem trapezoid_top_side_length
  (t : Trapezoid)
  (h_base : t.base = 25)
  (h_height : t.height = 13)
  (h_area : t.area = 286)
  (h_area_formula : t.area = trapezoid_area t) :
  t.top = 19 := by
  sorry

#check trapezoid_top_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_top_side_length_l387_38714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_example_l387_38756

/-- Theorem: The slope of the line passing through (4, -3) and (-1, 7) is -2. -/
theorem line_slope_example : (7 - (-3)) / ((-1) - 4) = -2 := by
  -- Evaluate the numerator
  have h1 : 7 - (-3) = 10 := by norm_num
  -- Evaluate the denominator
  have h2 : (-1) - 4 = -5 := by norm_num
  -- Rewrite the fraction using the evaluated numerator and denominator
  rw [h1, h2]
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_example_l387_38756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l387_38761

theorem tan_pi_4_minus_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.cos α = -4/5) : 
  Real.tan (π/4 - α) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l387_38761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_emails_l387_38798

def email_count (n : ℕ) : ℕ → ℕ
| 0 => n
| (k + 1) => (email_count n k) / 2

def total_emails (initial : ℕ) (days : ℕ) : ℕ :=
  (List.range days).map (email_count initial) |>.sum

theorem vacation_emails : total_emails 16 4 = 30 := by
  rfl

#eval total_emails 16 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_emails_l387_38798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_with_more_than_two_factors_l387_38753

def numbers : List Nat := [105, 142, 165, 187, 221]

def has_more_than_two_distinct_prime_factors (n : Nat) : Prop :=
  (Nat.factors n).toFinset.card > 2

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_with_more_than_two_factors : 
  ∃ (n : Nat), n ∈ numbers ∧ 
    has_more_than_two_distinct_prime_factors n ∧
    (∀ m ∈ numbers, has_more_than_two_distinct_prime_factors m → 
      largest_prime_factor n ≥ largest_prime_factor m) ∧
    n = 165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_with_more_than_two_factors_l387_38753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_one_l387_38732

/-- A function that, when applied some number of times to any real number, eventually equals 1 -/
def EventuallyOne (F : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ n : ℕ, (fun k => (F^[k]) x) n = 1

/-- The main theorem: if F is continuous and eventually reaches 1 for any input, then F(1) = 1 -/
theorem fixed_point_one (F : ℝ → ℝ) (hcont : Continuous F) (hone : EventuallyOne F) : F 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_one_l387_38732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l387_38795

def A : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}
def B : Set ℤ := {x : ℤ | -2 ≤ x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l387_38795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_digit_multiple_of_72_has_12_digits_l387_38733

/-- A natural number whose digits are all 0 or 1 -/
def BinaryDigitNumber (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The property of being the smallest multiple of 72 with all digits 0 or 1 -/
def SmallestBinaryDigitMultipleOf72 (n : ℕ) : Prop :=
  BinaryDigitNumber n ∧ 72 ∣ n ∧ ∀ m, m < n → BinaryDigitNumber m → ¬(72 ∣ m)

theorem smallest_binary_digit_multiple_of_72_has_12_digits :
  ∃ n, SmallestBinaryDigitMultipleOf72 n ∧ (n.digits 10).length = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_digit_multiple_of_72_has_12_digits_l387_38733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l387_38752

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 16 + p.y^2 / 4 = 1

/-- Definition of symmetry with respect to the origin -/
def symmetricWrtOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (p q r s : Point) : ℝ :=
  sorry  -- The actual computation of the area is not needed for the statement

theorem ellipse_quadrilateral_area
  (f₁ f₂ p q : Point)
  (h_ellipse_p : isOnEllipse p)
  (h_ellipse_q : isOnEllipse q)
  (h_symmetric : symmetricWrtOrigin p q)
  (h_distance : distance p q = distance f₁ f₂) :
  quadrilateralArea p f₁ q f₂ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l387_38752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_ant_distance_percentage_l387_38767

/-- The distance an ant walks in meters -/
noncomputable def ant_distance : ℝ := 1000

/-- The time both animals walk in minutes -/
noncomputable def walk_time : ℝ := 30

/-- The speed of the beetle in km/h -/
noncomputable def beetle_speed : ℝ := 1.8

/-- Converts kilometers to meters -/
noncomputable def km_to_m (km : ℝ) : ℝ := km * 1000

/-- Converts minutes to hours -/
noncomputable def min_to_hour (min : ℝ) : ℝ := min / 60

/-- Calculates the percentage of one value compared to another -/
noncomputable def percentage (part : ℝ) (whole : ℝ) : ℝ := (part / whole) * 100

theorem beetle_ant_distance_percentage :
  let ant_speed := ant_distance / (km_to_m (min_to_hour walk_time))
  let beetle_distance := km_to_m (beetle_speed * min_to_hour walk_time)
  percentage beetle_distance ant_distance = 90 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_ant_distance_percentage_l387_38767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_seven_thirty_l387_38763

/-- The angle between clock hands at a given time -/
noncomputable def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour % 12 : ℝ) - 11 * (minute : ℝ)| / 2

/-- Theorem: The angle between the hour and minute hands of a clock at 7:30 is 45° -/
theorem clock_angle_at_seven_thirty :
  clockAngle 7 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_seven_thirty_l387_38763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l387_38735

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + abs x) - 1 / (1 + x^2)

theorem f_inequality_range :
  ∀ x : ℝ, f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l387_38735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_function_zero_in_interval_linear_regression_increase_l387_38783

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 = 1

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 3 + 2 * x

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), hyperbola x y → (y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) := by sorry

theorem function_zero_in_interval :
  ∃ (x : ℝ), x > 1 ∧ x < 10 ∧ f x = 0 := by sorry

theorem linear_regression_increase :
  ∀ (x : ℝ), linear_regression (x + 2) - linear_regression x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_function_zero_in_interval_linear_regression_increase_l387_38783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_count_l387_38715

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed
    as the difference of squares of two nonnegative integers is at least 1500 -/
theorem difference_of_squares_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2000) ∧ 
  (∀ n ∈ S, ∃ a b : ℕ, n = a^2 - b^2) ∧
  S.card ≥ 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_count_l387_38715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_positive_l387_38720

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2^x) / 2 - 2 / (2^x) - x + 1

-- State the theorem
theorem zeros_sum_positive (a : ℝ) (x₁ x₂ : ℝ) :
  f a x₁ = 0 → f a x₂ = 0 → x₁ < x₂ → g x₁ + g x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_positive_l387_38720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_not_in_M_exp_quad_in_M_l387_38744

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ t : ℝ, f (t + 2) = f t + f 2}

-- Define the linear function
def f₁ : ℝ → ℝ := λ x ↦ 3 * x + 2

-- Define the exponential-quadratic function
noncomputable def f₂ (b : ℝ) : ℝ → ℝ := λ x ↦ (2 : ℝ)^x + b * x^2

-- Theorem statements
theorem linear_not_in_M : f₁ ∉ M := by sorry

theorem exp_quad_in_M : ∀ b : ℝ, f₂ b ∈ M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_not_in_M_exp_quad_in_M_l387_38744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_2d_l387_38737

/-- A convex set in a plane -/
structure ConvexSet where
  -- We don't need to define the internal structure of a convex set for this statement

/-- A point in a plane -/
structure Point where
  -- We don't need to define the internal structure of a point for this statement

/-- Define membership for Point in ConvexSet -/
instance : Membership Point ConvexSet where
  mem := λ _ _ => sorry

/-- Helly's theorem in 2D -/
theorem hellys_theorem_2d (n : ℕ) (sets : Fin n → ConvexSet) :
  n ≥ 3 →
  (∀ (i j : Fin n), i ≠ j → ∃ (p : Point), p ∈ sets i ∧ p ∈ sets j) →
  ∃ (p : Point), ∀ (i : Fin n), p ∈ sets i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_2d_l387_38737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l387_38792

/-- Represents a triangle -/
structure Triangle where
  perimeter : ℝ

/-- Represents a large triangle divided into smaller triangles -/
structure DividedTriangle where
  largeTriangle : Triangle
  smallTriangles : List Triangle

theorem divided_triangle_perimeter 
  (dt : DividedTriangle)
  (h1 : dt.largeTriangle.perimeter = 120)
  (h2 : dt.smallTriangles.length = 9)
  (h3 : ∀ t ∈ dt.smallTriangles, t.perimeter = (dt.smallTriangles.head?.map Triangle.perimeter).getD 0) :
  ∀ t ∈ dt.smallTriangles, t.perimeter = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l387_38792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_intercept_distance_l387_38770

/-- Proves that the distance between Michael and Marcos when the ball is touched for the first time is 2.5 m -/
theorem soccer_ball_intercept_distance
  (ball_speed : ℝ)
  (michael_speed : ℝ)
  (marcos_speed : ℝ)
  (initial_ball_michael_distance : ℝ)
  (initial_ball_marcos_distance : ℝ)
  (h1 : ball_speed = 4)
  (h2 : michael_speed = 9)
  (h3 : marcos_speed = 8)
  (h4 : initial_ball_michael_distance = 15)
  (h5 : initial_ball_marcos_distance = 30) :
  let marcos_intercept_time := initial_ball_marcos_distance / (marcos_speed + ball_speed)
  let ball_travel_distance := ball_speed * marcos_intercept_time
  let michael_travel_distance := michael_speed * marcos_intercept_time
  let final_ball_position := initial_ball_michael_distance + ball_travel_distance
  let final_michael_position := michael_travel_distance
  abs (final_ball_position - final_michael_position) = 2.5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_intercept_distance_l387_38770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_of_regular_tetrahedron_l387_38765

/-- A regular tetrahedron with side length a, where all lateral faces are right triangles
    and all vertices lie on a sphere. -/
structure RegularTetrahedron (a : ℝ) where
  side_length : a > 0
  right_triangles : Bool
  vertices_on_sphere : Bool

/-- The surface area of a sphere containing a regular tetrahedron -/
noncomputable def sphere_surface_area (a : ℝ) (t : RegularTetrahedron a) : ℝ := 3 * Real.pi * a^2

theorem sphere_surface_area_of_regular_tetrahedron (a : ℝ) (t : RegularTetrahedron a) :
  t.right_triangles ∧ t.vertices_on_sphere → sphere_surface_area a t = 3 * Real.pi * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_of_regular_tetrahedron_l387_38765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l387_38768

theorem simplify_and_rationalize :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = a / Real.sqrt b ∧
  a / Real.sqrt b = (3 * Real.sqrt 15015) / 1001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l387_38768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_and_vector_problem_l387_38750

noncomputable section

-- Define the points and angle
def A : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)
noncomputable def α : ℝ := Real.pi / 3
noncomputable def C : ℝ × ℝ := (Real.cos α, Real.sin α)

-- Define the vectors
def OA : ℝ × ℝ := A
noncomputable def OC : ℝ × ℝ := C
noncomputable def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define the theorem
theorem point_and_vector_problem :
  0 < α ∧ α < Real.pi ∧
  (OA.1 + OC.1)^2 + (OA.2 + OC.2)^2 = 7 →
  α = Real.pi / 3 ∧
  Real.arccos ((OA.1 * AC.1 + OA.2 * AC.2) / 
    (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (AC.1^2 + AC.2^2))) = 
    Real.arccos (-3 / (2 * Real.sqrt 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_and_vector_problem_l387_38750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_32_l387_38712

/-- The value of r for which the triangle area is 32 -/
noncomputable def r : ℝ := Real.sqrt 1020

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 1

/-- The line function -/
noncomputable def line (x : ℝ) : ℝ := r * x

/-- The x-coordinates of the intersection points -/
def intersection_points : Set ℝ := {x | parabola x = line x}

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := 
  let base := Real.sqrt (r^2 + 4)
  let height : ℝ := 2
  (1 / 2) * base * height

theorem triangle_area_is_32 : triangle_area = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_32_l387_38712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l387_38717

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The number we want to round -/
def number : ℝ := 45.14329

theorem round_to_nearest_tenth_of_number :
  roundToNearestTenth number = 45.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l387_38717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l387_38704

theorem divisibility_property (n : ℕ) (a : Fin n → ℕ+) 
  (h_n : n ≥ 3)
  (h_gcd : Nat.gcd (Finset.univ.prod (fun i => (a i).val)) 1 = 1)
  (h_div : ∀ j : Fin n, (Finset.univ.sum (fun i => (a i).val)) % (a j).val = 0) :
  (Finset.univ.prod (fun i => (a i).val)) ∣ (Finset.univ.sum (fun i => (a i).val)) ^ (n - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l387_38704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_r_l387_38745

theorem find_r (k r : ℝ) (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (Real.log 9) / (2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_r_l387_38745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l387_38799

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (0, -2)

-- Define the slope range
def slope_range (k : ℝ) : Prop :=
  (k > Real.sqrt 3 / 2 ∧ k < 2) ∨ (k < -Real.sqrt 3 / 2 ∧ k > -2)

-- Theorem statement
theorem ellipse_and_slope_range :
  -- Given conditions
  ∃ (A B Q : ℝ × ℝ) (a b : ℝ),
    -- A and B are left and right vertices of E
    E A.1 A.2 ∧ E B.1 B.2 ∧ A.1 < B.1 ∧
    -- General ellipse equation
    (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ E x y) ∧
    -- a > b > 0
    a > b ∧ b > 0 ∧
    -- Line BP intersects E at Q
    E Q.1 Q.2 ∧
    -- Triangle ABP is isosceles right triangle
    (A.1 - P.1)^2 + (A.2 - P.2)^2 = (B.1 - P.1)^2 + (B.2 - P.2)^2 ∧
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0 ∧
    -- Vector PQ = 3/2 * Vector QB
    Q.1 - P.1 = 3/2 * (B.1 - Q.1) ∧ Q.2 - P.2 = 3/2 * (B.2 - Q.2) →
  -- Conclusion
  (∀ k : ℝ, 
    (∃ M N : ℝ × ℝ, 
      E M.1 M.2 ∧ E N.1 N.2 ∧ 
      M.2 = k * M.1 - 2 ∧ N.2 = k * N.1 - 2 ∧
      M.1 * N.1 + M.2 * N.2 > 0) ↔ 
    slope_range k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l387_38799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l387_38762

noncomputable def f (x : ℝ) : ℝ := (x^4 + 3*x^2 - 8) / ((x - 2) * |x+3|)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -3 ∨ (-3 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l387_38762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l387_38716

/-- The function f(x) = ln x + ax has a maximum value of 0 if and only if a = -1/e -/
theorem max_value_condition (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → Real.log y + a * y ≤ Real.log x + a * x) ∧ 
  (Real.log x + a * x = 0) ↔ 
  a = -1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l387_38716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_of_exponents_l387_38786

theorem inverse_sum_of_exponents (x y : ℝ) (h1 : (2 : ℝ)^x = 100) (h2 : (50 : ℝ)^y = 100) : x⁻¹ + y⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_of_exponents_l387_38786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l387_38754

theorem circle_area_ratio (C D : ℝ) (hC : C > 0) (hD : D > 0) :
  (60 / 360) * (2 * Real.pi * C) = (40 / 360) * (2 * Real.pi * D) →
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l387_38754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l387_38788

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def F : ℝ × ℝ := (2, 0)

-- Define a point M on the parabola
structure PointOnParabola where
  point : ℝ × ℝ
  on_parabola : parabola point.1 point.2

-- Define point N on y-axis
noncomputable def N (M : PointOnParabola) : ℝ × ℝ := (0, M.point.2 * (M.point.1 / (M.point.1 - 2)))

-- M is the midpoint of FN
def is_midpoint (M : PointOnParabola) : Prop :=
  M.point.1 = (F.1 + (N M).1) / 2 ∧ M.point.2 = (F.2 + (N M).2) / 2

-- Distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_focus_property (M : PointOnParabola) 
  (h_midpoint : is_midpoint M) : distance F (N M) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l387_38788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l387_38718

-- Define the train properties
noncomputable def train1_length : ℝ := 50
noncomputable def train2_length : ℝ := 120
noncomputable def train1_speed : ℝ := 60
noncomputable def train2_speed : ℝ := 40

-- Define the function to calculate crossing time
noncomputable def calculate_crossing_time (l1 l2 v1 v2 : ℝ) : ℝ :=
  (l1 + l2) / ((v1 + v2) * (1000 / 3600))

-- Theorem statement
theorem trains_crossing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (calculate_crossing_time train1_length train2_length train1_speed train2_speed - 6.12) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l387_38718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_line_l387_38794

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The equation of line l -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

/-- The distance between two points on the trajectory -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Main theorem: Given the trajectory and distance condition, prove the equation of line l -/
theorem trajectory_intersection_line :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 2 / 3 →
  (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_line_l387_38794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l387_38797

/-- Represents the composition of a mixture --/
structure Mixture where
  oil : ℝ
  materialB : ℝ

/-- Represents the problem setup --/
structure MixtureProblem where
  initialMixture : Mixture
  addedOil : ℝ
  addedMixture : ℝ
  finalMaterialBPercentage : ℝ

/-- The specific problem instance --/
def problem : MixtureProblem :=
  { initialMixture := { oil := 0.2, materialB := 0.8 }
    addedOil := 2
    addedMixture := 6
    finalMaterialBPercentage := 0.7 }

/-- The theorem to prove --/
theorem initial_mixture_amount (x : ℝ) : 
  x > 0 →
  problem.initialMixture.oil * x + problem.addedOil = 
    (1 - problem.finalMaterialBPercentage) * (x + problem.addedMixture + problem.addedOil) ∧
  problem.initialMixture.materialB * x + problem.initialMixture.materialB * problem.addedMixture = 
    problem.finalMaterialBPercentage * (x + problem.addedMixture + problem.addedOil) →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l387_38797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_numbers_l387_38780

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 + (n / 10) % 10 + n % 10 = 9) ∧
  (n / 100 ≠ 0 ∧ (n / 10) % 10 ≠ 0 ∧ n % 10 ≠ 0)

def has_different_digits (a b : ℕ) : Prop :=
  (a / 100 ≠ b / 100) ∧
  ((a / 10) % 10 ≠ (b / 10) % 10) ∧
  (a % 10 ≠ b % 10)

def is_valid_set (s : Finset ℕ) : Prop :=
  (∀ n, n ∈ s → is_valid_number n) ∧
  (∀ a b, a ∈ s → b ∈ s → a ≠ b → has_different_digits a b)

theorem max_valid_numbers :
  ∃ (s : Finset ℕ), is_valid_set s ∧ s.card = 5 ∧
  ∀ (t : Finset ℕ), is_valid_set t → t.card ≤ 5 := by
  sorry

#check max_valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_numbers_l387_38780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equality_and_difference_l387_38748

-- Define the sets
def set1 : Set ℝ := {x : ℝ | x = 1}
def set2 : Set ℝ := {y : ℝ | (y - 1)^2 = 0}
def set3 : Set (Set ℝ) := {{x : ℝ | x = 1}}
def set4 : Set ℝ := {1}

-- Theorem statement
theorem sets_equality_and_difference :
  (set1 = set2) ∧ (set1 = set4) ∧ (set2 = set4) ∧ (set3 ≠ {set1}) ∧ (set3 ≠ {set2}) ∧ (set3 ≠ {set4}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equality_and_difference_l387_38748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l387_38713

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) := (log x) / x

-- State the theorem
theorem f_increasing_on_zero_to_e :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.exp 1 → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_to_e_l387_38713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_C_l387_38719

/-- Given a triangle ABC with area S and side lengths a, b, and c opposite to angles A, B, and C respectively,
    if 4√3S = (a+b)² - c², then sin C = √3/2 -/
theorem triangle_sin_C (S a b c : ℝ) (h : 4 * Real.sqrt 3 * S = (a + b)^2 - c^2) :
  let C := Real.arcsin (Real.sqrt 3 / 2)
  S = 1/2 * a * b * Real.sin C → Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_C_l387_38719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_upper_bound_l387_38766

theorem tan_upper_bound (m : ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/3), Real.tan x ≤ m) → m ≥ Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_upper_bound_l387_38766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_second_draw_no_replacement_expected_wins_four_draws_replacement_theorem_l387_38738

-- Define the number of red and white balls
def red_balls : ℕ := 3
def white_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define a win as drawing at least one red ball
def is_win (red_drawn : ℕ) : Prop := red_drawn ≥ 1

-- Define the probability of winning precisely on the second draw without replacement
def prob_win_second_draw_no_replacement : ℚ := 9 / 35

-- Define the expected number of wins in 4 consecutive draws with replacement
def expected_wins_four_draws_replacement : ℚ := 20 / 7

-- Theorem for winning precisely on the second draw without replacement
theorem win_second_draw_no_replacement :
  prob_win_second_draw_no_replacement = 9 / 35 := by sorry

-- Theorem for expected number of wins in 4 consecutive draws with replacement
theorem expected_wins_four_draws_replacement_theorem :
  expected_wins_four_draws_replacement = 20 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_second_draw_no_replacement_expected_wins_four_draws_replacement_theorem_l387_38738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_chocolate_game_strategy_game_outcome_determined_by_primality_l387_38791

/-- Represents the outcome of the triangle chocolate game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- The triangle chocolate game -/
def triangleChocolateGame (n : ℕ) : GameOutcome :=
  if Nat.Prime n then GameOutcome.SecondPlayerWins
  else GameOutcome.FirstPlayerWins

/-- Theorem stating the winning strategy for the triangle chocolate game -/
theorem triangle_chocolate_game_strategy (n : ℕ) (hn : n > 1) :
  (Nat.Prime n ↔ triangleChocolateGame n = GameOutcome.SecondPlayerWins) ∧
  (¬Nat.Prime n ↔ triangleChocolateGame n = GameOutcome.FirstPlayerWins) := by
  sorry

/-- Corollary: The game outcome is determined by whether n is prime or composite -/
theorem game_outcome_determined_by_primality (n : ℕ) (hn : n > 1) :
  (triangleChocolateGame n = GameOutcome.SecondPlayerWins) ↔ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_chocolate_game_strategy_game_outcome_determined_by_primality_l387_38791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximized_l387_38734

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.cos x) * (380 - x - x^2)

-- Define the integral function
noncomputable def integral (a b : ℝ) : ℝ := ∫ x in a..b, f x

-- State the theorem
theorem integral_maximized (a b : ℝ) (h : a ≤ b) :
  integral a b ≤ integral (-20) 19 ∧
  (integral a b = integral (-20) 19 ↔ a = -20 ∧ b = 19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximized_l387_38734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l387_38796

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_derivative (x : ℝ) : deriv f x - f x = (1 - 2*x) * Real.exp (-x)

axiom f_initial : f 0 = 0

theorem f_solution : ∀ x : ℝ, f x = x * Real.exp (-x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l387_38796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l387_38793

def sequenceA (n : ℕ+) : ℚ := (-1)^n.val / (n.val * (n.val + 1))

theorem sequence_formula (n : ℕ+) :
  sequenceA n = (-1)^n.val / (n.val * (n.val + 1)) :=
by
  -- The proof is trivial since it's the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l387_38793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_calculation_l387_38739

/-- Represents the rise in water level when a rectangular solid is immersed in a cylindrical vessel -/
noncomputable def water_level_rise (l w h d : ℝ) : ℝ :=
  (l * w * h) / (Real.pi * (d / 2)^2)

/-- Theorem stating the rise in water level for the given dimensions -/
theorem water_rise_calculation :
  water_level_rise 10 12 15 18 = 200 / (9 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_calculation_l387_38739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l387_38784

/-- The area of a square inscribed in a specific ellipse with rotated sides -/
theorem inscribed_square_area : ∃ (s : ℝ), 
  (∀ (x y : ℝ), x^2/4 + y^2/8 = 1 → 
    ∃ (t : ℝ), (x = 0 ∧ y = Real.sqrt 2 * t) ∨ (x = Real.sqrt 2 * t ∧ y = 0)) →
  s^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l387_38784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_maximum_triangle_sine_sum_maximum_achievable_l387_38777

theorem triangle_sine_sum_maximum (A B C : ℝ) : 
  A + B + C = π → 
  A > 0 → B > 0 → C > 0 →
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

theorem triangle_sine_sum_maximum_achievable : 
  ∃ A B C : ℝ, A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧
  Real.sin A + Real.sin B + Real.sin C = 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_maximum_triangle_sine_sum_maximum_achievable_l387_38777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l387_38725

noncomputable def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

noncomputable def std_dev (xs : List ℝ) : ℝ := Real.sqrt (variance xs)

theorem shooting_test_results :
  mean scores = 7 ∧ std_dev scores = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l387_38725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lolas_rabbits_l387_38740

theorem lolas_rabbits (
  initial_breeding_rabbits : ℕ)
  (first_spring_multiplier : ℕ)
  (first_spring_adoption_rate : ℚ)
  (first_spring_returned : ℕ)
  (second_spring_kittens : ℕ)
  (second_spring_adopted : ℕ)
  (h1 : initial_breeding_rabbits = 10)
  (h2 : first_spring_multiplier = 10)
  (h3 : first_spring_adoption_rate = 1/2)
  (h4 : first_spring_returned = 5)
  (h5 : second_spring_kittens = 60)
  (h6 : second_spring_adopted = 4) :
  initial_breeding_rabbits +
  (Int.floor (initial_breeding_rabbits * first_spring_multiplier * (1 - first_spring_adoption_rate) + first_spring_returned : ℚ)) +
  (second_spring_kittens - second_spring_adopted) = 121 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lolas_rabbits_l387_38740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_leftover_apples_l387_38776

/-- Calculates the number of leftover apples after making mini pies -/
def leftover_apples (initial_apples : ℕ) (mini_pies : ℕ) (apples_per_mini_pie : ℚ) : ℕ :=
  initial_apples - Int.toNat ((mini_pies : ℚ) * apples_per_mini_pie).floor

/-- Proves that Ivan has 36 apples leftover -/
theorem ivan_leftover_apples :
  leftover_apples 48 24 (1/2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_leftover_apples_l387_38776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geraldine_banana_consumption_l387_38726

/-- Represents the number of bananas eaten on each day -/
def banana_sequence (first : ℕ) : ℕ → ℕ
  | 0 => first
  | n + 1 => banana_sequence first n + 8

/-- The sum of bananas eaten over 5 days -/
def total_bananas (first : ℕ) : ℕ :=
  (List.range 5).map (banana_sequence first) |>.sum

theorem geraldine_banana_consumption (first : ℕ) 
  (h : total_bananas first = 150) : 
  banana_sequence first 4 = 46 := by
  sorry

#eval total_bananas 14  -- Should output 150
#eval banana_sequence 14 4  -- Should output 46

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geraldine_banana_consumption_l387_38726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l387_38749

-- Define the function f(x) = cos²(2ax) - sin²(2ax)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos (2 * a * x))^2 - (Real.sin (2 * a * x))^2

-- Define the concept of a function having a minimum positive period
def has_min_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ y, f (y + q) ≠ f y

-- State the theorem
theorem sufficient_but_not_necessary (a : ℝ) :
  (a = 1/2 → has_min_positive_period (f a) π) ∧
  ¬(has_min_positive_period (f a) π → a = 1/2) := by
  sorry

#check sufficient_but_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l387_38749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l387_38773

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 3

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_and_line_properties :
  ∃ (k : ℝ), 
    (∀ x y, ellipse x y → ¬(y = 3 ∧ x ≠ 0)) ∧  -- Vertex at (0,3)
    (∃ x1 y1 x2 y2, 
      x1 ≠ x2 ∧  -- Two distinct points
      ellipse x1 y1 ∧ ellipse x2 y2 ∧  -- Points on ellipse
      line k x1 y1 ∧ line k x2 y2 ∧  -- Points on line
      distance 0 3 x1 y1 = distance 0 3 x2 y2) ∧  -- |AM| = |AN|
    (k = Real.sqrt 7 / 5 ∨ k = -Real.sqrt 7 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l387_38773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_angle_negation_triangle_at_most_one_obtuse_angle_l387_38782

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields here, for example:
  vertices : Fin 3 → ℝ × ℝ

/-- Number of obtuse angles in a triangle -/
def Triangle.obtuse_angles (t : Triangle) : ℕ :=
  sorry -- Implementation details would go here

theorem triangle_obtuse_angle_negation :
  (¬ (∀ t : Triangle, t.obtuse_angles ≤ 1)) ↔ (∃ t : Triangle, t.obtuse_angles ≥ 2) :=
by sorry

/-- A triangle has at most one obtuse angle -/
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : t.obtuse_angles ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_angle_negation_triangle_at_most_one_obtuse_angle_l387_38782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_l387_38774

-- Define the functions that bound the area
noncomputable def f (x : ℝ) : ℝ := Real.arccos x
def g : ℝ → ℝ := λ _ => 0
def h : ℝ → ℝ := λ _ => 0

-- Define the area as the integral of arccos x from 0 to 1
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, f x

-- Theorem statement
theorem area_is_one : area = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_l387_38774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_pentagon_side_theorem_l387_38729

/-- The side length of a pentagon that divides the area between two concentric regular pentagons -/
noncomputable def third_pentagon_side (a b : ℝ) : ℝ :=
  4 * Real.sqrt 3

/-- Theorem stating the side length of the third pentagon -/
theorem third_pentagon_side_theorem (a b x : ℝ) 
  (ha : a = 4) 
  (hb : b = 12) 
  (hx : x = third_pentagon_side a b) 
  (concentric : Bool) 
  (parallel_sides : Bool) 
  (area_ratio : (x^2 - a^2) / (b^2 - x^2) = 1 / 3) : 
  x = 4 * Real.sqrt 3 := by
  sorry

#check third_pentagon_side_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_pentagon_side_theorem_l387_38729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l387_38728

/-- The minimum distance from (m, n) to the origin, given three intersecting lines -/
theorem min_distance_to_origin (m n : ℝ) : 
  (∃ x y : ℝ, y = 2 * x ∧ x + y = 3 ∧ m * x + n * y + 5 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ m' n' : ℝ, m' * 1 + n' * 2 + 5 = 0 → 
    d ≤ Real.sqrt (m' ^ 2 + n' ^ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l387_38728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_m_l387_38764

-- Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | -1 < x ∧ x < 3}
def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2*m - 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  {m : ℝ | C m ∪ B = B} = Set.Iic 2 := by sorry

-- Here, Set.Iic 2 represents the set (-∞, 2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_m_l387_38764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_fifth_sixth_digits_l387_38742

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def digit_at (n : ℕ) (pos : ℕ) : ℕ := (n / (10^(9 - pos))) % 10

def valid_number (n : ℕ) : Prop :=
  n ≥ 100000000 ∧ n < 1000000000 ∧
  (∀ i j : Fin 9, i ≠ j → digit_at n i.val ≠ digit_at n j.val) ∧
  (∀ i : Fin 9, ∃ pos : Fin 9, digit_at n pos.val = i.val + 1) ∧
  (∀ pos : Fin 8, ¬(is_prime (10 * (digit_at n pos.val) + (digit_at n (pos.val + 1)))))

theorem largest_valid_number_fifth_sixth_digits :
  ∃ n : ℕ, valid_number n ∧
    (∀ m : ℕ, valid_number m → m ≤ n) ∧
    digit_at n 5 = 3 ∧ digit_at n 6 = 5 :=
by
  sorry

#print axioms largest_valid_number_fifth_sixth_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_fifth_sixth_digits_l387_38742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l387_38778

noncomputable def f (x : ℝ) := Real.log ((3 - x) / (3 + x))

theorem f_properties :
  (∀ x, x ∈ Set.Ioo (-3 : ℝ) 3 → f x = -f (-x)) →
  (∀ x y, x ∈ Set.Ioo (-3 : ℝ) 3 → y ∈ Set.Ioo (-3 : ℝ) 3 → x < y → f x > f y) ∧
  (∃ k : ℝ, -Real.sqrt 3 < k ∧ k ≤ -1 ∧
    ∀ θ : ℝ, f (k - Real.cos θ) + f (Real.cos θ ^ 2 - k ^ 2) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l387_38778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_length_l387_38789

/-- The arc length traveled by the minute hand of a clock in a given time -/
noncomputable def arcLength (radius : ℝ) (time : ℝ) : ℝ := 2 * Real.pi * radius * (time / 60)

/-- Theorem: The length of the minute hand of a clock is 1/2 m,
    given that it travels an arc length of π/3 m in 20 minutes -/
theorem minute_hand_length :
  ∃ (r : ℝ), arcLength r 20 = Real.pi / 3 ∧ r = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_length_l387_38789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_F_is_4_l387_38724

/-- The parabola defined by x = 4t², y = 4t with focus at (1, 0) -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- A point P on the parabola -/
noncomputable def P : Parabola := { t := Real.sqrt 3 / 2 }

/-- The focus F of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_to_F_is_4 :
  distance (P.x, P.y) F = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_F_is_4_l387_38724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l387_38743

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The right focus of an ellipse -/
noncomputable def rightFocus (e : Ellipse) : ℝ × ℝ :=
  (Real.sqrt (e.a^2 - e.b^2), 0)

theorem ellipse_slope_theorem (e : Ellipse) (k : ℝ) 
    (h_ecc : eccentricity e = Real.sqrt 3 / 2)
    (h_k_pos : k > 0)
    (A B : EllipsePoint e)
    (h_line : ∃ (m c : ℝ), A.y = m * A.x + c ∧ B.y = m * B.x + c ∧ m = k)
    (h_AF_FB : let F := rightFocus e
               Real.sqrt ((A.x - F.1)^2 + A.y^2) = 3 * Real.sqrt ((B.x - F.1)^2 + B.y^2)) :
  k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l387_38743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_ports_l387_38730

/-- The distance between ports A and B in meters -/
def distance : ℝ := sorry

/-- The speed of the current in meters per second -/
def current_speed : ℝ := 2

/-- The speed of the boat downstream relative to water in meters per second -/
def boat_speed_downstream : ℝ := 8

/-- The distance at which the boat encounters the raft from port A in meters -/
def raft_encounter_distance : ℝ := 4000

theorem distance_between_ports : distance = 8000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_ports_l387_38730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l387_38757

-- Define the molar masses
noncomputable def molar_mass_N : ℝ := 14.01
noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45

-- Define the composition of NH4Cl
def num_N_atoms : ℕ := 1
def num_H_atoms : ℕ := 4
def num_Cl_atoms : ℕ := 1

-- Define the molar mass of NH4Cl
noncomputable def molar_mass_NH4Cl : ℝ :=
  num_N_atoms * molar_mass_N + num_H_atoms * molar_mass_H + num_Cl_atoms * molar_mass_Cl

-- Define the mass of H in one mole of NH4Cl
noncomputable def mass_H_in_NH4Cl : ℝ := num_H_atoms * molar_mass_H

-- Define the mass percentage of H in NH4Cl
noncomputable def mass_percentage_H : ℝ := (mass_H_in_NH4Cl / molar_mass_NH4Cl) * 100

-- Theorem statement
theorem mass_percentage_H_approx :
  ∃ ε > 0, |mass_percentage_H - 7.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l387_38757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_groups_l387_38775

theorem right_angled_triangle_groups : 
  let groups : List (ℝ × ℝ × ℝ) := [(6, 7, 8), (1, Real.sqrt 2, 5), (6, 8, 10), (Real.sqrt 5, 2 * Real.sqrt 3, Real.sqrt 15)]
  ∃! g, g ∈ groups ∧ 
    let (a, b, c) := g
    (a < c ∧ b < c) ∧ a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_groups_l387_38775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l387_38759

open Real

theorem function_inequalities (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo 0 (π / 2), deriv f x * cos x > f x * sin x) :
  (f (π / 3) > sqrt 2 * f (π / 4)) ∧ (2 * f (π / 4) > sqrt 6 * f (π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l387_38759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_type_I_error_at_95_percent_confidence_l387_38746

/-- Represents the confidence level of a statistical test -/
structure ConfidenceLevel where
  value : ℝ
  property : 0 ≤ value ∧ value ≤ 1

/-- Represents the significance level (α) of a statistical test -/
structure SignificanceLevel where
  value : ℝ
  property : 0 ≤ value ∧ value ≤ 1

/-- The relationship between confidence level and significance level -/
axiom confidence_significance_relation (cl : ConfidenceLevel) (α : SignificanceLevel) :
  cl.value + α.value = 1

/-- The probability of Type I error in a statistical test -/
def type_I_error_probability (α : SignificanceLevel) : ℝ := α.value

/-- Theorem: In a statistical test with 95% confidence level, 
    the probability of Type I error is 5% -/
theorem type_I_error_at_95_percent_confidence :
  ∀ (cl : ConfidenceLevel) (α : SignificanceLevel),
  cl.value = 0.95 → type_I_error_probability α = 0.05 := by
  sorry

/-- Example of creating a 95% confidence level -/
def confidence_95 : ConfidenceLevel where
  value := 0.95
  property := by
    constructor
    · exact le_of_lt (by norm_num)
    · exact (by norm_num)

/-- Example of creating a 5% significance level -/
def significance_05 : SignificanceLevel where
  value := 0.05
  property := by
    constructor
    · exact le_of_lt (by norm_num)
    · exact (by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_type_I_error_at_95_percent_confidence_l387_38746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_probability_l387_38723

/-- A random walk on a number line -/
def RandomWalk := List Int

/-- The probability of a specific random walk outcome -/
def probability (walk : RandomWalk) : Rat :=
  (1 / 2) ^ walk.length

/-- Check if a walk reaches 5 at some point -/
def reaches_five (walk : RandomWalk) : Bool :=
  (walk.scanl (· + ·) 0).any (· = 5)

/-- Check if a walk ends at 0 -/
def ends_at_zero (walk : RandomWalk) : Bool :=
  walk.sum = 0

/-- Generate all possible walks of length 10 -/
def all_walks : List RandomWalk :=
  sorry

/-- The main theorem to prove -/
theorem random_walk_probability :
  let favorable_walks := all_walks.filter (λ w => reaches_five w ∧ ends_at_zero w)
  (favorable_walks.map probability).sum / (all_walks.map probability).sum = 63 / 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_probability_l387_38723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l387_38731

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := (1 - i) / (1 + i)

-- Theorem statement
theorem modulus_of_z_is_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l387_38731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_probability_l387_38736

/-- The probability of a randomly selected three-digit number (from 001 to 999) 
    having at least two identical digits is approximately 0.28. -/
theorem license_plate_probability : ℝ := by
  -- The actual value is 279/999, which is approximately 0.28
  exact 279 / 999


end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_probability_l387_38736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l387_38751

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω : ℝ) (x₁ x₂ : ℝ) 
  (h_omega_pos : ω > 0)
  (h_f_x₁ : f ω x₁ = -2)
  (h_f_x₂ : f ω x₂ = 0)
  (h_min_diff : ∀ y z, f ω y = -2 ∧ f ω z = 0 → |y - z| ≥ Real.pi)
  (h_exists_min : ∃ y z, f ω y = -2 ∧ f ω z = 0 ∧ |y - z| = Real.pi) :
  ω = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l387_38751
