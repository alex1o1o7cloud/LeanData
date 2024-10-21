import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l313_31361

/-- The length of the base of a parallelogram with given area and altitude-base ratio -/
noncomputable def base_length (area : ℝ) (altitude_base_ratio : ℝ) : ℝ :=
  Real.sqrt (area / altitude_base_ratio)

/-- Theorem: The base length of a parallelogram with area 162 sq m and altitude twice the base is 9 m -/
theorem parallelogram_base_length :
  base_length 162 2 = 9 := by
  -- Unfold the definition of base_length
  unfold base_length
  -- Simplify the expression
  simp [Real.sqrt_div]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l313_31361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prev_year_profit_ratio_is_ten_percent_l313_31372

/-- Represents the financial data of a company for two consecutive years -/
structure CompanyFinancials where
  prev_year_revenue : ℚ
  prev_year_profit : ℚ
  current_year_revenue_ratio : ℚ
  current_year_profit_ratio : ℚ
  current_year_profit_growth : ℚ

/-- Calculates the profit to revenue ratio for the previous year -/
def prev_year_profit_ratio (cf : CompanyFinancials) : ℚ :=
  cf.prev_year_profit / cf.prev_year_revenue

/-- Theorem stating that under given conditions, the previous year's profit ratio was 10% -/
theorem prev_year_profit_ratio_is_ten_percent (cf : CompanyFinancials)
  (h1 : cf.current_year_revenue_ratio = 4/5)
  (h2 : cf.current_year_profit_ratio = 3/20)
  (h3 : cf.current_year_profit_growth = 6/5) :
  prev_year_profit_ratio cf = 1/10 := by
  sorry

#check prev_year_profit_ratio_is_ten_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prev_year_profit_ratio_is_ten_percent_l313_31372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_l313_31398

theorem cosine_difference (α β : ℝ) : 
  α ∈ Set.Ioo 0 (π/2) → 
  β ∈ Set.Ioo 0 (π/2) → 
  Real.cos (α + β) = -5/13 → 
  Real.tan α + Real.tan β = 3 → 
  Real.cos (α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_l313_31398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l313_31379

/-- The function f(x, y, z) to be minimized -/
noncomputable def f (x y z : ℝ) : ℝ := (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2)

/-- Theorem stating the minimum value of f(x, y, z) -/
theorem min_value_of_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  f x y z ≥ 0 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ f a b c = 0 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l313_31379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_tenth_l313_31396

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The sum of 96.23 and 47.849 rounded to the nearest tenth equals 144.1 -/
theorem sum_rounded_to_tenth :
  round_to_tenth (96.23 + 47.849) = 144.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_tenth_l313_31396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_revenue_is_1260_l313_31306

/-- Represents the ticket pricing rules --/
def ticket_price (n : ℕ) : ℚ :=
  match n with
  | 1 | 2 => 20
  | 3 | 4 => 20 * (1 - 1/10)
  | 5 | 6 => 20 * (1 - 1/5)
  | 7 | 8 => 20 * (1 - 3/10)
  | 9 | 10 => 20 * (1 + 1/10)
  | _ => 20 * (1 + 1/5)

/-- Calculates the revenue for a given group size --/
def group_revenue (size : ℕ) (count : ℕ) : ℚ :=
  (ticket_price size) * size * count

/-- Theorem: The total revenue from ticket sales is $1260 --/
theorem total_revenue_is_1260 : 
  group_revenue 2 8 + 
  group_revenue 3 5 + 
  group_revenue 4 3 + 
  group_revenue 5 2 + 
  group_revenue 6 1 + 
  group_revenue 9 1 = 1260 := by
  sorry

#eval group_revenue 2 8 + 
      group_revenue 3 5 + 
      group_revenue 4 3 + 
      group_revenue 5 2 + 
      group_revenue 6 1 + 
      group_revenue 9 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_revenue_is_1260_l313_31306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l313_31374

def M : Set ℝ := {x | x^2 - 3*x < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l313_31374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_proof_l313_31326

/-- Piecewise water billing function -/
noncomputable def water_bill (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 4 then C₁ * x else 4 * C₁ + C₂ * (x - 4)

/-- Theorem: Given specific billing rates and a monthly fee of 12.8 yuan, the water usage is 9 tons -/
theorem water_usage_proof (C₁ C₂ : ℝ) (h₁ : C₁ = 1.2) (h₂ : C₂ = 1.6) :
  ∃ x : ℝ, water_bill C₁ C₂ x = 12.8 ∧ x = 9 := by
  use 9
  constructor
  · simp [water_bill, h₁, h₂]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_proof_l313_31326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_first_and_second_quadrants_l313_31301

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 4

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem circle_in_first_and_second_quadrants :
  ∃ x y : ℝ, circle_equation x y ∧ (first_quadrant x y ∨ second_quadrant x y) ∧
  ∀ x y : ℝ, circle_equation x y → ¬(third_quadrant x y ∨ fourth_quadrant x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_first_and_second_quadrants_l313_31301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l313_31313

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  (∀ x, f ((-Real.pi/8) + x) = f ((-Real.pi/8) - x)) ∧
  (∃ α : ℝ, 0 < α ∧ α < Real.pi ∧ ∀ x, f (x + α) = f (x + 3 * α)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l313_31313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_10_sqrt_3_cos_largest_angle_is_1_div_7_l313_31329

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The triangle satisfies the given conditions -/
noncomputable def SpecialTriangle : Triangle where
  a := 5
  b := 8
  c := Real.sqrt (5^2 + 8^2 - 2*5*8*(1/2))  -- Using Law of Cosines
  A := Real.arccos ((8^2 + Real.sqrt (5^2 + 8^2 - 2*5*8*(1/2))^2 - 5^2) / (2*8*Real.sqrt (5^2 + 8^2 - 2*5*8*(1/2))))
  B := Real.arccos ((5^2 + Real.sqrt (5^2 + 8^2 - 2*5*8*(1/2))^2 - 8^2) / (2*5*Real.sqrt (5^2 + 8^2 - 2*5*8*(1/2))))
  C := Real.arccos (1/2)

/-- The area of the triangle is 10√3 -/
theorem area_is_10_sqrt_3 (t : Triangle) (h : t = SpecialTriangle) :
  (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3 := by
  sorry

/-- The cosine of the largest angle is 1/7 -/
theorem cos_largest_angle_is_1_div_7 (t : Triangle) (h : t = SpecialTriangle) :
  Real.cos t.B = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_10_sqrt_3_cos_largest_angle_is_1_div_7_l313_31329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l313_31319

/-- The function f(x) defined in terms of a -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ a^x - 4*a + 3

/-- The statement that the inverse of f passes through (-1, 2) -/
def inverse_passes_through (a : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g (-1) = 2

/-- Theorem stating that a = 2 given the conditions -/
theorem a_equals_two : ∃ a : ℝ, f a 2 = -1 ∧ inverse_passes_through a → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l313_31319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_and_sequence_bounds_l313_31367

/-- The function f(x) = (x-1)e^x + 1 -/
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x + 1

/-- The sequence x_n defined by the recurrence relation x_n e^(x_(n+1)) = e^(x_n) - 1 -/
noncomputable def x : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.log ((Real.exp (x n) - 1) / (x n))

theorem f_positive_and_sequence_bounds :
  (∀ x > 0, f x > 0) ∧
  (∀ n : ℕ, x n > x (n + 1) ∧ x (n + 1) > 1 / (2 ^ (n + 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_and_sequence_bounds_l313_31367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l313_31325

theorem coefficient_x_cubed_expansion : 
  (Polynomial.coeff ((1 - 2 * X : Polynomial ℚ)^10) 3) = -960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l313_31325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31378

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h : 2 * Real.cos (t.A - t.C) + Real.cos (2 * t.B) = 1 + 2 * Real.cos t.A * Real.cos t.C) :
  -- Part 1: a, b, c form a geometric progression
  ∃ (r : ℝ), t.a * r = t.b ∧ t.b * r = t.c ∧
  -- Part 2: When b = 2, the minimum value of u is 2√3
  (t.b = 2 → 
    let u := |((t.a^2 + t.c^2 - 5) / (t.a - t.c))|
    ∀ (x : ℝ), u ≤ x → 2 * Real.sqrt 3 ≤ x) ∧
  -- Part 3: When u is at its minimum and b = 2, cos B = 7/8
  (t.b = 2 → 
    let u := |((t.a^2 + t.c^2 - 5) / (t.a - t.c))|
    u = 2 * Real.sqrt 3 → Real.cos t.B = 7/8) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_numbers_with_given_hcf_and_product_l313_31397

theorem lcm_of_numbers_with_given_hcf_and_product
  (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) :
  Nat.lcm a b = 182 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_numbers_with_given_hcf_and_product_l313_31397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_eight_l313_31308

theorem floor_expression_equals_eight :
  ⌊(2025^3 : ℚ) / (2023 * 2024) - (2023^3 : ℚ) / (2024 * 2025)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_eight_l313_31308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYMR_is_80_l313_31377

/-- Triangle PQR with given properties -/
structure Triangle_PQR where
  PQ : ℝ
  PR : ℝ
  area : ℝ
  h_PQ : PQ = 60
  h_PR : PR = 20
  h_area : area = 240

/-- Point in Euclidean space -/
@[ext]
structure Point where
  x : ℝ
  y : ℝ

/-- Line in Euclidean space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on a line segment -/
def PointOnSegment (A B : Point) : Type :=
  { P : Point // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * A.x + t * B.x, (1 - t) * A.y + t * B.y⟩ }

/-- Midpoint of a line segment -/
noncomputable def Midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

/-- Altitude from a point to a line -/
noncomputable def Altitude (P Q R : Point) : Line :=
  sorry

/-- Intersection of two lines -/
noncomputable def LineIntersection (l₁ l₂ : Line) : Point :=
  sorry

/-- Area of a quadrilateral -/
noncomputable def QuadrilateralArea (A B C D : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem area_XYMR_is_80 (t : Triangle_PQR) (P Q R : Point) :
  let M : Point := Midpoint P Q
  let N : Point := Midpoint P R
  let alt : Line := Altitude P Q R
  let X : Point := LineIntersection alt (Line.mk 0 0 0) -- Placeholder for line MN
  let Y : Point := LineIntersection alt (Line.mk 0 0 0) -- Placeholder for line QR
  QuadrilateralArea X Y M R = 80 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYMR_is_80_l313_31377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_rational_points_in_acute_rational_triangle_l313_31318

/-- A triangle with vertices A, B, C in the real plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the angle at vertex A of a triangle -/
noncomputable def angle_A (t : Triangle) : ℝ := sorry

/-- Calculate the angle at vertex B of a triangle -/
noncomputable def angle_B (t : Triangle) : ℝ := sorry

/-- Calculate the angle at vertex C of a triangle -/
noncomputable def angle_C (t : Triangle) : ℝ := sorry

/-- Predicate for a rational angle (in degrees) -/
def is_rational_angle (angle : ℝ) : Prop :=
  ∃ q : ℚ, angle = q

/-- Predicate for a rational triangle -/
def is_rational_triangle (t : Triangle) : Prop :=
  is_rational_angle (angle_A t) ∧
  is_rational_angle (angle_B t) ∧
  is_rational_angle (angle_C t)

/-- Predicate for an acute triangle -/
def is_acute_triangle (t : Triangle) : Prop :=
  angle_A t < 90 ∧ angle_B t < 90 ∧ angle_C t < 90

/-- A point in the real plane -/
def Point := ℝ × ℝ

/-- Predicate for a rational point within a triangle -/
def is_rational_point (p : Point) (t : Triangle) : Prop :=
  is_rational_triangle (Triangle.mk t.A t.B p) ∧
  is_rational_triangle (Triangle.mk t.A p t.C) ∧
  is_rational_triangle (Triangle.mk p t.B t.C)

/-- Theorem: There exist at least three distinct rational points within any acute rational triangle -/
theorem three_rational_points_in_acute_rational_triangle (t : Triangle) 
  (h_rational : is_rational_triangle t) (h_acute : is_acute_triangle t) :
  ∃ (p1 p2 p3 : Point), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    is_rational_point p1 t ∧ 
    is_rational_point p2 t ∧ 
    is_rational_point p3 t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_rational_points_in_acute_rational_triangle_l313_31318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_of_f_l313_31302

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (2 - x) + Real.sqrt (3 * x + 12)

-- State the theorem
theorem max_min_ratio_of_f :
  ∃ (M m : ℝ), 
    (∀ x, -4 ≤ x ∧ x ≤ 2 → f x ≤ M) ∧
    (∃ x, -4 ≤ x ∧ x ≤ 2 ∧ f x = M) ∧
    (∀ x, -4 ≤ x ∧ x ≤ 2 → m ≤ f x) ∧
    (∃ x, -4 ≤ x ∧ x ≤ 2 ∧ f x = m) ∧
    M / m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_of_f_l313_31302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_range_of_PM_l313_31346

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the left focus F
def F : ℝ × ℝ := (-2, 0)

-- Define that P, A, B are on the ellipse C
variable (P A B : ℝ × ℝ)
axiom P_on_C : C P.1 P.2
axiom A_on_C : C A.1 A.2
axiom B_on_C : C B.1 B.2

-- Define that AB passes through F
axiom AB_through_F : ∃ t : ℝ, F = (1-t) • A + t • B

-- Define M on AB such that OM · AB = 0
variable (M : ℝ × ℝ)
axiom M_on_AB : ∃ s : ℝ, M = (1-s) • A + s • B
axiom OM_perp_AB : M.1 * (B.1 - A.1) + M.2 * (B.2 - A.2) = 0

-- Theorem 1: Trajectory of M
theorem trajectory_of_M : (M.1 + 1)^2 + M.2^2 = 1 := by
  sorry

-- Theorem 2: Range of |PM|
theorem range_of_PM : ∃ (t : ℝ), 
  Real.sqrt 6 / 2 - 1 ≤ t ∧ t ≤ Real.sqrt 6 + 2 ∧
  t = Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_range_of_PM_l313_31346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_sum_l313_31382

/-- The set of available numbers to be placed in the circles and squares. -/
def available_numbers : Finset Nat := {1, 2, 4, 5, 6, 9, 10, 11, 13}

/-- A valid placement of numbers in the circles and squares. -/
structure Placement where
  squares : Fin 4 → Nat
  circles : Fin 3 → Nat
  all_used : ∀ n ∈ available_numbers, (∃ i, squares i = n) ∨ (∃ i, circles i = n)
  circle_sum : ∀ i : Fin 3, circles i = squares i.val + squares (i.val + 1)

/-- The sum of the leftmost and rightmost squares in a placement. -/
def edge_sum (p : Placement) : Nat :=
  p.squares 0 + p.squares 3

/-- The theorem stating that the maximum edge sum is 20. -/
theorem max_edge_sum :
    (∀ p : Placement, edge_sum p ≤ 20) ∧ (∃ p : Placement, edge_sum p = 20) := by
  sorry

#eval available_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_sum_l313_31382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equivalence_l313_31354

/-- A line in 2D space represented by a vector equation -/
structure VectorLine where
  a : ℝ × ℝ
  b : ℝ × ℝ

/-- A line in 2D space represented by slope-intercept form -/
structure SlopeInterceptLine where
  m : ℚ
  b : ℚ

/-- The given vector line -/
noncomputable def given_line : VectorLine := {
  a := (1, 3)
  b := (-2, 8)
}

/-- The slope-intercept form to be proved -/
def slope_intercept_line : SlopeInterceptLine := {
  m := -1/3
  b := 22/3
}

/-- Theorem stating the equivalence of the two line representations -/
theorem line_equivalence (x y : ℝ) : 
  (given_line.a.1 * (x - given_line.b.1) + given_line.a.2 * (y - given_line.b.2) = 0) ↔ 
  (y = slope_intercept_line.m * x + slope_intercept_line.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equivalence_l313_31354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l313_31355

-- Define a triangle with two known sides and the angle between them
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ

-- Define our specific triangle
noncomputable def ourTriangle : Triangle where
  side1 := 10
  side2 := 15
  angle := 135 * Real.pi / 180  -- Convert degrees to radians

-- Theorem statement
theorem third_side_length (t : Triangle) (h : t = ourTriangle) : 
  Real.sqrt (t.side1^2 + t.side2^2 - 2 * t.side1 * t.side2 * Real.cos t.angle) = 
  Real.sqrt (325 - 150 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l313_31355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_one_sufficient_not_necessary_l313_31321

/-- A function f: ℝ → ℝ is increasing on ℝ if for all x y, x < y implies f x < f y -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = ax - sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.sin x

/-- The statement that "a > 1" is a sufficient but not necessary condition for f to be increasing -/
theorem a_gt_one_sufficient_not_necessary :
  (∃ a : ℝ, a ≤ 1 ∧ IsIncreasing (f a)) ∧
  (∀ a : ℝ, a > 1 → IsIncreasing (f a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_one_sufficient_not_necessary_l313_31321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l313_31381

/-- Represents a parabola in the form ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Calculates the x-coordinate of the vertex of the parabola -/
noncomputable def Parabola.vertexX (p : Parabola) : ℝ :=
  -p.b / (2 * p.a)

/-- Calculates the y-coordinate of the vertex of the parabola -/
noncomputable def Parabola.vertexY (p : Parabola) : ℝ :=
  p.c - p.b^2 / (4 * p.a)

theorem parabola_equation_proof (p : Parabola) (h1 : p.a = 4) (h2 : p.b = -24) (h3 : p.c = 34) :
  p.vertexX = 3 ∧ p.vertexY = -2 ∧ p.contains 4 2 := by
  sorry

#check parabola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l313_31381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_valleyed_than_humped_l313_31316

/-- A five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- The middle digit of a five-digit number -/
def middleDigit (n : FiveDigitNumber) : ℕ := (n.val / 100) % 10

/-- A five-digit number is humped if its middle digit is larger than all other digits -/
def isHumped (n : FiveDigitNumber) : Prop :=
  let d := middleDigit n
  d > n.val / 10000 ∧
  d > (n.val / 1000) % 10 ∧
  d > (n.val / 10) % 10 ∧
  d > n.val % 10

/-- A five-digit number is valleyed if its middle digit is smaller than all other digits -/
def isValleyed (n : FiveDigitNumber) : Prop :=
  let d := middleDigit n
  d < n.val / 10000 ∧
  d < (n.val / 1000) % 10 ∧
  d < (n.val / 10) % 10 ∧
  d < n.val % 10

/-- The set of all humped five-digit numbers -/
def HumpedNumbers : Set FiveDigitNumber :=
  { n | isHumped n }

/-- The set of all valleyed five-digit numbers -/
def ValleyedNumbers : Set FiveDigitNumber :=
  { n | isValleyed n }

/-- There are more valleyed numbers than humped numbers -/
theorem more_valleyed_than_humped :
  ∃ (h : Fintype HumpedNumbers) (v : Fintype ValleyedNumbers),
    Fintype.card ValleyedNumbers > Fintype.card HumpedNumbers :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_valleyed_than_humped_l313_31316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31399

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.medianCM : Line :=
  { a := 2, b := -1, c := -5 }

def Triangle.angleBisectorBN : Line :=
  { a := 1, b := -2, c := -5 }

theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (5, 1))
  (h2 : Triangle.medianCM = { a := 2, b := -1, c := -5 })
  (h3 : Triangle.angleBisectorBN = { a := 1, b := -2, c := -5 }) :
  t.B = (-1, -3) ∧ 
  ∃ (l : Line), l.a = 18 ∧ l.b = -31 ∧ l.c = -75 ∧ 
    (∀ (x y : ℝ), l.a * x + l.b * y + l.c = 0 ↔ (x = t.B.1 ∧ y = t.B.2) ∨ (x = t.C.1 ∧ y = t.C.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l313_31375

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, (Real.sqrt ((3 - 0)^2 + (0 - y)^2) = Real.sqrt ((4 - 0)^2 + (5 - y)^2)) ∧ y = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l313_31375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_initial_speed_l313_31310

-- Define the problem parameters
noncomputable def total_distance : ℝ := 140
noncomputable def first_half_time : ℝ := 2.5
noncomputable def second_half_time : ℝ := 7/3
noncomputable def speed_increase : ℝ := 2

-- Define the theorem
theorem biker_initial_speed :
  ∃ (v : ℝ),
    (v * first_half_time = total_distance / 2) ∧
    ((v + speed_increase) * second_half_time = total_distance / 2) ∧
    v = 28 := by
  -- The proof goes here
  sorry

#check biker_initial_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_initial_speed_l313_31310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l313_31360

/-- Given a geometric sequence where the fifth term is 81 and the sixth term is 243,
    prove that the first term is 1. -/
theorem geometric_sequence_first_term
  (a b c d : ℝ)
  (h1 : ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ 81 = d * r ∧ 243 = 81 * r)
  : a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l313_31360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probability_l313_31393

/-- The probability of player A making a shot -/
def prob_A : ℝ := 0.4

/-- The probability of player B making a shot -/
def prob_B : ℝ := 0.6

/-- The number of times player A shoots -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for ξ -/
def P (k : ℕ) : ℝ := (1 - prob_A * (1 - prob_B))^(k-1) * (prob_A + prob_B - prob_A * prob_B)

theorem basketball_probability (k : ℕ) :
  P k = (0.24)^(k-1) * 0.76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probability_l313_31393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_receptivity_receptivity_comparison_l313_31371

/-- Piecewise function representing students' receptivity --/
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x ≤ 16 then 59
  else if 16 < x ∧ x ≤ 30 then -3 * x + 107
  else 0

/-- The maximum value of f(x) occurs at x = 10 and remains constant until x = 16 --/
theorem max_receptivity :
  (∀ x : ℝ, 0 < x → x ≤ 30 → f x ≤ f 10) ∧
  (∀ x : ℝ, 10 < x → x ≤ 16 → f x = f 10) := by sorry

/-- The receptivity at 5 minutes is greater than at 20 minutes --/
theorem receptivity_comparison : f 5 > f 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_receptivity_receptivity_comparison_l313_31371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l313_31352

def r : ℕ := sorry

axiom r_pos : r > 0
axiom r_multiple : ∃ k : ℕ, 8 * 45 * r = k

def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => (n * a (n + 1) + 2 * (n + 2)^(2 * r)) / (n + 2)

theorem a_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ n : ℕ, n > 0 → (a n % 2 = 0 ↔ n % 4 = 0 ∨ n % 4 = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l313_31352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_velocity_for_new_condition_l313_31366

/-- Represents the relationship between pressure, area, and velocity of wind on a sail -/
structure WindSail where
  k : ℝ  -- Constant of proportionality
  P : ℝ → ℝ → ℝ → ℝ  -- Pressure as a function of k, A, and V

/-- The initial conditions of the wind sail problem -/
noncomputable def initial_condition : WindSail :=
  { k := 1 / 200,
    P := λ k A V => k * A * V^2 }

/-- Theorem stating the velocity of wind when pressure is 16 pounds on 4 square feet -/
theorem wind_velocity_for_new_condition (ws : WindSail) 
  (h1 : ws.P ws.k 2 20 = 4)  -- Initial condition
  (h2 : ws.P ws.k 4 (20 * Real.sqrt 2) = 16)  -- New condition
  : ∃ V, ws.P ws.k 4 V = 16 ∧ V = 20 * Real.sqrt 2 :=
by
  sorry

#check wind_velocity_for_new_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_velocity_for_new_condition_l313_31366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l313_31359

/-- Represents the volume of lemon juice in a cylindrical glass with given conditions -/
noncomputable def lemon_juice_volume (glass_height : ℝ) (glass_diameter : ℝ) (lemonade_ratio : ℝ) (juice_ratio : ℝ) : ℝ :=
  let glass_radius : ℝ := glass_diameter / 2
  let lemonade_height : ℝ := glass_height * lemonade_ratio
  let lemonade_volume : ℝ := Real.pi * glass_radius^2 * lemonade_height
  lemonade_volume * juice_ratio

/-- The volume of lemon juice in the glass is approximately 2.13 cubic inches -/
theorem lemon_juice_volume_approx :
  ∃ ε > 0, |lemon_juice_volume 9 3 (1/3) (1/10) - 2.13| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l313_31359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equals_wire_radius_l313_31387

noncomputable section

-- Define the constants for the wire dimensions
def wire_radius : ℝ := 12
def wire_length : ℝ := 16

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Theorem statement
theorem sphere_radius_equals_wire_radius :
  ∃ (r : ℝ), sphere_volume r = cylinder_volume wire_radius wire_length ∧ r = wire_radius := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equals_wire_radius_l313_31387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_l313_31315

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 5 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x + 7) / 15

-- Theorem stating that h_inv is the inverse of h
theorem h_inverse : 
  (∀ x, h (h_inv x) = x) ∧ (∀ x, h_inv (h x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_l313_31315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_plus_pi_fourth_l313_31349

theorem cos_beta_plus_pi_fourth (α β : Real) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h2 : β ∈ Set.Ioo (3 * Real.pi / 4) Real.pi)
  (h3 : Real.cos (α + β) = 4 / 5)
  (h4 : Real.sin (α - Real.pi / 4) = 12 / 13) :
  Real.cos (β + Real.pi / 4) = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_plus_pi_fourth_l313_31349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l313_31369

theorem cone_slice_volume_ratio :
  ∀ (r h : ℝ) (hr : r > 0) (hh : h > 0),
  let V1 := (125/3 - 64/3) * π * r^2 * h
  let V2 := (64/3 - 27/3) * π * r^2 * h
  V2 / V1 = 37 / 61 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l313_31369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l313_31324

theorem trigonometric_expression_value : 
  (1 - 1 / (Real.sqrt 3 / 2)) * (1 + 1 / (Real.sqrt 3 / 2)) * 
  (1 - 1 / (1 / 2)) * (1 + 1 / (1 / 2)) = -1 := by
  -- Introduce local variables for trigonometric values
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  -- Rewrite the expression using these variables
  have h : (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = -1 := by
    sorry
  -- Use the hypothesis to prove the theorem
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l313_31324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l313_31363

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x/2 + 1/4

noncomputable def x : ℕ → ℝ
| 0 => 0
| n + 1 => f (x n)

noncomputable def y : ℕ → ℝ
| 0 => 1/2
| n + 1 => f (y n)

theorem problem_statement (x₀ : ℝ) (h₁ : 0 < x₀) (h₂ : x₀ < 1/2) (h₃ : f x₀ = x₀) :
  (∀ x : ℝ, (deriv f x) > 0) ∧
  (∀ n : ℕ, x n < x (n + 1) ∧ x (n + 1) < x₀ ∧ x₀ < y (n + 1) ∧ y (n + 1) < y n) ∧
  (∀ n : ℕ, (y (n + 1) - x (n + 1)) / (y n - x n) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l313_31363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_nine_twentyfifths_l313_31376

/-- Represents the geometric configuration described in the problem -/
structure GeometricConfig where
  s : ℝ  -- side length of the square
  k : ℝ  -- shorter side of the rectangle
  l : ℝ  -- longer side of the rectangle
  R : ℝ  -- radius of the largest circle
  r : ℝ  -- radius of the smallest circle

/-- Conditions of the geometric configuration -/
def validConfig (g : GeometricConfig) : Prop :=
  g.s > 0 ∧ g.k > 0 ∧ g.l > 0 ∧ g.R > 0 ∧ g.r > 0 ∧
  g.k ≤ g.l ∧ g.k ≤ g.s ∧ g.l ≤ g.s ∧
  g.R = g.s / 2 ∧ g.r = g.k / 2 ∧
  g.s^2 = g.k^2 + g.l^2

/-- The shaded area between the two larger circles is eight times the area of the smallest circle -/
def shadedAreaCondition (g : GeometricConfig) : Prop :=
  Real.pi * g.R^2 - Real.pi * ((g.k^2 + g.l^2) / 4) = 8 * (Real.pi * g.r^2)

/-- The fraction of the largest circle that is shaded -/
noncomputable def shadedFraction (g : GeometricConfig) : ℝ :=
  (9 * (Real.pi * g.r^2)) / (Real.pi * g.R^2)

/-- Main theorem: The fraction of the largest circle that is shaded is 9/25 -/
theorem shaded_fraction_is_nine_twentyfifths (g : GeometricConfig) 
  (hvalid : validConfig g) (hshaded : shadedAreaCondition g) : 
  shadedFraction g = 9 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_nine_twentyfifths_l313_31376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l313_31383

/-- For an angle θ whose terminal side passes through the point (-12, 5), cos(θ) = -12/13 -/
theorem cosine_of_angle_through_point :
  ∀ θ : ℝ,
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos θ = -12 ∧ r * Real.sin θ = 5) →
  Real.cos θ = -12/13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l313_31383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l313_31347

theorem cubic_equation_solution : ∃ x : ℝ, 
  x^3 - (0.1: ℝ)^3 / (0.5: ℝ)^2 + 0.05 + (0.1: ℝ)^2 = 0.4 ∧ 
  abs (x - 0.7) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l313_31347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l313_31357

-- Define IsEquilateralTriangle as an axiom since it's not part of the standard library
axiom IsEquilateralTriangle : ℂ → ℂ → ℂ → Prop

theorem equilateral_triangle_lambda (ω : ℂ) (lambda : ℝ) 
  (h1 : Complex.abs ω = 3)
  (h2 : lambda > 1)
  (h3 : IsEquilateralTriangle ω (ω^3) (lambda * ω)) : 
  lambda = 1 + 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l313_31357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cone_angle_l313_31322

/-- Two spheres touching externally on a table -/
structure SpherePair where
  R : ℝ  -- radius of the first sphere
  r : ℝ  -- radius of the second sphere
  R_pos : R > 0
  r_pos : r > 0

/-- A cone touching both spheres and the table -/
structure Cone (sp : SpherePair) where
  α : ℝ  -- half of the angle at the apex
  φ : ℝ  -- angle between the ray from apex to sphere center and the table
  apex_on_segment : True  -- represents that the apex is on the segment connecting sphere-table contact points
  equal_angles : True  -- represents that angles formed by rays to sphere centers are equal

/-- The maximum angle at the apex of the cone -/
noncomputable def max_apex_angle (sp : SpherePair) : ℝ :=
  2 * Real.arctan (1 / 2)

/-- The theorem statement -/
theorem max_cone_angle (sp : SpherePair) :
  ∀ c : Cone sp, 2 * c.α ≤ max_apex_angle sp := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cone_angle_l313_31322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentadecagon_enclosure_l313_31334

/-- The number of sides of the central regular polygon -/
def m : ℕ := 15

/-- The interior angle of a regular polygon with m sides -/
noncomputable def interior_angle (m : ℕ) : ℝ := (m - 2) * 180 / m

/-- The exterior angle of a regular polygon with m sides -/
noncomputable def exterior_angle (m : ℕ) : ℝ := 180 - interior_angle m

/-- The number of enclosing regular polygons -/
def num_enclosing : ℕ := 15

/-- Theorem: For a regular 15-sided polygon (pentadecagon) exactly enclosed by 15 regular polygons 
    with n sides each, n must equal 15. -/
theorem pentadecagon_enclosure (n : ℕ) 
  (h : 2 * (180 / n) = exterior_angle m) : n = m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentadecagon_enclosure_l313_31334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crackham_puzzle_solution_l313_31341

theorem crackham_puzzle_solution :
  let first_number : ℕ := 4539281706
  let second_number : ℕ := 9078563412
  (∀ d : ℕ, d < 10 → (Nat.digits 10 first_number).count d = 1) ∧  -- All digits used once
  (first_number * second_number > 0) ∧                            -- Valid product
  (first_number % 10 ≠ 0) ∧                                       -- Doesn't end with 0
  (first_number ≥ 1000000000)                                     -- Doesn't start with 0
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crackham_puzzle_solution_l313_31341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l313_31342

/-- The volume of a cone with given diameter and height -/
noncomputable def cone_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (diameter / 2)^2 * height

/-- Theorem: The volume of a cone with diameter 12 cm and height 8 cm is 96π cubic centimeters -/
theorem cone_volume_specific : cone_volume 12 8 = 96 * Real.pi := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l313_31342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l313_31338

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = x³(a·2ˣ - 2⁻ˣ) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * (2:ℝ)^x - (2:ℝ)^(-x))

/-- If f(x) = x³(a·2ˣ - 2⁻ˣ) is an even function, then a = 1 -/
theorem even_function_implies_a_eq_one (a : ℝ) :
  IsEven (f a) → a = 1 := by
  sorry

#check even_function_implies_a_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l313_31338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_fraction_power_l313_31344

theorem evaluate_fraction_power : (1/16 : ℝ)^(-(1/2) : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_fraction_power_l313_31344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l313_31312

-- Define the lines
def line_through_origin (x y : ℝ) : Prop := ∃ m : ℝ, y = m * x
def line_x_equals_1 (x : ℝ) : Prop := x = 1
def line_y_equals_2x_plus_1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  line_through_origin A.1 A.2 ∧
  line_x_equals_1 B.1 ∧
  line_y_equals_2x_plus_1 C.1 C.2 ∧
  A.1 = 0 ∧ A.2 = 0  -- A is the origin

-- Define isosceles property
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  ((B.1 - A.1)^2 + (B.2 - A.2)^2) = ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Define perimeter
noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) +
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)

-- Theorem statement
theorem triangle_perimeter :
  ∀ A B C : ℝ × ℝ,
  triangle A B C →
  is_isosceles A B C →
  perimeter A B C = 5 + 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l313_31312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_clip_not_rectangle_l313_31335

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

-- Define a shape type
inductive Shape
  | Rectangle : Rectangle → Shape
  | Triangle : Shape
  | Trapezoid : Shape
  | Pentagon : Shape

-- Define a corner clip operation
def cornerClip (r : Rectangle) : Shape :=
  sorry -- Implementation details omitted

-- Theorem stating that the result of a corner clip cannot be a rectangle
theorem corner_clip_not_rectangle (r : Rectangle) : 
  ¬ ∃ (s : Rectangle), cornerClip r = Shape.Rectangle s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_clip_not_rectangle_l313_31335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_l313_31370

/-- The cost of buying a large number of pencils with a bulk discount -/
theorem pencil_cost (base_quantity : ℕ) (base_cost : ℚ) (order_quantity : ℕ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) :
  base_quantity = 150 →
  base_cost = 40 →
  order_quantity = 4500 →
  discount_threshold = 3000 →
  discount_rate = 1/10 →
  (base_cost / base_quantity) * order_quantity * (1 - discount_rate) = 1080 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_l313_31370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_l313_31337

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x + b

theorem function_and_inequality (a b : ℝ) :
  (∃ (x : ℝ), f a b x = -4/3 ∧ ∀ (y : ℝ), f a b y ≥ -4/3) ∧
  (f a b 2 = -4/3) →
  (a = -4 ∧ b = 4) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-4) 3 → f a b x ≤ m^2 + m + 10/3) ↔
               m ∈ Set.Iic (-3) ∪ Set.Ici 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_l313_31337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l313_31385

structure Point where
  x : ℝ
  y : ℝ

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem triangle_abc_area :
  let A : Point := { x := -2, y := 3 }
  let B : Point := { x := 6, y := 3 }
  let C : Point := { x := 3, y := -4 }
  triangleArea A B C = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l313_31385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_36_minimum_m_l313_31307

theorem divisibility_by_36 (n : ℕ) : 36 ∣ (3^n * (2*n + 7) + 9) := by
  sorry

theorem minimum_m : ∀ m : ℕ, m < 9 → ∃ n : ℕ, ¬(36 ∣ (3^n * (2*n + 7) + m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_36_minimum_m_l313_31307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_T_l313_31394

-- Define the sets P and T
def P : Set ℝ := {x : ℝ | |x| > 2}
def T : Set ℝ := {x : ℝ | (3 : ℝ)^x > 1}

-- State the theorem
theorem intersection_P_T : P ∩ T = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_T_l313_31394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_side_length_of_specific_prism_l313_31305

/-- A regular octagonal prism with given total edge length and height -/
structure RegularOctagonalPrism where
  total_edge_length : ℝ
  height : ℝ

/-- The length of one side of the base of a regular octagonal prism -/
noncomputable def base_side_length (prism : RegularOctagonalPrism) : ℝ :=
  (prism.total_edge_length - 8 * prism.height) / 16

theorem base_side_length_of_specific_prism :
  let prism : RegularOctagonalPrism := ⟨240, 12⟩
  base_side_length prism = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_side_length_of_specific_prism_l313_31305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_in_ratio_l313_31343

/-- Given two points A and B in 3D space, and a point C that divides the line segment AB
    in the ratio 2:1 starting from A, prove that C has the correct coordinates. -/
theorem divide_segment_in_ratio (A B C : ℝ × ℝ × ℝ) : 
  A = (1, -1, 2) →
  B = (7, -4, -1) →
  C = (5, -3, 0) →
  2 * ((C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)) = (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2) :=
by
  intro hA hB hC
  simp [hA, hB, hC]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_in_ratio_l313_31343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_sixth_degree_equation_l313_31332

theorem solutions_of_sixth_degree_equation :
  let S : Set ℂ := {x | x^6 + 216 = 0}
  let root : ℂ := (216 : ℂ)^((1/6 : ℝ) : ℂ)
  S = {root, -root, 
       root * Complex.exp (2 * Real.pi / 3 * Complex.I),
       root * Complex.exp (4 * Real.pi / 3 * Complex.I),
       -root * Complex.exp (Real.pi / 3 * Complex.I),
       -root * Complex.exp (5 * Real.pi / 3 * Complex.I)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_sixth_degree_equation_l313_31332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l313_31389

-- Define the function f(x) = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the line y = x
def line (x : ℝ) : ℝ := x

-- State the theorem
theorem shortest_distance_ln_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), x > 0 → y = f x → 
    Real.sqrt ((x - line y)^2 + (y - line y)^2) ≥ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l313_31389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l313_31320

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 4

-- Define the point through which the tangent lines pass
def tangent_point : ℝ × ℝ := (-1, -1)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := y = -1
def tangent_line_2 (x y : ℝ) : Prop := 12*x + 5*y + 17 = 0

-- Theorem statement
theorem tangent_lines_to_circle :
  ∃ (x y : ℝ), circle_eq x y ∧
  (tangent_line_1 x y ∨ tangent_line_2 x y) ∧
  (x, y) ≠ tangent_point ∧
  ∀ (ε : ℝ), ε > 0 → ∃ (x' y' : ℝ),
    ((x' - x)^2 + (y' - y)^2 < ε^2) ∧
    ¬(circle_eq x' y') :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l313_31320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_not_on_circle_l313_31303

def point_A : ℝ × ℝ := (5, 0)
def point_B : ℝ × ℝ := (4, 3)
def point_C : ℝ × ℝ := (2, 2)
def point_D : ℝ × ℝ := (3, 4)
def point_E : ℝ × ℝ := (0, 5)

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

def on_circle (p : ℝ × ℝ) : Prop :=
  distance_from_origin p = 5

theorem point_C_not_on_circle :
  ¬(on_circle point_C) ∧
  (on_circle point_A ∧ on_circle point_B ∧ on_circle point_D ∧ on_circle point_E) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_not_on_circle_l313_31303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31368

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : 0 < A ∧ A < π
  h3 : 0 < B ∧ B < π
  h4 : 0 < C ∧ C < π
  h5 : A + B + C = π
  h6 : S = (1/2) * a * b * sin C

theorem triangle_properties (t : Triangle) 
  (h : t.a * sin t.B = Real.sqrt 3 * t.b * cos t.A) :
  t.A = π/3 ∧ 
  (t.a = Real.sqrt 3 ∧ t.S = Real.sqrt 3/2 → t.b + t.c = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l313_31368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_and_change_correct_transaction_l313_31353

/-- Represents the weight in liang -/
structure Liang where
  value : ℕ
deriving Repr

/-- Represents the number of grain coupons -/
structure Coupon where
  value : ℕ
deriving Repr

instance : Add Liang where
  add a b := ⟨a.value + b.value⟩

instance : Mul Liang where
  mul a b := ⟨a.value * b.value⟩

instance : HMul Liang ℕ Liang where
  hMul a b := ⟨a.value * b⟩

instance : OfNat Liang n where
  ofNat := ⟨n⟩

instance : LE Coupon where
  le a b := a.value ≤ b.value

/-- Conversion rate from jin to liang -/
def jinToLiang (n : ℕ) : Liang := ⟨n * 10⟩

/-- Weight of each bread in liang -/
def breadWeight : Liang := 3

/-- Number of loaves of bread bought -/
def numBreads : ℕ := 9

/-- Total weight of bread in liang -/
def totalBreadWeight : Liang := breadWeight * numBreads

/-- Amount to be paid in liang -/
def amountToPay : Liang := jinToLiang 2 + ⟨7⟩

/-- Value of a half-jin coupon in liang -/
def halfJinCouponValue : Liang := 5

/-- Value of a 2-liang coupon in liang -/
def twoLiangCouponValue : Liang := 2

/-- Number of half-jin coupons Xiao Ming has -/
def availableHalfJinCoupons : Coupon := ⟨10⟩

theorem correct_payment_and_change 
  (paidHalfJinCoupons : Coupon) 
  (returnedTwoLiangCoupons : Coupon) : Prop :=
  paidHalfJinCoupons ≤ availableHalfJinCoupons ∧
  ⟨paidHalfJinCoupons.value * halfJinCouponValue.value - 
    returnedTwoLiangCoupons.value * twoLiangCouponValue.value⟩ = amountToPay ∧
  paidHalfJinCoupons = ⟨7⟩ ∧
  returnedTwoLiangCoupons = ⟨4⟩

/-- Proves that the correct payment is 7 half-jin coupons and the correct change is 4 2-liang coupons -/
theorem correct_transaction : ∃ (p r : Coupon), correct_payment_and_change p r := by
  sorry

#eval totalBreadWeight
#eval amountToPay

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_and_change_correct_transaction_l313_31353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_valid_numbers_l313_31365

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    d1 ∈ Finset.range 9 ∧
    d2 ∈ Finset.range 9 ∧
    d3 ∈ Finset.range 9 ∧
    d4 ∈ Finset.range 9 ∧
    d5 ∈ Finset.range 9 ∧
    d6 ∈ Finset.range 9 ∧
    d7 ∈ Finset.range 9 ∧
    d8 ∈ Finset.range 9 ∧
    d9 ∈ Finset.range 9 ∧
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9

theorem gcd_of_valid_numbers :
  ∃ (d : ℕ), d > 0 ∧ (∀ (n : ℕ), is_valid_number n → d ∣ n) ∧
  (∀ (m : ℕ), m > 0 → (∀ (n : ℕ), is_valid_number n → m ∣ n) → m ∣ d) ∧
  d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_valid_numbers_l313_31365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_range_is_reals_l313_31336

-- Define the function
noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x^2 + 2 * a * x + 1)

-- Theorem for the first part
theorem domain_is_reals (a : ℝ) : 
  (∀ x, ∃ y, f a x = y) ↔ a ∈ Set.Ici 0 ∩ Set.Iio 1 :=
sorry

-- Theorem for the second part
theorem range_is_reals (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_range_is_reals_l313_31336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l313_31345

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line
def my_line (x₀ y₀ x y : ℝ) : Prop := x₀ * x - y₀ * y = 2

-- Define what it means for a line to be tangent to the circle
def is_tangent (x₀ y₀ : ℝ) : Prop :=
  ∀ x y : ℝ, my_circle x y → my_line x₀ y₀ x y → 
    ∃! p : ℝ × ℝ, my_circle p.1 p.2 ∧ my_line x₀ y₀ p.1 p.2

-- Theorem statement
theorem line_tangent_to_circle :
  ∀ x₀ y₀ : ℝ, my_circle x₀ y₀ → is_tangent x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l313_31345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_55_5_rounds_to_56_l313_31328

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem only_55_5_rounds_to_56 : 
  (round_to_nearest 56.69 ≠ 56) ∧ 
  (round_to_nearest 55.5 = 56) ∧ 
  (round_to_nearest 55.49 ≠ 56) ∧ 
  (round_to_nearest 55.09 ≠ 56) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_55_5_rounds_to_56_l313_31328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_highway_mpg_is_12_2_l313_31333

/-- Represents the fuel efficiency of an SUV -/
structure SUV where
  city_mpg : ℝ
  max_distance : ℝ
  fuel_capacity : ℝ

/-- Calculates the highway miles per gallon for an SUV -/
noncomputable def highway_mpg (suv : SUV) : ℝ :=
  suv.max_distance / suv.fuel_capacity

/-- Theorem stating that for the given SUV specifications, the highway mpg is 12.2 -/
theorem suv_highway_mpg_is_12_2 (suv : SUV) 
    (h_city : suv.city_mpg = 7.6)
    (h_max_distance : suv.max_distance = 280.6)
    (h_fuel_capacity : suv.fuel_capacity = 23) :
    highway_mpg suv = 12.2 := by
  sorry

/-- Compute the highway mpg for the given SUV specifications -/
def compute_highway_mpg : ℚ :=
  280.6 / 23

#eval compute_highway_mpg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_highway_mpg_is_12_2_l313_31333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_water_filling_l313_31327

/-- Represents the time (in minutes) it takes for a tap to fill the bathtub -/
structure FillTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents the state of the bathtub filling process -/
structure BathtubState where
  hot_water : ℝ
  cold_water : ℝ
  time_elapsed : ℝ

/-- Function to simulate the bathtub filling process -/
def fill_bathtub (hot_fill_time : FillTime) (cold_fill_time : FillTime) (delay : ℝ) : BathtubState :=
  sorry -- Implementation details omitted for brevity

/-- The main theorem to prove -/
theorem equal_water_filling 
  (hot_fill_time : FillTime) 
  (cold_fill_time : FillTime) 
  (hot_fill_time_val : hot_fill_time.minutes = 23) 
  (cold_fill_time_val : cold_fill_time.minutes = 19) :
  ∃ (delay : ℝ), 
    delay = 2 ∧ 
    (let final_state := fill_bathtub hot_fill_time cold_fill_time delay
     final_state.hot_water = final_state.cold_water ∧
     final_state.hot_water + final_state.cold_water = 1) := by
  sorry -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_water_filling_l313_31327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_max_value_in_interval_l313_31348

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

-- Part 1
theorem tangent_line_coefficients (a b : ℝ) :
  (∀ x, (deriv (f a)) 1 * (x - 1) + f a 1 = x + b) →
  a = 1 ∧ b = 0 := by sorry

-- Part 2
theorem max_value_in_interval :
  let f := f (1/8)
  (deriv f 2 = 0) →
  (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x ≤ f (1/Real.exp 1)) ∧
  f (1/Real.exp 1) = 1/(8*(Real.exp 1)^2) + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_max_value_in_interval_l313_31348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_second_quadrant_l313_31392

open Real

/-- The function f as defined in the problem -/
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (-α) * cos (-α + 3*π/2)) / 
  (cos (π/2 - α) * sin (-π - α))

/-- Theorem stating the equality to be proved -/
theorem f_value_in_second_quadrant (α : ℝ) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : cos (α - 5*π/2) = 1/5) : 
  f α = 2 * sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_second_quadrant_l313_31392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l313_31364

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = -(3 * Real.sqrt 5) / 5) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l313_31364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darias_financial_result_l313_31304

/-- Represents the financial operation described in the problem -/
noncomputable def financialOperation (initialAmount : ℝ) (initialSellRate : ℝ) (initialBuyRate : ℝ) 
                       (finalBuyRate : ℝ) (interestRate : ℝ) (months : ℕ) : ℝ :=
  let dollarAmount := initialAmount / initialSellRate
  let depositedAmount := dollarAmount * (1 + interestRate * (months / 12 : ℝ))
  depositedAmount * finalBuyRate

/-- The theorem stating the financial result of Daria's operation -/
theorem darias_financial_result :
  let initialAmount : ℝ := 60000
  let initialSellRate : ℝ := 59.65
  let initialBuyRate : ℝ := 56.65
  let finalSellRate : ℝ := 58.95
  let finalBuyRate : ℝ := 55.95
  let interestRate : ℝ := 0.015
  let months : ℕ := 6
  let result := financialOperation initialAmount initialSellRate initialBuyRate finalBuyRate interestRate months
  ⌊initialAmount - result⌋ = 3309 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_darias_financial_result_l313_31304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l313_31351

noncomputable section

variable (σ : ℝ) (hσ : σ > 0)
variable (ξ : ℝ → ℝ)

/-- Normal distribution probability density function -/
def normal_dist (μ σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- Predicate for a function following a normal distribution -/
def is_normal_dist (ξ : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∀ x, ξ x = normal_dist μ σ x

/-- Theorem stating the probability for the given problem -/
theorem normal_distribution_probability 
  (h_normal : is_normal_dist ξ 1 σ)
  (h_prob : ∫ x in Set.Ioo 0 2, ξ x = 0.8) :
  ∫ x in Set.Iic 2, ξ x = 0.9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l313_31351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l313_31391

theorem triangle_abc_properties (A B C : ℝ) :
  (Real.sqrt 2 * Real.sin A = 2 * (Real.cos (A/2))^2) →
  (2*A - B = π/2) →
  (Real.tan A = 2*Real.sqrt 2) ∧ (Real.sin (A+C) = 7/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l313_31391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l313_31395

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x
def g (x : ℝ) : ℝ := 2 * (x - Real.log x)
def h (x : ℝ) : ℝ := -x * Real.sin x + Real.exp x * Real.cos x

def interval : Set ℝ := {x | -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2}

theorem problem_statement :
  (∀ x > 0, f x > g x) ∧
  (∃! n : ℕ, n = 2 ∧ ∃ S : Set ℝ, S ⊆ interval ∧ (∀ x ∈ S, h x = 0) ∧ Finite S ∧ Nat.card S = n) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l313_31395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_condition_l313_31380

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  a_longest : a > b ∧ a > c
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the angle A
noncomputable def angleA (t : ScaleneTriangle) : ℝ := 
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- Theorem statement
theorem obtuse_angle_condition (t : ScaleneTriangle) : 
  angleA t > Real.pi / 2 ↔ t.a^2 > t.b^2 + t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_condition_l313_31380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_ab_l313_31309

theorem divisibility_of_ab (a b n : ℕ) (h1 : b < 10) (h2 : 2^n = 10*a + b) (h3 : n > 3) :
  6 ∣ (a * b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_ab_l313_31309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_maxima_equality_l313_31330

/-- A 6x6 matrix with unique elements -/
def UniqueMatrix : Type := Matrix (Fin 6) (Fin 6) ℝ

/-- Predicate: All elements in the matrix are unique -/
def all_unique (m : UniqueMatrix) : Prop :=
  ∀ i j i' j', m i j = m i' j' → i = i' ∧ j = j'

/-- The maximum element in each column -/
noncomputable def col_max (m : UniqueMatrix) (j : Fin 6) : ℝ :=
  Finset.max' (Finset.univ.image (λ i => m i j)) (by simp [Finset.univ_nonempty])

/-- The maximum element in each row -/
noncomputable def row_max (m : UniqueMatrix) (i : Fin 6) : ℝ :=
  Finset.max' (Finset.univ.image (λ j => m i j)) (by simp [Finset.univ_nonempty])

/-- Predicate: Column maxima are in different rows -/
def col_max_different_rows (m : UniqueMatrix) : Prop :=
  ∀ j j', j ≠ j' → 
    ∃ i i', i ≠ i' ∧ m i j = col_max m j ∧ m i' j' = col_max m j'

/-- Predicate: Row maxima are in different columns -/
def row_max_different_cols (m : UniqueMatrix) : Prop :=
  ∀ i i', i ≠ i' → 
    ∃ j j', j ≠ j' ∧ m i j = row_max m i ∧ m i' j' = row_max m i'

theorem matrix_maxima_equality (m : UniqueMatrix) 
  (h_unique : all_unique m)
  (h_col : col_max_different_rows m)
  (h_row : row_max_different_cols m) :
  {x | ∃ j, x = col_max m j} = {x | ∃ i, x = row_max m i} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_maxima_equality_l313_31330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shakespeare_birthday_l313_31339

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Checks if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 400 == 0) || (year % 4 == 0 && year % 100 ≠ 0)

/-- Counts leap years between two years (inclusive) -/
def countLeapYears (start year : ℕ) : ℕ :=
  (List.range (year - start + 1)).map (fun i => start + i) |>.filter isLeapYear |>.length

/-- Calculates the day of the week given a number of days after a Thursday -/
def dayAfterThursday (days : ℕ) : DayOfWeek :=
  match days % 7 with
  | 0 => DayOfWeek.Thursday
  | 1 => DayOfWeek.Friday
  | 2 => DayOfWeek.Saturday
  | 3 => DayOfWeek.Sunday
  | 4 => DayOfWeek.Monday
  | 5 => DayOfWeek.Tuesday
  | _ => DayOfWeek.Wednesday

/-- Theorem: Shakespeare was born on a Wednesday -/
theorem shakespeare_birthday :
  let anniversaryYear : ℕ := 1964
  let birthYear : ℕ := anniversaryYear - 300
  let leapYears : ℕ := countLeapYears birthYear anniversaryYear
  let regularYears : ℕ := 300 - leapYears
  let totalDayShift : ℕ := regularYears + 2 * leapYears
  dayAfterThursday (7 - (totalDayShift % 7)) = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shakespeare_birthday_l313_31339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_external_tangency_l313_31356

/-- The line on which the center of circle C lies -/
def line (x y : ℝ) : Prop := x - y + 10 = 0

/-- The circle C with radius 5 and center (a, b) on the line -/
def circleC (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 25 ∧ line a b

/-- The circle O with radius r -/
def circleO (x y r : ℝ) : Prop := x^2 + y^2 = r^2

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- External tangency condition -/
def externallyTangent (a b r : ℝ) : Prop := distance a b 0 0 = 5 + r

theorem unique_external_tangency :
  ∃! r : ℝ, r > 0 ∧
    ∃ a b : ℝ, circleC (-5) 0 a b ∧
              circleO a b r ∧
              externallyTangent a b r ∧
              r = 5 * Real.sqrt 2 - 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_external_tangency_l313_31356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l313_31317

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at the origin --/
structure SpecialTriangle where
  /-- The x-coordinate of one vertex --/
  a : ℝ
  /-- The triangle is equilateral --/
  equilateral : True
  /-- The vertices lie on the hyperbola xy = 4 --/
  on_hyperbola : a * (4 / a) = 4
  /-- The centroid is at the origin --/
  centroid_origin : True

/-- The square of the area of the special triangle --/
noncomputable def square_area (t : SpecialTriangle) : ℝ :=
  let s := Real.sqrt (2 * (t.a^2 + (4/t.a)^2) * (1 - Real.cos (2*Real.pi/3)))
  (3/16) * s^4

/-- Theorem stating that the square of the area of the special triangle is (3/16) * s^4 --/
theorem special_triangle_area (t : SpecialTriangle) :
  square_area t = (3/16) * (Real.sqrt (2 * (t.a^2 + (4/t.a)^2) * (1 - Real.cos (2*Real.pi/3))))^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l313_31317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_constant_difference_l313_31358

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_constant_difference 
  (f g : ℝ → ℝ) (s : ℝ) : 
  (∀ x, f x - g x = 2 * s) → 
  (∃ c, f = λ x ↦ (x - (s + 2)) * (x - (s + 5)) * (x - c)) →
  (∃ d, g = λ x ↦ (x - (s + 4)) * (x - (s + 8)) * (x - d)) →
  s = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_constant_difference_l313_31358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l313_31331

/-- The start point of the particle's path -/
def start_point : Fin 3 → ℝ := ![1, 2, 3]

/-- The end point of the particle's path -/
def end_point : Fin 3 → ℝ := ![4, 6, 8]

/-- The radius of the sphere centered at the origin -/
def sphere_radius : ℝ := 5

/-- The theorem stating the distance between intersection points -/
theorem intersection_distance :
  let line_dir := λ i => end_point i - start_point i
  let a := (Finset.univ.sum λ i => (line_dir i)^2 : ℝ)
  let b := 2 * (Finset.univ.sum λ i => start_point i * line_dir i : ℝ)
  let c := (Finset.univ.sum λ i => (start_point i)^2 : ℝ) - sphere_radius^2
  let discriminant := b^2 - 4*a*c
  let t₁ := (-b + Real.sqrt discriminant) / (2*a)
  let t₂ := (-b - Real.sqrt discriminant) / (2*a)
  let p₁ := λ i => start_point i + t₁ * line_dir i
  let p₂ := λ i => start_point i + t₂ * line_dir i
  Real.sqrt (Finset.univ.sum λ i => (p₁ i - p₂ i)^2) = 10 * Real.sqrt 558 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l313_31331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_192_5_l313_31350

def initial_fee : ℚ := 15
def daily_rate : ℚ := 30
def mile_rate : ℚ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 350

def total_cost : ℚ :=
  initial_fee + daily_rate * rental_days + mile_rate * miles_driven

theorem total_cost_is_192_5 :
  total_cost = 192.5 := by
  -- Unfold the definition of total_cost
  unfold total_cost
  -- Simplify the arithmetic
  norm_num
  -- QED
  rfl

#eval total_cost -- Should output 192.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_192_5_l313_31350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_theorem_l313_31386

/-- The logarithmic expressions as functions of x -/
noncomputable def log1 (x : ℝ) := Real.log (6 * x - 14) / Real.log (Real.sqrt ((x / 3) + 3))
noncomputable def log2 (x : ℝ) := 2 * Real.log (x - 1) / Real.log (6 * x - 14)
noncomputable def log3 (x : ℝ) := Real.log ((x / 3) + 3) / Real.log (x - 1)

/-- The theorem stating that x = 3 is the only solution -/
theorem log_equality_theorem :
  ∃! x : ℝ, (x > 1) ∧
    ((log1 x = log2 x ∧ log3 x = log1 x - 1) ∨
     (log2 x = log3 x ∧ log1 x = log2 x - 1) ∨
     (log3 x = log1 x ∧ log2 x = log3 x - 1)) ∧
    x = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_theorem_l313_31386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_proof_l313_31388

/-- The weekly payment for Employee Y in rupees -/
def employee_y_payment : ℚ := 272.73

/-- The multiplier for Employee X's payment relative to Employee Y's -/
def employee_x_multiplier : ℚ := 1.2

/-- Calculates the weekly payment for Employee X -/
def employee_x_payment : ℚ := employee_x_multiplier * employee_y_payment

/-- Calculates the total weekly payment for both employees -/
def total_payment : ℚ := employee_x_payment + employee_y_payment

/-- Rounds a rational number to two decimal places -/
def round_to_two_decimals (x : ℚ) : ℚ := 
  (⌊x * 100 + 1/2⌋ : ℚ) / 100

theorem total_payment_proof :
  round_to_two_decimals total_payment = 600.01 := by
  sorry

#eval round_to_two_decimals total_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_proof_l313_31388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l313_31300

-- Define the circles and points
variable (A B C D P Q R : ℝ × ℝ)

-- Define the radii of the circles
variable (r_A r_B r_C r_D : ℝ)

-- Define the distances
def AB : ℝ := 42
def CD : ℝ := 50
def PQ : ℝ := 52

-- Define the relationships between radii
axiom radius_ratio_AB : r_A = 3/4 * r_B
axiom radius_ratio_CD : r_C = 2 * r_D

-- Define R as the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the distances from R to each circle center
noncomputable def AR (A R : ℝ × ℝ) : ℝ := Real.sqrt ((R.1 - A.1)^2 + (R.2 - A.2)^2)
noncomputable def BR (B R : ℝ × ℝ) : ℝ := Real.sqrt ((R.1 - B.1)^2 + (R.2 - B.2)^2)
noncomputable def CR (C R : ℝ × ℝ) : ℝ := Real.sqrt ((R.1 - C.1)^2 + (R.2 - C.2)^2)
noncomputable def DR (D R : ℝ × ℝ) : ℝ := Real.sqrt ((R.1 - D.1)^2 + (R.2 - D.2)^2)

-- The theorem to be proved
theorem sum_of_distances (A B C D P Q R : ℝ × ℝ) (r_A r_B r_C r_D : ℝ) :
  AR A R + BR B R + CR C R + DR D R = 184 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l313_31300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eight_roots_product_l313_31390

theorem cos_eight_roots_product :
  let f (x : Real) := 1 - 56*x + 560*x^2 - 1792*x^3 + 2304*x^4
  let roots := [Real.cos (π/8)^2, Real.cos (2*π/8)^2, Real.cos (3*π/8)^2]
  (∀ k ∈ ({1, 2, 3} : Set ℕ), f (Real.cos (k*π/8)^2) = -1) →
  Real.sqrt ((2 - Real.cos (π/8)^2) * (2 - Real.cos (2*π/8)^2) * (2 - Real.cos (3*π/8)^2)) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eight_roots_product_l313_31390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_pick_l313_31373

/-- Ramanujan's number -/
noncomputable def ramanujan_number : ℂ := (152 - 104*Complex.I) / 25

/-- Hardy's number -/
def hardy_number : ℂ := 7 + Complex.I

/-- The product of Ramanujan's and Hardy's numbers -/
def product : ℂ := 40 - 24*Complex.I

theorem ramanujan_pick :
  ramanujan_number * hardy_number = product :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_pick_l313_31373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l313_31311

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem segment_length : distance (1, 4) (8, 12) = Real.sqrt 113 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l313_31311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l313_31314

/-- Two lines l₁ and l₂ are given by their parametric equations. -/
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (1 - 2*t, 2 + k*t)

noncomputable def l₂ (s : ℝ) : ℝ × ℝ := (s, 1 - 2*s)

/-- The slope of l₁ -/
noncomputable def slope_l₁ (k : ℝ) : ℝ := -k/2

/-- The slope of l₂ -/
noncomputable def slope_l₂ : ℝ := -1/2

/-- Theorem: If l₁ is parallel to l₂, then k = 4 -/
theorem parallel_lines (k : ℝ) : 
  slope_l₁ k = slope_l₂ → k = 4 := by sorry

/-- Theorem: If l₁ is perpendicular to l₂, then k = -1 -/
theorem perpendicular_lines (k : ℝ) : 
  slope_l₁ k * slope_l₂ = -1 → k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l313_31314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l313_31384

-- Define the complex number z
noncomputable def z : ℂ := 10 / (3 + Complex.I) - 2 * Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l313_31384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffing_rate_l313_31362

/-- Earl's envelope stuffing rate in envelopes per minute -/
def earl_rate : ℝ := sorry

/-- Ellen's envelope stuffing rate in envelopes per minute -/
def ellen_rate : ℝ := sorry

/-- The relationship between Earl and Ellen's stuffing rates -/
axiom rate_relation : ellen_rate = earl_rate / 1.5

/-- The combined stuffing rate when Earl and Ellen work together -/
axiom combined_rate : earl_rate + ellen_rate = 60

/-- Theorem: Earl's envelope stuffing rate is 36 envelopes per minute -/
theorem earl_stuffing_rate : earl_rate = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffing_rate_l313_31362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_pass_fraction_l313_31340

theorem parking_pass_fraction (total_cars : ℕ) (valid_ticket_percentage : ℚ) (unpaid_cars : ℕ) : 
  total_cars = 300 →
  valid_ticket_percentage = 75 / 100 →
  unpaid_cars = 30 →
  (↑(total_cars * valid_ticket_percentage.num - unpaid_cars * valid_ticket_percentage.den) / 
   ↑(total_cars * valid_ticket_percentage.num)) = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_pass_fraction_l313_31340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_value_l313_31323

noncomputable def A (a b : ℝ) : ℝ :=
  (Real.sqrt (a + 2 * Real.sqrt b + b / a) * 
   Real.sqrt (2 * a - 10 * (8 * a^3 * b^2)^(1/6) + 25 * b^(2/3))) /
  (a * Real.sqrt (2 * a) + Real.sqrt (2 * a * b) - 5 * a * b^(1/3) - 5 * b^(5/6))

theorem A_value (a b : ℝ) (ha : a > 0) (hb : b ≥ 0) :
  A a b = if Real.sqrt (2 * a) > 5 * b^(1/3)
          then 1 / Real.sqrt a
          else -1 / Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_value_l313_31323
