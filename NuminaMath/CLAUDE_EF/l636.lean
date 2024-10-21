import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_trapezoid_l636_63695

/-- A trapezoid with an inscribed circle -/
structure TrapezoidWithIncircle where
  -- Points of the trapezoid
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- Points where the circle touches the sides
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Radius of the inscribed circle
  r : ℝ
  -- WZ is parallel to XY
  parallel_sides : (Z.2 - W.2) / (Z.1 - W.1) = (Y.2 - X.2) / (Y.1 - X.1)
  -- XY length is 10
  xy_length : Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 10
  -- ∠WXY is 45°
  angle_wxy : Real.arctan ((W.2 - X.2) / (W.1 - X.1)) - Real.arctan ((Y.2 - X.2) / (Y.1 - X.1)) = π / 4

theorem incircle_radius_of_trapezoid (t : TrapezoidWithIncircle) : t.r = 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_trapezoid_l636_63695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_l636_63665

-- Define the line equation
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y - 3 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the point P
def P (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem line_ellipse_intersection
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, line a b x y → ¬circle_eq x y)
  (h2 : a ≠ 0 ∨ b ≠ 0) :
  ∀ m k : ℝ, ∃! (x y : ℝ), 
    (y - b = m * (x - a)) ∧ 
    ellipse x y ∧
    (x ≠ a ∨ y ≠ b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_l636_63665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_implies_60_degrees_l636_63676

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition for the equation having two equal real roots
def has_equal_roots (t : Triangle) : Prop :=
  4 * (t.b^2 + t.c^2) = 4 * (t.a^2 + t.b * t.c)

-- Define the angle A in degrees
noncomputable def angle_A (t : Triangle) : ℝ :=
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)) * (180 / Real.pi)

-- Theorem statement
theorem equal_roots_implies_60_degrees (t : Triangle) :
  has_equal_roots t → angle_A t = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_implies_60_degrees_l636_63676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_left_after_expenses_l636_63671

noncomputable def monthly_salary : ℝ := 6000
noncomputable def house_rental : ℝ := 640
noncomputable def food_expense : ℝ := 380
noncomputable def electric_water_ratio : ℝ := 1/4
noncomputable def insurance_ratio : ℝ := 1/5
noncomputable def tax_ratio : ℝ := 1/10
noncomputable def transportation_ratio : ℝ := 3/100
noncomputable def emergency_ratio : ℝ := 2/100

theorem money_left_after_expenses :
  monthly_salary - (house_rental + food_expense + 
    electric_water_ratio * monthly_salary + 
    insurance_ratio * monthly_salary + 
    tax_ratio * monthly_salary + 
    transportation_ratio * monthly_salary + 
    emergency_ratio * monthly_salary) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_left_after_expenses_l636_63671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_constant_term_l636_63611

theorem expansion_constant_term (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x + a / x) * (3 * x - 2 / x)^5 = 3) →
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    (x + a / x) * (3 * x - 2 / x)^5 = c + x * (Polynomial.eval x (Polynomial.X : Polynomial ℝ)) + 
      (Polynomial.eval (1 / x) (Polynomial.X : Polynomial ℝ)) / x ∧ 
    c = 1440) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_constant_term_l636_63611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l636_63673

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1 - a^x) / (1 + a^x)

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 1) (h2 : f (a + 2) = 81) :
  a = 2 ∧
  (∀ x, g a (-x) = -(g a x)) ∧
  (∀ x y, x < y → g a x > g a y) ∧
  (∀ y, y ∈ Set.Ioo (-1) 1 ↔ ∃ x, g a x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l636_63673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_min_value_correct_l636_63640

/-- The function f(x) = x²|x-a| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * abs (x - a)

/-- The set of x that satisfies f(x)=x when a=2 -/
noncomputable def solution_set : Set ℝ := {0, 1, 1 + Real.sqrt 2}

/-- The minimum value of f(x) in the interval [1,2] -/
noncomputable def min_value (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 - a
  else if a ≤ 2 then 0
  else if a ≤ 7/3 then 4 * (a - 2)
  else a - 1

theorem solution_set_correct :
  ∀ x, x ∈ solution_set ↔ f 2 x = x := by
  sorry

theorem min_value_correct :
  ∀ a, ∃ x ∈ Set.Icc 1 2, f a x = min_value a ∧
  ∀ y ∈ Set.Icc 1 2, f a y ≥ min_value a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_min_value_correct_l636_63640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l636_63658

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 6/a| + |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≥ 2 * Real.sqrt 6) ∧
  (f a 3 < 7 → 2 < a ∧ a < 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l636_63658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_deg_l636_63619

theorem sin_2012_deg : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_deg_l636_63619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_top_width_l636_63679

/-- Represents the properties of a trapezoidal canal cross-section -/
structure CanalCrossSection where
  bottom_width : ℝ
  top_width : ℝ
  depth : ℝ
  area : ℝ

/-- The area of a trapezium is half the sum of parallel sides times the height -/
noncomputable def trapezium_area (c : CanalCrossSection) : ℝ :=
  (c.bottom_width + c.top_width) / 2 * c.depth

/-- Theorem: Given a canal with bottom width 8 m, area 840 sq. m, and depth 84 m, 
    the top width is 12 m -/
theorem canal_top_width (c : CanalCrossSection) 
  (h1 : c.bottom_width = 8)
  (h2 : c.area = 840)
  (h3 : c.depth = 84)
  (h4 : c.area = trapezium_area c) : 
  c.top_width = 12 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_top_width_l636_63679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_my_sequence_l636_63608

noncomputable def my_sequence (n : ℕ) : ℝ := (3^n - 2^n) / (3^(n+1) + 2^(n+1))

theorem limit_of_my_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |my_sequence n - 1/3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_my_sequence_l636_63608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l636_63681

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a4 : a 4 = 70)
  (h_a21 : a 21 = -100) :
  (∀ n : ℕ, a n = 110 - 10 * n) ∧
  (Finset.filter (fun n => a n ≥ 0) (Finset.range 100)).card = 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l636_63681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_price_doubling_time_l636_63678

/-- The time in years for the price of chicken to reach a target price -/
noncomputable def time_to_reach_price (initial_price : ℝ) (target_price : ℝ) (months_to_triple : ℕ) : ℝ :=
  let price_ratio := target_price / initial_price
  let num_triples := (Real.log price_ratio) / (Real.log 3)
  (num_triples * (months_to_triple : ℝ)) / 12

/-- Theorem stating that it takes 2 years for the price to reach R$ 81.00 -/
theorem chicken_price_doubling_time :
  time_to_reach_price 1 81 6 = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_reach_price 1 81 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_price_doubling_time_l636_63678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_from_square_dissection_l636_63610

/-- The Point type represents a point in 2D space. -/
def Point : Type := ℝ × ℝ

/-- The distance function calculates the Euclidean distance between two points. -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The orthogonal function checks if two vectors are perpendicular. -/
def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

/-- Given a square with side length 4, this theorem proves that the ratio of the height to the length
    of the rectangle formed by dissecting the square along specific lines is 9/100. -/
theorem rectangle_ratio_from_square_dissection :
  ∀ (M N P Q : Point),
  let square_side : ℝ := 4
  let third : ℝ := 1/3
  (distance P M = third * square_side) →
  (distance Q N = third * square_side) →
  (orthogonal (P.1 - M.1, P.2 - M.2) (Q.1 - N.1, Q.2 - N.2)) →
  let rectangle_height : ℝ := (16 : ℝ) / ((40/3 : ℝ))
  let rectangle_length : ℝ := 40/3
  (rectangle_height / rectangle_length = 9/100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_from_square_dissection_l636_63610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l636_63644

theorem floor_properties (x y : ℝ) (hx : x < 0) (hy : y ≥ 0) : 
  (∀ z : ℝ, ⌊z + 1⌋ = ⌊z⌋ + 1) ∧ 
  (∃ a b : ℝ, a < 0 ∧ b ≥ 0 ∧ ⌊a + b⌋ ≠ ⌊a⌋ + ⌊b⌋) ∧
  (∃ c d : ℝ, c < 0 ∧ d ≥ 0 ∧ ⌊c * d⌋ ≠ ⌊c⌋ * ⌊d⌋) := by
  sorry

#check floor_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l636_63644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_bounds_l636_63683

open Real

theorem tangent_line_sin_bounds (x₀ : ℝ) (a b : ℝ) : 
  x₀ ≥ 0 → 
  (∀ x ≥ 0, Real.sin x ≤ a * x + b) → 
  (a = Real.cos x₀) → 
  (Real.sin x₀ = a * x₀ + b) → 
  ∃ (min max : ℝ), ∀ x ∈ Set.Icc 0 (π/2), 
    min ≤ Real.cos x + Real.sin x - x * Real.cos x ∧ 
    Real.cos x + Real.sin x - x * Real.cos x ≤ max :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_bounds_l636_63683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_five_hours_l636_63684

/-- Represents the journey scenario with given conditions -/
structure Journey where
  routeLength : ℝ
  carSpeed : ℝ
  cycleSpeed : ℝ
  samCarDistance : ℝ
  janeReturnDistance : ℝ

/-- Calculates the total time of the journey -/
noncomputable def totalTime (j : Journey) : ℝ :=
  (j.samCarDistance / j.carSpeed) + ((j.routeLength - j.samCarDistance) / j.cycleSpeed)

/-- Theorem stating that the total time of the journey is 5 hours -/
theorem journey_time_is_five_hours (j : Journey) 
    (h1 : j.routeLength = 150)
    (h2 : j.carSpeed = 30)
    (h3 : j.cycleSpeed = 10)
    (h4 : j.samCarDistance = 150)
    (h5 : j.janeReturnDistance = 75)
    (h6 : (j.samCarDistance / j.carSpeed) + ((j.routeLength - j.samCarDistance) / j.cycleSpeed) = 
          (j.samCarDistance / j.carSpeed) + (j.janeReturnDistance / j.carSpeed) + 
          ((j.routeLength - (j.samCarDistance - j.janeReturnDistance)) / j.carSpeed))
    (h7 : (j.samCarDistance / j.carSpeed) + ((j.routeLength - j.samCarDistance) / j.cycleSpeed) = 
          ((j.samCarDistance - j.janeReturnDistance) / j.cycleSpeed) + 
          ((j.routeLength - (j.samCarDistance - j.janeReturnDistance)) / j.carSpeed)) : 
    totalTime j = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_five_hours_l636_63684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_geometric_triangle_l636_63666

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a, b, c form a geometric sequence and c = 2a, then cos B = 3/4 -/
theorem cosine_in_geometric_triangle (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  b^2 = a * c →
  c = 2 * a →
  Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c) →
  Real.cos B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_geometric_triangle_l636_63666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l636_63659

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 4) / Real.log a

-- State the theorem
theorem f_positive_range (a : ℝ) :
  (∀ x > 1, f a x > 0) ↔ (1 < a ∧ a < 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l636_63659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l636_63603

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define the line through origin with slope 1/6
def line_through_origin (x y : ℝ) : Prop :=
  y = (1/6) * x

-- Define the property that M is on line AB
def M_on_AB (a b xm ym : ℝ) : Prop :=
  ∃ t : ℝ, xm = t * a ∧ ym = (1 - t) * b

-- Define the property MA = 1/3 BM
def MA_eq_third_BM (a b xm ym : ℝ) : Prop :=
  (a - xm)^2 + ym^2 = (1/9) * (xm^2 + (ym - b)^2)

-- Main theorem
theorem ellipse_properties
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (xm ym : ℝ)
  (h_line : line_through_origin xm ym)
  (h_M_on_AB : M_on_AB a b xm ym)
  (h_MA_BM : MA_eq_third_BM a b xm ym)
  (p q : ℝ × ℝ)
  (h_pq_diameter : circle_C p.1 p.2 ∧ circle_C q.1 q.2 ∧ (p.1 + q.1 = 4 ∧ p.2 + q.2 = 2))
  (h_e_through_pq : ellipse a b p.1 p.2 ∧ ellipse a b q.1 q.2) :
  a = 2 * b ∧ ∀ (x y : ℝ), ellipse 4 2 x y ↔ ellipse a b x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l636_63603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_blue_ratio_l636_63604

/-- Represents the number of pairs of black socks -/
def black_socks : ℕ := 4

/-- Represents the number of pairs of blue socks -/
def blue_socks : ℕ := 16

/-- Represents the price of a pair of blue socks -/
def blue_price : ℝ := 1 -- We set this to 1 for simplicity

/-- The price of black socks is twice the price of blue socks -/
def black_price : ℝ := 2 * blue_price

/-- The original total cost -/
def original_cost : ℝ := (black_socks : ℝ) * black_price + (blue_socks : ℝ) * blue_price

/-- The reversed cost (after the mix-up) -/
def reversed_cost : ℝ := (black_socks : ℝ) * blue_price + (blue_socks : ℝ) * black_price

/-- The reversed cost is 50% more than the original cost -/
axiom cost_increase : reversed_cost = (3/2) * original_cost

/-- The theorem to prove -/
theorem black_to_blue_ratio : 
  (black_socks : ℚ) / blue_socks = 1 / 4 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_blue_ratio_l636_63604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_collinear_points_l636_63620

def A : ℝ × ℝ × ℝ := (1, -2, 11)
def B : ℝ × ℝ × ℝ := (4, 2, 3)

/-- Three points are collinear if the vector from the first point to the second is parallel to the vector from the first point to the third. -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (q.fst - p.fst, q.snd - p.snd, q.2.2 - p.2.2) = t • (r.fst - p.fst, r.snd - p.snd, r.2.2 - p.2.2)

theorem product_xy_collinear_points (x y : ℝ) :
  let C : ℝ × ℝ × ℝ := (x, y, 15)
  collinear A B C → x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_collinear_points_l636_63620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l636_63687

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The closed interval [-1, 2] --/
def I : Set ℝ := Set.Icc (-1) 2

theorem range_of_a :
  ∀ a : ℝ, a > 0 →
  (∀ x₁ ∈ I, ∃ x₀ ∈ I, g a x₁ = f x₀) →
  0 < a ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l636_63687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_numbers_count_l636_63675

theorem complex_numbers_count : 
  let S : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let valid_pairs := S.filter (λ b => b ≠ 0) ×ˢ S
  (valid_pairs.filter (λ (a, b) => a ≠ b)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_numbers_count_l636_63675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_equals_one_l636_63680

theorem cos_two_theta_equals_one (θ : ℝ) :
  (2 : ℝ)^(-5/2 + 3 * Real.cos θ) + 1 = (2 : ℝ)^(1/2 + 2 * Real.cos θ) →
  Real.cos (2 * θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_equals_one_l636_63680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_p_l636_63693

/-- Theorem: For a parabola y^2 = 2px with p > 0, if the focus is at (1/4, 0), then p = 1/2 -/
theorem parabola_focus_p (p : ℝ) : 
  p > 0 → 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (1/4 : ℝ) = p/2 → 
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_p_l636_63693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l636_63651

-- Define the system of inequalities
noncomputable def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ a + 2 ∧ x < 3*a - 2

-- Define the quadratic function
noncomputable def f (a x : ℝ) : ℝ :=
  (a - 3) * x^2 - x - 1/4

-- Define the number of intersection points
noncomputable def num_intersections (a : ℝ) : ℕ :=
  let discriminant := 1 - 4*(a-3)*(-1/4)
  if discriminant > 0 then 2
  else if discriminant = 0 then 1
  else 0

-- Theorem statement
theorem intersection_points (a : ℝ) :
  has_solution a → (num_intersections a = 1 ∨ num_intersections a = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l636_63651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_sum_l636_63672

def cross_sum_arrangement (a b c d e : ℕ) : Prop :=
  a + b + e = a + c + e ∧ a + b + e = b + d + e

theorem max_cross_sum : ∃ (a b c d e : ℕ),
  a ∈ ({2, 5, 8, 11, 14} : Set ℕ) ∧
  b ∈ ({2, 5, 8, 11, 14} : Set ℕ) ∧
  c ∈ ({2, 5, 8, 11, 14} : Set ℕ) ∧
  d ∈ ({2, 5, 8, 11, 14} : Set ℕ) ∧
  e ∈ ({2, 5, 8, 11, 14} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  cross_sum_arrangement a b c d e ∧
  a + b + e = 33 ∧
  ∀ (x y z w v : ℕ),
    x ∈ ({2, 5, 8, 11, 14} : Set ℕ) →
    y ∈ ({2, 5, 8, 11, 14} : Set ℕ) →
    z ∈ ({2, 5, 8, 11, 14} : Set ℕ) →
    w ∈ ({2, 5, 8, 11, 14} : Set ℕ) →
    v ∈ ({2, 5, 8, 11, 14} : Set ℕ) →
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
    z ≠ w ∧ z ≠ v ∧
    w ≠ v →
    cross_sum_arrangement x y z w v →
    x + y + v ≤ 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_sum_l636_63672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_special_pyramid_l636_63626

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  PA : ℝ
  PB : ℝ
  PC : ℝ

/-- The surface area of a sphere -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The circumscribed sphere surface area of a triangular pyramid -/
noncomputable def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sphereSurfaceArea (((pyramid.PA^2 + pyramid.PB^2 + pyramid.PC^2).sqrt) / 2)

theorem circumscribed_sphere_surface_area_of_special_pyramid :
  ∃ (pyramid : TriangularPyramid),
    pyramid.PA = 1 ∧
    pyramid.PB = 2 ∧
    pyramid.PC = 3 ∧
    circumscribedSphereSurfaceArea pyramid = 14 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_special_pyramid_l636_63626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_sine_in_special_triangle_l636_63655

theorem largest_angle_sine_in_special_triangle :
  ∀ (a b c : ℝ) (θ : ℝ),
  a = 1 ∧ b = 1 ∧ c = Real.sqrt 3 →
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
  Real.sin θ = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_sine_in_special_triangle_l636_63655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_with_equilateral_property_l636_63630

/-- A complex number z satisfies the equilateral triangle property if 0, z, and z^4 
    form the vertices of an equilateral triangle in the complex plane. -/
def has_equilateral_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  z ≠ z^4 ∧
  Complex.abs z = Complex.abs (z^4 - z) ∧
  Complex.abs z = Complex.abs z^4 ∧
  Complex.abs (z^4 - z) = Complex.abs z^4

/-- There are exactly two nonzero complex numbers that satisfy the equilateral triangle property. -/
theorem two_complex_numbers_with_equilateral_property :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, has_equilateral_property z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_with_equilateral_property_l636_63630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l636_63605

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_ms : ℝ := 5 / 18

/-- Length of the first train in meters -/
def train1_length : ℝ := 200

/-- Speed of the first train in km/h -/
def train1_speed_kmph : ℝ := 120

/-- Length of the second train in meters -/
def train2_length : ℝ := 300.04

/-- Speed of the second train in km/h -/
def train2_speed_kmph : ℝ := 80

/-- Time for trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ :=
  (train1_length + train2_length) / ((train1_speed_kmph * kmph_to_ms) + (train2_speed_kmph * kmph_to_ms))

theorem trains_crossing_time :
  Int.floor crossing_time = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l636_63605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_of_regular_quadrilateral_pyramid_l636_63623

/-- The radius of the circumscribed sphere of a regular quadrilateral pyramid -/
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  (5 * a) / (4 * Real.sqrt 3)

/-- Theorem: The radius of the circumscribed sphere of a regular quadrilateral pyramid
    with base side length a and side face angle 60° with the base plane is (5a) / (4√3) -/
theorem circumscribed_sphere_radius_of_regular_quadrilateral_pyramid
  (a : ℝ) (h : a > 0) :
  let base_side_length := a
  let side_face_angle := π / 3  -- 60° in radians
  circumscribed_sphere_radius a = (5 * base_side_length) / (4 * Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

#check circumscribed_sphere_radius
#check circumscribed_sphere_radius_of_regular_quadrilateral_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_of_regular_quadrilateral_pyramid_l636_63623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fold_f_of_neg_one_plus_i_l636_63602

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (z : ℂ) : ℂ :=
  if (z^(1/3)).im ≠ 0 then z^3 else -z^3

-- State the theorem
theorem four_fold_f_of_neg_one_plus_i :
  f (f (f (f (-1 + i)))) = -134217728 * i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fold_f_of_neg_one_plus_i_l636_63602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_profit_analysis_l636_63636

-- Define the piecewise daily sales revenue function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 3 then 9 / (x + 1) + 2 * x + 3
  else if 3 < x ∧ x ≤ 6 then 9 * m * x / (x + 3) + 3 * x
  else 21

-- Define the daily sales profit function
noncomputable def P (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 3 then 9 / (x + 1) + x + 3
  else if 3 < x ∧ x ≤ 6 then 9 * x / (2 * (x + 3)) + 2 * x
  else 21 - x

-- Theorem statement
theorem sales_profit_analysis :
  ∃ (m : ℝ),
    (∀ x, f m x = f m 3) ∧
    (∀ x, P x = f (1/2) x - x) ∧
    (∀ x, P x ≤ P 6) ∧
    P 6 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_profit_analysis_l636_63636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l636_63616

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- The line equation -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * x

/-- The distance between a point (x, y) and the line y = kx -/
noncomputable def distance_to_line (k x y : ℝ) : ℝ :=
  abs (k * x - y) / Real.sqrt (k^2 + 1)

/-- The theorem statement -/
theorem circle_line_intersection (k : ℝ) :
  (∃! (p1 p2 p3 : ℝ × ℝ),
    circle_equation p1.1 p1.2 ∧
    circle_equation p2.1 p2.2 ∧
    circle_equation p3.1 p3.2 ∧
    distance_to_line k p1.1 p1.2 = 2 ∧
    distance_to_line k p2.1 p2.2 = 2 ∧
    distance_to_line k p3.1 p3.2 = 2) →
  k = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l636_63616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_k_values_l636_63685

theorem quadratic_equation_k_values (k : ℕ) : 
  (∃ x y : ℤ, x^2 - k*x + 16 = 0 ∧ y^2 - k*y + 16 = 0 ∧ x ≠ y) → 
  (∃ m : ℕ, x * y = 8 * m) →
  k = 17 ∨ k = 10 ∨ k = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_k_values_l636_63685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_formula_l636_63618

-- Define the sequence a_n
def a : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | n + 2 => n + 1

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ → ℝ
  | 0 => 0
  | n + 1 => S n + a (n + 1)

-- State the given conditions
axiom a2_eq_1 : a 2 = 1
axiom S_relation : ∀ n : ℕ, 2 * S n = n * a n

-- Theorem for the general formula of a_n
theorem a_formula : ∀ n : ℕ, a n = n - 1 := by sorry

-- Define T_n as the sum of the first n terms of (a_n + 1) / 2^n
noncomputable def T : ℕ → ℝ
  | 0 => 0
  | n + 1 => T n + (a (n + 1) + 1) / (2 ^ (n + 1))

-- Theorem for the sum T_n
theorem T_formula : ∀ n : ℕ, T n = 2 - (n + 2) / (2 ^ n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_formula_l636_63618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l636_63622

noncomputable def original_function (x : ℝ) : ℝ := Real.cos (2 * x)

noncomputable def translated_function (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

theorem translation_equivalence :
  ∀ x : ℝ, translated_function x = original_function (x + Real.pi / 6) + 1 :=
by
  intro x
  simp [translated_function, original_function]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l636_63622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_proof_l636_63645

-- Define constants for the angles in radians
noncomputable def angle18 : ℝ := 18 * Real.pi / 180
noncomputable def angle63 : ℝ := 63 * Real.pi / 180

-- Define m and n
noncomputable def m : ℝ := 2 * Real.sin angle18
noncomputable def n : ℝ := 4 - m^2

-- Theorem statement
theorem golden_ratio_proof : (m + Real.sqrt n) / Real.sin angle63 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_proof_l636_63645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l636_63637

-- Define the function f
noncomputable def f (x : ℝ) := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

-- Define the domain
def domain : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x < Real.pi / 3}

-- Theorem statement
theorem f_range_theorem :
  ∀ x ∈ domain, (3/2 - Real.sqrt 2) ≤ f x ∧ f x ≤ 3 ∧
  (∃ x₁ ∈ domain, f x₁ = 3/2 - Real.sqrt 2) ∧
  (∃ x₂ ∈ domain, f x₂ = 3) :=
by
  sorry

#check f_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l636_63637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l636_63601

/-- Time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train of 360m length traveling at 72 km/h takes 25 seconds to pass a 140m bridge -/
theorem train_bridge_passing_time :
  time_to_pass_bridge 360 140 72 = 25 := by
  sorry

-- Remove #eval as it's not computable
-- #eval time_to_pass_bridge 360 140 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l636_63601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_function_value_l636_63689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 + a - 5) * (Real.log x / Real.log a)

theorem logarithmic_function_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (1/8) = -3 :=
by
  -- We'll use sorry to skip the proof for now
  sorry

#check logarithmic_function_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_function_value_l636_63689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_specific_complex_l636_63664

noncomputable def complex_distance (z : ℂ) : ℝ :=
  Real.sqrt (z.re ^ 2 + z.im ^ 2)

theorem distance_of_specific_complex : complex_distance ((2 / (1 + Complex.I)) * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_specific_complex_l636_63664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_c_l636_63657

theorem right_triangle_side_c (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * Real.pi / 180 →  -- Convert 30° to radians
  a = 1 →
  b = Real.sqrt 3 →
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  C = Real.pi / 2 →  -- Right angle
  a * Real.sin B = b * Real.sin A →  -- Sine rule
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  c = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_c_l636_63657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_exhibit_multiple_l636_63650

theorem zoo_exhibit_multiple (reptile_house rain_forest multiple : ℕ) :
  (reptile_house = multiple * rain_forest - 5) →
  (reptile_house = 16) →
  (rain_forest = 7) →
  multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_exhibit_multiple_l636_63650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_bushes_theorem_l636_63646

/-- The number of rose bushes needed to surround a circular garden -/
noncomputable def num_bushes (radius : ℝ) (spacing : ℝ) : ℕ :=
  ⌊(2 * Real.pi * radius) / spacing⌋₊

/-- Theorem stating that the number of rose bushes needed is approximately 47 -/
theorem rose_bushes_theorem (radius spacing : ℝ) 
  (h_radius : radius = 15)
  (h_spacing : spacing = 2) :
  num_bushes radius spacing = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_bushes_theorem_l636_63646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l636_63649

-- Define the rational function
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3)

-- Define the denominator of the function
def denom (x : ℝ) : ℝ := 4 * x^2 + 6 * x + 3

-- State the theorem
theorem vertical_asymptotes_sum (p q : ℝ) :
  (denom p = 0) → (denom q = 0) → (p ≠ q) → (p + q = -3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l636_63649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_circle_1976gon_l636_63694

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to specify fields for this problem
  mk :: -- Empty constructor

/-- The set of marked points in a regular polygon -/
def markedPoints (n : ℕ) (p : RegularPolygon n) : Set (ℝ × ℝ) :=
  sorry

/-- The maximum number of marked points that can lie on a single circle -/
def maxPointsOnCircle (n : ℕ) (p : RegularPolygon n) : ℕ :=
  sorry

theorem max_points_on_circle_1976gon :
  ∀ (p : RegularPolygon 1976),
    maxPointsOnCircle 1976 p = 1976 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_circle_1976gon_l636_63694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_maple_trees_l636_63654

theorem jerry_maple_trees :
  let pine_logs_per_tree : ℕ := 80
  let maple_logs_per_tree : ℕ := 60
  let walnut_logs_per_tree : ℕ := 100
  let pine_trees_cut : ℕ := 8
  let walnut_trees_cut : ℕ := 4
  let total_logs : ℕ := 1220
  let maple_trees_cut : ℕ :=
    (total_logs - (pine_logs_per_tree * pine_trees_cut + walnut_logs_per_tree * walnut_trees_cut)) / maple_logs_per_tree
  maple_trees_cut = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_maple_trees_l636_63654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_formula_l636_63691

/-- Definition of the sequence a_n -/
def a : ℕ → ℚ
  | 0 => 2/3  -- Add case for n = 0
  | 1 => 2/3
  | (n+2) => 2 * a (n+1) / (a (n+1) + 1)

/-- The sequence {1/a_n - 1} is geometric with first term 1/2 and common ratio 1/2 -/
theorem geometric_sequence (n : ℕ) : 1 / a n - 1 = (1/2)^n := by
  sorry

/-- The general term formula for a_n -/
theorem general_term_formula (n : ℕ) : a n = 2^n / (1 + 2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_formula_l636_63691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_tuning_day_approximation_l636_63667

noncomputable def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

def isSimplerFraction (a b c d : ℕ) : Prop :=
  a * d < b * c ∨ (a * d = b * c ∧ a + b < c + d)

noncomputable def π : ℝ := Real.pi

theorem fourth_tuning_day_approximation :
  ∃ (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ a₃ b₃ c₃ d₃ a₄ b₄ c₄ d₄ : ℕ),
    (31 : ℝ) / 10 < π ∧ π < (49 : ℝ) / 15 ∧
    (b₁ + d₁ : ℚ) / (a₁ + c₁) = 16 / 5 ∧
    (b₂ + d₂ : ℚ) / (a₂ + c₂) = 47 / 15 ∧
    (b₃ + d₃ : ℚ) / (a₃ + c₃) = 63 / 20 ∧
    (b₄ + d₄ : ℚ) / (a₄ + c₄) = 22 / 7 ∧
    (∀ x y, isSimplerFraction x y a₁ c₁ → (x + y : ℚ) / (a₁ + c₁) ≠ 16 / 5) ∧
    (∀ x y, isSimplerFraction x y a₂ c₂ → (x + y : ℚ) / (a₂ + c₂) ≠ 47 / 15) ∧
    (∀ x y, isSimplerFraction x y a₃ c₃ → (x + y : ℚ) / (a₃ + c₃) ≠ 63 / 20) ∧
    (∀ x y, isSimplerFraction x y a₄ c₄ → (x + y : ℚ) / (a₄ + c₄) ≠ 22 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_tuning_day_approximation_l636_63667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l636_63638

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem statement -/
theorem ellipse_fixed_point_theorem (e : Ellipse) (Q E : Point) (l : ℝ) :
  e.a = 2 →
  e.b = Real.sqrt 3 →
  isOnEllipse e Q →
  Q.x = 1 →
  Q.y = 3/2 →
  eccentricity e = 1/2 →
  E.x = 7 →
  E.y = 0 →
  l = 4 →
  ∃ F : Point,
    F.x = 1 ∧
    F.y = 0 ∧
    ∀ P : Point,
      isOnEllipse e P →
      P.x ≠ 2 →
      P.x ≠ -2 →
      ∃ M N : Point,
        M.x = l ∧
        N.x = l ∧
        (∃ k₁ k₂ : ℝ,
          M.y = k₁ * (M.x + 2) ∧
          N.y = k₂ * (N.x - 2) ∧
          k₁ * k₂ = -3/4 ∧
          (M.y - E.y) / (M.x - E.x) * (N.y - E.y) / (N.x - E.x) = -1 ∧
          (M.y - F.y) / (M.x - F.x) * (N.y - F.y) / (N.x - F.x) = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l636_63638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_values_l636_63677

noncomputable def f (x : ℝ) : ℝ := 
  (Real.cos (x - Real.pi/2)) / (Real.sin (7*Real.pi/2 + x)) * Real.cos (Real.pi - x)

theorem f_simplification_and_values (α : ℝ) 
  (h : f α = -5/13) : 
  (∀ x, f x = Real.sin x) ∧ 
  ((Real.cos α = 12/13 ∧ Real.tan α = -5/12) ∨ 
   (Real.cos α = -12/13 ∧ Real.tan α = 5/12)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_values_l636_63677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_log_power_property_main_theorem_l636_63668

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  f (a * b) = f a + f b := by sorry

theorem log_power_property (a : ℝ) (n : ℕ) (ha : 0 < a) : 
  f (a ^ n) = n * f a := by sorry

theorem main_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : f (a * b) = 1) :
  f (a^2) + f (b^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_log_power_property_main_theorem_l636_63668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l636_63634

-- Define the rectangle type
structure Rectangle where
  length : Nat
  width : Nat

-- Define the perimeter condition
def hasPerimeter40 (r : Rectangle) : Prop :=
  2 * r.length + 2 * r.width = 40

-- Define the condition that one dimension is even
def hasEvenDimension (r : Rectangle) : Prop :=
  r.length % 2 = 0 ∨ r.width % 2 = 0

-- Define the area function
def area (r : Rectangle) : Nat :=
  r.length * r.width

-- Theorem statement
theorem max_area_rectangle :
  ∃ (r : Rectangle), hasPerimeter40 r ∧ hasEvenDimension r ∧
    (∀ (r' : Rectangle), hasPerimeter40 r' ∧ hasEvenDimension r' → area r' ≤ area r) ∧
    area r = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l636_63634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parenthesization_equal_to_1292646_l636_63612

/-- The sequence of numbers in the numerator of the fraction -/
def numerator_seq : List ℕ := List.range 14 |>.map (λ i => 29 - i)

/-- The sequence of numbers in the denominator of the fraction -/
def denominator_seq : List ℕ := List.range 14 |>.map (λ i => 15 - i)

/-- A type representing the possible parenthesization of the fraction -/
inductive Parenthesization
| Divide : Parenthesization → Parenthesization → Parenthesization
| Single : ℕ → Parenthesization

/-- Evaluate a parenthesization to a rational number -/
def evaluate : Parenthesization → ℚ
| Parenthesization.Divide p q => (evaluate p) / (evaluate q)
| Parenthesization.Single n => n

/-- Check if a Parenthesization contains a specific number -/
def contains : Parenthesization → ℕ → Prop
| Parenthesization.Divide p q, n => contains p n ∨ contains q n
| Parenthesization.Single m, n => m = n

/-- The theorem stating the existence of a parenthesization resulting in 1292646 -/
theorem exists_parenthesization_equal_to_1292646 :
  ∃ (p_num p_den : Parenthesization),
    (∀ n, n ∈ numerator_seq → contains p_num n) ∧
    (∀ n, n ∈ denominator_seq → contains p_den n) ∧
    evaluate (Parenthesization.Divide p_num p_den) = 1292646 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parenthesization_equal_to_1292646_l636_63612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_theorem_l636_63656

/-- The number of years after which a tree's circumference exceeds 90 cm -/
def years_to_exceed_90cm (initial_circumference : ℝ) (annual_increase : ℝ) (x : ℝ) : Prop :=
  initial_circumference + annual_increase * x > 90

/-- Theorem: Given a tree with initial circumference 10 cm and annual increase of 3 cm,
    the number of years x after which the circumference exceeds 90 cm satisfies 3x + 10 > 90 -/
theorem tree_growth_theorem :
  ∀ x : ℝ, years_to_exceed_90cm 10 3 x ↔ 3 * x + 10 > 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_theorem_l636_63656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l636_63635

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define S_n as the sum of the first n terms
def S : ℕ → ℝ := sorry

-- The given condition relating S_n and a_n
axiom S_condition (n : ℕ) : S n = (4/3) * (a n - 1)

-- The expression to be minimized
noncomputable def expr (n : ℕ) : ℝ := (4^(n-2) + 1) * (16 / a n + 1)

-- Theorem stating the minimum value of the expression
theorem min_value_is_four :
  ∃ (n : ℕ), expr n = 4 ∧ ∀ (m : ℕ), expr m ≥ 4 := by
  sorry

-- Additional lemma to establish the sequence properties
lemma a_is_geometric (n : ℕ) : a n = 4^n := by
  sorry

-- Lemma for the AM-GM inequality used in the proof
lemma am_gm_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : (x + y) / 2 ≥ Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l636_63635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanAlpha_eq_2_implies_fraction_eq_neg8_sinAlpha_plus_cosAlpha_eq_one_fifth_implies_tanAlpha_eq_neg_four_thirds_l636_63662

-- Problem 1
theorem tanAlpha_eq_2_implies_fraction_eq_neg8 (α : Real) (h : Real.tan α = 2) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = -8 := by sorry

-- Problem 2
theorem sinAlpha_plus_cosAlpha_eq_one_fifth_implies_tanAlpha_eq_neg_four_thirds (α : Real)
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanAlpha_eq_2_implies_fraction_eq_neg8_sinAlpha_plus_cosAlpha_eq_one_fifth_implies_tanAlpha_eq_neg_four_thirds_l636_63662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l636_63633

/-- Given a geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_problem (a q : ℝ) :
  geometric_sum a q 2 = 80 ∧
  geometric_sum a q 6 = 665 →
  geometric_sum a q 4 = 260 := by
  sorry

#check geometric_sequence_sum_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l636_63633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l636_63696

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem f_is_increasing (a b c : ℝ) (h : a^2 - 3*b < 0) :
  StrictMonoOn (f a b c) Set.univ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l636_63696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_implies_a_equals_one_l636_63609

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*a*(Real.log (x^2 + 2)) / Real.log 2 + a^2 - 3

theorem unique_zero_point_implies_a_equals_one :
  (∃! x, f a x = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_implies_a_equals_one_l636_63609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_sum_positive_l636_63606

open Real

-- Define the function f
noncomputable def f (a k x : ℝ) : ℝ := -1/x - k*x + a * log x

-- Define the derivative of f
noncomputable def f_deriv (a k x : ℝ) : ℝ := 1/x^2 + a/x - k

-- Theorem statement
theorem f_deriv_sum_positive
  (a : ℝ)
  (h_a : a > 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : x₁ > 0)
  (h_x₂ : x₂ > 0)
  (h_x_neq : x₁ ≠ x₂)
  (k : ℝ)
  (h_f_eq : f a k x₁ = f a k x₂) :
  f_deriv a k x₁ + f_deriv a k x₂ > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_sum_positive_l636_63606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gallons_needed_l636_63621

/-- The number of drums to be painted -/
def num_drums : ℕ := 20

/-- The height of each drum in feet -/
def drum_height : ℝ := 15

/-- The diameter of each drum in feet -/
def drum_diameter : ℝ := 8

/-- The area that one gallon of paint can cover in square feet -/
def paint_coverage : ℝ := 300

/-- Calculate the lateral surface area of a single drum -/
noncomputable def drum_lateral_area : ℝ := 2 * Real.pi * (drum_diameter / 2) * drum_height

/-- Calculate the total lateral surface area of all drums -/
noncomputable def total_area : ℝ := num_drums * drum_lateral_area

/-- Calculate the number of gallons needed, rounded up to the nearest integer -/
noncomputable def gallons_needed : ℕ := (Int.ceil (total_area / paint_coverage)).toNat

/-- Theorem stating the minimum number of full gallons of paint needed -/
theorem min_gallons_needed : gallons_needed = 26 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gallons_needed_l636_63621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_h_l636_63688

open Real

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := sin (4 * x + π / 3)

/-- The function after doubling the abscissa -/
noncomputable def g (x : ℝ) : ℝ := f (x / 2)

/-- The final function after shifting right by π/6 -/
noncomputable def h (x : ℝ) : ℝ := g (x - π / 6)

/-- Theorem stating that (π/2, 0) is the symmetric center of h -/
theorem symmetric_center_of_h :
  ∀ x : ℝ, h (π - x) = h x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_h_l636_63688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l636_63652

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 4) / (x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l636_63652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_eval_l636_63661

theorem floor_ceil_fraction_eval : 
  ⌊⌈((12:ℝ)/5)^2⌉ + (11:ℝ)/3⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_eval_l636_63661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_existence_l636_63625

-- Define a circle on a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define the concept of two non-overlapping circles
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

-- Define a point outside both circles
def outside_circles (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  let (x, y) := p
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x - x1)^2 + (y - y1)^2 > c1.radius^2 ∧
  (x - x2)^2 + (y - y2)^2 > c2.radius^2

-- Define the property that every line through a point intersects at least one circle
def intersects_circles (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  ∀ (m b : ℝ), ∃ (x y : ℝ),
    (y = m * x + b) ∧
    ((x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∨
     (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2)

-- Define the region bounded by external tangent lines
def in_tangent_region (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  sorry -- This definition would involve complex geometric calculations

-- Define the set of points on external tangent lines
def external_tangent_lines (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  sorry -- This definition would involve complex geometric calculations

-- The main theorem
theorem intersection_point_existence (c1 c2 : Circle) 
  (h : non_overlapping c1 c2) :
  ∃ (M : Set (ℝ × ℝ)), 
    (∀ p ∈ M, outside_circles p c1 c2 ∧ intersects_circles p c1 c2) ∧
    (∀ p, p ∈ M ↔ (in_tangent_region p c1 c2 ∧ p ∉ external_tangent_lines c1 c2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_existence_l636_63625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_crayons_theorem_l636_63686

def crayons_remaining (box1 box2 box3 : ℕ) 
  (loss1 loss2 loss3 : ℚ)
  (broken2 broken3 : ℕ) : ℕ :=
  let remaining1 := box1 - Int.floor (loss1 * box1)
  let remaining2 := box2 - Int.floor (loss2 * box2)
  let remaining3 := box3 - Int.floor (loss3 * box3)
  (remaining1 + remaining2 + remaining3).toNat

theorem paul_crayons_theorem :
  crayons_remaining 479 352 621 (70/100) (25/100) (50/100) 8 15 = 719 := by
  -- Unfold the definition of crayons_remaining
  unfold crayons_remaining
  -- Simplify the arithmetic expressions
  simp
  -- The proof is completed
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_crayons_theorem_l636_63686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_exponential_l636_63639

theorem inequality_exponential (x y : ℝ) (h1 : x > y) (h2 : y < 0) :
  (1/2:ℝ)^x - (1/2:ℝ)^y < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_exponential_l636_63639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_value_l636_63631

/-- The sequence {aₙ} defined by the given recurrence relation -/
noncomputable def a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (Real.sqrt 3 * a n + 1) / (Real.sqrt 3 - a n)

/-- Theorem stating that the 2002nd term of the sequence equals 2 + 4√3 -/
theorem a_2002_value : a 2002 = 2 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_value_l636_63631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_of_det_three_l636_63660

open Matrix

variable {n : ℕ}
variable (M : Matrix (Fin n) (Fin n) ℝ)

theorem det_cube_of_det_three (h : Matrix.det M = 3) : Matrix.det (M^3) = 27 := by
  rw [Matrix.det_pow]
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_of_det_three_l636_63660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_P_in_AlPO4_l636_63670

/-- The molar mass of Aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of Phosphorus in g/mol -/
noncomputable def molar_mass_P : ℝ := 30.97

/-- The molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of Oxygen atoms in AlPO4 -/
def num_O_atoms : ℕ := 4

/-- The molar mass of AlPO4 in g/mol -/
noncomputable def molar_mass_AlPO4 : ℝ := molar_mass_Al + molar_mass_P + num_O_atoms * molar_mass_O

/-- The mass percentage of P in AlPO4 -/
noncomputable def mass_percentage_P : ℝ := (molar_mass_P / molar_mass_AlPO4) * 100

/-- Theorem stating that the mass percentage of P in AlPO4 is approximately 25.40% -/
theorem mass_percentage_P_in_AlPO4 : 
  (mass_percentage_P ≥ 25.39) ∧ (mass_percentage_P ≤ 25.41) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_P_in_AlPO4_l636_63670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_after_cutting_l636_63697

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An equilateral triangle with side length 5 -/
def equilateralTriangle : Triangle :=
  { a := 5, b := 5, c := 5 }

/-- A right triangle with legs of length 2 -/
noncomputable def smallRightTriangle : Triangle :=
  { a := 2, b := 2, c := 2 * Real.sqrt 2 }

/-- The perimeter of the remaining figure after cutting -/
def remainingPerimeter (big : Triangle) (small : Triangle) : ℝ :=
  big.a + (big.b - small.a) + (big.c - small.b)

theorem perimeter_after_cutting :
  remainingPerimeter equilateralTriangle smallRightTriangle = 11 := by
  sorry

#eval remainingPerimeter equilateralTriangle { a := 2, b := 2, c := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_after_cutting_l636_63697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_fifteen_meters_l636_63669

/-- The distance between consecutive trees in a yard with equally spaced trees -/
noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : ℝ :=
  yard_length / (num_trees - 1 : ℝ)

/-- Theorem: The distance between consecutive trees is 15 meters -/
theorem distance_is_fifteen_meters :
  let yard_length : ℝ := 255
  let num_trees : ℕ := 18
  distance_between_trees yard_length num_trees = 15 := by
  -- Unfold the definition of distance_between_trees
  unfold distance_between_trees
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_fifteen_meters_l636_63669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_region_area_l636_63641

-- Define the cone parameters
noncomputable def base_radius : ℝ := 2
noncomputable def cone_height : ℝ := 1

-- Define the maximum angle in radians (45 degrees)
noncomputable def max_angle : ℝ := Real.pi / 4

-- Define the theorem
theorem cone_base_region_area :
  let total_base_area := Real.pi * base_radius^2
  let restricted_area := Real.pi * cone_height^2
  (total_base_area - restricted_area) = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_region_area_l636_63641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l636_63698

theorem chocolate_milk_tea_sales (total : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) 
  (h_total : total = 50)
  (h_winter_melon : winter_melon_ratio = 2/5)
  (h_okinawa : okinawa_ratio = 3/10)
  (h_sum : winter_melon_ratio + okinawa_ratio < 1) :
  total - (winter_melon_ratio * ↑total).floor - (okinawa_ratio * ↑total).floor = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l636_63698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_trajectory_and_max_distance_l636_63617

/-- Given a polar coordinate system with pole at the origin of the rectangular coordinate system,
    polar axis coinciding with the positive x-axis, and equal length units in both systems. -/
structure CoordinateSystem where
  -- No additional structure needed, as the alignment is given in the problem description

/-- Line l in polar coordinates -/
noncomputable def line_l (θ : Real) : Real :=
  5 / (Real.sin (θ - Real.pi/3))

/-- Point P in parametric form -/
noncomputable def point_P (α : Real) : Real × Real :=
  (2 * Real.cos α, 2 * Real.sin α + 2)

/-- Theorem stating the trajectory of point P and its maximum distance from line l -/
theorem point_P_trajectory_and_max_distance (cs : CoordinateSystem) :
  (∀ (x y : Real), (∃ (α : Real), point_P α = (x, y)) ↔ x^2 + (y-2)^2 = 4) ∧
  (∃ (d : Real), d = 6 ∧ ∀ (α : Real), ∃ (θ : Real),
    d ≥ Real.sqrt ((point_P α).1 - line_l θ * Real.cos θ)^2 +
                   ((point_P α).2 - line_l θ * Real.sin θ)^2 ∧
    (∃ (α' : Real), d = Real.sqrt ((point_P α').1 - line_l θ * Real.cos θ)^2 +
                                   ((point_P α').2 - line_l θ * Real.sin θ)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_trajectory_and_max_distance_l636_63617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l636_63629

theorem complex_expression_value : 
  (((Real.sqrt 2 : ℂ) / (1 - Complex.I)) ^ 2018 + ((1 + Complex.I) / (1 - Complex.I)) ^ 6) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l636_63629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l636_63642

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 2}

-- Define the center of the circle
def center : ℝ × ℝ := (1, 2)

-- Define the point that the circle passes through
def passThrough : ℝ × ℝ := (0, 1)

-- Define point P
def P : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem circle_properties :
  -- 1. Standard equation of the circle
  (∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 - 1)^2 + (p.2 - 2)^2 = 2) ∧
  -- 2. Equations of tangent lines
  (∃ t₁ t₂ : Set (ℝ × ℝ),
    (∀ p : ℝ × ℝ, p ∈ t₁ ↔ 7 * p.1 - p.2 - 15 = 0) ∧
    (∀ p : ℝ × ℝ, p ∈ t₂ ↔ p.1 + p.2 - 1 = 0) ∧
    (∀ p : ℝ × ℝ, p ∈ t₁ ∩ C → p ≠ P) ∧
    (∀ p : ℝ × ℝ, p ∈ t₂ ∩ C → p ≠ P)) ∧
  -- 3. Length of tangent line segment
  (∃ q₁ q₂ : ℝ × ℝ,
    q₁ ∈ C ∧ q₂ ∈ C ∧
    ((P.1 - q₁.1)^2 + (P.2 - q₁.2)^2 = 8 ∨
     (P.1 - q₂.1)^2 + (P.2 - q₂.2)^2 = 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l636_63642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_iff_l636_63607

-- Define the parameter a
variable (a : ℝ)

-- Define the second root t₂
noncomputable def t₂ (a : ℝ) : ℝ := (a + 2) / (a + 1)

-- Define the condition for two distinct roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  t₂ a ≠ 1 ∧ (t₂ a > 2 ∨ t₂ a < -2)

-- Theorem statement
theorem two_distinct_roots_iff :
  has_two_distinct_roots a ↔ (a > -4/3 ∧ a < -1) ∨ (a > -1 ∧ a < 0) := by
  sorry

#check two_distinct_roots_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_iff_l636_63607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l636_63653

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

-- Define the bounded region
def bounded_region (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 ∧ f x ≤ g x

-- State the theorem
theorem area_between_curves : 
  (∫ x in Set.Icc 0 1, g x - f x) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l636_63653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_equivalence_quadrilateral_equal_sides_not_sufficient_for_square_square_has_equal_sides_not_square_may_have_equal_sides_l636_63632

-- Define basic structures
structure Triangle where
  sides : Fin 3 → ℝ
  
structure Quadrilateral where
  sides : Fin 4 → ℝ

-- Define properties
def isCongruent (t1 t2 : Triangle) : Prop := sorry

def hasEqualSidesTriangle (t : Triangle) : Prop :=
  ∀ i j : Fin 3, t.sides i = t.sides j

def isSquare (q : Quadrilateral) : Prop := sorry

def hasEqualSidesQuadrilateral (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Theorem statements
theorem triangle_congruence_equivalence :
  ∀ t1 t2 : Triangle, isCongruent t1 t2 ↔ hasEqualSidesTriangle t1 ∧ hasEqualSidesTriangle t2 := by sorry

theorem quadrilateral_equal_sides_not_sufficient_for_square :
  ∃ q : Quadrilateral, hasEqualSidesQuadrilateral q ∧ ¬isSquare q := by sorry

theorem square_has_equal_sides :
  ∀ q : Quadrilateral, isSquare q → hasEqualSidesQuadrilateral q := by sorry

theorem not_square_may_have_equal_sides :
  ∃ q : Quadrilateral, ¬isSquare q ∧ hasEqualSidesQuadrilateral q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_equivalence_quadrilateral_equal_sides_not_sufficient_for_square_square_has_equal_sides_not_square_may_have_equal_sides_l636_63632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_passes_bonnie_l636_63663

/-- Represents the length of the track in meters -/
noncomputable def track_length : ℝ := 500

/-- Represents the speed ratio of Annie to Bonnie -/
noncomputable def speed_ratio : ℝ := 1.3

/-- Calculates the number of laps Annie has run when she first passes Bonnie -/
noncomputable def annies_laps : ℝ :=
  let bonnies_distance := track_length / (speed_ratio - 1)
  speed_ratio * bonnies_distance / track_length

theorem annie_passes_bonnie :
  annies_laps = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_passes_bonnie_l636_63663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_material_point_combination_l636_63614

/-- Represents a material point in a three-component mixture -/
structure MaterialPoint where
  a : ℝ  -- concentration of component A
  b : ℝ  -- concentration of component B
  c : ℝ  -- concentration of component C
  m : ℝ  -- mass of the mixture
  nonneg : 0 ≤ m  -- mass is non-negative
  sum_to_one : a + b + c = 1  -- concentrations sum to 1

/-- Combines two material points -/
noncomputable def combine_material_points (K₁ K₂ : MaterialPoint) : MaterialPoint where
  a := (K₁.a * K₁.m + K₂.a * K₂.m) / (K₁.m + K₂.m)
  b := (K₁.b * K₁.m + K₂.b * K₂.m) / (K₁.m + K₂.m)
  c := (K₁.c * K₁.m + K₂.c * K₂.m) / (K₁.m + K₂.m)
  m := K₁.m + K₂.m
  nonneg := by sorry
  sum_to_one := by sorry

/-- Addition operation for MaterialPoint -/
noncomputable instance : Add MaterialPoint where
  add K₁ K₂ := combine_material_points K₁ K₂

/-- The theorem to be proved -/
theorem material_point_combination (K₁ K₂ : MaterialPoint) :
  combine_material_points K₁ K₂ = K₁ + K₂ := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_material_point_combination_l636_63614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l636_63648

theorem inverse_matrices_sum (a b c f : ℝ) : 
  (Matrix.det 
    (!![a, 1; c, 2] : Matrix (Fin 2) (Fin 2) ℝ) ≠ 0) →
  (Matrix.det 
    (!![4, b; f, 3] : Matrix (Fin 2) (Fin 2) ℝ) ≠ 0) →
  (!![a, 1; c, 2] : Matrix (Fin 2) (Fin 2) ℝ) * 
  (!![4, b; f, 3] : Matrix (Fin 2) (Fin 2) ℝ) = 
  (1 : Matrix (Fin 2) (Fin 2) ℝ) →
  a + b + c + f = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l636_63648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l636_63613

theorem inequality_proof (a b c : ℝ) : 
  a = 2^(-(1/3 : ℝ)) →
  b = (2^(Real.log 3 / Real.log 2))^(-(1/2 : ℝ)) →
  c = Real.cos (50 * π / 180) * Real.cos (10 * π / 180) + Real.cos (140 * π / 180) * Real.sin (170 * π / 180) →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l636_63613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_reciprocal_sum_l636_63699

/-- The ellipse on which point M moves -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point C -/
noncomputable def C : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- Point D -/
noncomputable def D : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem minimum_reciprocal_sum :
  ∀ M : ℝ × ℝ, ellipse M.1 M.2 →
    1 / distance M C + 1 / distance M D ≥ 1 ∧
    ∃ M' : ℝ × ℝ, ellipse M'.1 M'.2 ∧ 1 / distance M' C + 1 / distance M' D = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_reciprocal_sum_l636_63699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l636_63628

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * (a ^ (x - 1)) + 1

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 3 ∧ ∀ x : ℝ, f a x = x → (x = 1 ∧ f a x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l636_63628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_value_l636_63643

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n ≥ 3, a n = a (n - 1) - a (n - 2)

theorem sequence_2016_value (a : ℕ → ℤ) (h : sequence_a a) : a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_value_l636_63643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l636_63674

open Real

noncomputable def f (x : ℝ) : ℝ :=
  -1 / (x + 1) + 1/2 * log (abs (x^2 + x + 1)) + 1 / sqrt 3 * arctan ((2*x + 1) / sqrt 3)

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ -1) :
  deriv f x = (x^3 + 4*x^2 + 4*x + 2) / ((x + 1)^2 * (x^2 + x + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l636_63674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l636_63692

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt (x^2) else -Real.sqrt (x^2)

-- Define the function g
def g (x : ℝ) : ℝ := x

-- Theorem statement
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l636_63692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l636_63690

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2 + 2*x + 3) / Real.log 10

-- State the theorem
theorem f_decreasing_on_interval : 
  ∀ x y, 1 < x ∧ x < y ∧ y < 3 → f x > f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l636_63690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l636_63627

-- Define the circle C
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 3

-- Define point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 5

-- Theorem statement
theorem circle_and_line_problem :
  -- Part 1: Point P is not on the circle
  (point_P.1 - circle_center.1)^2 + (point_P.2 - circle_center.2)^2 ≠ circle_radius^2 ∧
  -- Part 2: The line equation is either x = 0 or 3x + 4y - 8 = 0
  (∀ x y : ℝ, (x = 0 ∨ 3*x + 4*y - 8 = 0) ↔
    -- Line passes through P
    (x = point_P.1 ∧ y = point_P.2) ∨
    -- Line intersects the circle at two points with distance chord_length
    (∃ x1 y1 x2 y2 : ℝ,
      (x1 - circle_center.1)^2 + (y1 - circle_center.2)^2 = circle_radius^2 ∧
      (x2 - circle_center.1)^2 + (y2 - circle_center.2)^2 = circle_radius^2 ∧
      (x = 0 ∨ 3*x + 4*y - 8 = 0) ∧
      (x1 = 0 ∨ 3*x1 + 4*y1 - 8 = 0) ∧
      (x2 = 0 ∨ 3*x2 + 4*y2 - 8 = 0) ∧
      (x1 - x2)^2 + (y1 - y2)^2 = chord_length^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l636_63627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l636_63624

/-- A function that checks if a 4-digit number satisfies all conditions -/
def satisfiesConditions (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- 4-digit positive integer
  n % 2 = 0 ∧  -- multiple of 2
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits.toFinset.card = 4 ∧  -- all digits are different
  (n / 1000) ≠ 0 ∧  -- leading digit is not 0
  7 ∈ digits.toFinset ∧  -- 7 is one of the digits
  digits.all (· ≤ 7)  -- 7 is the largest digit

/-- The main theorem stating that the count of numbers satisfying the conditions is 690 -/
theorem count_satisfying_numbers : (Finset.filter (fun n => satisfiesConditions n) (Finset.range 10000)).card = 690 :=
  sorry

#eval (Finset.filter (fun n => satisfiesConditions n) (Finset.range 10000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l636_63624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l636_63615

/-- The area of a regular hexadecagon inscribed in a circle with radius r -/
noncomputable def hexadecagon_area (r : ℝ) : ℝ :=
  16 * (1 / 2) * r^2 * Real.sin (22.5 * Real.pi / 180)

/-- Theorem stating that the area of a regular hexadecagon inscribed in a circle
    with radius r is approximately equal to 3.061464r² -/
theorem hexadecagon_area_approx (r : ℝ) (h : r > 0) :
  ∃ ε > 0, |hexadecagon_area r - 3.061464 * r^2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l636_63615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_9_fourth_power_l636_63682

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x : ℝ, x ≥ 1 → f (g x) = x^2
axiom gf_condition : ∀ x : ℝ, x ≥ 1 → g (f x) = x^4
axiom g_81 : g 81 = 81

-- State the theorem to be proved
theorem g_9_fourth_power : (g 9)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_9_fourth_power_l636_63682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_increase_l636_63647

/-- The internal angle of a regular polygon with n sides -/
noncomputable def internal_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

/-- Theorem: If increasing the number of sides of a regular polygon by 9 results in each angle
    increasing by 9°, then the original polygon has 15 sides -/
theorem polygon_sides_increase (n : ℕ) (h : n > 2) :
  internal_angle (n + 9) = internal_angle n + 9 → n = 15 := by
  sorry

#check polygon_sides_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_increase_l636_63647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_through_origin_l636_63600

/-- Circle C with center (0, 2) and radius 4 -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 16

/-- Line of symmetry for circle C -/
def symmetry_line (a b : ℝ) (x y : ℝ) : Prop := a*x + b*y = 12

/-- Line on which point S lies -/
def line_S (b : ℝ) (y : ℝ) : Prop := y + b = 0

/-- Point S -/
structure PointS (b : ℝ) where
  x : ℝ
  y : ℝ
  on_line : line_S b y

/-- Tangent points A and B -/
structure TangentPoints where
  ax : ℝ
  ay : ℝ
  bx : ℝ
  by_ : ℝ  -- Changed 'by' to 'by_' to avoid keyword conflict
  on_circle_A : circle_C ax ay
  on_circle_B : circle_C bx by_

/-- Theorem: Line AB passes through (0, 0) -/
theorem line_AB_through_origin (a b : ℝ) (S : PointS b) (AB : TangentPoints) :
  ∃ (t : ℝ), t * AB.ax = 8 * AB.ay ∧ t * AB.bx = 8 * AB.by_ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_through_origin_l636_63600
