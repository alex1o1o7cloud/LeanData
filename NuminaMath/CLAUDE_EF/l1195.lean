import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_six_in_fraction_l1195_119548

-- Define the fraction
def fraction : ℚ := 11 / 13

-- Define the repeating decimal representation
def decimal_rep : List ℕ := [8, 4, 6, 1, 5, 3]

-- Define the probability of selecting a 6
def prob_six : ℚ := 1 / 6

-- Theorem statement
theorem probability_of_six_in_fraction :
  fraction.num = 11 ∧
  fraction.den = 13 ∧
  decimal_rep.length = 6 ∧
  decimal_rep.count 6 = 1 →
  prob_six = 1 / (decimal_rep.length : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_six_in_fraction_l1195_119548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_l1195_119527

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*(a^2 + 1)*y + 4 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the property of bisecting the circumference
def bisects_circumference (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation x y ∧ circle_equation x y a

-- State the theorem
theorem circle_bisection (a : ℝ) :
  bisects_circumference a → a = 0 ∨ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_l1195_119527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1195_119562

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 30)
  (sum2 : a + b + c + d = 50)
  (sum3 : a + b + c + d + e + f = 70)
  (odd_sum : Odd (e + f)) :
  ∃ (n : ℕ), n = 1 ∧ n = (
    (if Odd a then 1 else 0) + 
    (if Odd b then 1 else 0) + 
    (if Odd c then 1 else 0) + 
    (if Odd d then 1 else 0) + 
    (if Odd e then 1 else 0) + 
    (if Odd f then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1195_119562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_no_rational_satisfying_equation_integers_satisfying_equation_l1195_119529

-- Part 1
theorem distance_between_points (a b : ℚ) : 
  |a - b| = |b - a| :=
sorry

-- Part 2
theorem no_rational_satisfying_equation :
  ¬∃ x : ℚ, |x + 1| + |x - 3| = x :=
sorry

-- Part 3
theorem integers_satisfying_equation :
  {x : ℤ | |x - 4| + |x - 3| + |x + 3| + |x + 4| = 14} =
  {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_no_rational_satisfying_equation_integers_satisfying_equation_l1195_119529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1195_119518

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 4

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x ∈ Set.Ioo 1 2, f m x < 0) → m ∈ Set.Iic (-5) := by
  sorry

-- Part II
theorem part_two (m : ℝ) : 
  (∀ x : ℤ, |((f m x) - x^2) / m| < 1 ↔ x = 1 ∨ x = 2) → 
  m ∈ Set.Ioo (-4) (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1195_119518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x2_is_9_l1195_119560

-- Define the expression
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 2 * x) - 9 * (3 * x^2 - 2)

-- Define a function to extract the coefficient of x^2
noncomputable def coefficientOfX2 (f : ℝ → ℝ) : ℝ :=
  (deriv (deriv f) 0) / 2

-- Theorem statement
theorem coefficient_of_x2_is_9 : coefficientOfX2 expression = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x2_is_9_l1195_119560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_sum_l1195_119501

/-- The number of disks -/
def n : ℕ := 16

/-- The radius of the large circle -/
def R : ℝ := 1

/-- The radius of a small disk -/
noncomputable def r : ℝ := Real.tan (Real.pi / (2 * n))

/-- The sum of the areas of the small disks -/
noncomputable def total_area : ℝ := n * Real.pi * r^2

/-- The theorem to be proved -/
theorem disk_area_sum :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b ≥ 0 ∧ c > 0 ∧
    (∀ (p : ℕ), Nat.Prime p → c % (p^2) ≠ 0) ∧
    total_area = Real.pi * (a - b * Real.sqrt c) ∧
    a + b + c = 6337 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_sum_l1195_119501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_segment_length_l1195_119578

/-- Given a quadratic function y = x^2 + ax + b with specific properties,
    prove that the length of segment OC is 1. -/
theorem quadratic_segment_length
  (a b : ℝ) 
  (h1 : b ≠ 0)
  (h2 : ∃ (x y : ℝ), x^2 + a*x + b = y ∧ y = x) -- Line AB perpendicular to y = x
  (h3 : (0, b) ∈ {(x, y) | y = x^2 + a*x + b}) -- Point B on the curve
  (h4 : (b, 0) ∈ {(x, y) | y = x^2 + a*x + b}) -- Point A on the curve
  (h5 : Real.sqrt ((0 - b)^2 + (b - 0)^2) = Real.sqrt ((0 - b)^2 + (b - b)^2)) -- A and B symmetric to y = x
  : abs (1 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_segment_length_l1195_119578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_in_each_column_l1195_119582

/-- Represents the state of the square at any given row -/
structure SquareState where
  row : Fin 100 → Fin 100

/-- Generates the next row state based on the current row state -/
def nextRowState (s : SquareState) : SquareState where
  row := λ i =>
    if i < 11 then s.row (i + 89)
    else if i < 81 then s.row (i - 11)
    else s.row (i - 81)

/-- The initial state of the square -/
def initialState : SquareState where
  row := λ i => i

/-- Generates the nth row state -/
def nthRowState : Nat → SquareState
  | 0 => initialState
  | n + 1 => nextRowState (nthRowState n)

theorem all_numbers_in_each_column :
  ∀ (col : Fin 100),
    ∀ (num : Fin 100),
      ∃ (row : Fin 100),
        (nthRowState row.val).row col = num := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_in_each_column_l1195_119582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1195_119570

/-- The inverse proportion function f(x) = -3/x --/
noncomputable def f (x : ℝ) : ℝ := -3 / x

/-- Predicate to check if a point (x, y) is in Quadrant II --/
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in Quadrant IV --/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the graph of f(x) = -3/x is in Quadrants II and IV --/
theorem inverse_proportion_quadrants :
  ∀ x : ℝ, x ≠ 0 → (in_quadrant_II x (f x) ∨ in_quadrant_IV x (f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1195_119570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_tangency_proofs_l1195_119539

noncomputable def distance_point_to_line (x₀ y₀ k b : ℝ) : ℝ :=
  (|k * x₀ - y₀ + b|) / Real.sqrt (1 + k^2)

structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

noncomputable def distance_parallel_lines (k b₁ b₂ : ℝ) : ℝ :=
  |b₂ - b₁| / Real.sqrt (1 + k^2)

theorem distance_and_tangency_proofs :
  -- Part 1: Distance from point to line
  (distance_point_to_line 2 (-3) (-1) 3 = 2 * Real.sqrt 2) ∧
  -- Part 2: Circle tangent to line
  (let q : Circle := { center_x := 0, center_y := 5, radius := 2 }
   distance_point_to_line q.center_x q.center_y (Real.sqrt 3) 9 = q.radius) ∧
  -- Part 3: Distance between parallel lines
  (distance_parallel_lines (-3) (-2) 6 = 4 * Real.sqrt 10 / 5) :=
by sorry

#check distance_and_tangency_proofs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_tangency_proofs_l1195_119539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_average_wage_l1195_119535

theorem worker_average_wage (total_days : ℕ) (first_period : ℕ) (last_period : ℕ) 
  (first_avg : ℚ) (last_avg : ℚ) (middle_day_wage : ℚ) 
  (h1 : total_days = 15)
  (h2 : first_period = 7)
  (h3 : last_period = 7)
  (h4 : first_avg = 87)
  (h5 : last_avg = 90)
  (h6 : middle_day_wage = 111) :
  (first_period * first_avg + last_period * last_avg + middle_day_wage) / total_days = 90 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_average_wage_l1195_119535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_215_in_quadrant_III_l1195_119555

/-- Represents the four quadrants of a coordinate system. -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determines the quadrant of an angle in degrees. -/
noncomputable def angleQuadrant (angle : ℝ) : Quadrant :=
  if 0 ≤ angle && angle < 90 then Quadrant.I
  else if 90 ≤ angle && angle < 180 then Quadrant.II
  else if 180 ≤ angle && angle < 270 then Quadrant.III
  else Quadrant.IV

/-- Theorem stating that an angle of 215° is in Quadrant III. -/
theorem angle_215_in_quadrant_III :
  angleQuadrant 215 = Quadrant.III :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_215_in_quadrant_III_l1195_119555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_combined_results_l1195_119508

theorem average_of_combined_results 
  (count1 count2 count3 : ℕ) 
  (avg1 avg2 avg3 : ℚ) 
  (h1 : count1 > 0)
  (h2 : count2 > 0)
  (h3 : count3 > 0) :
  (count1 * avg1 + count2 * avg2 + count3 * avg3) / (count1 + count2 + count3) = 
  (count1 * avg1 + count2 * avg2 + count3 * avg3) / (count1 + count2 + count3) := by
  sorry

#eval (30 * 20 + 20 * 30 + 25 * 40) / (30 + 20 + 25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_combined_results_l1195_119508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_sum_approx_l1195_119510

/-- The sum of the double series ∑_{n=3}^∞ ∑_{k=2}^{n-1} (k^2 / 3^(n+k)) -/
noncomputable def double_series_sum : ℝ := ∑' n, ∑' k, if 3 ≤ n ∧ 2 ≤ k ∧ k < n then (k^2 : ℝ) / 3^(n + k) else 0

/-- The sum of the double series is approximately equal to 3/14 -/
theorem double_series_sum_approx :
  ∃ ε > 0, |double_series_sum - 3/14| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_series_sum_approx_l1195_119510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_exists_l1195_119534

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- Add necessary fields here

/-- Represents a division of an equilateral triangle into convex polygons --/
structure TriangleDivision where
  polygons : List ConvexPolygon
  is_valid : Bool

/-- Represents a line in 2D space --/
structure Line where
  -- Add necessary fields here

/-- Checks if a line intersects a polygon --/
def line_intersects_polygon (line : Line) (polygon : ConvexPolygon) : Bool :=
  sorry -- Implementation details

/-- Checks if a line intersects at most 40 polygons in the division --/
def line_intersects_at_most_40 (division : TriangleDivision) (line : Line) : Prop :=
  (division.polygons.filter (λ p => line_intersects_polygon line p)).length ≤ 40

/-- The main theorem statement --/
theorem equilateral_triangle_division_exists : 
  ∃ (division : TriangleDivision), 
    division.is_valid ∧ 
    division.polygons.length ≥ 1000000 ∧ 
    ∀ (line : Line), line_intersects_at_most_40 division line :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_exists_l1195_119534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_k_range_l1195_119550

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x

-- Theorem for the tangent line
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x, m * x + b = x ∧
  (∃ δ > 0, ∀ h ∈ Set.Ioo (-δ) δ, 
    |f (1 + h) - (f 1 + m * h)| ≤ |h| * |f (1 + h) - (f 1 + m * h)| / |h|) :=
by sorry

-- Theorem for the range of k
theorem k_range (k : ℝ) :
  (∀ x ∈ Set.Ioi 1, k / x + x / 2 - f x / x < 0) →
  k ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_k_range_l1195_119550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_leftover_milk_l1195_119549

/-- Calculates the amount of milk left over after making milkshakes -/
theorem milkshake_leftover_milk (milk_per_shake ice_cream_per_shake total_milk total_ice_cream : ℕ) : 
  milk_per_shake = 4 →
  ice_cream_per_shake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  total_milk - milk_per_shake * (total_ice_cream / ice_cream_per_shake) = 8 := by
  intros h1 h2 h3 h4
  -- Replace the entire proof with sorry
  sorry

#check milkshake_leftover_milk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_leftover_milk_l1195_119549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_balloon_theorem_l1195_119524

/-- The maximum number of balloons Orvin can buy under the given conditions -/
def max_balloons : ℕ := 52

/-- The regular price of a balloon in cents -/
def p : ℕ → ℕ := fun _ => 1  -- We use a function to avoid 'variable' declaration

/-- Orvin's initial budget in cents -/
def initial_budget (p : ℕ → ℕ) : ℕ := 40 * p 0

/-- The cost of two balloons under the sale conditions in cents -/
def cost_of_two (p : ℕ → ℕ) : ℕ := p 0 + p 0 / 2

/-- The number of sets of two balloons Orvin can buy -/
def sets_bought (p : ℕ → ℕ) : ℕ := initial_budget p / cost_of_two p

theorem orvin_balloon_theorem (p : ℕ → ℕ) :
  2 * sets_bought p = max_balloons := by
  sorry

#eval max_balloons
#eval sets_bought p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_balloon_theorem_l1195_119524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_pricing_and_kitchen_max_l1195_119569

/-- Represents the unit price of colored tiles in yuan -/
def colored_price : ℕ → Prop := sorry

/-- Represents the unit price of single-colored tiles in yuan -/
def single_colored_price : ℕ → Prop := sorry

/-- Represents the total cost in yuan for a given number of colored and single-colored tiles -/
def total_cost : ℕ → ℕ → ℕ → Prop := sorry

/-- Represents the maximum number of colored tiles that can be purchased for the kitchen -/
def max_colored_kitchen : ℕ → Prop := sorry

theorem tile_pricing_and_kitchen_max (c s : ℕ) :
  (∀ x y, colored_price x → single_colored_price y →
    total_cost 40 60 5600 ∧ total_cost 50 50 6000) →
  (∀ a, max_colored_kitchen a →
    total_cost a (60 - a) 3400) →
  colored_price 80 ∧ single_colored_price 40 ∧ max_colored_kitchen 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_pricing_and_kitchen_max_l1195_119569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1195_119512

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (4 * x + 2) / Real.sqrt (x - 7)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 7} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1195_119512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1195_119547

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 4*m)

theorem function_properties (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- symmetric about y-axis
  (f m 2 > f m 3) →              -- f(2) > f(3)
  (m = 2) ∧                      -- m = 2
  (∀ a : ℝ, f m (a + 2) < f m (1 - 2*a) ↔ 
    (a > -1/3 ∧ a < 1/2) ∨ (a > 1/2 ∧ a < 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1195_119547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1195_119533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the relation of a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the relation of two lines intersecting
variable (intersect : Line → Line → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the theorem
theorem geometry_theorem 
  (l m : Line) (α β : Plane) 
  (h_diff_lines : l ≠ m) 
  (h_diff_planes : α ≠ β) :
  (∀ (l1 l2 : Line), line_in_plane l1 α → line_in_plane l2 α → 
    intersect l1 l2 → perp_line_line l l1 → perp_line_line l l2 → 
    perp_line_plane l α) ∧
  (line_in_plane l β → perp_line_plane l α → perp_plane_plane α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1195_119533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1195_119598

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def line_equation (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

def circle_equation (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem ellipse_properties (e : Ellipse) 
  (F : Point) (A : Point) (P : Point) (Q : Point) (M : Point) (l : Line) (C : Circle) :
  ellipse_equation e F →  -- F is on the ellipse
  ellipse_equation e A →  -- A is on the ellipse
  ellipse_equation e P →  -- P is on the ellipse
  F.y = 0 →               -- F is on x-axis
  A.x = 0 →               -- A is on y-axis
  A.y = e.b →             -- A is the top vertex
  Q.y = 0 →               -- Q is on x-axis
  Q.x > 0 →               -- Q is on positive x-axis
  (P.x - A.x) * (F.x - A.x) + (P.y - A.y) * (F.y - A.y) = 0 →  -- AP ⟂ AF
  8 * (Q.x - P.x) = 5 * (P.x - A.x) →  -- AP:PQ = 8:5
  M.x = -3 ∧ M.y = 0 →    -- M is at (-3, 0)
  l.m = Real.sqrt 3 / 3 →  -- l has inclination π/6
  line_equation l M →     -- l passes through M
  circle_equation C A →   -- A is on circle C
  circle_equation C Q →   -- Q is on circle C
  circle_equation C F →   -- F is on circle C
  ∃ p : Point, line_equation l p ∧ circle_equation C p ∧
    ∀ q : Point, line_equation l q → circle_equation C q → q = p →  -- l is tangent to C
  eccentricity e = 1/2 ∧ e.a = 2 ∧ e.b = Real.sqrt 3 :=
by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1195_119598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_separate_l1195_119526

-- Define the circle and line equations
def circle_eq (x y c : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + c = 0
def line_eq (x y c : ℝ) : Prop := 2*x + 2*y + c = 0

-- Theorem statement
theorem circle_line_separate (c : ℝ) (hc : c ≠ 0) : 
  ∀ x y : ℝ, ¬(circle_eq x y c ∧ line_eq x y c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_separate_l1195_119526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1195_119568

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y))

/-- Theorem: The vertical line x = 2 divides the triangle ABC into two equal areas -/
theorem equal_area_division (A B C : Point) (a : ℝ) : 
  A.x = 0 → A.y = 3 → B.x = 0 → B.y = 0 → C.x = 4 → C.y = 0 → a = 2 →
  let t := Triangle.mk A B C
  let leftArea := triangleArea (Triangle.mk A B ⟨a, 0⟩)
  let rightArea := triangleArea (Triangle.mk A ⟨a, 0⟩ C)
  leftArea = rightArea := by
    intro hAx hAy hBx hBy hCx hCy ha
    sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1195_119568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1195_119517

-- Define the function g(x)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + 2*x - m + m/x

-- State the theorem
theorem g_properties (m : ℝ) (h_m : m > 0) 
  (h_increasing : ∀ x ≥ 1, StrictMono (g m)) :
  (∃ (max_m : ℝ), max_m = 3 ∧ ∀ m' > 0, m' ≤ max_m) ∧
  (∃ (Q : ℝ × ℝ), Q = (0, -3) ∧ 
    ∀ x ≠ 0, g 3 (-x) = -g 3 x + 2*Q.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1195_119517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_congruence_l1195_119546

def sumOfDigits (n : ℕ) : ℕ := sorry

def isPairwiseCoprime (a b c : ℕ) : Prop := sorry

theorem sum_of_digits_congruence (m : ℕ) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ isPairwiseCoprime a b c ∧
    sumOfDigits (a * b) = m ∧
    sumOfDigits (a * c) = m ∧
    sumOfDigits (b * c) = m) →
  m % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_congruence_l1195_119546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_two_l1195_119589

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The logarithmic function we're considering --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a * x - 1) / (2 * x + 1))

/-- Theorem stating that if f is odd, then a must equal 2 --/
theorem odd_function_implies_a_equals_two :
  ∃ a : ℝ, IsOdd (f a) → a = 2 := by
  -- We claim that a = 2 satisfies the condition
  use 2
  -- Now we need to prove that if f with a=2 is odd, then indeed a = 2
  intro h
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_two_l1195_119589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_nine_fourths_l1195_119536

-- Define the curve y = √x
noncomputable def curve (x : ℝ) : ℝ := Real.sqrt x

-- Define the points A, B, and C
noncomputable def A : ℝ × ℝ := (1, curve 1)
noncomputable def B (m : ℝ) : ℝ × ℝ := (m, curve m)
noncomputable def C : ℝ × ℝ := (4, curve 4)

-- Define the area of the triangle ABC
noncomputable def triangle_area (m : ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B m
  let (x₃, y₃) := C
  (1/2) * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

-- Theorem statement
theorem max_area_at_nine_fourths (m : ℝ) (h1 : 1 < m) (h2 : m < 4) :
  ∃ (max_m : ℝ), max_m = 9/4 ∧ 
  ∀ (n : ℝ), 1 < n ∧ n < 4 → triangle_area n ≤ triangle_area max_m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_nine_fourths_l1195_119536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l1195_119583

/-- The diameter of each cylindrical pipe in cm -/
def pipe_diameter : ℝ := 8

/-- The number of pipes in each crate -/
def total_pipes : ℕ := 250

/-- The number of rows in the direct stacking method -/
def direct_rows : ℕ := 25

/-- The number of pipes per row in the direct stacking method -/
def direct_pipes_per_row : ℕ := 10

/-- The height of the direct stacking method in cm -/
def direct_height : ℝ := direct_rows * pipe_diameter

/-- The vertical distance between layers in the honeycomb pattern -/
noncomputable def honeycomb_layer_distance : ℝ := 4 * Real.sqrt 3 - 4

/-- The height of the honeycomb pattern stacking in cm -/
noncomputable def honeycomb_height : ℝ := direct_rows * honeycomb_layer_distance

/-- The positive difference in heights between the two packing methods -/
noncomputable def height_difference : ℝ := direct_height - honeycomb_height

theorem packing_height_difference : 
  height_difference = 300 - 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l1195_119583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_area_l1195_119532

noncomputable def f (x : ℝ) := x + Real.sin x

theorem f_range_and_area :
  let a := Real.pi / 2
  let b := Real.pi
  (∀ y ∈ Set.Icc (f a) (f b), ∃ x ∈ Set.Icc a b, f x = y) ∧
  (Set.Icc (f a) (f b) = Set.Icc (Real.pi / 2 + 1) Real.pi) ∧
  (∫ x in a..b, f x = 3 * Real.pi^2 / 8 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_area_l1195_119532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1195_119523

/-- The total shaded area given two circles with radii 3 and 6 feet -/
noncomputable def total_shaded_area (r₁ r₂ : ℝ) : ℝ :=
  let rect₁_area := r₁ * (2 * r₁)
  let semicircle₁_area := (1/2) * Real.pi * r₁^2
  let rect₂_area := r₂ * (2 * r₂)
  let semicircle₂_area := (1/2) * Real.pi * r₂^2
  (rect₁_area - semicircle₁_area) + (rect₂_area - semicircle₂_area)

/-- Theorem stating that the total shaded area for circles with radii 3 and 6 feet is 90 - (45π/2) -/
theorem shaded_area_calculation :
  total_shaded_area 3 6 = 90 - (45 * Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1195_119523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_pi_approximates_true_pi_l1195_119519

/-- Experimental value of π based on random point simulation in a square --/
noncomputable def experimental_pi (n m : ℕ) : ℝ :=
  9 * (m : ℝ) / (n : ℝ)

/-- The square used in the simulation --/
structure SimulationSquare where
  side_length : ℝ
  total_points : ℕ
  close_points : ℕ

/-- Properties of the simulation square --/
def valid_simulation_square (s : SimulationSquare) : Prop :=
  s.side_length = 3 ∧
  s.close_points ≤ s.total_points ∧
  s.total_points > 0

/-- Theorem: The experimental π value approximates the true π value --/
theorem experimental_pi_approximates_true_pi (s : SimulationSquare) 
  (h : valid_simulation_square s) :
  ∃ ε > 0, |experimental_pi s.total_points s.close_points - Real.pi| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_pi_approximates_true_pi_l1195_119519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_two_range_of_a_when_f_geq_four_l1195_119588

/-- The function f(x) defined as |x - a^2| + |x - 2a + 1| -/
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

/-- Theorem for part (1) -/
theorem solution_set_when_a_is_two :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

/-- Theorem for part (2) -/
theorem range_of_a_when_f_geq_four :
  {a : ℝ | ∀ x, f x a ≥ 4} = Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_two_range_of_a_when_f_geq_four_l1195_119588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_all_colors_l1195_119590

/-- Represents a coloring of edges in a complete graph -/
def Coloring (n : ℕ) := Fin (3*n + 1) → Fin (3*n + 1) → Fin 3

/-- A valid coloring satisfies the condition that each vertex is connected to
    exactly n edges of each color -/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ v : Fin (3*n + 1), ∀ color : Fin 3,
    (Finset.filter (λ w => c v w = color) Finset.univ).card = n

/-- The main theorem: In a complete graph with 3n + 1 vertices, where each edge
    is colored with one of three colors, and each vertex is connected to exactly
    n edges of each color, there exists a triangle with all three colors -/
theorem exists_triangle_with_all_colors (n : ℕ) (c : Coloring n) 
  (h : is_valid_coloring n c) :
  ∃ (v1 v2 v3 : Fin (3*n + 1)), 
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
    c v1 v2 ≠ c v2 v3 ∧ c v2 v3 ≠ c v1 v3 ∧ c v1 v2 ≠ c v1 v3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_all_colors_l1195_119590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_set_part2_a_range_l1195_119528

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 2 x ≥ 2 * x} = Set.Iic (5/2) :=
sorry

-- Part 2
theorem part2_a_range :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ (1/2) * a + 5) → a ∈ Set.Icc (-16/3) 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_set_part2_a_range_l1195_119528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1195_119531

theorem expression_evaluation : 
  (1/3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (60 * π / 180) + (π - 2016)^0 - (8 : ℝ)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1195_119531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l1195_119575

noncomputable section

-- Define the line l₀
def l₀ (x y : ℝ) : Prop := x + 2 * y + 4 = 0

-- Define the function y = 1/(1-x)
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

-- Part I
theorem part_I (m n : ℝ) (h₁ : m = 1/2) (h₂ : n = f m) :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    (y - n = k * (x - m)) ∧ (k * (-1/2) = -1) →
    2 * x - y + 1 = 0 := by sorry

-- Part II
theorem part_II (m n : ℝ) (h : l₀ m n) :
  ∀ (x y : ℝ), m * x + (n - 1) * y + n + 5 = 0 → x = 1 ∧ y = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l1195_119575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1195_119503

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)
def C (m : ℝ) : ℝ × ℝ := (4, m + 1)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC (m : ℝ) : ℝ × ℝ := ((C m).1 - B.1, (C m).2 - B.2)

theorem vector_properties (m : ℝ) :
  (AB.1^2 + AB.2^2 = 5) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ AB.1 * k = (BC m).1 ∧ AB.2 * k = (BC m).2) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1195_119503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_height_l1195_119507

/-- Represents a triangle -/
structure Triangle where
  -- You can add more fields if needed, but for this problem we only need a placeholder
  dummy : Unit

/-- Checks if two triangles are similar -/
def Similar (t1 t2 : Triangle) : Prop := sorry

/-- Calculates the ratio of areas between two triangles -/
def AreaRatio (t1 t2 : Triangle) : ℚ := sorry

/-- Calculates the height of a triangle -/
def Height (t : Triangle) : ℚ := sorry

/-- Given two similar triangles with an area ratio of 1:9 and the smaller triangle
    having a height of 5 cm, the corresponding height of the larger triangle is 15 cm. -/
theorem similar_triangles_height (triangle1 triangle2 : Triangle) 
  (h_similar : Similar triangle1 triangle2)
  (h_area_ratio : AreaRatio triangle1 triangle2 = 1 / 9)
  (h_height_small : Height triangle1 = 5) : Height triangle2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_height_l1195_119507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_and_binomial_expansion_l1195_119504

/-- Given a parabola y^2 = ax (a > 0) and the area enclosed by this parabola
    and the line x = 1 is 4/3, prove that the coefficient of x^(-18) in the
    expansion of (x + a/x)^20 is 20. -/
theorem parabola_area_and_binomial_expansion (a : ℝ) : 
  a > 0 → 
  (∫ x in (Set.Icc 0 1 : Set ℝ), 2 * Real.sqrt (a * x)) = 4/3 → 
  (Finset.range 21).sum (fun k => Nat.choose 20 k * a^k * (-1)^(19 - k)) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_and_binomial_expansion_l1195_119504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l1195_119521

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 6*y - 4

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 3)

/-- The point we're measuring the distance to -/
def point : ℝ × ℝ := (10, 10)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_circle_center :
  distance circle_center point = Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l1195_119521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_increasing_l1195_119580

-- Define the function f(x) = ∛x
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem cubic_root_increasing : Monotone f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_increasing_l1195_119580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_worked_56_hours_l1195_119541

/-- The total hours worked by Thomas, Toby, and Rebecca -/
def total_hours : ℕ := 157

/-- Thomas's working hours -/
def thomas_hours : ℕ → ℕ := id

/-- Toby's working hours in terms of Thomas's -/
def toby_hours (x : ℕ) : ℕ := 2 * x - 10

/-- Rebecca's working hours in terms of Thomas's -/
def rebecca_hours (x : ℕ) : ℕ := toby_hours x - 8

/-- Theorem stating that Rebecca worked 56 hours -/
theorem rebecca_worked_56_hours :
  (∃ x : ℕ, thomas_hours x + toby_hours x + rebecca_hours x = total_hours) →
  rebecca_hours 37 = 56 := by
  sorry

#eval rebecca_hours 37

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_worked_56_hours_l1195_119541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_apples_and_oranges_l1195_119571

/-- Represents a box containing apples and oranges -/
structure Box where
  apples : ℕ
  oranges : ℕ

/-- Represents the collection of all boxes -/
def Boxes := Fin 99 → Box

/-- Represents a selection of boxes -/
def Selection := Fin 50 → Fin 99

/-- The sum of apples in a selection of boxes -/
def sumApples (boxes : Boxes) (selection : Selection) : ℕ :=
  Finset.sum (Finset.range 50) fun i => (boxes (selection i)).apples

/-- The sum of oranges in a selection of boxes -/
def sumOranges (boxes : Boxes) (selection : Selection) : ℕ :=
  Finset.sum (Finset.range 50) fun i => (boxes (selection i)).oranges

/-- The total number of apples in all boxes -/
def totalApples (boxes : Boxes) : ℕ :=
  Finset.sum (Finset.range 99) fun i => (boxes i).apples

/-- The total number of oranges in all boxes -/
def totalOranges (boxes : Boxes) : ℕ :=
  Finset.sum (Finset.range 99) fun i => (boxes i).oranges

theorem half_apples_and_oranges (boxes : Boxes) :
  ∃ (selection : Selection),
    2 * sumApples boxes selection ≥ totalApples boxes ∧
    2 * sumOranges boxes selection ≥ totalOranges boxes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_apples_and_oranges_l1195_119571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_property_l1195_119509

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 6 = 1

-- Define the foci
def left_focus (F₁ : ℝ × ℝ) : Prop := ∃ c : ℝ, F₁ = (-c, 0) ∧ c^2 = 3
def right_focus (F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, F₂ = (c, 0) ∧ c^2 = 3

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem ellipse_foci_property (P F₁ F₂ : ℝ × ℝ) :
  ellipse P.1 P.2 →
  left_focus F₁ →
  right_focus F₂ →
  distance P F₁ = 2 →
  distance P F₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_property_l1195_119509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_list_l1195_119513

theorem max_value_in_list (S : List ℕ) (has_68 : 68 ∈ S) 
  (avg_with_68 : (S.sum + 68) / S.length = 56)
  (avg_without_68 : S.sum / (S.length - 1) = 55) :
  ∃ x ∈ S, x ≤ 649 ∧ ∀ y ∈ S, y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_list_l1195_119513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_specific_case_l1195_119552

/-- The length of a ladder leaning against a wall --/
noncomputable def ladder_length (initial_distance : ℝ) (slipped_distance : ℝ) : ℝ :=
  let initial_height := Real.sqrt (initial_distance^2 + slipped_distance^2 - 2 * slipped_distance - 80)
  Real.sqrt (initial_distance^2 + initial_height^2)

/-- Theorem stating the length of the ladder given specific conditions --/
theorem ladder_length_specific_case :
  abs (ladder_length 9 10.07212046142853 - 13.9965) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_specific_case_l1195_119552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l1195_119586

def my_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a (n + 2) + a n) ∧ 
  a 1 = 2 ∧ 
  a 2 = 5

theorem sixth_term_value (a : ℕ → ℤ) (h : my_sequence a) : a 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l1195_119586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_is_negative_45_l1195_119566

/-- Represents a geometric sequence with a given common ratio and sum of first two terms -/
structure GeometricSequence where
  r : ℝ  -- common ratio
  sum_first_two : ℝ  -- sum of first two terms

/-- Returns the nth term of a geometric sequence -/
noncomputable def nth_term (gs : GeometricSequence) (n : ℕ) : ℝ :=
  let a₁ := gs.sum_first_two / (1 + gs.r)
  a₁ * gs.r^(n - 1)

/-- Theorem stating that for a geometric sequence with common ratio -3 and sum of first two terms 10, the third term is -45 -/
theorem third_term_is_negative_45 (gs : GeometricSequence) 
  (h1 : gs.r = -3)
  (h2 : gs.sum_first_two = 10) :
  nth_term gs 3 = -45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_is_negative_45_l1195_119566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_neg_sqrt_two_l1195_119554

noncomputable def f (n : ℕ) : ℝ := Real.cos ((n * Real.pi / 2) + (Real.pi / 4))

theorem sum_of_f_equals_neg_sqrt_two :
  (Finset.range 2018).sum f = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_neg_sqrt_two_l1195_119554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1195_119581

theorem expression_equality : 
  (Real.sqrt 2 - 1) ^ (0 : ℤ) - (1 / 3) ^ (-1 : ℤ) - Real.sqrt 8 - Real.sqrt ((-2 : ℝ) ^ 2) = -4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1195_119581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equality_l1195_119597

theorem power_sum_equality (p : ℚ) (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (p ≤ 0 → ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (x + y : ℝ)^(p : ℝ) ≠ x^(p : ℝ) + y^(p : ℝ)) ∧
  ((0 < p ∧ p ≠ 1) → ((x + y : ℝ)^(p : ℝ) = x^(p : ℝ) + y^(p : ℝ) ↔ x = 0 ∨ y = 0)) ∧
  (p = 1 → (x + y : ℝ)^(p : ℝ) = x^(p : ℝ) + y^(p : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equality_l1195_119597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1195_119573

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (|x| + 1) / Real.log 2 + 2^x + 2^(-x)

-- State the theorem
theorem f_inequality_range (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x < -1/3 ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1195_119573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrices_intersect_infinite_parallel_lines_l1195_119595

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

structure CircularCone where

-- Define the concept of generatrix
def Generatrix (c : CircularCone) : Type := Line

-- Define the parallel relation
def Parallel : Line → Line → Prop := sorry

-- Define the parallel relation between a line and a plane
def ParallelToPlane : Line → Plane → Prop := sorry

-- Define the "in plane" relation for a line
def InPlane : Line → Plane → Prop := sorry

-- Define a membership relation for a point on a line
def PointOnLine : Point → Line → Prop := sorry

-- Theorem 1: The extensions of any two generatrices of a circular cone intersect at a point
theorem generatrices_intersect (c : CircularCone) (g1 g2 : Generatrix c) :
  ∃ (p : Point), PointOnLine p g1 ∧ PointOnLine p g2 :=
sorry

-- Theorem 2: If a line is parallel to a plane, then there are infinitely many lines in the plane parallel to it
theorem infinite_parallel_lines (a : Line) (α : Plane) (h : ParallelToPlane a α) :
  ∃ (s : Set Line), (∀ l ∈ s, InPlane l α ∧ Parallel l a) ∧ Set.Infinite s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrices_intersect_infinite_parallel_lines_l1195_119595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1195_119500

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := 1/x
noncomputable def h (x : ℝ) : ℝ := x^2
noncomputable def k (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem function_properties :
  (∃ y, y ≤ 0 ∧ f y > 1) ∧
  (∃ y, y > 2 ∧ g y > 1/2) ∧
  (∃ s : Set ℝ, s ≠ {x | -2 ≤ x ∧ x ≤ 2} ∧ 
    (∀ x ∈ s, 0 ≤ h x ∧ h x ≤ 4) ∧
    (∀ y, 0 ≤ y ∧ y ≤ 4 → ∃ x ∈ s, h x = y)) ∧
  (∀ x, 0 < x ∧ x ≤ 8 ↔ k x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1195_119500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1195_119565

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - 1

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ x, f a x ≥ 0 ↔ x ∈ Set.Iic (-1/2) ∪ Set.Ici 1) →
  (∃ a, ∀ x, f a x ≤ 1 - x^2 ↔ x ∈ Set.Icc (-2/3) 1) ∧
  (∃ m, ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a (Real.log x) + 5 > m * Real.log x) ∧
  (∀ m, (∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a (Real.log x) + 5 > m * Real.log x) → m < 4 * Real.sqrt 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1195_119565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_target_vertex_l1195_119596

-- Define a point in 3D space
structure Point3D where
  x : Int
  y : Int
  z : Int
deriving Inhabited

-- Define the set of initial vertices
def initialVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 1⟩, ⟨0, 1, 0⟩, ⟨1, 0, 0⟩,
  ⟨1, 1, 0⟩, ⟨0, 1, 1⟩, ⟨1, 0, 1⟩
]

-- Define the symmetry operation
def symmetryOperation (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

-- Define the target vertex
def targetVertex : Point3D := ⟨1, 1, 1⟩

-- Theorem: It's impossible to reach the target vertex from the initial vertices
theorem cannot_reach_target_vertex :
  ∀ (sequence : List Point3D),
    sequence.head? = some targetVertex →
    (∀ i, i > 0 → i < sequence.length →
      ∃ j, j < i ∧ sequence[i]? = some (symmetryOperation (sequence.get! j) (sequence.get! (j+1)))) →
    ¬(sequence.head? ∈ initialVertices.map some) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_target_vertex_l1195_119596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l1195_119579

-- Define the circle
def my_circle (a b : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 9/4

-- Define the parabola
def my_parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the conditions
def conditions (a b p : ℝ) : Prop :=
  p > 0 ∧
  my_circle a b 0 0 ∧
  my_parabola p a b ∧
  ∃ (y : ℝ), my_circle a b 0 y ∧ y = -p/2

-- Define the triangle area function
noncomputable def triangle_area (k : ℝ) : ℝ := 4*(1 + k^2)^(3/2)

-- State the theorem
theorem parabola_circle_theorem (a b p : ℝ) :
  conditions a b p →
  (∀ x y, my_parabola p x y ↔ x^2 = 4*y) ∧
  (∀ k, triangle_area k ≥ 4) ∧
  (triangle_area 0 = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l1195_119579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_unit_circle_l1195_119591

/-- The maximum value of √3y_A + x_B on the unit circle -/
theorem max_value_on_unit_circle : ∃ (M : ℝ),
  (∀ (x_A y_A x_B y_B : ℝ),
    (x_A^2 + y_A^2 = 1) →  -- A is on the unit circle
    (x_B^2 + y_B^2 = 1) →  -- B is on the unit circle
    (x_B = x_A * (1/2) - y_A * (Real.sqrt 3/2)) →  -- B is rotated π/3 from A
    (y_B = x_A * (Real.sqrt 3/2) + y_A * (1/2)) →  -- B is rotated π/3 from A
    (Real.sqrt 3 * y_A + x_B ≤ M)) ∧
  (∃ (x_A y_A x_B y_B : ℝ),
    (x_A^2 + y_A^2 = 1) ∧
    (x_B^2 + y_B^2 = 1) ∧
    (x_B = x_A * (1/2) - y_A * (Real.sqrt 3/2)) ∧
    (y_B = x_A * (Real.sqrt 3/2) + y_A * (1/2)) ∧
    (Real.sqrt 3 * y_A + x_B = M)) ∧
  M = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_unit_circle_l1195_119591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_prime_factors_l1195_119564

-- Define the properties of integers m and n
def has_four_prime_factors (m : ℕ) : Prop := ∃ p₁ p₂ p₃ p₄ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  ∃ k : ℕ, m = p₁ * p₂ * p₃ * (p₄^2) * k ∧ Nat.Coprime k (p₁ * p₂ * p₃ * p₄)

def has_three_prime_factors (n : ℕ) : Prop := ∃ q₁ q₂ q₃ : ℕ, Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧
  q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
  ∃ l : ℕ, n = q₁ * q₂ * q₃ * l ∧ Nat.Coprime l (q₁ * q₂ * q₃)

def distinct_prime_factors (m n : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → (p ∣ m → ¬(p ∣ n)) ∧ (p ∣ n → ¬(p ∣ m))

-- Theorem statement
theorem product_prime_factors (m n : ℕ) :
  has_four_prime_factors m →
  has_three_prime_factors n →
  distinct_prime_factors m n →
  Nat.gcd m n = 15 →
  ∃ p₁ p₂ p₃ p₄ p₅ p₆ p₇ : ℕ,
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ Nat.Prime p₆ ∧ Nat.Prime p₇ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧ p₁ ≠ p₇ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧ p₂ ≠ p₇ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧ p₃ ≠ p₇ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧ p₄ ≠ p₇ ∧
    p₅ ≠ p₆ ∧ p₅ ≠ p₇ ∧
    p₆ ≠ p₇ ∧
    ∃ k : ℕ, m * n = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇ * k ∧ Nat.Coprime k (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_prime_factors_l1195_119564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1195_119594

-- Define the sequence
def c : ℕ → ℕ
  | 0 => 3  -- Add this case to handle n = 0
  | 1 => 3
  | 2 => 5
  | n + 3 => c (n + 2) * c (n + 1)

-- State the theorem
theorem c_15_value : c 15 = 3^235 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1195_119594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_4_2048_closest_to_8_l1195_119540

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem harmonic_mean_4_2048_closest_to_8 :
  let hm := harmonic_mean 4 2048
  (abs (hm - 8) < abs (hm - 7)) ∧
  (abs (hm - 8) < abs (hm - 9)) ∧
  (abs (hm - 8) < abs (hm - 10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_4_2048_closest_to_8_l1195_119540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1195_119525

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x + 2 else -x^2 + 2*x + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (f (a^2 - 4*a) + f (-4) > 15) ↔ (a < -1 ∨ a > 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1195_119525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1195_119520

-- Define the system of equations
def system_of_equations (a x y : ℝ) : Prop :=
  (a * x - 5 * y = 5) ∧ (x / (x + y) = 5 / 7) ∧ (x - y = 3)

-- Theorem statement
theorem solve_for_a :
  ∀ a x y : ℝ, system_of_equations a x y → a = 3 :=
by
  intros a x y h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1195_119520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_journey_time_proof_l1195_119592

/-- The time required to travel between two towns at a given speed, given the time and speed of a previous journey -/
theorem travel_time (initial_time initial_speed new_speed : ℝ) :
  initial_time > 0 ∧ initial_speed > 0 ∧ new_speed > 0 →
  (initial_time * initial_speed) / new_speed = 
    initial_time * (initial_speed / new_speed) :=
by sorry

/-- Proof that the journey takes 4.8 hours at 50 mph given it takes 3 hours at 80 mph -/
theorem journey_time_proof :
  (3 : ℝ) * 80 / 50 = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_journey_time_proof_l1195_119592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_composite_l1195_119574

def is_valid_divisor_sequence (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin (k + 1) → ℕ),
    d 0 = 1 ∧ d (Fin.last k) = n ∧
    (∀ i : Fin k, d i < d (i.succ)) ∧
    (∀ m : ℕ, m ∣ n ↔ ∃ i : Fin (k + 1), d i = m) ∧
    (∀ i : Fin k, (d (i.succ) - d i) = (i.val + 1) * (d 1 - d 0))

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem unique_valid_composite : ∀ n : ℕ, 
  is_composite n → (is_valid_divisor_sequence n ↔ n = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_composite_l1195_119574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1195_119522

open Real

/-- The function f defined on the positive octant of ℝ³ --/
noncomputable def f (x y z : ℝ) : ℝ :=
  (x * sqrt y + y * sqrt z + z * sqrt x) / sqrt ((x + y) * (y + z) * (z + x))

theorem f_range :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  0 < f x y z ∧ f x y z ≤ 3 / (2 * sqrt 2) ∧
  (∀ ε > 0, ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ f x y z < ε) ∧
  (∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ f x y z = 3 / (2 * sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1195_119522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l1195_119599

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃! x, a*x^2 + b*x + c = 0

/-- The main theorem: if the line 4x + 7y + k = 0 is tangent to the parabola y^2 = 16x, then k = 49 -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y, 4*x + 7*y + k = 0 → y^2 = 16*x) →
  (∃! y, ∃ x, 4*x + 7*y + k = 0 ∧ y^2 = 16*x) →
  k = 49 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check line_tangent_to_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l1195_119599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1195_119563

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 + 3 * x + 5 * y = 25

-- Define the slope of the tangent at a given point
noncomputable def tangent_slope (x : ℝ) : ℝ :=
  -4/5 * x - 3/5

-- Theorem statement
theorem tangent_slope_at_one :
  tangent_slope 1 = -7/5 := by
  -- Unfold the definition of tangent_slope
  unfold tangent_slope
  -- Simplify the expression
  simp
  -- The result follows from arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1195_119563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_circle_tangency_l1195_119538

/-- The radius of the circle to which eight parabolas are tangent -/
noncomputable def circle_radius : ℝ := 1/4

theorem parabolas_circle_tangency (r : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + r = x → (x^2 - x + r = 0)) 
  (h2 : ∃! x : ℝ, x^2 - x + r = 0) : 
  r = circle_radius := by
  sorry

#check parabolas_circle_tangency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_circle_tangency_l1195_119538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1195_119585

theorem solve_exponential_equation (y : ℝ) :
  (27 : ℝ)^(3*y - 4) = (1/3 : ℝ)^(2*y + 6) → y = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1195_119585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1195_119557

/-- Given a line l: kx - y - 3 = 0 intersecting a circle O: x^2 + y^2 = 4 at points A and B,
    if OA · OB = 2, then k = ±√2. -/
theorem line_circle_intersection (k : ℝ) (A B : ℝ × ℝ) : 
  (∀ x y, k * x - y - 3 = 0 → x^2 + y^2 = 4) →  -- Line l intersects circle O
  A ∈ {p : ℝ × ℝ | k * p.1 - p.2 - 3 = 0 ∧ p.1^2 + p.2^2 = 4} →  -- A is on both line and circle
  B ∈ {p : ℝ × ℝ | k * p.1 - p.2 - 3 = 0 ∧ p.1^2 + p.2^2 = 4} →  -- B is on both line and circle
  A.1 * B.1 + A.2 * B.2 = 2 →  -- OA · OB = 2
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1195_119557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x2_f_l1195_119545

/-- The function f(x) = (1+x^2)(1-2x)^5 -/
noncomputable def f (x : ℝ) : ℝ := (1 + x^2) * (1 - 2*x)^5

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := deriv f

/-- The coefficient of x^2 in the expanded form of f'(x) -/
noncomputable def coeff_x2_f' : ℝ := sorry

theorem coeff_x2_f'_eq_neg270 : coeff_x2_f' = -270 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x2_f_l1195_119545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_point_rotation_l1195_119556

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  center : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For any equilateral triangle with side length a and a point P with
    positive distances p, q, and r to its vertices, there exists an equilateral
    triangle with side length a and a point Q with distances q, r, and p to its vertices -/
theorem equilateral_triangle_point_rotation
  (t : EquilateralTriangle)
  (p : Point)
  (dist_p dist_q dist_r : ℝ)
  (h_positive_p : dist_p > 0)
  (h_positive_q : dist_q > 0)
  (h_positive_r : dist_r > 0)
  (h_distances : ∃ (v1 v2 v3 : Point),
    distance p v1 = dist_p ∧
    distance p v2 = dist_q ∧
    distance p v3 = dist_r ∧
    distance v1 v2 = t.sideLength ∧
    distance v2 v3 = t.sideLength ∧
    distance v3 v1 = t.sideLength) :
  ∃ (t' : EquilateralTriangle) (q : Point),
    t'.sideLength = t.sideLength ∧
    ∃ (v1' v2' v3' : Point),
      distance q v1' = dist_q ∧
      distance q v2' = dist_r ∧
      distance q v3' = dist_p ∧
      distance v1' v2' = t'.sideLength ∧
      distance v2' v3' = t'.sideLength ∧
      distance v3' v1' = t'.sideLength := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_point_rotation_l1195_119556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_negativity_l1195_119515

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1 / x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 + x - b
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x / g b x

theorem fixed_point_and_negativity (a b : ℝ) :
  (∃ P : ℝ × ℝ, P.1 > 0 ∧ 
    f a P.1 = P.2 ∧ 
    g b P.1 = P.2 ∧ 
    (deriv (f a)) P.1 = P.2) →
  (a = 2 ∧ b = 2) ∧
  (∀ x : ℝ, x > 0 → x ≠ 1 → h a b x < 0) :=
by sorry

#check fixed_point_and_negativity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_negativity_l1195_119515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_forms_l1195_119577

-- Define the functions f and g
noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := Real.log (x + m) + n

def g (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem function_forms (m n a b : ℝ) (h_a : a ≠ 0) :
  (∀ x, x > -m → Real.exp (f m n x) = x + m) ∧
  (∀ x, Real.exp (f m n 1) * (x - 1) + Real.exp (f m n 1) = x) ∧
  (g a b 2 = -2) ∧
  (∀ x, x ≠ 2 → g a b x > -2) →
  (∀ x, x > 0 → f m n x = Real.log x) ∧
  (∀ x, g a b x = (1/2) * x^2 - 2 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_forms_l1195_119577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_s_positive_l1195_119584

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^2

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by
  sorry

-- State that there are no y ≤ 0 in the range of s(x)
theorem s_positive :
  ∀ x : ℝ, x ≠ 2 → s x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_s_positive_l1195_119584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sequence_l1195_119505

def sequenceA (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) * a n + 1

theorem composite_sequence (a : ℕ → ℕ) (h : sequenceA a) :
  ∀ k : ℕ, k ≥ 9 → ∃ m : ℕ, m > 1 ∧ m ∣ (a k - 22) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sequence_l1195_119505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_difference_l1195_119587

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_difference (a₁ d : ℝ) (n : ℕ) :
  S a₁ d (n + 2) - S a₁ d n = 36 →
  a₁ = 1 →
  d = 2 →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_difference_l1195_119587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_t_l1195_119506

/-- A function f is a "D-t function" if f(x) ≥ t for all x in the domain D -/
def is_D_t_function (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∀ x, f x ≥ t

/-- The specific function we're considering -/
noncomputable def f (t : ℝ) : ℝ → ℝ := λ x ↦ (x - t) * Real.exp x

theorem largest_integer_t : 
  (∀ t : ℝ, (∀ x : ℝ, f t x ≥ t) → t ≤ -1) ∧ 
  (∃ x : ℝ, f (-1) x < -1 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_t_l1195_119506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l1195_119558

/-- The polar coordinate equation of circle C is ρ^2 + 2√2 ρ sin(θ - π/4) - 4 = 0 -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ^2 + 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi/4) - 4 = 0

/-- The radius of circle C -/
noncomputable def circle_radius : ℝ := Real.sqrt 6

theorem circle_radius_proof :
  ∀ ρ θ : ℝ, circle_equation ρ θ → ρ = circle_radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l1195_119558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l1195_119530

noncomputable def distance_to_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def slope_to_origin (x y : ℝ) : ℝ :=
  y / x

/-- A line passing through a point (x₀, y₀) with slope m has equation y - y₀ = m(x - x₀) -/
def line_equation (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

theorem max_distance_line :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, line_equation 1 2 m x y ↔ x + 2*y = 5) ∧
    (∀ x y : ℝ, line_equation 1 2 m x y → 
      ∀ m' : ℝ, ∀ x' y' : ℝ, line_equation 1 2 m' x' y' → 
        distance_to_origin x y ≥ distance_to_origin x' y') ∧
    m * slope_to_origin 1 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l1195_119530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1195_119543

/-- Given a hyperbola with the specified properties, its eccentricity is 1 + √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t))) →
  F₁ ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t)) →
  F₂ ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t)) →
  P ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t)) →
  Q ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t)) →
  (P.1 - Q.1 = 0) →
  (F₁.2 = 0) →
  ((P.1 - F₂.1) * (Q.1 - F₂.1) + (P.2 - F₂.2) * (Q.2 - F₂.2) = 0) →
  (Real.sqrt (a^2 + b^2) / a = 1 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1195_119543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_equal_triangle_division_l1195_119511

/-- Predicate stating that an n-gon is regular -/
def IsRegularNGon (n : ℕ) : Prop :=
  sorry

/-- Predicate stating that a set of segments forms a valid set of diagonals for an n-gon -/
def IsDiagonalSet (n : ℕ) (d : Set (Fin n × Fin n)) : Prop :=
  sorry

/-- Predicate stating that a set of segments divides an n-gon into equal triangles -/
def DividesIntoEqualTriangles (n : ℕ) (d : Set (Fin n × Fin n)) : Prop :=
  sorry

/-- A regular n-gon can be divided by diagonals into equal triangles if and only if n is even and greater than 3 -/
theorem regular_ngon_equal_triangle_division (n : ℕ) : 
  (n > 3 ∧ ∃ (d : Set (Fin n × Fin n)), 
    IsRegularNGon n ∧ 
    IsDiagonalSet n d ∧ 
    DividesIntoEqualTriangles n d) ↔ 
  (Even n ∧ n > 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_equal_triangle_division_l1195_119511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1195_119572

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 30)
  (sum_abcd : a + b + c + d = 47)
  (sum_abcdef : a + b + c + d + e + f = 65) :
  ∃ (odd_count : ℕ), 
    odd_count ≥ 1 ∧
    odd_count = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) ∧
    ∀ (other_odd_count : ℕ), 
      (other_odd_count = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                         (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                         (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0)) →
      other_odd_count ≥ odd_count :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1195_119572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_intersection_l1195_119502

-- Define IsRegularOctagon as an uninterpreted predicate
axiom IsRegularOctagon : Set (ℝ × ℝ) → Prop

theorem octagon_intersection (a : ℝ) : a > 0 → 
  let A := {p : ℝ × ℝ | |p.1| + |p.2| = a}
  let B := {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}
  (∃ O : Set (ℝ × ℝ), IsRegularOctagon O ∧ O = A ∩ B) →
  a = 2 + Real.sqrt 2 ∨ a = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_intersection_l1195_119502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_35_and_200_l1195_119551

theorem multiples_of_15_between_35_and_200 : 
  ∃ n : ℕ, n = (Finset.filter (λ x ↦ 15 ∣ x ∧ 35 < x ∧ x < 200) (Finset.range 200)).card ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_35_and_200_l1195_119551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1195_119514

/-- Represents a cone with base radius and apex angle -/
structure Cone where
  baseRadius : ℝ
  apexAngle : ℝ

/-- Represents the configuration of cones and sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphereRadius : ℝ

/-- The specific problem configuration -/
noncomputable def problem : ConeSphereProblem :=
  { cone1 := { baseRadius := 32, apexAngle := Real.pi/3 }
  , cone2 := { baseRadius := 48, apexAngle := 2*Real.pi/3 }
  , cone3 := { baseRadius := 48, apexAngle := 2*Real.pi/3 }
  , sphereRadius := 13*(Real.sqrt 3 + 1) }

/-- Theorem stating that the sphere radius is correct for the given configuration -/
theorem sphere_radius_correct (p : ConeSphereProblem) : 
  p.cone1.baseRadius = 32 ∧ 
  p.cone1.apexAngle = Real.pi/3 ∧
  p.cone2.baseRadius = 48 ∧ 
  p.cone2.apexAngle = 2*Real.pi/3 ∧
  p.cone3.baseRadius = 48 ∧ 
  p.cone3.apexAngle = 2*Real.pi/3 →
  p.sphereRadius = 13*(Real.sqrt 3 + 1) :=
by
  sorry

#check sphere_radius_correct problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1195_119514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coloring_value_l1195_119553

/-- A coloring of a 6 × 6 grid -/
def Coloring := Fin 6 → Fin 6 → Bool

/-- The property that a coloring satisfies the problem conditions -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ (r1 r2 r3 : Fin 6) (c1 c2 c3 : Fin 6),
    r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 →
    ∃ (i j : Fin 3), ¬c ((List.get! [r1, r2, r3] i)) ((List.get! [c1, c2, c3] j))

/-- The number of colored cells in a coloring -/
def colored_cells (c : Coloring) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 6)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 6)) (λ j => 
      if c i j then 1 else 0))

/-- The theorem statement -/
theorem max_coloring_value :
  (∃ (k : ℕ) (c : Coloring), k = 4 ∧ colored_cells c = 6 * k ∧ valid_coloring c) ∧
  (∀ (k : ℕ) (c : Coloring), k > 4 → colored_cells c = 6 * k → ¬valid_coloring c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coloring_value_l1195_119553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_birdhouse_price_l1195_119516

/-- The price of a large birdhouse satisfies the given conditions -/
theorem large_birdhouse_price :
  ∀ (large_price : ℚ),
  (large_price * 2 + 16 * 2 + 7 * 3 = 97) →
  large_price = 22 := by
  intro large_price h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_birdhouse_price_l1195_119516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_optimal_speed_l1195_119544

/-- Represents the ship's speed in kilometers per hour -/
noncomputable def speed : ℝ → ℝ := sorry

/-- Represents the fuel cost per hour as a function of speed -/
noncomputable def fuel_cost : ℝ → ℝ := sorry

/-- Represents other costs per hour unrelated to speed -/
def other_costs : ℝ := 96

/-- The total cost per hour as a function of speed -/
noncomputable def total_cost (v : ℝ) : ℝ := fuel_cost v + other_costs

/-- The total cost per kilometer traveled as a function of speed -/
noncomputable def cost_per_km (v : ℝ) : ℝ := total_cost v / v

theorem ship_optimal_speed :
  /- Fuel cost is directly proportional to the cube of speed -/
  (∃ k : ℝ, ∀ v : ℝ, fuel_cost v = k * v^3) →
  /- When speed is 10 km/h, fuel cost per hour is 6 yuan -/
  (fuel_cost 10 = 6) →
  /- The speed that minimizes the total cost per kilometer traveled is 20 km/h -/
  (∃ v : ℝ, v = 20 ∧ ∀ u : ℝ, u > 0 → cost_per_km v ≤ cost_per_km u) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_optimal_speed_l1195_119544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_both_ways_time_l1195_119561

/-- Represents the time in minutes for a student's journey to and from school -/
structure SchoolJourney where
  walk_and_bus : ℕ  -- Time when walking to school and taking bus back
  bus_both_ways : ℕ  -- Time when taking bus both ways

/-- Calculates the time needed when walking both ways -/
def time_walking_both_ways (j : SchoolJourney) : ℕ :=
  2 * (j.walk_and_bus - j.bus_both_ways / 2)

theorem walking_both_ways_time (j : SchoolJourney) 
  (h1 : j.walk_and_bus = 90)
  (h2 : j.bus_both_ways = 30) :
  time_walking_both_ways j = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_both_ways_time_l1195_119561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l1195_119593

open Real Topology BigOperators

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (n * (a n)^2)

theorem series_convergence 
  (a : ℕ → ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha_conv : Summable a) :
  Summable (λ n => n / (∑ i in Finset.range n, b a (i + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l1195_119593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relationship_l1195_119542

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square (k : ℝ) (y : ℝ) : ℝ := k / (y ^ 2)

/-- Theorem stating the relationship between x and y -/
theorem inverse_square_relationship (k : ℝ) :
  (inverse_square k 3 = 1) →
  (inverse_square k 2 = 2.25) := by
  intro h
  -- The proof goes here
  sorry

#check inverse_square_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relationship_l1195_119542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_integer_solution_l1195_119559

noncomputable def nested_sqrt : ℕ → ℝ → ℝ
  | 0, x => 0
  | n + 1, x => Real.sqrt (x + nested_sqrt n x)

theorem nested_sqrt_integer_solution :
  ∀ x y : ℤ, nested_sqrt 1992 (x : ℝ) = (y : ℝ) → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_integer_solution_l1195_119559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1195_119576

-- Define the function f with domain [-1, 3]
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 3

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1195_119576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_height_l1195_119567

/-- The volume of a cone inscribed in a sphere of radius R with height h -/
noncomputable def volume_cone (R : ℝ) (h : ℝ) : ℝ := 
  (Real.pi / 3) * h * (2 * R * h - h^2)

/-- The height of a cone inscribed in a sphere that maximizes the cone's volume -/
theorem max_volume_cone_height (R : ℝ) (h : ℝ) : 
  R > 0 → -- The sphere has a positive radius
  (∀ h' : ℝ, volume_cone R h' ≤ volume_cone R h) → -- h maximizes the volume
  h = 4/3 * R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_height_l1195_119567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1195_119537

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area formula for the triangle -/
noncomputable def areaFormula (t : Triangle) : ℝ :=
  (Real.sqrt 3 / 6) * t.b * (t.b + t.c - t.a * Real.cos t.C)

theorem triangle_properties (t : Triangle) 
  (h1 : areaFormula t = (1/2) * t.a * t.b * Real.sin t.C)
  (h2 : t.b = 1)
  (h3 : t.c = 3) :
  t.A = π/3 ∧ Real.cos (2 * t.C - π/6) = -4 * Real.sqrt 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1195_119537
