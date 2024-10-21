import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_room_median_l1174_117419

theorem hotel_room_median : 
  let rooms : List ℕ := List.filter (λ x => x ≠ 15 ∧ x ≠ 20) (List.range 32)
  rooms.length = 29 ∧ 
  List.Sorted (· ≤ ·) rooms ∧
  rooms.get! 14 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_room_median_l1174_117419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_conditions_l1174_117406

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def P (x y : ℤ) : ℤ := x * (2 - ((y - x) * y - x^2)^2)

theorem polynomial_conditions :
  (∀ x y : ℤ, P x y ∈ Set.univ) ∧
  (∀ n : ℕ, ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ P x y > n) ∧
  (∀ x y : ℤ, x > 0 → y > 0 → P x y > 0 → P x y ∈ Set.univ → ∃ n : ℕ, P x y = fibonacci n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_conditions_l1174_117406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_when_selling_price_equals_cost_price_l1174_117440

/-- Calculates the profit percentage per pen when the selling price of a smaller quantity
    equals the cost price of a larger quantity. -/
theorem profit_percentage_when_selling_price_equals_cost_price
  (cost_quantity : ℕ) 
  (sell_quantity : ℕ) 
  (h1 : cost_quantity > sell_quantity) 
  (h2 : cost_quantity > 0) 
  (h3 : sell_quantity > 0) :
  let profit_per_pen := (1 / sell_quantity : ℚ) - (1 / cost_quantity : ℚ)
  let cost_per_pen := (1 / cost_quantity : ℚ)
  (profit_per_pen / cost_per_pen) * 100 = 50 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_when_selling_price_equals_cost_price_l1174_117440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1174_117430

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the circle
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem line_circle_intersection (m : ℝ) :
  (∃ x y, line_l x y ∧ line_l 1 1) →  -- l passes through (1,1)
  (∀ x y, line_l x y ↔ 2*x + 4*y + 9 = 0) →  -- l is parallel to 2x+4y+9=0
  (∃ x1 y1 x2 y2, line_l x1 y1 ∧ line_l x2 y2 ∧ 
    circle_eq x1 y1 m ∧ circle_eq x2 y2 m ∧ 
    perpendicular x1 y1 x2 y2) →  -- l intersects the circle at P and Q, OP ⟂ OQ
  m = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1174_117430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1174_117445

-- Define the given constants
def train_speed_kmh : ℚ := 144
def time_to_pass_pole : ℚ := 8
def stationary_train_length : ℚ := 400

-- Define the function to convert km/h to m/s
def kmh_to_ms (speed : ℚ) : ℚ := speed * 1000 / 3600

-- Define the function to calculate the length of the moving train
def moving_train_length (speed : ℚ) (time : ℚ) : ℚ := speed * time

-- Define the function to calculate the total distance to be covered
def total_distance (moving_length : ℚ) (stationary_length : ℚ) : ℚ := 
  moving_length + stationary_length

-- Define the function to calculate the time to cross
def time_to_cross (distance : ℚ) (speed : ℚ) : ℚ := distance / speed

-- State the theorem
theorem train_crossing_time : 
  time_to_cross 
    (total_distance 
      (moving_train_length 
        (kmh_to_ms train_speed_kmh) 
        time_to_pass_pole) 
      stationary_train_length) 
    (kmh_to_ms train_speed_kmh) = 18 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1174_117445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_a_formula_l1174_117410

def x : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => x n / (2 - x n)

def a (n : ℕ) : ℚ := 1 / x n

theorem x_bounds (n : ℕ) : 0 < x n ∧ x n < 1 := by sorry

theorem a_formula (n : ℕ) : a n = 2^n + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_a_formula_l1174_117410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_case_l1174_117476

theorem tan_sum_special_case :
  let α : Real := 5 * π / 180
  let β : Real := 40 * π / 180
  (Real.tan α + Real.tan β + Real.tan α * Real.tan β = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_case_l1174_117476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_k_l1174_117429

-- Define the hyperbola equation
def hyperbola_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + k * y^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity_k (k : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, hyperbola_equation k x y ↔ (x/a)^2 - (y/b)^2 = 1) ∧
    eccentricity a b = 2) →
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_k_l1174_117429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_row_unique_l1174_117403

/-- Represents the type of dot between two numbers -/
inductive Dot
  | Black
  | White
  | None

/-- Represents a 3x3 grid with dots between numbers -/
structure Grid where
  values : Fin 3 → Fin 3 → Nat
  horizontal_dots : Fin 3 → Fin 2 → Dot
  vertical_dots : Fin 2 → Fin 3 → Dot

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g.values i j ∈ Finset.range 6) ∧
  (∀ i, Function.Injective (g.values i)) ∧
  (∀ j, Function.Injective (λ i => g.values i j)) ∧
  (∀ i j, g.horizontal_dots i j = Dot.Black → g.values i (Fin.succ j) = 2 * g.values i j) ∧
  (∀ i j, g.horizontal_dots i j = Dot.White → g.values i (Fin.succ j) = g.values i j + 1 ∨ g.values i j = g.values i (Fin.succ j) + 1) ∧
  (∀ i j, g.vertical_dots i j = Dot.Black → g.values (Fin.succ i) j = 2 * g.values i j) ∧
  (∀ i j, g.vertical_dots i j = Dot.White → g.values (Fin.succ i) j = g.values i j + 1 ∨ g.values i j = g.values (Fin.succ i) j + 1)

theorem fourth_row_unique (g : Grid) (h : is_valid_grid g) :
  g.values 2 0 = 2 ∧ g.values 2 1 = 1 ∧ g.values 2 2 = 4 ∧ g.values 1 2 = 3 ∧ g.values 0 2 = 6 :=
by sorry

#check fourth_row_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_row_unique_l1174_117403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_not_lattice_point_l1174_117413

/-- A point in the 2D lattice. -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points. -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- The circumcenter of a triangle. -/
noncomputable def circumcenter (t : LatticeTriangle) : ℝ × ℝ := sorry

/-- Predicate to check if a point is a lattice point. -/
def isLatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

/-- Predicate to check if a triangle is similar to another triangle. -/
def isSimilar (t1 t2 : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a triangle is smaller than another triangle. -/
def isSmaller (t1 t2 : LatticeTriangle) : Prop := sorry

/-- Main theorem -/
theorem circumcenter_not_lattice_point (t : LatticeTriangle) :
  (∀ t' : LatticeTriangle, isSimilar t t' → ¬(isSmaller t' t)) →
  ¬(isLatticePoint (circumcenter t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_not_lattice_point_l1174_117413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_symmetric_holes_l1174_117458

/-- Represents a rectangular sheet of paper -/
structure Paper where
  width : ℚ
  height : ℚ
  holes : List (ℚ × ℚ)

/-- Represents the folding operations -/
inductive FoldOperation
  | BottomToTop
  | RightToLeft
  | TopToBottom

/-- Applies a single fold operation to the paper -/
def applyFold (p : Paper) (op : FoldOperation) : Paper :=
  match op with
  | FoldOperation.BottomToTop => { p with height := p.height / 2 }
  | FoldOperation.RightToLeft => { p with width := p.width / 2 }
  | FoldOperation.TopToBottom => { p with height := p.height / 2 }

/-- Applies all three fold operations in sequence -/
def foldPaper (p : Paper) : Paper :=
  applyFold (applyFold (applyFold p FoldOperation.BottomToTop) FoldOperation.RightToLeft) FoldOperation.TopToBottom

/-- Punches a hole in the folded paper -/
def punchHole (p : Paper) (x y : ℚ) : Paper :=
  { p with holes := (x, y) :: p.holes }

/-- Unfolds the paper and calculates the positions of all holes -/
def unfoldPaper (p : Paper) : Paper :=
  sorry

/-- Theorem stating that folding and punching results in 8 symmetrically distributed holes -/
theorem eight_symmetric_holes (p : Paper) (x y : ℚ) :
  (unfoldPaper (punchHole (foldPaper p) x y)).holes.length = 8 ∧
  -- Additional conditions for symmetry would be defined here
  True := by
  sorry

#check eight_symmetric_holes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_symmetric_holes_l1174_117458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_at_two_seconds_l1174_117442

/-- The height of an object thrown upward, considering air resistance -/
def height' (v0 k t : ℝ) : ℝ :=
  -16 * t^2 + v0 * t * (1 - k * t) + 140

/-- The theorem stating the height of the object at t=2 seconds -/
theorem height_at_two_seconds (v0 k : ℝ) 
  (h_v0 : v0 = 50) 
  (h_k : k = 0.05) : 
  height' v0 k 2 = 166 := by
  sorry

#eval height' 50 0.05 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_at_two_seconds_l1174_117442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1174_117405

/-- Sum of a finite geometric series -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 3
  let n : ℕ := 11
  geometric_sum a r n = 88573 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1174_117405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1174_117417

/-- For any triangle ABC with side lengths a, b, c and angles A, B, C opposite to these sides respectively,
    the equality a(b^2 + c^2) cos A + b(c^2 + a^2) cos B + c(a^2 + b^2) cos C = 3abc holds. -/
theorem triangle_cosine_sum (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- side lengths are positive
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = π →  -- sum of angles in a triangle is π
  a * Real.sin B = b * Real.sin A →  -- law of sines
  b * Real.sin C = c * Real.sin B →  -- law of sines
  c * Real.sin A = a * Real.sin C →  -- law of sines
  a * (b^2 + c^2) * Real.cos A + b * (c^2 + a^2) * Real.cos B + c * (a^2 + b^2) * Real.cos C = 3 * a * b * c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1174_117417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_equals_diameter_l1174_117428

theorem longest_chord_equals_diameter (r : ℝ) (h : r = 7) : 
  2 * r = 14 :=
by
  rw [h]
  ring

#check longest_chord_equals_diameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_equals_diameter_l1174_117428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_square_representations_l1174_117439

/-- A function that checks if a number can be expressed as the sum of 4 squares in 4 different ways -/
def hasFourSquareRepresentations (n : ℕ) : Prop :=
  ∃ (a b c d e f g h i j k l m n o p : ℕ),
    n = a^2 + b^2 + c^2 + d^2
    ∧ n = e^2 + f^2 + g^2 + h^2
    ∧ n = i^2 + j^2 + k^2 + l^2
    ∧ n = m^2 + n^2 + o^2 + p^2
    ∧ Finset.card {(a, b, c, d), (e, f, g, h), (i, j, k, l), (m, n, o, p)} = 4

/-- The theorem stating that 635318657 is the smallest number with four 4-square representations -/
theorem smallest_four_square_representations :
  (hasFourSquareRepresentations 635318657) ∧
  (∀ m : ℕ, m < 635318657 → ¬(hasFourSquareRepresentations m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_square_representations_l1174_117439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1174_117493

theorem inequality_solution (x : ℝ) :
  x ≠ 0 → x ≠ 1 → x ≠ -1/3 →
  ((x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 2 ↔
   x ∈ Set.Iic (-1/3) ∪
       Set.Ioo (-1/3) 0 ∪
       Set.Ioo 0 1 ∪
       Set.Ioc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1174_117493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_theorem_l1174_117489

/-- A parabola with focus F and directrix intersecting the x-axis at K -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- A point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * C.p * x

/-- The focus of the parabola -/
noncomputable def focus (C : Parabola) : ℝ × ℝ := (C.p / 2, 0)

/-- The directrix-x-axis intersection point -/
noncomputable def directrix_x_intersection (C : Parabola) : ℝ × ℝ := (-C.p / 2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_angle_theorem (C : Parabola) (M : PointOnParabola C) :
  let F := focus C
  let K := directrix_x_intersection C
  distance (M.x, M.y) F = C.p →
  angle (M.x, M.y) K F = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_theorem_l1174_117489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_205_l1174_117490

/-- The length of a bridge that can be crossed by a train with given parameters. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating that the bridge length is 205 meters given the specified conditions. -/
theorem bridge_length_is_205 :
  bridge_length 170 45 30 = 205 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num

-- We can't use #eval with noncomputable definitions, so we'll use #check instead
#check bridge_length 170 45 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_205_l1174_117490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1174_117496

-- Define the function g
noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := x^2 / (D*x^2 + E*x + F)

-- State the theorem
theorem sum_of_coefficients (D E F : ℤ) : 
  -- Vertical asymptotes at x = -3 and x = 4
  (∀ x : ℝ, D*x^2 + E*x + F = D*(x + 3)*(x - 4)) →
  -- Horizontal asymptote below 1 but above 0.5
  (0.5 < (1 : ℝ) / ↑D ∧ (1 : ℝ) / ↑D < 1) →
  -- g(x) > 0.5 for all x > 5
  (∀ x : ℝ, x > 5 → g D E F x > 0.5) →
  D + E + F = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1174_117496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_specific_case_l1174_117495

noncomputable def hyperbola_eccentricity (vertex_to_asymptote : ℝ) (focus_to_asymptote : ℝ) : ℝ :=
  (vertex_to_asymptote + focus_to_asymptote) / vertex_to_asymptote

theorem hyperbola_eccentricity_specific_case :
  hyperbola_eccentricity 2 6 = 4 :=
by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- Simplify the arithmetic
  simp [add_div]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_specific_case_l1174_117495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1174_117473

theorem inequality_solution (x : ℝ) : 
  (x^(1/4 : ℝ) + 3 / (x^(1/4 : ℝ) + 2) ≥ 0) ↔ (x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1174_117473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1174_117449

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := m * Real.sqrt (Real.log x - 1/4) + 2*x + (1/2)*n

-- State the theorem
theorem min_distance_to_origin (m n : ℝ) :
  (∃ a ∈ Set.Icc 2 4, f m n a = 0) →
  Real.sqrt (m^2 + n^2) ≥ 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1174_117449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l1174_117455

noncomputable def f (x : ℝ) : ℝ := 
  (1 + Real.cos (Real.pi * x)) / (Real.tan (Real.pi * x))^2

theorem limit_of_f_at_one :
  Filter.Tendsto f (nhds 1) (nhds (1/2)) := by
  sorry

#check limit_of_f_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l1174_117455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_of_circles_l1174_117462

/-- Function to reflect a point across a line -/
noncomputable def reflect_line (line : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Given two circles in a 2D plane, this theorem states that the line of symmetry
    between them has a specific equation. -/
theorem symmetry_line_of_circles :
  let circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}
  let circle2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + 4*p.2 - 1 = 0}
  let symmetry_line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  (∀ p, p ∈ circle1 ↔ reflect_line symmetry_line p ∈ circle2) →
  symmetry_line = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_of_circles_l1174_117462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_two_l1174_117425

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given a real number a, define z as (1 + ai) / (2 - i). -/
noncomputable def z (a : ℝ) : ℂ :=
  (1 + a * Complex.I) / (2 - Complex.I)

theorem pure_imaginary_implies_a_eq_two (a : ℝ) :
  isPureImaginary (z a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_two_l1174_117425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1174_117461

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

def sum_arithmetic_seq (a : ℕ → ℤ) (start finish step : ℕ) : ℤ :=
  Finset.sum (Finset.range ((finish - start) / step + 1)) (λ i => a (start + i * step))

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : arithmetic_sequence a (-2))
  (h2 : sum_arithmetic_seq a 1 97 3 = 50) :
  sum_arithmetic_seq a 3 99 3 = -82 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1174_117461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_of_milk_powder_inspection_l1174_117471

/-- Given a population of 100 bags of milk powder, if 5 bags are drawn for inspection,
    then the sample size is 5. -/
theorem sample_size_of_milk_powder_inspection :
  ∀ (population : ℕ) (drawn : ℕ),
  population = 100 →
  drawn = 5 →
  drawn = 5 :=
by
  intro population drawn h1 h2
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_of_milk_powder_inspection_l1174_117471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_circle_coverage_l1174_117486

-- Define a convex polygon
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop :=
  -- Add appropriate conditions for convexity
  sorry

-- Define a circle passing through three points
def CircleThroughPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  -- Define the set of points that form the circle through A, B, and C
  sorry

-- Define consecutive vertices in a polygon
def ConsecutiveVertices (P : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  -- Define what it means for A, B, C to be consecutive vertices in P
  sorry

-- Main theorem
theorem convex_polygon_circle_coverage (P : Set (ℝ × ℝ)) 
  (h : ConvexPolygon P) :
  ∃ (A B C : ℝ × ℝ), 
    A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ 
    ConsecutiveVertices P A B C ∧
    ∀ (x : ℝ × ℝ), x ∈ P → x ∈ CircleThroughPoints A B C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_circle_coverage_l1174_117486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l1174_117470

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define a line with equal intercepts
def equal_intercepts (a b c : ℝ) : Prop := ∃ k, a * k + c = 0 ∧ b * k + c = 0

-- Define tangency condition
def is_tangent (a b c : ℝ) : Prop := ∃ x y, circle_eq x y ∧ a * x + b * y + c = 0 ∧
  ∀ x' y', circle_eq x' y' → a * x' + b * y' + c ≥ 0

-- Theorem statement
theorem tangent_lines_with_equal_intercepts :
  (∀ a b c : ℝ, is_tangent a b c ∧ equal_intercepts a b c →
    (a = 1 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4)) ∧
  is_tangent 1 (-1) 0 ∧ equal_intercepts 1 (-1) 0 ∧
  is_tangent 1 1 (-4) ∧ equal_intercepts 1 1 (-4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l1174_117470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_d_finite_iff_coprime_10_l1174_117437

/-- Definition of a subdivisor -/
def is_subdivisor (d n : ℕ) : Prop :=
  ∃ (r l : ℕ), d ∣ (n % 10^(Nat.log 10 n - r - l)) / 10^l ∧ r + l < Nat.log 10 n

/-- Definition of the set A_d -/
def A (d : ℕ) : Set ℕ := {n : ℕ | ¬ is_subdivisor d n}

/-- The main theorem -/
theorem A_d_finite_iff_coprime_10 (d : ℕ) :
  Set.Finite (A d) ↔ Nat.Coprime d 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_d_finite_iff_coprime_10_l1174_117437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_l1174_117450

-- Define the function representing the curve
noncomputable def f (x : ℝ) := Real.cos x

-- Define the integral bounds
noncomputable def a : ℝ := -Real.pi/2
noncomputable def b : ℝ := Real.pi

-- State the theorem
theorem area_enclosed_by_cosine : 
  (∫ (x : ℝ) in a..b, abs (f x)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_l1174_117450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1174_117401

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1174_117401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_chart_ratio_is_one_fourth_l1174_117415

/-- Represents the work schedule of a programmer during a week. -/
structure ProgrammerSchedule where
  total_time : ℚ
  coding_fraction : ℚ
  debugging_time : ℚ

/-- Calculates the ratio of flow chart preparation time to total time. -/
def flow_chart_ratio (schedule : ProgrammerSchedule) : ℚ :=
  let coding_time := schedule.coding_fraction * schedule.total_time
  let flow_chart_time := schedule.total_time - coding_time - schedule.debugging_time
  flow_chart_time / schedule.total_time

/-- Theorem stating that the ratio of flow chart preparation time to total time is 1:4. -/
theorem flow_chart_ratio_is_one_fourth
    (schedule : ProgrammerSchedule)
    (h1 : schedule.total_time = 48)
    (h2 : schedule.coding_fraction = 3/8)
    (h3 : schedule.debugging_time = 18) :
  flow_chart_ratio schedule = 1/4 := by
  sorry

#eval flow_chart_ratio { total_time := 48, coding_fraction := 3/8, debugging_time := 18 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_chart_ratio_is_one_fourth_l1174_117415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1174_117446

def game_board : List ℕ := [25, 36]

def valid_move (board : List ℕ) (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ board ∧ b ∈ board ∧ a > b ∧ n = a - b ∧ n ∉ board

def game_over (board : List ℕ) : Prop :=
  ∀ (n : ℕ), ¬(valid_move board n)

inductive player_wins : ℕ → List ℕ → Prop
  | game_over_win (player : ℕ) (board : List ℕ) : 
      game_over board → player = 2 → player_wins player board
  | make_move (player : ℕ) (board : List ℕ) (n : ℕ) : 
      valid_move board n → player_wins (3 - player) (n :: board) → player_wins player board

theorem second_player_wins : player_wins 2 game_board := by
  sorry

#check second_player_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1174_117446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_mouse_food_theorem_l1174_117422

/-- The number of days that food for a given number of elephants can feed a specific number of mice -/
def elephant_mouse_food_days (elephant_count mouse_count : ℕ) : ℕ → ℕ := sorry

/-- Given that food for 12 elephants for 1 day feeds 1000 mice for 600 days -/
axiom base_case : elephant_mouse_food_days 12 1000 1 = 600

/-- The food consumption is proportional to the number of animals and days -/
axiom proportionality (e m d t : ℕ) : 
  elephant_mouse_food_days e m d * t = elephant_mouse_food_days (e * t) m d

theorem elephant_mouse_food_theorem (t : ℕ) : 
  elephant_mouse_food_days t 100 1 = 500 * t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_mouse_food_theorem_l1174_117422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_with_cap_equilibrium_l1174_117487

/-- A solid consisting of a circular cylinder with a hemispherical cap -/
structure CylinderWithCap where
  r : ℝ  -- radius
  h : ℝ  -- height of cylinder
  (r_pos : r > 0)
  (h_pos : h > 0)

/-- The equilibrium constant for a CylinderWithCap -/
noncomputable def equilibrium_constant : ℝ := Real.sqrt 2

/-- Helper function to calculate the height of the center of mass when tilted by angle θ -/
noncomputable def center_of_mass_height (s : CylinderWithCap) (θ : ℝ) : ℝ :=
  sorry

/-- Theorem stating the equilibrium condition for a CylinderWithCap -/
theorem cylinder_with_cap_equilibrium (s : CylinderWithCap) :
  s.r / s.h = equilibrium_constant ↔ 
  (∀ θ : ℝ, center_of_mass_height s θ = center_of_mass_height s 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_with_cap_equilibrium_l1174_117487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_of_parabola_l1174_117452

/-- Given a parabola with equation x² = -1/4 * y, prove that its latus rectum has equation y = 1/16 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  (x^2 = -(1/4) * y) → (y = 1/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_of_parabola_l1174_117452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_card_value_l1174_117434

def pick_and_replace (a b : ℕ) : ℕ := a + b + a * b

def initial_cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem final_card_value :
  ∃ (operations : List (ℕ × ℕ)),
    operations.length = 9 ∧
    (operations.foldl
      (λ (cards : List ℕ) (pair : ℕ × ℕ) =>
        let (a, b) := pair
        let new_card := pick_and_replace a b
        new_card :: (cards.filter (λ x => x ≠ a ∧ x ≠ b)))
      initial_cards).head? = some (Nat.factorial 11 - 1) := by
  sorry

#eval Nat.factorial 11 - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_card_value_l1174_117434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_effect_l1174_117424

noncomputable def house_sale : ℝ := 10000
noncomputable def store_sale : ℝ := 15000
noncomputable def car_sale : ℝ := 8000

noncomputable def house_loss_percent : ℝ := 25
noncomputable def store_gain_percent : ℝ := 25
noncomputable def car_loss_percent : ℝ := 20

noncomputable def total_sale : ℝ := house_sale + store_sale + car_sale

noncomputable def house_cost : ℝ := house_sale / (1 - house_loss_percent / 100)
noncomputable def store_cost : ℝ := store_sale / (1 + store_gain_percent / 100)
noncomputable def car_cost : ℝ := car_sale / (1 - car_loss_percent / 100)

noncomputable def total_cost : ℝ := house_cost + store_cost + car_cost

theorem transaction_effect :
  ∃ (ε : ℝ), abs (total_cost - total_sale - 2333.33) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_effect_l1174_117424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_max_l1174_117411

open Real

theorem tan_two_alpha_max (α β : ℝ) : 
  α ∈ Set.Ioo 0 (π/2) → 
  β ∈ Set.Ioo 0 (π/2) → 
  tan (α + β) = 2 * tan β → 
  ∃ (α_max : ℝ), α_max ∈ Set.Ioo 0 (π/2) ∧ 
    (∀ α' ∈ Set.Ioo 0 (π/2), tan (2*α') ≤ tan (2*α_max)) ∧
    tan (2*α_max) = 4*sqrt 2/7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_max_l1174_117411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1174_117407

/-- Polynomial type with real coefficients -/
def MyPolynomial (α : Type) := List α

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : MyPolynomial ℝ) (x : ℝ) : ℝ :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Divide a polynomial by (x - a) -/
def polyDivide (p : MyPolynomial ℝ) (a : ℝ) : MyPolynomial ℝ × ℝ :=
  sorry

theorem polynomial_remainder_theorem (p : MyPolynomial ℝ) (a : ℝ) :
  let (_, r) := polyDivide p a
  r = evalPoly p a := by
  sorry

#check polynomial_remainder_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1174_117407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_sold_in_week_l1174_117464

/-- Calculates the number of goldfish sold in a week given the specified conditions --/
def goldfish_sold (buy_price sell_price tank_cost daily_expense : ℚ) 
  (days : ℕ) (short_percentage : ℚ) : ℕ :=
  let profit_per_fish := sell_price - buy_price
  let total_expense := daily_expense * (days : ℚ)
  let available_amount := tank_cost * (1 - short_percentage)
  let total_sales := available_amount + total_expense
  (total_sales / profit_per_fish).floor.toNat

/-- Theorem stating that under the given conditions, 125 goldfish were sold in a week --/
theorem goldfish_sold_in_week : 
  goldfish_sold (1/2) (3/2) 200 5 7 (11/20) = 125 := by
  sorry

#eval goldfish_sold (1/2) (3/2) 200 5 7 (11/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_sold_in_week_l1174_117464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1174_117423

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

-- Define the derivative of f with respect to x
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 * a * x + 4 * (a - 3)

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x : ℝ, x < 3 → f_deriv a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3/4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1174_117423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marvin_ratio_l1174_117494

/-- Represents the number of math problems Marvin practiced today -/
def marvin_today : ℕ := sorry

/-- Represents the number of math problems Marvin practiced yesterday -/
def marvin_yesterday : ℕ := sorry

/-- The total number of math problems practiced by Marvin and Arvin -/
def total_problems : ℕ := sorry

/-- Arvin practices twice as many problems as Marvin each day -/
def arvin (day : ℕ) : ℕ := 2 * day

/-- The total number of problems practiced is 480 -/
axiom total_eq_480 : total_problems = 480

/-- Marvin practiced 40 problems yesterday -/
axiom marvin_yesterday_eq_40 : marvin_yesterday = 40

/-- The total problems is the sum of both Marvin's and Arvin's problems for both days -/
axiom total_sum : total_problems = marvin_today + marvin_yesterday + 
                                   (arvin marvin_today) + 
                                   (arvin marvin_yesterday)

theorem marvin_ratio : 
  (marvin_today : ℚ) / marvin_yesterday = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marvin_ratio_l1174_117494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_tshirt_purchase_l1174_117404

def tshirt_cost : ℚ := 9.95
def total_spent : ℚ := 199

theorem carrie_tshirt_purchase :
  (total_spent / tshirt_cost).floor = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_tshirt_purchase_l1174_117404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1174_117447

noncomputable def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

noncomputable def Variance (x y z : ℝ) : ℝ :=
  (x^2 + y^2 + z^2) / 3 - ((x + y + z) / 3)^2

theorem right_triangle_properties :
  ∀ a b : ℝ,
  RightTriangle a b 3 →
  (Variance a b 3 < 5) ∧
  (∃ (min_std_dev : ℝ),
    min_std_dev = Real.sqrt 2 - 1 ∧
    (∀ x y : ℝ, RightTriangle x y 3 → Real.sqrt (Variance x y 3) ≥ min_std_dev)) ∧
  (∀ x y : ℝ, RightTriangle x y 3 ∧ Real.sqrt (Variance x y 3) = Real.sqrt 2 - 1 →
    x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1174_117447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_purchase_cost_l1174_117448

/-- Calculates the total cost in USD for items purchased in yen, given the exchange rate -/
noncomputable def total_cost_usd (book_price_yen : ℝ) (souvenir_price_yen : ℝ) (exchange_rate : ℝ) : ℝ :=
  (book_price_yen + souvenir_price_yen) / exchange_rate

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem alice_purchase_cost :
  round_to_hundredth (total_cost_usd 500 300 110) = 7.27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_purchase_cost_l1174_117448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_quadrilateral_areas_l1174_117463

/-- A point on the perimeter of a square --/
structure PerimeterPoint where
  side : Fin 4
  position : Fin 4

/-- A quadrilateral formed by connecting four points on the perimeter of a square --/
structure PerimeterQuadrilateral where
  points : Fin 4 → PerimeterPoint

/-- The set of possible areas for quadrilaterals formed by connecting points on the perimeter of a square --/
def possibleAreas : Set ℝ :=
  {6, 7, 7.5, 8, 8.5, 9, 10}

/-- The side length of the square --/
def sideLength : ℝ := 4

/-- Calculate the area of a PerimeterQuadrilateral --/
noncomputable def area (q : PerimeterQuadrilateral) : ℝ := sorry

/-- Theorem stating that the area of any PerimeterQuadrilateral is in the set of possible areas --/
theorem perimeter_quadrilateral_areas :
  ∀ q : PerimeterQuadrilateral,
  area q ∈ possibleAreas := by
  sorry

#check perimeter_quadrilateral_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_quadrilateral_areas_l1174_117463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_differential_equation_unique_solution_l1174_117414

/-- The differential equation and its solution -/
theorem differential_equation_solution (t : ℝ) :
  let x : ℝ → ℝ := λ t => 1 - 2 * Real.exp t + Real.exp (4 * t)
  let x' : ℝ → ℝ := λ t => -2 * Real.exp t + 4 * Real.exp (4 * t)
  let x'' : ℝ → ℝ := λ t => -2 * Real.exp t + 16 * Real.exp (4 * t)
  (x'' t - 5 * x' t + 4 * x t = 4) ∧
  (x 0 = 0) ∧
  (x' 0 = 2) := by
  sorry

/-- The uniqueness of the solution -/
theorem differential_equation_unique_solution (y : ℝ → ℝ) :
  (∀ t, (deriv (deriv y)) t - 5 * (deriv y) t + 4 * y t = 4) →
  (y 0 = 0) →
  ((deriv y) 0 = 2) →
  (∀ t, y t = 1 - 2 * Real.exp t + Real.exp (4 * t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_differential_equation_unique_solution_l1174_117414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_square_lower_bound_l1174_117468

open MeasureTheory Real

/-- For any integrable function f: [0,1] → ℝ satisfying the given conditions,
    the integral of f² over [0,1] is greater than or equal to 4. -/
theorem integral_square_lower_bound
  (f : ℝ → ℝ)
  (h_integrable : IntervalIntegrable f volume 0 1)
  (h_int_f : ∫ x in Set.Icc 0 1, f x = 1)
  (h_int_xf : ∫ x in Set.Icc 0 1, x * f x = 1) :
  ∫ x in Set.Icc 0 1, (f x)^2 ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_square_lower_bound_l1174_117468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_wang_running_time_l1174_117412

/-- Represents the distance between City A and City B -/
noncomputable def distance : ℝ := 1

/-- Xiao Wang's cycling speed -/
noncomputable def cycling_speed : ℝ := 5 / 2

/-- Xiao Wang's running speed -/
noncomputable def running_speed : ℝ := cycling_speed * (1 - 1 / 2)

/-- Xiao Wang's walking speed -/
noncomputable def walking_speed : ℝ := running_speed * (1 - 1 / 2)

/-- The total time taken for cycling from A to B and walking back -/
noncomputable def total_time : ℝ := 2

theorem xiao_wang_running_time :
  distance / running_speed * 60 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_wang_running_time_l1174_117412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfEdgesApprox135_l1174_117456

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ  -- Length of the base side
  height : ℝ     -- Height of the pyramid
  base_side_positive : base_side > 0
  height_positive : height > 0

/-- Calculates the sum of the lengths of all edges of a right pyramid -/
noncomputable def sumOfEdges (p : RightPyramid) : ℝ :=
  let base_perimeter := 4 * p.base_side
  let diagonal := p.base_side * Real.sqrt 2
  let slant_height := Real.sqrt ((diagonal / 2) ^ 2 + p.height ^ 2)
  base_perimeter + 4 * slant_height

/-- Theorem stating that for a right pyramid with base side 15 cm and height 15 cm,
    the sum of its edge lengths is approximately 135 cm -/
theorem sumOfEdgesApprox135 :
    let p : RightPyramid := ⟨15, 15, by norm_num, by norm_num⟩
    abs (sumOfEdges p - 135) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfEdgesApprox135_l1174_117456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_fertilization_maintain_chromosomes_l1174_117436

/-- Represents the process of meiosis -/
inductive Meiosis : Type where
  | process : Meiosis

/-- Represents the process of fertilization -/
inductive Fertilization : Type where
  | process : Fertilization

/-- Represents an organism that reproduces sexually -/
structure SexualOrganism where
  meiosis : Meiosis
  fertilization : Fertilization

/-- Represents a somatic cell -/
structure SomaticCell where
  chromosomeCount : ℕ

/-- The combination of meiosis and fertilization maintains the chromosome count -/
axiom maintain_chromosome_count (o : SexualOrganism) (parent offspring : SomaticCell) :
  parent.chromosomeCount = offspring.chromosomeCount

/-- Theorem: For sexually reproducing organisms, meiosis and fertilization maintain
    the chromosome number in somatic cells across generations -/
theorem meiosis_fertilization_maintain_chromosomes (o : SexualOrganism) 
  (parent offspring : SomaticCell) :
  parent.chromosomeCount = offspring.chromosomeCount := by
  exact maintain_chromosome_count o parent offspring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_fertilization_maintain_chromosomes_l1174_117436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_base_diameter_l1174_117426

/-- A truncated cone with the given properties -/
structure TruncatedCone where
  R : ℝ  -- radius of the first base
  r : ℝ  -- radius of the second base
  h : ℝ  -- height of the truncated cone

/-- Volume of a truncated cone -/
noncomputable def volume (tc : TruncatedCone) : ℝ :=
  (Real.pi * tc.h / 3) * (tc.R^2 + tc.R * tc.r + tc.r^2)

/-- Theorem about the diameter of the other base in a truncated cone -/
theorem other_base_diameter
  (tc : TruncatedCone)
  (h_R : tc.R = 50)  -- initial radius of the first base (diameter 100 mm)
  (h_volume_increase : volume { R := 1.21 * tc.R, r := tc.r, h := tc.h } = 1.21 * volume tc)
  : 2 * tc.r = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_base_diameter_l1174_117426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_32_l1174_117477

def my_sequence : List ℕ := [2, 5, 11, 20, 32, 47]

def difference_increases_by_three (seq : List ℕ) : Prop :=
  ∀ i : ℕ, i + 3 < seq.length →
    (seq[i+2]! - seq[i+1]!) - (seq[i+1]! - seq[i]!) = 3

theorem x_equals_32 (seq : List ℕ) (h : seq = my_sequence) :
  difference_increases_by_three seq →
  seq[4]! = 32 := by
  intro hdiff
  rw [h]
  rfl

#eval my_sequence[4]!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_32_l1174_117477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1174_117433

-- Define the rectangular parallelepiped
noncomputable def AB : ℝ := 4
noncomputable def BC : ℝ := 2
noncomputable def CG : ℝ := 3

-- Define point N as the midpoint of FG
noncomputable def N : ℝ × ℝ × ℝ := (AB / 2, BC / 2, CG)

-- Define the base of the pyramid
noncomputable def base_area : ℝ := BC * Real.sqrt (AB^2 + CG^2)

-- Define the height of the pyramid
noncomputable def pyramid_height : ℝ := CG

-- Theorem statement
theorem pyramid_volume : 
  (1 / 3 : ℝ) * base_area * pyramid_height = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1174_117433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_range_l1174_117420

theorem increasing_sine_range (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin x) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/2) (2*π/3), Monotone (λ y ↦ f (ω * y))) →
  ω ∈ Set.Ioo 0 (3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_range_l1174_117420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_holds_m_range_correct_l1174_117472

-- Define the function f
def f (x : ℝ) : ℝ := |3*x - 2| + |x - 2|

-- Statement for part (I)
theorem solution_set_f : Set.Icc (-1) 3 = { x : ℝ | f x ≤ 8 } := by sorry

-- Statement for part (II)
theorem inequality_holds (x : ℝ) (m : ℝ) (h : 0 ≤ m ∧ m ≤ 1) : 
  f x ≥ (m^2 - m + 2) * |x| := by sorry

-- The range of m
def m_range : Set ℝ := Set.Icc 0 1

theorem m_range_correct : 
  ∀ m, m ∈ m_range ↔ ∀ x, f x ≥ (m^2 - m + 2) * |x| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_holds_m_range_correct_l1174_117472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l1174_117499

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  4 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l1174_117499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l1174_117479

noncomputable section

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := C P.1 P.2

-- Define the relationship between P and Q
def PQ_relation (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1, P.2 - Q.2) = (9*(F.1 - Q.1), 9*(F.2 - Q.2))

-- Define the slope of line OQ
noncomputable def slope_OQ (Q : ℝ × ℝ) : ℝ := Q.2 / Q.1

-- Theorem statement
theorem max_slope_OQ :
  ∃ (max_slope : ℝ),
    (∀ (P Q : ℝ × ℝ), on_parabola P → PQ_relation P Q → slope_OQ Q ≤ max_slope) ∧
    (∃ (P Q : ℝ × ℝ), on_parabola P ∧ PQ_relation P Q ∧ slope_OQ Q = max_slope) ∧
    max_slope = 1/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l1174_117479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fourth_root_l1174_117465

theorem simplified_fourth_root : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ 
  (((2^8 : ℝ) * (5^3 : ℝ))^(1/4 : ℝ) = (c : ℝ) * ((d : ℝ)^(1/4 : ℝ))) ∧ c + d = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fourth_root_l1174_117465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l1174_117441

theorem triangle_tan_a (A B C a b c : ℝ) : 
  -- Triangle conditions
  0 < A ∧ A < π/2 ∧
  B = π/4 ∧
  A + B + C = π ∧
  -- Side lengths form arithmetic sequence
  2 * b^2 = a^2 + c^2 ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C
  →
  Real.tan A = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_a_l1174_117441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_overlap_probability_l1174_117482

/-- The probability of a coin overlapping black regions in a square -/
theorem coin_overlap_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) : 
  square_side = 10 → 
  triangle_leg = 3 → 
  diamond_side = 3 * Real.sqrt 2 → 
  coin_diameter = 2 → 
  (30 + 12 * Real.sqrt 2 + Real.pi) / 64 = 
    ((4 * (1/2 * triangle_leg ^ 2) + 4 * (Real.pi / 4 + triangle_leg) + 
      diamond_side ^ 2 + Real.pi + 4 * diamond_side / Real.sqrt 2) / 
     (square_side - coin_diameter) ^ 2) :=
by
  -- Introduce assumptions
  intro h_square h_triangle h_diamond h_coin
  
  -- Perform calculations
  have reduced_square_side : ℝ := square_side - coin_diameter
  have reduced_square_area : ℝ := reduced_square_side ^ 2
  have triangle_area : ℝ := 4 * (1/2 * triangle_leg ^ 2)
  have triangle_extra_area : ℝ := 4 * (Real.pi / 4 + triangle_leg)
  have diamond_area : ℝ := diamond_side ^ 2
  have diamond_extra_area : ℝ := Real.pi + 4 * diamond_side / Real.sqrt 2
  
  -- Prove the equality
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_overlap_probability_l1174_117482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l1174_117454

theorem largest_divisor_of_n4_minus_n2 :
  ∃ (k : ℕ), k = 12 ∧ 
  (∀ (n : ℤ), (k : ℤ) ∣ (n^4 - n^2)) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℤ), ¬((m : ℤ) ∣ (n^4 - n^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l1174_117454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_true_statements_l1174_117427

/-- Represents a child in the circle with their height -/
structure Child where
  height : ℕ

/-- Represents the circle of children -/
def Circle := Vector Child 16

/-- Checks if a child's statement is true -/
def statementIsTrue (circle : Circle) (i : Fin 16) : Bool :=
  let rightIndex : Fin 16 := ⟨(i.val + 1) % 16, by sorry⟩
  let leftIndex : Fin 16 := ⟨(i.val - 1 + 16) % 16, by sorry⟩
  (circle.get rightIndex).height > (circle.get leftIndex).height

/-- Counts the number of true statements in the circle -/
def countTrueStatements (circle : Circle) : ℕ :=
  (List.range 16).filter (fun i => statementIsTrue circle ⟨i, by sorry⟩) |>.length

/-- The main theorem stating that the minimum number of true statements is 2 -/
theorem min_true_statements (circle : Circle) 
    (different_heights : ∀ i j : Fin 16, i ≠ j → (circle.get i).height ≠ (circle.get j).height) : 
    ∃ (c : Circle), countTrueStatements c = 2 ∧ 
    ∀ (c' : Circle), countTrueStatements c' ≥ 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_true_statements_l1174_117427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1174_117467

/-- Hyperbola C with given properties and point P -/
structure Hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) where
  F1 : ℝ × ℝ  -- Left focus
  F2 : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Point on right branch
  M : ℝ × ℝ   -- Center of inscribed circle
  G : ℝ × ℝ   -- Centroid of triangle PF1F2
  hC : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = P ∨ (x, y) = F1 ∨ (x, y) = F2
  hP : P.1 > 0  -- P is on the right branch
  hR : Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) - 
       Real.sqrt ((P.2 - F2.1)^2 + (P.2 - F2.2)^2) = 2 * a  -- Definition of hyperbola
  hI : ∀ (x y : ℝ), (x - M.1)^2 + (y - M.2)^2 = a^2 →
       (x - F1.1)^2 + (y - F1.2)^2 ≥ a^2 ∧
       (x - F2.1)^2 + (y - F2.2)^2 ≥ a^2 ∧
       (x - P.1)^2 + (y - P.2)^2 ≥ a^2  -- Inscribed circle property
  hG : G = ((P.1 + F1.1 + F2.1) / 3, (P.2 + F1.2 + F2.2) / 3)  -- Centroid definition
  hMG : M.2 = G.2  -- MG parallel to x-axis

/-- The eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Hyperbola a b ha hb) : 
  Real.sqrt (1 + b^2 / a^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1174_117467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drain_rate_is_twenty_l1174_117469

/-- Represents the tank and pipe system -/
structure TankSystem where
  capacity : ℚ
  pipeA_rate : ℚ
  pipeB_rate : ℚ
  fill_time : ℚ

/-- Calculates the drain rate of pipe C given a TankSystem -/
def calculate_drain_rate (system : TankSystem) : ℚ :=
  let cycle_count := system.fill_time / 3
  let net_fill_per_cycle := system.pipeA_rate + system.pipeB_rate - (system.capacity / cycle_count)
  system.pipeA_rate + system.pipeB_rate - net_fill_per_cycle

/-- Theorem stating that for the given system, the drain rate is 20 liters/minute -/
theorem drain_rate_is_twenty (system : TankSystem) 
  (h_capacity : system.capacity = 900)
  (h_pipeA : system.pipeA_rate = 40)
  (h_pipeB : system.pipeB_rate = 30)
  (h_fill_time : system.fill_time = 54) :
  calculate_drain_rate system = 20 := by
  sorry

#eval calculate_drain_rate { capacity := 900, pipeA_rate := 40, pipeB_rate := 30, fill_time := 54 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drain_rate_is_twenty_l1174_117469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qinzhou_huang_max_profit_l1174_117438

-- Define the yield function T(x)
noncomputable def T (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 25 then x^2 / 20 + 50
  else if 25 < x ∧ x ≤ 50 then 91 * x / (3 + x)
  else 0

-- Define the profit function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  30 * T x - 6 * x

-- State the theorem
theorem qinzhou_huang_max_profit :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 50 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 50 → f y ≤ f x) ∧
  (Int.floor (5 * f x + 0.5) : ℤ) = 11520 ∧
  (Int.floor (x + 0.5) : ℤ) = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qinzhou_huang_max_profit_l1174_117438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_changed_number_proof_l1174_117498

theorem changed_number_proof (numbers : List ℝ) (changed_index : Fin numbers.length) : 
  numbers.length = 9 →
  numbers.sum / numbers.length = 9 →
  let new_numbers := numbers.set changed_index 9
  new_numbers.sum / new_numbers.length = 8 →
  numbers.get changed_index = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_changed_number_proof_l1174_117498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinate_l1174_117491

/-- The equation of a hyperbola in the form x^2 - 2y^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2*y^2 = 1

/-- The x-coordinate of the right focus of the hyperbola -/
noncomputable def right_focus_x : ℝ := Real.sqrt 6 / 2

/-- Theorem stating that the x-coordinate of the right focus of the hyperbola x^2 - 2y^2 = 1 is √6/2 -/
theorem right_focus_coordinate (x y : ℝ) : 
  hyperbola_equation x y → x = right_focus_x ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinate_l1174_117491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_l1174_117457

theorem quadratic_minimum (a : ℝ) (h1 : a < -1) :
  (∃ f : ℝ → ℝ, f = λ x ↦ x^2 + a*x) ∧
  (∀ x, x^2 + a*x ≤ -x) ∧
  (∃ m, m = -1/2 ∧ ∀ x, x^2 + a*x ≥ m) →
  a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_l1174_117457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1174_117400

/-- The area of a trapezoid with vertices (0, 0), (0, 3), (4, 5), and (4, 1) is 14 square units. -/
theorem trapezoid_area : 14 = (1 / 2 : ℝ) * (3 + 4) * 4 := by
  -- Calculate the area using the trapezoid formula
  calc
    14 = (1 / 2 : ℝ) * 7 * 4 := by norm_num
    _ = (1 / 2 : ℝ) * (3 + 4) * 4 := by ring

#check trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1174_117400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_planes_is_one_or_three_l1174_117443

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Check if a point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk 
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

/-- Three lines intersecting pairwise -/
structure ThreePairwiseIntersectingLines where
  l1 : Line3D
  l2 : Line3D
  l3 : Line3D
  intersect_pairwise : 
    (∃ p : Point3D, point_on_line p l1 ∧ point_on_line p l2) ∧ 
    (∃ q : Point3D, point_on_line q l2 ∧ point_on_line q l3) ∧ 
    (∃ r : Point3D, point_on_line r l3 ∧ point_on_line r l1)

/-- The number of planes determined by three pairwise intersecting lines -/
noncomputable def num_planes_determined (lines : ThreePairwiseIntersectingLines) : Nat :=
  sorry

/-- Theorem: The number of planes determined by three pairwise intersecting lines is either 1 or 3 -/
theorem num_planes_is_one_or_three (lines : ThreePairwiseIntersectingLines) :
  num_planes_determined lines = 1 ∨ num_planes_determined lines = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_planes_is_one_or_three_l1174_117443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_inequality_l1174_117484

open Real BigOperators

variable (n : ℕ)
variable (x : Fin n → ℝ)

def x_pos (n : ℕ) (x : Fin n → ℝ) : Prop := ∀ i, 0 < x i

def x_n_plus_1 (n : ℕ) (x : Fin n → ℝ) : ℝ := ∑ i, x i

theorem sum_sqrt_inequality (h : x_pos n x) :
  ∑ i, sqrt (x i * (x_n_plus_1 n x - x i)) ≤ 
  sqrt (∑ i, x_n_plus_1 n x * (x_n_plus_1 n x - x i)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_inequality_l1174_117484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_exists_l1174_117478

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of 9 points
def NinePoints := Fin 9 → Point

-- Check if three points are collinear
def are_collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Check if four points form a convex quadrilateral
def is_convex_quadrilateral (p q r s : Point) : Prop :=
  ∀ (a b c : Point), a ∈ ({p, q, r, s} : Set Point) → b ∈ ({p, q, r, s} : Set Point) → c ∈ ({p, q, r, s} : Set Point) →
    a ≠ b → b ≠ c → a ≠ c → ¬are_collinear a b c

-- Check if five points form a convex pentagon
def is_convex_pentagon (p q r s t : Point) : Prop :=
  ∀ (a b c : Point), a ∈ ({p, q, r, s, t} : Set Point) → b ∈ ({p, q, r, s, t} : Set Point) → c ∈ ({p, q, r, s, t} : Set Point) →
    a ≠ b → b ≠ c → a ≠ c → ¬are_collinear a b c

-- Main theorem
theorem convex_pentagon_exists (points : NinePoints) 
  (h1 : ∃ (a b c d : Fin 9), is_convex_quadrilateral (points a) (points b) (points c) (points d))
  (h2 : ∀ (a b c : Fin 9), a ≠ b → b ≠ c → a ≠ c → ¬are_collinear (points a) (points b) (points c)) :
  ∃ (a b c d e : Fin 9), is_convex_pentagon (points a) (points b) (points c) (points d) (points e) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_exists_l1174_117478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_semiellipses_area_l1174_117475

theorem hexagon_minus_semiellipses_area :
  ∃ (s : ℝ) (n : ℕ),
    s > 0 ∧ n > 0 ∧
    let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
    let semiellipse_area := n * (Real.pi / 2 * s * (s / 2))
    hexagon_area - semiellipse_area = 48 * Real.sqrt 3 - 16 * Real.pi := by
  use 4, 4
  apply And.intro
  · exact zero_lt_four
  · apply And.intro
    · exact four_pos
    · sorry  -- The actual calculation proof goes here

#check hexagon_minus_semiellipses_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_semiellipses_area_l1174_117475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l1174_117497

def children_ages (ages : List ℕ) : Prop :=
  ages.length = 7 ∧
  ages.sum / ages.length = 8 ∧
  ages.Pairwise (· ≠ ·) ∧
  ∀ i j, i + 1 = j → j < ages.length → ages[j]! - ages[i]! = 3

theorem oldest_child_age (ages : List ℕ) (h : children_ages ages) :
  ages.maximum? = some 17 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l1174_117497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_problem_l1174_117408

-- Define the geometric series sum T(r) for -1 < r < 1
noncomputable def T (r : ℝ) : ℝ := 8 / (1 - r)

-- State the theorem
theorem geometric_series_sum_problem (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 1152) : T b + T (-b) = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_problem_l1174_117408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_shifted_l1174_117431

-- Define the function f'(x)
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the interval of decrease for f(x-1)
def interval_of_decrease : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem decreasing_interval_of_f_shifted :
  (∀ x, f' x = x^2 - 4*x + 3) →
  (∀ x ∈ interval_of_decrease, f' (x-1) < 0) :=
by
  intro h x hx
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_shifted_l1174_117431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l1174_117432

/-- The function f(x) = sin(x) * sin(x/2) * sin(x/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.sin (x/2) * Real.sin (x/3)

/-- The period of a function -/
def isPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The smallest positive period of f is 12π -/
theorem smallest_period_of_f :
  ∃ T : ℝ, T > 0 ∧ isPeriod f T ∧ ∀ T' : ℝ, T' > 0 → isPeriod f T' → T ≤ T' ∧ T = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l1174_117432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_6656_l1174_117418

theorem largest_prime_factor_of_6656 : (Nat.factors 6656).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_6656_l1174_117418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_independence_constant_l1174_117416

noncomputable def integral_independence (a : ℝ) : ℝ :=
  ∫ (x : ℝ) in Set.Ici 0, 1 / ((1 + x^2) * (1 + x^a))

theorem integral_independence_constant (a : ℝ) : 
  integral_independence a = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_independence_constant_l1174_117416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_greater_than_one_l1174_117483

noncomputable def S : ℝ := 1 / (2 - Real.rpow 7 (1/3)) + 1 / (Real.rpow 7 (1/3) - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.rpow 2 (1/3))

theorem S_greater_than_one : S > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_greater_than_one_l1174_117483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1174_117481

/-- The speed of a train in km/hr, given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_calculation :
  train_speed 100 2.5 = 144 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_div_eq_mul_div]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1174_117481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_positive_T_l1174_117421

/-- Sequence a_n with sum S_n -/
def a (n : ℕ+) (k : ℝ) : ℝ := 2 * (n : ℝ) + k - 1

/-- Sum of first n terms of a_n -/
def S (n : ℕ+) (k : ℝ) : ℝ := (n : ℝ) + k

/-- Sequence b_n -/
def b (n : ℕ+) : ℝ := 3 * (n : ℝ) + 2

/-- Sequence c_n -/
def c (n : ℕ+) (k : ℝ) : ℝ := a n k - k * b n

/-- Sum of first n terms of c_n -/
noncomputable def T (n : ℕ+) (k : ℝ) : ℝ := 
  ((n : ℝ) / 2) * (2 * c 1 k + ((n : ℝ) - 1) * (2 - 3 * k))

/-- Main theorem -/
theorem max_k_for_positive_T :
  ∀ n : ℕ+, (∀ m : ℕ+, T m 1 > 0) ∧ 
  (∀ k : ℕ+, k > 1 → ∃ m : ℕ+, T m k ≤ 0) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_positive_T_l1174_117421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_150_l1174_117402

/-- The decimal representation of 17/70 has a repeating sequence of 6 digits -/
def decimal_rep_17_70 : ℚ := 17 / 70

/-- The repeating sequence in the decimal representation of 17/70 -/
def repeating_sequence : List ℕ := [4, 1, 4, 2, 8, 5]

/-- The length of the repeating sequence -/
def sequence_length : ℕ := 6

/-- The position we're interested in -/
def target_position : ℕ := 150

theorem digit_at_150 : 
  (repeating_sequence.get! ((target_position - 1) % sequence_length)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_150_l1174_117402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1174_117492

/-- Theorem: For a triangle with sides a and b, angle α between them, 
    and angle bisector length l, the following equation holds:
    l = (2ab cos(α/2)) / (a+b) -/
theorem angle_bisector_length 
  (a b l : ℝ) 
  (α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < α) 
  (h4 : α < π) 
  (h5 : l > 0) 
  (h6 : 2 * a * b * Real.sin (α / 2) / l > 0) : 
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1174_117492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l1174_117409

-- Define the points
def A : Fin 3 → ℝ := ![(-3), 0, 1]
def B : Fin 3 → ℝ := ![2, 1, (-1)]
def C : Fin 3 → ℝ := ![(-2), 2, 0]
def D : Fin 3 → ℝ := ![1, 3, 2]

-- Define the plane passing through A, B, and C
def plane (p : Fin 3 → ℝ) : Prop :=
  ∃ (a b c : ℝ), a * p 0 + b * p 1 + c * p 2 = 0 ∧
                  a * A 0 + b * A 1 + c * A 2 = 0 ∧
                  a * B 0 + b * B 1 + c * B 2 = 0 ∧
                  a * C 0 + b * C 1 + c * C 2 = 0

-- Define the distance function
noncomputable def distance (p : Fin 3 → ℝ) (plane : (Fin 3 → ℝ) → Prop) : ℝ :=
  sorry -- Actual implementation of distance function

-- Theorem statement
theorem distance_to_plane :
  distance D plane = 10 / Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l1174_117409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_touches_sixteen_trees_l1174_117444

/-- Represents a rectangular path with trees -/
structure TreePath :=
  (total_trees : ℕ)
  (tree_spacing : ℚ)

/-- Represents a walk along the path -/
structure Walk :=
  (forward1 : ℚ)
  (backward : ℚ)
  (forward2 : ℚ)

/-- Calculates the number of trees touched during a walk segment -/
def trees_touched (distance : ℚ) (spacing : ℚ) : ℕ :=
  (distance / spacing).floor.toNat + 1

/-- Calculates the total number of trees touched during a walk -/
def total_trees_touched (path : TreePath) (walk : Walk) : ℕ :=
  trees_touched walk.forward1 path.tree_spacing +
  trees_touched walk.backward path.tree_spacing +
  trees_touched walk.forward2 path.tree_spacing

/-- Theorem: The walk touches 16 trees -/
theorem walk_touches_sixteen_trees :
  let path : TreePath := ⟨12, 5⟩
  let walk : Walk := ⟨32, 18, 22⟩
  total_trees_touched path walk = 16 := by
  sorry

#eval let path : TreePath := ⟨12, 5⟩
      let walk : Walk := ⟨32, 18, 22⟩
      total_trees_touched path walk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_touches_sixteen_trees_l1174_117444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_15_feet_l1174_117480

/-- The height of a ball thrown from a building's rooftop as a function of time -/
def ballHeight (t : ℝ) : ℝ := 60 - 9*t - 5*t^2

/-- Theorem stating that there exists a positive time when the ball reaches 15 feet -/
theorem ball_height_15_feet :
  ∃ t : ℝ, t > 0 ∧ ballHeight t = 15 ∧ |t - 2.233| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_15_feet_l1174_117480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_inequality_and_smallest_n_l1174_117466

-- Define the tower function T
def T : ℕ → ℕ → ℕ
| 0, _ => 1  -- Add a base case for n = 0
| 1, w => w
| n + 1, w => w ^ T n w

-- State the theorem
theorem tower_inequality_and_smallest_n :
  (∀ n : ℕ, n ≥ 1 → (4 * T n 3 < T (n + 2) 2 ∧ T (n + 2) 2 < T (n + 1) 3)) ∧
  (∀ n : ℕ, n < 1988 → T n 3 ≤ T 1989 2) ∧
  T 1988 3 > T 1989 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_inequality_and_smallest_n_l1174_117466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_zero_integral_implies_zero_l1174_117459

theorem continuous_function_zero_integral_implies_zero 
  (f : ℝ → ℝ) (a b : ℝ) (h_le : a ≤ b) (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_int : ∀ (n : ℕ), ∫ x in a..b, (x^n * f x) = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_zero_integral_implies_zero_l1174_117459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_is_parabola_specific_quadratic_is_parabola_l1174_117453

/-- A quadratic function of the form y = a(x-h)^2 + k, where a ≠ 0 -/
def QuadraticFunction (a h k : ℝ) : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k

/-- Predicate to define if a function is a parabola -/
def IsParabola (f : ℝ → ℝ) : Prop := sorry

/-- The graph of a quadratic function is a parabola -/
theorem quadratic_graph_is_parabola (a h k : ℝ) (ha : a ≠ 0) :
  IsParabola (QuadraticFunction a h k) :=
by sorry

/-- The specific quadratic function y = 3(x-2)^2 + 6 -/
def specific_quadratic : ℝ → ℝ := QuadraticFunction 3 2 6

/-- The graph of y = 3(x-2)^2 + 6 is a parabola -/
theorem specific_quadratic_is_parabola :
  IsParabola specific_quadratic :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_is_parabola_specific_quadratic_is_parabola_l1174_117453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_miles_is_twelve_l1174_117488

/-- Represents the number of miles driven on each day -/
structure DailyMiles where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ

/-- Calculates the average miles driven per day -/
def averageMiles (d : DailyMiles) : ℚ :=
  (d.monday + d.tuesday + d.wednesday) / 3

/-- Theorem stating that given the conditions, the miles driven on Monday is 12 -/
theorem monday_miles_is_twelve :
  ∀ d : DailyMiles,
  d.tuesday = 18 ∧
  d.wednesday = 21 ∧
  averageMiles d = 17 →
  d.monday = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_miles_is_twelve_l1174_117488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1174_117435

theorem triangle_area_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 + x*y = 529)
  (h2 : x^2 + z^2 + Real.sqrt 3 * x*z = 441)
  (h3 : z^2 + y^2 = 144) :
  Real.sqrt 3 * x*y + 2*y*z + x*z = 224 * Real.sqrt 5 := by
  sorry

#check triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1174_117435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_is_29_to_28_l1174_117474

-- Define the cars
structure Car where
  speed : ℝ
  time : ℝ

-- Define the groups
structure GroupComposition where
  carA_count : ℕ
  carB_count : ℕ
  carC_count : ℕ

-- Define the problem parameters
def carA : Car := { speed := 80, time := 4 }
def carB : Car := { speed := 100, time := 2 }
def carC : Car := { speed := 120, time := 3 }

def group1 : GroupComposition := { carA_count := 3, carB_count := 1, carC_count := 0 }
def group2 : GroupComposition := { carA_count := 0, carB_count := 2, carC_count := 2 }

-- Calculate distance for a car
def distance (car : Car) : ℝ := car.speed * car.time

-- Calculate total distance for a group
def groupDistance (group : GroupComposition) : ℝ :=
  group.carA_count * distance carA +
  group.carB_count * distance carB +
  group.carC_count * distance carC

-- Theorem to prove
theorem distance_ratio_is_29_to_28 :
  (groupDistance group1) / (groupDistance group2) = 29 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_is_29_to_28_l1174_117474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_inequality_range_restriction_l1174_117485

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 3) / (x + 1)

-- Theorem for the solution of f(x) > 1
theorem solution_inequality :
  ∀ x : ℝ, f x > 1 ↔ (-1 < x ∧ x < 1) ∨ x > 2 :=
by
  sorry

-- Theorem for the range of f(x) when x ∈ (1, 3)
theorem range_restriction :
  ∀ x : ℝ, 1 < x ∧ x < 3 →
  ∃ y : ℝ, f x = y ∧ 2 * Real.sqrt 6 - 4 ≤ y ∧ y < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_inequality_range_restriction_l1174_117485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l1174_117460

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem unused_sector_angle (R : ℝ) : 
  let r : ℝ := 15
  let V : ℝ := 675 * Real.pi
  let h : ℝ := V / ((1/3) * Real.pi * r^2)
  let s : ℝ := Real.sqrt (r^2 + h^2)
  let used_angle : ℝ := (30 * Real.pi / (2 * Real.pi * s)) * 360
  ∃ ε > 0, |360 - used_angle - 164.66| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l1174_117460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_poly_sum_theorem_l1174_117451

-- Define a monic polynomial of degree 5
def monicPoly (a b c d : ℝ) : ℝ → ℝ := λ x ↦ x^5 + a*x^4 + b*x^3 + c*x^2 + d*x

theorem monic_poly_sum_theorem (a b c d : ℝ) :
  let q := monicPoly a b c d
  (q 1 = 24) → (q 2 = 48) → (q 3 = 72) →
  ∃ s t : ℝ, q 0 + q 5 = (-1)*(-2)*(-3)*(-s)*(-t) + 4*3*2*(5-s)*(5-t) + 24*5 := by
  sorry

#check monic_poly_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_poly_sum_theorem_l1174_117451
