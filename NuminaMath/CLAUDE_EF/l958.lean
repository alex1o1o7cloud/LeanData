import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_c_highest_increase_l958_95876

structure Stock where
  name : String
  openPrice : ℚ
  closePrice : ℚ

def percentIncrease (s : Stock) : ℚ :=
  (s.closePrice - s.openPrice) / s.openPrice * 100

theorem stock_c_highest_increase (stockA stockB stockC : Stock)
  (hA : stockA.name = "A" ∧ stockA.openPrice = 28 ∧ stockA.closePrice = 29)
  (hB : stockB.name = "B" ∧ stockB.openPrice = 55 ∧ stockB.closePrice = 57)
  (hC : stockC.name = "C" ∧ stockC.openPrice = 75 ∧ stockC.closePrice = 78) :
  percentIncrease stockC > percentIncrease stockA ∧
  percentIncrease stockC > percentIncrease stockB :=
by
  sorry

#eval percentIncrease ⟨"A", 28, 29⟩
#eval percentIncrease ⟨"B", 55, 57⟩
#eval percentIncrease ⟨"C", 75, 78⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_c_highest_increase_l958_95876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sheet_perimeter_l958_95860

/-- Represents a rectangular sheet that can be folded. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Represents the first fold of the sheet. -/
def first_fold (r : Rectangle) : Rectangle :=
  { length := r.length, width := r.width - 10 }

/-- Represents the second fold of the sheet. -/
def second_fold (r : Rectangle) : Rectangle :=
  { length := r.length, width := r.width - 18 }

/-- Represents the third fold of the sheet. -/
def third_fold (r : Rectangle) : Rectangle :=
  { length := r.length, width := r.width - 26 }

/-- Given a rectangular sheet that can be folded twice, this theorem proves
    that the perimeter of the original sheet is 92 units, based on the
    differences in perimeters of the folded rectangles. -/
theorem original_sheet_perimeter
  (sheet : Rectangle)
  (h1 : perimeter (first_fold sheet) = perimeter (second_fold sheet) + 20)
  (h2 : perimeter (second_fold sheet) = perimeter (third_fold sheet) + 16)
  : perimeter sheet = 92 :=
by
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sheet_perimeter_l958_95860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l958_95829

theorem log_inequality : ∀ a b c : ℝ,
  a = Real.log 6 / Real.log 3 →
  b = Real.log 10 / Real.log 5 →
  c = Real.log 14 / Real.log 7 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l958_95829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_cubes_count_l958_95898

/-- Represents a cube with painted stripes -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  stripesPerFace : Nat
  stripeLength : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedUnitCubes cube
where
  paintedUnitCubes (c : PaintedCube) : Nat := sorry

/-- Theorem stating that a 6x6x6 cube with specific painted stripes has 144 unpainted unit cubes -/
theorem unpainted_cubes_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    stripesPerFace := 2,
    stripeLength := 6
  }
  unpaintedUnitCubes cube = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_cubes_count_l958_95898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l958_95890

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => Real.sqrt (sequence_a (n + 1) ^ 2 + 1 / sequence_a (n + 1))

theorem sequence_a_bounds :
  ∃ (α : ℝ), α = 1/3 ∧ ∀ (n : ℕ), n ≥ 1 →
    1/2 ≤ sequence_a n / n^α ∧ sequence_a n / n^α ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l958_95890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_y_squared_equals_five_l958_95835

theorem x_squared_plus_y_squared_equals_five 
  (x y : ℝ) (hy : y > 0)
  (h : Set.toFinset {x^2 + x + 1, -x, -x - 1} = Set.toFinset {-y, -y/2, y + 1}) :
  x^2 + y^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_y_squared_equals_five_l958_95835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l958_95816

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the condition AC = DB
def diagonalsEqual (q : Quadrilateral) : Prop :=
  dist q.A q.C = dist q.D q.B

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) (M : EuclideanSpace ℝ (Fin 2)) 
  (h : diagonalsEqual q) : 
  dist M q.A < dist M q.B + dist M q.C + dist M q.D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l958_95816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l958_95850

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 4) / (x - 1)

-- State the theorem
theorem f_minimum_value :
  ∀ x > 1, f x ≥ 5 ∧ (f x = 5 ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l958_95850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l958_95856

theorem tan_value_fourth_quadrant (α : ℝ) 
  (h1 : Real.sin α = -Real.sqrt 3 / 2)
  (h2 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) : 
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l958_95856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l958_95877

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the mowing rate of a lawn mower -/
structure MowingRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowing_time (area : LawnArea) (rate : MowingRate) : ℝ :=
  area.size / rate.rate

theorem carlos_finishes_first 
  (andy_lawn : LawnArea)
  (beth_lawn : LawnArea)
  (carlos_lawn : LawnArea)
  (andy_rate : MowingRate)
  (beth_rate : MowingRate)
  (carlos_rate : MowingRate)
  (h1 : andy_lawn.size = 3 * beth_lawn.size)
  (h2 : andy_lawn.size = 4 * carlos_lawn.size)
  (h3 : carlos_rate.rate = beth_rate.rate)
  (h4 : carlos_rate.rate = andy_rate.rate / 2) :
  mowing_time carlos_lawn carlos_rate < min (mowing_time andy_lawn andy_rate) (mowing_time beth_lawn beth_rate) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l958_95877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l958_95805

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem product_of_b_values (b : ℝ) : 
  (distance (3*b) (b+2) 6 3 = 3 * Real.sqrt 5) →
  (∃ b' : ℝ, (distance (3*b') (b'+2) 6 3 = 3 * Real.sqrt 5) ∧ b * b' = -0.8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l958_95805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_equivalent_to_at_least_one_l958_95883

-- Define propositions p and q
def p : Prop := sorry
def q : Prop := sorry

-- Define the statement "At least one of A and B exceeded 2 meters in their trial jump"
def at_least_one_exceeded : Prop := sorry

-- Theorem to prove
theorem p_or_q_equivalent_to_at_least_one : (p ∨ q) ↔ at_least_one_exceeded :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_equivalent_to_at_least_one_l958_95883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_multiple_of_five_l958_95855

def digits : Finset Nat := {1, 2, 4, 5, 6}

def is_multiple_of_five (n : Nat) : Bool := n % 5 = 0

def sum_of_digits (d₁ d₂ d₃ : Nat) : Nat := d₁ + d₂ + d₃

def valid_combination (d₁ d₂ d₃ : Nat) : Bool :=
  d₁ ∈ digits ∧ d₂ ∈ digits ∧ d₃ ∈ digits ∧ d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

theorem probability_sum_multiple_of_five :
  (Finset.filter (fun (t : Nat × Nat × Nat) =>
    let (d₁, d₂, d₃) := t
    valid_combination d₁ d₂ d₃ ∧ is_multiple_of_five (sum_of_digits d₁ d₂ d₃))
    (Finset.product digits (Finset.product digits digits))).card /
  (Finset.filter (fun (t : Nat × Nat × Nat) =>
    let (d₁, d₂, d₃) := t
    valid_combination d₁ d₂ d₃)
    (Finset.product digits (Finset.product digits digits))).card = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_multiple_of_five_l958_95855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_k_range_l958_95852

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then
    (k - 1) * x^2 - 3 * (k - 1) * x + (13 * k - 9) / 4
  else
    (1/2)^x - 1

theorem function_decreasing_implies_k_range (k : ℝ) :
  (∀ n : ℕ+, f k (n + 1) < f k n) → k < -1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_k_range_l958_95852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l958_95861

def A : Set ℝ := {x | x^2 - 6*x + 5 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x < a + 1}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty → a ∈ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l958_95861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_condition_l958_95845

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 2 * (1 - x) * Real.sin (a * x)

/-- The theorem statement -/
theorem two_intersection_points_condition (a : ℝ) : 
  a > 0 → 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ (Set.Ioo 0 1) ∧ x₂ ∈ (Set.Ioo 0 1) ∧ 
    f a x₁ = 2 * x₁ - 1 ∧ f a x₂ = 2 * x₂ - 1) → 
  a ∈ Set.Ioc (5 * Real.pi / 6) (13 * Real.pi / 6) := by
  sorry

#check two_intersection_points_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_condition_l958_95845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l958_95832

/-- Represents a square on the chessboard -/
inductive Square
| Light
| Dark

/-- A 9x9 chessboard with alternating light and dark squares -/
def Chessboard : Type := Fin 9 → Fin 9 → Square

/-- A standard chessboard pattern where squares alternate -/
def standardPattern : Chessboard :=
  fun row col => if (row.val + col.val) % 2 = 0 then Square.Dark else Square.Light

/-- Count the number of dark squares on the chessboard -/
def countDarkSquares (board : Chessboard) : Nat :=
  (List.sum (List.map (fun i => List.sum (List.map (fun j =>
    match board i j with
    | Square.Dark => 1
    | Square.Light => 0) (List.range 9))) (List.range 9)))

/-- Count the number of light squares on the chessboard -/
def countLightSquares (board : Chessboard) : Nat :=
  (List.sum (List.map (fun i => List.sum (List.map (fun j =>
    match board i j with
    | Square.Light => 1
    | Square.Dark => 0) (List.range 9))) (List.range 9)))

/-- Theorem: The number of dark squares exceeds the number of light squares by exactly one -/
theorem dark_exceeds_light_by_one :
  countDarkSquares standardPattern = countLightSquares standardPattern + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l958_95832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l958_95881

open Real

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), 2 * sin (2*x + Real.pi/6) - m = 0

def q (m : ℝ) : Prop := ∃ x > 0, x^2 - 2*m*x + 1 < 0

-- State the theorem
theorem m_range (m : ℝ) (h : p m ∧ ¬(q m)) : m ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l958_95881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l958_95879

/-- Represents a hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of b to a in the equation of the asymptotes y = ±(b/a)x -/
  asymptote_slope : ℝ
  /-- Assumption that the asymptote slope is positive -/
  asymptote_slope_pos : asymptote_slope > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.asymptote_slope ^ 2)

/-- Theorem: The eccentricity of a hyperbola with asymptotes y = ±2x is √5 -/
theorem hyperbola_eccentricity_sqrt_5 (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 2) : eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l958_95879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangle_area_formula_l958_95866

/-- The area of a regular triangle with perimeter a -/
noncomputable def regular_triangle_area (a : ℝ) : ℝ := (Real.sqrt 3 * a^2) / 36

/-- Theorem: The area of a regular triangle with perimeter a is (√3 * a^2) / 36 -/
theorem regular_triangle_area_formula (a : ℝ) (h : a > 0) :
  regular_triangle_area a = (Real.sqrt 3 * a^2) / 36 := by
  -- Unfold the definition of regular_triangle_area
  unfold regular_triangle_area
  -- The definition and the right-hand side are syntactically equal
  rfl

#check regular_triangle_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangle_area_formula_l958_95866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stimulus_tax_revenue_ratio_l958_95872

/-- Proves that the ratio of tax revenue to stimulus cost is 5, given the specified conditions. -/
theorem stimulus_tax_revenue_ratio
  (city_population : ℕ)
  (stimulus_amount : ℕ)
  (stimulus_percentage : ℚ)
  (government_profit : ℕ)
  (h_population : city_population = 1000)
  (h_stimulus : stimulus_amount = 2000)
  (h_percentage : stimulus_percentage = 1/5)
  (h_profit : government_profit = 1600000) :
  (government_profit + city_population * stimulus_percentage * stimulus_amount) /
  (city_population * stimulus_percentage * stimulus_amount) = 5 := by
  sorry

#check stimulus_tax_revenue_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stimulus_tax_revenue_ratio_l958_95872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_searchlight_revolutions_l958_95869

theorem searchlight_revolutions (p : ℝ) (h : p = 2/3) :
  ∃ r : ℕ, r > 0 ∧ r ≤ 2 ∧ p = (60 / r - 20 / r) / (60 / r) :=
by
  use 2
  constructor
  · exact Nat.succ_pos 1
  constructor
  · simp
  · field_simp
    rw [h]
    ring

#check searchlight_revolutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_searchlight_revolutions_l958_95869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_for_nested_F_l958_95814

/-- Definition of F_n(a) as described in the problem -/
def F (n : ℕ+) (a : ℕ) : ℕ :=
  (a / n) + (a % n)

/-- The statement to be proved -/
theorem largest_A_for_nested_F : ∃ (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ+),
  (∀ a : ℕ+, a ≤ 53590 →
    F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) ∧
  (∀ A : ℕ+, A > 53590 →
    ¬∃ (m₁ m₂ m₃ m₄ m₅ m₆ : ℕ+),
      ∀ a : ℕ+, a ≤ A →
        F m₆ (F m₅ (F m₄ (F m₃ (F m₂ (F m₁ a))))) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_for_nested_F_l958_95814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_range_l958_95839

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + Real.pi

def is_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f x₁ = (f b - f a) / (b - a)) ∧
    (deriv^[2] f x₂ = (f b - f a) / (b - a))

theorem mean_value_function_range (m : ℝ) :
  is_mean_value_function f 0 m → 3/4 < m ∧ m < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_range_l958_95839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_is_one_l958_95833

noncomputable section

-- Define the rectangle ABCD
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 2, 0)
def C : ℝ × ℝ × ℝ := (3, 2, 0)
def D : ℝ × ℝ × ℝ := (3, 0, 0)

-- Define the points A', B', C', D'
def A' : ℝ × ℝ × ℝ := (0, 0, 12)
def B' : ℝ × ℝ × ℝ := (0, 2, 6)
def C' : ℝ × ℝ × ℝ := (3, 2, 20)
def D' : ℝ × ℝ × ℝ := (3, 0, 24)

-- Define midpoints M and N
def M : ℝ × ℝ × ℝ := ((A'.1 + C'.1) / 2, (A'.2.1 + C'.2.1) / 2, (A'.2.2 + C'.2.2) / 2)
def N : ℝ × ℝ × ℝ := ((B'.1 + D'.1) / 2, (B'.2.1 + D'.2.1) / 2, (B'.2.2 + D'.2.2) / 2)

-- Define the distance function
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2)

theorem length_of_MN_is_one : distance M N = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_is_one_l958_95833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airfield_problem_l958_95800

/-- Represents an airfield -/
structure Airfield :=
  (id : Nat)

/-- Represents the distance between two airfields -/
def distance (a b : Airfield) : ℝ :=
  sorry  -- In reality, this would be a function calculating the distance

/-- Predicate to check if all distances are distinct -/
def all_distances_distinct (airfields : List Airfield) : Prop :=
  ∀ a b c d, a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) → distance a b ≠ distance c d

/-- The main theorem representing the problem -/
theorem airfield_problem :
  ∃ (airfields : List Airfield),
    airfields.length = 1985 ∧
    all_distances_distinct airfields ∧
    ∃ (landing_airfields : List Airfield),
      landing_airfields.length = 50 ∧
      (∀ a ∈ airfields, ∃ b ∈ landing_airfields,
        ∀ c ∈ airfields, c ≠ a → distance a b ≥ distance a c) :=
by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airfield_problem_l958_95800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l958_95858

noncomputable def slope_angle (a b c : ℝ) : ℝ :=
  Real.arctan (-a/b)

theorem line_slope_angle (x y : ℝ) :
  let line := {(x, y) | x + y - 3 = 0}
  let θ := slope_angle 1 1 (-3)
  (Real.tan θ = -1 ∧ 0 ≤ θ ∧ θ < Real.pi) → θ = 135 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l958_95858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l958_95831

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius. -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is on a given circle. -/
def IsOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents the measure of an arc on a circle. -/
noncomputable def ArcMeasure (c : Circle) (p1 p2 : Point) : ℝ := sorry

/-- Given a circle with center O and radius 10, where an arc AB spans 5/6 of the circle's
    circumference, the perimeter of the region formed by this arc and two radii OA and OB
    is equal to 20 + (50/3)π. -/
theorem shaded_region_perimeter (O : Point) (A B : Point) :
  let r : ℝ := 10
  let circle := Circle.mk O r
  let arc_fraction : ℝ := 5/6
  let arc_length := arc_fraction * (2 * π * r)
  let perimeter := 2 * r + arc_length
  IsOnCircle A circle → IsOnCircle B circle →
  ArcMeasure circle A B = arc_fraction * (2 * π) →
  perimeter = 20 + (50/3) * π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l958_95831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_16_l958_95888

-- Define the sector
noncomputable def Sector (perimeter : ℝ) (centralAngle : ℝ) : Type :=
  {r : ℝ // perimeter = 2 * r + centralAngle * r}

-- Define the area of a sector
noncomputable def sectorArea (p α : ℝ) (s : Sector p α) : ℝ :=
  (1/2) * α * s.val^2

-- Theorem statement
theorem sector_area_16 :
  ∀ (s : Sector 16 2), sectorArea 16 2 s = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_16_l958_95888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l958_95897

/-- A shop owner uses false weights that cheat by 10% while buying and selling. -/
def cheating_percentage : ℚ := 1/10

/-- The professed cost price of an article -/
def professed_cost : ℚ := 100

/-- The actual worth of goods received when buying for the professed cost -/
noncomputable def actual_worth_bought : ℚ := professed_cost * (1 + cheating_percentage)

/-- The actual worth of goods given when selling for the professed cost -/
noncomputable def actual_worth_sold : ℚ := professed_cost * (1 - cheating_percentage)

/-- The actual cost of goods sold for the professed cost -/
noncomputable def actual_cost : ℚ := actual_worth_bought * (actual_worth_sold / professed_cost)

/-- The profit made on the sale -/
noncomputable def profit : ℚ := professed_cost - actual_cost

/-- The percentage profit -/
noncomputable def percentage_profit : ℚ := (profit / actual_cost) * 100

/-- Theorem stating that the percentage profit is approximately 1.01% -/
theorem shop_owner_profit : 
  ∃ ε > 0, abs (percentage_profit - 101/100) < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l958_95897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_team_braiding_time_l958_95887

/-- The time it takes to braid all dancers' hair -/
noncomputable def total_braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) (prep_time_per_dancer : ℝ) : ℝ :=
  let minutes_per_braid : ℝ := (seconds_per_braid : ℝ) / 60
  let braiding_time_per_dancer : ℝ := minutes_per_braid * (braids_per_dancer : ℝ)
  let total_time_per_dancer : ℝ := braiding_time_per_dancer + prep_time_per_dancer
  (num_dancers : ℝ) * total_time_per_dancer

/-- Theorem stating that the total braiding time for the given conditions is 187.5 minutes -/
theorem dance_team_braiding_time :
  total_braiding_time 15 10 45 5 = 187.5 := by
  -- Unfold the definition of total_braiding_time
  unfold total_braiding_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_team_braiding_time_l958_95887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l958_95804

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3) + 1

-- State the theorem
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l958_95804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l958_95899

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*a*x + 1 < 0) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l958_95899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_distance_center_to_line_l958_95862

-- Define the line
def line (x y : ℝ) : Prop := x - y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (p : ℝ × ℝ), 
    line p.1 p.2 ∧ 
    circle_eq p.1 p.2 ∧
    ∀ (q : ℝ × ℝ), line q.1 q.2 → circle_eq q.1 q.2 → q = p :=
by
  sorry

-- Additional theorem to show that the distance from the center to the line is sqrt(2)
theorem distance_center_to_line :
  let d := |(-1 - 1) / Real.sqrt 2|
  d = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_distance_center_to_line_l958_95862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l958_95851

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 5 ∧ t.b = 7 ∧
  t.α = 40 * Real.pi / 180 ∧
  t.β - t.α = t.γ - t.β ∧
  t.α < t.β ∧ t.β < t.γ

-- Theorem statement
theorem triangle_side_value (t : Triangle) 
  (h : triangle_conditions t) : t.c = Real.sqrt 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l958_95851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marla_driving_time_l958_95830

/-- Represents the time Marla spends driving one way to her son's school -/
def driving_time : ℕ := sorry

/-- The total time Marla spends on the errand -/
def total_time : ℕ := 110

/-- The time Marla spends at parent-teacher night -/
def parent_teacher_time : ℕ := 70

/-- Theorem stating that Marla spends 20 minutes driving one way to her son's school -/
theorem marla_driving_time : driving_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marla_driving_time_l958_95830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shape_l958_95838

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the shape
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (floor p.1)^2 + (floor p.2)^2 = 50}

-- State the theorem
theorem area_of_shape : MeasureTheory.volume S = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shape_l958_95838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_winnings_l958_95859

theorem lottery_winnings (W : ℝ) : 
  W > 0 → 
  (let remaining_after_tax := W / 2;
   let remaining_after_loans := remaining_after_tax - (remaining_after_tax / 3);
   let remaining_after_savings := remaining_after_loans - 1000;
   let invested := 1000 / 5;
   remaining_after_savings - invested = 2802) → 
  W = 12006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_winnings_l958_95859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_4pi_l958_95841

/-- A rectangle with vertices (2, 9), (13, 9), (13, -4), and (2, -4) -/
def rectangle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 ≤ p.1 ∧ p.1 ≤ 13 ∧ -4 ≤ p.2 ∧ p.2 ≤ 9}

/-- A circle described by the equation (x - 2)^2 + (y + 4)^2 = 16 -/
def circleSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 4)^2 = 16}

/-- The area of intersection between the rectangle and the circle -/
noncomputable def intersectionArea : ℝ := sorry

theorem intersection_area_is_4pi :
  intersectionArea = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_4pi_l958_95841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l958_95875

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then (x + 1)^2 else 2*x + 2

-- State the theorem
theorem f_greater_than_one_range :
  {x : ℝ | f x > 1} = {x : ℝ | x < -2 ∨ x > -1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l958_95875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l958_95817

/-- The length of a platform given train acceleration and other parameters -/
theorem platform_length (v₀ t_platform t_man a : Real) :
  v₀ = 54 * 1000 / 3600 →  -- Initial speed in m/s
  t_platform = 22 →        -- Time to pass platform in seconds
  t_man = 20 →             -- Time to pass man in seconds
  let L_t := v₀ * t_man + (1/2) * a * t_man^2  -- Length of train
  let L_p := v₀ * t_platform + (1/2) * a * t_platform^2 - L_t  -- Length of platform
  L_p = 30 + 42 * a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l958_95817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_implies_x_value_l958_95891

theorem dot_product_zero_implies_x_value (a b : Fin 3 → ℝ) (x : ℝ) :
  a = ![2, -3, 1] →
  b = ![1, x, 4] →
  (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2) = 0 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_implies_x_value_l958_95891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_planes_from_perp_lines_perp_lines_from_perp_planes_l958_95842

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)

-- Define a membership relation for lines and planes
variable (inPlane : Line → Plane → Prop)

-- Define the premises
variable (α β : Plane)
variable (m n : Line)
variable (h_diff_planes : α ≠ β)
variable (h_diff_lines : m ≠ n)
variable (h_m_not_in_planes : ¬inPlane m α ∧ ¬inPlane m β)
variable (h_n_not_in_planes : ¬inPlane n α ∧ ¬inPlane n β)

-- Theorem 1: If m ⊥ n, n ⊥ β, and m ⊥ α, then α ⊥ β
theorem perp_planes_from_perp_lines 
  (h1 : perpLine m n) 
  (h2 : perpLinePlane n β) 
  (h3 : perpLinePlane m α) : 
  perp α β := by sorry

-- Theorem 2: If α ⊥ β, n ⊥ β, and m ⊥ α, then m ⊥ n
theorem perp_lines_from_perp_planes 
  (h1 : perp α β) 
  (h2 : perpLinePlane n β) 
  (h3 : perpLinePlane m α) : 
  perpLine m n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_planes_from_perp_lines_perp_lines_from_perp_planes_l958_95842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_product_sixth_root_l958_95812

theorem power_of_three_product_sixth_root : (3^12 * 3^18 : ℚ)^(1/6) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_product_sixth_root_l958_95812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_with_coprime_pair_no_smaller_n_guarantees_coprime_pair_l958_95870

def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1002}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem min_subset_size_with_coprime_pair :
  ∀ n : ℕ, n ≥ 502 →
    ∀ S : Finset ℕ, (↑S : Set ℕ) ⊆ A → S.card = n →
      ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ coprime a b :=
by sorry

theorem no_smaller_n_guarantees_coprime_pair :
  ∀ n : ℕ, n < 502 →
    ∃ S : Finset ℕ, (↑S : Set ℕ) ⊆ A ∧ S.card = n ∧
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(coprime a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_with_coprime_pair_no_smaller_n_guarantees_coprime_pair_l958_95870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l958_95884

noncomputable def f (a b x : ℝ) : ℝ := x^2 + a*x + b

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -2 then a^2/4 + a + 2
  else if a ≤ 2 then 1
  else a^2/4 - a + 2

theorem min_value_and_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a (a^2/4 + 1) x ≥ g a) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a b x = 0) →
  (0 ≤ b - 2*a ∧ b - 2*a ≤ 1) →
  b ∈ Set.Icc (-3 : ℝ) (9 - 4*Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l958_95884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_age_is_32_l958_95809

-- Define the present ages of the father and two sons
def father_age : ℕ → ℕ := sorry
def son1_age : ℕ → ℕ := sorry
def son2_age : ℕ → ℕ := sorry

-- Current year
def current_year : ℕ := sorry

-- Axioms based on the given conditions
axiom average_age : (father_age current_year + son1_age current_year + son2_age current_year) / 3 = 24

axiom sons_average_5_years_ago : 
  (son1_age (current_year - 5) + son2_age (current_year - 5)) / 2 = 15

axiom age_difference : son1_age current_year - son2_age current_year = 4

-- Theorem to prove
theorem father_age_is_32 : father_age current_year = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_age_is_32_l958_95809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l958_95846

/-- The number of pieces in a pack of gum -/
def x : ℚ := 115 / 6

/-- The number of pieces of cherry gum Chewbacca has -/
def cherry_gum : ℕ := 25

/-- The number of pieces of grape gum Chewbacca has -/
def grape_gum : ℕ := 35

/-- The equation that represents the equality of ratios -/
def gum_equation : Prop :=
  (cherry_gum - x) / grape_gum = cherry_gum / (grape_gum + 6 * x)

theorem gum_pack_size :
  gum_equation ∧ x > 0 → x = 115 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l958_95846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l958_95826

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle with width 2 and height 1 -/
def Rectangle := {p : Point | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

theorem smallest_distance_bound (points : Finset Point) 
    (h1 : points.card = 5)
    (h2 : ∀ p, p ∈ points → p ∈ Rectangle) :
    (∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 / 2) ∧
    ∀ b < Real.sqrt 5 / 2, ∃ pts : Finset Point, 
      pts.card = 5 ∧ 
      (∀ p, p ∈ pts → p ∈ Rectangle) ∧
      (∀ p1 p2, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → distance p1 p2 > b) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l958_95826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_function_iff_collinear_ln_restricted_is_cauchy_reciprocal_sum_not_cauchy_sqrt_2x_squared_plus_8_not_cauchy_sqrt_2x_squared_minus_8_is_cauchy_l958_95871

/-- A function f: ℝ → ℝ is a Cauchy function if there exist two different points
    on its graph that form a line passing through the origin. -/
def is_cauchy_function (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    let y₁ := f x₁
    let y₂ := f x₂
    (x₁ * x₂ + y₁ * y₂)^2 = (x₁^2 + y₁^2) * (x₂^2 + y₂^2)

theorem cauchy_function_iff_collinear (f : ℝ → ℝ) :
  is_cauchy_function f ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    let y₁ := f x₁
    let y₂ := f x₂
    ∃ (k : ℝ), k ≠ 0 ∧ x₂ = k * x₁ ∧ y₂ = k * y₁ :=
sorry

-- Example functions
noncomputable def ln_restricted (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 3 then Real.log x else 0

noncomputable def reciprocal_sum (x : ℝ) : ℝ := 
  if x > 0 then x + 1/x else 0

noncomputable def sqrt_2x_squared_plus_8 (x : ℝ) : ℝ := Real.sqrt (2 * x^2 + 8)

noncomputable def sqrt_2x_squared_minus_8 (x : ℝ) : ℝ := Real.sqrt (2 * x^2 - 8)

-- Theorems for each function
theorem ln_restricted_is_cauchy : is_cauchy_function ln_restricted := by
  sorry

theorem reciprocal_sum_not_cauchy : ¬ is_cauchy_function reciprocal_sum := by
  sorry

theorem sqrt_2x_squared_plus_8_not_cauchy : ¬ is_cauchy_function sqrt_2x_squared_plus_8 := by
  sorry

theorem sqrt_2x_squared_minus_8_is_cauchy : is_cauchy_function sqrt_2x_squared_minus_8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_function_iff_collinear_ln_restricted_is_cauchy_reciprocal_sum_not_cauchy_sqrt_2x_squared_plus_8_not_cauchy_sqrt_2x_squared_minus_8_is_cauchy_l958_95871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_12_value_l958_95815

def f (m : ℕ) : ℕ :=
  if m % 2 = 0 ∧ m > 0 then
    (List.range (m / 2)).foldl (fun acc i => acc * (2 * (i + 1))) 1
  else
    0

theorem f_12_value : f 12 = 46080 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if condition
  simp
  -- Evaluate the foldl expression
  norm_num
  -- The proof is complete
  rfl

#eval f 12  -- This will evaluate f 12 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_12_value_l958_95815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_asymptote_with_focus_as_center_l958_95873

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := 4*x - 3*y = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (5, 0)

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 9 = 0

-- Theorem statement
theorem circle_tangent_to_asymptote_with_focus_as_center :
  ∃ (x₀ y₀ : ℝ), 
    (hyperbola x₀ y₀) ∧ 
    (asymptote x₀ y₀) ∧
    (circle_equation x₀ y₀) ∧
    (∀ (x y : ℝ), circle_equation x y → (x - right_focus.1)^2 + (y - right_focus.2)^2 = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_asymptote_with_focus_as_center_l958_95873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_prism_volume_l958_95823

/-- The volume of a regular quadrangular prism -/
noncomputable def prism_volume (a : ℝ) (α : ℝ) : ℝ :=
  (a^3 * Real.sqrt (Real.cos (2 * α))) / Real.sin α

/-- Theorem: The volume of a regular quadrangular prism with base side length a and angle α
    between the prism's diagonal and a side face is equal to (a³ * √(cos(2α))) / sin(α) -/
theorem regular_quadrangular_prism_volume 
  (a : ℝ) 
  (α : ℝ) 
  (h1 : a > 0) 
  (h2 : 0 < α ∧ α < π / 2) : 
  prism_volume a α = (a^3 * Real.sqrt (Real.cos (2 * α))) / Real.sin α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_prism_volume_l958_95823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_chord_extensions_parallel_l958_95820

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
noncomputable def intersectionPoints (c1 c2 : Circle) : Point × Point := sorry

noncomputable def chordExtension (c : Circle) (p1 p2 : Point) : Point := sorry

noncomputable def lineFromPoints (p1 p2 : Point) : Line := sorry

-- Define a function to check if a point is on a circle
def onCircle (p : Point) (c : Circle) : Prop := 
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define parallelism for lines
def parallel (l1 l2 : Line) : Prop := 
  l1.a * l2.b = l1.b * l2.a

-- Theorem statement
theorem circles_chord_extensions_parallel 
  (c1 c2 : Circle) 
  (A B C D E F : Point) :
  let (A', B') := intersectionPoints c1 c2
  let E := chordExtension c2 A C
  let F := chordExtension c2 B D
  let CD := lineFromPoints C D
  let EF := lineFromPoints E F
  A = A' → 
  B = B' → 
  onCircle C c1 → 
  onCircle D c1 → 
  onCircle E c2 → 
  onCircle F c2 → 
  parallel CD EF :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_chord_extensions_parallel_l958_95820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_properties_l958_95865

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b

def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

theorem fibonacci_properties :
  (∃ m : ℕ, is_arithmetic_progression (fibonacci m) (fibonacci (m + 1)) (fibonacci (m + 2))) ∧
  (¬ ∃ m : ℕ, is_geometric_progression (fibonacci m) (fibonacci (m + 1)) (fibonacci (m + 2))) ∧
  (∃ t : ℚ, ∀ n : ℕ, is_arithmetic_progression (fibonacci n) (t * (fibonacci (n + 2))) (fibonacci (n + 4))) ∧
  (∃ s : List ℕ, s.length > 0 ∧ List.Sorted (· < ·) s ∧ (s.map fibonacci).sum = 2023) := by
  sorry

#eval fibonacci 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_properties_l958_95865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_gpa_at_least_3_25_l958_95885

/-- Grade points for each letter grade -/
def gradePoints : Char → ℕ
| 'A' => 4
| 'B' => 3
| 'C' => 2
| 'D' => 1
| _   => 0

/-- Calculate GPA from a list of grades -/
def calculateGPA (grades : List Char) : ℚ :=
  (grades.map gradePoints).sum / grades.length

/-- Probability of getting an A in Literature -/
def litProbA : ℚ := 1/4

/-- Probability of getting a B in Literature -/
def litProbB : ℚ := 1/5

/-- Probability of getting an A in Art -/
def artProbA : ℚ := 1/3

/-- Probability of getting a B in Art -/
def artProbB : ℚ := 1/4

/-- Theorem: Probability of achieving a GPA of at least 3.25 -/
theorem prob_gpa_at_least_3_25 : 
  let physicsGrade := 'A'
  let chemistryGrade := 'A'
  ∃ p : ℚ, p = 91/240 ∧ 
    (∀ litGrade artGrade : Char, 
      calculateGPA [physicsGrade, chemistryGrade, litGrade, artGrade] ≥ 3.25 →
      p = litProbA * artProbA + 
          litProbA * artProbB + 
          litProbB * artProbA) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_gpa_at_least_3_25_l958_95885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_k_l958_95894

/-- A triangle in the first quadrant formed by the x-axis, y-axis, and a line --/
structure Triangle where
  k : ℝ
  t : ℝ

/-- The area of the triangle --/
noncomputable def Triangle.area (tri : Triangle) : ℝ :=
  (tri.t^2) / (4 * (tri.k^2 - 1))

/-- Theorem: If the area of the triangle is 10 and t = 5, then k = 3/2 --/
theorem triangle_area_implies_k (tri : Triangle) 
  (h1 : tri.area = 10) 
  (h2 : tri.t = 5) : 
  tri.k = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_k_l958_95894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_l958_95854

noncomputable def f (a b x : ℝ) : ℝ := (x + a) / (x^2 + b*x + 1)

theorem odd_function_monotonicity (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is an odd function
  (a = 0 ∧ b = 0) ∧               -- Part 1: a = 0 and b = 0
  (∀ x y, 1 < x → x < y → f 0 0 x > f 0 0 y) -- Part 2: f is strictly decreasing on (1, +∞)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_l958_95854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l958_95806

/-- Calculates the balance of an account with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate / 2) ^ (periods * 2)

/-- Calculates the balance of an account with simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate * years)

/-- The positive difference between compound and simple interest accounts -/
theorem interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (years : ℕ) :
  principal > 0 →
  compound_rate > 0 →
  simple_rate > 0 →
  years > 0 →
  |compound_interest principal compound_rate years - simple_interest principal simple_rate years - 121| < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l958_95806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_gcd_lcm_set_exists_l958_95895

theorem distinct_gcd_lcm_set_exists : ∃ (C : Finset ℕ+), 
  (Finset.card C = 2020) ∧ 
  (∀ a b : ℕ+, a ∈ C → b ∈ C → a ≠ b → 
    ∀ c d : ℕ+, c ∈ C → d ∈ C → c ≠ d → (a ≠ c ∨ b ≠ d) → 
      (Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val ∧ 
       Nat.lcm a.val b.val ≠ Nat.lcm c.val d.val)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_gcd_lcm_set_exists_l958_95895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_tangent_to_circle_l958_95801

/-- The value of m for which the asymptotes of the hyperbola y² - x²/m² = 1 (m > 0)
    are tangent to the circle x² + y² - 4y + 3 = 0 -/
theorem hyperbola_asymptotes_tangent_to_circle :
  ∀ m : ℝ,
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2/m^2 = 1 → 
    ∃ k : ℝ, (y = m*x ∨ y = -m*x) ∧
    (k*x - y = 0 → |k*0 - 1*2| / Real.sqrt (k^2 + 1) = 1)) →
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → (x - 0)^2 + (y - 2)^2 = 1) →
  m = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_tangent_to_circle_l958_95801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l958_95857

theorem range_of_a (a : ℝ) : 
  (∀ α ∈ Set.Icc (π/6) (2*π/3), ∃ β ∈ Set.Icc (π/6) (2*π/3), Real.cos α ≥ Real.sin β + a) → 
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l958_95857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l958_95840

noncomputable def A : ℝ × ℝ := (2, 3)
noncomputable def B : ℝ × ℝ := (3, 8)

def line_equation (x : ℝ) : ℝ := 2 * x

def point_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line_equation p.1

def intersect_at_D (A' B' D : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    D = (A.1 + t₁ * (A'.1 - A.1), A.2 + t₁ * (A'.2 - A.2)) ∧
    D = (B.1 + t₂ * (B'.1 - B.1), B.2 + t₂ * (B'.2 - B.2))

def equal_angle_with_x_axis (A' B' D : ℝ × ℝ) : Prop :=
  (A'.2 - D.2) / (A'.1 - D.1) = -(B'.2 - D.2) / (B'.1 - D.1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem length_of_A'B' (A' B' D : ℝ × ℝ) :
  point_on_line A' ∧ 
  point_on_line B' ∧ 
  intersect_at_D A' B' D ∧
  equal_angle_with_x_axis A' B' D →
  distance A' B' = 5.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l958_95840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_laps_per_minute_approx_three_l958_95828

/-- The number of laps Johnny ran -/
noncomputable def total_laps : ℝ := 10

/-- The time it took Johnny to run the laps (in minutes) -/
noncomputable def total_time : ℝ := 3.33333

/-- The number of laps Johnny ran per minute -/
noncomputable def laps_per_minute : ℝ := total_laps / total_time

/-- Theorem stating that the number of laps per minute is approximately 3 -/
theorem johnny_laps_per_minute_approx_three : 
  ∃ ε > 0, |laps_per_minute - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_laps_per_minute_approx_three_l958_95828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l958_95813

/-- The function f(x) defined as sin x + cos(x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (x + Real.pi/6)

/-- Theorem stating that the range of f(x) is [-1, 1] -/
theorem f_range : ∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l958_95813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l958_95853

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- State the theorem
theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 Real.pi → f y ≥ f x) ∧
  f x = 5 * Real.pi / 6 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l958_95853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_l958_95847

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The slope of the line -/
  k : ℝ
  /-- The line equation: kx - y + 2 = 0 -/
  line : ℝ → ℝ → Prop := λ x y ↦ k * x - y + 2 = 0
  /-- The circle equation: (x - 1)^2 + y^2 = 9 -/
  circle : ℝ → ℝ → Prop := λ x y ↦ (x - 1)^2 + y^2 = 9
  /-- The line intersects the circle at two points -/
  intersects : ∃ A B : ℝ × ℝ, A ≠ B ∧ line A.1 A.2 ∧ line B.1 B.2 ∧ circle A.1 A.2 ∧ circle B.1 B.2

/-- The length of the chord for a given slope k -/
noncomputable def chord_length (k : ℝ) : ℝ := sorry

/-- The theorem stating that the chord is shortest when k = 1/2 -/
theorem shortest_chord (lci : LineCircleIntersection) :
  (∀ k' : ℝ, k' ≠ lci.k → chord_length lci.k ≤ chord_length k') →
  lci.k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_l958_95847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l958_95819

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + 2 * (Real.cos (x / 2))^2

theorem f_range : Set.range f = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l958_95819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l958_95834

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (2 : ℝ)^(x^2 - 5*x + 2) = (8 : ℝ)^(x - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l958_95834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l958_95880

def sequence_a (A : ℕ) : ℕ → ℕ
  | 0 => A^A  -- Added case for 0
  | 1 => A^A
  | n+1 => A^(sequence_a A n)

def sequence_b (A : ℕ) : ℕ → ℕ
  | 0 => A^(A+1)  -- Added case for 0
  | 1 => A^(A+1)
  | n+1 => 2^(sequence_b A n)

theorem a_less_than_b (A : ℕ) (h : A > 1) :
  ∀ n : ℕ, n ≥ 1 → sequence_a A n < sequence_b A n :=
by
  intro n hn
  sorry  -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l958_95880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l958_95874

noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_image :
  let center : ℂ := 1 + 2*Complex.I
  let scale : ℝ := 4
  let z : ℂ := -2 - 2*Complex.I
  dilation center scale z = -11 - 14*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l958_95874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l958_95827

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2 - x)) / x

theorem domain_of_f :
  {x : ℝ | x < 0 ∨ (0 < x ∧ x ≤ 2)} = {x : ℝ | f x ∈ Set.univ} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l958_95827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l958_95836

theorem equidistant_point (x y : ℝ) : 
  (abs y = abs x) ∧ 
  (abs y = abs (2*x + y - 4) / Real.sqrt 5) → 
  x = 4/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l958_95836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_real_root_iff_odd_l958_95892

/-- The sum of geometric series from 0 to n -/
noncomputable def geometricSum (x : ℝ) (n : ℕ) : ℝ :=
  (x^(n+1) - 1) / (x - 1)

/-- The polynomial P(X) = X^n + X^{n-1} + ... + 1 -/
noncomputable def P (n : ℕ) (x : ℝ) : ℝ :=
  geometricSum x n

theorem P_real_root_iff_odd (n : ℕ) (hn : n > 0) :
  (∃ x : ℝ, P n x = 0) ↔ Odd n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_real_root_iff_odd_l958_95892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_neg_thirteen_l958_95843

theorem det_B_equals_neg_thirteen (x y : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B - B⁻¹ = 1) → Matrix.det B = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_neg_thirteen_l958_95843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l958_95807

-- Define the sequence T
def T : ℕ → ℕ
| 0 => 6  -- Add this case for 0
| 1 => 6
| n + 2 => 6^(T (n + 1))

-- State the theorem
theorem t_100_mod_7 : T 100 ≡ 1 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l958_95807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_side_bisector_intersection_l958_95886

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- An angle bisector in a quadrilateral -/
noncomputable def AngleBisector (q : ConvexQuadrilateral) (i : Fin 4) : Set (ℝ × ℝ) :=
  sorry

/-- A side of a quadrilateral -/
def Side (q : ConvexQuadrilateral) (i : Fin 4) : Set (ℝ × ℝ) :=
  sorry

/-- The statement that no side intersects an angle bisector except at vertices -/
theorem no_side_bisector_intersection (q : ConvexQuadrilateral) :
  ∀ (i j : Fin 4), i ≠ j →
    (Side q i ∩ AngleBisector q j) ⊆ Set.range q.vertices :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_side_bisector_intersection_l958_95886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_most_8_l958_95863

-- Define the taxi fare structure
structure TaxiFare where
  base_fare : ℝ
  base_distance : ℝ
  additional_fare : ℝ
  total_fare : ℝ

-- Define the problem conditions
def taxi_problem (fare : TaxiFare) (distance : ℝ) : Prop :=
  fare.base_fare = 7 ∧
  fare.base_distance = 3 ∧
  fare.additional_fare = 2.4 ∧
  fare.total_fare = 19 ∧
  distance ≥ 0 ∧
  (if distance ≤ fare.base_distance
   then fare.total_fare = fare.base_fare
   else fare.total_fare = fare.base_fare + (⌈distance - fare.base_distance⌉) * fare.additional_fare)

-- Theorem to prove
theorem distance_at_most_8 (fare : TaxiFare) (distance : ℝ) :
  taxi_problem fare distance → distance ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_most_8_l958_95863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_approx_11_percent_l958_95882

/-- Calculates the second discount percentage given the original price, 
    first discount percentage, and final price -/
noncomputable def calculate_second_discount (original_price first_discount final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount / 100)
  100 * (1 - final_price / price_after_first_discount)

/-- Theorem stating that given the conditions in the problem, 
    the second discount is approximately 11% -/
theorem second_discount_approx_11_percent 
  (original_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (h1 : original_price = 70)
  (h2 : first_discount = 10)
  (h3 : final_price = 56.16) :
  ∃ ε > 0, |calculate_second_discount original_price first_discount final_price - 11| < ε := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_second_discount 70 10 56.16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_approx_11_percent_l958_95882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l958_95849

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 4) / Real.log (1/2)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x : ℝ, x ≥ 3/4 → (∀ y : ℝ, y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l958_95849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tetrahedron_l958_95808

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedron_volume (ab ac bc bd ad cd : ℝ) : ℝ :=
  let B : Matrix (Fin 5) (Fin 5) ℝ := ![
    ![0, 1, 1, 1, 1],
    ![1, 0, ab^2, ac^2, ad^2],
    ![1, ab^2, 0, bc^2, bd^2],
    ![1, ac^2, bc^2, 0, cd^2],
    ![1, ad^2, bd^2, cd^2, 0]
  ]
  (1/6) * Real.sqrt (- B.det)

/-- Theorem: The volume of tetrahedron ABCD with given edge lengths is 20 -/
theorem volume_of_specific_tetrahedron :
  tetrahedron_volume 6 4 5 5 4 3 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tetrahedron_l958_95808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_dollar_price_with_tax_twenty_one_is_smallest_l958_95896

theorem smallest_whole_dollar_price_with_tax :
  ∀ m : ℕ, m < 21 →
    ¬∃ x : ℕ, (105 * x) % 100 = 0 ∧
      ∃ y : ℕ, y = m ∧ 105 * x = 100 * y :=
by sorry

theorem twenty_one_is_smallest :
  ∃ x : ℕ, (105 * x) % 100 = 0 ∧
    ∃ y : ℕ, y = 21 ∧ 105 * x = 100 * y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_dollar_price_with_tax_twenty_one_is_smallest_l958_95896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l958_95811

def M : Set (ℝ × ℝ) := {a | ∃ lambda1 : ℝ, a = (1, 2) + lambda1 • (3, 4)}
def N : Set (ℝ × ℝ) := {b | ∃ lambda2 : ℝ, b = (-2, -2) + lambda2 • (4, 5)}

theorem intersection_of_M_and_N : M ∩ N = {(-2, -2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l958_95811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_pi_over_six_l958_95810

/-- The angle corresponding to a given slope -/
noncomputable def angle_from_slope (s : ℝ) : ℝ := Real.arctan s

theorem line_slope_is_pi_over_six :
  angle_from_slope ((Real.sqrt 3) / 3) = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_pi_over_six_l958_95810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_skew_lines_l958_95837

/-- The volume of a tetrahedron formed by two segments sliding on skew lines -/
theorem tetrahedron_volume_skew_lines 
  (a b d : ℝ) (φ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hd : d > 0) 
  (hφ : 0 < φ ∧ φ < π) : 
  ∃ (V : ℝ), V = (1/6) * a * b * d * Real.sin φ ∧ 
  ∀ (pos : ℝ × ℝ), V = (1/6) * a * b * d * Real.sin φ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_skew_lines_l958_95837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_cross_section_eccentricity_l958_95893

/-- The eccentricity of an elliptical cross-section formed by a plane intersecting a cylinder at a 45° angle with its base is √2/2. -/
theorem elliptical_cross_section_eccentricity :
  ∀ (R : ℝ), R > 0 →
  let cylinder := {(x, y, z) : ℝ × ℝ × ℝ | x^2 + y^2 = R^2}
  let plane := {(x, y, z) : ℝ × ℝ × ℝ | x + z = 0}
  let intersection := {p : ℝ × ℝ × ℝ | p ∈ cylinder ∩ plane}
  let major_axis := Real.sqrt 2 * R
  let minor_axis := R
  let eccentricity := Real.sqrt (1 - (minor_axis / major_axis)^2)
  eccentricity = Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_cross_section_eccentricity_l958_95893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_border_black_edges_l958_95864

/-- Represents a 5x5 grid where each 1x1 square has exactly 3 black edges -/
def Grid := Fin 5 → Fin 5 → Fin 4 → Bool

/-- The condition that adjacent squares share edge colors -/
def adjacent_consistent (g : Grid) : Prop :=
  ∀ i j k, i < 4 → g i j k = g (i + 1) j k ∧ 
           j < 4 → g i j k = g i (j + 1) k

/-- The number of black edges for a given square -/
def black_edges (g : Grid) (i j : Fin 5) : Nat :=
  (g i j 0).toNat + (g i j 1).toNat + (g i j 2).toNat + (g i j 3).toNat

/-- The condition that each square has exactly 3 black edges -/
def three_black_edges (g : Grid) : Prop :=
  ∀ i j, black_edges g i j = 3

/-- The number of black edges on the outer border of the grid -/
def border_black_edges (g : Grid) : Nat :=
  (g 0 0 0).toNat + (g 0 0 3).toNat + 
  (g 0 4 0).toNat + (g 0 4 1).toNat + 
  (g 4 0 2).toNat + (g 4 0 3).toNat + 
  (g 4 4 1).toNat + (g 4 4 2).toNat + 
  (Finset.sum (Finset.range 5) (fun j => (g 0 j 0).toNat)) + 
  (Finset.sum (Finset.range 5) (fun j => (g 4 j 2).toNat)) + 
  (Finset.sum (Finset.range 5) (fun i => (g i 0 3).toNat)) + 
  (Finset.sum (Finset.range 5) (fun i => (g i 4 1).toNat))

/-- The main theorem -/
theorem min_border_black_edges (g : Grid) 
  (h1 : adjacent_consistent g) 
  (h2 : three_black_edges g) : 
  border_black_edges g ≥ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_border_black_edges_l958_95864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_l958_95867

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_13 (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 5 + 2 * arithmetic_sequence a₁ d 10 = 4 →
  arithmetic_sum a₁ d 13 = 13 := by
  sorry

#check arithmetic_sequence_sum_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_l958_95867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sakshi_tanya_work_efficiency_l958_95824

theorem sakshi_tanya_work_efficiency (sakshi_days : ℕ) (tanya_efficiency : ℝ) :
  sakshi_days = 12 →
  tanya_efficiency = 1.2 →
  (sakshi_days : ℝ) / tanya_efficiency = 10 := by
  intros h_sakshi h_tanya
  rw [h_sakshi, h_tanya]
  norm_num

#check sakshi_tanya_work_efficiency

-- Example usage
example : (12 : ℝ) / 1.2 = 10 :=
  sakshi_tanya_work_efficiency 12 1.2 rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sakshi_tanya_work_efficiency_l958_95824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_price_problem_l958_95889

/-- Proves that the price of each towel bought second is 40 rupees given the problem conditions --/
theorem towel_price_problem : ∃ x : ℚ, x = 40 := by
  -- Define the number of towels in each group
  let n₁ : ℕ := 3
  let n₂ : ℕ := 5
  let n₃ : ℕ := 2

  -- Define the prices for the first and third groups
  let p₁ : ℚ := 100
  let p₃ : ℚ := 550

  -- Define the average price
  let avg_price : ℚ := 160

  -- Total number of towels
  let total_towels : ℕ := n₁ + n₂ + n₃

  -- Define x as the price of each towel bought second
  let x : ℚ := 40

  -- Equation for the average price
  have avg_price_eq : (n₁ * p₁ + n₂ * x + n₃ * p₃) / total_towels = avg_price := by
    -- Proof of the equation
    sorry

  -- Prove that x = 40 satisfies the equation
  have x_satisfies_eq : x = 40 := by
    -- Proof that x = 40
    sorry

  -- Conclude that there exists an x that satisfies the conditions
  exact ⟨x, x_satisfies_eq⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_price_problem_l958_95889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toaster_popularity_invariant_l958_95822

/-- A function representing the popularity of a toaster based on cost and age -/
noncomputable def popularity (k : ℝ) (c a : ℝ) : ℝ := k * a / c

/-- Theorem stating that if popularity is 16 for a $400 toaster that is 2 years old,
    then it's also 16 for an $800 toaster that is 4 years old -/
theorem toaster_popularity_invariant (k : ℝ) :
  popularity k 400 2 = 16 → popularity k 800 4 = 16 := by
  intro h
  -- Proof steps would go here, but we'll use sorry for now
  sorry

#check toaster_popularity_invariant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toaster_popularity_invariant_l958_95822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_emptying_time_is_six_hours_l958_95868

/-- Represents the tank system with a leak and an inlet pipe -/
structure TankSystem where
  capacity : ℝ
  inletRate : ℝ
  emptyingTimeWithInlet : ℝ

/-- Calculates the time taken for the leak to empty the tank when the inlet is closed -/
noncomputable def leakEmptyingTime (system : TankSystem) : ℝ :=
  let netEmptyingRate := system.capacity / (system.emptyingTimeWithInlet * 60)
  let leakRate := netEmptyingRate + system.inletRate
  system.capacity / leakRate / 60

/-- Theorem stating that for the given system, the leak empties the tank in 6 hours -/
theorem leak_emptying_time_is_six_hours (system : TankSystem) 
  (h1 : system.capacity = 5760)
  (h2 : system.inletRate = 4)
  (h3 : system.emptyingTimeWithInlet = 8) :
  leakEmptyingTime system = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_emptying_time_is_six_hours_l958_95868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_64_l958_95821

theorem cube_root_of_sqrt_64 : (64 : ℝ).sqrt ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_64_l958_95821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_without_average_between_l958_95803

theorem permutation_without_average_between (n : ℕ+) : ∃ σ : Fin n → Fin n, Function.Bijective σ ∧
  ∀ i j k : Fin n, i < k ∧ k < j → σ k ≠ (σ i + σ j) / (2 : Fin n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_without_average_between_l958_95803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_theorem_l958_95848

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The radius of the circle in feet -/
def r : ℝ := 50

/-- The distance traveled by one point to visit all non-adjacent points -/
noncomputable def distance_one_point : ℝ := 200 + 100 * Real.sqrt (2 - Real.sqrt 2)

/-- The total distance traveled by all points -/
noncomputable def total_distance : ℝ := n * distance_one_point

theorem circle_distance_theorem :
  total_distance = 1600 + 800 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_theorem_l958_95848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_a_min_value_of_c_l958_95818

-- Define the function f
noncomputable def f (a c x : ℝ) : ℝ := a * Real.log x + (x - c) * abs (x - c)

-- Helper function for the derivative of f
noncomputable def f' (a c x : ℝ) : ℝ :=
  if x ≥ c then (2*x^2 - 2*c*x + a) / x
  else (-2*x^2 + 2*c*x + a) / x

-- Theorem 1
theorem monotonicity_intervals (x : ℝ) (hx : x > 0) :
  let a : ℝ := -3/4
  let c : ℝ := 1/4
  (∀ y ∈ Set.Ioo 0 (3/4), HasDerivAt (f a c) (f' a c y) y ∧ f' a c y < 0) ∧
  (∀ y ∈ Set.Ioi (3/4), HasDerivAt (f a c) (f' a c y) y ∧ f' a c y > 0) :=
sorry

-- Theorem 2
theorem range_of_a (a : ℝ) :
  (∀ x > (a/2 + 1), f a (a/2 + 1) x ≥ 1/4) →
  a ∈ Set.Ioc (-2) (-1) :=
sorry

-- Theorem 3
theorem min_value_of_c (a c : ℝ) (ha : a < 0) (hc : c > 0) :
  let x₁ : ℝ := Real.sqrt (-a/2)
  let x₂ : ℝ := c
  (HasDerivAt (f a c) (f' a c x₁) x₁) ∧
  (HasDerivAt (f a c) (f' a c x₂) x₂) ∧
  (f' a c x₁ * f' a c x₂ = -1) →
  c ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_a_min_value_of_c_l958_95818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l958_95802

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the fixed point P
def P : ℝ × ℝ := (2, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the theorem
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 4 ∧
  ∀ (M : ℝ × ℝ), parabola M.1 M.2 →
    distance M P + distance M focus ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l958_95802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_1800_l958_95844

/-- Calculates the total profit given the investment conditions and third person's profit share -/
def total_profit_calculation (total_investment : ℕ) (second_extra : ℕ) (third_extra : ℕ) (third_profit : ℕ) : ℕ :=
  let first_investment := (total_investment - 3 * second_extra) / 3
  let second_investment := first_investment + second_extra
  let third_investment := second_investment + third_extra
  let profit_ratio := first_investment + second_investment + third_investment
  let profit_per_unit := third_profit / third_investment
  profit_per_unit * profit_ratio

/-- Proves that the total profit is 1800 given the specified conditions -/
theorem total_profit_is_1800 :
  total_profit_calculation 9000 1000 1000 800 = 1800 := by
  -- Proof goes here
  sorry

#eval total_profit_calculation 9000 1000 1000 800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_1800_l958_95844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_main_theorem_l958_95878

-- Define the slopes of the lines
noncomputable def slope_l1 : ℝ := -1/4
noncomputable def slope_l2 (k : ℝ) : ℝ := -k

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_k (k : ℝ) :
  perpendicular slope_l1 (slope_l2 k) → k = -4 :=
by
  intro h
  unfold perpendicular at h
  unfold slope_l1 slope_l2 at h
  field_simp at h
  exact h

-- Main theorem with sorry
theorem main_theorem (k : ℝ) :
  perpendicular slope_l1 (slope_l2 k) ↔ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_main_theorem_l958_95878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rafael_tuesday_hours_l958_95825

/-- Calculates the hours Rafael worked on Tuesday given his work schedule and pay -/
theorem rafael_tuesday_hours
  (monday_hours : ℕ)
  (remaining_hours : ℕ)
  (total_pay : ℕ)
  (hourly_rate : ℕ)
  (h1 : monday_hours = 10)
  (h2 : remaining_hours = 20)
  (h3 : total_pay = 760)
  (h4 : hourly_rate = 20) :
  total_pay / hourly_rate - remaining_hours - monday_hours = 8 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rafael_tuesday_hours_l958_95825
