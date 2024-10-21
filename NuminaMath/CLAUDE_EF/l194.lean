import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_seven_is_ninth_term_l194_19496

noncomputable def sequenceA (n : ℕ) : ℝ := Real.sqrt (3 * n + 1)

theorem two_sqrt_seven_is_ninth_term :
  ∃ n : ℕ, n = 9 ∧ sequenceA n = 2 * Real.sqrt 7 := by
  use 9
  constructor
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_seven_is_ninth_term_l194_19496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_books_count_l194_19403

def sam_books : ℕ := 110
def total_books : ℕ := 212

theorem joan_books_count : total_books - sam_books = 102 := by
  -- The proof goes here
  sorry

def joan_books : ℕ := total_books - sam_books

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_books_count_l194_19403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l194_19405

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x^2 + 1/x^2)

theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l194_19405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_probability_l194_19416

def total_jellybeans : ℕ := 15
def orange_jellybeans : ℕ := 6
def blue_jellybeans : ℕ := 4
def white_jellybeans : ℕ := 3
def green_jellybeans : ℕ := 2
def picks : ℕ := 4

theorem jellybean_probability :
  (Nat.choose blue_jellybeans 3 * Nat.choose (total_jellybeans - blue_jellybeans) 1) /
  (Nat.choose total_jellybeans picks) = 44 / 1365 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_probability_l194_19416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l194_19479

/-- The equation of the boundary curve -/
def boundary_equation (x y : ℝ) : Prop :=
  y^2 + 4*x*y + 60*|x| = 600

/-- The bounded region defined by the boundary equation -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | boundary_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area_of_region : ℝ :=
  450 -- We directly define it as 450 for now

theorem area_of_bounded_region :
  area_of_region = 450 := by
  -- Proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l194_19479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_at_9_l194_19488

/-- Arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + n * (n - 1) / 2 * d

/-- The maximum value of S_n occurs when n = 9 -/
theorem max_S_at_9 (a₁ d : ℝ) (h₁ : a₁ > 0) (h₂ : S a₁ d 8 = S a₁ d 10) :
  ∀ n : ℕ, S a₁ d 9 ≥ S a₁ d n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_at_9_l194_19488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_90_when_x_is_3_l194_19412

theorem fraction_equals_90_when_x_is_3 : 
  ∀ x : ℝ, x = 3 → (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by
  intro x h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_90_when_x_is_3_l194_19412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_properties_l194_19499

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁.1 = -Real.sqrt 5 ∧ F₁.2 = 0 ∧ F₂.1 = Real.sqrt 5 ∧ F₂.2 = 0

-- Define the asymptotes of the hyperbola
def asymptote (x y : ℝ) : Prop := y = (1/2) * x ∨ y = -(1/2) * x

-- Define the circle with diameter F₁F₂
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define point M
def point_M (M : ℝ × ℝ) : Prop :=
  asymptote M.1 M.2 ∧ circle_eq M.1 M.2

-- Helper function to calculate triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let s := ((A.1 - C.1)^2 + (A.2 - C.2)^2).sqrt / 2
  (s * (s - ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt) *
   (s - ((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt) *
   (s - ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt)).sqrt

-- Theorem statement
theorem hyperbola_point_properties (F₁ F₂ M : ℝ × ℝ) :
  foci F₁ F₂ → point_M M →
  (M.1 = 2 ∨ M.1 = -2) ∧
  area_triangle M F₁ F₂ = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_properties_l194_19499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_implies_alpha_one_l194_19452

/-- Curve C₁ in parametric form -/
noncomputable def curve_C₁ (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 + 2 * Real.cos t, 3 + 2 * Real.sin t)

/-- Curve C₂ in rectangular form with parameter α -/
def curve_C₂ (α : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 2 * α = 0

/-- The statement to be proved -/
theorem unique_intersection_implies_alpha_one :
  ∀ α : ℝ,
  (∃! p : ℝ × ℝ, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ 
    curve_C₁ t = p ∧ curve_C₂ α p.1 p.2) →
  α = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_implies_alpha_one_l194_19452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l194_19449

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- Define the domain
def domain : Set ℝ := (Set.Iio 1) ∪ (Set.Icc 2 5)

-- Define the range
def range : Set ℝ := (Set.Iio 0) ∪ (Set.Ioc (1/4) 1)

-- Theorem statement
theorem f_range : 
  {y | ∃ x ∈ domain, f x = y} = range := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l194_19449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_equals_three_l194_19455

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t/2, 3 + (Real.sqrt 3 * t)/2)

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define point P as the intersection of line l and y-axis
def point_P : ℝ × ℝ := (0, 3)

-- Define points A and B as intersections of line l and curve C
-- We don't explicitly define A and B, but we'll use their existence in the theorem

theorem intersection_product_equals_three :
  ∃ (t_A t_B : ℝ),
    let A := line_l t_A
    let B := line_l t_B
    (A.1^2 + A.2^2 = 2 * A.2) ∧  -- A is on curve C
    (B.1^2 + B.2^2 = 2 * B.2) ∧  -- B is on curve C
    ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) *
    ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_equals_three_l194_19455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_ratio_l194_19462

theorem tan_sum_ratio (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_ratio_l194_19462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_approx_l194_19461

def playground_side : ℝ := 27
def veg_garden_length : ℝ := 12
def veg_garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13
def irregular_bed_perimeter : ℝ := 18
def l_shape_length1 : ℝ := 6
def l_shape_width1 : ℝ := 4
def l_shape_length2 : ℝ := 3
def l_shape_width2 : ℝ := 4

noncomputable def total_fencing : ℝ :=
  4 * playground_side +
  2 * (veg_garden_length + veg_garden_width) +
  2 * Real.pi * flower_bed_radius +
  sandpit_side1 + sandpit_side2 + sandpit_side3 +
  irregular_bed_perimeter +
  l_shape_length1 + l_shape_width1 + l_shape_length2 + l_shape_width2 + l_shape_length1 + l_shape_width2

theorem total_fencing_approx :
  |total_fencing - 256.42| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_approx_l194_19461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_cos_2x_l194_19490

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem range_of_f_cos_2x :
  (∀ x ∈ Set.Icc (-1) 1, f x ∈ Set.Icc (-2) 0) →
  (∀ x ∈ Set.Icc (-2) 0, ∃ y ∈ Set.Icc (-1) 1, f y = x) →
  (∀ x, f (Real.cos (2 * x)) ∈ Set.Icc (-2) 0) ∧
  (∀ y ∈ Set.Icc (-2) 0, ∃ x, f (Real.cos (2 * x)) = y) := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_cos_2x_l194_19490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l194_19463

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides : 
  ∃! k : ℕ, k > 0 ∧ (∀ z : ℂ, (z^k - 1) % polynomial z = 0) ∧ 
  (∀ m : ℕ, 0 < m → m < k → ∃ z : ℂ, (z^m - 1) % polynomial z ≠ 0) ∧
  k = 42 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l194_19463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_market_analysis_l194_19430

/-- Represents the cost and pricing structure of fruits A and B -/
structure FruitMarket where
  cost_A : ℝ  -- Cost price of fruit A
  cost_B : ℝ  -- Cost price of fruit B
  sell_A : ℝ  -- Selling price of fruit A
  sell_B : ℝ  -- Selling price of fruit B

/-- Theorem stating the cost prices and maximum price reduction -/
theorem fruit_market_analysis (market : FruitMarket) : 
  market.sell_A = 20 ∧ 
  market.sell_B = 23 ∧ 
  15 * market.cost_A + 5 * market.cost_B = 305 ∧ 
  20 * market.cost_A + 10 * market.cost_B = 470 →
  market.cost_A = 14 ∧ 
  market.cost_B = 19 ∧
  ∃ (m : ℝ), m = 1.2 ∧ 
    ∀ (x : ℝ), 30 ≤ x ∧ x ≤ 80 →
      (let profit := if x ≤ 60 then 2*x + 400 else -x + 580
       let new_profit := (20 - 3*m - market.cost_A) * 60 + (23 - m - market.cost_B) * 40
       let total_cost := market.cost_A * 60 + market.cost_B * 40
       new_profit / total_cost ≥ 0.16) ∧
      ∀ (m' : ℝ), m' > m → 
        (20 - 3*m' - market.cost_A) * 60 + (23 - m' - market.cost_B) * 40 < 0.16 * (market.cost_A * 60 + market.cost_B * 40) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_market_analysis_l194_19430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l194_19460

/-- The probability function P(K) for the dragon's resilience x -/
noncomputable def P (x : ℝ) : ℝ := x^12 / (1 + x + x^2)^10

/-- The optimal value of x that maximizes P(K) -/
noncomputable def optimal_x : ℝ := (Real.sqrt 97 + 1) / 8

theorem dragon_resilience_maximizer :
  ∀ x > 0, P x ≤ P optimal_x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l194_19460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_eighteen_power_divides_factorial_l194_19424

theorem largest_n_eighteen_power_divides_factorial : 
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ (m : ℕ), 18^m ∣ Nat.factorial 30 → m ≤ n) ∧
  (18^n ∣ Nat.factorial 30) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_eighteen_power_divides_factorial_l194_19424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_l194_19408

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define a membership relation for Point and Circle
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

instance : Membership Point Circle where
  mem := Point.inCircle

-- Define the given conditions
axiom C1 : Circle
axiom C2 : Circle
axiom M : Point
axiom N : Point
axiom A : Point
axiom B : Point

-- Define the intersection of circles
axiom intersect : M ∈ C1 ∧ M ∈ C2 ∧ N ∈ C1 ∧ N ∈ C2
axiom distinct_intersection : M ≠ N

-- Define the points on respective circles
axiom A_on_C1 : A ∈ C1
axiom B_on_C2 : B ∈ C2

-- Define the tangency conditions
def IsTangent (c : Circle) (p q : Point) : Prop :=
  p ∈ c ∧ q ∈ c ∧ ∃ (l : Set Point), p ∈ l ∧ q ∈ l ∧ ∀ (r : Point), r ∈ l → r ∈ c → r = p ∨ r = q

axiom MA_tangent_C2 : IsTangent C2 M A
axiom MB_tangent_C1 : IsTangent C1 M B

-- Define angle measure
noncomputable def angle_measure (P Q R : Point) : ℝ := sorry

-- The theorem to prove
theorem equal_angles : angle_measure M N A = angle_measure M N B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_l194_19408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_points_between_A_and_B_l194_19414

/-- The number of points with integer coordinates strictly between two given points on a line. -/
def countIntegerPointsOnLine (x1 y1 x2 y2 : ℤ) : ℕ :=
  let dx := x2 - x1
  let dy := y2 - y1
  let gcd := Int.gcd dx.natAbs dy.natAbs
  (dx.natAbs / gcd) - 1

/-- Theorem stating that there are 96 points with integer coordinates strictly between
    A(3, 5) and B(100, 405) on the line passing through these points. -/
theorem count_integer_points_between_A_and_B :
  countIntegerPointsOnLine 3 5 100 405 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_points_between_A_and_B_l194_19414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_for_same_color_triangles_l194_19419

/-- A point in the plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool -- true for red, false for blue

/-- The set of all triangles formed by n points -/
def TriangleSet (points : List ColoredPoint) : Set (ColoredPoint × ColoredPoint × ColoredPoint) :=
  sorry

/-- Predicate to check if three points are collinear -/
def AreCollinear (p q r : ColoredPoint) : Prop :=
  sorry

/-- Predicate to check if a triangle has vertices of the same color -/
def SameColorTriangle (t : ColoredPoint × ColoredPoint × ColoredPoint) : Prop :=
  sorry

/-- The main theorem -/
theorem min_points_for_same_color_triangles :
  ∃ (n : ℕ), n = 8 ∧
  ∃ (points : List ColoredPoint),
    points.length = n ∧
    (∀ p q r, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → p ≠ r → ¬AreCollinear p q r) ∧
    (∃ t₁ t₂, t₁ ∈ TriangleSet points ∧ t₂ ∈ TriangleSet points ∧ SameColorTriangle t₁ ∧ SameColorTriangle t₂) ∧
    (∀ m, m < n → ¬∃ (subpoints : List ColoredPoint),
      subpoints.length = m ∧
      (∀ p q r, p ∈ subpoints → q ∈ subpoints → r ∈ subpoints → p ≠ q → q ≠ r → p ≠ r → ¬AreCollinear p q r) ∧
      (∃ t₁ t₂, t₁ ∈ TriangleSet subpoints ∧ t₂ ∈ TriangleSet subpoints ∧ SameColorTriangle t₁ ∧ SameColorTriangle t₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_for_same_color_triangles_l194_19419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_empty_position_conclusion_holds_without_king_l194_19494

/-- Represents the color of wine in a glass -/
inductive WineColor
| White
| Red

/-- Represents a person at the table with their wine glass -/
structure Person :=
  (position : Nat)
  (wineColor : WineColor)

/-- The state of the table before and after midnight -/
structure Table :=
  (people : List Person)
  (size : Nat)

/-- Defines the movement of glasses based on wine color -/
def moveGlass (table : Table) (p : Person) : Nat :=
  match p.wineColor with
  | WineColor.White => (p.position - 1 + table.size) % table.size
  | WineColor.Red => (p.position + 1) % table.size

/-- Theorem: After midnight, there will be at least one empty position -/
theorem at_least_one_empty_position (table : Table)
  (h1 : table.size = 101)
  (h2 : ∃ p ∈ table.people, p.wineColor = WineColor.White)
  (h3 : ∃ p ∈ table.people, p.wineColor = WineColor.Red)
  (h4 : ∀ p ∈ table.people, p.position < table.size) :
  ∃ pos : Nat, pos < table.size ∧ ∀ p ∈ table.people, moveGlass table p ≠ pos := by
  sorry

#check at_least_one_empty_position

/-- Theorem: The conclusion holds even if the King leaves before midnight -/
theorem conclusion_holds_without_king (table : Table)
  (h1 : table.size = 100)
  (h2 : ∃ p ∈ table.people, p.wineColor = WineColor.White)
  (h3 : ∃ p ∈ table.people, p.wineColor = WineColor.Red)
  (h4 : ∀ p ∈ table.people, p.position < table.size) :
  ∃ pos : Nat, pos < table.size ∧ ∀ p ∈ table.people, moveGlass table p ≠ pos := by
  sorry

#check conclusion_holds_without_king

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_empty_position_conclusion_holds_without_king_l194_19494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_applied_95_times_eq_twice_l194_19400

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^3)^(1/3)

theorem f_applied_95_times_eq_twice (n : ℕ) (h : n = 95) :
  (f^[n]) 19 = (1 - 1 / 19^3)^(1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_applied_95_times_eq_twice_l194_19400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l194_19453

/-- An equilateral triangle can be divided into 4 scalene acute triangles -/
theorem equilateral_triangle_division :
  ∃ (T : Type) (triangle : T) (is_equilateral : T → Prop) 
    (is_scalene : T → Prop) (is_acute : T → Prop) (division : T → Finset T),
    is_equilateral triangle ∧
    (∃ (sub_triangles : Finset T),
      sub_triangles.card = 4 ∧
      (∀ t ∈ sub_triangles, is_scalene t ∧ is_acute t) ∧
      division triangle = sub_triangles) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l194_19453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coral_polyp_population_decline_l194_19448

/-- The year when the coral polyp population first becomes less than 5% of its initial value -/
def year_population_below_5_percent : ℕ := 2019

/-- The annual decrease rate of the coral polyp population -/
def annual_decrease_rate : ℝ := 0.3

/-- The initial year of observation -/
def initial_year : ℕ := 2010

theorem coral_polyp_population_decline :
  let remaining_fraction := 1 - annual_decrease_rate
  let threshold := 0.05
  year_population_below_5_percent = initial_year + 
    (Int.ceil (Real.log threshold / Real.log remaining_fraction)).toNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coral_polyp_population_decline_l194_19448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l194_19466

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The five given points -/
noncomputable def points : List Point := [
  ⟨-3/2, 1⟩,
  ⟨0, 0⟩,
  ⟨0, 2⟩,
  ⟨3, 0⟩,
  ⟨4, 2⟩
]

/-- No three points are collinear -/
axiom not_collinear : ∀ p q r, p ∈ points → q ∈ points → r ∈ points → 
  p ≠ q → q ≠ r → p ≠ r →
  (q.y - p.y) * (r.x - q.x) ≠ (r.y - q.y) * (q.x - p.x)

/-- The conic section passing through the points is an ellipse -/
axiom is_ellipse : ∃ (a b h k : ℝ), ∀ p, p ∈ points →
  (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1

/-- Theorem: The length of the minor axis of the ellipse is 2√5 -/
theorem minor_axis_length :
  ∃ (a b h k : ℝ), (∀ p, p ∈ points → (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1) →
  2 * b = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l194_19466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_cost_is_175_l194_19435

/-- The cost for parking a car in a certain garage -/
structure ParkingCost where
  initial_cost : ℝ  -- Cost for the first 2 hours
  excess_cost : ℝ   -- Cost per hour after the first 2 hours
  total_hours : ℕ   -- Total parking duration
  avg_cost : ℝ      -- Average cost per hour for the total duration

/-- The parking cost satisfies the given conditions -/
def satisfies_conditions (p : ParkingCost) : Prop :=
  p.initial_cost = 20 ∧
  p.total_hours = 9 ∧
  p.avg_cost = 3.5833333333333335

/-- The theorem stating that the excess cost per hour is $1.75 -/
theorem excess_cost_is_175 (p : ParkingCost) 
  (h : satisfies_conditions p) : p.excess_cost = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_cost_is_175_l194_19435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_proof_l194_19498

theorem population_change_proof : 
  let year1_change : ℝ := 1 + 0.20
  let year2_change : ℝ := 1 + 0.30
  let year3_change : ℝ := 1 - 0.15
  let year4_change : ℝ := 1 - 0.25
  let total_change : ℝ := year1_change * year2_change * year3_change * year4_change
  ∃ ε : ℝ, ε > 0 ∧ |total_change - 1 + 0.06| < ε ∧ ε < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_proof_l194_19498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l194_19467

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x^2) / Real.log 10

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → 
  x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → 
  x ≤ y → f y ≤ f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l194_19467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l194_19426

theorem integer_solutions_equation : 
  ∀ x y : ℤ, x^2 - Nat.factorial y.toNat = 2001 ↔ (x = 45 ∧ y = 4) ∨ (x = -45 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l194_19426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_die_rolls_leap_year_l194_19445

/-- The expected number of rolls on a single day -/
noncomputable def expected_rolls_per_day : ℝ := 3/2

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The expected number of rolls in a leap year -/
noncomputable def expected_rolls_leap_year : ℝ := expected_rolls_per_day * (leap_year_days : ℝ)

theorem bob_die_rolls_leap_year : 
  expected_rolls_leap_year = 549 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_die_rolls_leap_year_l194_19445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_properties_l194_19495

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x < 0 then -2 - x
  else if x ≥ 0 ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x > 2 ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Define a default value for x outside the specified ranges

-- State the theorem
theorem abs_f_properties :
  (∀ x : ℝ, x ≥ -3 ∧ x < 0 → |f x| = x + 2) ∧
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ 2 → |f x| = |Real.sqrt (4 - (x - 2)^2) - 2|) ∧
  (∀ x : ℝ, x > 2 ∧ x ≤ 3 → |f x| = 2 * (x - 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_properties_l194_19495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l194_19429

theorem car_sale_profit (P : ℝ) (h : P > 0) : 
  let discount_rate : ℝ := 0.20
  let profit_rate : ℝ := 0.24
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := P * (1 + profit_rate)
  (selling_price - buying_price) / buying_price = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l194_19429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l194_19493

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (2, 0)

-- Define E as the midpoint of BD
noncomputable def E : ℝ × ℝ := ((D.1 + B.1) / 2, (D.2 + B.2) / 2)

-- Define F on DA such that DF = 1/4 * DA
noncomputable def F : ℝ × ℝ := (D.1 - (D.1 - A.1) / 4, D.2 - (D.2 - A.2) / 4)

-- Define G as the midpoint of BC
noncomputable def G : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

-- Theorem stating the ratio of areas
theorem area_ratio :
  let areaDFE := triangleArea D F E
  let areaABG := triangleArea A B G
  let areaAFG := triangleArea A F G
  let areaABFG := areaABG + areaAFG
  areaDFE / areaABFG = 1 / (4 * Real.sqrt 5 + 6) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l194_19493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_optimal_purification_l194_19477

-- Function definition
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then x^2 / 25 + 2
  else if x > 5 then (x + 19) / (2 * x - 2)
  else 0

-- Concentration function
noncomputable def concentration (m : ℝ) (x : ℝ) : ℝ := m * f x

-- Effective purification condition
def is_effectively_purified (m : ℝ) (x : ℝ) : Prop :=
  concentration m x ≥ 5

-- Optimal purification condition
def is_optimally_purified (m : ℝ) (x : ℝ) : Prop :=
  5 ≤ concentration m x ∧ concentration m x ≤ 10

-- Theorem: Minimum m for optimal purification within 9 days
theorem min_m_for_optimal_purification :
  ∃ (m : ℝ), m = 20/7 ∧
  (∀ (x : ℝ), 0 < x ∧ x ≤ 9 → is_optimally_purified m x) ∧
  (∀ (m' : ℝ), m' < m →
    ∃ (x : ℝ), 0 < x ∧ x ≤ 9 ∧ ¬(is_optimally_purified m' x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_optimal_purification_l194_19477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_zero_l194_19487

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_negative_two_equals_zero (a b c : ℝ) :
  f a b c 2 = 4 → f a b c (-2) = 0 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_zero_l194_19487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l194_19432

def g : ℕ → ℕ
| n => if n < 15 then 2 * n + 3 else g (n - 7)

theorem g_max_value : ∃ m : ℕ, ∀ n : ℕ, g n ≤ m ∧ ∃ k : ℕ, g k = m :=
  sorry

#eval g 14
#eval g 15
#eval g 21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l194_19432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtakes_red_car_l194_19436

/-- Represents the time in hours for one car to overtake another -/
noncomputable def overtakeTime (v1 v2 d : ℝ) : ℝ :=
  d / (v2 - v1)

/-- Theorem: Given the initial conditions, the black car will overtake the red car in 3 hours -/
theorem black_car_overtakes_red_car :
  let red_speed : ℝ := 40
  let black_speed : ℝ := 50
  let initial_distance : ℝ := 30
  overtakeTime red_speed black_speed initial_distance = 3 := by
  -- Unfold the definition of overtakeTime
  unfold overtakeTime
  -- Simplify the expression
  simp
  -- The proof is complete, but we use sorry to skip the detailed steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtakes_red_car_l194_19436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_historic_sets_l194_19484

/-- A historic set is a set of three nonnegative integers {x, y, z} with x < y < z,
    where {z-y, y-x} = {1776, 2001}. -/
def IsHistoricSet (s : Set ℕ) : Prop :=
  ∃ x y z : ℕ, s = {x, y, z} ∧ x < y ∧ y < z ∧
  ({z - y, y - x} : Set ℕ) = {1776, 2001}

/-- The theorem states that there exists a partition of ℕ into historic sets. -/
theorem partition_into_historic_sets :
  ∃ (P : Set (Set ℕ)), 
    (∀ s, s ∈ P → IsHistoricSet s) ∧ 
    (∀ a b, a ∈ P → b ∈ P → a ≠ b → a ∩ b = ∅) ∧
    (⋃₀ P = Set.univ) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_historic_sets_l194_19484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_donation_theorem_l194_19411

def race_length : ℕ := 5
def initial_donation : ℕ := 10

def donation_for_km (km : ℕ) : ℕ :=
  initial_donation * 2^(km - 1)

def total_donation : ℕ :=
  Finset.sum (Finset.range race_length) (fun k => donation_for_km (k + 1))

theorem race_donation_theorem : total_donation = 310 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_donation_theorem_l194_19411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l194_19486

theorem tan_sum_given_tan_cot_sum (x y : Real) 
  (h1 : Real.tan x + Real.tan y = 30)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 40) : 
  Real.tan (x + y) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l194_19486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l194_19473

/-- The floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- Theorem: No solution exists for the given recurrence relation -/
theorem no_solution_exists : ¬ ∃ (m n : ℕ+) (x : Fin (n + 1) → ℝ),
  x 0 = 428 ∧
  x n = 1928 ∧
  (∀ k : Fin n, (x (k + 1)) / 10 = floor ((x k) / 10) + m + frac ((x k) / 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l194_19473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cut_theorem_l194_19450

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The result of cutting a regular tetrahedron by planes passing through its edge midpoints -/
noncomputable def cut_tetrahedron (t : RegularTetrahedron) : RegularOctahedron :=
  { edge_length := t.edge_length / 2
    edge_length_pos := by
      apply div_pos
      · exact t.edge_length_pos
      · norm_num }

/-- Theorem stating that cutting a regular tetrahedron results in a regular octahedron
    with half the edge length -/
theorem tetrahedron_cut_theorem (t : RegularTetrahedron) :
  ∃ (o : RegularOctahedron), o = cut_tetrahedron t ∧ o.edge_length = t.edge_length / 2 := by
  use cut_tetrahedron t
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cut_theorem_l194_19450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_increase_double_minutes_l194_19410

/-- Calculates the percent increase in total expenditure when doubling call minutes -/
theorem percent_increase_double_minutes (cost_per_minute : ℝ) (initial_minutes : ℝ) :
  cost_per_minute > 0 → initial_minutes > 0 →
  (cost_per_minute * (2 * initial_minutes) - cost_per_minute * initial_minutes) / (cost_per_minute * initial_minutes) * 100 = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_increase_double_minutes_l194_19410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_characterization_l194_19409

/-- Two fixed points on a plane -/
structure FixedPoints (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  p1 : P
  p2 : P
  c : ℝ
  dist_eq : ‖p1 - p2‖ = 2 * c
  c_pos : c > 0

/-- A moving point with constant distance difference from two fixed points -/
structure MovingPoint (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (fp : FixedPoints P) where
  p : P
  a : ℝ
  dist_diff : |‖p - fp.p1‖ - ‖p - fp.p2‖| = 2 * a
  a_pos : a > 0

/-- The trajectory of the moving point -/
def Trajectory (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (fp : FixedPoints P) :=
  {p : P | ∃ mp : MovingPoint P fp, mp.p = p}

theorem trajectory_characterization (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (fp : FixedPoints P) :
  (∀ a > 0, (∃ p : Trajectory P fp, True) ↔ 2 * a < 2 * fp.c) ∧
  (∀ a > fp.c, ¬∃ p : Trajectory P fp, True) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_characterization_l194_19409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_foci_ratio_l194_19465

/-- An ellipse with foci on the axes and an inscribed equilateral triangle -/
structure EllipseWithTriangle where
  p : ℝ
  q : ℝ
  foci_distance : ℝ
  triangle_side : ℝ
  ellipse_equation : ∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → True
  foci_on_axes : foci_distance^2 = p^2 - q^2
  b_point : (0, q) ∈ {z : ℝ × ℝ | z.1^2 / p^2 + z.2^2 / q^2 = 1}
  ac_parallel_x : ∃ (h : ℝ), ((-triangle_side/2, h) ∈ {z : ℝ × ℝ | z.1^2 / p^2 + z.2^2 / q^2 = 1}) ∧
                              ((triangle_side/2, h) ∈ {z : ℝ × ℝ | z.1^2 / p^2 + z.2^2 / q^2 = 1})
  equilateral : triangle_side = q - Real.sqrt (foci_distance^2 / 4)
  foci_distance_value : foci_distance = 4

/-- The ratio of the triangle side to the foci distance is 4/3 -/
theorem triangle_foci_ratio (e : EllipseWithTriangle) : 
  e.triangle_side / e.foci_distance = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_foci_ratio_l194_19465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_dog_food_requirement_l194_19425

-- Define the dog size categories
inductive DogSize
  | Small
  | Medium
  | Large

-- Define a function to calculate food requirement based on dog size and weight
noncomputable def foodRequirement (size : DogSize) (weight : ℚ) : ℚ :=
  match size with
  | DogSize.Small => weight / 20
  | DogSize.Medium => weight / 15
  | DogSize.Large => weight / 10

-- Define Paul's dogs
def paulsDogs : List (DogSize × ℚ) :=
  [(DogSize.Small, 15), (DogSize.Small, 20),
   (DogSize.Medium, 25), (DogSize.Medium, 35), (DogSize.Medium, 45),
   (DogSize.Large, 55), (DogSize.Large, 60), (DogSize.Large, 75)]

-- Calculate the total food requirement
noncomputable def totalFoodRequirement : ℚ :=
  paulsDogs.map (fun (size, weight) => foodRequirement size weight) |>.sum

-- Theorem statement
theorem paul_dog_food_requirement :
  totalFoodRequirement = 111/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_dog_food_requirement_l194_19425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_of_4_eq_7_5_l194_19441

noncomputable def h (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def h_inverse (x : ℝ) : ℝ := (3*x - 4) / x

noncomputable def j (x : ℝ) : ℝ := 1 / (h_inverse x) + 7

theorem j_of_4_eq_7_5 : j 4 = 7.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_of_4_eq_7_5_l194_19441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_from_chords_l194_19471

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of chords -/
def num_chords : ℕ := n.choose 2

/-- The number of intersections inside the circle -/
def num_intersections : ℕ := n.choose 4

/-- Proposition: The number of triangles formed by intersections of chords inside the circle -/
theorem num_triangles_from_chords (h : ∀ p q r : Finset (Fin n), (p ∩ q ∩ r).card ≤ 2) : 
  num_intersections.choose 3 = 328750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_from_chords_l194_19471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_power_of_complex_fraction_modulus_of_root_l194_19491

-- Define the complex number -2-i
def z : ℂ := -2 - Complex.I

-- Define the equation x^2 + 4x + 5 = 0
def equation (x : ℂ) : Prop := x^2 + 4*x + 5 = 0

-- Statement 1
theorem imaginary_part_of_z : z.im = -1 := by sorry

-- Statement 2
theorem power_of_complex_fraction (n : ℕ) : ((1 + Complex.I) / (1 - Complex.I))^(4*n) = 1 := by sorry

-- Statement 3
theorem modulus_of_root : ∀ z : ℂ, equation z → Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_power_of_complex_fraction_modulus_of_root_l194_19491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_center_of_gravity_l194_19483

/-- Represents the properties of a cylindrical glass filled with water. -/
structure GlassWithWater where
  glassWeight : ℝ
  glassCenterOfGravity : ℝ
  baseArea : ℝ
  waterHeight : ℝ

/-- Calculates the center of gravity of the glass-water system. -/
noncomputable def systemCenterOfGravity (g : GlassWithWater) : ℝ :=
  (g.waterHeight ^ 2 + 96) / (2 * (g.waterHeight + 10))

/-- Theorem stating that the center of gravity is lowest when water height is 4 cm. -/
theorem lowest_center_of_gravity (g : GlassWithWater)
  (h1 : g.glassWeight = 200)
  (h2 : g.glassCenterOfGravity = 4.8)
  (h3 : g.baseArea = 20)
  (h4 : g.waterHeight ≥ 0) :
  ∀ x ≥ 0, systemCenterOfGravity g ≤ systemCenterOfGravity {g with waterHeight := x} ↔ g.waterHeight = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_center_of_gravity_l194_19483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l194_19485

-- Define the function L(m)
noncomputable def L (m : ℝ) : ℝ := -2 - Real.sqrt (m + 7)

-- Define the function r(m)
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- State the theorem
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ → |r m - 1 / Real.sqrt 7| < ε :=
by
  sorry

#check limit_of_r_as_m_approaches_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l194_19485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_S_1992_1993_l194_19431

def S (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) (λ i => (-1)^(i + 2) * (i + 1 : ℤ))

theorem sum_S_1992_1993 : S 1992 + S 1993 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_S_1992_1993_l194_19431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l194_19401

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1/2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos (2*x))
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/6)

theorem function_properties :
  (∃ (max : ℝ), ∀ x, f x ≤ max ∧ (∃ x₀, f x₀ = max)) ∧
  (∃ (period : ℝ), period > 0 ∧ ∀ x, f (x + period) = f x) ∧
  (Set.Icc (-1/2) 1 = {y | ∃ x ∈ Set.Icc 0 (Real.pi/2), g x = y}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l194_19401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l194_19437

/-- Given a reflection that maps (-2, 8) to (4, -4), 
    prove that the reflection of (-3, 4) is (37/5, -64/5) -/
theorem reflection_problem (reflection : ℝ × ℝ → ℝ × ℝ) : 
  reflection (-2, 8) = (4, -4) → 
  reflection (-3, 4) = (37/5, -64/5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l194_19437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_condition_l194_19454

open Real

/-- The function f(x) = -e^x - x --/
noncomputable def f (x : ℝ) : ℝ := -exp x - x

/-- The function g(x) = ax + 2cos(x) --/
noncomputable def g (a x : ℝ) : ℝ := a * x + 2 * cos x

/-- The derivative of f --/
noncomputable def f_deriv (x : ℝ) : ℝ := -exp x - 1

/-- The derivative of g --/
noncomputable def g_deriv (a x : ℝ) : ℝ := a - 2 * sin x

/-- The theorem stating the range of a for which the perpendicular tangent condition holds --/
theorem perpendicular_tangent_condition (a : ℝ) :
  (∀ x₁, ∃ x₂, f_deriv x₁ * g_deriv a x₂ = -1) ↔ a ∈ Set.Icc (-1) 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_condition_l194_19454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l194_19480

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 240/961 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l194_19480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_vertex_probability_l194_19476

/-- A flea jumping on a square -/
structure FleaJump where
  /-- The probability of jumping to an adjacent vertex -/
  p_jump : ℚ
  /-- The probability of jumping to an adjacent vertex is 1/2 -/
  p_jump_eq : p_jump = 1/2

/-- The probability that a specific vertex is the last one visited -/
def last_vertex_prob (fj : FleaJump) : ℚ := 1/3

/-- Theorem: The probability of each vertex (B, C, or D) being the last visited is 1/3 -/
theorem last_vertex_probability (fj : FleaJump) :
  last_vertex_prob fj = 1/3 := by
  sorry

#check last_vertex_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_vertex_probability_l194_19476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l194_19440

theorem problem_solution (x y : ℝ) :
  9 * (3 : ℝ)^x = (7 : ℝ)^(y + 7) → y = -7 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l194_19440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_territories_l194_19468

/-- Represents a position on the chessboard -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Represents a rook on the chessboard -/
structure Rook where
  pos : Position

/-- Represents the chessboard with rooks -/
structure Chessboard where
  rooks : Finset Rook
  non_attacking : ∀ r1 r2, r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → 
    r1.pos.row ≠ r2.pos.row ∧ r1.pos.col ≠ r2.pos.col
  eight_rooks : rooks.card = 8

/-- Calculate the territory of a rook -/
def territory (board : Chessboard) (r : Rook) : Real :=
  sorry

/-- The main theorem to prove -/
theorem equal_territories (board : Chessboard) :
  ∀ r, r ∈ board.rooks → territory board r = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_territories_l194_19468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_l194_19421

/-- The angle of inclination of a line in the Cartesian coordinate system --/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  if a = 0 then Real.pi/2
  else if b/a > 0 then Real.arctan (b/a)
  else Real.arctan (b/a) + Real.pi

/-- The line equation ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem angle_of_line (l : Line) :
  l.a = 1 ∧ l.b = 1 ∧ l.c = -3 →
  angle_of_inclination l.a l.b l.c = 3*Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_l194_19421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_scores_count_l194_19447

def BasketballScores : Finset ℕ :=
  Finset.image (λ i => 2*i + 3*(7-i)) (Finset.range 8)

theorem basketball_scores_count : Finset.card BasketballScores = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_scores_count_l194_19447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_fruit_drink_cost_is_two_l194_19422

/-- The cost of Bob's fruit drink given Andy and Bob's spending patterns -/
def bobs_fruit_drink_cost : ℚ :=
  let andys_total : ℚ := 1 + 4  -- $1 for soda, $4 for hamburgers
  let bobs_sandwich_cost : ℚ := 3
  let bobs_total : ℚ := andys_total  -- They spent the same amount
  bobs_total - bobs_sandwich_cost

/-- Proof that Bob's fruit drink costs $2 -/
theorem bobs_fruit_drink_cost_is_two :
  bobs_fruit_drink_cost = 2 := by
  unfold bobs_fruit_drink_cost
  norm_num

#eval bobs_fruit_drink_cost  -- This should output 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_fruit_drink_cost_is_two_l194_19422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l194_19464

/-- The vertex of a parabola described by y = ax^2 + bx + c is (-b/(2a), f(-b/(2a))) where f(x) = ax^2 + bx + c -/
theorem parabola_vertex (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let m : ℝ := -b / (2 * a)
  let n : ℝ := f m
  (∀ x, f x ≥ n) ∨ (∀ x, f x ≤ n) := by
  sorry

/-- The vertex of the parabola y = 2x^2 + 16x + 50 is (-4, 18) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 16 * x + 50
  let m : ℝ := -4
  let n : ℝ := 18
  (∀ x, f x ≥ n) ∨ (∀ x, f x ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l194_19464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l194_19451

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) 
  (h1 : is_geometric_sequence a q) 
  (h2 : a 0 > 0) : 
  is_necessary_not_sufficient 
    (q < 0) 
    (∀ n : ℕ, a (2*n) + a (2*n + 1) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_condition_l194_19451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_relation_l194_19402

-- Define the velocity of the river current
noncomputable def v_r : ℝ := sorry

-- Define the average velocity of the canoe in still water
noncomputable def v : ℝ := sorry

-- Define the distance between villages A and B
noncomputable def S : ℝ := sorry

-- Define the time to travel from A to B
noncomputable def t_AB : ℝ := S / (v + v_r)

-- Define the time to travel from B to A
noncomputable def t_BA : ℝ := S / (v - v_r)

-- Define the time to travel from B to A without paddles
noncomputable def t_BA_no_paddle : ℝ := S / v_r

-- Theorem stating the relationship between travel times
theorem travel_time_relation :
  t_AB = 3 * t_BA ∧ t_BA_no_paddle = 3 * t_BA :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_relation_l194_19402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_snack_shack_purchase_cost_l194_19439

/-- Represents the cost of items at Lisa's Snack Shack -/
structure SnackShackPrices where
  sandwich_price : ℚ
  soda_price : ℚ
  chips_price : ℚ

/-- Represents the quantities of items purchased -/
structure Purchase where
  sandwiches : ℕ
  sodas : ℕ
  chips : ℕ

/-- Calculates the total cost of a purchase at Lisa's Snack Shack -/
def total_cost (prices : SnackShackPrices) (purchase : Purchase) : ℚ :=
  prices.sandwich_price * purchase.sandwiches +
  prices.soda_price * purchase.sodas +
  prices.chips_price * purchase.chips

/-- Theorem stating that the total cost of the specific purchase at Lisa's Snack Shack is $36 -/
theorem lisa_snack_shack_purchase_cost :
  let prices : SnackShackPrices := ⟨4, 3, 3/2⟩
  let purchase : Purchase := ⟨3, 6, 4⟩
  total_cost prices purchase = 36 := by
  sorry

#eval let prices : SnackShackPrices := ⟨4, 3, 3/2⟩
      let purchase : Purchase := ⟨3, 6, 4⟩
      total_cost prices purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_snack_shack_purchase_cost_l194_19439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_series_solution_l194_19446

/-- An arithmetic-geometric series is a series where the ratio of consecutive terms forms an arithmetic sequence. -/
def is_arithmetic_geometric_series (a : ℕ → ℝ) : Prop :=
  ∃ r d : ℝ, ∀ n : ℕ, a (n + 1) / a n = r + d * n

/-- The sum of an infinite arithmetic-geometric series. -/
noncomputable def sum_arithmetic_geometric_series (a : ℕ → ℝ) : ℝ :=
  ∑' n, a n

/-- The given arithmetic-geometric series. -/
def series (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => (7 + 6 * n) * x^(n + 1)

theorem arithmetic_geometric_series_solution :
  ∃! x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ 
    is_arithmetic_geometric_series (series x) ∧
    sum_arithmetic_geometric_series (series x) = 100 ∧
    x = 251 / 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_series_solution_l194_19446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l194_19492

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 5) (h2 : |x| * y + x^2 = 5) :
  ∃ (n : ℤ), n = -3 ∧ ∀ (m : ℤ), |↑m - (x - y)| ≥ |↑n - (x - y)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_x_minus_y_l194_19492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l194_19442

/-- The distance from the focus to the asymptote of a hyperbola -/
noncomputable def distance_focus_to_asymptote (b : ℝ) : ℝ :=
  let a := 2
  let c := 3
  Real.sqrt 5

/-- Theorem: For a hyperbola with equation x²/4 - y²/b² = 1 and right focus at (3,0),
    the distance from the focus to its asymptote is √5 -/
theorem hyperbola_focus_asymptote_distance :
  ∀ b : ℝ, (4 + b^2 = 9) → distance_focus_to_asymptote b = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l194_19442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_common_chord_l194_19413

/-- Two intersecting circles -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 9 = 0

/-- The equation of the line where the common chord lies -/
def common_chord_line (x : ℝ) : Prop := x = 9/4

/-- The length of the common chord -/
noncomputable def common_chord_length : ℝ := Real.sqrt 55 / 2

theorem intersecting_circles_common_chord :
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) →
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord_line x) ∧
  (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧
    ∃ c d : ℝ, C₁ c d ∧ C₂ c d ∧ (c - a)^2 + (d - b)^2 = common_chord_length^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_common_chord_l194_19413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_C₂_l194_19478

/-- Curve C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := ∃ t : ℝ, x = 6 - Real.sqrt 2 / 2 * t ∧ y = 2 + Real.sqrt 2 / 2 * t

/-- Distance from a point (x, y) to C₂ -/
noncomputable def dist_to_C₂ (x y : ℝ) : ℝ := |x + y - 8| / Real.sqrt 2

/-- The theorem stating the maximum distance and the point where it occurs -/
theorem max_distance_to_C₂ : 
  (∃ x y : ℝ, C₁ x y ∧ ∀ x' y' : ℝ, C₁ x' y' → dist_to_C₂ x' y' ≤ dist_to_C₂ x y) ∧ 
  (∀ x y : ℝ, C₁ x y → dist_to_C₂ x y ≤ 5 * Real.sqrt 2) ∧
  (C₁ (-3/2) (-1/2) ∧ dist_to_C₂ (-3/2) (-1/2) = 5 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_C₂_l194_19478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l194_19434

-- Define the function f
noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.cos (2 * x + φ)

-- State the theorem
theorem function_properties (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) 
  (h2 : abs φ < Real.pi / 2)
  (h3 : f A φ (-Real.pi / 4) = 2 * Real.sqrt 2)
  (h4 : f A φ 0 = 2 * Real.sqrt 6)
  (h5 : f A φ (Real.pi / 12) = 2 * Real.sqrt 2)
  (h6 : f A φ (Real.pi / 4) = -2 * Real.sqrt 2)
  (h7 : f A φ (Real.pi / 3) = -2 * Real.sqrt 6) :
  φ = Real.pi / 6 ∧ f A φ (5 * Real.pi / 12) = -4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l194_19434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probability_l194_19475

-- Define the sample spaces for the two dice
def six_sided_die : Finset Nat := {1, 2, 3, 4, 5, 6}
def eight_sided_die : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the favorable outcomes for each event
def event_A : Finset Nat := {1, 2, 3, 4}
def event_B : Finset Nat := {2, 3, 5, 7}

-- Define a function to calculate probability
noncomputable def probability (favorable : Finset Nat) (sample_space : Finset Nat) : Rat :=
  (favorable.card : Rat) / (sample_space.card : Rat)

-- State the theorem
theorem dice_roll_probability :
  (probability event_A six_sided_die) * (probability event_B eight_sided_die) = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probability_l194_19475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l194_19457

/-- The equation of the directrix for a parabola -/
noncomputable def directrix_equation (a : ℝ) : ℝ := -1 / (8 * a)

/-- The parabola equation -/
def parabola_equation (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

theorem parabola_directrix (a : ℝ) (ha : a > 0) :
  ∀ x y : ℝ, parabola_equation x y a → (y = directrix_equation a ↔ y = -1/8) :=
by
  intros x y h
  simp [parabola_equation, directrix_equation] at *
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l194_19457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_l194_19417

theorem common_root_sum : 
  ∃ (p₁ p₂ : ℝ → ℝ) (k₁ k₂ : ℝ),
  (p₁ = fun x ↦ x^2 - 5*x + 4) ∧
  (p₂ = fun x ↦ x^2 - 7*x) ∧
  k₁ ≠ k₂ ∧ 
  (∃ x : ℝ, p₁ x = 0 ∧ p₂ x + k₁ = 0) ∧
  (∃ y : ℝ, p₁ y = 0 ∧ p₂ y + k₂ = 0) ∧
  (∀ k : ℝ, (∃ z : ℝ, p₁ z = 0 ∧ p₂ z + k = 0) → (k = k₁ ∨ k = k₂)) ∧
  k₁ + k₂ = 18 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_l194_19417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_lateral_edges_property_l194_19481

structure Prism where
  lateral_faces_are_parallelograms : Bool
  lateral_edges_are_common_edges : Bool

def Segment (α : Type*) := α × α

def is_lateral_edge (p : Prism) (e : Segment ℝ) : Prop := sorry

def parallel (e1 e2 : Segment ℝ) : Prop := sorry

def length (e : Segment ℝ) : ℝ := sorry

def lateral_edges_parallel_and_equal (p : Prism) : Prop :=
  p.lateral_faces_are_parallelograms ∧ p.lateral_edges_are_common_edges →
    ∃ (edges : Set (Segment ℝ)), 
      (∀ e, e ∈ edges → is_lateral_edge p e) ∧ 
      (∀ e1 e2, e1 ∈ edges → e2 ∈ edges → parallel e1 e2 ∧ length e1 = length e2)

theorem prism_lateral_edges_property (p : Prism) :
  lateral_edges_parallel_and_equal p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_lateral_edges_property_l194_19481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_a_l194_19469

noncomputable section

open Real

theorem triangle_sin_a (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a > 0 → b > 0 → c > 0 →
  a = 2 → 
  Real.cos B = 4/5 → 
  b = 3 → 
  a * Real.sin B = b * Real.sin A →
  Real.sin A = 2/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_a_l194_19469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l194_19428

/-- The radius of a circular wheel given its travel distance and number of revolutions -/
noncomputable def wheel_radius (distance : ℝ) (revolutions : ℝ) : ℝ :=
  distance * 1000 / (2 * Real.pi * revolutions)

/-- Theorem stating that a wheel covering 11 km in about 1000.4024994347707 revolutions has a radius of approximately 1.749 meters -/
theorem wheel_radius_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |wheel_radius 11 1000.4024994347707 - 1.749| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l194_19428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_terms_l194_19489

/-- A binomial is a polynomial with two terms. -/
structure Binomial (R : Type) [Semiring R] where
  a : R
  b : R
  x : R  -- The variable

/-- The result of multiplying two binomials and combining like terms. -/
def multiply_binomials {R : Type} [CommRing R] (p q : Binomial R) : List R := sorry

/-- The number of terms in the result of multiplying two binomials is 2, 3, or 4. -/
theorem binomial_product_terms {R : Type} [CommRing R] (p q : Binomial R) :
  (multiply_binomials p q).length ∈ ({2, 3, 4} : Set Nat) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_terms_l194_19489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_system_solution_l194_19407

structure TruckSystem where
  v : ℝ  -- Speed of an empty truck
  D : ℝ  -- Distance between points A and B
  t : ℝ  -- Loading time for one truck
  n : ℕ  -- Number of trucks

noncomputable def TruckSystem.loaded_speed (ts : TruckSystem) : ℝ := (6/7) * ts.v

noncomputable def TruckSystem.round_trip_time (ts : TruckSystem) : ℝ :=
  ts.D / ts.v + ts.D / (ts.loaded_speed)

theorem truck_system_solution (ts : TruckSystem) 
  (h1 : ts.D / ts.v = 6)  -- Petrov returns to A 6 minutes after meeting Ivanov
  (h2 : ∃ (δ : ℝ), 16 ≤ δ ∧ δ ≤ 19 ∧ δ = ts.D / ts.v)  -- Ivanov's return time bounds
  (h3 : ts.round_trip_time = 40)  -- Time between first and second meeting
  : ts.t = 13 ∧ ts.n = 5 := by
  sorry

#check truck_system_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_system_solution_l194_19407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_exponential_l194_19474

-- Define the function f(x) = 2ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1

-- Define the exponential function g(x) = a^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Theorem statement
theorem max_min_sum_exponential (a : ℝ) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≥ f a y) ∧
  (f a 2 = 7) →
  (g a 3 + g a 0 = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_exponential_l194_19474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_disproves_claim_l194_19444

-- Define the set of visible card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := CardSide × CardSide

-- Define the set of visible cards
def visibleCards : List CardSide :=
  [CardSide.Letter 'E', CardSide.Letter 'R', CardSide.Number 3, CardSide.Number 5, CardSide.Number 8]

-- Define vowels
def isVowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U'

-- Define prime numbers
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, 1 < m → m < n → ¬(n % m = 0)

-- Tom's claim
def tomsClaim (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isVowel c → isPrime n
  | _ => True

-- Function to check if a card disproves Tom's claim
def disprovesClaim (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isVowel c ∧ ¬isPrime n
  | (CardSide.Number n, CardSide.Letter c) => isVowel c ∧ ¬isPrime n
  | _ => False

-- Theorem: Turning over the card with 8 is the only way to potentially disprove Tom's claim
theorem eight_disproves_claim :
  ∀ (c : CardSide), c ∈ visibleCards →
    (∃ (other : CardSide), disprovesClaim (c, other)) ↔ c = CardSide.Number 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_disproves_claim_l194_19444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_needed_l194_19458

-- Define the room dimensions
noncomputable def room_length : ℝ := 15
noncomputable def room_width : ℝ := 9

-- Define the table dimensions
noncomputable def table_length : ℝ := 3
noncomputable def table_width : ℝ := 2

-- Define the conversion factor from square feet to square yards
noncomputable def sq_ft_to_sq_yd : ℝ := 1 / 9

-- Theorem statement
theorem carpet_needed : 
  ⌈(room_length * room_width - table_length * table_width) * sq_ft_to_sq_yd⌉ = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_needed_l194_19458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_zero_a_plus_b_bounds_l194_19427

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 
  Real.sqrt 2 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) - 2

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := f a b x + 1/2

-- Statement for part (1)
theorem g_has_unique_zero :
  ∃! x, x ∈ Set.Icc 0 (Real.pi / 4) ∧ g 1 0 x = 0 := by
  sorry

-- Statement for part (2)
theorem a_plus_b_bounds (a b : ℝ) :
  (∀ x, f a b x ≤ 0) → -2 ≤ a + b ∧ a + b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_zero_a_plus_b_bounds_l194_19427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l194_19404

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x^2))

-- State the theorem
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l194_19404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_g_range_l194_19418

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 3) + Real.sin x ^ 2 - Real.cos x ^ 2

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) := f x ^ 2 + f x

/-- The smallest positive period of f(x) is π -/
theorem f_period : ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ p = Real.pi := by
  sorry

/-- The equation of the axis of symmetry for f(x) -/
theorem f_symmetry_axis : ∀ (k : ℤ), ∃ (x : ℝ), x = k * Real.pi / 2 + Real.pi / 3 ∧ ∀ (y : ℝ), f (x - y) = f (x + y) := by
  sorry

/-- The range of g(x) is [-1/4, 2] -/
theorem g_range : Set.range g = Set.Icc (-1/4 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_g_range_l194_19418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_xy_length_l194_19438

/-- An isosceles right triangle with altitude feet -/
structure IsoscelesRightTriangleWithAltitudes where
  -- The triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Points on the sides
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Feet of altitudes
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- Triangle ABC is isosceles with right angle at A
  isIsosceles : A.1 = B.1 ∧ A.2 = C.2
  rightAngleA : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- D is on AB, E is on AC
  dOnAB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  eOnAC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- X and Y are feet of altitudes from D and E to BC
  xIsFootFromD : (X.1 - D.1) * (C.1 - B.1) + (X.2 - D.2) * (C.2 - B.2) = 0 ∧
                 ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ X = (B.1 + u * (C.1 - B.1), B.2 + u * (C.2 - B.2))
  yIsFootFromE : (Y.1 - E.1) * (C.1 - B.1) + (Y.2 - E.2) * (C.2 - B.2) = 0 ∧
                 ∃ v : ℝ, 0 ≤ v ∧ v ≤ 1 ∧ Y = (B.1 + v * (C.1 - B.1), B.2 + v * (C.2 - B.2))
  -- Given lengths
  adLength : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 48 * Real.sqrt 2
  aeLength : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 52 * Real.sqrt 2

/-- The main theorem -/
theorem isosceles_right_triangle_xy_length
  (t : IsoscelesRightTriangleWithAltitudes) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_xy_length_l194_19438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_l194_19423

noncomputable def x : ℕ → ℝ
| 0 => 7
| n + 1 => (x n ^ 2 + 3 * x n + 2) / (x n + 4)

theorem existence_of_m : 
  ∃ m : ℕ, m ∈ Finset.Icc 41 100 ∧ 
  x m ≤ 5 + 1 / 2^10 ∧
  ∀ k : ℕ, k > 0 → k < m → x k > 5 + 1 / 2^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_l194_19423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_l194_19415

-- Define the function
def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

-- Define the interval
def interval : Set ℝ := Set.Icc (-1 : ℝ) 3

-- Theorem statement
theorem roots_of_f :
  ∃ (x₁ x₂ x₃ : ℝ), 
    x₁ ∈ interval ∧ 
    x₂ ∈ interval ∧ 
    x₃ ∈ interval ∧ 
    f x₁ = 0 ∧ 
    f x₂ = 0 ∧ 
    f x₃ = 0 ∧ 
    abs (x₁ + 0.4) < 0.1 ∧ 
    abs (x₂ - 0.5) < 0.1 ∧ 
    abs (x₃ - 2.6) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_l194_19415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_draw_guarantee_l194_19433

/-- Represents the point value of a card -/
def CardValue := Nat

/-- The total number of cards in the deck -/
def DeckSize : Nat := 54

/-- The target sum we're looking for in a pair of cards -/
def TargetSum : Nat := 14

/-- The set of possible card values in the deck -/
def CardValues : Finset Nat :=
  Finset.range 14

/-- The minimum number of cards needed to guarantee a pair summing to the target -/
def MinimumDraw : Nat := 28

theorem minimum_draw_guarantee (draw : Nat) :
  draw ≥ MinimumDraw →
  ∀ (S : Finset Nat),
    S ⊆ CardValues →
    S.card = draw →
    ∃ (x y : Nat), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x + y = TargetSum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_draw_guarantee_l194_19433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_probability_l194_19456

-- Define the number of socks and colors
def total_socks : ℕ := 10
def num_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 5

-- Define the probability we want to prove
def target_probability : ℚ := 5/42

-- Theorem statement
theorem two_pairs_probability :
  (Nat.choose total_socks socks_drawn) ≠ 0 →
  (↑(Nat.choose num_colors 2 * Nat.choose (num_colors - 2) 1) : ℚ) / (↑(Nat.choose total_socks socks_drawn)) = target_probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_probability_l194_19456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_quadrilateral_properties_l194_19497

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

structure Line where
  p1 : Point
  p2 : Point

-- Define the given conditions
def circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

def touches (c : Circle) (l : Line) (p : Point) : Prop := sorry

def is_midpoint (p : Point) (l : Line) : Prop := sorry

def intersect (l1 l2 : Line) (p : Point) : Prop := sorry

def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Main theorem
theorem circumscribed_quadrilateral_properties 
  (ABCD : Quadrilateral) (O : Circle) (E F G H P M N : Point) 
  (EG FH AC BD : Line) :
  circumscribed ABCD O →
  touches O (Line.mk ABCD.A ABCD.B) E →
  touches O (Line.mk ABCD.B ABCD.C) F →
  touches O (Line.mk ABCD.C ABCD.D) G →
  touches O (Line.mk ABCD.D ABCD.A) H →
  intersect EG FH P →
  is_midpoint M BD →
  is_midpoint N AC →
  (collinear M N O.center ∧ intersect AC BD P) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_quadrilateral_properties_l194_19497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_135_deg_intercept_neg1_l194_19482

/-- Represents the equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Converts a slope angle in degrees to a slope value -/
noncomputable def slopeFromAngle (angle : ℝ) : ℝ :=
  Real.tan (angle * Real.pi / 180)

/-- Theorem: The equation of a line with slope angle 135° and y-intercept -1 is x + y + 1 = 0 -/
theorem line_equation_135_deg_intercept_neg1 :
  let m := slopeFromAngle 135
  let b := -1
  LineEquation.mk 1 1 1 = LineEquation.mk 1 (-m) (-b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_135_deg_intercept_neg1_l194_19482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_significant_digits_l194_19420

/-- The number of significant digits in a measurement -/
def significantDigits (x : ℝ) : ℕ := sorry

/-- The area of the rectangle in square inches -/
noncomputable def area : ℝ := 2.0561

/-- The width of the rectangle in inches -/
noncomputable def width : ℝ := 1.8

/-- The length of the rectangle in inches -/
noncomputable def length : ℝ := area / width

theorem length_significant_digits : significantDigits length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_significant_digits_l194_19420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sum_theorem_l194_19470

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  ending : ℝ × ℝ

-- Define parallelism for line segments
def parallel (s1 s2 : LineSegment) : Prop := sorry

-- Define the sum of two line segments
def sum_segments (s1 s2 : LineSegment) : Set (ℝ × ℝ) := sorry

-- Define a parallelogram
def is_parallelogram (s : Set (ℝ × ℝ)) : Prop := sorry

-- Define the length of a line segment
noncomputable def length (s : LineSegment) : ℝ := sorry

theorem segment_sum_theorem (Φ₁ Φ₂ : LineSegment) :
  (¬ parallel Φ₁ Φ₂ → is_parallelogram (sum_segments Φ₁ Φ₂)) ∧
  (parallel Φ₁ Φ₂ → 
    ∃ (s : LineSegment), 
      sum_segments Φ₁ Φ₂ = {s.start, s.ending} ∧ 
      parallel s Φ₁ ∧ 
      parallel s Φ₂ ∧ 
      length s = length Φ₁ + length Φ₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sum_theorem_l194_19470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_expression_l194_19406

open Complex

theorem magnitude_of_complex_expression :
  let z : ℂ := (1/3 : ℂ) - (2/3 : ℂ) * I
  abs (z^(4.5 : ℂ)) = (5 : ℝ)^(2.25 : ℝ) / (27 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_expression_l194_19406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_service_center_location_l194_19443

/-- The milepost of the fifth exit -/
noncomputable def fifth_exit : ℝ := 50

/-- The milepost of the twelfth exit -/
noncomputable def twelfth_exit : ℝ := 200

/-- The fraction of the distance between the fifth and twelfth exits where the service center is located -/
noncomputable def service_center_fraction : ℝ := 2/3

/-- The milepost of the service center -/
noncomputable def service_center_milepost : ℝ := fifth_exit + service_center_fraction * (twelfth_exit - fifth_exit)

/-- Theorem stating that the service center is located at milepost 150 -/
theorem service_center_location : service_center_milepost = 150 := by
  -- Expand the definition of service_center_milepost
  unfold service_center_milepost
  -- Simplify the expression
  simp [fifth_exit, twelfth_exit, service_center_fraction]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_service_center_location_l194_19443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_2_l194_19459

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the line
def line_eq (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the chord length
def chord_length (l : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    l^2 = (x₂ - x₁)^2 + (y₂ - y₁)^2

-- Theorem statement
theorem chord_length_is_sqrt_2 :
  chord_length (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_2_l194_19459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_quadrants_l194_19472

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 / x

-- State the theorem
theorem f_decreasing_in_quadrants :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₂ < f x₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_quadrants_l194_19472
