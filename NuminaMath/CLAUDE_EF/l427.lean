import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_equals_f_inv_l427_42711

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem statement
theorem unique_solution_f_equals_f_inv :
  ∃! x : ℝ, f x = f_inv x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_equals_f_inv_l427_42711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l427_42725

/-- Given a triangle ABC with sides a, b, and c, if its area is (a^2 + b^2 - c^2) / 4,
    then the internal angle C is 90°. -/
theorem triangle_right_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / 4 = (1 / 2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l427_42725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l427_42727

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that (n+i)^6 is an integer
def is_integer_power (n : ℤ) : Prop := ∃ (m : ℤ), (n : ℂ) + i ^ 6 = m

-- Theorem statement
theorem unique_integer_power :
  ∃! (n : ℤ), is_integer_power n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l427_42727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_sixteen_l427_42792

-- Define the variables as natural numbers
variable (A B C D E : ℕ)

-- Define the conditions
def condition1 (C E : ℕ) : Prop := C + E = 4
def condition2 (B E : ℕ) : Prop := B + E = 7
def condition3 (B D : ℕ) : Prop := B + D = 6
def condition4 (A : ℕ) : Prop := A = 6

-- Define the property that all variables are different
def all_different (A B C D E : ℕ) : Prop := 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

-- Theorem statement
theorem sum_equals_sixteen 
  (h1 : condition1 C E) (h2 : condition2 B E) (h3 : condition3 B D) (h4 : condition4 A) 
  (h5 : all_different A B C D E) : 
  A + B + C + D + E = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_sixteen_l427_42792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l427_42701

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3*a) / Real.log (1/2)

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 2 < x ∧ x < y → f a y < f a x) →
  a ∈ Set.Icc (-4 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l427_42701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_fifth_fourth_degree_l427_42755

/-- A polynomial function -/
def MyPolynomial (α : Type*) [Semiring α] := ℕ → α

/-- The degree of a polynomial -/
noncomputable def degree {α : Type*} [Semiring α] (p : MyPolynomial α) : ℕ := sorry

/-- The leading coefficient of a polynomial -/
noncomputable def leadingCoeff {α : Type*} [Semiring α] (p : MyPolynomial α) : α := sorry

/-- The number of intersection points between two polynomials -/
noncomputable def numIntersections {α : Type*} [Semiring α] (p q : MyPolynomial α) : ℕ := sorry

theorem max_intersections_fifth_fourth_degree (p q : MyPolynomial ℝ) :
  degree p = 5 →
  degree q = 4 →
  leadingCoeff p = 2 →
  leadingCoeff q = 1 →
  numIntersections p q ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_fifth_fourth_degree_l427_42755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glasses_cost_is_250_l427_42799

/-- The total cost of glasses given frame and lens prices, insurance coverage, and a coupon -/
noncomputable def total_cost_glasses (frame_price : ℝ) (lens_price : ℝ) (insurance_coverage_percent : ℝ) (coupon_value : ℝ) : ℝ :=
  (frame_price - coupon_value) + (lens_price * (1 - insurance_coverage_percent / 100))

/-- Theorem stating that the total cost of glasses is $250 under given conditions -/
theorem glasses_cost_is_250 :
  total_cost_glasses 200 500 80 50 = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_glasses_cost_is_250_l427_42799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l427_42726

/-- Given an initial amount P and an interest rate R, 
    calculates the final amount after 3 years of simple interest -/
noncomputable def finalAmount (P R : ℚ) : ℚ := P * (1 + 3 * R / 100)

/-- Represents the problem of finding the initial amount given two scenarios of simple interest -/
theorem initial_amount_proof (P R : ℚ) : 
  finalAmount P R = 956 ∧ 
  finalAmount P (R + 4) = 1061 → 
  P = 875 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l427_42726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_q_for_specific_triangle_l427_42703

/-- Triangle with an incircle that evenly trisects a median -/
structure TriangleWithTrisectedMedian where
  -- Side length PQ
  pq : ℝ
  -- Area of the triangle
  area : ℝ
  -- Assumption that the incircle evenly trisects the median PS
  trisects_median : True

/-- The sum of p and q for a triangle with trisected median -/
noncomputable def sum_p_q (t : TriangleWithTrisectedMedian) : ℕ :=
  let p := Int.floor t.area
  let q := Int.floor (t.area^2 / (p : ℝ)^2)
  p.natAbs + q.natAbs

/-- Theorem stating the sum of p and q for the specific triangle -/
theorem sum_p_q_for_specific_triangle :
  ∃ (t : TriangleWithTrisectedMedian),
    t.pq = 28 ∧
    (∃ (p q : ℕ), Nat.Prime q ∧ t.area = (p : ℝ) * Real.sqrt (q : ℝ)) ∧
    sum_p_q t = 199 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_q_for_specific_triangle_l427_42703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_a_range_l427_42722

-- Define the equation
noncomputable def equation (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*x + Real.log (a^2 - a) / Real.log a = 0

-- Define the condition that the equation has one positive and one negative root
def has_pos_neg_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ < 0 ∧ equation a x₁ ∧ equation a x₂

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  1 < a ∧ a < (1 + Real.sqrt 5) / 2

-- State the theorem
theorem equation_roots_imply_a_range :
  ∀ a : ℝ, has_pos_neg_roots a → a_range a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_a_range_l427_42722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l427_42762

/-- Represents a position on the chocolate bar -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the state of the chocolate bar game -/
structure ChocolateGame :=
  (n : Nat) -- width of the bar
  (m : Nat) -- height of the bar
  (currentPlayer : Bool) -- true for first player, false for second player
  (remainingSquares : Set Position)

/-- Represents a move in the game -/
structure Move :=
  (topLeft : Position)
  (bottomRight : Position)

/-- Checks if a move is valid -/
def isValidMove (game : ChocolateGame) (move : Move) : Prop :=
  move.topLeft.x ≤ move.bottomRight.x ∧
  move.topLeft.y ≤ move.bottomRight.y ∧
  move.bottomRight.x ≤ game.n ∧
  move.bottomRight.y ≤ game.m ∧
  (∃ p ∈ game.remainingSquares, 
    move.topLeft.x ≤ p.x ∧ p.x ≤ move.bottomRight.x ∧
    move.topLeft.y ≤ p.y ∧ p.y ≤ move.bottomRight.y)

/-- Applies a move to the game state -/
def applyMove (game : ChocolateGame) (move : Move) : ChocolateGame :=
  { game with
    currentPlayer := ¬game.currentPlayer,
    remainingSquares := {p ∈ game.remainingSquares | 
      p.x < move.topLeft.x ∨ p.x > move.bottomRight.x ∨
      p.y < move.topLeft.y ∨ p.y > move.bottomRight.y}
  }

/-- Checks if the game is over -/
def isGameOver (game : ChocolateGame) : Prop :=
  ⟨1, 1⟩ ∉ game.remainingSquares

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins (n m : Nat) :
  ∃ (strategy : ChocolateGame → Move),
    ∀ (game : ChocolateGame),
      game.n = n ∧ game.m = m ∧ game.currentPlayer = true →
      (isValidMove game (strategy game) ∧
       (isGameOver (applyMove game (strategy game)) ∨
        ∀ (opponentMove : Move),
          isValidMove (applyMove game (strategy game)) opponentMove →
          ¬isGameOver (applyMove (applyMove game (strategy game)) opponentMove))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l427_42762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l427_42791

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + (Real.sqrt 3 / 2) * Real.sin (2 * x)

structure Triangle (A B C : ℝ) where
  angle_A_valid : 0 < A ∧ A < π
  f_condition : f (A / 2) = 1
  area : (1 / 2) * Real.sin A = 3 * Real.sqrt 3

theorem min_side_a (A B C : ℝ) (t : Triangle A B C) :
  ∃ (a b c : ℝ), 
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧ 
    a ≥ 2 * Real.sqrt 3 ∧
    (a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 3 ∧ c = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l427_42791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l427_42795

def num_families : ℕ := 3
def family_size : ℕ := 3
def total_seats : ℕ := 9

theorem seating_arrangements :
  (∀ f, f ≤ num_families → family_size = 3) →
  total_seats = num_families * family_size →
  (Nat.factorial num_families)^num_families = (num_families!)^num_families :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l427_42795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cube_roots_l427_42716

theorem sum_of_cube_roots (a b c : ℕ+) : 
  (4 : ℝ) * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 3 (1/3)) = Real.rpow a.val (1/3) + Real.rpow b.val (1/3) - Real.rpow c.val (1/3) →
  a.val + b.val + c.val = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cube_roots_l427_42716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bilinear_upper_half_plane_condition_l427_42724

/-- A bilinear function that maps complex numbers to complex numbers -/
noncomputable def bilinear_function (a b c d : ℝ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

/-- The upper half-plane -/
def upper_half_plane : Set ℂ :=
  {z : ℂ | z.im > 0}

/-- Theorem stating the condition for mapping upper half-plane to upper half-plane -/
theorem bilinear_upper_half_plane_condition
  (a b c d : ℝ) :
  (∀ z ∈ upper_half_plane, bilinear_function a b c d z ∈ upper_half_plane) ↔
  a * d - b * c > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bilinear_upper_half_plane_condition_l427_42724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l427_42765

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem triangle_abc_area :
  let a : Point := ⟨0, 0⟩
  let b : Point := ⟨2, 0⟩
  let c : Point := ⟨2, 2⟩
  triangleArea a b c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l427_42765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_angle_bisector_l427_42713

-- Define an angle
structure Angle where
  measure : ℝ
  h : 0 ≤ measure ∧ measure < 2 * Real.pi

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define an angle bisector
noncomputable def angleBisector (a : Angle) : Line :=
{ slope := Real.tan (a.measure / 2),
  intercept := 0 }

-- Define axis of symmetry
noncomputable def axisOfSymmetry (a : Angle) : Line :=
{ slope := Real.tan (a.measure / 2),
  intercept := 0 }

-- Theorem: The axis of symmetry of an angle is the line on which its angle bisector lies
theorem axis_of_symmetry_is_angle_bisector (a : Angle) :
  axisOfSymmetry a = angleBisector a := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_angle_bisector_l427_42713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_a_annual_income_l427_42704

/-- Calculates A's annual income given the income ratios and C's monthly income -/
theorem calculate_a_annual_income 
  (ratio_a_b : ℚ) -- Ratio of A's income to B's income
  (b_increase_percent : ℚ) -- Percentage increase of B's income compared to C's
  (c_monthly_income : ℚ) -- C's monthly income in Rs.
  (h1 : ratio_a_b = 5 / 2) -- A's income is 5/2 times B's income
  (h2 : b_increase_percent = 12 / 100) -- B's income is 12% more than C's
  (h3 : c_monthly_income = 17000) -- C's monthly income is Rs. 17000
  : ℚ -- A's annual income in Rs.
  := by
  let b_monthly_income := c_monthly_income * (1 + b_increase_percent)
  let a_monthly_income := b_monthly_income * ratio_a_b
  let a_annual_income := a_monthly_income * 12
  have : a_annual_income = 571200 := by sorry
  exact 571200

#eval (5 : ℚ) / 2 * (1 + 12 / 100) * 17000 * 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_a_annual_income_l427_42704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l427_42738

-- Define the two lines
noncomputable def line1 (x y : ℝ) : Prop := 2*x + 7*y - 4 = 0
noncomputable def line2 (x y : ℝ) : Prop := 7*x - 21*y - 1 = 0

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (1, 2/7)

-- Define points A and B
noncomputable def point_A : ℝ × ℝ := (-3, 1)
noncomputable def point_B : ℝ × ℝ := (5, 7)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

-- State the theorem
theorem equidistant_line :
  ∃ (a b c : ℝ),
    (∀ (x y : ℝ), line1 x y ∧ line2 x y → a*x + b*y + c = 0) ∧
    (distance_to_line point_A.1 point_A.2 a b c = distance_to_line point_B.1 point_B.2 a b c) ∧
    ((a = 21 ∧ b = -28 ∧ c = -13) ∨ (a = 1 ∧ b = 0 ∧ c = -1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l427_42738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_problem_l427_42734

theorem triangle_tangent_problem (A B C : ℝ) :
  -- Triangle ABC
  A + B + C = Real.pi →
  -- Given conditions
  Real.tan A = 3/4 →
  Real.tan (A - B) = -1/3 →
  -- Conclusion
  Real.tan C = 79/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_problem_l427_42734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l427_42753

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Define proposition p
def p : Prop := ∀ x : ℝ, x ≠ 0 → f x ≥ 4

-- Define proposition q
def q : Prop := ∀ (A B : ℝ) (a b : ℝ),
  A > B ↔ a > b

-- Theorem to prove
theorem correct_answer : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l427_42753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_in_trapezoid_l427_42798

/-- Given a trapezoid with bases of lengths a and b, if a line parallel to the bases
    divides the trapezoid into two similar trapezoids, then the length of the segment
    of this line enclosed within the trapezoid is √(ab). -/
theorem parallel_line_in_trapezoid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (trapezoid : Set (ℝ × ℝ)) (parallel_line : Set (ℝ × ℝ))
    (isTrapezoid : Set (ℝ × ℝ) → Prop)
    (baseLength : Set (ℝ × ℝ) → ℝ → Prop)
    (isParallelToBase : Set (ℝ × ℝ) → Set (ℝ × ℝ) → Prop)
    (dividesIntoSimilarTrapezoids : Set (ℝ × ℝ) → Set (ℝ × ℝ) → Prop)
    (segmentLength : Set (ℝ × ℝ) → ℝ),
    isTrapezoid trapezoid ∧
    baseLength trapezoid a ∧
    baseLength trapezoid b ∧
    isParallelToBase parallel_line trapezoid ∧
    dividesIntoSimilarTrapezoids parallel_line trapezoid →
    segmentLength (parallel_line ∩ trapezoid) = Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_in_trapezoid_l427_42798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l427_42742

/-- Represents different units of measurement -/
inductive MeasurementUnit
  | Centimeter
  | Kilogram
  | Millimeter
  | Ton
  | Hour
  | Meter

/-- Represents a measurement with a value and a unit -/
structure Measurement where
  value : ℕ
  unit : MeasurementUnit

/-- Function to determine the appropriate unit for a given measurement -/
def appropriate_unit (m : Measurement) : MeasurementUnit :=
  match m with
  | ⟨140, _⟩ => MeasurementUnit.Centimeter  -- Height
  | ⟨23, _⟩ => MeasurementUnit.Kilogram     -- Weight
  | ⟨20, _⟩ => MeasurementUnit.Centimeter   -- Book length
  | ⟨7, _⟩ => MeasurementUnit.Millimeter    -- Book thickness
  | ⟨4, _⟩ => MeasurementUnit.Ton           -- Truck cargo
  | ⟨9, _⟩ => MeasurementUnit.Hour          -- Children's sleep
  | ⟨12, _⟩ => MeasurementUnit.Meter        -- Tree height
  | _ => MeasurementUnit.Centimeter         -- Default case

/-- Theorem stating that the appropriate_unit function returns the correct units -/
theorem correct_units :
  (appropriate_unit ⟨140, MeasurementUnit.Centimeter⟩ = MeasurementUnit.Centimeter) ∧
  (appropriate_unit ⟨23, MeasurementUnit.Kilogram⟩ = MeasurementUnit.Kilogram) ∧
  (appropriate_unit ⟨20, MeasurementUnit.Centimeter⟩ = MeasurementUnit.Centimeter) ∧
  (appropriate_unit ⟨7, MeasurementUnit.Millimeter⟩ = MeasurementUnit.Millimeter) ∧
  (appropriate_unit ⟨4, MeasurementUnit.Ton⟩ = MeasurementUnit.Ton) ∧
  (appropriate_unit ⟨9, MeasurementUnit.Hour⟩ = MeasurementUnit.Hour) ∧
  (appropriate_unit ⟨12, MeasurementUnit.Meter⟩ = MeasurementUnit.Meter) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l427_42742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l427_42735

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x + a / Real.exp x|

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l427_42735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_problem_l427_42749

theorem least_common_multiple_problem :
  ∃ (n : ℕ), n > 0 ∧ 
  Nat.lcm (Nat.lcm n 16) (Nat.lcm 18 24) = 144 ∧ 
  ∀ (m : ℕ), m > 0 → Nat.lcm (Nat.lcm m 16) (Nat.lcm 18 24) = 144 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_problem_l427_42749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_sum_l427_42740

/-- A parallelogram with integer coordinate vertices -/
structure IntParallelogram where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ

/-- Calculate the distance between two points with integer coordinates -/
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  Real.sqrt (((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) : ℝ)

/-- Calculate the perimeter of an IntParallelogram -/
noncomputable def perimeter (p : IntParallelogram) : ℝ :=
  distance p.a p.b + distance p.b p.c + distance p.c p.d + distance p.d p.a

/-- Calculate the area of an IntParallelogram -/
def area (p : IntParallelogram) : ℝ :=
  ((p.b.1 - p.a.1) * (p.b.2 - p.a.2) : ℝ)

/-- The main theorem -/
theorem parallelogram_sum (p : IntParallelogram) 
  (h1 : p.a = (0, 0)) 
  (h2 : p.b = (3, 4)) 
  (h3 : p.c = (10, 4)) 
  (h4 : p.d = (7, 0)) : 
  perimeter p + area p = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_sum_l427_42740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l427_42760

/-- The complex number z -/
noncomputable def z : ℂ := (4 + 2*Complex.I) / (1 + Complex.I)^2

/-- The line equation: x - 2y + m = 0 -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Theorem statement -/
theorem complex_point_on_line :
  ∃ (x y : ℝ), (z = x + y*Complex.I ∧ line_equation x y (-5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l427_42760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_condition_l427_42780

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem sin_odd_condition (φ : ℝ) :
  (φ = 0 → is_odd (f φ)) ∧ (∃ φ' : ℝ, φ' ≠ 0 ∧ is_odd (f φ')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_condition_l427_42780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_theorem_l427_42721

/-- The volume of a regular quadrilateral pyramid -/
noncomputable def regular_quadrilateral_pyramid_volume
  (b : ℝ)  -- base side length
  (α : ℝ)  -- dihedral angle between adjacent lateral faces
  (φ : ℝ)  -- angle between lateral edge and base plane
  : ℝ :=
  (b^3 * Real.sqrt 2 * Real.tan φ) / 6

/-- Theorem: Volume of a regular quadrilateral pyramid -/
theorem regular_quadrilateral_pyramid_volume_theorem
  (b : ℝ)  -- base side length
  (α : ℝ)  -- dihedral angle between adjacent lateral faces
  (φ : ℝ)  -- angle between lateral edge and base plane
  (h1 : b > 0)
  (h2 : 0 < α ∧ α < π)
  (h3 : 0 < φ ∧ φ < π/2)
  : regular_quadrilateral_pyramid_volume b α φ = (b^3 * Real.sqrt 2 * Real.tan φ) / 6 :=
by
  -- Unfold the definition of regular_quadrilateral_pyramid_volume
  unfold regular_quadrilateral_pyramid_volume
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_theorem_l427_42721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l427_42773

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := is_on_ellipse P.1 P.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum (P : ℝ × ℝ) (h : point_on_ellipse P) :
  ∀ Q : ℝ × ℝ, point_on_ellipse Q →
    distance Q point_A + distance Q left_focus ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l427_42773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_clock_sync_count_l427_42766

/-- Represents the speed of the broken clock relative to a normal clock -/
noncomputable def broken_clock_speed : ℚ := -5

/-- The number of seconds in 12 hours -/
def half_day_seconds : ℕ := 12 * 3600

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The time (in hours) it takes for the broken clock to sync with the normal clock -/
noncomputable def sync_time : ℚ := (half_day_seconds : ℚ) / (broken_clock_speed * 3600 - 3600)

/-- Theorem stating that the broken clock syncs with the normal clock 12 times in 24 hours -/
theorem broken_clock_sync_count : 
  ⌊(hours_per_day : ℚ) / sync_time⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_clock_sync_count_l427_42766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_saved_is_1152_l427_42747

/-- The number of small cubes -/
def num_small_cubes : ℕ := 64

/-- The edge length of each small cube in cm -/
def small_cube_edge : ℝ := 2

/-- The surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

/-- The total surface area of all small cubes -/
def total_small_cubes_area : ℝ := num_small_cubes * cube_surface_area small_cube_edge

/-- The edge length of the large cube -/
noncomputable def large_cube_edge : ℝ := small_cube_edge * (num_small_cubes^(1/3 : ℝ))

/-- The surface area of the large cube -/
noncomputable def large_cube_area : ℝ := cube_surface_area large_cube_edge

/-- The amount of paper saved -/
noncomputable def paper_saved : ℝ := total_small_cubes_area - large_cube_area

theorem paper_saved_is_1152 : paper_saved = 1152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_saved_is_1152_l427_42747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_return_theorem_l427_42736

/-- Calculates the combined yearly return percentage of two investments -/
noncomputable def combined_return_percentage (investment1 investment2 return1 return2 : ℝ) : ℝ :=
  let total_investment := investment1 + investment2
  let total_return := investment1 * return1 + investment2 * return2
  (total_return / total_investment) * 100

/-- Theorem stating that the combined yearly return percentage of the given investments is 13% -/
theorem investment_return_theorem :
  combined_return_percentage 500 1500 0.07 0.15 = 13 := by
  -- Proof goes here
  sorry

-- Use #eval only for computable functions
-- #eval combined_return_percentage 500 1500 0.07 0.15

-- Instead, we can use the following to check the result:
#check investment_return_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_return_theorem_l427_42736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_l427_42739

/-- Represents the time taken to fill a tank using three pipes simultaneously. -/
noncomputable def fillTime (timeA timeB timeC : ℝ) : ℝ :=
  1 / (1 / timeA + 1 / timeB + 1 / timeC)

/-- Theorem stating that three pipes with given fill times will fill a tank in 300/13 hours when used simultaneously. -/
theorem simultaneous_fill_time :
  fillTime 50 75 100 = 300 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_l427_42739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l427_42759

theorem negation_of_sin_inequality (h : ∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l427_42759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_positive_sum_l427_42772

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem largest_positive_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 > 0 →
  a 2003 + a 2004 > 0 →
  a 2003 * a 2004 < 0 →
  (∃ n : ℕ, sum_arithmetic a n > 0) →
  (∀ m : ℕ, m > 4006 → sum_arithmetic a m ≤ 0) →
  ∃ n : ℕ, n = 4006 ∧ sum_arithmetic a n > 0 ∧ ∀ m : ℕ, m > n → sum_arithmetic a m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_positive_sum_l427_42772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_constructible_l427_42743

/-- A pentagon with given side lengths and angles can be constructed -/
theorem pentagon_constructible (a b c d e : ℝ) (α φ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (hα : 0 < α ∧ α < π) (hφ : 0 < φ ∧ φ < π)
  (h_specific : a = 6 ∧ b = 9/2 ∧ c = 7.2 ∧ d = 8.5 ∧ e = 12 ∧ α = 2*π/3 ∧ φ = π/6) :
  ∃ (A B C D E : ℝ × ℝ), 
    let dist := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    let angle := λ (p q r : ℝ × ℝ) ↦ Real.arccos (
      ((q.1 - p.1) * (r.1 - p.1) + (q.2 - p.2) * (r.2 - p.2)) /
      (dist p q * dist p r)
    )
    dist A B = a ∧
    dist B C = b ∧
    dist C D = c ∧
    dist D E = d ∧
    dist E A = e ∧
    angle A B C = α ∧
    angle C A D = φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_constructible_l427_42743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l427_42758

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/2

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) + f x = 0) := by
  constructor
  · intro x hx
    sorry -- Proof for f x ≠ 0 when x ≠ 0
  · intro x hx
    sorry -- Proof for f (-x) + f x = 0 when x ≠ 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l427_42758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_iff_equal_l427_42771

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The main theorem -/
theorem floor_inequality_iff_equal (m n : ℕ+) :
  (∀ α β : ℝ, floor ((m + n : ℝ) * α) + floor ((m + n : ℝ) * β) ≥ 
    floor (m * α) + floor (m * β) + floor (n * (α + β))) ↔ m = n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_iff_equal_l427_42771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_count_l427_42752

def f (a x : ℝ) : ℝ := |x + 1| + |a * x + 1|

theorem min_value_count (min_val : ℝ) : 
  min_val = 3/2 → 
  (∃ (S : Finset ℝ), (∀ a ∈ S, ∃ x, f a x = min_val ∧ ∀ y, f a y ≥ min_val) ∧ Finset.card S = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_count_l427_42752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_locus_definition_is_not_correct_l427_42706

-- Define a type for points in a geometric space
variable (Point : Type)

-- Define predicates for being on the locus and satisfying conditions
variable (onLocus : Point → Prop)
variable (satisfiesConditions : Point → Prop)

-- Define the incorrect locus definition
def incorrectLocusDefinition (Point : Type) (onLocus satisfiesConditions : Point → Prop) : Prop :=
  (∀ p : Point, onLocus p → satisfiesConditions p) ∧
  (∃ p : Point, satisfiesConditions p ∧ onLocus p)

-- Theorem stating that the incorrect definition is not equivalent to the correct locus definition
theorem incorrect_locus_definition_is_not_correct (Point : Type) (onLocus satisfiesConditions : Point → Prop) :
  ¬(incorrectLocusDefinition Point onLocus satisfiesConditions ↔ 
    (∀ p : Point, onLocus p ↔ satisfiesConditions p)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_locus_definition_is_not_correct_l427_42706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l427_42746

/-- The function f(x) = 0.8^x - 1 -/
noncomputable def f (x : ℝ) : ℝ := (0.8 : ℝ)^x - 1

/-- The function g(x) = ln x -/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The function h(x) = f(x) - g(x) -/
noncomputable def h (x : ℝ) : ℝ := f x - g x

/-- Theorem: There exists an x in the interval (0, 1) such that h(x) = 0 -/
theorem zero_in_interval : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l427_42746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_boys_is_440_l427_42793

/-- Represents the school population and its composition -/
structure SchoolPopulation where
  total_students : ℕ
  junior_students : ℕ
  junior_boy_girl_ratio : Rat
  senior_girl_boy_diff : ℕ
  senior_boy_count : ℕ
  girl_percentage : Rat

/-- Calculates the total number of boys in the school -/
def total_boys (school : SchoolPopulation) : ℕ :=
  let junior_boys := (school.junior_students * school.junior_boy_girl_ratio.num) /
    (school.junior_boy_girl_ratio.num + school.junior_boy_girl_ratio.den)
  junior_boys.toNat + school.senior_boy_count

/-- Theorem stating that given the conditions, the total number of boys is 440 -/
theorem total_boys_is_440 (school : SchoolPopulation) 
  (h1 : school.total_students = 1000)
  (h2 : school.junior_students = 400)
  (h3 : school.junior_boy_girl_ratio = 3 / 2)
  (h4 : school.senior_girl_boy_diff = 200)
  (h5 : school.senior_boy_count * 2 + 200 = school.total_students - school.junior_students)
  (h6 : school.girl_percentage * (school.total_students - school.junior_students) = 
    (school.senior_boy_count + school.senior_girl_boy_diff) * 100) :
  total_boys school = 440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_boys_is_440_l427_42793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l427_42783

theorem cos_double_angle (θ : Real) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l427_42783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l427_42712

-- Define the circle
def circleSet (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (3, 2)
def M₁ : ℝ × ℝ := (2, 3)
def M₂ : ℝ × ℝ := (2, 4)

-- Define the line y = 0
def line_y_eq_0 : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Theorem statement
theorem circle_theorem :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center ∈ line_y_eq_0 ∧
    A ∈ circleSet center radius ∧
    B ∈ circleSet center radius ∧
    center = (-1, 0) ∧
    radius^2 = 20 ∧
    (M₁.1 - center.1)^2 + (M₁.2 - center.2)^2 < radius^2 ∧
    (M₂.1 - center.1)^2 + (M₂.2 - center.2)^2 > radius^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l427_42712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_l427_42750

/-- Two circles touching externally -/
structure TouchingCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  touch_point : ℝ × ℝ
  h_positive_radii : radius1 > 0 ∧ radius2 > 0
  h_touch_externally : dist center1 center2 = radius1 + radius2
  h_touch_point : dist center1 touch_point = radius1 ∧ dist center2 touch_point = radius2

/-- A point on the first circle -/
def PointOnCircle (tc : TouchingCircles) := { p : ℝ × ℝ // dist tc.center1 p = tc.radius1 }

/-- Tangent from a point to the second circle -/
noncomputable def Tangent (tc : TouchingCircles) (b : PointOnCircle tc) :=
  { t : ℝ × ℝ // dist b.val t = Real.sqrt ((dist tc.center2 b.val)^2 - tc.radius2^2) }

/-- Theorem: The ratio of BA to BT is constant -/
theorem constant_ratio (tc : TouchingCircles) (b : PointOnCircle tc) (t : Tangent tc b) :
  dist b.val tc.touch_point / dist b.val t.val = Real.sqrt (tc.radius1 / (tc.radius1 + tc.radius2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_l427_42750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trapezoid_area_l427_42744

/-- A trapezoid with one base and two sides of length 1 -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  side1 : ℝ
  side2 : ℝ
  base1_eq_one : base1 = 1
  side1_eq_one : side1 = 1
  side2_eq_one : side2 = 1

/-- The area of a trapezoid -/
noncomputable def area (t : Trapezoid) : ℝ :=
  (t.base1 + t.base2) * Real.sqrt (t.side1 * t.side2) / 2

/-- The maximum area of a trapezoid with one base and two sides of length 1 -/
theorem max_trapezoid_area :
  ∃ (t : Trapezoid), ∀ (s : Trapezoid), area t ≥ area s ∧ area t = 3 * Real.sqrt 3 / 4 := by
  sorry

#check max_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trapezoid_area_l427_42744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_min_a_value_l427_42733

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (2 * x)

-- Part 1
theorem cos_2alpha_value (α : ℝ) (h1 : f α = 1/2) (h2 : α ∈ Set.Ioo (5*π/12) (2*π/3)) :
  Real.cos (2 * α) = -(Real.sqrt 3 + Real.sqrt 15) / 8 := by sorry

-- Part 2
theorem min_a_value (a b : ℝ) (h1 : a < b) 
  (h2 : StrictMonoOn f (Set.Icc (a * π) (b * π))) :
  a ≥ 23/12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_min_a_value_l427_42733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l427_42728

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define the new hyperbola
def new_hyperbola (x y : ℝ) : Prop := y^2/4 - x^2/16 = 1

-- Define the point M
noncomputable def point_M : ℝ × ℝ := (2, Real.sqrt 5)

-- Theorem statement
theorem hyperbola_properties :
  -- The new hyperbola passes through point M
  new_hyperbola point_M.1 point_M.2 ∧
  -- The new hyperbola has the same asymptotes as the original hyperbola
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (y/x = 1/2 ∨ y/x = -1/2) →
    (original_hyperbola x y ↔ new_hyperbola x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l427_42728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l427_42729

-- Define the points
def D : ℝ × ℝ := (-5, 3)
def E : ℝ × ℝ := (9, 3)
def F : ℝ × ℝ := (6, -6)

-- Define the function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

-- State the theorem
theorem area_of_triangle_DEF : triangleArea D E F = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l427_42729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_game_winner_l427_42796

/-- Represents the state of the diagonal drawing game on a regular (2n+1)-gon. -/
structure GameState (n : ℕ) where
  (n_gt_one : n > 1)

/-- Represents a player in the game. -/
inductive Player
  | First
  | Second

/-- The initial number of legal moves in the game. -/
def initial_legal_moves (n : ℕ) : ℕ := (2*n + 1) * (n - 1)

/-- Determines the winner of the game based on the parity of n. -/
def winner (n : ℕ) : Player :=
  if n % 2 = 0 then Player.First else Player.Second

/-- Theorem stating the winning strategy for the diagonal drawing game. -/
theorem diagonal_game_winner (n : ℕ) (state : GameState n) :
  winner n = if n % 2 = 0 then Player.First else Player.Second := by
  sorry

/-- Lemma: The number of legal moves changes parity after each turn. -/
lemma legal_moves_parity_change (n : ℕ) (state : GameState n) :
  ∃ k : ℕ, (initial_legal_moves n) % 2 ≠ k % 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_game_winner_l427_42796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l427_42797

theorem solve_exponential_equation (m n : ℤ) : 
  8 * (2^m : ℚ)^n = 64 → |n| = 1 → (m = 3 ∨ m = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l427_42797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l427_42778

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + a / 36)

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x, f a x ∈ Set.univ

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x, 2^x - 4^x < 2*a - 3/4

-- Define the main theorem
theorem range_of_a : 
  {a : ℝ | ¬(p a ∧ q a)} = Set.Iic 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l427_42778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_deriv_is_derivative_l427_42710

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

-- Define the derivative of the function
noncomputable def f_deriv (x : ℝ) : ℝ := -x * Real.sin x

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo π (2 * π), f_deriv x > 0 := by
  sorry

-- Additional theorem to show that f_deriv is indeed the derivative of f
theorem f_deriv_is_derivative :
  ∀ x : ℝ, HasDerivAt f (f_deriv x) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_deriv_is_derivative_l427_42710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l427_42748

theorem congruent_integers_count : 
  (Finset.filter (fun n : Fin 500 => n.val % 8 = 5) Finset.univ).card = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l427_42748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l427_42769

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f(x) is increasing on ℝ for all a ∈ ℝ
theorem f_increasing (a : ℝ) : Monotone (f a) := by
  sorry

-- Theorem 2: f(x) is an odd function if and only if a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x, f a x = -(f a (-x))) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l427_42769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l427_42784

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part I
theorem part_one :
  let a : ℝ := -1
  {x : ℝ | f x a ≤ 2} = Set.Icc (-1/2) (1/2) :=
sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2) 1, f x a ≤ |2*x + 1|) →
  a ∈ Set.Icc 0 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l427_42784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l427_42777

noncomputable def M : ℝ × ℝ := (-5, 0)
noncomputable def N : ℝ × ℝ := (5, 0)

noncomputable def perimeter (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) +
  Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2) +
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)

theorem locus_of_P (P : ℝ × ℝ) (h1 : perimeter P = 36) (h2 : P.2 ≠ 0) :
  (P.1^2 / 169) + (P.2^2 / 144) = 1 := by
  sorry

#check locus_of_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l427_42777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l427_42700

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 16

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 = 8*y

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem smallest_distance :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ parabola_eq x2 y2 ∧
    (∀ (a1 b1 a2 b2 : ℝ), circle_eq a1 b1 → parabola_eq a2 b2 →
      distance x1 y1 x2 y2 ≤ distance a1 b1 a2 b2) ∧
    distance x1 y1 x2 y2 = Real.sqrt 34 - 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l427_42700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l427_42751

def M : Set ℝ := {1, 2, 3, 4}
def N : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

theorem intersection_M_N : M ∩ N = {3, 4} := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l427_42751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_foreign_liquid_l427_42719

/-- Represents the amount of liquid in each container after transfers -/
structure ContainerState where
  wine : ℝ
  water : ℝ

/-- Represents the initial state and transfer amount -/
structure LiquidTransfer where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_b_lt_a : b < a

/-- Calculates the final state of containers after transfers -/
noncomputable def finalState (lt : LiquidTransfer) : ContainerState × ContainerState :=
  let totalVolume := lt.a + lt.b
  let wineInA := lt.a * lt.a / totalVolume
  let waterInA := lt.a * lt.b / totalVolume
  let wineInB := lt.a * lt.b / totalVolume
  let waterInB := (lt.a * lt.a + lt.b * lt.b) / totalVolume
  ({ wine := wineInA, water := waterInA },
   { wine := wineInB, water := waterInB })

/-- Theorem stating that both containers have the same amount of "foreign" liquid after transfers -/
theorem equal_foreign_liquid (lt : LiquidTransfer) :
  let (finalA, finalB) := finalState lt
  finalA.water = finalB.wine := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_foreign_liquid_l427_42719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_value_l427_42782

/-- Triangle DEF with side lengths -/
def triangle_DEF : Triangle := sorry

/-- Rectangle WXYZ inscribed in triangle DEF -/
def rectangle_WXYZ : Rectangle := sorry

/-- Area of rectangle WXYZ as a function of its side length θ -/
def area_WXYZ (θ : ℝ) : ℝ := sorry

/-- Coefficient γ in the area formula -/
noncomputable def γ : ℝ := sorry

/-- Coefficient δ in the area formula -/
noncomputable def δ : ℝ := sorry

theorem delta_value : δ = Real.sqrt 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_value_l427_42782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_theorem_l427_42774

/-- The number of cards in the deck -/
def deck_size : ℕ := 32

/-- The number of ways to draw n cards from a deck of size k when order matters -/
def permutations (k n : ℕ) : ℕ :=
  Nat.factorial k / Nat.factorial (k - n)

/-- The number of ways to draw n cards from a deck of size k when order doesn't matter -/
def combinations (k n : ℕ) : ℕ :=
  permutations k n / Nat.factorial n

theorem card_drawing_theorem :
  (permutations deck_size 2 = 992) ∧
  (combinations deck_size 2 = 496) ∧
  (combinations deck_size 3 = 4960) := by
  sorry

#eval permutations deck_size 2
#eval combinations deck_size 2
#eval combinations deck_size 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_theorem_l427_42774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_coefficients_l427_42789

theorem product_of_coefficients (p q r : ℝ) : 
  (∃ Q : ℝ → ℝ, ∀ x, Q x = x^3 + p*x^2 + q*x + r) →
  (∀ x, x^3 + p*x^2 + q*x + r = 0 ↔ x = Real.cos (π/9) ∨ x = Real.cos (5*π/9) ∨ x = Real.cos (7*π/9)) →
  p * q * r = 1/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_coefficients_l427_42789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_converges_l427_42779

/-- The largest power of 2 that divides n -/
def v2 (n : ℕ+) : ℕ := sorry

/-- a_n is defined as e^(-k) where k is the largest power of 2 that divides n -/
noncomputable def a (n : ℕ+) : ℝ := Real.exp (-v2 n)

/-- b_n is defined as the product of a_i from i=1 to n -/
noncomputable def b (n : ℕ+) : ℝ := (Finset.range n).prod (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The series Σ(n=1 to ∞) b_n converges -/
theorem b_series_converges : ∃ (s : ℝ), Filter.Tendsto (fun n => (Finset.range n).sum (fun i => b ⟨i + 1, Nat.succ_pos i⟩)) Filter.atTop (nhds s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_converges_l427_42779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l427_42714

/-- Represents a batsman's score data -/
structure BatsmanScore where
  inningsBeforeLatest : Nat
  averageBeforeLatest : ℚ
  latestInningScore : ℚ
  averageIncrease : ℚ

/-- Calculates the new average after the latest inning -/
def newAverage (b : BatsmanScore) : ℚ :=
  (b.inningsBeforeLatest * b.averageBeforeLatest + b.latestInningScore) / (b.inningsBeforeLatest + 1)

/-- Theorem: The batsman's new average is 34 runs -/
theorem batsman_new_average (b : BatsmanScore)
  (h1 : b.inningsBeforeLatest = 16)
  (h2 : b.latestInningScore = 82)
  (h3 : b.averageIncrease = 3)
  (h4 : newAverage b = b.averageBeforeLatest + b.averageIncrease) :
  newAverage b = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l427_42714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l427_42737

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  focal_length a b / a

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : focal_length a b = 1) 
  (h4 : ellipse a b (Real.sqrt 6 / 2) 1) :
  -- Part 1: Standard equation
  (∃ (a' b' : ℝ), a' = Real.sqrt 3 ∧ b' = Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- Part 2: Eccentricity range
  (∃ (x y : ℝ), ellipse a b x y ∧ 
    (x + 2)^2 + y^2 = 2 * ((x + 1)^2 + y^2) →
    Real.sqrt 3 / 3 ≤ eccentricity a b ∧ 
    eccentricity a b ≤ Real.sqrt 2 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l427_42737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_3_4_5_l427_42757

/-- The volume of an octahedron formed by joining the centers of the faces of a right rectangular prism -/
noncomputable def octahedron_volume (a b c : ℝ) : ℝ :=
  (a * c) * (b / 2) / 3

/-- Theorem: The volume of an octahedron formed by joining the centers of the faces 
    of a right rectangular prism with dimensions 3 × 4 × 5 is equal to 10 -/
theorem octahedron_volume_3_4_5 : 
  octahedron_volume 3 4 5 = 10 := by
  -- Unfold the definition of octahedron_volume
  unfold octahedron_volume
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_3_4_5_l427_42757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pony_jeans_discount_rate_pony_jeans_discount_rate_proof_l427_42709

/-- The discount rate on Pony jeans given the conditions of the jean sale problem -/
theorem pony_jeans_discount_rate (fox_price pony_price total_savings fox_pairs pony_pairs pony_discount_rate : ℝ) : Prop :=
  let total_discount_rate : ℝ := 22
  let fox_discount_rate : ℝ := total_discount_rate - pony_discount_rate
  fox_price = 15 ∧
  pony_price = 18 ∧
  total_savings = 8.91 ∧
  fox_pairs = 3 ∧
  pony_pairs = 2 ∧
  fox_pairs * fox_price * (fox_discount_rate / 100) +
  pony_pairs * pony_price * (pony_discount_rate / 100) = total_savings →
  pony_discount_rate = 11

theorem pony_jeans_discount_rate_proof :
  ∃ (fox_price pony_price total_savings fox_pairs pony_pairs pony_discount_rate : ℝ),
    pony_jeans_discount_rate fox_price pony_price total_savings fox_pairs pony_pairs pony_discount_rate :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pony_jeans_discount_rate_pony_jeans_discount_rate_proof_l427_42709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_given_angles_l427_42761

/-- A type representing a line in 3D space -/
structure Line where
  -- Define the line structure (e.g., with a point and a direction vector)
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A type representing a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- A type representing an angle -/
def Angle := ℝ

/-- Function to check if a point is on a line -/
def Point.on_line (P : Point) (l : Line) : Prop :=
  sorry

/-- Function to compute the angle between two lines -/
def angle_between (l₁ l₂ : Line) : Angle :=
  sorry

/-- Given a general position line g and a point P not on g, there exists a line l
    passing through P that forms angles t₁₂ with g. -/
theorem line_with_given_angles (g : Line) (P : Point) (t₁₂ : Angle) 
    (h_not_on : ¬ P.on_line g) : 
    ∃ l : Line, P.on_line l ∧ angle_between g l = t₁₂ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_given_angles_l427_42761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_y_terms_l427_42788

theorem coefficient_sum_y_terms (x y : ℝ) : 
  let expanded := (5*x + 3*y + 2) * (2*x + 6*y + 7)
  let y_terms := [36*x*y, 18*y^2, 33*y]
  (y_terms.map (λ term => (Polynomial.coeff (Polynomial.C term) 1))).sum = 87 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_y_terms_l427_42788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l427_42718

-- Define the distance traveled
noncomputable def distance : ℝ := 60

-- Define the time taken
noncomputable def time : ℝ := 5

-- Define speed as distance divided by time
noncomputable def speed : ℝ := distance / time

-- Theorem stating that the speed is 12 miles per hour
theorem car_speed : speed = 12 := by
  -- Unfold the definitions
  unfold speed distance time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l427_42718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_87_l427_42781

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  faces : Fin 6 → ℕ
  consecutive : ∀ i j, i < j → faces i < faces j
  opposite_sum_equal : ∀ i, faces i + faces (5 - i) = faces 0 + faces 5
  one_pair_diff_6 : ∃ i, faces (5 - i) - faces i = 6
  smallest_12 : faces 0 = 12

/-- The sum of all numbers on the cube -/
def cube_sum (c : NumberedCube) : ℕ := (Finset.range 6).sum (λ i => c.faces i)

/-- Theorem stating that the sum of all numbers on the cube is 87 -/
theorem cube_sum_is_87 (c : NumberedCube) : cube_sum c = 87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_87_l427_42781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l427_42767

theorem trig_identity (α : ℝ) : 
  Real.sin (2 * π - α) ^ 2 + Real.cos (π + α) * Real.cos (π - α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l427_42767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilaterals_with_common_midpoints_equal_area_l427_42732

/-- Represents a quadrilateral in 2D geometry -/
structure QuadrilateralGeometry where
  -- Define the structure of a quadrilateral
  -- This is a placeholder and should be properly defined based on your geometric system
  vertices : Fin 4 → ℝ × ℝ

/-- Predicate to check if two quadrilaterals have common midpoints -/
def have_common_midpoints (Q₁ Q₂ : QuadrilateralGeometry) : Prop :=
  -- Define the condition for having common midpoints
  -- This is a placeholder and should be properly defined based on your geometric system
  sorry

/-- Function to calculate the area of a quadrilateral -/
noncomputable def area (Q : QuadrilateralGeometry) : ℝ :=
  -- Define how to calculate the area of a quadrilateral
  -- This is a placeholder and should be properly defined based on your geometric system
  sorry

/-- Two quadrilaterals with common midpoints have equal areas -/
theorem quadrilaterals_with_common_midpoints_equal_area 
  (Q₁ Q₂ : QuadrilateralGeometry) 
  (h : have_common_midpoints Q₁ Q₂) : 
  area Q₁ = area Q₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilaterals_with_common_midpoints_equal_area_l427_42732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l427_42715

-- Define the ellipse (C)
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line (l)
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the condition that line (l) doesn't pass through origin
def not_through_origin (k m : ℝ) : Prop := m ≠ 0

-- Define points A and B as intersections of ellipse and line
def intersection_points (a b k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ a b ∧ line x₁ y₁ k m ∧
    ellipse x₂ y₂ a b ∧ line x₂ y₂ k m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define point M as midpoint of AB
def midpoint_of (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

-- Define the condition on slopes
def slope_condition (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) * (y₀ / x₀) = -3/4

-- Main theorem
theorem ellipse_eccentricity (a b k m : ℝ) :
  a > b ∧ b > 0 ∧
  not_through_origin k m ∧
  intersection_points a b k m ∧
  (∃ x₀ y₀ x₁ y₁ x₂ y₂ : ℝ, 
    midpoint_of x₀ y₀ x₁ y₁ x₂ y₂ ∧
    slope_condition x₀ y₀ x₁ y₁ x₂ y₂) →
  Real.sqrt (1 - b^2 / a^2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l427_42715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_special_square_l427_42787

/-- A square with two vertices on a line and two on a parabola -/
structure SpecialSquare where
  -- Line equation: y = 3x - 20
  line : ℝ → ℝ
  line_eq : line = λ x => 3 * x - 20

  -- Parabola equation: y = x^2
  parabola : ℝ → ℝ
  parabola_eq : parabola = λ x => x^2

  -- Two vertices on the line
  v1_line : ℝ × ℝ
  v2_line : ℝ × ℝ
  v1_on_line : v1_line.2 = line v1_line.1
  v2_on_line : v2_line.2 = line v2_line.1

  -- Two vertices on the parabola
  v1_parabola : ℝ × ℝ
  v2_parabola : ℝ × ℝ
  v1_on_parabola : v1_parabola.2 = parabola v1_parabola.1
  v2_on_parabola : v2_parabola.2 = parabola v2_parabola.1

  -- Square property
  is_square : (v1_line.1 - v2_line.1)^2 + (v1_line.2 - v2_line.2)^2 =
              (v1_parabola.1 - v2_parabola.1)^2 + (v1_parabola.2 - v2_parabola.2)^2

/-- The smallest possible area of a SpecialSquare is 250 -/
theorem smallest_area_special_square :
  ∃ (s : SpecialSquare), ∀ (t : SpecialSquare),
    (s.v1_line.1 - s.v2_line.1)^2 ≤ (t.v1_line.1 - t.v2_line.1)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_special_square_l427_42787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l427_42731

/-- Given two parallel vectors a and b in the plane, prove that |a + 2b| = √10 -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), a = k • b) →
  ‖a + 2 • b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l427_42731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_set_with_non_power_sum_l427_42770

-- Define the property for the set S
def has_non_power_sum_property (S : Set ℕ) : Prop :=
  ∀ (T : Finset ℕ) (k : ℕ),
    T.toSet ⊆ S → k ≥ 2 →
    ¬ ∃ (n : ℕ), (T.sum id) = n^k

-- State the theorem
theorem exists_infinite_set_with_non_power_sum :
  ∃ (S : Set ℕ), Set.Infinite S ∧ has_non_power_sum_property S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_set_with_non_power_sum_l427_42770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l427_42702

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (F₁ F₂ P : ℝ × ℝ),
    let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1};
    F₁ ∈ C ∧ F₂ ∈ C ∧  -- F₁ and F₂ are foci of C
    ‖F₁ - F₂‖ = 6 ∧  -- focal length is 6
    P ∈ C ∧  -- P is on C
    ‖P - F₁‖ + ‖P - F₂‖ = 6 * a ∧  -- sum of distances
    ∃ θ : ℝ, θ = 30 * π / 180 ∧ 
      θ ≤ Real.arccos ((‖P - F₁‖^2 + ‖F₁ - F₂‖^2 - ‖P - F₂‖^2) / (2 * ‖P - F₁‖ * ‖F₁ - F₂‖))  -- smallest angle is 30°
    → a = Real.sqrt 3 ∧ b = Real.sqrt 6 :=
by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l427_42702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_in_triangle_l427_42745

theorem sin_C_in_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
                          (h2 : 0 < B ∧ B < π) (h3 : A + B + C = π)
                          (h4 : Real.cos A = 12/13) (h5 : Real.cos B = 3/5) :
  Real.sin C = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_in_triangle_l427_42745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unselected_digit_is_six_l427_42754

/-- The set of digits from 0 to 9 -/
def Digits : Finset ℕ := Finset.range 10

/-- The sum of all digits from 0 to 9 -/
def DigitsSum : ℕ := Finset.sum Digits id

/-- A function that checks if a number is a valid two-digit number -/
def IsTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A function that checks if a number is a valid three-digit number -/
def IsThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- A function that checks if a number is a valid four-digit number -/
def IsFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The theorem to be proved -/
theorem unselected_digit_is_six
  (selected : Finset ℕ)
  (h_size : selected.card = 9)
  (h_subset : selected ⊆ Digits)
  (a b c : ℕ)
  (h_two_digit : IsTwoDigit a)
  (h_three_digit : IsThreeDigit b)
  (h_four_digit : IsFourDigit c)
  (h_sum : a + b + c = 2010)
  (h_digits : ∀ d ∈ selected, d ∈ Finset.image (λ x => x % 10) Digits) :
  Digits \ selected = {6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unselected_digit_is_six_l427_42754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_liquid_x_percentage_l427_42720

/-- Represents the composition of a solution -/
structure Solution where
  total_mass : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The initial solution Y -/
def initial_solution_y : Solution :=
  { total_mass := 8
  , liquid_x_percent := 30
  , water_percent := 70 }

/-- The amount of water that evaporates -/
def evaporated_water : ℝ := 4

/-- The amount of solution Y added after evaporation -/
def added_solution_y : ℝ := 4

/-- Calculates the mass of liquid X in a solution -/
noncomputable def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.liquid_x_percent / 100

/-- Calculates the mass of water in a solution -/
noncomputable def water_mass (s : Solution) : ℝ :=
  s.total_mass * s.water_percent / 100

/-- Theorem stating that the final percentage of liquid X is 45% -/
theorem final_liquid_x_percentage : 
  let initial_liquid_x := liquid_x_mass initial_solution_y
  let initial_water := water_mass initial_solution_y
  let remaining_water := initial_water - evaporated_water
  let added_liquid_x := liquid_x_mass { initial_solution_y with total_mass := added_solution_y }
  let added_water := water_mass { initial_solution_y with total_mass := added_solution_y }
  let final_liquid_x := initial_liquid_x + added_liquid_x
  let final_water := remaining_water + added_water
  let final_total := final_liquid_x + final_water
  (final_liquid_x / final_total) * 100 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_liquid_x_percentage_l427_42720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_vessel_capacity_l427_42763

/-- Represents a vessel containing a mixture of alcohol and water -/
structure Vessel where
  capacity : ℝ
  alcohol_percentage : ℝ

/-- Calculates the amount of alcohol in a vessel -/
noncomputable def alcohol_amount (v : Vessel) : ℝ := v.capacity * v.alcohol_percentage / 100

theorem mixture_vessel_capacity 
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (total_liquid : ℝ)
  (new_concentration : ℝ)
  (h1 : vessel1.capacity = 2)
  (h2 : vessel1.alcohol_percentage = 35)
  (h3 : vessel2.capacity = 6)
  (h4 : vessel2.alcohol_percentage = 50)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 37)
  : ∃ (final_capacity : ℝ), final_capacity ≥ total_liquid := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_vessel_capacity_l427_42763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l427_42776

theorem triangle_side_length (a b c : ℝ) : 
  0 < a → 0 < b → 0 < c →  -- Positive sides
  a ≤ b → b ≤ c →  -- Ordered sides
  b + a = 9 →  -- Sum of roots
  b * a = 8 →  -- Product of roots
  ∀ x, x^2 - 9*x + 8 = 0 → x = a ∨ x = b →  -- Quadratic equation
  c^2 = b^2 + a^2 - 2*b*a*(Real.cos (π/3)) →  -- Law of Cosines
  c = Real.sqrt 57 :=  -- Length of side BC
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l427_42776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_collinearity_and_minimum_dot_product_l427_42786

/-- Given points A, B, and C in a 2D Cartesian coordinate system, 
    we prove that they are collinear when C has a specific x-coordinate,
    and find the point M on line OC that minimizes the dot product of MA and MB. -/
theorem point_collinearity_and_minimum_dot_product 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 4)) 
  (h_B : B = (2, 3)) :
  ∃ (x : ℝ), 
    (∃ (C : ℝ × ℝ), C = (x, 1) ∧ 
      (∃ (t : ℝ), A = t • B + (1 - t) • C)) ∧
    (let C : ℝ × ℝ := (3, 1)
     ∃ (M : ℝ × ℝ), 
       (∃ (l : ℝ), M = l • C) ∧
       ∀ (N : ℝ × ℝ), (∃ (m : ℝ), N = m • C) → 
         (M - A) • (M - B) ≤ (N - A) • (N - B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_collinearity_and_minimum_dot_product_l427_42786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_tangent_circles_l427_42764

-- Define the circles and their properties
def circle_A_radius : ℝ := 4
def circle_B_radius : ℝ := 9
def distance_between_centers : ℝ := 5

-- Define the area of a circle
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

-- Theorem statement
theorem area_between_tangent_circles :
  let area_A := circle_area circle_A_radius
  let area_B := circle_area circle_B_radius
  area_B - area_A = 65 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_tangent_circles_l427_42764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighter_dog_weight_l427_42723

/-- The weight of the lighter dog -/
def lighter_dog : ℝ := sorry

/-- The weight of the heavier dog -/
def heavier_dog : ℝ := sorry

/-- The weight of the cat -/
def cat : ℝ := sorry

/-- The total weight of all three animals is 36 pounds -/
axiom total_weight : lighter_dog + heavier_dog + cat = 36

/-- The heavier dog and cat weigh three times as much as the lighter dog -/
axiom heavier_dog_cat : heavier_dog + cat = 3 * lighter_dog

/-- The lighter dog and cat weigh twice as much as the heavier dog -/
axiom lighter_dog_cat : lighter_dog + cat = 2 * heavier_dog

/-- The weight of the lighter dog is 9 pounds -/
theorem lighter_dog_weight : lighter_dog = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighter_dog_weight_l427_42723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_l427_42775

noncomputable def f (x : ℝ) : ℝ := (x^3 + 11*x^2 + 38*x + 35) / (x + 3)

noncomputable def g (x : ℝ) : ℝ := x^2 + 8*x + 14

def d : ℝ := -3

theorem function_simplification :
  (∀ x : ℝ, x ≠ d → f x = g x) →
  (1 : ℝ) + 8 + 14 + d = 20 := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_l427_42775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_permutation_set_size_l427_42708

def is_valid_permutation_set (perms : Set (Fin 6 → Fin 6)) : Prop :=
  ∀ p q, p ∈ perms → q ∈ perms → p ≠ q →
    (∃ i : Fin 6, p i = q i) ∧ (∃ j : Fin 6, p j ≠ q j)

theorem max_valid_permutation_set_size :
  ∃ (perms : Finset (Fin 6 → Fin 6)), is_valid_permutation_set perms ∧ 
    Finset.card perms = 120 ∧
    ∀ (S : Finset (Fin 6 → Fin 6)), is_valid_permutation_set S → 
      Finset.card S ≤ 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_permutation_set_size_l427_42708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l427_42756

theorem log_inequality (x : ℝ) (h : x > 1) : Real.log (1 + x) > x / (2 * (1 + x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l427_42756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_outcomes_l427_42794

-- Define the probability of a child being a boy or a girl
def child_probability : ℚ := 1 / 2

-- Define the number of children
def num_children : ℕ := 5

-- Define the probability of all children being the same gender
def prob_all_same_gender : ℚ := child_probability ^ num_children

-- Define the probability of 3 children of one gender and 2 of the other
def prob_three_two : ℚ := (Nat.choose num_children 3) * (child_probability ^ num_children)

-- Define the probability of 4 children of one gender and 1 of the other
def prob_four_one : ℚ := 2 * (Nat.choose num_children 1) * (child_probability ^ num_children)

theorem most_likely_outcomes :
  prob_three_two = prob_four_one ∧
  prob_three_two > prob_all_same_gender ∧
  prob_four_one > prob_all_same_gender := by
  sorry

#eval prob_all_same_gender
#eval prob_three_two
#eval prob_four_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_outcomes_l427_42794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l427_42705

-- Define an odd function f on ℝ
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 4*x - 1 else -(4*(-x) - 1)

-- Theorem statement
theorem f_neg_one_eq_neg_three :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = 4*x - 1) →  -- definition for x ≥ 0
  f (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l427_42705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_i_replacement_l427_42741

def alphabet : List Char := ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def message : String := "Mini's visit is inspiring in Illinois, isn't it?"

def shift_letter (c : Char) (n : Nat) : Char :=
  let index := alphabet.indexOf c
  let new_index := (index + n) % 26
  alphabet[new_index]!

def count_occurrences (s : String) (c : Char) : Nat :=
  s.toList.filter (· == c) |>.length

theorem last_i_replacement (c : Char) :
  c = 'i' →
  count_occurrences message c = 10 →
  shift_letter c (10 * 10) = 'e' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_i_replacement_l427_42741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l427_42730

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / (3 * sequence_a (n + 1) + 1)

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = 1 / (3 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l427_42730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_properties_l427_42768

/-- A circle passing through two points with specific properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_A : (center.1 - 1)^2 + (center.2 + 2)^2 = radius^2
  passes_through_B : (center.1 + 1)^2 + (center.2 - 4)^2 = radius^2

/-- The line 2x - y - 4 = 0 -/
def special_line (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 - 4 = 0

/-- The theorem stating the properties of the special circles -/
theorem special_circle_properties :
  ∃ (c1 c2 : SpecialCircle),
    -- Circle with smallest perimeter
    (c1.center.1 = 0 ∧ c1.center.2 = 1 ∧ c1.radius^2 = 10) ∧
    -- Circle with center on the special line
    (special_line c2.center ∧ c2.center = (3, 2) ∧ c2.radius^2 = 20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_properties_l427_42768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_income_l427_42707

/-- Calculates the annual income from a stock investment --/
noncomputable def annual_income (amount_invested : ℝ) (dividend_rate : ℝ) (market_price : ℝ) : ℝ :=
  let nominal_value := (amount_invested / market_price) * 100
  nominal_value * (dividend_rate / 100)

/-- Theorem: The annual income from investing $6800 in stock with a 60% dividend rate
    and a market price of $136 per $100 of nominal value is $3000 --/
theorem stock_investment_income :
  annual_income 6800 60 136 = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_income_l427_42707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_not_f_one_l427_42790

/-- A quadratic function f(x) = ax^2 + bx + c that is symmetric about x=2 -/
def symmetric_quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

/-- The property that f(2+t) = f(2-t) for all real t -/
def is_symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f (2 + t) = f (2 - t)

theorem smallest_value_not_f_one (a b c : ℝ) :
  is_symmetric_about_two (symmetric_quadratic a b c) →
  ∃ x ∈ ({-1, 2, 5} : Set ℝ), (symmetric_quadratic a b c) x ≤ (symmetric_quadratic a b c) 1 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_not_f_one_l427_42790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_I_holds_law_II_does_not_hold_l427_42717

-- Define the @ operation
def op_at (a b : ℝ) : ℝ := a + 2 * b

-- Define the # operation
def op_hash (a b : ℝ) : ℝ := 2 * a - b

-- Theorem for Law I
theorem law_I_holds (x y z : ℝ) : 
  op_at x (op_hash y z) = op_hash (op_at x y) (op_at x z) := by
  -- Expand definitions and simplify
  simp [op_at, op_hash]
  -- Prove equality by arithmetic
  ring

-- Theorem for Law II
theorem law_II_does_not_hold : 
  ∃ x y z : ℝ, x + op_at y z ≠ op_at (x + y) (x + z) := by
  -- Provide a counterexample
  use 1, 1, 1
  -- Expand definitions and simplify
  simp [op_at]
  -- Prove inequality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_I_holds_law_II_does_not_hold_l427_42717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l427_42785

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to A, B, C respectively

-- Define our specific triangle
noncomputable def our_triangle : Triangle :=
  { A := Real.pi / 3,  -- 60° in radians
    B := 0,      -- We'll prove this is π/6 (30°)
    C := 0,      -- We don't need to specify C
    a := Real.sqrt 3,
    b := 1,
    c := 0 }     -- We don't need to specify c

-- State the theorem
theorem angle_B_is_30_degrees (t : Triangle) (h1 : t.A = Real.pi / 3) (h2 : t.a = Real.sqrt 3) (h3 : t.b = 1) :
  t.B = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l427_42785
