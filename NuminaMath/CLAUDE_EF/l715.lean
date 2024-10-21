import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l715_71574

/-- The volume of a pyramid with given base dimensions and height -/
noncomputable def pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_length * base_width * height

/-- Theorem stating the volume of a specific pyramid -/
theorem specific_pyramid_volume :
  let base_length : ℝ := 4
  let base_width : ℝ := 8
  let height : ℝ := 10
  abs (pyramid_volume base_length base_width height - 106.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l715_71574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_max_l715_71555

noncomputable def sequence_term (n : ℝ) : ℝ := (n - 2017.5) / (n - 2016.5)

theorem sequence_min_max :
  (∃ (n : ℝ), sequence_term n = -1) ∧
  (∃ (n : ℝ), sequence_term n = 3) ∧
  (∀ (n : ℝ), sequence_term n ≥ -1) ∧
  (∀ (n : ℝ), sequence_term n ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_max_l715_71555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_complex_numbers_l715_71583

/-- The measure of the angle between two complex numbers on the complex plane -/
noncomputable def angle_between (z₁ z₂ : ℂ) : ℝ :=
  Real.arctan ((z₂.im * z₁.re - z₁.im * z₂.re) / (z₁.re * z₂.re + z₁.im * z₂.im))

theorem angle_between_specific_complex_numbers :
  angle_between (2 + Complex.I) (1 / (3 + Complex.I)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_complex_numbers_l715_71583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l715_71580

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem third_vertex_coordinates (x : ℝ) :
  let p1 : Point := ⟨4, -5⟩
  let p2 : Point := ⟨0, 0⟩
  let p3 : Point := ⟨x, 0⟩
  x < 0 ∧ triangleArea p1 p2 p3 = 40 → x = -16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l715_71580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excited_cells_count_l715_71529

-- Define the state of a cell
inductive CellState
  | Rest
  | Excited

-- Define the chain of cells
def CellChain := ℤ → CellState

-- Define the number of 1s in the binary representation of a natural number
def countOnes : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1).mod 2 + countOnes (n / 2)

-- Define the evolution of the cell chain over time
def evolve : CellChain → ℕ → CellChain := sorry

-- Define the number of excited cells at a given time
def excitedCount (chain : CellChain) : ℕ := sorry

-- The main theorem
theorem excited_cells_count (t : ℕ) :
  let initialChain : CellChain := fun i => if i = 0 then CellState.Excited else CellState.Rest
  excitedCount (evolve initialChain t) = 2^(countOnes t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excited_cells_count_l715_71529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_implies_k_range_l715_71566

open Real

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := exp x / x + k/2 * x^2 - k*x

-- Define the derivative of f(x)
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := (exp x / x^2 + k) * (x - 1)

-- Theorem statement
theorem unique_critical_point_implies_k_range :
  ∀ k : ℝ, 
  (∀ x : ℝ, x > 0 → x ≠ 1 → f_deriv k x ≠ 0) →
  (f_deriv k 1 = 0) →
  k ≥ -exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_implies_k_range_l715_71566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_two_l715_71540

open Real

/-- The volume of n cones formed from n congruent sectors of a unit circle -/
noncomputable def total_volume (n : ℕ) : ℝ :=
  (Real.pi / 3) * (sqrt ((n : ℝ)^2 - 1) / (n : ℝ)^2)

/-- The theorem stating that the total volume is maximized when n = 2 -/
theorem volume_maximized_at_two :
  ∀ n : ℕ, n ≥ 2 → total_volume n ≤ total_volume 2 := by
  sorry

#check volume_maximized_at_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_two_l715_71540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_AC_l715_71556

open Real

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * sin (π - x) * cos (-x) + sin (π + x) * cos (π/2 - x)

/-- Theorem stating the dot product of vectors AB and AC -/
theorem dot_product_AB_AC : 
  ∃ (A B C : ℝ × ℝ), 
    (∀ x, f A.1 ≤ f x) ∧ 
    (∀ x, x ≠ B.1 → x ≠ C.1 → f x ≤ f B.1) ∧
    (∀ x, |x - A.1| < |B.1 - A.1| → x = A.1 ∨ f x < f B.1) ∧
    (∀ x, |x - A.1| < |C.1 - A.1| → x = A.1 ∨ f x < f C.1) ∧
    ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 4 - π^2/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_AC_l715_71556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l715_71589

/-- The circle described by the equation x^2 + y^2 - 2x + 2y = 2 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 2

/-- The line described by the equation 3x + 4y - 14 = 0 -/
def Line (x y : ℝ) : Prop :=
  3*x + 4*y - 14 = 0

/-- The distance from a point (x, y) to the line 3x + 4y - 14 = 0 -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |3*x + 4*y - 14| / Real.sqrt (3^2 + 4^2)

theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 1 ∧
  ∀ (x y : ℝ), Circle x y →
    d ≤ distanceToLine x y ∧
    ∃ (x' y' : ℝ), Circle x' y' ∧ distanceToLine x' y' = d := by
  sorry

#check min_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l715_71589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_properties_l715_71577

def is_strictly_increasing (a : Fin 2023 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

def median (a : Fin 2023 → ℝ) : ℝ := a ⟨1012, by norm_num⟩

noncomputable def average (a : Fin 2023 → ℝ) : ℝ :=
  (Finset.sum Finset.univ (λ i => a i)) / 2023

noncomputable def variance (a : Fin 2023 → ℝ) : ℝ :=
  (Finset.sum Finset.univ (λ i => (a i - average a) ^ 2)) / 2023

theorem data_properties (a : Fin 2023 → ℝ) (h : is_strictly_increasing a) :
  (median a = a ⟨1012, by norm_num⟩) ∧
  (average (λ i => a i + 2) = average a + 2) ∧
  (variance (λ i => 2 * a i + 1) = 4 * variance a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_properties_l715_71577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_reality_and_equality_l715_71516

theorem quadratic_roots_reality_and_equality (q n p : ℝ) :
  let discriminant := (q - n)^2 + 4 * p^2
  ∀ x : ℝ, x^2 - (q + n) * x + (q * n - p^2) = 0 → 
    ((∃ (y : ℝ), y ≠ x ∧ y^2 - (q + n) * y + (q * n - p^2) = 0) ↔ discriminant > 0) ∧
    ((∀ (y : ℝ), y^2 - (q + n) * y + (q * n - p^2) = 0 → y = x) ↔ q = n ∧ p = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_reality_and_equality_l715_71516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_tangent_l715_71567

/-- Given a circle with diameter AB, an equilateral triangle ABC, a point D on AB 
    such that AD = (2/n) * AB, and a line CD intersecting the circle at E, 
    where ∠AOE = α, prove the following equation. -/
theorem circle_triangle_tangent (n : ℝ) (α : ℝ) 
  (h1 : n ≠ 4) -- Ensure denominator is not zero
  (h2 : n^2 + 16*n - 32 ≥ 0) -- Ensure square root is real
  : Real.tan α = (Real.sqrt (n^2 + 16*n - 32) - n) / (n - 4) * (Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_tangent_l715_71567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l715_71561

theorem rectangle_area_theorem (x y : ℝ) :
  (Real.sqrt (x - y) = 2/5 ∧ Real.sqrt (x + y) = 2) →
  ((x = 0 ∧ y = 0) ∨
   (x = 2 ∧ y = 2) ∨
   (x = 2/25 ∧ y = -2/25) ∨
   (x = 52/25 ∧ y = 48/25)) ∧
  (let area := 2 * Real.sqrt 2 * (2/25) * Real.sqrt 2
   area = 8/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l715_71561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l715_71511

def S : Set ℝ := {x : ℝ | x ≠ 0}

theorem function_equation_solution 
  (f : ℝ → ℝ) 
  (c : ℝ) 
  (hc : c ≠ 0)
  (h : ∀ (x y : ℝ), x ∈ S → y ∈ S → x + y ≠ 0 → f x + f y = c * f (x * y * f (x + y)))
  (hf : ∀ x, x ∈ S → f x ∈ S) :
  (∀ (x : ℝ), x ∈ S → f x = (1 : ℝ) / x) ∧ c = 1 ∧ f 5 = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l715_71511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_college_cost_calculation_l715_71507

-- Define the parameters
def total_credits : ℕ := 18
def regular_credits : ℕ := 12
def lab_credits : ℕ := 6
def regular_credit_cost : ℚ := 450
def lab_credit_cost : ℚ := 550
def textbook_count : ℕ := 3
def textbook_cost : ℚ := 150
def online_resource_count : ℕ := 4
def online_resource_cost : ℚ := 95
def facilities_fee : ℚ := 200
def lab_fee : ℚ := 75
def regular_scholarship_rate : ℚ := 1/2
def lab_discount_rate : ℚ := 1/4
def interest_rate : ℚ := 1/25

-- Define the theorem
theorem college_cost_calculation :
  (let regular_class_cost := regular_credits * regular_credit_cost
   let lab_class_cost := lab_credits * lab_credit_cost
   let textbook_total_cost := textbook_count * textbook_cost
   let online_resource_total_cost := online_resource_count * online_resource_cost
   let lab_fee_total := lab_credits * lab_fee
   let total_cost := regular_class_cost + lab_class_cost + textbook_total_cost + 
                     online_resource_total_cost + facilities_fee + lab_fee_total
   let scholarship := regular_scholarship_rate * regular_class_cost
   let discount := lab_discount_rate * lab_class_cost
   let adjusted_cost := total_cost - scholarship - discount
   let interest := interest_rate * adjusted_cost
   let final_cost := adjusted_cost + interest
   final_cost) = 5881.2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_college_cost_calculation_l715_71507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l715_71551

theorem first_class_students (avg_first : ℚ) (avg_second : ℚ) (num_second : ℕ) (avg_total : ℚ) :
  avg_first = 40 →
  avg_second = 60 →
  num_second = 45 →
  avg_total = 205/4 →
  ∃ (num_first : ℕ), 
    (avg_first * num_first + avg_second * num_second) / (num_first + num_second) = avg_total ∧
    num_first = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l715_71551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_share_l715_71595

/-- Helper function to round to the nearest cent -/
def round_to_nearest_cent (x : ℚ) : ℚ :=
  (⌊x * 100 + 1/2⌋) / 100

/-- Calculate each person's share of a restaurant bill -/
theorem restaurant_bill_share (total_bill : ℚ) (num_people : ℕ) (tip_percent : ℚ) : 
  total_bill = 139 ∧ num_people = 6 ∧ tip_percent = 1/10 →
  round_to_nearest_cent ((total_bill + total_bill * tip_percent) / num_people) = 2548/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_share_l715_71595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_positions_l715_71500

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -3)

/-- The radius of the circle -/
def radius : ℝ := 5

/-- Point A -/
def A : ℝ × ℝ := (0, 0)

/-- Point B -/
def B : ℝ × ℝ := (-2, 1)

/-- Point C -/
def C : ℝ × ℝ := (3, 3)

/-- Point D -/
def D : ℝ × ℝ := (2, -1)

/-- The squared distance between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The circle with equation (x-1)^2 + (y+3)^2 = 25 -/
def inCircle (p : ℝ × ℝ) : Prop := distanceSquared p center ≤ radius^2

theorem circle_points_positions :
  inCircle A ∧
  distanceSquared B center = radius^2 ∧
  ¬inCircle C ∧
  inCircle D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_positions_l715_71500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_property_iff_form_l715_71554

-- Define the number of divisors function
def d (n : ℕ) : ℕ := sorry

-- Define the property we want to prove
def has_property (n : ℕ) : Prop :=
  d (n^3) = 5 * d n

-- Define the structure of n we're looking for
def is_correct_form (n : ℕ) : Prop :=
  ∃ (p₁ p₂ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁^3 * p₂

-- State the theorem
theorem divisor_property_iff_form (n : ℕ) :
  has_property n ↔ is_correct_form n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_property_iff_form_l715_71554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l715_71542

/-- A parabola with focus on the line x - 2y - 4 = 0 -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_line : focus.1 - 2 * focus.2 - 4 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation where
  | vertical (a : ℝ) : StandardEquation
  | horizontal (a : ℝ) : StandardEquation

/-- Predicate to check if a StandardEquation is vertical with coefficient a -/
def is_vertical (eq : StandardEquation) (a : ℝ) : Prop :=
  match eq with
  | StandardEquation.vertical x => x = a
  | _ => False

/-- Predicate to check if a StandardEquation is horizontal with coefficient a -/
def is_horizontal (eq : StandardEquation) (a : ℝ) : Prop :=
  match eq with
  | StandardEquation.horizontal x => x = a
  | _ => False

/-- The theorem stating the possible standard equations for the parabola -/
theorem parabola_standard_equation (p : Parabola) :
  (∃ (eq : StandardEquation), is_vertical eq 16) ∨
  (∃ (eq : StandardEquation), is_horizontal eq (-8)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l715_71542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_no_perpendicular_intersections_l715_71522

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1/4

-- Define the foci of the ellipse
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the dot product of vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Part 1: Point P satisfies the conditions
theorem point_P_satisfies_conditions :
  let P : ℝ × ℝ := (1, Real.sqrt 3 / 2)
  ellipse_C P.1 P.2 ∧
  P.1 > 0 ∧ P.2 > 0 ∧
  dot_product (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) = -5/4 :=
by sorry

-- Part 2: No line l exists such that OA ⊥ OB
theorem no_perpendicular_intersections :
  ∀ (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ),
  (∃ (x y : ℝ), circle_O x y ∧ l x y) →
  (ellipse_C A.1 A.2 ∧ l A.1 A.2) →
  (ellipse_C B.1 B.2 ∧ l B.1 B.2) →
  dot_product A B ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_no_perpendicular_intersections_l715_71522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_nine_three_l715_71588

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_through_point_nine_three (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = powerFunction α x) →  -- f is a power function with exponent α
  f 9 = 3 →                         -- f passes through the point (9, 3)
  f 100 = 10 :=                     -- prove that f(100) = 10
by
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_nine_three_l715_71588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bound_function_maximum_l715_71599

-- Part 1
theorem function_bound {a x : ℝ} (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  let f := λ y => a * y^2 + y - a
  |f x| ≤ 5/4 := by sorry

-- Part 2
theorem function_maximum :
  ∃ a : ℝ, let f := λ x => a * x^2 + x - a
  (∃ x, |x| ≤ 1 ∧ f x = 17/8) ∧ a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bound_function_maximum_l715_71599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QO_is_three_l715_71557

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def F : ℝ × ℝ := (2, 0)

-- Define the directrix l
def l (x : ℝ) : Prop := x = -2

-- Define a point P on the directrix
noncomputable def P : ℝ × ℝ := (-2, Real.sqrt 128)

-- Define Q as the intersection of PF and C
noncomputable def Q : ℝ × ℝ := (1, Real.sqrt 128 / 4)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the vector from F to P
noncomputable def FP : ℝ × ℝ := (P.1 - F.1, P.2 - F.2)

-- Define the vector from F to Q
noncomputable def FQ : ℝ × ℝ := (Q.1 - F.1, Q.2 - F.2)

-- State the theorem
theorem distance_QO_is_three (h1 : C Q.1 Q.2) (h2 : l P.1) 
  (h3 : FP = (4 * FQ.1, 4 * FQ.2)) : 
  Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QO_is_three_l715_71557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_alignment_and_angle_l715_71509

/-- Given three points A, B, C in a plane rectangular coordinate system lying on a straight line,
    prove that m = 1, n = 2, and cos ∠AOC = -√5/5 under the given conditions. -/
theorem point_alignment_and_angle (m n : ℝ) : 
  let A : ℝ × ℝ := (-3, m + 1)
  let B : ℝ × ℝ := (n, 3)
  let C : ℝ × ℝ := (7, 4)
  let O : ℝ × ℝ := (0, 0)
  let G : ℝ × ℝ := ((1/3) * (-3 + 7), (1/3) * ((m + 1) + 4))
  -- A, B, C are collinear
  (∃ (t : ℝ), B.1 - A.1 = t * (C.1 - B.1) ∧ B.2 - A.2 = t * (C.2 - B.2)) →
  -- OA ⊥ OB
  (A.1 * B.1 + A.2 * B.2 = 0) →
  -- G is the centroid of triangle AOC
  (G = ((1/3) * (O.1 + A.1 + C.1), (1/3) * (O.2 + A.2 + C.2))) →
  -- OG = (2/3)OB
  (G = ((2/3) * B.1, (2/3) * B.2)) →
  (m = 1 ∧ n = 2 ∧ 
   (A.1 * C.1 + A.2 * C.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (C.1^2 + C.2^2)) = -Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_alignment_and_angle_l715_71509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_implies_right_angle_l715_71563

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the incircle radius
noncomputable def incircleRadius (t : Triangle) : ℝ := (t.b + t.c - t.a) / 2

-- Define angle A
noncomputable def angleA (t : Triangle) : ℝ := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- Theorem statement
theorem incircle_radius_implies_right_angle (t : Triangle) 
  (h : incircleRadius t = (t.b + t.c - t.a) / 2) : 
  angleA t = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_implies_right_angle_l715_71563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l715_71570

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^6
  else if x ≤ -1 then -2*x - 1
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem coefficient_of_x_squared (x : ℝ) (h : x ≤ -1) :
  ∃ (a b c d e : ℝ), f (f x) = 60 * x^2 + a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x + 1 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l715_71570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l715_71562

/-- Calculate the overall gain percentage for three items sold with different profit/loss margins. -/
theorem overall_gain_percentage (sp1 sp2 sp3 : ℝ) (g1 g2 l3 : ℝ) : 
  sp1 = 100 →
  sp2 = 150 →
  sp3 = 200 →
  g1 = 0.20 →
  g2 = 0.15 →
  l3 = 0.05 →
  let cp1 := sp1 / (1 + g1)
  let cp2 := sp2 / (1 + g2)
  let cp3 := sp3 / (1 - l3)
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  let gain := tsp - tcp
  let gain_percentage := (gain / tcp) * 100
  abs (gain_percentage - 6.06) < 0.01 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l715_71562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l715_71591

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f x = f (f (f x) + y) + f (x * f y) * f (x + y)

/-- The main theorem stating that any function satisfying the equation must be of the form c/x -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x > 0 → f x = c / x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l715_71591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_rotation_is_reflection_l715_71544

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angles α, β, γ
def Triangle.angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define a rotation
def rotate (point center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

-- Define the incircle center
def Triangle.incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a reflection
def reflect (point center : ℝ × ℝ) : ℝ × ℝ := sorry

theorem composite_rotation_is_reflection 
  (t : Triangle) (α β γ : ℝ) (h : α + β + γ = π) :
  let B₁ := rotate t.B t.A α
  let B₂ := rotate B₁ t.B β
  let B₃ := rotate B₂ t.C γ
  let O := Triangle.incenter t
  let midpoint := ((t.B.1 + B₃.1) / 2, (t.B.2 + B₃.2) / 2)
  B₃ = reflect t.B midpoint := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_rotation_is_reflection_l715_71544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_intensity_reduction_l715_71515

/-- The intensity reduction factor for one piece of glass -/
noncomputable def intensity_reduction : ℝ := 0.9

/-- The threshold fraction of original intensity -/
noncomputable def threshold : ℝ := 1/4

/-- Approximation of lg 2 -/
noncomputable def lg2_approx : ℝ := 0.3

/-- Approximation of lg 3 -/
noncomputable def lg3_approx : ℝ := 0.477

/-- The minimum number of pieces of glass required to reduce light intensity below the threshold -/
def min_pieces : ℕ := 14

/-- Theorem stating the light intensity reduction properties -/
theorem light_intensity_reduction (k : ℝ) (k_pos : k > 0) :
  (∀ x < min_pieces, k * intensity_reduction ^ x > threshold * k) ∧
  (k * intensity_reduction ^ min_pieces < threshold * k) := by
  sorry

#check light_intensity_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_intensity_reduction_l715_71515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_a_theorem_l715_71512

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isObtuse (t : Triangle) : Prop := sorry
def sideABGreaterThanAC (t : Triangle) : Prop := sorry
def angleBIs45Degrees (t : Triangle) : Prop := sorry

-- Define the circumcenter and incenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define sin A
noncomputable def sinA (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_sin_a_theorem (t : Triangle) 
  (h1 : isObtuse t)
  (h2 : sideABGreaterThanAC t)
  (h3 : angleBIs45Degrees t)
  (h4 : Real.sqrt 2 * distance (circumcenter t) (incenter t) = 
        distance t.A t.B - distance t.A t.C) :
  sinA t = Real.sqrt 2 / 2 ∨ sinA t = Real.sqrt (Real.sqrt 2 - 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_a_theorem_l715_71512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_validColorings_nonneg_l715_71587

/-- The number of valid colorings for an n-sided polygon --/
def validColorings (n : ℕ) : ℤ :=
  2^n + 2 * (-1 : ℤ)^n

/-- Theorem: The number of valid colorings for an n-sided polygon
    where n ≥ 3 and no two adjacent sides share the same color
    out of three possible colors is 2ⁿ + 2 · (-1)ⁿ --/
theorem polygon_coloring (n : ℕ) (h : n ≥ 3) :
  validColorings n = (2 : ℤ)^n + 2 * (-1 : ℤ)^n :=
by
  sorry

/-- Verify that the formula gives a non-negative integer for all n ≥ 3 --/
theorem validColorings_nonneg (n : ℕ) (h : n ≥ 3) :
  (validColorings n).toNat ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_validColorings_nonneg_l715_71587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_statement_correct_l715_71532

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if two points determine a unique line
def two_points_determine_line (p1 p2 : Point) : Prop :=
  ∃! (l : Line), l.a * p1.x + l.b * p1.y + l.c = 0 ∧ l.a * p2.x + l.b * p2.y + l.c = 0

-- Define a function for the shortest distance between two points
noncomputable def shortest_distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define a function for opposite numbers
def are_opposite (a b : ℝ) : Prop :=
  a = -b

-- Theorem stating that only the first statement is correct
theorem only_first_statement_correct :
  (∀ (p1 p2 : Point), p1 ≠ p2 → two_points_determine_line p1 p2) ∧
  (∃ (p1 p2 : Point), ¬(shortest_distance p1 p2 = 0 → p1 = p2)) ∧
  (∃ (a : ℝ), |a| = -a ∧ a = 0) ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ are_opposite a b ∧ a / b ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_statement_correct_l715_71532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l715_71552

/-- Represents the simple interest calculation for a loan -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Proves that the principal is 5000 given the conditions of the problem -/
theorem principal_calculation (principal rate time interest : ℝ) 
  (h_rate : rate = 4)
  (h_time : time = 10)
  (h_interest : interest = 2000)
  (h_simple_interest : simple_interest principal rate time = interest) :
  principal = 5000 := by
  sorry

#check principal_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l715_71552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l715_71518

-- Define the variables
variable (W : ℝ)  -- Capacity of the pool in cubic meters
variable (V1 : ℝ)  -- Rate of the first valve in cubic meters per minute
variable (V2 : ℝ)  -- Rate of the second valve in cubic meters per minute

-- Define the conditions
axiom both_valves : V1 + V2 = W / 48
axiom first_valve : V1 = W / 120
axiom valve_difference : V2 = V1 + 50

-- Theorem to prove
theorem pool_capacity : W = 12000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l715_71518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l715_71547

/-- Probability of getting more heads than tails when flipping n coins -/
def Probability.of_getting_more_heads_than_tails (n : ℕ) : ℚ :=
  sorry -- Definition to be implemented

theorem coin_flip_probability (n : ℕ) (p : ℚ) : 
  n = 10 → p = 193 / 512 → 
  p = Probability.of_getting_more_heads_than_tails n :=
by sorry

#check coin_flip_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l715_71547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l715_71569

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * (Real.sin ((t.A + t.B) / 2))^2 = Real.sin t.C + 1 ∧
  t.a = Real.sqrt 2 ∧
  t.c = 1

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = π / 4 ∧ (1 / 2 : ℝ) * t.b * t.c = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l715_71569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l715_71539

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_perimeter_20 (a b c : ℕ) : Prop :=
  a + b + c = 20

def is_non_congruent (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  t1 ≠ t2 ∧ 
  t1 ≠ (t2.2.1, t2.2.2, t2.1) ∧ 
  t1 ≠ (t2.2.2, t2.1, t2.2.1)

theorem count_non_congruent_triangles : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)),
    (∀ t ∈ triangles, 
      is_valid_triangle t.1 t.2.1 t.2.2 ∧ 
      has_perimeter_20 t.1 t.2.1 t.2.2) ∧
    (∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → is_non_congruent t1 t2) ∧
    triangles.length = 8 ∧
    (∀ a b c : ℕ, 
      is_valid_triangle a b c → 
      has_perimeter_20 a b c → 
      (a, b, c) ∈ triangles ∨ 
      (b, c, a) ∈ triangles ∨ 
      (c, a, b) ∈ triangles) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l715_71539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_bead_necklaces_l715_71559

/-- Proves that Megan sold 7 bead necklaces given the conditions of the problem -/
theorem megan_bead_necklaces :
  ∀ (bead_necklaces : ℕ),
    (bead_necklaces + 3) * 9 = 90 →
    bead_necklaces = 7 := by
  intro bead_necklaces hypothesis
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_bead_necklaces_l715_71559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l715_71525

theorem dihedral_angle_cosine (r R : ℝ) (α β : ℝ) : 
  r > 0 ∧ R > 0 ∧  -- Both radii are positive
  R = 2 * r ∧      -- One radius is twice the other
  β = π / 4 ∧      -- The angle between centers line and edge is 45°
  (R - r) / (Real.sin (α / 2)) = (R + r) * Real.sin β  -- Geometric relationship
  →
  Real.cos α = 8 / 9 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l715_71525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l715_71582

/-- The growth rate from t=0 to t=2 -/
def growth_rate_0_2 : ℝ := 0.10

/-- The growth rate from t=2 to t=4 -/
def growth_rate_2_4 : ℝ := 0.20

/-- The growth rate from t=4 to t=6 -/
def growth_rate_4_6 : ℝ := 2 * growth_rate_2_4

/-- The total growth factor from t=0 to t=6 -/
def total_growth_factor : ℝ :=
  (1 + growth_rate_0_2) * (1 + growth_rate_2_4) * (1 + growth_rate_4_6)

/-- The total percentage increase from t=0 to t=6 -/
def total_percentage_increase : ℝ := (total_growth_factor - 1) * 100

theorem population_growth_theorem :
  ∃ ε > 0, |total_percentage_increase - 84.8| < ε := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l715_71582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_ricotta_for_short_side_cylinder_l715_71517

/-- Represents the dimensions of the dough rectangle in centimeters -/
structure DoughRectangle where
  length : ℝ
  width : ℝ

/-- Represents a cylinder formed from the dough rectangle -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Creates a cylinder by joining the longer sides of the rectangle -/
noncomputable def makeLongSideCylinder (d : DoughRectangle) (overlap : ℝ) : Cylinder :=
  { radius := (d.length - overlap) / (2 * Real.pi)
  , height := d.width }

/-- Creates a cylinder by joining the shorter sides of the rectangle -/
noncomputable def makeShortSideCylinder (d : DoughRectangle) (overlap : ℝ) : Cylinder :=
  { radius := (d.width - overlap) / (2 * Real.pi)
  , height := d.length }

/-- Converts volume to ricotta mass -/
noncomputable def volumeToRicotta (volume : ℝ) (initialVolume : ℝ) (initialMass : ℝ) : ℝ :=
  (volume * initialMass) / initialVolume

theorem more_ricotta_for_short_side_cylinder 
  (d : DoughRectangle) 
  (overlap : ℝ) 
  (initialRicottaMass : ℝ) : 
  let longCylinder := makeLongSideCylinder d overlap
  let shortCylinder := makeShortSideCylinder d overlap
  let longVolume := cylinderVolume longCylinder
  let shortVolume := cylinderVolume shortCylinder
  let ricottaDifference := volumeToRicotta shortVolume longVolume initialRicottaMass - initialRicottaMass
  d.length = 16 ∧ 
  d.width = 12 ∧ 
  overlap = 2 ∧ 
  initialRicottaMass = 500 →
  ricottaDifference = 235 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_ricotta_for_short_side_cylinder_l715_71517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_initial_peanuts_l715_71558

/-- The number of peanuts Carol initially collected. -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Carol's father gave her. -/
def fathers_peanuts : ℕ := 5

/-- The total number of peanuts Carol has after receiving peanuts from her father. -/
def total_peanuts : ℕ := 7

/-- Theorem stating that Carol initially collected 2 peanuts. -/
theorem carol_initial_peanuts : 
  initial_peanuts + fathers_peanuts = total_peanuts → initial_peanuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_initial_peanuts_l715_71558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_y_l715_71504

open Real

-- Define the original function
noncomputable def y (x : ℝ) : ℝ := (x^2 + 3) * log (x - 3)

-- State the theorem
theorem fourth_derivative_of_y (x : ℝ) (h : x ≠ 3) :
  (deriv^[4] y) x = (-2 * x^2 + 24 * x - 126) / (x - 3)^4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_y_l715_71504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_at_P_is_C_l715_71585

-- Define the type for people
inductive Person : Type
  | A | B | C | D

-- Define the visibility relation
def canSee (x y : Person) : Prop := 
  match x, y with
  | Person.A, _ => False
  | Person.B, Person.C => True
  | Person.B, _ => False
  | Person.C, Person.B => True
  | Person.C, Person.C => False  -- Added missing case
  | Person.C, Person.D => True
  | Person.C, Person.A => False
  | Person.D, Person.C => True
  | Person.D, _ => False

-- Define the point P
def P : Person := Person.C

-- Theorem statement
theorem person_at_P_is_C :
  (∀ y, ¬canSee Person.A y) ∧
  (∀ y, canSee Person.B y ↔ y = Person.C) ∧
  (∀ y, canSee Person.C y ↔ (y = Person.B ∨ y = Person.D)) ∧
  (∀ y, canSee Person.D y ↔ y = Person.C) →
  P = Person.C := by
  intro h
  rfl  -- reflexivity proves P = Person.C


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_at_P_is_C_l715_71585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l715_71513

theorem problem_solution (m n : ℝ) (h1 : (10 : ℝ)^m = 5) (h2 : ((10 : ℝ)^n)^2 = 2) : 
  m + 2*n - 3 = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l715_71513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_quadrilateral_l715_71505

/-- Represents a right-angled triangle with sides a, b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angled : a^2 + b^2 = c^2

/-- Represents the quadrilateral PTRQ -/
structure Quadrilateral where
  pt : ℝ
  tr : ℝ
  rq : ℝ
  pq : ℝ

def original_triangle : RightTriangle :=
  { a := 3
    b := 4
    c := 5
    right_angled := by sorry }

noncomputable def removed_triangle : RightTriangle :=
  { a := 2
    b := Real.sqrt 21
    c := 5
    right_angled := by sorry }

noncomputable def resulting_quadrilateral : Quadrilateral :=
  { pt := original_triangle.a - removed_triangle.a
    tr := removed_triangle.b
    rq := original_triangle.b
    pq := original_triangle.a }

theorem perimeter_of_quadrilateral :
  resulting_quadrilateral.pt + resulting_quadrilateral.tr +
  resulting_quadrilateral.rq + resulting_quadrilateral.pq = 8 + Real.sqrt 21 := by
  sorry

#eval "Theorem statement added successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_quadrilateral_l715_71505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_increasing_l715_71560

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubic_root_increasing :
  ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_increasing_l715_71560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_school_time_l715_71526

/-- Represents the walking parameters of a person -/
structure WalkingParams where
  steps_per_minute : ℝ
  step_length : ℝ

/-- Calculates the time taken to cover a distance at a given speed -/
noncomputable def time_taken (distance speed : ℝ) : ℝ := distance / speed

theorem jill_school_time (dave : WalkingParams) (jill : WalkingParams) (dave_time : ℝ) :
  dave.steps_per_minute = 100 →
  dave.step_length = 80 →
  dave_time = 20 →
  jill.steps_per_minute = 110 →
  jill.step_length = 70 →
  let dave_speed := dave.steps_per_minute * dave.step_length
  let distance := dave_speed * dave_time
  let jill_speed := jill.steps_per_minute * jill.step_length
  abs (time_taken distance jill_speed - 20.78) < 0.01 := by
  sorry

#check jill_school_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_school_time_l715_71526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_exists_l715_71506

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1 and y-intercept m -/
structure Line where
  m : ℝ

/-- Defines the conditions for the ellipse E -/
def ellipse_conditions (E : Ellipse) : Prop :=
  E.a^2 = 4 ∧ E.b^2 = 2 ∧ E.a^2 / 4 + E.b^2 / 2 = 1

/-- Defines the intersection of the line with the ellipse -/
def line_intersects_ellipse (l : Line) (E : Ellipse) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    y₁ = x₁ + l.m ∧ y₂ = x₂ + l.m

/-- Theorem stating the existence of a right triangle -/
theorem right_triangle_exists (E : Ellipse) (l : Line) :
  ellipse_conditions E →
  line_intersects_ellipse l E →
  (∃ (C : ℝ), C ≠ 0 ∧ 
    let A := (x₁, x₁ + l.m)
    let B := (x₂, x₂ + l.m)
    let C := (0, C)
    ((A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2)) ↔
  l.m = 3 * Real.sqrt 10 / 5 ∨ l.m = -3 * Real.sqrt 10 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_exists_l715_71506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l715_71581

theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - 4*a + 5) (-b^2 + 2*b - 6)
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l715_71581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_arithmetic_mean_l715_71596

noncomputable def first_set : List ℝ := [16, 20, 42]
noncomputable def second_set : List ℝ := [34, 51]

noncomputable def arithmetic_mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

theorem double_arithmetic_mean :
  arithmetic_mean (arithmetic_mean first_set :: second_set) = 37 := by
  -- Expand the definition of arithmetic_mean
  unfold arithmetic_mean
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- Complete the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_arithmetic_mean_l715_71596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l715_71553

theorem min_value_cubic_inverse (y : ℝ) (hy : y > 0) : 
  3 * y^3 + 4 * y^(-2 : ℝ) ≥ 7 ∧ 
  (3 * y^3 + 4 * y^(-2 : ℝ) = 7 ↔ y = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l715_71553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_2500_l715_71524

/-- Represents the simple interest calculation for a given principal, rate, and time. -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- 
Proves that if increasing the interest rate by 2% over 5 years results in Rs. 250 more interest,
then the principal must be Rs. 2500.
-/
theorem principal_is_2500 
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Original interest rate
  (h : simpleInterest P (R + 2) 5 = simpleInterest P R 5 + 250) -- Condition from the problem
  : P = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_2500_l715_71524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_properties_l715_71530

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 21 = 0

-- Define the point P
def P : ℝ × ℝ := (-6, 7)

-- Define the reflection property
def reflects_off_x_axis (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l x y ↔ l x (-y)

-- Define the tangency property
def is_tangent_to_circle (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ circle_eq x y ∧ ∀ x' y', l x' y' → circle_eq x' y' → (x', y') = (x, y)

-- Main theorem
theorem light_ray_properties :
  ∃ l : ℝ → ℝ → Prop,
    (l P.1 P.2) ∧
    (reflects_off_x_axis l) ∧
    (is_tangent_to_circle l) ∧
    ((∀ x y, l x y ↔ 3*x + 4*y - 10 = 0) ∨ (∀ x y, l x y ↔ 4*x + 3*y + 3 = 0)) ∧
    (∃ x y, l x y ∧ circle_eq x y ∧ Real.sqrt ((x - P.1)^2 + (y - P.2)^2) = 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_properties_l715_71530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l715_71538

/-- Helper function to calculate the area of a triangle given its sides and angles. -/
noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area is √3 when a = 2 and (2+b)(sin A - sin B) = (c-b)sin C. -/
theorem max_area_triangle_abc (a b c : ℝ) (A B C : ℝ) : 
  a = 2 → 
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  (∀ b' c' A' B' C',
    a = 2 → 
    (2 + b') * (Real.sin A' - Real.sin B') = (c' - b') * Real.sin C' →
    area_triangle a b' c' A' B' C' ≤ area_triangle a b c A B C) →
  area_triangle a b c A B C = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l715_71538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l715_71573

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ 3) :
  2 * x^(-(1/3 : ℝ)) / (x^((2/3 : ℝ)) - 3 * x^(-(1/3 : ℝ))) - 
  x^((2/3 : ℝ)) / (x^((5/3 : ℝ)) - x^((2/3 : ℝ))) - 
  (x + 1) / (x^2 - 4*x + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l715_71573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_g_properties_l715_71533

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given function f
noncomputable def f (x : ℝ) : ℝ := lg (x + 1)

-- Define h as the square root of f
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (f x)

-- Define the set of real numbers [0, +∞)
def nonnegative_reals : Set ℝ := { x | x ≥ 0 }

-- Statement 1: The domain of h is [0, +∞)
theorem h_domain : Set.range h = nonnegative_reals := by sorry

-- Define g as an even function on [-1, 1]
noncomputable def g (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then lg (1 - x)
  else if 0 ≤ x ∧ x ≤ 1 then lg (x + 1)
  else 0  -- undefined outside [-1, 1]

-- Statement 2: g is even and matches the given piecewise definition
theorem g_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, g (-x) = g x) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) 1, g x = f x) ∧
  (∀ x ∈ Set.Ioc (-1 : ℝ) 0, g x = lg (1 - x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_g_properties_l715_71533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l715_71531

theorem coefficient_x_cubed_in_expansion : 
  ∀ (coeffs : List ℤ), 
    (coeffs.sum = 64) → 
    (coeffs.length = 7) →
    (∀ k, 0 ≤ k ∧ k < coeffs.length → coeffs.get! k = (-1)^k * Nat.choose 6 k * 3^(6-k)) →
    coeffs.get! 3 = -540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l715_71531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_menus_l715_71546

/-- Represents the days of the week -/
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving Fintype, DecidableEq

/-- Represents the dessert options -/
inductive Dessert
| Cake | Pie | IceCream | Pudding
deriving Fintype, DecidableEq

/-- A dessert menu for a week -/
def WeekMenu := Day → Dessert

/-- Checks if two desserts are different -/
def different_desserts (d1 d2 : Dessert) : Prop := d1 ≠ d2

/-- Checks if a menu satisfies the consecutive day constraint -/
def valid_consecutive_days (menu : WeekMenu) : Prop :=
  different_desserts (menu Day.Sunday) (menu Day.Monday) ∧
  different_desserts (menu Day.Monday) (menu Day.Tuesday) ∧
  different_desserts (menu Day.Tuesday) (menu Day.Wednesday) ∧
  different_desserts (menu Day.Wednesday) (menu Day.Thursday) ∧
  different_desserts (menu Day.Thursday) (menu Day.Friday) ∧
  different_desserts (menu Day.Friday) (menu Day.Saturday) ∧
  different_desserts (menu Day.Saturday) (menu Day.Sunday)

/-- Checks if a menu satisfies all constraints -/
def valid_menu (menu : WeekMenu) : Prop :=
  menu Day.Friday = Dessert.Cake ∧
  menu Day.Monday = Dessert.Pie ∧
  valid_consecutive_days menu

instance : Fintype WeekMenu := by sorry

instance : DecidablePred valid_menu := by sorry

/-- The main theorem stating the number of valid dessert menus -/
theorem number_of_valid_menus :
  (Finset.filter valid_menu (Finset.univ : Finset WeekMenu)).card = 972 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_menus_l715_71546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_ages_sum_l715_71548

theorem consecutive_ages_sum (ages : List ℕ) : 
  ages.length = 7 ∧ 
  (∀ i j, i < j → i < ages.length → j < ages.length → ages[i]! + 1 = ages[j]!) ∧
  (ages.take 3).sum = 42 →
  (ages.reverse.take 3).sum = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_ages_sum_l715_71548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grain_cracker_price_calculation_l715_71568

/-- The price of a pack of grain crackers -/
noncomputable def grain_cracker_price : ℚ := 2.25

/-- The number of movie tickets in the reference set -/
def movie_tickets : ℕ := 6

/-- The number of grain cracker packs sold per reference set -/
def grain_crackers_sold : ℕ := 3

/-- The number of beverage bottles sold per reference set -/
def beverages_sold : ℕ := 4

/-- The price of each beverage bottle -/
def beverage_price : ℚ := 1.5

/-- The number of chocolate bars sold per reference set -/
def chocolate_bars_sold : ℕ := 4

/-- The price of each chocolate bar -/
def chocolate_bar_price : ℚ := 1

/-- The average snack sales per movie ticket -/
def average_snack_sales : ℚ := 2.79

theorem grain_cracker_price_calculation :
  (grain_cracker_price * grain_crackers_sold + 
   beverage_price * beverages_sold + 
   chocolate_bar_price * chocolate_bars_sold) / movie_tickets = average_snack_sales ∧
  grain_cracker_price = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grain_cracker_price_calculation_l715_71568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l715_71586

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ c : ℝ, c = -15 ∧ 
   c = (Nat.choose 5 1) * a * (x^2)^4 * (a/x)^1 ∧
   7 = 10 - 3 * 1) → 
  a = -3 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l715_71586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_at_5_l715_71543

def sequenceA (n : ℕ) : ℤ :=
  -56 + 12 * (n - 1)

def sum_n_terms (n : ℕ) : ℤ :=
  n * sequenceA 1 + (n * (n - 1) * 12) / 2

theorem sum_minimized_at_5 :
  ∀ k : ℕ, k ≥ 1 → sum_n_terms 5 ≤ sum_n_terms k := by
  sorry

#eval sum_n_terms 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_at_5_l715_71543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_over_reals_l715_71590

/-- Two polynomials with the same complex roots and same complex roots when subtracted by 1 are equal over the reals -/
theorem polynomials_equal_over_reals 
  (P Q : Polynomial ℂ) 
  (hP : P.degree > 0) 
  (hQ : Q.degree > 0) 
  (h0 : {z : ℂ | P.eval z = 0} = {z : ℂ | Q.eval z = 0})
  (h1 : {z : ℂ | P.eval z = 1} = {z : ℂ | Q.eval z = 1}) :
  ∀ x : ℝ, P.eval (x : ℂ) = Q.eval (x : ℂ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_over_reals_l715_71590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_positive_l715_71527

theorem sine_sum_positive (x : Real) (h : 0 < x ∧ x < Real.pi) :
  Real.sin x + (1/2) * Real.sin (2*x) + (1/3) * Real.sin (3*x) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_positive_l715_71527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l715_71508

/-- A translation in the complex plane that shifts 1 - 3i to 6 + 2i -/
def translation (z : ℂ) : ℂ := z + (6 + 2*Complex.I - (1 - 3*Complex.I))

/-- Theorem stating that the translation applied to 2 - i results in 7 + 4i -/
theorem translation_result : translation (2 - Complex.I) = 7 + 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l715_71508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l715_71593

theorem polynomial_remainder : ∃ q : Polynomial ℝ, 
  X^12 - X^6 + 1 = (X^2 - 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l715_71593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ls_parallel_pq_l715_71523

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points on the circle
variable (A C P Q : ℝ × ℝ)

-- Define other points
variable (L S I M N : ℝ × ℝ)

-- Define lines as pairs of points
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := {x : ℝ × ℝ | ∃ t : ℝ, x = (1 - t) • p + t • q}

-- Define angle
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define angle bisector
def angle_bisector (p q r : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define parallel lines
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem ls_parallel_pq 
  (h_circle : A ∈ circle ∧ C ∈ circle ∧ P ∈ circle ∧ Q ∈ circle)
  (h_L : L ∈ line A P ∧ L ∈ line C M)
  (h_S : S ∈ line A N ∧ S ∈ line C Q)
  (h_I : I ∈ line C P ∧ I ∈ line C Q ∧ 
         angle_bisector P C Q = line C I) :
  parallel (line L S) (line P Q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ls_parallel_pq_l715_71523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem_l715_71503

theorem min_value_problem (a b : ℝ) (h1 : a > b) (h2 : b > 1) 
  (h3 : 2 * (Real.log b / Real.log a) + 3 * (Real.log a / Real.log b) = 7) : 
  ∀ x y : ℝ, x > y ∧ y > 1 ∧ 2 * (Real.log y / Real.log x) + 3 * (Real.log x / Real.log y) = 7 → 
  a + 1 / (b^2 - 1) ≤ x + 1 / (y^2 - 1) ∧ 
  ∃ a0 b0 : ℝ, a0 > b0 ∧ b0 > 1 ∧ 2 * (Real.log b0 / Real.log a0) + 3 * (Real.log a0 / Real.log b0) = 7 ∧ 
  a0 + 1 / (b0^2 - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem_l715_71503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l715_71501

/-- Represents a rectangle with given length and width in meters -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle in square meters -/
noncomputable def areaInSquareMeters (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Converts square meters to hectares -/
noncomputable def squareMetersToHectares (area : ℝ) : ℝ :=
  area / 10000

theorem rectangle_area_theorem (r : Rectangle) 
  (h1 : r.length = 500) 
  (h2 : r.width = 60) : 
  areaInSquareMeters r = 30000 ∧ squareMetersToHectares (areaInSquareMeters r) = 3 := by
  sorry

#check rectangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l715_71501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_two_must_be_true_l715_71584

def Digit := Fin 10

structure Envelope :=
  (digit : Digit)

def Statement (e : Envelope) : Fin 4 → Prop
  | 0 => e.digit.val = 5
  | 1 => e.digit.val ≠ 6
  | 2 => e.digit.val = 7
  | 3 => e.digit.val ≠ 8

theorem statement_two_must_be_true (e : Envelope) :
  (∃ (i : Fin 4), ¬Statement e i) ∧ 
  (∀ (i j k : Fin 4), i ≠ j → i ≠ k → j ≠ k → Statement e i ∧ Statement e j ∧ Statement e k) →
  Statement e 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_two_must_be_true_l715_71584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l715_71571

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

-- State the theorem
theorem g_neither_even_nor_odd :
  ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l715_71571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_thirds_l715_71597

theorem power_negative_two_thirds (a : ℝ) (h : a ≠ 0) :
  a^(-(2/3 : ℝ)) = 1 / Real.sqrt (a^3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_thirds_l715_71597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_is_pi_l715_71545

/-- The smallest value of t such that the graph of r = sin θ for 0 ≤ θ ≤ t is the entire circle -/
noncomputable def smallest_t : ℝ := Real.pi

/-- The graph of r = sin θ is a circle -/
axiom is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = Real.sin θ ∧ (∃ x y : ℝ, x^2 + y^2 = r^2)

theorem smallest_t_is_pi :
  ∀ t : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ) →
  (∀ r : ℝ, -1 ≤ r ∧ r ≤ 1 → ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ r = Real.sin θ) →
  t ≥ smallest_t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_is_pi_l715_71545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_analysis_l715_71536

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x

-- Theorem statement
theorem extreme_point_analysis (a : ℝ) :
  (∃ x, x > 0 ∧ deriv (f a) x = 0 ∧ x = 2) →
  (a = -8 ∧
   f a 2 = 4 - 8 * Real.log 2 ∧
   (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1),
     f a x ≥ 4 - 8 * Real.log 2 ∧
     f a x ≤ 1/(Real.exp 1)^2 + 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_analysis_l715_71536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l715_71550

-- Define the line (noncomputable due to use of real numbers)
noncomputable def line (t : ℝ) : ℝ × ℝ := (2 + t, 4 - t)

-- Define the curve (circle) (noncomputable due to use of Real.sqrt and trigonometric functions)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 + Real.sqrt 2 * Real.cos θ, 5 + Real.sqrt 2 * Real.sin θ)

-- Theorem statement
theorem line_curve_intersection :
  ∃! p : ℝ × ℝ, ∃ t θ : ℝ, line t = p ∧ curve θ = p :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l715_71550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_and_angle_l715_71537

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

def line1 : Line2D := ⟨(1, 3), (4, -3)⟩
def line2 : Line2D := ⟨(2, -1), (2, 3)⟩

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / (vector_magnitude v * vector_magnitude w)

theorem lines_intersection_and_angle :
  ∃ (t u : ℝ),
    t = 11/18 ∧ 
    u = 13/18 ∧
    line1.point.1 + t * line1.direction.1 = line2.point.1 + u * line2.direction.1 ∧
    line1.point.2 + t * line1.direction.2 = line2.point.2 + u * line2.direction.2 ∧
    abs (cos_angle line1.direction line2.direction) = 1 / (5 * Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_and_angle_l715_71537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_theorem_l715_71534

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 9

-- Define the equation of the common chord
def commonChordEq (x y : ℝ) : Prop := 2*x - 2*(Real.sqrt 3)*y - 3 = 0

-- Define the length of the common chord
noncomputable def commonChordLength : ℝ := 3 * Real.sqrt 7 / 2

theorem common_chord_theorem :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → commonChordEq x y) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    commonChordEq x1 y1 ∧ commonChordEq x2 y2 ∧
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = commonChordLength) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_theorem_l715_71534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_m_value_l715_71598

/-- Given an ellipse with equation x²/(3m) + y²/m = 1 and focus at (1,0), prove that m = 1/2 -/
theorem ellipse_focus_m_value (m : ℝ) 
  (ellipse_eq : ∀ (x y : ℝ), x^2 / (3*m) + y^2 / m = 1) 
  (focus : ℝ × ℝ) (h_focus : focus = (1, 0)) : m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_m_value_l715_71598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_count_l715_71576

theorem solution_pairs_count : 
  let count := Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 7 * p.2 = 800 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 201 ×ˢ Finset.range 115)
  Finset.card count = 29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_count_l715_71576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l715_71535

theorem diophantine_equation_solutions (x y n : ℕ) :
  1 + 2^x + 2^(2*x + 1) = y^n ↔
    ((x = 4 ∧ y = 23 ∧ n = 2) ∨
     (∃ t : ℕ, t > 0 ∧ x = t ∧ y = 1 + 2^t + 2^(2*t + 1) ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l715_71535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l715_71549

/-- A marking configuration for a 9x9 grid --/
def Marking := Fin 9 → Fin 9 → Bool

/-- Checks if a 1x5 strip contains a marked cell --/
def strip_marked (m : Marking) (row col : Fin 9) (horizontal : Bool) : Prop :=
  ∃ i : Fin 5, 
    if horizontal then
      m row (col + i)
    else
      m (row + i) col

/-- A valid marking satisfies the strip condition for all strips --/
def valid_marking (m : Marking) : Prop :=
  ∀ row col : Fin 9, 
    strip_marked m row col true ∧ 
    strip_marked m row col false

/-- Counts the number of marked cells in a marking --/
def count_marked (m : Marking) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 9)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 9)) (λ j => 
      if m i j then 1 else 0))

/-- The main theorem: the minimum number of marked cells is 16 --/
theorem min_marked_cells :
  (∃ m : Marking, valid_marking m ∧ count_marked m = 16) ∧
  (∀ m : Marking, valid_marking m → count_marked m ≥ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l715_71549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_sail_velocity_l715_71521

/-- Represents the relationship between pressure, area, and velocity for wind on a sail -/
structure WindSail where
  k : ℝ
  P : ℝ → ℝ → ℝ
  h : ∀ A V, P A V = k * A * V^2

/-- Given initial conditions and final pressure and area, calculates the final velocity -/
noncomputable def final_velocity (ws : WindSail) (P1 A1 V1 P2 A2 : ℝ) : ℝ :=
  Real.sqrt ((P2 * A1 * V1^2) / (P1 * A2))

/-- Theorem stating that under given conditions, the final velocity is 16 -/
theorem wind_sail_velocity (ws : WindSail) 
    (h1 : ws.P 4 8 = 4) 
    (h2 : final_velocity ws 4 4 8 64 16 = 16) : 
  ws.P 16 16 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_sail_velocity_l715_71521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l715_71592

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -exp x - x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 3 * a * x + 2 * cos x

-- Define the derivatives of f and g
noncomputable def f' (x : ℝ) : ℝ := -exp x - 1
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a - 2 * sin x

-- State the theorem
theorem perpendicular_tangents_condition (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, (f' x) * (g' a y) = -1) ↔ 
  -1/3 ≤ a ∧ a ≤ 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l715_71592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l715_71520

theorem purely_imaginary_condition (a : ℝ) : 
  (↑(2 - a * Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ (↑(2 - a * Complex.I) / (1 + Complex.I)).re = 0 ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l715_71520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_x_l715_71579

noncomputable def f (a b : ℝ) : ℝ := 1/a + 4/b

-- Theorem for the minimum value of f(a, b)
theorem min_value_f :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 →
  f a b ≥ 9 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ f a₀ b₀ = 9 :=
by
  sorry

-- Theorem for the range of x
theorem range_of_x :
  ∀ x : ℝ,
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → f a b ≥ |2*x - 1| - |x + 1|) →
  -7 ≤ x ∧ x ≤ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_x_l715_71579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_properties_l715_71564

/-- A line parallel to y = -2x passing through (1, 2) -/
structure ParallelLine where
  k : ℚ
  b : ℚ
  parallel_condition : k = -2
  point_condition : 2 = k * 1 + b

/-- The intersection point of the line with the x-axis -/
def intersection (line : ParallelLine) : ℚ × ℚ :=
  (line.b / 2, 0)

theorem parallel_line_properties (line : ParallelLine) :
  line.b = 4 ∧ intersection line = (2, 0) := by
  sorry

#check parallel_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_properties_l715_71564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_sampling_types_l715_71519

-- Define the sampling methods
structure SamplingMethod where
  description : String

-- Define the types of sampling
inductive SamplingType
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the survey methods used
def method1 : SamplingMethod :=
  { description := "Random interviews with 24 students by student council members" }

def method2 : SamplingMethod :=
  { description := "Numbering students from 001 to 240 and selecting those with last digit 3" }

-- Helper function (defined as an axiom for now)
axiom sampling_type_of : SamplingMethod → SamplingType

-- Theorem to prove
theorem survey_sampling_types :
  ∃ (type1 type2 : SamplingType),
    (type1 = SamplingType.SimpleRandom ∧
     type2 = SamplingType.Systematic) ∧
    (type1 = sampling_type_of method1 ∧
     type2 = sampling_type_of method2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_sampling_types_l715_71519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l715_71575

/-- Calculate the area of a quadrilateral given its four vertices. -/
def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral with vertices at (1, 2), (1, 1), (3, 1), and (5, 5) is 6 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (3, 1)
  let D : ℝ × ℝ := (5, 5)
  let quad_area := area_quadrilateral A B C D
  quad_area = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l715_71575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l715_71510

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def triangleABC (y : ℝ) : Triangle where
  A := (-7, 0)
  B := (1, 0)
  C := (0, y)  -- C is on the y-axis, so x-coordinate is 0

-- Define the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.1 - t.A.1) * t.C.2) / 2

-- Theorem statement
theorem triangle_coordinates :
  ∀ (y : ℝ),
  let t := triangleABC y
  triangleArea t = 8 →
  (t.C = (0, 2) ∨ t.C = (0, -2)) :=
by
  intro y t h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l715_71510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_and_irreducibility_l715_71594

theorem polynomial_existence_and_irreducibility 
  (n : ℕ) 
  (m : Fin n → ℤ) 
  (h_distinct : ∀ (i j : Fin n), i ≠ j → m i ≠ m j) :
  ∃ (f : Polynomial ℤ), 
    (Polynomial.degree f = n) ∧ 
    (∀ (i : Fin n), Polynomial.eval (m i) f = -1) ∧ 
    Irreducible f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_and_irreducibility_l715_71594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equality_l715_71578

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt (2/75) = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equality_l715_71578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l715_71502

/-- Given a triangle ABC and a point D in its plane such that BC = 3CD, 
    prove that AD = -1/3 AB + 4/3 AC -/
theorem vector_relation (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  (C - B) = 3 • (D - C) → 
  (D - A) = -(1/3) • (B - A) + (4/3) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l715_71502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l715_71528

/-- Given an interest rate and time period, calculates the simple interest on a principal amount -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Given an interest rate and time period, calculates the compound interest on a principal amount -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating that if the difference between compound and simple interest
    is 15 for a 5% interest rate over 2 years, the principal amount is 6000 -/
theorem interest_difference_implies_principal :
  ∀ P : ℝ,
  compoundInterest P 5 2 - simpleInterest P 5 2 = 15 →
  P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l715_71528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_theorem_l715_71572

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Determines if a triangle is acute-angled --/
noncomputable def isAcuteAngled (t : Triangle) : Bool :=
  sorry

/-- The locus of centers of rectangles described around a triangle --/
noncomputable def locusOfCenters (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- A curvilinear triangle formed by arcs of semicircles on midlines --/
noncomputable def curvilinearTriangle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- Two arcs of semicircles on two midlines --/
noncomputable def twoArcsOfSemicircles (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

theorem locus_of_centers_theorem (t : Triangle) :
  locusOfCenters t = 
    if isAcuteAngled t then curvilinearTriangle t
    else twoArcsOfSemicircles t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_theorem_l715_71572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_participation_theorem_l715_71514

theorem debate_participation_theorem (g : ℕ) : 
  let b := g  -- Equal number of boys and girls
  let girls_participating := (2 : ℚ) / 3 * g
  let boys_participating := (3 : ℚ) / 5 * b
  let total_participating := girls_participating + boys_participating
  girls_participating / total_participating = (30 : ℚ) / 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_participation_theorem_l715_71514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_num_values_g_3_sum_values_g_3_product_num_and_sum_l715_71565

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * g y - x) = x * y - g x

/-- The theorem stating that there are exactly two functions satisfying the equation -/
theorem two_solutions :
  ∃! (f₁ f₂ : ℝ → ℝ), FunctionalEquation f₁ ∧ FunctionalEquation f₂ ∧
    (∀ g : ℝ → ℝ, FunctionalEquation g → (g = f₁ ∨ g = f₂)) ∧
    f₁ = (λ x ↦ x) ∧ f₂ = (λ x ↦ -x) := by
  sorry

/-- The number of possible values for g(3) is 2 -/
theorem num_values_g_3 : 
  ∃! n : ℕ, n = 2 ∧ 
    ∀ g : ℝ → ℝ, FunctionalEquation g → 
      ∃ (S : Finset ℝ), S.card = n ∧ (g 3 ∈ S) := by
  sorry

/-- The sum of all possible values of g(3) is 0 -/
theorem sum_values_g_3 :
  ∃! s : ℝ, s = 0 ∧
    ∀ g₁ g₂ : ℝ → ℝ, FunctionalEquation g₁ ∧ FunctionalEquation g₂ ∧ g₁ ≠ g₂ →
      g₁ 3 + g₂ 3 = s := by
  sorry

/-- The product of the number of possible values and their sum is 0 -/
theorem product_num_and_sum :
  ∃! p : ℝ, p = 0 ∧
    ∃ (n : ℕ) (s : ℝ),
      (∃! m : ℕ, m = n ∧ 
        ∀ g : ℝ → ℝ, FunctionalEquation g → 
          ∃ (S : Finset ℝ), S.card = m ∧ (g 3 ∈ S)) ∧
      (∃! t : ℝ, t = s ∧
        ∀ g₁ g₂ : ℝ → ℝ, FunctionalEquation g₁ ∧ FunctionalEquation g₂ ∧ g₁ ≠ g₂ →
          g₁ 3 + g₂ 3 = t) ∧
      p = n * s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_num_values_g_3_sum_values_g_3_product_num_and_sum_l715_71565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_seven_times_prime_l715_71541

theorem divisors_of_seven_times_prime (p : ℕ) (h_prime : Nat.Prime p) :
  let n := 7 * p
  (Finset.card (Nat.divisors n)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_seven_times_prime_l715_71541
