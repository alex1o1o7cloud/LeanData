import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l659_65940

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) - abs (x - 1)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a * x^2 - 3 * x + 3) / x

-- Theorem for the range of f
theorem range_of_f : Set.range f = Set.Icc (-3) 3 := by sorry

-- Theorem for the range of a
theorem range_of_a (h : ∀ (a : ℝ), a > 0 → ∀ (s t : ℝ), s ≥ 1 → t ≥ 0 → g a s ≥ f t) :
  {a : ℝ | a > 0 ∧ ∀ (s t : ℝ), s ≥ 1 → t ≥ 0 → g a s ≥ f t} = Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l659_65940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_number_with_limited_twos_l659_65963

/-- Represents the count of a specific digit in a number --/
def digitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Represents the total count of a specific digit in a range of numbers --/
def totalDigitCount (start : ℕ) (finish : ℕ) (d : ℕ) : ℕ := sorry

/-- The theorem stating the highest number reachable with limited twos --/
theorem highest_number_with_limited_twos :
  ∀ n : ℕ, (totalDigitCount 1 n 2 ≤ 25 ∧ totalDigitCount 1 (n + 1) 2 > 25) → n = 152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_number_with_limited_twos_l659_65963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outlier_count_l659_65982

def dataset : List ℚ := [8, 20, 36, 36, 44, 46, 46, 48, 56, 62]
def Q1 : ℚ := 36
def Q2 : ℚ := 45
def Q3 : ℚ := 48

def IQR : ℚ := Q3 - Q1

def lowerThreshold : ℚ := Q1 - 1.5 * IQR
def upperThreshold : ℚ := Q3 + 1.5 * IQR

noncomputable def isOutlier (x : ℚ) : Bool :=
  x < lowerThreshold || x > upperThreshold

noncomputable def countOutliers (data : List ℚ) : Nat :=
  data.filter isOutlier |>.length

theorem outlier_count : countOutliers dataset = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outlier_count_l659_65982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_four_l659_65960

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (D : ℝ × ℝ), 
    -- CD is an altitude
    D.1 = C.1 ∧ D.2 = B.2 ∧
    -- Triangle is 45°-45°-90°
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    -- CD = 2
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = 4

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem statement
theorem triangle_area_is_four (A B C : ℝ × ℝ) :
  Triangle A B C → TriangleArea A B C = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_four_l659_65960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l659_65958

-- Define the function f(x) = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, 0)

-- Define the tangent line function
def tangent_line (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem tangent_triangle_area : 
  let x_intercept := 1  -- The x-intercept is at x = 1
  let y_intercept := 1  -- The y-intercept is at y = 1
  (1/2 : ℝ) * x_intercept * y_intercept = 1/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l659_65958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l659_65975

/-- The ellipse (C) in the cartesian coordinate plane -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-4, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (4, 0)

/-- PF₁ is perpendicular to PF₂ -/
def PF₁_perp_PF₂ (P : ℝ × ℝ) : Prop :=
  let PF₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let PF₂ := (F₂.1 - P.1, F₂.2 - P.2)
  PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2 = 0

/-- The area of triangle PF₁F₂ -/
noncomputable def area_PF₁F₂ (P : ℝ × ℝ) : ℝ :=
  let PF₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let PF₂ := (F₂.1 - P.1, F₂.2 - P.2)
  abs (PF₁.1 * PF₂.2 - PF₁.2 * PF₂.1) / 2

theorem ellipse_area_theorem :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → PF₁_perp_PF₂ P → area_PF₁F₂ P = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l659_65975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l659_65945

/-- The distance between foci of a hyperbola -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between foci of the hyperbola x²/16 - y²/9 = 1 is 10 -/
theorem hyperbola_foci_distance :
  distance_between_foci 4 3 = 10 := by
  sorry

#check hyperbola_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l659_65945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_f_sign_l659_65950

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ  -- circumradius
  r : ℝ  -- inradius
  h_sides : a ≤ b ∧ b ≤ c

-- Define the function f
def f (t : Triangle) : ℝ := t.a + t.b - 2 * t.R - 2 * t.r

-- Define angle C
noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem triangle_f_sign (t : Triangle) :
  (f t > 0 ↔ angle_C t < π / 2) ∧
  (f t = 0 ↔ angle_C t = π / 2) ∧
  (f t < 0 ↔ angle_C t > π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_f_sign_l659_65950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l659_65953

/-- The time taken for two bugs to meet again at their starting point on tangent circles -/
theorem bugs_meet_time (r₁ r₂ v₁ v₂ : ℝ) (hr₁ : r₁ = 3) (hr₂ : r₂ = 7) 
  (hv₁ : v₁ = 4 * Real.pi) (hv₂ : v₂ = 6 * Real.pi) : 
  Nat.lcm 
    (Nat.floor ((2 * r₁ * Real.pi) / v₁)) 
    (Nat.floor ((2 * r₂ * Real.pi) / v₂)) = 21 := by
  sorry

#check bugs_meet_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l659_65953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_non_leap_year_l659_65976

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- Defines the action Alice takes based on her die roll -/
def action (roll : DieRoll) : Bool :=
  match roll with
  | .two | .three | .seven => false  -- Drinks tea (prime numbers except 5)
  | .four | .five | .six => false    -- Drinks coffee (composite numbers and 5)
  | .one => false                    -- Drinks tea (1 is considered prime in this context)
  | .eight => true                   -- Rolls again

/-- The probability of each roll on a fair eight-sided die -/
def rollProbability : DieRoll → ℚ
  | _ => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℚ :=
  8/7

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
noncomputable def expectedRollsPerYear : ℚ :=
  expectedRollsPerDay * daysInNonLeapYear

theorem expected_rolls_in_non_leap_year :
  expectedRollsPerYear = 417.14 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_non_leap_year_l659_65976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_to_cube_surface_area_l659_65911

/-- Represents a cuboid with a square base -/
structure Cuboid where
  base_side : ℝ
  height : ℝ

/-- Represents a cube -/
structure Cube where
  side : ℝ

/-- Calculate the volume of a cuboid -/
def cuboid_volume (c : Cuboid) : ℝ := c.base_side * c.base_side * c.height

/-- Calculate the volume of a cube -/
def cube_volume (c : Cube) : ℝ := c.side * c.side * c.side

/-- Calculate the surface area of a cube -/
def cube_surface_area (c : Cube) : ℝ := 6 * c.side * c.side

theorem cuboid_to_cube_surface_area
  (c : Cuboid)
  (h_height : c.height = c.base_side + 2)
  (h_volume : cuboid_volume c - cube_volume (Cube.mk c.base_side) = 50) :
  cube_surface_area (Cube.mk c.base_side) = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_to_cube_surface_area_l659_65911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_property_l659_65974

noncomputable def f (x a : ℝ) : ℝ := |x + 1| * Real.exp (-1 / x) - a

theorem three_roots_property (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a > 0 ∧ ∀ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 →
    x₂ - x₁ < a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_property_l659_65974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_AB_coordinates_l659_65944

-- Define the points
def A : ℝ × ℝ := (-1, -1)
def B : ℝ → ℝ × ℝ := λ x => (x, 5)
def C : ℝ × ℝ := (1, 3)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Define the vector AB
def vector_AB (x : ℝ) : ℝ × ℝ := ((B x).1 - A.1, (B x).2 - A.2)

-- Theorem statement
theorem collinear_implies_AB_coordinates :
  ∃ x : ℝ, collinear A (B x) C → vector_AB x = (3, 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_AB_coordinates_l659_65944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l659_65929

-- Define the points
def P : ℝ × ℝ := (5, 3)
def Q : ℝ × ℝ := (-P.1, P.2)  -- Reflection of P over y-axis
def R : ℝ × ℝ := (-Q.2, -Q.1)  -- Reflection of Q over y = -x

-- Calculate the area of triangle PQR
noncomputable def area_PQR : ℝ := 
  let base := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let height := |R.2 - P.2|
  (1/2) * base * height

-- Theorem statement
theorem area_of_triangle_PQR : area_PQR = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l659_65929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_minus_2_l659_65920

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 6/x - 6

-- Theorem statement
theorem min_value_f_minus_2 :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x : ℝ), f x - 2 ≥ m :=
by
  -- We'll use -1/2 as our minimum value
  use -1/2
  
  constructor
  · -- Prove that m = -1/2
    rfl
    
  · -- Prove that for all x, f x - 2 ≥ -1/2
    intro x
    -- We'll split the proof into two cases based on the definition of f
    by_cases h : x ≤ 1
    
    · -- Case 1: x ≤ 1
      simp [f, h]
      -- For x ≤ 1, we need to prove x^2 - 2 ≥ -1/2
      -- This is equivalent to x^2 ≥ 3/2
      -- Which is true for all real x
      sorry
      
    · -- Case 2: x > 1
      simp [f, h]
      -- For x > 1, we need to prove x + 6/x - 8 ≥ -1/2
      -- This is equivalent to x + 6/x ≥ 15/2
      -- We can prove this using AM-GM inequality
      sorry

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_minus_2_l659_65920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_has_property_P_k_l659_65903

/-- The type of binary sequences of length n -/
def BinarySeq (n : ℕ) := Fin n → Bool

/-- The set of binary sequences -/
def S (n : ℕ) : Set (BinarySeq n) := Set.univ

/-- The majority function for an odd number of sequences -/
def majority {n : ℕ} (seqs : List (BinarySeq n)) : BinarySeq n :=
  fun i => (seqs.map (fun seq => seq i)).count true > seqs.length / 2

/-- Property P_k for a set of binary sequences -/
def has_property_P_k (n k : ℕ) (s : Set (BinarySeq n)) : Prop :=
  ∀ (seqs : List (BinarySeq n)), seqs.length = 2 * k + 1 → (∀ seq ∈ seqs, seq ∈ s) →
    majority seqs ∈ s

/-- The main theorem: for all positive n and k, S has property P_k -/
theorem S_has_property_P_k (n k : ℕ) (h_n : n > 0) (h_k : k > 0) :
  has_property_P_k n k (S n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_has_property_P_k_l659_65903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l659_65970

def spinner_C : Finset ℕ := Finset.range 5 \ {0}
def spinner_D : Finset ℕ := Finset.range 4 \ {0}

def is_even (n : ℕ) : Bool := n % 2 = 0

def total_outcomes : ℕ := spinner_C.card * spinner_D.card

def even_product_outcomes : ℕ :=
  (spinner_C.filter (λ x => is_even x)).card * spinner_D.card +
  (spinner_C.filter (λ x => ¬(is_even x))).card * (spinner_D.filter (λ x => is_even x)).card

theorem probability_even_product :
  (even_product_outcomes : ℚ) / total_outcomes = 7 / 10 := by
  sorry

#eval spinner_C
#eval spinner_D
#eval total_outcomes
#eval even_product_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l659_65970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l659_65992

def point_A : Fin 3 → ℝ := ![1, -3, 4]
def point_B : Fin 3 → ℝ := ![-2, 2, -1]

theorem distance_between_points : 
  Real.sqrt (((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2) : ℝ) = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l659_65992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l659_65923

/-- The distance from a point in polar coordinates to a line in polar form -/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  |a * r * Real.cos θ + b * r * Real.sin θ + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (2, π/6) to the line ρ sin(θ - π/6) = 1 is 1 -/
theorem distance_point_to_line_problem :
  distance_point_to_line 2 (Real.pi/6) 1 (-Real.sqrt 3) 2 = 1 := by
  sorry

#check distance_point_to_line_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l659_65923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l659_65917

theorem solve_equation (m : ℝ) : (m - 6) ^ 4 = (1 / 16) ^ (2 : ℝ) → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l659_65917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_consecutive_even_product_consecutive_even_l659_65916

-- Define a function to generate n consecutive natural numbers starting from k
def consecutiveNaturals (k n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => k + i)

-- Theorem 1: Sum of even number of consecutive natural numbers is even
theorem sum_even_consecutive_even (k n : ℕ) :
  Even n → Even (List.sum (consecutiveNaturals k (2 * n))) := by
  sorry

-- Theorem 2: Product of any number of consecutive natural numbers is even
theorem product_consecutive_even (k n : ℕ) :
  n > 0 → Even (List.prod (consecutiveNaturals k n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_consecutive_even_product_consecutive_even_l659_65916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isometric_equilateral_triangle_area_l659_65961

/-- Definition of a Triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equilateral triangle predicate -/
def Triangle.Equilateral (t : Triangle) : Prop := sorry

/-- Side length of a triangle -/
def Triangle.SideLength (t : Triangle) : ℝ := sorry

/-- Area of a triangle -/
noncomputable def Triangle.Area (t : Triangle) : ℝ := sorry

/-- Isometric projection relation between two triangles -/
def IsometricProjection (t1 t2 : Triangle) : Prop := sorry

/-- The area of an isometric projection of an equilateral triangle -/
theorem isometric_equilateral_triangle_area (ABC : Triangle) (A'B'C' : Triangle) :
  Triangle.Equilateral ABC →
  Triangle.SideLength ABC = 10 →
  IsometricProjection ABC A'B'C' →
  Triangle.Area A'B'C' = (25 * Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isometric_equilateral_triangle_area_l659_65961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrationals_in_list_l659_65969

/-- A number type that can represent both rational and irrational numbers -/
inductive Number
| rational : ℚ → Number
| irrational : ℝ → Number

/-- The list of numbers from the problem -/
def number_list : List Number := sorry

/-- √121 is rational and equal to 11 -/
axiom sqrt_121_rational : ∃ (q : ℚ), q = 11 ∧ number_list.get? 0 = some (Number.rational q)

/-- -3.14 is a terminating decimal, hence rational -/
axiom neg_3_14_rational : ∃ (q : ℚ), number_list.get? 1 = some (Number.rational q)

/-- π is irrational -/
axiom pi_irrational : ∀ (q : ℚ), (q : ℝ) ≠ Real.pi

/-- -π/3 is in the list and is irrational -/
axiom neg_pi_third_in_list : ∃ (r : ℝ), r = -Real.pi/3 ∧ number_list.get? 2 = some (Number.irrational r)

/-- -0.77... is a repeating decimal, hence rational -/
axiom neg_0_77_rational : ∃ (q : ℚ), number_list.get? 3 = some (Number.rational q)

/-- 22/7 is a fraction, hence rational -/
axiom twentytwo_sevenths_rational : ∃ (q : ℚ), q = 22/7 ∧ number_list.get? 4 = some (Number.rational q)

/-- 1.6262262226... has a non-repeating, non-terminating pattern, hence irrational -/
axiom special_number_irrational : ∃ (r : ℝ), number_list.get? 5 = some (Number.irrational r)

/-- The main theorem: there are exactly 2 irrational numbers in the list -/
theorem two_irrationals_in_list : (number_list.filter fun n => match n with
  | Number.rational _ => false
  | Number.irrational _ => true).length = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrationals_in_list_l659_65969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l659_65994

/-- If a natural number n has exactly 4 divisors, then n^3 has exactly 10 divisors. -/
theorem divisors_of_cube (n : ℕ) (h : (Nat.divisors n).card = 4) : (Nat.divisors (n^3)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_cube_l659_65994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l659_65922

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  d = -2 ∧
  (∀ n, a n = 41 - 2 * n) ∧
  (∃ (n_max : ℕ), ∀ n, sum a n ≤ sum a n_max ∧ n_max = 20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l659_65922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l659_65919

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 3*y + 1 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3*y + 2 = 0

-- Define the center and radius of C1
noncomputable def center_C1 : ℝ × ℝ := (-1, -3/2)
noncomputable def radius_C1 : ℝ := 3/2

-- Define the center and radius of C2
noncomputable def center_C2 : ℝ × ℝ := (-2, -3/2)
noncomputable def radius_C2 : ℝ := Real.sqrt 17 / 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 1

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers < radius_C1 + radius_C2 ∧
  distance_between_centers > abs (radius_C1 - radius_C2) := by
  sorry

-- Additional theorem to state the conclusion
theorem circles_are_intersecting : 
  ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l659_65919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l659_65979

theorem complex_division_result : 
  (2 + 4 * Complex.I) / (1 + Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l659_65979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l659_65993

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate for triangles with perimeter 21 -/
def has_perimeter_21 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 21

/-- Predicate for non-congruent triangles -/
def non_congruent (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

theorem count_non_congruent_triangles : 
  ∃ (triangles : Finset IntTriangle),
    (∀ t, t ∈ triangles → has_perimeter_21 t) ∧
    (∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → non_congruent t1 t2) ∧
    triangles.card = 6 ∧
    (∀ t : IntTriangle, has_perimeter_21 t → 
      (∃ t', t' ∈ triangles ∧ ¬(non_congruent t t'))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l659_65993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l659_65959

noncomputable def m : ℝ × ℝ := (-1, Real.sqrt 3)

theorem angle_between_vectors (n : ℝ × ℝ) 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m.1 * m.1 + m.2 * m.2 - (m.1 * n.1 + m.2 * n.2) = 5)
  (h4 : n.1 * (m.1 + n.1) + n.2 * (m.2 + n.2) = 0) :
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / 
    (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l659_65959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_ratio_l659_65973

noncomputable section

/-- The speed of the current in the river -/
def current_speed : ℝ := 6

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 18

/-- The speed of the boat going downstream -/
def downstream_speed : ℝ := boat_speed + current_speed

/-- The speed of the boat going upstream -/
def upstream_speed : ℝ := boat_speed - current_speed

/-- The distance traveled in each direction (assumed to be 1 for simplicity) -/
def distance : ℝ := 1

/-- The time taken to travel downstream -/
noncomputable def time_downstream : ℝ := distance / downstream_speed

/-- The time taken to travel upstream -/
noncomputable def time_upstream : ℝ := distance / upstream_speed

/-- The total time for the round trip -/
noncomputable def total_time : ℝ := time_downstream + time_upstream

/-- The total distance for the round trip -/
def total_distance : ℝ := 2 * distance

/-- The average speed for the round trip -/
noncomputable def average_speed : ℝ := total_distance / total_time

/-- Theorem: The ratio of the average speed for the round trip to the speed in still water is 8/9 -/
theorem average_speed_ratio : average_speed / boat_speed = 8 / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_ratio_l659_65973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l659_65956

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_specific :
  let ρ : Real := 4
  let θ : Real := Real.pi
  let φ : Real := Real.pi / 3
  spherical_to_rectangular ρ θ φ = (-2 * Real.sqrt 3, 0, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_l659_65956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l659_65996

theorem cube_root_inequality (x : ℝ) :
  x.rpow (1/3) + 3 / (x.rpow (2/3) + 3 * x.rpow (1/3) + 4) < 0 ↔ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l659_65996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l659_65967

def sequence_rule (seq : List ℤ) : Prop :=
  ∀ i, i ≥ 2 → i < seq.length → seq[i]! = seq[i-1]! + seq[i-2]!

theorem find_x (seq : List ℤ) (h : sequence_rule seq) 
  (h_subseq : [-2, -1, 1, 0, 1, 1, 2, 3].isSuffixOf seq) :
  ∃ x y z, seq = x :: y :: z :: [-2, -1, 1, 0, 1, 1, 2, 3] ∧ x = 4 := by
  sorry

#check find_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l659_65967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l659_65907

noncomputable def f (x : ℝ) := 2 * Real.sin (3 * x + Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = 2 * Real.pi / 3 := by
  sorry

#check min_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l659_65907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_centrally_symmetric_polygon_in_triangle_l659_65938

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in a 2D plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List Point

/-- Determines if a polygon is centrally symmetric -/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Determines if a polygon is inscribed in a triangle -/
def isInscribed (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon -/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- The theorem stating that the largest centrally symmetric polygon 
    inscribed in a triangle is a hexagon with 2/3 the area of the triangle -/
theorem largest_centrally_symmetric_polygon_in_triangle 
  (t : Triangle) : 
  ∃ (p : Polygon), 
    isCentrallySymmetric p ∧ 
    isInscribed p t ∧ 
    (∀ (q : Polygon), isCentrallySymmetric q → isInscribed q t → area q ≤ area p) ∧
    (area p = (2/3) * triangleArea t) ∧
    p.vertices.length = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_centrally_symmetric_polygon_in_triangle_l659_65938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l659_65981

theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : crossing_time = 12) : 
  let total_distance := 2 * train_length
  let relative_speed := total_distance / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  train_speed_kmh = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l659_65981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_marbles_count_l659_65980

theorem blue_marbles_count (red yellow : ℕ) (prob_yellow : ℚ) (blue : ℕ) : 
  red = 11 → 
  yellow = 6 → 
  prob_yellow = 1/4 → 
  blue = (yellow / prob_yellow).num - red - yellow →
  blue = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_marbles_count_l659_65980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_A_given_B_l659_65904

/-- A fair die with 6 sides -/
def Die : Type := Fin 6

/-- The sample space of rolling two dice -/
def SampleSpace : Type := Die × Die

/-- Event A: The two numbers are different -/
def EventA (outcome : SampleSpace) : Prop :=
  outcome.1 ≠ outcome.2

/-- Event B: At least one 5 appears -/
def EventB (outcome : SampleSpace) : Prop :=
  outcome.1 = ⟨4, sorry⟩ ∨ outcome.2 = ⟨4, sorry⟩

/-- The probability measure for the sample space -/
noncomputable def P : Set SampleSpace → ℝ := sorry

theorem conditional_probability_A_given_B :
  P {outcome | EventA outcome ∧ EventB outcome} / P {outcome | EventB outcome} = 10 / 11 := by
  sorry

#check conditional_probability_A_given_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_A_given_B_l659_65904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_chord_product_l659_65936

/-- Given a regular nonagon inscribed in a unit circle, the product of the lengths of four specific chords equals 3 -/
theorem regular_nonagon_chord_product (A : Fin 9 → ℝ × ℝ) :
  (∀ i j : Fin 9, ‖A i - A j‖ = ‖A ((i + 1) % 9) - A ((j + 1) % 9)‖) →  -- regular nonagon
  (∀ i : Fin 9, ‖A i‖ = 1) →  -- inscribed in unit circle
  ‖A 0 - A 1‖ * ‖A 0 - A 2‖ * ‖A 0 - A 3‖ * ‖A 0 - A 4‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_chord_product_l659_65936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_2003_candy_game_strategy_l659_65942

/-- Represents the state of the candy game -/
inductive GameState
  | FirstPlayerTurn (candies : Nat)
  | SecondPlayerTurn (candies : Nat)
  | GameOver

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Defines a valid move in the candy game -/
def validMove (candies : Nat) (eaten : Nat) : Bool :=
  eaten = 1 || eaten = candies / 2

/-- Defines the winning strategy for the candy game -/
def winningStrategy (initialCandies : Nat) (player : Player) : Prop :=
  sorry -- Placeholder for the actual definition

/-- Theorem stating that the second player has a winning strategy for 2003 candies -/
theorem second_player_wins_2003 :
  winningStrategy 2003 Player.Second := by
  sorry

/-- Main theorem proving the existence of a winning strategy for the second player -/
theorem candy_game_strategy (initialCandies : Nat) :
  initialCandies = 2003 →
  winningStrategy initialCandies Player.Second := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_2003_candy_game_strategy_l659_65942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_two_l659_65933

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then |x - 1|
  else 3^x

-- State the theorem
theorem unique_solution_is_negative_two :
  ∃! x : ℝ, f x = 3 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_two_l659_65933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l659_65915

/-- A line passing through the origin and a given point -/
def line_through_origin_and_point (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ (t : ℝ), q = (t * p.1, t * p.2)}

/-- A line is perpendicular to the polar axis -/
def line_perpendicular_to_polar_axis (l : Set (ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ), p ∈ l → q ∈ l → p ≠ q → (p.1 - q.1) * 1 + (p.2 - q.2) * 0 = 0

/-- The polar equation of a line passing through (2, 0) and perpendicular to the polar axis -/
theorem polar_line_equation (ρ θ : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = ρ ∧ P.2 = θ ∧ P ∈ line_through_origin_and_point (2, 0)) ∧
  line_perpendicular_to_polar_axis (line_through_origin_and_point (2, 0)) →
  ρ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l659_65915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l659_65909

-- Define the hyperbola equation
def hyperbola_equation (m x y : ℝ) : Prop :=
  x^2 / (m + 9) + y^2 / 9 = 1

-- Define the eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt ((9 + (-(m + 9))) / 9)

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation m x y) ∧ eccentricity m = 2 → m = -36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l659_65909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_2_sqrt_5_l659_65932

/-- The chord length intercepted by a line and a circle in polar coordinates -/
noncomputable def chord_length (ρ_line : ℝ → ℝ → ℝ) (ρ_circle : ℝ) : ℝ :=
  let θ₁ := Real.arcsin (2 / 3) - Real.pi / 4
  let θ₂ := Real.pi - Real.arcsin (2 / 3) - Real.pi / 4
  2 * ρ_circle * Real.sin ((θ₂ - θ₁) / 2)

theorem chord_length_equals_2_sqrt_5 :
  chord_length (fun ρ θ ↦ ρ * Real.sin (θ + Real.pi / 4)) 3 = 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_2_sqrt_5_l659_65932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l659_65977

/-- A quadratic function f(x) = ax^2 - 4x + c -/
def quadratic_function (a c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 - 4 * x + c

/-- The solution set of f(x) < 0 -/
def solution_set (a c : ℝ) : Set ℝ := {x | quadratic_function a c x < 0}

theorem quadratic_function_properties (a c : ℝ) 
  (h : solution_set a c = Set.Ioo (-1) 5) : 
  a = 1 ∧ c = -5 ∧ 
  (∀ x ∈ Set.Icc 0 3, -9 ≤ quadratic_function a c x) ∧
  (∃ x ∈ Set.Icc 0 3, quadratic_function a c x = -9) ∧
  (∀ x ∈ Set.Icc 0 3, quadratic_function a c x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 3, quadratic_function a c x = -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l659_65977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_line_l659_65902

/-- The line passing through points A(3, 0) and B(1, 1) -/
def line_AB (x y : ℝ) : Prop := x + 2*y = 3

/-- The function to be minimized -/
noncomputable def f (x y : ℝ) : ℝ := (2:ℝ)^x + (4:ℝ)^y

theorem min_value_on_line :
  ∃ (x₀ y₀ : ℝ), line_AB x₀ y₀ ∧
  (∀ (x y : ℝ), line_AB x y → f x y ≥ f x₀ y₀) ∧
  f x₀ y₀ = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_line_l659_65902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_parallelogram_l659_65921

-- Define Point type if not already defined in Mathlib
structure Point where
  x : ℝ
  y : ℝ

-- Define Line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Quadrilateral (A B C D : Point) where
  -- Assuming Circle is defined elsewhere in the library
  tangent_to_circle : ∃ O : Point, True  -- Placeholder for external tangency condition

-- Define external angle bisector
def external_angle_bisector (A B C : Point) : Line := sorry

-- Define orthocenter
def orthocenter (A B C : Point) : Point := sorry

-- Define is_parallelogram
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Main theorem
theorem quadrilateral_parallelogram 
  (A B C D : Point) (O : Point) (K L M N K₁ L₁ M₁ N₁ : Point) 
  (h : Quadrilateral A B C D) : 
  external_angle_bisector A B K = external_angle_bisector B A K →
  external_angle_bisector B C L = external_angle_bisector C B L →
  external_angle_bisector C D M = external_angle_bisector D C M →
  external_angle_bisector D A N = external_angle_bisector A D N →
  K₁ = orthocenter A B K →
  L₁ = orthocenter B C L →
  M₁ = orthocenter C D M →
  N₁ = orthocenter D A N →
  is_parallelogram K₁ L₁ M₁ N₁ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_parallelogram_l659_65921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_paper_area_l659_65925

/-- Represents an L-shaped paper composed of two identical rectangles --/
structure LShapedPaper where
  longest_side : ℝ
  perimeter : ℝ

/-- Calculates the area of an L-shaped paper --/
noncomputable def area (paper : LShapedPaper) : ℝ :=
  let width := (paper.perimeter - 4 * paper.longest_side) / (-2)
  let length := paper.longest_side - width
  2 * (length * width)

/-- Theorem stating that an L-shaped paper with given dimensions has an area of 120 square cm --/
theorem l_shaped_paper_area :
  ∀ (paper : LShapedPaper),
    paper.longest_side = 16 ∧
    paper.perimeter = 52 →
    area paper = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_paper_area_l659_65925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_identity_l659_65999

theorem sqrt_identity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 0 < a^2 - b) :
  (Real.sqrt (a + Real.sqrt b) = Real.sqrt ((a + Real.sqrt (a^2 - b))/2) + Real.sqrt ((a - Real.sqrt (a^2 - b))/2)) ∧
  (Real.sqrt (a - Real.sqrt b) = Real.sqrt ((a + Real.sqrt (a^2 - b))/2) - Real.sqrt ((a - Real.sqrt (a^2 - b))/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_identity_l659_65999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_range_l659_65957

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / a^2 = 1

-- Define the circle
def circleO (O F₁ F₂ : ℝ × ℝ) : Prop :=
  O = (0, 0) ∧ dist O F₁ = dist O F₂ ∧ dist F₁ F₂ = 2 * dist O F₁

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 7 * x - 4

-- Main theorem
theorem hyperbola_foci_range (a : ℝ) (F₁ F₂ O : ℝ × ℝ) :
  a > 0 →
  hyperbola a F₁.1 F₁.2 →
  hyperbola a F₂.1 F₂.2 →
  circleO O F₁ F₂ →
  (∃ x y, line x y ∧ dist O (x, y) = dist O F₁) →
  a ∈ Set.Ioi 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_range_l659_65957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_self_intersection_l659_65965

/-- Represents the configuration of three rods connected in a "П" shape -/
structure RodConfiguration where
  α : Real
  β : Real

/-- The space of all possible rod configurations -/
def configurationSpace : Set RodConfiguration :=
  {c | 0 ≤ c.α ∧ c.α ≤ Real.pi ∧ 0 ≤ c.β ∧ c.β ≤ 2*Real.pi}

/-- Condition for self-intersection of rods -/
def hasSelfIntersection (c : RodConfiguration) : Prop :=
  0 ≤ c.β ∧ c.β < Real.pi/2 - c.α/2 ∧ 0 ≤ c.α ∧ c.α < Real.pi/2 - c.β/2

/-- The probability measure on the configuration space -/
noncomputable def P : Set RodConfiguration → Real :=
  sorry

/-- The theorem stating the probability of no self-intersection -/
theorem probability_no_self_intersection :
  P {c ∈ configurationSpace | ¬hasSelfIntersection c} = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_self_intersection_l659_65965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_after_dilutions_theorem_l659_65930

/-- Calculates the quantity of pure milk after two successive dilutions -/
noncomputable def milkAfterDilutions (initialCapacity : ℝ) (replacementVolume : ℝ) : ℝ :=
  let remainingMilk1 := initialCapacity - replacementVolume
  let milkFraction := remainingMilk1 / initialCapacity
  let milkRemoved := replacementVolume * milkFraction
  remainingMilk1 - milkRemoved

/-- The quantity of pure milk after two successive dilutions is 72.9 litres -/
theorem milk_after_dilutions_theorem :
  milkAfterDilutions 90 9 = 72.9 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_after_dilutions_theorem_l659_65930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l659_65972

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  let a := Real.pi / 4
  let b := Real.pi / 2
  (∀ x ∈ Set.Icc a b, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc a b, f x = 3) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc a b, f x = 2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc a b, |f x - m| < 2) ↔ m ∈ Set.Ioo 1 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l659_65972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OP_OQ_general_equation_l_l659_65984

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 + 2 * Real.sin θ = ρ

-- Define the intersection points P and Q
variable (P Q : ℝ × ℝ)

-- Define point A
def A : ℝ × ℝ := (0, 1)

-- Axioms for the given conditions
variable (α : ℝ)
axiom α_range : 0 < α ∧ α < Real.pi / 2
axiom ρ_nonneg : ∀ θ, ∃ ρ ≥ 0, curve_C ρ θ
axiom l_intersects_C : ∃ t₁ t₂, curve_C (line_l α t₁).1 (line_l α t₁).2 ∧ 
                                curve_C (line_l α t₂).1 (line_l α t₂).2 ∧
                                P = line_l α t₁ ∧ Q = line_l α t₂
axiom AP_AQ_relation : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
                       2 * Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2)

-- Theorem statements
theorem dot_product_OP_OQ : P.1 * Q.1 + P.2 * Q.2 = -1 := by sorry

theorem general_equation_l : 
  (∀ x y, y = (1/2) * x + 1 ↔ ∃ t, (x, y) = line_l α t) ∧
  (∀ x y, x - 2*y + 2 = 0 ↔ ∃ t, (x, y) = line_l α t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OP_OQ_general_equation_l_l659_65984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_AOB_l659_65900

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a circle equation in 2D space -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  eq : (x y : ℝ) → Prop

/-- Triangle AOB with given vertices -/
def triangleAOB : (Point2D × Point2D × Point2D) :=
  (⟨4, 0⟩, ⟨0, 3⟩, ⟨0, 0⟩)

/-- The proposed equation of the circumcircle -/
def proposedCircumcircle : CircleEquation :=
  ⟨-4, -3, 0, 0, fun x y => x^2 + y^2 - 4*x - 3*y = 0⟩

/-- Theorem: The proposed circle equation represents the circumcircle of triangle AOB -/
theorem circumcircle_of_triangle_AOB :
  let (A, B, O) := triangleAOB
  proposedCircumcircle.eq A.x A.y ∧
  proposedCircumcircle.eq B.x B.y ∧
  proposedCircumcircle.eq O.x O.y ∧
  ∀ (circle : CircleEquation),
    (circle.eq A.x A.y ∧ circle.eq B.x B.y ∧ circle.eq O.x O.y) →
    ∀ (x y : ℝ), circle.eq x y ↔ proposedCircumcircle.eq x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_AOB_l659_65900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l659_65987

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the inverse function g
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem tangent_line_property (x₁ x₂ : ℝ) 
  (h1 : x₁ > x₂) (h2 : x₂ > 0) 
  (h3 : ∃ (m b : ℝ), (∀ x, m * x + b = f x ↔ x = x₁) ∧ 
                     (∀ x, m * x + b = g x ↔ x = x₂)) :
  x₁ > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l659_65987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_abc_l659_65955

theorem weight_of_abc (a b c d e f i j : ℝ) : 
  (a + b + c + f + i) / 5 = 79 →
  (a + b + c + d + e + f + i + j) / 8 = 83 →
  (d + e + f + (d + 6) + (e - 8) + i + (j + 5)) / 7 = 84 →
  a + b + c = 237 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_abc_l659_65955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meeting_time_l659_65941

/-- The time (in seconds) it takes for two boys to meet on a circular track -/
noncomputable def timeTomeet (trackLength : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  trackLength / (speed1 + speed2)

/-- Conversion factor from km/hr to m/s -/
noncomputable def kmhrToms : ℝ := 1000 / 3600

theorem boys_meeting_time :
  let trackLength : ℝ := 4800
  let speed1 : ℝ := 61.3 * kmhrToms
  let speed2 : ℝ := 97.5 * kmhrToms
  ∃ ε > 0, |timeTomeet trackLength speed1 speed2 - 108.8| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meeting_time_l659_65941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_girls_equal_class_composition_l659_65905

/-- Represents a class with boys and girls -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ
  total : boys + girls = 30
  unique_boy_friends : ∀ i j, i ≠ j → i < boys → j < boys → 
    ∃ f : Fin boys → Fin girls, Function.Injective f
  unique_girl_friends : ∀ i j, i ≠ j → i < girls → j < girls → 
    ∃ f : Fin girls → Fin boys, Function.Injective f

/-- The number of boys equals the number of girls in the class -/
theorem boys_girls_equal (c : ClassComposition) : c.boys = c.girls := by
  sorry

/-- The number of boys (and girls) in the class is 15 -/
theorem class_composition (c : ClassComposition) : c.boys = 15 ∧ c.girls = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_girls_equal_class_composition_l659_65905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_100a_eq_10_sqrt_a_sqrt_300_approx_sqrt_a_eq_160_implies_a_eq_25600_sqrt_a_comparison_l659_65995

-- Define the approximation of √3
noncomputable def sqrt3_approx : ℝ := 1.732

-- Theorem 1: For a > 0, √(100a) = 10√a
theorem sqrt_100a_eq_10_sqrt_a (a : ℝ) (ha : a > 0) :
  Real.sqrt (100 * a) = 10 * Real.sqrt a := by sorry

-- Theorem 2: Given √3 ≈ 1.732, prove √300 ≈ 17.32
theorem sqrt_300_approx :
  abs (Real.sqrt 300 - 17.32) < 0.001 := by sorry

-- Theorem 3: If √a = 160, then a = 25600
theorem sqrt_a_eq_160_implies_a_eq_25600 (a : ℝ) :
  Real.sqrt a = 160 → a = 25600 := by sorry

-- Theorem 4: Comparison of √a and a for a > 0
theorem sqrt_a_comparison (a : ℝ) (ha : a > 0) :
  (0 < a ∧ a < 1 → a < Real.sqrt a) ∧
  (a = 1 → a = Real.sqrt a) ∧
  (a > 1 → a > Real.sqrt a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_100a_eq_10_sqrt_a_sqrt_300_approx_sqrt_a_eq_160_implies_a_eq_25600_sqrt_a_comparison_l659_65995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_or_line_l659_65939

-- Define the circles
variable (O₁ O₂ : ℝ × ℝ)
variable (r₁ r₂ : ℝ)

-- Define the power of a point with respect to a circle
def power (P O : ℝ × ℝ) (r : ℝ) : ℝ :=
  (P.1 - O.1)^2 + (P.2 - O.2)^2 - r^2

-- Define the set of points P that satisfy the power ratio condition
def locus (O₁ O₂ : ℝ × ℝ) (r₁ r₂ k : ℝ) : Set (ℝ × ℝ) :=
  {P | power P O₁ r₁ = k * power P O₂ r₂}

-- Theorem statement
theorem locus_is_circle_or_line (O₁ O₂ : ℝ × ℝ) (r₁ r₂ k : ℝ) :
  (k ≠ 1 → ∃ C r, locus O₁ O₂ r₁ r₂ k = {P | power P C r = 0}) ∧
  (k = 1 → ∃ a b c, locus O₁ O₂ r₁ r₂ k = {P : ℝ × ℝ | a * P.1 + b * P.2 + c = 0}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_or_line_l659_65939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_five_l659_65968

/-- The side length of the largest inscribable square in a rectangle with two equilateral triangles --/
noncomputable def largest_inscribed_square_side (rect_length rect_width : ℝ) (triangle_side : ℝ) : ℝ :=
  rect_width - (triangle_side * Real.sqrt 3 / 2)

/-- Theorem stating the largest inscribable square side length in the given configuration --/
theorem largest_square_side_is_five :
  let rect_length : ℝ := 20
  let rect_width : ℝ := 15
  let triangle_side : ℝ := 5 * Real.sqrt 3
  largest_inscribed_square_side rect_length rect_width triangle_side = 5 := by
  sorry

#check largest_square_side_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_five_l659_65968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l659_65924

theorem expression_equality : 
  -1^(4 : ℕ) + (Real.sqrt 3 - 1)^(0 : ℕ) + (1/3)^(-(2 : ℤ)) + |Real.sqrt 3 - 2| + Real.tan (60 * π / 180) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l659_65924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_one_minus_g_equals_four_to_500_l659_65989

theorem y_one_minus_g_equals_four_to_500 (y m g : ℝ) : 
  y = (3 + Real.sqrt 5)^500 → 
  m = ⌊y⌋ → 
  g = y - m → 
  y * (1 - g) = 4^500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_one_minus_g_equals_four_to_500_l659_65989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_theorem_l659_65910

/-- Represents a team's scoring information -/
structure Team where
  name : String
  scoreRate : Float

/-- Calculates the total score for a team given the duration -/
def totalScore (team : Team) (duration : Float) : Float :=
  team.scoreRate * duration

/-- Theorem: The difference between the highest and lowest scores is 28.8 points -/
theorem score_difference_theorem (wildcats panthers tigers : Team) 
  (duration : Float) 
  (h1 : wildcats.scoreRate = 2.5)
  (h2 : panthers.scoreRate = 1.3)
  (h3 : tigers.scoreRate = 1.8)
  (h4 : duration = 24) :
  max (totalScore wildcats duration) (max (totalScore panthers duration) (totalScore tigers duration)) -
  min (totalScore wildcats duration) (min (totalScore panthers duration) (totalScore tigers duration)) = 28.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_theorem_l659_65910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l659_65948

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a cubic polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (z : ℂ) : ℂ :=
  z^3 + p.a * z^2 + p.b * z + p.c

/-- The roots of the polynomial satisfy the given conditions -/
def has_specific_roots (p : CubicPolynomial) : Prop :=
  ∃ w : ℂ, (p.eval (w + 3*Complex.I) = 0) ∧ (p.eval (w + 9*Complex.I) = 0) ∧ (p.eval (2*w - 4) = 0)

theorem cubic_polynomial_sum_of_coefficients (p : CubicPolynomial) 
  (h : has_specific_roots p) : 
  |p.a + p.b + p.c| = 136 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l659_65948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_shift_l659_65931

/-- Given two points A and B on a grid, if A has coordinates (-3, 4) when B is the origin,
    then B has coordinates (3, -4) when A is the origin. -/
theorem coordinate_shift (A B : ℝ × ℝ) : 
  (A.1 = -3 ∧ A.2 = 4) → (B.1 = 3 ∧ B.2 = -4) :=
by
  intro h
  sorry

#check coordinate_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_shift_l659_65931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_e_l659_65914

def point := ℝ × ℝ

def reflect_over_y_axis (p : point) : point :=
  (-p.1, p.2)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem reflection_distance_e : 
  let E : point := (-2, -3)
  let E' : point := reflect_over_y_axis E
  distance E E' = 4 := by
  -- Proof steps would go here
  sorry

#eval reflect_over_y_axis (-2, -3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_e_l659_65914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_four_game_l659_65943

def cards : List ℕ := [9, 7, 3, 2]

def target : ℕ := 24

def valid_operation (op : ℕ → ℕ → ℕ) (a b : ℕ) : Prop :=
  (op a b = a + b) ∨ (op a b = a - b) ∨ (op a b = a * b) ∨ (∃ k, b ≠ 0 ∧ a = b * k)

def can_make_24 (nums : List ℕ) : Prop :=
  ∃ (op1 op2 op3 : ℕ → ℕ → ℕ) (perm : List ℕ),
    List.Perm perm nums ∧
    perm.length = 4 ∧
    valid_operation op1 perm[0]! perm[1]! ∧
    valid_operation op2 (op1 perm[0]! perm[1]!) perm[2]! ∧
    valid_operation op3 (op2 (op1 perm[0]! perm[1]!) perm[2]!) perm[3]! ∧
    op3 (op2 (op1 perm[0]! perm[1]!) perm[2]!) perm[3]! = target

theorem twenty_four_game :
  can_make_24 cards := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_four_game_l659_65943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l659_65954

/-- Represents a journey with cycling and walking portions -/
structure Journey where
  total_time : ℝ
  cycle_speed : ℝ
  walk_speed : ℝ

/-- Calculates the distance walked in a journey -/
noncomputable def distance_walked (j : Journey) : ℝ :=
  let total_distance := j.total_time * (j.cycle_speed * j.walk_speed) / (j.cycle_speed + j.walk_speed)
  total_distance / 2

theorem journey_distance_theorem (j : Journey) 
  (h1 : j.total_time = 1) 
  (h2 : j.cycle_speed = 15) 
  (h3 : j.walk_speed = 4) : 
  ∃ (ε : ℝ), abs (distance_walked j - 3.2) < ε ∧ ε < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l659_65954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l659_65991

def study_time_differences : List Int := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

def average_difference : ℚ :=
  (study_time_differences.sum : ℚ) / study_time_differences.length

theorem average_study_time_difference :
  abs ((average_difference : ℝ) - 4.29) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l659_65991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_is_nk_l659_65947

/-- A structure representing an arithmetic progression of nonnegative integers -/
structure ArithmeticProgression where
  difference : ℕ+
  first_term : ℕ

/-- The property that among any k consecutive nonnegative integers, at least one belongs to one of the n arithmetic progressions -/
def covers_k_consecutive (n : ℕ+) (progressions : Fin n → ArithmeticProgression) (k : ℕ+) : Prop :=
  ∀ start : ℕ, ∃ (i : Fin n) (j : ℕ), ∃ t ∈ Finset.range k,
    start + t = (progressions i).first_term + j * (progressions i).difference

/-- The theorem stating the maximum possible value of the minimum difference -/
theorem max_min_difference_is_nk {n k : ℕ+} 
  (progressions : Fin n → ArithmeticProgression)
  (h_covers : covers_k_consecutive n progressions k) :
  ∃ (i : Fin n), (progressions i).difference ≤ n * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_is_nk_l659_65947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l659_65901

/-- Two right circular cylinders with the same volume -/
def SameVolume (r1 h1 r2 h2 : ℝ) : Prop :=
  Real.pi * r1^2 * h1 = Real.pi * r2^2 * h2

/-- The radius of the second cylinder is 20% more than the radius of the first -/
def RadiusRelation (r1 r2 : ℝ) : Prop :=
  r2 = 1.2 * r1

theorem cylinder_height_relation (r1 h1 r2 h2 : ℝ) 
  (hv : SameVolume r1 h1 r2 h2) (hr : RadiusRelation r1 r2) :
  h1 = 1.44 * h2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l659_65901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_shift_l659_65952

theorem fixed_point_of_exponential_shift (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (λ x => a^(x - 2) - 3) 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_shift_l659_65952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_butcher_packages_total_weight_equation_l659_65906

/- Represent the number of four-pound packages delivered by each butcher -/
def x : ℕ := 10
def y : ℕ := 7
def z : ℕ := 8

/- Represent the delivery times for each butcher -/
def t1 : ℕ := 8
def t2 : ℕ := 10
def t3 : ℕ := 18

/- The total weight of ground beef in pounds -/
def total_weight : ℕ := 100

/- The weight of each package in pounds -/
def package_weight : ℕ := 4

theorem third_butcher_packages :
  z = 8 :=
by
  -- The proof goes here
  sorry

theorem total_weight_equation :
  package_weight * (x + y + z) = total_weight :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_butcher_packages_total_weight_equation_l659_65906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l659_65983

theorem order_of_values (a b c : ℝ) (ha : a = 60.5) (hb : b = 0.56) (hc : c = Real.log 0.56 / Real.log 10) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l659_65983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equality_l659_65934

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B : Set ℝ := {x | x ≥ 0 ∧ x ≤ 4}
def C : Set ℝ := {x | |x + 1| ≤ 2}

-- State the theorem
theorem complement_intersection_equality :
  (A \ (B ∩ C)) = Set.Icc (-3) 0 ∪ Set.Ioo 1 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equality_l659_65934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg_four_four_sqrt_three_l659_65928

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 ∧ y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rectangular_to_polar_neg_four_four_sqrt_three :
  let (r, θ) := rectangular_to_polar (-4) (4 * Real.sqrt 3)
  r = 8 ∧ θ = 2 * Real.pi / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg_four_four_sqrt_three_l659_65928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l659_65937

noncomputable section

open Real

theorem triangle_properties (a b c A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (angle_sum : A + B + C = π)
  (sine_law_AB : sin A / a = sin B / b)
  (sine_law_BC : sin B / b = sin C / c)
  (given_eq : sin C * sin (A - B) = sin B * sin (C - A))
  (A_eq_2B : A = 2 * B) :
  C = 5 * π / 8 ∧ 2 * a^2 = b^2 + c^2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l659_65937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_l659_65966

noncomputable def f (x : ℝ) := max (1 - x) (2^x)

theorem f_greater_than_four (x : ℝ) :
  f x > 4 ↔ x ∈ Set.Iio (-3) ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_l659_65966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_end_time_l659_65946

-- Define a custom time type
structure SchoolTime where
  hours : Nat
  minutes : Nat

-- Define the addition operation for SchoolTime and minutes
def addMinutes (t : SchoolTime) (m : Nat) : SchoolTime :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Theorem statement
theorem class_end_time 
  (start_time : SchoolTime) 
  (class_duration : Nat) : 
  start_time.hours = 8 ∧ 
  start_time.minutes = 10 ∧ 
  class_duration = 40 → 
  addMinutes start_time class_duration = { hours := 8, minutes := 50 } := by
  sorry

-- Example usage
def main : IO Unit := do
  let start_time : SchoolTime := { hours := 8, minutes := 10 }
  let class_duration : Nat := 40
  IO.println s!"Class starts at {start_time.hours}:{start_time.minutes}"
  IO.println s!"Class duration: {class_duration} minutes"
  let end_time := addMinutes start_time class_duration
  IO.println s!"Class ends at {end_time.hours}:{end_time.minutes}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_end_time_l659_65946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_function_group_l659_65926

noncomputable section

-- Define the function groups
def group_A1 (x : ℝ) : ℝ := Real.sqrt (x^2)
def group_A2 (x : ℝ) : ℝ := x -- Changed from Real.cuberoot (x^3)

def group_B1 (_ : ℝ) : ℝ := 1
def group_B2 (x : ℝ) : ℝ := x^0

def group_C1 (x : ℝ) : ℝ := 2*x + 1
def group_C2 (t : ℝ) : ℝ := 2*t + 1

def group_D1 (x : ℝ) : ℝ := x
def group_D2 (x : ℝ) : ℝ := (Real.sqrt x)^2

-- Define the notion of function equality
def functions_equal (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x = g x) ∧ (∀ x, (∃ y, f x = y) ↔ (∃ y, g x = y))

-- Theorem statement
theorem same_function_group :
  (¬ functions_equal group_A1 group_A2) ∧
  (¬ functions_equal group_B1 group_B2) ∧
  (functions_equal group_C1 group_C2) ∧
  (¬ functions_equal group_D1 group_D2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_function_group_l659_65926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_one_female_l659_65912

/-- The probability of selecting at most one female student from a group of 3 male and 2 female students when randomly choosing 2 students is 9/10. -/
theorem prob_at_most_one_female (n_male : ℕ) (n_female : ℕ) : 
  n_male = 3 → n_female = 2 → 
  (Nat.choose (n_male + n_female) 2 : ℚ)⁻¹ * 
  (Nat.choose n_male 2 + n_male * n_female : ℚ) = 9 / 10 := by
  intros h_male h_female
  rw [h_male, h_female]
  norm_num
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_one_female_l659_65912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l659_65908

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

def isOnEllipse (e : Ellipse) (x y : ℝ) : Prop := x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_properties :
  ∃ (e : Ellipse),
    e.a^2 = 12 ∧ e.b^2 = 4 ∧
    eccentricity e = Real.sqrt 6 / 3 ∧
    isOnEllipse e (2 * Real.sqrt 2) 0 ∧
    (∀ (m n : ℝ), isOnEllipse e m n → -1 ≤ n / (m - 4) ∧ n / (m - 4) ≤ 1) ∧
    (∃ (A B : ℝ × ℝ),
      isOnEllipse e A.1 A.2 ∧
      isOnEllipse e B.1 B.2 ∧
      B.2 - A.2 = B.1 - A.1 ∧  -- slope 1
      (A.1 - (-3))^2 + (A.2 - 2)^2 = (B.1 - (-3))^2 + (B.2 - 2)^2 ∧  -- isosceles
      (B.1 - A.1) * (B.2 - A.2) / 2 = 9/2) -- area
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l659_65908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_properties_l659_65962

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x / 4)

theorem f_satisfies_properties :
  (∀ x : ℝ, f (-x) + f x = 0) ∧
  (∀ x : ℝ, f x = f (4 - x)) :=
by
  constructor
  · intro x
    sorry -- Proof for the first property
  · intro x
    sorry -- Proof for the second property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_properties_l659_65962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l659_65990

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x < 1 then -1 else x - 2

-- State the theorem
theorem function_inequality (a : ℝ) :
  f (5 * a - 2) > f (2 * a^2) ↔ 3/5 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l659_65990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l659_65913

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  cosC : ℝ
  ha : a = 1
  hb : b = 2
  hcosC : cosC = 3/4

/-- The perimeter of the triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + Real.sqrt (t.a^2 + t.b^2 - 2*t.a*t.b*t.cosC)

/-- The sine of angle A -/
noncomputable def sinA (t : Triangle) : ℝ :=
  (t.a * Real.sqrt (1 - t.cosC^2)) / Real.sqrt (t.a^2 + t.b^2 - 2*t.a*t.b*t.cosC)

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  perimeter t = 3 + Real.sqrt 2 ∧ sinA t = Real.sqrt 14 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l659_65913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_win_probability_l659_65997

noncomputable def game_probability (p1 p2 p3 : ℝ) : ℝ :=
  p1 / (1 - (1 - p1) * (1 - p2) * (1 - p3))

theorem jane_win_probability :
  let jane_prob : ℝ := 1/3
  let bob_prob : ℝ := 2/3
  let alice_prob : ℝ := 1/3
  game_probability jane_prob bob_prob alice_prob = 9/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_win_probability_l659_65997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l659_65986

-- Define the circles and points
def circle_M (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - a)^2 = 1
def circle_N (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (-1, 0)
def point_Q : ℝ × ℝ := (-1, 2)

-- Define the statement
theorem circle_properties :
  -- Statement A
  (∃ (t1 t2 : ℝ × ℝ), t1 ≠ t2 ∧ 
    (∀ (x y : ℝ), circle_N x y → (t1.1 - x)^2 + (t1.2 - y)^2 ≥ (point_Q.1 - x)^2 + (point_Q.2 - y)^2) ∧
    (∀ (x y : ℝ), circle_N x y → (t2.1 - x)^2 + (t2.2 - y)^2 ≥ (point_Q.1 - x)^2 + (point_Q.2 - y)^2)) ∧
  -- Statement B
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_M 0 x1 y1 ∧ circle_M 0 x2 y2 ∧ circle_N x1 y1 ∧ circle_N x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 ≠ (Real.sqrt 15 / 4)^2) ∧
  -- Statement C
  (∀ a : ℝ, (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    circle_M a p1.1 p1.2 ∧ circle_M a p2.1 p2.2 ∧
    (p1.1 - point_B.1)^2 + (p1.2 - point_B.2)^2 = 1 ∧
    (p2.1 - point_B.1)^2 + (p2.2 - point_B.2)^2 = 1) ↔
    (-1 - Real.sqrt 7) / 2 < a ∧ a < (-1 + Real.sqrt 7) / 2) ∧
  -- Statement D
  (∀ p : ℝ × ℝ, circle_N p.1 p.2 →
    -2 ≤ ((p.1 - point_A.1) * (p.1 - point_B.1) + (p.2 - point_A.2) * (p.2 - point_B.2)) ∧
    ((p.1 - point_A.1) * (p.1 - point_B.1) + (p.2 - point_A.2) * (p.2 - point_B.2)) ≤ 18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l659_65986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_range_l659_65985

/-- The circle C with center (2, m) and radius 2 -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - m)^2 = 4

/-- The line l₁ passing through the origin -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x - y = 0

/-- The line l₂ passing through (4, -2) -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + a * y + 2 * a - 4 = 0

/-- The intersection point P of l₁ and l₂ -/
def intersection_point (a : ℝ) (x y : ℝ) : Prop :=
  line1 a x y ∧ line2 a x y

theorem circle_chord_range (m : ℝ) :
  m > 0 →
  (∃ A B : ℝ × ℝ, 
    circle_C m A.1 A.2 ∧
    circle_C m B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 ∧
    (∃ a : ℝ, ∃ P : ℝ × ℝ,
      intersection_point a P.1 P.2 ∧
      P.1 = (A.1 + B.1) / 2 ∧
      P.2 = (A.2 + B.2) / 2)) →
  Real.sqrt 5 - 2 ≤ m ∧ m ≤ Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_range_l659_65985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l659_65978

theorem election_votes_theorem (first_candidate_percentage : ℚ) (second_candidate_votes : ℕ) :
  first_candidate_percentage = 60 / 100 →
  second_candidate_votes = 480 →
  ∃ total_votes : ℕ,
    (first_candidate_percentage * total_votes + second_candidate_votes = total_votes) ∧
    total_votes = 1200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l659_65978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_two_l659_65927

theorem sum_of_reciprocals_equals_two (x y : ℝ) (h1 : (4 : ℝ)^x = 6) (h2 : (9 : ℝ)^y = 6) : 
  1/x + 1/y = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_two_l659_65927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_irrational_l659_65988

-- Define the given numbers
noncomputable def a : ℚ := 2/3
noncomputable def b : ℚ := 1.414
noncomputable def c : ℝ := Real.sqrt 3
noncomputable def d : ℚ := 3  -- √9 = 3

-- State the theorem
theorem sqrt_3_irrational :
  (a ∈ Set.univ) → (b ∈ Set.univ) → (d ∈ Set.univ) → c ∉ Set.univ :=
by
  intros ha hb hd
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_irrational_l659_65988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l659_65998

/-- The function f(x) = 2x / (2 + x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * x / (2 + x)

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => f (a n)

/-- The theorem stating the general term of the sequence -/
theorem a_formula (n : ℕ) : a n = 2 / (n + 1 : ℝ) := by
  sorry

/-- Auxiliary lemma: f(2) = 1 -/
lemma f_2_eq_1 : f 2 = 1 := by
  sorry

/-- Domain of f excludes -2 -/
lemma f_domain (x : ℝ) (h : x ≠ -2) : f x = 2 * x / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l659_65998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l659_65964

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 1
  | (n + 1) => sequence_a n + 1 / (4 * n^2 - 1)

theorem a_10_value : sequence_a 10 = 28/19 := by
  -- Computational proof
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l659_65964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l659_65935

/-- The ellipse C in the problem -/
def C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 = 1}

/-- The line l in the problem -/
def l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

/-- The intersection points of C and l -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := C ∩ l k

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Vector addition -/
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Vector length -/
noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_chord_theorem :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points (1/2) ∧ B ∈ intersection_points (1/2) ∧ A ≠ B ∧
  vec_length (vec_add (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2)) = vec_length (B.1 - A.1, B.2 - A.2) ∧
  vec_length (B.1 - A.1, B.2 - A.2) = 4 * Real.sqrt 65 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l659_65935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinates_l659_65971

/-- An equilateral triangle inscribed in a parabola with its centroid on a hyperbola -/
structure InscribedTriangle where
  /-- Vertices of the triangle -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  /-- Centroid of the triangle -/
  P : ℝ × ℝ
  /-- Triangle ABC is equilateral -/
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  /-- Triangle ABC is inscribed in the parabola x = y^2 -/
  inscribed_parabola : A.1 = A.2^2 ∧ B.1 = B.2^2 ∧ C.1 = C.2^2
  /-- P is the centroid of triangle ABC -/
  centroid : P.1 = (A.1 + B.1 + C.1) / 3 ∧ P.2 = (A.2 + B.2 + C.2) / 3
  /-- P lies on the hyperbola xy = 1 -/
  on_hyperbola : P.1 * P.2 = 1

/-- The main theorem: The centroid P has coordinates (3, 1/3) -/
theorem centroid_coordinates (t : InscribedTriangle) : t.P = (3, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinates_l659_65971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_evenness_l659_65949

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def has_real_domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y

noncomputable def power_function (a : ℝ) : ℝ → ℝ :=
  λ x ↦ x^a

theorem power_function_domain_and_evenness :
  ∀ a ∈ ({-1, 2, (1/2), 3} : Set ℝ),
    (has_real_domain (power_function a) ∧ is_even_function (power_function a)) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_evenness_l659_65949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_projections_l659_65918

/-- Represents the length of an edge in 3D space -/
def edge_length : ℝ := 2

/-- Represents the length of the edge's projection in the front view -/
def front_projection : ℝ := 2

/-- Theorem stating that the maximum value of a + b is 4 -/
theorem max_sum_of_projections :
  ∀ a b : ℝ, 
  a ≥ 0 → b ≥ 0 → 
  a ≤ edge_length → b ≤ edge_length → 
  front_projection = edge_length →
  a + b ≤ 4 :=
by
  intros a b ha hb ha_le hb_le hfront
  have h1 : a ≤ 2 := ha_le
  have h2 : b ≤ 2 := hb_le
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_projections_l659_65918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l659_65951

theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x, f x = (x + 3) * (2 * x - 1 / (4 * x * Real.sqrt x)) ^ 5) ∧
  (∃ c, ∀ x, f x = c + x * (f x - c) ∧ c = 15) := by
  -- The constant term in the expansion of (x+3)(2x- 1/(4x√x))^5 is equal to 15
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l659_65951
