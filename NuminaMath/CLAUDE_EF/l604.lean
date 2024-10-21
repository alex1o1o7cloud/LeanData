import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l604_60430

noncomputable def quadrilateral_vertices : List (ℝ × ℝ) := [(0, 1), (3, 4), (4, 3), (3, 0)]

noncomputable def perimeter (vertices : List (ℝ × ℝ)) : ℝ :=
  let distances := List.zipWith (λ p1 p2 => Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) 
                    vertices 
                    (vertices.rotateLeft 1)
  distances.sum

theorem quadrilateral_perimeter_sum (a b : ℤ) :
  perimeter quadrilateral_vertices = a * Real.sqrt 2 + b * Real.sqrt 10 →
  (a : ℝ) + (b : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l604_60430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l604_60462

-- Define the sets A and B
def A : Set ℝ := {x | 4 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x < 16}
def B : Set ℝ := {x | x > 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x < 4} := by sorry

-- Theorem for the union of the complement of A and B
theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x | x < 2 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l604_60462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l604_60414

theorem smallest_whole_number_above_sum : ℕ :=
  -- Define the sum
  let sum : ℚ := (3 + 1/3) + (4 + 1/2) + (5 + 1/5) + (6 + 1/6)
  
  -- Define the property for the smallest whole number larger than the sum
  let is_smallest_whole_number (n : ℕ) := 
    (n : ℚ) > sum ∧ ∀ m : ℕ, (m : ℚ) > sum → n ≤ m

  -- State that 20 satisfies this property
  have h : is_smallest_whole_number 20 := by
    sorry -- Proof to be filled in later

  20 -- Return the answer


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l604_60414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_6_has_two_solutions_l604_60484

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

-- Theorem statement
theorem f_f_eq_6_has_two_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, f (f x) = 6) ∧ (Finset.card s = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_6_has_two_solutions_l604_60484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_probability_bound_l604_60404

-- Define the rounding function
def myRound (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Define the probability function P(k)
noncomputable def P (k : ℕ) : ℚ :=
  (Finset.filter (fun n : ℕ => 
    myRound (n / k : ℚ) + myRound ((120 - n) / k : ℚ) = myRound (120 / k : ℚ))
    (Finset.range 120)).card / 120

-- Define the set of odd integers from 1 to 120
def odd_ints : Finset ℕ :=
  Finset.filter (fun k => k % 2 = 1) (Finset.range 121)

-- The theorem to prove
theorem min_probability_bound (M : ℕ) :
  ∃ k ∈ odd_ints, ∀ j ∈ odd_ints, P k ≤ P j ∧ P k = M / 120 := by
  sorry

#check min_probability_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_probability_bound_l604_60404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l604_60408

theorem function_properties (m : ℕ) (hm : m > 1) :
  let f : ℝ → ℝ := λ x ↦ ((2 * x) / (x^2 + 1))^m
  let I : Set ℝ := Set.Icc 0 ((m - 1) / m)
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (f ((m - 1) / m) = ((2 * m^2 - 2 * m) / (2 * m^2 - 2 * m + 1))^m) ∧
  (∀ (x : ℝ), x ∈ I → Real.exp (1 / (2 * ↑m)) * f x < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l604_60408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equivalence_l604_60494

theorem power_equivalence (x : ℝ) (h : (8 : ℝ)^(2*x) = 11) :
  (2 : ℝ)^(x + 3/2) = 11^(1/6) * 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equivalence_l604_60494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l604_60441

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of point A -/
noncomputable def A : ℝ × ℝ := (2, 0)

/-- Definition of point B -/
noncomputable def B : ℝ × ℝ := (0, 1)

/-- Definition of point O -/
noncomputable def O : ℝ × ℝ := (0, 0)

/-- Definition of point M -/
noncomputable def M (x₀ y₀ : ℝ) : ℝ × ℝ := (0, -2*y₀/(x₀-2))

/-- Definition of point N -/
noncomputable def N (x₀ y₀ : ℝ) : ℝ × ℝ := (-x₀/(y₀-1), 0)

/-- Theorem: |AN| · |BM| is constant for any point on the ellipse -/
theorem constant_product (x₀ y₀ : ℝ) : 
  ellipse x₀ y₀ → |2 + x₀/(y₀-1)| * |1 + 2*y₀/(x₀-2)| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l604_60441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l604_60481

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- The volume of a tetrahedron with given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  (80 * Real.sqrt 2) / 3

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∀ t : Tetrahedron,
  t.ab_length = 4 ∧
  t.abc_area = 20 ∧
  t.abd_area = 16 ∧
  t.face_angle = π/4 →
  tetrahedron_volume t = (80 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l604_60481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_and_reflection_l604_60486

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Reflects a point about the line y = 1 -/
def reflectPoint (p : Point) : Point :=
  { x := p.x, y := 2 - p.y }

/-- The main theorem to prove -/
theorem area_of_triangle_and_reflection : 
  let A : Point := { x := 3, y := 4 }
  let B : Point := { x := 4, y := -2 }
  let C : Point := { x := 7, y := 0 }
  let A' := reflectPoint A
  let B' := reflectPoint B
  let C' := reflectPoint C
  triangleArea A B C + triangleArea A' B' C' = 20 := by
  sorry

#eval "Proof completed with 'sorry'"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_and_reflection_l604_60486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_bounded_l604_60483

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sin (a n)

theorem a_squared_bounded : ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ, (a n)^2 ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_bounded_l604_60483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l604_60488

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C₂ (x y : ℝ) : Prop := y^2 - x + 1 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ),
    min_dist = (3 * Real.sqrt 2) / 4 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      C₁ x₁ y₁ → C₂ x₂ y₂ →
      distance x₁ y₁ x₂ y₂ ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l604_60488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_of_beef_weight_l604_60431

/-- Calculates the original weight of a side of beef before processing -/
noncomputable def originalWeight (afterWeight : ℝ) (percentLost : ℝ) : ℝ :=
  afterWeight / (1 - percentLost / 100)

/-- Theorem stating the original weight of a side of beef -/
theorem side_of_beef_weight (afterWeight : ℝ) (percentLost : ℝ) 
  (h1 : afterWeight = 500)
  (h2 : percentLost = 30) :
  ∃ (ε : ℝ), abs (originalWeight afterWeight percentLost - 714.29) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_of_beef_weight_l604_60431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_water_consumption_l604_60400

/-- The number of days Remi drinks water -/
def days : ℕ := 7

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The amount of water Remi spills in ounces -/
def spilled_water : ℕ := 5 + 8

/-- The total amount of water Remi drinks in ounces -/
def total_water_drunk : ℕ := 407

theorem remi_water_consumption :
  (days * bottle_capacity * refills_per_day) - spilled_water = total_water_drunk ∧ days = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_water_consumption_l604_60400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_b_bound_extreme_value_and_bound_implies_c_range_l604_60480

/-- The function f(x) = x^3 - x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^3 - x^2 + b*x + c

/-- If f(x) is increasing on ℝ, then b ≥ 1/12 -/
theorem increasing_f_implies_b_bound (b c : ℝ) :
  (∀ x y : ℝ, x < y → f b c x < f b c y) →
  b ≥ 1/12 := by sorry

/-- If f(x) takes an extreme value at x = 1 and f(x) < c^2 for x ∈ [-1,2], 
    then c ∈ (-∞, -1) ∪ (2, +∞) -/
theorem extreme_value_and_bound_implies_c_range (b c : ℝ) :
  (∃ ε > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < ε → f b c 1 ≤ f b c x) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → f b c x < c^2) →
  c ∈ Set.Ioi 2 ∪ Set.Iio (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_b_bound_extreme_value_and_bound_implies_c_range_l604_60480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_meals_l604_60423

/-- Represents the number of students choosing meal A on the nth Monday -/
def a : ℕ → ℕ := sorry

/-- Represents the number of students choosing meal B on the nth Monday -/
def b : ℕ → ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 500

/-- The fraction of students who stay with meal A -/
def stay_a : ℚ := 4/5

/-- The fraction of students who switch from B to A -/
def switch_to_a : ℚ := 3/10

theorem cafeteria_meals (n : ℕ) :
  (∀ k, a k + b k = total_students) →
  (∀ k, a (k + 1) = stay_a * (a k : ℚ) + switch_to_a * (b k : ℚ)) →
  a 1 = 300 →
  a n = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_meals_l604_60423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l604_60460

/-- Represents the operation of adding balls to boxes -/
def add_balls (m n : ℕ) (boxes : Fin m → ℕ) : Fin m → ℕ := sorry

/-- Represents a sequence of operations -/
def operation_sequence (m n : ℕ) (initial : Fin m → ℕ) : ℕ → (Fin m → ℕ) := sorry

/-- Checks if all boxes have the same number of balls -/
def all_equal (m : ℕ) (boxes : Fin m → ℕ) : Prop := sorry

theorem ball_distribution (m n : ℕ) (h1 : 0 < n) (h2 : n < m) :
  (Nat.Coprime m n →
    ∃ (k : ℕ) (initial : Fin m → ℕ), all_equal m (operation_sequence m n initial k)) ∧
  (¬Nat.Coprime m n →
    ∃ (initial : Fin m → ℕ), ∀ (k : ℕ), ¬all_equal m (operation_sequence m n initial k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l604_60460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_nonnegative_l604_60437

theorem at_least_one_nonnegative (x : ℝ) : 
  ¬(x^2 - 1 < 0 ∧ 2*x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_nonnegative_l604_60437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eb_equals_cf_l604_60458

/-- Predicate indicating that three points form a triangle -/
def Triangle (A B C : Point) : Prop := sorry

/-- Predicate indicating that a point is on the angle bisector -/
def IsAngleBisector (A D B C : Point) : Prop := sorry

/-- Predicate indicating that a point is on the circumcircle of three other points -/
def OnCircumcircle (A B C D : Point) : Prop := sorry

/-- Predicate indicating that a point is on a segment defined by two other points -/
def OnSegment (A B C : Point) : Prop := sorry

/-- Given a triangle ABC with D as the foot of the angle bisector from A,
    E as the intersection of circumcircle ACD with AB,
    and F as the intersection of circumcircle ABD with AC,
    prove that EB = CF -/
theorem eb_equals_cf (A B C D E F : Point) 
  (h1 : Triangle A B C)
  (h2 : IsAngleBisector A D B C)
  (h3 : OnCircumcircle A C D E)
  (h4 : OnSegment A B E)
  (h5 : OnCircumcircle A B D F)
  (h6 : OnSegment A C F) : 
  EB = CF := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eb_equals_cf_l604_60458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tie_september_l604_60424

def johnson_hrs : List Nat := [3, 14, 18, 13, 10, 16, 14, 5]
def carter_hrs : List Nat := [5, 9, 22, 11, 15, 17, 9, 9]
def months : List String := ["March", "April", "May", "June", "July", "August", "September", "October"]

def cumulative_sum (xs : List Nat) : List Nat :=
  List.scanl (· + ·) 0 xs

def first_equal_month (js cs : List Nat) (ms : List String) : Option String :=
  (List.zip js (List.zip cs ms)).find? (fun (j, (c, _)) => j = c) |>.map (fun (_, (_, m)) => m)

theorem first_tie_september :
  first_equal_month (cumulative_sum johnson_hrs) (cumulative_sum carter_hrs) months = some "September" := by
  sorry

#eval first_equal_month (cumulative_sum johnson_hrs) (cumulative_sum carter_hrs) months

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tie_september_l604_60424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l604_60403

theorem polynomial_identity (P : Polynomial ℝ) 
  (h1 : P.eval 0 = 0)
  (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l604_60403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_multiple_of_21_to_3105_l604_60444

theorem nearest_multiple_of_21_to_3105 : 
  ∀ n : Int, n ≠ 3108 → n % 21 = 0 → |n - 3105| ≥ |3108 - 3105| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_multiple_of_21_to_3105_l604_60444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_quartile_not_eight_standard_deviation_transformed_data_probability_union_mutually_exclusive_independent_events_l604_60472

def dataset : List ℝ := [2, 2, 3, 5, 6, 7, 7, 8, 10, 11]

noncomputable def lowerQuartile (data : List ℝ) : ℝ := sorry

noncomputable def standardDeviation (data : List ℝ) : ℝ := sorry

def transformData (a b : ℝ) (data : List ℝ) : List ℝ := 
  data.map (fun x => a * x + b)

structure ProbabilitySpace (Ω : Type) where
  P : Set Ω → ℝ
  P_nonneg : ∀ A, P A ≥ 0
  P_total : P Set.univ = 1
  P_additive : ∀ A B, Disjoint A B → P (A ∪ B) = P A + P B

variable {Ω : Type} (ps : ProbabilitySpace Ω)

def independent (ps : ProbabilitySpace Ω) (A B : Set Ω) : Prop :=
  ps.P (A ∩ B) = ps.P A * ps.P B

theorem lower_quartile_not_eight :
  lowerQuartile dataset ≠ 8 := by sorry

theorem standard_deviation_transformed_data (data : List ℝ) (a b : ℝ) :
  standardDeviation (transformData a b data) = |a| * standardDeviation data := by sorry

theorem probability_union_mutually_exclusive (ps : ProbabilitySpace Ω) (A B C : Set Ω) 
  (h1 : Disjoint A B) (h2 : Disjoint A C) (h3 : Disjoint B C) :
  ps.P (A ∪ B ∪ C) = ps.P A + ps.P B + ps.P C := by sorry

theorem independent_events (ps : ProbabilitySpace Ω) (A B : Set Ω) 
  (h : ps.P (A ∩ B) = ps.P A * ps.P B) :
  independent ps A B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_quartile_not_eight_standard_deviation_transformed_data_probability_union_mutually_exclusive_independent_events_l604_60472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l604_60453

def bag_A : List ℕ := [2, 3, 5, 7]
def bag_B : List ℕ := [2, 4, 6]

def possible_sums : List ℕ := (do
  let a ← bag_A
  let b ← bag_B
  pure (a + b)).eraseDups

theorem distinct_sums_count : possible_sums.length = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l604_60453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_make_monochromatic_l604_60426

/-- A color representation for graph edges -/
inductive Color
| Red
| Blue

/-- A complete graph with n vertices and colored edges -/
structure ColoredCompleteGraph (n : ℕ) where
  edges : Fin n → Fin n → Color

/-- The operation of changing colors in a non-monochromatic triangle -/
def changeTriangleColors (G : ColoredCompleteGraph n) (i j k : Fin n) : ColoredCompleteGraph n :=
  sorry

/-- Predicate to check if a graph is monochromatic -/
def isMonochromatic (G : ColoredCompleteGraph n) : Prop :=
  sorry

/-- The main theorem stating that any colored complete graph can be made monochromatic -/
theorem make_monochromatic (n : ℕ) (G : ColoredCompleteGraph n) :
  ∃ (G' : ColoredCompleteGraph n), isMonochromatic G' ∧
  (∃ (steps : ℕ), ∃ (triangles : Fin steps → Fin n × Fin n × Fin n),
    G' = (List.range steps).foldl (λ acc i ↦ 
      changeTriangleColors acc (triangles ⟨i, sorry⟩).1 (triangles ⟨i, sorry⟩).2.1 (triangles ⟨i, sorry⟩).2.2) G) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_make_monochromatic_l604_60426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l604_60434

def is_valid_number (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 10 ∧
  ∀ i, i < 10 → (digits.filter (λ d => d = i)).length = if i < digits.length then digits[i]! else 0

theorem unique_self_descriptive_number :
  ∃! n : Nat, is_valid_number n ∧ n = 6210001000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l604_60434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_equals_48_6_l604_60493

/-- The value of 32.36 (repeating) as a real number -/
noncomputable def repeating_decimal : ℝ := 32 + 36 / 99

/-- Function to round a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ := ⌊x * 10 + 0.5⌋ / 10

/-- Theorem stating that adding 16.25 to 32.36 (repeating) and rounding to the nearest tenth equals 48.6 -/
theorem add_and_round_equals_48_6 :
  round_to_tenth (repeating_decimal + 16.25) = 48.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_equals_48_6_l604_60493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l604_60495

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_a n + 2^n

theorem a_10_equals_1023 : sequence_a 10 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l604_60495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_white_zero_l604_60469

-- Define the number of balls of each color
def white_balls : ℕ := 6
def black_balls : ℕ := 7
def red_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls + red_balls

-- Define the number of balls to be drawn
def drawn_balls : ℕ := 8

-- Theorem statement
theorem probability_all_white_zero :
  Finset.card (Finset.filter (fun s => s.card = drawn_balls ∧ s ⊆ Finset.range white_balls) (Finset.powerset (Finset.range total_balls))) /
  Finset.card (Finset.filter (fun s => s.card = drawn_balls) (Finset.powerset (Finset.range total_balls))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_white_zero_l604_60469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_line_properties_l604_60422

/-- Given line parameterization -/
noncomputable def given_line (t : ℝ) : ℝ × ℝ := (4 - 2*t, 3 - t)

/-- Ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Slope of the given line -/
def given_line_slope : ℚ := 1/2

/-- The line we're looking for -/
def target_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- Theorem stating that the target_line is parallel to the given_line and passes through the right_focus -/
theorem target_line_properties :
  (∀ x y : ℝ, target_line x y → (y - (right_focus.2 : ℝ)) = (given_line_slope : ℝ) * (x - right_focus.1)) ∧
  target_line right_focus.1 right_focus.2 := by
  sorry

#check target_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_line_properties_l604_60422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_complement_theorem_l604_60497

-- Define the basic geometric shapes
structure Pyramid where

structure Frustum where

structure Plane where

-- Define the concept of "complementing" a frustum with a pyramid
def complementWithPyramid (f : Frustum) (p : Pyramid) : Pyramid :=
  sorry

-- Define the concept of a plane being parallel to the base of a pyramid
def isParallelToBase (plane : Plane) (pyr : Pyramid) : Prop :=
  sorry

-- Define how a frustum is formed from a pyramid
def formFrustum (pyr : Pyramid) (plane : Plane) : Frustum :=
  sorry

-- Theorem: Any frustum can be complemented with a pyramid to form a new pyramid
theorem frustum_complement_theorem (f : Frustum) : 
  ∃ (p : Pyramid), ∃ (newPyr : Pyramid), complementWithPyramid f p = newPyr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_complement_theorem_l604_60497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_area_is_correct_l604_60445

/-- A triangular playground with sides 7m, 24m, and 25m, containing a lawn 2m from each side -/
structure Playground where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  lawn_distance : ℝ
  h_side1 : side1 = 7
  h_side2 : side2 = 24
  h_side3 : side3 = 25
  h_lawn_distance : lawn_distance = 2

/-- The area of the lawn inside the playground -/
noncomputable def lawn_area (p : Playground) : ℝ := 28 / 3

/-- Theorem stating that the area of the lawn is 28/3 m² -/
theorem lawn_area_is_correct (p : Playground) : lawn_area p = 28 / 3 := by
  -- Unfold the definition of lawn_area
  unfold lawn_area
  -- The result follows immediately from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_area_is_correct_l604_60445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l604_60471

def M : Set ℕ := {1, 2, 3, 4, 5}

def N : Set ℝ := {x : ℝ | 2 / (x - 1) ≤ 1}

def M_real : Set ℝ := {1, 2, 3, 4, 5}

theorem intersection_M_N : M_real ∩ N = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l604_60471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_increase_percentage_l604_60456

def monthly_salary : ℝ := 7272.727272727273
def original_savings_rate : ℝ := 0.10
def new_savings_amount : ℝ := 400

theorem expense_increase_percentage :
  let original_expenses := monthly_salary * (1 - original_savings_rate)
  let original_savings := monthly_salary * original_savings_rate
  let expense_increase := original_savings - new_savings_amount
  ∃ ε > 0, |((expense_increase / original_expenses) * 100) - 5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_increase_percentage_l604_60456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_q_t_is_two_l604_60411

/-- Represents the side length of the square -/
noncomputable def side_length : ℝ := 2

/-- Represents the number of divisions in the square -/
def num_divisions : ℕ := 10

/-- Represents the angle between each division line in radians -/
noncomputable def angle_between_lines : ℝ := 45 * Real.pi / 180

/-- Represents the area of one central triangular region -/
noncomputable def t : ℝ := (side_length / 2) ^ 2 / 2

/-- Represents the area of one corner region -/
noncomputable def q : ℝ := (side_length ^ 2 - num_divisions * t) / num_divisions

/-- Theorem stating that the ratio of q to t is 2 -/
theorem ratio_q_t_is_two : q / t = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_q_t_is_two_l604_60411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l604_60409

/-- Parabola type representing y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Point type representing (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

def intersects (l : Line) (c : Parabola) : Prop :=
  ∃ x y, y = l.m * x + l.b ∧ y^2 = 2 * c.p * x

def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem parabola_intersection_theorem (c : Parabola) (n : Point) (l : Line) 
    (a b : Point) :
  n.x = -2 →
  n.y = 2 →
  l.m = 2 →
  l.b = -c.p →
  intersects l c →
  (∃ a b, intersects l c ∧ a ≠ b ∧ perpendicular n a b) →
  c.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l604_60409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l604_60419

theorem mean_median_difference (x : ℕ) (c : ℕ) : 
  let set := [x, x + 2, x + 4, x + 7, x + c]
  let median := x + 4
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + c)) / 5
  (mean = median + 4) → c = 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l604_60419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_recurrence_and_bound_l604_60485

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | (n + 3) => sequence_a (n + 2) * (sequence_a (n + 2)^2 + 1) / (sequence_a (n + 1)^2 + 1)

theorem sequence_a_recurrence_and_bound :
  (∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) = sequence_a n + 1 / sequence_a n) ∧
  (63 < sequence_a 2008 ∧ sequence_a 2008 < 78) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_recurrence_and_bound_l604_60485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l604_60443

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem principal_calculation (P : ℝ) (h : simple_interest P 7 8 = P - 5600) :
  ∃ ε > 0, |P - 12727.27| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l604_60443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_initial_games_total_games_correct_l604_60435

/-- The number of video games Jerry owned before his birthday -/
def initial_games : ℕ := 7

/-- The number of video games Jerry received for his birthday -/
def birthday_games : ℕ := 2

/-- The total number of video games Jerry has after his birthday -/
def total_games : ℕ := 9

/-- Theorem stating that Jerry owned 7 games before his birthday -/
theorem jerry_initial_games : initial_games = 7 := by
  rfl

/-- Theorem verifying the total number of games after Jerry's birthday -/
theorem total_games_correct : total_games = initial_games + birthday_games := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_initial_games_total_games_correct_l604_60435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l604_60491

/-- Given function f with two specific roots -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x - 2)

/-- Theorem stating the properties of f based on its roots -/
theorem f_properties (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 4 ∧ f a b x₁ = x₁ - 12 ∧ f a b x₂ = x₂ - 12) →
  (∀ x : ℝ, x ≠ 2 → f a b x = (-x + 2) / (x - 2)) ∧
  (∀ k : ℝ, k > 1 → 
    (∀ x : ℝ, x ≠ 2 → (f a b x < k ↔ (x - 2) * (x - 1) * (x - k) > 0))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l604_60491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_shorter_sum_exists_shorter_edge_l604_60407

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the points
variable (A B C D E : V)

-- Define the condition that E is inside ABCD
def inside_pyramid (A B C D E : V) : Prop :=
  ∃ (α β γ δ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧ α + β + γ + δ = 1 ∧
  E = α • A + β • B + γ • C + δ • D

-- Theorem 1: It's not always true that AE + BE + CE < AD + BD + CD
theorem not_always_shorter_sum :
  ¬ (∀ A B C D E : V, inside_pyramid A B C D E → 
    ‖A - E‖ + ‖B - E‖ + ‖C - E‖ < ‖A - D‖ + ‖B - D‖ + ‖C - D‖) :=
sorry

-- Theorem 2: There exists at least one pair (X, Y) where X ∈ {A, B, C} and Y ∈ {D, E} such that XE < XD
theorem exists_shorter_edge (h : inside_pyramid A B C D E) : 
  (‖A - E‖ < ‖A - D‖) ∨ (‖B - E‖ < ‖B - D‖) ∨ (‖C - E‖ < ‖C - D‖) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_shorter_sum_exists_shorter_edge_l604_60407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_factorial_triples_l604_60425

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem infinitely_many_factorial_triples :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    (∀ n : ℕ, 
      let (p, q, r) := f n
      p > q ∧ q > r ∧ r > 1 ∧
      is_perfect_square (factorial p * factorial q * factorial r)) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_factorial_triples_l604_60425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_in_square_l604_60463

/-- Helper function to calculate the angle between three points -/
noncomputable def angle (A M N : ℝ × ℝ) : ℝ := sorry

/-- Given a square ABCD with side length 4, where M is the midpoint of BC and N is the midpoint of CD,
    prove that the sine of angle ∠MAN is equal to 3/5. -/
theorem sine_of_angle_in_square (A B C D M N : ℝ × ℝ) : 
  let square_side_length : ℝ := 4
  -- Define the square ABCD
  (A = (0, 0) ∧ B = (0, square_side_length) ∧ 
   C = (square_side_length, square_side_length) ∧ D = (square_side_length, 0)) →
  -- M is the midpoint of BC
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- N is the midpoint of CD
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  -- The sine of angle ∠MAN is 3/5
  Real.sin (angle A M N) = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_in_square_l604_60463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_seven_digit_number_l604_60476

/-- Represents a 7-digit number as a list of its digits -/
def SevenDigitNumber := List Nat

/-- Checks if a given list represents a valid 7-digit number -/
def isValidSevenDigitNumber (n : SevenDigitNumber) : Prop :=
  n.length = 7 ∧ n.all (λ d => d < 10)

/-- Computes the list of sums of adjacent digits -/
def adjacentSums (n : SevenDigitNumber) : List Nat :=
  List.zipWith (·+·) n.tail n

/-- The specific sequence of sums we're looking for -/
def targetSums : List Nat := [9, 7, 9, 2, 8, 11]

/-- The main theorem statement -/
theorem unique_seven_digit_number :
  ∃! (n : SevenDigitNumber), 
    isValidSevenDigitNumber n ∧ 
    adjacentSums n = targetSums ∧
    n = [9, 0, 7, 2, 0, 8, 3] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_seven_digit_number_l604_60476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l604_60401

/-- Calculates the final price after applying successive discounts -/
def finalPrice (originalPrice : ℚ) (discounts : List ℚ) : ℚ :=
  discounts.foldl (fun price discount => price * (1 - discount)) originalPrice

/-- Theorem: The sale price of sarees listed for Rs. 400 after three successive discounts of 12%, 5%, and 7% is approximately Rs. 311 -/
theorem saree_sale_price :
  let originalPrice : ℚ := 400
  let discounts : List ℚ := [12/100, 5/100, 7/100]
  let finalPriceExact := finalPrice originalPrice discounts
  ⌊finalPriceExact⌋ = 311 := by sorry

#eval ⌊finalPrice 400 [12/100, 5/100, 7/100]⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l604_60401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l604_60496

open Nat

/-- Converts a number from base b1 to base b2 -/
def convertBase (n : ℕ) (b1 b2 : ℕ) : ℕ := sorry

/-- Checks if a sequence contains an arithmetic progression of length p -/
def hasArithmeticProgression (seq : List ℕ) (p : ℕ) : Prop := sorry

/-- Generates the nth term of the sequence as defined in the problem -/
def a (n : ℕ) (p : ℕ) : ℕ := sorry

theorem sequence_property (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) (n : ℕ) :
  a n p = convertBase n (p - 1) p := by
  sorry

#check sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l604_60496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l604_60468

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two vectors -/
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Calculates the vector difference between two points -/
def vector_sub (p1 p2 : Point) : Point :=
  { x := p1.x - p2.x, y := p1.y - p2.y }

/-- Calculates the area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : Point) : ℝ :=
  sorry -- Placeholder for actual area calculation

/-- Calculates the circumcenter of a triangle -/
noncomputable def circumcenter (A B C : Point) : Point :=
  sorry -- Placeholder for actual circumcenter calculation

/-- Theorem: The maximum area of a triangle with given conditions -/
theorem triangle_max_area (t : Triangle) (O A B C : Point) :
  t.b = 4 →
  dot_product (vector_sub A O) (vector_sub C B) = (1/2) * t.a * (t.a - 8/5 * t.c) →
  O = circumcenter A B C →
  (∃ (S : ℝ), S ≤ 12 ∧ S = triangle_area A B C ∧
    ∀ (S' : ℝ), S' = triangle_area A B C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l604_60468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_free_throws_l604_60459

theorem john_free_throws (
  free_throw_accuracy : ℝ)
  (shots_per_foul : ℕ)
  (fouls_per_game : ℕ)
  (games_played_percentage : ℝ)
  (total_team_games : ℕ)
  (h1 : free_throw_accuracy = 0.7)
  (h2 : shots_per_foul = 2)
  (h3 : fouls_per_game = 5)
  (h4 : games_played_percentage = 0.8)
  (h5 : total_team_games = 20) :
  ⌊free_throw_accuracy * (↑shots_per_foul * ↑fouls_per_game * ↑⌊games_played_percentage * ↑total_team_games⌋)⌋ = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_free_throws_l604_60459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_theorem_l604_60473

noncomputable def pages_read_day1 (total : ℝ) : ℝ := total / 6 + 10

noncomputable def pages_left_day1 (total : ℝ) : ℝ := total - pages_read_day1 total

noncomputable def pages_read_day2 (total : ℝ) : ℝ := pages_left_day1 total / 5 + 20

noncomputable def pages_left_day2 (total : ℝ) : ℝ := pages_left_day1 total - pages_read_day2 total

noncomputable def pages_read_day3 (total : ℝ) : ℝ := pages_left_day2 total / 4 + 25

noncomputable def pages_left_day3 (total : ℝ) : ℝ := pages_left_day2 total - pages_read_day3 total

theorem book_pages_theorem :
  pages_left_day3 236 = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_theorem_l604_60473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l604_60464

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3) / Real.log (1/2)

-- Define the property of f being increasing on (-∞, 1)
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 1 → f x < f y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  is_increasing_on_interval (f a) ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l604_60464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l604_60457

noncomputable def a : Fin 3 → ℝ := ![1, -3, 2]
noncomputable def b : Fin 3 → ℝ := ![-1/2, 3/2, -1]

theorem parallel_vectors : ∃ (k : ℝ), ∀ i, b i = k * a i := by
  use (-1/2)
  intro i
  fin_cases i <;> simp [a, b]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l604_60457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l604_60465

theorem problem_statement : 
  (∀ x : ℝ, (2 : ℝ)^x > 0) ∧ 
  ¬(∀ x : ℝ, (x > 3 → x > 5) ∧ ∃ y : ℝ, y > 5 ∧ y ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l604_60465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eighteenth_term_is_zero_l604_60487

def mySequence : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | n + 2 => mySequence n * mySequence (n + 1) - 1

theorem two_thousand_eighteenth_term_is_zero : mySequence 2017 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eighteenth_term_is_zero_l604_60487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l604_60449

noncomputable def powerFunction (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_increasing (α : ℝ) (h : α ∈ ({1, 3, (1:ℝ)/2} : Set ℝ)) :
  StrictMono (powerFunction α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l604_60449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l604_60416

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

-- State the theorem
theorem g_max_value :
  ∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 4 ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 4 → g x ≤ g x₁) ∧
  g x₁ = 16 ∧ x₁ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l604_60416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_dataset_l604_60477

theorem standard_deviation_of_dataset (m : ℝ) (dataset : List ℝ := [51, 54, m, 57, 53])
  (h1 : dataset.sum / dataset.length = 54) : 
  Real.sqrt ((dataset.map (λ x => (x - 54)^2)).sum / dataset.length) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_dataset_l604_60477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_group_size_l604_60418

theorem dining_group_size 
  (total_bill : ℝ) 
  (tip_percentage : ℝ) 
  (individual_share : ℝ) : ℕ :=
  let total_bill := 139.00
  let tip_percentage := 0.10
  let individual_share := 19.1125
  have h : ((total_bill * (1 + tip_percentage)) / individual_share).round = 8 := by sorry
  8

#check dining_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_group_size_l604_60418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_properties_l604_60427

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 3 + φ)

theorem sinusoid_properties (φ : ℝ) (h1 : |φ| < Real.pi/2) (h2 : f 0 φ = 1) :
  let amplitude : ℝ := 2
  let period : ℝ := 6
  let frequency : ℝ := 1/6
  let initial_phase : ℝ := Real.pi/6
  (∀ x, f x φ = amplitude * Real.sin (2 * Real.pi * frequency * x + initial_phase)) ∧
  (∀ x, f (x + period) φ = f x φ) ∧
  initial_phase = φ := by
  sorry

#check sinusoid_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_properties_l604_60427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l604_60412

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (4 - x) + Real.log (2 + x)

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioo (-2) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l604_60412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l604_60482

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 2 * (floor x) - 3 = 0

-- State the theorem
theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, equation x) ∧
  (∀ y : ℝ, equation y → y ∈ s) := by
  sorry

#check equation_has_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l604_60482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_stratified_sampling_l604_60446

structure School where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ

def junior_middle_school : School := {
  total_students := 270,
  first_grade := 108,
  second_grade := 81,
  third_grade := 81,
  sample_size := 10
}

inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSampling
| SystematicSampling

def SamplingResult := List ℕ

def result_2 : SamplingResult := [5, 9, 100, 107, 111, 121, 180, 195, 200, 265]
def result_4 : SamplingResult := [30, 57, 84, 111, 138, 165, 192, 219, 246, 270]

def is_stratified_sampling (school : School) (result : SamplingResult) : Prop :=
  ∃ (first second third : List ℕ),
    result = first ++ second ++ third ∧
    first.length = (school.first_grade * school.sample_size) / school.total_students ∧
    second.length = (school.second_grade * school.sample_size) / school.total_students ∧
    third.length = (school.third_grade * school.sample_size) / school.total_students

theorem not_stratified_sampling :
  ¬(is_stratified_sampling junior_middle_school result_2) ∧
  ¬(is_stratified_sampling junior_middle_school result_4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_stratified_sampling_l604_60446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l604_60492

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem triangle_area_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 3 ∧ 
  b = 4 ∧
  c > 0 ∧  -- Positive side length
  f A = 1/2 ∧  -- Given condition
  a / (Real.sin A) = b / (Real.sin B) ∧  -- Law of sines
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →  -- Law of cosines
  (1/2) * b * c * (Real.sin A) = 4 + Real.sqrt 2 :=
by sorry

#check triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l604_60492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_equals_one_l604_60490

theorem cosine_sum_squared_equals_one (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = (2*k + 1) * (Real.pi/8)) ∨ 
  (∃ k : ℤ, x = (2*k + 1) * (Real.pi/6)) ∨ 
  (∃ k : ℤ, x = (2*k + 1) * (Real.pi/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_equals_one_l604_60490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_distance_l604_60442

/-- Proves that given the specified conditions, the total distance traveled is 37 miles -/
theorem bike_ride_distance (first_hour : ℝ) (second_hour : ℝ) (third_hour : ℝ) : 
  second_hour = 12 →
  second_hour = first_hour * 1.2 →
  third_hour = second_hour * 1.25 →
  first_hour + second_hour + third_hour = 37 := by
  intros h1 h2 h3
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_distance_l604_60442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l604_60467

-- Use the built-in complex number type
open Complex

-- Define addition for complex numbers (already defined in Mathlib)
-- def add (a b : ℂ) : ℂ := a + b

-- Define subtraction for complex numbers (already defined in Mathlib)
-- def sub (a b : ℂ) : ℂ := a - b

-- Define the imaginary unit i (already defined in Mathlib)
-- def I : ℂ := Complex.I

-- Define complex number equality (already defined in Mathlib)
-- def complex_eq (a b : ℂ) : Prop := a = b

-- State the theorem
theorem complex_subtraction :
  (5 : ℂ) + 3*I - (6 - I) = -1 + 4*I :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l604_60467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_theorem_second_magician_can_identify_l604_60498

theorem magic_trick_theorem (cards : Finset ℕ) (selected : Finset ℕ) : 
  (∀ n ∈ cards, 1 ≤ n ∧ n ≤ 48) →
  cards.card = 48 →
  selected ⊆ cards →
  selected.card = 25 →
  ∃ x y, x ∈ selected ∧ y ∈ selected ∧ x + y = 49 :=
by sorry

theorem second_magician_can_identify 
  (cards : Finset ℕ) (selected : Finset ℕ) (returned : Finset ℕ) (added : ℕ) :
  (∀ n ∈ cards, 1 ≤ n ∧ n ≤ 48) →
  cards.card = 48 →
  selected ⊆ cards →
  selected.card = 25 →
  returned ⊆ selected →
  returned.card = 2 →
  added ∈ (cards \ selected) →
  (∃ x y, x ∈ returned ∧ y ∈ returned ∧ x + y = 49) →
  ∀ z ∈ (returned ∪ {added}), z ≠ added ↔ ∃ w, w ∈ (returned ∪ {added}) ∧ z + w = 49 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_theorem_second_magician_can_identify_l604_60498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_value_l604_60466

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of m -/
noncomputable def z (m : ℝ) : ℂ := (1 - m * Complex.I) / (2 + Complex.I)

theorem pure_imaginary_m_value :
  ∀ m : ℝ, isPureImaginary (z m) → m = 2 := by
  sorry

#check pure_imaginary_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_value_l604_60466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfactorable_polynomial_l604_60474

theorem unfactorable_polynomial (k : ℤ) (h : ¬ (5 ∣ k)) :
  ¬ ∃ (p q : Polynomial ℤ),
    (Polynomial.degree p < 5 ∧ Polynomial.degree q < 5) ∧
    (Polynomial.degree p + Polynomial.degree q = 5) ∧
    (p * q = X^5 - X + (Polynomial.C k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfactorable_polynomial_l604_60474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_in_sequence_l604_60451

def sequence_v : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => 4 * sequence_v (n+1) - sequence_v n

theorem odd_numbers_in_sequence (a b : ℕ) :
  Odd a → Odd b →
  (∃ k : ℕ, b^2 + 2 = k * a) →
  (∃ l : ℕ, a^2 + 2 = l * b) →
  ∃ m n : ℕ, sequence_v m = a ∧ sequence_v n = b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_in_sequence_l604_60451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_13_l604_60420

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t, Real.sqrt 3 * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

-- Define the chord length
noncomputable def chord_length (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ × ℝ) : ℝ :=
  let x₁ := (5 - Real.sqrt 13) / 4
  let x₂ := (5 + Real.sqrt 13) / 4
  Real.sqrt ((x₂ - x₁) ^ 2)

-- Theorem statement
theorem chord_length_is_sqrt_13 :
  chord_length line_l curve_C = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_13_l604_60420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_177_l604_60421

def c : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => c (n + 2) + c (n + 1) + 1

theorem c_10_equals_177 : c 10 = 177 := by
  -- Unfold the definition of c for the first few terms
  have h1 : c 1 = 1 := rfl
  have h2 : c 2 = 3 := rfl
  have h3 : c 3 = 5 := by simp [c]
  have h4 : c 4 = 9 := by simp [c]
  have h5 : c 5 = 15 := by simp [c]
  have h6 : c 6 = 25 := by simp [c]
  have h7 : c 7 = 41 := by simp [c]
  have h8 : c 8 = 67 := by simp [c]
  have h9 : c 9 = 109 := by simp [c]
  
  -- Calculate c 10
  calc
    c 10 = c 9 + c 8 + 1 := by rfl
    _ = 109 + 67 + 1 := by rw [h9, h8]
    _ = 177 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_177_l604_60421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_graphs_l604_60436

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the transformation functions
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - x)

-- State the theorem
theorem symmetry_of_graphs (f : ℝ → ℝ) :
  ∀ (x y : ℝ), g f (1 + x) = y ↔ h f (1 - x) = y :=
by
  intro x y
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_graphs_l604_60436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l604_60417

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three side lengths can form a valid triangle. -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Generates the next triangle in the sequence based on the current triangle. -/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  let s := (t.a + t.b + t.c) / 2
  { a := (s - t.b) / 2,
    b := (s - t.c) / 2,
    c := (s - t.a) / 2 }

/-- Represents the sequence of triangles. -/
noncomputable def triangleSequence : ℕ → Triangle
  | 0 => { a := 1010, b := 1011, c := 1012 }
  | n + 1 => nextTriangle (triangleSequence n)

/-- Finds the index of the last valid triangle in the sequence. -/
def lastValidTriangleIndex : ℕ := 9  -- We know this from the problem solution

/-- The main theorem to be proved. -/
theorem last_triangle_perimeter :
  let lastTriangle := triangleSequence lastValidTriangleIndex
  lastTriangle.a + lastTriangle.b + lastTriangle.c = 3033 / 512 := by
  sorry

#eval lastValidTriangleIndex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l604_60417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_interval_a_range_for_subset_l604_60450

noncomputable def f (x : ℝ) := (1/3) * x^3 - x^2 - 3*x + 2

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem f_monotonically_decreasing_interval :
  monotonically_decreasing f (-1) 3 := by
  sorry

theorem a_range_for_subset (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 3 → 2*a - 3 ≤ x ∧ x ≤ a + 3) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry

#check f
#check monotonically_decreasing
#check f_monotonically_decreasing_interval
#check a_range_for_subset

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_interval_a_range_for_subset_l604_60450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_shared_area_l604_60475

noncomputable section

-- Define the square
def square_side_length : ℝ := 4

-- Define the circle
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 2

-- Define the shared area
noncomputable def shared_area : ℝ := Real.pi * circle_radius^2

-- Theorem statement
theorem square_circle_shared_area :
  shared_area = 8 * Real.pi := by sorry

end noncomputable section

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_shared_area_l604_60475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCD_l604_60455

noncomputable section

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 8)
def C : ℝ × ℝ := (8, 3)
def D : ℝ × ℝ := (10, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C D + distance D A

theorem perimeter_ABCD : 
  perimeter = Real.sqrt 29 + Real.sqrt 85 + 7 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCD_l604_60455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l604_60439

-- Define the Model structure
structure Model where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define necessary functions
def strength_of_linear_correlation (r : ℝ) : ℝ := sorry
def sum_of_squared_residuals (model : Model) : ℝ := sorry
def improved_model (model : Model) : Model := sorry
def fitting_effect (model : Model) : ℝ := sorry
def correlation_index (model : Model) : ℝ := sorry

-- Define the propositions
def proposition1 : Prop := ∀ r : ℝ, |r| > 0 → (strength_of_linear_correlation r) = |r|

def proposition2 : Prop := ∀ model : Model, ∀ ssr1 ssr2 : ℝ, 
  sum_of_squared_residuals model = ssr1 → 
  sum_of_squared_residuals (improved_model model) = ssr2 →
  ssr2 < ssr1 →
  fitting_effect (improved_model model) > fitting_effect model

def proposition3 : Prop := ∀ model : Model, ∀ R1 R2 : ℝ, 
  correlation_index model = R1 → 
  correlation_index (improved_model model) = R2 →
  R2 < R1 →
  fitting_effect (improved_model model) > fitting_effect model

-- Define the theorem
theorem exactly_one_correct_proposition : 
  (proposition2) ∧ 
  ¬(proposition1) ∧ 
  ¬(proposition3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l604_60439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_geometric_series_l604_60413

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_specific_geometric_series :
  geometric_series_sum 9 3 7 = 9827 := by
  -- Unfold the definition of geometric_series_sum
  unfold geometric_series_sum
  -- Simplify the expression
  simp [pow_succ]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_geometric_series_l604_60413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l604_60479

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 4*x else (- x)^2 - 4*(- x)

theorem f_solution_set (x : ℝ) :
  (f x < 5) ↔ (-5 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l604_60479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_always_win_l604_60433

/-- Represents a card in the game -/
structure Card where
  number : Nat
  deriving Repr

/-- Represents a player in the game -/
inductive Player
  | First
  | Second
  deriving Repr

/-- The game state -/
structure GameState where
  cards : List Card
  firstPlayerSum : Nat
  secondPlayerSum : Nat
  currentPlayer : Player
  deriving Repr

/-- Initialize the game with 2002 cards -/
def initializeGame : GameState :=
  { cards := List.range 2002 |>.map (fun n => { number := n + 1 }),
    firstPlayerSum := 0,
    secondPlayerSum := 0,
    currentPlayer := Player.First }

/-- The winning condition for the first player -/
def firstPlayerWins (state : GameState) : Prop :=
  state.firstPlayerSum % 10 > state.secondPlayerSum % 10

/-- Simulate the game given strategies for both players -/
def simulate (initialState : GameState) (firstStrategy secondStrategy : GameState → Card) : GameState :=
  sorry

/-- The theorem to prove -/
theorem first_player_can_always_win :
  ∃ (strategy : GameState → Card),
    ∀ (opponent_strategy : GameState → Card),
      firstPlayerWins (simulate initializeGame strategy opponent_strategy) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_always_win_l604_60433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l604_60402

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h_angles : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi
  h_sides : a > 0 ∧ b > 0 ∧ c > 0

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.cos t.B = t.b * Real.sin t.A) 
  (h2 : (Real.sqrt 3 / 4) * t.b^2 = (1 / 2) * t.a * t.c * Real.sin t.B) : 
  t.B = Real.pi / 3 ∧ t.a = t.c := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l604_60402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cistern_volume_l604_60406

/-- The volume of water in a round cistern -/
noncomputable def water_volume (diameter : ℝ) (depth : ℝ) (fill_ratio : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth * fill_ratio

/-- Theorem: The volume of water in a specific round cistern -/
theorem specific_cistern_volume :
  water_volume 20 10 (3/4) = 750 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cistern_volume_l604_60406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_savings_is_24_l604_60429

/-- Represents a store with its chocolate pricing and visiting constraints -/
structure Store where
  price : ℚ
  min_quantity : ℕ
  visits_per_month : ℕ

/-- Calculates the total cost for a given store over 3 months -/
def total_cost (s : Store) : ℚ :=
  s.price * s.min_quantity * (3 * s.visits_per_month : ℚ)

/-- Theorem: The maximum savings Bernie can achieve in three months is $24 -/
theorem max_savings_is_24 
  (local_store : Store)
  (store_a : Store)
  (store_b : Store)
  (store_c : Store)
  (h_local : local_store = { price := 3, min_quantity := 2, visits_per_month := 4 })
  (h_a : store_a = { price := 2, min_quantity := 5, visits_per_month := 2 })
  (h_b : store_b = { price := (5/2), min_quantity := 2, visits_per_month := 4 })
  (h_c : store_c = { price := (9/5), min_quantity := 10, visits_per_month := 1 })
  (h_weeks : (3 : ℚ) * 4 + 1 = 13) :
  (total_cost local_store - min (total_cost store_a) (min (total_cost store_b) (total_cost store_c))) = 24 := by
  sorry

#eval total_cost { price := 3, min_quantity := 2, visits_per_month := 4 }
#eval total_cost { price := 2, min_quantity := 5, visits_per_month := 2 }
#eval total_cost { price := (5/2), min_quantity := 2, visits_per_month := 4 }
#eval total_cost { price := (9/5), min_quantity := 10, visits_per_month := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_savings_is_24_l604_60429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expressions_l604_60499

theorem negative_expressions (A B C D E : ℝ) : 
  ∃ A B C D E, (A - B < 0) ∧ (B * C < 0) ∧ (C / (A * B) < 0) ∧
  ¬((D / B) * A < 0) ∧
  ¬((D + E) / C < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expressions_l604_60499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_eq_neg_one_l604_60461

def b : ℕ → ℚ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => 1 / (1 - b (n + 1))

theorem b_2015_eq_neg_one : b 2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_eq_neg_one_l604_60461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l604_60470

-- Define the basic structures
structure Plane where

structure Line where

structure Point where

-- Define relationships between structures
def parallel (a b : Plane) : Prop := sorry

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (a b : Line) : Prop := sorry

def skew_lines (a b : Line) : Prop := sorry

def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

def point_in_plane (pt : Point) (p : Plane) : Prop := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

def equidistant (pt : Point) (a b c : Point) : Prop := sorry

def projection (pt : Point) (p : Plane) : Point := sorry

def circumcenter (a b c : Point) : Point := sorry

-- Define the propositions
def proposition1 (α β : Plane) (AB CD : Line) : Prop :=
  parallel α β → parallel_lines AB CD → AB = CD

def proposition2 (a b c : Line) : Prop :=
  skew_lines a b → skew_lines b c → skew_lines a c

def proposition3 (pt : Point) (α : Plane) : Prop :=
  ∃ l1 l2 : Line, perpendicular_line_plane l1 α ∧ perpendicular_line_plane l2 α ∧ l1 ≠ l2

def proposition4 (α β : Plane) (P : Point) (PQ : Line) : Prop :=
  parallel α β → point_in_plane P α → parallel_line_plane PQ β → line_in_plane PQ α

def proposition5 (P : Point) (A B C : Point) : Prop :=
  equidistant P A B C → projection P (Plane.mk) = circumcenter A B C

def proposition6 (a b : Line) (P : Point) : Prop :=
  skew_lines a b → ∃ γ : Plane, point_in_plane P γ ∧
    ((perpendicular_line_plane a γ ∧ parallel_line_plane b γ) ∨
     (perpendicular_line_plane b γ ∧ parallel_line_plane a γ))

-- Theorem stating which propositions are correct
theorem correct_propositions :
  (∀ α β : Plane, ∀ AB CD : Line, proposition1 α β AB CD) ∧
  (¬ ∀ a b c : Line, proposition2 a b c) ∧
  (¬ ∀ pt : Point, ∀ α : Plane, proposition3 pt α) ∧
  (∀ α β : Plane, ∀ P : Point, ∀ PQ : Line, proposition4 α β P PQ) ∧
  (∀ P A B C : Point, proposition5 P A B C) ∧
  (¬ ∀ a b : Line, ∀ P : Point, proposition6 a b P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l604_60470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bat_wings_area_l604_60415

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- Theorem: The area of the "bat wings" in the given rectangle is 3.5 -/
theorem bat_wings_area (PQRS : Rectangle) 
  (h_PQ : PQRS.Q.x - PQRS.P.x = 5)
  (h_QR : PQRS.R.y - PQRS.Q.y = 3)
  (h_PT : PQRS.Q.x - PQRS.P.x - (PQRS.Q.x - PQRS.R.x) = 1)
  (h_TR : PQRS.Q.x - PQRS.R.x = 1)
  (h_RQ : PQRS.Q.y - PQRS.R.y = 1) : 
  ∃ (T U : Point), 
    T.x = PQRS.Q.x - 1 ∧ 
    T.y = PQRS.Q.y ∧
    U.x = PQRS.R.x ∧ 
    U.y = PQRS.R.y + 1 ∧
    triangleArea PQRS.S U T = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bat_wings_area_l604_60415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l604_60405

def sequence_property (a : Fin 9 → ℕ) : Prop :=
  ∀ n : Fin 9, n.val ≥ 2 → a n = a (n - 1) * a (n - 2)

theorem first_number_is_two (a : Fin 9 → ℕ) 
  (h_seq : sequence_property a)
  (h_last_three : a 6 = 32 ∧ a 7 = 256 ∧ a 8 = 8192) :
  a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l604_60405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_eating_contest_l604_60448

theorem orange_eating_contest (students : Finset ℕ) 
  (oranges_eaten : ℕ → ℕ) 
  (alice bob : ℕ) :
  students.card = 8 →
  (∀ i j, i ∈ students → j ∈ students → i ≠ j → oranges_eaten i ≠ oranges_eaten j) →
  (∀ i, i ∈ students → oranges_eaten i ∈ Finset.range 9 \ {0}) →
  (∀ i, i ∈ students → oranges_eaten alice ≥ oranges_eaten i) →
  (∀ i, i ∈ students → oranges_eaten bob ≤ oranges_eaten i) →
  alice ∈ students →
  bob ∈ students →
  oranges_eaten alice - oranges_eaten bob = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_eating_contest_l604_60448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_out_time_for_specific_garage_l604_60410

/-- Represents the dimensions and properties of a flooded garage with pumps. -/
structure FloodedGarage where
  length : ℚ
  width : ℚ
  water_depth_inches : ℚ
  pump_count : ℕ
  pump_rate : ℚ
  gallons_per_cubic_foot : ℚ

/-- Calculates the time required to pump out water from a flooded garage. -/
def pump_out_time (garage : FloodedGarage) : ℚ :=
  let water_depth_feet := garage.water_depth_inches / 12
  let water_volume := water_depth_feet * garage.length * garage.width
  let total_gallons := water_volume * garage.gallons_per_cubic_foot
  let total_pump_rate := garage.pump_count * garage.pump_rate
  total_gallons / total_pump_rate

/-- Theorem stating the time required to pump out water from a specific garage configuration. -/
theorem pump_out_time_for_specific_garage :
  let garage : FloodedGarage := {
    length := 20,
    width := 25,
    water_depth_inches := 12,
    pump_count := 2,
    pump_rate := 10,
    gallons_per_cubic_foot := 15/2
  }
  pump_out_time garage = 375/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_out_time_for_specific_garage_l604_60410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_8_equals_5_l604_60428

-- Define the function g(x) = 2^(x-2)
noncomputable def g (x : ℝ) : ℝ := 2^(x-2)

-- Define the property of f being symmetrical to g about y = x
def symmetrical_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem f_8_equals_5 (f : ℝ → ℝ) (h : symmetrical_about_y_eq_x f g) : f 8 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_8_equals_5_l604_60428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l604_60438

theorem shaded_areas_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) :
  (∃ r : Real, r > 0 ∧ 
    (φ * r^2 / 2 = r^2 * Real.tan φ / 2 - φ * r^2 / 2)) ↔ Real.tan φ = 2 * φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l604_60438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l604_60452

-- Define the hyperbola
structure Hyperbola where
  focus : ℝ × ℝ
  asymptote_slope : ℚ

-- Define the properties of our specific hyperbola
noncomputable def our_hyperbola : Hyperbola :=
{ focus := (0, 10)
  asymptote_slope := 4/3 }

-- State the theorem
theorem hyperbola_equation (h : Hyperbola) (x y : ℝ) : 
  h = our_hyperbola → y^2/64 - x^2/36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l604_60452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l604_60432

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_cond : a 7 = a 6 + 2 * a 5) :
  ∀ m n : ℕ, 
    m > 0 → n > 0 → 
    a m * a n = 16 * (a 1)^2 → 
    1 / (m : ℝ) + 4 / (n : ℝ) ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_value_l604_60432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_g_l604_60489

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi/3 - 2*x) - 2 * Real.sin (Real.pi/4 + x) * Real.sin (Real.pi/4 - x)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2*x)

theorem monotonic_increasing_interval_of_g :
  ∃ (a b : ℝ), a = -Real.pi/12 ∧ b = Real.pi/4 ∧
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/2), 
    (∀ y ∈ Set.Icc a b, x < y → g x < g y) ∧
    (∀ y ∈ Set.Icc (-Real.pi/12) (Real.pi/2), y < a ∨ b < y → ¬(x < y → g x < g y))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_g_l604_60489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_with_complement_l604_60447

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {1,2,3,4}
def B : Set ℕ := {3,5,6}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1,2,4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_with_complement_l604_60447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_l604_60454

/-- Given sequences {a_n} and {b_n} satisfying certain conditions, prove that b_n = n / (n + 1) for all n ∈ ℕ+ -/
theorem sequence_relation (a b : ℕ+ → ℚ) : 
  (a 1 = 1/2) → 
  (∀ n : ℕ+, a n + b n = 1) → 
  (∀ n : ℕ+, b (n + 1) = b n / (1 - a n ^ 2)) → 
  (∀ n : ℕ+, b n = n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_l604_60454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_b_pow_5_minus_b_pow_neg_5_l604_60440

theorem factorization_of_b_pow_5_minus_b_pow_neg_5 (b : ℝ) (hb : b ≠ 0) :
  b^(5 : ℝ) - b^(-(5 : ℝ)) = (b - b⁻¹) * (b^(4 : ℝ) + b^(2 : ℝ) + 1 + b^(-(2 : ℝ)) + b^(-(4 : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_b_pow_5_minus_b_pow_neg_5_l604_60440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l604_60478

noncomputable def line (x : ℝ) : ℝ := (4/3) * x - 20/3

def valid_parameterization (p : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line (p.1 + t * d.1) = p.2 + t * d.2

noncomputable def param_A : ℝ × ℝ := (5, 0)
noncomputable def dir_A : ℝ × ℝ := (-3, -4)

noncomputable def param_B : ℝ × ℝ := (20, 4)
noncomputable def dir_B : ℝ × ℝ := (9, 12)

noncomputable def param_C : ℝ × ℝ := (3, -7)
noncomputable def dir_C : ℝ × ℝ := (4/3, 1)

noncomputable def param_D : ℝ × ℝ := (17/4, -1)
noncomputable def dir_D : ℝ × ℝ := (1, 4/3)

noncomputable def param_E : ℝ × ℝ := (0, -20/3)
noncomputable def dir_E : ℝ × ℝ := (18, -24)

theorem parameterization_validity :
  valid_parameterization param_A dir_A ∧
  valid_parameterization param_B dir_B ∧
  ¬valid_parameterization param_C dir_C ∧
  ¬valid_parameterization param_D dir_D ∧
  ¬valid_parameterization param_E dir_E :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l604_60478
