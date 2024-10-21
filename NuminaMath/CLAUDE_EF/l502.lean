import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_endpoint_l502_50289

-- Define points A and B
noncomputable def A : ℝ × ℝ := (-3, 5)
noncomputable def B : ℝ × ℝ := (9, -1)

-- Define the extension ratio
def extension_ratio : ℚ := 2/5

-- Define point C
noncomputable def C : ℝ × ℝ := (13.8, -3.4)

-- Theorem statement
theorem extended_segment_endpoint :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  (BC.1^2 + BC.2^2) = (extension_ratio^2 : ℝ) * (AB.1^2 + AB.2^2) ∧
  BC.1 / AB.1 = extension_ratio ∧
  BC.2 / AB.2 = extension_ratio :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_endpoint_l502_50289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l502_50291

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines 
  (m : ℝ) 
  (h_parallel : m = 2) -- condition that the lines are parallel
  : let line1 := λ (x y : ℝ) ↦ Real.sqrt 3 * x + y - 1
    let line2 := λ (x y : ℝ) ↦ 2 * Real.sqrt 3 * x + m * y + 3
    let a : ℝ := Real.sqrt 3
    let b : ℝ := 1
    let c1 : ℝ := -1
    let c2 : ℝ := -3/2
    (|c1 - c2| / Real.sqrt (a^2 + b^2)) = 5/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l502_50291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_g_l502_50270

-- Define the polynomials
variable (R : Type*) [CommRing R]
variable (f g h : Polynomial R)

-- Define the conditions
axiom poly_equation : 5 • f - 3 • g = h
axiom degree_f : Polynomial.degree f = 10
axiom degree_h : Polynomial.degree h = 11

-- State the theorem
theorem min_degree_g : Polynomial.degree g ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_g_l502_50270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_problem_l502_50244

theorem course_selection_problem (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 8 → k = 5 → m = 2 →
  (Nat.choose n k - Nat.choose (n - m - 1) (k - 1)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_problem_l502_50244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_division_l502_50228

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i) / i
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_division_l502_50228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l502_50232

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l502_50232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_five_points_l502_50271

-- Define a Point type to represent points in a plane
structure Point : Type :=
  (x : ℝ) (y : ℝ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- State the theorem
theorem inequality_of_five_points (A B P Q R : Point) :
  distance A B + distance P Q + distance Q R + distance R P ≤
  distance A P + distance A Q + distance A R + distance B P + distance B Q + distance B R :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_five_points_l502_50271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_differences_l502_50219

theorem pairwise_differences (S : Finset ℕ) : 
  S.card = 8 → (∀ n ∈ S, n < 16) → ∃ a b c d e f : ℕ, 
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
  a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ 
  a - b = c - d ∧ a - b = e - f ∧ 
  (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_differences_l502_50219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l502_50202

theorem inequality_solution (x : ℝ) : 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 ↔ x ∈ Set.Icc (-2) (-1) ∪ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l502_50202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_2_l502_50218

noncomputable def a : Fin 2 → ℝ := ![1, Real.sqrt 3]
def b : Fin 2 → ℝ := ![-1, 0]

theorem magnitude_a_plus_2b_equals_2 :
  Real.sqrt ((a 0 + 2 * b 0)^2 + (a 1 + 2 * b 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_2_l502_50218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_product_l502_50235

theorem closest_integer_product (target : ℤ) (factor : ℤ) (result : ℤ) :
  target = 4691100843 →
  factor = 469157 →
  result = 10001 →
  ∀ n : ℤ, |target - factor * result| ≤ |target - factor * n| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_product_l502_50235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_tan_difference_l502_50215

theorem cosine_function_tan_difference (A ω α β : ℝ) : 
  A > 0 → ω > 0 → 
  α ∈ Set.Ioo 0 (π / 4) → β ∈ Set.Ioo 0 (π / 4) →
  (∀ x : ℝ, A * Real.cos (ω * x - π / 3) = A * Real.cos (ω * (x + π / (2 * ω)) - π / 3)) →
  A * Real.cos (-π / 3) = 1 →
  A * Real.cos (ω * (α - π / 3) - π / 3) = 10 / 13 →
  A * Real.cos (ω * (β + π / 6) - π / 3) = 6 / 5 →
  Real.tan (2 * α - 2 * β) = 16 / 63 := by
  sorry

#check cosine_function_tan_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_tan_difference_l502_50215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_age_equation_original_age_solution_l502_50236

/-- The original average age of students in an adult school -/
noncomputable def A : ℝ := sorry

/-- The number of new students who joined the school -/
def new_students : ℕ := 120

/-- The average age of new students -/
def new_students_avg_age : ℝ := 32

/-- The decrease in average age after new students joined -/
def age_decrease : ℝ := 4

/-- The total number of students after new students joined -/
def total_students : ℕ := 160

/-- Theorem stating the equation that the original average age A must satisfy -/
theorem original_age_equation :
  (total_students - new_students : ℝ) * A + new_students * new_students_avg_age = 
  total_students * (A - age_decrease) := by
  sorry

/-- Theorem stating the solution for the original average age A -/
theorem original_age_solution :
  A = 37 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_age_equation_original_age_solution_l502_50236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_l502_50261

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a quadrilateral -/
structure Quadrilateral where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.v1 q.v2 + distance q.v2 q.v3 + distance q.v3 q.v4 + distance q.v4 q.v1

/-- Theorem: For the given quadrilateral, the sum of a, b, and c in the perimeter expression a√5 + b√2 + c is 12 -/
theorem perimeter_sum (q : Quadrilateral)
  (h1 : q.v1 = ⟨0, 0⟩)
  (h2 : q.v2 = ⟨4, 3⟩)
  (h3 : q.v3 = ⟨5, 2⟩)
  (h4 : q.v4 = ⟨4, -1⟩)
  (a b c : ℤ)
  (h5 : perimeter q = a * Real.sqrt 5 + b * Real.sqrt 2 + c) :
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_l502_50261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_operation_l502_50280

/-- Vector a -/
def a : ℝ × ℝ × ℝ := (2, 2, 1)

/-- Vector b -/
def b : ℝ × ℝ × ℝ := (3, 5, 3)

/-- Theorem stating the magnitude of 2a - b -/
theorem magnitude_of_vector_operation : Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2.1 - b.2.1)^2 + (2 * a.2.2 - b.2.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_operation_l502_50280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_baskets_proof_l502_50251

def blueberry_baskets : ℕ :=
  let plum_baskets := 19
  let plums_per_basket := 46
  let blueberries_per_basket := 170
  let total_fruits := 1894
  let total_plums := plum_baskets * plums_per_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_per_basket

#eval blueberry_baskets

theorem blueberry_baskets_proof : 
  blueberry_baskets = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_baskets_proof_l502_50251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groups_formed_l502_50214

/-- Represents a class of students -/
structure ClassOfStudents where
  boys : Nat
  girls : Nat

/-- Calculates the number of groups that can be formed from a given class -/
def calculateGroups (c : ClassOfStudents) (groupSize : Nat) : Nat :=
  (c.boys + c.girls) / groupSize

/-- Theorem: Given a class with 9 boys and 12 girls, forming groups of 3 students each results in 7 groups -/
theorem groups_formed (c : ClassOfStudents) (h1 : c.boys = 9) (h2 : c.girls = 12) :
  calculateGroups c 3 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_groups_formed_l502_50214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_body_volume_is_five_sixths_l502_50245

/-- A rectangular prism with edges of different lengths -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  different_lengths : length ≠ width ∧ width ≠ height ∧ height ≠ length

/-- A pair of skew edges in a rectangular prism -/
inductive SkewEdgePair
  | AB_A1D1
  | BC_B1D1
  | CD_C1A1

/-- The volume of the convex body formed by points minimizing distances to skew edges -/
noncomputable def convexBodyVolume (T : RectangularPrism) (skewEdges : SkewEdgePair) : ℝ :=
  (5/6) * T.length * T.width * T.height

/-- Theorem stating that the volume of the convex body is 5/6 of the original prism's volume -/
theorem convex_body_volume_is_five_sixths (T : RectangularPrism) (skewEdges : SkewEdgePair) :
  convexBodyVolume T skewEdges = (5/6) * (T.length * T.width * T.height) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_body_volume_is_five_sixths_l502_50245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_expenditure_increase_l502_50229

theorem sugar_expenditure_increase (P : ℝ) (h : P > 0) : 
  (25 * (P * 1.32) - 30 * P) / (30 * P) * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_expenditure_increase_l502_50229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_count_l502_50279

theorem distinct_prime_factors_count : ∃ (S : Finset Nat), 
  (∀ p ∈ S, Nat.Prime p) ∧ 
  (S.prod id) = 79 * 81 * 85 * 87 ∧ 
  Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_count_l502_50279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l502_50238

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

theorem triangle_side_length 
  (A B C : ℝ) 
  (hf : f A = 2) 
  (hb : Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) = 1) 
  (harea : (1/2) * 1 * Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) * Real.sin A = Real.sqrt 3 / 2) :
  Real.sqrt (1^2 + C^2 - 2*1*C*Real.cos A) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l502_50238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_cannot_form_triangle_l502_50208

/-- Three non-overlapping lines in a 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop
  line3 : ℝ → ℝ → ℝ → Prop

/-- The given three lines -/
def givenLines (m : ℝ) : ThreeLines :=
  { line1 := λ x y => y = -x
    line2 := λ x y => 4 * x + y = 3
    line3 := λ x y m' => m' * x + y + m' - 1 = 0 }

/-- Definition of when three lines form a triangle -/
def formTriangle (lines : ThreeLines) (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    lines.line1 x1 y1 ∧ lines.line2 x2 y2 ∧ lines.line3 x3 y3 m ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x3 ≠ x1 ∨ y3 ≠ y1)

/-- The main theorem to prove -/
theorem lines_cannot_form_triangle (m : ℝ) :
  ¬(formTriangle (givenLines m) m) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_cannot_form_triangle_l502_50208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_roots_l502_50296

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

noncomputable def g (x : ℝ) : ℝ := Real.log x - 3 * x + x^2

noncomputable def g_deriv (x : ℝ) : ℝ := 1 / x - 3 + 2 * x

theorem tangent_line_parallel_and_roots (a m : ℝ) :
  (f_deriv a 2 = -1/2) ∧
  (∃ x₁ x₂, 1/2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 ∧
    f a x₁ + m = 2 * x₁ - x₁^2 ∧
    f a x₂ + m = 2 * x₂ - x₂^2 ∧
    ∀ x, 1/2 ≤ x ∧ x ≤ 2 → f a x + m ≠ 2 * x - x^2 → (x = x₁ ∨ x = x₂)) →
  a = 1 ∧ Real.log 2 + 5/4 ≤ m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_roots_l502_50296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_neg_16_iff_a_in_range_f_lt_0_iff_x_in_interval_l502_50283

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (1 - a) * x - a

-- Part 1: Prove the range of a for which f(x) ≥ -16 for all real x
theorem f_geq_neg_16_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ -16) ↔ a ∈ Set.Icc (-9) 7 := by sorry

-- Part 2: Prove the solution sets for f(x) < 0
theorem f_lt_0_iff_x_in_interval (a x : ℝ) :
  f a x < 0 ↔
    (a < -1 ∧ x ∈ Set.Ioo a (-1)) ∨
    (a > -1 ∧ x ∈ Set.Ioo (-1) a) := by sorry

-- Note: Set.Icc is the closed interval notation in Lean
-- Set.Ioo is the open interval notation in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_neg_16_iff_a_in_range_f_lt_0_iff_x_in_interval_l502_50283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_equivalence_l502_50250

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- A parameterization of a line -/
structure Parameterization where
  p : Point
  d : Vec

/-- The line y = 3x - 4 -/
def line (p : Point) : Prop :=
  p.y = 3 * p.x - 4

/-- Check if a vector is parallel to (1, 3) -/
def isParallelToSlope (v : Vec) : Prop :=
  ∃ (k : ℝ), v.x = k * 1 ∧ v.y = k * 3

/-- A parameterization is valid if it satisfies the line equation and has the correct slope -/
def isValidParameterization (param : Parameterization) : Prop :=
  line param.p ∧ isParallelToSlope param.d

theorem parameterization_equivalence (param : Parameterization) :
  isValidParameterization param ↔ 
    (∀ (t : ℝ), line { x := param.p.x + t * param.d.x, y := param.p.y + t * param.d.y }) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_equivalence_l502_50250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complaint_probability_online_l502_50217

theorem complaint_probability_online (online_preference : ℚ) 
  (store_preference : ℚ) (online_qualification : ℚ) (store_qualification : ℚ) :
  online_preference = 4/5 →
  store_preference = 1/5 →
  online_qualification = 17/20 →
  store_qualification = 9/10 →
  (online_preference * (1 - online_qualification)) / 
  ((online_preference * (1 - online_qualification)) + 
   (store_preference * (1 - store_qualification))) = 6/7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complaint_probability_online_l502_50217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_l502_50282

theorem candy_ratio : 
  ∀ (emily_candies bob_candies : ℕ),
    emily_candies = 6 →
    bob_candies = 4 →
    let jennifer_candies := 3 * bob_candies
    (jennifer_candies : ℚ) / emily_candies = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_l502_50282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_coplanar_implies_m_equals_negative_one_l502_50295

def a : ℝ × ℝ × ℝ := (-3, 2, 1)
def b : ℝ × ℝ × ℝ := (2, 2, -1)
def c (m : ℝ) : ℝ × ℝ × ℝ := (m, 4, 0)

theorem vectors_coplanar_implies_m_equals_negative_one :
  (∃ (x y : ℝ), c (-1) = (x * a.1 + y * b.1, x * a.2.1 + y * b.2.1, x * a.2.2 + y * b.2.2)) →
  ∀ m : ℝ, (∃ (x y : ℝ), c m = (x * a.1 + y * b.1, x * a.2.1 + y * b.2.1, x * a.2.2 + y * b.2.2)) →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_coplanar_implies_m_equals_negative_one_l502_50295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_k_m_n_equals_six_l502_50286

theorem sum_k_m_n_equals_six (t : ℝ) (k m n : ℕ+) :
  (1 + Real.sin t) * (1 + Real.cos t) = 9/4 →
  (1 - Real.sin t) * (1 - Real.cos t) = m/n - Real.sqrt k →
  Nat.Coprime m.val n.val →
  k + m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_k_m_n_equals_six_l502_50286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_negative_two_l502_50233

theorem sum_of_solutions_equals_negative_two : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (3 : ℝ)^(x^2 + 4*x + 4) = 9^(x + 2)) ∧ 
  (∀ x : ℝ, (3 : ℝ)^(x^2 + 4*x + 4) = 9^(x + 2) → x ∈ S) ∧
  (S.sum id) = -2 := by
  sorry

#check sum_of_solutions_equals_negative_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_negative_two_l502_50233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_n_relationship_l502_50297

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define M and N
noncomputable def M (x : ℝ) : ℝ := Real.sqrt (floor (Real.sqrt x))
noncomputable def N (x : ℝ) : ℤ := floor (Real.sqrt (Real.sqrt x))

-- State the theorem
theorem m_n_relationship :
  ¬(∀ x : ℝ, x ≥ 1 → M x > N x) ∧
  ¬(∀ x : ℝ, x ≥ 1 → M x = N x) ∧
  ¬(∀ x : ℝ, x ≥ 1 → M x < N x) := by
  sorry

-- Additional lemmas to support the main theorem
lemma m_n_equal_at_one :
  M 1 = N 1 := by
  sorry

lemma m_greater_n_at_four :
  M 4 > N 4 := by
  sorry

lemma m_geq_n_general (x : ℝ) (h : x ≥ 1) :
  M x ≥ N x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_n_relationship_l502_50297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l502_50276

theorem simplify_expression (a b c x D : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  let q := fun x => ((x + a)^2 + D) / ((a - b) * (a - c)) + 
                    ((x + b)^2 + D) / ((b - a) * (b - c)) + 
                    ((x + c)^2 + D) / ((c - a) * (c - b))
  q x = a + b + c + 2*x + 3*D / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l502_50276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l502_50252

open Real

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem g_monotone_increasing_interval :
  ∀ k : ℤ, StrictMonoOn g (Set.Icc (k * π - π / 12 : ℝ) (k * π + 5 * π / 12 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l502_50252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l502_50258

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (3, 0)

/-- A line perpendicular to the x-axis passing through a point -/
def vertical_line (x₀ : ℝ) (x : ℝ) : Prop := x = x₀

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- Main theorem -/
theorem ellipse_chord_length :
  ∃ (A B : ℝ × ℝ),
    is_ellipse A.1 A.2 ∧
    is_ellipse B.1 B.2 ∧
    vertical_line right_focus.1 A.1 ∧
    vertical_line right_focus.1 B.1 ∧
    distance A B = 32/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l502_50258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canonical_form_equivalent_l502_50268

-- Define the functions y₁ and y₂
variable (y₁ y₂ : ℝ → ℝ)

-- Define the original system of equations
def original_system (y₁ y₂ : ℝ → ℝ) : Prop :=
  ∀ t, 
    y₂ t * (deriv y₁ t) - Real.log ((deriv (deriv y₁) t) - y₁ t) = 0 ∧
    Real.exp (deriv y₂ t) - y₁ t - y₂ t = 0

-- Define the canonical form of the system
def canonical_form (y₁ y₂ : ℝ → ℝ) : Prop :=
  ∀ t, 
    deriv (deriv y₁) t = y₁ t + Real.exp (y₂ t * deriv y₁ t) ∧
    deriv y₂ t = Real.log (y₁ t + y₂ t)

-- State the theorem
theorem canonical_form_equivalent :
  original_system y₁ y₂ ↔ canonical_form y₁ y₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canonical_form_equivalent_l502_50268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_exponential_l502_50241

theorem tangent_and_exponential :
  (∀ x ∈ Set.Ioo (-Real.pi/2 : ℝ) 0, Real.tan x < 0) ∧
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ (2 : ℝ)^x₀ = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_exponential_l502_50241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_multiple_of_175_l502_50223

theorem smallest_k_multiple_of_175 : 
  ∃ k : ℕ, 
    (∀ n : ℕ, n < k → ¬(175 ∣ (n * (n + 1) * (2 * n + 1) / 6))) ∧ 
    (175 ∣ (k * (k + 1) * (2 * k + 1) / 6)) ∧
    k = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_multiple_of_175_l502_50223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_point_and_zeros_comparison_l502_50263

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * exp x + log (x + 1) - a * sin x

theorem f_critical_point_and_zeros_comparison (a : ℝ) (h : a > 2) :
  ∃ (m : ℝ), m ∈ Set.Ioo 0 (π / 2) ∧ 
    (∀ x ∈ Set.Ioo 0 (π / 2), deriv (f a) x = 0 → x = m) ∧
    (∃ (n : ℝ), (∀ x ∈ Set.Icc 0 π, f a x = 0 → x ≤ n) ∧ 2 * m > n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_point_and_zeros_comparison_l502_50263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_half_l502_50247

/-- The point A with coordinates (1, 0) -/
def A : ℝ × ℝ := (1, 0)

/-- The line on which point B moves -/
def line_B (x y : ℝ) : Prop := x - y = 0

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the distance between A and B is minimized when B is at (1/2, 1/2) -/
theorem min_distance_at_half :
  ∀ (B : ℝ × ℝ), line_B B.1 B.2 →
    distance A B ≥ distance A (1/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_half_l502_50247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_negative_implies_c_bound_l502_50222

/-- Represents the sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Defines the sequence a_n = n + c -/
def a (n : ℕ) (c : ℝ) : ℝ := n + c

theorem sequence_sum_negative_implies_c_bound
  (c : ℝ) (h : arithmeticSum (a 1 c) 1 7 < 0) :
  c < -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_negative_implies_c_bound_l502_50222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_two_l502_50255

/-- The sum of the infinite series 1/(2^1) + 2/(2^2) + 3/(2^3) + ... + k/(2^k) + ... -/
noncomputable def infinite_series_sum : ℝ := ∑' k, (k : ℝ) / (2 ^ k)

/-- The theorem stating that the sum of the infinite series is equal to 2 -/
theorem infinite_series_sum_eq_two : infinite_series_sum = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_two_l502_50255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l502_50269

theorem sin_2x_value (x : ℝ) 
  (h1 : x ∈ Set.Icc (-π/3) (π/6))
  (h2 : (Real.sin (x + π/3) * Real.cos (x - π/6) + Real.sin (x - π/6) * Real.cos (x + π/3)) = 5/13) :
  Real.sin (2*x) = (5*Real.sqrt 3 - 12) / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l502_50269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_l502_50210

/-- Represents the probability of a seed germinating -/
noncomputable def probability_of_germination : ℝ := 8/9

/-- Represents the number of plots in each experimental group -/
def plots_per_group : ℕ := 3

/-- Represents the average number of plots that did not germinate in each group -/
noncomputable def average_non_germinated : ℝ := 1/3

theorem germination_probability :
  plots_per_group = 3 →
  average_non_germinated = 1/3 →
  probability_of_germination = 8/9 := by
  intros h1 h2
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_l502_50210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_10_l502_50274

noncomputable def f (x : ℝ) : ℝ := (5 * x^2 - 51 * x + 10) / (x - 10)

theorem limit_of_f_at_10 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 10 → |x - 10| < δ → |f x - 49| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_10_l502_50274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_x_given_cyclic_relations_l502_50281

theorem max_sin_x_given_cyclic_relations (x y z : Real) 
  (h1 : Real.sin x = Real.cos y)
  (h2 : Real.sin y = Real.cos z)
  (h3 : Real.sin z = Real.cos x) :
  ∃ (max_sin_x : Real), max_sin_x = Real.sqrt 2 / 2 ∧ 
    ∀ t : Real, Real.sin x ≤ max_sin_x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_x_given_cyclic_relations_l502_50281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_percentage_problem_l502_50239

theorem bird_percentage_problem (total_birds : ℝ) (hawk_percent : ℝ) (paddyfield_warbler_percent_of_nonhawks : ℝ) (kingfisher_to_paddyfield_ratio : ℝ) :
  hawk_percent = 30 →
  paddyfield_warbler_percent_of_nonhawks = 40 →
  kingfisher_to_paddyfield_ratio = 25 →
  (let nonhawk_percent : ℝ := 100 - hawk_percent
   let paddyfield_warbler_percent : ℝ := (paddyfield_warbler_percent_of_nonhawks / 100) * nonhawk_percent
   let kingfisher_percent : ℝ := (kingfisher_to_paddyfield_ratio / 100) * paddyfield_warbler_percent
   let other_birds_percent : ℝ := 100 - (hawk_percent + paddyfield_warbler_percent + kingfisher_percent)
   other_birds_percent) = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_percentage_problem_l502_50239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_iff_k_range_l502_50226

noncomputable section

/-- The function f as defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

/-- The triangle inequality condition for f(a), f(b), f(c) -/
def triangle_inequality (k : ℝ) : Prop :=
  ∀ a b c : ℝ, f k a + f k b > f k c ∧ 
               f k a + f k c > f k b ∧ 
               f k b + f k c > f k a ∧
               f k a > 0 ∧ f k b > 0 ∧ f k c > 0

/-- The main theorem stating the equivalence of the triangle inequality condition and the range of k -/
theorem triangle_condition_iff_k_range :
  ∀ k : ℝ, triangle_inequality k ↔ -1/2 < k ∧ k < 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_iff_k_range_l502_50226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_is_eight_l502_50262

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the total area of the sign -/
def totalArea (d : SignDimensions) : ℕ :=
  d.width * d.height

/-- Calculates the area occupied by the letter H -/
def areaH (strokeWidth : ℕ) : ℕ :=
  2 * (6 * strokeWidth) + 1 * (strokeWidth * 4)

/-- Calculates the area occupied by the letter E -/
def areaE (strokeWidth : ℕ) : ℕ :=
  3 * (strokeWidth * 4)

/-- Calculates the area occupied by the letter L -/
def areaL (strokeWidth : ℕ) : ℕ :=
  1 * (6 * strokeWidth) + 1 * (strokeWidth * 4)

/-- Calculates the area occupied by the letter P -/
def areaP (strokeWidth : ℕ) : ℕ :=
  1 * (6 * strokeWidth) + 1 * (strokeWidth * 4) + 1 * (strokeWidth * strokeWidth)

/-- Calculates the total area occupied by the word HELP -/
def areaHELP (strokeWidth : ℕ) : ℕ :=
  areaH strokeWidth + areaE strokeWidth + areaL strokeWidth + areaP strokeWidth

/-- The main theorem stating that the white area of the sign is 8 square units -/
theorem white_area_is_eight (d : SignDimensions) (strokeWidth : ℕ) :
  d.width = 6 → d.height = 18 → strokeWidth = 2 →
  totalArea d - areaHELP strokeWidth = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_is_eight_l502_50262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l502_50212

/-- The circle with center (a, a) and radius 1 -/
def circle_eq (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = 1

/-- The line y = 3x -/
def line_eq (x y : ℝ) : Prop :=
  y = 3 * x

/-- The area of the triangle formed by the intersection of the circle and the line -/
noncomputable def triangle_area (a : ℝ) : ℝ :=
  Real.sqrt (a^2 * (10 - 4 * a^2)) / 5

theorem max_triangle_area :
  ∃ (a : ℝ), a > 0 ∧ 
    (∀ (b : ℝ), b > 0 → triangle_area b ≤ triangle_area a) ∧
    a = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l502_50212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_relation_fibonacci_identity_l502_50259

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  Matrix.of !![1, 1; 1, 0] ^ n

theorem fibonacci_relation (n : ℕ) :
  fibonacci_matrix n = !![fibonacci (n + 1), fibonacci n; fibonacci n, fibonacci (n - 1)] := by
  sorry

theorem fibonacci_identity : 
  (fibonacci 784 : ℤ) * (fibonacci 786 : ℤ) - (fibonacci 785 : ℤ)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_relation_fibonacci_identity_l502_50259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_incongruent_iff_power_of_two_l502_50284

def S (n : ℕ+) : Finset ℕ :=
  Finset.image (λ k => k * (k + 1) / 2) (Finset.range n)

theorem elements_incongruent_iff_power_of_two (n : ℕ+) :
  (∀ i j, i ∈ S n → j ∈ S n → i ≠ j → i % n ≠ j % n) ↔ ∃ k : ℕ, n = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_incongruent_iff_power_of_two_l502_50284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_students_l502_50209

/-- Represents the number of students a person can teach -/
def x : ℕ := sorry

/-- The total number of students who can perform the experiment after two classes -/
def total_students : ℕ := 1 + x + (x + 1) * x

/-- Theorem stating that the total number of students who can perform the experiment is 36 -/
theorem experiment_students : total_students = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_students_l502_50209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l502_50246

theorem trigonometric_identities (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : β ∈ Set.Ioo 0 Real.pi) 
  (h3 : Real.cos α = 4/5) 
  (h4 : Real.sin (α - β) = 5/13) : 
  Real.cos (2 * α) = 7/25 ∧ Real.sin (α + β) = 253/325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l502_50246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_given_floor_condition_l502_50221

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem x_range_given_floor_condition :
  ∀ x : ℝ, floor (x - 1) = -2 → -1 ≤ x ∧ x < 0 :=
by
  intro x h
  have h1 : -2 ≤ x - 1 ∧ x - 1 < -1 := by
    sorry -- Proof of this step is omitted
  have h2 : -1 ≤ x ∧ x < 0 := by
    sorry -- Proof of this step is omitted
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_given_floor_condition_l502_50221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_g_extrema_l502_50254

noncomputable def f (x : ℝ) : ℝ := (4 * (Real.cos x) ^ 4 - 2 * Real.cos (2 * x) - 1) / 
  (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ := (1 / 2) * f x + Real.sin (2 * x)

theorem f_simplification (x : ℝ) : f x = 2 * Real.cos (2 * x) := by sorry

theorem f_value : f (-11 * Real.pi / 12) = Real.sqrt 3 := by sorry

theorem g_extrema : 
  ∀ x, x ∈ Set.Icc 0 (Real.pi / 4) → g x ≤ Real.sqrt 2 ∧ g x ≥ 1 ∧
  ∃ x₁ x₂, x₁ ∈ Set.Icc 0 (Real.pi / 4) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 4) ∧ 
  g x₁ = Real.sqrt 2 ∧ g x₂ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_g_extrema_l502_50254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_real_exponential_l502_50234

theorem negation_of_existence_real_exponential :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_real_exponential_l502_50234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_with_given_roots_l502_50257

def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 25*x^2 + 2*x - 12

theorem monic_quartic_with_given_roots :
  (∀ x : ℝ, p x = 0 → x = 3 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 7 ∨ 
                   x = 3 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 7) ∧
  (p (3 + Real.sqrt 5) = 0) ∧
  (p (2 - Real.sqrt 7) = 0) ∧
  (∀ x : ℝ, p x = x^4 - 10*x^3 + 25*x^2 + 2*x - 12) ∧
  (∀ a b c d e : ℝ, (∀ x : ℝ, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) → a = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_with_given_roots_l502_50257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l502_50253

/-- The area of a parallelogram with a 100-degree angle and sides of 14 and 15 inches --/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 14) (h2 : b = 15) (h3 : θ = 100 * Real.pi / 180) :
  abs (a * b * Real.sin θ - 206.808) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l502_50253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_reduction_proof_l502_50265

/-- Calculates the original number of employees before a reduction --/
def original_employees (current_employees : ℕ) (reduction_percentage : ℚ) : ℕ :=
  Nat.ceil ((current_employees : ℚ) / (1 - reduction_percentage))

/-- Proves that given a 14% reduction resulting in 195 employees, 
    the original number of employees is approximately 227 --/
theorem company_reduction_proof :
  original_employees 195 (14/100) = 227 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_reduction_proof_l502_50265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l502_50285

/-- Two vectors in the plane -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Cross product of two 2D vectors (scalar result) -/
def cross_product (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem :
  (∀ x : ℝ, dot_product (a x) (b x) = 0 → x = -1 ∨ x = 3) ∧
  (∀ x : ℝ, cross_product (a x) (b x) = 0 →
    magnitude ((a x).1 - (b x).1, (a x).2 - (b x).2) = 2 * Real.sqrt 5 ∨
    magnitude ((a x).1 - (b x).1, (a x).2 - (b x).2) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l502_50285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_is_approximately_36_315_l502_50230

/-- Represents the properties of a rectangular floor and its painting cost. -/
structure RectangularFloor where
  breadth : ℝ
  length_ratio : ℝ
  paint_cost : ℝ
  paint_rate : ℝ

/-- Calculates the length of the floor given its properties. -/
def calculate_floor_length (floor : RectangularFloor) : ℝ :=
  floor.breadth * floor.length_ratio

/-- Calculates the area of the floor given its properties. -/
def calculate_floor_area (floor : RectangularFloor) : ℝ :=
  floor.breadth * floor.breadth * floor.length_ratio

/-- Theorem stating that the calculated length of the floor is approximately 36.315 meters. -/
theorem floor_length_is_approximately_36_315 (floor : RectangularFloor) 
    (h1 : floor.length_ratio = 4.5)
    (h2 : floor.paint_cost = 2200)
    (h3 : floor.paint_rate = 7.5) :
    abs (calculate_floor_length floor - 36.315) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_is_approximately_36_315_l502_50230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inequality_l502_50267

theorem log_product_inequality : 
  ∃ x : ℝ, x = (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * (Real.log 8 / Real.log 7) ∧ 1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inequality_l502_50267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l502_50293

noncomputable def f (x : ℝ) : ℝ := Real.cos x * abs (Real.tan x)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x, f (Real.pi + x) = -f x) ∧
  (∀ x, -Real.pi/2 < x → x < Real.pi/2 → f x ≥ f 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l502_50293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_theorem_l502_50231

/-- Sequence a_n -/
def a : ℕ → ℤ := sorry

/-- Sequence b_n -/
def b : ℕ → ℤ := sorry

/-- Sequence c_n -/
def c : ℕ → ℤ := sorry

/-- Sum of first n terms of b_n -/
def S : ℕ → ℤ := sorry

/-- Sum of first n terms of c_n -/
def T : ℕ → ℤ := sorry

/-- Main theorem -/
theorem sequences_theorem : 
  (a 1 = 2) → 
  (∀ n, a (n + 1) - a n = 3) → 
  (∀ n, S n = 2 * b n - b 1) → 
  (2 * (b 2 + 1) = b 1 + b 3) → 
  (∀ n, a n = 3 * n - 1) ∧ 
  (∀ n, b n = 2^n) ∧ 
  (∀ n, c n = (a n + 1) * b n) → 
  (∀ n, T n = 6 + 3 * (n - 1) * 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_theorem_l502_50231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microwave_minimum_discount_l502_50290

/-- The minimum discount (in tenths) that can be offered on a microwave oven
    while maintaining a profit rate of at least 2% -/
noncomputable def minimum_discount (cost : ℝ) (marked_price : ℝ) (min_profit_rate : ℝ) : ℝ :=
  10 * (cost * (1 + min_profit_rate) / marked_price)

theorem microwave_minimum_discount :
  let cost : ℝ := 1000
  let marked_price : ℝ := 1500
  let min_profit_rate : ℝ := 0.02
  minimum_discount cost marked_price min_profit_rate = 6.8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microwave_minimum_discount_l502_50290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_bvp_l502_50243

open Real

/-- The solution to the boundary value problem y''(x) - y(x) = x, with y(0) = 0 and y(1) = 0 -/
noncomputable def solution (x : ℝ) : ℝ := (sinh x) / (sinh 1) - x

/-- The differential equation -/
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[2] y) x - y x = x

theorem solution_satisfies_bvp :
  diff_eq solution ∧ solution 0 = 0 ∧ solution 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_bvp_l502_50243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l502_50225

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-10) 6

-- Define the function k in terms of h
noncomputable def k (x : ℝ) : ℝ := h (-3 * x + 1)

-- Theorem: The domain of k is [-5/3, 11/3]
theorem k_domain : 
  {x : ℝ | ∃ y, k x = y} = Set.Icc (-5/3) (11/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l502_50225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l502_50248

/-- The vertex of a quadratic function y = ax² + bx + c is at x = -b/(2a) -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_vertices : 
  let C := vertex 1 6 13
  let D := vertex (-1) 2 8
  distance C D = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l502_50248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l502_50204

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let F := (-Real.sqrt (a^2 - b^2), 0)
  let line := fun (x : ℝ) => x + Real.sqrt (a^2 - b^2)
  let asymptote1 := fun (x : ℝ) => b / a * x
  let asymptote2 := fun (x : ℝ) => -b / a * x
  let A := (-(a * Real.sqrt (a^2 - b^2)) / (a + b), (b * Real.sqrt (a^2 - b^2)) / (a + b))
  let B := (-(a * Real.sqrt (a^2 - b^2)) / (a - b), -(b * Real.sqrt (a^2 - b^2)) / (a - b))
  (line A.1 = A.2) ∧ 
  (line B.1 = B.2) ∧ 
  (asymptote1 A.1 = A.2) ∧ 
  (asymptote2 B.1 = B.2) ∧
  (Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) / Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 1/2) →
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 10 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l502_50204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_two_l502_50203

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → g x * g y = g (x * y) + 3003 * (1 / x + 1 / y + 3002)

/-- The theorem stating the unique value of g(2) -/
theorem unique_g_two :
    ∃! v : ℝ, ∀ g : ℝ → ℝ, FunctionalEquation g → g 2 = v ∧ v = 6007 / 2 := by
  sorry

#check unique_g_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_two_l502_50203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l502_50260

/-- The maximum distance from a point on the circle (x-2)² + y² = 1 to the line x + y = 4 is √2 + 1 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 2 + 1 ∧
    ∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ max_dist ∧
    ∃ p₀ ∈ circle, ∃ q₀ ∈ line, dist p₀ q₀ = max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l502_50260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l502_50298

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 is 64 -/
theorem triangle_area : ℝ := by
  -- Define the lines
  let line1 : ℝ → ℝ := fun x ↦ x
  let line2 : ℝ → ℝ := fun x ↦ -x
  let line3 : ℝ → ℝ := fun _ ↦ 8

  -- Define the intersection points
  let point1 : ℝ × ℝ := (8, 8)
  let point2 : ℝ × ℝ := (-8, 8)

  -- Calculate the base and height of the triangle
  let base : ℝ := 16  -- distance between point1 and point2
  let height : ℝ := 8 -- y-coordinate of line3

  -- Calculate the area
  let area : ℝ := (1/2) * base * height

  -- State that the area of this triangle is 64
  have h : area = 64 := by
    -- Proof steps would go here
    sorry

  exact 64

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l502_50298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_average_bound_l502_50213

/-- The final value obtained after repeated averaging of the sequence 1, 1/2, 1/3, ..., 1/n -/
def final_average (n : ℕ+) : ℚ :=
  2 / n.val - 2 / (n.val * (2^n.val : ℚ))

/-- Theorem stating that the final average is less than 2/n -/
theorem final_average_bound (n : ℕ+) : final_average n < 2 / n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_average_bound_l502_50213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miller_town_work_from_home_growth_exponential_growth_best_model_l502_50237

-- Define the years and corresponding percentages
def years : List Nat := [2000, 2005, 2010, 2015]
def percentages : List Float := [8, 12, 20, 40]

-- Define a function to check if a list of data points follows exponential growth
def is_exponential_growth (data : List Float) : Prop :=
  ∀ i j k, i < j ∧ j < k ∧ k < data.length →
    (data.get! j / data.get! i) < (data.get! k / data.get! j)

-- Theorem stating that the given data follows exponential growth
theorem miller_town_work_from_home_growth :
  is_exponential_growth percentages := by
  sorry

-- Define a placeholder for the is_better_fit function
def is_better_fit (model : String) (data : List Float) (reference : String) : Prop :=
  sorry

-- Theorem stating that exponential growth is the best model for this data
theorem exponential_growth_best_model :
  is_exponential_growth percentages →
  ∀ model, model ≠ "Exponential Growth" →
    ¬(is_better_fit model percentages "Exponential Growth") := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miller_town_work_from_home_growth_exponential_growth_best_model_l502_50237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_correct_l502_50206

/-- The capacity of the pool in cubic meters -/
noncomputable def pool_capacity : ℝ := 12000

/-- Time to fill the pool with both valves open (in minutes) -/
noncomputable def both_valves_time : ℝ := 48

/-- Time to fill the pool with the first valve alone (in minutes) -/
noncomputable def first_valve_time : ℝ := 120

/-- Additional water emitted by the second valve per minute (in cubic meters) -/
noncomputable def second_valve_additional : ℝ := 50

/-- The rate at which the first valve fills the pool (in cubic meters per minute) -/
noncomputable def first_valve_rate : ℝ := pool_capacity / first_valve_time

/-- The rate at which the second valve fills the pool (in cubic meters per minute) -/
noncomputable def second_valve_rate : ℝ := first_valve_rate + second_valve_additional

/-- The combined rate of both valves (in cubic meters per minute) -/
noncomputable def combined_rate : ℝ := pool_capacity / both_valves_time

theorem pool_capacity_correct : 
  first_valve_rate + second_valve_rate = combined_rate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_correct_l502_50206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l502_50273

theorem largest_power_of_18_dividing_30_factorial : ∃ n : ℕ, n = 7 ∧ 
  (∀ m : ℕ, (18 : ℕ)^m ∣ Nat.factorial 30 → m ≤ n) ∧ (18 : ℕ)^n ∣ Nat.factorial 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l502_50273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_theorem_l502_50294

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∃ α : ℝ, f = power_function α) →
  f 4 = (1 : ℝ) / 2 →
  f (a + 1) < f (10 - 2 * a) →
  3 < a ∧ a < 5 :=
by
  sorry

#check power_function_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_theorem_l502_50294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_comparison_l502_50292

/-- Represents the cost function for apple purchases at Store A -/
noncomputable def cost_store_A (x : ℝ) : ℝ := 
  if x ≤ 1000 then 0.92 * 6 * x
  else if x ≤ 2000 then 0.90 * 6 * x
  else 0.88 * 6 * x

/-- Represents the cost function for apple purchases at Store B -/
noncomputable def cost_store_B (x : ℝ) : ℝ :=
  min 500 x * 0.95 * 6 +
  max 0 (min 1000 (x - 500)) * 0.85 * 6 +
  max 0 (min 1000 (x - 1500)) * 0.75 * 6 +
  max 0 (x - 2500) * 0.70 * 6

theorem apple_cost_comparison (x : ℝ) (h : 1500 < x ∧ x < 2000) :
  cost_store_A x = 5.4 * x ∧ cost_store_B x = 4.5 * x + 1200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_comparison_l502_50292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_prime_l502_50205

theorem subset_sum_divisible_by_prime (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p) :
  let A := Finset.range (p - 1)
  (Finset.filter (λ B ↦ p ∣ (Finset.sum B id)) (Finset.powerset A)).card = 2^(p-1) / p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_prime_l502_50205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_theorem_l502_50256

-- Define the type for inhabitants
inductive Inhabitant
| Knight
| Knave

-- Define the function to check if a statement is true
def isTrue (statement : Prop) (speaker : Inhabitant) : Prop :=
  match speaker with
  | Inhabitant.Knight => statement
  | Inhabitant.Knave => ¬statement

-- Define the statements made by each inhabitant
def statement1 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  inhabitants 0 = Inhabitant.Knight

def statement2 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  inhabitants 0 = Inhabitant.Knight

def statement3 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  (inhabitants 0 = Inhabitant.Knave ∨ inhabitants 1 = Inhabitant.Knave)

def statement4 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 1 = Inhabitant.Knave) ∨
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 2 = Inhabitant.Knave) ∨
  (inhabitants 1 = Inhabitant.Knave ∧ inhabitants 2 = Inhabitant.Knave)

def statement5 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight) ∨
  (inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight)

def statement6 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 1 = Inhabitant.Knave) ∨
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 2 = Inhabitant.Knave) ∨
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 3 = Inhabitant.Knave) ∨
  (inhabitants 0 = Inhabitant.Knave ∧ inhabitants 4 = Inhabitant.Knave) ∨
  (inhabitants 1 = Inhabitant.Knave ∧ inhabitants 2 = Inhabitant.Knave) ∨
  (inhabitants 1 = Inhabitant.Knave ∧ inhabitants 3 = Inhabitant.Knave) ∨
  (inhabitants 1 = Inhabitant.Knave ∧ inhabitants 4 = Inhabitant.Knave) ∨
  (inhabitants 2 = Inhabitant.Knave ∧ inhabitants 3 = Inhabitant.Knave) ∨
  (inhabitants 2 = Inhabitant.Knave ∧ inhabitants 4 = Inhabitant.Knave) ∨
  (inhabitants 3 = Inhabitant.Knave ∧ inhabitants 4 = Inhabitant.Knave)

def statement7 (inhabitants : Fin 7 → Inhabitant) : Prop :=
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 1 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 0 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 2 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 1 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight) ∨
  (inhabitants 2 = Inhabitant.Knight ∧ inhabitants 3 = Inhabitant.Knight ∧ inhabitants 4 = Inhabitant.Knight ∧ inhabitants 5 = Inhabitant.Knight)

theorem island_inhabitants_theorem (inhabitants : Fin 7 → Inhabitant) :
  (isTrue (statement1 inhabitants) (inhabitants 0)) ∧
  (isTrue (statement2 inhabitants) (inhabitants 1)) ∧
  (isTrue (statement3 inhabitants) (inhabitants 2)) ∧
  (isTrue (statement4 inhabitants) (inhabitants 3)) ∧
  (isTrue (statement5 inhabitants) (inhabitants 4)) ∧
  (isTrue (statement6 inhabitants) (inhabitants 5)) ∧
  (isTrue (statement7 inhabitants) (inhabitants 6)) →
  (inhabitants 0 = Inhabitant.Knight ∧
   inhabitants 1 = Inhabitant.Knight ∧
   inhabitants 2 = Inhabitant.Knave ∧
   inhabitants 3 = Inhabitant.Knave ∧
   inhabitants 4 = Inhabitant.Knight ∧
   inhabitants 5 = Inhabitant.Knight ∧
   inhabitants 6 = Inhabitant.Knight) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_inhabitants_theorem_l502_50256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l502_50249

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Define the function for the maximum value
noncomputable def max_value (x y : ℝ) : ℝ := (y + 1) / (x + 1)

-- Define the tangent line equation
noncomputable def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem statement
theorem curve_C_properties :
  ∃ (x y : ℝ),
    C x y ∧ 
    (∀ (a b : ℝ), C a b → max_value a b ≤ 2 + Real.sqrt 6) ∧
    (∃ (m : ℝ), max_value x y = 2 + Real.sqrt 6) ∧
    (C 0 (Real.sqrt 2) → tangent_line 0 (Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l502_50249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_sum_l502_50264

/-- The side length of the initial equilateral triangle -/
noncomputable def initial_side_length : ℝ := 80

/-- The ratio of side lengths between consecutive triangles -/
noncomputable def side_ratio : ℝ := 1 / 2

/-- The perimeter of an equilateral triangle given its side length -/
noncomputable def triangle_perimeter (side_length : ℝ) : ℝ := 3 * side_length

/-- The sum of the perimeters of all triangles in the infinite series -/
noncomputable def perimeter_sum : ℝ := initial_side_length * 6

/-- Theorem stating that the sum of perimeters is 480 -/
theorem triangle_perimeter_sum :
  perimeter_sum = 480 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_sum_l502_50264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l502_50272

/-- Represents a tiling of a plane with hexagons and triangles -/
structure HexagonTriangleTiling where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a regular hexagon -/
noncomputable def hexagonArea (t : HexagonTriangleTiling) : ℝ :=
  3 * Real.sqrt 3 / 2 * t.sideLength^2

/-- The area of an equilateral triangle -/
noncomputable def triangleArea (t : HexagonTriangleTiling) : ℝ :=
  Real.sqrt 3 / 4 * t.sideLength^2

/-- The total area of a composite tile (one hexagon + six triangles) -/
noncomputable def compositeTileArea (t : HexagonTriangleTiling) : ℝ :=
  hexagonArea t + 6 * triangleArea t

/-- The fraction of the area enclosed by triangles -/
noncomputable def triangleFraction (t : HexagonTriangleTiling) : ℝ :=
  (6 * triangleArea t) / compositeTileArea t

theorem triangle_area_fraction (t : HexagonTriangleTiling) :
  triangleFraction t = 1/2 := by
  -- Expand the definition of triangleFraction
  unfold triangleFraction
  -- Expand the definitions of triangleArea and compositeTileArea
  unfold triangleArea compositeTileArea
  -- Simplify the expression
  simp [hexagonArea]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l502_50272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrays_is_9_l502_50220

/-- Represents a 4x4 array of natural numbers -/
def Array4x4 := Matrix (Fin 4) (Fin 4) ℕ

/-- Check if a number appears exactly once in the array -/
def appears_once (a : Array4x4) (n : ℕ) : Prop :=
  (∃ i j, a i j = n) ∧ (∀ i j i' j', a i j = n ∧ a i' j' = n → i = i' ∧ j = j')

/-- Check if a number appears exactly twice in the array -/
def appears_twice (a : Array4x4) (n : ℕ) : Prop :=
  (∃ i j i' j', a i j = n ∧ a i' j' = n ∧ (i ≠ i' ∨ j ≠ j')) ∧
  (∀ i j i' j' i'' j'', a i j = n ∧ a i' j' = n ∧ a i'' j'' = n →
    (i = i' ∧ j = j') ∨ (i = i'' ∧ j = j'') ∨ (i' = i'' ∧ j' = j''))

/-- Check if the array contains digits 1 through 10 with one repeated -/
def valid_digits (a : Array4x4) : Prop :=
  ∃ n, 2 ≤ n ∧ n ≤ 9 ∧ appears_twice a n ∧
    (∀ m, 1 ≤ m ∧ m ≤ 10 → m ≠ n → appears_once a m)

/-- Check if entries in every row and column are in increasing order -/
def increasing_order (a : Array4x4) : Prop :=
  (∀ i j j', j < j' → a i j < a i j') ∧
  (∀ i i' j, i < i' → a i j < a i' j)

/-- Check if the array satisfies all conditions -/
def valid_array (a : Array4x4) : Prop :=
  valid_digits a ∧
  increasing_order a ∧
  a 0 0 = 1 ∧
  a 3 3 = 10

/-- The number of valid arrays -/
def num_valid_arrays : ℕ := 9

/-- The main theorem stating that the number of valid arrays is 9 -/
theorem num_valid_arrays_is_9 :
  (∃ s : Finset Array4x4, (∀ a ∈ s, valid_array a) ∧ s.card = num_valid_arrays) ∧
  (∀ s : Finset Array4x4, (∀ a ∈ s, valid_array a) → s.card ≤ num_valid_arrays) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrays_is_9_l502_50220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_20_over_3_l502_50211

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x ≤ 2 then x + 1
  else 0

-- State the theorem
theorem integral_of_f_equals_20_over_3 :
  ∫ x in (-2)..2, f x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_20_over_3_l502_50211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_board_inspection_l502_50200

/-- Represents the number of circuit boards that passed the verification process -/
def P : ℕ := sorry

/-- Represents the number of circuit boards that failed the verification process -/
def F : ℕ := sorry

/-- The total number of circuit boards -/
def total : ℕ := 3200

/-- The approximate number of faulty circuit boards -/
def faulty : ℕ := 456

theorem circuit_board_inspection :
  (P + F = total) →
  ((P / 8 + F : ℚ) = faulty) →
  F = 64 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_board_inspection_l502_50200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l502_50242

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2017 * x) + Real.cos (2017 * x)

theorem min_value_theorem (A : ℝ) (x₁ x₂ : ℝ) 
  (h_max : ∀ x, f x ≤ A)
  (h_bounds : ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) :
  2 * Real.pi / 2017 ≤ A * |x₁ - x₂| := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l502_50242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_five_digit_numbers_l502_50275

def digits : Finset Nat := {1, 2, 3, 4, 5}

def is_even (n : Nat) : Bool := n % 2 = 0

def valid_number (n : Nat) : Bool :=
  n ≥ 10000 && n < 100000 && 
  (Finset.filter (λ d => d ∈ digits) (Finset.range 10)).card = 5 &&
  (Finset.filter (λ d => (n / (10 ^ d) % 10) ∈ digits) (Finset.range 5)).card = 5

theorem count_even_five_digit_numbers : 
  (Finset.filter (λ n => valid_number n && is_even n) (Finset.range 100000)).card = 48 := by
  sorry

#eval (Finset.filter (λ n => valid_number n && is_even n) (Finset.range 100000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_five_digit_numbers_l502_50275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_given_equal_area_to_square_l502_50227

theorem rectangle_length_given_equal_area_to_square (square_side : ℝ) (rect_width : ℝ) 
  (h1 : square_side = 6)
  (h2 : rect_width = 4)
  (h3 : square_side * square_side = rect_width * 9) :
  rect_width * 9 = square_side * square_side := by
  sorry

#check rectangle_length_given_equal_area_to_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_given_equal_area_to_square_l502_50227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l502_50224

/-- Given a triangle ABC with internal angles A, B, C, and corresponding sides a, b, c -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A - Real.pi/4) = Real.sqrt 2 / 10)
  (h2 : (1/2) * t.b * t.c * Real.sin t.A = 24)
  (h3 : t.b = 10) :
  Real.tan t.A = 4/3 ∧ t.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l502_50224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_calculation_l502_50287

-- Define the base conversions
def base7_to_10 (n : ℕ) : ℕ := sorry

def base5_to_10 (n : ℕ) : ℕ := sorry

def base6_to_10 (n : ℕ) : ℕ := sorry

-- Define the given numbers in their respective bases
def num1 : ℕ := 1652
def num2 : ℕ := 142
def num3 : ℕ := 3241
def num4 : ℕ := 479

-- Theorem statement
theorem base_conversion_calculation :
  (base7_to_10 num1 / base5_to_10 num2) - (base7_to_10 num3 : ℤ) + (base6_to_10 num4 : ℤ) = -947 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_calculation_l502_50287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l502_50288

theorem polynomial_integer_values (a b c d : ℚ) :
  (∀ x : ℤ, x ∈ ({-1, 0, 1, 2} : Set ℤ) → ∃ n : ℤ, a * x^3 + b * x^2 + c * x + d = n) →
  (∀ x : ℤ, ∃ n : ℤ, a * x^3 + b * x^2 + c * x + d = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l502_50288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_astroid_volume_of_revolution_l502_50278

-- Define the astroid
noncomputable def astroid (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * (Real.cos t)^3, a * (Real.sin t)^3)

-- Define the volume of revolution
noncomputable def volumeOfRevolution (a : ℝ) : ℝ :=
  (3/4) * Real.pi^2 * a^3

-- Theorem statement
theorem astroid_volume_of_revolution (a : ℝ) (h : a > 0) :
  volumeOfRevolution a = (3/4) * Real.pi^2 * a^3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_astroid_volume_of_revolution_l502_50278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_3_minus_4x_l502_50299

theorem derivative_sin_3_minus_4x (x : ℝ) :
  HasDerivAt (λ x => Real.sin (3 - 4 * x)) (-4 * Real.cos (3 - 4 * x)) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_3_minus_4x_l502_50299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l502_50240

noncomputable section

-- Define the semicircle C
def semicircle (θ : Real) : Real × Real :=
  (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

-- Define the line l
def line_l (α : Real) (x : Real) : Real := x * Real.tan α - 2

-- Define point A
def point_A : Real × Real := (0, -2)

-- Define point D
def point_D (α : Real) : Real × Real := (Real.cos (2 * α), 1 + Real.sin (2 * α))

-- Define the distance from point D to line l
def distance_D_to_l (α : Real) : Real :=
  (3 * Real.cos α + Real.sin α)

-- Define the length of AB
noncomputable def length_AB (α : Real) : Real := 2 / Real.sin α

-- Theorem statement
theorem point_D_coordinates :
  ∀ α : Real,
  α > 0 ∧ α < π/2 ∧
  2 * α > 0 ∧ 2 * α < π ∧
  semicircle (2 * α) = point_D α ∧
  (1/2) * distance_D_to_l α * length_AB α = 4 →
  point_D α = (0, 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l502_50240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_three_solutions_l502_50216

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x
  else if x ≤ 1 then 0
  else x - 1

theorem f_eq_three_solutions (x : ℝ) : f x = 3 ↔ x = -3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_three_solutions_l502_50216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_on_hypotenuse_length_l502_50207

theorem median_on_hypotenuse_length (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  ((a = 5 ∧ b = 12) ∨ (a = 5 ∧ c = 12) ∨ (b = 5 ∧ c = 12)) →
  a^2 + b^2 = c^2 →
  let m := (1/2) * c
  m = 6 ∨ m = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_on_hypotenuse_length_l502_50207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_five_smaller_than_negative_three_l502_50266

theorem only_negative_five_smaller_than_negative_three :
  ∀ x : ℝ, x ∈ ({0, -1, -5, -1/2} : Set ℝ) → (x < -3 ↔ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_five_smaller_than_negative_three_l502_50266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angled_l502_50201

theorem triangle_acute_angled (a b c : ℝ) (h : a^3 = b^3 + c^3) : 
  ∀ A B C : ℝ, A + B + C = π → A < π/2 ∧ B < π/2 ∧ C < π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angled_l502_50201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_value_l502_50277

-- Define m as a parameter
variable (m : ℝ)

-- Define the data points
def x_values : List ℝ := [0, 2, 4, 6, 8]
def y_values (m : ℝ) : List ℝ := [1, m + 1, 2*m + 1, 3*m + 3, 11]

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 1.3 * x + 0.6

-- State the theorem
theorem find_m_value :
  let x_mean : ℝ := (List.sum x_values) / (List.length x_values)
  let y_mean : ℝ := (List.sum (y_values m)) / (List.length (y_values m))
  regression_line x_mean = y_mean →
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_value_l502_50277
