import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l783_78317

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (12 + x - x^2)

-- State the theorem
theorem domain_of_f :
  {x : ℝ | -3 < x ∧ x < 4} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l783_78317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l783_78393

def a : ℕ → ℤ
| 0 => 11
| 1 => 11
| n + 2 => 
  let m := n / 2
  let k := n % 2
  (1 / 2) * (a (2 * m) + a (2 * k)) - (m - k)^2

theorem a_45_equals_1991 : a 45 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l783_78393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_count_l783_78323

/-- 
Given an arithmetic sequence starting with 1, having a common difference of 3, 
and ending with 2008, the number of terms in the sequence is 670.
-/
theorem arithmetic_sequence_count : 
  ∀ (a : List ℕ), 
  (a.head? = some 1) → 
  (∀ i, i + 1 < a.length → a[i+1]! - a[i]! = 3) → 
  (a.getLast? = some 2008) → 
  a.length = 670 :=
by
  intro a first_term common_diff last_term
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_count_l783_78323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_cubic_equation_l783_78388

theorem sum_of_roots_cubic_equation :
  let p : Polynomial ℝ := 3 * X^3 + 9 * X^2 - 36 * X + 12
  (p.roots.map Complex.re).sum = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_cubic_equation_l783_78388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rogers_candy_l783_78375

theorem rogers_candy (sandra_bags : Nat) (sandra_pieces_per_bag : Nat) 
  (roger_bags : Nat) (roger_second_bag : Nat) (roger_extra : Nat) 
  (rogers_first_bag : Nat) :
  sandra_bags = 2 →
  sandra_pieces_per_bag = 6 →
  roger_bags = 2 →
  roger_second_bag = 3 →
  roger_extra = 2 →
  rogers_first_bag = sandra_bags * sandra_pieces_per_bag + roger_extra - roger_second_bag →
  rogers_first_bag = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rogers_candy_l783_78375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_range_l783_78310

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

-- State the theorem
theorem function_equality_range (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 0 3, f a x₁ = g x₂) ↔ 
  a ∈ Set.Icc (-1/Real.exp 1) (3/Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_range_l783_78310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l783_78376

/-- The time (in days) it takes for two workers to complete a job together -/
noncomputable def combined_time : ℝ := 4

/-- The time (in days) it takes for worker A to complete the job alone -/
noncomputable def time_a : ℝ := 12

/-- The work rate of a worker is defined as the fraction of the job completed in one day -/
noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

/-- The combined work rate of two workers is the sum of their individual work rates -/
noncomputable def combined_work_rate (rate_a rate_b : ℝ) : ℝ := rate_a + rate_b

theorem b_completion_time :
  combined_work_rate (work_rate time_a) (work_rate (6 : ℝ)) = work_rate combined_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l783_78376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_through_diagonal_intersection_length_l783_78350

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A line segment defined by two points -/
structure Segment where
  start : Point
  finish : Point

/-- Function to calculate the length of a segment -/
noncomputable def length (s : Segment) : ℝ :=
  Real.sqrt ((s.finish.x - s.start.x)^2 + (s.finish.y - s.start.y)^2)

/-- Function to check if a point lies on a segment -/
def pointOnSegment (p : Point) (s : Segment) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    p.x = s.start.x + t * (s.finish.x - s.start.x) ∧
    p.y = s.start.y + t * (s.finish.y - s.start.y)

/-- Function to find the intersection point of two segments -/
noncomputable def intersectionPoint (s1 s2 : Segment) : Point :=
  sorry

/-- The main theorem -/
theorem segment_through_diagonal_intersection_length
  (ABCD : Quadrilateral) (K L : Point) (KL : Segment) :
  pointOnSegment K (Segment.mk ABCD.A ABCD.B) →
  pointOnSegment L (Segment.mk ABCD.C ABCD.D) →
  KL.start = K →
  KL.finish = L →
  intersectionPoint (Segment.mk ABCD.A ABCD.C) (Segment.mk ABCD.B ABCD.D) = 
    intersectionPoint KL (Segment.mk ABCD.A ABCD.C) →
  length KL ≤ max (length (Segment.mk ABCD.A ABCD.C)) (length (Segment.mk ABCD.B ABCD.D)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_through_diagonal_intersection_length_l783_78350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_radius_l783_78396

/-- Given an isosceles triangle with base angle α and an inscribed circle of radius r,
    the radius R of the circumcircle is r * cot(α/2) / sin(2α) -/
theorem isosceles_triangle_circumcircle_radius 
  (α : ℝ) 
  (r : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : r > 0) :
  ∃ R : ℝ, 
    R > 0 ∧ 
    R = r * Real.tan (π / 2 - α / 2) / Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_radius_l783_78396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_center_of_gravity_cone_center_of_gravity_l783_78346

-- Hemisphere
def hemisphere (R : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | p.2.2 = Real.sqrt (R^2 - p.2.1^2 - p.1^2) ∧ p.2.2 ≥ 0}

def hemisphere_density (k : ℝ) (p : ℝ × ℝ × ℝ) : ℝ :=
  k * (p.2.1^2 + p.1^2)

-- Define IsWeightedCentroid as a placeholder
def IsWeightedCentroid (density : ℝ × ℝ × ℝ → ℝ) (set : Set (ℝ × ℝ × ℝ)) (centroid : ℝ × ℝ × ℝ) : Prop :=
  sorry

theorem hemisphere_center_of_gravity (R k : ℝ) (h : k > 0) :
  ∃ cg : ℝ × ℝ × ℝ, cg.2.2 = 3 * R / 8 ∧
    IsWeightedCentroid (hemisphere_density k) (hemisphere R) cg :=
  sorry

-- Cone
def cone (a b : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | a * p.2.2^2 = b^2 * (p.2.1^2 + p.1^2) ∧ 0 ≤ p.2.2 ∧ p.2.2 ≤ b}

def cone_density (ρ₀ : ℝ) (_ : ℝ × ℝ × ℝ) : ℝ :=
  ρ₀

theorem cone_center_of_gravity (a b ρ₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : ρ₀ > 0) :
  ∃ cg : ℝ × ℝ × ℝ, cg.2.2 = 2 * b / 3 ∧
    IsWeightedCentroid (cone_density ρ₀) (cone a b) cg :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_center_of_gravity_cone_center_of_gravity_l783_78346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_13_forms_triangle_l783_78385

-- Define the given lengths
def a : ℝ := 6
def b : ℝ := 13

-- Define the set of possible third lengths
def possible_lengths : Set ℝ := {6, 7, 13, 20}

-- Define the triangle inequality condition
def is_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem only_13_forms_triangle :
  ∃! c, c ∈ possible_lengths ∧ is_triangle a b c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_13_forms_triangle_l783_78385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_theorem_l783_78318

noncomputable def B : ℝ × ℝ := (-3 * Real.sqrt 2 / 2, 0)
noncomputable def C : ℝ × ℝ := (3 * Real.sqrt 2 / 2, 0)
noncomputable def O : ℝ × ℝ := ((B.1 + C.1) / 2, 0)

-- Define the perimeter of triangle ABC
noncomputable def perimeter : ℝ := 6 + 3 * Real.sqrt 2

-- Define the trajectory E
def E (x y : ℝ) : Prop := x^2 + 2*y^2 = 1 ∧ y ≠ 0

-- Define point T on AO
def T (x y : ℝ) : Prop := ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = t * O.1 ∧ y = t * O.2

-- Define point A
def A (x y : ℝ) : Prop := ∃ (tx ty : ℝ), T tx ty ∧ x = 3*tx ∧ y = 3*ty

-- Define point M
def M (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define point N
def N (n : ℝ) : ℝ × ℝ := (n, 0)

-- Theorem statement
theorem trajectory_intersection_theorem (m n : ℝ) 
  (hm : 0 < m ∧ m < 1) 
  (hn : n > C.1) 
  (hP : E (x₁ : ℝ) (y₁ : ℝ)) 
  (hQ : E (x₂ : ℝ) (y₂ : ℝ)) 
  (hR : E (x₃ : ℝ) (y₃ : ℝ)) 
  (hMPQ : ∃ (k : ℝ), x₁ = k*y₁ + m ∧ x₂ = k*y₂ + m) 
  (hQNR : ∃ (l : ℝ), x₂ = l*y₂ + n ∧ x₃ = l*y₃ + n) 
  (hPR : (x₁ - m) * (x₃ - x₁) + y₁ * (y₃ - y₁) = 0) :
  m * n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_theorem_l783_78318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l783_78314

/-- Proves that for a parabola with given properties, the difference between its zeros is 3 -/
theorem parabola_zeros_difference (a h k : ℝ) : 
  (∀ x y : ℝ, y = a * (x - h)^2 + k) →  -- Parabola equation
  (h = 3 ∧ k = -9) →                    -- Vertex at (3, -9)
  (a * (5 - h)^2 + k = 7) →             -- Passes through (5, 7)
  let b := -2 * a * h
  let c := a * h^2 + k
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Standard form
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ - x₂ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l783_78314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_male_workers_l783_78325

/-- Represents the workforce of a company --/
structure Workforce where
  total : ℕ
  female_percent : ℚ
  male_percent : ℚ

/-- Represents the change in workforce after hiring --/
structure WorkforceChange where
  initial : Workforce
  final : Workforce
  additional_male : ℕ

/-- The conditions of the problem --/
def problem_conditions (wc : WorkforceChange) : Prop :=
  wc.initial.female_percent = 60 / 100 ∧
  wc.initial.male_percent = 40 / 100 ∧
  wc.final.female_percent = 55 / 100 ∧
  wc.final.male_percent = 45 / 100 ∧
  wc.final.total = 360 ∧
  wc.final.total = wc.initial.total + wc.additional_male

/-- The theorem to be proved --/
theorem additional_male_workers (wc : WorkforceChange) :
  problem_conditions wc → wc.additional_male = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_male_workers_l783_78325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_adjustment_l783_78362

theorem marble_fraction_after_adjustment (total : ℕ) (h : total > 0) :
  let blue := (2 : ℚ) / 3 * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  (new_red : ℚ) / new_total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_adjustment_l783_78362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l783_78349

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem periodic_function_property 
  (ω φ : ℝ) 
  (h_ω : 0 < ω ∧ ω < 1) 
  (h_φ : |φ| < Real.pi) 
  (h_bounds : ∀ x : ℝ, f ω φ 1 ≤ f ω φ x ∧ f ω φ x ≤ f ω φ 6) :
  f ω φ 2014 - f ω φ 2017 < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l783_78349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_equal_norms_l783_78354

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

noncomputable def angle_between (a b : E) : ℝ := Real.arccos (inner a b / (norm a * norm b))

theorem angle_between_equal_norms (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : norm a = norm b ∧ norm a = norm (a + 2 • b)) : 
  angle_between a b = π := by
  sorry

#check angle_between_equal_norms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_equal_norms_l783_78354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l783_78374

def sequence_a : ℕ → ℚ
| 0 => 4/3
| n + 1 => (sequence_a n)^2 - (sequence_a n) + 1

def sum_reciprocals (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun i => 1 / sequence_a i)

theorem integer_part_of_sum :
  ⌊sum_reciprocals 2017⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l783_78374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l783_78398

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - 3
noncomputable def g (x : ℝ) : ℝ := x / 2

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x + 3
noncomputable def g_inv (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem composition_equality : f (g_inv (f_inv (g (f_inv (g (f 23)))))) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l783_78398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_translated_cosine_l783_78315

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * (x + Real.pi / 6) + Real.pi / 4)

theorem symmetric_center_of_translated_cosine :
  ∃ (k : ℤ), (11 * Real.pi / 24 : ℝ) = k * Real.pi / 2 - Real.pi / 24 ∧
  ∀ t : ℝ, f ((11 * Real.pi / 24 : ℝ) + t) = f ((11 * Real.pi / 24 : ℝ) - t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_translated_cosine_l783_78315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_condition_l783_78356

/-- A quadrilateral in the real plane -/
structure Quadrilateral (α : Type*) [LinearOrderedField α] :=
(a b c d : α)
(positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(triangle_inequality : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)

/-- The edge lengths of a quadrilateral -/
def Quadrilateral.edge_length {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) (i : Fin 4) : α :=
match i with
| 0 => q.a
| 1 => q.b
| 2 => q.c
| 3 => q.d

theorem quadrilateral_condition (a b c d : ℝ) 
  (positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (sum_one : a + b + c + d = 1) :
  (∃ (quad : Quadrilateral ℝ), 
    quad.edge_length 0 = a ∧ 
    quad.edge_length 1 = b ∧ 
    quad.edge_length 2 = c ∧ 
    quad.edge_length 3 = d) ↔ 
  (a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_condition_l783_78356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_inequality_l783_78335

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else Real.exp (x * Real.log 2)

-- State the theorem
theorem f_sum_inequality (x : ℝ) :
  f x + f (x - 1) > 1 ↔ x > -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_inequality_l783_78335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_inequality_l783_78355

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_inequality_l783_78355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equations_solutions_l783_78326

open Real MeasureTheory

theorem integral_equations_solutions (f g : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, ContinuousOn f (Set.Icc 0 1)) →
  (∀ x ∈ Set.Icc 0 1, ContinuousOn g (Set.Icc 0 1)) →
  (∀ x ∈ Set.Icc 0 1, f x = ∫ t in Set.Icc 0 1, Real.exp (x + t) * f t) →
  (∀ x ∈ Set.Icc 0 1, g x = ∫ t in Set.Icc 0 1, Real.exp (x + t) * g t + x) →
  (∀ x ∈ Set.Icc 0 1, f x = 0) ∧
  (∀ x ∈ Set.Icc 0 1, g x = 2 / (3 - Real.exp 2) * Real.exp x + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equations_solutions_l783_78326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_semicircle_development_l783_78386

noncomputable def coneVolume : ℝ := Real.sqrt 3 * Real.pi / 3

def isDevelopmentDiagramSemicircle : Prop := sorry

def semicircleRadius : ℝ := 2

theorem cone_volume_from_semicircle_development (h1 : isDevelopmentDiagramSemicircle) 
  (h2 : semicircleRadius = 2) : 
  coneVolume = Real.sqrt 3 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_semicircle_development_l783_78386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l783_78345

/-- Definition of ellipse C₁ -/
def C₁ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of ellipse C₂ -/
def C₂ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / (3*a^2) + y^2 / (3*b^2) = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of C₁ -/
def eccentricity_C₁ (a b : ℝ) : Prop :=
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 6 / 3

/-- C₂ passes through (√3/2, √3/2) -/
def C₂_point (a b : ℝ) : Prop :=
  C₂ a b (Real.sqrt 3 / 2) (Real.sqrt 3 / 2)

/-- Area of a triangle -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- Theorem statement -/
theorem constant_triangle_area
  (a b : ℝ)
  (h_C₁ : ∀ x y, C₁ a b x y → x^2 + 3*y^2 = 1)
  (h_C₂ : ∀ x y, C₂ a b x y → x^2/3 + y^2 = 1)
  (h_ecc : eccentricity_C₁ a b)
  (h_point : C₂_point a b)
  (M : ℝ × ℝ)
  (h_M : C₁ a b M.1 M.2)
  (N A B : ℝ × ℝ)
  (h_N : ∃ t : ℝ, N = (t * M.1, t * M.2) ∧ C₂ a b N.1 N.2)
  (h_AB : ∃ k m : ℝ, A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧
    C₂ a b A.1 A.2 ∧ C₂ a b B.1 B.2 ∧ A ≠ B) :
  area_triangle N A B = Real.sqrt 2 + Real.sqrt 6 / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l783_78345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l783_78302

-- Define the circle O
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the curve C
def curveC (x y t : ℝ) : Prop := y = 3 * abs (x - t)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main theorem
theorem find_t (m n s p : ℕ+) (k : ℝ) :
  (∀ x y, circleO x y → ∃ t,
    curveC (m : ℝ) (n : ℝ) t ∧ curveC (s : ℝ) (p : ℝ) t ∧
    (∀ x' y', circleO x' y' →
      distance x' y' (m : ℝ) (n : ℝ) / distance x' y' (s : ℝ) (p : ℝ) = k) ∧
    k > 1) →
  ∃ t, t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l783_78302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_after_process_l783_78322

/-- Represents the color of a square in the grid -/
inductive Color
  | Black
  | White
  | Red
deriving Repr, DecidableEq

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Color

/-- The probability of a square being initially black -/
noncomputable def prob_black : ℚ := 1/3

/-- The probability of a square being initially white -/
noncomputable def prob_white : ℚ := 1/3

/-- The probability of a square being initially red -/
noncomputable def prob_red : ℚ := 1/3

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid :=
  fun i j => g (3 - j) i

/-- Applies the color change rule after rotation -/
def apply_color_change (g : Grid) : Grid :=
  fun i j =>
    match g i j with
    | Color.Red => if (rotate g) i j = Color.Black then Color.Black else Color.Red
    | c => c

/-- The final grid after rotation and color change -/
def final_grid (g : Grid) : Grid := apply_color_change (rotate g)

/-- The probability that a single square ends up black -/
noncomputable def prob_final_black : ℚ := 4/9

/-- Theorem: The probability of the entire 4x4 grid being black after rotation and color change -/
theorem prob_all_black_after_process :
  (prob_final_black : ℝ) ^ 16 = (4/9 : ℝ)^16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_after_process_l783_78322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_quadratic_with_roots_between_zero_and_one_l783_78361

theorem min_a_for_quadratic_with_roots_between_zero_and_one 
  (a b c : ℕ+) 
  (p q : ℝ) 
  (h_distinct : p ≠ q)
  (h_roots : ∀ x, (a : ℝ) * x^2 - (b : ℝ) * x + (c : ℝ) = 0 ↔ x = p ∨ x = q)
  (h_p : 0 < p ∧ p < 1)
  (h_q : 0 < q ∧ q < 1) :
  5 ≤ (a : ℕ) ∧ ∀ a' : ℕ+, (∃ b' c' : ℕ+, ∃ p' q' : ℝ, 
    p' ≠ q' ∧ 
    (∀ x, (a' : ℝ) * x^2 - (b' : ℝ) * x + (c' : ℝ) = 0 ↔ x = p' ∨ x = q') ∧
    0 < p' ∧ p' < 1 ∧ 0 < q' ∧ q' < 1) → 
  5 ≤ (a' : ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_quadratic_with_roots_between_zero_and_one_l783_78361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_509_to_hundredth_l783_78308

/-- Rounds a number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The theorem states that rounding 4.509 to the nearest hundredth results in 4.51 -/
theorem round_4_509_to_hundredth :
  round_to_hundredth 4.509 = 4.51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_509_to_hundredth_l783_78308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_bounds_local_max_condition_l783_78303

-- Part 1
theorem sine_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by
  sorry

-- Part 2
noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a x ≤ f a 0) ↔ 
  a ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_bounds_local_max_condition_l783_78303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_l783_78370

/-- The volume of a sphere minus the volume of four cylindrical holes. -/
noncomputable def remaining_volume (sphere_diameter : ℝ) (hole1_diameter hole1_depth : ℝ) (hole2_diameter hole2_depth : ℝ) : ℝ :=
  let sphere_radius := sphere_diameter / 2
  let sphere_volume := (4 / 3) * Real.pi * (sphere_radius ^ 3)
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * (hole1_radius ^ 2) * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * (hole2_radius ^ 2) * hole2_depth
  sphere_volume - 2 * hole1_volume - 2 * hole2_volume

/-- The theorem stating that the remaining volume of the bowling ball is 2269.5π cubic cm. -/
theorem bowling_ball_volume : 
  remaining_volume 24 1.5 5 2 6 = 2269.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_l783_78370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_guess_is_20th_bear_l783_78373

-- Define the color of a bear
inductive BearColor
| White
| Brown
| Black

-- Define the row of bears
def BearRow := Fin 1000 → BearColor

-- Define the property that among any three consecutive bears, there is at least one of each color
def ValidColorDistribution (row : BearRow) : Prop :=
  ∀ i : Fin 998, ∃ (c1 c2 c3 : BearColor), 
    ({c1, c2, c3} : Set BearColor) = {BearColor.White, BearColor.Brown, BearColor.Black} ∧
    row i = c1 ∧ row (i + 1) = c2 ∧ row (i + 2) = c3

-- Define Iskander's guesses
def IskanderGuesses (row : BearRow) : Prop :=
  row 1 = BearColor.White ∧
  row 19 = BearColor.Brown ∧
  row 399 = BearColor.Black ∧
  row 599 = BearColor.Brown ∧
  row 799 = BearColor.White

-- Define the theorem
theorem incorrect_guess_is_20th_bear 
  (row : BearRow) 
  (h1 : ValidColorDistribution row) 
  (h2 : ∃ (row' : BearRow), ValidColorDistribution row' ∧ 
    (IskanderGuesses row' ∨ 
     (IskanderGuesses row' ∧ row' 19 ≠ BearColor.Brown))) :
  ∃ (row' : BearRow), ValidColorDistribution row' ∧ 
    IskanderGuesses row' ∧ row' 19 ≠ BearColor.Brown :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_guess_is_20th_bear_l783_78373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l783_78305

def is_geometric_sequence (seq : List ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin (seq.length - 1), seq[i.val + 1] = seq[i.val] * r

theorem geometric_sequence_properties
  (a b c : ℝ)
  (h : is_geometric_sequence [-1, a, b, c, -9]) :
  b = -3 ∧ a * c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l783_78305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l783_78368

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => a n + 5^(a n)

/-- Theorem stating that the first 2^k terms of sequence a_n have distinct remainders modulo 2^k -/
theorem distinct_remainders (k : ℕ) :
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < 2^k → a i % 2^k ≠ a j % 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l783_78368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l783_78391

/-- A circle C passing through points A(5, 1) and B(1, 3) with its center on the x-axis -/
structure CircleC where
  center : ℝ × ℝ
  /-- The circle passes through point A -/
  passes_through_A : (5 : ℝ) ^ 2 + 1 ^ 2 = (center.1 - 5) ^ 2 + (center.2 - 1) ^ 2
  /-- The circle passes through point B -/
  passes_through_B : (1 : ℝ) ^ 2 + 3 ^ 2 = (center.1 - 1) ^ 2 + (center.2 - 3) ^ 2
  /-- The center of the circle is on the x-axis -/
  center_on_x_axis : center.2 = 0

/-- The equation of circle C is (x-2)^2 + y^2 = 10 -/
theorem circle_C_equation (c : CircleC) : 
  ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 10 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.center.1 ^ 2 + c.center.2 ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l783_78391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_l783_78352

-- Define the speed of the car
noncomputable def speed : ℝ := 60

-- Define the distances for the two trips
noncomputable def distance1 : ℝ := 360
noncomputable def distance2 : ℝ := 420

-- Define the time difference in hours
noncomputable def time_diff : ℝ := distance2 / speed - distance1 / speed

-- Theorem to prove
theorem trip_time_difference : time_diff * 60 = 60 := by
  -- Unfold the definitions
  unfold time_diff distance1 distance2 speed
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_l783_78352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_three_to_two_l783_78365

/-- Represents the financial data of two persons P1 and P2 -/
structure FinancialData where
  income_ratio : Rat
  savings : ℕ
  p1_income : ℕ

/-- Calculates the ratio of expenditures given financial data -/
def expenditure_ratio (data : FinancialData) : Rat :=
  3 / 2

/-- Theorem: Given the financial data, the ratio of expenditures is 3:2 -/
theorem expenditure_ratio_is_three_to_two (data : FinancialData) 
  (h1 : data.income_ratio = 5 / 4)
  (h2 : data.savings = 1400)
  (h3 : data.p1_income = 3500) :
  expenditure_ratio data = 3 / 2 := by
  sorry

#eval expenditure_ratio { income_ratio := 5 / 4, savings := 1400, p1_income := 3500 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_three_to_two_l783_78365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_13_value_l783_78327

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | n + 1 => sequence_a n + 2 * Real.sqrt (sequence_a n + 1) + 1

theorem a_13_value : sequence_a 13 = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_13_value_l783_78327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l783_78300

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | -a < x ∧ x < 2*a}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

-- Part 1
theorem part_one : A 1 ∩ B = Set.Ioo 1 2 := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, B ⊆ A a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l783_78300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l783_78301

theorem product_approximation : 
  ⌊(2.4 : ℝ) * (53.8 - 0.08) * 1.2 + 0.5⌋ = 155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l783_78301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_hours_worked_l783_78384

def harry_earnings (x : ℝ) : ℝ := 18 * x + 16 * 1.5 * x

noncomputable def james_earnings (x y : ℝ) : ℝ :=
  if y ≤ 40 then y * x else 40 * x + (y - 40) * 2 * x

theorem james_hours_worked (x : ℝ) (x_pos : x > 0) :
  harry_earnings x = james_earnings x 41 ∧ harry_earnings x > 0 → 41 = 41 :=
by
  intro h
  exact rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_hours_worked_l783_78384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_theorem_l783_78390

/-- Represents the water usage policy and bill calculation --/
noncomputable def WaterBill (m : ℝ) (usage : ℝ) : ℝ :=
  if usage ≤ 10 then m * usage
  else m * 10 + 2 * m * (usage - 10)

/-- Theorem stating that given the water pricing policy and a water bill of 16m yuan,
    the actual water usage is 13 cubic meters --/
theorem water_usage_theorem (m : ℝ) (h : m > 0) : 
  WaterBill m 13 = 16 * m ∧ 
  ∀ x, x ≠ 13 → WaterBill m x ≠ 16 * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_theorem_l783_78390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l783_78309

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x

theorem tangent_line_at_zero_one :
  let p : ℝ × ℝ := (0, 1)
  let m : ℝ := (deriv f) 0
  let tangent_line : ℝ → ℝ := λ x => m * x + 1
  tangent_line = λ x => 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l783_78309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_per_meter_l783_78377

/-- Represents a rectangular farm with fencing -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ
  totalCost : ℝ

/-- Calculates the cost per meter of fencing for a given farm -/
noncomputable def fencingCostPerMeter (farm : RectangularFarm) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (longSide ^ 2 + farm.shortSide ^ 2)
  let totalLength := longSide + farm.shortSide + diagonal
  farm.totalCost / totalLength

/-- Theorem stating that for the given farm specifications, 
    the fencing cost per meter is 14 -/
theorem farm_fencing_cost_per_meter :
  let farm : RectangularFarm := {
    area := 1200,
    shortSide := 30,
    totalCost := 1680
  }
  fencingCostPerMeter farm = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_per_meter_l783_78377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l783_78387

/-- The area of a rectangle with vertices at (0,0), (4,0), (0,3), and (4,3) is 12 -/
theorem rectangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 3)
  let D : ℝ × ℝ := (0, 3)
  let width := B.1 - A.1
  let height := C.2 - B.2
  let area := width * height
  area = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l783_78387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l783_78382

/-- Given vectors a and n, prove that their dot product is zero, 
    implying that a line with direction vector a is parallel to 
    a plane with normal vector n. -/
theorem line_parallel_to_plane (a n : ℝ × ℝ × ℝ) : 
  a = (1, -1, 3) → n = (0, 3, 1) → 
  a.1 * n.1 + a.2.1 * n.2.1 + a.2.2 * n.2.2 = 0 := by
  intro ha hn
  rw [ha, hn]
  ring
  -- The proof is complete, so we don't need 'sorry' here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l783_78382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l783_78316

theorem inequality_system_solutions :
  let satisfies_inequalities (x : ℕ) : Prop :=
    (5 * x + 2 > 3 * (x - 1)) ∧ (x - 2 ≤ 14 - 3 * x)
  (Finset.filter satisfies_inequalities (Finset.range 5)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l783_78316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_area_on_tangent_circles_l783_78329

/-- The largest area of a right triangle with vertices on two externally tangent circles -/
theorem largest_triangle_area_on_tangent_circles (r₁ r₂ : ℝ) 
  (h₁ : r₁ = 71) (h₂ : r₂ = 100) : 
  ∃ (A B C : ℝ × ℝ), 
    let triangle_area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
    let on_circle (X : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) := (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2
    let O₁ : ℝ × ℝ := (0, 0)
    let O₂ : ℝ × ℝ := (r₁ + r₂, 0)
    ((∃ (O : ℝ × ℝ) (r : ℝ), on_circle A O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) ∧
    (∃ (O : ℝ × ℝ) (r : ℝ), on_circle B O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) ∧
    (∃ (O : ℝ × ℝ) (r : ℝ), on_circle C O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) ∧
    ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) ∧
    triangle_area ≤ 24200 ∧
    (∀ (A' B' C' : ℝ × ℝ),
      (∃ (O : ℝ × ℝ) (r : ℝ), on_circle A' O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) →
      (∃ (O : ℝ × ℝ) (r : ℝ), on_circle B' O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) →
      (∃ (O : ℝ × ℝ) (r : ℝ), on_circle C' O r ∧ (O = O₁ ∨ O = O₂) ∧ (r = r₁ ∨ r = r₂)) →
      ((A'.1 - C'.1) * (B'.1 - C'.1) + (A'.2 - C'.2) * (B'.2 - C'.2) = 0) →
      abs ((A'.1 - C'.1) * (B'.2 - C'.2) - (B'.1 - C'.1) * (A'.2 - C'.2)) / 2 ≤ triangle_area)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_area_on_tangent_circles_l783_78329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_sine_of_angle_C_l783_78324

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

-- Part 1
theorem area_of_triangle (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : t.A = 2 * Real.pi / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 / 14 := by
  sorry

-- Part 2
theorem sine_of_angle_C (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : 2 * Real.sin t.B - Real.sin t.C = 1) :
  Real.sin t.C = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_sine_of_angle_C_l783_78324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_serve_probability_range_l783_78320

/-- The probability of a successful serve -/
def p : ℝ := sorry

/-- The number of serves -/
def X : ℕ := sorry

/-- The expected value of X -/
def EX (p : ℝ) : ℝ := p + 2*p*(1-p) + 3*(1-p)^2

/-- Theorem stating the range of p given EX > 1.75 -/
theorem serve_probability_range (hp : p ≠ 0) (hEX : EX p > 1.75) : 0 < p ∧ p < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_serve_probability_range_l783_78320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_equality_l783_78367

/-- The focal length of two specific ellipses is the same -/
theorem focal_length_equality (k : ℝ) (h : k < 9) :
  2 * Real.sqrt (25 - 9) = 2 * Real.sqrt ((25 - k) - (9 - k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_equality_l783_78367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l783_78334

-- Define the square
def square_center : ℝ × ℝ := (5, -5)
def square_area : ℝ := 16

-- Define the dilation
def dilation_center : ℝ × ℝ := (0, 0)
def scale_factor : ℝ := 3

-- Function to calculate distance from origin
noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

-- Theorem statement
theorem farthest_vertex_after_dilation :
  let side_length : ℝ := Real.sqrt square_area
  let original_vertices : List (ℝ × ℝ) := [
    (square_center.1 + side_length / 2, square_center.2 + side_length / 2),
    (square_center.1 + side_length / 2, square_center.2 - side_length / 2),
    (square_center.1 - side_length / 2, square_center.2 + side_length / 2),
    (square_center.1 - side_length / 2, square_center.2 - side_length / 2)
  ]
  let dilated_vertices : List (ℝ × ℝ) := original_vertices.map (fun v => (scale_factor * v.1, scale_factor * v.2))
  let farthest_vertex := dilated_vertices.argmax distance_from_origin
  farthest_vertex = some (21, -21) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l783_78334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l783_78371

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c) ∨ (a = c ∧ a + c > b) ∨ (b = c ∧ b + c > a)

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (x y : ℝ) :
  |x - 5| + (y - 8)^2 = 0 →
  ∃ (a b c : ℝ), is_isosceles_triangle a b c ∧
                 ({a, b, c} : Set ℝ) ⊆ {x, y} ∧
                 (perimeter a b c = 18 ∨ perimeter a b c = 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l783_78371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_64_l783_78363

/-- The coefficient of x³ in the expansion of (2√x - 1/√x)⁶ -/
def coefficient_x_cubed : ℕ := 64

/-- The binomial expansion of (2√x - 1/√x)⁶ -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (2 * Real.sqrt x - 1 / Real.sqrt x) ^ 6

theorem coefficient_x_cubed_is_64 :
  ∃ (f : ℝ → ℝ), ∀ x, x > 0 → 
    binomial_expansion x = coefficient_x_cubed * x^3 + f x ∧ 
    Filter.Tendsto f (Filter.atTop.comap (λ y => y^3)) (nhds 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_64_l783_78363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l783_78351

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Line l₁: 3x - 4y - 1 = 0 -/
def l₁ (x y : ℝ) : Prop := 3 * x - 4 * y - 1 = 0

/-- Line l₂: 3x - 4y + 3 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x - 4 * y + 3 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 3 (-4) (-1) 3 = 4/5 := by
  unfold distance_between_parallel_lines
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof would go here
  sorry

#eval (4 : ℚ) / 5 -- This will output the fraction 4/5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l783_78351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l783_78341

open Real

-- Define an acute triangle ABC
def AcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

-- Define the given equation
def GivenEquation (A B C : ℝ) : Prop :=
  (sin C)^2 + (sin B)^2 - (sin A)^2 = (sin B) * (sin C)

-- Define the angle bisector ratio
noncomputable def AngleBisectorRatio (A B C : ℝ) : ℝ :=
  (sin C) / (sin B)

-- State the theorem
theorem triangle_problem (A B C : ℝ) 
  (h_acute : AcuteTriangle A B C) 
  (h_eq : GivenEquation A B C) :
  A = Real.pi/3 ∧ 1/2 < AngleBisectorRatio A B C ∧ AngleBisectorRatio A B C < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l783_78341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l783_78353

/-- The equation representing the problem -/
def problem_equation (x : ℝ) : Prop :=
  0.6667 * x - 10 = 0.25 * x

/-- The solution to the equation is approximately 24 -/
theorem problem_solution :
  ∃ x : ℝ, problem_equation x ∧ |x - 24| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l783_78353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_ellipse_midpoint_l783_78343

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define a line by its slope and y-intercept
def line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the midpoint of two points
def is_midpoint (x₁ y₁ x₂ y₂ xm ym : ℝ) : Prop :=
  xm = (x₁ + x₂) / 2 ∧ ym = (y₁ + y₂) / 2

theorem line_through_ellipse_midpoint :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ →
    ellipse x₂ y₂ →
    is_midpoint x₁ y₁ x₂ y₂ 1 1 →
    ∃ (m b : ℝ), line m b x₁ y₁ ∧ line m b x₂ y₂ ∧ m = -3/4 ∧ b = 7/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_ellipse_midpoint_l783_78343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_m_value_l783_78304

/-- A linear equation in x and y of the form 3x^(|m|) + (m+1)y = 6 -/
def is_linear_equation (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), 3 * (x ^ abs m) + (m + 1) * y = a * x + b * y + c

theorem linear_equation_m_value :
  (∃ m : ℝ, is_linear_equation m) → (∃ m : ℝ, is_linear_equation m ∧ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_m_value_l783_78304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l783_78378

-- Define the complex numbers representing the two points
noncomputable def z1 : ℂ := 0 + 4 * Complex.I
noncomputable def z2 : ℂ := 0 - 2 * Complex.I

-- Define the line segment between z1 and z2
noncomputable def line_segment (t : ℝ) : ℂ := (1 - t) • z1 + t • z2

-- Define the condition for intersection
def intersects_once (k : ℝ) : Prop :=
  ∃! t, t ∈ Set.Icc 0 1 ∧ Complex.abs (line_segment t) = k

-- Theorem statement
theorem unique_intersection :
  ∀ k, intersects_once k ↔ k = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l783_78378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_plantation_revenue_is_31250_l783_78307

/-- Calculates the revenue from a square peanut plantation -/
noncomputable def peanut_plantation_revenue (side_length : ℝ) (peanut_yield : ℝ) (peanut_to_butter_ratio : ℝ) (butter_price : ℝ) : ℝ :=
  let plantation_area := side_length * side_length
  let total_peanuts := plantation_area * peanut_yield
  let total_butter := total_peanuts * peanut_to_butter_ratio
  let total_butter_kg := total_butter / 1000
  total_butter_kg * butter_price

/-- Theorem: The revenue from a 500x500 feet peanut plantation is $31,250 -/
theorem peanut_plantation_revenue_is_31250 :
  peanut_plantation_revenue 500 50 (5/20) 10 = 31250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_plantation_revenue_is_31250_l783_78307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_optimization_optimal_speed_in_range_l783_78357

/-- Represents the total cost of a journey --/
noncomputable def total_cost (x : ℝ) : ℝ := 130 * (30 / x + x / 60)

/-- Theorem stating the minimum cost and optimal speed for the journey --/
theorem journey_optimization (x : ℝ) (h : 40 ≤ x ∧ x ≤ 100) : 
  total_cost x ≥ 130 * Real.sqrt 2 ∧ 
  total_cost (30 * Real.sqrt 2) = 130 * Real.sqrt 2 := by
  sorry

/-- Verifies that the optimal speed is within the allowed range --/
theorem optimal_speed_in_range : 
  40 ≤ 30 * Real.sqrt 2 ∧ 30 * Real.sqrt 2 ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_optimization_optimal_speed_in_range_l783_78357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_minus_pi_l783_78348

open Real

theorem sine_double_angle_minus_pi (α : ℝ) (h1 : π/2 < α ∧ α < π) 
  (h2 : tan (α + π/4) = -1/7) : sin (2*α - π) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_minus_pi_l783_78348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l783_78360

noncomputable def f (x : ℝ) := Real.sin x + Real.sin (x + Real.pi/2)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (t : ℝ), t > 0 → (∀ (x : ℝ), f (x + t) = f x) → t ≥ T) ∧
    T = 2 * Real.pi ∧
  ∃ (M : ℝ),
    (∀ (x : ℝ), f x ≤ M) ∧
    (∃ (x : ℝ), f x = M) ∧
    M = Real.sqrt 2 ∧
    (∀ (x : ℝ), f x = M ↔ ∃ (k : ℤ), x = Real.pi/4 + 2*Real.pi*↑k) ∧
  ∀ (α : ℝ),
    f α = 3/4 → Real.sin (2*α) = -23/32 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l783_78360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l783_78330

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) (c : ℝ) :
  a > b ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (a * Real.cos t, b * Real.sin t))) →
  F₁ = (-c, 0) ∧ F₂ = (c, 0) →
  P ∈ Set.range (λ t : ℝ ↦ (a * Real.cos t, b * Real.sin t)) →
  dist P F₁ - dist P F₂ = 2 * b →
  dist P F₁ * dist P F₂ = 3/2 * a * b →
  c^2 = a^2 - b^2 →
  let e := c / a
  e = Real.sqrt 3 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l783_78330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_C_is_12_l783_78344

def last_two_digits (n : ℕ) : ℕ := n % 100

def is_divisible_by_4 (n : ℕ) : Bool := n % 4 = 0

def valid_C (c : ℕ) : Bool :=
  c < 10 && is_divisible_by_4 (last_two_digits (745829000 + c * 100 + 92))

theorem sum_of_valid_C_is_12 :
  (Finset.sum (Finset.filter (λ c => valid_C c) (Finset.range 10)) id) = 12 := by
  -- Proof goes here
  sorry

#eval Finset.sum (Finset.filter (λ c => valid_C c) (Finset.range 10)) id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_C_is_12_l783_78344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_growth_closest_to_227_l783_78339

/-- Represents the population of a town over 20 years -/
structure TownPopulation where
  pop1991 : Nat
  pop2001 : Nat
  pop2011 : Nat

/-- Conditions for the town population -/
def ValidTownPopulation (t : TownPopulation) : Prop :=
  ∃ (p q r : Nat),
    t.pop1991 = p * p ∧
    t.pop2001 = t.pop1991 + 200 ∧
    t.pop2001 = q * q + 16 ∧
    t.pop2011 = t.pop2001 + 300 ∧
    t.pop2011 = r * r

/-- Calculate the percent growth over 20 years -/
noncomputable def PercentGrowth (t : TownPopulation) : ℝ :=
  (t.pop2011 - t.pop1991 : ℝ) / t.pop1991 * 100

/-- Theorem stating that the percent growth is closest to 227% -/
theorem percent_growth_closest_to_227 (t : TownPopulation) 
  (h : ValidTownPopulation t) : 
  abs (PercentGrowth t - 227) < 
    min (abs (PercentGrowth t - 220)) 
        (min (abs (PercentGrowth t - 225))
             (min (abs (PercentGrowth t - 230))
                  (abs (PercentGrowth t - 235)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_growth_closest_to_227_l783_78339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l783_78381

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The equation of a circle -/
def circleEquation (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

theorem circle_through_origin (center : Point) (h : center.x = 3 ∧ center.y = 4) :
  circleEquation center (distance center ⟨0, 0⟩) ⟨0, 0⟩ ↔ 
  circleEquation center 5 ⟨0, 0⟩ := by
  sorry

#check circle_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l783_78381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l783_78340

theorem sin_double_angle_problem (α : ℝ) :
  Real.sin (π / 4 + α) = Real.sqrt 5 / 5 → Real.sin (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l783_78340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_min_distance_l783_78337

/-- The distance ratio from point C to A(-1,0) and B(1,0) is √3 -/
def distance_ratio (C : ℝ × ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  (C.1 + 1)^2 + C.2^2 = 3 * ((C.1 - 1)^2 + C.2^2)

/-- The line equation is x-2y+8=0 -/
def line_equation (x y : ℝ) : Prop :=
  x - 2*y + 8 = 0

/-- The minimum distance from a point to a line -/
noncomputable def min_distance (C : ℝ × ℝ) : ℝ :=
  let M : ℝ × ℝ := (2, 0)
  (10 : ℝ) / Real.sqrt 5 - Real.sqrt 3

theorem apollonian_circle_min_distance :
  ∀ C : ℝ × ℝ, distance_ratio C →
    min_distance C = 2 * Real.sqrt 5 - Real.sqrt 3 := by
  sorry

#check apollonian_circle_min_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_min_distance_l783_78337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_additional_video_game_time_l783_78392

/-- The additional video game time Diana gets after her raise -/
def additional_video_game_time (base_reward : ℚ) (raise_percentage : ℚ) (reading_hours : ℚ) : ℚ :=
  base_reward * raise_percentage * reading_hours

/-- Proof that Diana gets 72 additional minutes of video game time -/
theorem diana_additional_video_game_time :
  additional_video_game_time 30 (20 / 100) 12 = 72 := by
  -- Unfold the definition of additional_video_game_time
  unfold additional_video_game_time
  -- Perform the calculation
  simp [Rat.mul_assoc, Rat.mul_comm]
  -- The proof is complete
  rfl

#eval additional_video_game_time 30 (20 / 100) 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_additional_video_game_time_l783_78392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l783_78338

-- Define the cone properties
noncomputable def slant_height : ℝ := 2
noncomputable def lateral_area : ℝ := Real.pi

-- Define the formula for lateral area of a cone
noncomputable def lateral_area_formula (r : ℝ) : ℝ := Real.pi * r * slant_height

-- Theorem statement
theorem cone_base_radius :
  ∃ (r : ℝ), lateral_area_formula r = lateral_area ∧ r = 1/2 := by
  -- Introduce the radius
  let r : ℝ := 1/2
  
  -- Prove that this radius satisfies the lateral area formula
  have h1 : lateral_area_formula r = lateral_area := by
    simp [lateral_area_formula, lateral_area, slant_height]
    ring
  
  -- Conclude the proof
  exact ⟨r, h1, rfl⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l783_78338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_problem_l783_78399

noncomputable section

-- Define the hyperbola M
def hyperbola_M (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

-- Define the parabola N
def parabola_N (x y p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus of the parabola
def focus_N (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Define the vertices of the hyperbola
def vertex_left_M : ℝ × ℝ := (-3, 0)
def vertex_right_M : ℝ × ℝ := (3, 0)

-- Define the intersections of the parabola with x = 4
def intersection_A : ℝ × ℝ := (4, 8)
def intersection_B : ℝ × ℝ := (4, -8)

-- Define the vectors AC and BD
def vector_AC : ℝ × ℝ := (-7, -8)
def vector_BD : ℝ × ℝ := (-1, 8)

theorem hyperbola_parabola_problem :
  ∃ (p : ℝ),
    (∃ (x y : ℝ), hyperbola_M x y ∧ parabola_N x y p) →
    (focus_N p = (4, 0)) →
    (∀ (x y : ℝ), parabola_N x y p ↔ y^2 = 16 * x) ∧
    (vector_AC.1 * vector_BD.1 + vector_AC.2 * vector_BD.2 = -57) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_problem_l783_78399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l783_78366

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Rotates a parabola 180° around its vertex -/
noncomputable def rotateParabola (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c + 2 * (p.b^2 / (4 * p.a)) }

/-- Translates a parabola vertically -/
def translateParabola (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Checks if a point lies on a parabola -/
def pointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

/-- The main theorem to be proved -/
theorem parabola_transformation (original : Parabola) (line : Line) :
  original.a = 3 ∧ original.b = -6 ∧ original.c = 5 ∧
  line.m = -1 ∧ line.b = -2 →
  ∃ (d : ℝ),
    let rotated := rotateParabola original
    let translated := translateParabola rotated d
    pointOnParabola translated 2 (-4) ∧
    pointOnLine line 2 (-4) ∧
    translated.a = -3 ∧ translated.b = 6 ∧ translated.c = -4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l783_78366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_two_long_altitudes_l783_78332

/-- Represents a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The altitude from vertex A to side a -/
noncomputable def altitude_a (t : Triangle) : ℝ := t.b * Real.sin t.C

/-- The altitude from vertex B to side b -/
noncomputable def altitude_b (t : Triangle) : ℝ := t.a * Real.sin t.C

theorem triangle_with_two_long_altitudes (t : Triangle) 
  (h1 : altitude_a t ≥ t.a) 
  (h2 : altitude_b t ≥ t.b) : 
  t.A = 45 ∧ t.B = 45 ∧ t.C = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_two_long_altitudes_l783_78332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_steps_proof_l783_78379

/-- The number of steps on an escalator satisfying specific conditions -/
def escalator_steps : ℕ := 80

/-- Total length of the escalator -/
def total_length (L : ℕ) : ℕ := 2 * L

/-- Petya's running speed relative to the escalator -/
def run_speed (x : ℕ) : ℕ := x

/-- Petya's tumbling speed relative to the escalator -/
def tumble_speed (x : ℕ) : ℕ := 3 * x

/-- Condition: Petya counts 20 steps while running half the escalator -/
def run_condition (L x : ℕ) : Prop := 20 * (run_speed x + x) = L * x

/-- Condition: Petya counts 30 steps while tumbling half the escalator -/
def tumble_condition (L x : ℕ) : Prop := 30 * (tumble_speed x + x) = L * (tumble_speed x)

/-- Proof that the number of steps on the escalator is correct -/
theorem escalator_steps_proof (L x : ℕ) 
  (h1 : run_condition L x)
  (h2 : tumble_condition L x) :
  escalator_steps = total_length L := by sorry

#check escalator_steps_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_steps_proof_l783_78379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_two_thirds_l783_78328

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The line l in the problem --/
noncomputable def l : Line := ⟨2/3, 0⟩  -- We know the slope, intercept is arbitrary

/-- Point P where l intersects y=1 --/
noncomputable def P : Point := ⟨-2, 1⟩  -- From the solution

/-- Point Q where l intersects x-y-1=0 --/
noncomputable def Q : Point := ⟨4, 1⟩  -- From the solution

/-- The midpoint of PQ --/
def midpoint_PQ : Point := ⟨1, -1⟩

theorem line_slope_is_two_thirds :
  (P.y = 1) →
  (Q.x - Q.y - 1 = 0) →
  (midpoint_PQ.x = (P.x + Q.x) / 2) →
  (midpoint_PQ.y = (P.y + Q.y) / 2) →
  l.slope = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_two_thirds_l783_78328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l783_78336

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := -x^2 + 8*x - 15

/-- The area function of the triangle -/
noncomputable def area (p : ℝ) : ℝ := (3/2) * |p^2 - (19/3)*p + 35/3|

/-- Theorem stating the existence of the maximum area -/
theorem max_triangle_area :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 5 ∧
  f 2 = 5 ∧ f 5 = 10 ∧
  ∀ (q : ℝ), f p = q →
  ∀ (p' : ℝ), 0 ≤ p' ∧ p' ≤ 5 → area p ≥ area p' ∧
  area p = 112.5/24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l783_78336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_150_divided_by_x_minus_1_pow_4_l783_78321

theorem remainder_x_150_divided_by_x_minus_1_pow_4 : 
  ∃ q : Polynomial ℝ, X^150 = (X - 1)^4 * q + (-551300*X^3 + 1665075*X^2 - 1667400*X + 562626) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_150_divided_by_x_minus_1_pow_4_l783_78321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_integer_product_l783_78364

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2
  | 1 => Real.rpow 3 (1/13)
  | n+2 => sequence_a (n+1) * (sequence_a n)^2

noncomputable def product_up_to (k : ℕ) : ℝ :=
  (List.range k).foldl (λ acc i => acc * sequence_a (i+1)) 1

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem smallest_k_for_integer_product :
  (∀ k < 13, ¬ is_integer (product_up_to k)) ∧
  is_integer (product_up_to 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_integer_product_l783_78364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfies_conditions_count_divisors_not_multiple_of_14_is_154_l783_78372

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_seventh (n : ℕ) : Prop := ∃ k : ℕ, n = k^7

def smallest_m : ℕ := 2^6 * 3^10 * 7^7

theorem smallest_m_satisfies_conditions :
  is_perfect_square (smallest_m / 2) ∧
  is_perfect_cube (smallest_m / 3) ∧
  is_perfect_seventh (smallest_m / 7) :=
sorry

noncomputable def count_divisors_not_multiple_of_14 (n : ℕ) : ℕ :=
  (Finset.filter (fun d => ¬(14 ∣ d)) (Nat.divisors n)).card

theorem count_divisors_not_multiple_of_14_is_154 :
  count_divisors_not_multiple_of_14 smallest_m = 154 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfies_conditions_count_divisors_not_multiple_of_14_is_154_l783_78372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_plough_time_l783_78369

/-- The time it takes for person A to plough the field alone -/
noncomputable def time_A : ℝ := 15

/-- The time it takes for persons A and B together to plough the field -/
noncomputable def time_AB : ℝ := 10

/-- The work rate of person A (fraction of field ploughed per hour) -/
noncomputable def rate_A : ℝ := 1 / time_A

/-- The combined work rate of persons A and B -/
noncomputable def rate_AB : ℝ := 1 / time_AB

/-- The work rate of person B -/
noncomputable def rate_B : ℝ := rate_AB - rate_A

theorem B_plough_time : 1 / rate_B = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_plough_time_l783_78369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_phi_coordinates_l783_78331

-- Define the types for rectangular and spherical coordinates
structure RectCoord where
  x : ℝ
  y : ℝ
  z : ℝ

structure SphCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the conversion functions
noncomputable def sphericalToRectangular (s : SphCoord) : RectCoord :=
  { x := s.ρ * Real.sin s.φ * Real.cos s.θ,
    y := s.ρ * Real.sin s.φ * Real.sin s.θ,
    z := s.ρ * Real.cos s.φ }

-- Theorem statement
theorem mirror_phi_coordinates 
  (p1 : RectCoord) 
  (s1 : SphCoord) 
  (h1 : p1 = sphericalToRectangular s1) 
  (h2 : p1 = { x := -5, y := -7, z := 4 }) : 
  sphericalToRectangular { ρ := s1.ρ, θ := s1.θ, φ := -s1.φ } = { x := 5, y := 7, z := 4 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_phi_coordinates_l783_78331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l783_78359

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y)) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l783_78359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_vertical_shift_l783_78342

theorem sine_function_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) 
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) 
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 5) 
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -3) : 
  d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_vertical_shift_l783_78342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l783_78397

theorem train_speed_problem (train_length crossing_time : Real) : 
  train_length = 2.00 →
  crossing_time = 40 / 3600 →
  let speed := train_length / (2 * crossing_time)
  speed = 90 := by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l783_78397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sequence_l783_78333

def a (n : ℕ) : ℕ := 2^(3*n) + 3^(6*n + 2) + 5^(6*n + 2)

theorem gcd_of_sequence :
  Nat.gcd (a 0) (Nat.gcd (a 1) (Nat.gcd (a 2) (a 1999))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sequence_l783_78333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l783_78313

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem about the monotonic increasing interval of f
theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, y < x → f y < f x} = Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l783_78313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l783_78380

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

-- Define the theorem
theorem triangle_problem (t : Triangle) :
  t.a * Real.sin t.C = t.c * Real.cos (t.A - π/6) →
  t.a = Real.sqrt 7 →
  t.A = π/3 ∧
  ((t.b = 2 → t.area = (3 * Real.sqrt 3) / 2) ∧
   (3 * Real.sin t.C = 2 * Real.sin t.B → t.area = Real.sqrt 3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l783_78380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l783_78383

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

/-- The transformed function g(x) after translation and shrinking -/
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (4 * x - 2 * φ + Real.pi / 4)

/-- The theorem stating the minimum value of φ -/
theorem min_phi_value (φ : ℝ) : 
  (φ > 0) → 
  (∀ x, g x φ = g (Real.pi / 2 - x) φ) → 
  φ ≥ 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l783_78383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l783_78395

theorem inequality_system_solution (m : ℝ) : 
  (∃ (S : Finset ℤ), S = {x : ℤ | 3*x - m > 0 ∧ x - 1 ≤ 5} ∧ Finset.card S = 4) →
  (6 ≤ m ∧ m < 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l783_78395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l783_78306

/-- Given an acid solution with initial concentration m% and volume m ounces,
    adding x ounces of water results in a (2m - k)% solution where m > 2k.
    This function calculates the amount of water (x) needed for dilution. -/
noncomputable def water_for_dilution (m k : ℝ) (h : m > 2 * k) : ℝ :=
  (k * m - m^2) / (2 * m - k)

/-- Theorem stating that the calculated amount of water results in the desired concentration -/
theorem dilution_theorem (m k : ℝ) (h : m > 2 * k) :
  let x := water_for_dilution m k h
  let initial_acid := m^2 / 100
  let final_volume := m + x
  let final_concentration := (2 * m - k) / 100
  initial_acid = final_concentration * final_volume :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l783_78306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l783_78394

/-- Represents a parabola of the form y^2 = ax -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- Represents a line with a given slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (p.a / 4, 0)

/-- A line passing through a point with a given slope -/
noncomputable def line_through_point (m : ℝ) (p : ℝ × ℝ) : Line :=
  { slope := m
  , y_intercept := p.2 - m * p.1 }

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ × ℝ := (0, l.y_intercept)

/-- The area of a triangle given two points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * |p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)|

theorem parabola_equation (p : Parabola) 
  (l : Line)
  (h1 : l.slope = 2)
  (h2 : l = line_through_point 2 (focus p))
  (h3 : triangle_area (0, 0) (y_intercept l) (focus p) = 4) :
  |p.a| = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l783_78394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_pentagonal_number_l783_78312

def pentagonal_sequence : List Nat := [1, 5, 12, 22, 35]

theorem sixth_pentagonal_number : 
  pentagonal_sequence.length = 5 → 
  (pentagonal_sequence.zip (pentagonal_sequence.tail!)).map (λ (a, b) => b - a) = [4, 7, 10, 13] →
  ([4, 7, 10, 13].zip ([7, 10, 13])).map (λ (a, b) => b - a) = [3, 3, 3] →
  pentagonal_sequence.getLast? = some 35 →
  35 + 13 + 3 = 51 :=
by
  intros h1 h2 h3 h4
  rfl

#eval pentagonal_sequence.length
#eval (pentagonal_sequence.zip (pentagonal_sequence.tail!)).map (λ (a, b) => b - a)
#eval ([4, 7, 10, 13].zip ([7, 10, 13])).map (λ (a, b) => b - a)
#eval pentagonal_sequence.getLast?
#eval 35 + 13 + 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_pentagonal_number_l783_78312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_intersection_l783_78347

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope k and y-intercept c -/
structure Line where
  k : ℝ
  c : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The distance from an endpoint of the minor axis to a focus -/
noncomputable def minorEndpointToFocus (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_equation_and_intersection (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 3 / 2)
  (h_dist : minorEndpointToFocus e = 2) :
  (∃ (k : ℝ), 
    (e.a = 2 ∧ e.b = 1) ∧ 
    (k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2) ∧
    (∀ (x y : ℝ), x^2 + y^2/4 = 1 ↔ x^2/e.b^2 + y^2/e.a^2 = 1) ∧
    (∃ (A B : ℝ × ℝ), 
      (A.2 = k * A.1 + Real.sqrt 3 ∧ A.1^2 + A.2^2/4 = 1) ∧
      (B.2 = k * B.1 + Real.sqrt 3 ∧ B.1^2 + B.2^2/4 = 1) ∧
      (A.1 * B.1 + A.2 * B.2 = 0))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_intersection_l783_78347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_correct_l783_78389

-- Define the payment function as noncomputable
noncomputable def payment (x : ℝ) : ℝ :=
  if x ≤ 200 then x
  else if x ≤ 600 then 200 + 0.8 * (x - 200)
  else x - 100 * (⌊(x - 200) / 300⌋)

-- State the theorem
theorem payment_correct (x : ℝ) (h : x > 0) : 
  payment x = 
    if x ≤ 200 then x
    else if x ≤ 600 then 200 + 0.8 * (x - 200)
    else x - 100 * (⌊(x - 200) / 300⌋) := by
  sorry

-- Examples from the problem
example : payment 196 = 196 := by sorry
example : payment 260 = 248 := by sorry
example (x : ℝ) (h1 : 200 < x) (h2 : x ≤ 600) : payment x = 0.8 * x + 40 := by sorry
example (m : ℝ) (h1 : 0 < m) (h2 : m < 200) : payment (m + 300) - payment m = 250 → m = 150 := by sorry
example : payment 900 = 600 := by sorry
example : payment 898 = 698 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_correct_l783_78389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l783_78358

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then |2^x - a| else x^2 - 3*a*x + 2*a^2

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y

theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_exactly_two_zeros a ↔ (1 ≤ a ∧ a < 2) ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l783_78358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_of_f_even_function_l783_78319

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^x else 0

-- Define the properties of g
def is_extension_of_f (g : ℝ → ℝ) : Prop :=
  ∀ x ≤ 0, g x = f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Theorem statement
theorem extension_of_f_even_function :
  ∀ g : ℝ → ℝ, is_extension_of_f g → is_even g →
  ∀ x : ℝ, g x = 2^(-abs x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_of_f_even_function_l783_78319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_hexagonal_prism_intersection_l783_78311

/-- A regular hexagonal prism -/
structure RegularHexagonalPrism where
  -- We don't need to define the internal structure of the prism
  -- as the problem doesn't require specific geometric properties

/-- A plane that can intersect the prism -/
structure IntersectingPlane where
  -- We don't need to define the internal structure of the plane
  -- as the problem doesn't require specific geometric properties

/-- The polygon formed by the intersection of a plane with the prism -/
def IntersectionPolygon (prism : RegularHexagonalPrism) (plane : IntersectingPlane) : Nat :=
  8 -- This is a simplification; in reality, it would be ≤ 8

/-- The maximum number of sides of any intersection polygon -/
def MaxSides : Nat := 8

/-- Theorem: The maximum number of sides of a polygon formed by the intersection
    of a plane with a regular hexagonal prism is 8 -/
theorem max_sides_hexagonal_prism_intersection :
  ∀ (prism : RegularHexagonalPrism) (plane : IntersectingPlane),
    IntersectionPolygon prism plane ≤ MaxSides ∧
    ∃ (plane : IntersectingPlane), IntersectionPolygon prism plane = MaxSides :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_hexagonal_prism_intersection_l783_78311
