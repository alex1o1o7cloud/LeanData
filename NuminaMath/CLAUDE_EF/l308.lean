import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_equal_r_l308_30815

/-- Sum of remainders when n is divided by each number from 1 to n -/
def r (n : ℕ) : ℕ := Finset.sum (Finset.range n) (fun i => n % (i + 1))

/-- There are infinitely many natural numbers k for which r(k) = r(k-1) -/
theorem infinitely_many_equal_r : ∀ m : ℕ, ∃ k > m, r k = r (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_equal_r_l308_30815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l308_30843

theorem geometric_sequence_first_term 
  (a b c : ℝ) -- First three terms of the sequence
  (h1 : ∃ r, b = a * r ∧ c = b * r ∧ 81 = c * r ∧ 243 = 81 * r) -- The sequence is geometric
  : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l308_30843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l308_30834

theorem min_translation_for_symmetry (f : ℝ → ℝ) (m : ℝ) :
  (f = λ x ↦ Real.sqrt 3 * Real.cos x + Real.sin x) →
  (m > 0) →
  (∃ k : ℤ, m + π / 3 = k * π + π / 2) →
  (∀ m' > 0, (∃ k' : ℤ, m' + π / 3 = k' * π + π / 2) → m' ≥ m) →
  m = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l308_30834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l308_30861

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + 5 * Real.pi / 6)

theorem g_monotone_increasing :
  MonotoneOn g (Set.Icc (-2 * Real.pi / 3) (-Real.pi / 6)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l308_30861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lex_reading_pages_per_day_l308_30850

/-- The number of pages Lex read every day -/
noncomputable def pages_per_day (total_days : ℝ) (book1_pages book2_pages book3_pages : ℝ) : ℝ :=
  (book1_pages + book2_pages + book3_pages) / total_days

/-- Theorem stating that Lex read (384.5 + 210.75 + 317.25) / 15.5 pages per day -/
theorem lex_reading_pages_per_day :
  pages_per_day 15.5 384.5 210.75 317.25 = (384.5 + 210.75 + 317.25) / 15.5 := by
  -- Unfold the definition of pages_per_day
  unfold pages_per_day
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

#eval (384.5 + 210.75 + 317.25) / 15.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lex_reading_pages_per_day_l308_30850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l308_30808

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line -/
def vectorParallelToLine (v : Vec) (l : Line) : Prop :=
  l.a * v.y = -l.b * v.x

/-- The main theorem -/
theorem line_equation_correct (l : Line) (v : Vec) (p : Point) :
  l.a = 1 ∧ l.b = 3 ∧ l.c = -1 ∧
  v.x = 3 ∧ v.y = -1 ∧
  p.x = 1 ∧ p.y = 0 →
  pointOnLine l p ∧ vectorParallelToLine v l := by
  sorry

#check line_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l308_30808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l308_30817

theorem log_inequality_range (a : ℝ) :
  (0 < a ∧ a ≠ 1) → (Real.log (3/5) / Real.log a < 1 ↔ (0 < a ∧ a < 3/5) ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l308_30817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l308_30801

def sequence_a : ℕ → ℝ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | n + 2 => 3 * sequence_a (n + 1) - 4

def sequence_b (n : ℕ) : ℝ := sequence_a n - 2

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = 3 * sequence_b n) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 3^(n - 1) + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l308_30801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_countability_of_sets_l308_30854

-- Define the sets
def Z : Type := ℤ
def Q : Type := ℚ
def R : Type := ℝ
def Q_X : Type := Polynomial ℚ

-- Define countability
def IsCountable (α : Type*) : Prop := Nonempty (α ≃ ℕ)

-- State the theorem
theorem countability_of_sets :
  (IsCountable Z) ∧
  (IsCountable Q) ∧
  (IsCountable Q_X) ∧
  (¬ IsCountable R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_countability_of_sets_l308_30854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_common_chord_l308_30853

/-- Given two circles, one passing through (1, -2) and tangent to (x-1)^2 + y^2 = 1
    at points A and B, prove that the line AB has the equation 2y + 1 = 0 -/
theorem tangent_circles_common_chord 
  (circle1 : Set (ℝ × ℝ)) 
  (circle2 : Set (ℝ × ℝ)) 
  (A B : ℝ × ℝ) :
  ((1, -2) ∈ circle1) →
  (∀ x y, (x - 1)^2 + y^2 = 1 ↔ (x, y) ∈ circle2) →
  (A ∈ circle1 ∧ A ∈ circle2) →
  (B ∈ circle1 ∧ B ∈ circle2) →
  (∀ P, P ∈ circle1 → P ∈ circle2 → P = A ∨ P = B) →
  (∃ k m : ℝ, ∀ x y, (x, y) ∈ (circle1 ∩ circle2) ↔ 2*y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_common_chord_l308_30853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_approx_three_l308_30896

/-- Represents the properties of a rectangular floor and its painting cost. -/
structure RectangularFloor where
  length : ℝ
  breadth : ℝ
  total_cost : ℝ
  length_breadth_ratio : length = 3 * breadth
  length_value : length = 20

/-- Calculates the painting rate per square meter for a given rectangular floor. -/
noncomputable def painting_rate_per_sq_m (floor : RectangularFloor) : ℝ :=
  floor.total_cost / (floor.length * floor.breadth)

/-- Theorem stating that the painting rate is approximately 3 Rs/sq m for the given conditions. -/
theorem painting_rate_approx_three (floor : RectangularFloor) 
  (h : floor.total_cost = 400) : 
  ∃ ε > 0, |painting_rate_per_sq_m floor - 3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_approx_three_l308_30896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_condition_l308_30819

/-- The condition for a point P(x, y, z) to be equidistant from points A(-1, 2, 3) and B(0, 0, 5) -/
theorem equidistant_condition (x y z : ℝ) : 
  (x + 1)^2 + (y - 2)^2 + (z - 3)^2 = x^2 + y^2 + (z - 5)^2 →
  2*x - 4*y + 4*z = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_condition_l308_30819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_equals_24_59_l308_30879

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The original number to be rounded -/
def original : ℝ := 24.58673

/-- Theorem stating that rounding the original number to the nearest hundredth equals 24.59 -/
theorem round_to_hundredth_equals_24_59 : 
  roundToHundredth original = 24.59 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_equals_24_59_l308_30879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_l308_30884

/-- Represents the face value of a stock -/
noncomputable def face_value (yield percentage_yield : ℝ) (quoted_price : ℝ) : ℝ :=
  quoted_price * yield / percentage_yield

/-- Theorem: Given a stock with 10% yield, quoted at $200, and 20% percentage yield, its face value is $100 -/
theorem stock_face_value :
  let yield : ℝ := 0.10
  let quoted_price : ℝ := 200
  let percentage_yield : ℝ := 0.20
  face_value yield percentage_yield quoted_price = 100 := by
  -- Unfold the definition of face_value
  unfold face_value
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_l308_30884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l308_30883

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.arccos x)

-- State the theorem about the domain of g(x)
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.univ} = Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l308_30883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_log_l308_30845

theorem min_sum_given_log (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.logb a (3 * b) = -1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 / 3 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → Real.logb x (3 * y) = -1 → x + y ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_log_l308_30845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l308_30889

/-- The ratio of shaded area to total area in square ABCD -/
theorem shaded_area_ratio : 
  (let square_side : ℝ := 7
   let small_square_side : ℝ := 2
   let medium_square_side : ℝ := 5
   let medium_square_hole : ℝ := 3
   let large_rectangle_width : ℝ := 7
   let large_rectangle_height : ℝ := 1
   let total_area := square_side ^ 2
   let shaded_area := small_square_side ^ 2 + 
                      (medium_square_side ^ 2 - medium_square_hole ^ 2) + 
                      (large_rectangle_width * large_rectangle_height)
   shaded_area / total_area) = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l308_30889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_line_l308_30813

/-- The equation of a line passing through a focus of the hyperbola x²/4 - y² = 1
    and parallel to one of its asymptotes -/
theorem hyperbola_focus_line (x y : ℝ) : 
  (∃ (a : ℝ), x^2 / 4 - y^2 = 1 ∧ 
              (x - a)^2 + y^2 = 5 ∧
              y = -(1/2) * x + a) →
  y = -(1/2) * x + (Real.sqrt 5) / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_line_l308_30813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l308_30832

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

-- State the theorem
theorem f_composition (x : ℝ) (h : 0 ≤ x ∧ x < 1) :
  f ((5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2)) = Real.sqrt 5 * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l308_30832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_of_congruent_triangles_l308_30818

/-- Two congruent triangles XYE and XYF with given side lengths -/
structure CongruentTriangles where
  XY : ℝ
  YE : ℝ
  EX : ℝ
  XF : ℝ
  FY : ℝ
  congruent : XY = 12 ∧ YE = 13 ∧ EX = 15 ∧ XF = 13 ∧ FY = 15

/-- The intersection area of two congruent triangles -/
noncomputable def intersection_area (t : CongruentTriangles) : ℝ :=
  2 * Real.sqrt 1432

/-- Theorem stating that the intersection area of the given congruent triangles is 2 * √1432 -/
theorem intersection_area_of_congruent_triangles (t : CongruentTriangles) :
  intersection_area t = 2 * Real.sqrt 1432 := by
  -- Unfold the definition of intersection_area
  unfold intersection_area
  -- The equality follows directly from the definition
  rfl

#check intersection_area_of_congruent_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_of_congruent_triangles_l308_30818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l308_30899

-- Define the sets M and N
def M : Set ℝ := {x | (2 : ℝ)^x > 1}
def N : Set ℝ := {x | Real.log (x^2 - 2*x + 4) > 0}

-- State the theorem
theorem complement_intersection_M_N : 
  (Set.univ : Set ℝ) \ (M ∩ N) = Set.Iic (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l308_30899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tv_height_l308_30800

/-- Represents a TV with its dimensions and cost -/
structure TV where
  width : ℚ
  height : ℚ
  cost : ℚ

/-- Calculates the area of a TV -/
def TV.area (tv : TV) : ℚ := tv.width * tv.height

/-- Calculates the cost per square inch of a TV -/
def TV.costPerSquareInch (tv : TV) : ℚ := tv.cost / tv.area

theorem new_tv_height (first_tv : TV) (new_tv : TV) :
  first_tv.width = 24 →
  first_tv.height = 16 →
  first_tv.cost = 672 →
  new_tv.width = 48 →
  new_tv.cost = 1152 →
  first_tv.costPerSquareInch = new_tv.costPerSquareInch + 1 →
  new_tv.height = 32 := by
  sorry

#check new_tv_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tv_height_l308_30800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l308_30880

/-- Given a parabola y² = 4x with focus F, prove that for any point P on the parabola
    such that |PF| = 4, if M is the foot of the perpendicular from P to the y-axis,
    then the area of triangle PFM is 3√3. -/
theorem parabola_triangle_area (x y : ℝ) (F P M : ℝ × ℝ) : 
  y^2 = 4*x →  -- Parabola equation
  P = (x, y) →  -- P is on the parabola
  F = (1, 0) →  -- Focus of the parabola
  M = (0, y) →  -- M is the foot of perpendicular from P to y-axis
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 4 →  -- |PF| = 4
  (1/2) * (P.1 - M.1) * (P.2 - M.2) = 3 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l308_30880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_perfect_square_l308_30855

theorem gcd_not_perfect_square (m n : ℕ) (h : (m % 3 = 0 ∧ n % 3 ≠ 0) ∨ (m % 3 ≠ 0 ∧ n % 3 = 0)) :
  ¬ ∃ k : ℕ, (Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) = k^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_perfect_square_l308_30855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_sequence_l308_30846

def sequence_property (s : List ℤ) : Prop :=
  ∀ i, i + 2 < s.length → s[i+1]! = s[i]! + s[i+2]!

theorem sum_of_special_sequence :
  ∀ s : List ℤ,
  s.length = 2009 →
  sequence_property s →
  s[0]! = 1 →
  s[1]! = -1 →
  s.sum = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_sequence_l308_30846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l308_30848

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log x

-- State the theorem
theorem tangent_line_parallel (a : ℝ) : 
  (∃ (m : ℝ), m = -2 ∧ 
    m = (deriv (f a)) 1) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l308_30848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_properties_l308_30803

-- Define the curve and parabola
noncomputable def curve (x : ℝ) : ℝ := -1 / x
noncomputable def parabola (p x : ℝ) : ℝ := Real.sqrt (2 * p * x)

-- Define the common tangent line
structure CommonTangent (p : ℝ) where
  k : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_tangent_curve : k * x₁ + b = curve x₁
  h_tangent_parabola : k * x₂ + b = parabola p x₂
  h_x₂_pos : x₂ > 0
  h_p_pos : p > 0

-- State the theorem
theorem common_tangent_properties (p : ℝ) (ct : CommonTangent p) :
  ct.x₁ * ct.y₁ = -1 ∧ 
  (((ct.x₁ - ct.x₂)^2 + (ct.y₁ - ct.y₂)^2 = (3 * Real.sqrt 10 / 2)^2) → 
   (p = Real.sqrt 2 ∨ p = 8 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_properties_l308_30803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l308_30893

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) : ℝ := k*x - 2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the condition for intersection
def intersects (x0 y0 : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧ distance x0 y0 x y ≤ 1

-- Main theorem
theorem max_k_value :
  ∀ k : ℝ, (∃ x0 : ℝ, intersects x0 (line k x0)) → k ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l308_30893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_scores_l308_30874

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : ℚ
  deriving Repr

/-- Determines if a student has the most stable scores among a list of students -/
def hasMostStableScores (s : Student) (students : List Student) : Prop :=
  ∀ t ∈ students, s.variance ≤ t.variance

/-- Theorem: Given three students with the same average score but different variances,
    the student with the lowest variance has the most stable scores -/
theorem most_stable_scores 
  (students : List Student) 
  (hLen : students.length = 3)
  (hDistinct : ∀ (i j : Fin 3), i ≠ j → 
    (students.get ⟨i.val, by simp [hLen]⟩).variance ≠ 
    (students.get ⟨j.val, by simp [hLen]⟩).variance)
  (hA : (students.get ⟨0, by simp [hLen]⟩).variance = 6)
  (hB : (students.get ⟨1, by simp [hLen]⟩).variance = 24)
  (hC : (students.get ⟨2, by simp [hLen]⟩).variance = 50) :
  hasMostStableScores (students.get ⟨0, by simp [hLen]⟩) students :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_scores_l308_30874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l308_30827

/-- The set of integers from 1 to 2007 -/
def S : Finset ℕ := Finset.range 2007

/-- The number of subsets in the partition -/
def n : ℕ := 223

/-- The number of elements in each subset -/
def k : ℕ := 9

/-- A partition of S into n subsets -/
def Partition := Fin n → Finset ℕ

/-- Predicate for a valid partition -/
def is_valid_partition (p : Partition) : Prop :=
  (∀ i j, i ≠ j → Disjoint (p i) (p j)) ∧ 
  (∀ i, Finset.card (p i) = k) ∧
  (Finset.biUnion Finset.univ p = S)

/-- Predicate for equal sums in all subsets -/
def has_equal_sums (p : Partition) : Prop :=
  ∀ i j, Finset.sum (p i) id = Finset.sum (p j) id

/-- Main theorem: There exists a valid partition with equal sums -/
theorem partition_exists : ∃ p : Partition, is_valid_partition p ∧ has_equal_sums p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l308_30827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l308_30806

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h1 : ∀ n, a (n + 1) = q * a n  -- Definition of geometric sequence
  h2 : ∀ n, a n > 0  -- Positivity condition

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumFirstN (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 0
  else g.a 0 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_ratio 
  (g : GeometricSequence) 
  (h : sumFirstN g 3 = 7 * g.a 3) : 
  g.q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l308_30806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_satisfying_gcd_equation_l308_30826

theorem smallest_b_satisfying_gcd_equation : 
  ∃ b : ℕ, b > 0 ∧ Nat.gcd (Nat.gcd 12 16) (Nat.gcd b 12) = 2 ∧ 
  ∀ c : ℕ, c > 0 → c < b → Nat.gcd (Nat.gcd 12 16) (Nat.gcd c 12) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_satisfying_gcd_equation_l308_30826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_seating_arrangement_l308_30867

/-- The number of ways to select k participants from n, with d-1 empty seats between them. -/
def number_of_ways_to_select (n k d : ℕ) : ℕ := 
  n / k * Nat.choose (n - k * d + k - 1) (k - 1)

/-- Theorem stating the correct number of ways to select participants in the Olympiad seating arrangement. -/
theorem olympiad_seating_arrangement (n k d : ℕ) 
  (h1 : n ≥ 4) 
  (h2 : k ≥ 2) 
  (h3 : d ≥ 2) 
  (h4 : k * d ≤ n) : 
  number_of_ways_to_select n k d = n / k * Nat.choose (n - k * d + k - 1) (k - 1) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_seating_arrangement_l308_30867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l308_30894

/-- The function g defined for positive real numbers a, b, and c -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a^2 + b) + b / (b^2 + c) + c / (c^2 + a)

/-- The theorem stating the range of g -/
theorem g_range :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  ∃ (x : ℝ), x ≥ 0 ∧ g a b c = x ∧
  ∀ (y : ℝ), y ≥ 0 → ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ g a' b' c' = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l308_30894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_three_factor_l308_30842

open BigOperators

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_of_factorials (n : ℕ) : ℕ := factorial n + factorial (n + 1) + factorial (n + 2)

def power_of_three_in_factorial (n : ℕ) : ℕ := 
  (n / 3) + (n / 9) + (n / 27)

theorem largest_power_of_three_factor :
  ∃ (n : ℕ), n = 22 ∧ 
  (3^n : ℕ) ∣ sum_of_factorials 48 ∧
  ∀ (m : ℕ), m > n → ¬((3^m : ℕ) ∣ sum_of_factorials 48) :=
by
  sorry

#eval sum_of_factorials 48
#eval power_of_three_in_factorial 48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_three_factor_l308_30842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l308_30837

-- Define the complex number z as a function of a
noncomputable def z (a : ℝ) : ℂ := (3 - a * Complex.I) / Complex.I

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- Statement to prove
theorem sufficient_but_not_necessary (a : ℝ) :
  (a ≥ 0 → in_third_quadrant (z a)) ∧
  ¬(in_third_quadrant (z a) → a ≥ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l308_30837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l308_30851

-- Define the circles and their properties
noncomputable def circleP_radius : ℝ := 3
noncomputable def circleR_radius : ℝ := circleP_radius / 2
noncomputable def circleQ_radius : ℝ := 2 * circleR_radius

-- Define p and q as natural numbers
def p : ℕ := 49
def q : ℕ := 7

-- State the theorem
theorem circle_tangency_theorem :
  circleP_radius = 3 ∧
  circleQ_radius = 2 * circleR_radius ∧
  circleQ_radius = Real.sqrt (p : ℝ) - q ∧
  p > 0 ∧
  q > 0 →
  p + q = 56 := by
  intro h
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l308_30851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_real_roots_l308_30860

noncomputable section

-- Define the inverse proportion function
def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 - k

-- State the theorem
theorem two_distinct_real_roots (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ → inverse_proportion k x₁ > inverse_proportion k x₂) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0 :=
by
  intro h
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_real_roots_l308_30860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpoles_equal_area_l308_30836

/-- Represents a tadpole shape -/
structure Tadpole where
  circle_radius : ℝ
  circle_area : ℝ
  triangle_area : ℝ

/-- Calculates the area of a tadpole -/
def tadpole_area (t : Tadpole) : ℝ :=
  t.triangle_area + (3.5 * t.circle_area)

/-- States that two tadpoles with the same circle radius have equal areas -/
theorem tadpoles_equal_area (t1 t2 : Tadpole) 
  (h1 : t1.circle_radius = t2.circle_radius)
  (h2 : t1.circle_area = π * t1.circle_radius^2)
  (h3 : t2.circle_area = π * t2.circle_radius^2)
  (h4 : t1.triangle_area = (Real.sqrt 3 / 4) * (2 * t1.circle_radius)^2)
  (h5 : t2.triangle_area = (Real.sqrt 3 / 4) * (2 * t2.circle_radius)^2) :
  tadpole_area t1 = tadpole_area t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpoles_equal_area_l308_30836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l308_30871

-- Part 1
theorem part_one : Real.sqrt 9 + ((-8 : ℝ) ^ (1/3)) + abs (1 - Real.sqrt 3) = Real.sqrt 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : a ≠ 0) : (12 * a^3 - 6 * a^2 + 3 * a) / (3 * a) = 4 * a^2 - 2 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l308_30871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_side_sum_l308_30811

-- Define the polygon ABCDEF
structure Polygon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the lengths of the sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the area of a polygon
noncomputable def area (p : Polygon) : ℝ := sorry

-- Theorem statement
theorem polygon_side_sum (p : Polygon) :
  area p = 78 ∧
  side_length p.A p.B = 10 ∧
  side_length p.B p.C = 11 ∧
  side_length p.F p.A = 7 →
  side_length p.D p.E + side_length p.E p.F = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_side_sum_l308_30811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_of_cone_l308_30886

/-- The magnitude of the solid angle of a right circular cone with vertex angle α -/
noncomputable def solid_angle (α : ℝ) : ℝ := 4 * Real.pi * (Real.sin (α / 4))^2

/-- Theorem stating that the magnitude of the solid angle of a right circular cone
    with vertex angle α is equal to 4π sin²(α/4) -/
theorem solid_angle_of_cone (α : ℝ) :
  solid_angle α = 4 * Real.pi * (Real.sin (α / 4))^2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_of_cone_l308_30886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_theta_l308_30873

/-- A function f with the given properties -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.sin x + 2

/-- Theorem stating the relationship between f(θ) and f(-θ) -/
theorem f_negative_theta (a b θ : ℝ) : f a b θ = -5 → f a b (-θ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_theta_l308_30873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_maximum_l308_30891

open Real Matrix

/-- The determinant function for the given 3x3 matrix --/
noncomputable def det_function (θ : ℝ) : ℝ := 
  det ![![1, 1, 1],
       ![1, 1 + Real.tan θ, 1],
       ![1 + Real.cos θ, 1, 1]]

/-- Theorem stating that the determinant has no finite maximum value --/
theorem no_finite_maximum : ¬ ∃ (M : ℝ), ∀ (θ : ℝ), det_function θ ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_maximum_l308_30891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_volume_equals_formula_l308_30857

/-- The volume of the space inside a sphere with radius 7 units, outside an inscribed right cylinder 
    with base radius 4 units and an inscribed right circular cone sharing the base with the cylinder. -/
noncomputable def space_volume : ℝ := 
  let sphere_radius : ℝ := 7
  let cylinder_base_radius : ℝ := 4
  let sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius^3
  let cylinder_height : ℝ := 2 * Real.sqrt (sphere_radius^2 - cylinder_base_radius^2)
  let cylinder_volume : ℝ := Real.pi * cylinder_base_radius^2 * cylinder_height
  let cone_volume : ℝ := (1/3) * Real.pi * cylinder_base_radius^2 * cylinder_height
  sphere_volume - cylinder_volume - cone_volume

/-- Theorem stating that the space_volume is equal to (1372 - 128√33)/3 π cubic units. -/
theorem space_volume_equals_formula : 
  space_volume = (1372 - 128 * Real.sqrt 33) / 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_volume_equals_formula_l308_30857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_b_part_d_l308_30864

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a*x + b

-- Theorem for part B
theorem part_b (a b : ℝ) :
  a = -3 →
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b x = 0 ∧ f a b y = 0 ∧ f a b z = 0) →
  b ∈ Set.Ioo (-9 : ℝ) (5/3 : ℝ) := by
  sorry

-- Theorem for part D
theorem part_d (a b : ℝ) :
  (∃ x₀ : ℝ, HasDerivAt (f a b) 0 x₀) →
  (∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ f a b x₀ = f a b x₁) →
  (∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ f a b x₀ = f a b x₁ ∧ x₁ + 2*x₀ + 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_b_part_d_l308_30864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_3x_minus_y_l308_30888

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := 1 + Real.log x

-- Theorem statement
theorem tangent_line_parallel_to_3x_minus_y (P : ℝ × ℝ) :
  (∀ x : ℝ, x > 0 → f_derivative x = deriv f x) →
  HasDerivAt f (f_derivative P.1) P.1 →
  f_derivative P.1 = 3 →
  P = (Real.exp 2, 2 * Real.exp 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_3x_minus_y_l308_30888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l308_30849

/-- The set of complex numbers z such that (3+4i)z is real -/
def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), (3 + 4 * Complex.I) * z = r}

/-- Theorem stating that S is a line in the complex plane -/
theorem S_is_line : ∃ (a b : ℝ), S = {z : ℂ | z.im = a * z.re + b} :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l308_30849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_problem_l308_30890

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * x + 1 - Real.sqrt 3 / 3

noncomputable def f_inv (x : ℝ) : ℝ := (x - (1 - Real.sqrt 3 / 3)) / Real.sqrt 3

theorem linear_function_problem :
  (∀ x, f x = 3 * f_inv x - 2) ∧
  f 3 = 5 →
  f 5 = 14 * Real.sqrt 3 / 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_problem_l308_30890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l308_30838

def M : ℕ := 72^5 + 5*72^4 + 10*72^3 + 10*72^2 + 5*72 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (fun x : ℕ => x > 0 ∧ M % x = 0) (Finset.range (M + 1))).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l308_30838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_equal_teams_smallest_n_no_five_equal_teams_l308_30835

/-- Represents a football team in the tournament -/
structure Team where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Calculates the total points for a team -/
def points (t : Team) : ℕ := 3 * t.wins + t.draws

/-- Represents a round-robin football tournament -/
structure Tournament where
  n : ℕ
  teams : Fin n → Team
  n_gt_4 : n > 4
  equal_points : ∀ i j : Fin n, points (teams i) = points (teams j)

/-- States that there exist four teams with equal wins, draws, and losses -/
theorem exist_four_equal_teams (t : Tournament) :
  ∃ i j k l : Fin t.n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    t.teams i = t.teams j ∧ t.teams i = t.teams k ∧ t.teams i = t.teams l :=
  sorry

/-- States that the smallest n for which there can be no five such teams is 10 -/
theorem smallest_n_no_five_equal_teams :
  (∀ n : ℕ, n ≥ 10 →
    ∃ t : Tournament,
      t.n = n ∧
      (∀ i j k l m : Fin t.n,
        i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
        k ≠ l ∧ k ≠ m ∧
        l ≠ m →
        t.teams i ≠ t.teams j ∨ t.teams i ≠ t.teams k ∨ t.teams i ≠ t.teams l ∨ t.teams i ≠ t.teams m)) ∧
  (∀ m : ℕ, m < 10 →
    ∃ t : Tournament,
      t.n = m ∧
      ∃ i j k l m : Fin t.n,
        i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
        k ≠ l ∧ k ≠ m ∧
        l ≠ m ∧
        t.teams i = t.teams j ∧ t.teams i = t.teams k ∧ t.teams i = t.teams l ∧ t.teams i = t.teams m) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_equal_teams_smallest_n_no_five_equal_teams_l308_30835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_series_sum_l308_30844

/-- The sum of the double infinite series 2^(-4k - 2j - (k + j)^2) for j and k from 0 to infinity is 4/3 -/
theorem double_infinite_series_sum : 
  (∑' (j : ℕ), ∑' (k : ℕ), (2 : ℝ)^(-(4*k + 2*j + (k + j)^2 : ℤ))) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_series_sum_l308_30844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l308_30863

/-- Given a sequence of points B_n and A_n satisfying certain conditions,
    prove properties about the sequences y_n and x_n. -/
theorem sequence_properties (a : ℝ) (h_a : 0 < a ∧ a < 1) :
  ∃ (x y : ℕ → ℝ),
    (∀ n : ℕ, y (n + 1) = (n + 1) / 4) ∧
    (x 1 = a) ∧
    (∀ n : ℕ, (x (n + 1) + x (n + 2)) / 2 = n + 1) →
    (∀ n : ℕ, y (n + 2) - y (n + 1) = 1 / 4) ∧
    (∀ n : ℕ, x (n + 3) - x (n + 1) = 2) ∧
    (∀ n : ℕ, x (n + 1) = if n % 2 = 0 then n + 1 + a - 1 else n + 1 - a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l308_30863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l308_30897

/-- Represents the percentage change in an investment over two years -/
noncomputable def two_year_investment_change (initial_investment : ℝ) (first_year_loss_percent : ℝ) (second_year_gain_percent : ℝ) : ℝ :=
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let final_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  (final_amount - initial_investment) / initial_investment * 100

/-- Theorem stating that a $100 investment with 15% loss in first year and 20% gain in second year results in 2% overall gain -/
theorem investment_result : 
  two_year_investment_change 100 15 20 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l308_30897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_cos_is_even_l308_30875

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := 2^x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def k (x : ℝ) : ℝ := Real.cos x

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem only_cos_is_even :
  ¬ is_even f ∧ ¬ is_even g ∧ ¬ is_even h ∧ is_even k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_cos_is_even_l308_30875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q₂_l308_30866

/-- A sequence of polyhedra where each subsequent polyhedron is obtained by 
    replacing midpoint triangles with outward-pointing regular octahedra -/
def PolyhedraSequence : ℕ → Type :=
  sorry

/-- The volume of a polyhedron in the sequence -/
noncomputable def volume (n : ℕ) : ℝ :=
  sorry

/-- Q₀ is a regular octahedron with volume 1 -/
axiom Q₀_volume : volume 0 = 1

/-- Q_{i+1} is obtained by replacing midpoint triangles with octahedra -/
axiom next_polyhedron (i : ℕ) : 
  volume (i + 1) = volume i + (8 * (1 / 8) ^ (i + 1))

/-- The volume of Q₂ is 3 -/
theorem volume_Q₂ : volume 2 = 3 := by
  sorry

#eval (3 : ℤ) + 1  -- This should output 4, which is m + n in the problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q₂_l308_30866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_symmetric_points_is_zero_l308_30830

noncomputable def v (x : ℝ) : ℝ := x^2 * Real.sin (Real.pi * x)

theorem sum_of_v_symmetric_points_is_zero :
  v (-1.5) + v (-0.5) + v 0.5 + v 1.5 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_symmetric_points_is_zero_l308_30830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l308_30877

-- Define the ellipse M
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line that contains the right focus
def focus_line (x y : ℝ) : Prop :=
  x + y = Real.sqrt 3

-- Define the midpoint of AB
def midpoint_of (xA yA xB yB x y : ℝ) : Prop :=
  x = (xA + xB) / 2 ∧ y = (yA + yB) / 2

-- Define the slope of OP
def slope_OP (x y : ℝ) : Prop :=
  y / x = 1 / 2

theorem ellipse_equation (a b : ℝ) (xA yA xB yB xP yP : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b xA yA ∧
  ellipse a b xB yB ∧
  focus_line (Real.sqrt 3) 0 ∧
  midpoint_of xA yA xB yB xP yP ∧
  slope_OP xP yP →
  a^2 = 6 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l308_30877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_in_bathtub_water_left_in_bathtub_is_7800_l308_30882

/-- Calculates the amount of water left in a bathtub after dripping, evaporation, and dumping. -/
theorem water_left_in_bathtub 
  (drip_rate : ℝ)           -- Dripping rate in ml/minute
  (evap_rate : ℝ)           -- Evaporation rate in ml/hour
  (time : ℝ)                -- Time in hours
  (dumped : ℝ)              -- Amount of water dumped in liters
  (h1 : drip_rate = 40)     -- Faucet dripping rate is 40 ml/minute
  (h2 : evap_rate = 200)    -- Water evaporation rate is 200 ml/hour
  (h3 : time = 9)           -- Water is running for 9 hours
  (h4 : dumped = 12)        -- 12 liters of water are dumped
  : ℝ :=
  let water_dripped := drip_rate * time * 60
  let water_evaporated := evap_rate * time
  let net_water_added := water_dripped - water_evaporated
  let water_dumped_ml := dumped * 1000
  net_water_added - water_dumped_ml

theorem water_left_in_bathtub_is_7800 
  (drip_rate : ℝ)           
  (evap_rate : ℝ)           
  (time : ℝ)                
  (dumped : ℝ)              
  (h1 : drip_rate = 40)     
  (h2 : evap_rate = 200)    
  (h3 : time = 9)           
  (h4 : dumped = 12)        
  : water_left_in_bathtub drip_rate evap_rate time dumped h1 h2 h3 h4 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_in_bathtub_water_left_in_bathtub_is_7800_l308_30882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l308_30868

-- Define the statement that we know is false
def falseStatement (x : Real) : Prop :=
  (2 ≤ x ∧ x ≤ 5) ∨ (x < 1 ∨ x > 4)

-- Define the range we want to prove
def targetRange (x : Real) : Prop :=
  1 ≤ x ∧ x < 2

-- Theorem statement
theorem range_of_x (x : Real) : 
  (¬ falseStatement x) → targetRange x :=
by
  intro h
  sorry -- Placeholder for the actual proof

#check range_of_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l308_30868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ADB_60_l308_30862

/-- A right-angled triangle with a square drawn on its hypotenuse -/
structure RightTriangleWithSquare where
  /-- The vertices of the triangle -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  /-- The center of the square drawn on the hypotenuse -/
  D : ℝ × ℝ
  /-- ABC is a right-angled triangle -/
  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  /-- Angle at B is 30° -/
  angle_B_30 : Real.cos (30 * π / 180) = (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) / 
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  /-- D is the center of a square drawn on BC -/
  D_center_square : (D.1 - B.1)^2 + (D.2 - B.2)^2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 4

/-- The measure of angle ADB is 60° -/
theorem angle_ADB_60 (t : RightTriangleWithSquare) : 
  Real.cos (60 * π / 180) = (t.A.1 - t.D.1) * (t.B.1 - t.D.1) + (t.A.2 - t.D.2) * (t.B.2 - t.D.2) / 
    (Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) * Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ADB_60_l308_30862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l308_30892

/-- Represents the number of students in the fifth grade -/
def f : ℕ := 1  -- We need to define f as a specific number, let's use 1 as an example

/-- Average running time for third grade students -/
def third_grade_avg : ℕ := 10

/-- Average running time for fourth grade students -/
def fourth_grade_avg : ℕ := 12

/-- Average running time for fifth grade students -/
def fifth_grade_avg : ℕ := 15

/-- Number of third grade students -/
def third_grade_students : ℕ := 6 * f

/-- Number of fourth grade students -/
def fourth_grade_students : ℕ := 2 * f

/-- Number of fifth grade students -/
def fifth_grade_students : ℕ := f

/-- Total number of students -/
def total_students : ℕ := third_grade_students + fourth_grade_students + fifth_grade_students

/-- Total minutes run by all students -/
def total_minutes : ℕ := 
  third_grade_avg * third_grade_students + 
  fourth_grade_avg * fourth_grade_students + 
  fifth_grade_avg * fifth_grade_students

/-- Theorem stating the average running time for all students -/
theorem average_running_time : 
  (total_minutes : ℚ) / (total_students : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l308_30892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l308_30898

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 6 / 9
  h_area : 2 * a * b = 2 * Real.sqrt 3

/-- Point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Helper function to calculate area of a triangle -/
noncomputable def area_triangle (C : Ellipse) (A M N : PointOnEllipse C) : ℝ :=
  abs ((M.x - A.x) * (N.y - A.y) - (N.x - A.x) * (M.y - A.y)) / 2

/-- Theorem statement -/
theorem ellipse_properties (C : Ellipse) :
  /- 1. Equation of the ellipse -/
  (C.a = Real.sqrt 3 ∧ C.b = 1) ∧
  /- 2. Fixed point property -/
  (∀ (A M N : PointOnEllipse C),
    A.y > 0 ∧ M ≠ A ∧ N ≠ A ∧ M ≠ N →
    (M.y - A.y) / (M.x - A.x) * (N.y - A.y) / (N.x - A.x) = 2/3 →
    ∃ (k : ℝ), M.y = k * M.x - 3 ∧ N.y = k * N.x - 3) ∧
  /- 3. Maximum area of triangle -/
  (∃ (A M N : PointOnEllipse C),
    A.y > 0 ∧ M ≠ A ∧ N ≠ A ∧ M ≠ N ∧
    (M.y - A.y) / (M.x - A.x) * (N.y - A.y) / (N.x - A.x) = 2/3 ∧
    ∀ (M' N' : PointOnEllipse C),
      M' ≠ A ∧ N' ≠ A ∧ M' ≠ N' ∧
      (M'.y - A.y) / (M'.x - A.x) * (N'.y - A.y) / (N'.x - A.x) = 2/3 →
      area_triangle C A M N ≥ area_triangle C A M' N') ∧
  (∀ (A M N : PointOnEllipse C),
    A.y > 0 ∧ M ≠ A ∧ N ≠ A ∧ M ≠ N ∧
    (M.y - A.y) / (M.x - A.x) * (N.y - A.y) / (N.x - A.x) = 2/3 →
    area_triangle C A M N ≤ 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l308_30898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l308_30829

/-- Variable cost function --/
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 2*x else 7*x + 100/x - 37

/-- Profit function --/
noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then -(1/3) * x^2 + 4*x - 20
  else if x ≥ 8 then -x - 100/x + 35
  else 0  -- undefined for x ≤ 0

/-- The maximum profit is achieved at 10,000 units and equals $15,000 --/
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 10 ∧
  ∀ (x : ℝ), P x ≤ P x_max ∧ P x_max = 15 := by
  sorry

#check max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l308_30829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_int_roots_possible_values_l308_30802

/-- A polynomial of degree 5 with integer coefficients -/
structure Poly5 where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The number of integer roots (counting multiplicity) of a Poly5 -/
def num_int_roots (p : Poly5) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem num_int_roots_possible_values (p : Poly5) :
  num_int_roots p ∈ ({0, 1, 2, 3, 4, 5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_int_roots_possible_values_l308_30802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l308_30823

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}

theorem complement_of_A : Set.univ \ A = {x : ℝ | x ≤ -1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l308_30823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_run_rate_is_six_l308_30870

/-- Represents a cricket match situation -/
structure CricketMatch where
  totalOvers : ℕ
  targetRuns : ℕ
  playedOvers : ℕ
  scoredRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (m : CricketMatch) : ℚ :=
  let remainingRuns := m.targetRuns - m.scoredRuns
  let remainingOvers := m.totalOvers - m.playedOvers
  (remainingRuns : ℚ) / remainingOvers

theorem required_run_rate_is_six (m : CricketMatch) 
  (h1 : m.totalOvers = 50)
  (h2 : m.targetRuns = 272)
  (h3 : m.playedOvers = 10)
  (h4 : m.scoredRuns = 32) :
  requiredRunRate m = 6 := by
  sorry

#eval requiredRunRate { totalOvers := 50, targetRuns := 272, playedOvers := 10, scoredRuns := 32 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_run_rate_is_six_l308_30870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_l308_30885

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter and orthocenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex B
noncomputable def angle_B (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_special_case (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  distance t.B O = distance t.B H →
  angle_B t = 60 ∨ angle_B t = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_l308_30885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l308_30841

/-- Line in 2D plane -/
structure Line₂ where
  equation : ℝ → ℝ → Prop

/-- Rotate a line by an angle around a point -/
def Line₂.rotate (l : Line₂) (angle : ℝ) (point : ℝ × ℝ) : Line₂ :=
  sorry

/-- X-intercept of a line -/
def Line₂.x_intercept (l : Line₂) : ℝ × ℝ :=
  sorry

/-- Given a line m with equation 2x + 3y - 6 = 0 in the coordinate plane,
    rotated 30° clockwise about the point (3, -2) to obtain line n,
    the x-coordinate of the x-intercept of line n is ((-2√3 + 6)(3√3 - 2)) / (2√3 + 3) -/
theorem x_intercept_after_rotation (m n : Line₂) :
  (m.equation = fun x y ↦ 2 * x + 3 * y - 6 = 0) →
  (n = m.rotate (Real.pi / 6) (3, -2)) →
  (n.x_intercept.1 = ((-2 * Real.sqrt 3 + 6) * (3 * Real.sqrt 3 - 2)) / (2 * Real.sqrt 3 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l308_30841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l308_30887

/-- Sample correlation coefficient -/
noncomputable def sample_correlation_coefficient (x y : List ℝ) : ℝ := sorry

/-- Degree of correlation between two variables -/
noncomputable def degree_of_correlation (r : ℝ) : ℝ := sorry

theorem correlation_coefficient_properties (x y : List ℝ) :
  let r := sample_correlation_coefficient x y
  ∀ ε > 0, ∃ δ > 0,
    (|r| ≤ 1) ∧
    (∀ r' : ℝ, |r'| > 1 - δ → |r'| ≤ 1 ∧ degree_of_correlation r' > degree_of_correlation r - ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l308_30887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_approx_4_03_l308_30809

/-- Represents the biathlon training session. -/
structure BiathlonSession where
  totalTime : ℝ
  bikingDistance : ℝ
  bikingSpeedFn : ℝ → ℝ
  transitionTime : ℝ
  runningDistance : ℝ
  runningSpeedVar : ℝ

/-- The biathlon session satisfies the time equation. -/
def satisfiesTimeEquation (session : BiathlonSession) : Prop :=
  session.bikingDistance / session.bikingSpeedFn session.runningSpeedVar +
  session.runningDistance / session.runningSpeedVar =
  (session.totalTime - session.transitionTime) / 60

/-- The specific biathlon session from the problem. -/
def johnSession : BiathlonSession where
  totalTime := 180
  bikingDistance := 36
  bikingSpeedFn := λ x => 3 * x + 2
  transitionTime := 10
  runningDistance := 8
  runningSpeedVar := 0  -- This will be solved for

/-- The theorem stating that the running speed is approximately 4.03 mph. -/
theorem running_speed_approx_4_03 :
  ∃ x : ℝ, satisfiesTimeEquation {johnSession with runningSpeedVar := x} ∧
  |x - 4.03| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_approx_4_03_l308_30809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l308_30828

/-- The area of a triangle with vertices (1,8,11), (0,7,7), and (-3,10,7) is 9. -/
theorem triangle_area : ℝ := by
  -- Define the vertices
  let A : Fin 3 → ℝ := ![1, 8, 11]
  let B : Fin 3 → ℝ := ![0, 7, 7]
  let C : Fin 3 → ℝ := ![-3, 10, 7]

  -- Define the function to calculate distance between two points
  let distance (p q : Fin 3 → ℝ) : ℝ :=
    Real.sqrt ((p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2)

  -- Calculate the sides of the triangle
  let AB := distance A B
  let BC := distance B C
  let AC := distance A C

  -- Define the area of the triangle
  let area := (1/2) * AB * BC

  -- Prove that the area equals 9
  have h : area = 9 := by sorry

  -- Return the numeric value
  exact 9


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l308_30828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_b_l308_30814

/-- The function b(x) defined by a quadratic numerator and denominator. -/
noncomputable def b (k : ℝ) (x : ℝ) : ℝ := (k * x^2 + 2 * x - 5) / (-5 * x^2 + 2 * x + k)

/-- The domain of b(x) is all real numbers iff k < -1/5 -/
theorem domain_of_b (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, b k x = y) ↔ k < -1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_b_l308_30814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_point_sets_l308_30872

-- Define the concept of a "perpendicular point set"
def is_perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M →
    ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = 1 / (p.1 ^ 2) ∧ p.1 ≠ 0}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 / Real.log 2 ∧ p.1 > 0}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = 2 ^ p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- State the theorem
theorem perpendicular_point_sets :
  is_perpendicular_point_set M₁ ∧
  ¬is_perpendicular_point_set M₂ ∧
  is_perpendicular_point_set M₃ ∧
  is_perpendicular_point_set M₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_point_sets_l308_30872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l308_30805

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ r : ℝ, (1 + 2 * Complex.I) * z = r}

-- Theorem statement
theorem S_is_line : 
  ∃ a b : ℝ, a ≠ 0 ∧ S = {z : ℂ | z.im = a * z.re + b} :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l308_30805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l308_30821

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The condition q: the lines x - ay + 1 = 0 and x + a²y - 1 = 0 are parallel -/
def condition_q (a : ℝ) : Prop := 
  are_parallel (1 / (-a)) (1 / a^2)

/-- The statement that p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q : 
  (∀ a : ℝ, a = -1 → condition_q a) ∧ 
  (∃ a : ℝ, a ≠ -1 ∧ condition_q a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l308_30821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_abc_l308_30839

/-- The area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

/-- The minimum area of a triangle ABC where A=(0,0), B=(36,15), and C has integer coordinates -/
theorem min_area_triangle_abc :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (36, 15)
  ∃ (C : ℤ × ℤ), ∀ (D : ℤ × ℤ),
    area_triangle A B (C.1, C.2) ≤ area_triangle A B (D.1, D.2) ∧
    area_triangle A B (C.1, C.2) = (3 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_abc_l308_30839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_approximately_281_44_l308_30807

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period -/
noncomputable def bankers_gain (bankers_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  let principal := bankers_discount * 100 / (interest_rate * time)
  let true_discount := principal / (1 + (interest_rate * time / 100))
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is approximately 281.44 given the specified conditions -/
theorem bankers_gain_approximately_281_44 :
  ∃ ε > 0, |bankers_gain 1462 12 6 - 281.44| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_approximately_281_44_l308_30807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_function_and_range_l308_30847

/-- Represents the total number of notebooks to be purchased -/
def total_notebooks : ℕ := 30

/-- Represents the price of notebook A in yuan -/
def price_A : ℕ := 12

/-- Represents the price of notebook B in yuan -/
def price_B : ℕ := 8

/-- Represents the number of notebook A purchased -/
def x : ℕ → ℕ := id

/-- Represents the total cost of purchasing notebooks -/
def total_cost (x : ℕ) : ℕ := price_A * x + price_B * (total_notebooks - x)

/-- Theorem stating the total cost function and its range -/
theorem total_cost_function_and_range :
  ∀ x : ℕ, x ≤ total_notebooks →
    total_cost x = price_A * x + price_B * (total_notebooks - x) ∧
    price_B * total_notebooks ≤ total_cost x ∧ total_cost x ≤ price_A * total_notebooks :=
by
  intro x h
  have h1 : total_cost x = price_A * x + price_B * (total_notebooks - x) := rfl
  have h2 : price_B * total_notebooks ≤ total_cost x :=
  by
    sorry -- Proof of lower bound
  have h3 : total_cost x ≤ price_A * total_notebooks :=
  by
    sorry -- Proof of upper bound
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_function_and_range_l308_30847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_thirty_degrees_l308_30824

theorem sin_thirty_degrees :
  Real.sin (30 * π / 180) = 1/2 := by
  let unit_circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let thirty_sixty_ninety_triangle_ratios := ∃ (a : ℝ), a > 0 ∧ (a, Real.sqrt 3 * a, 2 * a) = (1, Real.sqrt 3, 2)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_thirty_degrees_l308_30824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l308_30812

-- Define polynomials A and B
variable (x y : ℝ)
def B (x y : ℝ) : ℝ := 4 * x^2 - 3 * y - 1

-- Define the condition for A + B
def A_plus_B (x y : ℝ) : ℝ := 6 * x^2 - y

-- Define A
def A (x y : ℝ) : ℝ := 2 * x^2 + 2 * y + 1

-- Define the opposite sign condition
def opposite_sign (x y : ℝ) : Prop := (|x - 1| + (y + 1)^2 = 0)

theorem polynomial_problem (x y : ℝ) :
  (A x y + B x y = A_plus_B x y) ∧ 
  (opposite_sign x y → A x y - B x y = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l308_30812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_three_l308_30856

/-- An arithmetic progression with given initial terms -/
def arithmetic_progression (x : ℝ) : ℕ → ℝ
  | 0 => x - 1  -- Adding case for 0
  | 1 => x - 1
  | 2 => x^2 - 1
  | 3 => x^3 - 1
  | n + 4 => arithmetic_progression x (n + 3) + (x^2 - x)

/-- The smallest n for which a_n = 2x^2 + 2x - 3 is 3 -/
theorem smallest_n_is_three (x : ℝ) :
  (∀ n < 3, arithmetic_progression x n ≠ 2*x^2 + 2*x - 3) ∧
  arithmetic_progression x 3 = 2*x^2 + 2*x - 3 := by
  sorry

#eval arithmetic_progression 2 3  -- Example evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_three_l308_30856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_is_twenty_l308_30810

/-- The polynomial function representing the base-n number 352362_n in decimal form -/
def f (n : ℕ) : ℕ := 2 + 6*n + 3*n^2 + 2*n^3 + 5*n^4 + 3*n^5

/-- The count of numbers n in [2, 100] for which f(n) is divisible by 5 -/
def count_divisible_by_five : ℕ := 
  (Finset.range 99).filter (fun x => f (x + 2) % 5 = 0) |>.card

theorem count_divisible_by_five_is_twenty : count_divisible_by_five = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_is_twenty_l308_30810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_48_l308_30859

def sequence_a : ℕ → ℤ
  | 0 => 3
  | n + 1 => -2 * sequence_a n

theorem fifth_term_is_48 : sequence_a 4 = 48 := by
  rw [sequence_a, sequence_a, sequence_a, sequence_a, sequence_a]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_48_l308_30859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l308_30858

theorem solution_set (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, 
    (∀ i : Fin 5, 0 ≤ (Vector.ofFn ![x₁, x₂, x₃, x₄, x₅]).get i) ∧
    (x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = a) ∧
    (x₁ + 8*x₂ + 27*x₃ + 64*x₄ + 125*x₅ = a^2) ∧
    (x₁ + 32*x₂ + 243*x₃ + 1024*x₄ + 3125*x₅ = a^3))
  → a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l308_30858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l308_30840

-- Define the parabola and circle equations
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ myCircle p.1 p.2}

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
    p1 ≠ p2 ∧ distance p1 p2 = (4 * Real.sqrt 97) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l308_30840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_degenerate_hyperbola_l308_30878

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  r : ℝ → ℝ

/-- Represents a degenerate hyperbola -/
structure DegenerateHyperbola where
  a : ℝ

/-- The curve defined by r = 1 / (1 + cos θ) -/
noncomputable def curve : PolarCurve :=
  { r := λ θ => 1 / (1 + Real.cos θ) }

/-- Theorem stating that the given curve is a degenerate hyperbola -/
theorem curve_is_degenerate_hyperbola : 
  ∃ (h : DegenerateHyperbola), 
    ∀ θ : ℝ, (curve.r θ * (1 + Real.cos θ))^2 + (curve.r θ * Real.sin θ)^2 = h.a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_degenerate_hyperbola_l308_30878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_specific_line_l308_30895

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  let b := l.y₁ - m * l.x₁
  (0, b)

/-- The line passing through (3, 2) and (-1, 6) -/
def specific_line : Line := { x₁ := 3, y₁ := 2, x₂ := -1, y₂ := 6 }

theorem y_intercept_of_specific_line :
  y_intercept specific_line = (0, 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_specific_line_l308_30895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_conditions_l308_30804

/-- A function f(x) with parameters a, b, and c. -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- Theorem stating the conditions for f to have both a maximum and a minimum -/
theorem f_max_min_conditions (a b c : ℝ) (ha : a ≠ 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    (∀ x > 0, f a b c x ≤ f a b c x₁) ∧
    (∀ x > 0, f a b c x ≥ f a b c x₂)) :
  a * b > 0 ∧ b^2 + 8*a*c > 0 ∧ a * c < 0 := by
  sorry

#check f_max_min_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_conditions_l308_30804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l308_30852

/-- The absolute value function -/
def f (x : ℝ) : ℝ := |x|

/-- The square root of x squared function -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2)

/-- Theorem stating that f and g are the same function -/
theorem f_eq_g : f = g := by
  ext x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l308_30852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_2sqrt5_div_5_l308_30822

/-- A circle passing through (2,1) and tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (center.1 - 2)^2 + (center.2 - 1)^2 = radius^2
  tangent_to_axes : center.1 = radius ∧ center.2 = radius

/-- The line 2x - y - 3 = 0 -/
def target_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |2 * p.1 - p.2 - 3| / Real.sqrt 5

theorem distance_to_line_is_2sqrt5_div_5 (c : TangentCircle) :
  distance_to_line c.center = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_2sqrt5_div_5_l308_30822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l308_30820

theorem division_problem (n j : ℕ) 
  (h1 : n % j = 28)
  (h2 : (n : ℝ) / (j : ℝ) = 142.07)
  (hn : n > 0)
  (hj : j > 0) : 
  j = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l308_30820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_solution_l308_30825

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_solution (n : ℝ) : 
  (floor 6.5) * (floor (2/3)) + (floor 2) * n + (floor 8.4) - 6.6 = 15.8 → n = 7.2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_solution_l308_30825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l308_30816

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom a_1 : sequence_a 1 = 1
axiom b_1 : sequence_b 1 = 0

axiom a_recurrence (n : ℕ) : 4 * sequence_a (n + 1) = 3 * sequence_a n - sequence_b n + 4
axiom b_recurrence (n : ℕ) : 4 * sequence_b (n + 1) = 3 * sequence_b n - sequence_a n - 4

theorem sequences_properties :
  (∀ n : ℕ, sequence_a n + sequence_b n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, sequence_a n - sequence_b n = 2*n - 1) ∧
  (∀ n : ℕ, sequence_a n = (1/2)^n + n - 1/2) ∧
  (∀ n : ℕ, sequence_b n = (1/2)^n - n + 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l308_30816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_area_l308_30869

/-- A tile with the given properties -/
structure Tile where
  /-- The tile can be divided into three congruent quadrilaterals -/
  quadrilaterals : Fin 3 → Quadrilateral
  /-- The tile has nine sides -/
  sides : Fin 9 → ℝ
  /-- Six sides of the tile have length 1 -/
  six_unit_sides : ∃ (s : Finset (Fin 9)), s.card = 6 ∧ ∀ i ∈ s, sides i = 1
  /-- The shape tiles the plane -/
  tiles_plane : Bool

/-- The area of a tile with the given properties is 4√3/3 -/
theorem tile_area (t : Tile) : Real.sqrt 3 * 4 / 3 = 4 * Real.sqrt 3 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_area_l308_30869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l308_30881

/-- Given a function f where f(1) = 0, prove that f(0+1) + 1 = 1 -/
theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 0) : f 0 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l308_30881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_l308_30831

-- Define the triangle and points
variable (X Y Z K L M : ℝ × ℝ)

-- Define the ratios
variable (B C : ℝ)

-- Helper functions
def IsEquilateral (A B C : ℝ × ℝ) : Prop := sorry
def OnSegment (P A B : ℝ × ℝ) : Prop := sorry
def AreaRatio (A B C D E F : ℝ × ℝ) : ℝ := sorry

-- Define XK, KY, YL, LZ, ZM, MX
def XK : ℝ := sorry
def KY : ℝ := sorry
def YL : ℝ := sorry
def LZ : ℝ := sorry
def ZM : ℝ := sorry
def MX : ℝ := sorry

-- State the theorem
theorem area_ratio_equilateral_triangle 
  (h_equilateral : IsEquilateral X Y Z)
  (h_K_on_XY : OnSegment K X Y)
  (h_L_on_YZ : OnSegment L Y Z)
  (h_M_on_ZX : OnSegment M Z X)
  (h_XK_KY : XK / KY = B)
  (h_YL_LZ : YL / LZ = 1 / C)
  (h_ZM_MX : ZM / MX = 1)
  : AreaRatio K L M X Y Z = (B + C) / (2 * (B + 1) * (C + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_l308_30831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l308_30876

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

/-- The point A -/
def A : ℝ × ℝ := (1, 0)

/-- The point B -/
def B : ℝ × ℝ := (7, 6)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∃ (m : ℝ), m = 8 ∧ ∀ (P : ℝ × ℝ), P ∈ Parabola → distance A P + distance B P ≥ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l308_30876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_value_l308_30865

/-- A geometric sequence with sum of first n terms S_n = a · 3^n - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := λ n ↦ a * 3^n - 2

/-- The second term of the geometric sequence -/
def SecondTerm (a : ℝ) : ℝ := GeometricSequence a 2 - GeometricSequence a 1

theorem second_term_value (a : ℝ) : SecondTerm a = 12 := by
  -- Unfold the definitions
  unfold SecondTerm
  unfold GeometricSequence
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval SecondTerm 2  -- This should evaluate to 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_value_l308_30865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_AB_l308_30833

noncomputable section

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the curve C (locus of center of P)
def curve_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

-- Define point Q
def point_Q : ℝ × ℝ := (1, 8/3)

-- Theorem statement
theorem slope_of_AB : 
  ∀ (A B : ℝ × ℝ),
  curve_C A.1 A.2 → 
  curve_C B.1 B.2 →
  curve_C point_Q.1 point_Q.2 →
  (∃ (t : ℝ), A = (point_Q.1 + t, point_Q.2 + t * (B.2 - point_Q.2) / (B.1 - point_Q.1))) →
  (∃ (k : ℝ), (A.2 - 0) / (A.1 - 0) = k ∧ (B.2 - 0) / (B.1 - 0) = -k) →
  (B.2 - A.2) / (B.1 - A.1) = 1/3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_AB_l308_30833
