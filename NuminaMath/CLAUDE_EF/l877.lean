import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_7_equals_3_l877_87774

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  let y := (x - 1) / 2
  y^2 - 2*y

-- Theorem statement
theorem f_7_equals_3 : f 7 = 3 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_7_equals_3_l877_87774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_days_l877_87726

/-- Calculates the number of additional days required to complete a project given changes in workforce --/
def additional_days_required (initial_workers : ℕ) (initial_days : ℕ) (initial_hours_per_day : ℕ) 
  (days_before_change : ℕ) (additional_workers : ℕ) (new_workers_hours : ℕ) (original_workers_new_hours : ℕ) : ℕ :=
  let total_work_hours := initial_workers * initial_hours_per_day * initial_days
  let completed_work_hours := initial_workers * initial_hours_per_day * days_before_change
  let remaining_work_hours := total_work_hours - completed_work_hours
  let new_daily_work_hours := initial_workers * original_workers_new_hours + additional_workers * new_workers_hours
  (remaining_work_hours + new_daily_work_hours - 1) / new_daily_work_hours

/-- The theorem stating that under the given conditions, 3 additional days are required --/
theorem project_completion_days : 
  additional_days_required 6 8 8 3 4 6 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_days_l877_87726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_x_iff_geq_neg_one_l877_87754

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 1/2 * x - 1 else 1/x

theorem f_leq_x_iff_geq_neg_one (a : ℝ) : f a ≤ a ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_x_iff_geq_neg_one_l877_87754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l877_87757

theorem pure_imaginary_product (x : ℝ) : 
  ∃ (y : ℂ), (((x : ℂ) + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I) = Complex.I * y) 
    ↔ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 := by
  sorry

#check pure_imaginary_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l877_87757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l877_87728

theorem division_problem (x y : ℕ+) (h1 : (x : Int) % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l877_87728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l877_87790

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 3) - x - 3

theorem root_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l877_87790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_η_l877_87706

-- Define a type for binomial distribution
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  dummy : Unit

-- Define a function to represent the variance of a random variable
def variance (X : ℝ → ℝ) : ℝ := sorry

-- Define ξ as a function that represents a binomial distribution
def ξ : ℝ → ℝ := sorry

-- Define η in terms of ξ
def η : ℝ → ℝ := λ x => 5 * (ξ x) - 1

-- Theorem stating the variance of η
theorem variance_of_η : variance η = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_η_l877_87706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_strips_l877_87761

theorem overlap_area_of_strips (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  let β := α / 2
  let strip1_width := 1
  let strip2_width := 2
  let overlap_area := strip1_width * strip2_width / (2 * Real.sin β)
  overlap_area = 1 / Real.sin (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_strips_l877_87761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l877_87744

theorem min_value_trig_expression :
  (∀ x : ℝ, Real.sin x^4 + 2 * Real.cos x^4 + Real.sin x^2 ≥ 2/3) ∧
  (∃ y : ℝ, Real.sin y^4 + 2 * Real.cos y^4 + Real.sin y^2 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l877_87744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l877_87788

open BigOperators

def product_series (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 2), (1 + 1 / (k + 3 : ℚ))

theorem certain_number_proof :
  ∃ x : ℚ, (x / 11) * product_series 120 = 11 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l877_87788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87713

noncomputable def f (x : ℝ) := 6 * (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-(7 * Real.pi) / 12 + k * Real.pi) (-(Real.pi) / 12 + k * Real.pi))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_difference_is_zero_l877_87750

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 : Type := Fin 3 → Fin 3 → Int

/-- Creates the initial matrix with consecutive integers starting from 1 -/
def initialMatrix : Matrix3x3 :=
  λ i j => (i.val * 3 + j.val + 1 : Int)

/-- Reverses the order of elements in the first and third rows -/
def reverseFirstLastRows (m : Matrix3x3) : Matrix3x3 :=
  λ i j => if i = 0 || i = 2 then m i (2 - j) else m i j

/-- Calculates the sum of the main diagonal -/
def mainDiagonalSum (m : Matrix3x3) : Int :=
  m 0 0 + m 1 1 + m 2 2

/-- Calculates the sum of the anti-diagonal -/
def antiDiagonalSum (m : Matrix3x3) : Int :=
  m 0 2 + m 1 1 + m 2 0

/-- The main theorem to be proved -/
theorem diagonal_sum_difference_is_zero :
  let m := reverseFirstLastRows initialMatrix
  |mainDiagonalSum m - antiDiagonalSum m| = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_difference_is_zero_l877_87750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l877_87771

/-- A quadratic function with coefficient a, b, and c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the minimum point of a quadratic function -/
noncomputable def QuadraticFunction.min_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the y-intercept of a quadratic function -/
def QuadraticFunction.y_intercept (f : QuadraticFunction) : ℝ := f.c

/-- The roots of a quadratic function -/
noncomputable def QuadraticFunction.roots (f : QuadraticFunction) : ℝ × ℝ :=
  let d := Real.sqrt (f.b^2 - 4*f.a*f.c)
  ((-f.b - d) / (2*f.a), (-f.b + d) / (2*f.a))

theorem quadratic_function_properties (f : QuadraticFunction) 
  (h_min : f.min_x = 2)
  (h_roots : let (x₁, x₂) := f.roots; x₁ < 0 ∧ 0 < x₂)
  (h_tan : let (x₁, x₂) := f.roots
           let y := f.y_intercept
           (y / x₁).arctan - (y / x₂).arctan = 1) :
  f.b + 4*f.a = 0 ∧ f.a = 1/4 ∧ f.b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l877_87771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_midpoint_relation_l877_87740

/-- Given a triangle ABC with centroid G, arbitrary point P, and point D at the midpoint of AB,
    prove that PA^2 + PB^2 + PC^2 = 3⋅PG^2 + PD^2 + GA^2 + GB^2 + GC^2 -/
theorem triangle_centroid_midpoint_relation 
  (A B C P : EuclideanSpace ℝ (Fin 2)) 
  (G : EuclideanSpace ℝ (Fin 2)) 
  (hG : G = (1/3 : ℝ) • (A + B + C)) 
  (D : EuclideanSpace ℝ (Fin 2)) 
  (hD : D = (1/2 : ℝ) • (A + B)) : 
  ‖P - A‖^2 + ‖P - B‖^2 + ‖P - C‖^2 = 
  3 * ‖P - G‖^2 + ‖P - D‖^2 + ‖G - A‖^2 + ‖G - B‖^2 + ‖G - C‖^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_midpoint_relation_l877_87740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l877_87791

/-- Represents the two types of containers (lamps) --/
inductive ContainerType
| Yellow
| Blue

/-- Represents a distribution of objects (plants) among containers (lamps) --/
structure Distribution where
  yellowContainers : Fin 3 → Nat
  blueContainers : Fin 2 → Nat
  totalObjects : (yellowContainers 0 + yellowContainers 1 + yellowContainers 2 + 
                  blueContainers 0 + blueContainers 1) = 4

/-- The number of valid distributions --/
def validDistributions : Nat :=
  22

/-- Theorem stating that the number of valid distributions is 22 --/
theorem distribution_count : 
  ∃ (s : Finset Distribution), s.card = validDistributions ∧ 
    ∀ d : Distribution, d ∈ s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l877_87791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_other_sides_l877_87782

/-- A parallelogram with given properties -/
structure Parallelogram where
  -- Two adjacent sides
  side1 : (x y : ℝ) → x + y + 1 = 0
  side2 : (x y : ℝ) → 3 * x - y + 4 = 0
  -- Intersection point of diagonals
  diag_intersection : ℝ × ℝ := (3, 3)

/-- The other two sides of the parallelogram -/
def other_sides (p : Parallelogram) : (ℝ × ℝ → Prop) × (ℝ × ℝ → Prop) :=
  (λ (x, y) ↦ x + y - 13 = 0, λ (x, y) ↦ 3 * x - y - 16 = 0)

/-- Theorem stating that the other two sides are correct -/
theorem parallelogram_other_sides (p : Parallelogram) :
  other_sides p = (λ (x, y) ↦ x + y - 13 = 0, λ (x, y) ↦ 3 * x - y - 16 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_other_sides_l877_87782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87743

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sin (ω * x - Real.pi / 2)

theorem f_properties :
  (∀ x : ℝ, f (1/2) x ≤ Real.sqrt 2) ∧
  (∀ k : ℤ, f (1/2) (4 * ↑k * Real.pi + 3 * Real.pi / 2) = Real.sqrt 2) ∧
  (f 2 (Real.pi / 8) = 0) ∧
  (0 < 2 ∧ 2 < 10) ∧
  (∀ x : ℝ, f 2 (x + Real.pi) = f 2 x) ∧
  (∀ p : ℝ, 0 < p ∧ (∀ x : ℝ, f 2 (x + p) = f 2 x) → Real.pi ≤ p) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l877_87736

theorem trig_problem (θ : Real) (h1 : Real.sin θ = 3/5) (h2 : π/2 < θ ∧ θ < π) :
  (Real.tan θ = -3/4) ∧ (Real.cos (2*θ - π/3) = (7 - 24*Real.sqrt 3)/50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l877_87736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l877_87741

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 23 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 24 := by
  intros h_n h_initial_avg h_wrong_num h_correct_num
  -- The proof steps would go here
  sorry

#check correct_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l877_87741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l877_87779

/-- The spade operation for positive real numbers -/
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

/-- Theorem stating that spade(3, spade(4, 3)) = 30/11 -/
theorem spade_calculation :
  spade 3 (spade 4 3) = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l877_87779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_two_years_l877_87745

/-- Represents the height of a tree that triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that a tree tripling its height yearly reaches 9 feet after 2 years
    if it reaches 81 feet after 4 years -/
theorem tree_height_after_two_years
  (h : tree_height (tree_height h₀ 2) 2 = 81) :
  tree_height h₀ 2 = 9 :=
by
  sorry

where
  h₀ : ℝ := 1 -- Initial height (can be any non-zero value)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_two_years_l877_87745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l877_87798

noncomputable def f (x : ℝ) := 2^x + x - 5

theorem solution_interval (h1 : f 1 < 0) (h2 : f 2 > 0) :
  ∃ x, x ∈ Set.Ioo 1 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l877_87798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l877_87721

noncomputable def f (x : ℝ) := 7 * Real.sin (x - Real.pi / 6)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioo 0 (Real.pi / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l877_87721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l877_87793

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * Real.sin (Real.pi * x / 4 - Real.pi / 3)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi/2) (2*Real.pi/3), f (Real.cos x) ≤ a * Real.cos x + 1) ↔ a ∈ Set.Icc (-2) 7 :=
sorry

-- Part 2
theorem part_two (k : ℝ) :
  (∀ x1 ∈ Set.Icc (-2) (Real.sqrt 3), ∃ x2 ∈ Set.Ioo 0 4, g k x2 = f x1) ↔
  k ∈ Set.Ioi (-10 * Real.sqrt 3 / 3) ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l877_87793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_outcome_l877_87746

/-- Represents a line segment divided into three parts -/
structure DividedSegment where
  total : ℝ
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ
  sum_parts : part1 + part2 + part3 = total
  nonneg : 0 ≤ part1 ∧ 0 ≤ part2 ∧ 0 ≤ part3

/-- Check if three segments can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the game outcome based on segment lengths -/
theorem game_outcome (k l : ℝ) (hk : k > 0) (hl : l > 0) :
  (k > l → ∀ (segA segB : DividedSegment),
    segA.total = k → segB.total = l →
    ¬∃ (t1 t2 : Fin 6 → Nat),
      (∀ i : Fin 6, t1 i < 3 ∧ t2 i < 3) ∧
      (∀ i : Fin 6, t1 i ≠ t2 i) ∧
      canFormTriangle
        ((if t1 0 < 3 then segA.part1 else segB.part1) +
         (if t1 1 < 3 then segA.part2 else segB.part2) +
         (if t1 2 < 3 then segA.part3 else segB.part3))
        ((if t2 0 < 3 then segA.part1 else segB.part1) +
         (if t2 1 < 3 then segA.part2 else segB.part2) +
         (if t2 2 < 3 then segA.part3 else segB.part3))
        ((if (t1 3 < 3 ∧ t2 3 < 3) then segA.part1 else segB.part1) +
         (if (t1 4 < 3 ∧ t2 4 < 3) then segA.part2 else segB.part2) +
         (if (t1 5 < 3 ∧ t2 5 < 3) then segA.part3 else segB.part3))) ∧
  (k ≤ l → ∀ (segA : DividedSegment),
    segA.total = k →
    ∃ (segB : DividedSegment) (t1 t2 : Fin 6 → Nat),
      segB.total = l ∧
      (∀ i : Fin 6, t1 i < 3 ∧ t2 i < 3) ∧
      (∀ i : Fin 6, t1 i ≠ t2 i) ∧
      canFormTriangle
        ((if t1 0 < 3 then segA.part1 else segB.part1) +
         (if t1 1 < 3 then segA.part2 else segB.part2) +
         (if t1 2 < 3 then segA.part3 else segB.part3))
        ((if t2 0 < 3 then segA.part1 else segB.part1) +
         (if t2 1 < 3 then segA.part2 else segB.part2) +
         (if t2 2 < 3 then segA.part3 else segB.part3))
        ((if (t1 3 < 3 ∧ t2 3 < 3) then segA.part1 else segB.part1) +
         (if (t1 4 < 3 ∧ t2 4 < 3) then segA.part2 else segB.part2) +
         (if (t1 5 < 3 ∧ t2 5 < 3) then segA.part3 else segB.part3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_outcome_l877_87746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l877_87737

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  D : Real × Real

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a + 2 * t.b = t.c * Real.cos t.B + Real.sqrt 3 * t.c * Real.sin t.B ∧
  t.D.1 = 35 / 4 ∧  -- BD
  t.D.2 = 21 / 4    -- AD

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : TriangleProperties t) :
  t.C = 2 * Real.pi / 3 ∧ 
  (Real.pi * (Real.sqrt 3)^2 : Real) = 3 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l877_87737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lexie_mile_time_is_20_l877_87723

/-- The time it takes Celia to run 30 miles, in minutes -/
noncomputable def celia_time : ℚ := 300

/-- The distance Celia runs, in miles -/
noncomputable def celia_distance : ℚ := 30

/-- Celia's speed relative to Lexie's -/
noncomputable def speed_ratio : ℚ := 2

/-- The time it takes Lexie to run one mile, in minutes -/
noncomputable def lexie_mile_time : ℚ := celia_time / celia_distance * speed_ratio

theorem lexie_mile_time_is_20 : lexie_mile_time = 20 := by
  -- Unfold the definitions
  unfold lexie_mile_time celia_time celia_distance speed_ratio
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lexie_mile_time_is_20_l877_87723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l877_87797

-- Define the parabola
def is_on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the circle
def is_on_circle (Q : ℝ × ℝ) : Prop := Q.1^2 + (Q.2 - 4)^2 = 1

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def distance_to_directrix (P : ℝ × ℝ) : ℝ := P.1 + 1

theorem min_distance_sum :
  ∀ P Q : ℝ × ℝ,
  is_on_parabola P →
  is_on_circle Q →
  distance P Q + distance_to_directrix P ≥ Real.sqrt 17 - 1 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l877_87797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l877_87768

-- Define the ellipse
def Ellipse (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / (a^2 - 1) = 1}

-- Define the foci of the ellipse
noncomputable def LeftFocus (a : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - 1), 0)
noncomputable def RightFocus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - 1), 0)

-- Define the vector from a point to another point
def Vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define the vector addition
def VecAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define the vector scalar multiplication
def VecScale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define the vector norm
noncomputable def VecNorm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem max_vector_difference (a : ℝ) (ha : a > 1) :
  ∃ (M : ℝ), M = 2 * a ∧ 
  ∀ (P Q : ℝ × ℝ), P ∈ Ellipse a → Q ∈ Ellipse a →
  VecNorm (VecAdd (VecAdd (Vec P (LeftFocus a)) (Vec P (RightFocus a))) 
                  (VecScale (-2) (Vec P Q))) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l877_87768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_dot_product_l877_87742

/-- Given a sine function y = sin(2x + θ), prove that the dot product of vectors from an x-axis intersection point to the next maximum and minimum points equals 3π²/16 - 1 -/
theorem sine_intersection_dot_product (θ : ℝ) (a b c : ℝ) :
  (∃ x, Real.sin (2 * x + θ) = 0 ∧ x = a) →  -- P is an x-axis intersection point
  (∀ x ∈ Set.Icc a b, Real.sin (2 * x + θ) ≤ 1) →  -- A is the next maximum point
  Real.sin (2 * b + θ) = 1 →
  (∀ x ∈ Set.Icc b c, Real.sin (2 * x + θ) ≥ -1) →  -- B is the next minimum point
  Real.sin (2 * c + θ) = -1 →
  b - a = π / 4 →  -- Distance between P and A
  c - a = 3 * π / 4 →  -- Distance between P and B
  (b - a) * (c - a) - 1 = 3 * π^2 / 16 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_intersection_dot_product_l877_87742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l877_87748

/-- Given a square EFGH with side length 40 and a point Q inside the square,
    prove that the area of the quadrilateral formed by the centroids of
    triangles EFQ, FGQ, GHQ, and HEQ is 6400/9. -/
theorem centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) : 
  let square_side : ℝ := 40
  let EQ : ℝ := 16
  let FQ : ℝ := 34
  -- Square EFGH properties
  ((F.1 - E.1)^2 + (F.2 - E.2)^2 = square_side^2
  ∧ (G.1 - F.1)^2 + (G.2 - F.2)^2 = square_side^2
  ∧ (H.1 - G.1)^2 + (H.2 - G.2)^2 = square_side^2
  ∧ (E.1 - H.1)^2 + (E.2 - H.2)^2 = square_side^2
  -- Point Q inside the square
  ∧ (Q.1 - E.1)^2 + (Q.2 - E.2)^2 = EQ^2
  ∧ (Q.1 - F.1)^2 + (Q.2 - F.2)^2 = FQ^2)
  -- Centroid quadrilateral
  → let centroid_EFQ := ((E.1 + F.1 + Q.1) / 3, (E.2 + F.2 + Q.2) / 3)
    let centroid_FGQ := ((F.1 + G.1 + Q.1) / 3, (F.2 + G.2 + Q.2) / 3)
    let centroid_GHQ := ((G.1 + H.1 + Q.1) / 3, (G.2 + H.2 + Q.2) / 3)
    let centroid_HEQ := ((H.1 + E.1 + Q.1) / 3, (H.2 + E.2 + Q.2) / 3)
    -- Area of the centroid quadrilateral
    (centroid_FGQ.1 - centroid_HEQ.1)^2 + (centroid_FGQ.2 - centroid_HEQ.2)^2 +
    (centroid_GHQ.1 - centroid_EFQ.1)^2 + (centroid_GHQ.2 - centroid_EFQ.2)^2 = 2 * (6400 / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l877_87748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_rental_cost_l877_87751

/-- Calculates the rental cost for a car given the daily rate, per-mile rate, number of days, miles driven, and discount rate. -/
def rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) (discount_rate : ℚ) : ℚ :=
  let base_cost := daily_rate * days + mile_rate * miles
  let discount := if days > 4 then discount_rate * base_cost else 0
  base_cost - discount

/-- Proves that Alice's rental cost is $270 given the specified conditions. -/
theorem alice_rental_cost : 
  rental_cost 35 (1/4) 5 500 (1/10) = 270 := by
  -- Unfold the definition of rental_cost
  unfold rental_cost
  -- Simplify the arithmetic expressions
  simp [Rat.mul_def, Rat.add_def, Rat.sub_def]
  -- The proof is complete
  sorry

#eval rental_cost 35 (1/4) 5 500 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_rental_cost_l877_87751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_l877_87729

/-- Calculates the width of a cistern given its length, water depth, and total wet surface area. -/
theorem cistern_width (length depth total_area width : ℝ) 
  (h1 : length = 4)
  (h2 : depth = 1.25)
  (h3 : total_area = 62)
  (h4 : total_area = length * width + 2 * depth * length + 2 * depth * width) :
  width = 8 := by
  sorry

#check cistern_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_l877_87729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l877_87796

/-- Represents a train with a given length and speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- The time it takes for a train to cross a stationary point -/
noncomputable def crossTime (t : Train) : ℝ := t.length / t.speed

/-- The time it takes for two trains to cross each other -/
noncomputable def crossEachOther (t1 t2 : Train) : ℝ := (t1.length + t2.length) / (t1.speed + t2.speed)

theorem train_crossing_time 
  (t1 t2 : Train) 
  (h1 : t1.speed = t2.speed) 
  (h2 : crossTime t2 = 18) 
  (h3 : crossEachOther t1 t2 = 19) : 
  crossTime t1 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l877_87796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playhouse_siding_cost_l877_87767

/-- Calculates the cost of siding for a playhouse renovation --/
theorem playhouse_siding_cost :
  let wall_width : ℚ := 10
  let wall_height : ℚ := 7
  let roof_width : ℚ := 10
  let roof_height : ℚ := 6
  let siding_width : ℚ := 10
  let siding_height : ℚ := 12
  let siding_cost : ℚ := 30
  let wall_area := 2 * wall_width * wall_height
  let roof_area := 2 * (1/2 * roof_width * roof_height)
  let total_area := wall_area + roof_area
  let sections_needed := ⌈(total_area / (siding_width * siding_height))⌉
  sections_needed * siding_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_playhouse_siding_cost_l877_87767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_1968_l877_87747

/-- A sequence of numbers n + √(2n+1) for n from 1 to 1000 -/
noncomputable def rootSequence : Fin 1000 → ℝ := fun n => (n.val + 1 : ℝ) + Real.sqrt (2 * (n.val + 1) + 1)

/-- The set of roots for our polynomial -/
def rootSet : Set ℝ := {x | ∃ n : Fin 1000, x = rootSequence n}

/-- A polynomial with rational coefficients -/
structure RationalPoly where
  coeffs : List ℚ
  nonzero : coeffs ≠ []

/-- The degree of a polynomial -/
def degree (p : RationalPoly) : ℕ := p.coeffs.length - 1

/-- Evaluate a polynomial at a given point -/
noncomputable def evalPoly (coeffs : List ℚ) (x : ℝ) : ℝ :=
  coeffs.enum.foldr (fun ⟨i, a⟩ acc => acc + a * x ^ i) 0

/-- Predicate for a polynomial having all elements of rootSet as roots -/
def hasAllRoots (p : RationalPoly) : Prop :=
  ∀ x ∈ rootSet, evalPoly p.coeffs x = 0

theorem smallest_degree_1968 :
  ∀ p : RationalPoly, hasAllRoots p → degree p ≥ 1968 := by
  sorry

#eval 2 * 1000 - 32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_1968_l877_87747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_relations_l877_87763

/-- Given a triangle ABC where B is obtuse, prove specific trigonometric relations -/
theorem triangle_trig_relations (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_obtuse : Real.pi/2 < B) (h_B_less_pi : B < Real.pi)
  (h_eq1 : Real.cos A^2 + Real.cos B^2 + 2*Real.sin A*Real.sin B*Real.cos C = 16/7)
  (h_eq2 : Real.cos B^2 + Real.cos C^2 + 2*Real.sin B*Real.sin C*Real.cos A = 16/9) :
  Real.sin C = 1/Real.sqrt 7 ∧ 
  Real.cos C = Real.sqrt 6/Real.sqrt 7 ∧
  Real.sin A = 3/4 ∧ 
  Real.cos A = Real.sqrt 7/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_relations_l877_87763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_angles_sum_to_right_angle_l877_87760

/-- Given a vertical mast of height 1 meter that casts shadows of lengths 1, 2, and 3 meters,
    the sum of the angles formed by the mast and each shadow is 90°. -/
theorem shadow_angles_sum_to_right_angle (h : ℝ) (s₁ s₂ s₃ : ℝ) 
  (h_height : h = 1)
  (h_shadow1 : s₁ = 1)
  (h_shadow2 : s₂ = 2)
  (h_shadow3 : s₃ = 3) :
  Real.arctan (h / s₁) + Real.arctan (h / s₂) + Real.arctan (h / s₃) = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_angles_sum_to_right_angle_l877_87760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lcm_with_gcd_five_l877_87727

theorem smallest_lcm_with_gcd_five (k ℓ : ℕ) : 
  k ∈ Finset.range 9000 \ Finset.range 999 →
  ℓ ∈ Finset.range 9000 \ Finset.range 999 →
  Nat.gcd k ℓ = 5 →
  (∀ m n, m ∈ Finset.range 9000 \ Finset.range 999 →
         n ∈ Finset.range 9000 \ Finset.range 999 →
         Nat.gcd m n = 5 →
         Nat.lcm k ℓ ≤ Nat.lcm m n) →
  Nat.lcm k ℓ = 201000 := by
  sorry

#check smallest_lcm_with_gcd_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lcm_with_gcd_five_l877_87727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l877_87724

-- Define the train's length in meters
noncomputable def train_length : ℝ := 100

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 72

-- Convert km/hr to m/s
noncomputable def km_per_hour_to_meters_per_second (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

-- Calculate the time to cross the pole
noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (km_per_hour_to_meters_per_second speed_kmh)

-- Theorem statement
theorem train_crossing_time :
  time_to_cross train_length train_speed_kmh = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l877_87724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_parallel_l877_87730

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y = 0

-- Define what it means for a line to bisect a circle
def bisects (line : (ℝ → ℝ → Prop)) (circ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (cx cy r : ℝ), (∀ x y, circ x y ↔ (x - cx)^2 + (y - cy)^2 = r^2) ∧
                   line cx cy

-- Define what it means for two lines to be parallel
def parallel (line1 line2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d : ℝ), (∀ x y, line1 x y ↔ a*x + b*y + c = 0) ∧
                   (∀ x y, line2 x y ↔ a*x + b*y + d = 0)

-- The theorem to prove
theorem line_bisects_and_parallel : 
  bisects line_l my_circle ∧ parallel line_l given_line :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_and_parallel_l877_87730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_l877_87701

/-- The miles per gallon of Car Z at 45 mph -/
noncomputable def mpg_45 : ℝ := 50

/-- The miles per gallon of Car Z at 60 mph -/
noncomputable def mpg_60 : ℝ := 400 / 10

/-- The ratio of miles per gallon at 60 mph compared to 45 mph -/
noncomputable def efficiency_ratio : ℝ := 0.8

/-- Theorem stating the relationship between mpg at 45 mph and 60 mph -/
theorem car_z_efficiency :
  mpg_45 * efficiency_ratio = mpg_60 := by
  -- Expand definitions
  unfold mpg_45 mpg_60 efficiency_ratio
  -- Perform calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_l877_87701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_k_l877_87795

/-- Sum of an arithmetic progression with k terms, first term a, and common difference d -/
noncomputable def sum_ap (k : ℕ) (a d : ℝ) : ℝ := k * (2 * a + (k - 1) * d) / 2

/-- R is defined as the difference between s₄, s₂, and s₁ -/
noncomputable def R (k : ℕ) (a d : ℝ) : ℝ :=
  sum_ap (4 * k) a d - sum_ap (2 * k) a d - sum_ap k a d

/-- The theorem states that R depends only on d and k -/
theorem R_depends_on_d_and_k (k : ℕ) (a₁ a₂ d : ℝ) :
  R k a₁ d = R k a₂ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_k_l877_87795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_isosceles_trapezoid_l877_87738

/-- Represents a point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Represents an isosceles trapezoid with integer coordinates -/
structure IsoscelesTrapezoid where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint

/-- Checks if a line is horizontal or vertical -/
def isHorizontalOrVertical (p1 p2 : IntPoint) : Prop :=
  p1.x = p2.x ∨ p1.y = p2.y

/-- Checks if two lines are parallel -/
def areParallel (p1 p2 p3 p4 : IntPoint) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Calculates the slope of a line -/
noncomputable def lineSlope (p1 p2 : IntPoint) : ℚ :=
  (p2.y - p1.y : ℚ) / (p2.x - p1.x)

/-- Theorem: Sum of absolute values of all possible slopes for AB in the given isosceles trapezoid -/
theorem sum_of_slopes_isosceles_trapezoid (ABCD : IsoscelesTrapezoid) :
  ABCD.A = ⟨10, 50⟩ →
  ABCD.D = ⟨11, 57⟩ →
  ¬isHorizontalOrVertical ABCD.A ABCD.B →
  ¬isHorizontalOrVertical ABCD.B ABCD.C →
  ¬isHorizontalOrVertical ABCD.C ABCD.D →
  ¬isHorizontalOrVertical ABCD.D ABCD.A →
  areParallel ABCD.A ABCD.B ABCD.C ABCD.D →
  ¬areParallel ABCD.A ABCD.D ABCD.B ABCD.C →
  ∃ (slopes : List ℚ), (slopes.map abs).sum = 119 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_isosceles_trapezoid_l877_87738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l877_87731

theorem relationship_abc : 
  let a : ℝ := (1/2)^10
  let b : ℝ := (1/5)^(-(1/2 : ℝ))
  let c : ℝ := Real.log 10 / Real.log (1/3)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l877_87731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_four_l877_87755

/-- The length of a train in miles. -/
noncomputable def train_length : ℝ := 2

/-- The time it takes for the train to completely pass through the tunnel in minutes. -/
noncomputable def transit_time : ℝ := 4

/-- The speed of the train in miles per hour. -/
noncomputable def train_speed : ℝ := 90

/-- The length of the tunnel in miles. -/
noncomputable def tunnel_length : ℝ := train_speed * transit_time / 60 - train_length

theorem tunnel_length_is_four : tunnel_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_four_l877_87755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_product_of_smallest_primes_l877_87784

/-- rad(n) is the product of all prime divisors of n without multiplicities -/
noncomputable def rad (n : ℕ) : ℕ := 
  (Finset.filter Nat.Prime (Finset.range (n + 1))).prod (λ p => if n % p = 0 then p else 1)

/-- The sequence a_n defined by the recurrence relation -/
noncomputable def a : ℕ → ℕ
  | 0 => 1  -- Initial value set to 1, but this can be changed if needed
  | n + 1 => a n + rad (a n)

/-- The s-th smallest prime -/
noncomputable def nthSmallestPrime (s : ℕ) : ℕ := 
  (Finset.filter Nat.Prime (Finset.range (s * s))).toList.nthLe s sorry

theorem existence_of_product_of_smallest_primes :
  ∃ t s : ℕ, a t = (Finset.range s).prod (λ i => nthSmallestPrime (i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_product_of_smallest_primes_l877_87784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_inverse_x_sqrt_x_squared_plus_seven_l877_87794

open Real

theorem integral_of_inverse_x_sqrt_x_squared_plus_seven (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => -1 / Real.sqrt 7 * log ((Real.sqrt 7 + Real.sqrt (7 + x^2)) / x)) x =
    1 / (x * Real.sqrt (x^2 + 7)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_inverse_x_sqrt_x_squared_plus_seven_l877_87794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l877_87707

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Checks if a point lies on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a line in the form y = k(x-1) -/
structure Line where
  k : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem ellipse_theorem (e : Ellipse) (l : Line) :
  on_ellipse ⟨2, 0⟩ e ∧ 
  eccentricity e = Real.sqrt 2 / 2 →
  (∃ (m n : Point), 
    on_ellipse m e ∧ 
    on_ellipse n e ∧ 
    m.y = l.k * (m.x - 1) ∧ 
    n.y = l.k * (n.x - 1) ∧ 
    m ≠ n ∧
    triangle_area ⟨2, 0⟩ m n = Real.sqrt 10 / 3) →
  e.a = 2 ∧ e.b = Real.sqrt 2 ∧ (l.k = 1 ∨ l.k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l877_87707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87702

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Theorem statement
theorem f_properties :
  -- 1. f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- 2. f is decreasing on ℝ
  (∀ x y, x < y → f x > f y) ∧
  -- 3. For t ∈ (-1, 3], if f(t^2 - 2t) > f(-2t^2 + k), then k > 21
  (∀ t k, t > -1 ∧ t ≤ 3 → f (t^2 - 2*t) > f (-2*t^2 + k) → k > 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l877_87702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_chain_problem_l877_87780

/-- The length of a chain of springs with masses -/
noncomputable def spring_chain_length (n : ℕ) (l₀ : ℝ) (k : ℝ) (m : ℝ) (g : ℝ) : ℝ :=
  n * l₀ + (n * (n + 1) / 2) * (m * g / k)

/-- Theorem: The length of the specific spring chain is approximately 10.39 m -/
theorem spring_chain_problem :
  ∃ ε > 0, |spring_chain_length 10 0.5 200 2 9.8 - 10.39| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_chain_problem_l877_87780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l877_87783

noncomputable def f (x ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (A ω φ : ℝ) 
  (h1 : A > 0) (h2 : ω > 0) (h3 : |φ| < π / 2)
  (h4 : ∀ x, f (x + π / 2) ω φ = - f (x - π / 2) ω φ)
  (h5 : ∀ x, f (π / 6 + x) ω φ = f (π / 6 - x) ω φ) :
  ∀ x, f x ω φ = Real.sin (x + π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l877_87783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perry_tax_threshold_correct_l877_87772

/-- Represents the income tax calculation system -/
structure TaxSystem where
  lowRate : ℚ
  highRate : ℚ
  threshold : ℚ

/-- Calculates the tax for a given income and tax system -/
noncomputable def calculateTax (income : ℚ) (system : TaxSystem) : ℚ :=
  if income ≤ system.threshold then
    system.lowRate * income
  else
    system.lowRate * system.threshold + system.highRate * (income - system.threshold)

/-- Theorem stating that the given threshold is correct for Perry's tax situation -/
theorem perry_tax_threshold_correct (perryIncome perryTax : ℚ) :
  perryIncome = 10550 ∧ perryTax = 950 →
  ∃ (system : TaxSystem),
    system.lowRate = 8/100 ∧
    system.highRate = 10/100 ∧
    system.threshold = 5250 ∧
    calculateTax perryIncome system = perryTax :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perry_tax_threshold_correct_l877_87772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_3_prop_4_l877_87710

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

-- Proposition 3
theorem prop_3 (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ ∧ f x₁ = -1 ∧ f x₂ = -1 →
  ∃ k : ℤ, k ≠ 0 ∧ x₁ - x₂ = k * Real.pi :=
by sorry

-- Proposition 4
theorem prop_4 :
  ∀ x : ℝ, f (x + Real.pi/4) = f (-x - Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_3_prop_4_l877_87710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_voyage_solution_l877_87758

/-- Represents the ship's voyage parameters and cost function. -/
structure ShipVoyage where
  distance : ℝ  -- voyage distance in km
  other_cost : ℝ  -- other costs per hour in yuan
  fuel_cost_coeff : ℝ  -- coefficient for fuel cost

/-- Calculates the total cost of the voyage given a speed. -/
noncomputable def total_cost (v : ShipVoyage) (speed : ℝ) : ℝ :=
  (v.distance / speed) * (v.other_cost + v.fuel_cost_coeff * speed^2)

/-- The optimal speed and cost for the given voyage. -/
noncomputable def optimal_voyage (v : ShipVoyage) : ℝ × ℝ :=
  let optimal_speed := (v.other_cost / v.fuel_cost_coeff)^(1/3)
  let optimal_cost := total_cost v optimal_speed
  (optimal_speed, optimal_cost)

theorem optimal_voyage_solution :
  let v := ShipVoyage.mk 100 500 0.8
  optimal_voyage v = (25, 4000) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_voyage_solution_l877_87758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_comparison_l877_87787

-- Define the parameters
variable (a b : ℕ)

-- Define the perimeter of Triangle A
def perimeterA (a b : ℕ) : ℤ := 3 * (a^2 : ℤ) - 6 * (b : ℤ) + 8

-- Define the side lengths of Triangle B
def sideB1 (a b : ℕ) : ℤ := (a^2 : ℤ) - 2 * (b : ℤ)
def sideB2 (a b : ℕ) : ℤ := (a^2 : ℤ) - 3 * (b : ℤ)
def sideB3 (b : ℕ) : ℤ := -(b : ℤ) + 5

-- Define the perimeter of Triangle B
def perimeterB (a b : ℕ) : ℤ := sideB1 a b + sideB2 a b + sideB3 b

-- Theorem statement
theorem triangle_perimeter_comparison (a b : ℕ) :
  (sideB2 a b - sideB3 b = (a^2 : ℤ) - 2*(b : ℤ) - 5) →
  (perimeterA a b - perimeterB a b = 19) →
  (sideB3 b = -(b : ℤ) + 5) ∧
  (perimeterA a b > perimeterB a b) ∧
  (a = 4) := by
  sorry

#check triangle_perimeter_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_comparison_l877_87787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_eq_one_l877_87765

theorem product_of_roots_eq_one : 
  ∃ (r₁ r₂ : ℝ), (r₁^(Real.log r₁ / Real.log 5) = 25) ∧ 
                  (r₂^(Real.log r₂ / Real.log 5) = 25) ∧ 
                  (r₁ * r₂ = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_eq_one_l877_87765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l877_87752

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (d h₁ h₂ : ℝ) : ℝ := (1 / 2) * d * (h₁ + h₂)

/-- Theorem: The area of a quadrilateral with diagonal 22 cm and offsets 9 cm and 6 cm is 165 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 22 9 6 = 165 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp [mul_add, mul_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l877_87752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l877_87703

theorem min_value_expression (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + b + c : ℚ) / 2 - (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (a + b + c) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l877_87703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l877_87708

/-- The area of a regular octagon inscribed in a circle with radius s -/
noncomputable def regular_octagon_area (s : ℝ) : ℝ := 2 * s^2 * Real.sqrt 2

/-- Theorem: The area of a regular octagon inscribed in a circle with radius s is 2s²√2 -/
theorem regular_octagon_area_formula (s : ℝ) (h : s > 0) :
  regular_octagon_area s = 2 * s^2 * Real.sqrt 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l877_87708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l877_87777

/-- The curve C in the x-y plane --/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

/-- The line l in the x-y plane --/
def line_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 4}

/-- The distance from a point to a line --/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 + p.2 - 4| / Real.sqrt 2

/-- The theorem stating the maximum distance from curve C to line l --/
theorem max_distance_curve_to_line :
  ∃ (max_dist : ℝ), max_dist = 3 * Real.sqrt 2 ∧
  ∀ θ : ℝ, distance_to_line (curve_C θ) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l877_87777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l877_87764

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2 * y + 1 = 0

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Theorem: The slope of the line x + 2y + 1 = 0 is -1/2
theorem slope_of_line :
  ∃ (m b : ℝ), m = -1/2 ∧ b = -1/2 ∧
  ∀ (x y : ℝ), line_equation x y ↔ slope_intercept_form m b x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l877_87764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_journey_time_l877_87762

/-- Calculates the time taken for a journey with two legs -/
noncomputable def journey_time (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance1 / speed1 + distance2 / speed2) * 60

/-- Theorem: The squirrel's journey takes 84 minutes -/
theorem squirrel_journey_time :
  journey_time 2 5 3 3 = 84 := by
  -- Unfold the definition of journey_time
  unfold journey_time
  -- Perform the calculation
  simp [div_eq_mul_inv]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_journey_time_l877_87762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l877_87766

def M : Finset Char := {'a', 'b', 'c'}

theorem proper_subsets_count :
  Finset.card (Finset.powerset M \ {M}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l877_87766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l877_87799

theorem expression_simplification (x : ℕ) (h : x > 0) (h2 : x - 3 < 0) :
  (2 * x : ℚ) / (x^2 - 1) - 1 / (x - 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l877_87799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l877_87759

/-- Calculates the length of a platform given train length, speed, and time to pass. -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass : ℝ) :
  let train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
  let total_distance : ℝ := train_speed_mps * time_to_pass
  let platform_length : ℝ := total_distance - train_length
  train_length = 140 ∧ train_speed_kmph = 60 ∧ time_to_pass = 23.998080153587715 →
  abs (platform_length - 259.9680029317) < 0.000001 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l877_87759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l877_87734

-- Define vector a
noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)

-- Define the magnitude of vector b
def b_magnitude : ℝ := 1

-- Define the dot product of a and b
noncomputable def a_dot_b : ℝ := Real.sqrt 3

-- Theorem statement
theorem angle_between_vectors :
  let θ := Real.arccos (a_dot_b / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2)) * b_magnitude))
  θ = π / 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l877_87734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_angle_BPC_gt_90_eq_one_minus_pi_over_8_l877_87717

/-- A square with vertices A, B, C, and D -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- A point P inside the square -/
def P : ℝ × ℝ := sorry

/-- The angle BPC -/
def angle_BPC (s : Square) (p : ℝ × ℝ) : ℝ := sorry

/-- The probability that angle BPC is greater than 90 degrees -/
def prob_angle_BPC_gt_90 (s : Square) : ℝ := sorry

/-- Theorem: The probability that angle BPC is greater than 90 degrees is 1 - π/8 -/
theorem prob_angle_BPC_gt_90_eq_one_minus_pi_over_8 (s : Square) :
  prob_angle_BPC_gt_90 s = 1 - π / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_angle_BPC_gt_90_eq_one_minus_pi_over_8_l877_87717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_locus_l877_87785

/-- The critical value for parameter a -/
noncomputable def a₀ : ℝ := (3 * 2^(1/3)) / 2

/-- The parabola y = x^2 - a -/
noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 - a

/-- The hyperbola y = 1/x -/
noncomputable def hyperbola (x : ℝ) : ℝ := 1/x

/-- The number of intersection points between the parabola and hyperbola -/
noncomputable def num_intersections (a : ℝ) : ℕ :=
  sorry  -- This definition needs to be implemented

/-- The set of intersection points between the parabola and hyperbola -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = parabola a p.1 ∧ p.2 = hyperbola p.1}

/-- The center of the circumcircle of three points -/
noncomputable def circumcenter (p₁ p₂ p₃ : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- Definition of circumcenter calculation

/-- The locus of circumcenters for all valid a -/
def circumcenter_locus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a > a₀, ∃ p₁ p₂ p₃, p₁ ∈ intersection_points a ∧
    p₂ ∈ intersection_points a ∧ p₃ ∈ intersection_points a ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ ∧ p = circumcenter p₁ p₂ p₃}

theorem intersection_and_locus :
  (∀ a : ℝ, num_intersections a = 3 ↔ a > a₀) ∧
  circumcenter_locus = {p : ℝ × ℝ | p.1 = 1/2 ∧ p.2 < (2 - 3 * 2^(1/3)) / 4} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_locus_l877_87785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l877_87735

/-- Represents the average speed of a motorcyclist between two points -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Represents the problem of calculating the average speed of a motorcyclist -/
theorem motorcyclist_speed 
  (total_distance : ℝ)
  (total_time : ℝ)
  (distance_AB : ℝ)
  (time_AB : ℝ)
  (time_BC : ℝ)
  (distance_BC : ℝ)
  (h1 : total_distance = distance_AB + distance_BC)
  (h2 : average_speed total_distance total_time = 30)
  (h3 : distance_AB = 120)
  (h4 : time_AB = 3 * time_BC)
  (h5 : distance_BC = distance_AB / 2)
  (h6 : total_time = time_AB + time_BC) :
  average_speed distance_BC time_BC = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l877_87735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_arrangements_l877_87733

def total_students : ℕ := 360

def is_valid_arrangement (students_per_row : ℕ) : Bool :=
  students_per_row ≥ 18 &&
  (total_students / students_per_row) ≥ 12 &&
  total_students % students_per_row = 0

def possible_arrangements : List ℕ :=
  (List.range (total_students - 17)).filter is_valid_arrangement

theorem sum_of_possible_arrangements :
  List.sum possible_arrangements = 92 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_arrangements_l877_87733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_form_141_l877_87714

/-- Represents the sequence of digits formed by writing positive integers 
    with a first digit of 1 in increasing order -/
def digit_sequence : ℕ → ℕ := sorry

/-- Returns the nth digit in the sequence -/
def nth_digit (n : ℕ) : ℕ := digit_sequence n

/-- Checks if a number starts with the digit 1 -/
def starts_with_one (n : ℕ) : Prop :=
  (n.div (10^(n.log 10)) = 1)

/-- The sequence only contains numbers starting with 1 -/
axiom sequence_property : ∀ n, starts_with_one (digit_sequence n)

/-- The main theorem stating that the 1998th, 1999th, and 2000th digits form 141 -/
theorem digits_form_141 : 
  nth_digit 1998 = 1 ∧ nth_digit 1999 = 4 ∧ nth_digit 2000 = 1 :=
by
  sorry

#check digits_form_141

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_form_141_l877_87714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l877_87789

-- Define the slope angle of a line passing through two points
noncomputable def slopeAngle (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.arctan ((y2 - y1) / (x2 - x1))

-- Define the theorem
theorem slope_angle_specific_points :
  let angle := slopeAngle 3 0 4 (Real.sqrt 3)
  0 ≤ angle ∧ angle < π ∧ angle = π / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l877_87789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_l877_87770

-- Define the circles
noncomputable def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0
noncomputable def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
noncomputable def line_equation (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Define the length of the common chord
noncomputable def common_chord_length : ℝ := 24/5

-- Theorem statement
theorem intersection_chord (A B : ℝ × ℝ) :
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
  (line_equation A.1 A.2 ∧ line_equation B.1 B.2) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_l877_87770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_car_washes_l877_87704

/-- Represents the discount tiers for car washes -/
inductive DiscountTier
  | Tier1
  | Tier2
  | Tier3

/-- Returns the discount rate for a given tier -/
def discountRate (tier : DiscountTier) : Rat :=
  match tier with
  | DiscountTier.Tier1 => 9/10
  | DiscountTier.Tier2 => 8/10
  | DiscountTier.Tier3 => 7/10

/-- Returns the minimum number of car washes for a given tier -/
def minWashes (tier : DiscountTier) : Nat :=
  match tier with
  | DiscountTier.Tier1 => 10
  | DiscountTier.Tier2 => 15
  | DiscountTier.Tier3 => 20

/-- Returns the maximum number of car washes for a given tier -/
def maxWashes (tier : DiscountTier) : Nat :=
  match tier with
  | DiscountTier.Tier1 => 14
  | DiscountTier.Tier2 => 19
  | DiscountTier.Tier3 => 1000000  -- A large number instead of Nat.max_value

/-- The normal price of a single car wash -/
def normalPrice : Rat := 15

/-- The budget constraint -/
def budget : Rat := 250

/-- Calculates the cost of a given number of car washes in a specific tier -/
def cost (n : Nat) (tier : DiscountTier) : Rat :=
  n * normalPrice * discountRate tier

/-- Returns whether a given number of car washes is within the specified tier -/
def inTier (n : Nat) (tier : DiscountTier) : Prop :=
  minWashes tier ≤ n ∧ n ≤ maxWashes tier

/-- The main theorem: The maximum number of car washes that can be purchased is 23 -/
theorem max_car_washes :
  ∃ (n : Nat) (tier : DiscountTier),
    n = 23 ∧
    inTier n tier ∧
    cost n tier ≤ budget ∧
    ∀ (m : Nat) (t : DiscountTier),
      inTier m t → cost m t ≤ budget → m ≤ n :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_car_washes_l877_87704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_number_in_triangular_arrangement_l877_87773

/-- Represents the triangular arrangement of numbers -/
def TriangularArrangement (n : ℕ) : ℕ → ℕ → Prop :=
  λ row col ↦ (row - 1)^2 + col = n ∧ col ≤ row

/-- The row containing a given number -/
noncomputable def RowOf (n : ℕ) : ℕ := 
  Nat.ceil (Real.sqrt (n : ℝ))

theorem adjacent_number_in_triangular_arrangement :
  ∀ (n : ℕ), n = 350 →
  ∃ (row col : ℕ),
    TriangularArrangement n row col ∧
    TriangularArrangement 314 (row - 1) col :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_number_in_triangular_arrangement_l877_87773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l877_87781

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin (α/2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l877_87781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l877_87719

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (2 * t.A) = -1/3)
  (h2 : t.c = Real.sqrt 3)
  (h3 : Real.sin t.A = Real.sqrt 6 * Real.sin t.C) :
  -- Part I
  t.a = 3 * Real.sqrt 2 ∧
  -- Part II (assuming A is acute)
  (0 < t.A ∧ t.A < Real.pi/2 → 
    t.b = 5 ∧ 
    (1/2 * t.b * t.c * Real.sin t.A) = (5 * Real.sqrt 2) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l877_87719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l877_87739

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_problem (simple_principal : ℝ) 
  (simple_rate simple_time compound_rate compound_time : ℝ) :
  simple_principal = 3225 ∧
  simple_rate = 8 ∧
  simple_time = 5 ∧
  compound_rate = 15 ∧
  compound_time = 2 ∧
  (simple_interest simple_principal simple_rate simple_time) * 2 =
    compound_interest 1600 compound_rate compound_time :=
by sorry

#check interest_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l877_87739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_colorings_l877_87711

/-- Represents a coloring of a 7x7 chessboard with 2 yellow squares -/
def Coloring := Fin 49 × Fin 49

/-- Two colorings are equivalent if they can be obtained from each other by rotation -/
def coloring_equiv (c1 c2 : Coloring) : Prop := sorry

/-- The set of all distinct colorings under rotation equivalence -/
def distinct_colorings : Finset Coloring := sorry

/-- The number of distinct colorings is 300 -/
theorem chessboard_colorings :
  Finset.card distinct_colorings = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_colorings_l877_87711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_probabilities_l877_87778

/-- Given a set of cards with 9 hearts, 10 spades, and 11 diamonds, prove various probabilities and conditions. -/
theorem card_probabilities (m : ℕ) (h_m : m > 6) :
  let total_cards := 9 + 10 + 11
  let prob_heart := (9 : ℚ) / total_cards
  let remaining_spades := 10 - m
  let remaining_cards := 11 + remaining_spades
  let prob_diamond := (11 : ℚ) / remaining_cards
  (
    /- 1. Probability of drawing a heart -/
    prob_heart = 3 / 10 ∧
    /- 2. Condition for certain diamond draw -/
    (prob_diamond = 1 ↔ m = 10) ∧
    /- 3. Conditions for random diamond draw -/
    (prob_diamond < 1 ↔ m ∈ Finset.range 3 ∪ {9}) ∧
    /- 4. Minimum probability of drawing a diamond in random scenario -/
    (m = 7 → prob_diamond = 11 / 14)
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_probabilities_l877_87778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_18_4851_to_hundredth_l877_87732

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_18_4851_to_hundredth :
  roundToHundredth 18.4851 = 18.49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_18_4851_to_hundredth_l877_87732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_eq_neg_half_l877_87776

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- Define the base case for n = 0
  | (n + 1) => (1 + a n) / (1 - a n)

/-- The 15th term of the sequence equals -1/2 -/
theorem a_15_eq_neg_half : a 15 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_eq_neg_half_l877_87776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l877_87715

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the external point
def P : ℝ × ℝ := (1, 2)

-- Define the potential tangent lines
def tangent_line1 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0
def tangent_line2 (x : ℝ) : Prop := x = 1

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ x y, circle_eq x y → ¬(tangent_line1 x y ∧ x = P.1 ∧ y = P.2)) ∧
  (∀ x y, circle_eq x y → ¬(tangent_line2 x ∧ x = P.1 ∧ y = P.2)) ∧
  (∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ tangent_line1 p.1 p.2 ∧ 
    (∀ q : ℝ × ℝ, circle_eq q.1 q.2 → 
      ((p.1 - P.1)^2 + (p.2 - P.2)^2 ≤ (q.1 - P.1)^2 + (q.2 - P.2)^2))) ∧
  (∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ tangent_line2 p.1 ∧
    (∀ q : ℝ × ℝ, circle_eq q.1 q.2 → 
      ((p.1 - P.1)^2 + (p.2 - P.2)^2 ≤ (q.1 - P.1)^2 + (q.2 - P.2)^2))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l877_87715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l877_87756

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x - 1 else 1 - x

-- State the theorem
theorem solution_set_of_inequality (h : ∀ x, f (-x) = f x) :
  {x : ℝ | f (x - 1) > 1} = {x : ℝ | x < -1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l877_87756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l877_87720

-- Define the sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 8}
def B : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 2 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l877_87720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_decreasing_l877_87718

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sqrt (sequence_a n ^ 2 + 1 / sequence_a n)

noncomputable def sequence_b (n : ℕ) : ℝ := sequence_a (n + 1) - sequence_a n

theorem sequence_b_decreasing :
  ∀ n : ℕ, sequence_b n > sequence_b (n + 1) := by
  sorry

#check sequence_a
#check sequence_b
#check sequence_b_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_decreasing_l877_87718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nested_absolute_difference_l877_87716

def nested_absolute_difference (a : Fin 1990 → ℕ) : ℕ :=
  (List.range 1990).foldl (λ acc i => Int.natAbs (acc - a i.succ)) (a 0)

theorem max_nested_absolute_difference :
  ∃ (a : Fin 1990 → ℕ), Function.Bijective a ∧ (∀ i, a i ∈ Set.range (λ j : Fin 1990 => j.val + 1)) ∧
    (∀ (b : Fin 1990 → ℕ), Function.Bijective b → (∀ i, b i ∈ Set.range (λ j : Fin 1990 => j.val + 1)) →
      nested_absolute_difference a ≥ nested_absolute_difference b) ∧
    nested_absolute_difference a = 1989 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nested_absolute_difference_l877_87716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_15_times_5_11_l877_87700

theorem digits_of_2_15_times_5_11 : 
  (Nat.log 10 (2^15 * 5^11) + 1 : ℕ) = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_15_times_5_11_l877_87700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_right_quadrilateral_l877_87725

/-- Represents a quadrilateral with right angles at F and H -/
structure RightQuadrilateral :=
  (EF FG EH HG : ℝ)

/-- The area of a right quadrilateral -/
noncomputable def area (q : RightQuadrilateral) : ℝ := q.EF * q.FG / 2 + q.EH * q.HG / 2

/-- Theorem: Area of quadrilateral EFGH is 4√3 -/
theorem area_of_right_quadrilateral (q : RightQuadrilateral) :
  q.EF^2 + q.FG^2 = 16 →
  q.EH^2 + q.HG^2 = 16 →
  (q.EF = 2 ∨ q.FG = 2 ∨ q.EH = 2 ∨ q.HG = 2) →
  (q.EF ≠ q.FG ∧ q.EF ≠ q.EH ∧ q.EF ≠ q.HG ∧
   q.FG ≠ q.EH ∧ q.FG ≠ q.HG ∧ q.EH ≠ q.HG) →
  area q = 4 * Real.sqrt 3 := by
  sorry

#check area_of_right_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_right_quadrilateral_l877_87725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_powers_sum_20_factorial_l877_87709

theorem highest_powers_sum_20_factorial : ∃ (a b : ℕ), 
  (∀ k : ℕ, (4^(a+1) : ℕ) ∣ Nat.factorial 20 → (4^k : ℕ) ∣ Nat.factorial 20 → k ≤ a) ∧ 
  (∀ k : ℕ, (6^(b+1) : ℕ) ∣ Nat.factorial 20 → (6^k : ℕ) ∣ Nat.factorial 20 → k ≤ b) ∧ 
  a + b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_powers_sum_20_factorial_l877_87709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l877_87769

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given a rectangle ABCD with a point P inside, 
    if PA = 5, PD = 7, PC = 8, and AB = 9, then PB = √10 -/
theorem rectangle_point_distance 
  (ABCD : Rectangle) 
  (P : Point) 
  (h1 : distance P ABCD.A = 5)
  (h2 : distance P ABCD.D = 7)
  (h3 : distance P ABCD.C = 8)
  (h4 : distance ABCD.A ABCD.B = 9) :
  distance P ABCD.B = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l877_87769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l877_87792

def sequence_product (a : ℕ → ℕ) (n : ℕ) : ℕ := (Finset.range n).prod (λ i => a (i + 1))

theorem sequence_formula (a : ℕ → ℕ) :
  (∀ n : ℕ, n ≥ 2 → sequence_product a n = n^2) →
  (∀ n : ℕ, n ≥ 2 → a n = 2*n - 1) :=
by
  intro h
  intro n hn
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l877_87792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sunny_days_probability_l877_87775

/-- The probability of getting exactly 2 sunny days out of 5 days, 
    where the probability of rain each day is 0.75 -/
theorem two_sunny_days_probability : 
  let n : ℕ := 5  -- Total number of days
  let k : ℕ := 2  -- Number of desired sunny days
  let p : ℝ := 3/4  -- Probability of rain each day
  (Finset.range n).powersetCard k |>.sum
    (λ s => (1 - p) ^ k * p ^ (n - k)) = 135 / 512 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sunny_days_probability_l877_87775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_current_age_l877_87722

-- Define variables
def jill_age : ℕ → Prop := sorry
def roger_age : ℕ → Prop := sorry
def finley_age : ℕ := 40

-- Define conditions
axiom roger_age_relation : ∀ j, jill_age j → roger_age (2 * j + 5)

axiom future_age_difference : 
  ∀ j, jill_age j → roger_age (2 * j + 5) → 
    (2 * j + 5) - j = (finley_age + 15) - 30

-- Theorem to prove
theorem jill_current_age : ∃ j, jill_age j ∧ j = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_current_age_l877_87722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l877_87705

/-- The polynomial z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- Predicate to check if f(z) divides z^k - 1 -/
def divides (k : ℕ) : Prop := ∀ z : ℂ, ∃ q : ℂ, z^k - 1 = q * (f z)

theorem smallest_k_divides : 
  (divides 182) ∧ (∀ k : ℕ, 0 < k → k < 182 → ¬(divides k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l877_87705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_shoveling_time_l877_87786

def snow_volume (width length depth : ℝ) : ℝ := width * length * depth

def shoveling_rate (initial_rate : ℝ) (hour : ℕ) : ℝ := initial_rate - 2 * (hour.pred : ℝ)

def snow_removed (initial_rate : ℝ) (hours : ℕ) : ℝ :=
  (Finset.range hours).sum (λ h ↦ shoveling_rate initial_rate (h + 1))

theorem driveway_shoveling_time 
  (width length depth initial_rate : ℝ)
  (h_width : width = 5)
  (h_length : length = 12)
  (h_depth : depth = 2.5)
  (h_initial_rate : initial_rate = 25) :
  ∃ (hours : ℕ), 
    hours = 9 ∧ 
    snow_removed initial_rate hours ≥ snow_volume width length depth ∧
    snow_removed initial_rate (hours - 1) < snow_volume width length depth := by
  sorry

#eval snow_volume 5 12 2.5
#eval snow_removed 25 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_shoveling_time_l877_87786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_BG_FG_l877_87712

-- Define the rectangle ABCD
structure Rectangle (A B C D : ℝ × ℝ) : Prop where
  is_rectangle : (B.1 - A.1) * (C.1 - D.1) + (B.2 - A.2) * (C.2 - D.2) = 0
                 ∧ (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

-- Define midpoints
def Midpoint (P Q M : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define projection
def Projection (P Q R S : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, S = (R.1 + t * (Q.1 - R.1), R.2 + t * (Q.2 - R.2))
  ∧ (P.1 - S.1) * (Q.1 - R.1) + (P.2 - S.2) * (Q.2 - R.2) = 0

-- Define perpendicularity
def Perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- Theorem statement
theorem perpendicular_BG_FG 
  (A B C D E F G : ℝ × ℝ) 
  (h_rect : Rectangle A B C D) 
  (h_mid_E : Midpoint A B E) 
  (h_mid_F : Midpoint C D F) 
  (h_proj : Projection E A C G) : 
  Perpendicular B G F G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_BG_FG_l877_87712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oddProductProbabilityIsOne_l877_87753

/-- Represents a region on the dartboard -/
structure Region where
  isInner : Bool
  value : Nat

/-- Represents the dartboard configuration -/
def Dartboard : List Region :=
  [
    { isInner := true, value := 3 },
    { isInner := true, value := 4 },
    { isInner := true, value := 4 },
    { isInner := false, value := 1 },
    { isInner := false, value := 3 },
    { isInner := false, value := 3 }
  ]

/-- The probability of hitting any region on the dartboard -/
noncomputable def hitProbability (r : Region) : ℝ :=
  if r.isInner then 1 / 12 else 1 / 4

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- Calculates the probability of getting an odd product score -/
noncomputable def oddProductProbability : ℝ :=
  let oddRegions := Dartboard.filter (fun r => isOdd r.value)
  let oddProbability := oddRegions.map hitProbability |> List.sum
  oddProbability * oddProbability

theorem oddProductProbabilityIsOne : oddProductProbability = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oddProductProbabilityIsOne_l877_87753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_is_correct_l877_87749

/-- The minimum width of a rectangle with an area of at least 150 sq. ft and length twice its width -/
noncomputable def min_width : ℝ := 5 * Real.sqrt 3

/-- The theorem stating that min_width is the correct minimum width -/
theorem min_width_is_correct :
  ∀ w : ℝ, w > 0 → 2 * w^2 ≥ 150 → w ≥ min_width :=
by
  intro w hw h150
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check min_width_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_is_correct_l877_87749
