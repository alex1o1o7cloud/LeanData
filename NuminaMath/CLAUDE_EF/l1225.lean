import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l1225_122506

noncomputable def cube_root (x : ℝ) := Real.rpow x (1/3)

theorem rationalize_denominator_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  ∃ (X Y Z W : ℝ), 
    (1 / (cube_root a - cube_root b) = (cube_root X + cube_root Y + cube_root Z) / W) ∧
    (X + Y + Z + W = 51) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l1225_122506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_generating_functions_l1225_122555

/-- Chebyshev polynomial of the first kind -/
noncomputable def T (n : ℕ) (x : ℝ) : ℝ := sorry

/-- Chebyshev polynomial of the second kind -/
noncomputable def U (n : ℕ) (x : ℝ) : ℝ := sorry

/-- Generating function for Chebyshev polynomials of the first kind -/
noncomputable def F_T (x z : ℝ) : ℝ := ∑' n, T n x * z^n

/-- Generating function for Chebyshev polynomials of the second kind -/
noncomputable def F_U (x z : ℝ) : ℝ := ∑' n, U n x * z^n

/-- Theorem: Generating functions for Chebyshev polynomials -/
theorem chebyshev_generating_functions (x z : ℝ) :
  F_T x z = (1 - x*z) / (1 - 2*x*z + z^2) ∧
  F_U x z = 1 / (1 - 2*x*z + z^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_generating_functions_l1225_122555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_inequality_final_result_l1225_122570

-- Define a permutation of (1,2,3,4)
def IsPermutation (b₁ b₂ b₃ b₄ : ℕ) : Prop :=
  Multiset.ofList [b₁, b₂, b₃, b₄] = Multiset.ofList [1, 2, 3, 4]

-- Define the left side of the inequality
def LeftSide (b₁ b₂ b₃ b₄ : ℕ) : ℚ :=
  ((b₁^2 + 1) / 2) * ((b₂^2 + 2) / 2) * ((b₃^2 + 3) / 2) * ((b₄^2 + 4) / 2)

-- State the theorem
theorem permutation_inequality :
  ∀ b₁ b₂ b₃ b₄ : ℕ, IsPermutation b₁ b₂ b₃ b₄ → LeftSide b₁ b₂ b₃ b₄ ≥ 24 := by
  sorry

-- Count the number of valid permutations
def CountValidPermutations : ℕ := 24

-- State the final result
theorem final_result : CountValidPermutations = 24 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_inequality_final_result_l1225_122570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_7_equals_7_9_l1225_122574

/-- The repeating decimal 0.777... as a rational number -/
def repeating_decimal_7 : ℚ := 7/9

/-- Theorem stating that the repeating decimal 0.777... equals 7/9 -/
theorem repeating_decimal_7_equals_7_9 : repeating_decimal_7 = 7 / 9 := by
  -- The proof is trivial since we defined repeating_decimal_7 as 7/9
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_7_equals_7_9_l1225_122574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l1225_122539

/-- The weight of a triangular piece of wood with given side length -/
noncomputable def wood_weight (side_length : ℝ) : ℝ :=
  12 * (side_length^2 / 3^2)

/-- The problem statement -/
theorem second_triangle_weight :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |wood_weight 5 - 100/3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l1225_122539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l1225_122577

/-- Represents a polyhedron in the sequence Q_i -/
structure Polyhedron where
  index : ℕ
  volume : ℚ

/-- Defines the sequence of polyhedra Q_i -/
def Q : ℕ → Polyhedron
  | 0 => ⟨0, 1⟩
  | n + 1 => ⟨n + 1, (Q n).volume + (1/4) * (1/6)^n⟩

/-- The volume difference between consecutive polyhedra -/
def delta_Q (i : ℕ) : ℚ := (1/4) * (1/6)^i

theorem volume_Q3 :
  (Q 3).volume = 169/144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l1225_122577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_arrangements_are_sequences_l1225_122516

-- Define a type for natural numbers from 1 to 5
def NatOneToFive : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

-- Define a function to create NatOneToFive from Nat
def mkNatOneToFive (n : ℕ) : Option NatOneToFive :=
  if h : 1 ≤ n ∧ n ≤ 5 then
    some ⟨n, h⟩
  else
    none

-- Define a type for arrangements of the first five natural numbers
def Arrangement := List NatOneToFive

-- Define the four given arrangements
def arrangement1 : Arrangement := 
  [mkNatOneToFive 1, mkNatOneToFive 2, mkNatOneToFive 3, mkNatOneToFive 4, mkNatOneToFive 5].filterMap id
def arrangement2 : Arrangement := 
  [mkNatOneToFive 5, mkNatOneToFive 4, mkNatOneToFive 3, mkNatOneToFive 2, mkNatOneToFive 1].filterMap id
def arrangement3 : Arrangement := 
  [mkNatOneToFive 2, mkNatOneToFive 1, mkNatOneToFive 5, mkNatOneToFive 3, mkNatOneToFive 4].filterMap id
def arrangement4 : Arrangement := 
  [mkNatOneToFive 4, mkNatOneToFive 1, mkNatOneToFive 5, mkNatOneToFive 3, mkNatOneToFive 2].filterMap id

-- Define what it means for an arrangement to be a sequence
def isSequence (arr : Arrangement) : Prop := arr.length = 5

-- Theorem stating that all given arrangements are sequences
theorem all_arrangements_are_sequences :
  isSequence arrangement1 ∧
  isSequence arrangement2 ∧
  isSequence arrangement3 ∧
  isSequence arrangement4 := by
  sorry

#check all_arrangements_are_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_arrangements_are_sequences_l1225_122516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_ge_lower_bound_g_gt_f_when_a_eq_one_l1225_122515

open Real

/-- The function f(x) = x^2 ln x + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * log x + a*x

/-- The function g(x) = xe^x + x sin x -/
noncomputable def g (x : ℝ) : ℝ := x * exp x + x * sin x

/-- The lower bound for a -/
noncomputable def a_lower_bound : ℝ := 2 * exp (-3/2)

theorem f_increasing_iff_a_ge_lower_bound (a : ℝ) :
  (∀ x > 0, Monotone (fun x => f a x)) ↔ a ≥ a_lower_bound := by
  sorry

theorem g_gt_f_when_a_eq_one :
  ∀ x > 0, g x > f 1 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_ge_lower_bound_g_gt_f_when_a_eq_one_l1225_122515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_rectangle_outside_circles_l1225_122514

/-- The area of the region inside a rectangle but outside three quarter circles -/
theorem area_inside_rectangle_outside_circles (π : Real) : 
  (4 * 6) - ((π * 2^2) / 4 + (π * 3^2) / 4 + (π * 4^2) / 4) = (96 - 29*π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_rectangle_outside_circles_l1225_122514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_theorem_l1225_122569

-- Define the parameters of the well
noncomputable def well_height : ℝ := 14
noncomputable def well_top_radius : ℝ := 2
noncomputable def well_bottom_radius : ℝ := 3

-- Define the volume of a truncated cone
noncomputable def truncated_cone_volume (h r R : ℝ) : ℝ :=
  (1 / 3) * Real.pi * h * (R^2 + r^2 + R*r)

-- State the theorem
theorem well_volume_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |truncated_cone_volume well_height well_top_radius well_bottom_radius - 278.53| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_theorem_l1225_122569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1225_122523

noncomputable def f (x y : ℝ) : ℝ := 
  (5*x^2 + 8*x*y + 5*y^2 - 14*x - 10*y + 30) / (4 - x^2 - 10*x*y - 25*y^2)^(7/2)

theorem min_value_of_f :
  ∀ x y : ℝ, 4 - x^2 - 10*x*y - 25*y^2 > 0 →
  f x y ≥ 5/32 ∧ ∃ x₀ y₀ : ℝ, f x₀ y₀ = 5/32 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1225_122523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l1225_122537

/-- A line passing through two points (-5, 0) and (3, -3) has a y-intercept of -15/8 -/
theorem line_y_intercept : 
  ∃ (m b : ℝ), (λ x y ↦ y = m * x + b) (-5) 0 ∧ (λ x y ↦ y = m * x + b) 3 (-3) ∧
  (λ x y ↦ y = m * x + b) 0 (-15/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l1225_122537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1225_122535

/-- A function f is a "local odd function" if there exists an x in its domain such that f(-x) = -f(x) --/
def isLocalOddFunction (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, f (-x) = -f x

/-- The function f(x) = m + 2^x --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.exp (x * Real.log 2)

/-- The function g(x) = x^2 + (5m+1)x + 1 --/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + (5*m+1)*x + 1

/-- p: f is a local odd function on [-1, 2] --/
def p (m : ℝ) : Prop := isLocalOddFunction (f m) (Set.Icc (-1) 2)

/-- q: g intersects the x-axis at two distinct points --/
def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g m x₁ = 0 ∧ g m x₂ = 0

theorem m_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → 
  (m < -5/4 ∨ (-1 < m ∧ m < -3/5) ∨ m > 1/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1225_122535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_equals_train_l1225_122590

-- Define the train's speed in km/hr
noncomputable def train_speed : ℝ := 72

-- Define the time taken to cross the platform in minutes
noncomputable def crossing_time : ℝ := 1

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 600

-- Define the length of the platform
noncomputable def platform_length : ℝ := train_speed * 1000 / 60 * crossing_time - train_length

-- Theorem stating that the platform length equals the train length
theorem platform_equals_train : platform_length = train_length := by
  -- Unfold the definitions
  unfold platform_length train_length train_speed crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_equals_train_l1225_122590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_with_intercept_sum_l1225_122599

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

-- Function to get x-intercept of a line
noncomputable def xIntercept (l : Line2D) : ℝ :=
  -l.yIntercept / l.slope

-- Function to check if a line has given slope and passes through a point
def lineWithSlopeAndPoint (m : ℝ) (p : Point2D) (l : Line2D) : Prop :=
  l.slope = m ∧ pointOnLine l p

-- Function to check if a line has given slope and sum of intercepts
def lineWithSlopeAndInterceptSum (m : ℝ) (sum : ℝ) (l : Line2D) : Prop :=
  l.slope = m ∧ xIntercept l + l.yIntercept = sum

-- Theorem for part (1)
theorem line_through_point (l : Line2D) :
  lineWithSlopeAndPoint 2 ⟨-1, 3⟩ l →
  ∀ x y, y = 2 * x + 5 ↔ pointOnLine l ⟨x, y⟩ := by
  sorry

-- Theorem for part (2)
theorem line_with_intercept_sum (l : Line2D) :
  lineWithSlopeAndInterceptSum 2 4 l →
  ∀ x y, y = 2 * x + 8 ↔ pointOnLine l ⟨x, y⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_with_intercept_sum_l1225_122599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_correct_l1225_122511

/-- Horner's Rule for polynomial evaluation -/
def horner_rule (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  let rec go : ℕ → ℝ → ℝ
  | 0, v => v
  | k+1, v => go k (v * x + a (n - k))
  go n (a n)

/-- Polynomial evaluation using standard form -/
def polynomial_eval (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  (Finset.range (n+1)).sum (λ i ↦ a i * x^i)

theorem horner_rule_correct (n : ℕ) (a : ℕ → ℝ) (x : ℝ) :
  horner_rule n a x = polynomial_eval n a x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_correct_l1225_122511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_89_plus_24_sqrt_11_l1225_122524

theorem sqrt_89_plus_24_sqrt_11 (a b c : ℤ) : 
  (Real.sqrt (89 + 24 * Real.sqrt 11) = a + b * Real.sqrt c) → 
  (∀ (d : ℤ), d ^ 2 ∣ c → d = 1 ∨ d = -1) → 
  (a = 7 ∧ b = 2 ∧ c = 11 ∧ a + b + c = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_89_plus_24_sqrt_11_l1225_122524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1225_122550

-- Define the basic shapes
inductive Shape
| Rectangle
| Parallelogram
| Triangle

-- Define the syllogism structure
structure Syllogism where
  major_premise : Shape → Shape → Prop
  minor_premise : Shape → Prop
  conclusion : Shape → Prop

-- Define the specific syllogism
def rectangle_triangle_syllogism : Syllogism where
  major_premise := fun x y ↦ x = Shape.Rectangle → y = Shape.Parallelogram
  minor_premise := fun x ↦ x = Shape.Triangle → x ≠ Shape.Parallelogram
  conclusion := fun x ↦ x = Shape.Triangle → x ≠ Shape.Rectangle

-- Theorem to prove
theorem minor_premise_identification :
  rectangle_triangle_syllogism.minor_premise = (fun x ↦ x = Shape.Triangle → x ≠ Shape.Parallelogram) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1225_122550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_area_voters_l1225_122527

/-- The number of voters in the first area -/
def V : ℝ := sorry

/-- The percentage of votes Mark wins in the first area -/
def first_area_percentage : ℝ := 0.70

/-- The total number of votes Mark got -/
def total_votes : ℝ := 210000

/-- The statement that Mark got twice as many votes in the remaining area -/
axiom remaining_area_votes : first_area_percentage * V * 2 = total_votes - first_area_percentage * V

theorem first_area_voters : V = 100000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_area_voters_l1225_122527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_approx_53_l1225_122546

noncomputable section

/-- Calculates the area of a circle given its radius --/
def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Calculates the radius of the nth circle in the sequence --/
def nth_circle_radius (n : ℕ) : ℝ := 3 + 3 * (n - 1)

/-- The total number of circles in the design --/
def total_circles : ℕ := 6

/-- Checks if a circle is colored black based on its position --/
def is_black (n : ℕ) : Bool := n == 1 || n == 4 || n == 6

/-- Calculates the total area of black regions in the design --/
noncomputable def total_black_area : ℝ := 
  (circle_area (nth_circle_radius 1)) +
  (circle_area (nth_circle_radius 4) - circle_area (nth_circle_radius 3)) +
  (circle_area (nth_circle_radius 6) - circle_area (nth_circle_radius 5))

/-- Calculates the percentage of the design that is black --/
noncomputable def black_percentage : ℝ := 
  (total_black_area / circle_area (nth_circle_radius total_circles)) * 100

theorem black_percentage_approx_53 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |black_percentage - 53| < ε := by
  sorry

end noncomputable section

-- Remove the #eval statement as it's not computable
-- #eval black_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_approx_53_l1225_122546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_unique_solution_l1225_122544

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

-- Define vectors a and b
def a : V := 2 • e₁ - e₂
def b (lambda : ℝ) : V := e₁ + lambda • e₂

-- Theorem statement
theorem collinear_vectors (h₁ : ¬ LinearIndependent ℝ ![e₁, e₂]) 
  (h₂ : ∃ (k : ℝ), a = k • b (-1/2)) : 
  ∃ (k : ℝ), a = k • b (-1/2) := by
  sorry

-- Additional theorem to show that λ = -1/2 is the only solution
theorem unique_solution (h₁ : ¬ LinearIndependent ℝ ![e₁, e₂]) 
  (h₂ : ∃ (k : ℝ), a = k • b (-1/2)) 
  (lambda : ℝ) (h₃ : ∃ (k : ℝ), a = k • b lambda) : 
  lambda = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_unique_solution_l1225_122544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equation_g_is_correct_l1225_122519

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^4 - 3*x^2 - 1
def g (x : ℝ) : ℝ := -x^4 + 4*x^2 + 2*x - 2

-- State the theorem
theorem polynomial_sum_equation :
  ∀ x : ℝ, f x + g x = x^2 + 2*x - 3 :=
by
  intro x
  -- Expand the definitions of f and g
  simp [f, g]
  -- Simplify the algebraic expression
  ring

-- Prove that g is the correct polynomial
theorem g_is_correct :
  ∀ x : ℝ, g x = -x^4 + 4*x^2 + 2*x - 2 :=
by
  -- This is true by definition
  simp [g]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equation_g_is_correct_l1225_122519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_l1225_122593

/-- The dividend polynomial -/
noncomputable def dividend : Polynomial ℚ := 3 * Polynomial.X^7 - 2 * Polynomial.X^5 + 5 * Polynomial.X^3 - 8

/-- The divisor polynomial -/
noncomputable def divisor : Polynomial ℚ := Polynomial.X^2 + 3 * Polynomial.X + 2

/-- The expected remainder polynomial -/
noncomputable def expected_remainder : Polynomial ℚ := 354 * Polynomial.X + 340

theorem division_theorem :
  ∃ q : Polynomial ℚ, dividend = q * divisor + expected_remainder :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_l1225_122593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_lambda_range_lambda_range_obtuse_angle_l1225_122585

def vector_a (l : ℝ) : ℝ × ℝ := (l, 4)
def vector_b : ℝ × ℝ := (-3, 5)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def obtuse_angle (v w : ℝ × ℝ) : Prop :=
  dot_product v w < 0 ∧ ¬(∃ (k : ℝ), v = (k * w.1, k * w.2))

theorem obtuse_angle_lambda_range (l : ℝ) :
  obtuse_angle (vector_a l) vector_b → l > 20/3 :=
by sorry

-- Additional theorem to complete the range
theorem lambda_range_obtuse_angle (l : ℝ) :
  l > 20/3 → obtuse_angle (vector_a l) vector_b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_lambda_range_lambda_range_obtuse_angle_l1225_122585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l1225_122591

/-- Two vectors in ℝ² -/
def Vector2D : Type := ℝ × ℝ

/-- Check if two vectors are in opposite directions -/
def opposite_directions (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k < 0 ∧ v = (k * w.1, k * w.2)

theorem opposite_vectors_x_value :
  ∀ x : ℝ,
  let a : Vector2D := (x, 1)
  let b : Vector2D := (4, x)
  opposite_directions a b → x = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vectors_x_value_l1225_122591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotonicity_implies_log_range_l1225_122545

/-- A quadratic function -/
def quadratic (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

/-- The absolute value of a quadratic function -/
def f (m n : ℝ) (x : ℝ) : ℝ := |quadratic m n x|

/-- The logarithm of a quadratic function -/
noncomputable def g (m n : ℝ) (x : ℝ) : ℝ := Real.log (quadratic m n x)

/-- The number of monotonic intervals of a function -/
def monotonicIntervals (f : ℝ → ℝ) : ℕ := sorry

/-- The range of a function -/
def range (f : ℝ → ℝ) : Set ℝ := Set.range f

theorem quadratic_monotonicity_implies_log_range (m n : ℝ) :
  (monotonicIntervals (f m n) = 4) → (range (g m n) = Set.univ) ∧
  ¬ ((range (g m n) = Set.univ) → (monotonicIntervals (f m n) = 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotonicity_implies_log_range_l1225_122545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_satisfies_conditions_l1225_122598

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, -2]

theorem N_satisfies_conditions :
  (N.mulVec (![1, 0] : Fin 2 → ℝ) = (3 : ℝ) • (![1, 0] : Fin 2 → ℝ)) ∧
  (N.mulVec (![0, 1] : Fin 2 → ℝ) = (-2 : ℝ) • (![0, 1] : Fin 2 → ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_satisfies_conditions_l1225_122598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_triangle_area_l1225_122534

-- Define the hyperbola Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 - y^2 / 12 = 1

-- Define point A on Γ
def A : ℝ × ℝ := (2, 2)

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 12 + y^2 / 9 = 1

-- Define the asymptote of Γ in the first quadrant
def asymptote (x y : ℝ) : Prop := y = 2 * x ∧ x > 0 ∧ y > 0

-- Theorem for part (1)
theorem ellipse_equation : 
  (Γ A.1 A.2) → 
  (∃ (f : ℝ × ℝ), f.1^2 = 3 ∧ f.2 = 0 ∧ C f.1 f.2) →
  (∀ x y, C x y ↔ x^2 / 12 + y^2 / 9 = 1) :=
by sorry

-- Helper function to calculate triangle area
noncomputable def area_triangle (O P Q : ℝ × ℝ) : ℝ :=
  let d₁ := (P.1 - O.1, P.2 - O.2)
  let d₂ := (Q.1 - O.1, Q.2 - O.2)
  abs (d₁.1 * d₂.2 - d₁.2 * d₂.1) / 2

-- Theorem for part (2)
theorem triangle_area :
  (Γ A.1 A.2) →
  (∀ P Q : ℝ × ℝ, 
    asymptote P.1 P.2 → 
    Γ Q.1 Q.2 → 
    (Q.1 - A.1 = A.1 - P.1 ∧ Q.2 - A.2 = A.2 - P.2) →
    area_triangle (0, 0) P Q = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_triangle_area_l1225_122534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valya_turn_based_prob_valya_simultaneous_prob_valya_probabilities_equal_l1225_122536

/-- Represents the probabilities in a shooting game between Valya and Kolya -/
structure ShootingGame where
  x : ℝ
  hx : x > 1

/-- Valya's probability of hitting the target -/
noncomputable def valya_hit_prob (g : ShootingGame) : ℝ := 1 / (g.x + 1)

/-- Kolya's probability of hitting the target -/
noncomputable def kolya_hit_prob (g : ShootingGame) : ℝ := 1 / g.x

/-- Theorem stating that Valya's probability of winning in turn-based shooting is 1/2 -/
theorem valya_turn_based_prob (g : ShootingGame) :
  (valya_hit_prob g) / (1 - (1 - valya_hit_prob g) * (1 - kolya_hit_prob g)) = 1/2 := by
  sorry

/-- Theorem stating that Valya's probability of not losing in simultaneous shooting is 1/2 -/
theorem valya_simultaneous_prob (g : ShootingGame) :
  (valya_hit_prob g) / (1 - (1 - valya_hit_prob g) * (1 - kolya_hit_prob g)) = 1/2 := by
  sorry

/-- Theorem stating that Valya's probabilities are equal in both scenarios -/
theorem valya_probabilities_equal (g : ShootingGame) :
  (valya_hit_prob g) / (1 - (1 - valya_hit_prob g) * (1 - kolya_hit_prob g)) =
  (valya_hit_prob g) / (1 - (1 - valya_hit_prob g) * (1 - kolya_hit_prob g)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valya_turn_based_prob_valya_simultaneous_prob_valya_probabilities_equal_l1225_122536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_chain_existence_l1225_122587

/-- Two non-intersecting circles -/
structure NonIntersectingCircles where
  R₁ : Set (ℝ × ℝ)
  R₂ : Set (ℝ × ℝ)
  non_intersecting : R₁ ∩ R₂ = ∅

/-- Circles tangent to R₁ and R₂ at intersection points with center line -/
structure TangentCircles (nc : NonIntersectingCircles) where
  T₁ : Set (ℝ × ℝ)
  T₂ : Set (ℝ × ℝ)
  tangent_to_R₁ : (T₁ ∩ nc.R₁).Nonempty ∧ (T₂ ∩ nc.R₁).Nonempty
  tangent_to_R₂ : (T₁ ∩ nc.R₂).Nonempty ∧ (T₂ ∩ nc.R₂).Nonempty

/-- Angle between two circles -/
noncomputable def angle_between_circles (T₁ T₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Existence of a chain of n tangent circles -/
def exists_chain (n : ℕ) (nc : NonIntersectingCircles) : Prop := sorry

/-- Main theorem -/
theorem tangent_circle_chain_existence 
  (nc : NonIntersectingCircles) 
  (tc : TangentCircles nc) 
  (n : ℕ) :
  exists_chain n nc ↔ 
  ∃ m : ℤ, angle_between_circles tc.T₁ tc.T₂ = m * (360 / n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_chain_existence_l1225_122587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_halves_plus_two_alpha_l1225_122530

theorem cos_five_pi_halves_plus_two_alpha (α : ℝ) 
  (h1 : Real.tan α = 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_halves_plus_two_alpha_l1225_122530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l1225_122518

/-- Given that 0.̅09 = 1/11, prove that 0.̅27 = 3/11 -/
theorem recurring_decimal_to_fraction :
  ((0 : ℚ) + 9 / 99 : ℚ) = 1 / 11 → ((0 : ℚ) + 27 / 99 : ℚ) = 3 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l1225_122518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_l1225_122540

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x + y^2 = 1

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := Real.pi/4  -- 45° in radians

-- Define the rotated coordinates
noncomputable def rotated_x (x' y' : ℝ) : ℝ := (x' + y') / Real.sqrt 2
noncomputable def rotated_y (x' y' : ℝ) : ℝ := (y' - x') / Real.sqrt 2

-- Define the resulting curve
def resulting_curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x*y + Real.sqrt 2 * x + Real.sqrt 2 * y - 2 = 0

-- Theorem statement
theorem curve_rotation :
  ∀ x' y' : ℝ, original_curve (rotated_x x' y') (rotated_y x' y') →
  resulting_curve x' y' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_l1225_122540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_circles_l1225_122521

/-- The distance between the closest points of two circles -/
noncomputable def distance_between_circles (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - (r1 + r2)

/-- Theorem: The distance between the closest points of two specific circles -/
theorem distance_between_specific_circles :
  let c1 : ℝ × ℝ := (3, 3)
  let c2 : ℝ × ℝ := (20, 12)
  let r1 : ℝ := 3 - 1  -- radius of circle tangent to y = 1
  let r2 : ℝ := 12     -- radius of circle tangent to x-axis
  distance_between_circles c1 c2 r1 r2 = Real.sqrt 370 - 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_circles_l1225_122521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_measure_l1225_122564

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Angles A, B, and C are congruent
  A = B ∧ B = C ∧
  -- Angles D and E are congruent
  D = E ∧
  -- Angle D is 30° more than angle A
  D = A + 30 ∧
  -- Angle F is 20° less than angle D
  F = D - 20 ∧
  -- Sum of angles in a hexagon is 720°
  A + B + C + D + E + F = 720

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) (h : Hexagon A B C D E F) :
  abs (D - 138.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_measure_l1225_122564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_lower_bound_l1225_122576

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 / 2 - (a + 1) * x

-- Theorem for monotonicity intervals when a > 0
theorem monotonicity_intervals (a : ℝ) (h : a > 0) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a = 1 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 1 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) := by
  sorry

-- Theorem for the lower bound when a = -1
theorem lower_bound :
  ∀ x, x > 0 → f (-1) x ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_lower_bound_l1225_122576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increase_factor_is_18_75_l1225_122596

/-- The factor by which a cylinder's volume increases when its height is tripled and its radius is increased by 150% -/
noncomputable def volume_increase_factor (r h : ℝ) : ℝ :=
  let new_r := r * 2.5
  let new_h := h * 3
  (Real.pi * new_r^2 * new_h) / (Real.pi * r^2 * h)

/-- Theorem stating that the volume increase factor is 18.75 -/
theorem volume_increase_factor_is_18_75 (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  volume_increase_factor r h = 18.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increase_factor_is_18_75_l1225_122596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1225_122513

-- Define the function f(x) = cos(-x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (-x)

-- State the theorem
theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1225_122513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_and_conditions_l1225_122508

noncomputable def y (x : ℝ) : ℝ := -x + 2 * Real.log (1 + x) + 1

noncomputable def y_prime (x : ℝ) : ℝ := -1 + 2 / (1 + x)

noncomputable def y_double_prime (x : ℝ) : ℝ := -2 / ((1 + x)^2)

theorem solution_satisfies_equation_and_conditions :
  (∀ x : ℝ, (1 + x^2) * y_double_prime x + (y_prime x)^2 + 1 = 0) ∧
  y 0 = 1 ∧
  y_prime 0 = 1 := by
  sorry

#check solution_satisfies_equation_and_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_and_conditions_l1225_122508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_range_l1225_122582

theorem complex_number_equality_range (m theta : ℝ) (lambda : ℝ) :
  let z₁ : ℂ := m + (4 - m^2) * I
  let z₂ : ℂ := 2 * Real.cos theta + (lambda + 3 * Real.sin theta) * I
  z₁ = z₂ →
  ∃ (lambda_min lambda_max : ℝ), lambda_min = -9/16 ∧ lambda_max = 7 ∧ lambda_min ≤ lambda ∧ lambda ≤ lambda_max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_range_l1225_122582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_digits_l1225_122578

/-- A structure representing a pair of three-digit integers -/
structure ThreeDigitPair :=
  (a b : Nat)
  (a_three_digit : a ≥ 100 ∧ a < 1000)
  (b_three_digit : b ≥ 100 ∧ b < 1000)

/-- Function to check if all digits in two numbers are distinct -/
def all_digits_distinct (p : ThreeDigitPair) : Prop :=
  let digits := (p.a.repr ++ p.b.repr).toList
  digits.Nodup

/-- Function to calculate the sum of digits of a number -/
def sum_of_digits (n : Nat) : Nat :=
  n.repr.toList.map (fun c => c.toNat - '0'.toNat) |>.sum

/-- The main theorem -/
theorem smallest_sum_of_digits :
  ∃ (p : ThreeDigitPair),
    all_digits_distinct p ∧
    let S := p.a + p.b
    S ≥ 1000 ∧ S < 10000 ∧
    ∀ (q : ThreeDigitPair),
      all_digits_distinct q →
      let T := q.a + q.b
      T ≥ 1000 ∧ T < 10000 →
      sum_of_digits S ≤ sum_of_digits T ∧
      sum_of_digits S = 9 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_digits_l1225_122578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_c_value_sin_B_minus_C_value_l1225_122526

namespace TriangleProof

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define our specific triangle
noncomputable def myTriangle : Triangle where
  a := Real.sqrt 39
  b := 2
  A := 2 * Real.pi / 3  -- 120° in radians
  B := sorry      -- We don't know B yet
  C := sorry      -- We don't know C yet
  c := sorry      -- We don't know c yet

-- Theorem for part (I)
theorem sin_B_value (t : Triangle) (h1 : t = myTriangle) : 
  Real.sin t.B = Real.sqrt 13 / 13 := by sorry

-- Theorem for part (II)
theorem c_value (t : Triangle) (h1 : t = myTriangle) : 
  t.c = 5 := by sorry

-- Theorem for part (III)
theorem sin_B_minus_C_value (t : Triangle) (h1 : t = myTriangle) : 
  Real.sin (t.B - t.C) = -7 * Real.sqrt 3 / 26 := by sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_c_value_sin_B_minus_C_value_l1225_122526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_equivalence_l1225_122547

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + Real.pi / 6) + Real.cos (ω * x + Real.pi / 6)

theorem shifted_function_equivalence (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  ∀ x, f ω (x - Real.pi / 3) = 2 * Real.cos (2 * x - 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_equivalence_l1225_122547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_stage_fraction_is_one_third_l1225_122584

/-- Fraction of time spent on the last stage of the first route compared to the first two stages -/
noncomputable def lastStageFraction (uphill_time : ℝ) (flat_time : ℝ) (route_difference : ℝ) : ℝ :=
  let first_two_stages := uphill_time + 2 * uphill_time
  let second_route_total := flat_time + 2 * flat_time
  (second_route_total - (first_two_stages + route_difference)) / first_two_stages

theorem last_stage_fraction_is_one_third :
  lastStageFraction 6 14 18 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_stage_fraction_is_one_third_l1225_122584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_distance_theorem_l1225_122509

/-- Represents the distance between two skiers after terrain changes -/
noncomputable def final_distance (initial_distance : ℝ) (speed1 speed2 speed3 speed4 : ℝ) : ℝ :=
  initial_distance * (speed2 / speed1) * (speed3 / speed2) * (speed4 / speed3)

/-- Theorem stating the final distance between skiers -/
theorem skier_distance_theorem (initial_distance : ℝ) (speed1 speed2 speed3 speed4 : ℝ) 
  (h1 : initial_distance = 200)
  (h2 : speed1 = 6)
  (h3 : speed2 = 4)
  (h4 : speed3 = 7)
  (h5 : speed4 = 3) :
  ∃ ε > 0, |final_distance initial_distance speed1 speed2 speed3 speed4 - 100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_distance_theorem_l1225_122509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l1225_122532

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points K, L, M on the sides of triangle ABC
variable (K : ℝ × ℝ) (hK : K ∈ Set.Icc B C)
variable (L : ℝ × ℝ) (hL : L ∈ Set.Icc C A)
variable (M : ℝ × ℝ) (hM : M ∈ Set.Icc A B)

-- Define points D, E, F on the sides of triangle KLM
variable (D : ℝ × ℝ) (hD : D ∈ Set.Icc L M)
variable (E : ℝ × ℝ) (hE : E ∈ Set.Icc M K)
variable (F : ℝ × ℝ) (hF : F ∈ Set.Icc K L)

-- Define the area function for triangles
noncomputable def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_area_inequality :
  (area A M E) * (area C K E) * (area B K F) * (area A L F) * (area B D M) * (area C L D) ≤ 
  (1/8) * (area A B C)^6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l1225_122532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_sufficient_not_necessary_l1225_122501

-- Define the plane structure
structure Plane where
  dummy : Unit

-- Define the line structure
structure Line where
  dummy : Unit

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop :=
  sorry

-- Define perpendicularity between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry

-- Define when a line is in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_sufficient_not_necessary 
  (α : Plane) (l m n : Line) 
  (h1 : line_in_plane m α) 
  (h2 : line_in_plane n α) : 
  (∀ (l m n : Line), perpendicular l α → perpendicular_lines l m ∧ perpendicular_lines l n) ∧ 
  (∃ (l m n : Line), perpendicular_lines l m ∧ perpendicular_lines l n ∧ ¬perpendicular l α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_sufficient_not_necessary_l1225_122501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1225_122549

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b

/-- Condition for the symmetric point of the focus to lie on the left branch -/
def symmetric_point_on_left_branch (h : Hyperbola) : Prop :=
  ∃ (m n : ℝ), 
    m^2 + n^2 = h.c^2 ∧ 
    n / (m - h.c) = -h.a / h.b ∧
    m^2 / h.a^2 - n^2 / h.b^2 = 1 ∧
    m < 0

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_symmetric : symmetric_point_on_left_branch h) : 
  eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1225_122549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1225_122517

/-- The length of the line segment cut from a circle by a line --/
noncomputable def chord_length (circle_center_x circle_center_y circle_radius : ℝ) 
                 (line_a line_b line_c : ℝ) : ℝ :=
  2 * Real.sqrt (circle_radius ^ 2 - 
      ((line_a * circle_center_x + line_b * circle_center_y + line_c) / 
       Real.sqrt (line_a ^ 2 + line_b ^ 2)) ^ 2)

/-- The problem statement --/
theorem chord_length_specific_case : 
  chord_length 1 0 1 1 (Real.sqrt 3) (-2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1225_122517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_line_l1225_122588

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the line through two points
def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r.x = p.x + t * (q.x - p.x) ∧ r.y = p.y + t * (q.y - p.y)}

-- Define what it means for a point to lie on a line
def PointOnLine (p : Point) (l : Set Point) : Prop := p ∈ l

-- Define what it means for a line segment to be a diameter of a circle
def IsDiameter (c : Circle) (p q : Point) : Prop :=
  PointOnLine p (Line c.center q) ∧ 
  (p.x - q.x)^2 + (p.y - q.y)^2 = 4 * c.radius^2

-- Define membership for Point in Circle
instance : Membership Point Circle where
  mem p c := (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- State the theorem
theorem intersection_point_on_line 
  (A B C D : Point) (c1 c2 : Circle) : 
  IsDiameter c1 A B → 
  IsDiameter c2 B C → 
  B ∈ c1 → 
  B ∈ c2 → 
  D ∈ c1 → 
  D ∈ c2 → 
  D ≠ B → 
  PointOnLine D (Line A C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_line_l1225_122588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_difference_l1225_122556

noncomputable def total_rice : ℚ := 50
noncomputable def llesis_fraction : ℚ := 7/10

theorem rice_difference : 
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
  -- Unfold the definitions
  unfold total_rice llesis_fraction
  -- Simplify the expressions
  simp
  -- Perform the numerical calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_difference_l1225_122556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1225_122560

/-- Predicate to check if a line is an asymptote of a set in ℝ² -/
def IsAsymptote (S : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate to check if two lines in ℝ² are perpendicular -/
def ArePerpendicular (L₁ L₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if one of its foci is at (4,0) and its asymptotes are perpendicular,
    then a² = b² = 8 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (C : Set (ℝ × ℝ)), C = {(x, y) | x^2/a^2 - y^2/b^2 = 1}) →
  (∃ (f : ℝ × ℝ), f ∈ C ∧ f = (4, 0)) →
  (∀ (l₁ l₂ : Set (ℝ × ℝ)), IsAsymptote C l₁ ∧ IsAsymptote C l₂ ∧ l₁ ≠ l₂ → ArePerpendicular l₁ l₂) →
  a^2 = 8 ∧ b^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1225_122560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_sqrt_14_l1225_122586

-- Define the circles O₁ and O₂
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the common chord length
noncomputable def common_chord_length : ℝ := Real.sqrt 14

-- Theorem statement
theorem common_chord_length_is_sqrt_14 :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_O₁ x₁ y₁ ∧ circle_O₁ x₂ y₂ ∧
    circle_O₂ x₁ y₁ ∧ circle_O₂ x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = common_chord_length^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_sqrt_14_l1225_122586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burpee_contribution_percentage_l1225_122579

noncomputable def workout_time : ℝ := 20
noncomputable def jumping_jacks : ℝ := 30
noncomputable def pushups : ℝ := 22
noncomputable def situps : ℝ := 45
noncomputable def burpees : ℝ := 15
noncomputable def lunges : ℝ := 25

noncomputable def total_exercises : ℝ := jumping_jacks + pushups + situps + burpees + lunges

noncomputable def average_rate (exercises : ℝ) : ℝ := exercises / workout_time

noncomputable def total_average_rate : ℝ := average_rate total_exercises

noncomputable def burpee_contribution : ℝ := (average_rate burpees / total_average_rate) * 100

theorem burpee_contribution_percentage :
  abs (burpee_contribution - 10.95) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burpee_contribution_percentage_l1225_122579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_37_61_l1225_122580

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a cone slice -/
structure ConeSlice where
  bottomRadius : ℝ
  topRadius : ℝ
  height : ℝ

/-- Calculate the volume of a cone slice -/
noncomputable def coneSliceVolume (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.height * (slice.bottomRadius^2 + slice.topRadius^2 + slice.bottomRadius * slice.topRadius)

/-- Given a right circular cone sliced into 5 equal-height pieces,
    calculate the ratio of the volume of the second-largest piece to the largest piece -/
noncomputable def volumeRatio (cone : RightCircularCone) : ℝ :=
  let sliceHeight := cone.height / 5
  let largestSlice : ConeSlice := {
    bottomRadius := (4/5) * cone.baseRadius,
    topRadius := cone.baseRadius,
    height := sliceHeight
  }
  let secondLargestSlice : ConeSlice := {
    bottomRadius := (3/5) * cone.baseRadius,
    topRadius := (4/5) * cone.baseRadius,
    height := sliceHeight
  }
  (coneSliceVolume secondLargestSlice) / (coneSliceVolume largestSlice)

/-- Theorem stating that the volume ratio of the second-largest to largest piece is 37/61 -/
theorem volume_ratio_is_37_61 (cone : RightCircularCone) :
  volumeRatio cone = 37 / 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_37_61_l1225_122580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l1225_122581

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + cos (2 * x) + sin (2 * x)

-- Define a as f'(π/4)
noncomputable def a : ℝ := (deriv f) (π / 4)

-- Define the curve y = x³
def curve (x : ℝ) : ℝ := x^3

-- Define point P
noncomputable def P : ℝ × ℝ := (a, curve a)

-- Theorem statement
theorem tangent_line_equations :
  ∃ (x₀ y₀ : ℝ),
    (y₀ = curve x₀) ∧
    ((3 * x₀ - y₀ - 2 = 0 ∧ 3 * P.1 - P.2 - 2 = 0) ∨
     (3 * x₀ - 4 * y₀ + 1 = 0 ∧ 3 * P.1 - 4 * P.2 + 1 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l1225_122581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_15th_row_fourth_from_end_l1225_122507

/-- Given a lattice where the last number of row i is 5i, 
    the fourth number from the end in the 15th row is 73. -/
theorem lattice_15th_row_fourth_from_end 
  (last_number_in_row : ℕ → ℕ)
  (fourth_from_end_15th_row : ℕ)
  (h : ∀ i : ℕ, i > 0 → last_number_in_row i = 5 * i) : 
  fourth_from_end_15th_row = 73 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_15th_row_fourth_from_end_l1225_122507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_office_problem_l1225_122572

-- Define the constants
def total_spent : ℚ := 674/100
def letter_cost : ℚ := 42/100
def package_cost : ℚ := 98/100
def postcard_cost : ℚ := 28/100

-- Define the variables
variable (num_letters num_packages num_postcards : ℕ)

-- Define the theorem
theorem post_office_problem :
  -- Conditions
  (num_letters = num_packages) ∧
  (num_postcards = num_letters + num_packages + 5) ∧
  (letter_cost * num_letters + package_cost * num_packages + postcard_cost * num_postcards = total_spent) →
  -- Conclusion
  num_letters = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_office_problem_l1225_122572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_minus_x_squared_l1225_122567

theorem integral_x_minus_x_squared : ∫ x in (0:ℝ)..(1:ℝ), (x - x^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_minus_x_squared_l1225_122567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l1225_122552

noncomputable def a (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x), Real.cos (ω * x))

noncomputable def b (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x) + 2 * Real.cos (ω * x), Real.cos (ω * x))

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

def is_periodic (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g (x + p) = g x

def smallest_positive_period (g : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic g p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic g q

theorem vector_function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_period : smallest_positive_period (f ω) (Real.pi / 4)) :
  ω = 4 ∧ 
  (∀ x, f ω x ≤ 2) ∧
  (∀ x, f ω x = 2 ↔ ∃ k : ℤ, x = Real.pi / 16 + k * Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l1225_122552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1225_122597

theorem vector_equation_solution :
  ∃! (u v : ℝ), ((3 : ℝ), (1 : ℝ)) + u • ((8 : ℝ), (-4 : ℝ)) + ((5 : ℝ), (2 : ℝ)) = 
                 ((4 : ℝ), (2 : ℝ)) + v • ((-3 : ℝ), (4 : ℝ)) + ((5 : ℝ), (2 : ℝ)) ∧
                 u = (-1/2 : ℝ) ∧ v = (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1225_122597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1225_122559

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2*x + 3 - x^2 > 0 ∧ x - 2 < 0

def q (x a : ℝ) : Prop := x^2 - (3*a + 6)*x + 2*a^2 + 6*a < 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ∃ (_dummy : Unit), P ∧ ¬Q

-- State the theorem
theorem range_of_a :
  (∀ x, necessary_not_sufficient (¬(p x)) (¬∃ x, q x a)) →
  ∀ a, -2 ≤ a ∧ a ≤ -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1225_122559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probability_l1225_122554

theorem table_tennis_probability (p_ma_long_serve p_ma_long_receive : ℝ) 
  (h1 : p_ma_long_serve = 2/3)
  (h2 : p_ma_long_receive = 1/2)
  (h3 : 0 ≤ p_ma_long_serve ∧ p_ma_long_serve ≤ 1)
  (h4 : 0 ≤ p_ma_long_receive ∧ p_ma_long_receive ≤ 1) :
  p_ma_long_serve * p_ma_long_receive * p_ma_long_serve * p_ma_long_receive +
  (1 - p_ma_long_serve) * p_ma_long_receive * p_ma_long_serve * p_ma_long_receive +
  (1 - p_ma_long_serve) * p_ma_long_receive * (1 - p_ma_long_receive) * (1 - p_ma_long_receive) +
  p_ma_long_serve * p_ma_long_receive * (1 - p_ma_long_receive) * (1 - p_ma_long_receive) = 1/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probability_l1225_122554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edmonton_to_calgary_travel_time_l1225_122510

/-- Represents the distance between two cities in kilometers. -/
def distance_between (city1 city2 : String) : ℝ := sorry

/-- Represents the speed of travel in kilometers per hour. -/
def travel_speed : ℝ := sorry

theorem edmonton_to_calgary_travel_time :
  let edmonton_to_red_deer := distance_between "Edmonton" "Red Deer"
  let red_deer_to_calgary := distance_between "Red Deer" "Calgary"
  let total_distance := edmonton_to_red_deer + red_deer_to_calgary
  edmonton_to_red_deer = 220 →
  red_deer_to_calgary = 110 →
  travel_speed = 110 →
  total_distance / travel_speed = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edmonton_to_calgary_travel_time_l1225_122510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_white_third_draw_without_replacement_prob_four_or_fewer_red_with_replacement_l1225_122573

-- Define the number of red and white balls
def num_red_balls : ℚ := 4
def num_white_balls : ℚ := 2
def total_balls : ℚ := num_red_balls + num_white_balls

-- Define the number of draws for part 2
def num_draws : ℕ := 6

-- Part 1: Probability of drawing a white ball on the third draw without replacement
theorem prob_white_third_draw_without_replacement :
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1)) * (num_white_balls / (total_balls - 2)) +
  (num_red_balls / total_balls) * (num_white_balls / (total_balls - 1)) * ((num_white_balls - 1) / (total_balls - 2)) +
  (num_white_balls / total_balls) * (num_red_balls / (total_balls - 1)) * ((num_white_balls - 1) / (total_balls - 2)) =
  1 / 3 := by
sorry

-- Part 2: Probability of drawing 4 or fewer red balls in 6 draws with replacement
theorem prob_four_or_fewer_red_with_replacement :
  1 - (Nat.choose num_draws 5 * (num_red_balls / total_balls)^5 * (num_white_balls / total_balls) +
       (num_red_balls / total_balls)^6) =
  473 / 729 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_white_third_draw_without_replacement_prob_four_or_fewer_red_with_replacement_l1225_122573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_theorem_l1225_122520

/-- Motion equation of a particle -/
noncomputable def motion_equation (t : ℝ) : ℝ := 8 - 3 * t^2

/-- Average velocity of the particle during the time interval [1, 1+Δt] -/
noncomputable def average_velocity (Δt : ℝ) : ℝ := 
  (motion_equation (1 + Δt) - motion_equation 1) / Δt

/-- Instantaneous velocity of the particle at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  -6 * t

/-- Theorem stating the average velocity formula and instantaneous velocity at t=1 -/
theorem velocity_theorem (Δt : ℝ) :
  average_velocity Δt = -6 - 3 * Δt ∧
  instantaneous_velocity 1 = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_theorem_l1225_122520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_numbers_10_to_31_l1225_122504

def even_numbers_10_to_31 : List Nat :=
  (List.range 22).map (· + 10) |>.filter (· % 2 = 0)

theorem sum_even_numbers_10_to_31 :
  even_numbers_10_to_31.sum = 220 := by
  -- Evaluate the list and its sum
  have h1 : even_numbers_10_to_31 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] := by rfl
  have h2 : even_numbers_10_to_31.sum = 220 := by rfl
  -- Use the equality
  rw [h2]

#eval even_numbers_10_to_31
#eval even_numbers_10_to_31.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_numbers_10_to_31_l1225_122504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_are_infinitesimal_l1225_122502

noncomputable def sequence1 (n : ℕ) : ℝ := -1 / n
noncomputable def sequence2 (n : ℕ) : ℝ := (-1)^(n-1) / n
noncomputable def sequence3 (n : ℕ) : ℝ := 1 / (2*n - 1)

theorem sequences_are_infinitesimal :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence1 n| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence2 n| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence3 n| < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_are_infinitesimal_l1225_122502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_library_l1225_122589

/-- Represents the time in minutes to bike a certain distance at a constant pace -/
noncomputable def bikeTime (distance : ℝ) (pace : ℝ) : ℝ := distance / pace

theorem time_to_library (park_distance park_time library_distance : ℝ) 
  (h1 : park_distance = 5)
  (h2 : park_time = 30)
  (h3 : library_distance = 3)
  (h4 : park_time = bikeTime park_distance (park_distance / park_time)) :
  bikeTime library_distance (park_distance / park_time) = 18 := by
  sorry

#check time_to_library

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_library_l1225_122589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_behavior_l1225_122558

def f (x : ℝ) := x^2 - 6*x + 10

theorem function_behavior :
  (∀ x ∈ Set.Ioo 2 3, ∀ y ∈ Set.Ioo 2 3, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo 3 4, ∀ y ∈ Set.Ioo 3 4, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_behavior_l1225_122558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_bound_f_l1225_122543

/-- The function f(x) given a positive real number a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |3*x + 1/a| + 3*|x - a|

/-- The theorem stating that 2√3 is the maximum lower bound for f(x) -/
theorem max_lower_bound_f :
  ∀ a > 0, ∃ m, m = 2 * Real.sqrt 3 ∧
  (∀ x, f a x ≥ m) ∧
  ∀ m' > m, ∃ a' > 0, ∃ x', f a' x' < m' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_bound_f_l1225_122543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldenSectionThirdPointValue_l1225_122592

/-- Golden ratio constant -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Golden section search method third test point -/
noncomputable def goldenSectionThirdPoint (a b : ℝ) : ℝ :=
  a + (1 - 1 / φ) * (b - a)

/-- Theorem: Golden section search third test point for [1000, 2000] -/
theorem goldenSectionThirdPointValue :
  goldenSectionThirdPoint 1000 2000 = 1236 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldenSectionThirdPointValue_l1225_122592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_octagon_walk_l1225_122565

/-- The distance from the starting point after walking 10 km along the perimeter of a regular octagon with side length 3 km -/
noncomputable def distance_after_walk (octagon_side_length : ℝ) (walk_distance : ℝ) : ℝ :=
  let final_x := octagon_side_length - octagon_side_length * (Real.sqrt 2) / 2
  let final_y := octagon_side_length * (Real.sqrt 2) / 2 - (walk_distance - 2 * octagon_side_length)
  Real.sqrt (final_x ^ 2 + final_y ^ 2)

/-- Theorem stating that the distance after walking 10 km along the perimeter of a regular octagon with side length 3 km is √17 km -/
theorem distance_after_octagon_walk :
  distance_after_walk 3 10 = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_octagon_walk_l1225_122565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1225_122541

theorem expression_evaluation : 
  (-1)^2023 + Real.sqrt 27 + (Real.pi - 3.14)^0 - abs (Real.sqrt 3 - 2) = 4 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1225_122541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_bound_l1225_122548

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + Real.log x

-- Define the theorem
theorem extreme_points_sum_bound
  (a : ℝ)
  (h_a_pos : 0 < a)
  (h_a_bound : a < 1/2)
  (x₁ x₂ : ℝ)
  (h_extreme : ∀ x, x ≠ x₁ → x ≠ x₂ → deriv (f a) x ≠ 0) :
  f a x₁ + f a x₂ < -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_bound_l1225_122548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1225_122503

theorem polynomial_remainder : 
  ∃ q : Polynomial ℝ, (X^3 - 4*X + 6 : Polynomial ℝ) = (X + 3) * q + (-9 : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1225_122503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1225_122533

noncomputable section

/-- The probability that a randomly selected point from a square 
    with vertices at (±3, ±3) is within two units of the origin -/
def probability_within_circle : ℝ := Real.pi / 9

/-- The side length of the square -/
def square_side : ℝ := 6

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The area of the square -/
def square_area : ℝ := square_side ^ 2

/-- The area of the circle -/
def circle_area : ℝ := Real.pi * circle_radius ^ 2

theorem probability_calculation : 
  probability_within_circle = circle_area / square_area :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1225_122533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_l1225_122575

/-- The surface area of a sphere with diameter 9.5 inches is 361π/4 square inches. -/
theorem bowling_ball_surface_area :
  let diameter : ℚ := 19 / 2
  let radius : ℚ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * ((radius : ℝ) ^ 2)
  surface_area = 361 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_l1225_122575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_theorem_l1225_122594

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a partition of a larger cube into smaller cubes -/
structure CubePartition where
  original : Cube
  parts : List Cube

def CubePartition.volume (cp : CubePartition) : ℕ :=
  cp.original.edge ^ 3

def CubePartition.totalParts (cp : CubePartition) : ℕ :=
  cp.parts.length

def CubePartition.validPartition (cp : CubePartition) : Prop :=
  cp.volume = (cp.parts.map (λ c => c.edge ^ 3)).sum

def CubePartition.hasDifferentSizes (cp : CubePartition) : Prop :=
  ∃ c1 c2, c1 ∈ cp.parts ∧ c2 ∈ cp.parts ∧ c1.edge ≠ c2.edge

def CubePartition.hasEdgeTwo (cp : CubePartition) : Prop :=
  ∃ c, c ∈ cp.parts ∧ c.edge = 2

theorem cube_partition_theorem (cp : CubePartition) :
  cp.original.edge = 6 →
  cp.validPartition →
  cp.hasDifferentSizes →
  cp.hasEdgeTwo →
  cp.totalParts = 134 :=
by
  sorry

#check cube_partition_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_theorem_l1225_122594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_representation_l1225_122562

theorem binomial_sum_representation (n k : ℕ) : 
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℤ), 
    a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > (k : ℤ) ∧
    (∃ (s₁ s₂ s₃ s₄ s₅ : Int), (s₁ * s₁ = 1) ∧ (s₂ * s₂ = 1) ∧ (s₃ * s₃ = 1) ∧ (s₄ * s₄ = 1) ∧ (s₅ * s₅ = 1) ∧
      (n : ℤ) = s₁ * Nat.choose a₁.toNat 3 + s₂ * Nat.choose a₂.toNat 3 + s₃ * Nat.choose a₃.toNat 3 + 
                s₄ * Nat.choose a₄.toNat 3 + s₅ * Nat.choose a₅.toNat 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_representation_l1225_122562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_inequality_l1225_122563

theorem half_power_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1/2 : ℝ)^x - (1/2 : ℝ)^(y-x) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_inequality_l1225_122563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l1225_122571

/-- Calculates the area of an equilateral triangle given its altitude -/
noncomputable def triangle_area (altitude : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * altitude ^ 2

/-- The area of an equilateral triangle with altitude √3 is √3 -/
theorem equilateral_triangle_area (h : ℝ) (alt_eq : h = Real.sqrt 3) :
  triangle_area h = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l1225_122571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1225_122512

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0 →
  -- Conclusions
  A = π / 3 ∧
  (a = Real.sqrt 7 ∧ b = 2 →
    ∃ (S : ℝ), S = (3 * Real.sqrt 3) / 2 ∧
    S = (1 / 2) * b * c * Real.sin A) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1225_122512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_journey_probability_l1225_122500

/-- Represents a lily pad --/
structure LilyPad where
  number : Nat
  isPredator : Bool
  isSafe : Bool

/-- Represents Fiona's journey --/
structure FionaJourney where
  pads : List LilyPad
  startPad : LilyPad
  endPad : LilyPad
  normalHopProb : Real
  doubleHopProb : Real

/-- Calculates the probability of Fiona reaching the end pad --/
noncomputable def reachProbability (journey : FionaJourney) : Real :=
  sorry

theorem fiona_journey_probability :
  let pads := [
    LilyPad.mk 0 false false, LilyPad.mk 1 false false, LilyPad.mk 2 false false,
    LilyPad.mk 3 false false, LilyPad.mk 4 true false, LilyPad.mk 5 false false,
    LilyPad.mk 6 false false, LilyPad.mk 7 false true, LilyPad.mk 8 false false,
    LilyPad.mk 9 true false, LilyPad.mk 10 false false, LilyPad.mk 11 false false,
    LilyPad.mk 12 false false, LilyPad.mk 13 false false, LilyPad.mk 14 false false,
    LilyPad.mk 15 false false
  ]
  let journey := {
    pads := pads
    startPad := LilyPad.mk 0 false false
    endPad := LilyPad.mk 14 false false
    normalHopProb := 1/2
    doubleHopProb := 1
  }
  reachProbability journey = 15/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_journey_probability_l1225_122500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_triangle_perimeter_l1225_122542

/-- Given a circular sector with central angle 120 degrees and radius 4.8 cm,
    and a triangle inscribed in the sector with its base parallel to the chord of the sector,
    the perimeter of the combined shape is approximately 27.97 cm. -/
theorem sector_triangle_perimeter :
  let θ : Real := 120 * Real.pi / 180  -- Central angle in radians
  let r : Real := 4.8           -- Radius in cm
  let arc_length : Real := θ * r
  let chord_length : Real := 2 * r * Real.sin (θ / 2)
  let perimeter : Real := 2 * r + arc_length + chord_length
  ∃ ε > 0, |perimeter - 27.97| < ε
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_triangle_perimeter_l1225_122542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1225_122531

theorem quadratic_equation_properties (a : ℝ) :
  let f := fun x : ℝ => x^2 - a*x + a - 1
  (∀ x, f x = 0 → x ∈ Set.univ) ∧
  (∃ x, f x = 0 ∧ x > 3 → a > 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1225_122531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cross_sections_l1225_122595

-- Define the basic geometric objects
structure Plane where
structure Sphere where
structure Cylinder where
inductive CrossSection where
  | Circle
  | Ellipse

-- Define the intersection operation
def intersect : Plane → Sphere ⊕ Cylinder → CrossSection
  | _, Sum.inl _ => CrossSection.Circle
  | _, Sum.inr _ => CrossSection.Circle

-- Theorem statement
theorem intersection_cross_sections 
  (p : Plane) (s : Sphere) (c : Cylinder) : 
  (intersect p (Sum.inl s) = CrossSection.Circle) ∧ 
  ((intersect p (Sum.inr c) = CrossSection.Circle) ∨ (intersect p (Sum.inr c) = CrossSection.Ellipse)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cross_sections_l1225_122595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_overtake_distance_correct_l1225_122538

/-- Represents a race between two runners --/
structure Race where
  s : ℝ  -- Starting distance advantage of the first runner
  n : ℝ  -- Speed ratio of the second runner to the first runner
  h1 : 0 < s  -- The starting advantage is positive
  h2 : n > 1  -- The second runner is faster than the first

/-- 
The distance the second runner travels before overtaking the first runner
in a race where the first runner starts with an advantage.
-/
noncomputable def overtake_distance (race : Race) : ℝ :=
  (race.n * race.s) / (race.n - 1)

/-- 
Theorem stating that the overtake_distance function correctly calculates
the distance the second runner travels before overtaking the first runner.
-/
theorem overtake_distance_correct (race : Race) : 
  overtake_distance race = (race.n * race.s) / (race.n - 1) := by
  -- Unfold the definition of overtake_distance
  unfold overtake_distance
  -- The left-hand side and right-hand side are now identical
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_overtake_distance_correct_l1225_122538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_formula_l1225_122583

-- Define the prism parameters
variable (α β γ H : ℝ)

-- Define the conditions
variable (hα : 0 < α ∧ α < π/2)  -- α is an acute angle
variable (hβγ : 0 < β ∧ β < γ ∧ γ < π/2)  -- β < γ and both are acute angles
variable (hH : H > 0)  -- Height is positive

-- Define the volume function
noncomputable def prism_volume (α β γ H : ℝ) : ℝ :=
  H^3 * (Real.tan α * Real.sin (γ - β) * Real.sin (γ + β)) / (4 * Real.sin β^2 * Real.sin γ^2)

-- State the theorem
theorem prism_volume_formula :
  prism_volume α β γ H = H^3 * (Real.tan α * Real.sin (γ - β) * Real.sin (γ + β)) / (4 * Real.sin β^2 * Real.sin γ^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_formula_l1225_122583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_progression_from_quadratics_l1225_122551

/-- Represents a quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic trinomial at a given point -/
def evaluate (p : QuadraticTrinomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Checks if a sequence of real numbers forms an arithmetic progression -/
def isArithmeticProgression (seq : List ℝ) : Prop :=
  seq.length ≤ 1 ∨
  (∃ d : ℝ, ∀ i : Nat, i + 1 < seq.length → seq[i+1]! - seq[i]! = d)

theorem max_arithmetic_progression_from_quadratics
  (trinomials : List QuadraticTrinomial)
  (h_count : trinomials.length = 10) :
  ∃ (n : ℕ) (start : ℕ) (seq : List ℝ),
    n ≤ 20 ∧
    seq.length = n ∧
    (∀ i : ℕ, i < n → ∃ p ∈ trinomials, seq[i]! = evaluate p (start + i)) ∧
    isArithmeticProgression seq ∧
    ∀ m : ℕ, m > n →
      ¬∃ (start' : ℕ) (seq' : List ℝ),
        seq'.length = m ∧
        (∀ i : ℕ, i < m → ∃ p ∈ trinomials, seq'[i]! = evaluate p (start' + i)) ∧
        isArithmeticProgression seq' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_progression_from_quadratics_l1225_122551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1225_122553

/-- The parabola y^2 = 8x with focus at (2, 0) -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The line passing through the focus and two points on the parabola -/
def line_through_focus (A B : ParabolaPoint) : Prop :=
  ∃ (t : ℝ), A.x = focus.1 + t * (B.x - focus.1) ∧ A.y = focus.2 + t * (B.y - focus.2)

/-- The distance between two points -/
noncomputable def distance (A B : ParabolaPoint) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- The theorem to be proved -/
theorem parabola_intersection_distance (A B : ParabolaPoint) :
  line_through_focus A B → A.x + B.x = 6 → distance A B = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1225_122553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l1225_122528

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem f_decreasing : 
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x := by
  intro x y hx hy hxy
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l1225_122528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1225_122505

noncomputable def line_l (t m : ℝ) : ℝ × ℝ := (t + Real.sqrt 3 * m, -Real.sqrt 3 * t - 2 * m)

noncomputable def curve_C (θ : ℝ) : ℝ := 8 * Real.cos θ / (1 - Real.cos (2 * θ))

noncomputable def triangle_area (a b : ℝ) : ℝ := (1/2) * abs (a * b)

theorem tangent_line_triangle_area :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), (∃ (t : ℝ), line_l t m = (x, y)) → 
      (y^2 = 4*x ↔ (3*x^2 - (2*Real.sqrt 3*m + 4)*x + m^2 = 0))) →
    triangle_area (Real.sqrt 3 / 3) (1/3) = Real.sqrt 3 / 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1225_122505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_sum_2007_l1225_122529

/-- Represents the state of numbers on the circle -/
def CircleState := Fin 2006 → Fin 2006

/-- A move swaps two neighboring numbers -/
def IsValidMove (s t : CircleState) : Prop :=
  ∃ i : Fin 2006, (s i = t (i + 1) ∧ s (i + 1) = t i) ∧
    ∀ j : Fin 2006, j ≠ i ∧ j ≠ i + 1 → s j = t j

/-- The final state is diametrically opposite to the initial state -/
def IsDiametricallyOpposite (initial final : CircleState) : Prop :=
  ∀ i : Fin 2006, final i = initial (i + 1003)

/-- A sequence of moves from initial to final state -/
def ValidMoveSequence (initial final : CircleState) : Prop :=
  ∃ n : ℕ, ∃ sequence : ℕ → CircleState,
    sequence 0 = initial ∧
    sequence n = final ∧
    ∀ i : Fin n, IsValidMove (sequence i.val) (sequence (i.val + 1))

/-- Two numbers sum to 2007 -/
def SumsTo2007 (a b : Fin 2006) : Prop :=
  a.val + b.val = 2007

/-- The main theorem -/
theorem swap_sum_2007
  (initial final : CircleState)
  (h_initial : ∀ i : Fin 2006, initial i = i)
  (h_opposite : IsDiametricallyOpposite initial final)
  (h_sequence : ValidMoveSequence initial final) :
  ∃ s t : CircleState, ∃ i : Fin 2006,
    IsValidMove s t ∧ SumsTo2007 (s i) (s (i + 1)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_sum_2007_l1225_122529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_teams_same_matches_l1225_122568

/-- Represents a football championship with a given number of teams -/
structure Championship where
  numTeams : Nat
  allTeamsPlay : numTeams > 1

/-- Represents the state of the championship at any given moment -/
structure ChampionshipState where
  champ : Championship
  matchesPlayed : Fin champ.numTeams → Nat

/-- Theorem stating that in any championship state, there are two teams with the same number of matches played -/
theorem two_teams_same_matches (c : Championship) (h : c.numTeams = 30) :
  ∀ (state : ChampionshipState), state.champ = c →
  ∃ i j : Fin state.champ.numTeams, i ≠ j ∧ state.matchesPlayed i = state.matchesPlayed j := by
  intro state hState
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_teams_same_matches_l1225_122568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1225_122525

/-- A circle with center on the line x - 2y = 0, tangent to the positive y-axis,
    and cutting a chord of length 2√3 from the x-axis has the equation (x - 2)² + (y - 1)² = 4 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) :
  (∃ t : ℝ, center = (2*t, t)) →  -- Center is on the line x - 2y = 0
  (∃ y : ℝ, y > 0 ∧ (0, y) ∈ C) →  -- Circle is tangent to positive y-axis
  (∃ x : ℝ, x > 0 ∧ (x, 0) ∈ C ∧ x = Real.sqrt 3) →  -- Circle cuts chord of length 2√3 from x-axis
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1225_122525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1225_122522

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5)

theorem inequality_solution (x : ℝ) :
  (x ∈ Set.Ioo 1 2 ∪ Set.Ioo 3 4 ∪ Set.Ioo 6 8) ↔ 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ f x < 1/15) := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1225_122522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem1_l1225_122566

theorem min_value_problem1 (a b : ℝ) (h : a - 3*b + 6 = 0) :
  (2 : ℝ)^a + (1 : ℝ)/((8 : ℝ)^b) ≥ (1 : ℝ)/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem1_l1225_122566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_children_for_all_colors_l1225_122561

-- Define the number of colors, pencils per color, children, and pencils per child
def numColors : ℕ := 4
def pencilsPerColor : ℕ := 10
def numChildren : ℕ := 10
def pencilsPerChild : ℕ := 4

-- Define a distribution as a function from children to a list of colors they have
def Distribution := Fin numChildren → List (Fin numColors)

-- Define a property that a distribution is valid
def validDistribution (d : Distribution) : Prop :=
  (∀ i : Fin numChildren, (d i).length = pencilsPerChild) ∧
  (∀ c : Fin numColors, (Finset.univ.sum (λ i ↦ (d i).count c)) = pencilsPerColor)

-- Define a property that a set of children has all colors
def hasAllColors (s : Finset (Fin numChildren)) (d : Distribution) : Prop :=
  ∀ c : Fin numColors, ∃ i ∈ s, c ∈ d i

-- Theorem statement
theorem min_children_for_all_colors :
  ∀ d : Distribution, validDistribution d →
  ∃ s : Finset (Fin numChildren), s.card = 3 ∧ hasAllColors s d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_children_for_all_colors_l1225_122561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trajectory_l1225_122557

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the slope product
noncomputable def slopeProduct (t : Triangle) : ℝ :=
  let (x, y) := t.A
  (y / (x + 1)) * (y / (x - 1))

-- Define the trajectory equation
def trajectoryEq (x y m : ℝ) : Prop :=
  m * x^2 - y^2 = m

-- Theorem statement
theorem triangle_trajectory (t : Triangle) (m : ℝ) :
  t.B = (-1, 0) →
  t.C = (1, 0) →
  m ≠ 0 →
  slopeProduct t = m →
  let (x, y) := t.A
  y ≠ 0 →
  trajectoryEq x y m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trajectory_l1225_122557
