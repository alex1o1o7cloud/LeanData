import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relatively_prime_days_count_l1052_105252

/-- The number of days in the month -/
def days : ℕ := 30

/-- The month number -/
def month : ℕ := 30

/-- A day is relatively prime to the month if their GCD is 1 -/
def is_relatively_prime (day : ℕ) : Prop :=
  Nat.gcd day month = 1

/-- The count of relatively prime days in the month -/
noncomputable def count_relatively_prime_days : ℕ :=
  (Finset.range days).filter (λ d => (Nat.gcd (d + 1) month) = 1) |>.card

/-- Theorem stating the number of relatively prime days -/
theorem relatively_prime_days_count :
    count_relatively_prime_days = 8 ∧
    2 ∣ month ∧ 3 ∣ month ∧ 5 ∣ month := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relatively_prime_days_count_l1052_105252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1052_105240

/-- The sum of the infinite series 1/(4^1) + 2/(4^2) + 3/(4^3) + ... + k/(4^k) + ... -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

/-- Theorem: The sum of the infinite series is equal to 4/9 -/
theorem infiniteSeriesSum : infiniteSeries = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1052_105240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1052_105213

noncomputable def f (a b m n : ℕ+) (x : ℝ) : ℝ := 
  (Real.sin x) ^ (m : ℝ) / (a : ℝ) + (b : ℝ) / (Real.sin x) ^ (n : ℝ)

theorem min_value_of_f (a b m n : ℕ+) :
  ∃ (min_val : ℝ), ∀ (x : ℝ), 0 < x ∧ x < Real.pi → f a b m n x ≥ min_val ∧
  (∃ (x_min : ℝ), 0 < x_min ∧ x_min < Real.pi ∧ f a b m n x_min = min_val) ∧
  min_val = if (a : ℝ) * (b : ℝ) * (n : ℝ) ≥ (m : ℝ) 
            then 1 / (a : ℝ) + (b : ℝ)
            else ((m : ℝ) + (n : ℝ)) * ((1 / ((n : ℝ) * (a : ℝ))) ^ (n : ℝ) * ((b : ℝ) / (m : ℝ)) ^ (m : ℝ)) ^ (1 / ((m : ℝ) + (n : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1052_105213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_max_area_quadrilateral_l1052_105207

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2 - 3

-- Define points A, B, C, D, and P
variable (A B C D P : ℝ × ℝ)

-- Assume A, B, C, D are on the ellipse Γ
axiom A_on_Γ : Γ A.1 A.2
axiom B_on_Γ : Γ B.1 B.2
axiom C_on_Γ : Γ C.1 C.2
axiom D_on_Γ : Γ D.1 D.2

-- Assume P is on the parabola
axiom P_on_parabola : parabola P.1 P.2

-- Assume DC = 2AB
axiom DC_eq_2AB : ‖C - D‖ = 2 * ‖B - A‖

-- Assume P is the intersection of AD and BC
axiom P_is_intersection : ∃ (t₁ t₂ : ℝ), 
  P = A + t₁ • (D - A) ∧ P = B + t₂ • (C - B)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (p q r s : ℝ × ℝ) : ℝ :=
  (1/2) * |((r.1 - p.1) * (s.2 - p.2) - (s.1 - p.1) * (r.2 - p.2))|

-- Theorem 1: Maximum value of |AB| + |CD|
theorem max_sum_distances : 
  ∃ (max : ℝ), ∀ (A B C D : ℝ × ℝ), 
    Γ A.1 A.2 → Γ B.1 B.2 → Γ C.1 C.2 → Γ D.1 D.2 →
    ‖C - D‖ = 2 * ‖B - A‖ →
    distance A B + distance C D ≤ max ∧
    max = 6 := by sorry

-- Theorem 2: Maximum area of quadrilateral ABCD
theorem max_area_quadrilateral :
  ∃ (max : ℝ), ∀ (A B C D P : ℝ × ℝ),
    Γ A.1 A.2 → Γ B.1 B.2 → Γ C.1 C.2 → Γ D.1 D.2 →
    parabola P.1 P.2 →
    ‖C - D‖ = 2 * ‖B - A‖ →
    (∃ (t₁ t₂ : ℝ), P = A + t₁ • (D - A) ∧ P = B + t₂ • (C - B)) →
    area_quadrilateral A B C D ≤ max ∧
    max = ((3 * Real.sqrt 13 - 6) * Real.sqrt (2 + 2 * Real.sqrt 13)) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_max_area_quadrilateral_l1052_105207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_parameters_decreasing_function_k_range_l1052_105245

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem odd_function_parameters (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) → a = 1 ∧ b = 1 := by
  sorry

theorem decreasing_function :
  ∀ x y : ℝ, x < y → f 1 1 x > f 1 1 y := by
  sorry

theorem k_range (k : ℝ) :
  (∀ t : ℝ, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - k) < 0) → k < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_parameters_decreasing_function_k_range_l1052_105245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l1052_105264

noncomputable def f (x : ℝ) : ℝ := 6 / x

theorem inverse_proportion_ordering (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < 0) 
  (h4 : 0 < x₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l1052_105264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l1052_105282

-- Define the fixed point A
def A : ℝ × ℝ := (0, 3)

-- Define the fixed line L: y = -1
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -1}

-- Define the locus of points C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | dist p A = dist p (p.1, -1)}

-- Define what it means for a set to be a parabola
def IsParabola (S : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating that the locus C is a parabola
theorem locus_is_parabola : IsParabola C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l1052_105282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_ratio_proof_l1052_105295

/-- Given a substance with two elements, prove that the mass ratio is 1:8 -/
theorem mass_ratio_proof (total_mass : ℝ) (element1_mass : ℝ) 
  (h1 : total_mass = 171)
  (h2 : element1_mass = 19) :
  element1_mass / (total_mass - element1_mass) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_ratio_proof_l1052_105295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1052_105237

/-- Represents the time (in hours) to fill a cistern -/
noncomputable def fill_time (fill_rate : ℝ) (empty_rate : ℝ) : ℝ :=
  1 / (fill_rate - empty_rate)

theorem cistern_fill_time :
  let fill_rate : ℝ := 1 / 12  -- Rate at which pipe A fills the cistern
  let empty_rate : ℝ := 1 / 18  -- Rate at which pipe B empties the cistern
  fill_time fill_rate empty_rate = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1052_105237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_consecutive_intersection_l1052_105259

/-- Given a natural number n, this theorem states that the maximum number of subsets
    of the set {1, 2, ..., 2n+1} such that the intersection of any two subsets
    consists of one or several consecutive integers is ⌊((2n+1)²)/4⌋. -/
theorem max_subsets_with_consecutive_intersection (n : ℕ) :
  let S := Finset.range (2 * n + 1 + 1)
  ∃ (F : Finset (Finset ℕ)),
    (∀ A ∈ F, A ⊆ S) ∧
    (∀ A B, A ∈ F → B ∈ F → A ≠ B → ∃ a b : ℕ, A ∩ B = Finset.Icc a b) ∧
    F.card = ⌊((2 * n + 1)^2 : ℚ) / 4⌋ ∧
    (∀ G : Finset (Finset ℕ),
      (∀ A ∈ G, A ⊆ S) →
      (∀ A B, A ∈ G → B ∈ G → A ≠ B → ∃ a b : ℕ, A ∩ B = Finset.Icc a b) →
      G.card ≤ F.card) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_consecutive_intersection_l1052_105259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_reachable_l1052_105208

def initial_set : Set Nat := {1, 2, 4, 8, 16, 32}

def allowed_operation (s : Set Nat) : Set Nat :=
  {x | ∃ (a b : Nat), a ∈ s ∧ b ∈ s ∧ x = max a b - min a b ∧ x ≠ 0}

def reachable (target : Set Nat) : Prop :=
  ∃ (n : Nat) (f : Nat → Set Nat),
    f 0 = initial_set ∧
    (∀ i < n, f (i + 1) ⊆ allowed_operation (f i) ∪ (f i \ {a | ∃ b ∈ f i, a ∈ f i ∧ a ≠ b})) ∧
    f n = target

theorem fifteen_reachable : reachable {15} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_reachable_l1052_105208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_l1052_105291

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := x^2 = 4*y

-- Define the ellipse C₂
def C₂ (x y a b : ℝ) : Prop := y^2/a^2 + x^2/b^2 = 1

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define the common chord length
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 6

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y = k*x + F.2

-- Define the condition for vectors having the same direction
def same_direction (A B C D : ℝ × ℝ) : Prop :=
  (C.1 - A.1) / (D.1 - B.1) > 0 ∧ (C.2 - A.2) / (D.2 - B.2) > 0

theorem parabola_ellipse_intersection
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y, C₂ x y a b → C₂ x y a b)  -- C₂ is well-defined
  (h4 : ∃ x y, C₁ x y ∧ C₂ x y a b)  -- C₁ and C₂ intersect
  (h5 : ∀ x y, C₁ x y → C₂ x y a b → (x^2 + (y - F.2)^2) = common_chord_length^2 / 4)  -- Common chord length condition
  (h6 : ∀ k x y, l k x y → (C₁ x y ∨ C₂ x y a b))  -- l intersects C₁ and C₂
  (h7 : ∀ A B C D, (∃ k, l k A.1 A.2 ∧ l k B.1 B.2 ∧ l k C.1 C.2 ∧ l k D.1 D.2) →
                   C₁ A.1 A.2 → C₁ B.1 B.2 → C₂ C.1 C.2 a b → C₂ D.1 D.2 a b →
                   same_direction A B C D →
                   (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - B.1)^2 + (D.2 - B.2)^2 →
                   ∃ k, k = Real.sqrt 6 / 4 ∨ k = -Real.sqrt 6 / 4) :
  C₂ 0 3 3 (Real.sqrt 8) ∧ (∃ k, k = Real.sqrt 6 / 4 ∨ k = -Real.sqrt 6 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_l1052_105291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_160_l1052_105290

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | (n + 2) => 2 * (n + 2) / (n + 1) * sequence_a (n + 1)

theorem a_5_equals_160 : sequence_a 5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_160_l1052_105290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1052_105293

/-- The parabola equation y^2 = 12x -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The circle equation x^2 + y^2 - 4x - 6y = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between the intersection points of the parabola and circle is 3√5 -/
theorem intersection_distance : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ ∧ circle_eq x₁ y₁ ∧
  parabola x₂ y₂ ∧ circle_eq x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
  distance x₁ y₁ x₂ y₂ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1052_105293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1052_105226

theorem candy_problem (x : ℕ) : 
  (∃ (y : ℕ), x = 4 * y) →  -- x is divisible by 4
  (∃ (z : ℕ), 3 * x / 4 = 3 * z) →  -- 3/4 of x is an integer
  (∃ (w : ℕ), x / 2 = w) →  -- 1/2 of x is an integer
  (∃ (b : ℕ), 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 20 - b = 4) →  -- conditions for brother's candies and final amount
  x ∈ ({52, 56, 60} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1052_105226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l1052_105250

/-- The number of real solutions for a quadratic equation -/
noncomputable def numRealSolutions (b c : ℝ) : ℕ :=
  let Δ := b^2 - 4*c
  if Δ < 0 then 0
  else if Δ = 0 then 1
  else 2

/-- Theorem stating the relationship between the discriminant and the number of real solutions -/
theorem quadratic_solutions (b c : ℝ) :
  (numRealSolutions b c = 0 ↔ b^2 - 4*c < 0) ∧
  (numRealSolutions b c = 1 ↔ b^2 - 4*c = 0) ∧
  (numRealSolutions b c = 2 ↔ b^2 - 4*c > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l1052_105250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_G_l1052_105261

-- Define the letter G as a pair of vectors
def LetterG := Prod (Fin 2 → ℝ) (Fin 2 → ℝ)

-- Define the initial position of G
def initialG : LetterG := (![0, 1], ![1, 0])

-- Define the transformations
def rotate180 (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![-v 0, -v 1]

def reflectX (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![v 0, -v 1]

def rotate270CCW (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![v 1, -v 0]

-- Apply transformations to LetterG
def applyTransformation (g : LetterG) : LetterG :=
  let g1 := (rotate180 g.1, rotate180 g.2)
  let g2 := (reflectX g1.1, reflectX g1.2)
  (rotate270CCW g2.1, rotate270CCW g2.2)

-- Theorem statement
theorem final_position_of_G :
  applyTransformation initialG = (![1, 0], ![0, -1]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_G_l1052_105261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_distance_l1052_105288

-- Define a point in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a line in space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Function to check if a line intersects a sphere
def lineIntersectsSphere (l : Line3D) (s : Sphere) : Prop := sorry

-- Define membership for a point on a line
def pointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

-- Theorem statement
theorem sphere_intersection_distance 
  (X : Point3D) (S : Sphere) 
  (l1 l2 l3 : Line3D) 
  (p1 p2 p3 p4 p5 p6 : Point3D) :
  l1.point = X ∧ l2.point = X ∧ l3.point = X →
  lineIntersectsSphere l1 S ∧ lineIntersectsSphere l2 S ∧ lineIntersectsSphere l3 S →
  pointOnLine p1 l1 ∧ pointOnLine p2 l1 ∧ pointOnLine p3 l2 ∧ 
  pointOnLine p4 l2 ∧ pointOnLine p5 l3 ∧ pointOnLine p6 l3 →
  distance X p1 = 2 ∧ distance X p2 = 3 ∧ distance X p3 = 4 ∧ 
  distance X p4 = 5 ∧ distance X p5 = 6 →
  distance X p6 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_distance_l1052_105288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1052_105297

noncomputable section

-- Define the parabola C₁
def parabola (ρ : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2*ρ*x ∧ ρ > 0

-- Define the hyperbola C₂
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 - y^2/b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the focus of the parabola
noncomputable def focus (ρ : ℝ) : ℝ × ℝ :=
  (ρ/2, 0)

-- Define a point on the asymptote of the hyperbola
noncomputable def asymptote_point (a b ρ : ℝ) : ℝ × ℝ :=
  (ρ/2, ρ*b/(2*a))

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

-- Theorem statement
theorem hyperbola_eccentricity (ρ a b : ℝ) :
  parabola ρ (focus ρ).1 (focus ρ).2 →
  hyperbola a b (asymptote_point a b ρ).1 (asymptote_point a b ρ).2 →
  eccentricity a b = Real.sqrt 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1052_105297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1052_105215

theorem division_problem (m n : ℕ) (h1 : m % n = 12) (h2 : (m : ℚ) / n = 24.2) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1052_105215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_property_false_l1052_105229

-- Define the function f(x) = 6/x
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem stating that the given property is false
theorem inverse_proportion_property_false :
  ¬ (∀ (x₁ x₂ y₁ y₂ : ℝ), 
    (f x₁ = y₁) → (f x₂ = y₂) → (x₁ < x₂) → (y₁ > y₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_property_false_l1052_105229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_9_l1052_105299

/-- An arithmetic sequence {a_n} where a_5 is equal to the definite integral of (x+1) from -1 to 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧
  (a 5 = ∫ x in (-1)..1, (x + 1))

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sum_9 (a : ℕ → ℝ) :
  arithmetic_sequence a → arithmetic_sum a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_9_l1052_105299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l1052_105268

-- Define the curve
def f (x : ℝ) : ℝ := (2*x - 1)^3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line
def m : ℝ := 6

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := m*x - 5

-- Theorem statement
theorem tangent_line_at_point : 
  (tangent_line point.1 = point.2) ∧ 
  (m = deriv f point.1) ∧
  (∀ x, tangent_line x = m * (x - point.1) + point.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l1052_105268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_measurement_l1052_105286

/-- Represents a brick with length, width, and height -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the arrangement of three bricks as described in the problem -/
structure BrickArrangement where
  brick : Brick
  bottom_bricks : Fin 2 → Brick
  top_brick : Brick

/-- The diagonal of a brick -/
noncomputable def brick_diagonal (b : Brick) : ℝ :=
  Real.sqrt (b.length ^ 2 + b.width ^ 2 + b.height ^ 2)

/-- The distance between opposite corners of the top brick in the arrangement -/
noncomputable def top_brick_diagonal (arr : BrickArrangement) : ℝ :=
  Real.sqrt ((2 * arr.brick.length) ^ 2 + arr.brick.width ^ 2)

/-- Theorem stating that the diagonal of the top brick in the arrangement
    is equal to the diagonal of a single brick -/
theorem brick_diagonal_measurement (arr : BrickArrangement) 
    (h1 : arr.bottom_bricks 0 = arr.brick)
    (h2 : arr.bottom_bricks 1 = arr.brick)
    (h3 : arr.top_brick = arr.brick) :
  top_brick_diagonal arr = brick_diagonal arr.brick := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_measurement_l1052_105286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_one_minus_x_l1052_105239

-- Define the function f(x) = √(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x)

-- Theorem statement
theorem domain_of_sqrt_one_minus_x :
  ∀ x : ℝ, f x ∈ Set.Icc 0 1 ↔ x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_one_minus_x_l1052_105239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1052_105211

noncomputable def f (x : ℝ) := Real.sqrt (x + 1) + Real.log (x - 2)

theorem domain_of_f :
  {x : ℝ | x + 1 ≥ 0 ∧ x - 2 > 0} = {x : ℝ | x > 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1052_105211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_values_x_values_imply_perpendicular_l1052_105217

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then x else -2
def b (x : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then x - 1 else 1

-- Define dot product for our vectors
def dot (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity condition
def are_perpendicular (x : ℝ) : Prop :=
  dot (a x) (b x) = 0

-- Theorem statement
theorem perpendicular_vectors_x_values :
  ∀ x : ℝ, are_perpendicular x → x = 2 ∨ x = -1 := by
  sorry

-- Converse theorem
theorem x_values_imply_perpendicular :
  ∀ x : ℝ, (x = 2 ∨ x = -1) → are_perpendicular x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_values_x_values_imply_perpendicular_l1052_105217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1052_105269

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3) / Real.log (1/4)

-- Define the inner function g(x)
def g (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem f_monotone_increasing : 
  (∀ x ∈ Set.Icc 1 3, DifferentiableAt ℝ f x) → 
  (∀ x ∈ Set.Ioo (-1) 3, g x > 0) →
  (∀ x ∈ Set.Icc 1 3, (deriv g x) ≤ 0) →
  StrictMonoOn f (Set.Icc 1 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1052_105269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_angles_l1052_105283

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (y : ℝ) : ℝ := y^2

-- Define the intersection points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)

-- Define the angles of intersection
noncomputable def angle_at_O : ℝ := Real.pi / 2  -- 90 degrees in radians
noncomputable def angle_at_M : ℝ := Real.arctan (3 / 4)

-- Theorem statement
theorem curves_intersection_angles :
  (curve1 O.1 = O.2 ∧ O.2^2 = curve2 O.2) ∧
  (curve1 M.1 = M.2 ∧ M.2^2 = curve2 M.2) ∧
  (angle_at_O = Real.pi / 2) ∧
  (angle_at_M = Real.arctan (3 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_angles_l1052_105283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_representation_l1052_105281

theorem pythagorean_triple_representation (x y z : ℕ+) 
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x.val y.val = 1)
  (h3 : 2 ∣ y.val) :
  ∃ (a b : ℕ+), 
    x = a^2 - b^2 ∧
    y = 2 * a * b ∧
    z = a^2 + b^2 ∧
    a > b ∧
    (a.val % 2 = 1 ∧ b.val % 2 = 0 ∨ a.val % 2 = 0 ∧ b.val % 2 = 1) ∧
    Nat.gcd a.val b.val = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_representation_l1052_105281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1052_105222

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Definition of the function f for a tetrahedron ABCD -/
noncomputable def f (A B C D X : Point3D) : ℝ :=
  distance A X + distance B X + distance C X + distance D X

/-- Theorem stating the minimum value of f(X) for the given tetrahedron -/
theorem min_value_of_f (A B C D : Point3D) 
  (h1 : distance A D = 28) (h2 : distance B C = 28)
  (h3 : distance A C = 44) (h4 : distance B D = 44)
  (h5 : distance A B = 52) (h6 : distance C D = 52) :
  ∃ (m n : ℕ), m = 4 ∧ n = 678 ∧ 
  ∀ X, f A B C D X ≥ m * Real.sqrt n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1052_105222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_segments_l1052_105276

/-- Represents a triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Main theorem -/
theorem isosceles_triangle_segments (AB BC AC : ℝ) (AD DE EF DB EC FA : ℝ) :
  AB = 26 →
  AC = BC →
  AD = 7 →
  DE = 10 →
  EF = 12 →
  DB + 1 = EC →
  EC + 1 = FA →
  (Triangle.area { a := DE, b := EF, c := Real.sqrt ((DB^2) + (FA^2)) }) = 20 →
  DB = 3 ∧ EC = 4 ∧ FA = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_segments_l1052_105276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_range_l1052_105202

-- Define the variables and their relationships
noncomputable def x (t : ℝ) : ℝ := Real.sqrt t
noncomputable def y (t : ℝ) : ℝ := 2 * Real.sqrt (1 - t)

-- Define the expression
noncomputable def expr (t : ℝ) : ℝ := (y t + 2) / (x t + 2)

-- State the theorem
theorem expr_range :
  ∃ a b : ℝ, a = 2/3 ∧ b = 2 ∧
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → a ≤ expr t ∧ expr t ≤ b) ∧
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ expr t₁ = a ∧ expr t₂ = b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_range_l1052_105202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_applause_meter_rating_l1052_105218

theorem applause_meter_rating (arrow_position : ℝ) : 
  9.6 < arrow_position ∧ arrow_position < 9.8 → 
  ∀ rating ∈ ({9.3, 9.4, 9.5, 9.7, 9.9} : Set ℝ), 
  |arrow_position - 9.7| ≤ |arrow_position - rating| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_applause_meter_rating_l1052_105218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1052_105294

theorem hyperbola_standard_equation :
  ∀ (H : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ H ↔ ∃ (k : ℝ), x^2 - y^2 = k) →
  (0, 0) ∈ H →
  (∀ (x : ℝ), (x, x) ∉ H ∧ (x, -x) ∉ H) →
  (2, 1) ∈ H →
  ∀ (x y : ℝ), (x, y) ∈ H ↔ x^2 - y^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1052_105294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_time_proof_l1052_105248

/-- The time taken to cover a certain distance at the original speed -/
noncomputable def original_time : ℝ := 2/3

/-- The time taken to cover the same distance at the reduced speed -/
noncomputable def reduced_time (t : ℝ) : ℝ := t + 1/6

/-- The speed reduction factor -/
def speed_reduction : ℝ := 0.8

theorem original_time_proof (t : ℝ) :
  t * 1 = reduced_time t * speed_reduction →
  t = original_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_time_proof_l1052_105248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_weighted_average_l1052_105220

/-- Calculates the weighted average score given exam scores and weights -/
noncomputable def weighted_average (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) scores weights)) / (List.sum weights)

/-- Dacid's exam scores -/
def david_scores : List ℝ := [70, 63, 80, 63, 65]

/-- Weights for each exam -/
def exam_weights : List ℝ := [0.20, 0.30, 0.25, 0.15, 0.10]

theorem david_weighted_average :
  weighted_average david_scores exam_weights = 68.85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_weighted_average_l1052_105220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l1052_105292

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := 
  (cos (π/2 + α) * sin (3*π/2 - α)) / (cos (-π - α) * tan (π - α))

-- State the theorem
theorem f_value_at_negative_25pi_over_3 : f (-25*π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l1052_105292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_special_division_property_l1052_105228

theorem least_number_with_special_division_property : ∃ x : ℕ, 
  (x = 75) ∧ 
  (∀ y : ℕ, y < x → ¬(y / 5 = y % 34 + 8)) ∧ 
  (x / 5 = x % 34 + 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_special_division_property_l1052_105228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_sum_multiple_of_ten_l1052_105234

def number_set : Set ℕ := {n | 11 ≤ n ∧ n ≤ 30}

def sum_multiple_of_ten (a b : ℕ) : Prop := ∃ k : ℕ, a + b = 10 * k

theorem minimum_selection_for_sum_multiple_of_ten :
  (∀ s : Finset ℕ, (∀ x ∈ s, x ∈ number_set) → s.card = 11 →
    ∃ a b : ℕ, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ sum_multiple_of_ten a b) ∧
  (∃ s : Finset ℕ, (∀ x ∈ s, x ∈ number_set) ∧ s.card = 10 ∧
    ∀ a b : ℕ, a ∈ s → b ∈ s → a ≠ b → ¬sum_multiple_of_ten a b) :=
by sorry

#check minimum_selection_for_sum_multiple_of_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_sum_multiple_of_ten_l1052_105234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l1052_105233

def median (l : List ℕ) : ℕ := sorry

theorem max_element_of_list (l : List ℕ) : 
  l.length = 7 ∧ 
  median l = 5 ∧ 
  l.sum / l.length = 15 →
  l.maximum ≤ 87 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l1052_105233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_visit_planning_l1052_105216

theorem museum_visit_planning : 
  (Nat.choose 6 3) * (Nat.factorial 3) * (Nat.factorial 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_visit_planning_l1052_105216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_angles_theorem_l1052_105227

/-- A polygon in the plane -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  non_self_intersecting : Prop -- Placeholder for the non-self-intersecting property
  equal_side_lengths : Prop -- Placeholder for the equal side lengths property

/-- The number of interior angles greater than 180° in a polygon -/
def num_large_angles (n : ℕ) (p : Polygon n) : ℕ := sorry

/-- The maximum number of interior angles greater than 180° in any n-gon -/
def max_large_angles (n : ℕ) : ℕ := sorry

theorem max_large_angles_theorem (n : ℕ) (h : n ≥ 3) :
  max_large_angles n = if n ≥ 5 then n - 3 else 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_angles_theorem_l1052_105227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1052_105232

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 4)

theorem vector_problem :
  let angle := Real.arccos (((a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)) /
    (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) * Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)))
  let lambda := -(a.1^2 + a.2^2) / (a.1 * b.1 + a.2 * b.2)
  angle = 3 * Real.pi / 4 ∧ lambda = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1052_105232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_l1052_105289

/-- The volume of an inscribed cone in a larger cone -/
theorem inscribed_cone_volume (H α : ℝ) (H_pos : H > 0) (α_pos : α > 0) (α_lt_pi_div_2 : α < π / 2) :
  ∃ (V : ℝ), V = (1 / 3) * π * H^3 * (Real.sin α)^4 * (Real.cos α)^2 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_l1052_105289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l1052_105263

/-- A predicate that determines if an equation represents a circle -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ x y, f x y ↔ ((x - a)^2 + (y - b)^2 = r^2)

/-- The equation x^2 + y^2 - 2x - 4y + m = 0 -/
def equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

theorem circle_condition (m : ℝ) :
  is_circle (equation m) → m < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l1052_105263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1052_105287

noncomputable def point := ℝ × ℝ

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_points (A B : point) 
  (h1 : B.1 - A.1 = 50) 
  (h2 : A.2 - B.2 = 40) : 
  distance A B = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1052_105287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1052_105280

/-- Represents a rectangle ABCD with given dimensions -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- Calculates the volume of the solid generated by rotating a triangle around its base in a rectangle -/
noncomputable def rotationVolume (rect : Rectangle) : ℝ :=
  (2/3) * Real.pi * rect.BC^2 * rect.AB

/-- Theorem stating the volume of the solid generated by rotating the triangle BCO around CD in the given rectangle -/
theorem volume_of_rotation (rect : Rectangle) 
  (h1 : rect.AB = 10) 
  (h2 : rect.BC = 6) : 
  rotationVolume rect = 60 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1052_105280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_promotion_price_l1052_105205

/-- The effective price per pair of socks during a "buy five, get one free" promotion -/
noncomputable def effective_price (original_price : ℚ) : ℚ :=
  (5 * original_price) / 6

/-- Theorem: The effective price of socks during the promotion is 4.05 yuan -/
theorem socks_promotion_price :
  let original_price : ℚ := 486/100
  effective_price original_price = 405/100 := by
  -- Unfold the definition of effective_price
  unfold effective_price
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that the equality holds
  norm_num

-- This will not evaluate due to noncomputable definition
-- #eval effective_price (486/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_promotion_price_l1052_105205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_ratio_l1052_105273

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  baseEdgeLength : ℝ
  height : ℝ

/-- Represents a section of the pyramid parallel to its base -/
structure PyramidSection where
  distanceFromApex : ℝ
  area : ℝ

/-- Calculates the areas of sections in a regular hexagonal pyramid -/
noncomputable def calculateSectionAreas (pyramid : RegularHexagonalPyramid) : 
  Fin 4 → PyramidSection
| ⟨0, _⟩ => { distanceFromApex := pyramid.height / 4, area := 25 * pyramid.baseEdgeLength * pyramid.height / 4 }
| ⟨1, _⟩ => { distanceFromApex := pyramid.height / 2, area := 20 * pyramid.baseEdgeLength * pyramid.height / 4 }
| ⟨2, _⟩ => { distanceFromApex := 3 * pyramid.height / 4, area := 25 * pyramid.baseEdgeLength * pyramid.height / 4 }
| ⟨3, _⟩ => { distanceFromApex := pyramid.height, area := 9 * pyramid.baseEdgeLength * pyramid.height / 4 }

/-- The theorem stating the ratio of section areas in a regular hexagonal pyramid -/
theorem section_area_ratio (pyramid : RegularHexagonalPyramid) :
  let sections := calculateSectionAreas pyramid
  (sections 0).area / (sections 1).area = 25 / 20 ∧
  (sections 1).area / (sections 2).area = 20 / 25 ∧
  (sections 2).area / (sections 3).area = 25 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_ratio_l1052_105273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l1052_105203

theorem no_valid_n : ¬∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l1052_105203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_of_sampled_bags_l1052_105247

/-- Represents the difference from standard weight and the number of bags for each difference --/
structure WeightDifference :=
  (diff : Int)
  (count : Nat)

/-- The problem setup --/
def qualityTestSetup : List WeightDifference := [
  ⟨-2, 2⟩, ⟨-1, 3⟩, ⟨0, 2⟩, ⟨1, 2⟩, ⟨2, 1⟩
]

/-- The standard weight of each bag in grams --/
def standardWeight : Nat := 150

/-- The number of bags sampled --/
def numBags : Nat := 10

/-- Theorem stating the total weight of sampled bags --/
theorem total_weight_of_sampled_bags :
  (standardWeight * numBags : Int) + 
  (qualityTestSetup.map (λ wd => wd.diff * (wd.count : Int))).sum = 1497 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_of_sampled_bags_l1052_105247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_divisible_by_9_l1052_105209

def numbers : List Nat := [4272, 4281, 4290, 4311, 4320]

def is_divisible_by_9 (n : Nat) : Prop := n % 9 = 0

def units_digit (n : Nat) : Nat := n % 10

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem unique_non_divisible_by_9 :
  ∃! n, n ∈ numbers ∧ ¬is_divisible_by_9 n ∧ 
  units_digit n * tens_digit n = 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_divisible_by_9_l1052_105209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_evaluation_l1052_105206

theorem floor_evaluation : ⌊(4 : ℝ) * (7 - 1/3)⌋ = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_evaluation_l1052_105206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_points_form_arc_l1052_105200

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle --/
structure Chord where
  circle : Circle
  endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- A segment of a circle divided by a chord --/
structure CircleSegment where
  circle : Circle
  chord : Chord

/-- A pair of touching circles inscribed in a circle segment --/
structure InscribedCirclePair where
  segment : CircleSegment
  circle1 : Circle
  circle2 : Circle

/-- The set of points of contact between pairs of touching circles --/
def ContactPoints (segment : CircleSegment) : Set (ℝ × ℝ) :=
  {p | ∃ pair : InscribedCirclePair, pair.segment = segment ∧ 
       (p = pair.circle1.center ∨ p = pair.circle2.center)}

/-- An arc of a circle --/
structure CircleArc where
  circle : Circle
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The set of points on a circle arc --/
def PointsOnArc (arc : CircleArc) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
       p = (1 - t) • arc.startPoint + t • arc.endPoint ∧
       ‖p - arc.circle.center‖ = arc.circle.radius}

theorem contact_points_form_arc (S : Circle) (AB : Chord) 
  (h : AB.circle = S) : 
  ∃ arc : CircleArc, ContactPoints ⟨S, AB⟩ = PointsOnArc arc :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_points_form_arc_l1052_105200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1052_105204

/-- The hyperbola equation: x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The perpendicular line from the right focus to the x-axis -/
def perpendicular_line (x : ℝ) : Prop := x = 2

/-- The asymptotes of the hyperbola -/
noncomputable def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The area of the triangle formed by the perpendicular line and the asymptotes -/
noncomputable def triangle_area : ℝ := 4 * Real.sqrt 3

/-- Theorem: The area of the triangle formed by the perpendicular line from the right focus
    to the x-axis and the two asymptotes of the hyperbola x^2 - y^2/3 = 1 is 4√3 -/
theorem hyperbola_triangle_area : 
  ∀ (x y : ℝ), hyperbola x y → 
  ∃ (p q : ℝ × ℝ), 
    perpendicular_line (p.1) ∧ 
    perpendicular_line (q.1) ∧ 
    asymptote (p.1) (p.2) ∧ 
    asymptote (q.1) (q.2) ∧ 
    triangle_area = 4 * Real.sqrt 3 := by
  sorry

#check hyperbola_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1052_105204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_consecutive_integers_sum_55_l1052_105271

theorem greatest_consecutive_integers_sum_55 : 
  (∃ (a : ℤ), (Finset.sum (Finset.range 110) (λ i => a + i) = 55)) ∧ 
  (∀ n : ℕ, n > 110 → ¬∃ (a : ℤ), (Finset.sum (Finset.range n) (λ i => a + i) = 55)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_consecutive_integers_sum_55_l1052_105271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_l1052_105225

theorem sine_equality (α : ℝ) :
  Real.sin (2 * π / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5 →
  Real.sin (α + 7 * π / 6) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_l1052_105225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_k_value_l1052_105236

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then k = 4 -/
theorem parallel_planes_k_value (k : ℝ) : 
  let n1 : ℝ × ℝ × ℝ := (1, 2, -2)  -- normal vector of plane α
  let n2 : ℝ × ℝ × ℝ := (-2, -4, k) -- normal vector of plane β
  (∃ (t : ℝ), n1 = t • n2) →       -- planes are parallel if their normal vectors are scalar multiples
  k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_k_value_l1052_105236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_linearly_independent_l1052_105257

def a₁ : ℝ × ℝ × ℝ := (2, 1, 3)
def a₂ : ℝ × ℝ × ℝ := (1, 0, -1)
def a₃ : ℝ × ℝ × ℝ := (0, 0, 1)

theorem vectors_linearly_independent :
  ∀ (l1 l2 l3 : ℝ), l1 • a₁ + l2 • a₂ + l3 • a₃ = (0, 0, 0) → l1 = 0 ∧ l2 = 0 ∧ l3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_linearly_independent_l1052_105257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bag_problem_l1052_105254

def candy_sequence (n : ℕ) : ℕ := 2^n

def person_A_candies (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n+1)) (λ i => candy_sequence (2*i))

def person_B_candies (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => candy_sequence (2*i + 1))

theorem candy_bag_problem (n : ℕ) (h : person_A_candies n = 90) :
  person_A_candies n + person_B_candies n = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bag_problem_l1052_105254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1052_105224

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def ring_area (outer_r inner_r : ℝ) : ℝ := circle_area outer_r - circle_area inner_r

theorem black_to_white_area_ratio :
  let r1 : ℝ := 1
  let r2 : ℝ := 2
  let r3 : ℝ := 4
  let r4 : ℝ := 6
  let r5 : ℝ := 8
  let black_area : ℝ := circle_area r1 + ring_area r3 r2 + ring_area r5 r4
  let white_area : ℝ := ring_area r2 r1 + ring_area r4 r3
  black_area / white_area = 41 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1052_105224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l1052_105223

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The ellipse with equation x²/2 + y² = 1 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2

theorem ellipse_intersection_length (l : Line) (A B : Point) :
  isOnLine (Point.mk 0 2) l →
  isOnEllipse A ∧ isOnEllipse B →
  isOnLine A l ∧ isOnLine B l →
  A ≠ B →
  triangleArea (Point.mk 0 0) A B = Real.sqrt 2 / 2 →
  distance A B = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l1052_105223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l1052_105266

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  PS : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the area of triangle PQM given the tetrahedron -/
noncomputable def areaOfTrianglePQM (t : Tetrahedron) : ℝ := sorry

/-- Calculates the volume of the tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS is 7A/12, where A is the area of triangle PQM -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.PQ = 6)
  (h2 : t.PR = 4)
  (h3 : t.QR = 5)
  (h4 : t.PS = 5)
  (h5 : t.QS = 4)
  (h6 : t.RS = 7 * Real.sqrt 2 / 2) :
  volume t = 7 * areaOfTrianglePQM t / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l1052_105266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1052_105231

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  a + 2 * Real.sqrt (a * b) + (a * b * c) ^ (1/3 : ℝ) ≤ 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1052_105231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1052_105210

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of participants excluding the 12 lowest-scoring players
  total_players : ℕ := n + 12

  -- Each player plays exactly once against every other player
  games_played : total_players * (total_players - 1) / 2 = Nat.choose total_players 2

  -- Half of each player's points come from games against the 12 lowest-scoring players
  points_from_lowest : ℕ
  half_points_condition : 2 * points_from_lowest = n * 12

  -- The 12 lowest-scoring players get half their points from games among themselves
  lowest_internal_points : ℕ
  lowest_external_points : ℕ
  lowest_points_condition : 2 * lowest_internal_points = lowest_external_points
  lowest_total_points : lowest_internal_points + lowest_external_points = Nat.choose 12 2 + points_from_lowest

/-- The theorem stating that the total number of participants is 24 -/
theorem chess_tournament_participants : ∀ (ct : ChessTournament), ct.total_players = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1052_105210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l1052_105296

/-- The line l: 3x - √3y - 6 = 0 -/
def line_l (x y : ℝ) : Prop := 3 * x - Real.sqrt 3 * y - 6 = 0

/-- The curve C: x² + y² - 4y = 0 -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- A point P on the curve C -/
def point_on_C (P : ℝ × ℝ) : Prop := curve_C P.1 P.2

/-- A line with slope 30° passing through a point -/
def line_30_deg (P A : ℝ × ℝ) : Prop :=
  (A.2 - P.2) = (A.1 - P.1) * Real.tan (30 * Real.pi / 180)

/-- The intersection point A of line l and the 30° line through P -/
def intersection_point (P A : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ line_30_deg P A

/-- The distance between two points -/
noncomputable def distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2)

/-- The main theorem -/
theorem max_distance_theorem :
  ∀ P : ℝ × ℝ, point_on_C P →
  ∃ A : ℝ × ℝ, intersection_point P A ∧
  ∀ A' : ℝ × ℝ, intersection_point P A' →
  distance P A ≤ 6 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l1052_105296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_eq_l1052_105242

/-- The repeating decimal 0.7888... expressed as a rational number -/
def repeating_decimal : ℚ := 71 / 90

/-- Theorem stating that the repeating decimal 0.7888... is equal to 71/90 -/
theorem repeating_decimal_eq : repeating_decimal = 0.7 + 8 / 90 + 8 / 900 + 8 / 9000 + 8 / 90000 := by
  sorry

#eval repeating_decimal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_eq_l1052_105242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_relation_l1052_105278

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of f(x) > 0
def solution_set (a b c : ℝ) : Set ℝ := {x | f a b c x > 0}

-- Define the quadratic function g(x) = cx^2 - bx + a
def g (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 - b * x + a

-- Define the solution set of g(x) < 0
def inverse_solution_set (a b c : ℝ) : Set ℝ := {x | g a b c x < 0}

-- Theorem statement
theorem quadratic_inequality_relation 
  (a b c : ℝ) (h : solution_set a b c = Set.Ioo (-2) 1) :
  inverse_solution_set a b c = Set.Iic (-1) ∪ Set.Ioi (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_relation_l1052_105278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_doll_height_l1052_105270

/-- Represents the height of a nesting doll in a sequence -/
noncomputable def nestingDollHeight (n : ℕ) (first_height : ℝ) : ℝ :=
  first_height * (2/3) ^ (n - 1)

/-- Theorem stating the height of the biggest nesting doll -/
theorem biggest_doll_height :
  ∃ (first_height : ℝ),
    (nestingDollHeight 6 first_height = 32) ∧
    (first_height = 243) := by
  -- We'll use 243 as the first_height
  use 243
  constructor
  · -- Prove that nestingDollHeight 6 243 = 32
    sorry
  · -- Prove that 243 = 243
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_doll_height_l1052_105270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_less_than_one_l1052_105230

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

noncomputable def g (x : ℝ) : ℝ := Real.exp (x * Real.log 3) - 2

-- Define the sets M and N
def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

-- State the theorem
theorem M_intersect_N_eq_less_than_one : M ∩ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_less_than_one_l1052_105230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_15_prime_factors_l1052_105255

/-- The list of all prime numbers less than 50 -/
def primes_under_50 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

/-- The product of all prime numbers less than 50 -/
def product_of_primes : Nat := primes_under_50.prod

theorem smallest_number_with_15_prime_factors :
  (∀ p ∈ primes_under_50, product_of_primes % p = 0) ∧
  (List.length (Nat.factors product_of_primes) = 15) ∧
  (product_of_primes = 614889782588491410) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_15_prime_factors_l1052_105255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1052_105244

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x / Real.log x

-- State the theorem
theorem function_properties :
  ∃ (m : ℝ),
    -- Condition: Tangent line at (e², f(e²)) is perpendicular to 2x + y = 0
    (((deriv (f m)) (Real.exp 2)) * (-2) = 1) ∧
    -- 1. m = 2
    (m = 2) ∧
    -- 2. Monotonically decreasing intervals
    (∀ x, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < Real.exp 1) → deriv (f m) x < 0) ∧
    -- 3. Inequality holds for k = 2
    (∀ x, x > 0 → f m x > 2 / Real.log x + 2 * Real.sqrt x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1052_105244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_sequence_bound_l1052_105238

theorem unique_sum_sequence_bound (n k : ℕ) (a : Fin k → ℕ) : 
  (∀ i : Fin k, 1 ≤ a i) →
  (∀ i j : Fin k, i < j → a i < a j) →
  (∀ i : Fin k, a i ≤ n) →
  (∀ i j l m : Fin k, i ≤ j → l ≤ m → (i, j) ≠ (l, m) → a i + a j ≠ a l + a m) →
  (k : ℝ) ≤ Real.sqrt (2 * n) + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_sequence_bound_l1052_105238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_60_l1052_105274

/-- Represents the capacity of a water tank in gallons. -/
def tank_capacity (x : ℝ) : Prop :=
  x > 0 ∧ 0.9 * x - 0.4 * x = 30

/-- Theorem stating that the tank capacity is 60 gallons. -/
theorem tank_capacity_is_60 : ∃ x, tank_capacity x ∧ x = 60 := by
  use 60
  constructor
  · constructor
    · exact (show 60 > 0 from by norm_num)
    · norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_60_l1052_105274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l1052_105279

/-- The function f(x) = x³ + ax² - 9x - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

/-- The minimum value of f'(x) -/
noncomputable def min_slope (a : ℝ) : ℝ := -9 - a^2/3

theorem tangent_line_parallel_implies_a_value :
  ∀ a : ℝ, (∃ x : ℝ, f_derivative a x = min_slope a) ∧ 
           min_slope a = -12 →
  a = 3 ∨ a = -3 := by
  sorry

#check tangent_line_parallel_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l1052_105279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_integers_l1052_105235

theorem max_sum_of_integers (b : ℕ) (c : ℕ) : 
  (∃ (n : ℕ), b^2 + 60*b = n^2) →
  (∃ (k : ℕ), (Real.sqrt (b : ℝ) + Real.sqrt ((b + 60) : ℝ))^2 = k) →
  ¬(∃ (m : ℕ), (Real.sqrt (b : ℝ) + Real.sqrt ((b + 60) : ℝ)) = Real.sqrt (m : ℝ)) →
  2*b + 60 ≤ 156 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_integers_l1052_105235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_problem_l1052_105251

-- Define the parabola E
noncomputable def E (x y : ℝ) : Prop := y^2 = 8*x

-- Define the circle M
noncomputable def M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point Q on curve C
noncomputable def Q (x₀ y₀ : ℝ) : Prop := C x₀ y₀ ∧ x₀ ≥ 5

-- Define the area of triangle QAB
noncomputable def area_QAB (x₀ y₀ : ℝ) : ℝ := 2 * ((x₀ - 1) + 1 / (x₀ - 1) + 2)

theorem parabola_circle_tangent_problem :
  (∀ x y, E x y → C (x/2) (y/2)) ∧
  (∀ x₀ y₀, Q x₀ y₀ → area_QAB x₀ y₀ ≥ 25/2) ∧
  (∃ x₀ y₀, Q x₀ y₀ ∧ area_QAB x₀ y₀ = 25/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_problem_l1052_105251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1052_105241

-- Define the sequence a_n
noncomputable def a (n : ℕ+) : ℝ := 2 + 4 / (3^(n : ℝ) - 1)

-- Define the sequence b_n
noncomputable def b (n : ℕ+) (p : ℝ) : ℝ := (a n + p) / (a n - 2)

theorem sequence_properties :
  -- (1) Maximum term of the sequence a_n
  (∀ n : ℕ+, a n ≤ 4) ∧
  (∃ n : ℕ+, a n = 4) ∧
  
  -- (2) Condition for b_n to be a geometric sequence
  (∀ p : ℝ, (∀ n : ℕ+, ∃ r : ℝ, b (n + 1) p = r * b n p) ↔ (p = 2 ∨ p = -2)) ∧
  
  -- (3) Non-existence of arithmetic progression in a_n
  ∀ m n p : ℕ+, m < n → n < p → ¬(2 * a n = a m + a p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1052_105241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l1052_105214

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi/4) + Real.sin (x + Real.pi/3) * Real.cos (x + Real.pi/6)

theorem max_value_of_y :
  ∃ (max : ℝ), ∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/12 → y x ≤ max ∧
  max = 1/2 + (Real.sqrt 6 + Real.sqrt 2)/4 - Real.sqrt 3/4 :=
by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l1052_105214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1052_105258

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem f_properties :
  let a : ℝ := -Real.pi / 6
  let b : ℝ := Real.pi / 4
  (f (Real.pi / 6) = 2) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ 2) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ -1) ∧
  (∃ x, a ≤ x ∧ x ≤ b ∧ f x = 2) ∧
  (∃ x, a ≤ x ∧ x ≤ b ∧ f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1052_105258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_angle_sine_inequality_l1052_105212

theorem negation_of_angle_sine_inequality :
  (∀ (A B C : ℝ), A + B + C = π → (A > B → Real.sin A > Real.sin B)) ↔
  (∃ (A B C : ℝ), A + B + C = π ∧ A > B ∧ Real.sin A ≤ Real.sin B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_angle_sine_inequality_l1052_105212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cars_count_l1052_105272

theorem train_cars_count : ℕ :=
  let train_car_capacity : ℕ := 60
  let airplane_capacity : ℕ := 366
  let extra_passengers : ℕ := 228
  let airplane_count : ℕ := 2

  let train_capacity : ℕ → ℕ := λ x => x * train_car_capacity
  let airplanes_capacity : ℕ := airplane_count * airplane_capacity

  have h : ∃ x : ℕ, train_capacity x = airplanes_capacity + extra_passengers := by
    use 16
    simp [train_capacity, airplanes_capacity]
    norm_num

  Nat.find h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cars_count_l1052_105272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_index_theorem_correlation_index_approx_l1052_105267

/-- The correlation index between height and weight -/
def correlation_index : ℝ := 0.64

/-- The proportion of weight variation explained by height -/
def height_explanation : ℝ := 0.64

/-- The proportion of weight variation contributed by random errors -/
def random_error_contribution : ℝ := 0.36

/-- Theorem stating the relationship between height explanation, 
    random error contribution, and the correlation index -/
theorem correlation_index_theorem :
  height_explanation + random_error_contribution = 1 ∧
  correlation_index = height_explanation := by
  sorry

/-- Theorem stating that the correlation index is approximately equal to the height explanation -/
theorem correlation_index_approx :
  ∃ ε > 0, |correlation_index - height_explanation| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_index_theorem_correlation_index_approx_l1052_105267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_area_percentage_l1052_105260

noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

def circle_radii : List ℝ := [3, 6, 9, 12]

def black_rings : List (ℝ × ℝ) := [(6, 3), (9, 6)]

theorem black_area_percentage :
  let total_area := circle_area (circle_radii.getLast!)
  let black_area := (black_rings.map (fun (outer, inner) => 
    circle_area outer - circle_area inner)).sum
  black_area / total_area = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_area_percentage_l1052_105260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_35_donation_amount_l1052_105277

-- Define the basic parameters
def total_boxes : ℕ := 50
def cost_per_box : ℕ := 15
def type_b_price : ℕ := 30

-- Define the profit functions
def profit_a (x : ℕ) : ℤ := -2 * x^2 + 110 * x - 1200
def profit_b (x : ℕ) : ℤ := 30 * x - 450

-- Define the total profit function
def total_profit (x : ℕ) : ℤ := profit_a x + profit_b x

-- State the theorem
theorem max_profit_at_35 :
  ∀ x : ℕ, x ≥ 20 → total_profit x ≤ total_profit 35 ∧ total_profit 35 = 800 := by
  sorry

-- Define the sales volume functions
def sales_a (x : ℕ) : ℤ := 80 - 2 * x
def sales_b (x : ℕ) : ℤ := 2 * x - 30

-- State the conditions
axiom sales_volume_positive :
  ∀ x : ℕ, x ≥ 20 → sales_a x > 0 ∧ sales_b x > 0

axiom total_sales_equals_total_boxes :
  ∀ x : ℕ, x ≥ 20 → sales_a x + sales_b x = total_boxes

-- Remove the problematic axiom
-- axiom x_is_integer : ∀ x : ℕ, x ≥ 20 → x ∈ ℕ

-- Add a theorem for the donation amount
theorem donation_amount :
  ∃ a : ℚ, 0 < a ∧ a < 10 ∧ total_profit 35 - (2 * 35 - 30) * a = 722 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_35_donation_amount_l1052_105277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_has_winning_strategy_l1052_105219

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a piece on the board -/
inductive Piece where
  | White
  | Black

/-- Represents the game board -/
def Board := Cell → Option Piece

/-- Represents a player -/
inductive Player where
  | White
  | Black

/-- Initializes the board with pieces in the correct starting positions -/
def initBoard : Board :=
  fun c => 
    if c.row = 1 then some Piece.White
    else if c.row = 15 then some Piece.Black
    else none

/-- Checks if a move is valid according to the game rules -/
def isValidMove (board : Board) (fromCell toCell : Cell) (player : Player) : Prop :=
  match player with
  | Player.White => 
      fromCell.row < toCell.row ∧ 
      toCell.row ≤ 15 ∧
      fromCell.col = toCell.col ∧
      (∀ r, fromCell.row < r ∧ r < toCell.row → board ⟨r, fromCell.col⟩ ≠ some Piece.Black)
  | Player.Black => 
      fromCell.row > toCell.row ∧ 
      toCell.row ≥ 1 ∧
      fromCell.col = toCell.col ∧
      (∀ r, toCell.row < r ∧ r < fromCell.row → board ⟨r, fromCell.col⟩ ≠ some Piece.White)

/-- Represents a game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Represents a strategy for a player -/
def Strategy := GameState → Option (Cell × Cell)

/-- Predicate to check if White wins in n moves -/
def white_wins_in_n_moves (s : Strategy) (game : GameState) (n : Nat) : Prop :=
  sorry  -- The actual implementation would go here

/-- Theorem: The first player (White) has a winning strategy -/
theorem white_has_winning_strategy :
  ∃ (s : Strategy), ∀ (game : GameState), 
    game.board = initBoard → game.currentPlayer = Player.White →
    ∃ (n : Nat), white_wins_in_n_moves s game n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_has_winning_strategy_l1052_105219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_side_relationship_square_side_length_is_sqrt_2_l1052_105221

/-- The side length of a square given its diagonal length. -/
noncomputable def squareSideLength (diagonal : ℝ) : ℝ := diagonal / Real.sqrt 2

theorem square_diagonal_side_relationship (diagonal : ℝ) (side : ℝ) (h : diagonal = 2) :
  side = squareSideLength diagonal ↔ side * Real.sqrt 2 = diagonal := by sorry

theorem square_side_length_is_sqrt_2 :
  squareSideLength 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_side_relationship_square_side_length_is_sqrt_2_l1052_105221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1052_105262

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  (2 - a) + (3 - a) = 5 - 2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1052_105262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_special_triples_l1052_105246

/-- A triple of positive integers satisfying the given conditions -/
structure SpecialTriple where
  x : ℕ
  y : ℕ
  z : ℕ
  positive : x > 0 ∧ y > 0 ∧ z > 0
  distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z
  equation : x - y + z = 1
  divisibility : (x * y) % z = 0 ∧ (y * z) % x = 0 ∧ (z * x) % y = 0

/-- There exist infinitely many SpecialTriples -/
theorem infinitely_many_special_triples : ∀ n : ℕ, ∃ t : SpecialTriple, t.x > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_special_triples_l1052_105246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_equidistant_point_exists_l1052_105256

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the cyclists
structure Cyclist where
  circle : Circle
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

-- Define the system of two cyclists
structure TwoCyclistSystem where
  circle1 : Circle
  circle2 : Circle
  cyclist1 : Cyclist
  cyclist2 : Cyclist
  intersection_point : ℝ × ℝ

-- Helper function to calculate cyclist position at time t
noncomputable def cyclist_position (c : Cyclist) (t : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem fixed_equidistant_point_exists (system : TwoCyclistSystem) 
  (h1 : system.circle1.center ≠ system.circle2.center)
  (h2 : system.cyclist1.speed > 0)
  (h3 : system.cyclist2.speed > 0)
  (h4 : system.cyclist1.circle = system.circle1)
  (h5 : system.cyclist2.circle = system.circle2)
  (h6 : ∃ (t : ℝ), t > 0 ∧ cyclist_position system.cyclist1 t = cyclist_position system.cyclist2 t)
  : ∃ (p : ℝ × ℝ), ∀ (t : ℝ), 
    dist p (cyclist_position system.cyclist1 t) = dist p (cyclist_position system.cyclist2 t) := by
  sorry

-- Define the dist function (Euclidean distance)
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_equidistant_point_exists_l1052_105256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_all_shapes_l1052_105265

-- Define the shapes
class Rectangle
class Rhombus
class Square

-- Define the property of diagonals bisecting each other
def DiagonalsBisectEachOther (S : Type) : Prop :=
  ∀ s : S, ∃ (d1 d2 : S → ℝ × ℝ), d1 s = d2 s

-- State the theorem
theorem diagonals_bisect_in_all_shapes :
  DiagonalsBisectEachOther Rectangle ∧
  DiagonalsBisectEachOther Rhombus ∧
  DiagonalsBisectEachOther Square := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_all_shapes_l1052_105265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_formula_l1052_105284

/-- The volume of a cone with apex angle 2α and sum of height and slant height a -/
noncomputable def coneVolume (a : ℝ) (α : ℝ) : ℝ :=
  (a^3 * Real.pi * Real.cos α * (Real.sin (α/2))^2) / (6 * (Real.cos (α/2))^4)

/-- Theorem stating the volume of a cone with given conditions -/
theorem cone_volume_formula (a : ℝ) (α : ℝ) (h : a > 0) (h_angle : 0 < α ∧ α < Real.pi/2) :
  let height := a * Real.cos α / (1 + Real.cos α)
  let slantHeight := a / (1 + Real.cos α)
  let radius := a * Real.sin α / (1 + Real.cos α)
  (4/3) * Real.pi * radius^2 * height = coneVolume a α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_formula_l1052_105284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_selection_probability_l1052_105253

/-- Represents the selection process for a math competition -/
structure MathCompetitionSelection where
  total_students : ℕ
  eliminated : ℕ
  selected : ℕ
  remaining : ℕ

/-- The probability of a student being selected for the math competition -/
def selection_probability (mcs : MathCompetitionSelection) : ℚ :=
  mcs.selected / mcs.total_students

/-- Axiom for simple random sampling -/
axiom simple_random_sampling (k n : ℕ) : Prop

/-- Axiom for systematic sampling -/
axiom systematic_sampling (k n : ℕ) : Prop

/-- Theorem stating that the selection probability is equal for all students and equal to 50/2013 -/
theorem math_competition_selection_probability 
  (mcs : MathCompetitionSelection)
  (h1 : mcs.total_students = 2013)
  (h2 : mcs.eliminated = 13)
  (h3 : mcs.selected = 50)
  (h4 : mcs.remaining = mcs.total_students - mcs.eliminated)
  (h5 : mcs.remaining = 2000)
  (h6 : simple_random_sampling mcs.eliminated mcs.total_students)
  (h7 : systematic_sampling mcs.selected mcs.remaining) :
  selection_probability mcs = 50 / 2013 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_selection_probability_l1052_105253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_cyclist_catch_up_l1052_105243

/-- The time required for a hiker to catch up with a cyclist who stopped after passing -/
noncomputable def catch_up_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (stop_time : ℝ) : ℝ :=
  let distance := cyclist_speed * stop_time / 60
  distance / hiker_speed * 60

/-- Theorem stating that under the given conditions, the catch-up time is 20 minutes -/
theorem hiker_cyclist_catch_up :
  catch_up_time 7 28 5 = 20 := by
  -- Unfold the definition of catch_up_time
  unfold catch_up_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_cyclist_catch_up_l1052_105243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gather_cards_possible_l1052_105275

/-- Represents a configuration of cards in boxes -/
def CardConfiguration := List (List Nat)

/-- The instruction for moving a card -/
def moveCard (k : Nat) (config : CardConfiguration) : CardConfiguration :=
  sorry

/-- Checks if all cards are in one box -/
def allCardsInOneBox (config : CardConfiguration) : Prop :=
  sorry

/-- Theorem: For all positive integers n, there exists a sequence of moves
    that gathers all cards into one box -/
theorem gather_cards_possible (n : Nat) (h : n > 0) :
  ∃ (moves : List Nat), 
    let initial_config : CardConfiguration := (List.range n).map (λ x => [x])
    let final_config := moves.foldl (λ config k => moveCard k config) initial_config
    allCardsInOneBox final_config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gather_cards_possible_l1052_105275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_5_sqrt_2_l1052_105285

/-- Line l with parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (3 - (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

/-- Curve C with Cartesian equation -/
def curve_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 13

/-- Point A -/
def point_A : ℝ × ℝ := (3, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Sum of distances from A to intersection points is 5√2 -/
theorem sum_of_distances_is_5_sqrt_2 :
  ∃ (t1 t2 : ℝ),
    curve_C (line_l t1).1 (line_l t1).2 ∧
    curve_C (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    distance point_A (line_l t1) + distance point_A (line_l t2) = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_5_sqrt_2_l1052_105285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_y_l1052_105298

noncomputable def m : ℝ := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

noncomputable def y (x : ℝ) : ℝ := 2 * m * x + 3 / (x - 1) + 1

theorem min_value_y :
  ∀ x > 1, y x ≥ 2 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_y_l1052_105298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perimeter_l1052_105249

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Tangent relation between a circle and a line -/
def Tangent (c : Circle) (l : Line) : Prop :=
  sorry

/-- Intersection of two circles -/
def Intersection (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  sorry

/-- Perimeter of a set of points -/
noncomputable def Perimeter (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem intersection_perimeter (circles : Finset Circle) (n : ℕ) (circumference : ℝ) :
  circles.card = n ∧ 
  n = 4 ∧
  (∀ c ∈ circles, 2 * Real.pi * c.radius = circumference) ∧
  circumference = 24 ∧
  (∀ c1 ∈ circles, ∀ c2 ∈ circles, c1 ≠ c2 → ∃ l : Line, Tangent c1 l ∧ Tangent c2 l) →
  ∃ c1 ∈ circles, ∃ c2 ∈ circles, c1 ≠ c2 ∧ Perimeter (Intersection c1 c2) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perimeter_l1052_105249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_plating_accuracy_l1052_105201

/-- Represents the number of viable bacteria in a sample -/
def viableBacteria : ℕ → ℕ := sorry

/-- Represents the number of colonies on a plate after dilution plating -/
def coloniesOnPlate : ℕ → ℕ := sorry

/-- Represents the dilution factor applied to the sample -/
def dilutionFactor : ℕ → ℝ := sorry

/-- Indicates whether a dilution factor is considered sufficiently high -/
def isSufficientDilution : ℝ → Prop := sorry

/-- Defines an approximate equality relation for natural numbers -/
def approx_equal (a b : ℕ) : Prop := sorry

notation a " ≈ " b => approx_equal a b

/-- Theorem stating that under sufficient dilution, the number of colonies on the plate
    approximately equals the number of viable bacteria in the sample -/
theorem dilution_plating_accuracy (sample : ℕ) 
  (h : isSufficientDilution (dilutionFactor sample)) :
  coloniesOnPlate sample ≈ viableBacteria sample := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_plating_accuracy_l1052_105201
