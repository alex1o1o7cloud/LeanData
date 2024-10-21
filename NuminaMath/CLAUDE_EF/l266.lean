import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_in_triangle_l266_26611

-- Define the plane
variable {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] [FiniteDimensional ℝ P]
variable [Fact (finrank ℝ P = 2)]

-- Define the set S
variable {S : Set P}

-- Define the "can be seen from" relation
def CanBeSeenFrom (S : Set P) (A : P) : Prop :=
  ∀ P ∈ S, ∀ t ∈ Set.Icc (0 : ℝ) 1, (1 - t) • A + t • P ∈ S

-- Define a triangle
structure Triangle (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  A : P
  B : P
  C : P

-- State the theorem
theorem visibility_in_triangle {S : Set P} {T : Triangle P} :
  S.Nonempty →
  CanBeSeenFrom S T.A →
  CanBeSeenFrom S T.B →
  CanBeSeenFrom S T.C →
  ∀ D ∈ convexHull ℝ {T.A, T.B, T.C}, CanBeSeenFrom S D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_in_triangle_l266_26611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l266_26603

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + x - 6 < 0) ↔ ¬(∃ x : ℝ, x^2 + x - 6 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l266_26603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_problem_l266_26606

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define a point on the parabola
def point_on_parabola (p : ℝ) : Prop := parabola p 1 2

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem parabola_and_line_problem :
  ∃ (p : ℝ) (k : ℝ),
    point_on_parabola p ∧
    (∀ (x y : ℝ), parabola p x y → line_through_points 1 0 x y 0 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
      line_through_points 1 0 x₁ y₁ x₂ y₂ ∧
      distance x₁ y₁ x₂ y₂ = 10 ∧
      (∀ (x y : ℝ), line_through_points x₁ y₁ x₂ y₂ x y ↔ y = k*(x-1) ∨ y = -k*(x-1))) ∧
    p = 2 ∧
    k = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_problem_l266_26606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l266_26620

theorem y_value (y : ℕ) 
  (h1 : (Finset.filter (fun x => x ∣ y) (Finset.range (y + 1))).card = 18)
  (h2 : 18 ∣ y)
  (h3 : 24 ∣ y) :
  y = 288 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l266_26620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_negative_one_l266_26605

-- Define the function f as noncomputable
noncomputable def f (x m : ℝ) : ℝ := Real.log ((x + 1) / (x - 1)) + m + 1

-- State the theorem
theorem odd_function_implies_m_equals_negative_one :
  (∀ x, f (-x) m = -(f x m)) → m = -1 := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_negative_one_l266_26605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l266_26643

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l266_26643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_equation_and_point_P_coordinates_l266_26644

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 1}

-- Define the center of circle M
def center_M : ℝ × ℝ := (-1, 2)

-- Define the line that intercepts the chord
def chord_line (x : ℝ) : ℝ := x + 4

-- Define the length of the intercepted chord
noncomputable def chord_length : ℝ := Real.sqrt 2

-- Define the line l on which point P lies
def line_l (x : ℝ) : ℝ := x - 1

-- Define point Q on circle M
def point_Q : ℝ × ℝ := sorry

-- Define the relationship between vectors MP and QM
def vector_relation (P : ℝ × ℝ) : Prop :=
  ∃ (Q : ℝ × ℝ), Q ∈ circle_M ∧ 
  (P.1 - center_M.1, P.2 - center_M.2) = (4 : ℝ) • (center_M.1 - Q.1, center_M.2 - Q.2)

theorem circle_M_equation_and_point_P_coordinates :
  (∀ (x y : ℝ), (x, y) ∈ circle_M ↔ (x + 1)^2 + (y - 2)^2 = 1) ∧
  (∃ (P : ℝ × ℝ), P.2 = line_l P.1 ∧ vector_relation P ∧ 
    (P = (-1, -2) ∨ P = (3, 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_equation_and_point_P_coordinates_l266_26644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n0_x_gt_y_l266_26626

def x : ℕ → ℝ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | 2 => 6
  | (n + 3) => 2 * x (n + 2) + x (n + 1)

def y : ℕ → ℝ
  | 0 => 3  -- Adding case for 0
  | 1 => 3
  | 2 => 9
  | (n + 3) => y (n + 2) + 2 * y (n + 1)

theorem exists_n0_x_gt_y : ∃ n0 : ℕ, ∀ n > n0, x n > y n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n0_x_gt_y_l266_26626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_nine_count_l266_26658

theorem three_digit_divisible_by_nine_count : 
  (Finset.filter (fun n : Nat => 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0) (Finset.range 1000)).card = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_nine_count_l266_26658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_lines_l266_26661

-- Define the circles
def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 8*p.1 + 6*p.2 + 9 = 0}

-- Define the centers and radii
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (4, -3)
def radius2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Define a function to represent the number of common tangent lines
def number_of_common_tangent_lines (c1 c2 : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem common_tangent_lines :
  (distance_between_centers < radius1 + radius2) ∧
  (distance_between_centers > |radius1 - radius2|) →
  number_of_common_tangent_lines circle1 circle2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_lines_l266_26661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divisibility_l266_26685

theorem smallest_prime_divisibility (p : ℕ) : 
  Prime p ∧ 
  (∀ q : ℕ, Prime q ∧ (∀ n ∈ ({823, 618, 3648, 60, 3917, 4203, 1543, 2971} : Set ℕ), (q + 1) % n = 0) → p ≤ q) ∧
  (∀ n ∈ ({823, 618, 3648, 60, 3917, 4203, 1543, 2971} : Set ℕ), (p + 1) % n = 0) →
  p = Nat.lcm 823 (Nat.lcm 618 (Nat.lcm 3648 (Nat.lcm 60 (Nat.lcm 3917 (Nat.lcm 4203 (Nat.lcm 1543 2971)))))) - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divisibility_l266_26685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l266_26609

/-- The area of the region outside a smaller circle and inside two larger circles -/
theorem area_between_circles (r₁ r₂ d : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : d > 0) :
  let area := 2 * (r₁^2 * Real.arccos (r₂ / d) - r₂^2 * Real.arccos ((d^2 + r₂^2 - r₁^2) / (2 * d * r₂)) - 
               1/2 * Real.sqrt ((d + r₁ + r₂) * (d + r₁ - r₂) * (d - r₁ + r₂) * (-d + r₁ + r₂)))
  r₁ = 3 ∧ r₂ = 2 ∧ d = 4 → area = 3/2 * Real.pi - 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l266_26609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_goals_problem_l266_26621

def first_six_matches : List Nat := [2, 5, 4, 3, 6, 2]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem soccer_goals_problem (goals_7 goals_8 : ℕ) 
  (h1 : goals_7 < 10)
  (h2 : goals_8 < 10)
  (h3 : is_integer ((List.sum first_six_matches + goals_7) / 7))
  (h4 : is_integer ((List.sum first_six_matches + goals_7 + goals_8) / 8)) :
  goals_7 * goals_8 = 24 := by
  sorry

#check soccer_goals_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_goals_problem_l266_26621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_locus_equation_l266_26641

/-- Triangle AOB with O as the origin, A and B on the line x = 3, and ∠AOB = π/3 -/
structure TriangleAOB where
  t₁ : ℝ
  t₂ : ℝ
  A : ℝ × ℝ := (3, t₁)
  B : ℝ × ℝ := (3, t₂)
  O : ℝ × ℝ := (0, 0)
  angle_AOB : Real.cos (π/3) = 1/2

/-- The locus of the circumcenter P(x, y) of triangle AOB -/
def circumcenter_locus (x y : ℝ) : Prop :=
  x ≤ 2 ∧ (x - 4)^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that the locus of the circumcenter P(x, y) satisfies the given equation -/
theorem circumcenter_locus_equation :
  ∀ x y, circumcenter_locus x y ↔ 
    (x ≤ 2 ∧ (x - 4)^2 / 4 - y^2 / 12 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_locus_equation_l266_26641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l266_26638

/-- The time it takes Clea to walk down a non-operating escalator, in seconds -/
def time_non_operating : ℝ := 70

/-- The time it takes Clea to walk down an operating escalator, in seconds -/
def time_operating : ℝ := 28

/-- Clea's walking speed, in meters per second -/
noncomputable def clea_speed : ℝ := 1 / time_non_operating

/-- The length of the escalator, in meters -/
noncomputable def escalator_length : ℝ := time_non_operating * clea_speed

/-- The speed of the escalator, in meters per second -/
noncomputable def escalator_speed : ℝ := escalator_length / time_operating - clea_speed

/-- The time it takes Clea to ride down the operating escalator while standing, in seconds -/
noncomputable def time_standing : ℝ := escalator_length / escalator_speed

theorem escalator_ride_time : ⌊time_standing⌋ = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l266_26638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_theorem_l266_26686

open Real

/-- Represents a 2D point/vector -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Scalar product of two 2D vectors -/
def scalarProduct (v w : Point2D) : ℝ := v.x * w.x + v.y * w.y

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : Point2D) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- Vector subtraction -/
def vectorSub (v w : Point2D) : Point2D := Point2D.mk (v.x - w.x) (v.y - w.y)

/-- Vector addition -/
def vectorAdd (v w : Point2D) : Point2D := Point2D.mk (v.x + w.x) (v.y + w.y)

/-- Vector scalar multiplication -/
def vectorScalarMul (k : ℝ) (v : Point2D) : Point2D := Point2D.mk (k * v.x) (k * v.y)

theorem minimum_value_theorem :
  let A : Point2D := Point2D.mk 0 4
  let B : Point2D := Point2D.mk 0 2
  ∀ (t : ℝ) (C : Point2D),
    t < 0 →
    scalarProduct (vectorSub (vectorScalarMul 2 C) A) (vectorSub C B) = 0 →
    4 ≤ magnitude (vectorSub C (vectorAdd (vectorScalarMul ((1/4) * t) A) (vectorScalarMul ((1/2) * (log (-t) - 1)) B))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_theorem_l266_26686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l266_26684

/-- The area of a regular hexagon inscribed in a circle of radius 4 units is 24√3 square units. -/
theorem regular_hexagon_area (r : ℝ) (h : r = 4) : 
  (6 * ((Real.sqrt 3 / 4) * r^2)) = 24 * Real.sqrt 3 := by
  -- Substitute r = 4
  rw [h]
  -- Simplify
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l266_26684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_area_l266_26652

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
noncomputable def smaller_circle : Circle := sorry
noncomputable def larger_circle : Circle := sorry
noncomputable def P : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def A' : Point := sorry
noncomputable def B' : Point := sorry

-- Define helper functions
def distance (p1 p2 : Point) : ℝ := sorry
def circle_area (c : Circle) : ℝ := sorry
def on_circle (p : Point) (c : Circle) : Prop := sorry
def is_tangent_line (p1 p2 p3 : Point) (c1 c2 : Circle) : Prop := sorry

-- Define the conditions
axiom externally_tangent : 
  distance (Point.mk smaller_circle.center.1 smaller_circle.center.2) 
           (Point.mk larger_circle.center.1 larger_circle.center.2) = 
  smaller_circle.radius + larger_circle.radius

axiom common_tangent_PAB : 
  is_tangent_line P A B smaller_circle larger_circle

axiom common_tangent_PAB' : 
  is_tangent_line P A' B' smaller_circle larger_circle

axiom A_on_smaller : on_circle A smaller_circle
axiom A'_on_smaller : on_circle A' smaller_circle
axiom B_on_larger : on_circle B larger_circle
axiom B'_on_larger : on_circle B' larger_circle

axiom PA_length : distance P A = 5
axiom AB_length : distance A B = 5

-- The theorem to prove
theorem smaller_circle_area : 
  circle_area smaller_circle = 6.25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_area_l266_26652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_t_circ_is_brownian_bridge_l266_26642

/-- Ornstein-Uhlenbeck process -/
noncomputable def OrnsteinUhlenbeck (t : ℝ) : ℝ := sorry

/-- Definition of B_t° -/
noncomputable def B_t_circ (t : ℝ) : ℝ :=
  if 0 < t ∧ t < 1 then
    Real.sqrt (t * (1 - t)) * OrnsteinUhlenbeck (1/2) * Real.log (t / (1 - t))
  else
    0

/-- Brownian bridge -/
def is_brownian_bridge (B : ℝ → ℝ) : Prop := sorry

/-- Main theorem: B_t° is a Brownian bridge -/
theorem B_t_circ_is_brownian_bridge :
  is_brownian_bridge B_t_circ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_t_circ_is_brownian_bridge_l266_26642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_first_five_primes_l266_26649

theorem smallest_multiple_of_first_five_primes :
  ∀ n : ℕ, n > 0 → (∀ p : ℕ, p ∈ ({2, 3, 5, 7, 11} : Finset ℕ) → p ∣ n) → n ≥ 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_first_five_primes_l266_26649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l266_26664

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 2)

theorem f_extrema :
  (∀ x y : ℝ, x ∈ Set.Icc 4 5 → y ∈ Set.Icc 4 5 → x ≤ y → f y ≤ f x) →
  (∀ x : ℝ, x ∈ Set.Icc 4 5 → 2/3 ≤ f x ∧ f x ≤ 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 4 5 ∧ f x = 2/3) ∧
  (∃ x : ℝ, x ∈ Set.Icc 4 5 ∧ f x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l266_26664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l266_26608

theorem log_inequality (a b c : ℝ) (ha : a = Real.log 6 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) (hc : c = 3/2) :
  b > c ∧ c > a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l266_26608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_conditions_l266_26674

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def conditionA (t : Triangle) : Prop := t.a * t.b * Real.cos t.C > 0

def conditionB (t : Triangle) : Prop :=
  t.b^2 * (Real.sin t.C)^2 + t.c^2 * (Real.sin t.B)^2 = 2 * t.b * t.c * Real.cos t.B * Real.cos t.C

def conditionC (t : Triangle) : Prop :=
  (t.a - t.b) / (t.c + t.b) = Real.sin t.C / (Real.sin t.A + Real.sin t.B)

noncomputable def conditionD (t : Triangle) : Prop :=
  Real.tan t.A + Real.tan t.B + Real.tan t.C < 0

-- Define what it means for a triangle to be obtuse
def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- Theorem statement
theorem obtuse_triangle_conditions (t : Triangle) :
  (conditionA t → isObtuse t) ∧
  (conditionC t → isObtuse t) ∧
  (conditionD t → isObtuse t) ∧
  ¬(conditionB t → isObtuse t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_conditions_l266_26674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l266_26614

noncomputable def f (a x : ℝ) : ℝ := -a/2 * x^2 + (a+1)*x - Real.log x

theorem f_properties :
  ∀ (a : ℝ), ∀ (x : ℝ), x > 0 →
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 x_min ≤ f 0 y) ∧
  (a > 0 → ∃ (y z : ℝ), y > 0 ∧ z > 0 ∧ y ≠ z ∧ StrictMonoOn (f a) (Set.Ioo y z)) ∧
  (a > 2 ∧ a < 3 →
    ∀ (m : ℝ),
      (∀ (x₁ x₂ : ℝ), x₁ ≥ 1 ∧ x₁ ≤ 2 ∧ x₂ ≥ 1 ∧ x₂ ≤ 2 →
        (a^2 - 1)/2 * m + Real.log 2 > |f a x₁ - f a x₂|) →
      m ≥ 1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l266_26614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sum_zero_l266_26656

/-- The curve equation -/
noncomputable def curve_equation (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem: If y = x is an axis of symmetry of y = (ax + b) / (cx + d), then a + d = 0 -/
theorem symmetry_implies_sum_zero
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = curve_equation a b c d x → x = curve_equation a b c d y) :
  a + d = 0 := by
  sorry

#check symmetry_implies_sum_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sum_zero_l266_26656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_result_l266_26699

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- The first polynomial in the expression -/
noncomputable def p1 : Polynomial ℝ := 2 * Polynomial.X^5 + 3 * Polynomial.X^3 + Polynomial.X - 14

/-- The second polynomial in the expression -/
noncomputable def p2 : Polynomial ℝ := 3 * Polynomial.X^11 - 9 * Polynomial.X^8 + 9 * Polynomial.X^4 + 30

/-- The third polynomial in the expression -/
noncomputable def p3 : Polynomial ℝ := (Polynomial.X^3 + 5)^8

/-- The resulting polynomial from the given expression -/
noncomputable def result : Polynomial ℝ := p1 * p2 - p3

/-- Theorem stating that the degree of the resulting polynomial is 24 -/
theorem degree_of_result : degree result = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_result_l266_26699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_beats_runner_b_l266_26667

/-- The distance in kilometers that both runners cover -/
noncomputable def distance : ℝ := 3

/-- The time in seconds that runner a takes to cover the distance -/
noncomputable def time_a : ℝ := 3 * 60 + 18

/-- The time in seconds that runner b takes to cover the distance -/
noncomputable def time_b : ℝ := 3 * 60 + 40

/-- The speed of runner a in km/s -/
noncomputable def speed_a : ℝ := distance / time_a

/-- The speed of runner b in km/s -/
noncomputable def speed_b : ℝ := distance / time_b

/-- The distance runner a covers in the time it takes runner b to finish -/
noncomputable def distance_a : ℝ := speed_a * time_b

/-- The difference in distance covered by a and b -/
noncomputable def difference : ℝ := distance_a - distance

theorem runner_a_beats_runner_b : 
  ⌊difference * 1000⌋ = 333 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_beats_runner_b_l266_26667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l266_26622

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (3, 0)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_right_focus_to_line :
  distance_point_to_line right_focus.1 right_focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l266_26622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_decreasing_l266_26653

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

-- Define the increasing interval
def increasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi + 5 * Real.pi / 8) (k * Real.pi + 9 * Real.pi / 8)

-- Define the decreasing interval
def decreasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8)

-- Theorem stating that f is increasing on the specified intervals
theorem f_increasing (k : ℤ) : 
  StrictMonoOn f (increasing_interval k) := by sorry

-- Theorem stating that f is decreasing on the specified intervals
theorem f_decreasing (k : ℤ) : 
  StrictAntiOn f (decreasing_interval k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_decreasing_l266_26653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convoy_max_distance_l266_26600

/-- Represents a vehicle in the convoy --/
structure Vehicle where
  efficiency : ℚ  -- Miles per gallon (using rationals instead of reals)
  minFuel : ℚ     -- Minimum fuel required in gallons

/-- Represents the convoy of vehicles --/
structure Convoy where
  suv : Vehicle
  sedan : Vehicle
  motorcycle : Vehicle
  totalFuel : ℚ

/-- Calculates the maximum distance a convoy can travel --/
def maxConvoyDistance (c : Convoy) : ℚ :=
  min (c.suv.efficiency * c.suv.minFuel)
      (min (c.sedan.efficiency * c.sedan.minFuel)
           (c.motorcycle.efficiency * c.motorcycle.minFuel))

/-- The theorem to be proved --/
theorem convoy_max_distance (c : Convoy) :
  c.suv.efficiency = 61/5 →
  c.sedan.efficiency = 52 →
  c.motorcycle.efficiency = 70 →
  c.suv.minFuel = 10 →
  c.sedan.minFuel = 5 →
  c.motorcycle.minFuel = 2 →
  c.totalFuel = 21 →
  maxConvoyDistance c = 122 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convoy_max_distance_l266_26600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_fifth_l266_26660

/-- The function g satisfying the given properties -/
noncomputable def g : ℝ → ℝ := sorry

/-- g is defined on [0, 1] -/
axiom g_domain (x : ℝ) : 0 ≤ x ∧ x ≤ 1 → g x ∈ Set.Icc 0 1

/-- g(0) = 0 -/
axiom g_zero : g 0 = 0

/-- g is monotonically increasing -/
axiom g_monotone {x y : ℝ} : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y

/-- g(1 - x) = 1 - g(x) -/
axiom g_symmetry (x : ℝ) : 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x

/-- g(x/4) = g(x)/2 -/
axiom g_quarter (x : ℝ) : 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2

/-- The main theorem: g(1/5) = 1/4 -/
theorem g_one_fifth : g (1/5) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_fifth_l266_26660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l266_26618

open Set

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ :=
  {x : ℝ | x ≠ -2 ∧ (x^2 + 2*x + 2) / (x + 2) > 1}

-- Define the expected solution set
def expected_set : Set ℝ :=
  (Ioc (-2) (-1)) ∪ (Ioi 0)

-- Theorem stating that the solution set equals the expected set
theorem inequality_solution_set :
  solution_set = expected_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l266_26618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_squared_distance_l266_26627

/-- The minimum value of (a-c)^2 + (b-d)^2 is 8, given b = -a^2 + 3*ln(a) and d = c + 2 -/
theorem min_value_squared_distance :
  (∀ a c : ℝ, (a - c)^2 + (-a^2 + 3 * Real.log a - (c + 2))^2 ≥ 8) ∧ 
  (∃ a₀ c₀ : ℝ, (a₀ - c₀)^2 + (-a₀^2 + 3 * Real.log a₀ - (c₀ + 2))^2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_squared_distance_l266_26627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_speed_proof_l266_26630

/-- Calculates the speed given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem jim_speed_proof :
  let distance : ℝ := 84  -- km
  let time : ℝ := 7/4     -- hours (1.75 as a fraction)
  calculate_speed distance time = 48 := by
  unfold calculate_speed
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_speed_proof_l266_26630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_for_square_sequence_range_a_for_square_minus_a_sequence_l266_26633

/-- Property P(t) for a sequence a_n --/
def property_P (a : ℕ+ → ℝ) (t : ℝ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → (a m - a n) / ((m : ℝ) - (n : ℝ)) ≥ t

/-- Sequence a_n = n^2 --/
def seq_square (n : ℕ+) : ℝ := (n : ℝ) ^ 2

/-- Sequence a_n = n^2 - a/n --/
noncomputable def seq_square_minus_a (a : ℝ) (n : ℕ+) : ℝ := (n : ℝ) ^ 2 - a / (n : ℝ)

theorem max_t_for_square_sequence :
  (∀ t : ℝ, property_P seq_square t → t ≤ 3) ∧
  property_P seq_square 3 := by sorry

theorem range_a_for_square_minus_a_sequence :
  ∀ a : ℝ, property_P (seq_square_minus_a a) 7 → a ≥ 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_for_square_sequence_range_a_for_square_minus_a_sequence_l266_26633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l266_26677

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (2, 1)

-- Define a general point P on the locus
def point_P : ℝ × ℝ → Prop
| (x, y) => x + y - 5 = 0

-- State the theorem
theorem locus_of_P :
  ∀ (x y : ℝ),
  (∃ (B C : ℝ × ℝ),
    -- A is inside the circle M
    circle_M (point_A.1) (point_A.2) ∧
    -- B and C are on the circle M
    circle_M B.1 B.2 ∧ circle_M C.1 C.2 ∧
    -- A, B, and C are collinear
    ∃ (m b : ℝ), (point_A.2 = m * point_A.1 + b) ∧
                 (B.2 = m * B.1 + b) ∧
                 (C.2 = m * C.1 + b) ∧
    -- P is the intersection of tangents at B and C
    ∃ (t1 t2 : ℝ → ℝ),
      (t1 B.1 = B.2) ∧ (t2 C.1 = C.2) ∧
      (x = (t2 C.1 - t1 B.1) / ((t1 C.1 - t1 B.1) / (C.1 - B.1) - (t2 C.1 - t2 B.1) / (C.1 - B.1))) ∧
      (y = t1 x + (t1 B.1 - B.2))) →
  point_P (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l266_26677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l266_26696

noncomputable def curve (x : ℝ) : ℝ := x^4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 2009 = 0

noncomputable def curve_deriv : ℝ → ℝ := fun x => 4 * x^3

def is_tangent_line (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) : Prop :=
  l x₀ y₀ ∧ curve x₀ = y₀ ∧ ∀ x y, l x y → (y - y₀) = (curve_deriv x₀) * (x - x₀)

theorem tangent_line_equation (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) :
  is_tangent_line l x₀ y₀ →
  (∀ x y, l x y → perpendicular_line x y) →
  (∀ x y, l x y ↔ 4*x - y - 3 = 0) :=
by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l266_26696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_six_l266_26680

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 : ℝ)^(x^3 - 2*x^2 - 2*x + 1) = (8 : ℝ)^(x - 3)

-- Theorem statement
theorem sum_of_solutions_is_six :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x) ∧ (∀ x : ℝ, equation x → x ∈ S) ∧ (S.sum id = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_six_l266_26680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_is_31_l266_26671

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := { n : ℕ // 2000 ≤ n ∧ n ≤ 2099 }

/-- Calculates the payment for a given divisor -/
def payment (d : ℕ) : ℕ :=
  match d with
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | 9 => 9
  | 11 => 11
  | _ => 0

/-- Calculates the total payment for a number based on its divisibility -/
def totalPayment (n : FourDigitNumber) : ℕ :=
  (List.range 12).foldl (fun acc d => acc + if n.val % d = 0 then payment d else 0) 0

/-- The maximum possible payment for any number of the form 20** -/
def maxPayment : ℕ := 31

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), totalPayment n = maxPayment ∧
  ∀ (m : FourDigitNumber), totalPayment m ≤ maxPayment := by
  sorry

#eval totalPayment ⟨2079, by simp [Nat.le_refl]⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_is_31_l266_26671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_30000_l266_26628

/-- Represents the investment scenario for Tommy --/
structure InvestmentScenario where
  total_investment : ℚ
  fund_a_return : ℚ
  fund_b_return : ℚ
  fund_c_return : ℚ

/-- Calculates the maximum guaranteed profit for a given investment scenario --/
noncomputable def max_guaranteed_profit (scenario : InvestmentScenario) : ℚ :=
  let a := scenario.total_investment * 3 / 13
  let b := scenario.total_investment * 4 / 13
  let c := scenario.total_investment * 6 / 13
  min (min (a * scenario.fund_a_return) (b * scenario.fund_b_return)) (c * scenario.fund_c_return) - scenario.total_investment

/-- The theorem stating that the maximum guaranteed profit is $30,000 --/
theorem max_profit_is_30000 :
  ∃ (scenario : InvestmentScenario),
    scenario.total_investment = 90000 ∧
    scenario.fund_a_return = 3 ∧
    scenario.fund_b_return = 4 ∧
    scenario.fund_c_return = 6 ∧
    max_guaranteed_profit scenario = 30000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_30000_l266_26628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_complete_nat_sets_l266_26659

/-- A set of natural numbers is complete if for any natural numbers a and b,
    if a + b is in the set, then ab is also in the set. -/
def IsCompleteSet (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

/-- The set of all complete sets of natural numbers -/
def CompleteNatSets : Set (Set ℕ) :=
  {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, Set.univ}

theorem characterization_of_complete_nat_sets :
  ∀ A : Set ℕ, A.Nonempty → (IsCompleteSet A ↔ A ∈ CompleteNatSets) := by
  sorry

#check characterization_of_complete_nat_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_complete_nat_sets_l266_26659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_orthocenter_equilateral_l266_26639

/-- Triangle ABC with centroid G and orthocenter H -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- G is the centroid of triangle ABC -/
def is_centroid (t : TriangleABC) : Prop :=
  t.G = ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- H is the orthocenter of triangle ABC -/
def is_orthocenter (t : TriangleABC) : Prop :=
  ∃ (a_c : ℝ × ℝ), t.H = (t.A.1 + t.B.1 + t.C.1 - a_c.1, t.A.2 + t.B.2 + t.C.2 - a_c.2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- AG = AH -/
def AG_eq_AH (t : TriangleABC) : Prop :=
  distance t.A t.G = distance t.A t.H

/-- Triangle ABC is equilateral -/
def is_equilateral (t : TriangleABC) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

/-- Angle A in degrees -/
noncomputable def angle_A (t : TriangleABC) : ℝ := sorry

theorem centroid_orthocenter_equilateral (t : TriangleABC) 
  (h_centroid : is_centroid t) 
  (h_orthocenter : is_orthocenter t) 
  (h_AG_AH : AG_eq_AH t) : 
  is_equilateral t ∧ angle_A t = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_orthocenter_equilateral_l266_26639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_triangle_area_l266_26673

noncomputable section

-- Define the rectangle
def rectangle_length : ℝ := 6
def rectangle_width : ℝ := 4

-- Define the triangles
def triangle1_area_factor : ℝ := 5
def triangle2_base : ℝ := 8

-- Define the ratios
def rectangle_to_triangle1_ratio : ℝ := 2 / 5
def triangle2_to_triangle1_ratio : ℝ := 3 / 5

-- Theorem statement
theorem combined_triangle_area :
  let rectangle_area := rectangle_length * rectangle_width
  let triangle1_area := (rectangle_area * 5) / 2
  let triangle2_area := triangle2_to_triangle1_ratio * triangle1_area
  triangle1_area + triangle2_area = 96 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_triangle_area_l266_26673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_points_l266_26602

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x^3 / 3 - x^2 - x + 1

-- Define the derivative of the curve function
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 2 ↔ (x = 3 ∧ y = -2) ∨ (x = -1 ∧ y = 2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_points_l266_26602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_square_theorem_l266_26678

/-- Represents a 9x9 board with colored squares -/
def Board := Fin 9 → Fin 9 → Bool

/-- Counts the number of red squares in a board -/
def count_red (b : Board) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 9)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 9)) fun j =>
      if b i j then 1 else 0)

/-- Checks if a 2x2 block starting at (i, j) has at least 3 red squares -/
def has_three_red (b : Board) (i j : Fin 8) : Prop :=
  (if b i j then 1 else 0) + 
  (if b i (j.succ) then 1 else 0) + 
  (if b (i.succ) j then 1 else 0) + 
  (if b (i.succ) (j.succ) then 1 else 0) ≥ 3

/-- The main theorem -/
theorem red_square_theorem (b : Board) 
  (h : count_red b = 46) : 
  ∃ (i j : Fin 8), has_three_red b i j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_square_theorem_l266_26678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_A_l266_26613

noncomputable def A : ℝ := (21*62 + 22*63 + 23*64 + 24*65 + 25*66) / (21*61 + 22*62 + 23*63 + 24*64 + 25*65) * 199

theorem integer_part_of_A : ⌊A⌋ = 202 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_A_l266_26613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_school_year_hours_l266_26675

/-- Calculates the number of hours per week needed to work during the school year,
    given summer work details and desired school year earnings. -/
noncomputable def school_year_hours_per_week (summer_hours_per_week : ℚ) (summer_weeks : ℚ) 
  (summer_earnings : ℚ) (school_year_weeks : ℚ) (school_year_earnings : ℚ) : ℚ :=
  (school_year_earnings * summer_hours_per_week * summer_weeks) / 
  (summer_earnings * school_year_weeks)

/-- Theorem stating that given Julie's work conditions, she needs to work 14.4 hours
    per week during the school year to earn $6000. -/
theorem julie_school_year_hours : 
  school_year_hours_per_week 48 10 5000 40 6000 = 72/5 := by
  -- Unfold the definition and simplify
  unfold school_year_hours_per_week
  -- Perform the calculation
  norm_num
  -- QED

#eval (72 : ℚ) / 5 -- This should output 14.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_school_year_hours_l266_26675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_width_is_40_l266_26645

/-- A rectangular window with glass panes -/
structure Window where
  num_panes : ℕ
  rows : ℕ
  columns : ℕ
  pane_height_to_width_ratio : ℚ
  border_width : ℕ

/-- Calculate the total width of the window in inches -/
def window_width (w : Window) : ℕ :=
  w.columns * (w.pane_height_to_width_ratio.den : ℕ) + (w.columns + 1) * w.border_width

/-- The specific window described in the problem -/
def problem_window : Window :=
  { num_panes := 6
  , rows := 2
  , columns := 3
  , pane_height_to_width_ratio := 3 / 1
  , border_width := 1 }

theorem window_width_is_40 : 
  window_width problem_window = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_width_is_40_l266_26645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_production_times_l266_26651

-- Define the production rates for two workers
def worker_production_rate (time : ℝ) (rate1 rate2 : ℝ) : Prop :=
  rate1 * time - rate2 * time = 5 ∧ rate1 > rate2

-- Define the time difference to produce 100 parts
def time_difference (time1 time2 : ℝ) : Prop :=
  time2 - time1 = 2 ∧ time1 > 0 ∧ time2 > 0

-- Theorem statement
theorem worker_production_times
  (rate1 rate2 time1 time2 : ℝ)
  (h1 : worker_production_rate 2 rate1 rate2)
  (h2 : time_difference time1 time2)
  (h3 : rate1 * time1 = 100)
  (h4 : rate2 * time2 = 100) :
  time1 = 8 ∧ time2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_production_times_l266_26651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l266_26636

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side of length 4, height 6, and area 27,
    the other side has length 5 -/
theorem trapezium_other_side_length
  (t : Trapezium)
  (h1 : t.side1 = 4)
  (h2 : t.height = 6)
  (h3 : t.area = 27)
  (h4 : trapeziumArea t = t.area) :
  t.side2 = 5 := by
  sorry

#check trapezium_other_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l266_26636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l266_26672

/-- Represents an ellipse with equation x²/(10-m) + y²/(m-2) = 1 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Indicates that the major axis of an ellipse is along the x-axis -/
def majorAxisAlongX {m : ℝ} (e : Ellipse m) : Prop :=
  10 - m > m - 2

/-- Represents the focal length of an ellipse -/
noncomputable def focalLength {m : ℝ} (e : Ellipse m) : ℝ := 
  Real.sqrt ((10 - m) - (m - 2))

/-- 
Given an ellipse with equation x²/(10-m) + y²/(m-2) = 1,
if its major axis is along the x-axis and its focal length is 4,
then m equals 4.
-/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : majorAxisAlongX e) 
  (h2 : focalLength e = 4) : 
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l266_26672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l266_26689

/-- The constant term in the expansion of (x^4 + x^2 + 7)(x^6 + x^3 + 3)(2x^2 + 9) -/
def constant_term : ℕ := 189

/-- The first polynomial in the product -/
def p1 (x : ℝ) : ℝ := x^4 + x^2 + 7

/-- The second polynomial in the product -/
def p2 (x : ℝ) : ℝ := x^6 + x^3 + 3

/-- The third polynomial in the product -/
def p3 (x : ℝ) : ℝ := 2*x^2 + 9

/-- The polynomial part of the expansion -/
def polynomial_part (x : ℝ) : ℝ := sorry

theorem constant_term_expansion : 
  ∀ x : ℝ, (p1 x) * (p2 x) * (p3 x) = constant_term + x * (polynomial_part x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l266_26689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_circular_ellipse_shape_l266_26655

open Real

-- Define the ellipse equation
def ellipse_equation (α : ℝ) (x y : ℝ) : Prop :=
  x^2 / (tan α) + y^2 / (tan α^2 + 1) = 1

-- Define the condition on α
def alpha_condition (α : ℝ) : Prop :=
  0 < α ∧ α < π/2

-- Define the most circular ellipse equation
def most_circular_ellipse (x y : ℝ) : Prop :=
  x^2 + y^2/2 = 1

-- Theorem statement
theorem most_circular_ellipse_shape (α : ℝ) :
  alpha_condition α →
  (∀ x y : ℝ, ellipse_equation α x y ↔ most_circular_ellipse x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_circular_ellipse_shape_l266_26655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_count_l266_26694

def distribute_candy (n : ℕ) (red_min blue_min : ℕ) (white_max : ℕ) : ℕ :=
  let count := fun r b => Nat.choose n r * Nat.choose (n - r) b
  (Finset.range (n - blue_min + 1 - red_min)).sum fun r =>
    (Finset.range (n - r + 1 - blue_min)).sum fun b =>
      let r' := r + red_min
      let b' := b + blue_min
      if r' + b' ≤ n ∧ n - r' - b' ≤ white_max then count r' b' else 0

theorem candy_distribution_count :
  distribute_candy 8 2 2 3 = 2576 := by sorry

#eval distribute_candy 8 2 2 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_count_l266_26694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_quiz_scores_l266_26691

theorem lucas_quiz_scores :
  let first_three : List ℕ := [92, 78, 65]
  let total_quizzes : ℕ := 5
  let mean_score : ℕ := 84
  ∀ (scores : List ℕ),
    scores.length = total_quizzes →
    scores.take 3 = first_three →
    scores.sum / total_quizzes = mean_score →
    ∀ x ∈ scores, x < 95 →
    scores.Nodup →
    scores = [94, 92, 91, 78, 65].reverse :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_quiz_scores_l266_26691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_line_l_cartesian_max_distance_C_to_l_l266_26617

-- Define the curve C
noncomputable def curve_C (θ : Real) : Real × Real :=
  (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the line l in polar coordinates
def line_l (ρ θ : Real) : Prop :=
  Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4) = 3

-- Theorem for the ordinary equation of curve C
theorem curve_C_equation (x y : Real) : 
  (∃ θ, curve_C θ = (x, y)) ↔ x^2 / 3 + y^2 = 1 := by
  sorry

-- Theorem for the Cartesian equation of line l
theorem line_l_cartesian (x y : Real) :
  (∃ ρ θ, line_l ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ x - y + 3 = 0 := by
  sorry

-- Theorem for the maximum distance from curve C to line l
theorem max_distance_C_to_l :
  ∃ d, d = (5 * Real.sqrt 2) / 2 ∧
  ∀ θ, ∃ ρ θ', line_l ρ θ' →
    d ≥ Real.sqrt ((Real.sqrt 3 * Real.cos θ - ρ * Real.cos θ')^2 + 
                   (Real.sin θ - ρ * Real.sin θ')^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_line_l_cartesian_max_distance_C_to_l_l266_26617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_ratio_bounds_l266_26629

theorem two_digit_ratio_bounds :
  ∀ x y : ℕ,
    0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ x ≠ 0 →
    (let n := 10 * x + y;
     let r := (n : ℚ) / (x + y : ℚ)
     19/10 ≤ r ∧ r ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_ratio_bounds_l266_26629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_angles_l266_26601

/-- 
Two circles intersect at specific angles based on the relationship between 
their radii and the distance between their centers.
-/
theorem circle_intersection_angles (R r c : ℝ) (hR : R > 0) (hr : r > 0) (hc : c > 0) :
  (c = R - r → ∃ θ, θ = 0) ∧
  (c = Real.sqrt (R^2 + r^2) → ∃ θ, θ = π/2) ∧
  (c = R + r → ∃ θ, θ = π) :=
by
  sorry

#check circle_intersection_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_angles_l266_26601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pheromone_has_three_elements_pheromone_has_29_atoms_l266_26634

-- Define a structure to represent a chemical element
structure Element where
  symbol : String
  atomic_number : ℕ
deriving DecidableEq

-- Define a structure to represent a chemical compound
structure Compound where
  formula : List (Element × ℕ)

-- Define the pheromone compound
def pheromone : Compound :=
  { formula := [
      ({symbol := "C", atomic_number := 6}, 10),
      ({symbol := "H", atomic_number := 1}, 18),
      ({symbol := "O", atomic_number := 8}, 1)
    ]
  }

-- Theorem: The pheromone is composed of exactly three elements
theorem pheromone_has_three_elements : 
  (pheromone.formula.map Prod.fst).toFinset.card = 3 := by sorry

-- Theorem: A molecule of the pheromone contains 29 atoms
theorem pheromone_has_29_atoms :
  (pheromone.formula.map (λ (e, n) => n)).sum = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pheromone_has_three_elements_pheromone_has_29_atoms_l266_26634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l266_26692

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  (∃ a b c d e : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 6, 7, 8} ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e)

theorem smallest_valid_divisible_by_six :
  ∀ n : ℕ, is_valid_number n → n % 6 = 0 → n ≥ 12678 := by
  sorry

#check smallest_valid_divisible_by_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l266_26692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_saves_time_l266_26666

/-- Represents the time saved by taking Route Y instead of Route X --/
noncomputable def time_saved (route_x_distance : ℝ) (route_x_speed : ℝ) 
               (route_y_distance : ℝ) (route_y_normal_speed : ℝ) 
               (route_y_construction_distance : ℝ) (route_y_construction_speed : ℝ) : ℝ :=
  let route_x_time := route_x_distance / route_x_speed
  let route_y_normal_time := (route_y_distance - route_y_construction_distance) / route_y_normal_speed
  let route_y_construction_time := route_y_construction_distance / route_y_construction_speed
  let route_y_time := route_y_normal_time + route_y_construction_time
  (route_x_time - route_y_time) * 60

/-- Theorem stating that Route Y saves 2.4 minutes compared to Route X --/
theorem route_y_saves_time : 
  time_saved 8 40 7 50 1 25 = 2.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_saves_time_l266_26666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l266_26616

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (α γ : ℂ) (z : ℂ) : ℂ := (3 + 2*i)*z^3 + (3 + 2*i)*z^2 + α*z + γ

-- State the theorem
theorem min_abs_alpha_plus_gamma :
  ∃ α γ : ℂ, (f α γ 1).im = 0 ∧ (f α γ i).im = 0 ∧
  (∀ α' γ' : ℂ, (f α' γ' 1).im = 0 → (f α' γ' i).im = 0 → 
  Complex.abs α + Complex.abs γ ≤ Complex.abs α' + Complex.abs γ') ∧
  Complex.abs α + Complex.abs γ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l266_26616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l266_26683

/-- Given a finite set M with 10 elements, and two disjoint subsets A and B of M
    with 2 and 3 elements respectively, this theorem states that the number of
    subsets X of M such that neither A nor B is a subset of X is 672. -/
theorem subset_count (M A B : Finset ℕ) : 
  M.card = 10 → 
  A ⊆ M → 
  B ⊆ M → 
  A ∩ B = ∅ → 
  A.card = 2 → 
  B.card = 3 → 
  (M.powerset.filter (fun X => ¬(A ⊆ X) ∧ ¬(B ⊆ X))).card = 672 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l266_26683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l266_26623

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := n + 1

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := n * 2^n

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := (n - 1) * 2^(n + 1) + 2

-- Common difference of a_n
noncomputable def d : ℝ := 1

-- Sum of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n * (2 * (a 1) + (n - 1) * d) / 2

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = d) ∧  -- a_n is arithmetic with common difference d
  (d ≠ 0) ∧                                 -- common difference is not 0
  (S 3 = 9) ∧                               -- S_3 = 9
  ((a 3)^2 = (a 1) * (a 7)) ∧               -- a_1, a_3, a_7 form geometric sequence
  (∀ n : ℕ, n ≥ 1 → b n = (a n - 1) * 2^n)  -- definition of b_n
  →
  (∀ n : ℕ, n ≥ 1 → a n = n + 1) ∧          -- general formula for a_n
  (∀ n : ℕ, n ≥ 1 → T n = (n - 1) * 2^(n + 1) + 2)  -- sum of first n terms of b_n
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l266_26623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l266_26668

/-- The directrix of a parabola y = ax^2 + c is y = -1/(4a) + c -/
noncomputable def directrix (a : ℝ) (c : ℝ) : ℝ := -1/(4*a) + c

/-- The parabola equation y = 4x^2 - 3 -/
def parabola (x : ℝ) : ℝ := 4*x^2 - 3

theorem parabola_directrix :
  directrix 4 (-3) = -49/16 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l266_26668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_equations_l266_26647

/-- Given two non-zero real numbers m and n, and a complex number z satisfying
    |z + nⁱ| + |z - mⁱ| = n and |z + nⁱ| - |z - mⁱ| = -m,
    the shape formed in the complex plane is an ellipse with foci at (0, -n) and (0, m),
    where m < 0 and n > 0. -/
theorem ellipse_from_equations (m n : ℝ) (z : ℂ) 
    (hm : m ≠ 0) (hn : n ≠ 0)
    (eq1 : Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n)
    (eq2 : Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m) :
  ∃ (ellipse : Set ℂ),
    ellipse = {z : ℂ | Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n} ∧
    m < 0 ∧ n > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_equations_l266_26647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_selling_price_l266_26637

/-- The original selling price of Bill's product -/
noncomputable def original_selling_price : ℝ := 770

/-- The original purchase price of Bill's product -/
noncomputable def original_purchase_price : ℝ := original_selling_price / 1.1

/-- The new purchase price (10% less than the original) -/
noncomputable def new_purchase_price : ℝ := 0.9 * original_purchase_price

/-- The new selling price (30% profit on the new purchase price) -/
noncomputable def new_selling_price : ℝ := 1.3 * new_purchase_price

theorem bill_selling_price :
  (original_selling_price = 1.1 * original_purchase_price) ∧
  (new_selling_price = original_selling_price + 49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_selling_price_l266_26637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steamboat_speed_theorem_l266_26624

/-- The initial speed of two steamboats before receiving a radio signal to increase their speed -/
noncomputable def initial_speed (v₁ v₂ t₁ t₂ : ℝ) : ℝ :=
  (60 * v₁ * v₂) / (v₁ * t₂ - v₂ * t₁)

theorem steamboat_speed_theorem 
  (v₁ v₂ t₁ t₂ : ℝ) 
  (h1 : v₁ > 0) 
  (h2 : v₂ > 0) 
  (h3 : t₁ ≥ 0) 
  (h4 : t₂ ≥ 0) 
  (h5 : v₁ * t₂ ≠ v₂ * t₁) :
  ∃ (v : ℝ), 
    v > 0 ∧ 
    (v + v₁) * (1 + t₁ / 60) = (v + v₂) * (2 + t₂ / 60) ∧
    v = initial_speed v₁ v₂ t₁ t₂ := by
  sorry

#check steamboat_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steamboat_speed_theorem_l266_26624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_8_real_l266_26688

theorem sqrt_x_plus_8_real (x : ℝ) : x + 8 ≥ 0 ↔ x ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_8_real_l266_26688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l266_26670

-- Define the speeds for both cars
noncomputable def a : ℝ := 30
noncomputable def b : ℝ := 40
noncomputable def c : ℝ := 50
noncomputable def d : ℝ := 30
noncomputable def e : ℝ := 50
noncomputable def f : ℝ := 40

-- Define the average speed for Car P
noncomputable def w : ℝ := 3 / (1/a + 1/b + 1/c)

-- Define the average speed for Car Q
noncomputable def z : ℝ := (d + e + f) / 3

-- Theorem: The average speed of Car P is less than the average speed of Car Q
theorem car_speed_comparison : w < z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l266_26670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l266_26604

theorem sin_minus_cos_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.sin (2 * α) = -1/3) : 
  Real.sin α - Real.cos α = -2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l266_26604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_perimeter_l266_26679

/-- Represents a rectangular fence with posts --/
structure RectangularFence where
  num_posts : ℕ
  post_spacing : ℝ
  width_posts : ℕ
  length_posts : ℕ

/-- Calculates the outer perimeter of a rectangular fence --/
def outer_perimeter (fence : RectangularFence) : ℝ :=
  2 * (fence.post_spacing * ((fence.width_posts - 1) : ℝ) + 
       fence.post_spacing * ((fence.length_posts - 1) : ℝ))

/-- Theorem stating the outer perimeter of the specific fence --/
theorem fence_perimeter : 
  ∀ (fence : RectangularFence), 
    fence.num_posts = 36 ∧ 
    fence.post_spacing = 6 ∧ 
    fence.width_posts = 6 ∧ 
    fence.length_posts = 14 →
    outer_perimeter fence = 216 := by
  sorry

#eval outer_perimeter { num_posts := 36, post_spacing := 6, width_posts := 6, length_posts := 14 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_perimeter_l266_26679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l266_26654

/-- Given vectors in ℝ² --/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

/-- Parallel vectors --/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- Perpendicular vectors --/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Vector addition --/
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

/-- Scalar multiplication --/
def vec_mul (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (t * v.1, t * v.2)

/-- Vector subtraction --/
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

/-- Vector magnitude --/
noncomputable def vec_mag (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem :
  (∀ k : ℝ, parallel (vec_add a (vec_mul k c)) (vec_sub (vec_mul 2 b) a) → k = -16/13) ∧
  (∀ d : ℝ × ℝ, perpendicular (vec_add a b) (vec_sub d c) ∧ vec_mag (vec_sub d c) = Real.sqrt 5 →
    d = (6, 0) ∨ d = (2, 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l266_26654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l266_26631

/-- The function f that represents a rotation in the complex plane -/
noncomputable def f (z : ℂ) : ℂ := ((1 + Complex.I * Real.sqrt 3) * z + (4 * Real.sqrt 3 - 12 * Complex.I)) / 2

/-- The complex number c that the function rotates around -/
noncomputable def c : ℂ := -2 * Real.sqrt 3 - 2 * Complex.I

/-- Theorem stating that c is the fixed point of f -/
theorem rotation_fixed_point : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l266_26631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l266_26687

theorem impossible_arrangement : ¬ ∃ (M : Matrix (Fin 8) (Fin 8) ℕ), 
  (∀ i j, M i j ∈ Finset.range 65 \ {0}) ∧ 
  (∀ i j i' j', (i ≠ i' ∨ j ≠ j') → M i j ≠ M i' j') ∧
  (∀ i j, i < 7 ∧ j < 7 → |Int.ofNat (M i j * M (i+1) (j+1)) - Int.ofNat (M i (j+1) * M (i+1) j)| = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l266_26687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_z_in_third_quadrant_iff_l266_26610

noncomputable def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - Complex.I * m - 1 - 2 * Complex.I

theorem z_purely_imaginary_iff (m : ℝ) : 
  z m ∈ Set.range (λ y : ℝ => Complex.I * y) ↔ m = 1 := by sorry

theorem z_in_third_quadrant_iff (m : ℝ) :
  (z m).re < 0 ∧ (z m).im < 0 ↔ -1 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_z_in_third_quadrant_iff_l266_26610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_similar_triangles_perimeter_proof_l266_26662

/-- Given two similar triangles, one with sides 12, 12, and 16, and the other with longest side 40,
    the perimeter of the larger triangle is 100. -/
theorem similar_triangles_perimeter (perimeter_large : ℝ) : Prop :=
  let sides_small : Fin 3 → ℝ := λ i ↦ match i with
    | 0 => 12
    | 1 => 12
    | 2 => 16
  let longest_side_small := 16
  let longest_side_large := 40
  let perimeter_small := (sides_small 0) + (sides_small 1) + (sides_small 2)
  let ratio := longest_side_large / longest_side_small
  (∀ i : Fin 3, sides_small i > 0) ∧
  longest_side_small = max (sides_small 0) (max (sides_small 1) (sides_small 2)) ∧
  perimeter_large = ratio * perimeter_small ∧
  perimeter_large = 100

/-- Proof of the theorem -/
theorem similar_triangles_perimeter_proof : similar_triangles_perimeter 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_similar_triangles_perimeter_proof_l266_26662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_smaller_cubes_l266_26665

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes -/
structure CubeSet where
  cubes : List Cube

def volume (c : Cube) : ℕ := c.edge ^ 3

/-- The original cube with edge length 4 cm -/
def originalCube : Cube := ⟨4⟩

/-- The set of smaller cubes that the original cube is cut into -/
def smallerCubes : CubeSet := sorry

/-- All smaller cubes have whole number edge lengths -/
axiom smaller_cubes_whole_edge : ∀ c, c ∈ smallerCubes.cubes → c.edge > 0

/-- The smaller cubes are not all the same size -/
axiom not_all_same_size : ∃ c1 c2, c1 ∈ smallerCubes.cubes ∧ c2 ∈ smallerCubes.cubes ∧ c1.edge ≠ c2.edge

/-- The total volume of smaller cubes equals the volume of the original cube -/
axiom volume_conservation : (smallerCubes.cubes.map volume).sum = volume originalCube

theorem number_of_smaller_cubes : smallerCubes.cubes.length = 57 := by
  sorry

#check number_of_smaller_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_smaller_cubes_l266_26665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_length_relation_l266_26676

-- Define the triangles and their properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define similarity between triangles
def SimilarTriangles (t₁ t₂ : Triangle) : Prop := sorry

-- Define right triangle
def RightTriangle (t : Triangle) (rightAngleVertex : ℝ × ℝ) : Prop := sorry

-- Define length function
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem similar_triangles_length_relation 
  (XYZ PQR : Triangle)
  (h_similar : SimilarTriangles XYZ PQR)
  (h_right : RightTriangle XYZ XYZ.A)
  (h_YZ : length XYZ.B XYZ.C = 35)
  (h_XY : length XYZ.A XYZ.B = 20)
  (h_QR : length PQR.B PQR.C = 14) :
  length PQR.A PQR.C = 8 := by
  sorry

#check similar_triangles_length_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_length_relation_l266_26676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l266_26697

-- Define the function f(x)
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x) - 1

-- State the theorem
theorem omega_range_for_three_zeros (ω : ℝ) :
  ω > 0 →
  (∃ x₁ x₂ x₃ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 2 * Real.pi ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0 ∧
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f ω x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  2 ≤ ω ∧ ω < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l266_26697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_with_squared_min_l266_26698

def S : Finset ℤ := {2, 16, -4, 9, -2}

theorem smallest_sum_with_squared_min :
  ∃ a b c : ℤ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  let min_val := min a (min b c)
  let sum := min_val^2 + (if a = min_val then b + c else if b = min_val then a + c else a + b)
  sum = 16 ∧ ∀ x y z : ℤ, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
    let min_xyz := min x (min y z)
    let sum_xyz := min_xyz^2 + (if x = min_xyz then y + z else if y = min_xyz then x + z else x + y)
    sum_xyz ≥ 16 :=
by
  sorry

#check smallest_sum_with_squared_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_with_squared_min_l266_26698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l266_26625

/-- Given angles α and β with vertices at the origin and initial sides on the positive x-axis,
    prove that cos α = 56/65 under the given conditions. -/
theorem cos_alpha_value (α β : ℝ) : 
  0 < α ∧ α < π → 
  0 < β ∧ β < π → 
  Real.cos β = -5/13 →
  Real.sin (α + β) = 3/5 →
  Real.cos α = 56/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l266_26625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_partition_exists_l266_26663

/-- A type representing a point on a circle with an associated value --/
structure CirclePoint where
  angle : ℝ
  value : ℝ
  value_pos : 0 < value
  value_le_one : value ≤ 1

/-- A type representing a partition of the circle into three arcs --/
structure CirclePartition where
  cut1 : ℝ
  cut2 : ℝ
  cut3 : ℝ
  distinct : cut1 ≠ cut2 ∧ cut2 ≠ cut3 ∧ cut3 ≠ cut1

/-- Function to calculate the sum of values in an arc --/
noncomputable def arcSum (points : List CirclePoint) (start : ℝ) (endp : ℝ) : ℝ :=
  sorry

/-- Theorem stating that there exists a partition satisfying the condition --/
theorem circle_partition_exists (points : List CirclePoint) :
  ∃ (partition : CirclePartition),
    ∀ (i j : Fin 3),
      |arcSum points partition.cut1 partition.cut2 -
       arcSum points partition.cut2 partition.cut3| ≤ 1 ∧
      |arcSum points partition.cut2 partition.cut3 -
       arcSum points partition.cut3 partition.cut1| ≤ 1 ∧
      |arcSum points partition.cut3 partition.cut1 -
       arcSum points partition.cut1 partition.cut2| ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_partition_exists_l266_26663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l266_26619

def M : Set ℝ := { x : ℝ | x^2 - x ≤ 0 }

def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = 1 - |x| ∧ y > 0 }

theorem intersection_M_N : M ∩ N = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l266_26619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_decreasing_function_condition_l266_26646

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 / x - 2 * a / x^3

theorem tangent_parallel_to_x_axis (a : ℝ) :
  f_derivative a 1 = 0 → a = 1 := by sorry

theorem decreasing_function_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f_derivative a x ≤ 0) → a ≥ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_decreasing_function_condition_l266_26646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_equation_solve_cubic_equation_l266_26695

-- Part 1
theorem solve_quadratic_equation (x : ℝ) : 
  4 * (x - 1)^2 = 8 ↔ x = Real.sqrt 2 + 1 ∨ x = -Real.sqrt 2 + 1 := by sorry

-- Part 2
theorem solve_cubic_equation (x : ℝ) : 
  2 * x^3 = 8 → |x - 1.59| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_equation_solve_cubic_equation_l266_26695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_stack_thickness_l266_26648

theorem paper_stack_thickness (bundle_sheets : ℕ) (bundle_thickness : ℝ) (stack_height : ℝ) :
  bundle_sheets = 300 →
  bundle_thickness = 4 →
  stack_height = 10 →
  (stack_height / bundle_thickness * bundle_sheets : ℝ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_stack_thickness_l266_26648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_and_increasing_l266_26657

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 2) - 1

theorem f_zero_and_increasing :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, f x = 0) ∧
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_and_increasing_l266_26657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l266_26669

theorem band_arrangement_possibilities (total_members min_per_row max_per_row : ℕ) : 
  total_members = 120 → min_per_row = 4 → max_per_row = 30 →
  (Finset.filter (λ x ↦ min_per_row ≤ x ∧ x ≤ max_per_row) 
    (Nat.divisors total_members)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l266_26669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commuting_distress_analysis_l266_26635

/-- Contingency table data -/
structure ContingencyTable where
  young_high : Nat
  young_low : Nat
  middle_high : Nat
  middle_low : Nat

/-- Chi-square statistic calculation -/
noncomputable def chiSquare (ct : ContingencyTable) : Real :=
  let n := ct.young_high + ct.young_low + ct.middle_high + ct.middle_low
  let a := ct.young_high
  let b := ct.young_low
  let c := ct.middle_high
  let d := ct.middle_low
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d) : Real)

/-- Definition of S -/
noncomputable def S (ct : ContingencyTable) : Real :=
  (ct.young_high / (ct.young_high + ct.middle_high : Real)) * (ct.middle_low / (ct.young_low + ct.middle_low : Real)) /
  ((ct.middle_high / (ct.young_high + ct.middle_high : Real)) * (ct.young_low / (ct.young_low + ct.middle_low : Real)))

/-- Definition of T -/
noncomputable def T (ct : ContingencyTable) : Real :=
  (ct.young_high / (ct.young_high + ct.young_low : Real)) * (ct.middle_low / (ct.middle_high + ct.middle_low : Real)) /
  ((ct.young_low / (ct.young_high + ct.young_low : Real)) * (ct.middle_high / (ct.middle_high + ct.middle_low : Real)))

/-- The main theorem to be proved -/
theorem commuting_distress_analysis (ct : ContingencyTable) 
  (h1 : ct.young_high = 50)
  (h2 : ct.young_low = 60)
  (h3 : ct.middle_high = 30)
  (h4 : ct.middle_low = 60) :
  chiSquare ct > 2.706 ∧ S ct = 5/3 ∧ T ct = S ct := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commuting_distress_analysis_l266_26635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l266_26632

theorem divisibility_condition (n p : ℕ+) (h_prime : Nat.Prime p.val) (h_bound : n ≤ 2 * p) :
  (((p.val : ℤ) - 1) ^ (n.val : ℕ) + 1) % ((n : ℤ) ^ (p.val - 1)) = 0 ↔
    ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l266_26632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_worked_56_hours_l266_26607

/-- The total number of hours worked by Thomas, Toby, and Rebecca -/
def total_hours : ℕ := 157

/-- Thomas's work hours -/
def thomas_hours : ℕ → ℕ := id

/-- Toby's work hours -/
def toby_hours (x : ℕ) : ℕ := 2 * x - 10

/-- Rebecca's work hours -/
def rebecca_hours (x : ℕ) : ℕ := toby_hours x - 8

/-- The theorem stating that Rebecca worked 56 hours -/
theorem rebecca_worked_56_hours :
  ∃ x : ℕ, thomas_hours x + toby_hours x + rebecca_hours x = total_hours ∧ rebecca_hours x = 56 := by
  -- Proof goes here
  sorry

#check rebecca_worked_56_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_worked_56_hours_l266_26607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tileC_matches_rectangleIII_l266_26690

structure Tile where
  name : Char
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

def tiles : List Tile := [
  { name := 'A', top := 5, right := 3, bottom := 7, left := 2 },
  { name := 'B', top := 3, right := 6, bottom := 2, left := 8 },
  { name := 'C', top := 7, right := 9, bottom := 1, left := 3 },
  { name := 'D', top := 1, right := 8, bottom := 5, left := 9 }
]

def hasUniqueNumbers (t : Tile) : Bool :=
  let allNumbers := tiles.bind (λ tile => [tile.top, tile.right, tile.bottom, tile.left])
  (¬ allNumbers.contains t.right ∨ ¬ allNumbers.contains t.bottom) ∧
  (t.right ≠ t.bottom)

theorem tileC_matches_rectangleIII :
  ∃ t ∈ tiles, t.name = 'C' ∧ hasUniqueNumbers t ∧
  ∀ t' ∈ tiles, t' ≠ t → ¬ hasUniqueNumbers t' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tileC_matches_rectangleIII_l266_26690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_weight_in_pounds_l266_26640

/-- Conversion factor from kilograms to pounds -/
noncomputable def kg_to_pound : ℝ := 1 / 0.454

/-- Weight of the car in kilograms -/
def car_weight_kg : ℝ := 1250

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem stating the car weight in pounds -/
theorem car_weight_in_pounds :
  round_to_nearest (car_weight_kg * kg_to_pound) = 2753 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_weight_in_pounds_l266_26640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_size_increase_l266_26650

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt 1.44
  (π * R^2) / (π * r^2) = 1.44 →
  (R - r) / r = 0.2 :=
by
  intro h1
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_size_increase_l266_26650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair3_equal_l266_26693

-- Define the four pairs of expressions
noncomputable def pair1 : ℝ × ℝ := (-9, -1/9)
noncomputable def pair2 : ℝ × ℝ := (-|(-9)|, -(-9))
noncomputable def pair3 : ℝ × ℝ := (9, |(-9)|)
noncomputable def pair4 : ℝ × ℝ := (-9, |(-9)|)

-- Theorem stating that only pair3 contains equal values
theorem only_pair3_equal :
  pair1.1 ≠ pair1.2 ∧
  pair2.1 ≠ pair2.2 ∧
  pair3.1 = pair3.2 ∧
  pair4.1 ≠ pair4.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair3_equal_l266_26693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_max_k_value_product_inequality_l266_26681

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

-- Statement 1: f is monotone decreasing on (0, +∞)
theorem f_monotone_decreasing : 
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x :=
sorry

-- Statement 2: Maximum value of k is 3
theorem max_k_value : 
  ∃ k : ℕ, k = 3 ∧ 
  (∀ x : ℝ, x > 0 → f x > (k : ℝ) / (x + 1)) ∧
  (∀ m : ℕ, m > k → ∃ x : ℝ, x > 0 ∧ f x ≤ (m : ℝ) / (x + 1)) :=
sorry

-- Statement 3: Inequality for product of terms
theorem product_inequality (n : ℕ) : 
  Finset.prod (Finset.range n) (λ i => (1 + i * (i + 1))) > Real.exp (2 * n - 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_max_k_value_product_inequality_l266_26681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_journey_time_l266_26682

/-- Represents the time taken for a journey with walking and running portions -/
noncomputable def journey_time (walk_speed : ℝ) (walk_distance : ℝ) (run_distance : ℝ) : ℝ :=
  walk_distance / walk_speed + run_distance / (2 * walk_speed)

theorem wednesday_journey_time 
  (walk_speed : ℝ) 
  (total_distance : ℝ) 
  (hws : walk_speed > 0) 
  (htd : total_distance > 0) : 
  journey_time walk_speed (2 * total_distance / 3) (total_distance / 3) = 30 →
  journey_time walk_speed (total_distance / 3) (2 * total_distance / 3) = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_journey_time_l266_26682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishery_model_properties_l266_26612

/-- Fishery breeding model -/
structure FisheryModel where
  m : ℝ  -- Maximum breeding capacity
  k : ℝ  -- Proportionality constant
  h_m_pos : 0 < m
  h_k_pos : 0 < k

variable (model : FisheryModel)

/-- Annual increase of fish population -/
noncomputable def annual_increase (x : ℝ) : ℝ :=
  model.k * x * (1 - x / model.m)

/-- The domain of the annual increase function -/
def valid_breeding_amount (x : ℝ) : Prop :=
  0 < x ∧ x < model.m

/-- Maximum annual growth of fish population -/
noncomputable def max_annual_growth (model : FisheryModel) : ℝ :=
  (model.m * model.k) / 4

/-- Theorem stating the properties of the fishery model -/
theorem fishery_model_properties (model : FisheryModel) :
  (∀ x, valid_breeding_amount model x → 
    annual_increase model x = model.k * x * (1 - x / model.m)) ∧
  (∃ x, valid_breeding_amount model x ∧ 
    annual_increase model x = max_annual_growth model) ∧
  (∀ x, valid_breeding_amount model x → 
    annual_increase model x ≤ max_annual_growth model) ∧
  (0 < model.k ∧ model.k < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishery_model_properties_l266_26612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_price_is_five_l266_26615

/-- The price of an ice cream cone that satisfies the given conditions -/
noncomputable def ice_cream_price (expenses_ratio : ℚ) (cones_sold : ℕ) (profit : ℚ) : ℚ :=
  profit / (cones_sold * (1 - expenses_ratio))

/-- Theorem stating that the ice cream price is $5 under the given conditions -/
theorem ice_cream_price_is_five :
  ice_cream_price (4/5) 200 200 = 5 := by
  -- Unfold the definition of ice_cream_price
  unfold ice_cream_price
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_price_is_five_l266_26615
