import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1198_119802

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_gt_b : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The ratio of |BF| to |AB| for an ellipse -/
noncomputable def bf_ab_ratio (e : Ellipse) : ℝ := e.a / Real.sqrt (e.a^2 + e.b^2)

/-- Theorem: If the ratio of |BF| to |AB| is √3/2, then the eccentricity is √6/3 
    and the standard equation is x^2/6 + y^2/2 = 1 -/
theorem ellipse_properties (e : Ellipse) 
  (h_ratio : bf_ab_ratio e = Real.sqrt 3 / 2) :
  eccentricity e = Real.sqrt 6 / 3 ∧ 
  e.a^2 = 6 ∧ e.b^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1198_119802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1198_119806

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.sqrt (3 - x)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1198_119806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_vector_properties_l1198_119830

/-- Given two unit vectors with an angle θ ≠ π/2, we define an oblique coordinate system -/
structure ObliqueCoordSystem where
  e₁ : ℝ × ℝ
  e₂ : ℝ × ℝ
  θ : ℝ
  e₁_unit : e₁.1^2 + e₁.2^2 = 1
  e₂_unit : e₂.1^2 + e₂.2^2 = 1
  θ_ne_pi_div_2 : θ ≠ π/2

/-- Vector in oblique coordinates -/
def ObliqueVector (sys : ObliqueCoordSystem) := ℝ × ℝ

/-- Convert oblique coordinates to Cartesian coordinates -/
def toCartesian (sys : ObliqueCoordSystem) (v : ObliqueVector sys) : ℝ × ℝ :=
  (v.1 * sys.e₁.1 + v.2 * sys.e₂.1, v.1 * sys.e₁.2 + v.2 * sys.e₂.2)

/-- Vector subtraction in oblique coordinates -/
def sub (sys : ObliqueCoordSystem) (a b : ObliqueVector sys) : ObliqueVector sys :=
  (a.1 - b.1, a.2 - b.2)

/-- Scalar multiplication in oblique coordinates -/
def smul (sys : ObliqueCoordSystem) (l : ℝ) (a : ObliqueVector sys) : ObliqueVector sys :=
  (l * a.1, l * a.2)

theorem oblique_vector_properties (sys : ObliqueCoordSystem) 
    (a b : ObliqueVector sys) (l : ℝ) :
  (sub sys a b = (a.1 - b.1, a.2 - b.2)) ∧
  (smul sys l a = (l * a.1, l * a.2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_vector_properties_l1198_119830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1198_119871

/-- Diameter of a circle -/
def Diameter (P : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ P ∧ B ∈ P ∧ ∀ X ∈ P, dist X A + dist X B = dist A B

/-- Center of a circle -/
noncomputable def Center (P : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Given a circle with diameter endpoints (2, -3) and (10, 7), its center is (6, 2) -/
theorem circle_center (P : Set (ℝ × ℝ)) : 
  (∃ (A B : ℝ × ℝ), A = (2, -3) ∧ B = (10, 7) ∧ Diameter P A B) → 
  Center P = (6, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1198_119871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_max_l1198_119866

theorem triangle_sum_max (a b c d e f : ℕ) : 
  a ∈ Set.Icc 10 15 →
  b ∈ Set.Icc 10 15 →
  c ∈ Set.Icc 10 15 →
  d ∈ Set.Icc 10 15 →
  e ∈ Set.Icc 10 15 →
  f ∈ Set.Icc 10 15 →
  a + 1 = b →
  b + 1 = c →
  c + 1 = d →
  d + 1 = e →
  e + 1 = f →
  ∃ (S : ℕ), a + b + c = S ∧ c + d + e = S ∧ e + f + a = S →
  ∀ (T : ℕ), (∃ (x y z : ℕ), x + y + z = T ∧ z + d + e = T ∧ e + f + x = T) → T ≤ 39 :=
by
  sorry

#check triangle_sum_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_max_l1198_119866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_with_a_l1198_119851

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 0, -4)
noncomputable def unit_vector : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

theorem unit_vector_collinear_with_a :
  (∃ k : ℝ, k ≠ 0 ∧ vector_a = (k * unit_vector.1, k * unit_vector.2.1, k * unit_vector.2.2)) ∧
  (unit_vector.1^2 + unit_vector.2.1^2 + unit_vector.2.2^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_with_a_l1198_119851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l1198_119810

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the curve
def my_curve (x y t : ℝ) : Prop := y = 3 * abs (x - t)

-- Define the distance ratio condition
def distance_ratio (x y m n s p : ℝ) (k : ℝ) : Prop :=
  (x - m)^2 + (y - n)^2 = k^2 * ((x - s)^2 + (y - p)^2)

theorem find_t (m n s p : ℕ) (k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, my_circle x y → ∃ k > 1, distance_ratio x y m n s p k) →
  my_curve m n t →
  my_curve s p t →
  t = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l1198_119810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l1198_119823

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    (abs x + abs y ≤ 2) ∧
                    (abs x + abs z ≤ 1) ∧
                    (abs y + abs z ≤ 1)}

-- State the theorem
theorem volume_of_T : MeasureTheory.volume T = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l1198_119823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1198_119822

/-- Given an equilateral triangle ABC with side length 5 and a point P on the
    extension of BA such that |AP| = 9, this function represents |EA| - |DF|
    where E and F are the intersections of AD with the circumcircle of BPC,
    and D is a point on BC with |BD| = x. -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 45) / Real.sqrt (x^2 - 5*x + 25)

/-- The minimum value of f(x) is 4√5. -/
theorem min_value_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 5 → f y ≥ f x ∧ f x = 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1198_119822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_bound_l1198_119876

/-- The radius of a graph. -/
def radius {V : Type*} [Fintype V] (G : SimpleGraph V) : ℕ := sorry

/-- The maximum degree of a graph. -/
def maxDegree {V : Type*} [Fintype V] (G : SimpleGraph V) : ℕ := sorry

/-- The number of vertices in a graph. -/
def numVertices {V : Type*} [Fintype V] (G : SimpleGraph V) : ℕ := Fintype.card V

theorem vertex_bound {V : Type*} [Fintype V] (G : SimpleGraph V) (k d : ℕ) 
  (h_radius : radius G ≤ k) 
  (h_maxDegree : maxDegree G ≤ d) 
  (h_d : d ≥ 3) :
  (numVertices G : ℝ) < (d : ℝ) / (d - 2) * (d - 1) ^ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_bound_l1198_119876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1198_119847

noncomputable def f (x : ℝ) := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + π/2) = f x) ∧
  (∀ p, 0 < p → p < π/2 → ∃ x, f (x + p) ≠ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1198_119847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stick_value_with_100_l1198_119874

/-- A set of coins is a stick if all subset sums form a consecutive range of integers. -/
def is_stick (coins : Finset ℕ) : Prop :=
  let sums := (Finset.powerset coins).image (fun s ↦ s.sum id)
  ∀ m, m ∈ Finset.Icc 1 (coins.sum id) → m ∈ sums

/-- The minimum total value of a stick containing a coin of value 100. -/
theorem min_stick_value_with_100 :
    ∀ (n : ℕ) (coins : Finset ℕ),
      n ≥ 2 →
      coins.card = n →
      (∀ x y, x ∈ coins → y ∈ coins → x ≠ y → x ≠ y) →
      100 ∈ coins →
      is_stick coins →
      coins.sum id ≥ 199 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stick_value_with_100_l1198_119874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l1198_119815

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  C = (λ x y ↦ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  F₁ = (- Real.sqrt (a^2 - b^2), 0) ∧
  F₂ = (Real.sqrt (a^2 - b^2), 0)

-- Define points symmetric about the origin
def symmetric_points (P Q : ℝ × ℝ) : Prop :=
  Q = (-P.1, -P.2)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the area of a quadrilateral given its vertices
noncomputable def quadrilateral_area (A B C D : ℝ × ℝ) : ℝ :=
  let s₁ := distance A B
  let s₂ := distance B C
  let s₃ := distance C D
  let s₄ := distance D A
  let p := (s₁ + s₂ + s₃ + s₄) / 2
  Real.sqrt ((p - s₁) * (p - s₂) * (p - s₃) * (p - s₄))

-- State the theorem
theorem ellipse_quadrilateral_area
  (F₁ F₂ P Q : ℝ × ℝ) :
  foci F₁ F₂ ellipse →
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  symmetric_points P Q →
  distance P Q = distance F₁ F₂ →
  quadrilateral_area P F₁ Q F₂ = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l1198_119815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_age_l1198_119887

-- Define Anna's and Kati's current ages
def Anna : ℕ := sorry
def Kati : ℕ := sorry

-- Kati is in high school (age range 14-19)
axiom kati_high_school : 14 ≤ Kati ∧ Kati ≤ 19

-- Anna is older than Kati
axiom anna_older : Anna > Kati

-- Three years from now, Anna will be four times as old as Kati was when Anna was two years older than Kati is now
axiom age_relation : Anna + 3 = 4 * ((Kati + 2) - (Anna - Kati))

-- Theorem to prove
theorem anna_age : Anna = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_age_l1198_119887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_section_probability_l1198_119814

/-- The length of the pipe in centimeters -/
noncomputable def pipe_length : ℝ := 100

/-- The minimum required length for a usable pipe section in centimeters -/
noncomputable def min_section_length : ℝ := 25

/-- The probability that all three sections of the pipe are usable -/
noncomputable def probability_all_sections_usable : ℝ := 9/16

/-- Theorem stating that the probability of all three sections being usable is 9/16 -/
theorem pipe_section_probability :
  let total_area : ℝ := (pipe_length^2) / 2
  let usable_area : ℝ := ((pipe_length - min_section_length)^2) / 2
  usable_area / total_area = probability_all_sections_usable := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_section_probability_l1198_119814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_park_l1198_119805

/-- Represents the cycling speed in miles per minute -/
noncomputable def cycling_speed (distance_to_market : ℝ) (time_to_market : ℝ) : ℝ :=
  distance_to_market / time_to_market

/-- Calculates the time to cycle a given distance at a constant speed -/
noncomputable def time_to_cycle (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem time_to_park (distance_to_market : ℝ) (time_to_market : ℝ) (distance_to_park : ℝ)
    (h1 : distance_to_market = 5)
    (h2 : time_to_market = 30)
    (h3 : distance_to_park = 3) :
    time_to_cycle distance_to_park (cycling_speed distance_to_market time_to_market) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_park_l1198_119805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l1198_119881

/-- The volume of a cone with given slant height and height -/
noncomputable def cone_volume (slant_height : ℝ) (height : ℝ) : ℝ :=
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1/3) * Real.pi * radius^2 * height

/-- Theorem stating that the volume of a cone with slant height 15 and height 9 is 432π -/
theorem cone_volume_specific : cone_volume 15 9 = 432 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l1198_119881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_plus_sin_is_dependent_exp2_is_dependent_l1198_119853

-- Define the concept of a dependent function
def is_dependent_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ f x₁ * f x₂ = 1

-- Define the domain for the sine function
def sine_domain : Set ℝ := { x | -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2 }

-- State the theorem for √2 + sin x
theorem sqrt2_plus_sin_is_dependent :
  is_dependent_function (fun x ↦ Real.sqrt 2 + Real.sin x) sine_domain :=
sorry

-- State the theorem for 2ˣ
theorem exp2_is_dependent :
  is_dependent_function (fun x ↦ 2^x) Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_plus_sin_is_dependent_exp2_is_dependent_l1198_119853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_numbers_l1198_119827

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := S.filter (λ n => n % 2 = 0)

theorem probability_two_even_numbers :
  (Finset.card even_numbers).choose 2 / (Finset.card S).choose 2 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_numbers_l1198_119827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_l1198_119829

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 1 = 0

/-- The area of the circle defined by the given equation -/
noncomputable def CircleArea : ℝ := 4 * Real.pi

/-- Theorem stating that the area of the circle is 4π -/
theorem circle_area_is_4pi :
  ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), CircleEquation x y ↔ (x - 1)^2 + (y + 2)^2 = r^2) ∧
  CircleArea = π * r^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_l1198_119829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l1198_119878

open Set

theorem complement_intersection_theorem (U A B : Set ℝ) 
  (hU : U = univ)
  (hA : A = {x : ℝ | x ≤ -2})
  (hB : B = {x : ℝ | x < 1}) : 
  (Aᶜ) ∩ B = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l1198_119878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_l1198_119885

/-- Given two perpendicular lines passing through fixed points A and B,
    prove that the distance from their intersection point P to the midpoint C of AB
    is equal to √10/2. -/
theorem distance_to_midpoint (m : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  let C : ℝ × ℝ := ((A.fst + B.fst)/2, (A.snd + B.snd)/2)
  let line1 := {p : ℝ × ℝ | p.fst + m * p.snd = 0}
  let line2 := {p : ℝ × ℝ | m * p.fst - p.snd - m + 3 = 0}
  let P := Set.inter line1 line2
  ∀ p ∈ P, ‖p - C‖ = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_l1198_119885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seats_in_hall_l1198_119863

theorem total_seats_in_hall (filled_percentage : ℝ) (vacant_seats : ℕ) (total_seats : ℕ) : 
  filled_percentage = 62 ∧ vacant_seats = 228 ∧ total_seats = 600 → 
  (1 - filled_percentage / 100) * total_seats = vacant_seats := by
  sorry

#check total_seats_in_hall

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seats_in_hall_l1198_119863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_theta_l1198_119838

theorem sin_pi_half_minus_theta (θ : ℝ) (h : Real.cos θ = (1/3) * Real.tan (-π/4)) :
  Real.sin (π/2 - θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_theta_l1198_119838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_overall_correctness_l1198_119809

/-- Represents the correctness percentage for a part of the homework -/
def Correctness := Fin 101

/-- Represents the number of problems in the homework -/
def TotalProblems := Nat

/-- Calculates the number of correct answers given a total and a percentage -/
def correctAnswers (total : Nat) (percentage : Correctness) : Nat :=
  (total * percentage.val) / 100

theorem zoe_overall_correctness 
  (total : Nat)
  (chloe_first : Correctness)
  (chloe_last : Correctness)
  (chloe_overall : Correctness)
  (zoe_first : Correctness)
  (zoe_last : Correctness)
  (h_total : total % 3 = 0)
  (h_chloe_first : chloe_first.val = 70)
  (h_chloe_last : chloe_last.val = 90)
  (h_chloe_overall : chloe_overall.val = 82)
  (h_zoe_first : zoe_first.val = 85)
  (h_zoe_last : zoe_last.val = 95)
  : correctAnswers total ⟨88, by norm_num⟩ = 
    correctAnswers (total / 3) zoe_first + 
    correctAnswers (total / 3) chloe_overall + 
    correctAnswers (total / 3) zoe_last :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_overall_correctness_l1198_119809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_existence_l1198_119813

/-- Given non-constant polynomials P, Q, R, S with real coefficients, such that P(Q(x)) = R(S(x))
    and the degree of P is a multiple of the degree of R, prove that there exists a polynomial T
    with real coefficients such that P(x) = R(T(x)). -/
theorem polynomial_composition_existence
  (P Q R S : Polynomial ℝ)
  (hP : P.natDegree ≠ 0)
  (hQ : Q.natDegree ≠ 0)
  (hR : R.natDegree ≠ 0)
  (hS : S.natDegree ≠ 0)
  (h_comp : P.comp Q = R.comp S)
  (h_deg : ∃ k : ℕ, P.natDegree = k * R.natDegree) :
  ∃ T : Polynomial ℝ, P = R.comp T :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_existence_l1198_119813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coin_l1198_119875

/-- Represents a coin with a unique identifier and weight status -/
structure Coin where
  id : Nat
  isCounterfeit : Bool
  isHeavier : Bool

/-- Represents a weighing result -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing operation -/
def weighing (leftGroup rightGroup : List Coin) : WeighingResult :=
  sorry

/-- The main theorem stating that the counterfeit coin can be identified in n weighings -/
theorem identify_counterfeit_coin (n : Nat) (coins : List Coin)
    (h1 : n ≥ 2)
    (h2 : coins.length = (3^n - 3) / 2)
    (h3 : ∃! c : Coin, c ∈ coins ∧ c.isCounterfeit) :
    ∃ (strategy : Nat → List Coin → (List Coin × List Coin)),
      (∀ i : Nat, i < n → 
        let (left, right) := strategy i coins
        weighing left right ≠ WeighingResult.Equal) →
      ∃ (c : Coin), c ∈ coins ∧ c.isCounterfeit ∧
        (c.isHeavier ∨ ¬c.isHeavier) := by
  sorry

#check identify_counterfeit_coin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coin_l1198_119875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_national_marriage_council_theorem_l1198_119843

/-- Represents a valid group configuration for n couples -/
structure GroupConfiguration (n : ℕ) where
  male_groups : ℕ
  female_groups : ℕ
  min_group_size : ℕ
  max_group_size : ℕ
  total_groups : male_groups + female_groups = 17
  group_size_diff : max_group_size - min_group_size ≤ 1
  min_group_size_positive : min_group_size > 0
  total_members : n = male_groups * min_group_size + female_groups * max_group_size

/-- The set of valid n values -/
def valid_n_set : Set ℕ := {9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24}

/-- The main theorem stating the equivalence -/
theorem national_marriage_council_theorem (n : ℕ) (h : n ≤ 1996) :
  (∃ (config : GroupConfiguration n), True) ↔ n ∈ valid_n_set := by
  sorry

#check national_marriage_council_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_national_marriage_council_theorem_l1198_119843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sabroso_numbers_l1198_119897

-- Define a two-digit number
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the reverse of a two-digit number
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

-- Define a sabroso number
def isSabroso (n : ℕ) : Prop :=
  TwoDigitNumber n ∧ ∃ k : ℕ, n + reverse n = k * k

-- The set of two-digit sabroso numbers
def sabrosoSet : Set ℕ := {29, 38, 47, 56, 65, 74, 83, 92}

-- The theorem to prove
theorem two_digit_sabroso_numbers :
  ∀ n : ℕ, TwoDigitNumber n → (isSabroso n ↔ n ∈ sabrosoSet) :=
by sorry

-- Additional lemmas that might be useful for the proof
lemma reverse_of_two_digit (n : ℕ) (h : TwoDigitNumber n) :
  TwoDigitNumber (reverse n) :=
by sorry

lemma sum_with_reverse (n : ℕ) (h : TwoDigitNumber n) :
  n + reverse n = 11 * (n / 10 + n % 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sabroso_numbers_l1198_119897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l1198_119880

/-- The ellipse C1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2/5 = 1

/-- The hyperbola C2 -/
def C2 (x y : ℝ) : Prop := x^2/4 - y^2 = 1

/-- The length of the major axis of C1 -/
noncomputable def majorAxisLength : ℝ := 2 * Real.sqrt 5

/-- The asymptotic line equations of C2 -/
def asymptoticLines (x y : ℝ) : Prop := y = (1/2) * x ∨ y = -(1/2) * x

/-- The eccentricity of C1 -/
noncomputable def eccentricityC1 : ℝ := Real.sqrt (4/5)

/-- The eccentricity of C2 -/
noncomputable def eccentricityC2 : ℝ := Real.sqrt (5/4)

/-- The foci of C1 -/
def fociC1 : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- The foci of C2 -/
noncomputable def fociC2 : Set (ℝ × ℝ) := {(Real.sqrt 5, 0), (-Real.sqrt 5, 0)}

theorem ellipse_hyperbola_properties :
  (∀ x y, C1 x y → C2 x y →
    (majorAxisLength = 2 * Real.sqrt 5) ∧
    (∀ x y, asymptoticLines x y) ∧
    (eccentricityC1 * eccentricityC2 = 1) ∧
    (fociC1 ≠ fociC2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l1198_119880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l1198_119888

open Real MeasureTheory

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 - Real.log (Real.sin x)

-- Define the interval
noncomputable def a : ℝ := π / 3
noncomputable def b : ℝ := π / 2

-- State the theorem
theorem arc_length_of_curve :
  ∫ x in a..b, sqrt (1 + (deriv f x) ^ 2) = (Real.log 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l1198_119888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_temperature_proof_l1198_119877

/-- Represents the temperature change per unit of elevation change -/
noncomputable def temp_change_rate : ℚ := -3 / (1/2)

/-- Calculates the temperature at a given elevation -/
noncomputable def temperature_at_elevation (initial_temp : ℚ) (elevation : ℚ) : ℚ :=
  initial_temp + temp_change_rate * elevation

/-- Calculates the elevation given a temperature -/
noncomputable def elevation_at_temperature (initial_temp : ℚ) (temp : ℚ) : ℚ :=
  (temp - initial_temp) / temp_change_rate

theorem mountain_temperature_proof 
  (initial_temp : ℚ) 
  (elevation_first_campsite : ℚ) 
  (temp_second_campsite : ℚ) 
  (h1 : initial_temp = -2) 
  (h2 : elevation_first_campsite = 5/2) 
  (h3 : temp_second_campsite = -29) :
  (temperature_at_elevation initial_temp elevation_first_campsite = -17) ∧
  (elevation_at_temperature initial_temp temp_second_campsite = 9/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_temperature_proof_l1198_119877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_l1198_119836

theorem sin_to_cos_shift (x : ℝ) : 
  Real.sin (2 * (x + Real.pi/6) + Real.pi/6) = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_l1198_119836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proof_l1198_119832

theorem trigonometric_proof (x : ℝ) (h : Real.sin x - 2 * Real.cos x = 0) :
  Real.tan x = 2 ∧ (2 * Real.sin x * Real.cos x) / (Real.sin x ^ 2 - Real.cos x ^ 2) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_proof_l1198_119832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cosine_identification_l1198_119828

theorem unique_cosine_identification (x : ℝ) : 
  0 < x → x < π / 2 →  -- x is acute
  Real.sin x = 1 / 2 →  -- given sin x = 1/2
  (∀ y, 0 < y → y < π / 2 → Real.sin y = 1 / 2 → y = x) →  -- uniqueness of x
  (∀ y, 0 < y → y < π / 2 → Real.tan y = Real.tan x → y = x) →  -- uniqueness of tan x
  (∃ z, 0 < z ∧ z < π / 2 ∧ Real.cos z = Real.cos x ∧ z ≠ x) →  -- non-uniqueness of cos x
  Real.cos x = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cosine_identification_l1198_119828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1198_119844

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * Real.pi * base_radius^2 * cone_height
  volume = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1198_119844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_of_4_equals_53_div_7_l1198_119868

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 5 / (3 - x)

-- Define the inverse of h
noncomputable def h_inv (x : ℝ) : ℝ := (3 * x - 5) / x

-- Define the function j
noncomputable def j (x : ℝ) : ℝ := 1 / h_inv x + 7

-- Theorem statement
theorem j_of_4_equals_53_div_7 : j 4 = 53 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_of_4_equals_53_div_7_l1198_119868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1198_119845

open Real

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*log x
def g (a : ℝ) (x : ℝ) : ℝ := 2*a*x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Theorem statement
theorem function_properties :
  (∀ a : ℝ, a ≤ 0 → ¬∃ x : ℝ, x > 0 ∧ IsLocalExtr (f a) x) ∧
  (∀ a : ℝ, a > 0 → 
    (∃ x : ℝ, x > 0 ∧ IsLocalMin (f a) x ∧ f a x = a - a * log a) ∧
    (¬∃ x : ℝ, x > 0 ∧ IsLocalMax (f a) x)) ∧
  (∀ a : ℝ, a > 0 → 
    (∃! x : ℝ, x > 0 ∧ h a x = 0) → a = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1198_119845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l1198_119807

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def start : point := (2, -3)
def intermediate : point := (8, 9)
def end_point : point := (3, 2)

theorem total_distance_proof :
  distance start intermediate + distance intermediate end_point =
  6 * Real.sqrt 5 + Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l1198_119807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_2n_l1198_119870

theorem subset_sum_divisible_by_2n (n : ℕ) (a : Fin n → ℕ) 
  (h_n : n ≥ 4)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_range : ∀ i : Fin n, 0 < a i ∧ a i < 2*n) :
  ∃ s : Finset (Fin n), (s.sum a) % (2*n) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_2n_l1198_119870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercept_sum_l1198_119883

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the y-intercept of a quadratic function -/
def QuadraticFunction.y_intercept (f : QuadraticFunction) : ℝ :=
  f.c

/-- Calculates the x-intercept of a quadratic function -/
noncomputable def QuadraticFunction.x_intercept (f : QuadraticFunction) : ℝ :=
  (-f.b - Real.sqrt (f.b^2 - 4*f.a*f.c)) / (2*f.a)

/-- The main theorem: sum of y-intercepts and x-intercept for the given parabola -/
theorem parabola_intercept_sum :
  let f : QuadraticFunction := ⟨3, -9, 5⟩
  2 * f.y_intercept + f.x_intercept = (69 - Real.sqrt 21) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercept_sum_l1198_119883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1198_119826

-- Define the constants as noncomputable
noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (2/5)^(1/5)
noncomputable def c : ℝ := (2/5)^(3/5)

-- State the theorem
theorem ordering_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1198_119826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindromes_count_l1198_119818

/-- A five-digit palindrome is a number of the form ABCBA where A, B, and C are single digits. -/
def FiveDigitPalindrome : Type := ℕ

/-- The first digit of a five-digit palindrome, which must be non-zero. -/
def FirstDigit : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 9) (Finset.range 10)

/-- The set of all single digits. -/
def Digit : Finset ℕ := Finset.range 10

/-- The number of five-digit palindromes. -/
def NumFiveDigitPalindromes : ℕ := FirstDigit.card * Digit.card * Digit.card

theorem five_digit_palindromes_count :
  NumFiveDigitPalindromes = 900 := by
  unfold NumFiveDigitPalindromes
  unfold FirstDigit
  unfold Digit
  simp
  norm_num
  rfl

#eval NumFiveDigitPalindromes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindromes_count_l1198_119818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1198_119811

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem_solution (a : ℝ) (α β : ℝ) : 
  f a (π/4) = 0 ∧ 
  0 < α ∧ α < π/2 ∧ 
  0 < β ∧ β < π/2 ∧ 
  f a α = Real.sqrt 10 / 5 ∧ 
  f a β = 3 * Real.sqrt 5 / 5 →
  a = -1 ∧ Real.sin (α + β) = -Real.sqrt 2 / 10 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1198_119811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l1198_119891

noncomputable def expr_A : ℝ := 20 - 4 * Real.sqrt 15
noncomputable def expr_B : ℝ := 4 * Real.sqrt 15 - 20
noncomputable def expr_C : ℝ := 25 - 6 * Real.sqrt 17
noncomputable def expr_D : ℝ := 60 - 12 * Real.sqrt 30
noncomputable def expr_E : ℝ := 12 * Real.sqrt 30 - 60

theorem smallest_positive_number : 
  expr_C > 0 ∧ 
  (expr_A > 0 → expr_C < expr_A) ∧
  (expr_B > 0 → expr_C < expr_B) ∧
  (expr_D > 0 → expr_C < expr_D) ∧
  (expr_E > 0 → expr_C < expr_E) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l1198_119891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proofs_l1198_119896

theorem inequality_proofs (m n : ℝ) (h : m > n) (h' : n > 0) :
  (Real.sqrt (m^2 + m) > Real.sqrt (n^2 + n)) ∧
  (m - n > Real.sin m - Real.sin n) ∧
  ((Real.exp m - Real.exp n) / (m + n) > m - n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proofs_l1198_119896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_implies_a_nonpositive_h_max_value_is_zero_l1198_119842

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - a * x

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := Real.log x - x + 1

-- Theorem for part (I)
theorem f_derivative_positive_implies_a_nonpositive :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, (deriv (f a)) x > 0) → a ≤ 0 :=
by sorry

-- Theorem for part (II)
theorem h_max_value_is_zero :
  IsGreatest { y | ∃ x > 0, h x = y } 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_implies_a_nonpositive_h_max_value_is_zero_l1198_119842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_gross_profit_percentage_l1198_119856

/-- Calculate the gross profit percentage given the selling price and wholesale cost -/
noncomputable def gross_profit_percentage (selling_price wholesale_cost : ℝ) : ℝ :=
  ((selling_price - wholesale_cost) / wholesale_cost) * 100

/-- The gross profit percentage for a sleeping bag is approximately 14.01% -/
theorem sleeping_bag_gross_profit_percentage :
  let selling_price : ℝ := 28
  let wholesale_cost : ℝ := 24.56
  abs (gross_profit_percentage selling_price wholesale_cost - 14.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_gross_profit_percentage_l1198_119856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distant_visible_point_l1198_119859

def visible (x y : ℤ) : Prop := Int.gcd x y = 1

noncomputable def distance (x1 y1 x2 y2 : ℤ) : ℝ :=
  Real.sqrt (((x1 - x2) ^ 2 + (y1 - y2) ^ 2) : ℝ)

theorem existence_of_distant_visible_point (n : ℕ) :
  ∃ a b : ℤ, visible a b ∧
    ∀ x y : ℤ, visible x y → (x, y) ≠ (a, b) →
      (n : ℝ) < distance a b x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distant_visible_point_l1198_119859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_combinations_l1198_119841

def Digits : Finset ℕ := {0, 1, 2, 3, 5, 9}

theorem digit_combinations (d : Finset ℕ := Digits) :
  (∀ x ∈ d, x < 10) →
  (Finset.card d = 6) →
  (∃ (four_digit : Finset ℕ) (four_digit_odd : Finset ℕ) (four_digit_even : Finset ℕ) (all_numbers : Finset ℕ),
    (∀ n ∈ four_digit, n ≥ 1000 ∧ n < 10000) ∧
    (∀ n ∈ four_digit_odd, n ∈ four_digit ∧ n % 2 = 1) ∧
    (∀ n ∈ four_digit_even, n ∈ four_digit ∧ n % 2 = 0) ∧
    (∀ n ∈ all_numbers, n > 0) ∧
    (Finset.card four_digit = 300) ∧
    (Finset.card four_digit_odd = 192) ∧
    (Finset.card four_digit_even = 108) ∧
    (Finset.card all_numbers = 1631)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_combinations_l1198_119841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_percentage_approx_l1198_119848

/-- Represents the percentage of a round-trip completed by a technician -/
noncomputable def round_trip_percentage (original_distance : ℝ) : ℝ :=
  let return_distance := 1.2 * original_distance
  let total_distance := original_distance + return_distance
  let completed_distance := original_distance + 0.1 * return_distance
  (completed_distance / total_distance) * 100

/-- Proves that the round-trip percentage is approximately 50.91% -/
theorem round_trip_percentage_approx :
  ∀ d : ℝ, d > 0 → |round_trip_percentage d - 50.91| < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_percentage_approx_l1198_119848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1198_119890

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define the center of the circle
def F : ℝ × ℝ := (1, 0)

-- Define the ellipse
def my_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Theorem statement
theorem trajectory_is_ellipse :
  ∀ (B P : ℝ × ℝ),
    my_circle B.1 B.2 →
    (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1) → -- P is on perpendicular bisector of AB
    (P.1 - B.1) * (F.2 - B.2) = (P.2 - B.2) * (F.1 - B.1) → -- P is on line BF
    my_ellipse P.1 P.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1198_119890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_through_focus_l1198_119803

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y^2 = 4 * a * x

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq x y

/-- Theorem: Length of chord through focus of parabola y^2 = 4x -/
theorem chord_length_through_focus (p : Parabola) 
  (h_p : p.a = 1) 
  (A B : ParabolaPoint p) 
  (h_sum : A.x + B.x = 9) : 
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_through_focus_l1198_119803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_bags_price_l1198_119840

theorem halloween_bags_price (total_students : ℕ) (vampire_requests : ℕ) (pumpkin_requests : ℕ)
  (pack_size : ℕ) (pack_price : ℚ) (total_spent : ℚ) :
  total_students = 25 →
  vampire_requests = 11 →
  pumpkin_requests = 14 →
  pack_size = 5 →
  pack_price = 3 →
  total_spent = 17 →
  vampire_requests + pumpkin_requests = total_students →
  ∃ (individual_price : ℚ),
    individual_price = 1 ∧
    (((vampire_requests / pack_size) * pack_price +
     (pumpkin_requests / pack_size) * pack_price +
     (vampire_requests % pack_size + pumpkin_requests % pack_size) * individual_price) = total_spent) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_bags_price_l1198_119840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stoppage_time_l1198_119834

/-- Represents the speed and stoppage time of a bus -/
structure BusJourney where
  speed_without_stops : ℚ
  speed_with_stops : ℚ
  stoppage_time : ℚ

/-- Calculates the stoppage time per hour given the speeds with and without stops -/
def calculate_stoppage_time (speed_without_stops speed_with_stops : ℚ) : ℚ :=
  (1 - speed_with_stops / speed_without_stops) * 60

theorem bus_stoppage_time (bus : BusJourney) 
  (h1 : bus.speed_without_stops = 60)
  (h2 : bus.speed_with_stops = 20)
  : bus.stoppage_time = calculate_stoppage_time bus.speed_without_stops bus.speed_with_stops := by
  sorry

#eval calculate_stoppage_time 60 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stoppage_time_l1198_119834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_values_l1198_119889

noncomputable def floor_sum (x : ℝ) (p : ℤ) : ℤ :=
  ⌊(x - p : ℝ) / p⌋ + ⌊(-x - 1 : ℝ) / p⌋

theorem floor_sum_values (x : ℝ) (p : ℤ) (hp : p ≠ 0) :
  ∃ (n : ℤ), floor_sum x p = n ∧ n ∈ ({-3, -2, -1, 0} : Set ℤ) := by
  sorry

#check floor_sum_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_values_l1198_119889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1198_119857

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The parabola given by y = -3x^2 + 6x - 9 -/
def parabola : QuadraticFunction :=
  { a := -3
    b := 6
    c := -9 }

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ :=
  f.a * (vertex_x f)^2 + f.b * (vertex_x f) + f.c

/-- Determines if the vertex is a maximum or minimum point -/
def is_maximum (f : QuadraticFunction) : Prop := f.a < 0

theorem parabola_vertex :
  (vertex_x parabola = 1) ∧
  (vertex_y parabola = -6) ∧
  is_maximum parabola := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1198_119857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_acute_angles_diameter_perpendicular_bisects_chord_l1198_119873

-- Definition of a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  positive_radius : 0 < radius

-- Definition of a chord
structure Chord (circle : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle : (endpoint1.1 - circle.center.1)^2 + (endpoint1.2 - circle.center.2)^2 = circle.radius^2 ∧
              (endpoint2.1 - circle.center.1)^2 + (endpoint2.2 - circle.center.2)^2 = circle.radius^2

-- Statement A
theorem triangle_two_acute_angles (t : Triangle) : 
  (t.a < 90 ∧ t.b < 90) ∨ (t.a < 90 ∧ t.c < 90) ∨ (t.b < 90 ∧ t.c < 90) := by
  sorry

-- Statement B
theorem diameter_perpendicular_bisects_chord (c : Circle) (ch : Chord c) (d : (ℝ × ℝ) → (ℝ × ℝ) → ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 → d c.center p = c.radius) →
  (d c.center ch.endpoint1 = c.radius ∧ d c.center ch.endpoint2 = c.radius) →
  (ch.endpoint1.1 - ch.endpoint2.1) * (c.center.2 - ch.endpoint1.2) = 
  (ch.endpoint1.2 - ch.endpoint2.2) * (c.center.1 - ch.endpoint1.1) →
  ∃ m : ℝ × ℝ, d m ch.endpoint1 = d m ch.endpoint2 ∧
              (m.1 - c.center.1)^2 + (m.2 - c.center.2)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_acute_angles_diameter_perpendicular_bisects_chord_l1198_119873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_side_length_lower_bound_l1198_119801

/-- A triangle with side lengths a, b, and c, and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_area : S = 1

/-- The average side length of a triangle -/
noncomputable def average_side_length (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 3

/-- Theorem: For any triangle with area 1, the average length of its sides is not less than √2 -/
theorem average_side_length_lower_bound (t : Triangle) : 
  average_side_length t ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_side_length_lower_bound_l1198_119801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_sea_level_l1198_119869

/-- Represents the elevation relative to sea level in meters -/
def elevation : ℤ → ℝ := sorry

/-- Axiom: 100 meters above sea level is denoted as +100 -/
axiom above_sea_level : elevation 100 = 100

/-- Theorem: 75 meters below sea level is denoted as -75 meters -/
theorem below_sea_level : elevation (-75) = -75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_sea_level_l1198_119869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_104_l1198_119854

/-- Represents a rectangle in a plane --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle --/
noncomputable def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles in the plane --/
structure RectangleConfiguration where
  rect1 : Rectangle
  rect2 : Rectangle
  rect3 : Rectangle
  perpendicularIntersection : Bool
  thirdIntersectsFirst : Bool
  thirdIntersectsSecond : Bool

/-- Calculates the total shaded area formed by the intersections of rectangles --/
noncomputable def totalShadedArea (config : RectangleConfiguration) : ℝ :=
  let overlap12 := min config.rect1.width config.rect2.width * min config.rect1.height config.rect2.height
  config.rect1.area + config.rect2.area - overlap12 + config.rect3.area

/-- The main theorem stating the total shaded area for the given configuration --/
theorem total_shaded_area_is_104 (config : RectangleConfiguration) :
  config.rect1 = ⟨4, 15⟩ →
  config.rect2 = ⟨5, 12⟩ →
  config.rect3 = ⟨2, 2⟩ →
  config.perpendicularIntersection = true →
  config.thirdIntersectsFirst = true →
  config.thirdIntersectsSecond = false →
  totalShadedArea config = 104 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_104_l1198_119854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1198_119835

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2 * k + 1) * 4123) :
  Nat.gcd (Int.natAbs (4 * a^2 + 35 * a + 81)) (Int.natAbs (3 * a + 8)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1198_119835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_existence_l1198_119833

theorem sequence_inequality_existence (C : ℕ) (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = n / a n + C) 
  (hC : C > 0) (ha : ∀ n : ℕ, n > 0 → a n > 0) :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a (n + 2) > a n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_existence_l1198_119833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1198_119825

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 5 + 2 * t)

/-- Curve C defined by Cartesian equation -/
def on_curve_C (p : ℝ × ℝ) : Prop :=
  p.2^2 - p.1^2 = 4

/-- Point A -/
noncomputable def point_A : ℝ × ℝ := (0, Real.sqrt 5)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem to be proved -/
theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    t₁ ≠ t₂ ∧
    on_curve_C (line_l t₁) ∧
    on_curve_C (line_l t₂) ∧
    (1 / distance point_A (line_l t₁) + 1 / distance point_A (line_l t₂) = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1198_119825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1198_119898

theorem solve_exponential_equation :
  ∃ x : ℝ, (4 : ℝ) ^ (2 * x + 3) = (1 : ℝ) / 16 ↔ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1198_119898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_nonnegative_min_value_a_l1198_119804

-- Define the function f
def f (x : ℝ) := |2 * x + 1| - |x| - 2

-- Theorem for the solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = Set.Iic (-3) ∪ Set.Ici 1 := by sorry

-- Theorem for the minimum value of a
theorem min_value_a :
  (∃ a : ℝ, (∃ x : ℝ, f x - a ≤ |x|) ∧ 
   ∀ b : ℝ, (∃ x : ℝ, f x - b ≤ |x|) → b ≥ a) ∧
  (let a := -3; ∃ x : ℝ, f x - a ≤ |x|) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_nonnegative_min_value_a_l1198_119804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_one_l1198_119852

-- Define the open interval (0, +∞)
def OpenPositiveReals := {x : ℝ | x > 0}

-- Define the properties of function f
def MonotoneIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_value_at_one
  (f : ℝ → ℝ)
  (h_domain : ∀ x, x > 0 → f x ∈ OpenPositiveReals)
  (h_monotone : MonotoneIncreasing f)
  (h_functional : ∀ x, x > 0 → f (f x + 2 / x) = -1) :
  f 1 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_one_l1198_119852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_devin_basketball_chance_l1198_119837

/-- Calculates the chance of making the junior high basketball team --/
def basketball_team_chance (initial_height : ℕ) (growth : ℕ) (age : ℕ) (ppg : ℕ) : ℚ :=
  let final_height := initial_height + growth
  let base_chance := if age = 12 then 5 / 100 else 10 / 100
  let inch_increase := if age = 12 then 10 / 100 else 15 / 100
  let height_increase := (max (final_height - 66) 0 : ℚ) * inch_increase
  let ppg_bonus := if ppg > 10 then 5 / 100 else 0
  base_chance + height_increase + ppg_bonus

/-- Theorem stating Devin's chance of making the basketball team --/
theorem devin_basketball_chance :
  basketball_team_chance 65 3 13 8 = 40 / 100 := by
  -- Unfold the definition and simplify
  unfold basketball_team_chance
  simp
  -- Perform the calculation
  norm_num

#eval basketball_team_chance 65 3 13 8


end NUMINAMATH_CALUDE_ERRORFEEDBACK_devin_basketball_chance_l1198_119837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_S_and_R_l1198_119893

-- Define the circle and points
variable (circle : Set ℝ)
variable (A B R E C : ℝ)

-- Define the arcs and their measures
variable (arc_BR arc_RE : Set ℝ)
variable (measure_BR measure_RE : ℝ)

-- Define angles S and R
variable (angle_S angle_R : ℝ)

-- Assumptions
variable (h1 : A ∈ circle)
variable (h2 : B ∈ circle)
variable (h3 : R ∈ circle)
variable (h4 : E ∈ circle)
variable (h5 : C ∈ circle)
variable (h6 : measure_BR = 48)
variable (h7 : measure_RE = 54)
variable (h8 : angle_S = (measure_BR + measure_RE) / 2)
variable (h9 : angle_R = (360 - (measure_BR + measure_RE)) / 2)

-- Theorem statement
theorem sum_of_angles_S_and_R :
  angle_S + angle_R = 180 := by
  -- Proof steps would go here, but we'll use sorry as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_S_and_R_l1198_119893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_tree_planting_l1198_119895

/-- The total number of trees planted by the school --/
def total_trees (trees_2nd trees_3rd trees_4th trees_5th trees_6th trees_7th trees_8th : ℕ) : ℕ :=
  trees_2nd + trees_3rd + trees_4th + trees_5th + trees_6th + trees_7th + trees_8th

/-- Theorem stating the total number of trees planted by the school --/
theorem school_tree_planting :
  ∃ (trees_2nd trees_3rd trees_4th trees_5th trees_6th trees_7th trees_8th : ℕ),
    trees_2nd = 15 ∧
    trees_3rd = trees_2nd ^ 2 - 3 ∧
    trees_4th = 2 * trees_3rd + 10 ∧
    trees_5th = (trees_4th ^ 2) / 2 ∧
    trees_6th = 10 * Int.floor (Real.sqrt (trees_5th : ℝ)) - 4 ∧
    trees_7th = 3 * ((trees_3rd + trees_6th) / 2) ∧
    trees_8th = 15 + trees_2nd + trees_3rd + trees_4th - 10 ∧
    total_trees trees_2nd trees_3rd trees_4th trees_5th trees_6th trees_7th trees_8th = 109793 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_tree_planting_l1198_119895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_check_train_length_l1198_119846

-- Define the given conditions
noncomputable def train_speed_kmh : ℝ := 54
noncomputable def time_to_pass_tree : ℝ := 19

-- Define the conversion factor from km/h to m/s
noncomputable def km_per_hour_to_m_per_second : ℝ := 1000 / 3600

-- Define the theorem
theorem train_length : 
  train_speed_kmh * km_per_hour_to_m_per_second * time_to_pass_tree = 285 := by
  -- Proof steps would go here
  sorry

-- Define a function to compute the train length
noncomputable def compute_train_length : ℝ :=
  train_speed_kmh * km_per_hour_to_m_per_second * time_to_pass_tree

-- Assertion to check if the computed length matches the expected result
theorem check_train_length : compute_train_length = 285 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_check_train_length_l1198_119846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mapping_theorem_l1198_119860

-- Define a triangle as a triple of points in 2D space
def Triangle : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define congruence for triangles
def Congruent (t1 t2 : Triangle) : Prop := sorry

-- Define equal orientation for triangles
def EquallyOriented (t1 t2 : Triangle) : Prop := sorry

-- Define a rotation in 2D space
def Rotation : Type := ℝ × ℝ × ℝ

-- Define a translation in 2D space
def Translation : Type := ℝ × ℝ

-- Define the application of a rotation to a triangle
def ApplyRotation (r : Rotation) (t : Triangle) : Triangle := sorry

-- Define the application of a translation to a triangle
def ApplyTranslation (v : Translation) (t : Triangle) : Triangle := sorry

theorem triangle_mapping_theorem (ABC A₁B₁C₁ : Triangle) 
  (h1 : Congruent ABC A₁B₁C₁) (h2 : EquallyOriented ABC A₁B₁C₁) :
  (∃! r : Rotation, ApplyRotation r ABC = A₁B₁C₁) ∨ 
  (∃! v : Translation, ApplyTranslation v ABC = A₁B₁C₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mapping_theorem_l1198_119860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_cut_theorem_l1198_119824

/-- Represents a folded napkin -/
structure FoldedNapkin :=
  (layers : Nat)
  (is_square : Bool)

/-- Represents a cut on the folded napkin -/
inductive Cut
  | Straight : Cut

/-- Represents the result of cutting a folded napkin -/
def cut_result (n : FoldedNapkin) (c : Cut) : Finset Nat :=
  sorry  -- Implementation details omitted for simplicity

/-- A square napkin folded in half twice -/
def twice_folded_napkin : FoldedNapkin :=
  { layers := 4, is_square := true }

theorem napkin_cut_theorem :
  ∀ (c : Cut),
    (2 ∈ cut_result twice_folded_napkin c ∨ 4 ∈ cut_result twice_folded_napkin c) ∧
    (3 ∉ cut_result twice_folded_napkin c ∧ 5 ∉ cut_result twice_folded_napkin c) :=
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_cut_theorem_l1198_119824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_arithmetic_angles_and_geometric_sides_is_equilateral_l1198_119820

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def angles_form_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

def sides_form_geometric_sequence (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_with_arithmetic_angles_and_geometric_sides_is_equilateral
  (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : angles_form_arithmetic_sequence t)
  (h3 : sides_form_geometric_sequence t) :
  is_equilateral t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_arithmetic_angles_and_geometric_sides_is_equilateral_l1198_119820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_multiple_of_six_target_is_valid_and_multiple_of_six_l1198_119839

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    Finset.toSet {a, b, c, d, e} = Finset.toSet {2, 3, 6, 7, 9})

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem greatest_valid_multiple_of_six :
  ∀ (n : ℕ), is_valid_number n ∧ is_multiple_of_six n → n ≤ 97632 :=
by sorry

theorem target_is_valid_and_multiple_of_six :
  is_valid_number 97632 ∧ is_multiple_of_six 97632 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_multiple_of_six_target_is_valid_and_multiple_of_six_l1198_119839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_fixed_point_exists_l1198_119850

-- Define the line equation
noncomputable def line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the area of a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

-- Theorem for the area of the triangle
theorem triangle_area_is_8 :
  ∃ (x y : ℝ), line x y ∧ triangle_area x (-y) = 8 := by
  sorry

-- Define the parametric line equation
noncomputable def parametric_line (m x y : ℝ) : Prop := m * x + y + m = 0

-- Theorem for the fixed point
theorem fixed_point_exists :
  ∀ m : ℝ, parametric_line m (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_fixed_point_exists_l1198_119850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l1198_119894

theorem fraction_equality (x y z : ℝ) 
  (h : (y^2 + z^2 - x^2) / (2*y*z) + (z^2 + x^2 - y^2) / (2*z*x) + (x^2 + y^2 - z^2) / (2*x*y) = 1) :
  ∃ (a b : Fin 3), a ≠ b ∧ 
    ((fun i => match i with
      | 0 => (y^2 + z^2 - x^2) / (2*y*z)
      | 1 => (z^2 + x^2 - y^2) / (2*z*x)
      | 2 => (x^2 + y^2 - z^2) / (2*x*y)
      | _ => 0) a = 1 ∧
     (fun i => match i with
      | 0 => (y^2 + z^2 - x^2) / (2*y*z)
      | 1 => (z^2 + x^2 - y^2) / (2*z*x)
      | 2 => (x^2 + y^2 - z^2) / (2*x*y)
      | _ => 0) b = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l1198_119894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1198_119862

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  /-- Speed of the boat along the stream in km/hr -/
  alongStream : ℝ
  /-- Speed of the boat against the stream in km/hr -/
  againstStream : ℝ

/-- Calculates the speed of the boat in still water given its speeds along and against a stream -/
noncomputable def stillWaterSpeed (b : BoatSpeed) : ℝ :=
  (b.alongStream + b.againstStream) / 2

/-- Theorem stating that a boat traveling 21 km/hr along a stream and 9 km/hr against the stream
    has a speed of 15 km/hr in still water -/
theorem boat_speed_in_still_water (b : BoatSpeed)
    (h1 : b.alongStream = 21)
    (h2 : b.againstStream = 9) :
    stillWaterSpeed b = 15 := by
  sorry

#eval "Boat speed theorem defined successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1198_119862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1198_119865

/-- The equation of the given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 / 5 - y^2 / 3 = 1

/-- The equation of the sought hyperbola -/
def sought_hyperbola (x y : ℝ) : Prop := x^2 / 10 - y^2 / 6 = 1

/-- The equation of the directrix -/
def directrix (x : ℝ) : Prop := x = 5/2

/-- The slope of an asymptote of a hyperbola -/
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

theorem hyperbola_satisfies_conditions :
  ∃ (a b : ℝ), 
    (asymptote_slope 5 (Real.sqrt 3) = asymptote_slope (Real.sqrt 10) (Real.sqrt 6)) ∧
    (directrix (10 / (2 * Real.sqrt 10))) ∧
    (∀ x y : ℝ, sought_hyperbola x y ↔ x^2 / 10 - y^2 / 6 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1198_119865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1198_119819

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem statement -/
theorem ellipse_theorem (e : Ellipse) 
  (A : PointOnEllipse e) 
  (B : PointOnEllipse e)
  (F : PointOnEllipse e)
  (P : PointOnEllipse e)
  (h_A : A.x = -e.a ∧ A.y = 0)
  (h_B : B.x = e.a ∧ B.y = 0)
  (h_F : F.x = -c ∧ F.y = 0)
  (h_P_diff : P ≠ A ∧ P ≠ B) :
  let k_AP := (P.y - A.y) / (P.x - A.x)
  let k_BP := (P.y - B.y) / (P.x - B.x)
  -- Part 1
  k_AP * k_BP = -e.b^2 / e.a^2 ∧
  -- Part 2
  (∃ (M N : ℝ × ℝ) (lambda : ℝ),
    -- M is on line l (x = a)
    M.1 = e.a ∧
    -- M is on line AP
    M.2 = k_AP * (M.1 - A.x) + A.y ∧
    -- N is on line BP
    N.2 = k_BP * (N.1 - B.x) + B.y ∧
    -- MN is perpendicular to BP
    (M.2 - N.2) * k_BP = -(M.1 - N.1) ∧
    -- F is on line MN
    (F.y - M.2) * (N.1 - M.1) = (F.x - M.1) * (N.2 - M.2) ∧
    -- AF = lambda * FB
    e.a - F.x = lambda * (e.a + F.x) ∧
    lambda = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1198_119819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_30_l1198_119800

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The hour value at the given time -/
def given_hour : ℕ := 5

/-- The minute value at the given time -/
def given_minute : ℕ := 30

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  (h * (full_circle / clock_hours : ℝ) + m * (full_circle / (clock_hours * 60) : ℝ))

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  m * (full_circle / 60 : ℝ)

/-- The smaller angle between two angles on a circle -/
noncomputable def smaller_angle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (full_circle - abs (a - b))

theorem clock_angle_at_5_30 :
  smaller_angle (hour_hand_angle given_hour given_minute) (minute_hand_angle given_minute) = 15 := by
  sorry

#eval clock_hours
#eval full_circle
#eval given_hour
#eval given_minute

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_30_l1198_119800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_variance_bound_l1198_119817

noncomputable def variance (x y z : ℝ) : ℝ := 
  ((x - (x + y + z) / 3)^2 + (y - (x + y + z) / 3)^2 + (z - (x + y + z) / 3)^2) / 3

theorem angle_variance_bound (α β γ : ℝ) (h : α + β + γ = 2 * Real.pi) :
  variance α β γ < (2 * Real.pi^2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_variance_bound_l1198_119817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l1198_119812

theorem closest_integer_to_cube_root_250 :
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l1198_119812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelter_cat_count_l1198_119849

theorem shelter_cat_count : 
  ∀ (total_adults : ℕ) (female_percentage : ℚ) (litter_percentage : ℚ) (kittens_per_litter : ℕ),
    total_adults = 100 →
    female_percentage = 2/5 →
    litter_percentage = 2/3 →
    kittens_per_litter = 3 →
    (total_adults + 
     (Int.floor (↑total_adults * female_percentage * litter_percentage) * kittens_per_litter)) = 181 :=
by
  intros total_adults female_percentage litter_percentage kittens_per_litter
  intros h_total h_female h_litter h_kittens
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelter_cat_count_l1198_119849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_minimum_and_f_inequality_l1198_119858

noncomputable section

-- Define the function h
def h (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x + a

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x - a*Real.exp 1 + Real.exp 1 + 15/2

theorem h_minimum_and_f_inequality (a b : ℝ) :
  (a = 3) →
  (∀ x₁ ∈ Set.Icc (-1) 1, ∃ x₂ ∈ Set.Icc 1 2, h a x₁ ≥ f a b x₂) →
  b ∈ Set.Ici (17/8) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_minimum_and_f_inequality_l1198_119858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_for_special_function_l1198_119864

-- Define the function type
def ContinuousFunction := {f : ℝ → ℝ // ContinuousOn f (Set.Icc 0 1)}

-- State the theorem
theorem integral_bounds_for_special_function 
  (f : ContinuousFunction) 
  (h : ∀ x ∈ Set.Icc 0 1, f.1 (f.1 x) = 1) 
  (h_range : ∀ x ∈ Set.Icc 0 1, f.1 x ∈ Set.Icc 0 1) :
  ∃ (I : ℝ), I ∈ Set.Ioo (3/4) 1 ∧ 
  I = ∫ x in Set.Icc 0 1, f.1 x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_for_special_function_l1198_119864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_17_l1198_119879

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_17_l1198_119879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_kennedy_ratio_l1198_119886

/-- The number of airline passengers (in millions) traveling to or from the United States in 1979 -/
noncomputable def total_passengers : ℝ := 38.3

/-- The number of passengers (in millions) using Logan Airport -/
noncomputable def logan_passengers : ℝ := 1.5958333333333332

/-- The number of passengers (in millions) using Kennedy Airport -/
noncomputable def kennedy_passengers : ℝ := (1/3) * total_passengers

/-- The number of passengers (in millions) using Miami Airport -/
noncomputable def miami_passengers : ℝ := 4 * logan_passengers

/-- The theorem stating that the ratio of Miami Airport passengers to Kennedy Airport passengers is 1/2 -/
theorem miami_kennedy_ratio : miami_passengers / kennedy_passengers = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_kennedy_ratio_l1198_119886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l1198_119892

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (2^(x+1) + m) / (2^x - 1)

-- State the theorem
theorem odd_function_implies_m_equals_two :
  (∀ x, f x m = -f (-x) m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l1198_119892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equals_expected_l1198_119816

/-- The ratio of the volume of a regular hexagonal pyramid to the volume of a regular triangular pyramid -/
noncomputable def volume_ratio (a : ℝ) : ℝ :=
  let hex_base_area := (3 / 2) * a^2 * Real.sqrt 3
  let tri_base_area := (a^2 * Real.sqrt 3) / 4
  let slant_height := 2 * a
  let hex_height := Real.sqrt ((slant_height^2) - ((3 * a / (2 * Real.sqrt 3))^2))
  let tri_height := Real.sqrt ((slant_height^2) - ((a * Real.sqrt 3 / 6)^2))
  let hex_volume := (1 / 3) * hex_base_area * hex_height
  let tri_volume := (1 / 3) * tri_base_area * tri_height
  hex_volume / tri_volume

/-- The theorem stating the ratio of the volumes -/
theorem volume_ratio_equals_expected (a : ℝ) (h : a > 0) :
  volume_ratio a = 6 * Real.sqrt 1833 / 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equals_expected_l1198_119816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_minus_half_correct_l1198_119861

/-- The mathematical representation of "the square of x minus half of y" -/
noncomputable def square_minus_half (x y : ℝ) : ℝ := x^2 - y/2

/-- Theorem stating that the mathematical representation of "the square of x minus half of y" is correct -/
theorem square_minus_half_correct (x y : ℝ) : 
  square_minus_half x y = x^2 - y/2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_minus_half_correct_l1198_119861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_consumption_l1198_119808

/-- Represents the daily birdseed consumption for different bird types -/
structure BirdseedConsumption where
  parakeet : ℕ
  finch : ℕ
  parrot : ℕ

/-- Represents the number of birds Peter has -/
structure BirdCount where
  parakeets : ℕ
  parrots : ℕ
  finches : ℕ

/-- Calculates the total weekly birdseed consumption -/
def weeklyConsumption (consumption : BirdseedConsumption) (count : BirdCount) : ℕ :=
  7 * (consumption.parakeet * count.parakeets + 
       consumption.finch * count.finches + 
       consumption.parrot * count.parrots)

/-- Theorem: Each parrot eats 14 grams of birdseed per day -/
theorem parrot_consumption 
  (consumption : BirdseedConsumption)
  (count : BirdCount)
  (h1 : consumption.parakeet = 2)
  (h2 : consumption.finch = 1)
  (h3 : count.parakeets = 3)
  (h4 : count.parrots = 2)
  (h5 : count.finches = 4)
  (h6 : weeklyConsumption consumption count = 266) :
  consumption.parrot = 14 := by
  sorry

#eval weeklyConsumption 
  { parakeet := 2, finch := 1, parrot := 14 }
  { parakeets := 3, parrots := 2, finches := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_consumption_l1198_119808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1198_119884

/-- The ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_point : a^2 * 3 + b^2 * 3/4 = a^2 * b^2

/-- A line not passing through the origin -/
structure Line where
  k : ℝ
  m : ℝ
  h_m : m ≠ 0

/-- Helper function to check if a point is on the ellipse -/
def on_ellipse (C : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / C.a^2) + (p.2^2 / C.b^2) = 1

/-- Helper function to check if a point is on the line -/
def on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * p.1 + l.m

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2)) / 2

/-- The theorem statement -/
theorem max_triangle_area (C : Ellipse) (l : Line) :
  ∃ (A B : ℝ × ℝ),
    on_ellipse C A ∧ on_ellipse C B ∧
    on_line l A ∧ on_line l B ∧
    (∃ N : ℝ × ℝ, on_line l N ∧ N.2 = N.1 / 2 ∧ N = ((A.1 + B.1), (A.2 + B.2)) / 2) →
    (∀ A' B' : ℝ × ℝ,
      on_ellipse C A' ∧ on_ellipse C B' ∧ on_line l A' ∧ on_line l B' →
      area_triangle (0, 0) A B ≥ area_triangle (0, 0) A' B') ∧
    area_triangle (0, 0) A B = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1198_119884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1198_119855

/-- The distance to the place in kilometers -/
noncomputable def distance : ℝ := 840 / 19

/-- The rowing speed in still water in km/h -/
def rowing_speed : ℝ := 10

/-- The downstream current speed in km/h -/
def downstream_current : ℝ := 2

/-- The upstream current speed in km/h -/
def upstream_current : ℝ := 3

/-- The total time for the round trip in hours -/
def total_time : ℝ := 10

theorem distance_calculation :
  let downstream_speed := rowing_speed + downstream_current
  let upstream_speed := rowing_speed - upstream_current
  (distance / downstream_speed) + (distance / upstream_speed) = total_time := by
  sorry

/-- Approximate evaluation of the distance -/
def distance_approx : Float := 840 / 19

#eval distance_approx -- Should output approximately 44.21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1198_119855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_l1198_119867

/-- The probability of getting exactly two ones when rolling five fair dice -/
def p : ℚ := 1250 / 7776

/-- The number of rolls -/
def n : ℕ := 20

/-- X is a discrete random variable representing the number of rolls of five dice
    where exactly two dice show a one, given that the total number of rolls is twenty -/
def X : ℕ → ℚ := sorry

/-- The expected value of X -/
def expected_value : ℚ := n * p

theorem expected_value_of_X : expected_value = 25000 / 7776 := by
  unfold expected_value
  unfold n
  unfold p
  norm_num

#eval expected_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_l1198_119867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1198_119899

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the point of tangency
def p : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ m*x - y + b = 0) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| → |h| < δ → |(f (p.1 + h) - f p.1) / h - m| < ε) ∧
    (m * p.1 - p.2 + b = 0) ∧
    (m = 4 ∧ b = -1) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1198_119899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_fill_time_l1198_119821

/-- Represents the rate at which a pipe can fill a tank (in tanks per minute) -/
abbrev FillRate := ℝ

/-- Represents the time taken to fill a tank (in minutes) -/
abbrev FillTime := ℝ

/-- Represents the capacity of a tank (in liters) -/
abbrev TankCapacity := ℝ

/-- Represents the rate of a leak (in liters per hour) -/
abbrev LeakRate := ℝ

/-- Theorem stating the time taken by the slower pipe to fill the tank -/
theorem slower_pipe_fill_time 
  (r : FillRate) -- Rate of the slower (second) pipe
  (t : FillTime) -- Time taken by all pipes together
  (c : TankCapacity) -- Capacity of the tank
  (l : LeakRate) -- Rate of the leak
  (h1 : r > 0) -- The fill rate is positive
  (h2 : t > 0) -- The fill time is positive
  (h3 : c > 0) -- The tank capacity is positive
  (h4 : l ≥ 0) -- The leak rate is non-negative
  (h5 : 7 * r * t = 1) -- All three pipes together fill the tank in t minutes
  (h6 : l = 5) -- The leak rate is 5 liters per hour
  : FillTime := by
  -- The proof goes here
  sorry

#check slower_pipe_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_fill_time_l1198_119821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_third_l1198_119831

-- Define the angle 30 degrees in radians
noncomputable def angle_30 : Real := Real.pi / 6

-- State the theorem
theorem trigonometric_expression_equals_one_third :
  (Real.tan angle_30)^2 - (Real.sin angle_30)^2 = (1/3) * (Real.tan angle_30)^2 * (Real.cos angle_30)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_third_l1198_119831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_bounds_l1198_119882

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : 0 < side

-- Define a line segment within the triangle
structure SegmentInTriangle (t : EquilateralTriangle) where
  length : ℝ
  length_positive : 0 < length
  fits_in_triangle : length ≤ t.side

-- Define the sum of projections
noncomputable def sum_of_projections (t : EquilateralTriangle) (s : SegmentInTriangle t) (angle : ℝ) : ℝ :=
  2 * s.length * Real.cos angle

-- State the theorem
theorem projection_sum_bounds (t : EquilateralTriangle) (s : SegmentInTriangle t) :
  ∃ (max_sum min_sum : ℝ),
    (∀ (angle : ℝ), sum_of_projections t s angle ≤ max_sum) ∧
    (∀ (angle : ℝ), min_sum ≤ sum_of_projections t s angle) ∧
    max_sum = 2 * s.length ∧
    (s.length ≤ Real.sqrt 3 / 2 * t.side → min_sum = Real.sqrt 3 * s.length) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_bounds_l1198_119882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1198_119872

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ 
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the next triangle in the sequence based on the excircle --/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  let s := (t.a + t.b + t.c) / 2
  Triangle.mk (s - t.a) (s - t.b) (s - t.c)

/-- Recursively defines the sequence of triangles --/
noncomputable def triangle_sequence : ℕ → Triangle
  | 0 => Triangle.mk 1011 1012 1013
  | n + 1 => next_triangle (triangle_sequence n)

/-- Finds the index of the last valid triangle in the sequence --/
noncomputable def last_valid_triangle_index : ℕ := sorry

/-- The perimeter of a triangle --/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Main theorem: The perimeter of the last valid triangle is 1509/512 --/
theorem last_triangle_perimeter :
  perimeter (triangle_sequence last_valid_triangle_index) = 1509 / 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1198_119872
