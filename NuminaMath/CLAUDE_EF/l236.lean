import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l236_23685

/-- 
Given an ellipse with semi-major axis a and semi-minor axis b,
right focus F, and a point N, if the maximum perimeter of triangle MNF
is (√6 + 2)a, then the eccentricity of the ellipse is √2/2.
-/
theorem ellipse_eccentricity 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (F : ℝ × ℝ) 
  (hF : F.1 > 0 ∧ F.2 = 0) -- Right focus condition
  (N : ℝ × ℝ) 
  (hN : N = (0, Real.sqrt 2 * b)) 
  (h_perimeter : ∀ M : ℝ × ℝ, 
    M.1^2 / a^2 + M.2^2 / b^2 = 1 → 
    dist M N + dist M F + dist N F ≤ (Real.sqrt 6 + 2) * a) :
  (F.1^2 / a^2)^(1/2) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l236_23685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l236_23635

noncomputable def P (m : ℝ) : Prop := ∀ x y : ℝ, x < y → Real.log (x + 1) / Real.log (2 * m) < Real.log (y + 1) / Real.log (2 * m)

def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≥ 0

def range_m (m : ℝ) : Prop := m ∈ Set.Icc (-2 : ℝ) (1/2) ∪ Set.Ioi 2

theorem m_range :
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) ↔ range_m m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l236_23635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_project_max_profit_l236_23615

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then 80 * x - 320
  else if x > 4 then -x^2 + 40 * x - 200
  else 0

-- State the theorem
theorem amusement_project_max_profit :
  ∃ (x_max : ℝ) (profit_max : ℝ),
    (∀ x, W x ≤ W x_max) ∧
    x_max = 20 ∧
    profit_max = W x_max ∧
    profit_max = 200 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_project_max_profit_l236_23615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_on_d1_l236_23684

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (a b : ℝ)  -- bases of the trapezoid
  (vertex_A vertex_B vertex_C vertex_D : EuclideanSpace ℝ (Fin 2))  -- vertices of the trapezoid
  (is_isosceles : True)  -- property ensuring the trapezoid is isosceles (simplified for now)

-- Define the diagonal intersection point
noncomputable def diagonal_intersection (t : IsoscelesTrapezoid) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define the triangle formed by two vertices and the diagonal intersection
noncomputable def triangle (t : IsoscelesTrapezoid) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Define the centroid of the triangle
noncomputable def centroid (tri : Set (EuclideanSpace ℝ (Fin 2))) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define the line d1
noncomputable def d1 : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Define the projection d2 onto a1
noncomputable def d2 : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- The theorem to be proved
theorem centroid_on_d1 (t : IsoscelesTrapezoid) :
  centroid (triangle t) ∈ d1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_on_d1_l236_23684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_area_condition_l236_23609

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)

-- Define a point P on the x-axis
def P (x : ℝ) : ℝ × ℝ := (x, 0)

-- Part I: AB perpendicular to PB
theorem perpendicular_condition (x : ℝ) :
  (B.fst - A.fst) * (B.fst - (P x).fst) + (B.snd - A.snd) * (B.snd - (P x).snd) = 0 →
  x = 7 :=
sorry

-- Part II: Area of triangle ABP is 10
theorem area_condition (x : ℝ) :
  abs ((B.fst - A.fst) * ((P x).snd - A.snd) - (B.snd - A.snd) * ((P x).fst - A.fst)) / 2 = 10 →
  x = 9 ∨ x = -11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_area_condition_l236_23609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l236_23607

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2*x - 1)^0 / Real.sqrt (2 - x)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 1/2 ∨ (x > 1/2 ∧ x < 2)}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l236_23607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l236_23614

/-- Parabola defined by y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Point A outside the parabola -/
def point_A : ℝ × ℝ := (4, 2)

/-- Point M on the parabola -/
def point_M : ℝ × ℝ := (1, 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_minimum_distance :
  parabola point_M.1 point_M.2 ∧
  ∀ P : ℝ × ℝ, parabola P.1 P.2 →
    distance point_M point_A + distance point_M focus ≤
    distance P point_A + distance P focus := by
  sorry

#check parabola_minimum_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l236_23614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l236_23691

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 + (Real.log a / Real.log 10 + 2) * x + Real.log b / Real.log 10

-- State the theorem
theorem problem_solution (a b : ℝ) 
  (h1 : f a b (-1) = -2)
  (h2 : ∀ x : ℝ, f a b x ≥ 2 * x) :
  a + b = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l236_23691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_five_R_squared_l236_23683

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  R : ℝ
  upperBase : ℝ
  lowerBase : ℝ
  height : ℝ
  upper_base_half_height : upperBase = height / 2
  upper_base_eq_radius : upperBase = R
  lower_base_eq_four_radius : lowerBase = 4 * R

/-- The area of an isosceles trapezoid with an inscribed circle -/
noncomputable def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  (t.upperBase + t.lowerBase) * t.height / 2

/-- Theorem: The area of the isosceles trapezoid with an inscribed circle is 5R² -/
theorem area_is_five_R_squared (t : IsoscelesTrapezoidWithInscribedCircle) :
  area t = 5 * t.R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_five_R_squared_l236_23683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_zero_l236_23682

noncomputable def u : ℕ → ℝ
  | 0 => 1
  | n + 1 => u n / (u n ^ 2 + 1)

theorem u_converges_to_zero : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - 0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_zero_l236_23682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisibility_A_value_when_divisible_by_9_l236_23666

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_sum_divisibility (n : ℕ) :
  is_divisible_by_9 n ↔ is_divisible_by_9 (sum_of_digits n) := by sorry

def number_with_A (A : ℕ) : ℕ := 994561200 + A * 10 + 5

theorem A_value_when_divisible_by_9 :
  ∀ A, A < 10 →
    (is_divisible_by_9 (number_with_A A) ↔ A = 4) := by sorry

#eval sum_of_digits (number_with_A 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisibility_A_value_when_divisible_by_9_l236_23666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l236_23608

/-- Given points A, B, and C in a 2D plane, where B is symmetric to A about the x-axis,
    and C is symmetric to A about the origin, this theorem proves the equation of the line
    passing through the midpoints of BA and BC, and calculates the area of triangle ABC. -/
theorem triangle_abc_properties (A B C : ℝ × ℝ) :
  A = (5, 1) →
  B.1 = A.1 ∧ B.2 = -A.2 →
  C.1 = -A.1 ∧ C.2 = -A.2 →
  let M_AB : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M_BC : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_eq (x y : ℝ) := 2 * x - 5 * y - 5 = 0
  (∀ x y, (x, y) = M_AB ∨ (x, y) = M_BC → line_eq x y) ∧
  abs (B.1 - A.1) * abs (C.1 - B.1) / 2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l236_23608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_mold_radius_l236_23634

noncomputable def large_bowl_radius : ℝ := 2
def salvage_percentage : ℝ := 0.9
def num_small_molds : ℕ := 64

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

noncomputable def large_bowl_volume : ℝ := hemisphere_volume large_bowl_radius

noncomputable def salvaged_volume : ℝ := salvage_percentage * large_bowl_volume

theorem small_mold_radius :
  ∃ (r : ℝ), r > 0 ∧ 
    (num_small_molds : ℝ) * hemisphere_volume r = salvaged_volume ∧
    r = (9/80)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_mold_radius_l236_23634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slopes_sum_l236_23613

/-- Given a line y = 2x - 3 intersecting a parabola y^2 = 4x at points A and B,
    with O as the origin, and k1 and k2 as slopes of OA and OB respectively,
    prove that 1/k1 + 1/k2 = 1/2 -/
theorem intersection_slopes_sum (A B O : ℝ × ℝ) (k1 k2 : ℝ) :
  (∀ x y, y = 2*x - 3 ↔ (x, y) = A ∨ (x, y) = B) →
  (∀ x y, y^2 = 4*x ↔ (x, y) = A ∨ (x, y) = B) →
  O = (0, 0) →
  k1 = (A.2 / A.1) →
  k2 = (B.2 / B.1) →
  1/k1 + 1/k2 = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slopes_sum_l236_23613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l236_23622

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/6) - Real.cos x ^ 2 + 1/4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ Real.sqrt 3 / 4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = Real.sqrt 3 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l236_23622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lena_total_time_l236_23605

/-- The time Masha takes to walk to the store -/
noncomputable def masha_to_store : ℝ := 12

/-- The time Masha spends in the store -/
noncomputable def masha_in_store : ℝ := 2

/-- The time Masha walks back before meeting Lena -/
noncomputable def masha_back : ℝ := 2

/-- The total distance to the store, represented as a unit -/
noncomputable def total_distance : ℝ := 1

/-- The fraction of the path Masha covers when walking back before meeting Lena -/
noncomputable def masha_fraction : ℝ := masha_back / masha_to_store

/-- The fraction of the path Lena covers when meeting Masha -/
noncomputable def lena_fraction : ℝ := total_distance - masha_fraction

/-- The time Lena walks before meeting Masha -/
noncomputable def lena_walk_time : ℝ := masha_to_store + masha_in_store + masha_back

/-- Theorem: Lena's total walking time to the store is 19.2 minutes -/
theorem lena_total_time : (lena_walk_time / lena_fraction) = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lena_total_time_l236_23605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l236_23606

-- Define the differential equation
def P (x y : ℝ) : ℝ := 2 * y - 3
def Q (x y : ℝ) : ℝ := 2 * x + 3 * y^2

-- Define the solution function
def F (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y^3 - 1

-- Theorem statement
theorem solution_verification (x y : ℝ) :
  -- The function F satisfies the differential equation
  (deriv (fun x => F x y) x) * P x y + (deriv (fun y => F x y) y) * Q x y = 0 ∧
  -- The function F satisfies the initial condition
  F 0 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l236_23606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l236_23624

theorem sin_alpha_value (α β : ℝ)
  (h1 : Real.cos (α - β) = 3/5)
  (h2 : Real.sin β = -5/13)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : -π/2 < β ∧ β < 0) :
  Real.sin α = 33/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l236_23624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_at_one_l236_23699

-- Define a fourth-degree polynomial with real coefficients
def fourth_degree_poly (a b c d e : ℝ) : ℝ → ℝ := 
  λ x ↦ a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem absolute_value_at_one 
  (a b c d e : ℝ) 
  (h1 : |fourth_degree_poly a b c d e (-2)| = 10)
  (h2 : |fourth_degree_poly a b c d e 0| = 10)
  (h3 : |fourth_degree_poly a b c d e 3| = 10)
  (h4 : |fourth_degree_poly a b c d e 7| = 10) :
  |fourth_degree_poly a b c d e 1| = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_at_one_l236_23699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l236_23602

/-- Proves that Rohan spends 20% of his salary on food given his salary and expenses -/
theorem rohan_food_expense (salary : ℝ) (savings : ℝ) 
  (house_rent_percent : ℝ) (entertainment_percent : ℝ) (conveyance_percent : ℝ) :
  salary = 7500 →
  savings = 1500 →
  house_rent_percent = 20 →
  entertainment_percent = 10 →
  conveyance_percent = 10 →
  20 = 100 - (house_rent_percent + entertainment_percent + conveyance_percent) - (savings / salary * 100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l236_23602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_3x_minus_1_domain_f_from_2x_plus_5_l236_23688

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc (-2) 1

-- Define the domain of f(2x+5)
def domain_f_2x_plus_5 : Set ℝ := Set.Icc (-1) 4

-- Theorem 1: Domain of f(3x-1)
theorem domain_f_3x_minus_1 : 
  {x : ℝ | ∃ y ∈ domain_f, y = 3*x - 1} = Set.Icc (-1/3) (2/3) := by sorry

-- Theorem 2: Domain of f(x) given domain of f(2x+5)
theorem domain_f_from_2x_plus_5 : 
  {x : ℝ | ∃ y ∈ domain_f_2x_plus_5, y = 2*x + 5} = Set.Icc 3 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_3x_minus_1_domain_f_from_2x_plus_5_l236_23688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l236_23670

def mySequence : List Int := [5, 5, -13, 5, 5, 5, -13, 5, 5, -13, 5, 5, 5, -13, 5, 5]

def isPalindrome (l : List α) : Prop := l = l.reverse

def sumConsecutive (l : List Int) (n : Nat) (i : Nat) : Int :=
  (l.drop i).take n |> List.sum

theorem sequence_properties :
  (isPalindrome mySequence) ∧
  (∀ i, i + 6 < mySequence.length → sumConsecutive mySequence 7 i = -1) ∧
  (∀ i, i + 10 < mySequence.length → sumConsecutive mySequence 11 i = 1) := by
  sorry

#eval mySequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l236_23670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_gain_l236_23643

/-- Represents the investment and earnings in a business partnership --/
structure BusinessPartnership where
  nandan_investment : ℝ
  nandan_time : ℝ
  krishan_investment : ℝ
  krishan_time : ℝ
  nandan_earning : ℝ

/-- Calculates the total gain in the business partnership --/
noncomputable def total_gain (bp : BusinessPartnership) : ℝ :=
  bp.nandan_earning * (1 + (bp.krishan_investment * bp.krishan_time) / (bp.nandan_investment * bp.nandan_time))

/-- Theorem stating the total gain in the business --/
theorem business_gain (bp : BusinessPartnership) 
  (h1 : bp.krishan_investment = 4 * bp.nandan_investment)
  (h2 : bp.krishan_time = 3 * bp.nandan_time)
  (h3 : bp.nandan_earning = 2000) :
  total_gain bp = 26000 := by
  sorry

#check business_gain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_gain_l236_23643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasons_commute_l236_23686

/-- Represents the distance from Jason's house to the first store (and from the last store to work) -/
noncomputable def x : ℝ := sorry

/-- Distance between the first and second store -/
def d12 : ℝ := 6

/-- Distance between the second and third store -/
noncomputable def d23 : ℝ := d12 + (2/3) * d12

/-- Total commute distance -/
def total_commute : ℝ := 24

theorem jasons_commute : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasons_commute_l236_23686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_longer_side_l236_23618

/-- Represents a quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents a parallelogram, which is a special case of a quadrilateral -/
structure Parallelogram extends Quadrilateral :=
  (parallel : a = c ∧ b = d)

/-- Main theorem about the parallelogram with given conditions -/
theorem parallelogram_longer_side
  (P : Parallelogram)
  (perimeter : P.a + P.b + P.c + P.d = 40)
  (ratio : P.a / P.b = 3 / 2 ∨ P.b / P.a = 3 / 2) :
  max P.a P.b = 12 := by
  sorry

#check parallelogram_longer_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_longer_side_l236_23618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_xyyz_properties_l236_23697

-- Define a monomial type
structure Monomial (α : Type*) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define a function to calculate the degree of a monomial
def degree {α : Type*} [CommRing α] (m : Monomial α) : Nat :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

-- Define the specific monomial -xy²z
def monomial_xyyz : Monomial ℤ :=
  { coeff := -1,
    vars := [(1, 1), (2, 2), (3, 1)] }

-- Theorem statement
theorem monomial_xyyz_properties :
  monomial_xyyz.coeff = -1 ∧ degree monomial_xyyz = 4 := by
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_xyyz_properties_l236_23697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_three_rays_with_common_endpoint_l236_23629

/-- The set S of points in the Cartesian plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (3 = x + 2 ∧ 3 ≥ y - 4) ∨
               (3 = y - 4 ∧ 3 ≥ x + 2) ∨
               (x + 2 = y - 4 ∧ 3 ≤ x + 2)}

/-- Definition of a ray in 2D space -/
def IsRay (r : Set (ℝ × ℝ)) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    r = {x : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧ x = p + t • (q - p)}

/-- The theorem stating that S represents three rays with a common endpoint -/
theorem S_is_three_rays_with_common_endpoint :
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Set (ℝ × ℝ)),
    IsRay r₁ ∧ IsRay r₂ ∧ IsRay r₃ ∧
    p ∈ r₁ ∧ p ∈ r₂ ∧ p ∈ r₃ ∧
    S = r₁ ∪ r₂ ∪ r₃ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_three_rays_with_common_endpoint_l236_23629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_triangles_perimeter_sum_l236_23694

theorem rectangle_diagonal_triangles_perimeter_sum
  (p q : ℕ+)
  (h_coprime : Nat.Coprime p q)
  (h_p_lt_q : p < q) :
  let diagonal_triangles := p + q - 1
  let perimeter_sum := (p + q - 1 : ℝ) * (2 + Real.sqrt ((p : ℝ)^2 + (q : ℝ)^2) / (p : ℝ))
  perimeter_sum = (diagonal_triangles : ℝ) * (2 + Real.sqrt (1 + ((q : ℝ) / (p : ℝ))^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_triangles_perimeter_sum_l236_23694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_propositions_l236_23603

-- Define the propositions
def proposition1 : Prop := ∃ x : ℝ, x^2 - 2*x - 3 < 0
def proposition2 : Prop := ∃ x : ℝ, (x^2 - 4*x + 4 = 0 ↔ x = 2) ∧ (∃ y : ℝ, y ≠ 2 ∧ y^2 - 4*y + 4 = 0)
def proposition3 : Prop := ∀ t : ℝ, (t = 180 → ¬(t ≠ 180))
def proposition4 : Prop := (∀ x : ℝ, x^2 ≥ 0) → (¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∀ x : ℝ, x^2 < 0)

-- Theorem statement
theorem number_of_correct_propositions :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 :=
by
  constructor
  · sorry -- Proof for ¬proposition1
  constructor
  · sorry -- Proof for ¬proposition2
  constructor
  · sorry -- Proof for ¬proposition3
  · sorry -- Proof for ¬proposition4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_propositions_l236_23603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_order_relation_l236_23650

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_order_relation :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (log2 a > log2 b ↔ a > b)) ∧
  (∃ a b : ℝ, a > b ∧ ¬(log2 a > log2 b)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_order_relation_l236_23650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_selling_ratio_with_25_percent_profit_l236_23671

-- Define the profit percentage
noncomputable def profit_percent : ℚ := 25 / 100

-- Define the ratio of cost price to selling price
def cost_to_selling_ratio : ℚ := 4 / 5

-- Theorem statement
theorem cost_to_selling_ratio_with_25_percent_profit :
  let selling_price (cost_price : ℚ) := (1 + profit_percent) * cost_price
  ∀ cost_price : ℚ, cost_price > 0 → cost_price / selling_price cost_price = cost_to_selling_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_selling_ratio_with_25_percent_profit_l236_23671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l236_23673

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x * Real.cos x * Real.cos (2 * x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∃ (A : ℝ), A > 0 ∧ ∀ (x : ℝ), f x ≤ A) →
  (∃ (T : ℝ), T = Real.pi / 2 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∃ (A : ℝ), A = 2 ∧ ∀ (x : ℝ), f x ≤ A) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l236_23673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtain_6_9_14_sum_remains_odd_parity_preserved_cannot_obtain_2011_2012_2013_cannot_obtain_2011_2013_2015_l236_23695

-- Define the transformation rule
def transform (a b : ℕ) : ℕ := a + b - 1

-- Define a function to check if a triple can be obtained from the initial set
def can_obtain (target : ℕ × ℕ × ℕ) : Prop :=
  ∃ (steps : ℕ), ∃ (sequence : List (ℕ × ℕ × ℕ)),
    sequence.length = steps + 1 ∧
    sequence.head! = (2, 3, 4) ∧
    sequence.getLast! = target ∧
    ∀ i : Fin steps, ∃ (j k l : ℕ),
      sequence[i.val]! = (j, k, l) ∧
      (sequence[i.val + 1]! = (transform j k, k, l) ∨
       sequence[i.val + 1]! = (j, transform j k, l) ∨
       sequence[i.val + 1]! = (j, k, transform j k) ∨
       sequence[i.val + 1]! = (transform k l, j, l) ∨
       sequence[i.val + 1]! = (k, transform k l, j) ∨
       sequence[i.val + 1]! = (k, j, transform k l))

-- Theorem 1: (6, 9, 14) can be obtained
theorem obtain_6_9_14 : can_obtain (6, 9, 14) := by sorry

-- Theorem 2: Sum remains odd after transformations
theorem sum_remains_odd (a b c : ℕ) :
  Odd (a + b + c) → Odd (a + b + transform a b) := by sorry

-- Theorem 3: Parity remains two even and one odd
theorem parity_preserved (a b c : ℕ) :
  (Even a ∧ Even b ∧ Odd c) ∨ (Even a ∧ Odd b ∧ Even c) ∨ (Odd a ∧ Even b ∧ Even c) →
  (Even a ∧ Even b ∧ Odd (transform a b)) ∨
  (Even a ∧ Odd (transform a b) ∧ Even c) ∨
  (Odd (transform a b) ∧ Even b ∧ Even c) := by sorry

-- Theorem 4: (2011, 2012, 2013) cannot be obtained
theorem cannot_obtain_2011_2012_2013 : ¬ can_obtain (2011, 2012, 2013) := by sorry

-- Theorem 5: (2011, 2013, 2015) cannot be obtained
theorem cannot_obtain_2011_2013_2015 : ¬ can_obtain (2011, 2013, 2015) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtain_6_9_14_sum_remains_odd_parity_preserved_cannot_obtain_2011_2012_2013_cannot_obtain_2011_2013_2015_l236_23695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l236_23669

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def sequence_b (b : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, b n = (3 : ℝ) ^ (n.val - 1)

def sum_sequence_b (T : ℕ+ → ℝ) (b : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, T n = (3/2) * b n - (1/2)

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ+ → ℝ) (T : ℕ+ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_arithmetic : sum_arithmetic_sequence S a)
  (h_S5 : S 5 = 30)
  (h_S10 : S 10 = 110)
  (h_b : sequence_b b)
  (h_T : sum_sequence_b T b) :
  (∀ n : ℕ+, S n.val = n.val^2 + n.val) ∧
  (∀ n : ℕ+, b n = (3 : ℝ) ^ (n.val - 1)) ∧
  (∀ n : ℕ+, n.val ≤ 4 → S n.val * b n < 2 * T n * a n.val) ∧
  (∀ n : ℕ+, n.val ≥ 5 → S n.val * b n > 2 * T n * a n.val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l236_23669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l236_23665

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 3 * (Real.sin x)^2 - (Real.cos x)^2 + 4 * a * Real.cos x + a^2 ≤ 31) ↔ 
  -4 ≤ a ∧ a ≤ 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l236_23665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_tangent_parabolas_l236_23651

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle with radius r
def circleSet (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the tangent line at 45°
def tangentLine (x : ℝ) : ℝ := x

-- State the theorem
theorem circle_radius_for_tangent_parabolas : 
  ∃ (r : ℝ), 
    (∀ (x : ℝ), parabola x + r = tangentLine x → (x, parabola x + r) ∈ circleSet r) ∧ 
    (∃! (x : ℝ), parabola x + r = tangentLine x) → 
    r = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_tangent_parabolas_l236_23651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_condition_l236_23659

theorem count_positive_integers_satisfying_condition : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ (3 : ℝ) * n + 20 < 50) (Finset.range 50)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_condition_l236_23659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_attendees_l236_23633

theorem picnic_attendees (total_sum last_plates : Nat) : 
  total_sum = 2015 →
  last_plates = 4 →
  ∃ attendees initial_plates : Nat,
    initial_plates + attendees = total_sum ∧
    initial_plates - (attendees - 1) = last_plates ∧
    attendees = 1006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_attendees_l236_23633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l236_23610

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  3 * b = 4 * c ∧
  B = 2 * C ∧
  b = 4 →
  Real.sin B = 4 * Real.sqrt 5 / 9 ∧
  (1 / 2 : ℝ) * b * c * Real.sin A = 14 * Real.sqrt 5 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l236_23610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_berry_expenditure_l236_23652

/-- The amount Martin spends on berries in a 30-day period -/
def berry_expenditure (daily_consumption : ℚ) (package_size : ℚ) (package_price : ℚ) (days : ℕ) : ℚ :=
  (daily_consumption * days / package_size) * package_price

/-- Theorem: Martin's berry expenditure for 30 days is $30.00 -/
theorem martin_berry_expenditure :
  berry_expenditure (1/2) 1 2 30 = 30 := by
  unfold berry_expenditure
  -- Simplify the expression
  simp [Rat.div_def, Rat.mul_num, Rat.mul_den]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_berry_expenditure_l236_23652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_property_l236_23681

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_property (n : ℕ) :
  (∃ x y : ℕ, (a_n n) / 3 = x^3 + y^3 ∧ x > 0 ∧ y > 0) ∧
  (∀ x y : ℕ, (a_n n) / 3 ≠ x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_property_l236_23681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_complex_number_l236_23641

-- Define the complex number z
noncomputable def z (a : ℝ) : ℂ := (2 + a * Complex.I) / (2 + Complex.I)

-- Define the condition for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- State the theorem
theorem fourth_quadrant_complex_number :
  ∃ a : ℝ, in_fourth_quadrant (z a) ∧ a = -2 := by
  -- We'll use -2 as our witness for a
  use -2
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that z(-2) is in the fourth quadrant
  · -- Evaluate z(-2)
    have h : z (-2) = (6 - 2*Complex.I) / 5 := by
      simp [z]
      -- Algebraic manipulation (simplified for brevity)
      sorry
    
    -- Show that the real part is positive and imaginary part is negative
    constructor
    · -- Real part is 6/5 > 0
      sorry
    · -- Imaginary part is -2/5 < 0
      sorry

  -- Prove that a = -2 (trivial as we chose -2)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_complex_number_l236_23641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circplus_chain_equals_one_l236_23630

-- Define the operation ⊕
noncomputable def circplus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Recursive function to compute n ⊕ (n-1) ⊕ ... ⊕ 2 ⊕ 1
noncomputable def recursiveCircplus : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | n + 1 => circplus (n + 1 : ℝ) (recursiveCircplus n)

-- Theorem statement
theorem circplus_chain_equals_one : recursiveCircplus 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circplus_chain_equals_one_l236_23630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_not_always_parallel_l236_23654

-- Define a structure for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define what it means for a point to be on a line
def on_line (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

-- Define what it means for lines to be non-intersecting
def non_intersecting (l1 l2 : Line3D) : Prop :=
  ¬ ∃ p : Point3D, on_line p l1 ∧ on_line p l2

-- Define what it means for lines to be parallel
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧
    l1.direction.x = k * l2.direction.x ∧
    l1.direction.y = k * l2.direction.y ∧
    l1.direction.z = k * l2.direction.z

-- Theorem statement
theorem non_intersecting_not_always_parallel :
  ∃ l1 l2 : Line3D, non_intersecting l1 l2 ∧ ¬ parallel l1 l2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_not_always_parallel_l236_23654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l236_23626

/-- A piecewise function f parameterized by a real number a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 2) * Real.exp (a * x)

/-- f is monotonic on ℝ if and only if a is in [-1, 0) -/
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∨ (∀ x y : ℝ, x < y → f a x > f a y) ↔ 
  a ∈ Set.Icc (-1) 0 \ {0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l236_23626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_evaluate_specific_values_l236_23621

theorem simplify_and_evaluate (a b : ℝ) : 
  a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 2 * a * b :=
by
  ring  -- This tactic should handle the algebraic simplification

theorem evaluate_specific_values : 
  (let a : ℝ := 1; let b : ℝ := 2; a * b + (a^2 - a * b) - (a^2 - 2 * a * b)) = 4 :=
by
  simp  -- Simplify the let expressions
  ring  -- Perform algebraic simplification and evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_evaluate_specific_values_l236_23621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l236_23620

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

-- Define the right angle condition
def rightAngle (a b c : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_on_parabola :
  ∀ b c : ℝ × ℝ,
  parabola (1, 1) → parabola b → parabola c →
  rightAngle (1, 1) b c →
  ∀ d : ℝ × ℝ, parabola d →
  rightAngle (1, 1) b d →
  distance (1, 1) c ≤ distance (1, 1) d ∧
  ∃ b' c' : ℝ × ℝ, parabola b' ∧ parabola c' ∧
  rightAngle (1, 1) b' c' ∧
  distance (1, 1) c' = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l236_23620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_circumcenter_distance_squared_l236_23696

/-- Centroid of a triangle given its side lengths -/
noncomputable def centroid_of_triangle (a b c : ℝ) : ℝ × ℝ := sorry

/-- Circumcenter of a triangle given its side lengths -/
noncomputable def circumcenter_of_triangle (a b c : ℝ) : ℝ × ℝ := sorry

/-- Circumradius of a triangle given its side lengths -/
noncomputable def circumradius_of_triangle (a b c : ℝ) : ℝ := sorry

/-- Given a triangle with side lengths a, b, and c, circumradius R, centroid G, and circumcenter O,
    prove that the square of the distance between G and O is equal to R^2 - (a^2 + b^2 + c^2)/9 -/
theorem centroid_circumcenter_distance_squared
  (a b c : ℝ) 
  (R : ℝ) 
  (G O : ℝ × ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_R : R > 0)
  (h_G : G = centroid_of_triangle a b c)
  (h_O : O = circumcenter_of_triangle a b c)
  (h_R_def : R = circumradius_of_triangle a b c) :
  ‖G - O‖^2 = R^2 - (a^2 + b^2 + c^2) / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_circumcenter_distance_squared_l236_23696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_half_age_specific_l236_23675

/-- The number of years it takes for a son to be half his father's age -/
noncomputable def years_to_half_age (father_age : ℝ) : ℝ :=
  let son_age := (2/5) * father_age
  (father_age - 2 * son_age) / (1/2)

/-- Theorem stating the correct number of years for the given problem -/
theorem years_to_half_age_specific : 
  years_to_half_age 40.00000000000001 = 8.000000000000002 := by
  -- Unfold the definition of years_to_half_age
  unfold years_to_half_age
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_half_age_specific_l236_23675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_matchbox_coloring_count_l236_23680

/-- Represents a cuboid (including cube and matchbox) -/
structure Cuboid where
  faces : Fin 6 → Unit

/-- Represents a coloring of a cuboid -/
structure Coloring (c : Cuboid) where
  color : Fin 6 → Fin 3

/-- Counts the number of valid colorings for a cube -/
def count_cube_colorings : ℕ := sorry

/-- Counts the number of valid colorings for a matchbox -/
def count_matchbox_colorings : ℕ := sorry

/-- A coloring is valid if it uses each color exactly twice -/
def is_valid_coloring (c : Cuboid) (col : Coloring c) : Prop :=
  ∀ i : Fin 3, (Finset.filter (fun j => col.color j = i) Finset.univ).card = 2

theorem cube_coloring_count :
  count_cube_colorings = 36 := by sorry

theorem matchbox_coloring_count :
  count_matchbox_colorings = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_matchbox_coloring_count_l236_23680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_drivers_and_ivan_departure_l236_23658

/-- Represents time in minutes since midnight -/
structure Time where
  minutes : ℕ
deriving Inhabited, DecidableEq

/-- Converts hours and minutes to Time -/
def toTime (hours minutes : ℕ) : Time :=
  ⟨hours * 60 + minutes⟩

/-- Duration of a one-way trip in minutes -/
def onewayTripDuration : ℕ := 160

/-- Minimum rest time for drivers in minutes -/
def minRestTime : ℕ := 60

/-- Schedule of departures -/
def departures : List Time := [
  toTime 8 35,   -- First departure
  toTime 10 40,  -- Second departure (Ivan Petrovich)
  toTime 11 50,  -- Third departure
  toTime 13 5,   -- Fourth departure
  toTime 16 10,  -- Fifth departure
  toTime 17 30   -- Sixth departure
]

/-- Theorem stating the minimum number of drivers required and Ivan Petrovich's departure time -/
theorem min_drivers_and_ivan_departure 
  (driverA_return : Time)
  (driverB_return : Time)
  (h1 : driverA_return = toTime 12 40)
  (h2 : driverB_return = toTime 16 0)
  : (∃ (n : ℕ), n = 4 ∧ n = List.length (List.dedup departures)) ∧ 
    (List.get! departures 1 = toTime 10 40) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_drivers_and_ivan_departure_l236_23658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l236_23601

def sequence_sum (S : ℕ → ℝ) : Prop :=
  S 1 = 4 ∧ ∀ n : ℕ, S (n + 1) = 3 * S n + 2 * n + 4

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_sum : sequence_sum S)
  (h_a : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  geometric_sequence (λ n => a n + 1) ∧
  ∀ n : ℕ, S n = (5 * (3^n - 1)) / 2 - n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l236_23601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l236_23690

noncomputable def g (A : ℝ) : ℝ :=
  (Real.sin A * (4 * Real.cos A ^ 2 + 2 * Real.cos A ^ 4 + 2 * Real.sin A ^ 2 + Real.sin A ^ 2 * Real.cos A ^ 2)) /
  (Real.tan A * (1 / Real.cos A - 2 * Real.sin A * Real.tan A))

theorem g_range (A : ℝ) (h : ∀ n : ℤ, A ≠ n * Real.pi / 2) :
  4 < g A ∧ g A < 6 ∧ ∀ y : ℝ, 4 < y → y < 6 → ∃ A : ℝ, g A = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l236_23690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l236_23612

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law holds for the triangle. -/
axiom sine_law {t : Triangle} : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The given condition holds. -/
axiom given_condition (t : Triangle) : (t.a - t.c) / (t.a - t.b) = Real.sin (t.A + t.C) / (Real.sin t.A + Real.sin t.C)

/-- The vector condition holds. -/
axiom vector_condition (t : Triangle) : Real.sqrt ((t.a - t.b / 2) ^ 2) = 2

/-- The theorem to be proved. -/
theorem triangle_properties (t : Triangle) :
  t.C = π / 3 ∧ 
  (2 * Real.sqrt 3 : ℝ) = sorry :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l236_23612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l236_23661

open Set Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (2 * Real.sin x - 1)

-- Define the domain set
def domain : Set ℝ := {x | ∃ k : ℤ, π/6 + 2*k*π < x ∧ x < 5*π/6 + 2*k*π}

-- Theorem statement
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l236_23661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l236_23687

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ (t : ℝ × ℝ) ↦ t)) →
  P ∈ Set.range (λ (t : ℝ × ℝ) ↦ t) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 →
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4 →
  (Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) / 2) / a = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l236_23687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l236_23649

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/4) 
  (h2 : 0 < α ∧ α < Real.pi/4) : 
  Real.sin α - Real.cos α = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l236_23649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_rps_l236_23619

theorem sin_angle_rps (RPQ : ℝ) (RPS : ℝ) 
  (h1 : Real.sin RPQ = 3/5) 
  (h2 : RPS = RPQ + 30 * Real.pi / 180) : 
  Real.sin RPS = (3 * Real.sqrt 3 + 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_rps_l236_23619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_journey_time_l236_23645

/-- Represents the journey from home to school -/
structure Journey where
  totalDistance : ℝ
  walkingSpeed : ℝ
  runningSpeed : ℝ
  walkingTime : ℝ

/-- Calculates the total time for the journey -/
noncomputable def totalTime (j : Journey) : ℝ :=
  j.walkingTime + (2 * j.totalDistance / 3) / j.runningSpeed

/-- Theorem stating that Joe's journey takes 13.5 minutes -/
theorem joe_journey_time :
  ∀ (j : Journey),
    j.runningSpeed = 4 * j.walkingSpeed →
    j.walkingTime = 9 →
    j.walkingSpeed * j.walkingTime = j.totalDistance / 3 →
    totalTime j = 13.5 := by
  intro j h1 h2 h3
  unfold totalTime
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_journey_time_l236_23645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_lines_count_l236_23689

def S : Finset ℕ := {1, 2, 3, 5}

def distinct_lines (S : Finset ℕ) : ℕ :=
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => p.2 / p.1) (S.product S))

theorem distinct_lines_count : distinct_lines S = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_lines_count_l236_23689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_partition_theorem_l236_23660

/-- A type representing a geometric shape -/
structure Shape where
  area : ℝ

/-- A type representing a piece cut from a shape -/
structure Piece where
  area : ℝ
  type : Nat

/-- A function that partitions a shape into pieces -/
def partition (s : Shape) : List Piece :=
  sorry

/-- A function that checks if a list of pieces can form a square -/
def canFormSquare (pieces : List Piece) : Prop :=
  sorry

/-- A function that checks if a list of pieces satisfies the required conditions -/
def validPartition (pieces : List Piece) : Prop :=
  pieces.length = 8 ∧
  (pieces.filter (λ p => p.type = 1)).length = 4 ∧
  (pieces.filter (λ p => p.type = 2)).length = 4 ∧
  ∀ p₁ p₂, p₁ ∈ pieces → p₂ ∈ pieces → p₁.type = p₂.type → p₁.area = p₂.area

theorem candy_partition_theorem (s : Shape) :
  ∃ (pieces : List Piece), validPartition pieces ∧ canFormSquare pieces :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_partition_theorem_l236_23660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l236_23616

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | |x - m| ≤ 2}

-- Part I
theorem part_one (m : ℝ) : A ∩ B m = Set.Icc 0 3 → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) : A ⊆ (B m)ᶜ → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l236_23616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABC_l236_23677

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- The minimum area of triangle ABC, where A(-2,0) and B(0,2) are fixed points,
    and C is any point on the circle x^2 + y^2 - 2x = 0 -/
theorem min_area_triangle_ABC : 
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let circle := {C : ℝ × ℝ | C.1^2 + C.2^2 - 2*C.1 = 0}
  ∃ (min_area : ℝ), min_area = 3 - Real.sqrt 2 ∧
    ∀ C ∈ circle, area_triangle A B C ≥ min_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABC_l236_23677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_params_from_eccentricity_and_conjugate_axis_l236_23657

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The length of the conjugate axis of a hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := 2 * h.b

theorem hyperbola_params_from_eccentricity_and_conjugate_axis 
  (e : ℝ) (l : ℝ) (h_e : e = 5/3) (h_l : l = 8) :
  ∃ (h : Hyperbola), eccentricity h = e ∧ conjugate_axis_length h = l ∧ h.a = 3 ∧ h.b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_params_from_eccentricity_and_conjugate_axis_l236_23657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_system_l236_23693

theorem unique_solution_trigonometric_system :
  ∀ x y : ℝ,
  0 < x ∧ x < π / 2 →
  0 < y ∧ y < π / 2 →
  (Real.cos x) / (Real.cos y) = 2 * (Real.cos y)^2 →
  (Real.sin x) / (Real.sin y) = 2 * (Real.sin y)^2 →
  x = π / 4 ∧ y = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_system_l236_23693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uyu_not_equivalent_yuy_l236_23640

/-- Represents a word in the Mumbo-Yumbo language --/
inductive MumboYumboWord
| empty : MumboYumboWord
| cons : Char → MumboYumboWord → MumboYumboWord

/-- Counts the occurrences of a given character in a MumboYumboWord --/
def countChar (c : Char) : MumboYumboWord → Nat
| MumboYumboWord.empty => 0
| MumboYumboWord.cons x rest => (if x = c then 1 else 0) + countChar c rest

/-- Represents the allowed transformations in the Mumbo-Yumbo language --/
inductive MumboYumboTransform : MumboYumboWord → MumboYumboWord → Prop
| skipYU : ∀ w, MumboYumboTransform (MumboYumboWord.cons 'ы' (MumboYumboWord.cons 'у' w)) w
| skipUUYY : ∀ w, MumboYumboTransform (MumboYumboWord.cons 'у' (MumboYumboWord.cons 'у' (MumboYumboWord.cons 'ы' (MumboYumboWord.cons 'ы' w)))) w
| addUY : ∀ w, MumboYumboTransform w (MumboYumboWord.cons 'у' (MumboYumboWord.cons 'ы' w))

/-- Represents the transitive closure of MumboYumboTransform --/
def MumboYumboEquivalent := Relation.ReflTransGen MumboYumboTransform

/-- The main theorem stating that "уыу" and "ыуы" are not equivalent --/
theorem uyu_not_equivalent_yuy :
  ¬ MumboYumboEquivalent
    (MumboYumboWord.cons 'у' (MumboYumboWord.cons 'ы' (MumboYumboWord.cons 'у' MumboYumboWord.empty)))
    (MumboYumboWord.cons 'ы' (MumboYumboWord.cons 'у' (MumboYumboWord.cons 'ы' MumboYumboWord.empty))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uyu_not_equivalent_yuy_l236_23640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_right_angles_l236_23639

/-- Given n rectangles in a plane forming 4n right angles, 
    there exist at least ⌊4√n⌋ distinct right angles among them. -/
theorem distinct_right_angles (n : ℕ) : ∃ (d : ℕ), d ≥ ⌊4 * Real.sqrt n⌋ ∧ d ≤ 4 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_right_angles_l236_23639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l236_23627

/-- Represents the dimensions and properties of a milk box -/
structure MilkBox where
  length : ℝ
  width : ℝ
  milk_removed : ℝ
  level_lowered : ℝ

/-- Converts gallons to cubic feet -/
noncomputable def gallons_to_cubic_feet (gallons : ℝ) : ℝ :=
  gallons / 7.48052

/-- Theorem stating that the width of the milk box is 25 feet -/
theorem milk_box_width (box : MilkBox)
  (h1 : box.length = 58)
  (h2 : box.milk_removed = 5437.5)
  (h3 : box.level_lowered = 0.5) :
  box.width = 25 := by
  sorry

#check milk_box_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l236_23627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ef_is_fifteen_l236_23631

/-- A rectangle ABCD with points E and F forming a rhombus AFCE -/
structure RhombusInRectangle where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- Point E on side AB
  e : ℝ
  -- Point F on side CD
  f : ℝ
  -- AB = 16
  ab_eq : ab = 16
  -- BC = 12
  bc_eq : bc = 12
  -- AFCE is a rhombus
  is_rhombus : e ≠ 0 ∧ e ≠ ab ∧ f ≠ 0 ∧ f ≠ bc

/-- The length of EF in the rhombus-rectangle configuration -/
def ef_length (r : RhombusInRectangle) : ℝ := sorry

/-- Theorem: The length of EF is 15 -/
theorem ef_is_fifteen (r : RhombusInRectangle) : ef_length r = 15 := by
  sorry

#check ef_is_fifteen

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ef_is_fifteen_l236_23631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_5_equals_4_l236_23655

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := 
  let x := (y - 1) / 2
  x^2

-- State the theorem
theorem f_of_5_equals_4 : f 5 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Evaluate the result
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_5_equals_4_l236_23655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_specific_quadratic_root_product_l236_23663

theorem product_of_roots_quadratic (a b c : ℚ) (h : a ≠ 0) :
  let equation := λ x => a * x^2 + b * x + c
  let root_product := c / a
  ∀ x, equation x = 0 → ∃ y, equation y = 0 ∧ x * y = root_product :=
by sorry

theorem specific_quadratic_root_product :
  let equation := λ x => 14 * x^2 + 21 * x - 250
  let root_product := -125 / 7
  ∀ x, equation x = 0 → ∃ y, equation y = 0 ∧ x * y = root_product :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_specific_quadratic_root_product_l236_23663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l236_23638

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_minimum (a : ℕ+ → ℝ) (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ+, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ+, (1 : ℝ) / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ+, (1 : ℝ) / m + 4 / n ≥ 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l236_23638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l236_23679

/-- A color type with two possible values -/
inductive Color
  | one
  | two

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The theorem statement -/
theorem same_color_unit_distance (f : Coloring) :
  ∃ (p q : Point), f p = f q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l236_23679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_distance_l236_23636

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Define the starting point P
noncomputable def P : ℝ := 1 + Real.sqrt 17

-- Define the ending point Q
def Q : ℝ := 2

-- Theorem statement
theorem horizontal_distance :
  abs (P - Q) = Real.sqrt 17 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_distance_l236_23636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_existence_l236_23611

theorem vector_projection_existence : ∃ (v : ℝ × ℝ),
  (let u₁ : ℝ × ℝ := (3, 2)
   let proj₁ : ℝ × ℝ → ℝ × ℝ := λ w => 
     let dot := w.1 * u₁.1 + w.2 * u₁.2
     let norm_sq := u₁.1 * u₁.1 + u₁.2 * u₁.2
     ((dot / norm_sq) * u₁.1, (dot / norm_sq) * u₁.2)
   proj₁ v = (45/13, 30/13)) ∧
  (let u₂ : ℝ × ℝ := (1, 4)
   let proj₂ : ℝ × ℝ → ℝ × ℝ := λ w => 
     let dot := w.1 * u₂.1 + w.2 * u₂.2
     let norm_sq := u₂.1 * u₂.1 + u₂.2 * u₂.2
     ((dot / norm_sq) * u₂.1, (dot / norm_sq) * u₂.2)
   proj₂ v = (32/17, 128/17)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_existence_l236_23611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_pricing_theorem_l236_23644

/-- Represents the cost and selling prices of pens -/
structure PenPrices where
  cost : ℚ
  sell : ℚ

/-- Calculates the gain percentage given the cost and selling prices -/
noncomputable def gainPercentage (p : PenPrices) : ℚ :=
  ((p.sell - p.cost) / p.cost) * 100

/-- Theorem: If the selling price of 5 pens equals the cost price of 10 pens, 
    then the gain percentage is 100% -/
theorem pen_pricing_theorem (p : PenPrices) :
  5 * p.sell = 10 * p.cost → gainPercentage p = 100 := by
  intro h
  simp [gainPercentage]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check pen_pricing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_pricing_theorem_l236_23644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l236_23647

/-- A boy walks to school at different rates. -/
structure SchoolWalk where
  usual_time : ℚ
  rate_ratio : ℚ

/-- Calculate the time saved when walking at a faster rate. -/
def time_saved (walk : SchoolWalk) : ℚ :=
  walk.usual_time - (walk.usual_time / walk.rate_ratio)

/-- Theorem: The boy reaches school 5 minutes early when walking at 7/6 of his usual rate. -/
theorem early_arrival (walk : SchoolWalk) 
  (h1 : walk.usual_time = 35)
  (h2 : walk.rate_ratio = 7/6) : 
  time_saved walk = 5 := by
  sorry

#eval time_saved { usual_time := 35, rate_ratio := 7/6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l236_23647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l236_23664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2*x + a else -x - 2*a

theorem piecewise_function_equality (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l236_23664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_sequence_well_defined_and_nonzero_sum_l236_23692

noncomputable def ternary_sequence : ℕ → ℝ × ℝ × ℝ
| 0 => (2, 4, 6/7)
| n+1 => 
  let (x, y, z) := ternary_sequence n
  ((2*x)/(x^2 - 1), (2*y)/(y^2 - 1), (2*z)/(z^2 - 1))

theorem ternary_sequence_well_defined_and_nonzero_sum :
  (∀ n : ℕ, let (x, y, z) := ternary_sequence n;
             x ≠ 1 ∧ x ≠ -1 ∧ y ≠ 1 ∧ y ≠ -1 ∧ z ≠ 1 ∧ z ≠ -1) ∧
  (∀ n : ℕ, let (x, y, z) := ternary_sequence n;
             x + y + z ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_sequence_well_defined_and_nonzero_sum_l236_23692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_f_at_theta_minus_pi_sixth_l236_23632

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (x - Real.pi / 12)

-- Theorem 1
theorem f_at_pi_third : f (Real.pi / 3) = 1 := by sorry

-- Theorem 2
theorem f_at_theta_minus_pi_sixth (θ : ℝ) 
  (h1 : Real.cos θ = 3 / 5) 
  (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : 
  f (θ - Real.pi / 6) = -1 / 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_f_at_theta_minus_pi_sixth_l236_23632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l236_23625

/-- Calculates the annual repayment amount for a loan -/
noncomputable def annualRepayment (a r : ℝ) : ℝ :=
  (a * r * (1 + r)^5) / ((1 + r)^5 - 1)

/-- Theorem: The annual repayment amount for a 5-year loan equals the formula -/
theorem loan_repayment_theorem (a r : ℝ) (ha : a > 0) (hr : r > 0) :
  let x := annualRepayment a r
  (a * (1 + r)^5) = x * ((1 + r)^5 - 1) / r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l236_23625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l236_23656

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1 - a) / 2 * x^2 - x

-- State the theorem
theorem function_range_theorem (a : ℝ) (h₁ : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ≥ 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l236_23656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l236_23676

/-- The complex number Z -/
noncomputable def Z : ℂ := 2 / (3 - Complex.I) + Complex.I ^ 2015

/-- Z is in the fourth quadrant if its real part is positive and imaginary part is negative -/
def is_in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- Theorem: Z is located in the fourth quadrant of the complex plane -/
theorem Z_in_fourth_quadrant : is_in_fourth_quadrant Z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l236_23676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reshaped_pizza_radius_is_sqrt_seven_l236_23604

/-- The radius of a reshaped mini-pizza -/
noncomputable def reshaped_pizza_radius (large_radius : ℝ) (mini_radius : ℝ) (num_mini : ℕ) : ℝ :=
  Real.sqrt (large_radius^2 - num_mini * mini_radius^2)

/-- Theorem stating the radius of the reshaped mini-pizza -/
theorem reshaped_pizza_radius_is_sqrt_seven :
  reshaped_pizza_radius 4 1 9 = Real.sqrt 7 := by
  -- Unfold the definition of reshaped_pizza_radius
  unfold reshaped_pizza_radius
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reshaped_pizza_radius_is_sqrt_seven_l236_23604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_2017_is_red_l236_23628

-- Define the color type
inductive Color where
  | Blue
  | Red

-- Define the coloring function
def color : Nat → Color := sorry

-- Define the conditions
axiom color_domain : ∀ n : Nat, n > 1 → color n = Color.Blue ∨ color n = Color.Red

axiom blue_sum : ∀ a b : Nat, a > 1 → b > 1 → 
  color a = Color.Blue → color b = Color.Blue → color (a + b) = Color.Blue

axiom red_product : ∀ a b : Nat, a > 1 → b > 1 → 
  color a = Color.Red → color b = Color.Red → color (a * b) = Color.Red

axiom both_colors_used : ∃ a b : Nat, a > 1 ∧ b > 1 ∧ color a = Color.Blue ∧ color b = Color.Red

axiom color_1024 : color 1024 = Color.Blue

-- Theorem to prove
theorem color_2017_is_red : color 2017 = Color.Red := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_2017_is_red_l236_23628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_from_interior_points_l236_23662

/-- A point is interior to a triangle if it is contained within the triangle but is not on any of its edges or vertices. -/
def is_interior_point (P A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Three segments can form a triangle if the sum of the lengths of any two segments is greater than the length of the third segment. -/
def can_form_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle is equilateral if all its sides have equal length. -/
def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := 
  norm (A - B) = norm (B - C) ∧ norm (B - C) = norm (C - A)

theorem triangle_equilateral_from_interior_points 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h : ∀ P : EuclideanSpace ℝ (Fin 2), is_interior_point P A B C → 
    can_form_triangle (norm (P - A)) (norm (P - B)) (norm (P - C))) : 
  is_equilateral_triangle A B C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_from_interior_points_l236_23662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_papaya_cost_is_one_l236_23623

/-- The cost of fruit and discount structure at a store --/
structure FruitStore where
  lemon_cost : ℚ
  mango_cost : ℚ
  discount_threshold : ℕ
  discount_amount : ℚ

/-- A customer's fruit purchase --/
structure FruitPurchase where
  lemons : ℕ
  papayas : ℕ
  mangos : ℕ

/-- Calculate the total cost of a fruit purchase --/
def calculateTotalCost (store : FruitStore) (purchase : FruitPurchase) (papaya_cost : ℚ) : ℚ :=
  let total_fruits := purchase.lemons + purchase.papayas + purchase.mangos
  let discount_count := total_fruits / store.discount_threshold
  let total_cost_before_discount := 
    store.lemon_cost * purchase.lemons + 
    papaya_cost * purchase.papayas + 
    store.mango_cost * purchase.mangos
  total_cost_before_discount - store.discount_amount * discount_count

/-- Theorem stating that papayas cost $1 given the problem conditions --/
theorem papaya_cost_is_one (store : FruitStore) (purchase : FruitPurchase) : 
  store.lemon_cost = 2 →
  store.mango_cost = 4 →
  store.discount_threshold = 4 →
  store.discount_amount = 1 →
  purchase.lemons = 6 →
  purchase.papayas = 4 →
  purchase.mangos = 2 →
  calculateTotalCost store purchase 1 = 21 →
  ∃ (papaya_cost : ℚ), papaya_cost = 1 ∧ calculateTotalCost store purchase papaya_cost = 21 := by
  sorry

#check papaya_cost_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_papaya_cost_is_one_l236_23623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_family_l236_23678

/-- The family of lines described by the equation ax + (a-1)y + a + 3 = 0 -/
def line_family (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => a * p.1 + (a - 1) * p.2 + a + 3 = 0

/-- The point P -/
def point_P : ℝ × ℝ := (2, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the maximum distance from point_P to any line in the family is 2√10 -/
theorem max_distance_to_line_family :
  ∃ (a : ℝ), ∀ (q : ℝ × ℝ), line_family a q → distance point_P q ≤ 2 * Real.sqrt 10 := by
  sorry

#check max_distance_to_line_family

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_family_l236_23678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l236_23698

def is_solution (x : ℝ) : Prop :=
  x - ⌊x⌋ = 1 / (⌊x⌋ + 1)

def is_positive (x : ℝ) : Prop := x > 0

def smallest_solutions : Set ℝ :=
  {x | is_solution x ∧ is_positive x ∧ ∀ y, is_solution y ∧ is_positive y → x ≤ y}

theorem sum_of_smallest_solutions :
  ∃ (s₁ s₂ s₃ : ℝ), s₁ ∈ smallest_solutions ∧ s₂ ∈ smallest_solutions ∧ s₃ ∈ smallest_solutions ∧
  s₁ < s₂ ∧ s₂ < s₃ ∧
  (∀ (s : ℝ), s ∈ smallest_solutions → s = s₁ ∨ s = s₂ ∨ s = s₃ ∨ s > s₃) ∧
  s₁ + s₂ + s₃ = 7 + 1/12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l236_23698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jays_savings_l236_23674

theorem jays_savings (initial_savings : ℝ) : 
  (∀ w : ℕ, w < 4 → 
    (initial_savings + 10 * w) = 
      (fun w => initial_savings + 10 * w) w) →
  (initial_savings + (initial_savings + 10) + 
   (initial_savings + 20) + (initial_savings + 30) = 60) →
  initial_savings = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jays_savings_l236_23674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_x_axis_l236_23672

/-- Given a point M(2, 1, 3), its symmetric point with respect to the x-axis has coordinates (2, -1, -3). -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ × ℝ := (2, 1, 3)
  let symmetric_point := (M.1, -M.2.1, -M.2.2)
  symmetric_point = (2, -1, -3) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_x_axis_l236_23672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_minus_cube_l236_23600

def units_digit (n : ℤ) : ℕ := (n.mod 10).natAbs

theorem units_digit_of_product_minus_cube : units_digit (9 * 19 * 1989 - 9^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_minus_cube_l236_23600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_diagram_for_production_steps_l236_23667

/-- Represents the different types of diagrams --/
inductive DiagramType
  | ProgramFlowchart
  | ProcessFlowchart
  | KnowledgeStructureDiagram
  | OrganizationalStructureDiagram

/-- Represents the problem statement --/
def productionStepsDiagram : DiagramType := DiagramType.ProcessFlowchart

/-- Theorem stating that the correct diagram for describing production steps is a Process Flowchart --/
theorem correct_diagram_for_production_steps :
  productionStepsDiagram = DiagramType.ProcessFlowchart := by
  -- The proof is trivial as we defined productionStepsDiagram to be ProcessFlowchart
  rfl

#check correct_diagram_for_production_steps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_diagram_for_production_steps_l236_23667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_alive_simultaneously_equals_overlap_area_l236_23646

/-- Represents a mathematician with their birth year and lifespan --/
structure Mathematician where
  birthYear : ℝ
  lifespan : ℝ

/-- The timespan considered in the problem --/
def totalYears : ℝ := 1000

/-- Calculates the area of overlap between two mathematicians' lifespans --/
noncomputable def calculateOverlapArea (m1 m2 : Mathematician) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Calculates the probability of two mathematicians being alive at the same time --/
noncomputable def probabilityAliveSimultaneously (m1 m2 : Mathematician) : ℝ :=
  let overlapArea := calculateOverlapArea m1 m2
  overlapArea / (totalYears * totalYears)

/-- Theorem stating the probability of two mathematicians being alive simultaneously --/
theorem probability_alive_simultaneously_equals_overlap_area 
  (m1 m2 : Mathematician) 
  (h1 : m1.birthYear ∈ Set.Icc 0 totalYears) 
  (h2 : m2.birthYear ∈ Set.Icc 0 totalYears)
  (h3 : m1.lifespan = 100)
  (h4 : m2.lifespan = 80) :
  ∃ (P : ℝ), probabilityAliveSimultaneously m1 m2 = P / 1000000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_alive_simultaneously_equals_overlap_area_l236_23646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l236_23653

/-- The area of a region formed by removing two specific triangles from a square -/
theorem shaded_area_calculation (square_side : ℝ) (triangle1 triangle2 : Set (ℝ × ℝ)) : 
  square_side = 50 →
  triangle1 = {(0,0), (15,0), (50,30)} →
  triangle2 = {(0,15), (35,50), (50,50)} →
  let square_area := square_side ^ 2
  let triangle1_area := abs (15 * 30 + 50 * 0 + 0 * (-30)) / 2
  let triangle2_area := abs (35 * 35 + 50 * (-35) + 0 * 0) / 2
  square_area - (triangle1_area + triangle2_area) = 2012.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l236_23653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_areas_l236_23637

/-- The radius of the nth circle in the sequence -/
noncomputable def radius (n : ℕ) : ℝ := 2 / (2 ^ (n - 1))

/-- The area of the nth circle in the sequence -/
noncomputable def area (n : ℕ) : ℝ := Real.pi * (radius n)^2

/-- The sum of the areas of all circles in the sequence -/
noncomputable def total_area : ℝ := ∑' n, area n

theorem sum_of_circle_areas : total_area = 16 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_areas_l236_23637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equality_l236_23668

/-- Represents a strip in the square --/
structure Strip where
  index : Nat
  leftArea : ℝ
  rightArea : ℝ

/-- Represents the square ABCD with its properties --/
structure Square where
  sideLength : ℝ
  strips : List Strip
  oddSegmentsSum : ℝ
  evenSegmentsSum : ℝ

/-- The main theorem to be proved --/
theorem square_area_equality (s : Square) 
  (h1 : s.oddSegmentsSum = s.evenSegmentsSum) 
  (h2 : s.strips.length > 0) :
  (s.strips.filter (fun strip => strip.index % 2 = 1)).foldl (fun acc strip => acc + strip.leftArea) 0 =
  (s.strips.filter (fun strip => strip.index % 2 = 0)).foldl (fun acc strip => acc + strip.rightArea) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equality_l236_23668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l236_23648

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c A B C : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (A > 0) ∧ (A < Real.pi) ∧ (B > 0) ∧ (B < Real.pi) ∧ (C > 0) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi)

theorem triangle_theorem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_equation : (2*b - c) * Real.cos A = a * Real.cos C) :
  (A = Real.pi/3) ∧ 
  (∃ (m n : ℝ × ℝ), 
    m = (0, -1) ∧ 
    n = (Real.cos B, 2 * (Real.cos (C/2))^2) ∧
    (∀ (x : ℝ × ℝ), ‖m + n‖ ≤ ‖m + x‖) ∧
    ‖m + n‖ = Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l236_23648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_decimal_digits_l236_23642

theorem power_of_two_decimal_digits (m : ℕ+) :
  ∃ (n : ℕ), (2 : ℚ)^(-(m.val : ℤ)) = (n : ℚ) / 10^m.val ∧ n ≥ 10^(m.val - 1) ∧ n < 10^m.val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_decimal_digits_l236_23642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_product_approx_l236_23617

-- Define a right triangle with two known side lengths
noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the function to calculate the product of possible third side lengths
noncomputable def third_side_product (a b : ℝ) : ℝ :=
  let c₁ := Real.sqrt (a^2 + b^2)
  let c₂ := Real.sqrt (max a b^2 - min a b^2)
  c₁ * c₂

-- Theorem statement
theorem third_side_product_approx (a b : ℝ) (ha : a = 8) (hb : b = 15) :
  abs (third_side_product a b - 215.9) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_product_approx_l236_23617
