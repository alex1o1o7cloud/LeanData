import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_AOBD_l884_88486

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the vertex
def vertex : ℝ × ℝ := (0, 0)

-- Define the focus
def focus : ℝ × ℝ := (0, 1/4)

-- Define the chord length
def chord_length : ℝ := 2

-- Define the theorem
theorem area_of_quadrilateral_AOBD :
  ∀ (A B : ℝ × ℝ),
  let (xA, yA) := A
  let (xB, yB) := B
  -- A and B are on the parabola
  (yA = parabola xA) →
  (yB = parabola xB) →
  -- AB passes through the focus
  (∃ t : ℝ, (1 - t) • A + t • B = focus) →
  -- AB has length 2
  (Real.sqrt ((xB - xA)^2 + (yB - yA)^2) = chord_length) →
  -- D is on the y-axis
  let D := (0, (yA + yB)/2 + 1/2)
  -- The area of AOBD is 5√2/8
  (1/2 * chord_length * (D.2 - vertex.2) * Real.sqrt 2 / 2 = 5 * Real.sqrt 2 / 8) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_AOBD_l884_88486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l884_88440

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 5}

-- Define set N
def N : Set ℝ := {y | ∃ x : ℝ, x ≥ -2 ∧ y = Real.sqrt (x + 2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 0 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l884_88440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l884_88425

theorem money_distribution (total faruk vasim ranjith : ℝ) : 
  faruk + vasim + ranjith = total →
  faruk / 3 = vasim / 5 →
  faruk / 3 = ranjith / 9 →
  ranjith - faruk = 1800 →
  vasim = 1500 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l884_88425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_2008_coordinates_l884_88407

def tree_x : ℕ → ℤ
  | 0 => 1
  | k + 1 => tree_x k + 1 - 5 * (Int.floor ((k : ℚ) / 5)) + 5 * (Int.floor (((k - 1) : ℚ) / 5))

def tree_y : ℕ → ℤ
  | 0 => 1
  | k + 1 => tree_y k + (Int.floor ((k : ℚ) / 5)) - (Int.floor (((k - 1) : ℚ) / 5))

theorem tree_2008_coordinates :
  (tree_x 2008, tree_y 2008) = (3, 402) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_2008_coordinates_l884_88407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l884_88426

theorem new_average_age (initial_average : ℝ) (initial_group_size : ℕ) (new_person_age : ℝ) : 
  initial_average = 15 ∧ 
  initial_group_size = 10 ∧ 
  new_person_age = 37 → 
  (initial_average * (initial_group_size : ℝ) + new_person_age) / ((initial_group_size + 1) : ℝ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l884_88426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_sum_l884_88494

theorem tan_double_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 10) 
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 12) : 
  Real.tan (2 * (x + y)) = -120 / 3599 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_sum_l884_88494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l884_88461

/-- K's speed in miles per hour -/
noncomputable def x : ℝ := sorry

/-- M's speed in miles per hour -/
noncomputable def m_speed : ℝ := x - 1

/-- K's time to travel 60 miles -/
noncomputable def k_time : ℝ := 60 / x

/-- M's time to travel 60 miles -/
noncomputable def m_time : ℝ := 60 / m_speed

/-- The theorem stating that K's time to travel 60 miles is 6 hours -/
theorem k_travel_time :
  (m_time - k_time = 1) →  -- K takes 1 hour less time than M
  k_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l884_88461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l884_88439

noncomputable section

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := (1/2) * x^2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f x + g a x

-- Define the theorem
theorem problem_solution (a : ℝ) :
  -- Part 1
  (∀ x y : ℝ, y = f x - g a x → (6 * 1 - 2 * y - 5 = 0 → a = -2)) ∧
  -- Part 2
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (h a x₁ - h a x₂) / (x₁ - x₂) > 2 → a ≥ 1) ∧
  -- Part 3
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ 
    (deriv f x₀) + 1 / (deriv f x₀) < g a x₀ - (deriv (g a) x₀) → 
    a < -2 ∨ a > (Real.exp 2 + 1) / (Real.exp 1 - 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l884_88439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_share_for_180_members_l884_88427

/-- Represents a noble family with land inheritance rules --/
structure NobleFamily where
  total_members : ℕ
  founder_land : ℚ

/-- The smallest possible share of the original plot for any family member --/
def smallest_share (family : NobleFamily) : ℚ :=
  1 / (2 * 3^59)

/-- Theorem stating the smallest possible share for a family of 180 members --/
theorem smallest_share_for_180_members (family : NobleFamily) 
  (h_members : family.total_members = 180) :
  smallest_share family = 1 / (2 * 3^59) := by
  -- The proof goes here
  sorry

#eval smallest_share ⟨180, 1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_share_for_180_members_l884_88427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_line_parallel_implies_planes_parallel_two_intersecting_lines_parallel_implies_planes_parallel_l884_88419

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define a line in 3D space
structure Line where
  direction : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define parallelism between a line and a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop :=
  let (dx, dy, dz) := l.direction
  let (nx, ny, nz) := p.normal
  dx * nx + dy * ny + dz * nz = 0

-- Define parallelism between two planes
def planes_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
  let (nx1, ny1, nz1) := p1.normal
  let (nx2, ny2, nz2) := p2.normal
  (nx1, ny1, nz1) = (k * nx2, k * ny2, k * nz2)

-- Theorem 1
theorem every_line_parallel_implies_planes_parallel (p1 p2 : Plane) :
  (∀ (l : Line), (l.point = p1.point) → line_parallel_to_plane l p2) → planes_parallel p1 p2 :=
sorry

-- Theorem 2
theorem two_intersecting_lines_parallel_implies_planes_parallel (p1 p2 : Plane) (l1 l2 : Line) :
  (l1.point = p1.point) ∧ (l2.point = p1.point) ∧
  (∃ (x : ℝ × ℝ × ℝ), x = l1.point ∧ x = l2.point) ∧
  line_parallel_to_plane l1 p2 ∧ line_parallel_to_plane l2 p2 →
  planes_parallel p1 p2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_line_parallel_implies_planes_parallel_two_intersecting_lines_parallel_implies_planes_parallel_l884_88419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_complementary_angles_CD_fixed_point_l884_88423

-- Define the plane
variable (P : ℝ × ℝ)

-- Define given points and line
def F : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1, 0)
def l (x : ℝ) : Prop := x = -1

-- Define Q as the foot of perpendicular from P to l
def Q (P : ℝ × ℝ) : ℝ × ℝ := (-1, P.2)

-- Define the dot product condition
def dot_product_condition (P : ℝ × ℝ) : Prop :=
  let QP := (P.1 - (Q P).1, P.2 - (Q P).2)
  let QF := (F.1 - (Q P).1, F.2 - (Q P).2)
  let FP := (P.1 - F.1, P.2 - F.2)
  let FQ := ((Q P).1 - F.1, (Q P).2 - F.2)
  QP.1 * QF.1 + QP.2 * QF.2 = FP.1 * FQ.1 + FP.2 * FQ.2

-- Define the trajectory G
def G (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define A and B as intersection points of a line through F and G
variable (A B : ℝ × ℝ)

-- Define C and D
variable (C D : ℝ × ℝ)

-- AB is not perpendicular to x-axis
axiom AB_not_perpendicular : (B.2 - A.2) ≠ 0

-- Theorems to prove
theorem trajectory_equation (P : ℝ × ℝ) (h : dot_product_condition P) : G P :=
  sorry

theorem complementary_angles :
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  k_AB * k_CD = -1 :=
sorry

theorem CD_fixed_point : D.1 - C.1 ≠ 0 → C.2 / (C.1 - D.1) * 1 + C.2 * D.1 / (C.1 - D.1) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_complementary_angles_CD_fixed_point_l884_88423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_f_domain_min_value_constraint_min_value_achievable_l884_88421

/-- The function f(x) defined on real numbers --/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x + 5| - |x - 1| + t)

/-- Theorem stating the minimum value of t for f to be defined on all real numbers --/
theorem min_t_for_f_domain : ∀ x : ℝ, (∃ y : ℝ, f 6 x = y) ∧ (∀ t < 6, ¬∃ y : ℝ, f t x = y) := by sorry

/-- Theorem stating the minimum value of 4a+5b given the constraint --/
theorem min_value_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 / (a + 2*b) + 1 / (2*a + b) = 6) → 4*a + 5*b ≥ 3/2 := by sorry

/-- Theorem stating that 3/2 is achievable --/
theorem min_value_achievable : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 / (a + 2*b) + 1 / (2*a + b) = 6 ∧ 4*a + 5*b = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_f_domain_min_value_constraint_min_value_achievable_l884_88421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l884_88456

/-- The set of points (x,y) satisfying the inequality ||x|-2|+|y-3| ≤ 3 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs (abs p.1 - 2) + abs (p.2 - 3)) ≤ 3}

/-- The area of a set in ℝ² -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of S is 18 -/
theorem area_of_S : area S = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l884_88456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_quantity_l884_88460

/-- Represents the initial quantity of milk in container A -/
def A : ℝ := 1200

/-- Represents the quantity of milk in container B after initial pour -/
noncomputable def B : ℝ := (3/8) * A

/-- Represents the quantity of milk in container C after initial pour -/
noncomputable def C : ℝ := (5/8) * A

/-- Theorem stating that the initial quantity of milk in container A is 1200 liters -/
theorem initial_milk_quantity : A = 1200 :=
  by
    -- Define the equation representing the transfer of 150 liters
    have h1 : B + 150 = C - 150 := by sorry
    -- Substitute the expressions for B and C
    have h2 : (3/8) * A + 150 = (5/8) * A - 150 := by sorry
    -- Solve for A
    have h3 : 300 = (1/4) * A := by sorry
    -- Conclude that A = 1200
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_quantity_l884_88460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_spent_l884_88466

/-- Represents the time spent on research and expedition for an artifact -/
structure ArtifactTime where
  research : ℝ
  expedition : ℝ

/-- Calculates the total time spent on an artifact -/
def totalTime (a : ArtifactTime) : ℝ := a.research + a.expedition

/-- Represents the time spent on all seven artifacts -/
def allArtifacts : List ArtifactTime := [
  { research := 6, expedition := 24 },
  { research := 18, expedition := 48 },
  { research := 9, expedition := 48 },
  { research := 6, expedition := 72 },
  { research := 7.5, expedition := 45 },
  { research := 15, expedition := 120 },
  { research := 24, expedition := 209.25 }
]

/-- Theorem stating the total time spent on all artifacts -/
theorem total_time_spent : 
  (allArtifacts.map totalTime).sum = 651.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_spent_l884_88466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_count_l884_88412

theorem distinct_solutions_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ (x = x^2 + y^2 ∧ y = 2*x*y)) ∧ 
    Finset.card S = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_count_l884_88412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l884_88485

theorem discount_calculation (initial_discount : ℝ) (secondary_discount : ℝ) 
  (claimed_total_discount : ℝ) (h1 : initial_discount = 0.30) 
  (h2 : secondary_discount = 0.20) (h3 : claimed_total_discount = 0.50) : 
  (1 - (1 - initial_discount) * (1 - secondary_discount) = 0.44) ∧
  (claimed_total_discount - (1 - (1 - initial_discount) * (1 - secondary_discount)) = 0.06) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l884_88485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l884_88424

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (M : ℝ), M = 13 ∧ 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → y x ≤ M) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ y x = M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l884_88424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_plus_pi_3_l884_88453

theorem cos_2theta_plus_pi_3 (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) 
  (h2 : 1/Real.sin θ + 1/Real.cos θ = 2 * Real.sqrt 2) : 
  Real.cos (2*θ + π/3) = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_plus_pi_3_l884_88453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l884_88452

/-- Given a circle with center O and a chord AB intersected by a diameter at point E,
    prove that the radius is 11 when:
    1. The length of chord AB is 18
    2. The distance OE is 7
    3. E divides AB in the ratio 2:1 -/
theorem circle_radius_proof (O E A B : ℝ × ℝ) (R : ℝ) :
  let d := (fun p q : ℝ × ℝ ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d A B = 18 →
  d O E = 7 →
  d A E = 2 * d E B →
  d A B = d A E + d E B →
  (∀ P, d O P = R ↔ P ∈ {Q | d O Q = R}) →
  R = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l884_88452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_high_octane_amount_l884_88403

/-- Calculates the amount of high octane in a fuel mixture given the specified conditions. -/
theorem high_octane_amount (regular_octane : ℝ) (cost_ratio : ℝ) (cost_fraction : ℝ) : ℝ :=
  let high_octane := regular_octane / 4
  by
    have h1 : regular_octane = 4545 := by sorry
    have h2 : cost_ratio = 3 := by sorry
    have h3 : cost_fraction = 3 / 7 := by sorry
    have h4 : high_octane = 1136.25 := by sorry
    exact high_octane

-- The following line is commented out as it's not necessary for the theorem proof
-- #eval high_octane_amount 4545 3 (3/7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_high_octane_amount_l884_88403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_teacher_age_l884_88473

theorem new_teacher_age (initial_count : Nat) (initial_avg : Nat) (new_avg : Nat) (new_teacher_age : Nat) :
  initial_count = 20 →
  initial_avg = 49 →
  new_avg = 48 →
  (initial_count * initial_avg + new_teacher_age) / (initial_count + 1) = new_avg →
  new_teacher_age = 28 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_teacher_age_l884_88473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_greatest_common_factor_l884_88414

-- Define the polynomial
def polynomial (x : ℝ) : ℝ := 2 * x^2 + 6 * x^3

-- Define the common factor
def common_factor (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem is_greatest_common_factor :
  ∀ x : ℝ, 
  ∃ k : ℝ, polynomial x = common_factor x * k ∧ 
  ∀ f : ℝ → ℝ, (∃ m : ℝ → ℝ, ∀ y, polynomial y = f y * m y) → 
  ∃ n : ℝ → ℝ, ∀ y, common_factor y = f y * n y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_greatest_common_factor_l884_88414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_difference_l884_88483

theorem sine_cosine_difference (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = -Real.sqrt 5 / 2) 
  (h2 : 5 * Real.pi / 4 < α) 
  (h3 : α < 3 * Real.pi / 2) : 
  Real.cos α - Real.sin α = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_difference_l884_88483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l884_88415

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

noncomputable def g (x : ℝ) : ℝ := Real.exp (-x)

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

theorem function_inequality (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1/2 : ℝ) 2, f_deriv a x₁ > g x₂) ↔ 
  a > Real.exp (-2) - 5/4 :=
by sorry

#check function_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l884_88415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_chocolate_milk_l884_88492

/-- The amount of chocolate milk Holly drinks each time -/
def amount : ℝ := 0

/-- Holly starts with 16 ounces of chocolate milk -/
def initial_amount : ℝ := 16

/-- Holly buys a new 64-ounce container during lunch -/
def new_container : ℝ := 64

/-- Holly ends the day with 56 ounces of chocolate milk -/
def final_amount : ℝ := 56

/-- Theorem stating that Holly drinks 8 ounces of chocolate milk each time -/
theorem holly_chocolate_milk :
  initial_amount - amount + new_container - 2 * amount = final_amount →
  amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_holly_chocolate_milk_l884_88492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l884_88469

/-- Represents a time on a 12-hour digital clock --/
structure DigitalTime where
  hours : Fin 12
  minutes : Fin 60

/-- Checks if a DigitalTime is a palindrome --/
def isPalindrome (t : DigitalTime) : Prop :=
  sorry

/-- Checks if a DigitalTime is valid according to the problem conditions --/
def isValidTime (t : DigitalTime) : Prop :=
  sorry

/-- The set of all valid palindrome times --/
def validPalindromes : Finset DigitalTime :=
  sorry

/-- The main theorem --/
theorem count_palindromes : Finset.card validPalindromes = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l884_88469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l884_88449

theorem perpendicular_lines_a_value (a : ℝ) : 
  (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) →
  a = 1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l884_88449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l884_88442

/-- Represents a 5x5 grid with X's placed on it -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if a given position is within the 5x5 grid -/
def inBounds (row col : ℕ) : Prop := row < 5 ∧ col < 5

/-- Counts the number of X's in the grid -/
def countX (g : Grid) : ℕ := Finset.sum (Finset.univ : Finset (Fin 5)) (λ i => Finset.sum (Finset.univ : Finset (Fin 5)) (λ j => if g i j then 1 else 0))

/-- Checks if there are four X's in a row horizontally, vertically, or diagonally -/
def hasFourInARow (g : Grid) : Prop :=
  ∃ i j : Fin 5, (∀ k : Fin 4, g (i + k) j) ∨ 
                 (∀ k : Fin 4, g i (j + k)) ∨
                 (∀ k : Fin 4, g (i + k) (j + k)) ∨
                 (∀ k : Fin 4, g (i + k) (j - k))

/-- The main theorem stating that 13 is the maximum number of X's that can be placed -/
theorem max_x_placement :
  (∃ g : Grid, countX g = 13 ∧ ¬hasFourInARow g) ∧
  (∀ g : Grid, countX g > 13 → hasFourInARow g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l884_88442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l884_88480

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (A / 2) = -Real.sqrt 3 / 2 →
  a = 3 →
  b + c = 2 * Real.sqrt 3 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l884_88480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_angle_of_line_l884_88422

/-- The inclined angle of a line given by parametric equations -/
noncomputable def inclined_angle (x y : ℝ → ℝ) : ℝ :=
  Real.pi - Real.arctan ((y 1 - y 0) / (x 1 - x 0))

/-- The line given by the parametric equations x = 1 + t, y = 1 - t -/
def line_parameterization (t : ℝ) : ℝ × ℝ :=
  (1 + t, 1 - t)

theorem inclined_angle_of_line :
  inclined_angle (fun t => (line_parameterization t).1) (fun t => (line_parameterization t).2) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_angle_of_line_l884_88422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l884_88454

/-- The game state -/
inductive GameState
| A (n : ℕ)  -- Player A's turn
| B (n : ℕ)  -- Player B's turn

/-- The game result -/
inductive GameResult
| AWins
| BWins
| Draw

/-- Player A's move is valid -/
def valid_move_A (n m : ℕ) : Prop :=
  n ≤ m ∧ m ≤ n^2

/-- Player B's move is valid -/
def valid_move_B (n m : ℕ) : Prop :=
  ∃ (p : ℕ) (r : ℕ), Nat.Prime p ∧ r ≥ 1 ∧ n / m = p^r

/-- The game is over -/
def game_over (s : GameState) : Prop :=
  match s with
  | GameState.A 1990 => True
  | GameState.B 1 => True
  | _ => False

/-- Extract the number from a GameState -/
def get_n (s : GameState) : ℕ :=
  match s with
  | GameState.A n => n
  | GameState.B n => n

/-- The winning strategy for a player -/
def winning_strategy (n₀ : ℕ) (player : GameState → GameState) : Prop :=
  ∀ (opponent : GameState → GameState),
    (∀ s, valid_move_A (get_n s) (get_n (player s)) ∨ valid_move_B (get_n s) (get_n (player s))) →
    (∀ s, valid_move_A (get_n s) (get_n (opponent s)) ∨ valid_move_B (get_n s) (get_n (opponent s))) →
    ∃ (k : ℕ), game_over ((player ∘ opponent)^[k] (GameState.A n₀))

/-- The main theorem -/
theorem game_result (n₀ : ℕ) : n₀ > 1 →
  (n₀ ≥ 8 → ∃ (player_A : GameState → GameState), winning_strategy n₀ player_A) ∧
  (n₀ ∈ ({2, 3, 4, 5} : Set ℕ) → ∃ (player_B : GameState → GameState), winning_strategy n₀ player_B) ∧
  (n₀ ∈ ({6, 7} : Set ℕ) → ¬∃ (player : GameState → GameState), winning_strategy n₀ player) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l884_88454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l884_88464

-- Define the line
def line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ circleEq A.1 A.2 ∧
    line B.1 B.2 ∧ circleEq B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l884_88464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_theorem_l884_88491

/-- The doubling time of the bacteria population in minutes -/
def doubling_time : ℝ := 6

/-- The time taken for the population to reach 500,000 bacteria in minutes -/
def growth_time : ℝ := 53.794705707972525

/-- The final population of bacteria -/
def final_population : ℕ := 500000

/-- The initial population of bacteria (rounded to nearest integer) -/
def initial_population : ℕ := 1010

/-- Theorem stating that given the doubling time and growth time, 
    the initial population grows to the final population -/
theorem bacteria_growth_theorem :
  Int.floor ((final_population : ℝ) / (2 ^ (growth_time / doubling_time))) = initial_population := by
  sorry

#check bacteria_growth_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_theorem_l884_88491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_fourths_l884_88468

def standard_die : Finset ℕ := Finset.range 6

def valid_number (a b : ℕ) : Bool :=
  (10 ≤ a * 10 + b ∧ a * 10 + b ≤ 30) ∨ (10 ≤ b * 10 + a ∧ b * 10 + a ≤ 30)

def probability_valid_number : ℚ :=
  (Finset.filter (fun pair => valid_number pair.1 pair.2) 
    (standard_die.product standard_die)).card / (standard_die.card ^ 2 : ℚ)

theorem probability_is_three_fourths : 
  probability_valid_number = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_fourths_l884_88468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l884_88428

noncomputable def geometric_series (a b : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r) + b * (1 - r^(n-1)) / (1 - r)

theorem problem_solution (k : ℝ) :
  (∀ n : ℕ, geometric_series 5 (3*k) (1/5) n + 5 * (1/5)^n = 12) →
  k = 112/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l884_88428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_exactly_three_distinct_l884_88438

theorem not_exactly_three_distinct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬ (Finset.card {a, b, c, a^2 / b, b^2 / c, c^2 / a} = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_exactly_three_distinct_l884_88438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_total_tickets_l884_88475

/-- The total number of tickets sold for a play, given the number of reduced price tickets sold in the first week, the ratio of full price to reduced price tickets in the remaining weeks, and the total number of full price tickets sold. -/
def total_tickets_sold 
  (reduced_first_week : ℕ) 
  (full_to_reduced_ratio : ℕ) 
  (full_price_total : ℕ) : ℕ :=
  reduced_first_week + full_price_total

/-- Proves that the total number of tickets sold is 21900, given the specific conditions of the problem. -/
theorem prove_total_tickets : 
  total_tickets_sold 5400 5 16500 = 21900 := by
  rfl

#eval total_tickets_sold 5400 5 16500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_total_tickets_l884_88475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l884_88441

/-- Calculates the time taken to travel a given distance at a constant speed -/
noncomputable def travel_time (speed : ℝ) (distance : ℝ) : ℝ :=
  distance / speed

theorem bike_ride_time :
  let speed : ℝ := 1 / 4 -- 1 mile per 4 minutes
  let distance_to_bernard : ℝ := 3.5
  travel_time speed distance_to_bernard = 14 := by
  -- Unfold the definitions
  unfold travel_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l884_88441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_jars_count_final_jar_count_l884_88498

/-- Represents the number of jars of each size -/
def num_jars : ℕ := sorry

/-- Total volume of water in gallons -/
def total_water : ℕ := 14

/-- Theorem stating the total number of jars given the conditions -/
theorem total_jars_count : 
  total_water * 4 = num_jars * 4 + num_jars * 2 + num_jars → 
  3 * num_jars = 24 := by
  intro h
  sorry

/-- Final theorem proving the total number of jars -/
theorem final_jar_count : 3 * num_jars = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_jars_count_final_jar_count_l884_88498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l884_88450

/-- Line in 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculate the distance between a point and a line --/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Calculate the length of a chord given a circle and a line --/
noncomputable def chordLength (c : Circle) (l : Line) : ℝ :=
  2 * Real.sqrt (c.radius^2 - (distancePointToLine c.center l)^2)

/-- Theorem: The length of the chord intercepted by the given line and circle is 2√2 --/
theorem chord_length_is_2_sqrt_2 : 
  let l : Line := { a := 1, b := -1, c := -4 }
  let c : Circle := { center := (2, 0), radius := 2 }
  chordLength c l = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l884_88450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_expenses_l884_88420

/-- 
Given:
- James pays for everything for the prom with Susan.
- Tickets cost $100 each.
- Dinner costs $120.
- James leaves a 30% tip on the dinner.
- James charters a limo for a certain number of hours at $80 per hour.
- The total cost is $836.

Prove: James chartered the limo for 6 hours.
-/
theorem prom_expenses (ticket_cost : ℕ) (dinner_cost : ℕ) (tip_rate : ℚ) 
  (limo_rate : ℕ) (total_cost : ℕ) : 
  ticket_cost = 100 → 
  dinner_cost = 120 → 
  tip_rate = 30 / 100 → 
  limo_rate = 80 → 
  total_cost = 836 → 
  ∃ (limo_hours : ℕ), 
    2 * ticket_cost + dinner_cost + (tip_rate * ↑dinner_cost).floor + limo_rate * limo_hours = total_cost ∧ 
    limo_hours = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_expenses_l884_88420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l884_88437

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (m : ℝ), m = 6 ∧ ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ m := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l884_88437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l884_88478

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c ∧
  (t.A + t.B + t.C = Real.pi) ∧
  (t.a * Real.sin t.B = t.b * Real.sin t.A) ∧
  (t.a * Real.sin t.C = t.c * Real.sin t.A) ∧
  (t.b * Real.cos t.C + t.c * Real.cos t.B > 0) ∧
  t.a = Real.sqrt 3 / 2

-- Theorem statement
theorem triangle_side_sum_range (t : Triangle) 
  (h : triangle_conditions t) : 
  ∃ (x y : ℝ), x = Real.sqrt 3 / 2 ∧ y = 3 / 2 ∧ 
  x < t.b + t.c ∧ t.b + t.c < y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l884_88478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_proof_l884_88413

/-- The probability of an animal surviving one month -/
noncomputable def survival_prob : ℚ := 9/10

/-- The expected number of survivors after 3 months -/
noncomputable def expected_survivors : ℚ := 2187/5

/-- The number of newborn members in the group -/
def newborn_count : ℕ := 600

theorem newborn_count_proof :
  (newborn_count : ℚ) * survival_prob ^ 3 = expected_survivors :=
by
  -- Convert the goal to decimals for easier comparison
  have h1 : (600 : ℚ) * (9/10)^3 = 2187/5 := by norm_num
  -- Apply the equality
  exact h1

#eval newborn_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_proof_l884_88413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evolute_and_max_distance_l884_88448

-- Define the curve C2
noncomputable def C2 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

-- Define the evolute C3
def C3 (x y : ℝ) : Prop :=
  (x^2 + y^2)^2 = (1/9) * (4*x^2 + 49*y^2)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem evolute_and_max_distance :
  -- Part 1: The evolute of C2 is C3
  (∀ α : ℝ, ∃ x y : ℝ, C3 x y ∧ 
    x = (2/3) * Real.cos α ∧ 
    y = (7/3) * Real.sin α) ∧
  -- Part 2: The maximum distance between C2 and C3 is 4√3/3
  (∃ maxDist : ℝ, maxDist = 4 * Real.sqrt 3 / 3 ∧
    ∀ α β : ℝ, distance (C2 α) ((2/3) * Real.cos β, (7/3) * Real.sin β) ≤ maxDist) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evolute_and_max_distance_l884_88448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tolya_luga_upstream_time_l884_88465

/-- Represents the time in minutes for Tolya to swim downstream in the Volkhov River -/
noncomputable def volkhov_downstream : ℝ := 18

/-- Represents the time in minutes for Tolya to swim upstream in the Volkhov River -/
noncomputable def volkhov_upstream : ℝ := 60

/-- Represents the time in minutes for Tolya to swim downstream in the Luga River -/
noncomputable def luga_downstream : ℝ := 20

/-- Represents the distance in km between the beaches (same for both rivers) -/
noncomputable def distance : ℝ := 1  -- We set this to 1 as a placeholder, as the actual value doesn't affect the result

/-- Calculates Tolya's swimming speed in still water -/
noncomputable def swimming_speed : ℝ := (distance / volkhov_downstream + distance / volkhov_upstream) / 2

/-- Calculates the current speed of the Volkhov River -/
noncomputable def volkhov_current : ℝ := (distance / volkhov_downstream - distance / volkhov_upstream) / 2

/-- Calculates the current speed of the Luga River -/
noncomputable def luga_current : ℝ := distance / luga_downstream - swimming_speed

/-- Theorem: Given the conditions, Tolya's upstream swimming time in the Luga River is 45 minutes -/
theorem tolya_luga_upstream_time : 
  distance / (swimming_speed - luga_current) = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tolya_luga_upstream_time_l884_88465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_1_and_100_l884_88477

noncomputable def harmonic_mean (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem closest_integer_to_harmonic_mean_of_1_and_100 :
  let h := harmonic_mean 1 100
  ∀ x ∈ ({1, 2, 98, 100, 199} : Set ℝ), |h - 2| ≤ |h - x| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_1_and_100_l884_88477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_tangent_equation_l884_88490

/-- The curve function f(x) = x^2 - 2x - 3 -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Point P on the curve -/
noncomputable def P : ℝ × ℝ := (1, f 1)

/-- Point Q on the curve -/
noncomputable def Q : ℝ × ℝ := (4, f 4)

/-- The slope of the secant line PQ -/
noncomputable def secant_slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := 2*x - 2

/-- The slope of the tangent line at point P -/
noncomputable def tangent_slope : ℝ := f' P.1

theorem curve_properties :
  secant_slope = 3 ∧ tangent_slope = 0 := by sorry

theorem tangent_equation :
  ∀ x y : ℝ, y + 4 = 0 ↔ y - P.2 = tangent_slope * (x - P.1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_tangent_equation_l884_88490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_theorem_l884_88430

/-- The angle in the axial section of a cone, where a sphere with its center at the vertex
    of the cone touches its base and divides the volume of the cone in half. -/
noncomputable def cone_angle : ℝ := 2 * Real.arccos ((1 + Real.sqrt 17) / 8)

/-- The volume of the spherical sector formed by the sphere inside the cone. -/
noncomputable def sphere_sector_volume (R_m : ℝ) (h : ℝ) : ℝ := (2 / 3) * Real.pi * R_m^2 * h

/-- The volume of the cone. -/
noncomputable def cone_volume (R : ℝ) (H : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * H

/-- Theorem stating that the angle in the axial section of the cone satisfies the given conditions. -/
theorem cone_angle_theorem (R_m R H h : ℝ) (hpos : R_m > 0 ∧ R > 0 ∧ H > 0 ∧ h > 0) :
  sphere_sector_volume R_m h = (1 / 2) * cone_volume R H →
  H = R_m →
  h = R_m * (1 - Real.cos (cone_angle / 2)) →
  R = R_m * Real.tan (cone_angle / 2) →
  cone_angle = 2 * Real.arccos ((1 + Real.sqrt 17) / 8) := by
  sorry

#check cone_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_theorem_l884_88430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_from_5_or_6_squares_l884_88409

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  positive : 0 < sideLength

/-- Represents a rectangle formed by squares -/
structure Rectangle where
  squares : List Square
  isRectangle : Bool  -- This would be true if the squares form a valid rectangle

/-- Function to check if a list of squares can form a rectangle -/
def canFormRectangle (squares : List Square) : Bool :=
  sorry  -- Implementation details omitted

/-- Theorem stating that 5 or 6 different-sized squares cannot form a rectangle -/
theorem no_rectangle_from_5_or_6_squares :
  ∀ (n : ℕ) (squares : List Square),
    (n = 5 ∨ n = 6) →
    squares.length = n →
    (∀ (i j : Fin squares.length), i ≠ j →
      (squares.get i).sideLength ≠ (squares.get j).sideLength) →
    ¬(canFormRectangle squares) :=
by
  sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_from_5_or_6_squares_l884_88409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_exclamation_correct_exclamation_pattern_consistent_l884_88432

/-- Represents the alien's exclamation for a given day -/
inductive AlienExclamation
| Monday : AlienExclamation
| Tuesday : AlienExclamation
| Wednesday : AlienExclamation
| Thursday : AlienExclamation
| Friday : AlienExclamation
| Saturday : AlienExclamation

/-- Converts an AlienExclamation to a string -/
def alienExclamationToString : AlienExclamation → String
| AlienExclamation.Monday => "А"
| AlienExclamation.Tuesday => "АУ"
| AlienExclamation.Wednesday => "АУУА"
| AlienExclamation.Thursday => "АУУАУААУ"
| AlienExclamation.Friday => "АУУАУААУУААУАУУА"
| AlienExclamation.Saturday => "АУУАУААУУААУАУУАУААУАУУААУААУАУУ"

/-- Swaps 'А' with 'У' and vice versa in a string -/
def swapLetters (s : String) : String :=
  s.map (fun c => if c == 'А' then 'У' else if c == 'У' then 'А' else c)

/-- Generates the next day's exclamation based on the current day's exclamation -/
def nextDayExclamation (s : String) : String :=
  s ++ (swapLetters s.data.reverse.asString)

/-- Theorem: The alien's exclamation on Saturday follows the pattern and is correct -/
theorem saturday_exclamation_correct :
  alienExclamationToString AlienExclamation.Saturday =
  nextDayExclamation (alienExclamationToString AlienExclamation.Friday) :=
by sorry

/-- Theorem: The alien's exclamation pattern is consistent for all days -/
theorem exclamation_pattern_consistent :
  ∀ day : AlienExclamation,
    day ≠ AlienExclamation.Saturday →
    alienExclamationToString (match day with
      | AlienExclamation.Monday => AlienExclamation.Tuesday
      | AlienExclamation.Tuesday => AlienExclamation.Wednesday
      | AlienExclamation.Wednesday => AlienExclamation.Thursday
      | AlienExclamation.Thursday => AlienExclamation.Friday
      | AlienExclamation.Friday => AlienExclamation.Saturday
      | AlienExclamation.Saturday => AlienExclamation.Saturday) =
    nextDayExclamation (alienExclamationToString day) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_exclamation_correct_exclamation_pattern_consistent_l884_88432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l884_88495

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the domain of x
def X : Set ℝ := Set.Icc (-5) 5

-- Statement for part (1)
theorem part_one :
  let a := -1
  ∃ (x_max x_min : ℝ), x_max ∈ X ∧ x_min ∈ X ∧
    (∀ x ∈ X, f a x ≤ f a x_max) ∧
    (∀ x ∈ X, f a x ≥ f a x_min) ∧
    f a x_max = 37 ∧ f a x_min = 1 :=
by sorry

-- Define the minimum value function
noncomputable def f_min (a : ℝ) : ℝ :=
  if a < -5 then 27 + 10*a
  else if a > 5 then 27 - 10*a
  else 2 - a^2

-- Statement for part (2)
theorem part_two :
  ∀ a : ℝ, ∀ x ∈ X, f a x ≥ f_min a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l884_88495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_sufficient_not_necessary_for_similarity_l884_88479

/-- Two triangles in a Euclidean plane -/
structure Triangle :=
  (A B C : Real × Real)

/-- Congruence relation between two triangles -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Similarity relation between two triangles -/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Theorem stating that congruence is sufficient but not necessary for similarity -/
theorem congruence_sufficient_not_necessary_for_similarity :
  (∀ t1 t2 : Triangle, congruent t1 t2 → similar t1 t2) ∧
  (∃ t1 t2 : Triangle, similar t1 t2 ∧ ¬congruent t1 t2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_sufficient_not_necessary_for_similarity_l884_88479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l884_88433

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 10 * x + 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l884_88433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l884_88434

/-- Given two similar triangles with corresponding sides of 16 and 32,
    and a difference in perimeters of 36, the perimeter of the smaller triangle is 36. -/
theorem similar_triangles_perimeter (t1 t2 : Real) (h1 : t1 > 0 ∧ t2 > 0)
  (h2 : t2 / t1 = 2)
  (h3 : t2 - t1 = 36) :
  t1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_l884_88434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_sqrt_three_l884_88411

noncomputable def tan_sum (α β : Real) : Real := 
  (Real.tan α + Real.tan β) / (1 - Real.tan α * Real.tan β)

theorem tan_sum_sqrt_three (α β : Real) 
  (h : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) : 
  tan_sum α β = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_sqrt_three_l884_88411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_l884_88463

theorem temperature_difference : ∃ (diff : ℝ), diff = 9 := by
  -- Define the temperatures
  let temp1 : ℝ := 3
  let temp2 : ℝ := -6
  
  -- Define the difference
  let diff : ℝ := temp1 - temp2

  -- Prove the theorem
  use diff
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_l884_88463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l884_88447

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (x - 8*x^2 + 16*x^3) / (9 - x^3)

/-- Theorem stating the nonnegative interval for which the function is nonnegative -/
theorem f_nonnegative_iff (x : ℝ) :
  x ≥ 0 → (f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l884_88447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l884_88457

noncomputable section

-- Define the triangle and its properties
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Side lengths opposite to angles A, B, C

-- Define the vectors as functions
def m (c : ℝ) : ℝ × ℝ := (2, c)
def n (b A B C : ℝ) : ℝ × ℝ := (b/2 * Real.cos C - Real.sin A, Real.cos B)

-- State the theorem
theorem triangle_properties 
  (acute_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (angle_sum : A + B + C = π)
  (side_b : b = Real.sqrt 3)
  (vectors_perpendicular : (m c).1 * (n b A B C).1 + (m c).2 * (n b A B C).2 = 0) :
  B = π/3 ∧ 
  (∀ a' c', a' > 0 → c' > 0 → a' * c' * Real.sin B ≤ 3 * Real.sqrt 3 / 2) ∧
  (∃ a' c', a' > 0 ∧ c' > 0 ∧ a' * c' * Real.sin B = 3 * Real.sqrt 3 / 2 ∧ a' = Real.sqrt 3 ∧ c' = Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l884_88457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_cosine_l884_88435

noncomputable section

open Real

theorem triangle_angle_b_cosine 
  (A B C : ℝ → ℝ → ℝ → Real)
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_ratio : (B a b c).cos / (C a b c).cos = -b / (2*a + c)) : 
  (B a b c).cos = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_cosine_l884_88435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_distance_bounds_l884_88431

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line l
def l (t x y : ℝ) : Prop := x = 2 + t ∧ y = 2 - 2*t

-- Define a point P on curve C
def P_on_C (P : ℝ × ℝ) : Prop := C P.1 P.2

-- Define the angle between a line through P and line l
noncomputable def angle_with_l (P A : ℝ × ℝ) : ℝ := 30 * Real.pi / 180

-- Define the distance between two points
noncomputable def distance (P A : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

-- Theorem statement
theorem PA_distance_bounds :
  ∀ (P A : ℝ × ℝ) (t : ℝ),
  P_on_C P →
  l t A.1 A.2 →
  angle_with_l P A = 30 * Real.pi / 180 →
  distance P A ≤ 22 * Real.sqrt 5 / 5 ∧
  distance P A ≥ 2 * Real.sqrt 5 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_distance_bounds_l884_88431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l884_88496

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 4 else (2 : ℝ)^x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (f a) > f (f a + 1)) ↔ (a > -5/2 ∧ a ≤ -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l884_88496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l884_88482

noncomputable section

-- Define the interval
def I : Set ℝ := Set.Icc (-5/4) (7/4)

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x + Real.pi/4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.pi * x + Real.pi/4)

-- Define the intersection points
def A : ℝ × ℝ := (-1, -Real.sqrt 2 / 2)
def B : ℝ × ℝ := (0, Real.sqrt 2 / 2)
def C : ℝ × ℝ := (1, -Real.sqrt 2 / 2)

-- Theorem statement
theorem intersection_triangle_area : 
  ∃ (a b c : ℝ × ℝ), a ∈ I.prod (Set.range f) ∧ 
                      b ∈ I.prod (Set.range f) ∧ 
                      c ∈ I.prod (Set.range f) ∧
                      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                      f a.1 = g a.1 ∧ 
                      f b.1 = g b.1 ∧ 
                      f c.1 = g c.1 ∧
                      abs ((a.1 - c.1) * (b.2 - a.2) - (b.1 - a.1) * (c.2 - a.2)) / 2 = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l884_88482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l884_88417

/-- A function f : ℝ → ℝ is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = ax - sin x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.sin x

/-- Proposition p: a > 1 -/
def p (a : ℝ) : Prop := a > 1

/-- Proposition q: f(x) = ax - sin x is an increasing function on ℝ -/
def q (a : ℝ) : Prop := IsIncreasing (f a)

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, ¬p a ∧ q a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l884_88417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ECODF_l884_88400

-- Define the points
variable (A B C D E F O : ℝ × ℝ)

-- Define the radii of the circles
def radius_A : ℝ := 3
def radius_B : ℝ := 2

-- Define the distance OA
def distance_OA : ℝ := 3

-- O is the midpoint of AB
axiom is_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- OC is tangent to circle A
axiom tangent_OC : ‖O - C‖ = Real.sqrt (‖O - A‖^2 - radius_A^2)

-- OD is tangent to circle B
axiom tangent_OD : ‖O - D‖ = Real.sqrt (‖O - B‖^2 - radius_B^2)

-- EF is a common tangent to both circles
axiom common_tangent : ‖E - F‖ = ‖A - B‖ * (radius_A + radius_B) / ‖A - B‖

-- Theorem statement
theorem area_ECODF : 
  let area := ‖A - B‖ * (‖O - C‖ + ‖O - D‖) - (‖O - D‖ * ‖O - B‖) / 2
  area = (9 * Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ECODF_l884_88400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l884_88476

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else Real.log x + 2

theorem f_solution_set (x : ℝ) : f x > 3 ↔ x < -3 ∨ x > Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l884_88476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_ge_11_l884_88408

-- Define the constants from the problem
noncomputable def distance_to_shore : ℝ := 1.5
noncomputable def water_intake_rate : ℝ := 12
noncomputable def boat_capacity : ℝ := 45
noncomputable def rowing_speed : ℝ := 4
noncomputable def bailing_efficiency : ℝ := 0.9

-- Define the function to calculate the minimum bailing rate
noncomputable def min_bailing_rate : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  (water_intake_rate * time_to_shore - boat_capacity) / (bailing_efficiency * time_to_shore)

-- Theorem statement
theorem min_bailing_rate_ge_11 : min_bailing_rate ≥ 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_ge_11_l884_88408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_excluded_values_l884_88467

-- Define the quadratic equation
noncomputable def quadratic (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 6

-- Define the sum of roots using Vieta's formula
noncomputable def sum_of_roots : ℝ := 9 / 3

-- Theorem statement
theorem sum_of_excluded_values :
  (∃ x y : ℝ, x ≠ y ∧ quadratic x = 0 ∧ quadratic y = 0) ∧
  (∀ x y : ℝ, x ≠ y → quadratic x = 0 → quadratic y = 0 → x + y = sum_of_roots) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_excluded_values_l884_88467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l884_88472

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the focal points
def FocalPoints (c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0))

-- Define a point on the hyperbola
def PointOnHyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P ∈ Hyperbola a b

-- Define the angle between focal points and P
def AngleFPF (F₁ F₂ P : ℝ × ℝ) (angle : ℝ) : Prop := sorry

-- Define the distance ratio condition
def DistanceRatio (F₁ F₂ P : ℝ × ℝ) : Prop :=
  dist F₁ P = 3 * dist P F₂

-- Define eccentricity
noncomputable def Eccentricity (c a : ℝ) : ℝ := c / a

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (h₁ : FocalPoints c = (F₁, F₂))
  (h₂ : PointOnHyperbola P a b)
  (h₃ : AngleFPF F₁ F₂ P (2 * Real.pi / 3))
  (h₄ : DistanceRatio F₁ F₂ P) :
  Eccentricity c a = Real.sqrt 13 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l884_88472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l884_88443

def S (n : ℕ) : ℚ := n^2 + 2/3

def a : ℕ → ℚ
| 0 => 0  -- Add a case for 0 to cover all natural numbers
| 1 => 5/3
| (n+2) => 2*(n+2) - 1

theorem sequence_formula : 
  ∀ n : ℕ, n ≥ 1 → 
    (n = 1 ∧ a n = S n) ∨ 
    (n > 1 ∧ a n = S n - S (n-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l884_88443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l884_88402

/-- A set of consecutive odd integers -/
def ConsecutiveOddSet (median : ℤ) (greatest : ℤ) : Set ℤ :=
  {n | ∃ k, n = median + 2 * k ∧ n % 2 = 1 ∧ n ≤ greatest}

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ)
    (h_median : median = 126)
    (h_greatest : greatest = 153) :
    ∃ smallest, smallest ∈ ConsecutiveOddSet median greatest ∧
      (∀ n ∈ ConsecutiveOddSet median greatest, smallest ≤ n) ∧ smallest = 100 :=
  sorry

#check smallest_integer_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l884_88402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribable_square_theorem_l884_88484

/-- The side length of the largest inscribable square given a square of side length 12
    and two congruent equilateral triangles drawn inside as described in the problem. -/
noncomputable def largest_inscribable_square_side : ℝ := 6 - Real.sqrt 6

/-- The side length of the large square. -/
def large_square_side : ℝ := 12

/-- The side length of each equilateral triangle. -/
noncomputable def triangle_side : ℝ := 4 * Real.sqrt 6

/-- Theorem stating the relationship between the largest inscribable square side,
    the large square side, and the triangle side. -/
theorem largest_inscribable_square_theorem :
  ∃ (s : ℝ), s = largest_inscribable_square_side ∧
  s * Real.sqrt 2 = large_square_side * Real.sqrt 2 - triangle_side :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribable_square_theorem_l884_88484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l884_88444

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The cosine of the angle between the foci and a point on the ellipse -/
noncomputable def cos_foci_angle (e : Ellipse) (p : ℝ × ℝ) : ℝ := sorry

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / e.a^2) + (p.2^2 / e.b^2) = 1

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem ellipse_properties (e : Ellipse) 
  (h_major_axis : e.a = 2)
  (h_min_cos : ∀ p, ellipse_equation e p → cos_foci_angle e p ≥ 1/2) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ m : ℝ, -e.b < m ∧ m < e.b →
    (∃ (A B N : ℝ × ℝ),
      ellipse_equation e A ∧
      ellipse_equation e B ∧
      N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
      (∀ k : ℝ, 
        (dot_product A B - dot_product (0, m) N = 
         dot_product A B - dot_product (0, m) N) →
        m = Real.sqrt 3 / 2 ∨ m = -Real.sqrt 3 / 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l884_88444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_driver_net_pay_l884_88458

/-- Calculates the truck driver's net rate of pay per hour -/
noncomputable def netRateOfPay (travelTime : ℝ) (speed : ℝ) (fuelEfficiency : ℝ) (payRate : ℝ) (dieselCost : ℝ) : ℝ :=
  let totalDistance := travelTime * speed
  let dieselUsage := totalDistance / fuelEfficiency
  let earnings := payRate * totalDistance
  let dieselExpense := dieselCost * dieselUsage
  let netEarnings := earnings - dieselExpense
  netEarnings / travelTime

/-- The truck driver's net rate of pay is $24.75 per hour -/
theorem truck_driver_net_pay :
  netRateOfPay 3 45 15 0.75 3 = 24.75 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval netRateOfPay 3 45 15 0.75 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_driver_net_pay_l884_88458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hanas_speed_l884_88470

noncomputable def race_distance : ℝ := 100
noncomputable def megans_speed : ℝ := 5/4
noncomputable def time_difference : ℝ := 5

theorem hanas_speed (hanas_time : ℝ) (megans_time : ℝ)
  (h1 : megans_time = race_distance / megans_speed)
  (h2 : hanas_time = megans_time - time_difference)
  (h3 : hanas_time > 0) :
  race_distance / hanas_time = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hanas_speed_l884_88470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l884_88497

noncomputable def f (x : ℝ) := x^2 - Real.cos x

theorem relationship_abc :
  let a := f (3^(0.3 : ℝ))
  let b := f (Real.log 3 / Real.log Real.pi)
  let c := f (Real.log (1/9) / Real.log 3)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l884_88497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l884_88489

/-- Represents an acute-angled triangle ABC -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

/-- The theorem stating the relationship between sides, area, and angle C in the given triangle -/
theorem angle_C_measure (triangle : AcuteTriangle) 
  (side_a : triangle.a = 2 * Real.sqrt 3)
  (side_b : triangle.b = 2)
  (area : (1/2) * triangle.a * triangle.b * Real.sin triangle.C = Real.sqrt 3) :
  triangle.C = π / 6 := by
  sorry

#check angle_C_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l884_88489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_persons_meet_at_600m_l884_88471

/-- The distance between two points A and B in meters -/
noncomputable def distance : ℝ := 1000

/-- The time person A takes to travel from A to B in hours -/
noncomputable def time_A : ℝ := 1

/-- The time person B takes to travel from B to A in hours -/
noncomputable def time_B : ℝ := 40 / 60

/-- The delay in person B's start time compared to person A in hours -/
noncomputable def delay : ℝ := 20 / 60

/-- The speed of person A in meters per hour -/
noncomputable def speed_A : ℝ := distance / time_A

/-- The speed of person B in meters per hour -/
noncomputable def speed_B : ℝ := distance / time_B

/-- The meeting point of persons A and B -/
noncomputable def meeting_point : ℝ := 600

theorem persons_meet_at_600m :
  ∃ (t : ℝ), t > 0 ∧ t < time_A ∧
  speed_A * (t + delay) = meeting_point ∧
  speed_B * t = distance - meeting_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_persons_meet_at_600m_l884_88471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l884_88416

-- Define the hyperbola
noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Define the distance between foci
noncomputable def foci_distance (a b : ℝ) : ℝ := 2 * (b^2 / a)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y c : ℝ), 
    hyperbola a b x y ∧ 
    c = foci_distance a b / 2 ∧
    x > 0 ∧ 
    y > 0 ∧
    x = c ∧
    eccentricity a c = 1 + Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l884_88416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_less_than_a_implies_a_greater_than_neg_one_l884_88499

theorem sin_less_than_a_implies_a_greater_than_neg_one (a : ℝ) :
  (∃ x : ℝ, Real.sin x < a) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_less_than_a_implies_a_greater_than_neg_one_l884_88499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_proposition_proof_l884_88481

theorem compound_proposition_proof :
  (∃ x : ℝ, (2 : ℝ)^x ≥ (3 : ℝ)^x) ∧ (∃ x₀ : ℝ, x₀^3 = 1 - x₀^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_proposition_proof_l884_88481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_berry_distribution_l884_88418

theorem bird_berry_distribution (squirrels birds rabbits nuts berries : ℕ) :
  squirrels = 4 →
  birds = 3 →
  rabbits = 2 →
  nuts = 2 →
  berries = 10 →
  squirrels * 1 = nuts →
  (squirrels + birds) * 2 = berries →
  ∃ (bird_berries : ℕ), bird_berries = 2 ∧ birds * bird_berries = berries - squirrels :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_berry_distribution_l884_88418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l884_88455

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x - y - 6 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (1, 2)

-- Define the radius of the circle
noncomputable def radius : ℝ := Real.sqrt 5

-- Define the length of chord AB
noncomputable def chord_length : ℝ := Real.sqrt 10

theorem circle_and_line_intersection :
  (∀ x y, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  chord_length^2 = 4 * (radius^2 - (3*center.1 - center.2 - 6)^2 / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l884_88455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_form_curve_C_range_l884_88406

-- Define the curve C
noncomputable def curve_C (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, 3 * Real.sin φ)

-- Theorem 1: General form of curve C
theorem curve_C_general_form (x y : ℝ) :
  (∃ φ, curve_C φ = (x, y)) ↔ x^2/4 + y^2/9 = 1 := by sorry

-- Theorem 2: Range of 2x + y for points on curve C
theorem curve_C_range (x y : ℝ) :
  (∃ φ, curve_C φ = (x, y)) → -5 ≤ 2*x + y ∧ 2*x + y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_form_curve_C_range_l884_88406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_sum_of_sides_l884_88459

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

-- Define the relationship between sides and angles
def sides_angles_relation (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.c / Real.sin t.C = t.a / Real.sin t.A

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : Real :=
  1/2 * t.a * t.b * Real.sin t.C

-- Theorem 1
theorem angle_C_value (t : Triangle) 
  (h1 : is_acute_triangle t)
  (h2 : sides_angles_relation t)
  (h3 : given_condition t) : 
  t.C = Real.pi/3 :=
by sorry

-- Theorem 2
theorem sum_of_sides (t : Triangle)
  (h1 : is_acute_triangle t)
  (h2 : sides_angles_relation t)
  (h3 : given_condition t)
  (h4 : t.c = Real.sqrt 7)
  (h5 : triangle_area t = 3 * Real.sqrt 3 / 2) :
  t.a + t.b = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_sum_of_sides_l884_88459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_tank_dimensions_l884_88429

open Real

/-- Represents the cost of materials for a cylindrical tank -/
noncomputable def tank_cost (r h : ℝ) : ℝ := 40 * Real.pi * r^2 + 64 * Real.pi * r * h

/-- The theorem stating the optimal dimensions and minimum cost of the tank -/
theorem optimal_tank_dimensions :
  ∃ (r h : ℝ),
    r > 0 ∧ 
    h > 0 ∧
    Real.pi * r^2 * h = 20 * Real.pi ∧
    r = 2 ∧
    h = 5 ∧
    ∀ (r' h' : ℝ), r' > 0 → h' > 0 → Real.pi * r'^2 * h' = 20 * Real.pi → 
      tank_cost r h ≤ tank_cost r' h' :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_tank_dimensions_l884_88429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_at_center_l884_88446

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 4

-- Define the line
def line (t : ℝ) : ℝ × ℝ := (2*t - 1, 6*t - 1)

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 3)

-- Theorem statement
theorem line_intersects_circle_not_at_center :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
  circle_eq (line t₁).1 (line t₁).2 ∧
  circle_eq (line t₂).1 (line t₂).2 ∧
  line t₁ ≠ circle_center ∧
  line t₂ ≠ circle_center :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_at_center_l884_88446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l884_88462

def is_T_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)) ∧
  (∃ M : ℝ, ∀ n : ℕ, a n ≤ M)

noncomputable def sequence_8_minus_2_pow (n : ℕ) : ℝ := 8 - 2^(n : ℝ)

theorem T_sequence_properties :
  (is_T_sequence sequence_8_minus_2_pow) ∧
  (∀ a : ℕ → ℕ, is_T_sequence (λ n ↦ (a n : ℝ)) →
    (∀ n : ℕ, a n ≤ a (n + 1))) ∧
  (∀ a : ℕ → ℕ, is_T_sequence (λ n ↦ (a n : ℝ)) →
    ∃ n₀ : ℕ, ∀ n : ℕ, a (n₀ + n + 1) = a (n₀ + n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l884_88462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_twelve_five_l884_88474

theorem cos_arctan_twelve_five : Real.cos (Real.arctan (12 / 5)) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_twelve_five_l884_88474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l884_88488

/-- Represents a parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- The minimum distance from a point on the parabola to the focus -/
noncomputable def min_distance_to_focus (para : Parabola) : ℝ := para.p / 2

/-- The equation of the directrix of a parabola -/
def directrix_equation (para : Parabola) : ℝ → Prop :=
  fun x => x = -para.p / 2

theorem parabola_properties (para : Parabola) :
  min_distance_to_focus para = 1 →
  para.p = 2 ∧ directrix_equation para = fun x => x = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l884_88488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pos_implies_sin_double_pos_l884_88404

theorem tan_pos_implies_sin_double_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pos_implies_sin_double_pos_l884_88404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_floor_l884_88487

noncomputable def floor_length : ℝ := 10
noncomputable def floor_width : ℝ := 15
noncomputable def tile_length : ℝ := 6 / 12  -- 6 inches in feet
noncomputable def tile_width : ℝ := 9 / 12   -- 9 inches in feet

theorem tiles_needed_for_floor : 
  ⌈(floor_length * floor_width) / (tile_length * tile_width)⌉ = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_floor_l884_88487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_three_thirty_l884_88445

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees each hour mark represents on a clock face -/
noncomputable def degrees_per_hour : ℝ := 360 / clock_hours

/-- The position of the minute hand at 30 minutes past any hour, in degrees -/
def minute_hand_position : ℝ := 180

/-- The position of the hour hand at 3:30, in degrees -/
noncomputable def hour_hand_position : ℝ := 3 * degrees_per_hour + degrees_per_hour / 2

/-- The smaller angle between the hour and minute hands at 3:30 -/
noncomputable def clock_angle : ℝ := minute_hand_position - hour_hand_position

theorem clock_angle_at_three_thirty :
  clock_angle = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_three_thirty_l884_88445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_lucas_75_mod_7_l884_88493

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n+2 => modifiedLucas n + modifiedLucas (n+1)

theorem modified_lucas_75_mod_7 :
  modifiedLucas 75 ≡ 0 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_lucas_75_mod_7_l884_88493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cube_with_tunnel_l884_88451

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure CubeWithTunnel where
  edgeLength : ℝ
  E : Point3D
  I : Point3D
  J : Point3D
  K : Point3D

/-- Checks if a number is not divisible by the square of any prime -/
def isNotDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- Function to calculate the surface area of the cube with tunnel -/
def surfaceArea (cube : CubeWithTunnel) : ℝ :=
  sorry  -- Actual implementation would go here

/-- Main theorem about the surface area of the cube with tunnel -/
theorem surface_area_of_cube_with_tunnel (cube : CubeWithTunnel) 
  (h1 : cube.edgeLength = 8)
  (h2 : cube.E.x = 0 ∧ cube.E.y = 0 ∧ cube.E.z = 0)
  (h3 : cube.I.x = 2 ∧ cube.I.y = 0 ∧ cube.I.z = 0)
  (h4 : cube.J.x = 0 ∧ cube.J.y = 2 ∧ cube.J.z = 0)
  (h5 : cube.K.x = 0 ∧ cube.K.y = 0 ∧ cube.K.z = 2) :
  ∃ (m n p : ℕ), 
    (m > 0 ∧ n > 0 ∧ p > 0) ∧
    isNotDivisibleBySquareOfPrime p ∧
    (m + n + p = 90) ∧
    (surfaceArea cube = m + n * Real.sqrt p) :=
by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cube_with_tunnel_l884_88451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l884_88436

open Real

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 1

-- Define the circle passing through minor axis endpoint and foci
def circle_area (r : ℝ) : Prop := Real.pi * r^2 = 4*Real.pi/3

-- Define the line l passing through right focus
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1) ∧ k ≠ 0

-- Define point P as midpoint of AB
def midpoint_P (xA yA xB yB xP yP : ℝ) : Prop :=
  xP = (xA + xB) / 2 ∧ yP = (yA + yB) / 2

-- Define line PD perpendicular to AB
def line_PD (k xP yP xD : ℝ) : Prop :=
  xD = (k^2) / (3 + 4*k^2)

-- Define the length |DP|
def length_DP (xP yP xD : ℝ) : Prop :=
  (xP - xD)^2 + yP^2 = (3*sqrt 2 / 7)^2

theorem ellipse_and_line_theorem
  (a b c r k xA yA xB yB xP yP xD : ℝ) :
  ellipse_C a b xA yA →
  ellipse_C a b xB yB →
  focal_length c →
  circle_area r →
  line_l k xA yA →
  line_l k xB yB →
  midpoint_P xA yA xB yB xP yP →
  line_PD k xP yP xD →
  length_DP xP yP xD →
  (a = 2 ∧ b = sqrt 3) ∧ (k = 1 ∨ k = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l884_88436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_triangle_area_l884_88410

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop

/-- Chord type -/
structure Chord where
  start : ℝ × ℝ
  angle : ℝ

/-- Triangle type -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given ellipse -/
noncomputable def givenEllipse : Ellipse :=
  { a := Real.sqrt 2
    b := 1
    equation := fun x y ↦ x^2/2 + y^2 = 1 }

/-- Foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- Given chord -/
noncomputable def givenChord : Chord :=
  { start := F₂
    angle := Real.pi/4 }

/-- Function to calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Theorem statement -/
theorem ellipse_chord_triangle_area :
  ∃ (t : Triangle), t.C = F₁ ∧ triangleArea t = 4/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_triangle_area_l884_88410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l884_88401

/-- A rectangle with given side lengths -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ

/-- The diagonals of a rectangle -/
structure Diagonals where
  AC : ℝ
  BD : ℝ

/-- Helper function to calculate the area of triangle AOB -/
noncomputable def area_triangle_AOB (rect : Rectangle) (diag : Diagonals) : ℝ :=
  (1/2) * rect.AB * (diag.AC / 2)

/-- Theorem about properties of a specific rectangle -/
theorem rectangle_properties (rect : Rectangle) (diag : Diagonals) 
    (h1 : rect.AB = 30) (h2 : rect.BC = 40) (h3 : rect.CD = 30) (h4 : rect.DA = 40) :
  diag.AC = 50 ∧ diag.BD = 50 ∧ area_triangle_AOB rect diag = 375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l884_88401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l884_88405

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 1.5
noncomputable def b : ℝ := Real.exp 0.4
noncomputable def c : ℝ := Real.log 9 / Real.log 0.8

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l884_88405
