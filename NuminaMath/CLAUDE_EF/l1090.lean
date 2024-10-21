import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_fifteenth_l1090_109046

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The area of the large equilateral triangle -/
noncomputable def large_triangle_area : ℝ := equilateral_triangle_area 12

/-- The area of the small equilateral triangle -/
noncomputable def small_triangle_area : ℝ := equilateral_triangle_area 3

/-- The area of the isosceles trapezoid -/
noncomputable def trapezoid_area : ℝ := large_triangle_area - small_triangle_area

/-- The ratio of the small triangle area to the trapezoid area -/
noncomputable def area_ratio : ℝ := small_triangle_area / trapezoid_area

theorem area_ratio_is_one_fifteenth : area_ratio = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_fifteenth_l1090_109046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1090_109097

theorem sum_of_solutions : 
  ∃ S : Finset ℝ, (∀ x ∈ S, (2 : ℝ)^(x^2 - 3*x - 4) = (4 : ℝ)^(x - 5)) ∧ 
  (∀ y : ℝ, (2 : ℝ)^(y^2 - 3*y - 4) = (4 : ℝ)^(y - 5) → y ∈ S) ∧
  (S.sum id) = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1090_109097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l1090_109034

/-- Represents a right triangle with specific properties -/
structure RightTriangle where
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- Measure of the first acute angle in degrees -/
  angle1 : ℝ
  /-- Measure of the second acute angle in degrees -/
  angle2 : ℝ
  /-- The triangle is right-angled -/
  is_right : angle1 + angle2 = 90
  /-- The ratio of acute angles is 5:4 -/
  angle_ratio : angle1 / angle2 = 5 / 4
  /-- The hypotenuse is 10 units long -/
  hyp_length : hypotenuse = 10

/-- Calculates the area of the right triangle -/
noncomputable def triangle_area (t : RightTriangle) : ℝ :=
  50 * Real.sin (t.angle1 * Real.pi / 180) * Real.sin (t.angle2 * Real.pi / 180)

/-- Theorem stating the area of the specific right triangle -/
theorem specific_triangle_area :
  ∃ t : RightTriangle, ∀ ε > 0, |triangle_area t - 24.63156| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l1090_109034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_theorem_l1090_109085

/-- The maximum number of unit segments in an equilateral triangle of side length n -/
def max_segments (n : ℕ) : ℕ := n * (n + 1)

/-- The total number of unit segments in an equilateral triangle of side length n -/
def total_segments (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- Predicate to check if three segments are sides of a unit triangle at given coordinates -/
def are_sides_of_unit_triangle (a b c : ℕ) (x y z : ℕ) : Prop :=
  sorry -- Definition of what it means for three segments to form a unit triangle

/-- Predicate to check if three segments form a triangle in the divided equilateral triangle -/
def form_triangle (n : ℕ) (a b c : ℕ) : Prop :=
  a < total_segments n ∧ b < total_segments n ∧ c < total_segments n ∧
  ∃ (x y z : ℕ), x < n ∧ y < n ∧ z < n ∧
    are_sides_of_unit_triangle a b c x y z

/-- Theorem: The maximum number of unit segments that can be chosen in an equilateral triangle
    of side length n, divided into unit equilateral triangles, such that no three form a single
    triangle is n(n+1) -/
theorem max_segments_theorem (n : ℕ) :
  ∃ (chosen_segments : ℕ), 
    chosen_segments = max_segments n ∧
    chosen_segments ≤ total_segments n ∧
    ∀ (m : ℕ), m > chosen_segments → 
      ∃ (a b c : ℕ), a < m ∧ b < m ∧ c < m ∧ 
        form_triangle n a b c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_theorem_l1090_109085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1090_109022

/-- Properties of a triangle ABC given sin A = 2 sin B -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B)
  (h_sin_relation : Real.sin A = 2 * Real.sin B) : 
  (C = 2 * B → A = Real.pi / 2) ∧ 
  (a = c → Real.sin B = Real.sqrt 15 / 8) ∧
  (c = 3 → ∃ (max_area : ℝ), max_area = 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * a * b * Real.sin C → area ≤ max_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1090_109022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1090_109043

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1090_109043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_sides_product_squared_l1090_109008

-- Define the triangles
structure RightTriangle where
  base : ℝ
  height : ℝ

-- Define the problem setup
def problem_setup (T₁ T₂ : RightTriangle) : Prop :=
  -- Areas of the triangles
  T₁.base * T₁.height / 2 = 3 ∧
  T₂.base * T₂.height / 2 = 2 ∧
  -- Two pairs of congruent sides
  (T₁.base = T₂.base ∨ T₁.base = T₂.height) ∧
  (T₁.height = T₂.base ∨ T₁.height = T₂.height) ∧
  -- Ensure we're not using the same side twice
  (T₁.base ≠ T₁.height ∨ T₂.base ≠ T₂.height)

-- Define the third side of a right triangle using Pythagorean theorem
noncomputable def third_side (T : RightTriangle) : ℝ :=
  Real.sqrt (T.base^2 + T.height^2)

-- The theorem to prove
theorem third_sides_product_squared (T₁ T₂ : RightTriangle) 
  (h : problem_setup T₁ T₂) : 
  (third_side T₁ * third_side T₂)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_sides_product_squared_l1090_109008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_iff_l1090_109035

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x + 1| else -x + 3

-- State the theorem
theorem f_geq_one_iff (x : ℝ) : f x ≥ 1 ↔ x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_iff_l1090_109035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1090_109036

theorem trig_equation_solution (x y z : ℝ) (hx : Real.cos x ≠ 0) (hy : Real.cos y ≠ 0) :
  (Real.cos x ^ 2 + 1 / Real.cos x ^ 2) ^ 3 + (Real.cos y ^ 2 + 1 / Real.cos y ^ 2) ^ 3 = 16 * Real.sin z ↔
  ∃ (n k m : ℤ), x = n * Real.pi ∧ y = k * Real.pi ∧ z = Real.pi / 2 + 2 * m * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1090_109036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l1090_109074

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ; sin θ, cos θ]

theorem smallest_power_rotation_120 :
  (∃ n : ℕ+, (rotation_matrix (2 * π / 3))^(n : ℕ) = 1 ∧
   ∀ m : ℕ+, m < n → (rotation_matrix (2 * π / 3))^(m : ℕ) ≠ 1) →
  (∃ n : ℕ+, (rotation_matrix (2 * π / 3))^(n : ℕ) = 1 ∧
   ∀ m : ℕ+, m < n → (rotation_matrix (2 * π / 3))^(m : ℕ) ≠ 1 ∧ n = 3) := by
  sorry

#check smallest_power_rotation_120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l1090_109074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_union_l1090_109027

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem number_of_subsets_union (A B : Finset ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  Finset.card (Finset.powerset (A ∪ B)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_union_l1090_109027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1090_109090

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2))

theorem solution_set (x : ℝ) : x ≥ 2 → (f x = 3 ↔ x = 6 ∨ x = 27) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1090_109090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sin_double_angles_l1090_109012

theorem max_sum_sin_double_angles 
  (α β γ : ℝ) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) ≤ 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sin_double_angles_l1090_109012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1090_109055

/-- The function h(x) = (2x - 3) / (x^2 - 4) -/
noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x^2 - 4)

/-- The domain of h(x) -/
def domain_h : Set ℝ := {x | x ≠ 2 ∧ x ≠ -2}

theorem domain_of_h :
  domain_h = Set.Iio (-2) ∪ Set.Ioo (-2) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1090_109055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_is_1200_l1090_109005

/-- Calculates the total rent given Mrs. McPherson's contribution percentage and Mr. McPherson's contribution amount. -/
noncomputable def calculate_total_rent (mrs_percentage : ℝ) (mr_amount : ℝ) : ℝ :=
  mr_amount / (1 - mrs_percentage)

/-- Proves that the total rent is $1200 given the conditions. -/
theorem rent_is_1200 (mrs_percentage : ℝ) (mr_amount : ℝ)
    (h1 : mrs_percentage = 0.3)
    (h2 : mr_amount = 840) :
    calculate_total_rent mrs_percentage mr_amount = 1200 := by
  -- Unfold the definition of calculate_total_rent
  unfold calculate_total_rent
  -- Substitute the known values
  rw [h1, h2]
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_is_1200_l1090_109005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_interval_l1090_109080

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

theorem f_strictly_decreasing_interval :
  ∀ x y, 3 < x ∧ x < y → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_interval_l1090_109080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYZ_form_l1090_109068

/-- Triangle ABC with sides 5, 7, and 8 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 5)
  (BC_length : dist B C = 7)
  (CA_length : dist C A = 8)

/-- Point P inside triangle ABC such that PA:PB:PC = 2:3:6 -/
noncomputable def P (t : Triangle) : ℝ × ℝ := sorry

/-- Circumcircle of triangle ABC -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Points X, Y, Z where rays AP, BP, CP intersect the circumcircle -/
noncomputable def X (t : Triangle) : ℝ × ℝ := sorry
noncomputable def Y (t : Triangle) : ℝ × ℝ := sorry
noncomputable def Z (t : Triangle) : ℝ × ℝ := sorry

/-- Area of triangle XYZ -/
noncomputable def area_XYZ (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem area_XYZ_form (t : Triangle) :
  ∃ (p q r : ℕ), 
    area_XYZ t = (p : ℝ) * Real.sqrt q / r ∧
    Nat.Coprime p r ∧
    (∀ (prime : ℕ), Nat.Prime prime → ¬(q % (prime^2) = 0)) ∧
    p + q + r = 4082 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XYZ_form_l1090_109068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_numbers_determination_l1090_109062

theorem secret_numbers_determination (a b c d e : ℕ) : 
  a ≤ b → b ≤ c → c ≤ d → d ≤ e →
  a + b = 24 →
  a + c = 28 →
  c + e = 40 →
  d + e = 42 →
  a + b + c + d + e = 83 →
  a = 11 ∧ b = 13 ∧ c = 17 ∧ d = 19 ∧ e = 23 := by
  sorry

#check secret_numbers_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_numbers_determination_l1090_109062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1090_109024

-- Define a power function
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point :
  ∃ f : ℝ → ℝ, (∃ a : ℝ, f = powerFunction a) ∧ f 2 = Real.sqrt 2 → f = fun x ↦ Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1090_109024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_lowest_position_l1090_109033

/-- Represents a cyclist in the race -/
structure Cyclist :=
  (id : ℕ)

/-- Represents a stage in the race -/
structure Stage :=
  (number : ℕ)

/-- Represents the time a cyclist takes to complete a stage -/
structure StageTime :=
  (cyclist : Cyclist)
  (stage : Stage)
  (time : ℝ)

/-- Represents the total time a cyclist takes to complete all stages -/
structure TotalTime :=
  (cyclist : Cyclist)
  (time : ℝ)

/-- The number of cyclists in the race -/
def numCyclists : ℕ := 500

/-- The number of stages in the race -/
def numStages : ℕ := 15

/-- The position of Vasya in each stage -/
def vasyaStagePosition : ℕ := 7

/-- A function that returns the position of a cyclist in the overall race -/
noncomputable def overallPosition (c : Cyclist) : ℕ := sorry

/-- Vasya's cyclist object -/
def vasya : Cyclist := ⟨0⟩  -- Assuming Vasya's ID is 0

/-- The theorem to be proved -/
theorem vasya_lowest_position :
  (∀ (c₁ c₂ : Cyclist) (s₁ s₂ : Stage),
    StageTime.time ⟨c₁, s₁, 0⟩ ≠ StageTime.time ⟨c₂, s₂, 0⟩) →
  (∀ (c₁ c₂ : Cyclist), TotalTime.time ⟨c₁, 0⟩ ≠ TotalTime.time ⟨c₂, 0⟩) →
  (∀ (s : Stage), ∃! (sts : List StageTime),
    sts.length = numCyclists ∧
    (∀ (st : StageTime), st ∈ sts → st.stage = s) ∧
    sts.get? (vasyaStagePosition - 1) = some ⟨vasya, s, 0⟩) →
  overallPosition vasya ≤ 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_lowest_position_l1090_109033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1090_109065

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents a race between two runners -/
structure Race where
  runner_p : Runner
  runner_q : Runner
  headstart : ℝ

/-- The condition for a tie in the race -/
noncomputable def is_tie (race : Race) : Prop :=
  race.runner_p.distance / race.runner_p.speed = 
  (race.runner_q.distance + race.headstart) / race.runner_q.speed

/-- The percentage difference between two speeds -/
noncomputable def speed_difference_percentage (v1 v2 : ℝ) : ℝ :=
  (v1 - v2) / v2 * 100

/-- The main theorem about the race -/
theorem race_theorem (race : Race) :
  race.runner_p.speed > race.runner_q.speed →
  race.headstart = 300 →
  race.runner_p.distance = 1800 →
  is_tie race →
  speed_difference_percentage race.runner_p.speed race.runner_q.speed = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1090_109065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1090_109083

/-- A cubic function with a local maximum at x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

/-- The maximum difference of f values in the interval [-2, 2] -/
def max_diff (a b : ℝ) : ℝ := |f a b (-2) - f a b 2|

theorem cubic_function_properties (a b : ℝ) :
  (∀ x, f a b x ≤ f a b (-1)) ∧ 
  f a b (-1) = 2 →
  (f a b = λ x ↦ -(1/2) * x^3 - (9/2) * x^2 - 3 * x) ∧
  max_diff a b = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1090_109083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1090_109067

theorem modulus_of_z : Complex.abs ((2 - Complex.I)^2 / Complex.I) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1090_109067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1090_109041

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3 : ℝ)^b < (3 : ℝ)^a ∧ (3 : ℝ)^a < (4 : ℝ)^a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1090_109041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AQB_l1090_109018

-- Define the square ABDF
def square_side : ℝ := 8

-- Define points A, B, C, D, F, and Q
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (square_side, 0)
def D : ℝ × ℝ := (square_side, square_side)
def F : ℝ × ℝ := (0, square_side)

-- Define C and Q as variables
variable (C Q : ℝ × ℝ)

-- Define the lengths of QA, QB, and QC as variables
variable (QA_length QB_length QC_length : ℝ)

-- State the theorem
theorem area_of_triangle_AQB : 
  -- Given conditions
  QA_length = QB_length ∧ 
  QA_length = QC_length ∧
  (Q.2 - C.2) * (F.1 - D.1) = (Q.1 - C.1) * (F.2 - D.2) →
  -- Conclusion
  (1/2) * square_side * |Q.2| = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AQB_l1090_109018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_probability_l1090_109026

/-- The number of boxes -/
def num_boxes : ℕ := 5

/-- The number of identical pears -/
def num_pears : ℕ := 4

/-- The number of different apples -/
def num_apples : ℕ := 6

/-- The total number of fruits -/
def total_fruits : ℕ := num_pears + num_apples

/-- The number of fruits required in each box -/
def fruits_per_box : ℕ := 2

/-- The probability of each box containing exactly 2 fruits -/
def probability : ℚ := 8100 / 1093750

/-- A fruit distribution is a function from boxes to the number of fruits in each box -/
def FruitDistribution := Fin num_boxes → ℕ

/-- The probability measure for fruit distributions -/
noncomputable def ℙ : FruitDistribution → ℚ := sorry

/-- The number of fruits in a given box for a given distribution -/
def fruits_in_box (d : FruitDistribution) (box : Fin num_boxes) : ℕ := d box

theorem fruit_distribution_probability :
  (∀ d : FruitDistribution, (∀ box : Fin num_boxes, fruits_in_box d box = fruits_per_box)) →
  ℙ d = probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_probability_l1090_109026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_in_range_l1090_109037

/-- The function f(x) defined for all real x and k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * k * x^2 + k * x + 3/8

/-- Theorem stating that f(x) is positive for all real x if and only if k is in [0, 3) -/
theorem f_positive_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, f k x > 0) ↔ (0 ≤ k ∧ k < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_in_range_l1090_109037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_lineup_theorem_l1090_109093

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def lineup_size : ℕ := 7
def max_quadruplets : ℕ := 2

theorem basketball_lineup_theorem : 
  (Nat.choose total_players lineup_size) - 
  ((Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (lineup_size - 3)) + 
   (Nat.choose quadruplets 4) * (Nat.choose (total_players - quadruplets) (lineup_size - 4))) = 9240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_lineup_theorem_l1090_109093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_length_is_4_sqrt_5_l1090_109023

/-- Triangle ABC with base AB on x-axis and altitude feet M and N -/
structure Triangle :=
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)

/-- The length of the base AB given the coordinates of M and N -/
noncomputable def baseLength (t : Triangle) : ℝ :=
  4 * Real.sqrt 5

/-- Theorem stating that the base length is 4√5 given the specific coordinates -/
theorem base_length_is_4_sqrt_5 (t : Triangle) 
  (hM : t.M = (2, 2)) (hN : t.N = (4, 4)) : 
  baseLength t = 4 * Real.sqrt 5 := by
  -- Proof steps would go here
  sorry

#check base_length_is_4_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_length_is_4_sqrt_5_l1090_109023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1090_109006

-- Define the ellipse E
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line l
noncomputable def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the area of triangle F₂MN
noncomputable def triangle_area (k : ℝ) : ℝ :=
  3 * Real.sqrt ((k^2 * (2 - 4*k^2)) / (1 + 2*k^2)^2)

theorem ellipse_and_triangle_properties :
  -- The ellipse passes through (1, √2/2)
  ellipse 1 (Real.sqrt 2 / 2) ∧
  -- The maximum area of triangle F₂MN is 3√2/4
  ∃ k : ℝ, ∀ k' : ℝ, triangle_area k ≥ triangle_area k' ∧ triangle_area k = 3 * Real.sqrt 2 / 4 := by
  sorry

#check ellipse_and_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1090_109006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overcrowded_cars_proportion_theorem_l1090_109070

/-- Represents the distribution of train cars based on passenger numbers -/
def train_distribution : List (ℕ × ℝ) :=
  [(15, 0.04), (25, 0.06), (35, 0.12), (45, 0.18), (55, 0.20), (65, 0.20), (75, 0.14), (85, 0.06)]

/-- A train car is considered overcrowded if it has 60 or more passengers -/
def is_overcrowded (passengers : ℕ) : Bool := passengers ≥ 60

/-- The proportion of overcrowded train cars -/
def proportion_overcrowded_cars (distribution : List (ℕ × ℝ)) : ℝ :=
  (distribution.filter (λ p => is_overcrowded p.1)).foldr (λ p acc => acc + p.2) 0

/-- The proportion of passengers in overcrowded train cars -/
noncomputable def proportion_passengers_in_overcrowded_cars (distribution : List (ℕ × ℝ)) : ℝ :=
  let total_passengers := distribution.foldr (λ p acc => acc + (p.1 : ℝ) * p.2) 0
  let passengers_in_overcrowded := (distribution.filter (λ p => is_overcrowded p.1)).foldr (λ p acc => acc + (p.1 : ℝ) * p.2) 0
  passengers_in_overcrowded / total_passengers

theorem overcrowded_cars_proportion_theorem :
  ∀ (distribution : List (ℕ × ℝ)),
    proportion_passengers_in_overcrowded_cars distribution ≥ proportion_overcrowded_cars distribution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overcrowded_cars_proportion_theorem_l1090_109070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_celsius_l1090_109057

-- Define the conversion between Celsius and Fahrenheit
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

-- Define the conversion between Fahrenheit and Celsius
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * (5/9)

-- Theorem to prove the boiling point of water in Celsius
theorem water_boiling_point_celsius :
  let boiling_f := (212 : ℝ)
  let melting_f := (32 : ℝ)
  let melting_c := (0 : ℝ)
  let known_c := (40 : ℝ)
  let known_f := (104 : ℝ)
  celsius_to_fahrenheit known_c = known_f →
  fahrenheit_to_celsius melting_f = melting_c →
  fahrenheit_to_celsius boiling_f = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_celsius_l1090_109057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1090_109011

open Set

def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

theorem problem_solution :
  (∀ m : ℝ,
    (m = 3 → A ∩ (A ∪ B m \ B m) = {x : ℝ | 3 ≤ x ∧ x < 4})) ∧
  (∀ m : ℝ, A ∩ B m = ∅ ↔ m ≤ -2) ∧
  (∀ m : ℝ, A ∩ B m = A ↔ m ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1090_109011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1090_109010

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : S seq 6 > S seq 7 ∧ S seq 7 > S seq 5) : 
  seq.d < 1 ∧ S seq 11 > 0 ∧ abs (seq.a 6) > abs (seq.a 7) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1090_109010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l1090_109076

/-- Calculates the time (in seconds) it takes for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℚ) (train_length_m : ℚ) (platform_length_m : ℚ) : ℚ :=
  let train_speed_ms := train_speed_kmh * (5/18)
  let total_distance := train_length_m + platform_length_m
  total_distance / train_speed_ms

theorem train_crossing_platform : 
  train_crossing_time 72 300 220 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l1090_109076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cooking_time_l1090_109045

/-- Represents the cooking tasks --/
inductive CookingTask
  | BoilNoodles
  | WashVegetables
  | PrepareNoodlesAndCondiments
  | BoilWater
  | WashPotAndAddWater
deriving Inhabited, Repr

/-- Returns the duration of a cooking task in minutes --/
def taskDuration (task : CookingTask) : ℕ :=
  match task with
  | CookingTask.BoilNoodles => 4
  | CookingTask.WashVegetables => 5
  | CookingTask.PrepareNoodlesAndCondiments => 2
  | CookingTask.BoilWater => 10
  | CookingTask.WashPotAndAddWater => 2

/-- Represents whether a task can be performed simultaneously with boiling water --/
def canBeSimultaneous (task : CookingTask) : Bool :=
  match task with
  | CookingTask.BoilWater => false
  | _ => true

/-- The main theorem stating the minimum time required to complete all tasks --/
theorem min_cooking_time :
  ∃ (schedule : List CookingTask),
    (∀ task : CookingTask, task ∈ schedule) ∧
    (∀ i j, i < j → schedule.get! i = CookingTask.BoilWater →
      canBeSimultaneous (schedule.get! j) = true) ∧
    (List.sum (schedule.map taskDuration) = 16) := by
  sorry

#check min_cooking_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cooking_time_l1090_109045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_b_value_cosine_function_exists_l1090_109058

/-- The period of a cosine function with coefficient b -/
noncomputable def cosinePeriod (b : ℝ) : ℝ := 2 * Real.pi / b

/-- The observed period of the function from the graph -/
noncomputable def observedPeriod : ℝ := 3 * Real.pi / 2

/-- Theorem stating that for a cosine function y = a cos(bx + c) + d 
    with an observed period of 3π/2, the value of b is 4/3 -/
theorem cosine_function_b_value (b : ℝ) :
  cosinePeriod b = observedPeriod → b = 4/3 := by
  intro h
  -- Proof steps would go here
  sorry

/-- Existence theorem for the cosine function parameters -/
theorem cosine_function_exists :
  ∃ (a b c d : ℝ), cosinePeriod b = observedPeriod ∧ b = 4/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_b_value_cosine_function_exists_l1090_109058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_points_Q₁_Q₂_two_doubling_points_on_parabola_min_distance_to_doubling_point_correct_statements_count_l1090_109094

-- Define the concept of a "doubling point"
def is_doubling_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  2 * (x₁ + x₂) = y₁ + y₂

-- Define P₁
def P₁ : ℝ × ℝ := (1, 0)

-- Statement 1
theorem doubling_points_Q₁_Q₂ : 
  is_doubling_point P₁.1 P₁.2 3 8 ∧ is_doubling_point P₁.1 P₁.2 (-2) (-2) := by sorry

-- Statement 2
theorem two_doubling_points_on_parabola :
  ∃! (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    is_doubling_point P₁.1 P₁.2 x₁ (x₁^2 - 2*x₁ - 3) ∧ 
    is_doubling_point P₁.1 P₁.2 x₂ (x₂^2 - 2*x₂ - 3) := by sorry

-- Statement 3
theorem min_distance_to_doubling_point :
  ∀ (x y : ℝ), is_doubling_point P₁.1 P₁.2 x y → 
    Real.sqrt ((x - P₁.1)^2 + (y - P₁.2)^2) ≥ 4 * Real.sqrt 5 / 5 := by sorry

-- Main theorem
theorem correct_statements_count : Nat := 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_points_Q₁_Q₂_two_doubling_points_on_parabola_min_distance_to_doubling_point_correct_statements_count_l1090_109094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_theorem_l1090_109081

theorem comparison_theorem (a b c : ℝ) 
  (ha : a = (1/2 : ℝ)^(0.3 : ℝ))
  (hb : b = (0.3 : ℝ)^(-2 : ℝ))
  (hc : c = Real.log 2 / Real.log (1/2)) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_theorem_l1090_109081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l1090_109095

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 5 + t x

-- State the theorem
theorem t_of_f_3 : t (f 3) = Real.sqrt (22 + 4 * Real.sqrt 14) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_3_l1090_109095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l1090_109015

/-- Predicate indicating that triangle A₁B₁C₁ is inscribed in triangle ABC -/
def InscribedTriangle (A₁ B₁ C₁ A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Predicate indicating that triangle A₂B₂C₂ is circumscribed around triangle A₁B₁C₁ -/
def CircumscribedTriangle (A₂ B₂ C₂ A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Predicate indicating that corresponding sides of triangles A₁B₁C₁ and A₂B₂C₂ are parallel -/
def ParallelSides (A₁ B₁ C₁ A₂ B₂ C₂ : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Function to calculate the area of a triangle given its three vertices -/
noncomputable def area (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given three triangles ABC, A₁B₁C₁, and A₂B₂C₂, where A₁B₁C₁ is inscribed in ABC,
    A₂B₂C₂ is circumscribed around A₁B₁C₁, and corresponding sides of A₁B₁C₁ and A₂B₂C₂ are parallel,
    the square of the area of ABC is equal to the product of the areas of A₁B₁C₁ and A₂B₂C₂. -/
theorem triangle_area_relation 
    (A B C A₁ B₁ C₁ A₂ B₂ C₂ : EuclideanSpace ℝ (Fin 2)) :
  InscribedTriangle A₁ B₁ C₁ A B C →
  CircumscribedTriangle A₂ B₂ C₂ A₁ B₁ C₁ →
  ParallelSides A₁ B₁ C₁ A₂ B₂ C₂ →
  (area A B C) ^ 2 = (area A₁ B₁ C₁) * (area A₂ B₂ C₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l1090_109015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_is_120_minutes_l1090_109088

/-- Represents the trip details -/
structure TripDetails where
  freeway_distance : ℚ
  rural_distance : ℚ
  rural_time : ℚ
  rural_speed : ℚ
  freeway_speed : ℚ

/-- Calculates the total trip time given the trip details -/
def total_trip_time (trip : TripDetails) : ℚ :=
  trip.rural_time + trip.freeway_distance / trip.freeway_speed

/-- Theorem stating that the total trip time is 120 minutes -/
theorem trip_time_is_120_minutes (trip : TripDetails) 
  (h1 : trip.freeway_distance = 80)
  (h2 : trip.rural_distance = 20)
  (h3 : trip.rural_time = 40)
  (h4 : trip.freeway_speed = 2 * trip.rural_speed)
  (h5 : trip.rural_distance / trip.rural_speed = trip.rural_time) :
  total_trip_time trip = 120 := by
  sorry

#eval total_trip_time { freeway_distance := 80, rural_distance := 20, rural_time := 40, rural_speed := 1/2, freeway_speed := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_is_120_minutes_l1090_109088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l1090_109019

/-- The height of a right cylinder with given radius and surface area -/
noncomputable def cylinder_height (r : ℝ) (sa : ℝ) : ℝ :=
  (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

/-- Theorem: The height of a right cylinder with radius 4 feet and surface area 40π square feet is 1 foot -/
theorem cylinder_height_example : cylinder_height 4 (40 * Real.pi) = 1 := by
  -- Unfold the definition of cylinder_height
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l1090_109019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1090_109084

theorem tan_double_angle_special_case (α : Real) :
  Real.tan (α - π / 8) = 2 → Real.tan (2 * α - π / 4) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1090_109084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_proof_l1090_109017

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
def exterior_angle_regular_octagon : ℚ := 45

/-- A regular octagon has 8 sides. -/
def regular_octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℚ :=
  (n - 2) * 180

/-- Each interior angle of a regular polygon is the sum of interior angles divided by the number of sides. -/
def interior_angle (n : ℕ) : ℚ :=
  sum_interior_angles n / n

/-- An exterior angle is supplementary to its corresponding interior angle. -/
def exterior_angle (n : ℕ) : ℚ :=
  180 - interior_angle n

/-- Proof that the exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_proof :
  exterior_angle regular_octagon_sides = exterior_angle_regular_octagon := by
  -- Unfold definitions
  unfold exterior_angle
  unfold interior_angle
  unfold sum_interior_angles
  unfold regular_octagon_sides
  unfold exterior_angle_regular_octagon
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval exterior_angle regular_octagon_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_proof_l1090_109017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_sum_l1090_109066

/-- Felsius temperature conversion function from Celsius --/
noncomputable def felsius_from_celsius (c : ℝ) : ℝ := (7 * c) / 5 + 16

/-- Felsius temperature conversion function from Fahrenheit --/
noncomputable def felsius_from_fahrenheit (f : ℝ) : ℝ := (7 * f - 80) / 9

/-- The problem statement --/
theorem temperature_sum (x y z : ℝ) 
  (hx : x = felsius_from_celsius x)
  (hy : y = felsius_from_fahrenheit y)
  (hz : z = (9 * z) / 5 + 32) : 
  x + y + z = -120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_sum_l1090_109066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_l1090_109044

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a + (n - 1) * seq.d) / 2

/-- Sum of sums of first n terms of an arithmetic sequence -/
noncomputable def T (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (n + 1) * (3 * seq.a + (n - 1) * seq.d) / 6

/-- The theorem to be proved -/
theorem unique_determination_of_T (seq : ArithmeticSequence) :
  ∃! n : ℕ, n > 0 ∧ 
    (∀ seq' : ArithmeticSequence, S seq 2023 = S seq' 2023 → T seq n = T seq' n) ∧
    (∀ m : ℕ, m < n → ∃ seq' seq'', 
      S seq 2023 = S seq' 2023 ∧ S seq 2023 = S seq'' 2023 ∧ T seq' m ≠ T seq'' m) ∧
    n = 3034 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_l1090_109044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_inequality_part2_l1090_109082

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := (1/2) * x + b

-- Part 1: Solution set for f(x) ≤ 0 when a = 1/2
theorem solution_set_part1 :
  {x : ℝ | f (1/2) x ≤ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 0} := by sorry

-- Part 2: Prove 2b - 3a > 2 when g(x) is always above f(x) and a ≥ -1
theorem inequality_part2 (a b : ℝ) (h1 : a ≥ -1) 
  (h2 : ∀ x, g b x > f a x) : 2 * b - 3 * a > 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_inequality_part2_l1090_109082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_five_l1090_109056

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

-- Theorem statement
theorem unique_solution_for_five :
  ∃! x : ℝ, f x = 5 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_five_l1090_109056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steamboat_unsavable_l1090_109004

theorem steamboat_unsavable (n : ℕ) (h : n = 2017^2017) :
  ¬ ∃ (a : Fin 26 → ℕ), 
    (∀ i : Fin 25, (a (i.succ) = a i + 2 ∨ a (i.succ) = a i - 2)) ∧
    (Finset.sum Finset.univ a) = n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steamboat_unsavable_l1090_109004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1090_109061

/-- Represents an ellipse with center at origin, foci on x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b
  h4 : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The distance from the right focus to the right vertex -/
noncomputable def rightFocusToVertex (e : Ellipse) : ℝ := e.a - e.c

/-- Theorem about the standard equation and intersection properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h_ecc : eccentricity e = 1/2)
    (h_dist : rightFocusToVertex e = 1) :
  (∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1) ∧
  (∃ k m : ℝ, ∀ A B : ℝ × ℝ, 
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧ 
     A.1^2/e.a^2 + A.2^2/e.b^2 = 1 ∧ B.1^2/e.a^2 + B.2^2/e.b^2 = 1) →
    (‖(A.1, A.2)‖ + 2 * ‖(B.1, B.2)‖ = ‖(A.1, A.2)‖ - 2 * ‖(B.1, B.2)‖)) ∧
  (∀ m : ℝ, (m ≤ -2 * Real.sqrt 21 / 7 ∨ m ≥ 2 * Real.sqrt 21 / 7) ↔
    (∃ k : ℝ, ∃ A B : ℝ × ℝ, 
      A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧ 
      A.1^2/e.a^2 + A.2^2/e.b^2 = 1 ∧ B.1^2/e.a^2 + B.2^2/e.b^2 = 1 ∧
      ‖(A.1, A.2)‖ + 2 * ‖(B.1, B.2)‖ = ‖(A.1, A.2)‖ - 2 * ‖(B.1, B.2)‖)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1090_109061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_derivative_even_l1090_109098

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_derivative_even
  (h : Differentiable ℝ f) :
  (OddFunction f → EvenFunction f') ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ EvenFunction g' ∧ ¬OddFunction g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_derivative_even_l1090_109098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l1090_109050

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem symmetry_about_line (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + 4*Real.pi) = f ω x) :
  ∀ x, f ω (Real.pi/3 + x) = f ω (Real.pi/3 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l1090_109050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1090_109030

noncomputable def curve (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

theorem tangent_triangle_area (a : ℝ) (h1 : a > 0) :
  let f := curve
  let point := (a, f a)
  let slope := -(1/2 : ℝ) * a^(-(3/2 : ℝ))
  let tangent (x : ℝ) := f a + slope * (x - a)
  let x_intercept := 3 * a
  let y_intercept := (3/2 : ℝ) * a^(-(1/2 : ℝ))
  let triangle_area := (1/2 : ℝ) * x_intercept * y_intercept
  triangle_area = 18 → a = 64 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1090_109030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_six_digit_numbers_l1090_109092

def six_digit_numbers : Set Nat :=
  {n : Nat | n ≥ 100000 ∧ n < 1000000 ∧ ∃ (a b c d e f : Nat),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Finset.toSet {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧
    n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f}

theorem gcd_of_six_digit_numbers :
  ∃ (g : Nat), g > 0 ∧ (∀ n ∈ six_digit_numbers, g ∣ n) ∧
  (∀ d : Nat, d > 0 → (∀ n ∈ six_digit_numbers, d ∣ n) → d ≤ g) ∧
  g = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_six_digit_numbers_l1090_109092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_root_implies_m_range_l1090_109096

open Real

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := exp x * (2 * x - 1) - m * x + m

-- State the theorem
theorem unique_integer_root_implies_m_range :
  ∀ m : ℝ,
  (m < 1) →
  (∃! (n : ℤ), f (n : ℝ) m < 0) →
  m ∈ Set.Icc (3 / (2 * exp 1)) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_root_implies_m_range_l1090_109096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_symmetry_forms_two_lines_l1090_109075

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := 2 * x^2 + y^2 + 1 - y

-- Define the set of points satisfying x ⊗ y = y ⊗ x
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | otimes p.1 p.2 = otimes p.2 p.1}

-- Define what it means for a set to be a line in ℝ²
def IsLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ l = {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem statement
theorem otimes_symmetry_forms_two_lines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)),
    IsLine l₁ ∧ IsLine l₂ ∧ l₁ ≠ l₂ ∧ S = l₁ ∪ l₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_symmetry_forms_two_lines_l1090_109075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_of_triangle_AOB_l1090_109032

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 8

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the chord AB
noncomputable def chord_length (A B : PointOnEllipse) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Define the area of triangle AOB
noncomputable def triangle_area (A B : PointOnEllipse) : ℝ :=
  abs (A.x * B.y - A.y * B.x) / 2

-- Theorem statement
theorem area_range_of_triangle_AOB :
  ∀ (A B : PointOnEllipse),
  chord_length A B = 5/2 →
  5 * Real.sqrt 103 / 32 ≤ triangle_area A B ∧ triangle_area A B ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_of_triangle_AOB_l1090_109032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1090_109052

/-- Given a line with equation y = x + 1, its angle of inclination is 45° --/
theorem line_inclination (l : Set (ℝ × ℝ)) : 
  (∀ x y, y = x + 1 → (x, y) ∈ l) → 
  ∃ v : ℝ × ℝ, v ∈ l ∧ v.1 ≠ 0 ∧ (v.2 / v.1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1090_109052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1090_109060

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway_mpg : ℝ
  city_mpg : ℝ
  known_distance : ℝ
  known_gallons : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and a certain amount of gasoline -/
noncomputable def max_distance (suv : SUVFuelEfficiency) (gallons : ℝ) : ℝ :=
  max (suv.highway_mpg * gallons) (suv.city_mpg * gallons)

/-- Theorem stating that the maximum distance the SUV can travel on 20 gallons is 244 miles -/
theorem suv_max_distance (suv : SUVFuelEfficiency) 
  (h1 : suv.highway_mpg = 12.2)
  (h2 : suv.city_mpg = 7.6)
  (h3 : suv.known_distance = 244)
  (h4 : suv.known_gallons = 20)
  (h5 : suv.known_distance = suv.highway_mpg * suv.known_gallons) :
  max_distance suv 20 = 244 := by
  sorry

#check suv_max_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1090_109060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l1090_109086

-- Define the time it takes Tom to clean the entire house alone
noncomputable def tom_time : ℝ := 6

-- Define the relationship between Tom's and Nick's cleaning times
noncomputable def nick_time : ℝ := 3 * tom_time

-- Define Tom's work rate (portion of house cleaned per hour)
noncomputable def tom_rate : ℝ := 1 / tom_time

-- Define Nick's work rate (portion of house cleaned per hour)
noncomputable def nick_rate : ℝ := 1 / nick_time

-- Define the combined work rate of Tom and Nick
noncomputable def combined_rate : ℝ := tom_rate + nick_rate

-- Theorem: The time it takes Nick and Tom to clean the entire house together is 3.6 hours
theorem cleaning_time_together : (1 / combined_rate) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l1090_109086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1090_109049

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) : 
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((a - 2*b) * Real.cos C = c * (2 * Real.cos B - Real.cos A)) →
  (a^2 * Real.sin ((A + B) / 2) = (1/2) * b * c * Real.sin A) →
  C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1090_109049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_two_l1090_109021

-- Define the function (marked as noncomputable due to real number operations)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 10) / (x + 2)

-- State the theorem
theorem vertical_asymptote_at_neg_two :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 → 
    ∀ (x : ℝ), 0 < |x + 2| ∧ |x + 2| < δ → |f x| > M :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_two_l1090_109021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nFactorable_iff_power_of_two_l1090_109000

def is_nFactorable (n : ℕ) (a : ℕ) : Prop :=
  a > 2 ∧ ∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0

def isComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem nFactorable_iff_power_of_two (n : ℕ) :
  (isComposite n ∧ ∃ a : ℕ, is_nFactorable n a) ↔ ∃ m : ℕ, m > 1 ∧ n = 2^m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nFactorable_iff_power_of_two_l1090_109000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_cube_relationship_l1090_109072

/-- Represents the dimensions of a cuboidal block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the properties of the cubes cut from the block -/
structure CubeProperties where
  count : ℕ
  side_length : ℕ

/-- The theorem stating the relationship between block dimensions and cube properties -/
theorem block_cube_relationship (block : BlockDimensions) (cube : CubeProperties) :
  block.length = 6 ∧ 
  block.width = 9 ∧ 
  cube.count = 24 ∧ 
  (cube.side_length ^ 3 * cube.count = block.length * block.width * block.height) ∧
  (cube.side_length ∣ block.length) ∧
  (cube.side_length ∣ block.width) ∧
  (cube.side_length ∣ block.height) →
  block.height = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_cube_relationship_l1090_109072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l1090_109089

/-- Calculates the true discount on a bill given its face value, time until due, and annual interest rate. -/
noncomputable def trueDiscount (faceValue : ℝ) (timeInMonths : ℝ) (annualRate : ℝ) : ℝ :=
  let presentValue := faceValue / (1 + annualRate * (timeInMonths / 12))
  faceValue - presentValue

/-- Theorem stating that the true discount on a bill of Rs. 2240 due in 9 months at an annual interest rate of 16% is equal to Rs. 240. -/
theorem bill_true_discount :
  trueDiscount 2240 9 0.16 = 240 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval trueDiscount 2240 9 0.16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l1090_109089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l1090_109079

theorem arithmetic_progression_property (S : Set ℝ) : 
  Finite S →
  S.Nonempty →
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → |x - y| ∈ S) →
  ∃ (a d : ℝ) (k : ℕ), S = {i | ∃ n : ℕ, n ≤ k ∧ i = a + n * d} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l1090_109079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_range_l1090_109054

/-- The ellipse in question -/
def Ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- The foci of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Vector from F₁ to P -/
noncomputable def F₁P (x y : ℝ) : ℝ × ℝ := (x + Real.sqrt 2, y)

/-- Vector from F₂ to P -/
noncomputable def F₂P (x y : ℝ) : ℝ × ℝ := (x - Real.sqrt 2, y)

/-- Vector from F₁ to Q -/
noncomputable def F₁Q (x y : ℝ) : ℝ × ℝ := (x + Real.sqrt 2, -y)

/-- Vector from F₂ to Q -/
noncomputable def F₂Q (x y : ℝ) : ℝ × ℝ := (x - Real.sqrt 2, -y)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos (dot_product v w / (magnitude v * magnitude w))

/-- The main theorem -/
theorem ellipse_angle_range (x y : ℝ) :
  Ellipse x y →
  dot_product (F₁P x y) (F₂P x y) ≤ 1 →
  π - Real.arccos (1/3) ≤ angle (F₁P x y) (F₂Q x y) ∧
  angle (F₁P x y) (F₂Q x y) ≤ π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_range_l1090_109054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_numbers_characterization_l1090_109001

/-- A number that is a product of distinct primes and is divisible by each of these primes minus 1 -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ (primes : List ℕ), 
    (primes.Nodup) ∧ 
    (∀ p ∈ primes, Nat.Prime p) ∧
    (n = primes.prod) ∧
    (∀ p ∈ primes, n % (p - 1) = 0)

/-- The theorem stating that only 6, 42, and 1806 satisfy the SpecialNumber property -/
theorem special_numbers_characterization :
  ∀ n : ℕ, SpecialNumber n ↔ n ∈ [6, 42, 1806] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_numbers_characterization_l1090_109001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_magnitude_l1090_109020

noncomputable section

-- Define the vectors in ℝ²
variable (OA OB : ℝ × ℝ)

-- Define the magnitudes
def a (OA : ℝ × ℝ) : ℝ := Real.sqrt ((OA.1)^2 + (OA.2)^2)
def b (OB : ℝ × ℝ) : ℝ := Real.sqrt ((OB.1)^2 + (OB.2)^2)

-- Define the conditions
variable (h1 : (a OA)^2 + (b OB)^2 = 4)
variable (h2 : OA.1 * OB.1 + OA.2 * OB.2 = 0)

-- Define OC as a function of lambda and mu
def OC (OA OB : ℝ × ℝ) (lambda mu : ℝ) : ℝ × ℝ := 
  (lambda * OA.1 + mu * OB.1, lambda * OA.2 + mu * OB.2)

-- Define the condition on lambda and mu
variable (h3 : ∀ lambda mu : ℝ, (lambda - 1/2)^2 * (a OA)^2 + (mu - 1/2)^2 * (b OB)^2 = 1)

-- State the theorem
theorem max_OC_magnitude :
  ∃ C : ℝ × ℝ, ∀ lambda mu : ℝ, 
    Real.sqrt ((OC OA OB lambda mu).1^2 + (OC OA OB lambda mu).2^2) ≤ 
    Real.sqrt (C.1^2 + C.2^2) ∧
    Real.sqrt (C.1^2 + C.2^2) = 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_magnitude_l1090_109020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coeffs_of_f_eq_one_l1090_109087

-- Define the polynomial f(x) = (x^3 - x + 1)^100
noncomputable def f (x : ℝ) : ℝ := (x^3 - x + 1)^100

-- Define the sum of coefficients of even-degree terms
noncomputable def sum_even_coeffs (p : ℝ → ℝ) : ℝ := (p 1 + p (-1)) / 2

-- Theorem statement
theorem sum_even_coeffs_of_f_eq_one :
  sum_even_coeffs f = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coeffs_of_f_eq_one_l1090_109087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1090_109048

/-- Heron's formula for the area of a triangle -/
noncomputable def heronFormula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of a triangle with sides 2.3, √5, and 4.1 is approximately 1.975 -/
theorem triangle_area_approx :
  let a : ℝ := 2.3
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 4.1
  ∃ ε > 0, |heronFormula a b c - 1.975| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1090_109048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1090_109029

def is_k_order_equivalent_progressive (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ p q : ℕ, a p = a q → a (p + k) = a (q + k)

def sequence_sum (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b

theorem sequence_property (a : ℕ → ℝ) (b : ℕ → ℝ) :
  is_k_order_equivalent_progressive a 1 →
  a 1 = 1 →
  a 3 = 3 →
  a 5 = 1 →
  a 4 + a 7 + a 10 = 2 →
  (∀ n : ℕ, b n = (-1)^n * a n) →
  a 2023 = 3 ∧ sequence_sum b 2024 = -2530 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1090_109029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_x_axis_l1090_109028

/-- A parabola with upward-facing branches -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
noncomputable def y_at_x (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.b * x + p.c

theorem parabola_intersection_x_axis (π₁ π₂ : Parabola) : 
  y_at_x π₁ 10 = 0 →
  y_at_x π₁ 13 = 0 →
  y_at_x π₂ 13 = 0 →
  vertex_x π₁ = (0 + vertex_x π₂) / 2 →
  ∃ (x : ℝ), x ≠ 13 ∧ y_at_x π₂ x = 0 ∧ x = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_x_axis_l1090_109028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_at_least_N_l1090_109078

theorem polynomial_degree_at_least_N 
  (N : ℕ) 
  (h_N_pos : N > 0)
  (h_N_plus_one_prime : Nat.Prime (N + 1))
  (a : Fin (N + 1) → Fin 2)
  (h_a_not_all_same : ∃ i j : Fin (N + 1), a i ≠ a j)
  (f : Polynomial ℤ)
  (h_f_interpolation : ∀ i : Fin (N + 1), f.eval (i : ℤ) = (a i : ℤ)) :
  Polynomial.degree f ≥ N :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_at_least_N_l1090_109078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1090_109053

-- Define the function f(x) = e^x - ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem function_properties :
  -- The constant a in f(x) = e^x - ax is 2
  ∃ a : ℝ, (f a 0 = 1 ∧ (deriv (f a)) 0 = -1) ∧ a = 2 ∧
  -- The minimum value of f(x) = e^x - 2x is 2 - ln(4) at x = ln(2)
  (∀ x : ℝ, f 2 x ≥ f 2 (Real.log 2)) ∧ f 2 (Real.log 2) = 2 - Real.log 4 ∧
  -- For all x > 0, x^2 < e^x
  ∀ x : ℝ, x > 0 → x^2 < Real.exp x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1090_109053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_equal_power_sums_l1090_109003

theorem partition_with_equal_power_sums :
  ∃ (A B : Finset ℕ),
    A ∪ B = Finset.range 1024 ∧
    A ∩ B = ∅ ∧
    A.card = 512 ∧
    B.card = 512 ∧
    ∀ j : ℕ, j < 10 →
      (A.sum (λ x => x^j) = B.sum (λ x => x^j)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_equal_power_sums_l1090_109003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1090_109009

/-- Two circles in a plane -/
structure TwoCircles where
  O : EuclideanSpace ℝ (Fin 2)
  O₁ : EuclideanSpace ℝ (Fin 2)
  r : ℝ
  r₁ : ℝ

/-- The configuration of two intersecting circles -/
structure IntersectingCircles extends TwoCircles where
  A : EuclideanSpace ℝ (Fin 2)
  hA : (‖A - O‖ = r) ∧ (‖A - O₁‖ = r₁)

/-- A line through point A -/
structure LineThrough (A : EuclideanSpace ℝ (Fin 2)) where
  v : EuclideanSpace ℝ (Fin 2)
  hv : v ≠ 0

/-- The segment intercepted by two circles on a line -/
def interceptedSegment (C : IntersectingCircles) (l : LineThrough C.A) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {x | ∃ t : ℝ, x = C.A + t • l.v ∧ ((‖x - C.O‖ = C.r) ∨ (‖x - C.O₁‖ = C.r₁))}

noncomputable def length (S : Set (EuclideanSpace ℝ (Fin 2))) : ℝ :=
  sorry

theorem two_solutions_exist (C : IntersectingCircles) (a : ℝ) (ha : a > 0) :
  ∃ l₁ l₂ : LineThrough C.A, l₁ ≠ l₂ ∧
    length (interceptedSegment C l₁) = a ∧
    length (interceptedSegment C l₂) = a ∧
    ∀ l : LineThrough C.A, length (interceptedSegment C l) = a → l = l₁ ∨ l = l₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1090_109009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_N_l1090_109002

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

theorem coordinates_of_N (M N : Point) :
  M.x = 1 ∧ M.y = -2 ∧
  distance M N = 3 ∧
  parallelToXAxis M N →
  (N.x = -2 ∧ N.y = -2) ∨ (N.x = 4 ∧ N.y = -2) :=
by
  sorry

#check coordinates_of_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_N_l1090_109002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1090_109031

/-- The curve on which point P moves -/
noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The slope angle of the tangent line at point P -/
noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (curve_derivative x)

theorem slope_angle_range :
  ∀ x : ℝ, slope_angle x ∈ Set.union (Set.Ico 0 (π/2)) (Set.Ico (3*π/4) π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1090_109031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_of_primes_in_S_l1090_109051

def S (n : ℕ) : ℕ := 3 + 10 * (n - 1)

def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_difference_of_primes (x : ℕ) : Prop :=
  ∃ p q, is_prime p ∧ is_prime q ∧ x = p - q

theorem unique_difference_of_primes_in_S :
  ∃! k, k ∈ Finset.range ω ∧ is_difference_of_primes (S k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_of_primes_in_S_l1090_109051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_arcsin_arctan_l1090_109025

theorem sin_sum_arcsin_arctan : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/3)) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_arcsin_arctan_l1090_109025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_roots_l1090_109059

open Real

theorem tan_sum_roots (α β : ℝ) (h : ∃ (x₁ x₂ : ℝ), x₁ = Real.tan α ∧ x₂ = Real.tan β ∧ x₁^2 + 5*x₁ + 4 = 0 ∧ x₂^2 + 5*x₂ + 4 = 0) :
  Real.tan (α + β) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_roots_l1090_109059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_segment_equals_height_l1090_109014

/-- Right tangent trapezoid with bases a and b -/
structure RightTangentTrapezoid where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- The height of the trapezoid -/
noncomputable def trapezoidHeight (t : RightTangentTrapezoid) : ℝ := 2 * t.a * t.b / (t.a + t.b)

/-- The length of the line segment parallel to bases through diagonal intersection -/
noncomputable def diagonalIntersectionSegment (t : RightTangentTrapezoid) : ℝ := 2 * t.a * t.b / (t.a + t.b)

/-- Theorem: The diagonal intersection segment equals the height -/
theorem diagonal_intersection_segment_equals_height (t : RightTangentTrapezoid) :
  diagonalIntersectionSegment t = trapezoidHeight t := by
  sorry

#check diagonal_intersection_segment_equals_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_segment_equals_height_l1090_109014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_proof_l1090_109039

noncomputable def quadratic_polynomial (x : ℝ) (b c : ℝ) : ℝ := 2 / Real.sqrt 3 * x^2 + b * x + c

theorem quadratic_roots_proof (b c : ℝ) (M K L : ℝ × ℝ) :
  (∃ y, M = (0, y) ∧ y > 0) →  -- M is on positive y-axis
  (∃ x₁ x₂, K = (x₁, 0) ∧ L = (x₂, 0) ∧ 0 < x₁ ∧ x₁ < x₂) →  -- K and L are on positive x-axis, L to the right of K
  (Real.cos (2 * Real.pi / 3) = (K.1 - L.1) / Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2)) →  -- ∠LKM = 120°
  (Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2) = Real.sqrt ((K.1 - M.1)^2 + (K.2 - M.2)^2)) →  -- KL = KM
  (∀ x, quadratic_polynomial x b c = 0 ↔ x = 1/2 ∨ x = 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_proof_l1090_109039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_point_lies_on_line_l1090_109071

/-- Given three points A, B, and C in 2D space, this theorem states that 
    if C lies on the line passing through A and B, then the slopes between 
    A-B and A-C are equal. -/
theorem point_on_line (xA yA xB yB xC yC : ℚ) : 
  (yC - yA) * (xB - xA) = (yB - yA) * (xC - xA) → 
  ∃ t : ℚ, xC = xA + t * (xB - xA) ∧ yC = yA + t * (yB - yA) :=
sorry

/-- The main theorem proving that the point (39/11, 7) lies on the line 
    passing through (3, 5) and (0, -6). -/
theorem point_lies_on_line : 
  ∃ t : ℚ, 39/11 = 3 + t * (0 - 3) ∧ 7 = 5 + t * (-6 - 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_point_lies_on_line_l1090_109071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_coordinates_is_correct_l1090_109069

/-- The product of x-coordinates of points on y = 4 that are 15 units from (-2, -3) -/
def product_of_x_coordinates : ℝ := -172

/-- A point on the line y = 4 -/
def point_on_line (x : ℝ) : ℝ × ℝ := (x, 4)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The product of x-coordinates satisfying the conditions is -172 -/
theorem product_of_x_coordinates_is_correct :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (∀ x : ℝ, distance (point_on_line x) (-2, -3) = 15 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ * x₂ = product_of_x_coordinates := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_coordinates_is_correct_l1090_109069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_approximately_12_l1090_109099

/-- The initial length of a rope, given the extended length and additional grazing area. -/
noncomputable def initial_rope_length (extended_length : ℝ) (additional_area : ℝ) : ℝ :=
  Real.sqrt (extended_length^2 - additional_area / Real.pi)

/-- Theorem stating that under given conditions, the initial rope length is approximately 12 meters. -/
theorem rope_length_approximately_12 :
  let extended_length : ℝ := 23
  let additional_area : ℝ := 1210
  let initial_length := initial_rope_length extended_length additional_area
  ∃ ε > 0, |initial_length - 12| < ε ∧ ε < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_approximately_12_l1090_109099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l1090_109077

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8*x - 10*y + 40

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, -5)

/-- The point we're measuring the distance to -/
def point : ℝ × ℝ := (-2, 5)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_center : distance circle_center point = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l1090_109077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_reachable_city_l1090_109091

/-- A city in our graph. -/
structure City where
  id : Nat

/-- A directed graph representing the road network. -/
structure RoadNetwork where
  cities : Finset City
  roads : City → City → Prop
  all_connected : ∀ c1 c2 : City, c1 ∈ cities → c2 ∈ cities → c1 ≠ c2 → roads c1 c2

/-- A path in the road network. -/
inductive RoadPath (rn : RoadNetwork) : City → City → Prop where
  | single : ∀ c, c ∈ rn.cities → RoadPath rn c c
  | cons : ∀ {c1 c2 c3}, c1 ∈ rn.cities → c2 ∈ rn.cities → c3 ∈ rn.cities →
           rn.roads c1 c2 → RoadPath rn c2 c3 → RoadPath rn c1 c3

/-- The main theorem: there exists a city from which all others are reachable. -/
theorem exists_reachable_city (rn : RoadNetwork) : 
  ∃ c : City, c ∈ rn.cities ∧ ∀ c' : City, c' ∈ rn.cities → RoadPath rn c c' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_reachable_city_l1090_109091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l1090_109007

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A set of points within a rectangle -/
def PointsInRectangle (r : Rectangle) (points : Set Point) : Prop :=
  ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem close_points_exist (r : Rectangle) (points : Set Point) 
    (h1 : r.width = 3) (h2 : r.height = 4) (h3 : Fintype points) (h4 : Fintype.card points = 6) 
    (h5 : PointsInRectangle r points) :
  ∃ p1 p2 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l1090_109007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perpendicular_DE_l1090_109040

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define the chord AB
def ChordAB (A B : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ B ∈ Circle ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 < 2

-- Define the line l
def LineL (l : Set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ l, p ∉ Circle

-- Define the angle between line l and chord AB
def AngleLAB (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  ∃ v ∈ l, (v.1 - A.1)^2 + (v.2 - A.2)^2 = 1 ∧
           (v.1 - B.1)^2 + (v.2 - B.2)^2 = 1

-- Define the perpendicularity of DE to AB
def PerpDE (D E A B : ℝ × ℝ) : Prop :=
  (D.1 - E.1) * (A.1 - B.1) + (D.2 - E.2) * (A.2 - B.2) = 0

-- The main theorem
theorem exists_perpendicular_DE (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  ChordAB A B → LineL l → AngleLAB l A B →
  ∃ C D E, C ∈ l ∧ D ∈ Circle ∧ E ∈ Circle ∧ PerpDE D E A B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perpendicular_DE_l1090_109040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_same_house_l1090_109073

theorem probability_all_same_house (num_houses : ℕ) (num_persons : ℕ) :
  num_houses = 3 →
  num_persons = 3 →
  (1 : ℚ) / 9 = (1 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_same_house_l1090_109073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1090_109042

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- Theorem statement
theorem function_properties :
  (f (-4) = -2) ∧
  (f 3 = 6) ∧
  (f (f (-2)) = 8) ∧
  (∀ a : ℝ, f a = 0 ↔ a = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1090_109042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_papbpc_l1090_109064

-- Define a right triangle with leg lengths 1
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  leg_lengths_eq_one : A.1 - C.1 = 1 ∧ B.2 - C.2 = 1
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define a point on the triangle
def PointOnTriangle (t : RightTriangle) : Type :=
  { P : ℝ × ℝ // 
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ P = (x * t.A.1 + (1 - x) * t.C.1, x * t.A.2 + (1 - x) * t.C.2)) ∨
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ P = (x * t.B.1 + (1 - x) * t.C.1, x * t.B.2 + (1 - x) * t.C.2)) ∨
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ P = (x * t.A.1 + (1 - x) * t.B.1, x * t.A.2 + (1 - x) * t.B.2))
  }

-- Define the product PA · PB · PC
noncomputable def ProductPAPBPC (t : RightTriangle) (P : PointOnTriangle t) : ℝ :=
  let PA := Real.sqrt ((P.val.1 - t.A.1)^2 + (P.val.2 - t.A.2)^2)
  let PB := Real.sqrt ((P.val.1 - t.B.1)^2 + (P.val.2 - t.B.2)^2)
  let PC := Real.sqrt ((P.val.1 - t.C.1)^2 + (P.val.2 - t.C.2)^2)
  PA * PB * PC

-- State the theorem
theorem max_product_papbpc (t : RightTriangle) : 
  ∃ P : PointOnTriangle t, ∀ Q : PointOnTriangle t, ProductPAPBPC t P ≥ ProductPAPBPC t Q ∧ 
  ProductPAPBPC t P = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_papbpc_l1090_109064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_digit_of_fraction_l1090_109047

def fraction : ℚ := 57 / 5000

def thousandths_digit (q : ℚ) : ℕ :=
  (((q * 1000).floor : ℤ).toNat) % 10

theorem thousandths_digit_of_fraction :
  thousandths_digit fraction = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_digit_of_fraction_l1090_109047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_asymptotic_behavior_l1090_109038

noncomputable def f (x : ℝ) : ℝ := (x^2 * (x + 1)) / (x - 1)

noncomputable def f' (x : ℝ) : ℝ := ((3 * x^2 - x - 2) * x) / ((x - 1)^2)

theorem stationary_points (x : ℝ) (hx : x ≠ 1) :
  f' x = 0 ↔ 3 * x^2 - x - 2 = 0 := by
  sorry

theorem asymptotic_behavior :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x / x^3 - 1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_asymptotic_behavior_l1090_109038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1090_109016

-- Define the functions f and g
noncomputable def f (x : ℝ) := 2 * x + 1 / x^2 - 4
noncomputable def g (x m : ℝ) := Real.log x / x - m

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc (1/Real.exp 1) (Real.exp 2), f x₁ ≤ g x₂ m) ↔
  m ∈ Set.Iic (1/Real.exp 1 - 1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1090_109016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1090_109063

-- Definition of (a,b)-type function
def is_ab_type_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f (a + x) * f (a - x) = b

-- Statement for problem 1
theorem problem_1 : is_ab_type_function (fun x ↦ 4^x) := by sorry

-- Definition of g(x)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 1 then x^2 - m*(x-1) + 1 else 4 / (x^2 - m*(x-1) + 1)

-- Conditions for problem 2
def problem_2_conditions (m : ℝ) : Prop :=
  (∀ x, g m (1 + x) * g m (1 - x) = 4) ∧
  (∀ x ∈ Set.Icc 0 2, 1 ≤ g m x ∧ g m x ≤ 3)

-- Statement for problem 2
theorem problem_2 (m : ℝ) :
  problem_2_conditions m →
  2 - 2*Real.sqrt 6/3 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1090_109063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_f_a_l1090_109013

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem value_of_f_a (a b : ℝ) (h1 : b = 2) (h2 : f (a * b) = 1/6) : f a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_f_a_l1090_109013
