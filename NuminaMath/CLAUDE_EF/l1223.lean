import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_distribution_l1223_122385

/-- The number of ways to distribute stickers across sheets -/
def distribute_stickers (stickers : ℕ) (sheets : ℕ) : ℕ :=
  sorry

/-- At least one sheet remains empty -/
def at_least_one_empty (distribution : List ℕ) : Prop :=
  sorry

/-- The sheets are identical and only the number of stickers matters -/
def identical_sheets (distribution1 distribution2 : List ℕ) : Prop :=
  sorry

theorem sticker_distribution :
  ∃ (ways : ℕ),
    ways = distribute_stickers 10 5 ∧
    (∀ distribution : List ℕ,
      distribution.length = 5 ∧
      distribution.sum = 10 ∧
      at_least_one_empty distribution ∧
      (∀ d1 d2 : List ℕ, identical_sheets d1 d2 → (d1 = distribution ↔ d2 = distribution))) ∧
    ways = 23 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_distribution_l1223_122385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_15_10_l1223_122311

/-- The function J for nonzero real numbers -/
noncomputable def J (a b c : ℝ) : ℝ := a / b + b / c + c / a

/-- Theorem stating that J(3, 15, 10) = 151/30 -/
theorem J_3_15_10 :
  J 3 15 10 = 151 / 30 :=
by
  -- Unfold the definition of J
  unfold J
  -- Simplify the expression
  simp [add_assoc, add_comm, add_left_comm]
  -- Perform the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_15_10_l1223_122311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1223_122397

theorem absolute_value_inequality : ¬ (-|(-1.5 : ℝ)| = -(-1.5 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1223_122397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_C_B_range_of_a_when_C_contains_A_intersect_B_l1223_122325

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Theorem for part (I)
theorem intersection_A_complement_C_B :
  A ∩ (Set.univ \ B) = Set.Icc (-3) 2 := by sorry

-- Theorem for part (II)
theorem range_of_a_when_C_contains_A_intersect_B :
  ∀ a : ℝ, a ≠ 0 → (C a ⊇ (A ∩ B) → 4/3 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_C_B_range_of_a_when_C_contains_A_intersect_B_l1223_122325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisector_ratio_l1223_122306

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle α
variable (α : ℝ)

-- Helper functions (not proved)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry
def is_trisector (vertex point1 point2 point3 : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem trisector_ratio (t : Triangle) (h1 : distance t.B t.C = 3 * distance t.A t.C) 
  (h2 : angle t.A t.C t.B = α) : 
  ∃ (M N : ℝ × ℝ), 
    is_trisector t.C M t.A t.B ∧ 
    is_trisector t.C N t.A t.B ∧ 
    distance t.C M / distance t.C N = (2 * Real.cos (α / 3) + 3) / (6 * Real.cos (α / 3) + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisector_ratio_l1223_122306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_throws_23_percent_l1223_122322

/-- Represents the percentage of apples thrown away by a vendor over two days -/
noncomputable def apples_thrown_away (initial_apples : ℝ) : ℝ :=
  let remaining_after_first_sale := initial_apples * (1 - 0.6)
  let thrown_first_day := remaining_after_first_sale * 0.15
  let remaining_after_first_throw := remaining_after_first_sale - thrown_first_day
  let remaining_after_second_sale := remaining_after_first_throw * (1 - 0.5)
  let thrown_second_day := remaining_after_second_sale
  (thrown_first_day + thrown_second_day) / initial_apples * 100

/-- Theorem stating that the percentage of apples thrown away is 23% -/
theorem vendor_throws_23_percent :
  ∀ initial_apples : ℝ, initial_apples > 0 → apples_thrown_away initial_apples = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_throws_23_percent_l1223_122322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_4_3_or_5_l1223_122330

theorem probability_multiple_2_4_3_or_5 : 
  let n := 120
  let multiples := {x : ℕ | x ≤ n ∧ (2 ∣ x ∨ 4 ∣ x ∨ 3 ∣ x ∨ 5 ∣ x)}
  (Finset.card (Finset.filter (λ x => x ∈ multiples) (Finset.range (n+1))) : ℚ) / n = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_4_3_or_5_l1223_122330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l1223_122304

/-- A triangular pyramid with three mutually perpendicular lateral edges -/
structure TriangularPyramid where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  perpendicular : edge1 * edge2 = 0 ∧ edge2 * edge3 = 0 ∧ edge1 * edge3 = 0

/-- The surface area of a sphere -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 6π -/
theorem circumscribed_sphere_area (p : TriangularPyramid) 
  (h1 : p.edge1 = Real.sqrt 3)
  (h2 : p.edge2 = Real.sqrt 2)
  (h3 : p.edge3 = 1) :
  sphereSurfaceArea (Real.sqrt ((p.edge1^2 + p.edge2^2 + p.edge3^2) / 4)) = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l1223_122304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1223_122357

theorem triangle_side_count : 
  let possible_sides := {x : ℕ | x > 3 ∧ x < 13}
  Finset.card (Finset.filter (λ x => x ∈ possible_sides) (Finset.range 13)) = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1223_122357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_intervals_exist_l1223_122399

/-- A closed interval of real numbers -/
structure ClosedInterval where
  left : ℝ
  right : ℝ
  is_closed : left ≤ right

/-- A finite set of closed intervals of length 1 that cover [0, 50] -/
structure CoveringIntervals where
  intervals : Finset ClosedInterval
  unit_length : ∀ i ∈ intervals, i.right - i.left = 1
  covers_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → ∃ i ∈ intervals, i.left ≤ x ∧ x ≤ i.right

/-- Two intervals are disjoint if they do not overlap -/
def are_disjoint (i j : ClosedInterval) : Prop :=
  i.right ≤ j.left ∨ j.right ≤ i.left

/-- The main theorem stating that we can find at least 25 disjoint intervals -/
theorem disjoint_intervals_exist (cover : CoveringIntervals) :
  ∃ (subset : Finset ClosedInterval),
    subset ⊆ cover.intervals ∧
    subset.card ≥ 25 ∧
    ∀ i j, i ∈ subset → j ∈ subset → i ≠ j → are_disjoint i j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_intervals_exist_l1223_122399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_md_relation_l1223_122342

theorem abc_md_relation (a b c : ℕ+) :
  let D := Nat.gcd a.val (Nat.gcd b.val c.val)
  let M := Nat.lcm a.val (Nat.lcm b.val c.val)
  
  -- MD = abc
  (D * M = a.val * b.val * c.val) ∧
  
  -- MD ≤ abc
  (D * M ≤ a.val * b.val * c.val) ∧
  
  -- If a, b, c are pairwise coprime, then MD = abc
  ((Nat.Coprime a.val b.val ∧ Nat.Coprime b.val c.val ∧ Nat.Coprime a.val c.val) →
   D * M = a.val * b.val * c.val) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_md_relation_l1223_122342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1223_122398

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) : 
  -1 < b → b < 1 → 
  T b * T (-b) = 2250 →
  T b + T (-b) = 300 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1223_122398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1223_122379

/-- The area of a quadrilateral bounded by y = a, y = -b, x = -c, and y = x + k -/
noncomputable def quadrilateralArea (a b c k : ℝ) : ℝ :=
  (a + b) * (c + max (a - k) (-b - k)) / 2

theorem quadrilateral_area_theorem (a b c k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  let A := quadrilateralArea a b c k
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    x₁ = -c ∧ 
    x₂ = max (a - k) (-b - k) ∧
    y₁ = -b ∧
    y₂ = a ∧
    A = (y₂ - y₁) * (x₂ - x₁) / 2 := by
  sorry

#check quadrilateral_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1223_122379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_DAB_l1223_122312

/-- Parabola C: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus F of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P -/
def point_P : ℝ × ℝ := (-1, 1)

/-- Directrix l: x = -1 -/
def directrix (x : ℝ) : Prop := x = -1

/-- Point D: intersection of directrix and x-axis -/
def point_D : ℝ × ℝ := (-1, 0)

/-- Line PF: x + 2y - 1 = 0 -/
def line_PF (x y : ℝ) : Prop := x + 2*y - 1 = 0

/-- Area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Theorem: Area of triangle DAB is 4√5 -/
theorem area_DAB : ∃ (A B : ℝ × ℝ), 
  parabola A.1 A.2 ∧ 
  parabola B.1 B.2 ∧ 
  line_PF A.1 A.2 ∧ 
  line_PF B.1 B.2 ∧ 
  (area_triangle point_D A B = 4 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_DAB_l1223_122312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l1223_122362

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Finds the symmetric point with respect to the line θ = π/2 -/
noncomputable def symmetricPoint (p : PolarPoint) : PolarPoint :=
  { r := p.r,
    θ := if p.θ < Real.pi/2 then Real.pi - p.θ else 3*Real.pi - p.θ }

theorem symmetric_point_correct (M : PolarPoint) (h : M = ⟨2, -Real.pi/6⟩) :
  symmetricPoint M = ⟨2, 7*Real.pi/6⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l1223_122362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1223_122328

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | (n + 1) => 2 * sequence_a n - 1

theorem sequence_a_closed_form (n : ℕ) :
  sequence_a n = 2^(n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1223_122328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_range_l1223_122317

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 3) * Real.exp x
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (1 / 2) * x - Real.log x + t

-- State the theorem
theorem f_leq_g_range :
  ∀ t : ℝ, (∃ x : ℝ, x > 0 ∧ f (-1) x ≤ g t x) ↔ t ≤ Real.exp 2 - 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_range_l1223_122317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l1223_122310

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = 3x - 7 -/
noncomputable def m1 : ℝ := 3

/-- The slope of the second line 8y + (8/3)x = 16 -/
noncomputable def m2 : ℝ := -(8/3) / 8

/-- Theorem: The lines y = 3x - 7 and 8y + (8/3)x = 16 are perpendicular -/
theorem lines_perpendicular : perpendicular m1 m2 := by
  unfold perpendicular m1 m2
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l1223_122310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_another_day_choice_l1223_122361

theorem another_day_choice (day : String) :
  day = "another" →
  day ≠ "other" ∧ day ≠ "the other" ∧ day ≠ "others" :=
by
  intro h
  constructor
  · intro contra
    rw [contra] at h
    contradiction
  constructor
  · intro contra
    rw [contra] at h
    contradiction
  · intro contra
    rw [contra] at h
    contradiction

#check another_day_choice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_another_day_choice_l1223_122361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_proof_l1223_122365

noncomputable def f (k l m x : ℝ) : ℝ := k + m / (x - l)

noncomputable def g (k l m x : ℝ) : ℝ := f k l m (f k l m x)

theorem function_value_proof (k l m : ℝ) (hm : m ≠ 0) :
  (∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ), 
    ((f k l m x₁ = y₁ ∧ f k l m x₂ = y₂ ∧ f k l m x₃ = y₃) ∨
     (g k l m x₁ = y₁ ∧ g k l m x₂ = y₂ ∧ g k l m x₃ = y₃)) ∧
    ((x₁ = 1 ∧ y₁ = 4) ∨ (x₁ = 2 ∧ y₁ = 3) ∨ (x₁ = 2 ∧ y₁ = 4)) ∧
    ((x₂ = 1 ∧ y₂ = 4) ∨ (x₂ = 2 ∧ y₂ = 3) ∨ (x₂ = 2 ∧ y₂ = 4)) ∧
    ((x₃ = 1 ∧ y₃ = 4) ∨ (x₃ = 2 ∧ y₃ = 3) ∨ (x₃ = 2 ∧ y₃ = 4)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  f k l m (k + l + m) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_proof_l1223_122365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_two_equals_negative_six_l1223_122370

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x ≥ 0 then x^2 + x else -(((-x)^2) + (-x))

-- State the theorem
theorem f_minus_two_equals_negative_six :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x ≥ 0, f x = x^2 + x) → -- f(x) = x^2 + x for x ≥ 0
  f (-2) = -6 := by
  intros h_odd h_nonneg
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_two_equals_negative_six_l1223_122370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_F_l1223_122337

noncomputable def f (x : ℝ) := x + Real.sin x

def F (x y : ℝ) := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6*y + 11) + f (x^2 - 8*x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min max : ℝ), min = 13 ∧ max = 49 ∧ min ≤ F x y ∧ F x y ≤ max := by
  sorry

#check min_max_F

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_F_l1223_122337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_is_24_l1223_122331

def total_students : Nat := 360
def min_rows : Nat := 12
def min_students_per_row : Nat := 18

def is_valid_arrangement (students_per_row : Nat) : Bool :=
  students_per_row ≥ min_students_per_row &&
  (total_students / students_per_row) ≥ min_rows &&
  total_students % students_per_row = 0

def sum_of_valid_arrangements : Nat :=
  (List.range (total_students + 1)).filter is_valid_arrangement |>.sum

theorem sum_of_valid_arrangements_is_24 :
  sum_of_valid_arrangements = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_is_24_l1223_122331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_soda_price_l1223_122363

/-- Calculate the price of discounted soda cans -/
theorem discounted_soda_price
  (regular_price : ℚ)
  (discount_percent : ℚ)
  (num_cans : ℕ)
  (h1 : regular_price = 30 / 100)
  (h2 : discount_percent = 15 / 100)
  (h3 : num_cans = 72) :
  num_cans * (regular_price * (1 - discount_percent)) = 1836 / 100 := by
  sorry

#eval (72 : ℕ) * ((30 / 100 : ℚ) * (1 - 15 / 100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_soda_price_l1223_122363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_when_a_is_3_range_of_a_when_P_subset_Q_Q_equals_Ioo_neg5_1_l1223_122359

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 1 - (a + 1) / (x + 1) < 0}
def Q : Set ℝ := {x | |x + 2| < 3}

-- Theorem 1: P when a = 3
theorem P_when_a_is_3 : P 3 = Set.Ioo (-1) 3 := by sorry

-- Theorem 2: Range of a when P ∪ Q = Q
theorem range_of_a_when_P_subset_Q : 
  {a : ℝ | a > 0 ∧ P a ⊆ Set.Ioo (-5) 1} = Set.Ioc 0 1 := by sorry

-- Additional theorem to establish Q
theorem Q_equals_Ioo_neg5_1 : Q = Set.Ioo (-5) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_when_a_is_3_range_of_a_when_P_subset_Q_Q_equals_Ioo_neg5_1_l1223_122359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_sum_fourth_powers_l1223_122368

theorem equilateral_triangle_inscribed_circle_sum_fourth_powers 
  (O : ℝ × ℝ) (R : ℝ) (M N K F : ℝ × ℝ) :
  R > 0 →
  (∀ P ∈ ({M, N, K, F} : Set (ℝ × ℝ)), dist O P = R) →
  dist M N = dist N K →
  dist N K = dist K M →
  dist M N = R * Real.sqrt 3 →
  (dist F M)^4 + (dist F N)^4 + (dist F K)^4 = 18 * R^4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_sum_fourth_powers_l1223_122368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l1223_122303

/-- Represents the motorcyclist's journey --/
structure Journey where
  distance : ℝ
  initialSpeed : ℝ
  stopTime : ℝ
  speedIncrease : ℝ

/-- Calculates the total time of the return journey --/
noncomputable def returnTime (j : Journey) : ℝ :=
  1 + j.stopTime + (j.distance - j.initialSpeed) / (j.initialSpeed + j.speedIncrease)

/-- Theorem stating the initial speed of the motorcyclist --/
theorem motorcyclist_speed (j : Journey) 
  (h1 : j.distance = 120)
  (h2 : j.stopTime = 1/6)
  (h3 : j.speedIncrease = 6)
  (h4 : j.distance / j.initialSpeed = returnTime j) :
  j.initialSpeed = 48 := by
  sorry

#check motorcyclist_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l1223_122303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1223_122327

theorem simplification_and_evaluation :
  ∀ (m n : ℝ),
    (2*m + 3*n - 3*m + 5*n = -m + 8*n) ∧
    (2*(m^2 + m^2*n) - (2*m^2 - m*n^2) = 2*m^2*n + m*n^2) ∧
    (2*(-4)^2*(-1/2) + (-4)*(-1/2)^2 = -17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1223_122327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1223_122360

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the focal length
def focal_length : ℝ := 4

-- Define the left focus F
def F : ℝ × ℝ := (-2, 0)

-- Define a point T on the line x = -3
def T (m : ℝ) : ℝ × ℝ := (-3, m)

-- Define the line perpendicular to TF
def perpendicular_line (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y - 2

-- Define the intersections P and Q
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | C p.1 p.2 ∧ perpendicular_line m p.1 p.2}

-- Define the midpoint of PQ
noncomputable def midpoint_PQ (m : ℝ) : ℝ × ℝ :=
  (-6 / (m^2 + 3), 2*m / (m^2 + 3))

-- Define the ratio |TF|/|PQ|
noncomputable def ratio (m : ℝ) : ℝ :=
  Real.sqrt ((m^2 + 3)^2 / (24 * (m^2 + 1)))

theorem ellipse_properties :
  -- 1. The focal length is 4
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a^2 - b^2 = 4) ∧
  -- 2a. Line OT bisects line segment PQ
  (∀ m : ℝ, midpoint_PQ m = (m * (midpoint_PQ m).2, (midpoint_PQ m).2)) ∧
  -- 2b. |TF|/|PQ| is minimized when m = ±1
  (∀ m : ℝ, ratio m ≥ Real.sqrt 3 / 3) ∧
  (ratio 1 = Real.sqrt 3 / 3) ∧ (ratio (-1) = Real.sqrt 3 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1223_122360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_traversal_impossible_l1223_122324

/-- Represents the color of a field on the board -/
inductive FieldColor
  | Red
  | Black
  | White

/-- Represents the board configuration -/
structure Board :=
  (total_fields : Nat)
  (red_fields : Nat)
  (black_fields : Nat)
  (white_fields : Nat)
  (h_total : total_fields = red_fields + black_fields + white_fields)

/-- Represents a camel's movement on the board -/
structure CamelMovement :=
  (board : Board)
  (start_color : FieldColor)
  (same_color_frequency : Nat)
  (h_same_color : same_color_frequency = 3)

/-- Theorem stating the impossibility of the camel's traversal -/
theorem camel_traversal_impossible (b : Board) (cm : CamelMovement) 
  (h_board : b.total_fields = 25 ∧ b.red_fields = 9 ∧ b.black_fields = 8 ∧ b.white_fields = 8)
  (h_start : cm.start_color = FieldColor.Red)
  (h_board_match : cm.board = b) :
  ¬ ∃ (path : List FieldColor), path.length = b.total_fields ∧ List.Nodup path := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_traversal_impossible_l1223_122324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_subtriangle_probability_l1223_122355

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  isIsosceles : dist A B = dist A C
  base : ℝ
  baseLength : dist B C = base

/-- The probability that a random point in an isosceles triangle makes one subtriangle larger -/
noncomputable def probabilityLargerSubtriangle (t : IsoscelesTriangle) : ℝ :=
  1 / 2

/-- Theorem: In an isosceles triangle, the probability that a random point P makes
    the area of triangle ABP greater than the area of triangle ACP is 1/2 -/
theorem isosceles_triangle_subtriangle_probability
  (t : IsoscelesTriangle) :
  probabilityLargerSubtriangle t = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_subtriangle_probability_l1223_122355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_approx_l1223_122384

/-- Represents the reaction between KOH and HNO₃ -/
structure ReactionKOH_HNO3 where
  ω_KOH : ℝ
  ρ_KOH : ℝ
  V_KOH : ℝ
  M_KOH : ℝ
  C_M_HNO3 : ℝ
  V_HNO3 : ℝ
  ΔH : ℝ

/-- Calculates the number of moles of KOH -/
noncomputable def moles_KOH (r : ReactionKOH_HNO3) : ℝ :=
  r.ω_KOH * r.ρ_KOH * r.V_KOH / r.M_KOH

/-- Calculates the number of moles of HNO₃ -/
noncomputable def moles_HNO3 (r : ReactionKOH_HNO3) : ℝ :=
  r.C_M_HNO3 * r.V_HNO3

/-- Calculates the heat released in the reaction -/
noncomputable def heat_released (r : ReactionKOH_HNO3) : ℝ :=
  r.ΔH * (min (moles_KOH r) (moles_HNO3 r))

/-- Theorem: The heat released in the reaction is approximately 1.47 kJ -/
theorem heat_released_approx (r : ReactionKOH_HNO3)
  (h1 : r.ω_KOH = 0.062)
  (h2 : r.ρ_KOH = 1.055)
  (h3 : r.V_KOH = 22.7)
  (h4 : r.M_KOH = 56)
  (h5 : r.C_M_HNO3 = 2.00)
  (h6 : r.V_HNO3 = 0.0463)
  (h7 : r.ΔH = 55.6) :
  ∃ ε > 0, |heat_released r - 1.47| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_approx_l1223_122384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1223_122386

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 1]

theorem inverse_of_A : 
  A⁻¹ = !![1, 2; 3, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1223_122386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_angle_l1223_122302

/-- In a circle where the chord length corresponding to a central angle of 2 radians is 2,
    the arc length corresponding to this central angle is 2. -/
theorem arc_length_equals_angle (r : ℝ) (h : 2 * r * Real.sin 1 = 2) : r * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_angle_l1223_122302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_fence_together_time_l1223_122369

/-- The time it takes for two people to paint a fence together, given their individual painting times. -/
noncomputable def time_to_paint_together (time_person1 time_person2 : ℝ) : ℝ :=
  1 / (1 / time_person1 + 1 / time_person2)

/-- Proves that Jamshid and Taimour can paint the fence together in 5 hours. -/
theorem paint_fence_together_time 
  (taimour_time : ℝ) 
  (jamshid_time : ℝ) 
  (h1 : taimour_time = 15) 
  (h2 : jamshid_time = taimour_time / 2) : 
  time_to_paint_together taimour_time jamshid_time = 5 := by
  sorry

#check paint_fence_together_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_fence_together_time_l1223_122369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviations_below_mean_greater_than_44_l1223_122316

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  σ_pos : σ > 0

/-- Calculates the z-score for a given value in a normal distribution --/
noncomputable def zScore (nd : NormalDistribution) (x : ℝ) : ℝ :=
  (x - nd.μ) / nd.σ

theorem standard_deviations_below_mean_greater_than_44 
  (nd : NormalDistribution) 
  (h1 : nd.μ = 51) 
  (h2 : nd.σ = 2) 
  (x : ℝ) 
  (h3 : x > 44) : 
  zScore nd x > -3.5 := by
  sorry

#check standard_deviations_below_mean_greater_than_44

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviations_below_mean_greater_than_44_l1223_122316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_together_is_ten_l1223_122390

/-- The number of days a and b worked together before b left -/
noncomputable def days_worked_together (W : ℝ) : ℝ :=
  let combined_rate := W / 40
  let a_rate := W / 16
  let x := W / (4 * combined_rate)
  x

theorem days_worked_together_is_ten (W : ℝ) (W_pos : W > 0) :
  days_worked_together W = 10 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_together_is_ten_l1223_122390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1223_122352

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^(x^2 - 5*x + 5) > 1/2 ↔ x < 2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1223_122352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_equals_five_fourths_l1223_122350

/-- The vertices of triangle ABC -/
def A : ℝ × ℝ := (-4, 0)
def C : ℝ × ℝ := (4, 0)

/-- B is a point on the ellipse -/
noncomputable def B : ℝ × ℝ := sorry

/-- The ellipse equation -/
axiom ellipse_eq : (B.1)^2 / 25 + (B.2)^2 / 9 = 1

/-- Triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) := {A, B, C}

/-- Angle measure in a triangle -/
noncomputable def angle (t : Set (ℝ × ℝ)) (v : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The ratio of sines equals 5/4 -/
theorem sine_ratio_equals_five_fourths :
  (Real.sin (angle triangle_ABC A) + Real.sin (angle triangle_ABC C)) / Real.sin (angle triangle_ABC B) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_equals_five_fourths_l1223_122350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1223_122333

def A : Set ℝ := {-2, -1, 0, 5, 10, 20}
def B : Set ℝ := {x | 0 < x ∧ x < 10}

theorem intersection_A_B : A ∩ B = {5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1223_122333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1223_122353

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - (Real.exp x - 1) / (Real.exp x + 1)

theorem m_range (m : ℝ) : f (4 - m) - f m ≥ 8 - 4 * m → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1223_122353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_for_triple_g_l1223_122345

noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

theorem six_solutions_for_triple_g :
  ∃ (S : Finset ℝ), 
    S.card = 6 ∧ 
    (∀ x ∈ S, -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x → x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_for_triple_g_l1223_122345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1223_122376

/-- The solution set of the inequality 1/(x-1) < 1 -/
def p : Set ℝ := {x | x < 1 ∨ x > 2}

/-- The solution set of the inequality x^2 + (a-1)x - a > 0 -/
def q (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

/-- The theorem stating that p is a sufficient but not necessary condition for q,
    and the range of a is (-2, -1] -/
theorem solution_range (a : ℝ) : 
  (∀ x, x ∈ p → x ∈ q a) ∧ 
  (∃ x, x ∈ q a ∧ x ∉ p) ↔ 
  a ∈ Set.Ioc (-2) (-1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1223_122376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersections_l1223_122339

/-- A line in the 2D plane defined by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The x-coordinate of the intersection point of a line with the x-axis -/
noncomputable def xAxisIntersection (l : Line) : ℝ := -l.b / l.m

/-- The y-coordinate of the intersection point of a line with the y-axis -/
def yAxisIntersection (l : Line) : ℝ := l.b

theorem line_intersections (l : Line) (h : l.m = 2 ∧ l.b = -1) : 
  xAxisIntersection l = 0.5 ∧ yAxisIntersection l = -1 := by
  sorry

#check line_intersections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersections_l1223_122339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1223_122314

theorem trig_problem (α : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α + 5*π/4) = 7 ∧ 
  (Real.sin α)^2 + Real.sin (2*α) / ((Real.cos α)^2 + Real.cos (2*α)) = 33/23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1223_122314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_lung_capacity_association_expected_value_X_l1223_122380

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![22, 22],
    ![30, 6]]

-- Define the total number of students
def n : ℕ := 80

-- Define the K^2 formula
noncomputable def K_squared (a b c d : ℕ) : ℝ :=
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.5% confidence
def critical_value : ℝ := 7.879

-- Define the group composition
def group_composition : Fin 3 → ℕ
  | 0 => 1  -- unqualified
  | 1 => 4  -- good
  | 2 => 1  -- excellent

-- Define the scoring system
def score : Fin 3 → ℕ
  | 0 => 0  -- unqualified
  | 1 => 2  -- good
  | 2 => 3  -- excellent

-- Theorem 1: Association between height and lung capacity
theorem height_lung_capacity_association :
  K_squared 22 22 30 6 > critical_value :=
by sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  let X := λ i j => score i + score j
  let total_combinations := Nat.choose 6 2
  (Finset.sum (Finset.univ : Finset (Fin 3 × Fin 3))
    (λ (i, j) => X i j * (group_composition i * group_composition j))) /
    total_combinations = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_lung_capacity_association_expected_value_X_l1223_122380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parallel_to_xaxis_l1223_122346

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem min_distance_parallel_to_xaxis (A B C : Point) 
  (h1 : A.x = -3 ∧ A.y = 2) 
  (h2 : B.x = 3 ∧ B.y = 4) 
  (h3 : C.y = A.y) : -- AC is parallel to x-axis
  (∃ (C_min : Point), C_min.x = B.x ∧ C_min.y = A.y ∧
    ∀ (C' : Point), C'.y = A.y → distance B C_min ≤ distance B C') ∧
  (distance B (Point.mk 3 2) = 2) := by
  sorry

#check min_distance_parallel_to_xaxis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parallel_to_xaxis_l1223_122346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_used_l1223_122374

theorem oil_used (total_liquid water_used oil_used : ℚ) 
  (h1 : total_liquid = 1.33)
  (h2 : water_used = 1.17)
  (h3 : total_liquid = water_used + oil_used) :
  oil_used = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_used_l1223_122374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1223_122305

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (α/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1223_122305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_8963_to_hundredth_l1223_122343

/-- Rounds a given number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The theorem states that rounding 3.8963 to the nearest hundredth results in 3.90 -/
theorem round_3_8963_to_hundredth :
  roundToHundredth 3.8963 = 3.90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_8963_to_hundredth_l1223_122343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1223_122395

-- Define the hyperbolas
def C1 (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1
def C2 (x y : ℝ) : Prop := y^2/9 - x^2/16 = 1

-- Define asymptotes
def asymptote (x y : ℝ) : Prop := y = (3/4) * x ∨ y = -(3/4) * x

-- Define focal distance
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote x y ↔ (C1 x y ∨ C2 x y)) ∧
  (∀ x y : ℝ, ¬(C1 x y ∧ C2 x y)) ∧
  (focal_distance 4 3 = focal_distance 3 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1223_122395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_1_2015_was_monday_l1223_122309

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem june_1_2015_was_monday :
  dayAfter DayOfWeek.Thursday 151 = DayOfWeek.Monday := by
  sorry

#eval dayAfter DayOfWeek.Thursday 151

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_1_2015_was_monday_l1223_122309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_example_oplus_equation_solutions_l1223_122388

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a * b - a else a * b + b

-- Theorem 1
theorem oplus_example : oplus (3 - Real.sqrt 3) (Real.sqrt 3) = 4 * Real.sqrt 3 - 3 := by
  sorry

-- Theorem 2
theorem oplus_equation_solutions (x : ℝ) :
  oplus (2 * x) (x + 1) = 6 ↔ x = Real.sqrt 3 ∨ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_example_oplus_equation_solutions_l1223_122388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1223_122387

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  Real.sqrt 3 * t.c = 2 * t.a * Real.sin t.C

-- Part 1
theorem part1 (t : Triangle) (h : satisfiesConditions t) : 
  Real.sin t.A = Real.sqrt 3 / 2 :=
sorry

-- Part 2
def additionalConditions (t : Triangle) : Prop :=
  t.A < Real.pi / 2 ∧ t.a = 2 * Real.sqrt 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3

theorem part2 (t : Triangle) (h1 : satisfiesConditions t) (h2 : additionalConditions t) :
  (t.b = 4 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1223_122387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l1223_122318

/-- The system of equations has infinitely many real solutions -/
theorem infinite_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ (x y z : ℝ), (x, y, z) ∈ S ↔ x + y = 2 ∧ x * y - z^2 = 1) ∧ Infinite S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l1223_122318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_equality_iff_tangency_l1223_122320

-- Define the basic structures
structure Point where

structure Circle where

structure Triangle where

-- Define the given conditions
variable (A B C : Point)
variable (ABC : Triangle)
variable (Γ : Circle)
variable (P M N : Point)

-- Define the properties
def passes_through (c : Circle) (p : Point) : Prop := sorry

def tangent_to_side (c : Circle) (p : Point) (t : Triangle) : Prop := sorry

def intersects_side (c : Circle) (p : Point) (t : Triangle) : Prop := sorry

def smaller_arc (c : Circle) (p1 p2 : Point) : Set Point := sorry

def equal_arcs (a1 a2 : Set Point) : Prop := sorry

def circumcircle (t : Triangle) : Circle := sorry

def tangent_circles (c1 c2 : Circle) (p : Point) : Prop := sorry

-- State the theorem
theorem arc_equality_iff_tangency :
  passes_through Γ A ∧
  tangent_to_side Γ P ABC ∧
  intersects_side Γ M ABC ∧
  intersects_side Γ N ABC →
  equal_arcs (smaller_arc Γ M P) (smaller_arc Γ N P) ↔
  tangent_circles Γ (circumcircle ABC) A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_equality_iff_tangency_l1223_122320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_A_range_l1223_122301

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/4) * Real.cos (x/4) + (Real.cos (x/4))^2 + 1/2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = π
  side_angle_relation : (2*a - c) * Real.cos B = b * Real.cos C

-- State the theorem
theorem f_A_range (t : Triangle) : 
  ∃ (y : ℝ), 3/2 < y ∧ y < 2 ∧ y = f t.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_A_range_l1223_122301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1223_122396

-- Define the complex numbers
def B : ℂ := 5 - 2*Complex.I
def N : ℂ := -5 + 2*Complex.I
def T : ℂ := 2*Complex.I
def Q : ℂ := 3

-- State the theorem
theorem complex_arithmetic :
  B - N + T - Q = 7 - 2*Complex.I :=
by
  -- Expand the definitions
  simp [B, N, T, Q]
  -- Simplify the complex arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1223_122396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1223_122391

/-- Calculates the length of a train given the speeds of two trains and the time it takes for them to pass each other. -/
noncomputable def train_length (slower_speed faster_speed : ℝ) (passing_time : ℝ) : ℝ :=
  (slower_speed + faster_speed) * (1000 / 3600) * passing_time

/-- Theorem stating that given the conditions of the problem, the length of the faster train is 270 meters. -/
theorem faster_train_length :
  let slower_speed : ℝ := 36
  let faster_speed : ℝ := 45
  let passing_time : ℝ := 12
  train_length slower_speed faster_speed passing_time = 270 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1223_122391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_tiling_problem_l1223_122375

/-- Represents the dimensions of a rectangular object in feet -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Converts inches to feet -/
def inches_to_feet (inches : ℚ) : ℚ := inches / 12

/-- Calculates the area of a rectangular object given its dimensions in feet -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tiles_needed (floor : Dimensions) (tile : Dimensions) : ℕ :=
  Nat.ceil ((area floor) / (area tile))

theorem floor_tiling_problem :
  let floor : Dimensions := { length := 9, width := 12 }
  let tile : Dimensions := { length := inches_to_feet 6, width := inches_to_feet 4 }
  tiles_needed floor tile = 648 := by
  sorry

#eval tiles_needed { length := 9, width := 12 } { length := 1/2, width := 1/3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_tiling_problem_l1223_122375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_area_properties_l1223_122329

/-- A quadratic function satisfying certain conditions -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x

/-- The area function g(t) as described in the problem -/
noncomputable def g (t : ℝ) : ℝ :=
  -4/3 * t^3 + 3/2 * t^2 - 1/2 * t + 1/12

/-- Theorem stating the properties of f and g -/
theorem quadratic_and_area_properties :
  (∀ x, f x ≥ -1/4) ∧ 
  f 0 = 0 ∧ 
  f 1 = 0 ∧
  (∃ x, f x = -1/4) ∧
  (∀ t, 0 < t → t < 1/2 → g t ≥ g (1/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_area_properties_l1223_122329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4no3_h2o_equality_l1223_122366

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ

instance : Coe Moles ℝ where
  coe m := m.value

instance : OfNat Moles n where
  ofNat := ⟨n⟩

/-- The balanced chemical equation: NH4NO3 + NaOH → NaNO3 + NH3 + H2O -/
axiom balanced_equation : True

/-- The molar ratio of NH4NO3 to H2O in the reaction is 1:1 -/
axiom molar_ratio (nh4no3 h2o : Moles) : nh4no3 = h2o

/-- The amount of H2O produced in the reaction -/
def h2o_produced : Moles := 2

/-- The amount of NH4NO3 combined in the reaction -/
def nh4no3_combined : Moles := 2

/-- Theorem: The number of moles of NH4NO3 combined is equal to the number of moles of H2O produced -/
theorem nh4no3_h2o_equality : nh4no3_combined = h2o_produced := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4no3_h2o_equality_l1223_122366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_praveens_age_l1223_122381

theorem praveens_age : ∃ (age : ℝ), age + 10 = 3 * (age - 3) := by
  use 9.5
  norm_num

#eval (9.5 : ℚ) + 10 = 3 * (9.5 - 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_praveens_age_l1223_122381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical1_is_simplest_l1223_122373

-- Define the quadratic radicals
noncomputable def radical1 : ℝ := -Real.sqrt 3
noncomputable def radical2 : ℝ := Real.sqrt 20
noncomputable def radical3 : ℝ := Real.sqrt (1/2)
noncomputable def radical4 (a : ℝ) : ℝ := Real.sqrt (a^2)

-- Define a function to check if a radical is in its simplest form
def is_simplest_radical (r : ℝ) : Prop := sorry

-- Theorem stating that radical1 is the simplest among the given options
theorem radical1_is_simplest :
  is_simplest_radical radical1 ∧
  (¬ is_simplest_radical radical2) ∧
  (¬ is_simplest_radical radical3) ∧
  (∀ a : ℝ, ¬ is_simplest_radical (radical4 a)) :=
by sorry

#check radical1_is_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical1_is_simplest_l1223_122373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_is_one_monotonically_decreasing_interval_condition_ln_inequality_l1223_122349

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + 2 / (x + 1)

theorem min_value_f_when_a_is_one :
  ∀ x : ℝ, x ≥ 1 → f 1 x ≥ 1 := by
  sorry

theorem monotonically_decreasing_interval_condition (a : ℝ) :
  (∃ x y : ℝ, x < y ∧ x > 0 ∧ y > 0 ∧ f a x > f a y) ↔ a < 1/2 := by
  sorry

theorem ln_inequality (n : ℕ) :
  log (n + 1 : ℝ) > (Finset.range n).sum (λ k => 1 / (2 * (k + 1) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_is_one_monotonically_decreasing_interval_condition_ln_inequality_l1223_122349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_perpendicular_distances_constant_l1223_122334

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  sideLength : ℝ
  /-- The side length is positive -/
  sideLength_pos : 0 < sideLength

/-- An internal point of an equilateral triangle -/
structure InternalPoint (t : EquilateralTriangle) where
  /-- The x-coordinate of the point -/
  x : ℝ
  /-- The y-coordinate of the point -/
  y : ℝ
  /-- The point is inside the triangle -/
  is_internal : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ t.sideLength

/-- The perpendicular distance from a point to a side of the triangle -/
noncomputable def perpendicularDistance (t : EquilateralTriangle) (p : InternalPoint t) (side : Fin 3) : ℝ :=
  sorry

/-- The theorem stating that the sum of perpendicular distances is constant -/
theorem sum_perpendicular_distances_constant (t : EquilateralTriangle) :
  ∃ (c : ℝ), ∀ (p : InternalPoint t),
    perpendicularDistance t p 0 + perpendicularDistance t p 1 + perpendicularDistance t p 2 = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_perpendicular_distances_constant_l1223_122334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1223_122383

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 1 ∨ (1 < x ∧ x ≤ 2)}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1223_122383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_spheres_cover_circumscribed_sphere_l1223_122382

/-- The side length of the cube -/
noncomputable def s : ℝ := sorry

/-- The radius of the circumscribed sphere -/
noncomputable def R : ℝ := s * Real.sqrt 3 / 2

/-- The radius of the inscribed sphere -/
noncomputable def r : ℝ := s / 2

/-- The radius of each of the six smaller spheres tangent to the cube faces -/
noncomputable def R_small : ℝ := s * (Real.sqrt 3 - 1) / 2

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := 4 / 3 * Real.pi * radius ^ 3

theorem seven_spheres_cover_circumscribed_sphere : 
  sphere_volume r + 6 * sphere_volume R_small ≥ sphere_volume R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_spheres_cover_circumscribed_sphere_l1223_122382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1223_122340

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 3)

theorem omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : f ω (Real.pi / 6) = f ω (Real.pi / 3))
  (h3 : ∃ (m : ℝ), ∀ (x : ℝ), Real.pi / 6 < x → x < Real.pi / 3 → f ω x ≥ m)
  (h4 : ¬∃ (M : ℝ), ∀ (x : ℝ), Real.pi / 6 < x → x < Real.pi / 3 → f ω x ≤ M) :
  ω = 14 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1223_122340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1223_122344

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- Define set B
def B : Set ℝ := {x | Real.log x < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1223_122344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_of_two_sets_l1223_122351

theorem combined_mean_of_two_sets (set1 set2 : Finset ℝ) : 
  (set1.card = 4) →
  (set2.card = 8) →
  (Finset.sum set1 id / set1.card = 10) →
  (Finset.sum set2 id / set2.card = 20) →
  (∃ (subset : Finset ℝ), subset ⊆ set2 ∧ subset.card = 5 ∧ Finset.sum subset id = 120) →
  (Finset.sum (set1 ∪ set2) id / (set1 ∪ set2).card = 50/3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_of_two_sets_l1223_122351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1223_122335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + (a + 1) / x + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x : ℝ, x > 0 → DifferentiableAt ℝ (f 1) x) ∧
  (∀ y : ℝ, x - y + Real.log 2 + 4 = 0 ↔ y = (deriv (f 1)) 2 * (x - 2) + (f 1) 2) ∧
  (a > -1/2 →
    ((-1/2 < a ∧ a < 0) →
      (∀ x : ℝ, (0 < x ∧ x < 1) ∨ (-(1 + 1/a) < x) → (deriv (f a)) x < 0) ∧
      (∀ x : ℝ, 1 < x ∧ x < -(1 + 1/a) → (deriv (f a)) x > 0)) ∧
    (a ≥ 0 →
      (∀ x : ℝ, 0 < x ∧ x < 1 → (deriv (f a)) x < 0) ∧
      (∀ x : ℝ, x > 1 → (deriv (f a)) x > 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1223_122335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1223_122313

noncomputable def f (x : ℝ) : ℝ := |Real.log (x + 1)|

theorem solution_exists (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (-((b + 1) / (b + 2))))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = -2/5 ∧ b = -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1223_122313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1223_122358

/-- If the point (sin(5π/6), cos(5π/6)) lies on the terminal side of angle α, then sin(α) = -√3/2 -/
theorem sin_alpha_value (α : ℝ) : 
  (Real.sin (5 * Real.pi / 6), Real.cos (5 * Real.pi / 6)) ∈ 
    {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p = (r * Real.cos α, r * Real.sin α)} →
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1223_122358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cube_theorem_l1223_122356

def cube_side_length : ℕ := 3

def num_unpainted (n : ℕ) : ℕ := (n - 2)^3

def num_one_side (n : ℕ) : ℕ := 6 * (n - 2)^2

def num_two_sides (n : ℕ) : ℕ := 12 * (n - 2)

def num_three_sides : ℕ := 8

theorem painted_cube_theorem :
  let a := num_unpainted cube_side_length
  let b := num_one_side cube_side_length
  let c := num_two_sides cube_side_length
  let d := num_three_sides
  (a : Int) - b - c + d = -9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cube_theorem_l1223_122356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l1223_122332

noncomputable section

/-- The function f(x) = (2x + 3) / (kx - 2) -/
def f (k : ℝ) (x : ℝ) : ℝ := (2 * x + 3) / (k * x - 2)

/-- The set of real numbers k for which f^(-1)(x) = f(x) -/
def valid_k : Set ℝ := {k | k < -4/3 ∨ k > -4/3}

theorem inverse_equals_function (k : ℝ) :
  k ∈ valid_k ↔ Function.Bijective (f k) ∧ ∀ x, (f k) ((f k) x) = x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l1223_122332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1223_122321

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 45) 
  (h2 : goods_train_length = 340) 
  (h3 : passing_time = 8) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 108 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check goods_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1223_122321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l1223_122389

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (x c : ℝ) : Prop :=
  (3 * f (x - 2)) / f c + 4 = f (2 * x + 1)

-- Theorem statement
theorem constant_value :
  ∃ c : ℝ, equation 0.4 c ∧ c = 0 := by
  sorry

#check constant_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l1223_122389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tshirts_count_l1223_122336

/-- Represents the number of blue T-shirts in one pack -/
def blue_tshirts_per_pack : ℕ := 3

theorem blue_tshirts_count : blue_tshirts_per_pack = 3 := by
  -- Define the constants
  let white_packs : ℕ := 2
  let blue_packs : ℕ := 4
  let white_per_pack : ℕ := 5
  let cost_per_shirt : ℕ := 3
  let total_spent : ℕ := 66

  -- Calculate the total number of shirts
  let total_white : ℕ := white_packs * white_per_pack
  let total_blue : ℕ := blue_packs * blue_tshirts_per_pack
  let total_shirts : ℕ := total_white + total_blue

  -- Express the cost equation
  have cost_equation : total_spent = total_shirts * cost_per_shirt := by
    -- The proof of this equation would go here
    sorry

  -- The main proof
  calc
    blue_tshirts_per_pack
    _ = ((total_spent - (white_packs * white_per_pack * cost_per_shirt)) / (blue_packs * cost_per_shirt) : ℕ) := by
      -- The proof of this step would go here
      sorry
    _ = ((66 - (2 * 5 * 3)) / (4 * 3) : ℕ) := by rfl
    _ = 3 := by rfl

  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tshirts_count_l1223_122336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_drove_520_miles_l1223_122347

/-- Represents Karl's car and trip details -/
structure KarlsCar where
  /-- Miles per gallon of Karl's car -/
  mpg : ℚ
  /-- Capacity of Karl's gas tank in gallons -/
  tankCapacity : ℚ
  /-- Initial distance driven in miles -/
  initialDistance : ℚ
  /-- Amount of gas bought during the trip in gallons -/
  gasBought : ℚ
  /-- Fraction of tank full at the end of the trip -/
  endTankFraction : ℚ

/-- Calculates the total distance driven by Karl -/
def totalDistance (car : KarlsCar) : ℚ :=
  car.initialDistance + (car.tankCapacity - car.endTankFraction * car.tankCapacity) * car.mpg

/-- Theorem stating that Karl drove 520 miles -/
theorem karl_drove_520_miles (car : KarlsCar) 
  (h1 : car.mpg = 40)
  (h2 : car.tankCapacity = 12)
  (h3 : car.initialDistance = 400)
  (h4 : car.gasBought = 10)
  (h5 : car.endTankFraction = 3/4) :
  totalDistance car = 520 := by
  sorry

#eval totalDistance { mpg := 40, tankCapacity := 12, initialDistance := 400, gasBought := 10, endTankFraction := 3/4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_drove_520_miles_l1223_122347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l1223_122392

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_of_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l1223_122392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_all_quadrants_l1223_122372

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * a * x^2 - 2 * a * x + 2 * a + 1

/-- Theorem stating the range of a for which f passes through all four quadrants -/
theorem f_passes_through_all_quadrants :
  ∀ a : ℝ, (∃ x1 x2 x3 x4 : ℝ, 
    f a x1 > 0 ∧ x1 > 0 ∧
    f a x2 < 0 ∧ x2 > 0 ∧
    f a x3 > 0 ∧ x3 < 0 ∧
    f a x4 < 0 ∧ x4 < 0) ↔ 
  -6/5 < a ∧ a < -3/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_all_quadrants_l1223_122372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l1223_122377

/-- The function f(x) = ax^2 + 2x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

/-- The domain of x -/
def domain : Set ℝ := Set.Iic 2

theorem f_monotone_implies_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ domain → x₂ ∈ domain → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l1223_122377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l1223_122371

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / x^2 - 2 * x - Real.log x

-- State the theorem
theorem f_derivative_at_one :
  deriv f 1 = -5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l1223_122371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1223_122338

open Real

theorem indefinite_integral_proof (x : ℝ) :
  let f (x : ℝ) := (3 * x^3 + 6 * x^2 + 5 * x - 1) / ((x + 1)^2 * (x^2 + 2))
  let F (x : ℝ) := 1 / (x + 1) + (3 / 2) * log (x^2 + 2) + (1 / Real.sqrt 2) * arctan (x / Real.sqrt 2)
  deriv F x = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1223_122338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_AB_distance_product_l1223_122307

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (3 + t * Real.cos α, t * Real.sin α)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 / Real.cos θ, Real.tan θ)

-- Define the intersection points A and B
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l α t = p ∧ curve_C θ = p}

-- Part 1
theorem midpoint_of_AB (α : ℝ) (h : α = Real.pi / 3) :
  let points := intersection_points α
  ∃ A B, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧ 
    (A.1 + B.1) / 2 = 9 / 2 ∧ 
    (A.2 + B.2) / 2 = 3 * Real.sqrt 3 / 2 :=
sorry

-- Part 2
theorem distance_product (α : ℝ) (h1 : Real.tan α = 2) (h2 : line_l α 0 = (3, 0)) :
  let points := intersection_points α
  ∃ A B, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧ 
    let P := (3, 0)
    let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
    let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
    PA * PB = 40 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_AB_distance_product_l1223_122307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l1223_122378

/-- Represents a player in the game -/
inductive Player : Type
  | One
  | Two

/-- Represents the game state -/
structure GameState :=
  (n : ℕ)  -- length of the strip
  (cells : List Bool)  -- true for 'X', false for 'O', empty list for unused cells

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (pos : ℕ) : Prop :=
  pos < state.n ∧
  pos ∉ state.cells.enum.map Prod.fst ∧
  (pos = 0 ∨ state.cells.get? (pos - 1) ≠ some true) ∧
  (pos = state.n - 1 ∨ state.cells.get? pos ≠ some true)

/-- Defines the winning condition for a player -/
def HasWinningStrategy (n : ℕ) (player : Player) : Prop :=
  ∀ (opponent_strategy : GameState → ℕ),
    ∃ (player_strategy : GameState → ℕ),
      ∀ (game : GameState),
        game.n = n →
        (player = Player.One → ValidMove game (player_strategy game)) ∧
        (player = Player.Two →
          ValidMove game (opponent_strategy game) →
          ValidMove game (player_strategy game))

/-- The main theorem stating the winning strategies for different N -/
theorem winning_strategy :
  (HasWinningStrategy 1 Player.One) ∧
  (∀ n : ℕ, n > 1 → HasWinningStrategy n Player.Two) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l1223_122378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reckha_code_count_l1223_122394

/-- The original code -/
def original_code : Fin 1000 := ⟨145, by norm_num⟩

/-- The set of all possible three-digit codes -/
def all_codes : Finset (Fin 1000) := Finset.univ

/-- Check if two codes share exactly two digits in the same positions -/
def shares_two_digits (a b : Fin 1000) : Bool :=
  let a_digits := [a.val / 100, (a.val / 10) % 10, a.val % 10]
  let b_digits := [b.val / 100, (b.val / 10) % 10, b.val % 10]
  (a_digits.zip b_digits).filter (fun (x, y) => x == y) |>.length ≥ 2

/-- Check if a code is a transposition of the original code -/
def is_transposition (code : Fin 1000) : Bool :=
  let orig_digits := [original_code.val / 100, (original_code.val / 10) % 10, original_code.val % 10]
  let code_digits := [code.val / 100, (code.val / 10) % 10, code.val % 10]
  orig_digits.toFinset == code_digits.toFinset ∧ code ≠ original_code

/-- The set of valid codes for Reckha -/
def valid_codes : Finset (Fin 1000) :=
  all_codes.filter (fun code =>
    ¬(shares_two_digits original_code code) ∧
    ¬(is_transposition code))

theorem reckha_code_count :
  valid_codes.card = 970 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reckha_code_count_l1223_122394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_61_equals_9_l1223_122315

/-- Sequence defined by the given rules -/
def a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then a (n / 2 + 1) else a (n / 2 + 1) + a (n / 2 + 2)

/-- Theorem stating that the 61st term of the sequence is 9 -/
theorem a_61_equals_9 : a 61 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_61_equals_9_l1223_122315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1223_122300

theorem sum_remainder (p q r : ℕ) 
  (hp : p % 15 = 11)
  (hq : q % 15 = 13)
  (hr : r % 15 = 14) :
  (p + q + r) % 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1223_122300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_area_base_edge_l1223_122393

-- Define a regular square pyramid
structure RegularSquarePyramid where
  baseEdge : ℝ
  height : ℝ

-- Define the sphere radius
noncomputable def sphereRadius : ℝ := 1

-- Define the condition that all vertices are on the sphere surface
def verticesOnSphere (p : RegularSquarePyramid) : Prop :=
  2 * p.baseEdge^2 + p.height^2 = 4 * sphereRadius^2

-- Define the lateral surface area of the pyramid
noncomputable def lateralSurfaceArea (p : RegularSquarePyramid) : ℝ :=
  2 * p.baseEdge * p.height

-- State the theorem
theorem max_lateral_surface_area_base_edge :
  ∃ (p : RegularSquarePyramid), 
    verticesOnSphere p ∧ 
    (∀ (q : RegularSquarePyramid), verticesOnSphere q → 
      lateralSurfaceArea q ≤ lateralSurfaceArea p) ∧
    p.baseEdge = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_area_base_edge_l1223_122393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1223_122354

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 2 ∧
  Real.sin t.C = 2 * Real.sin t.B

/-- Helper function to calculate the area of a triangle -/
noncomputable def areaTriangle (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ 
  areaTriangle t = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1223_122354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_between_sine_functions_l1223_122348

/-- The horizontal shift between the graphs of y = 3sin(x + π/3) and y = 3sin(x - π/3) -/
noncomputable def horizontal_shift : ℝ := -2 * Real.pi / 3

/-- The first function: y = 3sin(x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 3)

/-- The second function: y = 3sin(x - π/3) -/
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (x - Real.pi / 3)

theorem shift_between_sine_functions :
  ∀ x : ℝ, f (x + horizontal_shift) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_between_sine_functions_l1223_122348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_abc_exists_sum_abc_l1223_122364

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_abc_exists : ∃! (a b c : ℕ+), 
  x^60 = 3*x^57 + 12*x^55 + 9*x^53 - x^30 + (a : ℝ)*x^26 + (b : ℝ)*x^24 + (c : ℝ)*x^20 := by
  sorry

theorem sum_abc (a b c : ℕ+) 
  (h : x^60 = 3*x^57 + 12*x^55 + 9*x^53 - x^30 + (a : ℝ)*x^26 + (b : ℝ)*x^24 + (c : ℝ)*x^20) :
  a + b + c = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_abc_exists_sum_abc_l1223_122364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_approximation_l1223_122319

noncomputable def ascent : ℝ := 800
noncomputable def old_grade : ℝ := 0.015
noncomputable def new_grade : ℝ := 0.01

noncomputable def old_length : ℝ := ascent / old_grade
noncomputable def new_length : ℝ := ascent / new_grade

noncomputable def additional_length : ℝ := new_length - old_length

theorem additional_track_length_approximation :
  ∃ ε > 0, |additional_length - 26667| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_approximation_l1223_122319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_2018_starting_with_2017_l1223_122326

theorem multiple_of_2018_starting_with_2017 : ∃ n : ℕ, 
  (2018 * n) / 10000 = 2017 ∧ (2018 * n) % 10000 < 10000 := by
  use 9996
  apply And.intro
  · norm_num
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_2018_starting_with_2017_l1223_122326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_height_volume_l1223_122341

/-- The volume of a hemispherical bowl in liters -/
def bowl_volume : ℝ := 8

/-- The radius of the hemispherical bowl -/
noncomputable def bowl_radius (v : ℝ) : ℝ := (3 * v / (2 * Real.pi)) ^ (1/3)

/-- The volume of liquid in a hemispherical bowl filled to a given height -/
noncomputable def liquid_volume (r h : ℝ) : ℝ := (Real.pi / 3) * h^2 * (3*r - h)

/-- Theorem stating that the volume of liquid filling half the height of the bowl is 2.5 liters -/
theorem half_height_volume :
  liquid_volume (bowl_radius bowl_volume) (bowl_radius bowl_volume / 2) = 2.5 := by
  sorry

#eval bowl_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_height_volume_l1223_122341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_octagon_is_135_l1223_122323

/-- The measure of the exterior angle formed by a regular square and a regular octagon sharing a common side -/
def exterior_angle_square_octagon : ℝ :=
  let square_interior_angle : ℝ := 90
  let octagon_interior_angle : ℝ := 135
  let total_angle : ℝ := 360
  total_angle - square_interior_angle - octagon_interior_angle

theorem exterior_angle_square_octagon_is_135 :
  exterior_angle_square_octagon = 135 := by
  unfold exterior_angle_square_octagon
  norm_num

#eval exterior_angle_square_octagon -- Should output 135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_octagon_is_135_l1223_122323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1223_122367

/-- Represents a hyperbola in 2D space -/
structure Hyperbola where
  -- Add necessary fields to define a hyperbola
  dummy : Unit

/-- Represents a line in 2D space -/
structure Line where
  -- Add necessary fields to define a line
  dummy : Unit

/-- Defines the property of a line being tangent to a hyperbola -/
def is_tangent (l : Line) (h : Hyperbola) : Prop :=
  sorry

/-- Defines the property of a line intersecting a hyperbola normally (non-tangentially) -/
def intersects_normally (l : Line) (h : Hyperbola) : Prop :=
  sorry

/-- Counts the number of intersection points between a line and a hyperbola -/
def intersection_count (l : Line) (h : Hyperbola) : ℕ :=
  sorry

/-- The main theorem stating the possible number of intersection points -/
theorem intersection_points_count (h : Hyperbola) (l1 l2 : Line) :
  is_tangent l1 h →
  intersects_normally l2 h →
  (intersection_count l1 h + intersection_count l2 h = 2 ∨
   intersection_count l1 h + intersection_count l2 h = 3) :=
by
  sorry

#check intersection_points_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1223_122367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1223_122308

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the rectangular equation of curve C
def curve_C_rect (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Theorem: The distance between intersection points A and B is √14
theorem intersection_distance :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    curve_C_rect x₁ y₁ ∧ curve_C_rect x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1223_122308
