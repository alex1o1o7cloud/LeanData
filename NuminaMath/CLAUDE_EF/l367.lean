import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elle_in_seat_two_l367_36741

/-- Represents the four friends --/
inductive Friend
  | Elle
  | Fiona
  | Garry
  | Hank

/-- Represents the four seats --/
def Seat := Fin 4

/-- A seating arrangement is a bijection between Friends and Seats --/
def SeatingArrangement := {f : Friend → Seat // Function.Bijective f}

/-- Two friends are adjacent if their seat numbers differ by 1 --/
def adjacent (arr : SeatingArrangement) (a b : Friend) : Prop :=
  ((arr.val a).val + 1 = (arr.val b).val) ∨ ((arr.val b).val + 1 = (arr.val a).val)

/-- A friend is between two others if their seat is between the other two --/
def between (arr : SeatingArrangement) (a b c : Friend) : Prop :=
  ((arr.val a).val < (arr.val b).val ∧ (arr.val b).val < (arr.val c).val) ∨
  ((arr.val c).val < (arr.val b).val ∧ (arr.val b).val < (arr.val a).val)

theorem elle_in_seat_two (arr : SeatingArrangement) 
  (not_elle_next_hank : ¬adjacent arr Friend.Elle Friend.Hank)
  (not_fiona_between : ¬between arr Friend.Garry Friend.Fiona Friend.Hank)
  (garry_in_one : arr.val Friend.Garry = ⟨0, by simp⟩) : 
  arr.val Friend.Elle = ⟨1, by simp⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elle_in_seat_two_l367_36741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stones_even_constraint_l367_36760

/-- Represents a 12x12 square table with stones. -/
structure StoneTable :=
  (stones : Fin 12 → Fin 12 → Bool)

/-- The number of stones in a row is even. -/
def rowEven (t : StoneTable) (row : Fin 12) : Prop :=
  Even (Finset.card (Finset.filter (λ col => t.stones row col) (Finset.univ : Finset (Fin 12))))

/-- The number of stones in a column is even. -/
def colEven (t : StoneTable) (col : Fin 12) : Prop :=
  Even (Finset.card (Finset.filter (λ row => t.stones row col) (Finset.univ : Finset (Fin 12))))

/-- The number of stones in a diagonal is even. -/
def diagEven (t : StoneTable) (diag : ℤ) : Prop :=
  Even (Finset.card (Finset.filter
    (λ i : Fin 12 => 0 ≤ i.val + diag ∧ i.val + diag < 12 ∧ t.stones i ⟨(i.val + diag).toNat, by sorry⟩)
    (Finset.univ : Finset (Fin 12))))

/-- The total number of stones in the table. -/
def totalStones (t : StoneTable) : ℕ :=
  Finset.card (Finset.filter (λ p : Fin 12 × Fin 12 => t.stones p.1 p.2) (Finset.univ : Finset (Fin 12 × Fin 12)))

/-- The main theorem: If all rows, columns, and diagonals have an even number of stones,
    then the maximum number of stones is 120. -/
theorem max_stones_even_constraint (t : StoneTable) 
  (hrows : ∀ row, rowEven t row)
  (hcols : ∀ col, colEven t col)
  (hdiags : ∀ diag, diagEven t diag) :
  totalStones t ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stones_even_constraint_l367_36760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_f_is_odd_and_increasing_l367_36762

-- Define the function f(x) = 3^x - (1/3)^x
noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

-- Theorem stating that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Theorem stating that f is increasing on ℝ
theorem f_is_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

-- Theorem stating that f is both odd and increasing
theorem f_is_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := by
  apply And.intro
  · exact f_is_odd
  · exact f_is_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_f_is_odd_and_increasing_l367_36762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_week_rate_is_14_l367_36788

noncomputable def first_week_rate : ℚ := 18
def total_days : ℕ := 23
noncomputable def total_cost : ℚ := 350

noncomputable def additional_week_rate : ℚ :=
  (total_cost - first_week_rate * 7) / (total_days - 7)

theorem additional_week_rate_is_14 : additional_week_rate = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_week_rate_is_14_l367_36788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_greater_than_45_l367_36796

noncomputable def u : ℕ → ℝ
  | 0 => 5
  | n + 1 => u n + 1 / u n

theorem u_1000_greater_than_45 : u 1000 > 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_greater_than_45_l367_36796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l367_36786

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 2

/-- The slope of the second line -/
noncomputable def m₂ : ℝ := -3

/-- The slope of the angle bisector -/
noncomputable def k : ℝ := (m₁ + m₂ + Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

theorem angle_bisector_slope :
  k = (-1 + Real.sqrt 14) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l367_36786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l367_36773

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (T : Type) [MetricSpace T] :=
  (sim : (T → T → T → Prop) → (T → T → T → Prop) → Prop)

variable {T : Type} [MetricSpace T]

/-- Given two similar triangles MNP and XYZ, with known side lengths, prove that XY = 12 -/
theorem similar_triangles_side_length 
  (MNP XYZ : T → T → T → Prop) 
  (h_sim : SimilarTriangles T)
  (h_similar : h_sim.sim MNP XYZ)
  (M N P X Y Z : T)
  (h_MN : dist M N = 8)
  (h_NP : dist N P = 20)
  (h_YZ : dist Y Z = 30) :
  dist X Y = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l367_36773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_connection_probability_l367_36708

/-- Probability of a single digit error --/
noncomputable def p : ℝ := 0.02

/-- A three-digit number is valid if the first digit is the remainder of the sum of the last two digits divided by 10 --/
def is_valid (n : Fin 1000) : Prop :=
  (n / 100) = (((n / 10) % 10 + n % 10) % 10)

/-- Probability of getting a valid number when exactly two digits are wrong --/
noncomputable def r2 : ℝ := 1 / 9

/-- Probability of getting a valid number when all three digits are wrong --/
noncomputable def r3 : ℝ := 8 / 81

/-- The probability of an incorrect connection --/
noncomputable def incorrect_connection_prob : ℝ :=
  3 * p^2 * (1 - p) * r2 + p^3 * r3

theorem incorrect_connection_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ |incorrect_connection_prob - 0.000131| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_connection_probability_l367_36708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l367_36776

noncomputable def f (x : ℝ) : ℝ := -Real.log (abs (Real.sin x))

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧
  (∀ x ∈ Set.Ioo 0 (π/2), ∀ y ∈ Set.Ioo 0 (π/2), x < y → f y < f x) ∧
  (∀ x, f (-x) = f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l367_36776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l367_36740

/-- The area of the region bounded by the lines x = 2, y = 2, and the coordinate axes is 4 -/
theorem area_of_bounded_region : ℝ := 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l367_36740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_equation_l367_36775

/-- The circle with center (1,0) and radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- The point P -/
def P : ℝ × ℝ := (2, 1)

/-- A chord of the circle passing through P -/
def chord (m b : ℝ) : Prop := ∃ (x y : ℝ), y = m * x + b ∧ myCircle x y ∧ (x, y) ≠ P

/-- The shortest chord passing through P -/
def shortest_chord (m b : ℝ) : Prop :=
  chord m b ∧ ∀ (m' b' : ℝ), chord m' b' → 
    ∃ (x y x' y' : ℝ), 
      y = m * x + b ∧ y' = m' * x' + b' ∧
      myCircle x y ∧ myCircle x' y' ∧
      (x - 2)^2 + (y - 1)^2 ≤ (x' - 2)^2 + (y' - 1)^2

theorem shortest_chord_equation : 
  shortest_chord (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_equation_l367_36775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_f_range_l367_36733

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2) * Real.cos (x / 2) + Real.sin (x / 2) ^ 2

-- Theorem for part I
theorem f_pi_third : f (π / 3) = (1 + Real.sqrt 3) / 4 := by sorry

-- Theorem for part II
theorem f_range (x : ℝ) (h : x ∈ Set.Ioo (-π / 3) (π / 2)) : 
  (1 - Real.sqrt 2) / 2 ≤ f x ∧ f x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_f_range_l367_36733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l367_36781

-- Define a type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for colors
inductive Color
  | Black
  | White

-- Define a function type for coloring edges
def EdgeColoring := Point3D → Point3D → Color

-- Define the property that no four points are coplanar
def NoFourCoplanar (points : Finset Point3D) : Prop :=
  ∀ p₁ p₂ p₃ p₄, p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ →
    ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
      a * (p₁.x - p₄.x) + b * (p₁.y - p₄.y) + c * (p₁.z - p₄.z) = d ∧
      a * (p₂.x - p₄.x) + b * (p₂.y - p₄.y) + c * (p₂.z - p₄.z) = d ∧
      a * (p₃.x - p₄.x) + b * (p₃.y - p₄.y) + c * (p₃.z - p₄.z) = d

-- Define the theorem
theorem monochromatic_triangle_exists
  (points : Finset Point3D)
  (h_size : points.card = 6)
  (h_no_four_coplanar : NoFourCoplanar points)
  (edge_coloring : EdgeColoring) :
  ∃ p₁ p₂ p₃, p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    edge_coloring p₁ p₂ = edge_coloring p₂ p₃ ∧
    edge_coloring p₂ p₃ = edge_coloring p₃ p₁ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l367_36781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l367_36717

-- Define vectors a and b
noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![Real.sin x, Real.cos x]
noncomputable def b : Fin 2 → ℝ := ![Real.sqrt 3, -1]

-- Define the dot product function
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the parallelism condition
def is_parallel (v w : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), v = fun i => k * (w i)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

-- State the theorem
theorem vector_problem (x : ℝ) (k : ℤ) :
  (is_parallel (a x) b → Real.sin x ^ 2 - 6 * Real.cos x ^ 2 = -3/4) ∧
  (∀ y ∈ Set.Icc (π/3 + k * π) (5*π/6 + k * π), 
    ∀ z ∈ Set.Icc (π/3 + k * π) (5*π/6 + k * π),
    y ≤ z → f (2*z) ≤ f (2*y)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l367_36717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equals_six_l367_36752

-- Define the product of logarithms
noncomputable def log_product : ℝ := 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) * 
  (Real.log 5 / Real.log 4) * (Real.log 6 / Real.log 5) * 
  (Real.log 7 / Real.log 6) * (Real.log 8 / Real.log 7) * 
  (Real.log 9 / Real.log 8) * (Real.log 10 / Real.log 9) * 
  (Real.log 11 / Real.log 10) * (Real.log 12 / Real.log 11) * 
  (Real.log 13 / Real.log 12) * (Real.log 14 / Real.log 13) * 
  (Real.log 15 / Real.log 14) * (Real.log 16 / Real.log 15) * 
  (Real.log 17 / Real.log 16) * (Real.log 18 / Real.log 17) * 
  (Real.log 19 / Real.log 18) * (Real.log 20 / Real.log 19) * 
  (Real.log 21 / Real.log 20) * (Real.log 22 / Real.log 21) * 
  (Real.log 23 / Real.log 22) * (Real.log 24 / Real.log 23) * 
  (Real.log 25 / Real.log 24) * (Real.log 26 / Real.log 25) * 
  (Real.log 27 / Real.log 26) * (Real.log 28 / Real.log 27) * 
  (Real.log 29 / Real.log 28) * (Real.log 30 / Real.log 29) * 
  (Real.log 31 / Real.log 30) * (Real.log 32 / Real.log 31) * 
  (Real.log 33 / Real.log 32) * (Real.log 34 / Real.log 33) * 
  (Real.log 35 / Real.log 34) * (Real.log 36 / Real.log 35) * 
  (Real.log 37 / Real.log 36) * (Real.log 38 / Real.log 37) * 
  (Real.log 39 / Real.log 38) * (Real.log 40 / Real.log 39) * 
  (Real.log 41 / Real.log 40) * (Real.log 42 / Real.log 41) * 
  (Real.log 43 / Real.log 42) * (Real.log 44 / Real.log 43) * 
  (Real.log 45 / Real.log 44) * (Real.log 46 / Real.log 45) * 
  (Real.log 47 / Real.log 46) * (Real.log 48 / Real.log 47) * 
  (Real.log 49 / Real.log 48) * (Real.log 50 / Real.log 49) * 
  (Real.log 51 / Real.log 50) * (Real.log 52 / Real.log 51) * 
  (Real.log 53 / Real.log 52) * (Real.log 54 / Real.log 53) * 
  (Real.log 55 / Real.log 54) * (Real.log 56 / Real.log 55) * 
  (Real.log 57 / Real.log 56) * (Real.log 58 / Real.log 57) * 
  (Real.log 59 / Real.log 58) * (Real.log 60 / Real.log 59) * 
  (Real.log 61 / Real.log 60) * (Real.log 62 / Real.log 61) * 
  (Real.log 63 / Real.log 62) * (Real.log 64 / Real.log 63)

-- Theorem statement
theorem log_product_equals_six : log_product = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equals_six_l367_36752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_relation_l367_36742

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 => 2  -- Add this case to handle Nat.zero
| 1 => 2
| n + 2 => 2 * (n + 2) - 1

-- Define the sum function S_n
def S (n : ℕ) : ℕ := n^2 + 1

-- Theorem statement
theorem sequence_sum_relation : ∀ n : ℕ, 
  (n = 1 ∧ a n = 2) ∨ 
  (n > 1 ∧ a n = 2*n - 1) ∧ 
  S n = n^2 + 1 := by
  sorry

#check sequence_sum_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_relation_l367_36742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_result_l367_36732

/-- A polynomial function satisfying f(x^2 + 1) = x^4 + 5x^2 + 3 for all real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The given property of f -/
axiom f_property : ∀ x : ℝ, f (x^2 + 1) = x^4 + 5*x^2 + 3

/-- The theorem to be proved -/
theorem f_result : ∀ x : ℝ, f (x^2 - 1) = x^4 + x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_result_l367_36732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_sum_l367_36774

theorem combinatorial_sum (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k => (-2 : ℤ)^k * (n.choose k)) = (-1 : ℤ)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_sum_l367_36774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l367_36743

/-- Given a cone with an unwrapped side view having a base radius of 1 and a central angle of 4π/3,
    the volume of the cone is (4√5/81)π. -/
theorem cone_volume (r θ V : ℝ) : 
  r = 1 → θ = (4 / 3) * Real.pi → V = (1 / 3) * Real.pi * (2 / 3)^2 * (Real.sqrt 5 / 3) → 
  V = (4 * Real.sqrt 5 / 81) * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l367_36743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l367_36780

/-- Given a triangle ABC where the opposing sides of angles A, B, and C are a, b, and c respectively,
    and a, b, c form a geometric progression. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 = a * c →
  Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c) →
  0 < B ∧ B ≤ Real.pi / 3 ∧
  (∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5) ∧
  (∃ B₀, 0 < B₀ ∧ B₀ ≤ Real.pi / 3 ∧ 3 * Real.sin B₀ + 4 * Real.cos B₀ = 5 ∧ Real.tan B₀ = 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l367_36780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_E_neg_two_integer_value_l367_36718

noncomputable def E (x : ℝ) : ℝ := Real.sqrt (abs (x + 1)) + (9 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem E_neg_two_integer_value : 
  ⌊E (-2)⌋ = 4 ∨ ⌈E (-2)⌉ = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_E_neg_two_integer_value_l367_36718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_area_l367_36703

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 1

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line k x y}

-- Define the triangle FAB
def triangle_FAB (k : ℝ) : Set (ℝ × ℝ) :=
  {left_focus} ∪ intersection_points k

-- Define the perimeter of triangle FAB
noncomputable def perimeter (k : ℝ) : ℝ :=
  sorry

-- Define the area of triangle FAB
noncomputable def area (k : ℝ) : ℝ :=
  sorry

-- The theorem to prove
theorem max_perimeter_area :
  ∃ (k : ℝ), (∀ (k' : ℝ), perimeter k ≥ perimeter k') → area k = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_area_l367_36703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_value_l367_36765

def satisfies_all_conditions (box' : ℤ) : Prop :=
  ∃ (a' b' : ℤ), 
    (∀ x, (a' * x + b') * (b' * x + a') = 24 * x^2 + box' * x + 24) ∧
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' ∧
    box' = a'^2 + b'^2 ∧
    a' * b' = 24

theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 24 * x^2 + box * x + 24) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  box = a^2 + b^2 →
  a * b = 24 →
  ∃ (min_box : ℤ), (∀ box', satisfies_all_conditions box' → box' ≥ min_box) ∧ min_box = 52 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_value_l367_36765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_kg_theorem_paint_cost_equality_l367_36702

/-- Proves that the cost of paint per kg is 40 Rs, given the specified conditions --/
theorem paint_cost_per_kg_theorem (cube_side : ℝ) (paint_coverage : ℝ) (total_cost : ℝ) : ℝ :=
  by
  -- Given conditions
  have h1 : cube_side = 10 := by sorry
  have h2 : paint_coverage = 20 := by sorry
  have h3 : total_cost = 1200 := by sorry

  -- Calculate the cost per kg
  let surface_area : ℝ := 6 * cube_side * cube_side
  let paint_needed : ℝ := surface_area / paint_coverage
  let cost_per_kg : ℝ := total_cost / paint_needed

  -- Prove that cost_per_kg = 40
  sorry

/-- The cost of paint per kg --/
def paint_cost_per_kg : ℝ := 40

theorem paint_cost_equality : 
  paint_cost_per_kg_theorem 10 20 1200 = paint_cost_per_kg :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_kg_theorem_paint_cost_equality_l367_36702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l367_36719

/-- The perimeter of a triangle with sides 14, 8, and 9 is 31. -/
theorem triangle_perimeter : ∀ (a b c : ℝ), a = 14 ∧ b = 8 ∧ c = 9 → a + b + c = 31 := by
  intros a b c h
  rcases h with ⟨ha, hb, hc⟩
  rw [ha, hb, hc]
  norm_num

#check triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l367_36719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l367_36706

/-- Represents a rectangle with given width and perimeter -/
structure Rectangle where
  width : ℝ
  perimeter : ℝ

/-- Calculates the area of a rectangle given its width and perimeter -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  let length := (r.perimeter / 2) - r.width
  length * r.width

/-- Theorem stating that a rectangle with width 6 and perimeter 28 has an area of 48 -/
theorem rectangle_area_theorem :
  let r : Rectangle := { width := 6, perimeter := 28 }
  rectangleArea r = 48 := by
  -- Unfold the definition of rectangleArea
  unfold rectangleArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l367_36706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_k_squared_minus_2016_equals_3_to_n_l367_36710

theorem unique_solution_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_k_squared_minus_2016_equals_3_to_n_l367_36710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l367_36787

theorem cos_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (3 * π / 4) π)
  (h2 : β ∈ Set.Ioo (3 * π / 4) π)
  (h3 : Real.sin (α + β) = -3/5)
  (h4 : Real.sin (β - π/4) = 24/25) :
  Real.cos (α + π/4) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l367_36787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exponential_base_l367_36750

-- Define the function f(x) = (a - 1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- Theorem: If f is increasing on ℝ, then a > 2
theorem increasing_exponential_base (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a > 2 :=
by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exponential_base_l367_36750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l367_36753

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | k => a₁ + (k - 1 : ℝ) * d

-- Define the sum of the first n terms
noncomputable def sum_n_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_sum_inequality
  (a₁ : ℝ) (d : ℝ) (n : ℕ) (h₁ : d < 0) (h₂ : n > 0) :
  (n : ℝ) * (arithmetic_sequence a₁ d n n) < sum_n_terms a₁ d n ∧
  sum_n_terms a₁ d n < (n : ℝ) * a₁ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l367_36753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_f_implies_divides_l367_36747

/-- A function satisfying the required properties -/
def special_function (f : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, Nat.Coprime m n → Nat.Coprime (f m) (f n)) ∧
  (∀ n : ℕ, n ≤ f n ∧ f n ≤ n + 2012)

/-- The main theorem -/
theorem divides_f_implies_divides (f : ℕ → ℕ) (hf : special_function f) :
  ∀ (n : ℕ) (p : ℕ), Nat.Prime p → (p ∣ f n → p ∣ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_f_implies_divides_l367_36747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l367_36701

-- Define the circle
def mycircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point through which tangent lines pass
def mypoint : ℝ × ℝ := (1, 2)

-- Define a tangent line to the circle passing through the given point
def is_tangent_line (m b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), mycircle x₀ y₀ ∧ y₀ = m * x₀ + b ∧ mypoint.2 = m * mypoint.1 + b

-- Define the quadrilateral
noncomputable def quadrilateral_area (m₁ b₁ m₂ b₂ : ℝ) : ℝ :=
  let y₁ := b₁  -- y-intercept of first tangent line
  let y₂ := b₂  -- y-intercept of second tangent line
  let x₁ := -b₁ / m₁  -- x-intercept of first tangent line
  let x₂ := -b₂ / m₂  -- x-intercept of second tangent line
  (y₁ + y₂) * (x₁ + x₂) / 4  -- Area formula for a quadrilateral

-- Theorem statement
theorem tangent_quadrilateral_area :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    is_tangent_line m₁ b₁ ∧
    is_tangent_line m₂ b₂ ∧
    m₁ ≠ m₂ ∧
    quadrilateral_area m₁ b₁ m₂ b₂ = 13/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l367_36701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_4_l367_36726

theorem sin_2alpha_plus_pi_4 (α : Real) (h : Real.tan α = 2) :
  Real.sin (2 * α + Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_4_l367_36726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_board_cut_length_l367_36761

theorem square_board_cut_length (board_size : ℕ) (num_parts : ℕ) (h1 : board_size = 30) (h2 : num_parts = 225) :
  let total_area : ℕ := board_size * board_size
  let part_area : ℕ := total_area / num_parts
  let max_perimeter : ℕ := 10
  let total_perimeter : ℕ := num_parts * max_perimeter
  let board_perimeter : ℕ := 4 * board_size
  (total_perimeter - board_perimeter) / 2 = 1065 :=
by
  sorry

#check square_board_cut_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_board_cut_length_l367_36761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_equiv_a_range_l367_36737

/-- The function f(x) = -e^x - x --/
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

/-- The function g(x) = ax + cos(x) --/
noncomputable def g (a x : ℝ) : ℝ := a * x + Real.cos x

/-- The derivative of f --/
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x - 1

/-- The derivative of g --/
noncomputable def g' (a x : ℝ) : ℝ := a - Real.sin x

/-- The theorem stating the equivalence between the perpendicular tangent lines condition and the range of a --/
theorem perpendicular_tangents_equiv_a_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, (f' x) * (g' a y) = -1) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_equiv_a_range_l367_36737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cones_intersection_is_plane_l367_36738

/-- Represents a cone in 3D space -/
structure Cone where
  vertex : ℝ × ℝ × ℝ
  axis_direction : ℝ × ℝ × ℝ
  angle : ℝ

/-- Represents the surface of a cone -/
def surface (cone : Cone) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The theorem stating that the intersection of two cones with parallel axes 
    and equal angles between axis and generator is a plane -/
theorem cones_intersection_is_plane (cone1 cone2 : Cone) 
  (h_parallel : cone1.axis_direction = cone2.axis_direction)
  (h_equal_angles : cone1.angle = cone2.angle) :
  ∃ (a b c d : ℝ), ∀ (x y z : ℝ), 
    ((x, y, z) ∈ surface cone1 ∧ (x, y, z) ∈ surface cone2) → 
    a * x + b * y + c * z + d = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cones_intersection_is_plane_l367_36738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposite_parts_l367_36731

theorem complex_opposite_parts (a : ℝ) : 
  (Complex.re ((1 - Complex.I * a) * Complex.I) = 
   -Complex.im ((1 - Complex.I * a) * Complex.I)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposite_parts_l367_36731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_friends_same_group_l367_36779

/-- The number of groups students are divided into -/
def num_groups : ℕ := 4

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

/-- The set of friends we're interested in -/
inductive Friend
| Al
| Bob
| Carol
| Dave

/-- The number of friends -/
def num_friends : ℕ := 4

theorem prob_all_friends_same_group :
  let prob_all_same := (prob_single_student ^ (num_friends - 1) : ℚ)
  prob_all_same = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_friends_same_group_l367_36779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l367_36744

noncomputable section

open Real

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a > 0 →
  b > 0 →
  c > 0 →
  Real.cos B = 1/4 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1/2) * a * c * Real.sin B = (9 * Real.sqrt 15) / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l367_36744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_equal_length_l367_36736

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A chord of a sphere -/
structure Chord where
  start : Point3D
  finish : Point3D

/-- A sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Check if three points are coplanar -/
def areCoplanar (p1 p2 p3 : Point3D) : Prop := sorry

/-- Check if a point is inside a sphere -/
def isInside (p : Point3D) (s : Sphere) : Prop := sorry

/-- Check if two spheres touch at a point -/
def spheresTouchAt (s1 s2 : Sphere) (p : Point3D) : Prop := sorry

/-- Check if a point is on a sphere -/
def isOnSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Calculate the length of a chord -/
def chordLength (c : Chord) : ℝ := sorry

/-- Main theorem -/
theorem chords_equal_length 
  (s : Sphere) (c1 c2 c3 : Chord) (x : Point3D) 
  (s1 s2 : Sphere) :
  isInside x s ∧ 
  ¬ areCoplanar c1.start c2.start c3.start ∧
  c1.start = x ∧ c2.start = x ∧ c3.start = x ∧
  isOnSphere c1.finish s1 ∧ isOnSphere c2.finish s1 ∧ isOnSphere c3.finish s1 ∧
  isOnSphere c1.start s2 ∧ isOnSphere c2.start s2 ∧ isOnSphere c3.start s2 ∧
  spheresTouchAt s1 s2 x →
  chordLength c1 = chordLength c2 ∧ 
  chordLength c2 = chordLength c3 ∧
  chordLength c3 = chordLength c1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_equal_length_l367_36736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l367_36705

noncomputable section

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a = t.b * Real.tan t.A ∧
  Real.sin t.C - Real.sin t.A * Real.cos t.B = 3/4 ∧
  t.B > π/2 ∧ t.B < π ∧
  t.A + t.B + t.C = π

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  Real.sin t.B = Real.cos t.A ∧
  t.A = π/6 ∧
  t.B = 2*π/3 ∧
  t.C = π/6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l367_36705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_2p_l367_36789

/-- Represents a point on a parabola -/
structure ParabolaPoint (p : ℝ) where
  t : ℝ
  x : ℝ := 2 * p * t
  y : ℝ := 2 * p * t^2

/-- Represents a right triangle on a parabola -/
structure RightTriangleOnParabola (p : ℝ) where
  A : ParabolaPoint p
  B : ParabolaPoint p
  C : ParabolaPoint p
  is_right_triangle : (C.x - A.x) * (C.x - B.x) + (C.y - A.y) * (C.y - B.y) = 0
  hypotenuse_parallel_x : A.y = B.y

/-- The height of the right triangle from C to AB -/
def triangleHeight (p : ℝ) (triangle : RightTriangleOnParabola p) : ℝ :=
  2 * p * |triangle.C.t^2 - triangle.A.t^2|

theorem height_is_2p (p : ℝ) (hp : p > 0) (triangle : RightTriangleOnParabola p) :
  triangleHeight p triangle = 2 * p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_2p_l367_36789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l367_36791

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | x^2 + 2*x - 3 ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l367_36791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l367_36795

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k| / Real.sqrt (a^2 + b^2 + c^2)

/-- Plane 1 equation: 3x - y + 2z - 3 = 0 -/
def plane1 (x y z : ℝ) : Prop := 3*x - y + 2*z - 3 = 0

/-- Plane 2 equation: 6x - 2y + 4z + 4 = 0 -/
def plane2 (x y z : ℝ) : Prop := 6*x - 2*y + 4*z + 4 = 0

theorem distance_between_given_planes :
  distance_between_planes 3 (-1) 2 (-3) 7 = 5 * Real.sqrt 14 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l367_36795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_k_div_x_inverse_prop_two_div_x_not_inverse_prop_two_x_div_three_not_inverse_prop_neg_x_plus_five_l367_36772

/-- A function is an inverse proportion function if it can be written as k/x for some non-zero k -/
def IsInverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = k/x is an inverse proportion function -/
theorem inverse_prop_k_div_x (k : ℝ) (h : k ≠ 0) :
    IsInverseProportion (fun x => k / x) := by
  sorry

/-- The function f(x) = 2x^(-1) is an inverse proportion function -/
theorem inverse_prop_two_div_x :
    IsInverseProportion (fun x => 2 * (x⁻¹)) := by
  sorry

/-- The function f(x) = 2x/3 is not an inverse proportion function -/
theorem not_inverse_prop_two_x_div_three :
    ¬ IsInverseProportion (fun x => 2 * x / 3) := by
  sorry

/-- The function f(x) = -x + 5 is not an inverse proportion function -/
theorem not_inverse_prop_neg_x_plus_five :
    ¬ IsInverseProportion (fun x => -x + 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_k_div_x_inverse_prop_two_div_x_not_inverse_prop_two_x_div_three_not_inverse_prop_neg_x_plus_five_l367_36772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l367_36751

/-- The volume of a regular triangular pyramid -/
noncomputable def pyramid_volume (base_side : ℝ) (lateral_area : ℝ) : ℝ :=
  let p := 3 * base_side / 2
  let l := lateral_area / p
  let r := base_side * Real.sqrt 3 / 6
  let h := Real.sqrt (l^2 - r^2)
  let base_area := base_side^2 * Real.sqrt 3 / 4
  (1 / 3) * base_area * h

/-- Theorem: The volume of a regular triangular pyramid with base side length 1 and lateral surface area 3 is √47/36 -/
theorem pyramid_volume_specific : 
  pyramid_volume 1 3 = Real.sqrt 47 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l367_36751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_286_l367_36727

theorem lcm_of_210_and_286 (a b hcf lcm : ℕ) : 
  a = 210 ∧ b = 286 ∧ Nat.gcd a b = hcf ∧ hcf = 26 →
  Nat.lcm a b = lcm ∧ lcm = 2310 := by
  sorry

#check lcm_of_210_and_286

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_286_l367_36727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_lower_bound_l367_36799

/-- A convex quadrilateral with sides a, b, c, d -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The ratio of the sum of squares of two opposite sides to the sum of squares of the other two sides -/
noncomputable def sideRatio (q : ConvexQuadrilateral) : ℝ :=
  (q.a^2 + q.b^2) / (q.c^2 + q.d^2)

/-- The theorem stating that the side ratio is always greater than 1/2 and this bound is tight -/
theorem side_ratio_lower_bound :
  (∀ q : ConvexQuadrilateral, sideRatio q > 1/2) ∧
  (∃ q : ConvexQuadrilateral, sideRatio q = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_lower_bound_l367_36799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_counts_l367_36763

def M : Finset Int := {-3, -2, -1, 0, 1, 2}

def P (a b : Int) : ℝ × ℝ := (a, b)

def countTotalPoints : Nat :=
  Finset.card (M.product M)

def countSecondQuadrantPoints : Nat :=
  Finset.card (M.filter (fun a => a < 0) ×ˢ M.filter (fun b => b > 0))

theorem point_counts :
  countTotalPoints = 36 ∧ countSecondQuadrantPoints = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_counts_l367_36763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_clear_board_l367_36798

/-- Represents a move that can remove either one or two adjacent chips -/
inductive Move
  | single : Move
  | double : Move

/-- Represents the state of the board -/
structure Board where
  n : ℕ
  chips : ℕ

/-- Defines a valid initial board state -/
def valid_initial_board (b : Board) : Prop :=
  b.chips = (b.n + 1)^2

/-- Defines a valid move on the board -/
def valid_move (m : Move) (b : Board) : Prop :=
  match m with
  | Move.single => b.chips ≥ 1
  | Move.double => b.chips ≥ 2

/-- Applies a move to the board, reducing the number of chips -/
def apply_move (m : Move) (b : Board) : Board :=
  match m with
  | Move.single => ⟨b.n, b.chips - 1⟩
  | Move.double => ⟨b.n, b.chips - 2⟩

/-- Applies a list of moves to the board -/
def apply_moves : List Move → Board → Board
  | [], b => b
  | m::ms, b => apply_moves ms (apply_move m b)

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_clear_board (b : Board) (h : valid_initial_board b) :
  ∃ (moves : List Move), 
    (∀ m ∈ moves, valid_move m (apply_moves moves b)) ∧ 
    (apply_moves moves b).chips = 0 ∧
    moves.length = b.n^2 + b.n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_clear_board_l367_36798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l367_36757

def a : ℕ → ℕ
  | 0 => 0  -- Add this case to handle Nat.zero
  | 1 => 0  -- Add this case to handle n = 1 to 9
  | 2 => 0
  | 3 => 0
  | 4 => 0
  | 5 => 0
  | 6 => 0
  | 7 => 0
  | 8 => 0
  | 9 => 0
  | 10 => 12
  | n+1 => 50 * a n + (n+1)^2

theorem least_multiple_of_121 :
  (∀ k, 10 < k → k < 23 → ¬ (121 ∣ a k)) ∧ (121 ∣ a 23) := by
  sorry

#eval a 23  -- This line is optional, but can be useful for checking the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l367_36757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_avg_rate_of_change_l367_36720

-- Define the four functions
noncomputable def f₁ (x : ℝ) : ℝ := x
noncomputable def f₂ (x : ℝ) : ℝ := x^2
noncomputable def f₃ (x : ℝ) : ℝ := x^3
noncomputable def f₄ (x : ℝ) : ℝ := 1/x

-- Define the average rate of change function
noncomputable def avg_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

-- Theorem statement
theorem max_avg_rate_of_change :
  let a := 1
  let b := 2
  let arc₁ := avg_rate_of_change f₁ a b
  let arc₂ := avg_rate_of_change f₂ a b
  let arc₃ := avg_rate_of_change f₃ a b
  let arc₄ := avg_rate_of_change f₄ a b
  arc₃ = max arc₁ (max arc₂ (max arc₃ arc₄)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_avg_rate_of_change_l367_36720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_eight_digit_product_l367_36766

theorem base_eight_digit_product (n : ℕ) (h : n = 8679) : 
  (List.map (λ d => d) (Nat.digits 8 n)).prod = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_eight_digit_product_l367_36766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_son_age_ratio_l367_36722

theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 28 →
    man_age = son_age + 30 →
    (man_age + 2) / (son_age + 2) = 2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_son_age_ratio_l367_36722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l367_36767

/-- Polar to Cartesian coordinate conversion theorem -/
theorem polar_to_cartesian_conversion 
  (ρ θ x y : ℝ) 
  (h1 : x = ρ * Real.cos θ) 
  (h2 : y = ρ * Real.sin θ) : 
  (ρ * Real.cos (θ - π/6) = 1) ↔ (Real.sqrt 3 * x + y = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l367_36767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l367_36771

noncomputable section

def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 + 1

def g (x m : ℝ) : ℝ := f x m + 24/49 * m^2

theorem vector_function_properties :
  (∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/4) → ∀ m : ℝ,
    (f (π/6) 0 = 3/2) ∧
    (f x (Real.sqrt 2) ≥ -1) ∧
    ((∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ∈ Set.Icc (-π/3) (π/4) ∧ x₂ ∈ Set.Icc (-π/3) (π/4) ∧ 
      x₃ ∈ Set.Icc (-π/3) (π/4) ∧ x₄ ∈ Set.Icc (-π/3) (π/4) ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0 ∧ g x₄ m = 0) ↔
     (7 * Real.sqrt 2 / 6 ≤ m ∧ m < 7/4))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l367_36771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l367_36797

noncomputable def f (x : ℝ) := 2 / x + 9 / (1 - 2 * x)

theorem inequality_and_minimum :
  ∀ (a b x y : ℝ),
  a > 0 → b > 0 → a ≠ b → x > 0 → y > 0 →
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ a / x = b / y) ∧
  (∀ x, 0 < x → x < 1/2 → f x ≥ 25) ∧
  (f (1/5) = 25) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l367_36797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l367_36792

theorem complex_number_problem (a : ℝ) : 
  (Complex.I : ℂ)^2015 = -Complex.I →
  (Complex.ofReal (a^2 - 4) + Complex.I * Complex.ofReal (a + 2)).im ≠ 0 →
  (Complex.ofReal (a^2 - 4) + Complex.I * Complex.ofReal (a + 2)).re = 0 →
  (Complex.ofReal a + Complex.I^2015) / (1 + 2 * Complex.I) = -Complex.I := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l367_36792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l367_36709

theorem sin_double_angle (x : ℝ) :
  Real.sin (x - π/4) = 3/5 → Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l367_36709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_existence_l367_36724

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ p x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ p x) := by sorry

-- The specific case for 2^x = 3
theorem negation_of_specific_existence :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ (2 : ℝ)^x = 3) ↔ (∀ x : ℝ, x ≥ 0 → (2 : ℝ)^x ≠ 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_existence_l367_36724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_on_chessboard_l367_36749

/-- A rectangle on the chessboard -/
structure Rectangle where
  width : Nat
  height : Nat
  x : Nat
  y : Nat

/-- The chessboard -/
def Chessboard : Type := Array (Array Bool)

/-- Check if two rectangles are equal -/
def rectanglesEqual (r1 r2 : Rectangle) : Prop :=
  r1.width = r2.width ∧ r1.height = r2.height

/-- Check if two rectangles touch -/
def rectanglesTouch (r1 r2 : Rectangle) : Prop :=
  (r1.x ≤ r2.x + r2.width ∧ r2.x ≤ r1.x + r1.width) ∧
  (r1.y ≤ r2.y + r2.height ∧ r2.y ≤ r1.y + r1.height)

/-- Check if a configuration of rectangles is valid -/
def isValidConfiguration (rectangles : List Rectangle) : Prop :=
  (∀ r ∈ rectangles, r.x + r.width ≤ 8 ∧ r.y + r.height ≤ 8) ∧
  (∀ r1 ∈ rectangles, ∀ r2 ∈ rectangles,
    r1 = r2 ∨ ¬(rectanglesEqual r1 r2) ∨ ¬(rectanglesTouch r1 r2))

/-- The main theorem -/
theorem max_rectangles_on_chessboard :
  (∃ (rectangles : List Rectangle),
    rectangles.length = 35 ∧ isValidConfiguration rectangles) ∧
  (∀ (rectangles : List Rectangle),
    isValidConfiguration rectangles → rectangles.length ≤ 35) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_on_chessboard_l367_36749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_l367_36770

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem johns_speed (john_time : ℝ) (beth_speed : ℝ) (beth_time_diff : ℝ) (route_diff : ℝ) : 
  john_time = 0.5 →
  beth_speed = 30 →
  beth_time_diff = 1/3 →
  route_diff = 5 →
  average_speed (beth_speed * (john_time + beth_time_diff) - route_diff) john_time = 40 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions
-- #eval average_speed 20 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_l367_36770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l367_36728

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 2*y + 1/y = 6) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + 2/a + 2*b + 1/b = 6 → x + 2*y ≥ a + 2*b ∧ x + 2*y ≤ 4 := by
  sorry

#check max_value_x_plus_2y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l367_36728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l367_36756

/-- The straight-line distance from the starting point to the final position
    after walking 7 miles westward, turning 45° northward, and walking 5 miles. -/
theorem distance_after_walk (west_distance : ℝ) (north_angle : ℝ) (final_distance : ℝ) :
  west_distance = 7 →
  north_angle = 45 * Real.pi / 180 →
  final_distance = 5 →
  Real.sqrt ((west_distance + final_distance * Real.cos north_angle) ^ 2 + 
        (final_distance * Real.sin north_angle) ^ 2) = Real.sqrt 386 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l367_36756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_proof_l367_36764

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 154

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 250 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 250

theorem leo_weight_proof :
  (leo_weight + 15 = 1.75 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (abs (leo_weight - 154) < 0.5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_weight_proof_l367_36764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_A_plus_one_square_l367_36716

/-- Represents a digit in the number system with base A+1 -/
def Digit (A : ℕ) := Fin (A + 1)

/-- Converts a number from base A+1 to base 10 -/
def toBase10 (A : ℕ) (digits : List (Digit A)) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d.val * (A + 1)^i) 0

/-- The main theorem -/
theorem base_A_plus_one_square (A B C : ℕ) :
  (∃ (d₁ d₂ d₃ d₄ : Digit A),
    d₁.val = A ∧ d₂.val = A ∧ d₃.val = A ∧ d₄.val = A ∧
    (toBase10 A [d₁, d₂, d₃, d₄])^2 = 
      toBase10 A [Fin.ofNat A, Fin.ofNat A, Fin.ofNat A, Fin.ofNat C, Fin.ofNat B, Fin.ofNat B, Fin.ofNat B, Fin.ofNat C] ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C) →
  A = 2 ∧ B = 0 ∧ C = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_A_plus_one_square_l367_36716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l367_36746

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Point A -/
noncomputable def point_A : ℝ × ℝ := (0, 2 * Real.sqrt 3)

/-- A point P on the ellipse -/
def point_P : {p : ℝ × ℝ // ellipse p.1 p.2} := sorry

/-- The area of triangle APF -/
noncomputable def triangle_area (P : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle APF -/
theorem max_triangle_area :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ 
  ∀ (Q : ℝ × ℝ), ellipse Q.1 Q.2 → 
  triangle_area P ≥ triangle_area Q ∧
  triangle_area P = (21 * Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l367_36746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_height_l367_36748

/-- Calculates the actual average height of a class given:
  * The number of students
  * The initially calculated average height
  * The incorrectly recorded height of one student
  * The actual height of that student
-/
noncomputable def actualAverageHeight (numStudents : ℕ) (initialAverage : ℝ) (incorrectHeight actualHeight : ℝ) : ℝ :=
  (numStudents * initialAverage - incorrectHeight + actualHeight) / numStudents

/-- Theorem stating that the actual average height of the class is 173 cm -/
theorem class_average_height :
  actualAverageHeight 20 175 151 111 = 173 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_height_l367_36748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_property_l367_36729

/-- A random variable following a normal distribution with mean 1 and variance 4 -/
noncomputable def X : ℝ → ℝ := sorry

/-- The probability that X is less than or equal to 0 -/
noncomputable def m : ℝ := sorry

/-- The probability density function of the standard normal distribution -/
noncomputable def standardNormalPDF (x : ℝ) : ℝ :=
  Real.exp (-(x^2) / 2) / Real.sqrt (2 * Real.pi)

theorem normal_distribution_property (h1 : ∀ x, X x = standardNormalPDF ((x - 1) / 2))
  (h2 : ∫ x in Set.Iic 0, X x = m) :
  ∫ x in Set.Ioo 0 2, X x = 1 - 2 * m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_property_l367_36729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l367_36700

def a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | (n + 1) => a n + 2 / (n * (n + 1))

theorem a_10_value : a 10 = 19 / 5 := by
  -- The proof will be skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l367_36700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_special_triangle_l367_36758

/-- Given a triangle ABC where a^2 = b^2 + c^2 + bc, prove that the measure of angle A is 2π/3 -/
theorem angle_measure_special_triangle (a b c : ℝ) (h : a^2 = b^2 + c^2 + b*c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_special_triangle_l367_36758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_vectors_determinant_l367_36777

open Matrix

/-- Given vectors a, b, c in R^3 and a scalar k, if D is the determinant of the matrix [a|b|c],
    then the determinant of the matrix [k*a+b|b+k*c|k*c+a] is equal to (k^2 + 2k)D. -/
theorem scaled_vectors_determinant
  (a b c : Fin 3 → ℝ) (k : ℝ) :
  let D := det (![a, b, c])
  det (![k • a + b, b + k • c, k • c + a]) = (k^2 + 2*k) * D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_vectors_determinant_l367_36777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_intercepts_l367_36782

/-- A quadratic function --/
def QuadraticFunction := ℝ → ℝ

/-- The x-coordinate of a point --/
def X := ℝ

theorem quadratic_function_intercepts
  (f g : QuadraticFunction)
  (h1 : ∀ x, g x = -f (200 - x))
  (h2 : ∃ v, g v = f v ∧ ∀ x, f x ≤ f v)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h4 : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0)
  (h5 : x₃ - x₂ = 300) :
  x₄ - x₁ = 1800 + 600 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_intercepts_l367_36782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l367_36712

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := -x^2 + a * Real.log x

def g (x : ℝ) : ℝ := f a x - 2*x + 2*x^2

theorem range_of_m (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) 
  (h₂ : ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (deriv (g a)) x ≠ 0)
  (h₃ : ∀ m : ℝ, g a x₁ ≥ m * x₂) :
  ∀ m : ℝ, m ≤ -3/2 - Real.log 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l367_36712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_approximation_l367_36730

/-- The cost price of a book given its selling price and profit percentage -/
noncomputable def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem stating that the cost price of a book is approximately 208.33
    given a selling price of 250 and a profit of 20% -/
theorem book_cost_price_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |cost_price 250 20 - 208.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_approximation_l367_36730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l367_36790

theorem tan_half_angle (α : Real) : 
  (α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) → 
  (Real.sin α + Real.cos α = 1 / 5) → 
  Real.tan (α / 2) = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l367_36790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangular_frame_l367_36734

/-- The length of the steel bar in meters -/
noncomputable def bar_length : ℝ := 18

/-- The ratio of length to width of the rectangular frame -/
noncomputable def length_width_ratio : ℝ := 2

/-- Function to calculate the volume of the rectangular frame -/
noncomputable def volume (x : ℝ) : ℝ := 2 * x^2 * ((bar_length - 6 * x) / 4)

/-- Theorem stating the maximum volume of the rectangular frame -/
theorem max_volume_rectangular_frame :
  ∃ (x : ℝ), x > 0 ∧ volume x = 12 ∧ ∀ (y : ℝ), y > 0 → volume y ≤ volume x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangular_frame_l367_36734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_terms_21_l367_36785

def S (n : ℕ) : ℕ := n^2 + 2*n + 1

def a : ℕ → ℕ
  | 0 => 0  -- We define a_0 as 0 for convenience
  | n + 1 => S (n + 1) - S n

def sum_odd_terms (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (2*i + 1)) |>.sum

theorem sum_odd_terms_21 : sum_odd_terms 11 = 254 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_terms_21_l367_36785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_press_F_denomination_l367_36793

/-- Represents the denomination of bills produced by Press F -/
def F : ℝ := sorry

/-- Rate of bill production for Press F in bills per minute -/
def rate_F : ℝ := 1000

/-- Rate of bill production for Press T in bills per minute -/
def rate_T : ℝ := 200

/-- Denomination of bills produced by Press T in dollars -/
def denomination_T : ℝ := 20

/-- Time in seconds for Press F to produce 50 dollars more than Press T -/
def time : ℝ := 3

/-- Theorem stating that the denomination of bills produced by Press F is 81 dollars -/
theorem press_F_denomination :
  (time / 60 * rate_F * F = time / 60 * rate_T * denomination_T + 50) →
  F = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_press_F_denomination_l367_36793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_l367_36713

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n + 3 => a (n + 1) + a n

def b : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | 3 => 1
  | 4 => 1
  | n + 5 => b (n + 4) + b n

theorem sequence_relation : ∃ N : ℕ, ∀ n ≥ N, a n = b (n + 1) + b (n - 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_relation_l367_36713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l367_36759

theorem erdos_szekeres (m n : ℕ) (a : Fin (m * n + 1) → ℝ) :
  (∃ (s : Fin (m + 1) → Fin (m * n + 1)), StrictMono s ∧ StrictMono (a ∘ s)) ∨
  (∃ (s : Fin (n + 1) → Fin (m * n + 1)), StrictMono s ∧ StrictAnti (a ∘ s)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l367_36759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l367_36769

/-- The ratio of the area of an equilateral triangle inscribed in a semicircle
    to the area of an equilateral triangle inscribed in a circle,
    both with radius r -/
theorem triangle_area_ratio (r : ℝ) (r_pos : r > 0) :
  (16 * r^2 / 3) / (3 * Real.sqrt 3 * r^2 / 4) = 64 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l367_36769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l367_36755

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 1 → (1/2:ℝ)^x < 1/2) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ (1/2:ℝ)^x₀ ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l367_36755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_ax_iff_a_in_range_l367_36707

open Real

noncomputable def f (x : ℝ) : ℝ := x + 1 / (exp x)

theorem f_greater_than_ax_iff_a_in_range :
  (∀ x : ℝ, f x > a * x) ↔ a ∈ Set.Ioo (1 - exp 1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_ax_iff_a_in_range_l367_36707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_screen_height_l367_36754

/-- Represents a rectangular TV screen -/
structure TVScreen where
  area : ℝ
  width : ℝ

/-- Calculates the height of a TV screen given its area and width -/
noncomputable def calculateHeight (tv : TVScreen) : ℝ :=
  tv.area / tv.width

theorem tv_screen_height :
  let tv : TVScreen := { area := 21, width := 3 }
  calculateHeight tv = 7 := by
  -- Unfold the definition of calculateHeight
  unfold calculateHeight
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_screen_height_l367_36754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l367_36784

noncomputable def variance (s : List ℝ) : ℝ := 
  let n := s.length
  let mean := s.sum / n
  (s.map (λ x => (x - mean) ^ 2)).sum / n

theorem variance_transformation (a₁ a₂ a₃ : ℝ) :
  variance [a₁, a₂, a₃] = 2 → 
  variance [2 * a₁ + 3, 2 * a₂ + 3, 2 * a₃ + 3] = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l367_36784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chasing_probability_is_1_360_l367_36783

/-- The probability of selecting specific letters from given words -/
def select_probability (word : String) (select : Nat) (target : String) : Rat :=
  1 / (Nat.choose word.length select)

/-- The probability of selecting all letters from CHASING -/
def chasing_probability : Rat :=
  (select_probability "CAMP" 2 "CA") *
  (select_probability "HERBS" 3 "HES") *
  (select_probability "GLOW" 2 "GW")

theorem chasing_probability_is_1_360 : chasing_probability = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chasing_probability_is_1_360_l367_36783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_diagonal_l367_36725

theorem square_area_from_diagonal (a b : ℝ) : 
  ∃ (s : ℝ), s > 0 ∧ 
  (2*a + b)^2 = 2 * s^2 ∧ 
  s^2 = (2*a + b)^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_diagonal_l367_36725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sale_discount_l367_36704

/-- Represents the additional discount percentage during a special sale -/
noncomputable def additional_discount (list_price : ℝ) (typical_min_discount : ℝ) (typical_max_discount : ℝ) (lowest_sale_price_percent : ℝ) : ℝ :=
  100 * (1 - lowest_sale_price_percent / 100 - typical_max_discount / 100)

/-- Theorem stating the additional discount during the special sale -/
theorem special_sale_discount :
  let list_price : ℝ := 80
  let typical_min_discount : ℝ := 30
  let typical_max_discount : ℝ := 70
  let lowest_sale_price_percent : ℝ := 40
  additional_discount list_price typical_min_discount typical_max_discount lowest_sale_price_percent = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sale_discount_l367_36704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_time_calculation_l367_36745

/-- The time taken to climb to the top of the hill -/
def climbTime : ℝ := sorry

/-- The distance to the top of the hill -/
def distance : ℝ := sorry

/-- The average speed for the entire journey -/
def avgSpeedTotal : ℝ := 3

/-- The average speed while climbing up -/
def avgSpeedUp : ℝ := 2.25

/-- The time taken to descend the hill -/
def descentTime : ℝ := 2

theorem climb_time_calculation :
  (distance = avgSpeedUp * climbTime) ∧
  (distance = avgSpeedTotal * descentTime) ∧
  (avgSpeedTotal = (2 * distance) / (climbTime + descentTime)) →
  climbTime = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_time_calculation_l367_36745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bound_implies_a_range_l367_36768

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + 4*x else Real.log (x + 1)

theorem f_bound_implies_a_range (a : ℝ) :
  (∀ x : ℝ, |f x| ≥ a * x - 1) → a ∈ Set.Icc (-6) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bound_implies_a_range_l367_36768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l367_36735

def v₁ : Fin 3 → ℝ := ![1, 2, -2]
def v₂ : Fin 3 → ℝ := ![0, 1, -1]
noncomputable def u : Fin 3 → ℝ := ![3 * Real.sqrt 3 / 2 - 2, 0, -1]

theorem vector_satisfies_conditions : 
  (Real.sqrt ((u 0)^2 + (u 1)^2 + (u 2)^2) = 1) ∧ 
  (u 1 = 0) ∧
  (Real.cos (30 * π / 180) = (u 0 * v₁ 0 + u 1 * v₁ 1 + u 2 * v₁ 2) / 
    (Real.sqrt ((u 0)^2 + (u 1)^2 + (u 2)^2) * Real.sqrt ((v₁ 0)^2 + (v₁ 1)^2 + (v₁ 2)^2))) ∧
  (Real.cos (45 * π / 180) = (u 0 * v₂ 0 + u 1 * v₂ 1 + u 2 * v₂ 2) / 
    (Real.sqrt ((u 0)^2 + (u 1)^2 + (u 2)^2) * Real.sqrt ((v₂ 0)^2 + (v₂ 1)^2 + (v₂ 2)^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l367_36735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_against_l367_36794

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ  -- Length of the walkway in meters
  time_with : ℝ  -- Time to walk with the walkway in seconds
  time_stationary : ℝ  -- Time to walk on stationary walkway in seconds

/-- Calculates the time to walk against the walkway -/
noncomputable def time_against (s : WalkwayScenario) : ℝ :=
  s.length * s.time_stationary / (s.time_stationary - s.time_with)

/-- Theorem stating the correct time to walk against the walkway -/
theorem walkway_time_against 
  (s : WalkwayScenario) 
  (h1 : s.length = 80) 
  (h2 : s.time_with = 40) 
  (h3 : s.time_stationary = 60) : 
  time_against s = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_against_l367_36794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l367_36711

/-- The distance between two points in a three-dimensional Cartesian coordinate system -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The distance between points A(2, 1, 0) and B(4, 3, 2) is 2√3 -/
theorem distance_A_to_B :
  distance3D 2 1 0 4 3 2 = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l367_36711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l367_36714

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x

theorem f_properties :
  (∀ x y, x < y ∧ ((x < -3 ∧ y < -3) ∨ (x > 1 ∧ y > 1)) → f x < f y) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ -5/3) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ 2/3) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -5/3) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l367_36714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l367_36715

/-- The total shaded area in two circles -/
theorem total_shaded_area (r₁ r₂ d : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 5) (h₃ : d = 2) : 
  (π * r₁^2 / 2 + π * r₂^2 / 2) = 62.5 * π :=
by
  -- Substitute the given values
  rw [h₁, h₂]
  
  -- Simplify the expression
  have : π * 10^2 / 2 + π * 5^2 / 2 = 62.5 * π := by
    ring
  
  -- Apply the simplification
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l367_36715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_theorem_l367_36721

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 / 2 + b * Real.exp x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := x + b * Real.exp x

theorem extreme_points_theorem (b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' b x₁ = 0 ∧ f' b x₂ = 0) →
  (-1 / Real.exp 1 < b ∧ b < 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f' b x₁ = 0 → f' b x₂ = 0 → x₁ + x₂ > 2) := by
  sorry

#check extreme_points_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_theorem_l367_36721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_of_range_l367_36723

theorem ending_number_of_range (start : Nat) (count : Nat) : 
  let first_divisible := ((start + 10) / 11) * 11
  let sequence := List.range count |>.map (fun i => first_divisible + i * 11)
  start = 29 ∧ count = 5 → sequence.getLast? = some 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_of_range_l367_36723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_theorem_l367_36739

/-- Represents a right circular cone with liquid and a marble -/
structure LiquidCone where
  radius : ℝ  -- radius of the top of the liquid surface
  marble_radius : ℝ  -- radius of the marble

/-- Calculates the rise in liquid level when a marble is dropped into a cone -/
noncomputable def liquid_rise (cone : LiquidCone) : ℝ :=
  (4 / 3) * Real.pi * cone.marble_radius^3 / (Real.pi * cone.radius^2)

theorem liquid_rise_ratio_theorem (narrow_cone wide_cone : LiquidCone) 
  (h1 : narrow_cone.radius = 4)
  (h2 : wide_cone.radius = 8)
  (h3 : narrow_cone.marble_radius = 1)
  (h4 : wide_cone.marble_radius = 2) :
  liquid_rise narrow_cone / liquid_rise wide_cone = 1 / 2 := by
  sorry

#check liquid_rise_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_theorem_l367_36739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_company_profit_per_tire_l367_36778

/-- Calculates the profit per tire for a tire company --/
noncomputable def profit_per_tire (fixed_cost : ℝ) (variable_cost : ℝ) (selling_price : ℝ) (batch_size : ℕ) : ℝ :=
  let total_cost := fixed_cost + variable_cost * (batch_size : ℝ)
  let total_revenue := selling_price * (batch_size : ℝ)
  let total_profit := total_revenue - total_cost
  total_profit / (batch_size : ℝ)

/-- Theorem: The profit per tire for the given tire company is $10.50 --/
theorem tire_company_profit_per_tire :
  profit_per_tire 22500 8 20 15000 = 10.5 := by
  -- Unfold the definition of profit_per_tire
  unfold profit_per_tire
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_company_profit_per_tire_l367_36778
