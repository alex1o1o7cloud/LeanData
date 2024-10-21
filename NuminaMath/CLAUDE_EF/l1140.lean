import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_increases_averages_l1140_114041

/-- Represents a group of students with their total grade and count --/
structure StudentGroup where
  totalGrade : ℚ
  count : ℕ

/-- Calculates the average grade of a student group --/
def averageGrade (group : StudentGroup) : ℚ :=
  group.totalGrade / group.count

/-- Represents the transfer of students between groups --/
def transferStudents (groupA groupB : StudentGroup) (grade1 grade2 : ℚ) : (StudentGroup × StudentGroup) :=
  let newGroupA : StudentGroup := ⟨groupA.totalGrade - grade1 - grade2, groupA.count - 2⟩
  let newGroupB : StudentGroup := ⟨groupB.totalGrade + grade1 + grade2, groupB.count + 2⟩
  (newGroupA, newGroupB)

/-- The main theorem to prove --/
theorem transfer_increases_averages (groupA groupB : StudentGroup) (grade1 grade2 : ℚ) :
  groupA.count = 10 →
  groupB.count = 10 →
  averageGrade groupA = 442/10 →
  averageGrade groupB = 388/10 →
  grade1 = 41 →
  grade2 = 44 →
  let (newGroupA, newGroupB) := transferStudents groupA groupB grade1 grade2
  averageGrade newGroupA > averageGrade groupA ∧ averageGrade newGroupB > averageGrade groupB :=
by
  sorry

#eval averageGrade ⟨442, 10⟩
#eval averageGrade ⟨388, 10⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_increases_averages_l1140_114041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domains_and_subset_l1140_114050

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.sqrt ((x - 1) / (x + 1))
noncomputable def g (a : ℝ) (x : ℝ) := Real.log ((x - a) * (x - 1))

-- Define the domains A and B
def A : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x > 1 ∨ x < a}

-- State the theorem
theorem domains_and_subset (a : ℝ) (h : a < 1) :
  A = Set.Iic (-1) ∪ Set.Ici 1 ∧
  B a = Set.Ioi 1 ∪ Set.Iic a ∧
  (B a ⊆ A ↔ a ∈ Set.Iic (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domains_and_subset_l1140_114050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1140_114018

theorem solve_exponential_equation :
  ∃ y : ℝ, 81 = 3 * (27 : ℝ) ^ (y - 2) ↔ y = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1140_114018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangle_solutions_l1140_114058

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 2 ∧
  t.b = 4 ∧
  t.A = Real.pi / 6 ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem two_triangle_solutions :
  ∃ (t1 t2 : Triangle), 
    satisfiesConditions t1 ∧ 
    satisfiesConditions t2 ∧ 
    t1 ≠ t2 ∧
    ∀ (t : Triangle), satisfiesConditions t → (t = t1 ∨ t = t2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangle_solutions_l1140_114058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1140_114074

-- Define the constants
noncomputable def a : ℝ := Real.rpow 3 1.2
def b : ℝ := 1  -- 3° in radians is equivalent to 1
noncomputable def c : ℝ := Real.rpow (1/3) (-0.9)

-- State the theorem
theorem relationship_abc : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1140_114074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1140_114059

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Calculates the total volume of the fruit drink -/
noncomputable def total_volume (drink : FruitDrink) : ℝ :=
  drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)

theorem fruit_drink_volume (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.15)
  (h2 : drink.watermelon_percent = 0.60)
  (h3 : drink.grape_ounces = 35) :
  total_volume drink = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1140_114059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C2_and_distance_AB_l1140_114097

open Real

-- Define the curves and points
noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ := (2 * cos α, 2 + 2 * sin α)

noncomputable def point_M (α : ℝ) : ℝ × ℝ := curve_C1 α

noncomputable def point_P (α : ℝ) : ℝ × ℝ := (4 * cos α, 4 + 4 * sin α)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 8 * sin θ

noncomputable def curve_C3 (θ : ℝ) : ℝ := 4 * cos θ

noncomputable def point_A : ℝ := curve_C3 (π / 3)

noncomputable def point_B : ℝ := curve_C2 (π / 3)

-- State the theorem
theorem curve_C2_and_distance_AB :
  (∀ θ, curve_C2 θ = 8 * sin θ) ∧
  abs (point_A - point_B) = 4 * sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C2_and_distance_AB_l1140_114097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_minimum_l1140_114007

open Real

/-- The function f(x) = (a - ln x) / x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - log x) / x

theorem f_properties (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ x ≤ Real.exp 1 ∧ f a x = -1) ↔ (a ≤ -1 ∨ (0 ≤ a ∧ a ≤ Real.exp 1)) := by
  sorry

theorem f_minimum (a : ℝ) :
  ∀ x > 0, f a x ≥ f a (Real.exp (a + 1)) ∧ f a (Real.exp (a + 1)) = -Real.exp (-a - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_minimum_l1140_114007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_96_l1140_114053

-- Define the side length of the square
def square_side : ℝ := 8

-- Define the area of the square
noncomputable def square_area : ℝ := square_side ^ 2

-- Define the overlapping area
noncomputable def overlap_area : ℝ := (3 / 4) * square_area

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := 2 * overlap_area

-- Theorem statement
theorem triangle_area_is_96 : triangle_area = 96 := by
  -- Unfold definitions
  unfold triangle_area overlap_area square_area square_side
  -- Simplify the expression
  simp [pow_two]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_96_l1140_114053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_in_third_quadrant_l1140_114092

theorem cosine_of_angle_in_third_quadrant (B : ℝ) : 
  (B > Real.pi ∧ B < 3 * Real.pi / 2) →  -- Angle B is in the third quadrant
  Real.sin B = 4/5 → 
  Real.cos B = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_in_third_quadrant_l1140_114092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l1140_114029

def grid_width : ℕ := 5
def grid_height : ℕ := 4
def start_point : (ℕ × ℕ) := (0, 0)
def end_point : (ℕ × ℕ) := (5, 4)
def blocked_point : (ℕ × ℕ) := (2, 2)

def valid_path (path : List (ℕ × ℕ)) : Prop :=
  path.length = 9 ∧
  path.head? = some start_point ∧
  path.getLast? = some end_point ∧
  blocked_point ∉ path ∧
  ∀ i, i < path.length - 1 →
    let (x₁, y₁) := path[i]!
    let (x₂, y₂) := path[i+1]!
    ((x₂ = x₁ + 1 ∧ y₂ = y₁) ∨ (x₂ = x₁ ∧ y₂ = y₁ + 1))

def count_valid_paths : ℕ := sorry

theorem valid_paths_count :
  count_valid_paths = 66 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l1140_114029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_ratio_l1140_114045

-- Define the triangle and points
variable (A B C D E P : ℝ × ℝ)

-- Define the ratios
noncomputable def ratio_CD_DB : ℝ := 4/1
noncomputable def ratio_AE_EB : ℝ := 4/3

-- Define the condition that P is on both CE and AD
def P_on_CE_and_AD (A B C D E P : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem intersection_point_ratio 
  (h1 : (C.1 - D.1) / (D.1 - B.1) = ratio_CD_DB) 
  (h2 : (A.1 - E.1) / (E.1 - B.1) = ratio_AE_EB)
  (h3 : P_on_CE_and_AD A B C D E P) :
  (C.1 - P.1) / (P.1 - E.1) = 4/3 := 
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_ratio_l1140_114045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_of_circle_l1140_114046

/-- Given a circle C containing points (8, 0) and (-8, 0), 
    the maximum possible radius of C is 8. -/
theorem max_radius_of_circle (C : Set (ℝ × ℝ)) : 
  ((8 : ℝ), 0) ∈ C → ((-8 : ℝ), 0) ∈ C → 
  ∃ (center : ℝ × ℝ) (r : ℝ), C = {p : ℝ × ℝ | dist p center = r} → r ≤ 8 ∧ 
  ∃ (C' : Set (ℝ × ℝ)) (center' : ℝ × ℝ), 
    ((8 : ℝ), 0) ∈ C' ∧ ((-8 : ℝ), 0) ∈ C' ∧ 
    C' = {p : ℝ × ℝ | dist p center' = 8} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_of_circle_l1140_114046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1140_114099

theorem work_completion_time (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  (1 / x = 1 / 15) →
  (1 / y = 1 / 30) →
  (1 / (x⁻¹ + y⁻¹) = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1140_114099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_divisibility_l1140_114042

/-- Given points in a coordinate system with specific properties, prove statements about the minimum distance and divisibility. -/
theorem min_distance_and_divisibility (a b c : ℕ+) (p : ℝ) (X : ℝ) : 
  let A : ℝ × ℝ := (0, a)
  let O : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (c, 0)
  let B : ℝ × ℝ := (c, b)
  let P : ℝ × ℝ := (p, 0)
  -- P is on line segment OC
  0 ≤ p ∧ p ≤ c →
  -- P minimizes AP + PB
  ∀ q : ℝ, 0 ≤ q ∧ q ≤ c → 
    Real.sqrt (p^2 + a^2) + Real.sqrt ((c - p)^2 + b^2) ≤ 
    Real.sqrt (q^2 + a^2) + Real.sqrt ((c - q)^2 + b^2) →
  -- X is the minimum distance
  X = Real.sqrt (p^2 + a^2) + Real.sqrt ((c - p)^2 + b^2) →
  (
    -- 1. X = √(c^2 + (a + b)^2)
    X = Real.sqrt (c^2 + (a + b)^2) ∧
    -- 2. If a, b, p, X are all positive integers, there exists an integer n ≥ 3 that divides both a and b
    (∃ p' : ℕ+, p = p') → (∃ X' : ℕ+, X = X') → ∃ n : ℕ, n ≥ 3 ∧ n ∣ a ∧ n ∣ b
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_divisibility_l1140_114042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1140_114063

theorem symmetry_condition (θ : ℝ) : 
  (Real.cos θ = -Real.cos (θ + π/6) ∧ Real.sin θ = Real.sin (θ + π/6)) → 
  θ = 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1140_114063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_rate_l1140_114030

/-- Calculates the value of an investment after a simple interest period -/
def invest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the investment conditions, the second certificate's annual interest rate is approximately 12.55% -/
theorem second_certificate_rate (initial_investment : ℝ) (first_rate : ℝ) (second_rate : ℝ) : 
  initial_investment = 20000 →
  first_rate = 0.08 →
  invest (invest initial_investment first_rate (1/4)) second_rate (1/4) = 21040 →
  abs (second_rate - 0.1255) < 0.0001 := by
  sorry

#check second_certificate_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_rate_l1140_114030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l1140_114019

def arithmeticSequence : List ℤ := List.range 10 |>.map (λ i => i - 3)

theorem arithmetic_mean_of_sequence : 
  (arithmeticSequence.sum : ℚ) / arithmeticSequence.length = 3/2 := by
  -- Proof steps would go here
  sorry

#eval arithmeticSequence
#eval arithmeticSequence.sum
#eval arithmeticSequence.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l1140_114019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_c_range_l1140_114096

noncomputable section

def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := (x - e) / (Real.exp x)

theorem f_max_value_and_c_range :
  (∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = Real.exp (-e - 1)) ∧
  (∀ c : ℝ, (∀ x : ℝ, x > 0 → 2 * abs (Real.log x - Real.log 2) ≥ f x + c - 1 / (e^2)) →
    c ≤ (e - 1) / (e^2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_c_range_l1140_114096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_difference_sequences_l1140_114073

-- Definition of a ratio-difference sequence
def is_ratio_difference_sequence (a : ℕ → ℝ) : Prop :=
  ∃ lambda : ℝ, ∀ n : ℕ, (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = lambda

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem ratio_difference_sequences :
  (¬ is_ratio_difference_sequence (λ n => (fibonacci n : ℝ))) ∧
  (∀ a : ℕ → ℝ, is_geometric_sequence a → is_ratio_difference_sequence a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_difference_sequences_l1140_114073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_closed_subset_with_one_from_each_pair_l1140_114006

-- Define the circle S¹
def Circle : Type := Unit

-- Define the antipodal map
noncomputable def antipodal (x : Circle) : Circle := sorry

-- Define the property of containing exactly one point from each pair of diametrically opposite points
def containsOneFromEachPair (A : Set Circle) : Prop :=
  ∀ x : Circle, (x ∈ A ∧ antipodal x ∉ A) ∨ (x ∉ A ∧ antipodal x ∈ A)

-- Theorem statement
theorem no_closed_subset_with_one_from_each_pair 
  [TopologicalSpace Circle] :
  ¬ ∃ A : Set Circle, IsClosed A ∧ containsOneFromEachPair A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_closed_subset_with_one_from_each_pair_l1140_114006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l1140_114036

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sine_transformation :
  ∀ x : ℝ, transformed_function (x + Real.pi / 6) = original_function (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l1140_114036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_polynomial_correct_l1140_114034

/-- Represents a chessboard configuration --/
structure ChessboardConfig where
  rows : Nat
  cols : Nat
  occupied : List (Nat × Nat)

/-- Represents a polynomial with integer coefficients --/
structure RookPolynomial where
  coeffs : List Int

/-- Computes the rook polynomial for a given chessboard configuration --/
def rookPolynomial (config : ChessboardConfig) : RookPolynomial :=
  sorry

/-- The specific chessboard configuration given in the problem --/
def givenConfig : ChessboardConfig :=
  { rows := 5
  , cols := 4
  , occupied := [(0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)]
  }

/-- The expected rook polynomial for the given configuration --/
def expectedPolynomial : RookPolynomial :=
  { coeffs := [1, 10, 25, 24, 6] }

theorem rook_polynomial_correct : rookPolynomial givenConfig = expectedPolynomial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_polynomial_correct_l1140_114034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_sum_l1140_114061

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := sin (4 * x + π / 4)

theorem three_zeros_sum (a : ℝ) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
  x₁ ∈ Set.Icc 0 (9 * π / 16) ∧ x₂ ∈ Set.Icc 0 (9 * π / 16) ∧ x₃ ∈ Set.Icc 0 (9 * π / 16) ∧
  f x₁ + a = 0 ∧ f x₂ + a = 0 ∧ f x₃ + a = 0 ∧
  (∀ x ∈ Set.Icc 0 (9 * π / 16), f x + a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₁ + 2 * x₂ + x₃ = 3 * π / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_sum_l1140_114061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_points_l1140_114084

/-- A point in R^2 -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance between two points in R^2 -/
noncomputable def distance (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The inequality condition for a point -/
def satisfies_inequality (p : Point2D) : Prop :=
  p.x^4 + p.y^4 ≤ p.x^3 + p.y^3

/-- The main theorem -/
theorem exist_four_points :
  ∃ (p₁ p₂ p₃ p₄ : Point2D),
    (satisfies_inequality p₁) ∧
    (satisfies_inequality p₂) ∧
    (satisfies_inequality p₃) ∧
    (satisfies_inequality p₄) ∧
    (distance p₁ p₂ > 1) ∧
    (distance p₁ p₃ > 1) ∧
    (distance p₁ p₄ > 1) ∧
    (distance p₂ p₃ > 1) ∧
    (distance p₂ p₄ > 1) ∧
    (distance p₃ p₄ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_points_l1140_114084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l1140_114011

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- Adding the case for 0 to handle the missing case error
  | 1 => 2
  | n + 1 => (n * sequence_a n + sequence_a n + 1) / n

def inequality_holds (t : ℝ) : Prop :=
  ∀ (n : ℕ) (a : ℝ), n ≥ 1 → a ∈ Set.Icc (-2 : ℝ) 2 →
    sequence_a (n + 1) / (n + 1 : ℝ) < 2 * t^2 + a * t - 1

theorem sequence_inequality_range :
  {t : ℝ | inequality_holds t} = Set.Iic (-2) ∪ Set.Ici 2 :=
by sorry

#check sequence_inequality_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l1140_114011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1140_114047

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase : ∃ (rect_length rect_width perimeter square_side rect_area square_area area_increase : ℝ),
  rect_length = 60 ∧
  rect_width = 20 ∧
  perimeter = 2 * (rect_length + rect_width) ∧
  square_side = perimeter / 4 ∧
  rect_area = rect_length * rect_width ∧
  square_area = square_side * square_side ∧
  area_increase = square_area - rect_area ∧
  area_increase = 400 := by
  -- Provide the values for all variables
  use 60, 20, 160, 40, 1200, 1600, 400
  -- Prove each part of the conjunction
  apply And.intro
  · rfl  -- rect_length = 60
  apply And.intro
  · rfl  -- rect_width = 20
  apply And.intro
  · -- perimeter = 2 * (rect_length + rect_width)
    norm_num
  apply And.intro
  · -- square_side = perimeter / 4
    norm_num
  apply And.intro
  · -- rect_area = rect_length * rect_width
    norm_num
  apply And.intro
  · -- square_area = square_side * square_side
    norm_num
  apply And.intro
  · -- area_increase = square_area - rect_area
    norm_num
  · -- area_increase = 400
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1140_114047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_26_l1140_114085

/-- The count of positive multiples of 6 less than 800 that end with the digit 4 -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => 0 < n ∧ n < 800 ∧ n % 6 = 0 ∧ n % 10 = 4) (Finset.range 800)).card

/-- Theorem stating that the count of such multiples is 26 -/
theorem count_multiples_eq_26 : count_multiples = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_26_l1140_114085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1140_114040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - abs x + 3 * a - 1

noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/2 then 4 * a - 2
  else if a > 1/4 then 3 * a - 1 / (4 * a) - 1
  else 7 * a - 3

theorem f_properties (a : ℝ) :
  (a = 0 → Set.Iic 0 = {x : ℝ | f 0 (2^x) + 2 ≥ 0}) ∧
  (a < 0 → ∀ x, f a x ≤ 3 * a - 1) ∧
  (a > 0 → ∀ x ∈ Set.Icc 1 2, f a x ≥ g a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1140_114040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_rectangle_l1140_114008

-- Define the curves
def hyperbola (x y : ℝ) : Prop := x * y = 20
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ hyperbola x y ∧ circle_eq x y}

-- Define a quadrilateral formed by these points
def quadrilateral : Set (ℝ × ℝ) :=
  {p | p ∈ intersection_points}

-- Define what it means for a set of points to form a rectangle
def IsRectangle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ, s = {a, b, c, d} ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2 ∧
    (a.1 - d.1)^2 + (a.2 - d.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    ((a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0)

-- Theorem statement
theorem intersection_forms_rectangle : 
  IsRectangle quadrilateral :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_rectangle_l1140_114008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_range_l1140_114017

theorem quadratic_root_range (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x ≤ 3 ∧ a * x^2 + x + 3 * a + 1 = 0) →
  (a ∈ Set.Icc (- 1 / 2) (- 1 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_range_l1140_114017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1140_114052

variable (R : ℝ)
variable (d : ℝ)

-- Define the area of triangle ABC as a function of d
noncomputable def triangleArea (R d : ℝ) : ℝ := d * Real.sqrt (R^2 - d^2)

-- State the theorem
theorem max_triangle_area (R : ℝ) (h : R > 0) :
  ∃ d : ℝ, d > 0 ∧ d < R ∧
  (∀ x : ℝ, 0 < x ∧ x < R → triangleArea R d ≥ triangleArea R x) ∧
  d = (Real.sqrt 2 / 2) * R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1140_114052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_seats_calculation_l1140_114005

theorem theater_seats_calculation (seats : ℕ) : 
  (seats > 0) →
  (0.65 * (seats : ℝ) + 0.50 * (seats : ℝ) - 28 = 0.57 * (2 * (seats : ℝ) - 40)) →
  seats = 520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_seats_calculation_l1140_114005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_standard_equation_l1140_114056

-- Define the circle C
noncomputable def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = (a - 1)^2}

-- Define the line l
noncomputable def Line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y - 1 = 0}

-- Define the chord length
noncomputable def ChordLength (a : ℝ) := 2 * Real.sqrt 3

-- Theorem statement
theorem circle_equation (a : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ Circle a → (x - 1)^2 + y^2 = (a - 1)^2) ∧
  (∃ (x y : ℝ), (x, y) ∈ Circle a ∧ (x, y) ∈ Line) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ Circle a ∧ (x₁, y₁) ∈ Line ∧
                        (x₂, y₂) ∈ Circle a ∧ (x₂, y₂) ∈ Line ∧
                        (x₂ - x₁)^2 + (y₂ - y₁)^2 = (ChordLength a)^2) →
  a = 3 ∨ a = -1 :=
by sorry

-- Corollary for the standard equation of the circle
theorem circle_standard_equation (x y : ℝ) :
  (x - 3)^2 + y^2 = 4 ∨ (x + 1)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_standard_equation_l1140_114056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pentagon_side_ratio_l1140_114089

/-- The ratio of the side length of a square to the side length of a regular pentagon with equal areas -/
noncomputable def square_to_pentagon_ratio : ℝ := Real.sqrt (Real.sqrt (5 * Real.sqrt 5) / 2)

/-- The area of a square with side length s -/
noncomputable def square_area (s : ℝ) : ℝ := s^2

/-- The area of a regular pentagon with side length s -/
noncomputable def pentagon_area (s : ℝ) : ℝ := (s^2 * Real.sqrt (5 * Real.sqrt 5)) / 2

/-- Theorem stating the ratio of side lengths of a square and pentagon with equal areas -/
theorem square_pentagon_side_ratio :
  ∀ (s_square s_pentagon : ℝ), s_square > 0 → s_pentagon > 0 →
  square_area s_square = pentagon_area s_pentagon →
  s_square / s_pentagon = square_to_pentagon_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pentagon_side_ratio_l1140_114089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1140_114012

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ max) ∧
    (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = max) ∧
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → min ≤ f x) ∧
    (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = min) ∧
    max = 2 ∧ min = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1140_114012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1140_114076

theorem trigonometric_problem (α β : ℝ)
  (h1 : Real.tan α = 1/3)
  (h2 : Real.cos β = Real.sqrt 5 / 5)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 3*π/2 < β ∧ β < 2*π) :
  (Real.tan (2*α) = 3/4) ∧ (α + β = 7*π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1140_114076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_on_vector_l1140_114055

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Positive projection of one vector onto another -/
def positive_projection (u v : V) : ℝ := ‖u‖ * (inner u v / (‖u‖ * ‖v‖))

/-- Theorem: Projection of sum of two vectors onto one of them -/
theorem projection_sum_on_vector 
  (a b : V) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : inner a b = ‖a‖ * ‖b‖ * Real.cos (π / 3)) : 
  positive_projection (a + b) a = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sum_on_vector_l1140_114055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l1140_114060

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let area : ℝ := (Real.sqrt 3 / 4) * side_length ^ 2
  let perimeter : ℝ := 3 * side_length
  area / perimeter = (5 * Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l1140_114060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_sum_l1140_114013

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ
  F : ℝ × ℝ

/-- Point B on the parabola -/
noncomputable def B : ℝ × ℝ := sorry

/-- Point M is the midpoint of BV -/
noncomputable def M (p : Parabola) : ℝ × ℝ := ((p.V.1 + B.1) / 2, (p.V.2 + B.2) / 2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_vertex_sum (p : Parabola) :
  distance B p.F = 26 ∧ distance B p.V = 25 →
  (∃ d₁ d₂ : ℝ, d₁ ≠ d₂ ∧ 
    (distance p.F p.V = d₁ ∨ distance p.F p.V = d₂) ∧
    d₁ + d₂ = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_sum_l1140_114013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_hundredth_3_1415_l1140_114066

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToNearestHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original : ℝ := 3.1415

theorem round_to_nearest_hundredth_3_1415 :
  roundToNearestHundredth original = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_hundredth_3_1415_l1140_114066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhett_rent_expense_l1140_114025

/-- Represents Rhett's financial situation and rent payments --/
structure RhettFinances where
  monthly_salary : ℚ
  tax_rate : ℚ
  late_payments : ℕ
  late_rent_fraction : ℚ

/-- Calculates the monthly rent expense given Rhett's financial situation --/
noncomputable def calculate_rent_expense (finances : RhettFinances) : ℚ :=
  let after_tax_salary := finances.monthly_salary * (1 - finances.tax_rate)
  let total_late_rent := after_tax_salary * finances.late_rent_fraction
  total_late_rent / finances.late_payments

/-- Theorem stating that Rhett's monthly rent expense is $1350 --/
theorem rhett_rent_expense :
  let finances : RhettFinances := {
    monthly_salary := 5000,
    tax_rate := 1/10,
    late_payments := 2,
    late_rent_fraction := 3/5
  }
  calculate_rent_expense finances = 1350 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhett_rent_expense_l1140_114025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_nonnegative_l1140_114079

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the tangent line parameters
variable (x₀ : ℝ)
variable (k b : ℝ)

-- State the theorem
theorem tangent_line_sum_nonnegative 
  (h₁ : x₀ > 0) 
  (h₂ : k = 1 / x₀) 
  (h₃ : b = f x₀ - k * x₀) :
  k + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_nonnegative_l1140_114079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_limit_l1140_114049

/-- The interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := 180 * (n - 2 : ℝ) / n

/-- The limit of the interior angle of a regular polygon as the number of sides approaches infinity is 180° -/
theorem interior_angle_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |interior_angle n - 180| < ε :=
by
  sorry

#check interior_angle_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_limit_l1140_114049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_form_l1140_114051

/-- 
Theorem: The standard form of a hyperbola with real axis length 4√5 and foci at (±5,0) is x²/20 - y²/5 = 1.
-/
theorem hyperbola_standard_form 
  (real_axis_length : ℝ) 
  (foci_distance : ℝ) 
  (h1 : real_axis_length = 4 * Real.sqrt 5) 
  (h2 : foci_distance = 5) : 
  ∃ (f : ℝ → ℝ → Prop), 
    f = (λ x y ↦ x^2 / 20 - y^2 / 5 = 1) ∧ 
    (∀ x y, f x y ↔ 
      (x^2 / ((real_axis_length/2)^2) - 
       y^2 / (foci_distance^2 - (real_axis_length/2)^2) = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_form_l1140_114051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_is_correct_l1140_114062

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

/-- The midpoint of the line segment -/
def problem_midpoint : ℝ × ℝ := (2, 1)

/-- The line l -/
def line_l (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0

/-- Theorem stating that the given line is correct -/
theorem line_equation_is_correct :
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧
    problem_midpoint = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (∀ (x y : ℝ), line_l x y ↔ ∃ t : ℝ, (x, y) = (1-t) • A + t • B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_is_correct_l1140_114062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_for_unit_area_triangles_l1140_114003

/-- The minimum width of an infinite strip that can accommodate any triangle of area 1 -/
noncomputable def min_strip_width : ℝ := (3 : ℝ) ^ (1/4)

/-- A triangle with area 1 -/
structure UnitAreaTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area_eq_one : a * b * Real.sin c / 2 = 1

/-- An infinite strip with a given width -/
structure InfiniteStrip where
  width : ℝ

/-- A predicate stating that a triangle fits within a strip -/
def fits_in_strip (t : UnitAreaTriangle) (s : InfiniteStrip) : Prop :=
  ∃ (h : ℝ), h ≤ s.width ∧ t.a * t.b * Real.sin t.c / (2 * h) = 1

/-- Theorem stating the minimum width condition for accommodating all unit area triangles -/
theorem min_width_for_unit_area_triangles :
  ∀ (s : InfiniteStrip),
    (∀ (t : UnitAreaTriangle), fits_in_strip t s) ↔ s.width ≥ min_strip_width :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_for_unit_area_triangles_l1140_114003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l1140_114033

theorem subset_existence (X : Finset ℤ) 
  (hX_size : X.card = 10000)
  (hX_not_multiple : ∀ x ∈ X, ¬(47 ∣ x)) :
  ∃ Y : Finset ℤ, Y ⊆ X ∧ Y.card = 2007 ∧
    ∀ a b c d e, a ∈ Y → b ∈ Y → c ∈ Y → d ∈ Y → e ∈ Y → ¬(47 ∣ (a - b + c - d + e)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l1140_114033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l1140_114070

theorem triangle_trig_identities (A : ℝ) (h : Real.sin A + Real.cos A = 7/13) :
  Real.tan A = -12/5 ∧ 2 * Real.sin A * Real.cos A - (Real.cos A)^2 = -145/169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l1140_114070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_square_perimeter_l1140_114002

/-- Given a square with perimeter 144 units divided into 9 congruent smaller squares,
    the perimeter of one of the smaller squares is 48 units. -/
theorem smaller_square_perimeter (large_square : ℝ → ℝ → Prop)
                                  (small_square : ℝ → ℝ → Prop)
                                  (h1 : ∀ x y, large_square x y ↔ 0 ≤ x ∧ x ≤ 144 / 4 ∧ 0 ≤ y ∧ y ≤ 144 / 4)
                                  (h2 : ∀ x y, small_square x y ↔ 0 ≤ x ∧ x ≤ 144 / 12 ∧ 0 ≤ y ∧ y ≤ 144 / 12)
                                  (h3 : ∀ x y, large_square x y → ∃ i j : ℕ, i < 3 ∧ j < 3 ∧
                                                                small_square (x - i * 144 / 12) (y - j * 144 / 12)) :
  ∀ x y, small_square x y → x * 4 + y * 4 = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_square_perimeter_l1140_114002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_M_l1140_114023

def M : Finset ℕ := {2, 4, 6}

theorem proper_subsets_of_M : Finset.card (Finset.powerset M \ {M}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_M_l1140_114023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_b_power_n_l1140_114071

theorem max_factors_b_power_n (b n : ℕ) (hb : b ≤ 20) (hn : n ≤ 20) :
  (∃ (k : ℕ), k ≤ 20 ∧ Nat.card (Nat.divisors (k^n)) = 861) ∧
  (∀ (m : ℕ), m ≤ 20 → Nat.card (Nat.divisors (m^n)) ≤ 861) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_b_power_n_l1140_114071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_calculation_l1140_114044

/-- Given a cone with a lateral area of 50π cm², if the angle between the slant height
    and the base is 60°, the radius of the base is 5 cm. -/
theorem cone_radius_calculation (lateral_area : ℝ) (angle : ℝ) :
  lateral_area = 50 * Real.pi ∧ angle = Real.pi / 3 →
  ∃ (radius : ℝ), radius = 5 ∧ lateral_area = Real.pi * radius * (2 * radius) :=
by
  intro h
  use 5
  constructor
  · rfl
  · rw [h.left]
    ring

#check cone_radius_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_calculation_l1140_114044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_values_and_derivative_l1140_114032

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 * Real.exp (x - 1) + a * x^3 + b * x^2

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := x * (x + 2) * (Real.exp (x - 1) - 1)

theorem extreme_points_imply_values_and_derivative :
  ∀ a b : ℝ,
  (∀ x : ℝ, x = -2 ∨ x = 1 → (deriv (f a b)) x = 0) →
  (a = -1/3 ∧ b = -1) ∧
  (∀ x : ℝ, (deriv (f a b)) x = f' a b x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_values_and_derivative_l1140_114032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_participation_l1140_114031

universe u

structure City where
  Person : Type u
  residents : Finset Person
  knows : Person → Finset Person
  knows_fraction : ∀ p ∈ residents, (knows p).card ≥ (residents.card : ℚ) * (30 : ℚ) / (100 : ℚ)

def will_vote {c : City} (voters candidates : Finset c.Person) : Prop :=
  ∀ v ∈ voters, ∃ cand ∈ candidates, cand ∈ c.knows v

theorem election_participation (c : City) :
  ∃ candidates : Finset c.Person, candidates.card = 2 ∧
    ∃ voters : Finset c.Person, will_vote voters candidates ∧
      voters.card ≥ (c.residents.card : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_participation_l1140_114031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_20_seconds_l1140_114072

-- Define the given parameters
noncomputable def platform_length : ℝ := 150.012
noncomputable def platform_pass_time : ℝ := 30
noncomputable def train_speed_kmh : ℝ := 54

-- Define the function to calculate the time to pass a stationary point
noncomputable def time_to_pass_point (platform_length : ℝ) (platform_pass_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let train_length : ℝ := train_speed_ms * platform_pass_time - platform_length
  train_length / train_speed_ms

-- Theorem statement
theorem train_pass_time_approx_20_seconds :
  ∃ ε > 0, |time_to_pass_point platform_length platform_pass_time train_speed_kmh - 20| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_20_seconds_l1140_114072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_theorem_l1140_114004

/-- Represents a cell on the chessboard -/
structure Cell where
  row : Fin 8
  col : Fin 8
  number : Fin 32

/-- Represents the chessboard configuration -/
def Board := List Cell

/-- Predicate to check if a board is valid -/
def is_valid_board (board : Board) : Prop :=
  board.length = 64 ∧
  ∀ n : Fin 32, (board.filter (λ c ↦ c.number = n)).length = 2

/-- Predicate to check if a selection of cells is valid -/
def is_valid_selection (selection : List Cell) : Prop :=
  selection.length = 32 ∧
  selection.Nodup ∧
  (∀ r : Fin 8, ∃ c ∈ selection, c.row = r) ∧
  (∀ c : Fin 8, ∃ cell ∈ selection, cell.col = c)

/-- Theorem stating that for any valid board, there exists a valid selection -/
theorem chessboard_selection_theorem (board : Board) 
  (h : is_valid_board board) :
  ∃ selection : List Cell, selection ⊆ board ∧ is_valid_selection selection := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_theorem_l1140_114004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_paper_area_l1140_114015

/-- Given a square paper with perimeter 32 cm, proves that the area of the remaining paper after removing 1/4 is 48 cm² -/
theorem remaining_paper_area (perimeter : ℝ) (h1 : perimeter = 32) : 
  (3 / 4) * ((perimeter / 4) * (perimeter / 4)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_paper_area_l1140_114015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_burritos_seven_empanadas_cost_l1140_114048

/-- The price of a burrito in dollars -/
def burrito_price : ℝ := sorry

/-- The price of an empanada in dollars -/
def empanada_price : ℝ := sorry

/-- The condition that four burritos and five empanadas cost $4.00 -/
axiom four_burritos_five_empanadas : 4 * burrito_price + 5 * empanada_price = 4

/-- The condition that six burritos and three empanadas cost $4.50 -/
axiom six_burritos_three_empanadas : 6 * burrito_price + 3 * empanada_price = 4.5

/-- The theorem stating that five burritos and seven empanadas cost $5.25 -/
theorem five_burritos_seven_empanadas_cost : 
  5 * burrito_price + 7 * empanada_price = 5.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_burritos_seven_empanadas_cost_l1140_114048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_characterization_non_translatable_proof_l1140_114082

/-- A parabola is defined by its coefficients a, b, and c in the form y = ax² + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The original parabola y = -1/2x² + x + 1 -/
def original_parabola : Parabola :=
  { a := -1/2, b := 1, c := 1 }

/-- A parabola p can be obtained by translating the original parabola -/
def is_translation (p : Parabola) : Prop :=
  p.a = original_parabola.a

theorem parabola_translation_characterization (p : Parabola) :
  ¬(is_translation p) ↔ p.a ≠ original_parabola.a := by
  sorry

/-- The parabola that cannot be obtained by translation -/
def non_translatable_parabola : Parabola :=
  { a := -1, b := 1, c := 1 }

theorem non_translatable_proof :
  ¬(is_translation non_translatable_parabola) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_characterization_non_translatable_proof_l1140_114082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_15_to_45_l1140_114095

def odd_sum (start : ℕ) (end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem odd_sum_15_to_45 : odd_sum 15 45 = 480 := by
  -- Proof goes here
  sorry

#eval odd_sum 15 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_15_to_45_l1140_114095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1140_114001

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem stating that a 100-meter long train moving at 126 km/hr takes approximately 2.857 seconds to cross an electric pole -/
theorem train_crossing_time_approx :
  ∃ ε > 0, |train_crossing_time 100 126 - 2.857| < ε :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1140_114001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_minus_c_equals_neg_log_288_l1140_114067

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.log 5000 / Real.log n)

noncomputable def b : ℝ := a 2 + a 3 + a 5

noncomputable def c : ℝ := a 8 + a 9 + a 10 + a 12

theorem b_minus_c_equals_neg_log_288 :
  b - c = -(Real.log 288 / Real.log 5000) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_minus_c_equals_neg_log_288_l1140_114067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_three_smallest_solutions_l1140_114083

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def satisfies_equation (x : ℝ) : Prop :=
  x - (floor x : ℝ) = 2 / ((floor x : ℝ) + 1)

-- Define a function to get the nth smallest positive solution
noncomputable def nth_smallest_solution (n : ℕ) : ℝ :=
  sorry

-- State the theorem
theorem product_of_three_smallest_solutions :
  (nth_smallest_solution 1) * (nth_smallest_solution 2) * (nth_smallest_solution 3) = 41 + 1/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_three_smallest_solutions_l1140_114083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_duck_percentage_is_15_percent_l1140_114010

/-- The percentage of green ducks in both ponds combined -/
def green_duck_percentage (small_pond_total : ℕ) (large_pond_total : ℕ)
  (small_pond_green_percent : ℚ) (large_pond_green_percent : ℚ) : ℚ :=
  let small_pond_green := (small_pond_green_percent * small_pond_total) / 100
  let large_pond_green := (large_pond_green_percent * large_pond_total) / 100
  let total_green := small_pond_green + large_pond_green
  let total_ducks := small_pond_total + large_pond_total
  (total_green / total_ducks) * 100

/-- Theorem stating that the percentage of green ducks is 15% given the problem conditions -/
theorem green_duck_percentage_is_15_percent :
  green_duck_percentage 30 50 20 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_duck_percentage_is_15_percent_l1140_114010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_volume_in_cone_l1140_114087

/-- The volume of a hemisphere inscribed in a cone -/
theorem hemisphere_volume_in_cone (l α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  let volume := (1 / 12) * Real.pi * l^3 * (Real.sin (2 * α))^3
  ∃ (hemisphere_volume : Real),
    hemisphere_volume = volume ∧
    hemisphere_volume > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_volume_in_cone_l1140_114087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l1140_114068

theorem cos_sin_sum (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = Real.pi) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = -(Real.pi^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l1140_114068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1140_114064

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x + a / Real.exp x|

theorem f_monotone_iff_a_in_range (a : ℝ) :
  (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f a x ≤ f a y) ↔ -1 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1140_114064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1140_114000

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h : seq.a 4 ^ 2 + seq.a 5 ^ 2 = seq.a 6 ^ 2 + seq.a 7 ^ 2) : 
  sum_n seq 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1140_114000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_complement_B_l1140_114021

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x + 2) * (x - 1) > 0}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x < 0}

-- Theorem to prove
theorem union_A_complement_B :
  A ∪ (U \ B) = {x : ℝ | x < -1 ∨ x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_complement_B_l1140_114021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l1140_114022

theorem complex_fraction_calculation : ((((3:ℚ)+2)⁻¹-1)⁻¹-1)⁻¹ - 1 = -13/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l1140_114022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1140_114037

noncomputable def f (x : ℝ) := (Real.cos x)^2 - (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (max_val min_val : ℝ),
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ max_val) ∧
    (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = max_val) ∧
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 → min_val ≤ f x) ∧
    (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = min_val) ∧
    max_val = 2 ∧ min_val = -1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1140_114037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1140_114075

-- Define the piecewise function R(x)
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 80 then x^2 + 200*x
  else if x ≥ 80 then (301*x^2 - 2750*x + 10000) / x
  else 0

-- Define the profit function W(x)
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 80 then -x^2 + 100*x - 1000
  else if 80 ≤ x ∧ x ≤ 150 then -x - 10000/x + 1750
  else 0

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x ∈ Set.Icc 0 150, W x ≤ W 100 ∧ W 100 = 1550 := by sorry

-- Additional conditions as lemmas
lemma fixed_cost : ℝ := 10

lemma market_price : ℝ := 300

lemma max_production : ℝ := 150

lemma R_at_10 : R 10 = 2100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1140_114075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_difference_in_H_l1140_114081

/-- The set H of floor values of i√2 for positive integers i -/
def H : Set ℤ := {x | ∃ i : ℤ, i > 0 ∧ x = ⌊i * Real.sqrt 2⌋}

/-- Theorem statement -/
theorem exists_difference_in_H :
  ∃ C : ℝ, ∀ n : ℕ, n > 0 → ∀ A : Finset ℕ,
    (A ⊆ Finset.range n) →
    (A.card : ℝ) ≥ C * Real.sqrt n →
    ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ ((a : ℤ) - b) ∈ H :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_difference_in_H_l1140_114081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_values_l1140_114088

noncomputable def triangle_ABC (a b : ℝ) (A : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 6 ∧ A = 45 * Real.pi / 180

theorem angle_C_values (a b : ℝ) (A : ℝ) (h : triangle_ABC a b A) :
  ∃ C, (C = 15 * Real.pi / 180 ∨ C = 75 * Real.pi / 180) ∧
    0 < C ∧ C < Real.pi ∧
    ∃ B, 0 < B ∧ B < Real.pi ∧ A + B + C = Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_values_l1140_114088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l1140_114094

theorem odd_prime_divisibility (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (∃ n : ℤ, (p : ℤ) ∣ n * (n + 1) * (n + 2) * (n + 3) + 1) ↔
  (∃ m : ℤ, (p : ℤ) ∣ m^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l1140_114094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_study_properties_l1140_114027

structure School where
  total_students : ℕ
  surveyed_students : ℕ

structure StatisticalStudy where
  school : School
  population : Set ℝ
  sample : Set ℝ
  individual : Type
  sample_size : ℕ

def weight_study (s : School) : StatisticalStudy where
  school := s
  population := {x : ℝ | ∃ _student : ℕ, true}
  sample := {x : ℝ | ∃ _student : ℕ, true}
  individual := ℕ
  sample_size := s.surveyed_students

theorem weight_study_properties (s : School) 
  (h1 : s.total_students = 4000) 
  (h2 : s.surveyed_students = 400) : 
  let study := weight_study s
  (study.population = {x : ℝ | ∃ _student : ℕ, true}) ∧
  (study.sample = {x : ℝ | ∃ _student : ℕ, true}) ∧
  (study.sample_size = 400) ∧
  (study.individual = ℕ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_study_properties_l1140_114027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_can_buy_ticket_l1140_114054

/-- Represents the denominations of coins available -/
inductive Coin where
  | Three
  | Five
deriving BEq, Repr

/-- Represents a person (visitor or cashier) with their money -/
structure Person where
  coins : List Coin
  money : Nat
  inv : money = (coins.filter (· == Coin.Three)).length * 3 + 
               (coins.filter (· == Coin.Five)).length * 5

/-- The ticket price -/
def ticketPrice : Nat := 4

/-- The number of visitors in the queue -/
def queueLength : Nat := 200

/-- Theorem stating that all visitors can buy a ticket -/
theorem all_can_buy_ticket 
  (visitors : Vector Person queueLength)
  (cashier : Person)
  (h1 : ∀ p ∈ visitors.toList, p.money = 22)
  (h2 : cashier.money = 22) :
  ∃ (finalCashier : Person), 
    finalCashier.money = cashier.money + queueLength * ticketPrice := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_can_buy_ticket_l1140_114054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_sequence_uniqueness_l1140_114028

/-- Represents a triple of positive coprime integers -/
structure CoprimeTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  coprime : Nat.Coprime a.val (Nat.gcd b.val c.val)

/-- Counts the number of elements in {a, b, c} that divide n -/
def s (t : CoprimeTriple) (n : ℕ) : ℕ :=
  (if n % t.a = 0 then 1 else 0) +
  (if n % t.b = 0 then 1 else 0) +
  (if n % t.c = 0 then 1 else 0)

/-- The sequence of all positive integers divisible by some element of {a, b, c} -/
noncomputable def k (t : CoprimeTriple) : ℕ → ℕ :=
  sorry

/-- The characteristic sequence of a CoprimeTriple -/
noncomputable def characteristicSequence (t : CoprimeTriple) : ℕ → ℕ :=
  fun n => s t (k t n)

/-- Two characteristic sequences are equal -/
def equalCharacteristicSequences (t1 t2 : CoprimeTriple) : Prop :=
  ∀ n, characteristicSequence t1 n = characteristicSequence t2 n

theorem characteristic_sequence_uniqueness (t1 t2 : CoprimeTriple) :
  equalCharacteristicSequences t1 t2 → t1 = t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_sequence_uniqueness_l1140_114028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_product_l1140_114093

/-- Area function for a triangle given its three sides using Heron's formula -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle with sides a, b, c, circumradius R, and inradius r, 
    the product of R and r equals the product of sides divided by twice the perimeter -/
theorem triangle_radii_product (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circum : R > 0)
  (h_in : r > 0)
  (h_R : R = (a * b * c) / (4 * area a b c))
  (h_r : r = (2 * area a b c) / (a + b + c)) :
  R * r = (a * b * c) / (2 * (a + b + c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radii_product_l1140_114093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_angle_theorem_l1140_114039

-- Define the circle O
variable (O : EuclideanSpace ℝ (Fin 2))

-- Define points Q, C, and D on the plane
variable (Q C D : EuclideanSpace ℝ (Fin 2))

-- Define the property that QCD is a triangle formed by tangents to circle O
def is_tangent_triangle (O Q C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the measure of an angle
noncomputable def angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem tangent_triangle_angle_theorem 
  (h_tangent : is_tangent_triangle O Q C D) 
  (h_angle_CQD : angle_measure C Q D = 50) : 
  angle_measure C O D = 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_angle_theorem_l1140_114039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_always_three_l1140_114077

noncomputable def star (x y : ℝ) : ℝ := (3 * x / y) * (y / x)

theorem star_always_three (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  star a (star b c) = 3 ∧ star (star a (star b c)) d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_always_three_l1140_114077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_not_divisible_l1140_114098

theorem consecutive_sum_not_divisible (K : ℕ) (h : Even K) :
  ∃ (perm : Fin (K - 1) → Fin (K - 1)), Function.Bijective perm ∧
    ∀ (i j : Fin (K - 1)), i ≤ j →
      ¬(K ∣ (Finset.range (j.val - i.val + 1)).sum (λ k => (perm ⟨i.val + k, sorry⟩).val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_not_divisible_l1140_114098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1140_114078

/-- A graph without cycles of length 3 -/
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop
  no_triangles : ∀ a b c, edges a b → edges b c → edges c a → False

/-- An assignment of natural numbers to vertices satisfying the conditions -/
def ValidAssignment (G : Graph) (n : ℕ) :=
  ∃ f : G.vertices → ℕ,
    (∀ v w, v ≠ w → f v ≠ f w) ∧
    (∀ v w, ¬G.edges v w → (f v + f w).gcd n = 1) ∧
    (∀ v w, G.edges v w → ∃ d > 1, d ∣ (f v + f w) ∧ d ∣ n)

/-- The main theorem stating that 35 is the smallest valid n -/
theorem smallest_valid_n :
  (∃ G : Graph, ValidAssignment G 35) ∧
  (∀ m < 35, ∀ G : Graph, ¬ValidAssignment G m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1140_114078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangent_triangle_is_acute_l1140_114009

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  O : ℝ × ℝ  -- Center of the inscribed circle
  r : ℝ      -- Radius of the inscribed circle
  A₁ : ℝ × ℝ  -- Point of tangency on BC
  B₁ : ℝ × ℝ  -- Point of tangency on CA
  C₁ : ℝ × ℝ  -- Point of tangency on AB

/-- The angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Check if an angle is acute -/
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem inscribed_circle_tangent_triangle_is_acute (T : InscribedCircleTriangle) :
  is_acute (angle (T.B₁ - T.A₁) (T.C₁ - T.A₁)) ∧
  is_acute (angle (T.A₁ - T.B₁) (T.C₁ - T.B₁)) ∧
  is_acute (angle (T.A₁ - T.C₁) (T.B₁ - T.C₁)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangent_triangle_is_acute_l1140_114009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l1140_114026

theorem undefined_fraction (a : ℝ) : 
  (a + 3) / (a^2 - 9) = 0 ↔ a = -3 ∨ a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l1140_114026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertex_diameter_circles_l1140_114057

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → (ℝ × ℝ)

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The plane containing the square and circles -/
def Plane := ℝ × ℝ

/-- Function to create a circle from two points (diameter endpoints) -/
noncomputable def circleFromDiameter (p q : Plane) : Circle := sorry

/-- Function to check if two circles are distinct -/
def areDistinctCircles (c1 c2 : Circle) : Prop := sorry

/-- Main theorem: There are exactly 3 distinct circles whose diameters are defined by vertices of the square -/
theorem square_vertex_diameter_circles (S : Square) : 
  ∃ (C : Finset Circle), 
    (∀ c ∈ C, ∃ (i j : Fin 4), i ≠ j ∧ c = circleFromDiameter (S.vertices i) (S.vertices j)) ∧
    (∀ (i j : Fin 4), i ≠ j → ∃ c ∈ C, c = circleFromDiameter (S.vertices i) (S.vertices j)) ∧
    (∀ c1 c2 : Circle, c1 ∈ C → c2 ∈ C → c1 ≠ c2 → areDistinctCircles c1 c2) ∧
    C.card = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertex_diameter_circles_l1140_114057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1140_114038

noncomputable section

-- Define the point C
def C : ℝ × ℝ := (2, -1)

-- Define the slope of the given line
def m₁ : ℝ := -1

-- Define the slope of the perpendicular line
def m₂ : ℝ := -1 / m₁

-- Statement to prove
theorem perpendicular_line_equation :
  let line_eq := fun (x y : ℝ) => x - y - 3
  (∀ x y, line_eq x y = 0 ↔ 
    (y - C.2 = m₂ * (x - C.1)) ∧ 
    m₁ * m₂ = -1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1140_114038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suspended_cube_sum_l1140_114080

/-- Represents a cube suspended above a plane -/
structure SuspendedCube where
  sideLength : ℝ
  vertexHeights : Fin 3 → ℝ
  distanceToPlane : ℝ

/-- The condition for a valid suspended cube configuration -/
def isValidSuspendedCube (c : SuspendedCube) : Prop :=
  c.sideLength = 8 ∧
  c.vertexHeights 0 = 8 ∧
  c.vertexHeights 1 = 9 ∧
  c.vertexHeights 2 = 10

/-- The distance from the closest vertex to the plane can be expressed as (p - √q) / u -/
noncomputable def distanceExpression (p q u : ℕ) : ℝ :=
  (p - Real.sqrt (q : ℝ)) / u

theorem suspended_cube_sum (c : SuspendedCube) (p q u : ℕ) :
  isValidSuspendedCube c →
  c.distanceToPlane = distanceExpression p q u →
  p + q + u = 312 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suspended_cube_sum_l1140_114080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1140_114043

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 7)
  (h2 : (x : ℝ) / (y : ℝ) = 86.1) : 
  y = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1140_114043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_point_on_line_with_distance_min_distance_line_segment_l1140_114090

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ k b : ℝ) : ℝ :=
  |k * x₀ - y₀ + b| / Real.sqrt (1 + k^2)

-- Statement 1
theorem distance_to_line (x₀ y₀ k b : ℝ) :
  distance_point_to_line 1 2 2 (-1) = Real.sqrt 5 / 5 := by sorry

-- Statement 2
theorem point_on_line_with_distance (x₀ : ℝ) :
  (x₀ = 13 ∨ x₀ = -7) ↔
  distance_point_to_line x₀ (x₀ + 2) 2 (-1) = 2 * Real.sqrt 5 := by sorry

-- Statement 3
theorem min_distance_line_segment (k : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2,
    distance_point_to_line x (k * x + 4) 1 2 ≥ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2,
    distance_point_to_line x (k * x + 4) 1 2 = Real.sqrt 2) →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_point_on_line_with_distance_min_distance_line_segment_l1140_114090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_f_gt_sin_g_cos_cos_gt_sin_sin_l1140_114014

open Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the condition for f and g
axiom fg_bound (x : ℝ) : -Real.pi/2 < f x - g x ∧ f x + g x < Real.pi/2

-- State the theorems to be proved
theorem cos_f_gt_sin_g (x : ℝ) : Real.cos (f x) > Real.sin (g x) := by sorry

theorem cos_cos_gt_sin_sin (x : ℝ) : Real.cos (Real.cos x) > Real.sin (Real.sin x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_f_gt_sin_g_cos_cos_gt_sin_sin_l1140_114014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_min_value_l1140_114069

theorem cos_2x_plus_2sin_x_min_value :
  (∀ x : ℝ, Real.cos (2 * x) + 2 * Real.sin x ≥ -3) ∧
  (∃ x : ℝ, Real.cos (2 * x) + 2 * Real.sin x = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_min_value_l1140_114069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_y_coordinate_l1140_114086

-- Define the triangle vertices
def v1 : ℝ × ℝ := (-1, 0)
def v2 : ℝ × ℝ := (7, 4)
def v3 (y : ℝ) : ℝ × ℝ := (7, y)

-- Define the area of the triangle
def triangle_area : ℝ := 32

-- Function to calculate the area of a triangle given three vertices
noncomputable def calculate_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)))

-- Theorem statement
theorem third_vertex_y_coordinate :
  ∃ y : ℝ, v3 y = (7, -4) ∧ calculate_area v1 v2 (v3 y) = triangle_area :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_y_coordinate_l1140_114086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1140_114035

/-- The probability of drawing all three colors in exactly 5 draws -/
def probability_all_colors_in_five_draws : ℚ :=
  14 / 81

/-- The set of possible ball colors -/
inductive BallColor
  | Red
  | Yellow
  | Blue
deriving Repr, DecidableEq

/-- A bag containing one ball of each color -/
def Bag := Finset BallColor

/-- The number of balls in the bag -/
def bag_size : ℕ := 3

/-- A sequence of drawn ball colors -/
def DrawSequence := List BallColor

/-- Check if a draw sequence contains all three colors -/
def contains_all_colors (seq : DrawSequence) : Prop :=
  seq.toFinset = {BallColor.Red, BallColor.Yellow, BallColor.Blue}

/-- The set of all possible 5-draw sequences -/
noncomputable def all_five_draw_sequences : Finset DrawSequence :=
  sorry

/-- The set of 5-draw sequences that contain all three colors -/
noncomputable def successful_five_draw_sequences : Finset DrawSequence :=
  sorry

theorem probability_theorem :
  (successful_five_draw_sequences.card : ℚ) / (all_five_draw_sequences.card : ℚ) = probability_all_colors_in_five_draws :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1140_114035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1140_114020

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 / 4^x) - (Real.log x / Real.log 4)

-- Theorem statement
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1140_114020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1140_114016

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x - 5) / ((x - 3)^2)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, f x < 0 ↔ (x < 3 ∨ (x > 3 ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1140_114016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1140_114065

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x+1)^2 + (y-5)^2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum :
  ∃ (m a : ℝ × ℝ),
    parabola m.1 m.2 ∧
    circle_eq a.1 a.2 ∧
    (∀ (m' a' : ℝ × ℝ),
      parabola m'.1 m'.2 →
      circle_eq a'.1 a'.2 →
      distance m a + distance m focus ≤ distance m' a' + distance m' focus) ∧
    distance m a + distance m focus = 5 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1140_114065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_term_without_x_l1140_114091

theorem binomial_expansion_term_without_x 
  (x y : ℝ) (x_nonzero : x ≠ 0) (y_nonzero : y ≠ 0) :
  ∃ (c : ℝ), c = -20 / y^3 ∧ 
  (∃ (terms : List ℝ), 
    (x / y - 1 / x)^6 = terms.sum ∧
    c ∈ terms ∧
    ∀ (term : ℝ), term ∈ terms → (term = c ∨ ∃ (n m : ℤ) (k : ℝ), term = x^n * y^m * k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_term_without_x_l1140_114091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_multiple_is_90_l1140_114024

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 9 ∨ d = 0

def smallest_valid_multiple (m : ℕ) : Prop :=
  m > 0 ∧ m % 18 = 0 ∧ is_valid_number m ∧
  ∀ k, 0 < k ∧ k < m ∧ k % 18 = 0 → ¬is_valid_number k

theorem smallest_valid_multiple_is_90 :
  smallest_valid_multiple 90 ∧ 90 / 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_multiple_is_90_l1140_114024
