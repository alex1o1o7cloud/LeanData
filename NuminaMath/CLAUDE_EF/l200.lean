import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_range_on_interval_l200_20013

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin x - Real.sqrt 3 * (Real.cos x)^2

-- Statement for the smallest positive period
theorem smallest_positive_period : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ p = Real.pi :=
by sorry

-- Statement for the maximum value
theorem max_value : 
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 1 - Real.sqrt 3 / 2 :=
by sorry

-- Statement for the range on [π/6, 2π/3]
theorem range_on_interval : 
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 6) ((2 * Real.pi) / 3) ∧ f x = y) ↔ 
  y ∈ Set.Icc (-(Real.sqrt 3) / 2) (1 - Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_range_on_interval_l200_20013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_product_l200_20078

def is_valid_element (x : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p > 26 → ¬(p ∣ x)

structure ValidSet :=
  (S : Finset ℕ)
  (distinct : S.card = 1985)
  (valid : ∀ x ∈ S, is_valid_element x)

theorem fourth_power_product (M : ValidSet) :
  ∃ (a b c d : ℕ) (n : ℕ),
    a ∈ M.S ∧ b ∈ M.S ∧ c ∈ M.S ∧ d ∈ M.S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b * c * d = n^4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_product_l200_20078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distances_and_height_l200_20054

/-- Represents a region on the map with its scale --/
structure Region where
  cm : ℚ  -- centimeters on the map
  km : ℚ  -- kilometers in reality

/-- Represents a distance on the map --/
structure MapDistance where
  region : Region
  cm : ℚ  -- centimeters on the map

/-- Represents a height difference on the map --/
structure HeightDifference where
  cm : ℚ  -- centimeters on the map
  m : ℚ   -- meters in reality

/-- The main theorem statement --/
theorem map_distances_and_height (regionA regionB regionC : Region)
    (distA distB : MapDistance) (totalPerimeter : ℚ) (heightAB : HeightDifference) :
    regionA.cm = 7 ∧ regionA.km = 35 ∧
    regionB.cm = 9 ∧ regionB.km = 45 ∧
    regionC.cm = 10 ∧ regionC.km = 30 ∧
    distA.region = regionA ∧ distA.cm = 15 ∧
    distB.region = regionB ∧ distB.cm = 10 ∧
    totalPerimeter = 355 ∧
    heightAB.cm = 2 ∧ heightAB.m = 500 →
    ∃ (distC : MapDistance) (heightBC : HeightDifference),
      distC.region = regionC ∧
      (distC.cm ≥ 76 ∧ distC.cm ≤ 77) ∧
      (distA.cm * regionA.km / regionA.cm +
       distB.cm * regionB.km / regionB.cm +
       distC.cm * regionC.km / regionC.cm = totalPerimeter) ∧
      heightBC.cm = 3 ∧
      heightBC.m = 750 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distances_and_height_l200_20054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l200_20098

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6) - 1

theorem symmetry_axis_of_f (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / ω = 2 * Real.pi / 3) :
  ∃ (k : ℤ), f ω (Real.pi / 9 + k * 2 * Real.pi / (3 * ω)) = f ω (Real.pi / 9 - k * 2 * Real.pi / (3 * ω)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l200_20098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l200_20014

theorem sequence_product (n : ℕ) (h : n > 1) : 
  let a : ℕ → ℕ := fun i => 2^i
  a (n-1) * a (n+1) = 4^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l200_20014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_number_l200_20028

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1)).im ≠ 0 ∧ 
  (a^2 - 3*a + 2 : ℂ).re = 0 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_number_l200_20028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_221_33_l200_20012

/-- The area of a triangle formed by the intersection of three lines -/
noncomputable def triangleArea (f g h : ℝ → ℝ) : ℝ :=
  let x₁ := (2 - 3) / (-3/4)
  let x₂ := (2 - 8) / 2
  let x₃ := (3 - 8) / (-11/4)
  let y₃ := -2 * x₃ + 8
  let base := x₂ - x₁
  let height := y₃ - 2
  1/2 * base * height

/-- The three lines that form the triangle -/
noncomputable def line1 : ℝ → ℝ := fun x ↦ 3/4 * x + 3
noncomputable def line2 : ℝ → ℝ := fun x ↦ -2 * x + 8
noncomputable def line3 : ℝ → ℝ := fun _ ↦ 2

theorem triangle_area_is_221_33 : 
  triangleArea line1 line2 line3 = 221/33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_221_33_l200_20012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l200_20080

/-- Represents a tetrahedron -/
structure Tetrahedron :=
  (vertices : Fin 4 → Type)
  (edges : ∀ (i j : Fin 4), i ≠ j → Type)

/-- Represents a path of the bug's movement -/
def BugPath (T : Tetrahedron) := List (Fin 4)

/-- A path is valid if it has length 4 and all elements are distinct -/
def ValidPath (T : Tetrahedron) (p : BugPath T) : Prop :=
  p.length = 4 ∧ p.Nodup

/-- The probability of choosing a specific path -/
def PathProbability (T : Tetrahedron) (p : BugPath T) : ℚ :=
  (1 : ℚ) / (3^3)

/-- The total number of possible paths -/
def TotalPaths : ℕ := 4 * 3^3

/-- The number of valid paths (visiting all vertices exactly once) -/
def ValidPathCount : ℕ := 12

theorem bug_probability (T : Tetrahedron) :
  (ValidPathCount : ℚ) / (TotalPaths : ℚ) = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l200_20080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_l200_20063

-- Define the parametric equations of the ellipse
noncomputable def ellipse_x (t : ℝ) : ℝ := 3 * (Real.sin t + 2) / (3 - Real.cos t)
noncomputable def ellipse_y (t : ℝ) : ℝ := 2 * (Real.cos t - 4) / (3 - Real.cos t)

-- Define the property that A, B, C, D, E, F are integers
def are_integers (A B C D E F : ℤ) : Prop := True

-- Define the GCD condition
def gcd_condition (A B C D E F : ℤ) : Prop :=
  Nat.gcd (A.natAbs) (Nat.gcd (B.natAbs) (Nat.gcd (C.natAbs) (Nat.gcd (D.natAbs) (Nat.gcd (E.natAbs) (F.natAbs))))) = 1

-- State the theorem
theorem ellipse_equation_sum :
  ∃ (A B C D E F : ℤ),
    (∀ x y : ℝ, (∃ t : ℝ, x = ellipse_x t ∧ y = ellipse_y t) →
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧
    are_integers A B C D E F ∧
    gcd_condition A B C D E F ∧
    A.natAbs + B.natAbs + C.natAbs + D.natAbs + E.natAbs + F.natAbs = 713 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_sum_l200_20063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_threeExamplesProvided_l200_20037

structure SelfInvestment where
  name : String
  benefit : String

def education : SelfInvestment :=
  { name := "Education"
  , benefit := "Enhances employability and earning potential" }

def physicalHealth : SelfInvestment :=
  { name := "Physical Health"
  , benefit := "Reduces future healthcare costs and enhances overall well-being" }

def reading : SelfInvestment :=
  { name := "Reading Books"
  , benefit := "Cultivates intellectual growth and contributes to personal and professional success" }

def examplesOfSelfInvestment : List SelfInvestment :=
  [education, physicalHealth, reading]

#eval examplesOfSelfInvestment.length

theorem threeExamplesProvided : examplesOfSelfInvestment.length = 3 := by
  rfl

#check threeExamplesProvided

end NUMINAMATH_CALUDE_ERRORFEEDBACK_threeExamplesProvided_l200_20037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l200_20027

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x + 1 else x^2

theorem f_composition_negative_two : f (f (-2)) = 5 := by
  -- Evaluate f(-2)
  have h1 : f (-2) = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 5 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f (-2)) = f 4 := by rw [h1]
    _          = 5   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l200_20027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l200_20069

/-- Calculates the length of a train given its speed in km/h and time in seconds to pass a fixed point. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating that a train traveling at 80 km/h passing a point in 8.999280057595392 seconds is 200 meters long. -/
theorem train_length_calculation :
  let speed := (80 : ℝ)
  let time := (8.999280057595392 : ℝ)
  trainLength speed time = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l200_20069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l200_20079

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number expression
noncomputable def complex_expr : ℂ := i^3 / (2 - i)

-- State the theorem
theorem complex_number_equality : complex_expr = (1/5 : ℂ) - (2/5 : ℂ) * i := by
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l200_20079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_k_l200_20039

/-- For a positive integer m, v₂(m) represents the highest power of 2 that divides m. -/
def v₂ (m : ℕ) : ℕ := sorry

/-- S(m) is the sum of the digits of m in binary representation. -/
def S (m : ℕ) : ℕ := sorry

/-- The property that k must satisfy for all positive integers n. -/
def satisfies_property (k : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ¬(2^((k-1)*n+1) ∣ (k*n).factorial / n.factorial)

/-- Main theorem: k satisfies the property if and only if it's a power of 2. -/
theorem characterization_of_k (k : ℕ) (hk : k > 0) :
  satisfies_property k ↔ ∃ a : ℕ, k = 2^a := by
  sorry

#check characterization_of_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_k_l200_20039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l200_20030

/-- The angle between two lines given by their equations -/
noncomputable def angle_between_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  Real.arccos (|a₁ * a₂ + b₁ * b₂| / (Real.sqrt (a₁^2 + b₁^2) * Real.sqrt (a₂^2 + b₂^2)))

/-- Theorem: The angle between the lines x + 5y - 3 = 0 and 2x - 3y + 4 = 0 is π/4 -/
theorem angle_between_specific_lines :
  angle_between_lines 1 5 (-3) 2 (-3) 4 = π / 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval angle_between_lines 1 5 (-3) 2 (-3) 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l200_20030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_valid_arrangements_l200_20096

/-- Represents a valid arrangement of numbers on an n × n chessboard -/
def ValidArrangement (n : ℕ) : Type :=
  Fin n → Fin n → Fin (n^2)

/-- Checks if a row forms an arithmetic progression -/
def IsArithmeticRow (n : ℕ) (arr : ValidArrangement n) (row : Fin n) : Prop :=
  ∃ d : ℤ, ∀ i j : Fin n, (arr row j : ℤ) - (arr row i : ℤ) = d * ((j : ℤ) - (i : ℤ))

/-- Checks if a column forms an arithmetic progression -/
def IsArithmeticColumn (n : ℕ) (arr : ValidArrangement n) (col : Fin n) : Prop :=
  ∃ d : ℤ, ∀ i j : Fin n, (arr j col : ℤ) - (arr i col : ℤ) = d * ((j : ℤ) - (i : ℤ))

/-- Checks if the arrangement is valid (all rows and columns are arithmetic progressions) -/
def IsValidArrangement (n : ℕ) (arr : ValidArrangement n) : Prop :=
  (∀ row : Fin n, IsArithmeticRow n arr row) ∧
  (∀ col : Fin n, IsArithmeticColumn n arr col)

/-- The main theorem stating that there are exactly 8 valid arrangements -/
theorem eight_valid_arrangements (n : ℕ) (h : n ≥ 3) :
  ∃! (arrangements : Finset (ValidArrangement n)),
    arrangements.card = 8 ∧
    ∀ arr ∈ arrangements, IsValidArrangement n arr :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_valid_arrangements_l200_20096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l200_20015

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 8^x)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ≤ 2/3) ∧
  (∀ y, y ∈ Set.range f → 0 ≤ y ∧ y < 2) ∧
  (∀ x, f x ≤ 1 → 1/3 * Real.log 3 / Real.log 2 ≤ x ∧ x ≤ 2/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l200_20015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l200_20016

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 * (n + 3) / (n + 2) * a n

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i => a i)

theorem sequence_properties :
  (∀ n : ℕ, a n = 2^n * (n + 2)) ∧
  (∀ n : ℕ, n > 0 → (Finset.range n).sum (λ i => 1 / S (i + 1)) ≤ n / (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l200_20016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_theorem_l200_20087

-- Define the total number of students
def total_students : ℕ := 1500

-- Define percentages for shirt colors
def blue_shirt_percentage : ℚ := 35/100
def red_shirt_percentage : ℚ := 20/100
def yellow_shirt_percentage : ℚ := 10/100
def green_shirt_percentage : ℚ := 5/100

-- Define percentages for patterns on blue shirts
def blue_stripes_percentage : ℚ := 40/100
def blue_polka_dots_percentage : ℚ := 15/100
def blue_floral_percentage : ℚ := 10/100

-- Define percentages for patterns on red shirts
def red_stripes_percentage : ℚ := 30/100
def red_polka_dots_percentage : ℚ := 25/100
def red_floral_percentage : ℚ := 5/100

-- Define percentages for patterns on yellow shirts
def yellow_stripes_percentage : ℚ := 20/100
def yellow_polka_dots_percentage : ℚ := 10/100

-- Define percentages for patterns on green shirts
def green_stripes_percentage : ℚ := 10/100
def green_polka_dots_percentage : ℚ := 30/100
def green_floral_percentage : ℚ := 50/100

-- Define percentages for accessories
def glasses_percentage : ℚ := 18/100
def hat_percentage : ℚ := 12/100
def scarf_percentage : ℚ := 10/100

theorem student_count_theorem :
  ∃ n : ℕ, n = 53 ∧
  n = (Int.toNat ⌊(total_students : ℚ) * blue_shirt_percentage * blue_stripes_percentage * glasses_percentage⌋) +
      (Int.toNat ⌊(total_students : ℚ) * red_shirt_percentage * red_polka_dots_percentage * hat_percentage⌋) +
      (Int.toNat ⌊(total_students : ℚ) * yellow_shirt_percentage * yellow_polka_dots_percentage * hat_percentage⌋) +
      (Int.toNat ⌊(total_students : ℚ) * green_shirt_percentage * green_floral_percentage * scarf_percentage⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_theorem_l200_20087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_fall_probabilities_l200_20052

/-- Represents the probability of a dwarf falling -/
def probability_of_falling (p : ℝ) : Prop := 0 < p ∧ p < 1

/-- Represents the number of dwarfs -/
def number_of_dwarfs (n : ℕ) : Prop := n > 0

/-- Probability that exactly k dwarfs fall -/
noncomputable def probability_exactly_k_fall (n k : ℕ) (p : ℝ) : ℝ := 
  p * (1 - p) ^ (n - k)

/-- Expected number of fallen dwarfs -/
noncomputable def expected_number_of_falls (n : ℕ) (p : ℝ) : ℝ := 
  n + 1 - 1 / p + (1 - p) ^ (n + 1) / p

theorem dwarf_fall_probabilities 
  (n : ℕ) 
  (k : ℕ) 
  (p : ℝ) 
  (h1 : number_of_dwarfs n) 
  (h2 : probability_of_falling p) :
  (probability_exactly_k_fall n k p = p * (1 - p) ^ (n - k)) ∧ 
  (expected_number_of_falls n p = n + 1 - 1 / p + (1 - p) ^ (n + 1) / p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_fall_probabilities_l200_20052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_red_segments_checkerboard_l200_20092

/-- The sum of red segment lengths on the diagonal of a checkerboard -/
noncomputable def sum_red_segments (m n : ℕ+) : ℝ :=
  (m.val * n.val + 1 : ℝ) / (2 * m.val * n.val) * Real.sqrt (m.val^2 + n.val^2)

/-- Theorem stating the sum of red segment lengths on the diagonal of a checkerboard -/
theorem sum_red_segments_checkerboard (m n : ℕ+) :
  let diagonal_length : ℝ := Real.sqrt (m.val^2 + n.val^2)
  let red_blue_difference : ℝ := diagonal_length / (m.val * n.val)
  let total_red_blue : ℝ := diagonal_length
  sum_red_segments m n = (total_red_blue + red_blue_difference) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_red_segments_checkerboard_l200_20092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_correct_l200_20032

/-- The sum of the series ∑_{n=0}^∞ (n+6) x^(7n) for x ∈ (-1, 1) -/
noncomputable def seriesSum (x : ℝ) : ℝ := (6 - 5 * x^7) / (1 - x^7)^2

/-- The series ∑_{n=0}^∞ (n+6) x^(7n) -/
def series (x : ℝ) (n : ℕ) : ℝ := (n + 6) * x^(7 * n)

theorem series_sum_correct (x : ℝ) (h : x ∈ Set.Ioo (-1) 1) :
  Summable (series x) ∧ tsum (series x) = seriesSum x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_correct_l200_20032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l200_20076

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  (arithmetic_sequence a₁ d 4 = 10) →
  (arithmetic_sequence a₁ d 10 = -2) →
  (∃ n : ℕ, arithmetic_sum a₁ d n = 60) →
  (∃ n : ℕ, n = 5 ∨ n = 12) := by
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l200_20076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l200_20085

noncomputable def f (x : ℝ) : ℝ := 1 / x - 2

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f x = 1 / x - 2) ∧
  (Set.range f = {y : ℝ | y < -2 ∨ y > -2}) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l200_20085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l200_20036

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a - 1 / b

-- State the theorem
theorem diamond_calculation :
  (diamond (diamond 1 2) 3) - (diamond 1 (diamond 2 3)) = -7/30 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [sub_eq_add_neg, add_assoc, add_comm, add_left_comm]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l200_20036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_is_linear_l200_20058

-- Define the concept of a linear function
noncomputable def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := λ x ↦ x^3
noncomputable def f2 : ℝ → ℝ := λ x ↦ -2*x + 1
noncomputable def f3 : ℝ → ℝ := λ x ↦ 2/x
noncomputable def f4 : ℝ → ℝ := λ x ↦ 2*x^2 + 1

-- Theorem stating that only f2 is a linear function
theorem only_f2_is_linear :
  ¬IsLinearFunction f1 ∧
  IsLinearFunction f2 ∧
  ¬IsLinearFunction f3 ∧
  ¬IsLinearFunction f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_is_linear_l200_20058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_l200_20077

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 4)

theorem symmetric_graph (φ : ℝ) : 
  (∀ x, f (2 * x - φ) = -f (-2 * x + φ)) → φ = 3 * Real.pi / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_l200_20077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_condition_l200_20045

theorem sin_cos_sum_condition (α β : ℝ) (k : ℤ) :
  (∃ k, α + β = 2 * k * Real.pi + Real.pi / 6) →
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 / 2 ∧
  ∃ α β, Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 / 2 ∧ ∀ k, α + β ≠ 2 * k * Real.pi + Real.pi / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_condition_l200_20045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_53_l200_20055

theorem remainder_sum_mod_53 (a b c : ℕ) 
  (ha : a % 53 = 33)
  (hb : b % 53 = 14)
  (hc : c % 53 = 9) :
  (a + b + c) % 53 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_53_l200_20055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_start_time_l200_20050

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Converts a Time to hours as a real number -/
noncomputable def timeToHours (t : Time) : ℝ :=
  t.hours + t.minutes / 60

/-- Calculates the travel time given distance and speed -/
noncomputable def travelTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Represents the train schedule problem -/
structure TrainSchedule where
  avgSpeed : ℝ
  distance : ℝ
  arrivalTime : Time
  haltDuration : ℝ

theorem train_start_time (schedule : TrainSchedule)
  (hSpeed : schedule.avgSpeed = 87)
  (hDistance : schedule.distance = 348)
  (hArrival : schedule.arrivalTime = ⟨13, 45, by norm_num⟩)
  (hHalt : schedule.haltDuration = 0.75) :
  let totalTime := travelTime schedule.distance schedule.avgSpeed + schedule.haltDuration
  let startTimeHours := timeToHours schedule.arrivalTime - totalTime
  ⌊startTimeHours⌋ = 9 ∧ (startTimeHours - ⌊startTimeHours⌋) * 60 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_start_time_l200_20050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l200_20082

def num_math_books : ℕ := 4
def num_science_books : ℕ := 4

def arrangement_count : ℕ := 5760

theorem book_arrangement_count :
  (num_science_books * (num_science_books - 1)) *
  num_math_books *
  Nat.factorial (num_math_books + num_science_books - 3) = arrangement_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l200_20082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_proof_l200_20086

-- Define the types for balls
structure Ball where
  color : String
  weight : ℚ
deriving Repr

-- Define the balance relation
def balances (a b : List Ball) : Prop :=
  (a.map Ball.weight).sum = (b.map Ball.weight).sum

theorem balance_proof :
  -- Given conditions
  (balances [Ball.mk "green" 1, Ball.mk "green" 1, Ball.mk "green" 1, Ball.mk "green" 1]
            [Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1]) →
  (balances [Ball.mk "yellow" 1, Ball.mk "yellow" 1, Ball.mk "yellow" 1]
            [Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1]) →
  (balances [Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1]
            [Ball.mk "white" 1, Ball.mk "white" 1, Ball.mk "white" 1, Ball.mk "white" 1, Ball.mk "white" 1]) →
  (balances [Ball.mk "red" 1, Ball.mk "red" 1]
            [Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1]) →
  -- Prove
  (balances [Ball.mk "green" 1, Ball.mk "green" 1, Ball.mk "green" 1, Ball.mk "green" 1, Ball.mk "green" 1,
             Ball.mk "yellow" 1, Ball.mk "yellow" 1, Ball.mk "yellow" 1,
             Ball.mk "red" 1, Ball.mk "red" 1, Ball.mk "red" 1]
            [Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1, Ball.mk "blue" 1,
             Ball.mk "blue" 1]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_proof_l200_20086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l200_20004

noncomputable def fixed_cost : ℝ := 20000
noncomputable def additional_cost_per_unit : ℝ := 100

noncomputable def total_revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

noncomputable def profit (x : ℝ) : ℝ := total_revenue x - fixed_cost - additional_cost_per_unit * x

theorem max_profit_at_300 :
  ∃ (max_x : ℝ), max_x = 300 ∧
  ∀ (x : ℝ), profit x ≤ profit max_x ∧
  profit max_x = 25000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l200_20004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l200_20091

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_inequality (a b c S : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S = area_triangle a b c) :
  S ≤ (a^2 + b^2 + c^2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l200_20091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_vectors_l200_20070

/-- Given a regular hexagon ABCDEF with vectors a and b, prove the vectors to various points -/
theorem regular_hexagon_vectors (a b : ℝ × ℝ) : 
  let ABCDEF : Set (ℝ × ℝ) := sorry
  let is_regular_hexagon : sorry := sorry
  let A : ℝ × ℝ := sorry
  let B : ℝ × ℝ := sorry
  let C : ℝ × ℝ := sorry
  let D : ℝ × ℝ := sorry
  let E : ℝ × ℝ := sorry
  let F : ℝ × ℝ := sorry
  let M : ℝ × ℝ := sorry
  let AB : ℝ × ℝ := B - A
  let AF : ℝ × ℝ := F - A
  let AD : ℝ × ℝ := D - A
  let BD : ℝ × ℝ := D - B
  let FD : ℝ × ℝ := D - F
  let BM : ℝ × ℝ := M - B

  AB = a ∧ 
  AF = b ∧ 
  M = (E + F) / 2 →
  AD = 2 • a + 2 • b ∧
  BD = a + 2 • b ∧
  FD = 2 • a + b ∧
  BM = (-1/2 : ℝ) • a + (3/2 : ℝ) • b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_vectors_l200_20070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l200_20035

noncomputable def h (x : ℝ) : ℝ := (4 * x^2 + 2 * x - 3) / (x - 5)

theorem h_domain :
  {x : ℝ | ∃ y, h x = y} = {x | x < 5 ∨ x > 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l200_20035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l200_20034

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x + 2)^2 / 9 + (y - 2)^2 / 9 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_circle_ellipse :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ ellipse_eq x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      circle_eq x3 y3 → ellipse_eq x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l200_20034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_neg_three_solutions_l200_20067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -(3^x) else 1 - x^2

-- State the theorem
theorem f_equals_neg_three_solutions :
  {x : ℝ | f x = -3} = {1, -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_neg_three_solutions_l200_20067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_length_l200_20033

-- Define the radius of the circular room
noncomputable def room_radius : ℝ := 75

-- Define the angle formed at the center
noncomputable def center_angle : ℝ := 120 * (Real.pi / 180)

-- Define the length of one side of the triangle
noncomputable def given_side : ℝ := 120

-- Theorem statement
theorem fly_path_length :
  let side1 := 2 * room_radius * Real.sin (center_angle / 2)
  let side2 := given_side
  let side3 := given_side
  side1 + side2 + side3 = 240 + 75 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_length_l200_20033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_in_interval_l200_20059

-- Define the function f(θ) = 2^(cos θ) - sin θ
noncomputable def f (θ : ℝ) : ℝ := 2^(Real.cos θ) - Real.sin θ

-- State the theorem
theorem two_solutions_in_interval :
  ∃ (θ₁ θ₂ : ℝ), θ₁ ≠ θ₂ ∧ 
  0 ≤ θ₁ ∧ θ₁ ≤ 2*Real.pi ∧
  0 ≤ θ₂ ∧ θ₂ ≤ 2*Real.pi ∧
  f θ₁ = 0 ∧ f θ₂ = 0 ∧
  ∀ θ, 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ f θ = 0 → θ = θ₁ ∨ θ = θ₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_in_interval_l200_20059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_dilution_l200_20044

/-- Represents a solution with a given mass and solute mass fraction -/
structure Solution where
  mass : ℝ
  soluteMassFraction : ℝ

/-- Calculates the mass of solute in a solution -/
noncomputable def soluteMass (s : Solution) : ℝ := s.mass * s.soluteMassFraction

/-- Dilutes a solution by adding water -/
noncomputable def dilute (s : Solution) (waterAdded : ℝ) : Solution where
  mass := s.mass + waterAdded
  soluteMassFraction := soluteMass s / (s.mass + waterAdded)

/-- Theorem stating the correct amount of water to add for dilution -/
theorem correct_dilution (initialSolution : Solution) 
    (h1 : initialSolution.mass = 50)
    (h2 : initialSolution.soluteMassFraction = 0.2)
    (targetFraction : ℝ) 
    (h3 : targetFraction = 0.1) :
    ∃ (waterAdded : ℝ), 
      waterAdded = 50 ∧ 
      (dilute initialSolution waterAdded).soluteMassFraction = targetFraction := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_dilution_l200_20044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l200_20094

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 9*x + 15) / (3*x + 4)

-- State the theorem
theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - (x + 3)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l200_20094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l200_20097

/-- Represents a single-digit number -/
def SingleDigit : Type := { n : ℕ // n < 10 }

/-- Converts a list of single digits to a natural number -/
def listToNat (digits : List SingleDigit) : ℕ :=
  digits.foldr (fun d acc => acc * 10 + d.val) 0

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution
  (A B C D E F : SingleDigit)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
                B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
                C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
                D ≠ E ∧ D ≠ F ∧
                E ≠ F)
  (h_eq1 : listToNat [A, B, C, D] + listToNat [C, D] = listToNat [C, ⟨8, by norm_num⟩, C, E, C])
  (h_eq2 : listToNat [C, ⟨8, by norm_num⟩, C, E, C] + listToNat [A, B, C, D] = 
           listToNat [F, ⟨8, by norm_num⟩, F, ⟨6, by norm_num⟩, C]) :
  listToNat [A, B, C, D, E, F] = 201973 :=
by sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l200_20097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_side_is_three_l200_20018

/-- Represents the dimensions of a square sandbox -/
structure Sandbox where
  side : ℝ
  is_square : side > 0

/-- Represents the properties of sand bags -/
structure SandBag where
  price : ℝ
  coverage : ℝ

/-- Calculates the side length of a square sandbox given the total cost and sand bag properties -/
noncomputable def calculate_sandbox_side (total_cost : ℝ) (bag : SandBag) : ℝ :=
  Real.sqrt ((total_cost / bag.price) * bag.coverage)

theorem sandbox_side_is_three 
  (sandbox : Sandbox) 
  (bag : SandBag)
  (h1 : bag.price = 4)
  (h2 : bag.coverage = 3)
  (h3 : calculate_sandbox_side 12 bag = sandbox.side) :
  sandbox.side = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_side_is_three_l200_20018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_cost_theorem_l200_20021

/-- Calculates the total cost of watermelons given the specified conditions -/
noncomputable def total_cost (watermelon_weight : ℝ) (daily_prices : List ℝ) (discount_rate : ℝ) 
  (tax_rate : ℝ) (num_watermelons : ℕ) : ℝ :=
  let avg_price := (daily_prices.sum) / daily_prices.length
  let total_weight := watermelon_weight * (num_watermelons : ℝ)
  let initial_cost := total_weight * avg_price
  let discounted_cost := if num_watermelons > 15 then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + tax_rate)
  final_cost

/-- Theorem stating the total cost of 18 watermelons under given conditions -/
theorem watermelon_cost_theorem : 
  let watermelon_weight : ℝ := 23
  let daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
  let discount_rate : ℝ := 0.10
  let tax_rate : ℝ := 0.05
  let num_watermelons : ℕ := 18
  abs (total_cost watermelon_weight daily_prices discount_rate tax_rate num_watermelons - 796.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_cost_theorem_l200_20021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l200_20084

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.cos x)^2 + 4 * Real.sqrt 6 * Real.cos x + 6) + 
  Real.sqrt (4 * (Real.cos x)^2 - 8 * Real.sqrt 6 * Real.cos x + 4 * Real.sqrt 2 * Real.sin x + 22)

theorem f_max_value : 
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ 2 * (Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l200_20084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l200_20008

/-- The length of the major axis of an ellipse with given foci and x-axis tangency -/
theorem ellipse_major_axis_length : ∀ (E : Set (ℝ × ℝ)),
  (∃ (X : ℝ × ℝ), X ∈ E ∧ X.2 = 0) →  -- E is tangent to x-axis
  (∀ (P : ℝ × ℝ), P ∈ E → 
    dist P (4, 15) + dist P (44, 50) = dist P (4, -15) + dist P (44, 50)) →  -- reflection property
  (∀ (P : ℝ × ℝ), P ∈ E → 
    dist P (4, 15) + dist P (44, 50) = dist (4, -15) (44, 50)) →  -- constant sum property
  dist (4, -15) (44, 50) = 5 * Real.sqrt 233 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l200_20008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameter_t_for_A_length_MN_l200_20000

noncomputable section

-- Define the polar coordinates of point A
def A_polar : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ := (3/2 - Real.sqrt 2 / 2 * t, 1/2 + Real.sqrt 2 / 2 * t)

-- Define the parametric equations of curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- State that point A lies on line l
axiom A_on_l : ∃ t : ℝ, line_l t = (Real.sqrt 2 * Real.cos (Real.pi / 4), Real.sqrt 2 * Real.sin (Real.pi / 4))

-- Define the intersection points M and N
axiom M_N_exist : ∃ M N : ℝ × ℝ, (∃ t : ℝ, line_l t = M) ∧ (∃ θ : ℝ, curve_C θ = M) ∧
                                 (∃ t : ℝ, line_l t = N) ∧ (∃ θ : ℝ, curve_C θ = N) ∧
                                 M ≠ N

-- Theorem 1: The parameter t corresponding to point A on line l is √2/2
theorem parameter_t_for_A : ∃ t : ℝ, t = Real.sqrt 2 / 2 ∧ line_l t = (Real.sqrt 2 * Real.cos (Real.pi / 4), Real.sqrt 2 * Real.sin (Real.pi / 4)) := by
  sorry

-- Theorem 2: The length of the chord MN is 4√2/5
theorem length_MN : ∀ M N : ℝ × ℝ, (∃ t : ℝ, line_l t = M) → (∃ θ : ℝ, curve_C θ = M) →
                                   (∃ t : ℝ, line_l t = N) → (∃ θ : ℝ, curve_C θ = N) →
                                   M ≠ N →
                                   Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 * Real.sqrt 2 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameter_t_for_A_length_MN_l200_20000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_shift_l200_20026

/-- Given a point with rectangular coordinates (2, 3, -6) and corresponding
    spherical coordinates (ρ, θ, φ), prove that the point with spherical
    coordinates (ρ, θ, φ - π/2) has rectangular coordinates (12/√13, 18/√13, √13). -/
theorem spherical_coordinate_shift (ρ θ φ : Real) :
  ρ * Real.sin φ * Real.cos θ = 2 ∧
  ρ * Real.sin φ * Real.sin θ = 3 ∧
  ρ * Real.cos φ = -6 →
  ρ * Real.sin (φ - Real.pi/2) * Real.cos θ = 12 / Real.sqrt 13 ∧
  ρ * Real.sin (φ - Real.pi/2) * Real.sin θ = 18 / Real.sqrt 13 ∧
  ρ * Real.cos (φ - Real.pi/2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_shift_l200_20026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l200_20073

-- Define the ellipse parameters
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Define the eccentricity
noncomputable def e : ℝ := 1 / 2

-- Define the roots of the equation ax² + bx - c = 0
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry

-- Theorem statement
theorem point_inside_circle 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = c / a) 
  (h4 : x₁ + x₂ = -b / a) 
  (h5 : x₁ * x₂ = -c / a) 
  (h6 : a^2 * e^2 + b^2 = a^2) :
  x₁^2 + x₂^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l200_20073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_specific_project_completion_time_l200_20048

/-- The time needed for two people to complete a project together, given their individual completion times. -/
theorem project_completion_time (time_A time_B : ℚ) (h_A : time_A > 0) (h_B : time_B > 0) :
  (1 / (1 / time_A + 1 / time_B)) = (time_A * time_B) / (time_A + time_B) := by
  sorry

/-- The specific case where person A takes 12 days and person B takes 8 days. -/
theorem specific_project_completion_time :
  (1 / (1 / 12 + 1 / 8)) = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_specific_project_completion_time_l200_20048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_3_l200_20093

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.tan x

-- State the theorem
theorem derivative_f_at_pi_over_3 :
  deriv f (π / 3) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_3_l200_20093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l200_20090

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (50 * Real.pi)

-- Define the equation
def equation (x : ℝ) : Prop := Real.sin x = (1/3) ^ x

-- Define the solution set
def solution_set : Set ℝ := {x ∈ interval | equation x}

-- Theorem statement
theorem solution_count : ∃ (s : Finset ℝ), s.card = 50 ∧ ∀ x ∈ s, x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l200_20090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l200_20081

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- The nth term of a geometric sequence -/
noncomputable def GeometricSequence.nthTerm (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * g.q^(n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSequence.sum (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_common_ratio (g : GeometricSequence) :
  g.nthTerm 5 = 2 * g.sum 4 + 3 →
  g.nthTerm 6 = 2 * g.sum 5 + 3 →
  g.q = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l200_20081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_split_slope_l200_20057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The line that passes through a given point with a given slope -/
def line_through_point (p : Point) (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - p.x) + p.y

/-- Whether a line splits the area of a circle equally -/
def splits_circle_equally (c : Circle) (line : ℝ → ℝ) : Prop := sorry

/-- The total area of multiple circles -/
def total_area (circles : List Circle) : ℝ := sorry

/-- Whether a line splits the total area of multiple circles equally -/
def splits_total_area_equally (circles : List Circle) (line : ℝ → ℝ) : Prop :=
  ∃ (area_above area_below : ℝ),
    area_above + area_below = total_area circles ∧
    area_above = area_below

/-- The main theorem -/
theorem equal_area_split_slope :
  let c1 := { center := ⟨15, 93⟩, radius := 4 : Circle }
  let c2 := { center := ⟨18, 77⟩, radius := 4 : Circle }
  let c3 := { center := ⟨20, 85⟩, radius := 4 : Circle }
  let circles := [c1, c2, c3]
  let p := ⟨18, 77⟩
  ∃ (m : ℝ),
    splits_total_area_equally circles (line_through_point p m) ∧
    |m| = 24/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_split_slope_l200_20057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_parallelogram_l200_20025

/-- Represents a quadrilateral with side lengths -/
structure Quadrilateral :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Represents a rectangle -/
structure Rectangle extends Quadrilateral :=
  (is_rectangle : side1 = side3 ∧ side2 = side4)

/-- Represents a parallelogram -/
structure Parallelogram extends Quadrilateral :=
  (is_parallelogram : side1 = side3 ∧ side2 = side4)

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ :=
  q.side1 + q.side2 + q.side3 + q.side4

/-- Calculates the area of a rectangle -/
def area_rectangle (r : Rectangle) : ℝ :=
  r.side1 * r.side2

/-- Calculates the area of a parallelogram -/
def area_parallelogram (p : Parallelogram) (height : ℝ) : ℝ :=
  p.side1 * height

theorem rectangle_to_parallelogram 
  (r : Rectangle) 
  (p : Parallelogram) 
  (h : r.side1 = p.side1 ∧ r.side2 = p.side2 ∧ r.side3 = p.side3 ∧ r.side4 = p.side4) 
  (height_p : ℝ) 
  (h_height : height_p < r.side2) : 
  area_parallelogram p height_p < area_rectangle r ∧ 
  perimeter (p.toQuadrilateral) = perimeter (r.toQuadrilateral) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_parallelogram_l200_20025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l200_20064

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - x - 1)

theorem f_strictly_increasing : 
  StrictMonoOn f (Set.Iio (1/2 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l200_20064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_twenty_minutes_l200_20062

/-- Represents the tank and pipe system -/
structure TankSystem where
  tankCapacity : ℕ
  pipeARate : ℕ
  pipeBRate : ℕ
  pipeCRate : ℕ

/-- Calculates the net water added in one cycle -/
def netWaterPerCycle (system : TankSystem) : ℕ :=
  system.pipeARate * 1 + system.pipeBRate * 2 - system.pipeCRate * 2

/-- Calculates the time required to fill the tank -/
noncomputable def timeToFillTank (system : TankSystem) : ℕ :=
  let cycleTime : ℕ := 5
  let numCycles : ℕ := system.tankCapacity / netWaterPerCycle system
  cycleTime * numCycles

/-- Theorem stating that the time to fill the tank is 20 minutes -/
theorem fill_time_is_twenty_minutes (system : TankSystem)
  (h1 : system.tankCapacity = 1000)
  (h2 : system.pipeARate = 200)
  (h3 : system.pipeBRate = 50)
  (h4 : system.pipeCRate = 25) :
  timeToFillTank system = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_twenty_minutes_l200_20062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l200_20002

/-- The limiting sum of a geometric series with first term a and common ratio r -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_sum_example :
  let a : ℝ := 5
  let r : ℝ := -1/2
  geometricSeriesSum a r = 10/3 := by
  -- Unfold the definition of geometricSeriesSum
  unfold geometricSeriesSum
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l200_20002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_probability_l200_20065

/-- The probability of hanging 4 Escher prints consecutively among 12 art pieces -/
theorem escher_prints_probability (total_pieces : ℕ) (escher_prints : ℕ) 
  (h1 : total_pieces = 12) (h2 : escher_prints = 4) : 
  (Nat.factorial (escher_prints - 1)) * (Nat.factorial (total_pieces - escher_prints + 1)) / 
  (Nat.factorial total_pieces) = 1 / 1320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_probability_l200_20065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_separation_theorem_l200_20005

/-- Represents the mass of the separated part from the first alloy piece -/
def x : ℝ := sorry

/-- Represents the percentage of copper in the first alloy piece -/
def p : ℝ := sorry

/-- Represents the percentage of copper in the second alloy piece -/
def q : ℝ := sorry

/-- The mass of the first alloy piece -/
def m₁ : ℝ := 6

/-- The mass of the second alloy piece -/
def m₂ : ℝ := 8

theorem alloy_separation_theorem (h_p_ne_q : p ≠ q) 
  (h_x_pos : x > 0) (h_x_lt_m₁ : x < m₁) (h_2x_lt_m₂ : 2 * x < m₂) :
  (p * x + q * (m₂ - 2 * x)) / (m₂ - x) = (p * (m₁ - x) + q * (2 * x)) / (m₁ + x) →
  x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_separation_theorem_l200_20005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_occupancy_percentage_l200_20061

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the maximum number of cubes that can fit along one dimension -/
def maxCubesAlongDimension (boxDim : ℕ) (cubeSide : ℕ) : ℕ :=
  boxDim / cubeSide

/-- Calculates the volume occupied by cubes in the box -/
def volumeOccupiedByCubes (box : BoxDimensions) (cube : CubeDimensions) : ℕ :=
  let maxLength := maxCubesAlongDimension box.length cube.side
  let maxWidth := maxCubesAlongDimension box.width cube.side
  let maxHeight := maxCubesAlongDimension box.height cube.side
  maxLength * maxWidth * maxHeight * (cube.side ^ 3)

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cube : CubeDimensions) : ℚ :=
  let occupied := volumeOccupiedByCubes box cube
  let total := boxVolume box
  (occupied : ℚ) / (total : ℚ) * 100

/-- The main theorem stating that the percentage of volume occupied by cubes is approximately 55.10% -/
theorem cubes_occupancy_percentage (box : BoxDimensions) (cube : CubeDimensions) :
    box.length = 8 → box.width = 7 → box.height = 14 → cube.side = 3 →
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |percentageOccupied box cube - 5510/100| < ε := by
  sorry

#eval percentageOccupied ⟨8, 7, 14⟩ ⟨3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_occupancy_percentage_l200_20061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_trip_distance_l200_20011

/-- Calculates the total distance traveled given the times for each direction of a round trip and the average speed. -/
def round_trip_distance (time_to : ℚ) (time_from : ℚ) (avg_speed : ℚ) : ℚ :=
  let total_time : ℚ := time_to + time_from
  avg_speed * total_time / 60

/-- Proves that for Margo's specific trip, the total distance is 2 miles. -/
theorem margo_trip_distance : round_trip_distance 15 25 3 = 2 := by
  -- Unfold the definition of round_trip_distance
  unfold round_trip_distance
  -- Simplify the arithmetic
  simp [add_comm, mul_comm, mul_assoc]
  -- The proof is complete
  rfl

#eval round_trip_distance 15 25 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_trip_distance_l200_20011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_sum_l200_20023

noncomputable section

-- Define the vertices of the triangle
def P : ℝ × ℝ := (-2, 10)
def Q : ℝ × ℝ := (3, -2)
def R : ℝ × ℝ := (10, -2)

-- Define the midpoint of PR
noncomputable def M : ℝ × ℝ := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)

-- Define the slope of the line QM
noncomputable def m : ℝ := (M.2 - Q.2) / (M.1 - Q.1)

-- Define the y-intercept of the line QM
noncomputable def b : ℝ := Q.2 - m * Q.1

-- Theorem statement
theorem area_bisecting_line_sum :
  m + b = -14 := by
  -- Proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_sum_l200_20023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_cannot_hit_opposite_middle_l200_20072

-- Define the rectangular billiard table
structure BilliardTable where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define the ball's position and direction
structure BallState where
  x : ℝ
  y : ℝ
  angle : ℝ

-- Define the initial state of the ball
noncomputable def initial_state (table : BilliardTable) : BallState :=
  { x := 0, y := 0, angle := Real.pi/4 }

-- Define a function to check if the ball is at the middle of a side
def is_middle_of_side (table : BilliardTable) (state : BallState) : Prop :=
  (state.x = 0 ∧ state.y = table.width / 2) ∨
  (state.x = table.length ∧ state.y = table.width / 2) ∨
  (state.y = 0 ∧ state.x = table.length / 2) ∨
  (state.y = table.width ∧ state.x = table.length / 2)

-- Define a function to simulate the ball's movement
noncomputable def move_ball (table : BilliardTable) (initial : BallState) : BallState :=
  sorry -- Placeholder for the actual implementation

-- Theorem statement
theorem ball_cannot_hit_opposite_middle (table : BilliardTable) :
  let first_hit := move_ball table (initial_state table)
  is_middle_of_side table first_hit →
  ¬(is_middle_of_side table (move_ball table first_hit) ∧
    ((first_hit.x = 0 ∧ (move_ball table first_hit).x = table.length) ∨
    (first_hit.x = table.length ∧ (move_ball table first_hit).x = 0) ∨
    (first_hit.y = 0 ∧ (move_ball table first_hit).y = table.width) ∨
    (first_hit.y = table.width ∧ (move_ball table first_hit).y = 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_cannot_hit_opposite_middle_l200_20072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_168_l200_20029

/-- Represents a trapezoid ABCD with points P and Q on its sides -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  PQ : ℝ
  isIsosceles : BC = DA
  APequalsCQ : True  -- We can't directly represent AP = CQ without more complex definitions

/-- The area of the quadrilateral formed by the intersection of a circle and trapezoid -/
noncomputable def quadrilateralArea (t : Trapezoid) : ℝ :=
  ((t.CD - t.AB) / 2) * Real.sqrt (t.DA^2 - ((t.CD - t.AB) / 2)^2)

/-- Theorem stating the area of the quadrilateral is 168 under given conditions -/
theorem quadrilateral_area_is_168 (t : Trapezoid)
  (h1 : t.AB = 17)
  (h2 : t.BC = 25)
  (h3 : t.CD = 31)
  (h4 : t.PQ = 25) :
  quadrilateralArea t = 168 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_168_l200_20029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_proof_l200_20009

-- Define the friends as a custom type
inductive Friend
| Daniel
| Emily
| Fiona
deriving Repr, DecidableEq

-- Define a type for age rankings
def AgeRanking := List Friend

-- Define the statements
def Statement1 (ranking : AgeRanking) : Prop := ranking.head? = some Friend.Emily
def Statement2 (ranking : AgeRanking) : Prop := ranking.head? ≠ some Friend.Fiona
def Statement3 (ranking : AgeRanking) : Prop := ranking.getLast? ≠ some Friend.Daniel

-- Define the condition that exactly one statement is true
def ExactlyOneStatementTrue (ranking : AgeRanking) : Prop :=
  (Statement1 ranking ∧ ¬Statement2 ranking ∧ ¬Statement3 ranking) ∨
  (¬Statement1 ranking ∧ Statement2 ranking ∧ ¬Statement3 ranking) ∨
  (¬Statement1 ranking ∧ ¬Statement2 ranking ∧ Statement3 ranking)

-- Define the correct ranking
def CorrectRanking : AgeRanking := [Friend.Fiona, Friend.Daniel, Friend.Emily]

-- The theorem to prove
theorem correct_ranking_proof :
  ∀ (ranking : AgeRanking),
    (ranking.length = 3) →
    (ranking.toFinset.card = 3) →
    (ExactlyOneStatementTrue ranking) →
    (ranking = CorrectRanking) := by
  sorry

#eval CorrectRanking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_proof_l200_20009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l200_20074

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 2)

theorem f_range :
  (∀ x, 0 < f x ∧ f x ≤ 1/2) ∧
  (∀ y, 0 < y → y ≤ 1/2 → ∃ x, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l200_20074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrick_jogging_time_l200_20042

/-- The time taken for Patrick to jog from his house to Aaron's house -/
noncomputable def jogging_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that Patrick's jogging time is 2 hours -/
theorem patrick_jogging_time :
  let distance : ℝ := 14
  let speed : ℝ := 7
  jogging_time distance speed = 2 := by
  -- Unfold the definition of jogging_time
  unfold jogging_time
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrick_jogging_time_l200_20042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l200_20066

/-- Sequence c defined recursively -/
def c : ℕ → ℕ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | 2 => 9
  | (n + 3) => c (n + 2) * c (n + 1)

/-- Theorem stating that the 10th term of sequence c is 3^89 -/
theorem c_10_equals_3_pow_89 : c 10 = 3^89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l200_20066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_targets_hit_lower_bound_l200_20024

/-- The expected number of hit targets given n boys and n targets -/
noncomputable def E (n : ℕ) : ℝ := n * (1 - (1 - 1/n)^n)

/-- Theorem: The expected number of hit targets is always greater than or equal to n/2 -/
theorem expected_targets_hit_lower_bound (n : ℕ) (hn : n > 0) : E n ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_targets_hit_lower_bound_l200_20024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l200_20095

def sequenceQ (n : ℕ) : ℚ :=
  (1234567 : ℚ) / (3 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  (∀ n : ℕ, n < 2 → is_integer (sequenceQ n)) ∧
  (∀ n : ℕ, n ≥ 2 → ¬ is_integer (sequenceQ n)) ∧
  sequenceQ 1 = 137174 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l200_20095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cost_19_water_usage_84_l200_20010

/-- Water pricing function based on usage -/
noncomputable def water_price (usage : ℝ) : ℝ :=
  if usage ≤ 20 then 3 else 4

/-- Total water cost calculation -/
noncomputable def water_cost (usage : ℝ) : ℝ :=
  if usage ≤ 20 then usage * water_price usage
  else 20 * water_price 20 + (usage - 20) * water_price usage

/-- Theorem for water cost of 19m³ -/
theorem water_cost_19 : water_cost 19 = 57 := by sorry

/-- Theorem for water usage given cost of 84 yuan -/
theorem water_usage_84 : ∃ (usage : ℝ), water_cost usage = 84 ∧ usage = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cost_19_water_usage_84_l200_20010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l200_20040

/-- The function f(x) = (x-2)ln(x) --/
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

/-- The function f has exactly two zeros in its domain --/
theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ f a = 0 ∧ f b = 0 ∧ 
  (∀ x, x > 0 → f x = 0 → x = a ∨ x = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l200_20040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_not_divisible_by_seven_l200_20019

theorem max_subset_not_divisible_by_seven :
  ∃ (S : Finset ℕ),
    (∀ x, x ∈ S → x ≤ 50) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → ¬(7 ∣ x + y)) ∧
    S.card = 22 ∧
    ∀ (T : Finset ℕ),
      (∀ x, x ∈ T → x ≤ 50) →
      (∀ x y, x ∈ T → y ∈ T → x ≠ y → ¬(7 ∣ x + y)) →
      T.card ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_not_divisible_by_seven_l200_20019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l200_20068

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 24

-- Define point P
def P : ℝ × ℝ := (2, -3)

-- Define the line
def my_line (x y : ℝ) : Prop := x - y - 5 = 0

-- Theorem statement
theorem chord_equation :
  ∀ (A B : ℝ × ℝ),
  (∀ x y, my_circle x y → (x, y) ≠ A → (x, y) ≠ B → ¬(my_line x y)) →  -- A and B are on the circle
  (∃ t : ℝ, A = (2 + t, -3 + t) ∧ B = (2 - t, -3 - t)) →  -- P bisects AB
  (∀ x y, (x, y) = A ∨ (x, y) = B → my_line x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l200_20068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l200_20001

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 - Real.sqrt 2 * t, Real.sqrt 5 + Real.sqrt 2 * t)

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 5)^2 = 5

-- Define point P
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 5)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    (∃ t₁, line_l t₁ = A) ∧
    (∃ t₂, line_l t₂ = B) ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l200_20001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_is_five_l200_20043

/-- The problem setup for finding the optimal k value --/
structure PointConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ → ℝ × ℝ
  D : ℝ × ℝ

/-- The distance function between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The total distance function to be minimized --/
noncomputable def totalDistance (config : PointConfig) (k : ℝ) : ℝ :=
  distance config.A (config.C k) + distance config.B (config.C k)

/-- The theorem stating that k = 5 minimizes the total distance --/
theorem optimal_k_is_five (config : PointConfig) 
    (h1 : config.A = (4, 3))
    (h2 : config.B = (1, 2))
    (h3 : config.C = fun k => (0, k))
    (h4 : config.D = (0, 5))
    (h5 : ∀ k, (config.C k).1 = (config.D).1) :
    ∃ k, k = 5 ∧ ∀ x, totalDistance config k ≤ totalDistance config x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_is_five_l200_20043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_correct_matching_is_one_sixth_l200_20053

/-- The number of famous people and baby photos -/
def n : ℕ := 3

/-- The set of all possible matchings -/
def matchings : Finset (Fin n → Fin n) :=
  Finset.univ.filter Function.Bijective

/-- The correct matching -/
def correct_matching : Fin n → Fin n :=
  id

/-- The probability of a correct matching -/
noncomputable def prob_correct_matching : ℚ :=
  1 / matchings.card

/-- Theorem: The probability of correctly matching all baby photos is 1/6 -/
theorem prob_correct_matching_is_one_sixth :
  prob_correct_matching = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_correct_matching_is_one_sixth_l200_20053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_height_l200_20046

noncomputable section

variable (S : ℝ)
variable (π : ℝ)

-- Define the surface area of a cylinder
def surface_area (r h : ℝ) : ℝ := 2 * π * r * h + 2 * π * r^2

-- Define the volume of a cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- State the theorem
theorem cylinder_max_volume_height (hS : S > 0) (hπ : π > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  surface_area π r h = S ∧
  (∀ (r' h' : ℝ), r' > 0 → h' > 0 → surface_area π r' h' = S → volume π r h ≥ volume π r' h') ∧
  h = (Real.sqrt (6 * π * S)) / (3 * π) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_height_l200_20046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_bought_two_pounds_of_meat_l200_20051

/-- Represents the grocery items Lauren bought --/
structure GroceryItems where
  meatPricePerPound : ℚ
  bunsPrice : ℚ
  lettucePrice : ℚ
  tomatoWeight : ℚ
  tomatoPricePerPound : ℚ
  picklesPrice : ℚ
  picklesCoupon : ℚ

/-- Represents Lauren's purchase --/
structure Purchase where
  items : GroceryItems
  billPaid : ℚ
  changeReceived : ℚ

/-- Calculates the amount of meat Lauren bought --/
def meatBought (p : Purchase) : ℚ :=
  let totalSpent := p.billPaid - p.changeReceived
  let otherItemsCost := p.items.bunsPrice + p.items.lettucePrice + 
                        (p.items.tomatoWeight * p.items.tomatoPricePerPound) + 
                        (p.items.picklesPrice - p.items.picklesCoupon)
  let meatCost := totalSpent - otherItemsCost
  meatCost / p.items.meatPricePerPound

/-- Theorem stating that Lauren bought 2 pounds of meat --/
theorem lauren_bought_two_pounds_of_meat (p : Purchase) 
  (h1 : p.items.meatPricePerPound = 7/2)
  (h2 : p.items.bunsPrice = 3/2)
  (h3 : p.items.lettucePrice = 1)
  (h4 : p.items.tomatoWeight = 3/2)
  (h5 : p.items.tomatoPricePerPound = 2)
  (h6 : p.items.picklesPrice = 5/2)
  (h7 : p.items.picklesCoupon = 1)
  (h8 : p.billPaid = 20)
  (h9 : p.changeReceived = 6) :
  meatBought p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_bought_two_pounds_of_meat_l200_20051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_perpendicular_lines_l200_20007

noncomputable def perpendicular_slope (m : ℝ) : ℝ := -1 / m

noncomputable def line_equation (m x₀ y₀ x : ℝ) : ℝ := m * (x - x₀) + y₀

noncomputable def intersection_x (m₁ b₁ m₂ b₂ : ℝ) : ℝ := (b₂ - b₁) / (m₁ - m₂)

noncomputable def intersection_y (m₁ b₁ m₂ b₂ : ℝ) : ℝ := m₁ * intersection_x m₁ b₁ m₂ b₂ + b₁

theorem intersection_point_of_perpendicular_lines :
  let m₁ : ℝ := 3
  let b₁ : ℝ := 4
  let x₀ : ℝ := 3
  let y₀ : ℝ := 2
  let m₂ : ℝ := perpendicular_slope m₁
  let b₂ : ℝ := y₀ - m₂ * x₀
  intersection_x m₁ b₁ m₂ b₂ = 3/10 ∧
  intersection_y m₁ b₁ m₂ b₂ = 49/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_perpendicular_lines_l200_20007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_theorem_l200_20099

/-- The maximum value of the minimum distance between any three points in a rectangle --/
noncomputable def max_min_distance (a b : ℝ) : ℝ :=
  if a / b ≥ 2 / Real.sqrt 3
  then Real.sqrt (a^2 / 4 + b^2)
  else 2 * Real.sqrt (a^2 + b^2 - Real.sqrt 3 * a * b)

/-- Theorem stating the maximum value of the minimum distance between any three points in a rectangle --/
theorem max_min_distance_theorem (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  ∀ (X Y Z : ℝ × ℝ),
    X.1 ≥ 0 ∧ X.1 ≤ b ∧ X.2 ≥ 0 ∧ X.2 ≤ a ∧
    Y.1 ≥ 0 ∧ Y.1 ≤ b ∧ Y.2 ≥ 0 ∧ Y.2 ≤ a ∧
    Z.1 ≥ 0 ∧ Z.1 ≤ b ∧ Z.2 ≥ 0 ∧ Z.2 ≤ a →
  (min (dist X Y) (min (dist Y Z) (dist Z X))) ≤ max_min_distance a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_theorem_l200_20099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_ratio_l200_20041

theorem square_triangle_area_ratio (a n m : ℝ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 0 < m) :
  (1/2 * (a / n^2) * (a / 2)) / (1/2 * (a / m^2) * (a / 2)) = m^2 / n^2 := by
  -- Simplify the left side of the equation
  have h4 : (1/2 * (a / n^2) * (a / 2)) / (1/2 * (a / m^2) * (a / 2)) = (a^2 / (4 * n^2)) / (a^2 / (4 * m^2)) := by
    field_simp
    ring
  
  -- Simplify further
  have h5 : (a^2 / (4 * n^2)) / (a^2 / (4 * m^2)) = m^2 / n^2 := by
    field_simp
    ring
  
  -- Combine the steps
  rw [h4, h5]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_ratio_l200_20041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_collinearity_l200_20056

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 32 = 0

-- Define the line passing through (0, 2) with slope k
def line_eq (k x y : ℝ) : Prop := y = k*x + 2

-- Define the center of the circle
def center : ℝ × ℝ := (6, 0)

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem circle_line_intersection_and_collinearity :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2) ↔ 
  (-3/4 < k ∧ k < 0) ∧
  ¬∃ A B : ℝ × ℝ, 
    A ≠ B ∧ 
    circle_eq A.1 A.2 ∧ 
    circle_eq B.1 B.2 ∧ 
    line_eq k A.1 A.2 ∧ 
    line_eq k B.1 B.2 ∧ 
    ∃ t : ℝ, (A.1 + B.1, A.2 + B.2) = (t * (center.1 - P.1), t * (center.2 - P.2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_collinearity_l200_20056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l200_20020

theorem smallest_undefined_inverse : 
  (∀ b : ℕ, b < 11 → (∃ x : ℤ, x * b % 77 = 1 ∨ x * b % 88 = 1)) ∧
  (∀ x : ℤ, x * 11 % 77 ≠ 1) ∧
  (∀ x : ℤ, x * 11 % 88 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l200_20020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_and_integral_l200_20089

noncomputable def P (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

noncomputable def Q (p q r s : ℝ) (x : ℝ) : ℝ := 4 * p * x^3 + 3 * q * x^2 + 2 * r * x + s

theorem polynomial_root_and_integral (p q r s t : ℝ) :
  P p q r s t (Real.sqrt (-5)) = 0 →
  Q p q r s (Real.sqrt (-2)) = 0 →
  (∫ (x : ℝ) in Set.Icc 0 1, P p q r s t x) = -52/5 →
  (p, q, r, s, t) = (3, 0, 12, 0, -15) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_and_integral_l200_20089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l200_20047

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 10*x - 10) / ((x+1)*(x-2)^3)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := Real.log (abs (x + 1)) + 1 / (x - 2)^2

-- Theorem statement
theorem integral_equality (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) : 
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l200_20047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_equal_folds_l200_20075

/-- A fold is represented by an angle from 0 to π -/
def Fold := { θ : Real // 0 ≤ θ ∧ θ ≤ Real.pi }

/-- A square piece of paper -/
structure Square where
  side : ℝ
  side_pos : side > 0

/-- A folding divides the square into two parts -/
def divides_equally (f : Fold) (s : Square) : Prop :=
  ∃ (p1 p2 : Set (ℝ × ℝ)), 
    p1 ∪ p2 = { (x, y) | 0 ≤ x ∧ x ≤ s.side ∧ 0 ≤ y ∧ y ≤ s.side } ∧
    p1 ∩ p2 = ∅ ∧
    MeasureTheory.volume p1 = MeasureTheory.volume p2

/-- The theorem stating that there are infinitely many ways to fold a square into two equal parts -/
theorem infinite_equal_folds (s : Square) : 
  ∃ (F : Set Fold), (Set.Infinite F) ∧ (∀ f ∈ F, divides_equally f s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_equal_folds_l200_20075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l200_20006

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive angles
  a + b + c = 180 →        -- Sum of interior angles
  a / 4 = b / 5 ∧ b / 5 = c / 6 →  -- Ratio of angles
  max a (max b c) = 72 :=  -- Largest angle is 72 degrees
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l200_20006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_branch_diameter_is_correct_l200_20022

/-- The maximum diameter of a branch that can pass through a 90° turn in a channel of width 1 -/
noncomputable def max_branch_diameter : ℝ := 2 * Real.sqrt 2 + 2

/-- Theorem stating that the maximum diameter of a branch that can pass through a 90° turn
    in a channel of width 1 is equal to 2√2 + 2 -/
theorem max_branch_diameter_is_correct :
  ∀ d : ℝ, d > 0 →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    ∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
    x * (Real.cos θ) + y * (Real.sin θ) ≤ 1 ∧
    x * (Real.sin θ) - y * (Real.cos θ) + d ≤ 1) ↔
  d ≤ max_branch_diameter := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_branch_diameter_is_correct_l200_20022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l200_20003

open Real

theorem indefinite_integral_proof (x : ℝ) : 
  let f := λ x : ℝ ↦ -1 / (x + 3) + (1 / 2) * log (x^2 + 3) + (2 / Real.sqrt 3) * arctan (x / Real.sqrt 3)
  let g := λ x : ℝ ↦ (x^3 + 9*x^2 + 21*x + 21) / ((x + 3)^2 * (x^2 + 3))
  deriv f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l200_20003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initially_calculated_average_height_l200_20060

theorem initially_calculated_average_height 
  (n : ℕ) 
  (wrong_height correct_height : ℝ) 
  (actual_average : ℝ) 
  (initial_average : ℝ) :
  n = 35 ∧ 
  wrong_height = 166 ∧ 
  correct_height = 106 ∧ 
  actual_average = 182 ∧
  n * initial_average - (wrong_height - correct_height) = n * actual_average →
  abs (initial_average - 183.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initially_calculated_average_height_l200_20060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_median_l200_20088

/-- Given a triangle ABC with vertices A(1,-1), B(-2,1), and C(3,-5),
    prove that the equation x - y - 2 = 0 represents the perpendicular
    line from vertex A to the median drawn from vertex B. -/
theorem perpendicular_to_median (A B C : ℝ × ℝ) : 
  A = (1, -1) → B = (-2, 1) → C = (3, -5) →
  let M : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let median_slope : ℝ := (M.2 - B.2) / (M.1 - B.1)
  let perpendicular_slope : ℝ := -1 / median_slope
  let perpendicular_eq (x y : ℝ) : Prop := x - y - 2 = 0
  perpendicular_eq A.1 A.2 ∧ 
  (∀ x y : ℝ, perpendicular_eq x y → 
    (y - A.2) = perpendicular_slope * (x - A.1)) ∧
  perpendicular_slope * median_slope = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_median_l200_20088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_average_speed_l200_20071

-- Define the total distance
variable (S : ℝ)

-- Assumption that S is positive
axiom S_pos : S > 0

-- Define Jack's speeds
def jack_speed_up1 : ℝ := 4
def jack_speed_up2 : ℝ := 2
def jack_speed_down : ℝ := 3

-- Define the meeting point
noncomputable def meeting_point : ℝ := S / 2

-- Define Jack's total time
noncomputable def jack_total_time : ℝ := (S / (2 * jack_speed_up1)) + (S / (2 * jack_speed_up2)) + (S / (2 * jack_speed_down))

-- Theorem to prove
theorem jenny_average_speed :
  (meeting_point / jack_total_time) = 12 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_average_speed_l200_20071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l200_20038

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - Real.log x - 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one : 
  ∃ (m b : ℝ), m = 2 * Real.exp 1 - 1 ∧ b = -Real.exp 1 ∧
  ∀ x, f x = m * (x - 1) + f 1 := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  ∀ a : ℝ, (∀ x > 0, f x ≥ a * x) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l200_20038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_circles_are_intersecting_l200_20017

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (1, 0)
def radius1 : ℝ := 1

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (0, -2)
def radius2 : ℝ := 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem stating that the circles are intersecting
theorem circles_intersecting :
  radius2 - radius1 < distance_between_centers ∧
  distance_between_centers < radius1 + radius2 := by
  sorry

-- Proof that the circles are intersecting
theorem circles_are_intersecting : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersecting_circles_are_intersecting_l200_20017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_range_l200_20031

-- Define the points and the trajectory
def P : ℝ × ℝ := (2, 1)
def Q : ℝ × ℝ := (-2, -1)
def O : ℝ × ℝ := (0, 0)

def trajectory (x y : ℝ) : Prop :=
  x^2/8 + y^2/2 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define the conditions
def symmetric_points (p q : ℝ × ℝ) : Prop :=
  q.1 = -p.1 ∧ q.2 = -p.2

noncomputable def slope_product (p q m : ℝ × ℝ) : ℝ :=
  ((m.2 - p.2) / (m.1 - p.1)) * ((m.2 - q.2) / (m.1 - q.1))

-- Define the area of a triangle
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ :=
  abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)) / 2

-- Define the theorem
theorem trajectory_and_area_range :
  ∀ (M : ℝ × ℝ),
  symmetric_points P Q →
  slope_product P Q M = -1/4 →
  trajectory M.1 M.2 ∧
  ∀ (A : ℝ × ℝ),
  trajectory A.1 A.2 →
  A ≠ P →
  0 < area_triangle P A O ∧ area_triangle P A O ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_range_l200_20031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l200_20083

theorem sin_minus_cos_for_special_angle (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π/2)) (h2 : Real.tan θ = 1/3) :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l200_20083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l200_20049

/-- Represents a ray with an origin point and a direction vector. -/
structure Ray where
  origin : EuclideanSpace ℝ (Fin 3)
  direction : EuclideanSpace ℝ (Fin 3)

/-- Represents an angular region formed by two rays. -/
structure AngularRegion where
  ray1 : Ray
  ray2 : Ray

/-- Defines a triangle as three distinct points. -/
def Triangle (A B C : EuclideanSpace ℝ (Fin 3)) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

/-- Checks if a point is in an angular region. -/
def PointInAngularRegion (X : EuclideanSpace ℝ (Fin 3)) (region : AngularRegion) : Prop := sorry

/-- Checks if a point lies on a ray. -/
def PointOnRay (X : EuclideanSpace ℝ (Fin 3)) (r : Ray) : Prop := sorry

/-- Checks if a line passes through a point. -/
def LineThroughPoint (A B X : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

/-- Main theorem for triangle construction. -/
theorem triangle_construction 
  (P : EuclideanSpace ℝ (Fin 3)) 
  (a b c : Ray) 
  (X Y Z : EuclideanSpace ℝ (Fin 3)) 
  (hab : a.origin = P ∧ b.origin = P ∧ c.origin = P)
  (hX : PointInAngularRegion X ⟨b, c⟩)
  (hY : PointInAngularRegion Y ⟨c, a⟩)
  (hZ : PointInAngularRegion Z ⟨a, b⟩) :
  ∃ (A B C : EuclideanSpace ℝ (Fin 3)), 
    Triangle A B C ∧ 
    PointOnRay A a ∧ 
    PointOnRay B b ∧ 
    PointOnRay C c ∧
    LineThroughPoint B C X ∧
    LineThroughPoint C A Y ∧
    LineThroughPoint A B Z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l200_20049
