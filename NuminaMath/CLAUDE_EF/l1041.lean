import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_correct_cdf_l1041_104151

-- Define the probability density function (PDF)
noncomputable def p (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0
  else if -1 ≤ x ∧ x ≤ 0 then x + 1
  else -x + 1

-- Define the cumulative distribution function (CDF)
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ -1 then 0
  else if -1 < x ∧ x ≤ 0 then (x + 1)^2 / 2
  else if 0 < x ∧ x ≤ 1 then 1 - (1 - x)^2 / 2
  else 1

-- Theorem stating that F is the correct CDF for the given PDF p
theorem F_is_correct_cdf :
  ∀ x : ℝ, F x = ∫ t in Set.Iic x, p t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_correct_cdf_l1041_104151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l1041_104191

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 - y^2 = a^2 ∧ a > 0

-- Define the circle (renamed to avoid conflict)
def circle_eq (c x y : ℝ) : Prop := (x - c)^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - y = 0

-- Define the intersection point M
def point_M (a c x y : ℝ) : Prop :=
  hyperbola a x y ∧ circle_eq c x y ∧ line x y

-- Define the foci
noncomputable def focus_F1 (a : ℝ) : ℝ × ℝ := (-a * Real.sqrt 2, 0)
noncomputable def focus_F2 (a : ℝ) : ℝ × ℝ := (a * Real.sqrt 2, 0)

-- Define area_triangle (placeholder)
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle (a c : ℝ) (x y : ℝ) :
  hyperbola a x y →
  circle_eq c x y →
  line x y →
  point_M a c x y →
  let f1 := focus_F1 a
  let f2 := focus_F2 a
  let m := (x, y)
  area_triangle f1 m f2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l1041_104191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sides_of_specific_polygon_l1041_104162

/-- A polygon with equal sides and known perimeter and side length -/
structure RegularPolygon where
  perimeter : ℝ
  side_length : ℝ
  perimeter_positive : perimeter > 0
  side_length_positive : side_length > 0

/-- The number of sides in a regular polygon -/
noncomputable def num_sides (p : RegularPolygon) : ℝ := p.perimeter / p.side_length

theorem num_sides_of_specific_polygon :
  ∃ (p : RegularPolygon), p.perimeter = 80 ∧ p.side_length = 16 ∧ num_sides p = 5 := by
  sorry

#eval "The theorem has been stated and the proof is skipped with 'sorry'."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sides_of_specific_polygon_l1041_104162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l1041_104138

/-- The area of a rhombus given the lengths of its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (1 / 2) * d1 * d2

/-- Theorem: The area of a rhombus with diagonals of length 6 and 8 is 24 -/
theorem rhombus_area_example : rhombusArea 6 8 = 24 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the expression
  simp
  -- Check that the result is equal to 24
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l1041_104138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_added_cards_l1041_104177

theorem sasha_added_cards (original : ℕ) (final : ℕ) (removed_fraction : ℚ) :
  original = 43 →
  final = 83 →
  removed_fraction = 1 / 6 →
  ∃ (added : ℕ), 
    final = original + added - Int.floor (removed_fraction * added) ∧
    added = 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_added_cards_l1041_104177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_nine_fourths_l1041_104140

-- Define the function g
noncomputable def g (a b c : ℝ) : ℝ :=
  if a + b + c ≤ 5 then
    (a * b - a * c + c) / (2 * a - c)
  else
    (a * b - b * c - c) / (-2 * b + c)

-- State the theorem
theorem g_sum_equals_nine_fourths :
  g 3 2 0 + g 1 3 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_nine_fourths_l1041_104140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_pythagorean_partition_l1041_104133

def T (n : ℕ) : Set ℕ := {x | 2 ≤ x ∧ x ≤ n}

def has_pythagorean_triplet (S : Set ℕ) : Prop :=
  ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x^2 + y^2 = z^2

def valid_partition (n : ℕ) : Prop :=
  ∀ A B : Set ℕ, A ∪ B = T n → A ∩ B = ∅ →
    has_pythagorean_triplet A ∨ has_pythagorean_triplet B

theorem smallest_n_with_pythagorean_partition :
  (∀ n < 5, ¬ valid_partition n) ∧ valid_partition 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_pythagorean_partition_l1041_104133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1041_104179

noncomputable def a (n : ℕ) : ℝ := n + 1

noncomputable def b (n : ℕ) : ℝ := 2^n

noncomputable def S (n : ℕ) : ℝ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

noncomputable def c (n : ℕ) : ℝ := (a n) / (b n)

noncomputable def T (n : ℕ) : ℝ := 3 - (n + 3) / 2^n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 3 - (n + 3) / 2^n) ∧
  (a 1 = 2) ∧
  (b 1 = 2) ∧
  (S 5 = 5 * b 2) ∧
  (S 5 = a 11 + b 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1041_104179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_arrangement_l1041_104143

theorem crystal_arrangement (n : ℕ) (h : n = 5) : 
  Nat.factorial n - 2 * Nat.factorial (n - 1) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_arrangement_l1041_104143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1041_104139

theorem cosine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 Real.pi → 
  y ∈ Set.Icc 0 Real.pi → 
  (∀ t : Real, (-2 * Real.cos t - 1/2 * Real.cos x * Real.cos y) * Real.cos x * Real.cos y - 1 - Real.cos x + Real.cos y - Real.cos (2 * t) < 0) → 
  0 ≤ x ∧ x < y ∧ y ≤ Real.pi := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1041_104139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_ten_dividing_factorial_30_l1041_104109

theorem largest_power_of_ten_dividing_factorial_30 :
  (∃ n : ℕ, 7 < n ∧ (10 ^ n : ℕ) ∣ Nat.factorial 30) → False ∧
  (10 ^ 7 : ℕ) ∣ Nat.factorial 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_ten_dividing_factorial_30_l1041_104109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100k_l1041_104104

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 701 * x + 10000 / x - 9450
  else 0

noncomputable def selling_price : ℝ := 0.7

noncomputable def profit (x : ℝ) : ℝ :=
  selling_price * x * 1000 - (fixed_cost + variable_cost x)

theorem max_profit_at_100k :
  ∃ (max_profit : ℝ), max_profit = 9000 ∧
  ∀ (x : ℝ), x > 0 → profit x ≤ max_profit ∧
  profit 100 = max_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100k_l1041_104104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1041_104129

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem f_properties :
  (∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f 1 ∧ deriv f 1 = 0) ∧
  (∃ (max_val : ℝ) (max_point : ℝ), 
    max_val = 2 ∧ 
    max_point = -1 ∧ 
    f max_point = max_val ∧
    ∀ x, f x ≤ max_val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1041_104129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_quadratic_without_max_not_universal_l1041_104128

-- Define what a universal proposition is
def is_universal_proposition (P : Prop) : Prop :=
  ∀ (x : Type), P

-- Define a quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define what it means for a function to have no maximum value
def has_no_maximum (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f x > y

-- Statement to prove
theorem existence_of_quadratic_without_max_not_universal :
  ¬(is_universal_proposition (∃ a b c : ℝ, has_no_maximum (quadratic_function a b c))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_quadratic_without_max_not_universal_l1041_104128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1041_104107

open Real

/-- The infinite series defined in the problem -/
noncomputable def infiniteSeries (n : ℕ) : ℝ := 
  (n^4 + 5*n^2 + 8*n + 12) / (2^n * (n^4 + 9))

/-- The sum of the infinite series starting from n = 3 -/
noncomputable def seriesSum : ℝ := ∑' n, if n ≥ 3 then infiniteSeries n else 0

/-- Theorem stating that the sum of the infinite series is equal to 1/4 -/
theorem infiniteSeriesSum : seriesSum = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1041_104107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1041_104112

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem min_value_of_f :
  ∃ (x : ℝ), π / 4 ≤ x ∧ x ≤ π / 2 ∧
  ∀ (y : ℝ), π / 4 ≤ y ∧ y ≤ π / 2 → f x ≤ f y ∧ f x = 1 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1041_104112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_8_l1041_104146

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  first_term_eq : a 1 = a 1  -- Trivial condition to define a₁
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a 1 + (n - 1 : ℝ) * seq.d)

/-- Theorem stating that the sum is maximum at n = 8 -/
theorem max_sum_at_8 (seq : ArithmeticSequence) :
  S seq 16 > 0 → S seq 17 < 0 → ∀ n, S seq n ≤ S seq 8 := by
  sorry

#check max_sum_at_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_8_l1041_104146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progressions_with_same_exponents_are_proportional_l1041_104148

/-- Set of exponents for a positive integer -/
def setOfExponents (n : ℕ) : Finset ℕ :=
  sorry

/-- Arithmetic progression -/
def arithmeticProgression (a₀ d : ℕ) (n : ℕ) : ℕ :=
  a₀ + d * n

theorem arithmetic_progressions_with_same_exponents_are_proportional
  (a₀ d₁ b₀ d₂ : ℕ) :
  (∀ n : ℕ, setOfExponents (arithmeticProgression a₀ d₁ n) = setOfExponents (arithmeticProgression b₀ d₂ n)) →
  ∃ k : ℚ, ∀ n : ℕ, (arithmeticProgression a₀ d₁ n : ℚ) = k * (arithmeticProgression b₀ d₂ n) :=
by
  sorry

#check arithmetic_progressions_with_same_exponents_are_proportional

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progressions_with_same_exponents_are_proportional_l1041_104148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1041_104116

open Real

/-- A function satisfying the given functional equation and limit condition -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x > 0 → y > 0 → f x > 0 → f y > 0 → f (x * f y) = y * f x) ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ x > M, f x < ε)

/-- The main theorem stating that the only function satisfying the conditions is f(x) = 1/x -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, x > 0 → f x = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1041_104116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_A_is_direct_proportion_l1041_104124

/-- A function representing a direct proportion --/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- Function A: y = -0.1x --/
def f_A : ℝ → ℝ := fun x ↦ -0.1 * x

/-- Function B: y = 2x² --/
def f_B : ℝ → ℝ := fun x ↦ 2 * x^2

/-- Function C: y² = 4x --/
noncomputable def f_C : ℝ → ℝ := fun x ↦ Real.sqrt (4 * x)

/-- Function D: y = 2x + 1 --/
def f_D : ℝ → ℝ := fun x ↦ 2 * x + 1

/-- Theorem stating that only f_A is a direct proportion --/
theorem only_f_A_is_direct_proportion :
  DirectProportion f_A ∧
  ¬DirectProportion f_B ∧
  ¬DirectProportion f_C ∧
  ¬DirectProportion f_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_A_is_direct_proportion_l1041_104124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsidy_proof_l1041_104192

-- Define the subsidy for each year
noncomputable def subsidy (year : Nat) (a x : ℝ) : ℝ :=
  match year with
  | 2017 => 19.8 - a
  | 2018 => 19.8
  | 2019 => 19.8 + a
  | 2020 => (19.8 + a) * (1 + x)
  | 2021 => (19.8 + a) * (1 + x)^2
  | _ => 0

-- Define the total subsidy over 5 years
noncomputable def total_subsidy (a x : ℝ) : ℝ :=
  (subsidy 2017 a x) + (subsidy 2018 a x) + (subsidy 2019 a x) + (subsidy 2020 a x) + (subsidy 2021 a x)

-- Define the conditions
theorem subsidy_proof (a x : ℝ) :
  (subsidy 2019 a x ≥ 1.15 * subsidy 2018 a x) ∧
  (total_subsidy a x > 5.31 * subsidy 2018 a x + 2.31 * a) →
  (a ≥ 2.97 ∧ x = 0.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsidy_proof_l1041_104192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l1041_104193

noncomputable def my_sequence (n : ℕ+) : ℚ :=
  (-1)^(n : ℤ) * (2 * n.val + 1 : ℚ) / (n.val^2 + 1 : ℚ)

theorem tenth_term_of_sequence :
  my_sequence 10 = 21 / 101 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l1041_104193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equation_l1041_104164

/-- The angle bisector of a point in a triangle -/
def angle_bisector (P : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The line passing through two points -/
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Given a triangle ABC with points A(3, -1) and B(10, 5), and the angle bisector
    of angle B having the equation x - 4y + 10 = 0, prove that the line BC
    has the equation 2x + 9y - 65 = 0 -/
theorem triangle_side_equation (A B C : ℝ × ℝ) :
  A = (3, -1) →
  B = (10, 5) →
  (∀ x y : ℝ, (x - 4*y + 10 = 0) ↔ ((x, y) ∈ angle_bisector B)) →
  ∃ k m : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, (k*x + m*y - 65 = 0) ↔ ((x, y) ∈ line_through B C)) ∧
            k / m = 2 / 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equation_l1041_104164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldies_total_earnings_l1041_104158

/-- Represents Goldie's earnings for pet-sitting tasks over two weeks -/
def goldies_earnings (dw_hours dw_rate med_hours med_rate feed_hours feed_rate clean_hours clean_rate play_hours play_rate : ℕ) : ℕ :=
  dw_hours * dw_rate + med_hours * med_rate + 
  feed_hours * feed_rate + clean_hours * clean_rate + play_hours * play_rate

theorem goldies_total_earnings : 
  goldies_earnings 12 5 8 8 10 6 15 4 5 3 = 259 := by
  sorry

#eval goldies_earnings 12 5 8 8 10 6 15 4 5 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldies_total_earnings_l1041_104158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_number_l1041_104168

def skip_pattern (n : ℕ) : ℕ := 4 * n - 1

def student_number : ℕ → ℕ
  | 0 => 1  -- Base case for the first student (Alice)
  | n + 1 => skip_pattern (student_number n)

theorem last_student_number :
  ∃ (k : ℕ), k ≤ 7 ∧ student_number k = 239 ∧ student_number (k + 1) > 500 := by
  sorry

#eval student_number 7  -- This will evaluate the 7th student's number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_number_l1041_104168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_l1041_104106

theorem absolute_value_inequality_solution (x : ℝ) :
  (3 ≤ |x - 2| ∧ |x - 2| ≤ 6) ↔ (x ∈ Set.Icc (-4) (-1) ∪ Set.Icc 5 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_l1041_104106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_range_l1041_104130

/-- Given a line segment AB of length 4 and a point P such that |PA| + |PB| = 6,
    prove that the minimum value of |PA| is 1 and the maximum value of |PA| is 5 -/
theorem pa_range (A B P : ℝ × ℝ) : 
  let d := λ X Y : ℝ × ℝ ↦ Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let AB : ℝ := d A B
  let PA : ℝ := d P A
  let PB : ℝ := d P B
  AB = 4 →
  PA + PB = 6 →
  (∀ P', d P' A + d P' B = 6 → PA ≤ d P' A) ∧
  (∃ P', d P' A + d P' B = 6 ∧ d P' A = 1) ∧
  (∃ P'', d P'' A + d P'' B = 6 ∧ d P'' A = 5) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_range_l1041_104130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_are_good_infinitely_many_bad_numbers_thirteen_is_bad_l1041_104122

/-- A positive integer that can be represented as the arithmetic mean of some positive integers,
    each of which is a non-negative integer power of 2 (the integers can be the same). -/
def is_good_number (n : ℕ+) : Prop :=
  ∃ (k : ℕ) (S : Finset ℕ), k > 0 ∧ 
    (∀ i, i ∈ S → ∃ j : ℕ, i = 2^j) ∧
    (n : ℚ) = (S.sum (λ i => i)) / k

/-- A positive integer that cannot be represented as the arithmetic mean of some pairwise different
    positive integers, each of which is a non-negative integer power of 2. -/
def is_bad_number (n : ℕ+) : Prop :=
  ¬∃ (S : Finset ℕ), S.Nonempty ∧ 
    (∀ i, i ∈ S → ∃ j : ℕ, i = 2^j) ∧
    (∀ i j, i ∈ S → j ∈ S → i ≠ j → i ≠ j) ∧
    (n : ℚ) = (S.sum (λ i => i)) / S.card

theorem all_positive_integers_are_good :
  ∀ n : ℕ+, is_good_number n := by
  sorry

theorem infinitely_many_bad_numbers :
  ∃ f : ℕ → ℕ+, (∀ n, f n < f (n + 1)) ∧ (∀ n, is_bad_number (f n)) := by
  sorry

theorem thirteen_is_bad : is_bad_number 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_are_good_infinitely_many_bad_numbers_thirteen_is_bad_l1041_104122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_time_l1041_104105

/-- Represents the time (in days) it takes for a worker to complete the job alone -/
structure WorkerTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the portion of work completed by a worker in one day -/
noncomputable def work_rate (w : WorkerTime) : ℚ := 1 / w.days

theorem c_work_time (a b c : WorkerTime) 
  (ha : a.days = 24)
  (hb : b.days = 30)
  (total_time : ℚ)
  (htotal : total_time = 11)
  (c_left_early : ℚ)
  (hc_left : c_left_early = 4)
  (h_completion : work_rate a * total_time + work_rate b * total_time + 
                  work_rate c * (total_time - c_left_early) = 1) :
  c.days = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_time_l1041_104105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_inequality_l1041_104123

/-- A predicate stating that four points form a convex quadrilateral -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ C = (x₃, y₃) ∧ D = (x₄, y₄) ∧
    (x₂ - x₁) * (y₃ - y₂) > (y₂ - y₁) * (x₃ - x₂) ∧
    (x₃ - x₂) * (y₄ - y₃) > (y₃ - y₂) * (x₄ - x₃) ∧
    (x₄ - x₃) * (y₁ - y₄) > (y₄ - y₃) * (x₁ - x₄) ∧
    (x₁ - x₄) * (y₂ - y₁) > (y₁ - y₄) * (x₂ - x₁)

/-- Ptolemy's inequality for convex quadrilaterals -/
theorem ptolemy_inequality (A B C D : ℝ × ℝ) (h : ConvexQuadrilateral A B C D) :
  ‖A - C‖^2 + ‖B - D‖^2 ≤ ‖A - B‖^2 + ‖B - C‖^2 + ‖C - D‖^2 + ‖D - A‖^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_inequality_l1041_104123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_processing_volume_max_profit_l1041_104114

/-- Represents the monthly processing volume in tons -/
noncomputable def x : ℝ → ℝ := fun t => t

/-- Represents the monthly processing cost in yuan -/
noncomputable def y : ℝ → ℝ := fun t => 0.5 * t^2 - 200 * t + 80000

/-- Represents the average processing cost per ton -/
noncomputable def avg_cost : ℝ → ℝ := fun t => y t / x t

/-- Represents the monthly profit -/
noncomputable def profit : ℝ → ℝ := fun t => 100 * x t - y t

/-- The theorem stating the optimal processing volume and minimum average cost -/
theorem optimal_processing_volume :
  ∃ t : ℝ, 400 ≤ t ∧ t ≤ 600 ∧
  (∀ s, 400 ≤ s ∧ s ≤ 600 → avg_cost t ≤ avg_cost s) ∧
  avg_cost t = 200 ∧ t = 400 := by
  sorry

/-- The theorem stating the maximum profit (which is negative) -/
theorem max_profit :
  ∃ t : ℝ, 400 ≤ t ∧ t ≤ 600 ∧
  (∀ s, 400 ≤ s ∧ s ≤ 600 → profit s ≤ profit t) ∧
  profit t = -40000 ∧ t = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_processing_volume_max_profit_l1041_104114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1041_104174

theorem fixed_point_theorem (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_a : f a < a) (h_b : f b > b) :
  ∃ ξ ∈ Set.Ioo a b, f ξ = ξ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1041_104174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_predicted_weight_is_20_l1041_104115

/-- Linear regression equation for predicting weight based on age -/
def weight_prediction (age : ℝ) : ℝ := 3 * age + 5

/-- Ages of the 5 children -/
noncomputable def children_ages : List ℝ := [3, 4, 5, 6, 7]

/-- Average age of the children -/
noncomputable def average_age : ℝ := (children_ages.sum) / (children_ages.length : ℝ)

/-- Theorem stating that the average predicted weight is 20 kg -/
theorem average_predicted_weight_is_20 :
  weight_prediction average_age = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_predicted_weight_is_20_l1041_104115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equiv_complement_A_union_B_l1041_104187

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (Ⅰ)
theorem A_equiv : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := by sorry

-- Theorem for part (Ⅱ)
theorem complement_A_union_B : (Set.compl A) ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equiv_complement_A_union_B_l1041_104187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1041_104178

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (2 * x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3/2} :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1041_104178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_point_on_specific_line_l1041_104142

/-- A point (x, y) lies on a line passing through two points (x₁, y₁) and (x₂, y₂) if and only if
    the slope between (x, y) and (x₁, y₁) is equal to the slope between (x₂, y₂) and (x₁, y₁). -/
theorem point_on_line (x y x₁ y₁ x₂ y₂ : ℝ) :
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁) ↔ 
  ∃ t : ℝ, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁) :=
by sorry

/-- The point (1, 7) lies on the line passing through (0, 1) and (-6, 0). -/
theorem point_on_specific_line : 
  (7 - 1) * (-6 - 0) = (1 - 0) * (0 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_point_on_specific_line_l1041_104142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_in_SetA_g_in_SetA_iff_a_range_g_in_SetA_for_all_a_iff_b_eq_1_l1041_104136

-- Define set A
def SetA := {f : ℝ → ℝ | ∃ x₀, f (x₀ + 1) + f x₀ = f 1}

-- Define the power function f(x) = x^(-1)
noncomputable def f (x : ℝ) : ℝ := x⁻¹

-- Define the logarithmic function g(x) = lg((2^x + a) / b)
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := Real.log ((2^x + a) / b) / Real.log 2

-- Theorem 1: f belongs to set A
theorem f_in_SetA : f ∈ SetA := by sorry

-- Theorem 2: When b = 1, g belongs to set A iff 0 ≤ a < 2
theorem g_in_SetA_iff_a_range (a : ℝ) :
  g a 1 ∈ SetA ↔ 0 ≤ a ∧ a < 2 := by sorry

-- Theorem 3: g belongs to set A for all a ∈ (0, 2) iff b = 1
theorem g_in_SetA_for_all_a_iff_b_eq_1 (b : ℝ) :
  (∀ a ∈ Set.Ioo 0 2, g a b ∈ SetA) ↔ b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_in_SetA_g_in_SetA_iff_a_range_g_in_SetA_for_all_a_iff_b_eq_1_l1041_104136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_factorial_sum_last_two_digits_l1041_104183

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_last_two_digits (n : ℕ) : ℕ := (n % 100) / 10 + n % 10

def fibonacci_factorial_sum (n : ℕ) : ℕ :=
  (List.range n).map (λ i => factorial (fibonacci i)) |> List.sum

theorem fibonacci_factorial_sum_last_two_digits :
  sum_last_two_digits (last_two_digits (fibonacci_factorial_sum 18)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_factorial_sum_last_two_digits_l1041_104183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_club_committee_probability_l1041_104121

theorem science_club_committee_probability :
  (let total_members : ℕ := 30
   let boys : ℕ := 8
   let girls : ℕ := 22
   let committee_size : ℕ := 7
   let total_combinations := Nat.choose total_members committee_size
   let all_boys_combinations := Nat.choose boys committee_size
   let all_girls_combinations := Nat.choose girls committee_size
   (total_combinations - (all_boys_combinations + all_girls_combinations)) / total_combinations = 10179 / 11110) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_club_committee_probability_l1041_104121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l1041_104118

noncomputable def data : List ℝ := [82, 91, 89, 88, 90]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem data_variance : variance data = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l1041_104118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1041_104157

open Real

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 8 * cos θ / sin θ
  (ρ * cos θ, ρ * sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ((2 + t) / 2, t)

-- Define the intersection points
def intersection_points (t : ℝ) : Prop :=
  ∃ θ, curve_C θ = line_l t

-- Theorem statement
theorem chord_length :
  ∀ t₁ t₂ : ℝ, 
  intersection_points t₁ ∧ intersection_points t₂ ∧ t₁ ≠ t₂ →
  abs (t₁ - t₂) = 32/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1041_104157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cut_ratio_l1041_104195

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (Real.sqrt 3 / 36) * a^2 = (1 / (4 * Real.pi)) * b^2 →
  a / b = 3 * Real.sqrt Real.pi / Real.rpow 27 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cut_ratio_l1041_104195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l1041_104155

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_15th_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 9 = 4)
  (h2 : S seq 15 = 30) :
  seq.a 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l1041_104155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_theorem_l1041_104172

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the frequency of bus stops in minutes -/
def stop_frequency : ℕ := 20

/-- Represents the start time of the bus route in hours past midnight -/
def start_time : ℕ := 13

/-- Represents the end time of the bus route in hours past midnight -/
def end_time : ℕ := 18

/-- Calculates the total number of bus stops given the start time, end time, and stop frequency -/
def total_stops (start : ℕ) (end_time : ℕ) (frequency : ℕ) : ℕ :=
  ((end_time - start) * minutes_per_hour) / frequency + 1

/-- Theorem stating that the total number of bus stops is 16 -/
theorem bus_stops_theorem : total_stops start_time end_time stop_frequency = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_theorem_l1041_104172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1041_104196

-- Define the function f(x) = a^(x-1) + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 1

-- Theorem statement
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  f a 1 = 1 ∧ f a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1041_104196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_theorem_l1041_104125

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid -/
noncomputable def surface_area_ratio (p q r : ℝ) : ℝ :=
  3 / (p * q + p * r + q * r)

/-- Theorem stating the ratio of surface areas -/
theorem surface_area_ratio_theorem (s p q r : ℝ) (hs : s > 0) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (6 * s^2) / (2 * s^2 * (p * q + p * r + q * r)) = surface_area_ratio p q r := by
  sorry

#check surface_area_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_theorem_l1041_104125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l1041_104169

noncomputable def y (x : ℝ) : ℝ := Real.exp x * (4 - 2 * x)

theorem particular_solution (x : ℝ) :
  (((deriv (deriv y)) x - 2 * (deriv y x) + y x = 0) ∧
  (y 0 = 4) ∧
  ((deriv y) 0 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l1041_104169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_coincide_l1041_104101

/-- Given a triangle ABC and points P, Q, R on its sides that divide the sides in the same ratio,
    the centroids of ABC and PQR coincide. -/
theorem centroids_coincide (A B C P Q R : ℝ × ℝ) (k : ℝ) :
  k > 0 ∧
  P = ((k * B.1 + A.1) / (k + 1), (k * B.2 + A.2) / (k + 1)) ∧
  Q = ((k * C.1 + B.1) / (k + 1), (k * C.2 + B.2) / (k + 1)) ∧
  R = ((k * A.1 + C.1) / (k + 1), (k * A.2 + C.2) / (k + 1)) →
  (A.1 + B.1 + C.1) / 3 = (P.1 + Q.1 + R.1) / 3 ∧
  (A.2 + B.2 + C.2) / 3 = (P.2 + Q.2 + R.2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_coincide_l1041_104101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1041_104147

/-- Triangle ABC with circumradius R, inradius r, and area S -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real
  r : Real
  S : Real

/-- Properties of the triangle -/
def TriangleProperties (t : Triangle) : Prop :=
  t.r = 3 ∧
  t.S = 6 ∧
  t.a * Real.cos t.A + t.b * Real.cos t.B + t.c * Real.cos t.C = t.R / 3

theorem triangle_theorem (t : Triangle) (h : TriangleProperties t) :
  (t.a + t.b + t.c = 4) ∧
  (t.R = 6) ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C) = 1 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1041_104147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l1041_104166

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The surface areas of the box faces -/
noncomputable def surface_areas (d : BoxDimensions) : Finset ℝ :=
  {d.A * d.B, d.A * d.C, d.B * d.C}

/-- The theorem stating the sum of dimensions given the surface areas -/
theorem box_dimensions_sum 
  (d : BoxDimensions) 
  (h1 : d.A > 0 ∧ d.B > 0 ∧ d.C > 0)
  (h2 : surface_areas d = {40, 90, 100}) : 
  d.A + d.B + d.C = 83 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l1041_104166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_driving_hours_l1041_104108

theorem jessica_driving_hours 
  (trip_duration : ℝ)
  (trips_per_day : ℝ)
  (school_days : ℕ)
  (h1 : trip_duration = 20 / 60)  -- 20 minutes converted to hours
  (h2 : trips_per_day = 2)        -- to and from school
  (h3 : school_days = 75)         -- number of school days to meet requirement
  : trip_duration * trips_per_day * (school_days : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_driving_hours_l1041_104108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1041_104186

-- Define the points
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (1, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 1, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 1)

-- Define the planes
def plane_OAB (p : ℝ × ℝ × ℝ) : ℝ := p.1
def plane_OBC (p : ℝ × ℝ × ℝ) : ℝ := p.2.1
def plane_OAC (p : ℝ × ℝ × ℝ) : ℝ := p.2.2
def plane_ABC (p : ℝ × ℝ × ℝ) : ℝ := p.1 + p.2.1 + p.2.2 - 1

-- Define the distance function
def distance_to_plane (p : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) : ℝ :=
  abs (plane p)

-- Define the equidistant property
def is_equidistant (p : ℝ × ℝ × ℝ) : Prop :=
  distance_to_plane p plane_OAB = distance_to_plane p plane_OBC ∧
  distance_to_plane p plane_OAB = distance_to_plane p plane_OAC ∧
  distance_to_plane p plane_OAB = distance_to_plane p plane_ABC / Real.sqrt 3

-- Theorem statement
theorem equidistant_points_count :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 5 ∧ ∀ p ∈ s, is_equidistant p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1041_104186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mozi_satellite_implications_l1041_104167

-- Define the concept of a quantum satellite
structure QuantumSatellite where
  name : String
  orbit_altitude : ℕ
  is_sun_synchronous : Bool
  can_communicate_with_ground : Bool

-- Define the concept of quantum communication
def quantum_communication (satellite : QuantumSatellite) : Prop :=
  satellite.can_communicate_with_ground = true

-- Define the concept of respecting objective laws and exerting subjective initiative
axiom respect_laws_and_exert_initiative : Prop

-- Define the concept of matter as unity of absolute motion and relative stillness
axiom matter_motion_stillness_unity : Prop

-- Theorem statement
theorem mozi_satellite_implications 
  (mozi : QuantumSatellite)
  (h1 : mozi.name = "Mozi")
  (h2 : mozi.orbit_altitude = 500)
  (h3 : mozi.is_sun_synchronous = true)
  (h4 : quantum_communication mozi) :
  respect_laws_and_exert_initiative ∧ matter_motion_stillness_unity := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mozi_satellite_implications_l1041_104167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1041_104170

theorem evaluate_expression : (125 : ℝ) ^ (1/3 : ℝ) * 8 ^ (1/3 : ℝ) / 32 ^ (-(1/5) : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1041_104170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_grass_area_is_100pi_l1041_104171

/-- Represents a circular grass plot with a tangential gravel path -/
structure GrassPlot where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the remaining grass area of a circular plot with a tangential path -/
noncomputable def remainingGrassArea (plot : GrassPlot) : ℝ :=
  Real.pi * (plot.diameter / 2) ^ 2

/-- Theorem stating that for a circular grass plot with diameter 20 feet and a 
    tangential gravel path of 4 feet wide, the remaining grass area is 100π square feet -/
theorem remaining_grass_area_is_100pi :
  let plot : GrassPlot := { diameter := 20, pathWidth := 4 }
  remainingGrassArea plot = 100 * Real.pi := by
  sorry

-- This eval is removed as it's not computable
-- #eval remainingGrassArea { diameter := 20, pathWidth := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_grass_area_is_100pi_l1041_104171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_intersection_area_sum_l1041_104145

/-- Represents a regular octahedron --/
structure Octahedron :=
  (side_length : ℝ)

/-- Represents the intersection polygon --/
structure IntersectionPolygon :=
  (area : ℝ)

/-- Checks if two numbers are relatively prime --/
def are_relatively_prime (a c : ℕ) : Prop :=
  Nat.Coprime a c

/-- Checks if a number is not divisible by the square of any prime --/
def not_divisible_by_prime_square (b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ b)

/-- The main theorem --/
theorem octahedron_intersection_area_sum 
  (oct : Octahedron) 
  (intersection : IntersectionPolygon) 
  (a b c : ℕ) :
  oct.side_length = 2 →
  intersection.area = (a : ℝ) * Real.sqrt (b : ℝ) / c →
  are_relatively_prime a c →
  not_divisible_by_prime_square b →
  a + b + c = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_intersection_area_sum_l1041_104145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1041_104173

-- Define the ellipse C
noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

-- Define the theorem
theorem ellipse_properties :
  ∀ a b : ℝ, a > b ∧ b > 0 →
  eccentricity a b = Real.sqrt 2 / 2 →
  ellipse_C a b 2 (Real.sqrt 2) →
  (∀ x y : ℝ, ellipse_C a b x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ k b : ℝ, k ≠ 0 → b ≠ 0 →
    ∃ x₁ y₁ x₂ y₂ : ℝ,
      ellipse_C a b x₁ y₁ ∧
      ellipse_C a b x₂ y₂ ∧
      y₁ = k * x₁ + b ∧
      y₂ = k * x₂ + b ∧
      x₁ ≠ x₂ →
      let xₘ := (x₁ + x₂) / 2
      let yₘ := (y₁ + y₂) / 2
      (yₘ / xₘ) * k = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1041_104173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_length_l1041_104137

-- Define the function f(x) as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

-- State the theorem
theorem tan_intersection_length (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ x1 x2 : ℝ, x1 < x2 ∧ f ω x1 = 1 ∧ f ω x2 = 1 ∧ x2 - x1 = π / 3) : 
  ω = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_length_l1041_104137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_l1041_104197

-- Define the equation (noncomputable due to use of absolute value and sqrt)
noncomputable def f (x a : ℝ) : ℝ := x + abs x - 4 * Real.sqrt (a * (x - 3) + 2)

-- Theorem statement
theorem two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x a = 0 ∧ f y a = 0) ↔ a < 1 ∨ a > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_l1041_104197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1041_104126

noncomputable def v1 : ℝ × ℝ × ℝ := (2, 4, -1)
noncomputable def v2 : ℝ × ℝ × ℝ := (-1, 1, 3)

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := v
  Real.sqrt (x^2 + y^2 + z^2)

theorem parallelogram_area : 
  magnitude (cross_product v1 v2) = Real.sqrt 230 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1041_104126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_max_disjoint_pairs_correct_l1041_104141

/-- 
Given a positive integer n, this function returns the maximum number of disjoint pairs 
of elements from the set {1, 2, ..., n} such that the sums of different pairs 
are different integers not exceeding n.
-/
def maxDisjointPairs (n : ℕ) : ℕ :=
  (2 * n - 1) / 5

/-- 
Theorem stating that maxDisjointPairs gives the correct maximum number of disjoint pairs
for any positive integer n.
-/
theorem max_disjoint_pairs_correct (n : ℕ) (hn : n ≥ 1) :
  ∀ (pairs : List (Fin n × Fin n)),
    (∀ (p q : Fin n × Fin n), p ∈ pairs → q ∈ pairs → p ≠ q → 
      (p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2)) →
    (∀ (p q : Fin n × Fin n), p ∈ pairs → q ∈ pairs → p ≠ q → 
      (p.1.val + p.2.val : ℕ) ≠ (q.1.val + q.2.val : ℕ)) →
    (∀ (p : Fin n × Fin n), p ∈ pairs → (p.1.val + p.2.val : ℕ) ≤ n) →
    pairs.length ≤ maxDisjointPairs n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_max_disjoint_pairs_correct_l1041_104141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_price_l1041_104159

theorem student_ticket_price 
  (adult_price : ℚ)
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (student_tickets : ℕ)
  (h1 : adult_price = 4)
  (h2 : total_tickets = 59)
  (h3 : total_revenue = 222.5)
  (h4 : student_tickets = 9) :
  (total_revenue - ((total_tickets - student_tickets : ℚ) * adult_price)) / (student_tickets : ℚ) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_price_l1041_104159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_difference_l1041_104120

theorem geometric_arithmetic_difference 
  (geom_first : ℝ) (geom_ratio : ℝ) (arith_terms : ℕ) 
  (h1 : geom_first > 0) 
  (h2 : 0 < geom_ratio) 
  (h3 : geom_ratio < 1) 
  (h4 : arith_terms > 0) :
  (geom_first / (1 - geom_ratio)) - 
  (arith_terms * (2 * geom_first + (arith_terms - 1) * 1) / 2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_difference_l1041_104120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_switch_l1041_104111

/-- Represents a chessboard with queens -/
structure Chessboard :=
  (black_queens : Fin 8 → Nat)
  (white_queens : Fin 8 → Nat)

/-- Represents a move of a queen -/
inductive Move
  | Black : Fin 8 → Nat → Move
  | White : Fin 8 → Nat → Move

/-- Checks if the queens have switched places -/
def switched_places (initial : Chessboard) (final : Chessboard) : Prop :=
  ∀ i, initial.black_queens i = final.white_queens i ∧
       initial.white_queens i = final.black_queens i

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Checks if a move sequence is valid (alternating moves) -/
def valid_sequence : MoveSequence → Prop
  | [] => True
  | [_] => True
  | Move.Black _ _ :: Move.White _ _ :: rest => valid_sequence rest
  | Move.White _ _ :: Move.Black _ _ :: rest => valid_sequence rest
  | _ => False

/-- Applies a single move to a chessboard -/
def apply_move : Chessboard → Move → Chessboard
  | board, Move.Black i new_pos => { board with black_queens := Function.update board.black_queens i new_pos }
  | board, Move.White i new_pos => { board with white_queens := Function.update board.white_queens i new_pos }

/-- Applies a sequence of moves to a chessboard -/
def apply_moves : Chessboard → MoveSequence → Chessboard
  | board, [] => board
  | board, move :: rest => apply_moves (apply_move board move) rest

/-- The main theorem -/
theorem min_moves_to_switch (initial : Chessboard) :
  ∃ (moves : MoveSequence),
    valid_sequence moves ∧
    switched_places initial (apply_moves initial moves) ∧
    moves.length = 23 ∧
    (∀ (other_moves : MoveSequence),
      valid_sequence other_moves →
      switched_places initial (apply_moves initial other_moves) →
      other_moves.length ≥ 23) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_switch_l1041_104111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_meet_at_start_l1041_104198

/-- Represents a person walking around a square loop -/
structure Walker where
  speed : ℚ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The square loop -/
def loopSize : ℕ := 24

/-- Time taken for walkers to meet -/
noncomputable def meetingTime (w1 w2 : Walker) : ℚ :=
  (loopSize : ℚ) / (w1.speed + w2.speed)

/-- Distance walked by a walker until meeting -/
def distanceWalked (w : Walker) (t : ℚ) : ℚ :=
  w.speed * t

theorem walkers_meet_at_start (jane hector : Walker) 
    (h1 : jane.speed = 3 * hector.speed)
    (h2 : jane.direction ≠ hector.direction) :
  distanceWalked jane (meetingTime jane hector) % (loopSize : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_meet_at_start_l1041_104198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_diff_140_l1041_104188

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def goldbach_pair (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = n

theorem largest_prime_diff_140 :
  ∃ p q : ℕ,
    goldbach_pair 140 ∧
    is_prime p ∧ is_prime q ∧
    p + q = 140 ∧
    p ≠ q ∧
    ∀ p' q' : ℕ, goldbach_pair 140 → is_prime p' → is_prime q' → p' + q' = 140 → p' ≠ q' →
      (q - p : ℤ).natAbs ≥ (q' - p' : ℤ).natAbs ∧
      (q - p : ℤ).natAbs = 112 :=
by
  sorry

#check largest_prime_diff_140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_diff_140_l1041_104188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_l_equation_line_l_prime_exists_l1041_104110

-- Define the circle C
noncomputable def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}

-- Define the tangent line
noncomputable def tangent_line : Set (ℝ × ℝ) :=
  {p | 3 * p.1 - 4 * p.2 = 0}

-- Define line l (parametric form)
noncomputable def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (t, k * t)}

-- Define point P as a function of k
noncomputable def point_P (k : ℝ) : ℝ × ℝ :=
  ((4 * k + 2) / (k^2 + 1), (4 * k^2 + 2 * k) / (k^2 + 1))

-- Define point Q as a function of k and m
noncomputable def point_Q (k m : ℝ) : ℝ × ℝ :=
  (1 / (m - k), k / (m - k))

-- Define line l'
noncomputable def line_l_prime : Set (ℝ × ℝ) :=
  {p | p.1 + 2 * p.2 + 2 = 0}

theorem circle_equation :
  circle_C = {p | (p.1 - 2)^2 + (p.2 - 4)^2 = 4} := by
  sorry

theorem line_l_equation (k : ℝ) (h : k = 1 ∨ k = 7) :
  line_l k = {p | p.2 = k * p.1} := by
  sorry

theorem line_l_prime_exists :
  ∃ k : ℝ, k > 3/4 ∧
    (∃ Q : ℝ × ℝ, Q ∈ line_l k ∧ Q ∈ line_l_prime ∧
      ‖(point_P k : ℝ × ℝ)‖ * ‖(Q : ℝ × ℝ)‖ = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_l_equation_line_l_prime_exists_l1041_104110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_range_l1041_104163

theorem sin_sqrt3_range (x : Real) (h : π/4 ≤ x ∧ x ≤ π/2) :
  ∃ y : Real, y = (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ∧ 1 ≤ y ∧ y ≤ 3/2 ∧
  ∀ z : Real, (∃ w : Real, π/4 ≤ w ∧ w ≤ π/2 ∧ z = (Real.sin w)^2 + Real.sqrt 3 * Real.sin w * Real.cos w) →
  1 ≤ z ∧ z ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_range_l1041_104163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_6x_mod_9_l1041_104199

theorem remainder_of_6x_mod_9 (x : ℕ) (h1 : x > 0) (h2 : x % 9 = 5) : (6 * x) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_6x_mod_9_l1041_104199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1041_104135

/-- Calculates the speed of a train given its length, the time it takes to pass a person
    running in the opposite direction, and the person's speed. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / passing_time * 3.6 - person_speed

/-- Theorem stating that a train of length 120 m passing a person running at 6 kmph
    in the opposite direction in 6 seconds has a speed of 66 kmph. -/
theorem train_speed_problem : train_speed 120 6 6 = 66 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1041_104135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l1041_104176

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ

/-- Applies a dilation transformation to a point -/
def dilate (p : Point) (center : Point) (scale : ℝ) : Point :=
  { x := center.x + scale * (p.x - center.x),
    y := center.y + scale * (p.y - center.y) }

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Helper function to get the set of vertices of a square -/
def set_of_vertices (s : Square) : Set Point :=
  sorry -- Implementation details omitted for brevity

/-- Theorem: The farthest vertex of the dilated square -/
theorem farthest_vertex_of_dilated_square (s : Square) 
  (h1 : s.center = { x := 5, y := -3 })
  (h2 : s.area = 9)
  (h3 : ∃ (v : Point), v ∈ set_of_vertices s ∧ 
       ∀ (u : Point), u ∈ set_of_vertices s → 
       distance (dilate v { x := 0, y := 0 } 3) { x := 0, y := 0 } ≥ 
       distance (dilate u { x := 0, y := 0 } 3) { x := 0, y := 0 }) :
  ∃ (v : Point), v ∈ set_of_vertices s ∧ 
  dilate v { x := 0, y := 0 } 3 = { x := 19.5, y := -13.5 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l1041_104176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_unit_problem_l1041_104113

theorem fractional_unit_problem :
  ∃ (fractional_unit : ℚ) (original_fraction : ℚ) (units_to_add : ℕ) (smallest_prime : ℕ),
    -- The fractional unit of 5/6 is 1/6
    fractional_unit * 5 = original_fraction ∧
    fractional_unit = 1 / 6 ∧
    original_fraction = 5 / 6 ∧
    -- The smallest prime number is 2
    smallest_prime = 2 ∧
    units_to_add = 7 ∧
    -- Adding 7 of these fractional units to 5/6 results in the smallest prime number
    original_fraction + fractional_unit * units_to_add = smallest_prime := by
  -- Provide existence of the required values
  use 1/6, 5/6, 7, 2
  -- Split the goal into individual components
  apply And.intro
  · -- Prove fractional_unit * 5 = original_fraction
    norm_num
  · apply And.intro
    · -- Prove fractional_unit = 1/6
      rfl
    · apply And.intro
      · -- Prove original_fraction = 5/6
        rfl
      · apply And.intro
        · -- Prove smallest_prime = 2
          rfl
        · apply And.intro
          · -- Prove units_to_add = 7
            rfl
          · -- Prove original_fraction + fractional_unit * units_to_add = smallest_prime
            norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_unit_problem_l1041_104113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_star_area_ratio_l1041_104149

theorem hexagonal_star_area_ratio (r : ℝ) (h : r = 3) :
  let circle_area := π * r^2
  let hexagon_side := r
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side^2
  hexagon_area / circle_area = 3 * Real.sqrt 3 / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_star_area_ratio_l1041_104149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_minus_T_l1041_104153

open BigOperators

def S : ℚ := ∑ n in Finset.range 25, (2^(2*n)) / ((2*n+1) * (2*n+3))

def T : ℚ := ∑ n in Finset.range 25, (2^(2*n)) / (2*n+3)

theorem S_minus_T : S - T = 1 - 2^49 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_minus_T_l1041_104153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_empty_time_l1041_104175

/-- Represents the time it takes for pipe B to empty the tank -/
def empty_time (t : ℝ) : Prop := t > 0

/-- Pipe A fills the tank at a rate of 1/8 tank per minute -/
noncomputable def fill_rate_A : ℝ := 1 / 8

/-- Both pipes are open for 66 minutes -/
def both_pipes_time : ℝ := 66

/-- Only pipe A is open for the remaining time -/
def remaining_time : ℝ := 30 - 66

/-- The tank is completely filled after 30 minutes -/
axiom tank_filled : ∀ t : ℝ, empty_time t → 
  (fill_rate_A - 1 / t) * both_pipes_time + fill_rate_A * remaining_time = 1

theorem pipe_B_empty_time : empty_time 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_empty_time_l1041_104175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_constant_area_l1041_104185

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  ((x - 1)^2 + y^2).sqrt / |x - 4| = 1/2

-- Define the ellipse that contains point P
def ellipse (x y : ℝ) : Prop := x^2/12 + y^2/9 = 1

-- Define vector equality
def vector_equality (xa ya xb yb xd yd xe ye : ℝ) : Prop :=
  (xb - xa, yb - ya) = (2 * (xe - xd), 2 * (ye - yd))

-- Helper function for triangle area
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  |((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))| / 2

theorem curve_equation_and_constant_area 
  (xa ya xb yb xd yd xe ye xp yp : ℝ) :
  (∀ x y, distance_ratio x y ↔ C x y) ∧
  (C xa ya ∧ C xb yb ∧ C xd yd ∧ C xe ye) ∧
  vector_equality xa ya xb yb xd yd xe ye ∧
  ellipse xp yp →
  (∀ x y, C x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ k, ∀ xa ya xb yb xp yp,
    C xa ya ∧ C xb yb ∧ ellipse xp yp →
    area_triangle xp yp xa ya xb yb = k) ∧
  (∀ xa ya xb yb xp yp,
    C xa ya ∧ C xb yb ∧ ellipse xp yp →
    area_triangle xp yp xa ya xb yb = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_constant_area_l1041_104185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_intercept_sum_l1041_104127

-- Define the slope of a line ax + by + c = 0
noncomputable def line_slope (a b : ℚ) : ℚ := -a / b

-- Define the x-intercept of a line ax + by + c = 0
noncomputable def x_intercept (a c : ℚ) : ℚ := -c / a

-- Define the y-intercept of a line ax + by + c = 0
noncomputable def y_intercept (b c : ℚ) : ℚ := -c / b

-- Theorem statement
theorem parallel_line_with_intercept_sum :
  line_slope 2 3 = line_slope 10 15 ∧
  x_intercept 10 (-36) + y_intercept 15 (-36) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_intercept_sum_l1041_104127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1041_104154

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 + 1)
def domain_f_x2_plus_1 : Set ℝ := Set.Icc (-1) 1

-- Define the domain of f(lg x)
def domain_f_lg_x : Set ℝ := Set.Icc 10 100

-- State the theorem
theorem domain_equivalence :
  (∀ x ∈ domain_f_x2_plus_1, f (x^2 + 1) ∈ Set.range f) →
  (∀ x ∈ domain_f_lg_x, f (Real.log x / Real.log 10) ∈ Set.range f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1041_104154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l1041_104102

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine rule for triangles -/
noncomputable def cosine_rule (t : Triangle) : ℝ :=
  t.a^2 + t.c^2 - 2 * t.a * t.c * Real.cos t.B

/-- The area of a triangle using two sides and the included angle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.a * t.c * Real.sin t.B

/-- Theorem about the side length and area of a specific triangle -/
theorem triangle_side_and_area (t : Triangle) 
  (ha : t.a = 4) 
  (hc : t.c = 3) 
  (hB : Real.cos t.B = 1/8) :
  t.b = Real.sqrt 22 ∧ triangle_area t = 9 * Real.sqrt 7 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l1041_104102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_x_axis_l1041_104131

-- Define a point in a 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the reflection of a point across the x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem reflection_across_x_axis :
  let A : Point := (2, 3)
  reflect_x A = (2, -3) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_x_axis_l1041_104131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1041_104119

noncomputable def a : ℝ := Real.log 27 / Real.log 8
noncomputable def b : ℝ := Real.log 49 / Real.log 25

theorem problem_solution : (6 : ℝ)^(a/b) + (7 : ℝ)^(b/a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1041_104119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l1041_104194

theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total = 24 →
  boys = 14 →
  girls = 10 →
  committee_size = 5 →
  (Nat.choose total committee_size - (Nat.choose boys committee_size + Nat.choose girls committee_size) : ℚ) /
    Nat.choose total committee_size = 4025 / 4251 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l1041_104194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_distance_l1041_104165

/-- An ellipse with semi-major axis 5 and semi-minor axis 3 -/
structure Ellipse :=
  (x : ℝ) (y : ℝ)
  (eq : x^2 / 25 + y^2 / 9 = 1)

/-- The foci of the ellipse -/
noncomputable def foci : ℝ × ℝ := (4, -4)

/-- A point P on the ellipse such that PF₁ ⊥ PF₂ -/
structure SpecialPoint (e : Ellipse) :=
  (p : ℝ × ℝ)
  (on_ellipse : p.1^2 / 25 + p.2^2 / 9 = 1)
  (perpendicular : let (f₁, f₂) := foci
                   (p.1 - f₁) * (p.1 - f₂) + (p.2 - 0) * (p.2 - 0) = 0)

/-- The foot of the perpendicular from P to F₁F₂ -/
noncomputable def H (e : Ellipse) (sp : SpecialPoint e) : ℝ :=
  let (f₁, f₂) := foci
  let (px, py) := sp.p
  (px * (f₂ - f₁) + py * 0) / (f₂ - f₁)

/-- The theorem to be proved -/
theorem special_point_distance (e : Ellipse) (sp : SpecialPoint e) :
  |sp.p.1 - H e sp| = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_distance_l1041_104165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_natural_l1041_104160

/-- A geometric progression with natural number terms at positions 1, 10, and 30 -/
structure GeometricProgressionWithNaturalTerms where
  a : ℕ → ℚ
  q : ℚ
  is_gp : ∀ n, a (n + 1) = a n * q
  a1_nat : ∃ (n : ℕ), a 1 = n
  a10_nat : ∃ (n : ℕ), a 10 = n
  a30_nat : ∃ (n : ℕ), a 30 = n

/-- The 20th term of a geometric progression with natural number terms at positions 1, 10, and 30 is also a natural number -/
theorem twentieth_term_is_natural (gp : GeometricProgressionWithNaturalTerms) : ∃ (n : ℕ), gp.a 20 = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_natural_l1041_104160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l1041_104134

/-- A function that rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- Definition of the repeating decimal 37.363636... -/
noncomputable def repeating_decimal : ℝ := 37 + (36 : ℝ) / 99

/-- Theorem stating that rounding the repeating decimal to the nearest hundredth gives 37.37 -/
theorem round_repeating_decimal_to_hundredth :
  round_to_hundredth repeating_decimal = 37.37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l1041_104134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_depth_conversion_l1041_104117

/-- Represents a right cylindrical tank -/
structure Tank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the volume of oil in the tank when it's on its side -/
noncomputable def volumeOnSide (tank : Tank) (depth : ℝ) : ℝ := sorry

/-- Calculates the depth of oil when the tank is upright, given the volume -/
noncomputable def depthWhenUpright (tank : Tank) (volume : ℝ) : ℝ := sorry

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

notation:50 a " ≈ " b => approx_eq a b 0.1

theorem oil_depth_conversion (tank : Tank) (h : tank.height = 20) (d : tank.baseDiameter = 6) :
  let volumeSide := volumeOnSide tank 4
  let depthUp := depthWhenUpright tank volumeSide
  depthUp ≈ 2.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_depth_conversion_l1041_104117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_imply_power_of_two_l1041_104152

theorem prime_divisors_imply_power_of_two (b m n : ℕ) 
  (hb : b > 1) 
  (hmn : m ≠ n) 
  (h_prime_divisors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_imply_power_of_two_l1041_104152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_k_min_upper_bound_l1041_104184

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

-- Part 1
theorem solve_k (k : ℝ) : 
  (∀ x, f x > k ↔ x < -3 ∨ x > -2) → k = -2/5 := by sorry

-- Part 2
theorem min_upper_bound : 
  (∃ t, ∀ x > 0, f x ≤ t) ∧ 
  (∀ t < Real.sqrt 6 / 6, ∃ x > 0, f x > t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_k_min_upper_bound_l1041_104184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_iff_m_neq_n_l1041_104100

/-- The stone-breaking game --/
def StoneGame (m n : ℕ) : Prop :=
  ∃ (strategy : ℕ → ℕ → Bool),
    ∀ (opponent_strategy : ℕ → ℕ → Bool),
      let game_result := sorry
      game_result = true

/-- Petya can ensure victory if and only if m ≠ n --/
theorem petya_wins_iff_m_neq_n (m n : ℕ) :
  StoneGame m n ↔ m ≠ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_iff_m_neq_n_l1041_104100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cube_diagonal_l1041_104182

theorem inscribed_sphere_cube_diagonal (V : ℝ) (h : V = 36 * Real.pi) :
  ∃ (s : ℝ), s > 0 ∧ (4 / 3 * Real.pi * s^3 = V) ∧
  (Real.sqrt (3 * (2 * s)^2) = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cube_diagonal_l1041_104182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1041_104189

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

-- State the theorem
theorem f_range :
  Set.range f = { y : ℝ | 15/8 ≤ y } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1041_104189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_implies_a_value_l1041_104132

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line equation -/
def line_eq (x y a : ℝ) : Prop := x - 2*y + a = 0

/-- The chord length -/
def chord_length : ℝ := 2

/-- Theorem stating that if the chord of the circle cut by the line has a length of 2, then a = 1 -/
theorem chord_length_implies_a_value (x y a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    line_eq x₁ y₁ a ∧ line_eq x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = 1 :=
by
  sorry

#check chord_length_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_implies_a_value_l1041_104132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_gcd_bound_pentagonal_gcd_three_exists_l1041_104103

/-- The nth pentagonal number -/
def P (n : ℕ+) : ℚ := n * (3 * n - 1) / 2

/-- The greatest common divisor of 6Pn and n+1 is at most 3 -/
theorem pentagonal_gcd_bound (n : ℕ+) : Nat.gcd (Int.natAbs (Int.floor (6 * P n))) (n + 1) ≤ 3 := by
  sorry

/-- There exists an n such that the greatest common divisor of 6Pn and n+1 is exactly 3 -/
theorem pentagonal_gcd_three_exists : ∃ n : ℕ+, Nat.gcd (Int.natAbs (Int.floor (6 * P n))) (n + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_gcd_bound_pentagonal_gcd_three_exists_l1041_104103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_proof_l1041_104180

/-- The sum of the infinite geometric series with first term 2 and common ratio -5/8 -/
noncomputable def geometric_series_sum : ℝ := 16 / 13

/-- The first term of the geometric series -/
noncomputable def a : ℝ := 2

/-- The common ratio of the geometric series -/
noncomputable def r : ℝ := -5 / 8

/-- Theorem: The sum of the infinite geometric series 2 - 5/4 + 25/64 - 125/1024 + ... is equal to 16/13 -/
theorem geometric_series_sum_proof : 
  (∑' n : ℕ, a * r ^ n) = geometric_series_sum := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_proof_l1041_104180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C_distance_intersection_points_l1041_104144

-- Define the curve C
noncomputable def curve_C (a : ℝ) : ℝ × ℝ := (4 * Real.cos a + 2, 4 * Real.sin a)

-- Define the line l
noncomputable def line_l (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos (Real.pi/6), ρ * Real.sin (Real.pi/6))

-- Theorem for the polar equation of curve C
theorem polar_equation_C :
  ∀ ρ θ : ℝ, (ρ * Real.cos θ, ρ * Real.sin θ) ∈ Set.range curve_C ↔ ρ^2 - 4*ρ*Real.cos θ = 12 :=
by sorry

-- Theorem for the distance between intersection points
theorem distance_intersection_points :
  let intersection_points := Set.inter (Set.range curve_C) (Set.range line_l)
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C_distance_intersection_points_l1041_104144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1041_104156

/-- Given a parabola and a hyperbola with specific properties, 
    prove that the eccentricity of the hyperbola is √10/3 -/
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (h_p_pos : p > 0) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_parabola : ∀ x y : ℝ, y^2 = 2*p*x)
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1)
  (h_focus : ∃ c : ℝ, c > 0 ∧ c = a * Real.sqrt (1 + b^2/a^2))
  (h_directrix : 2*b^2/a = 2*b/3) :
  Real.sqrt (1 + b^2/a^2) = Real.sqrt 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1041_104156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_minus_x_equals_point_one_l1041_104150

-- Define the round_to_tenths function
noncomputable def round_to_tenths (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

-- Define the values of a, b, and c
def a : ℝ := 5.45
def b : ℝ := 2.95
def c : ℝ := 3.74

-- Define x as the sum of a, b, and c rounded to the tenths place
noncomputable def x : ℝ := round_to_tenths (a + b + c)

-- Define y as the sum of a, b, and c individually rounded to the tenths place
noncomputable def y : ℝ := round_to_tenths a + round_to_tenths b + round_to_tenths c

-- Theorem statement
theorem y_minus_x_equals_point_one : y - x = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_minus_x_equals_point_one_l1041_104150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1041_104161

/-- The constant term in the expansion of (x^6 - 1/(x*√x))^5 is 5 -/
theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), c = 5 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
      |(λ x => (x^6 - 1/(x*Real.sqrt x))^5) x - c| < ε) :=
by
  -- We define c as 5
  let c := 5
  
  -- We assert the existence of such a c
  use c
  
  -- We prove that c = 5 and the limit condition
  constructor
  · -- Prove c = 5
    rfl
  
  · -- Prove the limit condition
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1041_104161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_is_seven_l1041_104181

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The property we want to prove -/
def HasInfinitelyManySolutions (a : ℕ) : Prop :=
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, S (f k) - S (f k + a) = 2018

theorem smallest_a_is_seven :
  (HasInfinitelyManySolutions 7 ∧
   ∀ a < 7, ¬HasInfinitelyManySolutions a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_is_seven_l1041_104181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_g_even_l1041_104190

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 4) * Real.cos (x / 4) - 2 * Real.sqrt 3 * (Real.sin (x / 4))^2 + Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≥ -2) ∧
  (∀ x, f x ≤ 2) ∧
  (∃ x, f x = -2) ∧
  (∃ x, f x = 2) ∧
  (∀ x, g x = g (-x)) := by
  sorry

-- The smallest positive period is 4π
theorem f_period : 
  ∀ x, f (x + 4 * Real.pi) = f x ∧ 
  ∀ S, 0 < S ∧ S < 4 * Real.pi → ∃ y, f (y + S) ≠ f y := by
  sorry

-- g is an even function
theorem g_even : ∀ x, g x = g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_g_even_l1041_104190
