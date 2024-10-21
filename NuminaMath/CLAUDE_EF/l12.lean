import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l12_1206

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x - 1)

-- State the theorem
theorem max_ab_value (a b : ℝ) :
  (∀ x, f a x ≥ b) → a * b ≤ (1/2) * Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l12_1206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l12_1285

/-- The function f(x) = -1/2 * x^2 + 13/2 -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 13/2

/-- Theorem: If f(x) has a minimum value of 2a and a maximum value of 2b in the interval (a, b],
    then [a, b] is either [1, 3] or [-2 - √17, 13/4] -/
theorem function_extrema (a b : ℝ) (h1 : a < b) :
  (∀ x ∈ Set.Ioo a b, f x ≥ 2*a) ∧ 
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
  (∀ x ∈ Set.Ioo a b, f x ≤ 2*b) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*b) →
  ((a = 1 ∧ b = 3) ∨ (a = -2 - Real.sqrt 17 ∧ b = 13/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l12_1285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_calculation_l12_1201

/-- The amount of money lent (in Rs.) -/
def loan_amount : ℚ := 3500

/-- The interest rate A charges B (as a rational number) -/
def rate_A_to_B : ℚ := 10 / 100

/-- The interest rate B charges C (as a rational number) -/
def rate_B_to_C : ℚ := 11 / 100

/-- B's total gain over the period (in Rs.) -/
def total_gain : ℚ := 105

/-- Calculate the number of years for which B gains the total_gain amount -/
def calculate_years (amount : ℚ) (rate_borrowed : ℚ) (rate_lent : ℚ) (gain : ℚ) : ℚ :=
  gain / (amount * (rate_lent - rate_borrowed))

theorem years_calculation :
  calculate_years loan_amount rate_A_to_B rate_B_to_C total_gain = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_calculation_l12_1201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_decimal_octal_conversion_l12_1233

-- Define the binary number 101101₂
def binary_number : ℕ := 45

-- Theorem statement
theorem binary_decimal_octal_conversion :
  binary_number = 45 ∧ 
  ∃ (octal_rep : List ℕ), octal_rep = [5, 5] ∧ 
    (octal_rep.foldr (λ d acc => acc * 8 + d) 0 = binary_number) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_decimal_octal_conversion_l12_1233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_105_l12_1208

-- Define factorial function
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

-- State the theorem
theorem sqrt_nine_factorial_over_105 :
  Real.sqrt (factorial 9 / 105) = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_105_l12_1208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l12_1202

theorem complex_equation_solution (x y : ℝ) :
  (x - Complex.I) * Complex.I = y + 2 * Complex.I →
  Complex.ofReal x + Complex.I * Complex.ofReal y = 2 + Complex.I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l12_1202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l12_1287

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x ^ 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M) ∧
  (∃ (m : ℝ), m = 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l12_1287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l12_1240

theorem inequality_solution_count : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 3*x^2 + 14*x + 15 ≤ 25) ∧ Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l12_1240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_loss_pacxod_acxod_l12_1281

def is_arithmetic_progression (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = seq[1]! - seq[0]!

def digits_to_number (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem min_loss_pacxod_acxod :
  ∀ (p a c x o d : Nat),
    [p, a, c, x, o, d].all (· < 10) →
    is_arithmetic_progression [p, a, c, x, o, d] →
    p = 1 →
    digits_to_number [p, a, c, x, o, d] - digits_to_number [a, c, x, o, d] = 58000 :=
by
  intros p a c x o d h_digits h_ap h_p
  -- The proof steps would go here
  sorry

#eval digits_to_number [1, 2, 3, 4, 5, 6] - digits_to_number [2, 3, 4, 5, 6]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_loss_pacxod_acxod_l12_1281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_paintings_l12_1231

theorem marcus_paintings (paintings_per_day : ℕ → ℕ) 
  (h_double : ∀ n ∈ Finset.range 4, paintings_per_day (n + 1) = 2 * paintings_per_day n)
  (h_total : (Finset.range 5).sum paintings_per_day = 62) : 
  paintings_per_day 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_paintings_l12_1231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_separation_min_distance_C2_to_l_l12_1254

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the curve C2
def curve_C2 (x y : ℝ) : Prop := (2*x)^2 + (2*y/Real.sqrt 3)^2 = 1

-- Distance function between a point and a line
noncomputable def dist_point_line (x y : ℝ) : ℝ := |x - y - 2| / Real.sqrt 2

theorem line_curve_separation :
  ∀ x y : ℝ, line_l x y → curve_C x y → dist_point_line x y > 1 := by
  sorry

theorem min_distance_C2_to_l :
  ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧
  (∀ x y : ℝ, curve_C2 x y → dist_point_line x y ≥ d) ∧
  (∃ x y : ℝ, curve_C2 x y ∧ dist_point_line x y = d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_separation_min_distance_C2_to_l_l12_1254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_comparison_l12_1269

-- Define the regression line type
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Define the data set
def DataSet := List (ℝ × ℝ)

-- Define functions to calculate correlation coefficient, variance, and regression slope
noncomputable def correlationCoefficient (data : DataSet) : ℝ := sorry
noncomputable def variance (data : DataSet) : ℝ := sorry
noncomputable def regressionSlope (data : DataSet) : ℝ := sorry

-- Theorem statement
theorem regression_comparison 
  (data1 data2 : DataSet) 
  (line1 line2 : RegressionLine) 
  (h_mean : (data1.map Prod.snd ++ data2.map Prod.snd).sum / (data1.length + data2.length : ℝ) = 0.4)
  (h_p1 : data1.any (λ p => p.2 = 0.3))
  (h_p2 : data2.any (λ p => p.2 = 0.4))
  (h_line1 : line1.slope = regressionSlope data1 ∧ 
             line1.intercept = 0.4 - line1.slope * ((data1.map Prod.fst).sum / data1.length))
  (h_line2 : line2.slope = regressionSlope data2 ∧ 
             line2.intercept = 0.4 - line2.slope * ((data2.map Prod.fst).sum / data2.length)) :
  correlationCoefficient data1 > correlationCoefficient data2 ∧ 
  variance data1 > variance data2 ∧ 
  line1.slope > line2.slope := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_comparison_l12_1269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_cycling_speed_l12_1275

theorem lucy_cycling_speed (total_distance usual_speed first_mile_speed : ℝ)
  (h1 : total_distance = 2)
  (h2 : usual_speed = 4)
  (h3 : first_mile_speed = 3) :
  let usual_time := total_distance / usual_speed
  let first_mile_time := 1 / first_mile_speed
  let remaining_time := usual_time - first_mile_time
  let remaining_distance := total_distance - 1
  remaining_distance / remaining_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_cycling_speed_l12_1275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_rational_in_sequence_l12_1256

def sequence_x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | n + 2 => if n % 2 = 0 then 1 + sequence_x (n / 2 + 1) else 1 / sequence_x (n + 1)

theorem unique_rational_in_sequence :
  ∀ r : ℚ, r > 0 → ∃! n : ℕ, sequence_x n = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_rational_in_sequence_l12_1256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_true_proposition_l12_1222

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define the relationships
def parallel (a b : Plane ⊕ Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane ⊕ Line) : Prop := sorry
def pointOutsidePlane (pt : Point) (p : Plane) : Prop := sorry

-- Define the propositions
def proposition1 : Prop :=
  ∀ (p1 p2 : Plane) (l1 l2 : Line),
    (parallel (Sum.inr l1) (Sum.inl p2) ∧ parallel (Sum.inr l2) (Sum.inl p2)) → parallel (Sum.inl p1) (Sum.inl p2)

def proposition2 : Prop :=
  ∀ (l1 l2 l3 : Line),
    (perpendicular l1 (Sum.inr l3) ∧ perpendicular l2 (Sum.inr l3)) → parallel (Sum.inr l1) (Sum.inr l2)

def proposition3 : Prop :=
  ∀ (pt : Point) (p : Plane),
    pointOutsidePlane pt p →
    ∃! (l : Line), parallel (Sum.inr l) (Sum.inl p)

def proposition4 : Prop :=
  ∀ (l1 l2 : Line) (p : Plane),
    (perpendicular l1 (Sum.inl p) ∧ perpendicular l2 (Sum.inl p)) → parallel (Sum.inr l1) (Sum.inr l2)

theorem only_one_true_proposition :
  (¬ proposition1) ∧ (¬ proposition2) ∧ (¬ proposition3) ∧ proposition4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_true_proposition_l12_1222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_set_equality_l12_1211

theorem positive_integer_set_equality : 
  {k : ℕ+ | ∀ x : ℝ, (x^2 - 1)^(2*k.val) + (x^2 + 2*x)^(2*k.val) + (2*x + 1)^(2*k.val) = 2*(1 + x + x^2)^(2*k.val)} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_set_equality_l12_1211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_divisibility_probability_l12_1224

/-- A three-digit palindrome -/
def ThreeDigitPalindrome : Type := { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) }

/-- The set of all three-digit palindromes -/
def AllThreeDigitPalindromes : Finset ThreeDigitPalindrome := sorry

/-- The set of three-digit palindromes divisible by 11 -/
def DivisibleByEleven : Finset ThreeDigitPalindrome := 
  AllThreeDigitPalindromes.filter (fun n => n.val % 11 = 0)

/-- The probability of a randomly chosen three-digit palindrome being divisible by 11 -/
theorem palindrome_divisibility_probability : 
  (DivisibleByEleven.card : ℚ) / (AllThreeDigitPalindromes.card : ℚ) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_divisibility_probability_l12_1224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_subtraction_l12_1274

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  (digits.reverse.enum.map (fun (i, d) => d * b ^ i)).sum

/-- The problem statement --/
theorem base_subtraction : 
  let base_9_num := [3, 2, 4]
  let base_6_num := [2, 3, 1]
  (to_base_10 base_9_num 9) - (to_base_10 base_6_num 6) = 174 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_subtraction_l12_1274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l12_1270

/-- Two parallel lines in the form ax + by + d = 0 -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  d1 : ℝ
  d2 : ℝ

/-- Distance between two parallel lines -/
noncomputable def distance (l : ParallelLines) : ℝ :=
  abs (l.d1 - l.d2) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem parallel_lines_distance (c : ℝ) :
  let lines := ParallelLines.mk 3 (-2) (-1) c
  distance lines = 2 * Real.sqrt 13 / 13 →
  c = 1 ∨ c = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l12_1270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_admission_ratio_l12_1213

/-- Finds the ratio of adults to children closest to 2 given admission fees and total collected --/
theorem admission_ratio (adult_fee child_fee total : ℕ) 
  (h1 : adult_fee = 30)
  (h2 : child_fee = 15)
  (h3 : total = 2250)
  (h4 : ∃ (a c : ℕ), a ≥ 1 ∧ c ≥ 1 ∧ a * adult_fee + c * child_fee = total) :
  ∃ (a c : ℕ), a ≥ 1 ∧ c ≥ 1 ∧ a * adult_fee + c * child_fee = total ∧ 
    (∀ (a' c' : ℕ), a' ≥ 1 → c' ≥ 1 → a' * adult_fee + c' * child_fee = total → 
      |((a : ℚ) / c) - 2| ≤ |((a' : ℚ) / c') - 2|) ∧
    (a : ℚ) / c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_admission_ratio_l12_1213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l12_1272

/-- Represents the number of people transferred between classes -/
def x : ℤ := sorry

/-- The equation representing the transfer of people between two classes -/
def transfer_equation : Prop :=
  54 + x = 2 * (48 - x)

/-- Theorem stating that the transfer_equation correctly represents the scenario -/
theorem transfer_equation_correct :
  transfer_equation ↔
  (x ≥ 0 ∧ x ≤ 48 ∧ (54 + x) = 2 * (48 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l12_1272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_radical_axis_l12_1292

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the power of a point with respect to a circle
def power (p : Point) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set Point :=
  {p : Point | power p c1 = power p c2}

-- Define a function to check if a point is on a circle
def on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Theorem statement
theorem intersection_points_on_radical_axis (c1 c2 : Circle) 
  (p q : Point) (h1 : on_circle p c1)
  (h2 : on_circle p c2)
  (h3 : on_circle q c1)
  (h4 : on_circle q c2)
  (h5 : p ≠ q) :
  p ∈ radical_axis c1 c2 ∧ q ∈ radical_axis c1 c2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_radical_axis_l12_1292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l12_1279

/-- Given a real number a and a function g(x) = e^x - a*x^2 - a*x with exactly two distinct critical points x₁ and x₂, prove that (x₁ + x₂) / 2 < ln(2a) -/
theorem critical_points_inequality (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) :
  let g := fun x : ℝ => Real.exp x - a * x^2 - a * x
  (∀ x, x ≠ x₁ → x ≠ x₂ → (deriv g x ≠ 0)) →
  deriv g x₁ = 0 →
  deriv g x₂ = 0 →
  (x₁ + x₂) / 2 < Real.log (2 * a) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l12_1279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_pairwise_distinct_distances_l12_1264

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A set of points has pairwise distinct distances if the distance between any two distinct pairs of points is different -/
def hasPairwiseDistinctDistances (S : Set Point) : Prop :=
  ∀ p q r s : Point, p ∈ S → q ∈ S → r ∈ S → s ∈ S → 
    (p ≠ q ∨ r ≠ s) → (p ≠ r ∨ q ≠ s) → distance p q ≠ distance r s

/-- Main theorem -/
theorem exists_subset_with_pairwise_distinct_distances (α : ℝ) (h : α ≤ 1/7) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → 
    ∀ S : Finset Point, S.card = n → 
      ∃ T : Finset Point, T ⊆ S ∧ T.card = ⌊(n : ℝ)^α⌋ ∧ 
        hasPairwiseDistinctDistances T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_pairwise_distinct_distances_l12_1264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_rectangles_similar_l12_1232

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle where
  sides : Fin 4 → ℝ
  right_angles : ∀ i : Fin 4, Real.cos (π / 2) = 0  -- Using Real.cos instead of angle

/-- Two figures are similar if they have the same shape, meaning corresponding angles are equal
    and corresponding sides are in proportion. -/
def Similar (r1 r2 : Rectangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ i : Fin 4, r1.sides i = k * r2.sides i

/-- Not all rectangles are similar. -/
theorem not_all_rectangles_similar : ¬ ∀ r1 r2 : Rectangle, Similar r1 r2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_rectangles_similar_l12_1232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_product_l12_1295

theorem complex_modulus_product : 
  Complex.abs ((6 * Real.cos (π / 3) - 12 * Real.sin (π / 3) * Complex.I) * 
   (5 * Real.cos (π / 6) + 10 * Real.sin (π / 6) * Complex.I)) = 
  15 * Real.sqrt 91 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_product_l12_1295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l12_1234

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 2

/-- The horizontal line function -/
def line : ℝ := 1

/-- The circle predicate -/
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- The area enclosed by the parabola, line, and circle -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is equal to 9π/4 - 4 -/
theorem area_calculation :
  enclosed_area = 9 * Real.pi / 4 - 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l12_1234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l12_1297

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.log x / Real.log m)^2 + 2 * (Real.log x / Real.log m) - 3

-- Part I
theorem part_one :
  ∀ x : ℝ, f 2 x < 0 ↔ 1/8 < x ∧ x < 2 :=
sorry

-- Part II
theorem part_two :
  ∀ m : ℝ, (m > 0 ∧ m ≠ 1) →
  ((∀ x : ℝ, x ∈ Set.Icc 2 4 → f m x < 0) ↔
   (m > 0 ∧ m < (4 : ℝ)⁻¹ ^ (1/3)) ∨ (m > 4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l12_1297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_swing_time_l12_1262

-- Define the average speed of the monkey in meters per second
noncomputable def average_speed : ℝ := 1.2

-- Define the total distance swung by the monkey in meters
noncomputable def total_distance : ℝ := 2160

-- Define the function to calculate time in minutes
noncomputable def time_in_minutes (speed : ℝ) (distance : ℝ) : ℝ :=
  (distance / speed) / 60

-- Theorem statement
theorem monkey_swing_time :
  time_in_minutes average_speed total_distance = 30 := by
  -- Unfold the definitions
  unfold time_in_minutes average_speed total_distance
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_swing_time_l12_1262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l12_1205

/-- The projection of vector m⃗ onto vector n⃗ -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)

/-- Theorem stating the value of f(θ + π/12) for specific conditions -/
theorem projection_value (θ : ℝ) (h1 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) (h2 : Real.cos θ = 4/5) :
  f (θ + Real.pi / 12) = (4 * Real.sqrt 3 - 3) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l12_1205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_theorem_l12_1296

/-- A type representing a line in a plane -/
def Line : Type := ℝ → ℝ → Prop

/-- A type representing a point in a plane -/
def Point : Type := ℝ × ℝ

/-- Function to determine if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop := l p.1 p.2

/-- Function to get the intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : Option Point := sorry

/-- A collection of 10 lines -/
def bus_routes : Finset Line := sorry

theorem bus_route_theorem :
  ∃ (routes : Finset Line),
    (routes.card = 10) ∧ 
    (∀ l1 l2, l1 ∈ routes → l2 ∈ routes → l1 ≠ l2 → ∃! p : Point, on_line p l1 ∧ on_line p l2) ∧
    (∀ subset : Finset Line, subset ⊆ routes → subset.card = 8 →
      ∃ p : Point, ∃ l1 l2, l1 ∈ routes ∧ l2 ∈ routes ∧ l1 ∉ subset ∧ l2 ∉ subset ∧
        on_line p l1 ∧ on_line p l2 ∧ ∀ l ∈ subset, ¬on_line p l) ∧
    (∀ subset : Finset Line, subset ⊆ routes → subset.card = 9 →
      ∀ p : Point, ∃ l ∈ subset, on_line p l) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_theorem_l12_1296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_26_l12_1294

/-- The area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

/-- Theorem: The area of the quadrilateral with vertices (2,1), (1,6), (6,7), and (7,2) is 26 -/
theorem quadrilateral_area_is_26 : 
  quadrilateralArea (2,1) (1,6) (6,7) (7,2) = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_26_l12_1294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_theorem_l12_1277

theorem x0_value_theorem (a c x₀ : ℝ) (ha : a ≠ 0) (hx₀ : 0 ≤ x₀ ∧ x₀ ≤ 1) :
  let f := fun x => a * x^2 + c
  (∫ x in Set.Icc 0 1, f x) = f x₀ → x₀ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_theorem_l12_1277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equidistant_point_on_all_curves_l12_1242

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-4, -2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the curves
def curve1 (x y : ℝ) : Prop := 4*x + 2*y = 3
def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 3
def curve3 (x y : ℝ) : Prop := x^2 + 2*y^2 = 3
def curve4 (x y : ℝ) : Prop := x^2 - 2*y = 3

-- Theorem statement
theorem exists_equidistant_point_on_all_curves :
  (∃ x y : ℝ, curve1 x y ∧ distance (x, y) A = distance (x, y) B) ∧
  (∃ x y : ℝ, curve2 x y ∧ distance (x, y) A = distance (x, y) B) ∧
  (∃ x y : ℝ, curve3 x y ∧ distance (x, y) A = distance (x, y) B) ∧
  (∃ x y : ℝ, curve4 x y ∧ distance (x, y) A = distance (x, y) B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equidistant_point_on_all_curves_l12_1242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_digit_after_decimal_l12_1238

def decimal_sum_1_11_1_13 : ℚ := 1 / 11 + 1 / 13

theorem thirtieth_digit_after_decimal (n : ℕ) : 
  n = 30 → 
  (((decimal_sum_1_11_1_13 - ⌊decimal_sum_1_11_1_13⌋) * 10^n).floor : ℤ) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_digit_after_decimal_l12_1238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l12_1236

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (0 : ℝ) Real.pi,
    (∀ y ∈ Set.Icc (Real.pi / 8 : ℝ) (5 * Real.pi / 8),
      x < y → f x > f y) ∧
    (∀ z ∈ Set.Icc (0 : ℝ) Real.pi,
      z < Real.pi / 8 ∨ z > 5 * Real.pi / 8 →
      ∃ w ∈ Set.Icc (0 : ℝ) Real.pi, z < w ∧ f z < f w) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l12_1236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l12_1246

-- Define the given distances
noncomputable def jonathan_distance : ℝ := 7.5

-- Define the relationships between runners
noncomputable def mercedes_distance : ℝ := 2.5 * jonathan_distance
noncomputable def davonte_distance : ℝ := mercedes_distance + 3.25
noncomputable def felicia_distance : ℝ := davonte_distance - 1.75
noncomputable def emilia_distance : ℝ := (jonathan_distance + davonte_distance + felicia_distance) / 3

-- Theorem to prove
theorem total_distance_theorem :
  mercedes_distance + davonte_distance + felicia_distance + emilia_distance = 77.5833 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l12_1246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l12_1239

theorem cube_root_sum_equals_one :
  let A := (5 - 2 * Real.sqrt 13) ^ (1/3 : ℝ) + (5 + 2 * Real.sqrt 13) ^ (1/3 : ℝ)
  A = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l12_1239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_of_ten_l12_1250

theorem smallest_power_of_ten (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 3010) :
  ∃ (m n : ℕ), (Nat.factorial a * Nat.factorial b * Nat.factorial c = m * 10^n) ∧
                (¬ 10 ∣ m) ∧
                (∀ k, k < n → ∃ l, Nat.factorial a * Nat.factorial b * Nat.factorial c = l * 10^k ∧ 10 ∣ l) →
  n = 746 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_of_ten_l12_1250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tower_height_l12_1276

/-- Represents a water tower with height and capacity -/
structure WaterTower where
  height : ℝ
  capacity : ℝ

/-- Calculates the height of a scaled-down water tower model -/
noncomputable def calculateModelHeight (original : WaterTower) (modelCapacity : ℝ) : ℝ :=
  original.height * (modelCapacity / original.capacity) ^ (1/3)

/-- Theorem stating that a scaled-down model of the given water tower should be approximately 0.5 meters tall -/
theorem scaled_water_tower_height : 
  let original : WaterTower := { height := 80, capacity := 200000 }
  let modelCapacity : ℝ := 0.05
  abs (calculateModelHeight original modelCapacity - 0.5) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tower_height_l12_1276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l12_1280

/-- A circle centered on the x-axis with radius 1 passing through (2, 1) has the equation (x - 2)^2 + y^2 = 1 -/
theorem circle_equation :
  (∃ (a : ℝ), (a - 2)^2 + 1^2 = 1^2) →
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 1 ↔ ∃ (t : ℝ), x = 2 + Real.cos t ∧ y = Real.sin t) :=
by
  intro h
  intro x y
  apply Iff.intro
  · sorry
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l12_1280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l12_1241

def A : Set ℝ := {x | (1/4 : ℝ) ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 32}
def B (m : ℝ) : Set ℝ := {x | x^2 - 3*m*x + (2*m+1)*(m-1) < 0}

theorem problem_solution :
  (∀ m : ℝ, m > 2 ∧ (A ∩ B m).Nonempty → 2 < m ∧ m < 6) ∧
  (∀ m : ℝ, B m ⊆ A → m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l12_1241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l12_1261

/-- Converts polar coordinates (r, θ) to rectangular coordinates (x, y) -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given polar coordinates -/
noncomputable def given_polar : ℝ × ℝ := (8, 7 * Real.pi / 4)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ := (4 * Real.sqrt 2, -4 * Real.sqrt 2)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular given_polar.1 given_polar.2 = expected_rectangular := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l12_1261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l12_1291

noncomputable section

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (Real.sin A + Real.sin B)^2 = (2 * Real.sin B + Real.sin C) * Real.sin C →
  Real.sin A > Real.sqrt 3 / 3 →
  (c - a = a * Real.cos C) ∧ 
  (c > a) ∧ 
  (C > π/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l12_1291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l12_1271

open Real

theorem symmetry_about_origin (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) 
  (h : log a + log b = 0) :
  let f := fun x => a^x
  let g := fun x => -(log x / log b)
  ∀ x y, f x = y ↔ g (-x) = -y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l12_1271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l12_1200

/-- The circle C₁ -/
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (-2 + Real.cos t, 1 + Real.sin t)

/-- The ellipse C₂ -/
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)

/-- The line l passing through (-4, 0) with slope angle π/4 -/
noncomputable def l (s : ℝ) : ℝ × ℝ := (-4 + (Real.sqrt 2 / 2) * s, (Real.sqrt 2 / 2) * s)

/-- The distance between intersection points of l and C₁ -/
noncomputable def intersection_distance : ℝ := Real.sqrt 2

theorem intersection_distance_is_sqrt_2 : 
  ∃ (s₁ s₂ : ℝ), 
    (C₁ (Real.arccos ((l s₁).1 + 2))).2 = (l s₁).2 ∧ 
    (C₁ (Real.arccos ((l s₂).1 + 2))).2 = (l s₂).2 ∧ 
    Real.sqrt ((s₁ - s₂) ^ 2) = intersection_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l12_1200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_is_seven_stephanie_four_times_job_job_half_tim_tim_twice_tina_tina_two_younger_freddy_l12_1266

-- Define the ages of the people
def job_age : ℚ := 5
def tim_age : ℚ := 2 * job_age
def tina_age : ℚ := tim_age / 2
def freddy_age : ℚ := tina_age + 2
def stephanie_age : ℚ := freddy_age + (5/2)

-- Theorem to prove Freddy's age
theorem freddy_is_seven :
  freddy_age = 7 :=
by sorry

-- Additional theorems to represent the given conditions
theorem stephanie_four_times_job :
  stephanie_age = 4 * job_age :=
by sorry

theorem job_half_tim :
  job_age = tim_age / 2 :=
by sorry

theorem tim_twice_tina :
  tim_age = 2 * tina_age :=
by sorry

theorem tina_two_younger_freddy :
  tina_age = freddy_age - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_is_seven_stephanie_four_times_job_job_half_tim_tim_twice_tina_tina_two_younger_freddy_l12_1266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_theorem_semicircle_area_approx_l12_1214

/-- The area of a semicircular region with perimeter 20 (including both the semicircular arc and the diameter) -/
noncomputable def semicircleArea : ℝ := 200 * Real.pi / (Real.pi + 2)^2

/-- Theorem stating that the area of the semicircular region is equal to 200π / (π + 2)² -/
theorem semicircle_area_theorem (perimeter : ℝ) (h : perimeter = 20) :
  semicircleArea = 200 * Real.pi / (Real.pi + 2)^2 := by
  sorry

/-- Theorem stating that the area of the semicircular region is approximately 23.8 -/
theorem semicircle_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |semicircleArea - 23.8| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_theorem_semicircle_area_approx_l12_1214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_nine_fifths_l12_1257

/-- The ratio of the average speed of a car to the average speed of a bike -/
def speed_ratio (tractor_distance : ℚ) (tractor_time : ℚ) (car_distance : ℚ) (car_time : ℚ) : ℚ :=
  let tractor_speed := tractor_distance / tractor_time
  let bike_speed := 2 * tractor_speed
  let car_speed := car_distance / car_time
  car_speed / bike_speed

/-- Theorem: The ratio of the average speed of a car to the average speed of a bike is 9:5 -/
theorem speed_ratio_is_nine_fifths :
  speed_ratio 575 23 450 5 = 9 / 5 := by
  -- Unfold the definition of speed_ratio
  unfold speed_ratio
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

#eval speed_ratio 575 23 450 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_nine_fifths_l12_1257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l12_1248

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the point C
def C (d : ℝ) : ℝ × ℝ := (0, d)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of reciprocals of distances
noncomputable def sum_reciprocals (A B : ℝ × ℝ) (C : ℝ × ℝ) : ℝ :=
  1 / distance A C + 1 / distance B C

-- Theorem statement
theorem chord_reciprocal_sum_constant (d : ℝ) :
  (∃ t : ℝ, ∀ A B : ℝ × ℝ,
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    sum_reciprocals A B (C d) = t) →
  (∃ t : ℝ, ∀ A B : ℝ × ℝ,
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    sum_reciprocals A B (C d) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l12_1248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l12_1221

-- Define the circle using parametric equations
noncomputable def circle_param (θ : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem distance_circle_center_to_line :
  let d := (λ (p : ℝ × ℝ) (a b c : ℝ) => |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2))
  d circle_center 1 (-1) (-3) = Real.sqrt 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l12_1221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_bracelet_arrangements_l12_1227

/-- The number of distinct arrangements of n beads on a bracelet, 
    considering rotational and reflectional symmetry -/
def distinct_bracelet_arrangements (n : ℕ) : ℕ := 
  if n ≤ 2 then 1 else (n - 1).factorial

/-- There are 360 distinct arrangements of 7 beads on a bracelet, 
    considering rotational and reflectional symmetry -/
theorem seven_bead_bracelet_arrangements : 
  distinct_bracelet_arrangements 7 = 360 := by
  -- Unfold the definition and simplify
  rw [distinct_bracelet_arrangements]
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_bracelet_arrangements_l12_1227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l12_1220

theorem sequence_inequality (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_cond : ∀ n, (a n)^2 ≤ a (n + 1)) : 
  ∀ n : ℕ, a n < 1 / (n : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l12_1220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l12_1263

noncomputable def circle_radius : ℝ := 5

def circle_centers : List (ℝ × ℝ) := [(0, 5), (5, 0), (0, -5), (-5, 0)]

noncomputable def quarter_circle_area (r : ℝ) : ℝ := (Real.pi * r^2) / 4

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ := (leg^2) / 2

def num_shaded_regions : ℕ := 8

theorem shaded_area_theorem :
  (num_shaded_regions : ℝ) * (quarter_circle_area circle_radius - isosceles_right_triangle_area circle_radius) = 50 * Real.pi - 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l12_1263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_production_increase_l12_1258

theorem cabbage_production_increase (current_output : ℕ) (side_increase : ℕ) : 
  current_output = 9801 ∧ side_increase = 1 →
  current_output - (Nat.sqrt current_output - side_increase)^2 = 197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_production_increase_l12_1258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l12_1267

theorem tangent_line_triangle_area (a : ℝ) : a > 0 →
  let f (x : ℝ) := x^(-(1/2 : ℝ))
  let tangent_slope := -(1/2 : ℝ) * a^(-(3/2 : ℝ))
  let y_intercept := (3/2 : ℝ) * a^(-(1/2 : ℝ))
  let x_intercept := 3 * a
  (1/2 : ℝ) * x_intercept * y_intercept = 18 →
  a = 64 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l12_1267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l12_1289

noncomputable def M : ℝ × ℝ → Prop := fun (x, y) ↦ 2 * Real.sqrt ((x - 1)^2 + y^2) = abs (x - 4)

def trajectory : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / 4 + y^2 / 3 = 1

def line (k m : ℝ) : ℝ → ℝ := fun x ↦ k * x + m

noncomputable def distance_to_line (k m : ℝ) : ℝ := abs m / Real.sqrt (1 + k^2)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem area_of_triangle (k m : ℝ) (A B : ℝ × ℝ) :
  (∀ x y, M (x, y) ↔ trajectory (x, y)) →
  distance_to_line k m = 1 →
  trajectory A ∧ trajectory B →
  A ≠ B →
  A.1 = line k m A.2 ∧ B.1 = line k m B.2 →
  dot_product A B = -3/2 →
  (1/2 : ℝ) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 7 / 5 :=
by sorry

#check area_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l12_1289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centrally_symmetric_not_axially_symmetric_count_l12_1286

-- Define the set of shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Circle
  | RegularPentagram
  | Parabola

-- Define central symmetry property
def is_centrally_symmetric (s : Shape) : Bool :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Circle => true
  | Shape.RegularPentagram => true
  | Shape.Parabola => false

-- Define axial symmetry property
def is_axially_symmetric (s : Shape) : Bool :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Circle => true
  | Shape.RegularPentagram => true
  | Shape.Parabola => true

-- Define the set of all shapes
def all_shapes : List Shape :=
  [Shape.EquilateralTriangle, Shape.Parallelogram, Shape.Circle, Shape.RegularPentagram, Shape.Parabola]

-- Theorem statement
theorem centrally_symmetric_not_axially_symmetric_count :
  (all_shapes.filter (fun s => is_centrally_symmetric s && !is_axially_symmetric s)).length = 0 := by
  -- The proof goes here
  sorry

#eval (all_shapes.filter (fun s => is_centrally_symmetric s && !is_axially_symmetric s)).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centrally_symmetric_not_axially_symmetric_count_l12_1286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_internal_point_l12_1219

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  let d₁ := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2).sqrt
  let d₂ := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2).sqrt
  let d₃ := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2).sqrt
  d₁ = d₂ ∧ d₂ = d₃

/-- Definition of a point being inside a triangle -/
def isInside (p : Point2D) (t : Triangle) : Prop :=
  sorry -- Placeholder definition

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : Point2D) : ℝ :=
  ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2).sqrt

/-- The main theorem -/
theorem probability_closer_to_internal_point
  (ABC : Triangle)
  (D : Point2D)
  (h_equilateral : isEquilateral ABC)
  (h_side_length : distance ABC.A ABC.B = 6)
  (h_inside : isInside D ABC)
  (h_AD : distance ABC.A D = 2)
  (h_BD : distance ABC.B D = 4)
  (h_CD : distance ABC.C D = 3) :
  ∃ (p : Point2D → Prop) (μ : Set Point2D → ℝ),
    (∀ P, isInside P ABC → (p P ↔ 
      distance P D < min (distance P ABC.A) (min (distance P ABC.B) (distance P ABC.C)))) ∧
    (μ {P | isInside P ABC ∧ p P} / μ {P | isInside P ABC} = 4 * Real.pi / (9 * Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_internal_point_l12_1219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l12_1251

theorem min_value_and_range (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x : ℝ, (abs (2*a + b) + abs (2*a - b)) / abs a ≥ 4) ∧
  (∀ x : ℝ, (abs (2*a + b) + abs (2*a - b) ≥ abs a * (abs (2 + x) + abs (2 - x))) → 
      -2 ≤ x ∧ x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l12_1251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_time_is_five_months_l12_1223

/-- Represents the financial parameters of the product manufacturing scenario -/
structure ProductParameters where
  loan : ℚ
  cost : ℚ
  price : ℚ
  tax_rate : ℚ
  monthly_production : ℚ

/-- Calculates the number of months required to earn back the loan -/
noncomputable def months_to_repay (params : ProductParameters) : ℚ :=
  params.loan / (params.monthly_production * (params.price - params.cost - params.price * params.tax_rate))

/-- Theorem stating that given the specific parameters, it takes at least 5 months to repay the loan -/
theorem repayment_time_is_five_months :
  let params : ProductParameters := {
    loan := 22000,
    cost := 5,
    price := 8,
    tax_rate := 1/10,
    monthly_production := 2000
  }
  ∃ (x : ℚ), x ≥ 5 ∧ months_to_repay params = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_time_is_five_months_l12_1223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l12_1299

/-- A function f: ℝ → ℝ defined as f(x) = x^2 + ax + b cos(x) -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a * x + b * Real.cos x

/-- The set of real roots of f -/
def root_set (a b : ℝ) : Set ℝ := {x | f a b x = 0}

/-- The set of real roots of f(f(x)) -/
def double_root_set (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = 0}

/-- The main theorem stating the range of a + b -/
theorem range_of_sum (a b : ℝ) :
  root_set a b = double_root_set a b ∧ (root_set a b).Nonempty →
  a + b ∈ Set.Icc 0 4 ∧ a + b ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l12_1299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l12_1230

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C = 1/2 ∧
  t.a = 2 * Real.sqrt 3 ∧
  t.b + t.c = 4

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) : 
  t.A = 2*Real.pi/3 ∧ (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l12_1230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l12_1265

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal axis length of a hyperbola -/
noncomputable def focal_axis_length (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- The distance between focus and asymptote of a hyperbola -/
noncomputable def focus_asymptote_distance (h : Hyperbola) : ℝ :=
  h.b * Real.sqrt (1 + h.b^2 / h.a^2) / Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (h_focal : focal_axis_length h = 4)
  (h_dist : focus_asymptote_distance h = Real.sqrt 3) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l12_1265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l12_1245

/-- The function f(x) = m - |x - 2| where m is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

/-- The solution set of f(x + 2) ≥ 0 is [-1, 1] -/
def solution_set (m : ℝ) : Set ℝ := {x | f m (x + 2) ≥ 0}

theorem problem_solution (m : ℝ) (h : solution_set m = Set.Icc (-1) 1) :
  (m = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/(2*b) + 1/(3*c) = m → a + 2*b + 3*c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l12_1245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_D_E_is_30_l12_1259

/-- Represents the ring pattern of a tree species -/
structure RingPattern :=
  (fat : List Nat)
  (thin : List Nat)

/-- Calculates the total number of rings in a pattern -/
def total_rings (pattern : RingPattern) : Nat :=
  (pattern.fat.sum + pattern.thin.sum)

/-- Represents a tree species with its ring pattern and number of ring groups -/
structure TreeSpecies :=
  (pattern : RingPattern)
  (ring_groups : Nat)

/-- Calculates the age of a tree species in years -/
def age (species : TreeSpecies) : Nat :=
  (total_rings species.pattern) * species.ring_groups

theorem age_difference_D_E_is_30 (species_D species_E : TreeSpecies)
  (hD : species_D.pattern = ⟨[1, 2, 1], [5, 1, 1]⟩)
  (hE : species_E.pattern = ⟨[5, 3, 4], [1, 1]⟩)
  (hDgroups : species_D.ring_groups = 45)
  (hEgroups : species_E.ring_groups = 30) :
  Int.natAbs (age species_D - age species_E) = 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_D_E_is_30_l12_1259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_value_l12_1273

theorem tan_x_value (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < π) 
  (h3 : (Real.cos (2*x)) / (Real.sqrt 2 * Real.cos (x + π/4)) = 1/5) : 
  Real.tan x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_value_l12_1273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l12_1247

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

-- Define the domain
def domain : Set ℝ := Set.Icc (1/2) 2

-- Theorem statement
theorem min_m_value (m : ℝ) :
  (∃ x₀ ∈ domain, f x₀ - m ≤ 0) →
  m ≥ 0 := by
  sorry

#check min_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l12_1247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equals_sqrt_of_square_l12_1244

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equals_sqrt_of_square_l12_1244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexahedron_unique_sphere_l12_1255

/-- A hexahedron is a polyhedron with 6 faces -/
structure Hexahedron where
  vertices : Finset (EuclideanSpace ℝ (Fin 3))
  faces : Finset (Finset (EuclideanSpace ℝ (Fin 3)))
  vertex_count : vertices.card = 8
  face_count : faces.card = 6
  face_is_quad : ∀ f ∈ faces, f.card = 4
  vertices_in_faces : ∀ v ∈ vertices, ∃ f ∈ faces, v ∈ f

/-- A face is cyclic if all its vertices lie on a circle -/
def is_cyclic (face : Finset (EuclideanSpace ℝ (Fin 3))) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 3)) (radius : ℝ), 
    ∀ v ∈ face, ‖v - center‖ = radius

/-- Main theorem statement -/
theorem hexahedron_unique_sphere (h : Hexahedron) 
  (h_cyclic : ∀ f ∈ h.faces, is_cyclic f) :
  ∃! (center : EuclideanSpace ℝ (Fin 3)) (radius : ℝ), 
    ∀ v ∈ h.vertices, ‖v - center‖ = radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexahedron_unique_sphere_l12_1255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l12_1212

noncomputable def f (x : ℝ) : ℝ := 3 + 4 * x

noncomputable def g (x : ℝ) : ℝ := (x - 3) / 4

theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l12_1212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_ratio_l12_1278

/-- Represents a cone -/
structure Cone where
  baseArea : ℝ
  height : ℝ

/-- Represents a cylinder -/
structure Cylinder where
  baseArea : ℝ
  height : ℝ

/-- Volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * c.baseArea * c.height

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := c.baseArea * c.height

/-- Theorem: When a cone and a cylinder have the same base area and volume,
    the height of the cylinder is 1/3 times the height of the cone -/
theorem cylinder_cone_height_ratio 
  (cone : Cone) (cylinder : Cylinder) 
  (h_base : cone.baseArea = cylinder.baseArea) 
  (h_volume : coneVolume cone = cylinderVolume cylinder) :
  cylinder.height = (1/3) * cone.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_ratio_l12_1278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_intersection_point_l12_1217

-- Define the parametric equations for C₁
noncomputable def C₁ (k : ℕ) (t : ℝ) : ℝ × ℝ := (Real.cos t ^ k, Real.sin t ^ k)

-- Define the polar equation for C₂
def C₂ (ρ θ : ℝ) : Prop := 4 * ρ * Real.cos θ - 16 * ρ * Real.sin θ + 3 = 0

-- Theorem 1: C₁ is a unit circle when k = 1
theorem C₁_is_unit_circle :
  ∀ (x y : ℝ), (∃ t, C₁ 1 t = (x, y)) ↔ x^2 + y^2 = 1 := by sorry

-- Theorem 2: (1/4, 1/4) is an intersection point of C₁ and C₂ when k = 4
theorem intersection_point :
  let x : ℝ := 1/4
  let y : ℝ := 1/4
  (∃ t, C₁ 4 t = (x, y)) ∧ (∃ ρ θ, C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_intersection_point_l12_1217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l12_1218

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The function g derived from f -/
def g (a b c m : ℝ) : ℝ → ℝ := λ x ↦ f a b c x - 2 * m * x + 2

/-- Theorem stating the properties of f and g -/
theorem quadratic_function_properties (a b c : ℝ) :
  (f a b c 0 = 0) →
  (∀ x, f a b c (x + 2) - f a b c x = 4 * x) →
  ((∀ x, f a b c x = x^2 - 2 * x) ∧
   (∀ m, (m ≤ 0 → ∀ x ≥ 1, g a b c m x ≥ 1 - 2 * m) ∧
         (m > 0 → ∀ x ≥ 1, g a b c m x ≥ -m^2 - 2 * m + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l12_1218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximation_cube_root_l12_1210

noncomputable def f (x : ℝ) := Real.rpow x (1/3)

def x₀ : ℝ := 2.7
def x : ℝ := 2.54

theorem linear_approximation_cube_root :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |f x - (f x₀ + (deriv f x₀) * (x - x₀))| < ε :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximation_cube_root_l12_1210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_roots_l12_1209

noncomputable def f (x : ℝ) : ℝ := if x ≤ -1 then -x - 4 else x^2 - 5

theorem f_roots : {a : ℝ | f a - 11 = 0} = {-15, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_roots_l12_1209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_theorem_l12_1228

/-- The volume of a cone in cubic inches --/
noncomputable def cone_volume : ℝ := 9720 * Real.pi

/-- The vertex angle of the vertical cross section in degrees --/
def vertex_angle : ℝ := 90

/-- The height of the cone in inches --/
def cone_height : ℝ := 38.7

/-- Theorem stating the relationship between the cone's volume, vertex angle, and height --/
theorem cone_height_theorem (v : ℝ) (angle : ℝ) (h : ℝ) 
  (hv : v = cone_volume) (hangle : angle = vertex_angle) (hh : h = cone_height) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  (1/3 : ℝ) * Real.pi * (h / Real.sqrt 2)^3 * Real.sqrt 2 = v ∧ 
  |h - 38.7| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_theorem_l12_1228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangle_area_l12_1204

/-- Given a rectangle with width 32 cm and length 64 cm, if the area of a square is twice
    the area of this rectangle, then the perimeter of the square is 256 cm. -/
theorem square_perimeter_from_rectangle_area (square_side : ℝ) :
  let rectangle_width : ℝ := 32
  let rectangle_length : ℝ := 64
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let square_area : ℝ := square_side * square_side
  square_area = 2 * rectangle_area →
  4 * square_side = 256 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangle_area_l12_1204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_property_theorem_l12_1226

/-- Property P: For any i, j where 1 ≤ i ≤ j ≤ n, either aᵢaⱼ or aⱼ/aᵢ belongs to set A -/
def property_P (A : Set ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → (a i * a j ∈ A ∨ a j / a i ∈ A)

theorem set_property_theorem (A : Set ℝ) (a : ℕ → ℝ) (n : ℕ) 
    (h_order : ∀ i, 1 ≤ i → i < n → a i < a (i + 1))
    (h_n : n ≥ 2)
    (h_P : property_P A a n) :
  (a 1 = 1) ∧ 
  ((Finset.sum (Finset.range n) (fun i => a (i + 1))) / 
   (Finset.sum (Finset.range n) (fun i => (a (i + 1))⁻¹)) = a n) ∧
  (n = 5 → ∃ r, ∀ i, 1 ≤ i → i ≤ 5 → a i = r^(i - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_property_theorem_l12_1226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l12_1283

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.sqrt 3 * (Real.cos x) ^ 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, 2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2 - Real.sqrt 3

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4

theorem triangle_area_proof (A B C : ℝ) (a b c : ℝ) : 
  a = 7 →
  f (A / 2 - π / 6) = Real.sqrt 3 →
  Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14 →
  triangle_area a b c = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l12_1283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_theorem_l12_1253

-- We use 'theorem' instead of 'def' as we're stating a mathematical theorem
theorem tan_addition_theorem (α β : ℝ) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) (h5 : α + β < π/2) :
  Real.tan (α + β) = (Real.tan α + Real.tan β) / (1 - Real.tan α * Real.tan β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_theorem_l12_1253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l12_1260

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ m : ℕ, 
    sum_n seq (m - 1) = -2 ∧ 
    sum_n seq m = 0 ∧ 
    sum_n seq (m + 1) = 3 → 
    m = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l12_1260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_pi_over_four_l12_1268

noncomputable def A : ℂ := 2 + Complex.I
noncomputable def B : ℂ := 10 / (3 + Complex.I)

theorem angle_AOB_is_pi_over_four :
  let O : ℂ := 0
  Complex.arg (A - O) - Complex.arg (B - O) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_pi_over_four_l12_1268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l12_1288

theorem square_of_3x_plus_4_when_x_is_2 :
  ∀ x : ℝ, x = 2 → (3 * x + 4)^2 = 100 := by
  intro x h
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l12_1288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l12_1225

/-- Given a triangle ABC where a cos B = b cos A, prove that the triangle is isosceles (A = B) -/
theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a * Real.cos B = b * Real.cos A →  -- Given condition
  A = B :=                 -- Conclusion: triangle is isosceles
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l12_1225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l12_1207

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Real.cos (2 * x - Real.pi / 6) - 2 * m + 3

theorem range_of_m (m : ℝ) 
  (h₁ : ∀ x₁ ∈ Set.Icc 0 (Real.pi / 4), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 4), g m x₁ = f x₂) 
  (h₂ : m > 0) : 
  m ∈ Set.Icc 1 (4 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l12_1207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_with_bisecting_diagonals_is_parallelogram_parallelogram_with_perpendicular_diagonals_is_rhombus_isosceles_trapezoid_has_equal_diagonals_not_all_bisecting_diameters_perpendicular_to_chord_l12_1298

-- Define the necessary geometric objects
structure Quadrilateral where
  -- Add necessary fields
  mk :: -- Empty constructor for now

structure Parallelogram extends Quadrilateral where
  -- Add necessary fields
  mk :: -- Empty constructor for now

structure Rhombus extends Parallelogram where
  -- Add necessary fields
  mk :: -- Empty constructor for now

structure IsoscelesTrapezoid extends Quadrilateral where
  -- Add necessary fields
  mk :: -- Empty constructor for now

structure Circle where
  -- Add necessary fields
  mk :: -- Empty constructor for now

-- Define the propositions
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

def diagonals_perpendicular (p : Parallelogram) : Prop := sorry

def diagonals_equal (t : IsoscelesTrapezoid) : Prop := sorry

def diameter_bisects_chord (c : Circle) (d : Set Point) (ch : Set Point) : Prop := sorry

def diameter_perpendicular_to_chord (c : Circle) (d : Set Point) (ch : Set Point) : Prop := sorry

-- State the theorems
theorem quadrilateral_with_bisecting_diagonals_is_parallelogram 
  (q : Quadrilateral) : diagonals_bisect q → Parallelogram := sorry

theorem parallelogram_with_perpendicular_diagonals_is_rhombus 
  (p : Parallelogram) : diagonals_perpendicular p → Rhombus := sorry

theorem isosceles_trapezoid_has_equal_diagonals 
  (t : IsoscelesTrapezoid) : diagonals_equal t := sorry

theorem not_all_bisecting_diameters_perpendicular_to_chord 
  : ¬ ∀ (c : Circle) (d ch : Set Point), 
    diameter_bisects_chord c d ch → diameter_perpendicular_to_chord c d ch := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_with_bisecting_diagonals_is_parallelogram_parallelogram_with_perpendicular_diagonals_is_rhombus_isosceles_trapezoid_has_equal_diagonals_not_all_bisecting_diameters_perpendicular_to_chord_l12_1298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_principal_l12_1215

/-- Simple interest calculation -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem total_interest_after_trebling_principal
  (P R : ℚ) -- P is principal, R is rate
  (h1 : simpleInterest P R 10 = 800)
  (h2 : P > 0)
  (h3 : R > 0) :
  simpleInterest P R 5 + simpleInterest (3 * P) R 5 = 520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_principal_l12_1215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l12_1237

/-- Represents a cistern with two pipes. -/
structure Cistern where
  fill_time_A : ℚ  -- Time for pipe A to fill the cistern
  empty_time_B : ℚ  -- Time for pipe B to empty the cistern

/-- Calculates the time to fill the cistern when both pipes are open. -/
def time_to_fill (c : Cistern) : ℚ :=
  1 / (1 / c.fill_time_A - 1 / c.empty_time_B)

/-- Theorem stating that for a cistern with given properties, 
    the time to fill when both pipes are open is 24 hours. -/
theorem cistern_fill_time (c : Cistern) 
  (h1 : c.fill_time_A = 8)
  (h2 : c.empty_time_B = 12) : 
  time_to_fill c = 24 := by
  sorry

#check cistern_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l12_1237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l12_1249

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The dot product of two vectors -/
def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

/-- The theorem to be proved -/
theorem ellipse_constant_dot_product (C : Ellipse) 
  (h_ecc : C.eccentricity = 1/2)
  (h_area : C.a * C.b = 2 * Real.sqrt 3)
  (h_eq : C.a = 2 ∧ C.b = Real.sqrt 3) :
  ∃ (D : Point), D.x = -11/8 ∧ D.y = 0 ∧
  ∀ (M N : Point), M.x^2 / 4 + M.y^2 / 3 = 1 → N.x^2 / 4 + N.y^2 / 3 = 1 →
  dot_product (Point.mk (M.x - D.x) M.y) (Point.mk (N.x - D.x) N.y) = -135/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l12_1249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l12_1290

theorem equation_solution (x : ℝ) : 
  x ≠ 2/3 →
  ((6*x + 2) / (3*x^2 + 6*x - 4) = 3*x / (3*x - 2)) ↔ 
  (x = Real.sqrt 3 / 3 ∨ x = -Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l12_1290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_inverse_floors_l12_1235

-- Define the floor function (x] as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem max_sum_of_inverse_floors (a b : ℝ) : 
  (floor a + floor b = 0) → (a + b ≤ 2) :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_inverse_floors_l12_1235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_identity_l12_1293

theorem hyperbolic_identity (x y z : ℝ) 
  (h1 : Real.cosh x + Real.cosh y + Real.cosh z = 0)
  (h2 : Real.sinh x + Real.sinh y + Real.sinh z = 0) : 
  Real.cosh (2*x) + Real.cosh (2*y) + Real.cosh (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_identity_l12_1293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_approx_55cm_l12_1203

/-- Represents the scale of the map -/
noncomputable def map_scale : ℝ := 25 / 2.5

/-- Conversion factor from inches to centimeters -/
noncomputable def inch_to_cm : ℝ := 2.54

/-- Actual distance in miles -/
noncomputable def actual_distance : ℝ := 216.54

/-- Theorem stating that the measured distance on the map is approximately 55 cm -/
theorem map_distance_approx_55cm :
  let map_distance_inches : ℝ := actual_distance / map_scale
  let map_distance_cm : ℝ := map_distance_inches * inch_to_cm
  ∃ ε > 0, |map_distance_cm - 55| < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_approx_55cm_l12_1203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l3_l12_1282

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given two lines l₁ and l₂, and a third line l₃ passing through specific points,
    prove that l₃ has a specific slope. -/
theorem slope_of_line_l3 (l₁ l₂ l₃ : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ 3 * x - 2 * y = 6) →
  A = (-2, -5) →
  A ∈ l₁ →
  (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ y = 0) →
  B ∈ l₁ ∧ B ∈ l₂ →
  (∃ m : ℝ, m > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ l₃ ↔ y - A.2 = m * (x - A.1)) →
  A ∈ l₃ →
  C ∈ l₂ ∧ C ∈ l₃ →
  area_triangle A B C = 6 →
  (∃ m : ℝ, m > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ l₃ ↔ y - A.2 = m * (x - A.1) ∧ m = 25 / 32) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l3_l12_1282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_rate_is_three_l12_1284

/-- The rate at which Zach fills water balloons -/
def zachRate : ℕ → ℕ := sorry

/-- The total number of water balloons filled by Max and Zach -/
def totalBalloons : ℕ := 170

/-- The number of water balloons that popped -/
def poppedBalloons : ℕ := 10

/-- The time Max spent filling water balloons (in minutes) -/
def maxTime : ℕ := 30

/-- The rate at which Max fills water balloons (balloons per minute) -/
def maxRate : ℕ := 2

/-- The time Zach spent filling water balloons (in minutes) -/
def zachTime : ℕ := 40

theorem zach_rate_is_three : zachRate zachTime = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_rate_is_three_l12_1284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_confetti_triangle_area_l12_1216

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

/-- Theorem: Given a 2m × 1m rectangle and 500 points on it, 
    there exist 3 points forming a triangle with area ≤ 50 cm² -/
theorem confetti_triangle_area 
  (rect : Rectangle) 
  (points : Finset Point) : 
  rect.width = 2 ∧ 
  rect.height = 1 ∧ 
  points.card = 500 ∧ 
  (∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ rect.width ∧ 0 ≤ p.y ∧ p.y ≤ rect.height) →
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    triangle_area p1 p2 p3 ≤ 50 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_confetti_triangle_area_l12_1216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l12_1252

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 4)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x < f y) ↔ x ∈ Set.Iic 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l12_1252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_angles_l12_1243

/-- Represents a triangle with angle bisectors intersecting at a point. -/
structure TriangleWithBisectors where
  /-- The angles of the original triangle -/
  angles : Fin 3 → ℝ
  /-- The sum of the angles is π -/
  angle_sum : angles 0 + angles 1 + angles 2 = π
  /-- All angles are positive -/
  angles_positive : ∀ i, 0 < angles i

/-- Represents the smaller triangle formed by connecting the intersection point of angle bisectors to two vertices -/
structure SmallerTriangle (T : TriangleWithBisectors) where
  /-- The indices of the two angles that are halved -/
  halved_angles : Fin 2 → Fin 3
  /-- The index of the angle that remains the same -/
  same_angle : Fin 3
  /-- The halved angles and same angle are distinct -/
  distinct_angles : halved_angles 0 ≠ halved_angles 1 ∧ 
                    halved_angles 0 ≠ same_angle ∧ 
                    halved_angles 1 ≠ same_angle

/-- The theorem stating the angles of the similar smaller triangle -/
theorem similar_triangle_angles (T : TriangleWithBisectors) 
  (S : SmallerTriangle T) (h : S.halved_angles 0 = 1 ∧ S.halved_angles 1 = 2 ∧ S.same_angle = 0) :
  (T.angles 0, T.angles 1 / 2, T.angles 2 / 2) = (π/7, 2*π/7, 4*π/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_angles_l12_1243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l12_1229

noncomputable def s : ℝ := Real.sqrt (Real.sqrt (3 / 7))

def cubic_equation (x : ℝ) : Prop := x^3 - 3/7*x - 1 = 0

noncomputable def geometric_series_sum (x : ℝ) : ℝ := x^3 / (1 - x^3)^2

theorem geometric_series_equality :
  cubic_equation s →
  ∃ (S : ℝ), S = s^3 + 2*s^6 + 3*s^9 + 4*s^12 + geometric_series_sum s ∧ S = 49*s/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l12_1229
