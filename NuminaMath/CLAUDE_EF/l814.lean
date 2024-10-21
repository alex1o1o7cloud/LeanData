import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_lowest_sales_revenue_l814_81474

-- Define the price function
noncomputable def f (x : ℕ) : ℝ := (1/2) * (x^2 - 12*x + 69)

-- Define the sales volume function
def g (x : ℕ) : ℝ := x + 12

-- Define the sales revenue function
noncomputable def y (x : ℕ) : ℝ := f x * g x

-- State the theorem for the lowest price
theorem lowest_price (x : ℕ) (h : 1 ≤ x ∧ x ≤ 12) : 
  f 6 = 16.5 ∧ ∀ x, 1 ≤ x ∧ x ≤ 12 → f x ≥ 16.5 := by
  sorry

-- State the theorem for the lowest sales revenue
theorem lowest_sales_revenue (x : ℕ) (h : 1 ≤ x ∧ x ≤ 12) :
  y 5 = 28.9 ∧ ∀ x, 1 ≤ x ∧ x ≤ 12 → y x ≥ 28.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_lowest_sales_revenue_l814_81474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_imaginary_no_purely_imaginary_l814_81456

/-- The complex number Z as a function of real number a -/
def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2*a - 15 : ℝ)*Complex.I

/-- Z is a real number if and only if a = 5 or a = -3 -/
theorem Z_is_real (a : ℝ) : (Z a).im = 0 ↔ a = 5 ∨ a = -3 := by sorry

/-- Z is an imaginary number if and only if a ≠ 5 and a ≠ -3 -/
theorem Z_is_imaginary (a : ℝ) : (Z a).re ≠ 0 ∧ (Z a).im ≠ 0 ↔ a ≠ 5 ∧ a ≠ -3 := by sorry

/-- There is no value of a for which Z is a purely imaginary number -/
theorem no_purely_imaginary : ¬∃ a : ℝ, (Z a).re = 0 ∧ (Z a).im ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_imaginary_no_purely_imaginary_l814_81456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_z_l814_81433

/-- Given a set of constraints on x and y, prove that the sum of the minimum
    and maximum values of x^2 + y^2 is 36. -/
theorem sum_of_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0)
  (h2 : x - 5 ≤ 0)
  (h3 : y + 2 ≥ 0) :
  let z := fun (x y : ℝ) => x^2 + y^2
  ∃ (min_z max_z : ℝ),
    (∀ x' y', x' - y' - 2 ≥ 0 → x' - 5 ≤ 0 → y' + 2 ≥ 0 → z x' y' ≥ min_z) ∧
    (∃ x' y', x' - y' - 2 ≥ 0 ∧ x' - 5 ≤ 0 ∧ y' + 2 ≥ 0 ∧ z x' y' = min_z) ∧
    (∀ x' y', x' - y' - 2 ≥ 0 → x' - 5 ≤ 0 → y' + 2 ≥ 0 → z x' y' ≤ max_z) ∧
    (∃ x' y', x' - y' - 2 ≥ 0 ∧ x' - 5 ≤ 0 ∧ y' + 2 ≥ 0 ∧ z x' y' = max_z) ∧
    min_z + max_z = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_z_l814_81433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_quadrants_l814_81498

/-- A function representing f(x) = a^x + b --/
noncomputable def f (a b x : ℝ) : ℝ := a^x + b

/-- The set of quadrants in which a function's graph lies --/
def graph_quadrants (f : ℝ → ℝ) : Set (Fin 4) :=
  {i | ∃ x y, f x = y ∧ 
    ((i = 0 ∧ x > 0 ∧ y > 0) ∨
     (i = 1 ∧ x < 0 ∧ y > 0) ∧
     (i = 2 ∧ x < 0 ∧ y < 0) ∨
     (i = 3 ∧ x > 0 ∧ y < 0))}

/-- Theorem stating the quadrants in which f(x) = a^x + b lies, given a > 1 and b < -1 --/
theorem f_quadrants (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  graph_quadrants (f a b) = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_quadrants_l814_81498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_at_three_implies_m_four_l814_81428

/-- The function y in terms of x and m -/
noncomputable def y (x m : ℝ) : ℝ := x + m / (x - 1)

/-- The theorem stating that if y attains its minimum at x = 3, then m = 4 -/
theorem minimum_at_three_implies_m_four :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, x > 1 → y x m ≥ y 3 m) →
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_at_three_implies_m_four_l814_81428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_implies_product_l814_81476

theorem sum_of_powers_implies_product (x : ℝ) :
  (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x = 512 →
  (x + 2) * (x - 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_implies_product_l814_81476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l814_81468

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

-- State the theorem
theorem t_of_f_7 : t (f 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l814_81468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l814_81483

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 - (Real.sin x)^2 - 2 * (Real.sin x) * (Real.cos x)

-- State the theorem
theorem f_range :
  ∃ (a b : ℝ), a = -Real.sqrt 2 ∧ b = 1 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l814_81483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_le_two_l814_81469

/-- A lattice point in a 2D plane -/
structure LatticePoint where
  x : Int
  y : Int

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : LatticePoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Calculate the area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let a := p1.x * (p2.y - p3.y)
  let b := p2.x * (p3.y - p1.y)
  let c := p3.x * (p1.y - p2.y)
  (a + b + c).natAbs / 2

/-- Main theorem -/
theorem exists_triangle_area_le_two 
  (points : Finset LatticePoint)
  (h_card : points.card = 6)
  (h_bound : ∀ p ∈ points, p.x.natAbs ≤ 2 ∧ p.y.natAbs ≤ 2)
  (h_not_collinear : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangleArea p1 p2 p3 ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_le_two_l814_81469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l814_81467

open Function Real

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)

-- Theorem statement
theorem unique_function_satisfying_equation :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = 2 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l814_81467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_correct_l814_81462

/-- The number of distinct ordered pairs of positive integers (m,n) satisfying 1/m + 1/n = 1/3 -/
def count_pairs : ℕ := 3

/-- The predicate that checks if a pair of positive integers satisfies the equation -/
def satisfies_equation (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 3

/-- The theorem stating that there are exactly 3 distinct ordered pairs satisfying the equation -/
theorem count_pairs_correct :
  ∃! (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ satisfies_equation p.1 p.2) ∧ S.card = count_pairs :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_correct_l814_81462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_l814_81455

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_l814_81455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonnegative_sums_l814_81449

/-- Represents a table of real numbers -/
def Table (m n : ℕ) := Fin m → Fin n → ℝ

/-- Applies a sign change to a row of the table -/
def changeRowSign (t : Table m n) (row : Fin m) : Table m n :=
  fun i j => if i = row then -t i j else t i j

/-- Applies a sign change to a column of the table -/
def changeColumnSign (t : Table m n) (col : Fin n) : Table m n :=
  fun i j => if j = col then -t i j else t i j

/-- Calculates the sum of a row in the table -/
def rowSum (t : Table m n) (row : Fin m) : ℝ :=
  (Finset.univ.sum fun j => t row j)

/-- Calculates the sum of a column in the table -/
def columnSum (t : Table m n) (col : Fin n) : ℝ :=
  (Finset.univ.sum fun i => t i col)

/-- Represents an operation to change signs in a row or column -/
inductive SignChangeOp (m n : ℕ)
| row (i : Fin m) : SignChangeOp m n
| col (j : Fin n) : SignChangeOp m n

/-- Applies a sequence of sign change operations to a table -/
def applySignChanges (t : Table m n) : List (SignChangeOp m n) → Table m n
| [] => t
| (SignChangeOp.row i) :: ops => applySignChanges (changeRowSign t i) ops
| (SignChangeOp.col j) :: ops => applySignChanges (changeColumnSign t j) ops

/-- Theorem stating that it's possible to make all row and column sums nonnegative -/
theorem exists_nonnegative_sums (m n : ℕ) :
  ∀ (t : Table m n), ∃ (ops : List (SignChangeOp m n)),
    let t' := applySignChanges t ops
    (∀ row, rowSum t' row ≥ 0) ∧
    (∀ col, columnSum t' col ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonnegative_sums_l814_81449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_one_second_initial_velocity_two_seconds_l814_81408

/-- The acceleration due to gravity in m/s^2 -/
noncomputable def g : ℝ := 9.8

/-- The initial velocity required for a ball to return after a given time -/
noncomputable def initialVelocity (returnTime : ℝ) : ℝ := g * (returnTime / 2)

/-- Theorem for part a -/
theorem initial_velocity_one_second :
  initialVelocity 1 = 4.9 := by sorry

/-- Theorem for part b -/
theorem initial_velocity_two_seconds :
  initialVelocity 2 = 9.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_one_second_initial_velocity_two_seconds_l814_81408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_OLK_l814_81454

noncomputable section

/-- Given a circle with radius r and center O circumscribed around triangle MKH,
    where HM = a and HK^2 - HM^2 = HM^2 - MK^2,
    the area of triangle OLK (L being the centroid of MKH) is (a / (2√3)) * √(r^2 - a^2/3) -/
theorem area_triangle_OLK (r a : ℝ) (h_positive : r > 0 ∧ a > 0) :
  let b := (2*a^2 - r^2).sqrt
  let c := (2*a^2 - r^2).sqrt
  let area := (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - a^2/3)
  b^2 - a^2 = a^2 - c^2 → area = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - a^2/3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_OLK_l814_81454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_decreasing_on_interval_l814_81484

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

-- State the theorem
theorem cos_2x_decreasing_on_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
    ∀ y ∈ Set.Icc 0 (Real.pi / 2), 
      x < y → f y < f x :=
by
  -- Placeholder for the proof
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_decreasing_on_interval_l814_81484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_food_cost_is_12_50_l814_81432

/-- Represents the cat expenses for Jenny in the first year --/
structure CatExpenses where
  totalSpent : ℚ
  adoptionFee : ℚ
  vetVisits : ℚ
  toyExpenditure : ℚ

/-- Calculates the monthly food cost for the cat --/
def monthlyFoodCost (expenses : CatExpenses) : ℚ :=
  (expenses.totalSpent - (expenses.adoptionFee / 2 + expenses.vetVisits / 2 + expenses.toyExpenditure)) / 12

/-- Theorem stating that the monthly food cost is $12.50 --/
theorem monthly_food_cost_is_12_50 (expenses : CatExpenses) 
    (h1 : expenses.totalSpent = 625)
    (h2 : expenses.adoptionFee = 50)
    (h3 : expenses.vetVisits = 500)
    (h4 : expenses.toyExpenditure = 200) :
    monthlyFoodCost expenses = 25/2 := by
  sorry

/-- Computes the monthly food cost for the given expenses --/
def compute_monthly_food_cost : ℚ :=
  monthlyFoodCost { totalSpent := 625, adoptionFee := 50, vetVisits := 500, toyExpenditure := 200 }

#eval compute_monthly_food_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_food_cost_is_12_50_l814_81432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_slope_implies_a_nonnegative_l814_81419

/-- The function f(x) = ln x + ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x + 2*a*x

theorem no_negative_slope_implies_a_nonnegative 
  (h : ∀ x > 0, f_derivative a x ≥ 0) : a ≥ 0 := by
  sorry

#check no_negative_slope_implies_a_nonnegative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_slope_implies_a_nonnegative_l814_81419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l814_81465

/-- The length of the chord cut by a circle on a line --/
theorem chord_length_circle_line : 
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
  let line : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}
  ∃ (a b : ℝ × ℝ), a ∈ circle ∧ a ∈ line ∧ b ∈ circle ∧ b ∈ line ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l814_81465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_2x2_square_l814_81461

/-- Represents a rectangular piece with width and height -/
structure Piece where
  width : ℕ
  height : ℕ

/-- The set of available pieces -/
def pieces : List Piece := [
  { width := 1, height := 1 },
  { width := 1, height := 1 },
  { width := 2, height := 1 },
  { width := 1, height := 2 },
  { width := 3, height := 1 }
]

/-- The total area of all pieces -/
def totalArea : ℕ := (pieces.map (fun p => p.width * p.height)).sum

/-- Predicate to check if a 2x2 square can be formed -/
def canForm2x2Square (ps : List Piece) : Prop :=
  ∃ (arrangement : List Piece), (∀ p ∈ arrangement, p ∈ ps) ∧ 
    arrangement.map (fun p => p.width * p.height) = [4]

/-- Theorem stating that a 2x2 square cannot be formed -/
theorem cannot_form_2x2_square : ¬ canForm2x2Square pieces := by
  sorry

#check cannot_form_2x2_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_2x2_square_l814_81461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l814_81445

noncomputable def f (x : ℝ) := Real.exp x * (x^2 - 6*x + 1)

theorem f_extrema_on_interval :
  let a := 0
  let b := 6
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = Real.exp b ∧
    f x_min = -4 * Real.exp 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l814_81445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cube_closed_sqrt_shift_closed_range_l814_81466

-- Definition of a closed function
def is_closed_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ x y, x < y → (f x < f y ∨ f x > f y))

-- Theorem 1: f(x) = -x³ is a closed function
theorem negative_cube_closed :
  is_closed_function (λ x ↦ -x^3) :=
sorry

-- Theorem 2: Range of k for f(x) = k + √(x+2) to be a closed function
theorem sqrt_shift_closed_range :
  ∀ k : ℝ, is_closed_function (λ x ↦ k + Real.sqrt (x + 2)) ↔ 
  k > -9/4 ∧ k ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cube_closed_sqrt_shift_closed_range_l814_81466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_additional_results_l814_81479

theorem average_of_additional_results 
  (n₁ : ℕ) 
  (n₂ : ℕ) 
  (avg₁ : ℝ) 
  (avg_total : ℝ) 
  (h₁ : n₁ = 45) 
  (h₂ : n₂ = 25) 
  (h₃ : avg₁ = 25) 
  (h₄ : avg_total = 32.142857142857146) : 
  (((n₁ + n₂) * avg_total - n₁ * avg₁) / n₂ : ℝ) = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_additional_results_l814_81479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_greater_than_function_implies_inequality_l814_81426

theorem derivative_greater_than_function_implies_inequality 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) :
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_greater_than_function_implies_inequality_l814_81426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distribution_X_l814_81413

/-- Hypergeometric distribution probability mass function -/
def hypergeometric_pmf (N k l r : ℕ) : ℚ :=
  (Nat.choose k r * Nat.choose (N - k) (l - r)) / Nat.choose N l

/-- Problem setup -/
def N : ℕ := 15  -- Total number of cows
def k : ℕ := 5   -- Number of high-yielding cows
def l : ℕ := 3   -- Number of cows selected

/-- Theorem stating the probability distribution of X -/
theorem probability_distribution_X :
  (hypergeometric_pmf N k l 0 = 24 / 91) ∧
  (hypergeometric_pmf N k l 1 = 45 / 91) ∧
  (hypergeometric_pmf N k l 2 = 20 / 91) ∧
  (hypergeometric_pmf N k l 3 = 2 / 91) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distribution_X_l814_81413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_speed_ratio_l814_81491

/-- Represents the commute scenario -/
structure CommuteScenario where
  bike_speed : ℝ
  subway_speed : ℝ
  bike_time_normal : ℝ
  subway_time_normal : ℝ
  bike_time_broken : ℝ

/-- The commute scenario satisfies the given conditions -/
def valid_commute (c : CommuteScenario) : Prop :=
  c.bike_time_normal = 10 ∧
  c.subway_time_normal = 40 ∧
  c.bike_time_broken = 210 ∧
  c.bike_speed > 0 ∧
  c.subway_speed > 0 ∧
  c.bike_time_normal * c.bike_speed + c.subway_time_normal * c.subway_speed =
    c.bike_time_broken * c.bike_speed

/-- Theorem: The subway speed is 5 times the bike speed -/
theorem subway_speed_ratio (c : CommuteScenario) (h : valid_commute c) :
  c.subway_speed = 5 * c.bike_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_speed_ratio_l814_81491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_and_n_l814_81441

-- Define the sets A and B
def A : Set ℝ := {x | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_implies_m_and_n (m n : ℝ) :
  A ∩ B m = Set.Ioo (-1) n → m < -1 ∧ n = 1 := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b) in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_and_n_l814_81441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l814_81447

/-- The speed of a bus given its wheel radius and revolutions per minute -/
noncomputable def bus_speed (wheel_radius : ℝ) (rpm : ℝ) : ℝ :=
  2 * Real.pi * wheel_radius * rpm * 60 / 100000

/-- Theorem stating that a bus with a wheel radius of 35 cm and 500.4549590536851 rpm 
    travels at approximately 66.037 km/h -/
theorem bus_speed_calculation :
  let wheel_radius : ℝ := 35
  let rpm : ℝ := 500.4549590536851
  abs (bus_speed wheel_radius rpm - 66.037) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_calculation_l814_81447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_l814_81427

noncomputable def arithmetic_sequence (a₁ : ℝ) (n : ℕ) : ℝ := a₁ + (2 * Real.pi / 3) * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (arithmetic_sequence a₁ n)}

theorem cosine_product (a₁ : ℝ) :
  ∃ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_l814_81427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l814_81495

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x + 1

-- Define the theorem
theorem function_value_theorem (α : ℝ) 
  (h1 : α ∈ Set.Icc π (3*π/2)) -- α is in the third quadrant
  (h2 : f (α - π/6) = 1/3) :
  (Real.cos (2*α)) / (1 + Real.cos (2*α) - Real.sin (2*α)) = (4 + Real.sqrt 2) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l814_81495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l814_81457

noncomputable def f (x : ℝ) := Real.cos x * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + Real.sin x ^ 2

theorem sin_alpha_value (α : ℝ) (h1 : π / 6 < α) (h2 : α < 2 * π / 3) 
  (h3 : f (α / 2) = 1 / 2) : 
  Real.sin α = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l814_81457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disinfectant_usage_l814_81451

-- Define the original average daily usage
def original_usage : ℝ → Prop := λ x => x > 0

-- Define the relation between original and new usage
def usage_relation (x : ℝ) : Prop :=
  ∃ (days : ℝ), 
    days > 0 ∧ 
    120 = x * days ∧ 
    120 = (x + 4) * (days / 2)

-- Theorem statement
theorem disinfectant_usage : 
  ∃ (x : ℝ), original_usage x ∧ usage_relation x ∧ x = 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disinfectant_usage_l814_81451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_profit_optimization_l814_81487

/-- Profit function for Factory A -/
noncomputable def profit (x : ℝ) : ℝ := 100 * (5 * x + 1 - 3 / x)

/-- Total profit for t hours -/
noncomputable def totalProfit (x t : ℝ) : ℝ := profit x * t

/-- Production time for given amount -/
noncomputable def productionTime (x amount : ℝ) : ℝ := amount / x

theorem factory_profit_optimization (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) :
  (∀ y, 3 ≤ y ∧ y ≤ 10 ↔ totalProfit y 2 ≥ 3000) ∧
  (totalProfit 6 (productionTime 6 900) = 457500 ∧
   ∀ y, 1 ≤ y ∧ y ≤ 10 → totalProfit y (productionTime y 900) ≤ 457500) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_profit_optimization_l814_81487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_is_three_dollars_l814_81475

/-- Represents the earnings and expenses of a limousine driver for a day's work -/
structure LimoDriverEarnings where
  hourlyWage : ℚ
  rideBonus : ℚ
  reviewBonus : ℚ
  hoursWorked : ℕ
  ridesGiven : ℕ
  goodReviews : ℕ
  gallonsOfGas : ℚ
  totalOwed : ℚ

/-- Calculates the price per gallon of gas based on the driver's earnings and expenses -/
def calculateGasPrice (earnings : LimoDriverEarnings) : ℚ :=
  let baseEarnings := earnings.hourlyWage * earnings.hoursWorked +
                      earnings.rideBonus * earnings.ridesGiven +
                      earnings.reviewBonus * earnings.goodReviews
  let gasReimbursement := earnings.totalOwed - baseEarnings
  gasReimbursement / earnings.gallonsOfGas

/-- Theorem stating that the calculated gas price is $3 per gallon given the specific conditions -/
theorem gas_price_is_three_dollars (earnings : LimoDriverEarnings)
  (h1 : earnings.hourlyWage = 15)
  (h2 : earnings.rideBonus = 5)
  (h3 : earnings.reviewBonus = 20)
  (h4 : earnings.hoursWorked = 8)
  (h5 : earnings.ridesGiven = 3)
  (h6 : earnings.goodReviews = 2)
  (h7 : earnings.gallonsOfGas = 17)
  (h8 : earnings.totalOwed = 226) :
  calculateGasPrice earnings = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_is_three_dollars_l814_81475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_case_pure_imaginary_case_zero_case_l814_81406

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

-- Theorem for the real number case
theorem real_case (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = 2 := by
  sorry

-- Theorem for the pure imaginary case
theorem pure_imaginary_case (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -1/2 := by
  sorry

-- Theorem for the zero case
theorem zero_case (m : ℝ) : z m = 0 ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_case_pure_imaginary_case_zero_case_l814_81406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l814_81463

/-- The volume of a cone formed from a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  (1/3 : ℝ) * Real.pi * (r/2)^2 * Real.sqrt (r^2 - (r/2)^2) = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l814_81463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_sixteen_l814_81486

theorem log_one_fourth_sixteen : Real.log 16 / Real.log (1/4) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_sixteen_l814_81486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l814_81434

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_eight (seq : ArithmeticSequence) 
  (h_geom_mean : (seq.a 4) ^ 2 = (seq.a 2) * (seq.a 7))
  (h_sum_five : S seq 5 = 50) :
  S seq 8 = 104 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l814_81434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_largest_area_l814_81420

-- Define the shapes
def triangle_shape (A B C : ℝ) : Prop := A = 60 ∧ B = 45 ∧ C = Real.sqrt 2

def rhomboid_shape (d1 d2 angle : ℝ) : Prop :=
  d1 = Real.sqrt 2 ∧ d2 = Real.sqrt 3 ∧ angle = 75

def circle_shape (r : ℝ) : Prop := r = 1

def square_shape (d : ℝ) : Prop := d = 2.5

-- Theorem statement
theorem circle_largest_area :
  ∀ (tA tB tC : ℝ) (rd1 rd2 rangle : ℝ) (cr : ℝ) (sd : ℝ),
    triangle_shape tA tB tC →
    rhomboid_shape rd1 rd2 rangle →
    circle_shape cr →
    square_shape sd →
    (π * cr^2 > 0.5 * tC^2 * Real.sin (tA * π / 180)) ∧
    (π * cr^2 > 0.5 * rd1 * rd2 * Real.sin (rangle * π / 180)) ∧
    (π * cr^2 > 0.25 * sd^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_largest_area_l814_81420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l814_81424

/-- Line l in parametric form -/
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

/-- Circle C in polar form -/
noncomputable def circle_C (θ : ℝ) : ℝ := 6 * Real.sin θ

/-- Point P -/
def point_P : ℝ × ℝ := (1, 2)

/-- Cartesian equation of circle C -/
def circle_C_cartesian (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

/-- Intersection points of line l and circle C -/
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t α ∧ circle_C_cartesian p.1 p.2}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Minimum value of |PA| + |PB| is 2√7 -/
theorem min_sum_distances :
  ∀ α : ℝ, ∃ A B, A ∈ intersection_points α → B ∈ intersection_points α →
    ∀ A' B', A' ∈ intersection_points α → B' ∈ intersection_points α →
      distance point_P A + distance point_P B ≤ distance point_P A' + distance point_P B'
  ∧ ∃ α : ℝ, ∃ A B, A ∈ intersection_points α ∧ B ∈ intersection_points α ∧
      distance point_P A + distance point_P B = 2 * Real.sqrt 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l814_81424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l814_81494

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines 3x + y - 3 = 0 and 6x + 2y + 1 = 0 is 7√10 / 20 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 6 2 (-6) 1 = 7 * Real.sqrt 10 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l814_81494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l814_81402

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 101054) :
  4 = (n % 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l814_81402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_condition_intersection_condition_l814_81444

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line l
def lineL (k x y : ℝ) : Prop := k * x - y - 5 * k + 4 = 0

-- Theorem for bisection case
theorem bisection_condition (k : ℝ) :
  (∀ x y : ℝ, circleC x y → lineL k x y) ↔ k = 1/2 :=
sorry

-- Theorem for intersection with chord length 6
theorem intersection_condition (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
    lineL k x₁ y₁ ∧ lineL k x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) ↔ k = -3/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_condition_intersection_condition_l814_81444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_90000_l814_81405

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_investment : ℕ  -- A's investment in rupees
  a_months : ℕ     -- Number of months A invested
  b_months : ℕ     -- Number of months B invested
  profit_ratio_num : ℕ  -- Numerator of profit sharing ratio
  profit_ratio_den : ℕ  -- Denominator of profit sharing ratio

/-- Calculates B's investment given the partnership details -/
def calculate_b_investment (p : Partnership) : ℕ :=
  (p.a_investment * p.a_months * p.profit_ratio_den) / (p.b_months * p.profit_ratio_num)

/-- Theorem stating that B's investment is 90,000 rupees given the problem conditions -/
theorem b_investment_is_90000 :
  let p : Partnership := {
    a_investment := 35000,
    a_months := 12,
    b_months := 7,
    profit_ratio_num := 2,
    profit_ratio_den := 3
  }
  calculate_b_investment p = 90000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_90000_l814_81405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_correct_l814_81411

/-- The original number of faculty members before a reduction -/
def original_faculty : ℕ := 244

/-- The percentage of faculty retained after reduction -/
def retained_percentage : ℚ := 80 / 100

/-- The number of faculty members after reduction -/
def reduced_faculty : ℕ := 195

/-- Theorem stating that the original faculty count is correct given the reduction -/
theorem faculty_reduction_correct : 
  ⌈(reduced_faculty : ℚ) / retained_percentage⌉ = original_faculty := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_correct_l814_81411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_is_135_l814_81471

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℚ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_angles : ℚ := 180 * (n - 2)  -- sum of interior angles formula
  sum_of_angles / n  -- each angle in a regular polygon

/-- Proof that the measure of each interior angle of a regular octagon is 135° -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  unfold regular_octagon_interior_angle
  norm_num

#eval regular_octagon_interior_angle  -- Should output 135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_is_135_l814_81471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l814_81404

/-- Calculates the amount of alcohol added to a solution -/
noncomputable def alcohol_added (initial_volume : ℝ) (initial_concentration : ℝ) 
                  (water_added : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added + 
    (final_concentration * (initial_volume + water_added) - initial_alcohol) / (1 - final_concentration)
  (final_concentration * final_volume - initial_alcohol)

theorem alcohol_solution_problem :
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.05
  let water_added : ℝ := 7.5
  let final_concentration : ℝ := 0.09
  abs (alcohol_added initial_volume initial_concentration water_added final_concentration - 2.5) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l814_81404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l814_81490

noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 6)

theorem f_monotone_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi/3 ∧ b = Real.pi/12 ∧
  (∀ x y, -Real.pi/3 ≤ x ∧ x < y ∧ y ≤ Real.pi/3 → f x ≤ f y) ∧
  (∀ x y, -Real.pi/3 ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ ε > 0, ∃ x y, b < x ∧ x < y ∧ y < b + ε ∧ f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l814_81490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proj_equation_l814_81401

def v (y : ℝ) : Fin 2 → ℝ := ![2, y]
def w : Fin 2 → ℝ := ![7, 5]

noncomputable def proj (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ((u • v) / (v • v)) • v

theorem proj_equation (y : ℝ) :
  proj (v y) w = ![(-14 : ℝ), -10] → y = -32.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proj_equation_l814_81401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l814_81493

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x + 1

theorem f_minimum_value (a : ℝ) :
  a ∈ Set.Ioo (1/3) 1 →
  (∀ t ∈ Set.Icc 2 3, ∀ x ∈ Set.Ioo 0 t, f a x ≥ f a t) →
  a ∈ Set.Icc (2 * Real.log 2 - 1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l814_81493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_sum_of_two_l814_81435

theorem smallest_n_sum_of_two : ∃ (n : ℕ) (S : Finset ℝ),
  n = 6 ∧
  S.card = n ∧
  (∀ x ∈ S, ∃ y z, y ∈ S ∧ z ∈ S ∧ y ≠ z ∧ x = y + z) ∧
  (∀ m : ℕ, m < n →
    ¬∃ (T : Finset ℝ), T.card = m ∧ (∀ x ∈ T, ∃ y z, y ∈ T ∧ z ∈ T ∧ y ≠ z ∧ x = y + z)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_sum_of_two_l814_81435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l814_81440

-- Define set A
def A : Set ℝ := {x | x + 1/2 ≥ 3/2 ∨ x + 1/2 ≤ -3/2}

-- Define set B
def B : Set ℝ := {x | x^2 + x < 6}

-- Theorem statement
theorem intersection_of_A_and_B : 
  A ∩ B = Set.Ioc (-3) (-2) ∪ Set.Ico 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l814_81440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_permutation_of_2016_l814_81481

def is_permutation_of_2016 (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    n = a * 1000 + b * 100 + c * 10 + d ∧
    Multiset.ofList [a, b, c, d] = Multiset.ofList [2, 0, 1, 6]

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem unique_square_permutation_of_2016 : 
  ∃! n : ℕ, is_permutation_of_2016 n ∧ is_four_digit n ∧ is_perfect_square n ∧ n = 2601 := by
  sorry

#check unique_square_permutation_of_2016

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_permutation_of_2016_l814_81481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circumcircle_area_l814_81450

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the right branch of the hyperbola
variable (P : ℝ × ℝ)

-- Distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem hyperbola_circumcircle_area :
  hyperbola P.1 P.2 →
  P.1 > 0 →
  distance P F₁ = 2 * distance P F₂ →
  let R := distance F₁ F₂ / (2 * Real.sqrt (1 - ((distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2)^2 / (4 * (distance P F₁)^2 * (distance P F₂)^2)))
  Real.pi * R^2 = 256 * Real.pi / 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circumcircle_area_l814_81450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reagent_dosage_and_cost_l814_81459

def standard_dosage : ℝ := 220
def bottle_dosages : List ℝ := [230, 226, 218, 223, 214, 225, 205, 212]
def adjustment_cost : ℝ := 10

theorem reagent_dosage_and_cost :
  let total_dosage := bottle_dosages.sum
  let deviations := bottle_dosages.map (λ x => x - standard_dosage)
  let total_adjustment := deviations.map abs |>.sum
  let total_cost := adjustment_cost * total_adjustment
  (total_dosage = 1753 ∧ total_cost = 550) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reagent_dosage_and_cost_l814_81459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_medians_theorem_l814_81464

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A skew median of a lateral face of a regular tetrahedron -/
structure SkewMedian (t : RegularTetrahedron) where

/-- The angle between two skew medians of the lateral faces of a regular tetrahedron -/
noncomputable def angle_between_skew_medians (t : RegularTetrahedron) : ℝ :=
  Real.arccos (1/6)

/-- The distance between two skew medians of the lateral faces of a regular tetrahedron -/
noncomputable def distance_between_skew_medians (t : RegularTetrahedron) : ℝ :=
  t.edge_length * Real.sqrt (2/35)

theorem skew_medians_theorem (t : RegularTetrahedron) 
  (m1 m2 : SkewMedian t) : 
  angle_between_skew_medians t = Real.arccos (1/6) ∧ 
  distance_between_skew_medians t = t.edge_length * Real.sqrt (2/35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_medians_theorem_l814_81464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_omega_range_l814_81458

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  4 * Real.sin (ω * x) * (Real.sin ((ω * x / 2) + (Real.pi / 4)))^2 + Real.cos (2 * ω * x) - 1

theorem f_increasing_iff_omega_range (ω : ℝ) :
  (ω > 0) →
  (∀ x₁ x₂ : ℝ, -Real.pi/3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2*Real.pi/3 → f ω x₁ < f ω x₂) ↔
  (0 < ω ∧ ω ≤ 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_omega_range_l814_81458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_A_intersection_value_l814_81499

-- Define the expression A
noncomputable def A (a b : ℝ) : ℝ := (a - a^2 / (a + b)) / (a^2 * b^2 / (a^2 - b^2))

-- Theorem 1: Simplification of A
theorem A_simplification (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a^2 - b^2 ≠ 0) (h3 : a ≠ 0) (h4 : b ≠ 0) :
  A a b = (a - b) / (a * b) := by sorry

-- Theorem 2: Value of A at intersection point
theorem A_intersection_value (a b : ℝ) (h1 : b = a - 3) (h2 : b = 2 / a) (h3 : a ≠ 0) :
  (a - b) / (a * b) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_A_intersection_value_l814_81499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_mutually_inscribed_triangles_l814_81417

/-- A triangle represented as a function from Fin 3 to ℝ × ℝ -/
def Triangle := Fin 3 → ℝ × ℝ

/-- The set of sides of a triangle -/
def Triangle.sides (T : Triangle) : Set (Set (ℝ × ℝ)) :=
  { s | ∃ i j : Fin 3, i ≠ j ∧ s = {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • (T i) + t • (T j)} }

/-- The set of vertices of a triangle -/
def Triangle.vertices (T : Triangle) : Set (ℝ × ℝ) :=
  { v | ∃ i : Fin 3, v = T i }

/-- A triangle is inscribed in another if its vertices are points on the sides of the other triangle, excluding the vertices. -/
def IsInscribed (T1 T2 : Triangle) : Prop :=
  ∀ v ∈ T1.vertices, ∃ s ∈ T2.sides, v ∈ s ∧ v ∉ T2.vertices

/-- There do not exist two triangles, each inscribed in the other -/
theorem no_mutually_inscribed_triangles :
  ¬ ∃ (T1 T2 : Triangle), IsInscribed T1 T2 ∧ IsInscribed T2 T1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_mutually_inscribed_triangles_l814_81417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quotient_abs_abs_one_minus_two_i_div_two_plus_i_l814_81446

noncomputable def complex_abs (z : ℂ) : ℝ := Complex.abs z

theorem complex_quotient_abs (z w : ℂ) :
  complex_abs (z / w) = complex_abs z / complex_abs w :=
sorry

theorem abs_one_minus_two_i_div_two_plus_i :
  complex_abs ((1 - 2 * Complex.I) / (2 + Complex.I)) = 1 :=
by
  have h1 : complex_abs (1 - 2 * Complex.I) = Real.sqrt 5 := by sorry
  have h2 : complex_abs (2 + Complex.I) = Real.sqrt 5 := by sorry
  calc
    complex_abs ((1 - 2 * Complex.I) / (2 + Complex.I))
      = complex_abs (1 - 2 * Complex.I) / complex_abs (2 + Complex.I) := by apply complex_quotient_abs
    _ = Real.sqrt 5 / Real.sqrt 5 := by rw [h1, h2]
    _ = 1 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quotient_abs_abs_one_minus_two_i_div_two_plus_i_l814_81446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_sales_l814_81412

/-- The number of books sold on Tuesday given the total stock, sales on other days, and the percentage of unsold books -/
def books_sold_tuesday (total_stock : ℕ) (sold_monday : ℕ) (sold_wednesday : ℕ) (sold_thursday : ℕ) (sold_friday : ℕ) (percent_unsold : ℚ) : ℕ :=
  let total_sold := total_stock - (total_stock * (percent_unsold / 100 : ℚ)).floor
  let other_days_sold := sold_monday + sold_wednesday + sold_thursday + sold_friday
  (total_sold - other_days_sold).toNat

/-- Theorem stating that the number of books sold on Tuesday is 50 -/
theorem tuesday_sales : books_sold_tuesday 1400 75 64 78 135 (71.28571428571429 : ℚ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_sales_l814_81412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_24_game_not_algorithm_l814_81430

-- Define the characteristics of an algorithm
def has_clear_steps (operation : String) : Prop := sorry
def has_finite_steps (operation : String) : Prop := sorry
def yields_correct_result (operation : String) : Prop := sorry

def is_algorithm (operation : String) : Prop :=
  has_clear_steps operation ∧ has_finite_steps operation ∧ yields_correct_result operation

-- Define the operations
def calculate_circle_area : String := "Calculating the area of a circle given its radius"
def calculate_24_game_possibility : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def find_line_equation : String := "Finding the equation of a line given two points in the coordinate plane"
def arithmetic_operations : String := "Operations of addition, subtraction, multiplication, and division"

-- Theorem statement
theorem only_24_game_not_algorithm :
  is_algorithm calculate_circle_area ∧
  is_algorithm find_line_equation ∧
  is_algorithm arithmetic_operations ∧
  ¬is_algorithm calculate_24_game_possibility :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_24_game_not_algorithm_l814_81430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l814_81480

theorem largest_multiple_of_seven (n : ℤ) : 
  (∀ k : ℤ, k > n → -(7 * k) ≤ -150) → 
  (-(7 * n) > -150) → 
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l814_81480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_S_eq_pow4_l814_81492

/-- Definition of S_i sequence -/
def S (i : ℕ) : ℕ :=
  match i with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | 6 => 111
  | 7 => 175
  | _ => 0  -- For undefined cases

/-- Sum of odd-indexed S up to 2n-1 -/
def sum_odd_S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => S (2 * i + 1))

/-- Theorem: The sum of odd-indexed S up to 2n-1 equals n^4 -/
theorem sum_odd_S_eq_pow4 (n : ℕ) : sum_odd_S n = n ^ 4 := by
  sorry

#check sum_odd_S_eq_pow4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_S_eq_pow4_l814_81492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_distance_is_correct_l814_81423

/-- The distance between points D and E in meters -/
def DE : ℝ := 200

/-- Dan's speed in meters per second -/
def dan_speed : ℝ := 10

/-- Eva's speed in meters per second -/
def eva_speed : ℝ := 15

/-- The angle between Dan's path and DE in radians -/
noncomputable def dan_angle : ℝ := Real.pi / 4

/-- The time at which Dan and Eva meet -/
noncomputable def meeting_time : ℝ := 
  ((-1000 * Real.sqrt 2) + Real.sqrt ((1000 * Real.sqrt 2)^2 + 4 * 125 * 40000)) / 250

/-- The distance Dan skis before meeting Eva -/
noncomputable def dan_distance : ℝ := dan_speed * meeting_time

theorem dan_distance_is_correct : 
  ∃ ε > 0, |dan_distance - 178.4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_distance_is_correct_l814_81423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Ω_infinite_l814_81477

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set Ω
def Ω : Set Point := sorry

-- Define the midpoint property
def isMidpoint (p q r : Point) : Prop :=
  p.x = (q.x + r.x) / 2 ∧ p.y = (q.y + r.y) / 2

-- Axiom: Ω is non-empty
axiom Ω_nonempty : Set.Nonempty Ω

-- Axiom: Each point in Ω is the midpoint of two other points in Ω
axiom midpoint_property : ∀ p, p ∈ Ω → ∃ q r, q ∈ Ω ∧ r ∈ Ω ∧ isMidpoint p q r

-- Theorem to prove
theorem Ω_infinite : Set.Infinite Ω := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Ω_infinite_l814_81477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_and_distance_l814_81415

/-- The ellipse C with equation x^2 + y^2/4 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/4 = 1}

/-- The line y = kx + 1 for a given k -/
def line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- The intersection points of C and the line -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := C ∩ line k

/-- Check if two vectors are perpendicular -/
def are_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_intersection_perpendicular_and_distance :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, A ∈ intersection_points k ∧ B ∈ intersection_points k ∧ A ≠ B) →
  (k = 1/2 ∨ k = -1/2 ↔ are_perpendicular A B) ∧
  (are_perpendicular A B → distance A B = 4 * Real.sqrt 65 / 17) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_and_distance_l814_81415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l814_81472

theorem vector_computation : 
  (4 : ℝ) • (![3, -5] : Fin 2 → ℝ) - (3 : ℝ) • (![2, -10] : Fin 2 → ℝ) = ![6, 10] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_computation_l814_81472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l814_81409

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- Theorem stating the range of eccentricity for a hyperbola under given conditions -/
theorem hyperbola_eccentricity_range
  (a b c : ℝ)
  (h : Hyperbola a b)
  (F : Point)  -- Right focus
  (A B : Point)    -- Intersection points of asymptote and circle
  (h_c : c^2 = a^2 + b^2)
  (h_c_pos : c > 0)
  (h_acute : Triangle)
  (h_acute_cond : ∀ (x y : ℝ), (x - c)^2 + y^2 = a^2 → b * x - a * y = 0 → 
    (x = A.x ∧ y = A.y) ∨ (x = B.x ∧ y = B.y))
  : Real.sqrt 6 / 2 < eccentricity h ∧ eccentricity h < Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l814_81409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_two_and_four_l814_81453

/-- The displacement function of a particle moving along a straight line -/
noncomputable def displacement (t : ℝ) : ℝ := (1/3) * t^3 - 3 * t^2 + 8 * t

/-- The velocity function of the particle -/
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

theorem velocity_zero_at_two_and_four :
  velocity 2 = 0 ∧ velocity 4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_two_and_four_l814_81453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l814_81478

noncomputable def f (x : ℝ) := Real.cos (Real.pi / 2 + 2 * x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, 0 < p → p < Real.pi → ∃ x, f (x + p) ≠ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l814_81478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l814_81421

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x ^ 2 / Real.log (y ^ 12)) * (Real.log y ^ 3 / Real.log (x ^ 10)) * 
  (Real.log x ^ 4 / Real.log (y ^ 8)) * (Real.log y ^ 8 / Real.log (x ^ 6)) * 
  (Real.log x ^ 10 / Real.log (y ^ 3)) = 5 / 18 * (Real.log x / Real.log y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l814_81421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_times_l814_81437

noncomputable def distance : ℝ := 360
noncomputable def speed_express : ℝ := 72
noncomputable def speed_slow : ℝ := 48
noncomputable def early_start : ℝ := 25 / 60

theorem train_meeting_times :
  (∃ t : ℝ, t * speed_express + t * speed_slow = distance ∧ t = 3) ∧
  (∃ t : ℝ, t * speed_slow + (t + early_start) * speed_express = distance ∧ t = 11 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_times_l814_81437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l814_81403

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    real axis length of 2, and passing through the point (2,3),
    prove that its asymptotes have the equation y = ±√3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2^2 / a^2 - 3^2 / b^2 = 1) →  -- Point (2,3) satisfies the equation
  (2 * a = 2) →                  -- Real axis length is 2
  (∀ x y : ℝ, y = Real.sqrt 3 * x ∨ y = -(Real.sqrt 3) * x ↔ 
    ∃ t : ℝ, x = a * t ∧ y = b * t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l814_81403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l814_81410

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  (1.2 * L * 1.2 * W - L * W) / (L * W) = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l814_81410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_topping_cost_is_two_l814_81460

/-- Represents the cost structure of a pizza order -/
structure PizzaOrder where
  baseCost : ℝ
  slices : ℕ
  toppingsCount : ℕ
  costPerSlice : ℝ
  secondTierToppingsCount : ℕ
  secondTierToppingCost : ℝ
  thirdTierToppingCost : ℝ

/-- Calculates the cost of the first topping given a pizza order -/
def firstToppingCost (order : PizzaOrder) : ℝ :=
  let totalCost := order.costPerSlice * (order.slices : ℝ)
  let baseCost := order.baseCost
  let secondTierCost := (order.secondTierToppingsCount : ℝ) * order.secondTierToppingCost
  let thirdTierCount := order.toppingsCount - order.secondTierToppingsCount - 1
  let thirdTierCost := (thirdTierCount : ℝ) * order.thirdTierToppingCost
  totalCost - baseCost - secondTierCost - thirdTierCost

/-- Theorem stating that the first topping costs $2.00 for the given pizza order -/
theorem first_topping_cost_is_two : 
  let order : PizzaOrder := {
    baseCost := 10
    slices := 8
    toppingsCount := 7
    costPerSlice := 2
    secondTierToppingsCount := 2
    secondTierToppingCost := 1
    thirdTierToppingCost := 0.5
  }
  firstToppingCost order = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_topping_cost_is_two_l814_81460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_is_negative_seven_l814_81489

/-- A straight line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  l.y₁ - m * l.x₁

/-- The line passing through (2, -3) and (6, 5) -/
def our_line : Line := {
  x₁ := 2
  y₁ := -3
  x₂ := 6
  y₂ := 5
}

theorem y_intercept_is_negative_seven :
  y_intercept our_line = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_is_negative_seven_l814_81489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_cases_stratified_l814_81414

/-- Represents a sampling case with 10 drawn numbers -/
structure SamplingCase where
  numbers : Fin 10 → Nat

/-- Checks if a sampling case follows the stratified sampling pattern -/
def is_stratified_sampling (case : SamplingCase) : Prop :=
  ∃ (a b c : Nat), a + b + c = 10 ∧
    a = (List.filter (λ n => n ≤ 108) (List.map case.numbers (List.range 10))).length ∧
    b = (List.filter (λ n => 109 ≤ n ∧ n ≤ 189) (List.map case.numbers (List.range 10))).length ∧
    c = (List.filter (λ n => 190 ≤ n ∧ n ≤ 270) (List.map case.numbers (List.range 10))).length

/-- The total number of students -/
def total_students : Nat := 270

/-- The number of students in the first grade -/
def first_grade_students : Nat := 108

/-- The number of students in the second grade -/
def second_grade_students : Nat := 81

/-- The number of students in the third grade -/
def third_grade_students : Nat := 81

/-- Case 2 of drawn numbers -/
def case2 : SamplingCase :=
  ⟨λ i => [6, 33, 60, 87, 114, 141, 168, 195, 222, 249][i.val]⟩

/-- Case 4 of drawn numbers -/
def case4 : SamplingCase :=
  ⟨λ i => [12, 39, 66, 93, 120, 147, 174, 201, 228, 255][i.val]⟩

/-- Theorem stating that both cases 2 and 4 can possibly be from stratified sampling -/
theorem both_cases_stratified :
  is_stratified_sampling case2 ∧ is_stratified_sampling case4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_cases_stratified_l814_81414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_seven_l814_81418

/-- Given a polynomial P(x) with the specified structure and root properties, prove that P(7) = 51840 -/
theorem polynomial_value_at_seven (g h i j k l : ℝ) : 
  let P : ℂ → ℂ := λ x => (3*x^5 - 45*x^4 + g*x^3 + h*x^2 + i*x + j) * (4*x^3 - 60*x^2 + k*x + l)
  (∃ (z : ℂ), P z = 0 ∧ (z = 1 ∨ z = 2 ∨ z = 3 ∨ z = 4 ∨ z = 5 ∨ z = 6)) →
  (P 7 : ℂ).re = 51840 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_seven_l814_81418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l814_81400

theorem divisibility_property (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  ∃ m : ℤ, (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4*k - 1) = m * (n^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l814_81400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_matrix_is_correct_l814_81473

def dilation_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/2, 0, 0],
    ![0, 1/2, 0],
    ![0, 0, 1/2]]

theorem dilation_matrix_is_correct :
  ∀ (v : Fin 3 → ℚ),
  Matrix.mulVec dilation_matrix v = (1/2 : ℚ) • v := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_matrix_is_correct_l814_81473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l814_81452

/-- Represents a cube with vertices and edges -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- Represents a path on the cube -/
def CubePath (cube : Cube) := List (Fin 8)

/-- The probability of choosing a specific edge at a vertex -/
def edgeProbability : ℚ := 1 / 3

/-- The number of moves the bug makes -/
def numMoves : ℕ := 6

/-- Checks if a path visits each vertex exactly once and ends at the opposite vertex -/
def isValidPath (cube : Cube) (path : CubePath cube) : Prop :=
  path.length = numMoves + 1 ∧
  path.toFinset = cube.vertices ∧
  path.head? ≠ path.getLast?

/-- The probability of a specific valid path -/
def validPathProbability : ℚ := edgeProbability ^ numMoves

/-- The main theorem to prove -/
theorem bug_probability (cube : Cube) :
  (∃ (path : CubePath cube), isValidPath cube path) →
  validPathProbability = 1 / 729 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l814_81452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABF_is_right_angle_l814_81425

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity (√5 - 1)/2 -/
structure Ellipse (a b : ℝ) : Prop where
  positive : 0 < b ∧ b < a
  equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
  eccentricity : (a^2 - b^2).sqrt / a = (Real.sqrt 5 - 1) / 2

/-- The right vertex of the ellipse -/
noncomputable def right_vertex (a b : ℝ) : ℝ × ℝ := (a, 0)

/-- The top point of the minor axis -/
noncomputable def top_minor_axis (a b : ℝ) : ℝ × ℝ := (0, b)

/-- The left focus of the ellipse -/
noncomputable def left_focus (a b : ℝ) : ℝ × ℝ := 
  let c := a * ((Real.sqrt 5 - 1) / 2)
  (-c, 0)

/-- The angle between the line from the right vertex to the top of the minor axis
    and the line from the top of the minor axis to the left focus -/
noncomputable def angle_ABF (a b : ℝ) : ℝ := 
  sorry

theorem angle_ABF_is_right_angle (a b : ℝ) (e : Ellipse a b) :
  angle_ABF a b = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABF_is_right_angle_l814_81425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_inclusion_l814_81416

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem set_operations_and_inclusion :
  (A ∩ B = Set.Icc 3 6) ∧
  (A ∪ B = Set.Ioo 2 8) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = Set.Ici 8 ∪ Set.Iic 2) ∧
  (∀ a : ℝ, A ⊆ C a → a < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_inclusion_l814_81416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_coffee_purchase_l814_81431

theorem janes_coffee_purchase (b m c n : ℕ) : 
  b + m + c = 6 →
  75 * b + 60 * m + 100 * c = 100 * n →
  c = 1 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_coffee_purchase_l814_81431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_7_oclock_l814_81448

/-- The smaller angle formed by the hands of a clock at 7 o'clock -/
noncomputable def clock_angle_at_7 : ℝ :=
  let total_hours : ℕ := 12
  let degrees_per_hour : ℝ := 360 / total_hours
  let hour_hand_angle : ℝ := 7 * degrees_per_hour
  let minute_hand_angle : ℝ := 0
  min (hour_hand_angle - minute_hand_angle) (360 - (hour_hand_angle - minute_hand_angle))

theorem smaller_angle_at_7_oclock : clock_angle_at_7 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_7_oclock_l814_81448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2009_with_property_l814_81407

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year < 10000

def digits_rearrangement (year : ℕ) : Set ℕ :=
  {n : ℕ | is_valid_year n ∧ ∃ (perm : List ℕ), perm.length = 4 ∧ 
    perm.prod = (Nat.digits 10 year).prod ∧ n = perm.foldl (λ acc d => acc * 10 + d) 0}

def has_property (year : ℕ) : Prop :=
  is_valid_year year ∧ ∀ n ∈ digits_rearrangement year, n ≥ year

theorem first_year_after_2009_with_property :
  ∀ year, is_valid_year year → year > 2009 → year < 2022 → ¬(has_property year) ∧
  has_property 2022 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2009_with_property_l814_81407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sales_tax_l814_81436

/-- Calculates the amount of sales tax Bill pays given his financial information -/
theorem bill_sales_tax (gross_salary : ℕ) (take_home_salary : ℕ) (property_tax : ℕ) 
  (income_tax_rate : ℚ) (h1 : gross_salary = 50000) (h2 : take_home_salary = 40000) 
  (h3 : property_tax = 2000) (h4 : income_tax_rate = 1/10) : 
  gross_salary - take_home_salary - (income_tax_rate * gross_salary).floor - property_tax = 3000 :=
by
  -- Convert gross_salary to ℚ for multiplication with income_tax_rate
  have gross_salary_rat : ℚ := gross_salary
  -- Calculate income tax
  let income_tax := (income_tax_rate * gross_salary_rat).floor
  -- Use the given hypotheses and perform the calculation
  rw [h1, h2, h3, h4]
  norm_num
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sales_tax_l814_81436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_specific_rectangle_l814_81497

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of selecting a point (x,y) from a rectangle such that x < y --/
noncomputable def probability_x_less_than_y (r : Rectangle) : ℝ :=
  let total_area := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  let favorable_area := (min (r.x_max - r.x_min) (r.y_max - r.y_min))^2 / 2
  favorable_area / total_area

/-- The main theorem --/
theorem probability_in_specific_rectangle :
  let r : Rectangle := {
    x_min := 0
    x_max := 4
    y_min := 0
    y_max := 3
    h_x := by norm_num
    h_y := by norm_num
  }
  probability_x_less_than_y r = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_specific_rectangle_l814_81497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_cylinder_has_two_identical_views_l814_81429

-- Define the geometric bodies
inductive GeometricBody
  | Cube
  | Sphere
  | TriangularPrism
  | Cylinder
deriving DecidableEq

-- Define the types of views
inductive ViewType
  | Rectangle
  | Circle
  | Square
  | Triangle
deriving DecidableEq

-- Function to get the three views of a geometric body
def getViews (body : GeometricBody) : List ViewType :=
  match body with
  | GeometricBody.Cube => [ViewType.Square, ViewType.Square, ViewType.Square]
  | GeometricBody.Sphere => [ViewType.Circle, ViewType.Circle, ViewType.Circle]
  | GeometricBody.TriangularPrism => [ViewType.Rectangle, ViewType.Rectangle, ViewType.Triangle]
  | GeometricBody.Cylinder => [ViewType.Rectangle, ViewType.Rectangle, ViewType.Circle]

-- Function to count the number of identical views
def countIdenticalViews (views : List ViewType) : Nat :=
  let uniqueViews := views.toFinset
  if uniqueViews.card == 1 then 3
  else if uniqueViews.card == 2 then 2
  else 1

-- Theorem statement
theorem only_cylinder_has_two_identical_views :
  ∀ body : GeometricBody,
    (body = GeometricBody.Cylinder) ↔
    (countIdenticalViews (getViews body) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_cylinder_has_two_identical_views_l814_81429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_sum_zero_l814_81438

theorem circular_table_sum_zero (a b c d e : ℤ) :
  a * (e + d - b - c) + b * (a + e - c - d) + c * (b + a - d - e) +
  d * (c + b - e - a) + e * (d + c - a - b) = 0 := by
  ring  -- This tactic should solve the equation algebraically

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_sum_zero_l814_81438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_existence_l814_81470

-- Define the set M
def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1987}

-- Define a function to calculate the number of 10-term arithmetic progressions
def num_arithmetic_progressions (n : ℕ) : ℕ :=
  Finset.sum (Finset.range 221) (fun d => n - 9 * d)

-- Define the number of colors
def num_colors : ℕ := 4

-- Theorem statement
theorem coloring_existence : 
  (num_arithmetic_progressions 1987 * num_colors : ℕ) + num_colors ^ 1977 < num_colors ^ 1987 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_existence_l814_81470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l814_81422

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Cartesian coordinates in 3D space -/
structure CartesianCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Convert spherical coordinates to Cartesian coordinates -/
noncomputable def sphericalToCartesian (s : SphericalCoord) : CartesianCoord :=
  { x := s.ρ * Real.sin s.φ * Real.cos s.θ
    y := s.ρ * Real.sin s.φ * Real.sin s.θ
    z := s.ρ * Real.cos s.φ }

/-- A plane in 3D space defined by ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point in Cartesian coordinates lies on a plane -/
def isOnPlane (p : CartesianCoord) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

theorem constant_theta_is_plane (c : ℝ) :
  ∃ (plane : Plane), ∀ (s : SphericalCoord),
    s.θ = c → isOnPlane (sphericalToCartesian s) plane := by
  -- Construct the plane y = tan(c) * x
  let plane : Plane := {
    a := Real.cos c
    b := -Real.sin c
    c := 0
    d := 0
  }
  
  -- Show that this plane satisfies the condition
  exists plane
  intro s hθ
  simp [isOnPlane, sphericalToCartesian]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l814_81422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_initial_water_consumption_l814_81482

/-- Proves that the initial amount of water Kim drank is 1.5 quarts -/
theorem kim_initial_water_consumption (total_ounces : ℝ) (can_ounces : ℝ) 
  (h1 : total_ounces = 60)
  (h2 : can_ounces = 12)
  (h3 : ∀ (quarts : ℝ), quarts * 32 = quarts * 32) :
  (total_ounces - can_ounces) / 32 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_initial_water_consumption_l814_81482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rational_roots_l814_81485

/-- A polynomial with integer coefficients of the form 16x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 = 0 -/
def polynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℚ) : ℚ :=
  16 * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 24

/-- The set of possible rational roots for the polynomial -/
def possible_rational_roots : Finset ℚ :=
  {1, 2, 3, 4, 6, 8, 12, 24, 1/2, 3/2, 1/4, 3/4, 1/8, 3/8, 1/16, 3/16,
   -1, -2, -3, -4, -6, -8, -12, -24, -1/2, -3/2, -1/4, -3/4, -1/8, -3/8, -1/16, -3/16}

/-- Theorem stating that the number of different possible rational roots is at most 16 -/
theorem count_rational_roots (b₄ b₃ b₂ b₁ : ℤ) :
  (possible_rational_roots.filter (λ x => polynomial b₄ b₃ b₂ b₁ x = 0)).card ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rational_roots_l814_81485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_8_sided_die_is_4_9_l814_81439

/-- The expected value of rolling an 8-sided die with given probabilities -/
noncomputable def expected_value_8_sided_die : ℝ :=
  let p1_4 := (1 : ℝ) / 10  -- Probability for numbers 1 to 4
  let p5_8 := (3 : ℝ) / 20  -- Probability for numbers 5 to 8
  (1 * p1_4 + 2 * p1_4 + 3 * p1_4 + 4 * p1_4) +
  (5 * p5_8 + 6 * p5_8 + 7 * p5_8 + 8 * p5_8)

/-- Theorem stating that the expected value of the 8-sided die is 4.9 -/
theorem expected_value_8_sided_die_is_4_9 :
  expected_value_8_sided_die = 4.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_8_sided_die_is_4_9_l814_81439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_intersection_fixed_point_l814_81443

-- Define the triangle ABC
variable (A B C : EuclideanPlane) (X : EuclideanPlane)

-- Define the incircle of a triangle
noncomputable def incircle (P Q R : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a point on the extension of a line segment
def pointOnExtension (P Q : EuclideanPlane) (X : EuclideanPlane) : Prop := sorry

-- Define the intersection of two sets
def setIntersection (S T : Set EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a line passing through two points
noncomputable def lineThroughPoints (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a fixed point
def fixedPoint (F : EuclideanPlane) (X : EuclideanPlane) : Prop := sorry

theorem incircle_intersection_fixed_point :
  ∃ (F : EuclideanPlane),
  fixedPoint F X ∧
  ∀ (P Q : EuclideanPlane),
  P ∈ setIntersection (incircle A B X) (incircle A C X) ∧
  Q ∈ setIntersection (incircle A B X) (incircle A C X) ∧
  P ≠ Q →
  F ∈ lineThroughPoints P Q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_intersection_fixed_point_l814_81443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l814_81496

open BigOperators Nat

/-- For any positive integer n, the sum of the products of each binomial coefficient
    and its index equals n · 2^(n-1). -/
theorem binomial_sum_identity (n : ℕ) (hn : 0 < n) :
  ∑ k in Finset.range n, k * (n.choose k) = n * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l814_81496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_eccentricity_l814_81488

/-- Represents a parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- A point on both the parabola and an asymptote of the hyperbola -/
noncomputable def common_point (c₁ : Parabola) (c₂ : Hyperbola) : ℝ × ℝ := (c₁.p / 2, c₁.p * c₂.b / (2 * c₂.a))

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (c : Hyperbola) : ℝ := Real.sqrt (c.a^2 + c.b^2) / c.a

theorem parabola_hyperbola_eccentricity (c₁ : Parabola) (c₂ : Hyperbola) :
  let f := focus c₁
  let a := common_point c₁ c₂
  (a.1 - f.1 = 0) → -- AF is perpendicular to x-axis
  eccentricity c₂ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_eccentricity_l814_81488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_500_l814_81442

theorem least_integer_greater_than_sqrt_500 : ∀ n : ℤ, n > Int.floor (Real.sqrt 500) → n ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_greater_than_sqrt_500_l814_81442
