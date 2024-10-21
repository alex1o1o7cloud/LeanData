import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_class_l892_89204

/-- 
Given a class where 12% of students are absent and 44 students are present,
prove that the total number of students in the class is 50.
-/
theorem total_students_in_class : 
  let absent_percentage : ℚ := 12 / 100
  let present_students : ℕ := 44
  (present_students : ℚ) / (1 - absent_percentage) = 50 := by
  -- Unfold the definitions
  simp
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_class_l892_89204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l892_89297

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_sum (a₁ lastTerm : ℝ) (n : ℕ) :
  let d := (lastTerm - a₁) / (n - 1 : ℝ)
  let x := arithmeticSequence a₁ d (n - 2)
  let y := arithmeticSequence a₁ d (n - 1)
  x + y = 72 :=
by
  -- Proof steps would go here
  sorry

#eval arithmeticSequence 3 6 6 + arithmeticSequence 3 6 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l892_89297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_28_l892_89225

theorem sum_of_solutions_equals_28 : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, |x^2 - 14*x + 44| = 4) ∧
  (∀ x : ℝ, |x^2 - 14*x + 44| = 4 → x ∈ S) ∧
  (S.sum id) = 28 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_28_l892_89225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l892_89202

theorem problem_solution :
  ∀ (x y z : ℤ), 
    x > 0 → y > 0 → z > 0 →
    x ≥ y → y ≥ z →
    x^2 - y^2 - z^2 + x*y = 4019 →
    x^2 + 4*y^2 + 4*z^2 - 4*x*y - 3*x*z - 3*y*z = -3997 →
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l892_89202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l892_89276

/-- Definition of an ellipse with given parameters -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_b : b = 4
  h_ecc : Real.sqrt (a^2 - b^2) / a = 3/5

/-- Definition of a triangle formed by points A, B on the ellipse and focus F₂ -/
def TriangleABF₂ (e : Ellipse) : ℝ := 4 * e.a

/-- Theorem: The perimeter of triangle ABF₂ is 20 -/
theorem triangle_perimeter (e : Ellipse) :
  TriangleABF₂ e = 20 := by
  sorry

#check triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l892_89276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_disk_sales_l892_89229

/-- The number of disks Maria must sell to make a profit of $120 -/
def disks_to_sell (buy_price : ℚ) (sell_price : ℚ) (fixed_cost : ℚ) (target_profit : ℚ) : ℕ :=
  Int.natAbs (Int.ceil ((target_profit + fixed_cost) / (sell_price - buy_price)))

/-- Theorem stating the number of disks Maria must sell to make a profit of $120 -/
theorem maria_disk_sales : 
  let buy_price : ℚ := 6 / 5
  let sell_price : ℚ := 7 / 4
  let fixed_cost : ℚ := 15
  let target_profit : ℚ := 120
  disks_to_sell buy_price sell_price fixed_cost target_profit = 246 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_disk_sales_l892_89229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l892_89209

def sequence_a : ℕ → ℚ
  | 0 => 2
  | 1 => 500
  | 2 => 2000
  | (n + 3) => sequence_a (n + 2) * sequence_a (n + 2) / sequence_a n

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n > 0 ∧ ∃ k : ℕ, sequence_a n = k) ∧
  (∃ k : ℕ, sequence_a 1999 = k * 2^2000) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l892_89209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixing_theorem_l892_89230

/-- Represents the intensity of paint as a real number between 0 and 1 -/
def Intensity := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- Calculates the resulting intensity when mixing two paints -/
noncomputable def mixPaint (i1 : Intensity) (v1 : ℝ) (i2 : Intensity) (v2 : ℝ) : Intensity :=
  ⟨(i1.val * v1 + i2.val * v2) / (v1 + v2), by {
    sorry -- Proof that the result is between 0 and 1
  }⟩

theorem paint_mixing_theorem (original : Intensity) (replacement : Intensity) :
  original.val = 0.1 →
  replacement.val = 0.2 →
  (mixPaint original 0.5 replacement 0.5).val = 0.15 := by
  sorry -- Proof of the theorem

#check paint_mixing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixing_theorem_l892_89230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_proof_l892_89233

theorem sin_alpha_proof (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin (α/2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_proof_l892_89233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_alt_l892_89295

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 16

-- Define the center and radius of the first circle
def center1 : ℝ × ℝ := (1, -2)
def radius1 : ℝ := 2

-- Define the center and radius of the second circle
def center2 : ℝ × ℝ := (4, 2)
def radius2 : ℝ := 4

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  abs (radius2 - radius1) < distance_between_centers ∧ 
  distance_between_centers < radius1 + radius2 := by
  sorry

-- Theorem stating that the circles intersect (alternative formulation)
theorem circles_intersect_alt : 
  2 < 5 ∧ 5 < 6 := by
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_alt_l892_89295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l892_89289

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, the area of the triangle is √2/2. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  -- Conditions
  (Real.sqrt 2 * c - c * Real.cos A = a * Real.cos C) →
  (b + c = Real.sqrt 2 + 1) →
  (a = Real.sqrt 3) →
  -- Conclusion
  (1/2 * b * c * Real.sin A = Real.sqrt 2/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l892_89289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_bank_area_calculation_l892_89226

/-- The area of a trapezium-shaped cross-section of a river bank -/
noncomputable def riverBankArea (topWidth bottomWidth depth : ℝ) : ℝ :=
  (1/2) * (topWidth + bottomWidth) * depth

theorem river_bank_area_calculation :
  let topWidth : ℝ := 12
  let bottomWidth : ℝ := 8
  let depth : ℝ := 50
  riverBankArea topWidth bottomWidth depth = 500 := by
  -- Unfold the definition of riverBankArea
  unfold riverBankArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_bank_area_calculation_l892_89226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_performance_smaller_total_sum_squares_l892_89286

/-- Represents a linear regression model -/
structure LinearRegressionModel where
  totalSumSquares : ℝ
  sumSquaresResiduals : ℝ
  sumSquaresRegression : ℝ

/-- Indicates that one model has better performance than another -/
def BetterPerformance (model1 model2 : LinearRegressionModel) : Prop := 
  model1.totalSumSquares < model2.totalSumSquares

/-- The sum of squares due to regression equals the total sum of squares of deviations 
    minus the sum of squares of residuals -/
axiom sum_squares_relation (model : LinearRegressionModel) :
  model.sumSquaresRegression = model.totalSumSquares - model.sumSquaresResiduals

/-- A smaller total sum of squares of deviations indicates better regression performance -/
theorem better_performance_smaller_total_sum_squares 
  (model1 model2 : LinearRegressionModel) 
  (h : model1.totalSumSquares < model2.totalSumSquares) :
  BetterPerformance model1 model2 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_performance_smaller_total_sum_squares_l892_89286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_snail_race_time_l892_89223

/-- Represents the time taken by a snail to complete the race -/
noncomputable def race_time (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

theorem third_snail_race_time :
  let first_snail_speed : ℝ := 2
  let second_snail_speed : ℝ := 2 * first_snail_speed
  let third_snail_speed : ℝ := 5 * second_snail_speed
  let first_snail_time : ℝ := 20
  let race_distance : ℝ := first_snail_speed * first_snail_time
  race_time third_snail_speed race_distance = 2 := by
  -- Unfold the definitions
  unfold race_time
  -- Simplify the expression
  simp
  -- The actual proof steps would go here
  sorry

#check third_snail_race_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_snail_race_time_l892_89223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_through_focus_l892_89280

noncomputable section

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, if a chord CD passes through a focus F and CF = 2, then DF = 2 -/
theorem ellipse_chord_through_focus
  (e : Ellipse)
  (C D F : Point)
  (h1 : e.a = 6 ∧ e.b = 4)
  (h2 : isOnEllipse C e ∧ isOnEllipse D e)
  (h3 : F.x = 2 * Real.sqrt 5 ∧ F.y = 0)
  (h4 : distance C F = 2)
  (h5 : ∃ t : ℝ, D = Point.mk (F.x + t * (C.x - F.x)) (F.y + t * (C.y - F.y)))
  : distance D F = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_through_focus_l892_89280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleanup_probability_l892_89253

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ :=
  ↑favorableOutcomes / ↑totalOutcomes

theorem cleanup_probability :
  let totalStudents : ℕ := 5
  let studentsToSelect : ℕ := 2
  let studentsExcluded : ℕ := 2
  let totalOutcomes : ℕ := choose totalStudents studentsToSelect
  let favorableOutcomes : ℕ := choose (totalStudents - studentsExcluded) studentsToSelect
  probability favorableOutcomes totalOutcomes = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleanup_probability_l892_89253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_value_l892_89293

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_max_value (a₁ d : ℝ) :
  (∀ n m : ℕ, n ≠ m → arithmetic_sequence a₁ d n ≠ arithmetic_sequence a₁ d m) →
  (∀ n : ℕ, arithmetic_sequence a₁ d (2 * n) = 2 * arithmetic_sequence a₁ d n - 3) →
  (arithmetic_sequence a₁ d 6)^2 = arithmetic_sequence a₁ d 1 * arithmetic_sequence a₁ d 21 →
  (∃ n : ℕ, S a₁ d n / 2^(n - 1) = 6) ∧
  (∀ n : ℕ, S a₁ d n / 2^(n - 1) ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_value_l892_89293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l892_89296

/-- A dilation in the complex plane -/
def complexDilation (center : ℂ) (factor : ℝ) (z : ℂ) : ℂ :=
  center + factor • (z - center)

/-- The problem statement -/
theorem dilation_problem : 
  let center : ℂ := 1 + 2*Complex.I
  let factor : ℝ := 2
  let z : ℂ := 2 + Complex.I
  complexDilation center factor z = 3 := by
  -- Unfold the definitions
  simp [complexDilation]
  -- Perform the complex arithmetic
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l892_89296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_statements_island_l892_89228

/-- Represents the type of inhabitant -/
inductive InhabitantType
  | Knight
  | Liar
deriving BEq, Repr

/-- Represents a statement made by an inhabitant about their partner -/
inductive Statement
  | IsKnight
  | IsLiar
deriving BEq, Repr

/-- A pair of inhabitants -/
structure Pair where
  first : InhabitantType
  second : InhabitantType

/-- The statement made by an inhabitant based on their type and their partner's type -/
def makeStatement (speaker : InhabitantType) (partner : InhabitantType) : Statement :=
  match speaker, partner with
  | InhabitantType.Knight, InhabitantType.Knight => Statement.IsKnight
  | InhabitantType.Knight, InhabitantType.Liar => Statement.IsLiar
  | InhabitantType.Liar, InhabitantType.Knight => Statement.IsKnight
  | InhabitantType.Liar, InhabitantType.Liar => Statement.IsKnight

theorem unequal_statements_island (inhabitants : List Pair) 
  (h1 : inhabitants.length = 617)
  (h2 : ∀ p ∈ inhabitants, p.first ≠ p.second → 
        (makeStatement p.first p.second = Statement.IsKnight ∧ 
         makeStatement p.second p.first = Statement.IsLiar) ∨
        (makeStatement p.first p.second = Statement.IsLiar ∧ 
         makeStatement p.second p.first = Statement.IsKnight)) :
  ¬((inhabitants.map (λ p => makeStatement p.first p.second)).count Statement.IsKnight = 
    (inhabitants.map (λ p => makeStatement p.first p.second)).count Statement.IsLiar) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_statements_island_l892_89228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_special_triangle_l892_89231

theorem cosine_in_special_triangle (A B : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < a ∧ 0 < b →  -- Ensure positive angles and side lengths
  b = (5 / 8) * a →                -- Given condition
  A = 2 * B →                      -- Given condition
  Real.cos A = 7 / 25 := by        -- Theorem to prove
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_special_triangle_l892_89231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l892_89235

noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else 3 * x - 6

theorem h_triple_equality (b : ℝ) :
  b < 0 →
  h (h (h (-0.5))) = h (h (h b)) ↔ b = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l892_89235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_gas_mileage_l892_89260

/-- Calculates the average gas mileage for a round trip given the distances and fuel efficiencies of each leg --/
noncomputable def average_gas_mileage (distance1 distance2 mpg1 mpg2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / mpg1 + distance2 / mpg2)

/-- The average gas mileage for the round trip is approximately 18 miles per gallon --/
theorem round_trip_gas_mileage :
  let distance1 : ℝ := 150
  let distance2 : ℝ := 180
  let mpg1 : ℝ := 25
  let mpg2 : ℝ := 15
  ∃ ε > 0, |average_gas_mileage distance1 distance2 mpg1 mpg2 - 18| < ε :=
by
  sorry

-- Remove the #eval line as it's not computable
-- #eval average_gas_mileage 150 180 25 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_gas_mileage_l892_89260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_order_approx_sqrt_and_cube_root_l892_89288

/-- Definition of k-order linear approximation --/
def k_order_linear_approximation (f : ℝ → ℝ) (a b k : ℝ) : Prop :=
  ∀ x l : ℝ, x = l * a + (1 - l) * b →
    |f x - (l * f a + (1 - l) * f b)| ≤ k

/-- The main theorem --/
theorem k_order_approx_sqrt_and_cube_root (k : ℝ) :
  (k_order_linear_approximation Real.sqrt 0 1 k ∧
   k_order_linear_approximation (fun x ↦ Real.rpow x (1/3)) 0 1 k) ↔
  1/4 ≤ k ∧ k < 2 * Real.sqrt 3 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_order_approx_sqrt_and_cube_root_l892_89288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l892_89262

-- Define the function f(x) = 1 - 2^x
noncomputable def f (x : ℝ) : ℝ := 1 - 2^x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Iio 1 := by sorry

-- Note: Set.Iio 1 represents the open interval (-∞, 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l892_89262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_triangle_sides_l892_89283

/-- Given a triangle ABC with sides BC = 3, AC = 4, and AB = 5, 
    the minimum value of PA · PB for any point P on the sides of the triangle is 25/64 -/
theorem min_dot_product_on_triangle_sides (A B C P : ℝ × ℝ) : 
  let d := (λ (X Y : ℝ × ℝ) ↦ Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2))
  (d B C = 3 ∧ d A C = 4 ∧ d A B = 5) →
  (∃ t : ℝ, (0 ≤ t ∧ t ≤ 1) ∧ 
    (P = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∨
     P = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2) ∨
     P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2))) →
  25/64 ≤ ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_triangle_sides_l892_89283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_gain_percentage_l892_89281

noncomputable def original_price : ℚ := 51724.14
def loss_percentage : ℚ := 13
noncomputable def final_sale_price : ℚ := 54000

noncomputable def man_sale_price : ℚ := original_price * (1 - loss_percentage / 100)

theorem friend_gain_percentage :
  (final_sale_price - man_sale_price) / man_sale_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_gain_percentage_l892_89281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_fractions_l892_89244

variable (x : ℝ)

theorem lcm_fractions (hx : x ≠ 0) :
  let f1 := 1 / (2 * x)
  let f2 := 1 / (4 * x)
  let f3 := 1 / (6 * x)
  let f4 := 1 / (12 * x)
  let lcm := 1 / (12 * x)
  (∀ k : ℝ, (∃ n : ℤ, k * f1 = n) ∧ (∃ n : ℤ, k * f2 = n) ∧ (∃ n : ℤ, k * f3 = n) ∧ (∃ n : ℤ, k * f4 = n) → k ≥ lcm) ∧
  ((∃ n : ℤ, lcm * f1 = n) ∧ (∃ n : ℤ, lcm * f2 = n) ∧ (∃ n : ℤ, lcm * f3 = n) ∧ (∃ n : ℤ, lcm * f4 = n)) :=
by
  sorry

#check lcm_fractions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_fractions_l892_89244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_doubles_in_nine_years_l892_89236

-- Define the annual growth rate
def growth_rate : ℝ := 0.08

-- Define the approximation for lg 2
def lg2_approx : ℝ := 0.3010

-- Define the approximation for lg 1.08
def lg1_08_approx : ℝ := 0.0334

-- Define the function to calculate the number of years for GDP to double
noncomputable def years_to_double (growth_rate : ℝ) : ℝ :=
  lg2_approx / (Real.log (1 + growth_rate) / Real.log 10)

-- Theorem stating that the number of years for GDP to double is approximately 9
theorem gdp_doubles_in_nine_years :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |years_to_double growth_rate - 9| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_doubles_in_nine_years_l892_89236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l892_89245

/-- The nth term of the sequence -/
def a (n : ℕ) : ℝ := (3^n)^(1/(3^n))

/-- The infinite product -/
noncomputable def infinite_product : ℝ := ∏' n, a n

/-- Theorem stating the equality of the infinite product and 3^(3/4) -/
theorem infinite_product_equals_three_to_three_fourths :
  infinite_product = 3^(3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l892_89245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l892_89255

/-- The acceleration of a plate on two cylindrical rollers. -/
theorem plate_acceleration (R r m α g : Real) (hR : R = 1) (hr : r = 0.75) 
  (hm : m = 75) (hα : α = Real.arccos 0.98) (hg : g = 10) :
  let a := g * Real.sin (α / 2)
  (a = 1 ∧ Real.arcsin (Real.sin (α / 2)) = Real.arcsin 0.1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l892_89255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harvest_season_duration_l892_89256

/-- Calculates the number of weeks in the harvest season given weekly earnings, weekly rent, and total earnings. -/
def harvest_weeks (weekly_earnings : ℕ) (weekly_rent : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / (weekly_earnings - weekly_rent))

/-- Theorem stating that given the specified earnings and rent, the harvest season lasts 265 weeks. -/
theorem harvest_season_duration :
  harvest_weeks 403 49 93899 = 265 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harvest_season_duration_l892_89256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l892_89212

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l892_89212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_proof_l892_89290

/-- Calculate the present value of an investment --/
noncomputable def present_value (future_value : ℝ) (interest_rate : ℝ) (time : ℝ) (compounding_periods : ℝ) : ℝ :=
  future_value / (1 + interest_rate / compounding_periods) ^ (compounding_periods * time)

/-- Prove that the initial investment needed is approximately $392,946.77 --/
theorem initial_investment_proof :
  let future_value : ℝ := 750000
  let interest_rate : ℝ := 0.045
  let time : ℝ := 15
  let compounding_periods : ℝ := 2
  let calculated_pv := present_value future_value interest_rate time compounding_periods
  ∃ ε > 0, |calculated_pv - 392946.77| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_proof_l892_89290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l892_89285

/-- A regular triangular pyramid with height h and all plane angles at the apex being right angles -/
structure RegularTriangularPyramid where
  h : ℝ
  apex_angles_right : True

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ := p.h^3 * Real.sqrt 3 / 2

/-- Theorem: The volume of a regular triangular pyramid with height h and all plane angles 
    at the apex being right angles is equal to (h^3 * √3) / 2 -/
theorem regular_triangular_pyramid_volume (p : RegularTriangularPyramid) :
  volume p = p.h^3 * Real.sqrt 3 / 2 := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l892_89285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_impossibility_l892_89299

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Represents the state of the three ants -/
structure AntState where
  a : Point
  b : Point
  c : Point

/-- Defines a valid move for the ants -/
def isValidMove (s1 s2 : AntState) : Prop :=
  (s1.a = s2.a ∧ s1.b = s2.b) ∨ (s1.a = s2.a ∧ s1.c = s2.c) ∨ (s1.b = s2.b ∧ s1.c = s2.c)

/-- Defines a sequence of valid moves -/
def isValidMoveSequence : List AntState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

/-- The main theorem stating the impossibility of reaching the final state -/
theorem ant_movement_impossibility : 
  ∀ (moves : List AntState),
    isValidMoveSequence moves →
    (moves.head? = some (AntState.mk (Point.mk 0 0) (Point.mk 0 1) (Point.mk 1 0))) →
    (moves.getLast? ≠ some (AntState.mk (Point.mk (-1) 0) (Point.mk 0 1) (Point.mk 1 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_impossibility_l892_89299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_prime_l892_89232

/-- The sequence defined by the recurrence relation -/
def recurrenceSequence (x₀ a b : ℕ) : ℕ → ℕ
  | 0 => x₀
  | n + 1 => recurrenceSequence x₀ a b n * a + b

/-- Theorem stating that the sequence cannot be entirely prime -/
theorem not_all_prime (x₀ a b : ℕ) : ∃ k : ℕ, ¬ Nat.Prime (recurrenceSequence x₀ a b k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_prime_l892_89232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_graph_l892_89270

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x)

-- Theorem stating that g is the result of transforming f
theorem transform_sin_graph (x : ℝ) : g x = f (3 * x) := by
  -- Unfold the definitions of f and g
  unfold f g
  -- The equality holds by definition
  rfl

-- Note: The original statement was incorrect. 
-- We're not proving g x = 3 * f x, but rather g x = f (3 * x)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_graph_l892_89270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AR_range_l892_89234

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define point A (right vertex of C)
def point_A : ℝ × ℝ := (2, 0)

-- Define point B
def point_B : ℝ × ℝ := (1/2, 0)

-- Define a line passing through B and intersecting C at P and Q
def line_PQ (m : ℝ) (y : ℝ) : ℝ := m * y + 1/2

-- Define the midpoint R of PQ
noncomputable def point_R (m : ℝ) : ℝ × ℝ :=
  (line_PQ m (-3*m/(2*(3*m^2 + 4))), -3*m/(2*(3*m^2 + 4)))

-- Define the slope of line AR
noncomputable def slope_AR (m : ℝ) : ℝ :=
  (point_R m).2 / ((point_R m).1 - 2)

-- Theorem statement
theorem slope_AR_range :
  ∀ m : ℝ, -1/8 ≤ slope_AR m ∧ slope_AR m ≤ 1/8 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_AR_range_l892_89234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_std_dev_from_mean_l892_89258

/-- Calculates the number of standard deviations a value is from the mean -/
noncomputable def zScore (x μ σ : ℝ) : ℝ := (x - μ) / σ

theorem value_std_dev_from_mean (μ σ x : ℝ) (hμ : μ = 15.5) (hσ : σ = 1.5) (hx : x = 12.5) :
  zScore x μ σ = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_std_dev_from_mean_l892_89258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_C_coordinates_l892_89272

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def dist_sq (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Defines an isosceles right triangle -/
structure IsoscelesRightTriangle where
  A : Point
  B : Point
  C : Point
  right_angle : dist_sq A B + dist_sq A C = dist_sq B C
  isosceles : dist_sq A C = dist_sq B C

theorem isosceles_right_triangle_C_coordinates :
  ∃ (C : Point), (C.x = 2 ∧ C.y = 3) ∨ (C.x = 4 ∧ C.y = -1) ∧
    ∃ (t : IsoscelesRightTriangle), t.A = { x := 1, y := 0 } ∧ t.B = { x := 3, y := 1 } ∧ t.C = C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_C_coordinates_l892_89272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_theorem_l892_89227

-- Define a cube face
inductive Face : Type
| One | Two | Three | Four | Five | Six

-- Define a function to convert Face to Nat
def faceToNat : Face → Nat
| Face.One   => 1
| Face.Two   => 2
| Face.Three => 3
| Face.Four  => 4
| Face.Five  => 5
| Face.Six   => 6

-- Define a cube
structure Cube :=
  (top : Face)
  (front : Face)
  (right : Face)
  (bottom : Face)
  (back : Face)
  (left : Face)

-- Define the property that opposite faces sum to 7
def oppositeFacesSum7 (c : Cube) : Prop :=
  (faceToNat c.top + faceToNat c.bottom = 7) ∧
  (faceToNat c.front + faceToNat c.back = 7) ∧
  (faceToNat c.left + faceToNat c.right = 7)

-- Define the roll right operation
def rollRight (c : Cube) : Cube :=
  { top := c.left,
    front := c.front,
    right := c.top,
    bottom := c.right,
    back := c.back,
    left := c.bottom }

-- Define the roll down operation
def rollDown (c : Cube) : Cube :=
  { top := c.back,
    front := c.top,
    right := c.right,
    bottom := c.front,
    back := c.bottom,
    left := c.left }

-- Define the paths
def path1 := [rollRight, rollDown, rollRight, rollDown]
def path2 := [rollRight, rollDown, rollRight, rollRight]

-- The theorem to prove
theorem cube_roll_theorem (c : Cube) (h : oppositeFacesSum7 c) :
  (c.top = Face.Six) →
  ((List.foldl (fun acc f => f acc) c path1).top = Face.Two) ∧
  ((List.foldl (fun acc f => f acc) c path2).top = Face.One) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_theorem_l892_89227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l892_89257

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 3) / Real.log a

theorem function_monotonicity 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1) 
  (h3 : f a 0 < 0) :
  ∃ (l r : ℝ), l = -1 ∧ r = 1 ∧ 
    ∀ x y, l ≤ x ∧ x < y ∧ y < r → f a x < f a y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l892_89257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cotangent_equality_l892_89298

theorem sqrt_cotangent_equality (x : ℝ) :
  (Real.sqrt (1 + Real.cos x / Real.sin x) = Real.sin x + Real.cos x) →
  (Real.sin x ≠ 0) →
  (Real.cos x ≠ 0) →
  (∃ n : ℤ, x = π / 4 + 2 * π * ↑n ∨ x = -π / 4 + 2 * π * ↑n ∨ x = 3 * π / 4 + 2 * π * ↑n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cotangent_equality_l892_89298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_special_triangle_third_side_l892_89208

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  ab : ℝ
  ac : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the specific triangle from the problem
def specialTriangle : Triangle :=
  { A := (0, 0),
    B := (10, 0),
    C := (0, 0),  -- Placeholder, exact position not necessary
    ab := 10,
    ac := 20,
    angleB := 3 * (1 : ℝ),  -- Placeholder, exact angle not given
    angleC := 1  -- Placeholder, exact angle not given
  }

-- Theorem statement
theorem third_side_length (t : Triangle) (h1 : t.ab = 10) (h2 : t.ac = 20) (h3 : t.angleB = 3 * t.angleC) :
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) = 10 * Real.sqrt 3 := by
  sorry

-- Application to our specific triangle
theorem special_triangle_third_side :
  Real.sqrt ((specialTriangle.B.1 - specialTriangle.C.1)^2 + (specialTriangle.B.2 - specialTriangle.C.2)^2) = 10 * Real.sqrt 3 := by
  apply third_side_length specialTriangle
  · rfl
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_special_triangle_third_side_l892_89208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l892_89239

/-- An ellipse with given eccentricity and passing through a specific point --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 2/3
  h_point : a^2 * 1^2 + b^2 * (Real.sqrt 3)^2 = a^2 * b^2

/-- The line passing through the right focus of the ellipse --/
structure FocusLine (e : Ellipse) where
  m : ℝ
  c : ℝ
  h_focus : c = Real.sqrt (e.a^2 - e.b^2)

/-- Triangle formed by origin and intersection points of ellipse and focus line --/
noncomputable def triangleArea (e : Ellipse) (l : FocusLine e) : ℝ :=
  2 * Real.sqrt 6 * Real.sqrt (l.m^2 + 1) / (l.m^2 + 3)

/-- Main theorem statement --/
theorem ellipse_properties (e : Ellipse) :
  (∀ x y, x^2/6 + y^2/2 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1) ∧
  (∃ l : FocusLine e, ∀ l' : FocusLine e, triangleArea e l ≥ triangleArea e l') ∧
  triangleArea e { m := 1, c := Real.sqrt (e.a^2 - e.b^2), h_focus := by rfl } = Real.sqrt 3 ∧
  triangleArea e { m := -1, c := Real.sqrt (e.a^2 - e.b^2), h_focus := by rfl } = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l892_89239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l892_89200

theorem vector_angle_cosine (a b : ℝ × ℝ) :
  a • (a + b) = 3 →
  ‖a‖ = 2 →
  ‖b‖ = 1 →
  a • b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l892_89200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_length_problem_l892_89213

open Nat Real InnerProductSpace

theorem vector_length_problem (a b : EuclideanSpace ℝ (Fin 3)) :
  let angle := Real.arccos (inner a b / (norm a * norm b))
  angle = π / 3 ∧ 
  norm b = 1 ∧ 
  norm (a - 2 • b) = Real.sqrt 7 →
  norm a = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_length_problem_l892_89213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_of_ten_l892_89252

theorem smallest_exponent_of_ten (a b c d : ℕ+) (m n : ℕ) : 
  a.val + b.val + c.val + d.val = 4014 →
  (a.val.factorial * b.val.factorial * c.val.factorial * d.val.factorial : ℕ) = m * 10^n →
  ¬(10 ∣ m) →
  n ≥ 1000 ∧ ∃ (a' b' c' d' : ℕ+) (m' : ℕ), 
    a'.val + b'.val + c'.val + d'.val = 4014 ∧
    (a'.val.factorial * b'.val.factorial * c'.val.factorial * d'.val.factorial : ℕ) = m' * 10^1000 ∧
    ¬(10 ∣ m') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_of_ten_l892_89252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_floor_f_l892_89279

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := 1/2 - Real.exp x / (1 + Real.exp x)

theorem range_of_sum_floor_f (x : ℝ) : 
  (floor (f x) + floor (f (-x))) ∈ ({-1, 0} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_floor_f_l892_89279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_896_to_hundredth_l892_89207

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The theorem states that rounding 3.896 to the nearest hundredth results in 3.90 -/
theorem round_3_896_to_hundredth :
  round_to_hundredth 3.896 = 3.90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_896_to_hundredth_l892_89207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₂_increasing_on_reals_l892_89219

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.exp (-x)
def f₂ (x : ℝ) : ℝ := x^3
noncomputable def f₃ (x : ℝ) : ℝ := Real.log x
def f₄ (x : ℝ) : ℝ := |x|

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem only_f₂_increasing_on_reals :
  (IsIncreasing f₂) ∧
  (¬IsIncreasing f₁) ∧
  (∃ x, f₃ x = 0) ∧  -- log is not defined for non-positive reals
  (¬IsIncreasing f₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₂_increasing_on_reals_l892_89219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_exercise_minutes_l892_89264

/-- The number of minutes Hannah needs to exercise on the tenth day to achieve an average of 110 minutes per day over a 10-day period, given her exercise routine for the first 9 days. -/
theorem hannah_exercise_minutes 
  (days_2hours : ℕ) 
  (days_1hour45min : ℕ) 
  (target_average : ℕ) 
  (total_days : ℕ) 
  (minutes_per_hour : ℕ) 
  (x : ℕ) :
  days_2hours = 4 →
  days_1hour45min = 5 →
  target_average = 110 →
  total_days = 10 →
  minutes_per_hour = 60 →
  (days_2hours * 2 * minutes_per_hour + 
   days_1hour45min * (1 * minutes_per_hour + 45) + 
   (total_days - days_2hours - days_1hour45min) * x) / total_days = target_average →
  x = 95 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_exercise_minutes_l892_89264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l892_89266

noncomputable def f (x : ℝ) : ℝ := -Real.sin (Real.pi * x / 4)

theorem f_satisfies_conditions :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 2) = f (-x + 2)) ∧  -- f(x+2) is even
  (∀ x₁ x₂, x₁ ∈ Set.Ioo 0 2 → x₂ ∈ Set.Ioo 0 2 → x₁ ≠ x₂ →
    (f x₁ - f x₂) / (x₁ - x₂) < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l892_89266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l892_89224

noncomputable section

variable (f : ℝ → ℝ)

axiom f_deriv : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_initial : f 0 = 3

theorem solution_set (x : ℝ) : 
  (Real.exp x * f x > Real.exp x + 2) ↔ (x > 0) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l892_89224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l892_89247

/-- The volume of a cone given its diameter and height -/
noncomputable def cone_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (diameter / 2)^2 * height

/-- Theorem: The volume of a cone with diameter 12 cm and height 9 cm is 108π cubic centimeters -/
theorem cone_volume_specific : cone_volume 12 9 = 108 * Real.pi := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l892_89247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l892_89263

-- Define the parameters
noncomputable def speed : ℝ := 12 -- km/hr
noncomputable def time : ℝ := 8/60 -- hours (8 minutes)
noncomputable def length_to_breadth_ratio : ℝ := 1/4

-- Define the theorem
theorem park_area (perimeter : ℝ) (length : ℝ) (breadth : ℝ) :
  perimeter = speed * time * 1000 →  -- Convert km to m
  perimeter = 2 * (length + breadth) →
  breadth = length / length_to_breadth_ratio →
  length * breadth = 102400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l892_89263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l892_89211

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclinationAngle (a b : ℝ) : ℝ := Real.arctan (-a / b)

/-- Prove that the inclination angle of the line x - √3y + 3 = 0 is π/6 -/
theorem line_inclination_angle : 
  inclinationAngle 1 (-Real.sqrt 3) = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l892_89211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l892_89271

theorem expression_value : 
  (0.064 : ℝ)^(-(1/3 : ℝ)) - (-7/8 : ℝ)^(0 : ℝ) + 16^(3/4 : ℝ) + (0.01 : ℝ)^(1/2 : ℝ) = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l892_89271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_female_pair_approx_l892_89216

-- Define the number of men and women
def num_men : ℕ := 7
def num_women : ℕ := 7

-- Define the total number of people
def total_people : ℕ := num_men + num_women

-- Define the number of pairs
def num_pairs : ℕ := total_people / 2

-- Define the probability function
noncomputable def probability_at_least_one_female_pair : ℚ :=
  1 - (Nat.choose num_women 0 * Nat.choose num_men num_pairs) / Nat.choose total_people (2 * num_pairs)

-- Theorem statement
theorem probability_female_pair_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((probability_at_least_one_female_pair : ℝ) - 0.96) < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_female_pair_approx_l892_89216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_logarithms_count_l892_89277

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a function to count distinct logarithmic values
def countDistinctLogarithms (S : Finset ℕ) : ℕ :=
  let baseAntilogPairs := S.filter (· ≠ 1) ×ˢ S.filter (· ≠ 1)
  let distinctPairs := baseAntilogPairs.filter (fun p => p.1 ≠ p.2)
  let logWithOneAsAntilog := S.filter (· ≠ 1)
  distinctPairs.card + logWithOneAsAntilog.card

-- Theorem statement
theorem distinct_logarithms_count :
  countDistinctLogarithms S = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_logarithms_count_l892_89277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l892_89201

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the left focus F
def F : ℝ × ℝ := (-2, 0)

-- Define a line passing through F
def line_AB (m : ℝ) (y : ℝ) : ℝ := m * y - 2

-- Define points A and B on the hyperbola
def point_on_hyperbola (x y : ℝ) : Prop :=
  hyperbola x y ∧ ∃ m : ℝ, x = line_AB m y

-- Define tangent lines l1 and l2
def tangent_line (x y : ℝ) (x' y' : ℝ) : Prop :=
  x * x' - y * y' / 3 = 1

-- Define point M as the intersection of l1 and l2
def point_M (x y : ℝ) : Prop :=
  ∃ (x_A y_A x_B y_B : ℝ),
    point_on_hyperbola x_A y_A ∧
    point_on_hyperbola x_B y_B ∧
    tangent_line x_A y_A x y ∧
    tangent_line x_B y_B x y

-- Helper function to calculate triangle area
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))

-- Theorem statement
theorem hyperbola_properties :
  ∀ (x_M y_M : ℝ),
    point_M x_M y_M →
    (∃ (m : ℝ), (x_M + 2) * m = y_M) ∧  -- MF ⟂ AB
    (∀ (x_A y_A x_B y_B : ℝ),
      point_on_hyperbola x_A y_A →
      point_on_hyperbola x_B y_B →
      tangent_line x_A y_A x_M y_M →
      tangent_line x_B y_B x_M y_M →
      area_triangle x_A y_A x_B y_B x_M y_M ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l892_89201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l892_89261

theorem complex_equality (a : ℝ) : 
  ((1 + 2*Complex.I) * (a + Complex.I)).re = 
  ((1 + 2*Complex.I) * (a + Complex.I)).im → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l892_89261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_open_interval_l892_89237

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x - 3 * a
  else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_in_open_interval :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  1 < a ∧ a < 2 :=
by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_open_interval_l892_89237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l892_89250

noncomputable section

open Real

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  (a + b) * (sin B - sin A) = sin C * (Real.sqrt 3 * a + c) →
  B = 5 * π / 6 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l892_89250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_30_l892_89287

/-- r_8(n) is the remainder when n is divided by 8 -/
def r_8 (n : ℕ) : ℕ := n % 8

/-- The sequence of nonnegative integers n satisfying r_8(7n) ≤ 3 -/
def seq : List ℕ := (List.range 32).filter (fun n => r_8 (7 * n) ≤ 3)

/-- The 15th term of the sequence is 30 -/
theorem fifteenth_term_is_30 : seq[14]? = some 30 := by
  -- Evaluate seq explicitly
  have h : seq = [0, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31] := by rfl
  -- Use this equality to show the 15th term (index 14) is 30
  rw [h]
  rfl

#eval seq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_30_l892_89287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manhattan_to_bronx_travel_time_l892_89203

/-- Represents the total travel time from Manhattan to the Bronx -/
noncomputable def total_travel_time (subway_speed subway_distance train_speed bus_speed bike_speed bike_time
                       layover1 layover2 layover3 : ℝ) : ℝ :=
  let train_distance := 2 * subway_distance
  let bus_distance := 0.5 * train_distance
  subway_distance / subway_speed +
  layover1 / 60 +
  train_distance / train_speed +
  layover2 / 60 +
  bus_distance / bus_speed +
  layover3 / 60 +
  bike_time

/-- The total travel time from Manhattan to the Bronx is approximately 14.333 hours -/
theorem manhattan_to_bronx_travel_time :
  ∃ ε > 0, |total_travel_time 20 30 45 15 10 8 30 45 15 - 14.333| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manhattan_to_bronx_travel_time_l892_89203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_cowboy_to_river_to_cabin_l892_89243

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflects a point across the line y = x -/
def reflect_across_y_eq_x (p : Point) : Point :=
  ⟨p.y, p.x⟩

/-- The cowboy's starting position -/
def cowboy_start : Point :=
  ⟨0, -5⟩

/-- The cabin's position relative to the cowboy's starting position -/
def cabin : Point :=
  ⟨10, -12⟩

/-- Theorem stating the shortest distance the cowboy can travel -/
theorem shortest_distance_cowboy_to_river_to_cabin :
  ∃ (river_point : Point),
    river_point.y = river_point.x ∧
    distance cowboy_start river_point + distance river_point cabin =
    5 + Real.sqrt 369 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_cowboy_to_river_to_cabin_l892_89243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_and_subset_l892_89221

-- Define the functions and their domains
noncomputable def f (x : ℝ) := Real.log (x^2 - x - 2) / Real.log 10
def A : Set ℝ := {x | x^2 - x - 2 > 0}

noncomputable def g (x : ℝ) := Real.sqrt (3 - abs x)
def B : Set ℝ := {x | 3 - abs x ≥ 0}

def C (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

-- State the theorem
theorem domain_intersection_and_subset (m : ℝ) : 
  (A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3}) ∧ 
  (C m ⊆ B ↔ m ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_and_subset_l892_89221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l892_89284

theorem pizza_area_increase (s : ℝ) (h : s > 0) : 
  (((1.2 * s) * (1.25 * s) - s^2) / s^2) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l892_89284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_max_l892_89268

theorem cube_face_product_sum_max :
  ∀ (a b c d e f : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  ({a, b, c, d, e, f} : Finset ℕ) = {2, 3, 4, 5, 6, 7} →
  (a + b) * (c + d) * (e + f) ≤ 729 :=
by
  intros a b c d e f h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_max_l892_89268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l892_89249

-- Define the points P and M
noncomputable def P : ℝ × ℝ := (1, 2)
noncomputable def M : ℝ × ℝ := (0, 3)

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - M.1)^2 + (y - M.2)^2 = 3^2

-- Theorem statement
theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
    C A.1 A.2 ∧ C B.1 B.2 ∧
    (∃ t₁ t₂ : ℝ, l t₁ = A ∧ l t₂ = B) ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) *
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l892_89249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postcards_leftover_proof_l892_89275

/-- The number of postcards left over when a total number of postcards are bundled into groups of a given size -/
def postcards_leftover (total : Int) (bundle_size : Int) : Int :=
  total % bundle_size

/-- Proof that 123 postcards bundled into groups of 15 leave 3 postcards leftover -/
theorem postcards_leftover_proof (total : Int) (bundle_size : Int) 
  (h1 : total = 123) (h2 : bundle_size = 15) : 
  postcards_leftover total bundle_size = 3 := by
  rw [postcards_leftover, h1, h2]
  norm_num

#eval postcards_leftover 123 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_postcards_leftover_proof_l892_89275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_theta_l892_89248

theorem tan_pi_4_plus_theta (θ : Real) :
  Real.tan (π / 4 + θ) = 3 → Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_theta_l892_89248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l892_89254

-- Define the sets P and Q
def P : Set ℝ := {x | x - 5*x + 4 < 0}
def Q : Set ℝ := {y | ∃ x, y = Real.sqrt (4 - 2^x)}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l892_89254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_lateral_area_l892_89210

/-- A prism with a regular triangular base and lateral edges perpendicular to the base. -/
structure TriangularPrism where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The height of the prism -/
  height : ℝ

/-- A sphere that touches all faces of the prism -/
structure InscribedSphere (p : TriangularPrism) where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The sphere touches all faces of the prism -/
  touches_all_faces : True

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The lateral surface area of a triangular prism -/
def lateral_surface_area (p : TriangularPrism) : ℝ := 3 * p.base_side * p.height

/-- 
  If a sphere with volume 4π/3 touches all faces of a triangular prism, 
  then the lateral surface area of the prism is 12√3.
-/
theorem prism_lateral_area (p : TriangularPrism) (s : InscribedSphere p) 
  (h : sphere_volume s.radius = (4 / 3) * Real.pi) : 
  lateral_surface_area p = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_lateral_area_l892_89210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_36_l892_89238

/-- The amount of credit extended by automobile finance companies in billions of dollars -/
noncomputable def auto_finance_credit : ℝ := 35

/-- The total consumer installment credit outstanding in billions of dollars -/
noncomputable def total_consumer_credit : ℝ := 291.6666666666667

/-- The fraction of automobile installment credit extended by automobile finance companies -/
noncomputable def auto_finance_fraction : ℝ := 1/3

/-- The percentage of consumer installment credit accounted for by automobile installment credit -/
noncomputable def auto_credit_percentage : ℝ := (auto_finance_credit / auto_finance_fraction) / total_consumer_credit * 100

theorem auto_credit_percentage_approx_36 :
  abs (auto_credit_percentage - 36) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_36_l892_89238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_marking_theorem_l892_89259

/-- A configuration of marks on a stick -/
def MarkConfiguration := List Nat

/-- Check if all distances from 1 to n appear exactly once in the configuration -/
def validConfiguration (config : MarkConfiguration) (n : Nat) : Prop :=
  ∀ d : Nat, d ≤ n → (∃! (i j : Nat), i < j ∧ j < config.length ∧ 
    (config.get? j).isSome ∧ (config.get? i).isSome ∧ 
    (config.get? j).get! - (config.get? i).get! = d)

/-- The theorem stating that only n = 3 and n = 6 are valid -/
theorem stick_marking_theorem :
  ∀ n : Nat, (∃ config : MarkConfiguration, validConfiguration config n) ↔ (n = 3 ∨ n = 6) := by
  sorry

#check stick_marking_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_marking_theorem_l892_89259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l892_89217

-- Define a real-valued function g on some interval
variable (g : ℝ → ℝ)
-- Define the interval
variable (a b : ℝ)

-- Define the area between y = g(x) and the x-axis over the interval [a, b]
noncomputable def area_g : ℝ := ∫ x in a..b, |g x|

-- Define the area between y = 2g(x + 3) - 1 and the x-axis over the interval [a, b]
noncomputable def area_transformed : ℝ := ∫ x in a..b, |2 * g (x + 3) - 1|

-- Theorem statement
theorem area_transformation (h : area_g g a b = 15) : area_transformed g a b = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l892_89217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_solution_interval_l892_89241

open Set

-- Define the function f on the interval [0,3]
def f : Set ℝ → Set ℝ := sorry

-- Define the inverse function of f
def f_inv : Set ℝ → Set ℝ := sorry

-- State the properties of f_inv
axiom f_inv_prop1 : f_inv (Ioc 0 1) = Ico 0 2
axiom f_inv_prop2 : f_inv (Ioi 2) = Ioo 0 1

-- Define the theorem
theorem max_solution_interval (x₀ : ℝ) : 
  (∃ x ∈ Ioo 0 x₀, f {x} = {x}) ∧ 
  (∀ y > x₀, ¬∃ x ∈ Ioo 0 y, f {x} = {x}) →
  x₀ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_solution_interval_l892_89241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_passes_through_point_l892_89222

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 2

-- State the theorem
theorem exponential_function_passes_through_point
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the exponent
  simp [Real.rpow_sub_one]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_passes_through_point_l892_89222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l892_89273

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_problem :
  (∀ x ∈ Set.Icc (-π/12) (π/3),
    Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ Real.sqrt 38 / 2 ∧
    Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≤ Real.sqrt 11) ∧
  (∀ α ∈ Set.Ioo (π/4) (π/2),
    f α = 12/5 → Real.tan (2*α + 3*π/4) = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l892_89273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_l892_89214

def A : Set (Fin 10 × Fin 10) := Set.univ

def is_valid_subset (X : Set (Fin 10 × Fin 10)) : Prop :=
  ∀ (s : ℕ → Fin 10), ∃ (n : ℕ), (s n, s (n + 1)) ∈ X

theorem min_subset_size :
  ∃ (X : Set (Fin 10 × Fin 10)), X ⊆ A ∧ is_valid_subset X ∧ Finset.card (Set.toFinite X).toFinset = 55 ∧
    ∀ (Y : Set (Fin 10 × Fin 10)), Y ⊆ A → is_valid_subset Y → Finset.card (Set.toFinite Y).toFinset ≥ 55 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_l892_89214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l892_89220

theorem problem_solution (x a b c : ℝ) : 
  x ≤ 0 →
  a - x = 2019 →
  b + |x| = 2020 →
  c + Real.sqrt (x^2) = 2021 →
  a * b * c = 24 →
  (((a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) ^ (1/3)) : ℝ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l892_89220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_eq_192_l892_89240

/-- The number of different arrangements for 7 students in a row,
    where one student must stand in the middle and two students must stand together. -/
def photo_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  2 * (remaining_students + 1) * (remaining_students.factorial)

theorem photo_arrangements_eq_192 : photo_arrangements = 192 := by
  sorry

#eval photo_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_eq_192_l892_89240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_difference_l892_89242

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

theorem geometric_sequence_difference (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, a n > 0) →
  is_geometric_sequence a →
  (∀ n : ℕ+, a n * a (n + 1) = (2 : ℝ) ^ (2 * n.val)) →
  a 6 - a 5 = 16 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_difference_l892_89242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratio_correct_l892_89251

/-- A triangular pyramid with lateral faces of equal areas -/
structure TriangularPyramid where
  α : ℝ
  β : ℝ
  γ : ℝ
  equal_lateral_areas : True  -- This represents the condition of equal lateral areas

/-- The ratio of radii for the inscribed and touching spheres -/
noncomputable def sphere_radii_ratio (p : TriangularPyramid) : ℝ :=
  (3 - Real.cos p.α - Real.cos p.β - Real.cos p.γ) /
  (3 + Real.cos p.α + Real.cos p.β + Real.cos p.γ)

theorem sphere_radii_ratio_correct (p : TriangularPyramid) :
  sphere_radii_ratio p =
  (3 - Real.cos p.α - Real.cos p.β - Real.cos p.γ) /
  (3 + Real.cos p.α + Real.cos p.β + Real.cos p.γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratio_correct_l892_89251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_limit_exceeders_l892_89215

theorem speed_limit_exceeders (total_motorists : ℝ) 
  (h1 : total_motorists > 0) 
  (ticket_receivers : ℝ) 
  (h2 : ticket_receivers = 0.1 * total_motorists) 
  (h3 : ticket_receivers > 0) :
  let non_ticket_ratio : ℝ := 0.4
  let exceeders : ℝ := ticket_receivers / (1 - non_ticket_ratio)
  ∃ ε > 0, |exceeders / total_motorists - 0.1667| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_limit_exceeders_l892_89215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_eating_problem_l892_89218

/-- Represents the daily consumption of grass per sheep -/
noncomputable def S : ℝ := sorry

/-- Represents the initial quantity of grass on one piece of land -/
noncomputable def C : ℝ := sorry

/-- Represents the rate at which grass grows daily -/
noncomputable def G : ℝ := sorry

theorem grass_eating_problem (h1 : 50 * S * 18 = 3 * C + 3 * G * 18)
                              (h2 : 40 * S * 12 = 2 * C + 2 * G * 12)
                              (h3 : G = 10 * S)
                              (h4 : C = 120 * S) :
  70 * S * 2 = C + G * 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_eating_problem_l892_89218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_of_9_eq_8_l892_89291

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def f (n : Nat) : Nat :=
  digit_sum (n^2 + 1)

def f_k : Nat → Nat → Nat
  | 0, n => n  -- Add base case for k = 0
  | 1, n => f n
  | k+1, n => f (f_k k n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_of_9_eq_8_l892_89291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_42_l892_89278

-- Define the circles and triangle
def small_circle_radius : ℝ := 3
def large_circle_radius : ℝ := 4
def center_distance : ℝ := 5

-- Define the properties of the triangle
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Sides of the triangle are tangent to the circles
  -- AB and BC are congruent
  -- Other properties of the triangle derived from the circles' configuration
  True -- Placeholder, replace with actual conditions when implementing

-- Define the area function for a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  -- Placeholder implementation
  0

-- Theorem statement
theorem triangle_area_is_42 (A B C : ℝ × ℝ) 
  (h : triangle_ABC A B C) : 
  area_triangle A B C = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_42_l892_89278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polo_shirt_cost_is_correct_l892_89294

/-- Calculates the total cost in USD for two polo shirts with given discounts, VAT, and exchange rate --/
def polo_shirt_cost (regular_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (vat_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_price1 := regular_price * (1 - discount1)
  let discounted_price2 := regular_price * (1 - discount2)
  let total_before_tax := discounted_price1 + discounted_price2
  let total_with_tax := total_before_tax * (1 + vat_rate)
  let total_usd := total_with_tax * exchange_rate
  total_usd

/-- The total cost of two polo shirts rounds to $82.84 USD --/
theorem polo_shirt_cost_is_correct : 
  ⌊polo_shirt_cost 50 0.4 0.3 0.08 1.18 * 100⌋ = 8284 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polo_shirt_cost_is_correct_l892_89294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_primes_is_one_tenth_l892_89274

/-- The set of integers from 1 to 30 inclusive -/
def S : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 30) (Finset.range 31)

/-- The set of prime numbers in S -/
def P : Finset ℕ := Finset.filter Nat.Prime S

/-- The number of elements in S -/
def S_count : ℕ := Finset.card S

/-- The number of elements in P -/
def P_count : ℕ := Finset.card P

/-- The probability of selecting two different prime numbers from S -/
noncomputable def prob_two_primes : ℚ := (P_count.choose 2 : ℚ) / (S_count.choose 2 : ℚ)

theorem prob_two_primes_is_one_tenth : prob_two_primes = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_primes_is_one_tenth_l892_89274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_vote_percentage_l892_89205

def election_votes : List Nat := [2500, 5000, 20000]

def total_votes : Nat := election_votes.sum

noncomputable def winning_votes : Nat := 
  match election_votes.maximum? with
  | some n => n
  | none => 0

noncomputable def winning_percentage : Real := 
  (winning_votes : Real) / (total_votes : Real) * 100

theorem winning_vote_percentage :
  ∀ ε > 0, |winning_percentage - 72.73| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_vote_percentage_l892_89205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_100_value_l892_89206

def c : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to make the function total
  | 1 => 2
  | 2 => 1
  | n + 3 => (2 - c (n + 2)) / (3 * c (n + 1))

theorem c_100_value : c 100 = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_100_value_l892_89206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_foci_distance_product_l892_89269

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a ≥ b)

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse a b) : ℝ × ℝ × ℝ × ℝ := by
  let c := Real.sqrt (a^2 - b^2)
  exact (-c, 0, c, 0)

/-- A tangent line to the ellipse at point (x, y) -/
structure TangentLine (e : Ellipse a b) (x y : ℝ) where
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (slope : ℝ)
  (intercept : ℝ)

/-- The distance from a point (x₀, y₀) to the line ax + by + c = 0 -/
noncomputable def distanceToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- The theorem to be proved -/
theorem tangent_line_foci_distance_product (a b : ℝ) (e : Ellipse a b) (x y : ℝ) (l : TangentLine e x y) :
  let (x₁, y₁, x₂, y₂) := foci e
  let d₁ := distanceToLine x₁ y₁ (-l.slope) 1 (-l.intercept)
  let d₂ := distanceToLine x₂ y₂ (-l.slope) 1 (-l.intercept)
  d₁ * d₂ = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_foci_distance_product_l892_89269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_from_B_and_C_l892_89282

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- The coordinates of point A -/
def A : ℝ × ℝ × ℝ := (0, -1, 0)

/-- The coordinates of point B -/
def B : ℝ × ℝ × ℝ := (2, 2, 4)

/-- The coordinates of point C -/
def C : ℝ × ℝ × ℝ := (0, 4, 2)

/-- Theorem: Point A is equidistant from points B and C -/
theorem equidistant_from_B_and_C : 
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = 
  distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 := by
  sorry

#eval A
#eval B
#eval C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_from_B_and_C_l892_89282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_2sqrt_l892_89267

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x

noncomputable def g (x : ℝ) : ℝ := (x / 4) ^ 2

theorem inverse_function_of_2sqrt (x : ℝ) (hx : x ≥ 0) :
  f (g x) = x ∧ g (f x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_2sqrt_l892_89267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sum_l892_89292

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The vertex of a quadratic function -/
noncomputable def vertex (q : QuadraticFunction) : ℝ × ℝ :=
  let x := -(q.b : ℝ) / (2 * (q.a : ℝ))
  (x, q.f x)

theorem quadratic_coefficient_sum (q : QuadraticFunction) 
  (h_vertex : vertex q = (2, -3))
  (h_point : q.f 0 = 1) :
  q.a - q.b + q.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_sum_l892_89292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l892_89246

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b * x + 9

noncomputable def f_prime (a b : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + b

theorem function_extrema (a b : ℝ) :
  (∀ x : ℝ, f_prime a b x = 0 ↔ x = 1 ∨ x = 3) →
  (a = 2 ∧ b = 3) ∧
  (f 2 3 1 = 31/3) ∧
  (f 2 3 3 = 9) ∧
  (∀ x : ℝ, x < 1 → f_prime 2 3 x > 0) ∧
  (∀ x : ℝ, 1 < x → x < 3 → f_prime 2 3 x < 0) ∧
  (∀ x : ℝ, x > 3 → f_prime 2 3 x > 0) :=
by sorry

#check function_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l892_89246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_vertices_l892_89265

noncomputable def D : ℝ × ℝ := (0, 0)
noncomputable def E : ℝ × ℝ := (10, 0)
noncomputable def F : ℝ × ℝ := (7, 5)
noncomputable def P : ℝ × ℝ := (4, 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem sum_of_distances_to_vertices :
  distance P D + distance P E + distance P F = 2 * Real.sqrt 5 + 2 * Real.sqrt 10 + 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_vertices_l892_89265
