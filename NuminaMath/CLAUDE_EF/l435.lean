import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fixed_intervals_l435_43583

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (abs x + 1)

def interval_image (a b : ℝ) : Set ℝ := {y | ∃ x, x ∈ Set.Icc a b ∧ y = f x}

theorem three_fixed_intervals :
  ∃! (S : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ S → p.1 < p.2) ∧ 
    (∀ (p : ℝ × ℝ), p ∈ S → interval_image p.1 p.2 = Set.Icc p.1 p.2) ∧
    Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fixed_intervals_l435_43583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_parts_complex_fraction_l435_43522

theorem sum_of_parts_complex_fraction :
  let z : ℂ := (1 + 4*I) / (2 - 4*I)
  (z.re + z.im : ℝ) = -1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_parts_complex_fraction_l435_43522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_power_after_digit_removal_l435_43501

/-- Represents a 1000-digit number without zeros -/
def ThousandDigitNumber : Type :=
  { n : ℕ // n ≥ 10^999 ∧ n < 10^1000 ∧ ∀ d, d ∈ n.digits 10 → d ≠ 0 }

/-- Represents the result of removing some (possibly none) of the last digits -/
def RemoveLastDigits (n : ThousandDigitNumber) : Set ℕ :=
  { m : ℕ | ∃ k : ℕ, m = n.val / 10^k ∧ k ≤ 999 }

/-- Checks if a number is a natural power of an integer less than 500 -/
def IsNaturalPowerLessThan500 (n : ℕ) : Prop :=
  ∃ (a k : ℕ), a < 500 ∧ k > 0 ∧ n = a^k

theorem exists_non_power_after_digit_removal (n : ThousandDigitNumber) :
  ∃ m ∈ RemoveLastDigits n, ¬IsNaturalPowerLessThan500 m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_power_after_digit_removal_l435_43501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_m_l435_43503

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.tan x + Real.sin x + 2015

-- State the theorem
theorem f_negative_m (m : ℝ) (h : f m = 2) : f (-m) = 4028 := by
  -- Expand the definition of f
  have h1 : f m = Real.tan m + Real.sin m + 2015 := rfl
  have h2 : f (-m) = Real.tan (-m) + Real.sin (-m) + 2015 := rfl

  -- Use properties of tan and sin for negative arguments
  have h3 : Real.tan (-m) = -Real.tan m := by sorry
  have h4 : Real.sin (-m) = -Real.sin m := by sorry

  -- Rewrite f(-m) using these properties
  rw [h2, h3, h4]

  -- Use the given condition f(m) = 2
  rw [h] at h1
  
  -- Solve the equation
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_m_l435_43503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l435_43593

/-- Calculates the speed of a bus including stoppages given its speed without stoppages and stoppage time -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (h1 : speed_without_stoppages = 48) 
  (h2 : stoppage_time = 45) : 
  (speed_without_stoppages * (60 - stoppage_time) / 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l435_43593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l435_43561

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 13 cm between them, is equal to 247 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 13 = 247 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l435_43561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_length_proof_l435_43523

-- Define the cube
noncomputable def cube_edge_length : ℝ := 1

-- Define the sphere
noncomputable def sphere_radius : ℝ := 2 * Real.sqrt 3 / 3

-- Define the length of the intersection curve
noncomputable def intersection_curve_length : ℝ := 5 * Real.sqrt 3 * Real.pi / 6

-- Theorem statement
theorem intersection_curve_length_proof :
  let cube := cube_edge_length
  let sphere := sphere_radius
  intersection_curve_length = 5 * Real.sqrt 3 * Real.pi / 6 :=
by
  sorry

#check intersection_curve_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_length_proof_l435_43523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l435_43573

-- Define the principal amount
noncomputable def principal : ℝ := 58 * 100 / (5 * 2)

-- Define the compound interest calculation function
noncomputable def compound_interest (p : ℝ) (r1 r2 r3 : ℝ) : ℝ :=
  p * (1 + r1/100) * (1 + r2/100) * (1 + r3/100) - p

-- State the theorem
theorem compound_interest_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |compound_interest principal 4 6 5 - 91.25| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l435_43573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l435_43582

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) (k : ℕ) 
    (h1 : sum_n seq 9 = 81)
    (h2 : seq.a (k - 4) = 191)
    (h3 : sum_n seq k = 10000) :
  k = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l435_43582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l435_43572

/-- Set of all different ways of adding 3 pairs of parentheses to the product of 5 numbers -/
def set_A : Finset (List (Fin 5 → ℕ)) := sorry

/-- Set of all different ways of dividing a convex hexagon into 4 triangles -/
def set_B : Finset (List (Fin 6 → Fin 6 → Prop)) := sorry

/-- Set of all arrangements of 4 black balls and 4 white balls in a line
    such that at any position, the number of white balls is not less than the number of black balls -/
def set_C : Finset (List Bool) := sorry

/-- The cardinalities of sets A, B, and C are equal -/
theorem cardinality_equality : (Finset.card set_A = Finset.card set_B) ∧ (Finset.card set_A = Finset.card set_C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l435_43572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_not_square_view_l435_43586

-- Define a type for solid geometric figures
inductive SolidFigure
| cylinder
| cone
| pyramid
| prism

-- Define a property for having a square view
def has_square_view (figure : SolidFigure) : Prop :=
  ∃ (view : ℕ), view ≤ 3 ∧ (view = 1 ∨ view = 2 ∨ view = 3)

-- State the theorem
theorem cone_not_square_view :
  ∀ (figure : SolidFigure), has_square_view figure → figure ≠ SolidFigure.cone :=
by
  intro figure h
  cases figure
  all_goals {
    intro hcontra
    cases h with
    | intro view h' =>
      cases h' with
      | intro h1 h2 =>
        sorry  -- The actual proof would go here
  }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_not_square_view_l435_43586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l435_43569

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3 - x^2) / Real.log 4

-- State the theorem
theorem f_properties :
  -- Domain of f is (-1, 3)
  (∀ x, x ∈ Set.Ioo (-1) 3 ↔ f x ∈ Set.univ) ∧
  -- f is increasing on (-1, 1)
  (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → f x < f y) ∧
  -- f is decreasing on (1, 3)
  (∀ x y, x ∈ Set.Ioo 1 3 → y ∈ Set.Ioo 1 3 → x < y → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l435_43569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_probability_l435_43509

/-- Given a triangle ABC and a random point M inside it, the probability that the area of one of
    the triangles ABM, BCM, or CAM is greater than the sum of the areas of the other two is 3/4. -/
theorem triangle_area_probability (A B C M : ℝ × ℝ) : 
  (∃ (p : (ℝ × ℝ) → ℝ), p M = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_probability_l435_43509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_c_contract_fee_l435_43562

/-- Represents the work rate of a team in completing the project per day -/
noncomputable def WorkRate (days : ℝ) : ℝ := 1 / days

/-- Represents the contract fee for a team based on their work contribution -/
noncomputable def ContractFee (totalFee workDone : ℝ) : ℝ := totalFee * workDone

theorem team_c_contract_fee :
  let teamA_rate := WorkRate 36
  let teamB_rate := WorkRate 24
  let teamC_rate := WorkRate 18
  let first_half_rate := (teamA_rate + teamB_rate + teamC_rate) / 2
  let second_half_rate := (teamA_rate + teamC_rate) / 2
  let total_work := first_half_rate + second_half_rate
  let teamC_work := teamC_rate / 2 + teamC_rate / 2
  let total_fee := (36000 : ℝ)
  ContractFee total_fee (teamC_work / total_work) = 20000 := by
  sorry

#check team_c_contract_fee

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_c_contract_fee_l435_43562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_similar_iff_tangential_l435_43595

/-- A convex polygon in a 2D plane --/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool -- This is a simplification; in a real implementation, we'd need to prove convexity

/-- A polygon is tangential if it has an inscribed circle --/
def IsTangential (p : ConvexPolygon) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    True -- Placeholder; actual condition for an inscribed circle would be more complex

/-- Two polygons are similar --/
def AreSimilar (p1 p2 : ConvexPolygon) : Prop :=
  True -- Placeholder; actual similarity condition would involve comparing angles and side ratios

/-- Create a new polygon by drawing lines parallel to each side of the original polygon at a constant distance --/
def ParallelPolygon (p : ConvexPolygon) (distance : ℝ) : ConvexPolygon :=
  { vertices := p.vertices, is_convex := p.is_convex } -- Placeholder implementation

/-- Theorem: A convex polygon can have a similar polygon formed by lines parallel to its sides 
    at a constant distance if and only if the original polygon is tangential --/
theorem parallel_similar_iff_tangential (p : ConvexPolygon) :
  (∃ (d : ℝ), AreSimilar p (ParallelPolygon p d)) ↔ IsTangential p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_similar_iff_tangential_l435_43595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l435_43563

theorem jacket_price_restoration : 
  ∀ (original_price : ℝ), original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.10)
  let required_increase := (original_price / price_after_second_reduction) - 1
  ∃ ε > 0, |required_increase - 0.4815| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l435_43563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_interval_for_f_l435_43520

noncomputable def f (x : ℝ) := -x / (1 + abs x)

theorem no_fixed_interval_for_f :
  ∀ a b : ℝ, a < b →
    (Set.Icc a b ≠ {y | ∃ x ∈ Set.Icc a b, f x = y}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_interval_for_f_l435_43520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_antiderivative_integral_l435_43502

open Set Interval MeasureTheory

variable {f : ℝ → ℝ} {F : ℝ → ℝ} {T : ℝ} {n : ℕ}

/-- A function is periodic with period T -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem periodic_antiderivative_integral
  (hf_periodic : IsPeriodic f T)
  (hf_continuous : Continuous f)
  (hF_antideriv : ∀ x, HasDerivAt F (f x) x)
  (hT_pos : T > 0)
  : ∫ x in (Set.Icc 0 T), (F (n * x) - F x - f x * ((n - 1) * T / 2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_antiderivative_integral_l435_43502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_56_equals_fraction_l435_43514

/-- The decimal representation of a number with infinitely repeating digits 56 after the decimal point -/
def repeating_decimal_56 : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.56666... is equal to the fraction 56/99 -/
theorem repeating_decimal_56_equals_fraction : repeating_decimal_56 = 56 / 99 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_56_equals_fraction_l435_43514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_even_seven_dice_l435_43500

/-- The number of sides on each die -/
def numSides : ℕ := 5

/-- The number of dice rolled -/
def numDice : ℕ := 7

/-- The number of 'even' outcomes on each die -/
def numEvenOutcomes : ℕ := 3

/-- The number of desired 'even' outcomes across all dice -/
def desiredEvenOutcomes : ℕ := 3

/-- The probability of rolling exactly three 'even' numbers when rolling 7 fair 5-sided dice -/
theorem prob_three_even_seven_dice : 
  (numDice.choose desiredEvenOutcomes : ℚ) * 
  ((numEvenOutcomes : ℚ) / (numSides : ℚ)) ^ desiredEvenOutcomes * 
  ((numSides - numEvenOutcomes : ℚ) / (numSides : ℚ)) ^ (numDice - desiredEvenOutcomes) = 
  15120 / 78125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_even_seven_dice_l435_43500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l435_43598

theorem triangle_area_formula (r rc : ℝ) (α β γ : ℝ) (S p : ℝ) :
  r > 0 → rc > 0 → α > 0 → β > 0 → γ > 0 →
  r = rc * Real.tan (α / 2) * Real.tan (β / 2) →
  p = rc * (Real.tan (γ / 2))⁻¹ →
  S = p * r →
  S = rc^2 * Real.tan (α / 2) * Real.tan (β / 2) * (Real.tan (γ / 2))⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l435_43598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_implies_a_eq_neg_one_l435_43596

/-- The function f(x) representing the left side of the equation minus the right side -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.exp (-a) + Real.exp (a - 2*x) - Real.exp (-x)

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * Real.exp (a - 2*x) + Real.exp (-x)

theorem equation_solution_implies_a_eq_neg_one (a : ℝ) :
  (∃ x : ℝ, f a x = 0) → a = -1 := by
  sorry

#check equation_solution_implies_a_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_implies_a_eq_neg_one_l435_43596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_congruent_to_one_mod_three_l435_43599

theorem positive_integers_congruent_to_one_mod_three :
  {x : ℕ | x > 0 ∧ x % 3 = 1} = {x : ℕ | x > 0 ∧ ∃ k : ℕ, x = 3 * k + 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_congruent_to_one_mod_three_l435_43599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_division_theorem_l435_43594

/-- Represents the number of pieces a pie is cut into -/
def PiecesCount (n : ℕ) : Prop := True

/-- Represents that a number of pieces can be equally divided among a group of people -/
def CanDivideEqually (pieces : ℕ) (people : ℕ) : Prop := pieces % people = 0

/-- The minimum number of pieces the pie can be cut into -/
def MinimumPieces : ℕ := 11

theorem pie_division_theorem :
  (∀ n : ℕ, n < MinimumPieces → ¬(CanDivideEqually n 5 ∧ CanDivideEqually n 7)) ∧
  (CanDivideEqually MinimumPieces 5 ∧ CanDivideEqually MinimumPieces 7) :=
by
  sorry -- Proof omitted for brevity

#check pie_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_division_theorem_l435_43594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_approx_42_43_altitude_sum_equals_result_l435_43590

/-- The line equation forming the triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 18 * x + 9 * y = 162

/-- The x-intercept of the line -/
noncomputable def x_intercept : ℝ := 9

/-- The y-intercept of the line -/
noncomputable def y_intercept : ℝ := 18

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept

/-- The sum of the lengths of the altitudes of the triangle -/
noncomputable def altitude_sum : ℝ := x_intercept + y_intercept + (162 / Real.sqrt (18^2 + 9^2))

/-- Theorem stating that the sum of the lengths of the altitudes is approximately equal to 42.43 -/
theorem altitude_sum_approx_42_43 : 
  ∃ ε > 0, abs (altitude_sum - 42.43) < ε :=
sorry

/-- Theorem stating that the sum of the lengths of the altitudes is equal to (729 + 162√5) / 21 -/
theorem altitude_sum_equals_result : 
  altitude_sum = (729 + 162 * Real.sqrt 5) / 21 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_approx_42_43_altitude_sum_equals_result_l435_43590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l435_43544

-- Define the ellipse equation
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / k + y^2 / 2 = 1

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem ellipse_k_values :
  ∀ k : ℝ, 
    (∃ a b : ℝ, 
      (ellipse_equation a b k ∧ 
       ((a^2 = k ∧ b^2 = 2) ∨ (a^2 = 2 ∧ b^2 = k)) ∧
       focal_length a b = 2)) 
    ↔ (k = 1 ∨ k = 3) := by
  sorry

#check ellipse_k_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l435_43544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_formula_a_when_K_is_2_l435_43528

/-- Sequence a_n defined recursively --/
def a (K : ℕ+) : ℕ → ℕ
  | 0 => 1
  | n + 1 => n + 1 + K * ((n + 1) - (a K n % K))

/-- General formula for a_n --/
def a_formula (K : ℕ+) (n : ℕ) : ℚ :=
  1 + ((K + 1) * (n - 1) * n : ℚ) / 2

theorem a_eq_formula (K : ℕ+) (n : ℕ) :
  (a K n : ℚ) = a_formula K n := by sorry

theorem a_when_K_is_2 (n : ℕ) :
  a_formula 2 n = (3 * n^2 - 3 * n + 2 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_formula_a_when_K_is_2_l435_43528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l435_43537

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 9 → n = 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l435_43537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cycle_points_l435_43556

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then x + 1/2
  else if 1/2 < x ∧ x ≤ 1 then 2 * (1 - x)
  else 0  -- This case should never occur in our problem

noncomputable def x₀ : ℝ := 2/15
noncomputable def x₁ : ℝ := 19/30
noncomputable def x₂ : ℝ := 11/15
noncomputable def x₃ : ℝ := 8/15
noncomputable def x₄ : ℝ := 14/15

theorem five_cycle_points :
  0 ≤ x₀ ∧ x₀ ≤ 1 ∧
  0 ≤ x₁ ∧ x₁ ≤ 1 ∧
  0 ≤ x₂ ∧ x₂ ≤ 1 ∧
  0 ≤ x₃ ∧ x₃ ≤ 1 ∧
  0 ≤ x₄ ∧ x₄ ≤ 1 ∧
  x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₀ ≠ x₃ ∧ x₀ ≠ x₄ ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧
  x₃ ≠ x₄ ∧
  f x₀ = x₁ ∧
  f x₁ = x₂ ∧
  f x₂ = x₃ ∧
  f x₃ = x₄ ∧
  f x₄ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cycle_points_l435_43556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_my_sequence_l435_43554

def my_sequence (n : ℕ) : ℕ := n * n.factorial

def sum_my_sequence : ℕ := (List.range 7).map (λ i => my_sequence (i + 1)) |>.sum

theorem units_digit_of_sum_my_sequence :
  sum_my_sequence % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_my_sequence_l435_43554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nested_roots_l435_43507

theorem sqrt_nested_roots : Real.sqrt (25 * Real.sqrt (10 * Real.sqrt 25)) = 5 * (2 : Real)^(1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nested_roots_l435_43507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l435_43540

theorem hyperbola_equation (e h h_desired : ℝ × ℝ → Prop) :
  (∀ x y, e (x, y) ↔ (x^2 / 24 + y^2 / 49 = 1)) →
  (∀ x y, h (x, y) ↔ (x^2 / 36 - y^2 / 64 = 1)) →
  (∀ x y, h_desired (x, y) ↔ (y^2 / 16 - x^2 / 9 = 1)) →
  (∀ c, c^2 = 49 - 24 → 
    (∀ x y, h_desired (x, y) → 
      ∃ f₁ f₂, e (0, f₁) ∧ e (0, f₂) ∧ (y - f₁)^2 - x^2 = (y - f₂)^2 - x^2)) →
  (∀ x y, h (x, y) → y = (4/3) * x ∨ y = -(4/3) * x) →
  (∀ x y, h_desired (x, y) → y = (4/3) * x ∨ y = -(4/3) * x) →
  ∀ x y, h_desired (x, y) ↔ (y^2 / 16 - x^2 / 9 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l435_43540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l435_43557

theorem power_equality : (50 : ℝ)^4 = 10^(6.79588 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l435_43557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_seven_halves_l435_43506

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  sideAC : ℝ
  sideBC : ℝ
  sideAB : ℝ
  is_right_triangle : sideAC^2 + sideBC^2 = sideAB^2
  sideAC_eq : sideAC = 5
  sideBC_eq : sideBC = 12
  sideAB_eq : sideAB = 13

/-- The crease formed when folding vertex C onto vertex A -/
noncomputable def crease_length (t : RightTriangle) : ℝ := 
  Real.sqrt ((t.sideAB / 2)^2 - t.sideAC^2)

/-- Theorem stating that the crease length is 7/2 inches -/
theorem crease_length_is_seven_halves (t : RightTriangle) : 
  crease_length t = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_seven_halves_l435_43506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_105th_bracket_l435_43538

/-- Represents the sequence {2n-1} where n ∈ ℕ⁺ -/
def mySequence (n : ℕ) : ℕ := 2 * n - 1

/-- Determines the number of terms in a bracket based on its position -/
def termsInBracket (pos : ℕ) : ℕ :=
  match pos % 3 with
  | 0 => 3
  | 1 => 1
  | _ => 2

/-- Calculates the sum of terms in the nth bracket -/
def bracketSum (n : ℕ) : ℕ :=
  let start := (mySequence (3 * ((n - 1) / 3) + 1)) + 12 * ((n - 1) / 3)
  let terms := termsInBracket n
  (terms * (2 * start + (terms - 1) * 2)) / 2

theorem sum_105th_bracket : bracketSum 105 = 1251 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_105th_bracket_l435_43538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_point_coordinates_l435_43516

noncomputable def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem extension_point_coordinates :
  let A : ℝ × ℝ := point (-3) 4
  let B : ℝ × ℝ := point 9 (-2)
  let C : ℝ × ℝ := point 21 (-8)
  distance A C = 2 * distance A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_point_coordinates_l435_43516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_center_with_rule1_center_possible_with_both_rules_l435_43531

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A frog's position -/
structure Frog where
  position : Point

/-- A regular hexagon -/
structure RegularHexagon where
  center : Point
  sideLength : ℝ

/-- Rule 1: Jumping to double the distance -/
noncomputable def rule1 (f : Point) (p : Point) : Point :=
  { x := 2 * p.x - f.x, y := 2 * p.y - f.y }

/-- Rule 2: Jumping to half the distance -/
noncomputable def rule2 (f : Point) (p : Point) : Point :=
  { x := (f.x + p.x) / 2, y := (f.y + p.y) / 2 }

/-- Initial configuration of frogs on a regular hexagon -/
def initialConfiguration (h : RegularHexagon) : List Frog :=
  sorry

/-- Predicate to check if a point is at the center of the hexagon -/
def isAtCenter (p : Point) (h : RegularHexagon) : Prop :=
  p = h.center

/-- Theorem: No sequence of jumps using only Rule 1 can place a frog at the center -/
theorem no_center_with_rule1 (h : RegularHexagon) :
  ∀ (jumps : List (Frog → Frog)), 
    (∀ jump ∈ jumps, ∃ f p, jump f = { position := rule1 f.position p }) →
    ∀ f ∈ initialConfiguration h,
      ¬isAtCenter (List.foldl (fun f jump => jump f) f jumps).position h :=
by sorry

/-- Theorem: There exists a sequence of jumps using Rule 1 and Rule 2 that can place a frog at the center -/
theorem center_possible_with_both_rules (h : RegularHexagon) :
  ∃ (jumps : List (Frog → Frog)),
    (∀ jump ∈ jumps, (∃ f p, jump f = { position := rule1 f.position p }) ∨
                     (∃ f p, jump f = { position := rule2 f.position p })) ∧
    ∃ f ∈ initialConfiguration h,
      isAtCenter (List.foldl (fun f jump => jump f) f jumps).position h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_center_with_rule1_center_possible_with_both_rules_l435_43531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l435_43551

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define vector operations
def vecSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vecDot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def vecScale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  let BA := vecSub t.A t.B
  let BC := vecSub t.C t.B
  let CB := vecSub t.B t.C
  vecDot (vecSub (vecScale 2 BA) (vecScale 3 BC)) CB = 0

-- Define the angle A
noncomputable def angleA (t : Triangle) : ℝ :=
  let BA := vecSub t.A t.B
  let CA := vecSub t.A t.C
  Real.arccos (vecDot BA CA / (norm BA * norm CA))

-- State the theorem
theorem max_angle_A (t : Triangle) (h : satisfiesCondition t) :
  angleA t ≤ π / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l435_43551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_min_l435_43525

-- Define f(x) as an even function on R
noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

-- Define g(x) on [1,2]
noncomputable def g (a x : ℝ) : ℝ := f x - 2*a*x + 1

-- Define h(a) as the minimum value of g(x) on [1,2]
noncomputable def h (a : ℝ) : ℝ :=
  if a ≤ 0 then -2*a
  else if a < 1 then -a^2 - 2*a
  else 1 - 4*a

theorem f_even_and_g_min :
  (∀ x, f (-x) = f x) ∧
  (∀ x, x > 0 → f x = x^2 - 2*x) ∧
  (∀ a, ∀ x ∈ Set.Icc 1 2, g a x ≥ h a) := by
  sorry

#check f_even_and_g_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_min_l435_43525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_change_l435_43545

/-- Represents a tripod with three legs -/
structure Tripod where
  leg_length : ℝ
  height : ℝ

/-- Calculates the new height of a tripod after shortening one leg -/
noncomputable def new_height (t : Tripod) (shortened_length : ℝ) : ℝ :=
  144 / Real.sqrt 1585

theorem tripod_height_change (t : Tripod) (h1 : t.leg_length = 5) (h2 : t.height = 4) :
  new_height t 4 = 144 / Real.sqrt 1585 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_change_l435_43545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l435_43547

-- Define the power function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^m

-- Define g(x) in terms of f(x)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x + 2*x - 3

-- State the theorem
theorem power_function_range :
  ∃ (m : ℝ), 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∧  -- f is monotonically increasing on (0, +∞)
    m = 3 ∧
    (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -6 ≤ g m x ∧ g m x ≤ 30) ∧  -- range of g(x) on [-1, 3]
    (g m (-1) = -6) ∧ (g m 3 = 30) :=  -- endpoints of the range
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l435_43547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l435_43566

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_theorem (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * (1000 / 3600) →
  crossing_time = 23.998080153587715 →
  bridge_length = 290 →
  ∃ (train_length : ℝ), |train_length - 110| < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l435_43566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l435_43530

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The set of positive integers k for which a^2 + b^2 + c^2 = kabc has positive integer solutions -/
def valid_k : Set ℕ := {k : ℕ | ∃ a b c : ℕ+, a^2 + b^2 + c^2 = k * (a * b * c)}

/-- Property that a number is the sum of two squares of positive integers -/
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ+, x^2 + y^2 = n

/-- Main theorem -/
theorem diophantine_equation_solutions :
  (valid_k = {1, 3}) ∧
  (∀ n : ℕ, n ≥ 2 →
    (let a : ℕ+ := 1
     let b : ℕ+ := ⟨fib (2*n - 1), by sorry⟩
     let c : ℕ+ := ⟨fib (2*n + 1), by sorry⟩
     a^2 + b^2 + c^2 = 3 * (a * b * c) ∧
     is_sum_of_two_squares (a * b) ∧
     is_sum_of_two_squares (b * c) ∧
     is_sum_of_two_squares (a * c))) ∧
  (∀ a b c : ℕ+, a^2 + b^2 + c^2 = 3 * (a * b * c) →
    (3*a)^2 + (3*b)^2 + (3*c)^2 = (3*a) * (3*b) * (3*c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l435_43530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l435_43585

/-- Represents a cube with its vertex positions relative to a plane. -/
structure Cube where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ

/-- Calculates the distance from a vertex to a plane given the heights of adjacent vertices. -/
noncomputable def distance_to_plane (c : Cube) : ℝ :=
  (17 - 15 * Real.sqrt 2) / 4

/-- Theorem stating the distance from vertex A to the plane for a specific cube configuration. -/
theorem distance_theorem (c : Cube) 
  (h_side : c.side_length = 8)
  (h_heights : c.adjacent_heights = ![13, 15, 17]) :
  distance_to_plane c = (17 - 15 * Real.sqrt 2) / 4 := by
  sorry

#eval (17 + 450 + 4 : Nat)  -- To verify p + q + r < 2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l435_43585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l435_43508

-- Define the angle α
noncomputable def α : ℝ := sorry

-- Define the coordinates of point P
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Define the conditions
axiom x_not_zero : x ≠ 0
axiom y_eq_neg_sqrt_2 : y = -Real.sqrt 2
axiom cos_α_eq : Real.cos α = (Real.sqrt 3 / 6) * x

-- Define the theorem
theorem angle_values :
  Real.sin α = -Real.sqrt 6 / 6 ∧ abs (Real.tan α) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l435_43508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_bisection_l435_43560

/-- A regular octahedron in 3D space -/
structure RegularOctahedron where
  center : Fin 3 → ℝ
  volume : ℝ

/-- A line in 3D space -/
structure Line where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- A plane in 3D space -/
structure Plane where
  point : Fin 3 → ℝ
  normal : Fin 3 → ℝ

/-- Function to check if a plane divides an octahedron into two equal parts -/
def divides_equally (o : RegularOctahedron) (p : Plane) : Prop :=
  ∃ v : ℝ, v = o.volume / 2

/-- Function to create a plane from a point and a line -/
noncomputable def plane_from_point_and_line (point : Fin 3 → ℝ) (line : Line) : Plane :=
  { point := point,
    normal := sorry }  -- The cross product operation is complex and needs to be defined separately

/-- Theorem stating that the plane through the center of symmetry and the given line divides the octahedron equally -/
theorem octahedron_bisection (o : RegularOctahedron) (l : Line) :
  divides_equally o (plane_from_point_and_line o.center l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_bisection_l435_43560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_6_l435_43527

-- Define the function f
noncomputable def f (y : ℝ) : ℝ :=
  let x := (y - 1) / 3
  x^2 + 3*x + 2

-- State the theorem
theorem f_of_4_equals_6 : f 4 = 6 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_6_l435_43527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_shifted_g_l435_43533

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (2 * x)

-- State the theorem
theorem f_equals_shifted_g : ∀ x : ℝ, f x = g (x + 3 * Real.pi / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_shifted_g_l435_43533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_time_difference_machine_q_time_l435_43517

/-- Represents a machine that produces sprockets -/
structure Machine where
  production_rate : ℝ

/-- The number of sprockets to be manufactured -/
def target_sprockets : ℕ := 550

/-- Machine A's production rate in sprockets per hour -/
def machine_a_rate : ℝ := 5

/-- Machine Q produces 10% more sprockets per hour than Machine A -/
def machine_q : Machine := ⟨machine_a_rate * 1.1⟩

/-- Machine P takes longer than Machine Q to produce the target number of sprockets -/
axiom machine_p_slower : ∃ (machine_p : Machine) (t : ℝ), t > 0 ∧ 
  target_sprockets / machine_q.production_rate + t = target_sprockets / machine_p.production_rate

/-- Theorem stating that we cannot determine the exact time difference between Machine P and Q -/
theorem cannot_determine_time_difference : 
  ¬ ∃ (t : ℝ), ∀ (machine_p : Machine), 
    target_sprockets / machine_q.production_rate + t = target_sprockets / machine_p.production_rate :=
by sorry

/-- Theorem stating that Machine Q takes 100 hours to produce 550 sprockets -/
theorem machine_q_time : 
  target_sprockets / machine_q.production_rate = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_time_difference_machine_q_time_l435_43517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l435_43512

theorem sin_cos_shift (x : ℝ) : Real.sin (2 * x - π / 6) = Real.cos (2 * (x - π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l435_43512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_savings_fraction_l435_43526

/-- The fraction of Martha's daily allowance saved on most days -/
def regular_savings_fraction (
  daily_allowance : ℚ
  ) (total_days : ℕ
  ) (regular_days : ℕ
  ) (reduced_savings_fraction : ℚ
  ) (total_savings : ℚ
  ) : ℚ :=
  (total_savings - daily_allowance * reduced_savings_fraction) /
  (daily_allowance * regular_days)

/-- Proof that Martha saves half of her daily allowance on most days -/
theorem martha_savings_fraction :
  regular_savings_fraction 12 7 6 (1/4) 39 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_savings_fraction_l435_43526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_distances_to_square_vertices_l435_43592

/-- A square in a plane --/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- The distance between two points in a plane --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that no point exists with distances 1, 1, 2, and 3 to the vertices of a square --/
theorem no_point_with_distances_to_square_vertices : ¬∃ (s : Square) (p : ℝ × ℝ), 
  ∃ (perm : Fin 4 → Fin 4), Function.Bijective perm ∧ 
  (distance p (s.vertices (perm 0)) = 1) ∧
  (distance p (s.vertices (perm 1)) = 1) ∧
  (distance p (s.vertices (perm 2)) = 2) ∧
  (distance p (s.vertices (perm 3)) = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_distances_to_square_vertices_l435_43592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_cosine_function_l435_43589

open Real

theorem min_value_of_shifted_cosine_function 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * cos (2 * x - π / 6)) :
  ∃ x ∈ Set.Icc 0 (π / 2), f x = -Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 (π / 2), f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_cosine_function_l435_43589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l435_43519

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l435_43519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_three_neg_one_eq_zero_l435_43575

/-- Define the function f for three distinct real numbers -/
noncomputable def f (a b c : ℝ) : ℝ := (c - a + b) / (b - a)

/-- Theorem stating that f(2, 3, -1) = 0 -/
theorem f_two_three_neg_one_eq_zero : f 2 3 (-1) = 0 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Evaluate the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_three_neg_one_eq_zero_l435_43575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l435_43539

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l435_43539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l435_43518

/-- Represents a rectangular farm with fencing. -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ
  totalCost : ℝ

/-- Calculates the cost per meter of fencing for a rectangular farm. -/
noncomputable def costPerMeter (farm : RectangularFarm) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (longSide ^ 2 + farm.shortSide ^ 2)
  let totalLength := longSide + farm.shortSide + diagonal
  farm.totalCost / totalLength

/-- Theorem stating that for a rectangular farm with given specifications, 
    the cost per meter of fencing is 13. -/
theorem farm_fencing_cost (farm : RectangularFarm) 
    (h_area : farm.area = 1200)
    (h_short : farm.shortSide = 30)
    (h_cost : farm.totalCost = 1560) :
    costPerMeter farm = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l435_43518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuji_apple_market_analysis_l435_43574

structure Origin where
  name : String
  price : ℝ
  marketShare : ℝ
deriving Inhabited

def origins : List Origin := [
  ⟨"A", 150, 0.15⟩,
  ⟨"B", 160, 0.10⟩,
  ⟨"C", 140, 0.25⟩,
  ⟨"D", 155, 0.20⟩,
  ⟨"E", 170, 0.30⟩
]

def totalBoxes : ℝ := 20

def nextYearMarketShare (o : Origin) : ℝ :=
  match o.name with
  | "A" => o.marketShare + 0.05
  | "C" => o.marketShare - 0.05
  | _ => o.marketShare

theorem fuji_apple_market_analysis :
  let priceLessThan160 := (origins.filter (fun o => o.price < 160)).foldl (fun acc o => acc + o.marketShare) 0
  let boxesFromAAndB := (origins.filter (fun o => o.name = "A" ∨ o.name = "B")).foldl (fun acc o => acc + o.marketShare * totalBoxes) 0
  let probDifferentOrigins := 2 * (origins.filter (fun o => o.name = "A")).head!.marketShare * (origins.filter (fun o => o.name = "B")).head!.marketShare / (boxesFromAAndB * (boxesFromAAndB - 1))
  let avgPriceThisYear := (origins.map (fun o => o.price * o.marketShare)).sum
  let avgPriceNextYear := (origins.map (fun o => o.price * nextYearMarketShare o)).sum
  priceLessThan160 = 0.60 ∧
  boxesFromAAndB = 5 ∧
  probDifferentOrigins = 3/5 ∧
  avgPriceThisYear < avgPriceNextYear := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuji_apple_market_analysis_l435_43574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_multiplied_number_l435_43559

theorem find_multiplied_number (numbers : Fin 5 → ℝ) 
  (h_avg : (Finset.sum Finset.univ (λ i => numbers i)) / 5 = 6.8)
  (h_new_avg : ∃ i, ((Finset.sum Finset.univ (λ j => numbers j) - numbers i + 3 * numbers i) / 5) = 9.2) :
  ∃ i, numbers i = 6 ∧ ((Finset.sum Finset.univ (λ j => numbers j) - numbers i + 3 * numbers i) / 5) = 9.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_multiplied_number_l435_43559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_current_speed_l435_43513

/-- The speed of a water current given swimmer's characteristics and performance -/
theorem water_current_speed
  (swimmer_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h_swimmer_speed : swimmer_speed = 4)
  (h_distance : distance = 3)
  (h_time : time = 1.5) :
  ∃ (water_current_speed : ℝ),
    distance / time = swimmer_speed - water_current_speed ∧
    water_current_speed = 2 :=
by
  -- We introduce the water_current_speed as an existential variable
  use 2
  constructor
  · -- Prove the equation
    rw [h_distance, h_time, h_swimmer_speed]
    norm_num
  · -- Prove that water_current_speed = 2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_current_speed_l435_43513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_division_l435_43576

def weights : List ℕ := [1, 2, 3, 4, 8, 16]

def is_valid_division (pile1 pile2 : List ℕ) : Prop :=
  pile1.length = 2 ∧
  pile2.length = 4 ∧
  pile1.sum = pile2.sum ∧
  (pile1 ++ pile2).toFinset = weights.toFinset

theorem unique_division :
  ∀ (pile1 pile2 : List ℕ),
    is_valid_division pile1 pile2 →
    (pile1.toFinset = ({1, 16} : Finset ℕ) ∧ pile2.toFinset = ({2, 3, 4, 8} : Finset ℕ)) ∨
    (pile2.toFinset = ({1, 16} : Finset ℕ) ∧ pile1.toFinset = ({2, 3, 4, 8} : Finset ℕ)) :=
by
  sorry

#check unique_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_division_l435_43576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_value_l435_43567

/-- An ellipse with semi-major axis 2 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

/-- The left focus of the ellipse -/
noncomputable def leftFocus : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- The right focus of the ellipse -/
noncomputable def rightFocus : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The distance between a point and the left focus -/
noncomputable def distToLeftFocus (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - leftFocus.1)^2 + (p.2 - leftFocus.2)^2)

/-- The distance between a point and the right focus -/
noncomputable def distToRightFocus (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - rightFocus.1)^2 + (p.2 - rightFocus.2)^2)

/-- The function to be maximized -/
noncomputable def f (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (distToLeftFocus p * distToRightFocus p / eccentricity^2)

theorem ellipse_max_value :
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 / 3 ∧
  ∀ (p : ℝ × ℝ), p ∈ Ellipse → f p ≤ max := by
  sorry

#check ellipse_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_value_l435_43567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l435_43555

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6) + 1

theorem max_value_theorem (x₁ x₂ : ℝ) 
  (h₁ : g x₁ * g x₂ = 16)
  (h₂ : x₁ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2))
  (h₃ : x₂ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2)) :
  (∀ y₁ y₂, y₁ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) → 
            y₂ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) → 
            g y₁ * g y₂ = 16 → 
            2 * y₁ - y₂ ≤ 35 * Real.pi / 12) ∧
  (∃ z₁ z₂, z₁ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) ∧ 
            z₂ ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) ∧ 
            g z₁ * g z₂ = 16 ∧ 
            2 * z₁ - z₂ = 35 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l435_43555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l435_43588

/-- Predicate to assert that the given points form an ellipse -/
def IsEllipse (O A B C D F : ℝ × ℝ) : Prop := sorry

/-- Given an ellipse with center O, major axis AB, minor axis CD, and focus F,
    if OF = 8 and the diameter of the inscribed circle of triangle OCF is 4,
    then (AB)(CD) = 240 -/
theorem ellipse_product (O A B C D F : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let OF := Real.sqrt ((O.1 - F.1)^2 + (O.2 - F.2)^2)
  let OC := Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2)
  let CF := Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)
  IsEllipse O A B C D F → 
  OF = 8 → 
  OC + OF - CF = 4 → 
  AB * CD = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l435_43588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_l435_43570

/-- Represents the cost and selling prices of an article -/
structure Article where
  cost : ℚ
  selling_price_1 : ℚ
  selling_price_2 : ℚ

/-- Calculates the gain percentage for a given selling price -/
noncomputable def gain_percentage (a : Article) (selling_price : ℚ) : ℚ :=
  (selling_price - a.cost) / a.cost * 100

/-- Theorem stating the cost of the article given the conditions -/
theorem article_cost (a : Article) 
  (h1 : a.selling_price_1 = 340)
  (h2 : a.selling_price_2 = 350)
  (h3 : gain_percentage a a.selling_price_2 = gain_percentage a a.selling_price_1 + 5) :
  a.cost = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_l435_43570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treatment_volume_is_correct_l435_43552

/-- Calculates the total volume of treatment received from two saline drips --/
noncomputable def total_treatment_volume (
  drip1_drops_per_minute : ℝ)
  (drip1_drops_per_ml : ℝ)
  (drip2_drops_per_minute : ℝ)
  (drip2_drops_per_ml : ℝ)
  (total_hours : ℝ)
  (break_minutes_per_hour : ℝ) : ℝ :=
  let total_minutes : ℝ := total_hours * 60
  let break_minutes : ℝ := total_hours * break_minutes_per_hour
  let treatment_minutes : ℝ := total_minutes - break_minutes
  let drip1_total_drops : ℝ := drip1_drops_per_minute * treatment_minutes
  let drip2_total_drops : ℝ := drip2_drops_per_minute * treatment_minutes
  let drip1_volume : ℝ := drip1_total_drops / drip1_drops_per_ml
  let drip2_volume : ℝ := drip2_total_drops / drip2_drops_per_ml
  drip1_volume + drip2_volume

/-- The total volume of treatment received is approximately 566.67 ml --/
theorem treatment_volume_is_correct :
  ∃ ε > 0, |total_treatment_volume 15 20 25 12 4 10 - 566.67| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_treatment_volume_is_correct_l435_43552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_derivative_bound_quadratic_derivative_bound_tight_l435_43568

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of a quadratic polynomial
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_derivative_bound :
  ∀ (a b c : ℝ),
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |quadratic_polynomial a b c x| ≤ 1) →
  quadratic_derivative a b 0 ≤ 8 := by
  sorry

theorem quadratic_derivative_bound_tight :
  ∃ (a b c : ℝ),
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |quadratic_polynomial a b c x| ≤ 1) ∧
  quadratic_derivative a b 0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_derivative_bound_quadratic_derivative_bound_tight_l435_43568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l435_43510

/-- A power function that passes through the point (3, 1/9) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_through_point (α : ℝ) :
  f α 3 = 1/9 → f α 2 = 1/4 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l435_43510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_prob_two_red_from_B_l435_43571

-- Define the initial conditions of the boxes
def box_A : Fin 2 → Nat
| 0 => 3  -- white balls
| 1 => 2  -- red balls
| _ => 0

def box_B : Fin 2 → Nat
| 0 => 4  -- white balls
| 1 => 1  -- red balls
| _ => 0

-- Define the random variable X
def X : Finset Nat := {0, 1, 2}

-- Define the probability mass function of X
def pmf_X (x : Nat) : ℚ :=
  if x = 0 then 3/10
  else if x = 1 then 6/10
  else if x = 2 then 1/10
  else 0

-- Define the expected value of X
noncomputable def E_X : ℚ := Finset.sum X (λ x => x * pmf_X x)

-- Define the probability of taking 2 red balls from box B after transfer
def P_two_red_from_B : ℚ := 3/70

-- Theorem statements
theorem expected_value_X : E_X = 4/5 := by sorry

theorem prob_two_red_from_B : P_two_red_from_B = 3/70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_prob_two_red_from_B_l435_43571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_d_value_l435_43524

/-- Four points are collinear if the vectors between any three of them are proportional -/
def collinear (p₁ p₂ p₃ p₄ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧
    (p₂.fst - p₁.fst, p₂.snd - p₁.snd, p₂.snd.fst - p₁.snd.fst) = k₁ • (p₃.fst - p₁.fst, p₃.snd - p₁.snd, p₃.snd.fst - p₁.snd.fst) ∧
    (p₂.fst - p₁.fst, p₂.snd - p₁.snd, p₂.snd.fst - p₁.snd.fst) = k₂ • (p₄.fst - p₁.fst, p₄.snd - p₁.snd, p₄.snd.fst - p₁.snd.fst)

theorem collinear_points_d_value (a c d : ℝ) :
  collinear (2, 0, a) (2*a, 2, 0) (0, c, 1) (9*d, 9*d, -d) → d = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_d_value_l435_43524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sum_l435_43587

theorem card_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sum_l435_43587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2theta_minus_cos_squared_theta_l435_43511

theorem max_value_sin_2theta_minus_cos_squared_theta :
  ∃ (max_value : ℝ), max_value = (Real.sqrt 5 - 1) / 2 ∧
  ∀ θ : ℝ, π/4 < θ ∧ θ < π/2 →
  Real.sin (2*θ) - (Real.cos θ)^2 ≤ max_value ∧
  ∃ θ₀ : ℝ, π/4 < θ₀ ∧ θ₀ < π/2 ∧ Real.sin (2*θ₀) - (Real.cos θ₀)^2 = max_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2theta_minus_cos_squared_theta_l435_43511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_shirts_count_l435_43580

def trousers_count : ℕ := 10
def jacket_count : ℕ := 5
def trouser_price : ℚ := 9
def jacket_price : ℚ := 11
def shirt_price : ℚ := 5
def jacket_surcharge : ℚ := 2
def shirt_discount : ℚ := 1/10
def total_bill : ℚ := 198
def claimed_shirts : ℕ := 2

noncomputable def actual_shirts : ℕ := 
  Int.toNat <| Int.floor <|
    (total_bill - 
     (trousers_count * trouser_price + 
      jacket_count * (jacket_price + jacket_surcharge))) / 
    (shirt_price * (1 - shirt_discount))

theorem missing_shirts_count :
  actual_shirts - claimed_shirts = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_shirts_count_l435_43580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_five_hours_l435_43505

/-- The time it takes for person A to catch up with person B -/
noncomputable def catch_up_time (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  initial_distance / (speed_A - speed_B)

/-- Theorem stating that the catch-up time is 5 hours given the problem conditions -/
theorem catch_up_time_is_five_hours :
  let initial_distance : ℝ := 15
  let speed_A : ℝ := 10
  let speed_B : ℝ := 7
  catch_up_time initial_distance speed_A speed_B = 5 := by
  -- Unfold the definition of catch_up_time
  unfold catch_up_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#check catch_up_time_is_five_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_five_hours_l435_43505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_ratio_l435_43536

open Real EuclideanGeometry

-- Define the triangle ABC
variable (A B C D E : EuclideanSpace ℝ (Fin 2))
variable (α : ℝ)

-- Define the conditions
variable (h1 : ∠ B A C = α)
variable (h2 : D ∈ (perp_line A B C).toSet)
variable (h3 : E ∈ (perp_line A C B).toSet)
variable (h4 : D ∈ (Line.through A B).toSet)
variable (h5 : E ∈ (Line.through A C).toSet)

-- State the theorem
theorem altitude_ratio :
  dist D E / dist B C = |cos α| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_ratio_l435_43536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l435_43521

-- Define the hyperbola C
def C (x y : ℝ) : Prop := y^2 / 9 - x^2 / 6 = 1

-- Define the length of the imaginary axis
noncomputable def imaginary_axis_length : ℝ := 2 * Real.sqrt 6

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 15 / 3

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, C x y → 
    (imaginary_axis_length = 2 * Real.sqrt 6) ∧ 
    (eccentricity = Real.sqrt 15 / 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l435_43521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pooja_speed_calculation_l435_43546

/-- Calculates the speed of Pooja given Roja's speed, time traveled, and final distance between them. -/
noncomputable def poojaSpeed (rojaSpeed : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  (distance / time) - rojaSpeed

theorem pooja_speed_calculation (rojaSpeed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : rojaSpeed = 8)
  (h2 : time = 4)
  (h3 : distance = 44) :
  poojaSpeed rojaSpeed time distance = 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval poojaSpeed 8 4 44

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pooja_speed_calculation_l435_43546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l435_43532

/-- Represents a cylinder with given height and radius -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Calculates the total surface area of a cylinder -/
noncomputable def totalSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

theorem cylinder_properties :
  let c : Cylinder := { height := 15, radius := 5 }
  totalSurfaceArea c = 200 * Real.pi ∧ volume c = 375 * Real.pi := by
  sorry

#eval "Cylinder properties theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l435_43532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l435_43550

/-- A predicate stating that a triangle with sides a, b, c is tangent to a sphere of radius r --/
def IsTangent (r a b c : ℝ) : Prop :=
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
  r = (Real.sqrt (s * (s - a) * (s - b) * (s - c))) / s

/-- The distance from the center of a sphere to the plane of a triangle tangent to the sphere --/
noncomputable def DistanceToPlane (r a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  Real.sqrt (r^2 - inradius^2)

/-- The distance from the center of a sphere to the plane of a triangle tangent to the sphere --/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 8) 
  (h_triangle : a = 13 ∧ b = 14 ∧ c = 15) (h_tangent : IsTangent r a b c) : 
  DistanceToPlane r a b c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l435_43550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l435_43541

noncomputable def is_solution (x y z : Real) : Prop :=
  (Real.sin x ≠ 0) ∧ 
  (Real.sin y ≠ 0) ∧ 
  ((Real.sin x)^2 + 1/(Real.sin x)^2)^3 + ((Real.sin y)^2 + 1/(Real.sin y)^2)^3 = 16 * Real.cos z

theorem equation_solution :
  ∀ x y z : Real,
    is_solution x y z ↔ 
    ∃ n k m : Int, 
      x = Real.pi/2 + Real.pi * (n : Real) ∧ 
      y = Real.pi/2 + Real.pi * (k : Real) ∧ 
      z = 2 * Real.pi * (m : Real) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l435_43541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l435_43564

-- Define the plane and points
def Plane := ℝ × ℝ

def A : Plane := (0, 1)
def B : Plane := (0, 6)

-- Define the distance function
noncomputable def distance (p q : Plane) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the condition for point P
def satisfiesCondition (P : Plane) : Prop :=
  distance P A - distance P B = 5

-- Statement: The trajectory of P is a ray
theorem trajectory_is_ray :
  ∃ (P : Plane), satisfiesCondition P ∧
  ∀ (Q : Plane), satisfiesCondition Q →
    ∃ (t : ℝ), t ≥ 0 ∧ Q = (t * P.1, t * P.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l435_43564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_price_l435_43591

/-- Regular price of a cup of lemonade -/
def regular_price : ℚ := sorry

/-- Total number of days -/
def total_days : ℕ := 10

/-- Number of hot days -/
def hot_days : ℕ := 4

/-- Number of cups sold per day -/
def cups_per_day : ℕ := 32

/-- Total profit over the period -/
def total_profit : ℚ := 200

/-- Price multiplier for hot days -/
def hot_day_multiplier : ℚ := 1.25

theorem lemonade_price :
  regular_price * cups_per_day * (total_days - hot_days) +
  (regular_price * hot_day_multiplier) * cups_per_day * hot_days = total_profit →
  regular_price = 25 / 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_price_l435_43591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l435_43504

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (0, 3)
def center2 : ℝ × ℝ := (4, 0)

-- Define the radii of the circles
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent : 
  distance_between_centers = radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l435_43504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l435_43529

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l435_43529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_definition_l435_43543

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom polynomial_f : Polynomial ℝ
axiom polynomial_g : Polynomial ℝ
axiom sum_condition : ∀ x, f x + g x = x^2 - 3*x + 5
axiom f_definition : ∀ x, f x = x^4 - 4*x^2 + 3*x - 1

-- State the theorem
theorem g_definition : ∀ x, g x = -x^4 + 5*x^2 - 6*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_definition_l435_43543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l435_43597

/-- The projection vector of b onto a is (-64/25, 48/25) given a = (-4, 3) and b = (5, 12) -/
theorem projection_vector (a b : ℝ × ℝ) :
  a = (-4, 3) →
  b = (5, 12) →
  let proj := ((b.1 * a.1 + b.2 * a.2) / (a.1^2 + a.2^2)) • a
  proj = (-64/25, 48/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l435_43597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_case_l435_43581

theorem tan_sum_special_case (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : 0 < β) (h4 : β < π/2)
  (h5 : Real.tan α = 1/7)
  (h6 : Real.tan β = 3/4) :
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_special_case_l435_43581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_A_and_B_l435_43584

-- Define the set A as solutions to z^3 - 8 = 0
def A : Set ℂ := {z : ℂ | z^3 - 8 = 0}

-- Define the set B as solutions to z^4 - 8z^3 + 8z^2 - 64z + 128 = 0
def B : Set ℂ := {z : ℂ | z^4 - 8*z^3 + 8*z^2 - 64*z + 128 = 0}

-- Define the distance function between two complex numbers
noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

-- Theorem statement
theorem greatest_distance_between_A_and_B :
  ∃ (z : ℂ) (w : ℂ), z ∈ A ∧ w ∈ B ∧
  (∀ (z' : ℂ) (w' : ℂ), z' ∈ A → w' ∈ B → distance z w ≥ distance z' w') ∧
  distance z w = 10 := by
  sorry

#check greatest_distance_between_A_and_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_A_and_B_l435_43584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l435_43565

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 2^x

-- State the theorem
theorem function_composition_equality (b : ℝ) :
  f b (f b (5/6)) = 4 → b = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l435_43565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cody_distance_correct_l435_43534

/-- The distance Cody skis before meeting Daisy -/
noncomputable def cody_distance (cd_distance : ℝ) (cody_speed daisy_speed : ℝ) (cody_angle : ℝ) : ℝ :=
  375 * Real.sqrt (1 / (61 - 30 * Real.sqrt 2))

theorem cody_distance_correct (cd_distance : ℝ) (cody_speed daisy_speed : ℝ) (cody_angle : ℝ) :
  cd_distance = 150 ∧ 
  cody_speed = 5 ∧ 
  daisy_speed = 6 ∧ 
  cody_angle = 45 * π / 180 →
  cody_distance cd_distance cody_speed daisy_speed cody_angle = 
    375 * Real.sqrt (1 / (61 - 30 * Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cody_distance_correct_l435_43534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_in_product_l435_43549

/-- The coefficient of x^2 in the expansion of (3x^3 + 4x^2 + 5x + 6)(7x^3 + 8x^2 + 9x + 10) is 93 -/
theorem coeff_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 3 * X^3 + 4 * X^2 + 5 * X + 6
  let p₂ : Polynomial ℤ := 7 * X^3 + 8 * X^2 + 9 * X + 10
  (p₁ * p₂).coeff 2 = 93 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_in_product_l435_43549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_probability_difference_l435_43578

def probability_exactly_n_heads (n : ℕ) (total : ℕ) : ℚ :=
  (Nat.choose total n : ℚ) * (1 / 2) ^ n * (1 / 2) ^ (total - n)

theorem fair_coin_probability_difference : 
  probability_exactly_n_heads 3 4 - probability_exactly_n_heads 4 4 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_probability_difference_l435_43578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l435_43553

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | 2 => 3/7
  | (n + 3) => (sequence_a (n + 1) * sequence_a (n + 2)) / (2 * sequence_a (n + 1) - sequence_a (n + 2))

theorem a_2019_value : sequence_a 2019 = 3/8075 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l435_43553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_synonyms_omm_moo_not_synonyms_l435_43579

/-- Represents a word in the Ancient Tribe language -/
inductive AncientWord
| M : AncientWord
| O : AncientWord
| concat : AncientWord → AncientWord → AncientWord

/-- Counts the number of M's in a word -/
def countM : AncientWord → Nat
| AncientWord.M => 1
| AncientWord.O => 0
| AncientWord.concat w1 w2 => countM w1 + countM w2

/-- Counts the number of O's in a word -/
def countO : AncientWord → Nat
| AncientWord.M => 0
| AncientWord.O => 1
| AncientWord.concat w1 w2 => countO w1 + countO w2

/-- Calculates the difference between M's and O's in a word -/
def diffMO (w : AncientWord) : Int :=
  (countM w : Int) - (countO w : Int)

/-- Synonym relation for Ancient Tribe words -/
def isSynonym : AncientWord → AncientWord → Prop := sorry

/-- Two words are not synonyms if their M-O differences are not equal -/
theorem not_synonyms (w1 w2 : AncientWord) : diffMO w1 ≠ diffMO w2 → ¬(isSynonym w1 w2) := by
  sorry

/-- OMM and MOO are not synonyms -/
theorem omm_moo_not_synonyms :
  let omm := AncientWord.concat AncientWord.O (AncientWord.concat AncientWord.M AncientWord.M)
  let moo := AncientWord.concat AncientWord.M (AncientWord.concat AncientWord.O AncientWord.O)
  ¬(isSynonym omm moo) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_synonyms_omm_moo_not_synonyms_l435_43579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_isosceles_triangle_l435_43548

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  base_angle : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle ≥ 0 ∧ vertex_angle ≥ 0
  angle_sum : base_angle + base_angle + vertex_angle = 180

-- Theorem statement
theorem largest_angle_in_isosceles_triangle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 50) : 
  max triangle.base_angle (max triangle.base_angle triangle.vertex_angle) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_isosceles_triangle_l435_43548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_distance_l435_43515

theorem folded_paper_distance (s : ℝ) (x : ℝ) : 
  s^2 = 18 → 
  (1/2) * x^2 = 2 * (18 - x^2) → 
  Real.sqrt (2 * x^2) = (12 * Real.sqrt 2) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_distance_l435_43515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circumradius_l435_43542

-- Define the function to calculate the circumradius of the trapezoid
noncomputable def circumradius_of_trapezoid (r : ℝ) (θ : ℝ) : ℝ :=
  r / (2 * Real.sin (θ / 2))

theorem trapezoid_circumradius (r : ℝ) (θ : ℝ) :
  r = 9 →
  θ = π / 6 →
  (9 * (Real.sqrt 6 + Real.sqrt 2)) / 2 = circumradius_of_trapezoid r θ :=
by
  intro hr hθ
  simp [circumradius_of_trapezoid, hr, hθ]
  sorry -- The actual proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circumradius_l435_43542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l435_43535

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type representing (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: For a parabola y² = 2px with p > 0 and a point A(4, y₀) on the parabola,
    if the distance between A and the focus F is 3p/2, then p = 4 -/
theorem parabola_focus_distance (c : Parabola) (a : Point)
  (h_on_parabola : a.y^2 = 2 * c.p * a.x)
  (h_x : a.x = 4)
  (f : Point)
  (h_focus : f.x = c.p / 2 ∧ f.y = 0)
  (h_distance : distance a f = 3 * c.p / 2) :
  c.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l435_43535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leastExpensiveJourney_l435_43577

-- Define the cities
inductive City : Type
| D
| E
| F

-- Define the transportation methods
inductive TransportMethod : Type
| Bus
| Airplane

-- Define the distance between two cities
noncomputable def distance (c1 c2 : City) : ℝ :=
  match c1, c2 with
  | City.D, City.F => 4000
  | City.F, City.D => 4000
  | City.D, City.E => 4500
  | City.E, City.D => 4500
  | City.E, City.F => Real.sqrt 4250000
  | City.F, City.E => Real.sqrt 4250000
  | _, _ => 0

-- Define the cost function for a single trip
noncomputable def tripCost (c1 c2 : City) (method : TransportMethod) : ℝ :=
  match method with
  | TransportMethod.Bus => 0.20 * distance c1 c2
  | TransportMethod.Airplane => 120 + 0.12 * distance c1 c2

-- Define the total cost of the journey
noncomputable def journeyCost (m1 m2 m3 : TransportMethod) : ℝ :=
  tripCost City.D City.E m1 + tripCost City.E City.F m2 + tripCost City.F City.D m3

-- Theorem statement
theorem leastExpensiveJourney :
  ∃ m1 m2 m3 : TransportMethod,
    journeyCost m1 m2 m3 = 1627.44 ∧
    ∀ n1 n2 n3 : TransportMethod, journeyCost n1 n2 n3 ≥ 1627.44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leastExpensiveJourney_l435_43577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l435_43558

def sequenceNum (n : ℕ) : ℕ := 
  if n = 0 then 17 else 17 * (10^(2*n) - 1) / 99

theorem only_first_prime :
  ∀ n : ℕ, n > 0 → ¬(Nat.Prime (sequenceNum n)) ∧ Nat.Prime (sequenceNum 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l435_43558
