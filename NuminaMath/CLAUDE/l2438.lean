import Mathlib

namespace NUMINAMATH_CALUDE_line_vector_at_4_l2438_243894

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_4 :
  (∃ (a d : ℝ × ℝ × ℝ),
    (∀ t : ℝ, line_vector t = a + t • d) ∧
    line_vector (-2) = (2, 6, 16) ∧
    line_vector 1 = (-1, -4, -8)) →
  line_vector 4 = (-4, -10, -32) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_at_4_l2438_243894


namespace NUMINAMATH_CALUDE_vector_magnitude_l2438_243802

/-- Given vectors a and b, if a is collinear with a + b, then |a - b| = 2√5 -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∃ (k : ℝ), a = k • (a + b)) →
  ‖a - b‖ = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2438_243802


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_ratios_ge_two_l2438_243840

theorem sum_of_reciprocal_ratios_ge_two (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_ratios_ge_two_l2438_243840


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_twelve_l2438_243831

/-- Given a function y of x, prove that a + b = 12 -/
theorem sum_of_a_and_b_is_twelve 
  (y : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, y x = a + b / (x + 1))
  (h2 : y (-2) = 2)
  (h3 : y (-6) = 6) :
  a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_twelve_l2438_243831


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2438_243841

def integerRange : List Int := List.range 10 |> List.map (λ x => x - 4)

theorem arithmetic_mean_of_range : 
  (integerRange.sum : ℚ) / integerRange.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2438_243841


namespace NUMINAMATH_CALUDE_crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l2438_243810

-- Define the regression equation
def crew_estimate (tonnage : ℝ) : ℝ := 9.5 + 0.0062 * tonnage

-- Define the tonnage range
def min_tonnage : ℝ := 192
def max_tonnage : ℝ := 3246

-- Theorem 1: Difference in crew members for 1000 tons difference
theorem crew_diff_1000_tons : 
  ∀ (x : ℝ), crew_estimate (x + 1000) - crew_estimate x = 6 := by sorry

-- Theorem 2: Estimated crew for minimum tonnage
theorem crew_estimate_min_tonnage : 
  ⌊crew_estimate min_tonnage⌋ = 11 := by sorry

-- Theorem 3: Estimated crew for maximum tonnage
theorem crew_estimate_max_tonnage : 
  ⌊crew_estimate max_tonnage⌋ = 30 := by sorry

end NUMINAMATH_CALUDE_crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l2438_243810


namespace NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l2438_243833

/-- The maximum number of faces perpendicular to the base in an n-sided pyramid -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of faces perpendicular to the base in an n-sided pyramid -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 2) :
  max_perpendicular_faces n = if n % 2 = 0 then n / 2 else (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l2438_243833


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2438_243890

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y + 1 = 0 -/
theorem intersection_line_equation 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁ * 2 + b₁ * 3 + 1 = 0)
  (h₂ : a₂ * 2 + b₂ * 3 + 1 = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧ 2 * a₁ + 3 * b₁ + k = 0 ∧ 2 * a₂ + 3 * b₂ + k = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2438_243890


namespace NUMINAMATH_CALUDE_hcl_moles_equal_one_l2438_243807

-- Define the chemical reaction
structure Reaction where
  naoh : ℕ  -- moles of Sodium hydroxide
  hcl : ℕ   -- moles of Hydrochloric acid
  h2o : ℕ   -- moles of Water produced

-- Define the balanced reaction
def balanced_reaction (r : Reaction) : Prop :=
  r.naoh = r.hcl ∧ r.naoh = r.h2o

-- Theorem statement
theorem hcl_moles_equal_one (r : Reaction) 
  (h1 : r.naoh = 1)  -- 1 mole of Sodium hydroxide is used
  (h2 : r.h2o = 1)   -- The reaction produces 1 mole of Water
  (h3 : balanced_reaction r) : -- The reaction is balanced
  r.hcl = 1 := by sorry

end NUMINAMATH_CALUDE_hcl_moles_equal_one_l2438_243807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l2438_243836

-- Arithmetic Sequence
theorem arithmetic_sequence_problem (d n a_n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  let a_1 := a_n - (n - 1) * d
  let S_n := n * (a_1 + a_n) / 2
  a_1 = -38 ∧ S_n = -360 := by sorry

-- Geometric Sequence
theorem geometric_sequence_problem (a_2 a_3 a_4 : ℚ) (h1 : a_2 + a_3 = 6) (h2 : a_3 + a_4 = 12) :
  let q := a_3 / a_2
  let a_1 := a_2 / q
  let S_10 := a_1 * (1 - q^10) / (1 - q)
  q = 2 ∧ S_10 = 1023 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l2438_243836


namespace NUMINAMATH_CALUDE_sum_of_digits_in_special_number_l2438_243885

theorem sum_of_digits_in_special_number (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  A ≠ B → A ≠ C → A ≠ D → A ≠ E → 
  B ≠ C → B ≠ D → B ≠ E → 
  C ≠ D → C ≠ E → 
  D ≠ E →
  (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E) % 9 = 0 →
  A + B + C + D + E = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_special_number_l2438_243885


namespace NUMINAMATH_CALUDE_actual_average_height_l2438_243883

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 183

/-- The incorrectly recorded height of the first boy in cm -/
def incorrect_height1 : ℝ := 166

/-- The actual height of the first boy in cm -/
def actual_height1 : ℝ := 106

/-- The incorrectly recorded height of the second boy in cm -/
def incorrect_height2 : ℝ := 175

/-- The actual height of the second boy in cm -/
def actual_height2 : ℝ := 190

/-- Conversion factor from cm to feet -/
def cm_to_feet : ℝ := 30.48

/-- Theorem stating that the actual average height of the boys is approximately 5.98 feet -/
theorem actual_average_height :
  let total_height := num_boys * initial_avg
  let corrected_total := total_height - (incorrect_height1 - actual_height1) + (actual_height2 - incorrect_height2)
  let actual_avg_cm := corrected_total / num_boys
  let actual_avg_feet := actual_avg_cm / cm_to_feet
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |actual_avg_feet - 5.98| < ε :=
by sorry

end NUMINAMATH_CALUDE_actual_average_height_l2438_243883


namespace NUMINAMATH_CALUDE_composite_figure_area_l2438_243849

/-- The area of a composite figure with specific properties -/
theorem composite_figure_area : 
  let equilateral_triangle_area := Real.sqrt 3 / 4
  let rectangle_area := 1
  let right_triangle_area := 1 / 2
  2 * equilateral_triangle_area + rectangle_area + right_triangle_area = Real.sqrt 3 / 2 + 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_figure_area_l2438_243849


namespace NUMINAMATH_CALUDE_difference_between_results_l2438_243813

theorem difference_between_results (x : ℝ) (h : x = 15) : 2 * x - (26 - x) = 19 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_results_l2438_243813


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2438_243863

theorem fraction_sum_equality : 
  (2 : ℚ) / 100 + 5 / 1000 + 5 / 10000 + 3 * (4 / 1000) = 375 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2438_243863


namespace NUMINAMATH_CALUDE_finish_time_l2438_243855

/-- Time (in days) for A and B to finish a can of coffee together -/
def coffee_together : ℝ := 10

/-- Time (in days) for A to finish a can of coffee alone -/
def coffee_A : ℝ := 12

/-- Time (in days) for A and B to finish a pound of tea together -/
def tea_together : ℝ := 12

/-- Time (in days) for B to finish a pound of tea alone -/
def tea_B : ℝ := 20

/-- A won't drink coffee if there's tea, and B won't drink tea if there's coffee -/
axiom preference : True

/-- The time it takes for A and B to finish a pound of tea and a can of coffee -/
def total_time : ℝ := 35

theorem finish_time : total_time = 35 := by sorry

end NUMINAMATH_CALUDE_finish_time_l2438_243855


namespace NUMINAMATH_CALUDE_bridget_middle_score_l2438_243851

/-- Represents the test scores of the four students -/
structure Scores where
  hannah : ℝ
  ella : ℝ
  cassie : ℝ
  bridget : ℝ

/-- Defines the conditions given in the problem -/
def SatisfiesConditions (s : Scores) : Prop :=
  (s.cassie > s.hannah) ∧ (s.cassie > s.ella) ∧
  (s.bridget ≥ s.hannah) ∧ (s.bridget ≥ s.ella)

/-- Defines what it means for a student to have the middle score -/
def HasMiddleScore (name : String) (s : Scores) : Prop :=
  match name with
  | "Bridget" => (s.bridget > min s.hannah s.ella) ∧ (s.bridget < max s.cassie s.ella)
  | _ => False

/-- The main theorem stating that if the conditions are satisfied, Bridget must have the middle score -/
theorem bridget_middle_score (s : Scores) :
  SatisfiesConditions s → HasMiddleScore "Bridget" s := by
  sorry


end NUMINAMATH_CALUDE_bridget_middle_score_l2438_243851


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2438_243869

theorem solve_quadratic_equation (x : ℝ) :
  2 * (x - 3)^2 - 98 = 0 → x = 10 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2438_243869


namespace NUMINAMATH_CALUDE_tree_table_profit_l2438_243880

/-- Calculates the profit from selling tables made from chopped trees --/
theorem tree_table_profit
  (trees : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (h1 : trees = 30)
  (h2 : planks_per_tree = 25)
  (h3 : planks_per_table = 15)
  (h4 : price_per_table = 300)
  (h5 : labor_cost = 3000)
  : (trees * planks_per_tree / planks_per_table) * price_per_table - labor_cost = 12000 := by
  sorry

#check tree_table_profit

end NUMINAMATH_CALUDE_tree_table_profit_l2438_243880


namespace NUMINAMATH_CALUDE_expected_balls_theorem_l2438_243832

/-- Represents a system of balls arranged in a circle -/
structure BallSystem :=
  (n : ℕ)  -- number of balls

/-- Represents a swap operation on the ball system -/
structure SwapOperation :=
  (isAdjacent : Bool)  -- whether the swap is between adjacent balls only

/-- Calculates the probability of a ball remaining in its original position after a swap -/
def probabilityAfterSwap (sys : BallSystem) (op : SwapOperation) : ℚ :=
  if op.isAdjacent then
    (sys.n - 2 : ℚ) / sys.n * 2 / 3 + 2 / sys.n
  else
    (sys.n - 2 : ℚ) / sys.n

/-- Calculates the expected number of balls in their original positions after two swaps -/
def expectedBallsInOriginalPosition (sys : BallSystem) (op1 op2 : SwapOperation) : ℚ :=
  sys.n * probabilityAfterSwap sys op1 * probabilityAfterSwap sys op2

theorem expected_balls_theorem (sys : BallSystem) (op1 op2 : SwapOperation) :
  sys.n = 8 ∧ ¬op1.isAdjacent ∧ op2.isAdjacent →
  expectedBallsInOriginalPosition sys op1 op2 = 2 := by
  sorry

#eval expectedBallsInOriginalPosition ⟨8⟩ ⟨false⟩ ⟨true⟩

end NUMINAMATH_CALUDE_expected_balls_theorem_l2438_243832


namespace NUMINAMATH_CALUDE_data_grouping_l2438_243811

theorem data_grouping (data : Set ℤ) (max_val min_val class_interval : ℤ) :
  max_val = 42 →
  min_val = 8 →
  class_interval = 5 →
  ∀ x ∈ data, min_val ≤ x ∧ x ≤ max_val →
  (max_val - min_val) / class_interval + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_data_grouping_l2438_243811


namespace NUMINAMATH_CALUDE_tenth_term_is_39_l2438_243838

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  second_term : a + d = 7
  fifth_term : a + 4 * d = 19

/-- The tenth term of the arithmetic sequence is 39 -/
theorem tenth_term_is_39 (seq : ArithmeticSequence) : seq.a + 9 * seq.d = 39 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_39_l2438_243838


namespace NUMINAMATH_CALUDE_smallest_d_value_l2438_243837

theorem smallest_d_value (c d : ℕ+) (h1 : c - d = 8) 
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : 
  d ≥ 4 ∧ ∃ (c' d' : ℕ+), c' - d' = 8 ∧ 
    Nat.gcd ((c'^3 + d'^3) / (c' + d')) (c' * d') = 16 ∧ d' = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l2438_243837


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l2438_243814

/-- Given 7 bugs, each eating 4 flowers, prove that the total number of flowers eaten is 28. -/
theorem bugs_eating_flowers :
  let number_of_bugs : ℕ := 7
  let flowers_per_bug : ℕ := 4
  number_of_bugs * flowers_per_bug = 28 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l2438_243814


namespace NUMINAMATH_CALUDE_value_range_of_f_l2438_243808

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem value_range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -3 ≤ y ∧ y ≤ 1} := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l2438_243808


namespace NUMINAMATH_CALUDE_carls_garden_area_l2438_243848

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_distance : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * (g.longer_side_posts - 1) * g.post_distance * g.post_distance

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area :
  ∀ g : Garden,
    g.total_posts = 36 ∧
    g.post_distance = 6 ∧
    g.longer_side_posts = 3 * g.shorter_side_posts ∧
    g.total_posts = 2 * (g.longer_side_posts + g.shorter_side_posts - 2) →
    garden_area g = 2016 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l2438_243848


namespace NUMINAMATH_CALUDE_sum_binomial_congruence_l2438_243881

theorem sum_binomial_congruence (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (∑' j, Nat.choose p j * Nat.choose (p + j) j) ≡ (2^p + 1) [ZMOD p^2] := by
  sorry

end NUMINAMATH_CALUDE_sum_binomial_congruence_l2438_243881


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l2438_243834

-- Define the plot dimensions
def length : ℝ := 60
def breadth : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 26.50

-- Calculate the perimeter
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
def total_cost : ℝ := perimeter * cost_per_meter

-- Theorem to prove
theorem fencing_cost_calculation :
  total_cost = 5300 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l2438_243834


namespace NUMINAMATH_CALUDE_janet_change_l2438_243854

def muffin_price : ℚ := 75 / 100
def num_muffins : ℕ := 12
def amount_paid : ℚ := 20

theorem janet_change :
  amount_paid - (num_muffins : ℚ) * muffin_price = 11 := by
  sorry

end NUMINAMATH_CALUDE_janet_change_l2438_243854


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2438_243828

theorem consecutive_integers_product_sum (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 930 → x + (x + 1) = 61 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2438_243828


namespace NUMINAMATH_CALUDE_mean_proportional_sqrt45_and_7_3_pi_l2438_243839

theorem mean_proportional_sqrt45_and_7_3_pi :
  let a := Real.sqrt 45
  let b := 7/3 * Real.pi
  Real.sqrt (a * b) = Real.sqrt (7 * Real.sqrt 5 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_sqrt45_and_7_3_pi_l2438_243839


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2438_243820

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  smallBase : ℝ
  largeBase : ℝ
  acuteAngle : ℝ
  heightPerp : ℝ

/-- A solid of revolution formed by rotating an isosceles trapezoid around its smaller base -/
structure SolidOfRevolution where
  trapezoid : IsoscelesTrapezoid
  surfaceArea : ℝ
  volume : ℝ

/-- The theorem stating the surface area and volume of the solid of revolution -/
theorem isosceles_trapezoid_rotation (t : IsoscelesTrapezoid) 
  (h1 : t.smallBase = 2)
  (h2 : t.largeBase = 3)
  (h3 : t.acuteAngle = π / 3)
  (h4 : t.heightPerp = 3) :
  ∃ (s : SolidOfRevolution), 
    s.trapezoid = t ∧ 
    s.surfaceArea = 4 * π * Real.sqrt 3 ∧ 
    s.volume = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2438_243820


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2438_243816

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 12) → (a + b + c = 33) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2438_243816


namespace NUMINAMATH_CALUDE_no_solution_condition_l2438_243886

theorem no_solution_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔ 
  (0 < b ∧ b < 4) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2438_243886


namespace NUMINAMATH_CALUDE_max_k_value_l2438_243809

open Real

noncomputable def f (x : ℝ) : ℝ := x * (1 + log x)

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2438_243809


namespace NUMINAMATH_CALUDE_arthur_walked_five_and_half_miles_l2438_243806

/-- The distance Arthur walked in miles -/
def arthurs_distance (east west north : ℕ) (block_length : ℚ) : ℚ :=
  (east + west + north : ℚ) * block_length

/-- Proof that Arthur walked 5.5 miles -/
theorem arthur_walked_five_and_half_miles :
  arthurs_distance 8 4 10 (1/4) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_five_and_half_miles_l2438_243806


namespace NUMINAMATH_CALUDE_complex_linear_combination_l2438_243817

theorem complex_linear_combination (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - 3*I) :
  2*a + 3*b = 12 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_linear_combination_l2438_243817


namespace NUMINAMATH_CALUDE_train_passengers_count_l2438_243819

/-- Represents the number of passengers in each carriage of a train -/
structure TrainCarriages :=
  (c1 c2 c3 c4 c5 : ℕ)

/-- Defines the condition for the number of neighbours a passenger has -/
def valid_neighbours (tc : TrainCarriages) : Prop :=
  ∀ i : Fin 5, 
    let neighbours := match i with
      | 0 => tc.c1 - 1 + tc.c2
      | 1 => tc.c1 + tc.c2 - 1 + tc.c3
      | 2 => tc.c2 + tc.c3 - 1 + tc.c4
      | 3 => tc.c3 + tc.c4 - 1 + tc.c5
      | 4 => tc.c4 + tc.c5 - 1
    (neighbours = 5 ∨ neighbours = 10)

/-- The main theorem stating that under the given conditions, 
    the total number of passengers is 17 -/
theorem train_passengers_count (tc : TrainCarriages) 
  (h1 : tc.c1 ≥ 1 ∧ tc.c2 ≥ 1 ∧ tc.c3 ≥ 1 ∧ tc.c4 ≥ 1 ∧ tc.c5 ≥ 1)
  (h2 : valid_neighbours tc) : 
  tc.c1 + tc.c2 + tc.c3 + tc.c4 + tc.c5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_count_l2438_243819


namespace NUMINAMATH_CALUDE_power_five_mod_thirteen_l2438_243896

theorem power_five_mod_thirteen : 5^2006 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_thirteen_l2438_243896


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l2438_243822

/-- A function g : ℝ → ℝ with the property that g(x) = g(3-x) for all x ∈ ℝ -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g -/
def IsAxisOfSymmetry (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) (h : SymmetricFunction g) :
  IsAxisOfSymmetry g := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_l2438_243822


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2438_243857

/-- Proves that given a shirt with a list price of 150, a final price of 105 after two successive discounts, and a second discount of 12.5%, the first discount percentage is 20%. -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5) :
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧ 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2438_243857


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2438_243875

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubseq (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sum (a : ℕ → ℤ) 
  (h_arith : arithmeticSeq a) (h_geom : geometricSubseq a) : 
  a 2 + a 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2438_243875


namespace NUMINAMATH_CALUDE_pi_estimation_l2438_243887

theorem pi_estimation (n m : ℕ) (h1 : n = 100) (h2 : m = 31) :
  let π_est := 4 * (n : ℝ) / (m : ℝ) - 3
  π_est = 81 / 25 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l2438_243887


namespace NUMINAMATH_CALUDE_vector_BC_l2438_243850

/-- Given two vectors AB and AC in 2D space, prove that the vector BC is their difference -/
theorem vector_BC (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, -1]) (h2 : AC = ![-4, 1]) :
  AC - AB = ![-6, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l2438_243850


namespace NUMINAMATH_CALUDE_rem_five_sevenths_three_fourths_l2438_243873

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_five_sevenths_three_fourths :
  rem (5/7) (3/4) = 5/7 := by sorry

end NUMINAMATH_CALUDE_rem_five_sevenths_three_fourths_l2438_243873


namespace NUMINAMATH_CALUDE_count_with_3_or_7_l2438_243825

/-- The set of digits that are neither 3 nor 7 -/
def other_digits : Finset Nat := {0, 1, 2, 4, 5, 6, 8, 9}

/-- The set of non-zero digits that are neither 3 nor 7 -/
def non_zero_other_digits : Finset Nat := {1, 2, 4, 5, 6, 8, 9}

/-- The count of four-digit numbers without 3 or 7 -/
def count_without_3_or_7 : Nat :=
  (Finset.card non_zero_other_digits) * (Finset.card other_digits)^3

/-- The total count of four-digit numbers -/
def total_four_digit_numbers : Nat := 9000

theorem count_with_3_or_7 :
  total_four_digit_numbers - count_without_3_or_7 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_count_with_3_or_7_l2438_243825


namespace NUMINAMATH_CALUDE_luthers_line_has_17_pieces_l2438_243843

/-- Represents Luther's latest clothing line -/
structure ClothingLine where
  silk_pieces : ℕ
  cashmere_pieces : ℕ
  blended_pieces : ℕ

/-- Calculates the total number of pieces in the clothing line -/
def total_pieces (line : ClothingLine) : ℕ :=
  line.silk_pieces + line.cashmere_pieces + line.blended_pieces

/-- Theorem: Luther's latest line has 17 pieces -/
theorem luthers_line_has_17_pieces :
  ∃ (line : ClothingLine),
    line.silk_pieces = 10 ∧
    line.cashmere_pieces = line.silk_pieces / 2 ∧
    line.blended_pieces = 2 ∧
    total_pieces line = 17 := by
  sorry

end NUMINAMATH_CALUDE_luthers_line_has_17_pieces_l2438_243843


namespace NUMINAMATH_CALUDE_inverse_proportion_value_l2438_243877

/-- For the inverse proportion function y = -8/x, when x = -2, y = 4 -/
theorem inverse_proportion_value : 
  let f : ℝ → ℝ := λ x => -8 / x
  f (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_value_l2438_243877


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l2438_243835

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def g (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l2438_243835


namespace NUMINAMATH_CALUDE_product_of_digits_3545_l2438_243882

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def units_digit (n : ℕ) : ℕ :=
  n % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem product_of_digits_3545 :
  let n := 3545
  ¬ is_divisible_by_3 n ∧ units_digit n * tens_digit n = 20 :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_3545_l2438_243882


namespace NUMINAMATH_CALUDE_stratified_sampling_l2438_243801

theorem stratified_sampling (total_employees : ℕ) (male_employees : ℕ) (sample_size : ℕ) :
  total_employees = 750 →
  male_employees = 300 →
  sample_size = 45 →
  (sample_size - (male_employees * sample_size / total_employees) : ℕ) = 27 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2438_243801


namespace NUMINAMATH_CALUDE_alicia_remaining_art_l2438_243878

/-- Represents the types of art in Alicia's collection -/
inductive ArtType
  | Medieval
  | Renaissance
  | Modern

/-- Calculates the remaining art pieces after donation -/
def remaining_art (initial : Nat) (donate_percent : Nat) : Nat :=
  initial - (initial * donate_percent / 100)

/-- Theorem stating the remaining art pieces after Alicia's donations -/
theorem alicia_remaining_art :
  (remaining_art 70 65 = 25) ∧
  (remaining_art 120 30 = 84) ∧
  (remaining_art 150 45 = 83) := by
  sorry

#check alicia_remaining_art

end NUMINAMATH_CALUDE_alicia_remaining_art_l2438_243878


namespace NUMINAMATH_CALUDE_no_quadratic_term_implies_m_value_l2438_243830

theorem no_quadratic_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x + m) * (x^2 + 2*x - 1) = a*x^3 + b*x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_term_implies_m_value_l2438_243830


namespace NUMINAMATH_CALUDE_problem_solution_l2438_243829

theorem problem_solution (a b : ℕ+) : 
  Nat.lcm a b = 2520 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 252 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2438_243829


namespace NUMINAMATH_CALUDE_smallest_n_for_non_integer_expression_l2438_243859

theorem smallest_n_for_non_integer_expression : ∃ n : ℕ, n > 0 ∧ n = 11 ∧
  ∃ k : ℕ, k < n ∧
    (∀ a m : ℕ, a % n = k ∧ m > 0 →
      ¬(∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1))) ∧
    (∀ n' : ℕ, 0 < n' ∧ n' < n →
      ∀ k' : ℕ, k' < n' →
        ∃ a m : ℕ, a % n' = k' ∧ m > 0 ∧
          ∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_n_for_non_integer_expression_l2438_243859


namespace NUMINAMATH_CALUDE_absent_fraction_l2438_243805

theorem absent_fraction (total : ℕ) (present : ℕ) 
  (h1 : total = 28) 
  (h2 : present = 20) : 
  (total - present : ℚ) / total = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_absent_fraction_l2438_243805


namespace NUMINAMATH_CALUDE_equation_implies_ratio_one_third_l2438_243864

theorem equation_implies_ratio_one_third 
  (a x y : ℝ) 
  (h_distinct : a ≠ x ∧ x ≠ y ∧ a ≠ y) 
  (h_eq : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_ratio_one_third_l2438_243864


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2438_243860

/-- The perimeter of a right triangle formed by specific lines --/
theorem triangle_perimeter : 
  ∀ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
  (∃ (m : ℝ), l₁ = {(x, y) | y = m * x}) →  -- l₁ passes through origin
  (l₂ = {(x, y) | x = 2}) →                 -- l₂ is x = 2
  (l₃ = {(x, y) | y = 2 - (Real.sqrt 5 / 5) * x}) →  -- l₃ is y = 2 - (√5/5)x
  (∃ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃) →  -- intersection points exist
  (∃ (v₁ v₂ : ℝ × ℝ), v₁ ∈ l₁ ∧ v₂ ∈ l₃ ∧ (v₁.1 - v₂.1) * (v₁.2 - v₂.2) = 0) →  -- right angle condition
  let perimeter := 2 + (12 * Real.sqrt 5 - 10) / 5 + 2 * Real.sqrt 6
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃ ∧
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) + 
    Real.sqrt ((p₂.1 - p₃.1)^2 + (p₂.2 - p₃.2)^2) + 
    Real.sqrt ((p₃.1 - p₁.1)^2 + (p₃.2 - p₁.2)^2) = perimeter :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2438_243860


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2438_243871

theorem triangle_third_side_length (a b c : ℕ) : 
  a = 2 → b = 14 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (b - a < c ∧ c - a < b ∧ c - b < a) →
  c = 14 := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2438_243871


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l2438_243895

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def drawn : ℕ := 3

/-- The probability of drawing at least one white ball -/
theorem prob_at_least_one_white :
  (1 - (Nat.choose num_red drawn / Nat.choose total_balls drawn : ℚ)) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l2438_243895


namespace NUMINAMATH_CALUDE_dolly_additional_tickets_l2438_243842

/-- The number of additional tickets Dolly needs to buy for amusement park rides -/
theorem dolly_additional_tickets : ℕ := by
  -- Define the number of rides Dolly wants for each attraction
  let ferris_wheel_rides : ℕ := 2
  let roller_coaster_rides : ℕ := 3
  let log_ride_rides : ℕ := 7

  -- Define the cost in tickets for each attraction
  let ferris_wheel_cost : ℕ := 2
  let roller_coaster_cost : ℕ := 5
  let log_ride_cost : ℕ := 1

  -- Define the number of tickets Dolly currently has
  let current_tickets : ℕ := 20

  -- Calculate the total number of tickets needed
  let total_tickets_needed : ℕ := 
    ferris_wheel_rides * ferris_wheel_cost +
    roller_coaster_rides * roller_coaster_cost +
    log_ride_rides * log_ride_cost

  -- Calculate the additional tickets needed
  let additional_tickets : ℕ := total_tickets_needed - current_tickets

  -- Prove that the additional tickets needed is 6
  have h : additional_tickets = 6 := by sorry

  exact 6

end NUMINAMATH_CALUDE_dolly_additional_tickets_l2438_243842


namespace NUMINAMATH_CALUDE_gcd_14m_21n_l2438_243893

theorem gcd_14m_21n (m n : ℕ+) (h : Nat.gcd m n = 18) : Nat.gcd (14 * m) (21 * n) = 126 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14m_21n_l2438_243893


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l2438_243815

theorem profit_percentage_previous_year 
  (revenue_previous : ℝ) 
  (profit_previous : ℝ) 
  (revenue_decline : ℝ) 
  (profit_percentage_current : ℝ) 
  (profit_ratio : ℝ) 
  (h1 : revenue_decline = 0.3)
  (h2 : profit_percentage_current = 0.1)
  (h3 : profit_ratio = 0.6999999999999999)
  (h4 : profit_previous * profit_ratio = 
        (1 - revenue_decline) * revenue_previous * profit_percentage_current) :
  profit_previous / revenue_previous = 0.1 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l2438_243815


namespace NUMINAMATH_CALUDE_amount_left_after_purchases_l2438_243821

def calculate_discounted_price (price : ℚ) (discount_percent : ℚ) : ℚ :=
  price * (1 - discount_percent / 100)

def initial_amount : ℚ := 60

def frame_price : ℚ := 15
def frame_discount : ℚ := 10

def wheel_price : ℚ := 25
def wheel_discount : ℚ := 5

def seat_price : ℚ := 8
def seat_discount : ℚ := 15

def handlebar_price : ℚ := 5
def handlebar_discount : ℚ := 0

def bell_price : ℚ := 3
def bell_discount : ℚ := 0

def hat_price : ℚ := 10
def hat_discount : ℚ := 25

def total_cost : ℚ :=
  calculate_discounted_price frame_price frame_discount +
  calculate_discounted_price wheel_price wheel_discount +
  calculate_discounted_price seat_price seat_discount +
  calculate_discounted_price handlebar_price handlebar_discount +
  calculate_discounted_price bell_price bell_discount +
  calculate_discounted_price hat_price hat_discount

theorem amount_left_after_purchases :
  initial_amount - total_cost = 45 / 100 := by sorry

end NUMINAMATH_CALUDE_amount_left_after_purchases_l2438_243821


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2438_243826

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 28)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2)
  : badminton + tennis - (total - neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2438_243826


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l2438_243870

def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}

theorem subset_implies_m_values (m : ℝ) : Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l2438_243870


namespace NUMINAMATH_CALUDE_direct_proportion_constant_zero_l2438_243891

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- Given that y = x + b is a direct proportion function, b must be zero -/
theorem direct_proportion_constant_zero (b : ℝ) :
  IsDirectProportionFunction (fun x ↦ x + b) → b = 0 := by
  sorry


end NUMINAMATH_CALUDE_direct_proportion_constant_zero_l2438_243891


namespace NUMINAMATH_CALUDE_function_existence_condition_l2438_243858

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
by sorry

end NUMINAMATH_CALUDE_function_existence_condition_l2438_243858


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_15_l2438_243868

theorem three_digit_divisible_by_15 : 
  (Finset.filter (fun k => 100 ≤ 15 * k ∧ 15 * k ≤ 999) (Finset.range 1000)).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_15_l2438_243868


namespace NUMINAMATH_CALUDE_point_d_is_multiple_of_fifteen_l2438_243862

/-- Represents a point on the number line -/
structure Point where
  value : ℤ

/-- Represents the number line with four special points -/
structure NumberLine where
  w : Point
  x : Point
  y : Point
  z : Point
  consecutive : w.value < x.value ∧ x.value < y.value ∧ y.value < z.value
  multiples_of_three : (w.value % 3 = 0 ∧ y.value % 3 = 0) ∨ (x.value % 3 = 0 ∧ z.value % 3 = 0)
  multiples_of_five : (w.value % 5 = 0 ∧ z.value % 5 = 0) ∨ (x.value % 5 = 0 ∧ y.value % 5 = 0)

/-- The point D, which is 5 units away from one of the multiples of 5 -/
def point_d (nl : NumberLine) : Point :=
  if nl.w.value % 5 = 0 then { value := nl.w.value + 5 }
  else if nl.x.value % 5 = 0 then { value := nl.x.value + 5 }
  else if nl.y.value % 5 = 0 then { value := nl.y.value + 5 }
  else { value := nl.z.value + 5 }

theorem point_d_is_multiple_of_fifteen (nl : NumberLine) :
  (point_d nl).value % 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_d_is_multiple_of_fifteen_l2438_243862


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l2438_243866

theorem fraction_to_zero_power :
  let f : ℚ := -574839201 / 1357924680
  f ≠ 0 →
  f^0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l2438_243866


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2438_243884

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is an even function with range (-∞, 4] -/
def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of f is (-∞, 4] -/
def has_range_neg_inf_to_4 (f : ℝ → ℝ) : Prop := 
  (∀ y, y ≤ 4 → ∃ x, f x = y) ∧ (∀ x, f x ≤ 4)

theorem quadratic_function_theorem (a b : ℝ) : 
  is_even_function (f · a b) → has_range_neg_inf_to_4 (f · a b) → 
  ∀ x, f x a b = -2 * x^2 + 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2438_243884


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2438_243844

/-- Given two vectors on a plane with specific properties, prove the magnitude of a third vector. -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = -1/2) →  -- angle between a and b is 120°
  (a.1^2 + a.2^2 = 1) →             -- magnitude of a is 1
  (b.1^2 + b.2^2 = 4) →             -- magnitude of b is 2
  (m.1 * a.1 + m.2 * a.2 = 1) →     -- m · a = 1
  (m.1 * b.1 + m.2 * b.2 = 1) →     -- m · b = 1
  m.1^2 + m.2^2 = 7/3 :=            -- |m|^2 = (√21/3)^2 = 21/9 = 7/3
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2438_243844


namespace NUMINAMATH_CALUDE_colby_mango_harvest_l2438_243856

theorem colby_mango_harvest (x : ℝ) :
  (x ≥ 0) →                             -- Non-negative harvest
  (x - 20 ≥ 0) →                        -- Enough to sell 20 kg to market
  (8 * ((x - 20) / 2) = 160) →          -- 160 mangoes left after sales
  (x = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_colby_mango_harvest_l2438_243856


namespace NUMINAMATH_CALUDE_area_of_curve_l2438_243872

-- Define the curve
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

-- Define the area enclosed by the curve
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_of_curve : enclosed_area = 2 := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l2438_243872


namespace NUMINAMATH_CALUDE_cos_squared_difference_eq_sqrt_three_half_l2438_243824

theorem cos_squared_difference_eq_sqrt_three_half :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_difference_eq_sqrt_three_half_l2438_243824


namespace NUMINAMATH_CALUDE_power_relation_l2438_243804

theorem power_relation (a b : ℕ) : 2^a = 8^(b+1) → 3^a / 27^b = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2438_243804


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l2438_243812

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Define the slope of the asymptote
def AsymptopeSlope (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the eccentricity
def Eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_minimum_value (a b e : ℝ) :
  Hyperbola a b →
  AsymptopeSlope a b →
  Eccentricity a b e →
  a > 0 →
  b > 0 →
  (∀ a' b' e' : ℝ, Hyperbola a' b' → AsymptopeSlope a' b' → Eccentricity a' b' e' →
    a' > 0 → b' > 0 → (a'^2 + e') / b' ≥ (a^2 + e) / b) →
  (a^2 + e) / b = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l2438_243812


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l2438_243879

theorem fraction_sum_integer (n : ℕ+) (h : ∃ (k : ℤ), (1/4 : ℚ) + (1/5 : ℚ) + (1/10 : ℚ) + (1/(n : ℚ)) = k) : n = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l2438_243879


namespace NUMINAMATH_CALUDE_book_purchase_theorem_l2438_243800

theorem book_purchase_theorem (total_A total_B both only_B : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_theorem_l2438_243800


namespace NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l2438_243846

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l2438_243846


namespace NUMINAMATH_CALUDE_triangle_coverage_theorem_l2438_243889

-- Define the original equilateral triangle
structure EquilateralTriangle where
  area : ℝ
  isEquilateral : Bool

-- Define a point inside the triangle
structure Point where
  x : ℝ
  y : ℝ
  insideTriangle : Bool

-- Define the smaller equilateral triangles
structure SmallerTriangle where
  area : ℝ
  sidesParallel : Bool
  containsPoint : Point → Bool

-- Main theorem
theorem triangle_coverage_theorem 
  (original : EquilateralTriangle)
  (points : Finset Point)
  (h1 : original.area = 1)
  (h2 : original.isEquilateral = true)
  (h3 : points.card = 5)
  (h4 : ∀ p ∈ points, p.insideTriangle = true) :
  ∃ (t1 t2 t3 : SmallerTriangle),
    (t1.sidesParallel = true ∧ t2.sidesParallel = true ∧ t3.sidesParallel = true) ∧
    (t1.area + t2.area + t3.area ≤ 0.64) ∧
    (∀ p ∈ points, t1.containsPoint p = true ∨ t2.containsPoint p = true ∨ t3.containsPoint p = true) :=
by sorry

end NUMINAMATH_CALUDE_triangle_coverage_theorem_l2438_243889


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2438_243874

theorem min_value_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2438_243874


namespace NUMINAMATH_CALUDE_max_min_difference_is_16_l2438_243892

def f (x : ℝ) := |x - 1| + |x - 2| + |x - 3|

theorem max_min_difference_is_16 :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-4 : ℝ) 4 ∧ x_min ∈ Set.Icc (-4 : ℝ) 4 ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x ≤ f x_max) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x_min ≤ f x) ∧
  f x_max - f x_min = 16 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_is_16_l2438_243892


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l2438_243853

theorem rectangular_field_perimeter
  (area : ℝ) (width : ℝ) (h_area : area = 300) (h_width : width = 15) :
  2 * (area / width + width) = 70 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l2438_243853


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2438_243845

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2438_243845


namespace NUMINAMATH_CALUDE_greatest_possible_award_l2438_243827

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 400)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : 2 * (total_prize / 5) = 3 * (num_winners / 5) * min_award) : 
  ∃ (max_award : ℕ), max_award = 100 ∧ 
  (∀ (award : ℕ), award ≤ max_award ∧ 
    (∃ (distribution : List ℕ), 
      distribution.length = num_winners ∧
      (∀ x ∈ distribution, min_award ≤ x) ∧
      distribution.sum = total_prize ∧
      award ∈ distribution)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l2438_243827


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_196_l2438_243861

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_196_l2438_243861


namespace NUMINAMATH_CALUDE_kims_earrings_l2438_243823

/-- Proves that Kim brings 5 pairs of earrings on the third day to have enough gumballs for 42 days -/
theorem kims_earrings (gumballs_per_pair : ℕ) (day1_pairs : ℕ) (daily_consumption : ℕ) (total_days : ℕ) :
  gumballs_per_pair = 9 →
  day1_pairs = 3 →
  daily_consumption = 3 →
  total_days = 42 →
  let day2_pairs := 2 * day1_pairs
  let day3_pairs := day2_pairs - 1
  let total_gumballs := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)
  total_gumballs = daily_consumption * total_days →
  day3_pairs = 5 := by
sorry


end NUMINAMATH_CALUDE_kims_earrings_l2438_243823


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2438_243852

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_solution_factorial_equation :
  ∃! (k n : ℕ), factorial n + 3 * n + 8 = k^2 ∧ k = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2438_243852


namespace NUMINAMATH_CALUDE_problem_statement_l2438_243818

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2438_243818


namespace NUMINAMATH_CALUDE_bob_garden_area_l2438_243897

/-- Calculates the area of a garden given property dimensions and garden proportions. -/
def garden_area (property_width property_length : ℝ) (garden_width_ratio garden_length_ratio : ℝ) : ℝ :=
  (property_width * garden_width_ratio) * (property_length * garden_length_ratio)

/-- Theorem stating that Bob's garden area is 28125 square feet. -/
theorem bob_garden_area :
  garden_area 1000 2250 (1/8) (1/10) = 28125 := by
  sorry

end NUMINAMATH_CALUDE_bob_garden_area_l2438_243897


namespace NUMINAMATH_CALUDE_negation_of_implication_l2438_243865

theorem negation_of_implication (x y : ℝ) : 
  ¬(x = 0 ∧ y = 0 → x * y = 0) ↔ (¬(x = 0 ∧ y = 0) → x * y ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2438_243865


namespace NUMINAMATH_CALUDE_tshirts_sold_l2438_243867

def profit_per_tshirt : ℕ := 62
def total_tshirt_profit : ℕ := 11346

theorem tshirts_sold : ℕ := by
  have h : profit_per_tshirt * 183 = total_tshirt_profit := by sorry
  exact 183

#check tshirts_sold

end NUMINAMATH_CALUDE_tshirts_sold_l2438_243867


namespace NUMINAMATH_CALUDE_octal_536_to_base7_l2438_243898

def octal_to_decimal (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  [1, 0, 1, 0]

theorem octal_536_to_base7 :
  decimal_to_base7 (octal_to_decimal 536) = [1, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_octal_536_to_base7_l2438_243898


namespace NUMINAMATH_CALUDE_center_is_five_l2438_243899

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if the grid satisfies the consecutive number condition -/
def isConsecutive (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, sharesEdge p1 p2 →
    (g p1.1 p1.2).val + 1 = (g p2.1 p2.2).val ∨
    (g p2.1 p2.2).val + 1 = (g p1.1 p1.2).val

/-- The sum of corner numbers in the grid -/
def cornerSum (g : Grid) : Nat :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- All numbers from 1 to 9 are used in the grid -/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n

theorem center_is_five (g : Grid)
  (h1 : isConsecutive g)
  (h2 : cornerSum g = 20)
  (h3 : usesAllNumbers g) :
  g 1 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_center_is_five_l2438_243899


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l2438_243876

structure Space where
  Plane : Type
  Line : Type
  parallel : Plane → Plane → Prop
  perpendicular : Line → Plane → Prop
  subset : Line → Plane → Prop

theorem line_perpendicular_to_parallel_planes 
  (S : Space) (α β : S.Plane) (m : S.Line) : 
  S.perpendicular m α → S.parallel α β → S.perpendicular m β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l2438_243876


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2438_243803

/-- Given a polynomial function f(x) = ax^7 - bx^5 + cx^3 + 2, 
    prove that f(5) + f(-5) = 4 -/
theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2438_243803


namespace NUMINAMATH_CALUDE_vector_magnitude_direction_comparison_l2438_243888

theorem vector_magnitude_direction_comparison
  (a b : ℝ × ℝ)
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : ∃ (k : ℝ), k > 0 ∧ a = k • b)
  (h4 : ‖a‖ > ‖b‖) :
  ¬ (∀ (x y : ℝ × ℝ), (∃ (k : ℝ), k > 0 ∧ x = k • y) → x > y) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_direction_comparison_l2438_243888


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2438_243847

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ)^2 - ((t.base : ℚ) / 2)^2).sqrt) / 2

/-- Theorem: Minimum perimeter of two noncongruent isosceles triangles with same area and base ratio 9:8 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t1.base = 8 * t2.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      9 * s1.base = 8 * s2.base →
      perimeter t1 ≤ perimeter s1 :=
by sorry

#eval perimeter { leg := 90, base := 144 } -- Expected output: 324

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2438_243847
