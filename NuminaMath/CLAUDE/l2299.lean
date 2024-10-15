import Mathlib

namespace NUMINAMATH_CALUDE_constant_zero_unique_solution_l2299_229991

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the derivative of a function
noncomputable def derivative (f : RealFunction) : RealFunction :=
  λ x => deriv f x

-- State the theorem
theorem constant_zero_unique_solution :
  ∃! f : RealFunction, ∀ x : ℝ, f x = derivative f x ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_zero_unique_solution_l2299_229991


namespace NUMINAMATH_CALUDE_mean_is_seven_l2299_229939

def pull_up_data : List (ℕ × ℕ) := [(9, 2), (8, 3), (6, 3), (5, 2)]

def total_students : ℕ := 10

theorem mean_is_seven :
  let sum := (pull_up_data.map (λ p => p.1 * p.2)).sum
  sum / total_students = 7 := by sorry

end NUMINAMATH_CALUDE_mean_is_seven_l2299_229939


namespace NUMINAMATH_CALUDE_rectangle_circumscribed_l2299_229964

/-- Two lines form a rectangle with the coordinate axes that can be circumscribed by a circle -/
theorem rectangle_circumscribed (k : ℝ) : 
  (∃ (x y : ℝ), x + 3*y - 7 = 0 ∧ k*x - y - 2 = 0) →
  (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 → (x + 3*y - 7 = 0 ∨ k*x - y - 2 = 0 ∨ x = 0 ∨ y = 0)) →
  (k = 3) := by
sorry

end NUMINAMATH_CALUDE_rectangle_circumscribed_l2299_229964


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2299_229961

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The common ratio of a geometric sequence is the constant factor between successive terms. -/
def CommonRatio (a : ℕ → ℚ) : ℚ :=
  a 1 / a 0

theorem geometric_sequence_ratio :
  ∀ a : ℕ → ℚ,
  IsGeometricSequence a →
  a 0 = 25 →
  a 1 = -50 →
  a 2 = 100 →
  a 3 = -200 →
  CommonRatio a = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2299_229961


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l2299_229958

theorem circumscribed_circle_equation (A B C : ℝ × ℝ) :
  A = (4, 1) → B = (6, -3) → C = (-3, 0) →
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ↔
      (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2)) ∧
    center = (1, -3) ∧ r^2 = 25 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l2299_229958


namespace NUMINAMATH_CALUDE_cos_negative_330_degrees_l2299_229943

theorem cos_negative_330_degrees : Real.cos (-(330 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_330_degrees_l2299_229943


namespace NUMINAMATH_CALUDE_problem_2011_l2299_229986

theorem problem_2011 : (2011^2 + 2011) / 2011 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_problem_2011_l2299_229986


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2299_229935

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2299_229935


namespace NUMINAMATH_CALUDE_john_weekly_production_l2299_229950

/-- The number of widgets John makes per week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Theorem stating that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_production_l2299_229950


namespace NUMINAMATH_CALUDE_lightsaber_to_other_toys_ratio_l2299_229922

-- Define the cost of other Star Wars toys
def other_toys_cost : ℕ := 1000

-- Define the total spent
def total_spent : ℕ := 3000

-- Define the cost of the lightsaber
def lightsaber_cost : ℕ := total_spent - other_toys_cost

-- Theorem statement
theorem lightsaber_to_other_toys_ratio :
  (lightsaber_cost : ℚ) / other_toys_cost = 2 := by sorry

end NUMINAMATH_CALUDE_lightsaber_to_other_toys_ratio_l2299_229922


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l2299_229931

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) 
  (h1 : 60 * (x + 1) + 120 * x + 180 * (x - 1) = 100 * (3 * x)) : 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l2299_229931


namespace NUMINAMATH_CALUDE_mr_a_loss_l2299_229982

/-- Represents the house transaction between Mr. A and Mr. B -/
def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (rent : ℝ) (gain_percent : ℝ) : ℝ :=
  let sale_price := initial_value * (1 - loss_percent)
  let repurchase_price := sale_price * (1 + gain_percent)
  repurchase_price - initial_value

/-- Theorem stating that Mr. A loses $144 in the transaction -/
theorem mr_a_loss :
  house_transaction 12000 0.12 1000 0.15 = 144 := by
  sorry

end NUMINAMATH_CALUDE_mr_a_loss_l2299_229982


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2299_229929

theorem perpendicular_lines_a_values (a : ℝ) : 
  ((3*a + 2) * (5*a - 2) + (1 - 4*a) * (a + 4) = 0) → (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2299_229929


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2299_229930

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2299_229930


namespace NUMINAMATH_CALUDE_folding_punching_theorem_l2299_229984

/-- Represents a rectangular piece of paper --/
structure Paper where
  width : ℝ
  height : ℝ
  (width_pos : width > 0)
  (height_pos : height > 0)

/-- Represents a fold operation on the paper --/
inductive Fold
  | BottomToTop
  | RightToHalfLeft
  | DiagonalBottomLeftToTopRight

/-- Represents a hole punched in the paper --/
structure Hole where
  x : ℝ
  y : ℝ

/-- Applies a sequence of folds to a paper --/
def applyFolds (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole in the folded paper --/
def punchHole (p : Paper) : Hole :=
  sorry

/-- Unfolds the paper and calculates the resulting hole pattern --/
def unfoldAndGetHoles (p : Paper) (folds : List Fold) (h : Hole) : List Hole :=
  sorry

/-- Checks if a list of holes is symmetric around the center and along two diagonals --/
def isSymmetricPattern (holes : List Hole) : Prop :=
  sorry

/-- The main theorem stating that the folding and punching process results in 8 symmetric holes --/
theorem folding_punching_theorem (p : Paper) :
  let folds := [Fold.BottomToTop, Fold.RightToHalfLeft, Fold.DiagonalBottomLeftToTopRight]
  let foldedPaper := applyFolds p folds
  let hole := punchHole foldedPaper
  let holePattern := unfoldAndGetHoles p folds hole
  (holePattern.length = 8) ∧ isSymmetricPattern holePattern :=
by
  sorry

end NUMINAMATH_CALUDE_folding_punching_theorem_l2299_229984


namespace NUMINAMATH_CALUDE_arrangements_five_not_adjacent_l2299_229917

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n distinct objects in a line, 
    where two specific objects are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ :=
  factorial n - 2 * factorial (n - 1)

theorem arrangements_five_not_adjacent :
  arrangements_not_adjacent 5 = 72 := by
  sorry

#eval arrangements_not_adjacent 5

end NUMINAMATH_CALUDE_arrangements_five_not_adjacent_l2299_229917


namespace NUMINAMATH_CALUDE_log_problem_l2299_229903

theorem log_problem (x : ℝ) (h : Real.log x / Real.log 7 - Real.log 3 / Real.log 7 = 2) :
  Real.log x / Real.log 13 = Real.log 52 / Real.log 13 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2299_229903


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l2299_229960

theorem sum_of_powers_equals_zero :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l2299_229960


namespace NUMINAMATH_CALUDE_line_slope_proof_l2299_229940

theorem line_slope_proof (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (α : ℝ), α ∈ Set.Icc 0 π ∧ Real.tan α = -Real.sqrt 3 / 3 ∧ α = 5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proof_l2299_229940


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2299_229989

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (h_distinct_xy : x ≠ y) (h_distinct_yz : y ≠ z) (h_distinct_zx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = 5 * k * (x - y) * (y - z) * (z - x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2299_229989


namespace NUMINAMATH_CALUDE_pascal_triangle_sum_rows_7_8_l2299_229969

theorem pascal_triangle_sum_rows_7_8 : ℕ := by
  -- Define the sum of numbers in row n of Pascal's Triangle
  let sum_row (n : ℕ) := 2^n
  
  -- Sum of Row 7
  let sum_row_7 := sum_row 7
  
  -- Sum of Row 8
  let sum_row_8 := sum_row 8
  
  -- Total sum of Rows 7 and 8
  let total_sum := sum_row_7 + sum_row_8
  
  -- Prove that the total sum equals 384
  have h : total_sum = 384 := by sorry
  
  exact 384


end NUMINAMATH_CALUDE_pascal_triangle_sum_rows_7_8_l2299_229969


namespace NUMINAMATH_CALUDE_system_solution_l2299_229963

theorem system_solution (x y z : ℝ) : 
  x + y + z = 9 ∧ 
  1/x + 1/y + 1/z = 1 ∧ 
  x*y + x*z + y*z = 27 → 
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2299_229963


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2299_229979

theorem product_sum_theorem (W F : ℕ) (c : ℕ) : 
  W > 20 → F > 20 → W * F = 770 → W + F = c → c = 57 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2299_229979


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_16_l2299_229934

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_16_l2299_229934


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2299_229972

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2299_229972


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l2299_229928

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l2299_229928


namespace NUMINAMATH_CALUDE_cube_root_eq_four_l2299_229936

theorem cube_root_eq_four (y : ℝ) :
  (y * (y^5)^(1/2))^(1/3) = 4 → y = 4^(6/7) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_eq_four_l2299_229936


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2299_229919

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = m * x + 4

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_eq x y ∧ line_eq m x y) →
  m^2 ≥ 0.48 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2299_229919


namespace NUMINAMATH_CALUDE_direct_product_is_group_l2299_229913

/-- Given two groups G and H, their direct product is also a group. -/
theorem direct_product_is_group {G H : Type*} [Group G] [Group H] :
  Group (G × H) :=
by sorry

end NUMINAMATH_CALUDE_direct_product_is_group_l2299_229913


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l2299_229965

theorem absolute_value_and_exponents : 
  |(-3 : ℝ)| + (Real.pi + 1)^(0 : ℝ) - (1/3 : ℝ)^(-1 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l2299_229965


namespace NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l2299_229949

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual yield of the bank deposit
def nominal_annual_yield : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit :
  ((1 + nominal_annual_yield)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l2299_229949


namespace NUMINAMATH_CALUDE_star_equal_is_diagonal_l2299_229947

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a * b * (a - b)

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The line y = x in ℝ² -/
def diagonal_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

theorem star_equal_is_diagonal :
  star_equal_set = diagonal_line := by sorry

end NUMINAMATH_CALUDE_star_equal_is_diagonal_l2299_229947


namespace NUMINAMATH_CALUDE_concentric_circles_area_l2299_229967

theorem concentric_circles_area (R r : ℝ) (h1 : R > r) (h2 : r > 0) 
  (h3 : R^2 - r^2 = 2500) : 
  π * (R^2 - r^2) = 2500 * π := by
  sorry

#check concentric_circles_area

end NUMINAMATH_CALUDE_concentric_circles_area_l2299_229967


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l2299_229941

/-- The number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- The theorem stating that after 7 days, there will be 2187 amoebas -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l2299_229941


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2299_229966

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2299_229966


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l2299_229906

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ x ∈ solutions, x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l2299_229906


namespace NUMINAMATH_CALUDE_sticks_at_20th_stage_l2299_229920

/-- The number of sticks in the nth stage of the pattern -/
def sticks : ℕ → ℕ
| 0 => 5  -- Initial stage (indexed as 0)
| n + 1 => if n < 10 then sticks n + 3 else sticks n + 4

/-- The theorem stating that the 20th stage (indexed as 19) has 68 sticks -/
theorem sticks_at_20th_stage : sticks 19 = 68 := by
  sorry

end NUMINAMATH_CALUDE_sticks_at_20th_stage_l2299_229920


namespace NUMINAMATH_CALUDE_clock_gains_seven_minutes_per_hour_l2299_229921

/-- A clock that gains time -/
structure GainingClock where
  start_time : Nat  -- Start time in hours (24-hour format)
  end_time : Nat    -- End time in hours (24-hour format)
  total_gain : Nat  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : Rat :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock starting at 9 AM, ending at 6 PM, and gaining 63 minutes
    will gain 7 minutes per hour -/
theorem clock_gains_seven_minutes_per_hour 
  (clock : GainingClock) 
  (h1 : clock.start_time = 9)
  (h2 : clock.end_time = 18)
  (h3 : clock.total_gain = 63) :
  minutes_gained_per_hour clock = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_gains_seven_minutes_per_hour_l2299_229921


namespace NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l2299_229974

/-- The perimeter of a regular hexagon with side length 2 inches is 12 inches. -/
theorem hexagon_perimeter : ℝ → Prop :=
  fun (side_length : ℝ) =>
    side_length = 2 →
    6 * side_length = 12

/-- Proof of the theorem -/
theorem hexagon_perimeter_proof : hexagon_perimeter 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l2299_229974


namespace NUMINAMATH_CALUDE_beryllium_hydroxide_formation_l2299_229909

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

/-- The balanced chemical equation for the reaction of beryllium carbide with water -/
def berylliumCarbideReaction : ChemicalReaction :=
  { reactants := [
      { formula := "Be2C", moles := 1 },
      { formula := "H2O", moles := 4 }
    ],
    products := [
      { formula := "Be(OH)2", moles := 2 },
      { formula := "CH4", moles := 1 }
    ]
  }

/-- Given 1 mole of Be2C and 4 moles of H2O, 2 moles of Be(OH)2 are formed -/
theorem beryllium_hydroxide_formation :
  ∀ (reaction : ChemicalReaction),
    reaction = berylliumCarbideReaction →
    ∃ (product : ChemicalSpecies),
      product ∈ reaction.products ∧
      product.formula = "Be(OH)2" ∧
      product.moles = 2 :=
by sorry

end NUMINAMATH_CALUDE_beryllium_hydroxide_formation_l2299_229909


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2299_229955

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2299_229955


namespace NUMINAMATH_CALUDE_romanian_sequence_swaps_l2299_229937

/-- A Romanian sequence is a sequence of 3n letters where I, M, and O each occur exactly n times. -/
def RomanianSequence (n : ℕ) := Vector (Fin 3) (3 * n)

/-- The number of swaps required to transform one sequence into another. -/
def swapsRequired (n : ℕ) (X Y : RomanianSequence n) : ℕ := sorry

/-- There exists a Romanian sequence Y for any Romanian sequence X such that
    at least 3n^2/2 swaps are required to transform X into Y. -/
theorem romanian_sequence_swaps (n : ℕ) :
  ∀ X : RomanianSequence n, ∃ Y : RomanianSequence n,
    swapsRequired n X Y ≥ (3 * n^2) / 2 := by sorry

end NUMINAMATH_CALUDE_romanian_sequence_swaps_l2299_229937


namespace NUMINAMATH_CALUDE_average_age_combined_l2299_229946

theorem average_age_combined (num_students : ℕ) (avg_age_students : ℝ)
                              (num_teachers : ℕ) (avg_age_teachers : ℝ)
                              (num_parents : ℕ) (avg_age_parents : ℝ) :
  num_students = 40 →
  avg_age_students = 10 →
  num_teachers = 4 →
  avg_age_teachers = 40 →
  num_parents = 60 →
  avg_age_parents = 34 →
  (num_students * avg_age_students + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
  (num_students + num_teachers + num_parents : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2299_229946


namespace NUMINAMATH_CALUDE_billy_has_24_balloons_l2299_229942

/-- The number of water balloons Billy is left with after the water balloon fight -/
def billys_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (num_people : ℕ) 
  (extra_milly : ℕ) (extra_tamara : ℕ) (extra_floretta : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / num_people

/-- Theorem stating that Billy is left with 24 water balloons -/
theorem billy_has_24_balloons : 
  billys_balloons 12 8 4 11 9 4 = 24 := by
  sorry

#eval billys_balloons 12 8 4 11 9 4

end NUMINAMATH_CALUDE_billy_has_24_balloons_l2299_229942


namespace NUMINAMATH_CALUDE_product_difference_square_l2299_229938

theorem product_difference_square (n : ℤ) : (n - 1) * (n + 1) - n^2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_product_difference_square_l2299_229938


namespace NUMINAMATH_CALUDE_other_denomination_is_50_l2299_229932

/-- Proves that the denomination of the other currency notes is 50 given the problem conditions --/
theorem other_denomination_is_50 
  (total_notes : ℕ) 
  (total_amount : ℕ) 
  (amount_other_denom : ℕ) 
  (h_total_notes : total_notes = 85)
  (h_total_amount : total_amount = 5000)
  (h_amount_other_denom : amount_other_denom = 3500) :
  ∃ (x y D : ℕ), 
    x + y = total_notes ∧ 
    100 * x + D * y = total_amount ∧
    D * y = amount_other_denom ∧
    D = 50 := by
  sorry

#check other_denomination_is_50

end NUMINAMATH_CALUDE_other_denomination_is_50_l2299_229932


namespace NUMINAMATH_CALUDE_kats_strength_training_time_l2299_229970

/-- Given Kat's training schedule, prove that she spends 1 hour on strength training each session -/
theorem kats_strength_training_time (
  strength_sessions : ℕ) 
  (boxing_sessions : ℕ) 
  (boxing_hours_per_session : ℚ)
  (total_weekly_hours : ℕ) 
  (h1 : strength_sessions = 3)
  (h2 : boxing_sessions = 4)
  (h3 : boxing_hours_per_session = 3/2)
  (h4 : total_weekly_hours = 9) :
  (total_weekly_hours - boxing_sessions * boxing_hours_per_session) / strength_sessions = 1 := by
  sorry

end NUMINAMATH_CALUDE_kats_strength_training_time_l2299_229970


namespace NUMINAMATH_CALUDE_candy_distribution_l2299_229914

/-- Given a total number of candy pieces and the number of pieces per student,
    calculate the number of students. -/
def number_of_students (total_candy : ℕ) (candy_per_student : ℕ) : ℕ :=
  total_candy / candy_per_student

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) 
    (h1 : total_candy = 344) 
    (h2 : candy_per_student = 8) :
    number_of_students total_candy candy_per_student = 43 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2299_229914


namespace NUMINAMATH_CALUDE_first_stop_passengers_l2299_229971

/-- The number of passengers who got on at the first stop of a bus route -/
def passengers_first_stop : ℕ :=
  sorry

/-- The net change in passengers at the second stop -/
def net_change_second_stop : ℤ := 2

/-- The net change in passengers at the third stop -/
def net_change_third_stop : ℤ := 2

/-- The total number of passengers after the third stop -/
def total_passengers : ℕ := 11

theorem first_stop_passengers :
  passengers_first_stop = 7 :=
sorry

end NUMINAMATH_CALUDE_first_stop_passengers_l2299_229971


namespace NUMINAMATH_CALUDE_sunflower_contest_total_l2299_229905

/-- Represents the total number of seeds eaten in a sunflower eating contest -/
def total_seeds_eaten (player1 player2 player3 : ℕ) : ℕ :=
  player1 + player2 + player3

/-- Theorem stating the total number of seeds eaten in the contest -/
theorem sunflower_contest_total :
  let player1 := 78
  let player2 := 53
  let player3 := player2 + 30
  total_seeds_eaten player1 player2 player3 = 214 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_contest_total_l2299_229905


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l2299_229983

theorem fundraiser_group_composition (p : ℕ) : 
  p > 0 ∧ 
  (p / 2 : ℚ) = (p / 2 : ℕ) ∧ 
  ((p / 2 - 2 : ℚ) / p = 2 / 5) → 
  p / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_group_composition_l2299_229983


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2299_229908

theorem factorial_sum_equality : 7 * Nat.factorial 6 + 6 * Nat.factorial 5 + 2 * Nat.factorial 5 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2299_229908


namespace NUMINAMATH_CALUDE_fair_coin_prob_diff_l2299_229925

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The probability of getting exactly 3 heads in 4 flips of a fair coin -/
def prob_3_heads : ℚ := prob_k_heads 4 3

/-- The probability of getting 4 heads in 4 flips of a fair coin -/
def prob_4_heads : ℚ := prob_k_heads 4 4

/-- The positive difference between the probability of exactly 3 heads
    and the probability of 4 heads in 4 flips of a fair coin -/
theorem fair_coin_prob_diff : prob_3_heads - prob_4_heads = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_prob_diff_l2299_229925


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l2299_229927

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- State the theorem
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sumOfDigits (N + M) = sumOfDigits N) ∧ 
    (∀ k : ℕ, k > 1 → sumOfDigits (N + k * M) ≠ sumOfDigits N) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l2299_229927


namespace NUMINAMATH_CALUDE_perpendicular_trapezoid_midline_l2299_229992

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularTrapezoid where
  /-- One diagonal of the trapezoid -/
  diagonal1 : ℝ
  /-- The angle between the other diagonal and the base -/
  angle : ℝ
  /-- The diagonals are perpendicular -/
  perpendicular : True
  /-- One diagonal is 6 units long -/
  diagonal1_length : diagonal1 = 6
  /-- The other diagonal forms a 30° angle with the base -/
  angle_is_30 : angle = 30 * π / 180

/-- The midline of a trapezoid with perpendicular diagonals -/
def midline (t : PerpendicularTrapezoid) : ℝ := sorry

/-- Theorem: The midline of a trapezoid with perpendicular diagonals,
    where one diagonal is 6 units long and the other forms a 30° angle with the base,
    is 6 units long -/
theorem perpendicular_trapezoid_midline (t : PerpendicularTrapezoid) :
  midline t = 6 := by sorry

end NUMINAMATH_CALUDE_perpendicular_trapezoid_midline_l2299_229992


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l2299_229923

theorem movie_ticket_cost (ticket_price : ℕ) (num_students : ℕ) (budget : ℕ) : 
  ticket_price = 29 → num_students = 498 → budget = 1500 → 
  ticket_price * num_students > budget :=
by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l2299_229923


namespace NUMINAMATH_CALUDE_yogurt_refund_calculation_l2299_229907

theorem yogurt_refund_calculation (total_packs : ℕ) (expired_percentage : ℚ) (price_per_pack : ℚ) : 
  total_packs = 80 →
  expired_percentage = 40 / 100 →
  price_per_pack = 12 →
  (total_packs : ℚ) * expired_percentage * price_per_pack = 384 := by
sorry

end NUMINAMATH_CALUDE_yogurt_refund_calculation_l2299_229907


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2299_229995

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x₁ y₁ : ℝ), O₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + r)^2) ∧
    (∀ (x₂ y₂ : ℝ), O₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (9 - r)^2)) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2299_229995


namespace NUMINAMATH_CALUDE_coach_votes_l2299_229993

theorem coach_votes (num_coaches : ℕ) (num_voters : ℕ) (votes_per_voter : ℕ) 
  (h1 : num_coaches = 36)
  (h2 : num_voters = 60)
  (h3 : votes_per_voter = 3)
  (h4 : num_voters * votes_per_voter % num_coaches = 0) :
  (num_voters * votes_per_voter) / num_coaches = 5 := by
sorry

end NUMINAMATH_CALUDE_coach_votes_l2299_229993


namespace NUMINAMATH_CALUDE_ab_gt_b_squared_l2299_229924

theorem ab_gt_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_gt_b_squared_l2299_229924


namespace NUMINAMATH_CALUDE_amanda_grass_seed_bags_l2299_229951

/-- The number of bags of grass seed needed for a specific lot -/
def grassSeedBags (lotLength lotWidth concreteLength concreteWidth bagCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteLength * concreteWidth
  let grassArea := totalArea - concreteArea
  (grassArea + bagCoverage - 1) / bagCoverage

theorem amanda_grass_seed_bags :
  grassSeedBags 120 60 40 40 56 = 100 := by
  sorry

end NUMINAMATH_CALUDE_amanda_grass_seed_bags_l2299_229951


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_200_l2299_229959

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  angle_BAD : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.CD + t.AD + t.BC

/-- Theorem stating that the perimeter of the given trapezoid is 200 units -/
theorem trapezoid_perimeter_is_200 (t : Trapezoid) 
  (h1 : t.AB = 40)
  (h2 : t.CD = 35)
  (h3 : t.AD = 70)
  (h4 : t.BC = 55)
  (h5 : t.angle_BAD = 30 * π / 180)  -- Convert 30° to radians
  : perimeter t = 200 := by
  sorry

#check trapezoid_perimeter_is_200

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_200_l2299_229959


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l2299_229980

/-- The area of a circle circumscribing an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area of the circle (using the circumradius formula)
  A = 48 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l2299_229980


namespace NUMINAMATH_CALUDE_shirt_trouser_combinations_l2299_229997

theorem shirt_trouser_combinations (shirt_styles : ℕ) (trouser_colors : ℕ) 
  (h1 : shirt_styles = 4) (h2 : trouser_colors = 3) : 
  shirt_styles * trouser_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_trouser_combinations_l2299_229997


namespace NUMINAMATH_CALUDE_xiao_jun_travel_box_probability_l2299_229987

-- Define the number of digits in the password
def password_length : ℕ := 6

-- Define the number of possible digits (0-9)
def possible_digits : ℕ := 10

-- Define the probability of guessing the correct last digit
def probability_of_success : ℚ := 1 / possible_digits

-- Theorem statement
theorem xiao_jun_travel_box_probability :
  probability_of_success = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_xiao_jun_travel_box_probability_l2299_229987


namespace NUMINAMATH_CALUDE_restaurant_group_cost_l2299_229985

/-- Represents the cost structure and group composition at a restaurant -/
structure RestaurantGroup where
  adult_meal_cost : ℚ
  adult_drink_cost : ℚ
  adult_dessert_cost : ℚ
  kid_meal_cost : ℚ
  kid_drink_cost : ℚ
  kid_dessert_cost : ℚ
  total_people : ℕ
  num_kids : ℕ

/-- Calculates the total cost for a restaurant group -/
def total_cost (g : RestaurantGroup) : ℚ :=
  let num_adults := g.total_people - g.num_kids
  let adult_cost := num_adults * (g.adult_meal_cost + g.adult_drink_cost + g.adult_dessert_cost)
  let kid_cost := g.num_kids * (g.kid_meal_cost + g.kid_drink_cost + g.kid_dessert_cost)
  adult_cost + kid_cost

/-- Theorem stating that the total cost for the given group is $87.50 -/
theorem restaurant_group_cost :
  let g : RestaurantGroup := {
    adult_meal_cost := 7
    adult_drink_cost := 4
    adult_dessert_cost := 3
    kid_meal_cost := 0
    kid_drink_cost := 2
    kid_dessert_cost := 3/2
    total_people := 13
    num_kids := 9
  }
  total_cost g = 175/2 := by sorry

end NUMINAMATH_CALUDE_restaurant_group_cost_l2299_229985


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l2299_229976

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * second
  (x + second + third) / 3 = 78 →
  x = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l2299_229976


namespace NUMINAMATH_CALUDE_length_of_A_l2299_229933

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 6)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (1 - t) • p + t • q = r

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ,
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect A A' C ∧
    intersect B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (12 / 7) * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_of_A_l2299_229933


namespace NUMINAMATH_CALUDE_expansion_simplification_l2299_229900

theorem expansion_simplification (y : ℝ) : (2*y - 3)*(2*y + 3) - (4*y - 1)*(y + 5) = -19*y - 4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l2299_229900


namespace NUMINAMATH_CALUDE_december_sales_fraction_l2299_229999

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let other_months_total := 11 * average_sales
  let december_sales := 6 * average_sales
  let annual_sales := other_months_total + december_sales
  december_sales / annual_sales = 6 / 17 := by
sorry

end NUMINAMATH_CALUDE_december_sales_fraction_l2299_229999


namespace NUMINAMATH_CALUDE_park_area_l2299_229994

/-- Proves that a rectangular park with given conditions has an area of 102400 square meters -/
theorem park_area (length breadth : ℝ) (speed : ℝ) (time : ℝ) : 
  length / breadth = 4 →
  speed = 12 →
  time = 8 / 60 →
  2 * (length + breadth) = speed * time * 1000 →
  length * breadth = 102400 :=
by sorry

end NUMINAMATH_CALUDE_park_area_l2299_229994


namespace NUMINAMATH_CALUDE_car_trip_duration_l2299_229944

theorem car_trip_duration (initial_speed initial_time remaining_speed average_speed : ℝ) 
  (h1 : initial_speed = 70)
  (h2 : initial_time = 4)
  (h3 : remaining_speed = 60)
  (h4 : average_speed = 65) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = average_speed ∧ 
    total_time = 8 := by
sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2299_229944


namespace NUMINAMATH_CALUDE_expected_throws_in_leap_year_l2299_229977

/-- The expected number of throws for a single day -/
def expected_throws_per_day : ℚ := 8/7

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The expected number of throws in a leap year -/
def expected_throws_leap_year : ℚ := expected_throws_per_day * leap_year_days

theorem expected_throws_in_leap_year :
  expected_throws_leap_year = 2928/7 := by sorry

end NUMINAMATH_CALUDE_expected_throws_in_leap_year_l2299_229977


namespace NUMINAMATH_CALUDE_equation_root_implies_m_equals_three_l2299_229973

theorem equation_root_implies_m_equals_three (x m : ℝ) :
  (x ≠ 3) →
  (x / (x - 3) = 2 - m / (3 - x)) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_equals_three_l2299_229973


namespace NUMINAMATH_CALUDE_intercepts_correct_l2299_229901

/-- The line equation is 5x - 2y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Proof that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by sorry

end NUMINAMATH_CALUDE_intercepts_correct_l2299_229901


namespace NUMINAMATH_CALUDE_chipped_marbles_bag_l2299_229904

def marbleBags : List Nat := [15, 20, 22, 31, 33, 37, 40]

def isValidDistribution (jane : List Nat) (george : List Nat) : Prop :=
  jane.length = 4 ∧ 
  george.length = 2 ∧ 
  (jane.sum : ℚ) = 1.5 * george.sum ∧ 
  (jane.sum + george.sum) % 5 = 0

theorem chipped_marbles_bag (h : ∃ (jane george : List Nat),
  (∀ x ∈ jane ++ george, x ∈ marbleBags) ∧
  isValidDistribution jane george) :
  33 ∈ marbleBags \ (jane ++ george) :=
sorry

end NUMINAMATH_CALUDE_chipped_marbles_bag_l2299_229904


namespace NUMINAMATH_CALUDE_farm_animals_l2299_229911

theorem farm_animals (total_legs : ℕ) (chicken_count : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) :
  total_legs = 44 →
  chicken_count = 4 →
  chicken_legs = 2 →
  buffalo_legs = 4 →
  ∃ (buffalo_count : ℕ),
    total_legs = chicken_count * chicken_legs + buffalo_count * buffalo_legs ∧
    chicken_count + buffalo_count = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2299_229911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_line_point_l2299_229912

/-- If k, -1, b are three numbers in arithmetic sequence, 
    then the line y = kx + b passes through the point (1, -2). -/
theorem arithmetic_sequence_line_point (k b : ℝ) : 
  (∃ d : ℝ, k = -1 - d ∧ b = -1 + d) → 
  k * 1 + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_line_point_l2299_229912


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2299_229998

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 8 = 6
  prop2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2299_229998


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2299_229916

theorem at_least_one_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) :
  a > 1 ∨ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2299_229916


namespace NUMINAMATH_CALUDE_central_angle_from_arc_length_l2299_229956

/-- Given a circle with radius 12 mm and an arc length of 144 mm, 
    the central angle in radians is equal to 12. -/
theorem central_angle_from_arc_length (R L θ : ℝ) : 
  R = 12 → L = 144 → L = R * θ → θ = 12 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_from_arc_length_l2299_229956


namespace NUMINAMATH_CALUDE_price_decrease_proof_l2299_229945

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 1300

/-- The percentage of the original price after the decrease -/
def price_decrease_percentage : ℝ := 24

/-- The price of the article after the decrease -/
def decreased_price : ℝ := 988

theorem price_decrease_proof : 
  (1 - price_decrease_percentage / 100) * original_price = decreased_price := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_proof_l2299_229945


namespace NUMINAMATH_CALUDE_jules_blocks_to_walk_l2299_229918

-- Define the given constants
def vacation_cost : ℚ := 1000
def family_members : ℕ := 5
def start_fee : ℚ := 2
def per_block_fee : ℚ := 1.25
def num_dogs : ℕ := 20

-- Define Jules' contribution
def jules_contribution : ℚ := vacation_cost / family_members

-- Define the function to calculate earnings based on number of blocks
def earnings (blocks : ℕ) : ℚ := num_dogs * (start_fee + per_block_fee * blocks)

-- Theorem statement
theorem jules_blocks_to_walk :
  ∃ (blocks : ℕ), earnings blocks ≥ jules_contribution ∧
    ∀ (b : ℕ), b < blocks → earnings b < jules_contribution :=
by sorry

end NUMINAMATH_CALUDE_jules_blocks_to_walk_l2299_229918


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l2299_229948

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h_total : total_votes = 7000)
  (h_winner : winner_percentage = 55 / 100)
  (h_loser : loser_votes = 2520) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l2299_229948


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l2299_229926

/-- The molecular weight of AlCl3 in g/mol -/
def molecular_weight_AlCl3 : ℝ := 132

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 528

theorem molecular_weight_calculation :
  molecular_weight_AlCl3 * given_moles = total_weight :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l2299_229926


namespace NUMINAMATH_CALUDE_jakes_earnings_theorem_l2299_229910

/-- Calculates Jake's weekly earnings based on Jacob's hourly rates and Jake's work schedule. -/
def jakes_weekly_earnings (jacobs_weekday_rate : ℕ) (jacobs_weekend_rate : ℕ) 
  (jakes_weekday_hours : ℕ) (jakes_weekend_hours : ℕ) : ℕ :=
  let jakes_weekday_rate := 3 * jacobs_weekday_rate
  let jakes_weekend_rate := 3 * jacobs_weekend_rate
  let weekday_earnings := jakes_weekday_rate * jakes_weekday_hours * 5
  let weekend_earnings := jakes_weekend_rate * jakes_weekend_hours * 2
  weekday_earnings + weekend_earnings

/-- Theorem stating that Jake's weekly earnings are $960. -/
theorem jakes_earnings_theorem : 
  jakes_weekly_earnings 6 8 8 5 = 960 := by
  sorry

end NUMINAMATH_CALUDE_jakes_earnings_theorem_l2299_229910


namespace NUMINAMATH_CALUDE_karens_round_trip_distance_l2299_229975

/-- The total distance Karen covers for a round trip to the library -/
def total_distance (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  2 * (shelves * books_per_shelf)

/-- Proof that Karen's round trip distance is 3200 miles -/
theorem karens_round_trip_distance :
  total_distance 4 400 = 3200 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_karens_round_trip_distance_l2299_229975


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2299_229981

theorem minimum_value_theorem (m n t : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m + n = 1) 
  (ht : t > 0) 
  (hmin : ∀ s > 0, s / m + 1 / n ≥ t / m + 1 / n) 
  (heq : t / m + 1 / n = 9) : 
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2299_229981


namespace NUMINAMATH_CALUDE_function_property_l2299_229988

/-- Given two functions f and g defined on ℝ satisfying certain properties, 
    prove that g(1) + g(-1) = 1 -/
theorem function_property (f g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f 1 = f 2)
  (h3 : f 1 ≠ 0) : 
  g 1 + g (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2299_229988


namespace NUMINAMATH_CALUDE_invisible_dots_count_l2299_229978

/-- The sum of numbers on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The number of dice rolled -/
def num_dice : Nat := 4

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 6 + 6 + 4 + 4 + 3 + 2 + 1

/-- The total number of dots on all dice -/
def total_dots : Nat := num_dice * standard_die_sum

theorem invisible_dots_count : total_dots - visible_sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l2299_229978


namespace NUMINAMATH_CALUDE_zero_in_A_l2299_229953

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end NUMINAMATH_CALUDE_zero_in_A_l2299_229953


namespace NUMINAMATH_CALUDE_fraction_simplification_l2299_229915

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2299_229915


namespace NUMINAMATH_CALUDE_complex_number_location_l2299_229902

/-- Given a complex number z satisfying (1-i)z = (1+i)^2, 
    prove that z has a negative real part and a positive imaginary part. -/
theorem complex_number_location (z : ℂ) (h : (1 - I) * z = (1 + I)^2) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2299_229902


namespace NUMINAMATH_CALUDE_clothing_retailer_optimal_strategy_l2299_229990

/-- Represents the clothing retailer's purchase and sales data --/
structure ClothingRetailer where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_per_item : ℝ
  base_price : ℝ
  base_sales : ℝ
  price_decrease : ℝ
  sales_increase : ℝ
  daily_profit : ℝ

/-- Theorem stating the initial purchase quantity and price, and the optimal selling price --/
theorem clothing_retailer_optimal_strategy (r : ClothingRetailer)
  (h1 : r.first_purchase_cost = 48000)
  (h2 : r.second_purchase_cost = 100000)
  (h3 : r.cost_increase_per_item = 10)
  (h4 : r.base_price = 300)
  (h5 : r.base_sales = 80)
  (h6 : r.price_decrease = 10)
  (h7 : r.sales_increase = 20)
  (h8 : r.daily_profit = 3600) :
  ∃ (initial_quantity : ℝ) (initial_price : ℝ) (selling_price : ℝ),
    initial_quantity = 200 ∧
    initial_price = 240 ∧
    selling_price = 280 ∧
    (selling_price - (initial_price + r.cost_increase_per_item)) *
      (r.base_sales + (r.base_price - selling_price) / r.price_decrease * r.sales_increase) = r.daily_profit :=
by sorry

end NUMINAMATH_CALUDE_clothing_retailer_optimal_strategy_l2299_229990


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l2299_229957

theorem square_root_fraction_equality :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (49 + 36) = 17 * Real.sqrt 85 / 85 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l2299_229957


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l2299_229968

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) :
  a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l2299_229968


namespace NUMINAMATH_CALUDE_price_quantity_difference_l2299_229962

/-- Given a price increase and quantity reduction, proves the difference in cost -/
theorem price_quantity_difference (P Q : ℝ) (h_pos_P : P > 0) (h_pos_Q : Q > 0) : 
  (P * 1.1 * (Q * 0.8)) - (P * Q) = -0.12 * (P * Q) := by
  sorry

#check price_quantity_difference

end NUMINAMATH_CALUDE_price_quantity_difference_l2299_229962


namespace NUMINAMATH_CALUDE_nancy_spend_l2299_229996

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_spend : total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_nancy_spend_l2299_229996


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2299_229954

theorem triangle_side_difference (a b c : ℝ) : 
  b = 8 → c = 3 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → a + c > b → b + c > a →
  (∃ (a_min a_max : ℕ), 
    (∀ x : ℕ, (x : ℝ) = a → a_min ≤ x ∧ x ≤ a_max) ∧
    (∀ y : ℕ, y < a_min → (y : ℝ) ≠ a) ∧
    (∀ z : ℕ, z > a_max → (z : ℝ) ≠ a) ∧
    a_max - a_min = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2299_229954


namespace NUMINAMATH_CALUDE_anthony_jim_shoe_difference_l2299_229952

-- Define the number of shoe pairs for each person
def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

-- Theorem statement
theorem anthony_jim_shoe_difference :
  anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_jim_shoe_difference_l2299_229952
