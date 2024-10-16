import Mathlib

namespace NUMINAMATH_CALUDE_brownies_per_neighbor_l2594_259437

/-- Calculates the number of brownies each neighbor receives given the following conditions:
  * Melanie baked 15 batches of brownies
  * Each batch contains 30 brownies
  * She set aside 13/15 of the brownies in each batch for a bake sale
  * She placed 7/10 of the remaining brownies in a container
  * She donated 3/5 of what was left to a local charity
  * She wants to evenly distribute the rest among x neighbors
-/
theorem brownies_per_neighbor (x : ℕ) (x_pos : x > 0) : 
  let total_brownies := 15 * 30
  let bake_sale_brownies := (13 / 15 : ℚ) * total_brownies
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies.floor
  let container_brownies := (7 / 10 : ℚ) * remaining_after_bake_sale
  let remaining_after_container := remaining_after_bake_sale - container_brownies.floor
  let charity_brownies := (3 / 5 : ℚ) * remaining_after_container
  let final_remaining := remaining_after_container - charity_brownies.floor
  (final_remaining / x : ℚ) = 8 / x := by
    sorry

#check brownies_per_neighbor

end NUMINAMATH_CALUDE_brownies_per_neighbor_l2594_259437


namespace NUMINAMATH_CALUDE_f_is_even_m_upper_bound_a_comparisons_l2594_259475

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem m_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x > 0 → m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3 := by sorry

theorem a_comparisons (a : ℝ) (h : a > (Real.exp 1 + Real.exp (-1)) / 2) :
  (a < Real.exp 1 → Real.exp (a - 1) < a^(Real.exp 1 - 1)) ∧
  (a = Real.exp 1 → Real.exp (a - 1) = a^(Real.exp 1 - 1)) ∧
  (a > Real.exp 1 → Real.exp (a - 1) > a^(Real.exp 1 - 1)) := by sorry

end NUMINAMATH_CALUDE_f_is_even_m_upper_bound_a_comparisons_l2594_259475


namespace NUMINAMATH_CALUDE_classmate_pairs_l2594_259485

theorem classmate_pairs (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_classmate_pairs_l2594_259485


namespace NUMINAMATH_CALUDE_integer_solution_inequality_system_l2594_259462

theorem integer_solution_inequality_system : 
  ∃! x : ℤ, 2 * x ≤ 1 ∧ x + 2 > 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequality_system_l2594_259462


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l2594_259497

/-- Represents the number of hamburgers in various states --/
structure HamburgerCount where
  served : ℕ
  leftOver : ℕ

/-- Calculates the total number of hamburgers initially made --/
def totalHamburgers (h : HamburgerCount) : ℕ :=
  h.served + h.leftOver

/-- The theorem stating that for the given values, the total hamburgers is 9 --/
theorem restaurant_hamburgers :
  let h : HamburgerCount := { served := 3, leftOver := 6 }
  totalHamburgers h = 9 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l2594_259497


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2594_259419

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 36 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 36 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2594_259419


namespace NUMINAMATH_CALUDE_fraction_simplification_l2594_259402

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 12) / ((x^2 - 4*x + 4) / (x^2 - 6*x + 9)) = 
  ((x - 1) * (x - 3)^2) / ((x - 4) * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2594_259402


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2594_259429

/-- Given an arithmetic sequence {aₙ} with S₁ = 10 and S₂ = 20, prove that S₁₀ = 100 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  S 1 = 10 →                            -- given S₁ = 10
  S 2 = 20 →                            -- given S₂ = 20
  S 10 = 100 := by                      -- prove S₁₀ = 100
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2594_259429


namespace NUMINAMATH_CALUDE_system_solvability_l2594_259495

-- Define the system of equations
def system_of_equations (x y z a b c : ℝ) : Prop :=
  (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c

-- Define the solvability condition
def solvability_condition (a b c : ℝ) : Prop :=
  a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0

-- Theorem statement
theorem system_solvability (a b c : ℝ) :
  (∃ x y z : ℝ, system_of_equations x y z a b c) ↔ solvability_condition a b c :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l2594_259495


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l2594_259412

theorem cube_sum_of_roots (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 211.75 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l2594_259412


namespace NUMINAMATH_CALUDE_juelz_sisters_count_l2594_259436

theorem juelz_sisters_count (total_pieces : ℕ) (eaten_percentage : ℚ) (pieces_per_sister : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  pieces_per_sister = 32 →
  (total_pieces - (eaten_percentage * total_pieces).num) / pieces_per_sister = 3 :=
by sorry

end NUMINAMATH_CALUDE_juelz_sisters_count_l2594_259436


namespace NUMINAMATH_CALUDE_cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l2594_259473

theorem cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2 :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l2594_259473


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l2594_259407

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l2594_259407


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2594_259443

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2594_259443


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2594_259450

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (2 - i) * z = i^3 →
  z = 1/5 - (2/5) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2594_259450


namespace NUMINAMATH_CALUDE_price_of_pants_l2594_259416

/-- Given Iris's shopping trip to the mall, this theorem proves the price of each pair of pants. -/
theorem price_of_pants (jacket_price : ℕ) (shorts_price : ℕ) (total_spent : ℕ) 
  (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  jacket_count = 3 →
  shorts_count = 2 →
  pants_count = 4 →
  total_spent = 90 →
  ∃ (pants_price : ℕ), 
    pants_price * pants_count + jacket_price * jacket_count + shorts_price * shorts_count = total_spent ∧
    pants_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_price_of_pants_l2594_259416


namespace NUMINAMATH_CALUDE_eight_row_triangle_pieces_l2594_259486

/-- Calculates the sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def total_rods (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in an n-row triangle -/
def total_connectors (n : ℕ) : ℕ := triangular_number (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

/-- Theorem: The total number of pieces in an eight-row triangle is 153 -/
theorem eight_row_triangle_pieces : total_pieces 8 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eight_row_triangle_pieces_l2594_259486


namespace NUMINAMATH_CALUDE_always_pair_with_difference_multiple_of_seven_l2594_259481

theorem always_pair_with_difference_multiple_of_seven :
  ∀ (S : Finset ℕ),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 3000) →
  S.card = 8 →
  (∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b).mod 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_always_pair_with_difference_multiple_of_seven_l2594_259481


namespace NUMINAMATH_CALUDE_triangle_special_angle_relation_l2594_259434

/-- In a triangle ABC where α = 3β = 6γ, the equation bc² = (a+b)(a-b)² holds true. -/
theorem triangle_special_angle_relation (a b c : ℝ) (α β γ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < α ∧ 0 < β ∧ 0 < γ →  -- positive angles
  α + β + γ = π →         -- sum of angles in a triangle
  α = 3*β →               -- given condition
  α = 6*γ →               -- given condition
  b*c^2 = (a+b)*(a-b)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_angle_relation_l2594_259434


namespace NUMINAMATH_CALUDE_max_value_DEABC_l2594_259476

/-- Represents a single-digit number -/
def SingleDigit := {n : ℕ // n < 10}

/-- Converts a three-digit number represented by its digits to a natural number -/
def threeDigitToNat (a b c : SingleDigit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by its digits to a natural number -/
def twoDigitToNat (d e : SingleDigit) : ℕ := 10 * d.val + e.val

/-- Converts a five-digit number represented by its digits to a natural number -/
def fiveDigitToNat (d e a b c : SingleDigit) : ℕ := 
  10000 * d.val + 1000 * e.val + 100 * a.val + 10 * b.val + c.val

theorem max_value_DEABC 
  (A B C D E : SingleDigit)
  (h1 : twoDigitToNat D E = A.val + B.val + C.val)
  (h2 : threeDigitToNat A B C + threeDigitToNat B C A + threeDigitToNat C A B + twoDigitToNat D E = 2016) :
  (∀ A' B' C' D' E', 
    twoDigitToNat D' E' = A'.val + B'.val + C'.val →
    threeDigitToNat A' B' C' + threeDigitToNat B' C' A' + threeDigitToNat C' A' B' + twoDigitToNat D' E' = 2016 →
    fiveDigitToNat D' E' A' B' C' ≤ fiveDigitToNat D E A B C) →
  fiveDigitToNat D E A B C = 18783 :=
sorry

end NUMINAMATH_CALUDE_max_value_DEABC_l2594_259476


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2594_259411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m n)
  (h2 : perpendicularLP n α)
  (h3 : ¬ intersects m α) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2594_259411


namespace NUMINAMATH_CALUDE_bangle_packing_optimal_solution_l2594_259400

/-- Represents the number of dozens of bangles that can be packed in each box size -/
structure BoxCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the maximum number of boxes available for each size -/
structure MaxBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of boxes used for packing -/
structure UsedBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Check if the given number of used boxes is within the maximum allowed -/
def isValidBoxCount (used : UsedBoxes) (max : MaxBoxes) : Prop :=
  used.small ≤ max.small ∧ used.medium ≤ max.medium ∧ used.large ≤ max.large

/-- Calculate the total number of dozens packed given the box capacities and used boxes -/
def totalPacked (capacity : BoxCapacity) (used : UsedBoxes) : Nat :=
  used.small * capacity.small + used.medium * capacity.medium + used.large * capacity.large

/-- Check if the given solution packs all bangles and uses the minimum number of boxes -/
def isOptimalSolution (totalDozens : Nat) (capacity : BoxCapacity) (max : MaxBoxes) (solution : UsedBoxes) : Prop :=
  isValidBoxCount solution max ∧
  totalPacked capacity solution = totalDozens ∧
  ∀ (other : UsedBoxes), isValidBoxCount other max → totalPacked capacity other = totalDozens →
    solution.small + solution.medium + solution.large ≤ other.small + other.medium + other.large

theorem bangle_packing_optimal_solution :
  let totalDozens : Nat := 40
  let capacity : BoxCapacity := { small := 2, medium := 3, large := 4 }
  let max : MaxBoxes := { small := 6, medium := 5, large := 4 }
  let solution : UsedBoxes := { small := 5, medium := 5, large := 4 }
  isOptimalSolution totalDozens capacity max solution := by
  sorry

end NUMINAMATH_CALUDE_bangle_packing_optimal_solution_l2594_259400


namespace NUMINAMATH_CALUDE_exponent_division_l2594_259496

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2594_259496


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2594_259474

theorem cylinder_volume_equality (y : ℝ) : y > 0 →
  (π * (7 + 4)^2 * 5 = π * 7^2 * (5 + y)) → y = 360 / 49 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2594_259474


namespace NUMINAMATH_CALUDE_least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l2594_259439

theorem least_number_with_remainder (n : ℕ) : 
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) →
  n ≥ 256 :=
by sorry

theorem two_five_six_satisfies_conditions : 
  (256 % 7 = 4) ∧ (256 % 9 = 4) ∧ (256 % 12 = 4) ∧ (256 % 18 = 4) :=
by sorry

theorem least_number_is_256 : 
  ∀ n : ℕ, n < 256 → 
  ¬((n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l2594_259439


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l2594_259479

theorem rachel_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 7) 
  (h2 : reading_pages = 3) : 
  math_pages - reading_pages = 4 := by
sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l2594_259479


namespace NUMINAMATH_CALUDE_max_visible_sum_three_cubes_l2594_259478

/-- Represents a cube with six faces numbered 1, 3, 5, 7, 9, 11 -/
def Cube := Fin 6 → Nat

/-- The set of numbers on each cube -/
def cubeNumbers : Finset Nat := {1, 3, 5, 7, 9, 11}

/-- A function to calculate the sum of visible faces when stacking cubes -/
def visibleSum (c1 c2 c3 : Cube) : Nat := sorry

/-- The theorem stating the maximum visible sum when stacking three cubes -/
theorem max_visible_sum_three_cubes :
  ∃ (c1 c2 c3 : Cube),
    (∀ (i : Fin 6), c1 i ∈ cubeNumbers ∧ c2 i ∈ cubeNumbers ∧ c3 i ∈ cubeNumbers) ∧
    (∀ (c1' c2' c3' : Cube),
      (∀ (i : Fin 6), c1' i ∈ cubeNumbers ∧ c2' i ∈ cubeNumbers ∧ c3' i ∈ cubeNumbers) →
      visibleSum c1' c2' c3' ≤ visibleSum c1 c2 c3) ∧
    visibleSum c1 c2 c3 = 101 :=
  sorry

end NUMINAMATH_CALUDE_max_visible_sum_three_cubes_l2594_259478


namespace NUMINAMATH_CALUDE_mean_median_difference_zero_l2594_259425

/-- Represents the score distribution in a classroom --/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score90 + score95 = 1

/-- Calculates the mean score given a score distribution --/
def mean_score (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 95 * d.score95

/-- Calculates the median score given a score distribution --/
def median_score (d : ScoreDistribution) : ℝ := 85

/-- The main theorem stating that the difference between mean and median is zero --/
theorem mean_median_difference_zero (d : ScoreDistribution) :
  d.score60 = 0.05 →
  d.score75 = 0.20 →
  d.score85 = 0.30 →
  d.score90 = 0.25 →
  mean_score d - median_score d = 0 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_zero_l2594_259425


namespace NUMINAMATH_CALUDE_ellipse_constant_dot_product_l2594_259431

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of vectors MA and MB
def dot_product_MA_MB (m x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - m) * (x2 - m) + y1 * y2

-- Statement of the theorem
theorem ellipse_constant_dot_product :
  ∃ (m : ℝ), 
    (∀ (k x1 y1 x2 y2 : ℝ), k ≠ 0 →
      ellipse_C x1 y1 → ellipse_C x2 y2 →
      moving_line k x1 y1 → moving_line k x2 y2 →
      dot_product_MA_MB m x1 y1 x2 y2 = -7/16) ∧
    m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constant_dot_product_l2594_259431


namespace NUMINAMATH_CALUDE_expected_value_biased_coin_l2594_259404

/-- Expected value of winnings for a biased coin flip -/
theorem expected_value_biased_coin : 
  let prob_heads : ℚ := 2/5
  let prob_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 1
  prob_heads * win_heads - prob_tails * loss_tails = 7/5 := by
sorry

end NUMINAMATH_CALUDE_expected_value_biased_coin_l2594_259404


namespace NUMINAMATH_CALUDE_min_value_expression_l2594_259468

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 4 ∧
  ((5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 4 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2594_259468


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2594_259488

/-- Given a rectangle with sides a and b and perimeter p, 
    the area is maximized when the rectangle is a square. -/
theorem rectangle_max_area (a b p : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_perimeter : p = 2 * (a + b)) :
  ∃ (max_area : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2 * (x + y) = p → x * y ≤ max_area ∧ 
  (x * y = max_area ↔ x = y) :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2594_259488


namespace NUMINAMATH_CALUDE_min_four_digit_quotient_l2594_259401

/-- A type representing a base-ten digit (1-9) -/
def Digit := { n : Nat // 1 ≤ n ∧ n ≤ 9 }

/-- The function to be minimized -/
def f (a b c d : Digit) : ℚ :=
  (1000 * a.val + 100 * b.val + 10 * c.val + d.val) / (a.val + b.val + c.val + d.val)

/-- The theorem stating the minimum value of the function -/
theorem min_four_digit_quotient :
  ∀ (a b c d : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    f a b c d ≥ 80.56 :=
sorry

end NUMINAMATH_CALUDE_min_four_digit_quotient_l2594_259401


namespace NUMINAMATH_CALUDE_math_city_intersections_l2594_259489

/-- Represents a street in Math City -/
structure Street where
  curved : Bool

/-- Represents Math City -/
structure MathCity where
  streets : Finset Street
  no_parallel : True  -- Assumption that no streets are parallel
  curved_count : Nat
  curved_additional_intersections : Nat

/-- Calculates the maximum number of intersections in Math City -/
def max_intersections (city : MathCity) : Nat :=
  let basic_intersections := city.streets.card.choose 2
  let additional_intersections := city.curved_count * city.curved_additional_intersections
  basic_intersections + additional_intersections

/-- Theorem stating the maximum number of intersections in the given scenario -/
theorem math_city_intersections :
  ∀ (city : MathCity),
    city.streets.card = 10 →
    city.curved_count = 2 →
    city.curved_additional_intersections = 3 →
    max_intersections city = 51 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l2594_259489


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l2594_259422

/-- The number of ways to buy three items (headphones, a keyboard, and a mouse) in a store with given inventory. -/
theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) (keyboard_mouse_sets : ℕ) (headphone_mouse_sets : ℕ) : 
  headphones = 9 → 
  mice = 13 → 
  keyboards = 5 → 
  keyboard_mouse_sets = 4 → 
  headphone_mouse_sets = 5 → 
  headphones * keyboard_mouse_sets + 
  keyboards * headphone_mouse_sets + 
  headphones * mice * keyboards = 646 := by
sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l2594_259422


namespace NUMINAMATH_CALUDE_total_fingers_folded_l2594_259482

/-- The number of fingers folded by Yoojung -/
def yoojung_fingers : ℕ := 2

/-- The number of fingers folded by Yuna -/
def yuna_fingers : ℕ := 5

/-- The total number of fingers folded by both Yoojung and Yuna -/
def total_fingers : ℕ := yoojung_fingers + yuna_fingers

theorem total_fingers_folded : total_fingers = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_fingers_folded_l2594_259482


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l2594_259414

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l2594_259414


namespace NUMINAMATH_CALUDE_expected_hit_targets_bound_l2594_259433

theorem expected_hit_targets_bound (n : ℕ) (hn : n > 0) :
  let p := 1 - (1 - 1 / n)^n
  n * p ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_hit_targets_bound_l2594_259433


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_c_equals_three_l2594_259480

theorem infinite_solutions_imply_c_equals_three :
  (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_c_equals_three_l2594_259480


namespace NUMINAMATH_CALUDE_cosine_of_inclination_angle_l2594_259432

/-- A line in 2D space represented by its parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line -/
def inclinationAngle (l : ParametricLine) : ℝ := sorry

/-- The given line with parametric equations x = -2 + 3t and y = 3 - 4t -/
def givenLine : ParametricLine := {
  x := λ t => -2 + 3*t,
  y := λ t => 3 - 4*t
}

/-- Theorem stating that the cosine of the inclination angle of the given line is -3/5 -/
theorem cosine_of_inclination_angle :
  Real.cos (inclinationAngle givenLine) = -3/5 := by sorry

end NUMINAMATH_CALUDE_cosine_of_inclination_angle_l2594_259432


namespace NUMINAMATH_CALUDE_min_polyline_distance_l2594_259471

-- Define the polyline distance
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the circle
def on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop :=
  2*x + y - 2*Real.sqrt 5 = 0

-- Theorem statement
theorem min_polyline_distance :
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      on_circle x₁ y₁ → on_line x₂ y₂ →
      polyline_distance x₁ y₁ x₂ y₂ ≥ min_dist) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      on_circle x₁ y₁ ∧ on_line x₂ y₂ ∧
      polyline_distance x₁ y₁ x₂ y₂ = min_dist) ∧
    min_dist = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_polyline_distance_l2594_259471


namespace NUMINAMATH_CALUDE_circle_area_proof_l2594_259491

/-- The area of a circle with center at (-5, 3) and touching the point (7, -4) is 193π. -/
theorem circle_area_proof : 
  let center : ℝ × ℝ := (-5, 3)
  let point : ℝ × ℝ := (7, -4)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  let area : ℝ := π * radius^2
  area = 193 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_proof_l2594_259491


namespace NUMINAMATH_CALUDE_salesman_profit_salesman_profit_is_442_l2594_259442

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (total_cost : ℕ) 
  (first_sale_quantity : ℕ) (first_sale_price : ℕ)
  (second_sale_quantity : ℕ) (second_sale_price : ℕ)
  (remaining_price : ℕ) : ℕ :=
  let remaining_quantity := total_backpacks - first_sale_quantity - second_sale_quantity
  let total_revenue := 
    first_sale_quantity * first_sale_price +
    second_sale_quantity * second_sale_price +
    remaining_quantity * remaining_price
  total_revenue - total_cost

/-- The salesman's profit is $442 --/
theorem salesman_profit_is_442 : 
  salesman_profit 48 576 17 18 10 25 22 = 442 := by
  sorry

end NUMINAMATH_CALUDE_salesman_profit_salesman_profit_is_442_l2594_259442


namespace NUMINAMATH_CALUDE_equation_solution_l2594_259477

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (3 * x) / (x - 1) = 2 + 1 / (x - 1) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2594_259477


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l2594_259435

theorem gcd_powers_of_two : Nat.gcd (2^2025 - 1) (2^2007 - 1) = 2^18 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l2594_259435


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2594_259413

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 15 / Real.sqrt 21) = 
  (2 * Real.sqrt 105) / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2594_259413


namespace NUMINAMATH_CALUDE_least_multiplier_for_72_to_be_multiple_of_112_l2594_259464

theorem least_multiplier_for_72_to_be_multiple_of_112 :
  (∃ n : ℕ+, (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) ∧
  (∃ n : ℕ+, n = 14 ∧ (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_least_multiplier_for_72_to_be_multiple_of_112_l2594_259464


namespace NUMINAMATH_CALUDE_ceiling_fraction_equality_l2594_259467

theorem ceiling_fraction_equality : 
  (⌈(23 : ℚ) / 9 - ⌈(35 : ℚ) / 23⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 : ℚ) * 23 / 35⌉⌉) = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_equality_l2594_259467


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l2594_259492

/-- Given a triangle ABC with point D such that BD = 2DC, prove that AD = (1/3)AB + (2/3)AC -/
theorem triangle_vector_relation (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - D) = 2 • (D - C) →
  (A - D) = (1 / 3) • (A - B) + (2 / 3) • (A - C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l2594_259492


namespace NUMINAMATH_CALUDE_book_pages_l2594_259458

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 8 → days = 12 → fraction_read = 2/3 →
  (pages_per_day * days : ℚ) / fraction_read = 144 := by
sorry

end NUMINAMATH_CALUDE_book_pages_l2594_259458


namespace NUMINAMATH_CALUDE_cherries_theorem_l2594_259446

def cherries_problem (initial : ℕ) (eaten : ℕ) : ℕ :=
  let remaining := initial - eaten
  let given_away := remaining / 2
  remaining - given_away

theorem cherries_theorem :
  cherries_problem 2450 1625 = 413 := by
  sorry

end NUMINAMATH_CALUDE_cherries_theorem_l2594_259446


namespace NUMINAMATH_CALUDE_first_divisor_problem_l2594_259452

theorem first_divisor_problem (n d : ℕ) : 
  n > 1 →
  n % d = 1 →
  n % 7 = 1 →
  (∀ m : ℕ, m > 1 ∧ m % d = 1 ∧ m % 7 = 1 → m ≥ n) →
  n = 175 →
  d = 29 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l2594_259452


namespace NUMINAMATH_CALUDE_min_fuel_for_four_thirds_distance_l2594_259410

/-- Represents the fuel efficiency of a truck -/
structure TruckEfficiency where
  L : ℝ  -- Capacity of the truck in liters
  a : ℝ  -- Distance the truck can travel with L liters
  h_positive : 0 < L ∧ 0 < a

/-- The minimum fuel required for a truck to reach a destination -/
def min_fuel_required (e : TruckEfficiency) (d : ℝ) : ℝ :=
  2 * e.L

theorem min_fuel_for_four_thirds_distance (e : TruckEfficiency) :
  min_fuel_required e ((4 / 3) * e.a) = 2 * e.L :=
by sorry

end NUMINAMATH_CALUDE_min_fuel_for_four_thirds_distance_l2594_259410


namespace NUMINAMATH_CALUDE_power_of_two_l2594_259487

theorem power_of_two (k : ℕ) (h : 2^k = 4) : 2^(3*k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l2594_259487


namespace NUMINAMATH_CALUDE_library_books_total_l2594_259490

theorem library_books_total (initial_books additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
  sorry

end NUMINAMATH_CALUDE_library_books_total_l2594_259490


namespace NUMINAMATH_CALUDE_prime_cube_difference_l2594_259445

theorem prime_cube_difference (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 11 * p = q^3 - r^3 → 
  p = 199 ∧ q = 13 ∧ r = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_l2594_259445


namespace NUMINAMATH_CALUDE_linear_function_properties_l2594_259415

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_properties :
  ∀ (k b : ℝ), k ≠ 0 →
  linear_function k b 2 = -3 →
  linear_function k b (-4) = 0 →
  (k = -1/2 ∧ b = -2) ∧
  (∀ (x m : ℝ), x > -2 → -x + m < linear_function k b x → m ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2594_259415


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2594_259449

theorem inequality_system_solution : 
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2594_259449


namespace NUMINAMATH_CALUDE_line_plane_parallelism_condition_l2594_259459

-- Define the concepts of line and plane
variable (m : Line) (α : Plane)

-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for a line to be parallel to countless lines in a plane
def line_parallel_to_countless_lines_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_parallelism_condition :
  (line_parallel_to_countless_lines_in_plane m α → line_parallel_to_plane m α) ∧
  ¬(line_parallel_to_plane m α → line_parallel_to_countless_lines_in_plane m α) := by sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_condition_l2594_259459


namespace NUMINAMATH_CALUDE_nina_total_homework_l2594_259403

/-- Represents the number of homework assignments for a student -/
structure Homework where
  math : ℕ
  reading : ℕ

/-- Calculates the total number of homework assignments -/
def totalHomework (hw : Homework) : ℕ := hw.math + hw.reading

theorem nina_total_homework :
  let ruby : Homework := { math := 6, reading := 2 }
  let nina : Homework := { math := 4 * ruby.math, reading := 8 * ruby.reading }
  totalHomework nina = 40 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_homework_l2594_259403


namespace NUMINAMATH_CALUDE_james_stickers_after_birthday_l2594_259418

/-- The number of stickers James had after his birthday -/
def total_stickers (initial : ℕ) (birthday : ℕ) : ℕ :=
  initial + birthday

/-- Theorem stating that James had 61 stickers after his birthday -/
theorem james_stickers_after_birthday :
  total_stickers 39 22 = 61 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_after_birthday_l2594_259418


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2594_259470

/-- Prove that given a journey of 225 km completed in 10 hours, 
    where the first half is traveled at 21 km/hr, 
    the speed for the second half of the journey is approximately 24.23 km/hr. -/
theorem journey_speed_calculation 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 225) 
  (h2 : total_time = 10) 
  (h3 : first_half_speed = 21) : 
  ∃ (second_half_speed : ℝ), 
    (abs (second_half_speed - 24.23) < 0.01) ∧ 
    (total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time) :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2594_259470


namespace NUMINAMATH_CALUDE_calculate_F_l2594_259451

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 5*a + 6
def F (a b c : ℝ) : ℝ := b^2 + a*c + 1

-- State the theorem
theorem calculate_F : F 3 (f 3) (f 5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_l2594_259451


namespace NUMINAMATH_CALUDE_trail_mix_weight_l2594_259448

def peanuts_weight : Float := 0.16666666666666666
def chocolate_chips_weight : Float := 0.16666666666666666
def raisins_weight : Float := 0.08333333333333333

theorem trail_mix_weight :
  peanuts_weight + chocolate_chips_weight + raisins_weight = 0.41666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l2594_259448


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2594_259408

theorem factorization_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2594_259408


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisibility_l2594_259438

theorem unique_three_digit_number_divisibility : ∃! a : ℕ, 
  100 ≤ a ∧ a < 1000 ∧ 
  (∃ k : ℕ, 504000 + a = 693 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisibility_l2594_259438


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l2594_259447

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- Define the line
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | p ∈ line ∧ ∃ t, curve t = p}

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l2594_259447


namespace NUMINAMATH_CALUDE_tangent_product_approximation_l2594_259406

theorem tangent_product_approximation :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  ∃ ε > 0, |(1 + Real.tan A) * (1 + Real.tan B) - 2.9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_product_approximation_l2594_259406


namespace NUMINAMATH_CALUDE_cheaper_candy_price_l2594_259424

/-- Proves that the price of the cheaper candy is $2 per pound -/
theorem cheaper_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheaper_weight : ℝ)
  (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheaper_weight = 64)
  (h4 : expensive_price = 3)
  : ∃ (cheaper_price : ℝ),
    cheaper_price * cheaper_weight + expensive_price * (total_weight - cheaper_weight) =
    mixture_price * total_weight ∧ cheaper_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_candy_price_l2594_259424


namespace NUMINAMATH_CALUDE_exponent_division_l2594_259499

theorem exponent_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2594_259499


namespace NUMINAMATH_CALUDE_row_properties_l2594_259484

/-- Definition of a number being in a row -/
def in_row (m n : ℕ) : Prop :=
  n ∣ m ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row m k

/-- The main theorem encompassing all parts of the problem -/
theorem row_properties :
  (∀ m < 50, m % 10 = 0 → ∃ k < 10, in_row m k) ∧
  (∀ n ≥ 3, in_row (n^2 - n) n ∧ in_row (n^2 - 2*n) n) ∧
  (∀ n > 30, in_row (n^2 - 10*n) n) ∧
  ¬in_row (30^2 - 10*30) 30 := by
  sorry

#check row_properties

end NUMINAMATH_CALUDE_row_properties_l2594_259484


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_indices_l2594_259457

theorem arithmetic_geometric_sequence_indices 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (k : ℕ → ℕ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ r, ∀ n, a (k (n + 1)) = a (k n) * r)
  (h4 : k 1 = 1)
  (h5 : k 2 = 2)
  (h6 : k 3 = 6) :
  k 4 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_indices_l2594_259457


namespace NUMINAMATH_CALUDE_correlation_strength_theorem_l2594_259417

-- Define the correlation coefficient r
def correlation_coefficient (r : ℝ) : Prop := -1 < r ∧ r < 1

-- Define the strength of correlation
def correlation_strength (r : ℝ) : ℝ := |r|

-- Theorem stating the relationship between |r| and correlation strength
theorem correlation_strength_theorem (r : ℝ) (h : correlation_coefficient r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', correlation_coefficient r' →
    correlation_strength r' < δ → correlation_strength r' < ε :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_theorem_l2594_259417


namespace NUMINAMATH_CALUDE_race_elimination_ratio_l2594_259461

/-- Proves the ratio of racers eliminated before the last leg to remaining racers after the second segment --/
theorem race_elimination_ratio :
  let initial_racers : ℕ := 100
  let first_elimination : ℕ := 10
  let final_racers : ℕ := 30
  let after_first_segment : ℕ := initial_racers - first_elimination
  let second_elimination : ℕ := after_first_segment / 3
  let after_second_segment : ℕ := after_first_segment - second_elimination
  let eliminated_before_last : ℕ := after_second_segment - final_racers
  (eliminated_before_last : ℚ) / (after_second_segment : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_race_elimination_ratio_l2594_259461


namespace NUMINAMATH_CALUDE_average_beef_sold_is_260_l2594_259454

/-- The average amount of beef sold per day over three days -/
def average_beef_sold (thursday_sales : ℕ) (saturday_sales : ℕ) : ℚ :=
  (thursday_sales + 2 * thursday_sales + saturday_sales) / 3

/-- Proof that the average amount of beef sold per day is 260 pounds -/
theorem average_beef_sold_is_260 :
  average_beef_sold 210 150 = 260 := by
  sorry

end NUMINAMATH_CALUDE_average_beef_sold_is_260_l2594_259454


namespace NUMINAMATH_CALUDE_car_efficiency_problem_l2594_259426

/-- The combined fuel efficiency of two cars -/
def combined_efficiency (e1 e2 : ℚ) : ℚ :=
  2 / (1 / e1 + 1 / e2)

theorem car_efficiency_problem :
  let ray_efficiency : ℚ := 50
  let tom_efficiency : ℚ := 15
  combined_efficiency ray_efficiency tom_efficiency = 300 / 13 := by
sorry

end NUMINAMATH_CALUDE_car_efficiency_problem_l2594_259426


namespace NUMINAMATH_CALUDE_min_value_of_f_l2594_259465

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2594_259465


namespace NUMINAMATH_CALUDE_two_sarees_four_shirts_cost_l2594_259456

/-- The price of a single saree -/
def saree_price : ℝ := sorry

/-- The price of a single shirt -/
def shirt_price : ℝ := sorry

/-- The cost of 2 sarees and 4 shirts equals the cost of 1 saree and 6 shirts -/
axiom price_equality : 2 * saree_price + 4 * shirt_price = saree_price + 6 * shirt_price

/-- The price of 12 shirts is $2400 -/
axiom twelve_shirts_price : 12 * shirt_price = 2400

/-- The theorem stating that 2 sarees and 4 shirts cost $1600 -/
theorem two_sarees_four_shirts_cost : 2 * saree_price + 4 * shirt_price = 1600 := by sorry

end NUMINAMATH_CALUDE_two_sarees_four_shirts_cost_l2594_259456


namespace NUMINAMATH_CALUDE_correct_transformation_l2594_259423

theorem correct_transformation (x : ℝ) : 2*x - 5 = 3*x + 3 → 2*x - 3*x = 3 + 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l2594_259423


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l2594_259420

/-- A convex polyhedron is represented by its number of vertices, edges, and maximum number of triangular faces sharing a common vertex. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  T : ℕ  -- maximum number of triangular faces sharing a common vertex

/-- The inequality V ≤ √E + T holds for any convex polyhedron. -/
theorem polyhedron_inequality (P : ConvexPolyhedron) : P.V ≤ Real.sqrt (P.E : ℝ) + P.T := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l2594_259420


namespace NUMINAMATH_CALUDE_expression_simplification_l2594_259466

theorem expression_simplification (x : ℤ) 
  (h1 : x - 3 * (x - 2) ≥ 2) 
  (h2 : 4 * x - 2 < 5 * x - 1) : 
  (3 / (x - 1) - x - 1) / ((x - 2) / (x - 1)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2594_259466


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l2594_259428

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) ∧ 
  (∀ (p q : ℝ → ℝ) (a b c : ℝ), (w^4 - 81 = p w * q w ∧ p a = 0 ∧ q b = 0) → 
    (c = 3 ∨ c = -3 ∨ (c^2 = -9 ∧ (∀ x : ℝ, x^2 ≠ -9)))) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l2594_259428


namespace NUMINAMATH_CALUDE_matching_arrangements_count_l2594_259469

def number_of_people : Nat := 5

/-- The number of arrangements where exactly two people sit in seats matching their numbers -/
def matching_arrangements : Nat :=
  (number_of_people.choose 2) * 2 * 1 * 1

theorem matching_arrangements_count : matching_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_matching_arrangements_count_l2594_259469


namespace NUMINAMATH_CALUDE_line_through_quadrants_line_passes_through_point_point_slope_equation_line_equation_l2594_259409

-- Statement 1
theorem line_through_quadrants (k b : ℝ) :
  (∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y > 0) ∧
  (∃ x y : ℝ, y = k * x + b ∧ x < 0 ∧ y > 0) ∧
  (∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y < 0) →
  k < 0 ∧ b > 0 :=
sorry

-- Statement 2
theorem line_passes_through_point (a : ℝ) :
  2 = a * 3 - 3 * a + 2 :=
sorry

-- Statement 3
theorem point_slope_equation :
  ∀ x y : ℝ, y + 1 = -Real.sqrt 3 * (x - 2) ↔ y = -Real.sqrt 3 * (x - 2) - 1 :=
sorry

-- Statement 4
theorem line_equation (x y : ℝ) :
  y = -2 * x + 3 ↔ y - 3 = -2 * (x - 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_quadrants_line_passes_through_point_point_slope_equation_line_equation_l2594_259409


namespace NUMINAMATH_CALUDE_luke_garage_sale_games_l2594_259494

/-- Represents the number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := sorry

/-- Represents the number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- Represents the number of games that didn't work -/
def non_working_games : ℕ := 2

/-- Represents the number of good games Luke ended up with -/
def good_games : ℕ := 2

theorem luke_garage_sale_games :
  games_from_garage_sale = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_garage_sale_games_l2594_259494


namespace NUMINAMATH_CALUDE_bananas_and_cantaloupe_cost_l2594_259444

/-- Represents the cost of various fruits -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  figs : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (costs : FruitCosts) : Prop :=
  costs.apples + costs.bananas + costs.cantaloupe + costs.dates + costs.figs = 30 ∧
  costs.dates = 3 * costs.apples ∧
  costs.cantaloupe = costs.apples - costs.bananas ∧
  costs.figs = costs.bananas

/-- The theorem stating that the cost of bananas and cantaloupe is 6 -/
theorem bananas_and_cantaloupe_cost (costs : FruitCosts) 
  (h : fruitProblemConditions costs) : 
  costs.bananas + costs.cantaloupe = 6 := by
  sorry


end NUMINAMATH_CALUDE_bananas_and_cantaloupe_cost_l2594_259444


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_for_forest_data_l2594_259455

/-- Represents types of statistical graphs -/
inductive StatisticalGraph
| LineGraph
| BarChart
| PieChart
| ScatterPlot
| Histogram

/-- Represents characteristics of data and analysis requirements -/
structure DataCharacteristics where
  continuous : Bool
  timeSpan : ℕ
  decreasingTrend : Bool

/-- Determines the most suitable graph type for given data characteristics -/
def mostSuitableGraph (data : DataCharacteristics) : StatisticalGraph :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given forest area data -/
theorem line_graph_most_suitable_for_forest_data :
  let forestData : DataCharacteristics := {
    continuous := true,
    timeSpan := 20,
    decreasingTrend := true
  }
  mostSuitableGraph forestData = StatisticalGraph.LineGraph :=
sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_for_forest_data_l2594_259455


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l2594_259427

/-- Proves that the number of meters of cloth sold is 40, given the conditions of the problem -/
theorem cloth_sale_meters : 
  -- C represents the cost price of 1 meter of cloth
  ∀ (C : ℝ), C > 0 →
  -- S represents the selling price of 1 meter of cloth
  ∀ (S : ℝ), S > C →
  -- The gain is the selling price of 10 meters
  let G := 10 * S
  -- The gain percentage is 1/3 (33.33333333333333%)
  let gain_percentage := (1 : ℝ) / 3
  -- M represents the number of meters sold
  ∃ (M : ℝ),
    -- The gain is equal to the gain percentage times the total cost
    G = gain_percentage * (M * C) ∧
    -- The selling price is the cost price plus the gain per meter
    S = C + G / M ∧
    -- The number of meters sold is 40
    M = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l2594_259427


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2594_259453

theorem binomial_divisibility (p n : ℕ) (hp : Prime p) (hn : p < n) 
  (hdiv : p ∣ (n + 1)) (hcoprime : Nat.gcd (n / p) (Nat.factorial (p - 1)) = 1) :
  p * (n / p)^2 ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2594_259453


namespace NUMINAMATH_CALUDE_total_smoothie_ingredients_l2594_259405

def strawberries : ℝ := 0.2
def yogurt : ℝ := 0.1
def orange_juice : ℝ := 0.2
def spinach : ℝ := 0.15
def protein_powder : ℝ := 0.05

theorem total_smoothie_ingredients :
  strawberries + yogurt + orange_juice + spinach + protein_powder = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_smoothie_ingredients_l2594_259405


namespace NUMINAMATH_CALUDE_fifteenth_term_is_101_l2594_259430

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 15th term of the arithmetic sequence with first term 3 and common difference 7 is 101 -/
theorem fifteenth_term_is_101 : arithmeticSequence 3 7 15 = 101 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_101_l2594_259430


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2594_259463

def initial_earnings : ℝ := 65
def new_earnings : ℝ := 72

theorem percentage_increase_proof :
  let difference := new_earnings - initial_earnings
  let percentage_increase := (difference / initial_earnings) * 100
  ∀ ε > 0, |percentage_increase - 10.77| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2594_259463


namespace NUMINAMATH_CALUDE_slope_product_no_circle_through_A_l2594_259440

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse
def P (x₀ y₀ : ℝ) : Prop := E x₀ y₀ ∧ (x₀, y₀) ≠ A ∧ (x₀, y₀) ≠ B

-- Theorem: Product of slopes of PA and PB is -1/4
theorem slope_product (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -1/4 := by sorry

-- No circle with diameter MN passes through A
-- This part is more complex and would require additional definitions and theorems
-- We'll represent it as a proposition without proof
theorem no_circle_through_A (M N : ℝ × ℝ) (hM : E M.1 M.2) (hN : E N.1 N.2) :
  ¬∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
    (center.1 - N.1)^2 + (center.2 - N.2)^2 = radius^2 ∧
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_slope_product_no_circle_through_A_l2594_259440


namespace NUMINAMATH_CALUDE_equation_solution_l2594_259441

theorem equation_solution (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2594_259441


namespace NUMINAMATH_CALUDE_game_outcome_l2594_259460

def p1 : Nat := 10^9 + 7
def p2 : Nat := 10^9 + 9

axiom p1_prime : Nat.Prime p1
axiom p2_prime : Nat.Prime p2

theorem game_outcome (x : Nat) : 
  (∀ y : Nat, y < p1 → (y^2 - y + 1) % p1 ≠ 0) ∧
  (∃ z : Nat, z < p2 ∧ (z^2 - z + 1) % p2 = 0) :=
sorry

end NUMINAMATH_CALUDE_game_outcome_l2594_259460


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l2594_259498

/-- Given a parabola y = ax^2 + bx + c with vertex (q, q+1) and y-intercept (0, -2q-1),
    where q ≠ -1/2, prove that b = 6 + 4/q -/
theorem parabola_coefficient_b (a b c q : ℝ) (h : q ≠ -1/2) :
  (∀ x y, y = a * x^2 + b * x + c) →
  (q + 1 = a * q^2 + b * q + c) →
  (-2 * q - 1 = c) →
  b = 6 + 4 / q := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l2594_259498


namespace NUMINAMATH_CALUDE_visitor_growth_and_optimal_price_l2594_259421

def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000
def cost_price : ℚ := 6
def initial_price : ℚ := 25
def initial_sales : ℕ := 300
def sales_increase : ℕ := 30
def target_profit : ℚ := 6300

theorem visitor_growth_and_optimal_price :
  -- Part 1: Average annual growth rate
  ∃ (growth_rate : ℚ),
    (1 + growth_rate) ^ 2 * visitors_2022 = visitors_2024 ∧
    growth_rate = 1/5 ∧
  -- Part 2: Optimal selling price
  ∃ (optimal_price : ℚ),
    optimal_price ≤ initial_price ∧
    (optimal_price - cost_price) *
      (initial_sales + sales_increase * (initial_price - optimal_price)) =
      target_profit ∧
    optimal_price = 20 :=
  sorry

end NUMINAMATH_CALUDE_visitor_growth_and_optimal_price_l2594_259421


namespace NUMINAMATH_CALUDE_age_ratio_is_three_to_two_l2594_259472

/-- Represents the ages of a man and his wife -/
structure Couple where
  man_age : ℕ
  wife_age : ℕ

/-- The conditions of the problem -/
def problem_conditions (c : Couple) : Prop :=
  c.man_age = 30 ∧
  c.wife_age = 30 ∧
  c.man_age - 10 = c.wife_age ∧
  ∃ k : ℚ, c.man_age = k * (c.wife_age - 10)

/-- The theorem to be proved -/
theorem age_ratio_is_three_to_two (c : Couple) 
  (h : problem_conditions c) : 
  c.man_age / (c.wife_age - 10) = 3 / 2 := by
  sorry

#check age_ratio_is_three_to_two

end NUMINAMATH_CALUDE_age_ratio_is_three_to_two_l2594_259472


namespace NUMINAMATH_CALUDE_weighted_average_plants_per_hour_l2594_259493

def total_rows : ℕ := 400
def carrot_rows : ℕ := 250
def potato_rows : ℕ := 150

def carrot_first_rows : ℕ := 100
def carrot_first_plants_per_row : ℕ := 275
def carrot_first_time : ℕ := 10

def carrot_remaining_rows : ℕ := 150
def carrot_remaining_plants_per_row : ℕ := 325
def carrot_remaining_time : ℕ := 20

def potato_first_rows : ℕ := 50
def potato_first_plants_per_row : ℕ := 300
def potato_first_time : ℕ := 12

def potato_remaining_rows : ℕ := 100
def potato_remaining_plants_per_row : ℕ := 400
def potato_remaining_time : ℕ := 18

theorem weighted_average_plants_per_hour :
  let total_plants := 
    (carrot_first_rows * carrot_first_plants_per_row + 
     carrot_remaining_rows * carrot_remaining_plants_per_row +
     potato_first_rows * potato_first_plants_per_row + 
     potato_remaining_rows * potato_remaining_plants_per_row)
  let total_time := 
    (carrot_first_time + carrot_remaining_time + 
     potato_first_time + potato_remaining_time)
  (total_plants : ℚ) / total_time = 2187.5 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_plants_per_hour_l2594_259493


namespace NUMINAMATH_CALUDE_acid_dilution_l2594_259483

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 ∧ 
  initial_concentration = 0.4 ∧ 
  water_added = 30 ∧ 
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_acid_dilution_l2594_259483
