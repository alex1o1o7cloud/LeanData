import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l1967_196745

/-- An arithmetic sequence with first term -5 and positive terms starting from the 10th term
    has a common difference d in the range (5/9, 5/8] -/
theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a n = -5 + (n - 1) * d) →  -- Definition of arithmetic sequence
  (a 1 = -5) →                         -- First term is -5
  (∀ n ≥ 10, a n > 0) →                -- Terms from 10th onwards are positive
  5/9 < d ∧ d ≤ 5/8 :=                 -- Range of common difference
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l1967_196745


namespace NUMINAMATH_CALUDE_four_students_seven_seats_l1967_196766

/-- The number of ways to arrange students in seats with adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (students : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 480 ways to arrange 4 students in 7 seats with 2 adjacent empty seats -/
theorem four_students_seven_seats : seating_arrangements 7 4 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_four_students_seven_seats_l1967_196766


namespace NUMINAMATH_CALUDE_problem_statement_l1967_196731

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + x^2 + (a - 1) * x + 1

/-- The theorem statement -/
theorem problem_statement (a : ℝ) :
  (∀ x > 0, Real.exp x > x + 1) →
  (∀ x > 0, f a x ≤ x^4 + Real.exp x) →
  a ≤ Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1967_196731


namespace NUMINAMATH_CALUDE_negative_a_exponent_division_l1967_196775

theorem negative_a_exponent_division (a : ℝ) : (-a)^10 / (-a)^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_exponent_division_l1967_196775


namespace NUMINAMATH_CALUDE_condition_analysis_l1967_196722

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * b^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ (a - b) * b^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l1967_196722


namespace NUMINAMATH_CALUDE_min_value_line_circle_l1967_196724

/-- Given a line ax + by + c - 1 = 0 (where b, c > 0) passing through the center of the circle x^2 + y^2 - 2y - 5 = 0, 
    the minimum value of 4/b + 1/c is 9. -/
theorem min_value_line_circle (a b c : ℝ) : 
  b > 0 → c > 0 → 
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (∃ x y : ℝ, a * x + b' * y + c' - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
    4/b + 1/c ≤ 4/b' + 1/c') →
  4/b + 1/c = 9 :=
by sorry


end NUMINAMATH_CALUDE_min_value_line_circle_l1967_196724


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l1967_196706

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of dogwood trees after planting is 83 -/
theorem dogwood_tree_count : total_trees 34 49 = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l1967_196706


namespace NUMINAMATH_CALUDE_angle_TSB_closest_to_27_l1967_196798

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The art gallery setup -/
structure ArtGallery where
  B : Point -- Bottom of painting
  T : Point -- Top of painting
  S : Point -- Spotlight position

/-- Definition of the art gallery setup based on given conditions -/
def setupGallery : ArtGallery :=
  { B := ⟨0, 1⟩,    -- Bottom of painting (0, 1)
    T := ⟨0, 3⟩,    -- Top of painting (0, 3)
    S := ⟨3, 4⟩ }   -- Spotlight position (3, 4)

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Theorem stating that the angle TSB is closest to 27° -/
theorem angle_TSB_closest_to_27 (g : ArtGallery) :
  let angleTSB := angle g.T g.S g.B
  ∀ x ∈ [27, 63, 34, 45, 18], |angleTSB - 27| ≤ |angleTSB - x| :=
by sorry

end NUMINAMATH_CALUDE_angle_TSB_closest_to_27_l1967_196798


namespace NUMINAMATH_CALUDE_archibald_win_percentage_l1967_196746

theorem archibald_win_percentage (archibald_wins brother_wins : ℕ) : 
  archibald_wins = 12 → brother_wins = 18 → 
  (archibald_wins : ℚ) / (archibald_wins + brother_wins : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_archibald_win_percentage_l1967_196746


namespace NUMINAMATH_CALUDE_digit_five_occurrences_l1967_196781

/-- The number of occurrences of a digit in a specific place value when writing numbers from 1 to n -/
def occurrences_in_place (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) * (10 ^ (place - 1))

/-- The total number of occurrences of the digit 5 when writing all integers from 1 to n -/
def total_occurrences (n : ℕ) : ℕ :=
  occurrences_in_place n 0 + occurrences_in_place n 1 + 
  occurrences_in_place n 2 + occurrences_in_place n 3

theorem digit_five_occurrences :
  total_occurrences 10000 = 4000 := by sorry

end NUMINAMATH_CALUDE_digit_five_occurrences_l1967_196781


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l1967_196756

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 918

-- Define a function that returns the molecular weight for any number of moles
def weight_for_moles (moles : ℝ) : ℝ := molecular_weight

-- Theorem stating that the molecular weight is constant regardless of the number of moles
theorem constant_molecular_weight (moles : ℝ) :
  weight_for_moles moles = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l1967_196756


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l1967_196704

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 10| + |x - 1|

theorem sum_of_max_and_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → g x ≤ max) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → min ≤ g x) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = min) ∧
    max + min = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l1967_196704


namespace NUMINAMATH_CALUDE_largest_root_cubic_equation_l1967_196788

theorem largest_root_cubic_equation (a₂ a₁ a₀ : ℝ) 
  (h₂ : |a₂| < 2) (h₁ : |a₁| < 2) (h₀ : |a₀| < 2) :
  ∃ r : ℝ, r > 0 ∧ r^3 + a₂*r^2 + a₁*r + a₀ = 0 ∧
  (∀ x : ℝ, x > 0 ∧ x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r) ∧
  (5/2 < r ∧ r < 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_root_cubic_equation_l1967_196788


namespace NUMINAMATH_CALUDE_sqrt_two_between_one_and_two_l1967_196764

theorem sqrt_two_between_one_and_two :
  1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_between_one_and_two_l1967_196764


namespace NUMINAMATH_CALUDE_digit_difference_after_reversal_l1967_196726

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  h_hundreds := n.h_units
  h_tens := n.h_tens
  h_units := n.h_hundreds

theorem digit_difference_after_reversal
  (numbers : Finset ThreeDigitNumber)
  (reversed : ThreeDigitNumber)
  (h_count : numbers.card = 10)
  (h_reversed_in : reversed ∈ numbers)
  (h_average_increase : (numbers.sum value + value (reverse reversed) - value reversed) / 10 - numbers.sum value / 10 = 198 / 10) :
  (reverse reversed).units - (reverse reversed).hundreds = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_after_reversal_l1967_196726


namespace NUMINAMATH_CALUDE_calculation_proofs_l1967_196759

theorem calculation_proofs :
  (4.5 * 0.9 + 5.5 * 0.9 = 9) ∧
  (1.6 * (2.25 + 10.5 / 1.5) = 14.8) ∧
  (0.36 / ((6.1 - 4.6) * 0.8) = 0.3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1967_196759


namespace NUMINAMATH_CALUDE_fraction_equality_l1967_196703

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 3 * y) / (x + 4 * y) = 3) : 
  (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1967_196703


namespace NUMINAMATH_CALUDE_unicycle_count_l1967_196758

theorem unicycle_count :
  ∀ (num_bicycles num_tricycles num_unicycles : ℕ),
    num_bicycles = 3 →
    num_tricycles = 4 →
    num_bicycles * 2 + num_tricycles * 3 + num_unicycles * 1 = 25 →
    num_unicycles = 7 := by
  sorry

end NUMINAMATH_CALUDE_unicycle_count_l1967_196758


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l1967_196730

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l1967_196730


namespace NUMINAMATH_CALUDE_city_distance_proof_l1967_196770

/-- Given a map distance between two cities and a map scale, calculates the actual distance between the cities. -/
def actualDistance (mapDistance : ℝ) (mapScale : ℝ) : ℝ :=
  mapDistance * mapScale

/-- Theorem stating that for a map distance of 120 cm and a scale of 1 cm : 20 km, the actual distance is 2400 km. -/
theorem city_distance_proof :
  let mapDistance : ℝ := 120
  let mapScale : ℝ := 20
  actualDistance mapDistance mapScale = 2400 := by
  sorry

#eval actualDistance 120 20

end NUMINAMATH_CALUDE_city_distance_proof_l1967_196770


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l1967_196725

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l1967_196725


namespace NUMINAMATH_CALUDE_focus_coordinates_l1967_196702

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1 ∧ a > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = 3 * x

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop :=
  y = 2 * a * x^2

-- State the theorem
theorem focus_coordinates (a : ℝ) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  (∃ x y : ℝ, parabola a x y ∧ x = 0 ∧ y = 1/8) :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l1967_196702


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1967_196795

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/216
  let r : ℚ := a₂ / a₁
  r = -16/27 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1967_196795


namespace NUMINAMATH_CALUDE_min_side_diff_is_one_l1967_196779

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  AB : ℕ
  BC : ℕ
  AC : ℕ

/-- The perimeter of the triangle -/
def Triangle.perimeter (t : Triangle) : ℕ := t.AB + t.BC + t.AC

/-- The difference between the longest and second longest sides -/
def Triangle.sideDiff (t : Triangle) : ℕ := t.AC - t.BC

/-- Predicate for a valid triangle satisfying the given conditions -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.AB ≤ t.BC ∧ t.BC < t.AC ∧ t.perimeter = 2020

theorem min_side_diff_is_one :
  ∃ (t : Triangle), t.isValid ∧
    ∀ (t' : Triangle), t'.isValid → t.sideDiff ≤ t'.sideDiff :=
by sorry

end NUMINAMATH_CALUDE_min_side_diff_is_one_l1967_196779


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1967_196747

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1967_196747


namespace NUMINAMATH_CALUDE_cuboid_volume_error_percentage_l1967_196717

/-- The error percentage in volume calculation for a cuboid with specific measurement errors -/
theorem cuboid_volume_error_percentage :
  let length_error := 1.08  -- 8% excess
  let breadth_error := 0.95 -- 5% deficit
  let height_error := 0.90  -- 10% deficit
  let volume_error := length_error * breadth_error * height_error
  let error_percentage := (volume_error - 1) * 100
  error_percentage = -2.74 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_error_percentage_l1967_196717


namespace NUMINAMATH_CALUDE_equation_solution_l1967_196754

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + x + 1) / (x + 1) = x + 2 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1967_196754


namespace NUMINAMATH_CALUDE_test_scores_l1967_196701

/-- Represents the score of a test -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  score : Nat

/-- Calculates the score for a given test result -/
def calculateScore (correct unanswered incorrect : Nat) : Nat :=
  6 * correct + unanswered

/-- Checks if a given score is achievable on the test -/
def isAchievableScore (s : Nat) : Prop :=
  ∃ (correct unanswered incorrect : Nat),
    correct + unanswered + incorrect = 25 ∧
    calculateScore correct unanswered incorrect = s

theorem test_scores :
  (isAchievableScore 130) ∧
  (isAchievableScore 131) ∧
  (isAchievableScore 133) ∧
  (isAchievableScore 138) ∧
  ¬(isAchievableScore 139) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l1967_196701


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_310_l1967_196738

theorem largest_common_divisor_408_310 : 
  (∀ n : ℕ, n > 2 → (n ∣ 408 ∧ n ∣ 310) → False) ∧ (2 ∣ 408 ∧ 2 ∣ 310) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_310_l1967_196738


namespace NUMINAMATH_CALUDE_range_of_a_l1967_196708

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

-- Note: Set.Icc represents a closed interval [1/2, 2]

end NUMINAMATH_CALUDE_range_of_a_l1967_196708


namespace NUMINAMATH_CALUDE_unique_non_range_value_l1967_196772

/-- The function g defined as (px + q) / (rx + s) -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- The theorem stating that 30 is the unique number not in the range of g -/
theorem unique_non_range_value
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_11 : g p q r s 11 = 11)
  (h_41 : g p q r s 41 = 41)
  (h_inverse : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, ∀ x, g p q r s x ≠ y ∧ y = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l1967_196772


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l1967_196718

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u v w : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hu : M.mulVec u = ![(-3), 4])
  (hv : M.mulVec v = ![2, (-7)])
  (hw : M.mulVec w = ![9, 0]) :
  M.mulVec (3 • u - 4 • v + 2 • w) = ![1, 40] := by
sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l1967_196718


namespace NUMINAMATH_CALUDE_diamond_commutative_l1967_196791

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

-- Theorem statement
theorem diamond_commutative : ∀ x y : ℝ, diamond x y = diamond y x := by
  sorry

end NUMINAMATH_CALUDE_diamond_commutative_l1967_196791


namespace NUMINAMATH_CALUDE_house_cost_l1967_196723

theorem house_cost (total : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total = 120000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.05)
  (h4 : total_interest = 3920) :
  ∃ (house_cost : ℝ),
    house_cost = 36000 ∧
    (1/3 * (total - house_cost) * interest_rate1 + 2/3 * (total - house_cost) * interest_rate2 = total_interest) :=
by
  sorry

end NUMINAMATH_CALUDE_house_cost_l1967_196723


namespace NUMINAMATH_CALUDE_max_large_chips_l1967_196736

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (composite : ℕ → ℕ) : 
  total = 72 →
  (∀ n, total = small n + large n) →
  (∀ n, small n = large n + composite n) →
  (∀ n, composite n ≥ 4) →
  (∃ max_large : ℕ, ∀ n, large n ≤ max_large ∧ (∃ m, large m = max_large)) →
  (∃ max_large : ℕ, max_large = 34 ∧ ∀ n, large n ≤ max_large) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l1967_196736


namespace NUMINAMATH_CALUDE_det_A_equals_two_l1967_196769

open Matrix

theorem det_A_equals_two (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A + 2 * A⁻¹ = 0) : 
  det A = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l1967_196769


namespace NUMINAMATH_CALUDE_complex_on_line_l1967_196767

theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → a = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l1967_196767


namespace NUMINAMATH_CALUDE_sqrt_x_squared_plus_6x_plus_9_l1967_196793

theorem sqrt_x_squared_plus_6x_plus_9 (x : ℝ) (h : x = Real.sqrt 5 - 3) :
  Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_plus_6x_plus_9_l1967_196793


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1967_196735

/-- Three points in ℝ³ are collinear if they lie on the same line. -/
def collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C - A = t • (B - A)

/-- The theorem states that if A(1,3,-2), B(2,5,1), and C(p,7,q-2) are collinear in ℝ³, 
    then p+q = 9. -/
theorem collinear_points_sum (p q : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 3, -2)
  let B : ℝ × ℝ × ℝ := (2, 5, 1)
  let C : ℝ × ℝ × ℝ := (p, 7, q-2)
  collinear A B C → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1967_196735


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l1967_196728

theorem intersection_sum_zero (α β : ℝ) : 
  (∃ x₀ : ℝ, 
    (x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1) ∧
    (x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1)) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l1967_196728


namespace NUMINAMATH_CALUDE_expression_simplification_l1967_196733

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 12) / 4) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1967_196733


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1967_196705

/-- Given a geometric sequence {a_n} with a_1 = 1 and a_5 = 1/9, 
    prove that the product a_2 * a_3 * a_4 = 1/27 -/
theorem geometric_sequence_product (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                                -- first term
  a 5 = 1 / 9 →                            -- fifth term
  a 2 * a 3 * a 4 = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1967_196705


namespace NUMINAMATH_CALUDE_complex_real_condition_l1967_196780

theorem complex_real_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((2 + Complex.I) * (1 - m * Complex.I)).im = 0 →
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1967_196780


namespace NUMINAMATH_CALUDE_ding_xiaole_jogging_distances_l1967_196715

/-- Represents the jogging distances for 4 days -/
structure JoggingData :=
  (days : Nat)
  (max_daily : ℝ)
  (min_daily : ℝ)

/-- Calculates the maximum total distance for the given jogging data -/
def max_total_distance (data : JoggingData) : ℝ :=
  data.max_daily * (data.days - 1) + data.min_daily

/-- Calculates the minimum total distance for the given jogging data -/
def min_total_distance (data : JoggingData) : ℝ :=
  data.min_daily * (data.days - 1) + data.max_daily

/-- Theorem stating the maximum and minimum total distances for Ding Xiaole's jogging -/
theorem ding_xiaole_jogging_distances :
  let data : JoggingData := ⟨4, 3.3, 2.4⟩
  max_total_distance data = 12.3 ∧ min_total_distance data = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ding_xiaole_jogging_distances_l1967_196715


namespace NUMINAMATH_CALUDE_trapezoid_area_difference_l1967_196786

/-- A trapezoid with specific properties -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  right_angles : ℕ
  
/-- The area difference between largest and smallest regions -/
def area_difference (t : Trapezoid) : ℝ := sorry

theorem trapezoid_area_difference :
  ∀ t : Trapezoid,
    t.side1 = 4 ∧ 
    t.side2 = 4 ∧ 
    t.side3 = 5 ∧ 
    t.side4 = Real.sqrt 17 ∧
    t.right_angles = 2 →
    240 * (area_difference t) = 240 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_difference_l1967_196786


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l1967_196794

-- Define the structure of a regular star polygon
structure RegularStarPolygon where
  n : ℕ  -- number of points
  A : ℝ  -- measure of each Aᵢ angle in degrees
  B : ℝ  -- measure of each Bᵢ angle in degrees

-- Define the properties of the regular star polygon
def is_valid_regular_star_polygon (p : RegularStarPolygon) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧  -- angles are positive
  p.A = p.B + 15 ∧     -- Aᵢ is 15° more than Bᵢ
  p.n * (p.A + p.B) = 360  -- sum of external angles is 360°

-- Theorem: A regular star polygon with the given conditions has 24 points
theorem regular_star_polygon_points (p : RegularStarPolygon) :
  is_valid_regular_star_polygon p → p.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l1967_196794


namespace NUMINAMATH_CALUDE_smaller_bucket_capacity_proof_l1967_196785

/-- The capacity of the smaller bucket in liters -/
def smaller_bucket_capacity : ℝ := 3

/-- The capacity of the medium bucket in liters -/
def medium_bucket_capacity : ℝ := 5

/-- The capacity of the larger bucket in liters -/
def larger_bucket_capacity : ℝ := 6

/-- The amount of water that can be added to the larger bucket after pouring from the medium bucket -/
def remaining_capacity : ℝ := 4

theorem smaller_bucket_capacity_proof :
  smaller_bucket_capacity = medium_bucket_capacity - (larger_bucket_capacity - remaining_capacity) :=
by sorry

end NUMINAMATH_CALUDE_smaller_bucket_capacity_proof_l1967_196785


namespace NUMINAMATH_CALUDE_tina_shoe_expense_l1967_196721

def savings_june : ℕ := 27
def savings_july : ℕ := 14
def savings_august : ℕ := 21
def spent_on_books : ℕ := 5
def amount_left : ℕ := 40

theorem tina_shoe_expense : 
  savings_june + savings_july + savings_august - spent_on_books - amount_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_tina_shoe_expense_l1967_196721


namespace NUMINAMATH_CALUDE_residue_of_7_500_mod_19_l1967_196739

theorem residue_of_7_500_mod_19 : 7^500 % 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_500_mod_19_l1967_196739


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1967_196743

/-- Represents a cricket team -/
structure CricketTeam where
  numPlayers : Nat
  captainAge : Nat
  captainBattingAvg : Nat
  wicketKeeperAge : Nat
  wicketKeeperBattingAvg : Nat
  youngestPlayerBattingAvg : Nat

/-- Calculate the average age of the team -/
def averageTeamAge (team : CricketTeam) : Rat :=
  sorry

theorem cricket_team_average_age 
  (team : CricketTeam)
  (h1 : team.numPlayers = 11)
  (h2 : team.captainAge = 25)
  (h3 : team.captainBattingAvg = 45)
  (h4 : team.wicketKeeperAge = team.captainAge + 5)
  (h5 : team.wicketKeeperBattingAvg = 35)
  (h6 : team.youngestPlayerBattingAvg = 42)
  (h7 : ∀ (remainingPlayersAvgAge : Rat),
        remainingPlayersAvgAge = (averageTeamAge team - 1) ∧
        (team.captainAge + team.wicketKeeperAge + remainingPlayersAvgAge * (team.numPlayers - 2)) / team.numPlayers = averageTeamAge team)
  (h8 : ∃ (youngestPlayerAge : Nat),
        youngestPlayerAge ≤ team.wicketKeeperAge - 15 ∧
        youngestPlayerAge > 0) :
  averageTeamAge team = 23 :=
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1967_196743


namespace NUMINAMATH_CALUDE_rectangle_length_is_16_l1967_196771

/-- Proves that the length of a rectangle is 16 cm given specific conditions --/
theorem rectangle_length_is_16 (b : ℝ) (c : ℝ) :
  b = 14 →
  c = 23.56 →
  ∃ (l : ℝ), l = 16 ∧ 
    2 * (l + b) = 4 * (c / π) ∧
    c / π = (2 * c) / (2 * π) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_is_16_l1967_196771


namespace NUMINAMATH_CALUDE_gcd_102_238_l1967_196776

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1967_196776


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l1967_196719

-- Define a function to represent angles with the same terminal side
def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

-- Define the main theorem
theorem angles_with_same_terminal_side :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 720 ∧ same_terminal_side θ (-1050) ↔ θ = 30 ∨ θ = 390 := by
  sorry


end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l1967_196719


namespace NUMINAMATH_CALUDE_decreasing_condition_direct_proportion_condition_l1967_196787

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (m^2 - 4)

-- Theorem for part 1
theorem decreasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function m x₁ > linear_function m x₂) ↔ m < 2 :=
sorry

-- Theorem for part 2
theorem direct_proportion_condition (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, linear_function m x = k * x) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_direct_proportion_condition_l1967_196787


namespace NUMINAMATH_CALUDE_find_a_value_l1967_196752

theorem find_a_value (a : ℕ) (h : a ^ 3 = 21 * 25 * 45 * 49) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1967_196752


namespace NUMINAMATH_CALUDE_meal_price_calculation_l1967_196751

theorem meal_price_calculation (beef_amount : ℝ) (pork_ratio : ℝ) (meat_per_meal : ℝ) (total_revenue : ℝ) :
  beef_amount = 20 →
  pork_ratio = 1 / 2 →
  meat_per_meal = 1.5 →
  total_revenue = 400 →
  (total_revenue / ((beef_amount + beef_amount * pork_ratio) / meat_per_meal) = 20) :=
by sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l1967_196751


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1967_196748

/-- Given a distance of 10000 meters and a time of 28 minutes, 
    prove that the average speed is approximately 595.24 cm/s. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) : 
  distance = 10000 ∧ time = 28 → 
  ∃ (speed : ℝ), abs (speed - 595.24) < 0.01 ∧ 
  speed = (distance * 100) / (time * 60) := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l1967_196748


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1967_196774

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 192 →
  Nat.gcd A B = 16 →
  A = 48 →
  B = 64 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1967_196774


namespace NUMINAMATH_CALUDE_tea_bags_count_l1967_196716

/-- Represents the number of tea bags in a box -/
def n : ℕ := sorry

/-- Represents the number of cups Natasha made -/
def natasha_cups : ℕ := 41

/-- Represents the number of cups Inna made -/
def inna_cups : ℕ := 58

/-- The number of cups made from Natasha's box is between 2n and 3n -/
axiom natasha_range : 2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n

/-- The number of cups made from Inna's box is between 2n and 3n -/
axiom inna_range : 2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n

/-- The number of tea bags in the box is 20 -/
theorem tea_bags_count : n = 20 := by sorry

end NUMINAMATH_CALUDE_tea_bags_count_l1967_196716


namespace NUMINAMATH_CALUDE_employees_with_advanced_degrees_l1967_196760

/-- Proves that the number of employees with advanced degrees is 78 given the conditions in the problem -/
theorem employees_with_advanced_degrees :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (male_college_only : ℕ) 
    (female_advanced : ℕ),
  total_employees = 148 →
  female_employees = 92 →
  male_college_only = 31 →
  female_advanced = 53 →
  ∃ (male_advanced : ℕ),
    male_advanced + female_advanced + male_college_only + (female_employees - female_advanced) = total_employees ∧
    male_advanced + female_advanced = 78 :=
by sorry

end NUMINAMATH_CALUDE_employees_with_advanced_degrees_l1967_196760


namespace NUMINAMATH_CALUDE_exactly_three_true_l1967_196709

theorem exactly_three_true : 
  (∀ x > 0, x > Real.sin x) ∧ 
  ((∀ x, x - Real.sin x = 0 → x = 0) ↔ (∀ x, x ≠ 0 → x - Real.sin x ≠ 0)) ∧ 
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧ 
  ¬(¬(∀ x : ℝ, x - Real.log x > 0) ↔ (∃ x : ℝ, x - Real.log x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_true_l1967_196709


namespace NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l1967_196773

/-- Given seven consecutive even integers whose sum is 448, 
    the smallest of these numbers is 58. -/
theorem smallest_of_seven_consecutive_evens (a : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ 
   (a + (a+2) + (a+4) + (a+6) + (a+8) + (a+10) + (a+12) = 448)) → 
  a = 58 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l1967_196773


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l1967_196768

theorem det_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 3]
  Matrix.det A = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l1967_196768


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l1967_196741

theorem sum_of_square_areas : 
  let square1_side : ℝ := 8
  let square2_side : ℝ := 10
  let square1_area : ℝ := square1_side * square1_side
  let square2_area : ℝ := square2_side * square2_side
  square1_area + square2_area = 164 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l1967_196741


namespace NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1967_196710

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 3*r*Real.sqrt 2 + 2 = 0 → 
  s^2 - 3*s*Real.sqrt 2 + 2 = 0 → 
  r^6 + s^6 = 2576 := by sorry

end NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1967_196710


namespace NUMINAMATH_CALUDE_johns_speed_l1967_196742

/-- Prove that John's speed during his final push was 4.2 m/s given the race conditions --/
theorem johns_speed (initial_gap : ℝ) (steve_speed : ℝ) (final_gap : ℝ) (push_duration : ℝ) : 
  initial_gap = 14 →
  steve_speed = 3.7 →
  final_gap = 2 →
  push_duration = 32 →
  (initial_gap + final_gap) / push_duration + steve_speed = 4.2 := by
sorry

end NUMINAMATH_CALUDE_johns_speed_l1967_196742


namespace NUMINAMATH_CALUDE_log_equality_l1967_196789

theorem log_equality (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 5 = (1/2) * (Real.log 2 / Real.log 5) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l1967_196789


namespace NUMINAMATH_CALUDE_min_sum_eccentricities_l1967_196792

theorem min_sum_eccentricities (e₁ e₂ : ℝ) (h₁ : e₁ > 0) (h₂ : e₂ > 0) 
  (h : 1 / (e₁^2) + 1 / (e₂^2) = 1) : 
  e₁ + e₂ ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_eccentricities_l1967_196792


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1967_196790

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1967_196790


namespace NUMINAMATH_CALUDE_right_triangle_area_l1967_196749

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1967_196749


namespace NUMINAMATH_CALUDE_max_leftover_oranges_l1967_196757

theorem max_leftover_oranges (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_oranges_l1967_196757


namespace NUMINAMATH_CALUDE_sector_area_l1967_196777

/-- Given a circular sector with a central angle of 2 radians and an arc length of 4 cm,
    the area of the sector is 4 cm². -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (h1 : θ = 2) (h2 : arc_length = 4) :
  (1/2) * arc_length * (arc_length / θ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1967_196777


namespace NUMINAMATH_CALUDE_sine_inequalities_l1967_196737

theorem sine_inequalities :
  (∀ x : ℝ, |Real.sin (2 * x)| ≤ 2 * |Real.sin x|) ∧
  (∀ n : ℕ, n > 0 → ∀ x : ℝ, |Real.sin (n * x)| ≤ n * |Real.sin x|) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequalities_l1967_196737


namespace NUMINAMATH_CALUDE_min_points_on_circle_l1967_196714

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for circles in a plane
def Circle : Type := Point × ℝ

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to count points on a circle
def countPointsOnCircle (points : List Point) (c : Circle) : Nat := sorry

-- Main theorem
theorem min_points_on_circle 
  (points : List Point) 
  (h1 : points.length = 10)
  (h2 : ∀ (sublist : List Point), sublist ⊆ points → sublist.length = 5 → 
        ∃ (c : Circle), (countPointsOnCircle sublist c) ≥ 4) :
  ∃ (c : Circle), (countPointsOnCircle points c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l1967_196714


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1967_196778

theorem complex_cube_sum_ratio (x y z : ℂ) 
  (hnonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (hsum : x + y + z = 30)
  (hdiff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1967_196778


namespace NUMINAMATH_CALUDE_composition_equality_l1967_196784

theorem composition_equality (a : ℝ) (h1 : a > 1) : 
  let f (x : ℝ) := x^2 + 2
  let g (x : ℝ) := x^2 + 2
  f (g a) = 12 → a = Real.sqrt (Real.sqrt 10 - 2) := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l1967_196784


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l1967_196712

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l1967_196712


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_solution_l1967_196763

theorem cryptarithmetic_puzzle_solution :
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A * B + C * D = 10 * E + F ∧
    B + C + D ≠ A ∧
    A = 2 * D ∧
    F = 8 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_solution_l1967_196763


namespace NUMINAMATH_CALUDE_expression_equality_l1967_196750

theorem expression_equality (x : ℝ) : x * (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2*x^3 - 4*x^2 + 10*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1967_196750


namespace NUMINAMATH_CALUDE_ellipse_constants_l1967_196761

/-- An ellipse with foci at (1, 1) and (1, 5) passing through (12, -4) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, 5)
  point : ℝ × ℝ := (12, -4)

/-- The standard form of an ellipse equation -/
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating the constants of the ellipse -/
theorem ellipse_constants (e : Ellipse) :
  ∃ (a b h k : ℝ),
    a > 0 ∧ b > 0 ∧
    a = 13 ∧ b = Real.sqrt 153 ∧ h = 1 ∧ k = 3 ∧
    standard_form a b h k e.point.1 e.point.2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_constants_l1967_196761


namespace NUMINAMATH_CALUDE_find_m_value_l1967_196729

theorem find_m_value (x y m : ℝ) 
  (eq1 : 3 * x + 7 * y = 5 * m - 3)
  (eq2 : 2 * x + 3 * y = 8)
  (eq3 : x + 2 * y = 5) : 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l1967_196729


namespace NUMINAMATH_CALUDE_factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l1967_196782

-- Factorization of 2x^2 - 8
theorem factorization_2x2_minus_8 (x : ℝ) :
  2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by sorry

-- Factorization of ax^2 - 2ax + a
theorem factorization_ax2_minus_2ax_plus_a (x a : ℝ) (ha : a ≠ 0) :
  a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_2x2_minus_8_factorization_ax2_minus_2ax_plus_a_l1967_196782


namespace NUMINAMATH_CALUDE_correct_answers_for_86_points_min_correct_for_first_prize_l1967_196762

/-- Represents a math competition with given parameters -/
structure MathCompetition where
  total_questions : ℕ
  full_score : ℕ
  correct_points : ℕ
  wrong_points : ℤ
  unanswered_points : ℕ

/-- Theorem for part (1) of the problem -/
theorem correct_answers_for_86_points (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0)
    (h6 : ∃ (x : ℕ), x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86) :
    ∃ (x : ℕ), x = 22 ∧ x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86 :=
  sorry

/-- Theorem for part (2) of the problem -/
theorem min_correct_for_first_prize (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0) :
    ∃ (x : ℕ), x ≥ 23 ∧ ∀ (y : ℕ), y * comp.correct_points + (comp.total_questions - y) * comp.wrong_points ≥ 90 → y ≥ x :=
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_86_points_min_correct_for_first_prize_l1967_196762


namespace NUMINAMATH_CALUDE_total_food_consumption_theorem_l1967_196783

/-- The total amount of food consumed daily by both sides in a war --/
def total_food_consumption (first_side_soldiers : ℕ) (food_per_soldier_first : ℕ) 
  (soldier_difference : ℕ) (food_difference : ℕ) : ℕ :=
  let second_side_soldiers := first_side_soldiers - soldier_difference
  let food_per_soldier_second := food_per_soldier_first - food_difference
  (first_side_soldiers * food_per_soldier_first) + 
  (second_side_soldiers * food_per_soldier_second)

/-- Theorem stating the total food consumption for both sides --/
theorem total_food_consumption_theorem :
  total_food_consumption 4000 10 500 2 = 68000 := by
  sorry

end NUMINAMATH_CALUDE_total_food_consumption_theorem_l1967_196783


namespace NUMINAMATH_CALUDE_log_product_equals_24_l1967_196734

theorem log_product_equals_24 :
  Real.log 9 / Real.log 2 * (Real.log 16 / Real.log 3) * (Real.log 27 / Real.log 7) = 24 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_24_l1967_196734


namespace NUMINAMATH_CALUDE_triangle_max_area_l1967_196707

open Real

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + c = 6 →
  (3 - cos A) * tan (B / 2) = sin A →
  ∃ (S : ℝ), S ≤ 2 * sqrt 2 ∧
    ∀ (S' : ℝ), S' = (1 / 2) * a * c * sin B → S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1967_196707


namespace NUMINAMATH_CALUDE_fraction_conforms_to_standard_notation_l1967_196713

/-- Rules for standard algebraic notation -/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  mixed_numbers_as_fractions : Bool
  division_as_fraction : Bool

/-- An algebraic expression -/
inductive AlgebraicExpression
  | Multiply : ℕ → Char → AlgebraicExpression
  | MixedNumber : ℕ → ℚ → Char → AlgebraicExpression
  | Fraction : Char → Char → ℕ → AlgebraicExpression
  | Divide : Char → ℕ → Char → AlgebraicExpression

/-- Function to check if an expression conforms to standard algebraic notation -/
def conforms_to_standard_notation (rules : AlgebraicNotationRules) (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.Fraction _ _ _ => true
  | _ => false

/-- Theorem stating that -b/a² conforms to standard algebraic notation -/
theorem fraction_conforms_to_standard_notation (rules : AlgebraicNotationRules) :
  conforms_to_standard_notation rules (AlgebraicExpression.Fraction 'b' 'a' 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_conforms_to_standard_notation_l1967_196713


namespace NUMINAMATH_CALUDE_nancy_balloon_count_l1967_196755

/-- Given that Mary has 7 balloons and Nancy has 4 times as many balloons as Mary,
    prove that Nancy has 28 balloons. -/
theorem nancy_balloon_count :
  ∀ (mary_balloons nancy_balloons : ℕ),
    mary_balloons = 7 →
    nancy_balloons = 4 * mary_balloons →
    nancy_balloons = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_nancy_balloon_count_l1967_196755


namespace NUMINAMATH_CALUDE_square_difference_equals_double_product_problem_instance_l1967_196720

theorem square_difference_equals_double_product (a b : ℕ) :
  (a + b)^2 - (a^2 + b^2) = 2 * a * b :=
by sorry

-- Specific instance for the given problem
theorem problem_instance : (25 + 15)^2 - (25^2 + 15^2) = 750 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_equals_double_product_problem_instance_l1967_196720


namespace NUMINAMATH_CALUDE_strip_length_is_four_l1967_196797

/-- The length of each square in the strip -/
def square_length : ℚ := 2/3

/-- The number of squares in the strip -/
def num_squares : ℕ := 6

/-- The total length of the strip -/
def strip_length : ℚ := square_length * num_squares

/-- Theorem: The strip composed of 6 squares, each with length 2/3, has a total length of 4 -/
theorem strip_length_is_four : strip_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_strip_length_is_four_l1967_196797


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1967_196744

theorem ice_cream_flavors (n : ℕ) (k : ℕ) : 
  n = 4 → k = 5 → (n + k - 1).choose k = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1967_196744


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l1967_196732

/-- A monic polynomial of degree 2 -/
def MonicQuadratic (g : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c

theorem unique_monic_quadratic (g : ℝ → ℝ) 
  (h_monic : MonicQuadratic g) 
  (h_g0 : g 0 = 2) 
  (h_g1 : g 1 = 6) : 
  ∀ x, g x = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_l1967_196732


namespace NUMINAMATH_CALUDE_bingo_last_column_permutations_l1967_196711

/-- The number of elements in the set to choose from -/
def n : ℕ := 10

/-- The number of elements to be chosen and arranged -/
def r : ℕ := 5

/-- The function to calculate the number of permutations -/
def permutations (n r : ℕ) : ℕ := (n - r + 1).factorial / (n - r).factorial

theorem bingo_last_column_permutations :
  permutations n r = 30240 := by sorry

end NUMINAMATH_CALUDE_bingo_last_column_permutations_l1967_196711


namespace NUMINAMATH_CALUDE_three_hundred_thousand_squared_minus_million_l1967_196765

theorem three_hundred_thousand_squared_minus_million : (300000 * 300000) - 1000000 = 89990000000 := by
  sorry

end NUMINAMATH_CALUDE_three_hundred_thousand_squared_minus_million_l1967_196765


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1967_196799

/-- Represents a shape created by joining seven unit cubes -/
structure SevenCubeShape where
  /-- The volume of the shape in cubic units -/
  volume : ℕ
  /-- The surface area of the shape in square units -/
  surface_area : ℕ
  /-- The shape is composed of seven unit cubes -/
  is_seven_cubes : volume = 7
  /-- The surface area is calculated based on the configuration of the seven cubes -/
  surface_area_calc : surface_area = 30

/-- Theorem stating that the ratio of volume to surface area for the SevenCubeShape is 7:30 -/
theorem volume_to_surface_area_ratio (shape : SevenCubeShape) :
  (shape.volume : ℚ) / shape.surface_area = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1967_196799


namespace NUMINAMATH_CALUDE_total_candy_is_54_l1967_196727

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of chocolate pieces given to each student -/
def chocolate_per_student : ℕ := 2

/-- The number of hard candy pieces given to each student -/
def hard_candy_per_student : ℕ := 3

/-- The number of gummy candy pieces given to each student -/
def gummy_per_student : ℕ := 1

/-- The total number of candy pieces given away -/
def total_candy : ℕ := num_students * (chocolate_per_student + hard_candy_per_student + gummy_per_student)

theorem total_candy_is_54 : total_candy = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_is_54_l1967_196727


namespace NUMINAMATH_CALUDE_assembly_line_arrangements_l1967_196740

theorem assembly_line_arrangements (n : ℕ) (arrangements : ℕ) 
  (h1 : n = 6) 
  (h2 : arrangements = 360) :
  arrangements = n.factorial / 2 := by
sorry

end NUMINAMATH_CALUDE_assembly_line_arrangements_l1967_196740


namespace NUMINAMATH_CALUDE_train_length_problem_l1967_196796

/-- Represents the problem of calculating the length of a train based on James's jogging --/
theorem train_length_problem (james_speed : ℝ) (train_speed : ℝ) (steps_forward : ℕ) (steps_backward : ℕ) :
  james_speed > train_speed →
  steps_forward = 400 →
  steps_backward = 160 →
  let train_length := (steps_forward * james_speed - steps_forward * train_speed + 
                       steps_backward * james_speed + steps_backward * train_speed) / 2
  train_length = 640 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l1967_196796


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_range_l1967_196700

theorem parabola_axis_of_symmetry_range 
  (a b c m n t : ℝ) 
  (h_a_pos : a > 0)
  (h_point1 : m = a + b + c)
  (h_point2 : n = 9*a + 3*b + c)
  (h_order : m < n ∧ n < c)
  (h_axis : t = -b / (2*a)) : 
  3/2 < t ∧ t < 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_range_l1967_196700


namespace NUMINAMATH_CALUDE_strawberry_cost_l1967_196753

/-- The cost function for strawberry picking -/
def cost_function (x : ℝ) : ℝ := 16 * x + 2.5

/-- Theorem: The cost for 5.5 kg of strawberries is 90.5 元 -/
theorem strawberry_cost : cost_function 5.5 = 90.5 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cost_l1967_196753
