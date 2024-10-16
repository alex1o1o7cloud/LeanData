import Mathlib

namespace NUMINAMATH_CALUDE_percentage_increase_l3421_342112

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  (x - 70) / 70 * 100 = 11 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3421_342112


namespace NUMINAMATH_CALUDE_restaurant_ratio_proof_l3421_342159

/-- Proves that the original ratio of cooks to waiters was 1:3 given the conditions -/
theorem restaurant_ratio_proof (cooks : ℕ) (waiters : ℕ) :
  cooks = 9 →
  (cooks : ℚ) / (waiters + 12 : ℚ) = 1 / 5 →
  (cooks : ℚ) / (waiters : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_ratio_proof_l3421_342159


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3421_342121

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3421_342121


namespace NUMINAMATH_CALUDE_total_yellow_balloons_l3421_342170

/-- The total number of yellow balloons given the number of balloons each person has -/
def total_balloons (fred_balloons sam_balloons mary_balloons : ℕ) : ℕ :=
  fred_balloons + sam_balloons + mary_balloons

/-- Theorem stating that the total number of yellow balloons is 18 -/
theorem total_yellow_balloons :
  total_balloons 5 6 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_balloons_l3421_342170


namespace NUMINAMATH_CALUDE_perp_line_plane_condition_l3421_342104

/-- A straight line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines the perpendicular relationship between a line and another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines the perpendicular relationship between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem stating that "m ⊥ n" is a necessary but not sufficient condition for "m ⊥ α" -/
theorem perp_line_plane_condition (m n : Line3D) (α : Plane3D) 
  (h : line_in_plane n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ¬(perpendicular_lines m n → perpendicular_line_plane m α) :=
sorry

end NUMINAMATH_CALUDE_perp_line_plane_condition_l3421_342104


namespace NUMINAMATH_CALUDE_probability_prime_sum_three_dice_l3421_342119

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The set of possible prime sums when rolling three 6-sided dice -/
def primeSums : Set ℕ := {3, 5, 7, 11, 13, 17}

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := numSides ^ 3

/-- The number of ways to roll a prime sum with three 6-sided dice -/
def primeOutcomes : ℕ := 58

/-- The probability of rolling a prime sum with three 6-sided dice -/
theorem probability_prime_sum_three_dice :
  (primeOutcomes : ℚ) / totalOutcomes = 58 / 216 := by
  sorry


end NUMINAMATH_CALUDE_probability_prime_sum_three_dice_l3421_342119


namespace NUMINAMATH_CALUDE_distance_is_100_miles_l3421_342142

/-- Represents the fuel efficiency of a car in miles per gallon. -/
def miles_per_gallon : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons. -/
def gallons_needed : ℝ := 5

/-- Calculates the distance to Grandma's house in miles. -/
def distance_to_grandma : ℝ := miles_per_gallon * gallons_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles. -/
theorem distance_is_100_miles : distance_to_grandma = 100 :=
  sorry

end NUMINAMATH_CALUDE_distance_is_100_miles_l3421_342142


namespace NUMINAMATH_CALUDE_cat_whiskers_count_l3421_342139

/-- The number of whiskers on Princess Puff's face -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers on Catman Do's face -/
def catman_do_whiskers : ℕ := 2 * princess_puff_whiskers - 6

/-- The number of whiskers on Sir Whiskerson's face -/
def sir_whiskerson_whiskers : ℕ := princess_puff_whiskers + catman_do_whiskers + 8

/-- Theorem stating the correct number of whiskers for each cat -/
theorem cat_whiskers_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  sir_whiskerson_whiskers = 44 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_count_l3421_342139


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3421_342105

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3421_342105


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3421_342141

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 24 →
  group2_students = 6 →
  group1_mean = 80 / 100 →
  group2_mean = 85 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 81 / 100 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l3421_342141


namespace NUMINAMATH_CALUDE_smallest_x_value_l3421_342115

theorem smallest_x_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  (∀ z : ℤ, ∃ w : ℤ, z * w + 7 * z + 6 * w = -8 → z ≥ -40) ∧ 
  (∃ w : ℤ, -40 * w + 7 * (-40) + 6 * w = -8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3421_342115


namespace NUMINAMATH_CALUDE_always_two_real_roots_find_m_value_l3421_342126

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m+1)*x + (3*m-6)

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: Given the condition, m = 3
theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁ = 0) 
  (h₂ : quadratic m x₂ = 0)
  (h₃ : x₁ + x₂ + x₁*x₂ = 7) : 
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_find_m_value_l3421_342126


namespace NUMINAMATH_CALUDE_set_membership_solution_l3421_342163

theorem set_membership_solution (x : ℝ) :
  let A : Set ℝ := {2, x, x^2 + x}
  6 ∈ A → x = 6 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_set_membership_solution_l3421_342163


namespace NUMINAMATH_CALUDE_cannot_be_square_difference_l3421_342131

/-- The square difference formula -/
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

/-- The expression that we want to prove cannot be computed using the square difference formula -/
def expression (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference :
  ¬∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end NUMINAMATH_CALUDE_cannot_be_square_difference_l3421_342131


namespace NUMINAMATH_CALUDE_fibCoeff_symmetry_l3421_342169

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Fibonacci coefficient -/
def fibCoeff (n k : ℕ) : ℚ :=
  if k ≤ n then
    (List.range k).foldl (λ acc i => acc * fib (n - i)) 1 /
    (List.range k).foldl (λ acc i => acc * fib (k - i)) 1
  else 0

/-- Symmetry property of Fibonacci coefficients -/
theorem fibCoeff_symmetry (n k : ℕ) (h : k ≤ n) :
  fibCoeff n k = fibCoeff n (n - k) := by
  sorry

end NUMINAMATH_CALUDE_fibCoeff_symmetry_l3421_342169


namespace NUMINAMATH_CALUDE_problem_solution_l3421_342124

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3421_342124


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3421_342186

theorem smallest_positive_integer_congruence : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (5 * x ≡ 18 [MOD 33]) ∧ 
  (x ≡ 4 [MOD 7]) ∧ 
  (∀ (y : ℕ), y > 0 → (5 * y ≡ 18 [MOD 33]) → (y ≡ 4 [MOD 7]) → x ≤ y) ∧
  x = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3421_342186


namespace NUMINAMATH_CALUDE_division_problem_l3421_342133

theorem division_problem (total : ℚ) (a b c : ℚ) : 
  total = 527 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  c = 372 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l3421_342133


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3421_342178

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ (3 * x : ℤ) - 5 < 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3421_342178


namespace NUMINAMATH_CALUDE_fraction_power_four_l3421_342173

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l3421_342173


namespace NUMINAMATH_CALUDE_distinct_roots_sum_of_squares_l3421_342109

theorem distinct_roots_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_of_squares_l3421_342109


namespace NUMINAMATH_CALUDE_circle_tangent_and_point_condition_l3421_342185

-- Define the given points and lines
def point_A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that the center of C is on line l
def center_on_line_l (C : Circle) : Prop :=
  C.center.2 = line_l C.center.1

-- Define the condition that the center of C is on y = x - 1
def center_on_diagonal (C : Circle) : Prop :=
  C.center.2 = C.center.1 - 1

-- Define the tangent line
def is_tangent_line (k b : ℝ) (C : Circle) : Prop :=
  let (cx, cy) := C.center
  (k * cx - cy + b)^2 = (k^2 + 1) * C.radius^2

-- Define the condition |MA| = 2|MO|
def condition_MA_MO (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx^2 + (my - 3)^2) = 4 * (mx^2 + my^2)

-- Main theorem
theorem circle_tangent_and_point_condition (C : Circle) :
  C.radius = 1 →
  center_on_line_l C →
  (center_on_diagonal C →
    (∃ k b, is_tangent_line k b C ∧ (k = 0 ∨ (k = -3/4 ∧ b = 3)))) ∧
  (∃ M, condition_MA_MO M → 
    C.center.1 ≥ 0 ∧ C.center.1 ≤ 12/5) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_and_point_condition_l3421_342185


namespace NUMINAMATH_CALUDE_profit_starts_in_fourth_year_option_one_more_profitable_l3421_342108

/-- Represents the financial data for the real estate investment --/
structure RealEstateInvestment where
  initialInvestment : ℝ
  firstYearRenovationCost : ℝ
  yearlyRenovationIncrease : ℝ
  annualRentalIncome : ℝ

/-- Calculates the renovation cost for a given year --/
def renovationCost (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  investment.firstYearRenovationCost + investment.yearlyRenovationIncrease * (year - 1)

/-- Calculates the cumulative profit up to a given year --/
def cumulativeProfit (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  year * investment.annualRentalIncome - investment.initialInvestment - 
    (Finset.range year).sum (fun i => renovationCost investment (i + 1))

/-- Theorem stating that the developer starts making a net profit in the 4th year --/
theorem profit_starts_in_fourth_year (investment : RealEstateInvestment) 
  (h1 : investment.initialInvestment = 810000)
  (h2 : investment.firstYearRenovationCost = 10000)
  (h3 : investment.yearlyRenovationIncrease = 20000)
  (h4 : investment.annualRentalIncome = 300000) :
  (cumulativeProfit investment 3 < 0) ∧ (cumulativeProfit investment 4 > 0) := by
  sorry

/-- Represents the two selling options --/
inductive SellingOption
  | OptionOne : SellingOption
  | OptionTwo : SellingOption

/-- Calculates the profit for a given selling option --/
def profitForOption (investment : RealEstateInvestment) (option : SellingOption) : ℝ :=
  match option with
  | SellingOption.OptionOne => 460000 -- Simplified for the sake of the statement
  | SellingOption.OptionTwo => 100000 -- Simplified for the sake of the statement

/-- Theorem stating that Option 1 is more profitable --/
theorem option_one_more_profitable (investment : RealEstateInvestment) :
  profitForOption investment SellingOption.OptionOne > profitForOption investment SellingOption.OptionTwo := by
  sorry

end NUMINAMATH_CALUDE_profit_starts_in_fourth_year_option_one_more_profitable_l3421_342108


namespace NUMINAMATH_CALUDE_min_cubes_needed_l3421_342195

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := ℕ → ℕ → ℕ → Bool

/-- Checks if a cube is present at the given coordinates -/
def has_cube (grid : CubeGrid) (x y z : ℕ) : Prop := grid x y z = true

/-- Checks if the grid satisfies the adjacency condition -/
def satisfies_adjacency (grid : CubeGrid) : Prop :=
  ∀ x y z, has_cube grid x y z →
    (has_cube grid (x+1) y z ∨ has_cube grid (x-1) y z ∨
     has_cube grid x (y+1) z ∨ has_cube grid x (y-1) z ∨
     has_cube grid x y (z+1) ∨ has_cube grid x y (z-1))

/-- Checks if the grid matches the given front view -/
def matches_front_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 0 1 0) ∧ (has_cube grid 0 2 0) ∧
  (has_cube grid 1 0 0) ∧ (has_cube grid 1 1 0) ∧
  (has_cube grid 2 0 0) ∧ (has_cube grid 2 1 0)

/-- Checks if the grid matches the given side view -/
def matches_side_view (grid : CubeGrid) : Prop :=
  (has_cube grid 0 0 0) ∧ (has_cube grid 1 0 0) ∧ (has_cube grid 2 0 0) ∧
  (has_cube grid 2 0 1) ∧
  (has_cube grid 2 0 2)

/-- Counts the number of cubes in the grid -/
def count_cubes (grid : CubeGrid) : ℕ :=
  sorry -- Implementation omitted

/-- The main theorem to be proved -/
theorem min_cubes_needed :
  ∃ (grid : CubeGrid),
    satisfies_adjacency grid ∧
    matches_front_view grid ∧
    matches_side_view grid ∧
    count_cubes grid = 5 ∧
    (∀ (other_grid : CubeGrid),
      satisfies_adjacency other_grid →
      matches_front_view other_grid →
      matches_side_view other_grid →
      count_cubes other_grid ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_needed_l3421_342195


namespace NUMINAMATH_CALUDE_min_people_with_both_hat_and_glove_l3421_342147

theorem min_people_with_both_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = n / 3 →
  hats = 2 * n / 3 →
  gloves + hats - both = n →
  both ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_people_with_both_hat_and_glove_l3421_342147


namespace NUMINAMATH_CALUDE_systematic_sample_validity_l3421_342160

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  (∀ i j, i < j → i < sample.length → j < sample.length → 
    sample[i]! < sample[j]! ∧ 
    (sample[j]! - sample[i]!) = (population_size / sample_size) * (j - i)) ∧
  (∀ n, n ∈ sample → n < population_size)

theorem systematic_sample_validity :
  is_valid_systematic_sample [1, 11, 21, 31, 41] 50 5 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_validity_l3421_342160


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l3421_342181

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def tens_digit_less_than_5 (n : ℕ) : Prop := (n / 10) % 10 < 5

def divisible_by_its_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0 ∧
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧
  n % hundreds = 0 ∧ n % tens = 0 ∧ n % ones = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, is_three_digit n →
    tens_digit_less_than_5 n →
    divisible_by_its_digits n →
    n ≤ 936 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l3421_342181


namespace NUMINAMATH_CALUDE_cubic_polynomial_real_root_l3421_342144

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    and -2 - 3i is a root of this polynomial, the real root of the polynomial is 5/2. -/
theorem cubic_polynomial_real_root (a b : ℝ) : 
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 125 = 0 ∧ x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_real_root_l3421_342144


namespace NUMINAMATH_CALUDE_conic_section_type_l3421_342180

/-- The equation √((x-2)² + y²) + √((x+2)² + y²) = 12 represents an ellipse -/
theorem conic_section_type : ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
  {(x, y) : ℝ × ℝ | Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 12} =
  {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} :=
by sorry

end NUMINAMATH_CALUDE_conic_section_type_l3421_342180


namespace NUMINAMATH_CALUDE_student_A_wrong_l3421_342164

-- Define the circle
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the points
def point_D : ℝ × ℝ := (5, 1)
def point_A : ℝ × ℝ := (-2, -1)

-- Function to check if a point is on the circle
def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Theorem statement
theorem student_A_wrong :
  is_on_circle point_D ∧ ¬is_on_circle point_A :=
sorry

end NUMINAMATH_CALUDE_student_A_wrong_l3421_342164


namespace NUMINAMATH_CALUDE_student_count_l3421_342102

theorem student_count (avg_student_age avg_with_teacher teacher_age : ℝ) 
  (h1 : avg_student_age = 15)
  (h2 : avg_with_teacher = 16)
  (h3 : teacher_age = 46) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_student_age + teacher_age = (n + 1 : ℝ) * avg_with_teacher ∧
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3421_342102


namespace NUMINAMATH_CALUDE_triangle_area_increase_l3421_342177

theorem triangle_area_increase (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let original_area := (1/2) * a * b * Real.sin θ
  let new_area := (1/2) * (3*a) * (2*b) * Real.sin θ
  new_area = 6 * original_area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_increase_l3421_342177


namespace NUMINAMATH_CALUDE_f_properties_l3421_342162

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), 1 ≤ f x ∧ f x ≤ 2) ∧
  (f 0 = 1) ∧
  (f (Real.pi / 6) = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3421_342162


namespace NUMINAMATH_CALUDE_expected_rainfall_l3421_342103

/-- The expected value of total rainfall over 7 days given specific weather conditions --/
theorem expected_rainfall (p_sunny p_light p_heavy : ℝ) (r_light r_heavy : ℝ) (days : ℕ) : 
  p_sunny + p_light + p_heavy = 1 →
  p_sunny = 0.3 →
  p_light = 0.4 →
  p_heavy = 0.3 →
  r_light = 3 →
  r_heavy = 6 →
  days = 7 →
  days * (p_sunny * 0 + p_light * r_light + p_heavy * r_heavy) = 21 :=
by sorry

end NUMINAMATH_CALUDE_expected_rainfall_l3421_342103


namespace NUMINAMATH_CALUDE_derivative_x_squared_at_one_l3421_342137

theorem derivative_x_squared_at_one :
  let f : ℝ → ℝ := fun x ↦ x^2
  deriv f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_x_squared_at_one_l3421_342137


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l3421_342132

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 15 * y) / (y + 15 * x) = 3) : 
  x / y = 0.8 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l3421_342132


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3421_342174

theorem sum_of_roots_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a ≠ b) →
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3421_342174


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3421_342154

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) :
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boy_ratio * girls = girl_ratio * boys ∧
    girls - boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3421_342154


namespace NUMINAMATH_CALUDE_root_equation_value_l3421_342187

theorem root_equation_value (m : ℝ) : m^2 + m - 1 = 0 → 2023 - m^2 - m = 2022 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3421_342187


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3421_342196

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3421_342196


namespace NUMINAMATH_CALUDE_product_plus_number_equals_result_l3421_342167

theorem product_plus_number_equals_result : ∃ x : ℝ,
  12.05 * 5.4 + x = 108.45000000000003 ∧ x = 43.38000000000003 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_number_equals_result_l3421_342167


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_2014000000_l3421_342155

theorem fifth_largest_divisor_of_2014000000 :
  ∃ (d : ℕ), d ∣ 2014000000 ∧
  (∀ (x : ℕ), x ∣ 2014000000 → x ≠ 2014000000 → x ≠ 1007000000 → x ≠ 503500000 → x ≠ 251750000 → x ≤ d) ∧
  d = 125875000 :=
by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_2014000000_l3421_342155


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l3421_342135

theorem marble_ratio_proof (initial_red : ℕ) (initial_blue : ℕ) (red_taken : ℕ) (total_left : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  total_left = 35 →
  ∃ (blue_taken : ℕ),
    blue_taken * red_taken = 4 * red_taken ∧
    initial_red + initial_blue = total_left + red_taken + blue_taken :=
by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l3421_342135


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3421_342116

theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3421_342116


namespace NUMINAMATH_CALUDE_dice_divisible_by_seven_l3421_342188

/-- A die is represented as a function from face index to digit -/
def Die := Fin 6 → Fin 6

/-- Property that opposite faces of a die sum to 7 -/
def OppositeFacesSum7 (d : Die) : Prop :=
  ∀ i : Fin 3, d i + d (i + 3) = 7

/-- A set of six dice -/
def DiceSet := Fin 6 → Die

/-- Property that all dice in a set have opposite faces summing to 7 -/
def AllDiceOppositeFacesSum7 (ds : DiceSet) : Prop :=
  ∀ i : Fin 6, OppositeFacesSum7 (ds i)

/-- A configuration of dice is a function from die position to face index -/
def DiceConfiguration := Fin 6 → Fin 6

/-- The number formed by a dice configuration -/
def NumberFormed (ds : DiceSet) (dc : DiceConfiguration) : ℕ :=
  (ds 0 (dc 0)) * 100000 + (ds 1 (dc 1)) * 10000 + (ds 2 (dc 2)) * 1000 +
  (ds 3 (dc 3)) * 100 + (ds 4 (dc 4)) * 10 + (ds 5 (dc 5))

theorem dice_divisible_by_seven (ds : DiceSet) (h : AllDiceOppositeFacesSum7 ds) :
  ∃ dc : DiceConfiguration, NumberFormed ds dc % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_dice_divisible_by_seven_l3421_342188


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3421_342198

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  ∃ r : ℝ, r ≠ 0 ∧ b = 10 * r ∧ (3/4) = b * r → 
  b = 5 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3421_342198


namespace NUMINAMATH_CALUDE_seven_digit_sum_theorem_l3421_342122

theorem seven_digit_sum_theorem :
  ∀ a b : ℕ,
  (a ≤ 9 ∧ a > 0) →
  (b ≤ 9) →
  (7 * a = 10 * a + b) →
  (a + b = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_seven_digit_sum_theorem_l3421_342122


namespace NUMINAMATH_CALUDE_bonus_pool_ratio_l3421_342176

theorem bonus_pool_ratio (P : ℕ) (k : ℕ) (h1 : P % 5 = 2) (h2 : (k * P) % 5 = 1) :
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_bonus_pool_ratio_l3421_342176


namespace NUMINAMATH_CALUDE_grain_equations_correct_l3421_342184

/-- Represents the amount of grain in sheng that one bundle can produce -/
structure GrainBundle where
  amount : ℝ

/-- High-quality grain bundle -/
def high_quality : GrainBundle := sorry

/-- Low-quality grain bundle -/
def low_quality : GrainBundle := sorry

/-- Theorem stating that the system of equations correctly represents the grain problem -/
theorem grain_equations_correct :
  (5 * high_quality.amount - 11 = 7 * low_quality.amount) ∧
  (7 * high_quality.amount - 25 = 5 * low_quality.amount) := by
  sorry

end NUMINAMATH_CALUDE_grain_equations_correct_l3421_342184


namespace NUMINAMATH_CALUDE_curve_composition_l3421_342100

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (-y^2 + 2*y + 8)) * Real.sqrt (x - y) = 0

-- Define the line segment
def line_segment (x y : ℝ) : Prop :=
  x = y ∧ -2 ≤ y ∧ y ≤ 4

-- Define the minor arc
def minor_arc (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 9 ∧ x ≥ 0

-- Theorem stating that the curve consists of a line segment and a minor arc
theorem curve_composition :
  ∀ x y : ℝ, curve_equation x y ↔ (line_segment x y ∨ minor_arc x y) :=
sorry

end NUMINAMATH_CALUDE_curve_composition_l3421_342100


namespace NUMINAMATH_CALUDE_certain_fraction_problem_l3421_342125

theorem certain_fraction_problem (a b x y : ℚ) : 
  (a / b) / (2 / 5) = (3 / 8) / (x / y) →
  a / b = 3 / 4 →
  x / y = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_problem_l3421_342125


namespace NUMINAMATH_CALUDE_smallest_m_for_positive_integer_solutions_l3421_342192

theorem smallest_m_for_positive_integer_solutions :
  ∃ (m : ℤ), m = -1 ∧
  (∀ k : ℤ, k < m →
    ¬∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*k + 7 ∧ x - 2*y = 4*k - 3) ∧
  (∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*m + 7 ∧ x - 2*y = 4*m - 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_positive_integer_solutions_l3421_342192


namespace NUMINAMATH_CALUDE_combination_square_numbers_examples_find_m_l3421_342138

def is_combination_square_numbers (a b c : Int) : Prop :=
  a < 0 ∧ b < 0 ∧ c < 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y z : Int), x^2 = a * b ∧ y^2 = b * c ∧ z^2 = a * c

theorem combination_square_numbers_examples :
  (is_combination_square_numbers (-4) (-16) (-25)) ∧
  (is_combination_square_numbers (-3) (-48) (-12)) ∧
  (is_combination_square_numbers (-2) (-18) (-72)) := by sorry

theorem find_m :
  ∀ m : Int, is_combination_square_numbers (-3) m (-12) ∧ 
  (∃ (x : Int), x^2 = -3 * m ∨ x^2 = m * (-12) ∨ x^2 = -3 * (-12)) ∧
  x = 12 → m = -48 := by sorry

end NUMINAMATH_CALUDE_combination_square_numbers_examples_find_m_l3421_342138


namespace NUMINAMATH_CALUDE_celine_erasers_l3421_342143

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ

/-- The conditions of the eraser collection problem -/
def EraserProblem (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica = 151

theorem celine_erasers (ec : EraserCollection) (h : EraserProblem ec) : ec.celine = 16 := by
  sorry

end NUMINAMATH_CALUDE_celine_erasers_l3421_342143


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3421_342146

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3421_342146


namespace NUMINAMATH_CALUDE_maryann_rescue_time_l3421_342166

/-- The time (in minutes) it takes Maryann to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℕ := 6

/-- The time (in minutes) it takes Maryann to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℕ := 8

/-- The number of friends Maryann needs to rescue -/
def number_of_friends : ℕ := 3

/-- The time it takes to free one friend -/
def time_per_friend : ℕ := cheap_handcuff_time + expensive_handcuff_time

/-- The total time it takes to free all friends -/
def total_rescue_time : ℕ := time_per_friend * number_of_friends

theorem maryann_rescue_time : total_rescue_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_maryann_rescue_time_l3421_342166


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_with_tangent_chord_l3421_342191

/-- The area between two concentric circles with a tangent chord -/
theorem area_between_concentric_circles_with_tangent_chord 
  (r : ℝ) -- radius of the smaller circle
  (c : ℝ) -- length of the chord of the larger circle
  (h1 : r = 40) -- given radius of the smaller circle
  (h2 : c = 120) -- given length of the chord
  : ∃ (A : ℝ), A = 3600 * Real.pi ∧ A = Real.pi * ((c / 2)^2 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_with_tangent_chord_l3421_342191


namespace NUMINAMATH_CALUDE_optimal_rate_l3421_342150

/- Define the initial conditions -/
def totalRooms : ℕ := 100
def initialRate : ℕ := 400
def initialOccupancy : ℕ := 50
def rateReduction : ℕ := 20
def occupancyIncrease : ℕ := 5

/- Define the revenue function -/
def revenue (rate : ℕ) : ℕ :=
  let occupancy := initialOccupancy + ((initialRate - rate) / rateReduction) * occupancyIncrease
  rate * occupancy

/- Theorem statement -/
theorem optimal_rate :
  ∀ (rate : ℕ), rate ≤ initialRate → revenue 300 ≥ revenue rate :=
sorry

end NUMINAMATH_CALUDE_optimal_rate_l3421_342150


namespace NUMINAMATH_CALUDE_identify_counterfeit_pile_l3421_342183

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : Nat)
  (hasRealCoin : Bool)

/-- Represents the result of weighing two sets of coins -/
inductive WeighResult
  | Equal
  | Unequal

/-- Function to weigh two sets of coins -/
def weigh (pile1 : CoinPile) (pile2 : CoinPile) (count : Nat) : WeighResult :=
  sorry

/-- Theorem stating that it's possible to identify the all-counterfeit pile -/
theorem identify_counterfeit_pile 
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)
  (h1 : pile1.count = 15)
  (h2 : pile2.count = 19)
  (h3 : pile3.count = 25)
  (h4 : pile1.hasRealCoin ∨ pile2.hasRealCoin ∨ pile3.hasRealCoin)
  (h5 : ¬(pile1.hasRealCoin ∧ pile2.hasRealCoin) ∧ 
        ¬(pile1.hasRealCoin ∧ pile3.hasRealCoin) ∧ 
        ¬(pile2.hasRealCoin ∧ pile3.hasRealCoin)) :
  ∃ (p : CoinPile), p ∈ [pile1, pile2, pile3] ∧ ¬p.hasRealCoin :=
sorry

end NUMINAMATH_CALUDE_identify_counterfeit_pile_l3421_342183


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l3421_342193

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has 33 more diagonals than sides if and only if it has 11 sides -/
theorem polygon_diagonals_sides (n : ℕ) : diagonals n = n + 33 ↔ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l3421_342193


namespace NUMINAMATH_CALUDE_function_always_positive_l3421_342168

theorem function_always_positive (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (∀ x, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_function_always_positive_l3421_342168


namespace NUMINAMATH_CALUDE_eleven_sided_polygon_diagonals_l3421_342161

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has an obtuse angle --/
def has_obtuse_angle (p : ConvexPolygon n) : Prop := sorry

theorem eleven_sided_polygon_diagonals :
  ∀ (p : ConvexPolygon 11), has_obtuse_angle p → num_diagonals 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_eleven_sided_polygon_diagonals_l3421_342161


namespace NUMINAMATH_CALUDE_third_set_candies_l3421_342130

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies across all three sets -/
def totalCandies (set1 set2 set3 : CandySet) : ℕ :=
  set1.hard + set1.chocolate + set1.gummy +
  set2.hard + set2.chocolate + set2.gummy +
  set3.hard + set3.chocolate + set3.gummy

theorem third_set_candies
  (set1 set2 set3 : CandySet)
  (h1 : set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate)
  (h2 : set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy)
  (h3 : set1.chocolate = set1.gummy)
  (h4 : set1.hard = set1.chocolate + 7)
  (h5 : set2.hard = set2.chocolate)
  (h6 : set2.gummy = set2.hard - 15)
  (h7 : set3.hard = 0) :
  set3.chocolate + set3.gummy = 29 := by
  sorry

#check third_set_candies

end NUMINAMATH_CALUDE_third_set_candies_l3421_342130


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3421_342156

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- 0.cdcdc... = (c*10 + d) / 99
  c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3421_342156


namespace NUMINAMATH_CALUDE_find_b_l3421_342157

-- Define the set A
def A (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 + b}

-- Theorem statement
theorem find_b : ∃ b : ℝ, (1, 5) ∈ A b ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l3421_342157


namespace NUMINAMATH_CALUDE_girls_divisible_by_nine_l3421_342194

theorem girls_divisible_by_nine (N : Nat) (m c d u : Nat) : 
  N < 10000 →
  N = 1000 * m + 100 * c + 10 * d + u →
  let B := m + c + d + u
  let G := N - B
  G % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_girls_divisible_by_nine_l3421_342194


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3421_342190

open Real

theorem trig_expression_simplification (α : ℝ) (h1 : sin α ≠ 0) (h2 : tan α ≠ 0) :
  (1 / sin α + 1 / tan α) * (1 - cos α) = sin α :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3421_342190


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l3421_342153

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct containers is 3^5 -/
theorem distribute_five_into_three : distribute 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l3421_342153


namespace NUMINAMATH_CALUDE_angle_of_inclination_l3421_342149

theorem angle_of_inclination (x y : ℝ) :
  let line_equation := (Real.sqrt 3) * x + y - 3 = 0
  let angle_of_inclination := 2 * Real.pi / 3
  line_equation → angle_of_inclination = Real.arctan (-(Real.sqrt 3)) + Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_l3421_342149


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3421_342148

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem triangle_area_proof (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2) 
  (h_f : f A = 1) (h_dot : 2 * Real.cos A = Real.sqrt 2) : 
  (1/2) * Real.sin A = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3421_342148


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3421_342127

theorem trig_identity_proof :
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) +
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3421_342127


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3421_342179

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3421_342179


namespace NUMINAMATH_CALUDE_inequality_characterization_l3421_342171

theorem inequality_characterization (x y : ℝ) :
  2 * |x + y| ≤ |x| + |y| ↔
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -(1/3) * x) ∨
  (x < 0 ∧ -(1/3) * x ≤ y ∧ y ≤ -3 * x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_characterization_l3421_342171


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l3421_342140

theorem cos_graph_transformation (x : ℝ) : 
  let f (x : ℝ) := Real.cos ((1/2 : ℝ) * x - π/6)
  let g (x : ℝ) := f (x + π/3)
  let h (x : ℝ) := g (2 * x)
  h x = Real.cos x := by sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l3421_342140


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3421_342114

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 8 + (x + 4)*(x + 6) - 10
  ∃ x₁ x₂ : ℝ, x₁ = -4 + Real.sqrt 5 ∧ x₂ = -4 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3421_342114


namespace NUMINAMATH_CALUDE_A_eq_ge_1989_l3421_342158

/-- The set of functions f: ℕ → ℕ satisfying f(f(x)) - 2f(x) + x = 0 for all x ∈ ℕ -/
def F : Set (ℕ → ℕ) :=
  {f | ∀ x : ℕ, f (f x) - 2 * f x + x = 0}

/-- The set A = {f(1989) | f ∈ F} -/
def A : Set ℕ :=
  {y | ∃ f ∈ F, f 1989 = y}

/-- Theorem stating that A is equal to {k : k ≥ 1989} -/
theorem A_eq_ge_1989 : A = {k : ℕ | k ≥ 1989} := by
  sorry


end NUMINAMATH_CALUDE_A_eq_ge_1989_l3421_342158


namespace NUMINAMATH_CALUDE_max_students_distribution_l3421_342145

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 4860) 
  (h_pencils : pencils = 3645) : 
  (Nat.gcd (2 * pens) (3 * pencils)) / 6 = 202 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3421_342145


namespace NUMINAMATH_CALUDE_rectangle_tiling_no_walls_l3421_342182

/-- A domino tiling of a rectangle. -/
def DominoTiling (m n : ℕ) := Unit

/-- A wall in a domino tiling. -/
def Wall (m n : ℕ) (tiling : DominoTiling m n) := Unit

/-- Predicate indicating if a tiling has no walls. -/
def HasNoWalls (m n : ℕ) (tiling : DominoTiling m n) : Prop :=
  ∀ w : Wall m n tiling, False

theorem rectangle_tiling_no_walls 
  (m n : ℕ) 
  (h_even : Even (m * n))
  (h_m : m ≥ 5)
  (h_n : n ≥ 5)
  (h_not_six : ¬(m = 6 ∧ n = 6)) :
  ∃ (tiling : DominoTiling m n), HasNoWalls m n tiling :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_no_walls_l3421_342182


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3421_342136

theorem cos_alpha_value (α : Real) 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3421_342136


namespace NUMINAMATH_CALUDE_probability_three_black_balls_l3421_342101

theorem probability_three_black_balls (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + black_balls + white_balls →
  red_balls = 10 →
  black_balls = 8 →
  white_balls = 3 →
  (Nat.choose black_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ) = 4 / 95 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_balls_l3421_342101


namespace NUMINAMATH_CALUDE_olivias_bags_l3421_342110

def total_cans : ℕ := 20
def cans_per_bag : ℕ := 5

theorem olivias_bags : 
  total_cans / cans_per_bag = 4 := by sorry

end NUMINAMATH_CALUDE_olivias_bags_l3421_342110


namespace NUMINAMATH_CALUDE_root_of_cubic_polynomials_l3421_342111

theorem root_of_cubic_polynomials (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = 0)
  (hk2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_polynomials_l3421_342111


namespace NUMINAMATH_CALUDE_sum_digits_base_8_999_l3421_342197

def base_8_representation (n : ℕ) : List ℕ := sorry

theorem sum_digits_base_8_999 : 
  (base_8_representation 999).sum = 19 := by sorry

end NUMINAMATH_CALUDE_sum_digits_base_8_999_l3421_342197


namespace NUMINAMATH_CALUDE_factorization_and_sum_of_coefficients_l3421_342175

theorem factorization_and_sum_of_coefficients :
  ∃ (a b c d e f : ℤ),
    (81 : ℚ) * x^4 - 256 * y^4 = (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) ∧
    (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) = (3 * x - 4 * y) * (3 * x + 4 * y) * (9 * x^2 + 16 * y^2) ∧
    a + b + c + d + e + f = 31 :=
by sorry

end NUMINAMATH_CALUDE_factorization_and_sum_of_coefficients_l3421_342175


namespace NUMINAMATH_CALUDE_sector_central_angle_l3421_342120

theorem sector_central_angle (perimeter : ℝ) (area : ℝ) (angle : ℝ) : 
  perimeter = 4 → area = 1 → angle = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3421_342120


namespace NUMINAMATH_CALUDE_cyclist_time_is_two_hours_l3421_342123

/-- Represents the scenario of two cyclists traveling between two points --/
structure CyclistScenario where
  s : ℝ  -- Base speed of cyclists without wind
  t : ℝ  -- Time taken by Cyclist 1 to travel from A to B
  wind_speed : ℝ := 3  -- Wind speed affecting both cyclists

/-- Conditions of the cyclist problem --/
def cyclist_problem (scenario : CyclistScenario) : Prop :=
  let total_time := 4  -- Total time after which they meet
  -- Distance covered by Cyclist 1 in total_time
  let dist_cyclist1 := scenario.s * total_time + scenario.wind_speed * (2 * scenario.t - total_time)
  -- Distance covered by Cyclist 2 in total_time
  let dist_cyclist2 := (scenario.s - scenario.wind_speed) * total_time
  -- They meet halfway of the total distance
  dist_cyclist1 = dist_cyclist2 + (scenario.s + scenario.wind_speed) * scenario.t

/-- The theorem stating that the time taken by Cyclist 1 to travel from A to B is 2 hours --/
theorem cyclist_time_is_two_hours (scenario : CyclistScenario) :
  cyclist_problem scenario → scenario.t = 2 := by
  sorry

#check cyclist_time_is_two_hours

end NUMINAMATH_CALUDE_cyclist_time_is_two_hours_l3421_342123


namespace NUMINAMATH_CALUDE_chef_michel_pies_l3421_342199

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered shepherd's pie slices -/
def shepherds_pie_customers : ℕ := 52

/-- Represents the number of customers who ordered chicken pot pie slices -/
def chicken_pot_pie_customers : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ := 
  (shepherds_pie_customers / shepherds_pie_pieces) + 
  (chicken_pot_pie_customers / chicken_pot_pie_pieces)

theorem chef_michel_pies : total_pies_sold = 29 := by
  sorry

end NUMINAMATH_CALUDE_chef_michel_pies_l3421_342199


namespace NUMINAMATH_CALUDE_cube_sphere_comparison_l3421_342128

theorem cube_sphere_comparison (a b R : ℝ) 
  (h1 : 6 * a^2 = 4 * Real.pi * R^2) 
  (h2 : b^3 = (4/3) * Real.pi * R^3) :
  a < b :=
by sorry

end NUMINAMATH_CALUDE_cube_sphere_comparison_l3421_342128


namespace NUMINAMATH_CALUDE_sum_three_numbers_l3421_342152

theorem sum_three_numbers (a b c M : ℚ) : 
  a + b + c = 120 ∧ 
  a - 9 = M ∧ 
  b + 9 = M ∧ 
  9 * c = M → 
  M = 1080 / 19 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l3421_342152


namespace NUMINAMATH_CALUDE_equation_equivalence_l3421_342118

theorem equation_equivalence (x y : ℝ) : 
  (5 * x + y = 1) ↔ (y = 1 - 5 * x) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3421_342118


namespace NUMINAMATH_CALUDE_boys_camp_total_l3421_342172

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 42 → total = 300 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3421_342172


namespace NUMINAMATH_CALUDE_abs_negative_2023_l3421_342113

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l3421_342113


namespace NUMINAMATH_CALUDE_difference_repetition_l3421_342129

theorem difference_repetition (a : Fin 20 → ℕ) 
  (h_order : ∀ i j, i < j → a i < a j) 
  (h_bound : a 19 ≤ 70) : 
  ∃ (j₁ k₁ j₂ k₂ j₃ k₃ j₄ k₄ : Fin 20), 
    k₁ < j₁ ∧ k₂ < j₂ ∧ k₃ < j₃ ∧ k₄ < j₄ ∧
    (a j₁ - a k₁ : ℤ) = (a j₂ - a k₂) ∧
    (a j₁ - a k₁ : ℤ) = (a j₃ - a k₃) ∧
    (a j₁ - a k₁ : ℤ) = (a j₄ - a k₄) :=
by sorry

end NUMINAMATH_CALUDE_difference_repetition_l3421_342129


namespace NUMINAMATH_CALUDE_f_6_equals_21_l3421_342134

def f (x : ℝ) : ℝ := (x - 1)^2 - 4

theorem f_6_equals_21 : f 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_21_l3421_342134


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3421_342151

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line l₁: ax + y + 1 = 0 -/
def l1 (a : ℝ) : Line2D :=
  ⟨a, 1, 1⟩

/-- The second line l₂: 2x + (a + 1)y + 3 = 0 -/
def l2 (a : ℝ) : Line2D :=
  ⟨2, a + 1, 3⟩

/-- a = 1 is sufficient but not necessary for the lines to be parallel -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → areParallel (l1 a) (l2 a)) ∧
  ¬(∀ a : ℝ, areParallel (l1 a) (l2 a) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3421_342151


namespace NUMINAMATH_CALUDE_inequality_holds_l3421_342165

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : a / c^2 > b / c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3421_342165


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3421_342189

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3421_342189


namespace NUMINAMATH_CALUDE_work_completion_time_l3421_342117

/-- Given that two workers A and B can complete a work together in 16 days,
    and A alone can complete the work in 24 days, prove that B alone will
    complete the work in 48 days. -/
theorem work_completion_time
  (joint_time : ℝ) (a_time : ℝ) (b_time : ℝ)
  (h1 : joint_time = 16)
  (h2 : a_time = 24)
  (h3 : (1 / joint_time) = (1 / a_time) + (1 / b_time)) :
  b_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3421_342117


namespace NUMINAMATH_CALUDE_photos_per_page_l3421_342106

theorem photos_per_page (total_photos : ℕ) (total_pages : ℕ) (h1 : total_photos = 736) (h2 : total_pages = 122) :
  total_photos / total_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_photos_per_page_l3421_342106


namespace NUMINAMATH_CALUDE_smallest_rational_l3421_342107

theorem smallest_rational (S : Set ℚ) (h : S = {-1, 0, 3, -1/3}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_rational_l3421_342107
