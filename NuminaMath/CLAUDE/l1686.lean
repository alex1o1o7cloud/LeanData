import Mathlib

namespace NUMINAMATH_CALUDE_tank_filling_ratio_l1686_168604

/-- Proves that the ratio of time B works alone to total time is 0.5 -/
theorem tank_filling_ratio : 
  ∀ (t_A t_B t_total : ℝ),
  t_A > 0 → t_B > 0 → t_total > 0 →
  (1 / t_A + 1 / t_B = 1 / 24) →
  t_total = 29.999999999999993 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ t_total ∧ 
    t / t_B + (t_total - t) / 24 = 1) →
  ∃ t : ℝ, t / t_total = 0.5 := by
sorry

end NUMINAMATH_CALUDE_tank_filling_ratio_l1686_168604


namespace NUMINAMATH_CALUDE_exactly_three_primes_39p_plus_1_perfect_square_l1686_168698

theorem exactly_three_primes_39p_plus_1_perfect_square :
  ∃! (s : Finset Nat), 
    (∀ p ∈ s, Nat.Prime p ∧ ∃ n : Nat, 39 * p + 1 = n^2) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_primes_39p_plus_1_perfect_square_l1686_168698


namespace NUMINAMATH_CALUDE_total_feet_count_l1686_168682

theorem total_feet_count (total_heads : ℕ) (hen_count : ℕ) (hen_feet cow_feet : ℕ) : 
  total_heads = 48 → 
  hen_count = 26 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  (hen_count * hen_feet) + ((total_heads - hen_count) * cow_feet) = 140 := by
sorry

end NUMINAMATH_CALUDE_total_feet_count_l1686_168682


namespace NUMINAMATH_CALUDE_regression_analysis_l1686_168626

structure RegressionData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  original_slope : ℝ
  original_intercept : ℝ
  x_mean : ℝ
  new_slope : ℝ

def positive_correlation (data : RegressionData) : Prop :=
  data.new_slope > 0

def new_regression_equation (data : RegressionData) : Prop :=
  ∃ new_intercept : ℝ, new_intercept = data.x_mean * (data.original_slope - data.new_slope) + data.original_intercept + 1

def decreased_rate_of_increase (data : RegressionData) : Prop :=
  data.new_slope < data.original_slope

theorem regression_analysis (data : RegressionData) 
  (h1 : data.original_slope = 2)
  (h2 : data.original_intercept = -1)
  (h3 : data.x_mean = 3)
  (h4 : data.new_slope = 1.2) :
  positive_correlation data ∧ 
  new_regression_equation data ∧ 
  decreased_rate_of_increase data := by
  sorry

end NUMINAMATH_CALUDE_regression_analysis_l1686_168626


namespace NUMINAMATH_CALUDE_max_sections_with_five_lines_l1686_168664

/-- The number of sections created by drawing n line segments through a rectangle -/
def num_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else num_sections (n - 1) + n

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_with_five_lines :
  num_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_with_five_lines_l1686_168664


namespace NUMINAMATH_CALUDE_domain_of_f_l1686_168639

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1) / Real.log 10

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l1686_168639


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1686_168640

theorem sphere_surface_area (V : ℝ) (S : ℝ) : 
  V = (32 / 3) * Real.pi → S = 4 * Real.pi * ((3 * V) / (4 * Real.pi))^(2/3) → S = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1686_168640


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l1686_168699

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 500
  let pitcher2_capacity : ℚ := 600
  let pitcher1_juice_ratio : ℚ := 1/4
  let pitcher2_juice_ratio : ℚ := 1/3
  let total_juice := pitcher1_capacity * pitcher1_juice_ratio + pitcher2_capacity * pitcher2_juice_ratio
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 13/44 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l1686_168699


namespace NUMINAMATH_CALUDE_rectangleB_is_leftmost_l1686_168679

-- Define a structure for rectangles
structure Rectangle where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def rectangleA : Rectangle := ⟨'A', 5, 2, 8, 10⟩
def rectangleB : Rectangle := ⟨'B', 2, 1, 6, 9⟩
def rectangleC : Rectangle := ⟨'C', 4, 7, 3, 0⟩
def rectangleD : Rectangle := ⟨'D', 9, 6, 5, 11⟩
def rectangleE : Rectangle := ⟨'E', 10, 4, 7, 2⟩

-- Define a list of all rectangles
def allRectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) (rectangles : List Rectangle) : Prop :=
  ∀ other ∈ rectangles, r.w ≤ other.w

-- Theorem statement
theorem rectangleB_is_leftmost :
  isLeftmost rectangleB allRectangles :=
sorry

end NUMINAMATH_CALUDE_rectangleB_is_leftmost_l1686_168679


namespace NUMINAMATH_CALUDE_senior_class_size_l1686_168652

theorem senior_class_size (total : ℕ) 
  (h1 : total / 5 = total / 5)  -- A fifth of the senior class is in the marching band
  (h2 : (total / 5) / 2 = (total / 5) / 2)  -- Half of the marching band plays brass instruments
  (h3 : ((total / 5) / 2) / 5 = ((total / 5) / 2) / 5)  -- A fifth of the brass instrument players play saxophone
  (h4 : (((total / 5) / 2) / 5) / 3 = (((total / 5) / 2) / 5) / 3)  -- A third of the saxophone players play alto saxophone
  (h5 : (((total / 5) / 2) / 5) / 3 = 4)  -- 4 students play alto saxophone
  : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_size_l1686_168652


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l1686_168607

/-- Given a cone whose lateral surface development forms a sector with a central angle of 120° and radius l, 
    prove that the ratio of its total surface area to its lateral surface area is 4:3. -/
theorem cone_surface_area_ratio (l : ℝ) (h : l > 0) : 
  let r := l / 3
  let lateral_area := π * l * r
  let base_area := π * r^2
  let total_area := lateral_area + base_area
  (total_area / lateral_area : ℝ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l1686_168607


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1686_168642

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1686_168642


namespace NUMINAMATH_CALUDE_sum_of_squares_l1686_168695

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 2) (h2 : x^3 + y^3 = 3) : x^2 + y^2 = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1686_168695


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1686_168681

/-- Given sets A and B, prove their intersection -/
theorem intersection_of_sets :
  let A : Set ℝ := {x | x < 1}
  let B : Set ℝ := {x | x^2 - x - 6 < 0}
  A ∩ B = {x | -2 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1686_168681


namespace NUMINAMATH_CALUDE_tangent_lines_range_l1686_168666

/-- The range of k values for which two tangent lines exist from (1, 2) to the circle -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  ((1:ℝ)^2 + 2^2 + k*1 + 2*2 + k^2 - 15 > 0) ↔ 
  (k > 2 ∧ k < 8/3 * Real.sqrt 3) ∨ (k > -8/3 * Real.sqrt 3 ∧ k < -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l1686_168666


namespace NUMINAMATH_CALUDE_specific_value_problem_l1686_168616

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 15 * x = specific_value) (h2 : x = 11) : 
  specific_value = 165 := by
  sorry

end NUMINAMATH_CALUDE_specific_value_problem_l1686_168616


namespace NUMINAMATH_CALUDE_investment_average_rate_l1686_168688

/-- Proves that given a total investment split between two rates with equal returns, the average rate is as expected -/
theorem investment_average_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.05 ∧ rate2 = 0.03)
  (h_equal_returns : ∃ (x : ℝ), x * rate1 = (total_investment - x) * rate2)
  : (((rate1 * (total_investment * rate1 / (rate1 + rate2))) + 
     (rate2 * (total_investment * rate2 / (rate1 + rate2)))) / total_investment) = 0.0375 :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l1686_168688


namespace NUMINAMATH_CALUDE_curve_properties_l1686_168625

-- Define the curve y = ax^3 + bx
def curve (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of the curve
def curve_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem curve_properties (a b : ℝ) :
  curve a b 2 = 2 ∧ 
  curve_derivative a b 2 = 9 →
  a * b = -3 ∧
  Set.Icc (-3/2 : ℝ) 3 ⊆ Set.Icc (-2 : ℝ) 18 ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3/2 : ℝ) 3 ∧ 
             x₂ ∈ Set.Icc (-3/2 : ℝ) 3 ∧
             curve a b x₁ = -2 ∧
             curve a b x₂ = 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_properties_l1686_168625


namespace NUMINAMATH_CALUDE_intersection_is_empty_l1686_168683

open Set

def A : Set ℝ := Ioc (-1) 3
def B : Set ℝ := {2, 4}

theorem intersection_is_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l1686_168683


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l1686_168621

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l1686_168621


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1686_168692

theorem quadratic_no_real_roots :
  ∀ x : ℝ, 2 * (x - 1)^2 + 2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1686_168692


namespace NUMINAMATH_CALUDE_jose_join_time_l1686_168600

theorem jose_join_time (tom_investment : ℕ) (jose_investment : ℕ) (total_profit : ℕ) (jose_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  total_profit = 54000 →
  jose_profit = 30000 →
  ∃ x : ℕ,
    x ≤ 12 ∧
    (tom_investment * 12) / (tom_investment * 12 + jose_investment * (12 - x)) =
    (total_profit - jose_profit) / total_profit ∧
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_jose_join_time_l1686_168600


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1686_168693

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + a = 0 ∧ x = 2) → 
  (a = 2 ∧ ∃ y : ℝ, y^2 - 3*y + a = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1686_168693


namespace NUMINAMATH_CALUDE_tan_half_angle_l1686_168684

theorem tan_half_angle (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.sin α + 2 * Real.cos α = 2) : Real.tan (α / 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_l1686_168684


namespace NUMINAMATH_CALUDE_gabby_fruit_count_l1686_168633

/-- The number of fruits Gabby picked in total -/
def total_fruits (watermelons peaches plums : ℕ) : ℕ := watermelons + peaches + plums

/-- The number of watermelons Gabby got -/
def watermelons : ℕ := 1

/-- The number of peaches Gabby got -/
def peaches : ℕ := watermelons + 12

/-- The number of plums Gabby got -/
def plums : ℕ := peaches * 3

theorem gabby_fruit_count :
  total_fruits watermelons peaches plums = 53 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_count_l1686_168633


namespace NUMINAMATH_CALUDE_subset_condition_l1686_168638

theorem subset_condition (a : ℝ) : 
  {x : ℝ | a ≤ x ∧ x < 7} ⊆ {x : ℝ | 2 < x ∧ x < 10} ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1686_168638


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1686_168662

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

def reverse_number (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem unique_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (reverse_number n = n + 7182) → n = 1909 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1686_168662


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1686_168665

-- Define the sets A and B
def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1686_168665


namespace NUMINAMATH_CALUDE_max_value_of_f_l1686_168686

theorem max_value_of_f (x : ℝ) : 
  x / (x^2 + 9) + 1 / (x^2 - 6*x + 21) + Real.cos (2 * Real.pi * x) ≤ 1.25 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1686_168686


namespace NUMINAMATH_CALUDE_mehki_age_l1686_168632

/-- Given the ages of Zrinka, Jordyn, and Mehki, prove Mehki's age is 22 years. -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) : 
  mehki = 22 := by
sorry

end NUMINAMATH_CALUDE_mehki_age_l1686_168632


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1686_168651

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1686_168651


namespace NUMINAMATH_CALUDE_A_intersect_B_l1686_168602

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1686_168602


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l1686_168654

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l1686_168654


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1686_168641

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1686_168641


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_450_l1686_168615

/-- The sum of positive divisors of a natural number n -/
noncomputable def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 450 is 2 -/
theorem distinct_prime_factors_of_divisor_sum_450 : 
  num_distinct_prime_factors (sum_of_divisors 450) = 2 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_450_l1686_168615


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1686_168668

/-- Given an ellipse and a hyperbola sharing a common focus, prove that a² = 11 under specific conditions --/
theorem ellipse_hyperbola_intersection (a b : ℝ) : 
  a > b → b > 0 →
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →  -- Ellipse C1
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / 2 - (y t)^2 / 8 = 1) →  -- Hyperbola C2
  (a^2 - b^2 = 10) →  -- Common focus condition
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = a^2) ∧ (B.1^2 + B.2^2 = a^2) ∧  -- A and B on circle
    (∃ (k : ℝ), A.2 = k * A.1 ∧ B.2 = k * B.1) ∧  -- A and B on asymptote
    (∃ (C D : ℝ × ℝ), 
      C.1^2 / a^2 + C.2^2 / b^2 = 1 ∧
      D.1^2 / a^2 + D.2^2 / b^2 = 1 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*a/3)^2)) →  -- C1 divides AB into three equal parts
  a^2 = 11 := by
sorry


end NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1686_168668


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l1686_168609

theorem smallest_equal_packages (hamburger_pack : ℕ) (bun_pack : ℕ) : 
  hamburger_pack = 10 → bun_pack = 15 → 
  (∃ (h b : ℕ), h * hamburger_pack = b * bun_pack ∧ 
   ∀ (h' b' : ℕ), h' * hamburger_pack = b' * bun_pack → h ≤ h') → 
  (∃ (h : ℕ), h * hamburger_pack = 3 * hamburger_pack ∧ 
   ∃ (b : ℕ), b * bun_pack = 3 * hamburger_pack) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l1686_168609


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1686_168690

/-- Represents a coloring of an infinite grid -/
def GridColoring := ℤ → ℤ → Bool

/-- Represents a move of the (m, n)-condylure -/
structure CondylureMove (m n : ℕ+) where
  horizontal : ℤ
  vertical : ℤ
  move_valid : (horizontal.natAbs = m ∧ vertical = 0) ∨ (horizontal = 0 ∧ vertical.natAbs = n)

/-- Theorem stating that for any positive m and n, there exists a grid coloring
    such that the (m, n)-condylure always lands on a different colored cell -/
theorem exists_valid_coloring (m n : ℕ+) :
  ∃ (coloring : GridColoring),
    ∀ (x y : ℤ) (move : CondylureMove m n),
      coloring (x + move.horizontal) (y + move.vertical) ≠ coloring x y :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1686_168690


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_one_l1686_168612

theorem simplify_and_evaluate_one (x y : ℚ) :
  x = 1/2 ∧ y = -1 →
  (1 * (2*x + y) * (2*x - y)) - 4*x*(x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_one_l1686_168612


namespace NUMINAMATH_CALUDE_position_of_2015_l1686_168667

/-- Represents a digit in the base-6 number system -/
inductive Digit : Type
| zero : Digit
| one : Digit
| two : Digit
| three : Digit
| four : Digit
| five : Digit

/-- Converts a base-6 number to its decimal equivalent -/
def toDecimal (n : List Digit) : Nat :=
  sorry

/-- Checks if a number is representable in base-6 using digits 0-5 -/
def isValidBase6 (n : Nat) : Prop :=
  sorry

/-- The sequence of numbers formed by digits 0-5 in ascending order -/
def base6Sequence : List Nat :=
  sorry

/-- The position of a number in the base6Sequence -/
def positionInSequence (n : Nat) : Nat :=
  sorry

/-- Theorem: The position of 2015 in the base-6 sequence is 443 -/
theorem position_of_2015 : positionInSequence 2015 = 443 :=
  sorry

end NUMINAMATH_CALUDE_position_of_2015_l1686_168667


namespace NUMINAMATH_CALUDE_fish_in_tank_l1686_168696

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total →   -- One third of the fish are blue
  2 * spotted = blue → -- Half of the blue fish have spots
  spotted = 10 →       -- There are 10 blue, spotted fish
  total = 60 :=        -- Prove that the total number of fish is 60
by sorry

end NUMINAMATH_CALUDE_fish_in_tank_l1686_168696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1686_168601

/-- An arithmetic sequence with common difference d and first term 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 2*d + (n - 1)*d

/-- The value of k for which a_k is the geometric mean of a_1 and a_{2k+1} -/
def k_value : ℕ := 3

theorem arithmetic_sequence_geometric_mean (d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence d
  (a k_value)^2 = a 1 * a (2*k_value + 1) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1686_168601


namespace NUMINAMATH_CALUDE_hexagon_parallelogram_theorem_l1686_168694

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a hexagon as a collection of 6 points
structure Hexagon :=
  (A B C D E F : Point)

-- Define a property for convex hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a quadrilateral as a collection of 4 points
structure Quadrilateral :=
  (P Q R S : Point)

-- Define a property for parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem hexagon_parallelogram_theorem (h : Hexagon) 
  (convex_h : is_convex h)
  (para_ABDE : is_parallelogram ⟨h.A, h.B, h.D, h.E⟩)
  (para_ACDF : is_parallelogram ⟨h.A, h.C, h.D, h.F⟩) :
  is_parallelogram ⟨h.B, h.C, h.E, h.F⟩ := by
  sorry

end NUMINAMATH_CALUDE_hexagon_parallelogram_theorem_l1686_168694


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1686_168620

theorem complex_arithmetic_equality : (98 * 76 - 679 * 8) / (24 * 6 + 25 * 25 * 3 - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1686_168620


namespace NUMINAMATH_CALUDE_negation_equivalence_l1686_168670

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1686_168670


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l1686_168644

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_i_times_one_plus_i :
  (i * (1 + i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l1686_168644


namespace NUMINAMATH_CALUDE_horner_method_result_l1686_168634

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Theorem statement
theorem horner_method_result : f (-2) = 325.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l1686_168634


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1686_168649

def z : ℂ := (2 + Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1686_168649


namespace NUMINAMATH_CALUDE_max_leftover_apples_l1686_168631

theorem max_leftover_apples (n : ℕ) (h : n > 0) : 
  ∃ (m : ℕ), m > 0 ∧ m < n ∧ 
  ∀ (total : ℕ), total ≥ n * (total / n) + m → total / n = (total - m) / n :=
by
  sorry

end NUMINAMATH_CALUDE_max_leftover_apples_l1686_168631


namespace NUMINAMATH_CALUDE_unique_a_value_l1686_168647

def U : Set ℤ := {-5, -3, 1, 2, 3, 4, 5, 6}

def A : Set ℤ := {x | x^2 - 7*x + 12 = 0}

def B (a : ℤ) : Set ℤ := {a^2, 2*a - 1, 6}

theorem unique_a_value : 
  ∃! a : ℤ, A ∩ B a = {4} ∧ B a ⊆ U ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l1686_168647


namespace NUMINAMATH_CALUDE_min_value_expression_l1686_168619

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ m) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1686_168619


namespace NUMINAMATH_CALUDE_two_oplus_neg_three_l1686_168629

/-- The ⊕ operation for rational numbers -/
def oplus (α β : ℚ) : ℚ := α * β + 1

/-- Theorem stating that 2 ⊕ (-3) = -5 -/
theorem two_oplus_neg_three : oplus 2 (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_two_oplus_neg_three_l1686_168629


namespace NUMINAMATH_CALUDE_smallest_n_for_252_terms_l1686_168677

def count_terms (n : ℕ) : ℕ := Nat.choose n 5

theorem smallest_n_for_252_terms : 
  (∀ k < 10, count_terms k ≠ 252) ∧ count_terms 10 = 252 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_252_terms_l1686_168677


namespace NUMINAMATH_CALUDE_seashells_to_glass_ratio_l1686_168660

/-- Represents the number of treasures Simon collected -/
structure Treasures where
  sandDollars : ℕ
  glasspieces : ℕ
  seashells : ℕ

/-- The conditions of Simon's treasure collection -/
def simonsTreasures : Treasures where
  sandDollars := 10
  glasspieces := 3 * 10
  seashells := 3 * 10

/-- The total number of treasures Simon collected -/
def totalTreasures : ℕ := 190

/-- Theorem stating that the ratio of seashells to glass pieces is 1:1 -/
theorem seashells_to_glass_ratio (t : Treasures) 
  (h1 : t.sandDollars = simonsTreasures.sandDollars)
  (h2 : t.glasspieces = 3 * t.sandDollars)
  (h3 : t.seashells = t.glasspieces)
  (h4 : t.sandDollars + t.glasspieces + t.seashells = totalTreasures) :
  t.seashells = t.glasspieces := by
  sorry

#check seashells_to_glass_ratio

end NUMINAMATH_CALUDE_seashells_to_glass_ratio_l1686_168660


namespace NUMINAMATH_CALUDE_min_value_a_l1686_168643

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * a * Real.exp (2 * x) - Real.log x + Real.log a ≥ 0) →
  a ≥ 1 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1686_168643


namespace NUMINAMATH_CALUDE_division_problem_l1686_168630

theorem division_problem (dividend quotient divisor remainder n : ℕ) : 
  dividend = 86 →
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1686_168630


namespace NUMINAMATH_CALUDE_triangle_condition_equivalent_to_m_gt_2_l1686_168622

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval [0,2]
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_condition_equivalent_to_m_gt_2 :
  (∀ m : ℝ, (∀ a b c : ℝ, a ∈ I → b ∈ I → c ∈ I → a ≠ b → b ≠ c → a ≠ c →
    triangle_inequality (f m a) (f m b) (f m c)) ↔ m > 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_equivalent_to_m_gt_2_l1686_168622


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1686_168659

theorem imaginary_part_of_complex_expression :
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1686_168659


namespace NUMINAMATH_CALUDE_factorization_equality_l1686_168680

theorem factorization_equality (x y : ℝ) :
  x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1686_168680


namespace NUMINAMATH_CALUDE_binomial_12_9_l1686_168685

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l1686_168685


namespace NUMINAMATH_CALUDE_sum_and_reverse_contradiction_l1686_168673

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem sum_and_reverse_contradiction :
  let sum := 137 + 276
  sum = 413 ∧ reverse_digits sum ≠ 534 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reverse_contradiction_l1686_168673


namespace NUMINAMATH_CALUDE_orange_roses_count_l1686_168672

theorem orange_roses_count (red_roses : ℕ) (pink_roses : ℕ) (yellow_roses : ℕ) 
  (total_picked : ℕ) (h1 : red_roses = 12) (h2 : pink_roses = 18) 
  (h3 : yellow_roses = 20) (h4 : total_picked = 22) :
  ∃ (orange_roses : ℕ), 
    orange_roses = 8 ∧ 
    total_picked = red_roses / 2 + pink_roses / 2 + yellow_roses / 4 + orange_roses / 4 :=
by sorry

end NUMINAMATH_CALUDE_orange_roses_count_l1686_168672


namespace NUMINAMATH_CALUDE_box_two_neg_one_zero_l1686_168627

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_two_neg_one_zero : box 2 (-1) 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_box_two_neg_one_zero_l1686_168627


namespace NUMINAMATH_CALUDE_fraction_equality_l1686_168676

theorem fraction_equality (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1686_168676


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1686_168650

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y + 5) : 
  x + y + z = 7 * x + 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1686_168650


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1686_168636

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 5, 2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 1]
  A * B = !![6, 21; -4, 32] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1686_168636


namespace NUMINAMATH_CALUDE_total_protest_days_equals_29_625_l1686_168618

/-- Calculates the total number of days spent at four protests -/
def total_protest_days (first_protest : ℝ) (second_increase : ℝ) (third_increase : ℝ) (fourth_increase : ℝ) : ℝ :=
  let second_protest := first_protest * (1 + second_increase)
  let third_protest := second_protest * (1 + third_increase)
  let fourth_protest := third_protest * (1 + fourth_increase)
  first_protest + second_protest + third_protest + fourth_protest

/-- Theorem stating that the total number of days spent at four protests equals 29.625 -/
theorem total_protest_days_equals_29_625 :
  total_protest_days 4 0.25 0.5 0.75 = 29.625 := by
  sorry

end NUMINAMATH_CALUDE_total_protest_days_equals_29_625_l1686_168618


namespace NUMINAMATH_CALUDE_inverse_value_l1686_168610

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the conditions
axiom has_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f
axiom f_at_2 : f 2 = -1

-- State the theorem
theorem inverse_value : f_inv (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_value_l1686_168610


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1686_168628

theorem max_imaginary_part_of_roots (z : ℂ) (θ : ℝ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  -π/2 ≤ θ ∧ θ ≤ π/2 →
  z.im = Real.sin θ →
  z.im ≤ Real.sin (84 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1686_168628


namespace NUMINAMATH_CALUDE_fuel_refills_l1686_168697

theorem fuel_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 40) (h2 : cost_per_refill = 10) :
  total_spent / cost_per_refill = 4 := by
  sorry

end NUMINAMATH_CALUDE_fuel_refills_l1686_168697


namespace NUMINAMATH_CALUDE_largest_x_floor_fraction_l1686_168661

open Real

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 11 / 12) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 11 / 12 → y ≤ 120 / 11) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_fraction_l1686_168661


namespace NUMINAMATH_CALUDE_expression_simplification_l1686_168689

theorem expression_simplification :
  (Real.sqrt 2 * 2^(1/2 : ℝ) * 2) + (18 / 3 * 2) - (8^(1/2 : ℝ) * 4) = 16 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1686_168689


namespace NUMINAMATH_CALUDE_triangle_solutions_l1686_168635

theorem triangle_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 2 + Real.sqrt 6) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧
    a / Real.sin A = c / Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_solutions_l1686_168635


namespace NUMINAMATH_CALUDE_courtyard_paving_cost_l1686_168658

/-- Calculates the cost of paving a rectangular courtyard -/
theorem courtyard_paving_cost 
  (ratio_long : ℝ) 
  (ratio_short : ℝ) 
  (diagonal : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_ratio : ratio_long / ratio_short = 4 / 3) 
  (h_diagonal : diagonal = 45) 
  (h_cost : cost_per_sqm = 0.5) : 
  ⌊(ratio_long * ratio_short * (diagonal^2 / (ratio_long^2 + ratio_short^2)) * cost_per_sqm * 100) / 100⌋ = 486 := by
sorry

end NUMINAMATH_CALUDE_courtyard_paving_cost_l1686_168658


namespace NUMINAMATH_CALUDE_day2_sale_is_1043_l1686_168669

/-- Represents the sales data for a grocer over 5 days -/
structure SalesData where
  average : ℕ
  day1 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ

/-- Calculates the sale on the second day given the sales data -/
def calculateDay2Sale (data : SalesData) : ℕ :=
  5 * data.average - (data.day1 + data.day3 + data.day4 + data.day5)

/-- Proves that the sale on the second day is 1043 given the specified sales data -/
theorem day2_sale_is_1043 (data : SalesData) 
    (h1 : data.average = 625)
    (h2 : data.day1 = 435)
    (h3 : data.day3 = 855)
    (h4 : data.day4 = 230)
    (h5 : data.day5 = 562) :
    calculateDay2Sale data = 1043 := by
  sorry

end NUMINAMATH_CALUDE_day2_sale_is_1043_l1686_168669


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1686_168674

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1686_168674


namespace NUMINAMATH_CALUDE_earth_capacity_theorem_l1686_168687

/-- Represents the Earth's resource capacity --/
structure EarthCapacity where
  peopleA : ℕ  -- Number of people in scenario A
  yearsA : ℕ   -- Number of years in scenario A
  peopleB : ℕ  -- Number of people in scenario B
  yearsB : ℕ   -- Number of years in scenario B

/-- Calculates the maximum sustainable population given Earth's resource capacity --/
def maxSustainablePopulation (capacity : EarthCapacity) : ℕ :=
  ((capacity.peopleB * capacity.yearsB - capacity.peopleA * capacity.yearsA) / (capacity.yearsB - capacity.yearsA))

/-- Theorem stating the maximum sustainable population for given conditions --/
theorem earth_capacity_theorem (capacity : EarthCapacity) 
  (h1 : capacity.peopleA = 11)
  (h2 : capacity.yearsA = 90)
  (h3 : capacity.peopleB = 9)
  (h4 : capacity.yearsB = 210) :
  maxSustainablePopulation capacity = 75 := by
  sorry

end NUMINAMATH_CALUDE_earth_capacity_theorem_l1686_168687


namespace NUMINAMATH_CALUDE_milan_bill_cost_l1686_168648

/-- Calculates the total cost of a long distance phone bill -/
def long_distance_bill_cost (monthly_fee : ℚ) (cost_per_minute : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + cost_per_minute * minutes_used

/-- Proves that Milan's long distance bill cost is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  long_distance_bill_cost monthly_fee cost_per_minute minutes_used = 2336 / 100 := by
  sorry

end NUMINAMATH_CALUDE_milan_bill_cost_l1686_168648


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l1686_168663

/-- For positive real numbers a and b where a > b, the sum of the infinite series
    1/(ba) + 1/(a(2a + b)) + 1/((2a + b)(3a + 2b)) + 1/((3a + 2b)(4a + 3b)) + ...
    is equal to 1/((a + b)b) -/
theorem infinite_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n : ℕ => 1 / ((n * a + (n - 1) * b) * ((n + 1) * a + n * b))
  tsum series = 1 / ((a + b) * b) := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l1686_168663


namespace NUMINAMATH_CALUDE_new_man_weight_l1686_168603

theorem new_man_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  replaced_weight = 68 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_new_man_weight_l1686_168603


namespace NUMINAMATH_CALUDE_square_plus_one_geq_double_l1686_168645

theorem square_plus_one_geq_double (x : ℝ) : x^2 + 1 ≥ 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_double_l1686_168645


namespace NUMINAMATH_CALUDE_remainder_mod_24_l1686_168623

theorem remainder_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_24_l1686_168623


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l1686_168655

theorem odd_prime_square_root (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k) :
  k = (p + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l1686_168655


namespace NUMINAMATH_CALUDE_triangle_inequality_l1686_168657

theorem triangle_inequality (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_inradius : r = (2 * area) / (a + b + c))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) / 16)) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1686_168657


namespace NUMINAMATH_CALUDE_jack_daily_reading_rate_l1686_168675

-- Define the number of books Jack reads in a year
def books_per_year : ℕ := 3285

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- State the theorem
theorem jack_daily_reading_rate :
  books_per_year / days_per_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_jack_daily_reading_rate_l1686_168675


namespace NUMINAMATH_CALUDE_system_solution_system_solution_values_no_solution_fractional_equation_l1686_168614

/-- Proves that the system of equations 2x - 7y = 5 and 3x - 8y = 10 has a unique solution -/
theorem system_solution : ∃! (x y : ℝ), 2*x - 7*y = 5 ∧ 3*x - 8*y = 10 := by sorry

/-- Proves that x = 6 and y = 1 is the solution to the system of equations -/
theorem system_solution_values : 
  ∀ (x y : ℝ), (2*x - 7*y = 5 ∧ 3*x - 8*y = 10) → (x = 6 ∧ y = 1) := by sorry

/-- Proves that the equation 3/(x-1) - (x+2)/(x(x-1)) = 0 has no solution -/
theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ 3/(x-1) - (x+2)/(x*(x-1)) = 0 := by sorry

end NUMINAMATH_CALUDE_system_solution_system_solution_values_no_solution_fractional_equation_l1686_168614


namespace NUMINAMATH_CALUDE_transmission_time_is_256_seconds_l1686_168608

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 40

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Theorem stating that the transmission time is 256 seconds -/
theorem transmission_time_is_256_seconds :
  (num_blocks * chunks_per_block) / transmission_rate = 256 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_256_seconds_l1686_168608


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l1686_168617

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 7 / Real.log 14 + 1) = 1.5 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l1686_168617


namespace NUMINAMATH_CALUDE_range_of_a_l1686_168605

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a ≤ f 2) : a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1686_168605


namespace NUMINAMATH_CALUDE_probability_of_white_after_red_l1686_168606

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 20

/-- Represents the initial number of red balls -/
def initial_red_balls : ℕ := 10

/-- Represents the initial number of white balls -/
def initial_white_balls : ℕ := 10

/-- Represents that the first person draws a red ball -/
def first_draw_red : Prop := true

/-- The probability of drawing a white ball after a red ball is drawn -/
def prob_white_after_red : ℚ := 10 / 19

theorem probability_of_white_after_red :
  first_draw_red →
  prob_white_after_red = initial_white_balls / (total_balls - 1) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_white_after_red_l1686_168606


namespace NUMINAMATH_CALUDE_total_is_sum_of_eaten_and_saved_l1686_168637

/-- The number of strawberries Micah picked in total -/
def total_strawberries : ℕ := sorry

/-- The number of strawberries Micah ate -/
def eaten_strawberries : ℕ := 6

/-- The number of strawberries Micah saved for his mom -/
def saved_strawberries : ℕ := 18

/-- Theorem stating that the total number of strawberries is the sum of eaten and saved strawberries -/
theorem total_is_sum_of_eaten_and_saved : 
  total_strawberries = eaten_strawberries + saved_strawberries :=
by sorry

end NUMINAMATH_CALUDE_total_is_sum_of_eaten_and_saved_l1686_168637


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_with_divisibility_properties_l1686_168611

theorem smallest_three_digit_number_with_divisibility_properties :
  ∃ (n : ℕ), 
    100 ≤ n ∧ n ≤ 999 ∧
    (n - 7) % 7 = 0 ∧
    (n - 8) % 8 = 0 ∧
    (n - 9) % 9 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n →
      ¬((m - 7) % 7 = 0 ∧ (m - 8) % 8 = 0 ∧ (m - 9) % 9 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_with_divisibility_properties_l1686_168611


namespace NUMINAMATH_CALUDE_one_third_of_1206_percent_of_400_l1686_168646

theorem one_third_of_1206_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_1206_percent_of_400_l1686_168646


namespace NUMINAMATH_CALUDE_min_distinct_terms_scalene_triangle_l1686_168691

/-- Represents a scalene triangle with side lengths and angles -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ
  angle_sum : α + β + γ = π
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ
  law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ

/-- The minimum number of distinct terms in the 6-tuple (a,b,c,α,β,γ) for a scalene triangle is 4 -/
theorem min_distinct_terms_scalene_triangle (t : ScaleneTriangle) :
  ∃ (s : Finset ℝ), s.card = 4 ∧ {t.a, t.b, t.c, t.α, t.β, t.γ} ⊆ s :=
sorry

end NUMINAMATH_CALUDE_min_distinct_terms_scalene_triangle_l1686_168691


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1686_168624

-- Equation 1
theorem solve_equation_one : 
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * x - 3
  ∃ x₁ x₂ : ℝ, x₁ = 1 + (Real.sqrt 10) / 2 ∧ 
              x₂ = 1 - (Real.sqrt 10) / 2 ∧ 
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

-- Equation 2
theorem solve_equation_two :
  let g : ℝ → ℝ := λ x => (x^2 + x)^2 - x^2 - x - 30
  ∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 2 ∧ 
              g x₁ = 0 ∧ g x₂ = 0 ∧
              ∀ x : ℝ, g x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1686_168624


namespace NUMINAMATH_CALUDE_min_a_value_l1686_168671

theorem min_a_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 4 5) :
  ∃ (a : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ a * x' ^ 2 + 2 * y' ^ 2) ∧ 
  (∀ (b : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ b * x' ^ 2 + 2 * y' ^ 2) → b ≥ -6) :=
by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_min_a_value_l1686_168671


namespace NUMINAMATH_CALUDE_min_value_ab_l1686_168613

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : b > 1)
  (h3 : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → x * y ≥ a * b) →
  a * b = Real.exp 1 := by
sorry


end NUMINAMATH_CALUDE_min_value_ab_l1686_168613


namespace NUMINAMATH_CALUDE_inequality_proof_l1686_168653

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^4 + y^2*z^2) / (x^(5/2)*(y+z)) + (y^4 + z^2*x^2) / (y^(5/2)*(z+x)) +
  (z^4 + x^2*y^2) / (z^(5/2)*(x+y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1686_168653


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l1686_168678

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 4 / 3 →
  c / a = 2 →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l1686_168678


namespace NUMINAMATH_CALUDE_cube_volume_decomposition_l1686_168656

theorem cube_volume_decomposition (x : ℝ) (hx : x > 0) :
  ∃ (y z : ℝ),
    y = (3/2 + Real.sqrt (3/2)) * x ∧
    z = (3/2 - Real.sqrt (3/2)) * x ∧
    y^3 + z^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_decomposition_l1686_168656
