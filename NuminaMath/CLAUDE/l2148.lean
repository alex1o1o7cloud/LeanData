import Mathlib

namespace NUMINAMATH_CALUDE_root_equation_q_value_l2148_214842

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l2148_214842


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l2148_214884

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 14 → d^2 = l^2 + w^2 + h^2 → w^2 = 122 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l2148_214884


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2148_214875

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 120)
  (h2 : throwers = 55)
  (h3 : 2 * (total_players - throwers) = 5 * (total_players - throwers - (total_players - throwers - throwers)))
  (h4 : throwers ≤ total_players) :
  throwers + (total_players - throwers - (2 * (total_players - throwers) / 5)) = 94 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2148_214875


namespace NUMINAMATH_CALUDE_total_salary_proof_l2148_214802

def salary_n : ℝ := 260

def salary_m : ℝ := 1.2 * salary_n

def total_salary : ℝ := salary_m + salary_n

theorem total_salary_proof : total_salary = 572 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_proof_l2148_214802


namespace NUMINAMATH_CALUDE_division_problem_l2148_214870

theorem division_problem (N : ℕ) : 
  (N / 3 = 4) ∧ (N % 3 = 3) → N = 15 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2148_214870


namespace NUMINAMATH_CALUDE_total_cost_of_all_lawns_l2148_214844

structure Lawn where
  length : ℕ
  breadth : ℕ
  lengthRoadWidth : ℕ
  breadthRoadWidth : ℕ
  costPerSqMeter : ℕ

def totalRoadArea (l : Lawn) : ℕ :=
  l.length * l.lengthRoadWidth + l.breadth * l.breadthRoadWidth

def totalCost (l : Lawn) : ℕ :=
  totalRoadArea l * l.costPerSqMeter

def lawnA : Lawn := ⟨80, 70, 8, 6, 3⟩
def lawnB : Lawn := ⟨120, 50, 12, 10, 4⟩
def lawnC : Lawn := ⟨150, 90, 15, 9, 5⟩

theorem total_cost_of_all_lawns :
  totalCost lawnA + totalCost lawnB + totalCost lawnC = 26240 := by
  sorry

#eval totalCost lawnA + totalCost lawnB + totalCost lawnC

end NUMINAMATH_CALUDE_total_cost_of_all_lawns_l2148_214844


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2148_214843

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((627 + y) % 510 = 0 ∧ (627 + y) % 4590 = 0 ∧ (627 + y) % 105 = 0)) ∧
  ((627 + x) % 510 = 0 ∧ (627 + x) % 4590 = 0 ∧ (627 + x) % 105 = 0) ∧
  x = 31503 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2148_214843


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_720_l2148_214864

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_720 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ 
    digit_product n = 720 → 
    n ≤ 98521 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_720_l2148_214864


namespace NUMINAMATH_CALUDE_inequality_proof_l2148_214828

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / (a^2 + (b + c)^2) +
  (c + a - b)^2 / (b^2 + (c + a)^2) +
  (a + b - c)^2 / (c^2 + (a + b)^2) ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2148_214828


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2148_214852

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2148_214852


namespace NUMINAMATH_CALUDE_range_of_a_l2148_214894

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 + 5*a*x + 6*a^2 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = A) → a < 0 → -1/2 ≥ a ∧ a > -4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2148_214894


namespace NUMINAMATH_CALUDE_smallest_five_digit_palindrome_divisible_by_three_l2148_214860

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The smallest five-digit palindrome divisible by 3 -/
def smallest_palindrome : ℕ := 10001

theorem smallest_five_digit_palindrome_divisible_by_three :
  is_five_digit_palindrome smallest_palindrome ∧ 
  smallest_palindrome % 3 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n → n % 3 = 0 → n ≥ smallest_palindrome := by
  sorry

#eval smallest_palindrome

end NUMINAMATH_CALUDE_smallest_five_digit_palindrome_divisible_by_three_l2148_214860


namespace NUMINAMATH_CALUDE_system_solutions_product_l2148_214801

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x^3 - 5*x*y^2 = 21 ∧ y^3 - 5*x^2*y = 28

-- Define the theorem
theorem system_solutions_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  system x₁ y₁ ∧ system x₂ y₂ ∧ system x₃ y₃ ∧
  (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃) →
  (11 - x₁/y₁) * (11 - x₂/y₂) * (11 - x₃/y₃) = 1729 :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_product_l2148_214801


namespace NUMINAMATH_CALUDE_fraction_inequality_l2148_214892

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2148_214892


namespace NUMINAMATH_CALUDE_randy_block_difference_l2148_214841

/-- Randy's block building problem -/
theorem randy_block_difference :
  ∀ (total_blocks house_blocks tower_blocks : ℕ),
    total_blocks = 90 →
    house_blocks = 89 →
    tower_blocks = 63 →
    house_blocks - tower_blocks = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_randy_block_difference_l2148_214841


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_l2148_214867

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_f_2_l2148_214867


namespace NUMINAMATH_CALUDE_line_intercepts_l2148_214889

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-axis intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-axis intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The x-intercept and y-intercept of the line 3x - 2y - 6 = 0 are 2 and -3 respectively -/
theorem line_intercepts : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end NUMINAMATH_CALUDE_line_intercepts_l2148_214889


namespace NUMINAMATH_CALUDE_coin_rotation_theorem_l2148_214873

/-- 
  Represents the number of degrees a coin rotates when rolling around another coin.
  
  coinA : The rolling coin
  coinB : The stationary coin
  radiusRatio : The ratio of coinB's radius to coinA's radius
  rotationDegrees : The number of degrees coinA rotates around its center
-/
def coinRotation (coinA coinB : ℝ) (radiusRatio : ℝ) (rotationDegrees : ℝ) : Prop :=
  coinA > 0 ∧ 
  coinB > 0 ∧ 
  radiusRatio = 2 ∧ 
  rotationDegrees = 3 * 360

theorem coin_rotation_theorem (coinA coinB radiusRatio rotationDegrees : ℝ) :
  coinRotation coinA coinB radiusRatio rotationDegrees →
  rotationDegrees = 1080 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_rotation_theorem_l2148_214873


namespace NUMINAMATH_CALUDE_factor_count_of_n_l2148_214800

-- Define the number we're working with
def n : ℕ := 8^2 * 9^3 * 10^4

-- Define a function to count distinct natural-number factors
def count_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem factor_count_of_n : count_factors n = 385 := by sorry

end NUMINAMATH_CALUDE_factor_count_of_n_l2148_214800


namespace NUMINAMATH_CALUDE_congruence_implies_b_zero_l2148_214815

theorem congruence_implies_b_zero (a b c m : ℤ) (h_m : m > 1) 
  (h_cong : ∀ n : ℕ, (a^n + b*n + c) % m = 0) : 
  b % m = 0 ∧ (b^2) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_implies_b_zero_l2148_214815


namespace NUMINAMATH_CALUDE_angle_A_is_45_degrees_l2148_214878

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- The sum of angles in a triangle is 180°
  A + B + C = Real.pi ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- State the theorem
theorem angle_A_is_45_degrees :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = Real.pi / 3 →
  A = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_is_45_degrees_l2148_214878


namespace NUMINAMATH_CALUDE_solve_for_a_l2148_214898

/-- Given that x + 2a - 6 = 0 and x = -2, prove that a = 4 -/
theorem solve_for_a (x a : ℝ) (h1 : x + 2*a - 6 = 0) (h2 : x = -2) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2148_214898


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l2148_214868

theorem min_draws_for_even_product (n : ℕ) (h : n = 16) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 9 ∧ 
  ∀ k : ℕ, k < odd_count + 1 → ∃ subset : Finset ℕ, 
    subset.card = k ∧ 
    subset ⊆ S ∧ 
    ∀ x ∈ subset, x % 2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l2148_214868


namespace NUMINAMATH_CALUDE_buratino_number_problem_l2148_214863

theorem buratino_number_problem (x : ℚ) : 4 * x + 15 = 15 * x + 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_buratino_number_problem_l2148_214863


namespace NUMINAMATH_CALUDE_unique_base_representation_l2148_214879

/-- The repeating base-k representation of a rational number -/
def repeatingBaseK (n d k : ℕ) : ℚ :=
  (4 : ℚ) / k + (7 : ℚ) / k^2

/-- The condition for the repeating base-k representation to equal the given fraction -/
def isValidK (k : ℕ) : Prop :=
  k > 0 ∧ repeatingBaseK 11 77 k = 11 / 77

theorem unique_base_representation :
  ∃! k : ℕ, isValidK k ∧ k = 17 :=
sorry

end NUMINAMATH_CALUDE_unique_base_representation_l2148_214879


namespace NUMINAMATH_CALUDE_tan_11_25_degrees_l2148_214818

theorem tan_11_25_degrees :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (11.25 * π / 180) = Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) - (d : ℝ)) ∧
    (a = 2 + 2) ∧ (b = 2) ∧ (c = 1) ∧ (d = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_11_25_degrees_l2148_214818


namespace NUMINAMATH_CALUDE_income_distribution_equation_l2148_214866

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children_percent : ℝ
  wife_percent : ℝ
  orphan_percent : ℝ
  remaining : ℝ

/-- Theorem stating the relationship between income and its distribution --/
theorem income_distribution_equation (d : IncomeDistribution) 
  (h1 : d.children_percent = 0.1)
  (h2 : d.wife_percent = 0.2)
  (h3 : d.orphan_percent = 0.1)
  (h4 : d.remaining = 500) :
  d.total - (2 * d.children_percent * d.total + 
             d.wife_percent * d.total + 
             d.orphan_percent * (d.total - (2 * d.children_percent * d.total + d.wife_percent * d.total))) = 
  d.remaining := by
  sorry

#eval 500 / 0.54  -- This will output the approximate total income

end NUMINAMATH_CALUDE_income_distribution_equation_l2148_214866


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2148_214893

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2148_214893


namespace NUMINAMATH_CALUDE_point_on_axes_with_inclination_l2148_214855

/-- Given point A(-2, 1) and the angle of inclination of line PA is 30°,
    prove that point P on the coordinate axes is either (-4, 0) or (0, 2). -/
theorem point_on_axes_with_inclination (A : ℝ × ℝ) (P : ℝ × ℝ) :
  A = (-2, 1) →
  (P.1 = 0 ∨ P.2 = 0) →
  (P.2 - A.2) / (P.1 - A.1) = Real.tan (30 * π / 180) →
  (P = (-4, 0) ∨ P = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_axes_with_inclination_l2148_214855


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l2148_214856

theorem find_number_to_multiply (x : ℝ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l2148_214856


namespace NUMINAMATH_CALUDE_production_rates_theorem_l2148_214809

-- Define the number of machines
def num_machines : ℕ := 5

-- Define the list of pairwise production numbers
def pairwise_production : List ℕ := [35, 39, 40, 49, 44, 46, 30, 41, 32, 36]

-- Define the function to check if a list of production rates is valid
def is_valid_production (rates : List ℕ) : Prop :=
  rates.length = num_machines ∧
  rates.sum = 98 ∧
  (∀ i j, i < j → i < rates.length → j < rates.length →
    (rates.get ⟨i, by sorry⟩ + rates.get ⟨j, by sorry⟩) ∈ pairwise_production)

-- Theorem statement
theorem production_rates_theorem :
  ∃ (rates : List ℕ), is_valid_production rates ∧ rates = [13, 17, 19, 22, 27] := by
  sorry

end NUMINAMATH_CALUDE_production_rates_theorem_l2148_214809


namespace NUMINAMATH_CALUDE_exam_questions_attempted_student_exam_result_l2148_214883

theorem exam_questions_attempted (correct_score : ℕ) (wrong_penalty : ℕ) 
  (total_score : ℤ) (correct_answers : ℕ) : ℕ :=
  let wrong_answers := total_score - correct_score * correct_answers
  correct_answers + wrong_answers.toNat

-- Statement of the problem
theorem student_exam_result : 
  exam_questions_attempted 4 1 130 38 = 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_questions_attempted_student_exam_result_l2148_214883


namespace NUMINAMATH_CALUDE_hexagon_shell_arrangements_l2148_214838

/-- The number of rotational symmetries in a regular hexagon -/
def hexagon_rotations : ℕ := 6

/-- The number of distinct points on the hexagon (corners and midpoints) -/
def total_points : ℕ := 12

/-- The number of distinct sea shells -/
def total_shells : ℕ := 12

/-- The number of distinct arrangements of sea shells on a regular hexagon,
    considering only rotational equivalence -/
def distinct_arrangements : ℕ := (Nat.factorial total_shells) / hexagon_rotations

theorem hexagon_shell_arrangements :
  distinct_arrangements = 79833600 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_shell_arrangements_l2148_214838


namespace NUMINAMATH_CALUDE_large_marshmallows_count_l2148_214862

/-- Represents the number of Rice Krispie Treats made -/
def rice_krispie_treats : ℕ := 5

/-- Represents the total number of marshmallows used -/
def total_marshmallows : ℕ := 18

/-- Represents the number of mini marshmallows used -/
def mini_marshmallows : ℕ := 10

/-- Represents the number of large marshmallows used -/
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

theorem large_marshmallows_count : large_marshmallows = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_marshmallows_count_l2148_214862


namespace NUMINAMATH_CALUDE_compound_ratio_proof_l2148_214896

theorem compound_ratio_proof (x y : ℝ) (y_nonzero : y ≠ 0) :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) * (4 / 5) * (x / y) = x / (17.5 * y) :=
by sorry

end NUMINAMATH_CALUDE_compound_ratio_proof_l2148_214896


namespace NUMINAMATH_CALUDE_parallel_lines_circle_intersection_l2148_214826

theorem parallel_lines_circle_intersection (r : ℝ) : 
  ∀ d : ℝ, 
    (17682 + (21/4) * d^2 = 42 * r^2) ∧ 
    (4394 + (117/4) * d^2 = 26 * r^2) → 
    d = Real.sqrt 127 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_intersection_l2148_214826


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l2148_214834

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  binomial_probability p n k = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l2148_214834


namespace NUMINAMATH_CALUDE_test_questions_l2148_214827

theorem test_questions (points_correct : ℕ) (points_incorrect : ℕ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  points_correct = 20 →
  points_incorrect = 5 →
  total_score = 325 →
  correct_answers = 19 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_questions - correct_answers) ∧
    total_score = points_correct * correct_answers - 
      points_incorrect * (total_questions - correct_answers) ∧
    total_questions = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l2148_214827


namespace NUMINAMATH_CALUDE_shaded_area_square_minus_circles_l2148_214808

/-- The shaded area of a square with side length 10 and four circles of radius 3√2 at its vertices -/
theorem shaded_area_square_minus_circles :
  let square_side : ℝ := 10
  let circle_radius : ℝ := 3 * Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let total_circles_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := square_area - total_circles_area
  shaded_area = 100 - 72 * π := by
  sorry

#check shaded_area_square_minus_circles

end NUMINAMATH_CALUDE_shaded_area_square_minus_circles_l2148_214808


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2148_214871

theorem inverse_proportion_problem (x y : ℝ) (h : x * y = 12) :
  x = 5 → y = 2.4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2148_214871


namespace NUMINAMATH_CALUDE_green_peaches_count_l2148_214899

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 1

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 4

/-- The total number of peaches in all baskets -/
def total_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := total_peaches - (num_baskets * red_peaches_per_basket)

theorem green_peaches_count : green_peaches_per_basket = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2148_214899


namespace NUMINAMATH_CALUDE_rationalized_denominator_product_l2148_214881

theorem rationalized_denominator_product (A B C : ℤ) : 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C → A * B * C = 180 := by
  sorry

end NUMINAMATH_CALUDE_rationalized_denominator_product_l2148_214881


namespace NUMINAMATH_CALUDE_atomic_number_calculation_l2148_214819

/-- Represents an atomic element -/
structure Element where
  massNumber : ℕ
  neutronCount : ℕ
  atomicNumber : ℕ

/-- The relation between mass number, neutron count, and atomic number in an element -/
def isValidElement (e : Element) : Prop :=
  e.massNumber = e.neutronCount + e.atomicNumber

theorem atomic_number_calculation (e : Element)
  (h1 : e.massNumber = 288)
  (h2 : e.neutronCount = 169)
  (h3 : isValidElement e) :
  e.atomicNumber = 119 := by
  sorry

#check atomic_number_calculation

end NUMINAMATH_CALUDE_atomic_number_calculation_l2148_214819


namespace NUMINAMATH_CALUDE_ellipse_area_theorem_l2148_214861

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- PF1 is perpendicular to PF2 -/
def PF1_perp_PF2 : Prop := sorry

/-- The area of triangle F1PF2 -/
def area_F1PF2 : ℝ := sorry

theorem ellipse_area_theorem :
  ellipse_equation P.1 P.2 →
  PF1_perp_PF2 →
  area_F1PF2 = 9 := by sorry

end NUMINAMATH_CALUDE_ellipse_area_theorem_l2148_214861


namespace NUMINAMATH_CALUDE_unique_solution_range_l2148_214804

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (a x : ℝ) : Prop :=
  lg (a * x + 1) = lg (x - 1) + lg (2 - x)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a > -1 ∧ a ≤ -1/2) ∨ a = 3 - 2 * Real.sqrt 3

-- Theorem statement
theorem unique_solution_range :
  ∀ a : ℝ, (∃! x : ℝ, equation a x) ↔ a_range a := by sorry

end NUMINAMATH_CALUDE_unique_solution_range_l2148_214804


namespace NUMINAMATH_CALUDE_vasya_incorrect_l2148_214807

theorem vasya_incorrect : ¬∃ (x y : ℤ), (x + y = 2021) ∧ ((10 * x + y = 2221) ∨ (x + 10 * y = 2221)) := by
  sorry

end NUMINAMATH_CALUDE_vasya_incorrect_l2148_214807


namespace NUMINAMATH_CALUDE_largest_base_5_to_base_7_l2148_214813

/-- The largest four-digit number in base-5 -/
def m : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Conversion of a natural number to its base-7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  sorry

theorem largest_base_5_to_base_7 :
  to_base_7 m = [1, 5, 5, 1] :=
sorry

end NUMINAMATH_CALUDE_largest_base_5_to_base_7_l2148_214813


namespace NUMINAMATH_CALUDE_uncle_bradley_bill_change_l2148_214845

theorem uncle_bradley_bill_change (total_amount : ℕ) (small_bill_denom : ℕ) (total_bills : ℕ) :
  total_amount = 1000 →
  small_bill_denom = 50 →
  total_bills = 13 →
  ∃ (large_bill_denom : ℕ),
    (3 * total_amount / 10 / small_bill_denom + (total_amount - 3 * total_amount / 10) / large_bill_denom = total_bills) ∧
    large_bill_denom = 100 := by
  sorry

#check uncle_bradley_bill_change

end NUMINAMATH_CALUDE_uncle_bradley_bill_change_l2148_214845


namespace NUMINAMATH_CALUDE_fraction_and_sum_problem_l2148_214880

theorem fraction_and_sum_problem :
  (5 : ℚ) / 40 = 0.125 ∧ 0.125 + 0.375 = 0.500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_and_sum_problem_l2148_214880


namespace NUMINAMATH_CALUDE_factorization_equality_l2148_214839

theorem factorization_equality (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2148_214839


namespace NUMINAMATH_CALUDE_mark_young_fish_count_l2148_214865

/-- Calculates the total number of young fish given the number of tanks, pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Proves that given 5 tanks, 6 pregnant fish per tank, and 25 young per fish, the total number of young fish is 750. -/
theorem mark_young_fish_count :
  total_young_fish 5 6 25 = 750 := by
  sorry

end NUMINAMATH_CALUDE_mark_young_fish_count_l2148_214865


namespace NUMINAMATH_CALUDE_tom_seashell_count_l2148_214831

/-- The number of days Tom spent at the beach -/
def days_at_beach : ℕ := 5

/-- The number of seashells Tom found each day -/
def seashells_per_day : ℕ := 7

/-- The total number of seashells Tom found during his beach trip -/
def total_seashells : ℕ := days_at_beach * seashells_per_day

/-- Theorem stating that the total number of seashells Tom found is 35 -/
theorem tom_seashell_count : total_seashells = 35 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashell_count_l2148_214831


namespace NUMINAMATH_CALUDE_vector_problem_l2148_214823

/-- Given vectors a and b in ℝ², if vector c satisfies the conditions
    (c + b) ⊥ a and (c - a) ∥ b, then c = (2, 1). -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, -1) → 
  b = (1, 2) → 
  ((c.1 + b.1, c.2 + b.2) • a = 0) →  -- (c + b) ⊥ a
  (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) →  -- (c - a) ∥ b
  c = (2, 1) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l2148_214823


namespace NUMINAMATH_CALUDE_value_of_a_l2148_214890

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 8
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (ha : a > 0) (h : f (g a) = 8) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2148_214890


namespace NUMINAMATH_CALUDE_rose_sale_earnings_l2148_214830

theorem rose_sale_earnings :
  ∀ (price : ℕ) (initial : ℕ) (remaining : ℕ),
    price = 7 →
    initial = 9 →
    remaining = 4 →
    (initial - remaining) * price = 35 :=
by sorry

end NUMINAMATH_CALUDE_rose_sale_earnings_l2148_214830


namespace NUMINAMATH_CALUDE_cistern_depth_is_correct_l2148_214859

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  total_wet_area : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wet_surface_area (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that for a cistern with given dimensions and wet surface area, the depth is 1.25 m --/
theorem cistern_depth_is_correct (c : Cistern) 
    (h1 : c.length = 6)
    (h2 : c.width = 4)
    (h3 : c.total_wet_area = 49)
    (h4 : wet_surface_area c = c.total_wet_area) :
    c.depth = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cistern_depth_is_correct_l2148_214859


namespace NUMINAMATH_CALUDE_spurs_basketball_distribution_l2148_214847

theorem spurs_basketball_distribution (num_players : ℕ) (total_basketballs : ℕ) 
  (h1 : num_players = 22) 
  (h2 : total_basketballs = 242) : 
  total_basketballs / num_players = 11 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketball_distribution_l2148_214847


namespace NUMINAMATH_CALUDE_value_between_seven_and_eight_l2148_214888

theorem value_between_seven_and_eight :
  7 < (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) ∧
  (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) < 8 := by
  sorry

end NUMINAMATH_CALUDE_value_between_seven_and_eight_l2148_214888


namespace NUMINAMATH_CALUDE_ellipse_properties_l2148_214849

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The ellipse satisfies the given conditions -/
def EllipseConditions (e : Ellipse) : Prop :=
  (e.a ^ 2 - e.b ^ 2) / e.a ^ 2 = 3 / 4 ∧  -- eccentricity is √3/2
  e.a - (e.a ^ 2 - e.b ^ 2).sqrt = 2       -- distance from upper vertex to focus is 2

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) (h : EllipseConditions e) :
  e.a = 2 ∧ e.b = 1 ∧
  (∀ k : ℝ, k = 1 → 
    (∃ S : ℝ → ℝ, (∀ m : ℝ, S m ≤ 1) ∧ (∃ m : ℝ, S m = 1))) ∧
  (∀ k : ℝ, (∀ m : ℝ, ∃ C : ℝ, 
    (∀ x : ℝ, (x - m)^2 + (k * (x - m))^2 + 
      ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x)^2 + 
      (k * ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x))^2 = C)) → 
    k = 1/2 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2148_214849


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2148_214812

/-- Given two trains moving in opposite directions, prove the speed of one train given the lengths, speed of the other train, and time to cross. -/
theorem train_speed_calculation (length1 length2 speed2 time_to_cross : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 200.04)
  (h3 : speed2 = 80)
  (h4 : time_to_cross = 9 / 3600) : 
  ∃ speed1 : ℝ, speed1 = 120.016 ∧ 
  (length1 + length2) / 1000 = (speed1 + speed2) * time_to_cross := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2148_214812


namespace NUMINAMATH_CALUDE_range_of_a_l2148_214805

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x + y + z = 1 → |a - 2| ≤ x^2 + 2*y^2 + 3*z^2) →
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2148_214805


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2148_214897

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c p : ℝ) (x₀ y₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 → x₀ > 0 → y₀ > 0 →
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- hyperbola equation
  y₀ = (b / a) * x₀ →  -- point on asymptote
  x₀^2 + y₀^2 = c^2 →  -- MF₁ ⊥ MF₂
  y₀^2 = 2 * p * x₀ →  -- parabola equation
  c / a = 2 + Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2148_214897


namespace NUMINAMATH_CALUDE_distance_between_chord_endpoints_l2148_214814

/-- In a circle with radius R, given two mutually perpendicular chords MN and PQ,
    where NQ = m, the distance between points M and P is √(4R² - m²). -/
theorem distance_between_chord_endpoints (R m : ℝ) (R_pos : R > 0) (m_pos : m > 0) :
  ∃ (M P : ℝ × ℝ),
    (∃ (N Q : ℝ × ℝ),
      (∀ (X : ℝ × ℝ), (X.1 - 0)^2 + (X.2 - 0)^2 = R^2 → 
        ((M.1 - N.1) * (P.1 - Q.1) + (M.2 - N.2) * (P.2 - Q.2) = 0) ∧
        ((N.1 - Q.1)^2 + (N.2 - Q.2)^2 = m^2)) →
      ((M.1 - P.1)^2 + (M.2 - P.2)^2 = 4 * R^2 - m^2)) :=
sorry

end NUMINAMATH_CALUDE_distance_between_chord_endpoints_l2148_214814


namespace NUMINAMATH_CALUDE_existence_of_x0_iff_b_negative_l2148_214840

open Real

theorem existence_of_x0_iff_b_negative (a b : ℝ) (ha : a > 0) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ log x₀ > a * sqrt x₀ + b / sqrt x₀) ↔ b < 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_x0_iff_b_negative_l2148_214840


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l2148_214886

/-- Represents a three-digit number where the first and last digits are the same -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.a

theorem three_digit_divisibility_by_seven (n : ThreeDigitNumber) :
  (n.toNum % 7 = 0) ↔ ((n.a + n.b) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l2148_214886


namespace NUMINAMATH_CALUDE_dale_had_two_eggs_l2148_214825

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (dale_eggs : ℕ) : ℝ :=
  (2 * 1 + dale_eggs * 3) + (1 * 1 + 2 * 3)

/-- Theorem: Dale had 2 eggs -/
theorem dale_had_two_eggs : 
  ∃ (dale_eggs : ℕ), breakfast_cost dale_eggs = 15 ∧ dale_eggs = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dale_had_two_eggs_l2148_214825


namespace NUMINAMATH_CALUDE_sport_water_amount_l2148_214824

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Standard formulation of the flavored drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- Sport formulation of the flavored drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Theorem: The amount of water in the sport formulation is 75 ounces -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.flavoring) * (sport_corn_syrup / sport_ratio.corn_syrup) * sport_ratio.flavoring = 75 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2148_214824


namespace NUMINAMATH_CALUDE_total_remaining_battery_life_l2148_214857

/-- Calculates the total remaining battery life for three calculators after a 2-hour usage -/
theorem total_remaining_battery_life 
  (capacity1 capacity2 capacity3 : ℝ)
  (used_fraction1 used_fraction2 used_fraction3 : ℝ)
  (exam_duration : ℝ)
  (h1 : capacity1 = 60)
  (h2 : capacity2 = 80)
  (h3 : capacity3 = 120)
  (h4 : used_fraction1 = 3/4)
  (h5 : used_fraction2 = 1/2)
  (h6 : used_fraction3 = 2/3)
  (h7 : exam_duration = 2) :
  (capacity1 * (1 - used_fraction1) - exam_duration) +
  (capacity2 * (1 - used_fraction2) - exam_duration) +
  (capacity3 * (1 - used_fraction3) - exam_duration) = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_battery_life_l2148_214857


namespace NUMINAMATH_CALUDE_train_speed_l2148_214853

def train_length : ℝ := 100
def tunnel_length : ℝ := 2300
def time_seconds : ℝ := 120

theorem train_speed :
  let total_distance := tunnel_length + train_length
  let speed_ms := total_distance / time_seconds
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_l2148_214853


namespace NUMINAMATH_CALUDE_first_page_drawings_count_l2148_214851

/-- The number of drawings on the first page of an art book. -/
def first_page_drawings : ℕ := 5

/-- The increase in the number of drawings on each subsequent page. -/
def drawing_increase : ℕ := 5

/-- The total number of pages considered. -/
def total_pages : ℕ := 5

/-- The total number of drawings on the first five pages. -/
def total_drawings : ℕ := 75

/-- Theorem stating that the number of drawings on the first page is 5,
    given the conditions of the problem. -/
theorem first_page_drawings_count :
  (first_page_drawings +
   (first_page_drawings + drawing_increase) +
   (first_page_drawings + 2 * drawing_increase) +
   (first_page_drawings + 3 * drawing_increase) +
   (first_page_drawings + 4 * drawing_increase)) = total_drawings :=
by sorry

end NUMINAMATH_CALUDE_first_page_drawings_count_l2148_214851


namespace NUMINAMATH_CALUDE_snow_depth_theorem_l2148_214821

/-- Calculates the final snow depth after seven days given initial conditions and daily changes --/
def snow_depth_after_seven_days (initial_snow : Real) 
  (day2_snow : Real) (day2_compaction : Real)
  (daily_melt : Real) (day4_cleared : Real)
  (day5_multiplier : Real)
  (day6_melt : Real) (day6_accumulate : Real) : Real :=
  let day1 := initial_snow
  let day2 := day1 + day2_snow * (1 - day2_compaction)
  let day3 := day2 - daily_melt
  let day4 := day3 - daily_melt - day4_cleared
  let day5 := day4 - daily_melt + day5_multiplier * (day1 + day2_snow)
  let day6 := day5 - day6_melt + day6_accumulate
  day6

/-- The final snow depth after seven days is approximately 2.1667 feet --/
theorem snow_depth_theorem : 
  ∃ ε > 0, |snow_depth_after_seven_days 0.5 (8/12) 0.1 (1/12) (6/12) 1.5 (3/12) (4/12) - 2.1667| < ε :=
by sorry

end NUMINAMATH_CALUDE_snow_depth_theorem_l2148_214821


namespace NUMINAMATH_CALUDE_range_of_a_max_value_sum_of_roots_l2148_214822

-- Define the function f
def f (x : ℝ) : ℝ := |x - 4| - |x + 2|

-- Part 1
theorem range_of_a (a : ℝ) :
  (∀ x, f x - a^2 + 5*a ≥ 0) → 2 ≤ a ∧ a ≤ 3 :=
sorry

-- Part 2
theorem max_value_sum_of_roots (M a b c : ℝ) :
  (∀ x, f x ≤ M) →
  a > 0 → b > 0 → c > 0 →
  a + b + c = M →
  (∃ (max_val : ℝ), ∀ a' b' c' : ℝ,
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = M →
    Real.sqrt (a' + 1) + Real.sqrt (b' + 2) + Real.sqrt (c' + 3) ≤ max_val ∧
    max_val = 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_max_value_sum_of_roots_l2148_214822


namespace NUMINAMATH_CALUDE_base_conversion_difference_l2148_214858

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

theorem base_conversion_difference :
  let base_5_num := to_base_10 [1, 3, 4, 2, 5] 5
  let base_8_num := to_base_10 [2, 3, 4, 1] 8
  base_5_num - base_8_num = 2697 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_difference_l2148_214858


namespace NUMINAMATH_CALUDE_xy_sum_square_l2148_214887

theorem xy_sum_square (x y : ℤ) 
  (h1 : x * y + x + y = 106) 
  (h2 : x^2 * y + x * y^2 = 1320) : 
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_square_l2148_214887


namespace NUMINAMATH_CALUDE_circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l2148_214806

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_1 (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 := by
  sorry

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_2 (x y : ℝ) :
  4*x^2 + 4*y^2 - 8*x + 4*y - 11 = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l2148_214806


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2148_214836

theorem quadratic_inequality_minimum (a b c : ℝ) : 
  (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) →
  (∃ m, ∀ a b c, (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) → 
    b - 2*c + 1/a ≥ m ∧ 
    (∃ a₀ b₀ c₀, (∀ x, a₀*x^2 + b₀*x + c₀ < 0 ↔ -1 < x ∧ x < 3) ∧ 
      b₀ - 2*c₀ + 1/a₀ = m)) ∧
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2148_214836


namespace NUMINAMATH_CALUDE_rosies_pies_l2148_214850

/-- Given that Rosie can make 3 pies out of 12 apples, 
    this theorem proves how many pies she can make out of 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) : 
  apples_per_three_pies = 12 → total_apples = 36 → (total_apples / apples_per_three_pies) * 3 = 27 := by
  sorry

#check rosies_pies

end NUMINAMATH_CALUDE_rosies_pies_l2148_214850


namespace NUMINAMATH_CALUDE_books_remaining_on_shelf_l2148_214854

theorem books_remaining_on_shelf (initial_books : Real) (books_taken : Real) 
  (h1 : initial_books = 38.0) (h2 : books_taken = 10.0) : 
  initial_books - books_taken = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_on_shelf_l2148_214854


namespace NUMINAMATH_CALUDE_spinner_final_direction_l2148_214846

/-- Represents the four cardinal directions --/
inductive Direction
| North
| East
| South
| West

/-- Represents a rotation of the spinner --/
structure Rotation :=
  (revolutions : ℚ)
  (clockwise : Bool)

/-- Calculates the final direction after applying a sequence of rotations --/
def finalDirection (initial : Direction) (rotations : List Rotation) : Direction :=
  sorry

/-- The sequence of rotations described in the problem --/
def problemRotations : List Rotation :=
  [⟨7/2, true⟩, ⟨21/4, false⟩, ⟨1/2, true⟩]

theorem spinner_final_direction :
  finalDirection Direction.North problemRotations = Direction.West :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l2148_214846


namespace NUMINAMATH_CALUDE_circle_intersects_unit_distance_l2148_214832

/-- A circle in a 2D Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (y : ℝ) : ℝ := |y|

/-- Predicate for a point being on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Theorem stating the equivalence between the circle properties and the radius range -/
theorem circle_intersects_unit_distance (c : Circle) :
  (c.center = (3, -5)) →
  (∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ onCircle c p1 ∧ onCircle c p2 ∧ 
    distanceFromXAxis p1.2 = 1 ∧ distanceFromXAxis p2.2 = 1) ↔
  (4 < c.radius ∧ c.radius < 6) :=
sorry

end NUMINAMATH_CALUDE_circle_intersects_unit_distance_l2148_214832


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l2148_214891

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 42 → x + x/2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l2148_214891


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2148_214848

/-- Represents the number of villages in each category -/
structure VillageCategories where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample sizes for each category -/
structure SampleSizes where
  first : ℕ
  secondAndThird : ℕ

/-- Checks if the sampling is stratified -/
def isStratifiedSampling (vc : VillageCategories) (ss : SampleSizes) : Prop :=
  (ss.first : ℚ) / vc.first = (ss.first + ss.secondAndThird : ℚ) / vc.total

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (vc : VillageCategories) (ss : SampleSizes) :
  vc.total = 300 →
  vc.first = 60 →
  vc.second = 100 →
  vc.third = vc.total - vc.first - vc.second →
  ss.first = 3 →
  isStratifiedSampling vc ss →
  ss.secondAndThird = 12 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2148_214848


namespace NUMINAMATH_CALUDE_monday_grading_percentage_l2148_214810

/-- The percentage of exams graded on Monday -/
def monday_percentage : ℝ := 40

/-- The total number of exams -/
def total_exams : ℕ := 120

/-- The percentage of remaining exams graded on Tuesday -/
def tuesday_percentage : ℝ := 75

/-- The number of exams left to grade after Tuesday -/
def exams_left : ℕ := 12

theorem monday_grading_percentage :
  monday_percentage = 40 ∧
  (total_exams : ℝ) - (monday_percentage / 100) * total_exams -
    (tuesday_percentage / 100) * ((100 - monday_percentage) / 100 * total_exams) = exams_left :=
by sorry

end NUMINAMATH_CALUDE_monday_grading_percentage_l2148_214810


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l2148_214833

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_set (s : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ s = {x | a ≤ x ∧ x ≤ b ∧ is_odd x ∧ ∀ y, a ≤ y ∧ y < x → is_odd y}

def median (s : Set ℤ) : ℤ := sorry

theorem smallest_integer_in_set (s : Set ℤ) :
  is_consecutive_odd_set s ∧ median s = 153 ∧ (∃ x ∈ s, ∀ y ∈ s, y ≤ x) ∧ 167 ∈ s →
  (∃ z ∈ s, ∀ w ∈ s, z ≤ w) ∧ 139 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l2148_214833


namespace NUMINAMATH_CALUDE_smallest_square_partition_l2148_214835

theorem smallest_square_partition : ∃ (n : ℕ), 
  (40 ∣ n) ∧ 
  (49 ∣ n) ∧ 
  (∀ (m : ℕ), (40 ∣ m) ∧ (49 ∣ m) → m ≥ n) ∧
  n = 1960 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l2148_214835


namespace NUMINAMATH_CALUDE_range_of_f_l2148_214877

-- Define the function f
def f : ℝ → ℝ := λ x => x^2 - 10*x - 4

-- State the theorem
theorem range_of_f :
  ∀ t : ℝ, t ∈ Set.Ioo 0 8 → ∃ y : ℝ, y ∈ Set.Icc (-29) (-4) ∧ y = f t ∧
  ∀ z : ℝ, z ∈ Set.Icc (-29) (-4) → ∃ s : ℝ, s ∈ Set.Ioo 0 8 ∧ z = f s :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2148_214877


namespace NUMINAMATH_CALUDE_game_specific_outcome_l2148_214869

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) 
                     (alex_wins : ℕ) 
                     (mel_wins : ℕ) 
                     (chelsea_wins : ℕ) : ℚ :=
  sorry

theorem game_specific_outcome : 
  game_probability 7 (3/5) 2 4 2 1 = 18144/1125 := by sorry

end NUMINAMATH_CALUDE_game_specific_outcome_l2148_214869


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2148_214803

/-- The perimeter of a trapezoid JKLM with given coordinates is 34 units. -/
theorem trapezoid_perimeter : 
  let j : ℝ × ℝ := (-2, -4)
  let k : ℝ × ℝ := (-2, 1)
  let l : ℝ × ℝ := (6, 7)
  let m : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist j k + dist k l + dist l m + dist m j
  perimeter = 34 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2148_214803


namespace NUMINAMATH_CALUDE_red_cars_count_l2148_214885

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 28 := by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l2148_214885


namespace NUMINAMATH_CALUDE_function_characterization_l2148_214876

theorem function_characterization (f : ℕ → ℕ) 
  (h_increasing : ∀ x y : ℕ, x ≤ y → f x ≤ f y)
  (h_square1 : ∀ n : ℕ, ∃ k : ℕ, f n + n + 1 = k^2)
  (h_square2 : ∀ n : ℕ, ∃ k : ℕ, f (f n) - f n = k^2) :
  ∀ x : ℕ, f x = x^2 + x :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2148_214876


namespace NUMINAMATH_CALUDE_sea_glass_ratio_l2148_214895

/-- Sea glass collection problem -/
theorem sea_glass_ratio : 
  ∀ (blanche_green blanche_red rose_red rose_blue dorothy_total : ℕ),
  blanche_green = 12 →
  blanche_red = 3 →
  rose_red = 9 →
  rose_blue = 11 →
  dorothy_total = 57 →
  ∃ (dorothy_red dorothy_blue : ℕ),
    dorothy_blue = 3 * rose_blue ∧
    dorothy_red + dorothy_blue = dorothy_total ∧
    2 * (blanche_red + rose_red) = dorothy_red :=
by sorry

end NUMINAMATH_CALUDE_sea_glass_ratio_l2148_214895


namespace NUMINAMATH_CALUDE_max_hands_in_dance_l2148_214829

/-- Represents a Martian participating in the dance --/
structure Martian :=
  (hands : Nat)
  (hands_le_three : hands ≤ 3)

/-- Represents the dance configuration --/
structure DanceConfiguration :=
  (participants : List Martian)
  (participant_count_le_seven : participants.length ≤ 7)

/-- Calculates the total number of hands in a dance configuration --/
def total_hands (config : DanceConfiguration) : Nat :=
  config.participants.foldl (λ sum martian => sum + martian.hands) 0

/-- Theorem: The maximum number of hands involved in the dance is 20 --/
theorem max_hands_in_dance :
  ∃ (config : DanceConfiguration),
    (∀ (other_config : DanceConfiguration),
      total_hands other_config ≤ total_hands config) ∧
    total_hands config = 20 ∧
    total_hands config % 2 = 0 :=
  sorry

end NUMINAMATH_CALUDE_max_hands_in_dance_l2148_214829


namespace NUMINAMATH_CALUDE_total_tabs_is_322_l2148_214872

def browser1_windows : ℕ := 4
def browser1_tabs_per_window : ℕ := 10

def browser2_windows : ℕ := 5
def browser2_tabs_per_window : ℕ := 12

def browser3_windows : ℕ := 6
def browser3_tabs_per_window : ℕ := 15

def browser4_windows : ℕ := browser1_windows
def browser4_tabs_per_window : ℕ := browser1_tabs_per_window + 5

def browser5_windows : ℕ := browser2_windows
def browser5_tabs_per_window : ℕ := browser2_tabs_per_window - 2

def browser6_windows : ℕ := 3
def browser6_tabs_per_window : ℕ := browser3_tabs_per_window / 2

def total_tabs : ℕ := 
  browser1_windows * browser1_tabs_per_window +
  browser2_windows * browser2_tabs_per_window +
  browser3_windows * browser3_tabs_per_window +
  browser4_windows * browser4_tabs_per_window +
  browser5_windows * browser5_tabs_per_window +
  browser6_windows * browser6_tabs_per_window

theorem total_tabs_is_322 : total_tabs = 322 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_322_l2148_214872


namespace NUMINAMATH_CALUDE_cost_price_proof_l2148_214874

/-- The cost price of a ball in rupees -/
def cost_price : ℝ := 90

/-- The number of balls sold -/
def balls_sold : ℕ := 13

/-- The selling price of all balls in rupees -/
def selling_price : ℝ := 720

/-- The number of balls whose cost price equals the loss -/
def loss_balls : ℕ := 5

theorem cost_price_proof :
  cost_price * balls_sold = selling_price + cost_price * loss_balls :=
sorry

end NUMINAMATH_CALUDE_cost_price_proof_l2148_214874


namespace NUMINAMATH_CALUDE_hotel_discount_l2148_214837

/-- Calculate the discount for a hotel stay given the number of nights, cost per night, and total amount paid. -/
theorem hotel_discount (nights : ℕ) (cost_per_night : ℕ) (total_paid : ℕ) : 
  nights = 3 → cost_per_night = 250 → total_paid = 650 → 
  nights * cost_per_night - total_paid = 100 := by
sorry

end NUMINAMATH_CALUDE_hotel_discount_l2148_214837


namespace NUMINAMATH_CALUDE_tax_growth_equation_l2148_214817

/-- Represents the average annual growth rate of taxes paid by a company over two years -/
def average_annual_growth_rate (initial_tax : ℝ) (final_tax : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_tax * (1 + x)^years = final_tax

/-- Theorem stating that the equation 40(1+x)^2 = 48.4 correctly represents the average annual growth rate -/
theorem tax_growth_equation (x : ℝ) :
  average_annual_growth_rate 40 48.4 2 x ↔ 40 * (1 + x)^2 = 48.4 :=
by sorry

end NUMINAMATH_CALUDE_tax_growth_equation_l2148_214817


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2148_214816

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    if a₄/a₈ = 2/3, then S₇/S₁₅ = 14/45 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n))
  (h_ratio : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2148_214816


namespace NUMINAMATH_CALUDE_min_value_expression_l2148_214811

theorem min_value_expression (m n : ℝ) (h1 : m > 1) (h2 : n > 0) (h3 : m^2 - 3*m + n = 0) :
  ∃ (min_val : ℝ), min_val = 9/2 ∧ ∀ (x : ℝ), (4/(m-1) + m/n) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2148_214811


namespace NUMINAMATH_CALUDE_butterfly_black_dots_l2148_214820

/-- The number of black dots per butterfly -/
def blackDotsPerButterfly (totalButterflies : ℕ) (totalBlackDots : ℕ) : ℕ :=
  totalBlackDots / totalButterflies

/-- Theorem stating that each butterfly has 12 black dots -/
theorem butterfly_black_dots :
  blackDotsPerButterfly 397 4764 = 12 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_black_dots_l2148_214820


namespace NUMINAMATH_CALUDE_problem_statement_l2148_214882

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2148_214882
