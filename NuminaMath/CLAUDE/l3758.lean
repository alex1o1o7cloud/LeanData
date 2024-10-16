import Mathlib

namespace NUMINAMATH_CALUDE_g_monotone_decreasing_l3758_375873

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the condition for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) :
  (∀ x < a / 3, g' a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_l3758_375873


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3758_375839

open Real

theorem least_positive_t_for_geometric_progression : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π/2 → 
    (∃ (r : ℝ), r > 0 ∧
      arcsin (sin α) * r = arcsin (sin (2*α)) ∧
      arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
      arcsin (sin (5*α)) * r = arcsin (sin (t*α)))) ∧
  (∀ (t' : ℝ), 0 < t' → t' < t →
    ¬(∀ (α : ℝ), 0 < α → α < π/2 → 
      (∃ (r : ℝ), r > 0 ∧
        arcsin (sin α) * r = arcsin (sin (2*α)) ∧
        arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
        arcsin (sin (5*α)) * r = arcsin (sin (t'*α))))) ∧
  t = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3758_375839


namespace NUMINAMATH_CALUDE_rectangle_dimensions_area_l3758_375830

theorem rectangle_dimensions_area (x : ℝ) : 
  (2*x - 3 > 0) → 
  (3*x + 4 > 0) → 
  (2*x - 3) * (3*x + 4) = 14*x - 6 → 
  x = (5 + Real.sqrt 41) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_area_l3758_375830


namespace NUMINAMATH_CALUDE_ten_points_chords_l3758_375822

/-- The number of different chords that can be drawn by connecting any two of n points on a circle. -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting any two of ten points on a circle is 45. -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_points_chords_l3758_375822


namespace NUMINAMATH_CALUDE_cube_split_contains_31_l3758_375868

def split_cube (m : ℕ) : List ℕ :=
  let start := 2 * m * m - 2 * m + 1
  List.range m |>.map (fun i => start + 2 * i)

theorem cube_split_contains_31 (m : ℕ) (h1 : m > 1) :
  31 ∈ split_cube m → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_contains_31_l3758_375868


namespace NUMINAMATH_CALUDE_snow_probability_l3758_375855

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l3758_375855


namespace NUMINAMATH_CALUDE_range_of_f_l3758_375831

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The range of f is [-1, +∞) --/
theorem range_of_f :
  Set.range f = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3758_375831


namespace NUMINAMATH_CALUDE_certain_number_proof_l3758_375805

theorem certain_number_proof (N x : ℝ) (h1 : N / (1 + 3 / x) = 1) (h2 : x = 1) : N = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3758_375805


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3758_375898

theorem fixed_point_on_line (m : ℝ) : (2*m - 1)*2 + (m + 3)*(-3) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3758_375898


namespace NUMINAMATH_CALUDE_initial_roses_l3758_375886

theorem initial_roses (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 4)
  (h2 : added = 25)
  (h3 : final = 23)
  (h4 : initial - thrown_away + added = final) : initial = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_roses_l3758_375886


namespace NUMINAMATH_CALUDE_find_m_value_l3758_375893

theorem find_m_value (m : ℚ) : 
  (∃ (x y : ℚ), m * x - y = 4 ∧ x = 4 ∧ y = 3) → m = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l3758_375893


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3758_375801

theorem inequality_and_equality_condition (x y z : ℝ) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 ∧
  (x^2 + y^4 + z^6 = x*y^2 + y^2*z^3 + x*z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3758_375801


namespace NUMINAMATH_CALUDE_omelet_distribution_l3758_375863

theorem omelet_distribution (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 36 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 := by
sorry

end NUMINAMATH_CALUDE_omelet_distribution_l3758_375863


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3758_375823

/-- Given a polynomial p(x) such that p(2) = 7 and p(5) = 11,
    prove that the remainder when p(x) is divided by (x-2)(x-5) is (4/3)x + (13/3) -/
theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 11) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + (4/3 * x + 13/3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3758_375823


namespace NUMINAMATH_CALUDE_book_length_ratio_l3758_375878

/- Define the variables -/
def starting_age : ℕ := 6
def starting_book_length : ℕ := 8
def current_book_length : ℕ := 480

/- Define the book length at twice the starting age -/
def book_length_twice_starting_age : ℕ := starting_book_length * 5

/- Define the book length 8 years after twice the starting age -/
def book_length_8_years_after : ℕ := book_length_twice_starting_age * 3

/- Theorem: The ratio of current book length to the book length 8 years after twice the starting age is 4:1 -/
theorem book_length_ratio :
  current_book_length / book_length_8_years_after = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_length_ratio_l3758_375878


namespace NUMINAMATH_CALUDE_sum_of_expressions_l3758_375871

def replace_asterisks (n : ℕ) : ℕ := 2^(n-1)

theorem sum_of_expressions : 
  (replace_asterisks 6) = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_expressions_l3758_375871


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3758_375841

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 3 ∧ x₁^2 - 2*x₁ = 2) ∧ 
  (x₂ = 1 - Real.sqrt 3 ∧ x₂^2 - 2*x₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3758_375841


namespace NUMINAMATH_CALUDE_quadratic_solution_l3758_375882

theorem quadratic_solution (b : ℝ) : 
  ((-5 : ℝ)^2 + b * (-5) - 45 = 0) → b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3758_375882


namespace NUMINAMATH_CALUDE_range_of_x_l3758_375803

theorem range_of_x (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) →
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3758_375803


namespace NUMINAMATH_CALUDE_x_squared_in_set_l3758_375812

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l3758_375812


namespace NUMINAMATH_CALUDE_jacob_wage_is_6_l3758_375884

-- Define the given conditions
def jake_earnings_multiplier : ℚ := 3
def jake_total_earnings : ℚ := 720
def work_days : ℕ := 5
def hours_per_day : ℕ := 8

-- Define Jake's hourly wage
def jake_hourly_wage : ℚ := jake_total_earnings / (work_days * hours_per_day)

-- Define Jacob's hourly wage
def jacob_hourly_wage : ℚ := jake_hourly_wage / jake_earnings_multiplier

-- Theorem to prove
theorem jacob_wage_is_6 : jacob_hourly_wage = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacob_wage_is_6_l3758_375884


namespace NUMINAMATH_CALUDE_pencils_left_l3758_375883

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took out -/
def pencils_taken : ℕ := 22

/-- The number of pencils Dan returned -/
def pencils_returned : ℕ := 5

/-- Theorem: The number of pencils left in the drawer is 17 -/
theorem pencils_left : initial_pencils - (pencils_taken - pencils_returned) = 17 := by
  sorry

#eval initial_pencils - (pencils_taken - pencils_returned)

end NUMINAMATH_CALUDE_pencils_left_l3758_375883


namespace NUMINAMATH_CALUDE_sock_combinations_l3758_375889

theorem sock_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  Nat.choose n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_sock_combinations_l3758_375889


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3758_375859

theorem proposition_equivalence (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) ↔ (∀ x, x ∉ B → x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3758_375859


namespace NUMINAMATH_CALUDE_prob_three_pass_min_students_scheme_A_l3758_375876

/-- Represents the two testing schemes -/
inductive Scheme
| A
| B

/-- Represents a student -/
structure Student where
  name : String
  scheme : Scheme

/-- Probability of passing for each scheme -/
def passProbability (s : Scheme) : ℚ :=
  match s with
  | Scheme.A => 2/3
  | Scheme.B => 1/2

/-- Group of students participating in the test -/
def testGroup : List Student := [
  ⟨"A", Scheme.A⟩, ⟨"B", Scheme.A⟩, ⟨"C", Scheme.A⟩,
  ⟨"D", Scheme.B⟩, ⟨"E", Scheme.B⟩
]

/-- Calculates the probability of exactly k students passing out of n students -/
def probExactlyKPass (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of exactly three students passing the test is 19/54 -/
theorem prob_three_pass :
  (probExactlyKPass 3 3 (passProbability Scheme.A) *
   probExactlyKPass 2 0 (passProbability Scheme.B)) +
  (probExactlyKPass 3 2 (passProbability Scheme.A) *
   probExactlyKPass 2 1 (passProbability Scheme.B)) +
  (probExactlyKPass 3 1 (passProbability Scheme.A) *
   probExactlyKPass 2 2 (passProbability Scheme.B)) = 19/54 := by
  sorry

/-- Expected number of passing students given n students choose scheme A -/
def expectedPass (n : ℕ) : ℚ := n * (passProbability Scheme.A) + (5 - n) * (passProbability Scheme.B)

/-- Theorem: The minimum number of students choosing scheme A for the expected number
    of passing students to be at least 3 is 3 -/
theorem min_students_scheme_A :
  (∀ m : ℕ, m < 3 → expectedPass m < 3) ∧
  expectedPass 3 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_pass_min_students_scheme_A_l3758_375876


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3758_375870

theorem neither_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∨ q) → ¬(p ∧ q)) ∧ (¬(p ∧ q) → (p ∨ q))) :=
by sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3758_375870


namespace NUMINAMATH_CALUDE_conic_eccentricity_l3758_375862

/-- Given that 1, m, and 9 form a geometric sequence, 
    the eccentricity of the conic section x²/m + y² = 1 is either √6/3 or 2 -/
theorem conic_eccentricity (m : ℝ) : 
  (1 * 9 = m^2) →  -- geometric sequence condition
  (∃ e : ℝ, (e = Real.sqrt 6 / 3 ∨ e = 2) ∧
   ∀ x y : ℝ, x^2 / m + y^2 = 1 → 
   e = if m > 0 
       then Real.sqrt (1 - 1 / m) 
       else Real.sqrt (1 - m) / Real.sqrt (-m)) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l3758_375862


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l3758_375800

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 1400
  let sellingPrice : ℚ := 1330
  percentageLoss costPrice sellingPrice = 5 := by
  sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l3758_375800


namespace NUMINAMATH_CALUDE_smallest_k_for_divisible_sum_of_squares_l3758_375810

/-- The sum of squares from 1 to n -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is divisible by 150 -/
def divisibleBy150 (n : ℕ) : Prop := ∃ m : ℕ, n = 150 * m

theorem smallest_k_for_divisible_sum_of_squares :
  (∀ k : ℕ, 0 < k ∧ k < 100 → ¬(divisibleBy150 (sumOfSquares k))) ∧
  (divisibleBy150 (sumOfSquares 100)) := by
  sorry

#check smallest_k_for_divisible_sum_of_squares

end NUMINAMATH_CALUDE_smallest_k_for_divisible_sum_of_squares_l3758_375810


namespace NUMINAMATH_CALUDE_gary_stickers_l3758_375897

theorem gary_stickers (initial_stickers : ℕ) : 
  (initial_stickers : ℚ) * (2/3) * (3/4) = 36 → initial_stickers = 72 := by
  sorry

end NUMINAMATH_CALUDE_gary_stickers_l3758_375897


namespace NUMINAMATH_CALUDE_no_solution_gcd_lcm_sum_l3758_375825

theorem no_solution_gcd_lcm_sum (x y : ℕ) : 
  Nat.gcd x y + Nat.lcm x y + x + y ≠ 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_gcd_lcm_sum_l3758_375825


namespace NUMINAMATH_CALUDE_power_product_equals_l3758_375815

theorem power_product_equals : (3 : ℕ)^4 * (6 : ℕ)^4 = 104976 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_l3758_375815


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3758_375881

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x + 3/2)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3758_375881


namespace NUMINAMATH_CALUDE_probability_all_co_captains_value_l3758_375811

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def num_teams : Nat := 3

def probability_all_co_captains : ℚ :=
  (1 : ℚ) / num_teams *
  (team_sizes.map (λ n => (co_captains_per_team : ℚ) / (n * (n - 1) * (n - 2)))).sum

theorem probability_all_co_captains_value :
  probability_all_co_captains = 59 / 2520 := by
  sorry

#eval probability_all_co_captains

end NUMINAMATH_CALUDE_probability_all_co_captains_value_l3758_375811


namespace NUMINAMATH_CALUDE_triangle_side_length_l3758_375838

theorem triangle_side_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Side lengths of the triangle
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) -- Side lengths are positive
  (h2 : B = Real.pi / 3) -- B = 60°
  (h3 : (1/2) * a * c * Real.sin B = Real.sqrt 3) -- Area of the triangle is √3
  (h4 : a^2 + c^2 = 3*a*c) -- Given equation
  : b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3758_375838


namespace NUMINAMATH_CALUDE_a_range_l3758_375818

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Define the theorem
theorem a_range (a : ℝ) : (A a ⊆ B) ↔ a ∈ Set.Icc (-2) 2 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3758_375818


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3758_375835

theorem complex_magnitude_example : Complex.abs (11 + 18 * Complex.I + 4 - 3 * Complex.I) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3758_375835


namespace NUMINAMATH_CALUDE_gcd_91_49_l3758_375833

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l3758_375833


namespace NUMINAMATH_CALUDE_q_value_l3758_375860

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- Define the property that the mean of zeros, twice the product of zeros, and sum of coefficients are equal
def property (p q d : ℝ) : Prop :=
  let sum_of_zeros := -p
  let product_of_zeros := -d
  let sum_of_coefficients := 1 + p + q + d
  (sum_of_zeros / 3 = 2 * product_of_zeros) ∧ (sum_of_zeros / 3 = sum_of_coefficients)

-- Theorem statement
theorem q_value (p q d : ℝ) :
  property p q d → Q p q d 0 = 4 → q = -37 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l3758_375860


namespace NUMINAMATH_CALUDE_exactly_one_correct_statement_l3758_375802

-- Define the type for geometric statements
inductive GeometricStatement
  | uniquePerpendicular
  | perpendicularIntersect
  | equalVertical
  | distanceDefinition
  | uniqueParallel

-- Function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.perpendicularIntersect => True
  | _ => False

-- Theorem stating that exactly one statement is correct
theorem exactly_one_correct_statement :
  ∃! (s : GeometricStatement), isCorrect s :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_statement_l3758_375802


namespace NUMINAMATH_CALUDE_runner_lead_l3758_375890

/-- Represents the relative speeds of runners in a race. -/
structure RunnerSpeeds where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The race setup with given conditions. -/
def raceSetup (s : RunnerSpeeds) : Prop :=
  s.b = (5/6) * s.a ∧ s.c = (3/4) * s.a

/-- The theorem statement. -/
theorem runner_lead (s : RunnerSpeeds) (h : raceSetup s) :
  150 - (s.c * (150 / s.a)) = 37.5 := by
  sorry

#check runner_lead

end NUMINAMATH_CALUDE_runner_lead_l3758_375890


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3758_375829

theorem geometric_arithmetic_sequence_ratio 
  (x y z : ℝ) 
  (h_geometric : ∃ q : ℝ, y = x * q ∧ z = y * q) 
  (h_arithmetic : ∃ d : ℝ, y + z = (x + y) + d ∧ z + x = (y + z) + d) :
  ∃ q : ℝ, (y = x * q ∧ z = y * q) ∧ (q = -2 ∨ q = 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3758_375829


namespace NUMINAMATH_CALUDE_flour_to_add_l3758_375885

/-- Given a recipe requiring a total amount of flour and an amount already added,
    calculate the remaining amount of flour to be added. -/
def remaining_flour (total : ℕ) (added : ℕ) : ℕ :=
  total - added

theorem flour_to_add : remaining_flour 10 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l3758_375885


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3758_375848

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3758_375848


namespace NUMINAMATH_CALUDE_triangle_max_area_l3758_375826

noncomputable def f (x : ℝ) : ℝ := Real.cos x^4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x^4

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) (h1 : f A = 1) (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π) (h5 : A + B + C = π) (h6 : a > 0 ∧ b > 0 ∧ c > 0) (h7 : a / Real.sin A = b / Real.sin B) (h8 : b / Real.sin B = c / Real.sin C) (h9 : (b^2 + c^2 + 2 * b * c * Real.cos A) / 4 = 7) : 
  (1/2) * b * c * Real.sin A ≤ 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3758_375826


namespace NUMINAMATH_CALUDE_problem_solution_l3758_375892

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem conditions
def problem_conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (p q r : ℕ), 
    Real.sqrt (log10 a) = p ∧
    Real.sqrt (log10 b) = q ∧
    log10 (Real.sqrt (a * b^2)) = r ∧
    p + q + r = 150

-- State the theorem
theorem problem_solution (a b : ℝ) : 
  problem_conditions a b → a^2 * b^3 = 10^443 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3758_375892


namespace NUMINAMATH_CALUDE_man_pants_count_l3758_375852

theorem man_pants_count (t_shirts : ℕ) (total_outfits : ℕ) (pants : ℕ) : 
  t_shirts = 8 → total_outfits = 72 → total_outfits = t_shirts * pants → pants = 9 := by
sorry

end NUMINAMATH_CALUDE_man_pants_count_l3758_375852


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l3758_375821

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l3758_375821


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3758_375842

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ α β : ℝ, (α + β = 6) ∧ (α * β = 8) ∧ (α ≠ β → (α - β)^2 = 36 - 4*8)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3758_375842


namespace NUMINAMATH_CALUDE_billy_candy_boxes_l3758_375874

/-- Given that Billy bought boxes of candy with 3 pieces per box and has a total of 21 pieces,
    prove that he bought 7 boxes. -/
theorem billy_candy_boxes : 
  ∀ (boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ),
    pieces_per_box = 3 →
    total_pieces = 21 →
    boxes * pieces_per_box = total_pieces →
    boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_billy_candy_boxes_l3758_375874


namespace NUMINAMATH_CALUDE_product_odd_even_is_odd_l3758_375888

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem product_odd_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry


end NUMINAMATH_CALUDE_product_odd_even_is_odd_l3758_375888


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3758_375877

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3758_375877


namespace NUMINAMATH_CALUDE_angle_difference_l3758_375844

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α)
  (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β)
  (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_difference_l3758_375844


namespace NUMINAMATH_CALUDE_refrigerator_price_l3758_375827

theorem refrigerator_price (P : ℝ) 
  (h1 : 1.18 * P = 18880) 
  (h2 : 0.8 * P + 125 + 250 = 13175) : 
  0.8 * P + 125 + 250 = 13175 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l3758_375827


namespace NUMINAMATH_CALUDE_modulus_of_pure_imaginary_z_l3758_375849

/-- If z = (x^2 - 1) + (x - 1)i where x is a real number and z is a pure imaginary number, then |z| = 2 -/
theorem modulus_of_pure_imaginary_z (x : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (x^2 - 1) (x - 1))
  (h2 : z.re = 0) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_pure_imaginary_z_l3758_375849


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3758_375819

/-- The line equation x cos θ + y sin θ = 1 is tangent to the circle x² + y² = 1 -/
theorem line_tangent_to_circle :
  ∀ θ : ℝ, 
  (∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 → x^2 + y^2 = 1) ∧
  (∃ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 ∧ x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3758_375819


namespace NUMINAMATH_CALUDE_new_person_age_l3758_375845

theorem new_person_age (T : ℕ) : 
  T > 0 →  -- Ensure total age is positive
  (T / 10 : ℚ) - ((T - 48 + 18) / 10 : ℚ) = 3 →
  18 = 18 := by sorry

end NUMINAMATH_CALUDE_new_person_age_l3758_375845


namespace NUMINAMATH_CALUDE_season_games_count_l3758_375847

/-- The number of teams in the sports conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the sports conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The total number of games in a complete season -/
def total_games : ℕ := 296

theorem season_games_count :
  total_teams = num_divisions * teams_per_division ∧
  (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * num_divisions +
  (teams_per_division * teams_per_division * inter_division_games) = total_games := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l3758_375847


namespace NUMINAMATH_CALUDE_books_read_in_week_l3758_375832

/-- The number of books Mrs. Hilt reads per day -/
def books_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mrs. Hilt reads 14 books in one week -/
theorem books_read_in_week : books_per_day * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_books_read_in_week_l3758_375832


namespace NUMINAMATH_CALUDE_total_cost_price_l3758_375834

theorem total_cost_price (watch_price bracelet_price necklace_price : ℕ) 
  (hw : watch_price = 144)
  (hb : bracelet_price = 250)
  (hn : necklace_price = 190) :
  watch_price + bracelet_price + necklace_price = 584 := by
  sorry

#check total_cost_price

end NUMINAMATH_CALUDE_total_cost_price_l3758_375834


namespace NUMINAMATH_CALUDE_hat_code_is_312_l3758_375895

def code_to_digit (c : Char) : Fin 6 :=
  match c with
  | 'M' => 0
  | 'A' => 1
  | 'T' => 2
  | 'H' => 3
  | 'I' => 4
  | 'S' => 5
  | _ => 0  -- Default case, should not occur in our problem

theorem hat_code_is_312 : 
  (code_to_digit 'H') * 100 + (code_to_digit 'A') * 10 + (code_to_digit 'T') = 312 := by
  sorry

end NUMINAMATH_CALUDE_hat_code_is_312_l3758_375895


namespace NUMINAMATH_CALUDE_wednesday_sites_count_l3758_375880

theorem wednesday_sites_count (monday_sites tuesday_sites : ℕ)
  (monday_avg tuesday_avg wednesday_avg overall_avg : ℚ)
  (h1 : monday_sites = 5)
  (h2 : tuesday_sites = 5)
  (h3 : monday_avg = 7)
  (h4 : tuesday_avg = 5)
  (h5 : wednesday_avg = 8)
  (h6 : overall_avg = 7) :
  ∃ wednesday_sites : ℕ,
    (monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg) /
    (monday_sites + tuesday_sites + wednesday_sites : ℚ) = overall_avg ∧
    wednesday_sites = 10 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sites_count_l3758_375880


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_three_even_one_odd_l3758_375896

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  3 = (digits.filter (λ d => d % 2 = 0)).length ∧
  1 = (digits.filter (λ d => d % 2 = 1)).length

theorem smallest_four_digit_divisible_by_9_three_even_one_odd :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → has_three_even_one_odd n → 1026 ≤ n := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_three_even_one_odd_l3758_375896


namespace NUMINAMATH_CALUDE_hall_volume_proof_l3758_375861

/-- Represents a rectangular wall with a width and height -/
structure RectWall where
  width : ℝ
  height : ℝ

/-- Represents a rectangular hall with length, width, and height -/
structure RectHall where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the area of a rectangular wall -/
def wallArea (w : RectWall) : ℝ := w.width * w.height

/-- Calculate the volume of a rectangular hall -/
def hallVolume (h : RectHall) : ℝ := h.length * h.width * h.height

theorem hall_volume_proof (h : RectHall) 
  (a1 a2 : RectWall) 
  (b1 b2 : RectWall) 
  (c1 c2 : RectWall) :
  h.length = 30 ∧ 
  h.width = 20 ∧ 
  h.height = 10 ∧
  a1.width = a2.width ∧
  b1.height = b2.height ∧
  c1.height = c2.height ∧
  b1.height = h.height ∧
  c1.height = h.height ∧
  wallArea a1 + wallArea a2 = wallArea b1 + wallArea b2 ∧
  wallArea c1 + wallArea c2 = 2 * h.length * h.width ∧
  a1.width + a2.width = h.width ∧
  b1.width + b2.width = h.length ∧
  c1.width + c2.width = h.width →
  hallVolume h = 6000 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_proof_l3758_375861


namespace NUMINAMATH_CALUDE_coin_flips_l3758_375836

/-- The number of times a coin is flipped -/
def n : ℕ := sorry

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads on the first 4 flips and not heads on the last flip -/
def p_event : ℚ := 1/32

theorem coin_flips : 
  p_heads = 1/2 → 
  p_event = (p_heads ^ 4) * ((1 - p_heads) ^ 1) * (p_heads ^ (n - 5)) → 
  n = 9 := by sorry

end NUMINAMATH_CALUDE_coin_flips_l3758_375836


namespace NUMINAMATH_CALUDE_arcsin_of_neg_one_l3758_375858

theorem arcsin_of_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end NUMINAMATH_CALUDE_arcsin_of_neg_one_l3758_375858


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_union_equals_A_iff_l3758_375813

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) - Real.sqrt (x + 2)}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}

-- Statement 1: When a = 2, A ∩ B = {2}
theorem intersection_when_a_is_two :
  A ∩ B 2 = {2} := by sorry

-- Statement 2: A ∪ B = A if and only if a ∈ (2, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_union_equals_A_iff_l3758_375813


namespace NUMINAMATH_CALUDE_cone_volume_in_cylinder_with_spheres_l3758_375846

/-- The volume of a cone inscribed in a cylinder with specific properties --/
theorem cone_volume_in_cylinder_with_spheres (r h : ℝ) : 
  r = 1 → 
  h = 12 / (3 + 2 * Real.sqrt 3) →
  ∃ (cone_volume : ℝ), 
    cone_volume = (2/3) * Real.pi ∧
    cone_volume = (1/3) * Real.pi * r^2 * 1 ∧
    ∃ (sphere_radius : ℝ),
      sphere_radius = 2 * Real.sqrt 3 - 3 ∧
      sphere_radius > 0 ∧
      sphere_radius < r ∧
      h = 2 * sphere_radius :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_in_cylinder_with_spheres_l3758_375846


namespace NUMINAMATH_CALUDE_sum_of_tags_is_1000_l3758_375864

/-- The sum of tagged numbers on four cards W, X, Y, Z -/
def sum_of_tags (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating the sum of tagged numbers is 1000 -/
theorem sum_of_tags_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = x + w →
  z = 400 →
  sum_of_tags w x y z = 1000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_tags_is_1000_l3758_375864


namespace NUMINAMATH_CALUDE_square_area_problem_l3758_375887

theorem square_area_problem (x : ℝ) (h : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l3758_375887


namespace NUMINAMATH_CALUDE_solution_product_l3758_375894

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 14 * r + 48 →
  (s - 3) * (3 * s + 11) = s^2 - 14 * s + 48 →
  r ≠ s →
  (r + 4) * (s + 4) = -226 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l3758_375894


namespace NUMINAMATH_CALUDE_double_earnings_in_ten_days_l3758_375851

/-- Calculates the number of additional days needed to earn twice the current amount --/
def additional_days_to_double_earnings (days_worked : ℕ) (total_earned : ℚ) : ℕ :=
  let daily_rate := total_earned / days_worked
  let target_amount := 2 * total_earned
  let total_days_needed := (target_amount / daily_rate).ceil.toNat
  total_days_needed - days_worked

/-- Theorem stating that for the given conditions, 10 additional days are needed --/
theorem double_earnings_in_ten_days :
  additional_days_to_double_earnings 10 250 = 10 := by
  sorry

#eval additional_days_to_double_earnings 10 250

end NUMINAMATH_CALUDE_double_earnings_in_ten_days_l3758_375851


namespace NUMINAMATH_CALUDE_larger_number_is_23_l3758_375824

theorem larger_number_is_23 (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 40) :
  x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_23_l3758_375824


namespace NUMINAMATH_CALUDE_coordinate_point_A_coordinate_point_B_l3758_375853

-- Definition of a coordinate point
def is_coordinate_point (x y : ℝ) : Prop := 2 * x - y = 1

-- Part 1
theorem coordinate_point_A : 
  ∀ a : ℝ, is_coordinate_point 3 a ↔ a = 5 :=
sorry

-- Part 2
theorem coordinate_point_B :
  ∀ b c : ℕ, (is_coordinate_point (b + c) (b + 5) ∧ b > 0 ∧ c > 0) ↔ 
  ((b = 2 ∧ c = 2) ∨ (b = 4 ∧ c = 1)) :=
sorry

end NUMINAMATH_CALUDE_coordinate_point_A_coordinate_point_B_l3758_375853


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3758_375850

/-- Given a rectangle with width x, length 4x, and area 120 square inches,
    prove that the width is √30 inches and the length is 4√30 inches. -/
theorem rectangle_dimensions (x : ℝ) (h1 : x > 0) (h2 : x * (4 * x) = 120) :
  x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 := by
  sorry

#check rectangle_dimensions

end NUMINAMATH_CALUDE_rectangle_dimensions_l3758_375850


namespace NUMINAMATH_CALUDE_polynomial_sequence_gcd_l3758_375879

/-- A sequence defined by polynomials with positive integer coefficients -/
def PolynomialSequence (p : ℕ → ℕ → ℕ) (a₀ : ℕ) : ℕ → ℕ :=
  fun n => p n a₀

/-- The theorem statement -/
theorem polynomial_sequence_gcd
  (p : ℕ → ℕ → ℕ)
  (h_p : ∀ n x, p n x > 0)
  (a₀ : ℕ)
  (a : ℕ → ℕ)
  (h_a : a = PolynomialSequence p a₀)
  (m k : ℕ) :
  Nat.gcd (a m) (a k) = a (Nat.gcd m k) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sequence_gcd_l3758_375879


namespace NUMINAMATH_CALUDE_inverse_uniqueness_non_commutative_non_unique_inverse_l3758_375816

/-- A binary operation on a type α -/
def BinaryOp (α : Type) := α → α → α

/-- The inverse of a binary operation -/
def InverseOp (α : Type) (op : BinaryOp α) := BinaryOp α

/-- Property of an inverse operation -/
def IsInverse {α : Type} (op : BinaryOp α) (inv : InverseOp α op) : Prop :=
  ∀ a b c : α, op a b = c → inv c b = a ∧ op (inv c b) b = c

/-- Uniqueness of inverse operation -/
theorem inverse_uniqueness {α : Type} (op : BinaryOp α) :
  ∃! inv : InverseOp α op, IsInverse op inv :=
sorry

/-- Non-uniqueness for non-commutative operations -/
theorem non_commutative_non_unique_inverse {α : Type} (op : BinaryOp α) :
  (∃ a b : α, op a b ≠ op b a) →
  ¬∃! inv : InverseOp α op, IsInverse op inv :=
sorry

end NUMINAMATH_CALUDE_inverse_uniqueness_non_commutative_non_unique_inverse_l3758_375816


namespace NUMINAMATH_CALUDE_sequence_theorem_l3758_375804

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0) ∧ (∀ n : ℕ+, a (n + 1) + 1 / (a n) < 2)

theorem sequence_theorem (a : ℕ+ → ℝ) (h : sequence_property a) :
  (∀ n : ℕ+, a (n + 2) < a (n + 1) ∧ a (n + 1) < 2) ∧
  (∀ n : ℕ+, a n > 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l3758_375804


namespace NUMINAMATH_CALUDE_derivative_of_y_l3758_375820

noncomputable def y (x : ℝ) : ℝ := Real.exp (-5 * x + 2)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -5 * Real.exp (-5 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3758_375820


namespace NUMINAMATH_CALUDE_hardcover_non_fiction_count_l3758_375828

/-- Represents the number of books in Thabo's collection -/
def total_books : ℕ := 500

/-- Represents the fraction of fiction books in the collection -/
def fiction_fraction : ℚ := 2/5

/-- Represents the fraction of non-fiction books in the collection -/
def non_fiction_fraction : ℚ := 3/5

/-- Represents the difference between paperback and hardcover non-fiction books -/
def non_fiction_difference : ℕ := 50

/-- Represents the ratio of paperback to hardcover fiction books -/
def fiction_ratio : ℕ := 2

/-- Theorem stating that the number of hardcover non-fiction books is 125 -/
theorem hardcover_non_fiction_count :
  ∃ (hardcover_non_fiction paperback_non_fiction hardcover_fiction paperback_fiction : ℕ),
    hardcover_non_fiction + paperback_non_fiction + hardcover_fiction + paperback_fiction = total_books ∧
    (hardcover_fiction + paperback_fiction : ℚ) = fiction_fraction * total_books ∧
    (hardcover_non_fiction + paperback_non_fiction : ℚ) = non_fiction_fraction * total_books ∧
    paperback_non_fiction = hardcover_non_fiction + non_fiction_difference ∧
    paperback_fiction = fiction_ratio * hardcover_fiction ∧
    hardcover_non_fiction = 125 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_non_fiction_count_l3758_375828


namespace NUMINAMATH_CALUDE_right_triangle_test_l3758_375837

theorem right_triangle_test : 
  -- Option A
  (3 : ℝ)^2 + 4^2 = 5^2 ∧
  -- Option B
  (1 : ℝ)^2 + 2^2 = (Real.sqrt 5)^2 ∧
  -- Option C
  (2 : ℝ)^2 + (2 * Real.sqrt 3)^2 ≠ 3^2 ∧
  -- Option D
  (1 : ℝ)^2 + (Real.sqrt 3)^2 = 2^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_test_l3758_375837


namespace NUMINAMATH_CALUDE_apples_eaten_per_day_l3758_375809

theorem apples_eaten_per_day 
  (initial_apples : ℕ) 
  (remaining_apples : ℕ) 
  (days : ℕ) 
  (h1 : initial_apples = 32) 
  (h2 : remaining_apples = 4) 
  (h3 : days = 7) :
  (initial_apples - remaining_apples) / days = 4 :=
by sorry

end NUMINAMATH_CALUDE_apples_eaten_per_day_l3758_375809


namespace NUMINAMATH_CALUDE_whiteboard_count_per_class_l3758_375891

/-- Given:
  - There are 5 classes in a building block at Oakland High.
  - Each whiteboard needs 20ml of ink for a day's use.
  - Ink costs 50 cents per ml.
  - It costs $100 to use the boards for one day.
Prove that each class uses 10 whiteboards. -/
theorem whiteboard_count_per_class (
  num_classes : ℕ)
  (ink_per_board : ℝ)
  (ink_cost_per_ml : ℝ)
  (total_daily_cost : ℝ)
  (h1 : num_classes = 5)
  (h2 : ink_per_board = 20)
  (h3 : ink_cost_per_ml = 0.5)
  (h4 : total_daily_cost = 100) :
  (total_daily_cost * num_classes) / (ink_per_board * ink_cost_per_ml) / num_classes = 10 := by
  sorry

end NUMINAMATH_CALUDE_whiteboard_count_per_class_l3758_375891


namespace NUMINAMATH_CALUDE_max_points_theorem_l3758_375840

/-- Represents a football tournament with the given conditions -/
structure Tournament where
  teams : Nat
  total_points : Nat
  draw_points : Nat
  win_points : Nat

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  t.teams * (t.teams - 1) / 2

/-- Represents the result of solving the tournament equations -/
structure TournamentResult where
  draws : Nat
  wins : Nat

/-- Solves the tournament equations to find the number of draws and wins -/
def solve_tournament (t : Tournament) : TournamentResult :=
  { draws := 23, wins := 5 }

/-- Calculates the maximum points a single team can obtain -/
def max_points (t : Tournament) (result : TournamentResult) : Nat :=
  (result.wins * t.win_points) + (t.teams - 1 - result.wins) * t.draw_points

/-- The main theorem stating the maximum points obtainable by a single team -/
theorem max_points_theorem (t : Tournament) 
  (h1 : t.teams = 8)
  (h2 : t.total_points = 61)
  (h3 : t.draw_points = 1)
  (h4 : t.win_points = 3) :
  max_points t (solve_tournament t) = 17 := by
  sorry

#eval max_points 
  { teams := 8, total_points := 61, draw_points := 1, win_points := 3 } 
  (solve_tournament { teams := 8, total_points := 61, draw_points := 1, win_points := 3 })

end NUMINAMATH_CALUDE_max_points_theorem_l3758_375840


namespace NUMINAMATH_CALUDE_quadratic_less_than_sqrt_l3758_375856

theorem quadratic_less_than_sqrt (x : ℝ) :
  x^2 - 3*x + 2 < Real.sqrt (x + 4) ↔ 1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_less_than_sqrt_l3758_375856


namespace NUMINAMATH_CALUDE_product_103_97_l3758_375866

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_103_97_l3758_375866


namespace NUMINAMATH_CALUDE_polynomial_from_root_relations_l3758_375857

theorem polynomial_from_root_relations (α β γ : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 44*x - 46 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∃ x₁ x₂ x₃ : ℝ, 
    α = x₁ + x₂ ∧ 
    β = x₁ + x₃ ∧ 
    γ = x₂ + x₃ ∧
    (∀ x, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_from_root_relations_l3758_375857


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3758_375807

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3758_375807


namespace NUMINAMATH_CALUDE_unique_sums_count_l3758_375875

def X : Finset ℕ := {1, 4, 5, 7}
def Y : Finset ℕ := {3, 4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((X.product Y).image (fun p => p.1 + p.2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l3758_375875


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3758_375843

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3758_375843


namespace NUMINAMATH_CALUDE_splittable_point_range_l3758_375872

/-- A function f is splittable at x_0 if f(x_0 + 1) = f(x_0) + f(1) -/
def IsSplittable (f : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  f (x_0 + 1) = f x_0 + f 1

/-- The logarithm function with base 5 -/
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

/-- The function f(x) = log_5(a / (2^x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log5 (a / (2^x + 1))

theorem splittable_point_range (a : ℝ) :
  (a > 0) → (∃ x_0 : ℝ, IsSplittable (f a) x_0) ↔ (3/2 < a ∧ a < 3) := by
  sorry


end NUMINAMATH_CALUDE_splittable_point_range_l3758_375872


namespace NUMINAMATH_CALUDE_edward_booth_tickets_l3758_375806

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth (total_tickets : ℕ) (cost_per_ride : ℕ) (possible_rides : ℕ) : ℕ :=
  total_tickets - (cost_per_ride * possible_rides)

/-- Proof that Edward spent 23 tickets at the 'dunk a clown' booth -/
theorem edward_booth_tickets : 
  tickets_spent_at_booth 79 7 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_edward_booth_tickets_l3758_375806


namespace NUMINAMATH_CALUDE_sum_and_ratio_problem_l3758_375867

theorem sum_and_ratio_problem (x y : ℝ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 500 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_ratio_problem_l3758_375867


namespace NUMINAMATH_CALUDE_total_cost_beef_vegetables_l3758_375814

/-- The total cost of beef and vegetables given their weights and prices -/
theorem total_cost_beef_vegetables 
  (beef_weight : ℝ) 
  (vegetable_weight : ℝ) 
  (vegetable_price : ℝ) 
  (beef_price_multiplier : ℝ) : 
  beef_weight = 4 →
  vegetable_weight = 6 →
  vegetable_price = 2 →
  beef_price_multiplier = 3 →
  beef_weight * (vegetable_price * beef_price_multiplier) + vegetable_weight * vegetable_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_beef_vegetables_l3758_375814


namespace NUMINAMATH_CALUDE_platform_length_l3758_375869

/-- The length of a platform given train parameters -/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_cross : ℝ)
  (h1 : train_length = 450)
  (h2 : train_speed_kmph = 126)
  (h3 : time_to_cross = 20)
  : ∃ (platform_length : ℝ), platform_length = 250 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3758_375869


namespace NUMINAMATH_CALUDE_exist_five_integers_sum_four_is_square_l3758_375808

theorem exist_five_integers_sum_four_is_square : ∃ (a₁ a₂ a₃ a₄ a₅ : ℤ),
  (a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₄ ≠ a₅) ∧
  (∃ n₁ : ℕ, a₂ + a₃ + a₄ + a₅ = n₁^2) ∧
  (∃ n₂ : ℕ, a₁ + a₃ + a₄ + a₅ = n₂^2) ∧
  (∃ n₃ : ℕ, a₁ + a₂ + a₄ + a₅ = n₃^2) ∧
  (∃ n₄ : ℕ, a₁ + a₂ + a₃ + a₅ = n₄^2) ∧
  (∃ n₅ : ℕ, a₁ + a₂ + a₃ + a₄ = n₅^2) :=
by sorry

end NUMINAMATH_CALUDE_exist_five_integers_sum_four_is_square_l3758_375808


namespace NUMINAMATH_CALUDE_parallelogram_height_base_difference_l3758_375865

theorem parallelogram_height_base_difference 
  (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 24) 
  (h_base : base = 4) 
  (h_parallelogram : area = base * height) : 
  height - base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_base_difference_l3758_375865


namespace NUMINAMATH_CALUDE_min_value_theorem_l3758_375854

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  4 * x^2 + 1 / x^3 ≥ 5 ∧
  (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3758_375854


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3758_375817

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3758_375817


namespace NUMINAMATH_CALUDE_fixed_point_of_parabolas_unique_fixed_point_l3758_375899

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabolas (t : ℝ) :
  let f (x : ℝ) := 5 * x^2 + 4 * t * x - 3 * t
  f (3/4) = 45/16 := by sorry

/-- The uniqueness of the fixed point -/
theorem unique_fixed_point (t₁ t₂ : ℝ) (x : ℝ) :
  let f₁ (x : ℝ) := 5 * x^2 + 4 * t₁ * x - 3 * t₁
  let f₂ (x : ℝ) := 5 * x^2 + 4 * t₂ * x - 3 * t₂
  f₁ x = f₂ x → x = 3/4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabolas_unique_fixed_point_l3758_375899
