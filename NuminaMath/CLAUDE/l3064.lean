import Mathlib

namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3064_306457

/-- Represents the profit sharing ratio between two investors -/
structure ProfitRatio where
  praveen : ℕ
  hari : ℕ

/-- Calculates the profit sharing ratio based on investments and durations -/
def calculate_profit_ratio (praveen_investment : ℕ) (praveen_duration : ℕ) 
                           (hari_investment : ℕ) (hari_duration : ℕ) : ProfitRatio :=
  let praveen_contribution := praveen_investment * praveen_duration
  let hari_contribution := hari_investment * hari_duration
  let gcd := Nat.gcd praveen_contribution hari_contribution
  { praveen := praveen_contribution / gcd
  , hari := hari_contribution / gcd }

/-- Theorem stating the profit sharing ratio for the given problem -/
theorem profit_sharing_ratio : 
  calculate_profit_ratio 3220 12 8280 7 = ProfitRatio.mk 2 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l3064_306457


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3064_306471

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_terms b n + b (n + 1)

theorem sequence_sum_theorem (a b c : ℕ → ℝ) (d : ℝ) :
  d > 0 ∧
  arithmetic_sequence a d ∧
  a 2 + a 5 = 12 ∧
  a 2 * a 5 = 27 ∧
  b 1 = 3 ∧
  (∀ n : ℕ, b (n + 1) = 2 * sum_of_terms b n + 3) ∧
  (∀ n : ℕ, c n = a n / b n) →
  ∀ n : ℕ, sum_of_terms c n = 1 - (n + 1 : ℝ) / 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3064_306471


namespace NUMINAMATH_CALUDE_two_hour_charge_l3064_306436

/-- Represents the pricing scheme of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 35

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 350) : totalCharge pricing 2 = 161 := by
  sorry

#check two_hour_charge

end NUMINAMATH_CALUDE_two_hour_charge_l3064_306436


namespace NUMINAMATH_CALUDE_friendship_fraction_l3064_306448

theorem friendship_fraction :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 →
    (1 : ℚ) / 3 * y = (2 : ℚ) / 5 * x →
    ((1 : ℚ) / 3 * y + (2 : ℚ) / 5 * x) / (x + y : ℚ) = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_friendship_fraction_l3064_306448


namespace NUMINAMATH_CALUDE_isabel_homework_problems_l3064_306489

/-- Given the number of pages for math and reading homework, and the number of problems per page,
    calculate the total number of problems to complete. -/
def total_problems (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Prove that Isabel's total number of homework problems is 30. -/
theorem isabel_homework_problems :
  total_problems 2 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problems_l3064_306489


namespace NUMINAMATH_CALUDE_smallest_positive_m_for_symmetry_l3064_306411

open Real

/-- The smallest positive value of m for which the function 
    y = sin(2(x-m) + π/6) is symmetric about the y-axis -/
theorem smallest_positive_m_for_symmetry : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x : ℝ), sin (2*(x-m) + π/6) = sin (2*(-x-m) + π/6)) ∧
  (∀ (m' : ℝ), 0 < m' ∧ m' < m → 
    ∃ (x : ℝ), sin (2*(x-m') + π/6) ≠ sin (2*(-x-m') + π/6)) ∧
  m = π/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_m_for_symmetry_l3064_306411


namespace NUMINAMATH_CALUDE_quadratic_maximum_l3064_306434

-- Define the quadratic function
def quadratic (p r s x : ℝ) : ℝ := x^2 + p*x + r + s

-- State the theorem
theorem quadratic_maximum (p s : ℝ) :
  let r : ℝ := 10 - s + p^2/4
  (∀ x, quadratic p r s x ≤ 10) ∧ 
  (quadratic p r s (-p/2) = 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l3064_306434


namespace NUMINAMATH_CALUDE_clock_cost_price_l3064_306404

theorem clock_cost_price (total_clocks : ℕ) (clocks_profit1 : ℕ) (clocks_profit2 : ℕ)
  (profit1 : ℚ) (profit2 : ℚ) (uniform_profit : ℚ) (revenue_difference : ℚ) :
  total_clocks = 200 →
  clocks_profit1 = 80 →
  clocks_profit2 = 120 →
  profit1 = 5 / 25 →
  profit2 = 7 / 25 →
  uniform_profit = 6 / 25 →
  revenue_difference = 200 →
  ∃ (cost_price : ℚ),
    cost_price * (clocks_profit1 * (1 + profit1) + clocks_profit2 * (1 + profit2)) -
    cost_price * (total_clocks * (1 + uniform_profit)) = revenue_difference ∧
    cost_price = 125 :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l3064_306404


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3064_306456

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3064_306456


namespace NUMINAMATH_CALUDE_cross_figure_perimeter_l3064_306444

/-- A cross-shaped figure formed by five identical squares -/
structure CrossFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 125 cm² -/
  total_area_eq : 5 * side_length^2 = 125

/-- The perimeter of a cross-shaped figure -/
def perimeter (f : CrossFigure) : ℝ :=
  16 * f.side_length

/-- Theorem: The perimeter of the cross-shaped figure is 80 cm -/
theorem cross_figure_perimeter (f : CrossFigure) : perimeter f = 80 := by
  sorry

end NUMINAMATH_CALUDE_cross_figure_perimeter_l3064_306444


namespace NUMINAMATH_CALUDE_min_value_expression_l3064_306488

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 16) / Real.sqrt (x - 4) ≥ 4 * Real.sqrt 5 ∧
  (∃ x₀ : ℝ, x₀ > 4 ∧ (x₀ + 16) / Real.sqrt (x₀ - 4) = 4 * Real.sqrt 5 ∧ x₀ = 24) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3064_306488


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3064_306496

theorem arithmetic_square_root_of_four :
  ∃ (x : ℝ), x > 0 ∧ x * x = 4 ∧ ∀ y : ℝ, y > 0 ∧ y * y = 4 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3064_306496


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3064_306409

/-- The perimeter of a semi-circle with radius 7 cm is 7π + 14 cm. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 7 → (π * r + 2 * r) = 7 * π + 14 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3064_306409


namespace NUMINAMATH_CALUDE_reflection_sum_l3064_306490

-- Define the reflection line
structure ReflectionLine where
  m : ℝ
  b : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the reflection operation
def reflect (p : Point) (l : ReflectionLine) : Point :=
  sorry

-- Theorem statement
theorem reflection_sum (l : ReflectionLine) :
  reflect ⟨2, -2⟩ l = ⟨-4, 4⟩ → l.m + l.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l3064_306490


namespace NUMINAMATH_CALUDE_special_line_equation_l3064_306492

/-- A line passing through two points -/
structure Line where
  p : ℝ × ℝ
  q : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (point : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * point.1 + eq.b * point.2 + eq.c = 0

/-- The line l passing through P(x, y) and Q(4x + 2y, x + 3y) -/
def specialLine (x y : ℝ) : Line :=
  { p := (x, y)
    q := (4*x + 2*y, x + 3*y) }

/-- The possible equations for the special line -/
def possibleEquations : List LineEquation :=
  [{ a := 1, b := -1, c := 0 },  -- x - y = 0
   { a := 1, b := -2, c := 0 }]  -- x - 2y = 0

theorem special_line_equation (x y : ℝ) :
  ∃ (eq : LineEquation), eq ∈ possibleEquations ∧
    satisfiesEquation (specialLine x y).p eq ∧
    satisfiesEquation (specialLine x y).q eq :=
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l3064_306492


namespace NUMINAMATH_CALUDE_binomial_60_3_l3064_306400

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3064_306400


namespace NUMINAMATH_CALUDE_consecutive_arithmetic_geometric_equality_l3064_306425

theorem consecutive_arithmetic_geometric_equality (a b c : ℝ) : 
  (∃ r : ℝ, b - a = r ∧ c - b = r) →  -- arithmetic progression condition
  (∃ q : ℝ, b / a = q ∧ c / b = q) →  -- geometric progression condition
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_consecutive_arithmetic_geometric_equality_l3064_306425


namespace NUMINAMATH_CALUDE_triangle_problem_l3064_306438

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  c = 1 →
  b * Real.sin A = a * Real.sin C →
  0 < A →
  A < Real.pi →
  -- Conclusions
  b = 1 ∧
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 → x * Real.sin y ≤ 1/2 * z * Real.sin x) →
  (∃ x y : Real, x > 0 → y > 0 → 1/2 * c * b * Real.sin x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3064_306438


namespace NUMINAMATH_CALUDE_frying_time_correct_l3064_306465

/-- Calculates the minimum time required to fry n pancakes -/
def min_frying_time (n : ℕ) : ℕ :=
  if n ≤ 2 then
    4
  else if n % 2 = 0 then
    2 * n
  else
    2 * (n - 1) + 2

theorem frying_time_correct :
  (min_frying_time 3 = 6) ∧ (min_frying_time 2016 = 4032) := by
  sorry

#eval min_frying_time 3
#eval min_frying_time 2016

end NUMINAMATH_CALUDE_frying_time_correct_l3064_306465


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_specific_cube_root_l3064_306427

theorem unique_integer_divisible_by_24_with_specific_cube_root : 
  ∃! n : ℕ+, (∃ k : ℕ, n = 24 * k) ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_specific_cube_root_l3064_306427


namespace NUMINAMATH_CALUDE_odd_square_plus_two_divisor_congruence_l3064_306407

theorem odd_square_plus_two_divisor_congruence (a d : ℤ) : 
  Odd a → a > 0 → d ∣ (a^2 + 2) → d % 8 = 1 ∨ d % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_plus_two_divisor_congruence_l3064_306407


namespace NUMINAMATH_CALUDE_jan_beth_money_difference_l3064_306497

theorem jan_beth_money_difference (beth_money jan_money : ℕ) : 
  beth_money + 35 = 105 →
  beth_money + jan_money = 150 →
  jan_money - beth_money = 10 := by
sorry

end NUMINAMATH_CALUDE_jan_beth_money_difference_l3064_306497


namespace NUMINAMATH_CALUDE_log_equation_proof_l3064_306421

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_equation_proof :
  (ln 5) ^ 2 + ln 2 * ln 50 = 1 := by sorry

end NUMINAMATH_CALUDE_log_equation_proof_l3064_306421


namespace NUMINAMATH_CALUDE_periodic_function_value_l3064_306449

/-- Given a function f(x) = a*sin(π*x + θ) + b*cos(π*x + θ) + 3,
    where a, b, θ are non-zero real numbers, and f(2016) = -1,
    prove that f(2017) = 7. -/
theorem periodic_function_value (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hθ : θ ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + θ) + b * Real.cos (π * x + θ) + 3
  f 2016 = -1 → f 2017 = 7 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l3064_306449


namespace NUMINAMATH_CALUDE_product_representation_l3064_306416

theorem product_representation (a : ℝ) (p : ℕ+) 
  (h1 : 12345 * 6789 = a * (10 : ℝ)^(p : ℝ))
  (h2 : 1 ≤ a ∧ a < 10) :
  p = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_representation_l3064_306416


namespace NUMINAMATH_CALUDE_a_upper_bound_l3064_306452

/-- Given a real number a, we define a function f and its derivative f' --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

/-- We define g as the sum of f and f' --/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f' a x

/-- Main theorem: If there exists x in [1, 3] such that g(x) ≤ 0, then a ≤ 9/4 --/
theorem a_upper_bound (a : ℝ) (h : ∃ x ∈ Set.Icc 1 3, g a x ≤ 0) : a ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l3064_306452


namespace NUMINAMATH_CALUDE_statement_B_only_incorrect_l3064_306486

-- Define the structure for a statistical statement
structure StatisticalStatement where
  label : Char
  content : String
  isCorrect : Bool

-- Define the four statements
def statementA : StatisticalStatement := {
  label := 'A',
  content := "The absolute value of the correlation coefficient approaches 1 as the linear correlation between two random variables strengthens.",
  isCorrect := true
}

def statementB : StatisticalStatement := {
  label := 'B',
  content := "In a three-shot target shooting scenario, \"at least two hits\" and \"exactly one hit\" are complementary events.",
  isCorrect := false
}

def statementC : StatisticalStatement := {
  label := 'C',
  content := "The accuracy of a model fit increases as the band of residual points in a residual plot narrows.",
  isCorrect := true
}

def statementD : StatisticalStatement := {
  label := 'B',
  content := "The variance of a dataset remains unchanged when a constant is added to each data point.",
  isCorrect := true
}

-- Define the list of all statements
def allStatements : List StatisticalStatement := [statementA, statementB, statementC, statementD]

-- Theorem: Statement B is the only incorrect statement
theorem statement_B_only_incorrect :
  ∃! s : StatisticalStatement, s ∈ allStatements ∧ ¬s.isCorrect :=
sorry

end NUMINAMATH_CALUDE_statement_B_only_incorrect_l3064_306486


namespace NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3064_306484

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3064_306484


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3064_306472

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 256 = -9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3064_306472


namespace NUMINAMATH_CALUDE_triangle_problem_l3064_306483

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S₁ S₂ S₃ : ℝ) 
  (h₁ : S₁ - S₂ + S₃ = Real.sqrt 3 / 2)
  (h₂ : Real.sin B = 1 / 3)
  (h₃ : S₁ = Real.sqrt 3 / 4 * a^2)
  (h₄ : S₂ = Real.sqrt 3 / 4 * b^2)
  (h₅ : S₃ = Real.sqrt 3 / 4 * c^2)
  (h₆ : a > 0 ∧ b > 0 ∧ c > 0)
  (h₇ : 0 < A ∧ A < π)
  (h₈ : 0 < B ∧ B < π)
  (h₉ : 0 < C ∧ C < π)
  (h₁₀ : A + B + C = π) :
  (∃ (S : ℝ), S = Real.sqrt 2 / 8 ∧ S = 1/2 * a * c * Real.sin B) ∧
  (Real.sin A * Real.sin C = Real.sqrt 2 / 3 → b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3064_306483


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3064_306412

/-- The equation of a reflected ray given an incident ray and a reflecting line. -/
theorem reflected_ray_equation (x y : ℝ) :
  (y = 2 * x + 1) →  -- incident ray
  (y = x) →          -- reflecting line
  (x - 2 * y - 1 = 0) -- reflected ray
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3064_306412


namespace NUMINAMATH_CALUDE_runner_speed_l3064_306437

/-- Given a runner who runs 5 days a week, 1.5 hours each day, and covers 60 miles in a week,
    prove that their running speed is 8 mph. -/
theorem runner_speed (days_per_week : ℕ) (hours_per_day : ℝ) (miles_per_week : ℝ) :
  days_per_week = 5 →
  hours_per_day = 1.5 →
  miles_per_week = 60 →
  miles_per_week / (days_per_week * hours_per_day) = 8 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_l3064_306437


namespace NUMINAMATH_CALUDE_spherical_segment_angle_l3064_306410

theorem spherical_segment_angle (r : ℝ) (α : ℝ) (h : r > 0) :
  (2 * π * r * (r * (1 - Real.cos (α / 2))) + π * (r * Real.sin (α / 2))^2 = π * r^2) →
  (Real.cos (α / 2))^2 + 2 * Real.cos (α / 2) - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_spherical_segment_angle_l3064_306410


namespace NUMINAMATH_CALUDE_triangle_side_length_l3064_306459

theorem triangle_side_length (A B : Real) (a b : Real) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 1 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3064_306459


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l3064_306415

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ (r s : ℤ), x^2 + b*x + 1764 = (x + r) * (x + s)) ∧ 
  (∀ (b' : ℕ), b' < b → ¬∃ (r s : ℤ), x^2 + b'*x + 1764 = (x + r) * (x + s)) → 
  b = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l3064_306415


namespace NUMINAMATH_CALUDE_kevin_max_sum_l3064_306414

def kevin_process (S : Finset ℕ) : Finset ℕ :=
  sorry

theorem kevin_max_sum :
  let initial_set : Finset ℕ := Finset.range 15
  let final_set := kevin_process initial_set
  Finset.sum final_set id = 360864 :=
sorry

end NUMINAMATH_CALUDE_kevin_max_sum_l3064_306414


namespace NUMINAMATH_CALUDE_ratio_bounds_l3064_306432

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_bounds_l3064_306432


namespace NUMINAMATH_CALUDE_inequality_condition_l3064_306455

theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_inequality_condition_l3064_306455


namespace NUMINAMATH_CALUDE_largest_good_set_size_l3064_306402

/-- A set of positive integers is "good" if there exists a coloring with 2008 colors
    of all positive integers such that no number in the set is the sum of two
    different positive integers of the same color. -/
def isGoodSet (S : Set ℕ) : Prop :=
  ∃ (f : ℕ → Fin 2008), ∀ n ∈ S, ∀ x y : ℕ, x ≠ y → f x = f y → n ≠ x + y

/-- The set S(a, t) = {a+1, a+2, ..., a+t} for a positive integer a and natural number t. -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

/-- The largest value of t for which S(a, t) is "good" for any positive integer a is 4014. -/
theorem largest_good_set_size :
  (∀ a : ℕ, a > 0 → isGoodSet (S a 4014)) ∧
  (∀ t : ℕ, t > 4014 → ∃ a : ℕ, a > 0 ∧ ¬isGoodSet (S a t)) :=
sorry

end NUMINAMATH_CALUDE_largest_good_set_size_l3064_306402


namespace NUMINAMATH_CALUDE_veggies_expense_correct_l3064_306433

/-- Calculates the amount spent on veggies given the total amount brought,
    expenses on other items, and the amount left after shopping. -/
def amount_spent_on_veggies (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                             (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ) : ℕ :=
  total_brought - (meat_expense + chicken_expense + eggs_expense + dog_food_expense + amount_left)

/-- Proves that the amount Trisha spent on veggies is correct given the problem conditions. -/
theorem veggies_expense_correct (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                                 (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ)
                                 (h1 : total_brought = 167)
                                 (h2 : meat_expense = 17)
                                 (h3 : chicken_expense = 22)
                                 (h4 : eggs_expense = 5)
                                 (h5 : dog_food_expense = 45)
                                 (h6 : amount_left = 35) :
  amount_spent_on_veggies total_brought meat_expense chicken_expense eggs_expense dog_food_expense amount_left = 43 :=
by
  sorry

#eval amount_spent_on_veggies 167 17 22 5 45 35

end NUMINAMATH_CALUDE_veggies_expense_correct_l3064_306433


namespace NUMINAMATH_CALUDE_jerry_games_won_l3064_306446

theorem jerry_games_won (ken dave jerry : ℕ) 
  (h1 : ken = dave + 5)
  (h2 : dave = jerry + 3)
  (h3 : ken + dave + jerry = 32) : 
  jerry = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_games_won_l3064_306446


namespace NUMINAMATH_CALUDE_gmat_score_difference_l3064_306473

theorem gmat_score_difference (x y : ℝ) (h1 : x > y) (h2 : x / y = 4) :
  x - y = 3 * y := by
sorry

end NUMINAMATH_CALUDE_gmat_score_difference_l3064_306473


namespace NUMINAMATH_CALUDE_skew_lines_equivalent_l3064_306422

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary fields to represent a plane

-- Define what it means for two lines to be parallel
def parallel (a b : Line3D) : Prop :=
  sorry

-- Define what it means for a line to be a subset of a plane
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define what it means for two lines to intersect
def intersect (a b : Line3D) : Prop :=
  sorry

-- Define skew lines according to the first definition
def skew_def1 (a b : Line3D) : Prop :=
  ¬(intersect a b) ∧ ¬(parallel a b)

-- Define skew lines according to the second definition
def skew_def2 (a b : Line3D) : Prop :=
  ¬∃ (p : Plane3D), line_subset_plane a p ∧ line_subset_plane b p

-- Theorem stating the equivalence of the two definitions
theorem skew_lines_equivalent (a b : Line3D) :
  skew_def1 a b ↔ skew_def2 a b :=
sorry

end NUMINAMATH_CALUDE_skew_lines_equivalent_l3064_306422


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3064_306466

theorem fraction_sum_equality (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3064_306466


namespace NUMINAMATH_CALUDE_smallest_cube_on_unit_cube_surface_l3064_306450

-- Define a cube type
structure Cube where
  edgeLength : ℝ

-- Define the unit cube K1
def K1 : Cube := ⟨1⟩

-- Define the property that a cube's vertices lie on the surface of K1
def verticesOnSurfaceOfK1 (c : Cube) : Prop := sorry

-- Theorem statement
theorem smallest_cube_on_unit_cube_surface :
  ∃ (minCube : Cube), 
    verticesOnSurfaceOfK1 minCube ∧ 
    minCube.edgeLength = 1 / Real.sqrt 2 ∧
    ∀ (c : Cube), verticesOnSurfaceOfK1 c → c.edgeLength ≥ minCube.edgeLength :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_on_unit_cube_surface_l3064_306450


namespace NUMINAMATH_CALUDE_distance_propositions_l3064_306475

-- Define the distance measure
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Define propositions
def proposition1 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x ∈ Set.Icc x₁ x₂ ∧ y ∈ Set.Icc y₁ y₂) →
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂

def proposition2 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x - x₁) * (x₂ - x) + (y - y₁) * (y₂ - y) = 0 →
  (distance x₁ y₁ x y)^2 + (distance x y x₂ y₂)^2 = (distance x₁ y₁ x₂ y₂)^2

def proposition3 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  distance x₁ y₁ x y + distance x y x₂ y₂ > distance x₁ y₁ x₂ y₂

-- Theorem statement
theorem distance_propositions :
  (∀ x₁ y₁ x₂ y₂ x y, proposition1 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition2 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition3 x₁ y₁ x₂ y₂ x y) :=
sorry

end NUMINAMATH_CALUDE_distance_propositions_l3064_306475


namespace NUMINAMATH_CALUDE_cubic_extrema_l3064_306499

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x^2 + 4 * x - 7

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * x + 4

/-- The discriminant of f' -/
def Δ (a : ℝ) : ℝ := (-4)^2 - 4 * 3 * a * 4

theorem cubic_extrema (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) ↔ 
  (a < 1/3 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_l3064_306499


namespace NUMINAMATH_CALUDE_line_symmetry_l3064_306464

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-axis -/
def y_axis : Line := { a := 1, b := 0, c := 0 }

/-- Check if two lines are symmetric with respect to a given line -/
def symmetric_wrt (l1 l2 axis : Line) : Prop :=
  -- Definition of symmetry with respect to a line
  sorry

/-- The original line x - y + 1 = 0 -/
def original_line : Line := { a := 1, b := -1, c := 1 }

/-- The proposed symmetric line x + y - 1 = 0 -/
def symmetric_line : Line := { a := 1, b := 1, c := -1 }

theorem line_symmetry : 
  symmetric_wrt original_line symmetric_line y_axis := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3064_306464


namespace NUMINAMATH_CALUDE_square_window_side_length_l3064_306408

/-- Given three rectangles with perimeters 8, 10, and 12 that form a square window,
    prove that the side length of the square window is 4. -/
theorem square_window_side_length 
  (a b c : ℝ) 
  (h1 : 2*b + 2*c = 8)   -- perimeter of bottom-left rectangle
  (h2 : 2*(a - b) + 2*a = 10) -- perimeter of top rectangle
  (h3 : 2*b + 2*(a - c) = 12) -- perimeter of right rectangle
  : a = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_window_side_length_l3064_306408


namespace NUMINAMATH_CALUDE_min_additional_marbles_l3064_306460

/-- The number of friends Tom has -/
def num_friends : ℕ := 12

/-- The initial number of marbles Tom has -/
def initial_marbles : ℕ := 40

/-- The sum of consecutive integers from 1 to n -/
def sum_consecutive (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional marbles Tom needs -/
theorem min_additional_marbles : 
  sum_consecutive num_friends - initial_marbles = 38 := by sorry

end NUMINAMATH_CALUDE_min_additional_marbles_l3064_306460


namespace NUMINAMATH_CALUDE_max_sum_problem_l3064_306403

def is_valid_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_problem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D)
  (h_integer : ∃ k : ℕ, k * (C + D) = A + B + 1)
  (h_max : ∀ A' B' C' D' : ℕ, 
    A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
    is_valid_digit A' ∧ is_valid_digit B' ∧ is_valid_digit C' ∧ is_valid_digit D' →
    (∃ k' : ℕ, k' * (C' + D') = A' + B' + 1) →
    (A' + B' + 1) / (C' + D') ≤ (A + B + 1) / (C + D)) :
  A + B + 1 = 18 := by
sorry

end NUMINAMATH_CALUDE_max_sum_problem_l3064_306403


namespace NUMINAMATH_CALUDE_grid_solution_l3064_306470

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions in the grid --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

/-- The theorem to prove --/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : contains_all_numbers g) 
  (h3 : given_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grid_solution_l3064_306470


namespace NUMINAMATH_CALUDE_sum_of_coefficients_quadratic_l3064_306440

theorem sum_of_coefficients_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4 ∧ a + b + c = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_quadratic_l3064_306440


namespace NUMINAMATH_CALUDE_two_students_all_pets_l3064_306451

/-- Represents the number of students in each section of the Venn diagram --/
structure PetOwnership where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  other : ℕ
  no_pets : ℕ
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_and_cats : ℕ
  cats_and_other : ℕ
  dogs_and_other : ℕ
  all_three : ℕ

/-- Theorem stating that 2 students have all three types of pets --/
theorem two_students_all_pets (po : PetOwnership) : po.all_three = 2 :=
  by
  have h1 : po.total = 40 := sorry
  have h2 : po.dogs = po.total / 2 := sorry
  have h3 : po.cats = po.total * 5 / 16 := sorry
  have h4 : po.other = 8 := sorry
  have h5 : po.no_pets = 7 := sorry
  have h6 : po.dogs_only = 12 := sorry
  have h7 : po.cats_only = 3 := sorry
  have h8 : po.other_only = 2 := sorry

  have total_pet_owners : po.dogs_only + po.cats_only + po.other_only + 
    po.dogs_and_cats + po.cats_and_other + po.dogs_and_other + po.all_three = 
    po.total - po.no_pets := sorry

  have dog_owners : po.dogs_only + po.dogs_and_cats + po.dogs_and_other + po.all_three = 
    po.dogs := sorry

  have cat_owners : po.cats_only + po.dogs_and_cats + po.cats_and_other + po.all_three = 
    po.cats := sorry

  have other_pet_owners : po.other_only + po.cats_and_other + po.dogs_and_other + 
    po.all_three = po.other := sorry

  sorry


end NUMINAMATH_CALUDE_two_students_all_pets_l3064_306451


namespace NUMINAMATH_CALUDE_shanghai_masters_matches_l3064_306418

/-- Represents the Shanghai Masters tennis tournament structure -/
structure ShangHaiMasters where
  totalPlayers : Nat
  groupCount : Nat
  playersPerGroup : Nat
  advancingPerGroup : Nat

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the Shanghai Masters tournament -/
def totalMatches (tournament : ShangHaiMasters) : Nat :=
  let groupMatches := tournament.groupCount * roundRobinMatches tournament.playersPerGroup
  let knockoutMatches := tournament.groupCount * tournament.advancingPerGroup / 2
  let finalMatches := 2
  groupMatches + knockoutMatches + finalMatches

/-- Theorem stating that the total number of matches in the Shanghai Masters is 16 -/
theorem shanghai_masters_matches :
  ∃ (tournament : ShangHaiMasters),
    tournament.totalPlayers = 8 ∧
    tournament.groupCount = 2 ∧
    tournament.playersPerGroup = 4 ∧
    tournament.advancingPerGroup = 2 ∧
    totalMatches tournament = 16 := by
  sorry


end NUMINAMATH_CALUDE_shanghai_masters_matches_l3064_306418


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3064_306480

/-- 
Given an arithmetic sequence with 30 terms, first term 4, and last term 88,
prove that the 8th term is equal to 676/29.
-/
theorem arithmetic_sequence_8th_term 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (n : ℕ) 
  (h₁ : a₁ = 4) 
  (h₂ : aₙ = 88) 
  (h₃ : n = 30) : 
  a₁ + 7 * ((aₙ - a₁) / (n - 1)) = 676 / 29 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3064_306480


namespace NUMINAMATH_CALUDE_guaranteed_scores_theorem_l3064_306420

/-- Represents a player in the card game -/
inductive Player : Type
| First : Player
| Second : Player

/-- The card game with given conditions -/
structure CardGame where
  first_player_cards : Finset Nat
  second_player_cards : Finset Nat
  total_turns : Nat

/-- Define the game with the given conditions -/
def game : CardGame :=
  { first_player_cards := Finset.range 1000 |>.image (fun n => 2 * n + 2),
    second_player_cards := Finset.range 1001 |>.image (fun n => 2 * n + 1),
    total_turns := 1000 }

/-- The score a player can guarantee for themselves -/
def guaranteed_score (player : Player) (g : CardGame) : Nat :=
  match player with
  | Player.First => g.total_turns - 1
  | Player.Second => 1

/-- Theorem stating the guaranteed scores for both players -/
theorem guaranteed_scores_theorem (g : CardGame) :
  (guaranteed_score Player.First g = 999) ∧
  (guaranteed_score Player.Second g = 1) :=
sorry

end NUMINAMATH_CALUDE_guaranteed_scores_theorem_l3064_306420


namespace NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3064_306454

/-- The weight difference between cats and puppies -/
theorem cat_puppy_weight_difference :
  let puppy_weights : List ℝ := [6.5, 7.2, 8, 9.5]
  let cat_weight : ℝ := 2.8
  let num_cats : ℕ := 16
  (num_cats : ℝ) * cat_weight - puppy_weights.sum = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_cat_puppy_weight_difference_l3064_306454


namespace NUMINAMATH_CALUDE_bottle_display_sum_l3064_306481

/-- Represents a triangular bottle display -/
structure BottleDisplay where
  firstRow : ℕ
  commonDiff : ℕ
  lastRow : ℕ

/-- Calculates the total number of bottles in the display -/
def totalBottles (display : BottleDisplay) : ℕ :=
  let n := (display.lastRow - display.firstRow) / display.commonDiff + 1
  n * (display.firstRow + display.lastRow) / 2

/-- Theorem stating the total number of bottles in the specific display -/
theorem bottle_display_sum :
  let display : BottleDisplay := ⟨3, 3, 30⟩
  totalBottles display = 165 := by
  sorry

end NUMINAMATH_CALUDE_bottle_display_sum_l3064_306481


namespace NUMINAMATH_CALUDE_three_people_on_third_stop_l3064_306423

/-- Represents the number of people on a bus and its changes at stops -/
structure BusRide where
  initial : ℕ
  first_off : ℕ
  second_off : ℕ
  second_on : ℕ
  third_off : ℕ
  final : ℕ

/-- Calculates the number of people who got on at the third stop -/
def people_on_third_stop (ride : BusRide) : ℕ :=
  ride.final - (ride.initial - ride.first_off - ride.second_off + ride.second_on - ride.third_off)

/-- Theorem stating that 3 people got on at the third stop -/
theorem three_people_on_third_stop (ride : BusRide) 
  (h_initial : ride.initial = 50)
  (h_first_off : ride.first_off = 15)
  (h_second_off : ride.second_off = 8)
  (h_second_on : ride.second_on = 2)
  (h_third_off : ride.third_off = 4)
  (h_final : ride.final = 28) :
  people_on_third_stop ride = 3 := by
  sorry

#eval people_on_third_stop { initial := 50, first_off := 15, second_off := 8, second_on := 2, third_off := 4, final := 28 }

end NUMINAMATH_CALUDE_three_people_on_third_stop_l3064_306423


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3064_306426

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a > 0

def q (a : ℝ) : Prop := a < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3064_306426


namespace NUMINAMATH_CALUDE_train_length_calculation_l3064_306458

/-- Calculates the length of a train given the speeds of a jogger and the train, 
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that given the specific conditions, the train length is 120 meters. -/
theorem train_length_calculation : 
  train_length (9 * (1000 / 3600)) (45 * (1000 / 3600)) 250 37 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3064_306458


namespace NUMINAMATH_CALUDE_book_selling_price_l3064_306462

theorem book_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 250 →
  profit_percentage = 20 →
  selling_price = cost_price * (1 + profit_percentage / 100) →
  selling_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l3064_306462


namespace NUMINAMATH_CALUDE_point_on_curve_with_perpendicular_tangent_l3064_306474

/-- The function f(x) = x^4 - x -/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 - 1

theorem point_on_curve_with_perpendicular_tangent :
  ∀ x y : ℝ,
  f x = y →                           -- Point P(x, y) is on the curve f(x) = x^4 - x
  (f' x) * (-3) = 1 →                 -- Tangent line is perpendicular to x + 3y = 0
  x = 1 ∧ y = 0 := by                 -- Then P has coordinates (1, 0)
sorry

end NUMINAMATH_CALUDE_point_on_curve_with_perpendicular_tangent_l3064_306474


namespace NUMINAMATH_CALUDE_class_average_score_l3064_306477

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_avg = 80 →
  group2_avg = 60 →
  (group1_students * group1_avg + group2_students * group2_avg) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l3064_306477


namespace NUMINAMATH_CALUDE_ratio_when_a_is_20_percent_more_than_b_l3064_306485

theorem ratio_when_a_is_20_percent_more_than_b (A B : ℝ) (h : A = 1.2 * B) : A / B = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_when_a_is_20_percent_more_than_b_l3064_306485


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3064_306435

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_one :
  (f' 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3064_306435


namespace NUMINAMATH_CALUDE_wall_height_calculation_l3064_306467

/-- Calculates the height of a wall given brick dimensions and number of bricks --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
                                (wall_length wall_width : ℝ)
                                (num_bricks : ℝ) :
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  wall_length = 200 →
  wall_width = 2 →
  num_bricks = 72.72727272727273 →
  ∃ (wall_height : ℝ), abs (wall_height - 436.3636363636364) < 0.0001 :=
by
  sorry

#check wall_height_calculation

end NUMINAMATH_CALUDE_wall_height_calculation_l3064_306467


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_4_l3064_306478

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def six_digit_number (B : ℕ) : ℕ := 400000 + 5000 + 784 + B * 10000

theorem largest_B_divisible_by_4 :
  ∀ B : ℕ, B ≤ 9 →
    (is_divisible_by_4 (six_digit_number B)) →
    (∀ C : ℕ, C ≤ 9 → is_divisible_by_4 (six_digit_number C) → C ≤ B) →
    B = 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_4_l3064_306478


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3064_306463

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 ≥ x) ↔ (∀ x : ℝ, x^2 < x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3064_306463


namespace NUMINAMATH_CALUDE_inequality_and_range_proof_fraction_comparison_l3064_306482

theorem inequality_and_range_proof :
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) →
    x ∈ Set.Icc (-2) 2) :=
sorry

theorem fraction_comparison :
  ∀ (a b : ℝ), a ∈ Set.Ioo 0 1 → b ∈ Set.Ioo 0 1 →
    1 / (a * b) + 1 > 1 / a + 1 / b :=
sorry

end NUMINAMATH_CALUDE_inequality_and_range_proof_fraction_comparison_l3064_306482


namespace NUMINAMATH_CALUDE_total_stickers_l3064_306453

/-- Given 25 stickers on each page and 35 pages of stickers, 
    the total number of stickers is 875. -/
theorem total_stickers (stickers_per_page pages : ℕ) : 
  stickers_per_page = 25 → pages = 35 → stickers_per_page * pages = 875 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l3064_306453


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3064_306498

def N : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 8; 4, 6, -2; -9, -3, 5]

def i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
def j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
def k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]

theorem matrix_N_satisfies_conditions :
  N * i = !![3; 4; -9] ∧
  N * j = !![1; 6; -3] ∧
  N * k = !![8; -2; 5] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3064_306498


namespace NUMINAMATH_CALUDE_vector_colinearity_l3064_306441

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (2, 4)

def colinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 * w.2 = t * v.2 * w.1

theorem vector_colinearity (k : ℝ) :
  colinear (a.1 + k * b.1, a.2 + k * b.2) c →
  k = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_vector_colinearity_l3064_306441


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3064_306429

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : B a ⊆ A → a ∈ ({1, -1, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3064_306429


namespace NUMINAMATH_CALUDE_solve_for_y_l3064_306445

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 12) (h2 : x = 6) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3064_306445


namespace NUMINAMATH_CALUDE_sum_of_divisors_180_l3064_306476

def sumOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_180 : sumOfDivisors 180 = 546 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_180_l3064_306476


namespace NUMINAMATH_CALUDE_geometric_progression_b_equals_four_l3064_306419

-- Define a geometric progression
def is_geometric_progression (seq : Fin 5 → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 4, seq (i + 1) = seq i * q

-- State the theorem
theorem geometric_progression_b_equals_four
  (seq : Fin 5 → ℝ)
  (h_gp : is_geometric_progression seq)
  (h_first : seq 0 = 1)
  (h_last : seq 4 = 16) :
  seq 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_b_equals_four_l3064_306419


namespace NUMINAMATH_CALUDE_order_cost_is_43_l3064_306479

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of fries in dollars -/
def fries_cost : ℕ := 2

/-- The number of sandwiches ordered -/
def num_sandwiches : ℕ := 3

/-- The number of sodas ordered -/
def num_sodas : ℕ := 7

/-- The number of fries ordered -/
def num_fries : ℕ := 5

/-- The total cost of the order -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas + fries_cost * num_fries

theorem order_cost_is_43 : total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_order_cost_is_43_l3064_306479


namespace NUMINAMATH_CALUDE_routes_in_3x3_grid_l3064_306442

/-- The number of different routes in a 3x3 grid from top-left to bottom-right -/
def numRoutes : ℕ := 20

/-- The size of the grid -/
def gridSize : ℕ := 3

/-- The total number of moves required to reach the destination -/
def totalMoves : ℕ := gridSize * 2

/-- The number of moves in one direction (either right or down) -/
def movesInOneDirection : ℕ := gridSize

theorem routes_in_3x3_grid :
  numRoutes = Nat.choose totalMoves movesInOneDirection := by sorry

end NUMINAMATH_CALUDE_routes_in_3x3_grid_l3064_306442


namespace NUMINAMATH_CALUDE_cos_1275_degrees_l3064_306493

theorem cos_1275_degrees :
  Real.cos (1275 * π / 180) = -(Real.sqrt 2 + Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_1275_degrees_l3064_306493


namespace NUMINAMATH_CALUDE_student_arrangement_l3064_306469

theorem student_arrangement (n : ℕ) (h : n = 5) :
  let total_arrangements := n.factorial
  let a_left_arrangements := 2 * (n - 1).factorial
  let a_left_b_right_arrangements := (n - 2).factorial
  let valid_arrangements := total_arrangements - a_left_arrangements + a_left_b_right_arrangements
  valid_arrangements = 78 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l3064_306469


namespace NUMINAMATH_CALUDE_village_population_l3064_306439

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 60 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 38400 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3064_306439


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l3064_306406

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 7 →
  james + 10 = 3 * (louise - 3) →
  james + louise = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l3064_306406


namespace NUMINAMATH_CALUDE_house_savings_l3064_306424

theorem house_savings (total_savings : ℕ) (years : ℕ) (people : ℕ) : 
  total_savings = 108000 → 
  years = 3 → 
  people = 2 → 
  (total_savings / (years * 12)) / people = 1500 := by
sorry

end NUMINAMATH_CALUDE_house_savings_l3064_306424


namespace NUMINAMATH_CALUDE_mark_friends_percentage_l3064_306428

/-- Calculates the percentage of friends kept initially -/
def friendsKeptPercentage (initialFriends : ℕ) (finalFriends : ℕ) (responseRate : ℚ) : ℚ :=
  let keptPercentage : ℚ := (2 * finalFriends - initialFriends : ℚ) / initialFriends
  keptPercentage * 100

/-- Proves that the percentage of friends Mark kept initially is 40% -/
theorem mark_friends_percentage :
  friendsKeptPercentage 100 70 (1/2) = 40 := by
  sorry

#eval friendsKeptPercentage 100 70 (1/2)

end NUMINAMATH_CALUDE_mark_friends_percentage_l3064_306428


namespace NUMINAMATH_CALUDE_linear_system_elimination_l3064_306461

theorem linear_system_elimination (x y : ℝ) : 
  (6 * x - 5 * y = 3) → 
  (3 * x + y = -15) → 
  (5 * (3 * x + y) + (6 * x - 5 * y) = 21 * x) ∧ 
  (5 * (-15) + 3 = -72) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_elimination_l3064_306461


namespace NUMINAMATH_CALUDE_difference_of_squares_l3064_306431

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3064_306431


namespace NUMINAMATH_CALUDE_square_difference_l3064_306413

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3064_306413


namespace NUMINAMATH_CALUDE_worker_savings_fraction_l3064_306443

/-- A worker saves a constant fraction of her constant monthly take-home pay. -/
structure Worker where
  /-- Monthly take-home pay -/
  P : ℝ
  /-- Fraction of monthly take-home pay saved -/
  f : ℝ
  /-- Monthly take-home pay is positive -/
  P_pos : P > 0
  /-- Savings fraction is between 0 and 1 -/
  f_range : 0 ≤ f ∧ f ≤ 1

/-- The theorem stating that if a worker's yearly savings equals 8 times
    her monthly non-savings, then she saves 2/5 of her income. -/
theorem worker_savings_fraction (w : Worker) 
    (h : 12 * w.f * w.P = 8 * (1 - w.f) * w.P) : 
    w.f = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_fraction_l3064_306443


namespace NUMINAMATH_CALUDE_range_of_k_l3064_306401

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

theorem range_of_k (k : ℝ) : A ⊇ B k ↔ -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3064_306401


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3064_306494

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ -a * x * (x + y)) ↔ -2 ≤ a ∧ a ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3064_306494


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3064_306468

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  ∃ k : ℕ, p + q + r + s = 60 * (2 * k + 1) :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3064_306468


namespace NUMINAMATH_CALUDE_river_speed_theorem_l3064_306430

/-- Represents the equation for a ship traveling upstream and downstream -/
def river_equation (s v d1 d2 : ℝ) : Prop :=
  d1 / (s + v) = d2 / (s - v)

/-- Theorem stating that the river equation holds for the given conditions -/
theorem river_speed_theorem (s v d1 d2 : ℝ) 
  (h_s : s > 0)
  (h_v : 0 < v ∧ v < s)
  (h_d1 : d1 > 0)
  (h_d2 : d2 > 0)
  (h_s_still : s = 30)
  (h_d1 : d1 = 144)
  (h_d2 : d2 = 96) :
  river_equation s v d1 d2 :=
sorry

end NUMINAMATH_CALUDE_river_speed_theorem_l3064_306430


namespace NUMINAMATH_CALUDE_average_volume_of_three_cubes_l3064_306495

theorem average_volume_of_three_cubes (a b c : ℕ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  (a^3 + b^3 + c^3) / 3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_average_volume_of_three_cubes_l3064_306495


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3064_306417

theorem simplify_nested_roots (x : ℝ) :
  (((x^16)^(1/8))^(1/4))^3 * (((x^16)^(1/4))^(1/8))^5 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3064_306417


namespace NUMINAMATH_CALUDE_blanket_price_problem_l3064_306447

theorem blanket_price_problem (unknown_rate : ℕ) : 
  (3 * 100 + 1 * 150 + 2 * unknown_rate) / 6 = 150 → unknown_rate = 225 := by
  sorry

end NUMINAMATH_CALUDE_blanket_price_problem_l3064_306447


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3064_306487

theorem a_greater_than_b (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3064_306487


namespace NUMINAMATH_CALUDE_right_triangle_legs_l3064_306405

/-- A right triangle with specific median and altitude properties -/
structure RightTriangle where
  -- The length of the median from the right angle vertex
  median : ℝ
  -- The length of the altitude from the right angle vertex
  altitude : ℝ
  -- Condition that the median is 5
  median_is_five : median = 5
  -- Condition that the altitude is 4
  altitude_is_four : altitude = 4

/-- The legs of a right triangle -/
structure TriangleLegs where
  -- The length of the first leg
  leg1 : ℝ
  -- The length of the second leg
  leg2 : ℝ

/-- Theorem stating the legs of the triangle given the median and altitude -/
theorem right_triangle_legs (t : RightTriangle) : 
  ∃ (legs : TriangleLegs), legs.leg1 = 2 * Real.sqrt 5 ∧ legs.leg2 = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l3064_306405


namespace NUMINAMATH_CALUDE_larger_integer_proof_l3064_306491

theorem larger_integer_proof (a b : ℕ+) : 
  (a : ℕ) + 3 = (b : ℕ) → a * b = 88 → b = 11 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l3064_306491
