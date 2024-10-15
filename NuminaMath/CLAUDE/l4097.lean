import Mathlib

namespace NUMINAMATH_CALUDE_kate_wands_proof_l4097_409739

/-- The number of wands Kate bought -/
def total_wands : ℕ := 3

/-- The cost of each wand -/
def wand_cost : ℕ := 60

/-- The selling price of each wand -/
def selling_price : ℕ := wand_cost + 5

/-- The total amount collected from sales -/
def total_collected : ℕ := 130

/-- Kate keeps one wand for herself -/
def kept_wands : ℕ := 1

theorem kate_wands_proof : 
  total_wands = (total_collected / selling_price) + kept_wands :=
by sorry

end NUMINAMATH_CALUDE_kate_wands_proof_l4097_409739


namespace NUMINAMATH_CALUDE_arithmetic_mean_multiplication_l4097_409755

theorem arithmetic_mean_multiplication (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let multiplied_set := List.map (· * 3) original_set
  (List.sum multiplied_set) / 5 = 3 * ((List.sum original_set) / 5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_multiplication_l4097_409755


namespace NUMINAMATH_CALUDE_min_value_theorem_l4097_409725

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / c + (b + c) / a + (c + d) / b ≥ 6 ∧
  ((a + b) / c + (b + c) / a + (c + d) / b = 6 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4097_409725


namespace NUMINAMATH_CALUDE_problem_statement_l4097_409798

theorem problem_statement :
  (∀ a : ℝ, (2 * a + 1) / a ≤ 1 ↔ a ∈ Set.Icc (-1) 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0) ↔ a ∈ Set.Ico (-1/2) (3/2)) ∧
  (∀ a : ℝ, ((2 * a + 1) / a ≤ 1 ∨ (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0)) ↔ a ∈ Set.Ico (-1) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4097_409798


namespace NUMINAMATH_CALUDE_total_days_2010_to_2015_l4097_409794

/-- Calculate the total number of days from 2010 through 2015, inclusive. -/
theorem total_days_2010_to_2015 : 
  let years := 6
  let leap_years := 1
  let common_year_days := 365
  let leap_year_days := 366
  years * common_year_days + leap_years * (leap_year_days - common_year_days) = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2015_l4097_409794


namespace NUMINAMATH_CALUDE_triangle_side_length_l4097_409799

theorem triangle_side_length (A B C : ℝ) (angleA : ℝ) (sideBC sideAB : ℝ) :
  angleA = 2 * Real.pi / 3 →
  sideBC = Real.sqrt 19 →
  sideAB = 2 →
  ∃ (sideAC : ℝ), sideAC = 3 ∧
    sideBC ^ 2 = sideAC ^ 2 + sideAB ^ 2 - 2 * sideAC * sideAB * Real.cos angleA :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4097_409799


namespace NUMINAMATH_CALUDE_exists_continuous_random_variable_point_P_in_plane_ABC_l4097_409784

-- Define a random variable type
def RandomVariable := ℝ → ℝ

-- Define vectors in ℝ³
def AB : Fin 3 → ℝ := ![2, -1, -4]
def AC : Fin 3 → ℝ := ![4, 2, 0]
def AP : Fin 3 → ℝ := ![0, -4, -8]

-- Theorem for the existence of a continuous random variable
theorem exists_continuous_random_variable :
  ∃ (X : RandomVariable), ∀ (a b : ℝ), a < b → ∃ (x : ℝ), a < X x ∧ X x < b :=
sorry

-- Function to check if a point is in a plane defined by three vectors
def is_in_plane (p v1 v2 : Fin 3 → ℝ) : Prop :=
  ∃ (a b : ℝ), p = λ i => a * v1 i + b * v2 i

-- Theorem for point P lying in plane ABC
theorem point_P_in_plane_ABC :
  is_in_plane AP AB AC :=
sorry

end NUMINAMATH_CALUDE_exists_continuous_random_variable_point_P_in_plane_ABC_l4097_409784


namespace NUMINAMATH_CALUDE_cassidy_profit_l4097_409772

/-- Cassidy's bread baking and selling scenario --/
def bread_scenario (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (midday_price : ℚ) (evening_price : ℚ) : Prop :=
  let morning_sold := total_loaves / 3
  let midday_remaining := total_loaves - morning_sold
  let midday_sold := midday_remaining / 2
  let evening_sold := midday_remaining - midday_sold
  let total_revenue := morning_sold * morning_price + midday_sold * midday_price + evening_sold * evening_price
  let total_cost := total_loaves * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 70

/-- Theorem stating Cassidy's profit is $70 --/
theorem cassidy_profit : 
  bread_scenario 60 1 3 2 (3/2) :=
sorry

end NUMINAMATH_CALUDE_cassidy_profit_l4097_409772


namespace NUMINAMATH_CALUDE_farm_animals_count_l4097_409726

/-- Represents a farm with hens, cows, and ducks -/
structure Farm where
  hens : ℕ
  cows : ℕ
  ducks : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows + f.ducks

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows + 2 * f.ducks

/-- Theorem stating the number of cows and the sum of hens and ducks in the farm -/
theorem farm_animals_count (f : Farm) 
  (h1 : total_heads f = 72) 
  (h2 : total_feet f = 212) : 
  f.cows = 34 ∧ f.hens + f.ducks = 38 := by
  sorry


end NUMINAMATH_CALUDE_farm_animals_count_l4097_409726


namespace NUMINAMATH_CALUDE_equation_solution_is_one_point_five_l4097_409756

theorem equation_solution_is_one_point_five :
  ∃! (x : ℝ), x > 0 ∧ (3 + x)^5 = (1 + 3*x)^4 ∧ x = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_is_one_point_five_l4097_409756


namespace NUMINAMATH_CALUDE_brothers_reading_percentage_l4097_409787

theorem brothers_reading_percentage
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (difference : ℕ)
  (h1 : total_books = 20)
  (h2 : peter_percentage = 2/5)
  (h3 : difference = 6)
  : (↑(peter_percentage * total_books - difference) / total_books : ℚ) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_reading_percentage_l4097_409787


namespace NUMINAMATH_CALUDE_parabola_f_value_l4097_409718

/-- A parabola with equation y = dx² + ex + f, vertex at (-1, 3), and passing through (0, 2) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : d * (-1)^2 + e * (-1) + f = 3
  point_condition : d * 0^2 + e * 0 + f = 2

/-- The f-value of the parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_f_value_l4097_409718


namespace NUMINAMATH_CALUDE_a_range_theorem_l4097_409721

/-- Proposition p: The inequality x^2 + 2ax + 4 > 0 holds true for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: The function f(x) = (3-2a)^x is increasing -/
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l4097_409721


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l4097_409723

def linear_function (x : ℝ) : ℝ := -3 * x - 2

theorem not_in_first_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_in_first_quadrant_l4097_409723


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l4097_409701

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def satisfies_condition (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p % 2 = 1 → p < n → is_prime (n - p)

theorem largest_satisfying_number :
  satisfies_condition 10 ∧ ∀ n : ℕ, n > 10 → ¬(satisfies_condition n) :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l4097_409701


namespace NUMINAMATH_CALUDE_min_votes_for_a_to_win_l4097_409751

/-- Represents the minimum number of votes candidate A needs to win the election -/
def min_votes_to_win (total_votes : ℕ) (first_votes : ℕ) (a_votes : ℕ) (b_votes : ℕ) (c_votes : ℕ) : ℕ :=
  let remaining_votes := total_votes - first_votes
  let a_deficit := b_votes - a_votes
  (remaining_votes - a_deficit) / 2 + a_deficit + 1

theorem min_votes_for_a_to_win :
  let total_votes : ℕ := 1500
  let first_votes : ℕ := 1000
  let a_votes : ℕ := 350
  let b_votes : ℕ := 370
  let c_votes : ℕ := 280
  min_votes_to_win total_votes first_votes a_votes b_votes c_votes = 261 := by
  sorry

#eval min_votes_to_win 1500 1000 350 370 280

end NUMINAMATH_CALUDE_min_votes_for_a_to_win_l4097_409751


namespace NUMINAMATH_CALUDE_min_value_of_f_l4097_409705

noncomputable def f (x : ℝ) := (Real.exp x - 1)^2 + (Real.exp 1 - x - 1)^2

theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4097_409705


namespace NUMINAMATH_CALUDE_number_operation_proof_l4097_409757

theorem number_operation_proof (N : ℤ) : 
  N % 5 = 0 → N / 5 = 4 → ((N - 10) * 3) - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_proof_l4097_409757


namespace NUMINAMATH_CALUDE_least_value_quadratic_l4097_409782

theorem least_value_quadratic (y : ℝ) :
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l4097_409782


namespace NUMINAMATH_CALUDE_largest_multiple_of_13_less_than_neg_124_l4097_409703

theorem largest_multiple_of_13_less_than_neg_124 :
  ∀ n : ℤ, n * 13 < -124 → n * 13 ≤ -130 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_13_less_than_neg_124_l4097_409703


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l4097_409717

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l4097_409717


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4097_409747

/-- Given a quadratic function f(x) = x^2 + bx - 5 with symmetric axis x = 2,
    prove that the solutions to f(x) = 2x - 13 are x₁ = 2 and x₂ = 4. -/
theorem quadratic_equation_solution (b : ℝ) :
  (∀ x, x^2 + b*x - 5 = x^2 + b*x - 5) →  -- f(x) is a well-defined function
  (-b/2 = 2) →                            -- symmetric axis is x = 2
  (∃ x₁ x₂, x₁ = 2 ∧ x₂ = 4 ∧
    (∀ x, x^2 + b*x - 5 = 2*x - 13 ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4097_409747


namespace NUMINAMATH_CALUDE_complex_number_problem_l4097_409780

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 2 + 3 * Complex.I) :
  α = 6 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l4097_409780


namespace NUMINAMATH_CALUDE_number_division_problem_l4097_409786

theorem number_division_problem :
  ∃ (N p q : ℝ),
    N / p = 8 ∧
    N / q = 18 ∧
    p - q = 0.20833333333333334 ∧
    N = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l4097_409786


namespace NUMINAMATH_CALUDE_probability_two_pairs_l4097_409715

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The number of ways to choose which two dice will form each pair -/
def numWaysToFormPairs : ℕ := Nat.choose numDice 2

/-- The probability of rolling exactly two pairs of matching numbers
    when four standard six-sided dice are tossed simultaneously -/
theorem probability_two_pairs :
  (numWaysToFormPairs : ℚ) * (1 : ℚ) * (1 / numSides : ℚ) * ((numSides - 1) / numSides : ℚ) * (1 / numSides : ℚ) = 5 / 36 := by
  sorry


end NUMINAMATH_CALUDE_probability_two_pairs_l4097_409715


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l4097_409774

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (4, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Theorem statement
theorem circumcircle_of_triangle_ABC :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 ∧
  ∀ (x y : ℝ), circle_equation x y →
    (x - A.1)^2 + (y - A.2)^2 =
    (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 =
    (x - C.1)^2 + (y - C.2)^2 :=
sorry


end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l4097_409774


namespace NUMINAMATH_CALUDE_girls_in_blues_class_l4097_409792

/-- Calculates the number of girls in a class given the total number of students and the ratio of girls to boys -/
def girlsInClass (totalStudents : ℕ) (girlRatio boyRatio : ℕ) : ℕ :=
  (totalStudents * girlRatio) / (girlRatio + boyRatio)

/-- Theorem: In a class of 56 students with a girl to boy ratio of 4:3, there are 32 girls -/
theorem girls_in_blues_class :
  girlsInClass 56 4 3 = 32 := by
  sorry


end NUMINAMATH_CALUDE_girls_in_blues_class_l4097_409792


namespace NUMINAMATH_CALUDE_range_of_a_l4097_409771

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (a - x) > 0}
def B : Set ℝ := {x | |x + 1| + |x - 2| ≤ 3}

-- Define the complement of A
def C_R_A (a : ℝ) : Set ℝ := (A a)ᶜ

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (C_R_A a ∪ B) = Set.univ) ↔ (∀ a : ℝ, a ∈ Set.Icc (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4097_409771


namespace NUMINAMATH_CALUDE_sin_sin_2x_max_value_l4097_409741

theorem sin_sin_2x_max_value (x : ℝ) (h : 0 < x ∧ x < π/2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ y : ℝ, y = Real.sin x * Real.sin (2 * x) → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_sin_sin_2x_max_value_l4097_409741


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4097_409711

theorem inequality_solution_set (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 6 * x + 4) ↔ 
  (4 - 2 * Real.sqrt 19 < x ∧ x < 3 + 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4097_409711


namespace NUMINAMATH_CALUDE_job_completion_time_l4097_409724

/-- Represents the number of days needed to complete a job given initial and additional workers -/
def days_to_complete_job (initial_workers : ℕ) (initial_days : ℕ) (total_work_days : ℕ) 
  (days_before_joining : ℕ) (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let work_done := initial_workers * days_before_joining
  let remaining_work := total_work_days - work_done
  days_before_joining + (remaining_work + total_workers - 1) / total_workers

/-- Theorem stating that under the given conditions, the job will be completed in 6 days -/
theorem job_completion_time :
  days_to_complete_job 6 8 48 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l4097_409724


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l4097_409788

/-- Given a function f(x) = ax + b/x + 5 where a ≠ 0 and b ≠ 0, if f(2) = 3, then f(-2) = 7 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = a * x + b / x + 5)
  (h2 : f 2 = 3) :
  f (-2) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l4097_409788


namespace NUMINAMATH_CALUDE_triangle_area_angle_relation_l4097_409768

theorem triangle_area_angle_relation (a b c : ℝ) (A : ℝ) (S : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (S = (1/4) * (b^2 + c^2 - a^2)) →
  (S = (1/2) * b * c * Real.sin A) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (A = π/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_angle_relation_l4097_409768


namespace NUMINAMATH_CALUDE_hexagon_triangles_count_l4097_409713

/-- The number of unit equilateral triangles needed to form a regular hexagon of side length n -/
def triangles_in_hexagon (n : ℕ) : ℕ := 6 * (n * (n + 1) / 2)

/-- Given that 6 unit equilateral triangles can form a regular hexagon with side length 1,
    prove that 126 unit equilateral triangles are needed to form a regular hexagon with side length 6 -/
theorem hexagon_triangles_count :
  triangles_in_hexagon 1 = 6 →
  triangles_in_hexagon 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangles_count_l4097_409713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l4097_409783

-- Arithmetic Sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_problem (a₁ d Sn : ℚ) (n : ℕ) :
  a₁ = 3/2 ∧ d = -1/2 ∧ Sn = -15 →
  n = 12 ∧ arithmetic_sequence a₁ d n = -4 :=
sorry

-- Geometric Sequence
def geometric_sequence (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * q ^ (n - 1)

def geometric_sum (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * (q ^ n - 1) / (q - 1)

theorem geometric_sequence_problem (a₁ q Sn : ℚ) (n : ℕ) :
  q = 2 ∧ geometric_sequence a₁ q n = 96 ∧ Sn = 189 →
  a₁ = 3 ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l4097_409783


namespace NUMINAMATH_CALUDE_six_thirteen_not_square_nor_cube_l4097_409758

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem six_thirteen_not_square_nor_cube :
  ¬(is_square (6^13)) ∧ ¬(is_cube (6^13)) :=
sorry

end NUMINAMATH_CALUDE_six_thirteen_not_square_nor_cube_l4097_409758


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_k_value_l4097_409766

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  root_property : a 4 ^ 2 - 20 * a 4 + 99 = 0 ∧ a 5 ^ 2 - 20 * a 5 + 99 = 0
  sum_property : ∀ n : ℕ, 0 < n → S n ≤ S k

/-- The theorem stating that k equals 9 for the special arithmetic sequence -/
theorem special_arithmetic_sequence_k_value 
  (seq : SpecialArithmeticSequence) : 
  ∃ k : ℕ, k = 9 ∧ (∀ n : ℕ, 0 < n → seq.S n ≤ seq.S k) :=
sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_k_value_l4097_409766


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4097_409754

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 8) / Real.sqrt (8 * x + 8) = 2 / Real.sqrt 5) → x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4097_409754


namespace NUMINAMATH_CALUDE_thirteen_people_handshakes_l4097_409796

/-- The number of handshakes in a room with n people, where each person shakes hands with everyone else. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a room with 13 people, where each person shakes hands with everyone else, the total number of handshakes is 78. -/
theorem thirteen_people_handshakes :
  handshakes 13 = 78 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_handshakes_l4097_409796


namespace NUMINAMATH_CALUDE_student_pairs_l4097_409716

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l4097_409716


namespace NUMINAMATH_CALUDE_track_circumference_is_620_l4097_409797

/-- Represents the circular track and the movement of A and B -/
structure Track :=
  (circumference : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)

/-- The conditions of the problem -/
def problem_conditions (track : Track) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > t1 ∧
    track.speed_B * t1 = 120 ∧
    track.speed_A * t1 + track.speed_B * t1 = track.circumference / 2 ∧
    track.speed_A * t2 = track.circumference - 50 ∧
    track.speed_B * t2 = track.circumference / 2 + 50

/-- The theorem stating that the track circumference is 620 yards -/
theorem track_circumference_is_620 (track : Track) :
  problem_conditions track → track.circumference = 620 := by
  sorry

#check track_circumference_is_620

end NUMINAMATH_CALUDE_track_circumference_is_620_l4097_409797


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l4097_409738

/-- Given that x² is inversely proportional to y⁴, prove that x² = 2.25 when y = 4,
    given that x = 6 when y = 2. -/
theorem inverse_proportion_problem (k : ℝ) (x y : ℝ → ℝ) :
  (∀ t, t > 0 → x t ^ 2 * y t ^ 4 = k) →  -- x² is inversely proportional to y⁴
  x 2 = 6 →                               -- x = 6 when y = 2
  y 2 = 2 →                               -- y = 2 at this point
  y 4 = 4 →                               -- y = 4 at the point we're calculating
  x 4 ^ 2 = 2.25 :=                       -- x² = 2.25 when y = 4
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l4097_409738


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l4097_409793

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (155 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l4097_409793


namespace NUMINAMATH_CALUDE_decimal_division_division_result_l4097_409735

theorem decimal_division (x y : ℚ) : x / y = (x * 1000) / (y * 1000) := by sorry

theorem division_result : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_decimal_division_division_result_l4097_409735


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l4097_409731

/-- Calculates the cost of a children's ticket given the total cost and other ticket prices --/
theorem childrens_ticket_cost 
  (adult_price : ℕ) 
  (senior_price : ℕ) 
  (total_cost : ℕ) 
  (num_adults : ℕ) 
  (num_seniors : ℕ) 
  (num_children : ℕ) :
  adult_price = 11 →
  senior_price = 9 →
  num_adults = 2 →
  num_seniors = 2 →
  num_children = 3 →
  total_cost = 64 →
  (total_cost - (num_adults * adult_price + num_seniors * senior_price)) / num_children = 8 :=
by
  sorry

#check childrens_ticket_cost

end NUMINAMATH_CALUDE_childrens_ticket_cost_l4097_409731


namespace NUMINAMATH_CALUDE_sum_of_digits_l4097_409744

/-- The decimal representation of 1/142857 -/
def decimal_rep : ℚ := 1 / 142857

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 7

/-- The sum of digits in one repeating sequence -/
def cycle_sum : ℕ := 7

/-- The number of digits we're considering after the decimal point -/
def digit_count : ℕ := 35

/-- Theorem: The sum of the first 35 digits after the decimal point
    in the decimal representation of 1/142857 is equal to 35 -/
theorem sum_of_digits :
  (digit_count / repeat_length) * cycle_sum = 35 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l4097_409744


namespace NUMINAMATH_CALUDE_no_real_solutions_l4097_409734

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 10 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4097_409734


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l4097_409777

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) (h₃ : a₃ = 4/3) :
  geometric_sequence a₁ (a₂ / a₁) 8 = 4/729 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l4097_409777


namespace NUMINAMATH_CALUDE_range_of_fraction_l4097_409714

theorem range_of_fraction (x y : ℝ) (h1 : |x + y| ≤ 2) (h2 : |x - y| ≤ 2) :
  ∃ (z : ℝ), z = y / (x - 4) ∧ -1/2 ≤ z ∧ z ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l4097_409714


namespace NUMINAMATH_CALUDE_monica_reading_plan_l4097_409708

def books_last_year : ℕ := 16
def books_this_year : ℕ := 2 * books_last_year
def books_next_year : ℕ := 69

theorem monica_reading_plan :
  books_next_year - (2 * books_this_year) = 5 := by sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l4097_409708


namespace NUMINAMATH_CALUDE_computer_price_increase_l4097_409743

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 → (d * 1.3 : ℝ) = 364 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l4097_409743


namespace NUMINAMATH_CALUDE_product_evaluation_l4097_409778

theorem product_evaluation : (2.5 : ℝ) * (50.5 + 0.15) = 126.625 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l4097_409778


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l4097_409712

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    A + 2 * B = 11 →
    P * A + B = 10 →
    P = 2 := by
  sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l4097_409712


namespace NUMINAMATH_CALUDE_rectangle_opposite_vertices_distance_sum_equal_l4097_409773

/-- Theorem: The sums of the squares of the distances from any point in space to opposite vertices of a rectangle are equal to each other. -/
theorem rectangle_opposite_vertices_distance_sum_equal 
  (a b x y z : ℝ) : 
  (x^2 + y^2 + z^2) + ((x - a)^2 + (y - b)^2 + z^2) = 
  ((x - a)^2 + y^2 + z^2) + (x^2 + (y - b)^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_opposite_vertices_distance_sum_equal_l4097_409773


namespace NUMINAMATH_CALUDE_min_flips_theorem_l4097_409761

/-- Represents the color of a hat -/
inductive HatColor
| Blue
| Red

/-- Represents a gnome with a hat -/
structure Gnome where
  hat : HatColor

/-- Represents the state of all gnomes -/
def GnomeState := Fin 1000 → Gnome

/-- Counts the number of hat flips needed to reach a given state -/
def countFlips (initial final : GnomeState) : ℕ := sorry

/-- Checks if a given state allows all gnomes to make correct statements -/
def isValidState (state : GnomeState) : Prop := sorry

/-- The main theorem stating the minimum number of flips required -/
theorem min_flips_theorem (initial : GnomeState) :
  ∃ (final : GnomeState),
    isValidState final ∧
    countFlips initial final = 998 ∧
    ∀ (other : GnomeState),
      isValidState other →
      countFlips initial other ≥ 998 := by
  sorry

end NUMINAMATH_CALUDE_min_flips_theorem_l4097_409761


namespace NUMINAMATH_CALUDE_correlated_normal_distributions_l4097_409776

/-- Given two correlated normal distributions with specified parameters,
    this theorem proves the relationship between a value in the first distribution
    and its corresponding value in the second distribution. -/
theorem correlated_normal_distributions
  (μ₁ μ₂ σ₁ σ₂ ρ : ℝ)
  (h_μ₁ : μ₁ = 14.0)
  (h_μ₂ : μ₂ = 21.0)
  (h_σ₁ : σ₁ = 1.5)
  (h_σ₂ : σ₂ = 3.0)
  (h_ρ : ρ = 0.7)
  (x₁ : ℝ)
  (h_x₁ : x₁ = μ₁ - 2 * σ₁) :
  ∃ x₂ : ℝ, x₂ = μ₂ + ρ * (σ₂ / σ₁) * (x₁ - μ₁) :=
by
  sorry


end NUMINAMATH_CALUDE_correlated_normal_distributions_l4097_409776


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inequality_general_tetrahedron_inequality_l4097_409791

/-- Represents a tetrahedron with a triangle inside it -/
structure Tetrahedron where
  /-- Areas of the triangle's projections on the four faces -/
  P : Fin 4 → ℝ
  /-- Areas of the tetrahedron's faces -/
  S : Fin 4 → ℝ
  /-- Condition that all areas are non-negative -/
  all_non_neg : ∀ i, P i ≥ 0 ∧ S i ≥ 0

/-- Theorem for regular tetrahedron -/
theorem regular_tetrahedron_inequality (t : Tetrahedron) (h_regular : ∀ i j, t.S i = t.S j) :
  t.P 0 ≤ t.P 1 + t.P 2 + t.P 3 :=
sorry

/-- Theorem for any tetrahedron -/
theorem general_tetrahedron_inequality (t : Tetrahedron) :
  t.P 0 * t.S 0 ≤ t.P 1 * t.S 1 + t.P 2 * t.S 2 + t.P 3 * t.S 3 :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inequality_general_tetrahedron_inequality_l4097_409791


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l4097_409710

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), ∃ (k : ℤ), 
    n / (25 - n) = k^2 ∧ 
    ∃ (m : ℤ), n = 3 * m :=
by sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l4097_409710


namespace NUMINAMATH_CALUDE_red_markers_count_l4097_409707

def total_markers : ℕ := 3343
def blue_markers : ℕ := 1028

theorem red_markers_count : total_markers - blue_markers = 2315 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l4097_409707


namespace NUMINAMATH_CALUDE_miranda_pillows_l4097_409706

-- Define the constants
def feathers_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

-- Theorem statement
theorem miranda_pillows :
  let goose_pounds : ℕ := goose_total_feathers / goose_feathers_per_pound
  let duck_pounds : ℕ := duck_total_feathers / duck_feathers_per_pound
  let total_pounds : ℕ := goose_pounds + duck_pounds
  let pillows : ℕ := total_pounds / feathers_per_pillow
  pillows = 10 := by sorry

end NUMINAMATH_CALUDE_miranda_pillows_l4097_409706


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4097_409709

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4097_409709


namespace NUMINAMATH_CALUDE_unique_solution_l4097_409749

/-- A pair of natural numbers (r, x) where r is the base and x is a number in that base -/
structure BaseNumber :=
  (r : ℕ)
  (x : ℕ)

/-- Check if a BaseNumber satisfies the given conditions -/
def satisfiesConditions (bn : BaseNumber) : Prop :=
  -- r is at most 70
  bn.r ≤ 70 ∧
  -- x is represented by repeating a pair of digits
  ∃ (n : ℕ) (a b : ℕ), 
    a < bn.r ∧ b < bn.r ∧
    bn.x = (a * bn.r + b) * (bn.r^(2*n) - 1) / (bn.r^2 - 1) ∧
  -- x^2 in base r consists of 4n ones
  ∃ (n : ℕ), bn.x^2 = (bn.r^(4*n) - 1) / (bn.r - 1)

/-- The theorem stating that (7, 26₇) is the only solution -/
theorem unique_solution : 
  ∀ (bn : BaseNumber), satisfiesConditions bn ↔ bn.r = 7 ∧ bn.x = 26 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l4097_409749


namespace NUMINAMATH_CALUDE_square_sum_is_seven_l4097_409763

theorem square_sum_is_seven (x y : ℝ) (h : (x^2 + 1) * (y^2 + 1) + 9 = 6 * (x + y)) : 
  x^2 + y^2 = 7 := by sorry

end NUMINAMATH_CALUDE_square_sum_is_seven_l4097_409763


namespace NUMINAMATH_CALUDE_apples_per_bucket_l4097_409742

theorem apples_per_bucket (total_apples : ℕ) (num_buckets : ℕ) 
  (h1 : total_apples = 56) (h2 : num_buckets = 7) :
  total_apples / num_buckets = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_per_bucket_l4097_409742


namespace NUMINAMATH_CALUDE_miles_left_to_run_l4097_409722

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Macy's daily running distance in miles -/
def daily_distance : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Theorem stating the number of miles left for Macy to run after 6 days -/
theorem miles_left_to_run : weekly_goal - (daily_distance * days_run) = 6 := by
  sorry

end NUMINAMATH_CALUDE_miles_left_to_run_l4097_409722


namespace NUMINAMATH_CALUDE_multiples_count_l4097_409781

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 3 = 0 ∨ x % 5 = 0) ∧ x % 6 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples 200 = 73 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_l4097_409781


namespace NUMINAMATH_CALUDE_cubic_factorization_l4097_409727

theorem cubic_factorization (m : ℝ) : m^3 - 6*m^2 + 9*m = m*(m-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4097_409727


namespace NUMINAMATH_CALUDE_audiobook_length_l4097_409702

/-- Proves that if a person listens to audiobooks for a certain amount of time each day
    and completes a certain number of audiobooks in a given number of days,
    then each audiobook has a specific length. -/
theorem audiobook_length
  (daily_listening_hours : ℝ)
  (total_days : ℕ)
  (num_audiobooks : ℕ)
  (h1 : daily_listening_hours = 2)
  (h2 : total_days = 90)
  (h3 : num_audiobooks = 6)
  : (daily_listening_hours * total_days) / num_audiobooks = 30 := by
  sorry

end NUMINAMATH_CALUDE_audiobook_length_l4097_409702


namespace NUMINAMATH_CALUDE_students_walking_home_l4097_409737

theorem students_walking_home (total : ℚ) (bus auto bike scooter : ℚ) : 
  total = 1 →
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  scooter = 1/10 →
  total - (bus + auto + bike + scooter) = 29/120 :=
by sorry

end NUMINAMATH_CALUDE_students_walking_home_l4097_409737


namespace NUMINAMATH_CALUDE_min_stone_product_l4097_409740

theorem min_stone_product (total_stones : ℕ) (black_stones : ℕ) : 
  total_stones = 40 → 
  black_stones ≥ 20 → 
  black_stones ≤ 32 → 
  (black_stones * (total_stones - black_stones)) ≥ 256 := by
sorry

end NUMINAMATH_CALUDE_min_stone_product_l4097_409740


namespace NUMINAMATH_CALUDE_new_average_score_l4097_409728

theorem new_average_score (n : ℕ) (a s : ℚ) (h1 : n = 4) (h2 : a = 78) (h3 : s = 88) :
  (n * a + s) / (n + 1) = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_l4097_409728


namespace NUMINAMATH_CALUDE_exponent_problem_l4097_409769

theorem exponent_problem (a : ℝ) (m n : ℤ) 
  (h1 : a ^ m = 2) (h2 : a ^ n = 3) : 
  a ^ (m + n) = 6 ∧ a ^ (m - 2*n) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l4097_409769


namespace NUMINAMATH_CALUDE_equation_solution_l4097_409762

theorem equation_solution : ∃! x : ℝ, (x - 6) / (x + 4) = (x + 3) / (x - 5) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4097_409762


namespace NUMINAMATH_CALUDE_parabola_c_value_l4097_409732

-- Define the parabola
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c 3 = -5

-- Define the point condition
def point_condition (a b c : ℝ) : Prop :=
  parabola a b c 1 = -3

theorem parabola_c_value :
  ∀ a b c : ℝ,
  vertex_condition a b c →
  point_condition a b c →
  c = -0.5 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l4097_409732


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_129_l4097_409704

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithCircle where
  -- The lengths of the parallel sides
  shorterBase : ℝ
  longerBase : ℝ
  -- The length of the leg (equal for both legs in an isosceles trapezoid)
  leg : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Conditions
  height_positive : height > 0
  radius_positive : radius > 0
  longer_than_shorter : longerBase > shorterBase
  circle_touches_base : shorterBase = 2 * radius
  circle_touches_leg : leg^2 = (longerBase - shorterBase)^2 / 4 + height^2

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  t.shorterBase + t.longerBase + 2 * t.leg

theorem trapezoid_perimeter_is_129
  (t : IsoscelesTrapezoidWithCircle)
  (h₁ : t.height = 36)
  (h₂ : t.radius = 11) :
  perimeter t = 129 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_129_l4097_409704


namespace NUMINAMATH_CALUDE_marbles_problem_l4097_409767

theorem marbles_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a)
  (h2 : brian = 3 * a)
  (h3 : caden = 6 * a)
  (h4 : daryl = 24 * a)
  (h5 : angela + brian + caden + daryl = 156) : 
  a = 78 / 17 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l4097_409767


namespace NUMINAMATH_CALUDE_calculate_sales_11_to_12_l4097_409790

/-- Sales data for a shopping mall during National Day Golden Week promotion -/
structure SalesData where
  sales_9_to_10 : ℝ
  height_ratio_11_to_12 : ℝ

/-- Theorem: Given the sales from 9:00 to 10:00 and the height ratio of the 11:00 to 12:00 bar,
    calculate the sales from 11:00 to 12:00 -/
theorem calculate_sales_11_to_12 (data : SalesData)
    (h1 : data.sales_9_to_10 = 25000)
    (h2 : data.height_ratio_11_to_12 = 4) :
    data.sales_9_to_10 * data.height_ratio_11_to_12 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_sales_11_to_12_l4097_409790


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l4097_409759

/-- The number of bottle caps Marilyn has after receiving some from Nancy -/
def total_bottle_caps (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem: Marilyn's total bottle caps is the sum of her initial count and what she received -/
theorem marilyn_bottle_caps (initial : Real) (received : Real) :
  total_bottle_caps initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l4097_409759


namespace NUMINAMATH_CALUDE_inequality_proof_l4097_409745

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4097_409745


namespace NUMINAMATH_CALUDE_no_real_curve_exists_l4097_409733

theorem no_real_curve_exists : ¬ ∃ (x y : ℝ), x^2 + y^2 - 2*x + 4*y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_curve_exists_l4097_409733


namespace NUMINAMATH_CALUDE_senate_subcommittee_combinations_l4097_409700

theorem senate_subcommittee_combinations (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) :
  total_republicans = 8 → 
  total_democrats = 6 → 
  subcommittee_republicans = 3 → 
  subcommittee_democrats = 2 → 
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 840 := by
sorry

end NUMINAMATH_CALUDE_senate_subcommittee_combinations_l4097_409700


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l4097_409750

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + m = 0 ↔ x = (-2 + Complex.I * Real.sqrt 88) / 8 ∨ x = (-2 - Complex.I * Real.sqrt 88) / 8) → 
  m = 13 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l4097_409750


namespace NUMINAMATH_CALUDE_trapezoid_area_proof_l4097_409746

/-- The area of a trapezoid bounded by the lines y = x + 1, y = 15, y = 8, and the y-axis -/
def trapezoid_area : ℝ := 73.5

/-- The line y = x + 1 -/
def line1 (x : ℝ) : ℝ := x + 1

/-- The line y = 15 -/
def line2 : ℝ := 15

/-- The line y = 8 -/
def line3 : ℝ := 8

theorem trapezoid_area_proof :
  let x1 := (line2 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 15
  let x2 := (line3 - 1 : ℝ)  -- x-coordinate where y = x + 1 intersects y = 8
  let base1 := x1
  let base2 := x2
  let height := line2 - line3
  (base1 + base2) * height / 2 = trapezoid_area := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_proof_l4097_409746


namespace NUMINAMATH_CALUDE_count_mgons_with_two_acute_angles_correct_l4097_409729

/-- Given integers m and n where 4 < m < n, and a regular (2n+1)-gon with vertices set P,
    this function computes the number of convex m-gons with vertices in P
    that have exactly two acute internal angles. -/
def count_mgons_with_two_acute_angles (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that the count_mgons_with_two_acute_angles function
    correctly computes the number of m-gons with two acute angles in a (2n+1)-gon. -/
theorem count_mgons_with_two_acute_angles_correct (m n : ℕ) 
    (h1 : 4 < m) (h2 : m < n) : 
  count_mgons_with_two_acute_angles m n = 
    (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) := by
  sorry

#check count_mgons_with_two_acute_angles_correct

end NUMINAMATH_CALUDE_count_mgons_with_two_acute_angles_correct_l4097_409729


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l4097_409785

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem complement_P_intersect_Q : 
  (Set.univ \ P 3) ∩ Q = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem P_proper_subset_Q_iff_a_in_range : 
  ∀ a : ℝ, (P a ⊂ Q ∧ P a ≠ Q) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l4097_409785


namespace NUMINAMATH_CALUDE_inequality_proof_l4097_409770

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4097_409770


namespace NUMINAMATH_CALUDE_solution_and_uniqueness_l4097_409748

def equation (x : ℤ) : Prop :=
  (x + 1)^3 + (x + 2)^3 + (x + 3)^3 = (x + 4)^3

theorem solution_and_uniqueness :
  equation 2 ∧ ∀ x : ℤ, x ≠ 2 → ¬(equation x) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_uniqueness_l4097_409748


namespace NUMINAMATH_CALUDE_charlotte_dan_mean_score_l4097_409764

def test_scores : List ℝ := [82, 84, 86, 88, 90, 92, 95, 97]

def total_score : ℝ := test_scores.sum

def ava_ben_mean : ℝ := 90

def num_tests : ℕ := 4

theorem charlotte_dan_mean_score :
  let ava_ben_total : ℝ := ava_ben_mean * num_tests
  let charlotte_dan_total : ℝ := total_score - ava_ben_total
  charlotte_dan_total / num_tests = 88.5 := by sorry

end NUMINAMATH_CALUDE_charlotte_dan_mean_score_l4097_409764


namespace NUMINAMATH_CALUDE_inequality_proof_minimum_value_proof_l4097_409720

-- Define the variables and conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Part I: Prove the inequality
theorem inequality_proof : Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) ≤ 4 := by
  sorry

-- Part II: Prove the minimum value
theorem minimum_value_proof : ∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1 / (a + 1) + 1 / (b + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_minimum_value_proof_l4097_409720


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4097_409752

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4097_409752


namespace NUMINAMATH_CALUDE_min_homeowners_l4097_409765

theorem min_homeowners (total : ℕ) (men women : ℕ) (men_ratio women_ratio : ℚ) : 
  total = 150 →
  men + women = total →
  men_ratio = 1/10 →
  women_ratio = 1/5 →
  men ≥ 0 →
  women ≥ 0 →
  ∃ (homeowners : ℕ), homeowners = 16 ∧ 
    ∀ (h : ℕ), h ≥ men_ratio * men + women_ratio * women → h ≥ homeowners :=
by sorry

end NUMINAMATH_CALUDE_min_homeowners_l4097_409765


namespace NUMINAMATH_CALUDE_larger_number_is_sixteen_l4097_409760

theorem larger_number_is_sixteen (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 27) :
  max a b = 16 := by sorry

end NUMINAMATH_CALUDE_larger_number_is_sixteen_l4097_409760


namespace NUMINAMATH_CALUDE_complex_number_location_l4097_409730

theorem complex_number_location (z : ℂ) : 
  z = (1/2 : ℝ) * Complex.abs z + Complex.I ^ 2015 → 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_location_l4097_409730


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_cubed_l4097_409719

theorem floor_negative_seven_fourths_cubed : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_cubed_l4097_409719


namespace NUMINAMATH_CALUDE_max_value_constraint_l4097_409753

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (max : ℝ), max = 37 / 2 ∧ ∀ (a b c : ℝ), 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → 8 * a + 3 * b + 10 * c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l4097_409753


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l4097_409736

theorem sales_tax_percentage 
  (total_bill : ℝ) 
  (tip_percentage : ℝ) 
  (food_price : ℝ) 
  (h1 : total_bill = 211.20)
  (h2 : tip_percentage = 0.20)
  (h3 : food_price = 160) :
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage = 0.10 ∧
    total_bill = food_price * (1 + sales_tax_percentage) * (1 + tip_percentage) := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l4097_409736


namespace NUMINAMATH_CALUDE_fifth_odd_integer_in_sequence_l4097_409779

theorem fifth_odd_integer_in_sequence (n : ℕ) (sum : ℕ) (h1 : n = 20) (h2 : sum = 400) :
  let seq := fun i => 2 * i - 1
  let first := (sum - n * (n - 1)) / (2 * n)
  seq (first + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_odd_integer_in_sequence_l4097_409779


namespace NUMINAMATH_CALUDE_seven_ninths_rounded_l4097_409775

/-- Rounds a rational number to a specified number of decimal places -/
noncomputable def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (↑(⌊q * 10^places + 1/2⌋) : ℚ) / 10^places

/-- The fraction 7/9 rounded to 2 decimal places equals 0.78 -/
theorem seven_ninths_rounded : round_to_decimal_places (7/9) 2 = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_seven_ninths_rounded_l4097_409775


namespace NUMINAMATH_CALUDE_difference_even_odd_sums_l4097_409795

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of first n positive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : 
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_odd_sums_l4097_409795


namespace NUMINAMATH_CALUDE_max_pairs_sum_bound_l4097_409789

theorem max_pairs_sum_bound (k : ℕ) 
  (pairs : Fin k → (ℕ × ℕ))
  (h_range : ∀ i, (pairs i).1 ∈ Finset.range 3000 ∧ (pairs i).2 ∈ Finset.range 3000)
  (h_order : ∀ i, (pairs i).1 < (pairs i).2)
  (h_distinct : ∀ i j, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                                (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2)
  (h_sum_distinct : ∀ i j, i ≠ j → (pairs i).1 + (pairs i).2 ≠ (pairs j).1 + (pairs j).2)
  (h_sum_bound : ∀ i, (pairs i).1 + (pairs i).2 ≤ 4000) :
  k ≤ 1599 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_bound_l4097_409789
