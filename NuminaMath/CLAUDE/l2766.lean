import Mathlib

namespace NUMINAMATH_CALUDE_shifted_function_l2766_276638

def g (x : ℝ) : ℝ := 5 * x^2

def f (x : ℝ) : ℝ := 5 * (x - 3)^2 - 2

theorem shifted_function (x : ℝ) : 
  f x = g (x - 3) - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_function_l2766_276638


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2766_276672

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2766_276672


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l2766_276648

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 153 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l2766_276648


namespace NUMINAMATH_CALUDE_monomial_sum_equation_solution_l2766_276610

theorem monomial_sum_equation_solution :
  ∀ (a b : ℝ) (m n : ℕ),
  (∃ (k : ℝ), ∀ (a b : ℝ), (1/3 * a^m * b^3) + (-2 * a^2 * b^n) = k * a^m * b^n) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1 ∧ x = -23) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_equation_solution_l2766_276610


namespace NUMINAMATH_CALUDE_orange_ratio_l2766_276607

def total_oranges : ℕ := 180
def alice_oranges : ℕ := 120

theorem orange_ratio : 
  let emily_oranges := total_oranges - alice_oranges
  (alice_oranges : ℚ) / emily_oranges = 2 := by
sorry

end NUMINAMATH_CALUDE_orange_ratio_l2766_276607


namespace NUMINAMATH_CALUDE_adoption_fee_is_50_l2766_276601

/-- Represents the adoption fee for the cat -/
def adoption_fee : ℝ := sorry

/-- Represents the total vet visit costs for the first year -/
def vet_costs : ℝ := 500

/-- Represents the monthly food cost -/
def monthly_food_cost : ℝ := 25

/-- Represents the cost of toys Jenny bought -/
def jenny_toy_costs : ℝ := 200

/-- Represents Jenny's total spending on the cat in the first year -/
def jenny_total_spending : ℝ := 625

/-- Theorem stating that the adoption fee is $50 -/
theorem adoption_fee_is_50 : adoption_fee = 50 := by
  sorry

end NUMINAMATH_CALUDE_adoption_fee_is_50_l2766_276601


namespace NUMINAMATH_CALUDE_two_yellow_marbles_prob_l2766_276685

/-- Represents the number of marbles of each color in the box -/
structure MarbleBox where
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Calculates the total number of marbles in the box -/
def totalMarbles (box : MarbleBox) : ℕ :=
  box.blue + box.yellow + box.orange

/-- Calculates the probability of drawing a yellow marble -/
def probYellow (box : MarbleBox) : ℚ :=
  box.yellow / (totalMarbles box)

/-- The probability of drawing two yellow marbles in succession with replacement -/
def probTwoYellow (box : MarbleBox) : ℚ :=
  (probYellow box) * (probYellow box)

theorem two_yellow_marbles_prob :
  let box : MarbleBox := { blue := 4, yellow := 5, orange := 6 }
  probTwoYellow box = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_yellow_marbles_prob_l2766_276685


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2766_276694

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x =>
    let correct_result := 2 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 95

-- The proof is omitted
theorem error_percentage_division_vs_multiplication_proof :
  ∀ x : ℝ, x ≠ 0 → error_percentage_division_vs_multiplication x :=
sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2766_276694


namespace NUMINAMATH_CALUDE_salary_change_l2766_276615

theorem salary_change (original : ℝ) (h : original > 0) :
  ∃ (increase : ℝ),
    (original * 0.7 * (1 + increase) = original * 0.91) ∧
    increase = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2766_276615


namespace NUMINAMATH_CALUDE_garbage_collection_average_l2766_276695

theorem garbage_collection_average (total_garbage : ℝ) 
  (h1 : total_garbage = 900) 
  (h2 : ∃ x : ℝ, total_garbage = x + x / 2) : 
  ∃ x : ℝ, x + x / 2 = total_garbage ∧ x = 600 :=
by sorry

end NUMINAMATH_CALUDE_garbage_collection_average_l2766_276695


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2766_276635

theorem quadratic_rational_solutions : 
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 15 * x + c₁.val = 0) ∧ 
    (∃ (y : ℚ), 7 * y^2 + 15 * y + c₂.val = 0) ∧ 
    c₁ ≠ c₂ ∧ 
    c₁.val * c₂.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2766_276635


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l2766_276666

theorem tangent_perpendicular_line (x₀ y₀ c : ℝ) : 
  y₀ = Real.exp x₀ →                     -- P is on the curve y = e^x
  x₀ + 2 * y₀ + c = 0 →                  -- Line passes through P
  2 * Real.exp x₀ = -1 →                 -- Line is perpendicular to tangent
  c = -4 - Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l2766_276666


namespace NUMINAMATH_CALUDE_pet_ownership_l2766_276623

theorem pet_ownership (S D C B : Finset ℕ) (h1 : S.card = 50)
  (h2 : ∀ s ∈ S, s ∈ D ∨ s ∈ C ∨ s ∈ B)
  (h3 : D.card = 30) (h4 : C.card = 35) (h5 : B.card = 10)
  (h6 : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l2766_276623


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l2766_276661

/-- 
Given two arithmetic progressions:
1) {5, 9, 13, 17, ...} with common difference 4
2) {4, 12, 20, 28, ...} with common difference 8
This theorem states that their largest common value less than 1000 is 993.
-/
theorem largest_common_value_less_than_1000 :
  let seq1 := fun n : ℕ => 5 + 4 * n
  let seq2 := fun n : ℕ => 4 + 8 * n
  ∃ (k1 k2 : ℕ), seq1 k1 = seq2 k2 ∧ 
                 seq1 k1 < 1000 ∧
                 ∀ (m1 m2 : ℕ), seq1 m1 = seq2 m2 → seq1 m1 < 1000 → seq1 m1 ≤ seq1 k1 ∧
                 seq1 k1 = 993 :=
by sorry


end NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l2766_276661


namespace NUMINAMATH_CALUDE_polynomial_value_l2766_276696

/-- Given that ax³ + bx + 1 = 2023 when x = 1, prove that ax³ + bx - 2 = -2024 when x = -1 -/
theorem polynomial_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 2023) → (a * (-1)^3 + b * (-1) - 2 = -2024) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l2766_276696


namespace NUMINAMATH_CALUDE_solve_strawberry_problem_l2766_276603

def strawberry_problem (christine_pounds rachel_pounds total_pies : ℕ) : Prop :=
  rachel_pounds = 2 * christine_pounds →
  christine_pounds = 10 →
  total_pies = 10 →
  (christine_pounds + rachel_pounds) / total_pies = 3

theorem solve_strawberry_problem :
  ∃ (christine_pounds rachel_pounds total_pies : ℕ),
    strawberry_problem christine_pounds rachel_pounds total_pies :=
by
  sorry

end NUMINAMATH_CALUDE_solve_strawberry_problem_l2766_276603


namespace NUMINAMATH_CALUDE_right_triangle_area_l2766_276618

/-- The area of a right triangle with base 12 cm and height 15 cm is 90 square centimeters. -/
theorem right_triangle_area (base height area : ℝ) : 
  base = 12 → height = 15 → area = (1 / 2) * base * height → area = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2766_276618


namespace NUMINAMATH_CALUDE_total_football_games_l2766_276691

def football_games_this_year : ℕ := 4
def football_games_last_year : ℕ := 9

theorem total_football_games : 
  football_games_this_year + football_games_last_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l2766_276691


namespace NUMINAMATH_CALUDE_emily_cards_l2766_276652

theorem emily_cards (x : ℕ) : x + 7 = 70 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_emily_cards_l2766_276652


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l2766_276667

theorem not_equivalent_fraction (h : 0.000000275 = 2.75 * 10^(-7)) : 
  (11/40) * 10^(-7) ≠ 2.75 * 10^(-7) := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l2766_276667


namespace NUMINAMATH_CALUDE_athletic_groups_l2766_276617

/-- Given a number of athletes and groups satisfying certain conditions,
    prove that there are 59 athletes and 8 groups. -/
theorem athletic_groups (x y : ℤ) 
  (eq1 : 7 * y + 3 = x) 
  (eq2 : 8 * y - 5 = x) : 
  x = 59 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_athletic_groups_l2766_276617


namespace NUMINAMATH_CALUDE_abs_condition_for_log_half_condition_l2766_276656

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Statement of the theorem
theorem abs_condition_for_log_half_condition (x : ℝ) :
  (∀ x, |x - 2| < 1 → log_half (x + 2) < 0) ∧
  (∃ x, log_half (x + 2) < 0 ∧ |x - 2| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_abs_condition_for_log_half_condition_l2766_276656


namespace NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l2766_276654

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c / l.a = -l.c / l.b

theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line2D, point_on_line ⟨0, 5⟩ l ∧ equal_intercepts l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l2766_276654


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l2766_276637

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_ratio := pd.a_investment + pd.b_investment + pd.c_investment
  let unit_profit := pd.b_profit * total_ratio / pd.b_investment
  let a_profit := unit_profit * pd.a_investment / total_ratio
  let c_profit := unit_profit * pd.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3000) :
  profit_difference pd = 1200 := by
  sorry

#eval profit_difference ⟨8000, 10000, 12000, 3000⟩

end NUMINAMATH_CALUDE_profit_difference_theorem_l2766_276637


namespace NUMINAMATH_CALUDE_constant_odd_function_is_zero_l2766_276670

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem constant_odd_function_is_zero (k : ℝ) (h : IsOdd (fun x ↦ k)) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_odd_function_is_zero_l2766_276670


namespace NUMINAMATH_CALUDE_bowtie_problem_l2766_276659

-- Define the bowtie operation
noncomputable def bowtie (c d : ℝ) : ℝ := c + 1 + Real.sqrt (d + Real.sqrt (d + Real.sqrt d))

-- State the theorem
theorem bowtie_problem (h : ℝ) (hyp : bowtie 8 h = 12) : h = 6 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_problem_l2766_276659


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2766_276663

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2766_276663


namespace NUMINAMATH_CALUDE_cistern_problem_l2766_276698

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

theorem cistern_problem :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  cistern_wet_surface_area length width depth = 49 := by
  sorry

end NUMINAMATH_CALUDE_cistern_problem_l2766_276698


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2766_276673

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2766_276673


namespace NUMINAMATH_CALUDE_second_investment_rate_l2766_276655

def contest_winnings : ℝ := 5000
def first_investment : ℝ := 1800
def first_interest_rate : ℝ := 0.05
def total_interest : ℝ := 298

def second_investment : ℝ := 2 * first_investment - 400

def first_interest : ℝ := first_investment * first_interest_rate

def second_interest : ℝ := total_interest - first_interest

theorem second_investment_rate (second_rate : ℝ) : 
  second_rate * second_investment = second_interest → second_rate = 0.065 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l2766_276655


namespace NUMINAMATH_CALUDE_construct_one_to_ten_l2766_276679

/-- A type representing the allowed operations in our constructions -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide
  | Exponentiate

/-- A type representing a construction using threes and operations -/
inductive Construction
  | Three : Construction
  | Op : Operation → Construction → Construction → Construction

/-- Evaluate a construction to a rational number -/
def evaluate : Construction → ℚ
  | Construction.Three => 3
  | Construction.Op Operation.Add a b => evaluate a + evaluate b
  | Construction.Op Operation.Subtract a b => evaluate a - evaluate b
  | Construction.Op Operation.Multiply a b => evaluate a * evaluate b
  | Construction.Op Operation.Divide a b => evaluate a / evaluate b
  | Construction.Op Operation.Exponentiate a b => (evaluate a) ^ (evaluate b).num

/-- Count the number of threes used in a construction -/
def countThrees : Construction → ℕ
  | Construction.Three => 1
  | Construction.Op _ a b => countThrees a + countThrees b

/-- Predicate to check if a construction is valid (uses exactly five threes) -/
def isValidConstruction (c : Construction) : Prop := countThrees c = 5

/-- Theorem: We can construct all numbers from 1 to 10 using five threes and allowed operations -/
theorem construct_one_to_ten :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 →
  ∃ c : Construction, isValidConstruction c ∧ evaluate c = n := by sorry

end NUMINAMATH_CALUDE_construct_one_to_ten_l2766_276679


namespace NUMINAMATH_CALUDE_square_plot_area_l2766_276674

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) : 
  price_per_foot = 58 → total_cost = 1624 → 
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    4 * side_length * price_per_foot = total_cost ∧ 
    side_length^2 = 49 := by
  sorry

#check square_plot_area

end NUMINAMATH_CALUDE_square_plot_area_l2766_276674


namespace NUMINAMATH_CALUDE_sum_of_cube_difference_l2766_276645

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c : ℕ+)^3 - a^3 - b^3 - c^3 = 210 →
  (a : ℕ) + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_difference_l2766_276645


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l2766_276602

theorem mixed_number_multiplication :
  99 * (24 / 25) * (-5) = -(499 + 4 / 5) :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l2766_276602


namespace NUMINAMATH_CALUDE_island_marriage_fraction_l2766_276626

theorem island_marriage_fraction (N : ℚ) :
  let M := (3/2) * N  -- Total number of men
  let W := (5/3) * N  -- Total number of women
  let P := M + W      -- Total population
  (2 * N) / P = 12/19 := by
  sorry

end NUMINAMATH_CALUDE_island_marriage_fraction_l2766_276626


namespace NUMINAMATH_CALUDE_lucky_number_2005_to_52000_l2766_276621

/-- A natural number is a lucky number if the sum of its digits is 7 -/
def is_lucky_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def lucky_number_sequence : ℕ → ℕ :=
  sorry

/-- The 2005th lucky number is the nth in the sequence -/
axiom a_2005_is_nth : ∃ n : ℕ, lucky_number_sequence n = 2005

theorem lucky_number_2005_to_52000 :
  ∃ n : ℕ, lucky_number_sequence n = 2005 ∧ lucky_number_sequence (5 * n) = 52000 :=
sorry

end NUMINAMATH_CALUDE_lucky_number_2005_to_52000_l2766_276621


namespace NUMINAMATH_CALUDE_isabel_circuit_length_l2766_276627

/-- The length of Isabel's running circuit in meters. -/
def circuit_length : ℕ := 365

/-- The number of times Isabel runs the circuit in the morning. -/
def morning_runs : ℕ := 7

/-- The number of times Isabel runs the circuit in the afternoon. -/
def afternoon_runs : ℕ := 3

/-- The total distance Isabel runs in a week, in meters. -/
def weekly_distance : ℕ := 25550

/-- The number of days in a week. -/
def days_in_week : ℕ := 7

theorem isabel_circuit_length :
  circuit_length * (morning_runs + afternoon_runs) * days_in_week = weekly_distance :=
sorry

end NUMINAMATH_CALUDE_isabel_circuit_length_l2766_276627


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2766_276612

theorem quadratic_roots_sum_of_squares (m : ℝ) 
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0)
  (h2 : ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 = 23 ∧ x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0) :
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2766_276612


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l2766_276693

theorem reciprocal_of_sum : (1 / (1/3 + 3/4)) = 12/13 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l2766_276693


namespace NUMINAMATH_CALUDE_megan_eggs_count_l2766_276619

theorem megan_eggs_count :
  ∀ (broken cracked perfect : ℕ),
  broken = 3 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken + cracked + perfect = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_eggs_count_l2766_276619


namespace NUMINAMATH_CALUDE_sugar_price_increase_l2766_276622

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) :
  original_price = 3 →
  consumption_reduction = 0.4 →
  let new_consumption := 1 - consumption_reduction
  let new_price := original_price / new_consumption
  new_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l2766_276622


namespace NUMINAMATH_CALUDE_mary_money_left_l2766_276668

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary will have 50 - 10p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l2766_276668


namespace NUMINAMATH_CALUDE_hexagon_four_identical_shapes_l2766_276671

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add necessary fields

/-- A line segment representing a cut in the hexagon -/
structure Cut where
  -- Add necessary fields

/-- Represents a shape resulting from cuts in the hexagon -/
structure Shape where
  -- Add necessary fields

/-- Checks if two shapes are identical -/
def are_identical (s1 s2 : Shape) : Prop := sorry

/-- Checks if a cut is along a symmetry axis of the hexagon -/
def is_symmetry_axis_cut (h : RegularHexagon) (c : Cut) : Prop := sorry

/-- Theorem: A regular hexagon can be divided into four identical shapes by cutting along its symmetry axes -/
theorem hexagon_four_identical_shapes (h : RegularHexagon) :
  ∃ (c1 c2 : Cut) (s1 s2 s3 s4 : Shape),
    is_symmetry_axis_cut h c1 ∧
    is_symmetry_axis_cut h c2 ∧
    are_identical s1 s2 ∧
    are_identical s1 s3 ∧
    are_identical s1 s4 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_four_identical_shapes_l2766_276671


namespace NUMINAMATH_CALUDE_mirror_pieces_l2766_276631

theorem mirror_pieces : ∃ P : ℕ, 
  (P > 0) ∧ 
  (P / 2 - 3 > 0) ∧
  ((P / 2 - 3) / 3 = 9) ∧
  (P = 60) := by
  sorry

end NUMINAMATH_CALUDE_mirror_pieces_l2766_276631


namespace NUMINAMATH_CALUDE_factoring_expression_l2766_276658

theorem factoring_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2766_276658


namespace NUMINAMATH_CALUDE_jakes_weight_l2766_276669

theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 40 = 3 * sister)
  (h2 : jake - (sister + 10) = brother)
  (h3 : jake + sister + brother = 300) :
  jake = 155 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l2766_276669


namespace NUMINAMATH_CALUDE_file_download_rate_l2766_276616

/-- Given a file download scenario, prove the download rate for the latter part. -/
theorem file_download_rate 
  (file_size : ℝ) 
  (initial_rate : ℝ) 
  (initial_size : ℝ) 
  (total_time : ℝ) 
  (h1 : file_size = 90) 
  (h2 : initial_rate = 5) 
  (h3 : initial_size = 60) 
  (h4 : total_time = 15) : 
  (file_size - initial_size) / (total_time - initial_size / initial_rate) = 10 := by
  sorry

end NUMINAMATH_CALUDE_file_download_rate_l2766_276616


namespace NUMINAMATH_CALUDE_initial_cost_of_article_l2766_276624

/-- 
Proves that the initial cost of an article is 3000, given the conditions of two successive discounts.
-/
theorem initial_cost_of_article (price_after_first_discount : ℕ) 
  (final_price : ℕ) (h1 : price_after_first_discount = 2100) 
  (h2 : final_price = 1050) : ℕ :=
  by
    sorry

#check initial_cost_of_article

end NUMINAMATH_CALUDE_initial_cost_of_article_l2766_276624


namespace NUMINAMATH_CALUDE_cubic_inequality_l2766_276609

theorem cubic_inequality (a b c : ℝ) :
  a^6 + b^6 + c^6 - 3*a^2*b^2*c^2 ≥ (1/2) * (a-b)^2 * (b-c)^2 * (c-a)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2766_276609


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2766_276692

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2766_276692


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_snack_bag_l2766_276613

theorem min_blue_eyes_and_snack_bag 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (snack_bag : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 14) 
  (h3 : snack_bag = 22) 
  (h4 : blue_eyes ≤ total_students) 
  (h5 : snack_bag ≤ total_students) : 
  ∃ (both : ℕ), both ≥ 1 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ snack_bag ∧ 
    (blue_eyes - both) + (snack_bag - both) ≤ total_students := by
  sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_snack_bag_l2766_276613


namespace NUMINAMATH_CALUDE_container_volume_transformation_l2766_276664

/-- Represents a rectangular container with dimensions height, length, and width -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.length * c.width

/-- Creates a new container by scaling the dimensions of an original container -/
def scaleContainer (c : Container) (h_scale l_scale w_scale : ℝ) : Container :=
  { height := c.height * h_scale,
    length := c.length * l_scale,
    width := c.width * w_scale }

theorem container_volume_transformation (original : Container) :
  volume original = 4 →
  volume (scaleContainer original 2 3 4) = 96 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_transformation_l2766_276664


namespace NUMINAMATH_CALUDE_solution_equivalence_l2766_276614

theorem solution_equivalence :
  (∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2766_276614


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2766_276678

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2766_276678


namespace NUMINAMATH_CALUDE_set_equation_solution_l2766_276660

theorem set_equation_solution (p a b : ℝ) : 
  let A := {x : ℝ | x^2 - p*x + 15 = 0}
  let B := {x : ℝ | x^2 - a*x - b = 0}
  (A ∪ B = {2, 3, 5} ∧ A ∩ B = {3}) → (p = 8 ∧ a = 5 ∧ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l2766_276660


namespace NUMINAMATH_CALUDE_student_combinations_l2766_276639

/-- The number of possible combinations when n people each have 2 choices -/
def combinations (n : ℕ) : ℕ := 2^n

/-- There are 5 students -/
def num_students : ℕ := 5

/-- Theorem: The number of combinations for 5 students with 2 choices each is 32 -/
theorem student_combinations : combinations num_students = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_combinations_l2766_276639


namespace NUMINAMATH_CALUDE_members_playing_neither_in_given_club_l2766_276632

/-- Represents a music club with members playing different instruments -/
structure MusicClub where
  total : ℕ
  guitar : ℕ
  piano : ℕ
  both : ℕ

/-- Calculates the number of members who don't play either instrument -/
def membersPlayingNeither (club : MusicClub) : ℕ :=
  club.total - (club.guitar + club.piano - club.both)

/-- Theorem stating the number of members not playing either instrument in the given club -/
theorem members_playing_neither_in_given_club :
  let club : MusicClub := {
    total := 80,
    guitar := 50,
    piano := 40,
    both := 25
  }
  membersPlayingNeither club = 15 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_neither_in_given_club_l2766_276632


namespace NUMINAMATH_CALUDE_edward_spent_sixteen_l2766_276690

def edward_book_purchase (initial_amount : ℕ) (remaining_amount : ℕ) (num_books : ℕ) : Prop :=
  ∃ (amount_spent : ℕ), 
    initial_amount = remaining_amount + amount_spent ∧
    amount_spent = 16

theorem edward_spent_sixteen : 
  edward_book_purchase 22 6 92 :=
sorry

end NUMINAMATH_CALUDE_edward_spent_sixteen_l2766_276690


namespace NUMINAMATH_CALUDE_triangle_rotation_l2766_276605

theorem triangle_rotation (a₁ a₂ a₃ : ℝ) (h1 : 12 * a₁ = 360) (h2 : 6 * a₂ = 360) (h3 : a₁ + a₂ + a₃ = 180) :
  ∃ n : ℕ, n * a₃ ≥ 360 ∧ ∀ m : ℕ, m * a₃ ≥ 360 → n ≤ m ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_l2766_276605


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l2766_276611

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l2766_276611


namespace NUMINAMATH_CALUDE_prism_volume_l2766_276653

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2766_276653


namespace NUMINAMATH_CALUDE_phone_bill_percentage_abigail_phone_bill_l2766_276646

theorem phone_bill_percentage (initial_amount : ℝ) (food_percentage : ℝ) 
  (entertainment_cost : ℝ) (final_amount : ℝ) : ℝ :=
  let food_cost := initial_amount * food_percentage
  let after_food := initial_amount - food_cost
  let before_phone_bill := after_food - entertainment_cost
  let phone_bill_cost := before_phone_bill - final_amount
  let phone_bill_percentage := (phone_bill_cost / after_food) * 100
  phone_bill_percentage

theorem abigail_phone_bill : 
  phone_bill_percentage 200 0.6 20 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_percentage_abigail_phone_bill_l2766_276646


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2766_276676

theorem meaningful_expression_range (m : ℝ) : 
  (∃ (x : ℝ), x = (m - 1).sqrt / (m - 2)) ↔ (m ≥ 1 ∧ m ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2766_276676


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2766_276606

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  ∃ h k : ℝ,
  centerLine h k ∧
  requiredCircle x y ∧
  (x - h)^2 + (y - k)^2 = (x + 1)^2 + (y - 1)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2766_276606


namespace NUMINAMATH_CALUDE_square_roots_problem_l2766_276649

theorem square_roots_problem (n : ℝ) (h_pos : n > 0) :
  (∃ x : ℝ, (x + 1)^2 = n ∧ (4 - 2*x)^2 = n) → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2766_276649


namespace NUMINAMATH_CALUDE_composition_result_l2766_276644

/-- Given two functions f and g, prove that f(g(2)) = 169 -/
theorem composition_result (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = 2*x^2 + x + 3) : 
  f (g 2) = 169 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l2766_276644


namespace NUMINAMATH_CALUDE_inequality_proof_l2766_276682

theorem inequality_proof (m n a : ℝ) (h : m > n) : a - m < a - n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2766_276682


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2766_276688

theorem binomial_coefficient_sum (n : ℕ) : 
  let m := (4 : ℕ) ^ n
  let k := (2 : ℕ) ^ n
  m + k = 1056 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2766_276688


namespace NUMINAMATH_CALUDE_centroid_locus_is_hyperbola_l2766_276630

/-- Given two complex points Z₁ and Z₂ with arguments θ and -θ respectively, 
    where 0 < θ < π/2, and the area of triangle OZ₁Z₂ is constant S, 
    prove that the locus of the centroid Z of triangle OZ₁Z₂ forms a hyperbola. -/
theorem centroid_locus_is_hyperbola 
  (θ : ℝ) 
  (h_θ_pos : 0 < θ) 
  (h_θ_lt_pi_half : θ < π/2) 
  (S : ℝ) 
  (h_S_pos : S > 0) 
  (Z₁ Z₂ : ℂ) 
  (h_Z₁_arg : Complex.arg Z₁ = θ) 
  (h_Z₂_arg : Complex.arg Z₂ = -θ) 
  (h_area : abs (Z₁.im * Z₂.re - Z₁.re * Z₂.im) / 2 = S) : 
  ∃ (a b : ℝ), ∀ (Z : ℂ), Z = (Z₁ + Z₂) / 3 → (Z.re / a)^2 - (Z.im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_centroid_locus_is_hyperbola_l2766_276630


namespace NUMINAMATH_CALUDE_earnings_difference_is_400_l2766_276625

/-- Represents the amount of jade Nancy has in grams -/
def total_jade : ℕ := 1920

/-- Represents the amount of jade needed for a giraffe statue in grams -/
def giraffe_jade : ℕ := 120

/-- Represents the price of a giraffe statue in dollars -/
def giraffe_price : ℕ := 150

/-- Represents the amount of jade needed for an elephant statue in grams -/
def elephant_jade : ℕ := 2 * giraffe_jade

/-- Represents the price of an elephant statue in dollars -/
def elephant_price : ℕ := 350

/-- Calculates the earnings difference between making all elephant statues
    and all giraffe statues from the total jade -/
def earnings_difference : ℕ :=
  (total_jade / elephant_jade) * elephant_price - (total_jade / giraffe_jade) * giraffe_price

theorem earnings_difference_is_400 : earnings_difference = 400 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_is_400_l2766_276625


namespace NUMINAMATH_CALUDE_special_function_property_l2766_276665

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (f 2 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x) = x * f y + f x + x^2)

/-- The number of possible values for f(1/2) -/
def num_values (f : ℝ → ℝ) : ℕ :=
  sorry

/-- The sum of all possible values for f(1/2) -/
def sum_values (f : ℝ → ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  (num_values f : ℝ) * sum_values f = -2 :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l2766_276665


namespace NUMINAMATH_CALUDE_negation_existence_to_universal_negation_of_existence_proposition_l2766_276680

theorem negation_existence_to_universal (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existence_to_universal_negation_of_existence_proposition_l2766_276680


namespace NUMINAMATH_CALUDE_correct_reasoning_combination_l2766_276686

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that maps reasoning types to their directions
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct combination is Inductive, Deductive, and Analogical
theorem correct_reasoning_combination :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
sorry

end NUMINAMATH_CALUDE_correct_reasoning_combination_l2766_276686


namespace NUMINAMATH_CALUDE_equation_solutions_l2766_276634

theorem equation_solutions :
  (∃ x : ℚ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -5/2) ∧
  (∃ x : ℚ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39/35) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2766_276634


namespace NUMINAMATH_CALUDE_luke_stickers_to_sister_l2766_276608

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  initial + bought + birthday - used - final

/-- Theorem stating the number of stickers Luke gave to his sister -/
theorem luke_stickers_to_sister :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_to_sister_l2766_276608


namespace NUMINAMATH_CALUDE_yoojeong_rabbits_l2766_276675

/-- The number of animals Minyoung has -/
def minyoung_animals : ℕ := 9 + 3 + 5

/-- The number of animals Yoojeong has -/
def yoojeong_animals : ℕ := minyoung_animals + 2

/-- The number of dogs Yoojeong has -/
def yoojeong_dogs : ℕ := 7

theorem yoojeong_rabbits :
  ∃ (cats rabbits : ℕ),
    yoojeong_animals = yoojeong_dogs + cats + rabbits ∧
    cats = rabbits - 2 ∧
    rabbits = 7 := by sorry

end NUMINAMATH_CALUDE_yoojeong_rabbits_l2766_276675


namespace NUMINAMATH_CALUDE_direct_proportion_constant_factor_l2766_276684

theorem direct_proportion_constant_factor 
  (k : ℝ) (x y : ℝ → ℝ) (t : ℝ) :
  (∀ t, y t = k * x t) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → x t₁ ≠ x t₂ → y t₁ / x t₁ = y t₂ / x t₂) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_constant_factor_l2766_276684


namespace NUMINAMATH_CALUDE_circle_equation_implies_y_to_x_equals_nine_l2766_276628

theorem circle_equation_implies_y_to_x_equals_nine (x y : ℝ) : 
  x^2 + y^2 - 4*x + 6*y + 13 = 0 → y^x = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_y_to_x_equals_nine_l2766_276628


namespace NUMINAMATH_CALUDE_brendas_age_l2766_276633

theorem brendas_age (addison janet brenda : ℝ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 10) 
  (h3 : addison = janet) : 
  brenda = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l2766_276633


namespace NUMINAMATH_CALUDE_triangle_area_l2766_276677

theorem triangle_area (A B C : Real) (a b c : Real) : 
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  -- Given conditions
  (Real.cos B = 1/4) →
  (b = 3) →
  (Real.sin C = 2 * Real.sin A) →
  -- Conclusion
  (1/2 * a * c * Real.sin B = Real.sqrt 15 / 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2766_276677


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l2766_276629

/-- The length of a rectangular metallic sheet, given its width and the dimensions of an open box formed from it. -/
theorem metallic_sheet_length (w h v : ℝ) (hw : w = 36) (hh : h = 8) (hv : v = 5440) : ∃ l : ℝ,
  l = 50 ∧ v = (l - 2 * h) * (w - 2 * h) * h :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l2766_276629


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2766_276681

theorem sufficient_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  ∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2766_276681


namespace NUMINAMATH_CALUDE_jane_sequin_count_l2766_276643

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Represents the sequin count problem for Jane's costume -/
def sequinCount : Prop :=
  let blueStars := 10 * 12
  let purpleSquares := 8 * 15
  let greenHexagons := 14 * 20
  let redCircles := arithmeticSum 10 5 5
  blueStars + purpleSquares + greenHexagons + redCircles = 620

theorem jane_sequin_count : sequinCount := by
  sorry

end NUMINAMATH_CALUDE_jane_sequin_count_l2766_276643


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2766_276651

/-- Given two plane vectors a and b with an angle of 120° between them,
    |a| = 1, |b| = 2, and a vector m satisfying m · a = m · b = 1,
    prove that |m| = √21/3 -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) :
  (∃ θ : ℝ, θ = 2 * π / 3 ∧ a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos θ) →
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  m • a = 1 →
  m • b = 1 →
  ‖m‖ = Real.sqrt 21 / 3 := by
  sorry

#check vector_magnitude_problem

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2766_276651


namespace NUMINAMATH_CALUDE_right_triangle_area_l2766_276650

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2766_276650


namespace NUMINAMATH_CALUDE_bed_sheet_problem_l2766_276647

/-- Calculates the length of a bed-sheet in meters given the total cutting time, 
    time per cut, and length of each piece. -/
def bed_sheet_length (total_time : ℕ) (time_per_cut : ℕ) (piece_length : ℕ) : ℚ :=
  (total_time / time_per_cut) * piece_length / 100

/-- Proves that a bed-sheet cut into 20cm pieces, taking 5 minutes per cut and 
    245 minutes total, is 9.8 meters long. -/
theorem bed_sheet_problem : bed_sheet_length 245 5 20 = 9.8 := by
  sorry

#eval bed_sheet_length 245 5 20

end NUMINAMATH_CALUDE_bed_sheet_problem_l2766_276647


namespace NUMINAMATH_CALUDE_theater_attendance_l2766_276657

/-- The number of men who spent Rs. 3 each on tickets -/
def num_men_standard : ℕ := 8

/-- The amount spent by each of the standard-paying men -/
def standard_price : ℚ := 3

/-- The total amount spent by all men -/
def total_spent : ℚ := 29.25

/-- The extra amount spent by the last man compared to the average -/
def extra_spent : ℚ := 2

theorem theater_attendance :
  ∃ (n : ℕ), n > 0 ∧
  (n : ℚ) * (total_spent / n) = 
    num_men_standard * standard_price + (total_spent / n + extra_spent) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_theater_attendance_l2766_276657


namespace NUMINAMATH_CALUDE_odd_cycle_existence_l2766_276697

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- The minimum degree of a graph is the minimum of the degrees of all vertices. -/
def min_degree {V : Type} (G : Graph V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each adjacent pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (p : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (c : List V) : Prop := sorry

/-- The length of a cycle is the number of edges in the cycle. -/
def cycle_length {V : Type} (c : List V) : ℕ := sorry

/-- A theorem stating that any graph with minimum degree at least 3 contains an odd cycle. -/
theorem odd_cycle_existence {V : Type} (G : Graph V) :
  min_degree G ≥ 3 → ∃ c : List V, is_cycle G c ∧ Odd (cycle_length c) := by
  sorry

end NUMINAMATH_CALUDE_odd_cycle_existence_l2766_276697


namespace NUMINAMATH_CALUDE_fathers_age_l2766_276640

theorem fathers_age (son_age father_age : ℕ) : 
  son_age = 10 →
  father_age = 4 * son_age →
  father_age + 20 = 2 * (son_age + 20) →
  father_age = 40 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_l2766_276640


namespace NUMINAMATH_CALUDE_power_sum_equality_two_variables_power_sum_equality_three_variables_l2766_276636

-- Part (a)
theorem power_sum_equality_two_variables (x y u v : ℝ) (h1 : x + y = u + v) (h2 : x^2 + y^2 = u^2 + v^2) :
  ∀ n : ℕ, x^n + y^n = u^n + v^n := by sorry

-- Part (b)
theorem power_sum_equality_three_variables (x y z u v t : ℝ) 
  (h1 : x + y + z = u + v + t) 
  (h2 : x^2 + y^2 + z^2 = u^2 + v^2 + t^2) 
  (h3 : x^3 + y^3 + z^3 = u^3 + v^3 + t^3) :
  ∀ n : ℕ, x^n + y^n + z^n = u^n + v^n + t^n := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_two_variables_power_sum_equality_three_variables_l2766_276636


namespace NUMINAMATH_CALUDE_marcus_and_leah_games_l2766_276600

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The function to calculate the number of games where two specific players play together -/
def games_together (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 2) (k - 2)

/-- The theorem stating that Marcus and Leah play together in 210 games -/
theorem marcus_and_leah_games : 
  games_together total_players players_per_game = 210 := by
  sorry

#eval games_together total_players players_per_game

end NUMINAMATH_CALUDE_marcus_and_leah_games_l2766_276600


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2766_276604

/-- A hyperbola with center at the origin and axes of symmetry along the coordinate axes -/
structure CenteredHyperbola where
  /-- The angle of inclination of one of the asymptotes -/
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : CenteredHyperbola) : ℝ :=
  sorry

/-- Theorem stating the possible eccentricities of a hyperbola with an asymptote angle of π/3 -/
theorem hyperbola_eccentricity (h : CenteredHyperbola) 
  (h_angle : h.asymptote_angle = π / 3) : 
  eccentricity h = 2 ∨ eccentricity h = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2766_276604


namespace NUMINAMATH_CALUDE_inequality_proof_l2766_276641

theorem inequality_proof (a b c : ℝ) (m n k : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ (a^(m:ℝ) * b^(n:ℝ) * c^(k:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(n:ℝ) * b^(k:ℝ) * c^(m:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(k:ℝ) * b^(m:ℝ) * c^(n:ℝ))^(1/(m+n+k:ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2766_276641


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2766_276699

theorem sqrt_D_irrational (a : ℤ) : 
  let b : ℤ := a + 2
  let c : ℤ := a^2 + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by
sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2766_276699


namespace NUMINAMATH_CALUDE_carlson_ate_66_candies_l2766_276662

/-- Represents the number of candies eaten by Carlson given the initial conditions --/
def carlson_candies : ℕ :=
  let initial_candies : ℕ := 300
  let boy_daily_consumption : ℕ := 1
  let carlson_sunday_consumption : ℕ := 2
  let days_per_week : ℕ := 7
  let start_day : ℕ := 2  -- Tuesday (0-based index, where 0 is Sunday)

  let weekly_consumption : ℕ := boy_daily_consumption * days_per_week + carlson_sunday_consumption
  let complete_weeks : ℕ := initial_candies / weekly_consumption

  complete_weeks * carlson_sunday_consumption

/-- Theorem stating that Carlson ate 66 candies --/
theorem carlson_ate_66_candies : carlson_candies = 66 := by
  sorry

end NUMINAMATH_CALUDE_carlson_ate_66_candies_l2766_276662


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l2766_276683

theorem complex_sum_equals_i : Complex.I + 1 + Complex.I^2 = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l2766_276683


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2766_276642

theorem polynomial_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2766_276642


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2766_276689

/-- The universal set U is the set of positive integers less than or equal to a -/
def U (a : ℝ) : Set ℕ := {x : ℕ | x > 0 ∧ x ≤ ⌊a⌋}

/-- Set P -/
def P : Set ℕ := {1, 2, 3}

/-- Set Q -/
def Q : Set ℕ := {4, 5, 6}

/-- The complement of set A in the universal set U -/
def complement (a : ℝ) (A : Set ℕ) : Set ℕ := (U a) \ A

theorem necessary_and_sufficient_condition (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ complement a P = Q := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2766_276689


namespace NUMINAMATH_CALUDE_mark_buys_extra_large_bags_l2766_276620

/-- Represents the types of balloon bags available --/
inductive BagType
  | Small
  | Medium
  | ExtraLarge

/-- Represents a bag of balloons with its price and quantity --/
structure BalloonBag where
  bagType : BagType
  price : ℕ
  quantity : ℕ

def mark_budget : ℕ := 24
def small_bag : BalloonBag := ⟨BagType.Small, 4, 50⟩
def extra_large_bag : BalloonBag := ⟨BagType.ExtraLarge, 12, 200⟩
def total_balloons : ℕ := 400

/-- Calculates the number of bags that can be bought with a given budget --/
def bags_bought (bag : BalloonBag) (budget : ℕ) : ℕ :=
  budget / bag.price

/-- Calculates the total number of balloons from a given number of bags --/
def total_balloons_from_bags (bag : BalloonBag) (num_bags : ℕ) : ℕ :=
  num_bags * bag.quantity

theorem mark_buys_extra_large_bags :
  bags_bought extra_large_bag mark_budget = 2 ∧
  total_balloons_from_bags extra_large_bag (bags_bought extra_large_bag mark_budget) = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_mark_buys_extra_large_bags_l2766_276620


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2766_276687

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2766_276687
