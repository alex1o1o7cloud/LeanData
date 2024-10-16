import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l1238_123838

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1238_123838


namespace NUMINAMATH_CALUDE_transformed_triangle_area_l1238_123837

-- Define the function f on the domain {x₁, x₂, x₃}
variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the area of a triangle given three points
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area 
  (h1 : triangleArea (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 27) :
  triangleArea (x₁/2, 3 * f x₁) (x₂/2, 3 * f x₂) (x₃/2, 3 * f x₃) = 40.5 :=
sorry

end NUMINAMATH_CALUDE_transformed_triangle_area_l1238_123837


namespace NUMINAMATH_CALUDE_money_left_after_trip_l1238_123882

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip :
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_trip_l1238_123882


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1238_123831

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b < a ∧ a < 0 → 1/b > 1/a) ∧
  ¬(∀ a b : ℝ, 1/b > 1/a → b < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1238_123831


namespace NUMINAMATH_CALUDE_oak_grove_total_books_l1238_123829

def public_library_books : ℕ := 1986
def school_library_books : ℕ := 5106

theorem oak_grove_total_books :
  public_library_books + school_library_books = 7092 :=
by sorry

end NUMINAMATH_CALUDE_oak_grove_total_books_l1238_123829


namespace NUMINAMATH_CALUDE_grid_coloring_inequality_l1238_123871

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the grid -/
def Grid (n : ℕ) := Fin n → Fin n → Cell

/-- Counts the number of black cells adjacent to a vertex -/
def countAdjacentBlack (g : Grid n) (i j : Fin n) : ℕ := sorry

/-- Determines if a vertex is red -/
def isRed (g : Grid n) (i j : Fin n) : Bool :=
  Odd (countAdjacentBlack g i j)

/-- Counts the number of red vertices in the grid -/
def countRedVertices (g : Grid n) : ℕ := sorry

/-- Represents an operation to change colors in a rectangle -/
structure Operation (n : ℕ) where
  topLeft : Fin n × Fin n
  bottomRight : Fin n × Fin n

/-- Applies an operation to the grid -/
def applyOperation (g : Grid n) (op : Operation n) : Grid n := sorry

/-- Checks if the grid is entirely white -/
def isAllWhite (g : Grid n) : Bool := sorry

/-- The minimum number of operations to make the grid white -/
noncomputable def minOperations (g : Grid n) : ℕ := sorry

theorem grid_coloring_inequality (n : ℕ) (g : Grid n) :
  let Y := countRedVertices g
  let X := minOperations g
  Y / 4 ≤ X ∧ X ≤ Y / 2 := by sorry

end NUMINAMATH_CALUDE_grid_coloring_inequality_l1238_123871


namespace NUMINAMATH_CALUDE_book_selling_price_l1238_123899

theorem book_selling_price (cost_price : ℚ) 
  (h1 : cost_price * (1 + 1/10) = 550) 
  (h2 : ∃ original_price : ℚ, original_price = cost_price * (1 - 1/10)) : 
  ∃ original_price : ℚ, original_price = 450 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l1238_123899


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1238_123866

/-- Given a hyperbola with eccentricity 2 and the same foci as the ellipse x²/25 + y²/9 = 1,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2/25 + y^2/9 = 1 → 
      ∃ (c : ℝ), c = 4 ∧ 
        (∀ (x y : ℝ), (x + c)^2 + y^2 = 25 ∨ (x - c)^2 + y^2 = 25)) ∧
    (∃ (c : ℝ), c/a = 2 ∧ c = 4)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1238_123866


namespace NUMINAMATH_CALUDE_clothing_store_profit_model_l1238_123842

/-- Represents the clothing store's sales and profit model -/
structure ClothingStore where
  originalCost : ℝ
  originalPrice : ℝ
  originalSales : ℝ
  salesIncrease : ℝ
  priceReduction : ℝ

/-- Calculate daily sales after price reduction -/
def dailySales (store : ClothingStore) : ℝ :=
  store.originalSales + store.salesIncrease * store.priceReduction

/-- Calculate profit per piece after price reduction -/
def profitPerPiece (store : ClothingStore) : ℝ :=
  store.originalPrice - store.originalCost - store.priceReduction

/-- Calculate total daily profit -/
def dailyProfit (store : ClothingStore) : ℝ :=
  dailySales store * profitPerPiece store

/-- The main theorem about the clothing store's profit model -/
theorem clothing_store_profit_model (store : ClothingStore) 
  (h1 : store.originalCost = 80)
  (h2 : store.originalPrice = 120)
  (h3 : store.originalSales = 20)
  (h4 : store.salesIncrease = 2) :
  (∀ x, dailySales { store with priceReduction := x } = 20 + 2 * x) ∧
  (∀ x, profitPerPiece { store with priceReduction := x } = 40 - x) ∧
  (dailyProfit { store with priceReduction := 20 } = 1200) ∧
  (∀ x, dailyProfit { store with priceReduction := x } ≠ 2000) := by
  sorry


end NUMINAMATH_CALUDE_clothing_store_profit_model_l1238_123842


namespace NUMINAMATH_CALUDE_matts_baseball_cards_value_l1238_123851

theorem matts_baseball_cards_value (n : ℕ) (x : ℚ) : 
  n = 8 →  -- Matt has 8 baseball cards
  2 * x + 3 = (3 * 2 + 9) →  -- He trades 2 cards for 3 $2 cards and 1 $9 card, making a $3 profit
  x = 6 :=  -- Each of Matt's baseball cards is worth $6
by
  sorry

end NUMINAMATH_CALUDE_matts_baseball_cards_value_l1238_123851


namespace NUMINAMATH_CALUDE_perpendicular_line_theorem_l1238_123817

/-- A figure in a plane -/
inductive PlaneFigure
  | Triangle
  | Trapezoid
  | CircleDiameters
  | HexagonSides

/-- Represents whether two lines in a figure are guaranteed to intersect -/
def guaranteed_intersection (figure : PlaneFigure) : Prop :=
  match figure with
  | PlaneFigure.Triangle => true
  | PlaneFigure.Trapezoid => false
  | PlaneFigure.CircleDiameters => true
  | PlaneFigure.HexagonSides => false

/-- A line perpendicular to two sides of a figure is perpendicular to the plane -/
def perpendicular_to_plane (figure : PlaneFigure) : Prop :=
  guaranteed_intersection figure

theorem perpendicular_line_theorem (figure : PlaneFigure) :
  perpendicular_to_plane figure ↔ (figure = PlaneFigure.Triangle ∨ figure = PlaneFigure.CircleDiameters) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_theorem_l1238_123817


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_over_2_l1238_123819

noncomputable def f (x : ℝ) := Real.sin x - 2 * Real.cos x

theorem tangent_line_at_pi_over_2 :
  let Q : ℝ × ℝ := (π / 2, 1)
  let m : ℝ := Real.cos (π / 2) + 2 * Real.sin (π / 2)
  let tangent_line (x : ℝ) := m * (x - Q.1) + Q.2
  ∀ x, tangent_line x = 2 * x - π + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_over_2_l1238_123819


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l1238_123874

theorem sqrt_expression_equals_two :
  Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 - abs (2 - Real.sqrt 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l1238_123874


namespace NUMINAMATH_CALUDE_tan_product_special_angles_l1238_123878

theorem tan_product_special_angles : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_special_angles_l1238_123878


namespace NUMINAMATH_CALUDE_fraction_in_lowest_terms_l1238_123898

theorem fraction_in_lowest_terms (n : ℤ) (h : Odd n) :
  Nat.gcd (Int.natAbs (2 * n + 2)) (Int.natAbs (3 * n + 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_in_lowest_terms_l1238_123898


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1238_123887

def vector_a (x : ℝ) : ℝ × ℝ := (6, x)

theorem sufficient_not_necessary :
  ∃ (x : ℝ), (x ≠ 8 ∧ ‖vector_a x‖ = 10) ∧
  ∀ (x : ℝ), (x = 8 → ‖vector_a x‖ = 10) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1238_123887


namespace NUMINAMATH_CALUDE_correct_division_result_l1238_123815

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_division_result_l1238_123815


namespace NUMINAMATH_CALUDE_total_money_l1238_123825

/-- Given three people A, B, and C with some money, prove their total amount. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 250)
  (hBC : B + C = 450)
  (hC : C = 100) : 
  A + B + C = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1238_123825


namespace NUMINAMATH_CALUDE_factor_expression_l1238_123896

theorem factor_expression (x : ℝ) : 54 * x^6 - 231 * x^13 = 3 * x^6 * (18 - 77 * x^7) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1238_123896


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1238_123881

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1238_123881


namespace NUMINAMATH_CALUDE_inequality_proof_l1238_123897

theorem inequality_proof (a b c d : ℝ) (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1238_123897


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1238_123816

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop :=
  2 * a * x^2 + 4 * x + b ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x = -1/a}

-- State the theorem
theorem quadratic_inequality_theorem (a b : ℝ) 
  (h1 : ∀ x, x ∈ solution_set a ↔ quadratic_inequality a b x)
  (h2 : a > b) :
  (ab = 2) ∧ 
  (∀ a b, (2*a + b^3) / (2 - b^2) ≥ 4) ∧
  (∃ a b, (2*a + b^3) / (2 - b^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1238_123816


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l1238_123886

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 8) / (a_n + 3) for n ∈ ℕ* is 5/2,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem min_value_arithmetic_sequence (d : ℝ) (h_d : d ≠ 0) :
  let a : ℕ → ℝ := λ n => 1 + (n - 1) * d
  let S : ℕ → ℝ := λ n => (n * (2 + (n - 1) * d)) / 2
  (a 1 = 1) ∧ 
  (a 1 * a 13 = (a 3)^2) →
  (∃ (n : ℕ), n > 0 ∧ (2 * S n + 8) / (a n + 3) = 5/2) ∧
  (∀ (n : ℕ), n > 0 → (2 * S n + 8) / (a n + 3) ≥ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l1238_123886


namespace NUMINAMATH_CALUDE_reciprocal_difference_fractions_l1238_123848

theorem reciprocal_difference_fractions : (1 / (1/4 - 1/5) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_fractions_l1238_123848


namespace NUMINAMATH_CALUDE_fraction_problem_l1238_123839

theorem fraction_problem : ∃ (a b : ℤ), 
  (a - 1 : ℚ) / b = 2/3 ∧ 
  (a - 2 : ℚ) / b = 1/2 ∧ 
  (a : ℚ) / b = 5/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1238_123839


namespace NUMINAMATH_CALUDE_function_periodicity_l1238_123835

/-- A function satisfying the given functional equation is periodic with period 4 -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 4) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1238_123835


namespace NUMINAMATH_CALUDE_no_real_curve_exists_l1238_123880

theorem no_real_curve_exists : ¬ ∃ (x y : ℝ), x^2 + y^2 - 2*x + 4*y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_curve_exists_l1238_123880


namespace NUMINAMATH_CALUDE_correct_calculation_l1238_123875

theorem correct_calculation (x : ℤ) : 954 - x = 468 → 954 + x = 1440 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1238_123875


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1238_123857

theorem cricket_bat_profit_percentage
  (cost_price_A : ℝ)
  (selling_price_C : ℝ)
  (profit_percentage_B : ℝ)
  (h1 : cost_price_A = 156)
  (h2 : selling_price_C = 234)
  (h3 : profit_percentage_B = 25)
  : (((selling_price_C / (1 + profit_percentage_B / 100)) - cost_price_A) / cost_price_A) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1238_123857


namespace NUMINAMATH_CALUDE_non_student_ticket_cost_l1238_123867

theorem non_student_ticket_cost
  (total_tickets : ℕ)
  (student_ticket_cost : ℚ)
  (total_amount : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 193)
  (h2 : student_ticket_cost = 1/2)
  (h3 : total_amount = 412/2)
  (h4 : student_tickets = 83) :
  (total_amount - student_ticket_cost * student_tickets) / (total_tickets - student_tickets) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_non_student_ticket_cost_l1238_123867


namespace NUMINAMATH_CALUDE_function_minimum_at_one_l1238_123850

/-- The function f(x) = (x^2 + 1) / (x + a) has a minimum at x = 1 -/
theorem function_minimum_at_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => (x^2 + 1) / (x + a)
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), x ≠ -a → f x ≥ f 1 :=
by
  sorry

end NUMINAMATH_CALUDE_function_minimum_at_one_l1238_123850


namespace NUMINAMATH_CALUDE_f_at_one_l1238_123822

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_at_one_l1238_123822


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_one_fourth_l1238_123820

theorem sin_15_cos_15_eq_one_fourth : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_one_fourth_l1238_123820


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_4_mod_7_l1238_123845

theorem largest_integer_less_than_100_with_remainder_4_mod_7 : ∃ n : ℕ, n = 95 ∧ 
  (∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n) ∧ n < 100 ∧ n % 7 = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_4_mod_7_l1238_123845


namespace NUMINAMATH_CALUDE_range_of_a_l1238_123856

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1238_123856


namespace NUMINAMATH_CALUDE_speedster_convertibles_l1238_123870

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = (2 : ℚ) / 3 * total →
  total - speedsters = 50 →
  convertibles = (4 : ℚ) / 5 * speedsters →
  convertibles = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l1238_123870


namespace NUMINAMATH_CALUDE_largest_office_number_l1238_123868

def house_number : List Nat := [9, 0, 2, 3, 4]

def office_number_sum : Nat := house_number.sum

def is_valid_office_number (n : List Nat) : Prop :=
  n.length = 10 ∧ n.sum = office_number_sum

def lexicographically_greater (a b : List Nat) : Prop :=
  ∃ i, (∀ j < i, a.get! j = b.get! j) ∧ a.get! i > b.get! i

theorem largest_office_number :
  ∃ (max : List Nat),
    is_valid_office_number max ∧
    (∀ n, is_valid_office_number n → lexicographically_greater max n ∨ max = n) ∧
    max = [9, 0, 5, 4, 0, 0, 0, 0, 0, 4] :=
sorry

end NUMINAMATH_CALUDE_largest_office_number_l1238_123868


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1238_123858

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * ((-2 : ℂ) + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1238_123858


namespace NUMINAMATH_CALUDE_equation_solutions_l1238_123808

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10)
  ∀ x : ℝ, f x = 0 ↔ x = -2 + 2*Real.sqrt 14 ∨ x = -2 - 2*Real.sqrt 14 ∨ 
                    x = (7 + Real.sqrt 89) / 2 ∨ x = (7 - Real.sqrt 89) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1238_123808


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_a_range_l1238_123865

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + (2*a - 2)*x
  else x^3 - (3*a + 3)*x^2 + a*x

-- Define the derivative of f(x)
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x + 2*a - 2
  else 3*x^2 - 6*(a + 1)*x + a

-- Theorem statement
theorem parallel_tangents_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f_prime a x₁ = f_prime a x₂ ∧ f_prime a x₂ = f_prime a x₃) →
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_a_range_l1238_123865


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1238_123849

theorem geometric_sequence_property (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n = 4 * q^(n-1)) →  -- a_n is a geometric sequence with a_1 = 4 and common ratio q
  (∀ n, S n = 4 * (1 - q^n) / (1 - q)) →  -- S_n is the sum of the first n terms
  (∃ r, ∀ n, (S (n+1) + 2) = r * (S n + 2)) →  -- {S_n + 2} is a geometric sequence
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1238_123849


namespace NUMINAMATH_CALUDE_john_earnings_160_l1238_123841

/-- Calculates John's weekly streaming earnings --/
def johnWeeklyEarnings (daysOff : ℕ) (hoursPerDay : ℕ) (ratePerHour : ℕ) : ℕ :=
  let daysStreaming := 7 - daysOff
  let hoursPerWeek := daysStreaming * hoursPerDay
  hoursPerWeek * ratePerHour

/-- Theorem: John's weekly earnings are $160 --/
theorem john_earnings_160 :
  johnWeeklyEarnings 3 4 10 = 160 := by
  sorry

#eval johnWeeklyEarnings 3 4 10

end NUMINAMATH_CALUDE_john_earnings_160_l1238_123841


namespace NUMINAMATH_CALUDE_no_factors_of_p_l1238_123846

def p (x : ℝ) : ℝ := x^4 - 3*x^2 + 5

theorem no_factors_of_p :
  (∀ x, p x ≠ (x^2 + 1) * (x^2 - 3*x + 5)) ∧
  (∀ x, p x ≠ (x - 1) * (x^3 + x^2 - 2*x - 5)) ∧
  (∀ x, p x ≠ (x^2 + 5) * (x^2 - 5)) ∧
  (∀ x, p x ≠ (x^2 + 2*x + 1) * (x^2 - 2*x + 4)) :=
by
  sorry

#check no_factors_of_p

end NUMINAMATH_CALUDE_no_factors_of_p_l1238_123846


namespace NUMINAMATH_CALUDE_equation_solution_l1238_123824

theorem equation_solution :
  let s : ℚ := 20
  let r : ℚ := 270 / 7
  2 * (r - 45) / 3 = (3 * s - 2 * r) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1238_123824


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l1238_123883

/-- Calculates the remaining money after Lidia's app purchase --/
def remaining_money (productivity_apps : ℕ) (productivity_cost : ℚ)
                    (gaming_apps : ℕ) (gaming_cost : ℚ)
                    (lifestyle_apps : ℕ) (lifestyle_cost : ℚ)
                    (initial_money : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := productivity_apps * productivity_cost +
                    gaming_apps * gaming_cost +
                    lifestyle_apps * lifestyle_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_money - final_cost

/-- Theorem stating that Lidia will be left with $6.16 after her app purchase --/
theorem lidia_remaining_money :
  remaining_money 5 4 7 5 3 3 66 (15/100) (10/100) = (616/100) :=
sorry


end NUMINAMATH_CALUDE_lidia_remaining_money_l1238_123883


namespace NUMINAMATH_CALUDE_students_per_computer_l1238_123877

theorem students_per_computer :
  let initial_students : ℕ := 82
  let additional_students : ℕ := 16
  let total_students : ℕ := initial_students + additional_students
  let computers_after_increase : ℕ := 49
  let students_per_computer : ℚ := initial_students / (total_students / computers_after_increase : ℚ)
  students_per_computer = 2 := by
sorry

end NUMINAMATH_CALUDE_students_per_computer_l1238_123877


namespace NUMINAMATH_CALUDE_brainiac_survey_l1238_123832

theorem brainiac_survey (R M : ℕ) : 
  R = 2 * M →                   -- Twice as many like rebus as math
  18 ≤ M ∧ 18 ≤ R →             -- 18 like both rebus and math
  20 ≤ M →                      -- 20 like math but not rebus
  R + M - 18 + 4 = 100          -- Total surveyed is 100
  := by sorry

end NUMINAMATH_CALUDE_brainiac_survey_l1238_123832


namespace NUMINAMATH_CALUDE_not_divisible_3n_minus_1_by_2n_minus_1_l1238_123892

theorem not_divisible_3n_minus_1_by_2n_minus_1 (n : ℕ) (h : n > 1) :
  ¬(2^n - 1 ∣ 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_3n_minus_1_by_2n_minus_1_l1238_123892


namespace NUMINAMATH_CALUDE_at_least_one_passes_l1238_123833

/-- Represents the number of questions in the exam pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions A can answer correctly -/
def a_correct : ℕ := 6

/-- Represents the number of questions B can answer correctly -/
def b_correct : ℕ := 8

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the minimum number of correct answers needed to pass -/
def pass_threshold : ℕ := 2

/-- Calculates the probability of an event -/
def prob (favorable : ℕ) (total : ℕ) : ℚ := (favorable : ℚ) / (total : ℚ)

/-- Calculates the probability of person A passing the exam -/
def prob_a_pass : ℚ := 
  prob (Nat.choose a_correct 2 * Nat.choose (total_questions - a_correct) 1 + 
        Nat.choose a_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Calculates the probability of person B passing the exam -/
def prob_b_pass : ℚ := 
  prob (Nat.choose b_correct 2 * Nat.choose (total_questions - b_correct) 1 + 
        Nat.choose b_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Theorem: The probability that at least one person passes the exam is 44/45 -/
theorem at_least_one_passes : 
  1 - (1 - prob_a_pass) * (1 - prob_b_pass) = 44 / 45 := by sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l1238_123833


namespace NUMINAMATH_CALUDE_f_monotone_increasing_local_max_condition_l1238_123812

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (x - 1) + Real.log x - (a + 1) * x

theorem f_monotone_increasing (x : ℝ) (hx : x > 0) :
  let f₁ := f 1
  (deriv f₁) x ≥ 0 := by sorry

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a x ≤ f a 1) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_local_max_condition_l1238_123812


namespace NUMINAMATH_CALUDE_raccoons_pepper_sprayed_l1238_123813

theorem raccoons_pepper_sprayed (num_raccoons : ℕ) (num_squirrels : ℕ) : 
  num_squirrels = 6 * num_raccoons →
  num_raccoons + num_squirrels = 84 →
  num_raccoons = 12 := by
sorry

end NUMINAMATH_CALUDE_raccoons_pepper_sprayed_l1238_123813


namespace NUMINAMATH_CALUDE_paper_area_difference_l1238_123801

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem paper_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by sorry

end NUMINAMATH_CALUDE_paper_area_difference_l1238_123801


namespace NUMINAMATH_CALUDE_linear_function_translation_l1238_123852

/-- Given a linear function y = 2x, when translated 3 units to the right along the x-axis,
    the resulting function is y = 2x - 6. -/
theorem linear_function_translation (x y : ℝ) :
  (y = 2 * x) →
  (y = 2 * (x - 3)) →
  (y = 2 * x - 6) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_translation_l1238_123852


namespace NUMINAMATH_CALUDE_evaluate_expression_l1238_123802

theorem evaluate_expression (a : ℝ) :
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1238_123802


namespace NUMINAMATH_CALUDE_sweater_cost_l1238_123840

theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 91 → 
  tshirt_cost = 6 → 
  shoes_cost = 11 → 
  remaining_amount = 50 → 
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_sweater_cost_l1238_123840


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1238_123844

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1238_123844


namespace NUMINAMATH_CALUDE_percentage_with_repeat_approx_l1238_123847

/-- The count of all possible five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def no_repeat_numbers : ℕ := 27216

/-- The count of five-digit numbers with at least one repeated digit -/
def repeat_numbers : ℕ := total_five_digit_numbers - no_repeat_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_with_repeat : ℚ :=
  (repeat_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_with_repeat_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 10 ∧ 
  abs (percentage_with_repeat - 698 / 10) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_with_repeat_approx_l1238_123847


namespace NUMINAMATH_CALUDE_problem_solution_l1238_123893

theorem problem_solution (x y : Real) 
  (h1 : x + Real.cos y = 3009)
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3009 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1238_123893


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l1238_123891

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l1238_123891


namespace NUMINAMATH_CALUDE_no_solution_exists_l1238_123855

theorem no_solution_exists : ¬∃ (x y : ℕ+), 
  (x^(y:ℕ) + 3 = y^(x:ℕ)) ∧ (3 * x^(y:ℕ) = y^(x:ℕ) + 8) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1238_123855


namespace NUMINAMATH_CALUDE_square_adjustment_theorem_l1238_123873

theorem square_adjustment_theorem (a b : ℤ) (k : ℤ) : 
  (∃ (b : ℤ), b^2 = a^2 + 2*k ∨ b^2 = a^2 - 2*k) → 
  (∃ (c : ℤ), a^2 + k = c^2 + (b-a)^2 ∨ a^2 - k = c^2 + (b-a)^2) :=
sorry

end NUMINAMATH_CALUDE_square_adjustment_theorem_l1238_123873


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1238_123843

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (25 - Real.sqrt n) = 3 → n = 256 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1238_123843


namespace NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l1238_123800

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- The other endpoint of a diameter given the circle and one endpoint --/
def otherDiameterEndpoint (c : Circle) (p : Point) : Point :=
  (2 * c.center.1 - p.1, 2 * c.center.2 - p.2)

theorem diameter_endpoint_coordinates :
  let c : Circle := { center := (3, 5) }
  let p : Point := (0, 1)
  otherDiameterEndpoint c p = (6, 9) := by
  sorry

#check diameter_endpoint_coordinates

end NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l1238_123800


namespace NUMINAMATH_CALUDE_chessboard_covering_impossible_l1238_123888

/-- Represents a chessboard with given dimensions -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile with given dimensions -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Determines if a chessboard can be covered by a given number of tiles -/
def can_cover (board : Chessboard) (tile : Tile) (num_tiles : Nat) : Prop :=
  ∃ (arrangement : Nat), 
    arrangement > 0 ∧ 
    tile.length * tile.width * num_tiles = board.rows * board.cols

/-- The main theorem stating that a 10x10 chessboard cannot be covered by 25 4x1 tiles -/
theorem chessboard_covering_impossible :
  ¬(can_cover (Chessboard.mk 10 10) (Tile.mk 4 1) 25) :=
sorry

end NUMINAMATH_CALUDE_chessboard_covering_impossible_l1238_123888


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1238_123805

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 + 2*I) / I
  Complex.im z = -1 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1238_123805


namespace NUMINAMATH_CALUDE_apps_deleted_l1238_123876

theorem apps_deleted (initial_apps new_apps final_apps : ℕ) :
  initial_apps = 10 →
  new_apps = 11 →
  final_apps = 4 →
  initial_apps + new_apps - final_apps = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l1238_123876


namespace NUMINAMATH_CALUDE_smallest_d_value_l1238_123806

theorem smallest_d_value (d : ℝ) : 
  (∃ d₀ : ℝ, d₀ > 0 ∧ Real.sqrt (40 + (4 * d₀ - 2)^2) = 10 * d₀ ∧
   ∀ d' : ℝ, d' > 0 → Real.sqrt (40 + (4 * d' - 2)^2) = 10 * d' → d₀ ≤ d') →
  d = (4 + Real.sqrt 940) / 42 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l1238_123806


namespace NUMINAMATH_CALUDE_spinner_probability_l1238_123834

theorem spinner_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1238_123834


namespace NUMINAMATH_CALUDE_probability_below_25_major_b_l1238_123854

/-- Represents the probability distribution of students in a graduating class -/
structure GraduatingClass where
  male_percentage : Real
  female_percentage : Real
  major_b_percentage : Real
  major_b_below_25_percentage : Real

/-- Theorem stating the probability of a randomly selected student being less than 25 years old and enrolled in Major B -/
theorem probability_below_25_major_b (gc : GraduatingClass) 
  (h1 : gc.male_percentage + gc.female_percentage = 1)
  (h2 : gc.major_b_percentage = 0.3)
  (h3 : gc.major_b_below_25_percentage = 0.6) :
  gc.major_b_percentage * gc.major_b_below_25_percentage = 0.18 := by
  sorry

#check probability_below_25_major_b

end NUMINAMATH_CALUDE_probability_below_25_major_b_l1238_123854


namespace NUMINAMATH_CALUDE_arithmetic_and_another_seq_sum_l1238_123803

/-- An arithmetic sequence with the given properties -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ 
  ∃ q : ℝ, q * a 2 = a 3 + 1 ∧ q * (a 3 + 1) = a 4

/-- Another sequence with the given sum property -/
def another_seq (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 / S n = 1 / n - 1 / (n + 1)

/-- The main theorem -/
theorem arithmetic_and_another_seq_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_seq a → another_seq b S → a 8 + b 8 = 144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_and_another_seq_sum_l1238_123803


namespace NUMINAMATH_CALUDE_wine_cost_theorem_l1238_123853

/-- The cost of a bottle of wine with a cork -/
def wineWithCorkCost (corkCost : ℝ) (extraCost : ℝ) : ℝ :=
  corkCost + (corkCost + extraCost)

/-- Theorem: The cost of a bottle of wine with a cork is $6.10 -/
theorem wine_cost_theorem (corkCost : ℝ) (extraCost : ℝ)
  (h1 : corkCost = 2.05)
  (h2 : extraCost = 2.00) :
  wineWithCorkCost corkCost extraCost = 6.10 := by
  sorry

#eval wineWithCorkCost 2.05 2.00

end NUMINAMATH_CALUDE_wine_cost_theorem_l1238_123853


namespace NUMINAMATH_CALUDE_combined_shape_is_pentahedron_l1238_123814

/-- A regular square pyramid -/
structure RegularSquarePyramid :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- The result of combining a regular square pyramid and a regular tetrahedron -/
def CombinedShape (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) :=
  pyramid.edge_length = tetrahedron.edge_length

/-- The number of faces in the resulting shape -/
def num_faces (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : ℕ := 5

theorem combined_shape_is_pentahedron 
  (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : 
  num_faces pyramid tetrahedron h = 5 := by sorry

end NUMINAMATH_CALUDE_combined_shape_is_pentahedron_l1238_123814


namespace NUMINAMATH_CALUDE_negative_three_star_negative_two_nested_star_op_l1238_123830

-- Define the custom operation
def star_op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Theorem statements
theorem negative_three_star_negative_two : star_op (-3) (-2) = 17 := by sorry

theorem nested_star_op : star_op (-2) (star_op (-3) (-2)) = -47 := by sorry

end NUMINAMATH_CALUDE_negative_three_star_negative_two_nested_star_op_l1238_123830


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l1238_123884

theorem smallest_n_with_common_factor : 
  ∀ n : ℕ, n > 0 → n < 38 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (11*n - 3) ∧ k ∣ (8*n + 4)) ∧
  ∃ k : ℕ, k > 1 ∧ k ∣ (11*38 - 3) ∧ k ∣ (8*38 + 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l1238_123884


namespace NUMINAMATH_CALUDE_telephone_answered_probability_l1238_123821

theorem telephone_answered_probability :
  let p1 : ℝ := 0.1  -- Probability of answering at first ring
  let p2 : ℝ := 0.3  -- Probability of answering at second ring
  let p3 : ℝ := 0.4  -- Probability of answering at third ring
  let p4 : ℝ := 0.1  -- Probability of answering at fourth ring
  1 - (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.9
  := by sorry

end NUMINAMATH_CALUDE_telephone_answered_probability_l1238_123821


namespace NUMINAMATH_CALUDE_alice_bob_distance_difference_l1238_123807

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed_a : ℝ) (speed_b : ℝ) (time : ℝ) : ℝ :=
  (speed_a - speed_b) * time

/-- Theorem: Alice bikes 30 miles more than Bob after 6 hours -/
theorem alice_bob_distance_difference :
  distance_difference 15 10 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_distance_difference_l1238_123807


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l1238_123879

/-- The probability of arranging n items in a row, where k specific items are placed consecutively. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n

/-- The probability of arranging 12 items in a row, where 4 specific items are placed consecutively, is 1/55. -/
theorem art_arrangement_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_probability_l1238_123879


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1238_123863

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1238_123863


namespace NUMINAMATH_CALUDE_only_14_and_28_satisfy_l1238_123864

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = 10 * x + y ∧ y ≤ 9 ∧ n = 14 * x

theorem only_14_and_28_satisfy :
  ∀ n : ℕ, satisfies_condition n ↔ n = 14 ∨ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_only_14_and_28_satisfy_l1238_123864


namespace NUMINAMATH_CALUDE_choose_two_from_five_l1238_123894

theorem choose_two_from_five (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_five_l1238_123894


namespace NUMINAMATH_CALUDE_special_rectangle_side_gt_12_l1238_123895

/-- A rectangle with sides a and b, where the area is three times the perimeter --/
structure SpecialRectangle where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a ≠ b
  h4 : a * b = 3 * (2 * (a + b))

/-- Theorem: For a SpecialRectangle, one of its sides is greater than 12 --/
theorem special_rectangle_side_gt_12 (rect : SpecialRectangle) : max rect.a rect.b > 12 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_side_gt_12_l1238_123895


namespace NUMINAMATH_CALUDE_division_reduction_l1238_123826

theorem division_reduction (x : ℕ) (h : x > 0) : 36 / x = 36 - 24 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l1238_123826


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l1238_123872

def A (a b : ℝ) : Set ℝ := {2, a, b}
def B (b : ℝ) : Set ℝ := {0, 2, b^2 - 2}

theorem set_equality_implies_values (a b : ℝ) :
  A a b = B b → ((a = 0 ∧ b = -1) ∨ (a = -2 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l1238_123872


namespace NUMINAMATH_CALUDE_remainder_is_zero_l1238_123860

def divisors : List ℕ := [12, 15, 20, 54]
def least_number : ℕ := 540

theorem remainder_is_zero (n : ℕ) (h : n ∈ divisors) : 
  least_number % n = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_is_zero_l1238_123860


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1238_123890

/-- Two lines intersect in the fourth quadrant if and only if m > -2/3 -/
theorem intersection_in_fourth_quadrant (m : ℝ) :
  (∃ x y : ℝ, 3 * x + 2 * y - 2 * m - 1 = 0 ∧
               2 * x + 4 * y - m = 0 ∧
               x > 0 ∧ y < 0) ↔
  m > -2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1238_123890


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l1238_123836

/-- Represents the ratios and amounts in a flavored drink formulation -/
structure DrinkFormulation where
  standard_ratio_flavoring : ℚ
  standard_ratio_corn_syrup : ℚ
  standard_ratio_water : ℚ
  sport_ratio_flavoring_corn_syrup_multiplier : ℚ
  sport_ratio_flavoring_water_multiplier : ℚ
  sport_corn_syrup_amount : ℚ

/-- Calculates the amount of water in the sport formulation -/
def water_amount (d : DrinkFormulation) : ℚ :=
  let sport_ratio_flavoring := d.standard_ratio_flavoring
  let sport_ratio_corn_syrup := d.standard_ratio_corn_syrup / d.sport_ratio_flavoring_corn_syrup_multiplier
  let sport_ratio_water := d.standard_ratio_water / d.sport_ratio_flavoring_water_multiplier
  let flavoring_amount := d.sport_corn_syrup_amount * (sport_ratio_flavoring / sport_ratio_corn_syrup)
  flavoring_amount * (sport_ratio_water / sport_ratio_flavoring)

theorem sport_formulation_water_amount 
  (d : DrinkFormulation)
  (h1 : d.standard_ratio_flavoring = 1)
  (h2 : d.standard_ratio_corn_syrup = 12)
  (h3 : d.standard_ratio_water = 30)
  (h4 : d.sport_ratio_flavoring_corn_syrup_multiplier = 3)
  (h5 : d.sport_ratio_flavoring_water_multiplier = 2)
  (h6 : d.sport_corn_syrup_amount = 5) :
  water_amount d = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l1238_123836


namespace NUMINAMATH_CALUDE_tan_2x_value_l1238_123859

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_2x_value (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + Real.cos x)
  (hf' : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 : ℝ) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_2x_value_l1238_123859


namespace NUMINAMATH_CALUDE_intersection_point_property_l1238_123809

/-- The x-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def a : ℝ := by sorry

/-- The y-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def b : ℝ := by sorry

/-- The intersection point satisfies the equation of y = 1/x -/
axiom inverse_prop : b = 1 / a

/-- The intersection point satisfies the equation of y = x + 2 -/
axiom linear : b = a + 2

theorem intersection_point_property : a - a * b - b = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_property_l1238_123809


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l1238_123810

theorem consecutive_numbers_problem (x y z w : ℚ) : 
  x > y ∧ y > z ∧  -- x, y, z are consecutive and in descending order
  w > x ∧  -- w is greater than x
  w = (5/3) * x ∧  -- ratio of x to w is 3:5
  w^2 = x * z ∧  -- w^2 = xz
  2*x + 3*y + 3*z = 5*y + 11 ∧  -- given equation
  x - y = y - z  -- x, y, z are equally spaced
  → z = 3 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l1238_123810


namespace NUMINAMATH_CALUDE_probability_theorem_l1238_123862

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_sequence : ℚ :=
  (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) /
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem :
  probability_specific_sequence * number_of_arrangements = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1238_123862


namespace NUMINAMATH_CALUDE_evaluate_expression_1_evaluate_expression_2_l1238_123861

-- Question 1
theorem evaluate_expression_1 :
  (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (5 + 1/3) + 3 * Real.sqrt (8 + 1/3)) = 115 := by
  sorry

-- Question 2
theorem evaluate_expression_2 :
  (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (2 + 2/3) - 3 * Real.sqrt (1 + 2/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_1_evaluate_expression_2_l1238_123861


namespace NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1238_123823

theorem temperature_difference_product (P : ℝ) : 
  (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4) →
  (P = 17 ∨ P = 9) :=
by sorry

theorem product_of_possible_P_values : 
  (∃ P : ℝ, (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4)) →
  17 * 9 = 153 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1238_123823


namespace NUMINAMATH_CALUDE_cube_equals_self_mod_thousand_l1238_123869

theorem cube_equals_self_mod_thousand (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 →
  (n^3 % 1000 = n) ↔ (n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_cube_equals_self_mod_thousand_l1238_123869


namespace NUMINAMATH_CALUDE_sin_five_zeros_l1238_123889

theorem sin_five_zeros (f : ℝ → ℝ) (ω : ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x)) →
  (∃! z : Finset ℝ, z.card = 5 ∧ (∀ x ∈ z, x ∈ Set.Icc 0 (3 * Real.pi) ∧ f x = 0)) →
  ω ∈ Set.Icc (4 / 3) (5 / 3) :=
sorry

end NUMINAMATH_CALUDE_sin_five_zeros_l1238_123889


namespace NUMINAMATH_CALUDE_simplify_expression_l1238_123811

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1238_123811


namespace NUMINAMATH_CALUDE_helium_pressure_change_l1238_123818

-- Define the variables and constants
variable (v₁ v₂ p₁ p₂ : ℝ)

-- State the given conditions
def initial_volume : ℝ := 3.6
def initial_pressure : ℝ := 8
def final_volume : ℝ := 4.5

-- Define the inverse proportionality relationship
def inverse_proportional (v₁ v₂ p₁ p₂ : ℝ) : Prop :=
  v₁ * p₁ = v₂ * p₂

-- State the theorem
theorem helium_pressure_change :
  v₁ = initial_volume →
  p₁ = initial_pressure →
  v₂ = final_volume →
  inverse_proportional v₁ v₂ p₁ p₂ →
  p₂ = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_helium_pressure_change_l1238_123818


namespace NUMINAMATH_CALUDE_nine_team_league_games_l1238_123827

/-- The total number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a 9-team league where each team plays 3 games with every other team, 
    the total number of games played is 108 -/
theorem nine_team_league_games :
  total_games 9 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_nine_team_league_games_l1238_123827


namespace NUMINAMATH_CALUDE_triangle_to_hexagon_proportionality_l1238_123828

-- Define the original triangle
structure Triangle where
  x : Real
  y : Real
  z : Real
  angle_sum : x + y + z = 180

-- Define the resulting hexagon
structure Hexagon where
  a : Real -- Length of vector a
  b : Real -- Length of vector b
  c : Real -- Length of vector c
  u : Real -- Length of vector u
  v : Real -- Length of vector v
  w : Real -- Length of vector w
  angle1 : Real -- (x-1)°
  angle2 : Real -- 181°
  angle3 : Real
  angle4 : Real
  angle5 : Real
  angle6 : Real
  angle_sum : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 720
  non_convex : angle2 > 180

-- Define the transformation from triangle to hexagon
def transform (t : Triangle) (h : Hexagon) : Prop :=
  h.angle1 = t.x - 1 ∧ h.angle2 = 181

-- Theorem to prove
theorem triangle_to_hexagon_proportionality (t : Triangle) (h : Hexagon) 
  (trans : transform t h) : 
  ∃ (k : Real), k > 0 ∧ 
    h.a / t.x = h.b / t.y ∧ 
    h.b / t.y = h.c / t.z ∧ 
    h.c / t.z = k :=
  sorry

end NUMINAMATH_CALUDE_triangle_to_hexagon_proportionality_l1238_123828


namespace NUMINAMATH_CALUDE_product_sum_bounds_l1238_123885

def X : Finset ℕ := Finset.range 11

theorem product_sum_bounds (A B : Finset ℕ) (hA : A ⊆ X) (hB : B ⊆ X) 
  (hAB : A ∪ B = X) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) :
  12636 ≤ (A.prod id) + (B.prod id) ∧ (A.prod id) + (B.prod id) ≤ 2 * Nat.factorial 11 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l1238_123885


namespace NUMINAMATH_CALUDE_probability_solution_l1238_123804

def probability_equation (p q : ℝ) : Prop :=
  q = 1 - p ∧ 
  (Nat.choose 10 7 : ℝ) * p^7 * q^3 = (Nat.choose 10 6 : ℝ) * p^6 * q^4

theorem probability_solution :
  ∀ p q : ℝ, probability_equation p q → p = 7/11 := by
  sorry

end NUMINAMATH_CALUDE_probability_solution_l1238_123804
