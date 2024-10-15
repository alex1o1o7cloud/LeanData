import Mathlib

namespace NUMINAMATH_CALUDE_constant_value_l3493_349379

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (constant : ℝ) :
  (∀ t, x t = constant - 3 * t) →
  (∀ t, y t = 2 * t - 3) →
  x 0.8 = y 0.8 →
  constant = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_value_l3493_349379


namespace NUMINAMATH_CALUDE_math_book_cost_l3493_349306

/-- Proves that the cost of a math book is $4 given the conditions of the book purchase problem -/
theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_books = 54)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 396) :
  ∃ (math_book_cost : ℕ), 
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧ 
    math_book_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_book_cost_l3493_349306


namespace NUMINAMATH_CALUDE_temperature_calculation_l3493_349302

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (M T W Th F : ℝ) : 
  M = 42 →
  (M + T + W + Th) / 4 = 48 →
  (T + W + Th + F) / 4 = 46 →
  F = 34 := by
  sorry

#check temperature_calculation

end NUMINAMATH_CALUDE_temperature_calculation_l3493_349302


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l3493_349319

/-- Represents the daily sales and profit scenario for a product -/
structure SalesScenario where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the daily profit for a given sales scenario -/
def dailyProfit (s : SalesScenario) : ℕ :=
  (s.initialSales + s.salesIncrease * s.priceReduction) * (s.initialProfit - s.priceReduction)

/-- Theorem: A price reduction of 25 yuan results in a daily profit of 2000 yuan -/
theorem optimal_price_reduction (s : SalesScenario) 
  (h1 : s.initialSales = 30)
  (h2 : s.initialProfit = 50)
  (h3 : s.salesIncrease = 2)
  (h4 : s.priceReduction = 25) :
  dailyProfit s = 2000 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 25 }

end NUMINAMATH_CALUDE_optimal_price_reduction_l3493_349319


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l3493_349327

theorem triangle_similarity_problem (DC CB AD AB ED : ℝ) (h1 : DC = 12) (h2 : CB = 9) 
  (h3 : AB = (1/3) * AD) (h4 : ED = (3/4) * AD) : 
  ∃ FC : ℝ, FC = 14.625 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l3493_349327


namespace NUMINAMATH_CALUDE_fraction_squared_l3493_349376

theorem fraction_squared (x : ℚ) : x^2 = 0.0625 → x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_l3493_349376


namespace NUMINAMATH_CALUDE_set_problems_l3493_349350

def U : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {1, 4}

def A (x : ℕ) : Set ℕ := {1, 2, x^2}

theorem set_problems (x : ℕ) (hx : x ∈ U) :
  (U \ B = {2, 3}) ∧ 
  (A x ∩ B = B → x = 1) ∧
  ¬∃ (y : ℕ), y ∈ U ∧ A y ∪ B = U :=
by sorry

end NUMINAMATH_CALUDE_set_problems_l3493_349350


namespace NUMINAMATH_CALUDE_is_perfect_square_l3493_349328

/-- Given a = 2992² + 2992² × 2993² + 2993², prove that a is a perfect square -/
theorem is_perfect_square (a : ℕ) (h : a = 2992^2 + 2992^2 * 2993^2 + 2993^2) :
  ∃ n : ℕ, a = n^2 := by sorry

end NUMINAMATH_CALUDE_is_perfect_square_l3493_349328


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3493_349353

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ((2 / a + 8 / b) ≥ 9) ∧
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 2 / a' + 8 / b' = 9 ∧ a' = 2 / 3 ∧ b' = 4 / 3) ∧
  (a^2 + b^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3493_349353


namespace NUMINAMATH_CALUDE_hall_volume_l3493_349393

/-- The volume of a rectangular hall with given dimensions and area constraint -/
theorem hall_volume (length width : ℝ) (h : length = 15 ∧ width = 12) 
  (area_constraint : 2 * (length * width) = 2 * (length + width) * ((2 * length * width) / (2 * (length + width)))) :
  length * width * ((2 * length * width) / (2 * (length + width))) = 8004 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_l3493_349393


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l3493_349334

def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

theorem quadratic_function_satisfies_conditions :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l3493_349334


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3493_349399

/-- The functional equation that f must satisfy for all real a, b, c -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a*b^2 + b*c^2 + c*a^2) - f (a^2*b + b^2*c + c^2*a)

/-- The set of possible functions that satisfy the equation -/
def PossibleFunctions (f : ℝ → ℝ) : Prop :=
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x + β)) ∨
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x^3 + β))

/-- The main theorem stating that any function satisfying the equation must be of the specified form -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → PossibleFunctions f := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3493_349399


namespace NUMINAMATH_CALUDE_quarters_found_l3493_349367

/-- The number of quarters Alyssa found in her couch -/
def num_quarters : ℕ := sorry

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of pennies Alyssa found -/
def num_pennies : ℕ := 7

/-- The total amount Alyssa found in cents -/
def total_amount : ℕ := 307

theorem quarters_found :
  num_quarters * quarter_value + num_pennies * penny_value = total_amount ∧
  num_quarters = 12 := by sorry

end NUMINAMATH_CALUDE_quarters_found_l3493_349367


namespace NUMINAMATH_CALUDE_plains_total_area_l3493_349301

/-- The total area of two plains, given their individual areas. -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- The theorem stating the total area of two plains. -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_plains_total_area_l3493_349301


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3493_349333

theorem necessary_and_sufficient_condition (p q : Prop) 
  (h1 : p → q) (h2 : q → p) : 
  (p ↔ q) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3493_349333


namespace NUMINAMATH_CALUDE_table_price_is_84_l3493_349330

/-- Represents the price of items in a store --/
structure StorePrice where
  chair : ℝ
  table : ℝ
  lamp : ℝ

/-- Conditions for the store pricing problem --/
def StorePricingConditions (p : StorePrice) : Prop :=
  (2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table)) ∧
  (p.chair + p.table = 96) ∧
  (p.lamp + p.chair = 0.5 * (2 * p.table + p.lamp))

/-- Theorem stating that under the given conditions, the price of a table is $84 --/
theorem table_price_is_84 (p : StorePrice) 
  (h : StorePricingConditions p) : p.table = 84 := by
  sorry

end NUMINAMATH_CALUDE_table_price_is_84_l3493_349330


namespace NUMINAMATH_CALUDE_complete_square_sum_l3493_349324

theorem complete_square_sum (x : ℝ) : ∃ (d e f : ℤ), 
  d > 0 ∧ 
  (25 * x^2 + 30 * x - 72 = 0 ↔ (d * x + e)^2 = f) ∧ 
  d + e + f = 89 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3493_349324


namespace NUMINAMATH_CALUDE_oldest_babysat_current_age_l3493_349388

/- Define the parameters of the problem -/
def jane_start_age : ℕ := 18
def jane_current_age : ℕ := 34
def years_since_stopped : ℕ := 10

/- Define the function to calculate the maximum age of a child Jane could baby-sit at a given age -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/- Theorem statement -/
theorem oldest_babysat_current_age :
  let jane_stop_age : ℕ := jane_current_age - years_since_stopped
  let max_child_age_when_stopped : ℕ := max_child_age jane_stop_age
  let oldest_babysat_age : ℕ := max_child_age_when_stopped + years_since_stopped
  oldest_babysat_age = 22 := by sorry

end NUMINAMATH_CALUDE_oldest_babysat_current_age_l3493_349388


namespace NUMINAMATH_CALUDE_fifth_number_12th_row_l3493_349323

/-- Given a lattice with the following properties:
  - Has 12 rows
  - Each row contains 7 consecutive numbers
  - The first number in Row 1 is 1
  - The first number in each row increases by 8 as the row number increases
  This function calculates the nth number in the mth row -/
def latticeNumber (m n : ℕ) : ℕ :=
  (1 + 8 * (m - 1)) + (n - 1)

/-- The theorem states that the fifth number in the 12th row of the described lattice is 93 -/
theorem fifth_number_12th_row : latticeNumber 12 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_12th_row_l3493_349323


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3493_349392

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3, 4}

-- Define set B
def B : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3493_349392


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3493_349317

def C (s : ℕ) : ℕ := s * (s + 1)

def product_C (m k n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (C (m + i + 1) - C k)) 1

def product_C_seq (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * C (i + 1)) 1

theorem divisibility_theorem (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : Nat.Prime (m + k + 1)) (h5 : m + k + 1 > n + 1) :
  ∃ z : ℤ, (product_C m k n : ℤ) = z * (product_C_seq n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3493_349317


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3493_349320

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3493_349320


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3493_349338

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) :
  (∀ n : ℕ, a * r^n = 3 * (a * r^(n+1) + a * r^(n+2))) →
  (∀ n : ℕ, a * r^n > 0) →
  r = (-1 + Real.sqrt (7/3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3493_349338


namespace NUMINAMATH_CALUDE_inequality_proof_l3493_349362

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3493_349362


namespace NUMINAMATH_CALUDE_inequality_squared_l3493_349321

theorem inequality_squared (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_squared_l3493_349321


namespace NUMINAMATH_CALUDE_min_value_of_m_l3493_349310

theorem min_value_of_m (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - x^2 - x + (-(a^3) + a^2 + a) = (x - a) * (x - b) * (x - c)) →
  ∀ m : ℝ, (∀ x : ℝ, x^3 - x^2 - x + m = (x - a) * (x - b) * (x - c)) → 
  m ≥ -5/27 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_m_l3493_349310


namespace NUMINAMATH_CALUDE_symmetric_point_reciprocal_function_l3493_349385

theorem symmetric_point_reciprocal_function (k : ℝ) : 
  let B : ℝ × ℝ := (Real.cos (π / 3), -Real.sqrt 3)
  let A : ℝ × ℝ := (B.1, -B.2)
  (A.2 = k / A.1) → k = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_reciprocal_function_l3493_349385


namespace NUMINAMATH_CALUDE_bookmarked_pages_march_end_l3493_349358

/-- Represents the number of bookmarked pages at the end of a month -/
def bookmarked_pages_at_month_end (
  pages_per_day : ℕ
) (initial_pages : ℕ) (days_in_month : ℕ) : ℕ :=
  initial_pages + pages_per_day * days_in_month

/-- Theorem: Given the conditions, prove that the total bookmarked pages at the end of March is 1330 -/
theorem bookmarked_pages_march_end :
  bookmarked_pages_at_month_end 30 400 31 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bookmarked_pages_march_end_l3493_349358


namespace NUMINAMATH_CALUDE_problem_statement_l3493_349335

theorem problem_statement (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : abs x = 2) : 
  -2 * m * n + (a + b) / 2023 + x^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3493_349335


namespace NUMINAMATH_CALUDE_percentage_less_than_150000_l3493_349395

/-- Represents the percentage of counties in a specific population range -/
structure PopulationRange where
  percentage : ℝ
  lower_bound : ℕ
  upper_bound : Option ℕ

/-- Proves that the percentage of counties with fewer than 150,000 residents is 83% -/
theorem percentage_less_than_150000 
  (less_than_10000 : PopulationRange)
  (between_10000_and_49999 : PopulationRange)
  (between_50000_and_149999 : PopulationRange)
  (more_than_150000 : PopulationRange)
  (h1 : less_than_10000.percentage = 21)
  (h2 : between_10000_and_49999.percentage = 44)
  (h3 : between_50000_and_149999.percentage = 18)
  (h4 : more_than_150000.percentage = 17)
  (h5 : less_than_10000.upper_bound = some 9999)
  (h6 : between_10000_and_49999.lower_bound = 10000 ∧ between_10000_and_49999.upper_bound = some 49999)
  (h7 : between_50000_and_149999.lower_bound = 50000 ∧ between_50000_and_149999.upper_bound = some 149999)
  (h8 : more_than_150000.lower_bound = 150000 ∧ more_than_150000.upper_bound = none)
  (h9 : less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage + more_than_150000.percentage = 100) :
  less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage = 83 := by
  sorry


end NUMINAMATH_CALUDE_percentage_less_than_150000_l3493_349395


namespace NUMINAMATH_CALUDE_davids_math_marks_l3493_349313

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 91)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 78)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l3493_349313


namespace NUMINAMATH_CALUDE_sphere_volume_derivative_l3493_349377

noncomputable section

-- Define the volume function for a sphere
def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

-- Define the surface area function for a sphere
def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- State the theorem
theorem sphere_volume_derivative (R : ℝ) (h : R > 0) :
  deriv sphere_volume R = sphere_surface_area R := by
  sorry

end

end NUMINAMATH_CALUDE_sphere_volume_derivative_l3493_349377


namespace NUMINAMATH_CALUDE_rice_and_flour_weights_l3493_349394

/-- The weight of a bag of rice in kilograms -/
def rice_weight : ℝ := 50

/-- The weight of a bag of flour in kilograms -/
def flour_weight : ℝ := 25

/-- The total weight of 8 bags of rice and 6 bags of flour in kilograms -/
def weight1 : ℝ := 550

/-- The total weight of 4 bags of rice and 7 bags of flour in kilograms -/
def weight2 : ℝ := 375

theorem rice_and_flour_weights :
  (8 * rice_weight + 6 * flour_weight = weight1) ∧
  (4 * rice_weight + 7 * flour_weight = weight2) := by
  sorry

end NUMINAMATH_CALUDE_rice_and_flour_weights_l3493_349394


namespace NUMINAMATH_CALUDE_sum_of_sign_ratios_l3493_349365

theorem sum_of_sign_ratios (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / |a| + b / |b| + (a * b) / |a * b| = 3 ∨ a / |a| + b / |b| + (a * b) / |a * b| = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sign_ratios_l3493_349365


namespace NUMINAMATH_CALUDE_sum_floor_equals_n_l3493_349357

/-- For any natural number n, the sum of floor((n+2^k)/(2^(k+1))) from k=0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_n_l3493_349357


namespace NUMINAMATH_CALUDE_power_of_81_l3493_349369

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l3493_349369


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3493_349348

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-3, -1)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def lies_on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem point_P_satisfies_conditions :
  lies_on_line A B P ∧ vector A P = (1/2 : ℝ) • vector A B := by sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3493_349348


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3493_349307

theorem arithmetic_calculation : (2 + 3 * 4 - 5) * 2 + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3493_349307


namespace NUMINAMATH_CALUDE_city_inhabitants_problem_l3493_349371

theorem city_inhabitants_problem :
  ∃ (n : ℕ), 
    (∃ (m : ℕ), n^2 + 100 = m^2 + 1) ∧ 
    (∃ (k : ℕ), n^2 + 200 = k^2) ∧ 
    n = 49 := by
  sorry

end NUMINAMATH_CALUDE_city_inhabitants_problem_l3493_349371


namespace NUMINAMATH_CALUDE_assign_25_to_4_l3493_349343

/-- The number of ways to assign different service providers to children -/
def assignProviders (n m : ℕ) : ℕ :=
  (n - 0) * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: Assigning 25 service providers to 4 children results in 303600 possibilities -/
theorem assign_25_to_4 : assignProviders 25 4 = 303600 := by
  sorry

#eval assignProviders 25 4

end NUMINAMATH_CALUDE_assign_25_to_4_l3493_349343


namespace NUMINAMATH_CALUDE_equation_solutions_l3493_349366

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3493_349366


namespace NUMINAMATH_CALUDE_intersection_x_coordinates_equal_l3493_349318

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

/-- Definition of the right focus -/
def rightFocus (a b : ℝ) : Prop :=
  ellipse a b 1 (Real.sqrt 3 / 2)

/-- Definition of the perpendicular chord -/
def perpendicularChord (a b : ℝ) : Prop :=
  ∃ y₁ y₂, y₁ ≠ y₂ ∧ ellipse a b 0 y₁ ∧ ellipse a b 0 y₂ ∧ y₂ - y₁ = 1

/-- Definition of a point on the ellipse -/
def pointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse a b x y

/-- Theorem statement -/
theorem intersection_x_coordinates_equal
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hrf : rightFocus a b)
  (hpc : perpendicularChord a b)
  (x₁ y₁ x₂ y₂ : ℝ)
  (hm : pointOnEllipse a b x₁ y₁)
  (hn : pointOnEllipse a b x₂ y₂)
  (hl : ∃ m, y₁ = m * (x₁ - 1) ∧ y₂ = m * (x₂ - 1)) :
  ∃ x_int, (∃ y_am, y_am = (y₁ / (x₁ + 1)) * (x_int + 1)) ∧
           (∃ y_bn, y_bn = (y₂ / (x₂ - 1)) * (x_int - 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinates_equal_l3493_349318


namespace NUMINAMATH_CALUDE_area_common_part_squares_l3493_349370

/-- The area of the common part of two squares -/
theorem area_common_part_squares (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 1)
  (h2 : large_side = 4 * small_side)
  (h3 : small_side > 0)
  (h4 : large_side > small_side) : 
  large_side^2 - (1/2 * small_side^2 + 1/2 * large_side^2) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_area_common_part_squares_l3493_349370


namespace NUMINAMATH_CALUDE_hairstylist_monthly_earnings_l3493_349329

/-- Represents the hairstylist's pricing and schedule --/
structure HairstylistData where
  normal_price : ℕ
  special_price : ℕ
  trendy_price : ℕ
  deluxe_price : ℕ
  mwf_normal : ℕ
  mwf_special : ℕ
  mwf_trendy : ℕ
  tth_normal : ℕ
  tth_special : ℕ
  tth_deluxe : ℕ
  weekend_trendy : ℕ
  weekend_deluxe : ℕ
  weeks_per_month : ℕ

/-- Calculates the monthly earnings of the hairstylist --/
def monthlyEarnings (data : HairstylistData) : ℕ :=
  let mwf_daily := data.mwf_normal * data.normal_price + data.mwf_special * data.special_price + data.mwf_trendy * data.trendy_price
  let tth_daily := data.tth_normal * data.normal_price + data.tth_special * data.special_price + data.tth_deluxe * data.deluxe_price
  let weekend_daily := data.weekend_trendy * data.trendy_price + data.weekend_deluxe * data.deluxe_price
  let weekly_total := 3 * mwf_daily + 2 * tth_daily + 2 * weekend_daily
  weekly_total * data.weeks_per_month

/-- Theorem stating the monthly earnings of the hairstylist --/
theorem hairstylist_monthly_earnings :
  let data : HairstylistData := {
    normal_price := 10
    special_price := 15
    trendy_price := 22
    deluxe_price := 30
    mwf_normal := 4
    mwf_special := 3
    mwf_trendy := 1
    tth_normal := 6
    tth_special := 2
    tth_deluxe := 3
    weekend_trendy := 10
    weekend_deluxe := 5
    weeks_per_month := 4
  }
  monthlyEarnings data = 5684 := by
  sorry

end NUMINAMATH_CALUDE_hairstylist_monthly_earnings_l3493_349329


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_m_equals_one_l3493_349390

/-- A circle with equation x^2 + y^2 + 2x - 4y = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 = 0}

/-- A line with equation 3x + y + m = 0 -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 + p.2 + m = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Symmetry of a circle about a line -/
def isSymmetric (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_symmetry_implies_m_equals_one :
  isSymmetric Circle (Line m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_m_equals_one_l3493_349390


namespace NUMINAMATH_CALUDE_A_equals_B_l3493_349325

-- Define sets A and B
def A : Set ℕ := {3, 2}
def B : Set ℕ := {2, 3}

-- Theorem stating that A and B are equal
theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l3493_349325


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l3493_349355

theorem smallest_constant_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0) (hx₄ : x₄ > 0) (hx₅ : x₅ > 0) :
  ∃ C : ℝ, C = 5^15 ∧ 
  (∀ D : ℝ, D < C → ∃ y₁ y₂ y₃ y₄ y₅ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧ y₅ > 0 ∧
    D * (y₁^2005 + y₂^2005 + y₃^2005 + y₄^2005 + y₅^2005) < 
    y₁*y₂*y₃*y₄*y₅ * (y₁^125 + y₂^125 + y₃^125 + y₄^125 + y₅^125)^16) ∧
  C * (x₁^2005 + x₂^2005 + x₃^2005 + x₄^2005 + x₅^2005) ≥ 
  x₁*x₂*x₃*x₄*x₅ * (x₁^125 + x₂^125 + x₃^125 + x₄^125 + x₅^125)^16 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l3493_349355


namespace NUMINAMATH_CALUDE_projection_sum_bound_l3493_349368

/-- A segment in the plane represented by its length and angle -/
structure Segment where
  length : ℝ
  angle : ℝ

/-- The theorem statement -/
theorem projection_sum_bound (segments : List Segment) 
  (total_length : (segments.map (λ s => s.length)).sum = 1) :
  ∃ θ : ℝ, (segments.map (λ s => s.length * |Real.cos (θ - s.angle)|)).sum < 2 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_projection_sum_bound_l3493_349368


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3493_349341

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1,2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x+y-3=0 or 2x-y=0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y = l.k * (x - 1) + 2) ∨
  (∀ x y, 2 * x - y = 0 ↔ y = l.k * (x - 1) + 2) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3493_349341


namespace NUMINAMATH_CALUDE_box_volume_increase_l3493_349347

theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 4860)
  (hs : 2 * (l * w + w * h + l * h) = 1860)
  (he : 4 * (l + w + h) = 224) :
  (l + 2) * (w + 3) * (h + 1) = 5964 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3493_349347


namespace NUMINAMATH_CALUDE_dormitory_places_l3493_349300

theorem dormitory_places : ∃ (x y : ℕ),
  (2 * x + 3 * y > 30) ∧
  (2 * x + 3 * y < 70) ∧
  (4 * (2 * x + 3 * y) = 5 * (3 * x + 2 * y)) ∧
  (2 * x + 3 * y = 50) :=
by sorry

end NUMINAMATH_CALUDE_dormitory_places_l3493_349300


namespace NUMINAMATH_CALUDE_fraction_equality_implies_five_l3493_349354

theorem fraction_equality_implies_five (a b : ℕ) (h : (a + 30) / b = a / (b - 6)) : 
  (a + 30) / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_five_l3493_349354


namespace NUMINAMATH_CALUDE_jens_birds_multiple_l3493_349396

theorem jens_birds_multiple (ducks chickens total_birds M : ℕ) : 
  ducks = 150 →
  total_birds = 185 →
  ducks = M * chickens + 10 →
  total_birds = ducks + chickens →
  M = 4 := by
sorry

end NUMINAMATH_CALUDE_jens_birds_multiple_l3493_349396


namespace NUMINAMATH_CALUDE_rain_in_tel_aviv_l3493_349314

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv. -/
def rain_probability : ℝ := 0.5

/-- The number of randomly chosen days. -/
def total_days : ℕ := 6

/-- The number of rainy days we're interested in. -/
def rainy_days : ℕ := 4

theorem rain_in_tel_aviv :
  binomial_probability total_days rainy_days rain_probability = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_in_tel_aviv_l3493_349314


namespace NUMINAMATH_CALUDE_congruence_problem_l3493_349351

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3493_349351


namespace NUMINAMATH_CALUDE_more_males_than_females_difference_in_population_l3493_349389

theorem more_males_than_females : Int → Int → Int
  | num_males, num_females =>
    num_males - num_females

theorem difference_in_population (num_males num_females : Int) 
  (h1 : num_males = 23) 
  (h2 : num_females = 9) : 
  more_males_than_females num_males num_females = 14 := by
  sorry

end NUMINAMATH_CALUDE_more_males_than_females_difference_in_population_l3493_349389


namespace NUMINAMATH_CALUDE_total_players_count_l3493_349303

/-- The number of players who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 20 := by sorry

end NUMINAMATH_CALUDE_total_players_count_l3493_349303


namespace NUMINAMATH_CALUDE_inequality_proof_l3493_349363

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  a / (a + 2*b)^(1/3) + b / (b + 2*c)^(1/3) + c / (c + 2*a)^(1/3) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3493_349363


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l3493_349364

/-- A quadrilateral with perpendicular diagonals and equal adjacent sides, but not all sides equal -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- Each pair of adjacent sides is equal -/
  adjacent_sides_equal : Bool
  /-- Not all sides are equal -/
  not_all_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perpendicular_diagonals ∧ q.adjacent_sides_equal ∧ q.not_all_sides_equal

/-- Theorem stating that the given quadrilateral is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perpendicular_diagonals = true) 
  (h2 : q.adjacent_sides_equal = true) 
  (h3 : q.not_all_sides_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l3493_349364


namespace NUMINAMATH_CALUDE_max_handshakes_l3493_349380

def number_of_people : ℕ := 30

theorem max_handshakes (n : ℕ) (h : n = number_of_people) : 
  Nat.choose n 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l3493_349380


namespace NUMINAMATH_CALUDE_negation_equivalence_l3493_349308

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3493_349308


namespace NUMINAMATH_CALUDE_percentage_of_14_to_70_l3493_349398

theorem percentage_of_14_to_70 : ∀ (x : ℚ), x = 14 / 70 * 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_14_to_70_l3493_349398


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l3493_349387

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2010 + Real.sqrt 2011 →
  Q = -(Real.sqrt 2010) - Real.sqrt 2011 →
  R = Real.sqrt 2010 - Real.sqrt 2011 →
  S = Real.sqrt 2011 - Real.sqrt 2010 →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l3493_349387


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3493_349326

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def quadratic_roots_prime (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 105 ∧ p * q = k

theorem unique_k_for_prime_roots : ∃! k : ℕ, quadratic_roots_prime k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3493_349326


namespace NUMINAMATH_CALUDE_new_distance_segment_l3493_349360

/-- New distance function between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Predicate to check if a point is on a line segment -/
def on_segment (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)

/-- Theorem: If C is on segment AB, then |AC| + |BC| = |AB| -/
theorem new_distance_segment (x₁ y₁ x₂ y₂ x y : ℝ) :
  on_segment x₁ y₁ x₂ y₂ x y →
  new_distance x₁ y₁ x y + new_distance x y x₂ y₂ = new_distance x₁ y₁ x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_new_distance_segment_l3493_349360


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3493_349309

/-- A quadratic equation kx^2 + 3x - 1 = 0 has real roots if and only if k ≥ -9/4 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3493_349309


namespace NUMINAMATH_CALUDE_new_student_weight_l3493_349378

/-- Given a group of students and their weights, calculate the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : new_avg = 14.8) :
  (initial_count + 1) * new_avg - initial_count * initial_avg = 11 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l3493_349378


namespace NUMINAMATH_CALUDE_optimal_optimism_coefficient_l3493_349342

theorem optimal_optimism_coefficient 
  (a b c x : ℝ) 
  (h1 : b > a) 
  (h2 : 0 < x ∧ x < 1) 
  (h3 : c = a + x * (b - a)) 
  (h4 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_optimal_optimism_coefficient_l3493_349342


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3493_349312

theorem simplify_and_evaluate (b : ℝ) : 
  (15 * b^5) / (75 * b^3) = b^2 / 5 ∧ 
  (15 * 4^5) / (75 * 4^3) = 16 / 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3493_349312


namespace NUMINAMATH_CALUDE_triangle_formation_l3493_349316

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 6 10 ∧
  ¬can_form_triangle 2 3 5 ∧
  ¬can_form_triangle 5 6 11 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3493_349316


namespace NUMINAMATH_CALUDE_election_winner_votes_l3493_349382

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (runner_up_percentage : ℚ) 
  (other_candidates_percentage : ℚ) 
  (invalid_votes_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 48/100 →
  runner_up_percentage = 34/100 →
  other_candidates_percentage = 15/100 →
  invalid_votes_percentage = 3/100 →
  (winner_percentage - runner_up_percentage) * total_votes = vote_difference →
  vote_difference = 2112 →
  winner_percentage * total_votes = 7241 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3493_349382


namespace NUMINAMATH_CALUDE_some_zims_not_cims_l3493_349386

variable (U : Type) -- Universe set

-- Define the sets
variable (Zim Bim Cim : Set U)

-- Define the conditions
variable (h1 : Zim ⊆ Bim)  -- All Zims are Bims
variable (h2 : ∃ x, x ∈ Bim ∧ x ∉ Cim)  -- Some (at least one) Bims are not Cims

-- Theorem to prove
theorem some_zims_not_cims : ∃ x, x ∈ Zim ∧ x ∉ Cim :=
sorry

end NUMINAMATH_CALUDE_some_zims_not_cims_l3493_349386


namespace NUMINAMATH_CALUDE_single_point_implies_d_eq_seven_l3493_349344

/-- The equation of the graph -/
def equation (x y d : ℝ) : ℝ := 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + d

/-- The graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, equation p.1 p.2 d = 0

/-- Theorem: If the equation represents a graph that consists of a single point, then d = 7 -/
theorem single_point_implies_d_eq_seven :
  ∃ d : ℝ, is_single_point d → d = 7 :=
sorry

end NUMINAMATH_CALUDE_single_point_implies_d_eq_seven_l3493_349344


namespace NUMINAMATH_CALUDE_range_of_a_l3493_349322

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0

theorem range_of_a (a : ℝ) : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → a ∈ Set.Ioo (-2 : ℝ) 1 ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3493_349322


namespace NUMINAMATH_CALUDE_wednesday_sales_l3493_349356

/-- Represents the number of crates of eggs sold on each day of the week -/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Calculates the total number of crates sold over 4 days -/
def total_sales (sales : EggSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday

/-- Theorem stating the number of crates sold on Wednesday -/
theorem wednesday_sales (sales : EggSales) 
  (h1 : sales.monday = 5)
  (h2 : sales.tuesday = 2 * sales.monday)
  (h3 : sales.thursday = sales.tuesday / 2)
  (h4 : total_sales sales = 28) :
  sales.wednesday = 8 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l3493_349356


namespace NUMINAMATH_CALUDE_min_cost_closed_chain_l3493_349359

/-- Represents the cost in cents to separate one link -/
def separation_cost : ℕ := 1

/-- Represents the cost in cents to attach one link -/
def attachment_cost : ℕ := 2

/-- Represents the number of pieces in the gold chain -/
def num_pieces : ℕ := 13

/-- Represents the number of links in each piece of the chain -/
def links_per_piece : ℕ := 80

/-- Calculates the total cost to separate and reattach one link -/
def link_operation_cost : ℕ := separation_cost + attachment_cost

/-- Theorem stating the minimum cost to form a closed chain -/
theorem min_cost_closed_chain : 
  ∃ (cost : ℕ), cost = (num_pieces - 1) * link_operation_cost ∧ 
  ∀ (other_cost : ℕ), other_cost ≥ cost := by sorry

end NUMINAMATH_CALUDE_min_cost_closed_chain_l3493_349359


namespace NUMINAMATH_CALUDE_cube_tetrahedrons_l3493_349397

/-- A cube is a three-dimensional shape with 8 vertices. -/
structure Cube where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 8

/-- A tetrahedron is a three-dimensional shape with 4 vertices. -/
structure Tetrahedron where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 4

/-- The number of ways to choose 4 vertices from 8 vertices. -/
def total_choices : ℕ := Nat.choose 8 4

/-- The number of sets of 4 vertices that cannot form a tetrahedron. -/
def invalid_choices : ℕ := 12

/-- The function that calculates the number of valid tetrahedrons. -/
def valid_tetrahedrons (c : Cube) : ℕ := total_choices - invalid_choices

/-- Theorem: The number of distinct tetrahedrons that can be formed using the vertices of a cube is 58. -/
theorem cube_tetrahedrons (c : Cube) : valid_tetrahedrons c = 58 := by
  sorry

end NUMINAMATH_CALUDE_cube_tetrahedrons_l3493_349397


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3493_349391

/-- Given a parabola with equation y = (1/m)x^2 where m < 0, 
    its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3493_349391


namespace NUMINAMATH_CALUDE_average_score_two_classes_l3493_349375

theorem average_score_two_classes (students1 students2 : ℕ) (avg1 avg2 : ℚ) :
  students1 = 20 →
  students2 = 30 →
  avg1 = 80 →
  avg2 = 70 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = 74 :=
by sorry

end NUMINAMATH_CALUDE_average_score_two_classes_l3493_349375


namespace NUMINAMATH_CALUDE_circle_area_irrational_l3493_349336

/-- If the diameter of a circle is rational, then its area is irrational. -/
theorem circle_area_irrational (d : ℚ) : Irrational (π * (d^2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_irrational_l3493_349336


namespace NUMINAMATH_CALUDE_gift_distribution_theorem_l3493_349345

/-- The number of ways to choose and distribute gifts -/
def giftDistributionWays (totalGifts classmates chosenGifts : ℕ) : ℕ :=
  (totalGifts.choose chosenGifts) * chosenGifts.factorial

/-- Theorem stating that choosing 3 out of 5 gifts and distributing to 3 classmates results in 60 ways -/
theorem gift_distribution_theorem :
  giftDistributionWays 5 3 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_theorem_l3493_349345


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3493_349332

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Hyperbola structure -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / 3 - y^2 = 1

/-- The theorem statement -/
theorem parabola_focus_distance (parab : Parabola) (hyper : Hyperbola) :
  (parab.equation 2 0 → hyper.equation 2 0) →  -- The foci coincide at (2, 0)
  (∀ b : ℝ, parab.equation 2 b →               -- For any point (2, b) on the parabola
    (2 - parab.p / 2)^2 + b^2 = 4^2) :=        -- The distance to the focus is 4
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3493_349332


namespace NUMINAMATH_CALUDE_symmetry_line_l3493_349337

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry condition
def is_symmetric (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (4 - x)

-- Define the line of symmetry
def line_of_symmetry (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, g (k + (x - k)) = g (k - (x - k))

-- Theorem statement
theorem symmetry_line (g : ℝ → ℝ) (h : is_symmetric g) :
  line_of_symmetry g 2 :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_l3493_349337


namespace NUMINAMATH_CALUDE_third_side_length_l3493_349346

/-- Two similar triangles with given side lengths -/
structure SimilarTriangles where
  -- Larger triangle
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ
  -- Smaller triangle
  d : ℝ
  e : ℝ
  -- Conditions
  ha : a = 16
  hb : b = 20
  hc : c = 24
  hangle : angle = 30 * π / 180
  hd : d = 8
  he : e = 12
  -- Similarity condition
  similar : a / d = b / e

/-- The third side of the smaller triangle is 12 cm -/
theorem third_side_length (t : SimilarTriangles) : t.c / t.d = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l3493_349346


namespace NUMINAMATH_CALUDE_digit_removal_theorem_l3493_349311

def original_number : ℕ := 111123445678

-- Function to check if a number is divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Function to represent the removal of digits
def remove_digits (n : ℕ) (removed : List ℕ) : ℕ := sorry

-- Function to count valid ways of digit removal
def count_valid_removals (n : ℕ) : ℕ := sorry

theorem digit_removal_theorem :
  count_valid_removals original_number = 60 := by sorry

end NUMINAMATH_CALUDE_digit_removal_theorem_l3493_349311


namespace NUMINAMATH_CALUDE_third_number_tenth_row_l3493_349361

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of integers in the first n rows of the triangular array -/
def numbers_in_rows (n : ℕ) : ℕ := triangular_number (n - 1)

/-- The kth number from the left on the nth row of the triangular array -/
def number_at_position (n : ℕ) (k : ℕ) : ℕ := 
  numbers_in_rows n + k

theorem third_number_tenth_row : 
  number_at_position 10 3 = 48 := by sorry

end NUMINAMATH_CALUDE_third_number_tenth_row_l3493_349361


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3493_349349

theorem binomial_expansion_sum : 
  let f : ℕ → ℕ → ℕ := λ m n => (Nat.choose 6 m) * (Nat.choose 4 n)
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3493_349349


namespace NUMINAMATH_CALUDE_light_beam_reflection_l3493_349374

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two points, returns the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p1.y - p2.y
    b := p2.x - p1.x
    c := p1.x * p2.y - p2.x * p1.y }

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The x-axis -/
def x_axis : Line :=
  { a := 0, b := 1, c := 0 }

/-- Reflects a point across the x-axis -/
def reflect_point_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem light_beam_reflection (M N : Point) 
    (h1 : M = { x := 4, y := 5 })
    (h2 : N = { x := 2, y := 0 })
    (h3 : point_on_line N x_axis) :
  ∃ (l : Line), 
    l = { a := 5, b := -2, c := -10 } ∧ 
    point_on_line M l ∧ 
    point_on_line N l ∧
    point_on_line (reflect_point_x_axis M) l :=
  sorry

end NUMINAMATH_CALUDE_light_beam_reflection_l3493_349374


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l3493_349384

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem intersection_complement_A_and_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l3493_349384


namespace NUMINAMATH_CALUDE_bar_and_line_charts_represent_amount_l3493_349372

-- Define bar charts and line charts as types that can represent data
def BarChart : Type := Unit
def LineChart : Type := Unit

-- Define a property for charts that can represent amount
def CanRepresentAmount (chart : Type) : Prop := True

-- State the theorem
theorem bar_and_line_charts_represent_amount :
  CanRepresentAmount BarChart ∧ CanRepresentAmount LineChart := by
  sorry

end NUMINAMATH_CALUDE_bar_and_line_charts_represent_amount_l3493_349372


namespace NUMINAMATH_CALUDE_tim_and_donna_dating_years_l3493_349352

/-- Represents the timeline of Tim and Donna's relationship -/
structure Relationship where
  meetYear : ℕ
  weddingYear : ℕ
  anniversaryYear : ℕ
  yearsBetweenMeetingAndDating : ℕ

/-- Calculate the number of years Tim and Donna dated before marriage -/
def yearsDatingBeforeMarriage (r : Relationship) : ℕ :=
  r.weddingYear - r.meetYear - r.yearsBetweenMeetingAndDating

/-- The main theorem stating that Tim and Donna dated for 3 years before marriage -/
theorem tim_and_donna_dating_years (r : Relationship) 
  (h1 : r.meetYear = 2000)
  (h2 : r.anniversaryYear = 2025)
  (h3 : r.anniversaryYear - r.weddingYear = 20)
  (h4 : r.yearsBetweenMeetingAndDating = 2) : 
  yearsDatingBeforeMarriage r = 3 := by
  sorry


end NUMINAMATH_CALUDE_tim_and_donna_dating_years_l3493_349352


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3493_349373

theorem quadratic_root_difference (p : ℝ) : 
  p > 0 → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + 1 = 0 ∧ 
    x₂^2 + p*x₂ + 1 = 0 ∧ 
    |x₁ - x₂| = 1) → 
  p = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3493_349373


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l3493_349331

theorem jelly_bean_problem (b c : ℕ) : 
  b = 2 * c →                 -- Initial condition
  b - 5 = 4 * (c - 5) →       -- Condition after eating jelly beans
  b = 30 :=                   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l3493_349331


namespace NUMINAMATH_CALUDE_marbles_remaining_example_l3493_349339

/-- The number of marbles remaining after distribution -/
def marblesRemaining (chris ryan alex : ℕ) : ℕ :=
  let total := chris + ryan + alex
  let chrisShare := total / 4
  let ryanShare := total / 4
  let alexShare := total / 3
  total - (chrisShare + ryanShare + alexShare)

/-- Theorem stating the number of marbles remaining in the specific scenario -/
theorem marbles_remaining_example : marblesRemaining 12 28 18 = 11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_example_l3493_349339


namespace NUMINAMATH_CALUDE_map_distance_between_mountains_l3493_349383

/-- Given a map with a known scale factor, this theorem proves that the distance
    between two mountains on the map is 312 inches, given their actual distance
    and a reference point. -/
theorem map_distance_between_mountains
  (actual_distance : ℝ)
  (map_reference : ℝ)
  (actual_reference : ℝ)
  (h1 : actual_distance = 136)
  (h2 : map_reference = 28)
  (h3 : actual_reference = 12.205128205128204)
  : (actual_distance / (actual_reference / map_reference)) = 312 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_between_mountains_l3493_349383


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l3493_349304

theorem xyz_product_magnitude (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) :
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l3493_349304


namespace NUMINAMATH_CALUDE_solve_equation_l3493_349340

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3493_349340


namespace NUMINAMATH_CALUDE_allen_age_difference_l3493_349381

/-- Allen's age problem -/
theorem allen_age_difference :
  ∀ (allen_age mother_age : ℕ),
  mother_age = 30 →
  allen_age + 3 + mother_age + 3 = 41 →
  mother_age - allen_age = 25 :=
by sorry

end NUMINAMATH_CALUDE_allen_age_difference_l3493_349381


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3493_349305

theorem fraction_power_equality : (123456 / 41152)^5 = 243 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3493_349305


namespace NUMINAMATH_CALUDE_polygon_perimeter_l3493_349315

/-- The perimeter of a polygon formed by removing a right triangle from a rectangle. -/
theorem polygon_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_height : ℝ) :
  rectangle_length = 10 →
  rectangle_width = 6 →
  triangle_height = 4 →
  ∃ (polygon_perimeter : ℝ),
    polygon_perimeter = 22 + 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_l3493_349315
