import Mathlib

namespace NUMINAMATH_CALUDE_line_contains_point_l402_40265

theorem line_contains_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -4 * 1) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l402_40265


namespace NUMINAMATH_CALUDE_intersection_M_N_l402_40232

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l402_40232


namespace NUMINAMATH_CALUDE_larger_number_of_product_35_sum_12_l402_40212

theorem larger_number_of_product_35_sum_12 :
  ∀ x y : ℕ,
  x * y = 35 →
  x + y = 12 →
  max x y = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_product_35_sum_12_l402_40212


namespace NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l402_40240

theorem integral_sqrt_x_2_minus_x (x : ℝ) : ∫ x in (0:ℝ)..1, Real.sqrt (x * (2 - x)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l402_40240


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l402_40210

theorem triangle_circumcircle_radius 
  (a : ℝ) 
  (A : ℝ) 
  (h1 : a = 2) 
  (h2 : A = 2 * π / 3) : 
  ∃ R : ℝ, R = (2 * Real.sqrt 3) / 3 ∧ 
  R = a / (2 * Real.sin A) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l402_40210


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_ratio_l402_40296

theorem rectangular_field_diagonal_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →  -- x and y are positive (representing sides of a rectangle)
    x + y - Real.sqrt (x^2 + y^2) = (2/3) * y →  -- diagonal walk saves 2/3 of longer side
    x / y = 8/9 :=  -- ratio of shorter to longer side
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_ratio_l402_40296


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l402_40234

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (4 - 5*x + y) = 9 → y = 77 + 5*x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l402_40234


namespace NUMINAMATH_CALUDE_curve_expression_bound_l402_40254

theorem curve_expression_bound :
  ∀ x y : ℝ, x^2 + (y^2)/4 = 4 → 
  ∃ t : ℝ, x = 2*Real.cos t ∧ y = 4*Real.sin t ∧ 
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_expression_bound_l402_40254


namespace NUMINAMATH_CALUDE_skyscraper_arrangement_impossible_l402_40237

/-- The number of cyclic permutations of n elements -/
def cyclic_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The maximum number of regions that n lines can divide a plane into -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of lines connecting n points -/
def connecting_lines (n : ℕ) : ℕ := n.choose 2

theorem skyscraper_arrangement_impossible :
  let n := 7
  let permutations := cyclic_permutations n
  let lines := connecting_lines n
  let regions := max_regions lines
  regions < permutations := by sorry

end NUMINAMATH_CALUDE_skyscraper_arrangement_impossible_l402_40237


namespace NUMINAMATH_CALUDE_reads_two_days_per_week_l402_40238

/-- A person's reading habits over a period of weeks -/
structure ReadingHabits where
  booksPerDay : ℕ
  totalBooks : ℕ
  totalWeeks : ℕ

/-- Calculate the number of days per week a person reads based on their reading habits -/
def daysPerWeek (habits : ReadingHabits) : ℚ :=
  (habits.totalBooks / habits.booksPerDay : ℚ) / habits.totalWeeks

/-- Theorem: Given the specific reading habits, prove that the person reads 2 days per week -/
theorem reads_two_days_per_week (habits : ReadingHabits)
  (h1 : habits.booksPerDay = 4)
  (h2 : habits.totalBooks = 48)
  (h3 : habits.totalWeeks = 6) :
  daysPerWeek habits = 2 := by
  sorry

end NUMINAMATH_CALUDE_reads_two_days_per_week_l402_40238


namespace NUMINAMATH_CALUDE_fraction_simplification_l402_40287

/-- Proves that for x = 198719871987, the fraction 198719871987 / (x^2 - (x-1)(x+1)) simplifies to 1987 -/
theorem fraction_simplification (x : ℕ) (h : x = 198719871987) :
  (x : ℚ) / (x^2 - (x-1)*(x+1)) = 1987 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l402_40287


namespace NUMINAMATH_CALUDE_percentage_of_employed_females_l402_40249

-- Define the given percentages
def total_employed_percent : ℝ := 64
def employed_males_percent : ℝ := 46

-- Define the theorem
theorem percentage_of_employed_females :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 28.125 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_employed_females_l402_40249


namespace NUMINAMATH_CALUDE_coin_division_problem_l402_40256

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l402_40256


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l402_40222

/-- Given a triangle with perimeter 720 cm and longest side 280 cm, 
    prove that the ratio of the sides can be expressed as k:l:1, where k + l = 1.5714 -/
theorem triangle_side_ratio (a b c : ℝ) (h_perimeter : a + b + c = 720) 
  (h_longest : c = 280) (h_c_longest : a ≤ c ∧ b ≤ c) :
  ∃ (k l : ℝ), k + l = 1.5714 ∧ (a / c = k ∧ b / c = l) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l402_40222


namespace NUMINAMATH_CALUDE_even_function_m_value_l402_40257

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^4 + (m-1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^4 + (m-1)*x + 1

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l402_40257


namespace NUMINAMATH_CALUDE_probability_x_equals_y_l402_40213

-- Define the range for x and y
def valid_range (x : ℝ) : Prop := -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi

-- Define the condition for x and y
def condition (x y : ℝ) : Prop := Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Define the total number of valid pairs
def total_pairs : ℕ := 121

-- Define the number of pairs where X = Y
def equal_pairs : ℕ := 11

-- State the theorem
theorem probability_x_equals_y :
  (∀ x y : ℝ, valid_range x → valid_range y → condition x y) →
  (equal_pairs : ℕ) / (total_pairs : ℕ) = 1 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_x_equals_y_l402_40213


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l402_40251

/-- Given a square with perimeter 256 units divided into 16 equal smaller squares,
    each further divided into two rectangles along a diagonal,
    the perimeter of one of these smaller rectangles is 32 + 16√2 units. -/
theorem small_rectangle_perimeter (large_square_perimeter : ℝ) 
  (h1 : large_square_perimeter = 256) 
  (num_divisions : ℕ) 
  (h2 : num_divisions = 16) : ℝ :=
by
  -- Define the perimeter of one small rectangle
  let small_rectangle_perimeter := 32 + 16 * Real.sqrt 2
  
  -- Prove that this is indeed the perimeter
  sorry

#check small_rectangle_perimeter

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l402_40251


namespace NUMINAMATH_CALUDE_power_of_product_l402_40273

theorem power_of_product (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l402_40273


namespace NUMINAMATH_CALUDE_zoo_ticket_sales_l402_40219

/-- Calculates the total money made from ticket sales at a zoo -/
theorem zoo_ticket_sales (total_people : ℕ) (adult_price kid_price : ℕ) (num_kids : ℕ) : 
  total_people = 254 → 
  adult_price = 28 → 
  kid_price = 12 → 
  num_kids = 203 → 
  (total_people - num_kids) * adult_price + num_kids * kid_price = 3864 := by
sorry

end NUMINAMATH_CALUDE_zoo_ticket_sales_l402_40219


namespace NUMINAMATH_CALUDE_molecular_weight_c4h10_l402_40284

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The number of Carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The number of moles of C4H10 -/
def mole_count : ℝ := 6

/-- Theorem: The molecular weight of 6 moles of C4H10 is 348.72 grams -/
theorem molecular_weight_c4h10 :
  (carbon_weight * carbon_count + hydrogen_weight * hydrogen_count) * mole_count = 348.72 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_c4h10_l402_40284


namespace NUMINAMATH_CALUDE_number_equation_solution_l402_40295

theorem number_equation_solution : ∃ x : ℚ, 3 + (1/2) * (1/3) * (1/5) * x = (1/15) * x ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l402_40295


namespace NUMINAMATH_CALUDE_sandwich_combinations_theorem_l402_40203

def num_meat_types : ℕ := 8
def num_cheese_types : ℕ := 7

def num_meat_combinations : ℕ := (num_meat_types * (num_meat_types - 1)) / 2
def num_cheese_combinations : ℕ := num_cheese_types

def total_sandwich_combinations : ℕ := num_meat_combinations * num_cheese_combinations

theorem sandwich_combinations_theorem : total_sandwich_combinations = 196 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_theorem_l402_40203


namespace NUMINAMATH_CALUDE_boat_license_combinations_l402_40224

/-- The number of possible letters for the first character of a boat license -/
def numLetters : ℕ := 3

/-- The number of possible digits for each of the six numeric positions in a boat license -/
def numDigits : ℕ := 10

/-- The number of numeric positions in a boat license -/
def numPositions : ℕ := 6

/-- The total number of unique boat license combinations -/
def totalCombinations : ℕ := numLetters * (numDigits ^ numPositions)

theorem boat_license_combinations :
  totalCombinations = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l402_40224


namespace NUMINAMATH_CALUDE_certain_number_is_even_l402_40276

theorem certain_number_is_even (z : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) :
  ∀ x : ℤ, (z * (2 + x + z) + 3) % 2 = 1 ↔ Even x :=
by sorry

end NUMINAMATH_CALUDE_certain_number_is_even_l402_40276


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l402_40262

def A : Set ℝ := {2, 3, 4, 5, 6}
def B : Set ℝ := {x : ℝ | x^2 - 8*x + 12 ≥ 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l402_40262


namespace NUMINAMATH_CALUDE_three_large_five_small_capacity_l402_40279

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 2 large trucks and 3 small trucks is 15.5 tons -/
axiom condition1 : 2 * large_truck_capacity + 3 * small_truck_capacity = 15.5

/-- The total capacity of 5 large trucks and 6 small trucks is 35 tons -/
axiom condition2 : 5 * large_truck_capacity + 6 * small_truck_capacity = 35

/-- Theorem: 3 large trucks and 5 small trucks can transport 24.5 tons -/
theorem three_large_five_small_capacity : 
  3 * large_truck_capacity + 5 * small_truck_capacity = 24.5 := by sorry

end NUMINAMATH_CALUDE_three_large_five_small_capacity_l402_40279


namespace NUMINAMATH_CALUDE_annual_interest_rate_is_eight_percent_l402_40226

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem annual_interest_rate_is_eight_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (total : ℝ) 
  (time : ℕ) 
  (h1 : principal + interest = total)
  (h2 : interest = 2828.80)
  (h3 : total = 19828.80)
  (h4 : time = 2) :
  compound_interest principal 0.08 time = interest := by
  sorry

#check annual_interest_rate_is_eight_percent

end NUMINAMATH_CALUDE_annual_interest_rate_is_eight_percent_l402_40226


namespace NUMINAMATH_CALUDE_sweet_potatoes_theorem_l402_40245

def sweet_potatoes_problem (total_harvested sold_to_adams sold_to_lenon : ℕ) : Prop :=
  total_harvested - (sold_to_adams + sold_to_lenon) = 45

theorem sweet_potatoes_theorem :
  sweet_potatoes_problem 80 20 15 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potatoes_theorem_l402_40245


namespace NUMINAMATH_CALUDE_sheepdog_roundup_percentage_l402_40229

theorem sheepdog_roundup_percentage (total_sheep : ℕ) 
  (sheep_in_pen : ℕ) (sheep_in_wilderness : ℕ) :
  sheep_in_pen = 81 →
  sheep_in_wilderness = 9 →
  sheep_in_wilderness = total_sheep / 10 →
  (sheep_in_pen : ℚ) / total_sheep * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_sheepdog_roundup_percentage_l402_40229


namespace NUMINAMATH_CALUDE_min_sum_fraction_l402_40266

def Digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def IsValidSelection (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def SumFraction (a b c d : Nat) : Rat :=
  a / b + c / d

theorem min_sum_fraction :
  ∃ (a b c d : Nat), IsValidSelection a b c d ∧
    (∀ (w x y z : Nat), IsValidSelection w x y z →
      SumFraction a b c d ≤ SumFraction w x y z) ∧
    SumFraction a b c d = 17 / 15 :=
  sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l402_40266


namespace NUMINAMATH_CALUDE_product_one_sum_lower_bound_l402_40209

theorem product_one_sum_lower_bound (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_lower_bound_l402_40209


namespace NUMINAMATH_CALUDE_equation_solutions_l402_40241

theorem equation_solutions :
  (∃ x₁ x₂, (3 * x₁ + 2)^2 = 16 ∧ (3 * x₂ + 2)^2 = 16 ∧ x₁ = 2/3 ∧ x₂ = -2) ∧
  (∃ x, (1/2) * (2 * x - 1)^3 = -4 ∧ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l402_40241


namespace NUMINAMATH_CALUDE_opposite_number_l402_40227

theorem opposite_number (a : ℝ) : -(3*a - 2) = -3*a + 2 := by sorry

end NUMINAMATH_CALUDE_opposite_number_l402_40227


namespace NUMINAMATH_CALUDE_remainder_problem_l402_40259

theorem remainder_problem (D : ℕ) (h1 : D = 13) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) :
  242 % D = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l402_40259


namespace NUMINAMATH_CALUDE_inequality_proof_l402_40285

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l402_40285


namespace NUMINAMATH_CALUDE_complement_A_in_U_l402_40288

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x ≤ 1}

-- Define set A
def A : Set ℝ := {x : ℝ | x < 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l402_40288


namespace NUMINAMATH_CALUDE_largest_y_value_l402_40247

theorem largest_y_value (y : ℝ) : 
  (y / 3 + 2 / (3 * y) = 1) → y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l402_40247


namespace NUMINAMATH_CALUDE_student_age_problem_l402_40223

theorem student_age_problem (total_students : ℕ) (total_average_age : ℕ) 
  (group1_students : ℕ) (group1_average_age : ℕ) 
  (group2_students : ℕ) (group2_average_age : ℕ) :
  total_students = 20 →
  total_average_age = 20 →
  group1_students = 9 →
  group1_average_age = 11 →
  group2_students = 10 →
  group2_average_age = 24 →
  (total_students * total_average_age - 
   (group1_students * group1_average_age + group2_students * group2_average_age)) = 61 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l402_40223


namespace NUMINAMATH_CALUDE_monkey_climb_l402_40283

/-- The height of the tree climbed by the monkey -/
def tree_height : ℕ := 21

/-- The net progress of the monkey per hour -/
def net_progress_per_hour : ℕ := 1

/-- The time taken by the monkey to reach the top of the tree -/
def total_hours : ℕ := 19

/-- The distance the monkey hops up in the last hour -/
def last_hop : ℕ := 3

theorem monkey_climb :
  tree_height = net_progress_per_hour * (total_hours - 1) + last_hop := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_l402_40283


namespace NUMINAMATH_CALUDE_two_digit_pairs_count_l402_40235

/-- Given two natural numbers x and y, returns true if they contain only two different digits --/
def hasTwoDigits (x y : ℕ) : Prop := sorry

/-- The number of pairs (x, y) where x and y are three-digit numbers, 
    x + y = 999, and x and y together contain only two different digits --/
def countTwoDigitPairs : ℕ := sorry

theorem two_digit_pairs_count : countTwoDigitPairs = 40 := by sorry

end NUMINAMATH_CALUDE_two_digit_pairs_count_l402_40235


namespace NUMINAMATH_CALUDE_polynomial_simplification_l402_40248

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 7 * q^3 + 3 * q + 8) + (5 - 9 * q^3 + 4 * q^2 - 2 * q) =
  4 * q^4 - 16 * q^3 + 4 * q^2 + q + 13 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l402_40248


namespace NUMINAMATH_CALUDE_m_range_l402_40242

/-- The range of m given the specified conditions -/
theorem m_range (h1 : ∀ x : ℝ, 2 * x > m * (x^2 + 1)) 
                (h2 : ∃ x₀ : ℝ, x₀^2 + 2*x₀ - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l402_40242


namespace NUMINAMATH_CALUDE_binomial_odd_iff_power_of_two_minus_one_l402_40255

theorem binomial_odd_iff_power_of_two_minus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Odd (Nat.choose n k)) ↔
  ∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_odd_iff_power_of_two_minus_one_l402_40255


namespace NUMINAMATH_CALUDE_problem_solution_l402_40208

theorem problem_solution : (3358 / 46) - 27 = 46 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l402_40208


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l402_40261

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Part 1
theorem solution_set_when_a_is_one :
  let a := 1
  {x : ℝ | f x a ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | -6 ≤ a ∧ a ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l402_40261


namespace NUMINAMATH_CALUDE_seed_purchase_calculation_l402_40216

/-- Given the cost of seeds and the amount spent by a farmer, 
    calculate the number of pounds of seeds purchased. -/
theorem seed_purchase_calculation 
  (seed_cost : ℝ) 
  (seed_amount : ℝ) 
  (farmer_spent : ℝ) 
  (h1 : seed_cost = 44.68)
  (h2 : seed_amount = 2)
  (h3 : farmer_spent = 134.04) :
  farmer_spent / (seed_cost / seed_amount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_seed_purchase_calculation_l402_40216


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l402_40264

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 55 → 
  a = 210 → 
  b = 605 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l402_40264


namespace NUMINAMATH_CALUDE_correct_answers_for_given_exam_l402_40244

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  totalScore : ℤ

/-- Calculates the number of correctly answered questions. -/
def correctAnswers (result : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that given the specific exam conditions, 
    the number of correctly answered questions is 44. -/
theorem correct_answers_for_given_exam : 
  let exam : Exam := { totalQuestions := 60, correctScore := 4, wrongScore := -1 }
  let result : ExamResult := { exam := exam, totalScore := 160 }
  correctAnswers result = 44 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_exam_l402_40244


namespace NUMINAMATH_CALUDE_new_student_info_is_unique_l402_40270

-- Define the possible values for each attribute
inductive Surname
  | Ji | Zhang | Chen | Huang
deriving Repr, DecidableEq

inductive Gender
  | Male | Female
deriving Repr, DecidableEq

inductive Specialty
  | Singing | Dancing | Drawing
deriving Repr, DecidableEq

-- Define a structure for student information
structure StudentInfo where
  surname : Surname
  gender : Gender
  totalScore : Nat
  specialty : Specialty
deriving Repr

-- Define the information provided by each classmate
def classmate_A : StudentInfo := ⟨Surname.Ji, Gender.Male, 260, Specialty.Singing⟩
def classmate_B : StudentInfo := ⟨Surname.Zhang, Gender.Female, 220, Specialty.Dancing⟩
def classmate_C : StudentInfo := ⟨Surname.Chen, Gender.Male, 260, Specialty.Singing⟩
def classmate_D : StudentInfo := ⟨Surname.Huang, Gender.Female, 220, Specialty.Drawing⟩
def classmate_E : StudentInfo := ⟨Surname.Zhang, Gender.Female, 240, Specialty.Singing⟩

-- Define the correct information
def correct_info : StudentInfo := ⟨Surname.Huang, Gender.Male, 240, Specialty.Dancing⟩

-- Define a function to check if a piece of information is correct
def is_correct_piece (info : StudentInfo) (correct : StudentInfo) : Bool :=
  info.surname = correct.surname ∨ 
  info.gender = correct.gender ∨ 
  info.totalScore = correct.totalScore ∨ 
  info.specialty = correct.specialty

-- Theorem statement
theorem new_student_info_is_unique :
  (is_correct_piece classmate_A correct_info) ∧
  (is_correct_piece classmate_B correct_info) ∧
  (is_correct_piece classmate_C correct_info) ∧
  (is_correct_piece classmate_D correct_info) ∧
  (is_correct_piece classmate_E correct_info) ∧
  (∀ info : StudentInfo, 
    info ≠ correct_info → 
    (¬(is_correct_piece classmate_A info) ∨
     ¬(is_correct_piece classmate_B info) ∨
     ¬(is_correct_piece classmate_C info) ∨
     ¬(is_correct_piece classmate_D info) ∨
     ¬(is_correct_piece classmate_E info))) :=
by sorry

end NUMINAMATH_CALUDE_new_student_info_is_unique_l402_40270


namespace NUMINAMATH_CALUDE_cafeteria_seating_capacity_l402_40202

theorem cafeteria_seating_capacity
  (total_tables : ℕ)
  (occupied_ratio : ℚ)
  (occupied_seats : ℕ)
  (h1 : total_tables = 15)
  (h2 : occupied_ratio = 9/10)
  (h3 : occupied_seats = 135) :
  (occupied_seats / occupied_ratio) / total_tables = 10 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_seating_capacity_l402_40202


namespace NUMINAMATH_CALUDE_not_divides_two_pow_minus_one_l402_40239

theorem not_divides_two_pow_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_two_pow_minus_one_l402_40239


namespace NUMINAMATH_CALUDE_function_inequality_l402_40269

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 4) = -f x)
  (h_decreasing : is_decreasing_on f 0 4) :
  f 13 < f 10 ∧ f 10 < f 15 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l402_40269


namespace NUMINAMATH_CALUDE_comparison_theorem_l402_40267

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ -(-3) > -|(-3)| := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l402_40267


namespace NUMINAMATH_CALUDE_system_solution_correct_l402_40274

theorem system_solution_correct (x y : ℚ) :
  x = 2 ∧ y = 1/2 → (x - 2*y = 1 ∧ 2*x + 2*y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_correct_l402_40274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l402_40277

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l402_40277


namespace NUMINAMATH_CALUDE_wednesday_earnings_l402_40280

/-- Represents the types of lawns --/
inductive LawnType
| Small
| Medium
| Large

/-- Represents the charge rates for different lawn types --/
def charge_rate (lt : LawnType) : ℕ :=
  match lt with
  | LawnType.Small => 5
  | LawnType.Medium => 7
  | LawnType.Large => 10

/-- Extra fee for lawns with large piles of leaves --/
def large_pile_fee : ℕ := 3

/-- Calculates the earnings for a given day --/
def daily_earnings (small_bags medium_bags large_bags large_piles : ℕ) : ℕ :=
  small_bags * charge_rate LawnType.Small +
  medium_bags * charge_rate LawnType.Medium +
  large_bags * charge_rate LawnType.Large +
  large_piles * large_pile_fee

/-- Represents the work done on Monday --/
def monday_work : ℕ := daily_earnings 4 2 1 1

/-- Represents the work done on Tuesday --/
def tuesday_work : ℕ := daily_earnings 2 1 2 1

/-- Total earnings after three days --/
def total_earnings : ℕ := 163

/-- Theorem stating that Wednesday's earnings are $76 --/
theorem wednesday_earnings :
  total_earnings - (monday_work + tuesday_work) = 76 := by sorry

end NUMINAMATH_CALUDE_wednesday_earnings_l402_40280


namespace NUMINAMATH_CALUDE_min_expense_is_2200_l402_40236

/-- Represents the types of trucks available --/
inductive TruckType
| A
| B

/-- Represents the characteristics of a truck type --/
structure TruckInfo where
  cost : ℕ
  capacity : ℕ

/-- The problem setup --/
def problem_setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ :=
  (λ t => match t with
    | TruckType.A => ⟨400, 20⟩
    | TruckType.B => ⟨300, 10⟩,
   4,  -- number of Type A trucks
   8,  -- number of Type B trucks
   100) -- total air conditioners to transport

/-- Calculate the minimum transportation expense --/
def min_transportation_expense (setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem min_expense_is_2200 :
  min_transportation_expense problem_setup = 2200 :=
sorry

end NUMINAMATH_CALUDE_min_expense_is_2200_l402_40236


namespace NUMINAMATH_CALUDE_ab_product_l402_40263

theorem ab_product (a b : ℚ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_product_l402_40263


namespace NUMINAMATH_CALUDE_inequality_proof_l402_40225

theorem inequality_proof (a b : ℝ) (h : a ≠ b) : a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l402_40225


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l402_40291

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- State the theorem
theorem sufficient_but_not_necessary : 
  (p 2) ∧ (∃ a : ℝ, a ≠ 2 ∧ p a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l402_40291


namespace NUMINAMATH_CALUDE_blueberries_per_basket_l402_40218

theorem blueberries_per_basket (initial_basket : ℕ) (additional_baskets : ℕ) (total_blueberries : ℕ) : 
  initial_basket > 0 →
  additional_baskets = 9 →
  total_blueberries = 200 →
  total_blueberries = (initial_basket + additional_baskets) * initial_basket →
  initial_basket = 20 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_per_basket_l402_40218


namespace NUMINAMATH_CALUDE_two_heart_three_l402_40258

/-- The ♥ operation defined as a ♥ b = ab³ - 2b + 3 -/
def heart (a b : ℝ) : ℝ := a * b^3 - 2*b + 3

/-- Theorem stating that 2 ♥ 3 = 51 -/
theorem two_heart_three : heart 2 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_two_heart_three_l402_40258


namespace NUMINAMATH_CALUDE_power_function_through_point_l402_40204

theorem power_function_through_point (α : ℝ) : 
  (∀ x : ℝ, x > 0 → (fun x => x^α) x = x^α) → 
  (2 : ℝ)^α = 4 → 
  α = 2 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l402_40204


namespace NUMINAMATH_CALUDE_jump_rope_competition_theorem_l402_40253

/-- Represents a jump rope competition for a class of students. -/
structure JumpRopeCompetition where
  totalStudents : ℕ
  initialParticipants : ℕ
  initialAverage : ℕ
  lateStudentScores : List ℕ

/-- Calculates the new average score for the entire class after late students participate. -/
def newAverageScore (comp : JumpRopeCompetition) : ℚ :=
  let initialTotal := comp.initialParticipants * comp.initialAverage
  let lateTotal := comp.lateStudentScores.sum
  let totalJumps := initialTotal + lateTotal
  totalJumps / comp.totalStudents

/-- The main theorem stating that for the given competition parameters, 
    the new average score is 21. -/
theorem jump_rope_competition_theorem (comp : JumpRopeCompetition) 
  (h1 : comp.totalStudents = 30)
  (h2 : comp.initialParticipants = 26)
  (h3 : comp.initialAverage = 20)
  (h4 : comp.lateStudentScores = [26, 27, 28, 29]) :
  newAverageScore comp = 21 := by
  sorry

#eval newAverageScore {
  totalStudents := 30,
  initialParticipants := 26,
  initialAverage := 20,
  lateStudentScores := [26, 27, 28, 29]
}

end NUMINAMATH_CALUDE_jump_rope_competition_theorem_l402_40253


namespace NUMINAMATH_CALUDE_max_luggage_length_l402_40215

theorem max_luggage_length : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 ∧
  length / width = 3 / 2 ∧
  length + width + 30 ≤ 160 →
  length ≤ 78 :=
by
  sorry

end NUMINAMATH_CALUDE_max_luggage_length_l402_40215


namespace NUMINAMATH_CALUDE_multiples_of_hundred_sequence_l402_40297

theorem multiples_of_hundred_sequence (start : ℕ) :
  (∃ seq : Finset ℕ,
    seq.card = 10 ∧
    (∀ n ∈ seq, n % 100 = 0) ∧
    (∀ n ∈ seq, start ≤ n ∧ n ≤ 1000) ∧
    1000 ∈ seq) →
  start = 100 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_hundred_sequence_l402_40297


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l402_40220

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x - 5| ∧ |x - 5| ≤ 10 ↔ (-5 ≤ x ∧ x ≤ 2) ∨ (8 ≤ x ∧ x ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l402_40220


namespace NUMINAMATH_CALUDE_drums_per_day_l402_40294

/-- Given that 90 drums are filled in 6 days, prove that 15 drums are filled per day -/
theorem drums_per_day :
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 90 →
    total_days = 6 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 15 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l402_40294


namespace NUMINAMATH_CALUDE_total_crackers_is_14c_l402_40201

/-- Represents the number of cracker packs eaten on each day --/
structure DailyPacks where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of crackers consumed based on daily pack consumption --/
def totalCrackers (d : DailyPacks) (c : ℕ) : ℕ :=
  (d.monday + d.tuesday + d.wednesday + d.thursday + d.friday) * c

/-- The consumption pattern for Nedy --/
def nedyConsumption : DailyPacks where
  monday := 2
  tuesday := 3
  wednesday := 1
  thursday := 2 * 1  -- Double Wednesday's amount
  friday := 2 * (2 + 1)  -- Twice the combined amount of Monday and Wednesday

theorem total_crackers_is_14c (c : ℕ) :
  totalCrackers nedyConsumption c = 14 * c := by
  sorry


end NUMINAMATH_CALUDE_total_crackers_is_14c_l402_40201


namespace NUMINAMATH_CALUDE_quadratic_uniqueness_l402_40282

/-- A quadratic function is uniquely determined by three distinct points -/
theorem quadratic_uniqueness (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :
  ∃! (a b c : ℝ), 
    y₁ = a * x₁^2 + b * x₁ + c ∧
    y₂ = a * x₂^2 + b * x₂ + c ∧
    y₃ = a * x₃^2 + b * x₃ + c := by
  sorry

#check quadratic_uniqueness

end NUMINAMATH_CALUDE_quadratic_uniqueness_l402_40282


namespace NUMINAMATH_CALUDE_transport_cost_calculation_l402_40250

/-- The problem statement for calculating transport cost --/
theorem transport_cost_calculation (purchase_price installation_cost sell_price : ℚ) : 
  purchase_price = 16500 →
  installation_cost = 250 →
  sell_price = 23100 →
  ∃ (labelled_price transport_cost : ℚ),
    purchase_price = labelled_price * (1 - 0.2) ∧
    sell_price = labelled_price * 1.1 + transport_cost + installation_cost ∧
    transport_cost = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_calculation_l402_40250


namespace NUMINAMATH_CALUDE_cost_price_approximation_l402_40271

/-- The cost price of a single toy given the selling conditions -/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy / (1 + gain_in_toys / num_toys)
  x

/-- Theorem stating the cost price of a toy given the problem conditions -/
theorem cost_price_approximation :
  let result := cost_price_of_toy 18 23100 3
  (result > 1099.99) ∧ (result < 1100.01) := by
  sorry

#eval cost_price_of_toy 18 23100 3

end NUMINAMATH_CALUDE_cost_price_approximation_l402_40271


namespace NUMINAMATH_CALUDE_ages_solution_l402_40221

/-- Represents the ages of three persons --/
structure Ages where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.eldest = ages.middle + 16 ∧
  ages.middle = ages.youngest + 8 ∧
  ages.eldest - 6 = 3 * (ages.youngest - 6) ∧
  ages.eldest - 6 = 2 * (ages.middle - 6)

/-- Theorem stating that the ages 18, 26, and 42 satisfy the problem conditions --/
theorem ages_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ 
    ages.youngest = 18 ∧ ages.middle = 26 ∧ ages.eldest = 42 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l402_40221


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l402_40299

-- Define the function f
def f (x : ℝ) : ℝ := 1 - |x - 2|

-- Theorem for the first part
theorem solution_set_f (x : ℝ) :
  f x > 1 - |x + 4| ↔ x > -1 :=
sorry

-- Theorem for the second part
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 2 (5/2), f x > |x - m|) ↔ m ∈ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l402_40299


namespace NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l402_40290

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3*x ≠ 6 ∧ (5*x - 4) / (x - 2) = (4*x + 10) / (3*x - 6) - 1

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ 1 - (x - 2) / (2 + x) = 16 / (x^2 - 4)

-- Theorem for the first equation
theorem no_solution_equation1 : ¬∃ x, equation1 x :=
  sorry

-- Theorem for the second equation
theorem unique_solution_equation2 : ∃! x, equation2 x ∧ x = 6 :=
  sorry

end NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l402_40290


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l402_40207

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its three edge lengths. -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The surface area of a rectangular solid is twice the sum of the areas of its three distinct faces. -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

/-- Theorem: For a rectangular solid with prime edge lengths and a volume of 1001 cubic units, 
    the total surface area is 622 square units. -/
theorem rectangular_solid_surface_area :
  ∀ a b c : ℕ,
  is_prime a ∧ is_prime b ∧ is_prime c →
  volume a b c = 1001 →
  surface_area a b c = 622 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l402_40207


namespace NUMINAMATH_CALUDE_union_of_sets_l402_40281

theorem union_of_sets : 
  let M : Set ℕ := {2, 3, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∪ N = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l402_40281


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l402_40278

theorem contrapositive_equivalence :
  (∀ x : ℝ, x > 10 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l402_40278


namespace NUMINAMATH_CALUDE_triangle_inequality_in_necklace_l402_40243

theorem triangle_inequality_in_necklace :
  ∀ (a : ℕ → ℕ),
  (∀ n, 290 ≤ a n ∧ a n ≤ 2023) →
  (∀ m n, m ≠ n → a m ≠ a n) →
  ∃ i, a i + a (i + 1) > a (i + 2) ∧
       a i + a (i + 2) > a (i + 1) ∧
       a (i + 1) + a (i + 2) > a i :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_in_necklace_l402_40243


namespace NUMINAMATH_CALUDE_farm_problem_l402_40205

/-- Proves that given the conditions of the farm problem, the number of hens is 24 -/
theorem farm_problem (hens cows : ℕ) : 
  hens + cows = 48 →
  2 * hens + 4 * cows = 144 →
  hens = 24 := by
sorry

end NUMINAMATH_CALUDE_farm_problem_l402_40205


namespace NUMINAMATH_CALUDE_tamika_always_wins_l402_40298

def tamika_set : Finset ℕ := {7, 11, 14}
def carlos_set : Finset ℕ := {2, 4, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ), a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    ∀ (c d : ℕ), c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
      a * b > c + d :=
by sorry

end NUMINAMATH_CALUDE_tamika_always_wins_l402_40298


namespace NUMINAMATH_CALUDE_sum_and_ratio_problem_l402_40272

theorem sum_and_ratio_problem (x y : ℚ) 
  (sum_eq : x + y = 520)
  (ratio_eq : x / y = 3 / 4) :
  y - x = 520 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_problem_l402_40272


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l402_40211

theorem quadratic_equal_roots : ∃ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ∧
  ∀ y : ℝ, 4 * y^2 - 4 * y + 1 = 0 → y = x := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l402_40211


namespace NUMINAMATH_CALUDE_prime_power_expression_l402_40230

theorem prime_power_expression (a b : ℕ) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ 
   (a^6 + 21*a^4*b^2 + 35*a^2*b^4 + 7*b^6) * (b^6 + 21*b^4*a^2 + 35*b^2*a^4 + 7*a^6) = p^k) ↔ 
  (∃ (i : ℕ), a = 2^i ∧ b = 2^i) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_expression_l402_40230


namespace NUMINAMATH_CALUDE_heartsuit_problem_l402_40233

def heartsuit (a b : ℝ) : ℝ := |a + b|

theorem heartsuit_problem : heartsuit (-3) (heartsuit 5 (-8)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_problem_l402_40233


namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_l402_40260

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem for the solution set of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_l402_40260


namespace NUMINAMATH_CALUDE_book_width_calculation_l402_40286

theorem book_width_calculation (length width area : ℝ) : 
  length = 2 → area = 6 → area = length * width → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_book_width_calculation_l402_40286


namespace NUMINAMATH_CALUDE_bucket_sand_problem_l402_40289

theorem bucket_sand_problem (capacity_A : ℝ) (initial_sand_A : ℝ) :
  capacity_A > 0 →
  initial_sand_A ≥ 0 →
  initial_sand_A ≤ capacity_A →
  let capacity_B := capacity_A / 2
  let sand_B := 3 / 8 * capacity_B
  let total_sand := initial_sand_A + sand_B
  total_sand = 0.4375 * capacity_A →
  initial_sand_A = 1 / 4 * capacity_A :=
by sorry

end NUMINAMATH_CALUDE_bucket_sand_problem_l402_40289


namespace NUMINAMATH_CALUDE_ball_box_arrangement_l402_40217

/-- The number of ways to place n different balls into k boxes -/
def total_arrangements (n k : ℕ) : ℕ := k^n

/-- The number of ways to place n different balls into k boxes, 
    with at least one ball in a specific box -/
def arrangements_with_specific_box (n k : ℕ) : ℕ := 
  total_arrangements n k - total_arrangements n (k-1)

theorem ball_box_arrangement : 
  arrangements_with_specific_box 3 6 = 91 := by sorry

end NUMINAMATH_CALUDE_ball_box_arrangement_l402_40217


namespace NUMINAMATH_CALUDE_sequence_sum_l402_40246

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, b = a * r ∧ c = a * r^2) →  -- geometric progression
  (d = a + 40) →                        -- arithmetic progression and difference
  a + b + c + d = 110 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l402_40246


namespace NUMINAMATH_CALUDE_dans_remaining_money_l402_40292

/-- Dan's initial money in dollars -/
def initial_money : ℝ := 5

/-- Cost of the candy bar in dollars -/
def candy_bar_cost : ℝ := 2

/-- Theorem: Dan's remaining money after buying the candy bar is $3 -/
theorem dans_remaining_money : 
  initial_money - candy_bar_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l402_40292


namespace NUMINAMATH_CALUDE_angle_between_vectors_l402_40228

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : 2 * ‖a‖ = ‖b‖)
  (h2 : ‖b‖ = ‖2 * a - b‖)
  (h3 : 2 * a - b ≠ 0) :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l402_40228


namespace NUMINAMATH_CALUDE_sammy_total_problems_l402_40200

/-- The total number of math problems Sammy had to do -/
def total_problems (finished : ℕ) (remaining : ℕ) : ℕ :=
  finished + remaining

/-- Theorem stating that Sammy's total math problems equal 9 -/
theorem sammy_total_problems :
  total_problems 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sammy_total_problems_l402_40200


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l402_40231

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_investment : ℕ) 
  (a_extra : ℕ) 
  (b_extra : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (profit_ratio_c : ℕ) 
  (total_profit : ℕ) 
  (h1 : total_investment = 120000)
  (h2 : a_extra = 6000)
  (h3 : b_extra = 8000)
  (h4 : profit_ratio_a = 4)
  (h5 : profit_ratio_b = 3)
  (h6 : profit_ratio_c = 2)
  (h7 : total_profit = 50000) :
  (profit_ratio_c : ℚ) / (profit_ratio_a + profit_ratio_b + profit_ratio_c : ℚ) * total_profit = 11111.11 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l402_40231


namespace NUMINAMATH_CALUDE_factor_polynomial_l402_40268

theorem factor_polynomial (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l402_40268


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l402_40214

/-- The polynomial expression -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^3 + x) - 8*(x^5 + x^3 + 3*x) + 6*(3*x^5 - x^2 + 4)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (f : ℝ → ℝ) : ℝ :=
  sorry

theorem leading_coefficient_of_p :
  leading_coefficient p = 15 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l402_40214


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l402_40206

theorem product_of_sum_of_squares (x₁ y₁ x₂ y₂ : ℝ) :
  ∃ u v : ℝ, (x₁^2 + y₁^2) * (x₂^2 + y₂^2) = u^2 + v^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l402_40206


namespace NUMINAMATH_CALUDE_shop_owner_profit_l402_40252

/-- Calculates the percentage profit of a shop owner who cheats with weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.14 →
  selling_cheat = 0.20 →
  (((1 + buying_cheat) / (1 - selling_cheat)) - 1) * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_shop_owner_profit_l402_40252


namespace NUMINAMATH_CALUDE_circle_intersection_range_l402_40293

-- Define the circles
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 4 = 0

def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the intersection condition
def intersect_at_all_times (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ circle_O x y

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect_at_all_times a ↔ 
    ((-2 * Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l402_40293


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l402_40275

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := -2 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 7
def h (x : ℝ) : ℝ := 6 * x^2 + 3 * x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -2 * x^3 - x^2 + 9 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l402_40275
