import Mathlib

namespace prob_art_second_given_pe_first_l1946_194623

def total_courses : ℕ := 6
def pe_courses : ℕ := 4
def art_courses : ℕ := 2

def prob_pe_first : ℚ := pe_courses / total_courses
def prob_art_second : ℚ := art_courses / (total_courses - 1)

theorem prob_art_second_given_pe_first :
  (prob_pe_first * prob_art_second) / prob_pe_first = 2 / 5 :=
by sorry

end prob_art_second_given_pe_first_l1946_194623


namespace intercept_sum_mod_50_l1946_194632

theorem intercept_sum_mod_50 : ∃! (x₀ y₀ : ℕ), 
  x₀ < 50 ∧ y₀ < 50 ∧ 
  (7 * x₀ ≡ 2 [MOD 50]) ∧
  (3 * y₀ ≡ 48 [MOD 50]) ∧
  ((x₀ + y₀) ≡ 2 [MOD 50]) := by
sorry

end intercept_sum_mod_50_l1946_194632


namespace micks_to_macks_l1946_194653

/-- Given the conversion rates between micks, mocks, and macks, 
    prove that 200/3 micks equal 30 macks. -/
theorem micks_to_macks 
  (h1 : (8 : ℚ) * mick = 3 * mock) 
  (h2 : (5 : ℚ) * mock = 6 * mack) : 
  (200 : ℚ) / 3 * mick = 30 * mack :=
by
  sorry


end micks_to_macks_l1946_194653


namespace sufficient_condition_not_necessary_sufficient_but_not_necessary_l1946_194689

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := 2 / m = (m - 1) / 1

/-- Sufficient condition: m = 2 implies the lines are parallel -/
theorem sufficient_condition : parallel 2 := by sorry

/-- Not necessary: there exists m ≠ 2 such that the lines are parallel -/
theorem not_necessary : ∃ m : ℝ, m ≠ 2 ∧ parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary : 
  (parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ parallel m) := by sorry

end sufficient_condition_not_necessary_sufficient_but_not_necessary_l1946_194689


namespace last_two_digits_sum_factorials_l1946_194640

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 15) = last_two_digits (sum_factorials 9) := by
  sorry

end last_two_digits_sum_factorials_l1946_194640


namespace symmetric_point_l1946_194620

/-- The point symmetric to P(2,-3) with respect to the origin is (-2,3). -/
theorem symmetric_point : 
  let P : ℝ × ℝ := (2, -3)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = (-2, 3) := by
sorry

end symmetric_point_l1946_194620


namespace average_speed_of_planets_l1946_194629

/-- Calculates the average speed of Venus, Earth, and Mars in miles per hour -/
theorem average_speed_of_planets (venus_speed earth_speed mars_speed : ℝ) 
  (h1 : venus_speed = 21.9)
  (h2 : earth_speed = 18.5)
  (h3 : mars_speed = 15) :
  (venus_speed * 3600 + earth_speed * 3600 + mars_speed * 3600) / 3 = 66480 := by
  sorry

#eval (21.9 * 3600 + 18.5 * 3600 + 15 * 3600) / 3

end average_speed_of_planets_l1946_194629


namespace divisibility_condition_l1946_194627

theorem divisibility_condition (n : ℕ+) :
  (6^n.val - 1) ∣ (7^n.val - 1) ↔ ∃ k : ℕ, n.val = 4 * k := by
  sorry

end divisibility_condition_l1946_194627


namespace min_value_theorem_min_value_achieved_l1946_194607

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 + 12 * Real.sqrt 6 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 26 + 12 * Real.sqrt 6 :=
by sorry

end min_value_theorem_min_value_achieved_l1946_194607


namespace count_lines_with_integer_chord_l1946_194648

/-- Represents a line in the form kx - y - 4k + 1 = 0 --/
structure Line where
  k : ℝ

/-- Represents the circle x^2 + (y + 1)^2 = 25 --/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 + 1)^2 = 25}

/-- Returns true if the line intersects the circle with a chord of integer length --/
def hasIntegerChord (l : Line) : Prop :=
  ∃ n : ℕ, ∃ p q : ℝ × ℝ,
    p ∈ Circle ∧ q ∈ Circle ∧
    l.k * p.1 - p.2 - 4 * l.k + 1 = 0 ∧
    l.k * q.1 - q.2 - 4 * l.k + 1 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = n^2

/-- The theorem to be proved --/
theorem count_lines_with_integer_chord :
  ∃! (s : Finset Line), s.card = 10 ∧ ∀ l ∈ s, hasIntegerChord l :=
sorry

end count_lines_with_integer_chord_l1946_194648


namespace smallest_perfect_square_multiple_l1946_194602

def n : ℕ := 2023

-- Define 2023 as 7 * 17^2
axiom n_factorization : n = 7 * 17^2

-- Define the function to check if a number is a perfect square
def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

-- Define the function to check if a number is a multiple of 2023
def is_multiple_of_2023 (x : ℕ) : Prop := ∃ k : ℕ, x = k * n

-- Theorem statement
theorem smallest_perfect_square_multiple :
  (7 * n = (7 * 17)^2) ∧
  is_perfect_square (7 * n) ∧
  is_multiple_of_2023 (7 * n) ∧
  (∀ m : ℕ, m < 7 * n → ¬(is_perfect_square m ∧ is_multiple_of_2023 m)) :=
sorry

end smallest_perfect_square_multiple_l1946_194602


namespace specific_system_is_linear_l1946_194625

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ a * x + b * y = c

/-- A system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def specificSystem : EquationSystem where
  eq1 := {
    a := 1
    b := 1
    c := 1
    eq := λ x y => x + y = 1
    eq_def := by sorry
  }
  eq2 := {
    a := 1
    b := -1
    c := 2
    eq := λ x y => x - y = 2
    eq_def := by sorry
  }

/-- Definition of a system of two linear equations -/
def isSystemOfTwoLinearEquations (system : EquationSystem) : Prop :=
  ∃ (x y : ℝ), 
    system.eq1.eq x y ∧ 
    system.eq2.eq x y ∧
    (∀ z, system.eq1.eq x z ↔ system.eq1.a * x + system.eq1.b * z = system.eq1.c) ∧
    (∀ z, system.eq2.eq x z ↔ system.eq2.a * x + system.eq2.b * z = system.eq2.c)

theorem specific_system_is_linear : isSystemOfTwoLinearEquations specificSystem := by
  sorry

end specific_system_is_linear_l1946_194625


namespace all_b_k_divisible_by_six_l1946_194647

/-- The number obtained by writing the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The sum of the squares of the digits of b_n -/
def g (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem all_b_k_divisible_by_six (k : ℕ) (h : 1 ≤ k ∧ k ≤ 50) : 
  6 ∣ g k := by sorry

end all_b_k_divisible_by_six_l1946_194647


namespace greatest_x_value_l1946_194697

theorem greatest_x_value (x : ℝ) : 
  (4 * x^2 + 6 * x + 3 = 5) → x ≤ (1/2 : ℝ) :=
by sorry

end greatest_x_value_l1946_194697


namespace complex_magnitude_example_l1946_194691

theorem complex_magnitude_example : Complex.abs (-5 + (8/3)*Complex.I) = 17/3 := by
  sorry

end complex_magnitude_example_l1946_194691


namespace cos_seventeen_pi_sixths_l1946_194645

theorem cos_seventeen_pi_sixths : Real.cos (17 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seventeen_pi_sixths_l1946_194645


namespace exponent_division_l1946_194628

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end exponent_division_l1946_194628


namespace basket_total_is_40_l1946_194643

/-- A basket containing apples and oranges -/
structure Basket where
  oranges : ℕ
  apples : ℕ

/-- The total number of fruit in the basket -/
def Basket.total (b : Basket) : ℕ := b.oranges + b.apples

theorem basket_total_is_40 (b : Basket) 
  (h1 : b.apples = 3 * b.oranges) 
  (h2 : b.oranges = 10) : 
  b.total = 40 := by
sorry

end basket_total_is_40_l1946_194643


namespace tuesday_sales_total_l1946_194662

/-- Represents the types of flowers sold in the shop -/
inductive FlowerType
  | Rose
  | Lilac
  | Gardenia
  | Tulip
  | Orchid

/-- Represents the sales data for a given day -/
structure SalesData where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ
  tulips : ℕ
  orchids : ℕ

/-- Calculate the total number of flowers sold -/
def totalFlowers (sales : SalesData) : ℕ :=
  sales.roses + sales.lilacs + sales.gardenias + sales.tulips + sales.orchids

/-- Apply Tuesday sales factors to Monday's sales -/
def applyTuesdayFactors (monday : SalesData) : SalesData :=
  { roses := monday.roses - monday.roses * 4 / 100,
    lilacs := monday.lilacs + monday.lilacs * 5 / 100,
    gardenias := monday.gardenias,
    tulips := monday.tulips - monday.tulips * 7 / 100,
    orchids := monday.orchids }

/-- Theorem: Given the conditions, the total number of flowers sold on Tuesday is 214 -/
theorem tuesday_sales_total (monday : SalesData)
  (h1 : monday.lilacs = 15)
  (h2 : monday.roses = 3 * monday.lilacs)
  (h3 : monday.gardenias = monday.lilacs / 2)
  (h4 : monday.tulips = 2 * (monday.roses + monday.gardenias))
  (h5 : monday.orchids = (monday.roses + monday.gardenias + monday.tulips) / 3)
  : totalFlowers (applyTuesdayFactors monday) = 214 := by
  sorry


end tuesday_sales_total_l1946_194662


namespace max_reciprocal_sum_l1946_194674

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 16 → 1/a + 1/b ≤ 1/x + 1/y) ∧ 1/x + 1/y = 9/32 := by
  sorry

end max_reciprocal_sum_l1946_194674


namespace negation_of_universal_proposition_l1946_194698

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, Real.exp x > Real.log x)) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) := by
  sorry

end negation_of_universal_proposition_l1946_194698


namespace correlation_coefficients_relation_l1946_194694

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficients_relation : r1 < 0 ∧ 0 < r2 := by sorry

end correlation_coefficients_relation_l1946_194694


namespace customer_money_problem_l1946_194678

/-- Represents the initial amount of money a customer has --/
structure Money where
  dollars : ℕ
  cents : ℕ

/-- Represents the conditions of the problem --/
def satisfiesConditions (m : Money) : Prop :=
  let totalCents := 100 * m.dollars + m.cents
  let remainingCents := totalCents / 2
  let remainingDollars := remainingCents / 100
  let remainingCentsOnly := remainingCents % 100
  remainingCentsOnly = m.dollars ∧ remainingDollars = 2 * m.cents

/-- The theorem to be proved --/
theorem customer_money_problem :
  ∃ (m : Money), satisfiesConditions m ∧ m.dollars = 99 ∧ m.cents = 98 := by
  sorry

end customer_money_problem_l1946_194678


namespace parallelogram_coordinate_sum_l1946_194677

/-- A parallelogram with vertices P, Q, R, S in 2D space -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sum_coordinates (point : ℝ × ℝ) : ℝ := point.1 + point.2

/-- Theorem: In a parallelogram PQRS with P(-3,-2), Q(1,-5), R(9,1), and P, R opposite vertices,
    the sum of coordinates of S is 9 -/
theorem parallelogram_coordinate_sum (PQRS : Parallelogram) 
    (h1 : PQRS.P = (-3, -2))
    (h2 : PQRS.Q = (1, -5))
    (h3 : PQRS.R = (9, 1))
    (h4 : PQRS.P.1 + PQRS.R.1 = PQRS.Q.1 + PQRS.S.1) 
    (h5 : PQRS.P.2 + PQRS.R.2 = PQRS.Q.2 + PQRS.S.2) :
    sum_coordinates PQRS.S = 9 := by
  sorry

end parallelogram_coordinate_sum_l1946_194677


namespace double_sides_same_perimeter_l1946_194614

/-- A regular polygon with n sides and side length s -/
structure RegularPolygon where
  n : ℕ
  s : ℝ
  h_n : n ≥ 3

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.n * p.s

theorem double_sides_same_perimeter (p : RegularPolygon) :
  ∃ (q : RegularPolygon), q.n = 2 * p.n ∧ perimeter q = perimeter p ∧ q.s = p.s / 2 := by
  sorry

end double_sides_same_perimeter_l1946_194614


namespace complex_number_in_first_quadrant_l1946_194671

def complex_number : ℂ := Complex.I * (1 - Complex.I)

theorem complex_number_in_first_quadrant : 
  complex_number.re > 0 ∧ complex_number.im > 0 := by
  sorry

end complex_number_in_first_quadrant_l1946_194671


namespace woodworker_tables_l1946_194680

/-- Calculates the number of tables made given the total number of furniture legs,
    number of chairs, legs per chair, and legs per table. -/
def tables_made (total_legs : ℕ) (chairs : ℕ) (legs_per_chair : ℕ) (legs_per_table : ℕ) : ℕ :=
  (total_legs - chairs * legs_per_chair) / legs_per_table

/-- Theorem stating that given 40 total furniture legs, 6 chairs made,
    4 legs per chair, and 4 legs per table, the number of tables made is 4. -/
theorem woodworker_tables :
  tables_made 40 6 4 4 = 4 := by
  sorry

end woodworker_tables_l1946_194680


namespace solution_set_x_squared_less_than_one_l1946_194644

theorem solution_set_x_squared_less_than_one :
  {x : ℝ | x^2 < 1} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end solution_set_x_squared_less_than_one_l1946_194644


namespace cost_price_calculation_l1946_194635

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) 
  (h1 : selling_price = 48)
  (h2 : profit_percentage = 20 / 100) :
  ∃ (cost_price : ℚ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 40 := by
  sorry

end cost_price_calculation_l1946_194635


namespace equation_solutions_l1946_194684

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end equation_solutions_l1946_194684


namespace greening_investment_equation_l1946_194686

theorem greening_investment_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  (initial_investment / 1000) * (1 + x)^2 = (final_investment / 1000) :=
by
  sorry

end greening_investment_equation_l1946_194686


namespace stock_investment_calculation_l1946_194617

/-- Given a stock with price 64 and dividend yield 1623%, prove that an investment
    earning 1900 in dividends is approximately 117.00 -/
theorem stock_investment_calculation (stock_price : ℝ) (dividend_yield : ℝ) (dividend_earned : ℝ) :
  stock_price = 64 →
  dividend_yield = 1623 →
  dividend_earned = 1900 →
  ∃ (investment : ℝ), abs (investment - 117.00) < 0.01 := by
sorry

end stock_investment_calculation_l1946_194617


namespace division_problem_l1946_194611

theorem division_problem (L S Q : ℕ) : 
  L - S = 1500 → 
  L = 1782 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end division_problem_l1946_194611


namespace sin_2x_given_cos_l1946_194622

theorem sin_2x_given_cos (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = -7 / 25 := by
  sorry

end sin_2x_given_cos_l1946_194622


namespace constant_term_proof_l1946_194668

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to find the maximum coefficient term
def max_coeff_term (n : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constant_term (n : ℕ) : ℕ := sorry

theorem constant_term_proof (n : ℕ) :
  max_coeff_term n = 6 → constant_term n = 180 :=
by sorry

end constant_term_proof_l1946_194668


namespace S_not_union_of_finite_arithmetic_progressions_l1946_194681

-- Define the set S
def S : Set ℕ := {n : ℕ | ∀ p q : ℕ, (3 : ℚ) / n ≠ 1 / p + 1 / q}

-- Define what it means for a set to be the union of finitely many arithmetic progressions
def is_union_of_finite_arithmetic_progressions (T : Set ℕ) : Prop :=
  ∃ (k : ℕ) (a b : Fin k → ℕ), T = ⋃ i, {n : ℕ | ∃ m : ℕ, n = a i + m * b i}

-- State the theorem
theorem S_not_union_of_finite_arithmetic_progressions :
  ¬(is_union_of_finite_arithmetic_progressions S) := by
  sorry

end S_not_union_of_finite_arithmetic_progressions_l1946_194681


namespace no_even_increasing_function_l1946_194626

open Function

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem stating that no function can be both even and increasing
theorem no_even_increasing_function : ¬ ∃ f : ℝ → ℝ, IsEven f ∧ IsIncreasing f := by
  sorry

end no_even_increasing_function_l1946_194626


namespace complement_of_B_l1946_194618

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 ∧ U x \ B x = {3}) ∨
   (x = Real.sqrt 3 ∧ U x \ B x = {Real.sqrt 3}) ∨
   (x = -Real.sqrt 3 ∧ U x \ B x = {-Real.sqrt 3})) :=
by sorry

end complement_of_B_l1946_194618


namespace power_two_ge_cube_l1946_194606

theorem power_two_ge_cube (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by sorry

end power_two_ge_cube_l1946_194606


namespace ellipse_perpendicular_points_sum_reciprocal_bounds_l1946_194683

/-- The maximum and minimum values of the sum of reciprocals of distances from the center to two perpendicular points on an ellipse -/
theorem ellipse_perpendicular_points_sum_reciprocal_bounds
  (a b : ℝ) (ha : 0 < b) (hab : b < a)
  (P Q : ℝ × ℝ)
  (hP : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1)
  (hQ : (Q.1 / a) ^ 2 + (Q.2 / b) ^ 2 = 1)
  (hPOQ : (P.1 * Q.1 + P.2 * Q.2) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt (Q.1^2 + Q.2^2)) = 0) :
  (a + b) / (a * b) ≤ 1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ∧
  1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ≤ Real.sqrt (2 * (a^2 + b^2)) / (a * b) :=
by sorry

end ellipse_perpendicular_points_sum_reciprocal_bounds_l1946_194683


namespace fraction_integer_pairs_l1946_194665

theorem fraction_integer_pairs (m n : ℕ+) :
  (∃ h : ℕ+, (m.val^2 : ℚ) / (2 * m.val * n.val^2 - n.val^3 + 1) = h.val) ↔
  (∃ k : ℕ+, (m = 2 * k ∧ n = 1) ∨
             (m = k ∧ n = 2 * k) ∨
             (m = 8 * k.val^4 - k.val ∧ n = 2 * k)) :=
by sorry

end fraction_integer_pairs_l1946_194665


namespace system_solution_and_sum_l1946_194634

theorem system_solution_and_sum :
  ∃ (x y : ℚ),
    (4 * x - 6 * y = -3) ∧
    (8 * x + 3 * y = 6) ∧
    (x = 9/20) ∧
    (y = 4/5) ∧
    (x + y = 5/4) := by
  sorry

end system_solution_and_sum_l1946_194634


namespace football_team_progress_l1946_194615

theorem football_team_progress (lost_yards gained_yards : ℤ) (h1 : lost_yards = 5) (h2 : gained_yards = 7) :
  gained_yards - lost_yards = 2 := by
  sorry

end football_team_progress_l1946_194615


namespace sqrt_expression_equals_three_halves_l1946_194605

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 8 - Real.sqrt (1/2)) / Real.sqrt 2 = 3/2 := by
  sorry

end sqrt_expression_equals_three_halves_l1946_194605


namespace zero_sum_and_product_implies_all_zero_l1946_194651

theorem zero_sum_and_product_implies_all_zero (a b c d : ℝ) 
  (sum_zero : a + b + c + d = 0)
  (product_zero : a*b + c*d + a*c + b*c + a*d + b*d = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
sorry

end zero_sum_and_product_implies_all_zero_l1946_194651


namespace quadratic_root_implies_m_value_l1946_194699

theorem quadratic_root_implies_m_value (m : ℝ) :
  (3^2 - 2*3 + m = 0) → m = -3 := by
  sorry

end quadratic_root_implies_m_value_l1946_194699


namespace stationery_box_sheets_l1946_194693

theorem stationery_box_sheets : ∀ (S E : ℕ),
  S - E = 30 →  -- Ann's condition
  2 * E = S →   -- Bob's condition
  3 * E = S - 10 →  -- Sue's condition
  S = 40 := by
sorry

end stationery_box_sheets_l1946_194693


namespace dog_weight_l1946_194654

theorem dog_weight (d l s : ℝ) 
  (total_weight : d + l + s = 36)
  (larger_comparison : d + l = 3 * s)
  (smaller_comparison : d + s = l) : 
  d = 9 := by
  sorry

end dog_weight_l1946_194654


namespace unique_value_not_in_range_l1946_194656

/-- A function f with specific properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The theorem stating the properties of f and its unique value not in its range -/
theorem unique_value_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h19 : f a b c d 19 = 19)
  (h97 : f a b c d 97 = 97)
  (hinv : ∀ x, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 :=
sorry

end unique_value_not_in_range_l1946_194656


namespace sin_minus_cos_value_l1946_194609

theorem sin_minus_cos_value (x : ℝ) (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) : 
  Real.sin x - Real.cos x = -1 := by
sorry

end sin_minus_cos_value_l1946_194609


namespace power_division_equals_square_l1946_194669

theorem power_division_equals_square (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end power_division_equals_square_l1946_194669


namespace max_m_inequality_l1946_194663

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  ∃ m : ℝ, m = 12 ∧ (3 / a + 1 / b ≥ m / (a + 3 * b)) :=
by sorry

end max_m_inequality_l1946_194663


namespace pink_crayons_count_l1946_194679

def total_crayons : ℕ := 24
def red_crayons : ℕ := 8
def blue_crayons : ℕ := 6
def green_crayons : ℕ := (2 * blue_crayons) / 3

theorem pink_crayons_count :
  total_crayons - red_crayons - blue_crayons - green_crayons = 6 := by
  sorry

end pink_crayons_count_l1946_194679


namespace min_seating_circular_table_l1946_194619

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : ℕ
  seated : ℕ

/-- Predicate to check if a seating arrangement is valid. -/
def validSeating (table : CircularTable) : Prop :=
  table.seated ≤ table.chairs ∧
  ∀ k : ℕ, k < table.seated → ∃ j : ℕ, j < table.seated ∧ j ≠ k ∧
    (((k + 1) % table.chairs = j) ∨ ((k + table.chairs - 1) % table.chairs = j))

/-- The theorem to be proved. -/
theorem min_seating_circular_table :
  ∃ (n : ℕ), n = 20 ∧
  validSeating ⟨60, n⟩ ∧
  ∀ m : ℕ, m < n → ¬validSeating ⟨60, m⟩ := by
  sorry

end min_seating_circular_table_l1946_194619


namespace midpoint_triangle_perimeter_l1946_194613

/-- A right prism with regular pentagonal bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Midpoint of an edge -/
structure Midpoint :=
  (edge : String)

/-- Triangle formed by three midpoints -/
structure MidpointTriangle :=
  (p1 : Midpoint)
  (p2 : Midpoint)
  (p3 : Midpoint)

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the midpoint triangle -/
theorem midpoint_triangle_perimeter 
  (prism : RightPrism) 
  (triangle : MidpointTriangle) 
  (h1 : prism.height = 25) 
  (h2 : prism.base_side_length = 15) 
  (h3 : triangle.p1 = Midpoint.mk "AB") 
  (h4 : triangle.p2 = Midpoint.mk "BC") 
  (h5 : triangle.p3 = Midpoint.mk "CD") : 
  perimeter prism triangle = 15 + 2 * Real.sqrt 212.5 :=
sorry

end midpoint_triangle_perimeter_l1946_194613


namespace triangle_part1_triangle_part2_l1946_194624

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Part 1
theorem triangle_part1 (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  ((A = π / 3 ∧ C = 5 * π / 12 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
   (A = 2 * π / 3 ∧ C = π / 12 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
sorry

-- Part 2
theorem triangle_part2 (h1 : Real.cos B / Real.cos C = -b / (2 * a + c)) 
                       (h2 : b = Real.sqrt 13) (h3 : a + c = 4) :
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
sorry

end triangle_part1_triangle_part2_l1946_194624


namespace fraction_reciprocal_difference_l1946_194685

theorem fraction_reciprocal_difference : 
  let f : ℚ := 4/5
  let r : ℚ := 5/4  -- reciprocal of f
  r - f = 9/20 := by sorry

end fraction_reciprocal_difference_l1946_194685


namespace consecutive_integers_median_l1946_194687

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 64 → sum = 4096 → sum = n * median → median = 64 := by
  sorry

end consecutive_integers_median_l1946_194687


namespace smallest_n_congruence_l1946_194670

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 19 * n ≡ 2701 [ZMOD 9] ∧ ∀ (m : ℕ), m > 0 → 19 * m ≡ 2701 [ZMOD 9] → n ≤ m :=
sorry

end smallest_n_congruence_l1946_194670


namespace binomial_coefficient_two_l1946_194608

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l1946_194608


namespace class_size_problem_l1946_194688

theorem class_size_problem (total : ℕ) (sum_fraction : ℕ) 
  (h1 : total = 85) 
  (h2 : sum_fraction = 42) : ∃ (a b : ℕ), 
  a + b = total ∧ 
  (3 * a) / 8 + (3 * b) / 5 = sum_fraction ∧ 
  a = 40 ∧ 
  b = 45 := by
  sorry

end class_size_problem_l1946_194688


namespace complex_inequality_l1946_194650

theorem complex_inequality (a b c : ℂ) (h : a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := by
  sorry

end complex_inequality_l1946_194650


namespace polynomial_division_remainder_l1946_194631

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^4 - 7*x^3 + 18*x^2 - 28*x + 15
  let g (x : ℝ) := x^2 - 3*x + 16/3
  let q (x : ℝ) := x^2 - 4*x + 10/3
  let r (x : ℝ) := 2*x + 103/9
  ∀ x, f x = g x * q x + r x :=
by sorry

end polynomial_division_remainder_l1946_194631


namespace steve_pages_written_l1946_194664

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_in_month : ℕ) (letter_frequency : ℕ) (regular_letter_time : ℕ) 
  (time_per_page : ℕ) (long_letter_time : ℕ) : ℕ :=
  let regular_letters := days_in_month / letter_frequency
  let pages_per_regular_letter := regular_letter_time / time_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := long_letter_time / (2 * time_per_page)
  regular_letter_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by
  sorry

end steve_pages_written_l1946_194664


namespace expression_simplification_l1946_194610

theorem expression_simplification (m n : ℚ) 
  (hm : m = 2) 
  (hn : n = -1/2) : 
  3 * (m^2 - m + n^2) - 2 * (1/2 * m^2 - m*n + 3/2 * n^2) = 0 := by
  sorry

end expression_simplification_l1946_194610


namespace car_speed_problem_l1946_194612

/-- Proves that given a 6-hour trip where the average speed for the first 4 hours is 35 mph
    and the average speed for the entire trip is 38 mph, the average speed for the remaining 2 hours is 44 mph. -/
theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (total_avg_speed : ℝ) :
  total_time = 6 →
  initial_time = 4 →
  initial_speed = 35 →
  total_avg_speed = 38 →
  let remaining_time := total_time - initial_time
  let total_distance := total_avg_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 44 := by
  sorry

end car_speed_problem_l1946_194612


namespace sister_age_2021_l1946_194655

def kelsey_birth_year (kelsey_age_1999 : ℕ) : ℕ := 1999 - kelsey_age_1999

def sister_birth_year (kelsey_birth : ℕ) (age_difference : ℕ) : ℕ := kelsey_birth - age_difference

def current_age (birth_year : ℕ) (current_year : ℕ) : ℕ := current_year - birth_year

theorem sister_age_2021 (kelsey_age_1999 : ℕ) (age_difference : ℕ) (current_year : ℕ) :
  kelsey_age_1999 = 25 →
  age_difference = 3 →
  current_year = 2021 →
  current_age (sister_birth_year (kelsey_birth_year kelsey_age_1999) age_difference) current_year = 50 :=
by sorry

end sister_age_2021_l1946_194655


namespace hyperbola_condition_l1946_194666

/-- Represents a curve defined by the equation x²/(4-t) + y²/(t-1) = 1 --/
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

/-- Defines when C is a hyperbola --/
def is_hyperbola (t : ℝ) : Prop := (4 - t) * (t - 1) < 0

/-- Theorem stating that C is a hyperbola iff t > 4 or t < 1 --/
theorem hyperbola_condition (t : ℝ) : 
  is_hyperbola t ↔ t > 4 ∨ t < 1 := by sorry

end hyperbola_condition_l1946_194666


namespace expression_simplification_l1946_194641

theorem expression_simplification :
  (12 - 2 * Real.sqrt 35 + Real.sqrt 14 + Real.sqrt 10) / (Real.sqrt 7 - Real.sqrt 5 + Real.sqrt 2) = 2 * Real.sqrt 7 - Real.sqrt 2 := by
  sorry

end expression_simplification_l1946_194641


namespace simplify_sqrt_product_simplify_log_product_l1946_194657

-- Part I
theorem simplify_sqrt_product (a : ℝ) (ha : 0 < a) :
  Real.sqrt (a^(1/4)) * Real.sqrt (a * Real.sqrt a) = Real.sqrt a := by sorry

-- Part II
theorem simplify_log_product :
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := by sorry

end simplify_sqrt_product_simplify_log_product_l1946_194657


namespace expression_value_l1946_194660

theorem expression_value (a b c d x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |x| = Real.sqrt 7) :
  x^2 + (a + b) * c * d * x + Real.sqrt (a + b) + (c * d)^(1/3) = 8 := by
  sorry

end expression_value_l1946_194660


namespace mike_toy_expenses_l1946_194636

theorem mike_toy_expenses : 
  let marbles_cost : ℚ := 9.05
  let football_cost : ℚ := 4.95
  let baseball_cost : ℚ := 6.52
  marbles_cost + football_cost + baseball_cost = 20.52 := by
sorry

end mike_toy_expenses_l1946_194636


namespace bicycle_owners_without_cars_proof_l1946_194633

/-- Represents the number of adults who own bicycles but not cars in a population where every adult owns either a bicycle, a car, or both. -/
def bicycle_owners_without_cars (total_adults bicycle_owners car_owners : ℕ) : ℕ :=
  bicycle_owners - (bicycle_owners + car_owners - total_adults)

/-- Theorem stating that in a population of 500 adults where each adult owns either a bicycle, a car, or both, 
    given that 450 adults own bicycles and 120 adults own cars, the number of bicycle owners who do not own a car is 380. -/
theorem bicycle_owners_without_cars_proof :
  bicycle_owners_without_cars 500 450 120 = 380 := by
  sorry

#eval bicycle_owners_without_cars 500 450 120

end bicycle_owners_without_cars_proof_l1946_194633


namespace inequality_system_solution_set_l1946_194675

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x | -3 ≤ x ∧ x < 1} := by
sorry

end inequality_system_solution_set_l1946_194675


namespace negation_of_existence_proposition_l1946_194616

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, |x| + x^2 < 0) ↔ (∀ x : ℝ, |x| + x^2 ≥ 0) := by sorry

end negation_of_existence_proposition_l1946_194616


namespace square_sum_identity_l1946_194646

theorem square_sum_identity (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(3 - x) + (3 - x)^2 = 16 := by
  sorry

end square_sum_identity_l1946_194646


namespace factory_output_percentage_l1946_194695

theorem factory_output_percentage (may_output june_output : ℝ) : 
  may_output = june_output * (1 - 0.2) → 
  (june_output - may_output) / may_output = 0.25 := by
sorry

end factory_output_percentage_l1946_194695


namespace P_k_at_neg_half_is_zero_l1946_194692

/-- The unique polynomial P_k such that P_k(n) = 1^k + 2^k + 3^k + ... + n^k for each positive integer n -/
noncomputable def P_k (k : ℕ+) : ℝ → ℝ :=
  sorry

/-- For any positive integer k, P_k(-1/2) = 0 -/
theorem P_k_at_neg_half_is_zero (k : ℕ+) : P_k k (-1/2) = 0 := by
  sorry

end P_k_at_neg_half_is_zero_l1946_194692


namespace prob_same_length_regular_hexagon_l1946_194682

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_regular_hexagon :
  prob_same_length = 17 / 35 := by sorry

end prob_same_length_regular_hexagon_l1946_194682


namespace fixed_point_of_exponential_function_l1946_194621

/-- The function f(x) = a^(x-1) + 2 has (1, 3) as a fixed point, where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end fixed_point_of_exponential_function_l1946_194621


namespace sum_of_numbers_with_lcm_and_ratio_l1946_194637

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 50 := by
sorry

end sum_of_numbers_with_lcm_and_ratio_l1946_194637


namespace max_profit_zongzi_l1946_194661

/-- Represents the cost and selling prices of zongzi types A and B -/
structure ZongziPrices where
  cost_a : ℚ
  cost_b : ℚ
  sell_a : ℚ
  sell_b : ℚ

/-- Represents the purchase quantities of zongzi types A and B -/
structure ZongziQuantities where
  qty_a : ℕ
  qty_b : ℕ

/-- Calculates the profit given prices and quantities -/
def profit (p : ZongziPrices) (q : ZongziQuantities) : ℚ :=
  (p.sell_a - p.cost_a) * q.qty_a + (p.sell_b - p.cost_b) * q.qty_b

/-- Theorem stating the maximum profit achievable under given conditions -/
theorem max_profit_zongzi (p : ZongziPrices) (q : ZongziQuantities) :
  p.cost_b = p.cost_a + 2 →
  1000 / p.cost_a = 1200 / p.cost_b →
  p.sell_a = 12 →
  p.sell_b = 15 →
  q.qty_a + q.qty_b = 200 →
  q.qty_a ≥ 2 * q.qty_b →
  ∃ (max_q : ZongziQuantities),
    max_q.qty_a = 134 ∧
    max_q.qty_b = 66 ∧
    ∀ (other_q : ZongziQuantities),
      other_q.qty_a + other_q.qty_b = 200 →
      other_q.qty_a ≥ 2 * other_q.qty_b →
      profit p max_q ≥ profit p other_q :=
sorry

end max_profit_zongzi_l1946_194661


namespace grocery_cost_l1946_194658

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ)
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 23) :
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 260.90 := by
  sorry

end grocery_cost_l1946_194658


namespace successor_arrangements_l1946_194639

/-- The number of distinct arrangements of letters in a word -/
def word_arrangements (total_letters : ℕ) (repeated_letters : List (Char × ℕ)) : ℕ :=
  Nat.factorial total_letters / (repeated_letters.map (λ (_, count) => Nat.factorial count)).prod

/-- Theorem: The number of distinct arrangements of SUCCESSOR is 30,240 -/
theorem successor_arrangements :
  word_arrangements 9 [('S', 3), ('C', 2)] = 30240 := by
  sorry

end successor_arrangements_l1946_194639


namespace a_divisible_by_power_of_three_l1946_194600

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (3 * (a n)^2 + 1) / 2 - a n

theorem a_divisible_by_power_of_three (k : ℕ) : 
  ∃ m : ℕ, a (3^k) = m * (3^k) := by sorry

end a_divisible_by_power_of_three_l1946_194600


namespace sufficient_not_necessary_l1946_194638

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > a) ∧ (∃ a, a ≤ 1 ∧ a^2 > a) := by sorry

end sufficient_not_necessary_l1946_194638


namespace three_digit_divisibility_l1946_194690

theorem three_digit_divisibility (a b c : ℕ) (p : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0)
  (h_p : Nat.Prime p) (h_abc : p ∣ (100 * a + 10 * b + c)) (h_cba : p ∣ (100 * c + 10 * b + a)) :
  p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) := by
  sorry

end three_digit_divisibility_l1946_194690


namespace ad_square_area_l1946_194676

/-- Given two joined right triangles ABC and ACD with squares on their sides -/
structure JoinedTriangles where
  /-- Area of square on side AB -/
  ab_square_area : ℝ
  /-- Area of square on side BC -/
  bc_square_area : ℝ
  /-- Area of square on side CD -/
  cd_square_area : ℝ
  /-- ABC is a right triangle -/
  abc_right : True
  /-- ACD is a right triangle -/
  acd_right : True

/-- The theorem stating the area of the square on AD -/
theorem ad_square_area (t : JoinedTriangles)
  (h1 : t.ab_square_area = 36)
  (h2 : t.bc_square_area = 9)
  (h3 : t.cd_square_area = 16) :
  ∃ (ad_square_area : ℝ), ad_square_area = 61 := by
  sorry

end ad_square_area_l1946_194676


namespace simplify_sqrt_expression_l1946_194652

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l1946_194652


namespace log_problem_l1946_194649

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_problem (x : ℝ) (h : log 3 (5 * x) = 3) : log x 125 = 3/2 :=
by
  sorry

end log_problem_l1946_194649


namespace third_term_coefficient_equals_4860_l1946_194630

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the coefficient of the third term in the expansion of (3a+2b)^6
def third_term_coefficient : ℕ := 
  binomial 6 2 * (3^4) * (2^2)

-- Theorem statement
theorem third_term_coefficient_equals_4860 : third_term_coefficient = 4860 := by
  sorry

end third_term_coefficient_equals_4860_l1946_194630


namespace spherical_coordinates_reflection_l1946_194604

theorem spherical_coordinates_reflection :
  ∀ (x y z : ℝ),
  (∃ (ρ θ φ : ℝ),
    ρ = 4 ∧ θ = 5 * π / 6 ∧ φ = π / 4 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ) →
  (∃ (ρ' θ' φ' : ℝ),
    ρ' = 2 * Real.sqrt 10 ∧ θ' = 5 * π / 6 ∧ φ' = 3 * π / 4 ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by sorry

end spherical_coordinates_reflection_l1946_194604


namespace john_daily_calories_is_3275_l1946_194667

/-- Calculates John's total daily calorie intake based on given meal and shake information. -/
def johnDailyCalories : ℕ :=
  let breakfastCalories : ℕ := 500
  let lunchCalories : ℕ := breakfastCalories + (breakfastCalories / 4)
  let dinnerCalories : ℕ := 2 * lunchCalories
  let shakeCalories : ℕ := 3 * 300
  breakfastCalories + lunchCalories + dinnerCalories + shakeCalories

/-- Theorem stating that John's total daily calorie intake is 3275 calories. -/
theorem john_daily_calories_is_3275 : johnDailyCalories = 3275 := by
  sorry

end john_daily_calories_is_3275_l1946_194667


namespace omega_squared_plus_four_omega_plus_forty_modulus_l1946_194603

theorem omega_squared_plus_four_omega_plus_forty_modulus (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 40) = 2 * Real.sqrt 1885 := by sorry

end omega_squared_plus_four_omega_plus_forty_modulus_l1946_194603


namespace triangle_angle_measure_l1946_194601

theorem triangle_angle_measure (A B : Real) (a b : Real) : 
  0 < A ∧ 0 < B ∧ 0 < a ∧ 0 < b →  -- Ensure positive values
  A = 2 * B →                      -- Condition: A = 2B
  a / b = Real.sqrt 2 →            -- Condition: a:b = √2:1
  A = 90 * (π / 180) :=            -- Conclusion: A = 90° (in radians)
by
  sorry

#check triangle_angle_measure

end triangle_angle_measure_l1946_194601


namespace triple_solution_l1946_194696

theorem triple_solution (a b c : ℝ) :
  a^2 + 2*b^2 - 2*b*c = 16 ∧ 2*a*b - c^2 = 16 →
  (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = -4 ∧ b = -4 ∧ c = -4) := by
sorry

end triple_solution_l1946_194696


namespace candidate_vote_percentage_l1946_194673

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (lost_by : ℕ)
  (h_total : total_votes = 20000)
  (h_lost : lost_by = 16000) :
  (total_votes - lost_by) / total_votes * 100 = 10 :=
by sorry

end candidate_vote_percentage_l1946_194673


namespace parabola_has_one_x_intercept_l1946_194659

/-- The parabola equation: x = -y^2 + 2y + 3 -/
def parabola (x y : ℝ) : Prop := x = -y^2 + 2*y + 3

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola x 0

/-- The parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end parabola_has_one_x_intercept_l1946_194659


namespace no_integer_solution_for_x2_plus_y2_eq_3z2_l1946_194642

theorem no_integer_solution_for_x2_plus_y2_eq_3z2 :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 3 * z^2 := by
  sorry

end no_integer_solution_for_x2_plus_y2_eq_3z2_l1946_194642


namespace f_decreasing_interval_f_max_value_l1946_194672

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem for the decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ (Set.Ioo 0 2), ∀ y ∈ (Set.Ioo 0 2), x < y → f x > f y :=
sorry

-- Theorem for the maximum value on [-4, 3]
theorem f_max_value :
  ∀ x ∈ (Set.Icc (-4) 3), f x ≤ 0 ∧ ∃ y ∈ (Set.Icc (-4) 3), f y = 0 :=
sorry

end f_decreasing_interval_f_max_value_l1946_194672
