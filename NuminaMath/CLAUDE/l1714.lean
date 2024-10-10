import Mathlib

namespace soccer_balls_count_l1714_171422

theorem soccer_balls_count (soccer : ℕ) (baseball : ℕ) (volleyball : ℕ) : 
  baseball = 5 * soccer →
  volleyball = 3 * soccer →
  baseball + volleyball = 160 →
  soccer = 20 :=
by sorry

end soccer_balls_count_l1714_171422


namespace product_of_numbers_l1714_171455

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 8) (h2 : x^2 + y^2 = 160) : x * y = 48 := by
  sorry

end product_of_numbers_l1714_171455


namespace fifth_invoice_number_l1714_171464

/-- Represents the systematic sampling process for invoices -/
def systematicSampling (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * interval

/-- Theorem stating that the fifth sampled invoice number is 215 -/
theorem fifth_invoice_number :
  systematicSampling 15 50 5 = 215 := by
  sorry

end fifth_invoice_number_l1714_171464


namespace work_rate_comparison_l1714_171469

theorem work_rate_comparison (x : ℝ) (work : ℝ) : 
  x > 0 →
  (x + 1) * 21 = x * 28 →
  x = 3 := by
sorry

end work_rate_comparison_l1714_171469


namespace angle_measure_when_supplement_is_four_times_complement_l1714_171406

theorem angle_measure_when_supplement_is_four_times_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end angle_measure_when_supplement_is_four_times_complement_l1714_171406


namespace four_solutions_l1714_171409

/-- S(n) denotes the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n such that n + S(n) + S(S(n)) = 2010 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 4 solutions -/
theorem four_solutions : count_solutions = 4 := by sorry

end four_solutions_l1714_171409


namespace equal_numbers_after_operations_l1714_171425

theorem equal_numbers_after_operations : ∃ (x a b : ℝ), 
  x > 0 ∧ a > 0 ∧ b > 0 ∧
  96 / a = x ∧
  28 - b = x ∧
  20 + b = x ∧
  6 * a = x := by
  sorry

end equal_numbers_after_operations_l1714_171425


namespace jaylen_green_beans_l1714_171441

/-- Prove that Jaylen has 7 green beans given the conditions of the vegetable problem. -/
theorem jaylen_green_beans :
  ∀ (jaylen_carrots jaylen_cucumbers jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans : ℕ),
  jaylen_carrots = 5 →
  jaylen_cucumbers = 2 →
  kristin_bell_peppers = 2 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  kristin_green_beans = 20 →
  jaylen_carrots + jaylen_cucumbers + jaylen_bell_peppers + jaylen_green_beans = 18 →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  jaylen_green_beans = 7 :=
by
  sorry

end jaylen_green_beans_l1714_171441


namespace sequence_properties_l1714_171436

/-- Sequence a_n with sum S_n = n^2 + pn -/
def S (n : ℕ) (p : ℝ) : ℝ := n^2 + p * n

/-- Sequence b_n with sum T_n = 3n^2 - 2n -/
def T (n : ℕ) : ℝ := 3 * n^2 - 2 * n

/-- a_n is the difference of consecutive S_n terms -/
def a (n : ℕ) (p : ℝ) : ℝ := S n p - S (n-1) p

/-- b_n is the difference of consecutive T_n terms -/
def b (n : ℕ) : ℝ := T n - T (n-1)

/-- c_n is the sequence formed by odd-indexed terms of b_n -/
def c (n : ℕ) : ℝ := b (2*n - 1)

theorem sequence_properties (p : ℝ) :
  (a 10 p = b 10) → p = 36 ∧ ∀ n, c n = 12 * n - 11 := by sorry

end sequence_properties_l1714_171436


namespace remainder_1997_pow_2000_mod_7_l1714_171470

theorem remainder_1997_pow_2000_mod_7 : 1997^2000 % 7 = 4 := by sorry

end remainder_1997_pow_2000_mod_7_l1714_171470


namespace min_value_abc_l1714_171457

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 3) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 54 ∧ 
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 3 ∧ 
    a'^2 + 8*a'*b' + 24*b'^2 + 16*b'*c' + 6*c'^2 = 54 :=
by sorry

end min_value_abc_l1714_171457


namespace complement_of_A_in_U_l1714_171446

def U : Set ℕ := {x : ℕ | (x + 1 : ℚ) / (x - 5 : ℚ) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 3} := by sorry

end complement_of_A_in_U_l1714_171446


namespace randy_money_problem_l1714_171479

theorem randy_money_problem (M : ℝ) : 
  M > 0 →
  (1/4 : ℝ) * (M - 10) = 5 →
  M = 30 := by
sorry

end randy_money_problem_l1714_171479


namespace absolute_value_of_z_l1714_171474

theorem absolute_value_of_z (z z₀ : ℂ) : 
  z₀ = 3 + Complex.I ∧ z * z₀ = 3 * z + z₀ → Complex.abs z = Real.sqrt 10 := by
  sorry

end absolute_value_of_z_l1714_171474


namespace max_balls_l1714_171428

theorem max_balls (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 
    (r ≥ 49 ∧ r ≤ 50) ∧ 
    (∀ k : ℕ, k > 0 → (7 * k ≤ r - 49) ∧ (r - 49 < 8 * k)) ∧
    (10 * r ≥ 9 * n)) → 
  n ≤ 210 :=
sorry

end max_balls_l1714_171428


namespace finite_fun_primes_l1714_171439

/-- A prime p is fun with respect to positive integers a and b if there exists a positive integer n
    satisfying the given conditions. -/
def IsFunPrime (p a b : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    p.Prime ∧
    (p ∣ a^(n.factorial) + b) ∧
    (p ∣ a^((n+1).factorial) + b) ∧
    (p < 2*n^2 + 1)

/-- The set of fun primes for given positive integers a and b is finite. -/
theorem finite_fun_primes (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  {p : ℕ | IsFunPrime p a b}.Finite :=
sorry

end finite_fun_primes_l1714_171439


namespace negative_three_below_zero_l1714_171465

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℝ
  unit : String

/-- Defines the concept of opposite meanings for temperatures -/
def oppositeMeaning (t1 t2 : Temperature) : Prop :=
  t1.value = -t2.value ∧ t1.unit = t2.unit

/-- Axiom: If two numbers have opposite meanings, they are respectively called positive and negative -/
axiom positive_negative_opposite (t1 t2 : Temperature) :
  oppositeMeaning t1 t2 → (t1.value > 0 ↔ t2.value < 0)

/-- Given: +10°C represents a temperature of 10°C above zero -/
axiom positive_ten_above_zero :
  ∃ (t : Temperature), t.value = 10 ∧ t.unit = "°C"

/-- Theorem: -3°C represents a temperature of 3°C below zero -/
theorem negative_three_below_zero :
  ∃ (t : Temperature), t.value = -3 ∧ t.unit = "°C" ∧
  ∃ (t_pos : Temperature), oppositeMeaning t t_pos ∧ t_pos.value = 3 :=
sorry

end negative_three_below_zero_l1714_171465


namespace optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l1714_171459

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price -/
def optimal_price : ℝ := 14

/-- Represents the maximum profit -/
def max_profit : ℝ := 360

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  profit_function optimal_price = max_profit :=
sorry

/-- Theorem stating that the profit function is concave down -/
theorem profit_function_concave_down :
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 →
  profit_function (t * x + (1 - t) * y) ≥ t * profit_function x + (1 - t) * profit_function y :=
sorry

end optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l1714_171459


namespace second_caterer_cheaper_l1714_171467

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a caterer given the number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonFee * people

/-- The first caterer's pricing model -/
def caterer1 : CatererCost := { basicFee := 150, perPersonFee := 18 }

/-- The second caterer's pricing model -/
def caterer2 : CatererCost := { basicFee := 250, perPersonFee := 14 }

/-- Theorem stating that for 26 or more people, the second caterer is less expensive -/
theorem second_caterer_cheaper (n : ℕ) (h : n ≥ 26) :
  totalCost caterer2 n < totalCost caterer1 n := by
  sorry


end second_caterer_cheaper_l1714_171467


namespace fourth_term_is_one_l1714_171475

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_is_one :
  let a₁ := (2 : ℝ)^(1/4)
  let a₂ := (2 : ℝ)^(1/6)
  let a₃ := (2 : ℝ)^(1/12)
  let r := a₂ / a₁
  geometric_progression a₁ r 4 = 1 := by
  sorry

end fourth_term_is_one_l1714_171475


namespace complex_power_equality_l1714_171403

theorem complex_power_equality (n : ℕ) (hn : n ≤ 1000) :
  ∀ t : ℝ, (Complex.cos t - Complex.I * Complex.sin t) ^ n = Complex.cos (n * t) - Complex.I * Complex.sin (n * t) := by
  sorry

end complex_power_equality_l1714_171403


namespace store_price_reduction_l1714_171466

theorem store_price_reduction (original_price : ℝ) (h_positive : original_price > 0) :
  let first_reduction := 0.12
  let final_percentage := 0.792
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_percentage / (1 - first_reduction))
  second_reduction = 0.1 := by
sorry

end store_price_reduction_l1714_171466


namespace a_2016_div_2017_l1714_171419

/-- The sequence a defined by the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) + 41 * a n

/-- The theorem stating that the 2016th term of the sequence is divisible by 2017 -/
theorem a_2016_div_2017 : 2017 ∣ a 2016 := by
  sorry

end a_2016_div_2017_l1714_171419


namespace meaningful_fraction_range_l1714_171450

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_range_l1714_171450


namespace valid_seating_count_l1714_171497

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 4

/-- Number of adjacent unoccupied seats -/
def num_adjacent_empty : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_count :
  seating_arrangements num_seats num_people num_adjacent_empty = 336 :=
sorry

end valid_seating_count_l1714_171497


namespace solution_of_linear_equation_l1714_171431

theorem solution_of_linear_equation (x y m : ℝ) 
  (h1 : x = -1)
  (h2 : y = 2)
  (h3 : m * x + 2 * y = 1) :
  m = 3 := by
sorry

end solution_of_linear_equation_l1714_171431


namespace sum_interior_angles_regular_polygon_l1714_171413

/-- The sum of interior angles of a regular polygon with exterior angles of 20 degrees -/
theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n * 20 = 360) : 
  (n - 2) * 180 = 2880 := by
  sorry

end sum_interior_angles_regular_polygon_l1714_171413


namespace intersection_complement_equals_l1714_171449

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 2} := by
  sorry

end intersection_complement_equals_l1714_171449


namespace polynomial_factorization_l1714_171452

theorem polynomial_factorization (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 + 2 * y^3 = 2 * y * (x - y)^2 := by
  sorry

end polynomial_factorization_l1714_171452


namespace triangle_theorem_l1714_171421

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c) :
  (∀ (h2 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C),
    Real.tan t.C = Real.sqrt 2) ∧
  (∀ (h3 : t.a = 2)
     (h4 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 2 / 2)
     (h5 : t.b > t.c),
    t.b = 3 * Real.sqrt 2 / 2 ∧ t.c = Real.sqrt 2 / 2) :=
by sorry

end triangle_theorem_l1714_171421


namespace initial_men_is_50_l1714_171487

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men for a given road project -/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- The theorem stating that for the given project conditions, the initial number of men is 50 -/
theorem initial_men_is_50 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 75)
  : initialMen project = 50 := by
  sorry

end initial_men_is_50_l1714_171487


namespace problem_statement_l1714_171485

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 - a*x
def g (x : ℝ) : ℝ := Real.log x
def h (x : ℝ) : ℝ := f a x + g x

theorem problem_statement :
  (∀ x > 0, f a x ≥ g x) ↔ a ≤ 1 ∧
  ∃ m : ℝ, m = 3/4 - Real.log 2 ∧
    (0 < x₁ ∧ x₁ < 1/2 ∧ 
     h a x₁ - h a x₂ > m ∧
     (∀ m' : ℝ, h a x₁ - h a x₂ > m' → m' ≤ m)) :=
by sorry

end problem_statement_l1714_171485


namespace arithmetic_sequence_common_difference_l1714_171498

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (a₁ d : ℝ) 
  (h_arith : a = arithmeticSequence a₁ d)
  (h_first : a 1 = 5)
  (h_sum : a 6 + a 8 = 58) :
  d = 4 := by
sorry

end arithmetic_sequence_common_difference_l1714_171498


namespace expression_value_l1714_171484

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 :=
by sorry

end expression_value_l1714_171484


namespace total_rainfall_2004_l1714_171453

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total rainfall in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end total_rainfall_2004_l1714_171453


namespace triangle_cos_2C_l1714_171432

theorem triangle_cos_2C (a b : ℝ) (S_ABC : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S_ABC = 12 →
  S_ABC = 1/2 * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 :=
by
  sorry

end triangle_cos_2C_l1714_171432


namespace original_equals_scientific_l1714_171499

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.74
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end original_equals_scientific_l1714_171499


namespace f_properties_l1714_171430

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi) + 2 * Real.sin (x - Real.pi / 2) * Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1 : ℝ) 1 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) Real.pi ∧ f x = y) :=
sorry

end f_properties_l1714_171430


namespace max_quarters_sasha_l1714_171427

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 48 / 10

theorem max_quarters_sasha (q : ℕ) : 
  q * quarter_value + (2 * q) * dime_value ≤ total_amount → 
  q ≤ 10 :=
sorry

end max_quarters_sasha_l1714_171427


namespace squats_calculation_l1714_171461

/-- 
Proves that if the number of squats increases by 5 each day for four consecutive days, 
and 45 squats are performed on the fourth day, then 30 squats were performed on the first day.
-/
theorem squats_calculation (initial_squats : ℕ) : 
  (∀ (day : ℕ), day < 4 → initial_squats + 5 * day = initial_squats + day * 5) →
  initial_squats + 5 * 3 = 45 →
  initial_squats = 30 := by
  sorry

end squats_calculation_l1714_171461


namespace expression_value_l1714_171418

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 3) :
  (x + 2*y)^2 - (x + y)*(2*x - y) = 23 := by
  sorry

end expression_value_l1714_171418


namespace parallel_lines_intersection_l1714_171480

/-- Given 9 parallel lines intersected by n parallel lines forming 1008 parallelograms, n must equal 127 -/
theorem parallel_lines_intersection (n : ℕ) : 
  (9 - 1) * (n - 1) = 1008 → n = 127 := by
  sorry

end parallel_lines_intersection_l1714_171480


namespace eighteen_bottles_needed_l1714_171468

/-- Calculates the minimum number of small bottles needed to fill a large bottle and a vase -/
def minimum_bottles (small_capacity : ℕ) (large_capacity : ℕ) (vase_capacity : ℕ) : ℕ :=
  let large_bottles := large_capacity / small_capacity
  let remaining_for_vase := vase_capacity
  let vase_bottles := (remaining_for_vase + small_capacity - 1) / small_capacity
  large_bottles + vase_bottles

/-- Theorem stating that 18 small bottles are needed to fill the large bottle and vase -/
theorem eighteen_bottles_needed :
  minimum_bottles 45 675 95 = 18 := by
  sorry

end eighteen_bottles_needed_l1714_171468


namespace fib_divisibility_spacing_l1714_171471

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Numbers in Fibonacci sequence divisible by m are equally spaced -/
theorem fib_divisibility_spacing (m : ℕ) (h : m > 0) :
  ∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, m ∣ fib n → m ∣ fib (n + d) :=
sorry

end fib_divisibility_spacing_l1714_171471


namespace exactly_two_girls_together_probability_l1714_171492

/-- The number of boys in the group -/
def num_boys : ℕ := 2

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange all students -/
def total_arrangements : ℕ := Nat.factorial total_students

/-- The number of ways to arrange students with exactly 2 girls together -/
def favorable_arrangements : ℕ := 
  Nat.choose 3 2 * Nat.factorial 2 * Nat.factorial 3

/-- The probability of exactly 2 out of 3 girls standing next to each other -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem exactly_two_girls_together_probability : 
  probability = 3 / 5 := by sorry

end exactly_two_girls_together_probability_l1714_171492


namespace joans_remaining_books_l1714_171444

/-- Calculates the number of remaining books after a sale. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem stating that Joan's remaining books is 7. -/
theorem joans_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end joans_remaining_books_l1714_171444


namespace BA_equals_AB_l1714_171420

variable {α : Type*} [CommRing α]

def matrix_eq (A B : Matrix (Fin 2) (Fin 2) α) : Prop :=
  ∀ i j, A i j = B i j

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A + B = A * B)
  (h2 : matrix_eq (A * B) !![5, 2; -2, 4]) :
  matrix_eq (B * A) !![5, 2; -2, 4] := by
  sorry

end BA_equals_AB_l1714_171420


namespace sum_of_digits_theorem_l1714_171447

-- Define the set A
def A : Set Int := {m | ∃ p q : Int, p > 0 ∧ q > 0 ∧ p * q = 2020 ∧ p + q = -m}

-- Define the set B
def B : Set Int := {n | ∃ r s : Int, r > 0 ∧ s > 0 ∧ r * s = n ∧ r + s = 2020}

-- Define a function to calculate the sum of digits
def sumOfDigits (n : Int) : Nat :=
  (n.natAbs.digits 10).sum

-- State the theorem
theorem sum_of_digits_theorem :
  ∃ a b : Int, a ∈ A ∧ b ∈ B ∧ (∀ m ∈ A, m ≤ a) ∧ (∀ n ∈ B, b ≤ n) ∧
  sumOfDigits (a + b) = 27 := by sorry

end sum_of_digits_theorem_l1714_171447


namespace dihedral_angle_cosine_l1714_171440

/-- Regular triangular pyramid with given properties -/
structure RegularPyramid where
  base_side : ℝ
  lateral_edge : ℝ
  base_side_eq_one : base_side = 1
  lateral_edge_eq_two : lateral_edge = 2

/-- Section that divides the pyramid volume equally -/
structure EqualVolumeSection (p : RegularPyramid) where
  passes_through_AB : Bool
  divides_equally : Bool

/-- Dihedral angle between the section and the base -/
def dihedralAngle (p : RegularPyramid) (s : EqualVolumeSection p) : ℝ := sorry

theorem dihedral_angle_cosine 
  (p : RegularPyramid) 
  (s : EqualVolumeSection p) : 
  Real.cos (dihedralAngle p s) = 2 * Real.sqrt 15 / 15 := by
  sorry

end dihedral_angle_cosine_l1714_171440


namespace school_population_l1714_171437

theorem school_population (boys : ℕ) (girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = 128 → 
  boys + girls = 208 := by
sorry

end school_population_l1714_171437


namespace no_integer_solution_a_l1714_171456

theorem no_integer_solution_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := by
  sorry

#check no_integer_solution_a

end no_integer_solution_a_l1714_171456


namespace expression_evaluation_l1714_171493

theorem expression_evaluation (c k : ℕ) (h1 : c = 4) (h2 : k = 2) :
  ((c^c - c*(c-1)^c + k)^c : ℕ) = 18974736 := by
  sorry

end expression_evaluation_l1714_171493


namespace distance_to_school_l1714_171404

theorem distance_to_school (normal_time normal_speed light_time : ℚ) 
  (h1 : normal_time = 20 / 60)
  (h2 : light_time = 10 / 60)
  (h3 : normal_time * normal_speed = light_time * (normal_speed + 15)) :
  normal_time * normal_speed = 5 := by
  sorry

end distance_to_school_l1714_171404


namespace action_figure_collection_l1714_171411

/-- The problem of calculating the total number of action figures needed for a complete collection. -/
theorem action_figure_collection
  (jerry_has : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_finish : ℕ)
  (h1 : jerry_has = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_finish = 72) :
  jerry_has + total_cost_to_finish / cost_per_figure = 16 :=
by sorry

end action_figure_collection_l1714_171411


namespace deductive_reasoning_correctness_l1714_171412

/-- Represents a deductive reasoning process -/
structure DeductiveReasoning where
  premise : Prop
  form : Prop
  conclusion : Prop

/-- Represents the correctness of a component in the reasoning process -/
def isCorrect (p : Prop) : Prop := p

theorem deductive_reasoning_correctness 
  (dr : DeductiveReasoning) 
  (h_premise : isCorrect dr.premise) 
  (h_form : isCorrect dr.form) : 
  isCorrect dr.conclusion :=
sorry

end deductive_reasoning_correctness_l1714_171412


namespace fermat_number_properties_l1714_171454

/-- Fermat number F_n -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Main theorem -/
theorem fermat_number_properties (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_factor : p ∣ F n) :
  (∃ x : ℤ, x^2 ≡ 2 [ZMOD p]) ∧ p ≡ 1 [ZMOD 2^(n+2)] := by sorry

end fermat_number_properties_l1714_171454


namespace union_of_M_and_N_l1714_171415

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < 1} := by
  sorry

end union_of_M_and_N_l1714_171415


namespace min_cuts_correct_l1714_171417

/-- The minimum number of cuts required to divide a cube of edge length 4 into 64 unit cubes -/
def min_cuts : ℕ := 6

/-- The edge length of the initial cube -/
def initial_edge_length : ℕ := 4

/-- The number of smaller cubes we want to create -/
def target_num_cubes : ℕ := 64

/-- The edge length of the smaller cubes -/
def target_edge_length : ℕ := 1

/-- Theorem stating that min_cuts is the minimum number of cuts required -/
theorem min_cuts_correct :
  (2 ^ min_cuts = target_num_cubes) ∧
  (∀ n : ℕ, n < min_cuts → 2 ^ n < target_num_cubes) :=
sorry

end min_cuts_correct_l1714_171417


namespace donut_problem_l1714_171416

theorem donut_problem (D : ℕ) : (D - 6) / 2 = 22 ↔ D = 50 := by
  sorry

end donut_problem_l1714_171416


namespace brothers_ages_product_l1714_171414

theorem brothers_ages_product (O Y : ℕ) 
  (h1 : O > Y)
  (h2 : O - Y = 12)
  (h3 : O + Y = (O - Y) + 40) : 
  O * Y = 640 := by
  sorry

end brothers_ages_product_l1714_171414


namespace increasing_sequence_condition_sufficient_condition_not_necessary_condition_l1714_171494

theorem increasing_sequence_condition (k : ℝ) : 
  (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) ↔ k > -3 :=
by sorry

theorem sufficient_condition (k : ℝ) :
  k ≥ -2 → ∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1)) :=
by sorry

theorem not_necessary_condition :
  ∃ k : ℝ, k < -2 ∧ (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) :=
by sorry

end increasing_sequence_condition_sufficient_condition_not_necessary_condition_l1714_171494


namespace consecutive_odd_numbers_multiple_l1714_171473

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_multiple (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- Three odd numbers
  b = a + 2 ∧ c = b + 2 ∧           -- Consecutive
  a = 7 ∧                           -- First number is 7
  ∃ m : ℤ, 8 * a = 3 * c + 5 + m * b -- Equation condition
  → m = 2 :=                        -- Multiple of second number is 2
by sorry

end consecutive_odd_numbers_multiple_l1714_171473


namespace counterexample_37_l1714_171442

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_37 : 
  is_prime 37 ∧ ¬(is_prime (37 - 2) ∨ is_prime (37 + 2)) :=
by sorry

end counterexample_37_l1714_171442


namespace bc_plus_ce_is_one_third_of_ad_l1714_171445

-- Define the points and lengths
variable (A B C D E : ℝ)
variable (AB AC AE BD CD ED BC CE AD : ℝ)

-- State the conditions
variable (h1 : B < C)
variable (h2 : C < E)
variable (h3 : E < D)
variable (h4 : AB = 3 * BD)
variable (h5 : AC = 7 * CD)
variable (h6 : AE = 5 * ED)
variable (h7 : AD = AB + BD + CD + ED)
variable (h8 : BC = AC - AB)
variable (h9 : CE = AE - AC)

-- State the theorem
theorem bc_plus_ce_is_one_third_of_ad :
  (BC + CE) / AD = 1 / 3 := by sorry

end bc_plus_ce_is_one_third_of_ad_l1714_171445


namespace math_club_teams_l1714_171407

theorem math_club_teams (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 2) = 90 := by
sorry

end math_club_teams_l1714_171407


namespace smallest_ambiguous_weight_correct_l1714_171477

/-- The smallest total weight of kittens for which the number of kittens is not uniquely determined -/
def smallest_ambiguous_weight : ℕ := 480

/-- The total weight of the two lightest kittens -/
def lightest_two_weight : ℕ := 80

/-- The total weight of the four heaviest kittens -/
def heaviest_four_weight : ℕ := 200

/-- Predicate to check if a given total weight allows for a unique determination of the number of kittens -/
def is_uniquely_determined (total_weight : ℕ) : Prop :=
  ∀ n m : ℕ, 
    (n ≠ m) → 
    (∃ (weights_n weights_m : List ℕ),
      (weights_n.length = n ∧ weights_m.length = m) ∧
      (weights_n.sum = total_weight ∧ weights_m.sum = total_weight) ∧
      (weights_n.take 2).sum = lightest_two_weight ∧
      (weights_m.take 2).sum = lightest_two_weight ∧
      (weights_n.reverse.take 4).sum = heaviest_four_weight ∧
      (weights_m.reverse.take 4).sum = heaviest_four_weight) →
    False

theorem smallest_ambiguous_weight_correct :
  (∀ w : ℕ, w < smallest_ambiguous_weight → is_uniquely_determined w) ∧
  ¬is_uniquely_determined smallest_ambiguous_weight :=
sorry

end smallest_ambiguous_weight_correct_l1714_171477


namespace equation_equivalence_l1714_171401

theorem equation_equivalence (a c x y : ℝ) (s t u : ℤ) : 
  (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) →
  ((a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5) →
  s * t * u = 18 := by
sorry

end equation_equivalence_l1714_171401


namespace present_age_ratio_l1714_171402

/-- Given Suji's present age and the future ratio of ages, find the present ratio of ages --/
theorem present_age_ratio (suji_age : ℕ) (future_ratio_abi : ℕ) (future_ratio_suji : ℕ) :
  suji_age = 24 →
  (future_ratio_abi : ℚ) / future_ratio_suji = 11 / 9 →
  ∃ (abi_age : ℕ),
    (abi_age + 3 : ℚ) / (suji_age + 3) = (future_ratio_abi : ℚ) / future_ratio_suji ∧
    (abi_age : ℚ) / suji_age = 5 / 4 :=
by sorry

end present_age_ratio_l1714_171402


namespace monkey_family_size_l1714_171426

/-- The number of monkeys in a family that collected bananas -/
def number_of_monkeys : ℕ := by sorry

theorem monkey_family_size :
  let total_piles : ℕ := 10
  let piles_type1 : ℕ := 6
  let hands_per_pile_type1 : ℕ := 9
  let bananas_per_hand_type1 : ℕ := 14
  let piles_type2 : ℕ := total_piles - piles_type1
  let hands_per_pile_type2 : ℕ := 12
  let bananas_per_hand_type2 : ℕ := 9
  let bananas_per_monkey : ℕ := 99

  let total_bananas : ℕ := 
    piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2

  number_of_monkeys = total_bananas / bananas_per_monkey := by sorry

end monkey_family_size_l1714_171426


namespace gcf_of_120_180_240_l1714_171496

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end gcf_of_120_180_240_l1714_171496


namespace min_value_of_sum_of_squares_l1714_171489

theorem min_value_of_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) ≥ 3 ∧
  ((a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) = 3 ↔ a = b ∧ b = c) :=
by sorry

end min_value_of_sum_of_squares_l1714_171489


namespace constant_term_binomial_expansion_l1714_171488

/-- The constant term in the expansion of (2/x + x)^4 is 24 -/
theorem constant_term_binomial_expansion :
  let n : ℕ := 4
  let a : ℚ := 2
  let b : ℚ := 1
  (Nat.choose n (n / 2)) * a^(n / 2) * b^(n / 2) = 24 := by
  sorry

end constant_term_binomial_expansion_l1714_171488


namespace complex_fraction_sum_l1714_171472

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2 * Complex.I) / (1 + Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end complex_fraction_sum_l1714_171472


namespace acute_angle_vector_range_l1714_171435

/-- The range of k for acute angle between vectors (2, 1) and (1, k) -/
theorem acute_angle_vector_range :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, k)
  -- Acute angle condition
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  -- Non-parallel condition
  (a.1 / a.2 ≠ b.1 / b.2) →
  -- Range of k
  (k > -2 ∧ k ≠ 1/2) :=
by sorry

end acute_angle_vector_range_l1714_171435


namespace generalized_distributive_laws_l1714_171495

variable {α : Type*}
variable {I : Type*}
variable (𝔍 : I → Type*)
variable (A : (i : I) → 𝔍 i → Set α)

def paths (𝔍 : I → Type*) := (i : I) → 𝔍 i

theorem generalized_distributive_laws :
  (⋃ i, ⋂ j, A i j) = (⋂ f : paths 𝔍, ⋃ i, A i (f i)) ∧
  (⋂ i, ⋃ j, A i j) = (⋃ f : paths 𝔍, ⋂ i, A i (f i)) :=
sorry

end generalized_distributive_laws_l1714_171495


namespace yellow_highlighters_count_l1714_171482

theorem yellow_highlighters_count (yellow pink blue : ℕ) : 
  pink = yellow + 7 →
  blue = pink + 5 →
  yellow + pink + blue = 40 →
  yellow = 7 := by
  sorry

end yellow_highlighters_count_l1714_171482


namespace game_points_proof_l1714_171490

def points_earned (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem game_points_proof :
  points_earned 7 8 2 = 40 := by
  sorry

end game_points_proof_l1714_171490


namespace infinitely_many_n_satisfying_conditions_l1714_171433

theorem infinitely_many_n_satisfying_conditions :
  ∀ k : ℕ, k > 0 →
  let n := k * (k + 1)
  ∃ m : ℕ, m^2 < n ∧ n < (m + 1)^2 ∧ n % m = 0 :=
by
  sorry

end infinitely_many_n_satisfying_conditions_l1714_171433


namespace square_root_sum_l1714_171443

theorem square_root_sum (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end square_root_sum_l1714_171443


namespace mia_fruit_probability_l1714_171408

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals

/-- The probability of eating at least two different kinds of fruit -/
def prob_different_fruits : ℚ := 1 - (num_fruit_types * prob_same_fruit)

theorem mia_fruit_probability :
  prob_different_fruits = 63 / 64 :=
sorry

end mia_fruit_probability_l1714_171408


namespace not_perfect_square_floor_sqrt_l1714_171429

theorem not_perfect_square_floor_sqrt (A : ℕ) (h : ∀ k : ℕ, k * k ≠ A) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ :=
sorry

end not_perfect_square_floor_sqrt_l1714_171429


namespace sqrt_product_equality_l1714_171460

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1714_171460


namespace selection_problem_l1714_171424

theorem selection_problem (n_boys m_boys n_girls m_girls : ℕ) 
  (h1 : n_boys = 5) (h2 : m_boys = 3) (h3 : n_girls = 4) (h4 : m_girls = 2) : 
  (Nat.choose n_boys m_boys) * (Nat.choose n_girls m_girls) = 
  (Nat.choose 5 3) * (Nat.choose 4 2) := by
  sorry

end selection_problem_l1714_171424


namespace trioball_playing_time_l1714_171483

theorem trioball_playing_time (num_children : ℕ) (game_duration : ℕ) (players_per_game : ℕ) :
  num_children = 3 →
  game_duration = 120 →
  players_per_game = 2 →
  ∃ (individual_time : ℕ),
    individual_time * num_children = players_per_game * game_duration ∧
    individual_time = 80 := by
  sorry

end trioball_playing_time_l1714_171483


namespace handshakes_in_gathering_l1714_171434

/-- The number of handshakes in a gathering of couples with specific rules -/
theorem handshakes_in_gathering (n : ℕ) (h : n = 6) : 
  (2 * n) * (2 * n - 3) / 2 = 54 := by sorry

end handshakes_in_gathering_l1714_171434


namespace black_squares_count_l1714_171463

/-- Represents a checkerboard with side length n -/
structure Checkerboard (n : ℕ) where
  is_corner_black : Bool
  is_alternating : Bool

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard 33) : ℕ :=
  sorry

/-- Theorem: The number of black squares on a 33x33 alternating checkerboard with black corners is 545 -/
theorem black_squares_count : 
  ∀ (board : Checkerboard 33), 
  board.is_corner_black = true → 
  board.is_alternating = true → 
  count_black_squares board = 545 := by
  sorry

end black_squares_count_l1714_171463


namespace odd_integer_divisor_form_l1714_171458

theorem odd_integer_divisor_form (n : ℕ) (hn : Odd n) (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) (heq : (1 : ℚ) / x + (1 : ℚ) / y = (4 : ℚ) / n) :
  ∃ (k : ℕ), ∃ (d : ℕ), d ∣ n ∧ d = 4 * k - 1 := by
  sorry

end odd_integer_divisor_form_l1714_171458


namespace greatest_power_of_8_dividing_20_factorial_l1714_171491

theorem greatest_power_of_8_dividing_20_factorial :
  (∃ n : ℕ+, 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) ∧
  (∃ n : ℕ+, n.val = 6 ∧ 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) :=
by sorry

end greatest_power_of_8_dividing_20_factorial_l1714_171491


namespace more_ones_than_zeros_mod_500_l1714_171451

/-- The number of positive integers less than or equal to 500 whose binary 
    representation contains more 1's than 0's -/
def N : ℕ := sorry

/-- Function to count 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Function to count 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem more_ones_than_zeros_mod_500 :
  N % 500 = 305 :=
sorry

end more_ones_than_zeros_mod_500_l1714_171451


namespace xy_value_l1714_171438

theorem xy_value (x y : ℝ) : |x - y + 6| + (y + 8)^2 = 0 → x * y = 112 := by
  sorry

end xy_value_l1714_171438


namespace money_sharing_problem_l1714_171486

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents the money distribution among three people -/
structure MoneyDistribution :=
  (amanda : ℕ) (ben : ℕ) (carlos : ℕ)

/-- Theorem stating that given a money ratio of 2:3:8 and Amanda's share of $30, 
    the total amount shared is $195 -/
theorem money_sharing_problem 
  (ratio : MoneyRatio) 
  (dist : MoneyDistribution) :
  ratio.a = 2 ∧ ratio.b = 3 ∧ ratio.c = 8 ∧ 
  dist.amanda = 30 ∧
  dist.amanda * ratio.b = dist.ben * ratio.a ∧
  dist.amanda * ratio.c = dist.carlos * ratio.a →
  dist.amanda + dist.ben + dist.carlos = 195 :=
by sorry

end money_sharing_problem_l1714_171486


namespace g_increasing_g_geq_h_condition_l1714_171476

noncomputable section

-- Define the functions g and h
def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 5 * Real.log x
def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- Theorem 1: g(x) is increasing when a > 5/2
theorem g_increasing (a : ℝ) : 
  (∀ x > 0, ∀ y > 0, x < y → g a x < g a y) ↔ a > 5/2 :=
sorry

-- Theorem 2: Condition for g(x₁) ≥ h(x₂) when a = 2
theorem g_geq_h_condition (m : ℝ) :
  (∃ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Icc 1 2, g 2 x₁ ≥ h m x₂) ↔ 
  m ≥ 8 - 5 * Real.log 2 :=
sorry

end g_increasing_g_geq_h_condition_l1714_171476


namespace triangle_problem_l1714_171410

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = Real.pi / 3 ∧ t.c = 3 := by
  sorry


end triangle_problem_l1714_171410


namespace guilty_cases_count_l1714_171462

theorem guilty_cases_count (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 17 →
  dismissed = 2 →
  delayed = 1 →
  (total - dismissed - delayed - (2 * (total - dismissed) / 3)) = 4 := by
sorry

end guilty_cases_count_l1714_171462


namespace original_number_proof_l1714_171400

theorem original_number_proof (x : ℝ) : x * 1.5 = 135 → x = 90 := by
  sorry

end original_number_proof_l1714_171400


namespace one_more_bird_than_storks_l1714_171405

/-- Given a fence with birds and storks, calculate the difference between the number of birds and storks -/
def bird_stork_difference (num_birds : ℕ) (num_storks : ℕ) : ℤ :=
  (num_birds : ℤ) - (num_storks : ℤ)

/-- Theorem: On a fence with 6 birds and 5 storks, there is 1 more bird than storks -/
theorem one_more_bird_than_storks :
  bird_stork_difference 6 5 = 1 := by
  sorry

end one_more_bird_than_storks_l1714_171405


namespace smallest_non_factor_product_l1714_171481

/-- The set of factors of 48 -/
def factors_of_48 : Set ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

/-- Proposition: The smallest product of two distinct factors of 48 that is not a factor of 48 is 18 -/
theorem smallest_non_factor_product :
  ∃ (x y : ℕ), x ∈ factors_of_48 ∧ y ∈ factors_of_48 ∧ x ≠ y ∧ x * y ∉ factors_of_48 ∧
  x * y = 18 ∧ ∀ (a b : ℕ), a ∈ factors_of_48 → b ∈ factors_of_48 → a ≠ b →
  a * b ∉ factors_of_48 → a * b ≥ 18 := by
  sorry

end smallest_non_factor_product_l1714_171481


namespace cubic_polynomial_root_inequality_l1714_171423

theorem cubic_polynomial_root_inequality (A B C : ℝ) (α β γ : ℂ) 
  (h : ∀ x : ℂ, x^3 + A*x^2 + B*x + C = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  (1 + |A| + |B| + |C|) / (Complex.abs α + Complex.abs β + Complex.abs γ) ≥ 1 := by
  sorry

end cubic_polynomial_root_inequality_l1714_171423


namespace parallelogram_area_l1714_171448

/-- The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 32 → 
  height = 18 → 
  area = base * height → 
  area = 576 := by sorry

end parallelogram_area_l1714_171448


namespace problem_solution_l1714_171478

theorem problem_solution (x : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → (10/100) * x = 17.85 := by
  sorry

end problem_solution_l1714_171478
