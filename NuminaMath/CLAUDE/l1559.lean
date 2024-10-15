import Mathlib

namespace NUMINAMATH_CALUDE_max_gold_coins_theorem_l1559_155929

/-- Represents a toy that can be created -/
structure Toy where
  planks : ℕ
  value : ℕ

/-- Calculates the maximum gold coins that can be earned given a number of planks and a list of toys -/
def maxGoldCoins (totalPlanks : ℕ) (toys : List Toy) : ℕ :=
  sorry

/-- The theorem stating the maximum gold coins that can be earned -/
theorem max_gold_coins_theorem :
  let windmill : Toy := ⟨5, 6⟩
  let steamboat : Toy := ⟨7, 8⟩
  let airplane : Toy := ⟨14, 19⟩
  let toys : List Toy := [windmill, steamboat, airplane]
  maxGoldCoins 130 toys = 172 := by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_theorem_l1559_155929


namespace NUMINAMATH_CALUDE_refrigerator_profit_theorem_l1559_155994

/-- Represents the financial details of a refrigerator sale --/
structure RefrigeratorSale where
  costPrice : ℝ
  markedPrice : ℝ
  discountPercentage : ℝ

/-- Calculates the profit from a refrigerator sale --/
def calculateProfit (sale : RefrigeratorSale) : ℝ :=
  sale.markedPrice * (1 - sale.discountPercentage) - sale.costPrice

/-- Theorem stating the profit for a specific refrigerator sale scenario --/
theorem refrigerator_profit_theorem (sale : RefrigeratorSale) 
  (h1 : sale.costPrice = 2000)
  (h2 : sale.markedPrice = 2750)
  (h3 : sale.discountPercentage = 0.15) :
  calculateProfit sale = 337.5 := by
  sorry

#eval calculateProfit { costPrice := 2000, markedPrice := 2750, discountPercentage := 0.15 }

end NUMINAMATH_CALUDE_refrigerator_profit_theorem_l1559_155994


namespace NUMINAMATH_CALUDE_euler_formula_second_quadrant_l1559_155972

theorem euler_formula_second_quadrant :
  let z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_euler_formula_second_quadrant_l1559_155972


namespace NUMINAMATH_CALUDE_calculation_proof_l1559_155998

theorem calculation_proof : |-4| + (1/3)⁻¹ - (Real.sqrt 2)^2 + 2035^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1559_155998


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1559_155947

theorem quadratic_inequality_problem (m n : ℝ) (h1 : ∀ x : ℝ, x^2 - 3*x + m < 0 ↔ 1 < x ∧ x < n) :
  m = 2 ∧ n = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → m*a + 2*n*b = 3 → a*b ≤ 9/32) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ m*a + 2*n*b = 3 ∧ a*b = 9/32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1559_155947


namespace NUMINAMATH_CALUDE_payment_plan_difference_l1559_155927

def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def num_monthly_payments : ℕ := 24
def monthly_payment : ℕ := 65

theorem payment_plan_difference :
  (down_payment + num_monthly_payments * monthly_payment) - purchase_price = 260 := by
  sorry

end NUMINAMATH_CALUDE_payment_plan_difference_l1559_155927


namespace NUMINAMATH_CALUDE_bruce_triple_age_in_six_years_l1559_155944

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple_age (bruce_age : ℕ) (son_age : ℕ) : ℕ :=
  let x : ℕ := (bruce_age - 3 * son_age) / 2
  x

/-- Theorem stating that it will take 6 years for Bruce to be three times as old as his son -/
theorem bruce_triple_age_in_six_years :
  years_until_triple_age 36 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_triple_age_in_six_years_l1559_155944


namespace NUMINAMATH_CALUDE_next_number_with_property_l1559_155931

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  ∀ n : ℕ, n > 1818 →
  (∀ m : ℕ, 1818 < m ∧ m < n → ¬has_property m) →
  has_property n →
  n = 1832 :=
sorry

end NUMINAMATH_CALUDE_next_number_with_property_l1559_155931


namespace NUMINAMATH_CALUDE_sin_2alpha_over_cos_squared_l1559_155954

theorem sin_2alpha_over_cos_squared (α : Real) 
  (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_over_cos_squared_l1559_155954


namespace NUMINAMATH_CALUDE_range_of_m_l1559_155957

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 9*a + b = a*b)
  (h : ∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) :
  ∃ m₀ : ℝ, m₀ = 3 ∧ ∀ m : ℝ, (∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) → m ≥ m₀ :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1559_155957


namespace NUMINAMATH_CALUDE_math_team_selection_l1559_155938

theorem math_team_selection (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l1559_155938


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1559_155956

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  totalMembers : ℕ
  averageAge : ℝ
  captainAgeDiff : ℝ
  remainingAverageAgeDiff : ℝ

/-- Theorem stating that the average age of the cricket team is 30 years -/
theorem cricket_team_average_age
  (team : CricketTeam)
  (h1 : team.totalMembers = 20)
  (h2 : team.averageAge = 30)
  (h3 : team.captainAgeDiff = 5)
  (h4 : team.remainingAverageAgeDiff = 3)
  : team.averageAge = 30 := by
  sorry

#check cricket_team_average_age

end NUMINAMATH_CALUDE_cricket_team_average_age_l1559_155956


namespace NUMINAMATH_CALUDE_total_pencils_l1559_155912

/-- Calculate the total number of pencils Asaf and Alexander have together -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) :
  asaf_age + alexander_age = 140 →
  asaf_age = 50 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1559_155912


namespace NUMINAMATH_CALUDE_second_sunday_on_13th_l1559_155968

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  /-- The day of the week on which the month starts -/
  startDay : DayOfWeek
  /-- The number of days in the month -/
  numDays : Nat
  /-- Predicate that is true if three Wednesdays fall on even dates -/
  threeWednesdaysOnEvenDates : Prop

/-- Given a month and a day number, returns the day of the week -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Predicate that is true if the given day is a Sunday -/
def isSunday (dow : DayOfWeek) : Prop :=
  sorry

/-- Returns the date of the nth occurrence of a specific day in the month -/
def nthOccurrence (m : Month) (dow : DayOfWeek) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that in a month where three Wednesdays fall on even dates, 
    the second Sunday of that month falls on the 13th -/
theorem second_sunday_on_13th (m : Month) :
  m.threeWednesdaysOnEvenDates → nthOccurrence m DayOfWeek.Sunday 2 = 13 :=
sorry

end NUMINAMATH_CALUDE_second_sunday_on_13th_l1559_155968


namespace NUMINAMATH_CALUDE_percent_democrat_voters_l1559_155993

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.75 * D + 0.30 * R = 57 →
  D = 60 := by
sorry

end NUMINAMATH_CALUDE_percent_democrat_voters_l1559_155993


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_47_l1559_155989

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  sum_of_digits n = 47

theorem smallest_number_with_digit_sum_47 :
  ∀ n : ℕ, is_valid_number n → n ≥ 299999 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_47_l1559_155989


namespace NUMINAMATH_CALUDE_flag_designs_count_l1559_155949

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of different flag designs possible -/
def num_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of different flag designs is 27 -/
theorem flag_designs_count : num_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l1559_155949


namespace NUMINAMATH_CALUDE_second_stock_percentage_l1559_155911

/-- Prove that the percentage of the second stock is 15% given the investment conditions --/
theorem second_stock_percentage
  (total_investment : ℚ)
  (first_stock_percentage : ℚ)
  (first_stock_face_value : ℚ)
  (second_stock_face_value : ℚ)
  (total_dividend : ℚ)
  (first_stock_investment : ℚ)
  (h1 : total_investment = 12000)
  (h2 : first_stock_percentage = 12 / 100)
  (h3 : first_stock_face_value = 120)
  (h4 : second_stock_face_value = 125)
  (h5 : total_dividend = 1360)
  (h6 : first_stock_investment = 4000.000000000002)
  : (total_dividend - (first_stock_investment / first_stock_face_value * first_stock_percentage)) /
    ((total_investment - first_stock_investment) / second_stock_face_value) = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_stock_percentage_l1559_155911


namespace NUMINAMATH_CALUDE_amount_with_r_l1559_155970

/-- Given three people (p, q, r) with a total amount of 6000 among them,
    where r has two-thirds of the total amount that p and q have together,
    prove that the amount with r is 2400. -/
theorem amount_with_r (total : ℕ) (amount_r : ℕ) : 
  total = 6000 →
  amount_r = (2 / 3 : ℚ) * (total - amount_r) →
  amount_r = 2400 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l1559_155970


namespace NUMINAMATH_CALUDE_cabbage_sales_theorem_l1559_155969

/-- Calculates the total kilograms of cabbage sold given the price per kilogram and earnings from three days -/
def total_cabbage_sold (price_per_kg : ℚ) (day1_earnings day2_earnings day3_earnings : ℚ) : ℚ :=
  (day1_earnings + day2_earnings + day3_earnings) / price_per_kg

/-- Theorem stating that given the specific conditions, the total cabbage sold is 48 kg -/
theorem cabbage_sales_theorem :
  let price_per_kg : ℚ := 2
  let day1_earnings : ℚ := 30
  let day2_earnings : ℚ := 24
  let day3_earnings : ℚ := 42
  total_cabbage_sold price_per_kg day1_earnings day2_earnings day3_earnings = 48 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_sales_theorem_l1559_155969


namespace NUMINAMATH_CALUDE_inequality_solution_l1559_155935

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1559_155935


namespace NUMINAMATH_CALUDE_expression_evaluation_l1559_155984

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1559_155984


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1559_155991

theorem quadratic_inequality_solution_set (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1559_155991


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_parallel_l1559_155992

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem planes_parallel_from_skew_lines_parallel 
  (h_distinct : α ≠ β)
  (h_different : a ≠ b)
  (h_skew : skew_lines a b)
  (h_a_alpha : parallel_line_plane a α)
  (h_b_alpha : parallel_line_plane b α)
  (h_a_beta : parallel_line_plane a β)
  (h_b_beta : parallel_line_plane b β) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_parallel_l1559_155992


namespace NUMINAMATH_CALUDE_two_thousand_two_in_sequence_l1559_155914

def next_in_sequence (b : ℕ) : ℕ :=
  b + (Nat.factors b).reverse.head!

def is_in_sequence (a : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, Nat.iterate next_in_sequence k a = n

theorem two_thousand_two_in_sequence (a : ℕ) :
  a > 1 → (is_in_sequence a 2002 ↔ a = 1859 ∨ a = 1991) :=
sorry

end NUMINAMATH_CALUDE_two_thousand_two_in_sequence_l1559_155914


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1559_155900

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 : ℚ) - (sum_arithmetic_sequence a₁ d 10 / 10 : ℚ) = 2 →
  sum_arithmetic_sequence a₁ d 2018 = -2018 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1559_155900


namespace NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l1559_155940

-- Define the property of being a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the property of being an increasing arithmetic sequence
def isIncreasingArithmeticSequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ a < b ∧ b < c ∧ c < d

-- Define the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_first_prime_in_sequence (p q r s : ℕ) :
  isPrime p → isPrime q → isPrime r → isPrime s →
  isIncreasingArithmeticSequence p q r s →
  q - p = 4 →
  p > 5 →
  onesDigit p = 9 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l1559_155940


namespace NUMINAMATH_CALUDE_percentage_difference_l1559_155964

theorem percentage_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * (x - y) / y :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1559_155964


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1559_155971

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1559_155971


namespace NUMINAMATH_CALUDE_set_equality_l1559_155917

theorem set_equality : 
  {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1559_155917


namespace NUMINAMATH_CALUDE_tan_graph_property_l1559_155990

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * Real.pi / 5))) →
  a * Real.tan (b * Real.pi / 10) = 1 →
  a * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_property_l1559_155990


namespace NUMINAMATH_CALUDE_chris_age_l1559_155901

/-- The ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.amy + ages.ben + ages.chris) / 3 = 10 ∧
  -- Five years ago, Chris was twice Amy's age
  ages.chris - 5 = 2 * (ages.amy - 5) ∧
  -- In 5 years, Ben's age will be half of Amy's age
  ages.ben + 5 = (ages.amy + 5) / 2

/-- The theorem to prove -/
theorem chris_age (ages : Ages) (h : satisfies_conditions ages) : 
  ∃ (ε : ℝ), ages.chris = 16 + ε ∧ abs ε < 1 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l1559_155901


namespace NUMINAMATH_CALUDE_find_xy_l1559_155930

/-- Define the ⊕ operation for pairs of real numbers -/
def oplus (a b c d : ℝ) : ℝ × ℝ := (a + c, b * d)

/-- Theorem statement -/
theorem find_xy : ∃ (x y : ℝ), oplus x 1 2 y = (4, 2) ∧ (x, y) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_find_xy_l1559_155930


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1559_155936

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1559_155936


namespace NUMINAMATH_CALUDE_sine_of_angle_through_point_l1559_155967

theorem sine_of_angle_through_point (α : Real) :
  let P : Real × Real := (Real.cos (3 * Real.pi / 4), Real.sin (3 * Real.pi / 4))
  (∃ k : Real, k > 0 ∧ (k * Real.cos α = P.1) ∧ (k * Real.sin α = P.2)) →
  Real.sin α = Real.sqrt 2 / 2 ∨ Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_through_point_l1559_155967


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l1559_155932

/-- A quadratic function f(x) = x^2 - 2x + m passing through three specific points -/
def QuadraticThroughPoints (m : ℝ) (y₁ y₂ y₃ : ℝ) : Prop :=
  let f := fun x => x^2 - 2*x + m
  f (-1) = y₁ ∧ f 2 = y₂ ∧ f 3 = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_point_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
    (h : QuadraticThroughPoints m y₁ y₂ y₃) : 
    y₂ < y₁ ∧ y₁ = y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l1559_155932


namespace NUMINAMATH_CALUDE_equation_solution_l1559_155904

theorem equation_solution (x : ℝ) : (40 / 80 = Real.sqrt (x / 80)) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1559_155904


namespace NUMINAMATH_CALUDE_partner_q_active_months_l1559_155962

/-- Represents the investment and activity of a partner in the business -/
structure Partner where
  investment : ℝ
  monthlyReturn : ℝ
  activeMonths : ℕ

/-- Represents the business venture with three partners -/
structure Business where
  p : Partner
  q : Partner
  r : Partner
  totalProfit : ℝ

/-- The main theorem stating that partner Q was active for 6 months -/
theorem partner_q_active_months (b : Business) : b.q.activeMonths = 6 :=
  by
  have h1 : b.p.investment / b.q.investment = 7 / 5.00001 := sorry
  have h2 : b.q.investment / b.r.investment = 5.00001 / 3.99999 := sorry
  have h3 : b.p.monthlyReturn / b.q.monthlyReturn = 7.00001 / 10 := sorry
  have h4 : b.q.monthlyReturn / b.r.monthlyReturn = 10 / 6 := sorry
  have h5 : b.p.activeMonths = 5 := sorry
  have h6 : b.r.activeMonths = 8 := sorry
  have h7 : b.totalProfit = 200000 := sorry
  have h8 : b.p.investment * b.p.monthlyReturn * b.p.activeMonths = 50000 := sorry
  sorry

end NUMINAMATH_CALUDE_partner_q_active_months_l1559_155962


namespace NUMINAMATH_CALUDE_butterfingers_count_l1559_155979

theorem butterfingers_count (total : ℕ) (mars : ℕ) (snickers : ℕ) (butterfingers : ℕ)
  (h1 : total = 12)
  (h2 : mars = 2)
  (h3 : snickers = 3)
  (h4 : total = mars + snickers + butterfingers) :
  butterfingers = 7 := by
  sorry

end NUMINAMATH_CALUDE_butterfingers_count_l1559_155979


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l1559_155915

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ Nᶜ = Set.univ :=
sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l1559_155915


namespace NUMINAMATH_CALUDE_not_all_lines_perp_when_planes_perp_l1559_155908

-- Define the basic geometric objects
variable (α β : Plane) (l : Line)

-- Define perpendicularity between planes
def perp_planes (p q : Plane) : Prop := sorry

-- Define a line being within a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- The statement to be proved
theorem not_all_lines_perp_when_planes_perp (α β : Plane) :
  perp_planes α β → ¬ (∀ l : Line, line_in_plane l α → perp_line_plane l β) := by
  sorry

end NUMINAMATH_CALUDE_not_all_lines_perp_when_planes_perp_l1559_155908


namespace NUMINAMATH_CALUDE_g_equality_l1559_155965

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 3*x^4 - 11*x^3 + x^2 + 5*x - 5

-- State the theorem
theorem g_equality (x : ℝ) : 2*x^5 + 4*x^3 - 5*x + 3 + g x = 3*x^4 - 7*x^3 + x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_g_equality_l1559_155965


namespace NUMINAMATH_CALUDE_translation_problem_l1559_155937

-- Part 1
def part1 (A B A' B' : ℝ × ℝ) : Prop :=
  A = (-2, -1) ∧ B = (1, -3) ∧ A' = (2, 3) → B' = (5, 1)

-- Part 2
def part2 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n) ∧ B = (2*n, m) ∧ A' = (3*m, n) ∧ B' = (6*n, m) → m = 2*n

-- Part 3
def part3 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n+1) ∧ B = (n-1, n-2) ∧ A' = (2*n-5, 2*m+3) ∧ B' = (2*m+3, n+3) →
  A = (6, 10) ∧ B = (8, 7)

theorem translation_problem :
  ∀ (A B A' B' : ℝ × ℝ) (m n : ℝ),
    part1 A B A' B' ∧
    part2 A B A' B' m n ∧
    part3 A B A' B' m n :=
by sorry

end NUMINAMATH_CALUDE_translation_problem_l1559_155937


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1559_155951

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1559_155951


namespace NUMINAMATH_CALUDE_negation_of_implication_intersection_l1559_155941

theorem negation_of_implication_intersection (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ ∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_intersection_l1559_155941


namespace NUMINAMATH_CALUDE_length_OP_greater_than_radius_l1559_155925

-- Define a circle with radius 5
def circle_radius : ℝ := 5

-- Define a point P outside the circle
def point_outside_circle (P : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Assume the circle center is at the origin
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) > circle_radius

-- Theorem statement
theorem length_OP_greater_than_radius (P : ℝ × ℝ) 
  (h : point_outside_circle P) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) > circle_radius :=
sorry

end NUMINAMATH_CALUDE_length_OP_greater_than_radius_l1559_155925


namespace NUMINAMATH_CALUDE_no_solutions_exist_l1559_155923

theorem no_solutions_exist : ¬∃ (x y : ℕ+) (m : ℕ), 
  (x : ℝ)^2 + (y : ℝ)^2 = (x : ℝ)^5 ∧ x = m^6 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l1559_155923


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_existence_of_m_outside_interval_l1559_155945

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) ↔ m < 2 :=
by sorry

theorem existence_of_m_outside_interval :
  ∃ m : ℝ, (m ≤ -2 ∨ m ≥ 2) ∧ (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_existence_of_m_outside_interval_l1559_155945


namespace NUMINAMATH_CALUDE_divisibility_floor_factorial_l1559_155921

theorem divisibility_floor_factorial (m n : ℤ) 
  (h1 : 1 < m) (h2 : m < n + 2) (h3 : n > 3) : 
  (m - 1) ∣ ⌊n! / m⌋ := by
  sorry

end NUMINAMATH_CALUDE_divisibility_floor_factorial_l1559_155921


namespace NUMINAMATH_CALUDE_unsold_books_l1559_155961

theorem unsold_books (initial_stock : ℕ) (mon tue wed thu fri : ℕ) :
  initial_stock = 800 →
  mon = 60 →
  tue = 10 →
  wed = 20 →
  thu = 44 →
  fri = 66 →
  initial_stock - (mon + tue + wed + thu + fri) = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l1559_155961


namespace NUMINAMATH_CALUDE_product_one_sum_at_least_two_l1559_155952

theorem product_one_sum_at_least_two (x : ℝ) (h1 : x > 0) (h2 : x * (1/x) = 1) : x + (1/x) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_at_least_two_l1559_155952


namespace NUMINAMATH_CALUDE_equation_solution_l1559_155928

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x + 1) + 1 / (x + 2) = 1 / x) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1559_155928


namespace NUMINAMATH_CALUDE_mairead_exercise_ratio_l1559_155907

/-- Proves the ratio of miles walked to miles jogged for Mairead's exercise routine -/
theorem mairead_exercise_ratio :
  let miles_ran : ℝ := 40
  let miles_walked_fraction : ℝ := 3 / 5 * miles_ran
  let total_distance : ℝ := 184
  let miles_walked_multiple : ℝ := total_distance - miles_ran - miles_walked_fraction
  let total_miles_walked : ℝ := miles_walked_fraction + miles_walked_multiple
  total_miles_walked / miles_ran = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_mairead_exercise_ratio_l1559_155907


namespace NUMINAMATH_CALUDE_stream_current_rate_l1559_155948

/-- The rate of the stream's current in miles per hour -/
def w : ℝ := 3

/-- The man's rowing speed in still water in miles per hour -/
def r : ℝ := 6

/-- The distance traveled downstream and upstream in miles -/
def d : ℝ := 18

/-- Theorem stating that given the conditions, the stream's current is 3 mph -/
theorem stream_current_rate : 
  (d / (r + w) + 4 = d / (r - w)) ∧ 
  (d / (3 * r + w) + 2 = d / (3 * r - w)) → 
  w = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_rate_l1559_155948


namespace NUMINAMATH_CALUDE_convenient_denominator_sum_or_diff_integer_l1559_155902

/-- A positive integer q is a convenient denominator for a real number α if 
    |α - p/q| < 1/(10q) for some integer p -/
def ConvenientDenominator (α : ℝ) (q : ℕ+) : Prop :=
  ∃ p : ℤ, |α - (p : ℝ) / q| < 1 / (10 * q)

theorem convenient_denominator_sum_or_diff_integer 
  (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) :
  (∀ q : ℕ+, ConvenientDenominator α q ↔ ConvenientDenominator β q) →
  (∃ n : ℤ, α + β = n) ∨ (∃ n : ℤ, α - β = n) := by
  sorry

end NUMINAMATH_CALUDE_convenient_denominator_sum_or_diff_integer_l1559_155902


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l1559_155910

/-- The total number of books in the 'crazy silly school' series -/
def total_books (x y : ℕ) : ℕ := x^2 + y

/-- Theorem stating that the total number of books is 177 when x = 13 and y = 8 -/
theorem crazy_silly_school_books : total_books 13 8 = 177 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l1559_155910


namespace NUMINAMATH_CALUDE_sock_selection_combinations_l1559_155963

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem sock_selection_combinations :
  choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_combinations_l1559_155963


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_l1559_155986

/-- The number of blue balls remains unchanged when red balls are removed from a box -/
theorem blue_balls_unchanged (initial_blue : ℕ) (initial_red : ℕ) (removed_red : ℕ) :
  initial_blue = initial_blue :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_l1559_155986


namespace NUMINAMATH_CALUDE_vehicle_purchase_problem_l1559_155939

/-- Represents the purchase price and profit information for new energy vehicles -/
structure VehicleInfo where
  priceA : ℝ  -- Purchase price of type A vehicle in million yuan
  priceB : ℝ  -- Purchase price of type B vehicle in million yuan
  profitA : ℝ  -- Profit from selling one type A vehicle in million yuan
  profitB : ℝ  -- Profit from selling one type B vehicle in million yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  countA : ℕ  -- Number of type A vehicles
  countB : ℕ  -- Number of type B vehicles

/-- Calculates the total cost of a purchase plan given vehicle info -/
def totalCost (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.priceA * plan.countA + info.priceB * plan.countB

/-- Calculates the total profit of a purchase plan given vehicle info -/
def totalProfit (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.profitA * plan.countA + info.profitB * plan.countB

/-- Theorem stating the properties of the vehicle purchase problem -/
theorem vehicle_purchase_problem (info : VehicleInfo) :
  (totalCost ⟨3, 2⟩ info = 95) →
  (totalCost ⟨4, 1⟩ info = 110) →
  (info.profitA = 0.012) →
  (info.profitB = 0.008) →
  (∃ (plans : List PurchasePlan),
    (∀ plan ∈ plans, totalCost plan info = 250) ∧
    (plans.length = 4) ∧
    (∃ maxProfit : ℝ, maxProfit = 18.4 ∧
      ∀ plan ∈ plans, totalProfit plan info ≤ maxProfit)) :=
sorry


end NUMINAMATH_CALUDE_vehicle_purchase_problem_l1559_155939


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l1559_155997

-- Define the length increase factor for day k
def increase_factor (k : ℕ) : ℚ := (2 * k + 2) / (2 * k + 1)

-- Define the total increase factor after n days
def total_increase (n : ℕ) : ℚ := (2 * n + 2) / 2

-- Theorem statement
theorem barry_sotter_magic (n : ℕ) : total_increase n = 50 ↔ n = 49 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l1559_155997


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1559_155919

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000000 ∧ n ≤ 99999999) ∧
  (∃ (a b c d : ℕ), 
    a + b + c + d = 12 ∧
    List.count 4 (Nat.digits 10 n) = 2 ∧
    List.count 0 (Nat.digits 10 n) = 2 ∧
    List.count 2 (Nat.digits 10 n) = 2 ∧
    List.count 6 (Nat.digits 10 n) = 2)

def largest_valid_number : ℕ := 66442200
def smallest_valid_number : ℕ := 20024466

theorem sum_of_largest_and_smallest :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  largest_valid_number + smallest_valid_number = 86466666 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1559_155919


namespace NUMINAMATH_CALUDE_carla_karen_age_difference_l1559_155977

-- Define the current ages
def karen_age : ℕ := 2
def frank_future_age : ℕ := 36
def years_until_frank_future : ℕ := 5

-- Define relationships between ages
def frank_age : ℕ := frank_future_age - years_until_frank_future
def ty_age : ℕ := frank_future_age / 3
def carla_age : ℕ := (ty_age - 4) / 2

-- Theorem to prove
theorem carla_karen_age_difference : carla_age - karen_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_carla_karen_age_difference_l1559_155977


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1559_155909

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 2002
  col : Fin 2002

/-- Represents the table arrangement -/
def TableArrangement := Cell → Fin (2002^2)

/-- Checks if a triplet satisfies the product condition -/
def satisfiesProductCondition (a b c : Fin (2002^2)) : Prop :=
  (a.val + 1) * (b.val + 1) = c.val + 1 ∨
  (a.val + 1) * (c.val + 1) = b.val + 1 ∨
  (b.val + 1) * (c.val + 1) = a.val + 1

/-- Checks if a cell satisfies the condition in its row or column -/
def cellSatisfiesCondition (t : TableArrangement) (cell : Cell) : Prop :=
  ∃ (a b c : Cell),
    ((a.row = cell.row ∧ b.row = cell.row ∧ c.row = cell.row) ∨
     (a.col = cell.col ∧ b.col = cell.col ∧ c.col = cell.col)) ∧
    satisfiesProductCondition (t a) (t b) (t c)

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement :
  ¬∃ (t : TableArrangement),
    (∀ (c₁ c₂ : Cell), c₁ ≠ c₂ → t c₁ ≠ t c₂) ∧
    (∀ (cell : Cell), cellSatisfiesCondition t cell) :=
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1559_155909


namespace NUMINAMATH_CALUDE_square_value_l1559_155959

theorem square_value (square : ℝ) : 
  (1.08 / 1.2) / 2.3 = 10.8 / square → square = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l1559_155959


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1559_155975

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to specify all properties of an isosceles triangle,
  -- just that it has a vertex angle
  vertexAngle : ℝ

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (has_40_degree_angle : ∃ (angle : ℝ), angle = 40 ∧ 
    (angle = triangle.vertexAngle ∨ 
     2 * angle + triangle.vertexAngle = 180)) :
  triangle.vertexAngle = 40 ∨ triangle.vertexAngle = 100 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1559_155975


namespace NUMINAMATH_CALUDE_square_difference_equals_324_l1559_155922

theorem square_difference_equals_324 : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_324_l1559_155922


namespace NUMINAMATH_CALUDE_spaceship_journey_theorem_l1559_155942

/-- Represents the travel schedule of a spaceship --/
structure SpaceshipJourney where
  totalJourneyTime : ℕ
  firstDayTravelTime1 : ℕ
  firstDayBreakTime1 : ℕ
  firstDayTravelTime2 : ℕ
  firstDayBreakTime2 : ℕ
  routineTravelTime : ℕ
  routineBreakTime : ℕ

/-- Calculates the total time the spaceship was not moving during its journey --/
def totalNotMovingTime (journey : SpaceshipJourney) : ℕ :=
  let firstDayBreakTime := journey.firstDayBreakTime1 + journey.firstDayBreakTime2
  let firstDayTotalTime := journey.firstDayTravelTime1 + journey.firstDayTravelTime2 + firstDayBreakTime
  let remainingTime := journey.totalJourneyTime - firstDayTotalTime
  let routineBlockTime := journey.routineTravelTime + journey.routineBreakTime
  let routineBlocks := remainingTime / routineBlockTime
  firstDayBreakTime + routineBlocks * journey.routineBreakTime

theorem spaceship_journey_theorem (journey : SpaceshipJourney) 
  (h1 : journey.totalJourneyTime = 72)
  (h2 : journey.firstDayTravelTime1 = 10)
  (h3 : journey.firstDayBreakTime1 = 3)
  (h4 : journey.firstDayTravelTime2 = 10)
  (h5 : journey.firstDayBreakTime2 = 1)
  (h6 : journey.routineTravelTime = 11)
  (h7 : journey.routineBreakTime = 1) :
  totalNotMovingTime journey = 8 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_journey_theorem_l1559_155942


namespace NUMINAMATH_CALUDE_particle_probability_l1559_155973

/-- Probability of reaching (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0,0) from (3,5) is 1385/19683 -/
theorem particle_probability : P 3 5 = 1385 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l1559_155973


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1559_155913

theorem fraction_subtraction : (5 : ℚ) / 12 - (3 : ℚ) / 18 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1559_155913


namespace NUMINAMATH_CALUDE_function_equality_l1559_155999

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 * (x - 1)^2 + 1

-- State the theorem
theorem function_equality (x : ℝ) (h : x ≥ 1) : 
  f (1 + Real.sqrt x) = 2 * x + 1 ∧ f x = 2 * x^2 - 4 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l1559_155999


namespace NUMINAMATH_CALUDE_negative_inequality_l1559_155983

theorem negative_inequality (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l1559_155983


namespace NUMINAMATH_CALUDE_coin_loss_recovery_l1559_155943

theorem coin_loss_recovery (x : ℚ) : 
  x > 0 → 
  let lost := x / 2
  let found := (4 / 5) * lost
  let remaining := x - lost + found
  x - remaining = x / 10 := by
sorry

end NUMINAMATH_CALUDE_coin_loss_recovery_l1559_155943


namespace NUMINAMATH_CALUDE_equation_solution_l1559_155903

theorem equation_solution (x : ℚ) : (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1559_155903


namespace NUMINAMATH_CALUDE_max_value_inequality_l1559_155976

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1559_155976


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l1559_155905

/-- A 2x2 matrix is not invertible if its determinant is zero. -/
def is_not_invertible (a b c d : ℚ) : Prop :=
  a * d - b * c = 0

/-- The matrix in question with x as a parameter. -/
def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 * x + 1, 9],
    ![4 - x, 10]]

/-- Theorem stating that the matrix is not invertible when x = 26/29. -/
theorem matrix_not_invertible :
  is_not_invertible (2 * (26/29) + 1) 9 (4 - (26/29)) 10 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l1559_155905


namespace NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_l1559_155978

/-- The perimeter of a semicircular cubicle with radius 14 is approximately 72 units. -/
theorem semicircular_cubicle_perimeter : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |14 * Real.pi + 28 - 72| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_l1559_155978


namespace NUMINAMATH_CALUDE_white_pairs_count_l1559_155924

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- The main theorem stating the number of coinciding white pairs -/
theorem white_pairs_count (half_count : TriangleCount) (coinciding : CoincidingPairs) : 
  half_count.red = 4 ∧ 
  half_count.blue = 4 ∧ 
  half_count.white = 6 ∧
  coinciding.red_red = 3 ∧
  coinciding.blue_blue = 2 ∧
  coinciding.red_blue = 1 →
  (half_count.white : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_count_l1559_155924


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1559_155955

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → Nat.gcd (8*m - 3) (5*m + 2) = 1) ∧ 
  Nat.gcd (8*43 - 3) (5*43 + 2) > 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1559_155955


namespace NUMINAMATH_CALUDE_prob_purple_second_l1559_155946

-- Define the bags
def bag_A : Nat × Nat := (5, 5)  -- (red, green)
def bag_B : Nat × Nat := (8, 2)  -- (purple, orange)
def bag_C : Nat × Nat := (3, 7)  -- (purple, orange)

-- Define the probability of drawing a red marble from Bag A
def prob_red_A : Rat := bag_A.1 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a green marble from Bag A
def prob_green_A : Rat := bag_A.2 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a purple marble from Bag B
def prob_purple_B : Rat := bag_B.1 / (bag_B.1 + bag_B.2)

-- Define the probability of drawing a purple marble from Bag C
def prob_purple_C : Rat := bag_C.1 / (bag_C.1 + bag_C.2)

-- Theorem: The probability of drawing a purple marble as the second marble is 11/20
theorem prob_purple_second : 
  prob_red_A * prob_purple_B + prob_green_A * prob_purple_C = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_purple_second_l1559_155946


namespace NUMINAMATH_CALUDE_roses_remaining_is_nine_l1559_155985

/-- Represents the number of roses in a dozen -/
def dozen : ℕ := 12

/-- Calculates the number of unwilted roses remaining after a series of events -/
def remaining_roses (initial_dozens : ℕ) (traded_dozens : ℕ) : ℕ :=
  let initial_roses := initial_dozens * dozen
  let after_trade := initial_roses + traded_dozens * dozen
  let after_first_wilt := after_trade / 2
  after_first_wilt / 2

/-- Proves that given the initial conditions and subsequent events, 
    the number of unwilted roses remaining is 9 -/
theorem roses_remaining_is_nine :
  remaining_roses 2 1 = 9 := by
  sorry

#eval remaining_roses 2 1

end NUMINAMATH_CALUDE_roses_remaining_is_nine_l1559_155985


namespace NUMINAMATH_CALUDE_smaller_solution_form_l1559_155920

theorem smaller_solution_form : ∃ (p q : ℤ),
  ∃ (x : ℝ),
    x^(1/4) + (40 - x)^(1/4) = 2 ∧
    x = p - Real.sqrt q ∧
    ∀ (y : ℝ), y^(1/4) + (40 - y)^(1/4) = 2 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_form_l1559_155920


namespace NUMINAMATH_CALUDE_september_percentage_l1559_155996

-- Define the total number of people surveyed
def total_people : ℕ := 150

-- Define the number of people born in September
def september_births : ℕ := 12

-- Define the percentage calculation function
def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- State the theorem
theorem september_percentage : percentage september_births total_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_september_percentage_l1559_155996


namespace NUMINAMATH_CALUDE_proposition_truth_l1559_155980

theorem proposition_truth (x y : ℝ) : x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l1559_155980


namespace NUMINAMATH_CALUDE_smallest_k_for_binomial_divisibility_l1559_155960

theorem smallest_k_for_binomial_divisibility (k : ℕ) : 
  (k ≥ 25 ∧ 49 ∣ Nat.choose (2 * k) k) ∧ 
  (∀ m : ℕ, m < 25 → ¬(49 ∣ Nat.choose (2 * m) m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_binomial_divisibility_l1559_155960


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_sixth_term_l1559_155950

theorem sum_of_fifth_and_sixth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 5 + a 6 = 152 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_sixth_term_l1559_155950


namespace NUMINAMATH_CALUDE_distance_climbed_l1559_155995

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase_steps : ℕ := 24

/-- The number of steps in the second staircase -/
def second_staircase_steps : ℕ := 3 * first_staircase_steps

/-- The number of steps in the third staircase -/
def third_staircase_steps : ℕ := second_staircase_steps - 20

/-- The height of each step in feet -/
def step_height : ℚ := 6/10

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

/-- The total distance climbed in feet -/
def total_distance : ℚ := (total_steps : ℚ) * step_height

theorem distance_climbed : total_distance = 888/10 := by
  sorry

end NUMINAMATH_CALUDE_distance_climbed_l1559_155995


namespace NUMINAMATH_CALUDE_inequality_proof_l1559_155934

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1559_155934


namespace NUMINAMATH_CALUDE_remainder_zero_prime_l1559_155966

theorem remainder_zero_prime (N : ℕ) (h_odd : Odd N) :
  (∀ i j, 2 ≤ i ∧ i < j ∧ j ≤ 1000 → N % i ≠ N % j) →
  (∃ k, 2 ≤ k ∧ k ≤ 1000 ∧ N % k = 0) →
  ∃ p, Prime p ∧ 500 < p ∧ p < 1000 ∧ N % p = 0 :=
sorry

end NUMINAMATH_CALUDE_remainder_zero_prime_l1559_155966


namespace NUMINAMATH_CALUDE_andy_max_cookies_l1559_155918

theorem andy_max_cookies (total_cookies : ℕ) (andy alexa alice : ℕ) : 
  total_cookies = 36 →
  alexa = 3 * andy →
  alice = 2 * andy →
  total_cookies = andy + alexa + alice →
  andy ≤ 6 ∧ ∃ (n : ℕ), n = 6 ∧ n = andy := by
  sorry

end NUMINAMATH_CALUDE_andy_max_cookies_l1559_155918


namespace NUMINAMATH_CALUDE_f_prime_at_zero_l1559_155953

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

theorem f_prime_at_zero : 
  (deriv f) 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_prime_at_zero_l1559_155953


namespace NUMINAMATH_CALUDE_isabel_song_count_l1559_155906

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating that Isabel bought 72 songs -/
theorem isabel_song_count :
  total_songs 4 5 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_song_count_l1559_155906


namespace NUMINAMATH_CALUDE_min_value_theorem_l1559_155987

theorem min_value_theorem (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_inequality : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1559_155987


namespace NUMINAMATH_CALUDE_function_bounded_by_identity_l1559_155926

/-- For a differentiable function f: ℝ → ℝ, if f(x) ≤ f'(x) for all x in ℝ, then f(x) ≤ x for all x in ℝ. -/
theorem function_bounded_by_identity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x ≤ deriv f x) : ∀ x, f x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_function_bounded_by_identity_l1559_155926


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1559_155916

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_slope_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1559_155916


namespace NUMINAMATH_CALUDE_second_chapter_pages_count_l1559_155982

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ
  two_chapters : first_chapter_pages + second_chapter_pages = total_pages

/-- The specific book described in the problem -/
def problem_book : Book where
  total_pages := 81
  first_chapter_pages := 13
  second_chapter_pages := 68
  two_chapters := by sorry

theorem second_chapter_pages_count :
  problem_book.second_chapter_pages = 68 := by sorry

end NUMINAMATH_CALUDE_second_chapter_pages_count_l1559_155982


namespace NUMINAMATH_CALUDE_goods_train_speed_l1559_155958

/-- The speed of a goods train passing another train in the opposite direction -/
theorem goods_train_speed (v_man : ℝ) (l_goods : ℝ) (t_pass : ℝ) :
  v_man = 40 →  -- Speed of man's train in km/h
  l_goods = 0.28 →  -- Length of goods train in km (280 m = 0.28 km)
  t_pass = 1 / 400 →  -- Time to pass in hours (9 seconds = 1/400 hours)
  ∃ v_goods : ℝ, v_goods = 72 ∧ (v_goods + v_man) * t_pass = l_goods :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l1559_155958


namespace NUMINAMATH_CALUDE_zack_traveled_18_countries_l1559_155981

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem to prove
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_18_countries_l1559_155981


namespace NUMINAMATH_CALUDE_angle_problem_l1559_155933

theorem angle_problem (x y : ℝ) : 
  x + y + 120 = 360 →
  x = 2 * y →
  x = 160 ∧ y = 80 :=
by sorry

end NUMINAMATH_CALUDE_angle_problem_l1559_155933


namespace NUMINAMATH_CALUDE_square_roots_equality_l1559_155974

theorem square_roots_equality (x a : ℝ) (hx : x > 0) :
  (3 * a - 14) ^ 2 = x ∧ (a - 2) ^ 2 = x → a = 4 ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equality_l1559_155974


namespace NUMINAMATH_CALUDE_square_of_198_l1559_155988

theorem square_of_198 : 
  (198 : ℕ)^2 = 200^2 - 2 * 200 * 2 + 2^2 := by
  have h1 : 198 = 200 - 2 := by sorry
  have h2 : ∀ (a b : ℕ), (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_square_of_198_l1559_155988
