import Mathlib

namespace polynomial_symmetry_l3836_383696

/-- A polynomial is symmetric with respect to a point if and only if it has a specific form. -/
theorem polynomial_symmetry (P : ℝ → ℝ) (a b : ℝ) :
  (∀ x, P (2*a - x) = 2*b - P x) ↔
  (∃ Q : ℝ → ℝ, ∀ x, P x = b + (x - a) * Q ((x - a)^2)) :=
by sorry

end polynomial_symmetry_l3836_383696


namespace equal_squares_exist_l3836_383656

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 10
  col : Fin 10

/-- Represents a square in the grid -/
structure Square where
  cell : Cell
  size : ℕ

/-- The theorem to be proved -/
theorem equal_squares_exist (squares : Finset Square) 
  (h1 : squares.card = 9)
  (h2 : ∀ s ∈ squares, s.cell.row < 10 ∧ s.cell.col < 10) :
  ∃ s1 s2 : Square, s1 ∈ squares ∧ s2 ∈ squares ∧ s1 ≠ s2 ∧ s1.size = s2.size :=
sorry

end equal_squares_exist_l3836_383656


namespace skirt_cost_l3836_383685

/-- Calculates the cost of each skirt in Marcia's wardrobe purchase --/
theorem skirt_cost (num_skirts num_pants num_blouses : ℕ)
  (blouse_price pant_price total_spend : ℚ) :
  num_skirts = 3 →
  num_pants = 2 →
  num_blouses = 5 →
  blouse_price = 15 →
  pant_price = 30 →
  total_spend = 180 →
  (total_spend - (num_blouses * blouse_price + pant_price * 1.5)) / num_skirts = 20 := by
  sorry

end skirt_cost_l3836_383685


namespace range_of_a_for_always_nonnegative_quadratic_l3836_383614

theorem range_of_a_for_always_nonnegative_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + a*x + a ≥ 0} = Set.Icc 0 4 := by sorry

end range_of_a_for_always_nonnegative_quadratic_l3836_383614


namespace sum_of_squares_power_l3836_383651

theorem sum_of_squares_power (a b n : ℕ+) : ∃ x y : ℤ, (a.val ^ 2 + b.val ^ 2) ^ n.val = x ^ 2 + y ^ 2 := by
  sorry

end sum_of_squares_power_l3836_383651


namespace partial_fraction_decomposition_sum_l3836_383610

theorem partial_fraction_decomposition_sum (x A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4) →
  A + B + C + D + E = 0 := by
sorry

end partial_fraction_decomposition_sum_l3836_383610


namespace meaningful_expression_range_l3836_383650

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = x / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end meaningful_expression_range_l3836_383650


namespace sum_of_powers_eight_l3836_383637

theorem sum_of_powers_eight (x : ℕ) : 
  x^8 + x^8 + x^8 + x^8 + x^5 = 4 * x^8 + x^5 :=
by sorry

end sum_of_powers_eight_l3836_383637


namespace inequality_proof_l3836_383684

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x * y + y * z + z * x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
sorry

end inequality_proof_l3836_383684


namespace three_digit_powers_intersection_l3836_383631

/-- A number is a three-digit number if it's between 100 and 999, inclusive. -/
def IsThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ := (n / 100) % 10

/-- A power of 3 -/
def PowerOf3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

/-- A power of 7 -/
def PowerOf7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem three_digit_powers_intersection :
  ∃ (n m : ℕ),
    IsThreeDigit n ∧ PowerOf3 n ∧
    IsThreeDigit m ∧ PowerOf7 m ∧
    HundredsDigit n = HundredsDigit m ∧
    HundredsDigit n = 3 ∧
    ∀ (k : ℕ),
      (∃ (p q : ℕ),
        IsThreeDigit p ∧ PowerOf3 p ∧
        IsThreeDigit q ∧ PowerOf7 q ∧
        HundredsDigit p = HundredsDigit q ∧
        HundredsDigit p = k) →
      k = 3 :=
by sorry

end three_digit_powers_intersection_l3836_383631


namespace tan_alpha_3_implies_fraction_eq_5_div_7_l3836_383653

theorem tan_alpha_3_implies_fraction_eq_5_div_7 (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end tan_alpha_3_implies_fraction_eq_5_div_7_l3836_383653


namespace phi_equality_iff_in_solution_set_l3836_383652

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- The set of solutions to the equation φ(2019n) = φ(n²) -/
def solution_set : Set ℕ+ := {1346, 2016, 2019}

/-- Theorem stating that n satisfies φ(2019n) = φ(n²) if and only if n is in the solution set -/
theorem phi_equality_iff_in_solution_set (n : ℕ+) : 
  phi (2019 * n) = phi (n * n) ↔ n ∈ solution_set := by
  sorry

end phi_equality_iff_in_solution_set_l3836_383652


namespace number_145_column_l3836_383688

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given position in the sequence --/
def column_for_position (n : ℕ) : Column :=
  match n % 11 with
  | 1 => Column.A
  | 2 => Column.B
  | 3 => Column.C
  | 4 => Column.D
  | 5 => Column.E
  | 6 => Column.F
  | 7 => Column.E
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 0 => Column.A
  | _ => Column.A  -- This case should never occur, but Lean requires it for completeness

theorem number_145_column :
  column_for_position 143 = Column.A :=
sorry

end number_145_column_l3836_383688


namespace sin_sum_of_complex_exponentials_l3836_383615

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = 4/5 + Complex.I * 3/5 →
  Complex.exp (Complex.I * δ) = -5/13 + Complex.I * 12/13 →
  Real.sin (γ + δ) = 33/65 := by
sorry

end sin_sum_of_complex_exponentials_l3836_383615


namespace synthetic_analytic_properties_l3836_383662

/-- Represents a reasoning approach in mathematics or logic -/
inductive ReasoningApproach
| Synthetic
| Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| Forward
| Backward

/-- Represents the relationship between cause and effect in reasoning -/
inductive CauseEffectRelation
| CauseToEffect
| EffectToCause

/-- Properties of a reasoning approach -/
structure ApproachProperties where
  direction : ReasoningDirection
  causeEffect : CauseEffectRelation

/-- Define properties of synthetic and analytic approaches -/
def approachProperties : ReasoningApproach → ApproachProperties
| ReasoningApproach.Synthetic => ⟨ReasoningDirection.Forward, CauseEffectRelation.CauseToEffect⟩
| ReasoningApproach.Analytic => ⟨ReasoningDirection.Backward, CauseEffectRelation.EffectToCause⟩

theorem synthetic_analytic_properties :
  (approachProperties ReasoningApproach.Synthetic).direction = ReasoningDirection.Forward ∧
  (approachProperties ReasoningApproach.Synthetic).causeEffect = CauseEffectRelation.CauseToEffect ∧
  (approachProperties ReasoningApproach.Analytic).direction = ReasoningDirection.Backward ∧
  (approachProperties ReasoningApproach.Analytic).causeEffect = CauseEffectRelation.EffectToCause :=
by sorry

end synthetic_analytic_properties_l3836_383662


namespace xy_equals_zero_l3836_383604

theorem xy_equals_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 := by
  sorry

end xy_equals_zero_l3836_383604


namespace sum_of_digits_9ab_l3836_383698

/-- Represents a number as a sequence of digits in base 10 -/
def DigitSequence (d : Nat) (n : Nat) : Nat :=
  (10^n - 1) / 9 * d

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := DigitSequence 9 2023
  let b := DigitSequence 6 2023
  sumOfDigits (9 * a * b) = 28314 := by
  sorry

end sum_of_digits_9ab_l3836_383698


namespace milk_cost_is_1_15_l3836_383646

/-- The cost of Anna's breakfast items and lunch sandwich, and the difference between lunch and breakfast costs -/
structure AnnasMeals where
  bagel_cost : ℚ
  juice_cost : ℚ
  sandwich_cost : ℚ
  lunch_breakfast_diff : ℚ

/-- Calculate the cost of the milk carton based on Anna's meal expenses -/
def milk_cost (meals : AnnasMeals) : ℚ :=
  let breakfast_cost := meals.bagel_cost + meals.juice_cost
  let lunch_cost := breakfast_cost + meals.lunch_breakfast_diff
  lunch_cost - meals.sandwich_cost

/-- Theorem stating that the cost of the milk carton is $1.15 -/
theorem milk_cost_is_1_15 (meals : AnnasMeals) 
  (h1 : meals.bagel_cost = 95/100)
  (h2 : meals.juice_cost = 85/100)
  (h3 : meals.sandwich_cost = 465/100)
  (h4 : meals.lunch_breakfast_diff = 4) :
  milk_cost meals = 115/100 := by
  sorry

end milk_cost_is_1_15_l3836_383646


namespace problem_statement_l3836_383672

theorem problem_statement (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 = c/a - b) : 
  ((a^2 * b^2 / c^2) - (2/c) + (1/(a^2 * b^2)) + (2*a*b / c^2) - (2/(a*b*c))) / 
  ((2/(a*b)) - (2*a*b/c)) / (101/c) = -1/202 := by
  sorry

end problem_statement_l3836_383672


namespace task_completion_time_l3836_383600

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Define the theorem
theorem task_completion_time 
  (start_time : Time)
  (end_third_task : Time)
  (num_tasks : Nat)
  (h1 : start_time = { hours := 9, minutes := 0 })
  (h2 : end_third_task = { hours := 11, minutes := 30 })
  (h3 : num_tasks = 4) :
  addMinutes end_third_task ((end_third_task.hours * 60 + end_third_task.minutes - 
    start_time.hours * 60 - start_time.minutes) / 3) = { hours := 12, minutes := 20 } :=
by sorry

end task_completion_time_l3836_383600


namespace olivia_supermarket_spending_l3836_383694

theorem olivia_supermarket_spending (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 54)
  (h2 : remaining_amount = 29) :
  initial_amount - remaining_amount = 25 := by
sorry

end olivia_supermarket_spending_l3836_383694


namespace opposite_of_seven_l3836_383608

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_seven : opposite 7 = -7 := by
  -- The proof goes here
  sorry

end opposite_of_seven_l3836_383608


namespace max_value_expr_min_value_sum_reciprocals_l3836_383640

/-- For x > 0, the expression 4 - 2x - 2/x is at most 0 --/
theorem max_value_expr (x : ℝ) (hx : x > 0) : 4 - 2*x - 2/x ≤ 0 := by
  sorry

/-- Given a + 2b = 1 where a and b are positive real numbers, 
    the expression 1/a + 1/b is at least 3 + 2√2 --/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + 2*b = 1) : 1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end max_value_expr_min_value_sum_reciprocals_l3836_383640


namespace quadratic_inequality_range_l3836_383677

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end quadratic_inequality_range_l3836_383677


namespace oil_bill_ratio_change_l3836_383667

theorem oil_bill_ratio_change (january_bill : ℚ) (february_bill : ℚ) : 
  january_bill = 120 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + 30) / january_bill = 3 / 2 := by
sorry

end oil_bill_ratio_change_l3836_383667


namespace complex_arithmetic_expression_equals_seven_l3836_383674

theorem complex_arithmetic_expression_equals_seven :
  (2 + 3/5 - (17/2 - 8/3) / (7/2)) * (15/2) = 7 := by sorry

end complex_arithmetic_expression_equals_seven_l3836_383674


namespace remainder_theorem_polynomial_remainder_l3836_383613

def f (x : ℝ) : ℝ := x^4 - 9*x^3 + 21*x^2 + x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 2 := by
  sorry

end remainder_theorem_polynomial_remainder_l3836_383613


namespace election_ratio_l3836_383609

theorem election_ratio (Vx Vy : ℝ) 
  (h1 : 0.64 * (Vx + Vy) = 0.76 * Vx + 0.4000000000000002 * Vy)
  (h2 : Vx > 0)
  (h3 : Vy > 0) :
  Vx / Vy = 2 := by
sorry

end election_ratio_l3836_383609


namespace right_triangle_area_with_incircle_tangency_l3836_383635

/-- 
Given a right triangle with hypotenuse length c, where the incircle's point of tangency 
divides the hypotenuse in the ratio 4:9, the area of the triangle is (36/169) * c^2.
-/
theorem right_triangle_area_with_incircle_tangency (c : ℝ) (h : c > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
    (4 / 13) * c * (9 / 13) * c = (1 / 2) * a * b ∧  -- Area calculation
    (1 / 2) * a * b = (36 / 169) * c^2  -- The final area formula
  := by sorry

end right_triangle_area_with_incircle_tangency_l3836_383635


namespace smallest_n_with_1981_zeros_l3836_383670

def count_trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem smallest_n_with_1981_zeros :
  ∃ (n : ℕ), count_trailing_zeros n = 1981 ∧
    ∀ (m : ℕ), m < n → count_trailing_zeros m < 1981 :=
by
  use 7935
  sorry

end smallest_n_with_1981_zeros_l3836_383670


namespace square_intersection_perimeter_l3836_383664

/-- Given a square with side length 2a and a line y = -x/3 intersecting it,
    the perimeter of one resulting quadrilateral divided by a is (8 + 2√10) / 3 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := -x/3
  let intersection_points := [(-a, a/3), (a, -a/3)]
  let quadrilateral_vertices := [(-a, a), (-a, a/3), (a, -a/3), (a, -a)]
  let perimeter := 
    2 * (a - a/3) +  -- vertical sides
    2 * a +          -- horizontal side
    Real.sqrt ((2*a)^2 + (2*a/3)^2)  -- diagonal
  perimeter / a = (8 + 2 * Real.sqrt 10) / 3 := by
  sorry


end square_intersection_perimeter_l3836_383664


namespace fourth_root_81_times_cube_root_27_times_sqrt_9_l3836_383629

theorem fourth_root_81_times_cube_root_27_times_sqrt_9 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end fourth_root_81_times_cube_root_27_times_sqrt_9_l3836_383629


namespace solution_set_of_inequality_l3836_383625

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 0 ↔ -4 < x ∧ x < 2 :=
sorry

end solution_set_of_inequality_l3836_383625


namespace largest_special_number_l3836_383603

/-- A number is a two-digit number if it's between 10 and 99 inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number ends in 4 if it leaves a remainder of 4 when divided by 10 -/
def ends_in_four (n : ℕ) : Prop := n % 10 = 4

/-- The set of two-digit numbers divisible by 6 and ending in 4 -/
def special_set : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_special_number : 
  ∃ (m : ℕ), m ∈ special_set ∧ ∀ (n : ℕ), n ∈ special_set → n ≤ m ∧ m = 84 :=
sorry

end largest_special_number_l3836_383603


namespace weight_of_b_l3836_383671

/-- Given the average weights of different combinations of people a, b, c, and d,
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 48 →
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (c + d) / 2 = 46 →
  b = 31 := by
  sorry

end weight_of_b_l3836_383671


namespace total_deduction_is_111_cents_l3836_383620

-- Define the hourly wage in cents
def hourly_wage : ℚ := 2500

-- Define the tax rate
def tax_rate : ℚ := 15 / 1000

-- Define the retirement contribution rate
def retirement_rate : ℚ := 3 / 100

-- Function to calculate the total deduction
def total_deduction (wage : ℚ) (tax : ℚ) (retirement : ℚ) : ℚ :=
  let tax_amount := wage * tax
  let after_tax := wage - tax_amount
  let retirement_amount := after_tax * retirement
  tax_amount + retirement_amount

-- Theorem stating that the total deduction is 111 cents
theorem total_deduction_is_111_cents :
  ⌊total_deduction hourly_wage tax_rate retirement_rate⌋ = 111 :=
sorry

end total_deduction_is_111_cents_l3836_383620


namespace m_range_l3836_383647

theorem m_range (m : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0) ∧
  ¬((∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
    (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0)) →
  m ∈ Set.Ioo 0 2 ∪ Set.Ici 4 :=
by sorry

end m_range_l3836_383647


namespace robert_ate_more_chocolates_l3836_383639

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 2 := by
  sorry

end robert_ate_more_chocolates_l3836_383639


namespace quadratic_two_distinct_roots_l3836_383683

/-- The quadratic equation (a+1)x^2 - 4x + 1 = 0 has two distinct real roots if and only if a < 3 and a ≠ -1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a + 1) * x^2 - 4 * x + 1 = 0 ∧ (a + 1) * y^2 - 4 * y + 1 = 0) ↔ 
  (a < 3 ∧ a ≠ -1) := by
sorry

end quadratic_two_distinct_roots_l3836_383683


namespace geometric_sequence_a4_l3836_383697

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 := by
  sorry

end geometric_sequence_a4_l3836_383697


namespace tangent_slope_squared_l3836_383605

/-- A line with slope m passing through the point (0, 2) -/
def line (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- An ellipse centered at the origin with semi-major axis 3 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The condition for the line to be tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃! x, ellipse x (line m x)

theorem tangent_slope_squared (m : ℝ) :
  is_tangent m → m^2 = 1/3 := by
  sorry

end tangent_slope_squared_l3836_383605


namespace optimal_purchase_minimizes_cost_l3836_383621

/-- Represents the prices and quantities of soccer balls for two brands. -/
structure SoccerBallPurchase where
  priceA : ℝ  -- Price of brand A soccer ball
  priceB : ℝ  -- Price of brand B soccer ball
  quantityA : ℝ  -- Quantity of brand A soccer balls
  quantityB : ℝ  -- Quantity of brand B soccer balls

/-- The optimal purchase strategy for soccer balls. -/
def optimalPurchase : SoccerBallPurchase := {
  priceA := 50,
  priceB := 80,
  quantityA := 60,
  quantityB := 20
}

/-- The total cost of the purchase. -/
def totalCost (p : SoccerBallPurchase) : ℝ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- Theorem stating the optimal purchase strategy minimizes cost under given conditions. -/
theorem optimal_purchase_minimizes_cost :
  let p := optimalPurchase
  (p.priceB = p.priceA + 30) ∧  -- Condition 1
  (1000 / p.priceA = 1600 / p.priceB) ∧  -- Condition 2
  (p.quantityA + p.quantityB = 80) ∧  -- Condition 3
  (p.quantityA ≥ 30) ∧  -- Condition 4
  (p.quantityA ≤ 3 * p.quantityB) ∧  -- Condition 5
  (∀ q : SoccerBallPurchase,
    (q.priceB = q.priceA + 30) →
    (1000 / q.priceA = 1600 / q.priceB) →
    (q.quantityA + q.quantityB = 80) →
    (q.quantityA ≥ 30) →
    (q.quantityA ≤ 3 * q.quantityB) →
    totalCost p ≤ totalCost q) :=
by
  sorry  -- Proof omitted

#check optimal_purchase_minimizes_cost

end optimal_purchase_minimizes_cost_l3836_383621


namespace total_purchase_ways_l3836_383658

/-- The number of oreo flavors --/
def oreo_flavors : ℕ := 5

/-- The number of milk flavors --/
def milk_flavors : ℕ := 3

/-- The total number of product types --/
def total_products : ℕ := oreo_flavors + milk_flavors

/-- The number of products they must purchase collectively --/
def purchase_count : ℕ := 3

/-- Represents the ways Alpha can choose items without repetition --/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_products k

/-- Represents the ways Beta can choose oreos with possible repetition --/
def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors

/-- The total number of ways Alpha and Beta can purchase 3 products collectively --/
def total_ways : ℕ := 
  alpha_choices 3 +
  alpha_choices 2 * beta_choices 1 +
  alpha_choices 1 * beta_choices 2 +
  beta_choices 3

theorem total_purchase_ways : total_ways = 351 := by sorry

end total_purchase_ways_l3836_383658


namespace max_volume_container_frame_l3836_383665

/-- Represents a rectangular container frame constructed from a steel bar -/
structure ContainerFrame where
  total_length : ℝ
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of the container frame -/
def volume (c : ContainerFrame) : ℝ :=
  c.length * c.width * c.height

/-- Checks if the container frame satisfies the given conditions -/
def is_valid_frame (c : ContainerFrame) : Prop :=
  c.total_length = 14.8 ∧
  c.length = c.width + 0.5 ∧
  2 * (c.length + c.width) + 4 * c.height = c.total_length

/-- Theorem stating the maximum volume and corresponding height -/
theorem max_volume_container_frame :
  ∃ (c : ContainerFrame),
    is_valid_frame c ∧
    c.height = 1.8 ∧
    volume c = 1.512 ∧
    ∀ (c' : ContainerFrame), is_valid_frame c' → volume c' ≤ volume c :=
sorry

end max_volume_container_frame_l3836_383665


namespace min_distance_to_line_l3836_383622

/-- The curve C in the x-y plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1}

/-- The distance function from a point on C to the line x - y - 4 = 0 -/
def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 4|

/-- Theorem: The minimum distance from C to the line x - y - 4 = 0 is 4 - √5 -/
theorem min_distance_to_line :
  ∃ (min_dist : ℝ), min_dist = 4 - Real.sqrt 5 ∧
  (∀ p ∈ C, distance_to_line p ≥ min_dist) ∧
  (∃ p ∈ C, distance_to_line p = min_dist) := by
  sorry

end min_distance_to_line_l3836_383622


namespace negation_equivalence_l3836_383626

theorem negation_equivalence (x y : ℤ) :
  ¬(Even (x + y) → Even x ∧ Even y) ↔ (¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end negation_equivalence_l3836_383626


namespace f_value_at_5pi_over_3_l3836_383643

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_over_3 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period f π)
  (h_domain : ∀ x ∈ Set.Icc 0 (π/2), f x = π/2 - x) :
  f (5*π/3) = π/6 := by
sorry

end f_value_at_5pi_over_3_l3836_383643


namespace distributor_profit_percentage_profit_percentage_is_87_point_5_l3836_383638

/-- Calculates the profit percentage for a distributor given specific conditions --/
theorem distributor_profit_percentage 
  (commission_rate : ℝ) 
  (cost_price : ℝ) 
  (final_price : ℝ) : ℝ :=
  let distributor_price := final_price / (1 - commission_rate)
  let profit := distributor_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- The profit percentage is approximately 87.5% given the specific conditions --/
theorem profit_percentage_is_87_point_5 :
  let result := distributor_profit_percentage 0.2 19 28.5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |result - 87.5| < ε :=
sorry

end distributor_profit_percentage_profit_percentage_is_87_point_5_l3836_383638


namespace average_exists_l3836_383642

theorem average_exists : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end average_exists_l3836_383642


namespace charlie_horns_l3836_383648

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The problem statement -/
theorem charlie_horns (charlie carli : Instruments) : 
  charlie.flutes = 1 →
  charlie.harps = 1 →
  carli.flutes = 2 * charlie.flutes →
  carli.horns = charlie.horns / 2 →
  carli.harps = 0 →
  charlie.flutes + charlie.horns + charlie.harps + 
    carli.flutes + carli.horns + carli.harps = 7 →
  charlie.horns = 2 := by
  sorry

#check charlie_horns

end charlie_horns_l3836_383648


namespace first_job_wages_proof_l3836_383660

/-- Calculates the amount received from the first job given total wages and second job details. -/
def first_job_wages (total_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  total_wages - second_job_hours * second_job_rate

/-- Proves that given the specified conditions, the amount received from the first job is $52. -/
theorem first_job_wages_proof :
  first_job_wages 160 12 9 = 52 := by
  sorry

end first_job_wages_proof_l3836_383660


namespace pool_water_increase_l3836_383678

theorem pool_water_increase (total_capacity : ℝ) (additional_water : ℝ) 
  (h1 : total_capacity = 1312.5)
  (h2 : additional_water = 300)
  (h3 : (0.8 : ℝ) * total_capacity = additional_water + (total_capacity - additional_water)) :
  let current_water := total_capacity - additional_water
  let new_water := current_water + additional_water
  (new_water - current_water) / current_water * 100 = 40 := by
  sorry

end pool_water_increase_l3836_383678


namespace document_typing_time_l3836_383699

theorem document_typing_time 
  (total_time : ℝ) 
  (susan_time : ℝ) 
  (jack_time : ℝ) 
  (h1 : total_time = 10) 
  (h2 : susan_time = 30) 
  (h3 : jack_time = 24) : 
  ∃ jonathan_time : ℝ, 
    jonathan_time = 40 ∧ 
    1 / total_time = 1 / jonathan_time + 1 / susan_time + 1 / jack_time :=
by
  sorry

#check document_typing_time

end document_typing_time_l3836_383699


namespace d₂₀₁₇_equidistant_points_l3836_383616

/-- The set S of integer coordinates (x, y) where 0 ≤ x, y ≤ 2016 -/
def S : Set (ℤ × ℤ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2016 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2016}

/-- The distance function d₂₀₁₇ -/
def d₂₀₁₇ (a b : ℤ × ℤ) : ℤ :=
  ((a.1 - b.1)^2 + (a.2 - b.2)^2) % 2017

/-- The theorem to be proved -/
theorem d₂₀₁₇_equidistant_points :
  ∃ O ∈ S,
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (2, 6) ∧
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (7, 11) →
  d₂₀₁₇ O (5, 5) = 1021 := by
  sorry

end d₂₀₁₇_equidistant_points_l3836_383616


namespace stone_137_is_5_l3836_383690

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 20

/-- Represents the target count number -/
def target_count : ℕ := 137

/-- Represents the original stone number we want to prove -/
def original_stone : ℕ := 5

/-- Function to determine the stone number given a count in the sequence -/
def stone_at_count (count : ℕ) : ℕ :=
  sorry

theorem stone_137_is_5 : stone_at_count target_count = original_stone := by
  sorry

end stone_137_is_5_l3836_383690


namespace composition_of_even_is_even_l3836_383645

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Theorem statement
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end composition_of_even_is_even_l3836_383645


namespace benedicts_house_size_l3836_383668

theorem benedicts_house_size (kennedy_house : ℕ) (benedict_house : ℕ) : 
  kennedy_house = 10000 ∧ kennedy_house = 4 * benedict_house + 600 → benedict_house = 2350 := by
  sorry

end benedicts_house_size_l3836_383668


namespace garden_area_l3836_383634

/-- The area of a garden surrounding a circular ground -/
theorem garden_area (d : ℝ) (w : ℝ) (h1 : d = 34) (h2 : w = 2) :
  let r := d / 2
  let R := r + w
  π * (R^2 - r^2) = π * 72 := by sorry

end garden_area_l3836_383634


namespace value_of_expression_l3836_383680

theorem value_of_expression : 70 * Real.sqrt ((8^10 + 4^10) / (8^4 + 4^11)) = 16 := by
  sorry

end value_of_expression_l3836_383680


namespace largest_base5_to_base10_l3836_383602

/-- Converts a base-5 number to base-10 --/
def base5To10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 5^2 + d1 * 5^1 + d0 * 5^0

/-- The largest three-digit base-5 number --/
def largestBase5 : Nat := base5To10 4 4 4

theorem largest_base5_to_base10 :
  largestBase5 = 124 := by sorry

end largest_base5_to_base10_l3836_383602


namespace triangle_angle_at_least_60_degrees_l3836_383654

theorem triangle_angle_at_least_60_degrees (A B C : ℝ) :
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → (A ≥ 60 ∨ B ≥ 60 ∨ C ≥ 60) :=
by sorry

end triangle_angle_at_least_60_degrees_l3836_383654


namespace selection_theorem_l3836_383611

theorem selection_theorem (n_volunteers : ℕ) (n_bokchoys : ℕ) : 
  n_volunteers = 4 → n_bokchoys = 3 → 
  (Nat.choose (n_volunteers + n_bokchoys) 4 - Nat.choose n_volunteers 4) = 34 := by
  sorry

end selection_theorem_l3836_383611


namespace custom_deck_combination_l3836_383649

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards per suit
def cards_per_suit : ℕ := 12

-- Define the number of face cards per suit
def face_cards_per_suit : ℕ := 3

-- Define the total number of cards in the deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Theorem statement
theorem custom_deck_combination : 
  (Nat.choose num_suits 3) * 3 * face_cards_per_suit * cards_per_suit * cards_per_suit = 5184 := by
  sorry

end custom_deck_combination_l3836_383649


namespace sum_difference_equals_result_l3836_383617

theorem sum_difference_equals_result : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end sum_difference_equals_result_l3836_383617


namespace sqrt_sum_equals_8_sqrt_3_l3836_383612

theorem sqrt_sum_equals_8_sqrt_3 : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equals_8_sqrt_3_l3836_383612


namespace quadrilateral_area_72_l3836_383607

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  (q.C.x - q.A.x) * (q.C.y - q.B.y)

/-- Theorem: The y-coordinate of B in quadrilateral ABCD that makes its area 72 square units -/
theorem quadrilateral_area_72 (q : Quadrilateral) 
    (h1 : q.A = ⟨0, 0⟩) 
    (h2 : q.B = ⟨8, q.B.y⟩)
    (h3 : q.C = ⟨8, 16⟩)
    (h4 : q.D = ⟨0, 16⟩)
    (h5 : area q = 72) : 
    q.B.y = 9 := by
  sorry


end quadrilateral_area_72_l3836_383607


namespace sequence_theorem_l3836_383682

/-- A sequence whose reciprocal forms an arithmetic sequence -/
def IsReciprocalArithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 / a (n + 1) = 1 / a n + 1 / a (n + 2)

/-- The main theorem -/
theorem sequence_theorem (x : ℕ → ℝ) (a : ℕ → ℝ) 
    (h_pos : ∀ n, x n > 0)
    (h_recip_arith : IsReciprocalArithmetic a)
    (h_x1 : x 1 = 3)
    (h_sum : x 1 + x 2 + x 3 = 39)
    (h_power : ∀ n, (x n) ^ (a n) = (x (n + 1)) ^ (a (n + 1)) ∧ 
                    (x n) ^ (a n) = (x (n + 2)) ^ (a (n + 2))) : 
  ∀ n, x n = 3^n := by
  sorry

end sequence_theorem_l3836_383682


namespace solution_set_equals_union_l3836_383632

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x^2 - 2| < 2}

-- State the theorem
theorem solution_set_equals_union : 
  solution_set = Set.union (Set.Ioo (-2) 0) (Set.Ioo 0 2) := by sorry

end solution_set_equals_union_l3836_383632


namespace sequence_general_term_l3836_383623

-- Define the sequence a_n and its partial sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n + 1

-- State the theorem
theorem sequence_general_term (a : ℕ → ℤ) :
  (∀ n : ℕ, S a n = 2 * a n + 1) →
  (∀ n : ℕ, a n = -2^(n-1)) :=
by sorry

end sequence_general_term_l3836_383623


namespace tangent_triangle_angles_correct_l3836_383663

structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  not_right : α ≠ Real.pi/2 ∧ β ≠ Real.pi/2 ∧ γ ≠ Real.pi/2

def tangent_triangle_angles (t : Triangle) : Set Real :=
  if t.α < Real.pi/2 ∧ t.β < Real.pi/2 ∧ t.γ < Real.pi/2 then
    {Real.pi - 2*t.α, Real.pi - 2*t.β, Real.pi - 2*t.γ}
  else
    {2*t.α - Real.pi, 2*t.γ, 2*t.β}

theorem tangent_triangle_angles_correct (t : Triangle) :
  ∃ (a b c : Real), tangent_triangle_angles t = {a, b, c} ∧ a + b + c = Real.pi :=
sorry

end tangent_triangle_angles_correct_l3836_383663


namespace survey_change_bounds_l3836_383624

theorem survey_change_bounds (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 1/2)
  (h2 : initial_no = 1/2)
  (h3 : final_yes = 7/10)
  (h4 : final_no = 3/10)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (x : ℚ), 1/5 ≤ x ∧ x ≤ 4/5 ∧ 
  (∃ (a b c d : ℚ), 
    a + c = initial_yes ∧
    b + d = initial_no ∧
    a + d = final_yes ∧
    b + c = final_no ∧
    c + d = x) :=
by sorry

end survey_change_bounds_l3836_383624


namespace barbara_candy_distribution_l3836_383641

/-- Represents the candy distribution problem --/
structure CandyProblem where
  original_candies : Nat
  bought_candies : Nat
  num_friends : Nat

/-- Calculates the number of candies each friend receives --/
def candies_per_friend (problem : CandyProblem) : Nat :=
  (problem.original_candies + problem.bought_candies) / problem.num_friends

/-- Theorem stating that each friend receives 4 candies --/
theorem barbara_candy_distribution :
  ∀ (problem : CandyProblem),
    problem.original_candies = 9 →
    problem.bought_candies = 18 →
    problem.num_friends = 6 →
    candies_per_friend problem = 4 :=
by
  sorry

#eval candies_per_friend { original_candies := 9, bought_candies := 18, num_friends := 6 }

end barbara_candy_distribution_l3836_383641


namespace fitted_bowling_ball_volume_l3836_383666

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let ball_diameter : ℝ := 40
  let ball_radius : ℝ := ball_diameter / 2
  let hole1_diameter : ℝ := 4
  let hole1_radius : ℝ := hole1_diameter / 2
  let hole2_diameter : ℝ := 2.5
  let hole2_radius : ℝ := hole2_diameter / 2
  let hole_depth : ℝ := 8
  let ball_volume : ℝ := (4 / 3) * π * (ball_radius ^ 3)
  let hole1_volume : ℝ := π * (hole1_radius ^ 2) * hole_depth
  let hole2_volume : ℝ := π * (hole2_radius ^ 2) * hole_depth
  ball_volume - hole1_volume - 2 * hole2_volume = 10609.67 * π :=
by sorry

end fitted_bowling_ball_volume_l3836_383666


namespace complex_equation_solution_l3836_383627

-- Define the complex numbers
def z1 (y : ℝ) : ℂ := 3 + y * Complex.I
def z2 : ℂ := 2 - Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ y : ℝ, z1 y / z2 = 1 + Complex.I ∧ y = 1 := by sorry

end complex_equation_solution_l3836_383627


namespace certain_value_proof_l3836_383681

theorem certain_value_proof (x w : ℝ) (h1 : 13 = x / (1 - w)) (h2 : w^2 = 1) : x = 26 := by
  sorry

end certain_value_proof_l3836_383681


namespace a_2_times_a_3_l3836_383695

def a : ℕ → ℤ
  | n => if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end a_2_times_a_3_l3836_383695


namespace binary_linear_equation_problem_l3836_383659

theorem binary_linear_equation_problem (m n : ℤ) : 
  (3 * m - 2 * n = -2) → 
  (3 * (m + 405) - 2 * (n - 405) = 2023) := by
  sorry

end binary_linear_equation_problem_l3836_383659


namespace inequality_and_equality_condition_l3836_383628

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ 0 < a ∧ a = b ∧ b < 1) := by
sorry

end inequality_and_equality_condition_l3836_383628


namespace cubic_polynomials_relation_l3836_383606

/-- Two monic cubic polynomials with specific roots and a relation between them -/
theorem cubic_polynomials_relation (k : ℝ) 
  (f g : ℝ → ℝ)
  (hf_monic : ∀ x, f x = x^3 + a * x^2 + b * x + c)
  (hg_monic : ∀ x, g x = x^3 + d * x^2 + e * x + i)
  (hf_roots : (k + 2) * (k + 6) * (f (k + 2)) = 0 ∧ (k + 2) * (k + 6) * (f (k + 6)) = 0)
  (hg_roots : (k + 4) * (k + 8) * (g (k + 4)) = 0 ∧ (k + 4) * (k + 8) * (g (k + 8)) = 0)
  (h_diff : ∀ x, f x - g x = x + k) : 
  k = 7 := by
  sorry

end cubic_polynomials_relation_l3836_383606


namespace harvest_duration_l3836_383687

def total_earnings : ℕ := 1216
def weekly_earnings : ℕ := 16

theorem harvest_duration :
  total_earnings / weekly_earnings = 76 :=
sorry

end harvest_duration_l3836_383687


namespace divisor_problem_l3836_383655

theorem divisor_problem (number : ℕ) (divisor : ℕ) : 
  number = 36 →
  ((number + 10) * 2 / divisor) - 2 = 88 / 2 →
  divisor = 2 := by
sorry

end divisor_problem_l3836_383655


namespace rental_ratio_l3836_383661

def comedies_rented : ℕ := 15
def action_movies_rented : ℕ := 5

theorem rental_ratio : 
  (comedies_rented : ℚ) / (action_movies_rented : ℚ) = 3 / 1 := by
  sorry

end rental_ratio_l3836_383661


namespace eulers_partition_theorem_l3836_383673

/-- The number of partitions of a natural number into distinct parts -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of a natural number into odd parts -/
def l (n : ℕ) : ℕ := sorry

/-- Euler's partition theorem: The number of partitions of a natural number
    into distinct parts is equal to the number of partitions into odd parts -/
theorem eulers_partition_theorem : ∀ n : ℕ, d n = l n := by sorry

end eulers_partition_theorem_l3836_383673


namespace intersection_theorem_l3836_383619

-- Define the set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define the set B
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_theorem : A ∩ B = {y | y > 1} := by sorry

end intersection_theorem_l3836_383619


namespace winner_is_C_l3836_383675

structure Singer :=
  (name : String)

def Singers : List Singer := [⟨"A"⟩, ⟨"B"⟩, ⟨"C"⟩, ⟨"D"⟩]

def Statement : Singer → Prop
| ⟨"A"⟩ => ∃ s : Singer, (s.name = "B" ∨ s.name = "C") ∧ s ∈ Singers
| ⟨"B"⟩ => ∀ s : Singer, (s.name = "A" ∨ s.name = "C") → s ∉ Singers
| ⟨"C"⟩ => ⟨"C"⟩ ∈ Singers
| ⟨"D"⟩ => ⟨"B"⟩ ∈ Singers
| _ => False

def Winner (s : Singer) : Prop :=
  s ∈ Singers ∧
  (∀ t : Singer, t ∈ Singers ∧ t ≠ s → t ∉ Singers) ∧
  (∃ (s1 s2 : Singer), s1 ≠ s2 ∧ Statement s1 ∧ Statement s2 ∧
    (∀ s3 : Singer, s3 ≠ s1 ∧ s3 ≠ s2 → ¬Statement s3))

theorem winner_is_C :
  Winner ⟨"C"⟩ ∧ (∀ s : Singer, s ≠ ⟨"C"⟩ → ¬Winner s) :=
sorry

end winner_is_C_l3836_383675


namespace logan_watch_hours_l3836_383633

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of minutes Logan watched television -/
def logan_watch_time : ℕ := 300

/-- Theorem: Logan watched television for 5 hours -/
theorem logan_watch_hours : logan_watch_time / minutes_per_hour = 5 := by
  sorry

end logan_watch_hours_l3836_383633


namespace jack_apples_proof_l3836_383618

def initial_apples : ℕ := 150
def jill_percentage : ℚ := 30 / 100
def june_percentage : ℚ := 20 / 100
def gift_apples : ℕ := 2

def remaining_apples : ℕ := 82

theorem jack_apples_proof :
  let after_jill := initial_apples - (initial_apples * jill_percentage).floor
  let after_june := after_jill - (after_jill * june_percentage).floor
  after_june - gift_apples = remaining_apples :=
by sorry

end jack_apples_proof_l3836_383618


namespace batsman_running_percentage_l3836_383630

/-- Calculates the percentage of runs made by running between the wickets -/
def runs_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_from_shots := boundary_runs + six_runs
  let runs_from_running := total_runs - runs_from_shots
  (runs_from_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  runs_percentage 125 5 5 = 60 :=
sorry

end batsman_running_percentage_l3836_383630


namespace initial_amount_proof_l3836_383669

/-- Proves that if an amount increases by 1/8 of itself every year and after two years
    it becomes 40500, then the initial amount was 32000. -/
theorem initial_amount_proof (A : ℚ) : 
  (A + A/8 + (A + A/8)/8 = 40500) → A = 32000 :=
by sorry

end initial_amount_proof_l3836_383669


namespace diamond_ruby_difference_l3836_383686

theorem diamond_ruby_difference (d r : ℕ) (h1 : d = 3 * r) : d - r = 2 * r := by
  sorry

end diamond_ruby_difference_l3836_383686


namespace book_pages_calculation_l3836_383689

/-- The number of pages Steve reads per day -/
def pages_per_day : ℕ := 100

/-- The number of days per week Steve reads -/
def reading_days_per_week : ℕ := 3

/-- The number of weeks Steve takes to read the book -/
def total_weeks : ℕ := 7

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * reading_days_per_week * total_weeks

theorem book_pages_calculation :
  total_pages = 2100 :=
by sorry

end book_pages_calculation_l3836_383689


namespace skateboard_price_after_discounts_l3836_383644

/-- Calculates the final price of an item after two consecutive percentage discounts -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $150 skateboard after 40% and 25% discounts is $67.50 -/
theorem skateboard_price_after_discounts :
  final_price 150 0.4 0.25 = 67.5 := by
  sorry

#eval final_price 150 0.4 0.25

end skateboard_price_after_discounts_l3836_383644


namespace probability_woman_lawyer_l3836_383693

theorem probability_woman_lawyer (total_members : ℕ) 
  (women_percentage : ℝ) (young_lawyer_percentage : ℝ) (old_lawyer_percentage : ℝ) 
  (h1 : women_percentage = 0.4)
  (h2 : young_lawyer_percentage = 0.3)
  (h3 : old_lawyer_percentage = 0.1)
  (h4 : young_lawyer_percentage + old_lawyer_percentage + 0.6 = 1) :
  (women_percentage * (young_lawyer_percentage + old_lawyer_percentage)) = 0.16 := by
  sorry

end probability_woman_lawyer_l3836_383693


namespace cost_to_selling_price_ratio_l3836_383676

/-- Given a 25% profit, prove that the ratio of cost price to selling price is 4 : 5 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0)
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
  sorry

end cost_to_selling_price_ratio_l3836_383676


namespace inverse_proportional_problem_l3836_383657

/-- Given that a and b are inversely proportional, their sum is 24, and their difference is 6,
    prove that when a = 5, b = 27. -/
theorem inverse_proportional_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
  (h2 : a + b = 24) (h3 : a - b = 6) : a = 5 → b = 27 := by
  sorry

end inverse_proportional_problem_l3836_383657


namespace minimum_value_expression_minimum_value_achievable_l3836_383692

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 ≥ 7.25 :=
by sorry

theorem minimum_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 = 7.25 :=
by sorry

end minimum_value_expression_minimum_value_achievable_l3836_383692


namespace tan_alpha_negative_two_l3836_383601

theorem tan_alpha_negative_two (α : Real) (h : Real.tan α = -2) :
  (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4/7 ∧
  3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := by
  sorry

end tan_alpha_negative_two_l3836_383601


namespace bob_ken_situp_difference_l3836_383636

-- Define the number of sit-ups each person can do
def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

-- Theorem statement
theorem bob_ken_situp_difference :
  bob_situps - ken_situps = 10 := by
  sorry

end bob_ken_situp_difference_l3836_383636


namespace geometric_sequence_formula_l3836_383691

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 9)
  (h_a4 : a 4 = 4) :
  ∀ n : ℕ, a n = 9 * (2/3)^(n - 2) :=
sorry

end geometric_sequence_formula_l3836_383691


namespace intersection_value_l3836_383679

theorem intersection_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1, 3} → 
  B = {1, m} → 
  A ∩ B = {1, 3} → 
  m = 3 := by
sorry

end intersection_value_l3836_383679
