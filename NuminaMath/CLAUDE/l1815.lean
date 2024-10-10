import Mathlib

namespace largest_integer_with_two_digit_square_l1815_181576

theorem largest_integer_with_two_digit_square : ∃ M : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10 ∧ n^2 < 100 → n ≤ M) ∧ 
  M^2 ≥ 10 ∧ M^2 < 100 ∧ 
  M = 9 := by
  sorry

end largest_integer_with_two_digit_square_l1815_181576


namespace nth_derivative_reciprocal_polynomial_l1815_181578

theorem nth_derivative_reciprocal_polynomial (k n : ℕ) (h : k > 0) :
  let f : ℝ → ℝ := λ x => 1 / (x^k - 1)
  let nth_derivative := (deriv^[n] f)
  ∃ P : ℝ → ℝ, (∀ x, nth_derivative x = P x / (x^k - 1)^(n + 1)) ∧
                P 1 = (-1)^n * n.factorial * k^n :=
by
  sorry

end nth_derivative_reciprocal_polynomial_l1815_181578


namespace problem_solution_l1815_181519

theorem problem_solution (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 5 * y = -11 := by
  sorry

end problem_solution_l1815_181519


namespace fraction_equals_zero_l1815_181517

theorem fraction_equals_zero (x : ℝ) : 
  (x - 2) / (1 - x) = 0 → x = 2 := by sorry

end fraction_equals_zero_l1815_181517


namespace all_hop_sequences_eventually_periodic_l1815_181534

/-- The biggest positive prime number that divides n -/
def f (n : ℕ) : ℕ := sorry

/-- The smallest positive prime number that divides n -/
def g (n : ℕ) : ℕ := sorry

/-- The next position after hopping from n -/
def hop (n : ℕ) : ℕ := f n + g n

/-- A sequence is eventually periodic if it reaches a cycle after some point -/
def EventuallyPeriodic (seq : ℕ → ℕ) : Prop :=
  ∃ (start cycle : ℕ), ∀ n ≥ start, seq (n + cycle) = seq n

/-- The sequence of hops starting from k -/
def hopSequence (k : ℕ) : ℕ → ℕ
  | 0 => k
  | n + 1 => hop (hopSequence k n)

theorem all_hop_sequences_eventually_periodic :
  ∀ k > 1, EventuallyPeriodic (hopSequence k) := by sorry

end all_hop_sequences_eventually_periodic_l1815_181534


namespace matrix_power_2023_l1815_181556

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l1815_181556


namespace first_year_after_2010_with_digit_sum_7_l1815_181518

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum7 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 7 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 7

theorem first_year_after_2010_with_digit_sum_7 : 
  isFirstYearAfter2010WithDigitSum7 2014 :=
sorry

end first_year_after_2010_with_digit_sum_7_l1815_181518


namespace product_to_power_minus_one_l1815_181544

theorem product_to_power_minus_one :
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 := by
  sorry

end product_to_power_minus_one_l1815_181544


namespace inequality_and_equality_condition_l1815_181587

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let expr := (a-b)*(a-c)/(a+b+c) + (b-c)*(b-d)/(b+c+d) + 
               (c-d)*(c-a)/(c+d+a) + (d-a)*(d-b)/(d+a+b)
  (expr ≥ 0) ∧ 
  (expr = 0 ↔ a = c ∧ b = d) :=
by sorry

end inequality_and_equality_condition_l1815_181587


namespace charity_share_is_75_l1815_181567

-- Define the quantities of each baked good (in dozens)
def cookie_dozens : ℕ := 6
def brownie_dozens : ℕ := 4
def muffin_dozens : ℕ := 3

-- Define the selling prices (in dollars)
def cookie_price : ℚ := 3/2
def brownie_price : ℚ := 2
def muffin_price : ℚ := 5/2

-- Define the costs to make each item (in dollars)
def cookie_cost : ℚ := 1/4
def brownie_cost : ℚ := 1/2
def muffin_cost : ℚ := 3/4

-- Define the number of charities
def num_charities : ℕ := 3

-- Define a function to calculate the profit for each type of baked good
def profit_per_type (dozens : ℕ) (price : ℚ) (cost : ℚ) : ℚ :=
  (dozens * 12 : ℚ) * (price - cost)

-- Define the total profit
def total_profit : ℚ :=
  profit_per_type cookie_dozens cookie_price cookie_cost +
  profit_per_type brownie_dozens brownie_price brownie_cost +
  profit_per_type muffin_dozens muffin_price muffin_cost

-- Theorem to prove
theorem charity_share_is_75 :
  total_profit / num_charities = 75 := by sorry

end charity_share_is_75_l1815_181567


namespace unique_solution_l1815_181536

theorem unique_solution (a b c : ℕ+) 
  (eq1 : b = a^2 - a)
  (eq2 : c = b^2 - b)
  (eq3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end unique_solution_l1815_181536


namespace solve_linear_system_l1815_181529

theorem solve_linear_system (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 5) 
  (eq2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := by
  sorry

end solve_linear_system_l1815_181529


namespace choir_dance_team_equation_l1815_181532

theorem choir_dance_team_equation (x : ℤ) : 
  (46 + x = 3 * (30 - x)) ↔ 
  (∃ (initial_choir initial_dance final_choir final_dance : ℤ),
    initial_choir = 46 ∧ 
    initial_dance = 30 ∧ 
    final_choir = initial_choir + x ∧ 
    final_dance = initial_dance - x ∧ 
    final_choir = 3 * final_dance) :=
by sorry

end choir_dance_team_equation_l1815_181532


namespace pages_to_read_in_third_week_l1815_181554

theorem pages_to_read_in_third_week 
  (total_pages : ℕ) 
  (first_week_fraction : ℚ) 
  (second_week_percent : ℚ) 
  (h1 : total_pages = 600)
  (h2 : first_week_fraction = 1/2)
  (h3 : second_week_percent = 30/100) :
  total_pages - 
  (first_week_fraction * total_pages).floor - 
  (second_week_percent * (total_pages - (first_week_fraction * total_pages).floor)).floor = 210 :=
by
  sorry

end pages_to_read_in_third_week_l1815_181554


namespace x_minus_y_is_perfect_square_l1815_181501

theorem x_minus_y_is_perfect_square (x y : ℕ+) 
  (h : 3 * x ^ 2 + x = 4 * y ^ 2 + y) : 
  ∃ (k : ℕ), x - y = k ^ 2 := by
  sorry

end x_minus_y_is_perfect_square_l1815_181501


namespace side_length_b_l1815_181589

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isArithmeticSequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

def hasCorrectArea (t : Triangle) : Prop :=
  1/2 * t.a * t.c * sin t.B = 5 * sqrt 3

-- Main theorem
theorem side_length_b (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : t.a = 4)
  (h3 : hasCorrectArea t) :
  t.b = sqrt 21 := by
  sorry


end side_length_b_l1815_181589


namespace sqrt_eight_same_type_as_sqrt_two_l1815_181583

/-- Two real numbers are of the same type if one is a rational multiple of the other -/
def same_type (a b : ℝ) : Prop := ∃ q : ℚ, a = q * b

/-- √2 is a real number -/
axiom sqrt_two : ℝ

/-- √8 is a real number -/
axiom sqrt_eight : ℝ

/-- The statement to be proved -/
theorem sqrt_eight_same_type_as_sqrt_two : same_type sqrt_eight sqrt_two := by sorry

end sqrt_eight_same_type_as_sqrt_two_l1815_181583


namespace fixed_point_of_parabola_l1815_181504

/-- Theorem: All parabolas of the form y = 4x^2 + 2tx - 3t pass through the point (3, 36) for any real t. -/
theorem fixed_point_of_parabola (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 3 = 36 := by
  sorry

end fixed_point_of_parabola_l1815_181504


namespace max_problems_solved_l1815_181543

theorem max_problems_solved (n : ℕ) (avg : ℕ) (h1 : n = 25) (h2 : avg = 6) :
  ∃ (max : ℕ), max = 126 ∧
  ∀ (problems : Fin n → ℕ),
  (∀ i, problems i ≥ 1) →
  (Finset.sum Finset.univ problems = n * avg) →
  ∀ i, problems i ≤ max :=
by sorry

end max_problems_solved_l1815_181543


namespace no_valid_assignment_for_45gon_l1815_181546

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = n

/-- Represents an assignment of digits to vertices of a polygon -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Checks if an assignment satisfies the pairwise condition -/
def SatisfiesPairwiseCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j →
    ∃ (v w : Fin n), v ≠ w ∧ 
      assignment v = i ∧ 
      assignment w = j ∧ 
      (v.val + 1) % n = w.val ∨ (w.val + 1) % n = v.val

/-- The main theorem stating that no valid assignment exists for a 45-gon -/
theorem no_valid_assignment_for_45gon :
  ¬∃ (assignment : DigitAssignment 45), 
    SatisfiesPairwiseCondition 45 assignment :=
sorry

end no_valid_assignment_for_45gon_l1815_181546


namespace banana_orange_equivalence_l1815_181549

-- Define the value of a banana in terms of oranges
def banana_value (banana_count : ℚ) (orange_count : ℕ) : Prop :=
  banana_count * (15 / 12) = orange_count

-- Theorem statement
theorem banana_orange_equivalence :
  banana_value (4 / 5 * 15) 12 →
  banana_value (3 / 4 * 8) 6 :=
by
  sorry

end banana_orange_equivalence_l1815_181549


namespace point_in_second_quadrant_l1815_181541

theorem point_in_second_quadrant (a : ℝ) : 
  (a - 3 < 0 ∧ a + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

#check point_in_second_quadrant

end point_in_second_quadrant_l1815_181541


namespace clerical_percentage_theorem_l1815_181572

/-- Represents the employee composition of a company -/
structure CompanyEmployees where
  total : ℕ
  clerical_ratio : ℚ
  management_ratio : ℚ
  clerical_reduction : ℚ

/-- Calculates the percentage of clerical employees after reduction -/
def clerical_percentage_after_reduction (c : CompanyEmployees) : ℚ :=
  let initial_clerical := c.clerical_ratio * c.total
  let reduced_clerical := initial_clerical - c.clerical_reduction * initial_clerical
  let total_after_reduction := c.total - (initial_clerical - reduced_clerical)
  (reduced_clerical / total_after_reduction) * 100

/-- Theorem stating the result of the employee reduction -/
theorem clerical_percentage_theorem (c : CompanyEmployees) 
  (h1 : c.total = 5000)
  (h2 : c.clerical_ratio = 3/7)
  (h3 : c.management_ratio = 1/3)
  (h4 : c.clerical_reduction = 3/8) :
  ∃ (ε : ℚ), abs (clerical_percentage_after_reduction c - 3194/100) < ε ∧ ε < 1/100 := by
  sorry

end clerical_percentage_theorem_l1815_181572


namespace equation_solution_l1815_181563

theorem equation_solution : ∀ x : ℚ, (2/3 : ℚ) - (1/4 : ℚ) = 1/x → x = 12/5 := by
  sorry

end equation_solution_l1815_181563


namespace mission_duration_percentage_l1815_181591

/-- Proves that given the conditions of the problem, the first mission took 60% longer than planned. -/
theorem mission_duration_percentage (planned_duration : ℕ) (second_mission_duration : ℕ) (total_duration : ℕ) :
  planned_duration = 5 →
  second_mission_duration = 3 →
  total_duration = 11 →
  ∃ (percentage : ℚ),
    percentage = 60 ∧
    total_duration = planned_duration + (percentage / 100) * planned_duration + second_mission_duration :=
by
  sorry

#check mission_duration_percentage

end mission_duration_percentage_l1815_181591


namespace missing_angles_sum_l1815_181551

-- Define the properties of our polygon
def ConvexPolygon (n : ℕ) (knownSum missingSum : ℝ) : Prop :=
  -- The polygon has n sides
  n > 2 ∧
  -- The sum of known angles is 1620°
  knownSum = 1620 ∧
  -- There are two missing angles
  -- The total sum (known + missing) is divisible by 180°
  ∃ (k : ℕ), (knownSum + missingSum) = 180 * k

-- State the theorem
theorem missing_angles_sum (n : ℕ) (knownSum missingSum : ℝ) 
  (h : ConvexPolygon n knownSum missingSum) : missingSum = 180 := by
  sorry

end missing_angles_sum_l1815_181551


namespace investment_value_after_two_years_l1815_181512

/-- Calculates the value of an investment after a given period --/
def investment_value (income : ℝ) (income_expenditure_ratio : ℝ × ℝ) 
  (savings_rate : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  let expenditure := income * income_expenditure_ratio.2 / income_expenditure_ratio.1
  let savings := income - expenditure
  let amount_saved := income * savings_rate
  let tax_deductions := income * tax_rate
  let net_investment := amount_saved - tax_deductions
  net_investment * (1 + interest_rate) ^ years

/-- Theorem stating the value of the investment after two years --/
theorem investment_value_after_two_years :
  investment_value 19000 (5, 4) 0.15 0.10 0.08 2 = 1108.08 := by
  sorry

end investment_value_after_two_years_l1815_181512


namespace triangle_side_difference_bound_l1815_181520

/-- Given a triangle ABC with side lengths a, b, c and corresponding opposite angles A, B, C,
    prove that if a = 1 and C - B = π/2, then √2/2 < c - b < 1 -/
theorem triangle_side_difference_bound (a b c A B C : Real) : 
  a = 1 → 
  C - B = π / 2 → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by
sorry

end triangle_side_difference_bound_l1815_181520


namespace gcd_5670_9800_l1815_181525

theorem gcd_5670_9800 : Nat.gcd 5670 9800 = 70 := by
  sorry

end gcd_5670_9800_l1815_181525


namespace scientific_notation_of_44_3_million_l1815_181539

theorem scientific_notation_of_44_3_million : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 :=
by sorry

end scientific_notation_of_44_3_million_l1815_181539


namespace problem_1_problem_2_l1815_181582

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = -1) : 
  (1 : ℝ) * (a + 3)^2 + (3 + a) * (3 - a) = 12 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (x - 2*y) * (x + 2*y) - (x + 2*y)^2 + 8*y^2 = -24 := by sorry

end problem_1_problem_2_l1815_181582


namespace sum_of_digits_1_to_5000_l1815_181538

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : sumOfDigitsUpTo 5000 = 229450 := by sorry

end sum_of_digits_1_to_5000_l1815_181538


namespace nested_expression_value_l1815_181523

theorem nested_expression_value : (2*(2*(2*(2*(2*(2+1)+1)+1)+1)+1)+1) = 127 := by
  sorry

end nested_expression_value_l1815_181523


namespace todd_remaining_money_l1815_181590

/-- Calculates the remaining money after Todd's purchases -/
def remaining_money (initial_amount : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (gum_count : ℕ) (soda_price : ℚ) (soda_count : ℕ) 
  (soda_discount : ℚ) : ℚ :=
  let candy_cost := candy_price * candy_count
  let gum_cost := gum_price * gum_count
  let soda_cost := soda_price * soda_count * (1 - soda_discount)
  let total_cost := candy_cost + gum_cost + soda_cost
  initial_amount - total_cost

/-- Theorem stating Todd's remaining money after purchases -/
theorem todd_remaining_money :
  remaining_money 50 2.5 7 1.5 5 3 3 0.2 = 17.8 := by
  sorry

end todd_remaining_money_l1815_181590


namespace line_parabola_tangency_false_l1815_181502

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the concept of a common point
def common_point (p : ℝ) (a b c : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line a b c x y

-- Define the concept of tangency
def is_tangent (p : ℝ) (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, common_point p a b c x y ∧
  ∀ x' y' : ℝ, common_point p a b c x' y' → x' = x ∧ y' = y

-- The theorem to be proved
theorem line_parabola_tangency_false :
  ¬(∀ p a b c : ℝ, (∃! x y : ℝ, common_point p a b c x y) → is_tangent p a b c) :=
sorry

end line_parabola_tangency_false_l1815_181502


namespace coloring_book_problem_l1815_181515

theorem coloring_book_problem (book1 book2 book3 book4 colored : ℕ) 
  (h1 : book1 = 44)
  (h2 : book2 = 35)
  (h3 : book3 = 52)
  (h4 : book4 = 48)
  (h5 : colored = 37) :
  book1 + book2 + book3 + book4 - colored = 142 := by
  sorry

end coloring_book_problem_l1815_181515


namespace power_fraction_equality_l1815_181505

theorem power_fraction_equality : (2^2016 + 2^2014) / (2^2016 - 2^2014) = 5/3 := by
  sorry

end power_fraction_equality_l1815_181505


namespace our_circle_center_and_radius_l1815_181537

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle -/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (1, 0) ∧ Circle.radius our_circle = 2 := by
  sorry

end our_circle_center_and_radius_l1815_181537


namespace system_solution_l1815_181564

theorem system_solution : ∃! (x y : ℝ), (x / 3 - (y + 1) / 2 = 1) ∧ (4 * x - (2 * y - 5) = 11) ∧ x = 0 ∧ y = -3 := by
  sorry

end system_solution_l1815_181564


namespace gcd_problem_l1815_181514

theorem gcd_problem (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ (k : ℕ), A = 4 * k ∧ B = 5 * k) : 
  Nat.gcd A B = 9 := by
  sorry

end gcd_problem_l1815_181514


namespace sandys_savings_ratio_l1815_181557

/-- The ratio of Sandy's savings this year to last year -/
theorem sandys_savings_ratio (S1 D1 : ℝ) (S1_pos : 0 < S1) (D1_pos : 0 < D1) :
  let Y := 0.06 * S1 + 0.08 * D1
  let X := 0.099 * S1 + 0.126 * D1
  X / Y = (0.099 + 0.126) / (0.06 + 0.08) := by
  sorry

end sandys_savings_ratio_l1815_181557


namespace lcm_1188_924_l1815_181595

theorem lcm_1188_924 : Nat.lcm 1188 924 = 8316 := by
  sorry

end lcm_1188_924_l1815_181595


namespace bens_class_girls_l1815_181581

theorem bens_class_girls (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 35) (h2 : girl_ratio = 3) (h3 : boy_ratio = 4) :
  ∃ (girls boys : ℕ), girls + boys = total ∧ girls * boy_ratio = boys * girl_ratio ∧ girls = 15 := by
sorry

end bens_class_girls_l1815_181581


namespace two_tap_system_solution_l1815_181506

/-- Represents the time it takes for a tap to fill a tank -/
structure TapTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents a system of two taps filling a tank -/
structure TwoTapSystem where
  tapA : TapTime
  tapB : TapTime
  timeDifference : tapA.minutes = tapB.minutes + 22
  combinedTime : (1 / tapA.minutes + 1 / tapB.minutes) * 60 = 1

theorem two_tap_system_solution (system : TwoTapSystem) :
  system.tapB.minutes = 110 ∧ system.tapA.minutes = 132 := by
  sorry

end two_tap_system_solution_l1815_181506


namespace foreign_trade_income_2007_2009_l1815_181507

/-- Represents the foreign trade income equation given the initial value,
    final value, and growth rate over a two-year period. -/
def foreign_trade_equation (initial : ℝ) (final : ℝ) (rate : ℝ) : Prop :=
  initial * (1 + rate)^2 = final

/-- Theorem stating that the foreign trade income equation holds for the given values. -/
theorem foreign_trade_income_2007_2009 :
  foreign_trade_equation 2.5 3.6 x = true :=
sorry

end foreign_trade_income_2007_2009_l1815_181507


namespace gcd_1549_1023_l1815_181531

theorem gcd_1549_1023 : Nat.gcd 1549 1023 = 1 := by
  sorry

end gcd_1549_1023_l1815_181531


namespace movie_ticket_cost_l1815_181584

theorem movie_ticket_cost (ticket_count : ℕ) (borrowed_movie_cost change paid : ℚ) : 
  ticket_count = 2 → 
  borrowed_movie_cost = 679/100 → 
  change = 137/100 → 
  paid = 20 → 
  ∃ (ticket_cost : ℚ), 
    ticket_cost * ticket_count + borrowed_movie_cost = paid - change ∧ 
    ticket_cost = 592/100 := by
  sorry

end movie_ticket_cost_l1815_181584


namespace mod_17_equivalence_l1815_181555

theorem mod_17_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 42762 % 17 = n % 17 ∧ n = 7 := by
  sorry

end mod_17_equivalence_l1815_181555


namespace haley_washing_machine_capacity_l1815_181571

/-- The number of pieces of clothing Haley's washing machine can wash at a time -/
def washing_machine_capacity (total_clothes : ℕ) (num_loads : ℕ) : ℕ :=
  total_clothes / num_loads

theorem haley_washing_machine_capacity :
  let total_shirts : ℕ := 2
  let total_sweaters : ℕ := 33
  let total_clothes : ℕ := total_shirts + total_sweaters
  let num_loads : ℕ := 5
  washing_machine_capacity total_clothes num_loads = 7 := by
  sorry

end haley_washing_machine_capacity_l1815_181571


namespace total_video_game_cost_l1815_181552

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of the video games -/
def total_cost : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total cost of video games is $9.43 -/
theorem total_video_game_cost : total_cost = 9.43 := by sorry

end total_video_game_cost_l1815_181552


namespace quadratic_equal_roots_l1815_181562

theorem quadratic_equal_roots (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a = -4 ∧ ∃ x : ℝ, x = 1/2 ∧ a * x^2 + 4 * x - 1 = 0) :=
by sorry

end quadratic_equal_roots_l1815_181562


namespace magnitude_n_equals_five_l1815_181586

/-- Given two vectors m and n in ℝ², prove that |n| = 5 -/
theorem magnitude_n_equals_five (m n : ℝ × ℝ) 
  (h1 : m.1 * n.1 + m.2 * n.2 = 0)  -- m is perpendicular to n
  (h2 : (m.1 - 2 * n.1, m.2 - 2 * n.2) = (11, -2))  -- m - 2n = (11, -2)
  (h3 : Real.sqrt (m.1^2 + m.2^2) = 5)  -- |m| = 5
  : Real.sqrt (n.1^2 + n.2^2) = 5 := by
  sorry

end magnitude_n_equals_five_l1815_181586


namespace clownfish_ratio_l1815_181550

/-- The aquarium scenario -/
structure Aquarium where
  total_fish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  blowfish_in_own_tank : ℕ
  clownfish_in_display : ℕ
  (equal_fish : clownfish = blowfish)
  (total_sum : clownfish + blowfish = total_fish)
  (blowfish_display : blowfish - blowfish_in_own_tank = clownfish - clownfish_in_display)

/-- The theorem to prove -/
theorem clownfish_ratio (aq : Aquarium) 
  (h1 : aq.total_fish = 100)
  (h2 : aq.blowfish_in_own_tank = 26)
  (h3 : aq.clownfish_in_display = 16) :
  (aq.clownfish - aq.clownfish_in_display) / (aq.clownfish - aq.blowfish_in_own_tank) = 1 / 3 := by
  sorry

end clownfish_ratio_l1815_181550


namespace max_value_expression_l1815_181533

theorem max_value_expression (x y : ℝ) :
  (Real.sqrt (8 - 4 * Real.sqrt 3) * Real.sin x - 3 * Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
  (3 + 2 * Real.sqrt (11 - Real.sqrt 3) * Real.cos y - Real.cos (2 * y)) ≤ 33 := by
  sorry

end max_value_expression_l1815_181533


namespace binomial_12_choose_10_l1815_181560

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end binomial_12_choose_10_l1815_181560


namespace three_heads_in_eight_tosses_l1815_181553

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / 2^n

/-- The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem three_heads_in_eight_tosses :
  probability_k_heads 8 3 = 7 / 32 := by
  sorry

end three_heads_in_eight_tosses_l1815_181553


namespace triangle_perimeter_l1815_181510

/-- Given a triangle with inradius 2.5 cm and area 30 cm², its perimeter is 24 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 30 → A = r * (p / 2) → p = 24 := by sorry

end triangle_perimeter_l1815_181510


namespace virus_length_scientific_notation_l1815_181542

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_length_scientific_notation :
  toScientificNotation 0.00000032 = ScientificNotation.mk 3.2 (-7) :=
sorry

end virus_length_scientific_notation_l1815_181542


namespace product_equals_fraction_l1815_181548

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18/5 := by
  sorry

end product_equals_fraction_l1815_181548


namespace find_number_l1815_181561

theorem find_number : ∃! x : ℝ, 10 * ((2 * (x^2 + 2) + 3) / 5) = 50 := by
  sorry

end find_number_l1815_181561


namespace solution_m_l1815_181596

theorem solution_m (x y m : ℝ) 
  (hx : x = 1) 
  (hy : y = 3) 
  (heq : 3 * m * x - 2 * y = 9) : m = 5 := by
  sorry

end solution_m_l1815_181596


namespace curve_properties_l1815_181580

/-- The curve function -/
def curve (c : ℝ) (x : ℝ) : ℝ := c * x^4 + x^2 - c

theorem curve_properties :
  ∀ (c : ℝ),
  -- The points (1, 1) and (-1, 1) lie on the curve for all values of c
  curve c 1 = 1 ∧ curve c (-1) = 1 ∧
  -- When c = -1/4, the curve is tangent to the line y = x at the point (1, 1)
  (let c := -1/4
   curve c 1 = 1 ∧ (deriv (curve c)) 1 = 1) ∧
  -- The curve intersects the line y = x at the points (1, 1) and (-1 + √2, -1 + √2)
  (∃ (x : ℝ), x ≠ 1 ∧ curve (-1/4) x = x ∧ x = -1 + Real.sqrt 2) :=
by sorry

end curve_properties_l1815_181580


namespace lewis_savings_l1815_181530

/-- Lewis's savings calculation -/
theorem lewis_savings (weekly_earnings weekly_rent harvest_weeks : ℕ) 
  (h1 : weekly_earnings = 491)
  (h2 : weekly_rent = 216)
  (h3 : harvest_weeks = 1181) : 
  (weekly_earnings - weekly_rent) * harvest_weeks = 324775 := by
  sorry

#eval (491 - 216) * 1181  -- To verify the result

end lewis_savings_l1815_181530


namespace locus_of_A_is_ellipse_l1815_181516

/-- Given ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- Point on the ellipse -/
def point_on_ellipse (B : ℝ × ℝ) : Prop := ellipse B.1 B.2

/-- Equilateral triangle property -/
def is_equilateral (A B : ℝ × ℝ) : Prop :=
  let FA := (A.1 - F.1, A.2 - F.2)
  let FB := (B.1 - F.1, B.2 - F.2)
  let AB := (B.1 - A.1, B.2 - A.2)
  FA.1^2 + FA.2^2 = FB.1^2 + FB.2^2 ∧ FA.1^2 + FA.2^2 = AB.1^2 + AB.2^2

/-- Counterclockwise arrangement -/
def is_counterclockwise (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1) > 0

/-- Locus of point A -/
def locus_A (A : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ), point_on_ellipse B ∧ is_equilateral A B ∧ is_counterclockwise A B

/-- Theorem statement -/
theorem locus_of_A_is_ellipse :
  ∀ (A : ℝ × ℝ), locus_A A ↔ 
    (A.1 - 2)^2 + A.2^2 + (A.1)^2 + (A.2 - 2*Real.sqrt 3)^2 = 36 :=
sorry

end locus_of_A_is_ellipse_l1815_181516


namespace max_value_fraction_l1815_181577

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (x + y) / (x - 1) ≤ 2/3 := by
  sorry

end max_value_fraction_l1815_181577


namespace correct_calculation_result_l1815_181509

theorem correct_calculation_result (x : ℝ) (h : 5 * x = 30) : 8 * x = 48 := by
  sorry

end correct_calculation_result_l1815_181509


namespace sqrt_2_irrational_in_set_l1815_181511

theorem sqrt_2_irrational_in_set (S : Set ℝ) : 
  S = {1/7, Real.sqrt 2, (8 : ℝ) ^ (1/3), 1.010010001} → 
  ∃ x ∈ S, Irrational x ∧ x = Real.sqrt 2 :=
by sorry

end sqrt_2_irrational_in_set_l1815_181511


namespace subset_implies_range_l1815_181574

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x < 2 * a - 1}
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem subset_implies_range (a : ℝ) : N a ⊆ M → a ≤ 3 := by
  sorry

end subset_implies_range_l1815_181574


namespace scientific_calculator_cost_l1815_181535

theorem scientific_calculator_cost
  (total_cost : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (graphing_cost : ℕ)
  (h1 : total_cost = 1625)
  (h2 : num_scientific = 20)
  (h3 : num_graphing = 25)
  (h4 : graphing_cost = 57)
  (h5 : num_scientific + num_graphing = 45) :
  ∃ (scientific_cost : ℕ),
    scientific_cost * num_scientific + graphing_cost * num_graphing = total_cost ∧
    scientific_cost = 10 :=
by sorry

end scientific_calculator_cost_l1815_181535


namespace tan_twenty_seventy_product_is_one_l1815_181547

theorem tan_twenty_seventy_product_is_one :
  Real.tan (20 * π / 180) * Real.tan (70 * π / 180) = 1 := by
  sorry

end tan_twenty_seventy_product_is_one_l1815_181547


namespace average_weight_of_four_friends_l1815_181599

/-- The average weight of four friends given their relative weights -/
theorem average_weight_of_four_friends 
  (jalen_weight : ℝ)
  (ponce_weight : ℝ)
  (ishmael_weight : ℝ)
  (mike_weight : ℝ)
  (h1 : jalen_weight = 160)
  (h2 : ponce_weight = jalen_weight - 10)
  (h3 : ishmael_weight = ponce_weight + 20)
  (h4 : mike_weight = ishmael_weight + ponce_weight + jalen_weight - 15) :
  (jalen_weight + ponce_weight + ishmael_weight + mike_weight) / 4 = 236.25 := by
  sorry

end average_weight_of_four_friends_l1815_181599


namespace complement_of_P_l1815_181524

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- State the theorem
theorem complement_of_P : 
  (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by sorry

end complement_of_P_l1815_181524


namespace min_perimeter_of_cross_sectional_triangle_l1815_181585

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  baseEdgeLength : ℝ
  lateralEdgeLength : ℝ

/-- Cross-sectional triangle in the pyramid -/
structure CrossSectionalTriangle (p : RegularTriangularPyramid) where
  intersectsLateralEdges : Bool

/-- The minimum perimeter of the cross-sectional triangle -/
def minPerimeter (p : RegularTriangularPyramid) (t : CrossSectionalTriangle p) : ℝ :=
  sorry

/-- Theorem: Minimum perimeter of cross-sectional triangle in given pyramid -/
theorem min_perimeter_of_cross_sectional_triangle 
  (p : RegularTriangularPyramid) 
  (t : CrossSectionalTriangle p)
  (h1 : p.baseEdgeLength = 4)
  (h2 : p.lateralEdgeLength = 8)
  (h3 : t.intersectsLateralEdges = true) :
  minPerimeter p t = 11 :=
sorry

end min_perimeter_of_cross_sectional_triangle_l1815_181585


namespace balance_theorem_l1815_181573

/-- Represents the weight of a ball in an arbitrary unit -/
@[ext] structure BallWeight where
  weight : ℚ

/-- Defines the weight relationships between different colored balls -/
structure BallWeights where
  red : BallWeight
  blue : BallWeight
  orange : BallWeight
  purple : BallWeight
  red_blue_balance : 4 * red.weight = 8 * blue.weight
  orange_blue_balance : 3 * orange.weight = 15/2 * blue.weight
  blue_purple_balance : 8 * blue.weight = 6 * purple.weight

/-- Theorem stating the balance of 68.5/3 blue balls with 5 red, 3 orange, and 4 purple balls -/
theorem balance_theorem (weights : BallWeights) :
  (68.5/3) * weights.blue.weight = 5 * weights.red.weight + 3 * weights.orange.weight + 4 * weights.purple.weight :=
by sorry

end balance_theorem_l1815_181573


namespace johns_final_push_time_l1815_181558

/-- The time of John's final push in a speed walking race --/
theorem johns_final_push_time (john_initial_distance_behind : ℝ)
                               (john_speed : ℝ)
                               (steve_speed : ℝ)
                               (john_final_distance_ahead : ℝ)
                               (h1 : john_initial_distance_behind = 16)
                               (h2 : john_speed = 4.2)
                               (h3 : steve_speed = 3.7)
                               (h4 : john_final_distance_ahead = 2) :
  let t : ℝ := (john_initial_distance_behind + john_final_distance_ahead) / (john_speed - steve_speed)
  t = 36 := by
  sorry

end johns_final_push_time_l1815_181558


namespace not_center_of_symmetry_l1815_181598

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := sorry

/-- The function y = tan(2x - π/4) -/
noncomputable def f (x : ℝ) : ℝ := tan (2 * x - Real.pi / 4)

/-- A point is a center of symmetry if it has the form (kπ/4 + π/8, 0) for some integer k -/
def is_center_of_symmetry (p : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, p.1 = k * Real.pi / 4 + Real.pi / 8 ∧ p.2 = 0

/-- The statement to be proved -/
theorem not_center_of_symmetry :
  ¬ is_center_of_symmetry (Real.pi / 4, 0) :=
sorry

end not_center_of_symmetry_l1815_181598


namespace quadratic_inequality_solution_set_l1815_181570

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x - 1 ≤ 0) → a ≤ -1/4 := by
  sorry

end quadratic_inequality_solution_set_l1815_181570


namespace scale_heights_theorem_l1815_181508

theorem scale_heights_theorem (n : ℕ) (adults children : Fin n → ℝ) 
  (h : ∀ i : Fin n, adults i > children i) :
  ∃ (scales : Fin n → ℕ+), 
    (∀ i j : Fin n, (scales i : ℝ) * adults i > (scales j : ℝ) * children j) := by
  sorry

end scale_heights_theorem_l1815_181508


namespace product_of_distinct_roots_l1815_181500

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end product_of_distinct_roots_l1815_181500


namespace magic_8_ball_theorem_l1815_181540

def magic_8_ball_probability : ℚ := 181440 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 7 →
  k = 4 →
  p = 3 / 7 →
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = magic_8_ball_probability :=
by sorry

end magic_8_ball_theorem_l1815_181540


namespace inequality_of_four_positive_reals_l1815_181593

theorem inequality_of_four_positive_reals (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := by
  sorry

end inequality_of_four_positive_reals_l1815_181593


namespace polynomial_characterization_l1815_181569

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def TripleCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the polynomial equality condition
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, TripleCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the quadratic-quartic polynomial form
def QuadraticQuarticForm (P : RealPolynomial) : Prop :=
  ∃ a₂ a₄ : ℝ, ∀ x : ℝ, P x = a₂ * x^2 + a₄ * x^4

-- The main theorem
theorem polynomial_characterization :
  ∀ P : RealPolynomial, PolynomialCondition P → QuadraticQuarticForm P :=
by
  sorry

end polynomial_characterization_l1815_181569


namespace larger_integer_value_l1815_181579

theorem larger_integer_value (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 2 → 
  (a : ℕ) * b = 216 → 
  max a b = 18 := by
sorry

end larger_integer_value_l1815_181579


namespace cosine_sine_equation_l1815_181503

theorem cosine_sine_equation (n : ℕ) :
  (∀ k : ℤ, (Real.cos (2 * k * Real.pi)) ^ n - (Real.sin (2 * k * Real.pi)) ^ n = 1) ∧
  (Even n → ∀ k : ℤ, (Real.cos ((2 * k + 1) * Real.pi)) ^ n - (Real.sin ((2 * k + 1) * Real.pi)) ^ n = 1) :=
by sorry

end cosine_sine_equation_l1815_181503


namespace problem_solution_l1815_181527

def A : Set ℝ := {-1, 0}
def B (x : ℝ) : Set ℝ := {0, 1, x+2}

theorem problem_solution (x : ℝ) (h : A ⊆ B x) : x = -3 := by
  sorry

end problem_solution_l1815_181527


namespace f_composition_equals_one_fourth_l1815_181575

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_equals_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end f_composition_equals_one_fourth_l1815_181575


namespace largest_multiple_of_seven_less_than_negative_eightyfive_l1815_181528

theorem largest_multiple_of_seven_less_than_negative_eightyfive :
  ∀ n : ℤ, n * 7 < -85 → n * 7 ≤ -91 :=
sorry

end largest_multiple_of_seven_less_than_negative_eightyfive_l1815_181528


namespace crank_slider_motion_l1815_181513

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ
  ab : ℝ
  am : ℝ
  ω : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Velocity vector -/
structure Velocity where
  vx : ℝ
  vy : ℝ

/-- Theorem for crank-slider mechanism motion -/
theorem crank_slider_motion (cs : CrankSlider) 
  (h1 : cs.oa = 90)
  (h2 : cs.ab = 90)
  (h3 : cs.am = 2/3 * cs.ab)
  (h4 : cs.ω = 10) :
  (∃ (m : ℝ → Point),
    (∀ t, m t = ⟨30 * Real.cos (10 * t) + 60, 60 * Real.sin (10 * t)⟩) ∧
    (∀ p : Point, (p.y)^2 + (30 - (p.x - 60) / (1/3))^2 = 3600) ∧
    (∃ (v : ℝ → Velocity), ∀ t, v t = ⟨-300 * Real.sin (10 * t), 600 * Real.cos (10 * t)⟩)) :=
by sorry

end crank_slider_motion_l1815_181513


namespace point_line_distance_constraint_l1815_181588

/-- Given a point P(4, a) and a line 4x - 3y - 1 = 0, if the distance from P to the line
    is no greater than 3, then a is in the range [0, 10]. -/
theorem point_line_distance_constraint (a : ℝ) : 
  let P : ℝ × ℝ := (4, a)
  let line (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0
  let distance := |4 * 4 - 3 * a - 1| / 5
  distance ≤ 3 → 0 ≤ a ∧ a ≤ 10 := by
sorry

end point_line_distance_constraint_l1815_181588


namespace number_composition_l1815_181545

/-- The number of hundreds in the given number -/
def hundreds : ℕ := 11

/-- The number of tens in the given number -/
def tens : ℕ := 11

/-- The number of units in the given number -/
def units : ℕ := 11

/-- The theorem stating that the number consisting of 11 hundreds, 11 tens, and 11 units is 1221 -/
theorem number_composition : 
  hundreds * 100 + tens * 10 + units = 1221 := by sorry

end number_composition_l1815_181545


namespace equation_roots_l1815_181522

/-- The equation has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ 
    a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 :=
sorry

end equation_roots_l1815_181522


namespace closest_multiple_of_15_to_2028_l1815_181526

def closest_multiple (n : ℕ) (m : ℕ) : ℕ :=
  m * ((n + m / 2) / m)

theorem closest_multiple_of_15_to_2028 :
  closest_multiple 2028 15 = 2025 :=
sorry

end closest_multiple_of_15_to_2028_l1815_181526


namespace particular_number_multiplication_l1815_181521

theorem particular_number_multiplication (x : ℤ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end particular_number_multiplication_l1815_181521


namespace second_car_speed_l1815_181597

/-- Given two cars traveling in the same direction for 3 hours, with one car
    traveling at 50 mph and ending up 60 miles ahead of the other car,
    prove that the speed of the second car is 30 mph. -/
theorem second_car_speed (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance_diff : ℝ)
    (h1 : speed1 = 50)
    (h2 : time = 3)
    (h3 : distance_diff = 60)
    (h4 : speed1 * time - speed2 * time = distance_diff) :
    speed2 = 30 := by
  sorry

end second_car_speed_l1815_181597


namespace largest_non_sum_of_composites_l1815_181565

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l1815_181565


namespace gcf_36_54_81_l1815_181568

theorem gcf_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by sorry

end gcf_36_54_81_l1815_181568


namespace tablet_diagonal_comparison_l1815_181566

theorem tablet_diagonal_comparison (d : ℝ) : 
  d > 0 →  -- d is positive (diagonal length)
  (6 / Real.sqrt 2)^2 = (d / Real.sqrt 2)^2 + 5.5 →  -- area comparison
  d = 5 := by
sorry

end tablet_diagonal_comparison_l1815_181566


namespace equation_solution_l1815_181559

theorem equation_solution :
  ∃ x : ℝ, 45 - 5 = 3 * x + 10 ∧ x = 10 := by
  sorry

end equation_solution_l1815_181559


namespace hannah_running_difference_l1815_181592

def monday_distance : ℕ := 9
def wednesday_distance : ℕ := 4816
def friday_distance : ℕ := 2095

theorem hannah_running_difference :
  (monday_distance * 1000) - (wednesday_distance + friday_distance) = 2089 := by
  sorry

end hannah_running_difference_l1815_181592


namespace base4_representation_has_four_digits_l1815_181594

/-- Converts a natural number from decimal to base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 75

/-- Theorem stating that the base 4 representation of 75 has four digits -/
theorem base4_representation_has_four_digits :
  (toBase4 decimalNumber).length = 4 := by
  sorry

end base4_representation_has_four_digits_l1815_181594
