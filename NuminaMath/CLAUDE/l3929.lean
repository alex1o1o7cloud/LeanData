import Mathlib

namespace NUMINAMATH_CALUDE_integer_power_sum_l3929_392933

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l3929_392933


namespace NUMINAMATH_CALUDE_last_number_is_30_l3929_392981

theorem last_number_is_30 (numbers : Fin 8 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5 + numbers 6 + numbers 7) / 8 = 25)
  (h2 : (numbers 0 + numbers 1) / 2 = 20)
  (h3 : (numbers 2 + numbers 3 + numbers 4) / 3 = 26)
  (h4 : numbers 5 = numbers 6 - 4)
  (h5 : numbers 5 = numbers 7 - 6) :
  numbers 7 = 30 := by
sorry

end NUMINAMATH_CALUDE_last_number_is_30_l3929_392981


namespace NUMINAMATH_CALUDE_no_savings_from_radio_offer_l3929_392991

-- Define the in-store price
def in_store_price : ℚ := 139.99

-- Define the radio offer components
def radio_payment : ℚ := 33.00
def num_payments : ℕ := 4
def shipping_charge : ℚ := 11.99

-- Calculate the total radio offer price
def radio_offer_price : ℚ := radio_payment * num_payments + shipping_charge

-- Define the conversion factor from dollars to cents
def dollars_to_cents : ℕ := 100

-- Theorem statement
theorem no_savings_from_radio_offer : 
  (radio_offer_price - in_store_price) * dollars_to_cents = 0 := by sorry

end NUMINAMATH_CALUDE_no_savings_from_radio_offer_l3929_392991


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l3929_392973

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 19) 
  (h2 : speed_against_stream = 11) : 
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l3929_392973


namespace NUMINAMATH_CALUDE_article_cost_l3929_392997

theorem article_cost (selling_price_high selling_price_low : ℝ) 
  (h1 : selling_price_high = 360)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 20)
  (h4 : ∀ cost, 
    (selling_price_high - cost) = (selling_price_low - cost) * 1.05) :
  ∃ cost : ℝ, cost = 60 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l3929_392997


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l3929_392918

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x + y = 10 ∧ 2*x + y = 16 → x = 6 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2 → x = 2 ∧ y = 3 := by sorry

-- Inequality 3
theorem inequality_three_solution (x : ℝ) : 
  10 - 4*(x - 4) ≤ 2*(x + 1) ↔ x ≥ 4 := by sorry

-- Inequality 4
theorem inequality_four_solution (y : ℝ) : 
  (y + 1)/6 - (2*y - 5)/4 ≥ 1 ↔ y ≤ 5/4 := by sorry

-- System 5
theorem system_five_solution (x : ℝ) : 
  x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 ≥ (x + 1)/2 → x ≤ -7 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l3929_392918


namespace NUMINAMATH_CALUDE_higher_profit_percentage_l3929_392936

/-- The profit percentage that results in $72 more profit than 9% on a cost price of $800 is 18% -/
theorem higher_profit_percentage (cost_price : ℝ) (additional_profit : ℝ) :
  cost_price = 800 →
  additional_profit = 72 →
  ∃ (P : ℝ), P * cost_price / 100 = (9 * cost_price / 100) + additional_profit ∧ P = 18 :=
by sorry

end NUMINAMATH_CALUDE_higher_profit_percentage_l3929_392936


namespace NUMINAMATH_CALUDE_x_value_l3929_392952

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 10 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3929_392952


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3929_392925

open Set Real

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ (∀ x ∈ Ioo 0 (π / 2), f x < 0)) ↔ (∃ x ∈ Ioo 0 (π / 2), f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3929_392925


namespace NUMINAMATH_CALUDE_largest_value_l3929_392905

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l3929_392905


namespace NUMINAMATH_CALUDE_parabolic_arch_height_l3929_392967

/-- Represents a parabolic arch --/
structure ParabolicArch where
  width : ℝ
  area : ℝ

/-- Calculates the height of a parabolic arch given its width and area --/
def archHeight (arch : ParabolicArch) : ℝ :=
  sorry

/-- Theorem stating that a parabolic arch with width 8 and area 160 has height 30 --/
theorem parabolic_arch_height :
  let arch : ParabolicArch := { width := 8, area := 160 }
  archHeight arch = 30 := by sorry

end NUMINAMATH_CALUDE_parabolic_arch_height_l3929_392967


namespace NUMINAMATH_CALUDE_ab_nonpositive_l3929_392951

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l3929_392951


namespace NUMINAMATH_CALUDE_bernoulli_expected_value_l3929_392974

/-- A random variable with a Bernoulli distribution -/
structure BernoulliRV (p : ℝ) :=
  (prob : 0 < p ∧ p < 1)

/-- The probability mass function for a Bernoulli random variable -/
def pmf (p : ℝ) (X : BernoulliRV p) (k : ℕ) : ℝ :=
  if k = 0 then (1 - p) else if k = 1 then p else 0

/-- The expected value of a Bernoulli random variable -/
def expectedValue (p : ℝ) (X : BernoulliRV p) : ℝ :=
  0 * pmf p X 0 + 1 * pmf p X 1

/-- Theorem: The expected value of a Bernoulli random variable is p -/
theorem bernoulli_expected_value (p : ℝ) (X : BernoulliRV p) :
  expectedValue p X = p := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_expected_value_l3929_392974


namespace NUMINAMATH_CALUDE_f_value_at_7_l3929_392955

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^7 + c * x^3 + d * x - 6

-- State the theorem
theorem f_value_at_7 (a b c d : ℝ) :
  f a b c d (-7) = 10 → f a b c d 7 = 11529580 * a - 22 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_l3929_392955


namespace NUMINAMATH_CALUDE_investment_proof_l3929_392962

/-- Represents the total amount invested -/
def total_investment : ℝ := 15280

/-- Represents the amount invested at 6% rate -/
def investment_at_6_percent : ℝ := 8200

/-- Represents the total simple interest yield in one year -/
def total_interest : ℝ := 1023

/-- First investment rate -/
def rate_1 : ℝ := 0.06

/-- Second investment rate -/
def rate_2 : ℝ := 0.075

theorem investment_proof :
  total_investment * rate_1 * (investment_at_6_percent / total_investment) +
  total_investment * rate_2 * (1 - investment_at_6_percent / total_investment) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l3929_392962


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l3929_392911

/-- Represents the relationship between price and sales --/
def sales_function (x : ℝ) : ℝ := -3 * x + 240

/-- Represents the profit function --/
def profit_function (x : ℝ) : ℝ := -3 * x^2 + 360 * x - 9600

/-- The cost price of apples --/
def cost_price : ℝ := 40

/-- The maximum allowed selling price --/
def max_price : ℝ := 55

/-- Theorem stating that the maximum profit is achieved at the maximum allowed price --/
theorem max_profit_at_max_price : 
  ∀ x, x ≥ cost_price → x ≤ max_price → profit_function x ≤ profit_function max_price :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l3929_392911


namespace NUMINAMATH_CALUDE_number_of_common_tangents_l3929_392958

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0

def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 6 = 0

def common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

theorem number_of_common_tangents :
  common_tangents circle_C1 circle_C2 = 3 :=
sorry

end NUMINAMATH_CALUDE_number_of_common_tangents_l3929_392958


namespace NUMINAMATH_CALUDE_value_range_of_f_l3929_392928

def f (x : Int) : Int := x + 1

theorem value_range_of_f :
  {y | ∃ x ∈ ({-1, 1} : Set Int), f x = y} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3929_392928


namespace NUMINAMATH_CALUDE_mirror_wall_area_ratio_l3929_392988

theorem mirror_wall_area_ratio :
  let mirror_side : ℝ := 21
  let wall_width : ℝ := 28
  let wall_length : ℝ := 31.5
  let mirror_area := mirror_side ^ 2
  let wall_area := wall_width * wall_length
  mirror_area / wall_area = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_area_ratio_l3929_392988


namespace NUMINAMATH_CALUDE_modulus_of_squared_complex_l3929_392912

theorem modulus_of_squared_complex (z : ℂ) (h : z^2 = 15 - 20*I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_squared_complex_l3929_392912


namespace NUMINAMATH_CALUDE_seating_arrangements_l3929_392956

/-- The number of ways to arrange n elements --/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := n.choose k

/-- The number of seating arrangements for 5 people in 5 seats --/
def totalArrangements : ℕ := arrangements 5

/-- The number of arrangements where 3 people are in their numbered seats --/
def threeInPlace : ℕ := choose 5 3 * arrangements 2

/-- The number of arrangements where all 5 people are in their numbered seats --/
def allInPlace : ℕ := 1

theorem seating_arrangements :
  totalArrangements - threeInPlace - allInPlace = 109 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3929_392956


namespace NUMINAMATH_CALUDE_day_of_week_n_minus_one_l3929_392965

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to add days to a given day
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

-- Define the theorem
theorem day_of_week_n_minus_one (n : Nat) :
  -- Given conditions
  (addDays DayOfWeek.Friday (150 % 7) = DayOfWeek.Friday) →
  (addDays DayOfWeek.Wednesday (210 % 7) = DayOfWeek.Wednesday) →
  -- Conclusion
  (addDays DayOfWeek.Monday 50 = DayOfWeek.Tuesday) :=
by
  sorry


end NUMINAMATH_CALUDE_day_of_week_n_minus_one_l3929_392965


namespace NUMINAMATH_CALUDE_basketball_success_rate_increase_success_rate_increase_approx_17_l3929_392927

/-- Calculates the increase in success rate percentage for basketball free throws -/
theorem basketball_success_rate_increase 
  (initial_success : Nat) 
  (initial_attempts : Nat) 
  (subsequent_success_rate : Rat) 
  (subsequent_attempts : Nat) : ℝ :=
  let total_success := initial_success + ⌊subsequent_success_rate * subsequent_attempts⌋
  let total_attempts := initial_attempts + subsequent_attempts
  let new_rate := (total_success : ℝ) / total_attempts
  let initial_rate := (initial_success : ℝ) / initial_attempts
  let increase := (new_rate - initial_rate) * 100
  ⌊increase + 0.5⌋

/-- The increase in success rate percentage is approximately 17 percentage points -/
theorem success_rate_increase_approx_17 :
  ⌊basketball_success_rate_increase 7 15 (3/4) 18 + 0.5⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_success_rate_increase_success_rate_increase_approx_17_l3929_392927


namespace NUMINAMATH_CALUDE_utilities_percentage_l3929_392919

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℝ
  research_and_development : ℝ
  equipment : ℝ
  supplies : ℝ
  transportation_degrees : ℝ
  total_budget : ℝ

/-- The theorem stating that given the specific budget allocation, the percentage spent on utilities is 5% -/
theorem utilities_percentage (budget : BudgetAllocation) : 
  budget.salaries = 60 ∧ 
  budget.research_and_development = 9 ∧ 
  budget.equipment = 4 ∧ 
  budget.supplies = 2 ∧ 
  budget.transportation_degrees = 72 ∧ 
  budget.total_budget = 100 →
  100 - (budget.salaries + budget.research_and_development + budget.equipment + budget.supplies + (budget.transportation_degrees * 100 / 360)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_utilities_percentage_l3929_392919


namespace NUMINAMATH_CALUDE_fixed_salary_is_1000_l3929_392994

/-- Calculates the commission for the old scheme -/
def old_commission (sales : ℝ) : ℝ := 0.05 * sales

/-- Calculates the commission for the new scheme -/
def new_commission (sales : ℝ) : ℝ := 0.025 * (sales - 4000)

/-- Theorem: The fixed salary in the new scheme is 1000 -/
theorem fixed_salary_is_1000 (total_sales : ℝ) (fixed_salary : ℝ) :
  total_sales = 12000 →
  fixed_salary + new_commission total_sales = old_commission total_sales + 600 →
  fixed_salary = 1000 := by
  sorry

#check fixed_salary_is_1000

end NUMINAMATH_CALUDE_fixed_salary_is_1000_l3929_392994


namespace NUMINAMATH_CALUDE_minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l3929_392910

theorem minimum_balls_drawn (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) : ℕ :=
  let total_balls := blue_balls + red_balls
  let min_blue := 2
  let min_red := 1
  8

theorem minimum_balls_drawn_correct (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ∀ n : ℕ, n ≥ minimum_balls_drawn blue_balls red_balls h_blue h_red →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls) :=
by
  sorry

theorem minimum_balls_drawn_minimal (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ¬∃ m : ℕ, m < minimum_balls_drawn blue_balls red_balls h_blue h_red ∧
  (∀ n : ℕ, n ≥ m →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls)) :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l3929_392910


namespace NUMINAMATH_CALUDE_function_decomposition_l3929_392982

-- Define the domain
def Domain : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ -1}

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x ∈ Domain, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x ∈ Domain, g (-x) = g x

-- State the theorem
theorem function_decomposition
  (f g : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_even : IsEven g)
  (h_sum : ∀ x ∈ Domain, f x + g x = 1 / (x - 1)) :
  (∀ x ∈ Domain, f x = x / (x^2 - 1)) ∧
  (∀ x ∈ Domain, g x = 1 / (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_decomposition_l3929_392982


namespace NUMINAMATH_CALUDE_horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l3929_392986

/-- Horner's Rule for a specific polynomial -/
def horner_v3 (x : ℝ) : ℝ := ((x + 3) * x - 1) * x

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem horner_rule_v3_equals_18 :
  horner_v3 2 = 18 := by sorry

theorem horner_rule_correctness (x : ℝ) :
  horner_v3 x = ((x + 3) * x - 1) * x := by sorry

theorem main_theorem : f 2 = ((((2 + 3) * 2 - 1) * 2 + 2) * 2 - 1) := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l3929_392986


namespace NUMINAMATH_CALUDE_difference_of_fractions_l3929_392915

theorem difference_of_fractions (n : ℕ) : 
  (n / 10 : ℚ) - (n / 1000 : ℚ) = 693 ↔ n = 7000 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l3929_392915


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3929_392907

open Real

/-- Cyclic sum of a function over three variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hsum : a + b + c = 3) :
    cyclicSum (fun x y z => 1 / (2 * x^2 + y^2 + z^2)) a b c ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3929_392907


namespace NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l3929_392940

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def maxVisibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: For a 12×12×12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  let c : Cube := ⟨12⟩
  maxVisibleUnitCubes c = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l3929_392940


namespace NUMINAMATH_CALUDE_correct_calculation_l3929_392921

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : (x - 7/5) * 5 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3929_392921


namespace NUMINAMATH_CALUDE_line_param_solution_l3929_392945

/-- Represents a 2D vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Represents the parameterization of a line -/
def lineParam (s h : ℝ) (t : ℝ) : Vec2 :=
  { x := s + 5 * t
    y := -2 + h * t }

/-- The equation of the line y = 3x - 11 -/
def lineEq (v : Vec2) : Prop :=
  v.y = 3 * v.x - 11

theorem line_param_solution :
  ∃ (s h : ℝ), ∀ (t : ℝ), lineEq (lineParam s h t) ∧ s = 3 ∧ h = 15 := by
  sorry

end NUMINAMATH_CALUDE_line_param_solution_l3929_392945


namespace NUMINAMATH_CALUDE_janessas_cards_l3929_392985

/-- The number of cards Janessa's father gave her. -/
def fathers_cards : ℕ := by sorry

theorem janessas_cards :
  let initial_cards : ℕ := 4
  let ebay_cards : ℕ := 36
  let discarded_cards : ℕ := 4
  let cards_to_dexter : ℕ := 29
  let cards_kept : ℕ := 20
  fathers_cards = 13 := by sorry

end NUMINAMATH_CALUDE_janessas_cards_l3929_392985


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3929_392944

/-- Given three angles around a point B, prove that ∠ABC = 60° -/
theorem angle_ABC_measure (ABC ABD CBD : ℝ) : 
  CBD = 90 → ABD = 30 → ABC + ABD + CBD = 180 → ABC = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3929_392944


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3929_392942

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (2 * Complex.I) = -20 - 14 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3929_392942


namespace NUMINAMATH_CALUDE_consecutive_multiples_of_twelve_l3929_392999

theorem consecutive_multiples_of_twelve (A B : ℕ) (h1 : Nat.gcd A B = 12) (h2 : A > B) (h3 : A - B = 12) :
  ∃ (m : ℕ), A = 12 * (m + 1) ∧ B = 12 * m :=
sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_twelve_l3929_392999


namespace NUMINAMATH_CALUDE_pump_x_time_is_4_hours_l3929_392949

/-- Represents the rate of a pump in terms of fraction of total water pumped per hour -/
structure PumpRate where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents the scenario of two pumps working on draining a flooded basement -/
structure BasementPumpScenario where
  pump_x : PumpRate
  pump_y : PumpRate
  total_water : ℝ
  total_water_positive : total_water > 0
  y_alone_time : ℝ
  y_alone_time_eq : pump_y.rate * y_alone_time = total_water
  combined_time : ℝ
  combined_time_eq : (pump_x.rate + pump_y.rate) * combined_time = total_water / 2

/-- The main theorem stating that pump X takes 4 hours to pump out half the water -/
theorem pump_x_time_is_4_hours (scenario : BasementPumpScenario) : 
  scenario.pump_x.rate * 4 = scenario.total_water / 2 ∧ 
  scenario.y_alone_time = 20 ∧ 
  scenario.combined_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_pump_x_time_is_4_hours_l3929_392949


namespace NUMINAMATH_CALUDE_farmer_animals_l3929_392906

theorem farmer_animals (goats cows pigs : ℕ) : 
  pigs = 2 * cows ∧ 
  cows = goats + 4 ∧ 
  goats + cows + pigs = 56 →
  goats = 11 := by
sorry

end NUMINAMATH_CALUDE_farmer_animals_l3929_392906


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l3929_392904

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l3929_392904


namespace NUMINAMATH_CALUDE_don_buys_from_shop_B_l3929_392943

/-- The number of bottles Don buys from Shop A -/
def bottlesFromA : ℕ := 150

/-- The number of bottles Don buys from Shop C -/
def bottlesFromC : ℕ := 220

/-- The total number of bottles Don buys -/
def totalBottles : ℕ := 550

/-- The number of bottles Don buys from Shop B -/
def bottlesFromB : ℕ := totalBottles - (bottlesFromA + bottlesFromC)

theorem don_buys_from_shop_B : bottlesFromB = 180 := by
  sorry

end NUMINAMATH_CALUDE_don_buys_from_shop_B_l3929_392943


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l3929_392901

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l3929_392901


namespace NUMINAMATH_CALUDE_liquid_level_rate_of_change_l3929_392923

/-- The rate of change of liquid level height in a cylindrical container -/
theorem liquid_level_rate_of_change 
  (d : ℝ) -- diameter of the base
  (drain_rate : ℝ) -- rate at which liquid is drained
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (hd : d = 2) -- given diameter
  (hdrain : drain_rate = 0.01) -- given drain rate
  : deriv h t = -drain_rate / (π * (d/2)^2) := by
  sorry

#check liquid_level_rate_of_change

end NUMINAMATH_CALUDE_liquid_level_rate_of_change_l3929_392923


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l3929_392924

/-- The minimum value of 2x^2 + 4xy + 5y^2 - 8x - 6y over all real numbers x and y is 3 -/
theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 6 * y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l3929_392924


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3929_392937

theorem angle_sum_is_pi_over_two (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2 →
  4 * Real.sin (2*a) + 3 * Real.sin (2*b) = 3 →
  2*a + b = π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3929_392937


namespace NUMINAMATH_CALUDE_middle_term_coefficient_l3929_392917

/-- Given a natural number n, returns the binomial expansion of (1-2x)^n -/
def binomialExpansion (n : ℕ) : List ℤ := sorry

/-- Returns the sum of coefficients of even-numbered terms in a list -/
def sumEvenTerms (coeffs : List ℤ) : ℤ := sorry

/-- Returns the middle coefficient of a list -/
def middleCoefficient (coeffs : List ℤ) : ℤ := sorry

theorem middle_term_coefficient (n : ℕ) :
  sumEvenTerms (binomialExpansion n) = 128 →
  middleCoefficient (binomialExpansion n) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_l3929_392917


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l3929_392983

/-- The 'crazy silly school' series problem -/
theorem crazy_silly_school_movies :
  ∀ (total_books watched_movies remaining_movies : ℕ),
    total_books = 21 →
    watched_movies = 4 →
    remaining_movies = 4 →
    watched_movies + remaining_movies = 8 :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l3929_392983


namespace NUMINAMATH_CALUDE_work_completion_time_l3929_392995

/-- Given two workers A and B who complete a work together in a certain number of days,
    this function calculates the time it takes for them to complete the work together. -/
def time_to_complete (time_A : ℝ) (time_together : ℝ) : ℝ :=
  time_together

/-- Theorem stating that if A and B complete the work in 9 days together,
    and A alone can do the work in 18 days, then A and B together can complete
    the work in 9 days. -/
theorem work_completion_time (time_A : ℝ) (time_together : ℝ)
    (h1 : time_A = 18)
    (h2 : time_together = 9) :
    time_to_complete time_A time_together = 9 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3929_392995


namespace NUMINAMATH_CALUDE_jellybean_difference_l3929_392998

theorem jellybean_difference (gigi_jellybeans rory_jellybeans lorelai_jellybeans : ℕ) : 
  gigi_jellybeans = 15 →
  rory_jellybeans > gigi_jellybeans →
  lorelai_jellybeans = 3 * (rory_jellybeans + gigi_jellybeans) →
  lorelai_jellybeans = 180 →
  rory_jellybeans - gigi_jellybeans = 30 := by
sorry

end NUMINAMATH_CALUDE_jellybean_difference_l3929_392998


namespace NUMINAMATH_CALUDE_beads_per_necklace_l3929_392993

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 18) (h2 : num_necklaces = 6) :
  total_beads / num_necklaces = 3 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l3929_392993


namespace NUMINAMATH_CALUDE_soybean_oil_conversion_l3929_392977

/-- Represents the problem of determining the amount of soybeans converted to soybean oil --/
theorem soybean_oil_conversion (total_soybeans : ℝ) (total_revenue : ℝ) 
  (tofu_conversion : ℝ) (oil_conversion : ℝ) (tofu_price : ℝ) (oil_price : ℝ) :
  total_soybeans = 460 ∧ 
  total_revenue = 1800 ∧
  tofu_conversion = 3 ∧
  oil_conversion = 1 / 6 ∧
  tofu_price = 3 ∧
  oil_price = 15 →
  ∃ (x : ℝ), 
    x = 360 ∧ 
    tofu_price * tofu_conversion * (total_soybeans - x) + oil_price * oil_conversion * x = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_soybean_oil_conversion_l3929_392977


namespace NUMINAMATH_CALUDE_vector_addition_proof_l3929_392992

theorem vector_addition_proof (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-2, 2)) :
  a + 2 • b = (-3, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l3929_392992


namespace NUMINAMATH_CALUDE_stating_arrangements_count_l3929_392968

/-- 
Given a positive integer n, this function returns the number of arrangements
of integers 1 to n, where each number (except the leftmost) differs by 1
from some number to its left.
-/
def countArrangements (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of arrangements of integers 1 to n,
where each number (except the leftmost) differs by 1 from some number to its left,
is equal to 2^(n-1).
-/
theorem arrangements_count (n : ℕ) (h : n > 0) :
  countArrangements n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_stating_arrangements_count_l3929_392968


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l3929_392971

/-- A quadratic equation kx² + (2k-1)x + k = 0 has two real roots if and only if k ≤ 1/4 and k ≠ 0 -/
theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, k * z^2 + (2*k - 1) * z + k = 0 ↔ (z = x ∨ z = y)) ↔ 
  (k ≤ 1/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l3929_392971


namespace NUMINAMATH_CALUDE_thomas_money_left_l3929_392964

/-- Calculates the money left over after selling books and buying records. -/
def money_left_over (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (record_price : ℚ) : ℚ :=
  num_books * book_price - num_records * record_price

/-- Proves that Thomas has $75 left over after selling his books and buying records. -/
theorem thomas_money_left : money_left_over 200 (3/2) 75 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_thomas_money_left_l3929_392964


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_greater_than_l3929_392975

theorem contradiction_assumption_for_greater_than (a b : ℝ) : 
  (¬(a > b) ↔ (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_greater_than_l3929_392975


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l3929_392941

/-- Calculates the number of strokes over par in a golf game. -/
def strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  let total_holes := rounds * 18
  let total_strokes := avg_strokes_per_hole * total_holes
  let total_par := par_per_hole * total_holes
  total_strokes - total_par

/-- Proves that given 9 rounds of golf, an average of 4 strokes per hole, 
    and a par value of 3 per hole, the number of strokes over par is 162. -/
theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 162 := by
  sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l3929_392941


namespace NUMINAMATH_CALUDE_value_of_t_l3929_392900

theorem value_of_t (u m j : ℝ) (A t : ℝ) (h : A = u^m / (2 + j)^t) :
  t = Real.log (u^m / A) / Real.log (2 + j) := by
  sorry

end NUMINAMATH_CALUDE_value_of_t_l3929_392900


namespace NUMINAMATH_CALUDE_complement_of_range_l3929_392902

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def domain : Set ℝ := Set.univ

def range : Set ℝ := {y | ∃ x, f x = y}

theorem complement_of_range :
  (domain \ range) = {x | x < -4} :=
sorry

end NUMINAMATH_CALUDE_complement_of_range_l3929_392902


namespace NUMINAMATH_CALUDE_division_problem_l3929_392948

theorem division_problem :
  ∃ x : ℝ, x > 0 ∧ 
    2 * x + (100 - 2 * x) = 100 ∧
    (300 - 6 * x) + x = 100 ∧
    x = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3929_392948


namespace NUMINAMATH_CALUDE_library_books_difference_l3929_392984

theorem library_books_difference (initial_books borrowed_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : borrowed_books = 18) :
  initial_books - borrowed_books = 57 := by
sorry

end NUMINAMATH_CALUDE_library_books_difference_l3929_392984


namespace NUMINAMATH_CALUDE_race_time_a_l3929_392989

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- B runs 960 meters in the time A finishes
  960 = b.speed * a.time ∧
  -- A finishes 8 seconds before B
  b.time = a.time + 8 ∧
  -- A and B have different speeds
  a.speed ≠ b.speed

/-- The theorem stating A's finishing time -/
theorem race_time_a (a b : Runner) (h : Race a b) : a.time = 200 :=
  sorry

#check race_time_a

end NUMINAMATH_CALUDE_race_time_a_l3929_392989


namespace NUMINAMATH_CALUDE_equation_solutions_l3929_392930

theorem equation_solutions : 
  ∀ x : ℝ, (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
             1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3929_392930


namespace NUMINAMATH_CALUDE_lindseys_money_is_36_l3929_392938

/-- Calculates the remaining money for Lindsey given her savings and spending. -/
def lindseys_remaining_money (sept_savings oct_savings nov_savings mom_bonus_threshold mom_bonus video_game_cost : ℕ) : ℕ :=
  let total_savings := sept_savings + oct_savings + nov_savings
  let with_bonus := total_savings + if total_savings > mom_bonus_threshold then mom_bonus else 0
  with_bonus - video_game_cost

/-- Proves that Lindsey's remaining money is $36 given her savings and spending. -/
theorem lindseys_money_is_36 :
  lindseys_remaining_money 50 37 11 75 25 87 = 36 := by
  sorry

#eval lindseys_remaining_money 50 37 11 75 25 87

end NUMINAMATH_CALUDE_lindseys_money_is_36_l3929_392938


namespace NUMINAMATH_CALUDE_course_selection_theorem_l3929_392954

def category_A_courses : ℕ := 3
def category_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to choose courses from two categories with the given constraints -/
def number_of_ways_to_choose : ℕ :=
  (Nat.choose category_A_courses 1 * Nat.choose category_B_courses 2) +
  (Nat.choose category_A_courses 2 * Nat.choose category_B_courses 1)

theorem course_selection_theorem :
  number_of_ways_to_choose = 30 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l3929_392954


namespace NUMINAMATH_CALUDE_train_length_and_speed_l3929_392922

/-- A train passes by an observer in t₁ seconds and through a bridge of length a meters in t₂ seconds at a constant speed. This theorem proves the formulas for the train's length and speed. -/
theorem train_length_and_speed (t₁ t₂ a : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > t₁) (h₃ : a > 0) :
  ∃ (L V : ℝ),
    L = (a * t₁) / (t₂ - t₁) ∧
    V = a / (t₂ - t₁) ∧
    L / t₁ = V ∧
    (L + a) / t₂ = V :=
by sorry

end NUMINAMATH_CALUDE_train_length_and_speed_l3929_392922


namespace NUMINAMATH_CALUDE_vector_problem_l3929_392935

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (a b : ℝ × ℝ) : Prop :=
  ¬∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_problem (a b c : ℝ × ℝ) (x : ℝ) :
  NonCollinear a b →
  a = (1, 2) →
  b = (x, 6) →
  ‖a - b‖ = 2 * Real.sqrt 5 →
  c = (2 • a) + b →
  c = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3929_392935


namespace NUMINAMATH_CALUDE_largest_integer_with_mean_seven_l3929_392909

theorem largest_integer_with_mean_seven (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 7 →
  ∀ x : ℕ, (x = a ∨ x = b ∨ x = c) → x ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_mean_seven_l3929_392909


namespace NUMINAMATH_CALUDE_simplify_expression_l3929_392914

theorem simplify_expression : 4^4 * 9^4 * 4^9 * 9^9 = 36^13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3929_392914


namespace NUMINAMATH_CALUDE_max_a_value_l3929_392972

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_a_value :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∀ b : ℝ, (∀ x : ℝ, determinant (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3929_392972


namespace NUMINAMATH_CALUDE_five_circles_arrangement_exists_four_circles_arrangement_not_exists_l3929_392980

-- Define a circle on a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a ray starting from a point
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry

-- Function to check if a ray intersects at least two circles from a list
def ray_intersects_at_least_two (r : Ray) (circles : List Circle) : Prop :=
  sorry

-- Function to check if a circle covers a point
def circle_covers_point (c : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Theorem for part (a)
theorem five_circles_arrangement_exists :
  ∃ (circles : List Circle), circles.length = 5 ∧
  ∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles :=
sorry

-- Theorem for part (b)
theorem four_circles_arrangement_not_exists :
  ¬ ∃ (circles : List Circle), circles.length = 4 ∧
  (∀ c ∈ circles, ¬ circle_covers_point c (0, 0)) ∧
  (∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles) :=
sorry

end NUMINAMATH_CALUDE_five_circles_arrangement_exists_four_circles_arrangement_not_exists_l3929_392980


namespace NUMINAMATH_CALUDE_base8_246_to_base10_l3929_392996

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- The base 10 representation of 246₈ is 166 -/
theorem base8_246_to_base10 : base8_to_base10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_246_to_base10_l3929_392996


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l3929_392987

theorem system_inequalities_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2*x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) →
  (a = 2 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l3929_392987


namespace NUMINAMATH_CALUDE_childrens_ticket_price_l3929_392929

/-- The cost of a children's ticket to the aquarium -/
def childrens_ticket_cost : ℝ := 20

/-- The cost of an adult ticket to the aquarium -/
def adult_ticket_cost : ℝ := 35

/-- The number of adults in Violet's family -/
def num_adults : ℕ := 1

/-- The number of children in Violet's family -/
def num_children : ℕ := 6

/-- The total cost of separate tickets for Violet's family -/
def total_separate_cost : ℝ := 155

theorem childrens_ticket_price : 
  childrens_ticket_cost * num_children + adult_ticket_cost * num_adults = total_separate_cost :=
by sorry

end NUMINAMATH_CALUDE_childrens_ticket_price_l3929_392929


namespace NUMINAMATH_CALUDE_john_paid_21_dollars_l3929_392947

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_21_dollars_l3929_392947


namespace NUMINAMATH_CALUDE_total_tickets_l3929_392931

def tate_initial_tickets : ℕ := 32
def additional_tickets : ℕ := 2

def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

def peyton_tickets : ℕ := tate_total_tickets / 2

theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_l3929_392931


namespace NUMINAMATH_CALUDE_parallel_planes_intersection_theorem_l3929_392913

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersects : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_intersection_theorem 
  (α β γ : Plane) (m n : Line) :
  parallel_planes α β →
  intersects α γ m →
  intersects β γ n →
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_intersection_theorem_l3929_392913


namespace NUMINAMATH_CALUDE_august_electricity_bill_l3929_392976

-- Define electricity prices for different seasons
def electricity_price (month : Nat) : Real :=
  if month ≤ 3 then 0.12
  else if month ≤ 6 then 0.10
  else if month ≤ 9 then 0.09
  else 0.11

-- Define appliance consumption rates
def oven_consumption : Real := 2.4
def ac_consumption : Real := 1.6
def fridge_consumption : Real := 0.15
def washer_consumption : Real := 0.5

-- Define appliance usage durations
def oven_usage : Nat := 25
def ac_usage : Nat := 150
def fridge_usage : Nat := 720
def washer_usage : Nat := 20

-- Define the month of August
def august : Nat := 8

-- Theorem: Coco's total electricity bill for August is $37.62
theorem august_electricity_bill :
  let price := electricity_price august
  let oven_cost := oven_consumption * oven_usage * price
  let ac_cost := ac_consumption * ac_usage * price
  let fridge_cost := fridge_consumption * fridge_usage * price
  let washer_cost := washer_consumption * washer_usage * price
  oven_cost + ac_cost + fridge_cost + washer_cost = 37.62 := by
  sorry


end NUMINAMATH_CALUDE_august_electricity_bill_l3929_392976


namespace NUMINAMATH_CALUDE_max_value_z_l3929_392908

/-- The maximum value of z given the constraints -/
theorem max_value_z (x y : ℝ) (h1 : x - y ≤ 0) (h2 : 4 * x - y ≥ 0) (h3 : x + y ≤ 3) :
  ∃ (z : ℝ), z = x + 2 * y - 1 / x ∧ z ≤ 4 ∧ ∀ (w : ℝ), w = x + 2 * y - 1 / x → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l3929_392908


namespace NUMINAMATH_CALUDE_inequality_preservation_l3929_392932

theorem inequality_preservation (x y : ℝ) (h : x > y) : x/2 > y/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3929_392932


namespace NUMINAMATH_CALUDE_sunway_taihulight_performance_l3929_392970

theorem sunway_taihulight_performance :
  (12.5 * (10^12 : ℝ)) = (1.25 * (10^13 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sunway_taihulight_performance_l3929_392970


namespace NUMINAMATH_CALUDE_workshop_salary_calculation_l3929_392963

/-- Calculates the average salary of non-technician workers in a workshop --/
theorem workshop_salary_calculation 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_technicians : ℚ) 
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : avg_salary_all = 850)
  (h4 : avg_salary_technicians = 1000) :
  let non_technicians := total_workers - technicians
  let total_salary := avg_salary_all * total_workers
  let technicians_salary := avg_salary_technicians * technicians
  let non_technicians_salary := total_salary - technicians_salary
  non_technicians_salary / non_technicians = 780 := by
  sorry


end NUMINAMATH_CALUDE_workshop_salary_calculation_l3929_392963


namespace NUMINAMATH_CALUDE_divides_n_squared_plus_one_l3929_392903

theorem divides_n_squared_plus_one (n : ℕ) : 
  (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divides_n_squared_plus_one_l3929_392903


namespace NUMINAMATH_CALUDE_prime_sum_difference_l3929_392920

theorem prime_sum_difference (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → p > 0 → q > 0 →
  p + p^2 + p^4 - q - q^2 - q^4 = 83805 →
  p = 17 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_difference_l3929_392920


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3929_392959

/-- A geometric sequence with a_1 = 1 and a_5 = 4 has a_3 = 2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  a 1 = 1 →
  a 5 = 4 →
  a 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3929_392959


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l3929_392957

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The condition that the function must satisfy for all positive integers x and y -/
def SatisfiesCondition (f : PositiveIntFunction) : Prop :=
  ∀ x y : ℕ+, ∃ k : ℕ, (x : ℤ)^2 - (y : ℤ)^2 + 2*(y : ℤ)*((f x : ℤ) + (f y : ℤ)) = (k : ℤ)^2

/-- The theorem stating that the identity function is the only function satisfying the condition -/
theorem unique_satisfying_function :
  ∃! f : PositiveIntFunction, SatisfiesCondition f ∧ ∀ n : ℕ+, f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l3929_392957


namespace NUMINAMATH_CALUDE_age_problem_contradiction_l3929_392961

/-- Demonstrates the contradiction in the given age problem -/
theorem age_problem_contradiction (A B C D : ℕ) : 
  (A + B = B + C + 11) →  -- Condition 1
  (A + B + D = B + C + D + 8) →  -- Condition 2
  (A + C = 2 * D) →  -- Condition 3
  False := by sorry


end NUMINAMATH_CALUDE_age_problem_contradiction_l3929_392961


namespace NUMINAMATH_CALUDE_bbq_ice_packs_l3929_392939

/-- Given a BBQ scenario, calculate the number of 1-pound bags of ice in a pack -/
theorem bbq_ice_packs (people : ℕ) (ice_per_person : ℕ) (pack_price : ℚ) (total_spent : ℚ) :
  people = 15 →
  ice_per_person = 2 →
  pack_price = 3 →
  total_spent = 9 →
  (people * ice_per_person) / (total_spent / pack_price) = 10 := by
  sorry

#check bbq_ice_packs

end NUMINAMATH_CALUDE_bbq_ice_packs_l3929_392939


namespace NUMINAMATH_CALUDE_min_value_M_min_value_expression_min_value_equality_condition_l3929_392966

open Real

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part I
theorem min_value_M : ∃ (M : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ ∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → M ≤ m :=
sorry

-- Part II
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) ≥ 2 :=
sorry

theorem min_value_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) = 2 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_M_min_value_expression_min_value_equality_condition_l3929_392966


namespace NUMINAMATH_CALUDE_box_with_balls_l3929_392979

theorem box_with_balls (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end NUMINAMATH_CALUDE_box_with_balls_l3929_392979


namespace NUMINAMATH_CALUDE_linear_system_solution_ratio_l3929_392953

/-- Given a system of linear equations with parameter k:
    x + ky + 3z = 0
    3x + ky - 2z = 0
    x + 6y - 5z = 0
    which has a nontrivial solution where x, y, z are all non-zero,
    prove that yz/x^2 = 2/3 -/
theorem linear_system_solution_ratio (k : ℝ) (x y z : ℝ) :
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  x + 6*y - 5*z = 0 →
  y*z / (x^2) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_ratio_l3929_392953


namespace NUMINAMATH_CALUDE_shape_reassembly_l3929_392934

/-- Represents a geometric shape with an area -/
structure Shape :=
  (area : ℝ)

/-- Represents the original rectangle -/
def rectangle : Shape :=
  { area := 1 }

/-- Represents the square -/
def square : Shape :=
  { area := 0.5 }

/-- Represents the triangle with a hole -/
def triangleWithHole : Shape :=
  { area := 0.5 }

/-- Represents the two parts after cutting the rectangle -/
def part1 : Shape :=
  { area := 0.5 }

def part2 : Shape :=
  { area := 0.5 }

theorem shape_reassembly :
  (rectangle.area = part1.area + part2.area) ∧
  (square.area = part1.area) ∧
  (triangleWithHole.area = part2.area) := by
  sorry

#check shape_reassembly

end NUMINAMATH_CALUDE_shape_reassembly_l3929_392934


namespace NUMINAMATH_CALUDE_middle_number_problem_l3929_392950

theorem middle_number_problem (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 24)
  (h_sum2 : x + z = 29)
  (h_sum3 : y + z = 34) :
  y = 14.5 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l3929_392950


namespace NUMINAMATH_CALUDE_diophantine_equation_properties_l3929_392990

theorem diophantine_equation_properties :
  (∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 365) ∧
  (∀ (n : ℕ+), n > 370 → ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = n) ∧
  (¬ ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 370) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_properties_l3929_392990


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3929_392960

/-- Given a triangle ABC where C = π/3, b = √2, and c = √3, prove that angle A = 5π/12 -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  C = π/3 → b = Real.sqrt 2 → c = Real.sqrt 3 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  A = 5*π/12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3929_392960


namespace NUMINAMATH_CALUDE_cube_root_of_point_on_line_l3929_392978

/-- For any point (a, b) on the graph of y = x - 1, the cube root of b - a is -1 -/
theorem cube_root_of_point_on_line (a b : ℝ) (h : b = a - 1) : 
  (b - a : ℝ) ^ (1/3 : ℝ) = -1 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_point_on_line_l3929_392978


namespace NUMINAMATH_CALUDE_art_fair_sales_l3929_392916

theorem art_fair_sales (total_customers : ℕ) (two_painting_buyers : ℕ) 
  (one_painting_buyers : ℕ) (four_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_customers = 20 →
  one_painting_buyers = 12 →
  four_painting_buyers = 4 →
  total_paintings_sold = 36 →
  two_painting_buyers + one_painting_buyers + four_painting_buyers = total_customers →
  2 * two_painting_buyers + one_painting_buyers + 4 * four_painting_buyers = total_paintings_sold →
  two_painting_buyers = 4 := by
sorry

end NUMINAMATH_CALUDE_art_fair_sales_l3929_392916


namespace NUMINAMATH_CALUDE_ratio_c_d_equals_one_over_320_l3929_392926

theorem ratio_c_d_equals_one_over_320 (a b c d : ℝ) : 
  8 = 0.02 * a → 
  2 = 0.08 * b → 
  d = 0.05 * a → 
  c = b / a → 
  c / d = 1 / 320 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_d_equals_one_over_320_l3929_392926


namespace NUMINAMATH_CALUDE_min_numbers_for_five_ones_digit_count_for_five_ones_l3929_392946

/-- Represents the sequence of digits when writing consecutive natural numbers -/
def digit_sequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list contains five consecutive ones -/
def has_five_consecutive_ones (l : List ℕ) : Prop :=
  sorry

/-- Counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  sorry

/-- Counts the total number of digits when writing the first n natural numbers -/
def total_digit_count (n : ℕ) : ℕ :=
  sorry

theorem min_numbers_for_five_ones :
  ∃ n : ℕ, n ≤ 112 ∧ has_five_consecutive_ones (digit_sequence n) ∧
  ∀ m : ℕ, m < n → ¬has_five_consecutive_ones (digit_sequence m) :=
sorry

theorem digit_count_for_five_ones :
  total_digit_count 112 = 228 :=
sorry

end NUMINAMATH_CALUDE_min_numbers_for_five_ones_digit_count_for_five_ones_l3929_392946


namespace NUMINAMATH_CALUDE_roots_sum_of_cubes_reciprocal_l3929_392969

theorem roots_sum_of_cubes_reciprocal (a b c : ℝ) (r s : ℂ) 
  (hr : a * r^2 + b * r - c = 0) 
  (hs : a * s^2 + b * s - c = 0) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (b^3 + 3*a*b*c) / c^3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_cubes_reciprocal_l3929_392969
