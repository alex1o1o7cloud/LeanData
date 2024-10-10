import Mathlib

namespace intersection_of_M_and_N_l249_24910

def M : Set ℕ := {x : ℕ | 0 < x ∧ x < 4}
def N : Set ℕ := {x : ℕ | 1 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l249_24910


namespace probability_equals_frequency_l249_24942

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + yellow_balls

/-- Represents the observed limiting frequency from the experiment -/
def observed_frequency : ℚ := 2/5

/-- Theorem stating that the probability of selecting a red ball equals the observed frequency -/
theorem probability_equals_frequency : 
  (red_balls : ℚ) / (total_balls : ℚ) = observed_frequency :=
sorry

end probability_equals_frequency_l249_24942


namespace peach_basket_ratios_and_percentages_l249_24954

/-- Represents the number of peaches of each color in the basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of peaches in the basket -/
def totalPeaches (basket : PeachBasket) : ℕ :=
  basket.red + basket.yellow + basket.green + basket.orange

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of a specific color to the total -/
def colorRatio (count : ℕ) (total : ℕ) : Ratio :=
  let gcd := Nat.gcd count total
  { numerator := count / gcd, denominator := total / gcd }

/-- Calculates the percentage of a specific color -/
def colorPercentage (count : ℕ) (total : ℕ) : Float :=
  (count.toFloat / total.toFloat) * 100

theorem peach_basket_ratios_and_percentages
  (basket : PeachBasket)
  (h_red : basket.red = 8)
  (h_yellow : basket.yellow = 14)
  (h_green : basket.green = 6)
  (h_orange : basket.orange = 4) :
  let total := totalPeaches basket
  (colorRatio basket.green total = Ratio.mk 3 16) ∧
  (colorRatio basket.yellow total = Ratio.mk 7 16) ∧
  (colorPercentage basket.green total = 18.75) ∧
  (colorPercentage basket.yellow total = 43.75) := by
  sorry


end peach_basket_ratios_and_percentages_l249_24954


namespace calculate_dime_piles_l249_24902

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in cents -/
def total_value : ℕ := 2100

/-- Calculates the number of piles of dimes given the conditions -/
theorem calculate_dime_piles : 
  ∃ (dime_piles : ℕ),
    dime_piles * coins_per_pile * dime_value + 
    quarter_piles * coins_per_pile * quarter_value +
    nickel_piles * coins_per_pile * nickel_value +
    penny_piles * coins_per_pile * penny_value = total_value ∧
    dime_piles = 6 := by
  sorry

end calculate_dime_piles_l249_24902


namespace min_median_length_l249_24927

/-- In a right triangle with height h dropped onto the hypotenuse,
    the minimum length of the median that bisects the longer leg is (3/2) * h. -/
theorem min_median_length (h : ℝ) (h_pos : h > 0) :
  ∃ (m : ℝ), m ≥ (3/2) * h ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = h^2 →
    ((x/2 + y)^2 + (h/2)^2).sqrt ≥ m :=
by sorry

end min_median_length_l249_24927


namespace mean_equality_implies_sum_l249_24908

theorem mean_equality_implies_sum (x y : ℝ) : 
  (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3 → x + y = 26.5 := by
sorry

end mean_equality_implies_sum_l249_24908


namespace abc_product_values_l249_24976

theorem abc_product_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 8/3) :
  a * b * c = 1 ∨ a * b * c = 37/3 := by
  sorry

end abc_product_values_l249_24976


namespace problem_statement_l249_24925

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7 where f(2) = -9, prove that f(-2) = -5 -/
theorem problem_statement (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -a * x^5 - x^3 + b * x - 7)
  (h2 : f 2 = -9) : 
  f (-2) = -5 := by sorry

end problem_statement_l249_24925


namespace greatest_integer_with_gcf_three_exists_141_max_141_solution_l249_24994

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem exists_141 : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem max_141 : ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ 141 :=
by
  sorry

theorem solution : (∃ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n) ∧
                   (∀ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n → n = 141) :=
by
  sorry

end greatest_integer_with_gcf_three_exists_141_max_141_solution_l249_24994


namespace max_gcd_of_eight_numbers_sum_595_l249_24956

/-- The maximum possible GCD of eight natural numbers summing to 595 -/
theorem max_gcd_of_eight_numbers_sum_595 :
  ∃ (a b c d e f g h : ℕ),
    a + b + c + d + e + f + g + h = 595 ∧
    ∀ (k : ℕ),
      k ∣ a ∧ k ∣ b ∧ k ∣ c ∧ k ∣ d ∧ k ∣ e ∧ k ∣ f ∧ k ∣ g ∧ k ∣ h →
      k ≤ 35 :=
by sorry

end max_gcd_of_eight_numbers_sum_595_l249_24956


namespace quadratic_derivative_bound_l249_24981

theorem quadratic_derivative_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
  sorry

end quadratic_derivative_bound_l249_24981


namespace min_value_inequality_l249_24996

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 :=
sorry

end min_value_inequality_l249_24996


namespace intersection_complement_M_and_N_l249_24916

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end intersection_complement_M_and_N_l249_24916


namespace special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l249_24985

/-- An equilateral hexagon with specific angle and area properties -/
structure SpecialHexagon where
  -- The hexagon is equilateral
  equilateral : Bool
  -- Alternating interior angles are 120° and 60°
  alternating_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- Conditions on the hexagon
  h_equilateral : equilateral = true
  h_alternating_angles : alternating_angles = true
  h_area : area = 12

/-- The perimeter of a SpecialHexagon is 4√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : ℝ :=
  4 * Real.sqrt 6

/-- The perimeter of a SpecialHexagon with area 12 is 4√6 -/
theorem special_hexagon_perimeter_is_4_sqrt_6 (h : SpecialHexagon) :
  special_hexagon_perimeter h = 4 * Real.sqrt 6 := by
  sorry

#check special_hexagon_perimeter_is_4_sqrt_6

end special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l249_24985


namespace max_value_polynomial_l249_24915

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 656^2 / 18 :=
by sorry

end max_value_polynomial_l249_24915


namespace zeros_of_f_l249_24977

def f (x : ℝ) : ℝ := (x - 1) * (x^2 - 2*x - 3)

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {1, -1, 3} := by sorry

end zeros_of_f_l249_24977


namespace sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l249_24939

theorem sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1 :
  0 < Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) ∧
  Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) < 1 := by
  sorry

end sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l249_24939


namespace binomial_coefficient_times_n_l249_24943

theorem binomial_coefficient_times_n (n : ℕ+) : n * Nat.choose 4 3 = 4 * n := by
  sorry

end binomial_coefficient_times_n_l249_24943


namespace expansion_of_5000_power_150_l249_24904

theorem expansion_of_5000_power_150 :
  ∃ (n : ℕ), 5000^150 = n * 10^450 ∧ 1 ≤ n ∧ n < 10 :=
by
  sorry

end expansion_of_5000_power_150_l249_24904


namespace complex_equation_solution_l249_24997

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + 2 * Complex.I * z = (4 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 8 := by
  sorry

end complex_equation_solution_l249_24997


namespace alex_cell_phone_cost_l249_24948

/-- Calculates the total cost of a cell phone plan -/
def calculate_total_cost (base_cost : ℚ) (cost_per_text : ℚ) (cost_per_extra_minute : ℚ) 
  (included_hours : ℕ) (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_cost := cost_per_text * texts_sent
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost := cost_per_extra_minute * extra_minutes
  base_cost + text_cost + extra_minutes_cost

/-- The total cost of Alex's cell phone plan is $48.50 -/
theorem alex_cell_phone_cost : 
  calculate_total_cost 20 (7/100) (15/100) 30 150 32 = 485/10 := by
  sorry

end alex_cell_phone_cost_l249_24948


namespace division_fraction_problem_l249_24978

theorem division_fraction_problem : (1 / 60) / ((2 / 3) - (1 / 5) - (2 / 5)) = 1 / 4 := by
  sorry

end division_fraction_problem_l249_24978


namespace bob_hair_growth_time_l249_24906

/-- Represents the growth of Bob's hair over time -/
def hair_growth (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * time

/-- Theorem stating the time it takes for Bob's hair to grow from 6 inches to 36 inches -/
theorem bob_hair_growth_time :
  let initial_length : ℝ := 6
  let final_length : ℝ := 36
  let monthly_growth_rate : ℝ := 0.5
  let years : ℝ := 5
  hair_growth initial_length (monthly_growth_rate * 12) years = final_length := by
  sorry

#check bob_hair_growth_time

end bob_hair_growth_time_l249_24906


namespace intersection_M_complement_N_l249_24912

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | (3 - x) / (x + 1) > 0}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.compl N) = {x : ℝ | x < -2 ∨ x ≥ 3} := by sorry

end intersection_M_complement_N_l249_24912


namespace triangle_point_distance_l249_24965

/-- Given a triangle ABC with AB = 8, BC = 20, CA = 16, and points D and E on BC
    such that CD = 8 and ∠BAE = ∠CAD, prove that BE = 2 -/
theorem triangle_point_distance (A B C D E : ℝ × ℝ) : 
  let dist := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  let angle := (fun (P Q R : ℝ × ℝ) => Real.arccos (((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
                (dist P Q * dist P R)))
  dist A B = 8 →
  dist B C = 20 →
  dist C A = 16 →
  D.1 = B.1 + 12 / 20 * (C.1 - B.1) ∧ D.2 = B.2 + 12 / 20 * (C.2 - B.2) →
  angle B A E = angle C A D →
  dist B E = 2 := by
sorry


end triangle_point_distance_l249_24965


namespace factorization_x4_minus_1_l249_24933

theorem factorization_x4_minus_1 (x : ℂ) : x^4 - 1 = (x + Complex.I) * (x - Complex.I) * (x - 1) * (x + 1) := by
  sorry

end factorization_x4_minus_1_l249_24933


namespace bicycle_journey_l249_24924

theorem bicycle_journey (t₅ t₁₅ : ℝ) (h_positive : t₅ > 0 ∧ t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 → t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
  sorry

end bicycle_journey_l249_24924


namespace selling_price_proof_l249_24982

/-- The selling price that results in the same profit as the loss -/
def selling_price : ℕ := 66

/-- The cost price of the article -/
def cost_price : ℕ := 59

/-- The price at which the article is sold at a loss -/
def loss_price : ℕ := 52

theorem selling_price_proof :
  (selling_price - cost_price = cost_price - loss_price) ∧
  (selling_price > cost_price) ∧
  (loss_price < cost_price) := by
  sorry

end selling_price_proof_l249_24982


namespace graph_x_squared_minus_y_squared_l249_24980

/-- The graph of x^2 - y^2 = 0 consists of two intersecting lines in the real plane -/
theorem graph_x_squared_minus_y_squared (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) :=
sorry

end graph_x_squared_minus_y_squared_l249_24980


namespace trig_identity_l249_24926

theorem trig_identity (α : ℝ) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := by sorry

end trig_identity_l249_24926


namespace sin_alpha_for_point_on_terminal_side_l249_24945

/-- 
If the terminal side of angle α passes through point P(m, 2m) where m > 0, 
then sin(α) = 2√5/5.
-/
theorem sin_alpha_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x = m ∧ y = 2*m ∧ 
       x = Real.cos α * Real.sqrt (m^2 + (2*m)^2) ∧ 
       y = Real.sin α * Real.sqrt (m^2 + (2*m)^2)) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end sin_alpha_for_point_on_terminal_side_l249_24945


namespace range_of_m_l249_24973

/-- Given a quadratic inequality and a function with specific domain,
    prove that the range of m is [-1, 0] -/
theorem range_of_m (a : ℝ) (m : ℝ) : 
  (a > 0 ∧ a ≠ 1) →
  (∀ x : ℝ, a * x^2 - a * x - 2 * a^2 > 1 ↔ -a < x ∧ x < 2*a) →
  (∀ x : ℝ, (1/a)^(x^2 + 2*m*x - m) - 1 ≥ 0) →
  m ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end range_of_m_l249_24973


namespace unique_modular_solution_l249_24991

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end unique_modular_solution_l249_24991


namespace arithmetic_progression_cos_sum_l249_24923

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_cos_sum (a : ℕ → ℝ) :
  is_arithmetic_progression a →
  a 1 + a 7 + a 13 = 4 * Real.pi →
  Real.cos (a 2 + a 12) = -1/2 := by
  sorry

end arithmetic_progression_cos_sum_l249_24923


namespace binary_decimal_octal_equivalence_l249_24993

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_decimal_octal_equivalence :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let decimal := 11
  let octal := [1, 3]
  binary_to_decimal binary = decimal ∧
  decimal_to_octal decimal = octal :=
by sorry

end binary_decimal_octal_equivalence_l249_24993


namespace lindas_coins_l249_24957

theorem lindas_coins (total_coins : ℕ) (nickel_value dime_value : ℚ) 
  (swap_increase : ℚ) (h1 : total_coins = 30) 
  (h2 : nickel_value = 5/100) (h3 : dime_value = 10/100)
  (h4 : swap_increase = 90/100) : ∃ (nickels : ℕ), 
  nickels * nickel_value + (total_coins - nickels) * dime_value = 180/100 := by
  sorry

end lindas_coins_l249_24957


namespace expression_exists_l249_24909

/-- Represents an expression formed by ones and operators --/
inductive Expression
  | One : Expression
  | Add : Expression → Expression → Expression
  | Mul : Expression → Expression → Expression

/-- Evaluates the expression --/
def evaluate : Expression → ℕ
  | Expression.One => 1
  | Expression.Add e1 e2 => evaluate e1 + evaluate e2
  | Expression.Mul e1 e2 => evaluate e1 * evaluate e2

/-- Swaps the operators in the expression --/
def swap_operators : Expression → Expression
  | Expression.One => Expression.One
  | Expression.Add e1 e2 => Expression.Mul (swap_operators e1) (swap_operators e2)
  | Expression.Mul e1 e2 => Expression.Add (swap_operators e1) (swap_operators e2)

/-- Theorem stating the existence of the required expression --/
theorem expression_exists : ∃ (e : Expression), 
  evaluate e = 2014 ∧ evaluate (swap_operators e) = 2014 := by
  sorry


end expression_exists_l249_24909


namespace line_equation_problem_l249_24922

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m

/-- Point in ℝ² -/
def Point := ℝ × ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (line : Set Point) : Point := sorry

/-- The problem statement -/
theorem line_equation_problem (lines : TwoLines) 
  (h1 : (0, 0) ∈ lines.ℓ ∩ lines.m)
  (h2 : ∀ x y, (x, y) ∈ lines.ℓ ↔ 3 * x + 4 * y = 0)
  (h3 : reflect (-2, 3) lines.ℓ = reflect (3, -2) lines.m) :
  ∀ x y, (x, y) ∈ lines.m ↔ 7 * x - 25 * y = 0 := by sorry

end line_equation_problem_l249_24922


namespace cosine_expression_simplification_fraction_simplification_l249_24938

-- Part 1
theorem cosine_expression_simplification :
  2 * Real.cos (45 * π / 180) - (-2 * Real.sqrt 3) ^ 0 + 1 / (Real.sqrt 2 + 1) - Real.sqrt 8 = -2 := by
  sorry

-- Part 2
theorem fraction_simplification (x : ℝ) (h : x = -Real.sqrt 2) :
  (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = Real.sqrt 2 := by
  sorry

end cosine_expression_simplification_fraction_simplification_l249_24938


namespace quartic_polynomial_e_value_l249_24983

/-- A polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  e : ℤ

/-- The sum of coefficients of the polynomial -/
def QuarticPolynomial.sumCoeffs (p : QuarticPolynomial) : ℤ :=
  p.a + p.b + p.c + p.e

/-- Predicate for a polynomial having all negative integer roots -/
def hasAllNegativeIntegerRoots (p : QuarticPolynomial) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ : ℕ+), 
    p.a = s₁ + s₂ + s₃ + s₄ ∧
    p.b = s₁*s₂ + s₁*s₃ + s₁*s₄ + s₂*s₃ + s₂*s₄ + s₃*s₄ ∧
    p.c = s₁*s₂*s₃ + s₁*s₂*s₄ + s₁*s₃*s₄ + s₂*s₃*s₄ ∧
    p.e = s₁*s₂*s₃*s₄

theorem quartic_polynomial_e_value (p : QuarticPolynomial) 
  (h1 : hasAllNegativeIntegerRoots p) 
  (h2 : p.sumCoeffs = 2023) : 
  p.e = 1540 := by
  sorry

end quartic_polynomial_e_value_l249_24983


namespace function_max_at_pi_third_l249_24919

/-- The function f(x) reaches its maximum value at x₀ = π/3 --/
theorem function_max_at_pi_third (x : ℝ) : 
  let f := λ x : ℝ => (Real.sqrt 3 / 2) * Real.sin x + (1 / 2) * Real.cos x
  ∃ (x₀ : ℝ), x₀ = π/3 ∧ ∀ y, f y ≤ f x₀ :=
by sorry

end function_max_at_pi_third_l249_24919


namespace same_color_probability_l249_24961

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 30

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 8

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 9

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 10

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 2

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating that the probability of rolling the same color on both dice is 25/90 -/
theorem same_color_probability :
  (purpleSides^2 + orangeSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 25 / 90 := by
  sorry


end same_color_probability_l249_24961


namespace union_M_N_equals_geq_one_l249_24905

-- Define set M
def M : Set ℝ := {x | x - 2 > 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- Theorem statement
theorem union_M_N_equals_geq_one : M ∪ N = {x | x ≥ 1} := by sorry

end union_M_N_equals_geq_one_l249_24905


namespace system_of_equations_solution_l249_24958

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * (x + 2 * y) - 5 * y = -1) ∧ (3 * (x - y) + y = 2) ∧ (x = -4) ∧ (y = -7) := by
  sorry

end system_of_equations_solution_l249_24958


namespace f_properties_l249_24979

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

theorem f_properties (a : ℝ) :
  (f_deriv a 1 = 1) →
  (a = 2) ∧
  (∃ m b : ℝ, m = 9 ∧ b = 3 ∧ ∀ x y : ℝ, y = f a x → m*(-1) - y + b = 0) :=
by sorry

end f_properties_l249_24979


namespace sum_of_sqrt_inequality_l249_24920

theorem sum_of_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  ∃ (c : ℝ), c = 7 ∧ 
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ c) ∧
  (∀ (c' : ℝ), c' < c → 
    ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 
      Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) > c') :=
by sorry

end sum_of_sqrt_inequality_l249_24920


namespace square_root_squared_l249_24989

theorem square_root_squared (a : ℝ) (ha : 0 ≤ a) : (Real.sqrt a) ^ 2 = a := by
  sorry

end square_root_squared_l249_24989


namespace hall_volume_l249_24955

theorem hall_volume (length width : ℝ) (h : ℝ) : 
  length = 6 ∧ width = 6 ∧ 2 * (length * width) = 2 * (length * h) + 2 * (width * h) → 
  length * width * h = 108 := by
sorry

end hall_volume_l249_24955


namespace sum_and_square_difference_implies_difference_l249_24953

theorem sum_and_square_difference_implies_difference
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) :
  x - y = 19 := by
  sorry

end sum_and_square_difference_implies_difference_l249_24953


namespace plane_through_points_l249_24968

def point1 : ℝ × ℝ × ℝ := (2, -3, 5)
def point2 : ℝ × ℝ × ℝ := (4, -3, 6)
def point3 : ℝ × ℝ × ℝ := (6, -4, 8)

def plane_equation (x y z : ℝ) : ℝ := x - 2*y + 2*z - 18

theorem plane_through_points :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd 1 2) (Nat.gcd 2 18) = 1) := by
  sorry

end plane_through_points_l249_24968


namespace quadratic_properties_l249_24999

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x, x > 1 → ∀ y, y > x → f y < f x) := by
  sorry

end quadratic_properties_l249_24999


namespace trapezoid_area_division_l249_24987

/-- Represents a trapezoid with a diagonal and a parallel line -/
structure Trapezoid where
  /-- The ratio in which the diagonal divides the area -/
  diagonal_ratio : Rat
  /-- The ratio in which the parallel line divides the area -/
  parallel_line_ratio : Rat

/-- Theorem about area division in a specific trapezoid -/
theorem trapezoid_area_division (T : Trapezoid) 
  (h : T.diagonal_ratio = 3 / 7) : 
  T.parallel_line_ratio = 3 / 2 := by
  sorry

end trapezoid_area_division_l249_24987


namespace set_equality_l249_24930

def S : Set ℕ := {x | ∃ k : ℤ, 12 = k * (6 - x)}

theorem set_equality : S = {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} := by sorry

end set_equality_l249_24930


namespace barry_vitamin_d3_days_l249_24940

/-- Calculates the number of days Barry was told to take vitamin D3 -/
def vitaminD3Days (capsules_per_bottle : ℕ) (capsules_per_day : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / capsules_per_day) * bottles_needed

theorem barry_vitamin_d3_days :
  vitaminD3Days 60 2 6 = 180 := by
  sorry

end barry_vitamin_d3_days_l249_24940


namespace tangent_line_equation_l249_24946

/-- The curve C defined by y = x^3 -/
def C : ℝ → ℝ := fun x ↦ x^3

/-- The point P through which the tangent line passes -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (a b c : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- The tangent line to curve C at point (x₀, C x₀) -/
def tangent_line (x₀ : ℝ) : ℝ → ℝ := fun x ↦ C x₀ + (3 * x₀^2) * (x - x₀)

theorem tangent_line_equation :
  ∃ x₀ : ℝ, 
    tangent_line x₀ P.1 = P.2 ∧ 
    ¬passes_through_fourth_quadrant 3 (-4) 1 ∧
    ∀ x, tangent_line x₀ x = 3*x - 4*(C x) + 1 := by
  sorry

end tangent_line_equation_l249_24946


namespace melinda_doughnuts_count_l249_24903

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of coffees Melinda bought -/
def melinda_coffees : ℕ := 6

/-- The number of doughnuts Melinda bought -/
def melinda_doughnuts : ℕ := 5

theorem melinda_doughnuts_count : 
  ∃ (coffee_cost : ℚ), 
    (harold_doughnuts : ℚ) * doughnut_cost + (harold_coffees : ℚ) * coffee_cost = harold_total ∧
    (melinda_doughnuts : ℚ) * doughnut_cost + (melinda_coffees : ℚ) * coffee_cost = melinda_total :=
by sorry

end melinda_doughnuts_count_l249_24903


namespace perfect_square_sums_l249_24931

theorem perfect_square_sums : ∃ (x y : ℕ+), 
  ∃ (a b c : ℕ+),
  (x + y : ℕ) = a^2 ∧
  (x^2 + y^2 : ℕ) = b^2 ∧
  (x^3 + y^3 : ℕ) = c^2 := by
  sorry

end perfect_square_sums_l249_24931


namespace parabola_segment_length_l249_24934

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Length of PQ on a parabola -/
theorem parabola_segment_length
  (p : ℝ)
  (hp : p > 0)
  (P Q : ParabolaPoint)
  (hP : P.y^2 = 2*p*P.x)
  (hQ : Q.y^2 = 2*p*Q.x)
  (h_sum : P.x + Q.x = 3*p) :
  |P.x - Q.x| + p = 4*p :=
by sorry

end parabola_segment_length_l249_24934


namespace hadley_books_l249_24971

theorem hadley_books (initial_books : ℕ) 
  (h1 : initial_books - 50 + 40 - 30 = 60) : initial_books = 100 := by
  sorry

end hadley_books_l249_24971


namespace clock_angle_at_3_15_l249_24936

/-- The angle in degrees that the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle in degrees that the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The starting position of the hour hand at 3:00 in degrees -/
def hour_hand_start : ℝ := 90

/-- The number of minutes past 3:00 -/
def minutes_past : ℝ := 15

/-- Calculates the position of the hour hand at 3:15 -/
def hour_hand_position : ℝ := hour_hand_start + hour_hand_speed * minutes_past

/-- Calculates the position of the minute hand at 3:15 -/
def minute_hand_position : ℝ := minute_hand_speed * minutes_past

/-- The acute angle between the hour hand and minute hand at 3:15 -/
def clock_angle : ℝ := |hour_hand_position - minute_hand_position|

theorem clock_angle_at_3_15 : clock_angle = 7.5 := by sorry

end clock_angle_at_3_15_l249_24936


namespace interest_rate_is_five_paise_l249_24972

/-- Calculates the interest rate in paise per rupee per month given the principal, time, and simple interest -/
def interest_rate_paise (principal : ℚ) (time_months : ℚ) (simple_interest : ℚ) : ℚ :=
  (simple_interest / (principal * time_months)) * 100

/-- Theorem stating that for the given conditions, the interest rate is 5 paise per rupee per month -/
theorem interest_rate_is_five_paise 
  (principal : ℚ) 
  (time_months : ℚ) 
  (simple_interest : ℚ) 
  (h1 : principal = 20)
  (h2 : time_months = 6)
  (h3 : simple_interest = 6) :
  interest_rate_paise principal time_months simple_interest = 5 := by
  sorry

end interest_rate_is_five_paise_l249_24972


namespace rationalize_denominator_l249_24911

theorem rationalize_denominator :
  36 / Real.sqrt 7 = (36 * Real.sqrt 7) / 7 := by sorry

end rationalize_denominator_l249_24911


namespace equation_solution_l249_24974

theorem equation_solution (α β : ℝ) : 
  (∀ x : ℝ, x ≠ -β → x ≠ 30 → x ≠ 70 → 
    (x - α) / (x + β) = (x^2 + 120*x + 1575) / (x^2 - 144*x + 1050)) →
  α + β = 5 := by sorry

end equation_solution_l249_24974


namespace solve_equation_l249_24921

theorem solve_equation (a : ℚ) : a + a / 3 = 8 / 3 → a = 2 := by
  sorry

end solve_equation_l249_24921


namespace min_value_sum_squares_l249_24975

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + Real.sqrt 3 * z = 1) : 
  ∀ (a b c : ℝ), a^2 + b^2 + c^2 ≥ (1/8 : ℝ) ∧ 
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = (1/8 : ℝ) ∧ x₀ + 2*y₀ + Real.sqrt 3 * z₀ = 1) :=
by sorry

end min_value_sum_squares_l249_24975


namespace fold_cut_result_l249_24986

/-- Represents the possible number of parts after cutting a folded square --/
inductive CutResult
  | OppositeMiddle : CutResult
  | AdjacentMiddle : CutResult

/-- Represents the dimensions of the original rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the result of folding and cutting the rectangle --/
def fold_and_cut (rect : Rectangle) (cut : CutResult) : Set ℕ :=
  match cut with
  | CutResult.OppositeMiddle => {11, 13}
  | CutResult.AdjacentMiddle => {31, 36, 37, 43}

/-- Theorem stating the result of folding and cutting the specific rectangle --/
theorem fold_cut_result (rect : Rectangle) (h1 : rect.width = 10) (h2 : rect.height = 12) :
  (fold_and_cut rect CutResult.OppositeMiddle = {11, 13}) ∧
  (fold_and_cut rect CutResult.AdjacentMiddle = {31, 36, 37, 43}) := by
  sorry

#check fold_cut_result

end fold_cut_result_l249_24986


namespace snake_owners_count_l249_24964

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  onlyBirds : Nat
  onlySnakes : Nat
  catsAndDogs : Nat
  dogsAndBirds : Nat
  catsAndBirds : Nat
  catsAndSnakes : Nat
  dogsAndSnakes : Nat
  allCategories : Nat

/-- Calculates the total number of snake owners --/
def totalSnakeOwners (po : PetOwners) : Nat :=
  po.onlySnakes + po.catsAndSnakes + po.dogsAndSnakes + po.allCategories

/-- Theorem stating that the total number of snake owners is 25 --/
theorem snake_owners_count (po : PetOwners) 
  (h1 : po.total = 75)
  (h2 : po.onlyDogs = 20)
  (h3 : po.onlyCats = 15)
  (h4 : po.onlyBirds = 8)
  (h5 : po.onlySnakes = 10)
  (h6 : po.catsAndDogs = 5)
  (h7 : po.dogsAndBirds = 4)
  (h8 : po.catsAndBirds = 3)
  (h9 : po.catsAndSnakes = 7)
  (h10 : po.dogsAndSnakes = 6)
  (h11 : po.allCategories = 2) :
  totalSnakeOwners po = 25 := by
  sorry

end snake_owners_count_l249_24964


namespace stream_speed_l249_24990

/-- Given a boat with speed 78 kmph in still water, if the time taken upstream is twice
    the time taken downstream, then the speed of the stream is 26 kmph. -/
theorem stream_speed (D : ℝ) (D_pos : D > 0) : 
  let boat_speed : ℝ := 78
  let stream_speed : ℝ := 26
  (D / (boat_speed - stream_speed) = 2 * (D / (boat_speed + stream_speed))) →
  stream_speed = 26 := by
sorry

end stream_speed_l249_24990


namespace paper_towel_case_rolls_l249_24967

/-- The number of rolls in a case of paper towels -/
def number_of_rolls : ℕ := 12

/-- The price of the case in dollars -/
def case_price : ℚ := 9

/-- The price of an individual roll in dollars -/
def individual_roll_price : ℚ := 1

/-- The savings percentage per roll when buying the case -/
def savings_percentage : ℚ := 25 / 100

theorem paper_towel_case_rolls :
  case_price = number_of_rolls * (individual_roll_price * (1 - savings_percentage)) :=
sorry

end paper_towel_case_rolls_l249_24967


namespace ladder_problem_l249_24929

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (base_distance : ℝ), 
    base_distance^2 + height_on_wall^2 = ladder_length^2 ∧ 
    base_distance = 5 :=
by sorry

end ladder_problem_l249_24929


namespace max_incorrect_answers_is_correct_l249_24944

/-- The passing threshold for the exam as a percentage -/
def pass_threshold : ℝ := 85

/-- The total number of questions in the exam -/
def total_questions : ℕ := 50

/-- The maximum number of questions that can be answered incorrectly while still passing -/
def max_incorrect_answers : ℕ := 7

/-- Theorem stating that max_incorrect_answers is the maximum number of questions
    that can be answered incorrectly while still passing the exam -/
theorem max_incorrect_answers_is_correct :
  ∀ n : ℕ, 
    (n ≤ max_incorrect_answers ↔ 
      (total_questions - n : ℝ) / total_questions * 100 ≥ pass_threshold) :=
by sorry

end max_incorrect_answers_is_correct_l249_24944


namespace vector_problem_l249_24995

/-- Given vectors a and b, if vector c satisfies the conditions, then c equals the expected result. -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ (k : ℝ), c + a = k • b) → -- (c+a) ∥ b
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) → -- c ⟂ (a+b)
  c = (-7/9, -7/3) := by
sorry


end vector_problem_l249_24995


namespace watch_correction_theorem_l249_24935

/-- The number of minutes in a day -/
def minutes_per_day : ℚ := 24 * 60

/-- The number of minutes the watch loses per day -/
def minutes_lost_per_day : ℚ := 5/2

/-- The number of days between March 15 at 1 PM and March 21 at 9 AM -/
def days_elapsed : ℚ := 5 + 5/6

/-- The correct additional minutes to set the watch -/
def n : ℚ := 14 + 14/23

theorem watch_correction_theorem :
  n = (minutes_per_day / (minutes_per_day - minutes_lost_per_day) - 1) * (days_elapsed * minutes_per_day) :=
by sorry

end watch_correction_theorem_l249_24935


namespace arithmetic_progressions_in_S_l249_24932

def S : Set ℤ := {n : ℤ | ∃ k : ℕ, n = ⌊k * Real.pi⌋}

theorem arithmetic_progressions_in_S :
  (∀ k : ℕ, ∃ (a d : ℤ) (f : Fin k → ℤ), (∀ i : Fin k, f i ∈ S) ∧ 
    (∀ i : Fin k, f i = a + i.val * d)) ∧
  ¬(∃ (a d : ℤ) (f : ℕ → ℤ), (∀ n : ℕ, f n ∈ S) ∧ 
    (∀ n : ℕ, f (n + 1) - f n = d)) :=
by sorry

end arithmetic_progressions_in_S_l249_24932


namespace second_year_associates_percentage_l249_24900

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  not_first_year : ℝ
  more_than_two_years : ℝ

/-- The theorem stating that the percentage of second-year associates is 30% -/
theorem second_year_associates_percentage
  (ap : AssociatePercentages)
  (h1 : ap.not_first_year = 60)
  (h2 : ap.more_than_two_years = 30) :
  ap.not_first_year - ap.more_than_two_years = 30 := by
  sorry

end second_year_associates_percentage_l249_24900


namespace negative_510_in_third_quadrant_l249_24901

-- Define a function to normalize an angle to the range [-360°, 0°)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360 - 360

-- Define a function to determine the quadrant of an angle
def quadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if -270 < normalizedAngle && normalizedAngle ≤ -180 then 3
  else if -180 < normalizedAngle && normalizedAngle ≤ -90 then 2
  else if -90 < normalizedAngle && normalizedAngle ≤ 0 then 1
  else 4

-- Theorem: -510° is in the third quadrant
theorem negative_510_in_third_quadrant : quadrant (-510) = 3 := by
  sorry

end negative_510_in_third_quadrant_l249_24901


namespace smallest_debt_resolution_l249_24960

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt resolution is valid if it can be expressed as a combination of pigs and goats -/
def is_valid_resolution (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 800

theorem smallest_debt_resolution :
  (is_valid_resolution smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_valid_resolution d)) :=
by sorry

end smallest_debt_resolution_l249_24960


namespace solution_set_inequality_l249_24907

theorem solution_set_inequality (x : ℝ) :
  (x - 5) * (x + 1) > 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 5 :=
by sorry

end solution_set_inequality_l249_24907


namespace smallest_multiplier_for_four_zeros_l249_24984

theorem smallest_multiplier_for_four_zeros (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(10000 ∣ (975 * 935 * 972 * m))) →
  (10000 ∣ (975 * 935 * 972 * n)) →
  n = 20 := by
sorry

end smallest_multiplier_for_four_zeros_l249_24984


namespace scientific_notation_correct_l249_24914

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we're working with (1.12 million) -/
def number : ℝ := 1.12e6

/-- The claimed scientific notation of the number -/
def claimed_notation : ScientificNotation := {
  coefficient := 1.12,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the claimed notation is correct for the given number -/
theorem scientific_notation_correct : 
  number = claimed_notation.coefficient * (10 : ℝ) ^ claimed_notation.exponent := by sorry

end scientific_notation_correct_l249_24914


namespace multiply_and_add_l249_24913

theorem multiply_and_add : 45 * 21 + 45 * 79 = 4500 := by sorry

end multiply_and_add_l249_24913


namespace complex_modulus_problem_l249_24918

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l249_24918


namespace square_with_ascending_digits_l249_24928

theorem square_with_ascending_digits : ∃ n : ℕ, 
  (n^2).repr.takeRight 5 = "23456" ∧ 
  n^2 = 54563456 := by
  sorry

end square_with_ascending_digits_l249_24928


namespace intersection_point_of_lines_l249_24959

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (3 * x + 4 * y = 12) ↔ (x = 28/13 ∧ y = 18/13) :=
by sorry

end intersection_point_of_lines_l249_24959


namespace spacefarer_resources_sum_l249_24937

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem spacefarer_resources_sum :
  let crystal := base3_to_base10 [0, 2, 1, 2]
  let rare_metals := base3_to_base10 [2, 0, 1, 2]
  let alien_tech := base3_to_base10 [2, 0, 1]
  crystal + rare_metals + alien_tech = 145 := by
sorry

end spacefarer_resources_sum_l249_24937


namespace final_x_value_l249_24962

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (x : ℕ)
  (S : ℕ)

/-- The initial state of the program -/
def initial_state : ProgramState :=
  { x := 3, S := 0 }

/-- Updates the program state for one iteration -/
def update_state (state : ProgramState) : ProgramState :=
  { x := state.x + 2, S := state.S + (state.x + 2) }

/-- Predicate to check if the loop should continue -/
def continue_loop (state : ProgramState) : Prop :=
  state.S < 10000

/-- The final state of the program after all iterations -/
noncomputable def final_state : ProgramState :=
  sorry  -- The actual computation of the final state

/-- Theorem stating that the final x value is 201 -/
theorem final_x_value :
  (final_state.x = 201) ∧ (final_state.S ≥ 10000) ∧ 
  (update_state final_state).S > 10000 :=
sorry

end final_x_value_l249_24962


namespace ace_king_queen_probability_l249_24952

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def num_queens : ℕ := 4

theorem ace_king_queen_probability :
  (num_aces / standard_deck) * (num_kings / (standard_deck - 1)) * (num_queens / (standard_deck - 2)) = 16 / 33150 := by
  sorry

end ace_king_queen_probability_l249_24952


namespace divisors_of_180_l249_24949

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

def count_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_180 :
  (largest_prime_factor (sum_of_divisors 180) = 13) ∧
  (count_divisors 180 = 18) := by sorry

end divisors_of_180_l249_24949


namespace silver_car_percentage_l249_24988

/-- Calculates the percentage of silver cars in a car dealership's inventory after a new shipment. -/
theorem silver_car_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/10)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 1/4)
  : ∃ (result : ℚ), abs (result - 53333/100000) < 1/10000 ∧
    result = (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) :=
by sorry

end silver_car_percentage_l249_24988


namespace angle_subtraction_theorem_l249_24992

/-- Represents an angle in degrees and minutes -/
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Addition of angles in degrees and minutes -/
def add_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.minutes + b.minutes
  let carry_degrees := total_minutes / 60
  { degrees := a.degrees + b.degrees + carry_degrees,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- Subtraction of angles in degrees and minutes -/
def sub_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.degrees * 60 + a.minutes - (b.degrees * 60 + b.minutes)
  { degrees := total_minutes / 60,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- The main theorem to prove -/
theorem angle_subtraction_theorem :
  sub_angles { degrees := 72, minutes := 24, valid := by sorry }
              { degrees := 28, minutes := 36, valid := by sorry } =
  { degrees := 43, minutes := 48, valid := by sorry } := by sorry

end angle_subtraction_theorem_l249_24992


namespace eggs_left_for_breakfast_l249_24950

def total_eggs : ℕ := 36

def eggs_for_crepes : ℕ := (2 * total_eggs) / 5

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (3 * eggs_after_crepes) / 7

def eggs_after_cupcakes : ℕ := eggs_after_crepes - eggs_for_cupcakes

def eggs_for_quiche : ℕ := eggs_after_cupcakes / 2

def eggs_left : ℕ := eggs_after_cupcakes - eggs_for_quiche

theorem eggs_left_for_breakfast : eggs_left = 7 := by
  sorry

end eggs_left_for_breakfast_l249_24950


namespace jolene_bicycle_fundraising_l249_24998

/-- Proves that Jolene raises enough money to buy the bicycle with some extra --/
theorem jolene_bicycle_fundraising (
  bicycle_cost : ℕ)
  (babysitting_families : ℕ)
  (babysitting_rate : ℕ)
  (car_washing_neighbors : ℕ)
  (car_washing_rate : ℕ)
  (dog_walking_count : ℕ)
  (dog_walking_rate : ℕ)
  (cash_gift : ℕ)
  (h1 : bicycle_cost = 250)
  (h2 : babysitting_families = 4)
  (h3 : babysitting_rate = 30)
  (h4 : car_washing_neighbors = 5)
  (h5 : car_washing_rate = 12)
  (h6 : dog_walking_count = 3)
  (h7 : dog_walking_rate = 15)
  (h8 : cash_gift = 40) :
  let total_raised := babysitting_families * babysitting_rate +
                      car_washing_neighbors * car_washing_rate +
                      dog_walking_count * dog_walking_rate +
                      cash_gift
  ∃ (extra : ℕ), total_raised = 265 ∧ total_raised > bicycle_cost ∧ extra = total_raised - bicycle_cost ∧ extra = 15 :=
by
  sorry


end jolene_bicycle_fundraising_l249_24998


namespace increasing_function_a_range_l249_24963

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a + 3)*x - 4*a + 3 else a^x

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end increasing_function_a_range_l249_24963


namespace janelle_marbles_l249_24966

/-- Calculates the total number of marbles Janelle has after a series of transactions. -/
def total_marbles (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (gifted_red : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (gift_red : ℕ) 
  (returned_blue : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let remaining_green := initial_green - gift_green
  let remaining_blue := total_blue - gift_blue + returned_blue
  let remaining_red := gifted_red - gift_red
  remaining_green + remaining_blue + remaining_red

/-- Proves that Janelle ends up with 197 marbles given the initial conditions and transactions. -/
theorem janelle_marbles : 
  total_marbles 26 12 15 7 9 12 3 8 = 197 := by
  sorry

end janelle_marbles_l249_24966


namespace a2016_equals_2025_l249_24970

/-- An arithmetic sequence with common difference 2 and a2007 = 2007 -/
def arithmetic_seq (n : ℕ) : ℕ :=
  2007 + 2 * (n - 2007)

/-- Theorem stating that the 2016th term of the sequence is 2025 -/
theorem a2016_equals_2025 : arithmetic_seq 2016 = 2025 := by
  sorry

end a2016_equals_2025_l249_24970


namespace tonya_needs_22_hamburgers_l249_24941

/-- The weight of each hamburger in ounces -/
def hamburger_weight : ℕ := 4

/-- The total weight in ounces eaten by last year's winner -/
def last_year_winner_weight : ℕ := 84

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record : ℕ := 
  (last_year_winner_weight / hamburger_weight) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's record -/
theorem tonya_needs_22_hamburgers : 
  hamburgers_to_beat_record = 22 := by sorry

end tonya_needs_22_hamburgers_l249_24941


namespace kristin_bell_peppers_count_l249_24951

/-- The number of bell peppers Kristin has -/
def kristin_bell_peppers : ℕ := 2

/-- The number of carrots Jaylen has -/
def jaylen_carrots : ℕ := 5

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers : ℕ := 2

/-- The number of green beans Kristin has -/
def kristin_green_beans : ℕ := 20

/-- The total number of vegetables Jaylen has -/
def jaylen_total_vegetables : ℕ := 18

theorem kristin_bell_peppers_count :
  (jaylen_carrots + jaylen_cucumbers + 
   (kristin_green_beans / 2 - 3) + 
   (2 * kristin_bell_peppers) = jaylen_total_vegetables) →
  kristin_bell_peppers = 2 := by
  sorry

end kristin_bell_peppers_count_l249_24951


namespace ball_size_ratio_l249_24917

/-- Given three balls A, B, and C with different sizes, where A is three times bigger than B,
    and B is half the size of C, prove that A is 1.5 times the size of C. -/
theorem ball_size_ratio :
  ∀ (size_A size_B size_C : ℝ),
  size_A > 0 → size_B > 0 → size_C > 0 →
  size_A = 3 * size_B →
  size_B = (1 / 2) * size_C →
  size_A = (3 / 2) * size_C :=
by
  sorry

end ball_size_ratio_l249_24917


namespace toy_purchase_cost_l249_24969

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) : 
  num_toys = 5 → 
  cost_per_toy = 3 → 
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 := by
  sorry

end toy_purchase_cost_l249_24969


namespace algal_bloom_characteristic_l249_24947

/-- Represents the characteristics of algal population growth --/
inductive AlgalGrowthCharacteristic
  | IrregularFluctuations
  | UnevenDistribution
  | RapidGrowthShortPeriod
  | SeasonalGrowthDecline

/-- Represents the nutrient level in a water body --/
inductive NutrientLevel
  | Oligotrophic
  | Mesotrophic
  | Eutrophic

/-- Represents an algal bloom event --/
structure AlgalBloom where
  nutrientLevel : NutrientLevel
  growthCharacteristic : AlgalGrowthCharacteristic

/-- Theorem stating that algal blooms in eutrophic water bodies are characterized by rapid growth in a short period --/
theorem algal_bloom_characteristic (bloom : AlgalBloom) 
  (h : bloom.nutrientLevel = NutrientLevel.Eutrophic) : 
  bloom.growthCharacteristic = AlgalGrowthCharacteristic.RapidGrowthShortPeriod := by
  sorry

end algal_bloom_characteristic_l249_24947
