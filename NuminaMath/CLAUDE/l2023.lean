import Mathlib

namespace total_luggage_is_142_l2023_202399

/-- Calculates the total number of luggage pieces allowed on an international flight --/
def totalLuggageAllowed (economyPassengers businessPassengers firstClassPassengers : ℕ) : ℕ :=
  let economyAllowance := 5
  let businessAllowance := 8
  let firstClassAllowance := 12
  economyPassengers * economyAllowance + businessPassengers * businessAllowance + firstClassPassengers * firstClassAllowance

/-- Theorem stating that the total luggage allowed for the given passenger numbers is 142 --/
theorem total_luggage_is_142 :
  totalLuggageAllowed 10 7 3 = 142 := by
  sorry

#eval totalLuggageAllowed 10 7 3

end total_luggage_is_142_l2023_202399


namespace positive_A_value_l2023_202320

def hash (k : ℝ) (A B : ℝ) : ℝ := A^2 + k * B^2

theorem positive_A_value (k : ℝ) (A : ℝ) :
  k = 3 →
  hash k A 7 = 196 →
  A > 0 →
  A = 7 := by
sorry

end positive_A_value_l2023_202320


namespace fraction_subtraction_decreases_l2023_202373

theorem fraction_subtraction_decreases (a b n : ℕ) 
  (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a - n : ℚ) / (b - n) < (a : ℚ) / b :=
by sorry

end fraction_subtraction_decreases_l2023_202373


namespace books_on_shelf_l2023_202319

/-- The number of books on a shelf after adding more books is equal to the sum of the initial number of books and the number of books added. -/
theorem books_on_shelf (initial_books additional_books : ℕ) :
  initial_books + additional_books = initial_books + additional_books :=
by sorry

end books_on_shelf_l2023_202319


namespace min_value_of_a_l2023_202379

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) ↔ a ≥ 4 := by
  sorry

end min_value_of_a_l2023_202379


namespace circle_distance_bounds_specific_circle_distances_l2023_202363

/-- Given a circle with radius r and a point M at distance d from the center,
    returns a pair of the minimum and maximum distances from M to any point on the circle -/
def minMaxDistances (r d : ℝ) : ℝ × ℝ :=
  (r - d, r + d)

theorem circle_distance_bounds (r d : ℝ) (hr : r > 0) (hd : 0 ≤ d ∧ d < r) :
  let (min, max) := minMaxDistances r d
  ∀ p : ℝ × ℝ, (p.1 - r)^2 + p.2^2 = r^2 →
    min^2 ≤ (p.1 - d)^2 + p.2^2 ∧ (p.1 - d)^2 + p.2^2 ≤ max^2 :=
by sorry

theorem specific_circle_distances :
  minMaxDistances 10 3 = (7, 13) :=
by sorry

end circle_distance_bounds_specific_circle_distances_l2023_202363


namespace no_99_cents_combination_l2023_202352

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of five coins -/
def CoinCombination := Vector Coin 5

/-- Calculates the total value of a coin combination in cents -/
def totalValue (combo : CoinCombination) : Nat :=
  combo.toList.map coinValue |>.sum

/-- Theorem: It's impossible to make 99 cents with exactly five coins -/
theorem no_99_cents_combination :
  ¬∃ (combo : CoinCombination), totalValue combo = 99 := by
  sorry


end no_99_cents_combination_l2023_202352


namespace total_amount_proof_l2023_202327

def calculate_total_amount (plant_price tool_price soil_price : ℝ)
  (plant_discount tool_discount : ℝ) (tax_rate : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_plant := plant_price * (1 - plant_discount)
  let discounted_tool := tool_price * (1 - tool_discount)
  let subtotal := discounted_plant + discounted_tool + soil_price
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax + surcharge

theorem total_amount_proof :
  calculate_total_amount 467 85 38 0.15 0.10 0.08 12 = 564.37 := by
  sorry

end total_amount_proof_l2023_202327


namespace opposite_of_fraction_l2023_202359

theorem opposite_of_fraction : 
  -(11 : ℚ) / 2022 = -(11 / 2022) := by sorry

end opposite_of_fraction_l2023_202359


namespace laser_beam_distance_laser_beam_distance_is_ten_l2023_202354

/-- The total distance traveled by a laser beam with given conditions -/
theorem laser_beam_distance : ℝ :=
  let start : ℝ × ℝ := (2, 3)
  let end_point : ℝ × ℝ := (6, 3)
  let reflected_end : ℝ × ℝ := (-6, -3)
  Real.sqrt ((start.1 - reflected_end.1)^2 + (start.2 - reflected_end.2)^2)

/-- Proof that the laser beam distance is 10 -/
theorem laser_beam_distance_is_ten : laser_beam_distance = 10 := by
  sorry

end laser_beam_distance_laser_beam_distance_is_ten_l2023_202354


namespace sum_of_60_digits_eq_180_l2023_202312

/-- The sum of the first 60 digits after the decimal point in the decimal expansion of 1/1234 -/
def sum_of_60_digits : ℕ :=
  -- Define the sum here
  180

/-- Theorem stating that the sum of the first 60 digits after the decimal point
    in the decimal expansion of 1/1234 is equal to 180 -/
theorem sum_of_60_digits_eq_180 :
  sum_of_60_digits = 180 := by
  sorry

end sum_of_60_digits_eq_180_l2023_202312


namespace smallest_positive_integer_3001m_24567n_l2023_202348

theorem smallest_positive_integer_3001m_24567n : 
  ∃ (m n : ℤ), 3001 * m + 24567 * n = (Nat.gcd 3001 24567 : ℤ) ∧
  ∀ (k : ℤ), (∃ (a b : ℤ), k = 3001 * a + 24567 * b) → k = 0 ∨ abs k ≥ (Nat.gcd 3001 24567 : ℤ) :=
by sorry

end smallest_positive_integer_3001m_24567n_l2023_202348


namespace range_of_m_for_linear_system_l2023_202395

/-- Given a system of linear equations and an inequality condition, 
    prove that m must be less than 1. -/
theorem range_of_m_for_linear_system (x y m : ℝ) : 
  3 * x + y = 3 * m + 1 →
  x + 2 * y = 3 →
  2 * x - y < 1 →
  m < 1 := by
sorry

end range_of_m_for_linear_system_l2023_202395


namespace unique_x_value_l2023_202355

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a + c, b - d)

/-- The theorem stating the unique value of x -/
theorem unique_x_value : ∃! x : ℤ, ∃ y : ℤ, 
  star (x, y) (3, 3) = star (5, 4) (2, 2) := by
  sorry

end unique_x_value_l2023_202355


namespace moscow_olympiad_1975_l2023_202309

theorem moscow_olympiad_1975 (a b c p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  p = b^c + a →
  q = a^b + c →
  r = c^a + b →
  q = r := by
sorry

end moscow_olympiad_1975_l2023_202309


namespace fluffy_carrots_l2023_202331

def carrot_sequence (first_day : ℕ) : ℕ → ℕ
  | 0 => first_day
  | n + 1 => 2 * carrot_sequence first_day n

def total_carrots (first_day : ℕ) : ℕ :=
  (carrot_sequence first_day 0) + (carrot_sequence first_day 1) + (carrot_sequence first_day 2)

theorem fluffy_carrots (first_day : ℕ) :
  total_carrots first_day = 84 → carrot_sequence first_day 2 = 48 := by
  sorry

end fluffy_carrots_l2023_202331


namespace symmetric_points_sum_l2023_202370

/-- Given two points P and Q symmetric with respect to the origin, prove that a + b = -11 --/
theorem symmetric_points_sum (a b : ℝ) :
  let P : ℝ × ℝ := (a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a + 2*b)
  (P.1 = -Q.1 ∧ P.2 = -Q.2) →
  a + b = -11 := by
sorry


end symmetric_points_sum_l2023_202370


namespace arithmetic_expression_equals_two_l2023_202329

theorem arithmetic_expression_equals_two :
  10 - 9 + 8 * 7 / 2 - 6 * 5 + 4 - 3 + 2 / 1 = 2 := by
  sorry

end arithmetic_expression_equals_two_l2023_202329


namespace scientific_notation_260000_l2023_202301

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number 260000 in scientific notation -/
def n : ScientificNotation :=
  { coefficient := 2.6
    exponent := 5
    h1 := by sorry }

theorem scientific_notation_260000 :
  represents n 260000 := by sorry

end scientific_notation_260000_l2023_202301


namespace smallest_label_same_as_1993_solution_l2023_202392

/-- The number of points on the circle -/
def num_points : ℕ := 2000

/-- The highest label used in the problem -/
def max_label : ℕ := 1993

/-- Function to calculate the position of a label -/
def label_position (n : ℕ) : ℕ :=
  (n * (n + 1) / 2 - 1) % num_points

/-- Theorem stating that 118 is the smallest positive integer that labels the same point as 1993 -/
theorem smallest_label_same_as_1993 :
  ∀ k : ℕ, 0 < k → k < 118 → label_position k ≠ label_position max_label ∧
  label_position 118 = label_position max_label := by
  sorry

/-- Main theorem proving the solution -/
theorem solution : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 0 < k → k < n → label_position k ≠ label_position max_label) ∧
  label_position n = label_position max_label ∧ n = 118 := by
  sorry

end smallest_label_same_as_1993_solution_l2023_202392


namespace kayak_rental_cost_l2023_202314

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℝ := 18

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℝ := 15

/-- Represents the number of kayaks rented -/
def num_kayaks : ℕ := 10

/-- Represents the number of canoes rented -/
def num_canoes : ℕ := 15

/-- Represents the total revenue for one day -/
def total_revenue : ℝ := 405

theorem kayak_rental_cost :
  (kayak_cost * num_kayaks + canoe_cost * num_canoes = total_revenue) ∧
  (num_canoes = num_kayaks + 5) ∧
  (3 * num_kayaks = 2 * num_canoes) :=
by sorry

end kayak_rental_cost_l2023_202314


namespace complex_expression_calculation_l2023_202347

theorem complex_expression_calculation (a b : ℂ) :
  a = 3 + 2*I ∧ b = 1 - 3*I → 4*a + 5*b + a*b = 26 - 14*I :=
by sorry

end complex_expression_calculation_l2023_202347


namespace simplify_fraction_l2023_202341

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l2023_202341


namespace three_intersection_points_l2023_202311

-- Define the four lines
def line1 (x y : ℝ) : Prop := 3 * y - 2 * x = 1
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := 4 * x - 6 * y = 5
def line4 (x y : ℝ) : Prop := 2 * x - 3 * y = 4

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop := line x y

-- Define a function to check if a point is an intersection of at least two lines
def is_intersection (x y : ℝ) : Prop :=
  (point_on_line x y line1 ∧ point_on_line x y line2) ∨
  (point_on_line x y line1 ∧ point_on_line x y line3) ∨
  (point_on_line x y line1 ∧ point_on_line x y line4) ∨
  (point_on_line x y line2 ∧ point_on_line x y line3) ∨
  (point_on_line x y line2 ∧ point_on_line x y line4) ∨
  (point_on_line x y line3 ∧ point_on_line x y line4)

-- Theorem stating that there are exactly 3 distinct intersection points
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry


end three_intersection_points_l2023_202311


namespace triangle_properties_l2023_202307

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end triangle_properties_l2023_202307


namespace kathleen_remaining_money_l2023_202358

def kathleen_problem (june_savings july_savings august_savings : ℕ)
                     (school_supplies_cost clothes_cost : ℕ)
                     (aunt_bonus_threshold aunt_bonus : ℕ) : ℕ :=
  let total_savings := june_savings + july_savings + august_savings
  let total_expenses := school_supplies_cost + clothes_cost
  let bonus := if total_savings > aunt_bonus_threshold then aunt_bonus else 0
  total_savings + bonus - total_expenses

theorem kathleen_remaining_money :
  kathleen_problem 21 46 45 12 54 125 25 = 46 := by sorry

end kathleen_remaining_money_l2023_202358


namespace percentage_problem_l2023_202377

/-- Given a number N and a percentage P, this theorem proves that
    if P% of N is 24 less than 50% of N, and N = 160, then P = 35. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 160 → 
  (P / 100) * N = (50 / 100) * N - 24 → 
  P = 35 := by
sorry

end percentage_problem_l2023_202377


namespace f_of_five_equals_102_l2023_202369

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) : 
  f 5 = 102 := by
  sorry

end f_of_five_equals_102_l2023_202369


namespace ellipse_hyperbola_tangency_l2023_202302

-- Define the ellipse equation
def ellipse (x y n : ℝ) : Prop := x^2 + n*(y-1)^2 = n

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - 4*(y+3)^2 = 4

-- Define the tangency condition (discriminant = 0)
def tangent_condition (n : ℝ) : Prop := (24-2*n)^2 - 4*(4+n)*40 = 0

-- Theorem statement
theorem ellipse_hyperbola_tangency :
  ∃ n₁ n₂ : ℝ, 
    (abs (n₁ - 62.20625) < 0.00001) ∧ 
    (abs (n₂ - 1.66875) < 0.00001) ∧
    (∀ x y : ℝ, ellipse x y n₁ ∧ hyperbola x y → tangent_condition n₁) ∧
    (∀ x y : ℝ, ellipse x y n₂ ∧ hyperbola x y → tangent_condition n₂) :=
sorry

end ellipse_hyperbola_tangency_l2023_202302


namespace sum_of_four_numbers_l2023_202364

theorem sum_of_four_numbers : 5678 + 6785 + 7856 + 8567 = 28886 := by
  sorry

end sum_of_four_numbers_l2023_202364


namespace angle_properties_l2023_202386

theorem angle_properties (α : ℝ) (h : Real.tan α = -4/3) :
  (2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 2) ∧
  ((2 * Real.sin (π - α) + Real.sin (π/2 - α) + Real.sin (4*π)) / 
   (Real.cos (3*π/2 - α) + Real.cos (-α)) = -5/7) :=
by sorry

end angle_properties_l2023_202386


namespace f_difference_l2023_202380

/-- The function f(x) = 3x^3 + 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 4 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(9x^2 + 9xh + 3h^2 + 4x + 2h - 4) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (9 * x^2 + 9 * x * h + 3 * h^2 + 4 * x + 2 * h - 4) := by
  sorry

end f_difference_l2023_202380


namespace arithmetic_sequence_product_inequality_l2023_202333

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  common_diff : ℝ
  common_diff_neq_zero : common_diff ≠ 0
  is_arithmetic : ∀ i j, i < j → a j - a i = common_diff * (j - i)

/-- For an arithmetic sequence of 8 terms with positive values and non-zero common difference,
    the product of the first and last terms is less than the product of the fourth and fifth terms -/
theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end arithmetic_sequence_product_inequality_l2023_202333


namespace difference_of_squares_l2023_202340

theorem difference_of_squares (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end difference_of_squares_l2023_202340


namespace f_behavior_l2023_202398

def f (x : ℝ) := 2 * x^3 - 7

theorem f_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → f x < M) :=
sorry

end f_behavior_l2023_202398


namespace twentieth_15gonal_number_l2023_202384

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

/-- Theorem: The 20th 15-gonal number is 2490 -/
theorem twentieth_15gonal_number : N 20 15 = 2490 := by sorry

end twentieth_15gonal_number_l2023_202384


namespace base_8_to_10_98765_l2023_202324

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [9, 8, 7, 6, 5]

-- Define the function to convert a base-8 number to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

-- Theorem statement
theorem base_8_to_10_98765 :
  base_8_to_10 base_8_number = 41461 := by
  sorry

end base_8_to_10_98765_l2023_202324


namespace alcohol_water_ratio_l2023_202362

/-- Given a mixture where the volume fraction of alcohol is 2/7 and the volume fraction of water is 3/7,
    the ratio of the volume of alcohol to the volume of water is 2:3. -/
theorem alcohol_water_ratio (mixture : ℚ → ℚ) (h1 : mixture 1 = 2/7) (h2 : mixture 2 = 3/7) :
  (mixture 1) / (mixture 2) = 2/3 := by
  sorry

end alcohol_water_ratio_l2023_202362


namespace remaining_tickets_l2023_202338

/-- Represents the number of tickets Tom won and spent at the arcade -/
def arcade_tickets (x y : ℕ) : Prop :=
  let whack_a_mole := 32
  let skee_ball := 25
  let space_invaders := x
  let hat := 7
  let keychain := 10
  let small_toy := 15
  y = (whack_a_mole + skee_ball + space_invaders) - (hat + keychain + small_toy)

/-- Theorem stating that the number of tickets Tom has left is 25 plus the number of tickets he won from 'space invaders' -/
theorem remaining_tickets (x y : ℕ) :
  arcade_tickets x y → y = 25 + x := by
  sorry

end remaining_tickets_l2023_202338


namespace max_value_of_prime_sum_diff_l2023_202313

theorem max_value_of_prime_sum_diff (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧  -- a, b, c are prime
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧                    -- a, b, c are distinct
  a + b * c = 37 →                           -- given equation
  ∀ x y z : ℕ, 
    Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x + y * z = 37 →
    x + y - z ≤ a + b - c ∧                  -- a + b - c is maximum
    a + b - c = 32                           -- the maximum value is 32
  := by sorry

end max_value_of_prime_sum_diff_l2023_202313


namespace volunteer_assignment_count_l2023_202328

/-- The number of ways to assign volunteers to tasks -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_count :
  assignment_count 5 3 = 150 := by
  sorry

end volunteer_assignment_count_l2023_202328


namespace circular_field_diameter_specific_field_diameter_l2023_202387

/-- The diameter of a circular field, given the cost per meter of fencing and the total cost. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 16 meters. -/
theorem specific_field_diameter :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 3 150.79644737231007 - 16| < ε :=
sorry

end circular_field_diameter_specific_field_diameter_l2023_202387


namespace pencil_gain_percent_l2023_202371

/-- 
Proves that if the cost price of 12 pencils equals the selling price of 8 pencils, 
then the gain percent is 50%.
-/
theorem pencil_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 12 * cost_price = 8 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end pencil_gain_percent_l2023_202371


namespace monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l2023_202332

def polynomial (x : ℂ) : ℂ := x^2 + 6*x + 13

theorem monic_quadratic_with_complex_root_and_discriminant_multiple_of_four :
  (∀ x : ℂ, polynomial x = x^2 + 6*x + 13) ∧
  (polynomial (-3 + 2*I) = 0) ∧
  (∃ k : ℤ, 6^2 - 4*(1:ℝ)*13 = 4*k) :=
by sorry

end monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l2023_202332


namespace unique_solution_l2023_202342

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  y^2 = 4*x^3 + x - 4 ∧
  z^2 = 4*y^3 + y - 4 ∧
  x^2 = 4*z^3 + z - 4

/-- The theorem stating that (1, 1, 1) is the only solution to the system --/
theorem unique_solution :
  ∀ x y z : ℝ, system x y z → x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_l2023_202342


namespace john_thrice_tom_age_l2023_202337

/-- Proves that John was thrice as old as Tom 6 years ago, given the conditions -/
theorem john_thrice_tom_age (tom_current_age john_current_age x : ℕ) : 
  tom_current_age = 16 →
  john_current_age + 4 = 2 * (tom_current_age + 4) →
  john_current_age - x = 3 * (tom_current_age - x) →
  x = 6 := by
  sorry

end john_thrice_tom_age_l2023_202337


namespace two_pedestrians_problem_l2023_202390

/-- Two pedestrians problem -/
theorem two_pedestrians_problem (meet_time : ℝ) (time_difference : ℝ) :
  meet_time = 2 ∧ time_difference = 5/3 →
  ∃ (distance_AB : ℝ) (speed_A : ℝ) (speed_B : ℝ),
    distance_AB = 18 ∧
    speed_A = 5 ∧
    speed_B = 4 ∧
    distance_AB = speed_A * meet_time + speed_B * meet_time ∧
    distance_AB / speed_A = meet_time + time_difference ∧
    distance_AB / speed_B = meet_time + (meet_time + time_difference) :=
by sorry

end two_pedestrians_problem_l2023_202390


namespace equation_four_real_solutions_l2023_202321

theorem equation_four_real_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 3*x - 4)^2 = 9) ∧ s.card = 4 := by
sorry

end equation_four_real_solutions_l2023_202321


namespace equation_solution_l2023_202306

theorem equation_solution :
  ∀ x : ℝ, x^6 - 19*x^3 = 216 ↔ x = 3 ∨ x = -2 :=
by sorry

end equation_solution_l2023_202306


namespace total_students_calculation_l2023_202305

theorem total_students_calculation (short_ratio : Rat) (tall_count : Nat) (average_count : Nat) :
  short_ratio = 2/5 →
  tall_count = 90 →
  average_count = 150 →
  ∃ (total : Nat), total = (tall_count + average_count) / (1 - short_ratio) ∧ total = 400 :=
by sorry

end total_students_calculation_l2023_202305


namespace mass_of_man_is_60kg_l2023_202397

/-- The mass of a man causing a boat to sink in water -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man is 60 kg -/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end mass_of_man_is_60kg_l2023_202397


namespace find_B_value_l2023_202326

theorem find_B_value (A B : ℕ) : 
  (100 ≤ 6 * 100 + A * 10 + 5) ∧ (6 * 100 + A * 10 + 5 < 1000) ∧ 
  (100 ≤ 1 * 100 + 0 * 10 + B) ∧ (1 * 100 + 0 * 10 + B < 1000) ∧
  (6 * 100 + A * 10 + 5 + 1 * 100 + 0 * 10 + B = 748) →
  B = 3 := by sorry

end find_B_value_l2023_202326


namespace action_figures_added_l2023_202343

theorem action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : 
  initial = 3 → removed = 1 → final = 6 → final - (initial - removed) = 4 :=
by sorry

end action_figures_added_l2023_202343


namespace intersection_of_sets_l2023_202394

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x + 2 = 0}
  let B : Set ℝ := {x | x^2 - 4 = 0}
  A ∩ B = {-2} := by sorry

end intersection_of_sets_l2023_202394


namespace equilateral_triangle_cosine_l2023_202349

/-- An acute angle in degrees -/
def AcuteAngle (x : ℝ) : Prop := 0 < x ∧ x < 90

/-- Cosine function for angles in degrees -/
noncomputable def cosDeg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

/-- Theorem: The only acute angle x (in degrees) that satisfies the conditions for an equilateral triangle with sides cos x, cos x, and cos 5x is 60° -/
theorem equilateral_triangle_cosine (x : ℝ) :
  AcuteAngle x ∧ cosDeg x = cosDeg (5 * x) → x = 60 := by
  sorry

end equilateral_triangle_cosine_l2023_202349


namespace train_bus_cost_difference_proof_l2023_202325

def train_bus_cost_difference (train_cost bus_cost : ℝ) : Prop :=
  (train_cost > bus_cost) ∧
  (train_cost + bus_cost = 9.65) ∧
  (bus_cost = 1.40) ∧
  (train_cost - bus_cost = 6.85)

theorem train_bus_cost_difference_proof :
  ∃ (train_cost bus_cost : ℝ), train_bus_cost_difference train_cost bus_cost :=
by sorry

end train_bus_cost_difference_proof_l2023_202325


namespace rectangle_area_increase_l2023_202378

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  1.11 * L * (B * (1 + 22/100)) = 1.3542 * (L * B) := by sorry

end rectangle_area_increase_l2023_202378


namespace mean_of_points_l2023_202323

def points : List ℝ := [81, 73, 83, 86, 73]

theorem mean_of_points : (points.sum / points.length : ℝ) = 79.2 := by
  sorry

end mean_of_points_l2023_202323


namespace seventh_oblong_number_l2023_202368

/-- Defines an oblong number for a given positive integer n -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that the 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end seventh_oblong_number_l2023_202368


namespace degree_of_polynomial_l2023_202334

def p (x : ℝ) : ℝ := (2*x^5 - 3*x^3 + x^2 - 14) * (3*x^11 - 9*x^8 + 9*x^5 + 30) - (x^3 + 5)^7

theorem degree_of_polynomial : 
  ∃ (a : ℝ) (q : ℝ → ℝ), a ≠ 0 ∧ 
  (∀ (x : ℝ), p x = a * x^21 + q x) ∧ 
  (∃ (N : ℝ), ∀ (x : ℝ), |x| > N → |q x| < |a| * |x|^21) :=
sorry

end degree_of_polynomial_l2023_202334


namespace S_is_valid_set_l2023_202316

-- Define the set of non-negative integers not exceeding 10
def S : Set ℕ := {n : ℕ | n ≤ 10}

-- Theorem stating that S is a valid set
theorem S_is_valid_set :
  -- S has definite elements
  (∀ n : ℕ, n ∈ S ↔ n ≤ 10) ∧
  -- S has disordered elements (always true for sets)
  True ∧
  -- S has distinct elements (follows from the definition of ℕ)
  (∀ a b : ℕ, a ∈ S → b ∈ S → a = b → a = b) :=
sorry

end S_is_valid_set_l2023_202316


namespace min_value_x_plus_2y_l2023_202356

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - x * y = 0) :
  x + 2 * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y - x * y = 0 ∧ x + 2 * y = 9 :=
by sorry

end min_value_x_plus_2y_l2023_202356


namespace bookstore_discount_theorem_l2023_202361

variable (Book : Type)
variable (bookstore : Set Book)
variable (discounted_by_20_percent : Book → Prop)

theorem bookstore_discount_theorem 
  (h : ¬ ∀ b ∈ bookstore, discounted_by_20_percent b) : 
  (∃ b ∈ bookstore, ¬ discounted_by_20_percent b) ∧ 
  (¬ ∀ b ∈ bookstore, discounted_by_20_percent b) := by
  sorry

end bookstore_discount_theorem_l2023_202361


namespace count_numbers_l2023_202360

/-- The number of digits available for creating numbers -/
def num_digits : ℕ := 5

/-- The set of digits available for creating numbers -/
def digit_set : Finset ℕ := {0, 1, 2, 3, 4}

/-- The number of digits required for each number -/
def num_places : ℕ := 4

/-- Function to calculate the number of four-digit numbers -/
def four_digit_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers -/
def four_digit_even_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repeating digits -/
def four_digit_no_repeat : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers without repeating digits -/
def four_digit_even_no_repeat : ℕ := sorry

theorem count_numbers :
  four_digit_numbers = 500 ∧
  four_digit_even_numbers = 300 ∧
  four_digit_no_repeat = 96 ∧
  four_digit_even_no_repeat = 60 := by sorry

end count_numbers_l2023_202360


namespace coat_price_calculations_l2023_202385

def original_price : ℝ := 500
def initial_reduction : ℝ := 300
def discount1 : ℝ := 0.1
def discount2 : ℝ := 0.15

theorem coat_price_calculations :
  let percent_reduction := (initial_reduction / original_price) * 100
  let reduced_price := original_price - initial_reduction
  let percent_increase := ((original_price - reduced_price) / reduced_price) * 100
  let price_after_initial_reduction := reduced_price
  let price_after_discount1 := price_after_initial_reduction * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  (percent_reduction = 60 ∧
   percent_increase = 150 ∧
   final_price = 153) := by sorry

end coat_price_calculations_l2023_202385


namespace xyz_sum_product_range_l2023_202366

theorem xyz_sum_product_range :
  ∀ x y z : ℝ,
  0 < x ∧ x < 1 →
  0 < y ∧ y < 1 →
  0 < z ∧ z < 1 →
  x + y + z = 2 →
  ∃ S : ℝ, S = x*y + y*z + z*x ∧ 1 < S ∧ S ≤ 4/3 ∧
  ∀ T : ℝ, (∃ a b c : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
                        a + b + c = 2 ∧ T = a*b + b*c + c*a) →
            1 < T ∧ T ≤ 4/3 :=
by sorry

end xyz_sum_product_range_l2023_202366


namespace perpendicular_solution_parallel_solution_l2023_202382

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x^2 - 1, x + 1)

-- Define perpendicularity condition
def perpendicular (x : ℝ) : Prop := a.1 * (b x).1 + a.2 * (b x).2 = 0

-- Define parallelism condition
def parallel (x : ℝ) : Prop := a.1 * (b x).2 = a.2 * (b x).1

-- Theorem for perpendicular case
theorem perpendicular_solution :
  ∀ x : ℝ, perpendicular x → x = -1 ∨ x = -2 := by sorry

-- Theorem for parallel case
theorem parallel_solution :
  ∀ x : ℝ, parallel x → 
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = Real.sqrt 10 ∨
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = 2 * Real.sqrt 10 / 9 := by sorry

end perpendicular_solution_parallel_solution_l2023_202382


namespace cafeteria_duty_assignments_l2023_202345

def class_size : ℕ := 28
def duty_size : ℕ := 4

theorem cafeteria_duty_assignments :
  (Nat.choose class_size duty_size = 20475) ∧
  (Nat.choose (class_size - 1) (duty_size - 1) = 2925) := by
  sorry

#check cafeteria_duty_assignments

end cafeteria_duty_assignments_l2023_202345


namespace retailer_profit_percentage_l2023_202315

/-- Calculates the profit percentage for a retailer selling a machine --/
theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_rate = 0.1)
  : (((retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price) * 100 = 20) := by
  sorry

#check retailer_profit_percentage

end retailer_profit_percentage_l2023_202315


namespace unique_solution_quadratic_l2023_202336

/-- The quadratic equation qx^2 - 8x + 2 = 0 has only one solution when q = 8 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end unique_solution_quadratic_l2023_202336


namespace equation_solution_l2023_202330

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 8) / (x - 4) = (x - 3) / (x + 6) ↔ x = -12 / 7) := by
sorry

end equation_solution_l2023_202330


namespace max_x_minus_y_l2023_202300

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 → x' - y' ≤ max :=
by sorry

end max_x_minus_y_l2023_202300


namespace path_combinations_l2023_202304

theorem path_combinations (ways_AB ways_BC : ℕ) (h1 : ways_AB = 2) (h2 : ways_BC = 3) :
  ways_AB * ways_BC = 6 := by
sorry

end path_combinations_l2023_202304


namespace base_conversion_problem_l2023_202346

theorem base_conversion_problem (n d : ℕ) (hn : n > 0) (hd : d ≤ 9) :
  3 * n^2 + 2 * n + d = 263 ∧ 3 * n^2 + 2 * n + 4 = 253 + 6 * d → n + d = 11 := by
  sorry

end base_conversion_problem_l2023_202346


namespace light_travel_100_years_l2023_202339

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- Theorem stating the distance light travels in 100 years -/
theorem light_travel_100_years :
  100 * light_year_distance = 587 * (10 ^ 12 : ℝ) :=
sorry

end light_travel_100_years_l2023_202339


namespace cubic_function_property_l2023_202367

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem cubic_function_property (a b : ℝ) :
  f a b 4 = 0 → f a b (-4) = 2 := by
  sorry

end cubic_function_property_l2023_202367


namespace distance_to_y_axis_l2023_202372

/-- 
Given a point P with coordinates (x, -8) where the distance from the x-axis to P 
is half the distance from the y-axis to P, prove that P is 16 units from the y-axis.
-/
theorem distance_to_y_axis (x : ℝ) :
  let p : ℝ × ℝ := (x, -8)
  let dist_to_x_axis := |p.2|
  let dist_to_y_axis := |p.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis →
  dist_to_y_axis = 16 := by
  sorry

end distance_to_y_axis_l2023_202372


namespace triangle_side_altitude_inequality_l2023_202344

/-- Given a triangle with sides a, b, c where a > b > c and corresponding altitudes h_a, h_b, h_c,
    prove that a + h_a > b + h_b > c + h_c. -/
theorem triangle_side_altitude_inequality 
  (a b c h_a h_b h_c : ℝ) 
  (h_positive : 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : c < b ∧ b < a) 
  (h_triangle : h_a * a = h_b * b ∧ h_b * b = h_c * c) : 
  a + h_a > b + h_b ∧ b + h_b > c + h_c :=
by sorry

end triangle_side_altitude_inequality_l2023_202344


namespace solution_set_equivalence_l2023_202391

-- Define the function f implicitly
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) > 0
def solution_set_f (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x > -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x > 0 ↔ solution_set_f x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end solution_set_equivalence_l2023_202391


namespace boys_who_love_marbles_l2023_202310

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 20

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 10

/-- The number of boys who love to play marbles -/
def num_boys : ℕ := total_marbles / marbles_per_boy

theorem boys_who_love_marbles : num_boys = 2 := by
  sorry

end boys_who_love_marbles_l2023_202310


namespace translator_assignment_count_l2023_202396

def total_translators : ℕ := 9
def english_only_translators : ℕ := 6
def korean_only_translators : ℕ := 2
def bilingual_translators : ℕ := 1
def groups_needing_korean : ℕ := 2
def groups_needing_english : ℕ := 3

def assignment_ways : ℕ := sorry

theorem translator_assignment_count : 
  assignment_ways = 900 := by sorry

end translator_assignment_count_l2023_202396


namespace tan_beta_value_l2023_202335

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by sorry

end tan_beta_value_l2023_202335


namespace savings_increase_l2023_202388

theorem savings_increase (income expenditure savings new_income new_expenditure new_savings : ℝ)
  (h1 : expenditure = 0.75 * income)
  (h2 : savings = income - expenditure)
  (h3 : new_income = 1.2 * income)
  (h4 : new_expenditure = 1.1 * expenditure)
  (h5 : new_savings = new_income - new_expenditure) :
  (new_savings - savings) / savings * 100 = 50 := by
sorry

end savings_increase_l2023_202388


namespace nonzero_digits_count_l2023_202357

-- Define the fraction
def f : ℚ := 80 / (2^4 * 5^9)

-- Define a function to count non-zero digits after decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 3 := by sorry

end nonzero_digits_count_l2023_202357


namespace prob_second_odd_given_first_even_l2023_202389

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- A card is even if its number is even -/
def isEven (c : Card) : Prop := c.val % 2 = 0

/-- A card is odd if its number is odd -/
def isOdd (c : Card) : Prop := c.val % 2 = 1

/-- The set of even cards -/
def evenCards : Finset Card := sorry

/-- The set of odd cards -/
def oddCards : Finset Card := sorry

theorem prob_second_odd_given_first_even :
  (Finset.card oddCards : ℚ) / (Finset.card allCards - 1 : ℚ) = 3/4 := by sorry

end prob_second_odd_given_first_even_l2023_202389


namespace tom_completion_time_l2023_202322

/-- The time it takes Tom to complete a wall on his own after working with Avery for one hour -/
theorem tom_completion_time (avery_rate tom_rate : ℚ) : 
  avery_rate = 1/2 →  -- Avery's rate in walls per hour
  tom_rate = 1/4 →    -- Tom's rate in walls per hour
  (avery_rate + tom_rate) * 1 = 3/4 →  -- Combined work in first hour
  (1 - (avery_rate + tom_rate) * 1) / tom_rate = 1 := by
  sorry

end tom_completion_time_l2023_202322


namespace inequality_solution_l2023_202393

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 4 ∧ x < 3 → 1 < x ∧ x < 3 := by
  sorry

end inequality_solution_l2023_202393


namespace same_parity_min_max_l2023_202317

/-- A set with elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end same_parity_min_max_l2023_202317


namespace consecutive_five_digit_numbers_l2023_202376

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * c + c

def abbbb (a b : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * b + 10 * b + b

theorem consecutive_five_digit_numbers :
  ∀ a b c : ℕ,
    a < 10 → b < 10 → c < 10 →
    is_five_digit (abccc a b c) →
    is_five_digit (abbbb a b) →
    (abccc a b c).succ = abbbb a b ∨ (abbbb a b).succ = abccc a b c →
    ((a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0)) :=
by sorry

end consecutive_five_digit_numbers_l2023_202376


namespace talia_father_age_l2023_202375

/-- Represents the ages of Talia, her mother, and her father -/
structure FamilyAges where
  talia : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgeProblem (ages : FamilyAges) : Prop :=
  (ages.talia + 7 = 20) ∧
  (ages.mother = 3 * ages.talia) ∧
  (ages.father + 3 = ages.mother)

/-- Theorem stating that given the conditions, Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  FamilyAgeProblem ages → ages.father = 36 := by
  sorry

end talia_father_age_l2023_202375


namespace two_digit_product_1365_l2023_202318

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ ones ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number --/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem two_digit_product_1365 :
  ∀ (ab cd : TwoDigitNumber),
    ab.toNat * cd.toNat = 1365 →
    ab.tens ≠ ab.ones →
    cd.tens ≠ cd.ones →
    ab.tens ≠ cd.tens →
    ab.tens ≠ cd.ones →
    ab.ones ≠ cd.tens →
    ab.ones ≠ cd.ones →
    ((ab.tens = 2 ∧ ab.ones = 1) ∧ (cd.tens = 6 ∧ cd.ones = 5)) ∨
    ((ab.tens = 6 ∧ ab.ones = 5) ∧ (cd.tens = 2 ∧ cd.ones = 1)) :=
by sorry

end two_digit_product_1365_l2023_202318


namespace inequality_proof_l2023_202374

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ+, (a + b)^n.val - a^n.val - b^n.val ≥ 2^(2*n.val) - 2^(n.val+1) := by
  sorry

end inequality_proof_l2023_202374


namespace circles_intersect_at_right_angle_l2023_202381

/-- Two circles intersect at right angles if and only if the sum of the squares of their radii equals the square of the distance between their centers. -/
theorem circles_intersect_at_right_angle (a b c : ℝ) :
  ∃ (x y : ℝ), (x^2 + y^2 - 2*a*x + b^2 = 0 ∧ x^2 + y^2 - 2*c*y - b^2 = 0) →
  (a^2 - b^2) + (b^2 + c^2) = a^2 + c^2 :=
sorry

end circles_intersect_at_right_angle_l2023_202381


namespace expression_equals_negative_one_l2023_202365

theorem expression_equals_negative_one (b y : ℝ) (hb : b ≠ 0) (hy1 : y ≠ b) (hy2 : y ≠ -b) :
  (((b / (b + y)) + (y / (b - y))) / ((y / (b + y)) - (b / (b - y)))) = -1 := by
  sorry

end expression_equals_negative_one_l2023_202365


namespace star_emilio_sum_difference_l2023_202308

/-- The sum of numbers from 1 to 50 -/
def starSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- The sum of numbers from 1 to 50 with '3' replaced by '2' -/
def emilioSum : ℕ := (List.range 50).map (· + 1) |>.map (replaceThreeWithTwo) |>.sum
  where
    replaceThreeWithTwo (n : ℕ) : ℕ :=
      let tens := n / 10
      let ones := n % 10
      if tens = 3 then 20 + ones
      else if ones = 3 then 10 * tens + 2
      else n

/-- The difference between Star's sum and Emilio's sum is 105 -/
theorem star_emilio_sum_difference : starSum - emilioSum = 105 := by
  sorry

end star_emilio_sum_difference_l2023_202308


namespace concentric_circles_area_ratio_l2023_202303

theorem concentric_circles_area_ratio : 
  let small_diameter : ℝ := 2
  let large_diameter : ℝ := 4
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := large_diameter / 2
  let small_area : ℝ := π * small_radius^2
  let large_area : ℝ := π * large_radius^2
  let area_between : ℝ := large_area - small_area
  area_between / small_area = 3 := by sorry

end concentric_circles_area_ratio_l2023_202303


namespace least_multiple_25_over_500_l2023_202350

theorem least_multiple_25_over_500 : 
  ∀ n : ℕ, n > 0 → 25 * n > 500 → 525 ≤ 25 * n :=
sorry

end least_multiple_25_over_500_l2023_202350


namespace simplify_expression_l2023_202383

theorem simplify_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (16 * x^2 * y^3) / (8 * x * y^2) = 16 := by
  sorry

end simplify_expression_l2023_202383


namespace system_solution_ratio_l2023_202351

theorem system_solution_ratio (a b x y : ℝ) :
  8 * x - 6 * y = a →
  12 * y - 18 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = -4 / 9 := by sorry

end system_solution_ratio_l2023_202351


namespace equation_solution_l2023_202353

theorem equation_solution : ∃ (x : ℝ), x^2 - 2*x - 8 = -(x + 2)*(x - 6) ↔ x = 5 ∨ x = -2 := by
  sorry

end equation_solution_l2023_202353
