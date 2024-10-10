import Mathlib

namespace f_has_maximum_l3071_307158

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem f_has_maximum : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 28/3 := by
  sorry

end f_has_maximum_l3071_307158


namespace ab_value_l3071_307144

theorem ab_value (a b : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l3071_307144


namespace necessary_not_sufficient_l3071_307199

theorem necessary_not_sufficient (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, e^a + 2*a = e^b + 3*b → a > b) ∧
  (∃ a b, a > b ∧ e^a + 2*a ≠ e^b + 3*b) :=
by sorry

end necessary_not_sufficient_l3071_307199


namespace circumcircle_equation_l3071_307166

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (6, 2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y = 0

-- State the theorem
theorem circumcircle_equation :
  circle_equation O.1 O.2 ∧
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  (∀ (x y : ℝ), circle_equation x y → (x - 3)^2 + (y - 1)^2 = 10) :=
sorry

end circumcircle_equation_l3071_307166


namespace arithmetic_sequence_a15_l3071_307179

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a15 (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 7 = 8 → a 8 = 7 → a 15 = 0 := by
  sorry

end arithmetic_sequence_a15_l3071_307179


namespace divisibility_problem_l3071_307169

theorem divisibility_problem (n : ℤ) : 
  n > 101 →
  n % 101 = 0 →
  (∀ d : ℤ, d ∣ n → 1 < d → d < n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) →
  n % 100 = 0 := by
sorry

end divisibility_problem_l3071_307169


namespace arithmetic_sequence_first_term_l3071_307172

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of the first n terms of an arithmetic sequence. -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.firstTerm + (n - 1 : ℚ) * seq.commonDiff)

/-- Sum of terms from index m to n (inclusive) of an arithmetic sequence. -/
def sumTermsMtoN (seq : ArithmeticSequence) (m n : ℕ) : ℚ :=
  sumFirstNTerms seq n - sumFirstNTerms seq (m - 1)

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : sumFirstNTerms seq 30 = 450)
  (h2 : sumTermsMtoN seq 31 60 = 1650) : 
  seq.firstTerm = -13/3 := by
  sorry

end arithmetic_sequence_first_term_l3071_307172


namespace abs_sum_lt_sum_abs_when_product_negative_l3071_307116

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by
  sorry

end abs_sum_lt_sum_abs_when_product_negative_l3071_307116


namespace cakes_served_during_lunch_l3071_307146

theorem cakes_served_during_lunch :
  ∀ (lunch_cakes dinner_cakes : ℕ),
    dinner_cakes = 9 →
    dinner_cakes = lunch_cakes + 3 →
    lunch_cakes = 6 := by
  sorry

end cakes_served_during_lunch_l3071_307146


namespace custom_mul_solution_l3071_307193

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2*a - b^2

/-- Theorem stating that if a * 3 = 3 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∀ a : ℝ, custom_mul a 3 = 3 → a = 6 := by
  sorry

end custom_mul_solution_l3071_307193


namespace segment_length_l3071_307178

/-- Given a line segment AB with points P and Q, prove that AB has length 35 -/
theorem segment_length (A B P Q : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 1 / 5 →  -- P divides AB in ratio 1:4
  (Q.1 - A.1) / (B.1 - A.1) = 2 / 7 →  -- Q divides AB in ratio 2:5
  abs (Q.1 - P.1) = 3 →                -- Distance between P and Q is 3
  abs (B.1 - A.1) = 35 := by            -- Length of AB is 35
sorry

end segment_length_l3071_307178


namespace f_prime_zero_l3071_307148

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

theorem f_prime_zero : (deriv f) 0 = -2 := by sorry

end f_prime_zero_l3071_307148


namespace exact_arrival_speed_l3071_307187

theorem exact_arrival_speed (d : ℝ) (t : ℝ) (h1 : d = 30 * (t + 1/30)) (h2 : d = 50 * (t - 1/30)) :
  d / t = 37.5 := by sorry

end exact_arrival_speed_l3071_307187


namespace hyperbola_center_l3071_307101

/-- The center of the hyperbola given by the equation (4x+8)^2/16 - (5y-5)^2/25 = 1 is (-2, 1) -/
theorem hyperbola_center : ∃ (h k : ℝ), 
  (∀ x y : ℝ, (4*x + 8)^2 / 16 - (5*y - 5)^2 / 25 = 1 ↔ 
    (x - h)^2 - (y - k)^2 = 1) ∧ 
  h = -2 ∧ k = 1 := by
sorry

end hyperbola_center_l3071_307101


namespace cubic_system_solution_method_l3071_307142

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The statement of the theorem -/
theorem cubic_system_solution_method
  (a b c d : ℝ) (p : ℝ → ℝ) (hp : p = CubicPolynomial a b c d) :
  ∃ (cubic : ℝ → ℝ) (quadratic : ℝ → ℝ),
    (∀ x y : ℝ, x = p y ∧ y = p x ↔ 
      (cubic x = 0 ∧ quadratic y = 0) ∨ 
      (cubic y = 0 ∧ quadratic x = 0)) :=
by sorry

end cubic_system_solution_method_l3071_307142


namespace stationary_train_length_is_1296_l3071_307189

/-- The length of a stationary train given the time it takes for another train to pass it. -/
def stationary_train_length (time_to_pass_pole : ℝ) (time_to_cross_stationary : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * time_to_cross_stationary - train_speed * time_to_pass_pole

/-- Theorem stating that the length of the stationary train is 1296 meters under the given conditions. -/
theorem stationary_train_length_is_1296 :
  stationary_train_length 5 25 64.8 = 1296 := by
  sorry

#eval stationary_train_length 5 25 64.8

end stationary_train_length_is_1296_l3071_307189


namespace rosie_pies_calculation_l3071_307147

-- Define the function that calculates the number of pies
def pies_from_apples (apples_per_3_pies : ℕ) (available_apples : ℕ) : ℕ :=
  (available_apples * 3) / apples_per_3_pies

-- Theorem statement
theorem rosie_pies_calculation :
  pies_from_apples 12 36 = 9 := by
  sorry

end rosie_pies_calculation_l3071_307147


namespace problem_statement_l3071_307120

theorem problem_statement (A B C D : ℤ) 
  (h1 : A - B = 30) 
  (h2 : C + D = 20) : 
  (B + C) - (A - D) = -10 := by
sorry

end problem_statement_l3071_307120


namespace lights_after_2011_toggles_l3071_307184

/-- Represents the state of a light (on or off) -/
inductive LightState
| On : LightState
| Off : LightState

/-- Represents a row of 7 lights -/
def LightRow := Fin 7 → LightState

def initialState : LightRow := fun i =>
  if i = 0 ∨ i = 2 ∨ i = 4 ∨ i = 6 then LightState.On else LightState.Off

def toggleLights : LightRow → LightRow := sorry

theorem lights_after_2011_toggles (initialState : LightRow) 
  (h1 : ∀ state, (toggleLights^[14]) state = state)
  (h2 : (toggleLights^[9]) initialState 0 = LightState.On ∧ 
        (toggleLights^[9]) initialState 3 = LightState.On ∧ 
        (toggleLights^[9]) initialState 5 = LightState.On) :
  (toggleLights^[2011]) initialState 0 = LightState.On ∧
  (toggleLights^[2011]) initialState 3 = LightState.On ∧
  (toggleLights^[2011]) initialState 5 = LightState.On :=
sorry

end lights_after_2011_toggles_l3071_307184


namespace touchdown_points_l3071_307164

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end touchdown_points_l3071_307164


namespace vacation_cost_difference_l3071_307122

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) : 
  total_cost = 375 → initial_people = 3 → new_people = 5 → 
  (total_cost / initial_people) - (total_cost / new_people) = 50 := by
sorry

end vacation_cost_difference_l3071_307122


namespace simplify_expression_1_simplify_expression_2_l3071_307170

-- Part 1
theorem simplify_expression_1 (x : ℝ) (hx : x ≠ 0) :
  5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (2 * x) = 2 * y :=
sorry

end simplify_expression_1_simplify_expression_2_l3071_307170


namespace sugar_profit_theorem_l3071_307108

/-- Represents the profit calculation for a sugar trader --/
def sugar_profit (total_quantity : ℝ) (quantity_at_unknown_profit : ℝ) (known_profit : ℝ) (overall_profit : ℝ) : Prop :=
  let quantity_at_known_profit := total_quantity - quantity_at_unknown_profit
  let unknown_profit := (overall_profit * total_quantity - known_profit * quantity_at_known_profit) / quantity_at_unknown_profit
  unknown_profit = 12

/-- Theorem stating the profit percentage on the rest of the sugar --/
theorem sugar_profit_theorem :
  sugar_profit 1600 1200 8 11 := by
  sorry

end sugar_profit_theorem_l3071_307108


namespace bianca_cupcake_sale_l3071_307126

/-- Bianca's cupcake sale problem --/
theorem bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) :
  initial + made_later - left_at_end = (initial + made_later) - left_at_end :=
by
  sorry

/-- Solving Bianca's cupcake sale problem --/
def solve_bianca_cupcake_sale (initial : ℕ) (made_later : ℕ) (left_at_end : ℕ) : ℕ :=
  initial + made_later - left_at_end

#eval solve_bianca_cupcake_sale 14 17 25

end bianca_cupcake_sale_l3071_307126


namespace two_special_integers_under_million_l3071_307195

theorem two_special_integers_under_million : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 1000000 ∧ 
      ∃ a b : Nat, n = 2 * a^2 ∧ n = 3 * b^3) ∧ 
    S.card = 2 := by
  sorry

end two_special_integers_under_million_l3071_307195


namespace molecular_weight_CuCO3_l3071_307140

/-- The atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.55

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular weight of CuCO3 in g/mol -/
def CuCO3_weight : ℝ := Cu_weight + C_weight + 3 * O_weight

/-- The number of moles of CuCO3 -/
def moles : ℝ := 8

/-- Theorem: The molecular weight of 8 moles of CuCO3 is 988.48 grams -/
theorem molecular_weight_CuCO3 : moles * CuCO3_weight = 988.48 := by
  sorry

end molecular_weight_CuCO3_l3071_307140


namespace quadratic_equation_roots_specific_quadratic_equation_roots_l3071_307133

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  discriminant = 0 → ∃! x : ℝ, a*x^2 + b*x + c = 0 :=
by sorry

theorem specific_quadratic_equation_roots :
  ∃! x : ℝ, x^2 + 6*x + 9 = 0 :=
by sorry

end quadratic_equation_roots_specific_quadratic_equation_roots_l3071_307133


namespace business_profit_l3071_307127

/-- Represents a business with spending and income -/
structure Business where
  spending : ℕ
  income : ℕ

/-- Calculates the profit of a business -/
def profit (b : Business) : ℕ := b.income - b.spending

/-- Theorem stating the profit for a business with given conditions -/
theorem business_profit :
  ∀ (b : Business),
  (b.spending : ℚ) / b.income = 5 / 9 →
  b.income = 108000 →
  profit b = 48000 := by
  sorry

end business_profit_l3071_307127


namespace gcd_factorial_problem_l3071_307139

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end gcd_factorial_problem_l3071_307139


namespace line_equation_sum_l3071_307125

/-- Proves that for a line with slope 8 passing through the point (-2, 4),
    if its equation is of the form y = mx + b, then m + b = 28. -/
theorem line_equation_sum (m b : ℝ) : 
  m = 8 ∧ 4 = m * (-2) + b → m + b = 28 := by
  sorry

end line_equation_sum_l3071_307125


namespace z_max_min_difference_l3071_307109

theorem z_max_min_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 5)
  (sum_squares_eq : x^2 + y^2 + z^2 = 20)
  (xy_eq : x * y = 2) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end z_max_min_difference_l3071_307109


namespace equation_solution_l3071_307159

theorem equation_solution :
  ∀ x : ℚ, x ≠ 4 → ((7 * x + 2) / (x - 4) = -6 / (x - 4) ↔ x = -8 / 7) :=
by sorry

end equation_solution_l3071_307159


namespace lcm_hcf_problem_l3071_307121

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 385, the other is 180 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end lcm_hcf_problem_l3071_307121


namespace rental_crossover_point_l3071_307143

/-- Represents the rental rates for a car agency -/
structure AgencyRates where
  dailyRate : ℝ
  mileRate : ℝ

/-- Theorem stating the crossover point for car rental agencies -/
theorem rental_crossover_point (days : ℝ) (agency1 agency2 : AgencyRates) 
  (h1 : agency1.dailyRate = 20.25)
  (h2 : agency1.mileRate = 0.14)
  (h3 : agency2.dailyRate = 18.25)
  (h4 : agency2.mileRate = 0.22)
  : ∃ m : ℝ, m = 25 * days ∧ 
    agency1.dailyRate * days + agency1.mileRate * m = agency2.dailyRate * days + agency2.mileRate * m :=
by sorry

end rental_crossover_point_l3071_307143


namespace functional_equation_solution_l3071_307173

/-- A continuous function satisfying f(x) = a^x * f(x/2) for all x -/
def FunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Continuous f ∧ a > 0 ∧ ∀ x, f x = a^x * f (x/2)

theorem functional_equation_solution {f : ℝ → ℝ} {a : ℝ} 
  (h : FunctionalEquation f a) : 
  ∃ C : ℝ, ∀ x, f x = C * a^(2*x) := by
  sorry

end functional_equation_solution_l3071_307173


namespace problem_statement_l3071_307137

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end problem_statement_l3071_307137


namespace max_value_expression_l3071_307181

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end max_value_expression_l3071_307181


namespace largest_angle_in_special_right_triangle_l3071_307177

theorem largest_angle_in_special_right_triangle :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (x : ℝ), a = 3*x ∧ b = 2*x →
  max a (max b c) = 90 :=
by
  sorry

end largest_angle_in_special_right_triangle_l3071_307177


namespace train_speed_calculation_l3071_307118

theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 145 ∧ bridge_length = 230 ∧ crossing_time = 30 →
  ((train_length + bridge_length) / crossing_time) * 3.6 = 45 := by
sorry

end train_speed_calculation_l3071_307118


namespace min_a_for_subset_l3071_307196

theorem min_a_for_subset (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, x^2 - 6*x ≤ a + 2) ↔ a ≥ -7 :=
sorry

end min_a_for_subset_l3071_307196


namespace sam_total_wins_l3071_307190

theorem sam_total_wins (first_period : Nat) (second_period : Nat)
  (first_win_rate : Rat) (second_win_rate : Rat) :
  first_period = 100 →
  second_period = 100 →
  first_win_rate = 1/2 →
  second_win_rate = 3/5 →
  (first_period * first_win_rate + second_period * second_win_rate : Rat) = 110 := by
  sorry

end sam_total_wins_l3071_307190


namespace product_and_gcd_conditions_l3071_307160

theorem product_and_gcd_conditions (a b : ℕ+) : 
  a * b = 864 ∧ Nat.gcd a b = 6 ↔ (a = 6 ∧ b = 144) ∨ (a = 144 ∧ b = 6) ∨ (a = 18 ∧ b = 48) ∨ (a = 48 ∧ b = 18) := by
  sorry

end product_and_gcd_conditions_l3071_307160


namespace polar_to_rectangular_coordinates_l3071_307197

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end polar_to_rectangular_coordinates_l3071_307197


namespace piggy_bank_dime_difference_l3071_307113

theorem piggy_bank_dime_difference :
  ∀ (nickels dimes half_dollars : ℕ),
    nickels + dimes + half_dollars = 100 →
    5 * nickels + 10 * dimes + 50 * half_dollars = 1350 →
    ∃ (max_dimes min_dimes : ℕ),
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≤ max_dimes) ∧
      (∀ d : ℕ, (∃ n h : ℕ, n + d + h = 100 ∧ 5 * n + 10 * d + 50 * h = 1350) → d ≥ min_dimes) ∧
      max_dimes - min_dimes = 162 :=
by sorry

end piggy_bank_dime_difference_l3071_307113


namespace circle_ratio_l3071_307150

theorem circle_ratio (r R : ℝ) (hr : r > 0) (hR : R > 0) 
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 := by
  sorry

end circle_ratio_l3071_307150


namespace julia_tuesday_playmates_l3071_307156

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 6

/-- The difference in number of kids between Monday and Tuesday -/
def difference : ℕ := 1

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_playmates : tuesday_kids = 5 := by
  sorry

end julia_tuesday_playmates_l3071_307156


namespace women_in_luxury_class_l3071_307112

theorem women_in_luxury_class 
  (total_passengers : ℕ) 
  (women_percentage : ℚ) 
  (luxury_class_percentage : ℚ) 
  (h1 : total_passengers = 300) 
  (h2 : women_percentage = 80 / 100) 
  (h3 : luxury_class_percentage = 15 / 100) : 
  ℕ := by
  sorry

#check women_in_luxury_class

end women_in_luxury_class_l3071_307112


namespace total_carvings_eq_56_l3071_307171

/-- The number of wood carvings that can be contained in each shelf -/
def carvings_per_shelf : ℕ := 8

/-- The number of shelves filled with carvings -/
def filled_shelves : ℕ := 7

/-- The total number of wood carvings displayed -/
def total_carvings : ℕ := carvings_per_shelf * filled_shelves

theorem total_carvings_eq_56 : total_carvings = 56 := by
  sorry

end total_carvings_eq_56_l3071_307171


namespace negation_forall_geq_zero_equivalent_exists_lt_zero_l3071_307124

theorem negation_forall_geq_zero_equivalent_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end negation_forall_geq_zero_equivalent_exists_lt_zero_l3071_307124


namespace inverse_function_solution_l3071_307114

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that the solution to g^(-1)(x) = 2 is x = (1 - 2d) / (2c) -/
theorem inverse_function_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := λ x => 1 / (c * x + d)
  ∃! x, g x = 2⁻¹ ∧ x = (1 - 2 * d) / (2 * c) :=
by sorry

end inverse_function_solution_l3071_307114


namespace roots_sum_of_squares_l3071_307136

theorem roots_sum_of_squares (a b c : ℝ) (r s : ℝ) : 
  r^2 - (a+b)*r + ab + c = 0 → 
  s^2 - (a+b)*s + ab + c = 0 → 
  r^2 + s^2 = a^2 + b^2 - 2*c := by
  sorry

end roots_sum_of_squares_l3071_307136


namespace sequence_property_l3071_307130

/-- Given a sequence a and its partial sum S, prove that a_n = 2^n + n for all n ∈ ℕ⁺ -/
theorem sequence_property (a : ℕ+ → ℕ) (S : ℕ+ → ℕ) 
  (h : ∀ n : ℕ+, 2 * S n = 4 * a n + (n - 4) * (n + 1)) :
  ∀ n : ℕ+, a n = 2^(n : ℕ) + n := by
  sorry

end sequence_property_l3071_307130


namespace birthday_cake_candles_l3071_307117

theorem birthday_cake_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 79 →
  yellow = 27 →
  red = 14 →
  blue = total - yellow - red →
  blue = 38 := by
sorry

end birthday_cake_candles_l3071_307117


namespace diophantine_equation_solutions_l3071_307180

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) :=
by sorry

end diophantine_equation_solutions_l3071_307180


namespace S_value_l3071_307175

/-- The sum Sₙ for n points on a line and a point off the line -/
def S (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating the value of Sₙ based on n -/
theorem S_value (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∃ (p : Fin n → ℝ × ℝ), (∀ i, p i ∈ l) ∧ (∀ i j, i ≠ j → p i ≠ p j))
  (h3 : Q ∉ l) :
  S n l Q = if n = 3 then 1 else 0 :=
sorry

end S_value_l3071_307175


namespace number_problem_l3071_307131

theorem number_problem (x : ℝ) : 
  (1.5 * x) / 7 = 271.07142857142856 → x = 1265 := by
  sorry

end number_problem_l3071_307131


namespace no_real_solutions_for_ratio_equation_l3071_307145

theorem no_real_solutions_for_ratio_equation :
  ¬∃ (x : ℝ), (x + 3) / (2*x + 5) = (5*x + 4) / (8*x + 5) :=
by sorry

end no_real_solutions_for_ratio_equation_l3071_307145


namespace sum_of_roots_symmetric_function_l3071_307162

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly four distinct real roots,
    then the sum of these roots is 12 -/
theorem sum_of_roots_symmetric_function
  (g : ℝ → ℝ) 
  (h_sym : SymmetricAboutThree g)
  (h_roots : ∃! (s₁ s₂ s₃ s₄ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₂ ≠ s₃ ∧ s₂ ≠ s₄ ∧ s₃ ≠ s₄ ∧ 
              g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0) :
  ∃ (s₁ s₂ s₃ s₄ : ℝ), g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0 ∧ s₁ + s₂ + s₃ + s₄ = 12 :=
by sorry

end sum_of_roots_symmetric_function_l3071_307162


namespace perimeter_AEC_l3071_307198

/-- A square with side length 2 and vertices A, B, C, D (in order) is folded so that C meets AB at C'.
    AC' = 1/4, and BC intersects AD at E. -/
structure FoldedSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  C' : ℝ × ℝ
  E : ℝ × ℝ
  h_square : A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2)
  h_C'_on_AB : C'.1 = 1/4 ∧ C'.2 = 0
  h_E_on_AD : E = (0, 2)

/-- The perimeter of triangle AEC' in a folded square is (√65 + 1)/4 -/
theorem perimeter_AEC'_folded_square (fs : FoldedSquare) :
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d fs.A fs.E + d fs.E fs.C' + d fs.C' fs.A = (Real.sqrt 65 + 1) / 4 := by
  sorry


end perimeter_AEC_l3071_307198


namespace semicircle_perimeter_l3071_307151

theorem semicircle_perimeter (r : ℝ) (h : r = 2.1) : 
  let perimeter := π * r + 2 * r
  perimeter = π * 2.1 + 4.2 := by sorry

end semicircle_perimeter_l3071_307151


namespace ellipse_h_plus_k_l3071_307165

/-- An ellipse with foci at (1, 2) and (4, 2), passing through (-1, 5) -/
structure Ellipse where
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- A point on the ellipse -/
  point : ℝ × ℝ
  /-- The center of the ellipse -/
  center : ℝ × ℝ
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- Constraint: focus1 is at (1, 2) -/
  focus1_def : focus1 = (1, 2)
  /-- Constraint: focus2 is at (4, 2) -/
  focus2_def : focus2 = (4, 2)
  /-- Constraint: point is at (-1, 5) -/
  point_def : point = (-1, 5)
  /-- Constraint: center is the midpoint of foci -/
  center_def : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2)
  /-- Constraint: a is positive -/
  a_pos : a > 0
  /-- Constraint: b is positive -/
  b_pos : b > 0
  /-- Constraint: sum of distances from point to foci equals 2a -/
  sum_distances : Real.sqrt ((point.1 - focus1.1)^2 + (point.2 - focus1.2)^2) +
                  Real.sqrt ((point.1 - focus2.1)^2 + (point.2 - focus2.2)^2) = 2 * a

/-- Theorem: The sum of h and k in the standard form equation of the ellipse is 4.5 -/
theorem ellipse_h_plus_k (e : Ellipse) : e.center.1 + e.center.2 = 4.5 := by
  sorry

end ellipse_h_plus_k_l3071_307165


namespace triangle_properties_l3071_307129

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → π/2 < t.C) ∧
  (Real.sin t.A > Real.sin t.B → t.a > t.b) := by
  sorry

end triangle_properties_l3071_307129


namespace binomial_square_constant_l3071_307186

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9*x^2 + 27*x + a = (b*x + c)^2) → a = 81/4 := by
  sorry

end binomial_square_constant_l3071_307186


namespace nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l3071_307152

theorem nine_n_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℤ), 
    (p₁ ≠ 0 ∧ q₁ ≠ 0 ∧ r₁ ≠ 0 ∧ p₂ ≠ 0 ∧ q₂ ≠ 0 ∧ r₂ ≠ 0 ∧ p₃ ≠ 0 ∧ q₃ ≠ 0 ∧ r₃ ≠ 0) ∧
    (9 * n = (p₁ * a + q₁ * b + r₁ * c)^2 + (p₂ * a + q₂ * b + r₂ * c)^2 + (p₃ * a + q₃ * b + r₃ * c)^2) :=
sorry

theorem nine_n_sum_of_squares_not_div_by_three (n a b c : ℕ) (h₁ : n = a^2 + b^2 + c^2) 
  (h₂ : ¬(3 ∣ a) ∨ ¬(3 ∣ b) ∨ ¬(3 ∣ c)) :
  ∃ (x y z : ℕ), 
    (¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) ∧
    (9 * n = x^2 + y^2 + z^2) :=
sorry

end nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l3071_307152


namespace election_total_votes_l3071_307100

/-- Represents the number of votes for each candidate in the election. -/
structure ElectionResult where
  winner : ℕ
  opponent1 : ℕ
  opponent2 : ℕ
  opponent3 : ℕ
  fourth_place : ℕ

/-- Conditions of the election result. -/
def valid_election_result (e : ElectionResult) : Prop :=
  e.winner = e.opponent1 + 53 ∧
  e.winner = e.opponent2 + 79 ∧
  e.winner = e.opponent3 + 105 ∧
  e.fourth_place = 199

/-- Calculates the total votes in the election. -/
def total_votes (e : ElectionResult) : ℕ :=
  e.winner + e.opponent1 + e.opponent2 + e.opponent3 + e.fourth_place

/-- Theorem stating that the total votes in the election is 1598. -/
theorem election_total_votes :
  ∀ e : ElectionResult, valid_election_result e → total_votes e = 1598 :=
by sorry

end election_total_votes_l3071_307100


namespace min_snakes_owned_l3071_307167

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership where
  total : ℕ
  onlyDogs : ℕ
  onlyCats : ℕ
  catsAndDogs : ℕ
  allThree : ℕ

/-- The given pet ownership data -/
def givenData : PetOwnership :=
  { total := 59
  , onlyDogs := 15
  , onlyCats := 10
  , catsAndDogs := 5
  , allThree := 3 }

/-- The minimum number of snakes owned -/
def minSnakes : ℕ := givenData.allThree

theorem min_snakes_owned (data : PetOwnership) : 
  data.allThree ≤ minSnakes := by sorry

end min_snakes_owned_l3071_307167


namespace hexadecagon_diagonals_l3071_307154

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by
  sorry

end hexadecagon_diagonals_l3071_307154


namespace tangent_line_cubic_curve_l3071_307192

/-- Given a cubic function f(x) = x³ + ax + b and a line g(x) = kx + 1 tangent to f at x = 1,
    prove that 2a + b = 1 when f(1) = 3. -/
theorem tangent_line_cubic_curve (a b k : ℝ) : 
  (∀ x, (x^3 + a*x + b) = 3 * x^2 + a) →  -- Derivative condition
  (1^3 + a*1 + b = 3) →                   -- Point (1, 3) lies on the curve
  (k*1 + 1 = 3) →                         -- Point (1, 3) lies on the line
  (k = 3*1^2 + a) →                       -- Slope of tangent equals derivative at x = 1
  (2*a + b = 1) := by
sorry

end tangent_line_cubic_curve_l3071_307192


namespace garden_walkway_area_l3071_307191

/-- Calculates the total area of walkways in a garden with specified dimensions and layout. -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_width : ℕ) (bed_height : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_width + (columns + 1) * walkway_width
  let total_height := rows * bed_height + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_width * bed_height
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 := by
  sorry

end garden_walkway_area_l3071_307191


namespace perpendicular_lines_a_value_l3071_307149

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => (a + 2) * x - y + 1 = 0
  (∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → (x2 - x1) * (y2 - y1) = 0) →
  a = -1 := by
sorry

end perpendicular_lines_a_value_l3071_307149


namespace largest_angle_of_pentagon_l3071_307174

/-- Represents the measures of interior angles of a convex pentagon --/
structure PentagonAngles where
  x : ℝ
  angle1 : ℝ := x - 3
  angle2 : ℝ := x - 2
  angle3 : ℝ := x - 1
  angle4 : ℝ := x
  angle5 : ℝ := x + 1

/-- The sum of interior angles of a pentagon is 540° --/
def sumOfPentagonAngles : ℝ := 540

theorem largest_angle_of_pentagon (p : PentagonAngles) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = sumOfPentagonAngles →
  p.angle5 = 110 := by
  sorry

end largest_angle_of_pentagon_l3071_307174


namespace white_lights_replacement_l3071_307185

/-- The number of white lights Malcolm had initially --/
def total_white_lights : ℕ := by sorry

/-- The number of red lights initially purchased --/
def initial_red_lights : ℕ := 16

/-- The number of yellow lights purchased --/
def yellow_lights : ℕ := 4

/-- The number of blue lights initially purchased --/
def initial_blue_lights : ℕ := 2 * yellow_lights

/-- The number of green lights purchased --/
def green_lights : ℕ := 8

/-- The number of purple lights purchased --/
def purple_lights : ℕ := 3

/-- The additional number of red lights needed --/
def additional_red_lights : ℕ := 10

/-- The additional number of blue lights needed --/
def additional_blue_lights : ℕ := initial_blue_lights / 4

theorem white_lights_replacement :
  total_white_lights = 
    initial_red_lights + additional_red_lights +
    yellow_lights +
    initial_blue_lights + additional_blue_lights +
    green_lights +
    purple_lights := by sorry

end white_lights_replacement_l3071_307185


namespace ratio_of_55_to_11_l3071_307138

theorem ratio_of_55_to_11 : 
  let certain_number : ℚ := 55
  let ratio := certain_number / 11
  ratio = 5 / 1 := by sorry

end ratio_of_55_to_11_l3071_307138


namespace multiple_of_reciprocal_l3071_307141

theorem multiple_of_reciprocal (x : ℝ) (m : ℝ) (h1 : x > 0) (h2 : x + 17 = m * (1 / x)) (h3 : x = 3) : m = 60 := by
  sorry

end multiple_of_reciprocal_l3071_307141


namespace max_sum_of_other_roots_l3071_307111

/-- Given a polynomial x^3 - kx^2 + 20x - 15 with 3 roots, one of which is 3,
    the sum of the other two roots is at most 5. -/
theorem max_sum_of_other_roots (k : ℝ) :
  let p : ℝ → ℝ := λ x => x^3 - k*x^2 + 20*x - 15
  ∃ (r₁ r₂ : ℝ), (p 3 = 0 ∧ p r₁ = 0 ∧ p r₂ = 0) → r₁ + r₂ ≤ 5 := by
  sorry

end max_sum_of_other_roots_l3071_307111


namespace perpendicular_implies_parallel_l3071_307132

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the non-coincidence relation for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincidence relation for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_implies_parallel 
  (m n : Line) (α β : Plane)
  (h1 : non_coincident_lines m n)
  (h2 : non_coincident_planes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end perpendicular_implies_parallel_l3071_307132


namespace missing_exponent_proof_l3071_307107

theorem missing_exponent_proof :
  (9 ^ 5.6 * 9 ^ 10.3) / 9 ^ 2.56256 = 9 ^ 13.33744 := by
  sorry

end missing_exponent_proof_l3071_307107


namespace minimum_coins_for_purchase_l3071_307135

def quarter : ℕ := 25
def dime : ℕ := 10
def nickel : ℕ := 5

def candy_bar : ℕ := 45
def chewing_gum : ℕ := 35
def chocolate_bar : ℕ := 65
def juice_pack : ℕ := 70
def cookies : ℕ := 80

def total_cost : ℕ := 2 * candy_bar + 3 * chewing_gum + chocolate_bar + 2 * juice_pack + cookies

theorem minimum_coins_for_purchase :
  ∃ (q d n : ℕ), 
    q * quarter + d * dime + n * nickel = total_cost ∧ 
    q + d + n = 20 ∧ 
    q = 19 ∧ 
    d = 0 ∧ 
    n = 1 ∧
    ∀ (q' d' n' : ℕ), 
      q' * quarter + d' * dime + n' * nickel = total_cost → 
      q' + d' + n' ≥ 20 :=
by sorry

end minimum_coins_for_purchase_l3071_307135


namespace remainder_b39_mod_125_l3071_307163

def reverse_concatenate (n : ℕ) : ℕ :=
  -- Definition of b_n
  sorry

theorem remainder_b39_mod_125 : reverse_concatenate 39 % 125 = 21 := by
  sorry

end remainder_b39_mod_125_l3071_307163


namespace middle_number_problem_l3071_307194

theorem middle_number_problem (x y z : ℕ) : 
  x < y → y < z → x + y = 20 → x + z = 25 → y + z = 29 → y = 12 := by
  sorry

end middle_number_problem_l3071_307194


namespace expression_evaluation_l3071_307176

theorem expression_evaluation :
  ∃ m : ℤ, (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 1372 := by
  sorry

end expression_evaluation_l3071_307176


namespace no_such_function_l3071_307153

theorem no_such_function : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x * (1 + y * f x) := by
  sorry

end no_such_function_l3071_307153


namespace deluxe_premium_time_fraction_l3071_307115

/-- Represents the production details of stereos by Company S -/
structure StereoProduction where
  basicFraction : ℚ
  deluxeFraction : ℚ
  premiumFraction : ℚ
  deluxeTimeFactor : ℚ
  premiumTimeFactor : ℚ

/-- Calculates the fraction of total production time spent on deluxe and premium stereos -/
def deluxePremiumTimeFraction (prod : StereoProduction) : ℚ :=
  let totalTime := prod.basicFraction + prod.deluxeFraction * prod.deluxeTimeFactor + 
                   prod.premiumFraction * prod.premiumTimeFactor
  let deluxePremiumTime := prod.deluxeFraction * prod.deluxeTimeFactor + 
                           prod.premiumFraction * prod.premiumTimeFactor
  deluxePremiumTime / totalTime

/-- Theorem stating that the fraction of time spent on deluxe and premium stereos is 123/163 -/
theorem deluxe_premium_time_fraction :
  let prod : StereoProduction := {
    basicFraction := 2/5,
    deluxeFraction := 3/10,
    premiumFraction := 1 - 2/5 - 3/10,
    deluxeTimeFactor := 8/5,
    premiumTimeFactor := 5/2
  }
  deluxePremiumTimeFraction prod = 123/163 := by sorry

end deluxe_premium_time_fraction_l3071_307115


namespace passengers_in_buses_l3071_307106

/-- Given that 456 passengers fit into 12 buses, 
    prove that 266 passengers fit into 7 buses. -/
theorem passengers_in_buses 
  (total_passengers : ℕ) 
  (total_buses : ℕ) 
  (target_buses : ℕ) 
  (h1 : total_passengers = 456) 
  (h2 : total_buses = 12) 
  (h3 : target_buses = 7) :
  (total_passengers / total_buses) * target_buses = 266 := by
  sorry

end passengers_in_buses_l3071_307106


namespace park_trees_after_planting_l3071_307104

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of walnut trees after planting is 211 -/
theorem park_trees_after_planting :
  total_trees 107 104 = 211 := by
  sorry

end park_trees_after_planting_l3071_307104


namespace game_probability_l3071_307105

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℚ
  mel : ℚ
  chelsea : ℚ

/-- Calculates the probability of a specific outcome in the game -/
def outcome_probability (probs : PlayerProbabilities) (alex_wins mel_wins chelsea_wins : ℕ) : ℚ :=
  (probs.alex ^ alex_wins) * (probs.mel ^ mel_wins) * (probs.chelsea ^ chelsea_wins)

/-- Calculates the number of ways to arrange wins in a given number of rounds -/
def arrangements (total_rounds alex_wins mel_wins chelsea_wins : ℕ) : ℕ :=
  Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins

/-- The main theorem stating the probability of the specific outcome -/
theorem game_probability : ∃ (probs : PlayerProbabilities),
  probs.alex = 1/4 ∧
  probs.mel = 2 * probs.chelsea ∧
  probs.alex + probs.mel + probs.chelsea = 1 ∧
  (outcome_probability probs 2 3 3 * arrangements 8 2 3 3 : ℚ) = 35/512 := by
  sorry

end game_probability_l3071_307105


namespace measure_11_grams_l3071_307102

/-- Represents the number of ways to measure a weight using given weights -/
def measure_ways (one_gram : ℕ) (two_gram : ℕ) (four_gram : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 4 ways to measure 11 grams
    given three 1-gram weights, four 2-gram weights, and two 4-gram weights -/
theorem measure_11_grams :
  measure_ways 3 4 2 11 = 4 := by
  sorry

end measure_11_grams_l3071_307102


namespace grade_distribution_l3071_307103

theorem grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) 
  (h1 : prob_A = 0.6 * prob_B)
  (h2 : prob_C = 1.3 * prob_B)
  (h3 : prob_D = 0.8 * prob_B)
  (h4 : prob_A + prob_B + prob_C + prob_D = 1)
  (h5 : total_students = 50) :
  ∃ (num_B : ℕ), num_B = 14 ∧ 
    (↑num_B : ℝ) / total_students = prob_B := by
  sorry

end grade_distribution_l3071_307103


namespace parallelepiped_edge_length_l3071_307128

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ
  total_cubes : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem: The total edge length of the specific parallelepiped is 96 cm -/
theorem parallelepiped_edge_length :
  ∀ (p : Parallelepiped),
    p.volume = p.length * p.width * p.height →
    p.total_cubes = 440 →
    p.edge_min = 5 →
    p.length ≥ p.edge_min →
    p.width ≥ p.edge_min →
    p.height ≥ p.edge_min →
    total_edge_length p = 96 := by
  sorry

#check parallelepiped_edge_length

end parallelepiped_edge_length_l3071_307128


namespace part_one_part_two_l3071_307161

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part 1
theorem part_one : (Set.univ \ A 1) ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) : A a ⊆ B ↔ a < -4 ∨ (0 ≤ a ∧ a ≤ 1/2) := by sorry

end part_one_part_two_l3071_307161


namespace original_number_form_l3071_307188

theorem original_number_form (N : ℤ) : 
  (∃ m : ℤ, (N + 3) = 9 * m) → ∃ k : ℤ, N = 9 * k + 3 :=
by sorry

end original_number_form_l3071_307188


namespace negation_of_proposition_l3071_307123

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) := by
  sorry

end negation_of_proposition_l3071_307123


namespace longest_side_of_special_rectangle_l3071_307168

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 80 feet. -/
theorem longest_side_of_special_rectangle : 
  ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 80 := by
sorry

end longest_side_of_special_rectangle_l3071_307168


namespace expand_binomials_l3071_307182

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end expand_binomials_l3071_307182


namespace oscillating_cosine_shift_l3071_307119

theorem oscillating_cosine_shift (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
sorry

end oscillating_cosine_shift_l3071_307119


namespace edward_final_earnings_l3071_307183

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end edward_final_earnings_l3071_307183


namespace multiply_mixed_number_l3071_307110

theorem multiply_mixed_number : 9 * (7 + 2/5) = 66 + 3/5 := by
  sorry

end multiply_mixed_number_l3071_307110


namespace set_intersection_and_subset_l3071_307134

def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a) / (x - (a^2 + 1)) < 0}

theorem set_intersection_and_subset (a : ℝ) :
  (a = 2 → A a ∩ B a = {x | 2 < x ∧ x < 5}) ∧
  (B a ⊆ A a ↔ a ∈ Set.Icc (-1) (-1/2) ∪ Set.Icc 2 3) :=
sorry

end set_intersection_and_subset_l3071_307134


namespace arc_length_radius_l3071_307157

/-- Given an arc length of 2.5π cm and a central angle of 75°, the radius of the circle is 6 cm. -/
theorem arc_length_radius (L : ℝ) (θ : ℝ) (R : ℝ) : 
  L = 2.5 * π ∧ θ = 75 → R = 6 := by
  sorry

end arc_length_radius_l3071_307157


namespace square_sum_equals_b_times_ab_plus_two_l3071_307155

theorem square_sum_equals_b_times_ab_plus_two
  (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end square_sum_equals_b_times_ab_plus_two_l3071_307155
