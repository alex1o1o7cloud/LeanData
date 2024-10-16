import Mathlib

namespace NUMINAMATH_CALUDE_first_player_wins_l553_55378

/-- Represents the state of the game with two piles of tokens -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Defines a valid move in the game -/
inductive ValidMove : GameState → GameState → Prop
  | single_pile (s t : GameState) (i : Fin 2) (n : Nat) :
      n > 0 →
      (i = 0 → t.pile1 = s.pile1 - n ∧ t.pile2 = s.pile2) →
      (i = 1 → t.pile1 = s.pile1 ∧ t.pile2 = s.pile2 - n) →
      ValidMove s t
  | both_piles (s t : GameState) (x y : Nat) :
      x > 0 →
      y > 0 →
      (x + y) % 2015 = 0 →
      t.pile1 = s.pile1 - x →
      t.pile2 = s.pile2 - y →
      ValidMove s t

/-- Defines the winning condition -/
def IsWinningState (s : GameState) : Prop :=
  ∀ t : GameState, ¬ValidMove s t

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → GameState),
    let initial_state := GameState.mk 10000 20000
    ∀ (opponent_move : GameState → GameState),
      ValidMove initial_state (strategy initial_state) →
      (∀ s, ValidMove s (opponent_move s) → ValidMove (opponent_move s) (strategy (opponent_move s))) →
      ∃ n : Nat, IsWinningState (Nat.iterate (λ s => strategy (opponent_move s)) n (strategy initial_state)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l553_55378


namespace NUMINAMATH_CALUDE_thomas_escalator_problem_l553_55345

/-- Thomas's escalator problem -/
theorem thomas_escalator_problem 
  (l : ℝ) -- length of the escalator
  (v : ℝ) -- speed of the escalator when working
  (r : ℝ) -- Thomas's running speed
  (w : ℝ) -- Thomas's walking speed
  (h1 : l / (v + r) = 15) -- Thomas runs down moving escalator in 15 seconds
  (h2 : l / (v + w) = 30) -- Thomas walks down moving escalator in 30 seconds
  (h3 : l / r = 20) -- Thomas runs down broken escalator in 20 seconds
  : l / w = 60 := by
  sorry

end NUMINAMATH_CALUDE_thomas_escalator_problem_l553_55345


namespace NUMINAMATH_CALUDE_tims_movie_marathon_l553_55300

/-- The duration of Tim's movie marathon --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_percentage : ℝ) (third_movie_difference : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_percentage)
  let first_two := first_movie + second_movie
  let third_movie := first_two - third_movie_difference
  first_movie + second_movie + third_movie

/-- Theorem stating the duration of Tim's movie marathon --/
theorem tims_movie_marathon :
  movie_marathon_duration 2 0.5 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tims_movie_marathon_l553_55300


namespace NUMINAMATH_CALUDE_domain_shift_l553_55362

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x-1)
def domain_f_minus_1 : Set ℝ := Set.Icc 1 2

-- Define the domain of f(x+2)
def domain_f_plus_2 : Set ℝ := Set.Icc (-2) (-1)

-- Theorem statement
theorem domain_shift (h : ∀ x ∈ domain_f_minus_1, f (x - 1) = f (x - 1)) :
  ∀ y ∈ domain_f_plus_2, f (y + 2) = f (y + 2) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l553_55362


namespace NUMINAMATH_CALUDE_monday_rainfall_duration_l553_55392

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_rate : ℝ
  monday_duration : ℝ
  tuesday_rate : ℝ
  tuesday_duration : ℝ
  wednesday_rate : ℝ
  wednesday_duration : ℝ
  total_rainfall : ℝ

/-- Theorem: The duration of rainfall on Monday is 7 hours -/
theorem monday_rainfall_duration (data : RainfallData) : data.monday_duration = 7 :=
  by
  have h1 : data.monday_rate = 1 := by sorry
  have h2 : data.tuesday_rate = 2 := by sorry
  have h3 : data.tuesday_duration = 4 := by sorry
  have h4 : data.wednesday_rate = 2 * data.tuesday_rate := by sorry
  have h5 : data.wednesday_duration = 2 := by sorry
  have h6 : data.total_rainfall = 23 := by sorry
  have h7 : data.total_rainfall = 
    data.monday_rate * data.monday_duration + 
    data.tuesday_rate * data.tuesday_duration + 
    data.wednesday_rate * data.wednesday_duration := by sorry
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_duration_l553_55392


namespace NUMINAMATH_CALUDE_no_integer_solutions_l553_55397

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 2*x*y + 9*z^2 = 150) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l553_55397


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l553_55375

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 6*x - 1 = 0 ↔ x = Real.sqrt 10 - 3 ∨ x = -Real.sqrt 10 - 3 :=
sorry

-- Problem 2
theorem fractional_equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ 1 →
  (x / (x + 2) = 2 / (x - 1) + 1 ↔ x = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l553_55375


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_23_l553_55399

theorem smallest_n_divisible_by_23 :
  ∃ (n : ℕ), (n^3 + 12*n^2 + 15*n + 180) % 23 = 0 ∧
  ∀ (m : ℕ), m < n → (m^3 + 12*m^2 + 15*m + 180) % 23 ≠ 0 :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_23_l553_55399


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l553_55373

theorem base_10_to_base_7 : 
  (1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0 : ℕ) = 2468 := by
  sorry

#eval 1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0

end NUMINAMATH_CALUDE_base_10_to_base_7_l553_55373


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l553_55396

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l553_55396


namespace NUMINAMATH_CALUDE_completing_square_result_l553_55328

theorem completing_square_result (x : ℝ) : x^2 + 4*x - 1 = 0 → (x + 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l553_55328


namespace NUMINAMATH_CALUDE_calculate_c_investment_c_investment_is_20000_l553_55346

/-- Calculates C's investment in a partnership given the investments of A and B,
    C's share of profit, and the total profit. -/
theorem calculate_c_investment (a_investment b_investment : ℕ)
                                (c_profit_share total_profit : ℕ) : ℕ :=
  let x := c_profit_share * (a_investment + b_investment + c_profit_share * total_profit / c_profit_share) / 
           (total_profit - c_profit_share)
  x

/-- Proves that C's investment is 20,000 given the specified conditions -/
theorem c_investment_is_20000 :
  calculate_c_investment 12000 16000 36000 86400 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_c_investment_c_investment_is_20000_l553_55346


namespace NUMINAMATH_CALUDE_total_cost_l553_55316

/-- The cost of a bottle of soda -/
def soda_cost : ℚ := sorry

/-- The cost of a bottle of mineral water -/
def mineral_cost : ℚ := sorry

/-- First condition: 2 bottles of soda and 1 bottle of mineral water cost 7 yuan -/
axiom condition1 : 2 * soda_cost + mineral_cost = 7

/-- Second condition: 4 bottles of soda and 3 bottles of mineral water cost 16 yuan -/
axiom condition2 : 4 * soda_cost + 3 * mineral_cost = 16

/-- Theorem: The cost of 10 bottles of soda and 10 bottles of mineral water is 45 yuan -/
theorem total_cost : 10 * soda_cost + 10 * mineral_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_l553_55316


namespace NUMINAMATH_CALUDE_natalie_shopping_result_l553_55364

/-- Calculates the amount of money Natalie has left after shopping -/
def money_left (initial_amount jumper_price tshirt_price heels_price jumper_discount_rate sales_tax_rate : ℚ) : ℚ :=
  let discounted_jumper_price := jumper_price * (1 - jumper_discount_rate)
  let total_before_tax := discounted_jumper_price + tshirt_price + heels_price
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_amount - total_after_tax

/-- Theorem stating that Natalie has $18.62 left after shopping -/
theorem natalie_shopping_result : 
  money_left 100 25 15 40 (10/100) (5/100) = 18.62 := by
  sorry

end NUMINAMATH_CALUDE_natalie_shopping_result_l553_55364


namespace NUMINAMATH_CALUDE_cook_remaining_potatoes_l553_55308

/-- Given a chef needs to cook potatoes with the following conditions:
  * The total number of potatoes to cook
  * The number of potatoes already cooked
  * The time it takes to cook each potato
  This function calculates the time required to cook the remaining potatoes. -/
def time_to_cook_remaining (total : ℕ) (cooked : ℕ) (time_per_potato : ℕ) : ℕ :=
  (total - cooked) * time_per_potato

/-- Theorem stating that it takes 36 minutes to cook the remaining potatoes. -/
theorem cook_remaining_potatoes :
  time_to_cook_remaining 12 6 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cook_remaining_potatoes_l553_55308


namespace NUMINAMATH_CALUDE_second_team_odd_second_team_odd_approx_l553_55388

/-- Calculates the odd for the second team in a four-team soccer bet -/
theorem second_team_odd (odd1 odd3 odd4 bet_amount expected_winnings : ℝ) : ℝ :=
  let total_odds := expected_winnings / bet_amount
  let second_team_odd := total_odds / (odd1 * odd3 * odd4)
  second_team_odd

/-- The calculated odd for the second team is approximately 5.23 -/
theorem second_team_odd_approx :
  let odd1 : ℝ := 1.28
  let odd3 : ℝ := 3.25
  let odd4 : ℝ := 2.05
  let bet_amount : ℝ := 5.00
  let expected_winnings : ℝ := 223.0072
  abs (second_team_odd odd1 odd3 odd4 bet_amount expected_winnings - 5.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_team_odd_second_team_odd_approx_l553_55388


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l553_55337

/-- Given that z = (a - i) / (2 - i) is a pure imaginary number, prove that a = -1/2 --/
theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (2 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l553_55337


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l553_55376

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l553_55376


namespace NUMINAMATH_CALUDE_function_simplification_l553_55352

theorem function_simplification (θ : Real) 
  (h1 : θ ∈ Set.Icc π (2 * π)) 
  (h2 : Real.tan θ = 2) : 
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ / 2) - Real.cos (θ / 2))) / 
   Real.sqrt (2 + 2 * Real.cos θ) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l553_55352


namespace NUMINAMATH_CALUDE_congruence_problem_l553_55347

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ -2023 [ZMOD 16] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l553_55347


namespace NUMINAMATH_CALUDE_square_of_2m2_plus_n2_l553_55312

theorem square_of_2m2_plus_n2 (m n : ℤ) :
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_2m2_plus_n2_l553_55312


namespace NUMINAMATH_CALUDE_series_sum_is_zero_l553_55311

/-- The sum of the series -1 + 0 + 1 - 2 + 0 + 2 - 3 + 0 + 3 - ... + (-4001) + 0 + 4001 -/
def seriesSum : ℤ := sorry

/-- The number of terms in the series -/
def numTerms : ℕ := 12003

/-- The series ends at 4001 -/
def lastTerm : ℕ := 4001

theorem series_sum_is_zero :
  seriesSum = 0 :=
by sorry

end NUMINAMATH_CALUDE_series_sum_is_zero_l553_55311


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l553_55354

/-- Proves that given a distance traveled at a certain speed, if the person were to travel an additional distance in the same time at a faster speed, we can calculate that faster speed. -/
theorem faster_speed_calculation (D : ℝ) (v_original : ℝ) (additional_distance : ℝ) (v_faster : ℝ) :
  D = 33.333333333333336 →
  v_original = 10 →
  additional_distance = 20 →
  D / v_original = (D + additional_distance) / v_faster →
  v_faster = 16 := by
  sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l553_55354


namespace NUMINAMATH_CALUDE_number_problem_l553_55304

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 50 + 30 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l553_55304


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l553_55303

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola lies on the line x + y - 1 = 0 -/
def focus_on_line (para : Parabola) (F : Point) : Prop :=
  F.x + F.y = 1

/-- The equation of the parabola is y² = 4x -/
def parabola_equation (para : Parabola) : Prop :=
  para.p = 2

/-- A line through the focus at 45° angle -/
def line_through_focus (F : Point) (A B : Point) : Prop :=
  (A.y - F.y) = (A.x - F.x) ∧ (B.y - F.y) = (B.x - F.x)

/-- A and B are on the parabola -/
def points_on_parabola (para : Parabola) (A B : Point) : Prop :=
  A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x

/-- The length of AB is 8 -/
def length_AB (A B : Point) : Prop :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8

theorem parabola_focus_theorem (para : Parabola) (F A B : Point) :
  focus_on_line para F →
  line_through_focus F A B →
  points_on_parabola para A B →
  parabola_equation para ∧ length_AB A B :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l553_55303


namespace NUMINAMATH_CALUDE_co2_formation_l553_55356

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ
  nahco3 : ℕ
  co2 : ℕ

-- Define the stoichiometric ratio
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.co2 = r.hcl

-- Define the theorem
theorem co2_formation (r : Reaction) (h1 : stoichiometric_ratio r) (h2 : r.hcl = 3) (h3 : r.nahco3 = 3) :
  r.co2 = min r.hcl r.nahco3 := by
  sorry

#check co2_formation

end NUMINAMATH_CALUDE_co2_formation_l553_55356


namespace NUMINAMATH_CALUDE_min_value_function_min_value_attained_l553_55391

theorem min_value_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 :=
sorry

theorem min_value_attained : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_min_value_attained_l553_55391


namespace NUMINAMATH_CALUDE_fraction_equality_l553_55334

theorem fraction_equality (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : (2/3) * x = k * (1/x)) : k = 2/27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l553_55334


namespace NUMINAMATH_CALUDE_max_ab_value_l553_55313

theorem max_ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1 / 2) * Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l553_55313


namespace NUMINAMATH_CALUDE_not_all_even_numbers_multiple_of_eight_l553_55384

theorem not_all_even_numbers_multiple_of_eight : ¬ (∀ n : ℕ, 2 ∣ n → 8 ∣ n) := by
  sorry

#check not_all_even_numbers_multiple_of_eight

end NUMINAMATH_CALUDE_not_all_even_numbers_multiple_of_eight_l553_55384


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l553_55393

-- Statement 1
theorem inequality_one (x : ℝ) (h : x ≥ 0) : x + 1 + 1 / (x + 1) ≥ 2 := by sorry

-- Statement 2
theorem inequality_two (x : ℝ) (h : x > 0) : (x + 1) / Real.sqrt x ≥ 2 := by sorry

-- Statement 3
theorem min_value_reciprocal : ∃ (m : ℝ), ∀ (x : ℝ), x + 1/x ≥ m ∧ ∃ (y : ℝ), y + 1/y = m := by sorry

-- Statement 4
theorem min_value_sqrt : ∃ (m : ℝ), ∀ (x : ℝ), Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ m ∧ 
  ∃ (y : ℝ), Real.sqrt (y^2 + 2) + 1 / Real.sqrt (y^2 + 2) = m := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l553_55393


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l553_55332

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l553_55332


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l553_55368

/-- The ratio of the volume of a sphere with radius p to the volume of a hemisphere with radius 3p is 1/13.5 -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 1 / 13.5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l553_55368


namespace NUMINAMATH_CALUDE_simplify_expression_l553_55343

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + (a + 2) / (1 - a)) = (2 + a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l553_55343


namespace NUMINAMATH_CALUDE_trapezoid_wings_area_l553_55357

/-- A trapezoid divided into four triangles -/
structure Trapezoid :=
  (A₁ : ℝ) -- Area of first triangle
  (A₂ : ℝ) -- Area of second triangle
  (A₃ : ℝ) -- Area of third triangle
  (A₄ : ℝ) -- Area of fourth triangle

/-- The theorem stating that if two triangles in the trapezoid have areas 4 and 9,
    then the sum of the areas of the other two triangles is 12 -/
theorem trapezoid_wings_area (T : Trapezoid) 
  (h₁ : T.A₁ = 4) 
  (h₂ : T.A₂ = 9) : 
  T.A₃ + T.A₄ = 12 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_wings_area_l553_55357


namespace NUMINAMATH_CALUDE_cos_zero_degrees_l553_55380

theorem cos_zero_degrees : Real.cos (0 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_cos_zero_degrees_l553_55380


namespace NUMINAMATH_CALUDE_triple_sum_arithmetic_sequence_l553_55350

def arithmetic_sequence (a₁ l n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * ((l - a₁) / (n - 1)))

def sum_arithmetic_sequence (a₁ l n : ℕ) : ℕ :=
  (n * (a₁ + l)) / 2

theorem triple_sum_arithmetic_sequence :
  let a₁ := 74
  let l := 107
  let n := 12
  3 * (sum_arithmetic_sequence a₁ l n) = 3258 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_arithmetic_sequence_l553_55350


namespace NUMINAMATH_CALUDE_cookies_in_class_l553_55324

/-- The number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies (mona jasmine rachel : ℕ) : ℕ := mona + jasmine + rachel

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class : ∃ (jasmine rachel : ℕ),
  jasmine = 20 - 5 ∧ 
  rachel = jasmine + 10 ∧
  total_cookies 20 jasmine rachel = 60 := by
sorry

end NUMINAMATH_CALUDE_cookies_in_class_l553_55324


namespace NUMINAMATH_CALUDE_v_2007_equals_1_l553_55371

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2007_equals_1 : v 2007 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2007_equals_1_l553_55371


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l553_55307

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {1,4}
def N : Finset Nat := {1,3,5}

theorem intersection_complement_equal : N ∩ (U \ M) = {3,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l553_55307


namespace NUMINAMATH_CALUDE_simple_interest_rate_l553_55310

/-- Simple interest calculation --/
theorem simple_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) : 
  P = 450 →
  t = 8 →
  I = P - 306 →
  I = P * (4 / 100) * t :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l553_55310


namespace NUMINAMATH_CALUDE_unhappy_no_skills_no_skills_purple_l553_55321

/-- Represents the properties of a snake --/
structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

/-- Tom's collection of snakes --/
def toms_snakes : Finset Snake := sorry

/-- The number of snakes in Tom's collection --/
axiom total_snakes : toms_snakes.card = 17

/-- The number of purple snakes --/
axiom purple_snakes : (toms_snakes.filter (fun s => s.purple)).card = 5

/-- All purple snakes are unhappy --/
axiom purple_unhappy : ∀ s ∈ toms_snakes, s.purple → ¬s.happy

/-- The number of happy snakes --/
axiom happy_snakes : (toms_snakes.filter (fun s => s.happy)).card = 7

/-- All happy snakes can add and subtract --/
axiom happy_skills : ∀ s ∈ toms_snakes, s.happy → s.can_add ∧ s.can_subtract

/-- No purple snakes can add or subtract --/
axiom purple_no_skills : ∀ s ∈ toms_snakes, s.purple → ¬s.can_add ∧ ¬s.can_subtract

theorem unhappy_no_skills :
  ∀ s ∈ toms_snakes, ¬s.happy → ¬s.can_add ∨ ¬s.can_subtract :=
sorry

theorem no_skills_purple :
  ∀ s ∈ toms_snakes, ¬s.can_add ∧ ¬s.can_subtract → s.purple :=
sorry

end NUMINAMATH_CALUDE_unhappy_no_skills_no_skills_purple_l553_55321


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l553_55383

/-- Given a rectangular prism with base edge length 3 cm and lateral face diagonal 3√5 cm,
    prove that its volume is 54 cm³ -/
theorem rectangular_prism_volume (base_edge : ℝ) (lateral_diagonal : ℝ) (volume : ℝ) :
  base_edge = 3 →
  lateral_diagonal = 3 * Real.sqrt 5 →
  volume = base_edge * base_edge * Real.sqrt (lateral_diagonal^2 - base_edge^2) →
  volume = 54 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l553_55383


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l553_55302

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 8 km/h. -/
theorem man_speed_in_still_water 
  (s : SwimmerSpeeds)
  (h1 : effectiveSpeed s true = 30 / 3)  -- Downstream condition
  (h2 : effectiveSpeed s false = 18 / 3) -- Upstream condition
  : s.manSpeed = 8 := by
  sorry

#check man_speed_in_still_water

end NUMINAMATH_CALUDE_man_speed_in_still_water_l553_55302


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l553_55360

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- Calculates the surface area of a rectangular prism --/
def surfaceArea (prism : RectangularPrism) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Represents the result of cutting a unit cube from a rectangular prism --/
structure CutPrism where
  original : RectangularPrism
  cut_from_corner : Bool

/-- Calculates the surface area of a prism after a unit cube is cut from it --/
def surfaceAreaAfterCut (cut : CutPrism) : ℝ :=
  surfaceArea cut.original

theorem surface_area_unchanged (cut : CutPrism) :
  surfaceArea cut.original = surfaceAreaAfterCut cut :=
sorry

#check surface_area_unchanged

end NUMINAMATH_CALUDE_surface_area_unchanged_l553_55360


namespace NUMINAMATH_CALUDE_complex_number_real_part_eq_imaginary_part_l553_55317

theorem complex_number_real_part_eq_imaginary_part (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - 2*i) * (a + i)
  (z.re + z.im = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_part_eq_imaginary_part_l553_55317


namespace NUMINAMATH_CALUDE_hiker_cyclist_catchup_time_l553_55395

/-- Proves that a hiker catches up to a cyclist in 10 minutes under specific conditions -/
theorem hiker_cyclist_catchup_time :
  let hiker_speed : ℝ := 4  -- km/h
  let cyclist_speed : ℝ := 12  -- km/h
  let stop_time : ℝ := 5 / 60  -- hours (5 minutes converted to hours)
  
  let distance_cyclist : ℝ := cyclist_speed * stop_time
  let distance_hiker : ℝ := hiker_speed * stop_time
  let distance_between : ℝ := distance_cyclist - distance_hiker
  
  let catchup_time : ℝ := distance_between / hiker_speed

  catchup_time * 60 = 10  -- Convert hours to minutes
  := by sorry

end NUMINAMATH_CALUDE_hiker_cyclist_catchup_time_l553_55395


namespace NUMINAMATH_CALUDE_A_power_difference_l553_55374

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]

theorem A_power_difference :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by sorry

end NUMINAMATH_CALUDE_A_power_difference_l553_55374


namespace NUMINAMATH_CALUDE_hypotenuse_length_l553_55370

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  c : ℝ  -- Length of the hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (sum_of_squares : t.a^2 + t.b^2 + t.c^2 = 1450) : 
  t.c = Real.sqrt 725 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l553_55370


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l553_55336

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -9) : 
  x^2 + y^2 = 22 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l553_55336


namespace NUMINAMATH_CALUDE_a_greater_than_b_l553_55301

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l553_55301


namespace NUMINAMATH_CALUDE_problem_statement_l553_55351

theorem problem_statement : (0.125 : ℝ)^2012 * (2^2012)^3 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l553_55351


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l553_55327

-- Define the equation
def equation (a x : ℝ) : Prop :=
  x + |x| = 2 * Real.sqrt (3 + 2*a*x - 4*a)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a | (0 < a ∧ a < 3/4) ∨ (a > 3)}

-- Theorem statement
theorem equation_two_distinct_roots (a : ℝ) :
  (∃ x y, x ≠ y ∧ equation a x ∧ equation a y) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l553_55327


namespace NUMINAMATH_CALUDE_triangle_properties_l553_55349

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : (Real.cos t.C) / (Real.sin t.C) = (Real.cos t.A + Real.cos t.B) / (Real.sin t.A + Real.sin t.B)) :
  t.C = π / 3 ∧ 
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l553_55349


namespace NUMINAMATH_CALUDE_double_counted_page_number_l553_55358

theorem double_counted_page_number :
  ∃! (n : ℕ) (x : ℕ), 
    1 ≤ x ∧ 
    x ≤ n ∧ 
    n * (n + 1) / 2 + x = 2550 ∧ 
    x = 65 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_page_number_l553_55358


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l553_55361

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧
  ∀ (x : ℝ), equation x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l553_55361


namespace NUMINAMATH_CALUDE_existence_condition_l553_55344

variable {M : Type u}
variable (A B C : Set M)

theorem existence_condition :
  (∃ X : Set M, (X ∪ A) \ (X ∩ B) = C) ↔ 
  ((A ∩ Bᶜ ∩ Cᶜ = ∅) ∧ (Aᶜ ∩ B ∩ C = ∅)) := by sorry

end NUMINAMATH_CALUDE_existence_condition_l553_55344


namespace NUMINAMATH_CALUDE_sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l553_55355

theorem sin_pi_plus_A_implies_cos_three_pi_half_minus_A 
  (A : ℝ) (h : Real.sin (π + A) = 1/2) : 
  Real.cos ((3/2) * π - A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l553_55355


namespace NUMINAMATH_CALUDE_john_lost_socks_l553_55326

/-- The number of individual socks lost given initial pairs and maximum remaining pairs -/
def socks_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

theorem john_lost_socks (initial_pairs : ℕ) (max_remaining_pairs : ℕ) 
  (h1 : initial_pairs = 10) (h2 : max_remaining_pairs = 7) : 
  socks_lost initial_pairs max_remaining_pairs = 6 := by
  sorry

#eval socks_lost 10 7

end NUMINAMATH_CALUDE_john_lost_socks_l553_55326


namespace NUMINAMATH_CALUDE_proportion_problem_l553_55325

-- Define the proportion relation
def in_proportion (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem proportion_problem :
  ∀ (a b c d : ℝ),
  in_proportion a b c d →
  a = 2 →
  b = 3 →
  c = 6 →
  d = 9 := by
sorry

end NUMINAMATH_CALUDE_proportion_problem_l553_55325


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l553_55329

theorem youngest_sibling_age (a b c d : ℕ) : 
  a + b + c + d = 180 →
  b = a + 2 →
  c = a + 4 →
  d = a + 6 →
  Even a →
  Even b →
  Even c →
  Even d →
  a = 42 := by
sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l553_55329


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l553_55319

/-- Represents a cube with painted diagonal stripes on each face -/
structure StripedCube where
  /-- The number of faces on the cube -/
  num_faces : Nat
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : Nat
  /-- The total number of possible stripe combinations -/
  total_combinations : Nat
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : Nat

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe (cube : StripedCube) : Rat :=
  cube.favorable_outcomes / cube.total_combinations

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  ∃ (cube : StripedCube),
    cube.num_faces = 6 ∧
    cube.orientations_per_face = 2 ∧
    cube.total_combinations = 2^6 ∧
    cube.favorable_outcomes = 6 ∧
    probability_continuous_stripe cube = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l553_55319


namespace NUMINAMATH_CALUDE_point_on_line_slope_is_one_inclination_angle_45_l553_55381

/-- A line passing through the point (-1, 2) with an inclination angle of 45° -/
def line_l (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- The point (-1, 2) lies on the line -/
theorem point_on_line : line_l (-1) 2 :=
  sorry

/-- The slope of the line is tan(45°) = 1 -/
theorem slope_is_one :
  ∀ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ → line_l x₁ y₁ → line_l x₂ y₂ →
  (y₂ - y₁) / (x₂ - x₁) = 1 :=
  sorry

/-- The inclination angle of the line is 45° -/
theorem inclination_angle_45 :
  ∀ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ → line_l x₁ y₁ → line_l x₂ y₂ →
  Real.arctan ((y₂ - y₁) / (x₂ - x₁)) = Real.pi / 4 :=
  sorry

end NUMINAMATH_CALUDE_point_on_line_slope_is_one_inclination_angle_45_l553_55381


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l553_55363

theorem largest_integer_with_remainder (n : ℕ) : 
  (∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 7 = 4 → 
  n = 95 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l553_55363


namespace NUMINAMATH_CALUDE_right_triangle_condition_l553_55322

theorem right_triangle_condition (a d : ℝ) (ha : a > 0) (hd : d > 1) :
  (a * d^2)^2 = a^2 + (a * d)^2 ↔ d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l553_55322


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l553_55318

/-- A triangle with consecutive even number side lengths. -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality holds for an EvenTriangle. -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a triangle with consecutive even number
    side lengths that satisfies the triangle inequality is 18. -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfiesTriangleInequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfiesTriangleInequality t' → perimeter t' ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l553_55318


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l553_55348

/-- The eccentricity of a hyperbola x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  x^2 - y^2 / 4 = 1 → e = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l553_55348


namespace NUMINAMATH_CALUDE_retailer_profit_is_ten_percent_l553_55340

/-- Calculates the profit percentage for a retailer selling pens --/
def profit_percentage (buy_quantity : ℕ) (buy_price : ℕ) (discount : ℚ) : ℚ :=
  let cost_price := buy_price
  let selling_price_per_pen := 1 - discount
  let total_selling_price := buy_quantity * selling_price_per_pen
  let profit := total_selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 10% for the given conditions --/
theorem retailer_profit_is_ten_percent :
  profit_percentage 40 36 (1/100) = 10 := by
  sorry

#eval profit_percentage 40 36 (1/100)

end NUMINAMATH_CALUDE_retailer_profit_is_ten_percent_l553_55340


namespace NUMINAMATH_CALUDE_odd_function_property_l553_55339

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_prop : ∀ x, f (1 + x) = f (-x))
  (h_value : f (-1/3) = 1/3) :
  f (5/3) = 1/3 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l553_55339


namespace NUMINAMATH_CALUDE_distance_walked_when_meeting_l553_55331

/-- 
Given two people walking towards each other from a distance of 50 miles,
each at a constant speed of 5 miles per hour, prove that one person
will have walked 25 miles when they meet.
-/
theorem distance_walked_when_meeting 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (h1 : initial_distance = 50)
  (h2 : speed = 5) : 
  (initial_distance / (2 * speed)) * speed = 25 :=
by sorry

end NUMINAMATH_CALUDE_distance_walked_when_meeting_l553_55331


namespace NUMINAMATH_CALUDE_max_diff_six_digit_even_numbers_l553_55379

/-- A function that checks if a natural number has only even digits -/
def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d % 2 ≠ 0

/-- The theorem stating the maximum difference between two 6-digit numbers with the given conditions -/
theorem max_diff_six_digit_even_numbers :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    has_only_even_digits a ∧
    has_only_even_digits b ∧
    (∀ n : ℕ, a < n ∧ n < b → has_odd_digit n) ∧
    b - a = 111112 ∧
    (∀ a' b' : ℕ,
      (100000 ≤ a' ∧ a' < 1000000) →
      (100000 ≤ b' ∧ b' < 1000000) →
      has_only_even_digits a' →
      has_only_even_digits b' →
      (∀ n : ℕ, a' < n ∧ n < b' → has_odd_digit n) →
      b' - a' ≤ 111112) :=
by sorry

end NUMINAMATH_CALUDE_max_diff_six_digit_even_numbers_l553_55379


namespace NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l553_55359

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def isAllNumeric (hex : List HexDigit) : Bool :=
  sorry

/-- Counts numbers up to n (exclusive) with only numeric hexadecimal digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

theorem hex_numeric_count_and_sum :
  countNumericHex 512 = 200 ∧ sumDigits 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l553_55359


namespace NUMINAMATH_CALUDE_min_positive_period_of_tan_2x_l553_55377

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem min_positive_period_of_tan_2x :
  ∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end NUMINAMATH_CALUDE_min_positive_period_of_tan_2x_l553_55377


namespace NUMINAMATH_CALUDE_x_y_inequalities_l553_55333

theorem x_y_inequalities (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) :
  x < -2 ∧ y < -1 := by sorry

end NUMINAMATH_CALUDE_x_y_inequalities_l553_55333


namespace NUMINAMATH_CALUDE_julie_newspaper_sheets_l553_55369

/-- The number of sheets used to print one newspaper -/
def sheets_per_newspaper (boxes : ℕ) (packages_per_box : ℕ) (sheets_per_package : ℕ) (total_newspapers : ℕ) : ℕ :=
  (boxes * packages_per_box * sheets_per_package) / total_newspapers

/-- Proof that Julie uses 25 sheets to print one newspaper -/
theorem julie_newspaper_sheets : 
  sheets_per_newspaper 2 5 250 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_julie_newspaper_sheets_l553_55369


namespace NUMINAMATH_CALUDE_lunes_area_equals_rectangle_area_l553_55389

/-- Given a rectangle with sides a and b, with half-circles drawn outward on each side
    and a circumscribing circle, the area of the lunes (crescent shapes) is equal to
    the area of the rectangle. -/
theorem lunes_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let semicircle_area := π * (a^2 + b^2) / 4
  let circumscribed_circle_area := π * (a^2 + b^2) / 4
  let rectangle_area := a * b
  let lunes_area := semicircle_area + rectangle_area - circumscribed_circle_area
  lunes_area = rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_lunes_area_equals_rectangle_area_l553_55389


namespace NUMINAMATH_CALUDE_polynomial_equality_l553_55338

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l553_55338


namespace NUMINAMATH_CALUDE_birds_in_tree_l553_55341

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l553_55341


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_to_all_lines_l553_55353

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane_perpendicular_to_all_lines
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_to_all_lines_l553_55353


namespace NUMINAMATH_CALUDE_unique_number_satisfying_equation_l553_55394

theorem unique_number_satisfying_equation : ∃! x : ℝ, ((x^3)^(1/3) * 4) / 2 + 5 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_equation_l553_55394


namespace NUMINAMATH_CALUDE_max_value_of_expression_l553_55398

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (2*x + y) + y / (2*y + z) + z / (2*z + x) ≤ 1 := by
  sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l553_55398


namespace NUMINAMATH_CALUDE_metro_earnings_l553_55309

/-- Calculates the earnings from ticket sales over a given period of time. -/
def calculate_earnings (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  ticket_cost * tickets_per_minute * minutes

/-- Proves that the earnings from ticket sales in 6 minutes is $90,
    given the ticket cost and average tickets sold per minute. -/
theorem metro_earnings :
  calculate_earnings 3 5 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_metro_earnings_l553_55309


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_B_l553_55385

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    with the slope of the line AB being 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_of_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_B_l553_55385


namespace NUMINAMATH_CALUDE_max_third_side_length_l553_55382

theorem max_third_side_length (a b c : ℕ) (ha : a = 7) (hb : b = 12) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l553_55382


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l553_55305

/-- An arithmetic sequence with first term 1 and sum of first n terms S_n -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithSeq) 
  (h : seq.S 19 / 19 - seq.S 17 / 17 = 6) : 
  seq.S 10 = 280 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l553_55305


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l553_55367

def fair_tickets : ℕ := 60
def fair_ticket_price : ℕ := 15
def baseball_ticket_price : ℕ := 10

theorem total_revenue_calculation :
  let baseball_tickets := fair_tickets / 3
  let fair_revenue := fair_tickets * fair_ticket_price
  let baseball_revenue := baseball_tickets * baseball_ticket_price
  fair_revenue + baseball_revenue = 1100 := by
sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l553_55367


namespace NUMINAMATH_CALUDE_exists_point_on_h_with_sum_40_l553_55372

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function h in terms of g
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := (g x - 2)^2

-- Theorem statement
theorem exists_point_on_h_with_sum_40 (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = (g x - 2)^2) (g_val : g 4 = 8) :
  ∃ x y, h x = y ∧ x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_exists_point_on_h_with_sum_40_l553_55372


namespace NUMINAMATH_CALUDE_monotonicity_condition_l553_55315

/-- The function f(x) = √(x² + 1) - ax is monotonic on [0,+∞) if and only if a ≥ 1, given that a > 0 -/
theorem monotonicity_condition (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → (Real.sqrt (x^2 + 1) - a * x < Real.sqrt (y^2 + 1) - a * y ∨
                               Real.sqrt (x^2 + 1) - a * x > Real.sqrt (y^2 + 1) - a * y)) ↔
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l553_55315


namespace NUMINAMATH_CALUDE_expression_bounds_l553_55314

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l553_55314


namespace NUMINAMATH_CALUDE_scientific_notation_of_2102000_l553_55330

theorem scientific_notation_of_2102000 :
  ∃ (a : ℝ) (n : ℤ), 2102000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.102 ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2102000_l553_55330


namespace NUMINAMATH_CALUDE_min_jumps_to_blue_l553_55335

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- A position on the 4x4 grid -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Checks if two positions are adjacent (share a side) -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col = p2.col + 1 ∨ p1.col + 1 = p2.col)) ∨
  (p1.col = p2.col ∧ (p1.row = p2.row + 1 ∨ p1.row + 1 = p2.row))

/-- The effect of jumping on a position, changing it and adjacent positions to blue -/
def jump (g : Grid) (p : Position) : Grid :=
  fun r c => if (r = p.row ∧ c = p.col) ∨ adjacent p ⟨r, c⟩ then true else g r c

/-- A sequence of jumps -/
def JumpSequence := List Position

/-- Apply a sequence of jumps to a grid -/
def applyJumps (g : Grid) : JumpSequence → Grid
  | [] => g
  | p::ps => applyJumps (jump g p) ps

/-- Check if all squares in the grid are blue -/
def allBlue (g : Grid) : Prop :=
  ∀ r c, g r c = true

/-- The initial all-red grid -/
def initialGrid : Grid :=
  fun _ _ => false

/-- Theorem: There exists a sequence of 4 jumps that turns the entire grid blue -/
theorem min_jumps_to_blue :
  ∃ (js : JumpSequence), js.length = 4 ∧ allBlue (applyJumps initialGrid js) :=
sorry


end NUMINAMATH_CALUDE_min_jumps_to_blue_l553_55335


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l553_55323

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 80)
  (h3 : S = a / (1 - r)) :
  a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l553_55323


namespace NUMINAMATH_CALUDE_tan_squared_sum_l553_55306

theorem tan_squared_sum (a b : ℝ) 
  (h1 : (Real.sin a)^2 / (Real.cos b)^2 + (Real.sin b)^2 / (Real.cos a)^2 = 2)
  (h2 : (Real.cos a)^3 / (Real.sin b)^3 + (Real.cos b)^3 / (Real.sin a)^3 = 4) :
  (Real.tan a)^2 / (Real.tan b)^2 + (Real.tan b)^2 / (Real.tan a)^2 = 30/13 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_l553_55306


namespace NUMINAMATH_CALUDE_complex_equation_solution_l553_55387

theorem complex_equation_solution (z : ℂ) : (2 * I) / z = 1 - I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l553_55387


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l553_55366

/-- Given that z = (2 + mi) / (1 + i) is a pure imaginary number, 
    prove that the imaginary part of z is -2. -/
theorem imaginary_part_of_pure_imaginary_z (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l553_55366


namespace NUMINAMATH_CALUDE_super_k_conference_l553_55390

theorem super_k_conference (n : ℕ) : 
  (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_super_k_conference_l553_55390


namespace NUMINAMATH_CALUDE_area_between_circles_first_quadrant_l553_55342

/-- The area of the region between two concentric circles with radii 15 and 9,
    extending only within the first quadrant, is equal to 36π. -/
theorem area_between_circles_first_quadrant :
  let r₁ : ℝ := 15
  let r₂ : ℝ := 9
  let full_area := π * (r₁^2 - r₂^2)
  let quadrant_area := full_area / 4
  quadrant_area = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_circles_first_quadrant_l553_55342


namespace NUMINAMATH_CALUDE_power_equality_no_quadratic_term_l553_55320

-- Define the variables
variable (x y a b : ℝ)

-- Theorem 1
theorem power_equality (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b := by sorry

-- Theorem 2
theorem no_quadratic_term (h : ∀ x, (x - 1) * (x^2 + a*x + 1) = x^3 + c*x + d) : a = 1 := by sorry

end NUMINAMATH_CALUDE_power_equality_no_quadratic_term_l553_55320


namespace NUMINAMATH_CALUDE_only_6_8_10_is_right_triangle_l553_55365

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that only (6, 8, 10) forms a right triangle among the given sets
theorem only_6_8_10_is_right_triangle :
  ¬(isRightTriangle 4 5 6) ∧
  ¬(isRightTriangle 5 7 9) ∧
  isRightTriangle 6 8 10 ∧
  ¬(isRightTriangle 7 8 9) :=
sorry

end NUMINAMATH_CALUDE_only_6_8_10_is_right_triangle_l553_55365


namespace NUMINAMATH_CALUDE_platform_length_l553_55386

/-- Calculates the length of a platform given the speed of a train, time to cross the platform, and length of the train. -/
theorem platform_length
  (train_speed_kmh : ℝ)
  (crossing_time_s : ℝ)
  (train_length_m : ℝ)
  (h1 : train_speed_kmh = 72)
  (h2 : crossing_time_s = 26)
  (h3 : train_length_m = 270) :
  let train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
  let total_distance : ℝ := train_speed_ms * crossing_time_s
  let platform_length : ℝ := total_distance - train_length_m
  platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l553_55386
