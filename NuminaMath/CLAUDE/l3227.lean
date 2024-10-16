import Mathlib

namespace NUMINAMATH_CALUDE_park_area_is_20000_l3227_322714

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  cyclingSpeed : ℝ
  cyclingTime : ℝ

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Calculates the perimeter of a rectangular park -/
def parkPerimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Theorem: Given the conditions, the area of the park is 20,000 square meters -/
theorem park_area_is_20000 (park : RectangularPark) 
    (h1 : park.length = park.breadth / 2)
    (h2 : park.cyclingSpeed = 6)  -- in km/hr
    (h3 : park.cyclingTime = 1/10)  -- 6 minutes in hours
    (h4 : parkPerimeter park = park.cyclingSpeed * park.cyclingTime * 1000) : 
    parkArea park = 20000 := by
  sorry


end NUMINAMATH_CALUDE_park_area_is_20000_l3227_322714


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_in_range_l3227_322780

open Set

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}
def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_equality_implies_a_in_range (a : ℝ) :
  A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_in_range_l3227_322780


namespace NUMINAMATH_CALUDE_unique_solution_l3227_322745

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def xyz_to_decimal (x y z : ℕ) : ℚ := (100 * x + 10 * y + z : ℚ) / 1000

theorem unique_solution (x y z : ℕ) :
  is_digit x ∧ is_digit y ∧ is_digit z →
  (1 : ℚ) / (x + y + z : ℚ) = xyz_to_decimal x y z →
  x = 1 ∧ y = 2 ∧ z = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3227_322745


namespace NUMINAMATH_CALUDE_reciprocal_sum_geometric_progression_l3227_322782

theorem reciprocal_sum_geometric_progression 
  (q : ℝ) (n : ℕ) (S : ℝ) (h1 : q ≠ 1) :
  let a := 3
  let r := q^2
  let original_sum := a * (1 - r^(2*n)) / (1 - r)
  let reciprocal_sum := (1/a) * (1 - (1/r)^(2*n)) / (1 - 1/r)
  S = original_sum →
  reciprocal_sum = 1/S :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_geometric_progression_l3227_322782


namespace NUMINAMATH_CALUDE_shift_sine_graph_l3227_322718

/-- The problem statement as a theorem -/
theorem shift_sine_graph (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π) : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => 2 * Real.sin (2 * x - 2 * φ)
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂| = 4 ∧ 
    (∀ y₁ y₂ : ℝ, |f y₁ - g y₂| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
    |x₁ - x₂| = π / 6) →
  φ = π / 3 ∨ φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_shift_sine_graph_l3227_322718


namespace NUMINAMATH_CALUDE_employee_income_change_l3227_322702

theorem employee_income_change 
  (payment_increase : Real) 
  (time_decrease : Real) : 
  payment_increase = 0.3333 → 
  time_decrease = 0.3333 → 
  let new_payment := 1 + payment_increase
  let new_time := 1 - time_decrease
  let income_change := new_payment * new_time - 1
  income_change = -0.1111 := by sorry

end NUMINAMATH_CALUDE_employee_income_change_l3227_322702


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l3227_322708

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x ^ 7 = 13 * y ^ 11) :
  ∃ (a b c d : ℕ),
    x = a ^ c * b ^ d ∧
    a.Prime ∧ b.Prime ∧
    (∀ (a' b' c' d' : ℕ), x = a' ^ c' * b' ^ d' → a' ^ c' * b' ^ d' ≥ a ^ c * b ^ d) ∧
    a + b + c + d = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l3227_322708


namespace NUMINAMATH_CALUDE_actual_height_of_boy_l3227_322730

/-- Proves that the actual height of a boy in a class of 35 boys is 226 cm, given the conditions of the problem. -/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ)
  (h1 : n = 35)
  (h2 : initial_avg = 181)
  (h3 : wrong_height = 166)
  (h4 : actual_avg = 179) :
  ∃ (actual_height : ℝ), actual_height = 226 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end NUMINAMATH_CALUDE_actual_height_of_boy_l3227_322730


namespace NUMINAMATH_CALUDE_jones_trip_time_comparison_l3227_322725

theorem jones_trip_time_comparison 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : distance2 = 300) 
  (h3 : speed_multiplier = 3) :
  let time1 := distance1 / (distance1 / time1)
  let time2 := distance2 / (speed_multiplier * (distance1 / time1))
  time2 = 2 * time1 := by
sorry

end NUMINAMATH_CALUDE_jones_trip_time_comparison_l3227_322725


namespace NUMINAMATH_CALUDE_cone_height_l3227_322774

/-- Given a cone with slant height 2√2 cm and lateral surface area 4 cm², its height is 2 cm. -/
theorem cone_height (s : ℝ) (A : ℝ) (h : ℝ) :
  s = 2 * Real.sqrt 2 →
  A = 4 →
  A = π * s * (Real.sqrt (s^2 - h^2)) →
  h = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l3227_322774


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3227_322787

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3227_322787


namespace NUMINAMATH_CALUDE_final_state_theorem_l3227_322720

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The initial state of the box -/
def initialState : BoxState :=
  { white := 2015, black := 2015 }

/-- The final state of the box -/
def finalState : BoxState :=
  { white := 2, black := 1 }

/-- Represents one step of the ball selection process -/
def selectBalls (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process should stop -/
def stopCondition (state : BoxState) : Prop :=
  state.white + state.black = 3

/-- Theorem stating that the process will end with 2 white balls and 1 black ball -/
theorem final_state_theorem (state : BoxState) :
  state = initialState →
  (∃ n : Nat, (selectBalls^[n] state) = finalState ∧ stopCondition (selectBalls^[n] state)) :=
sorry

end NUMINAMATH_CALUDE_final_state_theorem_l3227_322720


namespace NUMINAMATH_CALUDE_units_digit_of_99_factorial_l3227_322700

theorem units_digit_of_99_factorial (n : ℕ) : n = 99 → n.factorial % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_99_factorial_l3227_322700


namespace NUMINAMATH_CALUDE_choose_formula_l3227_322741

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of ways to choose k items from n items is given by n! / (k!(n-k)!) -/
theorem choose_formula (n k : ℕ) (h : k ≤ n) :
  choose n k = Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_choose_formula_l3227_322741


namespace NUMINAMATH_CALUDE_n_minus_m_range_l3227_322738

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then Real.exp x - 1 else 3/2 * x + 1

theorem n_minus_m_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) : 
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 := by sorry

end NUMINAMATH_CALUDE_n_minus_m_range_l3227_322738


namespace NUMINAMATH_CALUDE_larger_number_proof_l3227_322771

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : 
  max x y = 16 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3227_322771


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3227_322790

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_union_theorem : 
  (U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3227_322790


namespace NUMINAMATH_CALUDE_combination_equality_l3227_322740

theorem combination_equality (n : ℕ) : 
  Nat.choose n 14 = Nat.choose n 4 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3227_322740


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3227_322785

-- Define the conditions
def p (x : ℝ) : Prop := Real.log (x - 3) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3227_322785


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3227_322729

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 → f a x ≥ 3) ∧ 
  (∃ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 ∧ f a x = 3) → 
  a = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3227_322729


namespace NUMINAMATH_CALUDE_business_profit_l3227_322717

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + (2 * (0.25 * (0.75 * total_profit))) = 50000 → 
  total_profit = 80000 := by
sorry

end NUMINAMATH_CALUDE_business_profit_l3227_322717


namespace NUMINAMATH_CALUDE_problem_solution_l3227_322762

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3227_322762


namespace NUMINAMATH_CALUDE_second_number_problem_l3227_322713

theorem second_number_problem (A B : ℝ) : 
  A = 580 → 0.20 * A = 0.30 * B + 80 → B = 120 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l3227_322713


namespace NUMINAMATH_CALUDE_luigi_pizza_count_l3227_322751

/-- The number of pizzas Luigi bought -/
def num_pizzas : ℕ := 4

/-- The total cost of pizzas in dollars -/
def total_cost : ℕ := 80

/-- The number of pieces each pizza is cut into -/
def pieces_per_pizza : ℕ := 5

/-- The cost of each piece of pizza in dollars -/
def cost_per_piece : ℕ := 4

/-- Theorem stating that the number of pizzas Luigi bought is 4 -/
theorem luigi_pizza_count :
  num_pizzas = 4 ∧
  total_cost = 80 ∧
  pieces_per_pizza = 5 ∧
  cost_per_piece = 4 ∧
  total_cost = num_pizzas * pieces_per_pizza * cost_per_piece :=
by sorry

end NUMINAMATH_CALUDE_luigi_pizza_count_l3227_322751


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_gas_mixture_l3227_322759

/-- Represents different types of statistical charts -/
inductive StatChart
  | PieChart
  | LineChart
  | BarChart
  deriving Repr

/-- Represents a mixture of gases -/
structure GasMixture where
  components : List String
  proportions : List Float
  sum_to_one : proportions.sum = 1

/-- Determines if a chart type is suitable for representing a gas mixture -/
def is_suitable_chart (chart : StatChart) (mixture : GasMixture) : Prop :=
  match chart with
  | StatChart.PieChart => 
      mixture.components.length > 1 ∧ 
      mixture.proportions.all (λ p => p ≥ 0 ∧ p ≤ 1)
  | _ => False

/-- Theorem stating that a pie chart is the most suitable for representing a gas mixture -/
theorem pie_chart_most_suitable_for_gas_mixture (mixture : GasMixture) :
  ∀ (chart : StatChart), is_suitable_chart chart mixture → chart = StatChart.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_gas_mixture_l3227_322759


namespace NUMINAMATH_CALUDE_rita_trust_fund_growth_l3227_322727

/-- Calculates the final amount in a trust fund after compound interest --/
def trust_fund_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_investment * (1 + interest_rate) ^ years

/-- Proves that Rita's trust fund will grow to approximately $10468.87 after 25 years --/
theorem rita_trust_fund_growth :
  let initial_investment : ℝ := 5000
  let interest_rate : ℝ := 0.03
  let years : ℕ := 25
  let final_amount := trust_fund_growth initial_investment interest_rate years
  ∃ ε > 0, |final_amount - 10468.87| < ε :=
by sorry

end NUMINAMATH_CALUDE_rita_trust_fund_growth_l3227_322727


namespace NUMINAMATH_CALUDE_unique_cube_difference_l3227_322786

theorem unique_cube_difference (n : ℕ+) : 
  (∃ x y : ℕ+, (837 + n : ℕ) = y^3 ∧ (837 - n : ℕ) = x^3) ↔ n = 494 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_difference_l3227_322786


namespace NUMINAMATH_CALUDE_pizza_dough_liquids_l3227_322775

/-- Pizza dough recipe calculation -/
theorem pizza_dough_liquids (milk_ratio : ℚ) (flour_ratio : ℚ) (flour_amount : ℚ) :
  milk_ratio = 75 →
  flour_ratio = 375 →
  flour_amount = 1125 →
  let portions := flour_amount / flour_ratio
  let milk_amount := portions * milk_ratio
  let water_amount := milk_amount / 2
  milk_amount + water_amount = 337.5 := by
  sorry

#check pizza_dough_liquids

end NUMINAMATH_CALUDE_pizza_dough_liquids_l3227_322775


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3227_322789

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  x = 63 → 
  y = x + 2 → 
  Odd x → 
  Odd y → 
  x + y = 128 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3227_322789


namespace NUMINAMATH_CALUDE_problem_solution_l3227_322723

structure Problem where
  -- Define the parabola
  p : ℝ
  parabola : ℝ → ℝ → Prop
  parabola_def : ∀ x y, parabola x y ↔ y^2 = 2*p*x

  -- Define points O, P, and Q
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ

  -- Define line l
  l : ℝ → ℝ → Prop

  -- Conditions
  O_is_origin : O = (0, 0)
  P_coordinates : P = (2, 1)
  p_positive : p > 0
  l_perpendicular_to_OP : (l 2 1) ∧ (∀ x y, l x y → (x - 2) = (y - 1) * (2 / 1))
  Q_on_l : l Q.1 Q.2
  Q_on_parabola : parabola Q.1 Q.2
  OPQ_right_isosceles : (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = (P.1 - O.1)^2 + (P.2 - O.2)^2

theorem problem_solution (prob : Problem) : prob.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3227_322723


namespace NUMINAMATH_CALUDE_zongzi_profit_maximization_l3227_322757

/-- Problem statement for zongzi profit maximization --/
theorem zongzi_profit_maximization 
  (cost_A cost_B : ℚ)  -- Cost prices of type A and B zongzi
  (sell_A sell_B : ℚ)  -- Selling prices of type A and B zongzi
  (total : ℕ)          -- Total number of zongzi to purchase
  :
  (cost_B = cost_A + 2) →  -- Condition 1
  (1000 / cost_A = 1200 / cost_B) →  -- Condition 2
  (sell_A = 12) →  -- Condition 5
  (sell_B = 15) →  -- Condition 6
  (total = 200) →  -- Condition 3
  ∃ (m : ℕ),  -- Number of type A zongzi purchased
    (m ≥ 2 * (total - m)) ∧  -- Condition 4
    (m < total) ∧
    (∀ (n : ℕ), n ≥ 2 * (total - n) → n < total →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) ≥
      (sell_A - cost_A) * n + (sell_B - cost_B) * (total - n)) ∧
    ((sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) = 466) ∧
    (m = 134) :=
by sorry

end NUMINAMATH_CALUDE_zongzi_profit_maximization_l3227_322757


namespace NUMINAMATH_CALUDE_divide_inequality_l3227_322750

theorem divide_inequality (x : ℝ) : -6 * x > 2 ↔ x < -1/3 := by sorry

end NUMINAMATH_CALUDE_divide_inequality_l3227_322750


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3227_322731

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}
  S = {x | -1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3227_322731


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3227_322772

theorem trigonometric_identities (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 2) : 
  (Real.sin θ * Real.cos θ = 1/2) ∧ 
  ((Real.sin θ + Real.cos θ)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3227_322772


namespace NUMINAMATH_CALUDE_total_fish_count_l3227_322742

/-- The total number of fish for all three tuna companies -/
def total_fish (jerk_tuna_tuna jerk_tuna_mackerel : ℕ) : ℕ :=
  let tall_tuna_tuna := 2 * jerk_tuna_tuna
  let tall_tuna_mackerel := jerk_tuna_mackerel + (30 * jerk_tuna_mackerel) / 100
  let swell_tuna_tuna := tall_tuna_tuna + (50 * tall_tuna_tuna) / 100
  let swell_tuna_mackerel := jerk_tuna_mackerel + (25 * jerk_tuna_mackerel) / 100
  jerk_tuna_tuna + jerk_tuna_mackerel +
  tall_tuna_tuna + tall_tuna_mackerel +
  swell_tuna_tuna + swell_tuna_mackerel

theorem total_fish_count :
  total_fish 144 80 = 1148 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_count_l3227_322742


namespace NUMINAMATH_CALUDE_third_angle_of_triangle_l3227_322737

theorem third_angle_of_triangle (a b c : ℝ) : 
  a + b + c = 180 → a = 50 → b = 80 → c = 50 := by sorry

end NUMINAMATH_CALUDE_third_angle_of_triangle_l3227_322737


namespace NUMINAMATH_CALUDE_point_coordinates_l3227_322743

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) 
  (h1 : secondQuadrant M)
  (h2 : distanceToXAxis M = 5)
  (h3 : distanceToYAxis M = 3) :
  M.x = -3 ∧ M.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3227_322743


namespace NUMINAMATH_CALUDE_function_inequality_l3227_322764

/-- For any differentiable function f on ℝ, if (x + 1)f'(x) ≥ 0 for all x in ℝ, 
    then f(0) + f(-2) ≥ 2f(-1) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, (x + 1) * deriv f x ≥ 0) : 
  f 0 + f (-2) ≥ 2 * f (-1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3227_322764


namespace NUMINAMATH_CALUDE_inequality_proof_l3227_322756

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 17/18)
  (hb : b = Real.cos (1/3))
  (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3227_322756


namespace NUMINAMATH_CALUDE_arithmetic_parabola_common_point_l3227_322776

/-- Represents a parabola with coefficients forming an arithmetic progression -/
structure ArithmeticParabola where
  a : ℝ
  d : ℝ

/-- The equation of the parabola given by y = ax^2 + bx + c where b = a + d and c = a + 2d -/
def ArithmeticParabola.equation (p : ArithmeticParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + (p.a + p.d) * x + (p.a + 2 * p.d)

/-- Theorem stating that all arithmetic parabolas pass through the point (-2, 0) -/
theorem arithmetic_parabola_common_point (p : ArithmeticParabola) :
  p.equation (-2) 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_parabola_common_point_l3227_322776


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l3227_322768

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 12 = 0

-- Define the roots of the equation
noncomputable def r₁ : ℝ := sorry
noncomputable def r₂ : ℝ := sorry

-- State the theorem
theorem absolute_difference_of_roots : 
  quadratic_equation r₁ ∧ quadratic_equation r₂ → |r₁ - r₂| = 1 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l3227_322768


namespace NUMINAMATH_CALUDE_correct_num_double_burgers_l3227_322712

/-- Represents the number of double burgers Caleb bought. -/
def num_double_burgers : ℕ := 37

/-- Represents the number of single burgers Caleb bought. -/
def num_single_burgers : ℕ := 50 - num_double_burgers

/-- The total cost of all burgers in cents. -/
def total_cost : ℕ := 6850

/-- The cost of a single burger in cents. -/
def single_burger_cost : ℕ := 100

/-- The cost of a double burger in cents. -/
def double_burger_cost : ℕ := 150

/-- The total number of burgers. -/
def total_burgers : ℕ := 50

theorem correct_num_double_burgers :
  num_single_burgers * single_burger_cost + num_double_burgers * double_burger_cost = total_cost ∧
  num_single_burgers + num_double_burgers = total_burgers :=
by sorry

end NUMINAMATH_CALUDE_correct_num_double_burgers_l3227_322712


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l3227_322777

theorem sqrt_difference_approximation : 
  |Real.sqrt (49 + 121) - Real.sqrt (64 - 36) - 7.75| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l3227_322777


namespace NUMINAMATH_CALUDE_equal_domain_function_iff_a_range_l3227_322736

/-- A function that maps a set onto itself --/
def EqualDomainFunction (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ x ∈ A, f x ∈ A ∧ ∀ y ∈ A, ∃ x ∈ A, f x = y

/-- The quadratic function f(x) = a(x-1)^2 - 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 2

theorem equal_domain_function_iff_a_range :
  ∀ a < 0, (∃ m n : ℝ, m < n ∧ EqualDomainFunction (f a) (Set.Icc m n)) ↔ -1/12 < a ∧ a < 0 := by
  sorry

#check equal_domain_function_iff_a_range

end NUMINAMATH_CALUDE_equal_domain_function_iff_a_range_l3227_322736


namespace NUMINAMATH_CALUDE_bakery_problem_proof_l3227_322739

/-- Given the total number of muffins, muffins per box, and available boxes, 
    calculate the number of additional boxes needed --/
def additional_boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : ℕ :=
  (total_muffins / muffins_per_box) - available_boxes

/-- Proof that 9 additional boxes are needed for the given bakery problem --/
theorem bakery_problem_proof : 
  additional_boxes_needed 95 5 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bakery_problem_proof_l3227_322739


namespace NUMINAMATH_CALUDE_range_of_a_l3227_322798

/-- The function f(x) = x³ - 3x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- The function g(x) = 2ˣ - a -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2^x - a

/-- The theorem statement -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 0 2, |f x₁ - g a x₂| ≤ 2) → 
  a ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3227_322798


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l3227_322746

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 1) = q * a n) ∧ 
  q ≠ 1

theorem geometric_sequence_sum_inequality 
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) : 
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l3227_322746


namespace NUMINAMATH_CALUDE_mirror_area_l3227_322760

/-- Given a rectangular mirror centered within two frames, where the outermost frame measures
    100 cm by 140 cm, and both frames have a width of 15 cm on each side, the area of the mirror
    is 3200 cm². -/
theorem mirror_area (outer_length outer_width frame_width : ℕ) 
  (h1 : outer_length = 100)
  (h2 : outer_width = 140)
  (h3 : frame_width = 15) : 
  (outer_length - 2 * frame_width - 2 * frame_width) * 
  (outer_width - 2 * frame_width - 2 * frame_width) = 3200 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l3227_322760


namespace NUMINAMATH_CALUDE_remainder_2_pow_13_mod_3_l3227_322792

theorem remainder_2_pow_13_mod_3 : 2^13 ≡ 2 [ZMOD 3] := by sorry

end NUMINAMATH_CALUDE_remainder_2_pow_13_mod_3_l3227_322792


namespace NUMINAMATH_CALUDE_permutations_of_three_objects_l3227_322769

theorem permutations_of_three_objects (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_three_objects_l3227_322769


namespace NUMINAMATH_CALUDE_triangle_area_l3227_322783

theorem triangle_area (a b c A B C : Real) (h1 : A = π/4) (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) : 
  (1/2) * b * c * Real.sin A = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3227_322783


namespace NUMINAMATH_CALUDE_intersection_M_N_l3227_322719

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | -5 < x ∧ x < 5 }

theorem intersection_M_N : M ∩ N = { x | -3 < x ∧ x < 5 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3227_322719


namespace NUMINAMATH_CALUDE_harry_worked_35_hours_l3227_322701

/-- Represents the pay structure and hours worked for Harry and James -/
structure PayStructure where
  x : ℝ  -- Base hourly rate
  james_overtime_rate : ℝ  -- James' overtime rate as a multiple of x
  harry_hours : ℕ  -- Total hours Harry worked
  harry_overtime : ℕ  -- Hours Harry worked beyond 21
  james_hours : ℕ  -- Total hours James worked
  james_overtime : ℕ  -- Hours James worked beyond 40

/-- Calculates Harry's total pay -/
def harry_pay (p : PayStructure) : ℝ :=
  21 * p.x + p.harry_overtime * (1.5 * p.x)

/-- Calculates James' total pay -/
def james_pay (p : PayStructure) : ℝ :=
  40 * p.x + p.james_overtime * (p.james_overtime_rate * p.x)

/-- Theorem stating that Harry worked 35 hours given the problem conditions -/
theorem harry_worked_35_hours :
  ∀ (p : PayStructure),
    p.james_hours = 41 →
    p.james_overtime = 1 →
    p.harry_hours = p.harry_overtime + 21 →
    harry_pay p = james_pay p →
    p.harry_hours = 35 := by
  sorry


end NUMINAMATH_CALUDE_harry_worked_35_hours_l3227_322701


namespace NUMINAMATH_CALUDE_total_birds_in_marsh_l3227_322794

theorem total_birds_in_marsh (geese ducks swans : ℕ) 
  (h1 : geese = 58) 
  (h2 : ducks = 37) 
  (h3 : swans = 42) : 
  geese + ducks + swans = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_marsh_l3227_322794


namespace NUMINAMATH_CALUDE_sum_of_integers_l3227_322705

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3227_322705


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3227_322799

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 + 2*k*x + 1 = 0) ↔ k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3227_322799


namespace NUMINAMATH_CALUDE_debate_team_group_size_l3227_322726

theorem debate_team_group_size (boys girls groups : ℕ) (h1 : boys = 28) (h2 : girls = 4) (h3 : groups = 8) :
  (boys + girls) / groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l3227_322726


namespace NUMINAMATH_CALUDE_employee_y_pay_l3227_322784

/-- Represents the weekly pay of employees x, y, and z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total pay for all employees -/
def totalPay (pay : EmployeePay) : ℝ :=
  pay.x + pay.y + pay.z

/-- Theorem: Given the conditions, employee y's pay is 478.125 -/
theorem employee_y_pay :
  ∀ (pay : EmployeePay),
    totalPay pay = 1550 →
    pay.x = 1.2 * pay.y →
    pay.z = pay.y - 30 + 50 →
    pay.y = 478.125 := by
  sorry


end NUMINAMATH_CALUDE_employee_y_pay_l3227_322784


namespace NUMINAMATH_CALUDE_complex_root_of_unity_sum_l3227_322755

theorem complex_root_of_unity_sum (ω : ℂ) : 
  ω = -1/2 + (Complex.I * Real.sqrt 3) / 2 → ω^4 + ω^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_of_unity_sum_l3227_322755


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l3227_322710

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Prove that 16 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 16 → num_factors m ≠ 8) ∧ num_factors 16 = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l3227_322710


namespace NUMINAMATH_CALUDE_tammy_running_schedule_l3227_322747

/-- Calculates the number of loops per day given weekly distance goal, track length, and days per week -/
def loops_per_day (weekly_goal : ℕ) (track_length : ℕ) (days_per_week : ℕ) : ℕ :=
  (weekly_goal / track_length) / days_per_week

theorem tammy_running_schedule :
  loops_per_day 3500 50 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tammy_running_schedule_l3227_322747


namespace NUMINAMATH_CALUDE_bryan_total_books_l3227_322766

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The number of bookshelves Bryan has -/
def number_of_shelves : ℕ := 23

/-- The total number of books Bryan has -/
def total_books : ℕ := books_per_shelf * number_of_shelves

theorem bryan_total_books : total_books = 621 := by
  sorry

end NUMINAMATH_CALUDE_bryan_total_books_l3227_322766


namespace NUMINAMATH_CALUDE_sum_of_digits_of_N_l3227_322770

def N : ℕ := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_N : sum_of_digits N = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_N_l3227_322770


namespace NUMINAMATH_CALUDE_sum_at_thirteenth_position_l3227_322709

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all orientations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of numbers in the 13th position from the left across all orientations of a regular 100-gon is 10100 -/
theorem sum_at_thirteenth_position (p : RegularPolygon 100) :
  sum_at_position p 13 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_at_thirteenth_position_l3227_322709


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l3227_322744

theorem right_triangle_base_length 
  (height : ℝ) 
  (perimeter : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : height = 3) 
  (h2 : perimeter = 12) 
  (h3 : is_right_triangle = true) : 
  ∃ (base : ℝ), base = 4 ∧ 
  ∃ (hypotenuse : ℝ), 
    perimeter = base + height + hypotenuse ∧
    hypotenuse^2 = base^2 + height^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l3227_322744


namespace NUMINAMATH_CALUDE_exercise_239_theorem_existence_not_implied_l3227_322724

-- Define a property A for functions
def PropertyA (f : ℝ → ℝ) : Prop := sorry

-- Define periodicity for functions
def Periodic (f : ℝ → ℝ) : Prop := ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- The theorem from exercise 239
theorem exercise_239_theorem : ∀ f : ℝ → ℝ, PropertyA f → Periodic f := sorry

-- The statement we want to prove
theorem existence_not_implied :
  (∀ f : ℝ → ℝ, PropertyA f → Periodic f) →
  ¬(∃ f : ℝ → ℝ, PropertyA f) := sorry

end NUMINAMATH_CALUDE_exercise_239_theorem_existence_not_implied_l3227_322724


namespace NUMINAMATH_CALUDE_pumpkin_ravioli_weight_l3227_322722

theorem pumpkin_ravioli_weight (brother_ravioli_count : ℕ) (total_weight : ℝ) : 
  brother_ravioli_count = 12 → total_weight = 15 → 
  (total_weight / brother_ravioli_count : ℝ) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_ravioli_weight_l3227_322722


namespace NUMINAMATH_CALUDE_random_walk_properties_l3227_322749

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- Number of right steps
  b : ℕ  -- Number of left steps
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Main theorem about the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry


end NUMINAMATH_CALUDE_random_walk_properties_l3227_322749


namespace NUMINAMATH_CALUDE_units_digit_product_l3227_322796

theorem units_digit_product : (17 * 59 * 23) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l3227_322796


namespace NUMINAMATH_CALUDE_product_of_three_integers_l3227_322735

theorem product_of_three_integers (A B C : ℤ) 
  (sum_eq : A + B + C = 33)
  (largest_eq : C = 3 * B)
  (smallest_eq : A = C - 23) :
  A * B * C = 192 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l3227_322735


namespace NUMINAMATH_CALUDE_soccer_team_losses_l3227_322711

theorem soccer_team_losses (total_games : ℕ) (games_won : ℕ) (points_for_win : ℕ) 
  (points_for_draw : ℕ) (points_for_loss : ℕ) (total_points : ℕ) :
  total_games = 20 →
  games_won = 14 →
  points_for_win = 3 →
  points_for_draw = 1 →
  points_for_loss = 0 →
  total_points = 46 →
  ∃ (games_lost : ℕ) (games_drawn : ℕ),
    games_lost = 2 ∧
    games_won + games_drawn + games_lost = total_games ∧
    games_won * points_for_win + games_drawn * points_for_draw + games_lost * points_for_loss = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_losses_l3227_322711


namespace NUMINAMATH_CALUDE_problem_2003_2001_l3227_322778

theorem problem_2003_2001 : 2003^3 - 2001 * 2003^2 - 2001^2 * 2003 + 2001^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_2003_2001_l3227_322778


namespace NUMINAMATH_CALUDE_widest_strip_width_l3227_322765

theorem widest_strip_width (bolt_width_1 bolt_width_2 : ℕ) 
  (h1 : bolt_width_1 = 45) 
  (h2 : bolt_width_2 = 60) : 
  Nat.gcd bolt_width_1 bolt_width_2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_widest_strip_width_l3227_322765


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l3227_322779

/-- The least possible side length of a square when measured as 7 cm to the nearest centimeter -/
def least_possible_side : ℝ := 6.5

/-- The measured side length of the square to the nearest centimeter -/
def measured_side : ℕ := 7

/-- The least possible area of the square -/
def least_possible_area : ℝ := least_possible_side ^ 2

theorem least_possible_area_of_square :
  least_possible_side ≥ (measured_side : ℝ) - 0.5 ∧
  least_possible_side < (measured_side : ℝ) ∧
  least_possible_area = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l3227_322779


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l3227_322733

theorem same_remainder_divisor : ∃! (n : ℕ), n > 0 ∧ 
  ∃ (r : ℕ), r > 0 ∧ r < n ∧ 
  (2287 % n = r) ∧ (2028 % n = r) ∧ (1806 % n = r) :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l3227_322733


namespace NUMINAMATH_CALUDE_gold_coin_value_l3227_322704

theorem gold_coin_value :
  let silver_coin_value : ℕ := 25
  let gold_coins : ℕ := 3
  let silver_coins : ℕ := 5
  let cash : ℕ := 30
  let total_value : ℕ := 305
  ∃ (gold_coin_value : ℕ),
    gold_coin_value * gold_coins + silver_coin_value * silver_coins + cash = total_value ∧
    gold_coin_value = 50 :=
by sorry

end NUMINAMATH_CALUDE_gold_coin_value_l3227_322704


namespace NUMINAMATH_CALUDE_profit_difference_A_C_l3227_322758

-- Define the profit-sharing ratios
def ratio_A : ℕ := 3
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6
def ratio_D : ℕ := 7

-- Define B's profit share
def profit_B : ℕ := 2000

-- Theorem statement
theorem profit_difference_A_C : 
  let part_value : ℚ := profit_B / ratio_B
  let profit_A : ℚ := part_value * ratio_A
  let profit_C : ℚ := part_value * ratio_C
  profit_C - profit_A = 1200 := by sorry

end NUMINAMATH_CALUDE_profit_difference_A_C_l3227_322758


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3227_322781

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 30 (n + 80) = 15 ∧ 
  Nat.gcd (n + 30) 100 = 50 ∧
  (∀ m : ℕ, m > 1000 → 
    (Nat.gcd 30 (m + 80) = 15 ∧ Nat.gcd (m + 30) 100 = 50) → 
    m ≥ n) ∧
  n = 1270 :=
sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3227_322781


namespace NUMINAMATH_CALUDE_disinfectant_sales_problem_l3227_322748

-- Define the range of x
def valid_x (x : ℤ) : Prop := 8 ≤ x ∧ x ≤ 15

-- Define the linear function
def y (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def w (x : ℤ) : ℤ := (x - 8) * (-5 * x + 150)

theorem disinfectant_sales_problem :
  (∀ x : ℤ, valid_x x → 
    (x = 9 → y x = 105) ∧ 
    (x = 11 → y x = 95) ∧ 
    (x = 13 → y x = 85)) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 425 ∧ x = 13) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 525 ∧ x = 15 ∧ ∀ x' : ℤ, valid_x x' → w x' ≤ w x) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_sales_problem_l3227_322748


namespace NUMINAMATH_CALUDE_train_length_l3227_322761

/-- Proves that a train passing through a tunnel under specific conditions has a length of 100 meters -/
theorem train_length (tunnel_length : ℝ) (total_time : ℝ) (inside_time : ℝ) 
  (h1 : tunnel_length = 500)
  (h2 : total_time = 30)
  (h3 : inside_time = 20)
  (h4 : total_time > 0)
  (h5 : inside_time > 0)
  (h6 : total_time > inside_time) :
  ∃ (train_length : ℝ), 
    train_length = 100 ∧ 
    (tunnel_length + train_length) / total_time = (tunnel_length - train_length) / inside_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_l3227_322761


namespace NUMINAMATH_CALUDE_set_A_at_most_one_element_iff_a_in_range_l3227_322791

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

-- Theorem statement
theorem set_A_at_most_one_element_iff_a_in_range :
  ∀ a : ℝ, (∃ (x y : ℝ), x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ a ∈ {a : ℝ | a < -1 ∨ (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)} :=
by sorry

end NUMINAMATH_CALUDE_set_A_at_most_one_element_iff_a_in_range_l3227_322791


namespace NUMINAMATH_CALUDE_exists_element_with_mass_percentage_l3227_322752

/-- Molar mass of Hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of Bromine in g/mol -/
def molar_mass_Br : ℝ := 79.90

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of HBrO3 in g/mol -/
def molar_mass_HBrO3 : ℝ := molar_mass_H + molar_mass_Br + 3 * molar_mass_O

/-- Mass percentage of a certain element in HBrO3 -/
def target_mass_percentage : ℝ := 0.78

theorem exists_element_with_mass_percentage :
  ∃ (element_mass : ℝ), 
    0 < element_mass ∧ 
    element_mass ≤ molar_mass_HBrO3 ∧
    (element_mass / molar_mass_HBrO3) * 100 = target_mass_percentage :=
by sorry

end NUMINAMATH_CALUDE_exists_element_with_mass_percentage_l3227_322752


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_division_l3227_322763

/-- Given a triangle with sides a, b, and c, where c is the longest side,
    and a circle touching sides a and b with its center on side c,
    prove that the center divides c into segments of length x and y. -/
theorem circle_tangent_triangle_division (a b c x y : ℝ) : 
  a = 12 → b = 15 → c = 18 → c > a ∧ c > b →
  x + y = c → x / y = a / b →
  x = 8 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_division_l3227_322763


namespace NUMINAMATH_CALUDE_contradictory_statement_l3227_322721

theorem contradictory_statement (x : ℝ) :
  (∀ x, x + 3 ≥ 0 → x ≥ -3) ↔ (∀ x, x + 3 < 0 → x < -3) :=
by sorry

end NUMINAMATH_CALUDE_contradictory_statement_l3227_322721


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3227_322728

theorem fraction_power_equality : (81000 ^ 5) / (27000 ^ 5) = 243 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3227_322728


namespace NUMINAMATH_CALUDE_inequality_proof_l3227_322706

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3227_322706


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l3227_322753

/-- A point on a parabola with a specific distance to its directrix -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*x
  distance_to_directrix : x + 1/2 = 2

/-- The coordinates of the point are (3/2, ±√3) -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  p.x = 3/2 ∧ (p.y = Real.sqrt 3 ∨ p.y = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l3227_322753


namespace NUMINAMATH_CALUDE_larger_triangle_side_l3227_322797

/-- Two similar triangles with specified properties -/
structure SimilarTriangles where
  area_small : ℕ
  area_large : ℕ
  side_small : ℕ
  ratio : ℕ
  area_diff : area_large - area_small = 32
  area_ratio : area_large = ratio ^ 2 * area_small
  side_small_val : side_small = 4

/-- The theorem stating the corresponding side of the larger triangle -/
theorem larger_triangle_side (t : SimilarTriangles) : 
  ∃ (side_large : ℕ), side_large = 12 ∧ 
    side_large * side_large * t.area_small = t.side_small * t.side_small * t.area_large := by
  sorry

#check larger_triangle_side

end NUMINAMATH_CALUDE_larger_triangle_side_l3227_322797


namespace NUMINAMATH_CALUDE_three_boxes_of_five_balls_l3227_322793

/-- Calculates the total number of balls given the number of boxes and balls per box -/
def totalBalls (numBoxes : ℕ) (ballsPerBox : ℕ) : ℕ :=
  numBoxes * ballsPerBox

/-- Proves that the total number of balls is 15 when there are 3 boxes with 5 balls each -/
theorem three_boxes_of_five_balls :
  totalBalls 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_three_boxes_of_five_balls_l3227_322793


namespace NUMINAMATH_CALUDE_town_population_growth_l3227_322715

/-- Represents the population of a town over time -/
structure TownPopulation where
  pop1991 : Nat
  pop2006 : Nat
  pop2016 : Nat

/-- Conditions for the town population -/
def ValidTownPopulation (t : TownPopulation) : Prop :=
  ∃ (n m k : Nat),
    t.pop1991 = n * n ∧
    t.pop2006 = t.pop1991 + 120 ∧
    t.pop2006 = m * m - 1 ∧
    t.pop2016 = t.pop2006 + 180 ∧
    t.pop2016 = k * k

/-- Calculate percent growth -/
def PercentGrowth (initial : Nat) (final : Nat) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Main theorem stating the percent growth is 5% -/
theorem town_population_growth (t : TownPopulation) 
  (h : ValidTownPopulation t) : 
  PercentGrowth t.pop1991 t.pop2016 = 5 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l3227_322715


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l3227_322732

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

noncomputable def f_deriv : ℝ → ℝ := deriv f

theorem f_sum_equals_two :
  f 2017 + f_deriv 2017 + f (-2017) - f_deriv (-2017) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l3227_322732


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3227_322707

-- Define the expression
def expression (a b : ℝ) : ℝ := -4 * a^2 + b^2

-- Theorem: The expression can be factored using the difference of squares formula
theorem difference_of_squares_factorization (a b : ℝ) :
  ∃ (x y : ℝ), expression a b = (x + y) * (x - y) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3227_322707


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3227_322754

theorem rope_cutting_problem : Nat.gcd 42 (Nat.gcd 56 (Nat.gcd 63 77)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3227_322754


namespace NUMINAMATH_CALUDE_dilation_problem_l3227_322734

/-- Dilation of a complex number -/
def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-1 + 2*I) 4 (3 + 4*I) = 15 + 10*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l3227_322734


namespace NUMINAMATH_CALUDE_rabbit_count_l3227_322767

theorem rabbit_count (total_legs : ℕ) (rabbit_chicken_diff : ℕ) : 
  total_legs = 250 → rabbit_chicken_diff = 53 → 
  ∃ (rabbits : ℕ), 
    rabbits + rabbit_chicken_diff = total_legs / 2 ∧
    4 * rabbits + 2 * (rabbits + rabbit_chicken_diff) = total_legs ∧
    rabbits = 24 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l3227_322767


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l3227_322773

/-- Represents a 5x5 grid of dots -/
def Grid : Type := Unit

/-- The number of dots in the grid -/
def num_dots : ℕ := 25

/-- The number of ways to choose 4 collinear dots from the grid -/
def num_collinear_sets : ℕ := 54

/-- The total number of ways to choose 4 dots from the grid -/
def total_combinations : ℕ := 12650

/-- The probability of selecting 4 collinear dots when choosing 4 dots at random -/
def collinear_probability (g : Grid) : ℚ := 6 / 1415

theorem collinear_dots_probability (g : Grid) : 
  collinear_probability g = num_collinear_sets / total_combinations :=
by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l3227_322773


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3227_322703

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 < 0 ↔ -1/2 < x ∧ x < 1) →
  (∀ x, (a*x + 2) / (b*x + 1) < 0 ↔ x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3227_322703


namespace NUMINAMATH_CALUDE_angela_marbles_l3227_322795

theorem angela_marbles :
  ∀ (a : ℕ), 
  (∃ (b c d : ℕ),
    b = 3 * a ∧
    c = 2 * b ∧
    d = 4 * c ∧
    a + b + c + d = 204) →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_angela_marbles_l3227_322795


namespace NUMINAMATH_CALUDE_pear_juice_percentage_approx_19_23_l3227_322716

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  pears : ℕ
  pearJuice : ℚ
  oranges : ℕ
  orangeJuice : ℚ

/-- Represents the blend composition -/
structure Blend where
  pears : ℕ
  oranges : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (yield : JuiceYield) (blend : Blend) : ℚ :=
  let pear_juice := (blend.pears : ℚ) * yield.pearJuice / yield.pears
  let orange_juice := (blend.oranges : ℚ) * yield.orangeJuice / yield.oranges
  let total_juice := pear_juice + orange_juice
  pear_juice / total_juice * 100

theorem pear_juice_percentage_approx_19_23 (yield : JuiceYield) (blend : Blend) :
  yield.pears = 4 ∧ 
  yield.pearJuice = 10 ∧ 
  yield.oranges = 1 ∧ 
  yield.orangeJuice = 7 ∧
  blend.pears = 8 ∧
  blend.oranges = 12 →
  abs (pear_juice_percentage yield blend - 19.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_approx_19_23_l3227_322716


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3227_322788

theorem inequality_holds_iff (n : ℕ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3227_322788
