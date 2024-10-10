import Mathlib

namespace custom_multiplication_prove_specific_case_l330_33044

theorem custom_multiplication (x y : ℤ) : x * y = x * y - 2 * (x + y) := by sorry

theorem prove_specific_case : 1 * (-3) = 1 := by sorry

end custom_multiplication_prove_specific_case_l330_33044


namespace prime_pairs_satisfying_equation_l330_33097

theorem prime_pairs_satisfying_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    p * (p + 1) + q * (q + 1) = n * (n + 1) →
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) ∨ (p = 2 ∧ q = 2)) :=
by sorry

end prime_pairs_satisfying_equation_l330_33097


namespace square_difference_theorem_l330_33022

theorem square_difference_theorem (x : ℝ) (h : (x + 2) * (x - 2) = 1221) : 
  x^2 = 1225 ∧ (x + 1) * (x - 1) = 1224 := by
  sorry

end square_difference_theorem_l330_33022


namespace power_of_128_l330_33015

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  have h1 : (128 : ℝ) = 2^7 := by sorry
  sorry

end power_of_128_l330_33015


namespace zero_only_universal_prime_multiple_l330_33081

theorem zero_only_universal_prime_multiple : ∃! n : ℤ, ∀ p : ℕ, Prime p → ∃ k : ℤ, n * p = k * p :=
sorry

end zero_only_universal_prime_multiple_l330_33081


namespace medium_apple_cost_l330_33080

/-- Proves that the cost of a medium apple is $2 given the conditions in the problem -/
theorem medium_apple_cost (small_apple_cost big_apple_cost total_cost : ℝ)
  (small_medium_count big_count : ℕ) :
  small_apple_cost = 1.5 →
  big_apple_cost = 3 →
  small_medium_count = 6 →
  big_count = 8 →
  total_cost = 45 →
  ∃ (medium_apple_cost : ℝ),
    small_apple_cost * (small_medium_count / 2) +
    medium_apple_cost * (small_medium_count / 2) +
    big_apple_cost * big_count = total_cost ∧
    medium_apple_cost = 2 :=
by sorry

end medium_apple_cost_l330_33080


namespace place_values_in_9890_l330_33031

theorem place_values_in_9890 : 
  ∃ (thousands hundreds tens : ℕ),
    9890 = thousands * 1000 + hundreds * 100 + tens * 10 + (9890 % 10) ∧
    thousands = 9 ∧
    hundreds = 8 ∧
    tens = 9 :=
by sorry

end place_values_in_9890_l330_33031


namespace problem_statement_l330_33093

theorem problem_statement (p q : Prop) 
  (hp : p ↔ 3 % 2 = 1) 
  (hq : q ↔ 5 % 2 = 0) : 
  p ∨ q := by sorry

end problem_statement_l330_33093


namespace min_value_zero_iff_k_eq_one_l330_33025

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 3*x^2 - 4*k*x*y + (2*k^2 + 1)*y^2 - 6*x - 2*y + 4

/-- The theorem stating that the minimum value of f is 0 iff k = 1 -/
theorem min_value_zero_iff_k_eq_one :
  (∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f 1 x y ≥ m) ∧
  (∀ k : ℝ, k ≠ 1 → ¬∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f k x y ≥ m) :=
sorry

end min_value_zero_iff_k_eq_one_l330_33025


namespace solution_of_equations_solution_of_inequalities_l330_33065

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solution_of_equations : 
  ∃ x y : ℝ, system_of_equations x y ∧ x = 4 ∧ y = 5 := by sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solution_of_inequalities : 
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 2 := by sorry

end solution_of_equations_solution_of_inequalities_l330_33065


namespace lillian_initial_candies_l330_33014

-- Define the variables
def initial_candies : ℕ := sorry
def father_gave : ℕ := 5
def total_candies : ℕ := 93

-- State the theorem
theorem lillian_initial_candies : 
  initial_candies + father_gave = total_candies → initial_candies = 88 :=
by
  sorry

end lillian_initial_candies_l330_33014


namespace market_equilibrium_and_max_revenue_l330_33024

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - 4 * P

-- Define the supply function (to be proven)
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the tax revenue function
def tax_revenue (t : ℝ) (Q : ℝ) : ℝ := t * Q

-- Theorem statement
theorem market_equilibrium_and_max_revenue :
  -- Conditions
  let change_ratio : ℝ := 1.5
  let production_tax : ℝ := 90
  let producer_price : ℝ := 64

  -- Prove that the supply function is correct
  ∀ P, supply_function P = 6 * P - 312 ∧
  
  -- Prove that the maximum tax revenue is 8640
  ∃ t_optimal, 
    let Q_optimal := demand_function (producer_price + t_optimal)
    tax_revenue t_optimal Q_optimal = 8640 ∧
    ∀ t, tax_revenue t (demand_function (producer_price + t)) ≤ 8640 :=
by sorry

end market_equilibrium_and_max_revenue_l330_33024


namespace max_pieces_on_board_l330_33003

/-- Represents a piece on the grid -/
inductive Piece
| Red
| Blue

/-- Represents a cell on the grid -/
structure Cell :=
(row : Nat)
(col : Nat)
(piece : Option Piece)

/-- Represents the game board -/
structure Board :=
(cells : List Cell)
(rowCount : Nat)
(colCount : Nat)

/-- Checks if a cell contains a piece -/
def Cell.hasPiece (cell : Cell) : Bool :=
  cell.piece.isSome

/-- Counts the number of pieces on the board -/
def Board.pieceCount (board : Board) : Nat :=
  board.cells.filter Cell.hasPiece |>.length

/-- Checks if a piece sees exactly five pieces of the other color in its row and column -/
def Board.validPiecePlacement (board : Board) (cell : Cell) : Bool :=
  sorry

/-- Checks if all pieces on the board satisfy the placement rule -/
def Board.validBoard (board : Board) : Bool :=
  board.cells.all (Board.validPiecePlacement board)

theorem max_pieces_on_board (board : Board) :
  board.rowCount = 200 ∧ board.colCount = 200 ∧ board.validBoard →
  board.pieceCount ≤ 3800 :=
sorry

end max_pieces_on_board_l330_33003


namespace real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l330_33005

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 5*x + m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

-- Define the additional condition for part 2
def satisfies_root_condition (x₁ x₂ : ℝ) : Prop := 3*x₁ - 2*x₂ = 5

-- Theorem 1: Equation has real roots iff m ≤ 25/4
theorem real_roots_iff_m_le_25_4 :
  ∀ m : ℝ, has_real_roots m ↔ m ≤ 25/4 :=
sorry

-- Theorem 2: If equation has two real roots satisfying the condition, then m = 6
theorem m_eq_6_when_condition_satisfied :
  ∀ m : ℝ, has_two_distinct_real_roots m →
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ satisfies_root_condition x₁ x₂) →
  m = 6 :=
sorry

end real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l330_33005


namespace friend_total_time_l330_33033

def my_reading_time : ℝ := 3 * 60 -- 3 hours in minutes
def my_writing_time : ℝ := 60 -- 1 hour in minutes
def friend_reading_speed_ratio : ℝ := 4 -- friend reads 4 times as fast

theorem friend_total_time (friend_reading_time friend_writing_time : ℝ) :
  friend_reading_time = my_reading_time / friend_reading_speed_ratio →
  friend_writing_time = my_writing_time →
  friend_reading_time + friend_writing_time = 105 := by
sorry

end friend_total_time_l330_33033


namespace sin_increasing_on_interval_l330_33090

-- Define the sine function (already defined in Mathlib)
-- def sin : ℝ → ℝ := Real.sin

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- State the theorem
theorem sin_increasing_on_interval :
  StrictMonoOn Real.sin interval :=
sorry

end sin_increasing_on_interval_l330_33090


namespace complex_number_problem_l330_33094

open Complex

theorem complex_number_problem (z : ℂ) (a b : ℝ) : 
  z = ((1 + I)^2 + 2*(5 - I)) / (3 + I) →
  abs z = Real.sqrt 10 ∧
  (z * (z + a) = b + I → a = -7 ∧ b = -13) :=
by sorry

end complex_number_problem_l330_33094


namespace smallest_number_with_given_remainders_l330_33069

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_number_with_given_remainders_l330_33069


namespace sandy_savings_l330_33073

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ)
  (salary_increase_rate : ℝ) (savings_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase_rate = 0.10 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * last_year_savings_rate * last_year_salary) /
  (last_year_salary * (1 + salary_increase_rate)) = 0.09 := by
  sorry

end sandy_savings_l330_33073


namespace rectangle_area_increase_l330_33064

/-- Theorem: When the sides of a rectangle are increased by 35%, the area increases by 82.25% -/
theorem rectangle_area_increase (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let original_area := L * W
  let new_length := L * 1.35
  let new_width := W * 1.35
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 82.25 := by
  sorry

#check rectangle_area_increase

end rectangle_area_increase_l330_33064


namespace newspaper_delivery_start_l330_33099

def building_floors : ℕ := 20

def start_floor : ℕ → Prop
| f => ∃ (current : ℕ), 
    current = f + 5 - 2 + 7 ∧ 
    current = building_floors - 9

theorem newspaper_delivery_start : start_floor 1 := by
  sorry

end newspaper_delivery_start_l330_33099


namespace shelf_adjustment_theorem_l330_33007

/-- The number of items on the shelf -/
def total_items : ℕ := 12

/-- The initial number of items on the upper layer -/
def initial_upper : ℕ := 4

/-- The initial number of items on the lower layer -/
def initial_lower : ℕ := 8

/-- The number of items to be moved from lower to upper layer -/
def items_to_move : ℕ := 2

/-- The number of ways to adjust the items -/
def adjustment_ways : ℕ := Nat.choose initial_lower items_to_move

theorem shelf_adjustment_theorem : adjustment_ways = 840 := by sorry

end shelf_adjustment_theorem_l330_33007


namespace intersection_angle_cosine_l330_33020

/-- The cosine of the angle formed by the foci and an intersection point of an ellipse and hyperbola with common foci -/
theorem intersection_angle_cosine 
  (x y : ℝ) 
  (ellipse_eq : x^2/6 + y^2/2 = 1) 
  (hyperbola_eq : x^2/3 - y^2 = 1) 
  (is_intersection : x^2/6 + y^2/2 = 1 ∧ x^2/3 - y^2 = 1) : 
  ∃ (f₁_x f₁_y f₂_x f₂_y : ℝ), 
    let f₁ := (f₁_x, f₁_y)
    let f₂ := (f₂_x, f₂_y)
    let p := (x, y)
    let v₁ := (x - f₁_x, y - f₁_y)
    let v₂ := (x - f₂_x, y - f₂_y)
    (f₁.1^2/6 + f₁.2^2/2 < 1 ∧ f₂.1^2/6 + f₂.2^2/2 < 1) ∧  -- f₁ and f₂ are inside the ellipse
    (f₁.1^2/3 - f₁.2^2 > 1 ∧ f₂.1^2/3 - f₂.2^2 > 1) ∧      -- f₁ and f₂ are outside the hyperbola
    (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = 1/3 :=
by sorry

end intersection_angle_cosine_l330_33020


namespace equal_area_rectangles_width_l330_33086

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 20 inches
    and the other rectangle is 6 inches long, prove that the width of the second rectangle
    is 50 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
  (h1 : carol_length = 15)
  (h2 : carol_width = 20)
  (h3 : jordan_length = 6)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 50 := by
  sorry

end equal_area_rectangles_width_l330_33086


namespace multiple_properties_l330_33092

theorem multiple_properties (a b : ℤ) 
  (h1 : ∃ k : ℤ, a = 5 * k)
  (h2 : ∃ m : ℤ, a = 2 * m + 1)
  (h3 : ∃ n : ℤ, b = 10 * n) :
  (∃ p : ℤ, b = 5 * p) ∧ (∃ q : ℤ, a - b = 5 * q) := by
sorry

end multiple_properties_l330_33092


namespace quadratic_discriminant_l330_33088

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 2x^2 + (2 + 1/2)x + 1/2 has discriminant 2.25 -/
theorem quadratic_discriminant : discriminant 2 (2 + 1/2) (1/2) = 2.25 := by
  sorry

end quadratic_discriminant_l330_33088


namespace geometric_sequence_third_term_l330_33091

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 27, prove that a₃ = 9 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h4 : a 4 = 27)  -- Fourth term is 27
  (h_geom : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a n = a 1 * q^(n-1))  -- Definition of geometric sequence
  : a 3 = 9 := by
sorry

end geometric_sequence_third_term_l330_33091


namespace optimal_plan_is_correct_l330_33070

/-- Represents the number of cars a worker can install per month -/
structure WorkerProductivity where
  skilled : ℕ
  new : ℕ

/-- Represents the monthly salary of workers -/
structure WorkerSalary where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

def optimal_plan (prod : WorkerProductivity) (salary : WorkerSalary) : RecruitmentPlan :=
  sorry

theorem optimal_plan_is_correct (prod : WorkerProductivity) (salary : WorkerSalary) :
  let plan := optimal_plan prod salary
  prod.skilled * plan.skilled + prod.new * plan.new = 20 ∧
  ∀ other : RecruitmentPlan,
    prod.skilled * other.skilled + prod.new * other.new = 20 →
    salary.skilled * plan.skilled + salary.new * plan.new ≤
    salary.skilled * other.skilled + salary.new * other.new :=
by
  sorry

#check @optimal_plan_is_correct

end optimal_plan_is_correct_l330_33070


namespace only_third_equation_has_nontrivial_solution_l330_33063

theorem only_third_equation_has_nontrivial_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (Real.sqrt (a^2 + b^2) = a + 2*b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a - b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2 - b^2) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2*b - a*b^2) :=
by sorry

end only_third_equation_has_nontrivial_solution_l330_33063


namespace solve_equation_l330_33067

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 5) = 2) : x = -3 := by
  sorry

end solve_equation_l330_33067


namespace range_of_a_l330_33027

theorem range_of_a (a : ℝ) (ha : a ≠ 0) : 
  let A := {x : ℝ | x^2 - x - 6 < 0}
  let B := {x : ℝ | x^2 + 2*x - 8 ≥ 0}
  let C := {x : ℝ | x^2 - 4*a*x + 3*a^2 < 0}
  C ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
by sorry

end range_of_a_l330_33027


namespace negative_exponent_division_l330_33029

theorem negative_exponent_division (a : ℝ) : -a^6 / a^3 = -a^3 := by
  sorry

end negative_exponent_division_l330_33029


namespace negative_distribution_l330_33095

theorem negative_distribution (a b c : ℝ) : -(a - b + c) = -a + b - c := by
  sorry

end negative_distribution_l330_33095


namespace three_digit_divisible_by_13_and_3_l330_33036

theorem three_digit_divisible_by_13_and_3 : 
  (Finset.filter (fun n => n % 13 = 0 ∧ n % 3 = 0) (Finset.range 900)).card = 23 :=
by
  sorry

end three_digit_divisible_by_13_and_3_l330_33036


namespace B_power_87_l330_33058

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_87 : B ^ 87 = ![![0,  1, 0],
                                 ![-1, 0, 0],
                                 ![0,  0, 0]] := by
  sorry

end B_power_87_l330_33058


namespace monotonic_decreasing_implies_order_l330_33057

theorem monotonic_decreasing_implies_order (f : ℝ → ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) :
  f 3 < f 2 ∧ f 2 < f 1 :=
by sorry

end monotonic_decreasing_implies_order_l330_33057


namespace range_of_a_l330_33018

-- Define the set A
def A (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (a * x^2 + 2*(a-1)*x - 4)}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a = Set.Ici 0) ↔ Set.Ici 0 = {a : ℝ | 0 ≤ a} := by sorry

end range_of_a_l330_33018


namespace select_two_from_four_l330_33037

theorem select_two_from_four : Nat.choose 4 2 = 6 := by sorry

end select_two_from_four_l330_33037


namespace geometric_sequence_property_l330_33079

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prod : a 2 * a 4 = 1/2) :
  a 1 * a 3^2 * a 5 = 1/4 := by
  sorry

end geometric_sequence_property_l330_33079


namespace p_and_q_true_l330_33047

theorem p_and_q_true (h : ¬(¬(p ∧ q))) : p ∧ q := by
  sorry

end p_and_q_true_l330_33047


namespace game_points_total_l330_33077

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  eric_points + mark_points + samanta_points = 32 := by
sorry

end game_points_total_l330_33077


namespace race_distance_is_17_l330_33075

/-- Represents the relay race with given conditions -/
structure RelayRace where
  totalTime : Real
  sadieTime : Real
  sadieSpeed : Real
  arianaTime : Real
  arianaSpeed : Real
  sarahSpeed : Real

/-- Calculates the total distance of the relay race -/
def totalDistance (race : RelayRace) : Real :=
  let sadieDistance := race.sadieTime * race.sadieSpeed
  let arianaDistance := race.arianaTime * race.arianaSpeed
  let sarahTime := race.totalTime - race.sadieTime - race.arianaTime
  let sarahDistance := sarahTime * race.sarahSpeed
  sadieDistance + arianaDistance + sarahDistance

/-- Theorem stating that the total distance of the given race is 17 miles -/
theorem race_distance_is_17 (race : RelayRace) 
  (h1 : race.totalTime = 4.5)
  (h2 : race.sadieTime = 2)
  (h3 : race.sadieSpeed = 3)
  (h4 : race.arianaTime = 0.5)
  (h5 : race.arianaSpeed = 6)
  (h6 : race.sarahSpeed = 4) :
  totalDistance race = 17 := by
  sorry

#eval totalDistance { totalTime := 4.5, sadieTime := 2, sadieSpeed := 3, arianaTime := 0.5, arianaSpeed := 6, sarahSpeed := 4 }

end race_distance_is_17_l330_33075


namespace bugs_eat_flowers_l330_33043

/-- The number of flowers eaten by a group of bugs -/
def flowers_eaten (num_bugs : ℕ) (flowers_per_bug : ℕ) : ℕ :=
  num_bugs * flowers_per_bug

/-- Theorem: Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6 -/
theorem bugs_eat_flowers :
  flowers_eaten 3 2 = 6 := by
  sorry

end bugs_eat_flowers_l330_33043


namespace cyclic_triples_count_l330_33056

/-- Represents a round-robin tournament. -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins per team
  losses : ℕ  -- number of losses per team

/-- Calculates the number of cyclic triples in a tournament. -/
def cyclic_triples (t : Tournament) : ℕ :=
  if t.n * (t.n - 1) = 2 * (t.wins + t.losses) ∧ t.wins = 12 ∧ t.losses = 8
  then 665
  else 0

theorem cyclic_triples_count (t : Tournament) :
  t.n * (t.n - 1) = 2 * (t.wins + t.losses) →
  t.wins = 12 →
  t.losses = 8 →
  cyclic_triples t = 665 := by
  sorry

end cyclic_triples_count_l330_33056


namespace penelope_candy_count_l330_33050

/-- Given a ratio of M&M candies to Starbursts candies and a number of M&M candies,
    calculate the number of Starbursts candies. -/
def calculate_starbursts (mm_ratio : ℕ) (starbursts_ratio : ℕ) (mm_count : ℕ) : ℕ :=
  (mm_count / mm_ratio) * starbursts_ratio

/-- Theorem stating that given 5 M&M candies for every 3 Starbursts candies,
    and 25 M&M candies, there are 15 Starbursts candies. -/
theorem penelope_candy_count :
  calculate_starbursts 5 3 25 = 15 := by
  sorry

end penelope_candy_count_l330_33050


namespace no_genetic_recombination_in_dna_replication_l330_33071

-- Define the basic types
def Cell : Type := String
def Process : Type := String

-- Define the specific cell and processes
def spermatogonialCell : Cell := "spermatogonial cell"
def geneticRecombination : Process := "genetic recombination"
def dnaUnwinding : Process := "DNA unwinding"
def geneMutation : Process := "gene mutation"
def proteinSynthesis : Process := "protein synthesis"

-- Define a function to represent whether a process occurs during DNA replication
def occursInDnaReplication (c : Cell) (p : Process) : Prop := sorry

-- State the theorem
theorem no_genetic_recombination_in_dna_replication :
  occursInDnaReplication spermatogonialCell dnaUnwinding ∧
  occursInDnaReplication spermatogonialCell geneMutation ∧
  occursInDnaReplication spermatogonialCell proteinSynthesis →
  ¬ occursInDnaReplication spermatogonialCell geneticRecombination :=
by sorry

end no_genetic_recombination_in_dna_replication_l330_33071


namespace circle_and_line_properties_l330_33051

-- Define the circle from part 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line x + y = 1
def line1 (x y : ℝ) : Prop := x + y = 1

-- Define the line y = -2x
def line2 (x y : ℝ) : Prop := y = -2 * x

-- Define the circle from part 2
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line from part 2
def line3 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem circle_and_line_properties :
  -- Part 1
  (circle1 2 (-1)) ∧ 
  (∃ (x y : ℝ), circle1 x y ∧ line1 x y) ∧
  (∃ (x y : ℝ), circle1 x y ∧ line2 x y) ∧
  -- Part 2
  (¬ circle2 2 (-2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle2 x₁ y₁ ∧ circle2 x₂ y₂ ∧ 
    ((x₁ - 2) * (y₁ + 2) = 4) ∧ 
    ((x₂ - 2) * (y₂ + 2) = 4) ∧
    line3 x₁ y₁ ∧ line3 x₂ y₂) := by
  sorry

#check circle_and_line_properties

end circle_and_line_properties_l330_33051


namespace triangle_type_l330_33053

theorem triangle_type (A B C : ℝ) (BC AC : ℝ) (h : BC * Real.cos A = AC * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end triangle_type_l330_33053


namespace simplify_expression_l330_33017

theorem simplify_expression (x : ℝ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
sorry

end simplify_expression_l330_33017


namespace nested_subtraction_simplification_l330_33012

theorem nested_subtraction_simplification (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y := by
  sorry

end nested_subtraction_simplification_l330_33012


namespace flood_damage_conversion_l330_33089

/-- Calculates the equivalent amount in USD given an amount in AUD and the exchange rate -/
def convert_aud_to_usd (amount_aud : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount_aud * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion :
  let damage_aud : ℝ := 45000000
  let exchange_rate : ℝ := 0.75
  convert_aud_to_usd damage_aud exchange_rate = 33750000 := by
  sorry

#check flood_damage_conversion

end flood_damage_conversion_l330_33089


namespace original_integer_is_21_l330_33054

theorem original_integer_is_21 (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (a + b + d) / 3 + c = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (b + c + d) / 3 + a = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 := by
  sorry

end original_integer_is_21_l330_33054


namespace rectangle_area_rectangle_area_proof_l330_33032

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breath := (3 / 5) * circle_radius
  rectangle_length * rectangle_breath

theorem rectangle_area_proof :
  rectangle_area 2025 10 = 270 := by
  sorry

end rectangle_area_rectangle_area_proof_l330_33032


namespace square_area_proof_l330_33078

theorem square_area_proof (x : ℝ) 
  (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
  sorry

end square_area_proof_l330_33078


namespace length_MN_l330_33055

-- Define the points on the line
variable (A B C D M N : ℝ)

-- Define the conditions
axiom order : A < B ∧ B < C ∧ C < D
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (B + D) / 2
axiom length_AD : D - A = 68
axiom length_BC : C - B = 20

-- Theorem to prove
theorem length_MN : N - M = 24 := by sorry

end length_MN_l330_33055


namespace triangle_side_roots_l330_33074

theorem triangle_side_roots (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b) →
  3/4 < m ∧ m ≤ 1 := by
sorry

end triangle_side_roots_l330_33074


namespace average_age_decrease_l330_33096

/-- Proves that replacing a 46-year-old person with a 16-year-old person in a group of 10 decreases the average age by 3 years -/
theorem average_age_decrease (initial_avg : ℝ) : 
  let total_age := 10 * initial_avg
  let new_total_age := total_age - 46 + 16
  let new_avg := new_total_age / 10
  initial_avg - new_avg = 3 := by sorry

end average_age_decrease_l330_33096


namespace modular_congruence_unique_solution_l330_33049

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15893 [ZMOD 16] := by
  sorry

end modular_congruence_unique_solution_l330_33049


namespace bobby_has_two_pizzas_l330_33035

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 6

-- Define Mrs. Kaplan's number of slices
def kaplan_slices : ℕ := 3

-- Define the ratio of Mrs. Kaplan's slices to Bobby's slices
def kaplan_to_bobby_ratio : ℚ := 1 / 4

-- Define Bobby's number of pizzas
def bobby_pizzas : ℕ := 2

-- Theorem to prove
theorem bobby_has_two_pizzas :
  kaplan_slices = kaplan_to_bobby_ratio * (bobby_pizzas * slices_per_pizza) :=
by sorry

end bobby_has_two_pizzas_l330_33035


namespace apps_added_l330_33008

theorem apps_added (initial_apps : ℕ) (deleted_apps : ℕ) (final_apps : ℕ) :
  initial_apps = 10 →
  deleted_apps = 17 →
  final_apps = 4 →
  ∃ (added_apps : ℕ), (initial_apps + added_apps - deleted_apps = final_apps) ∧ (added_apps = 11) :=
by sorry

end apps_added_l330_33008


namespace part_one_part_two_l330_33041

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := |x - 3| < 1

/-- Part 1 of the theorem -/
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

/-- Part 2 of the theorem -/
theorem part_two :
  ∀ a : ℝ, a > 0 → 
  ((∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a)) →
  (4/3 ≤ a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l330_33041


namespace smallest_positive_coterminal_angle_l330_33072

/-- 
Given an angle of -660°, prove that the smallest positive angle 
with the same terminal side is 60°.
-/
theorem smallest_positive_coterminal_angle : 
  ∃ (k : ℤ), -660 + k * 360 = 60 ∧ 
  ∀ (m : ℤ), -660 + m * 360 > 0 → -660 + m * 360 ≥ 60 :=
by sorry

end smallest_positive_coterminal_angle_l330_33072


namespace tim_bodyguard_cost_l330_33009

/-- Calculates the total weekly cost for hiring bodyguards -/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Proves that the total weekly cost for Tim's bodyguards is $2240 -/
theorem tim_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end tim_bodyguard_cost_l330_33009


namespace employee_salary_l330_33042

theorem employee_salary (total_salary : ℝ) (m_percentage : ℝ) (n_salary : ℝ) : 
  total_salary = 616 →
  m_percentage = 1.20 →
  n_salary + m_percentage * n_salary = total_salary →
  n_salary = 280 := by
sorry

end employee_salary_l330_33042


namespace min_value_x_plus_inverse_y_l330_33001

theorem min_value_x_plus_inverse_y (x y : ℝ) (h1 : x ≥ 3) (h2 : x - y = 1) :
  ∃ m : ℝ, m = 7/2 ∧ ∀ z : ℝ, z ≥ 3 → ∀ w : ℝ, z - w = 1 → z + 1/w ≥ m :=
by sorry

end min_value_x_plus_inverse_y_l330_33001


namespace fair_haired_employees_percentage_l330_33026

theorem fair_haired_employees_percentage 
  (total_employees : ℕ) 
  (women_fair_hair_percentage : ℚ) 
  (fair_haired_women_percentage : ℚ) 
  (h1 : women_fair_hair_percentage = 10 / 100) 
  (h2 : fair_haired_women_percentage = 40 / 100) :
  (women_fair_hair_percentage * total_employees) / 
  (fair_haired_women_percentage * total_employees) = 25 / 100 := by
sorry

end fair_haired_employees_percentage_l330_33026


namespace arithmetic_computation_l330_33046

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end arithmetic_computation_l330_33046


namespace probability_rain_three_days_l330_33066

theorem probability_rain_three_days
  (prob_friday : ℝ)
  (prob_saturday : ℝ)
  (prob_sunday : ℝ)
  (prob_sunday_given_saturday : ℝ)
  (h1 : prob_friday = 0.3)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.4)
  (h4 : prob_sunday_given_saturday = 0.7)
  : prob_friday * prob_saturday * prob_sunday_given_saturday = 0.105 := by
  sorry

end probability_rain_three_days_l330_33066


namespace total_mail_delivered_l330_33084

/-- Represents the types of mail --/
inductive MailType
  | JunkMail
  | Magazine
  | Newspaper
  | Bill
  | Postcard

/-- Represents the mail distribution for a single house --/
structure HouseMailDistribution where
  junkMail : Nat
  magazines : Nat
  newspapers : Nat
  bills : Nat
  postcards : Nat

/-- Calculates the total pieces of mail for a single house --/
def totalMailForHouse (dist : HouseMailDistribution) : Nat :=
  dist.junkMail + dist.magazines + dist.newspapers + dist.bills + dist.postcards

/-- The mail distribution for the first house --/
def house1 : HouseMailDistribution :=
  { junkMail := 6, magazines := 5, newspapers := 3, bills := 4, postcards := 2 }

/-- The mail distribution for the second house --/
def house2 : HouseMailDistribution :=
  { junkMail := 4, magazines := 7, newspapers := 2, bills := 5, postcards := 3 }

/-- The mail distribution for the third house --/
def house3 : HouseMailDistribution :=
  { junkMail := 8, magazines := 3, newspapers := 4, bills := 6, postcards := 1 }

/-- Theorem stating that the total pieces of mail delivered to all three houses is 63 --/
theorem total_mail_delivered :
  totalMailForHouse house1 + totalMailForHouse house2 + totalMailForHouse house3 = 63 := by
  sorry

end total_mail_delivered_l330_33084


namespace complex_fourth_power_l330_33016

theorem complex_fourth_power (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end complex_fourth_power_l330_33016


namespace total_prime_factors_l330_33062

def expression (a b c : ℕ) := (4^a) * (7^b) * (11^c)

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem total_prime_factors (a b c : ℕ) :
  a = 11 → b = 7 → c = 2 → is_prime 7 → is_prime 11 →
  (∃ n : ℕ, expression a b c = 2^(2*a) * 7^b * 11^c ∧ 
   n = (2*a) + b + c ∧ n = 31) :=
sorry

end total_prime_factors_l330_33062


namespace min_value_a_plus_2b_l330_33061

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_a_plus_2b_l330_33061


namespace regular_dodecahedron_edges_l330_33011

/-- A regular dodecahedron is a polyhedron with 12 regular pentagonal faces -/
structure RegularDodecahedron where
  faces : Nat
  edges_per_face : Nat
  shared_edges : Nat

/-- Calculate the number of edges in a regular dodecahedron -/
def count_edges (d : RegularDodecahedron) : Nat :=
  (d.faces * d.edges_per_face) / d.shared_edges

/-- Theorem: A regular dodecahedron has 30 edges -/
theorem regular_dodecahedron_edges :
  ∀ d : RegularDodecahedron,
    d.faces = 12 →
    d.edges_per_face = 5 →
    d.shared_edges = 2 →
    count_edges d = 30 := by
  sorry

#check regular_dodecahedron_edges

end regular_dodecahedron_edges_l330_33011


namespace blue_contour_area_relation_l330_33039

/-- Represents the area of a blue contour on a sphere. -/
def blueContourArea (sphereRadius : ℝ) (contourArea : ℝ) : Prop :=
  contourArea ≥ 0 ∧ contourArea ≤ 4 * Real.pi * sphereRadius^2

/-- Theorem stating the relationship between blue contour areas on two concentric spheres. -/
theorem blue_contour_area_relation
  (r₁ : ℝ) (r₂ : ℝ) (a₁ : ℝ) (a₂ : ℝ)
  (h_r₁ : r₁ = 4)
  (h_r₂ : r₂ = 6)
  (h_a₁ : a₁ = 27)
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_contour₁ : blueContourArea r₁ a₁)
  (h_contour₂ : blueContourArea r₂ a₂)
  (h_proportion : a₁ / a₂ = (r₁ / r₂)^2) :
  a₂ = 60.75 :=
sorry

end blue_contour_area_relation_l330_33039


namespace point_in_second_quadrant_l330_33038

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-2 : ℝ) (3 : ℝ) :=
by sorry

end point_in_second_quadrant_l330_33038


namespace one_sixths_in_eleven_thirds_l330_33083

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by
  sorry

end one_sixths_in_eleven_thirds_l330_33083


namespace toothpicks_150th_stage_l330_33030

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  6 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 150th stage is 602 -/
theorem toothpicks_150th_stage : toothpicks 150 = 602 := by
  sorry

end toothpicks_150th_stage_l330_33030


namespace band_percentage_of_ticket_price_l330_33006

/-- Proves that the band receives 70% of the ticket price, given the concert conditions -/
theorem band_percentage_of_ticket_price : 
  ∀ (attendance : ℕ) (ticket_price : ℕ) (band_members : ℕ) (member_earnings : ℕ),
    attendance = 500 →
    ticket_price = 30 →
    band_members = 4 →
    member_earnings = 2625 →
    (band_members * member_earnings : ℚ) / (attendance * ticket_price) = 70 / 100 := by
  sorry

end band_percentage_of_ticket_price_l330_33006


namespace min_value_theorem_l330_33034

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 27) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → 3 * a + 2 * b + c ≥ 18 := by
  sorry

end min_value_theorem_l330_33034


namespace chessboard_tiling_impossible_l330_33040

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard after removal of two squares -/
def ModifiedChessboard : Type := Fin 62 → Square

/-- A function to check if a tiling with dominoes is valid -/
def IsValidTiling (board : ModifiedChessboard) (tiling : List (Fin 62 × Fin 62)) : Prop :=
  ∀ (pair : Fin 62 × Fin 62), pair ∈ tiling →
    (board pair.1 ≠ board pair.2) ∧ 
    (∀ (i : Fin 62), i ∉ [pair.1, pair.2] → 
      ∀ (other_pair : Fin 62 × Fin 62), other_pair ∈ tiling → i ∉ [other_pair.1, other_pair.2])

theorem chessboard_tiling_impossible :
  ∀ (board : ModifiedChessboard),
    (∃ (white_count black_count : Nat), 
      (white_count + black_count = 62) ∧
      (white_count = 30) ∧ (black_count = 32) ∧
      (∀ (i : Fin 62), (board i = Square.White ↔ i.val < white_count))) →
    ¬∃ (tiling : List (Fin 62 × Fin 62)), IsValidTiling board tiling ∧ tiling.length = 31 :=
by sorry

end chessboard_tiling_impossible_l330_33040


namespace two_digit_number_representation_l330_33048

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (num : TwoDigitNumber) : ℕ :=
  10 * num.tens + num.units

theorem two_digit_number_representation (n m : ℕ) (h : n ≥ 1 ∧ n ≤ 9 ∧ m ≤ 9) :
  let num : TwoDigitNumber := ⟨n, m, h⟩
  num.value = 10 * n + m := by
  sorry

end two_digit_number_representation_l330_33048


namespace parabola_directrix_l330_33076

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- Theorem: For a parabola with equation y = -1/8 x^2, its directrix has the equation y = 2 -/
theorem parabola_directrix :
  let p : Parabola := { a := -1/8, b := 0, c := 0 }
  directrix p = 2 := by sorry

end parabola_directrix_l330_33076


namespace expression_value_l330_33052

theorem expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 7) 
  (eq2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
sorry

end expression_value_l330_33052


namespace value_of_expression_l330_33068

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - z = 0)
  (eq2 : x + 4 * y - 15 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y - y*z) / (y^2 + z^2) = 2.4 := by
  sorry

end value_of_expression_l330_33068


namespace replaced_person_weight_l330_33085

/-- The weight of the replaced person in a group of 6 people -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person -/
theorem replaced_person_weight :
  weight_of_replaced_person 6 68 3.5 = 47 := by
  sorry

end replaced_person_weight_l330_33085


namespace max_value_of_expression_l330_33010

theorem max_value_of_expression (x : ℝ) :
  ∃ (max_x : ℝ), ∀ y, 1 - (y + 5)^2 ≤ 1 - (max_x + 5)^2 ∧ max_x = -5 :=
by sorry

end max_value_of_expression_l330_33010


namespace algebraic_expression_value_l330_33023

theorem algebraic_expression_value (m n : ℝ) (h : m ≠ n) 
  (h_equal : m^2 - 2*m + 3 = n^2 - 2*n + 3) : 
  let x := m + n
  (x^2 - 2*x + 3) = 3 := by
  sorry

end algebraic_expression_value_l330_33023


namespace yanna_kept_apples_l330_33021

def apples_kept (total bought : ℕ) (given_to_zenny given_to_andrea : ℕ) : ℕ :=
  bought - given_to_zenny - given_to_andrea

theorem yanna_kept_apples :
  apples_kept 60 18 6 = 36 := by
  sorry

end yanna_kept_apples_l330_33021


namespace probability_sum_twenty_l330_33000

/-- A dodecahedral die with faces labeled 1 through 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12 

/-- The sample space of rolling two dodecahedral dice -/
def TwoDiceRolls : Finset (ℕ × ℕ) :=
  DodecahedralDie.product DodecahedralDie

/-- The event of rolling a sum of 20 with two dodecahedral dice -/
def SumTwenty : Finset (ℕ × ℕ) :=
  TwoDiceRolls.filter (fun p => p.1 + p.2 = 20)

/-- The probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  event.card / sampleSpace.card

theorem probability_sum_twenty :
  probability SumTwenty TwoDiceRolls = 5 / 144 := by
  sorry

end probability_sum_twenty_l330_33000


namespace cooks_selection_theorem_l330_33059

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem cooks_selection_theorem (total_people : ℕ) (cooks_needed : ℕ) (invalid_combinations : ℕ) :
  total_people = 10 →
  cooks_needed = 3 →
  invalid_combinations = choose 8 1 →
  choose total_people cooks_needed - invalid_combinations = 112 := by
sorry

end cooks_selection_theorem_l330_33059


namespace shaded_to_unshaded_ratio_is_five_thirds_l330_33004

/-- Represents a square subdivided into smaller squares --/
structure SubdividedSquare where
  -- The side length of the largest square
  side_length : ℝ
  -- The number of subdivisions (levels of recursion)
  subdivisions : ℕ

/-- Calculates the ratio of shaded area to unshaded area in a subdivided square --/
def shaded_to_unshaded_ratio (square : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to unshaded area is 5/3 --/
theorem shaded_to_unshaded_ratio_is_five_thirds (square : SubdividedSquare) :
  shaded_to_unshaded_ratio square = 5 / 3 := by
  sorry


end shaded_to_unshaded_ratio_is_five_thirds_l330_33004


namespace pizza_slices_theorem_l330_33087

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Represents the ratio of different pizza sizes ordered -/
structure PizzaRatio where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) : ℕ :=
  let ratioSum := ratio.small + ratio.medium + ratio.large + ratio.extraLarge
  let pizzasPerRatio := totalPizzas / ratioSum
  (slices.small * ratio.small * pizzasPerRatio) +
  (slices.medium * ratio.medium * pizzasPerRatio) +
  (slices.large * ratio.large * pizzasPerRatio) +
  (slices.extraLarge * ratio.extraLarge * pizzasPerRatio)

theorem pizza_slices_theorem (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) :
  slices.small = 6 →
  slices.medium = 8 →
  slices.large = 12 →
  slices.extraLarge = 16 →
  ratio.small = 3 →
  ratio.medium = 2 →
  ratio.large = 4 →
  ratio.extraLarge = 1 →
  totalPizzas = 20 →
  totalSlices slices ratio totalPizzas = 196 := by
  sorry

#eval totalSlices ⟨6, 8, 12, 16⟩ ⟨3, 2, 4, 1⟩ 20

end pizza_slices_theorem_l330_33087


namespace ratio_of_sum_and_difference_l330_33019

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_sum_and_difference_l330_33019


namespace ticket_sales_proof_ticket_sales_result_l330_33082

theorem ticket_sales_proof (reduced_first_week : ℕ) (total_tickets : ℕ) : ℕ :=
  let reduced_price_tickets := reduced_first_week
  let full_price_tickets := 5 * reduced_price_tickets
  let total := reduced_price_tickets + full_price_tickets
  
  have h1 : reduced_first_week = 5400 := by sorry
  have h2 : total_tickets = 25200 := by sorry
  have h3 : total = total_tickets := by sorry
  
  full_price_tickets

theorem ticket_sales_result : ticket_sales_proof 5400 25200 = 21000 := by sorry

end ticket_sales_proof_ticket_sales_result_l330_33082


namespace chocolate_boxes_total_l330_33013

/-- The total number of chocolate pieces in multiple boxes -/
def total_pieces (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: The total number of chocolate pieces in 6 boxes with 500 pieces each is 3000 -/
theorem chocolate_boxes_total :
  total_pieces 6 500 = 3000 := by
  sorry

end chocolate_boxes_total_l330_33013


namespace workers_savings_l330_33028

/-- A worker's savings problem -/
theorem workers_savings (monthly_pay : ℝ) (saving_fraction : ℝ) : 
  monthly_pay > 0 →
  saving_fraction > 0 →
  saving_fraction < 1 →
  (12 * saving_fraction * monthly_pay) = (4 * (1 - saving_fraction) * monthly_pay) →
  saving_fraction = 1 / 4 := by
  sorry


end workers_savings_l330_33028


namespace field_trip_cost_l330_33060

/-- Calculate the total cost of renting buses and paying tolls for a field trip -/
theorem field_trip_cost (total_people : ℕ) (seats_per_bus : ℕ) 
  (rental_cost_per_bus : ℕ) (toll_per_bus : ℕ) : 
  total_people = 260 → 
  seats_per_bus = 41 → 
  rental_cost_per_bus = 300000 → 
  toll_per_bus = 7500 → 
  (((total_people + seats_per_bus - 1) / seats_per_bus) * 
   (rental_cost_per_bus + toll_per_bus)) = 2152500 := by
  sorry

end field_trip_cost_l330_33060


namespace square_function_difference_l330_33002

/-- For f(x) = x^2, prove that f(x) - f(x-1) = 2x - 1 for all real x -/
theorem square_function_difference (x : ℝ) : x^2 - (x-1)^2 = 2*x - 1 := by
  sorry

end square_function_difference_l330_33002


namespace jasons_hardcover_books_l330_33045

/-- Proves that Jason has 70 hardcover books given the problem conditions --/
theorem jasons_hardcover_books :
  let bookcase_limit : ℕ := 80
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let over_limit : ℕ := 33
  
  let total_weight : ℕ := bookcase_limit + over_limit
  let textbook_total_weight : ℕ := textbook_count * textbook_weight
  let knickknack_total_weight : ℕ := knickknack_count * knickknack_weight
  let hardcover_total_weight : ℕ := total_weight - textbook_total_weight - knickknack_total_weight
  
  (hardcover_total_weight : ℚ) / hardcover_weight = 70 := by sorry

end jasons_hardcover_books_l330_33045


namespace repeating_decimal_equiv_fraction_lowest_terms_l330_33098

/-- The repeating decimal 0.4̄37 as a real number -/
def repeating_decimal : ℚ := 433 / 990

theorem repeating_decimal_equiv : repeating_decimal = 0.4 + (37 / 990) := by sorry

theorem fraction_lowest_terms : ∀ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 → (433 * a = 990 * b) → (a = 990 ∧ b = 433) := by sorry

end repeating_decimal_equiv_fraction_lowest_terms_l330_33098
