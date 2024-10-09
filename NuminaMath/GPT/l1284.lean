import Mathlib

namespace divisor_of_subtracted_number_l1284_128417

theorem divisor_of_subtracted_number (n : ℕ) (m : ℕ) (h : n = 5264 - 11) : Nat.gcd n 5264 = 5253 :=
by
  sorry

end divisor_of_subtracted_number_l1284_128417


namespace quadratic_inequality_solution_l1284_128467

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2 * x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_l1284_128467


namespace min_g_l1284_128466

noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_g : ∃ x : ℝ, g x = 2 :=
by
  use 0
  sorry

end min_g_l1284_128466


namespace gcd_m_n_l1284_128476

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end gcd_m_n_l1284_128476


namespace problem1_problem2_l1284_128412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x * f a x - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

theorem problem1 (x : ℝ) (h₁ : x ≥ 5) : g 1 x < 1 :=
sorry

theorem problem2 (a : ℝ) (h₂ : a > Real.exp 2 / 4) : 
∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ h a x1 = 0 ∧ h a x2 = 0 :=
sorry

end problem1_problem2_l1284_128412


namespace A_days_to_complete_job_l1284_128408

noncomputable def time_for_A (x : ℝ) (work_left : ℝ) : ℝ :=
  let work_rate_A := 1 / x
  let work_rate_B := 1 / 30
  let combined_work_rate := work_rate_A + work_rate_B
  let completed_work := 4 * combined_work_rate
  let fraction_work_left := 1 - completed_work
  fraction_work_left

theorem A_days_to_complete_job : ∃ x : ℝ, time_for_A x 0.6 = 0.6 ∧ x = 15 :=
by {
  use 15,
  sorry
}

end A_days_to_complete_job_l1284_128408


namespace evaluate_expression_l1284_128419

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 :=
by
  sorry

end evaluate_expression_l1284_128419


namespace hyperbola_eccentricity_l1284_128471

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0)
  (h_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / 16 = 1 ↔ true)
  (eccentricity : a^2 + 16 / a^2 = (5 / 3)^2) : a = 3 :=
by
  sorry

end hyperbola_eccentricity_l1284_128471


namespace factorization_count_is_correct_l1284_128465

noncomputable def count_factorizations (n : Nat) (k : Nat) : Nat :=
  (Nat.choose (n + k - 1) (k - 1))

noncomputable def factor_count : Nat :=
  let alpha_count := count_factorizations 6 3
  let beta_count := count_factorizations 6 3
  let total_count := alpha_count * beta_count
  let unordered_factorizations := total_count - 15 * 3 - 1
  1 + 15 + unordered_factorizations / 6

theorem factorization_count_is_correct :
  factor_count = 139 := by
  sorry

end factorization_count_is_correct_l1284_128465


namespace sum_max_min_ratio_l1284_128495

theorem sum_max_min_ratio (x y : ℝ) 
  (h_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0) 
  : (∃ m_max m_min : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0 → y = m_max * x ∨ y = m_min * x) ∧ (m_max + m_min = 37 / 22)) :=
sorry

end sum_max_min_ratio_l1284_128495


namespace polynomial_roots_l1284_128413

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l1284_128413


namespace rabbit_travel_time_l1284_128453

theorem rabbit_travel_time (distance : ℕ) (speed : ℕ) (time_in_minutes : ℕ) 
  (h_distance : distance = 3) 
  (h_speed : speed = 6) 
  (h_time_eqn : time_in_minutes = (distance * 60) / speed) : 
  time_in_minutes = 30 := 
by 
  sorry

end rabbit_travel_time_l1284_128453


namespace fraction_irreducible_gcd_2_power_l1284_128451

-- Proof problem (a)
theorem fraction_irreducible (n : ℕ) : gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

-- Proof problem (b)
theorem gcd_2_power (n m : ℕ) : gcd (2^100 - 1) (2^120 - 1) = 2^20 - 1 :=
sorry

end fraction_irreducible_gcd_2_power_l1284_128451


namespace no_solution_absval_equation_l1284_128473

theorem no_solution_absval_equation (x : ℝ) : ¬ (|2*x - 5| = 3*x + 1) :=
by
  sorry

end no_solution_absval_equation_l1284_128473


namespace sum_expr_le_e4_l1284_128486

theorem sum_expr_le_e4
  (α β γ δ ε : ℝ) :
  (1 - α) * Real.exp α +
  (1 - β) * Real.exp (α + β) +
  (1 - γ) * Real.exp (α + β + γ) +
  (1 - δ) * Real.exp (α + β + γ + δ) +
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 :=
sorry

end sum_expr_le_e4_l1284_128486


namespace find_OQ_l1284_128440
-- Import the required math libarary

-- Define points on a line with the given distances
def O := 0
def A (a : ℝ) := 2 * a
def B (b : ℝ) := 4 * b
def C (c : ℝ) := 5 * c
def D (d : ℝ) := 7 * d

-- Given P between B and C such that ratio condition holds
def P (a b c d x : ℝ) := 
  B b ≤ x ∧ x ≤ C c ∧ 
  (A a - x) * (x - C c) = (B b - x) * (x - D d)

-- Calculate Q based on given ratio condition
def Q (b c d y : ℝ) := 
  C c ≤ y ∧ y ≤ D d ∧ 
  (C c - y) * (y - D d) = (B b - C c) * (C c - D d)

-- Main Proof Statement to prove OQ
theorem find_OQ (a b c d y : ℝ) 
  (hP : ∃ x, P a b c d x)
  (hQ : ∃ y, Q b c d y) :
  y = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) := by
  sorry

end find_OQ_l1284_128440


namespace quarters_needed_l1284_128491

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l1284_128491


namespace molecular_weight_is_122_l1284_128435

noncomputable def molecular_weight_of_compound := 
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  7 * atomic_weight_C + 6 * atomic_weight_H + 2 * atomic_weight_O

theorem molecular_weight_is_122 :
  molecular_weight_of_compound = 122 := by
  sorry

end molecular_weight_is_122_l1284_128435


namespace solve_equation_l1284_128464

open Real

noncomputable def f (x : ℝ) := 2017 * x ^ 2017 - 2017 + x
noncomputable def g (x : ℝ) := (2018 - 2017 * x) ^ (1 / 2017 : ℝ)

theorem solve_equation :
  ∀ x : ℝ, 2017 * x ^ 2017 - 2017 + x = (2018 - 2017 * x) ^ (1 / 2017 : ℝ) → x = 1 :=
by
  sorry

end solve_equation_l1284_128464


namespace wallpaper_expenditure_l1284_128423

structure Room :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

def cost_per_square_meter : ℕ := 75

def total_expenditure (room : Room) : ℕ :=
  let perimeter := 2 * (room.length + room.width)
  let area_of_walls := perimeter * room.height
  let area_of_ceiling := room.length * room.width
  let total_area := area_of_walls + area_of_ceiling
  total_area * cost_per_square_meter

theorem wallpaper_expenditure (room : Room) : 
  room = Room.mk 30 25 10 →
  total_expenditure room = 138750 :=
by 
  intros h
  rw [h]
  sorry

end wallpaper_expenditure_l1284_128423


namespace studentsInBandOrSports_l1284_128422

-- conditions definitions
def totalStudents : ℕ := 320
def studentsInBand : ℕ := 85
def studentsInSports : ℕ := 200
def studentsInBoth : ℕ := 60

-- theorem statement
theorem studentsInBandOrSports : studentsInBand + studentsInSports - studentsInBoth = 225 :=
by
  sorry

end studentsInBandOrSports_l1284_128422


namespace payment_relationship_l1284_128427

noncomputable def payment_amount (x : ℕ) (price_per_book : ℕ) (discount_percent : ℕ) : ℕ :=
  if x > 20 then ((x - 20) * (price_per_book * (100 - discount_percent) / 100) + 20 * price_per_book) else x * price_per_book

theorem payment_relationship (x : ℕ) (h : x > 20) : payment_amount x 25 20 = 20 * x + 100 := 
by
  sorry

end payment_relationship_l1284_128427


namespace isosceles_trapezoid_legs_squared_l1284_128410

theorem isosceles_trapezoid_legs_squared
  (A B C D : Type)
  (AB CD AD BC : ℝ)
  (isosceles_trapezoid : AB = 50 ∧ CD = 14 ∧ AD = BC)
  (circle_tangent : ∃ M : ℝ, M = 25 ∧ ∀ x : ℝ, MD = 7 ↔ AD = x ∧ BC = x) :
  AD^2 = 800 := 
by
  sorry

end isosceles_trapezoid_legs_squared_l1284_128410


namespace drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l1284_128400

-- Definitions for the initial conditions
def initial_white_balls := 2
def initial_black_balls := 3
def initial_red_balls := 5
def total_balls := initial_white_balls + initial_black_balls + initial_red_balls

-- Statement for part 1: Drawing a red ball is a random event
theorem drawing_red_ball_random : (initial_red_balls > 0) := by
  sorry

-- Statement for part 1: Drawing a yellow ball is impossible
theorem drawing_yellow_ball_impossible : (0 = 0) := by
  sorry

-- Statement for part 2: Probability of drawing a black ball
theorem probability_black_ball : (initial_black_balls : ℚ) / total_balls = 3 / 10 := by
  sorry

-- Definitions for the conditions in part 3
def additional_black_balls (x : ℕ) := initial_black_balls + x
def new_total_balls (x : ℕ) := total_balls + x

-- Statement for part 3: Finding the number of additional black balls
theorem number_of_additional_black_balls (x : ℕ)
  (h : (additional_black_balls x : ℚ) / new_total_balls x = 3 / 4) : x = 18 := by
  sorry

end drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l1284_128400


namespace return_percentage_is_6_5_l1284_128496

def investment1 : ℤ := 16250
def investment2 : ℤ := 16250
def profit_percentage1 : ℚ := 0.15
def loss_percentage2 : ℚ := 0.05
def total_investment : ℤ := 25000
def net_income : ℚ := investment1 * profit_percentage1 - investment2 * loss_percentage2
def return_percentage : ℚ := (net_income / total_investment) * 100

theorem return_percentage_is_6_5 : return_percentage = 6.5 := by
  sorry

end return_percentage_is_6_5_l1284_128496


namespace sarah_daily_candy_consumption_l1284_128430

def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def days : ℕ := 9

def total_candy : ℕ := neighbors_candy + sister_candy
def average_daily_consumption : ℕ := total_candy / days

theorem sarah_daily_candy_consumption : average_daily_consumption = 9 := by
  sorry

end sarah_daily_candy_consumption_l1284_128430


namespace domain_of_f_l1284_128459

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | f x ≠ f (-3)} = Iio (-3) ∪ Ioi (-3) :=
by
  sorry

end domain_of_f_l1284_128459


namespace daily_shampoo_usage_l1284_128436

theorem daily_shampoo_usage
  (S : ℝ)
  (h1 : ∀ t : ℝ, t = 14 → 14 * S + 14 * (S / 2) = 21) :
  S = 1 := by
  sorry

end daily_shampoo_usage_l1284_128436


namespace circle_symmetric_line_l1284_128409

theorem circle_symmetric_line (a b : ℝ) 
  (h1 : ∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0)
  (h2 : ∀ x y, (x, y) = (2, -1))
  (h3 : 2 * a + 2 * b - 1 = 0) :
  ab ≤ 1 / 16 := sorry

end circle_symmetric_line_l1284_128409


namespace find_m_l1284_128452

noncomputable def first_series_sum : ℝ := 
  let a1 : ℝ := 18
  let a2 : ℝ := 6
  let r : ℝ := a2 / a1
  a1 / (1 - r)

noncomputable def second_series_sum (m : ℝ) : ℝ := 
  let b1 : ℝ := 18
  let b2 : ℝ := 6 + m
  let s : ℝ := b2 / b1
  b1 / (1 - s)

theorem find_m : 
  (3 : ℝ) * first_series_sum = second_series_sum m → m = 8 := 
by 
  sorry

end find_m_l1284_128452


namespace solve_arithmetic_sequence_l1284_128480

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l1284_128480


namespace part1_a_value_part2_solution_part3_incorrect_solution_l1284_128407

-- Part 1: Given solution {x = 1, y = 1}, prove a = 3
theorem part1_a_value (a : ℤ) (h1 : 1 + 2 * 1 = a) : a = 3 := 
by 
  sorry

-- Part 2: Given a = -2, prove the solution is {x = 0, y = -1}
theorem part2_solution (x y : ℤ) (h1 : x + 2 * y = -2) (h2 : 2 * x - y = 1) : x = 0 ∧ y = -1 := 
by 
  sorry

-- Part 3: Given {x = -2, y = -2}, prove that it is not a solution
theorem part3_incorrect_solution (a : ℤ) (h1 : -2 + 2 * (-2) = a) (h2 : 2 * (-2) - (-2) = 1) : False := 
by 
  sorry

end part1_a_value_part2_solution_part3_incorrect_solution_l1284_128407


namespace original_price_of_movie_ticket_l1284_128431

theorem original_price_of_movie_ticket
    (P : ℝ)
    (new_price : ℝ)
    (h1 : new_price = 80)
    (h2 : new_price = 0.80 * P) :
    P = 100 :=
by
  sorry

end original_price_of_movie_ticket_l1284_128431


namespace percentage_of_b_l1284_128432

variable (a b c p : ℝ)

theorem percentage_of_b (h1 : 0.06 * a = 12) (h2 : p * b = 6) (h3 : c = b / a) : 
  p = 6 / (200 * c) := by
  sorry

end percentage_of_b_l1284_128432


namespace smallest_base_for_62_three_digits_l1284_128450

theorem smallest_base_for_62_three_digits: 
  ∃ b : ℕ, (b^2 ≤ 62 ∧ 62 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 62 ∧ 62 < n^3) → n ≥ b :=
by
  sorry

end smallest_base_for_62_three_digits_l1284_128450


namespace max_product_condition_l1284_128403

theorem max_product_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 12) (h3 : 0 ≤ y) (h4 : y ≤ 12) (h_eq : x * y = (12 - x) ^ 2 * (12 - y) ^ 2) : x * y ≤ 81 :=
sorry

end max_product_condition_l1284_128403


namespace compute_difference_a_b_l1284_128401

-- Define the initial amounts paid by Alex, Bob, and Carol
def alex_paid := 120
def bob_paid := 150
def carol_paid := 210

-- Define the total amount and equal share
def total_costs := alex_paid + bob_paid + carol_paid
def equal_share := total_costs / 3

-- Define the amounts Alex and Carol gave to Bob, satisfying their balances
def a := equal_share - alex_paid
def b := carol_paid - equal_share

-- Lean 4 statement to prove a - b = 30
theorem compute_difference_a_b : a - b = 30 := by
  sorry

end compute_difference_a_b_l1284_128401


namespace age_of_youngest_child_l1284_128479

theorem age_of_youngest_child
  (x : ℕ)
  (sum_of_ages : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) :
  x = 4 :=
sorry

end age_of_youngest_child_l1284_128479


namespace solve_problem_l1284_128443

-- Definitions from the conditions
def is_divisible_by (n k : ℕ) : Prop :=
  ∃ m, k * m = n

def count_divisors (limit k : ℕ) : ℕ :=
  Nat.div limit k

def count_numbers_divisible_by_neither_5_nor_7 (limit : ℕ) : ℕ :=
  let total := limit - 1
  let divisible_by_5 := count_divisors limit 5
  let divisible_by_7 := count_divisors limit 7
  let divisible_by_35 := count_divisors limit 35
  total - (divisible_by_5 + divisible_by_7 - divisible_by_35)

-- The statement to be proved
theorem solve_problem : count_numbers_divisible_by_neither_5_nor_7 1000 = 686 :=
by
  sorry

end solve_problem_l1284_128443


namespace claudia_coins_l1284_128402

variable (x y : ℕ)

theorem claudia_coins :
  (x + y = 15 ∧ ((145 - 5 * x) / 5) + 1 = 23) → y = 9 :=
by
  intro h
  -- The proof steps would go here, but we'll leave it as sorry for now.
  sorry

end claudia_coins_l1284_128402


namespace range_of_a_l1284_128404

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 ≤ a ∧ a < 8) :=
by
  sorry

end range_of_a_l1284_128404


namespace thomas_monthly_earnings_l1284_128416

def weekly_earnings : ℕ := 4550
def weeks_in_month : ℕ := 4
def monthly_earnings : ℕ := weekly_earnings * weeks_in_month

theorem thomas_monthly_earnings : monthly_earnings = 18200 := by
  sorry

end thomas_monthly_earnings_l1284_128416


namespace total_tickets_sold_l1284_128462

theorem total_tickets_sold 
  (A D : ℕ) 
  (cost_adv cost_door : ℝ) 
  (revenue : ℝ)
  (door_tickets_sold total_tickets : ℕ) 
  (h1 : cost_adv = 14.50) 
  (h2 : cost_door = 22.00)
  (h3 : revenue = 16640) 
  (h4 : door_tickets_sold = 672) : 
  (total_tickets = 800) :=
by
  sorry

end total_tickets_sold_l1284_128462


namespace hyperbola_eccentricity_l1284_128448

theorem hyperbola_eccentricity :
  let a := 2
  let b := 2 * Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (e = Real.sqrt 3) :=
by {
  sorry
}

end hyperbola_eccentricity_l1284_128448


namespace fuel_tank_capacity_l1284_128437

def ethanol_content_fuel_A (fuel_A : ℝ) : ℝ := 0.12 * fuel_A
def ethanol_content_fuel_B (fuel_B : ℝ) : ℝ := 0.16 * fuel_B

theorem fuel_tank_capacity (C : ℝ) :
  ethanol_content_fuel_A 122 + ethanol_content_fuel_B (C - 122) = 30 → C = 218 :=
by
  sorry

end fuel_tank_capacity_l1284_128437


namespace one_third_of_recipe_l1284_128489

noncomputable def recipe_flour_required : ℚ := 7 + 3 / 4

theorem one_third_of_recipe : (1 / 3) * recipe_flour_required = (2 : ℚ) + 7 / 12 :=
by
  sorry

end one_third_of_recipe_l1284_128489


namespace V3_is_correct_l1284_128454

-- Definitions of the polynomial and Horner's method applied at x = -4
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def V_3_value : ℤ := 
  let v0 := -4
  let v1 := v0 * 3 + 5
  let v2 := v0 * v1 + 6
  v0 * v2 + 79

theorem V3_is_correct : V_3_value = -57 := 
  by sorry

end V3_is_correct_l1284_128454


namespace second_year_selection_l1284_128477

noncomputable def students_from_first_year : ℕ := 30
noncomputable def students_from_second_year : ℕ := 40
noncomputable def selected_from_first_year : ℕ := 6
noncomputable def selected_from_second_year : ℕ := (selected_from_first_year * students_from_second_year) / students_from_first_year

theorem second_year_selection :
  students_from_second_year = 40 ∧ students_from_first_year = 30 ∧ selected_from_first_year = 6 →
  selected_from_second_year = 8 :=
by
  intros h
  sorry

end second_year_selection_l1284_128477


namespace smallest_number_is_D_l1284_128433

-- Define the given numbers in Lean
def A := 25
def B := 111
def C := 16 + 4 + 2  -- since 10110_{(2)} equals 22 in base 10
def D := 16 + 2 + 1  -- since 10011_{(2)} equals 19 in base 10

-- The Lean statement for the proof problem
theorem smallest_number_is_D : min (min A B) (min C D) = D := by
  sorry

end smallest_number_is_D_l1284_128433


namespace muffin_combinations_l1284_128488

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end muffin_combinations_l1284_128488


namespace scientific_notation_141260_million_l1284_128482

theorem scientific_notation_141260_million :
  (141260 * 10^6 : ℝ) = 1.4126 * 10^11 := 
sorry

end scientific_notation_141260_million_l1284_128482


namespace totalPeaches_l1284_128463

-- Definitions based on the given conditions
def redPeaches : Nat := 13
def greenPeaches : Nat := 3

-- Problem statement
theorem totalPeaches : redPeaches + greenPeaches = 16 := by
  sorry

end totalPeaches_l1284_128463


namespace simplest_form_option_l1284_128470

theorem simplest_form_option (x y : ℚ) :
  (∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (12 * (x - y) / (15 * (x + y)) ≠ 4 * (x - y) / 5 * (x + y))) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 + y^2) / (x + y) = a / b) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / ((x + y)^2) ≠ (x - y) / (x + y)) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / (x + y) ≠ x - y)) := sorry

end simplest_form_option_l1284_128470


namespace centroid_path_is_ellipse_l1284_128434

theorem centroid_path_is_ellipse
  (b r : ℝ)
  (C : ℝ → ℝ × ℝ)
  (H1 : ∃ t θ, C t = (r * Real.cos θ, r * Real.sin θ))
  (G : ℝ → ℝ × ℝ)
  (H2 : ∀ t, G t = (1 / 3 * (b + (C t).fst), 1 / 3 * ((C t).snd))) :
  ∃ a c : ℝ, ∀ t, (G t).fst^2 / a^2 + (G t).snd^2 / c^2 = 1 :=
sorry

end centroid_path_is_ellipse_l1284_128434


namespace same_graphs_at_x_eq_1_l1284_128442

theorem same_graphs_at_x_eq_1 :
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  y2 = 3 ∧ y3 = 3 ∧ y1 ≠ y2 := 
by
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  sorry

end same_graphs_at_x_eq_1_l1284_128442


namespace intersection_of_A_and_B_l1284_128426

-- Definitions of the sets A and B
def A : Set ℝ := { x | x^2 + 2*x - 3 < 0 }
def B : Set ℝ := { x | |x - 1| < 2 }

-- The statement to prove their intersection
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x < 1 } :=
by 
  sorry

end intersection_of_A_and_B_l1284_128426


namespace sum_cis_angles_l1284_128492

noncomputable def complex.cis (θ : ℝ) := Complex.exp (Complex.I * θ)

theorem sum_cis_angles :
  (complex.cis (80 * Real.pi / 180) + complex.cis (88 * Real.pi / 180) + complex.cis (96 * Real.pi / 180) + 
  complex.cis (104 * Real.pi / 180) + complex.cis (112 * Real.pi / 180) + complex.cis (120 * Real.pi / 180) + 
  complex.cis (128 * Real.pi / 180)) = r * complex.cis (104 * Real.pi / 180) := 
sorry

end sum_cis_angles_l1284_128492


namespace value_of_expression_l1284_128457

theorem value_of_expression : (3 + 2) - (2 + 1) = 2 :=
by
  sorry

end value_of_expression_l1284_128457


namespace anthony_more_than_mabel_l1284_128411

noncomputable def transactions := 
  let M := 90  -- Mabel's transactions
  let J := 82  -- Jade's transactions
  let C := J - 16  -- Cal's transactions
  let A := (3 / 2) * C  -- Anthony's transactions
  let P := ((A - M) / M) * 100 -- Percentage more transactions Anthony handled than Mabel
  P

theorem anthony_more_than_mabel : transactions = 10 := by
  sorry

end anthony_more_than_mabel_l1284_128411


namespace problem_I_problem_II_l1284_128441

def setA : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≤ 0}

theorem problem_I (a : ℝ) : (setB a ⊆ setA) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem problem_II (a : ℝ) : (setA ∩ setB a = {1}) ↔ a ≤ 1 := by
  sorry

end problem_I_problem_II_l1284_128441


namespace find_a_l1284_128446

theorem find_a (a : ℝ) (p : ℕ → ℝ) (h : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 → p k = a * (1 / 2) ^ k)
  (prob_sum : a * (1 / 2 + (1 / 2) ^ 2 + (1 / 2) ^ 3) = 1) : a = 8 / 7 :=
sorry

end find_a_l1284_128446


namespace h_h_of_2_l1284_128414

def h (x : ℝ) : ℝ := 4 * x^2 - 8

theorem h_h_of_2 : h (h 2) = 248 := by
  -- Proof goes here
  sorry

end h_h_of_2_l1284_128414


namespace second_trial_amount_691g_l1284_128405

theorem second_trial_amount_691g (low high : ℝ) (h_range : low = 500) (h_high : high = 1000) (h_method : ∃ x, x = 0.618) : 
  high - 0.618 * (high - low) = 691 :=
by
  sorry

end second_trial_amount_691g_l1284_128405


namespace rectangle_total_area_l1284_128447

-- Let s be the side length of the smaller squares
variable (s : ℕ)

-- Define the areas of the squares
def smaller_square_area := s ^ 2
def larger_square_area := (3 * s) ^ 2

-- Define the total_area
def total_area : ℕ := 2 * smaller_square_area s + larger_square_area s

-- Assert the total area of the rectangle ABCD is 11s^2
theorem rectangle_total_area (s : ℕ) : total_area s = 11 * s ^ 2 := 
by 
  -- the proof is skipped
  sorry

end rectangle_total_area_l1284_128447


namespace collinear_condition_perpendicular_condition_l1284_128472

-- Problem 1: Prove collinearity condition for k = -2
theorem collinear_condition (k : ℝ) : 
  (k - 5) * (-12) - (12 - k) * 6 = 0 ↔ k = -2 :=
sorry

-- Problem 2: Prove perpendicular condition for k = 2 or k = 11
theorem perpendicular_condition (k : ℝ) : 
  (20 + (k - 6) * (7 - k)) = 0 ↔ (k = 2 ∨ k = 11) :=
sorry

end collinear_condition_perpendicular_condition_l1284_128472


namespace new_kite_area_l1284_128484

def original_base := 7
def original_height := 6
def scale_factor := 2
def side_length := 2

def new_base := original_base * scale_factor
def new_height := original_height * scale_factor
def half_new_height := new_height / 2

def area_triangle := (1 / 2 : ℚ) * new_base * half_new_height
def total_area := 2 * area_triangle

theorem new_kite_area : total_area = 84 := by
  sorry

end new_kite_area_l1284_128484


namespace remainder_zero_l1284_128428

theorem remainder_zero (x : ℤ) :
  (x^5 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end remainder_zero_l1284_128428


namespace stepa_and_petya_are_wrong_l1284_128429

-- Define the six-digit number where all digits are the same.
def six_digit_same (a : ℕ) : ℕ := a * 111111

-- Define the sum of distinct prime divisors of 1001 and 111.
def prime_divisor_sum : ℕ := 7 + 11 + 13 + 3 + 37

-- Define the sum of prime divisors when a is considered.
def additional_sum (a : ℕ) : ℕ :=
  if (a = 2) || (a = 6) || (a = 8) then 2
  else if (a = 5) then 5
  else 0

-- Summarize the possible correct sums
def correct_sums (a : ℕ) : ℕ := prime_divisor_sum + additional_sum a

-- The proof statement
theorem stepa_and_petya_are_wrong (a : ℕ) :
  correct_sums a ≠ 70 ∧ correct_sums a ≠ 80 := 
by {
  sorry
}

end stepa_and_petya_are_wrong_l1284_128429


namespace part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l1284_128481

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x

theorem part1_min_value_of_f_when_a_is_1 : 
  (∃ x : ℝ, f 1 x = 1 / 2 ∧ (∀ y : ℝ, f 1 y ≥ f 1 x)) :=
sorry

theorem part2_range_of_a_for_f_ge_x :
  (∀ x : ℝ, x > 0 → f a x ≥ x) ↔ a ≥ 2 :=
sorry

end part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l1284_128481


namespace star_example_l1284_128483

section star_operation

variables (x y z : ℕ) 

-- Define the star operation as a binary function
def star (a b : ℕ) : ℕ := a * b

-- Given conditions
axiom star_idempotent : ∀ x : ℕ, star x x = 0
axiom star_associative : ∀ x y z : ℕ, star x (star y z) = (star x y) + z

-- Main theorem to be proved
theorem star_example : star 1993 1935 = 58 :=
sorry

end star_operation

end star_example_l1284_128483


namespace R_and_D_per_increase_l1284_128415

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l1284_128415


namespace find_f3_l1284_128460

variable (f : ℕ → ℕ)

axiom h : ∀ x : ℕ, f (x + 1) = x ^ 2

theorem find_f3 : f 3 = 4 :=
by
  sorry

end find_f3_l1284_128460


namespace chess_club_probability_l1284_128475

theorem chess_club_probability :
  let total_members := 20
  let boys := 12
  let girls := 8
  let total_ways := Nat.choose total_members 4
  let all_boys := Nat.choose boys 4
  let all_girls := Nat.choose girls 4
  total_ways ≠ 0 → 
  (1 - (all_boys + all_girls) / total_ways) = (4280 / 4845) :=
by
  sorry

end chess_club_probability_l1284_128475


namespace sum_of_powers_of_i_l1284_128449

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end sum_of_powers_of_i_l1284_128449


namespace Kyle_papers_delivered_each_week_l1284_128493

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l1284_128493


namespace milk_volume_in_ounces_l1284_128445

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end milk_volume_in_ounces_l1284_128445


namespace probability_of_drawing_red_or_green_l1284_128497

def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def total_marbles : ℕ := red_marbles + green_marbles + yellow_marbles
def favorable_marbles : ℕ := red_marbles + green_marbles
def probability_of_red_or_green : ℚ := favorable_marbles / total_marbles

theorem probability_of_drawing_red_or_green :
  probability_of_red_or_green = 7 / 13 := by
  sorry

end probability_of_drawing_red_or_green_l1284_128497


namespace certain_number_eq_l1284_128421

theorem certain_number_eq :
  ∃ y : ℝ, y + (y * 4) = 48 ∧ y = 9.6 :=
by
  sorry

end certain_number_eq_l1284_128421


namespace sum_first_last_l1284_128439

theorem sum_first_last (A B C D : ℕ) (h1 : (A + B + C) / 3 = 6) (h2 : (B + C + D) / 3 = 5) (h3 : D = 4) : A + D = 11 :=
by
  sorry

end sum_first_last_l1284_128439


namespace find_b_l1284_128444

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 :=
by
  sorry

end find_b_l1284_128444


namespace mass_of_CaSO4_formed_correct_l1284_128490

noncomputable def mass_CaSO4_formed 
(mass_CaO : ℝ) (mass_H2SO4 : ℝ)
(molar_mass_CaO : ℝ) (molar_mass_H2SO4 : ℝ) (molar_mass_CaSO4 : ℝ) : ℝ :=
  let moles_CaO := mass_CaO / molar_mass_CaO
  let moles_H2SO4 := mass_H2SO4 / molar_mass_H2SO4
  let limiting_reactant_moles := min moles_CaO moles_H2SO4
  limiting_reactant_moles * molar_mass_CaSO4

theorem mass_of_CaSO4_formed_correct :
  mass_CaSO4_formed 25 35 56.08 98.09 136.15 = 48.57 :=
by
  rw [mass_CaSO4_formed]
  sorry

end mass_of_CaSO4_formed_correct_l1284_128490


namespace conceived_number_is_seven_l1284_128461

theorem conceived_number_is_seven (x : ℕ) (h1 : x > 0) (h2 : (1 / 4 : ℚ) * (10 * x + 7 - x * x) - x = 0) : x = 7 := by
  sorry

end conceived_number_is_seven_l1284_128461


namespace chose_number_l1284_128456

theorem chose_number (x : ℝ) (h : (x / 12)^2 - 240 = 8) : x = 24 * Real.sqrt 62 :=
sorry

end chose_number_l1284_128456


namespace meat_needed_for_40_hamburgers_l1284_128458

theorem meat_needed_for_40_hamburgers (meat_per_10_hamburgers : ℕ) (hamburgers_needed : ℕ) (meat_per_hamburger : ℚ) (total_meat_needed : ℚ) :
  meat_per_10_hamburgers = 5 ∧ hamburgers_needed = 40 ∧
  meat_per_hamburger = meat_per_10_hamburgers / 10 ∧
  total_meat_needed = meat_per_hamburger * hamburgers_needed → 
  total_meat_needed = 20 := by
  sorry

end meat_needed_for_40_hamburgers_l1284_128458


namespace sum_of_possible_values_l1284_128478

-- Define the triangle's base and height
def triangle_base (x : ℝ) : ℝ := x - 2
def triangle_height (x : ℝ) : ℝ := x - 2

-- Define the parallelogram's base and height
def parallelogram_base (x : ℝ) : ℝ := x - 3
def parallelogram_height (x : ℝ) : ℝ := x + 4

-- Define the areas
def triangle_area (x : ℝ) : ℝ := 0.5 * (triangle_base x) * (triangle_height x)
def parallelogram_area (x : ℝ) : ℝ := (parallelogram_base x) * (parallelogram_height x)

-- Statement to prove
theorem sum_of_possible_values (x : ℝ) (h : parallelogram_area x = 3 * triangle_area x) : x = 8 ∨ x = 3 →
  (x = 8 ∨ x = 3) → 8 + 3 = 11 :=
by sorry

end sum_of_possible_values_l1284_128478


namespace find_k_value_l1284_128469

theorem find_k_value (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : (x₁ + x₂ + x₃ + x₄) = 18)
  (h2 : (x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄) = k)
  (h3 : (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄) = -200)
  (h4 : (x₁ * x₂ * x₃ * x₄) = -1984)
  (h5 : x₁ * x₂ = -32) :
  k = 86 :=
by sorry

end find_k_value_l1284_128469


namespace value_of_m_l1284_128438

theorem value_of_m (m : ℝ) : (∀ x : ℝ, x^2 + m * x + 9 = (x + 3)^2) → m = 6 :=
by
  intro h
  sorry

end value_of_m_l1284_128438


namespace Marley_fruits_total_is_31_l1284_128499

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l1284_128499


namespace golf_ratio_l1284_128487

-- Definitions based on conditions
def first_turn_distance : ℕ := 180
def excess_distance : ℕ := 20
def total_distance_to_hole : ℕ := 250

-- Derived definitions based on conditions
def second_turn_distance : ℕ := (total_distance_to_hole - first_turn_distance) + excess_distance

-- Lean proof problem statement
theorem golf_ratio : (second_turn_distance : ℚ) / first_turn_distance = 1 / 2 :=
by
  -- use sorry to skip the proof
  sorry

end golf_ratio_l1284_128487


namespace money_spent_correct_l1284_128485

-- Define conditions
def spring_income : ℕ := 2
def summer_income : ℕ := 27
def amount_after_supplies : ℕ := 24

-- Define the resulting money spent on supplies
def money_spent_on_supplies : ℕ :=
  (spring_income + summer_income) - amount_after_supplies

theorem money_spent_correct :
  money_spent_on_supplies = 5 := by
  sorry

end money_spent_correct_l1284_128485


namespace sin_600_eq_neg_sqrt_3_over_2_l1284_128474

theorem sin_600_eq_neg_sqrt_3_over_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_eq_neg_sqrt_3_over_2_l1284_128474


namespace inequality_holds_for_minimal_a_l1284_128420

theorem inequality_holds_for_minimal_a :
  ∀ (x : ℝ), (1 ≤ x) → (x ≤ 4) → (1 + x) * Real.log x + x ≤ x * 1.725 :=
by
  intros x h1 h2
  sorry

end inequality_holds_for_minimal_a_l1284_128420


namespace max_sum_e3_f3_g3_h3_i3_l1284_128418

theorem max_sum_e3_f3_g3_h3_i3 (e f g h i : ℝ) (h_cond : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  e^3 + f^3 + g^3 + h^3 + i^3 ≤ 5^(3/4) :=
sorry

end max_sum_e3_f3_g3_h3_i3_l1284_128418


namespace fraction_square_equality_l1284_128494

theorem fraction_square_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
    (h : a / b + c / d = 1) : 
    (a / b)^2 + c / d = (c / d)^2 + a / b :=
by
  sorry

end fraction_square_equality_l1284_128494


namespace basketball_scores_l1284_128406

theorem basketball_scores :
  (∃ P : Finset ℕ, P = { P | ∃ x : ℕ, x ∈ (Finset.range 8) ∧ P = x + 14 } ∧ P.card = 8) :=
by
  sorry

end basketball_scores_l1284_128406


namespace problem_c_l1284_128425

theorem problem_c (x y : ℝ) (h : x - 3 = y - 3): x - y = 0 :=
by
  sorry

end problem_c_l1284_128425


namespace fruit_selling_price_3640_l1284_128498

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l1284_128498


namespace sin_cos_sum_l1284_128468

-- Let theta be an angle in the second quadrant
variables (θ : ℝ)
-- Given the condition tan(θ + π / 4) = 1 / 2
variable (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2)
-- Given θ is in the second quadrant
variable (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi)

-- Prove sin θ + cos θ = - sqrt(10) / 5
theorem sin_cos_sum (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2) (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi) :
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_cos_sum_l1284_128468


namespace friends_who_dont_eat_meat_l1284_128455

-- Definitions based on conditions
def number_of_friends : Nat := 10
def burgers_per_friend : Nat := 3
def buns_per_pack : Nat := 8
def packs_of_buns : Nat := 3
def friends_dont_eat_meat : Nat := 1
def friends_dont_eat_bread : Nat := 1

-- Total number of buns Alex plans to buy
def total_buns : Nat := buns_per_pack * packs_of_buns

-- Calculation of friends needing buns
def friends_needing_buns : Nat := number_of_friends - friends_dont_eat_meat - friends_dont_eat_bread

-- Total buns needed
def buns_needed : Nat := friends_needing_buns * burgers_per_friend

theorem friends_who_dont_eat_meat :
  buns_needed = total_buns -> friends_dont_eat_meat = 1 := by
  sorry

end friends_who_dont_eat_meat_l1284_128455


namespace altitude_length_l1284_128424

theorem altitude_length {s t : ℝ} 
  (A B C : ℝ × ℝ) 
  (hA : A = (-s, s^2))
  (hB : B = (s, s^2))
  (hC : C = (t, t^2))
  (h_parabola_A : A.snd = (A.fst)^2)
  (h_parabola_B : B.snd = (B.fst)^2)
  (h_parabola_C : C.snd = (C.fst)^2)
  (hyp_parallel : A.snd = B.snd)
  (right_triangle : (t + s) * (t - s) + (t^2 - s^2)^2 = 0) :
  (s^2 - (t^2)) = 1 :=
by
  sorry

end altitude_length_l1284_128424
