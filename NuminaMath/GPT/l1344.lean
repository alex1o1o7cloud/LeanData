import Mathlib

namespace NUMINAMATH_GPT_table_filling_impossible_l1344_134431

theorem table_filling_impossible :
  ∀ (table : Fin 5 → Fin 8 → Fin 10),
  (∀ digit : Fin 10, ∃ row_set : Finset (Fin 5), row_set.card = 4 ∧
    (∀ row : Fin 5, row ∈ row_set → ∃ col_set : Finset (Fin 8), col_set.card = 4 ∧
      (∀ col : Fin 8, col ∈ col_set → table row col = digit))) →
  False :=
by
  sorry

end NUMINAMATH_GPT_table_filling_impossible_l1344_134431


namespace NUMINAMATH_GPT_four_spheres_max_intersections_l1344_134467

noncomputable def max_intersection_points (n : Nat) : Nat :=
  if h : n > 0 then n * 2 else 0

theorem four_spheres_max_intersections : max_intersection_points 4 = 8 := by
  sorry

end NUMINAMATH_GPT_four_spheres_max_intersections_l1344_134467


namespace NUMINAMATH_GPT_macy_miles_left_l1344_134499

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end NUMINAMATH_GPT_macy_miles_left_l1344_134499


namespace NUMINAMATH_GPT_total_food_eaten_l1344_134453

theorem total_food_eaten (num_puppies num_dogs : ℕ)
    (dog_food_per_meal dog_meals_per_day puppy_food_per_day : ℕ)
    (dog_food_mult puppy_meal_mult : ℕ)
    (h1 : num_puppies = 6)
    (h2 : num_dogs = 5)
    (h3 : dog_food_per_meal = 6)
    (h4 : dog_meals_per_day = 2)
    (h5 : dog_food_mult = 3)
    (h6 : puppy_meal_mult = 4)
    (h7 : puppy_food_per_day = (dog_food_per_meal / dog_food_mult) * puppy_meal_mult * dog_meals_per_day) :
    (num_dogs * dog_food_per_meal * dog_meals_per_day + num_puppies * puppy_food_per_day) = 108 := by
  -- conclude the theorem
  sorry

end NUMINAMATH_GPT_total_food_eaten_l1344_134453


namespace NUMINAMATH_GPT_trigonometric_expression_l1344_134485

theorem trigonometric_expression (x : ℝ) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := 
sorry

end NUMINAMATH_GPT_trigonometric_expression_l1344_134485


namespace NUMINAMATH_GPT_cost_for_Greg_l1344_134478

theorem cost_for_Greg (N P M : ℝ)
(Bill : 13 * N + 26 * P + 19 * M = 25)
(Paula : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := 
sorry

end NUMINAMATH_GPT_cost_for_Greg_l1344_134478


namespace NUMINAMATH_GPT_initial_pinecones_l1344_134438

theorem initial_pinecones (P : ℝ) :
  (0.20 * P + 2 * 0.20 * P + 0.25 * (0.40 * P) = 0.70 * P - 0.10 * P) ∧ (0.30 * P = 600) → P = 2000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_pinecones_l1344_134438


namespace NUMINAMATH_GPT_tan_C_over_tan_A_max_tan_B_l1344_134419

theorem tan_C_over_tan_A {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let tan_A := Real.tan A
  let tan_C := Real.tan C
  (Real.tan C / Real.tan A) = -3 :=
sorry

theorem max_tan_B {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let B := Real.arctan (Real.tan B)
  ∃ (x : ℝ), x = Real.tan B ∧ ∀ y, y = Real.tan B → y ≤ (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_tan_C_over_tan_A_max_tan_B_l1344_134419


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1344_134483

variable {A B : Prop}

theorem necessary_and_sufficient_condition (h1 : A → B) (h2 : B → A) : A ↔ B := 
by 
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1344_134483


namespace NUMINAMATH_GPT_train_around_probability_train_present_when_alex_arrives_l1344_134498

noncomputable def trainArrivalTime : Set ℝ := Set.Icc 15 45
noncomputable def trainWaitTime (t : ℝ) : Set ℝ := Set.Icc t (t + 15)
noncomputable def alexArrivalTime : Set ℝ := Set.Icc 0 60

theorem train_around (t : ℝ) (h : t ∈ trainArrivalTime) :
  ∀ (x : ℝ), x ∈ alexArrivalTime → x ∈ trainWaitTime t ↔ 15 ≤ t ∧ t ≤ 45 ∧ t ≤ x ∧ x ≤ t + 15 :=
sorry

theorem probability_train_present_when_alex_arrives :
  let total_area := 60 * 60
  let favorable_area := 1 / 2 * (15 + 15) * 15
  (favorable_area / total_area) = 1 / 16 :=
sorry

end NUMINAMATH_GPT_train_around_probability_train_present_when_alex_arrives_l1344_134498


namespace NUMINAMATH_GPT_find_y_l1344_134418

theorem find_y (DEG EFG y : ℝ) 
  (h1 : DEG = 150)
  (h2 : EFG = 40)
  (h3 : DEG = EFG + y) :
  y = 110 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1344_134418


namespace NUMINAMATH_GPT_general_term_of_sequence_l1344_134496

noncomputable def harmonic_mean {n : ℕ} (p : Fin n → ℝ) : ℝ :=
  n / (Finset.univ.sum (fun i => p i))

theorem general_term_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, harmonic_mean (fun i : Fin n => a (i + 1)) = 1 / (2 * n - 1))
    (h₂ : ∀ n : ℕ, (Finset.range n).sum a = 2 * n^2 - n) :
  ∀ n : ℕ, a n = 4 * n - 3 := by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1344_134496


namespace NUMINAMATH_GPT_non_adjective_primes_sum_l1344_134457

-- We will define the necessary components as identified from our problem

def is_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ∃ a : ℕ → ℕ, ∀ n : ℕ,
    a 0 % p = (1 + (1 / a 1) % p) ∧
    a 1 % p = (1 + (1 / (1 + (1 / a 2) % p)) % p) ∧
    a 2 % p = (1 + (1 / (1 + (1 / (1 + (1 / a 3) % p))) % p))

def is_not_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ¬ is_adjective_prime p

def first_three_non_adjective_primes_sum : ℕ :=
  3 + 7 + 23

theorem non_adjective_primes_sum :
  first_three_non_adjective_primes_sum = 33 := 
  sorry

end NUMINAMATH_GPT_non_adjective_primes_sum_l1344_134457


namespace NUMINAMATH_GPT_absolute_value_condition_necessary_non_sufficient_l1344_134416

theorem absolute_value_condition_necessary_non_sufficient (x : ℝ) :
  (abs (x - 1) < 2 → x^2 < x) ∧ ¬ (x^2 < x → abs (x - 1) < 2) := sorry

end NUMINAMATH_GPT_absolute_value_condition_necessary_non_sufficient_l1344_134416


namespace NUMINAMATH_GPT_cups_needed_correct_l1344_134492

-- Define the conditions
def servings : ℝ := 18.0
def cups_per_serving : ℝ := 2.0

-- Define the total cups needed calculation
def total_cups (servings : ℝ) (cups_per_serving : ℝ) : ℝ :=
  servings * cups_per_serving

-- State the proof problem
theorem cups_needed_correct :
  total_cups servings cups_per_serving = 36.0 :=
by
  sorry

end NUMINAMATH_GPT_cups_needed_correct_l1344_134492


namespace NUMINAMATH_GPT_shaded_area_of_intersections_l1344_134480

theorem shaded_area_of_intersections (r : ℝ) (n : ℕ) (intersect_origin : Prop) (radius_5 : r = 5) (four_circles : n = 4) : 
  ∃ (area : ℝ), area = 100 * Real.pi - 200 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_intersections_l1344_134480


namespace NUMINAMATH_GPT_find_n_l1344_134497

theorem find_n (a b c : ℝ) (h : a^2 + b^2 = c^2) (n : ℕ) (hn : n > 2) : 
  (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n)) → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1344_134497


namespace NUMINAMATH_GPT_guards_can_protect_point_l1344_134476

-- Define the conditions of the problem as Lean definitions
def guardVisionRadius : ℝ := 100

-- Define the proof statement
theorem guards_can_protect_point :
  ∃ (num_guards : ℕ), num_guards * 45 = 360 ∧ guardVisionRadius = 100 :=
by
  sorry

end NUMINAMATH_GPT_guards_can_protect_point_l1344_134476


namespace NUMINAMATH_GPT_part_I_part_II_l1344_134446

-- Define the function f
def f (x a : ℝ) := |x - a| + |x - 2|

-- Statement for part (I)
theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ a) : a ≥ 1 := sorry

-- Statement for part (II)
theorem part_II (m n p : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : p > 0) (h4 : m + 2 * n + 3 * p = 1) : 
  (3 / m) + (2 / n) + (1 / p) ≥ 6 + 2 * Real.sqrt 6 + 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_part_I_part_II_l1344_134446


namespace NUMINAMATH_GPT_ratio_movies_allowance_l1344_134435

variable (M A : ℕ)
variable (weeklyAllowance moneyEarned endMoney : ℕ)
variable (H1 : weeklyAllowance = 8)
variable (H2 : moneyEarned = 8)
variable (H3 : endMoney = 12)
variable (H4 : weeklyAllowance + moneyEarned - M = endMoney)
variable (H5 : A = 8)
variable (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1)

theorem ratio_movies_allowance (M A : ℕ) 
  (weeklyAllowance moneyEarned endMoney : ℕ)
  (H1 : weeklyAllowance = 8)
  (H2 : moneyEarned = 8)
  (H3 : endMoney = 12)
  (H4 : weeklyAllowance + moneyEarned - M = endMoney)
  (H5 : A = 8)
  (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1) :
  M / A = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_movies_allowance_l1344_134435


namespace NUMINAMATH_GPT_min_colors_5x5_grid_l1344_134432

def is_valid_coloring (grid : Fin 5 × Fin 5 → ℕ) (k : ℕ) : Prop :=
  ∀ i j : Fin 5, ∀ di dj : Fin 2, ∀ c : ℕ,
    (di ≠ 0 ∨ dj ≠ 0) →
    (grid (i, j) = c ∧ grid (i + di, j + dj) = c ∧ grid (i + 2 * di, j + 2 * dj) = c) → 
    False

theorem min_colors_5x5_grid : 
  ∀ (grid : Fin 5 × Fin 5 → ℕ), (∀ i j, grid (i, j) < 3) → is_valid_coloring grid 3 := 
by
  sorry

end NUMINAMATH_GPT_min_colors_5x5_grid_l1344_134432


namespace NUMINAMATH_GPT_triangle_angles_inequality_l1344_134415

theorem triangle_angles_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := 
sorry

end NUMINAMATH_GPT_triangle_angles_inequality_l1344_134415


namespace NUMINAMATH_GPT_find_n_in_geometric_series_l1344_134445

theorem find_n_in_geometric_series :
  let a1 : ℕ := 15
  let a2 : ℕ := 5
  let r1 := a2 / a1
  let S1 := a1 / (1 - r1: ℝ)
  let S2 := 3 * S1
  let r2 := (5 + n) / a1
  S2 = 15 / (1 - r2) →
  n = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_in_geometric_series_l1344_134445


namespace NUMINAMATH_GPT_calc_expression_l1344_134428

theorem calc_expression : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l1344_134428


namespace NUMINAMATH_GPT_part1_part2_l1344_134422

-- Let m be the cost price this year
-- Let x be the selling price per bottle
-- Assuming:
-- 1. The cost price per bottle increased by 4 yuan this year compared to last year.
-- 2. The quantity of detergent purchased for 1440 yuan this year equals to the quantity purchased for 1200 yuan last year.
-- 3. The selling price per bottle is 36 yuan with 600 bottles sold per week.
-- 4. Weekly sales increase by 100 bottles for every 1 yuan reduction in price.
-- 5. The selling price cannot be lower than the cost price.

-- Definition for improved readability:
def costPriceLastYear (m : ℕ) : ℕ := m - 4

-- Quantity equations
def quantityPurchasedThisYear (m : ℕ) : ℕ := 1440 / m
def quantityPurchasedLastYear (m : ℕ) : ℕ := 1200 / (costPriceLastYear m)

-- Profit Function
def profitFunction (m x : ℝ) : ℝ :=
  (x - m) * (600 + 100 * (36 - x))

-- Maximum Profit and Best Selling Price
def maxProfit : ℝ := 8100
def bestSellingPrice : ℝ := 33

theorem part1 (m : ℕ) (h₁ : 1440 / m = 1200 / costPriceLastYear m) : m = 24 := by
  sorry  -- Will be proved later

theorem part2 (m : ℝ) (x : ℝ)
    (h₀ : m = 24)
    (hx : 600 + 100 * (36 - x) > 0)
    (hx₁ : x ≥ m)
    : profitFunction m x ≤ maxProfit ∧ (∃! (y : ℝ), y = bestSellingPrice ∧ profitFunction m y = maxProfit) := by
  sorry  -- Will be proved later

end NUMINAMATH_GPT_part1_part2_l1344_134422


namespace NUMINAMATH_GPT_ticket_1000_wins_probability_l1344_134486

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ticket_1000_wins_probability_l1344_134486


namespace NUMINAMATH_GPT_original_stations_l1344_134471

theorem original_stations (m n : ℕ) (h : n > 1) (h_equation : n * (2 * m + n - 1) = 58) : m = 14 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_original_stations_l1344_134471


namespace NUMINAMATH_GPT_Lyle_friends_sandwich_juice_l1344_134487

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_Lyle_friends_sandwich_juice_l1344_134487


namespace NUMINAMATH_GPT_arithmetic_sequence_probability_correct_l1344_134425

noncomputable def arithmetic_sequence_probability : ℚ := 
  let total_ways := Nat.choose 5 3
  let arithmetic_sequences := 4
  (arithmetic_sequences : ℚ) / (total_ways : ℚ)

theorem arithmetic_sequence_probability_correct :
  arithmetic_sequence_probability = 0.4 := by
  unfold arithmetic_sequence_probability
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_probability_correct_l1344_134425


namespace NUMINAMATH_GPT_market_value_of_share_l1344_134469

-- Definitions from the conditions
def nominal_value : ℝ := 48
def dividend_rate : ℝ := 0.09
def desired_interest_rate : ℝ := 0.12

-- The proof problem (theorem statement) in Lean 4
theorem market_value_of_share : (nominal_value * dividend_rate / desired_interest_rate * 100) = 36 := 
by
  sorry

end NUMINAMATH_GPT_market_value_of_share_l1344_134469


namespace NUMINAMATH_GPT_f_g_of_4_l1344_134410

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

theorem f_g_of_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_GPT_f_g_of_4_l1344_134410


namespace NUMINAMATH_GPT_find_coefficients_l1344_134405

noncomputable def polynomial_h (x : ℚ) : ℚ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def polynomial_j (b c d x : ℚ) : ℚ := x^3 + b * x^2 + c * x + d

theorem find_coefficients :
  (∃ b c d : ℚ,
     (∀ s : ℚ, polynomial_h s = 0 → polynomial_j b c d (s^3) = 0) ∧
     (b, c, d) = (6, 12, 8)) :=
sorry

end NUMINAMATH_GPT_find_coefficients_l1344_134405


namespace NUMINAMATH_GPT_equalize_cheese_pieces_l1344_134412

-- Defining the initial masses of the three pieces of cheese
def cheese1 : ℕ := 5
def cheese2 : ℕ := 8
def cheese3 : ℕ := 11

-- State that the fox can cut 1g simultaneously from any two pieces
def can_equalize_masses (cut_action : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n1 n2 n3 _ : ℕ,
    cut_action cheese1 cheese2 cheese3 ∧
    (n1 = 0 ∧ n2 = 0 ∧ n3 = 0)

-- Introducing the fox's cut action
def cut_action (a b c : ℕ) : Prop :=
  (∃ x : ℕ, x ≥ 0 ∧ a - x ≥ 0 ∧ b - x ≥ 0 ∧ c ≤ cheese3) ∧
  (∃ y : ℕ, y ≥ 0 ∧ a - y ≥ 0 ∧ b ≤ cheese2 ∧ c - y ≥ 0) ∧
  (∃ z : ℕ, z ≥ 0 ∧ a ≤ cheese1 ∧ b - z ≥ 0 ∧ c - z ≥ 0) 

-- The theorem that proves it's possible to equalize the masses
theorem equalize_cheese_pieces : can_equalize_masses cut_action :=
by
  sorry

end NUMINAMATH_GPT_equalize_cheese_pieces_l1344_134412


namespace NUMINAMATH_GPT_avg_salary_rest_of_workers_l1344_134414

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_tech : ℝ) (total_workers : ℕ)
  (total_avg_salary : avg_all = 8000) (tech_avg_salary : avg_tech = 12000) (workers_count : total_workers = 30) :
  (20 * (total_workers * avg_all - 10 * avg_tech) / 20) = 6000 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_rest_of_workers_l1344_134414


namespace NUMINAMATH_GPT_min_ones_count_in_100_numbers_l1344_134495

def sum_eq_product (l : List ℕ) : Prop :=
  l.sum = l.prod

theorem min_ones_count_in_100_numbers : ∀ l : List ℕ, l.length = 100 → sum_eq_product l → l.count 1 ≥ 95 :=
by sorry

end NUMINAMATH_GPT_min_ones_count_in_100_numbers_l1344_134495


namespace NUMINAMATH_GPT_multiply_105_95_l1344_134474

theorem multiply_105_95 : 105 * 95 = 9975 :=
by
  sorry

end NUMINAMATH_GPT_multiply_105_95_l1344_134474


namespace NUMINAMATH_GPT_ratio_of_marbles_l1344_134437

noncomputable def marble_ratio : ℕ :=
  let initial_marbles := 40
  let marbles_after_breakfast := initial_marbles - 3
  let marbles_after_lunch := marbles_after_breakfast - 5
  let marbles_after_moms_gift := marbles_after_lunch + 12
  let final_marbles := 54
  let marbles_given_back_by_Susie := final_marbles - marbles_after_moms_gift
  marbles_given_back_by_Susie / 5

theorem ratio_of_marbles : marble_ratio = 2 := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_ratio_of_marbles_l1344_134437


namespace NUMINAMATH_GPT_maximum_x_y_value_l1344_134404

theorem maximum_x_y_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h1 : x + 2 * y ≤ 6) (h2 : 2 * x + y ≤ 6) : x + y ≤ 4 := 
sorry

end NUMINAMATH_GPT_maximum_x_y_value_l1344_134404


namespace NUMINAMATH_GPT_value_of_a_l1344_134426

theorem value_of_a (a b k : ℝ) (h1 : a = k / b^2) (h2 : a = 40) (h3 : b = 12) (h4 : b = 24) : a = 10 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1344_134426


namespace NUMINAMATH_GPT_min_S_l1344_134433

variable {x y : ℝ}
def condition (x y : ℝ) : Prop := (4 * x^2 + 5 * x * y + 4 * y^2 = 5)
def S (x y : ℝ) : ℝ := x^2 + y^2
theorem min_S (hx : condition x y) : S x y = (10 / 13) :=
sorry

end NUMINAMATH_GPT_min_S_l1344_134433


namespace NUMINAMATH_GPT_average_score_of_male_students_standard_deviation_of_all_students_l1344_134479

def students : ℕ := 5
def total_average_score : ℝ := 80
def male_student_variance : ℝ := 150
def female_student1_score : ℝ := 85
def female_student2_score : ℝ := 75
def male_student_average_score : ℝ := 80 -- From solution step (1)
def total_standard_deviation : ℝ := 10 -- From solution step (2)

theorem average_score_of_male_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  male_student_average_score = 80 :=
by sorry

theorem standard_deviation_of_all_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  total_standard_deviation = 10 :=
by sorry

end NUMINAMATH_GPT_average_score_of_male_students_standard_deviation_of_all_students_l1344_134479


namespace NUMINAMATH_GPT_musketeers_strength_order_l1344_134456

variables {A P R D : ℝ}

theorem musketeers_strength_order 
  (h1 : P + D > A + R)
  (h2 : P + A > R + D)
  (h3 : P + R = A + D) : 
  P > D ∧ D > A ∧ A > R :=
by
  sorry

end NUMINAMATH_GPT_musketeers_strength_order_l1344_134456


namespace NUMINAMATH_GPT_regular_polygon_sides_l1344_134449

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1344_134449


namespace NUMINAMATH_GPT_difference_between_mean_and_median_l1344_134402

namespace MathProof

noncomputable def percentage_72 := 0.12
noncomputable def percentage_82 := 0.30
noncomputable def percentage_87 := 0.18
noncomputable def percentage_91 := 0.10
noncomputable def percentage_96 := 1 - (percentage_72 + percentage_82 + percentage_87 + percentage_91)

noncomputable def num_students := 20
noncomputable def scores := [72, 72, 82, 82, 82, 82, 82, 82, 87, 87, 87, 87, 91, 91, 96, 96, 96, 96, 96, 96]

noncomputable def mean_score : ℚ := (72 * 2 + 82 * 6 + 87 * 4 + 91 * 2 + 96 * 6) / num_students
noncomputable def median_score : ℚ := 87

theorem difference_between_mean_and_median :
  mean_score - median_score = 0.1 := by
  sorry

end MathProof

end NUMINAMATH_GPT_difference_between_mean_and_median_l1344_134402


namespace NUMINAMATH_GPT_solve_quadratic_eq_solve_linear_system_l1344_134400

theorem solve_quadratic_eq (x : ℚ) : 4 * (x - 1) ^ 2 - 25 = 0 ↔ x = 7 / 2 ∨ x = -3 / 2 := 
by sorry

theorem solve_linear_system (x y : ℚ) : (2 * x - y = 4) ∧ (3 * x + 2 * y = 1) ↔ (x = 9 / 7 ∧ y = -10 / 7) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_solve_linear_system_l1344_134400


namespace NUMINAMATH_GPT_number_of_sodas_bought_l1344_134411

theorem number_of_sodas_bought
  (sandwich_cost : ℝ)
  (num_sandwiches : ℝ)
  (soda_cost : ℝ)
  (total_cost : ℝ)
  (h1 : sandwich_cost = 3.49)
  (h2 : num_sandwiches = 2)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 10.46) :
  (total_cost - num_sandwiches * sandwich_cost) / soda_cost = 4 := 
sorry

end NUMINAMATH_GPT_number_of_sodas_bought_l1344_134411


namespace NUMINAMATH_GPT_measure_angle_C_l1344_134455

theorem measure_angle_C (A B C : ℝ) (h1 : A = 60) (h2 : B = 60) (h3 : C = 60 - 10) (sum_angles : A + B + C = 180) : C = 53.33 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_C_l1344_134455


namespace NUMINAMATH_GPT_saree_final_price_l1344_134401

noncomputable def saree_original_price : ℝ := 5000
noncomputable def first_discount_rate : ℝ := 0.20
noncomputable def second_discount_rate : ℝ := 0.15
noncomputable def third_discount_rate : ℝ := 0.10
noncomputable def fourth_discount_rate : ℝ := 0.05
noncomputable def tax_rate : ℝ := 0.12
noncomputable def luxury_tax_rate : ℝ := 0.05
noncomputable def custom_fee : ℝ := 200
noncomputable def exchange_rate_to_usd : ℝ := 0.013

theorem saree_final_price :
  let price_after_first_discount := saree_original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let price_after_third_discount := price_after_second_discount * (1 - third_discount_rate)
  let price_after_fourth_discount := price_after_third_discount * (1 - fourth_discount_rate)
  let tax := price_after_fourth_discount * tax_rate
  let luxury_tax := price_after_fourth_discount * luxury_tax_rate
  let total_charges := tax + luxury_tax + custom_fee
  let total_price_rs := price_after_fourth_discount + total_charges
  let final_price_usd := total_price_rs * exchange_rate_to_usd
  abs (final_price_usd - 46.82) < 0.01 :=
by sorry

end NUMINAMATH_GPT_saree_final_price_l1344_134401


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1344_134489

theorem cyclic_sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
    (ab / (ab + a^5 + b^5)) + (bc / (bc + b^5 + c^5)) + (ca / (ca + c^5 + a^5)) ≤ 1 := by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1344_134489


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1344_134436

def speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def crossing_time_s : ℝ := 30

noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_m_s * crossing_time_s
noncomputable def bridge_length_m : ℝ := total_distance_m - train_length_m

theorem bridge_length_is_correct : bridge_length_m = 255 := by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1344_134436


namespace NUMINAMATH_GPT_emma_investment_l1344_134443

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end NUMINAMATH_GPT_emma_investment_l1344_134443


namespace NUMINAMATH_GPT_shorter_piece_length_l1344_134434

theorem shorter_piece_length (x : ℕ) (h1 : ∃ l : ℕ, x + l = 120 ∧ l = 2 * x + 15) : x = 35 :=
sorry

end NUMINAMATH_GPT_shorter_piece_length_l1344_134434


namespace NUMINAMATH_GPT_scientific_notation_for_70_million_l1344_134481

-- Define the parameters for the problem
def scientific_notation (x : ℕ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Problem statement
theorem scientific_notation_for_70_million :
  scientific_notation 70000000 7.0 7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_for_70_million_l1344_134481


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1344_134462

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1344_134462


namespace NUMINAMATH_GPT_vector_parallel_l1344_134427

theorem vector_parallel {x : ℝ} (h : (4 / x) = (-2 / 5)) : x = -10 :=
  by
  sorry

end NUMINAMATH_GPT_vector_parallel_l1344_134427


namespace NUMINAMATH_GPT_xiaoming_age_l1344_134472

theorem xiaoming_age
  (x x' : ℕ) 
  (h₁ : ∃ f : ℕ, f = 4 * x) 
  (h₂ : (x + 25) + (4 * x + 25) = 100) : 
  x = 10 :=
by
  obtain ⟨f, hf⟩ := h₁
  sorry

end NUMINAMATH_GPT_xiaoming_age_l1344_134472


namespace NUMINAMATH_GPT_abs_difference_of_mn_6_and_sum_7_l1344_134451

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end NUMINAMATH_GPT_abs_difference_of_mn_6_and_sum_7_l1344_134451


namespace NUMINAMATH_GPT_length_of_train_l1344_134441

-- We define the conditions
def crosses_platform_1 (L : ℝ) : Prop := 
  let v := (L + 100) / 15
  v = (L + 100) / 15

def crosses_platform_2 (L : ℝ) : Prop := 
  let v := (L + 250) / 20
  v = (L + 250) / 20

-- We state the main theorem we need to prove
theorem length_of_train :
  ∃ L : ℝ, crosses_platform_1 L ∧ crosses_platform_2 L ∧ (L = 350) :=
sorry

end NUMINAMATH_GPT_length_of_train_l1344_134441


namespace NUMINAMATH_GPT_total_volume_correct_l1344_134444

-- Defining the initial conditions
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Given the above conditions, define the total volume of all cubes.
def total_volume_of_all_cubes : ℕ := (carl_cubes * carl_side_length ^ 3) + (kate_cubes * kate_side_length ^ 3)

-- The statement we need to prove
theorem total_volume_correct :
  total_volume_of_all_cubes = 114 :=
by
  -- Skipping the proof with sorry as per the instruction
  sorry

end NUMINAMATH_GPT_total_volume_correct_l1344_134444


namespace NUMINAMATH_GPT_roman_coins_left_l1344_134488

theorem roman_coins_left (X Y : ℕ) (h1 : X * Y = 50) (h2 : (X - 7) * Y = 28) : X - 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_roman_coins_left_l1344_134488


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1344_134440

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1344_134440


namespace NUMINAMATH_GPT_gain_percentage_is_8_l1344_134482

variable (C S : ℝ) (D : ℝ)
variable (h1 : 20 * C * (1 - D / 100) = 12 * S)
variable (h2 : D ≥ 5 ∧ D ≤ 25)

theorem gain_percentage_is_8 :
  (12 * S * 1.08 - 20 * C * (1 - D / 100)) / (20 * C * (1 - D / 100)) * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_is_8_l1344_134482


namespace NUMINAMATH_GPT_simplify_fraction_l1344_134491

theorem simplify_fraction : (3 ^ 2016 - 3 ^ 2014) / (3 ^ 2016 + 3 ^ 2014) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1344_134491


namespace NUMINAMATH_GPT_problem_l1344_134407

noncomputable def k : ℝ := 2.9

theorem problem (k : ℝ) (hₖ : k > 1) 
    (h_sum : ∑' n, (7 * n + 2) / k^n = 20 / 3) : 
    k = 2.9 := 
sorry

end NUMINAMATH_GPT_problem_l1344_134407


namespace NUMINAMATH_GPT_average_score_is_67_l1344_134464

def scores : List ℕ := [55, 67, 76, 82, 55]
def num_of_subjects : ℕ := List.length scores
def total_score : ℕ := List.sum scores
def average_score : ℕ := total_score / num_of_subjects

theorem average_score_is_67 : average_score = 67 := by
  sorry

end NUMINAMATH_GPT_average_score_is_67_l1344_134464


namespace NUMINAMATH_GPT_only_set_C_forms_triangle_l1344_134450

def triangle_inequality (a b c : ℝ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_C_forms_triangle : 
  (¬ triangle_inequality 1 2 3) ∧ 
  (¬ triangle_inequality 2 3 6) ∧ 
  triangle_inequality 4 6 8 ∧ 
  (¬ triangle_inequality 5 6 12) := 
by 
  sorry

end NUMINAMATH_GPT_only_set_C_forms_triangle_l1344_134450


namespace NUMINAMATH_GPT_Sarah_ate_one_apple_l1344_134454

theorem Sarah_ate_one_apple:
  ∀ (total_apples apples_given_to_teachers apples_given_to_friends apples_left: ℕ), 
  total_apples = 25 →
  apples_given_to_teachers = 16 →
  apples_given_to_friends = 5 →
  apples_left = 3 →
  total_apples - (apples_given_to_teachers + apples_given_to_friends + apples_left) = 1 :=
by
  intros total_apples apples_given_to_teachers apples_given_to_friends apples_left
  intro ht ht gt hf
  sorry

end NUMINAMATH_GPT_Sarah_ate_one_apple_l1344_134454


namespace NUMINAMATH_GPT_worker_allocation_correct_l1344_134420

variable (x y : ℕ)
variable (H1 : x + y = 50)
variable (H2 : x = 30)
variable (H3 : y = 20)
variable (H4 : 120 * (50 - x) = 2 * 40 * x)

theorem worker_allocation_correct 
  (h₁ : x = 30) 
  (h₂ : y = 20) 
  (h₃ : x + y = 50) 
  (h₄ : 120 * (50 - x) = 2 * 40 * x) 
  : true := 
by
  sorry

end NUMINAMATH_GPT_worker_allocation_correct_l1344_134420


namespace NUMINAMATH_GPT_M_even_comp_M_composite_comp_M_prime_not_div_l1344_134484

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_composite (n : ℕ) : Prop :=  ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n
def M (n : ℕ) : ℕ := 2^n - 1

theorem M_even_comp (n : ℕ) (h1 : n ≠ 2) (h2 : is_even n) : is_composite (M n) :=
sorry

theorem M_composite_comp (n : ℕ) (h : is_composite n) : is_composite (M n) :=
sorry

theorem M_prime_not_div (p : ℕ) (h : Nat.Prime p) : ¬ (p ∣ M p) :=
sorry

end NUMINAMATH_GPT_M_even_comp_M_composite_comp_M_prime_not_div_l1344_134484


namespace NUMINAMATH_GPT_no_such_positive_integer_l1344_134408

theorem no_such_positive_integer (n : ℕ) (d : ℕ → ℕ)
  (h₁ : ∃ d1 d2 d3 d4 d5, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ d 5 = d5) 
  (h₂ : 1 ≤ d 1 ∧ d 1 < d 2 ∧ d 2 < d 3 ∧ d 3 < d 4 ∧ d 4 < d 5)
  (h₃ : ∀ i, 1 ≤ i → i ≤ 5 → d i ∣ n)
  (h₄ : ∀ i, 1 ≤ i → i ≤ 5 → ∀ j, i ≠ j → d i ≠ d j)
  (h₅ : ∃ x, 1 + (d 2)^2 + (d 3)^2 + (d 4)^2 + (d 5)^2 = x^2) :
  false :=
sorry

end NUMINAMATH_GPT_no_such_positive_integer_l1344_134408


namespace NUMINAMATH_GPT_number_of_ways_to_choose_museums_l1344_134409

-- Define the conditions
def number_of_grades : Nat := 6
def number_of_museums : Nat := 6
def number_of_grades_Museum_A : Nat := 2

-- Prove the number of ways to choose museums such that exactly two grades visit Museum A
theorem number_of_ways_to_choose_museums :
  (Nat.choose number_of_grades number_of_grades_Museum_A) * (5 ^ (number_of_grades - number_of_grades_Museum_A)) = Nat.choose 6 2 * 5 ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_museums_l1344_134409


namespace NUMINAMATH_GPT_ball_hits_ground_l1344_134429

theorem ball_hits_ground :
  ∃ (t : ℝ), (t = 2) ∧ (-4.9 * t^2 + 5.7 * t + 7 = 0) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_l1344_134429


namespace NUMINAMATH_GPT_not_perfect_square_of_sum_300_l1344_134417

theorem not_perfect_square_of_sum_300 : ¬(∃ n : ℕ, n = 10^300 - 1 ∧ (∃ m : ℕ, n = m^2)) :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_of_sum_300_l1344_134417


namespace NUMINAMATH_GPT_pyramid_side_length_l1344_134458

-- Definitions for our conditions
def area_of_lateral_face : ℝ := 150
def slant_height : ℝ := 25

-- Theorem statement
theorem pyramid_side_length (A : ℝ) (h : ℝ) (s : ℝ) (hA : A = area_of_lateral_face) (hh : h = slant_height) :
  A = (1 / 2) * s * h → s = 12 :=
by
  intro h_eq
  rw [hA, hh, area_of_lateral_face, slant_height] at h_eq
  -- Steps to verify s = 12
  sorry

end NUMINAMATH_GPT_pyramid_side_length_l1344_134458


namespace NUMINAMATH_GPT_draw_at_least_two_first_grade_products_l1344_134423

theorem draw_at_least_two_first_grade_products :
  let total_products := 9
  let first_grade := 4
  let second_grade := 3
  let third_grade := 2
  let total_draws := 4
  let ways_to_draw := Nat.choose total_products total_draws
  let ways_no_first_grade := Nat.choose (second_grade + third_grade) total_draws
  let ways_one_first_grade := Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (total_draws - 1)
  ways_to_draw - ways_no_first_grade - ways_one_first_grade = 81 := sorry

end NUMINAMATH_GPT_draw_at_least_two_first_grade_products_l1344_134423


namespace NUMINAMATH_GPT_root_in_interval_l1344_134448

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem root_in_interval : ∃ m, f m = 0 ∧ 2 < m ∧ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l1344_134448


namespace NUMINAMATH_GPT_loan_duration_l1344_134463

theorem loan_duration (P R SI : ℝ) (hP : P = 20000) (hR : R = 12) (hSI : SI = 7200) : 
  ∃ T : ℝ, T = 3 :=
by
  sorry

end NUMINAMATH_GPT_loan_duration_l1344_134463


namespace NUMINAMATH_GPT_find_f_24_25_26_l1344_134452

-- Given conditions
def homogeneous (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (n a b c : ℤ), f (n * a) (n * b) (n * c) = n * f a b c

def shift_invariance (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c n : ℤ), f (a + n) (b + n) (c + n) = f a b c + n

def symmetry (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), f a b c = f c b a

-- Proving the required value under the conditions
theorem find_f_24_25_26 (f : ℤ → ℤ → ℤ → ℝ)
  (homo : homogeneous f) 
  (shift : shift_invariance f) 
  (symm : symmetry f) : 
  f 24 25 26 = 25 := 
sorry

end NUMINAMATH_GPT_find_f_24_25_26_l1344_134452


namespace NUMINAMATH_GPT_valentine_giveaway_l1344_134406

theorem valentine_giveaway (initial : ℕ) (left : ℕ) (given : ℕ) (h1 : initial = 30) (h2 : left = 22) : given = initial - left → given = 8 :=
by
  sorry

end NUMINAMATH_GPT_valentine_giveaway_l1344_134406


namespace NUMINAMATH_GPT_fully_charge_tablet_time_l1344_134447

def time_to_fully_charge_smartphone := 26 -- 26 minutes to fully charge a smartphone
def total_charge_time := 66 -- 66 minutes to charge tablet fully and phone halfway
def halfway_charge_time := time_to_fully_charge_smartphone / 2 -- 13 minutes to charge phone halfway

theorem fully_charge_tablet_time : 
  ∃ T : ℕ, T + halfway_charge_time = total_charge_time ∧ T = 53 := 
by
  sorry

end NUMINAMATH_GPT_fully_charge_tablet_time_l1344_134447


namespace NUMINAMATH_GPT_tangent_line_eq_l1344_134413

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_eq
  (a b : ℝ)
  (h1 : 3 + 2*a + b = 2*a)
  (h2 : 12 + 4*a + b = -b)
  : ∀ x y : ℝ , (f a b 1 = -5/2 ∧
  y - (f a b 1) = -3 * (x - 1))
  → (6*x + 2*y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1344_134413


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l1344_134460

theorem quadratic_to_vertex_form : ∃ m n : ℝ, (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 :=
by sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l1344_134460


namespace NUMINAMATH_GPT_triangle_sequence_relation_l1344_134475

theorem triangle_sequence_relation (b d c k : ℤ) (h₁ : b % d = 0) (h₂ : c % k = 0) (h₃ : b^2 + (b + 2*d)^2 = (c + 6*k)^2) :
  c = 0 :=
sorry

end NUMINAMATH_GPT_triangle_sequence_relation_l1344_134475


namespace NUMINAMATH_GPT_problem_solution_set_l1344_134424

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_set_l1344_134424


namespace NUMINAMATH_GPT_distance_between_cities_l1344_134466

-- Conditions Initialization
variable (A B : ℝ)
variable (S : ℕ)
variable (x : ℕ)
variable (gcd_values : ℕ)

-- Conditions:
-- 1. The distance between cities A and B is an integer.
-- 2. Markers are placed every kilometer.
-- 3. At each marker, one side has distance to city A, and the other has distance to city B.
-- 4. The GCDs observed were only 1, 3, and 13.

-- Given these conditions, we prove the distance is exactly 39 kilometers.
theorem distance_between_cities (h1 : gcd_values = 1 ∨ gcd_values = 3 ∨ gcd_values = 13)
    (h2 : S = Nat.lcm 1 (Nat.lcm 3 13)) :
    S = 39 := by 
  sorry

end NUMINAMATH_GPT_distance_between_cities_l1344_134466


namespace NUMINAMATH_GPT_num_people_visited_iceland_l1344_134442

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end NUMINAMATH_GPT_num_people_visited_iceland_l1344_134442


namespace NUMINAMATH_GPT_cookies_left_l1344_134439

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end NUMINAMATH_GPT_cookies_left_l1344_134439


namespace NUMINAMATH_GPT_isosceles_trapezoid_problem_l1344_134465

variable (AB CD AD BC : ℝ)
variable (x : ℝ)

noncomputable def p_squared (AB CD AD BC : ℝ) (x : ℝ) : ℝ :=
  if AB = 100 ∧ CD = 25 ∧ AD = x ∧ BC = x then 1875 else 0

theorem isosceles_trapezoid_problem (h₁ : AB = 100)
                                    (h₂ : CD = 25)
                                    (h₃ : AD = x)
                                    (h₄ : BC = x) :
  p_squared AB CD AD BC x = 1875 := by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_problem_l1344_134465


namespace NUMINAMATH_GPT_farmer_earns_from_runt_pig_l1344_134403

def average_bacon_per_pig : ℕ := 20
def price_per_pound : ℕ := 6
def runt_pig_bacon : ℕ := average_bacon_per_pig / 2
def total_money_made (bacon_pounds : ℕ) (price_per_pound : ℕ) : ℕ := bacon_pounds * price_per_pound

theorem farmer_earns_from_runt_pig :
  total_money_made runt_pig_bacon price_per_pound = 60 :=
sorry

end NUMINAMATH_GPT_farmer_earns_from_runt_pig_l1344_134403


namespace NUMINAMATH_GPT_trajectory_of_C_is_ellipse_l1344_134470

theorem trajectory_of_C_is_ellipse :
  ∀ (C : ℝ × ℝ),
  ((C.1 + 4)^2 + C.2^2).sqrt + ((C.1 - 4)^2 + C.2^2).sqrt = 10 →
  (C.2 ≠ 0) →
  (C.1^2 / 25 + C.2^2 / 9 = 1) :=
by
  intros C h1 h2
  sorry

end NUMINAMATH_GPT_trajectory_of_C_is_ellipse_l1344_134470


namespace NUMINAMATH_GPT_seafoam_azure_ratio_l1344_134477

-- Define the conditions
variables (P S A : ℕ) 

-- Purple Valley has one-quarter as many skirts as Seafoam Valley
axiom h1 : P = S / 4

-- Azure Valley has 60 skirts
axiom h2 : A = 60

-- Purple Valley has 10 skirts
axiom h3 : P = 10

-- The goal is to prove the ratio of Seafoam Valley skirts to Azure Valley skirts is 2 to 3
theorem seafoam_azure_ratio : S / A = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_seafoam_azure_ratio_l1344_134477


namespace NUMINAMATH_GPT_dividend_is_correct_l1344_134468

def quotient : ℕ := 20
def divisor : ℕ := 66
def remainder : ℕ := 55

def dividend := (divisor * quotient) + remainder

theorem dividend_is_correct : dividend = 1375 := by
  sorry

end NUMINAMATH_GPT_dividend_is_correct_l1344_134468


namespace NUMINAMATH_GPT_sum_of_numbers_gt_1_1_equals_3_9_l1344_134494

noncomputable def sum_of_elements_gt_1_1 : Float :=
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  let numbers_gt_1_1 := List.filter (fun x => x > 1.1) numbers
  List.sum numbers_gt_1_1

theorem sum_of_numbers_gt_1_1_equals_3_9 :
  sum_of_elements_gt_1_1 = 3.9 := by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_gt_1_1_equals_3_9_l1344_134494


namespace NUMINAMATH_GPT_distance_from_A_to_C_l1344_134421

theorem distance_from_A_to_C (x y : ℕ) (d : ℚ)
  (h1 : d = x / 3) 
  (h2 : 13 + (d * 15) / (y - 13) = 2 * x)
  (h3 : y = 2 * x + 13) 
  : x + y = 26 := 
  sorry

end NUMINAMATH_GPT_distance_from_A_to_C_l1344_134421


namespace NUMINAMATH_GPT_avg_distance_is_600_l1344_134430

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end NUMINAMATH_GPT_avg_distance_is_600_l1344_134430


namespace NUMINAMATH_GPT_least_prime_b_l1344_134461

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end NUMINAMATH_GPT_least_prime_b_l1344_134461


namespace NUMINAMATH_GPT_distance_AK_l1344_134473

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

-- Define the line equations
noncomputable def line_AB (x : ℝ) : Prop := x = 0
noncomputable def line_CD (x y : ℝ) : Prop := y = (Real.sqrt 2) / (2 - Real.sqrt 2) * (x - 1)

-- Define the intersection point K
noncomputable def K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the desired distance
theorem distance_AK : distance A K = Real.sqrt 2 + 1 :=
by
  -- Proof details are omitted
  sorry

end NUMINAMATH_GPT_distance_AK_l1344_134473


namespace NUMINAMATH_GPT_baseball_card_decrease_l1344_134493

noncomputable def percentDecrease (V : ℝ) (P : ℝ) : ℝ :=
  V * (P / 100)

noncomputable def valueAfterDecrease (V : ℝ) (D : ℝ) : ℝ :=
  V - D

theorem baseball_card_decrease (V : ℝ) (H1 : V > 0) :
  let D1 := percentDecrease V 50
  let V1 := valueAfterDecrease V D1
  let D2 := percentDecrease V1 10
  let V2 := valueAfterDecrease V1 D2
  let totalDecrease := V - V2
  totalDecrease / V * 100 = 55 := sorry

end NUMINAMATH_GPT_baseball_card_decrease_l1344_134493


namespace NUMINAMATH_GPT_next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l1344_134490

-- Part (a)
theorem next_terms_arithmetic_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ), 
  a₀ = 3 → a₁ = 7 → a₂ = 11 → a₃ = 15 → a₄ = 19 → a₅ = 23 → d = 4 →
  (a₅ + d = 27) ∧ (a₅ + 2*d = 31) :=
by intros; sorry


-- Part (b)
theorem next_terms_alternating_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℕ),
  a₀ = 9 → a₁ = 1 → a₂ = 7 → a₃ = 1 → a₄ = 5 → a₅ = 1 →
  a₄ - 2 = 3 ∧ a₁ = 1 :=
by intros; sorry


-- Part (c)
theorem next_terms_interwoven_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ),
  a₀ = 4 → a₁ = 5 → a₂ = 8 → a₃ = 9 → a₄ = 12 → a₅ = 13 → d = 4 →
  (a₄ + d = 16) ∧ (a₅ + d = 17) :=
by intros; sorry


-- Part (d)
theorem next_terms_geometric_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅: ℕ), 
  a₀ = 1 → a₁ = 2 → a₂ = 4 → a₃ = 8 → a₄ = 16 → a₅ = 32 →
  (a₅ * 2 = 64) ∧ (a₅ * 4 = 128) :=
by intros; sorry

end NUMINAMATH_GPT_next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l1344_134490


namespace NUMINAMATH_GPT_largest_n_exists_l1344_134459

theorem largest_n_exists :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (1 + x i * x j)^2 ≤ 0.99 * (1 + x i^2) * (1 + x j^2))) ∧ n = 31 :=
sorry

end NUMINAMATH_GPT_largest_n_exists_l1344_134459
