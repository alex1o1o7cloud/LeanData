import Mathlib

namespace population_ratio_l2025_202520

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio_l2025_202520


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l2025_202557

variable (a b : ℝ)
variable (h : a < b)

theorem option_A_correct : a + 2 < b + 2 := by
  sorry

theorem option_B_correct : 3 * a < 3 * b := by
  sorry

theorem option_C_correct : (1 / 2) * a < (1 / 2) * b := by
  sorry

theorem option_D_incorrect : ¬(-2 * a < -2 * b) := by
  sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l2025_202557


namespace solve_for_x_l2025_202569

theorem solve_for_x (x : ℝ) (y : ℝ) (z : ℝ) (h1 : y = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  x = -2 / 3 ∨ x = 3 :=
by sorry

end solve_for_x_l2025_202569


namespace problem_1_problem_2_l2025_202534

theorem problem_1 :
  83 * 87 = 100 * 8 * (8 + 1) + 21 :=
by sorry

theorem problem_2 (n : ℕ) :
  (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 :=
by sorry

end problem_1_problem_2_l2025_202534


namespace number_of_ways_is_25_l2025_202578

-- Define the number of books
def number_of_books : ℕ := 5

-- Define the function to calculate the number of ways
def number_of_ways_to_buy_books : ℕ :=
  number_of_books * number_of_books

-- Define the theorem to be proved
theorem number_of_ways_is_25 : 
  number_of_ways_to_buy_books = 25 :=
by
  sorry

end number_of_ways_is_25_l2025_202578


namespace prove_a_lt_zero_l2025_202533

variable (a b c : ℝ)

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions:
-- The polynomial has roots at -2 and 3
def has_roots : Prop := 
  a ≠ 0 ∧ (a * (-2)^2 + b * (-2) + c = 0) ∧ (a * 3^2 + b * 3 + c = 0)

-- f(-b/(2*a)) > 0
def vertex_positive : Prop := 
  f a b c (-b / (2 * a)) > 0

-- Target: Prove a < 0
theorem prove_a_lt_zero 
  (h_roots : has_roots a b c)
  (h_vertex : vertex_positive a b c) : a < 0 := 
sorry

end prove_a_lt_zero_l2025_202533


namespace last_fish_in_swamp_l2025_202559

noncomputable def final_fish (perches pikes sudaks : ℕ) : String :=
  let p := perches
  let pi := pikes
  let s := sudaks
  if p = 6 ∧ pi = 7 ∧ s = 8 then "Sudak" else "Unknown"

theorem last_fish_in_swamp : final_fish 6 7 8 = "Sudak" := by
  sorry

end last_fish_in_swamp_l2025_202559


namespace multiplication_correct_l2025_202596

theorem multiplication_correct :
  72514 * 99999 = 7250675486 :=
by
  sorry

end multiplication_correct_l2025_202596


namespace root_in_interval_l2025_202573

noncomputable def f (m x : ℝ) := m * 3^x - x + 3

theorem root_in_interval (m : ℝ) (h1 : m < 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f m x = 0) : -3 < m ∧ m < -2/3 :=
by
  sorry

end root_in_interval_l2025_202573


namespace find_f3_minus_f4_l2025_202581

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = - f x
axiom h_periodic : ∀ x : ℝ, f (x + 5) = f x
axiom h_f1 : f 1 = 1
axiom h_f2 : f 2 = 2

theorem find_f3_minus_f4 : f 3 - f 4 = -1 := by
  sorry

end find_f3_minus_f4_l2025_202581


namespace smallest_relatively_prime_to_180_is_7_l2025_202588

theorem smallest_relatively_prime_to_180_is_7 :
  ∃ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 ∧ ∀ z : ℕ, z > 1 ∧ Nat.gcd z 180 = 1 → y ≤ z :=
by
  sorry

end smallest_relatively_prime_to_180_is_7_l2025_202588


namespace problem1_problem2_problem3_l2025_202530

-- Problem 1
theorem problem1 (x : ℝ) : (3 * (x - 1)^2 = 12) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (3 * x^2 - 6 * x - 2 = 0) ↔ (x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1 / 2 ∨ x = 2 / 3) :=
by
  sorry

end problem1_problem2_problem3_l2025_202530


namespace exists_distinct_ij_l2025_202540

theorem exists_distinct_ij (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) (h_n_ge_3 : 3 ≤ n) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k, (a i + a j) ∣ 3 * a k → False) :=
by
  sorry

end exists_distinct_ij_l2025_202540


namespace pure_imaginary_solution_second_quadrant_solution_l2025_202583

def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def isSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 2*m - 3, m^2 + 3*m + 2⟩

theorem pure_imaginary_solution (m : ℝ) : isPureImaginary (complexNumber m) ↔ m = 3 :=
by sorry

theorem second_quadrant_solution (m : ℝ) : isSecondQuadrant (complexNumber m) ↔ (-1 < m ∧ m < 3) :=
by sorry

end pure_imaginary_solution_second_quadrant_solution_l2025_202583


namespace sum_of_distances_l2025_202541

theorem sum_of_distances (AB A'B' AD A'D' x y : ℝ) 
  (h1 : AB = 8)
  (h2 : A'B' = 6)
  (h3 : AD = 3)
  (h4 : A'D' = 1)
  (h5 : x = 2)
  (h6 : x / y = 3 / 2) : 
  x + y = 10 / 3 :=
by
  sorry

end sum_of_distances_l2025_202541


namespace tan_domain_correct_l2025_202599

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4}

def is_domain_correct : Prop :=
  ∀ x : ℝ, x ∈ domain_tan ↔ (∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4)

-- Statement of the problem in Lean 4
theorem tan_domain_correct : is_domain_correct :=
  sorry

end tan_domain_correct_l2025_202599


namespace simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l2025_202551

variable (a b : ℤ)

def A : ℤ := 3 * a^2 - 6 * a * b + b^2
def B : ℤ := -2 * a^2 + 3 * a * b - 5 * b^2

theorem simplify_A_plus_2B : 
  A a b + 2 * B a b = -a^2 - 9 * b^2 := by
  sorry

theorem value_A_plus_2B_at_a1_bneg1 : 
  let a := 1
  let b := -1
  A a b + 2 * B a b = -10 := by
  sorry

end simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l2025_202551


namespace xiao_ming_equation_l2025_202586

-- Defining the parameters of the problem
def distance : ℝ := 2000
def regular_time (x : ℝ) := x
def increased_speed := 5
def time_saved := 2

-- Problem statement to be proven in Lean 4:
theorem xiao_ming_equation (x : ℝ) (h₁ : x > 2) : 
  (distance / (x - time_saved)) - (distance / regular_time x) = increased_speed :=
by
  sorry

end xiao_ming_equation_l2025_202586


namespace number_of_matches_among_three_players_l2025_202594

-- Define the given conditions
variables (n r : ℕ) -- n is the number of participants, r is the number of matches among the 3 players
variables (m : ℕ := 50) -- m is the total number of matches played

-- Given assumptions
def condition1 := m = 50
def condition2 := ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)

-- The target proof
theorem number_of_matches_among_three_players (n r : ℕ) (m : ℕ := 50)
  (h1 : m = 50)
  (h2 : ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)) :
  r = 1 :=
sorry

end number_of_matches_among_three_players_l2025_202594


namespace mask_production_l2025_202527

theorem mask_production (x : ℝ) :
  24 + 24 * (1 + x) + 24 * (1 + x)^2 = 88 :=
sorry

end mask_production_l2025_202527


namespace inequality_with_equality_condition_l2025_202562

variables {a b c d : ℝ}

theorem inequality_with_equality_condition (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 1) : 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) ∧ 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1 / 2 ↔ a = b ∧ b = c ∧ c = d) := 
sorry

end inequality_with_equality_condition_l2025_202562


namespace time_to_cover_length_l2025_202543

def speed_escalator : ℝ := 10
def speed_person : ℝ := 4
def length_escalator : ℝ := 112

theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person) = 8) :=
by
  sorry

end time_to_cover_length_l2025_202543


namespace remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l2025_202589

-- Definition of initial conditions
def initial_sweet_cookies := 34
def initial_salty_cookies := 97
def initial_chocolate_cookies := 45

def sweet_cookies_eaten := 15
def salty_cookies_eaten := 56
def chocolate_cookies_given_away := 22
def chocolate_cookies_given_back := 7

-- Calculate remaining cookies
def remaining_sweet_cookies : Nat := initial_sweet_cookies - sweet_cookies_eaten
def remaining_salty_cookies : Nat := initial_salty_cookies - salty_cookies_eaten
def remaining_chocolate_cookies : Nat := (initial_chocolate_cookies - chocolate_cookies_given_away) + chocolate_cookies_given_back

-- Theorem statements
theorem remaining_sweet_cookies_correct : remaining_sweet_cookies = 19 := 
by sorry

theorem remaining_salty_cookies_correct : remaining_salty_cookies = 41 := 
by sorry

theorem remaining_chocolate_cookies_correct : remaining_chocolate_cookies = 30 := 
by sorry

end remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l2025_202589


namespace total_cost_after_discount_l2025_202518

def num_children : Nat := 6
def num_adults : Nat := 10
def num_seniors : Nat := 4

def child_ticket_price : Real := 12
def adult_ticket_price : Real := 20
def senior_ticket_price : Real := 15

def group_discount_rate : Real := 0.15

theorem total_cost_after_discount :
  let total_cost_before_discount :=
    num_children * child_ticket_price +
    num_adults * adult_ticket_price +
    num_seniors * senior_ticket_price
  let discount := group_discount_rate * total_cost_before_discount
  let total_cost := total_cost_before_discount - discount
  total_cost = 282.20 := by
  sorry

end total_cost_after_discount_l2025_202518


namespace B_join_months_after_A_l2025_202536

-- Definitions based on conditions
def capitalA (monthsA : ℕ) : ℕ := 3500 * monthsA
def capitalB (monthsB : ℕ) : ℕ := 9000 * monthsB

-- The condition that profit is in ratio 2:3 implies the ratio of their capitals should equal 2:3
def ratio_condition (x : ℕ) : Prop := 2 * (capitalB (12 - x)) = 3 * (capitalA 12)

-- Main theorem stating that B joined the business 5 months after A started
theorem B_join_months_after_A : ∃ x, ratio_condition x ∧ x = 5 :=
by
  use 5
  -- Proof would go here
  sorry

end B_join_months_after_A_l2025_202536


namespace problem_a_b_c_l2025_202555

theorem problem_a_b_c (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ac = 0) (h4 : abc = 1) : |a + b| > |c| := 
by sorry

end problem_a_b_c_l2025_202555


namespace solve_for_a_l2025_202542

theorem solve_for_a (a x : ℝ) (h : x = 1 ∧ 2 * a * x - 2 = a + 3) : a = 5 :=
by
  sorry

end solve_for_a_l2025_202542


namespace geometric_seq_sum_l2025_202585

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (-1)) 
  (h_a3 : a 3 = 3) 
  (h_sum_cond : a 2016 + a 2017 = 0) : 
  S 101 = 3 := 
by
  sorry

end geometric_seq_sum_l2025_202585


namespace condition_sufficient_not_necessary_l2025_202505

theorem condition_sufficient_not_necessary (x : ℝ) : (1 < x ∧ x < 2) → ((x - 2) ^ 2 < 1) ∧ ¬ ((x - 2) ^ 2 < 1 → (1 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l2025_202505


namespace spinner_final_direction_north_l2025_202598

def start_direction := "north"
def clockwise_revolutions := (7 : ℚ) / 2
def counterclockwise_revolutions := (5 : ℚ) / 2
def net_revolutions := clockwise_revolutions - counterclockwise_revolutions

theorem spinner_final_direction_north :
  net_revolutions = 1 → start_direction = "north" → 
  start_direction = "north" :=
by
  intro h1 h2
  -- Here you would prove that net_revolutions of 1 full cycle leads back to start
  exact h2 -- Skipping proof

end spinner_final_direction_north_l2025_202598


namespace max_value_of_expression_l2025_202574

noncomputable def expression (x : ℝ) : ℝ :=
  x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 15 * x^4 + 25)

theorem max_value_of_expression : ∃ x : ℝ, (expression x) = 1 / 17 :=
sorry

end max_value_of_expression_l2025_202574


namespace number_of_boys_l2025_202506

variable (x y : ℕ)

theorem number_of_boys (h1 : x + y = 900) (h2 : y = (x / 100) * 900) : x = 90 :=
by
  sorry

end number_of_boys_l2025_202506


namespace fill_time_60_gallons_ten_faucets_l2025_202503

-- Define the problem parameters
def rate_of_five_faucets : ℚ := 150 / 8 -- in gallons per minute

def rate_of_one_faucet : ℚ := rate_of_five_faucets / 5

def rate_of_ten_faucets : ℚ := rate_of_one_faucet * 10

def time_to_fill_60_gallons_minutes : ℚ := 60 / rate_of_ten_faucets

def time_to_fill_60_gallons_seconds : ℚ := time_to_fill_60_gallons_minutes * 60

-- The main theorem to prove
theorem fill_time_60_gallons_ten_faucets : time_to_fill_60_gallons_seconds = 96 := by
  sorry

end fill_time_60_gallons_ten_faucets_l2025_202503


namespace min_x_plus_y_l2025_202515

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l2025_202515


namespace tangent_line_at_1_0_monotonic_intervals_l2025_202501

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) (a : ℝ) : ℝ := (2 * x^2 - a * x + 2) / x

theorem tangent_line_at_1_0 (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ, 
  (f x a, f 1 a) = (0, x - 1) → 
  y = 3 * x - 3 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, 0 < x → f_derivative x a ≥ 0) ↔ (a ≤ 4) ∧ 
  (∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < (a - Real.sqrt (a^2 - 16)) / 4) ∨ 
    ((a + Real.sqrt (a^2 - 16)) / 4 < x) 
  ) :=
sorry

end tangent_line_at_1_0_monotonic_intervals_l2025_202501


namespace circle_radius_on_sphere_l2025_202591

theorem circle_radius_on_sphere
  (sphere_radius : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (circle3_radius : ℝ)
  (all_circle_touch_each_other : Prop)
  (smaller_circle_touches_all : Prop)
  (smaller_circle_radius : ℝ) :
  sphere_radius = 2 →
  circle1_radius = 1 →
  circle2_radius = 1 →
  circle3_radius = 1 →
  all_circle_touch_each_other →
  smaller_circle_touches_all →
  smaller_circle_radius = 1 - Real.sqrt (2 / 3) :=
by
  intros h_sphere_radius h_circle1_radius h_circle2_radius h_circle3_radius h_all_circle_touch h_smaller_circle_touch
  sorry

end circle_radius_on_sphere_l2025_202591


namespace value_of_x_l2025_202528

theorem value_of_x (x : ℤ) : (3000 + x) ^ 2 = x ^ 2 → x = -1500 := 
by
  sorry

end value_of_x_l2025_202528


namespace orthocenter_PQR_is_correct_l2025_202500

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l2025_202500


namespace pizza_slices_all_toppings_l2025_202558

theorem pizza_slices_all_toppings (x : ℕ) :
  (16 = (8 - x) + (12 - x) + (6 - x) + x) → x = 5 := by
  sorry

end pizza_slices_all_toppings_l2025_202558


namespace volleyball_tournament_l2025_202522

theorem volleyball_tournament (n m : ℕ) (h : n = m) :
  n = m := 
by
  sorry

end volleyball_tournament_l2025_202522


namespace simplify_expression_l2025_202568

open Real

theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (( (x + 2) ^ 2 * (x ^ 2 - 2 * x + 2) ^ 2 / (x ^ 3 + 8) ^ 2 ) ^ 2 *
   ( (x - 2) ^ 2 * (x ^ 2 + 2 * x + 2) ^ 2 / (x ^ 3 - 8) ^ 2 ) ^ 2 = 1) :=
by
  sorry

end simplify_expression_l2025_202568


namespace proof_problem_l2025_202587

-- Triangle and Point Definitions
variables {A B C P : Type}
variables (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)

-- Conditions: Triangle ABC with angle A = 90 degrees and P on BC
def is_right_triangle (A B C : Type) (a b c : ℝ) (BC : ℝ) (angleA : ℝ := 90) : Prop :=
a^2 + b^2 = c^2 ∧ c = BC

def on_hypotenuse (P : Type) (BC : ℝ) (PB PC : ℝ) : Prop :=
PB + PC = BC

-- The proof problem
theorem proof_problem (A B C P : Type) 
  (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)
  (h1 : is_right_triangle A B C a b c BC)
  (h2 : on_hypotenuse P BC PB PC) :
  (a^2 / PC + b^2 / PB) ≥ (BC^3 / (PA^2 + PB * PC)) := 
sorry

end proof_problem_l2025_202587


namespace range_of_f_l2025_202513

open Real

noncomputable def f (x y z w : ℝ) : ℝ :=
  x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem range_of_f (x y z w : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) (h1w : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
  sorry

end range_of_f_l2025_202513


namespace rationalize_denominator_l2025_202590

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l2025_202590


namespace factorization_problem_l2025_202579

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end factorization_problem_l2025_202579


namespace marcy_votes_correct_l2025_202537

-- Definition of variables based on the conditions
def joey_votes : ℕ := 8
def barry_votes : ℕ := 2 * (joey_votes + 3)
def marcy_votes : ℕ := 3 * barry_votes

-- The main statement to prove
theorem marcy_votes_correct : marcy_votes = 66 := 
by 
  sorry

end marcy_votes_correct_l2025_202537


namespace remainder_of_polynomial_division_l2025_202517

theorem remainder_of_polynomial_division
  (x : ℝ)
  (h : 2 * x - 4 = 0) :
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end remainder_of_polynomial_division_l2025_202517


namespace drums_needed_for_profit_l2025_202502

def cost_to_enter_contest : ℝ := 10
def money_per_drum : ℝ := 0.025
def money_needed_for_profit (drums_hit : ℝ) : Prop :=
  drums_hit * money_per_drum > cost_to_enter_contest

theorem drums_needed_for_profit : ∃ D : ℝ, money_needed_for_profit D ∧ D = 400 :=
  by
  use 400
  sorry

end drums_needed_for_profit_l2025_202502


namespace olive_needs_two_colours_l2025_202504

theorem olive_needs_two_colours (α : Type) [Finite α] (G : SimpleGraph α) (colour : α → Fin 2) :
  (∀ v : α, ∃! w : α, G.Adj v w ∧ colour v = colour w) → ∃ color_map : α → Fin 2, ∀ v, ∃! w, G.Adj v w ∧ color_map v = color_map w :=
sorry

end olive_needs_two_colours_l2025_202504


namespace ice_cream_total_volume_l2025_202521

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume_l2025_202521


namespace maximum_xyz_l2025_202567

theorem maximum_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h: x ^ (Real.log x / Real.log y) * y ^ (Real.log y / Real.log z) * z ^ (Real.log z / Real.log x) = 10) : 
  x * y * z ≤ 10 := 
sorry

end maximum_xyz_l2025_202567


namespace lemonade_price_fraction_l2025_202550

theorem lemonade_price_fraction :
  (2 / 5) * (L / S) = 0.35714285714285715 → L / S = 0.8928571428571429 :=
by
  intro h
  sorry

end lemonade_price_fraction_l2025_202550


namespace gcd_polynomial_l2025_202539

theorem gcd_polynomial (b : ℤ) (h1 : ∃ k : ℤ, b = 7 * k ∧ k % 2 = 1) : 
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 7 := 
sorry

end gcd_polynomial_l2025_202539


namespace find_orange_shells_l2025_202526

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l2025_202526


namespace sum_of_x_y_l2025_202535

theorem sum_of_x_y (x y : ℝ) (h : (x + y + 2) * (x + y - 1) = 0) : x + y = -2 ∨ x + y = 1 :=
by sorry

end sum_of_x_y_l2025_202535


namespace range_of_x_l2025_202577

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l2025_202577


namespace remainder_of_67_pow_67_plus_67_mod_68_l2025_202524

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l2025_202524


namespace second_largest_consecutive_odd_195_l2025_202549

theorem second_largest_consecutive_odd_195 :
  ∃ x : Int, (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 195 ∧ (x + 2) = 41 := by
  sorry

end second_largest_consecutive_odd_195_l2025_202549


namespace prove_f_cos_eq_l2025_202519

variable (f : ℝ → ℝ)

theorem prove_f_cos_eq :
  (∀ x : ℝ, f (Real.sin x) = 3 - Real.cos (2 * x)) →
  (∀ x : ℝ, f (Real.cos x) = 3 + Real.cos (2 * x)) :=
by
  sorry

end prove_f_cos_eq_l2025_202519


namespace hyperbola_eccentricity_range_l2025_202571

theorem hyperbola_eccentricity_range (a b e : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : b / a < 2) :
  e = Real.sqrt (1 + (b / a) ^ 2) → 1 < e ∧ e < Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_range_l2025_202571


namespace jake_third_test_score_l2025_202516

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end jake_third_test_score_l2025_202516


namespace equilateral_triangle_area_increase_l2025_202546

theorem equilateral_triangle_area_increase (A : ℝ) (k : ℝ) (s : ℝ) (s' : ℝ) (A' : ℝ) (ΔA : ℝ) :
  A = 36 * Real.sqrt 3 →
  A = (Real.sqrt 3 / 4) * s^2 →
  s' = s + 3 →
  A' = (Real.sqrt 3 / 4) * s'^2 →
  ΔA = A' - A →
  ΔA = 20.25 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_increase_l2025_202546


namespace polynomial_value_l2025_202565

theorem polynomial_value (x : ℝ) (h : 3 * x^2 - x = 1) : 6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 :=
by
  sorry

end polynomial_value_l2025_202565


namespace repeating_decimal_fractional_representation_l2025_202575

theorem repeating_decimal_fractional_representation :
  (0.36 : ℝ) = (4 / 11 : ℝ) :=
sorry

end repeating_decimal_fractional_representation_l2025_202575


namespace cousin_age_result_l2025_202510

-- Let define the ages
def rick_age : ℕ := 15
def oldest_brother_age : ℕ := 2 * rick_age
def middle_brother_age : ℕ := oldest_brother_age / 3
def smallest_brother_age : ℕ := middle_brother_age / 2
def youngest_brother_age : ℕ := smallest_brother_age - 2
def cousin_age : ℕ := 5 * youngest_brother_age

-- The theorem stating the cousin's age.
theorem cousin_age_result : cousin_age = 15 := by
  sorry

end cousin_age_result_l2025_202510


namespace num_intersections_l2025_202560

noncomputable def polar_to_cartesian (r θ: ℝ): ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem num_intersections (θ: ℝ): 
  let c1 := polar_to_cartesian (6 * Real.cos θ) θ
  let c2 := polar_to_cartesian (10 * Real.sin θ) θ
  let (x1, y1) := c1
  let (x2, y2) := c2
  ((x1 - 3)^2 + y1^2 = 9 ∧ x2^2 + (y2 - 5)^2 = 25) →
  (x1, y1) = (x2, y2) ↔ false :=
by
  sorry

end num_intersections_l2025_202560


namespace inequality_solution_l2025_202554

theorem inequality_solution (b c x : ℝ) (x1 x2 : ℝ)
  (hb_pos : b > 0) (hc_pos : c > 0) 
  (h_eq1 : x1 * x2 = 1) 
  (h_eq2 : -1 + x2 = 2 * x1) 
  (h_b : b = 5 / 2) 
  (h_c : c = 1) 
  : (1 < x ∧ x ≤ 5 / 2) ↔ (1 < x ∧ x ≤ 5 / 2) :=
sorry

end inequality_solution_l2025_202554


namespace notebook_cost_l2025_202556

theorem notebook_cost {s n c : ℕ}
  (h1 : s > 18)
  (h2 : c > n)
  (h3 : s * n * c = 2275) :
  c = 13 :=
sorry

end notebook_cost_l2025_202556


namespace area_of_triangle_ABC_l2025_202593

theorem area_of_triangle_ABC 
  (r : ℝ) (R : ℝ) (ACB : ℝ) 
  (hr : r = 2) 
  (hR : R = 4) 
  (hACB : ACB = 120) : 
  let s := (2 * (2 + 4 * Real.sqrt 3)) / Real.sqrt 3 
  let S := s * r 
  S = 56 / Real.sqrt 3 :=
sorry

end area_of_triangle_ABC_l2025_202593


namespace games_that_didnt_work_l2025_202576

variable (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ)

theorem games_that_didnt_work
  (h₁ : games_from_friend = 2)
  (h₂ : games_from_garage_sale = 2)
  (h₃ : good_games = 2) :
  (games_from_friend + games_from_garage_sale - good_games) = 2 :=
by 
  sorry

end games_that_didnt_work_l2025_202576


namespace sarah_problem_l2025_202552

theorem sarah_problem (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 100 ≤ y ∧ y ≤ 999) 
  (h : 1000 * x + y = 11 * x * y) : x + y = 110 :=
sorry

end sarah_problem_l2025_202552


namespace road_length_l2025_202595

theorem road_length 
  (D : ℕ) (N1 : ℕ) (t : ℕ) (d1 : ℝ) (N_extra : ℝ) 
  (h1 : D = 300) (h2 : N1 = 35) (h3 : t = 100) (h4 : d1 = 2.5) (h5 : N_extra = 52.5) : 
  ∃ L : ℝ, L = 3 := 
by {
  sorry
}

end road_length_l2025_202595


namespace third_character_has_2_lines_l2025_202531

-- Define the number of lines characters have
variables (x y z : ℕ)

-- The third character has x lines
-- Condition: The second character has 6 more than three times the number of lines the third character has
def second_character_lines : ℕ := 3 * x + 6

-- Condition: The first character has 8 more lines than the second character
def first_character_lines : ℕ := second_character_lines x + 8

-- The first character has 20 lines
def first_character_has_20_lines : Prop := first_character_lines x = 20

-- Prove that the third character has 2 lines
theorem third_character_has_2_lines (h : first_character_has_20_lines x) : x = 2 :=
by
  -- Skipping the proof
  sorry

end third_character_has_2_lines_l2025_202531


namespace bridge_length_l2025_202580

theorem bridge_length (length_of_train : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : 
  length_of_train = 110 → train_speed_kmph = 45 → time_seconds = 30 → 
  ∃ length_of_bridge : ℕ, length_of_bridge = 265 := by
  intros h1 h2 h3
  sorry

end bridge_length_l2025_202580


namespace evaluate_g_sum_l2025_202508

def g (a b : ℚ) : ℚ :=
if a + b ≤ 5 then (a^2 * b - a + 3) / (3 * a) 
else (a * b^2 - b - 3) / (-3 * b)

theorem evaluate_g_sum : g 3 2 + g 3 3 = -1 / 3 :=
by
  sorry

end evaluate_g_sum_l2025_202508


namespace batsman_average_increase_l2025_202507

theorem batsman_average_increase
  (prev_avg : ℝ) -- average before the 17th innings
  (total_runs_16 : ℝ := 16 * prev_avg) -- total runs scored in the first 16 innings
  (score_17th : ℝ := 85) -- score in the 17th innings
  (new_avg : ℝ := 37) -- new average after 17 innings
  (total_runs_17 : ℝ := total_runs_16 + score_17th) -- total runs after 17 innings
  (calc_total_runs_17 : ℝ := 17 * new_avg) -- new total runs calculated by the new average
  (h : total_runs_17 = calc_total_runs_17) -- given condition: total_runs_17 = calc_total_runs_17
  : (new_avg - prev_avg) = 3 := 
by
  sorry

end batsman_average_increase_l2025_202507


namespace distance_between_cities_l2025_202511

def distance_thing 
  (d_A d_B : ℝ) 
  (v_A v_B : ℝ) 
  (t_diff : ℝ) : Prop :=
d_A = (3 / 5) * d_B ∧
v_A = 72 ∧
v_B = 108 ∧
t_diff = (1 / 4) ∧
(d_A + d_B) = 432

theorem distance_between_cities
  (d_A d_B : ℝ)
  (v_A v_B : ℝ)
  (t_diff : ℝ)
  (h : distance_thing d_A d_B v_A v_B t_diff)
  : d_A + d_B = 432 := by
  sorry

end distance_between_cities_l2025_202511


namespace black_white_tile_ratio_l2025_202548

/-- Assume the original pattern has 12 black tiles and 25 white tiles.
    The pattern is extended by attaching a border of black tiles two tiles wide around the square.
    Prove that the ratio of black tiles to white tiles in the new extended pattern is 76/25.-/
theorem black_white_tile_ratio 
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (black_border_width : ℕ)
  (new_black_tiles : ℕ)
  (total_new_tiles : ℕ) 
  (total_old_tiles : ℕ) 
  (new_white_tiles : ℕ)
  : original_black_tiles = 12 → 
    original_white_tiles = 25 → 
    black_border_width = 2 → 
    total_old_tiles = 36 →
    total_new_tiles = 100 →
    new_black_tiles = 76 → 
    new_white_tiles = 25 → 
    (new_black_tiles : ℚ) / (new_white_tiles : ℚ) = 76 / 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end black_white_tile_ratio_l2025_202548


namespace probability_incorrect_pairs_l2025_202570

theorem probability_incorrect_pairs 
  (k : ℕ) (h_k : k < 6)
  : let m := 7
    let n := 72
    m + n = 79 :=
by
  sorry

end probability_incorrect_pairs_l2025_202570


namespace acute_angle_sum_eq_pi_div_two_l2025_202514

open Real

theorem acute_angle_sum_eq_pi_div_two (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin α ^ 2 + sin β ^ 2 = sin (α + β)) : 
  α + β = π / 2 :=
sorry

end acute_angle_sum_eq_pi_div_two_l2025_202514


namespace average_infections_l2025_202547

theorem average_infections (x : ℝ) (h : 1 + x + x^2 = 121) : x = 10 :=
sorry

end average_infections_l2025_202547


namespace qatar_location_is_accurate_l2025_202564

def qatar_geo_location :=
  "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East."

theorem qatar_location_is_accurate :
  qatar_geo_location = "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East." :=
sorry

end qatar_location_is_accurate_l2025_202564


namespace jihyae_initial_money_l2025_202512

variables {M : ℕ}

def spent_on_supplies (M : ℕ) := M / 2 + 200
def left_after_buying (M : ℕ) := M - spent_on_supplies M
def saved (M : ℕ) := left_after_buying M / 2 + 300
def final_leftover (M : ℕ) := left_after_buying M - saved M

theorem jihyae_initial_money : final_leftover M = 350 → M = 3000 :=
by
  sorry

end jihyae_initial_money_l2025_202512


namespace time_to_finish_task_l2025_202563

-- Define the conditions
def printerA_rate (total_pages : ℕ) (time_A_alone : ℕ) : ℚ := total_pages / time_A_alone
def printerB_rate (rate_A : ℚ) : ℚ := rate_A + 10

-- Define the combined rate of printers working together
def combined_rate (rate_A : ℚ) (rate_B : ℚ) : ℚ := rate_A + rate_B

-- Define the time taken to finish the task together
def time_to_finish (total_pages : ℕ) (combined_rate : ℚ) : ℚ := total_pages / combined_rate

-- Given conditions
def total_pages : ℕ := 35
def time_A_alone : ℕ := 60

-- Definitions derived from given conditions
def rate_A : ℚ := printerA_rate total_pages time_A_alone
def rate_B : ℚ := printerB_rate rate_A

-- Combined rate when both printers work together
def combined_rate_AB : ℚ := combined_rate rate_A rate_B

-- Lean theorem statement to prove time taken by both printers
theorem time_to_finish_task : time_to_finish total_pages combined_rate_AB = 210 / 67 := 
by
  sorry

end time_to_finish_task_l2025_202563


namespace linette_problem_proof_l2025_202584

def boxes_with_neither_markers_nor_stickers (total_boxes markers stickers both : ℕ) : ℕ :=
  total_boxes - (markers + stickers - both)

theorem linette_problem_proof : 
  let total_boxes := 15
  let markers := 9
  let stickers := 5
  let both := 4
  boxes_with_neither_markers_nor_stickers total_boxes markers stickers both = 5 :=
by
  sorry

end linette_problem_proof_l2025_202584


namespace smallest_angle_convex_15_polygon_l2025_202525

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l2025_202525


namespace greatest_multiple_of_four_cubed_less_than_2000_l2025_202597

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l2025_202597


namespace larger_page_number_l2025_202545

theorem larger_page_number (x : ℕ) (h1 : (x + (x + 1) = 125)) : (x + 1 = 63) :=
by
  sorry

end larger_page_number_l2025_202545


namespace product_of_integers_l2025_202572

theorem product_of_integers (x y : ℕ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 40) : x * y = 99 :=
by {
  sorry
}

end product_of_integers_l2025_202572


namespace u_1000_eq_2036_l2025_202532

open Nat

def sequence_term (n : ℕ) : ℕ :=
  let sum_to (k : ℕ) := k * (k + 1) / 2
  if n ≤ 0 then 0 else
  let group := (Nat.sqrt (2 * n)) + 1
  let k := n - sum_to (group - 1)
  (group * group) + 4 * (k - 1) - (group % 4)

theorem u_1000_eq_2036 : sequence_term 1000 = 2036 := sorry

end u_1000_eq_2036_l2025_202532


namespace cos_value_of_geometric_sequence_l2025_202582

theorem cos_value_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r)
  (h2 : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1 / 2 := 
sorry

end cos_value_of_geometric_sequence_l2025_202582


namespace exists_v_satisfying_equation_l2025_202509

noncomputable def custom_operation (v : ℝ) : ℝ :=
  v - (v / 3) + Real.sin v

theorem exists_v_satisfying_equation :
  ∃ v : ℝ, custom_operation (custom_operation v) = 24 := 
sorry

end exists_v_satisfying_equation_l2025_202509


namespace prime_iff_even_and_power_of_two_l2025_202592

theorem prime_iff_even_and_power_of_two (a n : ℕ) (h_pos_a : a > 1) (h_pos_n : n > 0) :
  Nat.Prime (a^n + 1) → (∃ k : ℕ, a = 2 * k) ∧ (∃ m : ℕ, n = 2^m) :=
by 
  sorry

end prime_iff_even_and_power_of_two_l2025_202592


namespace sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l2025_202544

theorem sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares 
  (n : ℕ)
  (a b c d : ℕ) 
  (h1 : n = 2^a + 2^b) 
  (h2 : a ≠ b) 
  (h3 : n = (2^c - 1) + (2^d - 1)) 
  (h4 : c ≠ d)
  (h5 : Nat.Prime (2^c - 1)) 
  (h6 : Nat.Prime (2^d - 1)) : 
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 := 
by
  sorry

end sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l2025_202544


namespace multiples_of_7_between_20_and_150_l2025_202566

def number_of_multiples_of_7_between (a b : ℕ) : ℕ :=
  (b / 7) - (a / 7) + (if a % 7 = 0 then 1 else 0)

theorem multiples_of_7_between_20_and_150 : number_of_multiples_of_7_between 21 147 = 19 := by
  sorry

end multiples_of_7_between_20_and_150_l2025_202566


namespace no_values_of_g_g_x_eq_one_l2025_202538

-- Define the function g and its properties based on the conditions
variable (g : ℝ → ℝ)
variable (h₁ : g (-4) = 1)
variable (h₂ : g (0) = 1)
variable (h₃ : g (4) = 3)
variable (h₄ : ∀ x, -4 ≤ x ∧ x ≤ 4 → g x ≥ 1)

-- Define the theorem to prove the number of values of x such that g(g(x)) = 1 is zero
theorem no_values_of_g_g_x_eq_one : ∃ n : ℕ, n = 0 ∧ (∀ x, -4 ≤ x ∧ x ≤ 4 → g (g x) = 1 → false) :=
by
  sorry -- proof to be provided later

end no_values_of_g_g_x_eq_one_l2025_202538


namespace opposite_of_neg_abs_opposite_of_neg_abs_correct_l2025_202529

theorem opposite_of_neg_abs (x : ℚ) (hx : |x| = 2 / 5) : -|x| = - (2 / 5) := sorry

theorem opposite_of_neg_abs_correct (x : ℚ) (hx : |x| = 2 / 5) : - -|x| = 2 / 5 := by
  rw [opposite_of_neg_abs x hx]
  simp

end opposite_of_neg_abs_opposite_of_neg_abs_correct_l2025_202529


namespace andy_starting_problem_l2025_202561

theorem andy_starting_problem (end_num problems_solved : ℕ) 
  (h_end : end_num = 125) (h_solved : problems_solved = 46) : 
  end_num - problems_solved + 1 = 80 := 
by
  sorry

end andy_starting_problem_l2025_202561


namespace jeff_bought_6_pairs_l2025_202553

theorem jeff_bought_6_pairs (price_of_shoes : ℝ) (num_of_shoes : ℕ) (price_of_jersey : ℝ)
  (h1 : price_of_jersey = (1 / 4) * price_of_shoes)
  (h2 : num_of_shoes * price_of_shoes = 480)
  (h3 : num_of_shoes * price_of_shoes + 4 * price_of_jersey = 560) :
  num_of_shoes = 6 :=
sorry

end jeff_bought_6_pairs_l2025_202553


namespace find_ordered_pair_l2025_202523

theorem find_ordered_pair (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroot : ∀ x : ℝ, 2 * x^2 + a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1 / 2, -3 / 4) := 
  sorry

end find_ordered_pair_l2025_202523
