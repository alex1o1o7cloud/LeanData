import Mathlib

namespace min_trucks_for_crates_l277_27719

noncomputable def min_trucks (total_weight : ℕ) (max_weight_per_crate : ℕ) (truck_capacity : ℕ) : ℕ :=
  if total_weight % truck_capacity = 0 then total_weight / truck_capacity
  else total_weight / truck_capacity + 1

theorem min_trucks_for_crates :
  ∀ (total_weight max_weight_per_crate truck_capacity : ℕ),
    total_weight = 10 →
    max_weight_per_crate = 1 →
    truck_capacity = 3 →
    min_trucks total_weight max_weight_per_crate truck_capacity = 5 :=
by
  intros total_weight max_weight_per_crate truck_capacity h_total h_max h_truck
  rw [h_total, h_max, h_truck]
  sorry

end min_trucks_for_crates_l277_27719


namespace smallest_t_for_temperature_104_l277_27741

theorem smallest_t_for_temperature_104 : 
  ∃ t : ℝ, (-t^2 + 16*t + 40 = 104) ∧ (t > 0) ∧ (∀ s : ℝ, (-s^2 + 16*s + 40 = 104) ∧ (s > 0) → t ≤ s) :=
sorry

end smallest_t_for_temperature_104_l277_27741


namespace simplify_fraction_complex_l277_27785

open Complex

theorem simplify_fraction_complex :
  (3 - I) / (2 + 5 * I) = (1 / 29) - (17 / 29) * I := by
  sorry

end simplify_fraction_complex_l277_27785


namespace total_cost_production_l277_27750

-- Define the fixed cost and marginal cost per product as constants
def fixedCost : ℤ := 12000
def marginalCostPerProduct : ℤ := 200
def numberOfProducts : ℤ := 20

-- Define the total cost as the sum of fixed cost and total variable cost
def totalCost : ℤ := fixedCost + (marginalCostPerProduct * numberOfProducts)

-- Prove that the total cost is equal to 16000
theorem total_cost_production : totalCost = 16000 :=
by
  sorry

end total_cost_production_l277_27750


namespace zoe_total_cost_correct_l277_27720

theorem zoe_total_cost_correct :
  (6 * 0.5) + (6 * (1 + 2 * 0.75)) + (6 * 2 * 3) = 54 :=
by
  sorry

end zoe_total_cost_correct_l277_27720


namespace matrix_mult_correct_l277_27773

-- Definition of matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![7, -3],
  ![2, 4]
]

-- The goal is to prove that A * B yields the matrix C
def matrix_product : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![23, -5],
  ![24, -20]
]

theorem matrix_mult_correct : A * B = matrix_product := by
  -- Proof omitted
  sorry

end matrix_mult_correct_l277_27773


namespace vacation_cost_l277_27782

theorem vacation_cost (C : ℝ) (h : C / 3 - C / 4 = 60) : C = 720 := 
by sorry

end vacation_cost_l277_27782


namespace find_salary_June_l277_27713

variable (J F M A May_s June_s : ℝ)
variable (h1 : J + F + M + A = 4 * 8000)
variable (h2 : F + M + A + May_s = 4 * 8450)
variable (h3 : May_s = 6500)
variable (h4 : M + A + May_s + June_s = 4 * 9000)
variable (h5 : June_s = 1.2 * May_s)

theorem find_salary_June : June_s = 7800 := by
  sorry

end find_salary_June_l277_27713


namespace find_e_l277_27793

theorem find_e (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
    (h_lb1 : a + b = 32) (h_lb2 : a + c = 36) (h_lb3 : b + c = 37)
    (h_ub1 : c + e = 48) (h_ub2 : d + e = 51) : e = 27.5 :=
sorry

end find_e_l277_27793


namespace function_passes_through_one_one_l277_27742

noncomputable def f (a x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_through_one_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 := 
by
  sorry

end function_passes_through_one_one_l277_27742


namespace tan_theta_half_l277_27759

open Real

theorem tan_theta_half (θ : ℝ) 
  (h0 : 0 < θ) 
  (h1 : θ < π / 2) 
  (h2 : ∃ k : ℝ, (sin (2 * θ), cos θ) = k • (cos θ, 1)) : 
  tan θ = 1 / 2 := by 
sorry

end tan_theta_half_l277_27759


namespace solve_for_x_l277_27772

theorem solve_for_x (x : ℝ) (h₁ : 3 * x^2 - 9 * x = 0) (h₂ : x ≠ 0) : x = 3 := 
by {
  sorry
}

end solve_for_x_l277_27772


namespace chris_babysitting_hours_l277_27788

theorem chris_babysitting_hours (h : ℕ) (video_game_cost candy_cost earn_per_hour leftover total_cost : ℕ) :
  video_game_cost = 60 ∧
  candy_cost = 5 ∧
  earn_per_hour = 8 ∧
  leftover = 7 ∧
  total_cost = video_game_cost + candy_cost ∧
  earn_per_hour * h = total_cost + leftover
  → h = 9 := by
  intros
  sorry

end chris_babysitting_hours_l277_27788


namespace binom_7_4_plus_5_l277_27794

theorem binom_7_4_plus_5 : ((Nat.choose 7 4) + 5) = 40 := by
  sorry

end binom_7_4_plus_5_l277_27794


namespace cost_price_of_book_l277_27740

theorem cost_price_of_book
  (SP : Real)
  (profit_percentage : Real)
  (h1 : SP = 300)
  (h2 : profit_percentage = 0.20) :
  ∃ CP : Real, CP = 250 :=
by
  -- Proof of the statement
  sorry

end cost_price_of_book_l277_27740


namespace lcm_gcd_eq_product_l277_27706

theorem lcm_gcd_eq_product {a b : ℕ} (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 :=
  sorry

end lcm_gcd_eq_product_l277_27706


namespace infinite_solutions_c_l277_27743

theorem infinite_solutions_c (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 :=
sorry

end infinite_solutions_c_l277_27743


namespace stephen_total_distance_l277_27781

theorem stephen_total_distance :
  let mountain_height := 40000
  let ascent_fraction := 3 / 4
  let descent_fraction := 2 / 3
  let extra_distance_fraction := 0.10
  let normal_trips := 8
  let harsh_trips := 2
  let ascent_distance := ascent_fraction * mountain_height
  let descent_distance := descent_fraction * ascent_distance
  let normal_trip_distance := ascent_distance + descent_distance
  let harsh_trip_extra_distance := extra_distance_fraction * ascent_distance
  let harsh_trip_distance := ascent_distance + harsh_trip_extra_distance + descent_distance
  let total_normal_distance := normal_trip_distance * normal_trips
  let total_harsh_distance := harsh_trip_distance * harsh_trips
  let total_distance := total_normal_distance + total_harsh_distance
  total_distance = 506000 :=
by
  sorry

end stephen_total_distance_l277_27781


namespace stephanie_oranges_l277_27753

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l277_27753


namespace julia_tuesday_l277_27710

variable (M : ℕ) -- The number of kids Julia played with on Monday
variable (T : ℕ) -- The number of kids Julia played with on Tuesday

-- Conditions
def condition1 : Prop := M = T + 8
def condition2 : Prop := M = 22

-- Theorem to prove
theorem julia_tuesday : condition1 M T → condition2 M → T = 14 := by
  sorry

end julia_tuesday_l277_27710


namespace expression_value_l277_27729

theorem expression_value (x : ℕ) (h : x = 12) : (3 / 2 * x - 3 : ℚ) = 15 := by
  rw [h]
  norm_num
-- sorry to skip the proof if necessary
-- sorry 

end expression_value_l277_27729


namespace arina_should_accept_anton_offer_l277_27708

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end arina_should_accept_anton_offer_l277_27708


namespace steps_per_level_l277_27774

def number_of_steps_per_level (blocks_per_step total_blocks total_levels : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / total_levels

theorem steps_per_level (blocks_per_step : ℕ) (total_blocks : ℕ) (total_levels : ℕ) (h1 : blocks_per_step = 3) (h2 : total_blocks = 96) (h3 : total_levels = 4) :
  number_of_steps_per_level blocks_per_step total_blocks total_levels = 8 := 
by
  sorry

end steps_per_level_l277_27774


namespace number_of_tiles_per_row_l277_27757

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l277_27757


namespace carol_total_peanuts_l277_27790

-- Conditions as definitions
def carol_initial_peanuts : Nat := 2
def carol_father_peanuts : Nat := 5

-- Theorem stating that the total number of peanuts Carol has is 7
theorem carol_total_peanuts : carol_initial_peanuts + carol_father_peanuts = 7 := by
  -- Proof would go here, but we use sorry to skip
  sorry

end carol_total_peanuts_l277_27790


namespace even_blue_faces_cubes_correct_l277_27783

/-- A rectangular wooden block is 6 inches long, 3 inches wide, and 2 inches high.
    The block is painted blue on all six sides and then cut into 1 inch cubes.
    This function determines the number of 1-inch cubes that have a total number
    of blue faces that is an even number (in this case, 2 blue faces). -/
def count_even_blue_faces_cubes : Nat :=
  let length := 6
  let width := 3
  let height := 2
  let total_cubes := length * width * height
  
  -- Calculate corner cubes
  let corners := 8

  -- Calculate edges but not corners cubes
  let edge_not_corners := 
    (4 * (length - 2)) + 
    (4 * (width - 2)) + 
    (4 * (height - 2))

  -- Calculate even number of blue faces cubes 
  let even_number_blue_faces := edge_not_corners

  even_number_blue_faces

theorem even_blue_faces_cubes_correct : count_even_blue_faces_cubes = 20 := by
  -- Place your proof here.
  sorry

end even_blue_faces_cubes_correct_l277_27783


namespace equation_one_solution_equation_two_no_solution_l277_27725

-- Problem 1
theorem equation_one_solution (x : ℝ) (h : x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) : x = 0 := 
by 
  sorry

-- Problem 2
theorem equation_two_no_solution (x : ℝ) (h : 2 * x + 9 / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2) : False := 
by 
  sorry

end equation_one_solution_equation_two_no_solution_l277_27725


namespace castor_chess_players_l277_27703

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end castor_chess_players_l277_27703


namespace kids_on_excursions_l277_27747

theorem kids_on_excursions (total_kids : ℕ) (one_fourth_kids_tubing two := total_kids / 4) (half_tubers_rafting : ℕ := one_fourth_kids_tubing / 2) :
  total_kids = 40 → one_fourth_kids_tubing = 10 → half_tubers_rafting = 5 :=
by
  intros
  sorry

end kids_on_excursions_l277_27747


namespace scientific_notation_150_billion_l277_27798

theorem scientific_notation_150_billion : 150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_150_billion_l277_27798


namespace value_of_fraction_l277_27748

theorem value_of_fraction (a b : ℚ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 :=
sorry

end value_of_fraction_l277_27748


namespace find_f_at_9_over_2_l277_27701

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l277_27701


namespace people_in_group_l277_27761

theorem people_in_group
  (N : ℕ)
  (h1 : ∃ w1 w2 : ℝ, w1 = 65 ∧ w2 = 71 ∧ w2 - w1 = 6)
  (h2 : ∃ avg_increase : ℝ, avg_increase = 1.5 ∧ 6 = avg_increase * N) :
  N = 4 :=
sorry

end people_in_group_l277_27761


namespace emily_initial_toys_l277_27728

theorem emily_initial_toys : ∃ (initial_toys : ℕ), initial_toys = 3 + 4 :=
by
  existsi 7
  sorry

end emily_initial_toys_l277_27728


namespace initial_games_l277_27732

def games_given_away : ℕ := 91
def games_left : ℕ := 92

theorem initial_games :
  games_given_away + games_left = 183 :=
by
  sorry

end initial_games_l277_27732


namespace crates_of_oranges_l277_27797

theorem crates_of_oranges (C : ℕ) (h1 : ∀ crate, crate = 150) (h2 : ∀ box, box = 30) (num_boxes : ℕ) (total_fruits : ℕ) : 
  num_boxes = 16 → total_fruits = 2280 → 150 * C + 16 * 30 = 2280 → C = 12 :=
by
  intros num_boxes_eq total_fruits_eq fruit_eq
  sorry

end crates_of_oranges_l277_27797


namespace find_a_values_l277_27717

theorem find_a_values (a n : ℕ) (h1 : 7 * a * n - 3 * n = 2020) :
    a = 68 ∨ a = 289 := sorry

end find_a_values_l277_27717


namespace problem_solution_l277_27762

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l277_27762


namespace spending_on_other_items_is_30_percent_l277_27712

-- Define the total amount Jill spent excluding taxes
variable (T : ℝ)

-- Define the amounts spent on clothing, food, and other items as percentages of T
def clothing_spending : ℝ := 0.50 * T
def food_spending : ℝ := 0.20 * T
def other_items_spending (x : ℝ) : ℝ := x * T

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0.0
def other_items_tax_rate : ℝ := 0.10

-- Define the taxes paid on each category
def clothing_tax : ℝ := clothing_tax_rate * clothing_spending T
def food_tax : ℝ := food_tax_rate * food_spending T
def other_items_tax (x : ℝ) : ℝ := other_items_tax_rate * other_items_spending T x

-- Define the total tax paid as a percentage of the total amount spent excluding taxes
def total_tax_paid : ℝ := 0.05 * T

-- The main theorem stating that the percentage of the amount spent on other items is 30%
theorem spending_on_other_items_is_30_percent (x : ℝ) (h : total_tax_paid T = clothing_tax T + other_items_tax T x) :
  x = 0.30 :=
sorry

end spending_on_other_items_is_30_percent_l277_27712


namespace geometric_progression_problem_l277_27776

open Real

theorem geometric_progression_problem
  (a b c r : ℝ)
  (h1 : a = 20)
  (h2 : b = 40)
  (h3 : c = 10)
  (h4 : b = r * a)
  (h5 : c = r * b) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end geometric_progression_problem_l277_27776


namespace child_tickets_sold_l277_27700

noncomputable def price_adult_ticket : ℝ := 7
noncomputable def price_child_ticket : ℝ := 4
noncomputable def total_tickets_sold : ℝ := 900
noncomputable def total_revenue : ℝ := 5100

theorem child_tickets_sold : ∃ (C : ℝ), price_child_ticket * C + price_adult_ticket * (total_tickets_sold - C) = total_revenue ∧ C = 400 :=
by
  sorry

end child_tickets_sold_l277_27700


namespace provisions_last_days_after_reinforcement_l277_27726

-- Definitions based on the conditions
def initial_men := 2000
def initial_days := 40
def reinforcement_men := 2000
def days_passed := 20

-- Calculate the total provisions initially
def total_provisions := initial_men * initial_days

-- Calculate the remaining provisions after some days passed
def remaining_provisions := total_provisions - (initial_men * days_passed)

-- Total number of men after reinforcement
def total_men := initial_men + reinforcement_men

-- The Lean statement proving the duration the remaining provisions will last
theorem provisions_last_days_after_reinforcement :
  remaining_provisions / total_men = 10 := by
  sorry

end provisions_last_days_after_reinforcement_l277_27726


namespace giraffe_statue_price_l277_27767

variable (G : ℕ) -- Price of a giraffe statue in dollars

-- Conditions as definitions in Lean 4
def giraffe_jade_usage := 120 -- grams
def elephant_jade_usage := 2 * giraffe_jade_usage -- 240 grams
def elephant_price := 350 -- dollars
def total_jade := 1920 -- grams
def additional_profit_with_elephants := 400 -- dollars

-- Prove that the price of a giraffe statue is $150
theorem giraffe_statue_price : 
  16 * G + additional_profit_with_elephants = 8 * elephant_price → G = 150 :=
by
  intro h
  sorry

end giraffe_statue_price_l277_27767


namespace intersection_of_A_and_B_l277_27770

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end intersection_of_A_and_B_l277_27770


namespace polynomial_divisibility_p_q_l277_27707

theorem polynomial_divisibility_p_q (p' q' : ℝ) :
  (∀ x, x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0 → (x = -1 ∨ x = 2)) →
  p' = 0 ∧ q' = -9 :=
by sorry

end polynomial_divisibility_p_q_l277_27707


namespace sum_series_eq_4_div_9_l277_27786

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l277_27786


namespace toy_cars_ratio_proof_l277_27777

theorem toy_cars_ratio_proof (toys_original : ℕ) (toys_bought_last_month : ℕ) (toys_total : ℕ) :
  toys_original = 25 ∧ toys_bought_last_month = 5 ∧ toys_total = 40 →
  (toys_total - toys_original - toys_bought_last_month) / toys_bought_last_month = 2 :=
by
  sorry

end toy_cars_ratio_proof_l277_27777


namespace f_D_not_mapping_to_B_l277_27738

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B := {y : ℝ | 1 ≤ y ∧ y <= 4}
def f_D (x : ℝ) := 4 - x^2

theorem f_D_not_mapping_to_B : ¬ (∀ x ∈ A, f_D x ∈ B) := sorry

end f_D_not_mapping_to_B_l277_27738


namespace urban_general_hospital_problem_l277_27737

theorem urban_general_hospital_problem
  (a b c d : ℕ)
  (h1 : b = 3 * c)
  (h2 : a = 2 * b)
  (h3 : d = c / 2)
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1500) :
  5 * d = 1500 / 11 := by
  sorry

end urban_general_hospital_problem_l277_27737


namespace remainder_abc_l277_27727

theorem remainder_abc (a b c : ℕ) 
  (h₀ : a < 9) (h₁ : b < 9) (h₂ : c < 9)
  (h₃ : (a + 3 * b + 2 * c) % 9 = 0)
  (h₄ : (2 * a + 2 * b + 3 * c) % 9 = 3)
  (h₅ : (3 * a + b + 2 * c) % 9 = 6) : 
  (a * b * c) % 9 = 0 := by
  sorry

end remainder_abc_l277_27727


namespace molecular_weight_of_one_mole_l277_27749

-- Definitions derived from the conditions in the problem:

def molecular_weight_nine_moles (w : ℕ) : ℕ :=
  2664

def molecular_weight_one_mole (w : ℕ) : ℕ :=
  w / 9

-- The theorem to prove, based on the above definitions and conditions:
theorem molecular_weight_of_one_mole (w : ℕ) (hw : molecular_weight_nine_moles w = 2664) :
  molecular_weight_one_mole w = 296 :=
sorry

end molecular_weight_of_one_mole_l277_27749


namespace other_solution_of_quadratic_l277_27746

theorem other_solution_of_quadratic (x : ℚ) 
  (hx1 : 77 * x^2 - 125 * x + 49 = 0) (hx2 : x = 8/11) : 
  77 * (1 : ℚ)^2 - 125 * (1 : ℚ) + 49 = 0 :=
by sorry

end other_solution_of_quadratic_l277_27746


namespace prob_less_than_9_is_correct_l277_27709

-- Define the probabilities
def prob_ring_10 := 0.24
def prob_ring_9 := 0.28
def prob_ring_8 := 0.19

-- Define the condition for scoring less than 9, which does not include hitting the 10 or 9 ring.
def prob_less_than_9 := 1 - prob_ring_10 - prob_ring_9

-- Now we state the theorem we want to prove.
theorem prob_less_than_9_is_correct : prob_less_than_9 = 0.48 :=
by {
  -- Proof would go here
  sorry
}

end prob_less_than_9_is_correct_l277_27709


namespace mrs_sheridan_cats_l277_27711

theorem mrs_sheridan_cats (initial_cats : ℝ) (given_away_cats : ℝ) (remaining_cats : ℝ) :
  initial_cats = 17.0 → given_away_cats = 14.0 → remaining_cats = (initial_cats - given_away_cats) → remaining_cats = 3.0 :=
by
  intros
  sorry

end mrs_sheridan_cats_l277_27711


namespace factorize1_factorize2_factorize3_factorize4_l277_27763

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end factorize1_factorize2_factorize3_factorize4_l277_27763


namespace minimum_knights_in_tournament_l277_27791

def knights_tournament : Prop :=
  ∃ (N : ℕ), (∀ (x : ℕ), x = N / 4 →
    ∃ (k : ℕ), k = (3 * x - 1) / 7 → N = 20)

theorem minimum_knights_in_tournament : knights_tournament :=
  sorry

end minimum_knights_in_tournament_l277_27791


namespace points_on_line_l277_27739

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l277_27739


namespace no_root_l277_27775

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l277_27775


namespace largest_square_with_five_interior_lattice_points_l277_27751

theorem largest_square_with_five_interior_lattice_points :
  ∃ (s : ℝ), (∀ (x y : ℤ), 1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → ((⌊s⌋ - 1)^2 = 5) ∧ s^2 = 18 := sorry

end largest_square_with_five_interior_lattice_points_l277_27751


namespace keychain_arrangement_l277_27754

theorem keychain_arrangement (house car locker office key5 key6 : ℕ) :
  (∃ (A B : ℕ), house = A ∧ car = A ∧ locker = B ∧ office = B) →
  (∃ (arrangements : ℕ), arrangements = 24) :=
by
  sorry

end keychain_arrangement_l277_27754


namespace false_propositions_l277_27769

theorem false_propositions (p q : Prop) (hnp : ¬ p) (hq : q) :
  (¬ p) ∧ (¬ (p ∧ q)) ∧ (¬ ¬ q) :=
by {
  exact ⟨hnp, not_and_of_not_left q hnp, not_not_intro hq⟩
}

end false_propositions_l277_27769


namespace intersection_M_N_l277_27724

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end intersection_M_N_l277_27724


namespace terminating_decimal_of_fraction_l277_27736

theorem terminating_decimal_of_fraction (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 624) : 
  (∃ m : ℕ, 10^m * (n / 625) = k) → ∃ m, m = 624 :=
sorry

end terminating_decimal_of_fraction_l277_27736


namespace find_a2_plus_b2_l277_27789

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) : a^2 + b^2 = 68 :=
sorry

end find_a2_plus_b2_l277_27789


namespace negation_proposition_l277_27780

variables {a b c : ℝ}

theorem negation_proposition (h : a ≤ b) : a + c ≤ b + c :=
sorry

end negation_proposition_l277_27780


namespace y_when_x_is_4_l277_27755

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l277_27755


namespace jerry_mowing_income_l277_27744

theorem jerry_mowing_income (M : ℕ) (week_spending : ℕ) (money_weed_eating : ℕ) (weeks : ℕ)
  (H1 : week_spending = 5)
  (H2 : money_weed_eating = 31)
  (H3 : weeks = 9)
  (H4 : (M + money_weed_eating) = week_spending * weeks)
  : M = 14 :=
by {
  sorry
}

end jerry_mowing_income_l277_27744


namespace part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l277_27764

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l277_27764


namespace round_robin_teams_l277_27768

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end round_robin_teams_l277_27768


namespace vertex_coordinates_l277_27734

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 8

-- State the theorem for the coordinates of the vertex
theorem vertex_coordinates : 
  (∃ h k : ℝ, ∀ x : ℝ, parabola x = 2 * (x - h)^2 + k) ∧ h = 1 ∧ k = 8 :=
sorry

end vertex_coordinates_l277_27734


namespace oldest_child_age_correct_l277_27723

-- Defining the conditions
def jane_start_age := 16
def jane_current_age := 32
def jane_stopped_babysitting_years_ago := 10
def half (x : ℕ) := x / 2

-- Expressing the conditions
def jane_last_babysitting_age := jane_current_age - jane_stopped_babysitting_years_ago
def max_child_age_when_jane_stopped := half jane_last_babysitting_age
def years_since_jane_stopped := jane_stopped_babysitting_years_ago

def calculate_oldest_child_current_age (age : ℕ) : ℕ :=
  age + years_since_jane_stopped

def child_age_when_stopped := max_child_age_when_jane_stopped
def expected_oldest_child_current_age := 21

-- The theorem stating the equivalence
theorem oldest_child_age_correct : 
  calculate_oldest_child_current_age child_age_when_stopped = expected_oldest_child_current_age :=
by
  -- Proof here
  sorry

end oldest_child_age_correct_l277_27723


namespace factor_theorem_l277_27722

noncomputable def Q (b x : ℝ) : ℝ := x^4 - 3 * x^3 + b * x^2 - 12 * x + 24

theorem factor_theorem (b : ℝ) : (∃ x : ℝ, x = -2) ∧ (Q b x = 0) → b = -22 :=
by
  sorry

end factor_theorem_l277_27722


namespace coterminal_angle_l277_27721

theorem coterminal_angle (theta : ℝ) (lower : ℝ) (upper : ℝ) (k : ℤ) : 
  -950 = k * 360 + theta ∧ (lower ≤ theta ∧ theta ≤ upper) → theta = 130 :=
by
  -- Given conditions
  sorry

end coterminal_angle_l277_27721


namespace twin_ages_l277_27733

theorem twin_ages (x : ℕ) (h : (x + 1) ^ 2 = x ^ 2 + 15) : x = 7 :=
sorry

end twin_ages_l277_27733


namespace smallest_value_in_interval_l277_27779

open Real

noncomputable def smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : Prop :=
  1 / x^2 < x ∧
  1 / x^2 < x^2 ∧
  1 / x^2 < 2 * x^2 ∧
  1 / x^2 < 3 * x ∧
  1 / x^2 < sqrt x ∧
  1 / x^2 < 1 / x

theorem smallest_value_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) : smallest_value x h :=
by
  sorry

end smallest_value_in_interval_l277_27779


namespace ab_leq_one_l277_27718

theorem ab_leq_one (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

end ab_leq_one_l277_27718


namespace expected_up_right_paths_l277_27758

def lattice_points := {p : ℕ × ℕ // p.1 ≤ 5 ∧ p.2 ≤ 5}

def total_paths : ℕ := Nat.choose 10 5

def calculate_paths (x y : ℕ) : ℕ :=
  if h : x ≤ 5 ∧ y ≤ 5 then
    let F := total_paths * 25
    F / 36
  else
    0

theorem expected_up_right_paths : ∃ S, S = 175 :=
  sorry

end expected_up_right_paths_l277_27758


namespace simplification_l277_27795

-- Define all relevant powers
def pow2_8 : ℤ := 2^8
def pow4_5 : ℤ := 4^5
def pow2_3 : ℤ := 2^3
def pow_neg2_2 : ℤ := (-2)^2

-- Define the expression inside the parentheses
def inner_expr : ℤ := pow2_3 - pow_neg2_2

-- Define the exponentiation of the inner expression
def inner_expr_pow11 : ℤ := inner_expr^11

-- Define the entire expression
def full_expr : ℤ := (pow2_8 + pow4_5) * inner_expr_pow11

-- State the proof goal
theorem simplification : full_expr = 5368709120 := by
  sorry

end simplification_l277_27795


namespace train_passes_jogger_in_approximately_25_8_seconds_l277_27735

noncomputable def jogger_speed_kmh := 7
noncomputable def train_speed_kmh := 60
noncomputable def jogger_head_start_m := 180
noncomputable def train_length_m := 200

noncomputable def kmh_to_ms (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_ms := kmh_to_ms jogger_speed_kmh
noncomputable def train_speed_ms := kmh_to_ms train_speed_kmh

noncomputable def relative_speed_ms := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m
noncomputable def time_to_pass_sec := total_distance_to_cover_m / (relative_speed_ms : ℝ) 

theorem train_passes_jogger_in_approximately_25_8_seconds :
  abs (time_to_pass_sec - 25.8) < 0.1 := sorry

end train_passes_jogger_in_approximately_25_8_seconds_l277_27735


namespace ratio_of_hair_lengths_l277_27704

theorem ratio_of_hair_lengths 
  (logan_hair : ℕ)
  (emily_hair : ℕ)
  (kate_hair : ℕ)
  (h1 : logan_hair = 20)
  (h2 : emily_hair = logan_hair + 6)
  (h3 : kate_hair = 7)
  : kate_hair / emily_hair = 7 / 26 :=
by sorry

end ratio_of_hair_lengths_l277_27704


namespace problem_statements_correctness_l277_27771

theorem problem_statements_correctness :
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (12 ∣ 72 ∧ 12 ∣ 120) ∧ (7 ∣ 49 ∧ 7 ∣ 84) ∧ (7 ∣ 63) → 
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (7 ∣ 63) :=
by
  intro h
  sorry

end problem_statements_correctness_l277_27771


namespace proof_x_plus_y_sum_l277_27745

noncomputable def x_and_y_sum (x y : ℝ) : Prop := 31.25 / x = 100 / 9.6 ∧ 13.75 / x = y / 9.6

theorem proof_x_plus_y_sum (x y : ℝ) (h : x_and_y_sum x y) : x + y = 47 :=
sorry

end proof_x_plus_y_sum_l277_27745


namespace cookies_per_person_l277_27730

theorem cookies_per_person (cookies_per_bag : ℕ) (bags : ℕ) (damaged_cookies_per_bag : ℕ) (people : ℕ) (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_each : ℕ) :
  (cookies_per_bag = 738) →
  (bags = 295) →
  (damaged_cookies_per_bag = 13) →
  (people = 125) →
  (total_cookies = cookies_per_bag * bags) →
  (remaining_cookies = total_cookies - (damaged_cookies_per_bag * bags)) →
  (cookies_each = remaining_cookies / people) →
  cookies_each = 1711 :=
by
  sorry 

end cookies_per_person_l277_27730


namespace find_p_over_q_at_neg1_l277_27714

noncomputable def p (x : ℝ) : ℝ := (-27 / 8) * x
noncomputable def q (x : ℝ) : ℝ := (x + 5) * (x - 1)

theorem find_p_over_q_at_neg1 : p (-1) / q (-1) = 27 / 64 := by
  -- Skipping the proof
  sorry

end find_p_over_q_at_neg1_l277_27714


namespace minimize_expression_l277_27715

theorem minimize_expression (a b c d : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 :=
by
  sorry

end minimize_expression_l277_27715


namespace speed_of_boat_in_still_water_l277_27752

-- Define the given conditions
def speed_of_stream : ℝ := 4  -- Speed of the stream in km/hr
def distance_downstream : ℝ := 60  -- Distance traveled downstream in km
def time_downstream : ℝ := 3  -- Time taken to travel downstream in hours

-- The statement we need to prove
theorem speed_of_boat_in_still_water (V_b : ℝ) (V_d : ℝ) :
  V_d = distance_downstream / time_downstream →
  V_d = V_b + speed_of_stream →
  V_b = 16 :=
by
  intros Vd_eq D_eq
  sorry

end speed_of_boat_in_still_water_l277_27752


namespace solve_x_for_equation_l277_27792

theorem solve_x_for_equation (x : ℝ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) : x = -14 :=
by 
  sorry

end solve_x_for_equation_l277_27792


namespace sum_of_distances_l277_27731

theorem sum_of_distances (A B C : ℝ × ℝ) (hA : A.2^2 = 8 * A.1) (hB : B.2^2 = 8 * B.1) 
(hC : C.2^2 = 8 * C.1) (h_centroid : (A.1 + B.1 + C.1) / 3 = 2) : 
  dist (2, 0) A + dist (2, 0) B + dist (2, 0) C = 12 := 
sorry

end sum_of_distances_l277_27731


namespace simon_age_in_2010_l277_27784

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end simon_age_in_2010_l277_27784


namespace probability_blue_or_green_is_two_thirds_l277_27796

-- Definitions for the given conditions
def blue_faces := 3
def red_faces := 2
def green_faces := 1
def total_faces := blue_faces + red_faces + green_faces
def successful_outcomes := blue_faces + green_faces

-- Probability definition
def probability_blue_or_green := (successful_outcomes : ℚ) / total_faces

-- The theorem we want to prove
theorem probability_blue_or_green_is_two_thirds :
  probability_blue_or_green = (2 / 3 : ℚ) :=
by
  -- here would be the proof steps, but we replace them with sorry as per the instructions
  sorry

end probability_blue_or_green_is_two_thirds_l277_27796


namespace intersection_of_M_and_N_is_N_l277_27760

def M := {x : ℝ | x ≥ -1}
def N := {y : ℝ | y ≥ 0}

theorem intersection_of_M_and_N_is_N : M ∩ N = N := sorry

end intersection_of_M_and_N_is_N_l277_27760


namespace product_of_g_xi_l277_27787

noncomputable def x1 : ℂ := sorry
noncomputable def x2 : ℂ := sorry
noncomputable def x3 : ℂ := sorry
noncomputable def x4 : ℂ := sorry
noncomputable def x5 : ℂ := sorry

def f (x : ℂ) : ℂ := x^5 + x^2 + 1
def g (x : ℂ) : ℂ := x^3 - 2

axiom roots_of_f (x : ℂ) : f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5

theorem product_of_g_xi : (g x1) * (g x2) * (g x3) * (g x4) * (g x5) = -243 := sorry

end product_of_g_xi_l277_27787


namespace least_common_multiple_l277_27705

open Int

theorem least_common_multiple {a b c : ℕ} 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : Nat.lcm a c = 90 := 
sorry

end least_common_multiple_l277_27705


namespace problem_1_problem_2_l277_27799

noncomputable def f (a x : ℝ) : ℝ := |x + a| + |x + 1/a|

theorem problem_1 (x : ℝ) : f 2 x > 3 ↔ x < -(11 / 4) ∨ x > 1 / 4 := sorry

theorem problem_2 (a m : ℝ) (ha : a > 0) : f a m + f a (-1 / m) ≥ 4 := sorry

end problem_1_problem_2_l277_27799


namespace max_value_l277_27702

noncomputable def max_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z)^2 / (x^2 + y^2 + z^2)

theorem max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_fraction x y z hx hy hz ≤ 3 :=
sorry

end max_value_l277_27702


namespace problems_left_to_grade_l277_27716

def worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

theorem problems_left_to_grade : (worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end problems_left_to_grade_l277_27716


namespace negate_proposition_l277_27756

theorem negate_proposition : (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by
  sorry

end negate_proposition_l277_27756


namespace option_b_correct_l277_27778

theorem option_b_correct : (-(-2)) = abs (-2) := by
  sorry

end option_b_correct_l277_27778


namespace train_speed_l277_27766

def train_length : ℝ := 360 -- length of the train in meters
def crossing_time : ℝ := 6 -- time taken to cross the man in seconds

theorem train_speed (train_length crossing_time : ℝ) : 
  (train_length = 360) → (crossing_time = 6) → (train_length / crossing_time = 60) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end train_speed_l277_27766


namespace exists_small_area_triangle_l277_27765

def lattice_point (x y : ℤ) : Prop := |x| ≤ 2 ∧ |y| ≤ 2

def no_three_collinear (points : List (ℤ × ℤ)) : Prop :=
∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
(p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
¬ (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) = 0)

noncomputable def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
(1 / 2 : ℚ) * |(p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))|

theorem exists_small_area_triangle {points : List (ℤ × ℤ)}
  (h1 : points.length = 6)
  (h2 : ∀ (p : ℤ × ℤ), p ∈ points → lattice_point p.1 p.2)
  (h3 : no_three_collinear points) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  triangle_area p1 p2 p3 ≤ 2 := 
sorry

end exists_small_area_triangle_l277_27765
