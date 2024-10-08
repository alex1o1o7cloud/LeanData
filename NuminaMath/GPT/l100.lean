import Mathlib

namespace units_digit_base8_l100_100677

theorem units_digit_base8 (a b : ℕ) (h_a : a = 123) (h_b : b = 57) :
  let product := a * b
  let units_digit := product % 8
  units_digit = 7 := by
  sorry

end units_digit_base8_l100_100677


namespace consecutive_int_sqrt_l100_100545

theorem consecutive_int_sqrt (m n : ℤ) (h1 : m < n) (h2 : m < Real.sqrt 13) (h3 : Real.sqrt 13 < n) (h4 : n = m + 1) : m * n = 12 :=
sorry

end consecutive_int_sqrt_l100_100545


namespace abs_neg_one_half_eq_one_half_l100_100191

theorem abs_neg_one_half_eq_one_half : abs (-1/2) = 1/2 := 
by sorry

end abs_neg_one_half_eq_one_half_l100_100191


namespace hawks_points_l100_100116

theorem hawks_points (x y z : ℤ) 
  (h_total_points: x + y = 82)
  (h_margin: x - y = 18)
  (h_eagles_points: x = 12 + z) : 
  y = 32 := 
sorry

end hawks_points_l100_100116


namespace village_food_sales_l100_100340

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l100_100340


namespace andrew_spent_total_amount_l100_100334

/-- Conditions:
1. Andrew played a total of 7 games.
2. Cost distribution for games:
   - 3 games cost $9.00 each
   - 2 games cost $12.50 each
   - 2 games cost $15.00 each
3. Additional expenses:
   - $25.00 on snacks
   - $20.00 on drinks
-/
def total_cost_games : ℝ :=
  (3 * 9) + (2 * 12.5) + (2 * 15)

def cost_snacks : ℝ := 25
def cost_drinks : ℝ := 20

def total_spent (cost_games cost_snacks cost_drinks : ℝ) : ℝ :=
  cost_games + cost_snacks + cost_drinks

theorem andrew_spent_total_amount :
  total_spent total_cost_games 25 20 = 127 := by
  -- The proof is omitted
  sorry

end andrew_spent_total_amount_l100_100334


namespace team_leader_prize_l100_100734

theorem team_leader_prize 
    (number_of_students : ℕ := 10)
    (number_of_team_members : ℕ := 9)
    (team_member_prize : ℕ := 200)
    (additional_leader_prize : ℕ := 90)
    (total_prize : ℕ)
    (leader_prize : ℕ := total_prize - (number_of_team_members * team_member_prize))
    (average_prize : ℕ := (total_prize + additional_leader_prize) / number_of_students)
: leader_prize = 300 := 
by {
  sorry  -- Proof omitted
}

end team_leader_prize_l100_100734


namespace rockets_win_30_l100_100802

-- Given conditions
def hawks_won (h : ℕ) (w : ℕ) : Prop := h > w
def rockets_won (r : ℕ) (k : ℕ) (l : ℕ) : Prop := r > k ∧ r < l
def knicks_at_least (k : ℕ) : Prop := k ≥ 15
def clippers_won (c : ℕ) (l : ℕ) : Prop := c < l

-- Possible number of games won
def possible_games : List ℕ := [15, 20, 25, 30, 35, 40]

-- Prove Rockets won 30 games
theorem rockets_win_30 (h w r k l c : ℕ) 
  (h_w: hawks_won h w)
  (r_kl : rockets_won r k l)
  (k_15: knicks_at_least k)
  (c_l : clippers_won c l)
  (h_mem : h ∈ possible_games)
  (w_mem : w ∈ possible_games)
  (r_mem : r ∈ possible_games)
  (k_mem : k ∈ possible_games)
  (l_mem : l ∈ possible_games)
  (c_mem : c ∈ possible_games) :
  r = 30 :=
sorry

end rockets_win_30_l100_100802


namespace boxes_containing_neither_l100_100574

-- Define the conditions
def total_boxes : ℕ := 15
def boxes_with_pencils : ℕ := 8
def boxes_with_pens : ℕ := 5
def boxes_with_markers : ℕ := 3
def boxes_with_pencils_and_pens : ℕ := 2
def boxes_with_pencils_and_markers : ℕ := 1
def boxes_with_pens_and_markers : ℕ := 1
def boxes_with_all_three : ℕ := 0

-- The proof problem
theorem boxes_containing_neither (h: total_boxes = 15) : 
  total_boxes - ((boxes_with_pencils - boxes_with_pencils_and_pens - boxes_with_pencils_and_markers) + 
  (boxes_with_pens - boxes_with_pencils_and_pens - boxes_with_pens_and_markers) + 
  (boxes_with_markers - boxes_with_pencils_and_markers - boxes_with_pens_and_markers) + 
  boxes_with_pencils_and_pens + boxes_with_pencils_and_markers + boxes_with_pens_and_markers) = 3 := 
by
  -- Specify that we want to use the equality of the number of boxes
  sorry

end boxes_containing_neither_l100_100574


namespace postage_arrangements_11_cents_l100_100108

-- Definitions for the problem settings, such as stamp denominations and counts
def stamp_collection : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Function to calculate all unique arrangements of stamps that sum to a given value (11 cents)
def count_arrangements (total_cents : ℕ) : ℕ :=
  -- The implementation would involve a combinatorial counting taking into account the problem conditions
  sorry

-- The main theorem statement asserting the solution
theorem postage_arrangements_11_cents :
  count_arrangements 11 = 71 :=
  sorry

end postage_arrangements_11_cents_l100_100108


namespace compare_y_values_l100_100577

variable (a : ℝ) (y₁ y₂ : ℝ)
variable (h : a > 0)
variable (p1 : y₁ = a * (-1 : ℝ)^2 - 4 * a * (-1 : ℝ) + 2)
variable (p2 : y₂ = a * (1 : ℝ)^2 - 4 * a * (1 : ℝ) + 2)

theorem compare_y_values : y₁ > y₂ :=
by {
  sorry
}

end compare_y_values_l100_100577


namespace original_number_is_correct_l100_100951

noncomputable def original_number : ℝ :=
  let x := 11.26666666666667
  let y := 30.333333333333332
  x + y

theorem original_number_is_correct (x y : ℝ) (h₁ : 10 * x + 22 * y = 780) (h₂ : y = 30.333333333333332) : 
  original_number = 41.6 :=
by
  sorry

end original_number_is_correct_l100_100951


namespace find_phi_l100_100445

open Real

theorem find_phi (φ : ℝ) (hφ : |φ| < π / 2)
  (h_symm : ∀ x, sin (2 * x + φ) = sin (2 * ((2 * π / 3 - x) / 2) + φ)) :
  φ = -π / 6 :=
by
  sorry

end find_phi_l100_100445


namespace gina_snake_mice_eaten_in_decade_l100_100705

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l100_100705


namespace platform_length_605_l100_100615

noncomputable def length_of_platform (speed_kmh : ℕ) (accel : ℚ) (t_platform : ℚ) (t_man : ℚ) (dist_man_from_platform : ℚ) : ℚ :=
  let speed_ms := (speed_kmh : ℚ) * 1000 / 3600
  let distance_man := speed_ms * t_man + 0.5 * accel * t_man^2
  let train_length := distance_man - dist_man_from_platform
  let distance_platform := speed_ms * t_platform + 0.5 * accel * t_platform^2
  distance_platform - train_length

theorem platform_length_605 :
  length_of_platform 54 0.5 40 20 5 = 605 := by
  sorry

end platform_length_605_l100_100615


namespace pizzas_served_during_lunch_l100_100458

theorem pizzas_served_during_lunch {total_pizzas dinner_pizzas lunch_pizzas: ℕ} 
(h_total: total_pizzas = 15) (h_dinner: dinner_pizzas = 6) (h_eq: total_pizzas = dinner_pizzas + lunch_pizzas) : 
lunch_pizzas = 9 := by
  sorry

end pizzas_served_during_lunch_l100_100458


namespace man_twice_son_age_l100_100796

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 27) (h2 : M = S + 29) (h3 : M + Y = 2 * (S + Y)) : Y = 2 := 
by sorry

end man_twice_son_age_l100_100796


namespace remainder_n_pow_5_minus_n_mod_30_l100_100743

theorem remainder_n_pow_5_minus_n_mod_30 (n : ℤ) : (n^5 - n) % 30 = 0 := 
by sorry

end remainder_n_pow_5_minus_n_mod_30_l100_100743


namespace asymptote_of_hyperbola_l100_100988

theorem asymptote_of_hyperbola :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) → y = x / 2 ∨ y = - x / 2 :=
sorry

end asymptote_of_hyperbola_l100_100988


namespace find_a_l100_100425

-- Assuming the existence of functions and variables as per conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Defining the given conditions
axiom cond1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x - 5
axiom cond2 : f a = 6

-- Now stating the proof goal
theorem find_a : a = 7 / 4 := by
  sorry

end find_a_l100_100425


namespace intersection_complement_l100_100757

def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | x > 3}

theorem intersection_complement :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l100_100757


namespace allocation_schemes_correct_l100_100526

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l100_100526


namespace smallest_product_l100_100950

theorem smallest_product (S : Set ℤ) (hS : S = { -8, -3, -2, 2, 4 }) :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧ ∀ (x y : ℤ), x ∈ S → y ∈ S → x * y ≥ -32 :=
by
  sorry

end smallest_product_l100_100950


namespace ratio_a_to_c_l100_100771

-- Declaring the variables a, b, c, and d as real numbers.
variables (a b c d : ℝ)

-- Define the conditions given in the problem.
def ratio_conditions : Prop :=
  (a / b = 5 / 4) ∧ (c / d = 4 / 3) ∧ (d / b = 1 / 5)

-- State the theorem we need to prove based on the conditions.
theorem ratio_a_to_c (h : ratio_conditions a b c d) : a / c = 75 / 16 :=
by
  sorry

end ratio_a_to_c_l100_100771


namespace garden_roller_area_l100_100407

theorem garden_roller_area (D : ℝ) (A : ℝ) (π : ℝ) (L_new : ℝ) :
  D = 1.4 → A = 88 → π = 22/7 → L_new = 4 → A = 5 * (2 * π * (D / 2) * L_new) :=
by sorry

end garden_roller_area_l100_100407


namespace max_value_of_k_l100_100403

noncomputable def max_possible_k (x y : ℝ) (k : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < k ∧
  (3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x))

theorem max_value_of_k (x y : ℝ) (k : ℝ) :
  max_possible_k x y k → k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_value_of_k_l100_100403


namespace focus_of_parabola_l100_100509

theorem focus_of_parabola (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x, y = 3 * x → 3 / a = 3) :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 8) :=
by
  -- The proof goes here
  sorry

end focus_of_parabola_l100_100509


namespace sanoop_initial_tshirts_l100_100477

theorem sanoop_initial_tshirts (n : ℕ) (T : ℕ) 
(avg_initial : T = n * 526) 
(avg_remaining : T - 673 = (n - 1) * 505) 
(avg_returned : 673 = 673) : 
n = 8 := 
by 
  sorry

end sanoop_initial_tshirts_l100_100477


namespace minimum_value_ineq_l100_100920

noncomputable def problem_statement (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) → (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2)

theorem minimum_value_ineq (a b c : ℝ) (h : a + b + c = 3) : problem_statement a b c h :=
  sorry

end minimum_value_ineq_l100_100920


namespace fraction_product_correct_l100_100031

theorem fraction_product_correct : (3 / 5) * (4 / 7) * (5 / 9) = 4 / 21 :=
by
  sorry

end fraction_product_correct_l100_100031


namespace swimmer_speed_is_4_4_l100_100772

noncomputable def swimmer_speed_in_still_water (distance : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
(distance / time) + current_speed

theorem swimmer_speed_is_4_4 :
  swimmer_speed_in_still_water 7 2.5 3.684210526315789 = 4.4 :=
by
  -- This part would contain the proof to show that the calculated speed is 4.4
  sorry

end swimmer_speed_is_4_4_l100_100772


namespace log3_cubicroot_of_3_l100_100197

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem log3_cubicroot_of_3 :
  log_base_3 (3 ^ (1/3 : ℝ)) = 1 / 3 :=
by
  sorry

end log3_cubicroot_of_3_l100_100197


namespace maurice_rides_before_visit_l100_100510

-- Defining all conditions in Lean
variables
  (M : ℕ) -- Number of times Maurice had been horseback riding before visiting Matt
  (Matt_rides_with_M : ℕ := 8 * 2) -- Number of times Matt rode with Maurice (8 times, 2 horses each time)
  (Matt_rides_alone : ℕ := 16) -- Number of times Matt rode solo
  (total_Matt_rides : ℕ := Matt_rides_with_M + Matt_rides_alone) -- Total rides by Matt
  (three_times_M : ℕ := 3 * M) -- Three times the number of times Maurice rode before visiting
  (unique_horses_M : ℕ := 8) -- Total number of unique horses Maurice rode during his visit

-- Main theorem
theorem maurice_rides_before_visit  
  (h1: total_Matt_rides = three_times_M) 
  (h2: unique_horses_M = M) 
  : M = 10 := sorry

end maurice_rides_before_visit_l100_100510


namespace pizza_slice_volume_l100_100240

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l100_100240


namespace find_n_l100_100656

theorem find_n (n : ℕ) : (16 : ℝ)^(1/4) = 2^n ↔ n = 1 := by
  sorry

end find_n_l100_100656


namespace find_louis_age_l100_100162

variables (C L : ℕ)

-- Conditions:
-- 1. In some years, Carla will be 30 years old
-- 2. The sum of the current ages of Carla and Louis is 55

theorem find_louis_age (h1 : ∃ n, C + n = 30) (h2 : C + L = 55) : L = 25 :=
by {
  sorry
}

end find_louis_age_l100_100162


namespace num_prime_divisors_50_fact_l100_100981
open Nat -- To simplify working with natural numbers

-- We define the prime numbers less than or equal to 50.
def primes_le_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- The problem statement: Prove that the number of prime divisors of 50! which are less than or equal to 50 is 15.
theorem num_prime_divisors_50_fact : (primes_le_50.length = 15) :=
by 
  -- Here we use sorry to skip the proof.
  sorry

end num_prime_divisors_50_fact_l100_100981


namespace no_such_function_exists_l100_100066

def f (n : ℕ) : ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → (f n = f (f (n - 1)) + f (f (n + 1))) :=
by
  sorry

end no_such_function_exists_l100_100066


namespace johnny_fishes_l100_100780

theorem johnny_fishes
  (total_fishes : ℕ)
  (sony_ratio : ℕ)
  (total_is_40 : total_fishes = 40)
  (sony_is_4x_johnny : sony_ratio = 4)
  : ∃ (johnny_fishes : ℕ), johnny_fishes + sony_ratio * johnny_fishes = total_fishes ∧ johnny_fishes = 8 :=
by
  sorry

end johnny_fishes_l100_100780


namespace tan_15_simplification_l100_100898

theorem tan_15_simplification :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tan_15_simplification_l100_100898


namespace watermelon_yield_increase_l100_100534

noncomputable def yield_increase (initial_yield final_yield annual_increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_yield * (1 + annual_increase_rate) ^ years

theorem watermelon_yield_increase :
  ∀ (x : ℝ),
    (yield_increase 20 28.8 x 2 = 28.8) →
    (yield_increase 28.8 40 x 2 > 40) :=
by
  intros x hx
  have incEq : 20 * (1 + x) ^ 2 = 28.8 := hx
  sorry

end watermelon_yield_increase_l100_100534


namespace ac_work_time_l100_100568

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l100_100568


namespace machines_work_together_time_l100_100288

theorem machines_work_together_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 20) (h2 : rate2 = 1 / 30) :
  (1 / (rate1 + rate2)) = 12 :=
by
  sorry

end machines_work_together_time_l100_100288


namespace center_square_number_l100_100379

def in_center_square (grid : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := grid 1 1

theorem center_square_number
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (consecutive_share_edge : ∀ (i j : Fin 3) (n : ℕ), 
                              (i < 2 ∨ j < 2) →
                              (∃ d, d ∈ [(-1,0), (1,0), (0,-1), (0,1)] ∧ 
                              grid (i + d.1) (j + d.2) = n + 1))
  (corner_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20)
  (diagonal_sum_15 : 
    (grid 0 0 + grid 1 1 + grid 2 2 = 15) 
    ∨ 
    (grid 0 2 + grid 1 1 + grid 2 0 = 15))
  : in_center_square grid = 5 := sorry

end center_square_number_l100_100379


namespace Sarah_shampoo_conditioner_usage_l100_100368

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l100_100368


namespace equal_points_per_person_l100_100591

theorem equal_points_per_person :
  let blue_eggs := 12
  let blue_points := 2
  let pink_eggs := 5
  let pink_points := 3
  let golden_eggs := 3
  let golden_points := 5
  let total_people := 4
  (blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points) / total_people = 13 :=
by
  -- place the steps based on the conditions and calculations
  sorry

end equal_points_per_person_l100_100591


namespace incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l100_100593

structure Tetrahedron (α : Type*) [MetricSpace α] :=
(A B C D : α)

def Incenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry
def Circumcenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry

def equidistant_from_faces {α : Type*} [MetricSpace α] (T : Tetrahedron α) (I : α) : Prop := sorry
def equidistant_from_vertices {α : Type*} [MetricSpace α] (T : Tetrahedron α) (O : α) : Prop := sorry
def skew_edges_equal {α : Type*} [MetricSpace α] (T : Tetrahedron α) : Prop := sorry

theorem incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal
  {α : Type*} [MetricSpace α] (T : Tetrahedron α) :
  (∃ I, ∃ O, (Incenter T = I) ∧ (Circumcenter T = O) ∧ 
            (equidistant_from_faces T I) ∧ (equidistant_from_vertices T O)) ↔ (skew_edges_equal T) := 
sorry

end incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l100_100593


namespace remainder_2n_div_9_l100_100710

theorem remainder_2n_div_9 (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := 
sorry

end remainder_2n_div_9_l100_100710


namespace minimize_quadratic_function_l100_100075

def quadratic_function (x : ℝ) : ℝ := x^2 + 8*x + 7

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = -4 :=
by
  sorry

end minimize_quadratic_function_l100_100075


namespace hannah_bananas_l100_100573

theorem hannah_bananas (B : ℕ) (h1 : B / 4 = 15 / 3) : B = 20 :=
by
  sorry

end hannah_bananas_l100_100573


namespace sequence_general_formula_l100_100121

theorem sequence_general_formula (a : ℕ+ → ℝ) (h₀ : a 1 = 7 / 8)
  (h₁ : ∀ n : ℕ+, a (n + 1) = 1 / 2 * a n + 1 / 3) :
  ∀ n : ℕ+, a n = 5 / 24 * (1 / 2)^(n - 1 : ℕ) + 2 / 3 :=
by
  sorry

end sequence_general_formula_l100_100121


namespace arun_completes_work_alone_in_70_days_l100_100929

def arun_days (A : ℕ) : Prop :=
  ∃ T : ℕ, (A > 0) ∧ (T > 0) ∧ 
           (∀ (work_done_by_arun_in_1_day work_done_by_tarun_in_1_day : ℝ),
            work_done_by_arun_in_1_day = 1 / A ∧
            work_done_by_tarun_in_1_day = 1 / T ∧
            (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day = 1 / 10) ∧
            (4 * (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day) = 4 / 10) ∧
            (42 * work_done_by_arun_in_1_day = 6 / 10) )

theorem arun_completes_work_alone_in_70_days : arun_days 70 :=
  sorry

end arun_completes_work_alone_in_70_days_l100_100929


namespace percentage_difference_l100_100302

theorem percentage_difference (water_yesterday : ℕ) (water_two_days_ago : ℕ) (h1 : water_yesterday = 48) (h2 : water_two_days_ago = 50) : 
  (water_two_days_ago - water_yesterday) / water_two_days_ago * 100 = 4 :=
by
  sorry

end percentage_difference_l100_100302


namespace opposite_of_neg_3_l100_100281

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l100_100281


namespace claim1_claim2_l100_100397

theorem claim1 (n : ℤ) (hs : ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0) : 
  ∃ k : ℤ, n = 4 * k := 
sorry

theorem claim2 (n : ℕ) (h : n % 4 = 0) : 
  ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0 := 
sorry

end claim1_claim2_l100_100397


namespace range_of_m_for_inequality_l100_100554

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 := 
sorry

end range_of_m_for_inequality_l100_100554


namespace cylinder_height_l100_100778

theorem cylinder_height (base_area : ℝ) (h s : ℝ)
  (h_base : base_area > 0)
  (h_ratio : (1 / 3 * base_area * 4.5) / (base_area * h) = 1 / 6)
  (h_cone_height : s = 4.5) :
  h = 9 :=
by
  -- Proof omitted
  sorry

end cylinder_height_l100_100778


namespace FI_squared_correct_l100_100220

noncomputable def FI_squared : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 4)
  let D : ℝ × ℝ := (0, 4)
  let E : ℝ × ℝ := (3, 0)
  let H : ℝ × ℝ := (0, 1)
  let F : ℝ × ℝ := (4, 1)
  let G : ℝ × ℝ := (1, 4)
  let I : ℝ × ℝ := (3, 0)
  let J : ℝ × ℝ := (0, 1)
  let FI_squared := (4 - 3)^2 + (1 - 0)^2
  FI_squared

theorem FI_squared_correct : FI_squared = 2 :=
by
  sorry

end FI_squared_correct_l100_100220


namespace no_such_increasing_seq_exists_l100_100245

theorem no_such_increasing_seq_exists :
  ¬(∃ (a : ℕ → ℕ), (∀ m n : ℕ, a (m * n) = a m + a n) ∧ (∀ n : ℕ, a n < a (n + 1))) :=
by
  sorry

end no_such_increasing_seq_exists_l100_100245


namespace horse_revolutions_l100_100233

noncomputable def carousel_revolutions (r1 r2 d1 : ℝ) : ℝ :=
  (d1 * r1) / r2

theorem horse_revolutions :
  carousel_revolutions 30 10 40 = 120 :=
by
  sorry

end horse_revolutions_l100_100233


namespace find_c_l100_100635

theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 5 * x + 8 * y + c = 0 ∧ x + y = 26) : c = -80 :=
sorry

end find_c_l100_100635


namespace no_solution_in_natural_numbers_l100_100037

theorem no_solution_in_natural_numbers (x y z : ℕ) : ¬((2 * x) ^ (2 * x) - 1 = y ^ (z + 1)) := 
  sorry

end no_solution_in_natural_numbers_l100_100037


namespace trapezoid_is_proposition_l100_100265

-- Define what it means to be a proposition
def is_proposition (s : String) : Prop := ∃ b : Bool, (s = "A trapezoid is a quadrilateral" ∨ s = "Construct line AB" ∨ s = "x is an integer" ∨ s = "Will it snow today?") ∧ 
  (b → s = "A trapezoid is a quadrilateral") 

-- Main proof statement
theorem trapezoid_is_proposition : is_proposition "A trapezoid is a quadrilateral" :=
  sorry

end trapezoid_is_proposition_l100_100265


namespace arithmetic_sequence_sum_l100_100478

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l100_100478


namespace length_down_correct_l100_100866

variable (rate_up rate_down time_up time_down length_down : ℕ)
variable (h1 : rate_up = 8)
variable (h2 : time_up = 2)
variable (h3 : time_down = time_up)
variable (h4 : rate_down = (3 / 2) * rate_up)
variable (h5 : length_down = rate_down * time_down)

theorem length_down_correct : length_down = 24 := by
  sorry

end length_down_correct_l100_100866


namespace smallest_positive_multiple_of_18_with_digits_9_or_0_l100_100311

noncomputable def m : ℕ := 90
theorem smallest_positive_multiple_of_18_with_digits_9_or_0 : m = 90 ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 9) ∧ m % 18 = 0 → m / 18 = 5 :=
by
  intro h
  sorry

end smallest_positive_multiple_of_18_with_digits_9_or_0_l100_100311


namespace lea_notebooks_count_l100_100285

theorem lea_notebooks_count
  (cost_book : ℕ)
  (cost_binder : ℕ)
  (num_binders : ℕ)
  (cost_notebook : ℕ)
  (total_cost : ℕ)
  (h_book : cost_book = 16)
  (h_binder : cost_binder = 2)
  (h_num_binders : num_binders = 3)
  (h_notebook : cost_notebook = 1)
  (h_total : total_cost = 28) :
  ∃ num_notebooks : ℕ, num_notebooks = 6 ∧
    total_cost = cost_book + num_binders * cost_binder + num_notebooks * cost_notebook := 
by
  sorry

end lea_notebooks_count_l100_100285


namespace concert_attendance_l100_100795

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l100_100795


namespace price_increase_percentage_l100_100712

theorem price_increase_percentage (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 360) : 
  (new_price - original_price) / original_price * 100 = 20 := 
by
  sorry

end price_increase_percentage_l100_100712


namespace arithmetic_sequence_sum_l100_100272

open Nat

theorem arithmetic_sequence_sum (m n : Nat) (d : ℤ) (a_1 : ℤ)
    (hnm : n ≠ m)
    (hSn : (n * (2 * a_1 + (n - 1) * d) / 2) = n / m)
    (hSm : (m * (2 * a_1 + (m - 1) * d) / 2) = m / n) :
  ((m + n) * (2 * a_1 + (m + n - 1) * d) / 2) > 4 := by
  sorry

end arithmetic_sequence_sum_l100_100272


namespace identify_different_correlation_l100_100626

-- Define the concept of correlation
inductive Correlation
| positive
| negative

-- Define the conditions for each option
def option_A : Correlation := Correlation.positive
def option_B : Correlation := Correlation.positive
def option_C : Correlation := Correlation.negative
def option_D : Correlation := Correlation.positive

-- The statement to prove
theorem identify_different_correlation :
  (option_A = Correlation.positive) ∧ 
  (option_B = Correlation.positive) ∧ 
  (option_D = Correlation.positive) ∧ 
  (option_C = Correlation.negative) := 
sorry

end identify_different_correlation_l100_100626


namespace add_to_make_divisible_by_23_l100_100684

def least_addend_for_divisibility (n k : ℕ) : ℕ :=
  let remainder := n % k
  k - remainder

theorem add_to_make_divisible_by_23 : least_addend_for_divisibility 1053 23 = 5 :=
by
  sorry

end add_to_make_divisible_by_23_l100_100684


namespace new_interest_rate_l100_100139

theorem new_interest_rate 
    (i₁ : ℝ) (r₁ : ℝ) (p : ℝ) (additional_interest : ℝ) (i₂ : ℝ) (r₂ : ℝ)
    (h1 : r₁ = 0.05)
    (h2 : i₁ = 101.20)
    (h3 : additional_interest = 20.24)
    (h4 : i₂ = i₁ + additional_interest)
    (h5 : p = i₁ / (r₁ * 1))
    (h6 : i₂ = p * r₂ * 1) :
  r₂ = 0.06 :=
by
  sorry

end new_interest_rate_l100_100139


namespace num_undef_values_l100_100007

theorem num_undef_values : 
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, (x^2 + 4 * x - 5) * (x - 4) = 0 → x = -5 ∨ x = 1 ∨ x = 4 :=
by
  -- We are stating that there exists a natural number n such that n = 3
  -- and for all real numbers x, if (x^2 + 4*x - 5)*(x - 4) = 0,
  -- then x must be one of -5, 1, or 4.
  sorry

end num_undef_values_l100_100007


namespace numerator_equals_denominator_l100_100524

theorem numerator_equals_denominator (x : ℝ) (h : 4 * x - 3 = 5 * x + 2) : x = -5 :=
  by
    sorry

end numerator_equals_denominator_l100_100524


namespace corner_contains_same_color_cells_l100_100657

theorem corner_contains_same_color_cells (colors : Finset (Fin 120)) :
  ∀ (coloring : Fin 2017 × Fin 2017 → Fin 120),
  ∃ (corner : Fin 2017 × Fin 2017 → Prop), 
    (∃ cell1 cell2, corner cell1 ∧ corner cell2 ∧ coloring cell1 = coloring cell2) := 
by 
  sorry

end corner_contains_same_color_cells_l100_100657


namespace total_earnings_l100_100389

theorem total_earnings (x y : ℕ) 
  (h1 : 2 * x * y = 250) : 
  58 * (x * y) = 7250 := 
by
  sorry

end total_earnings_l100_100389


namespace probability_of_break_in_first_50_meters_l100_100219

theorem probability_of_break_in_first_50_meters (total_length favorable_length : ℝ) 
  (h_total_length : total_length = 320) 
  (h_favorable_length : favorable_length = 50) : 
  (favorable_length / total_length) = 0.15625 := 
sorry

end probability_of_break_in_first_50_meters_l100_100219


namespace car_service_month_l100_100548

-- Define the conditions
def first_service_month : ℕ := 3 -- Representing March as the 3rd month
def service_interval : ℕ := 7
def total_services : ℕ := 13

-- Define an auxiliary function to calculate months and reduce modulo 12
def nth_service_month (first_month : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  (first_month + (interval * (n - 1))) % 12

-- The theorem statement
theorem car_service_month : nth_service_month first_service_month service_interval total_services = 3 :=
by
  -- The proof steps will go here
  sorry

end car_service_month_l100_100548


namespace problem_l100_100087

-- Definitions and hypotheses based on the given conditions
variable (a b : ℝ)
def sol_set := {x : ℝ | -1/2 < x ∧ x < 1/3}
def quadratic_inequality (x : ℝ) := a * x^2 + b * x + 2

-- Statement expressing that the inequality holds for the given solution set
theorem problem
  (h : ∀ (x : ℝ), x ∈ sol_set → quadratic_inequality a b x > 0) :
  a - b = -10 :=
sorry

end problem_l100_100087


namespace sum_of_squares_of_roots_l100_100386

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 9) (h2 : s₁ * s₂ = 14) :
  s₁^2 + s₂^2 = 53 :=
by
  sorry

end sum_of_squares_of_roots_l100_100386


namespace selling_price_of_article_l100_100819

theorem selling_price_of_article (cost_price : ℕ) (gain_percent : ℕ) (profit : ℕ) (selling_price : ℕ) : 
  cost_price = 100 → gain_percent = 10 → profit = (gain_percent * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 110 :=
by
  intros
  sorry

end selling_price_of_article_l100_100819


namespace percentage_increase_on_sale_l100_100739

theorem percentage_increase_on_sale (P S : ℝ) (hP : P ≠ 0) (hS : S ≠ 0)
  (h_price_reduction : (0.8 : ℝ) * P * S * (1 + (X / 100)) = 1.44 * P * S) :
  X = 80 := by
  sorry

end percentage_increase_on_sale_l100_100739


namespace factorize_expression_l100_100761

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l100_100761


namespace prime_square_minus_one_divisible_by_24_l100_100911

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (h_prime : Prime p) (h_gt_3 : p > 3) : 
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
sorry

end prime_square_minus_one_divisible_by_24_l100_100911


namespace eq_has_unique_solution_l100_100487

theorem eq_has_unique_solution : 
  ∃! x : ℝ, (x ≠ 0)
    ∧ ((x < 0 → false) ∧ 
      (x > 0 → (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9)) :=
by sorry

end eq_has_unique_solution_l100_100487


namespace ellipse_slope_product_constant_l100_100124

noncomputable def ellipse_constant_slope_product (a b : ℝ) (P M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (N.1 = -M.1 ∧ N.2 = -M.2) ∧
  (∃ k_PM k_PN : ℝ, k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_PN = (P.2 - N.2) / (P.1 - N.1)) ∧
  ((P.2 - M.2) / (P.1 - M.1) * (P.2 - N.2) / (P.1 - N.1) = -b^2 / a^2)

theorem ellipse_slope_product_constant (a b : ℝ) (P M N : ℝ × ℝ) :
  ellipse_constant_slope_product a b P M N := 
sorry

end ellipse_slope_product_constant_l100_100124


namespace correct_addition_result_l100_100356

-- Define the particular number x and state the condition.
variable (x : ℕ) (h₁ : x + 21 = 52)

-- Assert that the correct result when adding 40 to x is 71.
theorem correct_addition_result : x + 40 = 71 :=
by
  -- Proof would go here; represented as a placeholder for now.
  sorry

end correct_addition_result_l100_100356


namespace find_f_neg_2010_6_l100_100415

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_one (x : ℝ) : f (x + 1) + f x = 3

axiom f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = 2 - x

theorem find_f_neg_2010_6 : f (-2010.6) = 1.4 := by {
  sorry
}

end find_f_neg_2010_6_l100_100415


namespace karen_locks_l100_100975

theorem karen_locks : 
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  Lboth = 60 :=
by
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  sorry

end karen_locks_l100_100975


namespace bookstore_shoe_store_common_sales_l100_100024

-- Define the conditions
def bookstore_sale_days (d: ℕ) : Prop := d % 4 = 0 ∧ d >= 4 ∧ d <= 28
def shoe_store_sale_days (d: ℕ) : Prop := (d - 2) % 6 = 0 ∧ d >= 2 ∧ d <= 26

-- Define the question to be proven as a theorem
theorem bookstore_shoe_store_common_sales : 
  ∃ (n: ℕ), n = 2 ∧ (
    ∀ (d: ℕ), 
      ((bookstore_sale_days d ∧ shoe_store_sale_days d) → n = 2) 
      ∧ (d < 4 ∨ d > 28 ∨ d < 2 ∨ d > 26 → n = 2)
  ) :=
sorry

end bookstore_shoe_store_common_sales_l100_100024


namespace initial_floor_l100_100150

theorem initial_floor (x y z : ℤ)
  (h1 : y = x - 7)
  (h2 : z = y + 3)
  (h3 : 13 = z + 8) :
  x = 9 :=
sorry

end initial_floor_l100_100150


namespace amy_total_tickets_l100_100319

def amy_initial_tickets : ℕ := 33
def amy_additional_tickets : ℕ := 21

theorem amy_total_tickets : amy_initial_tickets + amy_additional_tickets = 54 := by
  sorry

end amy_total_tickets_l100_100319


namespace parabola_sum_coefficients_l100_100470

theorem parabola_sum_coefficients :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, (x = 0 → a * (x^2) + b * x + c = 1)) ∧
    (∀ x : ℝ, (x = 2 → a * (x^2) + b * x + c = 9)) ∧
    (a * (1^2) + b * 1 + c = 4)
  → a + b + c = 4 :=
by sorry

end parabola_sum_coefficients_l100_100470


namespace doughnut_completion_time_l100_100745

noncomputable def time_completion : Prop :=
  let start_time : ℕ := 7 * 60 -- 7:00 AM in minutes
  let quarter_complete_time : ℕ := 10 * 60 + 20 -- 10:20 AM in minutes
  let efficiency_decrease_time : ℕ := 12 * 60 -- 12:00 PM in minutes
  let one_quarter_duration : ℕ := quarter_complete_time - start_time
  let total_time_before_efficiency_decrease : ℕ := 5 * 60 -- from 7:00 AM to 12:00 PM is 5 hours
  let remaining_time_without_efficiency : ℕ := 4 * one_quarter_duration - total_time_before_efficiency_decrease
  let adjusted_remaining_time : ℕ := remaining_time_without_efficiency * 10 / 9 -- decrease by 10% efficiency
  let total_job_duration : ℕ := total_time_before_efficiency_decrease + adjusted_remaining_time
  let completion_time := efficiency_decrease_time + adjusted_remaining_time
  completion_time = 21 * 60 + 15 -- 9:15 PM in minutes

theorem doughnut_completion_time : time_completion :=
  by 
    sorry

end doughnut_completion_time_l100_100745


namespace john_chips_consumption_l100_100148

/-- John starts the week with a routine. Every day, he eats one bag of chips for breakfast,
  two bags for lunch, and doubles the amount he had for lunch for dinner.
  Prove that by the end of the week, John consumed 49 bags of chips. --/
theorem john_chips_consumption : 
  ∀ (days_in_week : ℕ) (chips_breakfast : ℕ) (chips_lunch : ℕ) (chips_dinner : ℕ), 
    days_in_week = 7 ∧ chips_breakfast = 1 ∧ chips_lunch = 2 ∧ chips_dinner = 2 * chips_lunch →
    days_in_week * (chips_breakfast + chips_lunch + chips_dinner) = 49 :=
by
  intros days_in_week chips_breakfast chips_lunch chips_dinner
  sorry

end john_chips_consumption_l100_100148


namespace math_problem_l100_100203

variable (a b c d : ℝ)

theorem math_problem 
    (h1 : a + b + c + d = 6)
    (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
    36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
    4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := 
by
    sorry

end math_problem_l100_100203


namespace stanley_sold_4_cups_per_hour_l100_100983

theorem stanley_sold_4_cups_per_hour (S : ℕ) (Carl_Hour : ℕ) :
  (Carl_Hour = 7) →
  21 = (Carl_Hour * 3) →
  (21 - 9) = (S * 3) →
  S = 4 :=
by
  intros Carl_Hour_eq Carl_3hours Stanley_eq
  sorry

end stanley_sold_4_cups_per_hour_l100_100983


namespace polygon_sides_l100_100654

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l100_100654


namespace triangle_circle_distance_l100_100784

open Real

theorem triangle_circle_distance 
  (DE DF EF : ℝ)
  (hDE : DE = 12) (hDF : DF = 16) (hEF : EF = 20) :
  let s := (DE + DF + EF) / 2
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let ra := K / (s - EF)
  let DP := s - DF
  let DQ := s
  let DI := sqrt (DP^2 + r^2)
  let DE := sqrt (DQ^2 + ra^2)
  let distance := DE - DI
  distance = 24 * sqrt 2 - 4 * sqrt 10 :=
by
  sorry

end triangle_circle_distance_l100_100784


namespace solve_system_correct_l100_100581

noncomputable def solve_system (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n > 2 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k + x (k + 1) = x (k + 2) ^ 2) ∧ 
  x (n + 1) = x 1 ∧ x (n + 2) = x 2 →
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = 2

theorem solve_system_correct (n : ℕ) (x : ℕ → ℝ) : solve_system n x := 
sorry

end solve_system_correct_l100_100581


namespace final_building_height_l100_100017

noncomputable def height_of_final_building 
    (Crane1_height : ℝ)
    (Building1_height : ℝ)
    (Crane2_height : ℝ)
    (Building2_height : ℝ)
    (Crane3_height : ℝ)
    (Average_difference : ℝ) : ℝ :=
    Crane3_height / (1 + Average_difference)

theorem final_building_height
    (Crane1_height : ℝ := 228)
    (Building1_height : ℝ := 200)
    (Crane2_height : ℝ := 120)
    (Building2_height : ℝ := 100)
    (Crane3_height : ℝ := 147)
    (Average_difference : ℝ := 0.13)
    (HCrane1 : 1 + (Crane1_height - Building1_height) / Building1_height = 1.14)
    (HCrane2 : 1 + (Crane2_height - Building2_height) / Building2_height = 1.20)
    (HAvg : (1.14 + 1.20) / 2 = 1.13) :
    height_of_final_building Crane1_height Building1_height Crane2_height Building2_height Crane3_height Average_difference = 130 := 
sorry

end final_building_height_l100_100017


namespace age_product_difference_is_nine_l100_100360

namespace ArnoldDanny

def current_age := 4
def product_today (A : ℕ) := A * A
def product_next_year (A : ℕ) := (A + 1) * (A + 1)
def difference (A : ℕ) := product_next_year A - product_today A

theorem age_product_difference_is_nine :
  difference current_age = 9 :=
by
  sorry

end ArnoldDanny

end age_product_difference_is_nine_l100_100360


namespace conic_section_hyperbola_l100_100341

theorem conic_section_hyperbola (x y : ℝ) : 
  (2 * x - 7)^2 - 4 * (y + 3)^2 = 169 → 
  -- Explain that this equation is of a hyperbola
  true := 
sorry

end conic_section_hyperbola_l100_100341


namespace total_cases_of_candy_correct_l100_100303

-- Define the number of cases of chocolate bars and lollipops
def cases_of_chocolate_bars : ℕ := 25
def cases_of_lollipops : ℕ := 55

-- Define the total number of cases of candy
def total_cases_of_candy : ℕ := cases_of_chocolate_bars + cases_of_lollipops

-- Prove that the total number of cases of candy is 80
theorem total_cases_of_candy_correct : total_cases_of_candy = 80 := by
  sorry

end total_cases_of_candy_correct_l100_100303


namespace train_passes_bridge_in_expected_time_l100_100163

def train_length : ℕ := 360
def speed_kmph : ℕ := 45
def bridge_length : ℕ := 140

def speed_mps : ℚ := (speed_kmph * 1000) / 3600
def total_distance : ℕ := train_length + bridge_length
def time_to_pass : ℚ := total_distance / speed_mps

theorem train_passes_bridge_in_expected_time : time_to_pass = 40 := by
  sorry

end train_passes_bridge_in_expected_time_l100_100163


namespace triangle_cross_section_l100_100503

-- Definitions for the given conditions
inductive Solid
| Prism
| Pyramid
| Frustum
| Cylinder
| Cone
| TruncatedCone
| Sphere

-- The theorem statement of the proof problem
theorem triangle_cross_section (s : Solid) (cross_section_is_triangle : Prop) : 
  cross_section_is_triangle →
  (s = Solid.Prism ∨ s = Solid.Pyramid ∨ s = Solid.Frustum ∨ s = Solid.Cone) :=
sorry

end triangle_cross_section_l100_100503


namespace imaginary_unit_div_l100_100703

open Complex

theorem imaginary_unit_div (i : ℂ) (hi : i * i = -1) : (i / (1 + i) = (1 / 2) + (1 / 2) * i) :=
by
  sorry

end imaginary_unit_div_l100_100703


namespace height_of_triangle_l100_100831

theorem height_of_triangle (base height area : ℝ) (h1 : base = 6) (h2 : area = 24) (h3 : area = 1 / 2 * base * height) : height = 8 :=
by sorry

end height_of_triangle_l100_100831


namespace correct_answer_l100_100405

def vector := (Int × Int)

-- Definitions of vectors given in conditions
def m : vector := (2, 1)
def n : vector := (0, -2)

def vec_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scalar_mult (c : Int) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vec_dot (v1 v2 : vector) : Int :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition vector combined
def combined_vector := vec_add m (vec_scalar_mult 2 n)

-- The problem is to prove this:
theorem correct_answer : vec_dot (3, 2) combined_vector = 0 :=
  sorry

end correct_answer_l100_100405


namespace initial_salty_cookies_l100_100520

theorem initial_salty_cookies
  (initial_sweet_cookies : ℕ) 
  (ate_sweet_cookies : ℕ) 
  (ate_salty_cookies : ℕ) 
  (ate_diff : ℕ) 
  (H1 : initial_sweet_cookies = 39)
  (H2 : ate_sweet_cookies = 32)
  (H3 : ate_salty_cookies = 23)
  (H4 : ate_diff = 9) :
  initial_sweet_cookies - ate_diff = 30 :=
by sorry

end initial_salty_cookies_l100_100520


namespace projectile_reaches_49_first_time_at_1_point_4_l100_100128

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end projectile_reaches_49_first_time_at_1_point_4_l100_100128


namespace bus_total_people_l100_100528

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l100_100528


namespace required_fencing_l100_100444

-- Definitions from conditions
def length_uncovered : ℝ := 30
def area : ℝ := 720

-- Prove that the amount of fencing required is 78 feet
theorem required_fencing : 
  ∃ (W : ℝ), (area = length_uncovered * W) ∧ (2 * W + length_uncovered = 78) := 
sorry

end required_fencing_l100_100444


namespace find_b_l100_100052

theorem find_b (a b c : ℕ) (h1 : 2 * b = a + c) (h2 : b^2 = c * (a + 1)) (h3 : b^2 = a * (c + 2)) : b = 12 :=
by 
  sorry

end find_b_l100_100052


namespace value_of_each_bill_l100_100178

theorem value_of_each_bill (bank1_withdrawal bank2_withdrawal number_of_bills : ℕ)
  (h1 : bank1_withdrawal = 300) 
  (h2 : bank2_withdrawal = 300) 
  (h3 : number_of_bills = 30) : 
  (bank1_withdrawal + bank2_withdrawal) / number_of_bills = 20 :=
by
  sorry

end value_of_each_bill_l100_100178


namespace hall_length_l100_100408

theorem hall_length (L B A : ℝ) (h1 : B = 2 / 3 * L) (h2 : A = 2400) (h3 : A = L * B) : L = 60 := by
  -- proof steps here
  sorry

end hall_length_l100_100408


namespace jane_average_speed_l100_100072

theorem jane_average_speed :
  let total_distance := 200
  let total_time := 6
  total_distance / total_time = 100 / 3 :=
by
  sorry

end jane_average_speed_l100_100072


namespace MarysTotalCandies_l100_100129

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end MarysTotalCandies_l100_100129


namespace find_B_share_l100_100995

-- Definitions for the conditions
def proportion (a b c d : ℕ) := 6 * a = 3 * b ∧ 3 * b = 5 * c ∧ 5 * c = 4 * d

def condition (c d : ℕ) := c = d + 1000

-- Statement of the problem
theorem find_B_share (A B C D : ℕ) (x : ℕ) 
  (h1 : proportion (6*x) (3*x) (5*x) (4*x)) 
  (h2 : condition (5*x) (4*x)) : 
  B = 3000 :=
by 
  sorry

end find_B_share_l100_100995


namespace interest_difference_l100_100847

noncomputable def principal : ℝ := 6200
noncomputable def rate : ℝ := 5 / 100
noncomputable def time : ℝ := 10

noncomputable def interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem interest_difference :
  (principal - interest principal rate time) = 3100 := 
by
  sorry

end interest_difference_l100_100847


namespace q_value_at_2_l100_100093

-- Define the function q and the fact that (2, 3) is on its graph
def q : ℝ → ℝ := sorry

-- Condition: (2, 3) is on the graph of q(x)
axiom q_at_2 : q 2 = 3

-- Theorem: The value of q(2) is 3
theorem q_value_at_2 : q 2 = 3 := 
by 
  apply q_at_2

end q_value_at_2_l100_100093


namespace option_C_true_l100_100801

variable {a b : ℝ}

theorem option_C_true (h : a < b) : a / 3 < b / 3 := sorry

end option_C_true_l100_100801


namespace hanoi_moves_minimal_l100_100697

theorem hanoi_moves_minimal (n : ℕ) : ∃ m, 
  (∀ move : ℕ, move = 2^n - 1 → move = m) := 
by
  sorry

end hanoi_moves_minimal_l100_100697


namespace exists_int_solutions_for_equations_l100_100870

theorem exists_int_solutions_for_equations : 
  ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 :=
by
  sorry

end exists_int_solutions_for_equations_l100_100870


namespace selfish_subsets_equals_fibonacci_l100_100698

noncomputable def fibonacci : ℕ → ℕ
| 0           => 0
| 1           => 1
| (n + 2)     => fibonacci (n + 1) + fibonacci n

noncomputable def selfish_subsets_count (n : ℕ) : ℕ := 
sorry -- This will be replaced with the correct recursive function

theorem selfish_subsets_equals_fibonacci (n : ℕ) : 
  selfish_subsets_count n = fibonacci n :=
sorry

end selfish_subsets_equals_fibonacci_l100_100698


namespace degenerate_ellipse_value_c_l100_100689

theorem degenerate_ellipse_value_c (c : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0) ∧
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0 → (x+1)^2 + (y-7)^2 = 0) ↔ c = 52 :=
by
  sorry

end degenerate_ellipse_value_c_l100_100689


namespace book_pages_total_l100_100932

-- Define the conditions as hypotheses
def total_pages (P : ℕ) : Prop :=
  let read_first_day := P / 2
  let read_second_day := P / 4
  let read_third_day := P / 6
  let read_total := read_first_day + read_second_day + read_third_day
  let remaining_pages := P - read_total
  remaining_pages = 20

-- The proof statement
theorem book_pages_total (P : ℕ) (h : total_pages P) : P = 240 := sorry

end book_pages_total_l100_100932


namespace combined_mass_of_individuals_l100_100354

-- Define constants and assumptions
def boat_length : ℝ := 4 -- in meters
def boat_breadth : ℝ := 3 -- in meters
def sink_depth_first_person : ℝ := 0.01 -- in meters (1 cm)
def sink_depth_second_person : ℝ := 0.02 -- in meters (2 cm)
def density_water : ℝ := 1000 -- in kg/m³ (density of freshwater)

-- Define volumes displaced
def volume_displaced_first : ℝ := boat_length * boat_breadth * sink_depth_first_person
def volume_displaced_both : ℝ := boat_length * boat_breadth * (sink_depth_first_person + sink_depth_second_person)

-- Define weights (which are equal to the masses under the assumption of constant gravity)
def weight_first_person : ℝ := volume_displaced_first * density_water
def weight_both_persons : ℝ := volume_displaced_both * density_water

-- Statement to prove the combined weight
theorem combined_mass_of_individuals : weight_both_persons = 360 :=
by
  -- Skip the proof
  sorry

end combined_mass_of_individuals_l100_100354


namespace problem_1_problem_2_l100_100560

-- Problem (1)
theorem problem_1 (x a : ℝ) (h_a : a = 1) (hP : x^2 - 4*a*x + 3*a^2 < 0) (hQ1 : x^2 - x - 6 ≤ 0) (hQ2 : x^2 + 2*x - 8 > 0) :
  2 < x ∧ x < 3 := sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h_a_pos : 0 < a) (h_suff_neccess : (¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) → ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) ∧
                   ¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) ≠ ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) :
  1 < a ∧ a ≤ 2 := sorry

end problem_1_problem_2_l100_100560


namespace guard_team_size_l100_100499

theorem guard_team_size (b n s : ℕ) (h_total : b * s * n = 1001) (h_condition : s < n ∧ n < b) : s = 7 := 
by
  sorry

end guard_team_size_l100_100499


namespace distance_of_points_in_polar_coordinates_l100_100394

theorem distance_of_points_in_polar_coordinates
  (A : Real × Real) (B : Real × Real) (θ1 θ2 : Real)
  (hA : A = (5, θ1)) (hB : B = (12, θ2))
  (hθ : θ1 - θ2 = Real.pi / 2) : 
  dist (5 * Real.cos θ1, 5 * Real.sin θ1) (12 * Real.cos θ2, 12 * Real.sin θ2) = 13 := 
by sorry

end distance_of_points_in_polar_coordinates_l100_100394


namespace avg_speed_is_65_l100_100019

theorem avg_speed_is_65
  (speed1: ℕ) (speed2: ℕ) (time1: ℕ) (time2: ℕ)
  (h_speed1: speed1 = 85)
  (h_speed2: speed2 = 45)
  (h_time1: time1 = 1)
  (h_time2: time2 = 1) :
  (speed1 + speed2) / (time1 + time2) = 65 := by
  sorry

end avg_speed_is_65_l100_100019


namespace eggs_leftover_l100_100807

theorem eggs_leftover (eggs_abigail eggs_beatrice eggs_carson cartons : ℕ)
  (h_abigail : eggs_abigail = 37)
  (h_beatrice : eggs_beatrice = 49)
  (h_carson : eggs_carson = 14)
  (h_cartons : cartons = 12) :
  ((eggs_abigail + eggs_beatrice + eggs_carson) % cartons) = 4 :=
by
  sorry

end eggs_leftover_l100_100807


namespace profit_starts_from_third_year_most_beneficial_option_l100_100049

-- Define the conditions of the problem
def investment_cost := 144
def maintenance_cost (n : ℕ) := 4 * n^2 + 20 * n
def revenue_per_year := 1

-- Define the net profit function
def net_profit (n : ℕ) : ℤ :=
(revenue_per_year * n : ℤ) - (maintenance_cost n) - investment_cost

-- Question 1: Prove the project starts to make a profit from the 3rd year
theorem profit_starts_from_third_year (n : ℕ) (h : 2 < n ∧ n < 18) : 
net_profit n > 0 ↔ 3 ≤ n := sorry

-- Question 2: Prove the most beneficial option for company's development
theorem most_beneficial_option : (∃ o, o = 1) ∧ (∃ t1 t2, t1 = 264 ∧ t2 = 264 ∧ t1 < t2) := sorry

end profit_starts_from_third_year_most_beneficial_option_l100_100049


namespace quadratic_inequality_solution_l100_100837

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ ax^2 + bx + c > 0) :
  ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - bx + c > 0 := 
sorry

end quadratic_inequality_solution_l100_100837


namespace walking_rate_on_escalator_l100_100013

/-- If the escalator moves at 7 feet per second, is 180 feet long, and a person takes 20 seconds to cover this length, then the rate at which the person walks on the escalator is 2 feet per second. -/
theorem walking_rate_on_escalator 
  (escalator_rate : ℝ)
  (length : ℝ)
  (time : ℝ)
  (v : ℝ)
  (h_escalator_rate : escalator_rate = 7)
  (h_length : length = 180)
  (h_time : time = 20)
  (h_distance_formula : length = (v + escalator_rate) * time) :
  v = 2 :=
by
  sorry

end walking_rate_on_escalator_l100_100013


namespace quadratic_roots_integer_sum_eq_198_l100_100529

theorem quadratic_roots_integer_sum_eq_198 (x p q x1 x2 : ℤ) 
  (h_eqn : x^2 + p * x + q = 0)
  (h_roots : (x - x1) * (x - x2) = 0)
  (h_pq_sum : p + q = 198) :
  (x1 = 2 ∧ x2 = 200) ∨ (x1 = 0 ∧ x2 = -198) :=
sorry

end quadratic_roots_integer_sum_eq_198_l100_100529


namespace ocean_depth_l100_100905

theorem ocean_depth (t : ℕ) (v : ℕ) (h : ℕ)
  (h_t : t = 8)
  (h_v : v = 1500) :
  h = 6000 :=
by
  sorry

end ocean_depth_l100_100905


namespace smallest_n_for_terminating_decimal_l100_100798

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l100_100798


namespace find_length_BF_l100_100223

-- Define the conditions
structure Rectangle :=
  (short_side : ℝ)
  (long_side : ℝ)

def folded_paper (rect : Rectangle) : Prop :=
  rect.short_side = 12

def congruent_triangles (rect : Rectangle) : Prop :=
  rect.short_side = 12

-- Define the length of BF to prove
def length_BF (rect : Rectangle) : ℝ := 10

-- The theorem statement
theorem find_length_BF (rect : Rectangle) (h1 : folded_paper rect) (h2 : congruent_triangles rect) :
  length_BF rect = 10 := 
  sorry

end find_length_BF_l100_100223


namespace find_days_jane_indisposed_l100_100720

-- Define the problem conditions
def John_rate := 1 / 20
def Jane_rate := 1 / 10
def together_rate := John_rate + Jane_rate
def total_task := 1
def total_days := 10

-- The time Jane was indisposed
def days_jane_indisposed (x : ℝ) : Prop :=
  (total_days - x) * together_rate + x * John_rate = total_task

-- Statement we want to prove
theorem find_days_jane_indisposed : ∃ x : ℝ, days_jane_indisposed x ∧ x = 5 :=
by 
  sorry

end find_days_jane_indisposed_l100_100720


namespace arthur_walk_distance_l100_100050

def blocks_east : ℕ := 8
def blocks_north : ℕ := 15
def block_length : ℚ := 1 / 4

theorem arthur_walk_distance :
  (blocks_east + blocks_north) * block_length = 23 * (1 / 4) := by
  sorry

end arthur_walk_distance_l100_100050


namespace urea_formation_l100_100324

theorem urea_formation
  (CO2 NH3 Urea : ℕ) 
  (h_CO2 : CO2 = 1)
  (h_NH3 : NH3 = 2) :
  Urea = 1 := by
  sorry

end urea_formation_l100_100324


namespace recycling_program_earnings_l100_100572

-- Define conditions
def signup_earning : ℝ := 5.00
def referral_earning_tier1 : ℝ := 8.00
def referral_earning_tier2 : ℝ := 1.50
def friend_earning_signup : ℝ := 5.00
def friend_earning_tier2 : ℝ := 2.00

def initial_friend_count : ℕ := 5
def initial_friend_tier1_referrals_day1 : ℕ := 3
def initial_friend_tier1_referrals_week : ℕ := 2

def additional_friend_count : ℕ := 2
def additional_friend_tier1_referrals : ℕ := 1

-- Calculate Katrina's total earnings
def katrina_earnings : ℝ :=
  signup_earning +
  (initial_friend_count * referral_earning_tier1) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * referral_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * referral_earning_tier2) +
  (additional_friend_count * referral_earning_tier1) +
  (additional_friend_count * additional_friend_tier1_referrals * referral_earning_tier2)

-- Calculate friends' total earnings
def friends_earnings : ℝ :=
  (initial_friend_count * friend_earning_signup) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * friend_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * friend_earning_tier2) +
  (additional_friend_count * friend_earning_signup) +
  (additional_friend_count * additional_friend_tier1_referrals * friend_earning_tier2)

-- Calculate combined total earnings
def combined_earnings : ℝ := katrina_earnings + friends_earnings

-- The proof assertion
theorem recycling_program_earnings : combined_earnings = 190.50 :=
by sorry

end recycling_program_earnings_l100_100572


namespace buses_dispatched_theorem_l100_100563

-- Define the conditions and parameters
def buses_dispatched (buses: ℕ) (hours: ℕ) : ℕ :=
  buses * hours

-- Define the specific problem
noncomputable def buses_from_6am_to_4pm : ℕ :=
  let buses_per_hour := 5 / 2
  let hours         := 16 - 6
  buses_dispatched (buses_per_hour : ℕ) hours

-- State the theorem that needs to be proven
theorem buses_dispatched_theorem : buses_from_6am_to_4pm = 25 := 
by {
  -- This 'sorry' is a placeholder for the actual proof.
  sorry
}

end buses_dispatched_theorem_l100_100563


namespace no_integer_triplets_for_equation_l100_100865

theorem no_integer_triplets_for_equation (a b c : ℤ) : ¬ (a^2 + b^2 + 1 = 4 * c) :=
by
  sorry

end no_integer_triplets_for_equation_l100_100865


namespace shaded_area_l100_100362

/--
Given a larger square containing a smaller square entirely within it,
where the side length of the smaller square is 5 units
and the side length of the larger square is 10 units,
prove that the area of the shaded region (the area of the larger square minus the area of the smaller square) is 75 square units.
-/
theorem shaded_area :
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  area_larger - area_smaller = 75 := 
by
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  sorry

end shaded_area_l100_100362


namespace molecular_weight_of_7_moles_of_NH4_2SO4_l100_100978

theorem molecular_weight_of_7_moles_of_NH4_2SO4 :
  let N_weight := 14.01
  let H_weight := 1.01
  let S_weight := 32.07
  let O_weight := 16.00
  let N_atoms := 2
  let H_atoms := 8
  let S_atoms := 1
  let O_atoms := 4
  let moles := 7
  let molecular_weight := (N_weight * N_atoms) + (H_weight * H_atoms) + (S_weight * S_atoms) + (O_weight * O_atoms)
  let total_weight := molecular_weight * moles
  total_weight = 924.19 :=
by
  sorry

end molecular_weight_of_7_moles_of_NH4_2SO4_l100_100978


namespace math_problem_l100_100198

noncomputable def sqrt180 : ℝ := Real.sqrt 180
noncomputable def two_thirds_sqrt180 : ℝ := (2 / 3) * sqrt180
noncomputable def forty_percent_300_cubed : ℝ := (0.4 * 300)^3
noncomputable def forty_percent_180 : ℝ := 0.4 * 180
noncomputable def one_third_less_forty_percent_180 : ℝ := forty_percent_180 - (1 / 3) * forty_percent_180

theorem math_problem : 
  (two_thirds_sqrt180 * forty_percent_300_cubed) - one_third_less_forty_percent_180 = 15454377.6 :=
  by
    have h1 : sqrt180 = Real.sqrt 180 := rfl
    have h2 : two_thirds_sqrt180 = (2 / 3) * sqrt180 := rfl
    have h3 : forty_percent_300_cubed = (0.4 * 300)^3 := rfl
    have h4 : forty_percent_180 = 0.4 * 180 := rfl
    have h5 : one_third_less_forty_percent_180 = forty_percent_180 - (1 / 3) * forty_percent_180 := rfl
    sorry

end math_problem_l100_100198


namespace area_difference_l100_100026

theorem area_difference (T_area : ℝ) (omega_area : ℝ) (H1 : T_area = (25 * Real.sqrt 3) / 4) 
  (H2 : omega_area = 4 * Real.pi) (H3 : 3 * (X - Y) = T_area - omega_area) :
  X - Y = (25 * Real.sqrt 3) / 12 - (4 * Real.pi) / 3 :=
by 
  sorry

end area_difference_l100_100026


namespace probability_heads_twice_in_three_flips_l100_100422

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_heads_twice_in_three_flips :
  let p := 0.5
  let n := 3
  let k := 2
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end probability_heads_twice_in_three_flips_l100_100422


namespace expected_accidents_no_overtime_l100_100649

noncomputable def accidents_with_no_overtime_hours 
    (hours1 hours2 : ℕ) (accidents1 accidents2 : ℕ) : ℕ :=
  let slope := (accidents2 - accidents1) / (hours2 - hours1)
  let intercept := accidents1 - slope * hours1
  intercept

theorem expected_accidents_no_overtime : 
    accidents_with_no_overtime_hours 1000 400 8 5 = 3 :=
by
  sorry

end expected_accidents_no_overtime_l100_100649


namespace sum_medians_less_than_perimeter_l100_100016

noncomputable def median_a (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * b^2 + 2 * c^2 - a^2).sqrt

noncomputable def median_b (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * c^2 - b^2).sqrt

noncomputable def median_c (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * b^2 - c^2).sqrt

noncomputable def sum_of_medians (a b c : ℝ) : ℝ :=
  median_a a b c + median_b a b c + median_c a b c

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  perimeter a b c / 2

theorem sum_medians_less_than_perimeter (a b c : ℝ) :
  semiperimeter a b c < sum_of_medians a b c ∧ sum_of_medians a b c < perimeter a b c :=
by
  sorry

end sum_medians_less_than_perimeter_l100_100016


namespace history_students_count_l100_100632

theorem history_students_count
  (total_students : ℕ)
  (sample_students : ℕ)
  (physics_students_sampled : ℕ)
  (history_students_sampled : ℕ)
  (x : ℕ)
  (H1 : total_students = 1500)
  (H2 : sample_students = 120)
  (H3 : physics_students_sampled = 80)
  (H4 : history_students_sampled = sample_students - physics_students_sampled)
  (H5 : x = 1500 * history_students_sampled / sample_students) :
  x = 500 :=
by
  sorry

end history_students_count_l100_100632


namespace total_combinations_l100_100035

/-- Tim's rearrangement choices for the week -/
def monday_choices : Nat := 1
def tuesday_choices : Nat := 2
def wednesday_choices : Nat := 3
def thursday_choices : Nat := 2
def friday_choices : Nat := 1

theorem total_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 12 :=
by
  sorry

end total_combinations_l100_100035


namespace sum_of_possible_values_l100_100624

theorem sum_of_possible_values (x : ℝ) :
  (x + 3) * (x - 4) = 20 →
  ∃ a b, (a ≠ b) ∧ 
         ((x = a) ∨ (x = b)) ∧ 
         (x^2 - x - 32 = 0) ∧ 
         (a + b = 1) :=
by
  sorry

end sum_of_possible_values_l100_100624


namespace find_a_l100_100436

theorem find_a (a : ℝ) (h : ∃ x : ℝ, x = 2 ∧ x^2 + a * x - 2 = 0) : a = -1 := 
by 
  sorry

end find_a_l100_100436


namespace not_prime_257_1092_1092_l100_100145

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_257_1092_1092 :
  is_prime 1093 →
  ¬ is_prime (257 ^ 1092 + 1092) :=
by
  intro h_prime_1093
  -- Detailed steps are omitted, proof goes here
  sorry

end not_prime_257_1092_1092_l100_100145


namespace sum_of_discount_rates_l100_100268

theorem sum_of_discount_rates : 
  let fox_price := 15
  let pony_price := 20
  let fox_pairs := 3
  let pony_pairs := 2
  let total_savings := 9
  let pony_discount := 18.000000000000014
  let fox_discount := 4
  let total_discount_rate := fox_discount + pony_discount
  total_discount_rate = 22.000000000000014 := by
sorry

end sum_of_discount_rates_l100_100268


namespace solve_abs_inequality_l100_100086

/-- Given the inequality 2 ≤ |x - 3| ≤ 8, we want to prove that the solution is [-5 ≤ x ≤ 1] ∪ [5 ≤ x ≤ 11] --/
theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
sorry

end solve_abs_inequality_l100_100086


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l100_100033

variable (U A B : Set ℝ)
variable (x : ℝ)

def universal_set := { x | x ≤ 4 }
def set_A := { x | -2 < x ∧ x < 3 }
def set_B := { x | -3 < x ∧ x ≤ 3 }

theorem complement_U_A : (U \ A) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem intersection_A_B : (A ∩ B) = { x | -2 < x ∧ x < 3 } := sorry

theorem complement_U_intersection_A_B : (U \ (A ∩ B)) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem complement_U_A_intersection_B : ((U \ A) ∩ B) = { x | -3 < x ∧ x ≤ -2 ∨ x = 3 } := sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l100_100033


namespace valid_sequences_count_l100_100331

noncomputable def number_of_valid_sequences
(strings : List (List Nat))
(ball_A_shot : Nat)
(ball_B_shot : Nat) : Nat := 144

theorem valid_sequences_count :
  let strings := [[1, 2], [3, 4, 5], [6, 7, 8, 9]];
  let ball_A := 1;  -- Assuming A is the first ball in the first string
  let ball_B := 3;  -- Assuming B is the first ball in the second string
  ball_A = 1 →
  ball_B = 3 →
  ball_A_shot = 5 →
  ball_B_shot = 6 →
  number_of_valid_sequences strings ball_A_shot ball_B_shot = 144 :=
by
  intros strings ball_A ball_B hA hB hAShot hBShot
  sorry

end valid_sequences_count_l100_100331


namespace verify_Fermat_point_l100_100208

open Real

theorem verify_Fermat_point :
  let D := (0, 0)
  let E := (6, 4)
  let F := (3, -2)
  let Q := (2, 1)
  let distance (P₁ P₂ : ℝ × ℝ) : ℝ := sqrt ((P₂.1 - P₁.1)^2 + (P₂.2 - P₁.2)^2)
  distance D Q + distance E Q + distance F Q = 5 + sqrt 5 + sqrt 10 := by
sorry

end verify_Fermat_point_l100_100208


namespace expr1_correct_expr2_correct_expr3_correct_l100_100961

-- Define the expressions and corresponding correct answers
def expr1 : Int := 58 + 15 * 4
def expr2 : Int := 216 - 72 / 8
def expr3 : Int := (358 - 295) / 7

-- State the proof goals
theorem expr1_correct : expr1 = 118 := by
  sorry

theorem expr2_correct : expr2 = 207 := by
  sorry

theorem expr3_correct : expr3 = 9 := by
  sorry

end expr1_correct_expr2_correct_expr3_correct_l100_100961


namespace floor_ineq_l100_100959

theorem floor_ineq (α β : ℝ) : ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ :=
sorry

end floor_ineq_l100_100959


namespace coin_stack_count_l100_100271

theorem coin_stack_count
  (TN : ℝ := 1.95)
  (TQ : ℝ := 1.75)
  (SH : ℝ := 20)
  (n q : ℕ) :
  (n*Tℕ + q*TQ = SH) → (n + q = 10) :=
sorry

end coin_stack_count_l100_100271


namespace n_power_of_3_l100_100336

theorem n_power_of_3 (n : ℕ) (h_prime : Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end n_power_of_3_l100_100336


namespace linear_combination_harmonic_l100_100582

-- Define the harmonic property for a function
def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

-- The main statement to be proven in Lean
theorem linear_combination_harmonic
  (f g : ℤ × ℤ → ℝ) (a b : ℝ) (hf : is_harmonic f) (hg : is_harmonic g) :
  is_harmonic (fun p => a * f p + b * g p) :=
by
  sorry

end linear_combination_harmonic_l100_100582


namespace correct_operation_l100_100173

theorem correct_operation : ∀ (a b : ℤ), 3 * a^2 * b - 2 * b * a^2 = a^2 * b :=
by
  sorry

end correct_operation_l100_100173


namespace quadratic_inequality_solution_l100_100683

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (3 * x^2 - 5 * x - 2 < 0) ↔ (-1/3 < x ∧ x < 2) :=
by
  sorry

end quadratic_inequality_solution_l100_100683


namespace parabola_translation_eq_l100_100018

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := - (x - 2)^2 - 1

-- State the theorem to prove the translated function
theorem parabola_translation_eq :
  ∀ x : ℝ, translated_parabola x = - (x - 2)^2 - 1 :=
by
  sorry

end parabola_translation_eq_l100_100018


namespace mary_total_nickels_l100_100286

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l100_100286


namespace costco_container_holds_one_gallon_l100_100380

theorem costco_container_holds_one_gallon
  (costco_cost : ℕ := 8)
  (store_cost_per_bottle : ℕ := 3)
  (savings : ℕ := 16)
  (ounces_per_bottle : ℕ := 16)
  (ounces_per_gallon : ℕ := 128) :
  ∃ (gallons : ℕ), gallons = 1 :=
by
  sorry

end costco_container_holds_one_gallon_l100_100380


namespace pairball_playing_time_l100_100404

-- Define the conditions of the problem
def num_children : ℕ := 7
def total_minutes : ℕ := 105
def total_child_minutes : ℕ := 2 * total_minutes

-- Define the theorem to prove
theorem pairball_playing_time : total_child_minutes / num_children = 30 :=
by sorry

end pairball_playing_time_l100_100404


namespace degenerate_ellipse_single_point_l100_100376

theorem degenerate_ellipse_single_point (c : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → (x = -1 ∧ y = 6)) ↔ c = -39 :=
by
  sorry

end degenerate_ellipse_single_point_l100_100376


namespace shyne_total_plants_l100_100645

/-- Shyne's seed packets -/
def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10

/-- Seed packets purchased by Shyne -/
def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6

/-- Total number of plants grown by Shyne -/
def total_plants : ℕ := 116

theorem shyne_total_plants :
  eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = total_plants :=
by
  sorry

end shyne_total_plants_l100_100645


namespace desired_average_sale_l100_100195

def s1 := 2500
def s2 := 4000
def s3 := 3540
def s4 := 1520
def avg := 2890

theorem desired_average_sale : (s1 + s2 + s3 + s4) / 4 = avg := by
  sorry

end desired_average_sale_l100_100195


namespace holiday_customers_l100_100491

-- Define the normal rate of customers entering the store (175 people/hour)
def normal_rate : ℕ := 175

-- Define the holiday rate of customers entering the store
def holiday_rate : ℕ := 2 * normal_rate

-- Define the duration for which we are calculating the total number of customers (8 hours)
def duration : ℕ := 8

-- Define the correct total number of customers (2800 people)
def correct_total_customers : ℕ := 2800

-- The theorem that asserts the total number of customers in 8 hours during the holiday season is 2800
theorem holiday_customers : holiday_rate * duration = correct_total_customers := by
  sorry

end holiday_customers_l100_100491


namespace perpendicular_vectors_l100_100501

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l100_100501


namespace pyramid_base_sidelength_l100_100352

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l100_100352


namespace simplify_expression_l100_100926

theorem simplify_expression (a b : ℚ) : (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 := 
by 
  sorry

end simplify_expression_l100_100926


namespace percent_of_y_eq_l100_100472

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l100_100472


namespace sean_final_cost_l100_100067

noncomputable def totalCost (sodaCount soupCount sandwichCount saladCount : ℕ)
                            (pricePerSoda pricePerSoup pricePerSandwich pricePerSalad : ℚ)
                            (discountRate taxRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount := (sodaCount * pricePerSoda) +
                                (soupCount * pricePerSoup) +
                                (sandwichCount * pricePerSandwich) +
                                (saladCount * pricePerSalad)
  let discountedTotal := totalCostBeforeDiscount * (1 - discountRate)
  let finalCost := discountedTotal * (1 + taxRate)
  finalCost

theorem sean_final_cost : 
  totalCost 4 3 2 1 
            1 (2 * 1) (4 * (2 * 1)) (2 * (4 * (2 * 1)))
            0.1 0.05 = 39.69 := 
by
  sorry

end sean_final_cost_l100_100067


namespace line_y_axis_intersection_l100_100315

-- Conditions: Line contains points (3, 20) and (-9, -6)
def line_contains_points : Prop :=
  ∃ m b : ℚ, ∀ (x y : ℚ), ((x = 3 ∧ y = 20) ∨ (x = -9 ∧ y = -6)) → (y = m * x + b)

-- Question: Prove that the line intersects the y-axis at (0, 27/2)
theorem line_y_axis_intersection :
  line_contains_points → (∃ (y : ℚ), y = 27/2) :=
by
  sorry

end line_y_axis_intersection_l100_100315


namespace find_a_if_f_is_even_l100_100359

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 2 := by
  sorry

end find_a_if_f_is_even_l100_100359


namespace tutors_schedule_l100_100097

theorem tutors_schedule :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_schedule_l100_100097


namespace angle_of_inclination_of_line_l100_100638

theorem angle_of_inclination_of_line (x y : ℝ) (h : x - y - 1 = 0) : 
  ∃ α : ℝ, α = π / 4 := 
sorry

end angle_of_inclination_of_line_l100_100638


namespace minimum_value_f_is_correct_l100_100160

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l100_100160


namespace minimum_apples_l100_100355

theorem minimum_apples (n : ℕ) : 
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 9 = 7 → n = 97 := 
by 
  -- To be proved
  sorry

end minimum_apples_l100_100355


namespace parallel_line_slope_l100_100872

theorem parallel_line_slope (a b c : ℝ) (m : ℝ) :
  (5 * a + 10 * b = -35) →
  (∃ m : ℝ, b = m * a + c) →
  m = -1/2 :=
by sorry

end parallel_line_slope_l100_100872


namespace jog_to_coffee_shop_l100_100530

def constant_pace_jogging (time_to_park : ℕ) (dist_to_park : ℝ) (dist_to_coffee_shop : ℝ) : Prop :=
  time_to_park / dist_to_park * dist_to_coffee_shop = 6

theorem jog_to_coffee_shop
  (time_to_park : ℕ)
  (dist_to_park : ℝ)
  (dist_to_coffee_shop : ℝ)
  (h1 : time_to_park = 12)
  (h2 : dist_to_park = 1.5)
  (h3 : dist_to_coffee_shop = 0.75)
: constant_pace_jogging time_to_park dist_to_park dist_to_coffee_shop :=
by sorry

end jog_to_coffee_shop_l100_100530


namespace rectangle_height_l100_100466

-- Defining the conditions
def base : ℝ := 9
def area : ℝ := 33.3

-- Stating the proof problem
theorem rectangle_height : (area / base) = 3.7 :=
by
  sorry

end rectangle_height_l100_100466


namespace zoo_camels_l100_100962

theorem zoo_camels (x y : ℕ) (h1 : x - y = 10) (h2 : x + 2 * y = 55) : x + y = 40 :=
by sorry

end zoo_camels_l100_100962


namespace num_distinct_x_intercepts_l100_100516

def f (x : ℝ) : ℝ := (x - 5) * (x^3 + 5*x^2 + 9*x + 9)

theorem num_distinct_x_intercepts : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2) :=
sorry

end num_distinct_x_intercepts_l100_100516


namespace value_of_f_g3_l100_100863

def g (x : ℝ) : ℝ := 4 * x - 5
def f (x : ℝ) : ℝ := 6 * x + 11

theorem value_of_f_g3 : f (g 3) = 53 := by
  sorry

end value_of_f_g3_l100_100863


namespace diamond_value_l100_100709

def diamond (a b : Int) : Int :=
  a * b^2 - b + 1

theorem diamond_value : diamond (-1) 6 = -41 := by
  sorry

end diamond_value_l100_100709


namespace sequence_induction_l100_100103

theorem sequence_induction (a b : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : b 1 = 4)
  (h₃ : ∀ n : ℕ, 0 < n → 2 * b n = a n + a (n + 1))
  (h₄ : ∀ n : ℕ, 0 < n → (a (n + 1))^2 = b n * b (n + 1)) :
  (∀ n : ℕ, 0 < n → a n = n * (n + 1)) ∧ (∀ n : ℕ, 0 < n → b n = (n + 1)^2) :=
by
  sorry

end sequence_induction_l100_100103


namespace smallest_sum_of_squares_l100_100358

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l100_100358


namespace points_enclosed_in_circle_l100_100125

open Set

variable (points : Set (ℝ × ℝ))
variable (radius : ℝ)
variable (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
  ∃ (c : ℝ × ℝ), dist c A ≤ radius ∧ dist c B ≤ radius ∧ dist c C ≤ radius)

theorem points_enclosed_in_circle
  (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points →
    ∃ (c : ℝ × ℝ), dist c A ≤ 1 ∧ dist c B ≤ 1 ∧ dist c C ≤ 1) :
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ 1 :=
sorry

end points_enclosed_in_circle_l100_100125


namespace sum_of_possible_values_of_g_l100_100915

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (x : ℝ) : ℝ := 3 * x - 4

theorem sum_of_possible_values_of_g :
  let x1 := (9 + 3 * Real.sqrt 5) / 2
  let x2 := (9 - 3 * Real.sqrt 5) / 2
  g x1 + g x2 = 19 :=
by
  sorry

end sum_of_possible_values_of_g_l100_100915


namespace calculate_dividend_l100_100119

def faceValue : ℕ := 100
def premiumPercent : ℕ := 20
def dividendPercent : ℕ := 5
def investment : ℕ := 14400
def costPerShare : ℕ := faceValue + (premiumPercent * faceValue / 100)
def numberOfShares : ℕ := investment / costPerShare
def dividendPerShare : ℕ := faceValue * dividendPercent / 100
def totalDividend : ℕ := numberOfShares * dividendPerShare

theorem calculate_dividend :
  totalDividend = 600 := 
by
  sorry

end calculate_dividend_l100_100119


namespace remainder_5_pow_100_div_18_l100_100695

theorem remainder_5_pow_100_div_18 : (5 ^ 100) % 18 = 13 := 
  sorry

end remainder_5_pow_100_div_18_l100_100695


namespace find_x_l100_100793

theorem find_x (x : ℝ) :
  (x * 13.26 + x * 9.43 + x * 77.31 = 470) → (x = 4.7) :=
by
  sorry

end find_x_l100_100793


namespace rectangular_x_value_l100_100606

theorem rectangular_x_value (x : ℝ)
  (h1 : ∀ (length : ℝ), length = 4 * x)
  (h2 : ∀ (width : ℝ), width = x + 10)
  (h3 : ∀ (length width : ℝ), length * width = 2 * (2 * length + 2 * width))
  : x = (Real.sqrt 41 - 1) / 2 :=
by
  sorry

end rectangular_x_value_l100_100606


namespace intersection_polar_coords_l100_100363

noncomputable def polar_coord_intersection (rho theta : ℝ) : Prop :=
  (rho * (Real.sqrt 3 * Real.cos theta - Real.sin theta) = 2) ∧ (rho = 4 * Real.sin theta)

theorem intersection_polar_coords :
  ∃ (rho theta : ℝ), polar_coord_intersection rho theta ∧ rho = 2 ∧ theta = (Real.pi / 6) := 
sorry

end intersection_polar_coords_l100_100363


namespace eldest_boy_age_l100_100092

theorem eldest_boy_age (a b c : ℕ) (h1 : a + b + c = 45) (h2 : 3 * c = 7 * a) (h3 : 5 * c = 7 * b) : c = 21 := 
sorry

end eldest_boy_age_l100_100092


namespace larger_number_225_l100_100473

theorem larger_number_225 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a - b = 120) 
  (h4 : Nat.lcm a b = 105 * Nat.gcd a b) : 
  max a b = 225 :=
by
  sorry

end larger_number_225_l100_100473


namespace tank_never_fills_l100_100533

structure Pipe :=
(rate1 : ℕ) (rate2 : ℕ)

def net_flow (pA pB pC pD : Pipe) (time1 time2 : ℕ) : ℤ :=
  let fillA := pA.rate1 * time1 + pA.rate2 * time2
  let fillB := pB.rate1 * time1 + pB.rate2 * time2
  let drainC := pC.rate1 * time1 + pC.rate2 * time2
  let drainD := pD.rate1 * (time1 + time2)
  (fillA + fillB) - (drainC + drainD)

theorem tank_never_fills (pA pB pC pD : Pipe) (time1 time2 : ℕ)
  (hA : pA = Pipe.mk 40 20) (hB : pB = Pipe.mk 20 40) 
  (hC : pC = Pipe.mk 20 40) (hD : pD = Pipe.mk 30 30) 
  (hTime : time1 = 30 ∧ time2 = 30): 
  net_flow pA pB pC pD time1 time2 = 0 := by
  sorry

end tank_never_fills_l100_100533


namespace pyramid_volume_eq_l100_100118

noncomputable def volume_of_pyramid (base_length1 base_length2 height : ℝ) : ℝ :=
  (1 / 3) * base_length1 * base_length2 * height

theorem pyramid_volume_eq (base_length1 base_length2 height : ℝ) (h1 : base_length1 = 1) (h2 : base_length2 = 2) (h3 : height = 1) :
  volume_of_pyramid base_length1 base_length2 height = 2 / 3 := by
  sorry

end pyramid_volume_eq_l100_100118


namespace find_ordered_pair_l100_100099

-- We need to define the variables and conditions first.
variables (a c : ℝ)

-- Now we state the conditions.
def quadratic_has_one_solution : Prop :=
  a * c = 25 ∧ a + c = 12 ∧ a < c

-- Finally, we state the main goal to prove.
theorem find_ordered_pair (ha : quadratic_has_one_solution a c) :
  a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 :=
by sorry

end find_ordered_pair_l100_100099


namespace fraction_problem_l100_100328

-- Definitions of x and y based on the given conditions
def x : ℚ := 3 / 5
def y : ℚ := 7 / 9

-- The theorem stating the mathematical equivalence to be proven
theorem fraction_problem : (5 * x + 9 * y) / (45 * x * y) = 10 / 21 :=
by
  sorry

end fraction_problem_l100_100328


namespace sum_of_ages_l100_100808

theorem sum_of_ages (M C : ℝ) (h1 : M = C + 12) (h2 : M + 10 = 3 * (C - 6)) : M + C = 52 :=
by
  sorry

end sum_of_ages_l100_100808


namespace chloe_at_least_85_nickels_l100_100700

-- Define the given values
def shoe_cost : ℝ := 45.50
def ten_dollars : ℝ := 10.0
def num_ten_dollar_bills : ℕ := 4
def quarter_value : ℝ := 0.25
def num_quarters : ℕ := 5
def nickel_value : ℝ := 0.05

-- Define the statement to be proved
theorem chloe_at_least_85_nickels (n : ℕ) 
  (H1 : shoe_cost = 45.50)
  (H2 : ten_dollars = 10.0)
  (H3 : num_ten_dollar_bills = 4)
  (H4 : quarter_value = 0.25)
  (H5 : num_quarters = 5)
  (H6 : nickel_value = 0.05) :
  4 * ten_dollars + 5 * quarter_value + n * nickel_value >= shoe_cost → n >= 85 :=
by {
  sorry
}

end chloe_at_least_85_nickels_l100_100700


namespace dogs_in_shelter_l100_100692

theorem dogs_in_shelter (D C : ℕ) (h1 : D * 7 = 15 * C) (h2 : D * 11 = 15 * (C + 8)) :
  D = 30 :=
sorry

end dogs_in_shelter_l100_100692


namespace determine_angle_F_l100_100201

noncomputable def sin := fun x => Real.sin x
noncomputable def cos := fun x => Real.cos x
noncomputable def arcsin := fun x => Real.arcsin x
noncomputable def angleF (D E : ℝ) := 180 - (D + E)

theorem determine_angle_F (D E F : ℝ)
  (h1 : 2 * sin D + 5 * cos E = 7)
  (h2 : 5 * sin E + 2 * cos D = 4) :
  F = arcsin (9 / 10) ∨ F = 180 - arcsin (9 / 10) :=
  sorry

end determine_angle_F_l100_100201


namespace total_time_correct_l100_100196

-- Define the base speeds and distance
def speed_boat : ℕ := 8
def speed_stream : ℕ := 6
def distance : ℕ := 210

-- Define the speeds downstream and upstream
def speed_downstream : ℕ := speed_boat + speed_stream
def speed_upstream : ℕ := speed_boat - speed_stream

-- Define the time taken for downstream and upstream
def time_downstream : ℕ := distance / speed_downstream
def time_upstream : ℕ := distance / speed_upstream

-- Define the total time taken
def total_time : ℕ := time_downstream + time_upstream

-- The theorem to be proven
theorem total_time_correct : total_time = 120 := by
  sorry

end total_time_correct_l100_100196


namespace minimum_value_is_two_sqrt_two_l100_100167

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sqrt (x^2 + (2 - x)^2)) + (Real.sqrt ((2 - x)^2 + x^2))

theorem minimum_value_is_two_sqrt_two :
  ∃ x : ℝ, minimum_value_expression x = 2 * Real.sqrt 2 :=
by 
  sorry

end minimum_value_is_two_sqrt_two_l100_100167


namespace yellow_shirts_count_l100_100327

theorem yellow_shirts_count (total_shirts blue_shirts green_shirts red_shirts yellow_shirts : ℕ) 
  (h1 : total_shirts = 36) 
  (h2 : blue_shirts = 8) 
  (h3 : green_shirts = 11) 
  (h4 : red_shirts = 6) 
  (h5 : yellow_shirts = total_shirts - (blue_shirts + green_shirts + red_shirts)) :
  yellow_shirts = 11 :=
by
  sorry

end yellow_shirts_count_l100_100327


namespace final_state_of_marbles_after_operations_l100_100676

theorem final_state_of_marbles_after_operations :
  ∃ (b w : ℕ), b + w = 2 ∧ w = 2 ∧ (∀ n : ℕ, n % 2 = 0 → n = 100 - k * 2) :=
sorry

end final_state_of_marbles_after_operations_l100_100676


namespace integers_sum_eighteen_l100_100680

theorem integers_sum_eighteen (a b : ℕ) (h₀ : a ≠ b) (h₁ : a < 20) (h₂ : b < 20) (h₃ : Nat.gcd a b = 1) 
(h₄ : a * b + a + b = 95) : a + b = 18 :=
by
  sorry

end integers_sum_eighteen_l100_100680


namespace work_completion_time_l100_100114

-- Let's define the initial conditions
def total_days := 100
def initial_people := 10
def days1 := 20
def work_done1 := 1 / 4
def days2 (remaining_work_per_person: ℚ) := (3/4) / remaining_work_per_person
def remaining_people := initial_people - 2
def remaining_work_per_person_per_day := remaining_people * (work_done1 / (initial_people * days1))

-- Theorem stating that the total number of days to complete the work is 95
theorem work_completion_time : 
  days1 + days2 remaining_work_per_person_per_day = 95 := 
  by
    sorry -- Proof to be filled in

end work_completion_time_l100_100114


namespace school_dance_attendance_l100_100437

theorem school_dance_attendance (P : ℝ)
  (h1 : 0.1 * P = (P - (0.9 * P)))
  (h2 : 0.9 * P = (2/3) * (0.9 * P) + (1/3) * (0.9 * P))
  (h3 : 30 = (1/3) * (0.9 * P)) :
  P = 100 :=
by
  sorry

end school_dance_attendance_l100_100437


namespace find_a100_l100_100776

noncomputable def S (k : ℝ) (n : ℤ) : ℝ := k * (n ^ 2) + n
noncomputable def a (k : ℝ) (n : ℤ) : ℝ := S k n - S k (n - 1)

theorem find_a100 (k : ℝ) 
  (h1 : a k 10 = 39) :
  a k 100 = 399 :=
sorry

end find_a100_l100_100776


namespace intersection_of_sets_l100_100753

def setA : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def setB : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_sets :
  setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end intersection_of_sets_l100_100753


namespace garden_square_char_l100_100755

theorem garden_square_char (s q p x : ℕ) (h1 : p = 28) (h2 : q = p + x) (h3 : q = s^2) (h4 : p = 4 * s) : x = 21 :=
by
  sorry

end garden_square_char_l100_100755


namespace both_complementary_angles_acute_is_certain_event_l100_100038

def complementary_angles (A B : ℝ) : Prop :=
  A + B = 90

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem both_complementary_angles_acute_is_certain_event (A B : ℝ) (h1 : complementary_angles A B) (h2 : acute_angle A) (h3 : acute_angle B) : (A < 90) ∧ (B < 90) :=
by
  sorry

end both_complementary_angles_acute_is_certain_event_l100_100038


namespace savings_if_together_l100_100919

def window_price : ℕ := 100

def free_windows_for_six_purchased : ℕ := 2

def windows_needed_Dave : ℕ := 9
def windows_needed_Doug : ℕ := 10

def total_individual_cost (windows_purchased : ℕ) : ℕ :=
  100 * windows_purchased

def total_cost_with_deal (windows_purchased: ℕ) : ℕ :=
  let sets_of_6 := windows_purchased / 6
  let remaining_windows := windows_purchased % 6
  100 * (sets_of_6 * 6 + remaining_windows)

def combined_savings (windows_needed_Dave: ℕ) (windows_needed_Doug: ℕ) : ℕ :=
  let total_windows := windows_needed_Dave + windows_needed_Doug
  total_individual_cost windows_needed_Dave 
  + total_individual_cost windows_needed_Doug 
  - total_cost_with_deal total_windows

theorem savings_if_together : combined_savings windows_needed_Dave windows_needed_Doug = 400 :=
by
  sorry

end savings_if_together_l100_100919


namespace pasha_encoded_expression_l100_100252

theorem pasha_encoded_expression :
  2065 + 5 - 47 = 2023 :=
by
  sorry

end pasha_encoded_expression_l100_100252


namespace train_speed_l100_100651

theorem train_speed (train_length : ℝ) (man_speed_kmph : ℝ) (passing_time : ℝ) : 
  train_length = 160 → man_speed_kmph = 6 →
  passing_time = 6 → (train_length / passing_time + man_speed_kmph * 1000 / 3600) * 3600 / 1000 = 90 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- further proof steps are omitted
  sorry

end train_speed_l100_100651


namespace sum_of_numbers_in_ratio_with_lcm_l100_100440

theorem sum_of_numbers_in_ratio_with_lcm
  (x : ℕ)
  (h1 : Nat.lcm (2 * x) (Nat.lcm (3 * x) (5 * x)) = 120) :
  (2 * x) + (3 * x) + (5 * x) = 40 := 
sorry

end sum_of_numbers_in_ratio_with_lcm_l100_100440


namespace quadratic_real_roots_range_l100_100262

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l100_100262


namespace sum_le_square_l100_100699

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l100_100699


namespace average_water_per_day_l100_100586

-- Define the given conditions as variables/constants
def day1 := 318
def day2 := 312
def day3_morning := 180
def day3_afternoon := 162

-- Define the total water added on day 3
def day3 := day3_morning + day3_afternoon

-- Define the total water added over three days
def total_water := day1 + day2 + day3

-- Define the number of days
def days := 3

-- The proof statement: the average water added per day is 324 liters
theorem average_water_per_day : total_water / days = 324 :=
by
  -- Placeholder for the proof
  sorry

end average_water_per_day_l100_100586


namespace anne_gave_sweettarts_to_three_friends_l100_100143

theorem anne_gave_sweettarts_to_three_friends (sweettarts : ℕ) (eaten : ℕ) (friends : ℕ) 
  (h1 : sweettarts = 15) (h2 : eaten = 5) (h3 : sweettarts = friends * eaten) :
  friends = 3 := 
by 
  sorry

end anne_gave_sweettarts_to_three_friends_l100_100143


namespace find_a_minus_b_l100_100505

theorem find_a_minus_b (a b : ℝ) (h1: ∀ x : ℝ, (ax^2 + bx - 2 = 0 → x = -2 ∨ x = -1/4)) : (a - b = 5) :=
sorry

end find_a_minus_b_l100_100505


namespace exponential_difference_l100_100221

theorem exponential_difference (f : ℕ → ℕ) (x : ℕ) (h : f x = 3^x) : f (x + 2) - f x = 8 * f x :=
by sorry

end exponential_difference_l100_100221


namespace sufficient_condition_not_necessary_condition_l100_100812

theorem sufficient_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ab > 0 := by
  sorry

theorem not_necessary_condition (a b : ℝ) : ¬(a > 0 ∧ b > 0) → ab > 0 := by
  sorry

end sufficient_condition_not_necessary_condition_l100_100812


namespace lindsey_saved_in_november_l100_100278

def savings_sept : ℕ := 50
def savings_oct : ℕ := 37
def additional_money : ℕ := 25
def spent_on_video_game : ℕ := 87
def money_left : ℕ := 36

def total_savings_before_november := savings_sept + savings_oct
def total_savings_after_november (N : ℕ) := total_savings_before_november + N + additional_money

theorem lindsey_saved_in_november : ∃ N : ℕ, total_savings_after_november N - spent_on_video_game = money_left ∧ N = 11 :=
by
  sorry

end lindsey_saved_in_november_l100_100278


namespace net_percentage_change_is_correct_l100_100897

def initial_price : Float := 100.0

def price_after_first_year (initial: Float) := initial * (1 - 0.05)

def price_after_second_year (price1: Float) := price1 * (1 + 0.10)

def price_after_third_year (price2: Float) := price2 * (1 + 0.04)

def price_after_fourth_year (price3: Float) := price3 * (1 - 0.03)

def price_after_fifth_year (price4: Float) := price4 * (1 + 0.08)

def final_price := price_after_fifth_year (price_after_fourth_year (price_after_third_year (price_after_second_year (price_after_first_year initial_price))))

def net_percentage_change (initial final: Float) := ((final - initial) / initial) * 100

theorem net_percentage_change_is_correct :
  net_percentage_change initial_price final_price = 13.85 := by
  sorry

end net_percentage_change_is_correct_l100_100897


namespace arithmetic_sequence_term_20_l100_100256

theorem arithmetic_sequence_term_20
  (a : ℕ := 2)
  (d : ℕ := 4)
  (n : ℕ := 20) :
  a + (n - 1) * d = 78 :=
by
  sorry

end arithmetic_sequence_term_20_l100_100256


namespace min_f_abs_l100_100105

def f (x y : ℤ) : ℤ := 5 * x^2 + 11 * x * y - 5 * y^2

theorem min_f_abs (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : (∃ m, ∀ x y : ℤ, (x ≠ 0 ∨ y ≠ 0) → |f x y| ≥ m) ∧ 5 = 5 :=
by
  sorry -- proof goes here

end min_f_abs_l100_100105


namespace race_problem_equivalent_l100_100685

noncomputable def race_track_distance (D_paved D_dirt D_muddy : ℝ) : Prop :=
  let v1 := 100 -- speed on paved section in km/h
  let v2 := 70  -- speed on dirt section in km/h
  let v3 := 15  -- speed on muddy section in km/h
  let initial_distance := 0.5 -- initial distance in km (since 500 meters is 0.5 km)
  
  -- Time to cover paved section
  let t_white_paved := D_paved / v1
  let t_red_paved := (D_paved - initial_distance) / v1

  -- Times to cover dirt section
  let t_white_dirt := D_dirt / v2
  let t_red_dirt := D_dirt / v2 -- same time since both start at the same time on dirt

  -- Times to cover muddy section
  let t_white_muddy := D_muddy / v3
  let t_red_muddy := D_muddy / v3 -- same time since both start at the same time on mud

  -- Distances between cars on dirt and muddy sections
  ((t_white_paved - t_red_paved) * v2 = initial_distance) ∧ 
  ((t_white_paved - t_red_paved) * v3 = initial_distance)

-- Prove the distance between the cars when both are on the dirt and muddy sections is 500 meters
theorem race_problem_equivalent (D_paved D_dirt D_muddy : ℝ) : race_track_distance D_paved D_dirt D_muddy :=
by
  -- Insert proof here, for now we use sorry
  sorry

end race_problem_equivalent_l100_100685


namespace solve_for_t_l100_100707

theorem solve_for_t (p t : ℝ) (h1 : 5 = p * 3^t) (h2 : 45 = p * 9^t) : t = 2 :=
by
  sorry

end solve_for_t_l100_100707


namespace largest_triangle_perimeter_with_7_9_x_l100_100076

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def triangle_side_x_valid (x : ℕ) : Prop :=
  is_divisible_by_3 x ∧ 2 < x ∧ x < 16

theorem largest_triangle_perimeter_with_7_9_x (x : ℕ) (h : triangle_side_x_valid x) : 
  ∃ P : ℕ, P = 7 + 9 + x ∧ P = 31 :=
by
  sorry

end largest_triangle_perimeter_with_7_9_x_l100_100076


namespace keys_per_lock_l100_100449

-- Define the given conditions
def num_complexes := 2
def apartments_per_complex := 12
def total_keys := 72

-- Calculate the total number of apartments
def total_apartments := num_complexes * apartments_per_complex

-- The theorem statement to prove
theorem keys_per_lock : total_keys / total_apartments = 3 := 
by
  sorry

end keys_per_lock_l100_100449


namespace snail_returns_to_starting_point_l100_100566

-- Define the variables and conditions
variables (a1 a2 b1 b2 : ℕ)

-- Prove that snail can return to starting point after whole number of hours
theorem snail_returns_to_starting_point (h1 : a1 = a2) (h2 : b1 = b2) : (a1 + b1 : ℕ) = (a1 + b1 : ℕ) :=
by sorry

end snail_returns_to_starting_point_l100_100566


namespace percentage_of_men_not_speaking_french_or_spanish_l100_100547

theorem percentage_of_men_not_speaking_french_or_spanish 
  (total_employees : ℕ) 
  (men_percent women_percent : ℝ)
  (men_french percent men_spanish_percent men_other_percent : ℝ)
  (women_french_percent women_spanish_percent women_other_percent : ℝ)
  (h1 : men_percent = 60)
  (h2 : women_percent = 40)
  (h3 : men_french_percent = 55)
  (h4 : men_spanish_percent = 35)
  (h5 : men_other_percent = 10)
  (h6 : women_french_percent = 45)
  (h7 : women_spanish_percent = 25)
  (h8 : women_other_percent = 30) :
  men_other_percent = 10 := 
by
  sorry

end percentage_of_men_not_speaking_french_or_spanish_l100_100547


namespace mono_sum_eq_five_l100_100893

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l100_100893


namespace second_hand_travel_distance_l100_100168

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end second_hand_travel_distance_l100_100168


namespace ones_digit_of_p_l100_100305

theorem ones_digit_of_p (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hseq : q = p + 4 ∧ r = p + 8 ∧ s = p + 12) (hpg : p > 5) : (p % 10) = 9 :=
by
  sorry

end ones_digit_of_p_l100_100305


namespace black_white_ratio_extended_pattern_l100_100949

theorem black_white_ratio_extended_pattern
  (original_black : ℕ) (original_white : ℕ) (added_black : ℕ)
  (h1 : original_black = 10)
  (h2 : original_white = 26)
  (h3 : added_black = 20) :
  (original_black + added_black) / original_white = 30 / 26 :=
by sorry

end black_white_ratio_extended_pattern_l100_100949


namespace y_pow_expression_l100_100818

theorem y_pow_expression (y : ℝ) (h : y + 1/y = 3) : y^13 - 5 * y^9 + y^5 = 0 :=
sorry

end y_pow_expression_l100_100818


namespace q_true_given_not_p_and_p_or_q_l100_100805

theorem q_true_given_not_p_and_p_or_q (p q : Prop) (hnp : ¬p) (hpq : p ∨ q) : q :=
by
  sorry

end q_true_given_not_p_and_p_or_q_l100_100805


namespace min_value_of_reciprocals_l100_100030

theorem min_value_of_reciprocals (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  ∃ x, x = (1 / a) + (1 / b) ∧ x ≥ 4 := 
sorry

end min_value_of_reciprocals_l100_100030


namespace total_seashells_after_six_weeks_l100_100740

theorem total_seashells_after_six_weeks :
  ∀ (a b : ℕ) 
  (initial_a : a = 50) 
  (initial_b : b = 30) 
  (next_a : ∀ k : ℕ, k > 0 → a + 20 = (a + 20) * k) 
  (next_b : ∀ k : ℕ, k > 0 → b * 2 = (b * 2) * k), 
  (a + 20 * 5) + (b * 2 ^ 5) = 1110 :=
by
  intros a b initial_a initial_b next_a next_b
  sorry

end total_seashells_after_six_weeks_l100_100740


namespace original_four_digit_number_l100_100774

theorem original_four_digit_number : 
  ∃ x y z: ℕ, (x = 1 ∧ y = 9 ∧ z = 7 ∧ 1000 * x + 100 * y + 10 * z + y = 1979) ∧ 
  (1000 * y + 100 * z + 10 * y + x - (1000 * x + 100 * y + 10 * z + y) = 7812) ∧ 
  (1000 * y + 100 * z + 10 * y + x < 10000 ∧ 1000 * x + 100 * y + 10 * z + y < 10000) := 
sorry

end original_four_digit_number_l100_100774


namespace range_of_a_l100_100551

theorem range_of_a (x a : ℝ) (h1 : -2 < x) (h2 : x ≤ 1) (h3 : |x - 2| < a) : a ≤ 0 :=
sorry

end range_of_a_l100_100551


namespace find_a_from_coefficient_l100_100835

theorem find_a_from_coefficient :
  (∀ x : ℝ, (x + 1)^6 * (a*x - 1)^2 = 20 → a = 0 ∨ a = 5) :=
by
  sorry

end find_a_from_coefficient_l100_100835


namespace sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l100_100655

theorem sum_of_squares_divisible_by_7_implies_product_divisible_by_49 (a b : ℕ) 
  (h : (a * a + b * b) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l100_100655


namespace find_a_b_largest_x_l100_100053

def polynomial (a b x : ℤ) : ℤ := 2 * (a * x - 3) - 3 * (b * x + 5)

-- Given conditions
variables (a b : ℤ)
#check polynomial

-- Part 1: Prove the values of a and b
theorem find_a_b (h1 : polynomial a b 2 = -31) (h2 : a + b = 0) : a = -1 ∧ b = 1 :=
by sorry

-- Part 2: Given a and b found in Part 1, find the largest integer x such that P > 0
noncomputable def P (x : ℤ) : ℤ := -5 * x - 21

theorem largest_x {a b : ℤ} (ha : a = -1) (hb : b = 1) : ∃ x : ℤ, P x > 0 ∧ ∀ y : ℤ, (P y > 0 → y ≤ x) :=
by sorry

end find_a_b_largest_x_l100_100053


namespace gold_coin_multiple_l100_100216

theorem gold_coin_multiple (x y k : ℕ) (h₁ : x + y = 16) (h₂ : x ≠ y) (h₃ : x^2 - y^2 = k * (x - y)) : k = 16 :=
sorry

end gold_coin_multiple_l100_100216


namespace boat_downstream_distance_l100_100497

variable (speed_still_water : ℤ) (speed_stream : ℤ) (time_downstream : ℤ)

theorem boat_downstream_distance
    (h₁ : speed_still_water = 24)
    (h₂ : speed_stream = 4)
    (h₃ : time_downstream = 4) :
    (speed_still_water + speed_stream) * time_downstream = 112 := by
  sorry

end boat_downstream_distance_l100_100497


namespace product_ge_half_l100_100480

theorem product_ge_half (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3) (h_sum : x1 + x2 + x3 ≤ 1/2) :
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1/2 :=
by
  sorry

end product_ge_half_l100_100480


namespace sin_double_angle_l100_100711

theorem sin_double_angle (theta : ℝ) 
  (h : Real.sin (theta + Real.pi / 4) = 2 / 5) :
  Real.sin (2 * theta) = -17 / 25 := by
  sorry

end sin_double_angle_l100_100711


namespace tank_capacity_75_l100_100827

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l100_100827


namespace mark_sandwiches_l100_100176

/--
Each day of a 6-day workweek, Mark bought either an 80-cent donut or a $1.20 sandwich. 
His total expenditure for the week was an exact number of dollars.
Prove that Mark bought exactly 3 sandwiches.
-/
theorem mark_sandwiches (s d : ℕ) (h1 : s + d = 6) (h2 : ∃ k : ℤ, 120 * s + 80 * d = 100 * k) : s = 3 :=
by
  sorry

end mark_sandwiches_l100_100176


namespace farmer_farm_size_l100_100631

theorem farmer_farm_size 
  (sunflowers flax : ℕ)
  (h1 : flax = 80)
  (h2 : sunflowers = flax + 80) :
  (sunflowers + flax = 240) :=
by
  sorry

end farmer_farm_size_l100_100631


namespace bus_initial_passengers_l100_100756

theorem bus_initial_passengers (M W : ℕ) 
  (h1 : W = M / 2) 
  (h2 : M - 16 = W + 8) : 
  M + W = 72 :=
sorry

end bus_initial_passengers_l100_100756


namespace solve_triples_l100_100976

theorem solve_triples (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  a^2 + b^2 = n * Nat.lcm a b + n^2 ∧
  b^2 + c^2 = n * Nat.lcm b c + n^2 ∧
  c^2 + a^2 = n * Nat.lcm c a + n^2 →
  ∃ k : ℕ, 0 < k ∧ a = k ∧ b = k ∧ c = k :=
by
  intros h
  sorry

end solve_triples_l100_100976


namespace minimum_value_x_plus_4_div_x_l100_100791

theorem minimum_value_x_plus_4_div_x (x : ℝ) (hx : x > 0) : x + 4 / x ≥ 4 :=
sorry

end minimum_value_x_plus_4_div_x_l100_100791


namespace winning_candidate_votes_l100_100564

def total_votes : ℕ := 100000
def winning_percentage : ℚ := 42 / 100
def expected_votes : ℚ := 42000

theorem winning_candidate_votes : winning_percentage * total_votes = expected_votes := by
  sorry

end winning_candidate_votes_l100_100564


namespace probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l100_100345

theorem probability_exactly_2_boys_1_girl 
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (Nat.choose boys 2 * Nat.choose girls 1 / (Nat.choose total_group select) : ℚ) = 3 / 5 :=
by sorry

theorem probability_at_least_1_girl
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (1 - (Nat.choose boys select / Nat.choose total_group select : ℚ)) = 4 / 5 :=
by sorry

end probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l100_100345


namespace area_EFCD_l100_100892

-- Defining the geometrical setup and measurements of the trapezoid
variables (AB CD AD BC : ℝ) (h1 : AB = 10) (h2 : CD = 30) (h_altitude : ∃ h : ℝ, h = 18)

-- Defining the midpoints E and F of AD and BC respectively
variables (E F : ℝ) (h_E : E = AD / 2) (h_F : F = BC / 2)

-- Define the intersection of diagonals and the ratio condition
variables (AC BD G : ℝ) (h_ratio : ∃ r : ℝ, r = 1/2)

-- Proving the area of quadrilateral EFCD
theorem area_EFCD : EFCD_area = 225 :=
sorry

end area_EFCD_l100_100892


namespace verify_system_of_equations_l100_100147

/-- Define a structure to hold the conditions of the problem -/
structure TreePurchasing :=
  (cost_A : ℕ)
  (cost_B : ℕ)
  (diff_A_B : ℕ)
  (total_cost : ℕ)
  (x : ℕ)
  (y : ℕ)

/-- Given conditions for purchasing trees -/
def example_problem : TreePurchasing :=
  { cost_A := 100,
    cost_B := 80,
    diff_A_B := 8,
    total_cost := 8000,
    x := 0,
    y := 0 }

/-- The theorem to prove that the equations match given conditions -/
theorem verify_system_of_equations (data : TreePurchasing) (h_diff : data.x - data.y = data.diff_A_B) (h_cost : data.cost_A * data.x + data.cost_B * data.y = data.total_cost) : 
  (data.x - data.y = 8) ∧ (100 * data.x + 80 * data.y = 8000) :=
  by
    sorry

end verify_system_of_equations_l100_100147


namespace greatest_integer_l100_100585

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l100_100585


namespace units_digit_k_squared_plus_two_exp_k_eq_7_l100_100427

/-- Define k as given in the problem -/
def k : ℕ := 2010^2 + 2^2010

/-- Final statement that needs to be proved -/
theorem units_digit_k_squared_plus_two_exp_k_eq_7 : (k^2 + 2^k) % 10 = 7 := 
by
  sorry

end units_digit_k_squared_plus_two_exp_k_eq_7_l100_100427


namespace sqrt_meaningful_condition_l100_100412

theorem sqrt_meaningful_condition (a : ℝ) : 2 - a ≥ 0 → a ≤ 2 := by
  sorry

end sqrt_meaningful_condition_l100_100412


namespace susan_hourly_rate_l100_100565

-- Definitions based on conditions
def vacation_workdays : ℕ := 10 -- Susan is taking a two-week vacation equivalent to 10 workdays

def weekly_workdays : ℕ := 5 -- Susan works 5 days a week

def paid_vacation_days : ℕ := 6 -- Susan has 6 days of paid vacation

def hours_per_day : ℕ := 8 -- Susan works 8 hours a day

def missed_pay_total : ℕ := 480 -- Susan will miss $480 pay on her unpaid vacation days

-- Calculations
def unpaid_vacation_days : ℕ := vacation_workdays - paid_vacation_days

def daily_lost_pay : ℕ := missed_pay_total / unpaid_vacation_days

def hourly_rate : ℕ := daily_lost_pay / hours_per_day

theorem susan_hourly_rate :
  hourly_rate = 15 := by sorry

end susan_hourly_rate_l100_100565


namespace skittles_distribution_l100_100450

theorem skittles_distribution :
  let initial_skittles := 14
  let additional_skittles := 22
  let total_skittles := initial_skittles + additional_skittles
  let number_of_people := 7
  (total_skittles / number_of_people = 5) :=
by
  sorry

end skittles_distribution_l100_100450


namespace evaluate_expression_l100_100576

theorem evaluate_expression (a b c : ℝ) (h1 : a = 4) (h2 : b = -4) (h3 : c = 3) : (3 / (a + b + c) = 1) :=
by
  sorry

end evaluate_expression_l100_100576


namespace choose_starters_l100_100494

theorem choose_starters :
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  Nat.choose totalPlayers 6 - Nat.choose playersExcludingTwins 6 = 5005 :=
by
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  sorry

end choose_starters_l100_100494


namespace exact_one_solves_l100_100182

variables (p1 p2 : ℝ)

/-- The probability that exactly one of two persons solves the problem
    when their respective probabilities are p1 and p2. -/
theorem exact_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 + p2 - 2 * p1 * p2) := 
sorry

end exact_one_solves_l100_100182


namespace cubic_equation_roots_l100_100301

theorem cubic_equation_roots (a b c d : ℝ) (h_a : a ≠ 0) 
(h_root1 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
(h_root2 : a * (-3)^3 + b * (-3)^2 - 3 * c + d = 0) :
 (b + c) / a = -13 :=
by sorry

end cubic_equation_roots_l100_100301


namespace fraction_equality_l100_100765

-- Defining the hypotheses and the goal
theorem fraction_equality (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 10 :=
by
  sorry

end fraction_equality_l100_100765


namespace simplify_expression_l100_100500

theorem simplify_expression (x : ℕ) : (5 * x^4)^3 = 125 * x^(12) := by
  sorry

end simplify_expression_l100_100500


namespace number_of_passed_candidates_l100_100080

-- Definitions based on conditions:
def total_candidates : ℕ := 120
def avg_total_marks : ℝ := 35
def avg_passed_marks : ℝ := 39
def avg_failed_marks : ℝ := 15

-- The number of candidates who passed the examination:
theorem number_of_passed_candidates :
  ∃ (P F : ℕ), 
    P + F = total_candidates ∧
    39 * P + 15 * F = total_candidates * avg_total_marks ∧
    P = 100 :=
by
  sorry

end number_of_passed_candidates_l100_100080


namespace find_m_n_l100_100913

theorem find_m_n (x m n : ℤ) : (x + 2) * (x + 3) = x^2 + m * x + n → m = 5 ∧ n = 6 :=
by {
    sorry
}

end find_m_n_l100_100913


namespace complex_number_in_first_quadrant_l100_100446

noncomputable def z : ℂ := Complex.ofReal 1 + Complex.I

theorem complex_number_in_first_quadrant 
  (h : Complex.ofReal 1 + Complex.I = Complex.I / z) : 
  (0 < z.re ∧ 0 < z.im) :=
  sorry

end complex_number_in_first_quadrant_l100_100446


namespace quadratic_inequality_solution_l100_100924

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l100_100924


namespace polygon_sides_eq_eight_l100_100022

theorem polygon_sides_eq_eight (x : ℕ) (h : x ≥ 3) 
  (h1 : 2 * (x - 2) = 180 * (x - 2) / 90) 
  (h2 : ∀ x, x + 2 * (x - 2) = x * (x - 3) / 2) : 
  x = 8 :=
by
  sorry

end polygon_sides_eq_eight_l100_100022


namespace possible_values_of_cubic_sum_l100_100921

theorem possible_values_of_cubic_sum (x y z : ℂ) (h1 : (Matrix.of ![
    ![x, y, z],
    ![y, z, x],
    ![z, x, y]
  ] ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ))) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨ x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry

end possible_values_of_cubic_sum_l100_100921


namespace complete_square_form_l100_100650

theorem complete_square_form (a b x : ℝ) : 
  ∃ (p : ℝ) (q : ℝ), 
  (p = x ∧ q = 1 ∧ (x^2 + 2*x + 1 = (p + q)^2)) ∧ 
  (¬ ∃ (p q : ℝ), a^2 + 4 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + a*b + b^2 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + 4*a*b + b^2 = (a + p) * (a + q)) :=
  sorry

end complete_square_form_l100_100650


namespace total_discount_is_58_percent_l100_100488

-- Definitions and conditions
def sale_discount : ℝ := 0.4
def coupon_discount : ℝ := 0.3

-- Given an original price, the sale discount price and coupon discount price
def sale_price (original_price : ℝ) : ℝ := (1 - sale_discount) * original_price
def final_price (original_price : ℝ) : ℝ := (1 - coupon_discount) * (sale_price original_price)

-- Theorem statement: final discount is 58%
theorem total_discount_is_58_percent (original_price : ℝ) : (original_price - final_price original_price) / original_price = 0.58 :=
by intros; sorry

end total_discount_is_58_percent_l100_100488


namespace green_pen_count_l100_100349

theorem green_pen_count 
  (blue_pens green_pens : ℕ)
  (h_ratio : blue_pens = 5 * green_pens / 3)
  (h_blue_pens : blue_pens = 20)
  : green_pens = 12 :=
by
  sorry

end green_pen_count_l100_100349


namespace f_7_eq_minus_1_l100_100381

-- Define the odd function f with the given properties
def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) :=
  ∀ x, f (x + 2) = -f x

def f_restricted (f : ℝ → ℝ) :=
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 -> f x = x

-- The main statement: Under the given conditions, f(7) = -1
theorem f_7_eq_minus_1 (f : ℝ → ℝ)
  (H1 : is_odd_function f)
  (H2 : period_2 f)
  (H3 : f_restricted f) :
  f 7 = -1 :=
by
  sorry

end f_7_eq_minus_1_l100_100381


namespace problem1_problem2_problem3_l100_100339

-- Given conditions for the sequence
axiom pos_seq {a : ℕ → ℝ} : (∀ n : ℕ, 0 < a n)
axiom relation1 {a : ℕ → ℝ} (t : ℝ) : (∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
axiom relation2 {a : ℕ → ℝ} : 2 * (a 3) = (a 2) + (a 4)

-- Proof Requirements

-- (1) Find the value of (a1 + a3) / a2
theorem problem1 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  (a 1 + a 3) / a 2 = 2 :=
sorry

-- (2) Prove that the sequence is an arithmetic sequence
theorem problem2 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  ∀ n : ℕ, a (n+2) - a (n+1) = a (n+1) - a n :=
sorry

-- (3) Show p and r such that (1/a_k), (1/a_p), (1/a_r) form an arithmetic sequence
theorem problem3 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) (k : ℕ) (hk : k ≠ 0) :
  (k = 1 → ∀ p r : ℕ, ¬((k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p))) ∧ 
  (k ≥ 2 → ∃ p r : ℕ, (k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p) ∧ p = 2 * k - 1 ∧ r = k * (2 * k - 1)) :=
sorry

end problem1_problem2_problem3_l100_100339


namespace work_completion_days_l100_100261

theorem work_completion_days (x : ℕ) (h_ratio : 5 * 18 = 3 * 30) : 30 = 30 :=
by {
    sorry
}

end work_completion_days_l100_100261


namespace largest_n_base_conditions_l100_100797

theorem largest_n_base_conditions :
  ∃ n: ℕ, n < 10000 ∧ 
  (∃ a: ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b: ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c: ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1)) ∧
  n = 4369 :=
sorry

end largest_n_base_conditions_l100_100797


namespace difference_in_sums_l100_100531

def sum_of_digits (n : ℕ) : ℕ := (toString n).foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def Petrov_numbers := List.range' 1 2014 |>.filter (λ n => n % 2 = 1)
def Vasechkin_numbers := List.range' 2 2012 |>.filter (λ n => n % 2 = 0)

def sum_of_digits_Petrov := (Petrov_numbers.map sum_of_digits).sum
def sum_of_digits_Vasechkin := (Vasechkin_numbers.map sum_of_digits).sum

theorem difference_in_sums : sum_of_digits_Petrov - sum_of_digits_Vasechkin = 1007 := by
  sorry

end difference_in_sums_l100_100531


namespace min_value_y_l100_100836

theorem min_value_y (x : ℝ) (h : x > 0) : ∃ y, y = x + 4 / x^2 ∧ (∀ z, z = x + 4 / x^2 → y ≤ z) := 
sorry

end min_value_y_l100_100836


namespace find_value_of_a3_a6_a9_l100_100832

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

variables {a : ℕ → ℤ} (d : ℤ)

-- Given conditions
axiom cond1 : a 1 + a 4 + a 7 = 45
axiom cond2 : a 2 + a 5 + a 8 = 29

-- Lean 4 Statement
theorem find_value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 13 :=
sorry

end find_value_of_a3_a6_a9_l100_100832


namespace find_bc_l100_100353

noncomputable def setA : Set ℝ := {x | x^2 + x - 2 ≤ 0}
noncomputable def setB : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
noncomputable def setAB : Set ℝ := setA ∪ setB
noncomputable def setC (b c : ℝ) : Set ℝ := {x | x^2 + b * x + c > 0}

theorem find_bc (b c : ℝ) :
  (setAB ∩ setC b c = ∅) ∧ (setAB ∪ setC b c = Set.univ) →
  b = -1 ∧ c = -6 :=
by
  sorry

end find_bc_l100_100353


namespace gcd_of_powers_of_two_minus_one_l100_100290

theorem gcd_of_powers_of_two_minus_one : 
  gcd (2^1015 - 1) (2^1020 - 1) = 1 :=
sorry

end gcd_of_powers_of_two_minus_one_l100_100290


namespace josh_total_candies_l100_100402

def josh_initial_candies (initial_candies given_siblings : ℕ) : Prop :=
  ∃ (remaining_1 best_friend josh_eats share_others : ℕ),
    (remaining_1 = initial_candies - given_siblings) ∧
    (best_friend = remaining_1 / 2) ∧
    (josh_eats = 16) ∧
    (share_others = 19) ∧
    (remaining_1 = 2 * (josh_eats + share_others))

theorem josh_total_candies : josh_initial_candies 100 30 :=
by
  sorry

end josh_total_candies_l100_100402


namespace gambler_final_amount_l100_100943

-- Define initial amount of money
def initial_amount := 100

-- Define the multipliers
def win_multiplier := 4 / 3
def loss_multiplier := 2 / 3
def double_win_multiplier := 5 / 3

-- Define the gambler scenario (WWLWLWLW)
def scenario := [double_win_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier]

-- Function to compute final amount given initial amount, number of wins and losses, and the scenario
def final_amount (initial: ℚ) (multipliers: List ℚ) : ℚ :=
  multipliers.foldl (· * ·) initial

-- Prove that the final amount after all multipliers are applied is approximately equal to 312.12
theorem gambler_final_amount : abs (final_amount initial_amount scenario - 312.12) < 0.01 :=
by
  sorry

end gambler_final_amount_l100_100943


namespace yoongi_division_l100_100855

theorem yoongi_division (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end yoongi_division_l100_100855


namespace factors_of_2550_have_more_than_3_factors_l100_100612

theorem factors_of_2550_have_more_than_3_factors :
  ∃ n: ℕ, n = 5 ∧
    ∃ d: ℕ, d = 2550 ∧
    (∀ x < n, ∃ y: ℕ, y ∣ d ∧ (∃ z, z ∣ y ∧ z > 3)) :=
sorry

end factors_of_2550_have_more_than_3_factors_l100_100612


namespace shaded_area_of_modified_design_l100_100323

noncomputable def radius_of_circles (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  (side_length / grid_size) / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

noncomputable def shaded_area (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  let r := radius_of_circles side_length grid_size
  let total_circle_area := 9 * area_of_circle r
  area_of_square side_length - total_circle_area

theorem shaded_area_of_modified_design :
  shaded_area 24 3 = (576 - 144 * Real.pi) :=
by
  sorry

end shaded_area_of_modified_design_l100_100323


namespace find_A_salary_l100_100243

theorem find_A_salary (A B : ℝ) (h1 : A + B = 2000) (h2 : 0.05 * A = 0.15 * B) : A = 1500 :=
sorry

end find_A_salary_l100_100243


namespace contrapositive_question_l100_100495

theorem contrapositive_question (x : ℝ) :
  (x = 2 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 2) := 
sorry

end contrapositive_question_l100_100495


namespace biloca_path_proof_l100_100525

def diagonal_length := 5 -- Length of one diagonal as deduced from Pipoca's path
def tile_width := 3 -- Width of one tile as deduced from Tonica's path
def tile_length := 4 -- Length of one tile as deduced from Cotinha's path

def Biloca_path_length : ℝ :=
  3 * diagonal_length + 4 * tile_width + 2 * tile_length

theorem biloca_path_proof :
  Biloca_path_length = 43 :=
by
  sorry

end biloca_path_proof_l100_100525


namespace cupcakes_frosted_in_10_minutes_l100_100451

-- Definitions representing the given conditions
def CagneyRate := 15 -- seconds per cupcake
def LaceyRate := 40 -- seconds per cupcake
def JessieRate := 30 -- seconds per cupcake
def initialDuration := 3 * 60 -- 3 minutes in seconds
def totalDuration := 10 * 60 -- 10 minutes in seconds
def afterJessieDuration := totalDuration - initialDuration -- 7 minutes in seconds

-- Proof statement
theorem cupcakes_frosted_in_10_minutes : 
  let combinedRateBefore := (CagneyRate * LaceyRate) / (CagneyRate + LaceyRate)
  let combinedRateAfter := (CagneyRate * LaceyRate * JessieRate) / (CagneyRate * LaceyRate + LaceyRate * JessieRate + JessieRate * CagneyRate)
  let cupcakesBefore := initialDuration / combinedRateBefore
  let cupcakesAfter := afterJessieDuration / combinedRateAfter
  cupcakesBefore + cupcakesAfter = 68 :=
by
  sorry

end cupcakes_frosted_in_10_minutes_l100_100451


namespace find_m_l100_100190

def numFactorsOf2 (k : ℕ) : ℕ :=
  k / 2 + k / 4 + k / 8 + k / 16 + k / 32 + k / 64 + k / 128 + k / 256

theorem find_m : ∃ m : ℕ, m > 1990 ^ 1990 ∧ m = 3 ^ 1990 + numFactorsOf2 m :=
by
  sorry

end find_m_l100_100190


namespace arithmetic_operation_equals_l100_100374

theorem arithmetic_operation_equals :
  12.1212 + 17.0005 - 9.1103 = 20.0114 := 
by 
  sorry

end arithmetic_operation_equals_l100_100374


namespace smaller_cuboid_width_l100_100231

theorem smaller_cuboid_width
  (length_orig width_orig height_orig : ℕ)
  (length_small height_small : ℕ)
  (num_small_cuboids : ℕ)
  (volume_orig : ℕ := length_orig * width_orig * height_orig)
  (volume_small : ℕ := length_small * width_small * height_small)
  (H1 : length_orig = 18)
  (H2 : width_orig = 15)
  (H3 : height_orig = 2)
  (H4 : length_small = 5)
  (H5 : height_small = 3)
  (H6 : num_small_cuboids = 6)
  (H_volume_match : num_small_cuboids * volume_small = volume_orig)
  : width_small = 6 := by
  sorry

end smaller_cuboid_width_l100_100231


namespace triangle_shape_and_maximum_tan_B_minus_C_l100_100885

open Real

variable (A B C : ℝ)
variable (sin cos tan : ℝ → ℝ)

-- Given conditions
axiom sin2A_plus_3sin2C_equals_3sin2B : sin A ^ 2 + 3 * sin C ^ 2 = 3 * sin B ^ 2
axiom sinB_cosC_equals_2div3 : sin B * cos C = 2 / 3

-- Prove
theorem triangle_shape_and_maximum_tan_B_minus_C :
  (A = π / 2) ∧ (∀ x y : ℝ, (x = B - C) → tan x ≤ sqrt 2 / 4) :=
by sorry

end triangle_shape_and_maximum_tan_B_minus_C_l100_100885


namespace fraction_exponentiation_example_l100_100297

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end fraction_exponentiation_example_l100_100297


namespace sequence_converges_l100_100493

theorem sequence_converges (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
    ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(a n)^ (1/n) - l| < ε :=
by
  sorry

end sequence_converges_l100_100493


namespace smallest_value_satisfies_equation_l100_100181

theorem smallest_value_satisfies_equation : ∃ x : ℝ, (|5 * x + 9| = 34) ∧ x = -8.6 :=
by
  sorry

end smallest_value_satisfies_equation_l100_100181


namespace cubesWithTwoColoredFaces_l100_100165

structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

def numberOfSmallerCubes (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

def numberOfCubesWithTwoColoredFaces (d : CuboidDimensions) : ℕ :=
  2 * (d.length - 2) * 2 + 2 * (d.width - 2) * 2 + 2 * (d.height - 2) * 2

theorem cubesWithTwoColoredFaces :
  numberOfCubesWithTwoColoredFaces { length := 4, width := 3, height := 3 } = 16 := by
  sorry

end cubesWithTwoColoredFaces_l100_100165


namespace number_of_tiles_per_row_l100_100825

-- Definitions of conditions
def area : ℝ := 320
def length : ℝ := 16
def tile_size : ℝ := 1

-- Theorem statement
theorem number_of_tiles_per_row : (area / length) / tile_size = 20 := by
  sorry

end number_of_tiles_per_row_l100_100825


namespace sin_double_angle_l100_100944

variable {α : Real}

theorem sin_double_angle (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_l100_100944


namespace find_divisor_l100_100068

-- Define the initial number
def num := 1387

-- Define the number to subtract to make it divisible by some divisor
def least_subtract := 7

-- Define the resulting number after subtraction
def remaining_num := num - least_subtract

-- Define the divisor
def divisor := 23

-- The statement to prove: 1380 is divisible by 23
theorem find_divisor (num_subtract_div : num - least_subtract = remaining_num) 
                     (remaining_divisor : remaining_num = 1380) : 
                     ∃ k : ℕ, 1380 = k * divisor := by
  sorry

end find_divisor_l100_100068


namespace group_size_l100_100188

theorem group_size (boxes_per_man total_boxes : ℕ) (h1 : boxes_per_man = 2) (h2 : total_boxes = 14) :
  total_boxes / boxes_per_man = 7 := by
  -- Definitions and conditions from the problem
  have man_can_carry_2_boxes : boxes_per_man = 2 := h1
  have group_can_hold_14_boxes : total_boxes = 14 := h2
  -- Proof follows from these conditions
  sorry

end group_size_l100_100188


namespace alex_points_l100_100171

variable {x y : ℕ} -- x is the number of three-point shots, y is the number of two-point shots
variable (success_rate_3 success_rate_2 : ℚ) -- success rates for three-point and two-point shots
variable (total_shots : ℕ) -- total number of shots

def alex_total_points (x y : ℕ) (success_rate_3 success_rate_2 : ℚ) : ℚ :=
  3 * success_rate_3 * x + 2 * success_rate_2 * y

axiom condition_1 : success_rate_3 = 0.25
axiom condition_2 : success_rate_2 = 0.20
axiom condition_3 : total_shots = 40
axiom condition_4 : x + y = total_shots

theorem alex_points : alex_total_points x y 0.25 0.20 = 30 :=
by
  -- The proof would go here
  sorry

end alex_points_l100_100171


namespace value_of_a_l100_100416

theorem value_of_a (x : ℝ) (n : ℕ) (h : x > 0) (h_n : n > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → x + k ≥ k + 1) → a = n^n :=
by
  sorry

end value_of_a_l100_100416


namespace diagonal_lt_half_perimeter_l100_100575

theorem diagonal_lt_half_perimeter (AB BC CD DA AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0) 
  (h_triangle1 : AC < AB + BC) (h_triangle2 : AC < AD + DC) :
  AC < (AB + BC + CD + DA) / 2 :=
by {
  sorry
}

end diagonal_lt_half_perimeter_l100_100575


namespace problem_l100_100706

variable {x : ℝ}

theorem problem (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 :=
by
  sorry

end problem_l100_100706


namespace price_of_second_set_of_knives_l100_100399

def john_visits_houses_per_day : ℕ := 50
def percent_buying_per_day : ℝ := 0.20
def price_first_set : ℝ := 50
def weekly_sales : ℝ := 5000
def work_days_per_week : ℕ := 5

theorem price_of_second_set_of_knives
  (john_visits_houses_per_day : ℕ)
  (percent_buying_per_day : ℝ)
  (price_first_set : ℝ)
  (weekly_sales : ℝ)
  (work_days_per_week : ℕ) :
  0 < percent_buying_per_day ∧ percent_buying_per_day ≤ 1 ∧
  weekly_sales = 5000 ∧ 
  work_days_per_week = 5 ∧
  john_visits_houses_per_day = 50 ∧
  price_first_set = 50 → 
  (∃ price_second_set : ℝ, price_second_set = 150) :=
  sorry

end price_of_second_set_of_knives_l100_100399


namespace catch_up_time_l100_100859

-- Define the speeds of Person A and Person B.
def speed_A : ℝ := 10 -- kilometers per hour
def speed_B : ℝ := 7  -- kilometers per hour

-- Define the initial distance between Person A and Person B.
def initial_distance : ℝ := 15 -- kilometers

-- Prove the time it takes for person A to catch up with person B is 5 hours.
theorem catch_up_time :
  initial_distance / (speed_A - speed_B) = 5 :=
by
  -- Proof can be added here
  sorry

end catch_up_time_l100_100859


namespace plane_equation_l100_100369

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧ 
  (∀ (x y z : ℤ), (x, y, z) = (0, 0, 0) ∨ (x, y, z) = (2, 0, -2) → A * x + B * y + C * z + D = 0) ∧ 
  ∀ (x y z : ℤ), (A = 1 ∧ B = -5 ∧ C = 1 ∧ D = 0) := sorry

end plane_equation_l100_100369


namespace probability_three_defective_phones_l100_100513

theorem probability_three_defective_phones :
  let total_smartphones := 380
  let defective_smartphones := 125
  let P_def_1 := (defective_smartphones : ℝ) / total_smartphones
  let P_def_2 := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let P_def_3 := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let P_all_three_def := P_def_1 * P_def_2 * P_def_3
  abs (P_all_three_def - 0.0351) < 0.001 := 
by
  sorry

end probability_three_defective_phones_l100_100513


namespace sequence_periodicity_a5_a2019_l100_100902

theorem sequence_periodicity_a5_a2019 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → a n * a (n + 2) = 3 * a (n + 1)) :
  a 5 * a 2019 = 27 :=
sorry

end sequence_periodicity_a5_a2019_l100_100902


namespace initial_house_cats_l100_100070

theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 38) 
  (cats_sold : ℕ := 45) 
  (cats_left : ℕ := 18) 
  (initial_total_cats : ℕ := siamese_cats + H) 
  (after_sale_cats : ℕ := initial_total_cats - cats_sold) : 
  after_sale_cats = cats_left → H = 25 := 
by
  intro h
  sorry

end initial_house_cats_l100_100070


namespace f_always_positive_l100_100431

noncomputable def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, 0 < f x := by
  sorry

end f_always_positive_l100_100431


namespace bead_arrangement_probability_l100_100769

def total_beads := 6
def red_beads := 2
def white_beads := 2
def blue_beads := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 6  -- Based on valid patterns RWBRWB, RWBWRB, and all other permutations for each starting color

def probability_valid := valid_arrangements / total_arrangements

theorem bead_arrangement_probability : probability_valid = 1 / 15 :=
  by
  -- The context and details of the solution steps are omitted as they are not included in the Lean theorem statement.
  -- This statement will skip the proof
  sorry

end bead_arrangement_probability_l100_100769


namespace common_value_of_7a_and_2b_l100_100039

variable (a b : ℝ)

theorem common_value_of_7a_and_2b (h1 : 7 * a = 2 * b) (h2 : 42 * a * b = 674.9999999999999) :
  7 * a = 15 :=
by
  -- This place will contain the proof steps
  sorry

end common_value_of_7a_and_2b_l100_100039


namespace tape_pieces_needed_l100_100990

-- Define the setup: cube edge length and tape width
def edge_length (n : ℕ) : ℕ := n
def tape_width : ℕ := 1

-- Define the statement we want to prove
theorem tape_pieces_needed (n : ℕ) (h₁ : edge_length n > 0) : 2 * n = 2 * (edge_length n) :=
  by
  sorry

end tape_pieces_needed_l100_100990


namespace num_trailing_zeroes_500_factorial_l100_100492

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end num_trailing_zeroes_500_factorial_l100_100492


namespace sum_of_neg_ints_l100_100429

theorem sum_of_neg_ints (xs : List Int) (h₁ : ∀ x ∈ xs, x < 0)
  (h₂ : ∀ x ∈ xs, 3 < |x| ∧ |x| < 6) : xs.sum = -9 :=
sorry

end sum_of_neg_ints_l100_100429


namespace Hans_current_age_l100_100537

variable {H : ℕ} -- Hans' current age

-- Conditions
def Josiah_age (H : ℕ) := 3 * H
def Hans_age_in_3_years (H : ℕ) := H + 3
def Josiah_age_in_3_years (H : ℕ) := Josiah_age H + 3
def sum_of_ages_in_3_years (H : ℕ) := Hans_age_in_3_years H + Josiah_age_in_3_years H

-- Theorem to prove
theorem Hans_current_age : sum_of_ages_in_3_years H = 66 → H = 15 :=
by
  sorry

end Hans_current_age_l100_100537


namespace hyperbola_focus_proof_l100_100156

noncomputable def hyperbola_focus : ℝ × ℝ :=
  (-3, 2.5 + 2 * Real.sqrt 3)

theorem hyperbola_focus_proof :
  ∃ x y : ℝ, 
  -2 * x^2 + 4 * y^2 - 12 * x - 20 * y + 5 = 0 
  → (x = -3) ∧ (y = 2.5 + 2 * Real.sqrt 3) := 
by 
  sorry

end hyperbola_focus_proof_l100_100156


namespace smaller_of_two_numbers_in_ratio_l100_100884

theorem smaller_of_two_numbers_in_ratio (x y a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : x / y = a / b) (h3 : x + y = c) : 
  min x y = (a * c) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_in_ratio_l100_100884


namespace exponentiation_problem_l100_100856

theorem exponentiation_problem :
  (-0.125 ^ 2003) * (-8 ^ 2004) = -8 := 
sorry

end exponentiation_problem_l100_100856


namespace garden_area_in_square_meters_l100_100065

def garden_width_cm : ℕ := 500
def garden_length_cm : ℕ := 800
def conversion_factor_cm2_to_m2 : ℕ := 10000

theorem garden_area_in_square_meters : (garden_length_cm * garden_width_cm) / conversion_factor_cm2_to_m2 = 40 :=
by
  sorry

end garden_area_in_square_meters_l100_100065


namespace total_number_of_workers_is_49_l100_100306

-- Definitions based on the conditions
def avg_salary_all_workers := 8000
def num_technicians := 7
def avg_salary_technicians := 20000
def avg_salary_non_technicians := 6000

-- Prove that the total number of workers in the workshop is 49
theorem total_number_of_workers_is_49 :
  ∃ W, (avg_salary_all_workers * W = avg_salary_technicians * num_technicians + avg_salary_non_technicians * (W - num_technicians)) ∧ W = 49 := 
sorry

end total_number_of_workers_is_49_l100_100306


namespace buoy_radius_proof_l100_100335

/-
We will define the conditions:
- width: 30 cm
- radius_ice_hole: 15 cm (half of width)
- depth: 12 cm
Then prove the radius of the buoy (r) equals 15.375 cm.
-/
noncomputable def radius_of_buoy : ℝ :=
  let width : ℝ := 30
  let depth : ℝ := 12
  let radius_ice_hole : ℝ := width / 2
  let r : ℝ := (369 / 24)
  r    -- the radius of the buoy

theorem buoy_radius_proof : radius_of_buoy = 15.375 :=
by 
  -- We assert that the above definition correctly computes the radius.
  sorry   -- Actual proof omitted

end buoy_radius_proof_l100_100335


namespace binary_11101_to_decimal_l100_100690

theorem binary_11101_to_decimal : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 29) := by
  sorry

end binary_11101_to_decimal_l100_100690


namespace path_shorter_factor_l100_100029

-- Declare variables
variables (x y z : ℝ)

-- Define conditions as hypotheses
def condition1 := x = 3 * (y + z)
def condition2 := 4 * y = z + x

-- State the proof statement
theorem path_shorter_factor (condition1 : x = 3 * (y + z)) (condition2 : 4 * y = z + x) :
  (4 * y) / z = 19 :=
sorry

end path_shorter_factor_l100_100029


namespace least_number_divisible_l100_100222

theorem least_number_divisible (x : ℕ) (h1 : x = 857) 
  (h2 : (x + 7) % 24 = 0) 
  (h3 : (x + 7) % 36 = 0) 
  (h4 : (x + 7) % 54 = 0) :
  (x + 7) % 32 = 0 := 
sorry

end least_number_divisible_l100_100222


namespace KodyAgeIs32_l100_100434

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l100_100434


namespace combined_weight_l100_100785

theorem combined_weight (x y z : ℕ) (h1 : x + z = 78) (h2 : x + y = 69) (h3 : y + z = 137) : x + y + z = 142 :=
by
  -- Intermediate steps or any additional lemmas could go here
sorry

end combined_weight_l100_100785


namespace number_of_5_dollar_coins_l100_100207

-- Define the context and the proof problem
theorem number_of_5_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by sorry

end number_of_5_dollar_coins_l100_100207


namespace min_h_for_circle_l100_100815

theorem min_h_for_circle (h : ℝ) :
  (∀ x y : ℝ, (x - h)^2 + (y - 1)^2 = 1 → x + y + 1 ≥ 0) →
  h = Real.sqrt 2 - 2 :=
sorry

end min_h_for_circle_l100_100815


namespace find_a_b_l100_100622

theorem find_a_b (a b : ℝ) (h₁ : a^2 = 64 * b) (h₂ : a^2 = 4 * b) : a = 0 ∧ b = 0 :=
by
  sorry

end find_a_b_l100_100622


namespace problem1_problem2_l100_100109

open Nat

def binomial (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem problem1 : binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006 := by
  sorry

theorem problem2 : binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32 := by
  sorry

end problem1_problem2_l100_100109


namespace cafeteria_pies_l100_100857

theorem cafeteria_pies (total_apples handed_out_apples apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_apples = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_apples) / apples_per_pie = 5 :=
by {
  sorry
}

end cafeteria_pies_l100_100857


namespace find_b_l100_100225

theorem find_b (g : ℝ → ℝ) (g_inv : ℝ → ℝ) (b : ℝ) (h_g_def : ∀ x, g x = 1 / (3 * x + b)) (h_g_inv_def : ∀ x, g_inv x = (1 - 3 * x) / (3 * x)) :
  b = 3 :=
by
  sorry

end find_b_l100_100225


namespace total_hours_watched_l100_100459

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l100_100459


namespace apples_per_pie_l100_100681

-- Definitions of the conditions
def number_of_pies : ℕ := 10
def harvested_apples : ℕ := 50
def to_buy_apples : ℕ := 30
def total_apples_needed : ℕ := harvested_apples + to_buy_apples

-- The theorem to prove
theorem apples_per_pie :
  (total_apples_needed / number_of_pies) = 8 := 
sorry

end apples_per_pie_l100_100681


namespace range_of_a_l100_100875

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l100_100875


namespace fraction_of_population_married_l100_100671

theorem fraction_of_population_married
  (M W N : ℕ)
  (h1 : (2 / 3 : ℚ) * M = N)
  (h2 : (3 / 5 : ℚ) * W = N)
  : ((2 * N) : ℚ) / (M + W) = 12 / 19 := 
by
  sorry

end fraction_of_population_married_l100_100671


namespace determine_f_when_alpha_l100_100628

noncomputable def solves_functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
∀ (x y : ℝ), 0 < x → 0 < y → f (f x + y) = α * x + 1 / (f (1 / y))

theorem determine_f_when_alpha (α : ℝ) (f : ℝ → ℝ) :
  (α = 1 → ∀ x, 0 < x → f x = x) ∧ (α ≠ 1 → ∀ f, ¬ solves_functional_equation f α) := by
  sorry

end determine_f_when_alpha_l100_100628


namespace speed_of_boat_is_correct_l100_100806

theorem speed_of_boat_is_correct (t : ℝ) (V_b : ℝ) (V_s : ℝ) 
  (h1 : V_s = 19) 
  (h2 : ∀ t, (V_b - V_s) * (2 * t) = (V_b + V_s) * t) :
  V_b = 57 :=
by
  -- Proof will go here
  sorry

end speed_of_boat_is_correct_l100_100806


namespace fraction_of_bikinis_or_trunks_l100_100788

theorem fraction_of_bikinis_or_trunks (h_bikinis : Real := 0.38) (h_trunks : Real := 0.25) :
  h_bikinis + h_trunks = 0.63 :=
by
  sorry

end fraction_of_bikinis_or_trunks_l100_100788


namespace calculation_correct_l100_100984

theorem calculation_correct : 
  ((2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7)) = 45 := by
  sorry

end calculation_correct_l100_100984


namespace Cara_skate_distance_l100_100737

-- Definitions corresponding to the conditions
def distance_CD : ℝ := 150
def speed_Cara : ℝ := 10
def speed_Dan : ℝ := 6
def angle_Cara_CD : ℝ := 45

-- main theorem based on the problem and given conditions
theorem Cara_skate_distance : ∃ t : ℝ, distance_CD = 150 ∧ speed_Cara = 10 ∧ speed_Dan = 6
                            ∧ angle_Cara_CD = 45 
                            ∧ 10 * t = 253.5 :=
by
  sorry

end Cara_skate_distance_l100_100737


namespace expression_evaluation_correct_l100_100583

theorem expression_evaluation_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ( ( ( (x - 2) ^ 2 * (x ^ 2 + x + 1) ^ 2 ) / (x ^ 3 - 1) ^ 2 ) ^ 2 *
    ( ( (x + 2) ^ 2 * (x ^ 2 - x + 1) ^ 2 ) / (x ^ 3 + 1) ^ 2 ) ^ 2 ) 
  = (x^2 - 4)^4 := 
sorry

end expression_evaluation_correct_l100_100583


namespace seat_notation_l100_100883

theorem seat_notation (row1 col1 row2 col2 : ℕ) (h : (row1, col1) = (5, 2)) : (row2, col2) = (7, 3) :=
 by
  sorry

end seat_notation_l100_100883


namespace ball_max_height_l100_100395

theorem ball_max_height : 
  (∃ t : ℝ, 
    ∀ u : ℝ, -16 * u ^ 2 + 80 * u + 35 ≤ -16 * t ^ 2 + 80 * t + 35 ∧ 
    -16 * t ^ 2 + 80 * t + 35 = 135) :=
sorry

end ball_max_height_l100_100395


namespace solve_x_l100_100955

theorem solve_x : ∃ x : ℝ, 65 + (5 * x) / (180 / 3) = 66 ∧ x = 12 := by
  sorry

end solve_x_l100_100955


namespace find_x_l100_100542

-- Define the operation "※" as given
def star (a b : ℕ) : ℚ := (a + 2 * b) / 3

-- Given that 6 ※ x = 22 / 3, prove that x = 8
theorem find_x : ∃ x : ℕ, star 6 x = 22 / 3 ↔ x = 8 :=
by
  sorry -- Proof not required

end find_x_l100_100542


namespace walking_rate_ratio_l100_100515

variables (R R' : ℝ)

theorem walking_rate_ratio (h₁ : R * 21 = R' * 18) : R' / R = 7 / 6 :=
by {
  sorry
}

end walking_rate_ratio_l100_100515


namespace problem_xy_minimized_problem_x_y_minimized_l100_100228

open Real

theorem problem_xy_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 16 ∧ y = 2 ∧ x * y = 32 := 
sorry

theorem problem_x_y_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 8 + 2 * sqrt 2 ∧ y = 1 + sqrt 2 ∧ x + y = 9 + 4 * sqrt 2 := 
sorry

end problem_xy_minimized_problem_x_y_minimized_l100_100228


namespace num_8tuples_satisfying_condition_l100_100969

theorem num_8tuples_satisfying_condition :
  (∃! (y : Fin 8 → ℝ),
    (2 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + 
    (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + 
    (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 4 / 9) :=
sorry

end num_8tuples_satisfying_condition_l100_100969


namespace alyssa_puppies_l100_100107

theorem alyssa_puppies (initial now given : ℕ) (h1 : initial = 12) (h2 : now = 5) : given = 7 :=
by
  have h3 : given = initial - now := by sorry
  rw [h1, h2] at h3
  exact h3

end alyssa_puppies_l100_100107


namespace area_within_fence_l100_100824

def length_rectangle : ℕ := 15
def width_rectangle : ℕ := 12
def side_cutout_square : ℕ := 3

theorem area_within_fence : (length_rectangle * width_rectangle) - (side_cutout_square * side_cutout_square) = 171 := by
  sorry

end area_within_fence_l100_100824


namespace system_of_equations_solution_l100_100133

theorem system_of_equations_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -8) ∧ (5 * x + 9 * y = -18) ∧ x = -14 / 3 ∧ y = -32 / 9 :=
by {
  sorry  -- Proof goes here
}

end system_of_equations_solution_l100_100133


namespace inequality_proof_l100_100633

theorem inequality_proof {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 :=
by
  -- Proof goes here
  sorry

end inequality_proof_l100_100633


namespace milton_zoology_books_l100_100512

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l100_100512


namespace sum_of_areas_l100_100851

theorem sum_of_areas (radii : ℕ → ℝ) (areas : ℕ → ℝ) (h₁ : radii 0 = 2) 
  (h₂ : ∀ n, radii (n + 1) = radii n / 3) 
  (h₃ : ∀ n, areas n = π * (radii n) ^ 2) : 
  ∑' n, areas n = (9 * π) / 2 := 
by 
  sorry

end sum_of_areas_l100_100851


namespace cannot_all_be_zero_l100_100864

theorem cannot_all_be_zero :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, f i ∈ { x : ℕ | 1 ≤ x ∧ x ≤ 1989 }) ∧
                   (∀ i j, f (i + j) = f i - f j) ∧
                   (∃ n, ∀ i, f (i + n) = 0) :=
by
  sorry

end cannot_all_be_zero_l100_100864


namespace quadratic_real_roots_condition_sufficient_l100_100611

theorem quadratic_real_roots_condition_sufficient (m : ℝ) : (m < 1 / 4) → ∃ x : ℝ, x^2 + x + m = 0 :=
by
  sorry

end quadratic_real_roots_condition_sufficient_l100_100611


namespace polygons_sides_l100_100725

def sum_of_angles (x y : ℕ) : ℕ :=
(x - 2) * 180 + (y - 2) * 180

def num_diagonals (x y : ℕ) : ℕ :=
x * (x - 3) / 2 + y * (y - 3) / 2

theorem polygons_sides (x y : ℕ) (hx : x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99) 
(hs : sum_of_angles x y = 21 * (x + y + num_diagonals x y) - 39) :
x = 17 ∧ y = 3 ∨ x = 3 ∧ y = 17 :=
by
  sorry

end polygons_sides_l100_100725


namespace smallest_value_N_l100_100141

theorem smallest_value_N (N : ℕ) (a b c : ℕ) (h1 : N = a * b * c) (h2 : (a - 1) * (b - 1) * (c - 1) = 252) : N = 392 :=
sorry

end smallest_value_N_l100_100141


namespace original_price_of_cycle_l100_100044

theorem original_price_of_cycle (SP : ℕ) (P : ℕ) (h1 : SP = 1800) (h2 : SP = 9 * P / 10) : P = 2000 :=
by
  have hSP_eq : SP = 1800 := h1
  have hSP_def : SP = 9 * P / 10 := h2
  -- Now we need to combine these to prove P = 2000
  sorry

end original_price_of_cycle_l100_100044


namespace jane_total_drying_time_l100_100283

theorem jane_total_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let nail_art_1 := 8
  let nail_art_2 := 10
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + nail_art_1 + nail_art_2 + top_coat = 49 :=
by 
  sorry

end jane_total_drying_time_l100_100283


namespace number_of_shelves_l100_100338

def initial_bears : ℕ := 17
def shipment_bears : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 3 :=
by
  sorry

end number_of_shelves_l100_100338


namespace arithmetic_sequence_a14_l100_100054

theorem arithmetic_sequence_a14 (a : ℕ → ℤ) (h1 : a 4 = 5) (h2 : a 9 = 17) (h3 : 2 * a 9 = a 14 + a 4) : a 14 = 29 := sorry

end arithmetic_sequence_a14_l100_100054


namespace value_of_y_l100_100659

theorem value_of_y (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 :=
by
  sorry

end value_of_y_l100_100659


namespace ratio_of_areas_is_one_ninth_l100_100027

-- Define the side lengths of Square A and Square B
variables (x : ℝ)
def side_length_a := x
def side_length_b := 3 * x

-- Define the areas of Square A and Square B
def area_a := side_length_a x * side_length_a x
def area_b := side_length_b x * side_length_b x

-- The theorem to prove the ratio of areas
theorem ratio_of_areas_is_one_ninth : (area_a x) / (area_b x) = (1 / 9) :=
by sorry

end ratio_of_areas_is_one_ninth_l100_100027


namespace c_work_rate_l100_100347

theorem c_work_rate {A B C : ℚ} (h1 : A + B = 1/6) (h2 : B + C = 1/8) (h3 : C + A = 1/12) : C = 1/48 :=
by
  sorry

end c_work_rate_l100_100347


namespace combined_weight_is_150_l100_100131

-- Definitions based on conditions
def tracy_weight : ℕ := 52
def jake_weight : ℕ := tracy_weight + 8
def weight_range : ℕ := 14
def john_weight : ℕ := tracy_weight - 14

-- Proving the combined weight
theorem combined_weight_is_150 :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

end combined_weight_is_150_l100_100131


namespace exists_root_interval_l100_100442

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l100_100442


namespace count_special_positive_integers_l100_100329

theorem count_special_positive_integers : 
  ∃! n : ℕ, n < 10^6 ∧ 
  ∃ a b : ℕ, n = 2 * a^2 ∧ n = 3 * b^3 ∧ 
  ((n = 2592) ∨ (n = 165888)) :=
by
  sorry

end count_special_positive_integers_l100_100329


namespace part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l100_100906

-- Define the game rules and conditions for the proof
def takeMatches (total_matches : Nat) (taken_matches : Nat) : Nat :=
  total_matches - taken_matches

-- Part (a) statement
theorem part_a_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (b) statement
theorem part_b_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (c) generalized statement for game type (a)
theorem part_c_winner_a (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

-- Part (c) generalized statement for game type (b)
theorem part_c_winner_b (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

end part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l100_100906


namespace f_2013_eq_2_l100_100858

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = -f x
axiom h2 : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom h3 : f (-1) = -2

theorem f_2013_eq_2 : f 2013 = 2 := 
by 
  sorry

end f_2013_eq_2_l100_100858


namespace race_distance_correct_l100_100485

noncomputable def solve_race_distance : ℝ :=
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs

  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  Dp

theorem race_distance_correct :
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs
  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  time_p = time_q := by
  sorry

end race_distance_correct_l100_100485


namespace total_winter_clothing_l100_100189

theorem total_winter_clothing (boxes : ℕ) (scarves_per_box mittens_per_box : ℕ) (h_boxes : boxes = 8) (h_scarves : scarves_per_box = 4) (h_mittens : mittens_per_box = 6) : 
  boxes * (scarves_per_box + mittens_per_box) = 80 := 
by
  sorry

end total_winter_clothing_l100_100189


namespace johny_distance_l100_100227

noncomputable def distance_south : ℕ := 40
variable (E : ℕ)
noncomputable def distance_east : ℕ := E
noncomputable def distance_north (E : ℕ) : ℕ := 2 * E
noncomputable def total_distance (E : ℕ) : ℕ := distance_south + distance_east E + distance_north E

theorem johny_distance :
  ∀ E : ℕ, total_distance E = 220 → E - distance_south = 20 :=
by
  intro E
  intro h
  rw [total_distance, distance_north, distance_east, distance_south] at h
  sorry

end johny_distance_l100_100227


namespace sqrt_30_estimate_l100_100916

theorem sqrt_30_estimate : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end sqrt_30_estimate_l100_100916


namespace additional_donation_l100_100326

theorem additional_donation
  (t : ℕ) (c d₁ d₂ T a : ℝ)
  (h1 : t = 25)
  (h2 : c = 2.00)
  (h3 : d₁ = 15.00) 
  (h4 : d₂ = 15.00)
  (h5 : T = 100.00)
  (h6 : t * c + d₁ + d₂ + a = T) :
  a = 20.00 :=
by
  sorry

end additional_donation_l100_100326


namespace bridget_apples_l100_100177

/-!
# Problem statement
Bridget bought a bag of apples. She gave half of the apples to Ann. She gave 5 apples to Cassie,
and 2 apples to Dan. She kept 6 apples for herself. Prove that Bridget originally bought 26 apples.
-/

theorem bridget_apples (x : ℕ) 
  (H1 : x / 2 + 2 * (x % 2) / 2 - 5 - 2 = 6) : x = 26 :=
sorry

end bridget_apples_l100_100177


namespace quadratic_inequality_solution_l100_100814

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), x^2 - 9 * x + 14 ≤ 0 → 2 ≤ x ∧ x ≤ 7 :=
by
  intros x h
  sorry

end quadratic_inequality_solution_l100_100814


namespace multiply_transformed_l100_100759

theorem multiply_transformed : (268 * 74 = 19832) → (2.68 * 0.74 = 1.9832) :=
by
  intro h
  sorry

end multiply_transformed_l100_100759


namespace find_perpendicular_line_l100_100535

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l100_100535


namespace number_of_blue_faces_l100_100557

theorem number_of_blue_faces (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end number_of_blue_faces_l100_100557


namespace part1_growth_rate_part2_new_price_l100_100047

-- Definitions based on conditions
def purchase_price : ℕ := 30
def selling_price : ℕ := 40
def january_sales : ℕ := 400
def march_sales : ℕ := 576
def growth_rate (x : ℝ) : Prop := january_sales * (1 + x)^2 = march_sales

-- Part (1): Prove the monthly average growth rate
theorem part1_growth_rate : 
  ∃ (x : ℝ), growth_rate x ∧ x = 0.2 :=
by
  sorry

-- Definitions for part (2) - based on the second condition
def price_reduction (y : ℝ) : Prop := (selling_price - y - purchase_price) * (march_sales + 12 * y) = 4800

-- Part (2): Prove the new price for April
theorem part2_new_price :
  ∃ (y : ℝ), price_reduction y ∧ (selling_price - y) = 38 :=
by
  sorry

end part1_growth_rate_part2_new_price_l100_100047


namespace find_x_in_gp_l100_100074

theorem find_x_in_gp :
  ∃ x : ℤ, (30 + x)^2 = (10 + x) * (90 + x) ∧ x = 0 :=
by
  sorry

end find_x_in_gp_l100_100074


namespace cylinder_height_l100_100930

variable (r h : ℝ) (SA : ℝ)

theorem cylinder_height (h : ℝ) (r : ℝ) (SA : ℝ) (h_eq : h = 2) (r_eq : r = 3) (SA_eq : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h → h = 2 :=
by
  intros
  sorry

end cylinder_height_l100_100930


namespace shaded_region_area_computed_correctly_l100_100580

noncomputable def side_length : ℝ := 15
noncomputable def quarter_circle_radius : ℝ := side_length / 3
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
noncomputable def shaded_region_area : ℝ := square_area - circle_area

theorem shaded_region_area_computed_correctly : 
  shaded_region_area = 225 - 25 * Real.pi := 
by 
  -- This statement only defines the proof problem.
  sorry

end shaded_region_area_computed_correctly_l100_100580


namespace overlap_length_l100_100738

noncomputable def length_of_all_red_segments := 98 -- in cm
noncomputable def total_length := 83 -- in cm
noncomputable def number_of_overlaps := 6 -- count

theorem overlap_length :
  ∃ (x : ℝ), length_of_all_red_segments - total_length = number_of_overlaps * x ∧ x = 2.5 := by
  sorry

end overlap_length_l100_100738


namespace total_rent_calculation_l100_100169

variables (x y : ℕ) -- x: number of rooms rented for $40, y: number of rooms rented for $60
variable (rent_total : ℕ)

-- Condition: Each room at the motel was rented for either $40 or $60
-- Condition: If 10 of the rooms that were rented for $60 had instead been rented for $40, the total rent would have been reduced by 50 percent

theorem total_rent_calculation 
  (h1 : 40 * (x + 10) + 60 * (y - 10) = (40 * x + 60 * y) / 2) :
  40 * x + 60 * y = 800 :=
sorry

end total_rent_calculation_l100_100169


namespace locus_of_midpoint_of_tangents_l100_100457

theorem locus_of_midpoint_of_tangents 
  (P Q Q1 Q2 : ℝ × ℝ)
  (L : P.2 = P.1 + 2)
  (C : ∀ p, p = Q1 ∨ p = Q2 → p.2 ^ 2 = 4 * p.1)
  (Q_is_midpoint : Q = ((Q1.1 + Q2.1) / 2, (Q1.2 + Q2.2) / 2)) :
  ∃ x y, (y - 1)^2 = 2 * (x - 3 / 2) := sorry

end locus_of_midpoint_of_tangents_l100_100457


namespace evaluate_heartsuit_l100_100843

-- Define the given operation
def heartsuit (x y : ℝ) : ℝ := abs (x - y)

-- State the proof problem in Lean
theorem evaluate_heartsuit (a b : ℝ) (h_a : a = 3) (h_b : b = -1) :
  heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b)) = 4 :=
by
  -- acknowledging that it's correct without providing the solution steps
  sorry

end evaluate_heartsuit_l100_100843


namespace find_abc_l100_100845

noncomputable def f (a b c x : ℝ) := x^3 + a*x^2 + b*x + c
noncomputable def f' (a b x : ℝ) := 3*x^2 + 2*a*x + b

theorem find_abc (a b c : ℝ) :
  (f' a b -2 = 0) ∧
  (f' a b 1 = -3) ∧
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 :=
sorry

end find_abc_l100_100845


namespace find_c_l100_100820

theorem find_c 
  (b c : ℝ) 
  (h1 : 4 = 2 * (1:ℝ)^2 + b * (1:ℝ) + c)
  (h2 : 4 = 2 * (5:ℝ)^2 + b * (5:ℝ) + c) : 
  c = 14 := 
sorry

end find_c_l100_100820


namespace one_fourth_of_2_pow_30_eq_2_pow_x_l100_100364

theorem one_fourth_of_2_pow_30_eq_2_pow_x (x : ℕ) : (1 / 4 : ℝ) * (2:ℝ)^30 = (2:ℝ)^x → x = 28 := by
  sorry

end one_fourth_of_2_pow_30_eq_2_pow_x_l100_100364


namespace courtyard_brick_problem_l100_100073

noncomputable def area_courtyard (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_brick (length width : ℝ) : ℝ :=
  length * width

noncomputable def total_bricks_required (court_area brick_area : ℝ) : ℝ :=
  court_area / brick_area

theorem courtyard_brick_problem 
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ)
  (H1 : courtyard_length = 18)
  (H2 : courtyard_width = 12)
  (H3 : brick_length = 15 / 100)
  (H4 : brick_width = 13 / 100) :
  
  total_bricks_required (area_courtyard courtyard_length courtyard_width * 10000) 
                        (area_brick brick_length brick_width) 
  = 11077 :=
by
  sorry

end courtyard_brick_problem_l100_100073


namespace total_animals_in_farm_l100_100134

theorem total_animals_in_farm (C B : ℕ) (h1 : C = 5) (h2 : 2 * C + 4 * B = 26) : C + B = 9 :=
by
  sorry

end total_animals_in_farm_l100_100134


namespace natural_number_factors_of_M_l100_100098

def M : ℕ := (2^3) * (3^2) * (5^5) * (7^1) * (11^2)

theorem natural_number_factors_of_M : ∃ n : ℕ, n = 432 ∧ (∀ d, d ∣ M → d > 0 → d ≤ M) :=
by
  let number_of_factors := (3 + 1) * (2 + 1) * (5 + 1) * (1 + 1) * (2 + 1)
  use number_of_factors
  sorry

end natural_number_factors_of_M_l100_100098


namespace trigonometric_expression_value_l100_100000

variable {α : ℝ}
axiom tan_alpha_eq : Real.tan α = 2

theorem trigonometric_expression_value :
  (1 + 2 * Real.cos (Real.pi / 2 - α) * Real.cos (-10 * Real.pi - α)) /
  (Real.cos (3 * Real.pi / 2 - α) ^ 2 - Real.sin (9 * Real.pi / 2 - α) ^ 2) = 3 :=
by
  have h_tan_alpha : Real.tan α = 2 := tan_alpha_eq
  sorry

end trigonometric_expression_value_l100_100000


namespace find_numbers_l100_100085

theorem find_numbers (x y z : ℕ) :
  x + y + z = 35 → 
  2 * y = x + z + 1 → 
  y^2 = (x + 3) * z → 
  (x = 15 ∧ y = 12 ∧ z = 8) ∨ (x = 5 ∧ y = 12 ∧ z = 18) :=
by
  sorry

end find_numbers_l100_100085


namespace find_c1_in_polynomial_q_l100_100006

theorem find_c1_in_polynomial_q
  (m : ℕ)
  (hm : m ≥ 5)
  (hm_odd : m % 2 = 1)
  (D : ℕ → ℕ)
  (hD_q : ∃ (c3 c2 c1 c0 : ℤ), ∀ (m : ℕ), m % 2 = 1 ∧ m ≥ 5 → D m = (c3 * m^3 + c2 * m^2 + c1 * m + c0)) :
  ∃ (c1 : ℤ), c1 = 11 :=
sorry

end find_c1_in_polynomial_q_l100_100006


namespace find_coefficients_sum_l100_100387

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 : ℝ) (h : ∀ x : ℝ, x^3 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3) :
  a_1 + a_2 + a_3 = 19 :=
by
  sorry

end find_coefficients_sum_l100_100387


namespace find_a_m_range_c_l100_100954

noncomputable def f (x a : ℝ) := x^2 - 2*x + 2*a
def solution_set (f : ℝ → ℝ) (m : ℝ) := {x : ℝ | -2 ≤ x ∧ x ≤ m ∧ f x ≤ 0}

theorem find_a_m (a m : ℝ) : 
  (∀ x, f x a ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) → a = -4 ∧ m = 4 := by
  sorry

theorem range_c (c : ℝ) : 
  (∀ x, (c - 4) * x^2 + 2 * (c - 4) * x - 1 < 0) → 13 / 4 < c ∧ c < 4 := by
  sorry

end find_a_m_range_c_l100_100954


namespace maximum_value_of_sums_of_cubes_l100_100089

theorem maximum_value_of_sums_of_cubes 
  (a b c d e : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
sorry

end maximum_value_of_sums_of_cubes_l100_100089


namespace tan_alpha_plus_pi_div_four_l100_100481

theorem tan_alpha_plus_pi_div_four (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := 
by
  sorry

end tan_alpha_plus_pi_div_four_l100_100481


namespace numberOfRealSolutions_l100_100721

theorem numberOfRealSolutions :
  ∀ (x : ℝ), (-4*x + 12)^2 + 1 = (x - 1)^2 → (∃ a b : ℝ, (a ≠ b) ∧ (-4*a + 12)^2 + 1 = (a - 1)^2 ∧ (-4*b + 12)^2 + 1 = (b - 1)^2) := by
  sorry

end numberOfRealSolutions_l100_100721


namespace find_a_l100_100161

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem find_a (a : ℝ) : A ⊆ B a → a = 3 :=
by
  intro h
  sorry

end find_a_l100_100161


namespace diagonal_of_larger_screen_l100_100723

theorem diagonal_of_larger_screen (d : ℝ) 
  (h1 : ∃ s : ℝ, s^2 = 20^2 + 42) 
  (h2 : ∀ s, d = s * Real.sqrt 2) : 
  d = Real.sqrt 884 :=
by
  sorry

end diagonal_of_larger_screen_l100_100723


namespace smallest_a_gcd_77_88_l100_100418

theorem smallest_a_gcd_77_88 :
  ∃ (a : ℕ), a > 0 ∧ (∀ b, b > 0 → b < a → (gcd b 77 > 1 ∧ gcd b 88 > 1) → false) ∧ gcd a 77 > 1 ∧ gcd a 88 > 1 ∧ a = 11 :=
by
  sorry

end smallest_a_gcd_77_88_l100_100418


namespace cricket_initial_overs_l100_100901

/-- Prove that the number of initial overs played was 10. -/
theorem cricket_initial_overs 
  (target : ℝ) 
  (initial_run_rate : ℝ) 
  (remaining_run_rate : ℝ) 
  (remaining_overs : ℕ)
  (h_target : target = 282)
  (h_initial_run_rate : initial_run_rate = 4.6)
  (h_remaining_run_rate : remaining_run_rate = 5.9)
  (h_remaining_overs : remaining_overs = 40) 
  : ∃ x : ℝ, x = 10 := 
by
  sorry

end cricket_initial_overs_l100_100901


namespace average_seeds_per_apple_l100_100646

-- Define the problem conditions and the proof statement

theorem average_seeds_per_apple
  (A : ℕ)
  (total_seeds_requirement : ℕ := 60)
  (pear_seeds_avg : ℕ := 2)
  (grape_seeds_avg : ℕ := 3)
  (num_apples : ℕ := 4)
  (num_pears : ℕ := 3)
  (num_grapes : ℕ := 9)
  (shortfall : ℕ := 3)
  (collected_seeds : ℕ := num_apples * A + num_pears * pear_seeds_avg + num_grapes * grape_seeds_avg)
  (required_seeds : ℕ := total_seeds_requirement - shortfall) :
  collected_seeds = required_seeds → A = 6 := 
by
  sorry

end average_seeds_per_apple_l100_100646


namespace vector_addition_l100_100584

def v1 : ℝ × ℝ := (3, -6)
def v2 : ℝ × ℝ := (2, -9)
def v3 : ℝ × ℝ := (-1, 3)
def c1 : ℝ := 4
def c2 : ℝ := 5
def result : ℝ × ℝ := (23, -72)

theorem vector_addition :
  c1 • v1 + c2 • v2 - v3 = result :=
by
  sorry

end vector_addition_l100_100584


namespace find_a_l100_100519

theorem find_a (a b c : ℕ) (h₁ : a + b = c) (h₂ : b + 2 * c = 10) (h₃ : c = 4) : a = 2 := by
  sorry

end find_a_l100_100519


namespace area_of_support_is_15_l100_100467

-- Define the given conditions
def initial_mass : ℝ := 60
def reduced_mass : ℝ := initial_mass - 10
def area_reduction : ℝ := 5
def mass_per_area_increase : ℝ := 1

-- Define the area of the support and prove that it is 15 dm^2
theorem area_of_support_is_15 (x : ℝ) 
  (initial_mass_eq : initial_mass / x = initial_mass / x) 
  (new_mass_eq : reduced_mass / (x - area_reduction) = initial_mass / x + mass_per_area_increase) : 
  x = 15 :=
  sorry

end area_of_support_is_15_l100_100467


namespace andy_tomatoes_left_l100_100688

theorem andy_tomatoes_left :
  let plants := 50
  let tomatoes_per_plant := 15
  let total_tomatoes := plants * tomatoes_per_plant
  let tomatoes_dried := (2 / 3) * total_tomatoes
  let tomatoes_left_after_drying := total_tomatoes - tomatoes_dried
  let tomatoes_for_marinara := (1 / 2) * tomatoes_left_after_drying
  let tomatoes_left := tomatoes_left_after_drying - tomatoes_for_marinara
  tomatoes_left = 125 := sorry

end andy_tomatoes_left_l100_100688


namespace triangle_area_proof_l100_100523

noncomputable def triangle_area (a b c C : ℝ) : ℝ := 0.5 * a * b * Real.sin C

theorem triangle_area_proof:
  ∀ (A B C a b c : ℝ),
  ¬ (C = π/2) ∧
  c = 1 ∧
  C = π/3 ∧
  Real.sin C + Real.sin (A - B) = 3 * Real.sin (2*B) →
  triangle_area a b c C = 3 * Real.sqrt 3 / 28 :=
by
  intros A B C a b c h
  sorry

end triangle_area_proof_l100_100523


namespace train_pass_time_eq_4_seconds_l100_100691

-- Define the length of the train in meters
def train_length : ℕ := 40

-- Define the speed of the train in km/h
def train_speed_kmph : ℕ := 36

-- Conversion factor: 1 kmph = 1000 meters / 3600 seconds
def conversion_factor : ℚ := 1000 / 3600

-- Convert the train's speed from km/h to m/s
def train_speed_mps : ℚ := train_speed_kmph * conversion_factor

-- Calculate the time to pass the telegraph post
def time_to_pass_post : ℚ := train_length / train_speed_mps

-- The goal: prove the actual time is 4 seconds
theorem train_pass_time_eq_4_seconds : time_to_pass_post = 4 := by
  sorry

end train_pass_time_eq_4_seconds_l100_100691


namespace complex_magnitude_problem_l100_100008

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l100_100008


namespace find_two_irreducible_fractions_l100_100735

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l100_100735


namespace gcd_lcm_of_consecutive_naturals_l100_100992

theorem gcd_lcm_of_consecutive_naturals (m : ℕ) (h : m > 0) (n : ℕ) (hn : n = m + 1) :
  gcd m n = 1 ∧ lcm m n = m * n :=
by
  sorry

end gcd_lcm_of_consecutive_naturals_l100_100992


namespace min_bottles_to_fill_large_bottle_l100_100912

theorem min_bottles_to_fill_large_bottle (large_bottle_ml : Nat) (small_bottle1_ml : Nat) (small_bottle2_ml : Nat) (total_bottles : Nat) :
  large_bottle_ml = 800 ∧ small_bottle1_ml = 45 ∧ small_bottle2_ml = 60 ∧ total_bottles = 14 →
  ∃ x y : Nat, x * small_bottle1_ml + y * small_bottle2_ml = large_bottle_ml ∧ x + y = total_bottles :=
by
  intro h
  sorry

end min_bottles_to_fill_large_bottle_l100_100912


namespace union_of_A_and_B_l100_100010

def setA : Set ℝ := { x | -3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3 }
def setB : Set ℝ := { x | 1 < x }

theorem union_of_A_and_B :
  setA ∪ setB = { x | -1 ≤ x } := sorry

end union_of_A_and_B_l100_100010


namespace smallest_natural_number_k_l100_100539

theorem smallest_natural_number_k :
  ∃ k : ℕ, k = 4 ∧ ∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ n → a^(k) * (1 - a)^(n) < 1 / (n + 1)^3 :=
by
  sorry

end smallest_natural_number_k_l100_100539


namespace y_coordinates_difference_l100_100309

theorem y_coordinates_difference {m n k : ℤ}
  (h1 : m = 2 * n + 5)
  (h2 : m + 4 = 2 * (n + k) + 5) :
  k = 2 :=
by
  sorry

end y_coordinates_difference_l100_100309


namespace swimming_pool_area_l100_100388

open Nat

-- Define the width (w) and length (l) with given conditions
def width (w : ℕ) : Prop :=
  exists (l : ℕ), l = 2 * w + 40 ∧ 2 * w + 2 * l = 800

-- Define the area of the swimming pool
def pool_area (w l : ℕ) : ℕ :=
  w * l

theorem swimming_pool_area : 
  ∃ (w l : ℕ), width w ∧ width l -> pool_area w l = 33600 :=
by
  sorry

end swimming_pool_area_l100_100388


namespace flour_needed_l100_100300

theorem flour_needed (flour_per_40_cookies : ℝ) (cookies : ℕ) (desired_cookies : ℕ) (flour_needed : ℝ) 
  (h1 : flour_per_40_cookies = 3) (h2 : cookies = 40) (h3 : desired_cookies = 100) :
  flour_needed = 7.5 :=
by
  sorry

end flour_needed_l100_100300


namespace correct_exponent_operation_l100_100242

theorem correct_exponent_operation (a b : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  (6 * a^6 / (2 * a^2) ≠ 3 * a^3) ∧ 
  ((-a^2)^3 = -a^6) ∧ 
  ((-2 * a * b^2)^2 ≠ 2 * a^2 * b^4) :=
by
  sorry

end correct_exponent_operation_l100_100242


namespace probability_of_B_not_losing_is_70_l100_100731

-- Define the probabilities as given in the conditions
def prob_A_winning : ℝ := 0.30
def prob_draw : ℝ := 0.50

-- Define the probability of B not losing
def prob_B_not_losing : ℝ := 0.50 + (1 - prob_A_winning - prob_draw)

-- State the theorem
theorem probability_of_B_not_losing_is_70 :
  prob_B_not_losing = 0.70 := by
  sorry -- Proof to be filled in

end probability_of_B_not_losing_is_70_l100_100731


namespace integer_value_of_K_l100_100736

theorem integer_value_of_K (K : ℤ) : 
  (1000 < K^4 ∧ K^4 < 5000) ∧ K > 1 → K = 6 ∨ K = 7 ∨ K = 8 :=
by sorry

end integer_value_of_K_l100_100736


namespace parallelogram_height_l100_100206

theorem parallelogram_height (base height area : ℝ) (h_base : base = 9) (h_area : area = 33.3) (h_formula : area = base * height) : height = 3.7 :=
by
  -- Proof goes here, but currently skipped
  sorry

end parallelogram_height_l100_100206


namespace geometric_increasing_condition_l100_100259

structure GeometricSequence (a₁ q : ℝ) (a : ℕ → ℝ) :=
  (rec_rel : ∀ n : ℕ, a (n + 1) = a n * q)

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a₁ q : ℝ) (a : ℕ → ℝ) (h : GeometricSequence a₁ q a) :
  ¬ (q > 1 ↔ is_increasing a) := sorry

end geometric_increasing_condition_l100_100259


namespace calculate_expression_l100_100378

theorem calculate_expression :
  (5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 : ℝ) = 74 := by
  sorry

end calculate_expression_l100_100378


namespace graph_passes_through_fixed_point_l100_100237

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (∀ x y : ℝ, y = a * x + 2 → (x, y) = (-1, 2))

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
sorry

end graph_passes_through_fixed_point_l100_100237


namespace denny_followers_l100_100005

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l100_100005


namespace mitchell_total_pages_read_l100_100084

def pages_per_chapter : ℕ := 40
def chapters_read_before : ℕ := 10
def pages_read_11th_before : ℕ := 20
def chapters_read_after : ℕ := 2

def total_pages_read := 
  pages_per_chapter * chapters_read_before + pages_read_11th_before + pages_per_chapter * chapters_read_after

theorem mitchell_total_pages_read : total_pages_read = 500 := by
  sorry

end mitchell_total_pages_read_l100_100084


namespace tangent_line_to_ex_l100_100605

theorem tangent_line_to_ex (b : ℝ) : (∃ x0 : ℝ, (∀ x : ℝ, (e^x - e^x0 - (x - x0) * e^x0 = 0) ↔ y = x + b)) → b = 1 :=
by
  sorry

end tangent_line_to_ex_l100_100605


namespace solution_set_of_inequality_l100_100398

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem solution_set_of_inequality :
  {x : ℝ | f (2 * x + 1) + f (1) ≥ 0} = {x : ℝ | -1 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l100_100398


namespace total_glasses_l100_100294

theorem total_glasses
  (x y : ℕ)
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15) :
  12 * x + 16 * y = 480 :=
by
  sorry

end total_glasses_l100_100294


namespace calculate_120_percent_l100_100468

theorem calculate_120_percent (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end calculate_120_percent_l100_100468


namespace expr_value_l100_100811

theorem expr_value : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end expr_value_l100_100811


namespace stamp_arrangements_equals_76_l100_100194

-- Define the conditions of the problem
def stamps_available : List (ℕ × ℕ) := 
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), 
   (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), 
   (17, 17), (18, 18), (19, 19)]

-- Define a function to compute the number of different arrangements
noncomputable def count_stamp_arrangements : ℕ :=
  -- This is a placeholder for the actual implementation
  sorry

-- State the theorem to be proven
theorem stamp_arrangements_equals_76 : count_stamp_arrangements = 76 :=
sorry

end stamp_arrangements_equals_76_l100_100194


namespace range_of_a_l100_100028

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x = 1) ↔ a ≠ 0 := by
sorry

end range_of_a_l100_100028


namespace refreshment_stand_distance_l100_100945

theorem refreshment_stand_distance 
  (A B S : ℝ) -- Positions of the camps and refreshment stand
  (dist_A_highway : A = 400) -- Distance from the first camp to the highway
  (dist_B_A : B = 700) -- Distance from the second camp directly across the highway
  (equidistant : ∀ x, S = x ∧ dist (S, A) = dist (S, B)) : 
  S = 500 := -- Distance from the refreshment stand to each camp is 500 meters
sorry

end refreshment_stand_distance_l100_100945


namespace nec_but_not_suff_condition_l100_100854

variables {p q : Prop}

theorem nec_but_not_suff_condition (hp : ¬p) : 
  (p ∨ q → False) ↔ (¬p) ∧ ¬(¬p → p ∨ q) :=
by {
  sorry
}

end nec_but_not_suff_condition_l100_100854


namespace positive_sequence_unique_l100_100142

theorem positive_sequence_unique (x : Fin 2021 → ℝ) (h : ∀ i : Fin 2020, x i.succ = (x i ^ 3 + 2) / (3 * x i ^ 2)) (h' : x 2020 = x 0) : ∀ i, x i = 1 := by
  sorry

end positive_sequence_unique_l100_100142


namespace debby_bought_bottles_l100_100021

def bottles_per_day : ℕ := 109
def days_lasting : ℕ := 74

theorem debby_bought_bottles : bottles_per_day * days_lasting = 8066 := by
  sorry

end debby_bought_bottles_l100_100021


namespace range_of_a_l100_100541

noncomputable def f : ℝ → ℝ := sorry
variable (f_even : ∀ x : ℝ, f x = f (-x))
variable (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
variable (a : ℝ) (h : f a ≤ f 2)

theorem range_of_a (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
                   (h : f a ≤ f 2) :
                   a ≤ -2 ∨ a ≥ 2 :=
sorry

end range_of_a_l100_100541


namespace remy_sold_110_bottles_l100_100977

theorem remy_sold_110_bottles 
    (price_per_bottle : ℝ)
    (total_evening_sales : ℝ)
    (evening_more_than_morning : ℝ)
    (nick_fewer_than_remy : ℝ)
    (R : ℝ) 
    (total_morning_sales_is : ℝ) :
    price_per_bottle = 0.5 →
    total_evening_sales = 55 →
    evening_more_than_morning = 3 →
    nick_fewer_than_remy = 6 →
    total_morning_sales_is = total_evening_sales - evening_more_than_morning →
    (R * price_per_bottle) + ((R - nick_fewer_than_remy) * price_per_bottle) = total_morning_sales_is →
    R = 110 :=
by
  intros
  sorry

end remy_sold_110_bottles_l100_100977


namespace car_body_mass_l100_100556

theorem car_body_mass (m_model : ℕ) (scale : ℕ) : 
  m_model = 1 → scale = 11 → m_car = 1331 :=
by 
  intros h1 h2
  sorry

end car_body_mass_l100_100556


namespace range_of_a_l100_100766

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l100_100766


namespace min_objective_value_l100_100888

theorem min_objective_value (x y : ℝ) 
  (h1 : x + y ≥ 2) 
  (h2 : x - y ≤ 2) 
  (h3 : y ≥ 1) : ∃ (z : ℝ), z = x + 3 * y ∧ z = 4 :=
by
  -- Provided proof omitted
  sorry

end min_objective_value_l100_100888


namespace symmetric_line_eq_l100_100768

theorem symmetric_line_eq (x y : ℝ) :
  (y = 2 * x + 3) → (y - 1 = x + 1) → (x - 2 * y = 0) :=
by
  intros h1 h2
  sorry

end symmetric_line_eq_l100_100768


namespace problem_inequality_l100_100199

theorem problem_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 1 < a₁) (h₂ : 1 < a₂) (h₃ : 1 < a₃) (h₄ : 1 < a₄) (h₅ : 1 < a₅) :
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) :=
sorry

end problem_inequality_l100_100199


namespace probability_of_four_digit_number_divisible_by_3_l100_100636

def digits : List ℕ := [0, 1, 2, 3, 4, 5]

def count_valid_four_digit_numbers : Int :=
  let all_digits := digits
  let total_four_digit_numbers := 180
  let valid_four_digit_numbers := 96
  total_four_digit_numbers

def probability_divisible_by_3 : ℚ :=
  (96 : ℚ) / (180 : ℚ)

theorem probability_of_four_digit_number_divisible_by_3 :
  probability_divisible_by_3 = 8 / 15 :=
by
  sorry

end probability_of_four_digit_number_divisible_by_3_l100_100636


namespace unique_integer_solution_range_l100_100592

theorem unique_integer_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x + 3 > 5) ∧ (x - a ≤ 0) → (x = 2)) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end unique_integer_solution_range_l100_100592


namespace greatest_b_l100_100599

theorem greatest_b (b : ℝ) : (-b^2 + 9 * b - 14 ≥ 0) → b ≤ 7 := sorry

end greatest_b_l100_100599


namespace triangle_side_length_l100_100838

theorem triangle_side_length (AB AC BC BX CX : ℕ)
  (h1 : AB = 86)
  (h2 : AC = 97)
  (h3 : BX + CX = BC)
  (h4 : AX = AB)
  (h5 : AX = 86)
  (h6 : AB * AB * CX + AC * AC * BX = BC * (BX * CX + AX * AX))
  : BC = 61 := 
sorry

end triangle_side_length_l100_100838


namespace power_equality_l100_100717

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l100_100717


namespace fraction_multiplication_l100_100025

theorem fraction_multiplication :
  (2 / (3 : ℚ)) * (4 / 7) * (5 / 9) * (11 / 13) = 440 / 2457 :=
by
  sorry

end fraction_multiplication_l100_100025


namespace number_of_pens_bought_l100_100041

theorem number_of_pens_bought 
  (P : ℝ) -- Marked price of one pen
  (N : ℝ) -- Number of pens bought
  (discount : ℝ := 0.01)
  (profit_percent : ℝ := 29.130434782608695)
  (Total_Cost := 46 * P)
  (Selling_Price_per_Pen := P * (1 - discount))
  (Total_Revenue := N * Selling_Price_per_Pen)
  (Profit := Total_Revenue - Total_Cost)
  (actual_profit_percent := (Profit / Total_Cost) * 100) :
  actual_profit_percent = profit_percent → N = 60 := 
by 
  intro h
  sorry

end number_of_pens_bought_l100_100041


namespace enough_cat_food_for_six_days_l100_100936

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l100_100936


namespace total_problems_l100_100409

theorem total_problems (math_pages reading_pages problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end total_problems_l100_100409


namespace solve_equation_l100_100062

theorem solve_equation (x : ℝ) : ((x-3)^2 + 4*x*(x-3) = 0) → (x = 3 ∨ x = 3/5) :=
by
  sorry

end solve_equation_l100_100062


namespace p_is_necessary_but_not_sufficient_for_q_l100_100090

variable (x : ℝ)

def p := x > 4
def q := 4 < x ∧ x < 10

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l100_100090


namespace true_propositions_among_converse_inverse_contrapositive_l100_100716

theorem true_propositions_among_converse_inverse_contrapositive
  (x : ℝ)
  (h1 : x^2 ≥ 1 → x ≥ 1) :
  (if x ≥ 1 then x^2 ≥ 1 else true) ∧ 
  (if x^2 < 1 then x < 1 else true) ∧ 
  (if x < 1 then x^2 < 1 else true) → 
  ∃ n, n = 2 :=
by sorry

end true_propositions_among_converse_inverse_contrapositive_l100_100716


namespace polynomial_two_distinct_negative_real_roots_l100_100890

theorem polynomial_two_distinct_negative_real_roots :
  ∀ (p : ℝ), 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
    (x1^4 + p*x1^3 + 3*x1^2 + p*x1 + 4 = 0) ∧ 
    (x2^4 + p*x2^3 + 3*x2^2 + p*x2 + 4 = 0)) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
by
  sorry

end polynomial_two_distinct_negative_real_roots_l100_100890


namespace coin_sum_even_odd_l100_100179

theorem coin_sum_even_odd (S : ℕ) (h : S > 1) : 
  (∃ even_count, (even_count : ℕ) ∈ [0, 2, S]) ∧ (∃ odd_count, ((odd_count : ℕ) - 1) ∈ [0, 2, S]) :=
  sorry

end coin_sum_even_odd_l100_100179


namespace min_value_of_expression_min_value_achieved_at_l100_100810

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 
  3 * Real.sqrt x + 4 / (x^2) ≥ 4 * 4^(1/5) :=
sorry

theorem min_value_achieved_at (x : ℝ) (hx : 0 < x) (h : x = 4^(2/5)) :
  3 * Real.sqrt x + 4 / (x^2) = 4 * 4^(1/5) :=
sorry

end min_value_of_expression_min_value_achieved_at_l100_100810


namespace right_triangles_with_leg_2012_l100_100853

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l100_100853


namespace remainder_when_3m_divided_by_5_l100_100589

theorem remainder_when_3m_divided_by_5 (m : ℤ) (hm : m % 5 = 2) : (3 * m) % 5 = 1 := 
sorry

end remainder_when_3m_divided_by_5_l100_100589


namespace rhombus_area_l100_100823

theorem rhombus_area : 
  ∃ (d1 d2 : ℝ), (∀ (x : ℝ), x^2 - 14 * x + 48 = 0 → x = d1 ∨ x = d2) ∧
  (∀ (A : ℝ), A = d1 * d2 / 2 → A = 24) :=
by 
sorry

end rhombus_area_l100_100823


namespace triangle_area_l100_100003

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

theorem triangle_area
  (A : ℝ) (b c : ℝ)
  (h1 : f A = 1)
  (h2 : b * c = 2) 
  (h3 : (b * cos A) * (c * cos A) = sqrt 2) : 
  (1 / 2 * b * c * sin A = sqrt 2 / 2) := 
sorry

end triangle_area_l100_100003


namespace subcommittee_count_l100_100382

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end subcommittee_count_l100_100382


namespace root_equation_m_l100_100994

theorem root_equation_m (m : ℝ) : 
  (∃ (x : ℝ), x = -1 ∧ m*x^2 + x - m^2 + 1 = 0) → m = 1 :=
by 
  sorry

end root_equation_m_l100_100994


namespace cosine_sum_of_angles_l100_100166

theorem cosine_sum_of_angles (α β : ℝ) 
  (hα : Complex.exp (Complex.I * α) = (4 / 5) + (3 / 5) * Complex.I)
  (hβ : Complex.exp (Complex.I * β) = (-5 / 13) + (12 / 13) * Complex.I) :
  Real.cos (α + β) = -7 / 13 :=
by
  sorry

end cosine_sum_of_angles_l100_100166


namespace annual_raise_l100_100553

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l100_100553


namespace smallest_number_conditions_l100_100498

theorem smallest_number_conditions :
  ∃ m : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], m % k = 2) ∧ (m % 8 = 0) ∧ ( ∀ n : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], n % k = 2) ∧ (n % 8 = 0) → m ≤ n ) :=
sorry

end smallest_number_conditions_l100_100498


namespace intersection_of_A_and_B_l100_100727

open Set

variable {α : Type}

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 2, 3, 5}
def B : Set ℤ := {x | -1 < x ∧ x < 3}

-- Define the proof problem as a theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_of_A_and_B_l100_100727


namespace carol_first_six_probability_l100_100330

theorem carol_first_six_probability :
  let p := 1 / 6
  let q := 5 / 6
  let prob_cycle := q^4
  (p * q^3) / (1 - prob_cycle) = 125 / 671 :=
by
  sorry

end carol_first_six_probability_l100_100330


namespace find_other_root_l100_100773

variable {m : ℝ} -- m is a real number
variable (x : ℝ)

theorem find_other_root (h : x^2 + m * x - 5 = 0) (hx1 : x = -1) : x = 5 :=
sorry

end find_other_root_l100_100773


namespace sixty_percent_of_N_l100_100151

noncomputable def N : ℝ :=
  let x := (45 : ℝ)
  let frac := (3/4 : ℝ) * (1/3) * (2/5) * (1/2)
  20 * x / frac

theorem sixty_percent_of_N : (0.60 : ℝ) * N = 540 := by
  sorry

end sixty_percent_of_N_l100_100151


namespace solve_arithmetic_sequence_l100_100559

theorem solve_arithmetic_sequence :
  ∃ x > 0, (x * x = (4 + 25) / 2) :=
by
  sorry

end solve_arithmetic_sequence_l100_100559


namespace sangwoo_gave_away_notebooks_l100_100934

variables (n : ℕ)

theorem sangwoo_gave_away_notebooks
  (h1 : 12 - n + 34 - 3 * n = 30) :
  n = 4 :=
by
  sorry

end sangwoo_gave_away_notebooks_l100_100934


namespace greatest_possible_x_l100_100809

theorem greatest_possible_x : ∃ (x : ℕ), (x^2 + 5 < 30) ∧ ∀ (y : ℕ), (y^2 + 5 < 30) → y ≤ x :=
by
  sorry

end greatest_possible_x_l100_100809


namespace max_value_of_x_l100_100841

theorem max_value_of_x : ∃ x : ℝ, 
  ( (4*x - 16) / (3*x - 4) )^2 + ( (4*x - 16) / (3*x - 4) ) = 18 
  ∧ x = (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
sorry

end max_value_of_x_l100_100841


namespace coefficient_x_squared_l100_100155

variable {a w c d : ℝ}

/-- The coefficient of x^2 in the expanded form of the equation (ax + w)(cx + d) = 6x^2 + x - 12 -/
theorem coefficient_x_squared (h1 : (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
                             (h2 : abs a + abs w + abs c + abs d = 12) :
  a * c = 6 :=
  sorry

end coefficient_x_squared_l100_100155


namespace greatest_temp_diff_on_tuesday_l100_100762

def highest_temp_mon : ℝ := 5
def lowest_temp_mon : ℝ := 2
def highest_temp_tue : ℝ := 4
def lowest_temp_tue : ℝ := -1
def highest_temp_wed : ℝ := 0
def lowest_temp_wed : ℝ := -4

def temp_diff (highest lowest : ℝ) : ℝ :=
  highest - lowest

theorem greatest_temp_diff_on_tuesday : temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_mon lowest_temp_mon 
  ∧ temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_wed lowest_temp_wed := 
by
  sorry

end greatest_temp_diff_on_tuesday_l100_100762


namespace bridge_length_l100_100410

theorem bridge_length (train_length : ℕ) (crossing_time : ℕ) (train_speed_kmh : ℕ) :
  train_length = 500 → crossing_time = 45 → train_speed_kmh = 64 → 
  ∃ (bridge_length : ℝ), bridge_length = 300.1 :=
by
  intros h1 h2 h3
  have speed_mps := (train_speed_kmh * 1000) / 3600
  have total_distance := speed_mps * crossing_time
  have bridge_length_calculated := total_distance - train_length
  use bridge_length_calculated
  sorry

end bridge_length_l100_100410


namespace square_of_999_l100_100346

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end square_of_999_l100_100346


namespace expected_difference_l100_100744

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_composite (n : ℕ) : Prop := n = 4 ∨ n = 6 ∨ n = 8

def roll_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def probability_eat_sweetened : ℚ := 4 / 7
def probability_eat_unsweetened : ℚ := 3 / 7
def days_in_leap_year : ℕ := 366

def expected_days_unsweetened : ℚ := probability_eat_unsweetened * days_in_leap_year
def expected_days_sweetened : ℚ := probability_eat_sweetened * days_in_leap_year

theorem expected_difference :
  expected_days_sweetened - expected_days_unsweetened = 52.28 := by
  sorry

end expected_difference_l100_100744


namespace cosine_equation_solution_count_l100_100209

open Real

noncomputable def number_of_solutions : ℕ := sorry

theorem cosine_equation_solution_count :
  number_of_solutions = 2 :=
by
  -- Let x be an angle in [0, 2π].
  sorry

end cosine_equation_solution_count_l100_100209


namespace yang_hui_rect_eq_l100_100767

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l100_100767


namespace relatively_prime_positive_integers_l100_100518

theorem relatively_prime_positive_integers (a b : ℕ) (h1 : a > b) (h2 : gcd a b = 1) (h3 : (a^3 - b^3) / (a - b)^3 = 91 / 7) : a - b = 1 := 
by 
  sorry

end relatively_prime_positive_integers_l100_100518


namespace sum_gcd_lcm_l100_100250

theorem sum_gcd_lcm (a b : ℕ) (ha : a = 45) (hb : b = 4095) :
    Nat.gcd a b + Nat.lcm a b = 4140 :=
by
  sorry

end sum_gcd_lcm_l100_100250


namespace S_6_equals_12_l100_100799

noncomputable def S (n : ℕ) : ℝ := sorry -- Definition for the sum of the first n terms

axiom geometric_sequence_with_positive_terms (n : ℕ) : S n > 0

axiom S_3 : S 3 = 3

axiom S_9 : S 9 = 39

theorem S_6_equals_12 : S 6 = 12 := by
  sorry

end S_6_equals_12_l100_100799


namespace value_of_c_l100_100433

theorem value_of_c (c : ℝ) : (∀ x : ℝ, x * (4 * x + 1) < c ↔ x > -5 / 2 ∧ x < 3) → c = 27 :=
by
  intros h
  sorry

end value_of_c_l100_100433


namespace jacket_initial_reduction_percent_l100_100102

theorem jacket_initial_reduction_percent (P : ℝ) (x : ℝ) (h : P * (1 - x / 100) * 0.70 * 1.5873 = P) : x = 10 :=
sorry

end jacket_initial_reduction_percent_l100_100102


namespace train_speed_l100_100277

-- Definition for the given conditions
def distance : ℕ := 240 -- distance in meters
def time_seconds : ℕ := 6 -- time in seconds
def conversion_factor : ℕ := 3600 -- seconds to hour conversion factor
def meters_in_km : ℕ := 1000 -- meters to kilometers conversion factor

-- The proof goal
theorem train_speed (d : ℕ) (t : ℕ) (cf : ℕ) (mk : ℕ) (h1 : d = distance) (h2 : t = time_seconds) (h3 : cf = conversion_factor) (h4 : mk = meters_in_km) :
  (d * cf / t) / mk = 144 :=
by sorry

end train_speed_l100_100277


namespace loss_percentage_is_11_percent_l100_100869

-- Definitions based on conditions
def costPrice : ℝ := 1500
def sellingPrice : ℝ := 1335

-- The statement to prove
theorem loss_percentage_is_11_percent :
  ((costPrice - sellingPrice) / costPrice) * 100 = 11 := by
  sorry

end loss_percentage_is_11_percent_l100_100869


namespace angles_proof_l100_100578

-- Definitions (directly from the conditions)
variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

def complementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 90
def supplementary (θ₃ θ₄ : ℝ) : Prop := θ₃ + θ₄ = 180

-- Theorem statement
theorem angles_proof (h1 : complementary θ₁ θ₂) (h2 : supplementary θ₃ θ₄) (h3 : θ₁ = θ₃) :
  θ₂ + 90 = θ₄ :=
by
  sorry

end angles_proof_l100_100578


namespace find_larger_number_l100_100763

theorem find_larger_number (L S : ℤ) (h₁ : L - S = 1000) (h₂ : L = 10 * S + 10) : L = 1110 :=
sorry

end find_larger_number_l100_100763


namespace bicentric_quad_lemma_l100_100687

-- Define the properties and radii of the bicentric quadrilateral
variables (KLMN : Type) (r ρ h : ℝ)

-- Assuming quadrilateral KLMN is bicentric with given radii
def is_bicentric (KLMN : Type) := true

-- State the theorem we wish to prove
theorem bicentric_quad_lemma (br : is_bicentric KLMN) : 
  (1 / (ρ + h) ^ 2) + (1 / (ρ - h) ^ 2) = (1 / r ^ 2) :=
sorry

end bicentric_quad_lemma_l100_100687


namespace example_problem_l100_100790

theorem example_problem
  (h1 : 0.25 < 1) 
  (h2 : 0.15 < 0.25) : 
  3.04 / 0.25 > 1 :=
by
  sorry

end example_problem_l100_100790


namespace tan_neg_3900_eq_sqrt3_l100_100348

theorem tan_neg_3900_eq_sqrt3 : Real.tan (-3900 * Real.pi / 180) = Real.sqrt 3 := by
  -- Definitions of trigonometric values at 60 degrees
  have h_cos : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h_sin : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Using periodicity of the tangent function
  sorry

end tan_neg_3900_eq_sqrt3_l100_100348


namespace sin_cos_alpha_frac_l100_100963

theorem sin_cos_alpha_frac (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 := 
by
  sorry

end sin_cos_alpha_frac_l100_100963


namespace find_m_l100_100764

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let d : ℝ := |2| / Real.sqrt (m^2 + 1)
  d = 1

theorem find_m (m : ℝ) : tangent_condition m ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end find_m_l100_100764


namespace candy_bar_profit_l100_100522

theorem candy_bar_profit
  (bars_bought : ℕ)
  (cost_per_six : ℝ)
  (bars_sold : ℕ)
  (price_per_three : ℝ)
  (tax_rate : ℝ)
  (h1 : bars_bought = 800)
  (h2 : cost_per_six = 3)
  (h3 : bars_sold = 800)
  (h4 : price_per_three = 2)
  (h5 : tax_rate = 0.1) :
  let cost_per_bar := cost_per_six / 6
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_three / 3
  let total_revenue := bars_sold * price_per_bar
  let tax := tax_rate * total_revenue
  let after_tax_revenue := total_revenue - tax
  let profit_after_tax := after_tax_revenue - total_cost
  profit_after_tax = 80.02 := by
    sorry

end candy_bar_profit_l100_100522


namespace convert_3652_from_base7_to_base10_l100_100715

def base7ToBase10(n : ℕ) := 
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d0 * (7^0) + d1 * (7^1) + d2 * (7^2) + d3 * (7^3)

theorem convert_3652_from_base7_to_base10 : base7ToBase10 3652 = 1360 :=
by
  sorry

end convert_3652_from_base7_to_base10_l100_100715


namespace part1_part2_l100_100973

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp (2 * x) - a * Real.exp x - x * Real.exp x

theorem part1 :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 := sorry

theorem part2 (h : a = 1) :
  ∃ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) ∧
    (Real.log 2 / (2 * Real.exp 1) + 1 / (4 * Real.exp (2 * 1)) ≤ f a x₀ ∧
    f a x₀ < 1 / 4) := sorry

end part1_part2_l100_100973


namespace solve_for_r_l100_100267

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end solve_for_r_l100_100267


namespace ashok_borrowed_l100_100235

theorem ashok_borrowed (P : ℝ) (h : 11400 = P * (6 / 100 * 2 + 9 / 100 * 3 + 14 / 100 * 4)) : P = 12000 :=
by
  sorry

end ashok_borrowed_l100_100235


namespace only_odd_digit_squared_n_l100_100882

def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_odd_digit d

theorem only_odd_digit_squared_n (n : ℕ) :
  0 < n ∧ has_only_odd_digits (n * n) ↔ n = 1 ∨ n = 3 :=
sorry

end only_odd_digit_squared_n_l100_100882


namespace vector_addition_l100_100640

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 5)

-- State the theorem that we want to prove
theorem vector_addition : a + 3 • b = (-1, 18) :=
  sorry

end vector_addition_l100_100640


namespace probability_red_or_black_probability_red_black_or_white_l100_100506

-- We define the probabilities of events A, B, and C
def P_A : ℚ := 5 / 12
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 6

-- Define the probability of event D for completeness
def P_D : ℚ := 1 / 12

-- 1. Statement for the probability of drawing a red or black ball (P(A ⋃ B))
theorem probability_red_or_black :
  (P_A + P_B = 3 / 4) :=
by
  sorry

-- 2. Statement for the probability of drawing a red, black, or white ball (P(A ⋃ B ⋃ C))
theorem probability_red_black_or_white :
  (P_A + P_B + P_C = 11 / 12) :=
by
  sorry

end probability_red_or_black_probability_red_black_or_white_l100_100506


namespace reciprocal_of_sum_is_correct_l100_100152

theorem reciprocal_of_sum_is_correct : (1 / (1 / 4 + 1 / 6)) = 12 / 5 := by
  sorry

end reciprocal_of_sum_is_correct_l100_100152


namespace arithmetic_evaluation_l100_100137

theorem arithmetic_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := 
by
  sorry

end arithmetic_evaluation_l100_100137


namespace lines_parallel_m_value_l100_100474

theorem lines_parallel_m_value (m : ℝ) : 
  (∀ (x y : ℝ), (x + 2 * m * y - 1 = 0) → ((m - 2) * x - m * y + 2 = 0)) → m = 3 / 2 :=
by
  -- placeholder for mathematical proof
  sorry

end lines_parallel_m_value_l100_100474


namespace find_m_l100_100258

theorem find_m
  (h1 : ∃ (m : ℝ), ∃ (focus_parabola : ℝ × ℝ), focus_parabola = (0, 1/2)
       ∧ ∃ (focus_ellipse : ℝ × ℝ), focus_ellipse = (0, Real.sqrt (m - 2))
       ∧ focus_parabola = focus_ellipse) :
  ∃ (m : ℝ), m = 9/4 :=
by
  sorry

end find_m_l100_100258


namespace three_collinear_points_l100_100456

theorem three_collinear_points (f : ℝ → Prop) (h_black_or_white : ∀ (x : ℝ), f x = true ∨ f x = false)
: ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b = (a + c) / 2) ∧ ((f a = f b) ∧ (f b = f c)) :=
sorry

end three_collinear_points_l100_100456


namespace ice_cream_bar_price_l100_100511

theorem ice_cream_bar_price 
  (num_bars num_sundaes : ℕ)
  (total_cost : ℝ)
  (sundae_price ice_cream_bar_price : ℝ)
  (h1 : num_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_cost = 250.00)
  (h4 : sundae_price = 1.40)
  (total_price_condition : num_bars * ice_cream_bar_price + num_sundaes * sundae_price = total_cost) :
  ice_cream_bar_price = 0.60 :=
sorry

end ice_cream_bar_price_l100_100511


namespace no_solution_inequality_l100_100662

theorem no_solution_inequality (a b x : ℝ) (h : |a - b| > 2) : ¬(|x - a| + |x - b| ≤ 2) :=
sorry

end no_solution_inequality_l100_100662


namespace part_1_part_2_l100_100665

def p (a x : ℝ) : Prop :=
a * x - 2 ≤ 0 ∧ a * x + 1 > 0

def q (x : ℝ) : Prop :=
x^2 - x - 2 < 0

theorem part_1 (a : ℝ) :
  (∃ x : ℝ, (1/2 < x ∧ x < 3) ∧ p a x) → 
  (-2 < a ∧ a < 4) :=
sorry

theorem part_2 (a : ℝ) :
  (∀ x, p a x → q x) ∧ 
  (∃ x, q x ∧ ¬p a x) → 
  (-1/2 ≤ a ∧ a ≤ 1) :=
sorry

end part_1_part_2_l100_100665


namespace diana_wins_l100_100603

noncomputable def probability_diana_wins : ℚ :=
  45 / 100

theorem diana_wins (d : ℕ) (a : ℕ) (hd : 1 ≤ d ∧ d ≤ 10) (ha : 1 ≤ a ∧ a ≤ 10) :
  probability_diana_wins = 9 / 20 :=
by
  sorry

end diana_wins_l100_100603


namespace height_of_first_podium_l100_100452

noncomputable def height_of_podium_2_cm := 53.0
noncomputable def height_of_podium_2_mm := 7.0
noncomputable def height_on_podium_2_cm := 190.0
noncomputable def height_on_podium_1_cm := 232.0
noncomputable def height_on_podium_1_mm := 5.0

def expected_height_of_podium_1_cm := 96.2

theorem height_of_first_podium :
  let height_podium_2 := height_of_podium_2_cm + height_of_podium_2_mm / 10.0
  let height_podium_1 := height_on_podium_1_cm + height_on_podium_1_mm / 10.0
  let hyeonjoo_height := height_on_podium_2_cm - height_podium_2
  height_podium_1 - hyeonjoo_height = expected_height_of_podium_1_cm :=
by sorry

end height_of_first_podium_l100_100452


namespace geometric_seq_neither_necess_nor_suff_l100_100158

theorem geometric_seq_neither_necess_nor_suff (a_1 q : ℝ) (h₁ : a_1 ≠ 0) (h₂ : q ≠ 0) :
  ¬ (∀ n : ℕ, (a_1 * q > 0 → a_1 * q ^ n < a_1 * q ^ (n + 1)) ∧ (∀ n : ℕ, (a_1 * q ^ n < a_1 * q ^ (n + 1)) → a_1 * q > 0)) :=
by
  sorry

end geometric_seq_neither_necess_nor_suff_l100_100158


namespace smallest_four_digit_2_mod_11_l100_100287

theorem smallest_four_digit_2_mod_11 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 11 = 2 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 11 = 2 → n ≤ m) := 
by 
  use 1003
  sorry

end smallest_four_digit_2_mod_11_l100_100287


namespace total_sum_lent_l100_100871

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ)
  (h1 : second_part = 1648)
  (h2 : (x * 3 / 100 * 8) = (second_part * 5 / 100 * 3))
  (h3 : total_sum = x + second_part) :
  total_sum = 2678 := 
  sorry

end total_sum_lent_l100_100871


namespace martin_bell_ringing_l100_100889

theorem martin_bell_ringing (B S : ℕ) (hB : B = 36) (hS : S = B / 3 + 4) : S + B = 52 :=
sorry

end martin_bell_ringing_l100_100889


namespace stones_max_value_50_l100_100332

-- Define the problem conditions in Lean
def value_of_stones (x y z : ℕ) : ℕ := 14 * x + 11 * y + 2 * z

def weight_of_stones (x y z : ℕ) : ℕ := 5 * x + 4 * y + z

def max_value_stones {x y z : ℕ} (h_w : weight_of_stones x y z ≤ 18) (h_x : x ≥ 0) (h_y : y ≥ 0) (h_z : z ≥ 0) : Prop :=
  value_of_stones x y z ≤ 50

theorem stones_max_value_50 : ∃ (x y z : ℕ), weight_of_stones x y z ≤ 18 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ value_of_stones x y z = 50 :=
by
  sorry

end stones_max_value_50_l100_100332


namespace maximum_n_for_dart_probability_l100_100532

theorem maximum_n_for_dart_probability (n : ℕ) (h : n ≥ 1) :
  (∃ r : ℝ, r = 1 ∧
  ∃ A_square A_circles : ℝ, A_square = n^2 ∧ A_circles = n * π * r^2 ∧
  (A_circles / A_square) ≥ 1 / 2) → n ≤ 6 := by
  sorry

end maximum_n_for_dart_probability_l100_100532


namespace playground_perimeter_km_l100_100750

def playground_length : ℕ := 360
def playground_width : ℕ := 480

def perimeter_in_meters (length width : ℕ) : ℕ := 2 * (length + width)

def perimeter_in_kilometers (perimeter_m : ℕ) : ℕ := perimeter_m / 1000

theorem playground_perimeter_km :
  perimeter_in_kilometers (perimeter_in_meters playground_length playground_width) = 168 :=
by
  sorry

end playground_perimeter_km_l100_100750


namespace ellipse_hyperbola_tangent_l100_100822

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y - 1)^2 = 4) →
  (m = 6 ∨ m = 12) := by
  sorry

end ellipse_hyperbola_tangent_l100_100822


namespace no_polynomials_exist_l100_100411

open Polynomial

theorem no_polynomials_exist
  (a b : Polynomial ℂ) (c d : Polynomial ℂ) :
  ¬ (∀ x y : ℂ, 1 + x * y + x^2 * y^2 = a.eval x * c.eval y + b.eval x * d.eval y) :=
sorry

end no_polynomials_exist_l100_100411


namespace second_butcher_packages_l100_100096

theorem second_butcher_packages (a b c: ℕ) (weight_per_package total_weight: ℕ)
    (first_butcher_packages: ℕ) (third_butcher_packages: ℕ)
    (cond1: a = 10) (cond2: b = 8) (cond3: weight_per_package = 4)
    (cond4: total_weight = 100):
    c = (total_weight - (first_butcher_packages * weight_per_package + third_butcher_packages * weight_per_package)) / weight_per_package →
    c = 7 := 
by 
  have first_butcher_packages := 10
  have third_butcher_packages := 8
  have weight_per_package := 4
  have total_weight := 100
  sorry

end second_butcher_packages_l100_100096


namespace inequality_proof_l100_100057

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (a b c x y z : R) (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  x^2 + y^2 + z^2 < a^2 + b^2 + c^2 ∧ x^3 + y^3 + z^3 < a^3 + b^3 + c^3 :=
by
  sorry

end inequality_proof_l100_100057


namespace real_root_range_of_a_l100_100226

theorem real_root_range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ (0 ≤ a ∧ a ≤ 1/4) :=
by
  sorry

end real_root_range_of_a_l100_100226


namespace golden_ratio_problem_l100_100667

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_problem (m : ℝ) (x : ℝ) :
  (1000 ≤ m) → (1000 ≤ x) → (x ≤ m) →
  ((m - 1000) / (x - 1000) = phi ∧ (x - 1000) / (m - x) = phi) →
  (m = 2000 ∨ m = 2618) :=
by
  sorry

end golden_ratio_problem_l100_100667


namespace money_sister_gave_l100_100571

theorem money_sister_gave (months_saved : ℕ) (savings_per_month : ℕ) (total_paid : ℕ) 
  (h1 : months_saved = 3) 
  (h2 : savings_per_month = 70) 
  (h3 : total_paid = 260) : 
  (total_paid - (months_saved * savings_per_month) = 50) :=
by {
  sorry
}

end money_sister_gave_l100_100571


namespace sniper_B_has_greater_chance_of_winning_l100_100298

def pA (n : ℕ) : ℝ :=
  if n = 1 then 0.4 else if n = 2 then 0.1 else if n = 3 then 0.5 else 0

def pB (n : ℕ) : ℝ :=
  if n = 1 then 0.1 else if n = 2 then 0.6 else if n = 3 then 0.3 else 0

noncomputable def expected_score (p : ℕ → ℝ) : ℝ :=
  (1 * p 1) + (2 * p 2) + (3 * p 3)

theorem sniper_B_has_greater_chance_of_winning :
  expected_score pB > expected_score pA :=
by
  sorry

end sniper_B_has_greater_chance_of_winning_l100_100298


namespace sequence_general_term_l100_100719

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n > 1, a n = 2 * a (n-1) + 1) : a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l100_100719


namespace rainy_days_last_week_l100_100732

theorem rainy_days_last_week (n : ℤ) (R NR : ℕ) (h1 : n * R + 3 * NR = 20)
  (h2 : 3 * NR = n * R + 10) (h3 : R + NR = 7) : R = 2 :=
sorry

end rainy_days_last_week_l100_100732


namespace edward_rides_eq_8_l100_100714

-- Define the initial conditions
def initial_tickets : ℕ := 79
def spent_tickets : ℕ := 23
def cost_per_ride : ℕ := 7

-- Define the remaining tickets after spending at the booth
def remaining_tickets : ℕ := initial_tickets - spent_tickets

-- Define the number of rides Edward could go on
def number_of_rides : ℕ := remaining_tickets / cost_per_ride

-- The goal is to prove that the number of rides is equal to 8.
theorem edward_rides_eq_8 : number_of_rides = 8 := by sorry

end edward_rides_eq_8_l100_100714


namespace ticket_sales_revenue_l100_100469

theorem ticket_sales_revenue (total_tickets advance_tickets same_day_tickets price_advance price_same_day: ℕ) 
    (h1: total_tickets = 60) 
    (h2: price_advance = 20) 
    (h3: price_same_day = 30) 
    (h4: advance_tickets = 20) 
    (h5: same_day_tickets = total_tickets - advance_tickets):
    advance_tickets * price_advance + same_day_tickets * price_same_day = 1600 := 
by
  sorry

end ticket_sales_revenue_l100_100469


namespace initial_price_of_sugar_per_kg_l100_100974

theorem initial_price_of_sugar_per_kg
  (initial_price : ℝ)
  (final_price : ℝ)
  (required_reduction : ℝ)
  (initial_price_eq : initial_price = 6)
  (final_price_eq : final_price = 7.5)
  (required_reduction_eq : required_reduction = 0.19999999999999996) :
  initial_price = 6 :=
by
  sorry

end initial_price_of_sugar_per_kg_l100_100974


namespace ria_number_is_2_l100_100023

theorem ria_number_is_2 
  (R S : ℕ) 
  (consecutive : R = S + 1 ∨ S = R + 1) 
  (R_positive : R > 0) 
  (S_positive : S > 0) 
  (R_not_1 : R ≠ 1) 
  (Sylvie_does_not_know : S ≠ 1) 
  (Ria_knows_after_Sylvie : ∃ (R_known : ℕ), R_known = R) :
  R = 2 :=
sorry

end ria_number_is_2_l100_100023


namespace problem1_problem2_l100_100482

namespace ProofProblems

-- Problem 1: Prove the inequality
theorem problem1 (x : ℝ) (h : x + |2 * x - 1| < 3) : -2 < x ∧ x < 4 / 3 := 
sorry

-- Problem 2: Prove the value of x + y + z 
theorem problem2 (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2 * y + 3 * z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := 
sorry

end ProofProblems

end problem1_problem2_l100_100482


namespace Carlos_has_highest_result_l100_100210

def Alice_final_result : ℕ := 30 + 3
def Ben_final_result : ℕ := 34 + 3
def Carlos_final_result : ℕ := 13 * 3

theorem Carlos_has_highest_result : (Carlos_final_result > Alice_final_result) ∧ (Carlos_final_result > Ben_final_result) := by
  sorry

end Carlos_has_highest_result_l100_100210


namespace actual_miles_traveled_l100_100046

def skipped_digits_odometer (digits : List ℕ) : Prop :=
  digits = [0, 1, 2, 3, 6, 7, 8, 9]

theorem actual_miles_traveled (odometer_reading : String) (actual_miles : ℕ) :
  skipped_digits_odometer [0, 1, 2, 3, 6, 7, 8, 9] →
  odometer_reading = "000306" →
  actual_miles = 134 :=
by
  intros
  sorry

end actual_miles_traveled_l100_100046


namespace value_of_P_dot_Q_l100_100939

def P : Set ℝ := {x | Real.log x / Real.log 2 < 1}
def Q : Set ℝ := {x | abs (x - 2) < 1}
def P_dot_Q (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∧ x ∉ Q}

theorem value_of_P_dot_Q : P_dot_Q P Q = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end value_of_P_dot_Q_l100_100939


namespace lower_seat_tickets_l100_100770

theorem lower_seat_tickets (L U : ℕ) (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end lower_seat_tickets_l100_100770


namespace intersection_A_B_l100_100775

def A (x : ℝ) : Prop := x > 3
def B (x : ℝ) : Prop := x ≤ 4

theorem intersection_A_B : {x | A x} ∩ {x | B x} = {x | 3 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_l100_100775


namespace onions_total_l100_100213

-- Define the number of onions grown by Sara, Sally, and Fred
def sara_onions : ℕ := 4
def sally_onions : ℕ := 5
def fred_onions : ℕ := 9

-- Define the total onions grown
def total_onions : ℕ := sara_onions + sally_onions + fred_onions

-- Theorem stating the total number of onions grown
theorem onions_total : total_onions = 18 := by
  sorry

end onions_total_l100_100213


namespace next_number_in_sequence_is_131_l100_100669

/-- Define the sequence increments between subsequent numbers -/
def sequencePattern : List ℕ := [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

-- Function to apply a sequence of increments starting from an initial value
def computeNext (initial : ℕ) (increments : List ℕ) : ℕ :=
  increments.foldl (λ acc inc => acc + inc) initial

-- Function to get the sequence's nth element 
def sequenceNthElement (n : ℕ) : ℕ :=
  (computeNext 12 (sequencePattern.take n))

-- Proof that the next number in the sequence is 131 
theorem next_number_in_sequence_is_131 :
  sequenceNthElement 10 = 131 :=
  by
  -- Proof omitted
  sorry

end next_number_in_sequence_is_131_l100_100669


namespace pool_depth_is_10_feet_l100_100550

-- Definitions based on conditions
def hoseRate := 60 -- cubic feet per minute
def poolWidth := 80 -- feet
def poolLength := 150 -- feet
def drainingTime := 2000 -- minutes

-- Proof goal: the depth of the pool is 10 feet
theorem pool_depth_is_10_feet :
  ∃ (depth : ℝ), depth = 10 ∧ (hoseRate * drainingTime) = (poolWidth * poolLength * depth) :=
by
  use 10
  sorry

end pool_depth_is_10_feet_l100_100550


namespace cleaner_steps_l100_100504

theorem cleaner_steps (a b c : ℕ) (h1 : a < 10 ∧ b < 10 ∧ c < 10) (h2 : 100 * a + 10 * b + c > 100 * c + 10 * b + a) (h3 : 100 * a + 10 * b + c + 100 * c + 10 * b + a = 746) :
  (100 * a + 10 * b + c) * 2 = 944 ∨ (100 * a + 10 * b + c) * 2 = 1142 :=
by
  sorry

end cleaner_steps_l100_100504


namespace arithmetic_sequence_sum_l100_100489

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Problem statement in Lean 4
theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : S 9 = a 4 + a 5 + a 6 + 66) :
  a 2 + a 8 = 22 := by
  sorry

end arithmetic_sequence_sum_l100_100489


namespace stubborn_robot_returns_to_start_l100_100878

inductive Direction
| East | North | West | South

inductive Command
| STEP | LEFT

structure Robot :=
  (position : ℤ × ℤ)
  (direction : Direction)

def turnLeft : Direction → Direction
| Direction.East  => Direction.North
| Direction.North => Direction.West
| Direction.West  => Direction.South
| Direction.South => Direction.East

def moveStep : Robot → Robot
| ⟨(x, y), Direction.East⟩  => ⟨(x + 1, y), Direction.East⟩
| ⟨(x, y), Direction.North⟩ => ⟨(x, y + 1), Direction.North⟩
| ⟨(x, y), Direction.West⟩  => ⟨(x - 1, y), Direction.West⟩
| ⟨(x, y), Direction.South⟩ => ⟨(x, y - 1), Direction.South⟩

def executeCommand : Command → Robot → Robot
| Command.STEP, robot => moveStep robot
| Command.LEFT, robot => ⟨robot.position, turnLeft robot.direction⟩

def invertCommand : Command → Command
| Command.STEP => Command.LEFT
| Command.LEFT => Command.STEP

def executeSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand cmd r) robot

def executeInvertedSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand (invertCommand cmd) r) robot

def initialRobot : Robot := ⟨(0, 0), Direction.East⟩

def exampleProgram : List Command :=
  [Command.LEFT, Command.LEFT, Command.LEFT, Command.LEFT, Command.STEP, Command.STEP,
   Command.LEFT, Command.LEFT]

theorem stubborn_robot_returns_to_start :
  let robot := executeSequence exampleProgram initialRobot
  executeInvertedSequence exampleProgram robot = initialRobot :=
by
  sorry

end stubborn_robot_returns_to_start_l100_100878


namespace opposite_directions_l100_100082

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem opposite_directions (a b : V) (h : a + 4 • b = 0) : a = -4 • b := sorry

end opposite_directions_l100_100082


namespace alice_prank_combinations_l100_100952

theorem alice_prank_combinations : 
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 60 :=
by
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  exact (show 1 * 3 * 5 * 4 * 1 = 60 from sorry)

end alice_prank_combinations_l100_100952


namespace ratio_bisector_circumradius_l100_100443

theorem ratio_bisector_circumradius (h_a h_b h_c : ℝ) (ha_val : h_a = 1/3) (hb_val : h_b = 1/4) (hc_val : h_c = 1/5) :
  ∃ (CD R : ℝ), CD / R = 24 * Real.sqrt 2 / 35 :=
by
  sorry

end ratio_bisector_circumradius_l100_100443


namespace find_r_l100_100789

theorem find_r (r s : ℝ)
  (h1 : ∀ α β : ℝ, (α + β = -r) ∧ (α * β = s) → 
         ∃ t : ℝ, (t^2 - (α^2 + β^2) * t + (α^2 * β^2) = 0) ∧ |α^2 - β^2| = 8)
  (h_sum : ∃ α β : ℝ, α + β = 10) :
  r = -10 := by
  sorry

end find_r_l100_100789


namespace michael_boxes_l100_100264

theorem michael_boxes (total_blocks boxes_per_box : ℕ) (h1: total_blocks = 16) (h2: boxes_per_box = 2) :
  total_blocks / boxes_per_box = 8 :=
by
  sorry

end michael_boxes_l100_100264


namespace simplify_sqrt8_minus_sqrt2_l100_100947

theorem simplify_sqrt8_minus_sqrt2 :
  (Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2) :=
sorry

end simplify_sqrt8_minus_sqrt2_l100_100947


namespace find_a_plus_b_l100_100627

noncomputable def lines_intersect (a b : ℝ) : Prop := 
  (∃ x y : ℝ, (x = 1/3 * y + a) ∧ (y = 1/3 * x + b) ∧ (x = 3) ∧ (y = 6))

theorem find_a_plus_b (a b : ℝ) (h : lines_intersect a b) : a + b = 6 :=
sorry

end find_a_plus_b_l100_100627


namespace coleFenceCostCorrect_l100_100273

noncomputable def coleFenceCost : ℕ := 455

def woodenFenceCost : ℕ := 15 * 6
def woodenFenceNeighborContribution : ℕ := woodenFenceCost / 3
def coleWoodenFenceCost : ℕ := woodenFenceCost - woodenFenceNeighborContribution

def metalFenceCost : ℕ := 15 * 8
def coleMetalFenceCost : ℕ := metalFenceCost

def hedgeCost : ℕ := 30 * 10
def hedgeNeighborContribution : ℕ := hedgeCost / 2
def coleHedgeCost : ℕ := hedgeCost - hedgeNeighborContribution

def installationFee : ℕ := 75
def soilPreparationFee : ℕ := 50

def totalCost : ℕ := coleWoodenFenceCost + coleMetalFenceCost + coleHedgeCost + installationFee + soilPreparationFee

theorem coleFenceCostCorrect : totalCost = coleFenceCost := by
  -- Skipping the proof steps with sorry
  sorry

end coleFenceCostCorrect_l100_100273


namespace sequence_a_n_definition_l100_100917

theorem sequence_a_n_definition (a : ℕ+ → ℝ) 
  (h₀ : ∀ n : ℕ+, a (n + 1) = 2016 * a n / (2014 * a n + 2016))
  (h₁ : a 1 = 1) : 
  a 2017 = 1008 / (1007 * 2017 + 1) :=
sorry

end sequence_a_n_definition_l100_100917


namespace free_throw_percentage_l100_100846

theorem free_throw_percentage (p : ℚ) :
  (1 - p)^2 + 2 * p * (1 - p) = 16 / 25 → p = 3 / 5 :=
by
  sorry

end free_throw_percentage_l100_100846


namespace no_solution_for_k_l100_100239

theorem no_solution_for_k 
  (a1 a2 a3 a4 : ℝ) 
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) 
  (h_pos3 : a2 < a3) (h_pos4 : a3 < a4) 
  (x1 x2 x3 x4 k : ℝ) 
  (h1 : x1 + x2 + x3 + x4 = 1) 
  (h2 : a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = k) 
  (h3 : a1^2 * x1 + a2^2 * x2 + a3^2 * x3 + a4^2 * x4 = k^2) 
  (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hx4 : 0 ≤ x4) :
  false := 
sorry

end no_solution_for_k_l100_100239


namespace total_heartbeats_correct_l100_100613

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l100_100613


namespace work_done_by_A_alone_l100_100266

theorem work_done_by_A_alone (Wb : ℝ) (Wa : ℝ) (D : ℝ) :
  Wa = 3 * Wb →
  (Wb + Wa) * 18 = D →
  D = 72 → 
  (D / Wa) = 24 := 
by
  intros h1 h2 h3
  sorry

end work_done_by_A_alone_l100_100266


namespace distance_covered_downstream_l100_100956

-- Conditions
def boat_speed_still_water : ℝ := 16
def stream_rate : ℝ := 5
def time_downstream : ℝ := 6

-- Effective speed downstream
def effective_speed_downstream := boat_speed_still_water + stream_rate

-- Distance covered downstream
def distance_downstream := effective_speed_downstream * time_downstream

-- Theorem to prove
theorem distance_covered_downstream :
  (distance_downstream = 126) :=
by
  sorry

end distance_covered_downstream_l100_100956


namespace sandy_gain_percent_is_10_l100_100244

def total_cost (purchase_price repair_costs : ℕ) := purchase_price + repair_costs

def gain (selling_price total_cost : ℕ) := selling_price - total_cost

def gain_percent (gain total_cost : ℕ) := (gain / total_cost : ℚ) * 100

theorem sandy_gain_percent_is_10 
  (purchase_price : ℕ := 900)
  (repair_costs : ℕ := 300)
  (selling_price : ℕ := 1320) :
  gain_percent (gain selling_price (total_cost purchase_price repair_costs)) 
               (total_cost purchase_price repair_costs) = 10 := 
by
  simp [total_cost, gain, gain_percent]
  sorry

end sandy_gain_percent_is_10_l100_100244


namespace giraffe_ratio_l100_100260

theorem giraffe_ratio (g ng : ℕ) (h1 : g = 300) (h2 : g = ng + 290) : g / ng = 30 :=
by
  sorry

end giraffe_ratio_l100_100260


namespace number_of_y_axis_returns_l100_100263

-- Definitions based on conditions
noncomputable def unit_length : ℝ := 0.5
noncomputable def diagonal_length : ℝ := Real.sqrt 2 * unit_length
noncomputable def pen_length_cm : ℝ := 8000 * 100 -- converting meters to cm
noncomputable def circle_length (n : ℕ) : ℝ := ((3 + Real.sqrt 2) * n ^ 2 + 2 * n) * unit_length

-- The main theorem
theorem number_of_y_axis_returns : ∃ n : ℕ, circle_length n ≤ pen_length_cm ∧ circle_length (n+1) > pen_length_cm :=
sorry

end number_of_y_axis_returns_l100_100263


namespace shifted_roots_polynomial_l100_100751

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ :=
  x^3 - 5 * x + 7

-- Define the shifted polynomial
def shifted_polynomial (x : ℝ) : ℝ :=
  x^3 + 9 * x^2 + 22 * x + 19

-- Define the roots condition
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop :=
  p r = 0

-- State the theorem
theorem shifted_roots_polynomial :
  ∀ a b c : ℝ,
    is_root original_polynomial a →
    is_root original_polynomial b →
    is_root original_polynomial c →
    is_root shifted_polynomial (a - 3) ∧
    is_root shifted_polynomial (b - 3) ∧
    is_root shifted_polynomial (c - 3) :=
by
  intros a b c ha hb hc
  sorry

end shifted_roots_polynomial_l100_100751


namespace exists_root_in_interval_l100_100357

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem exists_root_in_interval : ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 := 
by
  -- Assuming f(2) < 0 and f(3) > 0
  have h1 : f 2 < 0 := sorry
  have h2 : f 3 > 0 := sorry
  -- From the intermediate value theorem, there exists a c in (2, 3) such that f(c) = 0
  sorry

end exists_root_in_interval_l100_100357


namespace min_value_expression_l100_100465

theorem min_value_expression :
  ∃ x : ℝ, (x+2) * (x+3) * (x+5) * (x+6) + 2024 = 2021.75 :=
sorry

end min_value_expression_l100_100465


namespace eval_expression_l100_100639

theorem eval_expression : 
  ( ( (476 * 100 + 424 * 100) * 2^3 - 4 * (476 * 100 * 424 * 100) ) * (376 - 150) ) / 250 = -7297340160 :=
by
  sorry

end eval_expression_l100_100639


namespace inequality_solution_set_l100_100205

theorem inequality_solution_set :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (x = 0)} = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  sorry

end inequality_solution_set_l100_100205


namespace smallest_value_of_2a_plus_1_l100_100115

theorem smallest_value_of_2a_plus_1 (a : ℝ) 
  (h : 6 * a^2 + 5 * a + 4 = 3) : 
  ∃ b : ℝ, b = 2 * a + 1 ∧ b = 0 := 
sorry

end smallest_value_of_2a_plus_1_l100_100115


namespace max_value_of_expression_achieve_max_value_l100_100909

theorem max_value_of_expression : 
  ∀ x : ℝ, -3 * x ^ 2 + 18 * x - 4 ≤ 77 :=
by
  -- Placeholder proof
  sorry

theorem achieve_max_value : 
  ∃ x : ℝ, -3 * x ^ 2 + 18 * x - 4 = 77 :=
by
  -- Placeholder proof
  sorry

end max_value_of_expression_achieve_max_value_l100_100909


namespace flower_pots_problem_l100_100834

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ := x + 5 * 0.15

theorem flower_pots_problem
  (x : ℝ)       -- cost of the smallest pot
  (total_cost : ℝ) -- total cost of all pots
  (h_total_cost : total_cost = 8.25)
  (h_price_relation : total_cost = 6 * x + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15)) :
  cost_of_largest_pot x = 1.75 :=
by
  sorry

end flower_pots_problem_l100_100834


namespace minimum_ellipse_area_l100_100435

theorem minimum_ellipse_area (a b : ℝ) (h₁ : 4 * (a : ℝ) ^ 2 * b ^ 2 = a ^ 2 + b ^ 4)
  (h₂ : (∀ x y : ℝ, ((x - 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1)) 
       ∧ (∀ x y : ℝ, ((x + 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1))) : 
  ∃ k : ℝ, (k = 16) ∧ (π * (4 * a * b) = k * π) :=
by sorry

end minimum_ellipse_area_l100_100435


namespace add_base8_l100_100502

/-- Define the numbers in base 8 --/
def base8_add (a b : Nat) : Nat := 
  sorry

theorem add_base8 : base8_add 0o12 0o157 = 0o171 := 
  sorry

end add_base8_l100_100502


namespace no_solution_for_inequalities_l100_100860

theorem no_solution_for_inequalities :
  ¬ ∃ x : ℝ, 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 8 * x - 5 := by 
  sorry

end no_solution_for_inequalities_l100_100860


namespace number_of_bottle_caps_put_inside_l100_100453

-- Definitions according to the conditions
def initial_bottle_caps : ℕ := 7
def final_bottle_caps : ℕ := 14
def additional_bottle_caps (initial final : ℕ) := final - initial

-- The main theorem to prove
theorem number_of_bottle_caps_put_inside : additional_bottle_caps initial_bottle_caps final_bottle_caps = 7 :=
by
  sorry

end number_of_bottle_caps_put_inside_l100_100453


namespace value_of_a_l100_100728

noncomputable def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ :=
  a * b^3 + c

theorem value_of_a :
  F a 2 3 = F a 3 4 → a = -1 / 19 :=
by
  sorry

end value_of_a_l100_100728


namespace total_money_collected_l100_100371

def number_of_people := 610
def price_adult := 2
def price_child := 1
def number_of_adults := 350

theorem total_money_collected :
  (number_of_people - number_of_adults) * price_child + number_of_adults * price_adult = 960 := by
  sorry

end total_money_collected_l100_100371


namespace distance_to_workplace_l100_100126

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace_l100_100126


namespace ratio_2_10_as_percent_l100_100184

-- Define the problem conditions as given
def ratio_2_10 := 2 / 10

-- Express the question which is to show the percentage equivalent of the ratio 2:10
theorem ratio_2_10_as_percent : (ratio_2_10 * 100) = 20 :=
by
  -- Starting statement
  sorry -- Proof is not required here

end ratio_2_10_as_percent_l100_100184


namespace parabola_directrix_p_l100_100321

/-- Given a parabola with equation y^2 = 2px and directrix x = -2, prove that p = 4 -/
theorem parabola_directrix_p (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x) ∧ (∀ x : ℝ, x = -2 → True) → p = 4 :=
by
  sorry

end parabola_directrix_p_l100_100321


namespace bounds_on_xyz_l100_100758

theorem bounds_on_xyz (a x y z : ℝ) (h1 : x + y + z = a)
                      (h2 : x^2 + y^2 + z^2 = (a^2) / 2)
                      (h3 : a > 0) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) :
                      (0 < x ∧ x ≤ (2 / 3) * a) ∧ 
                      (0 < y ∧ y ≤ (2 / 3) * a) ∧ 
                      (0 < z ∧ z ≤ (2 / 3) * a) :=
sorry

end bounds_on_xyz_l100_100758


namespace find_F_l100_100370

theorem find_F (C F : ℝ) (h1 : C = (4 / 7) * (F - 40)) (h2 : C = 35) : F = 101.25 :=
  sorry

end find_F_l100_100370


namespace rainfall_on_first_day_l100_100064

theorem rainfall_on_first_day (R1 R2 R3 : ℕ) 
  (hR2 : R2 = 34)
  (hR3 : R3 = R2 - 12)
  (hTotal : R1 + R2 + R3 = 82) : 
  R1 = 26 := by
  sorry

end rainfall_on_first_day_l100_100064


namespace min_val_m_l100_100138

theorem min_val_m (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h : 24 * m = n ^ 4) : m = 54 :=
sorry

end min_val_m_l100_100138


namespace Rachel_total_books_l100_100937

theorem Rachel_total_books :
  (8 * 15) + (4 * 15) + (3 * 15) + (5 * 15) = 300 :=
by {
  sorry
}

end Rachel_total_books_l100_100937


namespace maximize_profit_l100_100942

-- Define the variables
variables (x y a b : ℝ)
variables (P : ℝ)

-- Define the conditions and the proof goal
theorem maximize_profit
  (h1 : x + 3 * y = 240)
  (h2 : 2 * x + y = 130)
  (h3 : a + b = 100)
  (h4 : a ≥ 4 * b)
  (ha : a = 80)
  (hb : b = 20) :
  x = 30 ∧ y = 70 ∧ P = (40 * a + 90 * b) - (30 * a + 70 * b) := 
by
  -- We assume the solution steps are solved correctly as provided
  sorry

end maximize_profit_l100_100942


namespace smallest_whole_number_larger_than_perimeter_l100_100280

theorem smallest_whole_number_larger_than_perimeter {s : ℝ} (h1 : 16 < s) (h2 : s < 30) :
  61 > 7 + 23 + s :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l100_100280


namespace factorization_of_m_squared_minus_4_l100_100543

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l100_100543


namespace range_of_a_l100_100040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 := by
  sorry

end range_of_a_l100_100040


namespace correct_exponentiation_l100_100316

theorem correct_exponentiation (a : ℝ) : a^5 / a = a^4 := 
  sorry

end correct_exponentiation_l100_100316


namespace deepak_and_wife_meet_time_l100_100579

noncomputable def deepak_speed_kmph : ℝ := 20
noncomputable def wife_speed_kmph : ℝ := 12
noncomputable def track_circumference_m : ℝ := 1000

noncomputable def speed_to_m_per_min (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 60

noncomputable def deepak_speed_m_per_min : ℝ := speed_to_m_per_min deepak_speed_kmph
noncomputable def wife_speed_m_per_min : ℝ := speed_to_m_per_min wife_speed_kmph

noncomputable def combined_speed_m_per_min : ℝ :=
  deepak_speed_m_per_min + wife_speed_m_per_min

noncomputable def meeting_time_minutes : ℝ :=
  track_circumference_m / combined_speed_m_per_min

theorem deepak_and_wife_meet_time :
  abs (meeting_time_minutes - 1.875) < 0.01 :=
by
  sorry

end deepak_and_wife_meet_time_l100_100579


namespace range_of_a_l100_100970

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_l100_100970


namespace product_evaluation_l100_100135

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 :=
by 
sorry

end product_evaluation_l100_100135


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l100_100987

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l100_100987


namespace solve_system_l100_100123

theorem solve_system 
  (x y z : ℝ)
  (h1 : x + 2 * y = 10)
  (h2 : y = 3)
  (h3 : x - 3 * y + z = 7) :
  x = 4 ∧ y = 3 ∧ z = 12 :=
by
  sorry

end solve_system_l100_100123


namespace yard_length_l100_100111

theorem yard_length (n : ℕ) (d : ℕ) (k : ℕ) (h : k = n - 1) (hd : d = 5) (hn : n = 51) : (k * d) = 250 := 
by
  sorry

end yard_length_l100_100111


namespace part1_part2_l100_100279

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x - 3)
noncomputable def M := 3 / 2

theorem part1 (x : ℝ) (m : ℝ) : (∀ x, f x ≥ abs (m + 1)) → m ≤ M := sorry

theorem part2 (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b + c = M →  (b^2 / a + c^2 / b + a^2 / c) ≥ M := sorry

end part1_part2_l100_100279


namespace number_of_cars_l100_100521

variable (C B : ℕ)

-- Define the conditions
def number_of_bikes : Prop := B = 2
def total_number_of_wheels : Prop := 4 * C + 2 * B = 44

-- State the theorem
theorem number_of_cars (hB : number_of_bikes B) (hW : total_number_of_wheels C B) : C = 10 := 
by 
  sorry

end number_of_cars_l100_100521


namespace blanch_slices_eaten_for_dinner_l100_100170

theorem blanch_slices_eaten_for_dinner :
  ∀ (total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner : ℕ),
  total_slices = 15 →
  eaten_breakfast = 4 →
  eaten_lunch = 2 →
  eaten_snack = 2 →
  slices_left = 2 →
  eaten_dinner = total_slices - (eaten_breakfast + eaten_lunch + eaten_snack) - slices_left →
  eaten_dinner = 5 := by
  intros total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner
  intros h_total_slices h_eaten_breakfast h_eaten_lunch h_eaten_snack h_slices_left h_eaten_dinner
  rw [h_total_slices, h_eaten_breakfast, h_eaten_lunch, h_eaten_snack, h_slices_left] at h_eaten_dinner
  exact h_eaten_dinner

end blanch_slices_eaten_for_dinner_l100_100170


namespace translation_coordinates_l100_100083

-- Define starting point
def initial_point : ℤ × ℤ := (-2, 3)

-- Define the point moved up by 2 units
def move_up (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst, p.snd + d)

-- Define the point moved right by 2 units
def move_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst + d, p.snd)

-- Expected results after movements
def point_up : ℤ × ℤ := (-2, 5)
def point_right : ℤ × ℤ := (0, 3)

-- Proof statement
theorem translation_coordinates :
  move_up initial_point 2 = point_up ∧
  move_right initial_point 2 = point_right :=
by
  sorry

end translation_coordinates_l100_100083


namespace daughters_and_granddaughters_without_daughters_l100_100414

-- Given conditions
def melissa_daughters : ℕ := 10
def half_daughters_with_children : ℕ := melissa_daughters / 2
def grandchildren_per_daughter : ℕ := 4
def total_descendants : ℕ := 50

-- Calculations based on given conditions
def number_of_granddaughters : ℕ := total_descendants - melissa_daughters
def daughters_with_no_children : ℕ := melissa_daughters - half_daughters_with_children
def granddaughters_with_no_children : ℕ := number_of_granddaughters

-- The final result we need to prove
theorem daughters_and_granddaughters_without_daughters : 
  daughters_with_no_children + granddaughters_with_no_children = 45 := by
  sorry

end daughters_and_granddaughters_without_daughters_l100_100414


namespace simplify_and_evaluate_expression_l100_100895

variable (x y : ℚ)

theorem simplify_and_evaluate_expression (hx : x = 1) (hy : y = 1 / 2) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x - y) ^ 2 = 31 / 4 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l100_100895


namespace find_function_l100_100927

theorem find_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f (y)^3 + f (z)^3 = 3 * x * y * z) → 
  f = id :=
by sorry

end find_function_l100_100927


namespace total_fruits_l100_100002

-- Define the given conditions
variable (a o : ℕ)
variable (ratio : a = 2 * o)
variable (half_apples_to_ann : a / 2 - 3 = 4)
variable (apples_to_cassie : a - a / 2 - 3 = 0)
variable (oranges_kept : 5 = o - 3)

theorem total_fruits (a o : ℕ) (ratio : a = 2 * o) 
  (half_apples_to_ann : a / 2 - 3 = 4) 
  (apples_to_cassie : a - a / 2 - 3 = 0) 
  (oranges_kept : 5 = o - 3) : a + o = 21 := 
sorry

end total_fruits_l100_100002


namespace range_of_a_l100_100496

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (x^2 - 2 * a * x + 2) ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l100_100496


namespace kira_night_songs_l100_100361

-- Definitions for the conditions
def morning_songs : ℕ := 10
def later_songs : ℕ := 15
def song_size_mb : ℕ := 5
def total_new_songs_memory_mb : ℕ := 140

-- Assert the number of songs Kira downloaded at night
theorem kira_night_songs : (total_new_songs_memory_mb - (morning_songs * song_size_mb + later_songs * song_size_mb)) / song_size_mb = 3 :=
by
  sorry

end kira_night_songs_l100_100361


namespace total_money_taken_in_l100_100214

-- Define the conditions as constants
def total_tickets : ℕ := 800
def advanced_ticket_price : ℝ := 14.5
def door_ticket_price : ℝ := 22.0
def door_tickets_sold : ℕ := 672
def advanced_tickets_sold : ℕ := total_tickets - door_tickets_sold
def total_revenue_advanced : ℝ := advanced_tickets_sold * advanced_ticket_price
def total_revenue_door : ℝ := door_tickets_sold * door_ticket_price
def total_revenue : ℝ := total_revenue_advanced + total_revenue_door

-- State the mathematical proof problem
theorem total_money_taken_in : total_revenue = 16640.00 := by
  sorry

end total_money_taken_in_l100_100214


namespace exists_function_passing_through_point_l100_100587

-- Define the function that satisfies f(2) = 0
theorem exists_function_passing_through_point : ∃ f : ℝ → ℝ, f 2 = 0 := 
sorry

end exists_function_passing_through_point_l100_100587


namespace value_of_y_l100_100192

theorem value_of_y (x y : ℝ) (h1 : x + y = 5) (h2 : x = 3) : y = 2 :=
by
  sorry

end value_of_y_l100_100192


namespace total_earnings_correct_l100_100318

section
  -- Define the conditions
  def wage : ℕ := 8
  def hours_Monday : ℕ := 8
  def hours_Tuesday : ℕ := 2

  -- Define the calculation for the total earnings
  def earnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

  -- State the total earnings
  def total_earnings : ℕ := earnings hours_Monday wage + earnings hours_Tuesday wage

  -- Theorem: Prove that Will's total earnings in those two days is $80
  theorem total_earnings_correct : total_earnings = 80 := by
    sorry
end

end total_earnings_correct_l100_100318


namespace power_function_value_l100_100985

theorem power_function_value {α : ℝ} (h : 3^α = Real.sqrt 3) : (9 : ℝ)^α = 3 :=
by sorry

end power_function_value_l100_100985


namespace car_owners_without_motorcycles_l100_100091

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ) (total_own_vehicle : ℕ)
  (h1 : total_adults = 400) (h2 : car_owners = 350) (h3 : motorcycle_owners = 60) (h4 : bicycle_owners = 30)
  (h5 : total_own_vehicle = total_adults)
  : (car_owners - 10 = 340) :=
by
  sorry

end car_owners_without_motorcycles_l100_100091


namespace simplify_sqrt_sum_l100_100211

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l100_100211


namespace inequality_proof_l100_100313

theorem inequality_proof
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 
  64 * a * b * c * d * abs ((a - b) * (b - c) * (c - d) * (d - a)) := 
by
  sorry

end inequality_proof_l100_100313


namespace sun_city_population_l100_100862

theorem sun_city_population (W R S : ℕ) (h1 : W = 2000)
    (h2 : R = 3 * W - 500) (h3 : S = 2 * R + 1000) : S = 12000 :=
by
    -- Use the provided conditions (h1, h2, h3) to state the theorem
    sorry

end sun_city_population_l100_100862


namespace problem_1_problem_2_l100_100113

-- Problem 1 Lean statement
theorem problem_1 :
  (1 - 1^4 - (1/2) * (3 - (-3)^2)) = 2 :=
by sorry

-- Problem 2 Lean statement
theorem problem_2 :
  ((3/8 - 1/6 - 3/4) * 24) = -13 :=
by sorry

end problem_1_problem_2_l100_100113


namespace problem_statement_l100_100056

theorem problem_statement (x : ℝ) :
  (x - 2)^4 + 5 * (x - 2)^3 + 10 * (x - 2)^2 + 10 * (x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end problem_statement_l100_100056


namespace cos_2alpha_l100_100844

theorem cos_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 + α) = (1 : ℝ) / 3) : 
  Real.cos (2 * α) = (7 : ℝ) / 9 := 
by
  sorry

end cos_2alpha_l100_100844


namespace ax_product_zero_l100_100696

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l100_100696


namespace octal_subtraction_l100_100993

theorem octal_subtraction : (53 - 27 : ℕ) = 24 :=
by sorry

end octal_subtraction_l100_100993


namespace man_walking_time_l100_100224

theorem man_walking_time
  (T : ℕ) -- Let T be the time (in minutes) the man usually arrives at the station.
  (usual_arrival_home : ℕ) -- The time (in minutes) they usually arrive home, which is T + 30.
  (early_arrival : ℕ) (walking_start_time : ℕ) (early_home_arrival : ℕ)
  (usual_arrival_home_eq : usual_arrival_home = T + 30)
  (early_arrival_eq : early_arrival = T - 60)
  (walking_start_time_eq : walking_start_time = early_arrival)
  (early_home_arrival_eq : early_home_arrival = T)
  (time_saved : ℕ) (half_time_walk : ℕ)
  (time_saved_eq : time_saved = 30)
  (half_time_walk_eq : half_time_walk = time_saved / 2) :
  walking_start_time = half_time_walk := by
  sorry

end man_walking_time_l100_100224


namespace simplify_expression_l100_100490

-- Define the algebraic expression
def algebraic_expr (x : ℚ) : ℚ := (3 / (x - 1) - x - 1) * (x - 1) / (x^2 - 4 * x + 4)

theorem simplify_expression : algebraic_expr 0 = 1 :=
by
  -- The proof is skipped using sorry
  sorry

end simplify_expression_l100_100490


namespace rebecca_eggs_l100_100777

theorem rebecca_eggs (groups : ℕ) (eggs_per_group : ℕ) (total_eggs : ℕ) 
  (h1 : groups = 3) (h2 : eggs_per_group = 3) : total_eggs = 9 :=
by
  sorry

end rebecca_eggs_l100_100777


namespace diagonal_of_rectangular_prism_l100_100602

theorem diagonal_of_rectangular_prism (x y z : ℝ) (d : ℝ)
  (h_surface_area : 2 * x * y + 2 * x * z + 2 * y * z = 22)
  (h_edge_length : x + y + z = 6) :
  d = Real.sqrt 14 :=
by
  sorry

end diagonal_of_rectangular_prism_l100_100602


namespace partial_fraction_decomposition_l100_100783

noncomputable def A := 29 / 15
noncomputable def B := 13 / 12
noncomputable def C := 37 / 15

theorem partial_fraction_decomposition :
  let ABC := A * B * C;
  ABC = 13949 / 2700 :=
by
  sorry

end partial_fraction_decomposition_l100_100783


namespace compute_binomial_sum_l100_100104

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem compute_binomial_sum :
  binomial 12 11 + binomial 12 1 = 24 :=
by
  sorry

end compute_binomial_sum_l100_100104


namespace solve_for_q_l100_100726

theorem solve_for_q (k l q : ℝ) 
  (h1 : 3 / 4 = k / 48)
  (h2 : 3 / 4 = (k + l) / 56)
  (h3 : 3 / 4 = (q - l) / 160) :
  q = 126 :=
  sorry

end solve_for_q_l100_100726


namespace interval_of_monotonic_decrease_minimum_value_in_interval_l100_100289

noncomputable def f (x a : ℝ) : ℝ := 1 / x + a * Real.log x

-- Define the derivative of f
noncomputable def f_prime (x a : ℝ) : ℝ := (a * x - 1) / x^2

-- Prove that the interval of monotonic decrease is as specified
theorem interval_of_monotonic_decrease (a : ℝ) :
  if a ≤ 0 then ∀ x ∈ Set.Ioi (0 : ℝ), f_prime x a < 0
  else ∀ x ∈ Set.Ioo 0 (1/a), f_prime x a < 0 := sorry

-- Prove that, given x in [1/2, 1], the minimum value of f(x) is 0 when a = 2 / log 2
theorem minimum_value_in_interval :
  ∃ a : ℝ, (a = 2 / Real.log 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x a ≥ 0 ∧ (∃ y ∈ Set.Icc (1/2 : ℝ) 1, f y a = 0) := sorry

end interval_of_monotonic_decrease_minimum_value_in_interval_l100_100289


namespace quadratic_rewriting_l100_100508

theorem quadratic_rewriting (d e : ℤ) (f : ℤ) : 
  (16 * x^2 - 40 * x - 24) = (d * x + e)^2 + f → 
  d^2 = 16 → 
  2 * d * e = -40 → 
  d * e = -20 := 
by
  intros h1 h2 h3
  sorry

end quadratic_rewriting_l100_100508


namespace least_possible_area_of_square_l100_100562

theorem least_possible_area_of_square (s : ℝ) (h₁ : 4.5 ≤ s) (h₂ : s < 5.5) : 
  s * s ≥ 20.25 :=
sorry

end least_possible_area_of_square_l100_100562


namespace amount_subtracted_l100_100471

theorem amount_subtracted (N A : ℝ) (h1 : N = 100) (h2 : 0.80 * N - A = 60) : A = 20 :=
by 
  sorry

end amount_subtracted_l100_100471


namespace range_of_a_l100_100938

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, a*x^2 - 2*a*x + 3 ≤ 0) ↔ (0 ≤ a ∧ a < 3) := 
sorry

end range_of_a_l100_100938


namespace find_lost_bowls_l100_100658

def bowls_problem (L : ℕ) : Prop :=
  let total_bowls := 638
  let broken_bowls := 15
  let payment := 1825
  let fee := 100
  let safe_bowl_payment := 3
  let lost_broken_bowl_cost := 4
  100 + 3 * (total_bowls - L - broken_bowls) - 4 * (L + broken_bowls) = payment

theorem find_lost_bowls : ∃ L : ℕ, bowls_problem L ∧ L = 26 :=
  by
  sorry

end find_lost_bowls_l100_100658


namespace lines_parallel_l100_100908

def line1 (x y : ℝ) : Prop := x - y + 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem lines_parallel : 
  (∀ x y, line1 x y ↔ y = x + 2) ∧ 
  (∀ x y, line2 x y ↔ y = x + 1) ∧ 
  ∃ m₁ m₂ c₁ c₂, (∀ x y, (y = m₁ * x + c₁) ↔ line1 x y) ∧ (∀ x y, (y = m₂ * x + c₂) ↔ line2 x y) ∧ m₁ = m₂ ∧ c₁ ≠ c₂ :=
by
  sorry

end lines_parallel_l100_100908


namespace deepak_present_age_l100_100617

variable (R D : ℕ)

theorem deepak_present_age 
  (h1 : R + 22 = 26) 
  (h2 : R / D = 4 / 3) : 
  D = 3 := 
sorry

end deepak_present_age_l100_100617


namespace remainder_division_l100_100760

theorem remainder_division {N : ℤ} (k : ℤ) (h : N = 125 * k + 40) : N % 15 = 10 :=
sorry

end remainder_division_l100_100760


namespace ratio_of_ap_l100_100154

theorem ratio_of_ap (a d : ℕ) (h : 30 * a + 435 * d = 3 * (15 * a + 105 * d)) : a = 8 * d :=
by
  sorry

end ratio_of_ap_l100_100154


namespace sculpture_paint_area_correct_l100_100546

def sculpture_exposed_area (edge_length : ℝ) (num_cubes_layer1 : ℕ) (num_cubes_layer2 : ℕ) (num_cubes_layer3 : ℕ) : ℝ :=
  let area_top_layer1 := num_cubes_layer1 * edge_length ^ 2
  let area_side_layer1 := 8 * 3 * edge_length ^ 2
  let area_top_layer2 := num_cubes_layer2 * edge_length ^ 2
  let area_side_layer2 := 10 * edge_length ^ 2
  let area_top_layer3 := num_cubes_layer3 * edge_length ^ 2
  let area_side_layer3 := num_cubes_layer3 * 4 * edge_length ^ 2
  area_top_layer1 + area_side_layer1 + area_top_layer2 + area_side_layer2 + area_top_layer3 + area_side_layer3

theorem sculpture_paint_area_correct :
  sculpture_exposed_area 1 12 6 2 = 62 := by
  sorry

end sculpture_paint_area_correct_l100_100546


namespace suraj_average_after_9th_innings_l100_100130

theorem suraj_average_after_9th_innings (A : ℕ) 
  (h1 : 8 * A + 90 = 9 * (A + 6)) : 
  (A + 6) = 42 :=
by
  sorry

end suraj_average_after_9th_innings_l100_100130


namespace largest_n_l100_100941

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

axiom a1_gt_zero : a 1 > 0
axiom a2011_a2012_sum_gt_zero : a 2011 + a 2012 > 0
axiom a2011_a2012_prod_lt_zero : a 2011 * a 2012 < 0

-- Sum of first n terms of an arithmetic sequence
def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Problem statement to prove
theorem largest_n (H : is_arithmetic_sequence a) :
  ∀ n, (sequence_sum a 4022 > 0) ∧ (sequence_sum a 4023 < 0) → n = 4022 := by
  sorry

end largest_n_l100_100941


namespace simplify_expr_l100_100644

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) :
  (3/4) * (8/(x^2) + 12*x - 5) = 6/(x^2) + 9*x - 15/4 := by
  sorry

end simplify_expr_l100_100644


namespace parabola_min_value_l100_100713

theorem parabola_min_value (x : ℝ) : (∃ x, x^2 + 10 * x + 21 = -4) := sorry

end parabola_min_value_l100_100713


namespace arithmetic_neg3_plus_4_l100_100055

theorem arithmetic_neg3_plus_4 : -3 + 4 = 1 :=
by
  sorry

end arithmetic_neg3_plus_4_l100_100055


namespace find_a_value_l100_100925

theorem find_a_value (a : ℝ) (x : ℝ) :
  (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 →
  a + 1 ≠ 0 →
  a^2 + 1 = 9 →
  a = 2 * Real.sqrt 2 :=
by
  intro h1 h2 h3
  sorry

end find_a_value_l100_100925


namespace prop1_prop2_prop3_prop4_exists_l100_100004

variable {R : Type*} [LinearOrderedField R]
def f (b c x : R) : R := abs x * x + b * x + c

theorem prop1 (b c x : R) (h : b > 0) : 
  ∀ {x y : R}, x ≤ y → f b c x ≤ f b c y := 
sorry

theorem prop2 (b c : R) (h : b < 0) : 
  ¬ ∃ a : R, ∀ x : R, f b c x ≥ f b c a := 
sorry

theorem prop3 (b c x : R) : 
  f b c (-x) = f b c x + 2*c := 
sorry

theorem prop4_exists (c : R) : 
  ∃ b : R, ∃ x y z : R, f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z := 
sorry

end prop1_prop2_prop3_prop4_exists_l100_100004


namespace eq_satisfies_exactly_four_points_l100_100675

theorem eq_satisfies_exactly_four_points : ∀ (x y : ℝ), 
  (x^2 - 4)^2 + (y^2 - 4)^2 = 0 ↔ 
  (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = -2) := 
by
  sorry

end eq_satisfies_exactly_four_points_l100_100675


namespace exchange_ways_count_l100_100647

theorem exchange_ways_count : ∃ n : ℕ, n = 46 ∧ ∀ x y z : ℕ, x + 2 * y + 5 * z = 20 → n = 46 :=
by
  sorry

end exchange_ways_count_l100_100647


namespace sum_of_coordinates_of_point_D_l100_100383

theorem sum_of_coordinates_of_point_D
  (N : ℝ × ℝ := (6,2))
  (C : ℝ × ℝ := (10, -2))
  (h : ∃ D : ℝ × ℝ, (N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))) :
  ∃ (D : ℝ × ℝ), D.1 + D.2 = 8 := 
by
  obtain ⟨D, hD⟩ := h
  sorry

end sum_of_coordinates_of_point_D_l100_100383


namespace probability_recruitment_l100_100164

-- Definitions for conditions
def P_A : ℚ := 2/3
def P_A_not_and_B_not : ℚ := 1/12
def P_B_and_C : ℚ := 3/8

-- Independence of A, B, and C
axiom independence_A_B_C : ∀ {P_A P_B P_C : Prop}, 
  (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)

-- Definition of probabilities of B and C
def P_B : ℚ := 3/4
def P_C : ℚ := 1/2

-- Main theorem
theorem probability_recruitment : 
  P_A = 2/3 ∧ 
  P_A_not_and_B_not = 1/12 ∧ 
  P_B_and_C = 3/8 ∧ 
  (∀ {P_A P_B P_C : Prop}, 
    (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)) → 
  (P_B = 3/4 ∧ P_C = 1/2) ∧ 
  (2/3 * 3/4 * 1/2 + 1/3 * 3/4 * 1/2 + 2/3 * 1/4 * 1/2 + 2/3 * 3/4 * 1/2 = 17/24) := 
by sorry

end probability_recruitment_l100_100164


namespace fraction_of_roots_l100_100601

theorem fraction_of_roots (a b : ℝ) (h : a * b = -209) (h_sum : a + b = -8) : 
  (a * b) / (a + b) = 209 / 8 := 
by 
  sorry

end fraction_of_roots_l100_100601


namespace machine_working_time_l100_100749

theorem machine_working_time (y : ℝ) :
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) → y = 2 :=
by
  sorry

end machine_working_time_l100_100749


namespace sum_of_digits_of_special_number_l100_100042

theorem sum_of_digits_of_special_number :
  ∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ (100 * x + 10 * y + z = x.factorial + y.factorial + z.factorial) →
  (x + y + z = 10) :=
by
  sorry

end sum_of_digits_of_special_number_l100_100042


namespace circle_tangent_line_l100_100746

theorem circle_tangent_line (a : ℝ) : 
  ∃ (a : ℝ), a = 2 ∨ a = -8 := 
by 
  sorry

end circle_tangent_line_l100_100746


namespace average_percentage_decrease_l100_100248

theorem average_percentage_decrease :
  ∃ (x : ℝ), (5000 * (1 - x / 100)^3 = 2560) ∧ x = 20 :=
by
  sorry

end average_percentage_decrease_l100_100248


namespace division_value_l100_100432

theorem division_value (x y : ℝ) (h1 : (x - 5) / y = 7) (h2 : (x - 14) / 10 = 4) : y = 7 :=
sorry

end division_value_l100_100432


namespace children_in_circle_l100_100020

theorem children_in_circle (n m : ℕ) (k : ℕ) 
  (h1 : n = m) 
  (h2 : n + m = 2 * k) :
  ∃ k', n + m = 4 * k' :=
by
  sorry

end children_in_circle_l100_100020


namespace difference_of_numbers_l100_100730

variable (x y d : ℝ)

theorem difference_of_numbers
  (h1 : x + y = 5)
  (h2 : x - y = d)
  (h3 : x^2 - y^2 = 50) :
  d = 10 :=
by
  sorry

end difference_of_numbers_l100_100730


namespace find_XY_base10_l100_100333

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l100_100333


namespace min_red_beads_l100_100741

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end min_red_beads_l100_100741


namespace probability_of_selected_number_between_l100_100187

open Set

theorem probability_of_selected_number_between (s : Set ℤ) (a b x y : ℤ) 
  (h1 : a = 25) 
  (h2 : b = 925) 
  (h3 : x = 25) 
  (h4 : y = 99) 
  (h5 : s = Set.Icc a b) :
  (y - x + 1 : ℚ) / (b - a + 1 : ℚ) = 75 / 901 := 
by 
  sorry

end probability_of_selected_number_between_l100_100187


namespace slower_pipe_filling_time_l100_100291

theorem slower_pipe_filling_time
  (t : ℝ)
  (H1 : ∀ (time_slow : ℝ), time_slow = t)
  (H2 : ∀ (time_fast : ℝ), time_fast = t / 3)
  (H3 : 1 / t + 1 / (t / 3) = 1 / 40) :
  t = 160 :=
sorry

end slower_pipe_filling_time_l100_100291


namespace one_third_of_1206_is_100_5_percent_of_400_l100_100967

theorem one_third_of_1206_is_100_5_percent_of_400 (n m : ℕ) (f : ℝ) :
  n = 1206 → m = 400 → f = 1 / 3 → (n * f) / m * 100 = 100.5 :=
by
  intros h_n h_m h_f
  rw [h_n, h_m, h_f]
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l100_100967


namespace identical_machine_production_l100_100923

-- Definitions based on given conditions
def machine_production_rate (machines : ℕ) (rate : ℕ) :=
  rate / machines

def bottles_in_minute (machines : ℕ) (rate_per_machine : ℕ) :=
  machines * rate_per_machine

def total_bottles (bottle_rate_per_minute : ℕ) (minutes : ℕ) :=
  bottle_rate_per_minute * minutes

-- Theorem to prove based on the question == answer given conditions
theorem identical_machine_production :
  ∀ (machines_initial machines_final : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ),
    machines_initial = 6 →
    machines_final = 12 →
    bottles_per_minute = 270 →
    minutes = 4 →
    total_bottles (bottles_in_minute machines_final (machine_production_rate machines_initial bottles_per_minute)) minutes = 2160 := by
  intros
  sorry

end identical_machine_production_l100_100923


namespace triangle_base_and_area_l100_100597

theorem triangle_base_and_area
  (height : ℝ)
  (h_height : height = 12)
  (height_base_ratio : ℝ)
  (h_ratio : height_base_ratio = 2 / 3) :
  ∃ (base : ℝ) (area : ℝ),
  base = height / height_base_ratio ∧
  area = base * height / 2 ∧
  base = 18 ∧
  area = 108 :=
by
  sorry

end triangle_base_and_area_l100_100597


namespace complex_z_pow_2017_l100_100079

noncomputable def complex_number_z : ℂ := (1 + Complex.I) / (1 - Complex.I)

theorem complex_z_pow_2017 :
  (complex_number_z * (1 - Complex.I) = 1 + Complex.I) → (complex_number_z ^ 2017 = Complex.I) :=
by
  intro h
  sorry

end complex_z_pow_2017_l100_100079


namespace find_quadruples_l100_100608

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

theorem find_quadruples :
  ∀ x y z n : ℕ, is_solution x y z n ↔ 
  (x, y, z, n) = (1, 1, 1, 2) ∨
  (x, y, z, n) = (0, 0, 1, 1) ∨
  (x, y, z, n) = (0, 1, 0, 1) ∨
  (x, y, z, n) = (1, 0, 0, 1) ∨
  (x, y, z, n) = (0, 0, 0, 0) :=
by
  sorry

end find_quadruples_l100_100608


namespace circle_symmetric_line_a_value_l100_100702

theorem circle_symmetric_line_a_value :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, (x, y) = (-1, 2)) →
  (∀ x y : ℝ, ax + y + 1 = 0) →
  a = 3 :=
by
  sorry

end circle_symmetric_line_a_value_l100_100702


namespace ramu_profit_percent_l100_100421

noncomputable def profitPercent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

theorem ramu_profit_percent :
  profitPercent 42000 13000 61900 = 12.55 :=
by
  sorry

end ramu_profit_percent_l100_100421


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l100_100307

-- Part 1: Expression of the quadratic function
theorem quadratic_function_expression (a : ℝ) (h : a = 0) : 
  ∀ x, (x^2 + (a-2)*x + 3) = x^2 - 2*x + 3 :=
by sorry

-- Part 2: Range of y for 0 < x < 3
theorem quadratic_function_range (x y : ℝ) (h : ∀ x, y = x^2 - 2*x + 3) (hx : 0 < x ∧ x < 3) :
  2 ≤ y ∧ y < 6 :=
by sorry

-- Part 3: Range of m for y1 > y2
theorem quadratic_function_m_range (m y1 y2 : ℝ) (P Q : ℝ × ℝ)
  (h1 : P = (m - 1, y1)) (h2 : Q = (m, y2)) (h3 : y1 > y2) :
  m < 3 / 2 :=
by sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l100_100307


namespace limit_a_n_l100_100972

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l100_100972


namespace area_of_square_with_perimeter_l100_100337

def perimeter_of_square (s : ℝ) : ℝ := 4 * s

def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_square_with_perimeter (p : ℝ) (h : perimeter_of_square (3 * p) = 12 * p) : area_of_square (3 * p) = 9 * p^2 := by
  sorry

end area_of_square_with_perimeter_l100_100337


namespace intersection_A_B_l100_100722

def interval_A : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def interval_B : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem intersection_A_B :
  interval_A ∩ interval_B = { x | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
sorry

end intersection_A_B_l100_100722


namespace num_solutions_l100_100552

-- Define the problem and the condition
def matrix_eq (x : ℝ) : Prop :=
  3 * x^2 - 4 * x = 7

-- Define the main theorem to prove the number of solutions
theorem num_solutions : ∃! x : ℝ, matrix_eq x :=
sorry

end num_solutions_l100_100552


namespace range_of_m_l100_100009

theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, m * x0^2 + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) → -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l100_100009


namespace no_integer_solutions_for_equation_l100_100157

theorem no_integer_solutions_for_equation : ¬∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := 
  by sorry

end no_integer_solutions_for_equation_l100_100157


namespace cylindrical_to_rectangular_coords_l100_100014

/--
Cylindrical coordinates (r, θ, z)
Rectangular coordinates (x, y, z)
-/
theorem cylindrical_to_rectangular_coords (r θ z : ℝ) (hx : x = r * Real.cos θ)
    (hy : y = r * Real.sin θ) (hz : z = z) :
    (r, θ, z) = (5, Real.pi / 4, 2) → (x, y, z) = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  sorry

end cylindrical_to_rectangular_coords_l100_100014


namespace integer_solutions_of_quadratic_l100_100417

theorem integer_solutions_of_quadratic (k : ℤ) :
  ∀ x : ℤ, (6 - k) * (9 - k) * x^2 - (117 - 15 * k) * x + 54 = 0 ↔
  k = 3 ∨ k = 7 ∨ k = 15 ∨ k = 6 ∨ k = 9 :=
by
  sorry

end integer_solutions_of_quadratic_l100_100417


namespace domain_sqrt_tan_x_sub_sqrt3_l100_100486

open Real

noncomputable def domain := {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2}

theorem domain_sqrt_tan_x_sub_sqrt3 :
  {x | ∃ y : ℝ, y = sqrt (tan x - sqrt 3)} = domain :=
by
  sorry

end domain_sqrt_tan_x_sub_sqrt3_l100_100486


namespace quadratic_form_decomposition_l100_100594

theorem quadratic_form_decomposition (a b c : ℝ) (h : ∀ x : ℝ, 8 * x^2 + 64 * x + 512 = a * (x + b) ^ 2 + c) :
  a + b + c = 396 := 
sorry

end quadratic_form_decomposition_l100_100594


namespace value_of_5_S_3_l100_100752

def operation_S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem value_of_5_S_3 : operation_S 5 3 = 8 :=
by
  sorry

end value_of_5_S_3_l100_100752


namespace brother_reading_time_l100_100619

variable (my_time_in_hours : ℕ)
variable (speed_ratio : ℕ)

theorem brother_reading_time
  (h1 : my_time_in_hours = 3)
  (h2 : speed_ratio = 4) :
  my_time_in_hours * 60 / speed_ratio = 45 := 
by
  sorry

end brother_reading_time_l100_100619


namespace possible_values_for_N_l100_100406

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l100_100406


namespace recurring_decimal_sum_as_fraction_l100_100275

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l100_100275


namespace height_of_picture_frame_l100_100484

-- Define the given conditions
def width : ℕ := 6
def perimeter : ℕ := 30
def perimeter_formula (w h : ℕ) : ℕ := 2 * (w + h)

-- Prove that the height of the picture frame is 9 inches
theorem height_of_picture_frame : ∃ height : ℕ, height = 9 ∧ perimeter_formula width height = perimeter :=
by
  -- Proof goes here
  sorry

end height_of_picture_frame_l100_100484


namespace a5_is_16_S8_is_255_l100_100088

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * seq n

-- Definition of the geometric sum
def geom_sum (n : ℕ) : ℕ :=
  (2 ^ (n + 1) - 1)

-- Prove that a₅ = 16
theorem a5_is_16 : seq 5 = 16 :=
  by
  unfold seq
  sorry

-- Prove that the sum of the first 8 terms, S₈ = 255
theorem S8_is_255 : geom_sum 7 = 255 :=
  by 
  unfold geom_sum
  sorry

end a5_is_16_S8_is_255_l100_100088


namespace variance_of_scores_l100_100385

-- Define the student's scores
def scores : List ℕ := [130, 125, 126, 126, 128]

-- Define a function to calculate the mean
def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define a function to calculate the variance
def variance (l : List ℕ) : ℕ :=
  let avg := mean l
  (l.map (λ x => (x - avg) * (x - avg))).sum / l.length

-- The proof statement (no proof provided, use sorry)
theorem variance_of_scores : variance scores = 3 := by sorry

end variance_of_scores_l100_100385


namespace sum_of_transformed_numbers_l100_100986

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l100_100986


namespace alice_ride_top_speed_l100_100891

-- Define the conditions
variables (x y : Real) -- x is the hours at 25 mph, y is the hours at 15 mph.
def distance_eq : Prop := 25 * x + 15 * y + 10 * (9 - x - y) = 162
def time_eq : Prop := x + y ≤ 9

-- Define the final answer
def final_answer : Prop := x = 2.7

-- The statement to prove
theorem alice_ride_top_speed : distance_eq x y ∧ time_eq x y → final_answer x := sorry

end alice_ride_top_speed_l100_100891


namespace non_honda_red_percentage_l100_100232

-- Define the conditions
def total_cars : ℕ := 900
def honda_percentage_red : ℝ := 0.90
def total_percentage_red : ℝ := 0.60
def honda_cars : ℕ := 500

-- The statement to prove
theorem non_honda_red_percentage : 
  (0.60 * 900 - 0.90 * 500) / (900 - 500) * 100 = 22.5 := 
  by sorry

end non_honda_red_percentage_l100_100232


namespace TV_cost_is_1700_l100_100293

def hourlyRate : ℝ := 10
def workHoursPerWeek : ℝ := 30
def weeksPerMonth : ℝ := 4
def additionalHours : ℝ := 50

def weeklyEarnings : ℝ := hourlyRate * workHoursPerWeek
def monthlyEarnings : ℝ := weeklyEarnings * weeksPerMonth
def additionalEarnings : ℝ := hourlyRate * additionalHours

def TVCost : ℝ := monthlyEarnings + additionalEarnings

theorem TV_cost_is_1700 : TVCost = 1700 := sorry

end TV_cost_is_1700_l100_100293


namespace Joan_bought_72_eggs_l100_100175

def dozen := 12
def dozens_Joan_bought := 6
def eggs_Joan_bought := dozens_Joan_bought * dozen

theorem Joan_bought_72_eggs : eggs_Joan_bought = 72 := by
  sorry

end Joan_bought_72_eggs_l100_100175


namespace sufficiency_but_not_necessary_l100_100120

theorem sufficiency_but_not_necessary (x y : ℝ) : |x| + |y| ≤ 1 → x^2 + y^2 ≤ 1 ∧ ¬(x^2 + y^2 ≤ 1 → |x| + |y| ≤ 1) :=
by
  sorry

end sufficiency_but_not_necessary_l100_100120


namespace usual_time_to_catch_bus_l100_100236

theorem usual_time_to_catch_bus (S T : ℝ) (h1 : S / ((5/4) * S) = (T + 5) / T) : T = 25 :=
by sorry

end usual_time_to_catch_bus_l100_100236


namespace solve_fractional_equation_l100_100623

theorem solve_fractional_equation (x : ℝ) (h : (3 * x + 6) / (x ^ 2 + 5 * x - 6) = (3 - x) / (x - 1)) (hx : x ≠ 1) : x = -4 := 
sorry

end solve_fractional_equation_l100_100623


namespace books_read_in_common_l100_100112

theorem books_read_in_common (T D B total X : ℕ) 
  (hT : T = 23) 
  (hD : D = 12) 
  (hB : B = 17) 
  (htotal : total = 47)
  (h_eq : (T - X) + (D - X) + B + 1 = total) : 
  X = 3 :=
by
  -- Here would go the proof details.
  sorry

end books_read_in_common_l100_100112


namespace equal_circle_radius_l100_100896

theorem equal_circle_radius (r R : ℝ) (h1: r > 0) (h2: R > 0)
  : ∃ x : ℝ, x = r * R / (R + r) :=
by 
  sorry

end equal_circle_radius_l100_100896


namespace original_number_l100_100598

theorem original_number (N : ℕ) (h : ∃ k : ℕ, N + 1 = 9 * k) : N = 8 :=
sorry

end original_number_l100_100598


namespace product_of_fractions_l100_100159

theorem product_of_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = 1 / 7 :=
  sorry

end product_of_fractions_l100_100159


namespace total_floor_area_is_correct_l100_100257

-- Define the combined area of the three rugs
def combined_area_of_rugs : ℕ := 212

-- Define the area covered by exactly two layers of rug
def area_covered_by_two_layers : ℕ := 24

-- Define the area covered by exactly three layers of rug
def area_covered_by_three_layers : ℕ := 24

-- Define the total floor area covered by the rugs
def total_floor_area_covered : ℕ :=
  combined_area_of_rugs - area_covered_by_two_layers - 2 * area_covered_by_three_layers

-- The theorem stating the total floor area covered
theorem total_floor_area_is_correct : total_floor_area_covered = 140 := by
  sorry

end total_floor_area_is_correct_l100_100257


namespace ashton_pencils_left_l100_100174

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l100_100174


namespace find_unknown_number_l100_100255

theorem find_unknown_number (x : ℤ) :
  (20 + 40 + 60) / 3 = 5 + (20 + 60 + x) / 3 → x = 25 :=
by
  sorry

end find_unknown_number_l100_100255


namespace distinct_integer_values_b_for_quadratic_l100_100964

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l100_100964


namespace prove_intersection_area_is_correct_l100_100317

noncomputable def octahedron_intersection_area 
  (side_length : ℝ) (cut_height_factor : ℝ) : ℝ :=
  have height_triangular_face := Real.sqrt (side_length^2 - (side_length / 2)^2)
  have plane_height := cut_height_factor * height_triangular_face
  have proportional_height := plane_height / height_triangular_face
  let new_side_length := proportional_height * side_length
  have hexagon_area := (3 * Real.sqrt 3 / 2) * (new_side_length^2) / 2 
  (3 * Real.sqrt 3 / 2) * (new_side_length^2)

theorem prove_intersection_area_is_correct 
  : 
  octahedron_intersection_area 2 (3 / 4) = 9 * Real.sqrt 3 / 8 :=
  sorry 

example : 9 + 3 + 8 = 20 := 
  by rfl

end prove_intersection_area_is_correct_l100_100317


namespace simplify_expression_l100_100850

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  ( ((x+1)^2 * (x^2 - x + 1)^2 / (x^3 + 1)^2)^2 *
    ((x-1)^2 * (x^2 + x + 1)^2 / (x^3 - 1)^2)^2
  ) = 1 :=
by
  sorry

end simplify_expression_l100_100850


namespace fruit_boxes_needed_l100_100204

noncomputable def fruit_boxes : ℕ × ℕ × ℕ :=
  let baskets : ℕ := 7
  let peaches_per_basket : ℕ := 23
  let apples_per_basket : ℕ := 19
  let oranges_per_basket : ℕ := 31
  let peaches_eaten : ℕ := 7
  let apples_eaten : ℕ := 5
  let oranges_eaten : ℕ := 3
  let peaches_box_size : ℕ := 13
  let apples_box_size : ℕ := 11
  let oranges_box_size : ℕ := 17

  let total_peaches := baskets * peaches_per_basket
  let total_apples := baskets * apples_per_basket
  let total_oranges := baskets * oranges_per_basket

  let remaining_peaches := total_peaches - peaches_eaten
  let remaining_apples := total_apples - apples_eaten
  let remaining_oranges := total_oranges - oranges_eaten

  let peaches_boxes := (remaining_peaches + peaches_box_size - 1) / peaches_box_size
  let apples_boxes := (remaining_apples + apples_box_size - 1) / apples_box_size
  let oranges_boxes := (remaining_oranges + oranges_box_size - 1) / oranges_box_size

  (peaches_boxes, apples_boxes, oranges_boxes)

theorem fruit_boxes_needed :
  fruit_boxes = (12, 12, 13) := by 
  sorry

end fruit_boxes_needed_l100_100204


namespace identify_a_b_l100_100899

theorem identify_a_b (a b : ℝ) (h : ∀ x y : ℝ, (⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋)) : 
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) :=
sorry

end identify_a_b_l100_100899


namespace range_of_a_l100_100230

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) → (-4 ≤ a ∧ a ≤ 4) :=
by 
  sorry

end range_of_a_l100_100230


namespace smallest_N_satisfying_frequencies_l100_100461

def percentageA := 1 / 5
def percentageB := 3 / 8
def percentageC := 1 / 4
def percentageD := 1 / 8
def percentageE := 1 / 20

def Divisible (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = k * d

theorem smallest_N_satisfying_frequencies :
  ∃ N : ℕ, 
    Divisible N 5 ∧ 
    Divisible N 8 ∧ 
    Divisible N 4 ∧ 
    Divisible N 20 ∧ 
    N = 40 := sorry

end smallest_N_satisfying_frequencies_l100_100461


namespace number_halfway_between_l100_100786

theorem number_halfway_between :
  ∃ x : ℚ, x = (1/12 + 1/14) / 2 ∧ x = 13 / 168 :=
sorry

end number_halfway_between_l100_100786


namespace parity_of_expression_l100_100663

theorem parity_of_expression (a b c : ℕ) (h_apos : 0 < a) (h_aodd : a % 2 = 1) (h_beven : b % 2 = 0) :
  (3^a + (b+1)^2 * c) % 2 = if c % 2 = 0 then 1 else 0 :=
sorry

end parity_of_expression_l100_100663


namespace find_p_l100_100366

variable (a b c p : ℚ)

theorem find_p (h1 : 5 / (a + b) = p / (a + c)) (h2 : p / (a + c) = 8 / (c - b)) : p = 13 := by
  sorry

end find_p_l100_100366


namespace smallest_square_number_l100_100149

theorem smallest_square_number (x y : ℕ) (hx : ∃ a, x = a ^ 2) (hy : ∃ b, y = b ^ 3) 
  (h_simp: ∃ c d, x / (y ^ 3) = c ^ 3 / d ^ 2 ∧ c > 1 ∧ d > 1): x = 64 := by
  sorry

end smallest_square_number_l100_100149


namespace average_temp_is_correct_l100_100215

-- Define the temperatures for each day
def sunday_temp : ℕ := 40
def monday_temp : ℕ := 50
def tuesday_temp : ℕ := 65
def wednesday_temp : ℕ := 36
def thursday_temp : ℕ := 82
def friday_temp : ℕ := 72
def saturday_temp : ℕ := 26

-- Define the total number of days in the week
def days_in_week : ℕ := 7

-- Define the total temperature for the week
def total_temperature : ℕ := sunday_temp + monday_temp + tuesday_temp + 
                             wednesday_temp + thursday_temp + friday_temp + 
                             saturday_temp

-- Define the average temperature calculation
def average_temperature : ℕ := total_temperature / days_in_week

-- The theorem to be proved
theorem average_temp_is_correct : average_temperature = 53 := by
  sorry

end average_temp_is_correct_l100_100215


namespace num_first_graders_in_class_l100_100953

def numKindergartners := 14
def numSecondGraders := 4
def totalStudents := 42

def numFirstGraders : Nat := totalStudents - (numKindergartners + numSecondGraders)

theorem num_first_graders_in_class :
  numFirstGraders = 24 :=
by
  sorry

end num_first_graders_in_class_l100_100953


namespace find_initial_number_l100_100367

theorem find_initial_number (x : ℕ) (h : ∃ y : ℕ, x * y = 4 ∧ y = 2) : x = 2 :=
by
  sorry

end find_initial_number_l100_100367


namespace intersection_M_N_l100_100390

def M : Set ℝ := { x | -1 < x ∧ x < 1 }
def N : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l100_100390


namespace polynomial_transformation_l100_100660

theorem polynomial_transformation (x y : ℂ) (h : y = x + 1/x) : x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by
  sorry

end polynomial_transformation_l100_100660


namespace closest_integers_to_2013_satisfy_trig_eq_l100_100430

noncomputable def closestIntegersSatisfyingTrigEq (x : ℝ) : Prop := 
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2)

theorem closest_integers_to_2013_satisfy_trig_eq : closestIntegersSatisfyingTrigEq (1935 * (Real.pi / 180)) ∧ closestIntegersSatisfyingTrigEq (2025 * (Real.pi / 180)) :=
sorry

end closest_integers_to_2013_satisfy_trig_eq_l100_100430


namespace sum_of_distances_minimized_l100_100792

theorem sum_of_distances_minimized (x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) : 
  abs (x - 0) + abs (x - 50) = 50 := 
by
  sorry

end sum_of_distances_minimized_l100_100792


namespace solutions_exist_l100_100420

theorem solutions_exist (k : ℤ) : ∃ x y : ℤ, (x = 3 * k + 2) ∧ (y = 7 * k + 4) ∧ (7 * x - 3 * y = 2) :=
by {
  -- Proof will be filled in here
  sorry
}

end solutions_exist_l100_100420


namespace velocity_of_current_l100_100254

theorem velocity_of_current
  (v c : ℝ) 
  (h1 : 32 = (v + c) * 6) 
  (h2 : 14 = (v - c) * 6) :
  c = 1.5 :=
by
  sorry

end velocity_of_current_l100_100254


namespace simplify_fraction_l100_100172

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 :=
by
  sorry

end simplify_fraction_l100_100172


namespace prove_trig_inequality_l100_100455

noncomputable def trig_inequality : Prop :=
  (0 < 1 / 2) ∧ (1 / 2 < Real.pi / 6) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.sin x < Real.sin y) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.cos x > Real.cos y) →
  (Real.cos (1 / 2) > Real.tan (1 / 2) ∧ Real.tan (1 / 2) > Real.sin (1 / 2))

theorem prove_trig_inequality : trig_inequality :=
by
  sorry

end prove_trig_inequality_l100_100455


namespace total_slices_is_78_l100_100629

-- Definitions based on conditions
def ratio_buzz_waiter (x : ℕ) : Prop := (5 * x) + (8 * x) = 78
def waiter_condition (x : ℕ) : Prop := (8 * x) - 20 = 28

-- Prove that the total number of slices is 78 given conditions
theorem total_slices_is_78 (x : ℕ) (h1 : ratio_buzz_waiter x) (h2 : waiter_condition x) : (5 * x) + (8 * x) = 78 :=
by
  sorry

end total_slices_is_78_l100_100629


namespace greater_number_l100_100527

theorem greater_number (x y : ℕ) (h_sum : x + y = 50) (h_diff : x - y = 16) : x = 33 :=
by
  sorry

end greater_number_l100_100527


namespace range_of_a_in_third_quadrant_l100_100794

theorem range_of_a_in_third_quadrant (a : ℝ) :
  let Z_re := a^2 - 2*a
  let Z_im := a^2 - a - 2
  (Z_re < 0 ∧ Z_im < 0) → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_in_third_quadrant_l100_100794


namespace solve_abs_equation_l100_100282

theorem solve_abs_equation (y : ℝ) (h8 : y < 8) (h_eq : |y - 8| + 2 * y = 12) : y = 4 :=
sorry

end solve_abs_equation_l100_100282


namespace divisors_of_30240_l100_100679

theorem divisors_of_30240 : 
  ∃ s : Finset ℕ, (s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ d ∈ s, (30240 % d = 0)) ∧ (s.card = 9) :=
by
  sorry

end divisors_of_30240_l100_100679


namespace problem1_problem2_l100_100742

-- Problem 1

def a : ℚ := -1 / 2
def b : ℚ := -1

theorem problem1 :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3 / 4 :=
by
  sorry

-- Problem 2

def x : ℚ := 1 / 2
def y : ℚ := -2 / 3
axiom condition2 : abs (2 * x - 1) + (3 * y + 2)^2 = 0

theorem problem2 :
  5 * x^2 - (2 * x * y - 3 * (x * y / 3 + 2) + 5 * x^2) = 19 / 3 :=
by
  have h : abs (2 * x - 1) + (3 * y + 2)^2 = 0 := condition2
  sorry

end problem1_problem2_l100_100742


namespace find_a_if_parallel_l100_100247

-- Definitions of the vectors and the scalar a
def vector_m : ℝ × ℝ := (2, 1)
def vector_n (a : ℝ) : ℝ × ℝ := (4, a)

-- Condition for parallel vectors
def are_parallel (m n : ℝ × ℝ) : Prop :=
  m.1 / n.1 = m.2 / n.2

-- Lean 4 statement
theorem find_a_if_parallel (a : ℝ) (h : are_parallel vector_m (vector_n a)) : a = 2 :=
by
  sorry

end find_a_if_parallel_l100_100247


namespace evaluate_x2_y2_l100_100982

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l100_100982


namespace thickness_and_width_l100_100185
noncomputable def channelThicknessAndWidth (L W v₀ h₀ θ g : ℝ) : ℝ × ℝ :=
let K := W * h₀ * v₀
let v := v₀ + Real.sqrt (2 * g * Real.sin θ * L)
let x := K / (v * W)
let y := K / (h₀ * v)
(x, y)

theorem thickness_and_width :
  channelThicknessAndWidth 10 3.5 1.4 0.4 (12 * Real.pi / 180) 9.81 = (0.072, 0.629) :=
by
  sorry

end thickness_and_width_l100_100185


namespace mickey_horses_per_week_l100_100804

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l100_100804


namespace determine_c_l100_100782

noncomputable def ab5c_decimal (a b c : ℕ) : ℕ :=
  729 * a + 81 * b + 45 + c

theorem determine_c (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : ∃ k : ℕ, ab5c_decimal a b c = k^2) :
  c = 0 ∨ c = 7 :=
by
  sorry

end determine_c_l100_100782


namespace inequality_solution_l100_100686

theorem inequality_solution (x : ℝ) (h : 1 / (x - 2) < 4) : x < 2 ∨ x > 9 / 4 :=
sorry

end inequality_solution_l100_100686


namespace pizza_remained_l100_100958

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end pizza_remained_l100_100958


namespace camp_weights_l100_100704

theorem camp_weights (m_e_w : ℕ) (m_e_w1 : ℕ) (c_w : ℕ) (m_e_w2 : ℕ) (d : ℕ)
  (h1 : m_e_w = 30) 
  (h2 : m_e_w1 = 28) 
  (h3 : c_w = 56)
  (h4 : m_e_w = m_e_w1 + d)
  (h5 : m_e_w1 = m_e_w2 + d)
  (h6 : c_w = m_e_w + m_e_w1 + d) :
  m_e_w = 28 ∧ m_e_w2 = 26 := 
by {
    sorry
}

end camp_weights_l100_100704


namespace geometric_first_term_l100_100180

theorem geometric_first_term (a r : ℝ) (h1 : a * r^3 = 720) (h2 : a * r^6 = 5040) : 
a = 720 / 7 :=
by
  sorry

end geometric_first_term_l100_100180


namespace colony_fungi_day_l100_100829

theorem colony_fungi_day (n : ℕ): 
  (4 * 2^n > 150) = (n = 6) :=
sorry

end colony_fungi_day_l100_100829


namespace Anne_carrying_four_cats_weight_l100_100840

theorem Anne_carrying_four_cats_weight : 
  let w1 := 2
  let w2 := 1.5 * w1
  let m1 := 2 * w1
  let m2 := w1 + w2
  w1 + w2 + m1 + m2 = 14 :=
by
  sorry

end Anne_carrying_four_cats_weight_l100_100840


namespace least_number_divisible_l100_100694

-- Define the numbers as given in the conditions
def given_number : ℕ := 3072
def divisor1 : ℕ := 57
def divisor2 : ℕ := 29
def least_number_to_add : ℕ := 234

-- Define the LCM
noncomputable def lcm_57_29 : ℕ := Nat.lcm divisor1 divisor2

-- Prove that adding least_number_to_add to given_number makes it divisible by both divisors
theorem least_number_divisible :
  (given_number + least_number_to_add) % divisor1 = 0 ∧ 
  (given_number + least_number_to_add) % divisor2 = 0 := 
by
  -- Proof should be provided here
  sorry

end least_number_divisible_l100_100694


namespace polynomial_divisibility_condition_l100_100678

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^5 - x^4 + x^3 - p * x^2 + q * x - 6

theorem polynomial_divisibility_condition (p q : ℝ) :
  (f (-1) p q = 0) ∧ (f 2 p q = 0) → 
  (p = 0) ∧ (q = -9) := by
  sorry

end polynomial_divisibility_condition_l100_100678


namespace paper_cups_count_l100_100718

variables (P C : ℝ) (x : ℕ)

theorem paper_cups_count :
  100 * P + x * C = 7.50 ∧ 20 * P + 40 * C = 1.50 → x = 200 :=
sorry

end paper_cups_count_l100_100718


namespace domain_of_sqrt_function_l100_100868

theorem domain_of_sqrt_function :
  {x : ℝ | (1 / (Real.log x / Real.log 2) - 2 ≥ 0) ∧ (x > 0) ∧ (x ≠ 1)} 
  = {x : ℝ | 1 < x ∧ x ≤ Real.sqrt 10} :=
sorry

end domain_of_sqrt_function_l100_100868


namespace find_dividend_l100_100928

def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 16) (h_quotient : quotient = 8) (h_remainder : remainder = 4) :
  dividend divisor quotient remainder = 132 :=
by
  sorry

end find_dividend_l100_100928


namespace ten_millions_in_hundred_million_hundred_thousands_in_million_l100_100077

theorem ten_millions_in_hundred_million :
  (100 * 10^6) / (10 * 10^6) = 10 :=
by sorry

theorem hundred_thousands_in_million :
  (1 * 10^6) / (100 * 10^3) = 10 :=
by sorry

end ten_millions_in_hundred_million_hundred_thousands_in_million_l100_100077


namespace number_properties_l100_100966

-- Define what it means for a digit to be in a specific place
def digit_at_place (n place : ℕ) (d : ℕ) : Prop := 
  (n / 10 ^ place) % 10 = d

-- The given number
def specific_number : ℕ := 670154500

-- Conditions: specific number has specific digit in defined places
theorem number_properties : (digit_at_place specific_number 7 7) ∧ (digit_at_place specific_number 2 5) :=
by
  -- Proof of the theorem
  sorry

end number_properties_l100_100966


namespace compound_oxygen_atoms_l100_100036

theorem compound_oxygen_atoms 
  (C_atoms : ℕ)
  (H_atoms : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_O : ℝ) :
  C_atoms = 4 →
  H_atoms = 8 →
  total_molecular_weight = 88 →
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  atomic_weight_O = 16.00 →
  (total_molecular_weight - (C_atoms * atomic_weight_C + H_atoms * atomic_weight_H)) / atomic_weight_O = 2 := 
by 
  intros;
  sorry

end compound_oxygen_atoms_l100_100036


namespace total_time_for_phd_l100_100058

def acclimation_period : ℕ := 1 -- in years
def basics_learning_phase : ℕ := 2 -- in years
def research_factor : ℝ := 1.75 -- 75% more time on research
def research_time_without_sabbaticals_and_conferences : ℝ := basics_learning_phase * research_factor
def first_sabbatical : ℝ := 0.5 -- in years (6 months)
def second_sabbatical : ℝ := 0.25 -- in years (3 months)
def first_conference : ℝ := 0.3333 -- in years (4 months)
def second_conference : ℝ := 0.4166 -- in years (5 months)
def additional_research_time : ℝ := first_sabbatical + second_sabbatical + first_conference + second_conference
def total_research_phase_time : ℝ := research_time_without_sabbaticals_and_conferences + additional_research_time
def dissertation_factor : ℝ := 0.5 -- half as long as acclimation period
def time_spent_writing_without_conference : ℝ := dissertation_factor * acclimation_period
def dissertation_conference : ℝ := 0.25 -- in years (3 months)
def total_dissertation_writing_time : ℝ := time_spent_writing_without_conference + dissertation_conference

theorem total_time_for_phd : 
  (acclimation_period + basics_learning_phase + total_research_phase_time + total_dissertation_writing_time) = 8.75 :=
by
  sorry

end total_time_for_phd_l100_100058


namespace x_value_when_y_2000_l100_100127

noncomputable def x_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) : ℝ :=
  if hy : y = 2000 then (1 / (50 : ℝ)^(1/3)) else x

-- Theorem statement
theorem x_value_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) :
  x_when_y_2000 x y hxy_pos hxy_inv h_init = 1 / (50 : ℝ)^(1/3) :=
sorry

end x_value_when_y_2000_l100_100127


namespace minimum_value_l100_100343

def f (x : ℝ) : ℝ := |x - 4| + |x + 7| + |x - 5|

theorem minimum_value : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = 4 :=
by
  -- Sorry is used here to skip the proof
  sorry

end minimum_value_l100_100343


namespace simplify_expression_l100_100479

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
    (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 :=
sorry

end simplify_expression_l100_100479


namespace quotient_of_0_009_div_0_3_is_0_03_l100_100787

-- Statement:
theorem quotient_of_0_009_div_0_3_is_0_03 (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 :=
by
  sorry

end quotient_of_0_009_div_0_3_is_0_03_l100_100787


namespace min_value_expression_l100_100596

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 4) :
  ∃ c : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y * z = 4 → 
  (2 * (x / y) + 3 * (y / z) + 4 * (z / x)) ≥ c) ∧ c = 6 :=
by
  sorry

end min_value_expression_l100_100596


namespace calculate_a_minus_b_l100_100874

theorem calculate_a_minus_b : 
  ∀ (a b : ℚ), (y = a * x + b) 
  ∧ (y = 4 ↔ x = 3) 
  ∧ (y = 22 ↔ x = 10) 
  → (a - b = 6 + 2 / 7)
:= sorry

end calculate_a_minus_b_l100_100874


namespace greatest_monthly_drop_l100_100117

-- Definition of monthly price changes
def price_change_jan : ℝ := -1.00
def price_change_feb : ℝ := 2.50
def price_change_mar : ℝ := 0.00
def price_change_apr : ℝ := -3.00
def price_change_may : ℝ := -1.50
def price_change_jun : ℝ := 1.00

-- Proving the month with the greatest monthly drop in price
theorem greatest_monthly_drop :
  (price_change_apr < price_change_jan) ∧
  (price_change_apr < price_change_feb) ∧
  (price_change_apr < price_change_mar) ∧
  (price_change_apr < price_change_may) ∧
  (price_change_apr < price_change_jun) :=
by
  sorry

end greatest_monthly_drop_l100_100117


namespace lines_parallel_m_values_l100_100881

theorem lines_parallel_m_values (m : ℝ) :
    (∀ x y : ℝ, (m - 2) * x - y - 1 = 0 ↔ 3 * x - m * y = 0) ↔ (m = -1 ∨ m = 3) :=
by
  sorry

end lines_parallel_m_values_l100_100881


namespace imo1983_q24_l100_100384

theorem imo1983_q24 :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ 
    (∀ x ∈ S, x > 0 ∧ x ≤ 10^5) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → (x + z ≠ 2 * y)) :=
sorry

end imo1983_q24_l100_100384


namespace Bo_knew_percentage_l100_100393

-- Definitions from the conditions
def total_flashcards := 800
def words_per_day := 16
def days := 40
def total_words_to_learn := words_per_day * days
def known_words := total_flashcards - total_words_to_learn

-- Statement that we need to prove
theorem Bo_knew_percentage : (known_words.toFloat / total_flashcards.toFloat) * 100 = 20 :=
by
  sorry  -- Proof is omitted as per the instructions

end Bo_knew_percentage_l100_100393


namespace max_elements_in_S_l100_100555

theorem max_elements_in_S : ∀ (S : Finset ℕ), 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → 
    (∃ c ∈ S, Nat.Coprime c a ∧ Nat.Coprime c b) ∧
    (∃ d ∈ S, ∃ x y : ℕ, x ∣ a ∧ x ∣ b ∧ x ∣ d ∧ y ∣ a ∧ y ∣ b ∧ y ∣ d)) →
  S.card ≤ 72 :=
by sorry

end max_elements_in_S_l100_100555


namespace a_5_eq_neg1_l100_100136

-- Given conditions
def S (n : ℕ) : ℤ := n^2 - 10 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem a_5_eq_neg1 : a 5 = -1 :=
by sorry

end a_5_eq_neg1_l100_100136


namespace identity_proof_l100_100373

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
    2 / (a - b) + 2 / (b - c) + 2 / (c - a) :=
by
  sorry

end identity_proof_l100_100373


namespace max_value_m_l100_100618

theorem max_value_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2 * x - 1 < m) → m ≤ 5 :=
by
  sorry

end max_value_m_l100_100618


namespace czechoslovak_inequality_l100_100238

-- Define the triangle and the points
structure Triangle (α : Type) [LinearOrderedRing α] :=
(A B C : α × α)

variables {α : Type} [LinearOrderedRing α]

-- Define the condition that O is on the segment AB but is not a vertex
def on_segment (O A B : α × α) : Prop :=
  ∃ x : α, 0 < x ∧ x < 1 ∧ O = (A.1 + x * (B.1 - A.1), A.2 + x * (B.2 - A.2))

-- Define the dot product for vectors
def dot (u v: α × α) : α := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem czechoslovak_inequality (T : Triangle α) (O : α × α) (hO : on_segment O T.A T.B) :
  dot O T.C * dot T.A T.B < dot T.A O * dot T.B T.C + dot T.B O * dot T.A T.C :=
sorry

end czechoslovak_inequality_l100_100238


namespace digit_equation_l100_100887

-- Definitions for digits and the equation components
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x ≤ 9

def three_digit_number (A B C : ℤ) : ℤ := 100 * A + 10 * B + C
def two_digit_number (A D : ℤ) : ℤ := 10 * A + D
def four_digit_number (A D C : ℤ) : ℤ := 1000 * A + 100 * D + 10 * D + C

-- Statement of the theorem
theorem digit_equation (A B C D : ℤ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) :
  three_digit_number A B C * two_digit_number A D = four_digit_number A D C :=
sorry

end digit_equation_l100_100887


namespace jonah_profit_l100_100419

def cost_per_pineapple (quantity : ℕ) : ℝ :=
  if quantity > 50 then 1.60 else if quantity > 40 then 1.80 else 2.00

def total_cost (quantity : ℕ) : ℝ :=
  cost_per_pineapple quantity * quantity

def bundle_revenue (bundles : ℕ) : ℝ :=
  bundles * 20

def single_ring_revenue (rings : ℕ) : ℝ :=
  rings * 4

def total_revenue (bundles : ℕ) (rings : ℕ) : ℝ :=
  bundle_revenue bundles + single_ring_revenue rings

noncomputable def profit (quantity bundles rings : ℕ) : ℝ :=
  total_revenue bundles rings - total_cost quantity

theorem jonah_profit : profit 60 35 150 = 1204 := by
  sorry

end jonah_profit_l100_100419


namespace container_ratio_l100_100980

theorem container_ratio (A B : ℝ) (h : (4 / 5) * A = (2 / 3) * B) : (A / B) = (5 / 6) :=
by
  sorry

end container_ratio_l100_100980


namespace compare_xyz_l100_100965

noncomputable def x := (0.5 : ℝ)^(0.5 : ℝ)
noncomputable def y := (0.5 : ℝ)^(1.3 : ℝ)
noncomputable def z := (1.3 : ℝ)^(0.5 : ℝ)

theorem compare_xyz : z > x ∧ x > y := by
  sorry

end compare_xyz_l100_100965


namespace max_profit_l100_100060

def fixed_cost : ℝ := 20
def variable_cost_per_unit : ℝ := 10

def total_cost (Q : ℝ) := fixed_cost + variable_cost_per_unit * Q

def revenue (Q : ℝ) := 40 * Q - Q^2

def profit (Q : ℝ) := revenue Q - total_cost Q

def Q_optimized : ℝ := 15

theorem max_profit : profit Q_optimized = 205 := by
  sorry -- Proof goes here.

end max_profit_l100_100060


namespace rectangular_garden_width_l100_100609

theorem rectangular_garden_width
  (w : ℝ)
  (h₁ : ∃ l, l = 3 * w)
  (h₂ : ∃ A, A = l * w ∧ A = 507) : 
  w = 13 :=
by
  sorry

end rectangular_garden_width_l100_100609


namespace quadratic_inequality_real_roots_l100_100842

theorem quadratic_inequality_real_roots (c : ℝ) (h_pos : 0 < c) (h_ineq : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
sorry

end quadratic_inequality_real_roots_l100_100842


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l100_100813

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l100_100813


namespace mean_difference_is_882_l100_100960

variable (S : ℤ) (N : ℤ) (S_N_correct : N = 1000)

def actual_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 98000) / N

def incorrect_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 980000) / N

theorem mean_difference_is_882 
  (S : ℤ) 
  (N : ℤ) 
  (S_N_correct : N = 1000) 
  (S_in_range : 8200 ≤ S) 
  (S_actual : S + 98000 ≤ 980000) :
  incorrect_mean S N - actual_mean S N = 882 := 
by
  /- Proof steps would go here -/
  sorry

end mean_difference_is_882_l100_100960


namespace derivative_at_one_l100_100800

theorem derivative_at_one (f : ℝ → ℝ) (df : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2) 
  (h₂ : ∀ x, df x = 2 * x) : 
  df 1 = 2 :=
by sorry

end derivative_at_one_l100_100800


namespace termite_ridden_fraction_l100_100251

theorem termite_ridden_fraction:
  ∀ T: ℝ, (3 / 4) * T = 1 / 4 → T = 1 / 3 :=
by
  intro T
  intro h
  sorry

end termite_ridden_fraction_l100_100251


namespace cuberoot_eight_is_512_l100_100997

-- Define the condition on x
def cuberoot_is_eight (x : ℕ) : Prop := 
  x^(1 / 3) = 8

-- The statement to be proved
theorem cuberoot_eight_is_512 : ∃ x : ℕ, cuberoot_is_eight x ∧ x = 512 := 
by 
  -- Proof is omitted
  sorry

end cuberoot_eight_is_512_l100_100997


namespace arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l100_100590

-- Definitions and theorems for the given conditions

-- (1) General formula for the arithmetic sequence
theorem arithmetic_sequence_formula (a S : Nat → Int) (n : Nat) (h1 : a 2 = -1)
  (h2 : S 9 = 5 * S 5) : 
  ∀ n, a n = -8 * n + 15 := 
sorry

-- (2) Minimum value of t - s
theorem min_value_t_minus_s (b : Nat → Rat) (T : Nat → Rat) 
  (h3 : ∀ n, b n = 1 / ((-8 * (n + 1) + 15) * (-8 * (n + 2) + 15))) 
  (h4 : ∀ n, s ≤ T n ∧ T n ≤ t) : 
  t - s = 1 / 72 := 
sorry

-- (3) Maximum value of k
theorem max_value_k (S a : Nat → Int) (k : Rat)
  (h5 : ∀ n, n ≥ 3 → S n / a n ≤ n^2 / (n + k)) :
  k = 80 / 9 := 
sorry

end arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l100_100590


namespace reciprocal_of_neg4_is_neg_one_fourth_l100_100826

theorem reciprocal_of_neg4_is_neg_one_fourth (x : ℝ) (h : x * -4 = 1) : x = -1/4 := 
by 
  sorry

end reciprocal_of_neg4_is_neg_one_fourth_l100_100826


namespace problem1_problem2_problem3_problem4_l100_100012

theorem problem1 : 9 - 5 - (-4) + 2 = 10 := by
  sorry

theorem problem2 : (- (3 / 4) + 7 / 12 - 5 / 9) / (-(1 / 36)) = 26 := by
  sorry

theorem problem3 : -2^4 - ((-5) + 1 / 2) * (4 / 11) + (-2)^3 / (abs (-3^2 + 1)) = -15 := by
  sorry

theorem problem4 : (100 - 1 / 72) * (-36) = -(3600) + (1 / 2) := by
  sorry

end problem1_problem2_problem3_problem4_l100_100012


namespace negation_of_p_l100_100880

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : (¬p) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_p_l100_100880


namespace mike_seashells_l100_100569

theorem mike_seashells (initial total : ℕ) (h1 : initial = 79) (h2 : total = 142) :
    total - initial = 63 :=
by
  sorry

end mike_seashells_l100_100569


namespace result_after_subtraction_l100_100540

theorem result_after_subtraction (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 :=
by
  sorry

end result_after_subtraction_l100_100540


namespace men_build_fountain_l100_100991

theorem men_build_fountain (m1 m2 : ℕ) (l1 l2 d1 d2 : ℕ) (work_rate : ℚ)
  (h1 : m1 * d1 = l1 * work_rate)
  (h2 : work_rate = 56 / (20 * 7))
  (h3 : l1 = 56)
  (h4 : l2 = 42)
  (h5 : m1 = 20)
  (h6 : m2 = 35)
  (h7 : d1 = 7)
  : d2 = 3 :=
sorry

end men_build_fountain_l100_100991


namespace percentage_of_non_technicians_l100_100935

theorem percentage_of_non_technicians (total_workers technicians non_technicians permanent_technicians permanent_non_technicians temporary_workers : ℝ)
  (h1 : technicians = 0.5 * total_workers)
  (h2 : non_technicians = total_workers - technicians)
  (h3 : permanent_technicians = 0.5 * technicians)
  (h4 : permanent_non_technicians = 0.5 * non_technicians)
  (h5 : temporary_workers = 0.5 * total_workers) :
  (non_technicians / total_workers) * 100 = 50 :=
by
  -- Proof is omitted
  sorry

end percentage_of_non_technicians_l100_100935


namespace isosceles_triangle_altitude_l100_100673

open Real

theorem isosceles_triangle_altitude (DE DF DG EG GF EF : ℝ) (h1 : DE = 5) (h2 : DF = 5) (h3 : EG = 2 * GF)
(h4 : DG = sqrt (DE^2 - GF^2)) (h5 : EF = EG + GF) (h6 : EF = 3 * GF) : EF = 5 :=
by
  -- Proof would go here
  sorry

end isosceles_triangle_altitude_l100_100673


namespace newLampTaller_l100_100063

-- Define the heights of the old and new lamps
def oldLampHeight : ℝ := 1
def newLampHeight : ℝ := 2.33

-- Define the proof statement
theorem newLampTaller : newLampHeight - oldLampHeight = 1.33 :=
by
  sorry

end newLampTaller_l100_100063


namespace breadth_of_rectangular_plot_l100_100106

theorem breadth_of_rectangular_plot (b : ℝ) (A : ℝ) (l : ℝ)
  (h1 : A = 20 * b)
  (h2 : l = b + 10)
  (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l100_100106


namespace people_eating_vegetarian_l100_100059

theorem people_eating_vegetarian (only_veg : ℕ) (both_veg_nonveg : ℕ) (total_veg : ℕ) :
  only_veg = 13 ∧ both_veg_nonveg = 6 → total_veg = 19 := 
by
  sorry

end people_eating_vegetarian_l100_100059


namespace triangle_OMN_area_l100_100372

noncomputable def rho (theta : ℝ) : ℝ := 4 * Real.cos theta + 2 * Real.sin theta

theorem triangle_OMN_area :
  let l1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
  let l2 (x y : ℝ) := y = Real.sqrt 3 * x
  let C (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5
  let OM := 2 * Real.sqrt 3 + 1
  let ON := 2 + Real.sqrt 3
  let angle_MON := Real.pi / 6
  let area_OMN := (1 / 2) * OM * ON * Real.sin angle_MON
  (4 * (Real.sqrt 3 + 2) + 5 * Real.sqrt 3 = 8 + 5 * Real.sqrt 3) → 
  area_OMN = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end triangle_OMN_area_l100_100372


namespace part1_l100_100438

variables (a c : ℝ × ℝ)
variables (a_parallel_c : ∃ k : ℝ, c = (k * a.1, k * a.2))
variables (a_value : a = (1,2))
variables (c_magnitude : (c.1 ^ 2 + c.2 ^ 2) = (3 * Real.sqrt 5) ^ 2)

theorem part1: c = (3, 6) ∨ c = (-3, -6) :=
by
  sorry

end part1_l100_100438


namespace eval_expression_at_neg3_l100_100670

def evaluate_expression (x : ℤ) : ℚ :=
  (5 + x * (5 + x) - 4 ^ 2 : ℤ) / (x - 4 + x ^ 3 : ℤ)

theorem eval_expression_at_neg3 :
  evaluate_expression (-3) = -17 / 20 := by
  sorry

end eval_expression_at_neg3_l100_100670


namespace cross_covers_two_rectangles_l100_100350

def Chessboard := Fin 8 × Fin 8

def is_cross (center : Chessboard) (point : Chessboard) : Prop :=
  (point.1 = center.1 ∧ (point.2 = center.2 - 1 ∨ point.2 = center.2 + 1)) ∨
  (point.2 = center.2 ∧ (point.1 = center.1 - 1 ∨ point.1 = center.1 + 1)) ∨
  (point = center)

def Rectangle_1x3 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Alina's rectangles
def Rectangle_1x2 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Polina's rectangles

theorem cross_covers_two_rectangles :
  ∃ center : Chessboard, 
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) :=
sorry

end cross_covers_two_rectangles_l100_100350


namespace max_students_l100_100193

theorem max_students 
  (x : ℕ) 
  (h_lt : x < 100)
  (h_mod8 : x % 8 = 5) 
  (h_mod5 : x % 5 = 3) 
  : x = 93 := 
sorry

end max_students_l100_100193


namespace measure_angle_ABC_l100_100652

theorem measure_angle_ABC (x : ℝ) (h1 : ∃ θ, θ = 180 - x ∧ x / 2 = (180 - x) / 3) : x = 72 :=
by
  sorry

end measure_angle_ABC_l100_100652


namespace total_cost_price_is_584_l100_100276

-- Define the costs of individual items
def cost_watch : ℕ := 144
def cost_bracelet : ℕ := 250
def cost_necklace : ℕ := 190

-- The proof statement: the total cost price is 584
theorem total_cost_price_is_584 : cost_watch + cost_bracelet + cost_necklace = 584 :=
by
  -- We skip the proof steps here, assuming the above definitions are correct.
  sorry

end total_cost_price_is_584_l100_100276


namespace students_taking_either_not_both_l100_100536

theorem students_taking_either_not_both (students_both : ℕ) (students_physics : ℕ) (students_only_chemistry : ℕ) :
  students_both = 12 →
  students_physics = 22 →
  students_only_chemistry = 9 →
  students_physics - students_both + students_only_chemistry = 19 :=
by
  intros h_both h_physics h_chemistry
  rw [h_both, h_physics, h_chemistry]
  repeat { sorry }

end students_taking_either_not_both_l100_100536


namespace blue_balloons_l100_100876

theorem blue_balloons (total_balloons red_balloons green_balloons purple_balloons : ℕ)
  (h1 : total_balloons = 135)
  (h2 : red_balloons = 45)
  (h3 : green_balloons = 27)
  (h4 : purple_balloons = 32) :
  total_balloons - (red_balloons + green_balloons + purple_balloons) = 31 :=
by
  sorry

end blue_balloons_l100_100876


namespace first_class_circular_permutations_second_class_circular_permutations_l100_100081

section CircularPermutations

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def perm_count (a b c : ℕ) : ℕ :=
  factorial (a + b + c) / (factorial a * factorial b * factorial c)

theorem first_class_circular_permutations : perm_count 2 2 4 / 8 = 52 := by
  sorry

theorem second_class_circular_permutations : perm_count 2 2 4 / 2 / 4 = 33 := by
  sorry

end CircularPermutations

end first_class_circular_permutations_second_class_circular_permutations_l100_100081


namespace blue_beads_count_l100_100413

-- Define variables and conditions
variables (r b : ℕ)

-- Define the conditions
def condition1 : Prop := r = 30
def condition2 : Prop := r / 3 = b / 2

-- State the theorem
theorem blue_beads_count (h1 : condition1 r) (h2 : condition2 r b) : b = 20 :=
sorry

end blue_beads_count_l100_100413


namespace car_travel_distance_l100_100011

-- Definitions based on the conditions
def car_speed : ℕ := 60  -- The actual speed of the car
def faster_speed : ℕ := car_speed + 30  -- Speed if the car traveled 30 km/h faster
def time_difference : ℚ := 0.5  -- 30 minutes less in hours

-- The distance D we need to prove
def distance_traveled : ℚ := 90

-- Main statement to be proven
theorem car_travel_distance : ∀ (D : ℚ),
  (D / car_speed) = (D / faster_speed) + time_difference →
  D = distance_traveled :=
by
  intros D h
  sorry

end car_travel_distance_l100_100011


namespace total_ingredients_l100_100998

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l100_100998


namespace water_dispenser_capacity_l100_100344

theorem water_dispenser_capacity :
  ∀ (x : ℝ), (0.25 * x = 60) → x = 240 :=
by
  intros x h
  sorry

end water_dispenser_capacity_l100_100344


namespace cost_of_bench_l100_100396

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l100_100396


namespace distance_3_units_l100_100931

theorem distance_3_units (x : ℤ) (h : |x + 2| = 3) : x = -5 ∨ x = 1 := by
  sorry

end distance_3_units_l100_100931


namespace radius_of_circle_l100_100325

theorem radius_of_circle : 
  ∀ (r : ℝ), 3 * (2 * Real.pi * r) = 2 * Real.pi * r ^ 2 → r = 3 :=
by
  intro r
  intro h
  sorry

end radius_of_circle_l100_100325


namespace magic_square_sum_l100_100653

-- Given conditions
def magic_square (S : ℕ) (a b c d e : ℕ) :=
  (30 + b + 27 = S) ∧
  (30 + 33 + a = S) ∧
  (33 + c + d = S) ∧
  (a + 18 + e = S) ∧
  (30 + c + e = S)

-- Prove that the sum a + d is 38 given the sums of the 3x3 magic square are equivalent
theorem magic_square_sum (a b c d e S : ℕ) (h : magic_square S a b c d e) : a + d = 38 :=
  sorry

end magic_square_sum_l100_100653


namespace maximum_people_shaked_hands_l100_100342

-- Given conditions
variables (N : ℕ) (hN : N > 4)
def has_not_shaken_hands_with (a b : ℕ) : Prop := sorry -- This should define the shaking hand condition

-- Main statement
theorem maximum_people_shaked_hands (h : ∃ i, has_not_shaken_hands_with i 2) :
  ∃ k, k = N - 3 := 
sorry

end maximum_people_shaked_hands_l100_100342


namespace number_of_petri_dishes_l100_100153

noncomputable def total_germs : ℝ := 0.036 * 10^5
noncomputable def germs_per_dish : ℝ := 99.99999999999999

theorem number_of_petri_dishes : 36 = total_germs / germs_per_dish :=
by sorry

end number_of_petri_dishes_l100_100153


namespace smallest_positive_integer_l100_100607

theorem smallest_positive_integer (n : ℕ) (h : 629 * n ≡ 1181 * n [MOD 35]) : n = 35 :=
sorry

end smallest_positive_integer_l100_100607


namespace minimize_J_l100_100365

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J_l100_100365


namespace digits_of_number_l100_100308

theorem digits_of_number (d : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9) (h2 : (10 * (50 + d) + 2) % 6 = 0) : (5 * 10 + d) * 10 + 2 = 522 :=
by sorry

end digits_of_number_l100_100308


namespace opposite_of_neg_2023_l100_100664

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l100_100664


namespace tapA_turned_off_time_l100_100634

noncomputable def tapA_rate := 1 / 45
noncomputable def tapB_rate := 1 / 40
noncomputable def tapB_fill_time := 23

theorem tapA_turned_off_time :
  ∃ t : ℕ, t * (tapA_rate + tapB_rate) + tapB_fill_time * tapB_rate = 1 ∧ t = 9 :=
by
  sorry

end tapA_turned_off_time_l100_100634


namespace largest_square_side_l100_100849

theorem largest_square_side {m n : ℕ} (h1 : m = 72) (h2 : n = 90) : Nat.gcd m n = 18 :=
by
  sorry

end largest_square_side_l100_100849


namespace average_annual_growth_rate_l100_100668

variable (a b : ℝ)

theorem average_annual_growth_rate :
  ∃ x : ℝ, (1 + x)^2 = (1 + a) * (1 + b) ∧ x = Real.sqrt ((1 + a) * (1 + b)) - 1 := by
  sorry

end average_annual_growth_rate_l100_100668


namespace problem_solution_l100_100032

-- Lean 4 statement of the proof problem
theorem problem_solution (m : ℝ) (U : Set ℝ := Univ) (A : Set ℝ := {x | x^2 + 3*x + 2 = 0}) 
  (B : Set ℝ := {x | x^2 + (m + 1)*x + m = 0}) (h : ∀ x, x ∈ (U \ A) → x ∉ B) : 
  m = 1 ∨ m = 2 :=
by 
  -- This is where the proof would normally go
  sorry

end problem_solution_l100_100032


namespace debby_total_photos_l100_100754

theorem debby_total_photos (friends_photos family_photos : ℕ) (h1 : friends_photos = 63) (h2 : family_photos = 23) : friends_photos + family_photos = 86 :=
by sorry

end debby_total_photos_l100_100754


namespace find_m_of_power_fn_and_increasing_l100_100475

theorem find_m_of_power_fn_and_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) > 0) →
  m^2 - m - 5 = 1 →
  1 < m →
  m = 3 :=
sorry

end find_m_of_power_fn_and_increasing_l100_100475


namespace chromosome_stability_due_to_meiosis_and_fertilization_l100_100604

/-- Definition of reducing chromosome number during meiosis -/
def meiosis_reduces_chromosome_number (n : ℕ) : ℕ := n / 2

/-- Definition of restoring chromosome number during fertilization -/
def fertilization_restores_chromosome_number (n : ℕ) : ℕ := n * 2

/-- Axiom: Sexual reproduction involves meiosis and fertilization to maintain chromosome stability -/
axiom chromosome_stability (n m : ℕ) (h1 : meiosis_reduces_chromosome_number n = m) 
  (h2 : fertilization_restores_chromosome_number m = n) : n = n

/-- Theorem statement in Lean 4: The chromosome number stability in sexually reproducing organisms is maintained due to meiosis and fertilization -/
theorem chromosome_stability_due_to_meiosis_and_fertilization 
  (n : ℕ) (h_meiosis: meiosis_reduces_chromosome_number n = n / 2) 
  (h_fertilization: fertilization_restores_chromosome_number (n / 2) = n) : 
  n = n := 
by
  apply chromosome_stability
  exact h_meiosis
  exact h_fertilization

end chromosome_stability_due_to_meiosis_and_fertilization_l100_100604


namespace sin_minus_cos_sqrt_l100_100296

theorem sin_minus_cos_sqrt (θ : ℝ) (b : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : Real.cos (2 * θ) = b) :
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) :=
sorry

end sin_minus_cos_sqrt_l100_100296


namespace paul_packed_total_toys_l100_100400

def small_box_small_toys : ℕ := 8
def medium_box_medium_toys : ℕ := 12
def large_box_large_toys : ℕ := 7
def large_box_small_toys : ℕ := 3
def small_box_medium_toys : ℕ := 5

def small_box : ℕ := small_box_small_toys + small_box_medium_toys
def medium_box : ℕ := medium_box_medium_toys
def large_box : ℕ := large_box_large_toys + large_box_small_toys

def total_toys : ℕ := small_box + medium_box + large_box

theorem paul_packed_total_toys : total_toys = 35 :=
by sorry

end paul_packed_total_toys_l100_100400


namespace train_length_l100_100048

/-- A train crosses a tree in 120 seconds. It takes 230 seconds to pass a platform 1100 meters long.
    How long is the train? -/
theorem train_length (L : ℝ) (V : ℝ)
    (h1 : V = L / 120)
    (h2 : V = (L + 1100) / 230) :
    L = 1200 :=
by
  sorry

end train_length_l100_100048


namespace negation_of_existence_l100_100661

theorem negation_of_existence (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 ≤ x + 2)) ↔ (∀ x : ℝ, x > 0 → x^2 > x + 2) :=
by
  sorry

end negation_of_existence_l100_100661


namespace cost_per_book_l100_100312

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end cost_per_book_l100_100312


namespace abc_div_def_eq_1_div_20_l100_100140

-- Definitions
variables (a b c d e f : ℝ)

-- Conditions
axiom condition1 : a / b = 1 / 3
axiom condition2 : b / c = 2
axiom condition3 : c / d = 1 / 2
axiom condition4 : d / e = 3
axiom condition5 : e / f = 1 / 10

-- Proof statement
theorem abc_div_def_eq_1_div_20 : (a * b * c) / (d * e * f) = 1 / 20 :=
by 
  -- The actual proof is omitted, as the problem only requires the statement.
  sorry

end abc_div_def_eq_1_div_20_l100_100140


namespace sqrt_expression_simplify_l100_100391

theorem sqrt_expression_simplify : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 :=
by 
  sorry

end sqrt_expression_simplify_l100_100391


namespace largest_int_starting_with_8_l100_100234

theorem largest_int_starting_with_8 (n : ℕ) : 
  (n / 100 = 8) ∧ (n >= 800) ∧ (n < 900) ∧ ∀ (d : ℕ), (d ∣ n ∧ d ≠ 0 ∧ d ≠ 7) → d ∣ 864 → (n ≤ 864) :=
sorry

end largest_int_starting_with_8_l100_100234


namespace calculate_product_l100_100588

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l100_100588


namespace relationship_between_P_and_Q_l100_100186

-- Define the sets P and Q
def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem relationship_between_P_and_Q : P ⊇ Q :=
by
  sorry

end relationship_between_P_and_Q_l100_100186


namespace find_factor_l100_100724

-- Define the conditions
def number : ℕ := 9
def expr1 (f : ℝ) : ℝ := (number + 2) * f
def expr2 : ℝ := 24 + number

-- The proof problem statement
theorem find_factor (f : ℝ) : expr1 f = expr2 → f = 3 := by
  sorry

end find_factor_l100_100724


namespace num_entrees_ordered_l100_100218

-- Define the conditions
def appetizer_cost: ℝ := 10
def entree_cost: ℝ := 20
def tip_rate: ℝ := 0.20
def total_spent: ℝ := 108

-- Define the theorem to prove the number of entrees ordered
theorem num_entrees_ordered : ∃ E : ℝ, (entree_cost * E) + appetizer_cost + (tip_rate * ((entree_cost * E) + appetizer_cost)) = total_spent ∧ E = 4 := 
by
  sorry

end num_entrees_ordered_l100_100218


namespace cylindrical_to_rectangular_l100_100351

theorem cylindrical_to_rectangular (r θ z : ℝ) 
  (h₁ : r = 7) (h₂ : θ = 5 * Real.pi / 4) (h₃ : z = 6) : 
  (r * Real.cos θ, r * Real.sin θ, z) = 
  (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, 6) := 
by 
  sorry

end cylindrical_to_rectangular_l100_100351


namespace colored_line_midpoint_l100_100873

theorem colored_line_midpoint (L : ℝ → Prop) (p1 p2 : ℝ) :
  (L p1 → L p2) →
  (∃ A B C : ℝ, L A = L B ∧ L B = L C ∧ 2 * B = A + C ∧ L A = L C) :=
sorry

end colored_line_midpoint_l100_100873


namespace length_of_first_square_flag_l100_100672

theorem length_of_first_square_flag
  (x : ℝ)
  (h1x : x * 5 + 10 * 7 + 5 * 5 = 15 * 9) : 
  x = 8 :=
by
  sorry

end length_of_first_square_flag_l100_100672


namespace registration_methods_count_l100_100078

theorem registration_methods_count (students : Fin 4) (groups : Fin 3) : (3 : ℕ)^4 = 81 :=
by
  sorry

end registration_methods_count_l100_100078


namespace line_plane_intersection_l100_100322

theorem line_plane_intersection :
  ∃ (x y z : ℝ), (∃ t : ℝ, x = -3 + 2 * t ∧ y = 1 + 3 * t ∧ z = 1 + 5 * t) ∧ (2 * x + 3 * y + 7 * z - 52 = 0) ∧ (x = -1) ∧ (y = 4) ∧ (z = 6) :=
sorry

end line_plane_intersection_l100_100322


namespace parabola_directrix_l100_100979

theorem parabola_directrix (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x ↔ x = -1 → p = 2) :=
by
  sorry

end parabola_directrix_l100_100979


namespace transform_polynomial_l100_100229

variable (x z : ℝ)

theorem transform_polynomial (h1 : z = x - 1 / x) (h2 : x^4 - 3 * x^3 - 2 * x^2 + 3 * x + 1 = 0) :
  x^2 * (z^2 - 3 * z) = 0 :=
sorry

end transform_polynomial_l100_100229


namespace count_triangles_on_cube_count_triangles_not_in_face_l100_100217

open Nat

def num_triangles_cube : ℕ := 56
def num_triangles_not_in_face : ℕ := 32

theorem count_triangles_on_cube (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 = num_triangles_cube) :=
  sorry

theorem count_triangles_not_in_face (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 - (6 * 4) = num_triangles_not_in_face) :=
  sorry

end count_triangles_on_cube_count_triangles_not_in_face_l100_100217


namespace tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l100_100183

noncomputable def f (x : ℝ) := x^3 - x + 3
noncomputable def df (x : ℝ) := 3 * x^2 - 1

theorem tangent_line_equation_at_x_1 : 
  let k := df 1
  let y := f 1
  (2 = k) ∧ (y = 3) ∧ ∀ x y, y - 3 = 2 * (x - 1) ↔ 2 * x - y + 1 = 0 := 
by 
  sorry

theorem intervals_of_monotonic_increase : 
  let x1 := - (Real.sqrt 3) / 3
  let x2 := (Real.sqrt 3) / 3
  ∀ x, (df x > 0 ↔ (x < x1) ∨ (x > x2)) ∧ 
       (df x < 0 ↔ (x1 < x ∧ x < x2)) := 
by 
  sorry

end tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l100_100183


namespace proof_of_k_bound_l100_100817

noncomputable def sets_with_nonempty_intersection_implies_k_bound (k : ℝ) : Prop :=
  let M := {x : ℝ | -1 ≤ x ∧ x < 2}
  let N := {x : ℝ | x ≤ k + 3}
  M ∩ N ≠ ∅ → k ≥ -4

theorem proof_of_k_bound (k : ℝ) : sets_with_nonempty_intersection_implies_k_bound k := by
  intro h
  have : -1 ≤ k + 3 := sorry
  linarith

end proof_of_k_bound_l100_100817


namespace treble_of_doubled_and_increased_l100_100460

theorem treble_of_doubled_and_increased (initial_number : ℕ) (result : ℕ) : 
  initial_number = 15 → (initial_number * 2 + 5) * 3 = result → result = 105 := 
by 
  intros h1 h2
  rw [h1] at h2
  linarith

end treble_of_doubled_and_increased_l100_100460


namespace distance_between_stations_l100_100996

/-- Two trains start at the same time from two stations and proceed towards each other. 
    The first train travels at 20 km/hr and the second train travels at 25 km/hr. 
    When they meet, the second train has traveled 60 km more than the first train. -/
theorem distance_between_stations
    (t : ℝ) -- The time in hours when they meet
    (x : ℝ) -- The distance traveled by the slower train
    (d1 d2 : ℝ) -- Distances traveled by the two trains respectively
    (h1 : 20 * t = x)
    (h2 : 25 * t = x + 60) :
  d1 + d2 = 540 :=
by
  sorry

end distance_between_stations_l100_100996


namespace negation_of_existence_l100_100968

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by
  sorry

end negation_of_existence_l100_100968


namespace m_plus_n_is_23_l100_100910

noncomputable def find_m_plus_n : ℕ := 
  let A := 12
  let B := 4
  let C := 3
  let D := 3

  -- Declare the radius of E
  let radius_E : ℚ := (21 / 2)
  
  -- Let radius_E be written as m / n where m and n are relatively prime
  let (m : ℕ) := 21
  let (n : ℕ) := 2

  -- Calculate m + n
  m + n

theorem m_plus_n_is_23 : find_m_plus_n = 23 :=
by
  -- Proof is omitted
  sorry

end m_plus_n_is_23_l100_100910


namespace semicircle_radius_l100_100900

noncomputable def radius_of_inscribed_semicircle (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21) : ℝ :=
  let AB := Real.sqrt (21^2 + 10^2)
  let s := 2 * Real.sqrt 541
  let area := 20 * 21
  (area) / (s * 2)

theorem semicircle_radius (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21)
  : radius_of_inscribed_semicircle BD height h_base h_height = 210 / Real.sqrt 541 :=
sorry

end semicircle_radius_l100_100900


namespace find_m_if_z_is_pure_imaginary_l100_100514

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_m_if_z_is_pure_imaginary (m : ℝ) (z : ℂ) (i : ℂ) (h_i_unit : i^2 = -1) (h_z : z = (1 + i) / (1 - i) + m * (1 - i)) :
  is_pure_imaginary z → m = 0 := 
by
  sorry

end find_m_if_z_is_pure_imaginary_l100_100514


namespace average_marks_l100_100375

theorem average_marks (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 10) : (M + C) / 2 = 35 := 
by
  sorry

end average_marks_l100_100375


namespace area_of_centroid_path_l100_100747

theorem area_of_centroid_path (A B C O G : ℝ) (r : ℝ) (h1 : A ≠ B) 
  (h2 : 2 * r = 30) (h3 : ∀ C, C ≠ A ∧ C ≠ B ∧ dist O C = r) 
  (h4 : dist O G = r / 3) : 
  (π * (r / 3)^2 = 25 * π) :=
by 
  -- def AB := 2 * r -- given AB is a diameter of the circle
  -- def O := (A + B) / 2 -- center of the circle
  -- def G := (A + B + C) / 3 -- centroid of triangle ABC
  sorry

end area_of_centroid_path_l100_100747


namespace expression_meaningful_l100_100828

theorem expression_meaningful (x : ℝ) : (∃ y, y = 4 / (x - 5)) ↔ x ≠ 5 :=
by
  sorry

end expression_meaningful_l100_100828


namespace q_value_l100_100922

-- Define the conditions and the problem statement
theorem q_value (a b m p q : ℚ) (h1 : a * b = 3) 
  (h2 : (a + 1 / b) * (b + 1 / a) = q) : 
  q = 16 / 3 :=
by
  sorry

end q_value_l100_100922


namespace solve_for_x_l100_100310

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end solve_for_x_l100_100310


namespace repeating_decimal_arithmetic_l100_100830

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l100_100830


namespace simplify_expr1_simplify_expr2_l100_100507

-- Problem 1
theorem simplify_expr1 (x y : ℝ) : x^2 - 5 * y - 4 * x^2 + y - 1 = -3 * x^2 - 4 * y - 1 :=
by sorry

-- Problem 2
theorem simplify_expr2 (a b : ℝ) : 7 * a + 3 * (a - 3 * b) - 2 * (b - 3 * a) = 16 * a - 11 * b :=
by sorry

end simplify_expr1_simplify_expr2_l100_100507


namespace simplify_expansion_l100_100246

theorem simplify_expansion (x : ℝ) : 
  (3 * x - 6) * (x + 8) - (x + 6) * (3 * x + 2) = -2 * x - 60 :=
by
  sorry

end simplify_expansion_l100_100246


namespace sixty_percent_of_total_is_960_l100_100666

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end sixty_percent_of_total_is_960_l100_100666


namespace polygon_sides_from_diagonals_l100_100454

theorem polygon_sides_from_diagonals (D : ℕ) (hD : D = 16) : 
  ∃ n : ℕ, 2 * D = n * (n - 3) ∧ n = 7 :=
by
  use 7
  simp [hD]
  norm_num
  sorry

end polygon_sides_from_diagonals_l100_100454


namespace value_of_x_l100_100641

theorem value_of_x 
  (x : ℚ) 
  (h₁ : 6 * x^2 + 19 * x - 7 = 0) 
  (h₂ : 18 * x^2 + 47 * x - 21 = 0) : 
  x = 1 / 3 := 
  sorry

end value_of_x_l100_100641


namespace A_can_finish_remaining_work_in_6_days_l100_100907

-- Condition: A can finish the work in 18 days
def A_work_rate := 1 / 18

-- Condition: B can finish the work in 15 days
def B_work_rate := 1 / 15

-- Given B worked for 10 days
def B_days_worked := 10

-- Calculation of the remaining work
def remaining_work := 1 - B_days_worked * B_work_rate

-- Calculation of the time for A to finish the remaining work
def A_remaining_days := remaining_work / A_work_rate

-- The theorem to prove
theorem A_can_finish_remaining_work_in_6_days : A_remaining_days = 6 := 
by 
  -- The proof is not required, so we use sorry to skip it.
  sorry

end A_can_finish_remaining_work_in_6_days_l100_100907


namespace masha_can_climb_10_steps_l100_100821

def ways_to_climb_stairs : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => ways_to_climb_stairs (n + 1) + ways_to_climb_stairs n

theorem masha_can_climb_10_steps : ways_to_climb_stairs 10 = 89 :=
by
  -- proof omitted here as per instruction
  sorry

end masha_can_climb_10_steps_l100_100821


namespace projection_ratio_zero_l100_100439

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end projection_ratio_zero_l100_100439


namespace f_800_l100_100043

-- Definitions of hypothesis from conditions given
def f : ℕ → ℤ := sorry
axiom f_mul (x y : ℕ) : f (x * y) = f x + f y
axiom f_10 : f 10 = 10
axiom f_40 : f 40 = 18

-- Proof problem statement: prove that f(800) = 32
theorem f_800 : f 800 = 32 := 
by
  sorry

end f_800_l100_100043


namespace equation_represents_pair_of_lines_l100_100051

theorem equation_represents_pair_of_lines : ∀ x y : ℝ, 9 * x^2 - 25 * y^2 = 0 → 
                    (x = (5/3) * y ∨ x = -(5/3) * y) :=
by sorry

end equation_represents_pair_of_lines_l100_100051


namespace find_numbers_l100_100253

theorem find_numbers (x y z : ℝ) 
  (h1 : x = 280)
  (h2 : y = 200)
  (h3 : z = 220) :
  (x = 1.4 * y) ∧
  (x / z = 14 / 11) ∧
  (z - y = 0.125 * (x + y) - 40) :=
by
  sorry

end find_numbers_l100_100253


namespace compute_c_plus_d_l100_100101

-- Define the conditions
variables (c d : ℕ) 

-- Conditions:
-- Positive integers
axiom pos_c : 0 < c
axiom pos_d : 0 < d

-- Contains 630 terms
axiom term_count : d - c = 630

-- The product of the logarithms equals 2
axiom log_product : (Real.log d) / (Real.log c) = 2

-- Theorem to prove
theorem compute_c_plus_d : c + d = 1260 :=
sorry

end compute_c_plus_d_l100_100101


namespace increasing_function_range_l100_100957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x y : ℝ, x < y → f a x < f a y) : 
  3 / 2 ≤ a ∧ a < 2 := by
  sorry

end increasing_function_range_l100_100957


namespace num_triangles_from_decagon_l100_100202

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l100_100202


namespace values_of_a2_add_b2_l100_100708

theorem values_of_a2_add_b2 (a b : ℝ) (h1 : a^3 - 3 * a * b^2 = 11) (h2 : b^3 - 3 * a^2 * b = 2) : a^2 + b^2 = 5 := 
by
  sorry

end values_of_a2_add_b2_l100_100708


namespace cycle_time_to_library_l100_100061

theorem cycle_time_to_library 
  (constant_speed : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_speed)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  (18 : ℕ) = (30 * distance_to_library / distance_to_park) :=
by
  intros
  -- The proof would go here
  sorry

end cycle_time_to_library_l100_100061


namespace binary_to_decimal_l100_100377

/-- The binary number 1011 (base 2) equals 11 (base 10). -/
theorem binary_to_decimal : (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 11 := by
  sorry

end binary_to_decimal_l100_100377


namespace infinitely_many_n_prime_l100_100401

theorem infinitely_many_n_prime (p : ℕ) [Fact (Nat.Prime p)] : ∃ᶠ n in at_top, p ∣ 2^n - n := 
sorry

end infinitely_many_n_prime_l100_100401


namespace candy_difference_l100_100904

def given_away : ℕ := 6
def left : ℕ := 5
def difference : ℕ := given_away - left

theorem candy_difference :
  difference = 1 :=
by
  sorry

end candy_difference_l100_100904


namespace proof_a_eq_b_pow_n_l100_100600

theorem proof_a_eq_b_pow_n 
  (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := 
by 
  sorry

end proof_a_eq_b_pow_n_l100_100600


namespace cindy_hit_section_8_l100_100284

inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen
deriving DecidableEq

structure DartContest :=
(player : Player)
(score : ℕ)

def ContestConditions (dc : DartContest) : Prop :=
  match dc with
  | ⟨Player.Alice, 10⟩ => True
  | ⟨Player.Ben, 6⟩ => True
  | ⟨Player.Cindy, 9⟩ => True
  | ⟨Player.Dave, 15⟩ => True
  | ⟨Player.Ellen, 19⟩ => True
  | _ => False

def isScoreSection8 (dc : DartContest) : Prop :=
  dc.player = Player.Cindy ∧ dc.score = 8

theorem cindy_hit_section_8 
  (cond : ∀ (dc : DartContest), ContestConditions dc) : 
  ∃ (dc : DartContest), isScoreSection8 dc := by
  sorry

end cindy_hit_section_8_l100_100284


namespace division_reciprocal_multiplication_l100_100441

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l100_100441


namespace sequence_a_n_correctness_l100_100914

theorem sequence_a_n_correctness (a : ℕ → ℚ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = 2 * a n + 1) : a 2 = 1.5 := by
  sorry

end sequence_a_n_correctness_l100_100914


namespace sandy_correct_sums_l100_100948

theorem sandy_correct_sums (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x - 2 * y = 50) : x = 22 :=
  by
  sorry

end sandy_correct_sums_l100_100948


namespace find_m_l100_100270

theorem find_m 
  (m : ℕ) 
  (hm_pos : 0 < m) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := 
sorry

end find_m_l100_100270


namespace interval_of_increase_of_f_l100_100621

noncomputable def f (x : ℝ) := Real.logb (0.5) (x - x^2)

theorem interval_of_increase_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → ∃ ε > 0, ∀ y : ℝ, y ∈ Set.Ioo (x - ε) (x + ε) → f y > f x :=
  by
    sorry

end interval_of_increase_of_f_l100_100621


namespace union_when_a_eq_2_condition_1_condition_2_condition_3_l100_100094

open Set

def setA (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_when_a_eq_2 : setA 2 ∪ setB = {x | -1 ≤ x ∧ x ≤ 3} :=
sorry

theorem condition_1 (a : ℝ) : 
  (setA a ∪ setB = setB) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_2 (a : ℝ) :
  (∀ x, (x ∈ setA a ↔ x ∈ setB)) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_3 (a : ℝ) :
  (setA a ∩ setB = ∅) → (a < -2 ∨ 4 < a) :=
sorry

end union_when_a_eq_2_condition_1_condition_2_condition_3_l100_100094


namespace total_miles_l100_100483

theorem total_miles (miles_Katarina miles_Harriet miles_Tomas miles_Tyler : ℕ)
  (hK : miles_Katarina = 51)
  (hH : miles_Harriet = 48)
  (hT : miles_Tomas = 48)
  (hTy : miles_Tyler = 48) :
  miles_Katarina + miles_Harriet + miles_Tomas + miles_Tyler = 195 :=
  by
    sorry

end total_miles_l100_100483


namespace probability_of_finding_transmitter_l100_100146

def total_license_plates : ℕ := 900
def inspected_vehicles : ℕ := 18

theorem probability_of_finding_transmitter : (inspected_vehicles : ℝ) / (total_license_plates : ℝ) = 0.02 :=
by
  sorry

end probability_of_finding_transmitter_l100_100146


namespace base_conversion_subtraction_l100_100015

theorem base_conversion_subtraction :
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  n1 - n2 = 7422 :=
by
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  show n1 - n2 = 7422
  sorry

end base_conversion_subtraction_l100_100015


namespace cos_seven_pi_over_six_l100_100448

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l100_100448


namespace system_unique_solution_l100_100001

theorem system_unique_solution 
  (x y z : ℝ) 
  (h1 : x + y + z = 3 * x * y) 
  (h2 : x^2 + y^2 + z^2 = 3 * x * z) 
  (h3 : x^3 + y^3 + z^3 = 3 * y * z) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) 
  (hz : 0 ≤ z) : 
  (x = 1 ∧ y = 1 ∧ z = 1) := 
sorry

end system_unique_solution_l100_100001


namespace triangle_side_lengths_l100_100833

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ (a^2 + b^2 + c^2 = 2050) ∧ (c^2 = a^2 + b^2)

theorem triangle_side_lengths :
  ∃ b c : ℝ, side_lengths 10 b c ∧ b = Real.sqrt 925 ∧ c = Real.sqrt 1025 :=
by
  sorry

end triangle_side_lengths_l100_100833


namespace milk_tea_sales_l100_100122

-- Definitions
def relationship (x y : ℕ) : Prop := y = 10 * x + 2

-- Theorem statement
theorem milk_tea_sales (x y : ℕ) :
  relationship x y → (y = 822 → x = 82) :=
by
  intros h_rel h_y
  sorry

end milk_tea_sales_l100_100122


namespace unique_solution_l100_100320

theorem unique_solution (k n : ℕ) (hk : k > 0) (hn : n > 0) (h : (7^k - 3^n) ∣ (k^4 + n^2)) : (k = 2 ∧ n = 4) :=
by
  sorry

end unique_solution_l100_100320


namespace units_digit_product_l100_100428

theorem units_digit_product :
  let nums : List Nat := [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
  let product := nums.prod
  (product % 10) = 9 :=
by
  sorry

end units_digit_product_l100_100428


namespace mom_chicken_cost_l100_100304

def cost_bananas : ℝ := 2 * 4 -- bananas cost
def cost_pears : ℝ := 2 -- pears cost
def cost_asparagus : ℝ := 6 -- asparagus cost
def total_expenses_other_than_chicken : ℝ := cost_bananas + cost_pears + cost_asparagus -- total cost of other items
def initial_money : ℝ := 55 -- initial amount of money
def remaining_money_after_other_purchases : ℝ := initial_money - total_expenses_other_than_chicken -- money left after covering other items

theorem mom_chicken_cost : 
  (remaining_money_after_other_purchases - 28 = 11) := 
by
  sorry

end mom_chicken_cost_l100_100304


namespace sum_of_infinite_perimeters_l100_100940

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_infinite_perimeters (a : ℝ) :
  let first_perimeter := 3 * a
  let common_ratio := (1/3 : ℝ)
  let S := geometric_series_sum first_perimeter common_ratio 0
  S = (9 * a / 2) :=
by
  sorry

end sum_of_infinite_perimeters_l100_100940


namespace p_iff_q_l100_100918

theorem p_iff_q (a b : ℝ) : (a > b) ↔ (a^3 > b^3) :=
sorry

end p_iff_q_l100_100918


namespace denominator_expression_l100_100894

theorem denominator_expression (x y a b E : ℝ)
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / E = 3)
  (h3 : a / b = 4.5) : E = 3 * b - y :=
sorry

end denominator_expression_l100_100894


namespace division_remainder_l100_100292

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l100_100292


namespace solve_for_a_b_l100_100034

open Complex

theorem solve_for_a_b (a b : ℝ) (h : (mk 1 2) / (mk a b) = mk 1 1) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
sorry

end solve_for_a_b_l100_100034


namespace alpha_half_in_II_IV_l100_100781

theorem alpha_half_in_II_IV (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) : 
  (k * π - π / 4 < (α / 2) ∧ (α / 2) < k * π) :=
by
  sorry

end alpha_half_in_II_IV_l100_100781


namespace history_book_cost_l100_100567

def total_books : ℕ := 90
def cost_math_book : ℕ := 4
def total_price : ℕ := 397
def math_books_bought : ℕ := 53

theorem history_book_cost :
  ∃ (H : ℕ), H = (total_price - (math_books_bought * cost_math_book)) / (total_books - math_books_bought) ∧ H = 5 :=
by
  sorry

end history_book_cost_l100_100567


namespace find_n_l100_100274

theorem find_n (n : ℕ) (hn : n * n! - n! = 5040 - n!) : n = 7 :=
by
  sorry

end find_n_l100_100274


namespace arithmetic_mean_solution_l100_100463

-- Define the Arithmetic Mean statement
theorem arithmetic_mean_solution (x : ℝ) (h : (x + 5 + 17 + 3 * x + 11 + 3 * x + 6) / 5 = 19) : 
  x = 8 :=
by
  sorry -- Proof is not required as per the instructions

end arithmetic_mean_solution_l100_100463


namespace intersection_line_constant_l100_100879

-- Definitions based on conditions provided:
def circle1_eq (x y : ℝ) : Prop := (x + 6)^2 + (y - 2)^2 = 144
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 9)^2 = 65

-- The theorem statement
theorem intersection_line_constant (c : ℝ) : 
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧ x + y = c) ↔ c = 6 :=
by
  sorry

end intersection_line_constant_l100_100879


namespace gh_of_2_l100_100989

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem gh_of_2 :
  g (h 2) = 3269 :=
by
  sorry

end gh_of_2_l100_100989


namespace line_passes_through_first_and_fourth_quadrants_l100_100517

theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) (H : b * k < 0) :
  (∃x₁, k * x₁ + b > 0) ∧ (∃x₂, k * x₂ + b < 0) :=
by
  sorry

end line_passes_through_first_and_fourth_quadrants_l100_100517


namespace george_hourly_rate_l100_100095

theorem george_hourly_rate (total_hours : ℕ) (total_amount : ℕ) (h1 : total_hours = 7 + 2)
  (h2 : total_amount = 45) : 
  total_amount / total_hours = 5 := 
by sorry

end george_hourly_rate_l100_100095


namespace ordered_pair_sqrt_l100_100476

/-- Problem statement: Given positive integers a and b such that a < b, prove that:
sqrt (1 + sqrt (40 + 24 * sqrt 5)) = sqrt a + sqrt b, if (a, b) = (1, 6). -/
theorem ordered_pair_sqrt (a b : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : a < b) :
  Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b :=
by
  sorry -- The proof is not required in this task.

end ordered_pair_sqrt_l100_100476


namespace fish_left_in_tank_l100_100886

-- Define the initial number of fish and the number of fish moved
def initialFish : Real := 212.0
def movedFish : Real := 68.0

-- Define the number of fish left in the tank
def fishLeft (initialFish : Real) (movedFish : Real) : Real := initialFish - movedFish

-- Theorem stating the problem
theorem fish_left_in_tank : fishLeft initialFish movedFish = 144.0 := by
  sorry

end fish_left_in_tank_l100_100886


namespace reflection_problem_l100_100674

theorem reflection_problem 
  (m b : ℝ)
  (h : ∀ (P Q : ℝ × ℝ), 
        P = (2,2) ∧ Q = (8,4) → 
        ∃ mid : ℝ × ℝ, 
        mid = ((P.fst + Q.fst) / 2, (P.snd + Q.snd) / 2) ∧ 
        ∃ m' : ℝ, m' ≠ 0 ∧ P.snd - m' * P.fst = Q.snd - m' * Q.fst) :
  m + b = 15 := 
sorry

end reflection_problem_l100_100674


namespace tan_diff_identity_l100_100558

theorem tan_diff_identity (α : ℝ) (hα : 0 < α ∧ α < π) (h : Real.sin α = 4 / 5) :
  Real.tan (π / 4 - α) = -1 / 7 ∨ Real.tan (π / 4 - α) = -7 :=
sorry

end tan_diff_identity_l100_100558


namespace sum_of_four_smallest_divisors_eq_11_l100_100610

noncomputable def common_divisors_sum : ℤ :=
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  let smallest_four := common_divisors.take 4
  smallest_four.sum

theorem sum_of_four_smallest_divisors_eq_11 :
  common_divisors_sum = 11 := by
  sorry

end sum_of_four_smallest_divisors_eq_11_l100_100610


namespace integer_a_values_l100_100132

theorem integer_a_values (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x - 7 = 0) ↔ a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3 :=
by
  sorry

end integer_a_values_l100_100132


namespace isosceles_triangle_perimeter_l100_100269

theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_eqn : ∀ x : ℕ, (x - 4) * (x - 2) = 0 → x = 4 ∨ x = 2)
  (h_isosceles : ∃ a b : ℕ, (a = 4 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 4)) :
  a + a + b = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l100_100269


namespace fractional_part_wall_in_12_minutes_l100_100299

-- Definitions based on given conditions
def time_to_paint_wall : ℕ := 60
def time_spent_painting : ℕ := 12

-- The goal is to prove that the fraction of the wall Mark can paint in 12 minutes is 1/5
theorem fractional_part_wall_in_12_minutes (t_pw: ℕ) (t_sp: ℕ) (h1: t_pw = 60) (h2: t_sp = 12) : 
  (t_sp : ℚ) / (t_pw : ℚ) = 1 / 5 :=
by 
  sorry

end fractional_part_wall_in_12_minutes_l100_100299


namespace prism_faces_l100_100144

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l100_100144


namespace fish_population_estimate_l100_100549

theorem fish_population_estimate
  (N : ℕ) 
  (tagged_initial : ℕ)
  (caught_again : ℕ)
  (tagged_again : ℕ)
  (h1 : tagged_initial = 60)
  (h2 : caught_again = 60)
  (h3 : tagged_again = 2)
  (h4 : (tagged_initial : ℚ) / N = (tagged_again : ℚ) / caught_again) :
  N = 1800 :=
by
  sorry

end fish_population_estimate_l100_100549


namespace number_not_equal_54_l100_100392

def initial_number : ℕ := 12
def target_number : ℕ := 54
def total_time : ℕ := 60

theorem number_not_equal_54 (n : ℕ) (time : ℕ) : (time = total_time) → (n = initial_number) → 
  (∀ t : ℕ, t ≤ time → (n = n * 2 ∨ n = n / 2 ∨ n = n * 3 ∨ n = n / 3)) → n ≠ target_number :=
by
  sorry

end number_not_equal_54_l100_100392


namespace problem_b_value_l100_100069

theorem problem_b_value (b : ℤ)
  (h1 : 0 ≤ b)
  (h2 : b ≤ 20)
  (h3 : (3 - b) % 17 = 0) : b = 3 :=
sorry

end problem_b_value_l100_100069


namespace find_rate_of_current_l100_100249

-- Given speed of the boat in still water (km/hr)
def boat_speed : ℤ := 20

-- Given time of travel downstream (hours)
def time_downstream : ℚ := 24 / 60

-- Given distance travelled downstream (km)
def distance_downstream : ℤ := 10

-- To find: rate of the current (km/hr)
theorem find_rate_of_current (c : ℚ) 
  (h1 : distance_downstream = (boat_speed + c) * time_downstream) : 
  c = 5 := 
by sorry

end find_rate_of_current_l100_100249


namespace Shara_savings_l100_100903

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l100_100903


namespace alcohol_quantity_in_mixture_l100_100314

theorem alcohol_quantity_in_mixture : 
  ∃ (A W : ℕ), (A = 8) ∧ (A * 3 = 4 * W) ∧ (A * 5 = 4 * (W + 4)) :=
by
  sorry -- This is a placeholder; the proof itself is not required.

end alcohol_quantity_in_mixture_l100_100314


namespace circle_center_l100_100295

theorem circle_center (x y : ℝ) : 
    (∃ x y : ℝ, x^2 - 8*x + y^2 - 4*y = 16) → (x, y) = (4, 2) := by
  sorry

end circle_center_l100_100295


namespace coefficient_of_x9_in_polynomial_is_240_l100_100561

-- Define the polynomial (1 + 3x - 2x^2)^5
noncomputable def polynomial : ℕ → ℝ := (fun x => (1 + 3*x - 2*x^2)^5)

-- Define the term we are interested in (x^9)
def term := 9

-- The coefficient we want to prove
def coefficient := 240

-- The goal is to prove that the coefficient of x^9 in the expansion of (1 + 3x - 2x^2)^5 is 240
theorem coefficient_of_x9_in_polynomial_is_240 : polynomial 9 = coefficient := sorry

end coefficient_of_x9_in_polynomial_is_240_l100_100561


namespace find_f4_l100_100701

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (h1 : f 1 = 5)
variable (h2 : f 2 = 8)
variable (h3 : f 3 = 11)
variable (h4 : ∀ x, f x = a * x + b)

theorem find_f4 : f 4 = 14 := by
  sorry

end find_f4_l100_100701


namespace trigonometric_identity_l100_100544

theorem trigonometric_identity
  (α : Real)
  (hcos : Real.cos α = -4/5)
  (hquad : π/2 < α ∧ α < π) :
  (-Real.sin (2 * α) / Real.cos α) = -6/5 := 
by
  sorry

end trigonometric_identity_l100_100544


namespace cannot_bisect_segment_with_ruler_l100_100779

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler_l100_100779


namespace geom_sequence_product_l100_100538

theorem geom_sequence_product (q a1 : ℝ) (h1 : a1 * (a1 * q) * (a1 * q^2) = 3) (h2 : (a1 * q^9) * (a1 * q^10) * (a1 * q^11) = 24) :
  (a1 * q^12) * (a1 * q^13) * (a1 * q^14) = 48 :=
by
  sorry

end geom_sequence_product_l100_100538


namespace average_income_l100_100693

-- Lean statement to express the given mathematical problem
theorem average_income (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : A = 3000) :
  (A + C) / 2 = 4200 :=
by
  sorry

end average_income_l100_100693


namespace numberOfTermsArithmeticSequence_l100_100630

theorem numberOfTermsArithmeticSequence (a1 d l : ℕ) (h1 : a1 = 3) (h2 : d = 4) (h3 : l = 2012) :
  ∃ n : ℕ, 3 + (n - 1) * 4 ≤ 2012 ∧ (n : ℕ) = 502 :=
by {
  sorry
}

end numberOfTermsArithmeticSequence_l100_100630


namespace measure_of_angle_F_l100_100803

theorem measure_of_angle_F (D E F : ℝ) (hD : D = E) 
  (hF : F = D + 40) (h_sum : D + E + F = 180) : F = 140 / 3 + 40 :=
by
  sorry

end measure_of_angle_F_l100_100803


namespace minimum_value_inequality_l100_100729

theorem minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → (2 / x + 1 / y) ≥ 9)) :=
by
  -- skipping the proof
  sorry

end minimum_value_inequality_l100_100729


namespace children_l100_100200

theorem children's_book_pages (P : ℝ)
  (h1 : P > 0)
  (c1 : ∃ P_rem, P_rem = P - (0.2 * P))
  (c2 : ∃ P_today, P_today = (0.35 * (P - (0.2 * P))))
  (c3 : ∃ Pages_left, Pages_left = (P - (0.2 * P) - (0.35 * (P - (0.2 * P)))) ∧ Pages_left = 130) :
  P = 250 := by
  sorry

end children_l100_100200


namespace order_of_logs_l100_100045

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem order_of_logs (a_def : a = Real.log 6 / Real.log 3)
                      (b_def : b = Real.log 10 / Real.log 5)
                      (c_def : c = Real.log 14 / Real.log 7) : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l100_100045


namespace coefficient_of_m5n4_in_expansion_l100_100933

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_of_m5n4_in_expansion : binomial_coefficient 9 5 = 126 := by
  sorry

end coefficient_of_m5n4_in_expansion_l100_100933


namespace fortieth_sequence_number_l100_100971

theorem fortieth_sequence_number :
  (∃ r n : ℕ, ((r * (r + 1)) - 40 = n) ∧ (40 ≤ r * (r + 1)) ∧ (40 > (r - 1) * r) ∧ n = 2 * r) :=
sorry

end fortieth_sequence_number_l100_100971


namespace percentage_x_equals_y_l100_100614

theorem percentage_x_equals_y (x y z : ℝ) (p : ℝ)
    (h1 : 0.45 * z = 0.39 * y)
    (h2 : z = 0.65 * x)
    (h3 : y = (p / 100) * x) : 
    p = 75 := 
sorry

end percentage_x_equals_y_l100_100614


namespace land_profit_each_son_l100_100241

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end land_profit_each_son_l100_100241


namespace umbrella_cost_l100_100424

theorem umbrella_cost (number_of_umbrellas : Nat) (total_cost : Nat) (h1 : number_of_umbrellas = 3) (h2 : total_cost = 24) :
  (total_cost / number_of_umbrellas) = 8 :=
by
  -- The proof will go here
  sorry

end umbrella_cost_l100_100424


namespace g_is_odd_l100_100426

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end g_is_odd_l100_100426


namespace prime_k_values_l100_100848

theorem prime_k_values (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by
  sorry

end prime_k_values_l100_100848


namespace bucket_P_turns_to_fill_the_drum_l100_100110

-- Define the capacities of the buckets
def capacity_P := 3
def capacity_Q := 1

-- Define the total number of turns for both buckets together to fill the drum
def turns_together := 60

-- Define the total capacity of the drum that gets filled in the given scenario of the problem
def total_capacity := turns_together * (capacity_P + capacity_Q)

-- The question: How many turns does it take for bucket P alone to fill this total capacity?
def turns_P_alone : ℕ :=
  total_capacity / capacity_P

theorem bucket_P_turns_to_fill_the_drum :
  turns_P_alone = 80 :=
by
  sorry

end bucket_P_turns_to_fill_the_drum_l100_100110


namespace factory_selection_and_probability_l100_100637

/-- Total number of factories in districts A, B, and C --/
def factories_A := 18
def factories_B := 27
def factories_C := 18

/-- Total number of factories and sample size --/
def total_factories := factories_A + factories_B + factories_C
def sample_size := 7

/-- Number of factories selected from districts A, B, and C --/
def selected_from_A := factories_A * sample_size / total_factories
def selected_from_B := factories_B * sample_size / total_factories
def selected_from_C := factories_C * sample_size / total_factories

/-- Number of ways to choose 2 factories out of the 7 --/
noncomputable def comb_7_2 := Nat.choose 7 2

/-- Number of favorable outcomes where at least one factory comes from district A --/
noncomputable def favorable_outcomes := 11

/-- Probability that at least one of the 2 factories comes from district A --/
noncomputable def probability := favorable_outcomes / comb_7_2

theorem factory_selection_and_probability :
  selected_from_A = 2 ∧ selected_from_B = 3 ∧ selected_from_C = 2 ∧ probability = 11 / 21 := by
  sorry

end factory_selection_and_probability_l100_100637


namespace correct_equation_l100_100620

theorem correct_equation (x : ℕ) : 8 * x - 3 = 7 * x + 4 :=
by sorry

end correct_equation_l100_100620


namespace G5_units_digit_is_0_l100_100071

def power_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  (base ^ exp) % modulus

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 2

theorem G5_units_digit_is_0 : (G 5) % 10 = 0 :=
by
  sorry

end G5_units_digit_is_0_l100_100071


namespace solve_x_squared_eq_nine_l100_100100

theorem solve_x_squared_eq_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) :=
by
  -- Proof by sorry placeholder
  sorry

end solve_x_squared_eq_nine_l100_100100


namespace fg_of_3_eq_29_l100_100852

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 :=
by
  sorry

end fg_of_3_eq_29_l100_100852


namespace percent_value_in_quarters_l100_100999

def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def num_nickels : ℕ := 80
def num_quarters : ℕ := 40

def value_in_nickels : ℕ := num_nickels * nickel_value
def value_in_quarters : ℕ := num_quarters * quarter_value
def total_value : ℕ := value_in_nickels + value_in_quarters

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value = 5 / 7 :=
by
  sorry

end percent_value_in_quarters_l100_100999


namespace poodle_barked_24_times_l100_100877

-- Defining the conditions and question in Lean
def poodle_barks (terrier_barks_per_hush times_hushed: ℕ) : ℕ :=
  2 * terrier_barks_per_hush * times_hushed

theorem poodle_barked_24_times (terrier_barks_per_hush times_hushed: ℕ) :
  terrier_barks_per_hush = 2 → times_hushed = 6 → poodle_barks terrier_barks_per_hush times_hushed = 24 :=
by
  intros
  sorry

end poodle_barked_24_times_l100_100877


namespace distance_between_parallel_lines_l100_100861

theorem distance_between_parallel_lines
  (O A B C D P Q : ℝ) -- Points on the circle with P and Q as defined midpoints
  (r d : ℝ) -- Radius of the circle and distance between the parallel lines
  (h_AB : dist A B = 36) -- Length of chord AB
  (h_CD : dist C D = 36) -- Length of chord CD
  (h_BC : dist B C = 40) -- Length of chord BC
  (h_OA : dist O A = r) 
  (h_OB : dist O B = r)
  (h_OC : dist O C = r)
  (h_PQ_parallel : dist P Q = d) -- Midpoints
  : d = 4 * Real.sqrt 19 / 3 :=
sorry

end distance_between_parallel_lines_l100_100861


namespace sum_x_y_z_l100_100212

noncomputable def a : ℝ := -Real.sqrt (9/27)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 7)^2 / 9)

theorem sum_x_y_z (ha : a = -Real.sqrt (9 / 27)) (hb : b = Real.sqrt ((3 + Real.sqrt 7) ^ 2 / 9)) (h_neg_a : a < 0) (h_pos_b : b > 0) :
  ∃ x y z : ℕ, (a + b)^3 = (x * Real.sqrt y) / z ∧ x + y + z = 718 := 
sorry

end sum_x_y_z_l100_100212


namespace average_speed_for_trip_l100_100625

theorem average_speed_for_trip :
  ∀ (walk_dist bike_dist drive_dist tot_dist walk_speed bike_speed drive_speed : ℝ)
  (h1 : walk_dist = 5) (h2 : bike_dist = 35) (h3 : drive_dist = 80)
  (h4 : tot_dist = 120) (h5 : walk_speed = 5) (h6 : bike_speed = 15)
  (h7 : drive_speed = 120),
  (tot_dist / (walk_dist / walk_speed + bike_dist / bike_speed + drive_dist / drive_speed)) = 30 :=
by
  intros
  sorry

end average_speed_for_trip_l100_100625


namespace average_is_equal_l100_100946

theorem average_is_equal (x : ℝ) :
  (1 / 3) * (2 * x + 4 + 5 * x + 3 + 3 * x + 8) = 3 * x - 5 → 
  x = -30 :=
by
  sorry

end average_is_equal_l100_100946


namespace arithmetic_geometric_sequence_l100_100867

-- Let {a_n} be an arithmetic sequence
-- And let a_1, a_2, a_3 form a geometric sequence
-- Given that a_5 = 1, we aim to prove that a_10 = 1
theorem arithmetic_geometric_sequence (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_geom : a 1 * a 3 = (a 2) ^ 2)
  (h_a5 : a 5 = 1) :
  a 10 = 1 :=
sorry

end arithmetic_geometric_sequence_l100_100867


namespace factor_expression_l100_100464

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l100_100464


namespace function_expression_and_min_value_l100_100643

def f (x b : ℝ) := x^2 - 2*x + b

theorem function_expression_and_min_value 
    (a b : ℝ)
    (condition1 : f (2 ^ a) b = b)
    (condition2 : f a b = 4) :
    f a b = 5 
    ∧ 
    ∃ c : ℝ, f (2^c) 5 = 4 ∧ c = 0 :=
by
  sorry

end function_expression_and_min_value_l100_100643


namespace find_f_7_over_2_l100_100462

section
variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x : ℝ, f (-x) = -f (x)
axiom even_shift_fn : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom range_x : Π x : ℝ, -1 ≤ x ∧ x ≤ 0 → f (x) = 2 * x^2

-- Prove that f(7/2) = 1/2
theorem find_f_7_over_2 : f (7 / 2) = 1 / 2 :=
sorry
end

end find_f_7_over_2_l100_100462


namespace calories_per_cookie_l100_100839

theorem calories_per_cookie :
  ∀ (cookies_per_bag bags_per_box total_calories total_number_cookies : ℕ),
  cookies_per_bag = 20 →
  bags_per_box = 4 →
  total_calories = 1600 →
  total_number_cookies = cookies_per_bag * bags_per_box →
  (total_calories / total_number_cookies) = 20 :=
by sorry

end calories_per_cookie_l100_100839


namespace find_a_l100_100816

theorem find_a (a : ℝ) (h_pos : a > 0)
  (h_eq : ∀ (f g : ℝ → ℝ), (f = λ x => x^2 + 10) → (g = λ x => x^2 - 6) → f (g a) = 14) :
  a = 2 * Real.sqrt 2 ∨ a = 2 :=
by 
  sorry

end find_a_l100_100816


namespace last_year_ticket_cost_l100_100447

theorem last_year_ticket_cost (this_year_cost : ℝ) (increase_percentage : ℝ) (last_year_cost : ℝ) :
  this_year_cost = last_year_cost * (1 + increase_percentage) ↔ last_year_cost = 85 :=
by
  let this_year_cost := 102
  let increase_percentage := 0.20
  sorry

end last_year_ticket_cost_l100_100447


namespace equation_b_not_symmetric_about_x_axis_l100_100423

def equationA (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equationB (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equationC (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equationD (x y : ℝ) : Prop := x + y^2 = -1

def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, f x y ↔ f x (-y)

theorem equation_b_not_symmetric_about_x_axis : 
  ¬ symmetric_about_x_axis (equationB) :=
sorry

end equation_b_not_symmetric_about_x_axis_l100_100423


namespace tangent_line_equation_l100_100733

-- Definitions used as conditions in the problem
def curve (x : ℝ) : ℝ := 2 * x - x^3
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Lean 4 statement representing the proof problem
theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := 1
  let m := deriv curve x₀
  m = -1 ∧ curve x₀ = y₀ →
  ∀ x y : ℝ, x + y - 2 = 0 → curve x₀ + m * (x - x₀) = y :=
by
  -- Proof would go here
  sorry

end tangent_line_equation_l100_100733


namespace total_spending_l100_100748

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l100_100748


namespace recreation_proof_l100_100682

noncomputable def recreation_percentage_last_week (W : ℝ) (P : ℝ) :=
  let last_week_spent := (P/100) * W
  let this_week_wages := (70/100) * W
  let this_week_spent := (20/100) * this_week_wages
  this_week_spent = (70/100) * last_week_spent

theorem recreation_proof :
  ∀ (W : ℝ), recreation_percentage_last_week W 20 :=
by
  intros
  sorry

end recreation_proof_l100_100682


namespace score_standard_deviation_l100_100570

theorem score_standard_deviation (mean std_dev : ℝ)
  (h1 : mean = 76)
  (h2 : mean - 2 * std_dev = 60) :
  100 = mean + 3 * std_dev :=
by
  -- Insert proof here
  sorry

end score_standard_deviation_l100_100570


namespace a_is_perfect_square_l100_100595

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l100_100595


namespace geometric_sequence_general_formula_l100_100616

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q : ℝ, ∀ n : ℕ, a n = a1 * q ^ (n - 1)

variables (a : ℕ → ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- The final statement to prove
theorem geometric_sequence_general_formula (h : geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end geometric_sequence_general_formula_l100_100616


namespace bacteria_growth_relation_l100_100642

variable (w1: ℝ := 10.0) (w2: ℝ := 16.0) (w3: ℝ := 25.6)

theorem bacteria_growth_relation :
  (w2 / w1) = (w3 / w2) :=
by
  sorry

end bacteria_growth_relation_l100_100642


namespace smallest_sum_of_squares_l100_100648

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 91) : x^2 + y^2 ≥ 109 :=
sorry

end smallest_sum_of_squares_l100_100648
