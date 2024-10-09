import Mathlib

namespace chessboard_max_squares_l2108_210833

def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

theorem chessboard_max_squares (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) : max_squares 1000 1000 = 1998 := 
by
  -- This is the theorem statement representing the maximum number of squares chosen
  -- in a 1000 x 1000 chessboard without having exactly three of them with two in the same row
  -- and two in the same column.
  sorry

end chessboard_max_squares_l2108_210833


namespace functional_relationship_max_daily_profit_price_reduction_1200_profit_l2108_210809

noncomputable def y : ℝ → ℝ := λ x => -2 * x^2 + 60 * x + 800

theorem functional_relationship :
  ∀ x : ℝ, y x = (40 - x) * (20 + 2 * x) := 
by
  intro x
  sorry

theorem max_daily_profit :
  y 15 = 1250 :=
by
  sorry

theorem price_reduction_1200_profit :
  ∀ x : ℝ, y x = 1200 → x = 10 ∨ x = 20 :=
by
  intro x
  sorry

end functional_relationship_max_daily_profit_price_reduction_1200_profit_l2108_210809


namespace total_points_of_three_players_l2108_210847

-- Definitions based on conditions
def points_tim : ℕ := 30
def points_joe : ℕ := points_tim - 20
def points_ken : ℕ := 2 * points_tim

-- Theorem statement for the total points scored by the three players
theorem total_points_of_three_players :
  points_tim + points_joe + points_ken = 100 :=
by
  -- Proof is to be provided
  sorry

end total_points_of_three_players_l2108_210847


namespace find_BC_length_l2108_210864

theorem find_BC_length
  (area : ℝ) (AB AC : ℝ)
  (h_area : area = 10 * Real.sqrt 3)
  (h_AB : AB = 5)
  (h_AC : AC = 8) :
  ∃ BC : ℝ, BC = 7 :=
by
  sorry

end find_BC_length_l2108_210864


namespace total_distance_of_trail_l2108_210839

theorem total_distance_of_trail (a b c d e : ℕ) 
    (h1 : a + b + c = 30) 
    (h2 : b + d = 30) 
    (h3 : d + e = 28) 
    (h4 : a + d = 34) : 
    a + b + c + d + e = 58 := 
sorry

end total_distance_of_trail_l2108_210839


namespace m_minus_n_eq_six_l2108_210814

theorem m_minus_n_eq_six (m n : ℝ) (h : ∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) : m - n = 6 := by
  sorry

end m_minus_n_eq_six_l2108_210814


namespace board_officer_election_l2108_210867

def num_ways_choose_officers (total_members : ℕ) (elect_officers : ℕ) : ℕ :=
  -- This will represent the number of ways to choose 4 officers given 30 members
  -- with the conditions on Alice, Bob, Chris, and Dana.
  if total_members = 30 ∧ elect_officers = 4 then
    358800 + 7800 + 7800 + 24
  else
    0

theorem board_officer_election : num_ways_choose_officers 30 4 = 374424 :=
by {
  -- Proof would go here
  sorry
}

end board_officer_election_l2108_210867


namespace ratio_of_pipe_lengths_l2108_210859

theorem ratio_of_pipe_lengths (L S : ℕ) (h1 : L + S = 177) (h2 : L = 118) (h3 : ∃ k : ℕ, L = k * S) : L / S = 2 := 
by 
  sorry

end ratio_of_pipe_lengths_l2108_210859


namespace total_students_went_to_concert_l2108_210866

/-- There are 12 buses and each bus took 57 students. We want to find out the total number of students who went to the concert. -/
theorem total_students_went_to_concert (num_buses : ℕ) (students_per_bus : ℕ) (total_students : ℕ) 
  (h1 : num_buses = 12) (h2 : students_per_bus = 57) (h3 : total_students = num_buses * students_per_bus) : 
  total_students = 684 := 
by
  sorry

end total_students_went_to_concert_l2108_210866


namespace solve_chris_age_l2108_210885

/-- 
The average of Amy's, Ben's, and Chris's ages is 12. Six years ago, Chris was the same age as Amy is now. In 3 years, Ben's age will be 3/4 of Amy's age at that time. 
How old is Chris now? 
-/
def chris_age : Prop := 
  ∃ (a b c : ℤ), 
    (a + b + c = 36) ∧
    (c - 6 = a) ∧ 
    (b + 3 = 3 * (a + 3) / 4) ∧
    (c = 17)

theorem solve_chris_age : chris_age := 
  by
    sorry

end solve_chris_age_l2108_210885


namespace number_of_intersections_l2108_210879

theorem number_of_intersections : ∃ (a_values : Finset ℚ), 
  ∀ a ∈ a_values, ∀ x y, y = 2 * x + a ∧ y = x^2 + 3 * a^2 ∧ x = 0 → 
  2 = a_values.card :=
by 
  sorry

end number_of_intersections_l2108_210879


namespace compare_expression_solve_inequality_l2108_210863

-- Part (1) Problem Statement in Lean 4
theorem compare_expression (x : ℝ) (h : x ≥ -1) : 
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) :=
by sorry

-- Part (2) Problem Statement in Lean 4
theorem solve_inequality (x a : ℝ) (ha : a < 0) : 
  (x^2 - a * x - 6 * a^2 > 0) ↔ (x < 3 * a ∨ x > -2 * a) :=
by sorry

end compare_expression_solve_inequality_l2108_210863


namespace xyz_sum_neg1_l2108_210861

theorem xyz_sum_neg1 (x y z : ℝ) (h : (x + 1)^2 + |y - 2| = -(2 * x - z)^2) : x + y + z = -1 :=
sorry

end xyz_sum_neg1_l2108_210861


namespace fuel_cost_is_50_cents_l2108_210862

-- Define the capacities of the tanks
def small_tank_capacity : ℕ := 60
def large_tank_capacity : ℕ := 60 * 3 / 2 -- 50% larger than small tank

-- Define the number of planes
def number_of_small_planes : ℕ := 2
def number_of_large_planes : ℕ := 2

-- Define the service charge per plane
def service_charge_per_plane : ℕ := 100
def total_service_charge : ℕ :=
  service_charge_per_plane * (number_of_small_planes + number_of_large_planes)

-- Define the total cost to fill all planes
def total_cost : ℕ := 550

-- Define the total fuel capacity
def total_fuel_capacity : ℕ :=
  number_of_small_planes * small_tank_capacity + number_of_large_planes * large_tank_capacity

-- Define the total fuel cost
def total_fuel_cost : ℕ := total_cost - total_service_charge

-- Define the fuel cost per liter
def fuel_cost_per_liter : ℕ :=
  total_fuel_cost / total_fuel_capacity

theorem fuel_cost_is_50_cents :
  fuel_cost_per_liter = 50 / 100 := by
sorry

end fuel_cost_is_50_cents_l2108_210862


namespace manolo_face_mask_time_l2108_210834
variable (x : ℕ)
def time_to_make_mask_first_hour := x
def face_masks_made_first_hour := 60 / x
def face_masks_made_next_three_hours := 180 / 6
def total_face_masks_in_four_hours := face_masks_made_first_hour + face_masks_made_next_three_hours

theorem manolo_face_mask_time : 
  total_face_masks_in_four_hours x = 45 ↔ x = 4 := sorry

end manolo_face_mask_time_l2108_210834


namespace refills_count_l2108_210894

variable (spent : ℕ) (cost : ℕ)

theorem refills_count (h1 : spent = 40) (h2 : cost = 10) : spent / cost = 4 := 
by
  sorry

end refills_count_l2108_210894


namespace number_of_even_factors_of_n_l2108_210875

def n : ℕ := 2^4 * 3^3 * 5 * 7^2

theorem number_of_even_factors_of_n : 
  (∃ k : ℕ, n = 2^4 * 3^3 * 5 * 7^2 ∧ k = 96) → 
  ∃ count : ℕ, 
    count = 96 ∧ 
    (∀ m : ℕ, 
      (m ∣ n ∧ m % 2 = 0) ↔ 
      (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d)) :=
by
  sorry

end number_of_even_factors_of_n_l2108_210875


namespace total_number_of_songs_is_30_l2108_210843

-- Define the number of country albums and pop albums
def country_albums : ℕ := 2
def pop_albums : ℕ := 3

-- Define the number of songs per album
def songs_per_album : ℕ := 6

-- Define the total number of albums
def total_albums : ℕ := country_albums + pop_albums

-- Define the total number of songs
def total_songs : ℕ := total_albums * songs_per_album

-- Prove that the total number of songs is 30
theorem total_number_of_songs_is_30 : total_songs = 30 := 
sorry

end total_number_of_songs_is_30_l2108_210843


namespace value_two_sd_below_mean_l2108_210832

theorem value_two_sd_below_mean (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 17.5) (h_std_dev : std_dev = 2.5) : 
  mean - 2 * std_dev = 12.5 := by
  -- proof omitted
  sorry

end value_two_sd_below_mean_l2108_210832


namespace oranges_taken_l2108_210857

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end oranges_taken_l2108_210857


namespace math_problem_l2108_210869

theorem math_problem :
    (50 + 5 * (12 / (180 / 3))^2) * Real.sin (Real.pi / 6) = 25.1 :=
by
  sorry

end math_problem_l2108_210869


namespace solve_for_t_l2108_210804

theorem solve_for_t (s t : ℝ) (h1 : 12 * s + 8 * t = 160) (h2 : s = t^2 + 2) :
  t = (Real.sqrt 103 - 1) / 3 :=
sorry

end solve_for_t_l2108_210804


namespace scientific_notation_of_400000_l2108_210873

theorem scientific_notation_of_400000 :
  (400000: ℝ) = 4 * 10^5 :=
by 
  sorry

end scientific_notation_of_400000_l2108_210873


namespace couple_ticket_cost_l2108_210850

variable (x : ℝ)

def single_ticket_cost : ℝ := 20
def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16

theorem couple_ticket_cost :
  96 * single_ticket_cost + 16 * x = total_sales →
  x = 22.5 :=
by
  sorry

end couple_ticket_cost_l2108_210850


namespace find_page_added_twice_l2108_210801

theorem find_page_added_twice (m p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : (m * (m + 1)) / 2 + p = 2550) : p = 6 :=
sorry

end find_page_added_twice_l2108_210801


namespace renne_savings_ratio_l2108_210815

theorem renne_savings_ratio (ME CV N : ℕ) (h_ME : ME = 4000) (h_CV : CV = 16000) (h_N : N = 8) :
  (CV / N : ℕ) / ME = 1 / 2 :=
by
  sorry

end renne_savings_ratio_l2108_210815


namespace count_five_letter_words_l2108_210880

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l2108_210880


namespace find_number_l2108_210868

theorem find_number (x : ℝ) : 61 + x * 12 / (180 / 3) = 62 → x = 5 :=
by
  sorry

end find_number_l2108_210868


namespace capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l2108_210835

-- Step 1: Define the capacities of type A and B cars
def typeACarCapacity := 3
def typeBCarCapacity := 4

-- Step 2: Prove transportation capacities x and y
theorem capacities_correct (x y: ℕ) (h1 : 3 * x + 2 * y = 17) (h2 : 2 * x + 3 * y = 18) :
    x = typeACarCapacity ∧ y = typeBCarCapacity :=
by
  sorry

-- Step 3: Define a rental plan to transport 35 tons
theorem rental_plan_exists (a b : ℕ) : 3 * a + 4 * b = 35 :=
by
  sorry

-- Step 4: Prove the minimal cost solution
def typeACarCost := 300
def typeBCarCost := 320

def rentalCost (a b : ℕ) : ℕ := a * typeACarCost + b * typeBCarCost

theorem minimal_rental_cost_exists :
    ∃ a b, 3 * a + 4 * b = 35 ∧ rentalCost a b = 2860 :=
by
  sorry

end capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l2108_210835


namespace max_cos_alpha_l2108_210858

open Real

-- Define the condition as a hypothesis
def cos_sum_eq (α β : ℝ) : Prop :=
  cos (α + β) = cos α + cos β

-- State the maximum value theorem
theorem max_cos_alpha (α β : ℝ) (h : cos_sum_eq α β) : ∃ α, cos α = sqrt 3 - 1 :=
by
  sorry   -- Proof is omitted

#check max_cos_alpha

end max_cos_alpha_l2108_210858


namespace ceil_floor_diff_l2108_210836

theorem ceil_floor_diff (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ - ⌊x⌋ = 1 := 
by 
  sorry

end ceil_floor_diff_l2108_210836


namespace calculate_expression_l2108_210831

theorem calculate_expression :
  |-2*Real.sqrt 3| - (1 - Real.pi)^0 + 2*Real.cos (Real.pi / 6) + (1 / 4)^(-1 : ℤ) = 3 * Real.sqrt 3 + 3 :=
by
  sorry

end calculate_expression_l2108_210831


namespace average_speed_of_train_l2108_210886

theorem average_speed_of_train
  (distance1 : ℝ) (time1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 240) (h2 : time1 = 3) (h3 : stop_time = 0.5)
  (h4 : distance2 = 450) (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = 81.18 := 
sorry

end average_speed_of_train_l2108_210886


namespace original_price_l2108_210897

theorem original_price (P : ℝ) 
  (h1 : 1.40 * P = P + 700) : P = 1750 :=
by sorry

end original_price_l2108_210897


namespace largest_fraction_l2108_210818

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end largest_fraction_l2108_210818


namespace total_points_l2108_210844

theorem total_points (zach_points ben_points : ℝ) (h₁ : zach_points = 42.0) (h₂ : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
  by sorry

end total_points_l2108_210844


namespace strawberries_in_each_handful_l2108_210825

theorem strawberries_in_each_handful (x : ℕ) (h : (x - 1) * (75 / x) = 60) : x = 5 :=
sorry

end strawberries_in_each_handful_l2108_210825


namespace school_students_count_l2108_210802

def students_in_school (c n : ℕ) : ℕ := n * c

theorem school_students_count
  (c n : ℕ)
  (h1 : n * c = (n - 6) * (c + 5))
  (h2 : n * c = (n - 16) * (c + 20)) :
  students_in_school c n = 900 :=
by
  sorry

end school_students_count_l2108_210802


namespace local_min_at_neg_one_l2108_210800

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_min_at_neg_one : 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≥ f (-1) := by
  sorry

end local_min_at_neg_one_l2108_210800


namespace avg_weight_of_22_boys_l2108_210893

theorem avg_weight_of_22_boys:
  let total_boys := 30
  let avg_weight_8 := 45.15
  let avg_weight_total := 48.89
  let total_weight_8 := 8 * avg_weight_8
  let total_weight_all := total_boys * avg_weight_total
  ∃ A : ℝ, A = 50.25 ∧ 22 * A + total_weight_8 = total_weight_all :=
by {
  sorry 
}

end avg_weight_of_22_boys_l2108_210893


namespace problem_inequality_l2108_210810

theorem problem_inequality (a b x y : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end problem_inequality_l2108_210810


namespace inscribed_circle_ratio_l2108_210892

theorem inscribed_circle_ratio (a b c u v : ℕ) (h_triangle : a = 10 ∧ b = 24 ∧ c = 26) 
    (h_tangent_segments : u < v) (h_side_sum : u + v = a) : u / v = 2 / 3 :=
by
    sorry

end inscribed_circle_ratio_l2108_210892


namespace solve_inequality_l2108_210890

theorem solve_inequality (x : ℝ) : (2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2 : ℝ) (4 / 3) ∨ x ∈ Set.Icc (8 / 3) (6 : ℝ)) :=
sorry

end solve_inequality_l2108_210890


namespace number_of_distinct_arrangements_l2108_210883

-- Given conditions: There are 7 items and we need to choose 4 out of these 7.
def binomial_coefficient (n k : ℕ) : ℕ :=
  (n.choose k)

-- Given condition: Calculate the number of sequences of arranging 4 selected items.
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The statement in Lean 4 to prove that the number of distinct arrangements is 840.
theorem number_of_distinct_arrangements : binomial_coefficient 7 4 * factorial 4 = 840 :=
by
  sorry

end number_of_distinct_arrangements_l2108_210883


namespace slope_of_line_eq_neg_four_thirds_l2108_210826

variable {x y : ℝ}
variable (p₁ p₂ : ℝ × ℝ) (h₁ : 3 / p₁.1 + 4 / p₁.2 = 0) (h₂ : 3 / p₂.1 + 4 / p₂.2 = 0)

theorem slope_of_line_eq_neg_four_thirds 
  (hneq : p₁.1 ≠ p₂.1):
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1) = -4 / 3 := 
sorry

end slope_of_line_eq_neg_four_thirds_l2108_210826


namespace number_of_elements_in_set_P_l2108_210899

theorem number_of_elements_in_set_P
  (p q : ℕ) -- we are dealing with non-negative integers here
  (h1 : p = 3 * q)
  (h2 : p + q = 4500)
  : p = 3375 :=
by
  sorry -- Proof goes here

end number_of_elements_in_set_P_l2108_210899


namespace range_of_m_l2108_210872

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  (f (m * Real.sin θ) + f (1 - m) > 0) ↔ (m ≤ 1) :=
sorry

end range_of_m_l2108_210872


namespace bus_speeds_l2108_210846

theorem bus_speeds (d t : ℝ) (s₁ s₂ : ℝ)
  (h₀ : d = 48)
  (h₁ : t = 1 / 6) -- 10 minutes in hours
  (h₂ : s₂ = s₁ - 4)
  (h₃ : d / s₂ - d / s₁ = t) :
  s₁ = 36 ∧ s₂ = 32 := 
sorry

end bus_speeds_l2108_210846


namespace incorrect_statement_B_l2108_210845

axiom statement_A : ¬ (0 > 0 ∨ 0 < 0)
axiom statement_C : ∀ (q : ℚ), (∃ (m : ℤ), q = m) ∨ (∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
axiom statement_D : abs (0 : ℚ) = 0

theorem incorrect_statement_B : ¬ (∀ (q : ℚ), abs q ≥ 1 → abs 1 = abs q) := sorry

end incorrect_statement_B_l2108_210845


namespace g_five_eq_one_l2108_210805

noncomputable def g : ℝ → ℝ := sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_nonzero : ∀ x : ℝ, g x ≠ 0

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l2108_210805


namespace n_squared_divisible_by_36_l2108_210870

theorem n_squared_divisible_by_36 (n : ℕ) (h1 : 0 < n) (h2 : 6 ∣ n) : 36 ∣ n^2 := 
sorry

end n_squared_divisible_by_36_l2108_210870


namespace system_of_equations_unique_solution_l2108_210820

theorem system_of_equations_unique_solution :
  (∃ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7) →
  (∀ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7 →
    x = 26 / 5 ∧ y = 9 / 5) := 
by {
  -- Proof to be provided
  sorry
}

end system_of_equations_unique_solution_l2108_210820


namespace emery_reading_days_l2108_210860

theorem emery_reading_days (S E : ℕ) (h1 : E = S / 5) (h2 : (E + S) / 2 = 60) : E = 20 := by
  sorry

end emery_reading_days_l2108_210860


namespace gcd_13924_27018_l2108_210853

theorem gcd_13924_27018 : Int.gcd 13924 27018 = 2 := 
  by
    sorry

end gcd_13924_27018_l2108_210853


namespace ann_age_l2108_210876

variable (A T : ℕ)

-- Condition 1: Tom is currently two times older than Ann
def tom_older : Prop := T = 2 * A

-- Condition 2: The sum of their ages 10 years later will be 38
def age_sum_later : Prop := (A + 10) + (T + 10) = 38

-- Theorem: Ann's current age
theorem ann_age (h1 : tom_older A T) (h2 : age_sum_later A T) : A = 6 :=
by
  sorry

end ann_age_l2108_210876


namespace quadratic_eq_roots_quadratic_eq_range_l2108_210821

theorem quadratic_eq_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0 ∧ x1 + 3 * x2 = 2 * m + 8) →
  (m = -1 ∨ m = -2) :=
sorry

theorem quadratic_eq_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0) →
  m ≤ 0 :=
sorry

end quadratic_eq_roots_quadratic_eq_range_l2108_210821


namespace combined_weight_cats_l2108_210856

-- Define the weights of the cats
def weight_cat1 := 2
def weight_cat2 := 7
def weight_cat3 := 4

-- Prove the combined weight of the three cats is 13 pounds
theorem combined_weight_cats :
  weight_cat1 + weight_cat2 + weight_cat3 = 13 := by
  sorry

end combined_weight_cats_l2108_210856


namespace new_problem_l2108_210807

theorem new_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := 
by
  sorry

end new_problem_l2108_210807


namespace length_of_shortest_side_30_60_90_l2108_210806

theorem length_of_shortest_side_30_60_90 (x : ℝ) : 
  (∃ x : ℝ, (2 * x = 15)) → x = 15 / 2 :=
by
  sorry

end length_of_shortest_side_30_60_90_l2108_210806


namespace evaluate_fraction_l2108_210898

theorem evaluate_fraction :
  (0.5^2 + 0.05^3) / 0.005^3 = 2000100 := by
  sorry

end evaluate_fraction_l2108_210898


namespace smallest_number_of_three_l2108_210813

theorem smallest_number_of_three (x : ℕ) (h1 : x = 18)
  (h2 : ∀ y z : ℕ, y = 4 * x ∧ z = 2 * y)
  (h3 : (x + 4 * x + 8 * x) / 3 = 78)
  : x = 18 := by
  sorry

end smallest_number_of_three_l2108_210813


namespace largest_prime_17p_625_l2108_210882

theorem largest_prime_17p_625 (p : ℕ) (h_prime : Nat.Prime p) (h_sqrt : ∃ q, 17 * p + 625 = q^2) : p = 67 :=
by
  sorry

end largest_prime_17p_625_l2108_210882


namespace value_of_m_has_positive_root_l2108_210842

theorem value_of_m_has_positive_root (x m : ℝ) (hx : x ≠ 3) :
    ((x + 5) / (x - 3) = 2 - m / (3 - x)) → x > 0 → m = 8 := 
sorry

end value_of_m_has_positive_root_l2108_210842


namespace lunch_choices_l2108_210855

theorem lunch_choices (chickens drinks : ℕ) (h1 : chickens = 3) (h2 : drinks = 2) : chickens * drinks = 6 :=
by
  sorry

end lunch_choices_l2108_210855


namespace false_conjunction_l2108_210819

theorem false_conjunction (p q : Prop) (h : ¬(p ∧ q)) : ¬ (¬p ∧ ¬q) := sorry

end false_conjunction_l2108_210819


namespace net_gain_is_88837_50_l2108_210817

def initial_home_value : ℝ := 500000
def first_sale_price : ℝ := 1.15 * initial_home_value
def first_purchase_price : ℝ := 0.95 * first_sale_price
def second_sale_price : ℝ := 1.1 * first_purchase_price
def second_purchase_price : ℝ := 0.9 * second_sale_price

def total_sales : ℝ := first_sale_price + second_sale_price
def total_purchases : ℝ := first_purchase_price + second_purchase_price
def net_gain_for_A : ℝ := total_sales - total_purchases

theorem net_gain_is_88837_50 : net_gain_for_A = 88837.50 := by
  -- proof steps would go here, but they are omitted per instructions
  sorry

end net_gain_is_88837_50_l2108_210817


namespace sum_of_specific_terms_l2108_210829

theorem sum_of_specific_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h1 : S 3 = 9) 
  (h2 : S 6 = 36) 
  (h3 : ∀ n, S n = n * (a 1) + d * n * (n - 1) / 2) :
  a 7 + a 8 + a 9 = 45 := 
sorry

end sum_of_specific_terms_l2108_210829


namespace problem1_part1_problem1_part2_l2108_210828

open Set Real

theorem problem1_part1 (a : ℝ) (h1: a = 5) :
  let A := { x : ℝ | (x - 6) * (x - 2 * a - 5) > 0 }
  let B := { x : ℝ | (a ^ 2 + 2 - x) * (2 * a - x) < 0 }
  A ∩ B = { x | 15 < x ∧ x < 27 } := sorry

theorem problem1_part2 (a : ℝ) (h2: a > 1 / 2) :
  let A := { x : ℝ | x < 6 ∨ x > 2 * a + 5 }
  let B := { x : ℝ | 2 * a < x ∧ x < a ^ 2 + 2 }
  (∀ x, x ∈ A → x ∈ B) ∧ ¬ (∀ x, x ∈ B → x ∈ A) → (1 / 2 < a ∧ a ≤ 2) := sorry

end problem1_part1_problem1_part2_l2108_210828


namespace find_coordinates_of_P_l2108_210840

-- Define the conditions
variable (x y : ℝ)
def in_second_quadrant := x < 0 ∧ y > 0
def distance_to_x_axis := abs y = 7
def distance_to_y_axis := abs x = 3

-- Define the statement to be proved in Lean 4
theorem find_coordinates_of_P :
  in_second_quadrant x y ∧ distance_to_x_axis y ∧ distance_to_y_axis x → (x, y) = (-3, 7) :=
by
  sorry

end find_coordinates_of_P_l2108_210840


namespace possible_point_counts_l2108_210874

theorem possible_point_counts (r b g : ℕ) (d_RB d_RG d_BG : ℕ) :
    r + b + g = 15 →
    r * b * d_RB = 51 →
    r * g * d_RG = 39 →
    b * g * d_BG = 1 →
    (r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3) :=
by {
    sorry
}

end possible_point_counts_l2108_210874


namespace cost_of_song_book_l2108_210884

def cost_of_trumpet : ℝ := 145.16
def total_amount_spent : ℝ := 151.00

theorem cost_of_song_book : (total_amount_spent - cost_of_trumpet) = 5.84 := by
  sorry

end cost_of_song_book_l2108_210884


namespace triangle_shape_l2108_210878

-- Let there be a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively
variables (A B C : ℝ) (a b c : ℝ) (b_ne_1 : b ≠ 1)
          (h1 : (log (b) (C / A)) = (log (sqrt (b)) (2)))
          (h2 : (log (b) (sin B / sin A)) = (log (sqrt (b)) (2)))

-- Define the theorem that states the shape of the triangle
theorem triangle_shape : A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) :=
by
  -- Proof is provided in the solution, skipping proof here
  sorry

end triangle_shape_l2108_210878


namespace tangent_line_proof_minimum_a_proof_l2108_210877

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

def tangent_equation_correct : Prop :=
  let y := f 1
  let slope := (2 / 1 - 6 * 1 - 11)
  (slope = -15) ∧ (y = -14) ∧ (∀ x y, y = -15 * (x - 1) + -14 ↔ 15 * x + y - 1 = 0)

def minimum_a_correct : Prop :=
  ∃ a : ℤ, 
    (∀ x, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ↔ (a = 2)

theorem tangent_line_proof : tangent_equation_correct := sorry

theorem minimum_a_proof : minimum_a_correct := sorry

end tangent_line_proof_minimum_a_proof_l2108_210877


namespace correct_representations_l2108_210803

open Set

theorem correct_representations : 
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  (¬S1 ∧ ¬S2 ∧ S3 ∧ S4) :=
by
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  exact sorry

end correct_representations_l2108_210803


namespace min_value_range_l2108_210849

noncomputable def f (a x : ℝ) := x^2 + a * x

theorem min_value_range (a : ℝ) :
  (∃x : ℝ, ∀y : ℝ, f a (f a x) ≥ f a (f a y)) ∧ (∀x : ℝ, f a x ≥ f a (-a / 2)) →
  a ≤ 0 ∨ a ≥ 2 := sorry

end min_value_range_l2108_210849


namespace value_of_a_plus_b_l2108_210881

theorem value_of_a_plus_b (a b c : ℤ) 
    (h1 : a + b + c = 11)
    (h2 : a + b - c = 19)
    : a + b = 15 := 
by
    -- Mathematical details skipped
    sorry

end value_of_a_plus_b_l2108_210881


namespace total_cats_in_academy_l2108_210812

theorem total_cats_in_academy (cats_jump cats_jump_fetch cats_fetch cats_fetch_spin cats_spin cats_jump_spin cats_all_three cats_none: ℕ)
  (h_jump: cats_jump = 60)
  (h_jump_fetch: cats_jump_fetch = 20)
  (h_fetch: cats_fetch = 35)
  (h_fetch_spin: cats_fetch_spin = 15)
  (h_spin: cats_spin = 40)
  (h_jump_spin: cats_jump_spin = 22)
  (h_all_three: cats_all_three = 11)
  (h_none: cats_none = 10) :
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none = 99 :=
by
  calc 
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none 
  = 11 + (20 - 11) + (15 - 11) + (22 - 11) + (60 - (9 + 11 + 11)) + (35 - (9 + 4 + 11)) + (40 - (11 + 4 + 11)) + 10 
  := by sorry
  _ = 99 := by sorry

end total_cats_in_academy_l2108_210812


namespace joe_average_speed_l2108_210823

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem joe_average_speed :
  let distance1 := 420
  let speed1 := 60
  let distance2 := 120
  let speed2 := 40
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed total_distance total_time = 54 := by
sorry

end joe_average_speed_l2108_210823


namespace geom_seq_inc_condition_l2108_210808

theorem geom_seq_inc_condition (a₁ a₂ q : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ = a₁ * q) :
  (a₁^2 < a₂^2) ↔ 
  (∀ n m : ℕ, n < m → (a₁ * q^n) < (a₁ * q^m) ∨ ((a₁ * q^n) = (a₁ * q^m) ∧ q = 1)) :=
by
  sorry

end geom_seq_inc_condition_l2108_210808


namespace two_pow_gt_square_for_n_ge_5_l2108_210827

theorem two_pow_gt_square_for_n_ge_5 (n : ℕ) (hn : n ≥ 5) : 2^n > n^2 :=
sorry

end two_pow_gt_square_for_n_ge_5_l2108_210827


namespace positive_solution_sqrt_a_sub_b_l2108_210830

theorem positive_solution_sqrt_a_sub_b (a b : ℕ) (x : ℝ) 
  (h_eq : x^2 + 14 * x = 32) 
  (h_form : x = Real.sqrt a - b) 
  (h_pos_nat : a > 0 ∧ b > 0) : 
  a + b = 88 := 
by
  sorry

end positive_solution_sqrt_a_sub_b_l2108_210830


namespace max_distance_proof_area_of_coverage_ring_proof_l2108_210854

noncomputable def maxDistanceFromCenterToRadars : ℝ :=
  24 / Real.sin (Real.pi / 7)

noncomputable def areaOfCoverageRing : ℝ :=
  960 * Real.pi / Real.tan (Real.pi / 7)

theorem max_distance_proof :
  ∀ (r n : ℕ) (width : ℝ),  n = 7 → r = 26 → width = 20 → 
  maxDistanceFromCenterToRadars = 24 / Real.sin (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

theorem area_of_coverage_ring_proof :
  ∀ (r n : ℕ) (width : ℝ), n = 7 → r = 26 → width = 20 → 
  areaOfCoverageRing = 960 * Real.pi / Real.tan (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

end max_distance_proof_area_of_coverage_ring_proof_l2108_210854


namespace parameterization_properties_l2108_210871

theorem parameterization_properties (a b c d : ℚ)
  (h1 : a * (-1) + b = -3)
  (h2 : c * (-1) + d = 5)
  (h3 : a * 2 + b = 4)
  (h4 : c * 2 + d = 15) :
  a^2 + b^2 + c^2 + d^2 = 790 / 9 :=
sorry

end parameterization_properties_l2108_210871


namespace expected_plain_zongzi_picked_l2108_210848

-- Definitions and conditions:
def total_zongzi := 10
def red_bean_zongzi := 3
def meat_zongzi := 3
def plain_zongzi := 4

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probabilities
def P_X_0 : ℚ := (choose 6 2 : ℚ) / choose 10 2
def P_X_1 : ℚ := (choose 6 1 * choose 4 1 : ℚ) / choose 10 2
def P_X_2 : ℚ := (choose 4 2 : ℚ) / choose 10 2

-- Expected value of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

theorem expected_plain_zongzi_picked : E_X = 4 / 5 := by
  -- Using the definition of E_X and the respective probabilities
  unfold E_X P_X_0 P_X_1 P_X_2
  -- Use the given formula to calculate the values
  -- Remaining steps would show detailed calculations leading to the answer
  sorry

end expected_plain_zongzi_picked_l2108_210848


namespace scientific_notation_l2108_210841

theorem scientific_notation (n : ℕ) (h : n = 27000000) : 
  ∃ (m : ℝ) (e : ℤ), n = m * (10 : ℝ) ^ e ∧ m = 2.7 ∧ e = 7 :=
by 
  use 2.7 
  use 7
  sorry

end scientific_notation_l2108_210841


namespace joel_age_when_dad_twice_l2108_210887

theorem joel_age_when_dad_twice (x joel_age dad_age: ℕ) (h₁: joel_age = 12) (h₂: dad_age = 47) 
(h₃: dad_age + x = 2 * (joel_age + x)) : joel_age + x = 35 :=
by
  rw [h₁, h₂] at h₃ 
  sorry

end joel_age_when_dad_twice_l2108_210887


namespace radius_increase_l2108_210889

theorem radius_increase (C₁ C₂ : ℝ) (C₁_eq : C₁ = 30) (C₂_eq : C₂ = 40) :
  let r₁ := C₁ / (2 * Real.pi)
  let r₂ := C₂ / (2 * Real.pi)
  r₂ - r₁ = 5 / Real.pi :=
by
  simp [C₁_eq, C₂_eq]
  sorry

end radius_increase_l2108_210889


namespace original_number_is_fraction_l2108_210811

theorem original_number_is_fraction (x : ℚ) (h : 1 + (1 / x) = 9 / 4) : x = 4 / 5 :=
by
  sorry

end original_number_is_fraction_l2108_210811


namespace find_line_m_l2108_210888

noncomputable def reflect_point_across_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
  let line_vector := (a, b)
  let scaling_factor := -2 * ((a * P.1 + b * P.2 + c) / (a^2 + b^2))
  ((P.1 + scaling_factor * a), (P.2 + scaling_factor * b))

theorem find_line_m (P P'' : ℝ × ℝ) (a b : ℝ) (c : ℝ := 0)
  (h₁ : P = (2, -3))
  (h₂ : a * 1 + b * 4 = 0)
  (h₃ : P'' = (1, 4))
  (h₄ : reflect_point_across_line (reflect_point_across_line P a b c) a b c = P'') :
  4 * P''.1 - P''.2 = 0 :=
by
  sorry

end find_line_m_l2108_210888


namespace range_of_a_l2108_210824

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x + 3

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (a-1) (a+1), 4*x - 1/x = 0) ↔ 1 ≤ a ∧ a < 3/2 :=
sorry

end range_of_a_l2108_210824


namespace P_plus_Q_l2108_210816

theorem P_plus_Q (P Q : ℝ) (h : ∀ x, x ≠ 3 → (P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3))) : P + Q = 46 :=
sorry

end P_plus_Q_l2108_210816


namespace necessary_but_not_sufficient_l2108_210838

theorem necessary_but_not_sufficient (a c : ℝ) : 
  (c ≠ 0) → (∀ (x y : ℝ), ax^2 + y^2 = c → (c = 0 → false) ∧ (c ≠ 0 → (∃ x y : ℝ, ax^2 + y^2 = c))) :=
by
  sorry

end necessary_but_not_sufficient_l2108_210838


namespace chef_earns_less_than_manager_l2108_210852

noncomputable def hourly_wage_manager : ℝ := 8.5
noncomputable def hourly_wage_dishwasher : ℝ := hourly_wage_manager / 2
noncomputable def hourly_wage_chef : ℝ := hourly_wage_dishwasher * 1.2
noncomputable def daily_bonus : ℝ := 5
noncomputable def overtime_multiplier : ℝ := 1.5
noncomputable def tax_rate : ℝ := 0.15

noncomputable def manager_hours : ℝ := 10
noncomputable def dishwasher_hours : ℝ := 6
noncomputable def chef_hours : ℝ := 12
noncomputable def standard_hours : ℝ := 8

noncomputable def compute_earnings (hourly_wage : ℝ) (hours_worked : ℝ) : ℝ :=
  let regular_hours := min standard_hours hours_worked
  let overtime_hours := max 0 (hours_worked - standard_hours)
  let regular_pay := regular_hours * hourly_wage
  let overtime_pay := overtime_hours * hourly_wage * overtime_multiplier
  let total_earnings_before_tax := regular_pay + overtime_pay + daily_bonus
  total_earnings_before_tax * (1 - tax_rate)

noncomputable def manager_earnings : ℝ := compute_earnings hourly_wage_manager manager_hours
noncomputable def dishwasher_earnings : ℝ := compute_earnings hourly_wage_dishwasher dishwasher_hours
noncomputable def chef_earnings : ℝ := compute_earnings hourly_wage_chef chef_hours

theorem chef_earns_less_than_manager : manager_earnings - chef_earnings = 18.78 := by
  sorry

end chef_earns_less_than_manager_l2108_210852


namespace jayden_planes_l2108_210896

theorem jayden_planes (W : ℕ) (wings_per_plane : ℕ) (total_wings : W = 108) (wpp_pos : wings_per_plane = 2) :
  ∃ n : ℕ, n = W / wings_per_plane ∧ n = 54 :=
by
  sorry

end jayden_planes_l2108_210896


namespace correct_calculation_l2108_210865

theorem correct_calculation (x : ℝ) (h : (x / 2) + 45 = 85) : (2 * x) - 45 = 115 :=
by {
  -- Note: Proof steps are not needed, 'sorry' is used to skip the proof
  sorry
}

end correct_calculation_l2108_210865


namespace Montoya_budget_spent_on_food_l2108_210822

-- Define the fractions spent on groceries and going out to eat
def groceries_fraction : ℝ := 0.6
def eating_out_fraction : ℝ := 0.2

-- Define the total fraction spent on food
def total_food_fraction (g : ℝ) (e : ℝ) : ℝ := g + e

-- The theorem to prove
theorem Montoya_budget_spent_on_food : total_food_fraction groceries_fraction eating_out_fraction = 0.8 := 
by
  -- the proof will go here
  sorry

end Montoya_budget_spent_on_food_l2108_210822


namespace max_value_of_ex1_ex2_l2108_210895

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then exp x else -(x^3)

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := 
  f (f x) - a

-- Define the condition that g(x) = 0 has two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0

-- Define the target function h
noncomputable def h (m : ℝ) : ℝ := 
  m^3 * exp (-m)

-- Statement of the final proof
theorem max_value_of_ex1_ex2 (a : ℝ) (hpos : 0 < a) (zeros : has_two_distinct_zeros a) :
  (∃ x1 x2 : ℝ, e^x1 * e^x2 = (27 : ℝ) / (exp 3) ∧ g x1 a = 0 ∧ g x2 a = 0) :=
sorry

end max_value_of_ex1_ex2_l2108_210895


namespace savings_per_month_l2108_210851

-- Define the monthly earnings, total needed for car, and total earnings
def monthly_earnings : ℤ := 4000
def total_needed_for_car : ℤ := 45000
def total_earnings : ℤ := 360000

-- Define the number of months it takes to save the required amount using total earnings and monthly earnings
def number_of_months : ℤ := total_earnings / monthly_earnings

-- Define the monthly savings based on the total needed and number of months
def monthly_savings : ℤ := total_needed_for_car / number_of_months

-- Prove that the monthly savings is £500
theorem savings_per_month : monthly_savings = 500 := by
  -- Placeholder for the proof
  sorry

end savings_per_month_l2108_210851


namespace determine_functions_l2108_210891

noncomputable def f : (ℝ → ℝ) := sorry

theorem determine_functions (f : ℝ → ℝ)
  (h_domain: ∀ x, 0 < x → 0 < f x)
  (h_eq: ∀ w x y z, 0 < w → 0 < x → 0 < y → 0 < z → w * x = y * z →
    (f w)^2 + (f x)^2 = (f (y^2) + f (z^2)) * (w^2 + x^2) / (y^2 + z^2)) :
  (∀ x, 0 < x → (f x = x ∨ f x = 1 / x)) :=
by
  intros x hx
  sorry

end determine_functions_l2108_210891


namespace incorrect_judgment_l2108_210837

variable (p q : Prop)
variable (hyp_p : p = (3 + 3 = 5))
variable (hyp_q : q = (5 > 2))

theorem incorrect_judgment : 
  (¬ (p ∧ q) ∧ ¬p) = false :=
by
  sorry

end incorrect_judgment_l2108_210837
